#include "graph_passes.hpp"

#include "graph_checks.hpp"
#include "scene.hpp"

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>
#include <aurora/render_graph.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace render_demo {
namespace {
using aurora::render::Access;
using aurora::render::PassBuilder;
using aurora::render::PassContext;
using aurora::render::PassId;

constexpr std::string_view VignetteWgsl = R"(
struct VignetteParams {
  strength: f32,
  inner: f32,
  outer: f32,
  pad: f32,
};
@group(0) @binding(1) var<uniform> params: VignetteParams;

var<private> positions: array<vec2f, 3> = array(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(positions[vi], 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * aurora_frame.inv_efb_size;
  let d = length(uv - vec2f(0.5)) * 1.41421356;
  let f = 1.0 - params.strength * smoothstep(params.inner, params.outer, d);
  return vec4f(f, f, f, 1.0);
}
)";

struct VignettePass {
  std::string anchor;
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  // Render worker only
  wgpu::RenderPipeline pipeline;
  wgpu::PipelineLayout layout;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  u32 samples = 0;
};

VignettePass sVignette;

wgpu::RenderPipeline create_vignette_pipeline(const PassContext& ctx, wgpu::ColorTargetState target, u32 samples) {
  const std::string code = std::string{aurora::render::FrameWgsl} + std::string{VignetteWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.vignette"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  // Multiplies the scene by the shader's factor, so the pass never has to sample gx.color
  const wgpu::BlendState blend{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  target.blend = &blend;
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "demo.vignette",
      .layout = ctx.pipeline_layout(),
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples},
      .fragment = &fragment,
  };
  return ctx.device().CreateRenderPipeline(&descriptor);
}

void declare_vignette(PassBuilder& builder, void* userdata) {
  const auto& pass = *static_cast<const VignettePass*>(userdata);
  builder.use(aurora::render::resource::GxColor, Access::ColorRMW);
  if (!pass.anchor.empty()) {
    builder.after(pass.anchor);
  }
}

void encode_vignette(PassContext& ctx, void* userdata) {
  auto& pass = *static_cast<VignettePass*>(userdata);
  constexpr aurora::render::TexRef Color{0};
  const wgpu::ColorTargetState target = ctx.color_target(Color);
  const u32 samples = ctx.sample_count(Color);
  if (!pass.pipeline || pass.format != target.format || pass.samples != samples ||
      pass.layout.Get() != ctx.pipeline_layout().Get()) {
    pass.pipeline = create_vignette_pipeline(ctx, target, samples);
    pass.layout = ctx.pipeline_layout();
    pass.format = target.format;
    pass.samples = samples;
  }
  const wgpu::RenderPassEncoder render = ctx.begin_render();
  if (!render) {
    return;
  }
  render.SetPipeline(pass.pipeline);
  render.Draw(3);
  render.End();
  // Counted only once the pass began and ended, so N/N proves every frame drew the vignette
  pass.encodes.fetch_add(1, std::memory_order_relaxed);
}

constexpr uint64_t ProbeBytes = 4 * sizeof(f32);
constexpr auto PipelineWait = std::chrono::seconds{10};

constexpr std::string_view DepthProbeWgsl = R"(
@group(1) @binding(0) var gx_depth: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> probes: array<f32, 4>;

@compute @workgroup_size(4)
fn cs_main(@builtin(local_invocation_index) i: u32) {
  let size = textureDimensions(gx_depth);
  if (i < 3u) {
    let y = min(u32(probe_rows[i] * f32(size.y)), size.y - 1u);
    probes[i] = textureLoad(gx_depth, vec2u(size.x / 2u, y), 0).r;
  } else {
    probes[3] = f32(size.x);
  }
}
)";

enum class ProbeResult : u32 { Pending, Ready, Failed };

struct DepthProbePass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  // Set by the main thread to the first frame recorded after every GX pipeline it needs had compiled
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  // Main thread only
  bool waitedForPipelines = false;
  wgpu::Buffer output;
  wgpu::Buffer readback;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  // Render worker only
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  bool copied = false;
  u32 copiedFrame = 0;
  bool mapRequested = false;
  // Written by the map callback, which may run on any thread
  std::atomic<ProbeResult> result = ProbeResult::Pending;
  std::mutex mutex;
  std::array<f32, 4> values{};
};

DepthProbePass sProbe;

// Built from DepthProbeRows with enough digits to round-trip, so the shader reads exactly the rows the check assumes
std::string depth_probe_wgsl() {
  char rows[160];
  std::snprintf(rows, sizeof(rows), "var<private> probe_rows: array<f32, 3> = array(%.9g, %.9g, %.9g);\n",
                DepthProbeRows[0], DepthProbeRows[1], DepthProbeRows[2]);
  return std::string{aurora::render::FrameWgsl} + rows + std::string{DepthProbeWgsl};
}

wgpu::ComputePipeline create_depth_probe_pipeline(const PassContext& ctx, const wgpu::BindGroupLayout& own) {
  const std::string code = depth_probe_wgsl();
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.depth_probe"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), own};
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "demo.depth_probe",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "demo.depth_probe",
      .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  return ctx.device().CreateComputePipeline(&descriptor);
}

void declare_depth_probe(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxDepth, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_depth_probe(PassContext& ctx, void* userdata) {
  auto& probe = *static_cast<DepthProbePass*>(userdata);
  probe.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!probe.pipeline || probe.passLayout.Get() != ctx.pass_layout().Get()) {
    probe.pipeline = create_depth_probe_pipeline(ctx, probe.layout);
    probe.passLayout = ctx.pass_layout();
  }
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(probe.pipeline);
  compute.SetBindGroup(2, probe.group);
  compute.DispatchWorkgroups(1);
  compute.End();
  if (!probe.copied && ctx.frame() >= probe.armedFrame.load(std::memory_order_acquire)) {
    ctx.encoder().CopyBufferToBuffer(probe.output, 0, probe.readback, 0, ProbeBytes);
    probe.copied = true;
    probe.copiedFrame = ctx.frame();
  }
}

void depth_probe_mapped(DepthProbePass& probe, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    probe.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const f32*>(probe.readback.GetConstMappedRange(0, ProbeBytes));
  if (mapped != nullptr) {
    std::lock_guard lock{probe.mutex};
    std::memcpy(probe.values.data(), mapped, ProbeBytes);
  }
  probe.readback.Unmap();
  probe.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

// Runs once the frame that copied the probe has been submitted, so the map waits on that submission
void depth_probe_submitted(u32 frame, void* userdata) {
  auto& probe = *static_cast<DepthProbePass*>(userdata);
  if (!probe.copied || probe.mapRequested || frame != probe.copiedFrame) {
    return;
  }
  probe.mapRequested = true;
  probe.readback.MapAsync(
      wgpu::MapMode::Read, 0, ProbeBytes, wgpu::CallbackMode::AllowSpontaneous,
      [&probe](wgpu::MapAsyncStatus status, wgpu::StringView) { depth_probe_mapped(probe, status); });
}

void wait_for_pipelines() {
  const auto deadline = std::chrono::steady_clock::now() + PipelineWait;
  while (pipelines_compiling() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }
}

bool register_depth_probe(std::string& error) {
  const wgpu::Device device = aurora::gfx::device();
  const wgpu::BufferDescriptor outputDescriptor{
      .label = "demo.depth_probe output",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
      .size = ProbeBytes,
  };
  sProbe.output = device.CreateBuffer(&outputDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "demo.depth_probe readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = ProbeBytes,
  };
  sProbe.readback = device.CreateBuffer(&readbackDescriptor);
  const wgpu::BindGroupLayoutEntry layoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute,
      .buffer = {.type = wgpu::BufferBindingType::Storage},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.depth_probe",
      .entryCount = 1,
      .entries = &layoutEntry,
  };
  sProbe.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::BindGroupEntry groupEntry{.binding = 0, .buffer = sProbe.output, .size = ProbeBytes};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.depth_probe",
      .layout = sProbe.layout,
      .entryCount = 1,
      .entries = &groupEntry,
  };
  sProbe.group = device.CreateBindGroup(&groupDescriptor);
  sProbe.id = aurora::render::register_pass({
      .name = "demo.depth_probe",
      .stage = aurora::render::Stage::FrameEnd,
      .kind = aurora::render::PassKind::Compute,
      .declare = declare_depth_probe,
      .encode = encode_depth_probe,
      .afterSubmit = depth_probe_submitted,
      .userdata = &sProbe,
  });
  if (sProbe.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.depth_probe";
    return false;
  }
  return true;
}
constexpr uint64_t MotionProbeBytes = MotionProbeFloats * sizeof(f32);

constexpr std::string_view MotionProbeWgsl = R"(
struct MotionProbeParams {
  ndc: array<vec4f, 2>,
  frame: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(1) var<uniform> params: MotionProbeParams;
@group(1) @binding(0) var gbuf_motion: texture_2d<f32>;
@group(1) @binding(1) var gbuf_reactive: texture_2d<f32>;
@group(1) @binding(2) var gx_depth: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> probes: array<f32, 36>;

@compute @workgroup_size(4)
fn cs_main(@builtin(local_invocation_index) i: u32) {
  let size = textureDimensions(gbuf_motion);
  let pair = params.ndc[i / 2u];
  let ndc = select(pair.xy, pair.zw, (i & 1u) != 0u);
  let pos = vec2f((ndc.x * 0.5 + 0.5) * f32(size.x), (0.5 - ndc.y * 0.5) * f32(size.y));
  let px = vec2u(clamp(pos, vec2f(0.0), vec2f(size) - vec2f(1.0)));
  let base = i * 8u;
  let motion = textureLoad(gbuf_motion, px, 0).xy;
  probes[base] = motion.x;
  probes[base + 1u] = motion.y;
  probes[base + 2u] = textureLoad(gbuf_reactive, px, 0).r;
  probes[base + 3u] = textureLoad(gx_depth, px, 0).r;
  probes[base + 4u] = f32(px.x);
  probes[base + 5u] = f32(px.y);
  probes[base + 6u] = 0.0;
  probes[base + 7u] = 0.0;
  if (i == 0u) {
    probes[32] = f32(params.frame);
    probes[33] = f32(size.x);
    probes[34] = f32(size.y);
    probes[35] = 1.0;
  }
}
)";

struct MotionProbePass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  // Main thread only
  bool waitedForPipelines = false;
  wgpu::Buffer output;
  wgpu::Buffer readback;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  // Render worker only
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  bool copied = false;
  u32 copiedFrame = 0;
  bool mapRequested = false;
  std::atomic<ProbeResult> result = ProbeResult::Pending;
  std::mutex mutex;
  std::array<f32, MotionProbeFloats> values{};
};

MotionProbePass sMotion;

wgpu::ComputePipeline create_motion_probe_pipeline(const PassContext& ctx, const wgpu::BindGroupLayout& own) {
  const std::string code = std::string{aurora::render::FrameWgsl} + std::string{MotionProbeWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.motion_probe"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), own};
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "demo.motion_probe",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "demo.motion_probe",
      .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  return ctx.device().CreateComputePipeline(&descriptor);
}

void declare_motion_probe(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::Motion, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::Reactive, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::GxDepth, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_motion_probe(PassContext& ctx, void* userdata) {
  auto& probe = *static_cast<MotionProbePass*>(userdata);
  probe.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!probe.pipeline || probe.passLayout.Get() != ctx.pass_layout().Get()) {
    probe.pipeline = create_motion_probe_pipeline(ctx, probe.layout);
    probe.passLayout = ctx.pass_layout();
  }
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(probe.pipeline);
  compute.SetBindGroup(2, probe.group);
  compute.DispatchWorkgroups(1);
  compute.End();
  if (!probe.copied && ctx.frame() >= probe.armedFrame.load(std::memory_order_acquire)) {
    ctx.encoder().CopyBufferToBuffer(probe.output, 0, probe.readback, 0, MotionProbeBytes);
    probe.copied = true;
    probe.copiedFrame = ctx.frame();
  }
}

void motion_probe_mapped(MotionProbePass& probe, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    probe.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const f32*>(probe.readback.GetConstMappedRange(0, MotionProbeBytes));
  if (mapped != nullptr) {
    std::lock_guard lock{probe.mutex};
    std::memcpy(probe.values.data(), mapped, MotionProbeBytes);
  }
  probe.readback.Unmap();
  probe.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void motion_probe_submitted(u32 frame, void* userdata) {
  auto& probe = *static_cast<MotionProbePass*>(userdata);
  if (!probe.copied || probe.mapRequested || frame != probe.copiedFrame) {
    return;
  }
  probe.mapRequested = true;
  probe.readback.MapAsync(
      wgpu::MapMode::Read, 0, MotionProbeBytes, wgpu::CallbackMode::AllowSpontaneous,
      [&probe](wgpu::MapAsyncStatus status, wgpu::StringView) { motion_probe_mapped(probe, status); });
}

bool register_motion_probe(std::string& error) {
  const wgpu::Device device = aurora::gfx::device();
  const wgpu::BufferDescriptor outputDescriptor{
      .label = "demo.motion_probe output",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
      .size = MotionProbeBytes,
  };
  sMotion.output = device.CreateBuffer(&outputDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "demo.motion_probe readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = MotionProbeBytes,
  };
  sMotion.readback = device.CreateBuffer(&readbackDescriptor);
  const wgpu::BindGroupLayoutEntry layoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute,
      .buffer = {.type = wgpu::BufferBindingType::Storage},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.motion_probe",
      .entryCount = 1,
      .entries = &layoutEntry,
  };
  sMotion.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::BindGroupEntry groupEntry{.binding = 0, .buffer = sMotion.output, .size = MotionProbeBytes};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.motion_probe",
      .layout = sMotion.layout,
      .entryCount = 1,
      .entries = &groupEntry,
  };
  sMotion.group = device.CreateBindGroup(&groupDescriptor);
  sMotion.id = aurora::render::register_pass({
      .name = "demo.motion_probe",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Compute,
      .paramsSize = sizeof(MotionProbeParams),
      .declare = declare_motion_probe,
      .encode = encode_motion_probe,
      .afterSubmit = motion_probe_submitted,
      .userdata = &sMotion,
  });
  if (sMotion.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.motion_probe";
    return false;
  }
  return true;
}

constexpr std::string_view MotionViewWgsl = R"(
@group(1) @binding(0) var gbuf_motion: texture_2d<f32>;

var<private> positions: array<vec2f, 3> = array(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(positions[vi], 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let m = textureLoad(gbuf_motion, vec2u(pos.xy), 0).xy;
  if (any(abs(m) >= vec2f(2.0))) {
    return vec4f(1.0, 0.0, 1.0, 1.0);
  }
  return vec4f(0.5 + 32.0 * m, 0.5, 1.0);
}
)";

struct MotionViewPass {
  PassId id = aurora::render::InvalidPass;
  // Render worker only
  wgpu::RenderPipeline pipeline;
  wgpu::PipelineLayout layout;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};

MotionViewPass sMotionView;

void declare_motion_view(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::Motion, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::GxColor, Access::ColorRMW);
}

void encode_motion_view(PassContext& ctx, void* userdata) {
  auto& view = *static_cast<MotionViewPass*>(userdata);
  constexpr aurora::render::TexRef Color{1};
  wgpu::ColorTargetState target = ctx.color_target(Color);
  if (!view.pipeline || view.format != target.format || view.layout.Get() != ctx.pipeline_layout().Get()) {
    const std::string code{MotionViewWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.motion_view"};
    const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
    // EFB alpha is the game's shadow stencil, so the view replaces RGB only
    target.writeMask = wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
    const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
    const wgpu::RenderPipelineDescriptor descriptor{
        .label = "demo.motion_view",
        .layout = ctx.pipeline_layout(),
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .multisample = {.count = ctx.sample_count(Color)},
        .fragment = &fragment,
    };
    view.pipeline = ctx.device().CreateRenderPipeline(&descriptor);
    view.layout = ctx.pipeline_layout();
    view.format = target.format;
  }
  const wgpu::RenderPassEncoder render = ctx.begin_render();
  if (!render) {
    return;
  }
  render.SetPipeline(view.pipeline);
  render.Draw(3);
  render.End();
}

bool register_motion_view(std::string& error) {
  sMotionView.id = aurora::render::register_pass({
      .name = "demo.motion_view",
      .stage = aurora::render::Stage::BeforeHud,
      .kind = aurora::render::PassKind::Render,
      .declare = declare_motion_view,
      .encode = encode_motion_view,
      .userdata = &sMotionView,
  });
  if (sMotionView.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.motion_view";
    return false;
  }
  return true;
}
} // namespace

// Aurora's pipeline threads update the counter through atomic_ref, so it is read the same way
bool pipelines_compiling() noexcept {
  auto& queued = const_cast<uint32_t&>(aurora_get_stats()->queuedPipelines);
#if defined(__cpp_lib_atomic_ref)
  return std::atomic_ref{queued}.load(std::memory_order_relaxed) != 0;
#else
  return __atomic_load_n(&queued, __ATOMIC_RELAXED) != 0;
#endif
}

bool register_graph_passes(const GraphPassOptions& options, std::string& error) {
  if (options.vignette) {
    sVignette.anchor = options.vignetteAnchor;
    sVignette.id = aurora::render::register_pass({
        .name = "demo.vignette",
        .stage = aurora::render::Stage::BeforeHud,
        .kind = aurora::render::PassKind::Render,
        .paramsSize = sizeof(VignetteParams),
        .declare = declare_vignette,
        .encode = encode_vignette,
        .userdata = &sVignette,
    });
    if (sVignette.id == aurora::render::InvalidPass) {
      error = "register_pass refused demo.vignette";
      return false;
    }
  }
  if (options.depthCheck && !register_depth_probe(error)) {
    return false;
  }
  if (options.motionCheck && !register_motion_probe(error)) {
    return false;
  }
  if (options.motionView && !register_motion_view(error)) {
    return false;
  }
  return options.gpuCull == 0 ||
         register_cull_passes(
             {.instances = options.gpuCull, .check = options.cullCheck, .timings = options.cullTimings}, error);
}

void update_graph_passes(u32 frame) {
  update_cull_passes(frame);
  if (sProbe.id != aurora::render::InvalidPass && frame >= DepthCheckFrame &&
      sProbe.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX) {
    // Draws whose pipeline is still compiling are skipped, so depth read before they finish misses part of the scene
    if (!std::exchange(sProbe.waitedForPipelines, true)) {
      wait_for_pipelines();
    }
    if (!pipelines_compiling()) {
      sProbe.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
    }
  }
  if (sMotion.id != aurora::render::InvalidPass) {
    MotionProbeParams params{.frame = frame};
    const auto points = motion_probe_points(frame);
    for (u32 i = 0; i < MotionProbeCount; ++i) {
      params.ndc[i * 2] = points[i][0];
      params.ndc[i * 2 + 1] = points[i][1];
    }
    aurora::render::set_pass_params(sMotion.id, &params, sizeof(params));
    if (frame >= MotionCheckFrame && sMotion.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX) {
      if (!std::exchange(sMotion.waitedForPipelines, true)) {
        wait_for_pipelines();
      }
      if (!pipelines_compiling()) {
        sMotion.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
      }
    }
  }
  if (sVignette.id == aurora::render::InvalidPass) {
    return;
  }
  const VignetteParams params = vignette_params(frame);
  aurora::render::set_pass_params(sVignette.id, &params, sizeof(params));
}

void finish_graph_passes() {
  finish_cull_passes();
  // An unarmed or disabled probe copied nothing, so there is no readback to wait for
  const auto waiting = [](PassId id, const std::atomic_uint32_t& encodes, const std::atomic_uint32_t& armed) {
    return id != aurora::render::InvalidPass && encodes.load(std::memory_order_relaxed) != 0 &&
           armed.load(std::memory_order_relaxed) != UINT32_MAX;
  };
  const bool depth = waiting(sProbe.id, sProbe.encodes, sProbe.armedFrame);
  const bool motion = waiting(sMotion.id, sMotion.encodes, sMotion.armedFrame);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (((depth && sProbe.result.load(std::memory_order_acquire) == ProbeResult::Pending) ||
          (motion && sMotion.result.load(std::memory_order_acquire) == ProbeResult::Pending)) &&
         std::chrono::steady_clock::now() < deadline) {
    aurora::gfx::synchronize();
    aurora::gfx::device().Tick();
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
}

GraphPassReport graph_pass_report() {
  GraphPassReport report{
      .vignetteEncodes = sVignette.encodes.load(std::memory_order_relaxed),
      .probeEncodes = sProbe.encodes.load(std::memory_order_relaxed),
      .probeReady = sProbe.result.load(std::memory_order_acquire) == ProbeResult::Ready,
      .source = aurora::render::scene_depth_source(),
      .cull = cull_report(),
      .motionEncodes = sMotion.encodes.load(std::memory_order_relaxed),
      .motionReady = sMotion.result.load(std::memory_order_acquire) == ProbeResult::Ready,
  };
  if (report.probeReady) {
    std::lock_guard lock{sProbe.mutex};
    report.probe = sProbe.values;
  }
  if (report.motionReady) {
    std::lock_guard lock{sMotion.mutex};
    report.motion = sMotion.values;
  }
  return report;
}

void unregister_graph_passes() {
  const bool cull = unregister_cull_passes();
  if (!cull && sVignette.id == aurora::render::InvalidPass && sProbe.id == aurora::render::InvalidPass &&
      sMotion.id == aurora::render::InvalidPass && sMotionView.id == aurora::render::InvalidPass) {
    return;
  }
  for (PassId* id : {&sVignette.id, &sProbe.id, &sMotion.id, &sMotionView.id}) {
    if (*id != aurora::render::InvalidPass) {
      aurora::render::unregister_pass(*id);
      *id = aurora::render::InvalidPass;
    }
  }
  // The worker may still be inside encode, so the pipelines and buffers outlive it until the queue drains
  aurora::gfx::synchronize();
  release_cull_passes();
  sVignette.pipeline = {};
  sVignette.layout = {};
  sProbe.pipeline = {};
  sProbe.passLayout = {};
  sProbe.group = {};
  sProbe.layout = {};
  sProbe.readback = {};
  sProbe.output = {};
  sProbe.waitedForPipelines = false;
  sMotion.pipeline = {};
  sMotion.passLayout = {};
  sMotion.group = {};
  sMotion.layout = {};
  sMotion.readback = {};
  sMotion.output = {};
  sMotion.waitedForPipelines = false;
  sMotionView.pipeline = {};
  sMotionView.layout = {};
}
} // namespace render_demo

#include "graph_passes.hpp"

#include "graph_checks.hpp"
#include "scene.hpp"

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>
#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

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

struct FogBandParams {
  f32 a;
  f32 b;
  f32 c;
  u32 rgb;
};

struct FogCheckParams {
  FogBandParams bands[2];
  u32 reversedZ;
  u32 pad[3];
};
static_assert(sizeof(FogCheckParams) == 48);

constexpr uint64_t FogCheckBytes = FogCheckWords * sizeof(u32);

// The lock bit picks the band, so band 0 must be the lock-0 PERSP_LIN band
constexpr std::string_view FogCheckWgsl = R"(
struct FogBand {
  a: f32,
  b: f32,
  c: f32,
  rgb: u32,
};
struct FogCheckParams {
  bands: array<FogBand, 2>,
  reversed_z: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(1) var<uniform> params: FogCheckParams;
@group(1) @binding(0) var gbuf_fog: texture_2d<u32>;
@group(1) @binding(1) var gx_depth: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> counts: array<atomic<u32>, 16>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gbuf_fog);
  if (all(id.xy == vec2u(0u))) {
    atomicStore(&counts[10], size.x);
    atomicStore(&counts[11], size.y);
    atomicStore(&counts[12], 1u);
  }
  if (any(id.xy >= size)) {
    return;
  }
  let rec = textureLoad(gbuf_fog, id.xy, 0);
  let d = textureLoad(gx_depth, id.xy, 0).r;
  let reversed = params.reversed_z != 0u;
  let written = any(rec != vec4u(0u));
  if (d == select(1.0, 0.0, reversed)) {
    if (written) {
      atomicAdd(&counts[1], 1u);
    }
    return;
  }
  atomicAdd(&counts[0], 1u);
  if (!written) {
    atomicAdd(&counts[2], 1u);
    return;
  }
  let lock = rec.a & 1u;
  let band = params.bands[lock];
  if (((rec.r << 16u) | (rec.g << 8u) | rec.b) != band.rgb) {
    atomicAdd(&counts[3], 1u);
  }
  let zw = select(d, 1.0 - d, reversed);
  let f = clamp(band.a / (band.b - zw) - band.c, 0.0, 1.0);
  let w_ref = select(1.0 - exp2(-8.0 * f), f, lock == 0u);
  let q = rec.a >> 1u;
  let err = abs(f32(q) / 127.0 - w_ref);
  if (err > 0.55 / 127.0) {
    atomicAdd(&counts[4], 1u);
  }
  atomicMax(&counts[9], bitcast<u32>(err));
  if (lock == 0u) {
    atomicAdd(&counts[5], 1u);
    atomicMax(&counts[7], 127u - min(q, 127u));
    atomicMax(&counts[8], q);
  } else {
    atomicAdd(&counts[6], 1u);
  }
}
)";

struct FogCheckPass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  // Main thread only
  bool waitedForPipelines = false;
  FogCheckParams params{};
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
  std::array<u32, FogCheckWords> values{};
};

FogCheckPass sFog;

f32 truncate_fog_float(f32 value) noexcept { return std::bit_cast<f32>(std::bit_cast<u32>(value) & 0xFFFFF000u); }

// Rebuilt from GXSetFog's f32 maths and the BP register fields, so the check can hold q to 0.55 of a code
FogBandParams fog_band_params(const FogParams& fog) noexcept {
  const f32 aCoeff = (fog.farZ * fog.nearZ) / ((fog.farZ - fog.nearZ) * (fog.end - fog.start));
  const f32 bCoeff = fog.farZ / (fog.farZ - fog.nearZ);
  const f32 cCoeff = fog.start / (fog.end - fog.start);
  f32 bMantissa = bCoeff;
  int bExponent = 0;
  while (bMantissa > 1.f) {
    bMantissa *= 0.5f;
    ++bExponent;
  }
  while (bMantissa > 0.f && bMantissa < 0.5f) {
    bMantissa *= 2.f;
    --bExponent;
  }
  const f32 a = aCoeff / std::ldexp(1.f, bExponent + 1);
  const u32 bMantissaRaw = static_cast<u32>(8.388638e6f * bMantissa) & 0xFFFFFFu;
  const int bScale = (bExponent + 1) & 0x1F;
  return {
      .a = std::ldexp(truncate_fog_float(a), bScale),
      .b = std::ldexp(static_cast<f32>(bMantissaRaw) / 8388638.f, bScale - 1),
      .c = truncate_fog_float(cCoeff),
      .rgb = u32{fog.color.r} << 16 | u32{fog.color.g} << 8 | u32{fog.color.b},
  };
}

wgpu::ComputePipeline create_fog_check_pipeline(const PassContext& ctx, const wgpu::BindGroupLayout& own) {
  const std::string code = std::string{aurora::render::FrameWgsl} + std::string{FogCheckWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.fog_check"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), own};
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "demo.fog_check",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "demo.fog_check",
      .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  return ctx.device().CreateComputePipeline(&descriptor);
}

void declare_fog_check(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::Fog, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::GxDepth, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_fog_check(PassContext& ctx, void* userdata) {
  auto& check = *static_cast<FogCheckPass*>(userdata);
  check.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!check.pipeline || check.passLayout.Get() != ctx.pass_layout().Get()) {
    check.pipeline = create_fog_check_pipeline(ctx, check.layout);
    check.passLayout = ctx.pass_layout();
  }
  constexpr aurora::render::TexRef Fog{0};
  const wgpu::Extent3D size = ctx.extent(Fog);
  ctx.encoder().ClearBuffer(check.output, 0, FogCheckBytes);
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(check.pipeline);
  compute.SetBindGroup(2, check.group);
  compute.DispatchWorkgroups((size.width + 7) / 8, (size.height + 7) / 8);
  compute.End();
  if (!check.copied && ctx.frame() >= check.armedFrame.load(std::memory_order_acquire)) {
    ctx.encoder().CopyBufferToBuffer(check.output, 0, check.readback, 0, FogCheckBytes);
    check.copied = true;
    check.copiedFrame = ctx.frame();
  }
}

void fog_check_mapped(FogCheckPass& check, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    check.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const u32*>(check.readback.GetConstMappedRange(0, FogCheckBytes));
  if (mapped != nullptr) {
    std::lock_guard lock{check.mutex};
    std::memcpy(check.values.data(), mapped, FogCheckBytes);
  }
  check.readback.Unmap();
  check.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void fog_check_submitted(u32 frame, void* userdata) {
  auto& check = *static_cast<FogCheckPass*>(userdata);
  if (!check.copied || check.mapRequested || frame != check.copiedFrame) {
    return;
  }
  check.mapRequested = true;
  check.readback.MapAsync(wgpu::MapMode::Read, 0, FogCheckBytes, wgpu::CallbackMode::AllowSpontaneous,
                          [&check](wgpu::MapAsyncStatus status, wgpu::StringView) { fog_check_mapped(check, status); });
}

bool register_fog_check(std::string& error) {
  const auto& bands = fog_bands();
  sFog.params = {
      .bands = {fog_band_params(bands[0]), fog_band_params(bands[1])},
      .reversedZ = aurora::gfx::uses_reversed_z() ? 1u : 0u,
  };
  const wgpu::Device device = aurora::gfx::device();
  const wgpu::BufferDescriptor outputDescriptor{
      .label = "demo.fog_check output",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
      .size = FogCheckBytes,
  };
  sFog.output = device.CreateBuffer(&outputDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "demo.fog_check readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = FogCheckBytes,
  };
  sFog.readback = device.CreateBuffer(&readbackDescriptor);
  const wgpu::BindGroupLayoutEntry layoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute,
      .buffer = {.type = wgpu::BufferBindingType::Storage},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.fog_check",
      .entryCount = 1,
      .entries = &layoutEntry,
  };
  sFog.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::BindGroupEntry groupEntry{.binding = 0, .buffer = sFog.output, .size = FogCheckBytes};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.fog_check",
      .layout = sFog.layout,
      .entryCount = 1,
      .entries = &groupEntry,
  };
  sFog.group = device.CreateBindGroup(&groupDescriptor);
  sFog.id = aurora::render::register_pass({
      .name = "demo.fog_check",
      .stage = aurora::render::Stage::OpaqueDone,
      .kind = aurora::render::PassKind::Compute,
      .paramsSize = sizeof(FogCheckParams),
      .declare = declare_fog_check,
      .encode = encode_fog_check,
      .afterSubmit = fog_check_submitted,
      .userdata = &sFog,
  });
  if (sFog.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.fog_check";
    return false;
  }
  return true;
}

constexpr u32 AoDynamicSlots = 6;

struct AoCheckParams {
  u32 probeCount;
  u32 frame;
  u32 flags;
  u32 dynamicCount;
  f32 dynamic[AoDynamicSlots * 4];
};
static_assert(sizeof(AoCheckParams) == 112);
static_assert(AoTrailProbeCount <= AoDynamicSlots * 2);

constexpr u32 AoCheckHeaderWords = 8;
constexpr u32 AoCheckProbeWords = 3;
constexpr uint64_t AoCheckBytes =
    (AoCheckHeaderWords + (AoCheckMaxProbes + AoDynamicSlots * 2) * AoCheckProbeWords) * sizeof(u32);
constexpr size_t AoCheckRing = 16;
constexpr auto AoCompileWait = std::chrono::milliseconds{50};

enum AoCheckWord : u32 { AoReceivers, AoPixels, AoWidth, AoHeight, AoWritten, AoFrame, AoDynamic };
enum AoCheckFlag : u32 { AoAccumulate = 1u << 0, AoCopyStats = 1u << 1 };
enum class AoSlotState : u32 { Free, Copied, Mapping };

constexpr std::string_view AoCheckWgsl = R"(
struct AoCheckParams {
  probe_count: u32,
  frame: u32,
  flags: u32,
  dynamic_count: u32,
  dynamic: array<vec4f, 6>,
};
struct AoProbe {
  ndc: vec2f,
  offset_px: f32,
  cls: u32,
};
@group(0) @binding(1) var<uniform> params: AoCheckParams;
@group(1) @binding(0) var ao_final: texture_2d<f32>;
@group(1) @binding(1) var scene_normal: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> results: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read> probes: array<AoProbe>;
@group(2) @binding(2) var<storage, read_write> stats: array<vec4f>;

fn receiver_code(p: vec2i) -> u32 {
  return u32(round(saturate(textureLoad(scene_normal, p, 0).a) * 3.0));
}

fn probe_centre(index: u32, size: vec2u) -> vec2f {
  if (index < params.probe_count) {
    let probe = probes[index];
    return (probe.ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size) + vec2f(probe.offset_px, 0.0);
  }
  let dynamic_index = index - params.probe_count;
  let slot = params.dynamic[dynamic_index / 2u];
  let ndc = select(slot.xy, slot.zw, (dynamic_index & 1u) != 0u);
  return (ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let size = textureDimensions(ao_final);
  if (all(id.xy == vec2u(0u))) {
    atomicStore(&results[2], size.x);
    atomicStore(&results[3], size.y);
    atomicStore(&results[4], 1u);
    atomicStore(&results[5], params.frame);
    atomicStore(&results[6], params.dynamic_count);
  }
  let index = id.y * groups.x * 8u + id.x;
  if (index < params.probe_count + params.dynamic_count) {
    let p = clamp(vec2i(floor(probe_centre(index, size))), vec2i(0), vec2i(size) - 1);
    let ao = textureLoad(ao_final, p, 0);
    let base = 8u + index * 3u;
    atomicStore(&results[base], bitcast<u32>(ao.r));
    atomicStore(&results[base + 1u], receiver_code(p));
    atomicStore(&results[base + 2u], u32(round(ao.g * 255.0)) | (u32(round(ao.b * 255.0)) << 8u));
  }
  if (any(id.xy >= size)) {
    return;
  }
  atomicAdd(&results[1], 1u);
  let receiver = receiver_code(vec2i(id.xy)) >= 1u;
  if (receiver) {
    atomicAdd(&results[0], 1u);
  }
  if ((params.flags & 1u) != 0u && receiver) {
    let a = textureLoad(ao_final, vec2i(id.xy), 0).r;
    let pixel = id.y * size.x + id.x;
    stats[pixel] = stats[pixel] + vec4f(a, a * a, 1.0, 0.0);
  }
}
)";

struct AoCheckSlot {
  wgpu::Buffer readback;
  std::atomic<AoSlotState> state = AoSlotState::Free;
  u32 frame = 0;
};

struct AoCheckPass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  bool waitedForPipelines = false;
  std::chrono::steady_clock::time_point deadline;
  AoCheckParams params{};
  wgpu::Buffer output;
  wgpu::Buffer probeBuffer;
  wgpu::Buffer stats;
  wgpu::Buffer statsReadback;
  wgpu::Extent3D statsSize{};
  std::atomic<AoSlotState> statsState = AoSlotState::Free;
  bool statsCopied = false;
  u32 statsFrame = 0;
  u32 sigmaFrames = 0;
  u32 statsFrames = 0;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  std::array<AoCheckSlot, AoCheckRing> slots;
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  std::mutex mutex;
  std::vector<AoProbe> probes;
  std::vector<size_t> contactProbes;
  u32 frames = 0;
  u32 failedMaps = 0;
  u32 receivers = 0;
  u32 pixels = 0;
  std::vector<double> sums;
  std::vector<u32> receiverFrames;
  AoTemporalSummary temporal;
};

AoCheckPass sAo;

wgpu::ComputePipeline create_ao_check_pipeline(const PassContext& ctx, const wgpu::BindGroupLayout& own) {
  const std::string code{AoCheckWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.ao_check"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), own};
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "demo.ao_check",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = "demo.ao_check",
      .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
      .compute = {.module = module, .entryPoint = "cs_main"},
  };
  return ctx.device().CreateComputePipeline(&descriptor);
}

void create_ao_stats(AoCheckPass& check, const wgpu::Device& device, const wgpu::Extent3D& size) {
  const uint64_t bytes = std::max<uint64_t>(uint64_t{size.width} * size.height * 4 * sizeof(f32), 16);
  const wgpu::BufferDescriptor statsDescriptor{
      .label = "demo.ao_check stats",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
      .size = bytes,
  };
  check.stats = device.CreateBuffer(&statsDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "demo.ao_check stats readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = bytes,
  };
  check.statsReadback = device.CreateBuffer(&readbackDescriptor);
  check.statsSize = size;
  check.sigmaFrames = 0;
  const uint64_t probeBytes = check.probes.size() * sizeof(AoProbe);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = check.output, .size = AoCheckBytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = check.probeBuffer, .size = probeBytes},
      wgpu::BindGroupEntry{.binding = 2, .buffer = check.stats, .size = bytes},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.ao_check",
      .layout = check.layout,
      .entryCount = groupEntries.size(),
      .entries = groupEntries.data(),
  };
  check.group = device.CreateBindGroup(&groupDescriptor);
}

void declare_ao_check(PassBuilder& builder, void*) {
  builder.after(aurora::render::phase::Ao);
  builder.use(aurora::render::resource::AoFinal, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::GxNormal, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_ao_check(PassContext& ctx, void* userdata) {
  auto& check = *static_cast<AoCheckPass*>(userdata);
  check.encodes.fetch_add(1, std::memory_order_relaxed);
  AoCheckParams params{};
  if (const std::span<const std::byte> bytes = ctx.params(); bytes.size() >= sizeof(params)) {
    std::memcpy(&params, bytes.data(), sizeof(params));
  }
  if (!check.pipeline || check.passLayout.Get() != ctx.pass_layout().Get()) {
    check.pipeline = create_ao_check_pipeline(ctx, check.layout);
    check.passLayout = ctx.pass_layout();
  }
  constexpr aurora::render::TexRef AoFinal{0};
  const wgpu::Extent3D size = ctx.extent(AoFinal);
  if (!check.stats || check.statsSize.width != size.width || check.statsSize.height != size.height) {
    create_ao_stats(check, ctx.device(), size);
  }
  ctx.encoder().ClearBuffer(check.output, 0, AoCheckBytes);
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(check.pipeline);
  compute.SetBindGroup(2, check.group);
  compute.DispatchWorkgroups((size.width + 7) / 8, (size.height + 7) / 8);
  compute.End();
  check.sigmaFrames += (params.flags & AoAccumulate) != 0 ? 1 : 0;
  if ((params.flags & AoCopyStats) != 0 && !check.statsCopied &&
      check.statsState.load(std::memory_order_acquire) == AoSlotState::Free) {
    ctx.encoder().CopyBufferToBuffer(check.stats, 0, check.statsReadback, 0, check.stats.GetSize());
    check.statsCopied = true;
    check.statsFrame = ctx.frame();
    check.statsFrames = check.sigmaFrames;
    check.statsState.store(AoSlotState::Copied, std::memory_order_release);
  }
  if (ctx.frame() < check.armedFrame.load(std::memory_order_acquire)) {
    return;
  }
  for (AoCheckSlot& slot : check.slots) {
    if (slot.state.load(std::memory_order_acquire) == AoSlotState::Free) {
      ctx.encoder().CopyBufferToBuffer(check.output, 0, slot.readback, 0, AoCheckBytes);
      slot.frame = ctx.frame();
      slot.state.store(AoSlotState::Copied, std::memory_order_release);
      break;
    }
  }
}

void ao_check_frame(AoCheckPass& check, const u32* mapped) {
  const u32 frame = mapped[AoFrame];
  const size_t probeCount = check.probes.size();
  const auto value = [&](size_t index) { return std::bit_cast<f32>(mapped[AoCheckHeaderWords + index * 3]); };
  const auto receiver = [&](size_t index) { return mapped[AoCheckHeaderWords + index * 3 + 1] >= 1; };
  if (frame >= AoSigmaFirst && frame <= AoSigmaLast) {
    ++check.frames;
    check.receivers = mapped[AoReceivers];
    check.pixels = mapped[AoPixels];
    for (size_t i = 0; i < probeCount; ++i) {
      check.sums[i] += value(i);
      check.receiverFrames[i] += receiver(i) ? 1 : 0;
    }
  }
  if (frame >= AoTrailFirst && frame <= AoTrailLast) {
    AoTrailFrame row{.frame = frame};
    for (size_t i = 0; i < probeCount; ++i) {
      if (check.probes[i].cls == AoProbeClass::Open && receiver(i)) {
        row.open += value(i);
        ++row.openReceivers;
      }
    }
    const size_t dynamic = std::min<size_t>(mapped[AoDynamic], AoDynamicSlots * 2);
    for (size_t i = probeCount; i < probeCount + dynamic; ++i) {
      if (receiver(i)) {
        row.trail += value(i);
        ++row.trailReceivers;
      }
    }
    row.open = row.openReceivers != 0 ? row.open / row.openReceivers : 0.0;
    row.trail = row.trailReceivers != 0 ? row.trail / row.trailReceivers : 0.0;
    check.temporal.trail.push_back(row);
  }
  for (size_t k = 0; k < AoContactFrames.size(); ++k) {
    if (frame != AoContactFrames[k]) {
      continue;
    }
    std::vector<f32>& band = check.temporal.contact[k];
    band.clear();
    for (const size_t i : check.contactProbes) {
      band.push_back(receiver(i) ? value(i) : std::numeric_limits<f32>::quiet_NaN());
    }
  }
}

void ao_check_mapped(AoCheckPass& check, AoCheckSlot& slot, wgpu::MapAsyncStatus status) {
  const auto* mapped = status == wgpu::MapAsyncStatus::Success
                           ? static_cast<const u32*>(slot.readback.GetConstMappedRange(0, AoCheckBytes))
                           : nullptr;
  {
    std::lock_guard lock{check.mutex};
    if (mapped == nullptr || mapped[AoWritten] != 1) {
      ++check.failedMaps;
    } else {
      ao_check_frame(check, mapped);
    }
  }
  if (status == wgpu::MapAsyncStatus::Success) {
    slot.readback.Unmap();
  }
  slot.state.store(AoSlotState::Free, std::memory_order_release);
}

void ao_stats_mapped(AoCheckPass& check, wgpu::MapAsyncStatus status) {
  const uint64_t bytes = check.statsReadback.GetSize();
  const auto* values = status == wgpu::MapAsyncStatus::Success
                           ? static_cast<const f32*>(check.statsReadback.GetConstMappedRange(0, bytes))
                           : nullptr;
  AoSigma sigma{.frames = check.statsFrames, .pixels = check.statsSize.width * check.statsSize.height};
  if (values != nullptr && sigma.frames != 0) {
    double deviation = 0.0;
    double mean = 0.0;
    for (size_t i = 0; i < sigma.pixels; ++i) {
      const double n = values[i * 4 + 2];
      if (n + 0.5 < sigma.frames) {
        continue;
      }
      const double m = values[i * 4] / n;
      deviation += std::sqrt(std::max(values[i * 4 + 1] / n - m * m, 0.0));
      mean += m;
      ++sigma.covered;
    }
    sigma.sigma = sigma.covered != 0 ? deviation / sigma.covered : 0.0;
    sigma.mean = sigma.covered != 0 ? mean / sigma.covered : 0.0;
  }
  {
    std::lock_guard lock{check.mutex};
    check.temporal.sigma = sigma;
  }
  if (status == wgpu::MapAsyncStatus::Success) {
    check.statsReadback.Unmap();
  }
  check.statsState.store(AoSlotState::Free, std::memory_order_release);
}

void ao_check_submitted(u32 frame, void* userdata) {
  auto& check = *static_cast<AoCheckPass*>(userdata);
  if (check.statsState.load(std::memory_order_acquire) == AoSlotState::Copied && check.statsFrame == frame) {
    check.statsState.store(AoSlotState::Mapping, std::memory_order_release);
    check.statsReadback.MapAsync(
        wgpu::MapMode::Read, 0, check.statsReadback.GetSize(), wgpu::CallbackMode::AllowSpontaneous,
        [&check](wgpu::MapAsyncStatus status, wgpu::StringView) { ao_stats_mapped(check, status); });
  }
  for (AoCheckSlot& slot : check.slots) {
    if (slot.state.load(std::memory_order_acquire) != AoSlotState::Copied || slot.frame != frame) {
      continue;
    }
    slot.state.store(AoSlotState::Mapping, std::memory_order_release);
    slot.readback.MapAsync(
        wgpu::MapMode::Read, 0, AoCheckBytes, wgpu::CallbackMode::AllowSpontaneous,
        [&check, &slot](wgpu::MapAsyncStatus status, wgpu::StringView) { ao_check_mapped(check, slot, status); });
  }
}

bool register_ao_check(std::string& error) {
  sAo.probes = ao_check_probes();
  if (sAo.probes.empty() || sAo.probes.size() > AoCheckMaxProbes) {
    error = "demo.ao_check has no room for its probes";
    return false;
  }
  sAo.params = {.probeCount = static_cast<u32>(sAo.probes.size())};
  sAo.sums.assign(sAo.probes.size(), 0.0);
  sAo.receiverFrames.assign(sAo.probes.size(), 0);
  sAo.contactProbes.clear();
  for (size_t i = 0; i < sAo.probes.size(); ++i) {
    if (sAo.probes[i].cls == AoProbeClass::Contact) {
      sAo.contactProbes.push_back(i);
    }
  }
  const wgpu::Device device = aurora::gfx::device();
  const wgpu::BufferDescriptor outputDescriptor{
      .label = "demo.ao_check output",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
      .size = AoCheckBytes,
  };
  sAo.output = device.CreateBuffer(&outputDescriptor);
  const uint64_t probeBytes = sAo.probes.size() * sizeof(AoProbe);
  const wgpu::BufferDescriptor probeDescriptor{
      .label = "demo.ao_check probes",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
      .size = probeBytes,
  };
  sAo.probeBuffer = device.CreateBuffer(&probeDescriptor);
  device.GetQueue().WriteBuffer(sAo.probeBuffer, 0, sAo.probes.data(), probeBytes);
  for (AoCheckSlot& slot : sAo.slots) {
    const wgpu::BufferDescriptor readbackDescriptor{
        .label = "demo.ao_check readback",
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
        .size = AoCheckBytes,
    };
    slot.readback = device.CreateBuffer(&readbackDescriptor);
    slot.state.store(AoSlotState::Free, std::memory_order_relaxed);
  }
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.ao_check",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  sAo.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  sAo.id = aurora::render::register_pass({
      .name = "demo.ao_check",
      .stage = aurora::render::Stage::OpaqueDone,
      .kind = aurora::render::PassKind::Compute,
      .paramsSize = sizeof(AoCheckParams),
      .declare = declare_ao_check,
      .encode = encode_ao_check,
      .afterSubmit = ao_check_submitted,
      .userdata = &sAo,
  });
  if (sAo.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.ao_check";
    return false;
  }
  return true;
}

void update_ao_check(u32 frame) {
  const bool armed = sAo.armedFrame.load(std::memory_order_relaxed) != UINT32_MAX;
  sAo.params.frame = frame;
  sAo.params.flags = (armed && frame >= AoSigmaFirst && frame <= AoSigmaLast ? AoAccumulate : 0u) |
                     (frame > AoSigmaLast ? AoCopyStats : 0u);
  const std::vector<std::array<f32, 2>> trail = ao_trail_probes(frame);
  sAo.params.dynamicCount = static_cast<u32>(std::min<size_t>(trail.size(), AoDynamicSlots * 2));
  for (size_t i = 0; i < sAo.params.dynamicCount; ++i) {
    sAo.params.dynamic[i * 2] = trail[i][0];
    sAo.params.dynamic[i * 2 + 1] = trail[i][1];
  }
  aurora::render::set_pass_params(sAo.id, &sAo.params, sizeof(sAo.params));
}

constexpr u32 FwdEnvHeaderWords = 4;
constexpr u32 FwdEnvSampleWords = 8;

// Reads each probe pixel after the forward window, where the strips and the card have blended
constexpr std::string_view FwdEnvCheckWgsl = R"(
struct FwdProbe {
  ndc: vec2f,
  cls: u32,
  pad: u32,
};
@group(1) @binding(0) var gx_color: texture_2d<f32>;
@group(1) @binding(1) var gbuf_fog: texture_2d<u32>;
@group(1) @binding(2) var ao_final: texture_2d<f32>;
@group(1) @binding(3) var scene_normal: texture_2d<f32>;
@group(1) @binding(4) var gx_depth: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> results: array<u32>;
@group(2) @binding(1) var<storage, read> probes: array<FwdProbe>;

fn pack_unorm(v: vec4f) -> u32 {
  let b = vec4u(round(saturate(v) * 255.0));
  return b.r | (b.g << 8u) | (b.b << 16u) | (b.a << 24u);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gx_color);
  if (id.x == 0u) {
    results[0] = size.x;
    results[1] = size.y;
    results[2] = 1u;
  }
  if (id.x >= arrayLength(&probes)) {
    return;
  }
  let centre = (probes[id.x].ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size);
  let p = clamp(vec2i(floor(centre)), vec2i(0), vec2i(size) - 1);
  let base = 4u + id.x * 8u;
  let rec = textureLoad(gbuf_fog, p, 0);
  let n = textureLoad(scene_normal, p, 0);
  results[base] = pack_unorm(textureLoad(gx_color, p, 0));
  results[base + 1u] = rec.r | (rec.g << 8u) | (rec.b << 16u) | (rec.a << 24u);
  results[base + 2u] = pack_unorm(textureLoad(ao_final, p, 0));
  results[base + 3u] = (pack_unorm(vec4f(n.rgb, 0.0)) & 0xFFFFFFu) | (u32(round(saturate(n.a) * 3.0)) << 24u);
  results[base + 4u] = bitcast<u32>(textureLoad(gx_depth, p, 0).r);
  results[base + 5u] = u32(p.x) | (u32(p.y) << 16u);
}
)";

struct FwdEnvCheckPass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  // Main thread only
  bool waitedForPipelines = false;
  std::chrono::steady_clock::time_point deadline;
  std::vector<FwdEnvProbe> probes;
  uint64_t bytes = 0;
  wgpu::Buffer output;
  wgpu::Buffer readback;
  wgpu::Buffer probeBuffer;
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
  std::vector<u32> values;
};

FwdEnvCheckPass sFwdEnv;

void declare_fwd_env_check(PassBuilder& builder, void*) {
  using aurora::render::Missing;
  namespace resource = aurora::render::resource;
  for (const std::string_view name :
       {resource::GxColor, resource::Fog, resource::AoFinal, resource::GxNormal, resource::GxDepth}) {
    builder.use(name, Access::Sampled, Missing::DisablePass);
  }
  builder.side_effect();
}

void encode_fwd_env_check(PassContext& ctx, void* userdata) {
  auto& check = *static_cast<FwdEnvCheckPass*>(userdata);
  check.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!check.pipeline || check.passLayout.Get() != ctx.pass_layout().Get()) {
    const std::string code{FwdEnvCheckWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.fwd_env_check"};
    const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), check.layout};
    const wgpu::PipelineLayoutDescriptor layoutDescriptor{
        .label = "demo.fwd_env_check",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
    };
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "demo.fwd_env_check",
        .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
        .compute = {.module = ctx.device().CreateShaderModule(&moduleDescriptor), .entryPoint = "cs_main"},
    };
    check.pipeline = ctx.device().CreateComputePipeline(&descriptor);
    check.passLayout = ctx.pass_layout();
  }
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(check.pipeline);
  compute.SetBindGroup(2, check.group);
  compute.DispatchWorkgroups((static_cast<u32>(check.probes.size()) + 63) / 64);
  compute.End();
  if (!check.copied && ctx.frame() >= check.armedFrame.load(std::memory_order_acquire)) {
    ctx.encoder().CopyBufferToBuffer(check.output, 0, check.readback, 0, check.bytes);
    check.copied = true;
    check.copiedFrame = ctx.frame();
  }
}

void fwd_env_check_mapped(FwdEnvCheckPass& check, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    check.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const u32*>(check.readback.GetConstMappedRange(0, check.bytes));
  if (mapped != nullptr) {
    std::lock_guard lock{check.mutex};
    check.values.assign(mapped, mapped + check.bytes / sizeof(u32));
  }
  check.readback.Unmap();
  check.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void fwd_env_check_submitted(u32 frame, void* userdata) {
  auto& check = *static_cast<FwdEnvCheckPass*>(userdata);
  if (!check.copied || check.mapRequested || frame != check.copiedFrame) {
    return;
  }
  check.mapRequested = true;
  check.readback.MapAsync(
      wgpu::MapMode::Read, 0, check.bytes, wgpu::CallbackMode::AllowSpontaneous,
      [&check](wgpu::MapAsyncStatus status, wgpu::StringView) { fwd_env_check_mapped(check, status); });
}

bool register_fwd_env_check(std::string& error) {
  sFwdEnv.probes = fwd_env_probes();
  if (sFwdEnv.probes.empty()) {
    error = "demo.fwd_env_check has no probes on screen";
    return false;
  }
  sFwdEnv.bytes = (FwdEnvHeaderWords + sFwdEnv.probes.size() * FwdEnvSampleWords) * sizeof(u32);
  const wgpu::Device device = aurora::gfx::device();
  const auto buffer = [&](const char* label, wgpu::BufferUsage usage, uint64_t size) {
    const wgpu::BufferDescriptor descriptor{.label = label, .usage = usage, .size = size};
    return device.CreateBuffer(&descriptor);
  };
  sFwdEnv.output =
      buffer("demo.fwd_env_check output", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, sFwdEnv.bytes);
  sFwdEnv.readback =
      buffer("demo.fwd_env_check readback", wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, sFwdEnv.bytes);
  const uint64_t probeBytes = sFwdEnv.probes.size() * sizeof(FwdEnvProbe);
  sFwdEnv.probeBuffer =
      buffer("demo.fwd_env_check probes", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, probeBytes);
  device.GetQueue().WriteBuffer(sFwdEnv.probeBuffer, 0, sFwdEnv.probes.data(), probeBytes);
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.fwd_env_check",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  sFwdEnv.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = sFwdEnv.output, .size = sFwdEnv.bytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = sFwdEnv.probeBuffer, .size = probeBytes},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.fwd_env_check",
      .layout = sFwdEnv.layout,
      .entryCount = groupEntries.size(),
      .entries = groupEntries.data(),
  };
  sFwdEnv.group = device.CreateBindGroup(&groupDescriptor);
  sFwdEnv.id = aurora::render::register_pass({
      .name = "demo.fwd_env_check",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Compute,
      .declare = declare_fwd_env_check,
      .encode = encode_fwd_env_check,
      .afterSubmit = fwd_env_check_submitted,
      .userdata = &sFwdEnv,
  });
  if (sFwdEnv.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.fwd_env_check";
    return false;
  }
  return true;
}

constexpr u32 UnlitHeaderWords = 4;
constexpr u32 UnlitSampleWords = 8;

constexpr std::string_view UnlitCheckWgsl = R"(
struct UnlitProbe {
  ndc: vec2f,
  cls: u32,
  pad: u32,
};
@group(1) @binding(0) var gx_color: texture_2d<f32>;
@group(1) @binding(1) var gbuf_surface: texture_2d<u32>;
@group(2) @binding(0) var<storage, read_write> results: array<u32>;
@group(2) @binding(1) var<storage, read> probes: array<UnlitProbe>;

fn pack_unorm(v: vec4f) -> u32 {
  let b = vec4u(round(saturate(v) * 255.0));
  return b.r | (b.g << 8u) | (b.b << 16u) | (b.a << 24u);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gx_color);
  if (id.x == 0u) {
    results[0] = size.x;
    results[1] = size.y;
    results[2] = 1u;
  }
  if (id.x >= arrayLength(&probes)) {
    return;
  }
  let centre = (probes[id.x].ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size);
  let p = clamp(vec2i(floor(centre)), vec2i(0), vec2i(size) - 1);
  let rec = textureLoad(gbuf_surface, min(vec2u(p), textureDimensions(gbuf_surface) - 1u), 0);
  let base = 4u + id.x * 8u;
  results[base] = pack_unorm(textureLoad(gx_color, p, 0));
  results[base + 1u] = rec.x;
  results[base + 2u] = rec.y;
  results[base + 3u] = rec.z;
  results[base + 4u] = rec.w;
}
)";

struct UnlitCheckSlot {
  wgpu::Buffer readback;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  bool copied = false;
  u32 copiedFrame = 0;
  bool mapRequested = false;
  std::atomic<ProbeResult> result = ProbeResult::Pending;
  std::vector<u32> values;
};

struct UnlitCheckPass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  // Main thread only
  bool waitedForPipelines = false;
  std::vector<UnlitProbe> probes;
  uint64_t bytes = 0;
  wgpu::Buffer output;
  wgpu::Buffer probeBuffer;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  // Render worker only
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  std::array<UnlitCheckSlot, 2> slots;
  std::mutex mutex;
};

UnlitCheckPass sUnlit;

void declare_unlit_check(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxColor, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::SurfaceV1, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_unlit_check(PassContext& ctx, void* userdata) {
  auto& check = *static_cast<UnlitCheckPass*>(userdata);
  check.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!check.pipeline || check.passLayout.Get() != ctx.pass_layout().Get()) {
    const std::string code{UnlitCheckWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.unlit_check"};
    const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), check.layout};
    const wgpu::PipelineLayoutDescriptor layoutDescriptor{
        .label = "demo.unlit_check",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
    };
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "demo.unlit_check",
        .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
        .compute = {.module = ctx.device().CreateShaderModule(&moduleDescriptor), .entryPoint = "cs_main"},
    };
    check.pipeline = ctx.device().CreateComputePipeline(&descriptor);
    check.passLayout = ctx.pass_layout();
  }
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(check.pipeline);
  compute.SetBindGroup(2, check.group);
  compute.DispatchWorkgroups((static_cast<u32>(check.probes.size()) + 63) / 64);
  compute.End();
  for (UnlitCheckSlot& slot : check.slots) {
    if (!slot.copied && ctx.frame() >= slot.armedFrame.load(std::memory_order_acquire)) {
      ctx.encoder().CopyBufferToBuffer(check.output, 0, slot.readback, 0, check.bytes);
      slot.copied = true;
      slot.copiedFrame = ctx.frame();
      break;
    }
  }
}

void unlit_check_mapped(UnlitCheckSlot& slot, uint64_t bytes, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    slot.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const u32*>(slot.readback.GetConstMappedRange(0, bytes));
  if (mapped != nullptr) {
    std::lock_guard lock{sUnlit.mutex};
    slot.values.assign(mapped, mapped + bytes / sizeof(u32));
  }
  slot.readback.Unmap();
  slot.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void unlit_check_submitted(u32 frame, void* userdata) {
  auto& check = *static_cast<UnlitCheckPass*>(userdata);
  for (UnlitCheckSlot& slot : check.slots) {
    if (!slot.copied || slot.mapRequested || frame != slot.copiedFrame) {
      continue;
    }
    slot.mapRequested = true;
    const uint64_t bytes = check.bytes;
    slot.readback.MapAsync(
        wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::AllowSpontaneous,
        [&slot, bytes](wgpu::MapAsyncStatus status, wgpu::StringView) { unlit_check_mapped(slot, bytes, status); });
  }
}

bool register_unlit_check(std::string& error) {
  sUnlit.probes = unlit_probes();
  if (sUnlit.probes.empty()) {
    error = "demo.unlit_check has no probes on screen";
    return false;
  }
  sUnlit.bytes = (UnlitHeaderWords + sUnlit.probes.size() * UnlitSampleWords) * sizeof(u32);
  const wgpu::Device device = aurora::gfx::device();
  const auto buffer = [&](const char* label, wgpu::BufferUsage usage, uint64_t size) {
    const wgpu::BufferDescriptor descriptor{.label = label, .usage = usage, .size = size};
    return device.CreateBuffer(&descriptor);
  };
  sUnlit.output =
      buffer("demo.unlit_check output", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, sUnlit.bytes);
  for (UnlitCheckSlot& slot : sUnlit.slots) {
    slot.readback =
        buffer("demo.unlit_check readback", wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, sUnlit.bytes);
  }
  const uint64_t probeBytes = sUnlit.probes.size() * sizeof(UnlitProbe);
  sUnlit.probeBuffer =
      buffer("demo.unlit_check probes", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, probeBytes);
  device.GetQueue().WriteBuffer(sUnlit.probeBuffer, 0, sUnlit.probes.data(), probeBytes);
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.unlit_check",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  sUnlit.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = sUnlit.output, .size = sUnlit.bytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = sUnlit.probeBuffer, .size = probeBytes},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.unlit_check",
      .layout = sUnlit.layout,
      .entryCount = groupEntries.size(),
      .entries = groupEntries.data(),
  };
  sUnlit.group = device.CreateBindGroup(&groupDescriptor);
  sUnlit.id = aurora::render::register_pass({
      .name = "demo.unlit_check",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Compute,
      .declare = declare_unlit_check,
      .encode = encode_unlit_check,
      .afterSubmit = unlit_check_submitted,
      .userdata = &sUnlit,
  });
  if (sUnlit.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.unlit_check";
    return false;
  }
  return true;
}

void update_unlit_check(u32 frame) {
  if (sUnlit.id == aurora::render::InvalidPass) {
    return;
  }
  // Draws whose pipeline is still compiling are skipped, so the ground must be compiled before the first read
  if (frame >= UnlitPipelineWaitFrame && !std::exchange(sUnlit.waitedForPipelines, true)) {
    wait_for_pipelines();
  }
  const std::array<u32, 2> frames{UnlitCheckFirstFrame, UnlitCheckSecondFrame};
  for (size_t i = 0; i < frames.size(); ++i) {
    if (frame == frames[i]) {
      sUnlit.slots[i].armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
    }
  }
}

constexpr u32 OverlayProbeWords = 4;
constexpr u32 OverlayPipelineWaitFrame = 4;

constexpr std::string_view OverlayProbeWgsl = R"(
struct OverlayProbe {
  ndc: vec2f,
  pad: vec2f,
};
@group(1) @binding(0) var gx_color: texture_2d<f32>;
@group(2) @binding(0) var<storage, read_write> results: array<u32>;
@group(2) @binding(1) var<storage, read> probe: OverlayProbe;

@compute @workgroup_size(1)
fn cs_main() {
  let size = textureDimensions(gx_color);
  let centre = (probe.ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size);
  let p = clamp(vec2i(floor(centre)), vec2i(0), vec2i(size) - 1);
  let b = vec4u(round(saturate(textureLoad(gx_color, p, 0)) * 255.0));
  results[0] = b.r | (b.g << 8u) | (b.b << 16u) | (b.a << 24u);
  results[1] = size.x;
  results[2] = size.y;
  results[3] = 1u;
}
)";

struct OverlayProbePass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  u32 slotCount = 0;
  wgpu::Buffer output;
  wgpu::Buffer probeBuffer;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  // Render worker only
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  std::array<UnlitCheckSlot, 2> slots;
  std::mutex mutex;
};

// A reads between the composite and gx.overlay, where a deferred quad has not drawn yet; B reads at PRE_POST
OverlayProbePass sOverlayA;
OverlayProbePass sOverlayB;
bool sOverlayWaited = false;

void declare_overlay_probe_a(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxColor, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.after(aurora::render::phase::Composite);
  builder.before(aurora::render::phase::Overlay);
  builder.side_effect();
}

void declare_overlay_probe_b(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxColor, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_overlay_probe(PassContext& ctx, void* userdata) {
  auto& probe = *static_cast<OverlayProbePass*>(userdata);
  probe.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!probe.pipeline || probe.passLayout.Get() != ctx.pass_layout().Get()) {
    const std::string code{OverlayProbeWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.overlay_probe"};
    const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), probe.layout};
    const wgpu::PipelineLayoutDescriptor layoutDescriptor{
        .label = "demo.overlay_probe",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
    };
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "demo.overlay_probe",
        .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
        .compute = {.module = ctx.device().CreateShaderModule(&moduleDescriptor), .entryPoint = "cs_main"},
    };
    probe.pipeline = ctx.device().CreateComputePipeline(&descriptor);
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
  // Exactly the armed frame, since the frames on either side of 32 replay and would hide a missed restore
  for (u32 i = 0; i < probe.slotCount; ++i) {
    UnlitCheckSlot& slot = probe.slots[i];
    if (!slot.copied && ctx.frame() == slot.armedFrame.load(std::memory_order_acquire)) {
      ctx.encoder().CopyBufferToBuffer(probe.output, 0, slot.readback, 0, OverlayProbeWords * sizeof(u32));
      slot.copied = true;
      slot.copiedFrame = ctx.frame();
    }
  }
}

void overlay_probe_mapped(OverlayProbePass& probe, UnlitCheckSlot& slot, wgpu::MapAsyncStatus status) {
  constexpr uint64_t Bytes = OverlayProbeWords * sizeof(u32);
  if (status != wgpu::MapAsyncStatus::Success) {
    slot.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const u32*>(slot.readback.GetConstMappedRange(0, Bytes));
  if (mapped != nullptr) {
    std::lock_guard lock{probe.mutex};
    slot.values.assign(mapped, mapped + OverlayProbeWords);
  }
  slot.readback.Unmap();
  slot.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void overlay_probe_submitted(u32 frame, void* userdata) {
  auto& probe = *static_cast<OverlayProbePass*>(userdata);
  for (u32 i = 0; i < probe.slotCount; ++i) {
    UnlitCheckSlot& slot = probe.slots[i];
    if (!slot.copied || slot.mapRequested || frame != slot.copiedFrame) {
      continue;
    }
    slot.mapRequested = true;
    slot.readback.MapAsync(
        wgpu::MapMode::Read, 0, OverlayProbeWords * sizeof(u32), wgpu::CallbackMode::AllowSpontaneous,
        [&probe, &slot](wgpu::MapAsyncStatus status, wgpu::StringView) { overlay_probe_mapped(probe, slot, status); });
  }
}

bool register_overlay_probe(OverlayProbePass& probe, const char* name, aurora::render::Stage stage,
                            aurora::render::PassDeclareCallback declare, u32 slots, std::string& error) {
  const wgpu::Device device = aurora::gfx::device();
  const auto buffer = [&](const char* label, wgpu::BufferUsage usage, uint64_t size) {
    const wgpu::BufferDescriptor descriptor{.label = label, .usage = usage, .size = size};
    return device.CreateBuffer(&descriptor);
  };
  constexpr uint64_t Bytes = OverlayProbeWords * sizeof(u32);
  probe.slotCount = slots;
  probe.output = buffer(name, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, Bytes);
  for (u32 i = 0; i < slots; ++i) {
    probe.slots[i].readback = buffer(name, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, Bytes);
  }
  const std::array<f32, 4> ndc{overlay_probe()[0], overlay_probe()[1], 0.f, 0.f};
  probe.probeBuffer = buffer(name, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, sizeof(ndc));
  device.GetQueue().WriteBuffer(probe.probeBuffer, 0, ndc.data(), sizeof(ndc));
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = name,
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  probe.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = probe.output, .size = Bytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = probe.probeBuffer, .size = sizeof(ndc)},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = name,
      .layout = probe.layout,
      .entryCount = groupEntries.size(),
      .entries = groupEntries.data(),
  };
  probe.group = device.CreateBindGroup(&groupDescriptor);
  probe.id = aurora::render::register_pass({
      .name = name,
      .stage = stage,
      .kind = aurora::render::PassKind::Compute,
      .declare = declare,
      .encode = encode_overlay_probe,
      .afterSubmit = overlay_probe_submitted,
      .userdata = &probe,
  });
  if (probe.id == aurora::render::InvalidPass) {
    error = std::string{"register_pass refused "} + name;
    return false;
  }
  return true;
}

bool register_overlay_check(std::string& error) {
  return register_overlay_probe(sOverlayA, "demo.overlay_probe_a", aurora::render::Stage::OpaqueDone,
                                declare_overlay_probe_a, 1, error) &&
         register_overlay_probe(sOverlayB, "demo.overlay_probe_b", aurora::render::Stage::PrePost,
                                declare_overlay_probe_b, 2, error);
}

void update_overlay_check(u32 frame) {
  if (sOverlayA.id == aurora::render::InvalidPass) {
    return;
  }
  // Draws whose pipeline is still compiling are skipped, so the quad and the box must be compiled before the reads
  if (frame >= OverlayPipelineWaitFrame && !std::exchange(sOverlayWaited, true)) {
    wait_for_pipelines();
  }
  if (frame == OverlayCheckFrame) {
    sOverlayA.slots[0].armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
    sOverlayB.slots[0].armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
  } else if (frame == OverlayLateFrame) {
    sOverlayB.slots[1].armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
  }
}

bool overlay_pending(const OverlayProbePass& probe) {
  if (probe.id == aurora::render::InvalidPass || probe.encodes.load(std::memory_order_relaxed) == 0) {
    return false;
  }
  for (u32 i = 0; i < probe.slotCount; ++i) {
    const UnlitCheckSlot& slot = probe.slots[i];
    if (slot.armedFrame.load(std::memory_order_relaxed) != UINT32_MAX &&
        slot.result.load(std::memory_order_acquire) == ProbeResult::Pending) {
      return true;
    }
  }
  return false;
}

void release_overlay_probe(OverlayProbePass& probe) {
  probe.pipeline = {};
  probe.passLayout = {};
  probe.group = {};
  probe.layout = {};
  probe.probeBuffer = {};
  probe.output = {};
  for (UnlitCheckSlot& slot : probe.slots) {
    slot.readback = {};
  }
}

constexpr u32 CompanionHeaderWords = 4;
constexpr u32 CompanionSampleWords = 8;

constexpr std::string_view CompanionCheckWgsl = R"(
struct CompanionProbe {
  ndc: vec2f,
  quad: u32,
  cell: u32,
};
@group(1) @binding(0) var gx_normal: texture_2d<f32>;
@group(1) @binding(1) var gbuf_surface: texture_2d<u32>;
@group(2) @binding(0) var<storage, read_write> results: array<u32>;
@group(2) @binding(1) var<storage, read> probes: array<CompanionProbe>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(gx_normal);
  if (id.x == 0u) {
    results[0] = size.x;
    results[1] = size.y;
    results[2] = 1u;
  }
  if (id.x >= arrayLength(&probes)) {
    return;
  }
  let centre = (probes[id.x].ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size);
  let p = clamp(vec2i(floor(centre)), vec2i(0), vec2i(size) - 1);
  let n = bitcast<vec4u>(textureLoad(gx_normal, p, 0));
  let rec = textureLoad(gbuf_surface, min(vec2u(p), textureDimensions(gbuf_surface) - 1u), 0);
  let base = 4u + id.x * 8u;
  results[base] = n.x;
  results[base + 1u] = n.y;
  results[base + 2u] = n.z;
  results[base + 3u] = n.w;
  results[base + 4u] = rec.x;
  results[base + 5u] = rec.y;
  results[base + 6u] = rec.z;
  results[base + 7u] = rec.w;
}
)";

struct CompanionCheckPass {
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  // Main thread only
  std::vector<CompanionProbe> probes;
  uint64_t bytes = 0;
  wgpu::Buffer output;
  wgpu::Buffer probeBuffer;
  wgpu::BindGroupLayout layout;
  wgpu::BindGroup group;
  // Render worker only
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout passLayout;
  std::array<UnlitCheckSlot, CompanionPhaseCount> slots;
  std::mutex mutex;
};

CompanionCheckPass sCompanion;

void declare_companion_check(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxNormal, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.use(aurora::render::resource::SurfaceV1, Access::Sampled, aurora::render::Missing::DisablePass);
  builder.side_effect();
}

void encode_companion_check(PassContext& ctx, void* userdata) {
  auto& check = *static_cast<CompanionCheckPass*>(userdata);
  check.encodes.fetch_add(1, std::memory_order_relaxed);
  if (!check.pipeline || check.passLayout.Get() != ctx.pass_layout().Get()) {
    const std::string code{CompanionCheckWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.companion_probe"};
    const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), check.layout};
    const wgpu::PipelineLayoutDescriptor layoutDescriptor{
        .label = "demo.companion_probe",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
    };
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "demo.companion_probe",
        .layout = ctx.device().CreatePipelineLayout(&layoutDescriptor),
        .compute = {.module = ctx.device().CreateShaderModule(&moduleDescriptor), .entryPoint = "cs_main"},
    };
    check.pipeline = ctx.device().CreateComputePipeline(&descriptor);
    check.passLayout = ctx.pass_layout();
  }
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetPipeline(check.pipeline);
  compute.SetBindGroup(2, check.group);
  compute.DispatchWorkgroups((static_cast<u32>(check.probes.size()) + 63) / 64);
  compute.End();
  for (UnlitCheckSlot& slot : check.slots) {
    if (!slot.copied && ctx.frame() >= slot.armedFrame.load(std::memory_order_acquire)) {
      ctx.encoder().CopyBufferToBuffer(check.output, 0, slot.readback, 0, check.bytes);
      slot.copied = true;
      slot.copiedFrame = ctx.frame();
      break;
    }
  }
}

void companion_check_mapped(UnlitCheckSlot& slot, uint64_t bytes, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    slot.result.store(ProbeResult::Failed, std::memory_order_release);
    return;
  }
  const auto* mapped = static_cast<const u32*>(slot.readback.GetConstMappedRange(0, bytes));
  if (mapped != nullptr) {
    std::lock_guard lock{sCompanion.mutex};
    slot.values.assign(mapped, mapped + bytes / sizeof(u32));
  }
  slot.readback.Unmap();
  slot.result.store(mapped != nullptr ? ProbeResult::Ready : ProbeResult::Failed, std::memory_order_release);
}

void companion_check_submitted(u32 frame, void* userdata) {
  auto& check = *static_cast<CompanionCheckPass*>(userdata);
  for (UnlitCheckSlot& slot : check.slots) {
    if (!slot.copied || slot.mapRequested || frame != slot.copiedFrame) {
      continue;
    }
    slot.mapRequested = true;
    const uint64_t bytes = check.bytes;
    slot.readback.MapAsync(
        wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::AllowSpontaneous,
        [&slot, bytes](wgpu::MapAsyncStatus status, wgpu::StringView) { companion_check_mapped(slot, bytes, status); });
  }
}

bool register_companion_check(std::string& error) {
  sCompanion.probes = companion_probes();
  if (sCompanion.probes.empty()) {
    error = "demo.companion_probe has no probes on screen";
    return false;
  }
  sCompanion.bytes = (CompanionHeaderWords + sCompanion.probes.size() * CompanionSampleWords) * sizeof(u32);
  const wgpu::Device device = aurora::gfx::device();
  const auto buffer = [&](const char* label, wgpu::BufferUsage usage, uint64_t size) {
    const wgpu::BufferDescriptor descriptor{.label = label, .usage = usage, .size = size};
    return device.CreateBuffer(&descriptor);
  };
  sCompanion.output =
      buffer("demo.companion_probe output", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, sCompanion.bytes);
  for (UnlitCheckSlot& slot : sCompanion.slots) {
    slot.readback = buffer("demo.companion_probe readback", wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
                           sCompanion.bytes);
  }
  const uint64_t probeBytes = sCompanion.probes.size() * sizeof(CompanionProbe);
  sCompanion.probeBuffer =
      buffer("demo.companion_probe probes", wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, probeBytes);
  device.GetQueue().WriteBuffer(sCompanion.probeBuffer, 0, sCompanion.probes.data(), probeBytes);
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.companion_probe",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  sCompanion.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = sCompanion.output, .size = sCompanion.bytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = sCompanion.probeBuffer, .size = probeBytes},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.companion_probe",
      .layout = sCompanion.layout,
      .entryCount = groupEntries.size(),
      .entries = groupEntries.data(),
  };
  sCompanion.group = device.CreateBindGroup(&groupDescriptor);
  sCompanion.id = aurora::render::register_pass({
      .name = "demo.companion_probe",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Compute,
      .declare = declare_companion_check,
      .encode = encode_companion_check,
      .afterSubmit = companion_check_submitted,
      .userdata = &sCompanion,
  });
  if (sCompanion.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.companion_probe";
    return false;
  }
  return true;
}

void update_companion_check(u32 frame) {
  if (sCompanion.id == aurora::render::InvalidPass) {
    return;
  }
  // Draws whose pipeline is still compiling are skipped, and each phase brings new companion pipelines
  if (frame == CompanionPipelineWaitFrame || frame == CompanionMappedWaitFrame || frame == CompanionMatWaitFrame) {
    wait_for_pipelines();
  }
  const std::array<u32, CompanionPhaseCount> frames{CompanionBareFrame, CompanionMappedFrame,
                                                    CompanionBlendOneProbeFrame, CompanionMatProbeFrame};
  for (size_t i = 0; i < frames.size(); ++i) {
    if (frame == frames[i]) {
      sCompanion.slots[i].armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
    }
  }
}

void update_fwd_env_check(u32 frame) {
  if (sFwdEnv.id == aurora::render::InvalidPass || frame < FwdEnvCheckFrame ||
      sFwdEnv.armedFrame.load(std::memory_order_relaxed) != UINT32_MAX) {
    return;
  }
  if (!std::exchange(sFwdEnv.waitedForPipelines, true)) {
    wait_for_pipelines();
    sFwdEnv.deadline = std::chrono::steady_clock::now() + PipelineWait;
  }
  // AO compiles its own pipelines, so the strips are judged only once its history has had frames to settle
  const u32 encodes = sFwdEnv.encodes.load(std::memory_order_relaxed);
  const bool late = std::chrono::steady_clock::now() >= sFwdEnv.deadline;
  if (!pipelines_compiling() && (encodes >= FwdEnvSettleFrames || (late && encodes != 0))) {
    sFwdEnv.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
  } else if (!late) {
    std::this_thread::sleep_for(AoCompileWait);
  }
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
  if (options.fogCheck && !register_fog_check(error)) {
    return false;
  }
  if (options.aoCheck && !register_ao_check(error)) {
    return false;
  }
  if (options.unlitCheck && !register_unlit_check(error)) {
    return false;
  }
  if (options.companionCheck && !register_companion_check(error)) {
    return false;
  }
  if (options.fwdEnvCheck && !register_fwd_env_check(error)) {
    return false;
  }
  if (options.overlayCheck && !register_overlay_check(error)) {
    return false;
  }
  return options.gpuCull == 0 ||
         register_cull_passes(
             {.instances = options.gpuCull, .check = options.cullCheck, .timings = options.cullTimings}, error);
}

void update_graph_passes(u32 frame) {
  update_cull_passes(frame);
  update_fwd_env_check(frame);
  update_unlit_check(frame);
  update_companion_check(frame);
  update_overlay_check(frame);
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
  if (sFog.id != aurora::render::InvalidPass) {
    aurora::render::set_pass_params(sFog.id, &sFog.params, sizeof(sFog.params));
    if (frame >= FogCheckFrame && sFog.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX) {
      if (!std::exchange(sFog.waitedForPipelines, true)) {
        wait_for_pipelines();
      }
      if (!pipelines_compiling()) {
        sFog.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
      }
    }
  }
  if (sAo.id != aurora::render::InvalidPass) {
    update_ao_check(frame);
    if (frame >= AoCheckFrame && sAo.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX) {
      if (!std::exchange(sAo.waitedForPipelines, true)) {
        wait_for_pipelines();
        sAo.deadline = std::chrono::steady_clock::now() + PipelineWait;
      }
      if (!pipelines_compiling() && sAo.encodes.load(std::memory_order_relaxed) != 0) {
        sAo.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
      } else if (std::chrono::steady_clock::now() < sAo.deadline) {
        std::this_thread::sleep_for(AoCompileWait);
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
  const bool fog = waiting(sFog.id, sFog.encodes, sFog.armedFrame);
  const bool fwdEnv = waiting(sFwdEnv.id, sFwdEnv.encodes, sFwdEnv.armedFrame);
  const auto unlitPending = [] {
    return sUnlit.id != aurora::render::InvalidPass && sUnlit.encodes.load(std::memory_order_relaxed) != 0 &&
           std::ranges::any_of(sUnlit.slots, [](const UnlitCheckSlot& slot) {
             return slot.armedFrame.load(std::memory_order_relaxed) != UINT32_MAX &&
                    slot.result.load(std::memory_order_acquire) == ProbeResult::Pending;
           });
  };
  const auto companionPending = [] {
    return sCompanion.id != aurora::render::InvalidPass && sCompanion.encodes.load(std::memory_order_relaxed) != 0 &&
           std::ranges::any_of(sCompanion.slots, [](const UnlitCheckSlot& slot) {
             return slot.armedFrame.load(std::memory_order_relaxed) != UINT32_MAX &&
                    slot.result.load(std::memory_order_acquire) == ProbeResult::Pending;
           });
  };
  const auto aoPending = [] {
    return sAo.statsState.load(std::memory_order_acquire) != AoSlotState::Free ||
           std::ranges::any_of(sAo.slots, [](const AoCheckSlot& slot) {
             return slot.state.load(std::memory_order_acquire) != AoSlotState::Free;
           });
  };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (((depth && sProbe.result.load(std::memory_order_acquire) == ProbeResult::Pending) ||
          (motion && sMotion.result.load(std::memory_order_acquire) == ProbeResult::Pending) ||
          (fog && sFog.result.load(std::memory_order_acquire) == ProbeResult::Pending) ||
          (fwdEnv && sFwdEnv.result.load(std::memory_order_acquire) == ProbeResult::Pending) || unlitPending() ||
          companionPending() || (sAo.id != aurora::render::InvalidPass && aoPending()) || overlay_pending(sOverlayA) ||
          overlay_pending(sOverlayB)) &&
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
      .fogEncodes = sFog.encodes.load(std::memory_order_relaxed),
      .fogReady = sFog.result.load(std::memory_order_acquire) == ProbeResult::Ready,
  };
  if (report.probeReady) {
    std::lock_guard lock{sProbe.mutex};
    report.probe = sProbe.values;
  }
  if (report.motionReady) {
    std::lock_guard lock{sMotion.mutex};
    report.motion = sMotion.values;
  }
  if (report.fogReady) {
    std::lock_guard lock{sFog.mutex};
    report.fog = sFog.values;
  }
  if (sFwdEnv.id != aurora::render::InvalidPass) {
    report.fwdEnv.encodes = sFwdEnv.encodes.load(std::memory_order_relaxed);
    report.fwdEnv.ready = sFwdEnv.result.load(std::memory_order_acquire) == ProbeResult::Ready;
    std::lock_guard lock{sFwdEnv.mutex};
    const std::vector<u32>& words = sFwdEnv.values;
    if (report.fwdEnv.ready && words.size() >= FwdEnvHeaderWords && words[2] == 1) {
      report.fwdEnv.width = words[0];
      report.fwdEnv.height = words[1];
      for (size_t i = 0; i < sFwdEnv.probes.size(); ++i) {
        const size_t base = FwdEnvHeaderWords + i * FwdEnvSampleWords;
        if (base + FwdEnvSampleWords > words.size()) {
          break;
        }
        report.fwdEnv.samples.push_back({.cls = sFwdEnv.probes[i].cls,
                                         .color = words[base],
                                         .fog = words[base + 1],
                                         .ao = words[base + 2],
                                         .normal = words[base + 3],
                                         .depthBits = words[base + 4]});
      }
    }
  }
  if (sUnlit.id != aurora::render::InvalidPass) {
    report.unlit.encodes = sUnlit.encodes.load(std::memory_order_relaxed);
    report.unlit.ready = std::ranges::all_of(sUnlit.slots, [](const UnlitCheckSlot& slot) {
      return slot.result.load(std::memory_order_acquire) == ProbeResult::Ready;
    });
    std::lock_guard lock{sUnlit.mutex};
    for (size_t s = 0; s < sUnlit.slots.size() && report.unlit.ready; ++s) {
      const std::vector<u32>& words = sUnlit.slots[s].values;
      if (words.size() < UnlitHeaderWords || words[2] != 1) {
        report.unlit.ready = false;
        break;
      }
      report.unlit.width = words[0];
      report.unlit.height = words[1];
      std::vector<UnlitSample>& samples = s == 0 ? report.unlit.before : report.unlit.after;
      for (size_t i = 0; i < sUnlit.probes.size(); ++i) {
        const size_t base = UnlitHeaderWords + i * UnlitSampleWords;
        if (base + UnlitSampleWords > words.size()) {
          break;
        }
        samples.push_back({.cls = sUnlit.probes[i].cls,
                           .color = words[base],
                           .surface = {words[base + 1], words[base + 2], words[base + 3], words[base + 4]}});
      }
    }
  }
  if (sCompanion.id != aurora::render::InvalidPass) {
    report.companion.encodes = sCompanion.encodes.load(std::memory_order_relaxed);
    report.companion.ready = std::ranges::all_of(sCompanion.slots, [](const UnlitCheckSlot& slot) {
      return slot.result.load(std::memory_order_acquire) == ProbeResult::Ready;
    });
    std::lock_guard lock{sCompanion.mutex};
    for (size_t s = 0; s < sCompanion.slots.size() && report.companion.ready; ++s) {
      const std::vector<u32>& words = sCompanion.slots[s].values;
      if (words.size() < CompanionHeaderWords || words[2] != 1) {
        report.companion.ready = false;
        break;
      }
      report.companion.width = words[0];
      report.companion.height = words[1];
      for (size_t i = 0; i < sCompanion.probes.size(); ++i) {
        const size_t base = CompanionHeaderWords + i * CompanionSampleWords;
        if (base + CompanionSampleWords > words.size()) {
          break;
        }
        CompanionSample sample{.quad = sCompanion.probes[i].quad, .cell = sCompanion.probes[i].cell};
        for (size_t c = 0; c < 4; ++c) {
          sample.normal[c] = std::bit_cast<f32>(words[base + c]);
          sample.surface[c] = words[base + 4 + c];
        }
        report.companion.phases[s].push_back(sample);
      }
    }
  }
  if (sOverlayA.id != aurora::render::InvalidPass && sOverlayB.id != aurora::render::InvalidPass) {
    auto& overlay = report.overlay;
    overlay.encodesA = sOverlayA.encodes.load(std::memory_order_relaxed);
    overlay.encodesB = sOverlayB.encodes.load(std::memory_order_relaxed);
    overlay.ready = true;
    const std::array<std::pair<OverlayProbePass*, u32>, OverlaySampleCount> samples{
        {{&sOverlayA, 0}, {&sOverlayB, 0}, {&sOverlayB, 1}}};
    for (u32 i = 0; i < OverlaySampleCount; ++i) {
      auto& [probe, slot] = samples[i];
      std::lock_guard lock{probe->mutex};
      const std::vector<u32>& words = probe->slots[slot].values;
      if (probe->slots[slot].result.load(std::memory_order_acquire) != ProbeResult::Ready ||
          words.size() < OverlayProbeWords || words[3] != 1) {
        overlay.ready = false;
        continue;
      }
      overlay.colors[i] = words[0];
      overlay.width = words[1];
      overlay.height = words[2];
    }
  }
  if (sAo.id != aurora::render::InvalidPass) {
    std::lock_guard lock{sAo.mutex};
    report.ao = {
        .encodes = sAo.encodes.load(std::memory_order_relaxed),
        .frames = sAo.frames,
        .failedMaps = sAo.failedMaps,
        .receivers = sAo.receivers,
        .pixels = sAo.pixels,
        .probes = sAo.probes,
        .sums = sAo.sums,
        .receiverFrames = sAo.receiverFrames,
        .temporal = sAo.temporal,
    };
  }
  return report;
}

void unregister_graph_passes() {
  const bool cull = unregister_cull_passes();
  if (!cull && sVignette.id == aurora::render::InvalidPass && sProbe.id == aurora::render::InvalidPass &&
      sMotion.id == aurora::render::InvalidPass && sMotionView.id == aurora::render::InvalidPass &&
      sFog.id == aurora::render::InvalidPass && sAo.id == aurora::render::InvalidPass &&
      sFwdEnv.id == aurora::render::InvalidPass && sUnlit.id == aurora::render::InvalidPass &&
      sCompanion.id == aurora::render::InvalidPass && sOverlayA.id == aurora::render::InvalidPass &&
      sOverlayB.id == aurora::render::InvalidPass) {
    return;
  }
  for (PassId* id : {&sVignette.id, &sProbe.id, &sMotion.id, &sMotionView.id, &sFog.id, &sAo.id, &sFwdEnv.id,
                     &sUnlit.id, &sCompanion.id, &sOverlayA.id, &sOverlayB.id}) {
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
  sFog.pipeline = {};
  sFog.passLayout = {};
  sFog.group = {};
  sFog.layout = {};
  sFog.readback = {};
  sFog.output = {};
  sFog.waitedForPipelines = false;
  sAo.pipeline = {};
  sAo.passLayout = {};
  sAo.group = {};
  sAo.layout = {};
  sAo.probeBuffer = {};
  sAo.output = {};
  sAo.stats = {};
  sAo.statsReadback = {};
  for (AoCheckSlot& slot : sAo.slots) {
    slot.readback = {};
  }
  sAo.waitedForPipelines = false;
  sFwdEnv.pipeline = {};
  sFwdEnv.passLayout = {};
  sFwdEnv.group = {};
  sFwdEnv.layout = {};
  sFwdEnv.probeBuffer = {};
  sFwdEnv.readback = {};
  sFwdEnv.output = {};
  sFwdEnv.waitedForPipelines = false;
  sUnlit.pipeline = {};
  sUnlit.passLayout = {};
  sUnlit.group = {};
  sUnlit.layout = {};
  sUnlit.probeBuffer = {};
  sUnlit.output = {};
  for (UnlitCheckSlot& slot : sUnlit.slots) {
    slot.readback = {};
  }
  sUnlit.waitedForPipelines = false;
  sCompanion.pipeline = {};
  sCompanion.passLayout = {};
  sCompanion.group = {};
  sCompanion.layout = {};
  sCompanion.probeBuffer = {};
  sCompanion.output = {};
  for (UnlitCheckSlot& slot : sCompanion.slots) {
    slot.readback = {};
  }
  release_overlay_probe(sOverlayA);
  release_overlay_probe(sOverlayB);
  sOverlayWaited = false;
}

} // namespace render_demo

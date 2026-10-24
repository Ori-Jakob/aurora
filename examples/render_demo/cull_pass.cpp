#include "cull_pass.hpp"

#include "cull_reference.hpp"
#include "scene.hpp"

#include <aurora/gfx.hpp>
#include <aurora/render.hpp>
#include <aurora/render_caps.hpp>
#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace render_demo {
namespace {
using aurora::render::Access;
using aurora::render::BufRef;
using aurora::render::PassBuilder;
using aurora::render::PassContext;
using aurora::render::PassId;
using aurora::render::TexRef;

constexpr std::string_view CullPassName = "demo.cull";
constexpr std::string_view DrawPassName = "demo.cull_draw";
constexpr std::string_view OutName = "demo.cull.out";
constexpr BufRef CullOut{0};
constexpr TexRef DrawColor{0};
constexpr TexRef DrawDepth{1};
constexpr BufRef DrawIds{2};
constexpr u32 TimingStartFrame = 60;
constexpr std::array<f32, 18> OctahedronVertices{1.f, 0.f,  0.f, -1.f, 0.f, 0.f, 0.f, 1.f, 0.f,
                                                 0.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f, 0.f, -1.f};
constexpr std::array<u32, CullIndexCount> OctahedronIndices{0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4,
                                                            2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
constexpr std::array<const char*, 3> CullEntryPoints{"cs_count", "cs_scan", "cs_scatter"};

enum class Readback : u32 { Pending, Ready, Failed };

struct CullState {
  CullPassOptions options;
  CullLayout layout;
  std::string clamp;
  u32 requested = 0;
  bool registered = false;
  PassId cullId = aurora::render::InvalidPass;
  PassId drawId = aurora::render::InvalidPass;
  std::atomic_uint32_t cullEncodes = 0;
  std::atomic_uint32_t drawEncodes = 0;
  std::atomic_uint32_t armedFrame = UINT32_MAX;
  // Main thread only
  std::vector<CullInstance> instances;
  wgpu::Buffer instanceBuffer;
  wgpu::Buffer mesh;
  wgpu::Buffer indices;
  wgpu::Buffer readback;
  wgpu::BindGroupLayout groupLayout;
  wgpu::BindGroup group;
  std::vector<f32> cullSamples;
  std::vector<f32> drawSamples;
  std::array<f32, 2> lastSample{-1.f, -1.f};
  // Render worker only
  std::array<wgpu::ComputePipeline, 3> cullPipelines;
  wgpu::BindGroupLayout cullPassLayout;
  wgpu::RenderPipeline drawPipeline;
  wgpu::BindGroupLayout drawPassLayout;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined;
  u32 samples = 0;
  bool copied = false;
  u32 copiedFrame = 0;
  bool mapRequested = false;
  // Written by the map callback, which may run on any thread
  std::atomic<Readback> result = Readback::Pending;
  std::mutex mutex;
  std::vector<u32> words;
};

CullState sCull;

wgpu::Buffer create_buffer(const wgpu::Device& device, const char* label, wgpu::BufferUsage usage, u64 size,
                           const void* data) {
  const wgpu::BufferDescriptor descriptor{
      .label = label,
      .usage = usage,
      .size = size,
      .mappedAtCreation = data != nullptr,
  };
  wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
  if (data != nullptr) {
    std::memcpy(buffer.GetMappedRange(0, size), data, size);
    buffer.Unmap();
  }
  return buffer;
}

wgpu::ShaderModule create_module(const wgpu::Device& device, const std::string& body, const char* label) {
  const std::string code = std::string{aurora::render::FrameWgsl} + body;
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl, .label = label};
  return device.CreateShaderModule(&descriptor);
}

wgpu::PipelineLayout create_layout(const PassContext& ctx, const wgpu::BindGroupLayout& own, const char* label) {
  const std::array<wgpu::BindGroupLayout, 3> layouts{ctx.frame_layout(), ctx.pass_layout(), own};
  const wgpu::PipelineLayoutDescriptor descriptor{
      .label = label,
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  return ctx.device().CreatePipelineLayout(&descriptor);
}

void create_cull_pipelines(const PassContext& ctx, CullState& s) {
  const wgpu::ShaderModule module = create_module(ctx.device(), cull_compute_wgsl(s.layout), "demo.cull");
  const wgpu::PipelineLayout layout = create_layout(ctx, s.groupLayout, "demo.cull");
  for (size_t i = 0; i < CullEntryPoints.size(); ++i) {
    const wgpu::ComputePipelineDescriptor descriptor{
        .label = "demo.cull",
        .layout = layout,
        .compute = {.module = module, .entryPoint = CullEntryPoints[i]},
    };
    s.cullPipelines[i] = ctx.device().CreateComputePipeline(&descriptor);
  }
  s.cullPassLayout = ctx.pass_layout();
}

wgpu::RenderPipeline create_draw_pipeline(const PassContext& ctx, const CullState& s,
                                          const wgpu::ColorTargetState& target, u32 samples,
                                          wgpu::TextureFormat depthFormat) {
  const wgpu::ShaderModule module = create_module(ctx.device(), cull_draw_wgsl(s.layout.instances), "demo.cull_draw");
  const std::array<wgpu::VertexAttribute, 2> attributes{{
      {.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0},
      {.format = wgpu::VertexFormat::Uint32, .offset = 0, .shaderLocation = 1},
  }};
  const std::array<wgpu::VertexBufferLayout, 2> buffers{{
      {.stepMode = wgpu::VertexStepMode::Vertex,
       .arrayStride = 3 * sizeof(f32),
       .attributeCount = 1,
       .attributes = &attributes[0]},
      {.stepMode = wgpu::VertexStepMode::Instance,
       .arrayStride = sizeof(u32),
       .attributeCount = 1,
       .attributes = &attributes[1]},
  }};
  const wgpu::DepthStencilState depth{
      .format = depthFormat,
      .depthWriteEnabled = wgpu::OptionalBool::False,
      .depthCompare =
          aurora::gfx::uses_reversed_z() ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual,
  };
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "demo.cull_draw",
      .layout = create_layout(ctx, s.groupLayout, "demo.cull_draw"),
      .vertex = {.module = module, .entryPoint = "vs_main", .bufferCount = buffers.size(), .buffers = buffers.data()},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depth,
      .multisample = {.count = samples},
      .fragment = &fragment,
  };
  return ctx.device().CreateRenderPipeline(&descriptor);
}

void declare_cull(PassBuilder& builder, void* userdata) {
  const auto& s = *static_cast<const CullState*>(userdata);
  builder.create_buffer(
      OutName, {.size = s.layout.bytes(), .extraUsage = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopySrc},
      Access::StorageWrite);
}

void declare_draw(PassBuilder& builder, void*) {
  builder.use(aurora::render::resource::GxColor, Access::ColorRMW);
  builder.use(aurora::render::resource::GxDepth, Access::DepthRead);
  builder.use_buffer(OutName, Access::Vertex);
}

void encode_cull(PassContext& ctx, void* userdata) {
  auto& s = *static_cast<CullState*>(userdata);
  if (!s.cullPipelines[0] || s.cullPassLayout.Get() != ctx.pass_layout().Get()) {
    create_cull_pipelines(ctx, s);
  }
  const aurora::render::BufferView out = ctx.buffer(CullOut);
  const wgpu::ComputePassEncoder compute = ctx.begin_compute();
  if (!compute) {
    return;
  }
  compute.SetBindGroup(2, s.group);
  compute.SetPipeline(s.cullPipelines[0]);
  compute.DispatchWorkgroups(s.layout.groups);
  compute.SetPipeline(s.cullPipelines[1]);
  compute.DispatchWorkgroups(1);
  compute.SetPipeline(s.cullPipelines[2]);
  compute.DispatchWorkgroups(s.layout.groups);
  compute.End();
  if (s.readback && !s.copied && ctx.frame() >= s.armedFrame.load(std::memory_order_acquire)) {
    ctx.encoder().CopyBufferToBuffer(out.buffer, out.offset, s.readback, 0, s.layout.bytes());
    s.copied = true;
    s.copiedFrame = ctx.frame();
  }
  s.cullEncodes.fetch_add(1, std::memory_order_relaxed);
}

void encode_draw(PassContext& ctx, void* userdata) {
  auto& s = *static_cast<CullState*>(userdata);
  const wgpu::ColorTargetState target = ctx.color_target(DrawColor);
  const u32 samples = ctx.sample_count(DrawColor);
  const wgpu::TextureFormat depthFormat = ctx.format(DrawDepth);
  if (!s.drawPipeline || s.colorFormat != target.format || s.samples != samples || s.depthFormat != depthFormat ||
      s.drawPassLayout.Get() != ctx.pass_layout().Get()) {
    s.drawPipeline = create_draw_pipeline(ctx, s, target, samples, depthFormat);
    s.drawPassLayout = ctx.pass_layout();
    s.colorFormat = target.format;
    s.samples = samples;
    s.depthFormat = depthFormat;
  }
  const aurora::render::BufferView ids = ctx.buffer(DrawIds);
  const wgpu::RenderPassEncoder render = ctx.begin_render();
  if (!render) {
    return;
  }
  render.SetPipeline(s.drawPipeline);
  render.SetBindGroup(2, s.group);
  render.SetVertexBuffer(0, s.mesh);
  render.SetVertexBuffer(1, ids.buffer, ids.offset + s.layout.ids_offset(),
                         static_cast<u64>(s.layout.instances) * sizeof(u32));
  render.SetIndexBuffer(s.indices, wgpu::IndexFormat::Uint32);
  render.DrawIndexedIndirect(ids.buffer, ids.offset);
  render.End();
  s.drawEncodes.fetch_add(1, std::memory_order_relaxed);
}

void cull_mapped(CullState& s, wgpu::MapAsyncStatus status) {
  if (status != wgpu::MapAsyncStatus::Success) {
    s.result.store(Readback::Failed, std::memory_order_release);
    return;
  }
  const u64 bytes = s.layout.bytes();
  const auto* mapped = static_cast<const u32*>(s.readback.GetConstMappedRange(0, bytes));
  if (mapped != nullptr) {
    std::lock_guard lock{s.mutex};
    s.words.assign(mapped, mapped + bytes / sizeof(u32));
  }
  s.readback.Unmap();
  s.result.store(mapped != nullptr ? Readback::Ready : Readback::Failed, std::memory_order_release);
}

void cull_submitted(u32 frame, void* userdata) {
  auto& s = *static_cast<CullState*>(userdata);
  if (!s.copied || s.mapRequested || frame != s.copiedFrame) {
    return;
  }
  s.mapRequested = true;
  s.readback.MapAsync(wgpu::MapMode::Read, 0, s.layout.bytes(), wgpu::CallbackMode::AllowSpontaneous,
                      [&s](wgpu::MapAsyncStatus status, wgpu::StringView) { cull_mapped(s, status); });
}

std::optional<std::string> missing_capability(const aurora::render::Caps& caps) {
  using aurora::render::ShaderStage;
  if (aurora::render::stage_binding_limits(caps, ShaderStage::Compute).storageBuffers < 2) {
    return std::string{"compute storage buffers"};
  }
  if (aurora::render::stage_binding_limits(caps, ShaderStage::Vertex).storageBuffers < 1) {
    return std::string{"vertex storage buffers"};
  }
  if (caps.limits.maxComputeInvocationsPerWorkgroup < CullWorkgroupSize ||
      caps.limits.maxComputeWorkgroupSizeX < CullWorkgroupSize) {
    return std::string{"compute workgroup size"};
  }
  return std::nullopt;
}

bool fits(u32 instances, const aurora::render::DeviceLimits& limits) noexcept {
  const u64 maxBinding = std::min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const CullLayout layout = cull_layout(instances);
  return static_cast<u64>(instances) * sizeof(CullInstance) <= maxBinding && layout.bytes() <= maxBinding &&
         layout.groups <= limits.maxComputeWorkgroupsPerDimension;
}

f32 percentile(std::vector<f32> samples, f32 q) {
  std::ranges::sort(samples);
  const f32 position = q * static_cast<f32>(samples.size() - 1);
  const auto lo = static_cast<size_t>(position);
  const size_t hi = std::min(lo + 1, samples.size() - 1);
  return samples[lo] + (samples[hi] - samples[lo]) * (position - static_cast<f32>(lo));
}

std::array<Plane, 6> camera_planes(u32 frame) {
  const SceneCamera camera = scene_camera(frame);
  return frustum_planes(view_projection(camera.projection, camera.view, aurora::gfx::uses_reversed_z()));
}
} // namespace

bool register_cull_passes(const CullPassOptions& options, std::string& error) {
  auto& s = sCull;
  s.options = options;
  s.requested = options.instances;
  const aurora::render::Caps& caps = aurora::render::caps();
  if (const auto missing = missing_capability(caps)) {
    s.clamp = "unavailable (" + *missing + ")";
    return true;
  }
  u32 instances = std::min(options.instances, MaxCullInstances);
  if (!fits(instances, caps.limits)) {
    instances -= instances % CullWorkgroupSize;
    while (instances != 0 && !fits(instances, caps.limits)) {
      instances -= CullWorkgroupSize;
    }
    if (instances == 0) {
      s.clamp = "unavailable (storage buffer size)";
      return true;
    }
    s.clamp = "storage buffer size";
  }
  s.layout = cull_layout(instances);
  s.instances.resize(instances);
  fill_cull_instances(s.instances);
  const wgpu::Device device = aurora::gfx::device();
  s.instanceBuffer = create_buffer(device, "demo.cull instances", wgpu::BufferUsage::Storage,
                                   static_cast<u64>(instances) * sizeof(CullInstance), s.instances.data());
  s.mesh = create_buffer(device, "demo.cull mesh", wgpu::BufferUsage::Vertex, sizeof(OctahedronVertices),
                         OctahedronVertices.data());
  s.indices = create_buffer(device, "demo.cull indices", wgpu::BufferUsage::Index, sizeof(OctahedronIndices),
                            OctahedronIndices.data());
  if (options.check) {
    s.readback = create_buffer(device, "demo.cull readback", wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
                               s.layout.bytes(), nullptr);
  }
  const wgpu::BindGroupLayoutEntry layoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute | wgpu::ShaderStage::Vertex,
      .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "demo.cull instances",
      .entryCount = 1,
      .entries = &layoutEntry,
  };
  s.groupLayout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::BindGroupEntry groupEntry{
      .binding = 0,
      .buffer = s.instanceBuffer,
      .size = static_cast<u64>(instances) * sizeof(CullInstance),
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "demo.cull instances",
      .layout = s.groupLayout,
      .entryCount = 1,
      .entries = &groupEntry,
  };
  s.group = device.CreateBindGroup(&groupDescriptor);
  s.cullId = aurora::render::register_pass({
      .name = "demo.cull",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Compute,
      .paramsSize = sizeof(CullParams),
      .declare = declare_cull,
      .encode = encode_cull,
      .afterSubmit = options.check ? cull_submitted : nullptr,
      .userdata = &s,
  });
  if (s.cullId == aurora::render::InvalidPass) {
    error = "register_pass refused demo.cull";
    return false;
  }
  s.drawId = aurora::render::register_pass({
      .name = "demo.cull_draw",
      .stage = aurora::render::Stage::PrePost,
      .kind = aurora::render::PassKind::Render,
      .paramsSize = sizeof(CullDrawParams),
      .declare = declare_draw,
      .encode = encode_draw,
      .userdata = &s,
  });
  if (s.drawId == aurora::render::InvalidPass) {
    error = "register_pass refused demo.cull_draw";
    return false;
  }
  s.registered = true;
  return true;
}

void update_cull_passes(u32 frame) {
  auto& s = sCull;
  if (!s.registered) {
    return;
  }
  if (s.options.timings && frame >= TimingStartFrame) {
    std::array<std::optional<f32>, 2> sample;
    for (const aurora::render::PassTiming& timing : aurora::render::pass_timings()) {
      if (timing.name == CullPassName) {
        sample[0] = timing.gpuMs;
      } else if (timing.name == DrawPassName) {
        sample[1] = timing.gpuMs;
      }
    }
    // The span repeats the last profiled frame until a newer one resolves
    if (sample[0] && sample[1] && (*sample[0] != s.lastSample[0] || *sample[1] != s.lastSample[1])) {
      s.lastSample = {*sample[0], *sample[1]};
      s.cullSamples.push_back(*sample[0]);
      s.drawSamples.push_back(*sample[1]);
    }
  }
  const SceneCamera camera = scene_camera(frame);
  const std::array<f32, 16> viewProj = view_projection(camera.projection, camera.view, aurora::gfx::uses_reversed_z());
  const CullParams params{.planes = frustum_planes(viewProj), .frameTag = frame, .pad = {}};
  const CullDrawParams drawParams{.viewProj = viewProj};
  aurora::render::set_pass_params(s.cullId, &params, sizeof(params));
  aurora::render::set_pass_params(s.drawId, &drawParams, sizeof(drawParams));
  if (s.options.check && frame == CullCheckFrame && s.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX) {
    s.armedFrame.store(aurora::gfx::current_frame(), std::memory_order_release);
  }
}

void finish_cull_passes() {
  auto& s = sCull;
  if (!s.registered || !s.readback || s.armedFrame.load(std::memory_order_relaxed) == UINT32_MAX ||
      s.cullEncodes.load(std::memory_order_relaxed) == 0) {
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (s.result.load(std::memory_order_acquire) == Readback::Pending && std::chrono::steady_clock::now() < deadline) {
    aurora::gfx::synchronize();
    aurora::gfx::device().Tick();
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
}

CullReport cull_report() {
  auto& s = sCull;
  CullReport report{
      .registered = s.registered,
      .clamp = s.clamp,
      .requested = s.requested,
      .instances = s.layout.instances,
      .cullEncodes = s.cullEncodes.load(std::memory_order_relaxed),
      .drawEncodes = s.drawEncodes.load(std::memory_order_relaxed),
      .timingSamples = static_cast<u32>(s.cullSamples.size()),
  };
  if (!s.cullSamples.empty()) {
    report.cullMs = percentile(s.cullSamples, 0.5f);
    report.drawMs = percentile(s.drawSamples, 0.5f);
    report.cullP90Ms = percentile(s.cullSamples, 0.9f);
    report.drawP90Ms = percentile(s.drawSamples, 0.9f);
  }
  if (!s.options.check || !s.registered) {
    return report;
  }
  report.checkFrame = CullCheckFrame;
  if (aurora::render::caps().backend == BACKEND_NULL) {
    report.check = CullCheck::SkippedNull;
    return report;
  }
  if (s.result.load(std::memory_order_acquire) != Readback::Ready) {
    report.check = CullCheck::NotReady;
    report.failure = s.copied ? "readback did not complete" : "the armed frame never copied";
    return report;
  }
  const u32 armed = s.armedFrame.load(std::memory_order_relaxed);
  if (s.copiedFrame != armed) {
    report.check = CullCheck::Failed;
    report.failure = "copied at gfx frame " + std::to_string(s.copiedFrame) + ", armed " + std::to_string(armed);
    return report;
  }
  const std::array<Plane, 6> planes = camera_planes(CullCheckFrame);
  std::lock_guard lock{s.mutex};
  const CullCheckResult result = check_cull_readback(s.words, s.layout, s.instances, planes, CullCheckFrame);
  report.visible = result.visible;
  report.ambiguous = result.ambiguous;
  report.check = result.failure ? CullCheck::Failed : CullCheck::Passed;
  report.failure = result.failure.value_or(std::string{});
  return report;
}

bool unregister_cull_passes() {
  auto& s = sCull;
  for (PassId* id : {&s.cullId, &s.drawId}) {
    if (*id != aurora::render::InvalidPass) {
      aurora::render::unregister_pass(*id);
      *id = aurora::render::InvalidPass;
    }
  }
  return static_cast<bool>(s.instanceBuffer);
}

void release_cull_passes() {
  auto& s = sCull;
  s.cullPipelines = {};
  s.cullPassLayout = {};
  s.drawPipeline = {};
  s.drawPassLayout = {};
  s.group = {};
  s.groupLayout = {};
  s.readback = {};
  s.indices = {};
  s.mesh = {};
  s.instanceBuffer = {};
  s.registered = false;
}
} // namespace render_demo

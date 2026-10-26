#include "fsr_pass.hpp"

#include "../compose.hpp"
#include "../../taa.hpp"
#include "../../temporal_resolve.hpp"
#include "../../../graph_bind.hpp"
#include "../../../graph_layout.hpp"
#include "../../../hints.hpp"
#include "../../../pass_context.hpp"
#include "../../../stage_data.hpp"

#include "../../../../gfx/pipeline_cache.hpp"
#include "../../../../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string>

namespace aurora::render::detail::upscale::fsr_pass {
namespace {
using wgpu::TextureFormat;

constexpr uint64_t LayoutKey = 0x4653'5233'4E4F'4445ull;
constexpr uint64_t ApplyLayoutKey = 0x4653'5233'4150'504Cull;
constexpr uint32_t ParamsOffset = 256;
constexpr uint32_t ApplyOffset = 512;
constexpr float DefaultUnitsPerMeter = 100.f;
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

struct Local {
  uint16_t color = 0;
  uint16_t depth = 0;
  uint16_t motion = 0;
  uint16_t reactive = 0;
  uint16_t dilatedDepth = 0;
  uint16_t dilatedMotion = 0;
  uint16_t farthestDepth = 0;
  uint16_t farthestDepthMip1 = 0;
  uint16_t lumaMip5 = 0;
  uint16_t spd = 0;
  uint16_t shadingChange = 0;
  uint16_t dilatedReactive = 0;
  uint16_t lumaInstability = 0;
  uint16_t luma = 0;
  uint16_t accumulation = 0;
  uint16_t frameInfo = 0;
  uint16_t lumaHistory = 0;
  uint16_t upscaled = 0;
  uint16_t locks = 0;
  uint16_t recon = 0;
  uint8_t inPlace = 0;
  uint8_t locksR8 = 0;
};
static_assert(sizeof(Local) <= PassLocalSize);

struct ApplyParams {
  float sharpness = 0.f;
  float pad[3]{};
};

struct Uniforms {
  FrameUniforms frame;
  fsr::Params params;
  std::array<std::byte, ApplyOffset - ParamsOffset - sizeof(fsr::Params)> gap{};
  ApplyParams apply;
};
static_assert(offsetof(Uniforms, params) == ParamsOffset);
static_assert(offsetof(Uniforms, apply) == ApplyOffset);

struct Prepared {
  wgpu::BindGroupLayout frameLayout;
  bool requested = false;
  bool locksR8 = false;
  fsr::Pipelines compute;
  wgpu::BindGroupLayout applyLayout;
  wgpu::PipelineLayout applyPipelineLayout;
  TextureFormat applyFormat = TextureFormat::Undefined;
  shaders::PipelineId apply;
};

enum class Dummy : uint8_t { Luma, Accumulation, FrameInfo, Rgba16 };
constexpr std::array<TextureFormat, 4> DummyFormats{TextureFormat::R16Float, TextureFormat::R8Unorm,
                                                    TextureFormat::RGBA32Float, TextureFormat::RGBA16Float};

struct Worker {
  wgpu::Buffer uniforms;
  wgpu::BindGroupLayout frameLayout;
  wgpu::BindGroup frameGroup;
  std::array<wgpu::Sampler, 4> samplers;
  std::array<wgpu::TextureView, DummyFormats.size()> dummies;
  bool forceReset = true;
  uint32_t framesSinceReset = 0;
  std::array<float, 2> lastJitter{};
};

PassId g_pass = InvalidPassId;

std::mutex g_mutex;
Prepared g_prepared;
std::atomic_bool g_live{false};

// Game thread
std::optional<std::chrono::steady_clock::time_point> g_lastLatch;

// Render worker only
Worker g_worker;
testing::Frame g_lastFrame;

std::atomic_uint32_t g_encodes{0};
std::atomic_uint32_t g_resets{0};
std::atomic_uint32_t g_published{0};
std::atomic_uint32_t g_applied{0};
std::atomic_uint32_t g_skipped{0};
std::atomic_uint32_t g_skippedNoScene{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

Local read_local(std::span<const std::byte, PassLocalSize> bytes) noexcept {
  Local local;
  std::memcpy(&local, bytes.data(), sizeof(local));
  return local;
}

void ensure_apply_layout_locked() {
  if (g_prepared.applyLayout) {
    return;
  }
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(ApplyParams)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "aurora.fsr apply",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_prepared.applyLayout = webgpu::g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "aurora.fsr apply",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_prepared.applyLayout,
  };
  g_prepared.applyPipelineLayout = webgpu::g_device.CreatePipelineLayout(&pipelineDescriptor);
}

bool request_locked(const wgpu::BindGroupLayout& frameLayout, bool locksR8, bool inPlace, TextureFormat colorFormat) {
  if (!webgpu::g_device || !frameLayout) {
    return false;
  }
  if (!g_prepared.requested || g_prepared.frameLayout.Get() != frameLayout.Get() || g_prepared.locksR8 != locksR8) {
    fsr::Pipelines compute;
    if (!fsr::request_pipelines(frameLayout, LayoutKey, locksR8, compute)) {
      return false;
    }
    g_prepared.frameLayout = frameLayout;
    g_prepared.locksR8 = locksR8;
    g_prepared.compute = compute;
    g_prepared.requested = true;
  }
  if (inPlace && g_prepared.applyFormat != colorFormat) {
    ensure_apply_layout_locked();
    shaders::RenderPipelineDesc desc = testing::apply_desc(colorFormat);
    desc.layout = g_prepared.applyPipelineLayout;
    desc.layoutKey = ApplyLayoutKey;
    g_prepared.apply = shaders::request_render(desc);
    g_prepared.applyFormat = colorFormat;
  }
  return true;
}

bool ready_locked() noexcept {
  return g_prepared.requested && fsr::pipelines_ready(g_prepared.compute) &&
         (!g_prepared.apply || shaders::ready(g_prepared.apply));
}

bool pipelines_ready() noexcept {
  std::lock_guard lock{g_mutex};
  return ready_locked();
}

Prepared snapshot() {
  std::lock_guard lock{g_mutex};
  return g_prepared;
}

wgpu::Sampler create_sampler(const wgpu::Device& device, wgpu::FilterMode filter, wgpu::AddressMode address,
                             wgpu::CompareFunction compare = wgpu::CompareFunction::Undefined) {
  const wgpu::SamplerDescriptor descriptor{
      .label = "aurora.fsr",
      .addressModeU = address,
      .addressModeV = address,
      .addressModeW = address,
      .magFilter = filter,
      .minFilter = filter,
      .mipmapFilter =
          filter == wgpu::FilterMode::Linear ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest,
      .compare = compare,
  };
  return device.CreateSampler(&descriptor);
}

const wgpu::BindGroup& frame_group(const wgpu::Device& device, const wgpu::BindGroupLayout& layout) {
  if (!g_worker.uniforms) {
    const wgpu::BufferDescriptor descriptor{
        .label = "aurora.fsr uniforms",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Uniforms),
    };
    g_worker.uniforms = device.CreateBuffer(&descriptor);
    g_worker.samplers = {
        create_sampler(device, wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge),
        create_sampler(device, wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge),
        create_sampler(device, wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
        create_sampler(device, wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge,
                       wgpu::CompareFunction::GreaterEqual),
    };
  }
  if (!g_worker.frameGroup || g_worker.frameLayout.Get() != layout.Get()) {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = g_worker.uniforms, .size = FrameUniformsSize},
        wgpu::BindGroupEntry{.binding = 1, .buffer = g_worker.uniforms, .size = MaxPassParamsSize},
        wgpu::BindGroupEntry{.binding = 2, .sampler = g_worker.samplers[0]},
        wgpu::BindGroupEntry{.binding = 3, .sampler = g_worker.samplers[1]},
        wgpu::BindGroupEntry{.binding = 4, .sampler = g_worker.samplers[2]},
        wgpu::BindGroupEntry{.binding = 5, .sampler = g_worker.samplers[3]},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .label = "aurora.fsr frame",
        .layout = layout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_worker.frameGroup = device.CreateBindGroup(&descriptor);
    g_worker.frameLayout = layout;
  }
  return g_worker.frameGroup;
}

const wgpu::TextureView& dummy(const wgpu::Device& device, Dummy kind) {
  const auto index = static_cast<size_t>(kind);
  if (!g_worker.dummies[index]) {
    const wgpu::TextureDescriptor descriptor{
        .label = "aurora.fsr empty history",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1, 1},
        .format = DummyFormats[index],
    };
    g_worker.dummies[index] = device.CreateTexture(&descriptor).CreateView();
  }
  return g_worker.dummies[index];
}

Params read_params(std::span<const std::byte> bytes) noexcept {
  Params params;
  if (bytes.size() >= sizeof(params)) {
    std::memcpy(&params, bytes.data(), sizeof(params));
  }
  return params;
}

float units_per_meter(const SceneHints& hints) noexcept {
  const float value = hints.environment.value.unitsPerMeter;
  return hints.has(SceneHasEnvironment) && std::isfinite(value) && value > 0.f ? value : DefaultUnitsPerMeter;
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  const Settings* settings = setup.settings;
  if (settings == nullptr || settings->debugView != 0 || setup.env == nullptr || setup.caps == nullptr ||
      !setup.caps->textureFormatsTier1 || !setup.gx_available(GxResource::Motion) ||
      temporal_resolve::installed().pass != Pass) {
    builder.skip();
    return;
  }
  const GraphExtent scene = setup.env->sceneSize;
  const bool inPlace = scene.width == 0 || scene == setup.env->efbSize;
  const bool locksR8 = setup.caps->textureFormatsTier2;
  builder.after(phase::Taa);
  builder.manual_bindings();
  Local local{.inPlace = static_cast<uint8_t>(inPlace), .locksR8 = static_cast<uint8_t>(locksR8)};
  local.color = builder.use(resource::GxColor, inPlace ? Access::ColorRMW : Access::Sampled, UseOptions{}).index;
  local.depth =
      builder
          .use(resource::GxDepth, Access::Sampled,
               UseOptions{.missing = Missing::BindDummyBlack, .sampleType = wgpu::TextureSampleType::UnfilterableFloat})
          .index;
  local.motion = builder.use(resource::Motion, Access::Sampled, UseOptions{.missing = Missing::DisablePass}).index;
  local.reactive =
      builder.use(resource::Reactive, Access::Sampled, UseOptions{.missing = Missing::BindDummyBlack}).index;
  const fsr::Refs refs = fsr::declare(builder, {.quality = settings->upscalerQuality, .locksR8 = locksR8});
  local.dilatedDepth = refs.dilatedDepth.index;
  local.dilatedMotion = refs.dilatedMotion.index;
  local.farthestDepth = refs.farthestDepth.index;
  local.farthestDepthMip1 = refs.farthestDepthMip1.index;
  local.lumaMip5 = refs.lumaMip5.index;
  local.spd = refs.spd.index;
  local.shadingChange = refs.shadingChange.index;
  local.dilatedReactive = refs.dilatedReactive.index;
  local.lumaInstability = refs.lumaInstability.index;
  local.luma = refs.luma.index;
  local.accumulation = refs.accumulation.index;
  local.frameInfo = refs.frameInfo.index;
  local.lumaHistory = refs.lumaHistory.index;
  local.upscaled = refs.upscaled.index;
  local.locks = refs.locks.index;
  local.recon = refs.recon.index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

bool enabled(const StageData& stage, const void*) {
  const bool scene = stage.markerSeen && !stage.flushedAtFinish && stage.sceneHints != nullptr &&
                     stage.sceneHints->has(SceneHasMainCamera);
  if (!scene) {
    bump(g_skippedNoScene);
    return false;
  }
  return pipelines_ready();
}

void skip_frame() {
  g_worker.forceReset = true;
  bump(g_skipped);
}

void encode(PassContext& ctx, const void*) {
  const Local l = read_local(ctx.local());
  const StageData& stage = ctx.stage_data();
  const SceneHints* hints = stage.sceneHints.get();
  const Prepared prepared = snapshot();
  const wgpu::Device& device = ctx.device();
  const bool pipelines = prepared.requested && prepared.frameLayout.Get() == ctx.frame_layout().Get() &&
                         fsr::pipelines_ready(prepared.compute) && (l.inPlace == 0 || shaders::ready(prepared.apply));
  if (!pipelines || !device || hints == nullptr) {
    skip_frame();
    return;
  }
  const TexRef upscaledRef{l.upscaled};
  const bool reset = g_worker.forceReset || !hints->motionValid || !ctx.prev_valid(upscaledRef);
  const auto view = [&](uint16_t ref, uint32_t mip = AllMips) { return ctx.view(TexRef{ref}, mip); };
  bool historyBound = false;
  const auto prev = [&](uint16_t ref, Dummy kind) {
    wgpu::TextureView previous = reset ? wgpu::TextureView{} : ctx.prev(TexRef{ref});
    historyBound = historyBound || static_cast<bool>(previous);
    return previous ? previous : dummy(device, kind);
  };
  const fsr::Bindings bindings{
      .color = view(l.color),
      .depth = view(l.depth),
      .motion = view(l.motion),
      .reactive = view(l.reactive),
      .dilatedDepth = view(l.dilatedDepth),
      .dilatedMotion = view(l.dilatedMotion),
      .farthestDepth = view(l.farthestDepth),
      .farthestDepthMip1 = view(l.farthestDepthMip1),
      .lumaMip5 = view(l.lumaMip5),
      .spd = view(l.spd),
      .spdMips = {view(l.spd, 0), view(l.spd, 1), view(l.spd, 2)},
      .shadingChange = view(l.shadingChange),
      .dilatedReactive = view(l.dilatedReactive),
      .lumaInstability = view(l.lumaInstability),
      .luma = view(l.luma),
      .prevLuma = prev(l.luma, Dummy::Luma),
      .accumulation = view(l.accumulation),
      .prevAccumulation = prev(l.accumulation, Dummy::Accumulation),
      .frameInfo = view(l.frameInfo),
      .prevFrameInfo = prev(l.frameInfo, Dummy::FrameInfo),
      .lumaHistory = view(l.lumaHistory),
      .prevLumaHistory = prev(l.lumaHistory, Dummy::Rgba16),
      .upscaled = view(l.upscaled),
      .prevUpscaled = prev(l.upscaled, Dummy::Rgba16),
      .locks = view(l.locks),
      .recon = ctx.buffer(BufRef{l.recon}),
  };
  const wgpu::Extent3D renderExtent = ctx.extent(TexRef{l.color});
  const wgpu::Extent3D outputExtent = ctx.extent(upscaledRef);
  const GraphExtent render{renderExtent.width, renderExtent.height};
  const GraphExtent output{outputExtent.width, outputExtent.height};
  if (render.width == 0 || render.height == 0 || output.width == 0 || output.height == 0) {
    skip_frame();
    return;
  }
  const Params latched = read_params(ctx.params());
  const std::array<float, 2> jitter{hints->jitterPx[0] * latched.jitterSign[0],
                                    hints->jitterPx[1] * latched.jitterSign[1]};
  const std::array<float, 2> prevJitter = reset ? jitter : g_worker.lastJitter;
  const fsr::FrameInputs inputs{
      .jitter = {jitter[0], jitter[1]},
      .prevJitter = {prevJitter[0], prevJitter[1]},
      .motionScale = {fsr::MotionScale[0] * latched.motionSign[0], fsr::MotionScale[1] * latched.motionSign[1]},
      .camera = fsr::camera_from_projection(hints->mainCamera.gxProj, units_per_meter(*hints))
                    .value_or(fsr::Camera{.nearZ = 1.f, .farZ = 100000.f, .fovY = 1.f, .viewToMeters = 0.01f}),
      .frameTimeMs = latched.frameTimeMs,
      .framesSinceReset = reset ? 0 : g_worker.framesSinceReset,
      .jitterPhases = fsr::jitter_phases(static_cast<float>(output.width) / static_cast<float>(render.width)),
      .render = render,
  };
  Uniforms uniforms;
  FrameConstants& constants = uniforms.frame.constants;
  constants.efbSize[0] = render.width;
  constants.efbSize[1] = render.height;
  constants.outputSize[0] = output.width;
  constants.outputSize[1] = output.height;
  constants.invEfbSize[0] = 1.f / static_cast<float>(render.width);
  constants.invEfbSize[1] = 1.f / static_cast<float>(render.height);
  constants.invOutputSize[0] = 1.f / static_cast<float>(output.width);
  constants.invOutputSize[1] = 1.f / static_cast<float>(output.height);
  constants.frameIndex = stage.frameIndex;
  constants.flags = FrameFlagReversedZ;
  constants.msaaSamples = 1;
  uniforms.params = fsr::make_params(inputs);
  uniforms.params.reactiveMax = ReactiveMax;
  uniforms.apply.sharpness = latched.sharpness;
  const fsr::FrameGroup frame{
      .group = frame_group(device, prepared.frameLayout),
      .offsets = {0, ParamsOffset},
      .offsetCount = 2,
  };
  ctx.encoder().WriteBuffer(g_worker.uniforms, 0, &uniforms, sizeof(uniforms));
  if (!fsr::encode(ctx.encoder(), prepared.compute, frame, bindings, render, output)) {
    skip_frame();
    return;
  }
  wgpu::Texture published;
  if (l.inPlace != 0) {
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0, .buffer = g_worker.uniforms, .offset = ApplyOffset, .size = sizeof(ApplyParams)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = bindings.upscaled},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .label = "aurora.fsr apply",
        .layout = prepared.applyLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const wgpu::BindGroup group = device.CreateBindGroup(&descriptor);
    const std::array<ColorTarget, 1> colors{{{.ref = TexRef{l.color}}}};
    if (const wgpu::RenderPassEncoder pass = ctx.begin_render(colors, nullptr, "aurora.fsr apply")) {
      pass.SetPipeline(shaders::render_pipeline(prepared.apply));
      pass.SetBindGroup(0, group);
      pass.Draw(3);
      pass.End();
      bump(g_applied);
    }
  } else {
    // A history outlives the graph's last use, so compose can still read it after PrePost
    compose::publish_source({.view = bindings.upscaled, .format = TextureFormat::RGBA16Float});
    published = ctx.texture(upscaledRef);
    bump(g_published);
  }
  g_worker.forceReset = false;
  g_worker.framesSinceReset = inputs.framesSinceReset + 1;
  g_worker.lastJitter = jitter;
  g_lastFrame = {.reset = reset, .historyBound = historyBound, .params = uniforms.params, .published = published};
  bump(g_encodes);
  if (reset) {
    bump(g_resets);
  }
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::PrePost,
      .kind = PassKind::Encoder,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
  };
}

void initialize() {
  if (g_pass == InvalidPassId) {
    g_pass = register_pass(pass_desc());
  }
}

void shutdown() noexcept {
  if (g_pass != InvalidPassId) {
    unregister_pass(g_pass);
    g_pass = InvalidPassId;
  }
  g_live.store(false, std::memory_order_relaxed);
  g_lastLatch.reset();
  g_worker = {};
  g_lastFrame = {};
  {
    std::lock_guard lock{g_mutex};
    g_prepared = {};
  }
  fsr::shutdown();
}

bool prepare(bool inPlace, wgpu::TextureFormat colorFormat) {
  const wgpu::BindGroupLayout frameLayout = frame_layout();
  const bool locksR8 = caps().textureFormatsTier2;
  std::lock_guard lock{g_mutex};
  return request_locked(frameLayout, locksR8, inPlace, colorFormat) && ready_locked();
}

void on_latch(const Settings& settings, const CompiledGraph* graph, const Signs& signs) {
  const float frameTimeMs =
      fsr::frame_delta_ms(g_lastLatch, std::chrono::steady_clock::now(), gfx::blocking_pipelines());
  if (g_pass != InvalidPassId) {
    const Params params{
        .frameTimeMs = frameTimeMs,
        .sharpness = taa::rcas_strength(settings.taaSharpness),
        .jitterSign = {signs.jitter[0], signs.jitter[1]},
        .motionSign = {signs.motion[0], signs.motion[1]},
    };
    set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
  }
  bool live = false;
  bool inPlace = false;
  if (graph != nullptr) {
    const NodeId node = graph->find_node(Pass);
    if (node != InvalidNode && graph->nodes[node].kind == NodeKind::Pass &&
        graph->nodes[node].status == CullReason::None) {
      live = true;
      inPlace = read_local(graph->nodes[node].local).inPlace != 0;
    }
  }
  if (live) {
    const wgpu::BindGroupLayout frameLayout = frame_layout();
    const bool locksR8 = caps().textureFormatsTier2;
    std::lock_guard lock{g_mutex};
    request_locked(frameLayout, locksR8, inPlace, webgpu::g_frameBuffer.format);
  }
  g_live.store(live, std::memory_order_relaxed);
}

bool jitter_ready() noexcept { return g_live.load(std::memory_order_relaxed) && hints_enabled() && pipelines_ready(); }

bool ready() noexcept { return jitter_ready(); }

namespace testing {
Counters counters() noexcept {
  return {
      .encodes = g_encodes.load(std::memory_order_relaxed),
      .resets = g_resets.load(std::memory_order_relaxed),
      .published = g_published.load(std::memory_order_relaxed),
      .applied = g_applied.load(std::memory_order_relaxed),
      .skipped = g_skipped.load(std::memory_order_relaxed),
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
  };
}

Frame last_frame() noexcept { return g_lastFrame; }

void reset() noexcept {
  g_lastFrame = {};
  g_worker.forceReset = true;
  g_encodes.store(0, std::memory_order_relaxed);
  g_resets.store(0, std::memory_order_relaxed);
  g_published.store(0, std::memory_order_relaxed);
  g_applied.store(0, std::memory_order_relaxed);
  g_skipped.store(0, std::memory_order_relaxed);
  g_skippedNoScene.store(0, std::memory_order_relaxed);
}

shaders::RenderPipelineDesc apply_desc(wgpu::TextureFormat format) {
  shaders::RenderPipelineDesc desc{
      .label = "aurora.fsr apply",
      .shader = ApplyShader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .colorCount = 1,
  };
  desc.colors[0] = {.format = format, .writeMask = RgbMask};
  return desc;
}
} // namespace testing
} // namespace aurora::render::detail::upscale::fsr_pass

#include "native_upscale.hpp"

#include "upscale.hpp"
#include "fsr/fsr.hpp"
#include "../taa.hpp"
#include "../temporal_resolve.hpp"
#include "../../hints.hpp"
#include "../../native/native.hpp"
#include "../../pass_context.hpp"
#include "../../stage_data.hpp"

#include "../../../gfx/pipeline_cache.hpp"
#include "../../../logging.hpp"
#include "../../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <span>
#include <string>

namespace aurora::render::detail::upscale::native_upscale {
namespace {
constexpr Module Log{"aurora::render::upscale"};
constexpr std::string_view TokenHistory = "native_upscale.token";
constexpr uint64_t ColorLayoutKey = 0x4E41'5455'5053'434Cull;
constexpr uint64_t DepthLayoutKey = 0x4E41'5455'5053'4450ull;
constexpr std::array<std::string_view, 1> DepthDefines{"NATIVE_DEPTH"};
constexpr float DefaultUnitsPerMeter = 100.f;

struct Local {
  uint16_t color = 0;
  uint16_t depth = 0;
  uint16_t motion = 0;
  uint16_t reactive = 0;
  uint16_t token = 0;
  uint8_t linear = 0;
  uint8_t reactiveUsed = 0;
  Plan plan;
};
static_assert(sizeof(Local) <= PassLocalSize);

struct Slot {
  temporal_resolve::Resolver resolver;
  NativeUpscaler* upscaler = nullptr;
  Plan plan;
  GraphExtent scene;
};

struct InputPipeline {
  wgpu::BindGroupLayout layout;
  wgpu::PipelineLayout pipelineLayout;
  shaders::PipelineId id;
};

struct InputGroup {
  wgpu::TextureView source;
  wgpu::BindGroup group;
};

struct Worker {
  bool forceReset = true;
  InputGroup color;
  InputGroup depth;
};

struct StandIn {
  bool set = false;
  NativeUpscaler* upscaler = nullptr;
};

PassId g_pass = InvalidPassId;

std::mutex g_mutex;
InputPipeline g_linearInput;
InputPipeline g_depthInput;
std::atomic_bool g_live{false};
std::atomic_bool g_linear{false};
std::atomic<NativeUpscaler*> g_active{nullptr};

// Game thread
std::array<Slot, 2> g_slots;
const Slot* g_installed = nullptr;
bool g_requested = false;
StandIn g_standIn;
std::optional<std::chrono::steady_clock::time_point> g_lastLatch;

// Render worker only
Worker g_worker;

std::atomic_uint32_t g_encodes{0};
std::atomic_uint32_t g_resets{0};
std::atomic_uint32_t g_skipped{0};
std::atomic_uint32_t g_skippedNoScene{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

bool wants_dlss(const Settings& settings) noexcept { return settings.upscaler == Upscaler::Dlss; }
bool wants_xess(const Settings& settings) noexcept { return settings.upscaler == Upscaler::Xess; }
bool wants_none(const Settings&) noexcept { return false; }

template <size_t Index>
float slot_mip_bias(float ratio) noexcept {
  const NativeUpscaler* upscaler = g_slots[Index].upscaler;
  return upscaler != nullptr ? upscaler->mip_bias(ratio) : std::log2(1.f / ratio);
}

Local read_local(std::span<const std::byte, PassLocalSize> bytes) noexcept {
  Local local;
  std::memcpy(&local, bytes.data(), sizeof(local));
  return local;
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

bool request_input_locked(InputPipeline& pipeline, bool depth) {
  if (pipeline.id) {
    return true;
  }
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return false;
  }
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = depth ? wgpu::TextureSampleType::Depth : wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D},
  };
  const char* label = depth ? "aurora.native_upscale depth" : "aurora.native_upscale linear";
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.label = label, .entryCount = 1, .entries = &entry};
  pipeline.layout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = label,
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &pipeline.layout,
  };
  pipeline.pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  shaders::RenderPipelineDesc desc = testing::input_desc(depth);
  desc.layout = pipeline.pipelineLayout;
  desc.layoutKey = depth ? DepthLayoutKey : ColorLayoutKey;
  pipeline.id = shaders::request_render(desc);
  return true;
}

bool request_inputs_locked(bool linear) {
  return request_input_locked(g_depthInput, true) && (!linear || request_input_locked(g_linearInput, false));
}

bool inputs_ready_locked(bool linear) noexcept {
  return g_depthInput.id && shaders::ready(g_depthInput.id) &&
         (!linear || (g_linearInput.id && shaders::ready(g_linearInput.id)));
}

bool pipelines_ready() noexcept {
  std::lock_guard lock{g_mutex};
  return inputs_ready_locked(g_linear.load(std::memory_order_relaxed));
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  const Settings* settings = setup.settings;
  const Slot* slot = g_installed;
  if (settings == nullptr || settings->debugView != 0 || setup.env == nullptr || slot == nullptr ||
      slot->upscaler == nullptr || temporal_resolve::installed().pass != Pass ||
      !setup.gx_available(GxResource::Motion)) {
    builder.skip();
    return;
  }
  const NativeUpscaler& upscaler = *slot->upscaler;
  const bool linear = upscaler.color_encoding() == ColorEncoding::Linear;
  const bool reactive = upscaler.wants_reactive();
  builder.after(phase::Taa);
  builder.manual_bindings();
  Local local{
      .linear = static_cast<uint8_t>(linear), .reactiveUsed = static_cast<uint8_t>(reactive), .plan = slot->plan};
  local.color = builder.use(resource::GxColor, linear ? Access::Sampled : Access::CopySrc, UseOptions{}).index;
  local.depth = builder.use(resource::GxDepth, Access::Sampled, UseOptions{}).index;
  local.motion = builder.use(resource::Motion, Access::CopySrc, UseOptions{.missing = Missing::DisablePass}).index;
  if (reactive) {
    local.reactive =
        builder.use(resource::Reactive, Access::CopySrc, UseOptions{.missing = Missing::DisablePass}).index;
  }
  local.token = builder
                    .history(TokenHistory,
                             TexDesc{
                                 .format = wgpu::TextureFormat::R8Unorm,
                                 .sizeClass = SizeClass::Absolute,
                                 .width = 1,
                                 .height = 1,
                             },
                             Access::ColorWrite)
                    .index;
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

void encode_input(PassContext& ctx, const InputPipeline& pipeline, InputGroup& cache, const wgpu::TextureView& source,
                  const wgpu::TextureView& target, const char* label) {
  if (cache.source.Get() != source.Get() || !cache.group) {
    const wgpu::BindGroupEntry entry{.binding = 0, .textureView = source};
    const wgpu::BindGroupDescriptor descriptor{
        .label = label,
        .layout = pipeline.layout,
        .entryCount = 1,
        .entries = &entry,
    };
    cache.group = ctx.device().CreateBindGroup(&descriptor);
    cache.source = source;
  }
  const wgpu::RenderPassColorAttachment attachment{
      .view = target,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor descriptor{
      .label = label,
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
  };
  const wgpu::RenderPassEncoder pass = ctx.encoder().BeginRenderPass(&descriptor);
  pass.SetPipeline(shaders::render_pipeline(pipeline.id));
  pass.SetBindGroup(0, cache.group);
  pass.Draw(3);
  pass.End();
}

void copy_texture(const wgpu::CommandEncoder& encoder, const wgpu::Texture& from, const wgpu::Texture& to,
                  GraphExtent size) {
  const wgpu::TexelCopyTextureInfo source{.texture = from};
  const wgpu::TexelCopyTextureInfo destination{.texture = to};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  encoder.CopyTextureToTexture(&source, &destination, &extent);
}

void encode(PassContext& ctx, const void*) {
  const Local l = read_local(ctx.local());
  const StageData& stage = ctx.stage_data();
  const SceneHints* hints = stage.sceneHints.get();
  const wgpu::Extent3D renderExtent = ctx.extent(TexRef{l.color});
  const GraphExtent render{renderExtent.width, renderExtent.height};
  const GraphExtent output = output_size();
  if (hints == nullptr || render.width == 0 || render.height == 0 || output.width == 0 || output.height == 0 ||
      ctx.format(TexRef{l.depth}) != wgpu::TextureFormat::Depth32Float) {
    skip_frame();
    return;
  }
  const Params params = read_params(ctx.params());
  const bool reset = g_worker.forceReset || !hints->motionValid || !ctx.prev_valid(TexRef{l.token});
  const fsr::Camera camera =
      fsr::camera_from_projection(hints->mainCamera.gxProj, units_per_meter(*hints))
          .value_or(fsr::Camera{.nearZ = 1.f, .farZ = 100000.f, .fovY = 1.f, .viewToMeters = 0.01f});
  const Frame frame{
      .plan = l.plan,
      .render = render,
      .output = output,
      .colorFormat = ctx.format(TexRef{l.color}),
      .jitter = {hints->jitterPx[0] * params.jitterSign[0], hints->jitterPx[1] * params.jitterSign[1]},
      .motionScale = {fsr::MotionScale[0] * static_cast<float>(render.width) * params.motionSign[0],
                      fsr::MotionScale[1] * static_cast<float>(render.height) * params.motionSign[1]},
      .camera = {.nearZ = camera.nearZ, .farZ = camera.farZ, .fovY = camera.fovY, .viewToMeters = camera.viewToMeters},
      .frameTimeMs = params.frameTimeMs,
      .sharpness = params.sharpness,
      .reset = reset,
  };
  Targets targets;
  if (!native::begin_upscale(frame, targets)) {
    skip_frame();
    return;
  }
  InputPipeline linearInput;
  InputPipeline depthInput;
  {
    std::lock_guard lock{g_mutex};
    linearInput = g_linearInput;
    depthInput = g_depthInput;
  }
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  if (l.linear != 0) {
    encode_input(ctx, linearInput, g_worker.color, ctx.view(TexRef{l.color}), targets.colorView,
                 "aurora.native_upscale linear");
  } else {
    copy_texture(encoder, ctx.texture(TexRef{l.color}), targets.color, render);
  }
  encode_input(ctx, depthInput, g_worker.depth, ctx.view(TexRef{l.depth}), targets.depthView,
               "aurora.native_upscale depth");
  copy_texture(encoder, ctx.texture(TexRef{l.motion}), targets.motion, render);
  if (l.reactiveUsed != 0 && targets.reactive) {
    copy_texture(encoder, ctx.texture(TexRef{l.reactive}), targets.reactive, render);
  }
  g_worker.forceReset = false;
  bump(g_encodes);
  if (reset) {
    bump(g_resets);
  }
}

NativeUpscaler* stand_in_env(float ratio) noexcept {
#ifndef NDEBUG
  static const bool copy = [] {
    const char* value = SDL_getenv("AURORA_NATIVE_TEST_STEP");
    const bool on = value != nullptr && std::string_view{value} == "copy";
    if (on) {
      Log.warn("AURORA_NATIVE_TEST_STEP=copy: a native copy stands in for the resolver");
    }
    return on;
  }();
  return copy ? native::test_upscaler(ratio) : nullptr;
#else
  (void)ratio;
  return nullptr;
#endif
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
  if (g_requested || g_installed != nullptr) {
    native::release_upscaler();
  }
  g_slots = {};
  g_installed = nullptr;
  g_requested = false;
  g_lastLatch.reset();
  g_live.store(false, std::memory_order_relaxed);
  g_linear.store(false, std::memory_order_relaxed);
  g_active.store(nullptr, std::memory_order_relaxed);
  g_worker = {};
  std::lock_guard lock{g_mutex};
  g_linearInput = {};
  g_depthInput = {};
}

NativeUpscaler* stand_in(float ratio) noexcept { return g_standIn.set ? g_standIn.upscaler : stand_in_env(ratio); }

std::optional<Selection> select(NativeUpscaler& upscaler, UpscalerQuality quality, GraphExtent output) {
  const Plan plan{.upscaler = &upscaler, .quality = quality, .output = output};
  GraphExtent render;
  g_requested = true;
  if (!native::prepare_upscaler(plan, render)) {
    return std::nullopt;
  }
  const ColorEncoding encoding = upscaler.color_encoding();
  if (g_installed != nullptr && g_installed->plan == plan && g_installed->scene == render) {
    return Selection{.resolver = &g_installed->resolver, .scene = render, .encoding = encoding};
  }
  const size_t index = g_installed == &g_slots[0] ? 1 : 0;
  Slot& slot = g_slots[index];
  const float ratio = static_cast<float>(output.width) / static_cast<float>(render.width);
  const Upscaler option = upscaler.option();
  slot = {
      .resolver =
          {
              .pass = Pass,
              .jitterPhases = upscaler.jitter_phases(ratio),
              .wanted = option == Upscaler::Dlss   ? &wants_dlss
                        : option == Upscaler::Xess ? &wants_xess
                                                   : &wants_none,
              .jitter_ready = &jitter_ready,
              .mip_bias = index == 0 ? &slot_mip_bias<0> : &slot_mip_bias<1>,
          },
      .upscaler = &upscaler,
      .plan = plan,
      .scene = render,
  };
  return Selection{.resolver = &slot.resolver, .scene = render, .encoding = encoding};
}

bool prepare_pipelines(ColorEncoding encoding) {
  const bool linear = encoding == ColorEncoding::Linear;
  std::lock_guard lock{g_mutex};
  return request_inputs_locked(linear) && inputs_ready_locked(linear);
}

void installed(const temporal_resolve::Resolver* resolver) noexcept {
  g_installed = nullptr;
  for (const Slot& slot : g_slots) {
    if (resolver == &slot.resolver) {
      g_installed = &slot;
    }
  }
  NativeUpscaler* upscaler = g_installed != nullptr ? g_installed->upscaler : nullptr;
  g_linear.store(upscaler != nullptr && upscaler->color_encoding() == ColorEncoding::Linear, std::memory_order_relaxed);
  g_active.store(upscaler, std::memory_order_relaxed);
}

void release() noexcept {
  if (!g_requested) {
    return;
  }
  native::release_upscaler();
  g_requested = false;
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
  if (graph != nullptr) {
    const NodeId node = graph->find_node(Pass);
    live = node != InvalidNode && graph->nodes[node].kind == NodeKind::Pass &&
           graph->nodes[node].status == CullReason::None;
  }
  if (live) {
    std::lock_guard lock{g_mutex};
    request_inputs_locked(g_linear.load(std::memory_order_relaxed));
  }
  g_live.store(live, std::memory_order_relaxed);
}

bool is_native(const temporal_resolve::Resolver& resolver) noexcept { return resolver.pass == Pass; }

bool jitter_ready() noexcept { return g_live.load(std::memory_order_relaxed) && hints_enabled() && pipelines_ready(); }

bool ready() noexcept { return jitter_ready(); }

Upscaler active() noexcept {
  const NativeUpscaler* upscaler = g_active.load(std::memory_order_relaxed);
  return upscaler != nullptr ? upscaler->option() : Upscaler::Off;
}

std::string_view display_name(UpscalerQuality quality) noexcept {
  const NativeUpscaler* upscaler = g_active.load(std::memory_order_relaxed);
  return upscaler != nullptr ? upscaler->display_name(quality) : std::string_view{};
}

std::string_view path() noexcept { return native::upscaler_path(active()); }

namespace testing {
Counters counters() noexcept {
  return {
      .encodes = g_encodes.load(std::memory_order_relaxed),
      .resets = g_resets.load(std::memory_order_relaxed),
      .skipped = g_skipped.load(std::memory_order_relaxed),
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
  };
}

void set_stand_in(NativeUpscaler* upscaler) noexcept { g_standIn = {.set = true, .upscaler = upscaler}; }

void reset() noexcept {
  g_standIn = {};
  g_worker.forceReset = true;
  g_encodes.store(0, std::memory_order_relaxed);
  g_resets.store(0, std::memory_order_relaxed);
  g_skipped.store(0, std::memory_order_relaxed);
  g_skippedNoScene.store(0, std::memory_order_relaxed);
}

shaders::RenderPipelineDesc input_desc(bool depth) {
  shaders::RenderPipelineDesc desc{
      .label = depth ? "aurora.native_upscale depth" : "aurora.native_upscale linear",
      .shader = InputShader,
      .vertexEntry = "vs_main",
      .fragmentEntry = depth ? "fs_depth" : "fs_linear",
      .defines = depth ? std::span<const std::string_view>{DepthDefines} : std::span<const std::string_view>{},
      .colorCount = 1,
  };
  desc.colors[0] = {.format = depth ? wgpu::TextureFormat::R32Float : wgpu::TextureFormat::RGBA16Float};
  return desc;
}
} // namespace testing
} // namespace aurora::render::detail::upscale::native_upscale

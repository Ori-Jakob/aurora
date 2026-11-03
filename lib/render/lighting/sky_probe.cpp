#include "sky_probe.hpp"

#include "clusters.hpp"
#include "deferred.hpp"

#include "../debug/views.hpp"
#include "../frame_state.hpp"
#include "../graph_layout.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../post/upscale/fsr/fsr.hpp"
#include "../scene/classify.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gfx/pipeline_cache.hpp"
#include "../../gx/hints.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::detail::lighting::sky_probe {
namespace {
constexpr Module Log{"aurora::render::lighting"};
constexpr double Pi = std::numbers::pi;
constexpr TexRef RadianceRef{3};
constexpr TexRef SourceRef{0};
constexpr TexRef ProbeRef{1};
constexpr BufRef Sh9Ref{2};
constexpr TexRef ViewTarget{5};
constexpr uint32_t PrevGroupIndex = 2;
constexpr uint64_t LayoutSalt = 0x534B595052424Full;
constexpr uint64_t RetainFrames = 8;
constexpr size_t CacheCapacity = 4;
constexpr std::array<std::string_view, 1> ProjectDefines{"SKY_PROJECT"};

struct LatchSlot {
  bool declared = false;
  bool armed = false;
  std::array<uint32_t, MaxPassPipelines> project{};
  std::array<uint32_t, MaxPassPipelines> filter{};
};

struct PipelineLayoutEntry {
  wgpu::BindGroupLayout passLayout;
  wgpu::PipelineLayout layout;
};

struct PrevEntry {
  wgpu::BindGroupLayout layout;
  wgpu::TextureView prev;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

struct FilterEntry {
  wgpu::Texture probe;
  wgpu::Buffer sh9;
  uint64_t offset = 0;
  wgpu::BindGroupLayout layout;
  std::array<wgpu::BindGroup, ProbeMips> downsample;
  wgpu::BindGroup sh;
  uint64_t lastUsed = 0;
};

PassId g_project = InvalidPassId;
PassId g_filter = InvalidPassId;
PassId g_view = InvalidPassId;
std::mutex g_slotMutex;
LatchSlot g_slot;
std::atomic_bool g_armed{false};
std::atomic_uint32_t g_alphaBits{std::bit_cast<uint32_t>(1.f)};
std::optional<std::chrono::steady_clock::time_point> g_lastLatch;
std::shared_ptr<const SceneHints> g_hintsOverride;
std::optional<Hook> g_hookOverride;
std::optional<uint32_t> g_skyDrawsOverride;
std::optional<float> g_alphaOverride;

std::mutex g_layoutMutex;
wgpu::BindGroupLayout g_prevLayout;
std::vector<PipelineLayoutEntry> g_pipelineLayouts;
std::vector<PrevEntry> g_prevCache;
std::vector<FilterEntry> g_filterCache;
wgpu::Texture g_dummyTexture;
wgpu::TextureView g_dummyView;

std::atomic_uint32_t g_observed{0};
std::atomic_uint32_t g_held{0};
std::atomic_uint32_t g_resets{0};
std::atomic_uint32_t g_filtered{0};
std::atomic_uint32_t g_constant{0};
std::atomic_uint32_t g_viewDraws{0};
std::atomic_uint32_t g_lastMode{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

Hook env_hook() noexcept {
  static const Hook value = [] {
    const char* text = SDL_getenv(std::string{HookEnv}.c_str());
    if (text == nullptr) {
      return Hook{};
    }
    if (const std::optional<Hook> parsed = parse_hook(text)) {
      Log.warn("{}={}: the sky probe runs a test configuration, for verification only", HookEnv, text);
      return *parsed;
    }
    Log.warn("{}={} is not off, constant:<value>, physical or global; ignored", HookEnv, text);
    return Hook{};
  }();
  return value;
}

const SceneHints* stage_hints(const StageData& stage) noexcept {
  return g_hintsOverride != nullptr ? g_hintsOverride.get() : stage.sceneHints.get();
}

bool camera_usable(const SceneHints* hints) noexcept {
  return hints != nullptr && hints->has(SceneHasMainCamera) && gx::hints::main_camera_usable(hints->mainCamera.gxProj);
}

bool node_live(const CompiledGraph& graph, std::string_view name) noexcept {
  const NodeId node = graph.find_node(name);
  return node != InvalidNode && graph.nodes[node].kind == NodeKind::Pass &&
         graph.nodes[node].status == CullReason::None;
}

LatchSlot latched_slot() {
  std::lock_guard lock{g_slotMutex};
  return g_slot;
}

float unit(float value, float fallback) noexcept {
  return std::isfinite(value) ? std::clamp(value, 0.f, 1.f) : fallback;
}

uint32_t sky_draws_now() noexcept { return g_skyDrawsOverride.value_or(scene::sky_draws()); }

bool observing(const SceneHints* hints, uint32_t skyDraws, bool constant, bool rotationOk) noexcept {
  const bool visible =
      hints == nullptr || !hints->has(SceneHasEnvironment) || unit(hints->environment.value.skyVisibility, 0.f) > 0.f;
  return !constant && skyDraws > 0 && rotationOk && visible;
}

bool row_selected(uint16_t view) noexcept {
  const debug::GbufView* row = debug::find_gbuf_view(view);
  return row != nullptr && row->resource == ViewResource;
}

bool scene_enabled(const StageData& stage, const void*) {
  return armed() && !stage.flushedAtFinish && camera_usable(stage_hints(stage));
}

void declare_project(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.use(resource::GxColor, Access::Sampled, Missing::DisablePass);
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::Fog, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.history(Radiance, radiance_desc(), Access::StorageWrite);
}

wgpu::PipelineLayout project_layout(const PipelineSetup& setup) {
  if (!webgpu::g_device || !setup.frameLayout || !setup.passLayout) {
    return {};
  }
  std::lock_guard lock{g_layoutMutex};
  if (!g_prevLayout) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
    };
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "sky.probe_project history",
        .entryCount = 1,
        .entries = &entry,
    };
    g_prevLayout = webgpu::g_device.CreateBindGroupLayout(&descriptor);
  }
  for (const PipelineLayoutEntry& entry : g_pipelineLayouts) {
    if (entry.passLayout.Get() == setup.passLayout.Get()) {
      return entry.layout;
    }
  }
  const std::array groups{setup.frameLayout, setup.passLayout, g_prevLayout};
  const wgpu::PipelineLayoutDescriptor descriptor{
      .label = "sky.probe_project",
      .bindGroupLayoutCount = groups.size(),
      .bindGroupLayouts = groups.data(),
  };
  wgpu::PipelineLayout layout = webgpu::g_device.CreatePipelineLayout(&descriptor);
  g_pipelineLayouts.push_back({.passLayout = setup.passLayout, .layout = layout});
  return layout;
}

wgpu::BindGroupLayout prev_layout() {
  std::lock_guard lock{g_layoutMutex};
  return g_prevLayout;
}

void request_project(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  const wgpu::PipelineLayout layout = project_layout(setup);
  if (!layout) {
    return;
  }
  ids[0] = shaders::request_compute({
                                        .label = "sky.probe_project",
                                        .shader = Shader,
                                        .entryPoint = "project",
                                        .defines = ProjectDefines,
                                        .layout = layout,
                                        .layoutKey = setup.layoutHash ^ LayoutSalt,
                                    })
               .value;
}

const wgpu::TextureView& dummy_prev(const wgpu::Device& device) {
  if (!g_dummyView && device) {
    const wgpu::TextureDescriptor descriptor{
        .label = "sky.probe empty history",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1, 1},
        .format = Format,
    };
    g_dummyTexture = device.CreateTexture(&descriptor);
    g_dummyView = g_dummyTexture.CreateView();
  }
  return g_dummyView;
}

wgpu::BindGroup prev_group(const PassContext& ctx, const wgpu::BindGroupLayout& layout, const wgpu::TextureView& prev) {
  const uint64_t frame = ctx.frame_id();
  for (PrevEntry& entry : g_prevCache) {
    if (entry.layout.Get() == layout.Get() && entry.prev.Get() == prev.Get()) {
      entry.lastUsed = frame;
      return entry.group;
    }
  }
  if (g_prevCache.size() >= CacheCapacity) {
    g_prevCache.erase(std::ranges::min_element(g_prevCache, {}, &PrevEntry::lastUsed));
  }
  const wgpu::BindGroupEntry entry{.binding = 0, .textureView = prev};
  const wgpu::BindGroupDescriptor descriptor{
      .label = "sky.probe_project history",
      .layout = layout,
      .entryCount = 1,
      .entries = &entry,
  };
  return g_prevCache
      .emplace_back(PrevEntry{
          .layout = layout,
          .prev = prev,
          .group = ctx.device().CreateBindGroup(&descriptor),
          .lastUsed = frame,
      })
      .group;
}

template <typename T>
T read_params(const PassContext& ctx) noexcept {
  T params;
  const std::span<const std::byte> bytes = ctx.params();
  std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(T)));
  return params;
}

void encode_project(PassContext& ctx, const void*) {
  const ProjectParams params = read_params<ProjectParams>(ctx);
  const bool reset = (params.flags & ProjectReset) != 0 || !ctx.prev_valid(RadianceRef);
  wgpu::TextureView prev = reset ? wgpu::TextureView{} : ctx.prev(RadianceRef);
  if (!prev) {
    prev = dummy_prev(ctx.device());
  }
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::BindGroupLayout layout = prev_layout();
  if (!pipeline || !layout || !prev) {
    return;
  }
  const wgpu::BindGroup group = prev_group(ctx, layout, prev);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  constexpr uint32_t Groups = ProbeSize / ProjectGroup;
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(PrevGroupIndex, group);
  pass.DispatchWorkgroups(Groups, Groups);
  pass.End();
  bump((params.flags & ProjectObserve) != 0 ? g_observed : g_held);
  if (reset) {
    bump(g_resets);
  }
  if (hook().mode == HookMode::Constant) {
    bump(g_constant);
  }
}

void fill_project(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(ProjectParams)) {
    return;
  }
  ProjectParams params;
  fill_project_params(stage_hints(stage), sky_draws_now(), hook().mode == HookMode::Constant,
                      std::bit_cast<float>(g_alphaBits.load(std::memory_order_relaxed)), params);
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void declare_filter(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  // Declared in every mode, because binding numbers follow use order
  builder.use(Radiance, Access::Sampled, Missing::DisablePass);
  builder.create(Probe, probe_desc(), Access::StorageWrite);
  builder.create_buffer(Sh9, {.size = Sh9Bytes}, Access::StorageWrite);
}

void request_filter(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 4> Entries{{
      {"sky.probe_filter gain", "gain"},
      {"sky.probe_filter fill", "fill"},
      {"sky.probe_filter downsample", "downsample"},
      {"sky.probe_filter sh9", "sh9"},
  }};
  for (size_t i = 0; i < Entries.size(); ++i) {
    ids[i] = shaders::request_compute({
                                          .label = Entries[i].first,
                                          .shader = Shader,
                                          .entryPoint = Entries[i].second,
                                          .layout = setup.layout,
                                          .layoutKey = setup.layoutHash,
                                      })
                 .value;
  }
}

wgpu::BindGroup filter_group(const PassContext& ctx, const wgpu::BindGroupLayout& layout, uint32_t source,
                             uint32_t target, const BufferView& sh9) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .textureView = ctx.view(ProbeRef, source)},
      wgpu::BindGroupEntry{.binding = 1, .textureView = ctx.view(ProbeRef, target)},
      wgpu::BindGroupEntry{.binding = 2, .buffer = sh9.buffer, .offset = sh9.offset, .size = sh9.size},
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "sky.probe_filter level",
      .layout = layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return ctx.device().CreateBindGroup(&descriptor);
}

const FilterEntry& filter_groups(const PassContext& ctx, const wgpu::Texture& probe, const BufferView& sh9) {
  const wgpu::BindGroupLayout& layout = ctx.pass_layout();
  const uint64_t frame = ctx.frame_id();
  for (FilterEntry& entry : g_filterCache) {
    if (entry.probe.Get() == probe.Get() && entry.sh9.Get() == sh9.buffer.Get() && entry.offset == sh9.offset &&
        entry.layout.Get() == layout.Get()) {
      entry.lastUsed = frame;
      return entry;
    }
  }
  if (g_filterCache.size() >= CacheCapacity) {
    g_filterCache.erase(std::ranges::min_element(g_filterCache, {}, &FilterEntry::lastUsed));
  }
  FilterEntry& entry = g_filterCache.emplace_back(
      FilterEntry{.probe = probe, .sh9 = sh9.buffer, .offset = sh9.offset, .layout = layout, .lastUsed = frame});
  for (uint32_t level = 1; level < ProbeMips; ++level) {
    entry.downsample[level] = filter_group(ctx, layout, level - 1, level, sh9);
  }
  entry.sh = filter_group(ctx, layout, ShMip, 0, sh9);
  return entry;
}

void encode_filter(PassContext& ctx, const void*) {
  std::array<wgpu::ComputePipeline, 4> pipelines;
  for (uint32_t i = 0; i < pipelines.size(); ++i) {
    pipelines[i] = shaders::compute_pipeline(ctx.pipeline(i));
    if (!pipelines[i]) {
      return;
    }
  }
  const wgpu::Texture probe = ctx.texture(ProbeRef);
  const BufferView sh9 = ctx.buffer(Sh9Ref);
  if (!probe || probe.GetMipLevelCount() != ProbeMips || !sh9.buffer || sh9.size < Sh9Bytes) {
    return;
  }
  const FilterEntry& groups = filter_groups(ctx, probe, sh9);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  constexpr auto count = [](uint32_t size) { return (size + FilterGroup - 1) / FilterGroup; };
  pass.SetPipeline(pipelines[0]);
  pass.DispatchWorkgroups(1);
  pass.SetPipeline(pipelines[1]);
  pass.DispatchWorkgroups(count(ProbeSize), count(ProbeSize));
  pass.SetPipeline(pipelines[2]);
  for (uint32_t level = 1; level < ProbeMips; ++level) {
    pass.SetBindGroup(PassGroupIndex, groups.downsample[level]);
    pass.DispatchWorkgroups(count(ProbeSize >> level), count(ProbeSize >> level));
  }
  pass.SetPipeline(pipelines[3]);
  pass.SetBindGroup(PassGroupIndex, groups.sh);
  pass.DispatchWorkgroups(1);
  pass.End();
  bump(g_filtered);
}

void fill_filter(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(FilterParams)) {
    return;
  }
  const SceneHints* hints = stage_hints(stage);
  const Hook current = hook();
  FilterParams params;
  const bool rotationOk = fill_filter_params(hints, current, false, params);
  if (observing(hints, sky_draws_now(), current.mode == HookMode::Constant, rotationOk)) {
    params.flags |= FilterObserved;
  }
  std::memcpy(bytes.data(), &params, sizeof(params));
  const bool hasEnv = (params.flags & FilterHasEnv) != 0;
  const testing::Vec3 ambient{testing::srgb_to_linear(params.ambient[0]), testing::srgb_to_linear(params.ambient[1]),
                              testing::srgb_to_linear(params.ambient[2])};
  g_lastMode.store(static_cast<uint32_t>(testing::mode_for(current, hasEnv, ambient)), std::memory_order_relaxed);
}

void declare_view(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup) || !row_selected(setup.debug_view())) {
    builder.skip();
    return;
  }
  builder.after(phase::Lighting);
  builder.use(Probe, Access::Sampled, Missing::BindDummyBlack);
  builder.use(Radiance, Access::Sampled, Missing::BindDummyBlack);
  builder.use_buffer(Sh9, Access::Uniform, UseOptions{.missing = Missing::BindDummyBlack, .dummySize = Sh9Bytes});
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::Fog, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.create(ViewResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                 Access::StorageWrite);
}

void request_view(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "sky.probe_view",
                                        .shader = ViewShader,
                                        .entryPoint = "view",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

bool view_enabled(const StageData& stage, const void*) { return !stage.flushedAtFinish; }

void encode_view(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::Texture target = ctx.texture(ViewTarget);
  if (!pipeline || !target) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  constexpr auto count = [](uint32_t size) { return (size + ViewGroup - 1) / ViewGroup; };
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(count(target.GetWidth()), count(target.GetHeight()));
  pass.End();
  bump(g_viewDraws);
}

void fill_view(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(ViewParams)) {
    return;
  }
  const debug::GbufView* row = stage.frame != nullptr ? debug::find_gbuf_view(stage.frame->debugView) : nullptr;
  ViewParams params;
  params.mode = row != nullptr && row->name == MaskRow ? 1u : 0u;
  std::memcpy(bytes.data(), &params, sizeof(params));
}

using Vec3 = testing::Vec3;

double dot3(const Vec3& a, const Vec3& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec3 oct_point(double u, double v) noexcept {
  const double ex = u * 2.0 - 1.0;
  const double ez = v * 2.0 - 1.0;
  const double y = 1.0 - std::abs(ex) - std::abs(ez);
  double x = ex;
  double z = ez;
  if (y < 0.0) {
    x = (1.0 - std::abs(ez)) * (ex >= 0.0 ? 1.0 : -1.0);
    z = (1.0 - std::abs(ex)) * (ez >= 0.0 ? 1.0 : -1.0);
  }
  return {x, y, z};
}

constexpr double band_factor(size_t i) noexcept {
  if (i == 0) {
    return Pi;
  }
  return i < 4 ? 2.0 * Pi / 3.0 : Pi / 4.0;
}
} // namespace

std::optional<Hook> parse_hook(std::string_view text) noexcept {
  if (text == "off") {
    return Hook{.mode = HookMode::Off};
  }
  if (text == "physical") {
    return Hook{.mode = HookMode::Physical};
  }
  if (text == "global") {
    return Hook{.mode = HookMode::Global};
  }
  constexpr std::string_view Prefix = "constant:";
  if (!text.starts_with(Prefix)) {
    return std::nullopt;
  }
  text.remove_prefix(Prefix.size());
  float value = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
      value < 0.f || value > 65504.f) {
    return std::nullopt;
  }
  return Hook{.mode = HookMode::Constant, .constant = value};
}

Hook hook() noexcept { return g_hookOverride.value_or(env_hook()); }

Policy policy_of(const Hook& hook) noexcept {
  return hook.mode == HookMode::Physical ? Policy::Physical : Policy::MatchAmbient;
}

float blend_alpha(float deltaMs) noexcept {
  const float dt = std::isfinite(deltaMs) ? std::clamp(deltaMs, 0.f, MaxDeltaMs) / 1000.f : 0.f;
  return 1.f - std::exp(-dt / BlendTau);
}

void fill_project_params(const SceneHints* hints, uint32_t skyDraws, bool constant, float alpha,
                         ProjectParams& out) noexcept {
  out = {};
  fill_stage_camera(hints, out.camera);
  const bool rotationOk = fill_view_rotation(hints, out.rotation);
  out.flags = (observing(hints, skyDraws, constant, rotationOk) ? uint32_t{ProjectObserve} : 0u) |
              (hints == nullptr || !hints->motionValid ? uint32_t{ProjectReset} : 0u);
  out.alpha = alpha;
}

bool fill_filter_params(const SceneHints* hints, const Hook& hook, bool observed, FilterParams& out) noexcept {
  out = {};
  const bool rotationOk = fill_view_rotation(hints, out.rotation);
  const bool hasEnv = hints != nullptr && hints->has(SceneHasEnvironment);
  out.fog[3] = 1.f;
  if (hasEnv) {
    const AuroraEnvironment& env = hints->environment.value;
    for (size_t i = 0; i < 3; ++i) {
      out.fog[i] = unit(env.fogColor[i], 0.f);
      out.ambient[i] = unit(env.ambientColor[i], 0.f);
    }
    out.fog[3] = unit(env.skyVisibility, 0.f);
  }
  out.flags =
      (hook.mode == HookMode::Constant ? uint32_t{FilterConstant} : 0u) | (hasEnv ? uint32_t{FilterHasEnv} : 0u) |
      (hook.mode == HookMode::Physical ? uint32_t{FilterPhysical} : 0u) |
      (hook.mode == HookMode::Global ? uint32_t{FilterGlobal} : 0u) | (observed ? uint32_t{FilterObserved} : 0u);
  out.constant = hook.constant;
  return rotationOk;
}

bool wanted(const PassSetup& setup) noexcept {
  return lighting_wanted(setup) && setup.gx_available(GxResource::Fog) && hook().mode != HookMode::Off;
}

bool armed() noexcept { return g_armed.load(std::memory_order_relaxed); }

bool ready(const Settings& settings) noexcept {
  if (!settings.pbr) {
    return true;
  }
  const LatchSlot slot = latched_slot();
  return !slot.declared || (slot.project[0] != 0 && slot.filter[0] != 0 && shaders::pipelines_ready(slot.project) &&
                            shaders::pipelines_ready(slot.filter) && hints_enabled());
}

bool producing() noexcept {
  const LatchSlot slot = latched_slot();
  return !slot.declared || (slot.project[0] != 0 && slot.filter[0] != 0 && shaders::pipelines_ready(slot.project) &&
                            shaders::pipelines_ready(slot.filter));
}

PassDesc project_pass_desc() {
  return {
      .name = std::string{ProjectPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = ProjectOrder,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(ProjectParams),
      .declare = declare_project,
      .enabled = scene_enabled,
      .encode = encode_project,
      .pipelines = request_project,
      .stageParams = fill_project,
  };
}

PassDesc filter_pass_desc() {
  return {
      .name = std::string{FilterPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = FilterOrder,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(FilterParams),
      .declare = declare_filter,
      .enabled = scene_enabled,
      .encode = encode_filter,
      .pipelines = request_filter,
      .stageParams = fill_filter,
  };
}

PassDesc view_pass_desc() {
  return {
      .name = std::string{ViewPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Optional,
      .order = 2,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(ViewParams),
      .declare = declare_view,
      .enabled = view_enabled,
      .encode = encode_view,
      .pipelines = request_view,
      .stageParams = fill_view,
  };
}

void initialize() {
  if (g_project == InvalidPassId) {
    g_project = register_pass(project_pass_desc());
    g_filter = register_pass(filter_pass_desc());
    g_view = register_pass(view_pass_desc());
  }
}

void shutdown() noexcept {
  for (PassId* pass : {&g_project, &g_filter, &g_view}) {
    if (*pass != InvalidPassId) {
      unregister_pass(*pass);
      *pass = InvalidPassId;
    }
  }
  {
    std::lock_guard lock{g_slotMutex};
    g_slot = {};
  }
  g_armed.store(false, std::memory_order_relaxed);
  g_lastLatch.reset();
  g_prevCache.clear();
  g_filterCache.clear();
  g_dummyView = {};
  g_dummyTexture = {};
  {
    std::lock_guard lock{g_layoutMutex};
    g_pipelineLayouts.clear();
    g_prevLayout = {};
  }
  const testing::Counters c = testing::counters();
  constexpr std::string_view Line = "Sky probe: observed {} frames, held {}, reset {}, constant {}; last mode {}";
  if (c.observed != 0 || c.held != 0) {
    Log.info(Line, c.observed, c.held, c.resets, c.constant, g_lastMode.load(std::memory_order_relaxed));
  }
  for (std::atomic_uint32_t* counter :
       {&g_observed, &g_held, &g_resets, &g_filtered, &g_constant, &g_viewDraws, &g_lastMode}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void on_latch(const PassSetup& latched, const CompiledGraph* graph) {
  LatchSlot slot;
  if (graph != nullptr && node_live(*graph, ProjectPass) && node_live(*graph, FilterPass)) {
    slot.declared = true;
    slot.project = graph->nodes[graph->find_node(ProjectPass)].pipelineIds;
    slot.filter = graph->nodes[graph->find_node(FilterPass)].pipelineIds;
  }
  const Settings* settings = latched.settings;
  const bool classView = debug::find_view(latched.debug_view()) != nullptr;
  slot.armed = slot.declared && settings != nullptr && settings->pbr && settings->legacyBlend < 1.f &&
               node_live(*graph, deferred::Pass) && !classView;
  const float deltaMs =
      upscale::fsr::frame_delta_ms(g_lastLatch, std::chrono::steady_clock::now(), gfx::blocking_pipelines());
  g_alphaBits.store(std::bit_cast<uint32_t>(g_alphaOverride.value_or(blend_alpha(deltaMs))), std::memory_order_relaxed);
  std::lock_guard lock{g_slotMutex};
  g_slot = slot;
  g_armed.store(slot.armed, std::memory_order_relaxed);
}

void worker_tick(uint64_t frameId) noexcept {
  const auto stale = [frameId](const auto& entry) { return frameId > entry.lastUsed + RetainFrames; };
  std::erase_if(g_prevCache, stale);
  std::erase_if(g_filterCache, stale);
}

namespace testing {
Counters counters() noexcept {
  return {
      .observed = g_observed.load(std::memory_order_relaxed),
      .held = g_held.load(std::memory_order_relaxed),
      .resets = g_resets.load(std::memory_order_relaxed),
      .filtered = g_filtered.load(std::memory_order_relaxed),
      .constant = g_constant.load(std::memory_order_relaxed),
      .viewDraws = g_viewDraws.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_hintsOverride.reset();
  g_hookOverride.reset();
  g_skyDrawsOverride.reset();
  g_alphaOverride.reset();
  for (std::atomic_uint32_t* counter :
       {&g_observed, &g_held, &g_resets, &g_filtered, &g_constant, &g_viewDraws, &g_lastMode}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept { g_hintsOverride = std::move(hints); }

void set_hook(std::optional<Hook> hook) noexcept { g_hookOverride = hook; }

void set_sky_draws(std::optional<uint32_t> draws) noexcept { g_skyDrawsOverride = draws; }

void set_alpha(std::optional<float> alpha) noexcept { g_alphaOverride = alpha; }

Vec3 oct_decode(double u, double v) noexcept {
  const Vec3 p = oct_point(u, v);
  const double length = std::sqrt(dot3(p, p));
  return {p[0] / length, p[1] / length, p[2] / length};
}

std::array<double, 2> oct_encode(const Vec3& dir) noexcept {
  const double l1 = std::abs(dir[0]) + std::abs(dir[1]) + std::abs(dir[2]);
  if (!(l1 > 1e-20)) {
    return {0.5, 0.5};
  }
  double x = dir[0] / l1;
  double z = dir[2] / l1;
  if (dir[1] < 0.0) {
    const double fx = (1.0 - std::abs(z)) * (x >= 0.0 ? 1.0 : -1.0);
    const double fz = (1.0 - std::abs(x)) * (z >= 0.0 ? 1.0 : -1.0);
    x = fx;
    z = fz;
  }
  return {x * 0.5 + 0.5, z * 0.5 + 0.5};
}

double texel_weight(double u, double v, double size) noexcept {
  const Vec3 p = oct_point(u, v);
  const double length = std::sqrt(dot3(p, p));
  return 4.0 / (size * size * length * length * length);
}

std::array<double, 9> sh_basis(const Vec3& d) noexcept {
  return {0.28209479177,
          0.48860251190 * d[1],
          0.48860251190 * d[2],
          0.48860251190 * d[0],
          1.09254843059 * d[0] * d[1],
          1.09254843059 * d[1] * d[2],
          0.31539156525 * (3.0 * d[2] * d[2] - 1.0),
          1.09254843059 * d[0] * d[2],
          0.54627421529 * (d[0] * d[0] - d[1] * d[1])};
}

Sh sh_project(const std::vector<Vec3>& texels, uint32_t size, const Vec3& base) {
  Sh sh{};
  double total = 0.0;
  for (uint32_t y = 0; y < size; ++y) {
    for (uint32_t x = 0; x < size; ++x) {
      const double u = (x + 0.5) / size;
      const double v = (y + 0.5) / size;
      const double w = texel_weight(u, v, size);
      const std::array<double, 9> basis = sh_basis(oct_decode(u, v));
      const Vec3& l = texels[size_t{y} * size + x];
      for (size_t i = 0; i < 9; ++i) {
        for (size_t c = 0; c < 3; ++c) {
          sh[i][c] += w * basis[i] * (l[c] - base[c]);
        }
      }
      total += w;
    }
  }
  const double norm = 4.0 * Pi / total;
  for (size_t i = 0; i < 9; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      sh[i][c] *= norm * band_factor(i);
    }
  }
  for (size_t c = 0; c < 3; ++c) {
    sh[0][c] += Pi * std::sqrt(4.0 * Pi) * base[c];
  }
  return sh;
}

Vec3 sh_irradiance(const Sh& sh, const Vec3& n) noexcept {
  const std::array<double, 9> basis = sh_basis(n);
  Vec3 e{};
  for (size_t i = 0; i < 9; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      e[c] += sh[i][c] * basis[i];
    }
  }
  for (double& c : e) {
    c = std::max(c, 0.0);
  }
  return e;
}

std::array<double, 3> project_fwd(const StageCamera& camera, const Vec3& fwd, double width, double height) noexcept {
  if (fwd[2] <= 0.0) {
    return {0.0, 0.0, 0.0};
  }
  const double cx = camera.p02 - 2.0 * camera.jitterPx[0] / width;
  const double cy = camera.p12 + 2.0 * camera.jitterPx[1] / height;
  const double ndcX = fwd[0] * camera.p00 / fwd[2] - cx;
  const double ndcY = fwd[1] * camera.p11 / fwd[2] - cy;
  return {0.5 * (ndcX + 1.0), 0.5 * (1.0 - ndcY), 1.0};
}

Vec3 view_dir_fwd(const ViewRotation& r, const Vec3& w) noexcept {
  const auto row = [&](int i) { return r.rows[i][0] * w[0] + r.rows[i][1] * w[1] + r.rows[i][2] * w[2]; };
  return {row(0), row(1), -row(2)};
}

Vec3 world_dir(const ViewRotation& r, const Vec3& f) noexcept {
  Vec3 out{};
  for (int c = 0; c < 3; ++c) {
    out[c] = r.rows[0][c] * f[0] + r.rows[1][c] * f[1] - r.rows[2][c] * f[2];
  }
  return out;
}

double gain(Policy policy, bool hasEnv, const Vec3& linearAmbient, double upperMean) noexcept {
  if (policy != Policy::MatchAmbient || !hasEnv) {
    return 1.0;
  }
  return std::clamp(luminance(linearAmbient) / std::max(upperMean, 1e-6), double{GainMin}, double{GainMax});
}

Mode mode_for(const Hook& hook, bool hasEnv, const Vec3& linearAmbient) noexcept {
  if (hook.mode == HookMode::Constant || hook.mode == HookMode::Physical || hook.mode == HookMode::Global) {
    return Mode::Raw;
  }
  return hasEnv && luminance(linearAmbient) >= AmbientFloor ? Mode::Pixel : Mode::PixelFlat;
}

Vec3 pixel_scale(Mode mode, const Vec3& probeAmbient, const Vec3& a) noexcept {
  if (mode != Mode::Pixel) {
    return {1.0, 1.0, 1.0};
  }
  Vec3 out{};
  for (size_t c = 0; c < 3; ++c) {
    out[c] = std::clamp(a[c] / std::max(probeAmbient[c], double{AmbientFloor}), 0.0, double{ScaleMax});
  }
  return out;
}

Vec3 env_irradiance(Mode mode, const Sh& sh, const Vec3& probeAmbient, const Vec3& n, const Vec3& a,
                    bool baked) noexcept {
  if (mode == Mode::PixelFlat || (mode == Mode::Pixel && baked)) {
    return {Pi * a[0], Pi * a[1], Pi * a[2]};
  }
  const Vec3 s = pixel_scale(mode, probeAmbient, a);
  const Vec3 e = sh_irradiance(sh, n);
  return {s[0] * e[0], s[1] * e[1], s[2] * e[2]};
}

double luminance(const Vec3& rgb) noexcept { return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]; }

double srgb_to_linear(double stored) noexcept {
  const double s = std::clamp(stored, 0.0, 1.0);
  return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}
} // namespace testing
} // namespace aurora::render::detail::lighting::sky_probe

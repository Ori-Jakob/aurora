#include "exposure.hpp"

#include "clusters.hpp"
#include "deferred.hpp"

#include "../debug/views.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../post/composite.hpp"
#include "../post/upscale/fsr/fsr.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gfx/pipeline_cache.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::detail::lighting::exposure {
namespace {
constexpr Module Log{"aurora::render::lighting"};
constexpr TexRef StateRef{5};
constexpr uint32_t PrevGroupIndex = 2;
constexpr uint64_t LayoutSalt = 0x4558504F53555245ull;
constexpr uint64_t RetainFrames = 8;
constexpr size_t CacheCapacity = 4;

struct LatchSlot {
  bool declared = false;
  bool armed = false;
  std::array<uint32_t, MaxPassPipelines> ids{};
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

PassId g_pass = InvalidPassId;
std::mutex g_slotMutex;
LatchSlot g_slot;
std::atomic_bool g_armed{false};
bool g_wasArmed = false;
std::optional<std::chrono::steady_clock::time_point> g_lastLatch;
std::shared_ptr<const SceneHints> g_hintsOverride;
std::optional<float> g_deltaOverride;
std::optional<Target> g_targetOverride;

std::mutex g_layoutMutex;
wgpu::BindGroupLayout g_prevLayout;
std::vector<PipelineLayoutEntry> g_pipelineLayouts;
std::vector<PrevEntry> g_prevCache;
wgpu::Texture g_dummyTexture;
wgpu::TextureView g_dummyView;

std::atomic_uint32_t g_runs{0};
std::atomic_uint32_t g_snaps{0};
std::atomic_uint32_t g_manual{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

Target env_target() noexcept {
  static const Target value = [] {
    const char* text = SDL_getenv(std::string{TargetEnv}.c_str());
    if (text == nullptr) {
      return Target::Sb;
    }
    if (const std::optional<Target> parsed = parse_target(text)) {
      Log.warn("{}={}: overriding the exposure target", TargetEnv, text);
      return *parsed;
    }
    Log.warn("{}={} is not sb or s; ignored", TargetEnv, text);
    return Target::Sb;
  }();
  return value;
}

const SceneHints* stage_hints(const StageData& stage) noexcept {
  return g_hintsOverride != nullptr ? g_hintsOverride.get() : stage.sceneHints.get();
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

template <typename T>
T read_params(std::span<const std::byte> bytes) noexcept {
  T params;
  std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(T)));
  return params;
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Exposure);
  builder.use(resource::SceneHdr, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(composite::LegacyTex, Access::Sampled, Missing::DisablePass);
  builder.use(resource::Fog, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use(resource::AoFinal, Access::Sampled, Missing::BindDummyWhite);
  builder.use(resource::GxNormal, Access::Sampled, Missing::BindDummyBlack);
  builder.history(State, state_desc(), Access::StorageWrite);
  builder.create_buffer(Partials, {.size = PartialBytes}, Access::StorageWrite);
  builder.create_buffer(Resource, {.size = BufferBytes}, Access::StorageWrite);
}

wgpu::PipelineLayout pipeline_layout(const PipelineSetup& setup) {
  if (!webgpu::g_device || !setup.frameLayout || !setup.passLayout) {
    return {};
  }
  std::lock_guard lock{g_layoutMutex};
  if (!g_prevLayout) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .texture = {.sampleType = wgpu::TextureSampleType::Uint, .viewDimension = wgpu::TextureViewDimension::e2D},
    };
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "exposure.meter history",
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
      .label = "exposure.meter",
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

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  const wgpu::PipelineLayout layout = pipeline_layout(setup);
  if (!layout) {
    return;
  }
  constexpr std::array<std::pair<std::string_view, std::string_view>, 2> Entries{{
      {"exposure.meter sweep", "meter"},
      {"exposure.meter solve", "solve"},
  }};
  for (size_t i = 0; i < Entries.size(); ++i) {
    ids[i] = shaders::request_compute({
                                          .label = Entries[i].first,
                                          .shader = Shader,
                                          .entryPoint = Entries[i].second,
                                          .layout = layout,
                                          .layoutKey = setup.layoutHash ^ LayoutSalt,
                                      })
                 .value;
  }
}

// Zeroed at creation, so its flags read invalid and the solve snaps
const wgpu::TextureView& dummy_prev(const wgpu::Device& device) {
  if (!g_dummyView && device) {
    const wgpu::TextureDescriptor descriptor{
        .label = "exposure.state empty history",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {2, 1, 1},
        .format = StateFormat,
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
      .label = "exposure.meter history",
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

bool enabled(const StageData& stage, const void*) {
  // H exists exactly when light.deferred runs, and the meter has nothing to read without it
  return armed() && !stage.flushedAtFinish && deferred::frame_enabled(stage);
}

void encode(PassContext& ctx, const void*) {
  const Params params = read_params<Params>(ctx.params());
  const bool reset = (params.flags & FlagReset) != 0 || !ctx.prev_valid(StateRef);
  wgpu::TextureView prev = reset ? wgpu::TextureView{} : ctx.prev(StateRef);
  if (!prev) {
    prev = dummy_prev(ctx.device());
  }
  const wgpu::ComputePipeline meter = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::ComputePipeline solve = shaders::compute_pipeline(ctx.pipeline(1));
  const wgpu::BindGroupLayout layout = prev_layout();
  if (!meter || !solve || !layout || !prev) {
    return;
  }
  const wgpu::BindGroup group = prev_group(ctx, layout, prev);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetBindGroup(PrevGroupIndex, group);
  const bool manual = params.mode == static_cast<uint32_t>(Exposure::Manual);
  if (!manual) {
    pass.SetPipeline(meter);
    pass.DispatchWorkgroups(Groups);
  }
  pass.SetPipeline(solve);
  pass.DispatchWorkgroups(1);
  pass.End();
  bump(g_runs);
  if (reset || (params.flags & FlagSnap) != 0) {
    bump(g_snaps);
  }
  if (manual) {
    bump(g_manual);
  }
}

void fill_stage(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params = read_params<Params>(bytes);
  const SceneHints* hints = stage_hints(stage);
  params.envBias = params.mode == static_cast<uint32_t>(Exposure::MatchLegacy) ? 0.f : env_bias(hints);
  if (hints == nullptr || !hints->motionValid) {
    params.flags |= FlagReset;
  }
  std::memcpy(bytes.data(), &params, sizeof(params));
}
struct Window {
  float center = 0.f;
  float step = SnapStepEv;
  bool snap = true;
  float prevS = 0.f;
  bool history = false;
};

Window window_of(const Params& params, const std::array<uint32_t, 8>& prev) noexcept {
  const float prevS = std::bit_cast<float>(prev[0]);
  const bool valid = (prev[1] & StateValid) != 0 && std::abs(prevS) <= MaxEv;
  const bool modeChanged = ((prev[1] & StateModeMask) >> StateModeShift) != params.mode;
  const bool tonemapChanged = ((prev[1] & StateAgx) != 0) != ((params.flags & FlagAgx) != 0);
  const bool owed = (prev[1] & (StateSnapped | StateHeld)) == (StateSnapped | StateHeld);
  const bool snap = !valid || (params.flags & (FlagSnap | FlagReset)) != 0 || modeChanged || tonemapChanged || owed;
  constexpr float Edge = MaxEv - 15.5f * StepEv;
  const float s = valid ? prevS : 0.f;
  return {.center = snap ? 0.f : std::clamp(s, -Edge, Edge),
          .step = snap ? SnapStepEv : StepEv,
          .snap = snap,
          .prevS = s,
          .history = valid};
}
} // namespace

bool wanted(const PassSetup& setup) noexcept { return lighting_wanted(setup) && setup.gx_available(GxResource::Fog); }

bool armed() noexcept { return g_armed.load(std::memory_order_relaxed); }

bool pipelines_live() noexcept {
  const LatchSlot slot = latched_slot();
  return slot.declared && slot.ids[0] != 0 && slot.ids[1] != 0 && shaders::pipelines_ready(slot.ids);
}

bool ready(const Settings&) noexcept { return !latched_slot().declared || (pipelines_live() && hints_enabled()); }

std::optional<Target> parse_target(std::string_view text) noexcept {
  if (text == "sb") {
    return Target::Sb;
  }
  if (text == "s") {
    return Target::S;
  }
  return std::nullopt;
}

Target target() noexcept { return g_targetOverride.value_or(env_target()); }

float env_bias(const SceneHints* hints) noexcept {
  if (hints == nullptr || !hints->has(SceneHasEnvironment)) {
    return 0.f;
  }
  const float bias = hints->environment.value.exposureBias;
  return std::isfinite(bias) ? std::clamp(bias, -MaxEv, MaxEv) : 0.f;
}

void fill_params(const Settings& settings, Target target, float deltaMs, bool snap, Params& out) noexcept {
  out = {};
  out.mode = static_cast<uint32_t>(settings.exposure);
  out.flags = (settings.tonemap == Tonemap::AgX ? uint32_t{FlagAgx} : 0u) |
              (target == Target::S ? uint32_t{FlagTargetS} : 0u) | (snap ? uint32_t{FlagSnap} : 0u);
  const bool match = settings.exposure == Exposure::MatchLegacy;
  out.bias = !match && std::isfinite(settings.exposureBias) ? std::clamp(settings.exposureBias, -MaxEv, MaxEv) : 0.f;
  out.dt = (std::isfinite(deltaMs) ? std::clamp(deltaMs, 0.f, MaxDeltaMs) : 0.f) / 1000.f;
  out.anchor = ManualAnchorEv;
}

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_stage,
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
  {
    std::lock_guard lock{g_slotMutex};
    g_slot = {};
  }
  g_armed.store(false, std::memory_order_relaxed);
  g_wasArmed = false;
  g_lastLatch.reset();
  g_prevCache.clear();
  g_dummyView = {};
  g_dummyTexture = {};
  {
    std::lock_guard lock{g_layoutMutex};
    g_pipelineLayouts.clear();
    g_prevLayout = {};
  }
  const testing::Counters c = testing::counters();
  if (c.runs != 0) {
    Log.info("Exposure meter ran on {} frames ({} snapped, {} manual)", c.runs, c.snaps,
             g_manual.load(std::memory_order_relaxed));
  }
  for (std::atomic_uint32_t* counter : {&g_runs, &g_snaps, &g_manual}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void on_latch(const PassSetup& latched, const CompiledGraph* graph) {
  LatchSlot slot;
  if (graph != nullptr && node_live(*graph, Pass)) {
    slot.declared = true;
    slot.ids = graph->nodes[graph->find_node(Pass)].pipelineIds;
  }
  const Settings* settings = latched.settings;
  const bool classView = debug::find_view(latched.debug_view()) != nullptr;
  slot.armed = slot.declared && settings != nullptr && settings->pbr && settings->legacyBlend < 1.f && !classView;
  const float deltaMs = g_deltaOverride.value_or(
      upscale::fsr::frame_delta_ms(g_lastLatch, std::chrono::steady_clock::now(), gfx::blocking_pipelines()));
  if (g_pass != InvalidPassId && settings != nullptr) {
    Params params;
    // A meter that was off has no history worth smoothing from, so it snaps on its first frame back
    fill_params(*settings, target(), deltaMs, slot.armed && !g_wasArmed, params);
    set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
  }
  g_wasArmed = slot.armed;
  std::lock_guard lock{g_slotMutex};
  g_slot = slot;
  g_armed.store(slot.armed, std::memory_order_relaxed);
}

void worker_tick(uint64_t frameId) noexcept {
  std::erase_if(g_prevCache, [frameId](const PrevEntry& entry) { return frameId > entry.lastUsed + RetainFrames; });
}

namespace testing {
Counters counters() noexcept {
  return {
      .runs = g_runs.load(std::memory_order_relaxed),
      .snaps = g_snaps.load(std::memory_order_relaxed),
      .manual = g_manual.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_hintsOverride.reset();
  g_deltaOverride.reset();
  g_targetOverride.reset();
  for (std::atomic_uint32_t* counter : {&g_runs, &g_snaps, &g_manual}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept { g_hintsOverride = std::move(hints); }

void set_delta_ms(std::optional<float> deltaMs) noexcept { g_deltaOverride = deltaMs; }

void set_target(std::optional<Target> target) noexcept { g_targetOverride = target; }

float exp2_exact(float s) noexcept {
  const float whole = std::floor(s);
  const float f = s - whole;
  return std::ldexp(f == 0.f ? 1.f : std::exp2(f), static_cast<int>(whole));
}

float candidate(float center, float step, uint32_t k) noexcept {
  return center + (static_cast<float>(k) - 15.5f) * step;
}

bool snaps(const Params& params, const std::array<uint32_t, 8>& prev) noexcept { return window_of(params, prev).snap; }

float center(const Params& params, const std::array<uint32_t, 8>& prev) noexcept {
  return window_of(params, prev).center;
}

float smoothing_alpha(float dt, float tau) noexcept { return 1.f - std::exp(-dt / tau); }

Solved solve(const Params& params, std::span<const float> partials, const std::array<uint32_t, 8>& prev) noexcept {
  const Window w = window_of(params, prev);
  Solved out;
  out.previous = w.prevS;
  bool held = false;
  bool pinned = false;
  const bool manual = params.mode == static_cast<uint32_t>(Exposure::Manual);
  if (manual) {
    out.target = params.anchor + params.bias + params.envBias;
  } else {
    std::array<float, PartialWords> sums{};
    for (uint32_t word = 0; word < PartialWords; ++word) {
      for (uint32_t g = 0; g < Groups && size_t{g} * PartialWords + word < partials.size(); ++g) {
        sums[word] += partials[size_t{g} * PartialWords + word];
      }
    }
    out.weight = sums[WordWeight];
    out.count = sums[WordCount];
    held = out.count < HoldFraction * static_cast<float>(GridCells) || !(out.weight > 0.f);
    if (held && (w.history || !(out.weight > 0.f))) {
      out.target = w.prevS;
    } else {
      const bool match = params.mode == static_cast<uint32_t>(Exposure::MatchLegacy);
      out.t = match ? sums[WordTarget] / out.weight : std::log2(Key) + params.bias + params.envBias;
      const auto f = [&](uint32_t k) { return sums[k] / out.weight - out.t; };
      if (f(0) >= 0.f) {
        out.target = candidate(w.center, w.step, 0);
        pinned = f(0) > 0.f;
        out.fLo = out.fHi = f(0);
      } else if (f(Candidates - 1) <= 0.f) {
        out.target = candidate(w.center, w.step, Candidates - 1);
        pinned = f(Candidates - 1) < 0.f;
        out.fLo = out.fHi = f(Candidates - 1);
      } else {
        uint32_t k = 0;
        for (uint32_t i = 0; i + 1 < Candidates; ++i) {
          if (f(i + 1) > 0.f) {
            k = i;
            break;
          }
        }
        out.fLo = f(k);
        out.fHi = f(k + 1);
        out.target = candidate(w.center, w.step, k) + w.step * -out.fLo / (out.fHi - out.fLo);
      }
    }
  }
  float s = out.target;
  if (!manual && !w.snap && !held) {
    const bool match = params.mode == static_cast<uint32_t>(Exposure::MatchLegacy);
    const float tau = match ? MatchTau : out.target > w.prevS ? AutoUpTau : AutoDownTau;
    s = w.prevS + (out.target - w.prevS) * smoothing_alpha(params.dt, tau);
  }
  out.s = std::isnan(s) ? 0.f : std::clamp(s, -MaxEv, MaxEv);
  out.flags = StateValid | ((params.mode << StateModeShift) & StateModeMask) |
              ((params.flags & FlagAgx) != 0 ? uint32_t{StateAgx} : 0u) | (w.snap ? uint32_t{StateSnapped} : 0u) |
              (held ? uint32_t{StateHeld} : 0u) | (pinned ? uint32_t{StatePinned} : 0u) |
              ((params.flags & FlagTargetS) != 0 ? uint32_t{StateTargetS} : 0u);
  return out;
}

std::array<uint32_t, 8> texels(const Solved& state) noexcept {
  return {std::bit_cast<uint32_t>(state.s),      state.flags,
          std::bit_cast<uint32_t>(state.target), std::bit_cast<uint32_t>(state.weight),
          std::bit_cast<uint32_t>(state.t),      std::bit_cast<uint32_t>(state.fLo),
          std::bit_cast<uint32_t>(state.fHi),    std::bit_cast<uint32_t>(state.count)};
}

std::array<float, 4> words(const Solved& state) noexcept {
  const bool snapped = (state.flags & StateSnapped) != 0;
  return {exp2_exact(state.s), exp2_exact(-state.s), snapped ? 1.f : exp2_exact(state.s - state.previous), 1.f};
}
} // namespace testing
} // namespace aurora::render::detail::lighting::exposure

#include "ao.hpp"

#include "denoise.hpp"
#include "gtao.hpp"
#include "hiz.hpp"
#include "ssil_radiance.hpp"
#include "upsample.hpp"

#include "../debug/views.hpp"
#include "../hints.hpp"
#include "../post/composite.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gx/hints.hpp"
#include "../../internal.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace aurora::render::detail::ao {
namespace {
constexpr Module Log{"aurora::render::ao"};
constexpr std::array<std::string_view, 5> ChainPasses{hiz::Pass, gtao::Pass, upsample::Pass, denoise::Pass,
                                                      ssil_radiance::Pass};
static_assert(ChainPasses.size() <= MaxAoPasses);

struct LatchSlot {
  bool live = false;
  std::array<uint32_t, MaxPassPipelines * MaxAoPasses> ids{};
};

std::array<PassId, MaxAoPasses> g_passes{};
LatchSlot g_slot;
std::atomic_bool g_classView{false};
std::atomic<float> g_strength{1.f};
std::shared_ptr<const SceneHints> g_cameraOverride;
std::optional<bool> g_sinkOverride;
std::optional<float> g_giStart;
std::optional<SsilEnv> g_ssilEnvOverride;
std::optional<float> g_albedoOverride;
std::atomic_bool g_ssilLive{false};
uint32_t g_ssilEpoch = 0;

std::atomic_uint32_t g_skippedNoScene{0};
std::atomic_uint32_t g_cameraFills{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

std::optional<float> parse_float(std::string_view text) {
  float value = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<float> parse_metres(std::string_view name, const char* value) {
  const std::string_view text{value};
  const std::optional<float> metres = parse_float(text);
  if (metres && *metres > 0.f) {
    Log.warn("{}={}: overriding the AO default for verification", name, text);
    return metres;
  }
  Log.warn("{}={} is not a positive number of metres; ignored", name, text);
  return std::nullopt;
}

std::optional<float> parse_bounded(std::string_view name, const char* value, float high) {
  const std::string_view text{value};
  const std::optional<float> parsed = parse_float(text);
  if (parsed && *parsed >= 0.f && *parsed <= high) {
    Log.warn("{}={}: overriding the AO default for verification", name, text);
    return parsed;
  }
  Log.warn("{}={} is not a number from 0 to {}; ignored", name, text, high);
  return std::nullopt;
}

uint32_t parse_choice(std::string_view name, std::string_view text, std::string_view standard, std::string_view other,
                      uint32_t flag) {
  if (text == other) {
    Log.warn("{}={}: overriding the AO default for verification", name, text);
    return flag;
  }
  if (text != standard) {
    Log.warn("{}={} is not {} or {}; ignored", name, text, standard, other);
  }
  return 0;
}

bool parse_on(std::string_view name) {
  const char* value = SDL_getenv(std::string{name}.c_str());
  if (value == nullptr) {
    return false;
  }
  if (std::string_view{value} == "on") {
    Log.warn("{}=on: set for verification only", name);
    return true;
  }
  Log.warn("{}={} is not on; ignored", name, value);
  return false;
}

const EnvOverrides& env_overrides() noexcept {
  static const EnvOverrides env = [] {
    EnvOverrides out;
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_RADIUS")) {
      out.radius = parse_metres("AURORA_RENDER_AO_RADIUS", value);
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_THICKNESS")) {
      out.thickness = parse_metres("AURORA_RENDER_AO_THICKNESS", value);
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_DIVISOR")) {
      if (const std::optional<uint32_t> divisor = parse_divisor(value)) {
        Log.warn("AURORA_RENDER_AO_DIVISOR={}: forcing the AO working divisor for verification", value);
        out.divisor = *divisor;
      } else {
        Log.warn("AURORA_RENDER_AO_DIVISOR={} is not 1 or 2; ignored", value);
      }
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_STRENGTH")) {
      out.strength = parse_bounded("AURORA_RENDER_AO_STRENGTH", value, std::numeric_limits<float>::max());
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_FLOOR")) {
      out.floor = parse_bounded("AURORA_RENDER_AO_FLOOR", value, 1.f);
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_NORMALS")) {
      out.flags |= parse_choice("AURORA_RENDER_AO_NORMALS", value, "gbuffer", "depth", FlagNormalFromDepth);
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_SECTORS")) {
      out.flags |= parse_choice("AURORA_RENDER_AO_SECTORS", value, "round", "ceil", FlagSectorCeil);
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_AO_TEMPORAL")) {
      out.flags |= parse_choice("AURORA_RENDER_AO_TEMPORAL", value, "on", "off", FlagTemporalOff);
    }
    if (identity_forced()) {
      out.flags |= FlagIdentityForced;
    }
    return out;
  }();
  return env;
}

const SceneHints* stage_hints(const StageData& stage) noexcept {
  return g_cameraOverride != nullptr ? g_cameraOverride.get() : stage.sceneHints.get();
}

bool camera_usable(const SceneHints* hints) noexcept {
  return hints != nullptr && hints->has(SceneHasMainCamera) && gx::hints::main_camera_usable(hints->mainCamera.gxProj);
}

bool node_live(const CompiledGraph& graph, std::string_view name) noexcept {
  const NodeId node = graph.find_node(name);
  return node != InvalidNode && graph.nodes[node].kind == NodeKind::Pass &&
         graph.nodes[node].status == CullReason::None;
}

SsilEnv env_ssil() noexcept {
  static const SsilEnv request = [] {
    const char* value = SDL_getenv("AURORA_RENDER_SSIL");
    if (value == nullptr) {
      return SsilEnv::Unset;
    }
    const std::optional<SsilEnv> parsed = parse_ssil(value);
    if (!parsed) {
      Log.warn("AURORA_RENDER_SSIL={} is not on, off or furnace; ignored", value);
      return SsilEnv::Unset;
    }
    Log.warn("AURORA_RENDER_SSIL={}: overriding Indirect Light, for verification only", value);
    return *parsed;
  }();
  return request;
}

float env_albedo() noexcept {
  static const float albedo = [] {
    const char* value = SDL_getenv("AURORA_RENDER_SSIL_ALBEDO");
    if (value == nullptr) {
      return SsilLegacyAlbedo;
    }
    return parse_bounded("AURORA_RENDER_SSIL_ALBEDO", value, 1.f).value_or(SsilLegacyAlbedo);
  }();
  return albedo;
}
} // namespace

uint32_t working_divisor(Quality quality) noexcept { return working_divisor(quality, env_overrides().divisor); }

std::optional<Quality> parse_quality(std::string_view text) noexcept {
  constexpr std::array<std::string_view, 5> Names{"off", "low", "medium", "high", "ultra"};
  for (size_t i = 0; i < Names.size(); ++i) {
    if (text == Names[i]) {
      return static_cast<Quality>(i);
    }
  }
  return std::nullopt;
}

std::optional<uint32_t> parse_divisor(std::string_view text) noexcept {
  if (text == "1") {
    return 1u;
  }
  if (text == "2") {
    return 2u;
  }
  return std::nullopt;
}

std::optional<Quality> env_ao_request() noexcept {
  static const std::optional<Quality> request = []() -> std::optional<Quality> {
    const char* value = SDL_getenv("AURORA_RENDER_AO");
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::optional<Quality> quality = parse_quality(value);
    if (!quality) {
      Log.warn("AURORA_RENDER_AO={} is not off, low, medium, high or ultra; ignored", value);
    } else if (*quality == Quality::Off) {
      Log.warn("AURORA_RENDER_AO=off: keeping AO off regardless of settings");
    } else {
      Log.warn("AURORA_RENDER_AO={}: building and requesting AO regardless of settings, for verification only", value);
    }
    return quality;
  }();
  return request;
}

BuiltOptions with_dev_built(BuiltOptions built, std::optional<Quality> request) noexcept {
  if (request && *request != Quality::Off) {
    built[static_cast<size_t>(Feature::Ao)] |= BuiltOptionMask;
  }
  return built;
}

float gi_start_metres([[maybe_unused]] const Settings& settings) noexcept { return g_giStart.value_or(NoGiStart); }

Params latched_params(const Settings& settings, const EnvOverrides& env) noexcept {
  Settings tuning;
  tuning.aoRadius = env.radius.value_or(settings.aoRadius);
  tuning.aoStrength = env.strength.value_or(settings.aoStrength);
  tuning = sanitize_settings(tuning);
  const QualityRow row = quality_row(settings.ao);
  Params params;
  params.sliceCount = row.slices;
  params.stepsPerSlice = row.steps;
  params.flags = env.flags;
  params.strength = tuning.aoStrength;
  // SSIL comes out of this loop, so clamping AO to d0 also ends SSIL there
  params.radiusMetres = near_field(tuning.aoRadius, gi_start_metres(settings)).aoMetres;
  params.thicknessMetres = env.thickness.value_or(ThicknessMetres);
  params.minVisibility = env.floor.value_or(MinVisibility);
  return params;
}

bool identity_forced() noexcept {
  static const bool forced = parse_on("AURORA_RENDER_COMPOSITE_IDENTITY");
  return forced;
}

bool sink_forced() noexcept {
  static const bool forced = parse_on("AURORA_RENDER_AO_SINK");
  return g_sinkOverride.value_or(forced);
}

bool wanted(const PassSetup& setup) noexcept {
  return setup.settings != nullptr && setup.settings->ao != Quality::Off && setup.sceneDepth == DepthSource::Direct &&
         setup.gx_available(GxResource::Motion) && setup.gx_available(GxResource::Normal);
}

bool ssil_wanted(const PassSetup& setup) noexcept {
  // The composite writes the radiance SSIL gathers and is its only consumer without PBR
  return wanted(setup) && composite::declare_wanted(setup) && setup.settings->ssil;
}

bool ssil_live(const CompiledGraph& graph) noexcept {
  return node_live(graph, ssil_radiance::Pass) && node_live(graph, gtao::Pass);
}

bool ssil_ready(const Settings& settings) noexcept {
  return !settings.ssil || settings.ao == Quality::Off ||
         (g_ssilLive.load(std::memory_order_relaxed) && pipelines_live() && hints_enabled());
}

std::optional<SsilEnv> parse_ssil(std::string_view text) noexcept {
  if (text == "on") {
    return SsilEnv::On;
  }
  if (text == "off") {
    return SsilEnv::Off;
  }
  if (text == "furnace") {
    return SsilEnv::Furnace;
  }
  return std::nullopt;
}

SsilEnv env_ssil_request() noexcept { return g_ssilEnvOverride.value_or(env_ssil()); }

BuiltOptions with_ssil_dev_built(BuiltOptions built, SsilEnv request) noexcept {
  if (request == SsilEnv::On || request == SsilEnv::Furnace) {
    built[static_cast<size_t>(Feature::Ssil)] |= SsilBuiltOption;
  }
  return built;
}

bool ssil_furnace() noexcept { return env_ssil_request() == SsilEnv::Furnace; }

float legacy_albedo() noexcept { return g_albedoOverride.value_or(env_albedo()); }

bool scene_ready(const StageData& stage, const void*) noexcept {
  const bool ready =
      !stage.flushedAtFinish && camera_usable(stage_hints(stage)) && !g_classView.load(std::memory_order_relaxed);
  if (!ready) {
    bump(g_skippedNoScene);
  }
  return ready;
}

void fill_camera(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, bytes.data(), sizeof(Params));
  const bool valid = fill_stage_camera(stage_hints(stage), params.camera);
  const float upm = valid ? params.camera.unitsPerMeter : DefaultUnitsPerMetre;
  const float radius =
      std::isfinite(params.radiusMetres) && params.radiusMetres > 0.f ? params.radiusMetres : RadiusMetres;
  const float thickness =
      std::isfinite(params.thicknessMetres) && params.thicknessMetres > 0.f ? params.thicknessMetres : ThicknessMetres;
  params.effectRadius = radius * upm;
  params.thickness = thickness * upm;
  // The hint frame restarts with every replayed run, while the gfx frame index keeps counting
  const uint64_t noiseFrame = stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : stage.frameIndex;
  params.noiseIndex = static_cast<uint32_t>(noiseFrame % NoisePeriod);
  std::memcpy(bytes.data(), &params, sizeof(Params));
  bump(g_cameraFills);
}

void on_latch(const PassSetup& latched, const CompiledGraph* graph) {
  Params params = latched_params(*latched.settings, env_overrides());
  const bool ssil = graph != nullptr && ssil_live(*graph);
  const uint32_t epoch = composite::units_epoch();
  if (ssil && epoch != g_ssilEpoch) {
    params.flags |= FlagSsilReset;
  }
  g_ssilEpoch = epoch;
  g_ssilLive.store(ssil, std::memory_order_relaxed);
  for (const PassId pass : g_passes) {
    if (pass != InvalidPassId) {
      set_pass_params(pass, std::as_bytes(std::span{&params, 1}));
    }
  }
  g_strength.store(params.strength, std::memory_order_relaxed);
  g_classView.store(debug::find_view(latched.debug_view()) != nullptr, std::memory_order_relaxed);
  LatchSlot slot;
  if (graph != nullptr && node_live(*graph, hiz::Pass)) {
    slot.live = true;
    for (size_t i = 0; i < ChainPasses.size(); ++i) {
      if (node_live(*graph, ChainPasses[i])) {
        const auto& ids = graph->nodes[graph->find_node(ChainPasses[i])].pipelineIds;
        std::ranges::copy(ids, slot.ids.begin() + static_cast<std::ptrdiff_t>(i * MaxPassPipelines));
      }
    }
  }
  g_slot = slot;
}

bool pipelines_live() noexcept { return g_slot.live && g_slot.ids[0] != 0 && shaders::pipelines_ready(g_slot.ids); }

bool ready(const Settings& settings) noexcept {
  return settings.ao == Quality::Off || (pipelines_live() && hints_enabled());
}

bool motion_valid(const StageData& stage) noexcept {
  const SceneHints* hints = stage_hints(stage);
  return hints != nullptr && hints->motionValid;
}

float effective_strength() noexcept { return g_strength.load(std::memory_order_relaxed); }

void initialize() {
  if (g_passes[0] == InvalidPassId) {
    g_passes[0] = register_pass(hiz::pass_desc());
    g_passes[1] = register_pass(gtao::pass_desc());
    g_passes[2] = register_pass(upsample::pass_desc());
    g_passes[3] = register_pass(denoise::pass_desc());
    g_passes[4] = register_pass(ssil_radiance::pass_desc());
  }
}

void shutdown() noexcept {
  for (PassId& pass : g_passes) {
    if (pass != InvalidPassId) {
      unregister_pass(pass);
      pass = InvalidPassId;
    }
  }
  g_slot = {};
  g_classView.store(false, std::memory_order_relaxed);
  g_strength.store(1.f, std::memory_order_relaxed);
  g_ssilLive.store(false, std::memory_order_relaxed);
  hiz::clear_cache();
  ssil_radiance::clear_cache();
  denoise::shutdown();
}

void worker_tick(uint64_t frameId) noexcept {
  hiz::worker_tick(frameId);
  ssil_radiance::worker_tick(frameId);
  denoise::worker_tick(frameId);
}

namespace testing {
Counters counters() noexcept {
  return {
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
      .cameraFills = g_cameraFills.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cameraOverride.reset();
  g_sinkOverride.reset();
  g_giStart.reset();
  g_ssilEnvOverride.reset();
  g_albedoOverride.reset();
  g_ssilEpoch = composite::units_epoch();
  g_skippedNoScene.store(0, std::memory_order_relaxed);
  g_cameraFills.store(0, std::memory_order_relaxed);
  hiz::testing::reset();
  gtao::testing::reset();
  upsample::testing::reset();
  denoise::testing::reset();
  ssil_radiance::testing::reset();
}

void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept { g_cameraOverride = std::move(hints); }

void set_sink_override(std::optional<bool> sink) noexcept { g_sinkOverride = sink; }

void set_gi_start(std::optional<float> metres) noexcept { g_giStart = metres; }

void set_ssil_env(std::optional<SsilEnv> env) noexcept { g_ssilEnvOverride = env; }

void set_legacy_albedo(std::optional<float> albedo) noexcept { g_albedoOverride = albedo; }

void unregister(std::string_view pass) noexcept {
  for (size_t i = 0; i < ChainPasses.size(); ++i) {
    if (ChainPasses[i] == pass && g_passes[i] != InvalidPassId) {
      unregister_pass(g_passes[i]);
      g_passes[i] = InvalidPassId;
    }
  }
}
} // namespace testing
} // namespace aurora::render::detail::ao

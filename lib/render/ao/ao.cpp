#include "ao.hpp"

#include "hiz.hpp"

#include "../debug/views.hpp"
#include "../hints.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gx/hints.hpp"
#include "../../internal.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <atomic>
#include <charconv>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

namespace aurora::render::detail::ao {
namespace {
constexpr Module Log{"aurora::render::ao"};

struct EnvOverrides {
  std::optional<float> radius;
  std::optional<float> thickness;
  uint32_t divisor = 0;
};

struct LatchSlot {
  bool live = false;
  std::array<uint32_t, MaxPassPipelines> ids{};
};

std::array<PassId, MaxAoPasses> g_passes{};
LatchSlot g_slot;
std::atomic_bool g_classView{false};
std::shared_ptr<const SceneHints> g_cameraOverride;

std::atomic_uint32_t g_skippedNoScene{0};
std::atomic_uint32_t g_cameraFills{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

std::optional<float> parse_metres(std::string_view name, const char* value) {
  const std::string_view text{value};
  float metres = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), metres);
  if (error == std::errc{} && end == text.data() + text.size() && std::isfinite(metres) && metres > 0.f) {
    Log.warn("{}={}: overriding the AO default for verification", name, text);
    return metres;
  }
  Log.warn("{}={} is not a positive number of metres; ignored", name, text);
  return std::nullopt;
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

Params latched_params(const Settings& settings) noexcept {
  const EnvOverrides& env = env_overrides();
  const QualityRow row = quality_row(settings.ao);
  Params params;
  params.sliceCount = row.slices;
  params.stepsPerSlice = row.steps;
  params.radiusMetres = env.radius.value_or(RadiusMetres);
  params.thicknessMetres = env.thickness.value_or(ThicknessMetres);
  return params;
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

bool wanted(const PassSetup& setup) noexcept {
  return setup.settings != nullptr && setup.settings->ao != Quality::Off && setup.sceneDepth == DepthSource::Direct &&
         setup.gx_available(GxResource::Motion) && setup.gx_available(GxResource::Normal);
}

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
  params.noiseIndex = stage.frameIndex % NoisePeriod;
  std::memcpy(bytes.data(), &params, sizeof(Params));
  bump(g_cameraFills);
}

void on_latch(const Settings& settings, const CompiledGraph* graph) {
  const Params params = latched_params(settings);
  for (const PassId pass : g_passes) {
    if (pass != InvalidPassId) {
      set_pass_params(pass, std::as_bytes(std::span{&params, 1}));
    }
  }
  g_classView.store(debug::find_view(debug::selected_view_id(settings)) != nullptr, std::memory_order_relaxed);
  LatchSlot slot;
  if (graph != nullptr) {
    const NodeId node = graph->find_node(hiz::Pass);
    if (node != InvalidNode && graph->nodes[node].kind == NodeKind::Pass &&
        graph->nodes[node].status == CullReason::None) {
      slot.live = true;
      slot.ids = graph->nodes[node].pipelineIds;
    }
  }
  g_slot = slot;
}

bool pipelines_live() noexcept { return g_slot.live && g_slot.ids[0] != 0 && shaders::pipelines_ready(g_slot.ids); }

void initialize() {
  if (g_passes[0] == InvalidPassId) {
    g_passes[0] = register_pass(hiz::pass_desc());
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
  hiz::clear_cache();
}

void worker_tick(uint64_t frameId) noexcept { hiz::worker_tick(frameId); }

namespace testing {
Counters counters() noexcept {
  return {
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
      .cameraFills = g_cameraFills.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cameraOverride.reset();
  g_skippedNoScene.store(0, std::memory_order_relaxed);
  g_cameraFills.store(0, std::memory_order_relaxed);
  hiz::testing::reset();
}

void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept { g_cameraOverride = std::move(hints); }
} // namespace testing
} // namespace aurora::render::detail::ao

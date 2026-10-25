#include "hints.hpp"

#include "scene/classify.hpp"
#include "scene/motion.hpp"
#include "../gfx/recording.hpp"
#include "../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <string_view>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render::hints"};
constexpr int NoMarker = -1;

// -1 leaves the decision to AURORA_RENDER_HINTS
int g_override = -1;
bool g_installed = false;

// FIFO thread during the frame; game thread in begin_hint_frame and gfx::finish, while the FIFO is idle
SceneHints g_live;
bool g_frozen = false;
int g_lastMarker = NoMarker;
bool g_epochSeen = false;
uint32_t g_epochBaseline = 0;
std::shared_ptr<const SceneHints> g_snapshot;

bool decide_enabled() noexcept {
  if (g_override >= 0) {
    return g_override != 0;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_HINTS")) {
    return std::string_view{value} != "0";
  }
  return true;
}

void clear_frame() noexcept {
  g_snapshot.reset();
  g_live.present = 0;
  g_live.mainCamera = {};
  g_live.lights.clear();
  g_live.environment = {};
  g_live.influencers.clear();
  g_live.windSources.clear();
  g_live.sceneEpoch = 0;
  g_live.sceneEpochChanged = false;
  g_live.motionValid = false;
  g_live.motionReason = MotionReason::NoCamera;
  g_live.jitterPx[0] = 0.f;
  g_live.jitterPx[1] = 0.f;
  g_frozen = false;
  g_lastMarker = NoMarker;
}

void on_reset(void*) noexcept {
  clear_frame();
  g_live.hintFrame = 0;
  g_epochSeen = false;
  g_epochBaseline = 0;
  scene::reset();
  scene::motion::reset();
}

void on_camera(void*, const gx::hints::Camera& camera) noexcept {
  scene::on_camera(camera);
  if (camera.role != AURORA_CAMERA_MAIN) {
    return;
  }
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_CAMERA);
    return;
  }
  g_live.mainCamera = camera;
  const bool usable = gx::hints::main_camera_usable(camera.gxProj);
  // The last MAIN camera before the freeze decides, so an unusable one clears an earlier usable one
  if (usable) {
    g_live.present |= SceneHasMainCamera;
  } else {
    g_live.present &= ~SceneHasMainCamera;
  }
  const auto v = scene::motion::on_main_camera(camera, usable);
  g_live.motionValid = v.valid;
  g_live.motionReason = v.reason;
  g_live.jitterPx[0] = v.jitterPx[0];
  g_live.jitterPx[1] = v.jitterPx[1];
}

void on_object_begin(void*, const gx::hints::ObjectBegin& object) noexcept { scene::on_object_begin(object); }

void on_object_end(void*) noexcept { scene::on_object_end(); }

void on_scope_push(void*, const gx::hints::ScopePush& scope) noexcept { scene::on_scope_push(scope); }

void on_scope_pop(void*) noexcept { scene::on_scope_pop(); }

void on_stage(void*, Stage stage) noexcept {
  // Mirrors record_stage, which refuses a marker that does not come after every marker it accepted this frame
  if (const int index = static_cast<int>(stage); index > g_lastMarker) {
    g_lastMarker = index;
    if (stage == Stage::SceneBegin) {
      g_frozen = true;
    }
  }
  scene::on_stage(stage);
  gfx::record_stage(stage);
}

void on_lights(void*, std::span<const AuroraLight> lights) noexcept {
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_LIGHTS);
    return;
  }
  g_live.lights.assign(lights.begin(), lights.end());
  g_live.present |= SceneHasLights;
}

void on_environment(void*, const gx::hints::Environment& environment) noexcept {
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_ENVIRONMENT);
    return;
  }
  g_live.environment = environment;
  g_live.present |= SceneHasEnvironment;
}

void on_material(void*, uint32_t materialClass) noexcept { scene::on_material(materialClass); }

void on_skeleton(void*, const gx::hints::Skeleton&) noexcept {}

void on_scene_epoch(void*, uint32_t epoch) noexcept {
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_SCENE_EPOCH);
    return;
  }
  g_live.sceneEpoch = epoch;
  g_live.sceneEpochChanged = g_epochSeen && epoch != g_epochBaseline;
  g_live.present |= SceneHasEpoch;
  g_epochSeen = true;
  g_epochBaseline = epoch;
}

void on_tev_policy(void*, const AuroraTevPolicy& policy) noexcept { scene::on_tev_policy(policy); }

void on_influencers(void*, std::span<const AuroraInfluencer> influencers) noexcept {
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_INFLUENCERS);
    return;
  }
  g_live.influencers.assign(influencers.begin(), influencers.end());
  g_live.present |= SceneHasInfluencers;
}

void on_wind_sources(void*, std::span<const AuroraWindSource> sources) noexcept {
  if (g_frozen) {
    gx::hints::count_late(GX_AURORA_RENDER_WIND_SOURCES);
    return;
  }
  g_live.windSources.assign(sources.begin(), sources.end());
  g_live.present |= SceneHasWindSources;
}

constexpr gx::hints::Sink g_sink{
    .reset = &on_reset,
    .camera = &on_camera,
    .objectBegin = &on_object_begin,
    .objectEnd = &on_object_end,
    .scopePush = &on_scope_push,
    .scopePop = &on_scope_pop,
    .stage = &on_stage,
    .lights = &on_lights,
    .environment = &on_environment,
    .material = &on_material,
    .skeleton = &on_skeleton,
    .sceneEpoch = &on_scene_epoch,
    .tevPolicy = &on_tev_policy,
    .influencers = &on_influencers,
    .windSources = &on_wind_sources,
};
} // namespace

bool hints_enabled() noexcept { return g_installed; }

void install_hints() noexcept {
  if (!decide_enabled()) {
    Log.info("Render hints: decoding disabled by AURORA_RENDER_HINTS=0");
    return;
  }
  gx::hints::set_sink(&g_sink);
  g_installed = true;
  scene::install();
}

void remove_hints() noexcept {
  scene::remove();
  if (gx::hints::sink() == &g_sink) {
    gx::hints::set_sink(nullptr);
  }
  g_installed = false;
  on_reset(nullptr);
}

void begin_hint_frame() noexcept {
  scene::begin_frame();
  clear_frame();
  ++g_live.hintFrame;
}

std::shared_ptr<const SceneHints> scene_hints() {
  if (!g_snapshot) {
    g_frozen = true;
    g_snapshot = std::make_shared<const SceneHints>(g_live);
  }
  return g_snapshot;
}

const SceneHints& live_scene_hints() noexcept { return g_live; }

bool scene_hints_frozen() noexcept { return g_frozen; }

namespace testing {
void set_hints_override(int value) noexcept { g_override = value; }

void reset_hints() noexcept {
  remove_hints();
  g_override = -1;
}
} // namespace testing
} // namespace aurora::render::detail

#include "classify.hpp"

#include "../fakeshadow/texture_gate.hpp"
#include "../frame_state.hpp"
#include "../gx_ext/records.hpp"
#include "../hints.hpp"
#include "../../gx/draw_route.hpp"
#include "../../gx/gx.hpp"
#include "../../logging.hpp"

#include <aurora/gfx.hpp>
#include <SDL3/SDL_stdinc.h>

#include <array>
#include <cstring>
#include <mutex>
#include <string_view>
#include <utility>

namespace aurora::render::scene {
namespace {
constexpr Module Log{"aurora::render::scene"};

static_assert(static_cast<uint8_t>(Role::Main) == AURORA_CAMERA_MAIN);
static_assert(static_cast<uint8_t>(Role::Sky) == AURORA_CAMERA_SKY);
static_assert(static_cast<uint8_t>(Role::Reflection) == AURORA_CAMERA_REFLECTION);
static_assert(static_cast<uint8_t>(Role::Shadow) == AURORA_CAMERA_SHADOW);
static_assert(static_cast<uint8_t>(Role::Offscreen) == AURORA_CAMERA_OFFSCREEN);
static_assert(static_cast<uint8_t>(Role::Ui) == AURORA_CAMERA_UI);
static_assert(InheritRole == AURORA_CAMERA_INHERIT);

static_assert(category::Static == AURORA_CAT_STATIC && category::Dynamic == AURORA_CAT_DYNAMIC &&
              category::Skinned == AURORA_CAT_SKINNED && category::Terrain == AURORA_CAT_TERRAIN &&
              category::Foliage == AURORA_CAT_FOLIAGE && category::Water == AURORA_CAT_WATER &&
              category::Sky == AURORA_CAT_SKY && category::Particle == AURORA_CAT_PARTICLE &&
              category::Decal == AURORA_CAT_DECAL && category::Ui == AURORA_CAT_UI &&
              category::Player == AURORA_CAT_PLAYER && category::Emissive == AURORA_CAT_EMISSIVE &&
              category::Overlay == AURORA_CAT_OVERLAY && category::NoShadowCast == AURORA_CAT_NO_SHADOW_CAST &&
              category::NoShadowReceive == AURORA_CAT_NO_SHADOW_RECEIVE && category::NoPbr == AURORA_CAT_NO_PBR &&
              category::NoCapture == AURORA_CAT_NO_CAPTURE && category::NoRt == AURORA_CAT_NO_RT &&
              category::ScreenRead == AURORA_CAT_SCREEN_READ && category::FakeShadow == AURORA_CAT_FAKE_SHADOW &&
              category::CloudReceiver == AURORA_CAT_CLOUD_RECEIVER && category::NoAo == AURORA_CAT_NO_AO);
static_assert(object_flag::CaptureOnly == AURORA_OBJ_CAPTURE_ONLY && object_flag::Teleport == AURORA_OBJ_TELEPORT &&
              object_flag::CpuDeformed == AURORA_OBJ_CPU_DEFORMED &&
              object_flag::BakedVertexLight == AURORA_OBJ_BAKED_VERTEX_LIGHT &&
              object_flag::DoubleSided == AURORA_OBJ_DOUBLE_SIDED && object_flag::Billboard == AURORA_OBJ_BILLBOARD &&
              object_flag::Immobile == AURORA_OBJ_IMMOBILE);
static_assert(sizeof(gx::GXState::proj) == sizeof(gx::hints::Camera::gxProj));

// FIFO thread during the frame; game thread while the FIFO is idle
Classifier g_classifier;
bool g_reserved = false;
bool g_sealed = false;
// 0 collects only on demand, 1 always with the full key, 2 always with the object-only key
int g_env = 0;
// -1 leaves the decision to AURORA_RENDER_IDENTITY
int g_override = -1;
// Raised while a debug view colours draws by identity
bool g_demand = false;
ReportRequest g_request;
const ViewHooks* g_hooks = nullptr;
const ObserverHooks* g_observer = nullptr;

std::mutex g_reportMutex;
std::shared_ptr<const FrameReport> g_report;

struct CountWarning {
  uint32_t FrameStats::* stat = nullptr;
  uint32_t HintCounts::* hint = nullptr;
  std::string_view text;
};

constexpr std::array<CountWarning, 9> CountWarnings{{
    {.hint = &HintCounts::staleMarkers, .text = "late or repeated STAGE markers this frame"},
    {.hint = &HintCounts::scopeOverflow, .text = "SCOPE_PUSHes past 16 open scopes this frame"},
    {.hint = &HintCounts::scopeUnderflow, .text = "SCOPE_POPs with no open scope this frame"},
    {.hint = &HintCounts::nestedObjects, .text = "nested OBJECT_BEGINs this frame"},
    {.hint = &HintCounts::objectUnderflow, .text = "OBJECT_ENDs with no open object this frame"},
    {.hint = &HintCounts::openScopes, .text = "scopes still open at frame end"},
    {.hint = &HintCounts::openObjects, .text = "objects still open at frame end"},
    {.stat = &FrameStats::droppedRecords, .text = "draw records past the record buffer this frame"},
    {.stat = &FrameStats::droppedObjects, .text = "objects past the object table this frame"},
}};
std::array<bool, CountWarnings.size()> g_warned{};

void warn_counts(const FrameStats& stats) noexcept {
  for (size_t i = 0; i < CountWarnings.size(); ++i) {
    const CountWarning& warning = CountWarnings[i];
    const uint32_t value = warning.hint != nullptr ? stats.hints.*warning.hint : stats.*warning.stat;
    if (value != 0 && !std::exchange(g_warned[i], true)) {
      Log.warn("Render hints: {} {}", value, warning.text);
    }
  }
}

int read_identity_env() noexcept {
  const char* value = SDL_getenv("AURORA_RENDER_IDENTITY");
  if (value == nullptr) {
    return 0;
  }
  const std::string_view text{value};
  if (text == "1" || text == "2") {
    Log.info("Render hints: identity collection on by AURORA_RENDER_IDENTITY={} ({} merge key)", text,
             text == "1" ? "full" : "object-only");
    return text == "1" ? 1 : 2;
  }
  if (text != "0") {
    Log.warn("Render hints: AURORA_RENDER_IDENTITY={} is not 0, 1 or 2; ignored", text);
  }
  return 0;
}

void apply_merge_key() noexcept {
  // A view needs per-draw identity, so it always gets the full key
  g_classifier.set_merge_key(g_env == 2 && !g_demand ? MergeKey::ObjectOnly : MergeKey::Full);
}

gx::draw_route::Decision route(void*, const gx::draw_route::Query& query) noexcept {
  const detail::SceneHints& hints = detail::live_scene_hints();
  const gx::GXState& state = gx::g_gxState;
  const bool mainCamera = hints.has(detail::SceneHasMainCamera);
  // projType goes stale after a full projection load, so perspective is read from the matrix
  const DrawInputs inputs{
      .mainCamera = mainCamera,
      .projectionMatch = mainCamera && std::memcmp(&state.proj, &hints.mainCamera.gxProj, sizeof(state.proj)) == 0,
      .perspective = state.proj[3][3] == 0.f,
      .offscreen = gfx::is_offscreen(),
      .depthWrite = state.depthCompare && state.depthUpdate,
      .blended = blended(state.blendMode, state.blendFacSrc, state.blendFacDst, state.blendOp),
      .colorWrite = state.colorUpdate,
      .indexed = query.indexed,
      .vertices = query.vtxCount,
  };
  const DrawClass& cls = g_classifier.route(inputs);
  if (cls.suppressed) {
    return {.route = gx::draw_route::Route::Suppress};
  }
  // Each command carries one draw record, so a changed class must start its own command
  const bool records = gx_ext::records::on_route(cls);
  // A merged follower skips the push hook, so a changed class after a flagged lead must start its own command
  const bool breakMerge = cls.keyChanged && (identity_collecting() || records || fakeshadow::lead_flagged());
  if (breakMerge) {
    g_classifier.count_merge_break();
  }
  return {.route = gx::draw_route::Route::Push, .breakMerge = breakMerge};
}

// The router ran first for this same draw, so the classifier's last draw is the one being pushed
bool push(void*, gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info) noexcept {
  const DrawClass& cls = g_classifier.last();
  const fakeshadow::Edit gate = fakeshadow::push(draw, config, info, cls);
  // An empty draw has nothing to recolour
  const bool changed = gate != fakeshadow::Edit::Skipped && g_hooks != nullptr && g_hooks->push != nullptr &&
                       g_hooks->push(draw, config, info, cls);
  // Before the inspector's observer, which reads the record locator
  gx_ext::records::stamp(draw, cls);
  // Last, so the inspector records the draw that is actually pushed
  if (g_observer != nullptr && g_observer->observe != nullptr) {
    g_observer->observe(draw, config, info, cls);
  }
  return changed || gate != fakeshadow::Edit::Untouched;
}

constexpr gx::draw_route::Router g_router{.route = &route, .push = &push, .userData = nullptr};
} // namespace

void install() noexcept {
  if (!std::exchange(g_reserved, true)) {
    g_classifier.reserve();
  }
  g_env = read_identity_env();
  fakeshadow::install();
  apply_merge_key();
  gx::draw_route::set_router(&g_router);
}

void remove() noexcept {
  if (gx::draw_route::router() == &g_router) {
    gx::draw_route::set_router(nullptr);
  }
  fakeshadow::remove();
  reset();
}

void reset() noexcept {
  g_classifier.reset();
  fakeshadow::reset_frame();
  g_sealed = false;
  std::shared_ptr<const FrameReport> previous;
  std::lock_guard lock{g_reportMutex};
  g_report.swap(previous);
}

void begin_frame() noexcept {
  // Seals before the receiver clears its live hints, so the report still sees this frame's camera and epoch
  seal_frame();
  g_classifier.begin_frame();
  g_sealed = false;
}

void apply_frame_state(const detail::FrameState& state) noexcept {
  // First, so the identity demand the view sets decides this frame's collection and merge key
  if (g_hooks != nullptr && g_hooks->beginFrame != nullptr) {
    g_hooks->beginFrame(state);
  }
  fakeshadow::begin_frame(state);
  g_classifier.set_record(state.inspectorOpen || g_request.records);
  g_classifier.set_objects(state.inspectorOpen || g_request.objects);
  apply_merge_key();
}

void seal_frame() noexcept {
  if (std::exchange(g_sealed, true)) {
    return;
  }
  g_classifier.end_frame();
  fakeshadow::seal();
  const FrameStats& stats = g_classifier.stats();
  warn_counts(stats);
  const detail::SceneHints& hints = detail::live_scene_hints();
  auto report = std::make_shared<FrameReport>();
  report->stats = stats;
  report->mainCamera = hints.has(detail::SceneHasMainCamera);
  report->cutEpoch = hints.mainCamera.cutEpoch;
  report->sceneEpochKnown = hints.has(detail::SceneHasEpoch);
  report->sceneEpoch = hints.sceneEpoch;
  const auto draws = g_classifier.draws();
  report->draws.assign(draws.begin(), draws.end());
  const auto objects = g_classifier.objects();
  report->objects.assign(objects.begin(), objects.end());
  // The lock goes before the swapped-out report does, so readers never wait on freeing it
  std::shared_ptr<const FrameReport> published = std::move(report);
  std::lock_guard lock{g_reportMutex};
  g_report.swap(published);
}

void request_report(const ReportRequest& request) noexcept { g_request = request; }

void set_view_hooks(const ViewHooks* hooks) noexcept { g_hooks = hooks; }

void set_observer_hooks(const ObserverHooks* hooks) noexcept { g_observer = hooks; }

void set_identity_demand(bool demand) noexcept { g_demand = demand; }

bool identity_collecting() noexcept {
  if (g_override >= 0) {
    return g_override != 0;
  }
  return g_env == 1 || g_env == 2 || g_demand;
}

ViewCounts& view_counts() noexcept { return g_classifier.stats().view; }

void on_camera(const gx::hints::Camera& camera) noexcept { g_classifier.camera(static_cast<Role>(camera.role)); }

void on_object_begin(const gx::hints::ObjectBegin& object) noexcept {
  g_classifier.object_begin({
      .id = object.id,
      .subId = object.subId,
      .generation = object.generation,
      .categories = object.categories,
      .flags = object.flags,
  });
}

void on_object_end() noexcept { g_classifier.object_end(); }

void on_scope_push(const gx::hints::ScopePush& scope) noexcept {
  g_classifier.scope_push({
      .categories = scope.categories,
      .flags = scope.flags,
      .role = static_cast<uint8_t>(scope.role),
  });
}

void on_scope_pop() noexcept { g_classifier.scope_pop(); }

void on_stage(Stage stage) noexcept { g_classifier.stage(stage); }

void on_material(uint32_t materialClass) noexcept { g_classifier.material(materialClass); }

void on_tev_policy(const AuroraTevPolicy& policy) noexcept { g_classifier.tev_policy(policy.mask); }

std::shared_ptr<const FrameReport> latest_report() noexcept {
  std::lock_guard lock{g_reportMutex};
  return g_report;
}

bool blended(GXBlendMode mode, GXBlendFactor src, GXBlendFactor dst, GXLogicOp op) noexcept {
  switch (mode) {
  case GX_BM_BLEND:
    return src != GX_BL_ONE || dst != GX_BL_ZERO;
  case GX_BM_LOGIC:
    return op != GX_LO_COPY;
  case GX_BM_SUBTRACT:
    return true;
  default:
    return false;
  }
}

namespace testing {
Classifier& classifier() noexcept { return g_classifier; }

void set_identity_override(int value) noexcept { g_override = value; }

void set_identity_env(int value) noexcept {
  g_env = value;
  apply_merge_key();
}

void reset() noexcept {
  g_override = -1;
  g_env = 0;
  g_demand = false;
  g_request = {};
  g_hooks = nullptr;
  g_observer = nullptr;
  fakeshadow::testing::reset();
  scene::reset();
}
} // namespace testing
} // namespace aurora::render::scene

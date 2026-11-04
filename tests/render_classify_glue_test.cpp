#include <gtest/gtest.h>

#include "gfx/recording.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "hint_bytes.hpp"
#include "render/ao/denoise.hpp"
#include "render/ao/gtao.hpp"
#include "render/ao/hiz.hpp"
#include "render/ao/upsample.hpp"
#include "render/frame_state.hpp"
#include "render/geometry/hzb.hpp"
#include "render/graph.hpp"
#include "render/gx_ext/fwd_env.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/post/composite.hpp"
#include "render/post/taa.hpp"
#include "render/post/upscale/fsr/fsr_pass.hpp"
#include "render/scene/classify.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/overlay.hpp"
#include "render/settings.hpp"
#include "render/stage_data.hpp"
#include "render_test_caps.hpp"

#include <SDL3/SDL_stdinc.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::scene {
namespace {
using gx::draw_route::Decision;
using gx::draw_route::Route;

// C_MTXPerspective(60, 4:3, 1, 1000), as GXSetProjection receives it
Mat4x4<float> perspective(float scale = 1.f) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f * scale;
  proj.m1[1] = 1.7320508f * scale;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

gx::hints::Camera camera(AuroraCameraRole role, const Mat4x4<float>& proj, uint32_t cutEpoch = 0) {
  return {
      .role = role,
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .cutEpoch = cutEpoch,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
}

class RenderClassifyGlue : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
  }

  void TearDown() override {
    detail::remove_hints();
    detail::testing::reset_hints();
    gx_ext::records::testing::reset();
    testing::reset();
    detail::testing::reset();
    gx::g_gxState = {};
  }

  static const gx::hints::Sink& sink() { return *gx::hints::sink(); }

  static void send(const gx::hints::Camera& value) { sink().camera(sink().userData, value); }
  static void stage(Stage value) { sink().stage(sink().userData, value); }
  static void begin_object(uint64_t id, uint32_t subId = 0, uint32_t flags = 0) {
    sink().objectBegin(sink().userData, {.id = id, .subId = subId, .generation = 1, .flags = flags});
  }
  static void end_object() { sink().objectEnd(sink().userData); }

  static Decision draw() {
    const gx::draw_route::Router* router = gx::draw_route::router();
    return router->route(router->userData, {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3});
  }

  static const DrawClass& last() { return testing::classifier().last(); }
  static const FrameStats& stats() { return testing::classifier().stats(); }

  // A usable MAIN camera with the GX state set to its projection, inside the opaque window
  static gx::hints::Camera main_scene() {
    const gx::hints::Camera main = camera(AURORA_CAMERA_MAIN, perspective());
    send(main);
    stage(Stage::SceneBegin);
    gx::g_gxState.proj = main.gxProj;
    return main;
  }
};

TEST_F(RenderClassifyGlue, InstallSetsRouterRemoveClears) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router, nullptr);
  EXPECT_NE(router->route, nullptr);
  EXPECT_NE(router->push, nullptr);
  EXPECT_EQ(router->userData, nullptr);
  detail::remove_hints();
  EXPECT_EQ(gx::draw_route::router(), nullptr);

  detail::testing::set_hints_override(0);
  detail::install_hints();
  EXPECT_EQ(gx::hints::sink(), nullptr);
  EXPECT_EQ(gx::draw_route::router(), nullptr);
}

TEST_F(RenderClassifyGlue, ProjectionMatchUsesGxProj) {
  const gx::hints::Camera main = main_scene();
  EXPECT_EQ(draw().route, Route::Push);
  EXPECT_EQ(last().check, MainCheck::Passed);
  EXPECT_TRUE(last().mainClass);
  EXPECT_EQ(last().kind, Kind::MainOpaque);

  gx::g_gxState.proj.m0[0] = std::nextafter(main.gxProj.m0[0], 2.f);
  draw();
  EXPECT_EQ(last().check, MainCheck::Projection);
  EXPECT_FALSE(last().mainClass);
}

TEST_F(RenderClassifyGlue, FullProjectionLoadReadsPerspectiveFromMatrix) {
  main_scene();
  gx::g_gxState.projType = GX_ORTHOGRAPHIC;
  draw();
  EXPECT_EQ(last().check, MainCheck::Passed);

  gx::g_gxState.proj = gx::hints::gx_projection(perspective(), GX_ORTHOGRAPHIC);
  gx::g_gxState.projType = GX_PERSPECTIVE;
  draw();
  EXPECT_EQ(last().check, MainCheck::Ortho);
}

TEST_F(RenderClassifyGlue, DegenerateMainCameraCountsAsAbsent) {
  // A zero field of view makes the cotangent infinite, so the projection is not usable
  const float inf = std::numeric_limits<float>::infinity();
  Mat4x4<float> proj = perspective();
  proj.m0[0] = inf;
  proj.m1[1] = inf;
  const gx::hints::Camera degenerate = camera(AURORA_CAMERA_MAIN, proj);
  send(degenerate);
  stage(Stage::SceneBegin);
  gx::g_gxState.proj = degenerate.gxProj;
  gx::g_gxState.proj.m0[0] = 1.f;
  gx::g_gxState.proj.m1[1] = 1.f;
  draw();
  EXPECT_FALSE(detail::live_scene_hints().has(detail::SceneHasMainCamera));
  EXPECT_EQ(last().key.role, Role::Main);
  EXPECT_FALSE(last().key.mainCamera);
  EXPECT_EQ(last().check, MainCheck::NoCamera);
}

TEST_F(RenderClassifyGlue, LateMainCameraChangesRoleOnly) {
  main_scene();
  const gx::hints::Camera late = camera(AURORA_CAMERA_MAIN, perspective(2.f));
  send(late);
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_CAMERA).late, 1u);
  gx::g_gxState.proj = late.gxProj;
  draw();
  EXPECT_EQ(last().key.role, Role::Main);
  EXPECT_EQ(last().check, MainCheck::Projection);
  EXPECT_EQ(stats().hints.mainCameras, 2u);
}

TEST_F(RenderClassifyGlue, NonMainCameraReachesClassifier) {
  main_scene();
  sink().scopePush(sink().userData, {});
  send(camera(AURORA_CAMERA_REFLECTION, perspective()));
  draw();
  EXPECT_EQ(last().key.role, Role::Reflection);
  EXPECT_TRUE(last().roleOnly);
  sink().scopePop(sink().userData);
  draw();
  EXPECT_EQ(last().key.role, Role::Main);
  EXPECT_EQ(stats().hints.cameras, 2u);
}

TEST_F(RenderClassifyGlue, CallbacksForwardDrawHints) {
  sink().scopePush(sink().userData, {.categories = AURORA_CAT_FOLIAGE, .role = AURORA_CAMERA_SKY});
  sink().objectBegin(
      sink().userData,
      {.id = 0x10, .subId = 2, .generation = 3, .categories = AURORA_CAT_TERRAIN, .flags = AURORA_OBJ_BILLBOARD});
  sink().material(sink().userData, 0x1FF);
  sink().tevPolicy(sink().userData, AuroraTevPolicy{.mask = AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K1, AURORA_TEV_CH_R)});
  draw();
  const ClassKey& key = last().key;
  EXPECT_EQ(key.role, Role::Sky);
  EXPECT_EQ(key.categories, category::Foliage | category::Terrain);
  EXPECT_EQ(key.flags, object_flag::Billboard);
  EXPECT_EQ(key.materialClass, 0xFFu);
  EXPECT_EQ(key.tevPolicy, AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K1, AURORA_TEV_CH_R));
  EXPECT_EQ(key.object, (ObjectRef{.id = 0x10, .subId = 2, .generation = 3, .valid = true}));
  end_object();
  sink().scopePop(sink().userData);
  EXPECT_EQ(stats().hints.objectEnds, 1u);
  EXPECT_EQ(stats().hints.scopePops, 1u);
  EXPECT_EQ(stats().tevPolicyDraws, 1u);
}

TEST_F(RenderClassifyGlue, CaptureOnlyIsSuppressed) {
  begin_object(1, 0, AURORA_OBJ_CAPTURE_ONLY);
  const Decision hidden = draw();
  EXPECT_EQ(hidden.route, Route::Suppress);
  EXPECT_FALSE(hidden.breakMerge);
  end_object();
  EXPECT_EQ(draw().route, Route::Push);
  EXPECT_EQ(stats().suppressed, 1u);
}

TEST_F(RenderClassifyGlue, IdentityModes) {
  uint64_t id = 1;
  const auto next_object = [&id](uint32_t subId = 0) {
    end_object();
    begin_object(subId == 0 ? ++id : id, subId);
    return draw().breakMerge;
  };
  begin_object(id);
  draw();
  EXPECT_FALSE(identity_collecting());
  EXPECT_FALSE(next_object());
  EXPECT_EQ(stats().mergeBreaks, 0u);

  testing::set_identity_env(1);
  EXPECT_TRUE(identity_collecting());
  EXPECT_TRUE(next_object());
  EXPECT_TRUE(next_object(1));
  EXPECT_EQ(stats().mergeBreaks, 2u);

  testing::set_identity_env(2);
  EXPECT_TRUE(identity_collecting());
  EXPECT_FALSE(next_object(2));
  EXPECT_TRUE(next_object());
  EXPECT_EQ(stats().mergeBreaks, 3u);

  testing::set_identity_env(0);
  testing::set_identity_override(1);
  EXPECT_TRUE(identity_collecting());
  EXPECT_TRUE(next_object(3));
  testing::set_identity_override(0);
  testing::set_identity_env(1);
  EXPECT_FALSE(identity_collecting());
  EXPECT_FALSE(next_object());
  EXPECT_EQ(stats().mergeBreaks, 4u);
}

TEST_F(RenderClassifyGlue, StageForwardsBeforeRecordStage) {
  stage(Stage::SceneBegin);
  draw();
  EXPECT_EQ(last().key.window, Window::Opaque);
  stage(Stage::SceneBegin);
  aurora::test::process_hints(aurora::test::stage_hint(Stage::BeforeHud));
  draw();
  EXPECT_EQ(last().key.window, Window::Hud);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::SceneBegin)], 0);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::BeforeHud)], 1);
  EXPECT_EQ(stats().hints.markers, 2u);
  EXPECT_EQ(stats().hints.staleMarkers, 1u);
  EXPECT_EQ(stats().windows[static_cast<size_t>(Window::Hud)], 1u);
}

TEST_F(RenderClassifyGlue, BeginHintFrameSealsBeforeClear) {
  aurora::test::process_hints(aurora::test::camera_hint(AURORA_CAMERA_MAIN, true, 7));
  aurora::test::process_hints(aurora::test::epoch_hint(3));
  draw();
  detail::begin_hint_frame();
  const std::shared_ptr<const FrameReport> report = latest_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->mainCamera);
  EXPECT_EQ(report->cutEpoch, 7u);
  EXPECT_TRUE(report->sceneEpochKnown);
  EXPECT_EQ(report->sceneEpoch, 3u);
  EXPECT_EQ(report->stats.draws, 1u);
  EXPECT_EQ(report->stats.hints.mainCameras, 1u);
  EXPECT_EQ(stats().draws, 0u);

  // An explicit seal and the next frame's seal publish one report between them
  seal_frame();
  const std::shared_ptr<const FrameReport> sealed = latest_report();
  EXPECT_NE(sealed, report);
  EXPECT_FALSE(sealed->mainCamera);
  detail::begin_hint_frame();
  EXPECT_EQ(latest_report(), sealed);
  EXPECT_EQ(sealed->stats.frame + 1, stats().frame);
}

TEST_F(RenderClassifyGlue, RequestReportEnablesRecordsAndObjects) {
  const auto frame_with_object = [] {
    detail::latch_frame();
    begin_object(1);
    draw();
    end_object();
    detail::latch_frame();
    return latest_report();
  };
  const auto plain = frame_with_object();
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(plain->stats.draws, 1u);
  EXPECT_TRUE(plain->draws.empty());
  EXPECT_TRUE(plain->objects.empty());

  request_report({.records = true, .objects = true});
  const auto requested = frame_with_object();
  ASSERT_EQ(requested->draws.size(), 1u);
  EXPECT_EQ(requested->draws[0].vertices, 3u);
  ASSERT_EQ(requested->objects.size(), 1u);
  EXPECT_EQ(requested->objects[0].ref.id, 1u);
  EXPECT_EQ(requested->objects[0].draws, 1u);

  request_report({});
  const auto cleared = frame_with_object();
  EXPECT_TRUE(cleared->draws.empty());
  EXPECT_TRUE(cleared->objects.empty());
}

TEST_F(RenderClassifyGlue, ResetKeepsRequestAndIdentityMode) {
  request_report({.records = true, .objects = true});
  testing::set_identity_env(2);
  gx::hints::reset_state();
  detail::latch_frame();
  begin_object(1);
  draw();
  end_object();
  begin_object(1, 1);
  EXPECT_FALSE(draw().breakMerge);
  end_object();
  begin_object(2);
  EXPECT_TRUE(draw().breakMerge);
  end_object();
  detail::latch_frame();

  const std::shared_ptr<const FrameReport> report = latest_report();
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->draws.size(), 3u);
  EXPECT_EQ(report->objects.size(), 3u);
  EXPECT_EQ(report->stats.mergeBreaks, 1u);
  EXPECT_TRUE(identity_collecting());
}

struct PushProbe {
  static inline int frames = 0;
  static inline int pushes = 0;
  static inline DrawClass last;

  static void begin(const detail::FrameState&) noexcept { ++frames; }
  static bool push(gx::DrawData& draw, const gx::PipelineConfig&, const gx::ShaderInfo&,
                   const DrawClass& cls) noexcept {
    ++pushes;
    last = cls;
    draw.pipeline = 99;
    return true;
  }
};

TEST_F(RenderClassifyGlue, PushForwardsToViewHooks) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router, nullptr);
  gx::DrawData data{};
  data.pipeline = 7;
  const gx::PipelineConfig config{};
  const gx::ShaderInfo info{};
  EXPECT_FALSE(router->push(router->userData, data, config, info));
  EXPECT_EQ(data.pipeline, 7u);

  static constexpr ViewHooks Hooks{.beginFrame = &PushProbe::begin, .push = &PushProbe::push};
  PushProbe::frames = 0;
  PushProbe::pushes = 0;
  set_view_hooks(&Hooks);
  apply_frame_state(detail::FrameState{});
  EXPECT_EQ(PushProbe::frames, 1);
  main_scene();
  begin_object(0x42);
  draw();
  // The router classified this draw just before its push, so the hook sees its class
  EXPECT_TRUE(router->push(router->userData, data, config, info));
  EXPECT_EQ(PushProbe::pushes, 1);
  EXPECT_EQ(data.pipeline, 99u);
  EXPECT_EQ(PushProbe::last.key.object.id, 0x42u);
  EXPECT_TRUE(PushProbe::last.mainClass);
  set_view_hooks(nullptr);
  EXPECT_FALSE(router->push(router->userData, data, config, info));
  EXPECT_EQ(PushProbe::pushes, 1);
}

struct ObserverProbe {
  static inline std::string order;
  static inline uint32_t pipeline = 0;
  static inline uint32_t index = UINT32_MAX;

  static bool push(gx::DrawData& draw, const gx::PipelineConfig&, const gx::ShaderInfo&, const DrawClass&) noexcept {
    order += 'v';
    draw.pipeline = 99;
    return true;
  }
  static void observe(gx::DrawData& draw, const gx::PipelineConfig&, const gx::ShaderInfo&,
                      const DrawClass& cls) noexcept {
    order += 'o';
    pipeline = static_cast<uint32_t>(draw.pipeline);
    index = cls.index;
  }
};

TEST_F(RenderClassifyGlue, ObserverRunsLast) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router, nullptr);
  const gx::PipelineConfig config{};
  const gx::ShaderInfo info{};
  static constexpr ObserverHooks Observer{.observe = &ObserverProbe::observe};
  static constexpr ViewHooks Hooks{.push = &ObserverProbe::push};
  ObserverProbe::order.clear();
  set_observer_hooks(&Observer);
  main_scene();
  draw();
  gx::DrawData data{};
  data.pipeline = 7;
  EXPECT_FALSE(router->push(router->userData, data, config, info));
  EXPECT_EQ(ObserverProbe::order, "o");
  EXPECT_EQ(ObserverProbe::pipeline, 7u);
  EXPECT_EQ(ObserverProbe::index, last().index);

  set_view_hooks(&Hooks);
  draw();
  EXPECT_TRUE(router->push(router->userData, data, config, info));
  EXPECT_EQ(ObserverProbe::order, "ovo");
  EXPECT_EQ(ObserverProbe::pipeline, 99u);
  EXPECT_EQ(ObserverProbe::index, last().index);

  testing::reset();
  EXPECT_FALSE(router->push(router->userData, data, config, info));
  EXPECT_EQ(ObserverProbe::order, "ovo");
}

TEST_F(RenderClassifyGlue, RecordsBreakMergeOnClassChange) {
  main_scene();
  begin_object(1);
  draw();
  end_object();
  begin_object(2);
  EXPECT_FALSE(draw().breakMerge);

  gx_ext::records::testing::set_active(true, true, false);
  end_object();
  begin_object(3);
  EXPECT_TRUE(draw().breakMerge);
  EXPECT_FALSE(draw().breakMerge);
  end_object();
  begin_object(4);
  EXPECT_TRUE(draw().breakMerge);
  EXPECT_EQ(stats().mergeBreaks, 2u);
  EXPECT_FALSE(identity_collecting());
}

struct RecordProbe {
  static inline uint32_t pushes = 0;
  static inline uint32_t observedPad = 0;

  static gfx::Range push(std::span<const uint8_t> bytes) noexcept {
    ++pushes;
    return {16 * pushes, static_cast<uint32_t>(bytes.size())};
  }
  static void observe(gx::DrawData& draw, const gx::PipelineConfig&, const gx::ShaderInfo&, const DrawClass&) noexcept {
    observedPad = draw.immediateData._pad;
  }
};

TEST_F(RenderClassifyGlue, RecordStampPrecedesObserver) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router, nullptr);
  static constexpr ObserverHooks Observer{.observe = &RecordProbe::observe};
  RecordProbe::pushes = 0;
  RecordProbe::observedPad = 0;
  gx_ext::records::testing::set_push(&RecordProbe::push);
  gx_ext::records::testing::set_active(true, true, false);
  set_observer_hooks(&Observer);
  main_scene();
  begin_object(0x42);
  draw();
  const gx::PipelineConfig config{};
  const gx::ShaderInfo info{};
  gx::DrawData data{};
  data.instanceCount = 1;
  EXPECT_FALSE(router->push(router->userData, data, config, info));
  EXPECT_EQ(RecordProbe::pushes, 2u);
  EXPECT_EQ(data.immediateData._pad, 32u / 4);
  EXPECT_EQ(RecordProbe::observedPad, data.immediateData._pad);

  gx::DrawData empty{};
  draw();
  EXPECT_FALSE(router->push(router->userData, empty, config, info));
  EXPECT_EQ(empty.immediateData._pad, 0u);
  EXPECT_EQ(RecordProbe::observedPad, 0u);
}

TEST_F(RenderClassifyGlue, IdentityDemandStartsCollection) {
  EXPECT_FALSE(identity_collecting());
  set_identity_demand(true);
  EXPECT_TRUE(identity_collecting());
  set_identity_demand(false);
  EXPECT_FALSE(identity_collecting());
}

TEST_F(RenderClassifyGlue, BlendedRule) {
  EXPECT_FALSE(blended(GX_BM_NONE, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR));
  EXPECT_FALSE(blended(GX_BM_BLEND, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR));
  EXPECT_FALSE(blended(GX_BM_LOGIC, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_COPY));
  EXPECT_TRUE(blended(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR));
  EXPECT_TRUE(blended(GX_BM_BLEND, GX_BL_DSTALPHA, GX_BL_ZERO, GX_LO_CLEAR));
  EXPECT_TRUE(blended(GX_BM_SUBTRACT, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR));
  EXPECT_TRUE(blended(GX_BM_LOGIC, GX_BL_ONE, GX_BL_ZERO, GX_LO_XOR));

  main_scene();
  gx::g_gxState.blendMode = GX_BM_BLEND;
  draw();
  EXPECT_EQ(last().kind, Kind::MainBlended);
  gx::g_gxState.blendMode = GX_BM_NONE;
  gx::g_gxState.colorUpdate = false;
  draw();
  EXPECT_EQ(last().kind, Kind::MainDepthOnly);
}

TEST_F(RenderClassifyGlue, OverlayArmingTruthTable) {
  using overlay::ArmInputs;
  using overlay::Env;
  const ArmInputs pbr{.hints = true, .gbufferLayout = true, .surfaceLayout = true, .pbr = true, .legacyBlend = 0.f};
  EXPECT_TRUE(overlay::pbr_can_change_pixels(pbr));
  EXPECT_TRUE(overlay::armed_for(pbr));
  const auto with = [](ArmInputs in, auto change) {
    change(in);
    return in;
  };
  EXPECT_TRUE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.legacyBlend = 0.999f; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.legacyBlend = 1.f; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.hints = false; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.gbufferLayout = false; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.surfaceLayout = false; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.pbr = false; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.identityForced = true; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.classView = true; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.passAfterOverlay = true; })));
  EXPECT_FALSE(overlay::armed_for(with(pbr, [](ArmInputs& in) { in.env = Env::Off; })));

  // The verification modes arm on any G-buffer frame, but never past the same vetoes
  const ArmInputs tierB{.hints = true, .gbufferLayout = true, .legacyBlend = 1.f};
  EXPECT_FALSE(overlay::armed_for(tierB));
  for (const Env env : {Env::Force, Env::Drop, Env::Restore}) {
    ArmInputs in = tierB;
    in.env = env;
    EXPECT_TRUE(overlay::armed_for(in));
    EXPECT_FALSE(overlay::armed_for(with(in, [](ArmInputs& v) { v.hints = false; })));
    EXPECT_FALSE(overlay::armed_for(with(in, [](ArmInputs& v) { v.gbufferLayout = false; })));
    EXPECT_FALSE(overlay::armed_for(with(in, [](ArmInputs& v) { v.identityForced = true; })));
    EXPECT_FALSE(overlay::armed_for(with(in, [](ArmInputs& v) { v.classView = true; })));
    EXPECT_FALSE(overlay::armed_for(with(in, [](ArmInputs& v) { v.passAfterOverlay = true; })));
  }
}

TEST_F(RenderClassifyGlue, OverlayEnvParsedOnce) {
  using overlay::Env;
  overlay::testing::reset();
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_OVERLAY_DEFER", "force", true);
  EXPECT_EQ(overlay::env(), Env::Force);
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_OVERLAY_DEFER", "drop", true);
  EXPECT_EQ(overlay::env(), Env::Force);
  const std::array<std::pair<const char*, Env>, 6> cases{{
      {"drop", Env::Drop},
      {"restore", Env::Restore},
      {"off", Env::Off},
      {"0", Env::Off},
      {"on", Env::Unset},
      {"force", Env::Force},
  }};
  for (const auto& [text, expected] : cases) {
    overlay::testing::reset();
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_OVERLAY_DEFER", text, true);
    EXPECT_EQ(overlay::env(), expected) << text;
  }
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_OVERLAY_DEFER");
  overlay::testing::reset();
  EXPECT_EQ(overlay::env(), Env::Unset);
}

bool g_applied = false;
bool applied_source(const detail::StageData&) noexcept { return g_applied; }

TEST_F(RenderClassifyGlue, OverlayDispositionFollowsMode) {
  using gfx::OverlayDisposition;
  using overlay::Env;
  const detail::StageData opaqueDone{.stage = Stage::OpaqueDone};
  overlay::testing::set_env(Env::Drop);
  EXPECT_EQ(overlay::disposition(&opaqueDone), OverlayDisposition::Discard);
  overlay::testing::set_env(Env::Restore);
  EXPECT_EQ(overlay::disposition(&opaqueDone), OverlayDisposition::Restore);
  overlay::testing::set_env(Env::Force);
  EXPECT_EQ(overlay::disposition(nullptr), OverlayDisposition::Replay);

  // Without W5's source nothing can say the PBR term ran, so every frame keeps the original order
  overlay::testing::set_env(Env::Unset);
  EXPECT_EQ(overlay::disposition(&opaqueDone), OverlayDisposition::Restore);
  overlay::set_pbr_applied_source(&applied_source);
  g_applied = true;
  EXPECT_EQ(overlay::disposition(&opaqueDone), OverlayDisposition::Replay);
  EXPECT_EQ(overlay::disposition(nullptr), OverlayDisposition::Restore);
  g_applied = false;
  EXPECT_EQ(overlay::disposition(&opaqueDone), OverlayDisposition::Restore);
  overlay::set_pbr_applied_source(nullptr);
}

TEST_F(RenderClassifyGlue, OverlayKindDefersOnlyWhileArmed) {
  main_scene();
  sink().scopePush(sink().userData, {.categories = AURORA_CAT_PARTICLE | AURORA_CAT_OVERLAY | AURORA_CAT_NO_PBR});
  gx::g_gxState.blendMode = GX_BM_BLEND;
  gx::g_gxState.blendFacSrc = GX_BL_SRCALPHA;
  gx::g_gxState.blendFacDst = GX_BL_INVSRCALPHA;
  gx::g_gxState.depthUpdate = false;
  EXPECT_EQ(draw().route, Route::Push);
  EXPECT_EQ(last().kind, Kind::Overlay);

  overlay::testing::set_armed(true);
  const uint32_t before = stats().draws;
  EXPECT_EQ(draw().route, Route::Defer);
  // Classified and counted like an in-place draw, so records and reports do not change
  EXPECT_EQ(stats().draws, before + 1);
  gx::g_gxState.depthUpdate = true;
  EXPECT_EQ(draw().route, Route::Push);
  gx::g_gxState.depthUpdate = false;
  gx::g_gxState.colorUpdate = false;
  EXPECT_EQ(draw().route, Route::Push);
  gx::g_gxState.colorUpdate = true;
  sink().scopePush(sink().userData, {.categories = AURORA_CAT_FAKE_SHADOW});
  EXPECT_EQ(draw().route, Route::Push);
  sink().scopePop(sink().userData);
  EXPECT_EQ(draw().route, Route::Defer);
  sink().scopePop(sink().userData);
  EXPECT_EQ(draw().route, Route::Push);
  EXPECT_NE(last().kind, Kind::Overlay);

  sink().scopePush(sink().userData, {.categories = AURORA_CAT_OVERLAY});
  stage(Stage::OpaqueDone);
  EXPECT_EQ(draw().route, Route::Push);
  EXPECT_NE(last().kind, Kind::Overlay);
  sink().scopePop(sink().userData);
}

std::shared_ptr<const detail::CompiledGraph> builtin_graph(const Settings& settings) {
  static const Caps caps = test::full_caps();
  const std::vector<detail::PassDesc> passes{
      detail::hzb::opaque_pass_desc(),        detail::ao::hiz::pass_desc(),     detail::ao::gtao::pass_desc(),
      detail::ao::upsample::pass_desc(),      detail::ao::denoise::pass_desc(), detail::composite::copy_pass_desc(),
      detail::composite::pass_desc(),         gx_ext::fwd_env::pass_desc(),     detail::taa::pass_desc(),
      detail::upscale::fsr_pass::pass_desc(),
  };
  const std::vector<detail::GxReadDesc> reads{gx_ext::fwd_env::read_desc()};
  detail::GraphEnvironment env;
  env.efbSize = {1280, 720};
  env.outputSize = {1920, 1080};
  env.sceneSize = {1280, 720};
  const auto set = [&](detail::GxResource resource, wgpu::TextureFormat format) {
    env.gxFormats[static_cast<size_t>(resource)] = format;
    env.gxUsage[static_cast<size_t>(resource)] = wgpu::TextureUsage::RenderAttachment |
                                                 wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
                                                 wgpu::TextureUsage::CopyDst;
  };
  set(detail::GxResource::Color, wgpu::TextureFormat::BGRA8Unorm);
  set(detail::GxResource::Depth, wgpu::TextureFormat::Depth32Float);
  set(detail::GxResource::Normal, wgpu::TextureFormat::RGB10A2Unorm);
  set(detail::GxResource::Motion, wgpu::TextureFormat::RG16Float);
  set(detail::GxResource::Reactive, wgpu::TextureFormat::R8Unorm);
  set(detail::GxResource::Fog, wgpu::TextureFormat::RGBA8Uint);
  set(detail::GxResource::Surface, wgpu::TextureFormat::RGBA32Uint);
  return detail::compile_graph(detail::GraphInputs{
      .passes = passes,
      .gxReads = reads,
      .settings = &settings,
      .caps = &caps,
      .env = env,
      .sceneDepthSource = DepthSource::Direct,
  });
}

TEST_F(RenderClassifyGlue, OverlayPassAfterIgnoresBuiltIns) {
  Settings settings;
  settings.ao = Quality::High;
  settings.taa = true;
  settings.upscaler = Upscaler::Fsr;
  const std::shared_ptr<const detail::CompiledGraph> graph = builtin_graph(settings);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(graph->find_node(detail::composite::Pass), detail::InvalidNode);
  EXPECT_EQ(graph->nodes[graph->find_node(detail::composite::Pass)].status, detail::CullReason::None);
  EXPECT_FALSE(overlay::pass_after_overlay(*graph));
}

TEST_F(RenderClassifyGlue, ResetClearsEverything) {
  detail::begin_hint_frame();
  sink().scopePush(sink().userData, {.categories = AURORA_CAT_DECAL});
  begin_object(4);
  draw();
  seal_frame();
  ASSERT_NE(latest_report(), nullptr);
  gx::hints::reset_state();
  EXPECT_EQ(stats().frame, 0u);
  EXPECT_EQ(latest_report(), nullptr);
  draw();
  EXPECT_FALSE(last().key.object.valid);
  EXPECT_EQ(last().key.categories, 0u);
  EXPECT_EQ(stats().unclassified, 1u);
}
TEST_F(RenderClassifyGlue, RetainedKeepsMerging) {
  struct Counts {
    uint64_t routed = 0;
    uint64_t mergeBreaks = 0;
    uint64_t substituted = 0;
    uint64_t merged = 0;
    uint64_t commands = 0;
    uint64_t entries = 0;
  };
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router->merge, nullptr);
  std::vector<uint8_t> fifo(4096, 0);
  gx::PipelineConfig config{};
  config.shaderConfig.vtxStride = 12;
  const gx::ShaderInfo info{};
  const auto run = [&](bool retainedOn) {
    retained::testing::reset();
    retained::set_demand(retained::Demand::Tool, retainedOn);
    Counts counts;
    gx::DrawData command{};
    for (uint32_t frame = 0; frame < 3; ++frame) {
      detail::begin_hint_frame();
      retained::latch_gate(Settings{}, test::full_caps());
      apply_frame_state(detail::FrameState{});
      main_scene();
      uint32_t offset = 0;
      bool mergeable = false;
      const auto gx_draw = [&](uint32_t vtxCount, bool stateChange) {
        const uint32_t size = vtxCount * 12;
        const Decision decision = router->route(
            router->userData,
            {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = vtxCount, .vertices = {fifo.data() + offset, size}});
        ++counts.routed;
        mergeable = mergeable && !stateChange;
        if (mergeable && decision.breakMerge) {
          ++counts.mergeBreaks;
        }
        if (mergeable && !decision.breakMerge) {
          command.vtxCount += vtxCount;
          command.vertRange.size += size;
          router->merge(router->userData,
                        {.vertOffset = offset, .vertSize = size, .command = &command, .config = &config});
          ++counts.merged;
        } else {
          command = {};
          command.vertRange = {offset, size};
          command.vtxCount = vtxCount;
          command.instanceCount = 1;
          command.immediateData.vtxStart = offset;
          const bool substituted = router->push(router->userData, command, config, info);
          counts.substituted += substituted ? 1 : 0;
          mergeable = !substituted;
          ++counts.commands;
        }
        offset += size;
      };
      begin_object(0x10);
      gx_draw(3, false);
      gx_draw(3, false);
      gx_draw(3, true);
      end_object();
      begin_object(0x11);
      gx_draw(4, false);
      gx_draw(4, false);
      end_object();
      gx_draw(3, false);
      detail::StageData stage{.stage = Stage::OpaqueDone};
      retained::prepare(stage);
      retained::seal(stage);
      counts.entries += stage.retained != nullptr ? stage.retained->entries.size() : 0;
      seal_frame();
    }
    retained::testing::reset();
    return counts;
  };
  for (const int identity : {0, 1}) {
    testing::set_identity_override(identity);
    const Counts off = run(false);
    const Counts on = run(true);
    EXPECT_EQ(on.routed, off.routed) << identity;
    EXPECT_EQ(on.mergeBreaks, off.mergeBreaks) << identity;
    EXPECT_EQ(on.substituted, off.substituted) << identity;
    EXPECT_EQ(on.merged, off.merged) << identity;
    EXPECT_EQ(on.commands, off.commands) << identity;
    EXPECT_EQ(off.entries, 0u);
    EXPECT_EQ(on.entries, on.routed);
    EXPECT_EQ(on.merged, identity == 0 ? 12u : 6u);
    EXPECT_EQ(on.mergeBreaks, identity == 0 ? 0u : 6u);
  }
}

TEST_F(RenderClassifyGlue, CaptureOnlyWhileRetainedAndMain) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router->capture, nullptr);
  const auto frame = [&](bool on) {
    retained::set_demand(retained::Demand::Tool, on);
    detail::begin_hint_frame();
    retained::latch_gate(Settings{}, test::full_caps());
    apply_frame_state(detail::FrameState{});
    main_scene();
  };
  const auto route = [&](GXPrimitive prim = GX_TRIANGLES) {
    return router->route(router->userData, {.prim = prim, .fmt = GX_VTXFMT0, .vtxCount = 3}).route;
  };
  retained::testing::reset();
  frame(false);
  begin_object(0x50, 0, AURORA_OBJ_CAPTURE_ONLY);
  EXPECT_EQ(route(), Route::Suppress);
  end_object();
  seal_frame();

  frame(true);
  begin_object(0x50, 0, AURORA_OBJ_CAPTURE_ONLY);
  EXPECT_EQ(route(), Route::Capture);
  EXPECT_EQ(route(GX_LINES), Route::Suppress);
  EXPECT_EQ(route(GX_POINTS), Route::Suppress);
  sink().scopePush(sink().userData, {});
  send(camera(AURORA_CAMERA_REFLECTION, perspective()));
  EXPECT_EQ(route(), Route::Suppress);
  EXPECT_EQ(last().key.role, Role::Reflection);
  sink().scopePop(sink().userData);
  gx::g_gxState.vtxDesc[GX_VA_POS] = GX_INDEX16;
  gx::g_gxState.arrays[GX_VA_POS].gpu = true;
  gx::g_gxState.gpuArrayMask = 1;
  EXPECT_EQ(route(), Route::Suppress);
  gx::g_gxState.gpuArrayMask = 0;
  gx::g_gxState.arrays[GX_VA_POS] = {};
  gx::g_gxState.vtxDesc[GX_VA_POS] = GX_DIRECT;
  EXPECT_EQ(route(), Route::Capture);
  end_object();
  EXPECT_EQ(route(), Route::Push);
  seal_frame();
  retained::testing::reset();
}

TEST_F(RenderClassifyGlue, CapturedShapeKeepsItsSlot) {
  retained::testing::reset();
  retained::set_demand(retained::Demand::Tool, true);
  const gx::draw_route::Router* router = gx::draw_route::router();
  std::vector<uint8_t> fifo(256, 0);
  gx::PipelineConfig config{};
  config.shaderConfig.vtxStride = 12;
  const gx::ShaderInfo info{};
  std::array<uint32_t, 2> slots{};
  for (uint32_t frame = 0; frame < 2; ++frame) {
    detail::begin_hint_frame();
    retained::latch_gate(Settings{}, test::full_caps());
    apply_frame_state(detail::FrameState{});
    main_scene();
    begin_object(0x60, 0, frame == 0 ? 0 : AURORA_OBJ_CAPTURE_ONLY);
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t offset = i * 36;
      const gx::draw_route::Query query{
          .prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3, .vertices = {fifo.data() + offset, 36}};
      const Decision decision = router->route(router->userData, query);
      if (frame == 0) {
        EXPECT_EQ(decision.route, Route::Push);
        gx::DrawData command{.vertRange = {offset, 36}, .vtxCount = 3, .instanceCount = 1};
        router->push(router->userData, command, config, info);
      } else {
        EXPECT_EQ(decision.route, Route::Capture);
        const gx::DrawImmediateData immediates{.vtxStart = offset};
        router->capture(router->userData, {.prim = GX_TRIANGLES,
                                           .vertOffset = offset,
                                           .vertSize = 36,
                                           .vtxCount = 3,
                                           .immediates = &immediates,
                                           .attrs = config.shaderConfig.attrs.data(),
                                           .vtxStride = 12});
      }
    }
    end_object();
    detail::StageData stage{.stage = Stage::OpaqueDone};
    retained::prepare(stage);
    retained::seal(stage);
    ASSERT_NE(stage.retained, nullptr);
    ASSERT_EQ(stage.retained->entries.size(), 2u);
    for (uint32_t i = 0; i < 2; ++i) {
      const retained::Entry& entry = stage.retained->entries[i];
      EXPECT_EQ((entry.flags & retained::entry_flag::CaptureOnly) != 0, frame == 1) << i;
      if (frame == 0) {
        slots[i] = entry.slot;
      } else {
        EXPECT_EQ(entry.slot, slots[i]) << i;
        EXPECT_EQ(entry.record, 0u) << i;
      }
    }
    seal_frame();
  }
  EXPECT_EQ(retained::frame_counts().created, 0u);
  EXPECT_EQ(retained::frame_counts().newOccurrence, 0u);
  const retained::testing::Instance instance = retained::testing::instance(slots[1]);
  EXPECT_TRUE(instance.captureOnlySeen);
  EXPECT_TRUE(instance.visibleSeen);
  retained::testing::reset();
}
} // namespace
} // namespace aurora::render::scene

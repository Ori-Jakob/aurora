#include <gtest/gtest.h>

#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "hint_bytes.hpp"
#include "render/frame_state.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/scene/classify.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"

#include <cmath>
#include <limits>
#include <string>

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
} // namespace
} // namespace aurora::render::scene

#include <gtest/gtest.h>

#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/lru_cache.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture.hpp"
#if AURORA_ENABLE_RENDER
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/settings.hpp"
#include "render/stage.hpp"
#endif
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace aurora::gfx {
namespace {

constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth24Plus;

class GfxRecordingTest : public ::testing::Test {
protected:
  void SetUp() override {
    webgpu::g_graphicsConfig.surfaceConfiguration.format = ColorFormat;
    webgpu::g_graphicsConfig.depthFormat = DepthFormat;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer.size = {640, 480, 1};
    webgpu::g_frameBuffer.format = ColorFormat;
    webgpu::g_depthBuffer.size = {640, 480, 1};
    webgpu::g_depthBuffer.format = DepthFormat;
    detail::testing::suppress_render_worker(true);
    detail::begin_recording(frame, 0);
  }

  void TearDown() override {
    if (recordingActive) {
      if (is_offscreen()) {
        end_offscreen();
      }
      finish();
      detail::end_recording();
    }
    detail::shutdown_recording();
  }

  void seed(uint32_t width, uint32_t height) {
    detail::testing::seed_offscreen_cache(width, height, ColorFormat, DepthFormat);
  }

  bool cached(uint32_t width, uint32_t height) {
    return detail::testing::is_offscreen_cached(width, height, ColorFormat, DepthFormat);
  }

  void next_frame() {
    finish();
    detail::end_recording();
    frame = {};
    detail::begin_recording(frame, 0);
  }

  void copy_current_offscreen() {
    const auto& pass = frame.renderPasses.back();
    const auto& size = pass.colorAttachments[SceneColorAttachmentIndex].size;
    auto target = std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size,
                                               ColorFormat, 1, GX_TF_RGBA8);
    resolve_pass_into(std::move(target), {0, 0, static_cast<int32_t>(size.width), static_cast<int32_t>(size.height)},
                      false, false, false, {}, 1.f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);
  }

  size_t count_efb_passes() const {
    return static_cast<size_t>(
        std::ranges::count_if(frame.renderPasses, [](const auto& pass) { return pass.label.starts_with("EFB"); }));
  }

  detail::FramePacket frame;
  bool recordingActive = true;
};

TEST_F(GfxRecordingTest, CreateRestoreReturnsToEfb) {
  seed(320, 180);
  begin_offscreen(320, 180);
  ASSERT_TRUE(is_offscreen());

  end_offscreen();

  EXPECT_FALSE(is_offscreen());
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, EfbPassUsesDiscoveredSceneLayout) {
  ASSERT_FALSE(frame.renderPasses.empty());
  const auto discovered = scene_render_target_layout();
  const auto targetLayout = frame.renderPasses.front().target_layout();

  EXPECT_EQ(targetLayout.key, discovered.key);
  EXPECT_EQ(targetLayout.colorAttachmentCount, discovered.colorAttachmentCount);
  EXPECT_EQ(targetLayout.colorAttachments[SceneColorAttachmentIndex].semantic, ColorAttachmentSemantic::SceneColor);
}

TEST_F(GfxRecordingTest, CopiedOffscreenPassIsRetainedOnRestore) {
  seed(320, 180);
  begin_offscreen(320, 180);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
}

TEST_F(GfxRecordingTest, ReplacementRetainsCopiedPassesAndDiscardsContinuations) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 5u);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_TRUE(frame.renderPasses[2].has_consumer());
  EXPECT_FALSE(frame.renderPasses[2].discardable);
  EXPECT_TRUE(frame.renderPasses[3].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, RepeatedUncopiedCreatesDiscardEarlierPasses) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  begin_offscreen(160, 90);
  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, PublicCreatePassRejectsExistingOffscreenPass) {
  seed(320, 180);
  seed(160, 90);
  ASSERT_TRUE(create_pass(320, 180));

  EXPECT_FALSE(create_pass(160, 90));

  ResolvedTargets ignored;
  EXPECT_TRUE(resolve_pass({.color = false, .depth = false}, ignored));
}

TEST_F(GfxRecordingTest, FinalizedPassesAreSealedOrDeliberatelyDiscarded) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  end_offscreen();
  finish();

  ASSERT_FALSE(frame.renderPasses.empty());
  for (const auto& pass : frame.renderPasses) {
    EXPECT_TRUE(pass.sealed);
    if (!pass.has_consumer() && pass.label.starts_with("Offscreen")) {
      EXPECT_TRUE(pass.discardable);
    }
  }
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxRecordingTest, OffscreenBudgetIncludesColorAndDepth) {
  seed(2048, 2048); // 16 MiB color + 16 MiB depth
  seed(4096, 1024); // Another 32 MiB
  ASSERT_EQ(detail::testing::offscreen_cache_stats().bytes, RenderTextureCacheLimits.bytes);
  ASSERT_EQ(detail::testing::offscreen_cache_stats().entries, 2);

  begin_offscreen(2048, 2048); // Refresh the older entry within the same frame.
  end_offscreen();
  seed(1024, 4096);

  EXPECT_TRUE(cached(2048, 2048));
  EXPECT_FALSE(cached(4096, 1024));
  EXPECT_TRUE(cached(1024, 4096));
  EXPECT_EQ(detail::testing::offscreen_cache_stats().bytes, RenderTextureCacheLimits.bytes);
}

TEST_F(GfxRecordingTest, OffscreenBudgetBoundsContinuousSizeChanges) {
  for (uint32_t width = 256; width < 2048; ++width) {
    seed(width, 1024);
    begin_offscreen(width, 1024);
    end_offscreen();
    EXPECT_LE(detail::testing::offscreen_cache_stats().bytes, RenderTextureCacheLimits.bytes);
  }
  EXPECT_FALSE(cached(256, 1024));
  EXPECT_TRUE(cached(2047, 1024));
}

TEST_F(GfxRecordingTest, OversizedOffscreenDoesNotDisplaceReusableSizes) {
  seed(320, 240);
  seed(4096, 4096); // 128 MiB including depth
  EXPECT_TRUE(cached(320, 240));
  EXPECT_FALSE(cached(4096, 4096));
  EXPECT_EQ(detail::testing::offscreen_cache_stats().entries, 1);
}

TEST_F(GfxRecordingTest, OffscreenSweepKeepsHotSizesAndExpiresIdleOnes) {
  seed(320, 240);
  seed(160, 120);
  for (uint64_t i = 0; i < RenderTextureCacheLimits.idleFrames.value(); ++i) {
    begin_offscreen(320, 240);
    end_offscreen();
    next_frame();
  }
  EXPECT_TRUE(cached(160, 120)); // Exactly 32 frames old is still retained.
  for (uint64_t i = 0; i < RenderTextureCacheLimits.sweepFrames; ++i) {
    begin_offscreen(320, 240);
    end_offscreen();
    next_frame();
  }
  EXPECT_TRUE(cached(320, 240));
  EXPECT_FALSE(cached(160, 120));
  EXPECT_EQ(detail::testing::offscreen_cache_stats().entries, 1);
}

TEST_F(GfxRecordingTest, OffscreenReuseSurvivesEfbResizeCycles) {
  seed(320, 240);
  seed(640, 360);
  for (uint64_t i = 0; i < 4 * RenderTextureCacheLimits.idleFrames.value(); ++i) {
    const uint32_t width = i % 2 == 0 ? 640 : 1280;
    const uint32_t height = i % 2 == 0 ? 480 : 720;
    webgpu::g_frameBuffer.size = {width, height, 1};
    webgpu::g_depthBuffer.size = {width, height, 1};
    next_frame();
    ASSERT_TRUE(cached(width / 2, height / 2));
    begin_offscreen(width / 2, height / 2);
    const auto size = get_render_target_size();
    EXPECT_EQ(size.x, width / 2);
    EXPECT_EQ(size.y, height / 2);
    end_offscreen();
  }
  EXPECT_EQ(detail::testing::offscreen_cache_stats().entries, 2);
}

TEST_F(GfxRecordingTest, OffscreenKeyIncludesColorAndDepthFormats) {
  constexpr auto OtherColor = wgpu::TextureFormat::BGRA8Unorm;
  constexpr auto OtherDepth = wgpu::TextureFormat::Depth32Float;
  seed(320, 240);
  detail::testing::seed_offscreen_cache(320, 240, OtherColor, DepthFormat);
  detail::testing::seed_offscreen_cache(320, 240, OtherColor, OtherDepth);
  ASSERT_EQ(detail::testing::offscreen_cache_stats().entries, 3);
  EXPECT_EQ(detail::testing::offscreen_cache_stats().bytes, 3 * 320 * 240 * 8);

  for (const auto [color, depth] :
       {std::pair{ColorFormat, DepthFormat}, std::pair{OtherColor, DepthFormat}, std::pair{OtherColor, OtherDepth}}) {
    webgpu::g_graphicsConfig.surfaceConfiguration.format = color;
    webgpu::g_graphicsConfig.depthFormat = depth;
    begin_offscreen(320, 240);
    const auto layout = get_render_target_layout();
    EXPECT_EQ(layout.colorAttachments[SceneColorAttachmentIndex].format, color);
    EXPECT_EQ(layout.depthStencilFormat, depth);
    end_offscreen();
  }
}

TEST_F(GfxRecordingTest, StageWithoutFrameStateIsIgnored) {
  ASSERT_EQ(frame.renderState, nullptr);
  const size_t opCount = frame.ops.size();
  const size_t passCount = frame.renderPasses.size();

  for (uint8_t stage = 0; stage <= static_cast<uint8_t>(render::Stage::FrameEnd); ++stage) {
    record_stage(static_cast<render::Stage>(stage));
  }

  EXPECT_EQ(frame.ops.size(), opCount);
  EXPECT_EQ(frame.renderPasses.size(), passCount);
  EXPECT_TRUE(frame.stages.empty());
  EXPECT_FALSE(frame.renderPasses.front().sealed);
}

TEST_F(GfxRecordingTest, SubmitHooksWithoutStateAreNoOps) {
  gfx::before_submit();
  gfx::after_frame();
}

#if AURORA_ENABLE_RENDER
using render::Stage;

constexpr auto PassOp = detail::FrameOpType::RenderPass;
constexpr auto CopyOp = detail::FrameOpType::TextureCopy;
constexpr auto StageOp = detail::FrameOpType::RenderStage;

class GfxRenderStageTest;

struct TestPass {
  GfxRenderStageTest* owner = nullptr;
  std::string name;
  bool enabled = true;
};

struct PassShape {
  std::string label;
  bool sealed = false;
  bool discardable = false;

  bool operator==(const PassShape&) const = default;
};

struct FrameShape {
  std::vector<detail::FrameOpType> ops;
  std::vector<PassShape> passes;

  bool operator==(const FrameShape&) const = default;
};

class GfxRenderStageTest : public GfxRecordingTest {
protected:
  void SetUp() override {
    render::detail::testing::reset();
    GfxRecordingTest::SetUp();
  }

  void TearDown() override {
    GfxRecordingTest::TearDown();
    render::detail::testing::reset();
  }

  // The base fixture's first frame began before any pass was registered, so it has no FrameState to schedule from
  void latch_new_frame(size_t expectedPasses) {
    finish();
    detail::end_recording();
    frame = {};
    frame.renderState = render::detail::latch_frame();
    detail::begin_recording(frame, 0);
    ASSERT_NE(frame.renderState, nullptr);
    ASSERT_NE(frame.renderState->graph, nullptr);
    ASSERT_EQ(frame.renderState->graph->passes.size(), expectedPasses);
  }

  render::detail::PassId add_pass(std::string name, Stage stage, uint16_t order = 0, bool enabled = true,
                                  bool runsWithoutMarker = false) {
    auto& pass = passes.emplace_back(TestPass{.owner = this, .name = name, .enabled = enabled});
    return render::detail::register_pass({
        .name = std::move(name),
        .stage = stage,
        .order = order,
        .runsWithoutMarker = runsWithoutMarker,
        .enabled = pass_enabled,
        .encode = encode_pass,
        .state = &pass,
    });
  }

  std::vector<detail::FrameOpType> op_types() const {
    std::vector<detail::FrameOpType> types;
    for (const auto& op : frame.ops) {
      types.push_back(op.type);
    }
    return types;
  }

  FrameShape frame_shape() const {
    FrameShape shape{.ops = op_types()};
    for (const auto& pass : frame.renderPasses) {
      shape.passes.push_back({.label = pass.label, .sealed = pass.sealed, .discardable = pass.discardable});
    }
    return shape;
  }

  std::vector<std::string> run(const render::detail::StageData& stage) {
    encoded.clear();
    render::detail::run_stage(wgpu::CommandEncoder{}, stage);
    return encoded;
  }

  void run_marker_script(bool markers, FrameShape& shape) {
    ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
    ASSERT_EQ(frame.renderState->graph, render::detail::empty_graph());
    const auto marker = [markers](Stage stage) {
      if (markers) {
        record_stage(stage);
      }
    };
    marker(Stage::SceneBegin);
    begin_offscreen(320, 180);
    marker(Stage::OpaqueDone);
    end_offscreen();
    marker(Stage::PrePost);
    ResolvedTargets out;
    ASSERT_TRUE(resolve_pass({.color = false, .depth = false}, out));
    marker(Stage::SceneEnd);
    begin_offscreen(320, 180);
    marker(Stage::BeforeHud);
    end_offscreen();
    marker(Stage::FrameEnd);
    finish();
    EXPECT_TRUE(frame.stages.empty());
    shape = frame_shape();
  }

  static bool pass_enabled(const render::detail::StageData&, const void* state) {
    return static_cast<const TestPass*>(state)->enabled;
  }

  static void encode_pass(render::detail::PassContext&, const void* state) {
    const auto* pass = static_cast<const TestPass*>(state);
    pass->owner->encoded.push_back(pass->name);
  }

  static bool check_stage_data(const render::detail::StageData& stage, const void* state) {
    const auto* pass = static_cast<const TestPass*>(state);
    auto& self = *pass->owner;
    self.stageDataChecked = pass == &self.passes.back() && stage.stage == Stage::OpaqueDone &&
                            stage.frameId == self.frame.frameId && stage.frameIndex == self.frame.frameIndex &&
                            stage.markerSeen && !stage.deferred && stage.frame == self.frame.renderState.get();
    return self.stageDataChecked;
  }

  std::deque<TestPass> passes;
  std::vector<std::string> encoded;
  bool stageDataChecked = false;
};

TEST_F(GfxRenderStageTest, EmptyGraphEmitsNoOps) {
  seed(320, 180);
  FrameShape withMarkers;
  ASSERT_NO_FATAL_FAILURE(run_marker_script(true, withMarkers));
  FrameShape withoutMarkers;
  ASSERT_NO_FATAL_FAILURE(run_marker_script(false, withoutMarkers));

  EXPECT_FALSE(withMarkers.ops.empty());
  EXPECT_EQ(withMarkers.ops, withoutMarkers.ops);
  EXPECT_EQ(withMarkers.passes, withoutMarkers.passes);
}

TEST_F(GfxRenderStageTest, DisabledPassesEmitNoOps) {
  add_pass("opaque", Stage::OpaqueDone, 0, false);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  record_stage(Stage::OpaqueDone);

  EXPECT_TRUE(frame.ops.empty());
  EXPECT_TRUE(frame.stages.empty());
  ASSERT_EQ(frame.renderPasses.size(), 1u);
  EXPECT_FALSE(frame.renderPasses[0].sealed);
}

TEST_F(GfxRenderStageTest, EnabledStageSealsAndResumes) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  const auto& resumed = frame.renderPasses[1];
  EXPECT_EQ(resumed.label, "EFB");
  EXPECT_FALSE(resumed.sealed);
  EXPECT_FALSE(resumed.clearDepth);
  EXPECT_FALSE(resumed.colorAttachments[SceneColorAttachmentIndex].clear);
  EXPECT_EQ(resumed.colorAttachments[SceneColorAttachmentIndex].loadOp, wgpu::LoadOp::Undefined);
  ASSERT_EQ(frame.stages.size(), 1u);
  const auto& stage = frame.stages[0];
  EXPECT_EQ(stage.stage, Stage::OpaqueDone);
  EXPECT_TRUE(stage.markerSeen);
  EXPECT_FALSE(stage.deferred);
  EXPECT_FALSE(stage.flushedAtFinish);
  EXPECT_EQ(stage.passCount, 1u);
  EXPECT_TRUE(stage.is_enabled(0));
  EXPECT_EQ(frame.ops[1].stageData, &frame.stages[0]);
}

TEST_F(GfxRenderStageTest, StagesKeepMarkerOrder) {
  add_pass("scene", Stage::SceneBegin);
  add_pass("opaque", Stage::OpaqueDone);
  add_pass("post", Stage::PrePost);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(3));

  record_stage(Stage::SceneBegin);
  record_stage(Stage::OpaqueDone);
  record_stage(Stage::PrePost);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp, PassOp, StageOp, PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 3u);
  EXPECT_EQ(frame.stages[0].stage, Stage::SceneBegin);
  EXPECT_EQ(frame.stages[1].stage, Stage::OpaqueDone);
  EXPECT_EQ(frame.stages[2].stage, Stage::PrePost);
  ASSERT_EQ(frame.renderPasses.size(), 4u);
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  for (size_t i = 1; i < 3; ++i) {
    EXPECT_TRUE(frame.renderPasses[i].sealed) << i;
    EXPECT_TRUE(frame.renderPasses[i].discardable) << i;
  }
}

TEST_F(GfxRenderStageTest, PassesRunByOrderThenRegistration) {
  add_pass("b", Stage::OpaqueDone, 1);
  add_pass("a", Stage::OpaqueDone, 0);
  add_pass("c", Stage::OpaqueDone, 1);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(3));

  record_stage(Stage::OpaqueDone);

  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"a", "b", "c"}));
}

TEST_F(GfxRenderStageTest, EnabledMaskSpansWords) {
  constexpr uint32_t Count = 70;
  for (uint32_t i = 0; i < Count; ++i) {
    add_pass((i < 10 ? "p0" : "p") + std::to_string(i), Stage::PrePost, 0, i % 3 == 0);
  }
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(Count));

  record_stage(Stage::PrePost);

  ASSERT_EQ(frame.stages.size(), 1u);
  const auto& stage = frame.stages[0];
  EXPECT_EQ(stage.passCount, Count);
  EXPECT_EQ(stage.enabled.size(), 2u);
  std::vector<std::string> expected;
  for (uint32_t i = 0; i < Count; ++i) {
    EXPECT_EQ(stage.is_enabled(i), i % 3 == 0) << i;
    if (i % 3 == 0) {
      expected.push_back(passes[i].name);
    }
  }
  EXPECT_FALSE(stage.is_enabled(Count));
  EXPECT_EQ(run(stage), expected);
}

TEST_F(GfxRenderStageTest, EnabledSeesStageData) {
  auto& pass = passes.emplace_back(TestPass{.owner = this, .name = "checked"});
  ASSERT_NE(render::detail::register_pass({
                .name = "checked",
                .stage = Stage::OpaqueDone,
                .enabled = check_stage_data,
                .encode = encode_pass,
                .state = &pass,
            }),
            render::detail::InvalidPassId);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  frame.frameId = 7;
  frame.frameIndex = 3;

  record_stage(Stage::OpaqueDone);

  EXPECT_TRUE(stageDataChecked);
  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
}

TEST_F(GfxRenderStageTest, MissingMarkerRunsAtNextMarker) {
  add_pass("scene", Stage::SceneBegin, 0, true, true);
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 2u);
  EXPECT_EQ(frame.stages[0].stage, Stage::SceneBegin);
  EXPECT_FALSE(frame.stages[0].markerSeen);
  EXPECT_FALSE(frame.stages[0].deferred);
  EXPECT_EQ(frame.stages[1].stage, Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages[1].markerSeen);
  EXPECT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"scene"}));
}

TEST_F(GfxRenderStageTest, MissingMarkerSkipsMarkerBoundPasses) {
  add_pass("scene", Stage::SceneBegin);
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"opaque"}));
}

TEST_F(GfxRenderStageTest, LaterStagesRunWithoutMarker) {
  add_pass("post", Stage::PrePost);
  add_pass("hud", Stage::BeforeHud);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  record_stage(Stage::SceneEnd);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::PrePost);
  EXPECT_FALSE(frame.stages[0].markerSeen);
  EXPECT_FALSE(frame.stages[0].flushedAtFinish);

  finish();

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp, PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 2u);
  EXPECT_EQ(frame.stages[1].stage, Stage::BeforeHud);
  EXPECT_FALSE(frame.stages[1].markerSeen);
  EXPECT_TRUE(frame.stages[1].flushedAtFinish);
}

TEST_F(GfxRenderStageTest, FinishRunsRemainingStagesAfterFinalPass) {
  add_pass("end", Stage::FrameEnd);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  finish();

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::FrameEnd);
  EXPECT_TRUE(frame.stages[0].flushedAtFinish);
  EXPECT_FALSE(frame.stages[0].markerSeen);
  ASSERT_EQ(frame.renderPasses.size(), 1u);
  EXPECT_TRUE(frame.renderPasses[0].captureDepthSnapshot);
  EXPECT_EQ(frame.ops[0].renderPass, &frame.renderPasses[0]);
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxRenderStageTest, OffscreenStageDefersUntilRestore) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  seed(320, 180);
  begin_offscreen(320, 180);

  record_stage(Stage::OpaqueDone);

  EXPECT_TRUE(frame.stages.empty());
  EXPECT_EQ(std::ranges::count(op_types(), StageOp), 0);

  end_offscreen();

  ASSERT_EQ(op_types(), (std::vector{PassOp, PassOp, StageOp}));
  EXPECT_TRUE(frame.ops[0].renderPass->label.starts_with("Offscreen"));
  EXPECT_TRUE(frame.ops[1].renderPass->label.starts_with("EFB"));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].markerSeen);
  EXPECT_TRUE(frame.stages[0].deferred);
  EXPECT_FALSE(frame.stages[0].flushedAtFinish);
  EXPECT_EQ(count_efb_passes(), 2u);
  EXPECT_FALSE(frame.renderPasses.back().sealed);
  EXPECT_FALSE(frame.renderPasses.back().clearDepth);
}

TEST_F(GfxRenderStageTest, ChainedOffscreensFlushOnce) {
  add_pass("scene", Stage::SceneBegin);
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  seed(320, 180);
  seed(160, 90);

  begin_offscreen(320, 180);
  record_stage(Stage::SceneBegin);
  begin_offscreen(160, 90);
  record_stage(Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages.empty());
  end_offscreen();

  ASSERT_EQ(op_types(), (std::vector{PassOp, PassOp, PassOp, StageOp, StageOp}));
  EXPECT_TRUE(frame.ops[2].renderPass->label.starts_with("EFB"));
  ASSERT_EQ(frame.stages.size(), 2u);
  EXPECT_EQ(frame.stages[0].stage, Stage::SceneBegin);
  EXPECT_EQ(frame.stages[1].stage, Stage::OpaqueDone);
  for (const auto& stage : frame.stages) {
    EXPECT_TRUE(stage.markerSeen);
    EXPECT_TRUE(stage.deferred);
  }
  EXPECT_EQ(count_efb_passes(), 2u);
}

TEST_F(GfxRenderStageTest, ResolvePassFromOffscreenFlushes) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  seed(320, 180);
  ASSERT_TRUE(create_pass(320, 180));

  record_stage(Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages.empty());
  ResolvedTargets out;
  ASSERT_TRUE(resolve_pass({.color = false, .depth = false}, out));

  EXPECT_FALSE(is_offscreen());
  ASSERT_EQ(op_types(), (std::vector{PassOp, PassOp, StageOp}));
  EXPECT_TRUE(frame.ops[1].renderPass->label.starts_with("EFB"));
  EXPECT_TRUE(frame.ops[1].renderPass->sealed);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].deferred);
}

TEST_F(GfxRenderStageTest, StageWithoutOpenPassRunsAtFinish) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  queue_texture_copy({}, {}, {1, 1, 1});

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, CopyOp}));
  EXPECT_TRUE(frame.stages.empty());

  finish();

  EXPECT_EQ(op_types(), (std::vector{PassOp, CopyOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages[0].markerSeen);
  EXPECT_TRUE(frame.stages[0].deferred);
  EXPECT_TRUE(frame.stages[0].flushedAtFinish);
}

TEST_F(GfxRenderStageTest, RepeatedAndLateMarkersAreIgnored) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  record_stage(Stage::OpaqueDone);
  ASSERT_EQ(frame.ops.size(), 2u);
  record_stage(Stage::OpaqueDone);
  record_stage(Stage::SceneBegin);

  EXPECT_EQ(frame.ops.size(), 2u);
  EXPECT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.renderPasses.size(), 2u);
}

TEST_F(GfxRenderStageTest, DeferredMarkerBlocksEarlierStage) {
  add_pass("scene", Stage::SceneBegin);
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  seed(320, 180);
  begin_offscreen(320, 180);

  record_stage(Stage::OpaqueDone);
  record_stage(Stage::SceneBegin);
  end_offscreen();

  EXPECT_EQ(std::ranges::count(op_types(), StageOp), 1);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages[0].markerSeen);
  EXPECT_TRUE(frame.stages[0].deferred);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"opaque"}));
}

TEST_F(GfxRenderStageTest, AfterFrameIsNotAMarker) {
  record_stage(Stage::AfterFrame);
  record_stage(static_cast<Stage>(200));
  EXPECT_TRUE(frame.ops.empty());

  add_pass("scene", Stage::SceneBegin, 0, true, true);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  record_stage(Stage::AfterFrame);
  record_stage(static_cast<Stage>(200));
  EXPECT_TRUE(frame.ops.empty());
  EXPECT_TRUE(frame.stages.empty());

  record_stage(Stage::SceneBegin);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].markerSeen);
}

TEST_F(GfxRenderStageTest, RegistrationAppliesAtNextLatch) {
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
  ASSERT_NE(add_pass("scene", Stage::SceneBegin), render::detail::InvalidPassId);

  record_stage(Stage::SceneBegin);
  EXPECT_TRUE(frame.ops.empty());

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  record_stage(Stage::SceneBegin);
  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
}

TEST_F(GfxRenderStageTest, GraphPointerStableWithoutChanges) {
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
  const auto first = frame.renderState;
  EXPECT_EQ(first->graph, render::detail::empty_graph());
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
  EXPECT_EQ(frame.renderState, first);

  const auto id = add_pass("scene", Stage::SceneBegin);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto registered = frame.renderState;
  EXPECT_NE(registered->graph, render::detail::empty_graph());
  EXPECT_NE(registered->graph->graphHash, 0u);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  EXPECT_EQ(frame.renderState, registered);

  render::detail::unregister_pass(id);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
  EXPECT_EQ(frame.renderState->graph, render::detail::empty_graph());
  EXPECT_EQ(frame.renderState->graph->graphHash, 0u);
}

TEST_F(GfxRenderStageTest, ReRegisteredPassReplacesGraph) {
  const auto id = add_pass("a", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto first = frame.renderState->graph;

  render::detail::unregister_pass(id);
  const auto again = add_pass("a", Stage::OpaqueDone);
  ASSERT_NE(again, render::detail::InvalidPassId);
  EXPECT_NE(again, id);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto second = frame.renderState->graph;

  EXPECT_NE(second, first);
  EXPECT_EQ(second->graphHash, first->graphHash);
  EXPECT_EQ(second->passes[0].desc.state, &passes.back());
  EXPECT_NE(first->passes[0].desc.state, second->passes[0].desc.state);
}

TEST_F(GfxRenderStageTest, RegisterRejectsInvalidPasses) {
  using render::detail::InvalidPassId;
  using render::detail::register_pass;
  EXPECT_EQ(register_pass({.encode = encode_pass}), InvalidPassId);
  EXPECT_EQ(register_pass({.name = "no encode"}), InvalidPassId);
  EXPECT_EQ(register_pass({.name = "native", .kind = render::detail::PassKind::Native, .encode = encode_pass}),
            InvalidPassId);
  EXPECT_EQ(register_pass({.name = "after", .stage = Stage::AfterFrame, .encode = encode_pass}), InvalidPassId);
  const auto kept = add_pass("kept", Stage::PrePost);
  ASSERT_NE(kept, InvalidPassId);
  EXPECT_EQ(add_pass("kept", Stage::SceneBegin), InvalidPassId);

  const auto stale = add_pass("stale", Stage::PrePost);
  ASSERT_NE(stale, InvalidPassId);
  render::detail::unregister_pass(stale);
  const auto reused = add_pass("reused", Stage::PrePost);
  ASSERT_NE(reused, InvalidPassId);
  render::detail::unregister_pass(InvalidPassId);
  render::detail::unregister_pass(stale);

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_EQ(frame.renderState->graph->passes[0].id, kept);
  EXPECT_EQ(frame.renderState->graph->passes[1].id, reused);
}

TEST_F(GfxRenderStageTest, StageDataAddressesStayValid) {
  constexpr std::array Order{Stage::SceneBegin, Stage::OpaqueDone, Stage::PrePost,
                             Stage::SceneEnd,   Stage::BeforeHud,  Stage::FrameEnd};
  for (const auto stage : Order) {
    add_pass(std::string{render::stage_name(stage)}, stage);
  }
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(Order.size()));

  for (const auto stage : Order) {
    record_stage(stage);
  }

  ASSERT_EQ(frame.stages.size(), Order.size());
  size_t next = 0;
  for (const auto& op : frame.ops) {
    if (op.type != StageOp) {
      EXPECT_EQ(op.stageData, nullptr);
      continue;
    }
    ASSERT_LT(next, frame.stages.size());
    EXPECT_EQ(op.index, next);
    EXPECT_EQ(op.stageData, &frame.stages[next]);
    EXPECT_EQ(op.stageData->stage, Order[next]);
    ++next;
  }
  EXPECT_EQ(next, Order.size());
}
#endif

} // namespace
} // namespace aurora::gfx

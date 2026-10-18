#include <gtest/gtest.h>

#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/lru_cache.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture.hpp"
#if AURORA_ENABLE_RENDER
#include "hint_bytes.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/graph_api.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_layout.hpp"
#include "render/hints.hpp"
#include "render/scene/classify.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/stage.hpp"
#endif
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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
  std::function<void(render::detail::InternalPassBuilder&)> declare;
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
    render::detail::testing::reset_external_passes();
    GfxRecordingTest::SetUp();
  }

  void TearDown() override {
    GfxRecordingTest::TearDown();
    render::detail::testing::reset();
    render::detail::testing::reset_external_passes();
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
    const auto& nodes = frame.renderState->graph->nodes;
    ASSERT_EQ(static_cast<size_t>(
                  std::ranges::count(nodes, render::detail::NodeKind::Pass, &render::detail::CompiledNode::kind)),
              expectedPasses);
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

  render::detail::PassId add_declared(std::string name, uint16_t order, bool enabled,
                                      std::function<void(render::detail::InternalPassBuilder&)> declare) {
    auto& pass =
        passes.emplace_back(TestPass{.owner = this, .name = name, .enabled = enabled, .declare = std::move(declare)});
    return render::detail::register_pass({
        .name = std::move(name),
        .stage = Stage::OpaqueDone,
        .order = order,
        .declare = declare_test_pass,
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
    const auto marker = [this, markers](Stage stage) {
      if (markers && hintMarkers) {
        aurora::test::process_hints(aurora::test::stage_hint(stage));
      } else if (markers) {
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

  static void declare_side_effect(render::detail::InternalPassBuilder& builder, const render::detail::PassSetup&,
                                  const void*) {
    builder.side_effect();
  }

  static void declare_test_pass(render::detail::InternalPassBuilder& builder, const render::detail::PassSetup&,
                                const void* state) {
    static_cast<const TestPass*>(state)->declare(builder);
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
  bool hintMarkers = false;
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
  EXPECT_EQ(second->nodes[0].state, &passes.back());
  EXPECT_NE(first->nodes[0].state, second->nodes[0].state);
}

TEST_F(GfxRenderStageTest, RecompilePublishesNewGraph) {
  using render::detail::InvalidPassId;
  const auto register_declared = [this](const std::string& stateName) {
    auto& pass = passes.emplace_back(TestPass{.owner = this, .name = stateName});
    return render::detail::register_pass({
        .name = "a",
        .stage = Stage::OpaqueDone,
        .declare = declare_side_effect,
        .encode = encode_pass,
        .state = &pass,
    });
  };
  const auto id = register_declared("s1");
  ASSERT_NE(id, InvalidPassId);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto first = frame.renderState->graph;

  render::detail::unregister_pass(id);
  ASSERT_NE(register_declared("s2"), InvalidPassId);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto second = frame.renderState->graph;

  EXPECT_NE(second, first);
  EXPECT_EQ(second->graphHash, first->graphHash);
  const auto node = second->find_node("a");
  ASSERT_NE(node, render::detail::InvalidNode);
  EXPECT_EQ(second->nodes[node].state, &passes.back());
  EXPECT_FALSE(second->nodes[node].opaque);

  record_stage(Stage::OpaqueDone);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"s2"}));
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
  EXPECT_EQ(frame.renderState->graph->nodes[0].id, kept);
  EXPECT_EQ(frame.renderState->graph->nodes[1].id, reused);
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

constexpr render::detail::TexDesc StageTexture{.format = wgpu::TextureFormat::RGBA8Unorm};

void create_stage_texture(render::detail::InternalPassBuilder& b) {
  b.create("t", StageTexture, render::detail::Access::StorageWrite);
}

TEST_F(GfxRenderStageTest, StructurallyCulledStageEmitsNoOp) {
  ASSERT_NE(add_declared("lonely", 0, true, create_stage_texture), render::detail::InvalidPassId);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto& graph = *frame.renderState->graph;
  EXPECT_TRUE(graph.stage_empty(Stage::OpaqueDone));
  EXPECT_EQ(graph.nodes[graph.find_node("lonely")].status, render::detail::CullReason::NoConsumer);

  record_stage(Stage::OpaqueDone);

  EXPECT_TRUE(frame.ops.empty());
  EXPECT_TRUE(frame.stages.empty());
  EXPECT_EQ(frame.renderPasses.size(), 1u);
}

void read_stage_texture(render::detail::InternalPassBuilder& b, render::detail::Missing missing) {
  b.use("t", render::detail::Access::Sampled, missing);
  b.side_effect();
}

TEST_F(GfxRenderStageTest, PerFrameCullSuppressesOp) {
  add_declared("w", 0, false, create_stage_texture);
  add_declared("r", 1, true, [](auto& b) { read_stage_texture(b, render::detail::Missing::DisablePass); });
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(std::ranges::count(op_types(), StageOp), 0);
  EXPECT_TRUE(frame.stages.empty());

  passes[0].enabled = true;
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].is_enabled(0));
  EXPECT_TRUE(frame.stages[0].is_enabled(1));
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"w", "r"}));
}

TEST_F(GfxRenderStageTest, DummyBitsReachStageData) {
  add_declared("w", 0, false, create_stage_texture);
  add_declared("r", 1, true, [](auto& b) { read_stage_texture(b, render::detail::Missing::BindDummyBlack); });
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  const auto& stage = frame.stages[0];
  EXPECT_FALSE(stage.is_enabled(0));
  EXPECT_TRUE(stage.is_enabled(1));
  const auto& graph = *frame.renderState->graph;
  const auto uses = graph.node_uses(graph.find_node("r"));
  ASSERT_EQ(uses.size(), 1u);
  EXPECT_TRUE(stage.dummy_bound(uses[0].stageUse));
  EXPECT_EQ(run(stage), (std::vector<std::string>{"r"}));
}

TEST_F(GfxRenderStageTest, StagesPushUniformsOnlyForPasses) {
  constexpr std::array Markers{Stage::SceneBegin, Stage::OpaqueDone, Stage::PrePost,
                               Stage::SceneEnd,   Stage::BeforeHud,  Stage::FrameEnd};
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(0));
  frame.frameId = 11;
  const size_t empty = frame.uniforms.size();
  for (const auto stage : Markers) {
    record_stage(stage);
  }
  EXPECT_TRUE(frame.stages.empty());
  EXPECT_EQ(frame.uniforms.size(), empty);
  EXPECT_FALSE(render::detail::testing::frame_objects_created());

  add_pass("pushes", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  frame.frameId = 12;
  const size_t before = frame.uniforms.size();
  for (const auto stage : Markers) {
    record_stage(stage);
  }
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_GE(frame.uniforms.size(), before + render::detail::FrameUniformsSize);
  EXPECT_GE(frame.stages[0].bindings.constantsOffset, before);
  EXPECT_TRUE(frame.stages[0].bindings.params.empty());
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"pushes"}));
}

class GfxHistoryStageTest : public GfxRenderStageTest {
protected:
  void add_history_passes() {
    add_declared("h", 0, true, [](auto& b) { b.history("hist", StageTexture, render::detail::Access::StorageWrite); });
    add_declared("p", 1, true, [](auto& b) {
      b.use_prev("hist");
      b.side_effect();
    });
  }

  bool prev_valid_at_marker() {
    record_stage(Stage::OpaqueDone);
    EXPECT_EQ(frame.stages.size(), 1u);
    if (frame.stages.empty()) {
      return false;
    }
    const auto& graph = *frame.renderState->graph;
    const auto slot = graph.resources[graph.find_resource("hist")].historySlot;
    EXPECT_NE(slot, render::detail::InvalidResource);
    return frame.stages[0].prev_valid(slot);
  }
};

TEST_F(GfxHistoryStageTest, PrevBitsAcrossFrames) {
  add_history_passes();
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_FALSE(prev_valid_at_marker());
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_TRUE(prev_valid_at_marker());

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  render::detail::invalidate_frame_histories();
  EXPECT_FALSE(prev_valid_at_marker());
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_TRUE(prev_valid_at_marker());
}

void declare_external_sink(render::PassBuilder& builder, void*) { builder.side_effect(); }

void encode_external_a(render::PassContext&, void*) {}

void encode_external_b(render::PassContext&, void*) {}

render::ExternalPassDesc external_sink(render::PassEncodeCallback encode) {
  return {
      .name = "mod.pass",
      .kind = render::PassKind::Compute,
      .declare = declare_external_sink,
      .encode = encode,
  };
}

TEST_F(GfxRenderStageTest, SteadyLatchesKeepGraphAndEpoch) {
  add_pass("steady", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto state = frame.renderState;
  const auto graph = state->graph;
  const uint64_t hash = graph->graphHash;
  const uint64_t epoch = render::detail::external_pass_epoch();
  EXPECT_EQ(graph->externals, nullptr);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
    EXPECT_EQ(frame.renderState, state) << i;
    EXPECT_EQ(frame.renderState->graph, graph) << i;
    EXPECT_EQ(frame.renderState->graph->graphHash, hash) << i;
    EXPECT_EQ(render::detail::external_pass_epoch(), epoch) << i;
  }
  EXPECT_TRUE(render::detail::untraceable_external_pass().empty());
}

TEST_F(GfxRenderStageTest, ExternalPassRunsWithoutMarkers) {
  ASSERT_NE(render::register_pass(external_sink(encode_external_a)), render::InvalidPass);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  finish();

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  const auto& stage = frame.stages[0];
  EXPECT_EQ(stage.stage, Stage::BeforeHud);
  EXPECT_FALSE(stage.markerSeen);
  EXPECT_TRUE(stage.flushedAtFinish);
  EXPECT_TRUE(stage.is_enabled(0));
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxRenderStageTest, ReRegisterSameDeclarationAdoptsNewGraph) {
  const auto first = render::register_pass(external_sink(encode_external_a));
  ASSERT_NE(first, render::InvalidPass);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto before = frame.renderState->graph;

  render::unregister_pass(first);
  const auto second = render::register_pass(external_sink(encode_external_b));
  ASSERT_NE(second, render::InvalidPass);
  ASSERT_NE(second, first);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto after = frame.renderState->graph;

  EXPECT_NE(after, before);
  EXPECT_EQ(after->graphHash, before->graphHash);
  const auto node = after->find_node("mod.pass");
  ASSERT_NE(node, render::detail::InvalidNode);
  const auto& state = *static_cast<const render::detail::ExternalNodeState*>(after->nodes[node].state);
  EXPECT_EQ(state.id, second);
  EXPECT_EQ(state.desc.encode, encode_external_b);
  EXPECT_EQ(after->nodes[node].id, second);

  finish();
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].is_enabled(0));
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxHistoryStageTest, SameHashRecompileKeepsCullState) {
  add_history_passes();
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_FALSE(prev_valid_at_marker());
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  EXPECT_TRUE(prev_valid_at_marker());
  const auto before = frame.renderState->graph;

  render::apply_settings({.legacyBlend = 0.5f});
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));

  EXPECT_NE(frame.renderState->graph, before);
  EXPECT_EQ(frame.renderState->graph->graphHash, before->graphHash);
  EXPECT_TRUE(prev_valid_at_marker());
}

using render::DepthSource;

constexpr std::array AllDepthSources{DepthSource::Unavailable, DepthSource::Direct, DepthSource::Copy};

struct PassDetail {
  std::string label;
  bool sealed = false;
  bool discardable = false;
  bool captureDepthSnapshot = false;
  bool depthSnapshot = false;
  bool clearDepth = false;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;

  bool operator==(const PassDetail&) const = default;
};

struct RecordedRun {
  FrameShape shape;
  std::vector<PassDetail> passes;
  size_t uniformBytes = 0;
  size_t stages = 0;

  bool operator==(const RecordedRun&) const = default;
};

TEST_F(GfxRenderStageTest, SceneDepthSourceChangeRecompiles) {
  ASSERT_NE(render::register_pass(external_sink(encode_external_a)), render::InvalidPass);
  detail::testing::override_scene_depth_source(DepthSource::Direct);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  const auto direct = frame.renderState->graph;
  EXPECT_EQ(direct->sceneDepthSource, DepthSource::Direct);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  EXPECT_EQ(frame.renderState->graph, direct);

  detail::testing::override_scene_depth_source(DepthSource::Copy);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  EXPECT_NE(frame.renderState->graph, direct);
  EXPECT_EQ(frame.renderState->graph->sceneDepthSource, DepthSource::Copy);
}

// Real snapshots stay on here, so any attempt to make one would fail without a device
TEST_F(GfxRenderStageTest, FramesWithoutDepthSamplersIgnoreTheSource) {
  const auto recorded = [this] {
    RecordedRun run{.shape = frame_shape(), .uniformBytes = frame.uniforms.size(), .stages = frame.stages.size()};
    for (const auto& pass : frame.renderPasses) {
      run.passes.push_back({
          .label = pass.label,
          .sealed = pass.sealed,
          .discardable = pass.discardable,
          .captureDepthSnapshot = pass.captureDepthSnapshot,
          .depthSnapshot = static_cast<bool>(pass.snapshotDepthDst),
          .clearDepth = pass.clearDepth,
          .depthStoreOp = pass.depthStoreOp,
      });
    }
    return run;
  };
  seed(320, 180);
  std::vector<RecordedRun> legacy;
  for (const auto source : AllDepthSources) {
    detail::testing::override_scene_depth_source(source);
    FrameShape shape;
    ASSERT_NO_FATAL_FAILURE(run_marker_script(true, shape));
    legacy.push_back(recorded());
  }
  EXPECT_EQ(legacy[1], legacy[0]);
  EXPECT_EQ(legacy[2], legacy[0]);

  add_pass("opaque", Stage::OpaqueDone);
  std::vector<RecordedRun> staged;
  for (const auto source : AllDepthSources) {
    detail::testing::override_scene_depth_source(source);
    ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
    // Frame constants are pushed once per frame id, so each run needs its own
    frame.frameId = 100 + staged.size();
    record_stage(Stage::OpaqueDone);
    finish();
    staged.push_back(recorded());
    ASSERT_EQ(frame.stages.size(), 1u);
    EXPECT_FALSE(frame.stages[0].sceneDepth->sampled);
  }
  EXPECT_EQ(staged[1], staged[0]);
  EXPECT_EQ(staged[2], staged[0]);
  for (uint8_t stage = 0; stage < render::StageCount; ++stage) {
    EXPECT_EQ(detail::testing::stage_depth_snapshot_requests(static_cast<Stage>(stage)), 0u);
    EXPECT_EQ(detail::testing::stage_depth_snapshot_generation(static_cast<Stage>(stage)), 0u);
  }
  for (const auto& pass : legacy[0].passes) {
    EXPECT_FALSE(pass.depthSnapshot) << pass.label;
  }
}

struct DepthReader {
  render::Missing missing = render::Missing::DisablePass;
  bool enabled = true;
  bool samples = true;
};

void declare_depth_reader(render::PassBuilder& builder, void* userdata) {
  const auto& reader = *static_cast<const DepthReader*>(userdata);
  if (reader.samples) {
    builder.use(render::resource::GxDepth, render::Access::Sampled, reader.missing);
  }
  builder.side_effect();
}

bool depth_reader_enabled(std::span<const std::byte>, void* userdata) {
  return static_cast<const DepthReader*>(userdata)->enabled;
}

class GfxSceneDepthTest : public GfxRenderStageTest {
protected:
  void SetUp() override {
    GfxRenderStageTest::SetUp();
    detail::testing::fake_depth_snapshots(true);
  }

  void add_reader(const char* name, Stage stage, DepthReader reader = {}) {
    DepthReader& stored = readers.emplace_back(reader);
    ASSERT_NE(render::register_pass({
                  .name = name,
                  .stage = stage,
                  .kind = render::PassKind::Compute,
                  .declare = declare_depth_reader,
                  .enabled = depth_reader_enabled,
                  .encode = encode_external_a,
                  .userdata = &stored,
              }),
              render::InvalidPass);
  }

  static render::detail::GraphEnvironment depth_environment() {
    render::detail::GraphEnvironment env;
    env.efbSize = {webgpu::g_frameBuffer.size.width, webgpu::g_frameBuffer.size.height};
    env.outputSize = env.efbSize;
    const auto color = static_cast<size_t>(render::detail::GxResource::Color);
    const auto depth = static_cast<size_t>(render::detail::GxResource::Depth);
    env.gxFormats[color] = ColorFormat;
    env.gxUsage[color] = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                         wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    env.gxFormats[depth] = DepthFormat;
    env.gxUsage[depth] = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
    return env;
  }

  // The latch takes its environment from the device, which these tests lack, so the frame's graph is compiled here
  void compile_new_frame(DepthSource source) {
    finish();
    detail::end_recording();
    detail::testing::override_scene_depth_source(source);
    render::detail::latch_external_passes();
    const auto set = render::detail::snapshot_external_passes();
    std::vector<render::detail::PassDesc> descs;
    render::detail::append_external_passes(*set, &render::detail::external_encode, descs);
    const auto graph = render::detail::compile_graph({
        .passes = descs,
        .env = depth_environment(),
        .externals = set,
        .sceneDepthSource = source,
    });
    frame = {};
    frame.renderState = std::make_shared<const render::detail::FrameState>(render::detail::FrameState{.graph = graph});
    detail::begin_recording(frame, 0);
  }

  static uint32_t requests(Stage stage) { return detail::testing::stage_depth_snapshot_requests(stage); }
  static uint32_t generation(Stage stage) { return detail::testing::stage_depth_snapshot_generation(stage); }

  std::deque<DepthReader> readers;
};

TEST_F(GfxSceneDepthTest, CopySourceRequestsSnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));

  record_stage(Stage::OpaqueDone);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(requests(Stage::OpaqueDone), 1u);
  EXPECT_EQ(generation(Stage::OpaqueDone), 1u);
  const auto& depth = frame.stages[0].sceneDepth;
  ASSERT_NE(depth, nullptr);
  EXPECT_EQ(depth->source, DepthSource::Copy);
  EXPECT_EQ(depth->sampledFormat, wgpu::TextureFormat::R32Float);
  EXPECT_EQ(depth->size.width, 640u);
  EXPECT_EQ(depth->size.height, 480u);
}

TEST_F(GfxSceneDepthTest, CaptureFollowsTheLatchedSource) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));
  detail::testing::override_scene_depth_source(DepthSource::Direct);

  record_stage(Stage::OpaqueDone);

  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].sceneDepth->source, DepthSource::Copy);
  EXPECT_EQ(frame.stages[0].sceneDepth->sampledFormat, wgpu::TextureFormat::R32Float);
  EXPECT_EQ(requests(Stage::OpaqueDone), 1u);
}

TEST_F(GfxSceneDepthTest, BackToBackSamplingStagesEachRequestSnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.post", Stage::PrePost));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));

  record_stage(Stage::OpaqueDone);
  EXPECT_EQ(requests(Stage::OpaqueDone), 1u);
  EXPECT_EQ(requests(Stage::PrePost), 0u);
  record_stage(Stage::PrePost);

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp, PassOp, StageOp}));
  ASSERT_EQ(frame.renderPasses.size(), 3u);
  const auto& continuation = frame.renderPasses[1];
  EXPECT_TRUE(continuation.sealed);
  EXPECT_FALSE(continuation.has_content());
  EXPECT_EQ(frame.ops[2].renderPass, &continuation);
  EXPECT_EQ(requests(Stage::PrePost), 1u);
  EXPECT_EQ(generation(Stage::OpaqueDone), 1u);
  EXPECT_EQ(generation(Stage::PrePost), 1u);
  ASSERT_EQ(frame.stages.size(), 2u);
  EXPECT_NE(frame.stages[0].sceneDepth, frame.stages[1].sceneDepth);
  EXPECT_EQ(frame.stages[1].sceneDepth->sampledFormat, wgpu::TextureFormat::R32Float);
}

TEST_F(GfxSceneDepthTest, StagesInOneBatchShareOneSnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.hud", Stage::BeforeHud));
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.end", Stage::FrameEnd));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));

  finish();

  EXPECT_EQ(requests(Stage::BeforeHud), 1u);
  EXPECT_EQ(requests(Stage::FrameEnd), 0u);
  ASSERT_EQ(frame.stages.size(), 2u);
  EXPECT_EQ(frame.stages[0].stage, Stage::BeforeHud);
  EXPECT_EQ(frame.stages[1].stage, Stage::FrameEnd);
  ASSERT_NE(frame.stages[0].sceneDepth, nullptr);
  EXPECT_EQ(frame.stages[0].sceneDepth, frame.stages[1].sceneDepth);
  EXPECT_EQ(frame.stages[1].sceneDepth->sampledFormat, wgpu::TextureFormat::R32Float);
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxSceneDepthTest, FinishBatchCapturesOnFinalPass) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.hud", Stage::BeforeHud));
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.end", Stage::FrameEnd));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));

  finish();

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp, StageOp}));
  ASSERT_EQ(frame.renderPasses.size(), 1u);
  EXPECT_TRUE(frame.renderPasses[0].captureDepthSnapshot);
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_EQ(frame.ops[0].renderPass, &frame.renderPasses[0]);
  EXPECT_EQ(requests(Stage::BeforeHud), 1u);
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxSceneDepthTest, FinishWithoutOpenPassHasNoCopySnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.hud", Stage::BeforeHud));
  for (const auto source : {DepthSource::Copy, DepthSource::Direct}) {
    ASSERT_NO_FATAL_FAILURE(compile_new_frame(source));
    queue_texture_copy({}, {}, {1, 1, 1});

    finish();

    EXPECT_EQ(op_types(), (std::vector{PassOp, CopyOp, StageOp}));
    ASSERT_EQ(frame.stages.size(), 1u);
    const auto& depth = *frame.stages[0].sceneDepth;
    EXPECT_EQ(depth.source, source);
    EXPECT_EQ(depth.attachmentFormat, DepthFormat);
    EXPECT_EQ(depth.size.width, 640u);
    if (source == DepthSource::Copy) {
      EXPECT_FALSE(depth.sampled);
      EXPECT_EQ(depth.sampledFormat, wgpu::TextureFormat::Undefined);
    } else {
      EXPECT_EQ(depth.sampledFormat, DepthFormat);
    }
  }
  EXPECT_EQ(requests(Stage::BeforeHud), 0u);
  EXPECT_EQ(generation(Stage::BeforeHud), 0u);
}

TEST_F(GfxSceneDepthTest, DirectSourceAddsNoSnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Direct));
  record_stage(Stage::OpaqueDone);
  const auto sampling = op_types();
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].sceneDepth->source, DepthSource::Direct);
  EXPECT_EQ(frame.stages[0].sceneDepth->sampledFormat, frame.renderPasses[0].depthStencilFormat);
  EXPECT_FALSE(frame.renderPasses[0].snapshotDepthDst);
  EXPECT_EQ(requests(Stage::OpaqueDone), 0u);
  EXPECT_EQ(generation(Stage::OpaqueDone), 0u);

  readers[0].samples = false;
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Direct));
  record_stage(Stage::OpaqueDone);
  EXPECT_EQ(op_types(), sampling);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].sceneDepth->sampledFormat, wgpu::TextureFormat::Undefined);
}

TEST_F(GfxSceneDepthTest, InactiveSamplerAddsNoSnapshot) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.plain", Stage::OpaqueDone, {.samples = false}));
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.sampler", Stage::OpaqueDone, {.enabled = false}));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));

  record_stage(Stage::OpaqueDone);

  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_FALSE(frame.stages[0].sceneDepth->sampled);
  EXPECT_EQ(frame.stages[0].sceneDepth->sampledFormat, wgpu::TextureFormat::Undefined);
  EXPECT_EQ(requests(Stage::OpaqueDone), 0u);
  EXPECT_EQ(generation(Stage::OpaqueDone), 0u);
}

TEST_F(GfxSceneDepthTest, AttachmentFieldsAlwaysFilled) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone, {.missing = render::Missing::BindDummyBlack}));
  for (const auto source : AllDepthSources) {
    ASSERT_NO_FATAL_FAILURE(compile_new_frame(source));
    record_stage(Stage::OpaqueDone);
    ASSERT_EQ(frame.stages.size(), 1u) << static_cast<int>(source);
    const auto& sealed = frame.renderPasses[0];
    const auto& depth = *frame.stages[0].sceneDepth;
    EXPECT_EQ(depth.source, source);
    EXPECT_EQ(depth.attachmentFormat, sealed.depthStencilFormat);
    EXPECT_EQ(depth.attachmentSamples, sealed.msaaSamples);
    EXPECT_EQ(depth.size.width, sealed.colorAttachments[SceneColorAttachmentIndex].size.width);
    EXPECT_EQ(depth.size.height, sealed.colorAttachments[SceneColorAttachmentIndex].size.height);
  }
}

TEST_F(GfxSceneDepthTest, SnapshotTextureReusedAcrossFrames) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  for (uint32_t i = 1; i <= 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));
    record_stage(Stage::OpaqueDone);
    EXPECT_EQ(generation(Stage::OpaqueDone), 1u) << i;
    EXPECT_EQ(requests(Stage::OpaqueDone), i);
  }
}

TEST_F(GfxSceneDepthTest, SnapshotRecreatedOnEfbResize) {
  ASSERT_NO_FATAL_FAILURE(add_reader("mod.opaque", Stage::OpaqueDone));
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));
  record_stage(Stage::OpaqueDone);
  EXPECT_EQ(generation(Stage::OpaqueDone), 1u);

  webgpu::g_frameBuffer.size = {1280, 720, 1};
  webgpu::g_depthBuffer.size = {1280, 720, 1};
  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));
  record_stage(Stage::OpaqueDone);
  EXPECT_EQ(generation(Stage::OpaqueDone), 2u);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].sceneDepth->size.width, 1280u);

  ASSERT_NO_FATAL_FAILURE(compile_new_frame(DepthSource::Copy));
  record_stage(Stage::OpaqueDone);
  EXPECT_EQ(generation(Stage::OpaqueDone), 2u);
}

std::vector<std::string> g_capturedWarnings;

void capture_warning(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  if (level == LOG_WARNING) {
    g_capturedWarnings.emplace_back(message, len);
  }
}

// Restores the previous log callback even when an assertion leaves the test early
class ScopedWarningCapture {
public:
  ScopedWarningCapture() noexcept : m_previous(aurora::g_config.logCallback) {
    g_capturedWarnings.clear();
    aurora::g_config.logCallback = &capture_warning;
  }
  ~ScopedWarningCapture() { aurora::g_config.logCallback = m_previous; }
  ScopedWarningCapture(const ScopedWarningCapture&) = delete;
  ScopedWarningCapture& operator=(const ScopedWarningCapture&) = delete;

  [[nodiscard]] size_t count(std::string_view text) const {
    return static_cast<size_t>(std::ranges::count_if(
        g_capturedWarnings, [text](const std::string& line) { return line.find(text) != std::string::npos; }));
  }

private:
  AuroraLogCallback m_previous;
};

class GfxRenderStageHintTest : public GfxRenderStageTest {
protected:
  void SetUp() override {
    GfxRenderStageTest::SetUp();
    gx::hints::testing::reset_counters();
    render::detail::testing::set_hints_override(1);
    render::detail::install_hints();
    // The missing-marker warning fires once per process, so repeated and whole-binary runs start it fresh
    render::detail::testing::reset_missing_marker_warnings();
  }

  void TearDown() override {
    GfxRenderStageTest::TearDown();
    render::detail::testing::reset_hints();
  }

  static bool needs_lights(const render::detail::StageData& stage, const void*) {
    return stage.sceneHints != nullptr && stage.sceneHints->has(render::detail::SceneHasLights);
  }
};

TEST_F(GfxRenderStageHintTest, HintMarkersWithEmptyGraphEmitNoOps) {
  seed(320, 180);
  hintMarkers = true;
  FrameShape withHints;
  ASSERT_NO_FATAL_FAILURE(run_marker_script(true, withHints));
  FrameShape withoutMarkers;
  ASSERT_NO_FATAL_FAILURE(run_marker_script(false, withoutMarkers));

  EXPECT_FALSE(withHints.ops.empty());
  EXPECT_EQ(withHints.ops, withoutMarkers.ops);
  EXPECT_EQ(withHints.passes, withoutMarkers.passes);
  EXPECT_EQ(render::hint_counters(GX_AURORA_RENDER_STAGE).decoded, 6u);
}

TEST_F(GfxRenderStageHintTest, HintMarkerSealsAndResumes) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));

  EXPECT_EQ(op_types(), (std::vector{PassOp, StageOp}));
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[1].sealed);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);
  EXPECT_TRUE(frame.stages[0].markerSeen);
  EXPECT_TRUE(frame.stages[0].is_enabled(0));
}

TEST_F(GfxRenderStageHintTest, StageDataCarriesSceneHints) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  aurora::test::process_hints(aurora::test::lights_hint(3));
  aurora::test::process_hints(aurora::test::environment_hint());
  aurora::test::process_hints(aurora::test::stage_hint(Stage::SceneBegin));
  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));

  ASSERT_EQ(frame.stages.size(), 1u);
  const auto& hints = frame.stages[0].sceneHints;
  ASSERT_NE(hints, nullptr);
  EXPECT_EQ(hints->lights.size(), 3u);
  EXPECT_TRUE(hints->has(render::detail::SceneHasLights | render::detail::SceneHasEnvironment));
  EXPECT_EQ(hints->environment.version, 2u);
}

TEST_F(GfxRenderStageHintTest, StageDataWithoutHintsHasEmptySnapshot) {
  add_pass("opaque", Stage::OpaqueDone);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));

  record_stage(Stage::SceneBegin);
  record_stage(Stage::OpaqueDone);

  ASSERT_EQ(frame.stages.size(), 1u);
  ASSERT_NE(frame.stages[0].sceneHints, nullptr);
  EXPECT_EQ(frame.stages[0].sceneHints->present, 0u);
}

TEST_F(GfxRenderStageHintTest, EnabledSeesSceneHints) {
  auto& pass = passes.emplace_back(TestPass{.owner = this, .name = "lit"});
  ASSERT_NE(render::detail::register_pass({
                .name = "lit",
                .stage = Stage::OpaqueDone,
                .enabled = needs_lights,
                .encode = encode_pass,
                .state = &pass,
            }),
            render::detail::InvalidPassId);
  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));
  EXPECT_TRUE(frame.stages.empty());

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(1));
  aurora::test::process_hints(aurora::test::lights_hint(1));
  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_TRUE(frame.stages[0].is_enabled(0));
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"lit"}));
}

TEST_F(GfxRenderStageHintTest, MissingMarkerWarnsOnlyWithMainCamera) {
  const ScopedWarningCapture warnings;
  add_pass("scene", Stage::SceneBegin);
  add_pass("opaque", Stage::OpaqueDone);

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  aurora::test::process_hints(aurora::test::camera_hint(AURORA_CAMERA_MAIN, false));
  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));
  EXPECT_EQ(warnings.count("marker missing"), 0u);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);

  ASSERT_NO_FATAL_FAILURE(latch_new_frame(2));
  aurora::test::process_hints(aurora::test::camera_hint(AURORA_CAMERA_MAIN, true));
  aurora::test::process_hints(aurora::test::stage_hint(Stage::OpaqueDone));
  EXPECT_EQ(warnings.count("marker missing"), 1u);
  ASSERT_EQ(frame.stages.size(), 1u);
  EXPECT_EQ(frame.stages[0].stage, Stage::OpaqueDone);
  EXPECT_EQ(run(frame.stages[0]), (std::vector<std::string>{"opaque"}));
}

TEST_F(GfxRenderStageHintTest, MarkerAcceptanceMatchesRecorder) {
  constexpr std::array MarkerStages{Stage::SceneBegin, Stage::OpaqueDone, Stage::PrePost,
                                    Stage::SceneEnd,   Stage::BeforeHud,  Stage::FrameEnd};
  // Passes that run without a marker make every stage emit, so markerSeen alone tells accepted from synthesized
  for (const Stage stage : MarkerStages) {
    add_pass(std::string{render::stage_name(stage)}, stage, 0, true, true);
  }
  seed(320, 180);
  enum class Step : uint8_t { Marker, BeginOffscreen, EndOffscreen };
  struct Action {
    Step step = Step::Marker;
    Stage stage = Stage::SceneBegin;
  };
  const auto mark = [](Stage stage) { return Action{.step = Step::Marker, .stage = stage}; };
  const Action begin{.step = Step::BeginOffscreen};
  const Action end{.step = Step::EndOffscreen};
  const std::vector<std::vector<Action>> sequences{
      {mark(Stage::SceneBegin), mark(Stage::OpaqueDone), mark(Stage::PrePost), mark(Stage::SceneEnd),
       mark(Stage::BeforeHud), mark(Stage::FrameEnd)},
      {mark(Stage::SceneBegin), mark(Stage::SceneBegin), mark(Stage::OpaqueDone)},
      {mark(Stage::OpaqueDone), mark(Stage::SceneBegin), mark(Stage::PrePost)},
      {mark(Stage::PrePost), mark(Stage::OpaqueDone), mark(Stage::SceneEnd), mark(Stage::SceneEnd)},
      {begin, mark(Stage::SceneBegin), end, mark(Stage::OpaqueDone)},
      {begin, mark(Stage::OpaqueDone), mark(Stage::SceneBegin), end, mark(Stage::PrePost)},
      {mark(Stage::FrameEnd), mark(Stage::BeforeHud)},
      {},
  };
  for (size_t i = 0; i < sequences.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(latch_new_frame(MarkerStages.size()));
    for (const Action& action : sequences[i]) {
      if (action.step == Step::BeginOffscreen) {
        begin_offscreen(320, 180);
      } else if (action.step == Step::EndOffscreen) {
        end_offscreen();
      } else {
        aurora::test::process_hints(aurora::test::stage_hint(action.stage));
      }
    }
    finish();
    uint32_t recorded = 0;
    for (const auto& data : frame.stages) {
      recorded |= data.markerSeen ? 1u << static_cast<uint32_t>(data.stage) : 0u;
    }
    uint32_t classified = 0;
    const render::scene::FrameStats& stats = render::scene::testing::classifier().stats();
    for (const Stage stage : MarkerStages) {
      classified |= stats.markerDraw[static_cast<size_t>(stage)] >= 0 ? 1u << static_cast<uint32_t>(stage) : 0u;
    }
    EXPECT_EQ(frame.stages.size(), MarkerStages.size()) << "sequence " << i;
    EXPECT_EQ(classified, recorded) << "sequence " << i;
    EXPECT_EQ(stats.hints.markers, static_cast<uint32_t>(std::popcount(recorded))) << "sequence " << i;
  }
}
#endif

} // namespace
} // namespace aurora::gfx

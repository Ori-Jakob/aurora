#include <gtest/gtest.h>

#include "gfx/pipeline_cache.hpp"
#include "gfx/scene_targets.hpp"
#include "gx/pipeline.hpp"

#include <cstdint>

namespace aurora::gfx {
namespace {
using detail::PipelinePriority;

SceneLayoutBase test_base() {
  return {.color = wgpu::TextureFormat::BGRA8Unorm, .depth = wgpu::TextureFormat::Depth32Float, .size = {64, 64, 1}};
}

// Every warmed pipeline resolved over two matched frames, so only the switch rules decide
SceneLayoutInput warming_input() {
  return {.want = true,
          .live = false,
          .warming = true,
          .supported = true,
          .progress = {.ready = 4, .total = 4},
          .matchedFrames = SceneLayoutMatchedFrames};
}

gx::PipelineConfig stochastic_config(uint32_t dstAlpha) {
  gx::PipelineConfig config{};
  config.dstAlpha = dstAlpha;
  config.shaderConfig.ext.stochasticMask = 1;
  return config;
}

class SceneLayoutWarmTest : public ::testing::Test {
protected:
  void TearDown() override {
    detail::reset_scene_layout();
    // Warm requests only queue here, so shutting the cache down drops them unbuilt
    shutdown_pipeline_cache();
  }
};

TEST(SceneLayout, FirstLatchAppliesDirectly) {
  EXPECT_EQ(next_scene_layout_action({.want = true, .firstLatch = true, .supported = true}), SceneLayoutAction::Switch);
  EXPECT_EQ(next_scene_layout_action({.live = true, .firstLatch = true, .supported = true}), SceneLayoutAction::Switch);
  EXPECT_EQ(next_scene_layout_action({.want = true, .live = true, .firstLatch = true, .supported = true}),
            SceneLayoutAction::None);
  EXPECT_EQ(next_scene_layout_action({.want = true, .supported = true}), SceneLayoutAction::BeginWarm);
}

TEST(SceneLayout, SwitchWaitsForEveryWarmPipeline) {
  SceneLayoutInput input = warming_input();
  input.progress = {.ready = 3, .total = 4};
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::None);
  input.progress.ready = 4;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, FailedVariantsCountAsResolved) {
  SceneLayoutInput input = warming_input();
  input.progress = {.ready = 3, .total = 4, .failed = 1};
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, HardFailureRefusesSwitch) {
  SceneLayoutInput input = warming_input();
  input.progress = {.ready = 3, .total = 4, .failed = 1};
  input.hardFailed = 1;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Refuse);
  input.progress = {.ready = 1, .total = 4, .failed = 1};
  input.elapsedMs = SceneLayoutWarmTimeoutMs;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Refuse);
  input.want = false;
  input.live = true;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, NeedsTwoMatchedFrames) {
  SceneLayoutInput input = warming_input();
  for (uint32_t frames = 0; frames < SceneLayoutMatchedFrames; ++frames) {
    input.matchedFrames = frames;
    EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::None) << frames;
  }
  input.matchedFrames = SceneLayoutMatchedFrames;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST_F(SceneLayoutWarmTest, UnmatchedFramesDoNotCount) {
  testing::begin_warm(compose_scene_layout({.gbuffer = true}, test_base()), 1);
  ASSERT_NE(layout_warm(), nullptr);
  EXPECT_EQ(testing::latch_warm(), 0u);
  EXPECT_EQ(testing::latch_warm(), 0u);
  warm_scene_variant(stochastic_config(0));
  EXPECT_EQ(testing::latch_warm(), 1u);
  EXPECT_EQ(testing::latch_warm(), 1u);
  // A config already warmed still marks its frame as matched
  warm_scene_variant(stochastic_config(0));
  EXPECT_EQ(testing::latch_warm(), 2u);
  const testing::WarmTally tally = testing::warm_tally();
  EXPECT_EQ(tally.pairs, 1u);
  EXPECT_EQ(tally.progress.total, 2u);
  detail::reset_scene_layout();
  EXPECT_EQ(layout_warm(), nullptr);
}

TEST(SceneLayout, SwitchesAfterFiveSeconds) {
  SceneLayoutInput input = warming_input();
  input.progress = {.ready = 1, .total = 4};
  input.matchedFrames = 0;
  input.elapsedMs = SceneLayoutWarmTimeoutMs - 1;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::None);
  input.elapsedMs = SceneLayoutWarmTimeoutMs;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, ToggleBackCancelsWarm) {
  SceneLayoutInput input = warming_input();
  input.want = false;
  input.progress = {.ready = 1, .total = 4};
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Cancel);
  input.warming = false;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::None);
}

TEST(SceneLayout, BlockingPipelinesSwitchAtOnce) {
  EXPECT_EQ(next_scene_layout_action({.want = true, .blocking = true, .supported = true}), SceneLayoutAction::Switch);
  SceneLayoutInput input = warming_input();
  input.progress = {.ready = 0, .total = 4};
  input.blocking = true;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, PendingKeyChangeRestartsWarm) {
  SceneLayoutInput input = warming_input();
  input.pendingKeyChanged = true;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Restart);
  input.pendingKeyChanged = false;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Switch);
}

TEST(SceneLayout, PendingLayoutExcludesDebugId) {
  const SceneLayoutBase base = test_base();
  const RenderTargetLayout pending = pending_scene_layout(false, true, base);
  for (uint32_t i = 0; i < pending.colorAttachmentCount; ++i) {
    EXPECT_NE(pending.colorAttachments[i].semantic, ColorAttachmentSemantic::DebugId) << i;
  }
  EXPECT_EQ(pending.key, pipeline_base_layout(compose_scene_layout({.gbuffer = true, .debugId = true}, base)).key);
  EXPECT_EQ(pending_scene_layout(true, true, base).key, pending.key);
  EXPECT_EQ(pending_scene_layout(true, false, base).key,
            pipeline_base_layout(compose_scene_layout({.normal = true, .debugId = true}, base)).key);
}

TEST(SceneLayout, UnsupportedDeviceNeverWarms) {
  EXPECT_EQ(next_scene_layout_action({.want = true}), SceneLayoutAction::None);
  EXPECT_EQ(next_scene_layout_action({.want = true, .firstLatch = true}), SceneLayoutAction::None);
  SceneLayoutInput input = warming_input();
  input.supported = false;
  EXPECT_EQ(next_scene_layout_action(input), SceneLayoutAction::Cancel);
}

TEST_F(SceneLayoutWarmTest, WarmSetCapsAndCountsOverflow) {
  testing::begin_warm(compose_scene_layout({.gbuffer = true}, test_base()), 1);
  gx::PipelineConfig config{};
  for (uint32_t i = 0; i <= MaxSceneLayoutWarmConfigs; ++i) {
    config.dstAlpha = i;
    warm_scene_variant(config);
  }
  config.dstAlpha = 0;
  warm_scene_variant(config);
  const testing::WarmTally tally = testing::warm_tally();
  EXPECT_EQ(tally.pairs, MaxSceneLayoutWarmConfigs);
  EXPECT_EQ(tally.overflow, 1u);
  EXPECT_EQ(tally.progress.total, MaxSceneLayoutWarmConfigs);
  // No pipeline thread runs here, and warm requests never take the inline build path
  EXPECT_EQ(tally.progress.ready + tally.progress.failed, 0u);
}

TEST(SceneLayout, QueueOrderNormalWarmBackground) {
  EXPECT_EQ(detail::next_queue(true, true, true), PipelinePriority::Normal);
  EXPECT_EQ(detail::next_queue(true, false, true), PipelinePriority::Normal);
  EXPECT_EQ(detail::next_queue(false, true, true), PipelinePriority::Warm);
  EXPECT_EQ(detail::next_queue(false, false, true), PipelinePriority::Background);
  EXPECT_FALSE(detail::next_queue(false, false, false).has_value());
}

TEST(SceneLayout, WarmPromotionRules) {
  using enum PipelinePriority;
  EXPECT_EQ(detail::promoted(Background, Background), Background);
  EXPECT_EQ(detail::promoted(Background, Warm), Warm);
  EXPECT_EQ(detail::promoted(Background, Normal), Normal);
  EXPECT_EQ(detail::promoted(Background, Blocking), Blocking);
  EXPECT_EQ(detail::promoted(Warm, Background), Warm);
  EXPECT_EQ(detail::promoted(Warm, Warm), Warm);
  EXPECT_EQ(detail::promoted(Warm, Normal), Normal);
  EXPECT_EQ(detail::promoted(Warm, Blocking), Blocking);
  EXPECT_EQ(detail::promoted(Normal, Warm), Normal);
  EXPECT_EQ(detail::promoted(Normal, Blocking), Normal);
}
} // namespace
} // namespace aurora::gfx

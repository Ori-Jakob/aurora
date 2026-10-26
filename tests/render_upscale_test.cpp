#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/frame_packet.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/recording.hpp"
#include "gx/gx.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/jobs.hpp"
#include "render/pass_context.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/post/upscale/upscale.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aurora::render::detail {
namespace {
using gfx::detail::FrameOpType;

constexpr uint32_t OutputWidth = 160;
constexpr uint32_t OutputHeight = 120;
constexpr GraphExtent SceneAtTwo{80, 60};

struct PassShape {
  std::string label;
  bool sealed = false;
  bool discardable = false;
  uint32_t attachments = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  bool operator==(const PassShape&) const = default;
};

struct FrameShape {
  std::vector<FrameOpType> ops;
  std::vector<PassShape> passes;
  bool operator==(const FrameShape&) const = default;
};

bool always_enabled(const StageData&, const void*) { return true; }

void encode_nothing(PassContext&, const void*) {}

webgpu::TextureWithSampler create_depth(const wgpu::Device& device, uint32_t width, uint32_t height,
                                        uint32_t samples = 1) {
  const wgpu::TextureDescriptor descriptor{
      .label = "Test EFB depth",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .size = {width, height, 1},
      .format = wgpu::TextureFormat::Depth32Float,
      .sampleCount = samples,
  };
  wgpu::Texture texture = device.CreateTexture(&descriptor);
  wgpu::TextureView view = texture.CreateView();
  return {.texture = std::move(texture),
          .view = std::move(view),
          .size = descriptor.size,
          .format = wgpu::TextureFormat::Depth32Float};
}

class UpscaleTest : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset();
    m_options = {.backend = wgpu::BackendType::Null, .efbSize = {OutputWidth, OutputHeight, 1}, .createEfb = true};
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    webgpu::g_depthBuffer = create_depth(m_device->device, OutputWidth, OutputHeight);
    jobs::initialize(1);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    gfx::detail::testing::suppress_render_worker(true);
    upscale::testing::reset();
    m_savedViewport = {gx::g_gxState.logicalViewport, gx::g_gxState.renderViewport};
    m_savedScissor = {gx::g_gxState.logicalScissor, gx::g_gxState.renderScissor};
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      if (m_recording) {
        end_frame();
      }
      gfx::detail::shutdown_recording();
      webgpu::set_scene_render_size(0, 0);
      upscale::testing::reset();
      upscale::shutdown();
      shaders::shutdown();
      jobs::shutdown();
      testing::reset();
      gfx::set_blocking_pipelines(false);
      gfx::detail::testing::suppress_render_worker(false);
      gx::g_gxState.logicalViewport = m_savedViewport[0];
      gx::g_gxState.renderViewport = m_savedViewport[1];
      gx::g_gxState.logicalScissor = m_savedScissor[0];
      gx::g_gxState.renderScissor = m_savedScissor[1];
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void add_pass(std::string name, Stage stage) {
    register_pass({.name = std::move(name), .stage = stage, .enabled = always_enabled, .encode = encode_nothing});
  }

  void start_frame() {
    if (m_recording) {
      end_frame();
    }
    m_frame = {};
    m_frame.frameId = ++m_frameId;
    m_frame.renderState = latch_frame();
    gfx::detail::begin_recording(m_frame, 0);
    m_recording = true;
  }

  void end_frame() {
    gfx::finish();
    gfx::detail::end_recording();
    m_recording = false;
  }

  [[nodiscard]] size_t count_ops(FrameOpType type) const {
    return static_cast<size_t>(std::ranges::count(m_frame.ops, type, &gfx::detail::FrameOp::type));
  }

  [[nodiscard]] std::vector<FrameOpType> op_types() const {
    std::vector<FrameOpType> types;
    for (const auto& op : m_frame.ops) {
      types.push_back(op.type);
    }
    return types;
  }

  [[nodiscard]] FrameShape shape() const {
    FrameShape out{.ops = op_types()};
    for (const auto& pass : m_frame.renderPasses) {
      const auto& color = pass.colorAttachments[gfx::SceneColorAttachmentIndex];
      out.passes.push_back({
          .label = pass.label,
          .sealed = pass.sealed,
          .discardable = pass.discardable,
          .attachments = pass.colorAttachmentCount,
          .width = color.size.width,
          .height = color.size.height,
      });
    }
    return out;
  }

  [[nodiscard]] const gfx::detail::RenderPass& current_pass() const { return m_frame.renderPasses.back(); }

  FrameShape record_scene_frame() {
    start_frame();
    for (const Stage stage :
         {Stage::SceneBegin, Stage::OpaqueDone, Stage::PrePost, Stage::SceneEnd, Stage::BeforeHud, Stage::FrameEnd}) {
      gfx::record_stage(stage);
    }
    end_frame();
    return shape();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  bool m_recording = false;
  uint64_t m_frameId = 0;
  gfx::detail::FramePacket m_frame;
  std::array<gfx::Viewport, 2> m_savedViewport{};
  std::array<gfx::ClipRect, 2> m_savedScissor{};
};

using SceneSplit = UpscaleTest;
using Upscale = UpscaleTest;

TEST_F(SceneSplit, OffLeavesOpStreamUnchanged) {
  add_pass("test.pre_post", Stage::PrePost);
  add_pass("test.before_hud", Stage::BeforeHud);
  const FrameShape off = record_scene_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 0u);
  EXPECT_TRUE(m_frame.sceneComposes.empty());
  for (const auto& pass : m_frame.renderPasses) {
    EXPECT_EQ(pass.colorAttachments[gfx::SceneColorAttachmentIndex].view.Get(), webgpu::g_frameBuffer.view.Get());
  }

  upscale::testing::set_split_env(2.f);
  record_scene_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);

  upscale::testing::set_split_env(std::nullopt);
  const FrameShape back = record_scene_frame();
  EXPECT_FALSE(upscale::split_live());
  EXPECT_EQ(back, off);
}

TEST_F(SceneSplit, OffAliasesEfbTargets) {
  const auto expect_aliased = [] {
    EXPECT_FALSE(webgpu::scene_split_active());
    for (const Stage stage : {Stage::SceneBegin, Stage::PrePost, Stage::SceneEnd, Stage::FrameEnd}) {
      EXPECT_EQ(webgpu::scene_color_for(stage).texture.Get(), webgpu::g_frameBuffer.texture.Get());
      EXPECT_EQ(webgpu::scene_depth_for(stage).texture.Get(), webgpu::g_depthBuffer.texture.Get());
    }
    EXPECT_EQ(webgpu::scene_extent().width, OutputWidth);
    EXPECT_EQ(webgpu::scene_extent().height, OutputHeight);
  };
  expect_aliased();
  upscale::testing::set_split_env(2.f);
  EXPECT_TRUE(upscale::on_latch(Settings{}));
  EXPECT_TRUE(webgpu::scene_split_active());
  EXPECT_NE(webgpu::scene_color_for(Stage::PrePost).texture.Get(), webgpu::g_frameBuffer.texture.Get());
  EXPECT_EQ(webgpu::scene_color_for(Stage::SceneEnd).texture.Get(), webgpu::g_frameBuffer.texture.Get());
  EXPECT_EQ(webgpu::scene_extent().width, SceneAtTwo.width);
  EXPECT_EQ(webgpu::scene_extent().height, SceneAtTwo.height);
  upscale::testing::set_split_env(std::nullopt);
  EXPECT_TRUE(upscale::on_latch(Settings{}));
  expect_aliased();
  EXPECT_FALSE(upscale::on_latch(Settings{}));
}

TEST_F(SceneSplit, SwitchesOnceAtPrePost) {
  add_pass("test.pre_post", Stage::PrePost);
  add_pass("test.before_hud", Stage::BeforeHud);
  upscale::testing::set_split_env(2.f);
  start_frame();
  ASSERT_TRUE(upscale::split_live());
  EXPECT_EQ(current_pass().colorAttachments[gfx::SceneColorAttachmentIndex].size.width, SceneAtTwo.width);
  gfx::record_stage(Stage::SceneBegin);
  gfx::record_stage(Stage::OpaqueDone);
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 0u);
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(op_types(), (std::vector{FrameOpType::RenderPass, FrameOpType::RenderStage, FrameOpType::SceneCompose}));
  EXPECT_EQ(gfx::current_efb_mode(), gfx::EfbMode::Output);
  const auto& output = current_pass();
  EXPECT_EQ(output.colorAttachmentCount, 1u);
  EXPECT_EQ(output.colorAttachments[gfx::SceneColorAttachmentIndex].view.Get(), webgpu::g_frameBuffer.view.Get());
  EXPECT_EQ(output.depthStencilView.Get(), webgpu::g_depthBuffer.view.Get());
  EXPECT_FALSE(output.clearDepth);
  EXPECT_FALSE(output.colorAttachments[gfx::SceneColorAttachmentIndex].clear);
  for (const Stage stage : {Stage::SceneEnd, Stage::BeforeHud, Stage::FrameEnd}) {
    gfx::record_stage(stage);
  }
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
  ASSERT_EQ(m_frame.sceneComposes.size(), 1u);
  const SceneComposeData& data = m_frame.sceneComposes[0];
  EXPECT_EQ(data.mode, ComposeMode::Split);
  EXPECT_EQ(data.ratio[0], 2.f);
  EXPECT_EQ(data.ratio[1], 2.f);
  EXPECT_EQ(data.frameId, m_frame.frameId);
  ASSERT_EQ(m_frame.stages.size(), 2u);
  EXPECT_EQ(m_frame.stages[0].stage, Stage::PrePost);
  EXPECT_EQ(m_frame.stages[1].stage, Stage::BeforeHud);
  const auto& scene = m_frame.renderPasses.front();
  EXPECT_EQ(scene.colorAttachments[gfx::SceneColorAttachmentIndex].view.Get(), webgpu::scene_color().view.Get());
  EXPECT_EQ(scene.depthStencilView.Get(), webgpu::scene_depth().view.Get());
}

TEST_F(SceneSplit, SwitchesWithEmptyPrePostBatch) {
  upscale::testing::set_split_env(2.f);
  start_frame();
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(op_types(), (std::vector{FrameOpType::RenderPass, FrameOpType::SceneCompose}));
  EXPECT_TRUE(m_frame.renderPasses.front().sealed);
  EXPECT_EQ(gfx::current_efb_mode(), gfx::EfbMode::Output);
  EXPECT_EQ(current_pass().colorAttachments[gfx::SceneColorAttachmentIndex].size.width, OutputWidth);
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
}

TEST_F(SceneSplit, ForcedAtSceneEndWithoutPrePost) {
  add_pass("test.pre_post", Stage::PrePost);
  add_pass("test.scene_end", Stage::SceneEnd);
  upscale::testing::set_split_env(2.f);
  start_frame();
  gfx::record_stage(Stage::SceneBegin);
  gfx::record_stage(Stage::SceneEnd);
  EXPECT_EQ(op_types(), (std::vector{FrameOpType::RenderPass, FrameOpType::RenderStage, FrameOpType::SceneCompose,
                                     FrameOpType::RenderPass, FrameOpType::RenderStage}));
  ASSERT_EQ(m_frame.stages.size(), 2u);
  EXPECT_EQ(m_frame.stages[0].stage, Stage::PrePost);
  EXPECT_FALSE(m_frame.stages[0].markerSeen);
  EXPECT_EQ(m_frame.stages[1].stage, Stage::SceneEnd);
  EXPECT_TRUE(m_frame.stages[1].markerSeen);
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
}

TEST_F(SceneSplit, MarkerWhileOffscreenSwitchesAtRestore) {
  upscale::testing::set_split_env(2.f);
  start_frame();
  gfx::begin_offscreen(32, 16);
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 0u);
  EXPECT_EQ(gfx::current_efb_mode(), gfx::EfbMode::Scene);
  gfx::end_offscreen();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
  EXPECT_EQ(gfx::current_efb_mode(), gfx::EfbMode::Output);
  EXPECT_EQ(current_pass().colorAttachments[gfx::SceneColorAttachmentIndex].view.Get(),
            webgpu::g_frameBuffer.view.Get());
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
}

TEST_F(SceneSplit, SwitchesAtFinishWithoutMarkers) {
  upscale::testing::set_split_env(2.f);
  start_frame();
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 1u);
  const auto& last = m_frame.renderPasses.back();
  EXPECT_TRUE(last.sealed);
  EXPECT_TRUE(last.captureDepthSnapshot);
  EXPECT_EQ(last.colorAttachments[gfx::SceneColorAttachmentIndex].view.Get(), webgpu::g_frameBuffer.view.Get());
  EXPECT_EQ(m_frame.ops.back().type, FrameOpType::RenderPass);
}

TEST_F(SceneSplit, ComposeEncodedOnWorker) {
  upscale::testing::set_split_env(2.f);
  start_frame();
  gfx::record_stage(Stage::PrePost);
  end_frame();
  const auto op = std::ranges::find(m_frame.ops, FrameOpType::SceneCompose, &gfx::detail::FrameOp::type);
  ASSERT_NE(op, m_frame.ops.end());
  ASSERT_TRUE(gfx::detail::op_has_payload(*op));
  m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
  wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  scene_compose(encoder, *op->sceneCompose);
  const wgpu::CommandBuffer commands = encoder.Finish();
  EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  EXPECT_EQ(upscale::compose::testing::counters().encoded, 1u);
  EXPECT_EQ(upscale::compose::testing::counters().skipped, 0u);
}

TEST_F(SceneSplit, ViewportRemappedToOutput) {
  gx::g_gxState.logicalViewport = {0.f, 0.f, 640.f, 480.f, 0.f, 1.f};
  gx::g_gxState.logicalScissor = {0, 0, 640, 480};
  upscale::testing::set_split_env(2.f);
  start_frame();
  const auto viewport_of = [](const gfx::detail::RenderPass& pass) {
    for (const auto& command : pass.commands) {
      if (command.type == gfx::detail::CommandType::SetViewport) {
        return command.data.setViewport;
      }
    }
    return gfx::Viewport{};
  };
  EXPECT_EQ(viewport_of(current_pass()).width, static_cast<float>(SceneAtTwo.width));
  gfx::record_stage(Stage::PrePost);
  const gfx::Viewport viewport = viewport_of(current_pass());
  EXPECT_EQ(viewport.width, static_cast<float>(OutputWidth));
  EXPECT_EQ(viewport.height, static_cast<float>(OutputHeight));
  EXPECT_EQ(gx::g_gxState.renderViewport.width, static_cast<float>(OutputWidth));
  EXPECT_EQ(gx::g_gxState.renderScissor.width, static_cast<int32_t>(OutputWidth));
  EXPECT_EQ(gx::g_gxState.renderScissor.height, static_cast<int32_t>(OutputHeight));
}

TEST_F(SceneSplit, CopyAfterSplitUsesOutputScale) {
  upscale::testing::set_split_env(2.f);
  start_frame();
  EXPECT_EQ(gfx::get_render_target_size(), (Vec2<uint32_t>{SceneAtTwo.width, SceneAtTwo.height}));
  EXPECT_EQ(current_pass().copySourceTexture.Get(), webgpu::scene_color().texture.Get());
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(gfx::get_render_target_size(), (Vec2<uint32_t>{OutputWidth, OutputHeight}));
  EXPECT_EQ(current_pass().copySourceTexture.Get(), webgpu::g_frameBuffer.texture.Get());
  EXPECT_EQ(current_pass().copySourceDepthView.Get(), webgpu::g_depthBuffer.view.Get());
  const gfx::ClipRect copy = gx::map_logical_scissor({0, 0, 320, 240});
  EXPECT_EQ(copy.width, static_cast<int32_t>(OutputWidth / 2));
  EXPECT_EQ(copy.height, static_cast<int32_t>(OutputHeight / 2));
}

TEST_F(SceneSplit, OffWhenMsaaOrCompat) {
  constexpr GraphExtent Output{OutputWidth, OutputHeight};
  EXPECT_EQ(upscale::wanted_scene_size({.output = Output, .ratio = 2.f, .sceneTargets = true}), SceneAtTwo);
  EXPECT_EQ(upscale::wanted_scene_size({.output = Output, .ratio = 2.f, .sceneTargets = false}), GraphExtent{});
  EXPECT_EQ(upscale::wanted_scene_size({.output = Output, .ratio = 2.f, .sceneTargets = true, .moduleEnabled = false}),
            GraphExtent{});
  EXPECT_EQ(upscale::wanted_scene_size({.output = Output, .ratio = 0.f, .sceneTargets = true}), GraphExtent{});
  EXPECT_EQ(upscale::wanted_scene_size({.output = {1, 1}, .ratio = 2.f, .sceneTargets = true}), GraphExtent{});
  upscale::testing::set_split_env(2.f);
  for (const bool msaa : {true, false}) {
    SCOPED_TRACE(msaa ? "msaa 4" : "compat");
    test::TestDeviceOptions options{.backend = wgpu::BackendType::Null,
                                    .core = msaa,
                                    .efbSize = {OutputWidth, OutputHeight, 1},
                                    .msaaSamples = msaa ? 4u : 1u,
                                    .createEfb = true};
    std::string error;
    const auto device = test::create_test_device(options, error);
    ASSERT_NE(device, nullptr) << error;
    const test::ScopedGlobalDevice globals{*device, options};
    EXPECT_FALSE(webgpu::scene_targets_supported());
    EXPECT_FALSE(upscale::on_latch(Settings{}));
    EXPECT_FALSE(upscale::split_live());
    EXPECT_FALSE(webgpu::scene_split_active());
  }
}

TEST_F(Upscale, SwitchWaitsForWarm) {
  add_pass("test.pre_post", Stage::PrePost);
  upscale::compose::testing::hold_pipelines(true);
  upscale::testing::set_split_env(2.f);
  start_frame();
  EXPECT_FALSE(upscale::split_live());
  EXPECT_FALSE(webgpu::scene_split_active());
  end_frame();
  EXPECT_EQ(count_ops(FrameOpType::SceneCompose), 0u);
  EXPECT_EQ(m_frame.renderState->graph->sceneSize, (GraphExtent{OutputWidth, OutputHeight}));

  upscale::compose::testing::hold_pipelines(false);
  start_frame();
  EXPECT_TRUE(upscale::split_live());
  EXPECT_EQ(upscale::scene_size(), SceneAtTwo);
  EXPECT_EQ(m_frame.renderState->graph->sceneSize, SceneAtTwo);
  const UpscaleStatus status = upscale::status();
  EXPECT_EQ(status.active, Upscaler::Off);
  EXPECT_EQ(status.sceneSize[0], SceneAtTwo.width);
  EXPECT_EQ(status.outputSize[0], OutputWidth);
}

TEST(UpscaleRules, SplitRatioParses) {
  EXPECT_EQ(upscale::parse_split_ratio("1.5"), 1.5f);
  EXPECT_EQ(upscale::parse_split_ratio("3"), 3.f);
  EXPECT_EQ(upscale::parse_split_ratio("1"), std::nullopt);
  EXPECT_EQ(upscale::parse_split_ratio("3.5"), std::nullopt);
  EXPECT_EQ(upscale::parse_split_ratio("two"), std::nullopt);
  EXPECT_EQ(upscale::parse_split_ratio("1.5x"), std::nullopt);
  EXPECT_EQ(upscale::split_size({2560, 1440}, 1.5f), (GraphExtent{1706, 960}));
  EXPECT_EQ(upscale::split_size({1, 1}, 3.f), (GraphExtent{1, 1}));
}
} // namespace
} // namespace aurora::render::detail

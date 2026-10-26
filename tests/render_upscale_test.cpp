#include <gtest/gtest.h>

#include "render_fsr_rig.hpp"
#include "render_test_device.hpp"

#include "gfx/frame_packet.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture.hpp"
#include "gx/gx.hpp"
#include "gx/shader_info.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/jobs.hpp"
#include "render/pass_context.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/post/upscale/fsr/fsr.hpp"
#include "render/post/upscale/fsr/fsr_pass.hpp"
#include "render/post/temporal_resolve.hpp"
#include "render/post/upscale/upscale.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace fsr_pass = upscale::fsr_pass;
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

constexpr gfx::Viewport FullViewport{0.f, 0.f, 640.f, 480.f, 0.f, 1.f};
gfx::TextureBind bias_texture(bool arbitraryMips) {
  auto ref = std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                               wgpu::Extent3D{128, 128, 1}, wgpu::TextureFormat::RGBA8Unorm, 8u,
                                               static_cast<uint32_t>(GX_TF_RGBA8));
  ref->hasArbitraryMips = arbitraryMips;
  ref->isReplacement = arbitraryMips;
  GXTexObj_ obj{};
  obj.mWidth = 64;
  obj.mHeight = 64;
  obj.mode0 = 16u << 9;
  return {obj, std::move(ref)};
}

class GxTextureBias : public UpscaleTest {
protected:
  void TearDown() override {
    gx::g_gxState.textures[0] = {};
    temporal_resolve::install(nullptr);
    UpscaleTest::TearDown();
  }

  static std::vector<uint8_t> uniform_bytes() {
    gx::ShaderInfo info{};
    info.sampledTextures.set(0);
    return gx::testing::uniform_bytes(info);
  }

  static float sampled_bias() {
    const std::vector<uint8_t> bytes = uniform_bytes();
    std::array<float, 4> sizeBias{};
    EXPECT_GE(bytes.size(), sizeof(sizeBias));
    std::memcpy(sizeBias.data(), bytes.data() + bytes.size() - sizeof(sizeBias), sizeof(sizeBias));
    EXPECT_EQ(sizeBias[0], 64.f);
    return sizeBias[2];
  }

  static bool uniform_dirty() { return (gx::g_gxState.dirty & gx::DirtyUniform) != 0; }
};

TEST_F(GxTextureBias, UniformBytesUnchangedWithoutUpscale) {
  gx::g_gxState.textures[0] = bias_texture(true);
  start_frame();
  EXPECT_FALSE(gx::g_upscaleSampling.active);
  gx::set_logical_viewport(FullViewport);
  const std::vector<uint8_t> before = uniform_bytes();
  gx::g_upscaleSampling = {.active = false, .outputPerRender = 2.f, .mipBias = -1.f};
  EXPECT_EQ(uniform_bytes(), before);
  end_frame();

  upscale::testing::set_split_env(2.f);
  upscale::testing::set_mip_bias_env(-1.f);
  start_frame();
  EXPECT_TRUE(gx::g_upscaleSampling.active);
  gx::set_logical_viewport(FullViewport);
  EXPECT_NE(uniform_bytes(), before);
  end_frame();

  upscale::testing::set_split_env(std::nullopt);
  start_frame();
  EXPECT_FALSE(upscale::split_live());
  EXPECT_FALSE(gx::g_upscaleSampling.active);
  gx::set_logical_viewport(FullViewport);
  EXPECT_EQ(uniform_bytes(), before);
  gx::g_gxState.dirty = 0;
  gfx::begin_offscreen(32, 16);
  EXPECT_FALSE(uniform_dirty());
  gfx::end_offscreen();
  EXPECT_FALSE(uniform_dirty());
}

TEST_F(GxTextureBias, SceneDrawsGetUpscaleBias) {
  gx::g_gxState.textures[0] = bias_texture(false);
  start_frame();
  gx::set_logical_viewport(FullViewport);
  const float base = sampled_bias();
  EXPECT_EQ(base, 0.5f);
  end_frame();

  upscale::testing::set_split_env(2.f);
  start_frame();
  ASSERT_TRUE(upscale::split_live());
  EXPECT_TRUE(gfx::in_scene_pass());
  EXPECT_TRUE(gx::g_upscaleSampling.active);
  EXPECT_EQ(gx::g_upscaleSampling.outputPerRender, 2.f);
  EXPECT_EQ(gx::g_upscaleSampling.mipBias, -1.f);
  gx::set_logical_viewport(FullViewport);
  EXPECT_EQ(sampled_bias(), base - 1.f);
  end_frame();

  upscale::testing::set_mip_bias_env(-1.f);
  start_frame();
  EXPECT_EQ(gx::g_upscaleSampling.mipBias, -2.f);
  gx::set_logical_viewport(FullViewport);
  EXPECT_EQ(sampled_bias(), base - 2.f);
  end_frame();

  static constexpr temporal_resolve::Resolver Biased{
      .pass = "test.biased",
      .mip_bias = [](float ratio) noexcept { return -0.25f * ratio; },
  };
  temporal_resolve::install(&Biased);
  upscale::testing::set_mip_bias_env(std::nullopt);
  start_frame();
  EXPECT_EQ(gx::g_upscaleSampling.mipBias, -0.5f);
  gx::set_logical_viewport(FullViewport);
  EXPECT_EQ(sampled_bias(), base - 0.5f);
}

TEST_F(GxTextureBias, OffscreenAndPostSplitUnbiased) {
  gx::g_gxState.textures[0] = bias_texture(false);
  upscale::testing::set_split_env(2.f);
  start_frame();
  gx::set_logical_viewport(FullViewport);
  const float base = gx::g_gxState.textures[0].sample_lod_bias();
  EXPECT_EQ(sampled_bias(), base - 1.f);

  gx::g_gxState.dirty = 0;
  gfx::begin_offscreen(32, 16);
  EXPECT_FALSE(gfx::in_scene_pass());
  EXPECT_TRUE(uniform_dirty());
  EXPECT_EQ(sampled_bias(), base);

  gx::g_gxState.dirty = 0;
  gfx::end_offscreen();
  EXPECT_TRUE(gfx::in_scene_pass());
  EXPECT_TRUE(uniform_dirty());
  EXPECT_EQ(sampled_bias(), base - 1.f);

  gx::g_gxState.dirty = 0;
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(gfx::current_efb_mode(), gfx::EfbMode::Output);
  EXPECT_FALSE(gfx::in_scene_pass());
  EXPECT_TRUE(uniform_dirty());
  EXPECT_EQ(sampled_bias(), base);

  gfx::begin_offscreen(32, 16);
  EXPECT_EQ(sampled_bias(), base);
  gfx::end_offscreen();
  EXPECT_FALSE(gfx::in_scene_pass());
  EXPECT_EQ(sampled_bias(), base);
}

TEST_F(GxTextureBias, ArbitraryMipReplacementKeepsOutputScale) {
  gx::g_gxState.textures[0] = bias_texture(true);
  start_frame();
  gx::set_logical_viewport(FullViewport);
  ASSERT_EQ(gx::g_gxState.renderViewport.width, static_cast<float>(OutputWidth));
  const float output = sampled_bias();
  EXPECT_EQ(output, 0.5f - 3.f);
  end_frame();

  upscale::testing::set_split_env(2.f);
  start_frame();
  gx::set_logical_viewport(FullViewport);
  ASSERT_EQ(gx::g_gxState.renderViewport.width, static_cast<float>(SceneAtTwo.width));
  EXPECT_EQ(sampled_bias(), output + gx::g_upscaleSampling.mipBias);
  gfx::record_stage(Stage::PrePost);
  EXPECT_EQ(gx::g_gxState.renderViewport.width, static_cast<float>(OutputWidth));
  EXPECT_EQ(sampled_bias(), output);
}

TEST(UpscaleRules, MipBiasParses) {
  EXPECT_EQ(upscale::parse_mip_bias("-1"), -1.f);
  EXPECT_EQ(upscale::parse_mip_bias("0.5"), 0.5f);
  EXPECT_EQ(upscale::parse_mip_bias("-4.5"), std::nullopt);
  EXPECT_EQ(upscale::parse_mip_bias("bias"), std::nullopt);
  EXPECT_EQ(upscale::parse_mip_bias("-1x"), std::nullopt);
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

TEST(FsrHost, RenderSizesMatchAmd) {
  const GraphExtent output{2560, 1440};
  EXPECT_EQ(upscale::fsr::render_size(output, UpscalerQuality::NativeAa), output);
  EXPECT_EQ(upscale::fsr::render_size(output, UpscalerQuality::Quality), (GraphExtent{1706, 960}));
  EXPECT_EQ(upscale::fsr::render_size(output, UpscalerQuality::Balanced), (GraphExtent{1505, 847}));
  EXPECT_EQ(upscale::fsr::render_size(output, UpscalerQuality::Performance), (GraphExtent{1280, 720}));
  EXPECT_EQ(upscale::fsr::render_size(output, UpscalerQuality::UltraPerformance), (GraphExtent{853, 480}));
  EXPECT_EQ(upscale::fsr::render_size({4779, 2688}, UpscalerQuality::Quality), (GraphExtent{3186, 1792}));
  EXPECT_EQ(upscale::fsr::render_size({64, 40}, UpscalerQuality::UltraPerformance), (GraphExtent{32, 32}));
}

TEST(FsrHost, JitterPhasesMatchAmd) {
  constexpr std::array Qualities{UpscalerQuality::NativeAa, UpscalerQuality::Quality, UpscalerQuality::Balanced,
                                 UpscalerQuality::Performance, UpscalerQuality::UltraPerformance};
  for (const uint32_t width : {1920u, 2560u, 4779u}) {
    for (const UpscalerQuality quality : Qualities) {
      const GraphExtent render = upscale::fsr::render_size({width, width / 2}, quality);
      const float ratio = static_cast<float>(width) / static_cast<float>(render.width);
      const auto amd = static_cast<uint32_t>(8.f * std::pow(ratio, 2.f));
      EXPECT_EQ(upscale::fsr::jitter_phases(upscale::fsr::mode_ratio(quality)), std::max(8u, amd))
          << width << " " << static_cast<int>(quality);
    }
  }
  EXPECT_EQ(upscale::fsr::jitter_phases(1.5f), 18u);
  EXPECT_EQ(upscale::fsr::jitter_phases(1.7f), 23u);
  EXPECT_EQ(upscale::fsr::jitter_phases(3.f), 72u);
}

TEST(FsrHost, ParamsFitPassParams) {
  static_assert(sizeof(upscale::fsr::Params) <= MaxPassParamsSize);
  EXPECT_EQ(sizeof(upscale::fsr::Params) % 16, 0u);
  const upscale::fsr::Params params = upscale::fsr::make_params({
      .camera = {.nearZ = 10.f, .farZ = 10000.f, .fovY = 1.f, .viewToMeters = 0.01f},
      .frameTimeMs = 2000.f,
      .framesSinceReset = 3,
      .jitterPhases = 18,
      .render = {1706, 960},
  });
  const auto view_depth = [&](float depth) { return params.deviceToView[1] / (depth - params.deviceToView[0]); };
  EXPECT_NEAR(view_depth(1.f), 10.f, 1e-3f);
  EXPECT_NEAR(view_depth(0.f), 10000.f, 1.f);
  EXPECT_EQ(params.deltaTime, 1.f);
  EXPECT_EQ(params.frameIndex, 3.f);
  EXPECT_EQ(params.jitterSequenceLength, 18.f);
  EXPECT_EQ(params.spdMips, 10u);
  EXPECT_EQ(upscale::fsr::spd_tiles({1706, 960}), (GraphExtent{27, 15}));
}
TEST(UpscaleRules, EnvOverridesParse) {
  EXPECT_EQ(parse_upscaler_override("fsr"), (UpscalerOverride{.upscaler = Upscaler::Fsr}));
  EXPECT_EQ(parse_upscaler_override("fsr:quality"),
            (UpscalerOverride{.upscaler = Upscaler::Fsr, .quality = UpscalerQuality::Quality}));
  EXPECT_EQ(parse_upscaler_override("auto:ultra"),
            (UpscalerOverride{.upscaler = Upscaler::Auto, .quality = UpscalerQuality::UltraPerformance}));
  EXPECT_EQ(parse_upscaler_override("xess:native"),
            (UpscalerOverride{.upscaler = Upscaler::Xess, .quality = UpscalerQuality::NativeAa}));
  EXPECT_EQ(parse_upscaler_override("off"), (UpscalerOverride{.upscaler = Upscaler::Off}));
  EXPECT_EQ(parse_upscaler_override("fsr:fast"), std::nullopt);
  EXPECT_EQ(parse_upscaler_override("taa"), std::nullopt);
  EXPECT_EQ(parse_upscaler_override(""), std::nullopt);
  EXPECT_EQ(upscale::parse_signs("1,1,1,1"), upscale::Signs{});
  EXPECT_EQ(upscale::parse_signs("-1,+1,1,-1"), (upscale::Signs{.jitter = {-1.f, 1.f}, .motion = {1.f, -1.f}}));
  EXPECT_EQ(upscale::parse_signs("1,1,1"), std::nullopt);
  EXPECT_EQ(upscale::parse_signs("1,1,1,1,1"), std::nullopt);
  EXPECT_EQ(upscale::parse_signs("1,2,1,1"), std::nullopt);
  EXPECT_EQ(upscale::parse_color("stored"), ColorEncoding::Stored);
  EXPECT_EQ(upscale::parse_color("linear"), ColorEncoding::Linear);
  EXPECT_EQ(upscale::parse_color("gamma"), std::nullopt);
}

class FsrGraph : public UpscaleTest {
protected:
  void SetUp() override {
    UpscaleTest::SetUp();
    if (IsSkipped() || !m_started) {
      return;
    }
    m_rig = std::make_unique<fsr_rig::Rig>(*m_device, m_options);
    if (!m_rig->ok()) {
      GTEST_SKIP() << "the base G-buffer tier needs a core device";
    }
  }

  void TearDown() override {
    m_rig.reset();
    UpscaleTest::TearDown();
  }

  [[nodiscard]] const CompiledNode& node(std::string_view name) const {
    const NodeId id = m_rig->node(name);
    EXPECT_NE(id, InvalidNode) << name;
    return m_rig->m_graph->nodes[id];
  }

  [[nodiscard]] const CompiledResource* resource(std::string_view name) const {
    const ResourceId id = m_rig->m_graph->find_resource(name);
    return id != InvalidResource ? &m_rig->m_graph->resources[id] : nullptr;
  }

  std::unique_ptr<fsr_rig::Rig> m_rig;
};

TEST_F(FsrGraph, SkippedWhenNotInstalled) {
  m_rig->use(m_rig->compile(fsr_rig::fsr_settings(UpscalerQuality::Quality)));
  EXPECT_EQ(temporal_resolve::installed().pass, taa::Pass);
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::Skipped);
  EXPECT_EQ(resource("fsr.upscaled"), nullptr);
  EXPECT_TRUE(m_rig->m_graph->histories.empty());
  EXPECT_TRUE(m_rig->m_graph->stage_empty(Stage::PrePost));
  EXPECT_FALSE(fsr_pass::jitter_ready());
  EXPECT_EQ(upscale::status().active, Upscaler::Off);
  EXPECT_EQ(m_rig->run(1), std::nullopt);
  EXPECT_EQ(fsr_pass::testing::counters().encodes, 0u);
  EXPECT_EQ(upscale::compose::testing::counters().encoded, 0u);
}

TEST_F(FsrGraph, ReplacesTaaWhenInstalled) {
  Settings settings = fsr_rig::fsr_settings(UpscalerQuality::Quality);
  settings.taa = true;
  m_rig->install(settings);
  EXPECT_EQ(temporal_resolve::installed().pass, fsr_pass::Pass);
  EXPECT_EQ(temporal_resolve::jitter_phases(), 18u);
  EXPECT_EQ(node(taa::Pass).status, CullReason::Skipped);
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::None);
  EXPECT_EQ(node(fsr_pass::Pass).stage, Stage::PrePost);
  EXPECT_EQ(resource(taa::History), nullptr);
  const CompiledResource* upscaled = resource("fsr.upscaled");
  ASSERT_NE(upscaled, nullptr);
  EXPECT_EQ(upscaled->kind, ResourceKind::History);
  EXPECT_EQ(upscaled->extent.width, OutputWidth);
  EXPECT_EQ(upscaled->extent.height, OutputHeight);
  const CompiledResource* luma = resource("fsr.luma");
  ASSERT_NE(luma, nullptr);
  EXPECT_EQ(luma->extent.width, 106u);
  EXPECT_EQ(luma->extent.height, 80u);
  EXPECT_TRUE(upscale::split_live());
  EXPECT_EQ(upscale::scene_size(), (GraphExtent{106, 80}));
  EXPECT_TRUE(fsr_pass::jitter_ready());
  const UpscaleStatus status = upscale::status();
  EXPECT_EQ(status.active, Upscaler::Fsr);
  EXPECT_EQ(status.quality, UpscalerQuality::Quality);
  EXPECT_EQ(status.displayName, "FSR");
  EXPECT_EQ(status.sceneSize[0], 106u);
  EXPECT_EQ(status.outputSize[0], OutputWidth);
  EXPECT_TRUE(status.ready);
  EXPECT_TRUE(upscale::ready(settings));
  EXPECT_EQ(m_rig->run(1), std::nullopt);
  EXPECT_EQ(fsr_pass::testing::counters().encodes, 1u);
  EXPECT_EQ(taa::testing::counters().resolves, 0u);

  m_rig->install(Settings{.taa = true});
  EXPECT_EQ(temporal_resolve::installed().pass, taa::Pass);
  EXPECT_FALSE(upscale::split_live());
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::Skipped);
  EXPECT_EQ(node(taa::Pass).status, CullReason::None);
  EXPECT_EQ(upscale::status().active, Upscaler::Off);
}

TEST_F(FsrGraph, RunsWithTaaOff) {
  scene::motion::testing::set_jitter_env(scene::motion::testing::JitterEnv::Unset);
  const Settings settings = fsr_rig::fsr_settings(UpscalerQuality::NativeAa);
  m_rig->install(settings);
  EXPECT_FALSE(settings.taa);
  EXPECT_FALSE(upscale::split_live());
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::None);
  ASSERT_NE(resource("fsr.upscaled"), nullptr);
  EXPECT_EQ(resource("fsr.upscaled")->extent.width, OutputWidth);
  bool jittered = false;
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    const StageData post = m_rig->build(frame);
    EXPECT_TRUE(scene::motion::testing::jitter_active());
    ASSERT_NE(post.sceneHints, nullptr);
    jittered = jittered || post.sceneHints->jitterPx[0] != 0.f || post.sceneHints->jitterPx[1] != 0.f;
    EXPECT_EQ(m_rig->encode(post, true), std::nullopt);
    const fsr_pass::testing::Frame last = fsr_pass::testing::last_frame();
    EXPECT_EQ(last.params.jitter[0], post.sceneHints->jitterPx[0]);
    EXPECT_EQ(last.params.jitter[1], post.sceneHints->jitterPx[1]);
  }
  EXPECT_TRUE(jittered);
  EXPECT_EQ(fsr_pass::testing::counters().encodes, 3u);
  EXPECT_EQ(fsr_pass::testing::counters().applied, 3u);
  EXPECT_EQ(fsr_pass::testing::counters().published, 0u);
  EXPECT_EQ(upscale::compose::testing::counters().encoded, 0u);
}

TEST_F(FsrGraph, SkippedUnderDebugView) {
  Settings settings = fsr_rig::fsr_settings(UpscalerQuality::Quality);
  settings.debugView = 16;
  m_rig->install(settings);
  EXPECT_EQ(temporal_resolve::installed().pass, fsr_pass::Pass);
  EXPECT_TRUE(upscale::split_live());
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::Skipped);
  EXPECT_FALSE(fsr_pass::jitter_ready());
  EXPECT_FALSE(upscale::ready(settings));
  EXPECT_EQ(m_rig->run(1), std::nullopt);
  EXPECT_EQ(fsr_pass::testing::counters().encodes, 0u);
  EXPECT_EQ(upscale::compose::testing::counters().encoded, 1u);
  EXPECT_EQ(upscale::compose::testing::counters().sourced, 0u);

  settings.debugView = 0;
  m_rig->install(settings);
  EXPECT_EQ(node(fsr_pass::Pass).status, CullReason::None);
  EXPECT_EQ(m_rig->run(2), std::nullopt);
  EXPECT_EQ(upscale::compose::testing::counters().encoded, 2u);
  EXPECT_EQ(upscale::compose::testing::counters().sourced, 1u);
}

TEST_F(FsrGraph, PublishesHistoryNotTransient) {
  m_rig->install(fsr_rig::fsr_settings(UpscalerQuality::Quality));
  const CompiledResource* upscaled = resource("fsr.upscaled");
  ASSERT_NE(upscaled, nullptr);
  ASSERT_EQ(upscaled->kind, ResourceKind::History);
  std::vector<WGPUTexture> published;
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    EXPECT_EQ(m_rig->run(frame), std::nullopt);
    const wgpu::Texture texture = fsr_pass::testing::last_frame().published;
    ASSERT_TRUE(texture);
    EXPECT_EQ(texture.Get(), transient_pool().history_write(upscaled->historySlot).texture.Get());
    EXPECT_EQ(texture.GetFormat(), wgpu::TextureFormat::RGBA16Float);
    EXPECT_EQ(texture.GetWidth(), OutputWidth);
    published.push_back(texture.Get());
  }
  EXPECT_NE(published[0], published[1]);
  EXPECT_EQ(published[0], published[2]);
  EXPECT_EQ(fsr_pass::testing::counters().published, 3u);
  EXPECT_EQ(upscale::compose::testing::counters().sourced, 3u);
}

TEST_F(FsrGraph, ResetNeverSamplesHistory) {
  m_rig->install(fsr_rig::fsr_settings(UpscalerQuality::Quality));
  const auto expect_frame = [&](uint64_t frame, const fsr_rig::FrameOptions& options, bool reset, float index) {
    SCOPED_TRACE(frame);
    EXPECT_EQ(m_rig->run(frame, options), std::nullopt);
    const fsr_pass::testing::Frame last = fsr_pass::testing::last_frame();
    EXPECT_EQ(last.reset, reset);
    EXPECT_EQ(last.historyBound, !reset);
    EXPECT_EQ(last.params.frameIndex, index);
  };
  expect_frame(1, {}, true, 0.f);
  expect_frame(2, {}, false, 1.f);
  expect_frame(3, {}, false, 2.f);
  expect_frame(4, {.refuseStorage = true}, true, 0.f);
  expect_frame(5, {}, false, 1.f);
  expect_frame(6, {.invalidate = true}, true, 0.f);
  expect_frame(7, {.cutEpoch = 1}, true, 0.f);
  expect_frame(8, {.cutEpoch = 1}, false, 1.f);
  EXPECT_EQ(fsr_pass::testing::counters().resets, 4u);
}

TEST_F(FsrGraph, EncodesValidationClean) {
  for (const UpscalerQuality quality :
       {UpscalerQuality::NativeAa, UpscalerQuality::Quality, UpscalerQuality::UltraPerformance}) {
    SCOPED_TRACE(static_cast<int>(quality));
    fsr_pass::testing::reset();
    m_rig->install(fsr_rig::fsr_settings(quality, 0.5f));
    ASSERT_EQ(node(fsr_pass::Pass).status, CullReason::None);
    for (uint64_t frame = 1; frame <= 4; ++frame) {
      EXPECT_EQ(m_rig->run(frame), std::nullopt);
    }
    EXPECT_EQ(fsr_pass::testing::counters().encodes, 4u);
    EXPECT_EQ(fsr_pass::testing::counters().resets, 1u);
    EXPECT_EQ(fsr_pass::testing::counters().skipped, 0u);
    EXPECT_EQ(fsr_pass::testing::last_frame().params.reactiveMax, fsr_pass::ReactiveMax);
  }
}
} // namespace
} // namespace aurora::render::detail

// Capture lists: route-time selection, push-time clones, the OPAQUE_DONE hand-off and the replay's encoder reset

#include <gtest/gtest.h>

#include "gfx/encoding.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/graph_api.hpp"
#include "render/scene/capture_list.hpp"
#include "render/scene/classifier.hpp"
#include "render/settings.hpp"
#include "render/stage_data.hpp"
#include "webgpu/gpu.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace aurora::render::scene::capture_list {
namespace {
namespace gfx = aurora::gfx;

constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth24Plus;

bool select_fake_shadow(const DrawClass& cls, void*) noexcept {
  return (cls.key.categories & category::FakeShadow) != 0;
}

bool select_all(const DrawClass&, void*) noexcept { return true; }

gfx::PipelineRef g_readyRef = 0;
uint32_t g_readyCalls = 0;
bool g_readyExtZero = false;
std::vector<gx::PipelineConfig> g_warmed;

gfx::PipelineRef hook_ready(const gx::PipelineConfig& config, const gfx::RenderTargetLayout&) noexcept {
  ++g_readyCalls;
  g_readyExtZero = config.shaderConfig.ext == gx::RenderExtConfig{};
  return g_readyRef;
}

void hook_warm(const gx::PipelineConfig& config, const gfx::RenderTargetLayout&) noexcept {
  g_warmed.push_back(config);
}

constexpr testing::PipelineHooks Hooks{.ready = hook_ready, .warm = hook_warm};

bool always(const detail::StageData&, const void*) { return true; }

void encode_nothing(detail::PassContext&, const void*) {}

gx::DrawData draw_data(uint32_t id) {
  gx::DrawData draw{};
  draw.pipeline = 9;
  draw.bindGroups.textureBindGroup = 3;
  draw.bindGroups.companionBindGroup = 5;
  draw.vtxCount = id;
  draw.instanceCount = 1;
  draw.immediateData._pad = 77;
  return draw;
}

DrawClass opaque_class(uint32_t categories = 0) {
  DrawClass cls;
  cls.key.window = Window::Opaque;
  cls.key.categories = categories;
  return cls;
}

gx::DrawData decode(const gfx::detail::Command& command) {
  gx::DrawData draw;
  std::memcpy(&draw, command.data.draw.payload.data(), sizeof(draw));
  return draw;
}

std::vector<gx::DrawData> list_draws(const gfx::CaptureListFrame::List& list) {
  std::vector<gx::DrawData> draws;
  for (const auto& command : list.commands) {
    if (command.type == gfx::detail::CommandType::Draw) {
      draws.push_back(decode(command));
    }
  }
  return draws;
}

class CaptureListTest : public ::testing::Test {
protected:
  void SetUp() override {
    detail::testing::reset();
    detail::testing::reset_external_passes();
    testing::reset();
    savedUniformAlignment = std::exchange(gfx::detail::resources().limits.minUniformBufferOffsetAlignment, 256);
    webgpu::g_graphicsConfig.surfaceConfiguration.format = ColorFormat;
    webgpu::g_graphicsConfig.depthFormat = DepthFormat;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer.size = {640, 480, 1};
    webgpu::g_frameBuffer.format = ColorFormat;
    webgpu::g_depthBuffer.size = {640, 480, 1};
    webgpu::g_depthBuffer.format = DepthFormat;
    gfx::detail::testing::suppress_render_worker(true);
    gfx::detail::begin_recording(frame, 0);
    g_readyRef = 0;
    g_readyCalls = 0;
    g_readyExtZero = false;
    g_warmed.clear();
  }

  void TearDown() override {
    gfx::finish();
    gfx::detail::end_recording();
    gfx::detail::shutdown_recording();
    if (pass != detail::InvalidPassId) {
      detail::unregister_pass(pass);
    }
    testing::reset();
    detail::testing::reset();
    detail::testing::reset_external_passes();
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = savedUniformAlignment;
  }

  ConsumerId install(bool (*select)(const DrawClass&, void*) noexcept, const gfx::RenderTargetLayout& layout) {
    const ConsumerId id = register_consumer({.name = "test", .select = select, .layout = layout});
    set_enabled(id, true);
    apply_frame_state();
    return id;
  }

  // An OPAQUE_DONE pass gives the frame a stage that takes the lists
  void latch_with_opaque_pass() {
    pass = detail::register_pass({
        .name = "test.capture_list",
        .stage = Stage::OpaqueDone,
        .enabled = always,
        .encode = encode_nothing,
    });
    next_frame(true);
  }

  void next_frame(bool latch) {
    gfx::finish();
    gfx::detail::end_recording();
    frame = {};
    if (latch) {
      frame.renderState = detail::latch_frame();
    }
    gfx::detail::begin_recording(frame, 0);
  }

  std::shared_ptr<const gfx::CaptureListFrame> opaque_done_lists() const {
    for (const auto& stage : frame.stages) {
      if (stage.stage == Stage::OpaqueDone) {
        return stage.captureList;
      }
    }
    return nullptr;
  }

  gfx::detail::FramePacket frame;
  detail::PassId pass = detail::InvalidPassId;
  uint32_t savedUniformAlignment = 0;
};

TEST_F(CaptureListTest, SelectByCategoryIgnoresNoCapture) {
  const ConsumerId id = install(select_fake_shadow, gfx::get_render_target_layout());
  ASSERT_EQ(id, 0u);
  const gx::draw_route::Query query{};
  // NO_CAPTURE keeps a draw out of retained captures, not out of a consumer that asks for it by category
  EXPECT_EQ(select(opaque_class(category::FakeShadow | category::NoCapture), query), 1u);
  EXPECT_EQ(select(opaque_class(category::Static), query), 0u);
  set_enabled(id, false);
  EXPECT_EQ(select(opaque_class(category::FakeShadow), query), 1u);
  apply_frame_state();
  EXPECT_EQ(select(opaque_class(category::FakeShadow), query), 0u);
  EXPECT_EQ(stats().selected, 2u);
}

TEST_F(CaptureListTest, LateWindowsAndGpuArraysAreRefused) {
  install(select_all, gfx::get_render_target_layout());
  const gx::draw_route::Query query{};
  DrawClass translucent = opaque_class();
  translucent.key.window = Window::Translucent;
  EXPECT_EQ(select(translucent, query), 0u);
  DrawClass preScene = opaque_class();
  preScene.key.window = Window::PreScene;
  EXPECT_EQ(select(preScene, query), 1u);

  auto& state = gx::g_gxState;
  const auto savedMask = state.gpuArrayMask;
  const GXAttrType savedDesc = state.vtxDesc[GX_VA_POS];
  state.gpuArrayMask = 1u;
  state.vtxDesc[GX_VA_POS] = GX_INDEX16;
  EXPECT_EQ(select(opaque_class(), query), 0u);
  state.vtxDesc[GX_VA_POS] = GX_DIRECT;
  // A GPU array the draw does not index is no reason to refuse it
  EXPECT_EQ(select(opaque_class(), query), 1u);
  state.gpuArrayMask = savedMask;
  state.vtxDesc[GX_VA_POS] = savedDesc;
  EXPECT_EQ(stats().gpuRefused, 1u);
  EXPECT_EQ(stats().selected, 2u);
}

TEST_F(CaptureListTest, SameLayoutCompanionClone) {
  const ConsumerId id = install(select_all, gfx::get_render_target_layout());
  testing::set_pipeline_hooks(&Hooks);
  gx::PipelineConfig config{};
  config.shaderConfig.ext.layer0 = 1;
  on_push(draw_data(3), config, opaque_class(), 1);
  const auto lists = gfx::take_capture_lists();
  ASSERT_NE(lists, nullptr);
  const auto draws = list_draws(lists->lists[id]);
  ASSERT_EQ(draws.size(), 1u);
  // A companion-lane pipeline skips a draw without its group, so the same layout keeps every bind group
  EXPECT_EQ(draws[0].pipeline, 9u);
  EXPECT_EQ(draws[0].bindGroups.companionBindGroup, 5u);
  EXPECT_EQ(draws[0].bindGroups.textureBindGroup, 3u);
  EXPECT_EQ(draws[0].immediateData._pad, 77u);
  EXPECT_EQ(lists->lists[id].layoutKey, gfx::get_render_target_layout().key);
  EXPECT_EQ(g_readyCalls, 0u);
  EXPECT_EQ(stats().cloned, 1u);
  EXPECT_EQ(stats().pending, 0u);
}

TEST_F(CaptureListTest, CrossLayoutClonePendsUntilWarm) {
  gfx::RenderTargetLayout other = gfx::get_render_target_layout();
  other.colorAttachments[1] = {.semantic = gfx::ColorAttachmentSemantic::Normal,
                               .format = wgpu::TextureFormat::RGB10A2Unorm};
  other.colorAttachmentCount = 2;
  gfx::detail::finalize_render_target_layout(other);
  ASSERT_NE(other.key, gfx::get_render_target_layout().key);
  const ConsumerId id = install(select_all, other);
  testing::set_pipeline_hooks(&Hooks);
  gx::PipelineConfig config{};
  config.shaderConfig.ext.output = 1;
  config.shaderConfig.ext.layer0 = 1;

  on_push(draw_data(3), config, opaque_class(), 1);
  EXPECT_EQ(gfx::take_capture_lists(), nullptr);
  EXPECT_EQ(stats().pending, 1u);
  ASSERT_EQ(g_warmed.size(), 1u);
  EXPECT_EQ(g_warmed[0].shaderConfig.ext, gx::RenderExtConfig{});
  EXPECT_TRUE(g_readyExtZero);

  g_readyRef = 42;
  on_push(draw_data(4), config, opaque_class(), 1);
  const auto lists = gfx::take_capture_lists();
  ASSERT_NE(lists, nullptr);
  const auto draws = list_draws(lists->lists[id]);
  ASSERT_EQ(draws.size(), 1u);
  EXPECT_EQ(draws[0].vtxCount, 4u);
  EXPECT_EQ(draws[0].pipeline, 42u);
  // The legacy pipeline from another layout has no companion lane
  EXPECT_EQ(draws[0].bindGroups.companionBindGroup, 0u);
  EXPECT_EQ(draws[0].bindGroups.textureBindGroup, 3u);
  EXPECT_EQ(draws[0].immediateData._pad, 77u);
  EXPECT_EQ(lists->lists[id].layoutKey, other.key);
  EXPECT_EQ(stats().cloned, 1u);
  EXPECT_EQ(stats().pending, 1u);
}

TEST_F(CaptureListTest, ClonesKeepOrderViewportAndScissor) {
  const ConsumerId id = install(select_all, gfx::get_render_target_layout());
  const gfx::Viewport first{0.f, 0.f, 320.f, 240.f, 0.f, 1.f};
  const gfx::Viewport second{16.f, 8.f, 128.f, 64.f, 0.f, 1.f};
  const gfx::ClipRect scissor{4, 4, 100, 50};
  const gx::PipelineConfig config{};
  gfx::set_viewport(first);
  on_push(draw_data(1), config, opaque_class(), 1);
  on_push(draw_data(2), config, opaque_class(), 1);
  gfx::set_viewport(second);
  gfx::set_scissor(scissor);
  on_push(draw_data(3), config, opaque_class(), 1);
  const auto lists = gfx::take_capture_lists();
  ASSERT_NE(lists, nullptr);
  const auto& commands = lists->lists[id].commands;
  using gfx::detail::CommandType;
  ASSERT_EQ(commands.size(), 7u);
  const std::vector<CommandType> expected{CommandType::SetViewport, CommandType::SetScissor,  CommandType::Draw,
                                          CommandType::Draw,        CommandType::SetViewport, CommandType::SetScissor,
                                          CommandType::Draw};
  for (size_t i = 0; i < commands.size(); ++i) {
    EXPECT_EQ(commands[i].type, expected[i]) << i;
  }
  EXPECT_EQ(commands[0].data.setViewport, first);
  EXPECT_EQ(commands[4].data.setViewport, second);
  EXPECT_EQ(commands[5].data.setScissor, scissor);
  EXPECT_EQ(decode(commands[2]).vtxCount, 1u);
  EXPECT_EQ(decode(commands[3]).vtxCount, 2u);
  EXPECT_EQ(decode(commands[6]).vtxCount, 3u);
  EXPECT_EQ(lists->lists[id].draws, 3u);
}

TEST_F(CaptureListTest, OpaqueDoneStageHoldsTheListAndRefusesLateClones) {
  latch_with_opaque_pass();
  const uint64_t discarded = stats().discarded;
  const ConsumerId id = install(select_all, gfx::get_render_target_layout());
  const gx::PipelineConfig config{};
  on_push(draw_data(1), config, opaque_class(), 1);
  on_push(draw_data(2), config, opaque_class(), 1);
  gfx::record_stage(Stage::OpaqueDone);
  const auto lists = opaque_done_lists();
  ASSERT_NE(lists, nullptr);
  EXPECT_EQ(lists->lists[id].draws, 2u);
  EXPECT_EQ(list_draws(lists->lists[id]).size(), 2u);

  on_push(draw_data(3), config, opaque_class(), 1);
  EXPECT_EQ(stats().lateRefused, 1u);
  EXPECT_EQ(stats().cloned, 2u);

  next_frame(true);
  EXPECT_EQ(gfx::take_capture_lists(), nullptr);
  EXPECT_EQ(stats().discarded, discarded);
}

TEST_F(CaptureListTest, UntakenListsAreDiscardedAtFrameEnd) {
  const ConsumerId id = install(select_all, gfx::get_render_target_layout());
  ASSERT_NE(id, InvalidConsumer);
  const uint64_t before = stats().discarded;
  on_push(draw_data(1), gx::PipelineConfig{}, opaque_class(), 1);
  next_frame(false);
  EXPECT_EQ(stats().discarded - before, 1u);
  EXPECT_EQ(gfx::take_capture_lists(), nullptr);
}

TEST_F(CaptureListTest, ReplayResetsEncoderState) {
  gfx::detail::testing::set_current_pipeline(reinterpret_cast<WGPURenderPipeline>(uintptr_t{0x10}));
  gx::testing::mark_pass_bindings();
  ASSERT_FALSE(gx::testing::pass_bindings_reset());
  const gfx::detail::RenderPass info = gfx::detail::replay_pass_info({64, 32, 1});
  EXPECT_EQ(info.colorAttachments[gfx::SceneColorAttachmentIndex].size.width, 64u);
  EXPECT_EQ(info.colorAttachments[gfx::SceneColorAttachmentIndex].size.height, 32u);
  // A first clone whose pipeline matches the last one bound elsewhere must still set it
  gfx::detail::encode_command_list(wgpu::RenderPassEncoder{}, {}, info);
  EXPECT_EQ(gfx::detail::testing::current_pipeline(), nullptr);
  EXPECT_TRUE(gx::testing::pass_bindings_reset());
}
} // namespace
} // namespace aurora::render::scene::capture_list

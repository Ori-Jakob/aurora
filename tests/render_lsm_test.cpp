// The legacy shadow mask: which game draws it clones, when it arms, and the graph it adds

#include <gtest/gtest.h>

#include "render_test_caps.hpp"

#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "render/fakeshadow/lsm.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/post/composite.hpp"
#include "render/scene/capture_list.hpp"
#include "render/scene/classifier.hpp"
#include "render/stage_data.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::fakeshadow::lsm {
namespace {
namespace cat = scene::category;

scene::DrawClass fake_shadow(scene::Window window = scene::Window::Opaque) {
  scene::DrawClass cls;
  cls.key.role = scene::Role::Main;
  cls.key.mainCamera = true;
  cls.key.window = window;
  cls.key.categories = cat::FakeShadow | cat::Decal | cat::NoCapture;
  return cls;
}

void declare_reader(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.after(phase::Composite);
  builder.use(Resource, Access::Sampled, Missing::BindDummyWhite);
  builder.side_effect();
}

void encode_nothing(detail::PassContext&, const void*) {}

detail::PassDesc reader_desc() {
  return {
      .name = "test.lsm_reader",
      .stage = Stage::OpaqueDone,
      .kind = detail::PassKind::Compute,
      .declare = declare_reader,
      .encode = encode_nothing,
  };
}

class Lsm : public ::testing::Test {
protected:
  void SetUp() override {
    scene::capture_list::testing::reset();
    testing::reset();
    gx::g_gxState = {};
    // The game's fakes test depth without writing it
    gx::g_gxState.depthUpdate = false;
    caps = test::full_caps();
  }

  void TearDown() override {
    shutdown();
    testing::reset();
    detail::composite::testing::set_exposure_units(std::nullopt);
    scene::capture_list::testing::reset();
    gx::g_gxState = {};
  }

  detail::GraphEnvironment env(uint32_t samples = 1) const {
    detail::GraphEnvironment out;
    const auto set = [&out](detail::GxResource gx, wgpu::TextureFormat format, wgpu::TextureUsage usage) {
      out.gxFormats[static_cast<size_t>(gx)] = format;
      out.gxUsage[static_cast<size_t>(gx)] = usage;
    };
    constexpr auto Attach = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
    set(detail::GxResource::Color, wgpu::TextureFormat::BGRA8Unorm, Attach);
    set(detail::GxResource::Depth, wgpu::TextureFormat::Depth32Float, Attach);
    set(detail::GxResource::Normal, wgpu::TextureFormat::RGB10A2Unorm, Attach);
    set(detail::GxResource::Fog, wgpu::TextureFormat::RGBA8Uint, Attach);
    set(detail::GxResource::Surface, wgpu::TextureFormat::RGBA16Uint, Attach);
    out.gxSampleCount = samples;
    out.efbSize = {640, 480};
    out.outputSize = {640, 480};
    out.sceneSize = {640, 480};
    return out;
  }

  std::shared_ptr<detail::CompiledGraph> compile(const Settings& settings, const detail::GraphEnvironment& graphEnv) {
    passes = {pass_desc(), reader_desc()};
    detail::GraphInputs inputs{.passes = passes, .settings = &settings, .caps = &caps, .env = graphEnv};
    inputs.sceneDepthSource = DepthSource::Direct;
    return detail::compile_graph(inputs);
  }

  static Settings pbr(float legacyBlend) {
    Settings settings;
    settings.pbr = true;
    settings.legacyBlend = legacyBlend;
    return settings;
  }

  static const detail::CompiledNode* node(const detail::CompiledGraph& graph, std::string_view name) {
    const detail::NodeId id = graph.find_node(name);
    return id == detail::InvalidNode ? nullptr : &graph.nodes[id];
  }

  void latch(const Settings& settings, const detail::GraphEnvironment& graphEnv, const detail::CompiledGraph* graph) {
    on_latch({.settings = &settings, .caps = &caps, .env = &graphEnv, .sceneDepth = DepthSource::Direct}, graph);
    apply_frame_state();
    scene::capture_list::apply_frame_state();
  }

  Caps caps;
  std::vector<detail::PassDesc> passes;
};

TEST_F(Lsm, SelectsFakeShadowOnly) {
  EXPECT_TRUE(select_draw(fake_shadow()));
  scene::DrawClass other = fake_shadow();
  other.key.categories = cat::Static | cat::Decal;
  EXPECT_FALSE(select_draw(other));
  other = fake_shadow();
  other.offscreen = true;
  EXPECT_FALSE(select_draw(other));
  other = fake_shadow();
  other.key.role = scene::Role::Shadow;
  EXPECT_FALSE(select_draw(other));
  other = fake_shadow();
  other.key.mainCamera = false;
  EXPECT_FALSE(select_draw(other));

  initialize();
  testing::set_consumed(true);
  const Settings settings = pbr(0.f);
  const detail::GraphEnvironment graphEnv = env();
  latch(settings, graphEnv, nullptr);
  const uint8_t bit = static_cast<uint8_t>(1u << testing::consumer());
  const gx::draw_route::Query query{};
  EXPECT_EQ(scene::capture_list::select(fake_shadow(), query), bit);
  EXPECT_EQ(scene::capture_list::select(other, query), 0u);
  EXPECT_EQ(stats().selected, 1u);
  // The mask attaches scene depth read-only, so a depth-writing fake is refused rather than replayed
  gx::g_gxState.depthCompare = true;
  gx::g_gxState.depthUpdate = true;
  EXPECT_EQ(scene::capture_list::select(fake_shadow(), query), 0u);
  EXPECT_EQ(stats().depthRefused, 1u);
  gx::g_gxState.depthUpdate = false;
  EXPECT_EQ(scene::capture_list::select(fake_shadow(), query), bit);
  EXPECT_EQ(stats().selected, 2u);
}

TEST_F(Lsm, RefusedAfterOpaqueDone) {
  initialize();
  testing::set_consumed(true);
  const Settings settings = pbr(0.f);
  const detail::GraphEnvironment graphEnv = env();
  latch(settings, graphEnv, nullptr);
  const gx::draw_route::Query query{};
  for (const scene::Window window : {scene::Window::Translucent, scene::Window::Post, scene::Window::SceneEnd,
                                     scene::Window::Hud, scene::Window::Final}) {
    EXPECT_FALSE(select_draw(fake_shadow(window))) << static_cast<int>(window);
    EXPECT_EQ(scene::capture_list::select(fake_shadow(window), query), 0u) << static_cast<int>(window);
  }
  EXPECT_EQ(stats().selected, 0u);
}

TEST_F(Lsm, EnabledOnlyWhenConsumed) {
  initialize();
  const detail::GraphEnvironment graphEnv = env();
  const Settings blendZero = pbr(0.f);
  const auto graph = compile(blendZero, graphEnv);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(node(*graph, Pass), nullptr);
  ASSERT_EQ(node(*graph, Pass)->status, detail::CullReason::None);
  const gx::draw_route::Query query{};
  struct Case {
    bool flagPbr;
    uint16_t view;
    bool withGraph;
    bool consumed;
  };
  for (const Case& c : {Case{true, 0, true, true}, Case{false, 0, true, false}, Case{true, 1, true, false},
                        Case{true, 0, false, false}}) {
    detail::composite::testing::set_exposure_units(c.flagPbr);
    Settings settings = blendZero;
    settings.debugView = c.view;
    latch(settings, graphEnv, c.withGraph ? graph.get() : nullptr);
    EXPECT_EQ(consumed(), c.consumed);
    EXPECT_EQ(consumed_fifo(), c.consumed);
    EXPECT_EQ(scene::capture_list::select(fake_shadow(), query) != 0, c.consumed);
  }
  EXPECT_EQ(stats().consumedLatches, 1u);
  EXPECT_EQ(stats().declaredLatches, 3u);
}

TEST_F(Lsm, DeclareIgnoresBlend) {
  const detail::GraphEnvironment graphEnv = env();
  const auto zero = compile(pbr(0.f), graphEnv);
  const auto half = compile(pbr(0.5f), graphEnv);
  const auto one = compile(pbr(1.f), graphEnv);
  ASSERT_NE(zero, nullptr);
  ASSERT_NE(half, nullptr);
  ASSERT_NE(one, nullptr);
  ASSERT_NE(node(*zero, Pass), nullptr);
  EXPECT_EQ(zero->graphHash, half->graphHash);
  EXPECT_EQ(zero->graphHash, one->graphHash);
  EXPECT_EQ(node(*zero, Pass)->status, detail::CullReason::None);
  const auto off = compile(Settings{}, graphEnv);
  ASSERT_NE(off, nullptr);
  ASSERT_NE(node(*off, Pass), nullptr);
  EXPECT_EQ(node(*off, Pass)->status, detail::CullReason::Skipped);
  const auto msaa = compile(pbr(0.f), env(4));
  ASSERT_NE(msaa, nullptr);
  ASSERT_NE(node(*msaa, Pass), nullptr);
  EXPECT_EQ(node(*msaa, Pass)->status, detail::CullReason::Skipped);
  detail::GraphEnvironment noSurface = graphEnv;
  noSurface.gxFormats[static_cast<size_t>(detail::GxResource::Surface)] = wgpu::TextureFormat::Undefined;
  const Settings blend = pbr(0.f);
  EXPECT_FALSE(declared({.settings = &blend, .caps = &caps, .env = &noSurface}));
  EXPECT_TRUE(declared({.settings = &blend, .caps = &caps, .env = &graphEnv}));
}

TEST_F(Lsm, ClearUsesEfbClearAlpha) {
  const detail::PassDesc desc = pass_desc();
  ASSERT_NE(desc.stageParams, nullptr);
  EXPECT_EQ(desc.paramsSize, sizeof(Params));
  const detail::StageData stage;
  for (const float alpha : {0.f, 0.25f, 1.f, 3.f}) {
    gx::g_gxState.clearColor = {0.2f, 0.4f, 0.6f, alpha};
    std::array<std::byte, sizeof(Params)> bytes{};
    desc.stageParams(stage, nullptr, bytes);
    Params params;
    std::memcpy(&params, bytes.data(), sizeof(params));
    EXPECT_EQ(params.clearAlpha, std::min(alpha, 1.f)) << alpha;
  }
}

TEST_F(Lsm, DepthReadNoCopy) {
  const detail::GraphEnvironment graphEnv = env();
  const auto graph = compile(pbr(0.f), graphEnv);
  ASSERT_NE(graph, nullptr);
  const detail::NodeId id = graph->find_node(Pass);
  ASSERT_NE(id, detail::InvalidNode);
  const detail::CompiledNode& mask = graph->nodes[id];
  EXPECT_EQ(mask.passKind, detail::PassKind::Render);
  EXPECT_EQ(mask.stage, Stage::OpaqueDone);
  EXPECT_EQ(mask.order, Order);
  const auto uses = graph->node_uses(id);
  ASSERT_EQ(uses.size(), 2u);
  EXPECT_EQ(graph->resources[uses[0].resource].name, Resource);
  EXPECT_EQ(uses[0].access, Access::ColorWrite);
  EXPECT_EQ(graph->resources[uses[0].resource].tex.format, Format);
  EXPECT_EQ(graph->resources[uses[1].resource].name, resource::GxDepth);
  EXPECT_EQ(uses[1].access, Access::DepthRead);
  for (const detail::CompiledResource& resource : graph->resources) {
    EXPECT_NE(resource.name, "lsm.depth");
  }
  const gfx::RenderTargetLayout target = layout(wgpu::TextureFormat::Depth32Float);
  EXPECT_EQ(target.colorAttachmentCount, 1u);
  EXPECT_EQ(target.colorAttachments[0].format, Format);
  EXPECT_EQ(target.depthStencilFormat, wgpu::TextureFormat::Depth32Float);
  EXPECT_EQ(target.sampleCount, 1u);
  EXPECT_NE(target.key, layout(wgpu::TextureFormat::Depth24Plus).key);
}

TEST_F(Lsm, CulledWithoutClones) {
  initialize();
  const detail::PassDesc desc = pass_desc();
  ASSERT_NE(desc.enabled, nullptr);
  const detail::StageData stage;
  testing::set_consumed(true);
  apply_frame_state();
  EXPECT_TRUE(consumed_fifo());
  EXPECT_EQ(scene::capture_list::queued(testing::consumer()), 0u);
  EXPECT_FALSE(desc.enabled(stage, nullptr));
  testing::set_consumed(false);
  apply_frame_state();
  EXPECT_FALSE(desc.enabled(stage, nullptr));
  EXPECT_FALSE(frame_enabled(stage));
}

TEST_F(Lsm, OwnerSurvivesCaptureListReset) {
  initialize();
  const uint32_t first = testing::consumer();
  ASSERT_NE(first, scene::capture_list::InvalidConsumer);
  scene::capture_list::testing::reset();
  const scene::capture_list::ConsumerId other = scene::capture_list::register_consumer(
      {.name = "other", .select = [](const scene::DrawClass&, void*) noexcept { return true; }});
  ASSERT_EQ(other, first);
  testing::set_consumed(true);
  const Settings settings = pbr(0.f);
  const detail::GraphEnvironment graphEnv = env();
  latch(settings, graphEnv, nullptr);
  EXPECT_NE(testing::consumer(), other);
  EXPECT_EQ(scene::capture_list::select(fake_shadow(), {}), static_cast<uint8_t>(1u << testing::consumer()));
}

TEST(LsmRules, TermDecodesFogAndKeepsWhiteExact) {
  EXPECT_EQ(term({1.f, 1.f, 1.f}, 0.7f, {0.9f, 0.5f, 0.1f}), (std::array<float, 3>{1.f, 1.f, 1.f}));
  EXPECT_EQ(term({0.5f, 0.25f, 0.f}, 0.f, {0.3f, 0.3f, 0.3f}), (std::array<float, 3>{0.5f, 0.25f, 0.f}));
  // A shadow of strength a under fog q blends to q*fc*a + 1 - a over white, which decodes back to 1 - a
  const float a = 0.4f;
  const float q = 0.5f;
  const std::array<float, 3> fc{0.8f, 0.6f, 0.2f};
  std::array<float, 3> mask{};
  for (size_t c = 0; c < 3; ++c) {
    mask[c] = q * fc[c] * a + 1.f - a;
  }
  const std::array<float, 3> m = term(mask, q, fc);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(m[c], 1.f - a, 1e-6f) << c;
  }
  EXPECT_EQ(term({0.2f, 0.2f, 0.2f}, 1.f, {1.f, 0.999f, 0.5f})[0], 1.f);
  EXPECT_EQ(term({0.f, 0.f, 0.f}, 0.5f, {1.f, 1.f, 1.f}), (std::array<float, 3>{0.f, 0.f, 0.f}));
}
} // namespace
} // namespace aurora::render::fakeshadow::lsm

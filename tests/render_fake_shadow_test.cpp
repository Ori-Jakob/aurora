#include <gtest/gtest.h>

#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "gx/texture.hpp"
#include "render/fakeshadow/texture_gate.hpp"
#include "render/frame_state.hpp"
#include "render/hints.hpp"
#include "render/material/registry.hpp"
#include "render/scene/classify.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"

#include <SDL3/SDL_stdinc.h>
#include <aurora/texture.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace aurora::render::fakeshadow {
namespace {
namespace category = scene::category;

constexpr gfx::BindGroupRef BaseGroup = 0x1111;
constexpr gfx::BindGroupRef NeutralGroups = 0x100000;
constexpr uint16_t AllBase = 0x1FFF;

Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

Rule rule(uint16_t id, Neutral neutral, Gate gate, uint16_t scope) {
  return {
      .materialId = id,
      .neutral = neutral,
      .gateBit = static_cast<uint8_t>(1u << static_cast<uint32_t>(gate)),
      .scopeMask = scope != 0 ? scope : AllBase,
      .scoped = scope != 0,
  };
}

scene::DrawClass main_class(uint32_t categories) {
  scene::DrawClass cls;
  cls.key.role = scene::Role::Main;
  cls.key.categories = categories;
  return cls;
}

size_t at(Outcome outcome) { return static_cast<size_t>(outcome); }

gx::PipelineConfig alpha_blend(GXBlendFactor dst = GX_BL_INVSRCALPHA) {
  gx::PipelineConfig config{};
  config.blendMode = GX_BM_BLEND;
  config.blendFacSrc = GX_BL_SRCALPHA;
  config.blendFacDst = dst;
  return config;
}

void kikage_stage(gx::PipelineConfig& config, GXTexMapID map = GX_TEXMAP0) {
  gx::TevStage& stage = config.shaderConfig.tevStages[0];
  stage.alphaPass = {GX_CA_ZERO, GX_CA_RASA, GX_CA_TEXA, GX_CA_ZERO};
  stage.texMapId = map;
  stage.texCoordId = GX_TEXCOORD0;
  config.shaderConfig.tevStageCount = 1;
}

struct Builder {
  static inline int calls = 0;
  static inline uint8_t white = 0;
  static inline uint8_t clear = 0;
  static inline bool baseLevel = true;

  static gx::GXBindGroups build(const gx::ShaderInfo&, uint8_t whiteMask, uint8_t clearMask, bool level) noexcept {
    ++calls;
    white = whiteMask;
    clear = clearMask;
    baseLevel = level;
    return {.textureBindGroup = NeutralGroups + calls * 0x10000 + whiteMask * 0x100 + clearMask};
  }
};

struct Probe {
  static inline int views = 0;
  static inline int observes = 0;
  static inline uint32_t observedInstances = UINT32_MAX;

  static bool push(gx::DrawData&, const gx::PipelineConfig&, const gx::ShaderInfo&, const scene::DrawClass&) noexcept {
    ++views;
    return true;
  }
  static void observe(gx::DrawData& draw, const gx::PipelineConfig&, const gx::ShaderInfo&,
                      const scene::DrawClass&) noexcept {
    ++observes;
    observedInstances = draw.instanceCount;
  }
};

struct Source {
  static inline bool value = false;
  static bool get(const detail::FrameState&) noexcept { return value; }
};

class RenderFakeShadow : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
    testing::set_neutral_builder(&Builder::build);
    Builder::calls = 0;
    Probe::views = 0;
    Probe::observes = 0;
    Probe::observedInstances = UINT32_MAX;
    Source::value = false;
  }

  void TearDown() override {
    detail::remove_hints();
    detail::testing::reset_hints();
    scene::testing::reset();
    detail::testing::reset();
    texture::clear_material_manifests();
    material::latch();
    gx::g_gxState = {};
  }

  static const gx::hints::Sink& sink() { return *gx::hints::sink(); }

  static void main_scene() {
    const Mat4x4<float> proj = perspective();
    const gx::hints::Camera main{
        .role = AURORA_CAMERA_MAIN,
        .proj = proj,
        .type = GX_PERSPECTIVE,
        .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
    };
    sink().camera(sink().userData, main);
    sink().stage(sink().userData, Stage::SceneBegin);
    gx::g_gxState.proj = main.gxProj;
  }

  static void scope(uint32_t categories, AuroraCameraRole role = AURORA_CAMERA_INHERIT) {
    sink().scopePush(sink().userData, {.categories = categories, .role = role});
  }
  static void pop() { sink().scopePop(sink().userData); }

  static void latch(bool textureMaterials = true, GameShadows game = GameShadows::Auto) {
    detail::FrameState state;
    state.settings.textureMaterials = textureMaterials;
    state.settings.shadows.game = game;
    scene::apply_frame_state(state);
  }

  static std::shared_ptr<const TextureReport> seal_report() {
    detail::begin_hint_frame();
    return latest_report();
  }

  void bind(uint32_t slot, uint32_t id) {
    gfx::TextureBind& bind = gx::g_gxState.textures[slot];
    bind.ref =
        std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                          wgpu::Extent3D{32, 32, 1}, wgpu::TextureFormat::RGBA8Unorm, 1, GX_TF_RGBA8);
    bind.material = {id, 0};
    m_info.sampledTextures.set(slot);
  }

  static gx::DrawData draw_data(gfx::BindGroupRef base = BaseGroup) {
    gx::DrawData draw{};
    draw.bindGroups.textureBindGroup = base;
    draw.instanceCount = 1;
    return draw;
  }

  static gx::draw_route::Decision route() {
    const gx::draw_route::Router* router = gx::draw_route::router();
    return router->route(router->userData, {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3});
  }

  bool push(gx::DrawData& draw) const {
    const gx::draw_route::Router* router = gx::draw_route::router();
    return router->push(router->userData, draw, m_config, m_info);
  }

  bool push_draw(gx::DrawData& draw) const {
    route();
    return push(draw);
  }

  bool edited() const {
    gx::DrawData draw = draw_data();
    return push_draw(draw);
  }

  gx::PipelineConfig m_config{};
  gx::ShaderInfo m_info{};
};

TEST_F(RenderFakeShadow, InactiveIsUntouched) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  const auto expect_untouched = [this] {
    gx::DrawData draw = draw_data();
    EXPECT_FALSE(push_draw(draw));
    EXPECT_EQ(draw.bindGroups.textureBindGroup, BaseGroup);
    EXPECT_EQ(draw.instanceCount, 1u);
    EXPECT_FALSE(lead_flagged());
    EXPECT_EQ(last_edit().flagged, 0u);
  };

  testing::set_rules(rules);
  latch();
  expect_untouched();

  testing::set_rules({});
  set_force(Gate::Light, Force::On);
  latch();
  expect_untouched();

  testing::set_rules(rules);
  latch(false);
  expect_untouched();
  EXPECT_EQ(Builder::calls, 0);

  latch();
  EXPECT_TRUE(edited());
  EXPECT_EQ(Builder::calls, 1);
}

TEST_F(RenderFakeShadow, WhiteOnlyInScope) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain | category::Static)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  scope(category::Static | category::Terrain);
  bind(0, 5);

  gx::DrawData draw = draw_data();
  EXPECT_TRUE(push_draw(draw));
  EXPECT_EQ(Builder::calls, 1);
  EXPECT_EQ(Builder::white, 0x1u);
  EXPECT_EQ(Builder::clear, 0x0u);
  EXPECT_FALSE(Builder::baseLevel);
  EXPECT_NE(draw.bindGroups.textureBindGroup, BaseGroup);
  EXPECT_EQ(draw.instanceCount, 1u);
  EXPECT_EQ(last_edit().edit, Edit::Substituted);
  EXPECT_TRUE(last_edit().applied);
  EXPECT_EQ(last_edit().flagged, 0x1u);
  EXPECT_EQ(last_edit().white, 0x1u);

  pop();
  scope(category::Water);
  draw = draw_data();
  EXPECT_FALSE(push_draw(draw));
  EXPECT_EQ(draw.bindGroups.textureBindGroup, BaseGroup);

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->counts.size(), 1u);
  const OutcomeCounts& counts = report->counts[0];
  EXPECT_EQ(counts[at(Outcome::Bound)], 2u);
  EXPECT_EQ(counts[at(Outcome::InScope)], 1u);
  EXPECT_EQ(counts[at(Outcome::White)], 1u);
  EXPECT_EQ(counts[at(Outcome::OutOfScope)], 1u);
}

TEST_F(RenderFakeShadow, UnscopedMatchesBaseCategoriesOnly) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, 0)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  bind(0, 5);
  EXPECT_FALSE(edited());
  scope(category::Particle);
  EXPECT_TRUE(edited());
  pop();
  scope(category::NoPbr);
  EXPECT_FALSE(edited());

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->table->unscoped, 1u);
  const OutcomeCounts& counts = report->counts[0];
  EXPECT_EQ(counts[at(Outcome::Bound)], 3u);
  EXPECT_EQ(counts[at(Outcome::White)], 1u);
  EXPECT_EQ(counts[at(Outcome::OutOfScope)], 2u);
}

TEST_F(RenderFakeShadow, ExclusionsKeepDraw) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Decal | category::Terrain)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  bind(0, 5);

  scope(category::FakeShadow | category::Decal);
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().flagged, 0x1u);
  pop();
  scope(category::Terrain, AURORA_CAMERA_REFLECTION);
  EXPECT_FALSE(edited());
  pop();
  scope(category::Terrain);
  m_info.sampledIndTextures.set(0);
  EXPECT_FALSE(edited());
  m_info.sampledIndTextures.reset();
  gx::g_gxState.textures[0].material = {};
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().flagged, 0u);
  gx::g_gxState.textures[0] = {};
  gx::g_gxState.textures[0].material = {5, 0};
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().flagged, 0u);
  EXPECT_EQ(Builder::calls, 0);

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  const OutcomeCounts& counts = report->counts[0];
  EXPECT_EQ(counts[at(Outcome::Bound)], 3u);
  EXPECT_EQ(counts[at(Outcome::EntryOwned)], 1u);
  EXPECT_EQ(counts[at(Outcome::OutOfRole)], 1u);
  EXPECT_EQ(counts[at(Outcome::RefusedIndirect)], 1u);
  EXPECT_EQ(counts[at(Outcome::InScope)], 0u);

  scene::DrawClass cls = main_class(category::Terrain);
  const Slot slot{.index = 0, .rule = 0};
  Outcome result{};
  const DrawFacts drawFacts{.boundMask = 0x1};
  EXPECT_EQ(decide({&slot, 1}, cls, drawFacts, 0x1, rules, {&result, 1}).white, 0x1u);
  EXPECT_EQ(result, Outcome::White);
  cls.offscreen = true;
  EXPECT_EQ(decide({&slot, 1}, cls, drawFacts, 0x1, rules, {&result, 1}).white, 0x0u);
  EXPECT_EQ(result, Outcome::OutOfRole);
}

TEST_F(RenderFakeShadow, GuardsRefuse) {
  const std::array rules{
      rule(1, Neutral::White, Gate::Light, category::Terrain),
      rule(2, Neutral::Clear, Gate::Light, category::Terrain),
      rule(3, Neutral::Skip, Gate::Light, category::Terrain),
  };
  const scene::DrawClass cls = main_class(category::Terrain);
  const auto outcome = [&](uint16_t ruleIndex, const gx::PipelineConfig& config, uint8_t slotIndex = 0) {
    const Slot slot{.index = slotIndex, .rule = ruleIndex};
    DrawFacts drawFacts = facts(config);
    drawFacts.boundMask = static_cast<uint8_t>(1u << slotIndex);
    Outcome result{};
    decide({&slot, 1}, cls, drawFacts, 0x1, rules, {&result, 1});
    return result;
  };

  const gx::PipelineConfig plain{};
  EXPECT_EQ(outcome(0, plain), Outcome::White);
  EXPECT_EQ(outcome(1, plain), Outcome::RefusedClear);
  EXPECT_EQ(outcome(2, plain), Outcome::Skipped);

  gx::PipelineConfig alphaTest{};
  alphaTest.shaderConfig.alphaCompare.comp0 = GX_GEQUAL;
  EXPECT_TRUE(facts(alphaTest).alphaTest);
  EXPECT_EQ(outcome(0, alphaTest), Outcome::RefusedCoverage);
  EXPECT_EQ(outcome(1, alphaTest), Outcome::RefusedCoverage);
  EXPECT_EQ(outcome(2, alphaTest), Outcome::Skipped);

  gx::PipelineConfig kikage = alpha_blend();
  kikage_stage(kikage);
  const DrawFacts kikageFacts = facts(kikage);
  EXPECT_EQ(kikageFacts.texAlphaMask, 0x1u);
  EXPECT_TRUE(kikageFacts.alphaConsumed);
  EXPECT_TRUE(kikageFacts.srcAlphaBlend);
  EXPECT_FALSE(kikageFacts.writes);
  EXPECT_EQ(outcome(0, kikage), Outcome::RefusedWhite);
  EXPECT_EQ(outcome(1, kikage), Outcome::Clear);
  EXPECT_EQ(outcome(0, kikage, 1), Outcome::White);

  gx::PipelineConfig opaqueKikage{};
  kikage_stage(opaqueKikage);
  EXPECT_EQ(outcome(0, opaqueKikage), Outcome::White);
  opaqueKikage.alphaUpdate = true;
  opaqueKikage.dstAlpha = UINT32_MAX;
  EXPECT_EQ(outcome(0, opaqueKikage), Outcome::RefusedWhite);
  EXPECT_EQ(outcome(2, opaqueKikage), Outcome::RefusedWrites);

  EXPECT_EQ(outcome(1, alpha_blend(GX_BL_ONE)), Outcome::Clear);
  gx::PipelineConfig inverse = alpha_blend();
  inverse.blendFacSrc = GX_BL_ONE;
  EXPECT_EQ(outcome(1, inverse), Outcome::RefusedClear);

  gx::PipelineConfig depth{};
  depth.depthCompare = true;
  EXPECT_EQ(outcome(2, depth), Outcome::Skipped);
  depth.depthUpdate = true;
  EXPECT_EQ(outcome(2, depth), Outcome::RefusedWrites);
}

TEST_F(RenderFakeShadow, SkipNeedsEveryBoundSlotNeutralised) {
  const std::array rules{
      rule(1, Neutral::Skip, Gate::Light, category::Terrain),
      rule(2, Neutral::White, Gate::Light, category::Terrain),
      rule(3, Neutral::White, Gate::Light, category::Water),
  };
  const scene::DrawClass cls = main_class(category::Terrain);
  std::array<Outcome, 2> results{};
  const auto run = [&](std::span<const Slot> slots, DrawFacts drawFacts) {
    return decide(slots, cls, drawFacts, 0x1, rules, {results.data(), slots.size()});
  };

  const std::array skipOnly{Slot{.index = 0, .rule = 0}};
  Decision decision = run(skipOnly, {.boundMask = 0b11});
  EXPECT_EQ(results[0], Outcome::RefusedShared);
  EXPECT_EQ(decision.skip | decision.white | decision.clear, 0u);
  decision = run(skipOnly, {.boundMask = 0b01});
  EXPECT_EQ(results[0], Outcome::Skipped);
  EXPECT_EQ(decision.skip, 0b01u);

  const std::array withWhite{Slot{.index = 0, .rule = 0}, Slot{.index = 1, .rule = 1}};
  decision = run(withWhite, {.boundMask = 0b11});
  EXPECT_EQ(results[0], Outcome::Skipped);
  EXPECT_EQ(results[1], Outcome::Skipped);
  EXPECT_EQ(decision.skip, 0b11u);
  EXPECT_EQ(decision.white, 0u);

  decision = run(withWhite, {.boundMask = 0b111});
  EXPECT_EQ(results[0], Outcome::RefusedShared);
  EXPECT_EQ(results[1], Outcome::White);
  EXPECT_EQ(decision.skip, 0u);
  EXPECT_EQ(decision.white, 0b10u);

  const std::array withScoped{Slot{.index = 0, .rule = 0}, Slot{.index = 1, .rule = 2}};
  decision = run(withScoped, {.boundMask = 0b11});
  EXPECT_EQ(results[0], Outcome::RefusedShared);
  EXPECT_EQ(results[1], Outcome::OutOfScope);
  EXPECT_EQ(decision.skip | decision.white | decision.clear, 0u);

  decision = run(withWhite, {.alphaConsumed = true, .texAlphaMask = 0b10, .boundMask = 0b11});
  EXPECT_EQ(results[0], Outcome::RefusedShared);
  EXPECT_EQ(results[1], Outcome::RefusedWhite);
  EXPECT_EQ(decision.skip | decision.white | decision.clear, 0u);
}

TEST_F(RenderFakeShadow, SkipEmptiesDrawAndSkipsView) {
  const std::array rules{rule(5, Neutral::Skip, Gate::Light, category::Terrain)};
  static constexpr scene::ViewHooks Hooks{.push = &Probe::push};
  static constexpr scene::ObserverHooks Observer{.observe = &Probe::observe};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  scene::set_view_hooks(&Hooks);
  scene::set_observer_hooks(&Observer);
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, 5);

  gx::DrawData draw = draw_data();
  EXPECT_TRUE(push_draw(draw));
  EXPECT_EQ(draw.instanceCount, 0u);
  EXPECT_EQ(draw.bindGroups.textureBindGroup, BaseGroup);
  EXPECT_EQ(Probe::views, 0);
  EXPECT_EQ(Probe::observes, 1);
  EXPECT_EQ(Probe::observedInstances, 0u);
  EXPECT_EQ(last_edit().edit, Edit::Skipped);
  EXPECT_EQ(last_edit().skip, 0x1u);
  EXPECT_EQ(Builder::calls, 0);

  m_config.depthCompare = true;
  m_config.depthUpdate = true;
  draw = draw_data();
  EXPECT_TRUE(push_draw(draw));
  EXPECT_EQ(draw.instanceCount, 1u);
  EXPECT_EQ(Probe::views, 1);
  EXPECT_EQ(Probe::observes, 2);
  EXPECT_EQ(last_edit().edit, Edit::Untouched);
}

TEST_F(RenderFakeShadow, LastEditPerDraw) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  EXPECT_TRUE(edited());
  EXPECT_EQ(last_edit().white, 0x1u);

  gx::g_gxState.textures[0].material = {};
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().edit, Edit::Untouched);
  EXPECT_EQ(last_edit().flagged, 0u);
  EXPECT_EQ(last_edit().white, 0u);

  gx::g_gxState.textures[0].material = {5, 0};
  pop();
  scope(category::Water);
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().flagged, 0x1u);
  EXPECT_EQ(last_edit().white, 0u);
  EXPECT_FALSE(last_edit().applied);
}

TEST_F(RenderFakeShadow, GateFlipAppliesNextFrame) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  const uint64_t generation = gx::texture::current_bind_generation();

  set_force(Gate::Light, Force::Off);
  latch();
  EXPECT_FALSE(edited());
  set_force(Gate::Light, Force::On);
  EXPECT_FALSE(edited());
  latch();
  gx::DrawData draw = draw_data();
  EXPECT_TRUE(push_draw(draw));
  EXPECT_NE(draw.bindGroups.textureBindGroup, BaseGroup);
  set_force(Gate::Light, Force::Off);
  latch();
  draw = draw_data();
  EXPECT_FALSE(push_draw(draw));
  EXPECT_EQ(draw.bindGroups.textureBindGroup, BaseGroup);
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);
}

TEST_F(RenderFakeShadow, AutoNeedsSourceAndNotKeep) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  const auto on = [this](GameShadows game) {
    latch(true, game);
    return edited();
  };

  EXPECT_FALSE(on(GameShadows::Auto));
  set_gate_source(Gate::Light, &Source::get);
  Source::value = true;
  EXPECT_FALSE(on(GameShadows::Keep));
  EXPECT_TRUE(on(GameShadows::Auto));
  EXPECT_TRUE(on(GameShadows::Remove));
  set_gate_source(Gate::Cloud, &Source::get);
  Source::value = false;
  EXPECT_FALSE(on(GameShadows::Auto));
  set_force(Gate::Light, Force::On);
  EXPECT_TRUE(on(GameShadows::Keep));
  set_force(Gate::Light, Force::Off);
  Source::value = true;
  EXPECT_FALSE(on(GameShadows::Auto));

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  const GateView light = report->gates[static_cast<size_t>(Gate::Light)];
  EXPECT_TRUE(light.live);
  EXPECT_FALSE(light.effective);
  EXPECT_EQ(light.force, Force::Off);
  EXPECT_TRUE(report->gates[static_cast<size_t>(Gate::Cloud)].effective);
  EXPECT_FALSE(report->gates[static_cast<size_t>(Gate::Ao)].live);
}

TEST_F(RenderFakeShadow, MergeBreakFollowsFlaggedLead) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  bind(0, 5);

  scope(category::Water);
  EXPECT_FALSE(edited());
  EXPECT_TRUE(lead_flagged());
  EXPECT_FALSE(route().breakMerge);
  pop();
  scope(category::Terrain);
  EXPECT_TRUE(route().breakMerge);
  gx::DrawData draw = draw_data();
  EXPECT_TRUE(push(draw));
  EXPECT_FALSE(lead_flagged());
  pop();
  scope(category::Water);
  EXPECT_FALSE(route().breakMerge);

  gx::g_gxState.textures[0].material = {};
  EXPECT_FALSE(edited());
  EXPECT_FALSE(lead_flagged());

  gx::g_gxState.textures[0].material = {5, 0};
  EXPECT_FALSE(edited());
  EXPECT_TRUE(lead_flagged());
  set_force(Gate::Light, Force::Off);
  latch();
  EXPECT_FALSE(lead_flagged());
  EXPECT_FALSE(edited());
  EXPECT_FALSE(lead_flagged());
  pop();
  scope(category::Terrain);
  EXPECT_FALSE(route().breakMerge);
}

TEST_F(RenderFakeShadow, ObserveCountsWithoutChanging) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  set_stats_enabled(true);
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  gx::DrawData draw = draw_data();
  EXPECT_FALSE(push_draw(draw));
  EXPECT_EQ(draw.bindGroups.textureBindGroup, BaseGroup);
  EXPECT_EQ(Builder::calls, 0);
  EXPECT_FALSE(lead_flagged());
  EXPECT_EQ(last_edit().flagged, 0x1u);
  EXPECT_FALSE(last_edit().applied);

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->observing);
  EXPECT_FALSE(report->active);
  const OutcomeCounts& counts = report->counts[0];
  EXPECT_EQ(counts[at(Outcome::Bound)], 1u);
  EXPECT_EQ(counts[at(Outcome::InScope)], 1u);
  EXPECT_EQ(counts[at(Outcome::GateOff)], 1u);

  set_stats_enabled(false);
  latch();
  main_scene();
  scope(category::Terrain);
  EXPECT_FALSE(edited());
  EXPECT_EQ(last_edit().flagged, 0u);
}

TEST_F(RenderFakeShadow, MemoBuildsOncePerFramePerVariant) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain),
                         rule(6, Neutral::White, Gate::Light, category::Terrain)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, 5);

  gx::DrawData first = draw_data();
  gx::DrawData second = draw_data();
  EXPECT_TRUE(push_draw(first));
  EXPECT_TRUE(push_draw(second));
  EXPECT_EQ(Builder::calls, 1);
  EXPECT_EQ(first.bindGroups.textureBindGroup, second.bindGroups.textureBindGroup);

  gx::DrawData otherBase = draw_data(BaseGroup + 1);
  EXPECT_TRUE(push_draw(otherBase));
  EXPECT_EQ(Builder::calls, 2);
  bind(1, 6);
  EXPECT_TRUE(edited());
  EXPECT_EQ(Builder::calls, 3);
  EXPECT_EQ(Builder::white, 0b11u);

  latch();
  gx::DrawData next = draw_data();
  EXPECT_TRUE(push_draw(next));
  EXPECT_EQ(Builder::calls, 4);
}

TEST_F(RenderFakeShadow, ReportTotalsAndReset) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain),
                         rule(6, Neutral::Clear, Gate::Cloud, category::Water)};
  testing::set_rules(rules);
  set_force(Gate::Light, Force::On);
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, 5);
  EXPECT_TRUE(edited());
  EXPECT_TRUE(edited());
  gx::g_gxState.textures[0].material = {6, 0};
  EXPECT_FALSE(edited());

  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->active);
  EXPECT_FALSE(report->observing);
  EXPECT_FALSE(report->hintsOff);
  EXPECT_TRUE(report->gates[static_cast<size_t>(Gate::Light)].effective);
  EXPECT_EQ(report->gates[static_cast<size_t>(Gate::Light)].force, Force::On);
  ASSERT_EQ(report->table->rules.size(), 2u);
  ASSERT_EQ(report->counts.size(), 2u);
  EXPECT_EQ(report->counts[0][at(Outcome::White)], 2u);
  EXPECT_EQ(report->counts[1][at(Outcome::OutOfScope)], 1u);

  latch();
  main_scene();
  scope(category::Terrain);
  gx::g_gxState.textures[0].material = {5, 0};
  EXPECT_TRUE(edited());
  reset_frame();
  EXPECT_EQ(force(Gate::Light), Force::On);
  EXPECT_FALSE(lead_flagged());
  EXPECT_EQ(last_edit().flagged, 0u);
  EXPECT_TRUE(edited());
  const auto after = seal_report();
  ASSERT_NE(after, report);
  EXPECT_EQ(after->counts[0][at(Outcome::White)], 1u);
  EXPECT_EQ(after->table, report->table);

  set_force(Gate::Light, Force::Off);
  latch();
  const auto idle = seal_report();
  ASSERT_NE(idle, after);
  EXPECT_FALSE(idle->active);
  latch();
  EXPECT_EQ(seal_report(), idle);
}

TEST_F(RenderFakeShadow, RulesFromRegistrySnapshot) {
  constexpr std::string_view Clear = "tex1_16x16_000000000000b001_14";
  constexpr std::string_view Skip = "tex1_16x16_000000000000b002_14";
  constexpr std::string_view Manifest = R"({"version":1,"materials":{
    "tex1_16x16_000000000000b001_14":{"fakeShadow":{"neutral":"clear","gate":"light","scope":["STATIC","TERRAIN"]}},
    "tex1_16x16_000000000000b002_14":{"fakeShadow":{"neutral":"skip","gate":"cloud"}},
    "tex1_16x16_000000000000b003_14":{"flags":["stochastic"]}}})";
  ASSERT_EQ(texture::register_material_manifest(Manifest, {.layer = texture::MaterialLayer::User}).status,
            texture::MaterialManifestStatus::Accepted);
  material::latch();
  latch();
  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  const auto snapshot = material::latched_snapshot();
  EXPECT_EQ(report->table->generation, snapshot->generation);
  ASSERT_EQ(report->table->rules.size(), 2u);
  EXPECT_EQ(report->table->unscoped, 1u);
  const RuleInfo& clear = report->table->rules[0].key == Clear ? report->table->rules[0] : report->table->rules[1];
  const RuleInfo& skip = report->table->rules[0].key == Skip ? report->table->rules[0] : report->table->rules[1];
  EXPECT_EQ(clear.key, Clear);
  EXPECT_EQ(clear.rule.neutral, Neutral::Clear);
  EXPECT_EQ(clear.rule.gateBit, 1u << static_cast<uint32_t>(Gate::Light));
  EXPECT_EQ(clear.rule.scopeMask, category::Static | category::Terrain);
  EXPECT_TRUE(clear.rule.scoped);
  EXPECT_EQ(clear.rule.materialId, snapshot->find(*texture::parse_material_key(Clear)));
  EXPECT_EQ(skip.key, Skip);
  EXPECT_EQ(skip.rule.neutral, Neutral::Skip);
  EXPECT_EQ(skip.rule.gateBit, 1u << static_cast<uint32_t>(Gate::Cloud));
  EXPECT_EQ(skip.rule.scopeMask, AllBase);
  EXPECT_FALSE(skip.rule.scoped);

  set_force(Gate::Light, Force::On);
  m_config = alpha_blend();
  latch();
  main_scene();
  scope(category::Terrain);
  bind(0, clear.rule.materialId);
  EXPECT_TRUE(edited());
  EXPECT_EQ(Builder::clear, 0x1u);
  EXPECT_EQ(Builder::white, 0x0u);
}

TEST_F(RenderFakeShadow, GateEnvList) {
  using testing::parse_gate_list;
  EXPECT_EQ(parse_gate_list("light,ao").mask, 0b101u);
  EXPECT_FALSE(parse_gate_list("light,ao").unknown);
  EXPECT_EQ(parse_gate_list("all").mask, 0b111u);
  EXPECT_EQ(parse_gate_list("0").mask, 0u);
  EXPECT_EQ(parse_gate_list("").mask, 0u);
  EXPECT_EQ(parse_gate_list("light,,light").mask, 0b001u);
  EXPECT_EQ(parse_gate_list("cloud,sun").mask, 0b010u);
  EXPECT_TRUE(parse_gate_list("cloud,sun").unknown);

  detail::remove_hints();
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_FAKE_SHADOW_GATES", "light,cloud", true);
  detail::install_hints();
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_FAKE_SHADOW_GATES");
  EXPECT_EQ(force(Gate::Light), Force::On);
  EXPECT_EQ(force(Gate::Cloud), Force::On);
  EXPECT_EQ(force(Gate::Ao), Force::Auto);
  detail::remove_hints();
  EXPECT_EQ(force(Gate::Light), Force::Auto);
  EXPECT_EQ(force(Gate::Cloud), Force::Auto);
}

TEST_F(RenderFakeShadow, NeutralFillerMasksSlots) {
  const auto view = [](uintptr_t value) { return reinterpret_cast<WGPUTextureView>(value); };
  const gx::detail::TextureFillViews views{
      .empty = view(0x10),
      .white = view(0x20),
      .clear = view(0x30),
      .emptySampler = reinterpret_cast<WGPUSampler>(uintptr_t{0x40}),
  };
  for (uint32_t slot = 0; slot < 5; ++slot) {
    bind(slot, 0);
  }
  m_info.sampledTextures.reset(4);
  m_info.sampledTextures.set(5);
  m_info.sampledIndTextures.set(3);
  m_info.sampledTextures.reset(3);

  std::array<WGPUBindGroupEntry, gx::MaxTextures * 2> entries{};
  gx::detail::fill_texture_entries(m_info, 0b0101, 0b1110, false, views, entries);
  const std::array<WGPUTextureView, gx::MaxTextures> expected{
      view(0x20), view(0x30), view(0x20), view(0x30), view(0x10), view(0x10), view(0x10), view(0x10),
  };
  for (uint32_t slot = 0; slot < gx::MaxTextures; ++slot) {
    EXPECT_EQ(entries[slot * 2].binding, slot * 2);
    EXPECT_EQ(entries[slot * 2 + 1].binding, slot * 2 + 1);
    EXPECT_EQ(entries[slot * 2].textureView, expected[slot]) << slot;
    EXPECT_EQ(entries[slot * 2 + 1].sampler, views.emptySampler) << slot;
  }
}
} // namespace
} // namespace aurora::render::fakeshadow

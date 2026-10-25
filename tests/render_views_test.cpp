#include <gtest/gtest.h>

#include "gfx/resources.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/scene/classify.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace aurora::render::debug {
namespace {
namespace category = scene::category;

constexpr scene::ViewHooks Hooks{.beginFrame = &begin_view_frame, .push = &push_view};

detail::FrameState frame_with_view(uint16_t id) { return {.settings = Settings{.debugView = id}}; }

// A draw that passes every span test: after a usable MAIN camera, in the opaque window, on the EFB, writing colour
scene::DrawClass span_draw() {
  scene::DrawClass cls;
  cls.key.role = scene::Role::Main;
  cls.key.window = scene::Window::Opaque;
  cls.key.mainCamera = true;
  cls.kind = scene::Kind::Foreign;
  cls.check = scene::MainCheck::Ortho;
  return cls;
}

bool all_distinct(std::span<const Rgb8> colors) {
  std::set<std::tuple<uint8_t, uint8_t, uint8_t>> seen;
  for (const Rgb8& color : colors) {
    if (!seen.insert({color.r, color.g, color.b}).second) {
      return false;
    }
  }
  return true;
}

class RenderViews : public ::testing::Test {
protected:
  void SetUp() override {
    m_alignment = gfx::detail::resources().limits.minUniformBufferOffsetAlignment;
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
    scene::testing::reset();
    testing::reset_views();
    set_view_override(0);
    set_view_bit(category::Static);
    gx::g_gxState = {};
  }

  void TearDown() override {
    scene::set_view_hooks(nullptr);
    shutdown_classification();
    scene::testing::reset();
    gx::g_gxState = {};
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = m_alignment;
  }

  // Stage 0 reads K0 into alpha; the last stage's colour samples a texture and writes TEVREG1 unclamped
  static gx::PipelineConfig two_stage_config() {
    gx::PipelineConfig config{};
    auto& shader = config.shaderConfig;
    shader.tevStageCount = 2;
    shader.fogType = GX_FOG_PERSP_LIN;
    shader.fogRangeEnabled = 1;
    auto& first = shader.tevStages[0];
    first.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC};
    first.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST};
    first.kaSel = GX_TEV_KASEL_K0_A;
    first.channelId = GX_COLOR0A0;
    auto& last = shader.tevStages[1];
    last.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_CPREV, GX_CC_ZERO};
    last.colorOp = {
        .op = GX_TEV_SUB, .bias = GX_TB_ADDHALF, .scale = GX_CS_SCALE_2, .outReg = GX_TEVREG1, .clamp = false};
    last.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV};
    last.kcSel = GX_TEV_KCSEL_K1_R;
    last.texCoordId = GX_TEXCOORD0;
    last.texMapId = GX_TEXMAP0;
    return config;
  }

  // Three stages sample K0, K1 and K2 as colour; the last one also reads K3 into alpha through kaSel
  static gx::PipelineConfig all_konst_config(GXTevKAlphaSel lastAlpha) {
    gx::PipelineConfig config{};
    auto& shader = config.shaderConfig;
    shader.tevStageCount = 3;
    constexpr std::array<GXTevKColorSel, 3> Colors{GX_TEV_KCSEL_K0, GX_TEV_KCSEL_K1, GX_TEV_KCSEL_K2};
    for (uint32_t i = 0; i < 3; ++i) {
      auto& stage = shader.tevStages[i];
      stage.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST};
      stage.kcSel = Colors[i];
    }
    auto& last = shader.tevStages[2];
    last.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST};
    last.kaSel = lastAlpha;
    return config;
  }

  uint32_t m_alignment = 0;
};

TEST_F(RenderViews, TableIdsAndNamesUnique) {
  std::set<uint16_t> ids;
  std::set<std::string_view> names;
  std::set<std::string_view> labels;
  for (const DebugView& view : ClassViews) {
    EXPECT_TRUE(ids.insert(view.id).second) << view.id;
    EXPECT_TRUE(names.insert(view.name).second) << view.name;
    EXPECT_TRUE(labels.insert(view.label).second) << view.label;
    EXPECT_GE(view.id, 1u);
    EXPECT_LE(view.id, LastReservedClassView);
    EXPECT_EQ(find_view(view.id), &view);
  }
  EXPECT_EQ(ids, (std::set<uint16_t>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(find_view(0), nullptr);
  EXPECT_EQ(find_view(9), nullptr);
  EXPECT_EQ(find_view(100), nullptr);
}

TEST_F(RenderViews, ParseView) {
  const auto parsed = [](std::string_view text) {
    const auto selection = parse_view(text);
    return selection ? std::optional{std::pair{selection->id, selection->bit}} : std::nullopt;
  };
  using Pair = std::pair<uint16_t, uint32_t>;
  EXPECT_EQ(parsed("class.kind"), (Pair{2, 0}));
  EXPECT_EQ(parsed("kind"), (Pair{2, 0}));
  EXPECT_EQ(parsed("2"), (Pair{2, 0}));
  EXPECT_EQ(parsed("100"), (Pair{100, 0}));
  EXPECT_EQ(parsed("class.role"), (Pair{1, 0}));
  EXPECT_EQ(parsed("main"), (Pair{3, 0}));
  EXPECT_EQ(parsed("bit"), (Pair{8, category::Static}));
  EXPECT_EQ(parsed("class.bit"), (Pair{8, category::Static}));
  EXPECT_EQ(parsed("8"), (Pair{8, category::Static}));
  EXPECT_EQ(parsed("bit:decal"), (Pair{8, category::Decal}));
  EXPECT_EQ(parsed("bit:cloudReceiver"), (Pair{8, category::CloudReceiver}));
  EXPECT_EQ(parsed("65535"), (Pair{65535, 0}));
  for (const std::string_view bad : {"", "class.", "0", "70000", "65536", "kinds", "bit:nope", "bit:", "-1", "2x"}) {
    EXPECT_EQ(parsed(bad), std::nullopt) << bad;
  }
}

TEST_F(RenderViews, PalettesAreDistinct) {
  EXPECT_TRUE(all_distinct(RoleColors));
  EXPECT_TRUE(all_distinct(KindColors));
  EXPECT_TRUE(all_distinct(CheckColors));
  EXPECT_TRUE(all_distinct(WindowColors));
  std::vector<Rgb8> categories;
  for (const CategoryColor& entry : CategoryColors) {
    categories.push_back(entry.color);
    EXPECT_EQ(std::popcount(entry.bit), 1);
    EXPECT_FALSE(scene::category_name(static_cast<uint32_t>(std::countr_zero(entry.bit))).empty());
  }
  EXPECT_TRUE(all_distinct(categories));
  // Unset grey means "nothing to show" and only that: no role but None, no class, check, window or category uses it
  for (size_t i = 0; i < RoleColors.size(); ++i) {
    EXPECT_EQ(RoleColors[i] == UnsetColor, static_cast<scene::Role>(i) == scene::Role::None) << i;
  }
  EXPECT_EQ(std::ranges::count(KindColors, UnsetColor), 0);
  EXPECT_EQ(std::ranges::count(CheckColors, UnsetColor), 0);
  EXPECT_EQ(std::ranges::count(WindowColors, UnsetColor), 0);
  EXPECT_EQ(std::ranges::count(categories, UnsetColor), 0);
  EXPECT_NE(BitOnColor, UnsetColor);
}

TEST_F(RenderViews, HashColorStableAndBright) {
  for (uint64_t key = 0; key < 10'000; ++key) {
    const Rgb8 color = hash_color(key * 0x9E3779B97F4A7C15ull);
    EXPECT_EQ(color, hash_color(key * 0x9E3779B97F4A7C15ull));
    ASSERT_GE(color.r, 64);
    ASSERT_GE(color.g, 64);
    ASSERT_GE(color.b, 64);
  }
  EXPECT_NE(hash_color(0x10), hash_color(0x11));
  EXPECT_EQ(lighten({0, 128, 255}), (Rgb8{127, 191, 255}));
}

TEST_F(RenderViews, CategoryPriority) {
  const auto color = [](uint32_t categories) {
    scene::DrawClass cls = span_draw();
    cls.key.categories = categories;
    return view_color(ClassView::Category, 0, cls);
  };
  const auto entry = [](uint32_t bit) { return std::ranges::find(CategoryColors, bit, &CategoryColor::bit)->color; };
  EXPECT_EQ(color(category::Terrain | category::Static), entry(category::Terrain));
  EXPECT_EQ(color(category::FakeShadow | category::Decal), entry(category::FakeShadow));
  EXPECT_EQ(color(category::Foliage | category::Emissive), entry(category::Foliage));
  EXPECT_EQ(color(category::CloudReceiver), UnsetColor);
  EXPECT_EQ(color(0), UnsetColor);
}

TEST_F(RenderViews, ViewColorFollowsClass) {
  scene::DrawClass cls = span_draw();
  cls.key.role = scene::Role::Reflection;
  cls.kind = scene::Kind::MainBlended;
  cls.check = scene::MainCheck::Projection;
  cls.key.window = scene::Window::Translucent;
  cls.key.categories = category::Decal | category::Static;
  EXPECT_EQ(view_color(ClassView::Role, 0, cls), RoleColors[static_cast<size_t>(scene::Role::Reflection)]);
  EXPECT_EQ(view_color(ClassView::Kind, 0, cls), KindColors[static_cast<size_t>(scene::Kind::MainBlended)]);
  EXPECT_EQ(view_color(ClassView::MainCheck, 0, cls), CheckColors[static_cast<size_t>(scene::MainCheck::Projection)]);
  EXPECT_EQ(view_color(ClassView::Window, 0, cls), WindowColors[static_cast<size_t>(scene::Window::Translucent)]);

  EXPECT_EQ(view_color(ClassView::Object, 0, cls), UnsetColor);
  EXPECT_EQ(view_color(ClassView::Scope, 0, cls), UnsetColor);
  cls.key.scope = 0x1234;
  EXPECT_EQ(view_color(ClassView::Object, 0, cls), lighten(hash_color(0x1234)));
  EXPECT_EQ(view_color(ClassView::Scope, 0, cls), hash_color(0x1234));
  cls.key.object = {.id = 0x10, .subId = 2, .generation = 1, .valid = true};
  EXPECT_EQ(view_color(ClassView::Object, 0, cls), hash_color(0x10 ^ (uint64_t{2} << 48)));

  EXPECT_EQ(view_color(ClassView::Bit, category::Static, cls), BitOnColor);
  EXPECT_EQ(view_color(ClassView::Bit, category::Terrain, cls), UnsetColor);
  EXPECT_EQ(view_color(ClassView::Off, 0, cls), UnsetColor);
}

TEST_F(RenderViews, SpanRule) {
  EXPECT_TRUE(in_view_span(span_draw()));
  scene::DrawClass preScene = span_draw();
  preScene.key.window = scene::Window::PreScene;
  EXPECT_TRUE(in_view_span(preScene));

  scene::DrawClass beforeCamera = span_draw();
  beforeCamera.key.mainCamera = false;
  EXPECT_FALSE(in_view_span(beforeCamera));
  scene::DrawClass post = span_draw();
  post.key.window = scene::Window::Post;
  EXPECT_FALSE(in_view_span(post));
  scene::DrawClass hud = span_draw();
  hud.key.window = scene::Window::Hud;
  EXPECT_FALSE(in_view_span(hud));
  scene::DrawClass offscreen = span_draw();
  offscreen.offscreen = true;
  EXPECT_FALSE(in_view_span(offscreen));
  scene::DrawClass colorless = span_draw();
  colorless.colorWrite = false;
  EXPECT_FALSE(in_view_span(colorless));
}

TEST_F(RenderViews, VariantRewritesOnlyLastColour) {
  const gx::PipelineConfig config = two_stage_config();
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  ASSERT_EQ(info.sampledKColors.to_ulong(), 0b0001u);
  gx::PipelineConfig variant{};
  ASSERT_TRUE(make_view_variant(config, info, variant));

  const gx::TevStage& before = config.shaderConfig.tevStages[1];
  const gx::TevStage& after = variant.shaderConfig.tevStages[1];
  EXPECT_EQ(after.colorPass.a, GX_CC_ZERO);
  EXPECT_EQ(after.colorPass.b, GX_CC_ZERO);
  EXPECT_EQ(after.colorPass.c, GX_CC_ZERO);
  EXPECT_EQ(after.colorPass.d, GX_CC_KONST);
  EXPECT_EQ(after.colorOp.op, GX_TEV_ADD);
  EXPECT_EQ(after.colorOp.bias, GX_TB_ZERO);
  EXPECT_EQ(after.colorOp.scale, GX_CS_SCALE_1);
  EXPECT_TRUE(after.colorOp.clamp);
  EXPECT_EQ(after.colorOp.outReg, GX_TEVREG1);
  EXPECT_EQ(after.kcSel, GX_TEV_KCSEL_K3);
  EXPECT_EQ(after.alphaPass, before.alphaPass);
  EXPECT_EQ(after.alphaOp, before.alphaOp);
  EXPECT_EQ(after.kaSel, before.kaSel);
  EXPECT_EQ(variant.shaderConfig.fogType, GX_FOG_NONE);
  EXPECT_EQ(static_cast<int>(variant.shaderConfig.fogRangeEnabled), 0);

  gx::PipelineConfig restored = variant;
  gx::TevStage& stage = restored.shaderConfig.tevStages[1];
  stage.colorPass = before.colorPass;
  stage.colorOp = before.colorOp;
  stage.kcSel = before.kcSel;
  restored.shaderConfig.fogType = config.shaderConfig.fogType;
  restored.shaderConfig.fogRangeEnabled = config.shaderConfig.fogRangeEnabled;
  EXPECT_EQ(std::memcmp(&restored, &config, sizeof(config)), 0);
}

TEST_F(RenderViews, VariantDropsTextureExtBits) {
  gx::PipelineConfig config = two_stage_config();
  config.shaderConfig.ext = {.output = 2, .pom = 1, .stochasticMask = 1, .layer0 = 1, .layer1 = 2};
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  gx::PipelineConfig variant{};
  ASSERT_TRUE(make_view_variant(config, info, variant));
  EXPECT_EQ(variant.shaderConfig.ext.stochasticMask, 0u);
  EXPECT_EQ(variant.shaderConfig.ext.layer0, 0u);
  EXPECT_EQ(variant.shaderConfig.ext.layer1, 0u);
  EXPECT_EQ(variant.shaderConfig.ext.pom, 0u);
  EXPECT_EQ(variant.shaderConfig.ext.output, 2u);
}

TEST_F(RenderViews, VariantSkipsPackedAlphaCompare) {
  gx::PipelineConfig config = two_stage_config();
  config.shaderConfig.tevStages[1].alphaOp.op = GX_TEV_COMP_R8_GT;
  gx::PipelineConfig variant{};
  EXPECT_FALSE(make_view_variant(config, gx::build_shader_info(config.shaderConfig), variant));
  config.shaderConfig.tevStages[1].alphaOp.op = GX_TEV_COMP_BGR24_EQ;
  EXPECT_FALSE(make_view_variant(config, gx::build_shader_info(config.shaderConfig), variant));
  // A8 compares read alpha only, so they keep their alpha under the variant
  config.shaderConfig.tevStages[1].alphaOp.op = GX_TEV_COMP_A8_GT;
  EXPECT_TRUE(make_view_variant(config, gx::build_shader_info(config.shaderConfig), variant));
  gx::PipelineConfig empty{};
  EXPECT_FALSE(make_view_variant(empty, gx::build_shader_info(empty.shaderConfig), variant));
}

TEST_F(RenderViews, VariantKonstFallbackKeepsAlpha) {
  gx::PipelineConfig variant{};
  const gx::PipelineConfig readsK3Colour = all_konst_config(GX_TEV_KASEL_K3_R);
  const gx::ShaderInfo readsInfo = gx::build_shader_info(readsK3Colour.shaderConfig);
  ASSERT_TRUE(readsInfo.sampledKColors.all());
  EXPECT_FALSE(make_view_variant(readsK3Colour, readsInfo, variant));

  const gx::PipelineConfig readsK3Alpha = all_konst_config(GX_TEV_KASEL_K3_A);
  const gx::ShaderInfo alphaInfo = gx::build_shader_info(readsK3Alpha.shaderConfig);
  ASSERT_TRUE(alphaInfo.sampledKColors.all());
  ASSERT_TRUE(make_view_variant(readsK3Alpha, alphaInfo, variant));
  EXPECT_EQ(variant.shaderConfig.tevStages[2].kcSel, GX_TEV_KCSEL_K3);

  // A packed compare in an earlier stage reads K3's colour through its colour arguments
  gx::PipelineConfig packed = readsK3Alpha;
  auto& compare = packed.shaderConfig.tevStages[1];
  compare.alphaOp.op = GX_TEV_COMP_GR16_EQ;
  compare.colorPass = {GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO};
  compare.kcSel = GX_TEV_KCSEL_K3_G;
  compare.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST};
  compare.kaSel = GX_TEV_KASEL_K1_A;
  const gx::ShaderInfo packedInfo = gx::build_shader_info(packed.shaderConfig);
  ASSERT_TRUE(packedInfo.sampledKColors.all());
  EXPECT_FALSE(make_view_variant(packed, packedInfo, variant));
  // Without the fallback the variant takes a free register, so K3's colour stays the draw's own
  compare.kaSel = GX_TEV_KASEL_1;
  packed.shaderConfig.tevStages[2].alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO};
  const gx::ShaderInfo freeInfo = gx::build_shader_info(packed.shaderConfig);
  ASSERT_FALSE(freeInfo.sampledKColors.test(1));
  ASSERT_TRUE(make_view_variant(packed, freeInfo, variant));
  EXPECT_EQ(variant.shaderConfig.tevStages[2].kcSel, GX_TEV_KCSEL_K1);

  // Stage 0 writes K3's colour to C0 and stage 1 compares C0 into alpha, so K3 reaches alpha without a KONST arg
  gx::PipelineConfig indirect{};
  auto& shader = indirect.shaderConfig;
  shader.tevStageCount = 3;
  auto& writer = shader.tevStages[0];
  writer.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST};
  writer.kcSel = GX_TEV_KCSEL_K3;
  writer.colorOp.outReg = GX_TEVREG0;
  auto& reader = shader.tevStages[1];
  reader.colorPass = {GX_CC_C0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST};
  reader.kcSel = GX_TEV_KCSEL_K0;
  reader.alphaOp.op = GX_TEV_COMP_R8_GT;
  auto& last = shader.tevStages[2];
  last.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST};
  last.kcSel = GX_TEV_KCSEL_K1;
  last.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST};
  last.kaSel = GX_TEV_KASEL_K2_A;
  const gx::ShaderInfo indirectInfo = gx::build_shader_info(shader);
  ASSERT_TRUE(indirectInfo.sampledKColors.all());
  EXPECT_FALSE(make_view_variant(indirect, indirectInfo, variant));
}

TEST_F(RenderViews, VariantShaderInfoSamplesKonst) {
  const gx::PipelineConfig config = two_stage_config();
  gx::PipelineConfig variant{};
  ASSERT_TRUE(make_view_variant(config, gx::build_shader_info(config.shaderConfig), variant));
  const gx::ShaderInfo info = gx::build_shader_info(variant.shaderConfig);
  EXPECT_TRUE(info.sampledKColors.test(3));
  EXPECT_TRUE(info.sampledKColors.test(0));
  EXPECT_FALSE(info.usesFog);
}

TEST_F(RenderViews, ViewOffLeavesDrawUntouched) {
  begin_view_frame(frame_with_view(0));
  EXPECT_EQ(testing::active_view(), ClassView::Off);
  EXPECT_FALSE(scene::identity_collecting());
  const gx::PipelineConfig config = two_stage_config();
  gx::DrawData draw{};
  draw.pipeline = 77;
  draw.uniformRange = {512, 256};
  const gx::DrawData original = draw;
  EXPECT_FALSE(push_view(draw, config, gx::build_shader_info(config.shaderConfig), span_draw()));
  EXPECT_EQ(std::memcmp(&draw, &original, sizeof(draw)), 0);
  const scene::ViewCounts& counts = scene::view_counts();
  EXPECT_EQ(counts.substituted + counts.untouched + counts.pending + counts.overBudget, 0u);
}

TEST_F(RenderViews, UnknownIdSelectsNothing) {
  begin_view_frame(frame_with_view(100));
  EXPECT_EQ(testing::active_view(), ClassView::Off);
  EXPECT_FALSE(scene::identity_collecting());
  begin_view_frame(frame_with_view(LastReservedClassView));
  EXPECT_EQ(testing::active_view(), ClassView::Off);
}

TEST_F(RenderViews, OverrideWinsOverSettings) {
  set_view_override(6);
  begin_view_frame(frame_with_view(2));
  EXPECT_EQ(testing::active_view(), ClassView::Object);
  EXPECT_TRUE(scene::identity_collecting());
  set_view_override(0);
  begin_view_frame(frame_with_view(2));
  EXPECT_EQ(testing::active_view(), ClassView::Kind);
  // A view left on by a test or a shutdown must not keep identity collection running
  testing::reset_views();
  EXPECT_FALSE(scene::identity_collecting());
  begin_view_frame(frame_with_view(2));
  ASSERT_TRUE(scene::identity_collecting());
  shutdown_classification();
  EXPECT_FALSE(scene::identity_collecting());
}

TEST_F(RenderViews, ColourlessDrawUntouched) {
  begin_view_frame(frame_with_view(1));
  const gx::PipelineConfig config = two_stage_config();
  scene::DrawClass colorless = span_draw();
  colorless.colorWrite = false;
  gx::DrawData draw{};
  draw.pipeline = 77;
  EXPECT_FALSE(push_view(draw, config, gx::build_shader_info(config.shaderConfig), colorless));
  EXPECT_EQ(draw.pipeline, 77u);
  EXPECT_EQ(scene::view_counts().untouched, 1u);
}

TEST_F(RenderViews, ApplyFrameStateRunsTheViewFirst) {
  scene::set_view_hooks(&Hooks);
  scene::testing::set_identity_env(2);
  scene::Classifier& classifier = scene::testing::classifier();
  const auto subId_changes_key = [&](uint16_t view) {
    scene::apply_frame_state(frame_with_view(view));
    classifier.begin_frame();
    classifier.object_begin({.id = 7, .subId = 0, .generation = 1});
    classifier.route({});
    classifier.object_end();
    classifier.object_begin({.id = 7, .subId = 1, .generation = 1});
    return classifier.route({}).keyChanged;
  };
  // The object-only key ignores sub-ids, but a view colours per draw and needs the full key
  EXPECT_FALSE(subId_changes_key(0));
  EXPECT_TRUE(subId_changes_key(6));
  EXPECT_TRUE(scene::identity_collecting());
  EXPECT_FALSE(subId_changes_key(0));
  scene::testing::set_identity_env(0);
  scene::apply_frame_state(frame_with_view(0));
  EXPECT_FALSE(scene::identity_collecting());
}
} // namespace
} // namespace aurora::render::debug

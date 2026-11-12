#include <gtest/gtest.h>

#include "gfx/resources.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "render/debug/decomp_view.hpp"
#include "render/debug/retained_view.hpp"
#include "render/debug/views.hpp"
#include "render/debug/vsm_view.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/scene/classify.hpp"
#include "render/scene/identity.hpp"

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

TEST_F(RenderViews, GbufViewTableFollowsClassRange) {
  EXPECT_EQ(FirstGbufView, LastReservedClassView + 1);
  std::set<std::string_view> names;
  for (const DebugView& view : ClassViews) {
    names.insert(view.name);
  }
  uint16_t expected = FirstGbufView;
  std::set<uint32_t> targetModes;
  for (const GbufView& view : GbufViews) {
    expected += expected == 35 ? 1 : 0;
    EXPECT_EQ(view.id, expected++);
    EXPECT_TRUE(names.insert(view.name).second) << view.name;
    EXPECT_EQ(find_gbuf_view(view.id), &view);
    EXPECT_EQ(find_view(view.id), nullptr);
    if (view.kind == GbufViewKind::Decomposition) {
      EXPECT_FALSE(view.uintSource) << view.name;
      EXPECT_TRUE(view.name.starts_with("composite.")) << view.name;
      continue;
    }
    if (view.kind == GbufViewKind::Retained) {
      EXPECT_FALSE(view.uintSource) << view.name;
      EXPECT_TRUE(view.name.starts_with("retained.")) << view.name;
      continue;
    }
    if (view.resource.starts_with("light.") || view.resource.starts_with("sky.")) {
      EXPECT_EQ(view.mode, 5u) << view.name;
      continue;
    }
    if (shadow_view(view)) {
      EXPECT_EQ(view.mode, view.resource == VsmViewResource ? 18u : 7u) << view.name;
      continue;
    }
    EXPECT_TRUE(targetModes.insert(view.mode).second) << view.name;
    EXPECT_EQ(view.uintSource, view.resource == "gbuf.fog" || view.resource == resource::SurfaceV1) << view.name;
    const bool gx = view.resource.starts_with("gbuf.") || view.resource.starts_with("gx.");
    EXPECT_EQ(gx, view.id <= 22) << view.name;
    EXPECT_EQ(gx, detail::find_gx_resource(view.resource).has_value()) << view.name;
    if (gx && view.id <= 22) {
      EXPECT_EQ(view.mode, static_cast<uint32_t>(view.id - FirstGbufView));
    }
  }
  EXPECT_EQ(expected, 53u);
  EXPECT_EQ(targetModes, (std::set<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}));
  EXPECT_EQ(find_gbuf_view(LastReservedClassView), nullptr);
  EXPECT_EQ(find_gbuf_view(35), nullptr);
  EXPECT_EQ(find_gbuf_view(16)->stage, Stage::PrePost);
  EXPECT_EQ(find_gbuf_view(18)->anchor, "phase.composite");
  EXPECT_EQ(find_gbuf_view(22)->label, "G-buffer: receiver code");
}

TEST_F(RenderViews, ParseGbufViewNames) {
  const auto id = [](std::string_view text) {
    const auto selection = parse_view(text);
    return selection ? std::optional{selection->id} : std::nullopt;
  };
  for (const GbufView& view : GbufViews) {
    EXPECT_EQ(id(view.name), view.id) << view.name;
    if (view.kind != GbufViewKind::Retained) {
      EXPECT_EQ(id(view.name.substr(view.name.find('.') + 1)), view.id) << view.name;
    }
  }
  EXPECT_EQ(id("motion"), 16u);
  EXPECT_EQ(id("fog"), 18u);
  EXPECT_EQ(id("fog.color"), 19u);
  EXPECT_EQ(id("gx.normal.code"), 22u);
  EXPECT_EQ(id("normal"), 21u);
  EXPECT_EQ(id("17"), 17u);
  EXPECT_EQ(parse_view("gbuf.motion")->bit, 0u);
  for (const std::string_view bad : {"gbuf.", "gbuf.fog.", "fog.colour", "gx.motion"}) {
    EXPECT_EQ(id(bad), std::nullopt) << bad;
  }
}

TEST_F(RenderViews, DecompositionRowsAppendFrom23) {
  struct Row {
    uint16_t id;
    std::string_view name;
    std::string_view resource;
    std::string_view anchor;
    GbufViewKind kind;
    uint32_t mode;
  };
  constexpr std::array<Row, 7> Rows{{
      {23, "ao.final", "ao.final", "phase.composite", GbufViewKind::Target, 7},
      {24, "composite.defog", "gx.color", "phase.volumetrics", GbufViewKind::Decomposition, 0},
      {25, "composite.identity", "legacyTex", "phase.composite", GbufViewKind::Decomposition, 1},
      {26, "composite.coverage", "gbuf.fog", "phase.volumetrics", GbufViewKind::Decomposition, 2},
      {27, "composite.multiply", "gx.color", "phase.volumetrics", GbufViewKind::Decomposition, 3},
      {28, "ao.raw", "ao.raw", "phase.ao", GbufViewKind::Target, 8},
      {29, "ao.hiz", "ao.hiz", "phase.ao", GbufViewKind::Target, 9},
  }};
  for (const Row& row : Rows) {
    const GbufView* view = find_gbuf_view(row.id);
    ASSERT_NE(view, nullptr) << row.id;
    EXPECT_EQ(view->name, row.name);
    EXPECT_EQ(view->resource, row.resource) << row.name;
    EXPECT_EQ(view->anchor, row.anchor) << row.name;
    EXPECT_EQ(view->stage, Stage::OpaqueDone) << row.name;
    EXPECT_EQ(view->kind, row.kind) << row.name;
    EXPECT_EQ(view->mode, row.mode) << row.name;
    EXPECT_FALSE(view->uintSource) << row.name;
    EXPECT_EQ(find_view(row.id), nullptr) << row.name;
  }
  EXPECT_EQ(decomp_mode(*find_gbuf_view(25)), DecompMode::Identity);
  EXPECT_EQ(decomp_mode(*find_gbuf_view(27)), DecompMode::Multiply);
  for (uint16_t id = FirstGbufView; id <= 22; ++id) {
    EXPECT_EQ(find_gbuf_view(id)->kind, GbufViewKind::Target) << id;
  }
}

TEST_F(RenderViews, SurfaceRowsAppendFrom30) {
  struct Row {
    uint16_t id;
    std::string_view name;
    std::string_view label;
  };
  constexpr std::array<Row, 5> Rows{{
      {30, "gbuf.surface.albedo", "Surface: base colour"},
      {31, "gbuf.surface.material", "Surface: roughness, metalness, occlusion"},
      {32, "gbuf.surface.class", "Surface: class"},
      {33, "gbuf.surface.ambient", "Surface: ambient"},
      {34, "gbuf.surface.emissive", "Surface: emissive"},
  }};
  for (const Row& row : Rows) {
    const GbufView* view = find_gbuf_view(row.id);
    ASSERT_NE(view, nullptr) << row.id;
    EXPECT_EQ(view->name, row.name);
    EXPECT_EQ(view->label, row.label) << row.name;
    EXPECT_EQ(view->resource, resource::SurfaceV1) << row.name;
    EXPECT_EQ(view->anchor, "phase.composite") << row.name;
    EXPECT_EQ(view->stage, Stage::OpaqueDone) << row.name;
    EXPECT_EQ(view->kind, GbufViewKind::Target) << row.name;
    EXPECT_EQ(view->mode, static_cast<uint32_t>(row.id - 20)) << row.name;
    EXPECT_TRUE(view->uintSource) << row.name;
    EXPECT_EQ(parse_view(row.name)->id, row.id) << row.name;
  }
  EXPECT_EQ(parse_view("surface.class")->id, 32u);
}

TEST_F(RenderViews, LightRowsNeedPbr) {
  const GbufView* view = find_gbuf_view(36);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->name, "light.heatmap");
  EXPECT_EQ(view->label, "Lights: lights per cluster");
  EXPECT_EQ(view->resource, "light.heatmap");
  EXPECT_EQ(view->anchor, "phase.composite");
  EXPECT_EQ(view->stage, Stage::OpaqueDone);
  EXPECT_EQ(view->mode, 5u);
  EXPECT_FALSE(view->uintSource);
  EXPECT_EQ(view->kind, GbufViewKind::Target);
  EXPECT_EQ(parse_view("light.heatmap")->id, 36u);
  EXPECT_EQ(parse_view("heatmap")->id, 36u);
  EXPECT_EQ(find_view(36), nullptr);

  EXPECT_EQ(view_blocked(*view, nullptr), "needs PBR Lighting");
  auto graph = std::make_shared<detail::CompiledGraph>();
  detail::FrameState frame{.settings = Settings{.ao = Quality::High}, .graph = graph};
  graph->sceneDepthSource = DepthSource::Direct;
  EXPECT_EQ(view_blocked(*view, &frame), "needs PBR Lighting");
  frame.settings.pbr = true;
  EXPECT_EQ(view_blocked(*view, &frame), std::nullopt);
  for (const DepthSource source : {DepthSource::Copy, DepthSource::Unavailable}) {
    graph->sceneDepthSource = source;
    EXPECT_EQ(view_blocked(*view, &frame), "needs readable scene depth");
  }
  const GbufView aoRow = *find_gbuf_view(23);
  frame.settings.pbr = false;
  EXPECT_NE(view_blocked(aoRow, &frame), "needs PBR Lighting");
  GbufView sky = *view;
  sky.resource = "sky.probe";
  EXPECT_EQ(view_blocked(sky, &frame), "needs PBR Lighting");
}

TEST_F(RenderViews, SkyProbeRowsNeedPbr) {
  const GbufView* probe = find_gbuf_view(37);
  const GbufView* mask = find_gbuf_view(38);
  ASSERT_NE(probe, nullptr);
  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(probe->name, "sky.probe");
  EXPECT_EQ(probe->label, "Sky probe: radiance, filtered, irradiance");
  EXPECT_EQ(mask->name, "sky.probe.mask");
  EXPECT_EQ(mask->label, "Sky probe: sky-test mask");
  for (const GbufView* view : {probe, mask}) {
    EXPECT_EQ(view->resource, "sky.probe.view");
    EXPECT_EQ(view->anchor, "phase.composite");
    EXPECT_EQ(view->stage, Stage::OpaqueDone);
    EXPECT_EQ(view->mode, 5u);
    EXPECT_FALSE(view->uintSource);
    EXPECT_EQ(view->kind, GbufViewKind::Target);
    EXPECT_EQ(find_view(view->id), nullptr);
    EXPECT_EQ(view_blocked(*view, nullptr), "needs PBR Lighting");
  }
  EXPECT_EQ(parse_view("sky.probe")->id, 37u);
  EXPECT_EQ(parse_view("probe")->id, 37u);
  EXPECT_EQ(parse_view("sky.probe.mask")->id, 38u);
  EXPECT_EQ(parse_view("probe.mask")->id, 38u);
  auto graph = std::make_shared<detail::CompiledGraph>();
  detail::FrameState frame{.settings = Settings{.pbr = true}, .graph = graph};
  graph->sceneDepthSource = DepthSource::Direct;
  EXPECT_EQ(view_blocked(*mask, &frame), std::nullopt);
  graph->sceneDepthSource = DepthSource::Copy;
  EXPECT_EQ(view_blocked(*mask, &frame), "needs readable scene depth");
}

TEST_F(RenderViews, SsilRowsAppendFrom39) {
  struct Row {
    uint16_t id;
    std::string_view name;
    std::string_view resource;
    std::string_view anchor;
    uint32_t mode;
  };
  constexpr std::array<Row, 4> Rows{{
      {39, "ssil.irr", resource::SsilIrradiance, "phase.composite", 15},
      {40, "ssil.irr.coverage", resource::SsilIrradiance, "phase.composite", 16},
      {41, "ssil.radiance", "ssil.radiance", "phase.ao", 17},
      {42, "ssil.radiance.normal", "ssil.normal", "phase.ao", 18},
  }};
  for (const Row& row : Rows) {
    const GbufView* view = find_gbuf_view(row.id);
    ASSERT_NE(view, nullptr) << row.id;
    EXPECT_EQ(view->name, row.name);
    EXPECT_EQ(view->resource, row.resource);
    EXPECT_EQ(view->anchor, row.anchor);
    EXPECT_EQ(view->mode, row.mode);
    EXPECT_EQ(view->kind, GbufViewKind::Target);
    EXPECT_FALSE(view->uintSource);
    EXPECT_TRUE(view->label.starts_with("Indirect light: ")) << view->label;
    EXPECT_EQ(parse_view(row.name)->id, row.id);
    EXPECT_EQ(view_blocked(*view, nullptr), std::optional<std::string>{"needs Indirect Light"});
  }
  EXPECT_EQ(find_gbuf_view(43)->kind, GbufViewKind::Retained);
}

TEST_F(RenderViews, ShortNamesAreUnique) {
  std::set<std::string_view> names;
  for (const DebugView& view : ClassViews) {
    EXPECT_TRUE(names.insert(view.name).second) << view.name;
    EXPECT_TRUE(names.insert(view.name.substr(std::string_view{"class."}.size())).second) << view.name;
  }
  size_t retained = 0;
  for (const GbufView& view : GbufViews) {
    EXPECT_TRUE(names.insert(view.name).second) << view.name;
    if (view.kind == GbufViewKind::Retained) {
      ++retained;
      continue;
    }
    EXPECT_TRUE(names.insert(view.name.substr(view.name.find('.') + 1)).second) << view.name;
  }
  EXPECT_EQ(names.size(), 2 * (ClassViews.size() + GbufViews.size()) - retained);
}

TEST_F(RenderViews, ParseDecompositionViewNames) {
  const auto id = [](std::string_view text) {
    const auto selection = parse_view(text);
    return selection ? std::optional{selection->id} : std::nullopt;
  };
  using Pair = std::pair<std::string_view, uint16_t>;
  for (const auto& [text, expected] :
       {Pair{"ao.final", 23}, Pair{"final", 23}, Pair{"composite.defog", 24}, Pair{"defog", 24},
        Pair{"composite.identity", 25}, Pair{"identity", 25}, Pair{"composite.coverage", 26}, Pair{"coverage", 26},
        Pair{"composite.multiply", 27}, Pair{"multiply", 27}, Pair{"ao.raw", 28}, Pair{"raw", 28}, Pair{"ao.hiz", 29},
        Pair{"hiz", 29}, Pair{"29", 29}}) {
    EXPECT_EQ(id(text), expected) << text;
  }
  EXPECT_EQ(parse_view("composite.identity")->bit, 0u);
  for (const std::string_view bad : {"composite.", "ao.", "composite.ident", "ao.final.r", "composite"}) {
    EXPECT_EQ(id(bad), std::nullopt) << bad;
  }
}

TEST_F(RenderViews, DecompositionColorsAreFrozen) {
  EXPECT_EQ(identity_class_color(IdentityClass::Unchanged), (Rgb8{0, 0, 0}));
  EXPECT_EQ(identity_class_color(IdentityClass::Lifted), (Rgb8{255, 128, 0}));
  EXPECT_EQ(identity_class_color(IdentityClass::Brightened), (Rgb8{0, 255, 255}));
  EXPECT_EQ(identity_class_color(IdentityClass::CodeZero), (Rgb8{255, 0, 0}));
  EXPECT_EQ(identity_class_color(IdentityClass::AlphaChanged), (Rgb8{255, 255, 0}));
  EXPECT_EQ(identity_grey(0), 63);
  EXPECT_EQ(identity_grey(1), 71);
  EXPECT_EQ(identity_grey(24), 255);
  EXPECT_EQ(identity_grey(1000), 255);
  EXPECT_EQ(coverage_touched(coverage_class_color(CoverageClass::Receiver)), (Rgb8{168, 208, 255}));
  EXPECT_EQ(coverage_touched(coverage_class_color(CoverageClass::FoggedReceiver)), (Rgb8{158, 228, 158}));
  EXPECT_EQ(coverage_touched(coverage_class_color(CoverageClass::LockedReceiver)), (Rgb8{255, 208, 128}));
  EXPECT_EQ(coverage_class_color(CoverageClass::Untouched), (Rgb8{64, 64, 64}));
  EXPECT_EQ(coverage_class_color(CoverageClass::FoggedNonReceiver), (Rgb8{40, 40, 110}));
  EXPECT_EQ(coverage_class_color(CoverageClass::Receiver), (Rgb8{80, 160, 255}));
  EXPECT_EQ(coverage_class_color(CoverageClass::FoggedReceiver), (Rgb8{60, 200, 60}));
  EXPECT_EQ(coverage_class_color(CoverageClass::LockedReceiver), (Rgb8{255, 160, 0}));
  EXPECT_EQ(coverage_class_color(CoverageClass::Bad), (Rgb8{255, 0, 255}));
  EXPECT_EQ(DefogBadColor, (Rgb8{255, 0, 255}));
  EXPECT_TRUE(all_distinct(IdentityColors));
  EXPECT_TRUE(all_distinct(CoverageColors));
}

TEST_F(RenderViews, DecompositionMirrorsClassify) {
  using decomp_view::Texel;
  constexpr Texel Legacy{100, 100, 100, 7};
  constexpr Texel NoFog{};
  constexpr Texel Fogged{101, 50, 50, 127 << 1};
  const auto grey = [](uint8_t v) { return Rgb8{v, v, v}; };
  EXPECT_EQ(decomp_view::identity_color(Legacy, Legacy, NoFog, 3), (Rgb8{0, 0, 0}));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {99, 100, 100, 7}, NoFog, 3), grey(71));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {90, 100, 97, 7}, NoFog, 1), grey(143));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {101, 100, 100, 7}, Fogged, 2), (Rgb8{255, 128, 0}));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {101, 100, 100, 7}, NoFog, 2), (Rgb8{0, 255, 255}));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {101, 101, 100, 7}, Fogged, 2), (Rgb8{0, 255, 255}));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {99, 100, 100, 7}, NoFog, 0), (Rgb8{255, 0, 0}));
  EXPECT_EQ(decomp_view::identity_color(Legacy, {100, 100, 100, 8}, NoFog, 0), (Rgb8{255, 255, 0}));

  constexpr Texel Bad{110, 0, 0, 127 << 1};
  EXPECT_EQ(decomp_view::coverage_color(Bad, 3, Legacy, 0), (Rgb8{255, 0, 255}));
  EXPECT_EQ(decomp_view::defog_color(Legacy, Bad), (Rgb8{255, 0, 255}));
  EXPECT_EQ(decomp_view::defog_color(Legacy, NoFog), (Rgb8{100, 100, 100}));
  EXPECT_EQ(decomp_view::coverage_color(NoFog, 0, Legacy, 0), (Rgb8{64, 64, 64}));
  EXPECT_EQ(decomp_view::coverage_color({10, 10, 10, 2}, 0, Legacy, 9), (Rgb8{40, 40, 110}));
  EXPECT_EQ(decomp_view::coverage_color(NoFog, 1, Legacy, 0), (Rgb8{80, 160, 255}));
  EXPECT_EQ(decomp_view::coverage_color({0, 0, 0, 1}, 1, Legacy, 0), (Rgb8{80, 160, 255}));
  EXPECT_EQ(decomp_view::coverage_color({10, 10, 10, 2}, 2, Legacy, 0), (Rgb8{60, 200, 60}));
  EXPECT_EQ(decomp_view::coverage_color({10, 10, 10, 2}, 2, Legacy, 1), (Rgb8{158, 228, 158}));
  EXPECT_EQ(decomp_view::coverage_color({10, 10, 10, 3}, 3, Legacy, 0), (Rgb8{255, 160, 0}));
  EXPECT_EQ(decomp_view::multiply_color(Legacy, 0, 51), (Rgb8{100, 100, 100}));
  EXPECT_EQ(decomp_view::multiply_color({200, 100, 0, 0}, 1, 51), (Rgb8{40, 20, 0}));
}

TEST_F(RenderViews, GbufIdSelectsNoClassView) {
  for (const GbufView& view : GbufViews) {
    begin_view_frame(frame_with_view(view.id));
    EXPECT_EQ(testing::active_view(), ClassView::Off) << view.name;
    EXPECT_FALSE(scene::identity_collecting()) << view.name;
  }
  set_view_override(FirstGbufView);
  EXPECT_EQ(selected_view_id(Settings{.debugView = 2}), FirstGbufView);
  set_view_override(0);
  EXPECT_EQ(selected_view_id(Settings{.debugView = 2}), 2u);
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

TEST_F(RenderViews, RetainedRowsHaveFreshIds) {
  struct Row {
    uint16_t id;
    std::string_view name;
    RetainedMode mode;
  };
  constexpr std::array<Row, 5> Rows{{
      {43, "retained.wireframe", RetainedMode::Wireframe},
      {44, "retained.clusters", RetainedMode::Clusters},
      {45, "retained.coverage", RetainedMode::Coverage},
      {46, "retained.atlas", RetainedMode::AtlasPage},
      {47, "retained.albedo", RetainedMode::Albedo},
  }};
  for (const Row& row : Rows) {
    const GbufView* view = find_gbuf_view(row.id);
    ASSERT_NE(view, nullptr) << row.id;
    EXPECT_EQ(view->name, row.name);
    EXPECT_EQ(view->kind, GbufViewKind::Retained);
    EXPECT_EQ(retained_mode(*view), row.mode);
    EXPECT_EQ(view->stage, Stage::OpaqueDone);
    EXPECT_TRUE(view->label.starts_with("Retained: ")) << view->label;
    EXPECT_EQ(find_view(row.id), nullptr);
  }
  EXPECT_LE(GbufViews.back().id, MaxDebugViewId);
  EXPECT_EQ(retained_view::mode_entries(RetainedMode::AtlasPage).size(), 1u);
  EXPECT_EQ(retained_view::mode_entries(RetainedMode::Albedo).size(), 1u);
  EXPECT_LE(retained_view::mode_entries(RetainedMode::Wireframe).size(), detail::MaxPassPipelines);
}

TEST_F(RenderViews, VisSunRowHasFreshId) {
  const GbufView* view = find_gbuf_view(48);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->name, "vis.sun");
  EXPECT_EQ(view->label, "Sun shadows: visibility");
  EXPECT_EQ(view->resource, resource::SunVisibility);
  EXPECT_EQ(view->anchor, "phase.composite");
  EXPECT_EQ(view->stage, Stage::OpaqueDone);
  EXPECT_EQ(view->mode, 7u);
  EXPECT_FALSE(view->uintSource);
  EXPECT_EQ(view->kind, GbufViewKind::Target);
  EXPECT_TRUE(shadow_view(*view));
  EXPECT_EQ(find_view(48), nullptr);
  EXPECT_EQ(parse_view("vis.sun")->id, 48u);
  EXPECT_EQ(parse_view("sun")->id, 48u);
  for (const GbufView& other : GbufViews) {
    EXPECT_EQ(shadow_view(other), other.id >= 48 && other.id <= 52) << other.name;
  }
  const std::optional<std::string> blocked = view_blocked(*view, nullptr);
  ASSERT_TRUE(blocked.has_value());
  EXPECT_EQ(*blocked, "needs real-time shadows");
}

TEST_F(RenderViews, ShadowRowsHaveFreshIds) {
  struct Row {
    uint16_t id;
    std::string_view name;
    std::string_view shortName;
    VsmViewMode mode;
  };
  constexpr std::array<Row, 4> Rows{{
      {49, "vsm.level", "level", VsmViewMode::Level},
      {50, "vsm.pages", "pages", VsmViewMode::Pages},
      {51, "vsm.table", "table", VsmViewMode::Table},
      {52, "vsm.pool", "pool", VsmViewMode::Pool},
  }};
  for (const Row& row : Rows) {
    const GbufView* view = find_gbuf_view(row.id);
    ASSERT_NE(view, nullptr) << row.name;
    EXPECT_LE(view->id, MaxDebugViewId);
    EXPECT_EQ(view->name, row.name);
    EXPECT_TRUE(view->label.starts_with("Sun shadows: ")) << view->label;
    EXPECT_EQ(view->resource, VsmViewResource);
    EXPECT_EQ(view->anchor, "phase.composite");
    EXPECT_EQ(view->stage, Stage::OpaqueDone);
    EXPECT_EQ(view->mode, 18u) << "the snapshot copies the view's RGB";
    EXPECT_EQ(view->kind, GbufViewKind::Target);
    EXPECT_FALSE(view->uintSource);
    EXPECT_TRUE(shadow_view(*view));
    EXPECT_EQ(find_view(row.id), nullptr);
    EXPECT_EQ(parse_view(row.name)->id, row.id);
    EXPECT_EQ(parse_view(row.shortName)->id, row.id);
    EXPECT_EQ(vsm_view::row_mode(*view), row.mode);
    EXPECT_EQ(vsm_view::selected_mode(row.id), row.mode);
  }
  EXPECT_EQ(vsm_view::row_mode(*find_gbuf_view(48)), std::nullopt);
  EXPECT_EQ(vsm_view::selected_mode(0), std::nullopt);
  EXPECT_EQ(find_gbuf_view(53), nullptr);
  EXPECT_EQ(GbufViews.back().id, 52u);
  EXPECT_EQ(static_cast<uint32_t>(VsmViewMode::NoSun) + 1, VsmViewModeCount);
  std::set<std::tuple<uint8_t, uint8_t, uint8_t>> levels;
  for (const Rgb8 c : VsmLevelColors) {
    EXPECT_TRUE(levels.insert({c.r, c.g, c.b}).second) << "levels need distinct colours";
  }
  EXPECT_EQ(vsm_cached_color(0), vsm_table_color(VsmTableState::Cached));
  EXPECT_EQ(vsm_cached_color(48), vsm_cached_color(1000));
  EXPECT_EQ(halve(vsm_page_color(VsmPageState::Rendered)), (Rgb8{127, 80, 0}));
}

TEST_F(RenderViews, ShadowRowsBlockedWithoutShadows) {
  auto graph = std::make_shared<detail::CompiledGraph>();
  graph->sceneDepthSource = DepthSource::Direct;
  detail::FrameState frame{.settings = Settings{.pbr = true, .ao = Quality::High}, .graph = graph};
  for (uint16_t id = 48; id <= 52; ++id) {
    const GbufView* view = find_gbuf_view(id);
    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view_blocked(*view, nullptr), "needs real-time shadows") << view->name;
    frame.settings.shadows.quality = Quality::Off;
    EXPECT_EQ(view_blocked(*view, &frame), "needs real-time shadows") << view->name;
    frame.settings.shadows.quality = Quality::High;
    EXPECT_EQ(view_blocked(*view, &frame), std::nullopt) << view->name;
  }
}

TEST_F(RenderViews, RegistryListsEveryBuiltView) {
  const std::span<const DebugViewInfo> views = registry();
  ASSERT_EQ(views.size(), ClassViews.size() + GbufViews.size());
  uint16_t last = 0;
  std::set<std::string_view> names;
  for (const DebugViewInfo& view : views) {
    EXPECT_GT(view.id, last) << view.name;
    last = view.id;
    EXPECT_LE(view.id, MaxDebugViewId);
    EXPECT_FALSE(view.id >= 9 && view.id <= 15) << view.id;
    EXPECT_NE(view.id, 35u);
    EXPECT_TRUE(names.insert(view.name).second) << view.name;
    EXPECT_TRUE(known_view(view.id));
    const DebugView* cls = find_view(view.id);
    const GbufView* gbuf = find_gbuf_view(view.id);
    ASSERT_TRUE(cls != nullptr || gbuf != nullptr) << view.id;
    EXPECT_EQ(view.label, cls != nullptr ? cls->label : gbuf->label);
  }
  const std::span<const DebugViewInfo> api = debug_views();
  ASSERT_EQ(api.size(), views.size());
  EXPECT_EQ(api.data(), views.data());
  EXPECT_TRUE(known_view(0));
  for (const uint16_t reserved : {uint16_t{9}, uint16_t{15}, uint16_t{35}, uint16_t{53}, uint16_t{100}}) {
    EXPECT_FALSE(known_view(reserved)) << reserved;
  }
}

TEST_F(RenderViews, ParseSelectsRetainedRowsByName) {
  const auto id = [](std::string_view text) {
    const auto selection = parse_view(text);
    return selection ? std::optional{selection->id} : std::nullopt;
  };
  EXPECT_EQ(id("retained.wireframe"), 43u);
  EXPECT_EQ(id("retained.clusters"), 44u);
  EXPECT_EQ(id("retained.coverage"), 45u);
  EXPECT_EQ(id("wireframe"), std::nullopt);
  EXPECT_EQ(id("clusters"), std::nullopt);
  EXPECT_EQ(id("coverage"), 26u);
  EXPECT_EQ(id("45"), 45u);
  EXPECT_EQ(parse_view("retained.coverage")->bit, 0u);
  EXPECT_EQ(id("retained."), std::nullopt);
  EXPECT_EQ(id("retained.wire"), std::nullopt);
}

TEST_F(RenderViews, RetainedRowBlockedWithoutRetainedScene) {
  ASSERT_FALSE(scene::retained::available(caps()));
  for (const uint16_t row : {uint16_t{43}, uint16_t{44}, uint16_t{45}, uint16_t{46}, uint16_t{47}}) {
    const std::optional<std::string> blocked = view_blocked(*find_gbuf_view(row), nullptr);
    ASSERT_TRUE(blocked.has_value()) << row;
    EXPECT_EQ(*blocked, "needs the retained scene");
  }
}

TEST_F(RenderViews, RetainedRowSetsAndClearsDemand) {
  constexpr auto View = static_cast<uint8_t>(scene::retained::Demand::View);
  const auto demanded = [] { return (scene::retained::testing::demand() & View) != 0; };
  ASSERT_FALSE(demanded());
  set_view_override(43);
  views_on_latch();
  EXPECT_TRUE(demanded());
  set_view_override(16);
  views_on_latch();
  EXPECT_FALSE(demanded());
  set_view_override(1);
  views_on_latch();
  EXPECT_FALSE(demanded());
  set_view_override(45);
  views_on_latch();
  EXPECT_TRUE(demanded());
  testing::reset_views();
  EXPECT_FALSE(demanded());
  views_on_latch();
  EXPECT_TRUE(demanded());
  shutdown_classification();
  EXPECT_FALSE(demanded());
}

TEST_F(RenderViews, RetainedViewClampsToCaps) {
  namespace flag = scene::streamout::instance_flag;
  std::vector<scene::streamout::LiveInstance> live(20000);
  for (uint32_t i = 0; i < live.size(); ++i) {
    live[i] = {.slot = i % scene::retained::MaxSlots,
               .indexCount = 3,
               .clusterFirst = i * 10,
               .clusterCount = 10,
               .flags = flag::Valid,
               .classBits = scene::retained::class_bit::Static};
  }
  const std::vector<uint64_t> seen(scene::retained::MaxSlots / 64, ~uint64_t{0});
  const retained_view::Packed clusters = retained_view::pack(live, seen, RetainedMode::Clusters);
  EXPECT_EQ(clusters.instances.size(), retained_view::MaxViewInstances);
  EXPECT_EQ(clusters.overInstances, live.size() - retained_view::MaxViewInstances);
  EXPECT_LE(clusters.pairs.size(), retained_view::MaxViewPairs);
  EXPECT_EQ(clusters.pairs.size() + clusters.overPairs, uint64_t{retained_view::MaxViewInstances} * 10);
  EXPECT_GT(clusters.overPairs, 0u);
  uint32_t clustered = 0;
  for (const retained_view::ViewInstance& view : clusters.instances) {
    clustered += (view.flags & retained_view::instance_flag::Clustered) != 0 ? 1 : 0;
  }
  EXPECT_EQ(clustered * 10, clusters.pairs.size());
  EXPECT_EQ(clusters.pairs.front(), (std::array<uint32_t, 2>{0, 0}));
  EXPECT_EQ(clusters.pairs[11], (std::array<uint32_t, 2>{1, 11}));
  const retained_view::Packed wire =
      retained_view::pack(std::span{live}.first(10), seen, RetainedMode::Wireframe, {.instances = 4, .pairs = 8});
  EXPECT_EQ(wire.instances.size(), 4u);
  EXPECT_EQ(wire.overInstances, 6u);
  EXPECT_TRUE(wire.pairs.empty());
  EXPECT_EQ(wire.overPairs, 0u);
}

TEST_F(RenderViews, RetainedViewColoursFollowClass) {
  namespace flag = scene::streamout::instance_flag;
  namespace bit = scene::retained::class_bit;
  using scene::streamout::LiveInstance;
  const auto cls = [](uint32_t flags, uint32_t classBits, uint32_t clusterCount) {
    return retained_view::classify({.clusterCount = clusterCount, .flags = flags, .classBits = classBits});
  };
  EXPECT_EQ(cls(flag::Valid, bit::Static, 4), RetainedClass::StaticClustered);
  EXPECT_EQ(cls(flag::Valid, bit::Immobile, 0), RetainedClass::StaticUnclustered);
  EXPECT_EQ(cls(flag::Valid, bit::Dynamic, 0), RetainedClass::Dynamic);
  EXPECT_EQ(cls(flag::Valid | flag::Palette, bit::Dynamic | bit::Palette, 2), RetainedClass::Palette);
  EXPECT_EQ(cls(flag::Valid | flag::CpuDeformed | flag::Palette, bit::Dynamic, 0), RetainedClass::Deformed);
  EXPECT_EQ(cls(flag::Valid | flag::GpuArrays, bit::Static, 0), RetainedClass::Other);
  EXPECT_EQ(cls(flag::Valid, 0, 0), RetainedClass::Other);
  EXPECT_TRUE(all_distinct(RetainedClassColors));

  std::vector<uint64_t> seen(scene::retained::MaxSlots / 64, 0);
  seen[0] = uint64_t{1} << 3;
  const std::array<LiveInstance, 3> live{{
      {.slot = 3, .indexCount = 6, .flags = flag::Valid, .classBits = bit::Dynamic},
      {.slot = 4, .indexCount = 6, .flags = flag::Valid, .classBits = bit::Dynamic},
      {.slot = 5, .indexCount = 6, .flags = flag::Valid | flag::Late, .classBits = bit::Dynamic},
  }};
  const retained_view::Packed packed = retained_view::pack(live, seen, RetainedMode::Coverage);
  ASSERT_EQ(packed.instances.size(), 3u);
  const Rgb8 dynamic = retained_class_color(RetainedClass::Dynamic);
  EXPECT_EQ(packed.instances[0].color, retained_view::pack_color(dynamic, true));
  EXPECT_EQ(packed.instances[1].color, retained_view::pack_color(dynamic, false));
  EXPECT_EQ(packed.instances[2].color, retained_view::pack_color(dynamic, true));
  EXPECT_EQ(retained_view::pack_color({255, 160, 0}, false), 0xFF00507Fu);
  EXPECT_EQ(packed.instances[1].slot, 4u);
  EXPECT_EQ(packed.instances[1].indexCount, 6u);

  const std::array<LiveInstance, 2> statics{{
      {.classBits = bit::Static, .origin = {120.0, 9000.0, -300.0}},
      {.classBits = bit::Dynamic, .origin = {90000.0, 0.0, 0.0}},
  }};
  EXPECT_FLOAT_EQ(retained_view::fit_extent(statics, {0.0, 0.0, 0.0}), retained_view::MinCoverageExtent);
  EXPECT_FLOAT_EQ(retained_view::fit_extent(statics, {0.0, 0.0, -4300.0}), 4200.f);
  EXPECT_FLOAT_EQ(retained_view::fit_extent(statics, {-1.0e5, 0.0, 0.0}), retained_view::MaxCoverageExtent);
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

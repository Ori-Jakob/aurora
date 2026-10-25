#include <gtest/gtest.h>

#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "render/debug/fake_shadows.hpp"
#include "render/fakeshadow/texture_gate.hpp"
#include "render/frame_state.hpp"
#include "render/hints.hpp"
#include "render/json.hpp"
#include "render/material/gx_lookup.hpp"
#include "render/material/registry.hpp"
#include "render/scene/classify.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"

#include <aurora/texture.hpp>

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aurora::render::debug::fake_shadows {
namespace {
namespace category = scene::category;
using fakeshadow::Force;
using fakeshadow::Gate;
using fakeshadow::Neutral;
using fakeshadow::Outcome;

constexpr uint32_t Side = 8;

size_t at(Outcome outcome) { return static_cast<size_t>(outcome); }

fakeshadow::Rule rule(uint16_t id, Neutral neutral, Gate gate, uint16_t scope) {
  return {
      .materialId = id,
      .neutral = neutral,
      .gateBit = static_cast<uint8_t>(1u << static_cast<uint32_t>(gate)),
      .scopeMask = scope != 0 ? scope : uint16_t{0x1FFF},
      .scoped = scope != 0,
  };
}

Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

gx::GXBindGroups stub_builder(const gx::ShaderInfo&, uint8_t whiteMask, uint8_t clearMask, bool) noexcept {
  return {.textureBindGroup = 0x100000u + whiteMask * 0x100u + clearMask};
}

class RenderFakeShadowsTab : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
    fakeshadow::testing::set_neutral_builder(&stub_builder);
  }

  void TearDown() override {
    testing::reset();
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

  static void scope(uint32_t categories) {
    sink().scopePush(sink().userData, {.categories = categories, .role = AURORA_CAMERA_INHERIT});
  }
  static void pop() { sink().scopePop(sink().userData); }

  static void latch() {
    on_latch();
    scene::apply_frame_state(detail::FrameState{});
  }

  static std::shared_ptr<const fakeshadow::TextureReport> seal_report() {
    detail::begin_hint_frame();
    return fakeshadow::latest_report();
  }

  void bind(uint32_t slot, gfx::TextureMaterialRef material) {
    gfx::TextureBind& bind = gx::g_gxState.textures[slot];
    bind.ref =
        std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                          wgpu::Extent3D{Side, Side, 1}, wgpu::TextureFormat::RGBA8Unorm, 1, GX_TF_I8);
    bind.material = material;
    m_info.sampledTextures.set(slot);
  }

  void draws(uint32_t categories, int count) const {
    scope(categories);
    for (int i = 0; i < count; ++i) {
      const gx::draw_route::Router* router = gx::draw_route::router();
      router->route(router->userData, {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3});
      gx::DrawData draw{};
      draw.bindGroups.textureBindGroup = 0x1111;
      draw.instanceCount = 1;
      router->push(router->userData, draw, m_config, m_info);
    }
    pop();
  }

  gx::PipelineConfig m_config{};
  gx::ShaderInfo m_info{};
};

TEST_F(RenderFakeShadowsTab, RowsFollowReport) {
  EXPECT_TRUE(build_rows(nullptr, nullptr).rows.empty());

  auto table = std::make_shared<fakeshadow::RuleTable>();
  table->generation = 3;
  table->rules = {
      {rule(5, Neutral::White, Gate::Light, category::Terrain | category::Static), "tex1_8x8_0000000000000005_9"},
      {rule(6, Neutral::Skip, Gate::Cloud, 0), "tex1_8x8_0000000000000006_9"},
      {rule(7, Neutral::Clear, Gate::Ao, category::Water), "tex1_8x8_0000000000000007_9"},
  };
  table->unscoped = 1;
  fakeshadow::TextureReport report;
  report.table = table;
  report.counts.resize(2);
  report.counts[0][at(Outcome::Bound)] = 4;
  report.counts[0][at(Outcome::InScope)] = 3;
  report.counts[0][at(Outcome::White)] = 3;
  report.counts[0][at(Outcome::OutOfScope)] = 1;

  const Rows rows = build_rows(&report, nullptr);
  ASSERT_EQ(rows.rows.size(), 3u);
  const Row& white = rows.rows[0];
  EXPECT_EQ(white.materialId, 5u);
  EXPECT_EQ(white.key, "tex1_8x8_0000000000000005_9");
  EXPECT_EQ(white.neutral, Neutral::White);
  EXPECT_EQ(white.gateBit, 1u << static_cast<uint32_t>(Gate::Light));
  EXPECT_EQ(scope_text(white.scopeMask, white.scoped), "STATIC TERRAIN");
  EXPECT_FALSE(white.registered);
  EXPECT_FALSE(white.unseen);
  EXPECT_EQ(white.counts, report.counts[0]);
  EXPECT_EQ(scope_text(rows.rows[1].scopeMask, rows.rows[1].scoped), "all");
  EXPECT_TRUE(rows.rows[1].unseen);
  EXPECT_TRUE(rows.rows[2].unseen);
  EXPECT_EQ(rows.unscoped, 1u);
  EXPECT_EQ(rows.unseen, 2u);
  EXPECT_EQ(rows.totals[at(Outcome::Bound)], 4u);
  EXPECT_EQ(rows.totals[at(Outcome::White)], 3u);
  EXPECT_EQ(scope_text(0, true), "none");

  const std::vector<uint8_t> bytes(Side * Side, 0x5A);
  const texture::TextureSourceKey key = *texture::compute_source_key(GX_TF_I8, Side, Side, bytes);
  const std::string stem = texture::format_material_key(key);
  const std::string manifest = fmt::format(
      R"({{"version":1,"materials":{{"{}":{{"fakeShadow":{{"neutral":"clear","gate":"cloud","scope":["WATER"]}}}}}}}})",
      stem);
  ASSERT_EQ(texture::register_material_manifest(manifest, {.layer = texture::MaterialLayer::User}).status,
            texture::MaterialManifestStatus::Accepted);
  material::latch();
  const auto snapshot = material::latched_snapshot();
  const uint16_t id = snapshot->find(key);
  ASSERT_NE(id, 0u);
  table->rules[0].rule.materialId = id;
  table->rules[0].key.clear();
  const Rows joined = build_rows(&report, snapshot.get());
  EXPECT_TRUE(joined.rows[0].registered);
  EXPECT_EQ(joined.rows[0].key, stem);
  EXPECT_EQ(joined.rows[0].neutral, Neutral::Clear);
  EXPECT_EQ(joined.rows[0].gateBit, 1u << static_cast<uint32_t>(Gate::Cloud));
  EXPECT_EQ(scope_text(joined.rows[0].scopeMask, joined.rows[0].scoped), "WATER");
  EXPECT_FALSE(joined.rows[1].registered);
  EXPECT_EQ(joined.rows[1].key, "tex1_8x8_0000000000000006_9");

  const json::Doc doc = json::read(report_json(&report, joined));
  ASSERT_NE(doc, nullptr);
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "generation")), 3u);
  yyjson_val* list = yyjson_obj_get(root, "rules");
  ASSERT_EQ(yyjson_arr_size(list), 3u);
  yyjson_val* first = yyjson_arr_get(list, 0);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(first, "key")), stem.c_str());
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(first, "neutral")), "clear");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(yyjson_obj_get(first, "counts"), "white")), 3u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(yyjson_arr_get(list, 1), "scope")), "all");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(yyjson_obj_get(root, "totals"), "bound")), 4u);
}

TEST_F(RenderFakeShadowsTab, PerKeyCountsInAndOutOfScope) {
  std::vector<uint8_t> bytes(Side * Side);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i * 13 + 7);
  }
  const texture::TextureSourceKey key = *texture::compute_source_key(GX_TF_I8, Side, Side, bytes);
  const std::string manifest = fmt::format(
      R"({{"version":1,"materials":{{"{}":{{"fakeShadow":{{"neutral":"white","gate":"light","scope":["TERRAIN","STATIC"]}}}}}}}})",
      texture::format_material_key(key));
  ASSERT_EQ(texture::register_material_manifest(manifest, {.layer = texture::MaterialLayer::User}).status,
            texture::MaterialManifestStatus::Accepted);
  material::latch();

  queue_force(Gate::Light, Force::On);
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::Auto);
  latch();
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::On);
  main_scene();
  bind(0, material::testing::gx_find(key));
  ASSERT_NE(gx::g_gxState.textures[0].material.id, 0u);
  draws(category::Terrain, 3);
  draws(category::Water, 2);
  draws(category::FakeShadow | category::Decal, 1);
  const auto report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->active);
  const Rows rows = build_rows(report.get(), material::latched_raw());
  ASSERT_EQ(rows.rows.size(), 1u);
  const Row& row = rows.rows[0];
  EXPECT_TRUE(row.registered);
  EXPECT_EQ(row.key, texture::format_material_key(key));
  EXPECT_EQ(scope_text(row.scopeMask, row.scoped), "STATIC TERRAIN");
  EXPECT_EQ(row.counts[at(Outcome::Bound)], 6u);
  EXPECT_EQ(row.counts[at(Outcome::InScope)], 3u);
  EXPECT_EQ(row.counts[at(Outcome::White)], 3u);
  EXPECT_EQ(row.counts[at(Outcome::OutOfScope)], 2u);
  EXPECT_EQ(row.counts[at(Outcome::EntryOwned)], 1u);
  EXPECT_EQ(rows.totals, row.counts);

  testing::force_visible(1);
  queue_force(Gate::Light, Force::Off);
  latch();
  main_scene();
  draws(category::Terrain, 3);
  draws(category::Water, 2);
  draws(category::FakeShadow | category::Decal, 1);
  const auto off = seal_report();
  ASSERT_NE(off, nullptr);
  EXPECT_FALSE(off->active);
  EXPECT_TRUE(off->observing);
  const Rows offRows = build_rows(off.get(), material::latched_raw());
  ASSERT_EQ(offRows.rows.size(), 1u);
  EXPECT_EQ(offRows.rows[0].counts[at(Outcome::Bound)], 6u);
  EXPECT_EQ(offRows.rows[0].counts[at(Outcome::White)], 0u);
  EXPECT_EQ(offRows.rows[0].counts[at(Outcome::GateOff)], 3u);
}

TEST_F(RenderFakeShadowsTab, ForceQueueAppliesAtLatch) {
  const std::array rules{rule(5, Neutral::White, Gate::Light, category::Terrain)};
  fakeshadow::testing::set_rules(rules);

  queue_force(Gate::Light, Force::Off);
  queue_force(Gate::Light, Force::On);
  EXPECT_EQ(shown_force(Gate::Light), Force::On);
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::Auto);
  latch();
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::On);
  auto report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->active);
  EXPECT_TRUE(report->gates[static_cast<size_t>(Gate::Light)].effective);
  EXPECT_EQ(report->gates[static_cast<size_t>(Gate::Light)].force, Force::On);

  queue_force(Gate::Light, Force::Auto);
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::On);
  latch();
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::Auto);
  report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_FALSE(report->active);
  EXPECT_FALSE(report->observing);

  testing::force_visible(1);
  latch();
  report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_TRUE(report->observing);
  testing::force_visible(0);
  latch();
  report = seal_report();
  ASSERT_NE(report, nullptr);
  EXPECT_FALSE(report->observing);

  fakeshadow::set_force(Gate::Cloud, Force::On);
  queue_force(Gate::Ao, Force::Off);
  latch();
  queue_force(Gate::Light, Force::On);
  shutdown();
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::Auto);
  EXPECT_EQ(fakeshadow::force(Gate::Ao), Force::Auto);
  EXPECT_EQ(fakeshadow::force(Gate::Cloud), Force::On);
  on_latch();
  EXPECT_EQ(fakeshadow::force(Gate::Light), Force::Auto);
}
} // namespace
} // namespace aurora::render::debug::fake_shadows

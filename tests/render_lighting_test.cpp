#include <gtest/gtest.h>

#include "render_depth_device.hpp"
#include "render_lighting_reference.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_layout.hpp"
#include "render/graph_pool.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/lighting/clusters.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/scene/receiver_category.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_camera.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace clusters = lighting::clusters;
namespace ref = aurora::test::lighting;
namespace rc = scene::receiver_category;
using aurora::test::scene_depth;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float Near = 10.f;
constexpr float Far = 100000.f;
constexpr uint16_t HeatmapView = 36;

Mat4x4<float> perspective(float n, float f, float p02 = 0.f, float p12 = 0.f) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m0[2] = p02;
  proj.m1[1] = 1.7320508f;
  proj.m1[2] = p12;
  proj.m2[2] = std::isinf(f) ? 0.f : -n / (f - n);
  proj.m2[3] = std::isinf(f) ? -n : -(f * n) / (f - n);
  proj.m3[2] = -1.f;
  return proj;
}

Mat3x4<float> identity_view() {
  return {Vec4<float>{1.f, 0.f, 0.f, 0.f}, Vec4<float>{0.f, 1.f, 0.f, 0.f}, Vec4<float>{0.f, 0.f, 1.f, 0.f}};
}

std::shared_ptr<SceneHints> scene_hints(std::vector<AuroraLight> lights, const Mat4x4<float>& proj,
                                        std::array<float, 2> jitter = {}, const Mat3x4<float>& view = identity_view()) {
  auto hints = std::make_shared<SceneHints>();
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .view = view,
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
  hints->present = SceneHasMainCamera | SceneHasLights;
  hints->lights = std::move(lights);
  hints->motionValid = true;
  hints->jitterPx[0] = jitter[0];
  hints->jitterPx[1] = jitter[1];
  return hints;
}

AuroraLight point(uint32_t id, std::array<float, 3> position, float radius, uint16_t flags = 0) {
  AuroraLight light{};
  light.id = id;
  light.type = AURORA_LIGHT_POINT;
  light.flags = flags;
  std::ranges::copy(position, light.position);
  light.color[0] = light.color[1] = light.color[2] = 0.5f;
  light.radius = radius;
  light.spotInnerCos = 1.f;
  light.spotOuterCos = -1.f;
  light.falloffExponent = 2.f;
  light.volumetricScale = 1.f;
  return light;
}

AuroraLight directional(uint32_t id, std::array<float, 3> direction, uint16_t flags = 0) {
  AuroraLight light = point(id, {}, 0.f, flags);
  light.type = AURORA_LIGHT_DIRECTIONAL;
  std::ranges::copy(direction, light.direction);
  return light;
}

AuroraLight spot(uint32_t id, std::array<float, 3> position, std::array<float, 3> direction, float radius, float inner,
                 float outer) {
  AuroraLight light = point(id, position, radius);
  light.type = AURORA_LIGHT_SPOT;
  std::ranges::copy(direction, light.direction);
  light.spotInnerCos = inner;
  light.spotOuterCos = outer;
  return light;
}

struct Prepared {
  clusters::LightListHeader header;
  clusters::UploadStats stats;
  std::vector<clusters::GpuLight> lights = std::vector<clusters::GpuLight>(clusters::MaxLights);

  [[nodiscard]] std::span<const clusters::GpuLight> used() const {
    return std::span{lights}.first(header.dirCount + header.localCount);
  }
};

Prepared prepare(const SceneHints& hints, uint32_t skipMask = 0) {
  Prepared out;
  clusters::prepare_lights(hints, skipMask, clusters::LightSpan{out.lights.data(), clusters::MaxLights}, out.header,
                           out.stats);
  return out;
}

clusters::LightListHeader default_header() { return prepare(SceneHints{}).header; }

TEST(ReceiverCategory, ByteTableIsFrozen) {
  constexpr std::array<std::pair<uint32_t, uint8_t>, 8> Table{{
      {AURORA_CAT_STATIC, 0x01},
      {AURORA_CAT_DYNAMIC, 0x02},
      {AURORA_CAT_SKINNED, 0x04},
      {AURORA_CAT_TERRAIN, 0x08},
      {AURORA_CAT_FOLIAGE, 0x10},
      {AURORA_CAT_WATER, 0x20},
      {AURORA_CAT_DECAL, 0x40},
      {AURORA_CAT_PLAYER, 0x80},
  }};
  uint32_t covered = 0;
  for (const auto& [category, byte] : Table) {
    EXPECT_EQ(rc::compress(category), byte) << category;
    EXPECT_EQ(rc::light_byte(category).byte, byte) << category;
    EXPECT_FALSE(rc::light_byte(category).widened) << category;
    covered |= category;
  }
  EXPECT_EQ(covered, rc::Covered);
  EXPECT_EQ(rc::Covered, 0x53Fu);
  EXPECT_EQ(rc::compress(rc::Covered), 0xFFu);
  for (uint32_t bit = 0; bit < 32; ++bit) {
    if ((rc::Covered & (1u << bit)) == 0) {
      EXPECT_EQ(rc::compress(1u << bit), 0u) << bit;
    }
  }
  EXPECT_EQ(rc::compress(AURORA_CAT_STATIC | AURORA_CAT_TERRAIN), 0x09u);
}

TEST(ReceiverCategory, CameraFillMaskIs0x82) {
  const rc::LightByte fill = rc::light_byte(AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER);
  EXPECT_EQ(fill.byte, 0x82u);
  EXPECT_FALSE(fill.widened);
  EXPECT_EQ(rc::light_byte(0x402).byte, 0x82u);
  EXPECT_EQ(rc::compress(scene::category::Dynamic | scene::category::Player | scene::category::Skinned), 0x86u);
}

TEST(ReceiverCategory, MaskOutsideSetIsAll) {
  EXPECT_EQ(rc::light_byte(0).byte, 0u);
  EXPECT_FALSE(rc::light_byte(0).widened);
  for (const uint32_t mask : {uint32_t{AURORA_CAT_EMISSIVE}, uint32_t{AURORA_CAT_SKY},
                              uint32_t{AURORA_CAT_STATIC | AURORA_CAT_EMISSIVE}, 0x80000000u}) {
    const rc::LightByte byte = rc::light_byte(mask);
    EXPECT_EQ(byte.byte, 0u) << mask;
    EXPECT_TRUE(byte.widened) << mask;
  }
  EXPECT_EQ(rc::light_byte(rc::Covered).byte, 0xFFu);
  EXPECT_FALSE(rc::light_byte(rc::Covered).widened);
}

TEST(LightUpload, DropsInvalid) {
  constexpr float Nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float Inf = std::numeric_limits<float>::infinity();
  std::vector<AuroraLight> lights;
  lights.push_back(point(1, {0.f, 0.f, -100.f}, 50.f));
  lights.push_back(point(2, {Nan, 0.f, -100.f}, 50.f));
  AuroraLight hot = point(3, {0.f, 0.f, -100.f}, 50.f);
  hot.color[1] = Inf;
  lights.push_back(hot);
  AuroraLight black = point(4, {0.f, 0.f, -100.f}, 50.f);
  black.color[0] = black.color[1] = black.color[2] = 0.f;
  lights.push_back(black);
  lights.push_back(point(5, {0.f, 0.f, -100.f}, 0.f));
  lights.push_back(point(6, {0.f, 0.f, -100.f}, -1.f));
  lights.push_back(point(7, {0.f, 0.f, -100.f}, Inf));
  lights.push_back(directional(8, {0.f, 0.f, 0.f}));
  lights.push_back(spot(9, {0.f, 0.f, -100.f}, {0.f, 1e-9f, 0.f}, 50.f, 0.9f, 0.5f));
  lights.push_back(spot(10, {0.f, 0.f, -100.f}, {0.f, -1.f, 0.f}, 50.f, Nan, 0.5f));
  AuroraLight badType = point(11, {0.f, 0.f, -100.f}, 50.f);
  badType.type = 3;
  lights.push_back(badType);
  lights.push_back(directional(12, {0.f, -1.f, 0.f}));
  AuroraLight negative = point(13, {0.f, 0.f, -100.f}, 50.f);
  negative.color[0] = -1.f;
  negative.falloffExponent = Nan;
  negative.sourceRadius = Inf;
  negative.volumetricScale = Nan;
  lights.push_back(negative);
  const Prepared p = prepare(*scene_hints(lights, perspective(Near, Far)));
  EXPECT_EQ(p.stats.invalid, 10u);
  EXPECT_EQ(p.stats.skipped, 0u);
  ASSERT_EQ(p.header.dirCount, 1u);
  ASSERT_EQ(p.header.localCount, 2u);
  EXPECT_EQ(p.lights[0].v4[1], 12u);
  EXPECT_EQ(p.lights[1].v4[1], 1u);
  const clusters::GpuLight& kept = p.lights[2];
  EXPECT_EQ(kept.v4[1], 13u);
  EXPECT_EQ(kept.v2[0], 0.f);
  EXPECT_EQ(kept.v3[0], clusters::DefaultFalloff);
  EXPECT_EQ(kept.v3[2], 0.f);
  EXPECT_EQ(kept.v3[3], 1.f);
  for (const clusters::GpuLight& light : p.used()) {
    for (const float value : {light.v0[0], light.v0[1], light.v0[2], light.v0[3], light.v1[0], light.v1[1], light.v1[2],
                              light.v2[0], light.v2[1], light.v2[2], light.v2[3]}) {
      EXPECT_TRUE(std::isfinite(value)) << light.v4[1];
    }
  }
  AuroraLight bright = point(14, {0.f, 0.f, -100.f}, 50.f);
  bright.color[0] = 1e9f;
  bright.falloffExponent = 9.f;
  const Prepared clamped = prepare(*scene_hints({bright}, perspective(Near, Far)));
  ASSERT_EQ(clamped.header.localCount, 1u);
  EXPECT_EQ(clamped.lights[0].v2[0], clusters::MaxColour);
  EXPECT_EQ(clamped.lights[0].v3[0], clusters::MaxFalloff);
}

TEST(LightUpload, SkipMaskDropsFlaggedLights) {
  const std::vector<AuroraLight> lights{
      directional(1, {0.f, -1.f, 0.f}, AURORA_LIGHT_CAMERA_FILL | AURORA_LIGHT_NO_SHADOW),
      directional(2, {0.f, -1.f, 0.f}, AURORA_LIGHT_KEY_ROOM),
      point(3, {0.f, 0.f, -100.f}, 50.f, AURORA_LIGHT_FLICKER),
  };
  const auto hints = scene_hints(lights, perspective(Near, Far));
  const Prepared all = prepare(*hints);
  EXPECT_EQ(all.header.dirCount, 2u);
  EXPECT_EQ(all.header.flags & clusters::HeaderSkipMask, 0u);
  const Prepared fill = prepare(*hints, AURORA_LIGHT_CAMERA_FILL);
  EXPECT_EQ(fill.header.dirCount, 1u);
  EXPECT_EQ(fill.header.localCount, 1u);
  EXPECT_EQ(fill.lights[0].v4[1], 2u);
  EXPECT_EQ(fill.stats.skipped, 1u);
  EXPECT_NE(fill.header.flags & clusters::HeaderSkipMask, 0u);
  const Prepared key = prepare(*hints, AURORA_LIGHT_KEY_ROOM);
  EXPECT_EQ(key.header.dirCount, 1u);
  EXPECT_EQ(key.lights[0].v4[1], 1u);
  EXPECT_EQ(clusters::light_flags(key.lights[0]), uint32_t{AURORA_LIGHT_CAMERA_FILL | AURORA_LIGHT_NO_SHADOW});
  const Prepared both = prepare(*hints, AURORA_LIGHT_CAMERA_FILL | AURORA_LIGHT_KEY_ROOM | AURORA_LIGHT_FLICKER);
  EXPECT_EQ(both.header.dirCount + both.header.localCount, 0u);
  EXPECT_EQ(both.stats.skipped, 3u);
}

TEST(LightUpload, OrdersSpotCones) {
  const std::vector<AuroraLight> lights{
      spot(1, {0.f, 0.f, -100.f}, {0.f, 0.f, -1.f}, 50.f, 0.5f, 0.9f),
      spot(2, {0.f, 0.f, -100.f}, {0.f, 0.f, -1.f}, 50.f, 0.95f, 0.8f),
      spot(3, {0.f, 0.f, -100.f}, {0.f, 0.f, -1.f}, 50.f, 0.7f, 0.7f),
      spot(4, {0.f, 0.f, -100.f}, {0.f, 0.f, -1.f}, 50.f, 3.f, -3.f),
      point(5, {0.f, 0.f, -100.f}, 50.f),
  };
  const Prepared p = prepare(*scene_hints(lights, perspective(Near, Far)));
  ASSERT_EQ(p.header.localCount, 5u);
  const auto cone_at = [](const clusters::GpuLight& light, float cosine) {
    const float t = std::clamp(cosine * light.v1[3] + light.v3[1], 0.f, 1.f);
    return t * t;
  };
  struct Cone {
    float inner;
    float outer;
  };
  for (const auto& [index, cone] :
       {std::pair{0, Cone{0.9f, 0.5f}}, std::pair{1, Cone{0.95f, 0.8f}}, std::pair{3, Cone{1.f, -1.f}}}) {
    const clusters::GpuLight& light = p.lights[index];
    EXPECT_EQ(clusters::light_type(light), uint32_t{AURORA_LIGHT_SPOT});
    EXPECT_NEAR(light.v1[3], 1.f / (cone.inner - cone.outer), 1e-3f) << index;
    EXPECT_NEAR(cone_at(light, cone.inner), 1.f, 1e-5f) << index;
    EXPECT_NEAR(cone_at(light, cone.outer), 0.f, 1e-5f) << index;
  }
  EXPECT_FLOAT_EQ(p.lights[2].v1[3], 1e4f);
  EXPECT_EQ(cone_at(p.lights[2], 0.71f), 1.f);
  EXPECT_EQ(cone_at(p.lights[2], 0.69f), 0.f);
  EXPECT_EQ(p.lights[4].v1[3], 0.f);
  EXPECT_EQ(p.lights[4].v3[1], 1.f);
  EXPECT_EQ(p.lights[0].v1[2], 1.f);
}

TEST(LightUpload, DirectionalsFirstStableOrder) {
  const std::vector<AuroraLight> lights{
      point(10, {0.f, 0.f, -100.f}, 50.f),
      directional(20, {0.f, -1.f, 0.f}),
      point(30, {0.f, 0.f, -200.f}, 50.f),
      directional(40, {1.f, 0.f, 0.f}),
      spot(50, {0.f, 0.f, -300.f}, {0.f, 0.f, -1.f}, 50.f, 0.9f, 0.5f),
  };
  const Prepared p = prepare(*scene_hints(lights, perspective(Near, Far)));
  ASSERT_EQ(p.header.dirCount, 2u);
  ASSERT_EQ(p.header.localCount, 3u);
  constexpr std::array<std::pair<uint32_t, uint32_t>, 5> Want{{{20, 1}, {40, 3}, {10, 0}, {30, 2}, {50, 4}}};
  for (size_t i = 0; i < Want.size(); ++i) {
    EXPECT_EQ(p.lights[i].v4[1], Want[i].first) << i;
    EXPECT_EQ(clusters::hint_index(p.lights[i]), Want[i].second) << i;
  }
  EXPECT_EQ(p.header.flags & clusters::HeaderTruncated, 0u);
  EXPECT_EQ(p.header.version, 1u);
}

TEST(LightUpload, TruncatesDirectionalsAt16) {
  std::vector<AuroraLight> lights;
  for (uint32_t i = 0; i < 20; ++i) {
    lights.push_back(directional(100 + i, {0.f, -1.f, 0.f}));
  }
  lights.push_back(point(1, {0.f, 0.f, -100.f}, 50.f));
  lights.push_back(point(2, {0.f, 0.f, -100.f}, 50.f));
  const Prepared p = prepare(*scene_hints(lights, perspective(Near, Far)));
  EXPECT_EQ(p.header.dirCount, clusters::MaxDirectional);
  EXPECT_EQ(p.header.localCount, 2u);
  EXPECT_EQ(p.stats.truncated, 4u);
  EXPECT_NE(p.header.flags & clusters::HeaderTruncated, 0u);
  for (uint32_t i = 0; i < clusters::MaxDirectional; ++i) {
    EXPECT_EQ(p.lights[i].v4[1], 100 + i);
  }
  EXPECT_EQ(p.lights[16].v4[1], 1u);
}

TEST(LightUpload, TruncatesAt256) {
  std::vector<AuroraLight> lights;
  for (uint32_t i = 0; i < 300; ++i) {
    lights.push_back(point(1000 + i, {0.f, 0.f, -100.f - static_cast<float>(i)}, 50.f));
  }
  for (uint32_t i = 0; i < 10; ++i) {
    lights.push_back(directional(i, {0.f, -1.f, 0.f}));
  }
  const Prepared p = prepare(*scene_hints(lights, perspective(Near, Far)));
  EXPECT_EQ(p.header.dirCount, 10u);
  EXPECT_EQ(p.header.localCount, 246u);
  EXPECT_EQ(p.stats.truncated, 54u);
  EXPECT_NE(p.header.flags & clusters::HeaderTruncated, 0u);
  EXPECT_EQ(p.lights[10].v4[1], 1000u);
  EXPECT_EQ(p.lights[255].v4[1], 1245u);
  Prepared stressed = prepare(*scene_hints({directional(1, {0.f, -1.f, 0.f})}, perspective(Near, Far)));
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(scene_hints({}, perspective(Near, Far)).get(), camera));
  EXPECT_EQ(
      clusters::append_stress_lights(300, camera, clusters::LightSpan{stressed.lights.data(), 256}, stressed.header),
      255u);
  EXPECT_EQ(stressed.header.localCount, 255u);
  EXPECT_NE(stressed.header.flags & clusters::HeaderStress, 0u);
  EXPECT_EQ(stressed.header.flags & clusters::HeaderTruncated, 0u);
}

TEST(LightUpload, ViewFrameMatchesReconstruction) {
  const Mat4x4<float> proj = perspective(Near, Far, 0.03f, -0.02f);
  const double angle = 0.6;
  const float c = static_cast<float>(2.0 * std::cos(angle));
  const float s = static_cast<float>(2.0 * std::sin(angle));
  const Mat3x4<float> view{Vec4<float>{c, 0.f, -s, 300.f}, Vec4<float>{0.f, 2.f, 0.f, -40.f},
                           Vec4<float>{s, 0.f, c, -1200.f}};
  const std::array<float, 3> world{150.f, 75.f, -900.f};
  AuroraLight light = point(1, world, 40.f);
  light.sourceRadius = 3.f;
  const auto hints = scene_hints({light, directional(2, {0.f, 0.f, -1.f})}, proj, {}, view);
  const Prepared p = prepare(*hints);
  ASSERT_EQ(p.header.localCount, 1u);
  ASSERT_EQ(p.header.dirCount, 1u);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
  std::array<double, 3> v{};
  const std::array<const Vec4<float>*, 3> rows{&view.m0, &view.m1, &view.m2};
  for (size_t i = 0; i < 3; ++i) {
    v[i] = (*rows[i])[0] * double{world[0]} + (*rows[i])[1] * double{world[1]} + (*rows[i])[2] * double{world[2]} +
           (*rows[i])[3];
  }
  const double z = -v[2];
  ASSERT_GT(z, 0.0);
  const double ndcX = (double{camera.p00} * v[0] + double{camera.p02} * v[2]) / z;
  const double ndcY = (double{camera.p11} * v[1] + double{camera.p12} * v[2]) / z;
  const ref::Vec3d rebuilt = ref::view_position_fwd(camera, (ndcX + 1.0) / 2.0, (1.0 - ndcY) / 2.0, z, 1280, 720);
  const clusters::GpuLight& local = p.lights[1];
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(local.v0[i], rebuilt[i], 1e-3 * std::max(std::abs(rebuilt[i]), 1.0)) << i;
  }
  EXPECT_NEAR(local.v0[3], 80.f, 1e-3f);
  EXPECT_NEAR(local.v3[2], 6.f, 1e-4f);
  EXPECT_NEAR(local.v2[3], 254.f / (80.f * 80.f), 1e-6f);
  const clusters::GpuLight& sun = p.lights[0];
  EXPECT_NEAR(sun.v1[0], s / 2.f, 1e-5f);
  EXPECT_NEAR(sun.v1[1], 0.f, 1e-6f);
  EXPECT_NEAR(sun.v1[2], c / 2.f, 1e-5f);
  EXPECT_NEAR(std::hypot(sun.v1[0], sun.v1[1], sun.v1[2]), 1.f, 1e-6f);
  EXPECT_EQ(p.header.unitsPerMetre, DefaultUnitsPerMetre);
  EXPECT_EQ(p.header.sliceNear, clusters::SliceNearMetres * DefaultUnitsPerMetre);
}

TEST(LightUpload, LayoutIsFrozen) {
  EXPECT_EQ(sizeof(clusters::LightListHeader), 32u);
  EXPECT_EQ(offsetof(clusters::LightListHeader, flags), 12u);
  EXPECT_EQ(offsetof(clusters::LightListHeader, sliceNear), 16u);
  EXPECT_EQ(offsetof(clusters::LightListHeader, sliceLogScale), 20u);
  EXPECT_EQ(offsetof(clusters::LightListHeader, unitsPerMetre), 24u);
  EXPECT_EQ(sizeof(clusters::GpuLight), 80u);
  EXPECT_EQ(offsetof(clusters::GpuLight, v1), 16u);
  EXPECT_EQ(offsetof(clusters::GpuLight, v2), 32u);
  EXPECT_EQ(offsetof(clusters::GpuLight, v3), 48u);
  EXPECT_EQ(offsetof(clusters::GpuLight, v4), 64u);
  EXPECT_EQ(clusters::ListBytes, 20512u);
  EXPECT_EQ(clusters::GridBytes, 110592u);
  EXPECT_EQ(sizeof(clusters::Params), 64u);
  EXPECT_EQ(offsetof(clusters::Params, flags), 48u);
  EXPECT_EQ(clusters::ClusterCount, 3456u);
  EXPECT_EQ(clusters::cluster_index(15, 8, 23), clusters::ClusterCount - 1);
  EXPECT_EQ(clusters::cluster_index(1, 0, 0), 1u);
  EXPECT_EQ(clusters::cluster_index(0, 1, 0), 16u);
  EXPECT_EQ(clusters::cluster_index(0, 0, 1), 144u);
  AuroraLight light = point(0xABCD, {0.f, 0.f, -100.f}, 50.f, AURORA_LIGHT_FLICKER | AURORA_LIGHT_CAMERA_FILL);
  light.categoryMask = AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER;
  light.shadowPriority = 7;
  const Prepared p = prepare(*scene_hints({directional(1, {0.f, -1.f, 0.f}), light}, perspective(Near, Far)));
  ASSERT_EQ(p.header.localCount, 1u);
  EXPECT_EQ(p.lights[1].v4[0], 1u | 0x82u << 8 | uint32_t{AURORA_LIGHT_FLICKER | AURORA_LIGHT_CAMERA_FILL} << 16);
  EXPECT_EQ(p.lights[1].v4[1], 0xABCDu);
  EXPECT_EQ(p.lights[1].v4[2], 7u | 1u << 8);
  EXPECT_EQ(p.lights[1].v4[3], 0u);
  light.categoryMask = AURORA_CAT_EMISSIVE;
  const Prepared widened = prepare(*scene_hints({light}, perspective(Near, Far)));
  EXPECT_EQ(clusters::light_byte(widened.lights[0]), 0u);
  EXPECT_EQ(widened.stats.widened, 1u);
  EXPECT_NE(widened.header.flags & clusters::HeaderMaskWidened, 0u);
}

TEST(LightGrid, SliceFormulaIsFrozen) {
  const clusters::LightListHeader header = default_header();
  EXPECT_EQ(header.sliceNear, 50.f);
  EXPECT_FLOAT_EQ(header.sliceLogScale, static_cast<float>(24.0 / std::log(2000.0)));
  EXPECT_EQ(clusters::SliceScale[0], 1.f);
  EXPECT_EQ(clusters::SliceScale[24], 2000.f);
  for (uint32_t s = 0; s <= clusters::Slices; ++s) {
    EXPECT_EQ(clusters::SliceScale[s], static_cast<float>(std::pow(2000.0, s / 24.0))) << s;
  }
  for (uint32_t s = 0; s < clusters::Slices; ++s) {
    const float lo = header.sliceNear * clusters::SliceScale[s];
    const float mid = header.sliceNear * std::sqrt(clusters::SliceScale[s] * clusters::SliceScale[s + 1]);
    EXPECT_EQ(clusters::slice_of(mid, header), s) << s;
    EXPECT_EQ(clusters::slice_of(lo * 1.001f, header), s) << s;
  }
  EXPECT_EQ(clusters::slice_of(0.f, header), 0u);
  EXPECT_EQ(clusters::slice_of(-5.f, header), 0u);
  EXPECT_EQ(clusters::slice_of(49.f, header), 0u);
  EXPECT_EQ(clusters::slice_of(std::numeric_limits<float>::quiet_NaN(), header), 0u);
  EXPECT_EQ(clusters::slice_of(100000.f * 10.f, header), 23u);
  EXPECT_EQ(clusters::slice_of(1e30f, header), 23u);
  EXPECT_EQ(clusters::slice_of(std::numeric_limits<float>::infinity(), header), 23u);
  EXPECT_EQ(clusters::pixel_cluster(0, 0, 64, 64, 1.f, header), 0u);
  EXPECT_EQ(clusters::pixel_cluster(63, 63, 64, 64, 1e30f, header), clusters::ClusterCount - 1);
}

TEST(LightGrid, PixelCentresInsideTheirFroxel) {
  for (const auto [width, height] :
       {std::pair{61u, 37u}, std::pair{2560u, 1440u}, std::pair{4779u, 2688u}, std::pair{4301u, 2688u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    for (const auto [size, tiles] : {std::pair{width, clusters::TilesX}, std::pair{height, clusters::TilesY}}) {
      uint32_t last = 0;
      size_t bad = 0;
      for (uint32_t p = 0; p < size; ++p) {
        const uint32_t tile = clusters::tile_of(p, size, tiles);
        const uint64_t centre = uint64_t{2 * p + 1} * tiles;
        bad += tile >= tiles || centre < uint64_t{2} * size * tile || centre >= uint64_t{2} * size * (tile + 1) ||
                       tile < last
                   ? 1
                   : 0;
        const float u = (static_cast<float>(p) + 0.5f) / static_cast<float>(size);
        const float step = tiles == 16 ? 1.f / 16.f : 1.f / 9.f;
        bad += u < static_cast<float>(tile) * step * (1.f - 1e-6f) ||
                       u > static_cast<float>(tile + 1) * step * (1.f + 1e-6f)
                   ? 1
                   : 0;
        last = tile;
      }
      EXPECT_EQ(bad, 0u) << tiles;
      EXPECT_EQ(last, tiles - 1) << tiles;
    }
  }
  EXPECT_EQ(clusters::tile_of(597, 4779, 16), 2u);
  EXPECT_EQ(clusters::tile_of(596, 4779, 16), 1u);
  EXPECT_GE((597.5f / 4779.f), 2.f / 16.f);
}

std::vector<AuroraLight> seeded_lights(std::mt19937& rng, uint32_t count, uint32_t directionals) {
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  std::vector<AuroraLight> lights;
  for (uint32_t i = 0; i < directionals && i < count; ++i) {
    lights.push_back(directional(0x100 + i, {unit(rng) - 0.5f, -1.f, unit(rng) - 0.5f}));
  }
  for (uint32_t i = static_cast<uint32_t>(lights.size()); i < count; ++i) {
    const float z = unit(rng) < 0.1f ? -2000.f * unit(rng) : 30.f + 120000.f * std::pow(unit(rng), 2.f);
    const float spread = std::max(std::abs(z), 500.f) * 1.4f;
    const std::array<float, 3> position{(2.f * unit(rng) - 1.f) * spread, (2.f * unit(rng) - 1.f) * spread, -z};
    const float reach = std::max(std::abs(z), 1000.f);
    float radius = reach * (0.03f + 0.5f * unit(rng));
    if (i % 37 == 5) {
      radius = 160000.f;
    }
    if (i % 4 == 3) {
      lights.push_back(spot(0x1000 + i, position, {unit(rng) - 0.5f, unit(rng) - 0.5f, -1.f}, radius, 0.9f, 0.6f));
    } else {
      lights.push_back(point(0x1000 + i, position, radius));
    }
  }
  return lights;
}

TEST(LightGrid, ReferenceIsConservative) {
  std::mt19937 rng{0x11C7u};
  constexpr uint32_t Width = 2560;
  constexpr uint32_t Height = 1440;
  const auto hints = scene_hints(seeded_lights(rng, 256, 0), perspective(Near, Far, 0.02f, -0.01f), {0.3f, -0.2f});
  const Prepared p = prepare(*hints);
  ASSERT_EQ(p.header.localCount, 256u);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
  const std::vector<ref::Mask> masks = ref::assign_reference(p.header, p.used(), camera, Width, Height,
                                                             1.0 + clusters::RadiusInflate, clusters::DepthInflate);
  std::uniform_int_distribution<uint32_t> pick(0, clusters::ClusterCount - 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  uint64_t inside = 0;
  uint64_t samples = 0;
  for (uint32_t c = 0; c < 200; ++c) {
    const uint32_t cluster = pick(rng);
    const uint32_t tx = cluster % clusters::TilesX;
    const uint32_t ty = (cluster / clusters::TilesX) % clusters::TilesY;
    const uint32_t slice = cluster / (clusters::TilesX * clusters::TilesY);
    const std::array<float, 2> depths = ref::cluster_depths(camera, p.header, slice);
    for (uint32_t k = 0; k < 64; ++k) {
      const auto px = static_cast<uint32_t>((tx + unit(rng)) * Width / clusters::TilesX);
      const auto py = static_cast<uint32_t>((ty + unit(rng)) * Height / clusters::TilesY);
      const double lo = std::max<double>(depths[0], Near);
      const double hi = std::min<double>(std::max(depths[1], depths[0]), Far);
      const float z = static_cast<float>(lo * std::pow(hi / lo, unit(rng)));
      if (px >= Width || py >= Height || clusters::pixel_cluster(px, py, Width, Height, z, p.header) != cluster) {
        continue;
      }
      ++samples;
      const ref::Vec3d pos = ref::view_position_fwd(camera, (px + 0.5) / Width, (py + 0.5) / Height, z, Width, Height);
      for (uint32_t i = 0; i < p.header.localCount; ++i) {
        const clusters::GpuLight& light = p.lights[i];
        const double d = std::hypot(pos[0] - light.v0[0], pos[1] - light.v0[1], pos[2] - light.v0[2]);
        if (d <= light.v0[3]) {
          ++inside;
          ASSERT_NE(masks[cluster][i / 32] & (1u << (i % 32)), 0u)
              << "cluster " << cluster << " pixel " << px << "," << py << " z " << z << " light " << i;
        }
      }
    }
  }
  EXPECT_GT(samples, 2000u);
  EXPECT_GT(inside, 1000u);
}

struct Readback {
  wgpu::Buffer buffer;
  wgpu::Texture texture;
  std::string_view resource;
  bool isBuffer = true;
  int calls = 0;
};

void declare_readback(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& reader = *static_cast<const Readback*>(state);
  builder.after(phase::Lighting);
  if (reader.isBuffer) {
    builder.use_buffer(reader.resource, Access::CopySrc);
  } else {
    builder.use(reader.resource, Access::CopySrc);
  }
  builder.side_effect();
}

void encode_readback(PassContext& ctx, const void* state) {
  auto& reader = *static_cast<Readback*>(const_cast<void*>(state));
  ++reader.calls;
  if (!reader.isBuffer) {
    reader.texture = ctx.texture(TexRef{0});
    return;
  }
  const BufferView view = ctx.buffer(BufRef{0});
  if (view.buffer && reader.buffer) {
    ctx.encoder().CopyBufferToBuffer(view.buffer, view.offset, reader.buffer, 0,
                                     std::min(view.size, reader.buffer.GetSize()));
  }
}

PassDesc readback_desc(Readback& reader, std::string_view name) {
  return {
      .name = std::string{name},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 5,
      .declare = declare_readback,
      .encode = encode_readback,
      .state = &reader,
  };
}

Settings pbr_settings(float legacyBlend = 0.f) {
  Settings settings;
  settings.pbr = true;
  settings.legacyBlend = legacyBlend;
  return settings;
}

class LightDeviceTest : public ::testing::Test {
protected:
  bool open(const test::TestDeviceOptions& options) {
    m_options = options;
    m_options.createEfb = true;
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      return false;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    gpu_heap::shutdown();
    reset_module_state();
    jobs::initialize(1);
    shaders::initialize();
    register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
    testing::set_hints_override(1);
    install_hints();
    m_started = true;
    clusters::initialize();
    return use_pbr_tier();
  }

  void TearDown() override {
    if (m_started) {
      debug::set_view_override(0);
      webgpu::set_gbuffer(false);
      testing::reset_hints();
      shaders::shutdown();
      jobs::shutdown();
      reset_module_state();
      gpu_heap::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    clusters::shutdown();
    clusters::testing::reset();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  bool use_pbr_tier() {
    if (!webgpu::set_gbuffer(true)) {
      m_error = "the base G-buffer tier needs a core device";
      return false;
    }
    render::Caps c = render::caps();
    c.limits.maxColorAttachmentBytesPerSample = 128;
    webgpu::set_caps(c);
    if (!webgpu::set_scene_tier(webgpu::SceneTier::Pbr)) {
      m_error = "the PBR G-buffer tier is unavailable";
      return false;
    }
    return true;
  }

  void use_globals(const wgpu::Extent3D& size) {
    webgpu::set_gbuffer(false);
    m_globals.reset();
    m_options.efbSize = size;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
    clusters::initialize();
    ASSERT_TRUE(use_pbr_tier());
  }

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(const Settings& settings,
                                                       DepthSource source = DepthSource::Direct) {
    m_settings = settings;
    m_env = gather_graph_environment();
    m_env.viewOverride = m_view;
    std::shared_ptr<CompiledGraph> graph = compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = source});
    prepare_graph_bindings(*graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = source});
    return graph;
  }

  void latch(const CompiledGraph* graph) {
    clusters::on_latch({.settings = &m_settings, .env = &m_env}, graph);
    latch_pass_params();
  }

  static void wait_ready(const CompiledGraph& graph) {
    for (const CompiledNode& node : graph.nodes) {
      if (node.kind != NodeKind::Pass || node.status != CullReason::None) {
        continue;
      }
      for (const uint32_t id : node.pipelineIds) {
        if (id != 0) {
          ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << node.name;
        }
      }
    }
  }

  std::shared_ptr<CompiledGraph> live_graph(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    wait_ready(*graph);
    latch(graph.get());
    return graph;
  }

  static CullReason status(const CompiledGraph& graph, std::string_view name) {
    const NodeId node = graph.find_node(name);
    return node != InvalidNode ? graph.nodes[node].status : CullReason::Skipped;
  }

  static bool pass_enabled(const CompiledGraph& graph, const StageData& stage, std::string_view name) {
    const NodeId node = graph.find_node(name);
    return node != InvalidNode && graph.nodes[node].programIndex != InvalidNode &&
           stage.is_enabled(graph.nodes[node].programIndex);
  }

  StageData build(const FrameState& state, uint64_t frameId, std::shared_ptr<const SceneDepth> depth) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    StageData opaque{
        .stage = Stage::OpaqueDone,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = true,
        .frame = &state,
        .sceneDepth = std::move(depth),
    };
    build_stage_data(opaque);
    return opaque;
  }

  static gfx::Range uniform_sink(const uint8_t* data, size_t length) {
    const size_t alignment = std::max<size_t>(gfx::detail::resources().limits.minUniformBufferOffsetAlignment, 1);
    const size_t offset = (s_uniforms.size() + alignment - 1) / alignment * alignment;
    s_uniforms.resize(offset + length);
    std::memcpy(s_uniforms.data() + offset, data, length);
    return {static_cast<uint32_t>(offset), static_cast<uint32_t>(length)};
  }

  std::optional<std::string> encode(StageData& opaque) const {
    if (opaque.any_enabled()) {
      FrameUniforms frame;
      const GraphExtent efb = opaque.frame->graph->efb_extent(Stage::OpaqueDone);
      frame.constants.efbSize[0] = efb.width;
      frame.constants.efbSize[1] = efb.height;
      s_uniforms.assign(FrameUniformsSize, 0);
      std::memcpy(s_uniforms.data(), &frame, sizeof(frame));
      build_stage_bindings(opaque, &uniform_sink);
      s_uniforms.resize((s_uniforms.size() + 3) & ~size_t{3});
      m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, s_uniforms.data(), s_uniforms.size());
    }
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    if (opaque.any_enabled()) {
      run_stage(encoder, opaque);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    return test::pop_validation_error(*m_device);
  }

  [[nodiscard]] std::shared_ptr<const SceneDepth> depth_of(uint32_t width, uint32_t height) const {
    return scene_depth(test::make_depth(*m_device, width, height));
  }

  wgpu::Buffer staging(uint64_t size) const {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
                                            .size = size};
    return m_device->device.CreateBuffer(&descriptor);
  }

  std::vector<ref::Mask> read_masks(const wgpu::Buffer& buffer) const {
    const std::optional<std::vector<uint32_t>> words = test::read_buffer(*m_device, buffer, 0, clusters::GridBytes);
    std::vector<ref::Mask> masks(clusters::ClusterCount);
    if (!words || words->size() != size_t{clusters::ClusterCount} * clusters::MaskWords) {
      ADD_FAILURE() << "grid readback failed";
      return masks;
    }
    for (size_t i = 0; i < masks.size(); ++i) {
      std::copy_n(words->begin() + static_cast<std::ptrdiff_t>(i * clusters::MaskWords), clusters::MaskWords,
                  masks[i].begin());
    }
    return masks;
  }

  static inline std::vector<uint8_t> s_uniforms;
  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
  uint16_t m_view = 0;
};

class LightClustersGraph : public LightDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 64, 1}})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    clusters::testing::set_hints_override(scene_hints({}, perspective(Near, Far)));
  }
};

TEST_F(LightClustersGraph, DeclaresOnlyUnderPbrSurface) {
  Readback grid{.resource = clusters::Grid};
  register_pass(readback_desc(grid, "t.grid_read"));
  const auto live = compile(pbr_settings());
  EXPECT_EQ(status(*live, clusters::Pass), CullReason::None);
  const CompiledResource& list = live->resources[live->find_resource(clusters::List)];
  EXPECT_TRUE(list.buffer);
  EXPECT_EQ(list.bufferSize, clusters::ListBytes);
  EXPECT_TRUE(list.bufferUsage & wgpu::BufferUsage::CopyDst);
  EXPECT_TRUE(list.bufferUsage & wgpu::BufferUsage::Storage);
  EXPECT_EQ(live->resources[live->find_resource(clusters::Grid)].bufferSize, clusters::GridBytes);
  EXPECT_EQ(status(*compile(pbr_settings(1.f)), clusters::Pass), CullReason::None);
  EXPECT_EQ(status(*compile(Settings{}), clusters::Pass), CullReason::Skipped);
  for (const DepthSource source : {DepthSource::Copy, DepthSource::Unavailable}) {
    EXPECT_EQ(status(*compile(pbr_settings(), source), clusters::Pass), CullReason::Skipped)
        << scene_depth_source_name(source);
  }
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  EXPECT_EQ(status(*compile(pbr_settings()), clusters::Pass), CullReason::Skipped);
  EXPECT_TRUE(clusters::ready(Settings{}));
  m_settings = pbr_settings();
  latch(compile(m_settings).get());
  EXPECT_FALSE(clusters::ready(m_settings));
  EXPECT_FALSE(clusters::armed());
}

TEST_F(LightClustersGraph, CulledWithoutConsumer) {
  const auto graph = compile(pbr_settings());
  EXPECT_EQ(status(*graph, clusters::Pass), CullReason::NoConsumer);
  EXPECT_EQ(graph->nodes[graph->find_node(clusters::Pass)].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
  EXPECT_EQ(status(*graph, clusters::HeatmapPass), CullReason::Skipped);
  const bool logged = std::ranges::any_of(graph->diagnostics, [](const Diagnostic& d) {
    return d.message == "render graph: nothing reads what 'light.clusters' writes; culled";
  });
  EXPECT_TRUE(logged);
  latch(graph.get());
  EXPECT_FALSE(clusters::armed());
}

TEST_F(LightClustersGraph, HeatmapDeclaresOnlyWhenSelected) {
  Readback heat{.resource = clusters::HeatmapResource, .isBuffer = false};
  register_pass(readback_desc(heat, "t.heatmap_read"));
  EXPECT_EQ(status(*compile(pbr_settings()), clusters::HeatmapPass), CullReason::Skipped);
  for (const uint16_t other : {uint16_t{21}, uint16_t{2}, uint16_t{30}}) {
    m_view = other;
    EXPECT_EQ(status(*compile(pbr_settings()), clusters::HeatmapPass), CullReason::Skipped) << other;
  }
  m_view = HeatmapView;
  const auto graph = live_graph(pbr_settings());
  EXPECT_EQ(status(*graph, clusters::HeatmapPass), CullReason::None);
  EXPECT_EQ(status(*graph, clusters::Pass), CullReason::None);
  const CompiledResource& heatmap = graph->resources[graph->find_resource(clusters::HeatmapResource)];
  EXPECT_EQ(heatmap.tex.format, wgpu::TextureFormat::RGBA8Unorm);
  EXPECT_EQ(heatmap.extent.width, 64u);
  EXPECT_EQ(status(*compile(Settings{}), clusters::HeatmapPass), CullReason::Skipped);
  EXPECT_TRUE(clusters::ready(m_settings));
  const FrameState state{.graph = graph};
  StageData stage = build(state, 1, depth_of(64, 64));
  EXPECT_TRUE(pass_enabled(*graph, stage, clusters::Pass));
  EXPECT_TRUE(pass_enabled(*graph, stage, clusters::HeatmapPass));
  EXPECT_EQ(encode(stage), std::nullopt);
  EXPECT_EQ(heat.calls, 1);
  clusters::testing::set_hints_override(std::make_shared<SceneHints>());
  StageData noCamera = build(state, 2, depth_of(64, 64));
  EXPECT_FALSE(pass_enabled(*graph, noCamera, clusters::Pass));
  EXPECT_FALSE(pass_enabled(*graph, noCamera, clusters::HeatmapPass));
}

class LightClustersGpu : public LightDeviceTest {
protected:
  void SetUp() override {
    const wgpu::BackendType backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!open({.backend = backend})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    if (select_scene_depth_source(m_device->core, 1, true) != DepthSource::Direct) {
      GTEST_SKIP() << "needs direct scene depth";
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
  }

  std::vector<ref::Mask> run_grid(const std::shared_ptr<const SceneHints>& hints, uint32_t width, uint32_t height,
                                  uint64_t frame = 1) {
    clusters::testing::set_hints_override(hints);
    Readback grid{.buffer = staging(clusters::GridBytes), .resource = clusters::Grid};
    register_pass(readback_desc(grid, "t.grid_read"));
    const auto graph = live_graph(pbr_settings());
    const FrameState state{.graph = graph};
    StageData stage = build(state, frame, depth_of(width, height));
    EXPECT_TRUE(pass_enabled(*graph, stage, clusters::Pass));
    EXPECT_EQ(encode(stage), std::nullopt);
    EXPECT_EQ(grid.calls, 1);
    return read_masks(grid.buffer);
  }
};

TEST_F(LightClustersGpu, MasksBracketReference) {
  std::mt19937 rng{0xC1A57u};
  uint64_t frame = 0;
  for (const auto [width, height] : {std::pair{61u, 37u}, std::pair{2560u, 1440u}, std::pair{4779u, 2688u}}) {
    use_globals({width, height, 1});
    Readback grid{.buffer = staging(clusters::GridBytes), .resource = clusters::Grid};
    register_pass(readback_desc(grid, "t.grid_read"));
    const auto graph = live_graph(pbr_settings());
    const FrameState state{.graph = graph};
    const auto depth = depth_of(width, height);
    for (const bool jitter : {false, true}) {
      for (const uint32_t count : {0u, 1u, 20u, 256u}) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " jitter " + std::to_string(jitter) +
                     " lights " + std::to_string(count));
        const std::array<float, 2> offset = jitter ? std::array{0.37f, -0.41f} : std::array{0.f, 0.f};
        const auto hints =
            scene_hints(seeded_lights(rng, count, count >= 20 ? 2 : 0), perspective(Near, Far, 0.015f, -0.01f), offset);
        clusters::testing::set_hints_override(hints);
        StageData stage = build(state, ++frame, depth);
        ASSERT_TRUE(pass_enabled(*graph, stage, clusters::Pass));
        ASSERT_EQ(encode(stage), std::nullopt);
        const std::vector<ref::Mask> gpu = read_masks(grid.buffer);
        const Prepared p = prepare(*hints);
        StageCamera camera;
        ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
        const double radius = 1.0 + clusters::RadiusInflate;
        const std::vector<ref::Mask> low = ref::assign_reference(p.header, p.used(), camera, width, height,
                                                                 radius * (1.0 - 1e-4), clusters::DepthInflate * 0.99);
        const std::vector<ref::Mask> high = ref::assign_reference(p.header, p.used(), camera, width, height,
                                                                  radius * (1.0 + 1e-4), clusters::DepthInflate * 1.01);
        size_t missing = 0;
        size_t extra = 0;
        size_t bits = 0;
        uint32_t first = UINT32_MAX;
        for (uint32_t cluster = 0; cluster < clusters::ClusterCount; ++cluster) {
          for (uint32_t w = 0; w < clusters::MaskWords; ++w) {
            const uint32_t got = gpu[cluster][w];
            missing += std::popcount(low[cluster][w] & ~got);
            extra += std::popcount(got & ~high[cluster][w]);
            bits += std::popcount(got);
            if (first == UINT32_MAX && ((low[cluster][w] & ~got) != 0 || (got & ~high[cluster][w]) != 0)) {
              first = cluster;
            }
          }
        }
        EXPECT_EQ(missing, 0u) << "first cluster " << first;
        EXPECT_EQ(extra, 0u) << "first cluster " << first;
        if (count >= 20) {
          EXPECT_GT(bits, size_t{count});
        }
      }
    }
    EXPECT_EQ(grid.calls, 8);
  }
}

TEST_F(LightClustersGpu, ZeroLightsWriteZeroGrid) {
  std::mt19937 rng{0x2E40u};
  use_globals({320, 180, 1});
  const auto lit = scene_hints(seeded_lights(rng, 64, 0), perspective(Near, Far));
  clusters::testing::set_hints_override(lit);
  Readback grid{.buffer = staging(clusters::GridBytes), .resource = clusters::Grid};
  register_pass(readback_desc(grid, "t.grid_read"));
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  StageData first = build(state, 1, depth_of(320, 180));
  ASSERT_EQ(encode(first), std::nullopt);
  const auto any = [](const std::vector<ref::Mask>& masks) {
    return std::ranges::any_of(masks, [](const ref::Mask& m) { return std::ranges::any_of(m, std::identity{}); });
  };
  ASSERT_TRUE(any(read_masks(grid.buffer)));
  auto dark = scene_hints({}, perspective(Near, Far));
  dark->present = SceneHasMainCamera;
  clusters::testing::set_hints_override(dark);
  StageData second = build(state, 2, depth_of(320, 180));
  ASSERT_EQ(encode(second), std::nullopt);
  EXPECT_FALSE(any(read_masks(grid.buffer)));
  EXPECT_EQ(clusters::testing::counters().frames, 2u);
  EXPECT_EQ(clusters::testing::counters().maxLights, 64u);
}

TEST_F(LightClustersGpu, BruteForceSetsAllWords) {
  use_globals({128, 72, 1});
  clusters::testing::set_brute_force(true);
  const std::vector<ref::Mask> masks =
      run_grid(scene_hints({point(1, {0.f, 0.f, -100.f}, 50.f)}, perspective(Near, Far)), 128, 72);
  for (const ref::Mask& mask : masks) {
    for (const uint32_t word : mask) {
      ASSERT_EQ(word, 0xFFFFFFFFu);
    }
  }
  clusters::testing::set_brute_force(std::nullopt);
}

class LightHeatmapGpu : public LightClustersGpu {};

TEST_F(LightHeatmapGpu, CountsMatchMasks) {
  constexpr uint32_t Width = 61;
  constexpr uint32_t Height = 37;
  use_globals({Width, Height, 1});
  m_view = HeatmapView;
  std::mt19937 rng{0x4EA7u};
  const Mat4x4<float> proj = perspective(Near, Far);
  const auto hints = scene_hints(seeded_lights(rng, 40, 2), proj);
  clusters::testing::set_hints_override(hints);
  Readback grid{.buffer = staging(clusters::GridBytes), .resource = clusters::Grid};
  Readback heat{.resource = clusters::HeatmapResource, .isBuffer = false};
  register_pass(readback_desc(grid, "t.grid_read"));
  register_pass(readback_desc(heat, "t.heatmap_read"));
  const auto graph = live_graph(pbr_settings());
  ASSERT_EQ(status(*graph, clusters::HeatmapPass), CullReason::None);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
  const Prepared p = prepare(*hints);
  std::vector<float> stored(size_t{Width} * Height);
  for (uint32_t y = 0; y < Height; ++y) {
    for (uint32_t x = 0; x < Width; ++x) {
      const uint32_t slice = (x * 7 + y * 3) % 19;
      if ((x + y) % 11 == 0) {
        stored[size_t{y} * Width + x] = 0.f;
        continue;
      }
      const double z =
          p.header.sliceNear * std::sqrt(double{clusters::SliceScale[slice]} * double{clusters::SliceScale[slice + 1]});
      stored[size_t{y} * Width + x] = static_cast<float>(double{camera.depthAdd} - double{camera.depthMul} / z);
    }
  }
  const wgpu::Texture depth = test::make_depth(*m_device, Width, Height);
  test::fill_depth(*m_device, depth, stored);
  const FrameState state{.graph = graph};
  StageData stage = build(state, 1, scene_depth(depth));
  ASSERT_TRUE(pass_enabled(*graph, stage, clusters::HeatmapPass));
  ASSERT_EQ(encode(stage), std::nullopt);
  ASSERT_EQ(heat.calls, 1);
  ASSERT_TRUE(heat.texture);
  const std::vector<ref::Mask> masks = read_masks(grid.buffer);
  const std::vector<float> texels = test::read_texture_mip(*m_device, heat.texture, 0);
  ASSERT_EQ(texels.size(), size_t{Width} * Height);
  const uint32_t locals = p.header.localCount;
  size_t bad = 0;
  size_t lit = 0;
  std::string first;
  for (uint32_t y = 0; y < Height; ++y) {
    for (uint32_t x = 0; x < Width; ++x) {
      const float d = stored[size_t{y} * Width + x];
      const uint32_t cluster = clusters::pixel_cluster(x, y, Width, Height, ref::view_depth(camera, d), p.header);
      uint32_t count = 0;
      for (uint32_t i = 0; i < locals; ++i) {
        count += (masks[cluster][i / 32] >> (i % 32)) & 1u;
      }
      lit += count != 0 ? 1 : 0;
      const bool border = (x > 0 && clusters::tile_of(x, Width, 16) != clusters::tile_of(x - 1, Width, 16)) ||
                          (y > 0 && clusters::tile_of(y, Height, 9) != clusters::tile_of(y - 1, Height, 9));
      const std::array<uint8_t, 3> want = ref::heatmap_rgb(count, border, d <= 0.f);
      const auto texel = std::bit_cast<uint32_t>(texels[size_t{y} * Width + x]);
      const std::array<uint32_t, 4> got{texel & 0xFF, (texel >> 8) & 0xFF, (texel >> 16) & 0xFF, texel >> 24};
      bool ok = got[3] == std::min(count, 255u);
      for (size_t c = 0; c < 3; ++c) {
        ok = ok && std::abs(static_cast<int>(got[c]) - static_cast<int>(want[c])) <= 1;
      }
      if (!ok && bad++ == 0) {
        first = std::to_string(x) + "," + std::to_string(y) + " count " + std::to_string(count) + " got " +
                std::to_string(got[0]) + "/" + std::to_string(got[1]) + "/" + std::to_string(got[2]) + "/" +
                std::to_string(got[3]);
      }
    }
  }
  EXPECT_EQ(bad, 0u) << first;
  EXPECT_GT(lit, 0u);
  EXPECT_EQ(clusters::testing::counters().heatmapDispatches, 1u);
}
} // namespace
} // namespace aurora::render::detail

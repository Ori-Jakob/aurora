#include <gtest/gtest.h>

#include "render_test_caps.hpp"

#include "gx/hints.hpp"
#include "render/graph.hpp"
#include "render/hints.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/streamout.hpp"
#include "render/graph_bind.hpp"
#include "render/shadows/vsm_alloc.hpp"
#include "render/shadows/vsm_check.hpp"
#include "render/shadows/vsm_cull.hpp"
#include "render/shadows/vsm_invalidate.hpp"
#include "render/shadows/vsm_pages.hpp"
#include "render/stage_data.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace aurora::render::shadows::vsm {
namespace {
using detail::SceneHints;
using detail::StageData;

constexpr double Pi = std::numbers::pi;
constexpr double Texel0 = Texel0Metres * DefaultUnitsPerMetre;

std::array<double, 3> sun_dir(double elevationDeg, double azimuthDeg) {
  const double e = elevationDeg * Pi / 180.0;
  const double a = azimuthDeg * Pi / 180.0;
  return {std::cos(e) * std::sin(a), -std::sin(e), std::cos(e) * std::cos(a)};
}

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double angle(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  const std::array<double, 3> c{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  return std::atan2(std::sqrt(dot(c, c)), dot(a, b));
}

AuroraLight directional(uint32_t id, const std::array<double, 3>& d, uint8_t priority, uint16_t flags = 0,
                        uint32_t mask = 0) {
  AuroraLight light{};
  light.id = id;
  light.type = AURORA_LIGHT_DIRECTIONAL;
  light.shadowPriority = priority;
  light.flags = flags;
  light.categoryMask = mask;
  for (size_t i = 0; i < 3; ++i) {
    light.direction[i] = static_cast<float>(d[i]);
  }
  return light;
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

std::shared_ptr<const SceneHints> make_hints(uint64_t hintFrame, std::vector<AuroraLight> lights = {}) {
  auto hints = std::make_shared<SceneHints>();
  hints->hintFrame = hintFrame;
  hints->present = detail::SceneHasMainCamera | detail::SceneHasLights;
  const Mat4x4<float> proj = perspective();
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .view = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}},
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
  hints->lights = std::move(lights);
  return hints;
}

StageData producing_stage(uint64_t frameId, uint64_t hintFrame = 30) {
  StageData stage{.stage = Stage::OpaqueDone, .frameId = frameId, .markerSeen = true};
  stage.sceneHints = make_hints(hintFrame);
  auto frame = std::make_shared<scene::retained::Frame>();
  frame->plan = std::make_shared<const scene::streamout::Plan>();
  stage.retained = std::move(frame);
  return stage;
}

SunPlan producing_plan() {
  return {.producing = true, .lightId = 7, .dirWorld = sun_dir(40.0, 30.0), .elevationDeg = 40.0, .idle = Idle::None};
}

Views clipmap_views(const std::array<double, 3>& eye, const std::array<double, 3>& dir) {
  SunClipmap clipmap;
  (void)clipmap.update({.eye = eye, .dir = dir, .lightId = 1, .texel0 = Texel0});
  Views views{};
  clipmap.fill(views);
  return views;
}

class RenderVsmGraph : public ::testing::Test {
protected:
  void SetUp() override { testing::reset(); }
  void TearDown() override { testing::reset(); }
};

TEST(RenderVsmPages, ToroidalIndexWrapsNegativeCoords) {
  EXPECT_EQ(page_index(0, -1, -1), 31u + 32u * 31u);
  EXPECT_EQ(page_index(0, -32, 0), 0u);
  EXPECT_EQ(page_index(1024, 33, -33), 1024u + 1u + 32u * 31u);
  for (int32_t p = -100; p <= 100; ++p) {
    EXPECT_EQ(page_index(0, p, 0), page_index(0, p + 32, 0)) << p;
    EXPECT_LT(page_index(0, p, p), PagesPerView) << p;
  }
  for (const int32_t origin : {-5, -32, -33, 0, 7, 1000}) {
    for (uint32_t slot = 0; slot < WindowPages; ++slot) {
      const int32_t page = abs_page(origin, slot);
      EXPECT_GE(page, origin);
      EXPECT_LT(page, origin + static_cast<int32_t>(WindowPages));
      EXPECT_EQ(static_cast<uint32_t>(page) & 31u, slot) << origin;
    }
  }
}

TEST(RenderVsmPages, CoarseLevelsComeFirstInTable) {
  EXPECT_EQ(level_view(SunLevels - 1), 0u);
  EXPECT_EQ(level_view(0), SunLevels - 1);
  const Views views = clipmap_views({100.0, 50.0, -20.0}, sun_dir(45.0, 10.0));
  EXPECT_EQ(views.header.viewCount, SunLevels);
  uint32_t previousOffset = 0;
  float previousTexel = 1e30f;
  for (uint32_t i = 0; i < SunLevels; ++i) {
    const ViewRecord& v = views.views[i];
    EXPECT_EQ(v.table_offset(), i * PagesPerView);
    EXPECT_EQ(v.kind(), ViewKind::SunLevel);
    EXPECT_NE(v.view_flags() & view_flag::Active, 0u);
    EXPECT_EQ(v.pages_x(), WindowPages);
    EXPECT_EQ(v.pages_y(), WindowPages);
    EXPECT_LT(v.texel, previousTexel) << i;
    EXPECT_GE(v.table_offset(), previousOffset);
    previousOffset = v.table_offset();
    previousTexel = v.texel;
  }
  EXPECT_FLOAT_EQ(views.views[level_view(0)].texel, static_cast<float>(Texel0));
  EXPECT_FLOAT_EQ(views.views[0].texel, static_cast<float>(std::ldexp(Texel0, SunLevels - 1)));
  for (uint32_t i = SunLevels; i < MaxViews; ++i) {
    EXPECT_EQ(views.views[i].flags, 0u);
  }
}

TEST(RenderVsmPages, WindowShiftKeepsOverlap) {
  const std::array<int32_t, 2> prev{7, -3};
  for (int32_t dx = -33; dx <= 33; ++dx) {
    for (int32_t dy = -33; dy <= 33; ++dy) {
      const std::array<int32_t, 2> next{prev[0] + dx, prev[1] + dy};
      uint32_t kept = 0;
      for (uint32_t sy = 0; sy < WindowPages; ++sy) {
        for (uint32_t sx = 0; sx < WindowPages; ++sx) {
          if (!kept_after_shift(sx, sy, prev, next)) {
            continue;
          }
          ++kept;
          EXPECT_EQ(abs_page(prev[0], sx), abs_page(next[0], sx));
          EXPECT_EQ(abs_page(prev[1], sy), abs_page(next[1], sy));
        }
      }
      const auto overlap = [](int32_t d) { return static_cast<uint32_t>(std::max(0, 32 - std::abs(d))); };
      EXPECT_EQ(kept, overlap(dx) * overlap(dy)) << dx << ", " << dy;
    }
  }
}

TEST(RenderVsmPages, LevelSelectionStaysInsideWindow) {
  const Views views = clipmap_views({100978.0, 1500.0, -2500.0}, sun_dir(35.0, 120.0));
  std::mt19937 rng{3};
  std::uniform_real_distribution<float> logFootprint(-4.f, 5.f);
  std::uniform_real_distribution<float> offset(-1.f, 1.f);
  std::uniform_int_distribution<int> reach(0, 20);
  constexpr std::array<float, 4> Biases{0.f, 0.5f, -0.5f, 0.9f};
  uint32_t outside = 0;
  for (int i = 0; i < 20000; ++i) {
    const float scale = std::ldexp(1.f, reach(rng));
    const std::array<float, 3> p{offset(rng) * scale, offset(rng) * scale, offset(rng) * scale};
    const float footprint = std::pow(10.f, logFootprint(rng));
    const float bias = Biases[static_cast<size_t>(i) % Biases.size()];
    const uint32_t level = select_level(views, footprint, p, bias);
    const float wanted = std::clamp(std::ceil(std::log2(footprint / static_cast<float>(Texel0)) + bias), 0.f, 11.f);
    if (level == SunLevels) {
      ++outside;
      continue;
    }
    EXPECT_GE(static_cast<float>(level), wanted);
    const std::array<float, 3> c = view_coords(views.views[level_view(level)], p);
    constexpr float Margin = FilterMarginTexels;
    EXPECT_GE(c[0], Margin);
    EXPECT_LT(c[0], WindowTexels - Margin);
    EXPECT_GE(c[1], Margin);
    EXPECT_LT(c[1], WindowTexels - Margin);
    if (level > static_cast<uint32_t>(wanted)) {
      const std::array<float, 3> below = view_coords(views.views[level_view(level - 1)], p);
      EXPECT_FALSE(below[0] >= Margin && below[0] < WindowTexels - Margin && below[1] >= Margin &&
                   below[1] < WindowTexels - Margin);
    }
  }
  EXPECT_GT(outside, 0u);
  EXPECT_LT(outside, 20000u);
}

TEST(RenderVsmPages, CameraRelativeSplitAt1e5) {
  const std::array<double, 3> unit{0.6, 0.012, -0.79975};
  const double length = std::sqrt(dot(unit, unit));
  const std::array<double, 3> eye{unit[0] / length * 100978.0, unit[1] / length * 100978.0,
                                  unit[2] / length * 100978.0};
  ASSERT_NEAR(std::sqrt(dot(eye, eye)), 100978.0, 1e-6);
  SunClipmap clipmap;
  (void)clipmap.update({.eye = eye, .dir = sun_dir(52.0, 210.0), .lightId = 1, .texel0 = Texel0});
  Views views{};
  clipmap.fill(views);
  const std::array<double, 3>& anchor = clipmap.anchor();
  std::mt19937 rng{11};
  std::uniform_real_distribution<double> unitOffset(-1.0, 1.0);
  double worstCam = 0.0;
  double worstAnchor = 0.0;
  double worstDepth = 0.0;
  for (uint32_t level = 0; level < SunLevels; ++level) {
    const ViewRecord& v = views.views[level_view(level)];
    const double radius = (WindowTexels / 2) * static_cast<double>(v.texel);
    for (int i = 0; i < 10000; ++i) {
      const std::array<double, 3> p{eye[0] + unitOffset(rng) * radius, eye[1] + unitOffset(rng) * radius,
                                    eye[2] + unitOffset(rng) * radius};
      std::array<double, 3> reference{};
      for (size_t r = 0; r < 3; ++r) {
        const float* row = v.anchorRel + r * 4;
        reference[r] = row[0] * (p[0] - anchor[0]) + row[1] * (p[1] - anchor[1]) + row[2] * (p[2] - anchor[2]) + row[3];
      }
      const std::array<float, 3> rel{static_cast<float>(p[0] - eye[0]), static_cast<float>(p[1] - eye[1]),
                                     static_cast<float>(p[2] - eye[2])};
      const std::array<float, 3> fromAnchor{static_cast<float>(p[0] - anchor[0]), static_cast<float>(p[1] - anchor[1]),
                                            static_cast<float>(p[2] - anchor[2])};
      const std::array<float, 3> cam = view_coords(v, rel);
      const std::array<float, 3> anc = view_coords_anchor(v, fromAnchor);
      for (size_t r = 0; r < 2; ++r) {
        worstCam = std::max(worstCam, std::abs(cam[r] - reference[r]));
        worstAnchor = std::max(worstAnchor, std::abs(anc[r] - reference[r]));
      }
      worstDepth = std::max({worstDepth, std::abs(cam[2] - reference[2]), std::abs(anc[2] - reference[2])});
    }
  }
  EXPECT_LE(worstCam, 1e-3) << "camera-relative texels";
  EXPECT_LE(worstAnchor, 1e-3) << "anchor-relative texels";
  EXPECT_LE(worstDepth, 1e-6) << "depth";
}

TEST(RenderVsmPages, LatchScheduleMatchesQueue) {
  const std::array<double, 3> eye{2500.0, 300.0, -4000.0};
  SunClipmap clipmap;
  SunClipmap::Result first = clipmap.update({.eye = eye, .dir = sun_dir(20.0, 70.0), .lightId = 3, .texel0 = Texel0});
  EXPECT_TRUE(first.relatchAll);
  std::array<std::array<double, 3>, SunLevels> twin{};
  twin.fill(sun_dir(20.0, 70.0));
  std::array<uint32_t, SunLevels> counts{};
  for (int frame = 1; frame <= 600; ++frame) {
    const std::array<double, 3> dir = sun_dir(20.0 + 0.01 * frame, 70.0);
    int expected = -1;
    for (uint32_t k = 0; k < SunLevels; ++k) {
      if (angle(twin[k], dir) > latch_threshold(k)) {
        expected = static_cast<int>(k);
        break;
      }
    }
    const SunClipmap::Result r = clipmap.update({.eye = eye, .dir = dir, .lightId = 3, .texel0 = Texel0});
    ASSERT_FALSE(r.relatchAll) << frame;
    ASSERT_EQ(r.level, expected) << frame;
    EXPECT_FALSE(r.recentre);
    if (expected >= 0) {
      twin[static_cast<size_t>(expected)] = dir;
      ++counts[static_cast<size_t>(expected)];
    }
    for (uint32_t k = 0; k < SunLevels; ++k) {
      EXPECT_LE(angle(clipmap.latches()[k].dir, dir), latch_threshold(k) + SunLevels * 0.01 * Pi / 180.0);
    }
  }
  EXPECT_GE(counts[0], 50u);
  EXPECT_LE(counts[0], 53u);
  EXPECT_GE(counts[1], 25u);
  EXPECT_LE(counts[1], 27u);
  for (uint32_t k = 4; k < SunLevels; ++k) {
    EXPECT_GE(counts[k], 5u) << k;
    EXPECT_LE(counts[k], 6u) << k;
  }

  const std::array<double, 3> still = sun_dir(26.0, 70.0);
  (void)clipmap.update({.eye = eye, .dir = still, .lightId = 3, .texel0 = Texel0});
  for (int frame = 0; frame < 100; ++frame) {
    const SunClipmap::Result r = clipmap.update({.eye = eye, .dir = still, .lightId = 3, .texel0 = Texel0});
    if (frame > SunLevels) {
      EXPECT_FALSE(r.relatchAll);
      EXPECT_EQ(r.level, -1) << frame;
    }
  }

  const SunClipmap::Result swapped = clipmap.update({.eye = eye, .dir = still, .lightId = 4, .texel0 = Texel0});
  EXPECT_TRUE(swapped.relatchAll);
  EXPECT_TRUE(swapped.keyLight);
  Views views{};
  clipmap.fill(views);
  for (uint32_t k = 0; k < SunLevels; ++k) {
    EXPECT_NE(views.views[level_view(k)].view_flags() & view_flag::Freed, 0u) << k;
  }

  const std::array<double, 3> jump = sun_dir(26.0 + 4.0 * latch_threshold(0) * 180.0 / Pi + 0.01, 70.0);
  const SunClipmap::Result lagged = clipmap.update({.eye = eye, .dir = jump, .lightId = 4, .texel0 = Texel0});
  EXPECT_TRUE(lagged.relatchAll);
  EXPECT_TRUE(lagged.lagged);
  EXPECT_FALSE(lagged.keyLight);

  const std::array<double, 3> d = clipmap.latches()[0].dir;
  const double push = depth_range(Texel0, 0) / 4.0 + 10.0;
  const std::array<double, 3> raised{eye[0] - d[0] * push, eye[1] - d[1] * push, eye[2] - d[2] * push};
  const SunClipmap::Result recentred = clipmap.update({.eye = raised, .dir = jump, .lightId = 4, .texel0 = Texel0});
  EXPECT_FALSE(recentred.relatchAll);
  EXPECT_EQ(recentred.level, 0);
  EXPECT_TRUE(recentred.recentre);
  EXPECT_TRUE(exempt_frame(recentred, {.coverageComplete = true}));
  clipmap.fill(views);
  EXPECT_NE(views.views[level_view(0)].view_flags() & view_flag::Recentred, 0u);
  EXPECT_FALSE(exempt_frame({}, {.coverageComplete = true}));
  EXPECT_TRUE(exempt_frame({}, {.coverageComplete = false}));
  EXPECT_TRUE(exempt_frame({}, {.cut = true, .coverageComplete = true}));
}

TEST(RenderVsmPages, SelectsKeyBodyOnly) {
  SceneHints hints;
  hints.present = detail::SceneHasLights;
  AuroraLight point{};
  point.id = 1;
  point.type = AURORA_LIGHT_POINT;
  point.shadowPriority = 255;
  hints.lights = {
      point,
      directional(2, sun_dir(-30.0, 0.0), 0, AURORA_LIGHT_NO_SHADOW),
      directional(3, sun_dir(60.0, 0.0), 255, AURORA_LIGHT_CAMERA_FILL),
      directional(4, sun_dir(60.0, 0.0), 255, AURORA_LIGHT_KEY_ROOM),
      directional(5, sun_dir(60.0, 0.0), 255, 0, AURORA_CAT_PLAYER),
      directional(6, sun_dir(45.0, 90.0), 255),
      directional(8, sun_dir(15.0, 90.0), 255),
  };
  const SunSelection sun = select_sun(hints);
  ASSERT_TRUE(sun.found);
  EXPECT_EQ(sun.lightId, 6u);
  EXPECT_EQ(sun.hintIndex, 5u);
  EXPECT_NEAR(sun.elevationDeg, 45.0, 1e-4);
  EXPECT_NEAR(dot(sun.dirWorld, sun.dirWorld), 1.0, 1e-12);

  hints.lights[6].shadowPriority = 254;
  hints.lights.push_back(directional(9, sun_dir(30.0, 90.0), 200));
  EXPECT_EQ(select_sun(hints).lightId, 6u);
  hints.lights.resize(5);
  EXPECT_FALSE(select_sun(hints).found);
  hints.lights.push_back(directional(10, {0.0, 0.0, 0.0}, 255));
  EXPECT_FALSE(select_sun(hints).found);
  hints.present = 0;
  hints.lights.push_back(directional(11, sun_dir(30.0, 90.0), 255));
  EXPECT_FALSE(select_sun(hints).found);
}

TEST(RenderVsmPages, PlanHysteresisAtThreeDegrees) {
  PlanInputs in{.active = true, .chainReady = true, .mainCamera = true};
  in.sun = {.found = true, .lightId = 5, .dirWorld = sun_dir(10.0, 0.0)};
  bool lowSun = false;
  const auto at = [&](double elevation) {
    in.sun.elevationDeg = elevation;
    return decide_plan(in, lowSun);
  };
  EXPECT_FALSE(at(2.9).producing);
  EXPECT_EQ(at(2.99).idle, Idle::LowSun);
  EXPECT_TRUE(at(3.0).producing);
  EXPECT_TRUE(at(2.6).producing);
  EXPECT_TRUE(at(2.5).producing);
  EXPECT_FALSE(at(2.49).producing);
  EXPECT_FALSE(at(2.9).producing);
  EXPECT_TRUE(at(3.1).producing);
  const SunPlan on = at(40.0);
  EXPECT_EQ(on.lightId, 5u);
  EXPECT_EQ(on.idle, Idle::None);

  in.noSun = true;
  EXPECT_EQ(at(40.0).idle, Idle::SunDisabled);
  in.noSun = false;
  in.mainCamera = false;
  EXPECT_EQ(at(40.0).idle, Idle::NoCamera);
  in.mainCamera = true;
  in.chainReady = false;
  EXPECT_EQ(at(40.0).idle, Idle::Pipelines);
  in.active = false;
  EXPECT_EQ(at(40.0).idle, Idle::Inactive);
  in.active = true;
  in.chainReady = true;
  in.sun.found = false;
  EXPECT_EQ(at(40.0).idle, Idle::NoSun);
  in.sun.found = true;
  EXPECT_FALSE(at(2.9).producing);
}

TEST_F(RenderVsmGraph, PlanIsLatchedOnly) {
  testing::set_plan(producing_plan());
  EXPECT_TRUE(plan().producing);
  testing::set_active(true);
  testing::set_pipeline_query([](uint32_t) noexcept { return true; });
  testing::set_compact_ready(true);
  testing::force_consumed(true);
  const StageData stage = producing_stage(1);
  EXPECT_TRUE(frame_producing(stage));
  StageData flushed = producing_stage(1);
  flushed.flushedAtFinish = true;
  EXPECT_FALSE(frame_producing(flushed));
  StageData noPlan = producing_stage(1);
  noPlan.retained = std::make_shared<scene::retained::Frame>();
  EXPECT_FALSE(frame_producing(noPlan));
  StageData noCamera = producing_stage(1);
  noCamera.sceneHints = std::make_shared<SceneHints>();
  EXPECT_FALSE(frame_producing(noCamera));
  testing::set_plan({});
  EXPECT_FALSE(frame_producing(stage));
}

TEST(RenderVsmPages, PlanReusesDirectionWhenLightMissing) {
  const std::array<double, 3> planned = sun_dir(30.0, 0.0);
  const std::array<double, 3> moved = sun_dir(31.0, 0.0);
  const std::shared_ptr<const SceneHints> withLight = make_hints(1, {directional(7, moved, 255)});
  const std::array<double, 3> taken = frame_direction(withLight.get(), 7, planned);
  EXPECT_NEAR(angle(taken, moved), 0.0, 1e-6);
  const std::shared_ptr<const SceneHints> other = make_hints(1, {directional(8, moved, 255)});
  EXPECT_NEAR(angle(frame_direction(other.get(), 7, planned), planned), 0.0, 1e-12);
  const std::shared_ptr<const SceneHints> zero = make_hints(1, {directional(7, {0.0, 0.0, 0.0}, 255)});
  EXPECT_NEAR(angle(frame_direction(zero.get(), 7, planned), planned), 0.0, 1e-12);
  EXPECT_NEAR(angle(frame_direction(nullptr, 7, planned), planned), 0.0, 1e-12);
}

TEST(RenderVsmPages, PoolExtentFollowsPages) {
  const auto check = [](uint32_t pages, uint32_t maxDim, uint32_t outPages, uint32_t width, uint32_t height) {
    const PoolExtent e = pool_extent(pages, maxDim);
    EXPECT_TRUE(e.fits) << pages;
    EXPECT_EQ(e.pages, outPages) << pages;
    EXPECT_EQ(e.width, width) << pages;
    EXPECT_EQ(e.height, height) << pages;
    EXPECT_EQ(e.pagesX * PageTexels, width) << pages;
    EXPECT_GE((e.width / PageTexels) * (e.height / PageTexels), e.pages) << pages;
  };
  check(256, 16384, 256, 4096, 1024);
  check(1024, 16384, 1024, 4096, 4096);
  check(2048, 16384, 2048, 4096, 8192);
  check(4096, 16384, 4096, 8192, 8192);
  check(3000, 16384, 3000, 8192, 6016);
  check(100, 16384, 256, 4096, 1024);
  check(4096, 8192, 4096, 8192, 8192);
  EXPECT_FALSE(pool_extent(4096, 8192).reduced);
  check(4096, 4096, 1024, 4096, 4096);
  EXPECT_TRUE(pool_extent(4096, 4096).reduced);
  check(2048, 4096, 1024, 4096, 4096);
  EXPECT_FALSE(pool_extent(256, 2048).fits);
  EXPECT_FALSE(pool_extent(1024, 0).fits);
}

TEST(RenderVsmPages, EntryBitsRoundTrip) {
  const std::array<uint32_t, 9> masks{
      entry::PhysMask, entry::Fresh,    entry::Allocated, entry::Cached,
      entry::Dirty,    entry::Priority, entry::RenderNow, entry::RenderIndexMask << entry::RenderIndexShift,
      entry::Skip};
  uint32_t all = 0;
  for (const uint32_t mask : masks) {
    EXPECT_EQ(all & mask, 0u) << std::hex << mask;
    all |= mask;
  }
  EXPECT_EQ(MaxPoolPages - 1, entry::PhysMask);
  for (const uint32_t phys : {0u, 1u, 2047u, 4095u}) {
    for (const uint32_t index : {0u, 17u, 4095u}) {
      const uint32_t e =
          phys | entry::Allocated | entry::Dirty | entry::RenderNow | index << entry::RenderIndexShift | entry::Skip;
      EXPECT_EQ(e & entry::PhysMask, phys);
      EXPECT_EQ((e >> entry::RenderIndexShift) & entry::RenderIndexMask, index);
      EXPECT_NE(e & entry::Skip, 0u);
      EXPECT_EQ(e & entry::Cached, 0u);
    }
  }
}

TEST(RenderVsmPages, ZeroedPhysIsFree) {
  Views views = clipmap_views({0.0, 0.0, 0.0}, sun_dir(50.0, 0.0));
  views.header.poolPages = 1024;
  std::vector<uint32_t> table(MaxVirtualPages, 0);
  std::vector<uint32_t> phys(MaxPoolPages * PhysWords, 0);
  const auto words = [&] {
    std::vector<uint32_t> out(ViewsBytes / 4);
    std::memcpy(out.data(), &views, ViewsBytes);
    return out;
  };
  EXPECT_EQ(check::verify(views, words(), table, phys), "");
  const UpdateStats none = apply_update(views, table, phys);
  EXPECT_EQ(none.freedWrap + none.freedRelatch + none.freedRecentre + none.freedFull, 0u);

  table[level_view(0) * PagesPerView + 5] = 9 | entry::Allocated | entry::Cached;
  phys[9 * PhysWords] = level_view(0) * PagesPerView + 5 + 1;
  phys[9 * PhysWords + 1] = 4;
  EXPECT_EQ(check::verify(views, words(), table, phys), "");
  phys[9 * PhysWords] = 7;
  EXPECT_NE(check::verify(views, words(), table, phys), "");
  phys[9 * PhysWords] = level_view(0) * PagesPerView + 5 + 1;
  table[3] = 1 | entry::Allocated;
  EXPECT_NE(check::verify(views, words(), table, phys), "");
  table[3] = 0;
  table[12 * PagesPerView] = 2 | entry::Allocated;
  phys[2 * PhysWords] = 12 * PagesPerView + 1;
  EXPECT_NE(check::verify(views, words(), table, phys), "");
  table[12 * PagesPerView] = 0;
  phys[2 * PhysWords] = 0;
  std::vector<uint32_t> wrong = words();
  wrong[7] ^= 1u;
  EXPECT_NE(check::verify(views, wrong, table, phys), "");

  views.header.flags |= header_flag::Full;
  const UpdateStats full = apply_update(views, table, phys);
  EXPECT_EQ(full.freedFull, 1u);
  EXPECT_TRUE(std::ranges::all_of(table, [](uint32_t e) { return e == 0; }));
  EXPECT_TRUE(std::ranges::all_of(phys, [](uint32_t w) { return w == 0; }));
  EXPECT_EQ(check::verify(views, words(), table, phys), "");
}

TEST(RenderVsmPages, UpdateTwinFreesByViewFlags) {
  SunClipmap clipmap;
  std::array<double, 3> eye{1000.0, 200.0, 3000.0};
  const std::array<double, 3> dir = sun_dir(40.0, 15.0);
  (void)clipmap.update({.eye = eye, .dir = dir, .lightId = 1, .texel0 = Texel0});
  std::vector<uint32_t> table(MaxVirtualPages, 0);
  std::vector<uint32_t> phys(MaxPoolPages * PhysWords, 0);
  uint32_t next = 0;
  for (uint32_t v = 0; v < SunLevels * PagesPerView; v += 4) {
    table[v] = next | entry::Allocated | entry::Cached;
    phys[next * PhysWords] = v + 1;
    phys[next * PhysWords + 1] = 3;
    ++next;
  }
  eye[0] += 200.0;
  eye[2] -= 120.0;
  (void)clipmap.update({.eye = eye, .dir = dir, .lightId = 1, .texel0 = Texel0});
  Views views{};
  clipmap.fill(views);
  views.header.poolPages = MaxPoolPages;
  const std::vector<uint32_t> before = table;
  const UpdateStats stats = apply_update(views, table, phys);
  uint32_t freed = 0;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    if (before[v] == 0) {
      continue;
    }
    const ViewRecord& view = views.views[v / PagesPerView];
    const uint32_t local = v % PagesPerView;
    const bool keep = (view.view_flags() & view_flag::Wrap) == 0 ||
                      kept_after_shift(local % WindowPages, local / WindowPages,
                                       {view.prevOrigin[0], view.prevOrigin[1]}, {view.origin[0], view.origin[1]});
    EXPECT_EQ(table[v], keep ? before[v] : 0u) << v;
    EXPECT_EQ(phys[(before[v] & entry::PhysMask) * PhysWords], keep ? v + 1 : 0u) << v;
    freed += keep ? 0 : 1;
  }
  EXPECT_GT(stats.freedWrap, 0u);
  EXPECT_EQ(stats.freedWrap, freed);
  EXPECT_EQ(stats.freedRelatch + stats.freedRecentre + stats.freedFull, 0u);
}

TEST(RenderVsmCheck, ParsesEnvForms) {
  const std::optional<check::Config> plain = check::parse("1");
  ASSERT_TRUE(plain);
  EXPECT_TRUE(plain->on);
  EXPECT_EQ(plain->first, 30u);
  EXPECT_EQ(plain->last, 300u);
  EXPECT_FALSE(plain->full);
  const std::optional<check::Config> range = check::parse("1,10,20");
  ASSERT_TRUE(range);
  EXPECT_EQ(range->first, 10u);
  EXPECT_EQ(range->last, 20u);
  const std::optional<check::Config> full = check::parse("1,10,200,full=90");
  ASSERT_TRUE(full);
  EXPECT_EQ(full->full, 90u);
  for (const std::string_view bad :
       {"0", "2", "1,", "1,5", "1,9,3", "1,a,9", "1,1,9,full", "1,1,9,full=x", "1,1,9,9"}) {
    EXPECT_FALSE(check::parse(bad)) << bad;
  }
}

TEST_F(RenderVsmGraph, ChainFitsStorageLimit) {
  initialize();
  const std::array scene{detail::create_persistent("scene.instances", detail::PersistentKind::Buffer),
                         detail::create_persistent("scene.materials", detail::PersistentKind::Buffer)};
  testing::set_scene_handles(scene);
  const std::array geometry{detail::create_persistent("scene.indices", detail::PersistentKind::Buffer),
                            detail::create_persistent("scene.vertices", detail::PersistentKind::Buffer),
                            detail::create_persistent("scene.clusters", detail::PersistentKind::Buffer)};
  testing::set_geometry_handles(geometry);
  testing::set_active(true);
  check::testing::set_config(check::Config{.on = true});
  Caps caps = test::full_caps();
  caps.limits.maxComputeWorkgroupSizeX = 256;
  caps.limits.maxComputeWorkgroupsPerDimension = 65535;
  const Settings settings;
  const std::array<detail::PassDesc, PassCount> passes = pass_descs();
  const auto compile = [&](uint32_t computeStorage, bool retained) {
    detail::GraphInputs inputs{.passes = passes, .settings = &settings, .caps = &caps};
    inputs.env.gxFormats[static_cast<size_t>(detail::GxResource::Depth)] = wgpu::TextureFormat::Depth32Float;
    inputs.env.gxFormats[static_cast<size_t>(detail::GxResource::Normal)] = wgpu::TextureFormat::RGB10A2Unorm;
    for (const detail::GxResource gx : {detail::GxResource::Depth, detail::GxResource::Normal}) {
      inputs.env.gxUsage[static_cast<size_t>(gx)] =
          wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
    }
    inputs.sceneDepthSource = DepthSource::Direct;
    inputs.env.limits.compute.storageBuffers = computeStorage;
    inputs.env.efbSize = {1280, 720};
    inputs.env.outputSize = {1280, 720};
    inputs.env.sceneSize = {1280, 720};
    inputs.env.retained = retained;
    return detail::compile_graph(inputs);
  };
  for (const uint32_t limit : {10u, MaxPassStorageBuffers}) {
    const auto graph = compile(limit, true);
    ASSERT_NE(graph, nullptr);
    uint32_t nodes = 0;
    for (const detail::CompiledNode& node : graph->nodes) {
      if (node.kind != detail::NodeKind::Pass || !node.name.starts_with("vsm.")) {
        continue;
      }
      ++nodes;
      EXPECT_EQ(node.status, detail::CullReason::None) << node.name << " at " << limit;
      EXPECT_LE(node.bindings[2].storageBuffers, MaxPassStorageBuffers) << node.name;
      EXPECT_EQ(node.stage, Stage::OpaqueDone) << node.name;
    }
    EXPECT_EQ(nodes, PassCount);
    const auto storage = [&](std::string_view name) {
      const detail::NodeId node = graph->find_node(name);
      EXPECT_NE(node, detail::InvalidNode) << name;
      return node != detail::InvalidNode ? graph->nodes[node].bindings[2].storageBuffers : 0u;
    };
    EXPECT_EQ(storage(UpdatePass), 4u);
    EXPECT_EQ(storage(ScanPass), 8u);
    EXPECT_EQ(storage(DirtyPass), 6u);
    EXPECT_EQ(storage(MarkPass), 3u);
    EXPECT_EQ(storage(ClassifyPass), 7u);
    EXPECT_EQ(storage(AssignPass), 8u);
    EXPECT_EQ(storage(CommitPass), 3u);
    EXPECT_EQ(storage(ChunkBoundsPass), 6u);
    EXPECT_EQ(storage(CullCandidatesPass), 5u);
    EXPECT_EQ(storage(CullCountPass), 7u);
    EXPECT_EQ(storage(CullSkipPass), 8u);
    EXPECT_EQ(storage(CullEmitPass), 8u);
    for (const std::string_view name : {AllocListsPass, ClearPass, CullListPass, CullScanPass}) {
      const detail::NodeId node = graph->find_node(name);
      ASSERT_NE(node, detail::InvalidNode) << name;
      EXPECT_TRUE(graph->nodes[node].manualBindings) << name;
    }
    EXPECT_TRUE(graph->nodes[graph->find_node(ClearPass)].sink);
    EXPECT_EQ(graph->nodes[graph->find_node(ClearPass)].passKind, PassKind::Encoder);
    const auto& order = graph->stages[static_cast<size_t>(Stage::OpaqueDone)].passes;
    const auto position = [&](std::string_view name) {
      return std::ranges::find(order, graph->find_node(name)) - order.begin();
    };
    EXPECT_LT(position(DirtyPass), position(MarkPass));
    EXPECT_LT(position(MarkPass), position(ClassifyPass));
    EXPECT_LT(position(ClassifyPass), position(AllocListsPass));
    EXPECT_LT(position(AllocListsPass), position(AssignPass));
    EXPECT_LT(position(ScanListPass), position(ChunkBoundsPass));
    EXPECT_LT(position(ChunkBoundsPass), position(DirtyPass));
    EXPECT_LT(position(AssignPass), position(ClearPass));
    EXPECT_LT(position(ClearPass), position(CommitPass));
    EXPECT_LT(position(CommitPass), position(CheckPass));
    const detail::NodeId update = graph->find_node(UpdatePass);
    ASSERT_NE(update, detail::InvalidNode);
    EXPECT_TRUE(graph->nodes[update].sink);
    EXPECT_TRUE(graph->nodes[graph->find_node(ScanPass)].sink);
    EXPECT_TRUE(graph->nodes[graph->find_node(DirtyPass)].sink);
    const detail::NodeId list = graph->find_node(ScanListPass);
    ASSERT_NE(list, detail::InvalidNode);
    EXPECT_TRUE(graph->nodes[list].manualBindings);
    for (const std::string_view name : {Table, Phys, ViewsName, StatsName, History, CastBits}) {
      const detail::ResourceId id = graph->find_resource(name);
      ASSERT_NE(id, detail::InvalidResource) << name;
      EXPECT_EQ(graph->resources[id].kind, detail::ResourceKind::Persistent) << name;
    }
  }
  const auto off = compile(10, false);
  const detail::NodeId skipped = off->find_node(UpdatePass);
  ASSERT_NE(skipped, detail::InvalidNode);
  EXPECT_NE(off->nodes[skipped].status, detail::CullReason::None);
  testing::set_active(false);
  const auto inactive = compile(10, true);
  EXPECT_NE(inactive->nodes[inactive->find_node(UpdatePass)].status, detail::CullReason::None);
  EXPECT_NE(inactive->nodes[inactive->find_node(ScanPass)].status, detail::CullReason::None);
  shutdown();
  for (detail::PersistentHandle h : scene) {
    detail::destroy_persistent(h);
  }
  for (detail::PersistentHandle h : geometry) {
    detail::destroy_persistent(h);
  }
  testing::set_scene_handles(std::nullopt);
  testing::set_geometry_handles(std::nullopt);
}

TEST_F(RenderVsmGraph, CullRunsBeforeClear) {
  initialize();
  const std::array handles{detail::create_persistent("scene.instances", detail::PersistentKind::Buffer),
                           detail::create_persistent("scene.materials", detail::PersistentKind::Buffer),
                           detail::create_persistent("scene.indices", detail::PersistentKind::Buffer),
                           detail::create_persistent("scene.vertices", detail::PersistentKind::Buffer),
                           detail::create_persistent("scene.clusters", detail::PersistentKind::Buffer)};
  testing::set_scene_handles(std::array{handles[0], handles[1]});
  testing::set_geometry_handles(std::array{handles[2], handles[3], handles[4]});
  testing::set_active(true);
  Caps caps = test::full_caps();
  caps.limits.maxComputeWorkgroupSizeX = 256;
  caps.limits.maxComputeWorkgroupsPerDimension = 65535;
  const Settings settings;
  const std::array<detail::PassDesc, PassCount> passes = pass_descs();
  detail::GraphInputs inputs{.passes = passes, .settings = &settings, .caps = &caps};
  for (const detail::GxResource gx : {detail::GxResource::Depth, detail::GxResource::Normal}) {
    inputs.env.gxFormats[static_cast<size_t>(gx)] =
        gx == detail::GxResource::Depth ? wgpu::TextureFormat::Depth32Float : wgpu::TextureFormat::RGB10A2Unorm;
    inputs.env.gxUsage[static_cast<size_t>(gx)] = wgpu::TextureUsage::TextureBinding;
  }
  inputs.sceneDepthSource = DepthSource::Direct;
  inputs.env.efbSize = {1280, 720};
  inputs.env.outputSize = {1280, 720};
  inputs.env.sceneSize = {1280, 720};
  inputs.env.retained = true;
  const auto graph = detail::compile_graph(inputs);
  ASSERT_NE(graph, nullptr);
  const auto& order = graph->stages[static_cast<size_t>(Stage::OpaqueDone)].passes;
  std::vector<std::string_view> names;
  for (const detail::NodeId node : order) {
    if (graph->nodes[node].name.starts_with("vsm.")) {
      names.emplace_back(graph->nodes[node].name);
    }
  }
  const std::vector<std::string_view> want{
      UpdatePass,   ScanPass,       ScanListPass, ChunkBoundsPass,    DirtyPass,    MarkPass,
      ClassifyPass, AllocListsPass, AssignPass,   CullCandidatesPass, CullListPass, CullCountPass,
      CullScanPass, CullSkipPass,   CullEmitPass, ClearPass,          CommitPass};
  EXPECT_EQ(names, want);
  for (const std::string_view name : want) {
    EXPECT_EQ(graph->nodes[graph->find_node(name)].status, detail::CullReason::None) << name;
  }
  shutdown();
  for (detail::PersistentHandle h : handles) {
    detail::destroy_persistent(h);
  }
  testing::set_scene_handles(std::nullopt);
  testing::set_geometry_handles(std::nullopt);
}

bool g_pendingSecond = true;

TEST_F(RenderVsmGraph, ChainIsAllOrNothing) {
  testing::set_active(true);
  testing::set_plan(producing_plan());
  testing::force_consumed(true);
  testing::set_compact_ready(true);
  register_chain_pipeline(shaders::PipelineId{1});
  register_chain_pipeline(shaders::PipelineId{2});
  register_chain_pipeline(shaders::PipelineId{2});
  on_latch({}, nullptr);
  EXPECT_EQ(testing::chain_pipelines().size(), 2u);
  g_pendingSecond = true;
  testing::set_pipeline_query([](uint32_t id) noexcept { return id != 2 || !g_pendingSecond; });
  check::testing::set_config(check::Config{.on = true});
  const std::array<detail::PassDesc, PassCount> passes = pass_descs();
  const StageData frame1 = producing_stage(1);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(chain_enabled(frame1));
    for (const detail::PassDesc& pass : passes) {
      EXPECT_FALSE(pass.enabled(frame1, pass.state)) << pass.name;
    }
    g_pendingSecond = false;
  }
  EXPECT_FALSE(chain_enabled(frame1));
  on_latch({}, nullptr);
  EXPECT_FALSE(chain_ready_latched());
  const StageData frame2 = producing_stage(2);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(chain_enabled(frame2));
    EXPECT_TRUE(passes[0].enabled(frame2, passes[0].state));
    g_pendingSecond = true;
  }
  on_latch({}, nullptr);
  EXPECT_TRUE(chain_ready_latched());
  testing::set_compact_ready(false);
  EXPECT_FALSE(chain_enabled(producing_stage(3)));
}

bool g_termAnswer = false;

TEST_F(RenderVsmGraph, ChainOffWithoutConsumer) {
  testing::set_active(true);
  testing::set_plan(producing_plan());
  testing::force_consumed(false);
  testing::set_compact_ready(true);
  testing::set_pipeline_query([](uint32_t) noexcept { return true; });
  check::testing::set_config(check::Config{.on = true});
  const std::array<detail::PassDesc, PassCount> passes = pass_descs();
  const StageData stage = producing_stage(1, check::DefaultFirst);
  ASSERT_TRUE(check::check_frame(check::DefaultFirst));
  for (const detail::PassDesc& pass : passes) {
    EXPECT_FALSE(pass.enabled(stage, pass.state)) << pass.name;
  }
  g_termAnswer = false;
  register_consumer_term([](const StageData&) { return g_termAnswer; });
  EXPECT_FALSE(consumed(stage));
  g_termAnswer = true;
  EXPECT_TRUE(consumed(stage));
  for (const detail::PassDesc& pass : passes) {
    EXPECT_TRUE(pass.enabled(stage, pass.state)) << pass.name;
  }
  g_termAnswer = false;
  testing::force_consumed(true);
  for (const detail::PassDesc& pass : passes) {
    EXPECT_TRUE(pass.enabled(stage, pass.state)) << pass.name;
  }
  const detail::PassDesc& checkPass = passes[PassCount - 1];
  const StageData window = producing_stage(2, check::DefaultFirst + 1);
  EXPECT_TRUE(checkPass.enabled(window, checkPass.state)) << "the check copies stats on every frame in its window";
  const StageData other = producing_stage(3, check::DefaultLast + 2);
  EXPECT_TRUE(passes[0].enabled(other, passes[0].state));
  EXPECT_FALSE(checkPass.enabled(other, checkPass.state));
}

constexpr uint32_t FullChain = chain_bit::Update | chain_bit::Scan | chain_bit::ScanList | chain_bit::Dirty;

FrameInput frame_input(uint32_t frame) {
  return {
      .frameIndex = frame,
      .hintFrame = frame,
      .eye = {500.0, 120.0, -2300.0},
      .planDir = sun_dir(40.0, 30.0),
      .lightId = 7,
      .texel0 = Texel0,
      .producing = true,
      .coverageComplete = true,
      .pages = 1024,
      .pagesX = 32,
      .budget = 64,
      .generation = 1,
      .chainMask = FullChain,
  };
}

FrameDecision step(const FrameInput& in, bool complete = true) {
  const FrameDecision d = decide_frame(in);
  if (complete) {
    mark_encoded(FullChain);
  }
  return d;
}

class RenderVsmInvalidate : public ::testing::Test {
protected:
  void SetUp() override { testing::reset(); }
  void TearDown() override { testing::reset(); }
};

TEST_F(RenderVsmInvalidate, FullReasons) {
  uint32_t frame = 1;
  EXPECT_EQ(step(frame_input(frame++)).reasons, full_reason::First);
  const FrameDecision steady = step(frame_input(frame++));
  EXPECT_FALSE(steady.full);
  EXPECT_EQ(steady.reasons, 0u);
  EXPECT_EQ(steady.views.header.fullReason, 0u);

  const auto with = [&](FrameSignals signals) {
    FrameInput in = frame_input(frame++);
    in.signals = signals;
    return step(in);
  };
  EXPECT_EQ(with({.retainedReset = true}).reasons, full_reason::RetainedReset);
  EXPECT_EQ(with({.clearInstances = true}).reasons, full_reason::ClearInstances);
  const FrameDecision held = with({.unapplied = true});
  EXPECT_FALSE(held.full);
  EXPECT_TRUE(held.held);
  EXPECT_TRUE(frame_held());
  EXPECT_NE(held.views.header.flags & header_flag::Held, 0u);
  EXPECT_NE(held.views.header.flags & header_flag::Exempt, 0u);
  const FrameDecision landed = step(frame_input(frame++));
  EXPECT_EQ(landed.reasons, full_reason::Unapplied);
  EXPECT_FALSE(frame_held());
  const FrameDecision epoch = with({.sceneEpoch = true});
  EXPECT_NE(epoch.reasons & full_reason::SceneEpoch, 0u);
  EXPECT_EQ(epoch.reasons & ~(full_reason::SceneEpoch | full_reason::Anchor), 0u);
  EXPECT_EQ(step(frame_input(frame++)).reasons, 0u);

  ++frame;
  EXPECT_EQ(step(frame_input(frame++)).reasons, full_reason::Gap);
  (void)step(frame_input(frame++), false);
  EXPECT_EQ(step(frame_input(frame++)).reasons, full_reason::ChainIncomplete);
  FrameInput pool = frame_input(frame++);
  pool.generation = 2;
  EXPECT_EQ(step(pool).reasons, full_reason::PoolGeneration);
  FrameInput light = frame_input(frame++);
  light.generation = 2;
  light.lightId = 8;
  EXPECT_EQ(step(light).reasons, full_reason::KeyLight);
  FrameInput far = frame_input(frame++);
  far.generation = 2;
  far.lightId = 8;
  far.eye[0] += 2.0 * AnchorReach;
  EXPECT_EQ(step(far).reasons, full_reason::Anchor);
  FrameInput quiet = far;
  quiet.frameIndex = quiet.hintFrame = frame++;
  const FrameDecision last = step(quiet);
  EXPECT_FALSE(last.full);
  EXPECT_EQ(last.views.header.clock, 2u);
  const testing::WorkerCounts counts = testing::worker_counts();
  EXPECT_EQ(counts.held, 1u);
  for (const uint32_t bit :
       {full_reason::RetainedReset, full_reason::ClearInstances, full_reason::Unapplied, full_reason::SceneEpoch}) {
    EXPECT_EQ(counts.reasons[static_cast<size_t>(std::countr_zero(bit))], 1u) << bit;
  }
}

TEST_F(RenderVsmInvalidate, BreakThenSteadyFreesOnce) {
  uint32_t frame = 1;
  (void)step(frame_input(frame++));
  (void)step(frame_input(frame++));
  uint32_t fulls = 0;
  FrameInput broken = frame_input(frame++);
  broken.signals.unapplied = true;
  fulls += step(broken).full ? 1 : 0;
  broken = frame_input(frame++);
  broken.signals.unapplied = true;
  fulls += step(broken).full ? 1 : 0;
  FrameInput resync = frame_input(frame++);
  resync.signals.clearInstances = true;
  const FrameDecision first = step(resync);
  EXPECT_TRUE(first.full);
  EXPECT_EQ(first.reasons, full_reason::Unapplied | full_reason::ClearInstances);
  fulls += first.full ? 1 : 0;

  std::vector<uint32_t> table(MaxVirtualPages, 0);
  std::vector<uint32_t> phys(MaxPoolPages * PhysWords, 0);
  (void)apply_update(first.views, table, phys);
  uint32_t next = 0;
  for (uint32_t v = 0; v < SunLevels * PagesPerView && next < MaxPoolPages; v += 3) {
    table[v] = next | entry::Allocated | entry::Cached;
    phys[next * PhysWords] = v + 1;
    ++next;
  }
  UpdateStats freed;
  for (int i = 0; i < 10; ++i) {
    const FrameDecision d = step(frame_input(frame++));
    fulls += d.full ? 1 : 0;
    EXPECT_FALSE(d.exempt) << i;
    const UpdateStats s = apply_update(d.views, table, phys);
    freed.freedWrap += s.freedWrap;
    freed.freedRelatch += s.freedRelatch;
    freed.freedRecentre += s.freedRecentre;
    freed.freedFull += s.freedFull;
  }
  EXPECT_EQ(fulls, 1u);
  EXPECT_EQ(freed.freedWrap + freed.freedRelatch + freed.freedRecentre + freed.freedFull, 0u);
}

TEST(RenderVsmInvalidateRules, CasterPredicateExcludesClassesAndCategories) {
  namespace class_bit = scene::retained::class_bit;
  namespace flag = scene::streamout::instance_flag;
  const std::array<std::pair<uint32_t, uint32_t>, 5> classes{{
      {0, 0},
      {class_bit::CpuDeformed, flag::CpuDeformed},
      {class_bit::Billboard, flag::Billboard},
      {class_bit::Water, flag::Water},
      {class_bit::Blended, flag::Blended},
  }};
  const std::array<uint32_t, 10> categories{AURORA_CAT_DYNAMIC,     AURORA_CAT_STATIC | AURORA_CAT_TERRAIN,
                                            AURORA_CAT_PLAYER,      AURORA_CAT_NO_SHADOW_CAST,
                                            AURORA_CAT_SKY,         AURORA_CAT_DECAL,
                                            AURORA_CAT_FAKE_SHADOW, AURORA_CAT_UI,
                                            AURORA_CAT_OVERLAY,     AURORA_CAT_PARTICLE};
  for (const auto& [classBit, recordFlag] : classes) {
    for (const uint32_t category : categories) {
      const scene::retained::InstanceInfo info{
          .slot = 3, .classBits = classBit | class_bit::Dynamic, .categories = category};
      constexpr uint32_t Excluded = AURORA_CAT_NO_SHADOW_CAST | AURORA_CAT_SKY | AURORA_CAT_DECAL |
                                    AURORA_CAT_FAKE_SHADOW | AURORA_CAT_UI | AURORA_CAT_OVERLAY | AURORA_CAT_PARTICLE;
      const bool expected = classBit == 0 && (category & Excluded) == 0;
      EXPECT_EQ(is_caster(info), expected) << classBit << " " << category;
      EXPECT_EQ(record_is_caster(flag::Valid | recordFlag | flag::Latched, category), expected)
          << classBit << " " << category;
      EXPECT_FALSE(record_is_caster(recordFlag, category));
    }
  }
  EXPECT_FALSE(is_caster({.categories = AURORA_CAT_DYNAMIC}));
  EXPECT_TRUE(record_is_caster(flag::Valid | flag::GpuArrays | flag::Palette, AURORA_CAT_DYNAMIC));
}
Views alloc_views(uint32_t poolPages = 1024, uint32_t budget = 1024, uint32_t clock = 5) {
  Views views = clipmap_views({500.0, 120.0, -2300.0}, sun_dir(40.0, 30.0));
  views.header.poolPages = poolPages;
  views.header.poolPagesX = 32;
  views.header.budget = budget;
  views.header.clock = clock;
  return views;
}

uint32_t key_bits(const MarkKeys& keys, uint32_t bits) {
  uint32_t n = 0;
  for (uint32_t i = 0; i < keys.count; ++i) {
    n += (keys.keys[i] >> 16) == bits ? 1 : 0;
  }
  return n;
}

std::array<float, 3> corner_point(const ViewRecord& coarse, const ViewRecord& fine, float fineTexels) {
  std::array<double, 3> bx{};
  std::array<double, 3> by{};
  for (size_t i = 0; i < 3; ++i) {
    bx[i] = double{coarse.camRel[i]} * coarse.texel;
    by[i] = double{coarse.camRel[4 + i]} * coarse.texel;
  }
  const double cx = 128.0 * 17 - coarse.camRel[3];
  const double cy = 128.0 * 15 - coarse.camRel[7];
  std::array<float, 3> p{};
  for (size_t i = 0; i < 3; ++i) {
    p[i] = static_cast<float>(coarse.texel * (cx * bx[i] + cy * by[i]) + fineTexels * fine.texel * (bx[i] + by[i]));
  }
  return p;
}

TEST(RenderVsmAlloc, LevelSelectionHonoursBias) {
  Views views = alloc_views();
  const float bias = static_cast<float>(std::log2(2688.0 / 1440.0));
  EXPECT_NEAR(bias, 0.9f, 0.01f);
  const auto f = static_cast<float>(Texel0 * std::exp2(1.4));
  EXPECT_EQ(level_wanted(f, static_cast<float>(Texel0), 0.f), 2u);
  EXPECT_EQ(level_wanted(f, static_cast<float>(Texel0), bias), 3u);
  EXPECT_EQ(level_wanted(f, static_cast<float>(Texel0), -0.5f), 1u);
  const std::array<float, 3> near{10.f, -20.f, 30.f};
  views.header.levelBias = 0.f;
  EXPECT_EQ(select_level(views, f, near, views.header.levelBias), 2u);
  views.header.levelBias = bias;
  EXPECT_EQ(select_level(views, f, near, views.header.levelBias), 3u);
  const MarkKeys keys = mark_keys(views, near, f);
  ASSERT_GE(keys.count, 2u);
  EXPECT_EQ((keys.keys[0] & 0xFFFFu) / PagesPerView, level_view(3));
  EXPECT_EQ(level_wanted(1e9f, static_cast<float>(Texel0), 0.f), SunLevels - 1);
  EXPECT_EQ(level_wanted(1e-9f, static_cast<float>(Texel0), 0.f), 0u);
}

TEST(RenderVsmAlloc, FallbackLevelAlwaysRequested) {
  const Views views = alloc_views();
  std::mt19937 rng{1234};
  std::uniform_real_distribution<float> offset(-3000.f, 3000.f);
  std::uniform_real_distribution<float> scale(-2.f, 9.f);
  uint32_t checked = 0;
  for (int i = 0; i < 4000; ++i) {
    const std::array<float, 3> p{offset(rng), offset(rng) * 0.2f, offset(rng)};
    const auto f = static_cast<float>(Texel0 * std::exp2(scale(rng)));
    const uint32_t level = select_level(views, f, p, views.header.levelBias);
    const MarkKeys keys = mark_keys(views, p, f);
    if (level >= SunLevels) {
      EXPECT_EQ(keys.count, 0u);
      continue;
    }
    const uint32_t pixel = key_bits(keys, request_bit::Pixel);
    const uint32_t fallback = key_bits(keys, request_bit::Fallback);
    EXPECT_GE(pixel, 1u);
    EXPECT_LE(pixel, 4u);
    EXPECT_LE(fallback, 4u);
    if (level + 1 < SunLevels) {
      const std::array<float, 3> c = view_coords(views.views[level_view(level + 1)], p);
      if (c[0] >= 0.f && c[0] < WindowTexels && c[1] >= 0.f && c[1] < WindowTexels) {
        EXPECT_GE(fallback, 1u) << i;
        ++checked;
      }
    } else {
      EXPECT_EQ(fallback, 0u);
    }
    for (uint32_t k = 0; k < keys.count; ++k) {
      const uint32_t view = (keys.keys[k] & 0xFFFFu) / PagesPerView;
      EXPECT_EQ(view, level_view((keys.keys[k] >> 16) == request_bit::Pixel ? level : level + 1));
    }
  }
  EXPECT_GT(checked, 1000u);
}

TEST(RenderVsmAlloc, MarginCoversPageCorners) {
  Views views = alloc_views();
  views.header.levelBias = 0.f;
  for (const uint32_t level : {0u, 3u, 7u}) {
    const ViewRecord& fine = views.views[level_view(level)];
    const ViewRecord& coarse = views.views[level_view(level + 1)];
    const std::array<float, 3> p = corner_point(coarse, fine, 1.f);
    const auto f = static_cast<float>(fine.texel * 0.7);
    ASSERT_EQ(select_level(views, f, p, 0.f), level);
    const MarkKeys keys = mark_keys(views, p, f);
    EXPECT_EQ(key_bits(keys, request_bit::Pixel), 4u) << level;
    EXPECT_EQ(key_bits(keys, request_bit::Fallback), 4u) << level;
    const std::array<float, 3> c = view_coords(fine, p);
    EXPECT_NEAR(std::fmod(c[0], 128.f), 1.f, 1e-2f);
    EXPECT_NEAR(std::fmod(c[1], 128.f), 1.f, 1e-2f);
    const std::array<float, 3> inside = corner_point(coarse, fine, 60.f);
    const MarkKeys one = mark_keys(views, inside, f);
    EXPECT_EQ(key_bits(one, request_bit::Pixel), 1u) << level;
  }
}

struct AllocFixture {
  Views views;
  std::vector<uint32_t> requests = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> table = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> phys = std::vector<uint32_t>(size_t{MaxPoolPages} * PhysWords, 0);
  std::vector<uint32_t> list = std::vector<uint32_t>(render_list::Words, 0);
  uint32_t nextPhys = 0;

  void own(uint32_t v, uint32_t bits, uint32_t lastUsed) {
    const uint32_t p = nextPhys++;
    table[v] = p | entry::Allocated | bits;
    phys[size_t{p} * PhysWords] = v + 1;
    phys[size_t{p} * PhysWords + 1] = lastUsed;
  }
  [[nodiscard]] uint32_t listed(uint32_t i) const { return list[render_list::Pages + 2 * i]; }
};

TEST(RenderVsmAlloc, AssignOrderIsPriorityNewReuse) {
  AllocFixture a{.views = alloc_views()};
  const uint32_t base = level_view(2) * PagesPerView;
  for (const uint32_t v : {base + 40, base + 9}) {
    a.own(v, entry::Cached | entry::Dirty | entry::Priority, 5);
    a.requests[v] = request_bit::Pixel;
  }
  for (const uint32_t v : {base + 30, base + 3}) {
    a.own(v, entry::Cached | entry::Dirty, 5);
    a.requests[v] = request_bit::Pixel;
  }
  a.own(base + 50, entry::Cached, 5);
  a.requests[base + 50] = request_bit::Fallback;
  for (const uint32_t v : {base + 20, base + 1, base + 700}) {
    a.requests[v] = request_bit::Pixel;
  }
  a.own(base + 60, entry::Cached | entry::Dirty, 2);
  const AllocCounts c = allocate(a.views, a.requests, a.table, a.phys, a.list);
  EXPECT_EQ(c.priority, 2u);
  EXPECT_EQ(c.fresh, 3u);
  EXPECT_EQ(c.reuse, 2u);
  EXPECT_EQ(c.hits, 1u);
  EXPECT_EQ(c.freedDirty, 1u);
  EXPECT_EQ(c.rendered, 7u);
  const std::array<uint32_t, 7> order{base + 9, base + 40, base + 1, base + 20, base + 700, base + 3, base + 30};
  for (uint32_t i = 0; i < order.size(); ++i) {
    EXPECT_EQ(a.listed(i), order[i]) << i;
    const uint32_t e = a.table[order[i]];
    EXPECT_NE(e & entry::RenderNow, 0u) << i;
    EXPECT_EQ((e >> entry::RenderIndexShift) & entry::RenderIndexMask, i) << i;
    EXPECT_EQ(e & entry::PhysMask, a.list[render_list::Pages + 2 * i + 1]) << i;
    EXPECT_EQ(e & entry::Fresh, i >= 2 && i < 5 ? entry::Fresh : 0u) << i;
  }
  EXPECT_EQ(a.table[base + 60], 0u);
  const std::vector<uint32_t> again = a.list;
  EXPECT_EQ(commit(a.list, a.table), 7u);
  for (const uint32_t v : order) {
    EXPECT_EQ(a.table[v] & (entry::RenderNow | entry::Dirty | entry::Priority | entry::Fresh), 0u) << v;
    EXPECT_NE(a.table[v] & entry::Cached, 0u) << v;
  }
  EXPECT_EQ(a.list, again);
}

TEST(RenderVsmAlloc, BudgetCountsOnlyNonPriority) {
  AllocFixture a{.views = alloc_views(1024, 8)};
  const uint32_t base = level_view(1) * PagesPerView;
  for (uint32_t i = 0; i < 5; ++i) {
    a.own(base + 500 + i, entry::Cached | entry::Dirty | entry::Priority, 5);
    a.requests[base + 500 + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 3; ++i) {
    a.own(base + 600 + i, entry::Cached | entry::Dirty, 5);
    a.requests[base + 600 + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 40; ++i) {
    a.requests[base + i] = request_bit::Pixel;
  }
  const AllocCounts c = allocate(a.views, a.requests, a.table, a.phys, a.list);
  EXPECT_EQ(c.rendered, 13u);
  EXPECT_EQ(c.deferredNew, 32u);
  EXPECT_EQ(c.deferredReuse, 3u);
  EXPECT_EQ(c.playerDeferred, 0u);
  EXPECT_EQ(c.overflowPool, 0u);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(a.listed(i), base + 500 + i);
  }
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(a.listed(5 + i), base + i);
  }
  for (uint32_t i = 8; i < 40; ++i) {
    EXPECT_EQ(a.table[base + i], 0u) << i;
  }
  const AllocBudget held = alloc_budget(5, 40, 3, 900, 8, true);
  EXPECT_EQ(held.count, 0u);
  const AllocBudget exempt = alloc_budget(5, 40, 3, 900, 1024, false);
  EXPECT_EQ(exempt.count, 48u);
  const AllocBudget pool = alloc_budget(0, 40, 0, 25, 1024, false);
  EXPECT_EQ(pool.overflowPool, 15u);
  const AllocBudget capped = alloc_budget(10, 9000, 0, 9000, 9000, false);
  EXPECT_EQ(capped.count, MaxRenderPages);
  EXPECT_EQ(capped.playerDeferred, 0u);
}

TEST(RenderVsmAlloc, EvictionTakesOldestBucketFirst) {
  AllocFixture a{.views = alloc_views(MinPoolPages, 1024, 200)};
  const uint32_t base = level_view(4) * PagesPerView;
  const std::array<uint32_t, 4> ages{0, 1, 8, 64};
  for (uint32_t i = 0; i < MinPoolPages; ++i) {
    a.own(base + i, entry::Cached, 200 - ages[i % 4]);
  }
  for (uint32_t i = 0; i < MinPoolPages; i += 4) {
    a.requests[base + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 200; ++i) {
    a.requests[base + 300 + i] = request_bit::Pixel;
  }
  const std::vector<uint32_t> before = a.table;
  const AllocCounts c = allocate(a.views, a.requests, a.table, a.phys, a.list);
  EXPECT_EQ(c.fresh, 200u);
  EXPECT_EQ(c.evictable, 192u);
  EXPECT_EQ(c.freePages, 0u);
  EXPECT_EQ(c.overflowPool, 8u);
  EXPECT_EQ(c.evicted, 192u);
  uint32_t lastBand = 0;
  uint32_t lastPage = 0;
  for (uint32_t i = 0; i < 192; ++i) {
    const uint32_t page = a.list[render_list::Pages + 2 * i + 1];
    ASSERT_LT(page, MinPoolPages);
    const uint32_t band = page % 4 == 3 ? 1 : page % 4 == 2 ? 2 : 3;
    EXPECT_NE(page % 4, 0u) << "an age-0 page was evicted";
    EXPECT_GE(band, lastBand) << i;
    if (band == lastBand) {
      EXPECT_GT(page, lastPage) << i;
    }
    lastBand = band;
    lastPage = page;
    EXPECT_EQ(a.table[base + page], 0u) << "the evicted owner keeps its entry";
  }
  for (uint32_t i = 192; i < 200; ++i) {
    EXPECT_EQ(a.listed(i), NoPage);
  }
  for (uint32_t i = 0; i < MinPoolPages; i += 4) {
    EXPECT_EQ(a.table[base + i], before[base + i]);
    EXPECT_EQ(a.phys[size_t{i} * PhysWords + 1], 200u);
  }
}

TEST_F(RenderVsmInvalidate, ClockRestartsOnFull) {
  uint32_t frame = 1;
  EXPECT_EQ(step(frame_input(frame++)).views.header.clock, 1u);
  for (uint32_t i = 2; i < 6; ++i) {
    EXPECT_EQ(step(frame_input(frame++)).views.header.clock, i);
  }
  FrameInput reset = frame_input(frame++);
  reset.signals.clearInstances = true;
  const FrameDecision full = step(reset);
  EXPECT_TRUE(full.full);
  EXPECT_EQ(full.views.header.clock, 1u);
  EXPECT_EQ(full.views.header.budget, full.views.header.poolPages);
  EXPECT_EQ(step(frame_input(frame++)).views.header.clock, 2u);
  AllocFixture a{.views = full.views};
  a.views.header.poolPages = 256;
  a.requests[level_view(0) * PagesPerView + 33] = request_bit::Pixel;
  (void)allocate(a.views, a.requests, a.table, a.phys, a.list);
  EXPECT_EQ(a.phys[1], 1u);
}

TEST_F(RenderVsmGraph, CommitNeedsEveryChainBit) {
  FrameInput in = frame_input(1);
  in.chainMask = FullChain | chain_bit::Commit;
  (void)decide_frame(in);
  mark_encoded(FullChain);
  EXPECT_EQ(expected_chain(), FullChain | chain_bit::Commit);
  in = frame_input(2);
  in.chainMask = FullChain | chain_bit::Commit;
  EXPECT_EQ(decide_frame(in).reasons, full_reason::ChainIncomplete);
  mark_encoded(FullChain | chain_bit::Commit);
  in = frame_input(3);
  in.chainMask = FullChain | chain_bit::Commit;
  EXPECT_EQ(decide_frame(in).reasons, 0u);
}
TEST(RenderVsmCull, RectCoversCorners) {
  const Views views = alloc_views();
  const ViewRecord& view = views.views[level_view(2)];
  const TexelRect corner{.lo = {128.f * 16 - 3.f, 128.f * 9 - 2.f}, .hi = {128.f * 16 + 2.f, 128.f * 9 + 3.f}};
  EXPECT_EQ(page_rect(view, corner), (std::array<int32_t, 4>{15, 8, 16, 9}));
  EXPECT_TRUE(rect_holds_centre(corner));
  const TexelRect inside{.lo = {128.f * 16 + 3.f, 128.f * 9 + 3.f}, .hi = {128.f * 16 + 9.f, 128.f * 9 + 9.f}};
  EXPECT_EQ(page_rect(view, inside), (std::array<int32_t, 4>{16, 9, 16, 9}));
  const TexelRect edge{.lo = {-40.f, 4090.f}, .hi = {-1.f, 4200.f}};
  const std::array<int32_t, 4> clamped = page_rect(view, edge);
  EXPECT_GT(clamped[0], clamped[2]);
  EXPECT_EQ(clamped[1], 31);
  EXPECT_EQ(clamped[3], 31);
  const Views overhead = clipmap_views({500.0, 120.0, -2300.0}, {0.0, -1.0, 0.0});
  const ViewRecord& top = overhead.views[level_view(2)];
  std::array<uint32_t, 12> xform{};
  for (uint32_t r = 0; r < 3; ++r) {
    for (uint32_t c = 0; c < 2; ++c) {
      xform[r * 4 + c] = std::bit_cast<uint32_t>(top.anchorRel[c * 4 + r] * top.texel);
    }
  }
  const Box local{.lo = {10.f, 20.f, 0.f}, .hi = {11.f, 21.f, 0.f}};
  const TexelRect t = texel_rect(top, box_through(local, xform));
  EXPECT_NEAR(t.hi[0] - t.lo[0], 1.f, 1e-3f);
  EXPECT_NEAR(t.hi[1] - t.lo[1], 1.f, 1e-3f);
  EXPECT_EQ(row_bits(0, 31), 0xFFFFFFFFu);
  EXPECT_EQ(row_bits(3, 5), 0x38u);
  EXPECT_EQ(row_bits(31, 31), 0x80000000u);
  EXPECT_EQ(row_bits(0, 0), 1u);
}

TEST(RenderVsmCull, TexelCentreSkipIsExact) {
  EXPECT_TRUE(rect_holds_centre({.lo = {10.3f, 4.3f}, .hi = {10.7f, 4.7f}}));
  EXPECT_FALSE(rect_holds_centre({.lo = {10.55f, 4.3f}, .hi = {10.95f, 4.7f}}));
  EXPECT_FALSE(rect_holds_centre({.lo = {10.3f, 4.55f}, .hi = {10.7f, 4.95f}}));
  EXPECT_TRUE(rect_holds_centre({.lo = {10.5f, 4.5f}, .hi = {10.5f, 4.5f}}));
  EXPECT_TRUE(rect_holds_centre({.lo = {10.5005f, 4.3f}, .hi = {10.9f, 4.7f}}));
  EXPECT_FALSE(rect_holds_centre({.lo = {10.502f, 4.3f}, .hi = {10.9f, 4.7f}}));
  EXPECT_TRUE(rect_holds_centre({.lo = {-3.6f, -0.7f}, .hi = {-3.4f, -0.4f}}));
  EXPECT_FALSE(rect_holds_centre({.lo = {-3.4f, -0.7f}, .hi = {-3.0f, -0.4f}}));
  EXPECT_TRUE(rect_holds_centre({.lo = {0.f, 0.f}, .hi = {300.f, 300.f}}));
}

TEST(RenderVsmCull, ChunkCountCoversIndexRange) {
  for (uint32_t triangles = 0; triangles < 2000; ++triangles) {
    const uint32_t n = triangles * 3;
    const uint32_t chunks = chunk_count(n);
    EXPECT_GE(chunks * ChunkTriangles, triangles);
    EXPECT_LT(chunks == 0 ? 0 : (chunks - 1) * ChunkTriangles, std::max(triangles, 1u));
    EXPECT_LE(chunks, (n + IndexGranule - 1) / IndexGranule);
  }
  EXPECT_EQ(chunk_count(372), 1u);
  EXPECT_EQ(chunk_count(375), 2u);
  EXPECT_EQ(chunk_slot(128, 3), 5u);
  EXPECT_EQ(chunk_slot(130, 0), NoChunk);
}

TEST(RenderVsmCull, ChunkBoxSlotsAreUnique) {
  std::mt19937 rng{91};
  std::uniform_int_distribution<uint32_t> size(3, 6000);
  std::vector<bool> used(ChunkBoxCount, false);
  uint32_t first = 0;
  uint32_t instances = 0;
  while (true) {
    const uint32_t n = size(rng) / 3 * 3;
    const uint32_t granules = (n + IndexGranule - 1) / IndexGranule;
    if (first / IndexGranule + granules > ChunkBoxCount) {
      break;
    }
    for (uint32_t c = 0; c < chunk_count(n); ++c) {
      const uint32_t slot = chunk_slot(first, c);
      ASSERT_LT(slot, ChunkBoxCount);
      EXPECT_FALSE(used[slot]) << first << " " << c;
      used[slot] = true;
    }
    first += granules * IndexGranule;
    ++instances;
  }
  EXPECT_GT(instances, 100u);
  const Box bounds{.lo = {-50.f, 3.f, 1000.f}, .hi = {20.f, 90.f, 1400.f}};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  for (int i = 0; i < 500; ++i) {
    Box box;
    for (uint32_t k = 0; k < 3; ++k) {
      const float a = bounds.lo[k] + unit(rng) * (bounds.hi[k] - bounds.lo[k]);
      const float b = bounds.lo[k] + unit(rng) * (bounds.hi[k] - bounds.lo[k]);
      box.lo[k] = std::min(a, b);
      box.hi[k] = std::max(a, b);
    }
    const Box out = chunk_box_unpack(chunk_box_pack(box, bounds), bounds);
    for (uint32_t k = 0; k < 3; ++k) {
      EXPECT_LE(out.lo[k], box.lo[k] + 1e-4f) << i;
      EXPECT_GE(out.hi[k], box.hi[k] - 1e-4f) << i;
      EXPECT_LE(out.hi[k] - out.lo[k],
                box.hi[k] - box.lo[k] + 2.f * (bounds.hi[k] - bounds.lo[k]) / ChunkQuanta + 1e-3f);
    }
  }
}

TEST(RenderVsmCull, PairPackRoundTrips) {
  for (const uint32_t slot : {0u, 1u, 9000u, Slots - 1}) {
    for (const uint32_t ri : {0u, 7u, MaxRenderPages - 1}) {
      for (const uint32_t item : {0u, 262143u, PairChunk | 5u, PairChunk | 65535u}) {
        const Pair p = pair_unpack(pair_pack(item, slot, ri));
        EXPECT_EQ(p.slot, slot);
        EXPECT_EQ(p.renderIndex, ri);
        EXPECT_EQ(p.chunk, (item & PairChunk) != 0);
        EXPECT_EQ(p.item, item & ~PairChunk);
        EXPECT_EQ(pair_key(ri, item, slot) >> 46, ri);
      }
    }
  }
}

TEST(RenderVsmCull, SkipIsADeterministicSuffix) {
  std::mt19937 rng{5};
  std::uniform_int_distribution<uint32_t> count(0, 40);
  std::uniform_int_distribution<uint32_t> empty(0, 3);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<uint32_t> counts(PageCountWords, 0);
    const uint32_t pages = 50 + trial * 7;
    for (uint32_t r = 0; r < pages; ++r) {
      if (empty(rng) != 0) {
        counts[r] = count(rng);
        counts[MaxRenderPages + r] = count(rng) / 4;
      }
    }
    const uint32_t maxPairs = 200 + trial * 31;
    const uint32_t first = skip_first(counts, maxPairs);
    EXPECT_EQ(skip_first(counts, maxPairs), first);
    uint64_t kept = 0;
    uint64_t prefix = 0;
    for (uint32_t r = 0; r < MaxRenderPages; ++r) {
      const uint32_t n = counts[r] + counts[MaxRenderPages + r];
      if (r < first) {
        kept += n;
      } else if (n != 0) {
        EXPECT_GT(prefix + n, maxPairs) << trial << " page " << r;
      }
      prefix += n;
    }
    EXPECT_LE(kept, maxPairs) << trial;
    if (first < MaxRenderPages) {
      EXPECT_NE(counts[first] + counts[MaxRenderPages + first], 0u);
    }
  }
}
} // namespace
} // namespace aurora::render::shadows::vsm

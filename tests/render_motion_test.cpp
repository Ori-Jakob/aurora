#include <gtest/gtest.h>

#include "gfx/frame.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/scene/classifier.hpp"
#include "render/scene/classify.hpp"
#include "render/scene/motion.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace aurora::render::scene {
namespace {
namespace rx = gx::render_ext;
using detail::MotionReason;
using motion::DrawMotion;
using motion::Validity;

constexpr uint32_t SourceMask = rx::MotionSourceMask;

// Lays blocks out the way push_storage_packed lays out frame storage
struct FakeStorage {
  static inline std::vector<uint8_t> bytes;
  static inline size_t capacity = SIZE_MAX;
  static inline uint32_t pushes = 0;

  static void reset() {
    bytes.clear();
    capacity = SIZE_MAX;
    pushes = 0;
  }

  static gfx::Range push(std::span<const uint8_t> data) noexcept {
    ++pushes;
    const size_t begin = std::max<size_t>((bytes.size() + 15) & ~size_t{15}, 16);
    if (begin + data.size() > capacity) {
      return {};
    }
    bytes.resize(begin);
    bytes.insert(bytes.end(), data.begin(), data.end());
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(data.size())};
  }

  template <typename T>
  static T at(uint32_t word) {
    T value{};
    if (size_t{word} * 4 + sizeof(T) <= bytes.size()) {
      std::memcpy(&value, bytes.data() + size_t{word} * 4, sizeof(T));
    }
    return value;
  }
};

// C_MTXPerspective(60, 4:3, 1, 1000), as GXSetProjection receives it
Mat4x4<float> perspective(float scale = 1.f) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f * scale;
  proj.m1[1] = 1.7320508f * scale;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

Mat3x4<float> translation(float x, float y, float z) {
  return {{1.f, 0.f, 0.f, x}, {0.f, 1.f, 0.f, y}, {0.f, 0.f, 1.f, z}};
}

Mat3x4<float> yaw(float radians, float x, float y, float z) {
  const float c = std::cos(radians);
  const float s = std::sin(radians);
  return {{c, 0.f, s, x}, {0.f, 1.f, 0.f, y}, {-s, 0.f, c, z}};
}

gx::hints::Camera camera(const Mat3x4<float>& view = translation(0.f, 0.f, 0.f), uint32_t cutEpoch = 0,
                         float scale = 1.f) {
  const Mat4x4<float> proj = perspective(scale);
  return {.role = AURORA_CAMERA_MAIN,
          .view = view,
          .proj = proj,
          .type = GX_PERSPECTIVE,
          .cutEpoch = cutEpoch,
          .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE)};
}

DrawClass draw_class(uint64_t object, uint16_t ordinal = 0, Window window = Window::Opaque, Role role = Role::Main,
                     uint32_t flags = 0, uint32_t categories = 0) {
  DrawClass cls;
  cls.key.role = role;
  cls.key.window = window;
  cls.key.mainCamera = true;
  cls.key.flags = flags;
  cls.key.categories = categories;
  if (object != 0) {
    cls.key.object = {.id = object, .subId = 0, .generation = 1, .valid = true};
  }
  cls.kind = role == Role::Sky ? Kind::Sky : Kind::MainOpaque;
  cls.ordinal = ordinal;
  return cls;
}

gx::PipelineConfig palette_config() {
  gx::PipelineConfig config{};
  config.shaderConfig.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
  return config;
}

DrawMotion draw(const DrawClass& cls, uint32_t slot = 0, const gx::PipelineConfig& config = {}) {
  gx::DrawData data{};
  data.instanceCount = 1;
  data.immediateData.currentPnMtx = slot;
  return motion::draw_motion(data, config, cls);
}

std::array<float, 12> words_of(const Mat3x4<float>& m) {
  std::array<float, 12> out{};
  std::memcpy(out.data(), &m, sizeof(m));
  return out;
}

class RenderMotionTest : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
    FakeStorage::reset();
    gfx::detail::testing::reset_streaming_request();
    gx_ext::records::testing::set_push(&FakeStorage::push);
    motion::testing::set_push(&FakeStorage::push);
    motion::testing::set_jitter_env(motion::testing::JitterEnv::Unset);
    motion::testing::set_check_range(0, 0);
  }

  void TearDown() override {
    motion::testing::reset();
    gx_ext::records::testing::reset();
    detail::remove_hints();
    detail::testing::reset_hints();
    testing::reset();
    detail::testing::reset();
    gfx::detail::testing::reset_streaming_request();
    gx::g_gxState = {};
  }

  static const gx::hints::Sink& sink() { return *gx::hints::sink(); }

  // What latch_frame does around the motion latch: a new hint frame, then the records frame state
  static void begin_frame(bool taa = false, bool live = true, bool debugId = false) {
    detail::begin_hint_frame();
    Settings settings;
    settings.taa = taa;
    motion::latch(settings, live);
    gx_ext::records::testing::set_active(true, true, debugId);
  }

  // Ends with the GX state on the camera's projection, so draws match the MAIN rows
  static Validity send(const gx::hints::Camera& main) {
    const Validity validity = motion::on_main_camera(main, gx::hints::main_camera_usable(main.gxProj));
    gx::g_gxState.proj = main.gxProj;
    return validity;
  }

  // One frame with a camera and nothing drawn, so the next frame has a previous camera
  static void warm(const gx::hints::Camera& main = camera(), bool taa = false) {
    begin_frame(taa);
    send(main);
  }

  static motion::MotionFrame camera_tail() {
    // The camera record is the last 128-byte block the frame pushed before any draw
    return FakeStorage::at<motion::MotionFrame>(static_cast<uint32_t>(FakeStorage::bytes.size() / 4 - 28));
  }
};

TEST_F(RenderMotionTest, EffectivelyOpaqueMatchesBlended) {
  constexpr std::array Modes{GX_BM_NONE, GX_BM_BLEND, GX_BM_LOGIC, GX_BM_SUBTRACT};
  constexpr std::array Factors{GX_BL_ZERO, GX_BL_ONE, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_BL_DSTALPHA};
  constexpr std::array Ops{GX_LO_CLEAR, GX_LO_COPY, GX_LO_XOR, GX_LO_NOOP};
  size_t opaque = 0;
  for (const auto mode : Modes) {
    for (const auto src : Factors) {
      for (const auto dst : Factors) {
        for (const auto op : Ops) {
          gx::PipelineConfig config{};
          config.blendMode = mode;
          config.blendFacSrc = src;
          config.blendFacDst = dst;
          config.blendOp = op;
          EXPECT_EQ(gx::effectively_opaque(config), !blended(mode, src, dst, op));
          opaque += gx::effectively_opaque(config) ? 1 : 0;
        }
      }
    }
  }
  EXPECT_EQ(opaque, Factors.size() * Factors.size() * Ops.size() + Ops.size() + Factors.size() * Factors.size());
}

TEST_F(RenderMotionTest, InactiveLayoutDoesNothing) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  begin_frame(true, false);
  EXPECT_FALSE(motion::testing::active());
  EXPECT_FALSE(motion::testing::jitter_active());
  EXPECT_EQ(motion::frame_jitter(), (std::array{0.f, 0.f}));
  const Validity validity = send(camera());
  EXPECT_FALSE(validity.valid);
  EXPECT_EQ(validity.reason, MotionReason::Inactive);
  EXPECT_EQ(validity.jitterPx[0], 0.f);
  EXPECT_EQ(draw(draw_class(7)).flags, 0u);
  EXPECT_EQ(FakeStorage::pushes, 0u);
  EXPECT_EQ(motion::previous_camera(), nullptr);

  detail::remove_hints();
  begin_frame(true, true);
  EXPECT_FALSE(motion::testing::active());
  EXPECT_EQ(motion::latest_stats().frames, 0u);
}

TEST_F(RenderMotionTest, SceneHintsCarryMotionValidity) {
  const gx::hints::Camera main = camera();
  begin_frame();
  EXPECT_FALSE(detail::live_scene_hints().motionValid);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::NoCamera);
  sink().camera(sink().userData, main);
  EXPECT_FALSE(detail::live_scene_hints().motionValid);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::NoPrevious);

  begin_frame();
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::NoCamera);
  ASSERT_NE(motion::previous_camera(), nullptr);
  sink().camera(sink().userData, main);
  EXPECT_TRUE(detail::live_scene_hints().motionValid);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::Valid);
  EXPECT_EQ(detail::live_scene_hints().jitterPx[0], 0.f);
  EXPECT_EQ(detail::live_scene_hints().jitterPx[1], 0.f);
  EXPECT_EQ(camera_tail().valid, 1u);

  gx::hints::Camera unusable = main;
  unusable.gxProj.m3[2] = 0.f;
  sink().camera(sink().userData, unusable);
  EXPECT_FALSE(detail::live_scene_hints().motionValid);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::Unusable);
  EXPECT_EQ(camera_tail().valid, 0u);
  sink().stage(sink().userData, Stage::SceneBegin);
  sink().camera(sink().userData, main);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::Unusable);
}

TEST_F(RenderMotionTest, PaletteSnapshotRoundTrips) {
  warm();
  for (uint32_t i = 0; i < gx::MaxPnMtx; ++i) {
    gx::g_gxState.pnMtx[i].pos = translation(static_cast<float>(i), 1.f, 2.f);
  }
  const std::array<float, 12> before3 = words_of(gx::g_gxState.pnMtx[3].pos);
  EXPECT_EQ(draw(draw_class(9), 0, palette_config()).flags & SourceMask, 0u);

  begin_frame();
  send(camera());
  for (uint32_t i = 0; i < gx::MaxPnMtx; ++i) {
    gx::g_gxState.pnMtx[i].pos = translation(-1.f, 0.f, 0.f);
  }
  const DrawMotion motion = draw(draw_class(9), 0, palette_config());
  EXPECT_EQ(motion.flags & SourceMask, rx::MotionSourcePalette);
  EXPECT_NE(motion.flags & rx::RecordFlagPrevProjFrame, 0u);
  ASSERT_NE(motion.prevMtxOff, 0u);
  for (uint32_t i = 0; i < gx::MaxPnMtx; ++i) {
    const auto previous = FakeStorage::at<Mat3x4<float>>(motion.prevMtxOff + i * 12);
    EXPECT_EQ(previous.m0[3], static_cast<float>(i)) << i;
    EXPECT_EQ(previous.m1[3], 1.f) << i;
  }
  EXPECT_EQ(words_of(FakeStorage::at<Mat3x4<float>>(motion.prevMtxOff + 3 * 12)), before3);
}

TEST_F(RenderMotionTest, SingleMatrixUsesCurrentSlot) {
  warm();
  gx::g_gxState.pnMtx[4].pos = translation(4.f, 5.f, 6.f);
  gx::g_gxState.pnMtx[0].pos = translation(9.f, 9.f, 9.f);
  draw(draw_class(3), 4);

  begin_frame();
  send(camera());
  const DrawMotion motion = draw(draw_class(3), 4);
  EXPECT_EQ(motion.flags & SourceMask, rx::MotionSourceObject);
  EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(motion.prevMtxOff).m0[3], 4.f);
  EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(motion.prevMtxOff).m2[3], 6.f);
}

TEST_F(RenderMotionTest, SingleMatrixTexSlotAndBadIndex) {
  warm();
  gx::g_gxState.texMtxs[3] = translation(13.f, 0.f, 0.f);
  draw(draw_class(3), 13);
  draw(draw_class(4), 20);

  begin_frame();
  EXPECT_EQ(motion::latest_stats().frame.badIndex, 1u);
  send(camera());
  const DrawMotion tex = draw(draw_class(3), 13);
  EXPECT_EQ(tex.flags & SourceMask, rx::MotionSourceObject);
  EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(tex.prevMtxOff).m0[3], 13.f);
  const DrawMotion bad = draw(draw_class(4), 20);
  EXPECT_EQ(bad.flags, rx::MotionSourceCamera | rx::RecordFlagPrevProjFrame);
  EXPECT_EQ(bad.prevMtxOff, 0u);
  begin_frame();
  EXPECT_EQ(motion::latest_stats().frame.badIndex, 1u);
  EXPECT_EQ(motion::latest_stats().frame.camera, 1u);
}

TEST_F(RenderMotionTest, OrdinalFallbackFindsMergedLead) {
  warm();
  gx::g_gxState.pnMtx[0].pos = translation(0.f, 0.f, 0.f);
  draw(draw_class(5, 0));
  gx::g_gxState.pnMtx[0].pos = translation(2.f, 0.f, 0.f);
  draw(draw_class(5, 2));
  draw(draw_class(6, 0), 0, palette_config());

  begin_frame();
  send(camera());
  const DrawMotion merged = draw(draw_class(5, 1));
  EXPECT_EQ(merged.flags & SourceMask, rx::MotionSourceObject);
  EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(merged.prevMtxOff).m0[3], 0.f);
  const DrawMotion later = draw(draw_class(5, 5));
  EXPECT_EQ(later.flags & SourceMask, rx::MotionSourceObject);
  EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(later.prevMtxOff).m0[3], 2.f);
  EXPECT_EQ(draw(draw_class(5, 6)).flags & SourceMask, rx::MotionSourceCamera);
  EXPECT_EQ(draw(draw_class(6, 1)).flags & SourceMask, rx::MotionSourceCamera);
  begin_frame();
  const motion::Counts& counts = motion::latest_stats().frame;
  EXPECT_EQ(counts.fallback, 2u);
  EXPECT_EQ(counts.exact, 0u);
  EXPECT_EQ(counts.miss, 2u);
}

TEST_F(RenderMotionTest, DuplicateKeyIsAmbiguous) {
  warm();
  draw(draw_class(8));
  draw(draw_class(8));
  draw(draw_class(9));

  begin_frame();
  EXPECT_EQ(motion::latest_stats().frame.ambiguous, 1u);
  send(camera());
  EXPECT_EQ(draw(draw_class(8)).flags & SourceMask, rx::MotionSourceCamera);
  EXPECT_EQ(draw(draw_class(9)).flags & SourceMask, rx::MotionSourceObject);
}

TEST_F(RenderMotionTest, CutEpochOrMissingCameraInvalidates) {
  warm(camera(translation(0.f, 0.f, 0.f), 0));
  begin_frame();
  draw(draw_class(1));
  Validity validity = send(camera(translation(0.f, 0.f, 0.f), 1));
  EXPECT_FALSE(validity.valid);
  EXPECT_EQ(validity.reason, MotionReason::Cut);
  EXPECT_EQ(draw(draw_class(1)).flags, 0u);

  begin_frame();
  EXPECT_EQ(draw(draw_class(1)).flags, 0u);

  begin_frame();
  EXPECT_EQ(motion::previous_camera(), nullptr);
  EXPECT_EQ(send(camera(translation(0.f, 0.f, 0.f), 1)).reason, MotionReason::NoPrevious);

  sink().sceneEpoch(sink().userData, 4);
  begin_frame();
  EXPECT_TRUE(send(camera(translation(0.f, 0.f, 0.f), 1)).valid);
  begin_frame();
  sink().sceneEpoch(sink().userData, 5);
  validity = send(camera(translation(0.f, 0.f, 0.f), 1));
  EXPECT_EQ(validity.reason, MotionReason::SceneEpoch);
  EXPECT_EQ(draw(draw_class(1)).flags, 0u);
}

TEST_F(RenderMotionTest, TeleportImmobileObjectlessUseCamera) {
  warm();
  for (const uint64_t object : {1u, 2u}) {
    draw(draw_class(object));
  }
  begin_frame();
  send(camera());
  constexpr uint32_t Camera = rx::MotionSourceCamera | rx::RecordFlagPrevProjFrame;
  EXPECT_EQ(draw(draw_class(1, 0, Window::Opaque, Role::Main, object_flag::Teleport)).flags, Camera);
  EXPECT_EQ(draw(draw_class(2, 0, Window::Opaque, Role::Main, object_flag::Immobile)).flags, Camera);
  EXPECT_EQ(draw(draw_class(0)).flags, Camera);
  gx::g_gxState.proj = perspective(2.f);
  gx::g_gxState.proj.m3[2] = -1.f;
  EXPECT_EQ(draw(draw_class(0)).flags, 0u);
  EXPECT_EQ(draw(draw_class(1, 0, Window::Opaque, Role::Main, object_flag::Teleport)).flags, 0u);
}

TEST_F(RenderMotionTest, TeleportStillStores) {
  warm();
  draw(draw_class(1, 0, Window::Opaque, Role::Main, object_flag::Teleport));
  draw(draw_class(2, 0, Window::Opaque, Role::Main, object_flag::Immobile));
  gx::DrawData empty{};
  motion::draw_motion(empty, {}, draw_class(3));

  begin_frame();
  send(camera());
  EXPECT_EQ(draw(draw_class(1)).flags & SourceMask, rx::MotionSourceObject);
  EXPECT_EQ(draw(draw_class(2)).flags & SourceMask, rx::MotionSourceCamera);
  EXPECT_EQ(draw(draw_class(3)).flags & SourceMask, rx::MotionSourceObject);
}

TEST_F(RenderMotionTest, ProjectionRowsRule) {
  const Mat4x4<float> main = camera().gxProj;
  Mat4x4<float> depthRange = main;
  depthRange.m2[2] *= 0.5f;
  EXPECT_TRUE(motion::detail::rows_match(main, main));
  EXPECT_TRUE(motion::detail::rows_match(depthRange, main));
  Mat4x4<float> wider = main;
  wider.m0[0] = std::nextafter(wider.m0[0], 2.f);
  EXPECT_FALSE(motion::detail::rows_match(wider, main));
  Mat4x4<float> signedZero = main;
  signedZero.m1[3] = -0.f;
  EXPECT_FALSE(motion::detail::rows_match(signedZero, main));

  warm();
  draw(draw_class(1));
  draw(draw_class(2));
  begin_frame();
  send(camera());
  gx::g_gxState.proj = depthRange;
  EXPECT_EQ(draw(draw_class(1)).flags, rx::MotionSourceObject | rx::RecordFlagPrevProjFrame);
  gx::g_gxState.proj = wider;
  EXPECT_EQ(draw(draw_class(2)).flags, rx::MotionSourceObject);
  EXPECT_EQ(draw(draw_class(0)).flags, 0u);
}

TEST_F(RenderMotionTest, WindowSkyRoleRules) {
  warm();
  begin_frame();
  send(camera());
  const auto source = [](const DrawClass& cls) { return draw(cls).flags & SourceMask; };
  EXPECT_EQ(source(draw_class(0, 0, Window::Opaque)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(0, 0, Window::Translucent)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(0, 0, Window::PreScene)), 0u);
  EXPECT_EQ(source(draw_class(0, 0, Window::PreScene, Role::Sky)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(0, 0, Window::Opaque, Role::Sky)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(0, 0, Window::Post)), 0u);
  EXPECT_EQ(source(draw_class(0, 0, Window::Opaque, Role::Reflection)), 0u);
  EXPECT_EQ(source(draw_class(0, 0, Window::Opaque, Role::None)), 0u);
  DrawClass offscreen = draw_class(0);
  offscreen.offscreen = true;
  EXPECT_EQ(draw(offscreen).flags, 0u);
  gx::g_gxState.proj = gx::hints::gx_projection(perspective(), GX_ORTHOGRAPHIC);
  EXPECT_EQ(source(draw_class(0)), 0u);

  draw(draw_class(11, 0, Window::Post));
  draw(draw_class(12, 0, Window::Opaque, Role::Reflection));
  draw(draw_class(13, 0, Window::Translucent, Role::Sky));
  begin_frame();
  send(camera());
  EXPECT_EQ(source(draw_class(11)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(12)), rx::MotionSourceCamera);
  EXPECT_EQ(source(draw_class(13, 0, Window::Translucent, Role::Sky)), rx::MotionSourceObject);
}

TEST_F(RenderMotionTest, JitterHaltonAndEnv) {
  const auto expect = [](uint32_t phase, float x, float y) {
    const auto jitter = motion::detail::halton_jitter(phase);
    EXPECT_NEAR(jitter[0], x, 1e-6f) << phase;
    EXPECT_NEAR(jitter[1], y, 1e-6f) << phase;
  };
  expect(1, 0.f, 1.f / 3.f - 0.5f);
  expect(2, -0.25f, 2.f / 3.f - 0.5f);
  expect(3, 0.25f, 1.f / 9.f - 0.5f);
  expect(8, 0.0625f - 0.5f, 8.f / 9.f - 0.5f);
  for (uint32_t phase = 1; phase <= 8; ++phase) {
    const auto jitter = motion::detail::halton_jitter(phase);
    EXPECT_LT(std::abs(jitter[0]), 0.5f);
    EXPECT_LT(std::abs(jitter[1]), 0.5f);
  }
  const auto expected = motion::detail::halton_jitter(gfx::current_frame() % 8 + 1);

  begin_frame(false);
  EXPECT_FALSE(motion::testing::jitter_active());
  EXPECT_EQ(motion::frame_jitter(), (std::array{0.f, 0.f}));
  begin_frame(true);
  EXPECT_TRUE(motion::testing::jitter_active());
  EXPECT_EQ(motion::frame_jitter(), expected);
  motion::testing::set_jitter_env(motion::testing::JitterEnv::Off);
  begin_frame(true);
  EXPECT_FALSE(motion::testing::jitter_active());
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  begin_frame(false);
  EXPECT_TRUE(motion::testing::jitter_active());
  EXPECT_EQ(motion::frame_jitter(), expected);
  begin_frame(false, false);
  EXPECT_FALSE(motion::testing::jitter_active());
}

TEST_F(RenderMotionTest, JitterBitOnlyBeforePrePost) {
  const gx::draw_route::Router* router = gx::draw_route::router();
  ASSERT_NE(router, nullptr);
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  const gx::hints::Camera main = camera();
  const auto pushed = [&]() {
    router->route(router->userData, {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3});
    gx::DrawData data{};
    data.instanceCount = 1;
    router->push(router->userData, data, gx::PipelineConfig{}, gx::ShaderInfo{});
    EXPECT_NE(data.immediateData._pad, 0u);
    return FakeStorage::at<gx_ext::records::DrawRecord>(data.immediateData._pad).flags;
  };
  begin_frame(false, true, true);
  sink().camera(sink().userData, main);
  begin_frame(false, true, true);
  sink().camera(sink().userData, main);
  gx::g_gxState.proj = main.gxProj;
  sink().stage(sink().userData, Stage::SceneBegin);
  sink().objectBegin(sink().userData, {.id = 1, .generation = 1});
  EXPECT_NE(pushed() & rx::RecordFlagJitter, 0u);
  sink().scopePush(sink().userData, {.role = AURORA_CAMERA_UI});
  EXPECT_EQ(pushed() & rx::RecordFlagJitter, 0u);
  sink().scopePop(sink().userData);
  EXPECT_NE(pushed() & rx::RecordFlagJitter, 0u);
  sink().stage(sink().userData, Stage::OpaqueDone);
  EXPECT_NE(pushed() & rx::RecordFlagJitter, 0u);
  sink().stage(sink().userData, Stage::PrePost);
  EXPECT_EQ(pushed() & rx::RecordFlagJitter, 0u);
  sink().objectEnd(sink().userData);
  DrawClass offscreen = draw_class(0);
  offscreen.offscreen = true;
  EXPECT_EQ(draw(offscreen).flags, 0u);
}

TEST_F(RenderMotionTest, ReprojectIdentityForStaticCamera) {
  const Mat3x4<float> view = yaw(0.7f, 3.f, -2.f, 40.f);
  const Mat3x4<float> identity = motion::detail::reproject_matrix(view, view);
  const std::array<float, 12> expected{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f};
  const std::array<float, 12> actual = words_of(identity);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6f) << i;
  }
}

TEST_F(RenderMotionTest, ReprojectMatchesPrevTimesInverse) {
  const Mat3x4<float> prev = yaw(0.3f, 1.f, 2.f, -30.f);
  const Mat3x4<float> cur = yaw(0.35f, 1.5f, 2.f, -31.f);
  const Mat3x4<float> reproject = motion::detail::reproject_matrix(prev, cur);
  const auto apply = [](const Mat3x4<float>& m, const std::array<float, 3>& p) {
    std::array<float, 3> out{};
    const Vec4<float>* rows[3] = {&m.m0, &m.m1, &m.m2};
    for (int r = 0; r < 3; ++r) {
      out[r] = (*rows[r])[0] * p[0] + (*rows[r])[1] * p[1] + (*rows[r])[2] * p[2] + (*rows[r])[3];
    }
    return out;
  };
  for (const std::array<float, 3> world : {std::array{0.f, 0.f, 0.f}, std::array{10.f, -3.f, 25.f}}) {
    const auto viaReproject = apply(reproject, apply(cur, world));
    const auto direct = apply(prev, world);
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(viaReproject[i], direct[i], 1e-4f);
    }
  }

  warm(camera(prev));
  begin_frame();
  send(camera(cur));
  const motion::MotionFrame tail = camera_tail();
  EXPECT_EQ(words_of(tail.reproject), words_of(reproject));
  const Mat4x4<float> prevProj = camera(prev).gxProj;
  EXPECT_TRUE(tail.prevProj[0] == prevProj.m0);
  EXPECT_TRUE(tail.prevProj[1] == prevProj.m1);
  EXPECT_TRUE(tail.prevProj[2] == prevProj.m3);
}

TEST_F(RenderMotionTest, PrevArenaPushedOncePerFrame) {
  warm();
  for (const uint64_t object : {1u, 2u, 3u}) {
    gx::g_gxState.pnMtx[0].pos = translation(static_cast<float>(object), 0.f, 0.f);
    draw(draw_class(object));
  }
  begin_frame();
  send(camera());
  const uint32_t pushes = FakeStorage::pushes;
  std::array<uint32_t, 3> offsets{};
  for (uint32_t i = 0; i < 3; ++i) {
    offsets[i] = draw(draw_class(i + 1)).prevMtxOff;
    EXPECT_EQ(FakeStorage::at<Mat3x4<float>>(offsets[i]).m0[3], static_cast<float>(i + 1));
  }
  EXPECT_EQ(FakeStorage::pushes, pushes + 1);
  EXPECT_EQ(offsets[1] - offsets[0], 12u);
  EXPECT_EQ(offsets[2] - offsets[1], 12u);
  draw(draw_class(0));
  EXPECT_EQ(FakeStorage::pushes, pushes + 1);
}

TEST_F(RenderMotionTest, PushRefusalFallsBack) {
  warm();
  draw(draw_class(1));
  draw(draw_class(2));
  begin_frame();
  send(camera());
  FakeStorage::capacity = FakeStorage::bytes.size();
  constexpr uint32_t Camera = rx::MotionSourceCamera | rx::RecordFlagPrevProjFrame;
  EXPECT_EQ(draw(draw_class(1)).flags, Camera);
  EXPECT_EQ(draw(draw_class(2)).flags, Camera);
  begin_frame();
  EXPECT_EQ(motion::latest_stats().frame.pushRefused, 1u);
  EXPECT_EQ(motion::latest_stats().frame.miss, 2u);
  EXPECT_GE(gfx::detail::testing::requested_streaming_sizes().storage,
            gfx::detail::streaming_sizes().storage + 2 * sizeof(Mat3x4<float>));
}

TEST_F(RenderMotionTest, CameraFrameRefusalDisablesMotion) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  warm();
  draw(draw_class(1));
  begin_frame();
  FakeStorage::capacity = FakeStorage::bytes.size();
  const Validity validity = send(camera());
  EXPECT_FALSE(validity.valid);
  EXPECT_EQ(validity.reason, MotionReason::StorageFull);
  EXPECT_EQ(validity.jitterPx[0], 0.f);
  EXPECT_EQ(validity.jitterPx[1], 0.f);
  EXPECT_EQ(draw(draw_class(1)).flags, 0u);
}

TEST_F(RenderMotionTest, AppliedJitterFollowsFrameDecision) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  const gx::hints::Camera main = camera();
  const auto expect_unjittered = [&](const Validity& validity, MotionReason reason) {
    EXPECT_EQ(validity.reason, reason);
    EXPECT_EQ(validity.jitterPx[0], 0.f);
    EXPECT_EQ(validity.jitterPx[1], 0.f);
    EXPECT_EQ(draw(draw_class(0)).flags & rx::RecordFlagJitter, 0u);
  };

  begin_frame();
  sink().camera(sink().userData, main);
  gx::g_gxState.proj = main.gxProj;
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::NoPrevious);
  EXPECT_EQ(detail::live_scene_hints().jitterPx[0], 0.f);
  EXPECT_EQ(camera_tail().jitterPx[0], 0.f);
  EXPECT_EQ(camera_tail().jitterPx[1], 0.f);
  EXPECT_EQ(draw(draw_class(0)).flags & rx::RecordFlagJitter, 0u);

  begin_frame();
  sink().camera(sink().userData, main);
  const auto jitter = motion::frame_jitter();
  ASSERT_NE(jitter[1], 0.f);
  EXPECT_EQ(detail::live_scene_hints().motionReason, MotionReason::Valid);
  EXPECT_EQ(detail::live_scene_hints().jitterPx[0], jitter[0]);
  EXPECT_EQ(detail::live_scene_hints().jitterPx[1], jitter[1]);
  EXPECT_EQ(camera_tail().jitterPx[0], jitter[0]);
  EXPECT_EQ(camera_tail().jitterPx[1], jitter[1]);
  EXPECT_NE(draw(draw_class(0)).flags & rx::RecordFlagJitter, 0u);

  begin_frame();
  expect_unjittered(send(camera(translation(0.f, 0.f, 0.f), 1)), MotionReason::Cut);
  EXPECT_EQ(camera_tail().jitterPx[1], 0.f);
  begin_frame();
  sink().sceneEpoch(sink().userData, 1);
  EXPECT_TRUE(send(camera(translation(0.f, 0.f, 0.f), 1)).valid);
  begin_frame();
  sink().sceneEpoch(sink().userData, 2);
  expect_unjittered(send(camera(translation(0.f, 0.f, 0.f), 1)), MotionReason::SceneEpoch);
  begin_frame();
  gx::hints::Camera unusable = main;
  unusable.gxProj.m0[0] = 0.f;
  expect_unjittered(send(unusable), MotionReason::Unusable);
  warm(main);
  begin_frame();
  FakeStorage::capacity = FakeStorage::bytes.size();
  expect_unjittered(send(main), MotionReason::StorageFull);
}

TEST_F(RenderMotionTest, NoFifoAllocation) {
  warm();
  const uint64_t fingerprint = motion::testing::storage_fingerprint();
  for (uint32_t frame = 0; frame < 3; ++frame) {
    begin_frame();
    send(camera());
    for (uint32_t i = 0; i < 2000; ++i) {
      draw(draw_class(i + 1, 0), 0, palette_config());
    }
    EXPECT_EQ(motion::testing::storage_fingerprint(), fingerprint);
  }
  begin_frame();
  EXPECT_GT(motion::latest_stats().frame.overflow, 0u);
  EXPECT_EQ(motion::testing::storage_fingerprint(), fingerprint);
}

TEST_F(RenderMotionTest, ClassifierOrdinals) {
  Classifier classifier;
  classifier.reserve();
  classifier.begin_frame();
  const DrawInputs inputs{.perspective = true};
  EXPECT_EQ(classifier.route(inputs).ordinal, 0u);
  classifier.object_begin({.id = 1});
  EXPECT_EQ(classifier.route(inputs).ordinal, 0u);
  EXPECT_EQ(classifier.route(inputs).ordinal, 1u);
  DrawInputs offscreen = inputs;
  offscreen.offscreen = true;
  EXPECT_EQ(classifier.route(offscreen).ordinal, 0u);
  EXPECT_EQ(classifier.route(inputs).ordinal, 2u);
  classifier.object_end();
  EXPECT_EQ(classifier.route(inputs).ordinal, 0u);
  classifier.object_begin({.id = 2, .flags = object_flag::CaptureOnly});
  EXPECT_TRUE(classifier.route(inputs).suppressed);
  EXPECT_EQ(classifier.last().ordinal, 0u);
  classifier.object_begin({.id = 3});
  EXPECT_EQ(classifier.route(inputs).ordinal, 0u);
  EXPECT_EQ(classifier.route(inputs).ordinal, 1u);
  classifier.begin_frame();
  classifier.object_begin({.id = 3});
  EXPECT_EQ(classifier.route(inputs).ordinal, 0u);
}

TEST_F(RenderMotionTest, CheckVerdictNeedsPlayerSamples) {
  motion::Counts passing;
  passing.player[0] = {.eligible = 90, .exact = 89, .fallback = 1};
  passing.player[1] = {.eligible = 10, .exact = 10};
  passing.framesInWindow = 100;
  passing.framesWithoutPlayer = 5;
  passing.immobileSamples = 4;
  passing.immobileMaxPx = 0.04f;
  EXPECT_FALSE(motion::detail::check_failure(passing).has_value());
  EXPECT_TRUE(motion::detail::check_failure(motion::Counts{}).has_value());
  const auto fails = [&](auto&& edit) {
    motion::Counts counts = passing;
    edit(counts);
    return motion::detail::check_failure(counts).has_value();
  };
  EXPECT_TRUE(fails([](motion::Counts& c) { c.player[0].exact = 87; }));
  EXPECT_TRUE(fails([](motion::Counts& c) { c.player[0] = {.eligible = 90, .exact = 87, .fallback = 3}; }));
  EXPECT_TRUE(fails([](motion::Counts& c) { c.framesWithoutPlayer = 6; }));
  EXPECT_TRUE(fails([](motion::Counts& c) { c.immobileSamples = 0; }));
  EXPECT_TRUE(fails([](motion::Counts& c) { c.immobileMaxPx = 0.06f; }));
  EXPECT_TRUE(fails([](motion::Counts& c) { c.pushRefused = 1; }));

  motion::testing::set_check_range(1, 5);
  warm();
  draw(draw_class(1, 0, Window::Opaque, Role::Main, 0, category::Player));
  draw(draw_class(2, 0, Window::Opaque, Role::Main, object_flag::CpuDeformed, category::Player));
  begin_frame();
  send(camera());
  draw(draw_class(1, 0, Window::Opaque, Role::Main, 0, category::Player));
  draw(draw_class(2, 0, Window::Opaque, Role::Main, object_flag::CpuDeformed, category::Player));
  draw(draw_class(3, 0, Window::Opaque, Role::Main, 0, category::Player), 0, palette_config());
  begin_frame();
  const motion::Counts& check = motion::latest_stats().check;
  EXPECT_EQ(check.framesInWindow, 1u);
  EXPECT_EQ(check.player[0].eligible, 1u);
  EXPECT_EQ(check.player[0].exact, 1u);
  EXPECT_EQ(check.player[2].exact, 1u);
  EXPECT_EQ(check.player[1].eligible, 1u);
  EXPECT_EQ(check.player[1].miss, 1u);
  EXPECT_EQ(motion::latest_stats().total.framesInWindow, 1u);
}

TEST_F(RenderMotionTest, ImmobileResidualUsesOnScreenPoint) {
  const Mat4x4<float> proj = camera().gxProj;
  const Mat3x4<float> curView = yaw(0.2f, 0.f, -1.f, -5.f);
  const Mat3x4<float> prevView = yaw(0.18f, 0.3f, -1.f, -5.2f);
  const Mat3x4<float> model = translation(4.f, 0.f, -20.f);
  const auto concat = [](const Mat3x4<float>& a, const Mat3x4<float>& b) {
    Mat3x4<float> out{};
    const Vec4<float>* ar[3] = {&a.m0, &a.m1, &a.m2};
    const Vec4<float>* br[3] = {&b.m0, &b.m1, &b.m2};
    Vec4<float>* orows[3] = {&out.m0, &out.m1, &out.m2};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        (*orows[r])[c] = (*ar[r])[0] * (*br[0])[c] + (*ar[r])[1] * (*br[1])[c] + (*ar[r])[2] * (*br[2])[c] +
                         (c == 3 ? (*ar[r])[3] : 0.f);
      }
    }
    return out;
  };
  const Mat3x4<float> reproject = motion::detail::reproject_matrix(prevView, curView);
  const auto residual = [&](const Mat3x4<float>& prevModel, const Mat4x4<float>& curProj) {
    return motion::detail::immobile_residual_px(concat(curView, model), concat(prevView, prevModel), reproject, curProj,
                                                proj, 640.f, 480.f);
  };
  const auto still = residual(model, proj);
  ASSERT_TRUE(still.has_value());
  EXPECT_LT(*still, 1e-3f);

  // A one-unit view-space shift at depth sqrt(near * far) moves p00 / depth in NDC
  const Mat3x4<float> shift = concat(translation(1.f, 0.f, 0.f), curView);
  const auto shifted = motion::detail::immobile_residual_px(concat(curView, model), concat(shift, model),
                                                            translation(0.f, 0.f, 0.f), proj, proj, 640.f, 480.f);
  ASSERT_TRUE(shifted.has_value());
  EXPECT_NEAR(*shifted, 1.2990381f / std::sqrt(1000.f) * 320.f, 0.01f);

  Mat4x4<float> infinite = proj;
  infinite.m2[2] = 0.f;
  infinite.m2[3] = -1.f;
  const auto far = motion::detail::immobile_residual_px(concat(curView, model), concat(shift, model),
                                                        translation(0.f, 0.f, 0.f), infinite, proj, 640.f, 480.f);
  ASSERT_TRUE(far.has_value());
  EXPECT_NEAR(*far, 1.2990381f / 64.f * 320.f, 0.01f);

  const Mat3x4<float> behind = concat(translation(0.f, 0.f, 100.f), curView);
  EXPECT_FALSE(motion::detail::immobile_residual_px(concat(curView, model), concat(behind, model),
                                                    translation(0.f, 0.f, 0.f), proj, proj, 640.f, 480.f)
                   .has_value());

  motion::testing::set_check_range(0, 10);
  gx::g_gxState.renderViewport = {0.f, 0.f, 640.f, 480.f, 0.f, 1.f};
  warm(camera(prevView));
  gx::g_gxState.pnMtx[0].pos = concat(prevView, model);
  draw(draw_class(5, 0, Window::Opaque, Role::Main, object_flag::Immobile));
  begin_frame();
  send(camera(curView));
  gx::g_gxState.pnMtx[0].pos = concat(curView, model);
  EXPECT_EQ(draw(draw_class(5, 0, Window::Opaque, Role::Main, object_flag::Immobile)).flags & SourceMask,
            rx::MotionSourceCamera);
  begin_frame();
  EXPECT_EQ(motion::latest_stats().frame.immobileSamples, 1u);
  EXPECT_LT(motion::latest_stats().frame.immobileMaxPx, 1e-3f);
}
} // namespace
} // namespace aurora::render::scene

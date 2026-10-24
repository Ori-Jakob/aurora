#include <gtest/gtest.h>

#include "hint_bytes.hpp"
#include "gx/gpu_channel.hpp"
#include "render/module.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"
#include "webgpu/gpu.hpp"
#if AURORA_ENABLE_RENDER
#include "render/hints.hpp"
#endif

#include <utility>
#include <vector>

namespace aurora::render::test {
namespace {
namespace wire = gx::hints::wire;
using aurora::test::camera_hint;
using aurora::test::environment_hint;
using aurora::test::epoch_hint;
using aurora::test::lights_hint;
using aurora::test::make_hint;
using aurora::test::process_hints;
using aurora::test::stage_hint;

// One valid hint for every declared ID, in ID order
std::vector<std::pair<uint16_t, std::vector<uint8_t>>> one_of_each() {
  using wire::Writer;
  std::vector<std::pair<uint16_t, std::vector<uint8_t>>> out;
  const auto add = [&](uint16_t id, uint16_t version, uint32_t size, auto&& fill) {
    out.emplace_back(id, make_hint(id, version, size, fill));
  };
  add(GX_AURORA_SET_ARRAY_IMMUTABLE, 1, wire::SetArrayImmutableV1, [](Writer& w) {
    w.u64(0x80200000);
    w.u32(0x90);
    w.u8(0);
    w.u32(1);
  });
  add(GX_AURORA_RELEASE_ARRAY_IMMUTABLE, 1, wire::ReleaseArrayImmutableV1, [](Writer& w) {
    w.u64(0x80200000);
    w.u32(1);
  });
  out.emplace_back(GX_AURORA_RENDER_CAMERA, camera_hint(AURORA_CAMERA_MAIN, true));
  add(GX_AURORA_RENDER_OBJECT_BEGIN, 1, wire::ObjectBeginV1, [](Writer& w) {
    w.u64(0x10);
    w.u32(0);
    w.u32(1);
    w.u32(AURORA_CAT_TERRAIN);
    w.u32(0);
  });
  add(GX_AURORA_RENDER_OBJECT_END, 1, wire::EmptyV1, [](Writer&) {});
  add(GX_AURORA_RENDER_SCOPE_PUSH, 1, wire::ScopePushV1, [](Writer& w) {
    w.u32(AURORA_CAT_EMISSIVE);
    w.u32(0);
    w.u8(AURORA_CAMERA_INHERIT);
  });
  add(GX_AURORA_RENDER_SCOPE_POP, 1, wire::EmptyV1, [](Writer&) {});
  out.emplace_back(GX_AURORA_RENDER_STAGE, stage_hint(Stage::OpaqueDone));
  out.emplace_back(GX_AURORA_RENDER_LIGHTS, lights_hint(1));
  out.emplace_back(GX_AURORA_RENDER_ENVIRONMENT, environment_hint());
  add(GX_AURORA_RENDER_MATERIAL, 1, wire::MaterialV1, [](Writer& w) { w.u32(1); });
  add(GX_AURORA_RENDER_SKELETON, 1, wire::SkeletonHeader, [](Writer& w) {
    w.u64(0x10);
    w.u16(0);
  });
  out.emplace_back(GX_AURORA_RENDER_SCENE_EPOCH, epoch_hint(1));
  add(GX_AURORA_RENDER_TEV_POLICY, 1, wire::TevPolicyV1, [](Writer& w) {
    wire::write(w, AuroraTevPolicy{.mask = AURORA_TEV_POLICY_BIT(AURORA_TEV_REG_K1, AURORA_TEV_CH_R)});
  });
  add(GX_AURORA_RENDER_INFLUENCERS, 1, wire::CountedHeader, [](Writer& w) { w.u16(0); });
  add(GX_AURORA_RENDER_WIND_SOURCES, 1, wire::CountedHeader, [](Writer& w) { w.u16(0); });
  add(GX_AURORA_RENDER_GRASS_PATCHES, 1, wire::GrassPatchesHeader, [](Writer& w) {
    w.u32(0);
    w.u32(0);
    w.f32(0.5f);
    w.u8(0);
  });
  add(GX_AURORA_RENDER_SKY, 1, wire::SkyV1, [](Writer& w) { wire::write(w, AuroraSky{.layerCount = 0}); });
  add(GX_AURORA_RENDER_FOG_VOLUMES, 1, wire::CountedHeader, [](Writer& w) { w.u16(0); });
  constexpr uint64_t Handle = 1ull << 32 | 1;
  constexpr uint64_t System = 1ull << 32 | 1;
  add(GX_AURORA_GPU_UPLOAD, 1, wire::GpuUploadHeader + 4, [](Writer& w) {
    w.u64(Handle);
    w.u32(0x01020304);
  });
  add(GX_AURORA_GPU_RESERVE, 1, wire::GpuReserveV1, [](Writer& w) {
    w.u64(Handle + 1);
    w.u32(16);
    w.u32(1);
  });
  add(GX_AURORA_GPU_DISPATCH, 1, wire::EmptyV1, [](Writer&) {});
  add(GX_AURORA_LOAD_ARRAY_GPU, 1, wire::LoadArrayGpuV1, [](Writer& w) {
    w.u8(GX_VA_TEX7);
    w.u64(0);
    w.u32(0);
  });
  add(GX_AURORA_GPU_UPDATE_PERSISTENT, 1, wire::GpuUpdateHeader + 4, [](Writer& w) {
    w.u64(System);
    w.u32(0);
    w.u32(7);
  });
  add(GX_AURORA_GPU_CREATE_SYSTEM, 1, wire::GpuCreateHeader + 1 + wire::GpuCreateKernelHeader + 4 + 1, [](Writer& w) {
    w.u64(System);
    w.u32(4);
    w.u32(0);
    w.u32(0);
    w.u8(1);
    w.u8(1);
    w.u32(1);
    w.u8('s');
    w.u8(4);
    for (const char c : {'m', 'a', 'i', 'n'}) {
      w.u8(static_cast<uint8_t>(c));
    }
    w.u16(64);
    w.u16(1);
    w.u16(1);
    w.u8(0);
    w.u8(';');
  });
  add(GX_AURORA_GPU_DESTROY_SYSTEM, 1, wire::GpuDestroyV1, [](Writer& w) { w.u64(System); });
  return out;
}

class RenderHintsTest : public ::testing::Test {
protected:
  void SetUp() override { reset(); }
  void TearDown() override { reset(); }

  static void reset() {
    gx::hints::testing::reset_counters();
    gx::gpu_channel::reset();
#if AURORA_ENABLE_RENDER
    detail::testing::reset_hints();
#endif
    detail::testing::reset();
  }

#if AURORA_ENABLE_RENDER
  static void install() {
    detail::testing::set_hints_override(1);
    detail::install_hints();
    detail::begin_hint_frame();
  }
#endif
};

#if AURORA_ENABLE_RENDER
TEST_F(RenderHintsTest, InitializeInstallsSinkShutdownRemovesIt) {
  detail::testing::set_caps(full_caps());
  webgpu::g_graphicsConfig.msaaSamples = 1;
  EXPECT_EQ(gx::hints::sink(), nullptr);
  detail::initialize();
  EXPECT_NE(gx::hints::sink(), nullptr);
  EXPECT_TRUE(hints_enabled());
  EXPECT_TRUE(detail::hints_enabled());
  detail::shutdown();
  EXPECT_EQ(gx::hints::sink(), nullptr);
  EXPECT_FALSE(hints_enabled());
}

TEST_F(RenderHintsTest, OverrideDisablesDecoding) {
  detail::testing::set_hints_override(0);
  detail::install_hints();
  EXPECT_EQ(gx::hints::sink(), nullptr);
  EXPECT_FALSE(hints_enabled());
  process_hints(lights_hint(2));
  process_hints(stage_hint(Stage::SceneBegin));
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_LIGHTS).skipped, 1u);
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_STAGE).skipped, 1u);
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_STAGE).decoded, 0u);
}

TEST_F(RenderHintsTest, FrameValuesFreezeAtSceneBegin) {
  install();
  process_hints(lights_hint(2));
  process_hints(stage_hint(Stage::SceneBegin));
  EXPECT_TRUE(detail::scene_hints_frozen());
  process_hints(lights_hint(5));
  const auto hints = detail::scene_hints();
  ASSERT_NE(hints, nullptr);
  EXPECT_EQ(hints->lights.size(), 2u);
  EXPECT_TRUE(hints->has(detail::SceneHasLights));
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_LIGHTS).decoded, 2u);
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_LIGHTS).late, 1u);
}

TEST_F(RenderHintsTest, RejectedSceneBeginDoesNotFreeze) {
  install();
  process_hints(stage_hint(Stage::OpaqueDone));
  process_hints(stage_hint(Stage::SceneBegin));
  process_hints(lights_hint(3));
  EXPECT_FALSE(detail::scene_hints_frozen());
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_LIGHTS).late, 0u);
  EXPECT_EQ(detail::scene_hints()->lights.size(), 3u);
  EXPECT_TRUE(detail::scene_hints_frozen());
}

TEST_F(RenderHintsTest, FreezeWithoutMarkerHappensAtFirstSnapshot) {
  install();
  process_hints(lights_hint(1));
  EXPECT_FALSE(detail::scene_hints_frozen());
  const auto hints = detail::scene_hints();
  EXPECT_TRUE(detail::scene_hints_frozen());
  process_hints(environment_hint());
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_ENVIRONMENT).late, 1u);
  EXPECT_FALSE(hints->has(detail::SceneHasEnvironment));
  EXPECT_FALSE(detail::live_scene_hints().has(detail::SceneHasEnvironment));
}

TEST_F(RenderHintsTest, SnapshotSharedWithinFrame) {
  install();
  process_hints(lights_hint(1));
  const auto first = detail::scene_hints();
  EXPECT_EQ(detail::scene_hints(), first);
  const uint64_t frame = first->hintFrame;
  detail::begin_hint_frame();
  const auto next = detail::scene_hints();
  EXPECT_NE(next, first);
  EXPECT_TRUE(next->lights.empty());
  EXPECT_EQ(next->present, 0u);
  EXPECT_EQ(next->hintFrame, frame + 1);
  EXPECT_EQ(first->lights.size(), 1u);
}

TEST_F(RenderHintsTest, EpochChangeDetection) {
  install();
  std::vector<bool> changed;
  for (const uint32_t epoch : {5u, 5u, 6u}) {
    detail::begin_hint_frame();
    process_hints(epoch_hint(epoch));
    const auto& live = detail::live_scene_hints();
    EXPECT_TRUE(live.has(detail::SceneHasEpoch));
    EXPECT_EQ(live.sceneEpoch, epoch);
    changed.push_back(live.sceneEpochChanged);
  }
  EXPECT_EQ(changed, (std::vector{false, false, true}));
}

TEST_F(RenderHintsTest, ResetClearsStickyState) {
  install();
  process_hints(epoch_hint(5));
  gx::hints::reset_state();
  detail::begin_hint_frame();
  process_hints(epoch_hint(6));
  EXPECT_FALSE(detail::live_scene_hints().sceneEpochChanged);
  detail::begin_hint_frame();
  process_hints(epoch_hint(7));
  EXPECT_TRUE(detail::live_scene_hints().sceneEpochChanged);
}

TEST_F(RenderHintsTest, MainCameraUsableDecides) {
  install();
  process_hints(camera_hint(AURORA_CAMERA_MAIN, true, 4));
  EXPECT_TRUE(detail::live_scene_hints().has(detail::SceneHasMainCamera));
  EXPECT_EQ(detail::live_scene_hints().mainCamera.cutEpoch, 4u);
  process_hints(camera_hint(AURORA_CAMERA_MAIN, false));
  EXPECT_FALSE(detail::live_scene_hints().has(detail::SceneHasMainCamera));
  process_hints(camera_hint(AURORA_CAMERA_MAIN, true));
  process_hints(stage_hint(Stage::SceneBegin));
  process_hints(camera_hint(AURORA_CAMERA_MAIN, false));
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_CAMERA).late, 1u);
  process_hints(camera_hint(AURORA_CAMERA_REFLECTION, true));
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_CAMERA).late, 1u);
  EXPECT_EQ(hint_counters(GX_AURORA_RENDER_CAMERA).decoded, 5u);
  EXPECT_TRUE(detail::scene_hints()->has(detail::SceneHasMainCamera));
}
#endif

TEST_F(RenderHintsTest, EveryIdCountedOnce) {
#if AURORA_ENABLE_RENDER
  install();
#endif
  const auto hints = one_of_each();
  for (const auto& [id, bytes] : hints) {
    process_hints(bytes);
  }
  for (const auto& [id, bytes] : hints) {
    const HintCounters counters = hint_counters(id);
    const bool decodes =
        id == GX_AURORA_SET_ARRAY_IMMUTABLE || id == GX_AURORA_RELEASE_ARRAY_IMMUTABLE ||
        (AURORA_ENABLE_RENDER != 0 && id >= GX_AURORA_RENDER_CAMERA && id <= GX_AURORA_RENDER_WIND_SOURCES) ||
        gx::gpu_channel::is_command(id);
    EXPECT_EQ(counters.decoded, decodes ? 1u : 0u) << std::hex << id;
    EXPECT_EQ(counters.skipped, decodes ? 0u : 1u) << std::hex << id;
    EXPECT_EQ(counters.malformed, 0u) << std::hex << id;
    EXPECT_EQ(counters.bytes, bytes.size()) << std::hex << id;
  }
  EXPECT_EQ(hint_counters(0x0001), HintCounters{});
  EXPECT_EQ(hints_enabled(), AURORA_ENABLE_RENDER != 0);
}
} // namespace
} // namespace aurora::render::test

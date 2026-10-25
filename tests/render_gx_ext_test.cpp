#include <gtest/gtest.h>

#include "gx/gx.hpp"
#include "gx/render_ext.hpp"
#include "gx/texture.hpp"
#include "render/frame_state.hpp"
#include "render/gx_ext/ext_provider.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/material/registry.hpp"
#include "render/scene/classifier.hpp"
#include "webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>
#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace aurora::render::gx_ext {
namespace {
constexpr std::string_view FlaggedKey = "tex1_16x16_000000000000a001_14";
constexpr std::string_view PlainKey = "tex1_16x16_000000000000a004_14";
constexpr std::string_view Flagged =
    R"({"version":1,"materials":{"tex1_16x16_000000000000a001_14":{"flags":["stochastic"]}}})";
constexpr std::string_view Plain =
    R"({"version":1,"materials":{"tex1_16x16_000000000000a004_14":{"class":"ground/grass"}}})";

detail::FrameState frame(bool stochasticTiling, bool textureMaterials) {
  detail::FrameState state;
  state.settings.stochasticTiling = stochasticTiling;
  state.settings.textureMaterials = textureMaterials;
  return state;
}

gx::ShaderInfo eligible(u8 mask) {
  gx::ShaderInfo info;
  info.stochasticEligible = mask;
  return info;
}

gfx::TextureBind repeat_bind() {
  gfx::TextureBind bind;
  bind.ref =
      std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                        wgpu::Extent3D{8, 8, 1}, wgpu::TextureFormat::RGBA8Unorm, 1, GX_TF_RGBA8);
  bind.texObj.mode0 = GX_REPEAT | GX_REPEAT << 2;
  return bind;
}

gfx::TextureMaterialRef material_ref(std::string_view key) {
  const auto snapshot = material::latched_snapshot();
  const uint16_t id = snapshot->find(*texture::parse_material_key(key));
  return {id, snapshot->record(id).summary};
}

// Lays records out the way push_storage_packed lays out frame storage
struct FakeStorage {
  static inline std::vector<uint8_t> bytes;
  static inline size_t capacity = SIZE_MAX;
  // 1-based push number from which every push lands at offset 0; 0 never
  static inline uint32_t zeroFrom = 0;
  static inline uint32_t pushes = 0;

  static void reset() {
    bytes.clear();
    capacity = SIZE_MAX;
    zeroFrom = 0;
    pushes = 0;
  }

  static gfx::Range push(std::span<const uint8_t> data) noexcept {
    ++pushes;
    if (zeroFrom != 0 && pushes >= zeroFrom) {
      return {0, static_cast<uint32_t>(data.size())};
    }
    const size_t begin = std::max<size_t>((bytes.size() + 15) & ~size_t{15}, 16);
    if (begin + data.size() > capacity) {
      return {};
    }
    bytes.resize(begin);
    bytes.insert(bytes.end(), data.begin(), data.end());
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(data.size())};
  }

  template <typename T>
  static T at(uint32_t locator) {
    T value{};
    if (size_t{locator} * 4 + sizeof(T) <= bytes.size()) {
      std::memcpy(&value, bytes.data() + size_t{locator} * 4, sizeof(T));
    }
    return value;
  }
};

scene::DrawClass draw_class(scene::Window window = scene::Window::Opaque, uint64_t object = 0) {
  scene::DrawClass cls;
  cls.key.role = scene::Role::Main;
  cls.key.window = window;
  if (object != 0) {
    cls.key.object = {.id = object, .subId = 0, .generation = 1, .valid = true};
  }
  return cls;
}

uint32_t stamp(const scene::DrawClass& cls, uint32_t instances = 1) {
  gx::DrawData draw{};
  draw.instanceCount = instances;
  records::stamp(draw, cls);
  return draw.immediateData._pad;
}

struct ScopedTargets {
  ~ScopedTargets() {
    webgpu::g_graphicsConfig.gbuffer = gbuffer;
    webgpu::g_graphicsConfig.debugIdBuffer = debugId;
  }
  bool gbuffer = webgpu::g_graphicsConfig.gbuffer;
  bool debugId = webgpu::g_graphicsConfig.debugIdBuffer;
};

struct ScopedHints {
  ScopedHints() {
    detail::testing::set_hints_override(1);
    detail::install_hints();
  }
  ~ScopedHints() { detail::testing::reset_hints(); }
};

class RenderGxExt : public ::testing::Test {
protected:
  void SetUp() override { reset_all(); }
  void TearDown() override { reset_all(); }

  static void reset_all() {
    testing::reset();
    texture::clear_material_manifests();
    material::latch();
    gx::g_gxState = {};
    gx::render_ext::g_drawInputs = 0;
    FakeStorage::reset();
  }

  static void add(std::string_view text) {
    ASSERT_EQ(texture::register_material_manifest(text, {.layer = texture::MaterialLayer::User}).status,
              texture::MaterialManifestStatus::Accepted);
    material::latch();
  }

  static gx::RenderExtConfig derive(const gx::ShaderInfo& info, gx::render_ext::StochasticLanes& lanes) {
    const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
    EXPECT_NE(hooks, nullptr);
    return hooks != nullptr
               ? hooks->derive(hooks->userData, gx::PipelineConfig{}, info, gfx::RenderTargetLayout{}, lanes)
               : gx::RenderExtConfig{};
  }
};

TEST_F(RenderGxExt, GateNeedsSettingsAndFlaggedMaterial) {
  add(Plain);
  apply_frame_state(frame(true, true));
  EXPECT_FALSE(testing::active());
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);

  add(Flagged);
  apply_frame_state(frame(false, true));
  EXPECT_FALSE(testing::active());
  apply_frame_state(frame(true, false));
  EXPECT_FALSE(testing::active());
  apply_frame_state(frame(true, true));
  EXPECT_TRUE(testing::active());
  EXPECT_NE(gx::render_ext::hooks(), nullptr);
  apply_frame_state(frame(false, true));
  EXPECT_FALSE(testing::active());
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);
}

TEST_F(RenderGxExt, RegistryChangeWhileActiveInvalidates) {
  add(Flagged);
  apply_frame_state(frame(true, true));
  ASSERT_TRUE(testing::active());
  const uint64_t generation = gx::texture::current_bind_generation();
  apply_frame_state(frame(true, true));
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);

  add(Plain);
  apply_frame_state(frame(true, true));
  EXPECT_TRUE(testing::active());
  EXPECT_GT(gx::texture::current_bind_generation(), generation);
}

TEST_F(RenderGxExt, ForceFlagsEveryEligibleRepeatSlot) {
  testing::set_force(true);
  apply_frame_state(frame(false, false));
  ASSERT_TRUE(testing::active());
  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(0b101), lanes).stochasticMask, 0b101u);
  EXPECT_TRUE(lanes[0] == gx::render_ext::pack({}));
  EXPECT_TRUE(lanes[2] == gx::render_ext::pack({}));

  gfx::TextureBind bind = repeat_bind();
  EXPECT_TRUE(gx::render_ext::stochastic_bind_ok(bind));
  bind.texObj.mode0 = GX_CLAMP | GX_REPEAT << 2;
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(bind));
  bind.texObj.mode0 = GX_REPEAT | GX_MIRROR << 2;
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(bind));
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(gfx::TextureBind{}));

  testing::set_force(false);
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(testing::active());
}

TEST_F(RenderGxExt, EnvOnOpensGateButNeedsFlags) {
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_STOCHASTIC", "on", true);
  testing::reset();
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(testing::active());
  add(Flagged);
  apply_frame_state(frame(false, false));
  EXPECT_TRUE(testing::active());
  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(1), lanes).stochasticMask, 0u);
  gx::g_gxState.textures[0].material = material_ref(FlaggedKey);
  EXPECT_EQ(derive(eligible(1), lanes).stochasticMask, 1u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_STOCHASTIC");
}

TEST_F(RenderGxExt, DeriveUsesManifestParamsAndDefaults) {
  add(R"({"version":1,"materials":{)"
      R"("tex1_16x16_000000000000a001_14":{"flags":["stochastic"],"stochastic":{"scale":4,"rotation":0.5}},)"
      R"("tex1_16x16_000000000000a002_14":{"flags":["stochastic"]},)"
      R"("tex1_16x16_000000000000a003_14":{"flags":["stochastic","cutout"]},)"
      R"("tex1_16x16_000000000000a004_14":{"class":"ground/grass"}}})");
  apply_frame_state(frame(true, true));
  ASSERT_TRUE(testing::active());
  const gfx::TextureMaterialRef params = material_ref(FlaggedKey);
  const gfx::TextureMaterialRef defaults = material_ref("tex1_16x16_000000000000a002_14");
  ASSERT_NE(params.id, 0u);
  ASSERT_NE(defaults.id, 0u);
  auto& textures = gx::g_gxState.textures;
  textures[0].material = params;
  textures[1].material = defaults;
  textures[2].material = material_ref("tex1_16x16_000000000000a003_14");
  textures[3].material = material_ref(PlainKey);
  textures[4].material = {params.id, 0};
  textures[5].material = {0, material::Stochastic};
  textures[6].material = {60000, material::Stochastic};
  textures[7].material = defaults;

  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(0xFF), lanes).stochasticMask, 0b10000011u);
  EXPECT_TRUE(lanes[0] == gx::render_ext::pack({.scale = 4.f, .rotation = 0.5f}));
  EXPECT_TRUE(lanes[1] == gx::render_ext::pack({}));
  EXPECT_TRUE(lanes[7] == gx::render_ext::pack({}));
  EXPECT_EQ(derive(eligible(0b10), lanes).stochasticMask, 0b10u);
}

TEST_F(RenderGxExt, RecordsInactiveWritesNothing) {
  const ScopedTargets targets;
  records::testing::set_push(&FakeStorage::push);
  apply_frame_state(frame(false, false));
  gx::render_ext::g_drawInputs = 5;
  EXPECT_FALSE(records::on_route(draw_class()));
  EXPECT_EQ(gx::render_ext::g_drawInputs, 0u);
  EXPECT_EQ(stamp(draw_class()), 0u);

  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.debugIdBuffer = true;
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(records::on_route(draw_class()));
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 7)), 0u);
  EXPECT_EQ(FakeStorage::pushes, 0u);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().frames, 0u);
  EXPECT_EQ(records::stats().records, 0u);
}

TEST_F(RenderGxExt, RecordsStampNonEmptyCommands) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  const uint32_t first = stamp(draw_class(scene::Window::Opaque, 0x123456789ABCDEF0ull));
  EXPECT_EQ(first, 32u / 4);
  const auto header = FakeStorage::at<records::FrameRecord>(16 / 4);
  EXPECT_EQ(header.frameSeq, static_cast<uint32_t>(detail::live_scene_hints().hintFrame));
  EXPECT_EQ(header.flags, 0u);
  const auto record = FakeStorage::at<records::DrawRecord>(first);
  EXPECT_EQ(record.frameOff, 16u / 4);
  EXPECT_EQ(record.flags, 0u);
  EXPECT_EQ(record.objectLo, 0x9ABCDEF0u);
  EXPECT_EQ(record.objectHi, 0x12345678u);
  EXPECT_EQ(record.prevMtxOff, 0u);
  EXPECT_EQ(record.materialOff0, 0u);
  EXPECT_EQ(record.materialOff1, 0u);
  EXPECT_EQ(record.tevPolicyOff, 0u);

  const uint32_t objectless = stamp(draw_class(scene::Window::Translucent));
  EXPECT_EQ(objectless, 64u / 4);
  EXPECT_EQ(FakeStorage::at<records::DrawRecord>(objectless).objectLo, 0u);
  EXPECT_EQ(FakeStorage::at<records::DrawRecord>(objectless).frameOff, 16u / 4);
  EXPECT_NE(stamp(draw_class(scene::Window::PreScene, 9)), 0u);
  EXPECT_EQ(FakeStorage::pushes, 4u);
}

TEST_F(RenderGxExt, RecordsReuseEqualRecordExceptWithDebugId) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  const uint32_t a = stamp(draw_class(scene::Window::Opaque, 7));
  EXPECT_EQ(stamp(draw_class(scene::Window::Translucent, 7)), a);
  const uint32_t b = stamp(draw_class(scene::Window::Opaque, 8));
  EXPECT_NE(b, a);
  EXPECT_GT(stamp(draw_class(scene::Window::Opaque, 7)), b);
  EXPECT_EQ(FakeStorage::pushes, 4u);

  records::testing::set_active(true, false, true);
  const uint32_t c = stamp(draw_class(scene::Window::Opaque, 7));
  const uint32_t d = stamp(draw_class(scene::Window::Opaque, 7));
  const uint32_t e = stamp(draw_class(scene::Window::Opaque, 7));
  EXPECT_GT(c, 0u);
  EXPECT_GT(d, c);
  EXPECT_GT(e, d);
  EXPECT_EQ(FakeStorage::at<records::DrawRecord>(d).frameOff, FakeStorage::at<records::DrawRecord>(c).frameOff);

  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().records, 6u);
  EXPECT_EQ(records::stats().reused, 1u);
  EXPECT_EQ(records::stats().codes[0], 7u);
}

TEST_F(RenderGxExt, RecordsSkipEmptyOffscreenAndPostWindows) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  scene::DrawClass offscreen = draw_class(scene::Window::Opaque, 1);
  offscreen.offscreen = true;
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1), 0), 0u);
  EXPECT_EQ(stamp(offscreen), 0u);
  for (const scene::Window window :
       {scene::Window::Post, scene::Window::SceneEnd, scene::Window::Hud, scene::Window::Final}) {
    EXPECT_EQ(stamp(draw_class(window, 1)), 0u);
  }
  EXPECT_EQ(FakeStorage::pushes, 0u);
  for (const scene::Window window : {scene::Window::PreScene, scene::Window::Opaque, scene::Window::Translucent}) {
    EXPECT_NE(stamp(draw_class(window, 1)), 0u);
  }

  records::testing::set_active(true, false, true);
  EXPECT_EQ(stamp(draw_class(scene::Window::Hud, 1), 0), 0u);
  EXPECT_EQ(stamp(offscreen), 0u);
  EXPECT_NE(stamp(draw_class(scene::Window::Post, 1)), 0u);
  EXPECT_NE(stamp(draw_class(scene::Window::Hud, 1)), 0u);
  EXPECT_NE(stamp(draw_class(scene::Window::Final, 1)), 0u);
}

TEST_F(RenderGxExt, RecordsDropOnFullStorage) {
  records::testing::set_push(&FakeStorage::push);
  FakeStorage::capacity = 64;
  records::testing::set_active(true, false, true);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 32u / 4);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 2)), 0u);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 3)), 0u);

  FakeStorage::reset();
  FakeStorage::capacity = 16;
  records::testing::set_active(true, true, false);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);
  EXPECT_EQ(FakeStorage::pushes, 2u);

  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().records, 1u);
  EXPECT_EQ(records::stats().dropped, 4u);
}

TEST_F(RenderGxExt, RecordsTreatOffsetZeroAsDropped) {
  records::testing::set_push(&FakeStorage::push);
  FakeStorage::zeroFrom = 1;
  records::testing::set_active(true, true, false);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);

  FakeStorage::reset();
  FakeStorage::zeroFrom = 2;
  records::testing::set_active(true, true, false);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);
  EXPECT_EQ(FakeStorage::pushes, 2u);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);
  EXPECT_EQ(FakeStorage::pushes, 3u);

  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().records, 0u);
  EXPECT_EQ(records::stats().dropped, 3u);
}

TEST_F(RenderGxExt, RecordStatsPublishAtLatch) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  stamp(draw_class(scene::Window::Opaque, 1));
  stamp(draw_class(scene::Window::Opaque, 1));
  stamp(draw_class(scene::Window::Opaque, 2));
  EXPECT_EQ(records::stats().records, 0u);
  EXPECT_EQ(records::stats().frames, 0u);

  apply_frame_state(frame(false, false));
  records::Stats stats = records::stats();
  EXPECT_EQ(stats.frames, 1u);
  EXPECT_EQ(stats.records, 2u);
  EXPECT_EQ(stats.reused, 1u);
  EXPECT_EQ(stats.dropped, 0u);
  EXPECT_EQ(stats.codes[0], 3u);
  EXPECT_EQ(stats.codes[1] + stats.codes[2] + stats.codes[3], 0u);

  apply_frame_state(frame(false, false));
  stats = records::stats();
  EXPECT_EQ(stats.frames, 1u);
  EXPECT_EQ(stats.records, 2u);
  records::shutdown();
  EXPECT_EQ(records::stats().records, 0u);
}

TEST_F(RenderGxExt, RecordsActiveFollowsLayout) {
  const ScopedTargets targets;
  const ScopedHints hints;
  records::testing::set_push(&FakeStorage::push);
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(records::on_route(draw_class()));
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);

  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  gx::render_ext::g_drawInputs = 5;
  EXPECT_TRUE(records::on_route(draw_class()));
  EXPECT_EQ(gx::render_ext::g_drawInputs, 0u);
  const uint32_t a = stamp(draw_class(scene::Window::Opaque, 1));
  EXPECT_NE(a, 0u);
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), a);
  EXPECT_EQ(stamp(draw_class(scene::Window::Post, 1)), 0u);

  webgpu::g_graphicsConfig.gbuffer = false;
  webgpu::g_graphicsConfig.debugIdBuffer = true;
  FakeStorage::reset();
  apply_frame_state(frame(false, false));
  EXPECT_TRUE(records::on_route(draw_class()));
  const uint32_t b = stamp(draw_class(scene::Window::Opaque, 1));
  EXPECT_NE(b, 0u);
  EXPECT_GT(stamp(draw_class(scene::Window::Opaque, 1)), b);
  EXPECT_NE(stamp(draw_class(scene::Window::Post, 1)), 0u);

  webgpu::g_graphicsConfig.gbuffer = true;
  detail::remove_hints();
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(records::on_route(draw_class()));
  EXPECT_EQ(stamp(draw_class(scene::Window::Opaque, 1)), 0u);
}
} // namespace
} // namespace aurora::render::gx_ext

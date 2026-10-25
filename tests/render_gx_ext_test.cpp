#include <gtest/gtest.h>

#include "gfx/scene_targets.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
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
  records::stamp(draw, gx::PipelineConfig{}, cls);
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
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
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
scene::DrawClass receiver_class(uint32_t categories, scene::Kind kind = scene::Kind::MainOpaque,
                                scene::Window window = scene::Window::Opaque) {
  scene::DrawClass cls = draw_class(window, 1);
  cls.kind = kind;
  cls.key.categories = categories;
  return cls;
}

uint32_t stamped_code(const scene::DrawClass& cls) {
  const uint32_t locator = stamp(cls);
  EXPECT_NE(locator, 0u);
  const uint32_t flags = FakeStorage::at<records::DrawRecord>(locator).flags;
  return (flags & records::flag::ReceiverMask) >> records::flag::ReceiverShift;
}

TEST_F(RenderGxExt, ReceiverCodesFollowCategories) {
  namespace cat = scene::category;
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  EXPECT_EQ(stamped_code(receiver_class(0)), 3u);
  EXPECT_EQ(stamped_code(receiver_class(cat::Static | cat::Terrain)), 3u);
  EXPECT_EQ(stamped_code(receiver_class(cat::NoShadowReceive)), 1u);
  EXPECT_EQ(stamped_code(receiver_class(cat::NoShadowReceive | cat::CloudReceiver)), 1u);
  EXPECT_EQ(stamped_code(receiver_class(cat::CloudReceiver | cat::Foliage)), 2u);
  for (const uint32_t category : {cat::Overlay, cat::Particle, cat::NoAo, cat::Sky, cat::Ui, cat::FakeShadow}) {
    EXPECT_EQ(stamped_code(receiver_class(category)), 0u) << category;
    EXPECT_EQ(stamped_code(receiver_class(category | cat::CloudReceiver)), 0u) << category;
  }
  for (const scene::Kind kind : {scene::Kind::MainBlended, scene::Kind::Overlay, scene::Kind::Sky, scene::Kind::Foreign,
                                 scene::Kind::MainDepthOnly}) {
    EXPECT_EQ(stamped_code(receiver_class(cat::Terrain, kind)), 0u) << static_cast<int>(kind);
  }
  for (const scene::Window window : {scene::Window::PreScene, scene::Window::Translucent}) {
    EXPECT_EQ(stamped_code(receiver_class(cat::Terrain, scene::Kind::MainOpaque, window)), 0u)
        << static_cast<int>(window);
  }
  scene::DrawClass offscreen = receiver_class(cat::Terrain);
  offscreen.offscreen = true;
  EXPECT_EQ(stamp(offscreen), 0u);

  apply_frame_state(frame(false, false));
  const records::Stats stats = records::stats();
  EXPECT_EQ(stats.codes[1], 2u);
  EXPECT_EQ(stats.codes[2], 1u);
  EXPECT_EQ(stats.codes[3], 2u);
  EXPECT_EQ(stats.codes[0], 19u);
}

// C_MTXPerspective(60, 4:3, 1, 1000), as GXSetProjection receives it
Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

gfx::RenderTargetLayout scene_layout(bool gbuffer) {
  return gfx::compose_scene_layout({.gbuffer = gbuffer}, {.color = wgpu::TextureFormat::BGRA8Unorm,
                                                          .depth = wgpu::TextureFormat::Depth32Float,
                                                          .size = {64, 64, 1},
                                                          .samples = 1});
}

gx::PipelineConfig writer(bool depth, bool color) {
  gx::PipelineConfig config{};
  config.depthCompare = true;
  config.depthUpdate = depth;
  config.colorUpdate = color;
  return config;
}

uint8_t derived_output(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_NE(hooks, nullptr);
  if (hooks == nullptr) {
    return 0xFF;
  }
  gx::render_ext::StochasticLanes lanes{};
  const gx::RenderExtConfig ext = hooks->derive(hooks->userData, config, gx::ShaderInfo{}, layout, lanes);
  EXPECT_EQ(ext.stochasticMask, 0u);
  return ext.output;
}

TEST_F(RenderGxExt, OpaqueParticleAndNoAoKeepCodeZero) {
  const ScopedTargets targets;
  const ScopedHints hints;
  records::testing::set_push(&FakeStorage::push);
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  ASSERT_TRUE(records::gbuffer());
  const gx::hints::Sink& sink = *gx::hints::sink();
  const gx::draw_route::Router& router = *gx::draw_route::router();
  const gx::hints::Camera main{
      .role = AURORA_CAMERA_MAIN,
      .proj = perspective(),
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(perspective(), GX_PERSPECTIVE),
  };
  sink.camera(sink.userData, main);
  sink.stage(sink.userData, Stage::SceneBegin);
  gx::GXState& state = gx::g_gxState;
  state.proj = main.gxProj;
  state.depthCompare = true;
  state.depthUpdate = true;
  state.colorUpdate = true;
  state.blendMode = GX_BM_NONE;
  const gfx::RenderTargetLayout gbuffer = scene_layout(true);
  const auto routed = [&](uint32_t expectedCode) {
    EXPECT_EQ(router.route(router.userData, {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3}).route,
              gx::draw_route::Route::Push);
    EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
    const gx::PipelineConfig config = writer(true, true);
    EXPECT_EQ(derived_output(config, gbuffer), 1u);
    gx::DrawData draw{};
    draw.instanceCount = 1;
    router.push(router.userData, draw, config, gx::ShaderInfo{});
    ASSERT_NE(draw.immediateData._pad, 0u);
    const uint32_t flags = FakeStorage::at<records::DrawRecord>(draw.immediateData._pad).flags;
    EXPECT_EQ((flags & records::flag::ReceiverMask) >> records::flag::ReceiverShift, expectedCode);
  };

  sink.objectBegin(sink.userData, {.id = 1, .generation = 1, .categories = AURORA_CAT_PARTICLE});
  routed(0);
  sink.objectEnd(sink.userData);
  sink.scopePush(sink.userData, {.categories = AURORA_CAT_NO_AO});
  sink.objectBegin(sink.userData, {.id = 2, .generation = 1, .categories = AURORA_CAT_STATIC});
  routed(0);
  sink.objectEnd(sink.userData);
  sink.scopePop(sink.userData);
  sink.objectBegin(sink.userData, {.id = 3, .generation = 1, .categories = AURORA_CAT_STATIC | AURORA_CAT_TERRAIN});
  routed(3);
  sink.objectEnd(sink.userData);
}

TEST_F(RenderGxExt, OutputMarksInWindowWritersOnly) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  const gfx::RenderTargetLayout gbuffer = scene_layout(true);
  const gfx::RenderTargetLayout legacy = scene_layout(false);
  ASSERT_TRUE(gfx::layout_has_gbuffer(gbuffer));
  const gx::PipelineConfig depthOnly = writer(true, false);
  const gx::PipelineConfig colorOnly = writer(false, true);
  const gx::PipelineConfig both = writer(true, true);
  gx::PipelineConfig noTest = writer(true, false);
  noTest.depthCompare = false;

  for (const scene::Kind kind : {scene::Kind::MainOpaque, scene::Kind::MainBlended, scene::Kind::Overlay,
                                 scene::Kind::Sky, scene::Kind::Foreign, scene::Kind::MainDepthOnly}) {
    scene::DrawClass cls = draw_class();
    cls.kind = kind;
    cls.key.role = kind == scene::Kind::Sky ? scene::Role::Sky : scene::Role::Main;
    records::on_route(cls);
    EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
    for (const gx::PipelineConfig& config : {depthOnly, colorOnly, both}) {
      EXPECT_EQ(derived_output(config, gbuffer), 1u) << static_cast<int>(kind);
      EXPECT_EQ(derived_output(config, legacy), 0u) << static_cast<int>(kind);
    }
    EXPECT_EQ(derived_output(writer(false, false), gbuffer), 0u);
    EXPECT_EQ(derived_output(noTest, gbuffer), 0u);
  }

  scene::DrawClass offscreen = draw_class();
  offscreen.offscreen = true;
  for (const scene::DrawClass& cls : {draw_class(scene::Window::PreScene), draw_class(scene::Window::Translucent),
                                      draw_class(scene::Window::Post), offscreen}) {
    records::on_route(cls);
    EXPECT_EQ(gx::render_ext::g_drawInputs, 0u);
    EXPECT_EQ(derived_output(both, gbuffer), 0u);
  }

  EXPECT_EQ(output_for(both, true), 1u);
  EXPECT_EQ(output_for(depthOnly, true), 1u);
  EXPECT_EQ(output_for(colorOnly, true), 1u);
  EXPECT_EQ(output_for(noTest, true), 0u);
  EXPECT_EQ(output_for(both, false), 0u);
}

TEST_F(RenderGxExt, HooksInstalledForGbufferWithoutStochastic) {
  const ScopedTargets targets;
  const ScopedHints hints;
  add(Flagged);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);

  webgpu::g_graphicsConfig.gbuffer = true;
  const uint64_t generation = gx::texture::current_bind_generation();
  apply_frame_state(frame(false, false));
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  ASSERT_NE(hooks, nullptr);
  EXPECT_FALSE(testing::active());
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);
  gx::g_gxState.textures[0].material = material_ref(FlaggedKey);
  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(hooks->derive(hooks->userData, writer(true, true), eligible(1), scene_layout(true), lanes).stochasticMask,
            0u);

  apply_frame_state(frame(true, true));
  EXPECT_TRUE(testing::active());
  EXPECT_EQ(gx::render_ext::hooks(), hooks);
  EXPECT_EQ(hooks->derive(hooks->userData, writer(true, true), eligible(1), scene_layout(true), lanes).stochasticMask,
            1u);

  webgpu::g_graphicsConfig.gbuffer = false;
  apply_frame_state(frame(false, false));
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);

  webgpu::g_graphicsConfig.gbuffer = true;
  detail::remove_hints();
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(records::gbuffer());
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);
}
} // namespace
} // namespace aurora::render::gx_ext

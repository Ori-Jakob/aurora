#include <gtest/gtest.h>

#include "gfx/scene_targets.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "gx/shader_info.hpp"
#include "gx/texture.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/gx_ext/ext_provider.hpp"
#include "render/gx_ext/fwd_env.hpp"
#include "render/gx_ext/records.hpp"
#include "render/gx_ext/surface_table.hpp"
#include "render/gx_ext/unlit.hpp"
#include "render/fakeshadow/lsm.hpp"
#include "render/hints.hpp"
#include "render/material/classes.hpp"
#include "render/material/gx_lookup.hpp"
#include "render/material/registry.hpp"
#include "render/post/composite.hpp"
#include "render/scene/classifier.hpp"
#include "webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>
#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
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
  records::stamp(draw, gx::PipelineConfig{}, gx::ShaderInfo{}, cls);
  return draw.immediateData._pad;
}

struct ScopedTargets {
  ~ScopedTargets() {
    webgpu::g_graphicsConfig.gbuffer = gbuffer;
    webgpu::g_graphicsConfig.debugIdBuffer = debugId;
    webgpu::g_graphicsConfig.surfaceBuffer = surface;
  }
  bool gbuffer = webgpu::g_graphicsConfig.gbuffer;
  bool debugId = webgpu::g_graphicsConfig.debugIdBuffer;
  bool surface = webgpu::g_graphicsConfig.surfaceBuffer;
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
  EXPECT_EQ(record.material, 0u);
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

gx::PipelineConfig translucent(bool depthUpdate) {
  gx::PipelineConfig config{};
  config.blendMode = GX_BM_BLEND;
  config.blendFacSrc = GX_BL_SRCALPHA;
  config.blendFacDst = GX_BL_INVSRCALPHA;
  config.depthCompare = true;
  config.depthUpdate = depthUpdate;
  config.colorUpdate = true;
  return config;
}

bool stamped_reactive(scene::Window window, scene::Role role, const gx::PipelineConfig& config,
                      uint32_t categories = 0) {
  scene::DrawClass cls = draw_class(window, 1);
  cls.key.role = role;
  cls.key.categories = categories;
  cls.kind = role == scene::Role::Sky ? scene::Kind::Sky : scene::Kind::MainBlended;
  gx::DrawData draw{};
  draw.instanceCount = 1;
  records::stamp(draw, config, gx::ShaderInfo{}, cls);
  EXPECT_NE(draw.immediateData._pad, 0u);
  return (FakeStorage::at<records::DrawRecord>(draw.immediateData._pad).flags & records::flag::Reactive) != 0;
}

TEST_F(RenderGxExt, ReactiveFlagMarksInWindowTranslucentsOnly) {
  using scene::Role;
  using scene::Window;
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  EXPECT_TRUE(stamped_reactive(Window::Opaque, Role::Main, translucent(false)));
  EXPECT_TRUE(stamped_reactive(Window::Translucent, Role::Main, translucent(false)));
  EXPECT_TRUE(stamped_reactive(Window::Translucent, Role::None, translucent(false)));
  EXPECT_FALSE(stamped_reactive(Window::PreScene, Role::Sky, translucent(false)));
  EXPECT_FALSE(stamped_reactive(Window::PreScene, Role::Main, translucent(false)));
  EXPECT_FALSE(stamped_reactive(Window::Opaque, Role::Sky, translucent(false)));
  EXPECT_FALSE(stamped_reactive(Window::Translucent, Role::Sky, translucent(false)));
  EXPECT_FALSE(stamped_reactive(Window::Translucent, Role::Main, translucent(true)));
  gx::PipelineConfig opaque = translucent(false);
  opaque.blendMode = GX_BM_NONE;
  EXPECT_FALSE(stamped_reactive(Window::Opaque, Role::Main, opaque));
  gx::PipelineConfig depthOnly = translucent(true);
  depthOnly.colorUpdate = false;
  EXPECT_FALSE(stamped_reactive(Window::Opaque, Role::Main, depthOnly));
}

TEST_F(RenderGxExt, ReactiveFlagSkipsTerrainLayersAndDecals) {
  namespace cat = scene::category;
  using scene::Role;
  using scene::Window;
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  EXPECT_FALSE(stamped_reactive(Window::Opaque, Role::Main, translucent(false), cat::Static | cat::Terrain));
  EXPECT_FALSE(stamped_reactive(Window::Opaque, Role::Main, translucent(false), cat::Decal | cat::FakeShadow));
  EXPECT_FALSE(
      stamped_reactive(Window::Translucent, Role::Main, translucent(false), cat::Terrain | cat::CloudReceiver));
  EXPECT_TRUE(stamped_reactive(Window::Translucent, Role::Main, translucent(false), cat::Static | cat::Water));
  EXPECT_TRUE(stamped_reactive(Window::Translucent, Role::Main, translucent(false), cat::Dynamic | cat::Foliage));
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

TEST_F(RenderGxExt, FogLayoutMarksInWindowOpaqueWriters) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  const gfx::RenderTargetLayout gbuffer = scene_layout(true);
  const gfx::RenderTargetLayout legacy = scene_layout(false);
  ASSERT_NE(gx::shader_targets(gbuffer).fog, UINT32_MAX);
  ASSERT_EQ(gx::shader_targets(legacy).fog, UINT32_MAX);
  gx::PipelineConfig colorOnly = writer(false, true);
  colorOnly.blendMode = GX_BM_NONE;
  const gx::PipelineOptions options{.colorUpdate = true};
  const auto marked = [&](const gfx::RenderTargetLayout& layout) {
    gx::PipelineConfig config = colorOnly;
    config.shaderConfig.ext.output = derived_output(colorOnly, layout);
    return config;
  };

  records::on_route(draw_class());
  ASSERT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
  const gx::PipelineConfig inWindow = marked(gbuffer);
  EXPECT_GE(inWindow.shaderConfig.ext.output, 1u);
  EXPECT_TRUE(gx::writes_fog_record(inWindow, options));
  EXPECT_EQ(marked(legacy).shaderConfig.ext.output, 0u);
  EXPECT_FALSE(gx::writes_fog_record(marked(legacy), options));

  scene::DrawClass offscreen = draw_class();
  offscreen.offscreen = true;
  for (const scene::DrawClass& cls : {draw_class(scene::Window::PreScene), draw_class(scene::Window::Translucent),
                                      draw_class(scene::Window::Post), offscreen}) {
    records::on_route(cls);
    const gx::PipelineConfig config = marked(gbuffer);
    EXPECT_EQ(config.shaderConfig.ext.output, 0u) << static_cast<int>(cls.key.window);
    EXPECT_FALSE(gx::writes_fog_record(config, options)) << static_cast<int>(cls.key.window);
  }
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

namespace cat = scene::category;

scene::DrawClass forward_class(uint32_t categories = cat::Terrain) {
  scene::DrawClass cls = draw_class(scene::Window::Translucent, 1);
  cls.kind = scene::Kind::MainBlended;
  cls.mainClass = true;
  cls.key.categories = categories;
  return cls;
}

gx::PipelineConfig layer_config(bool clamped = true) {
  gx::PipelineConfig config = translucent(false);
  config.shaderConfig.tevStageCount = 1;
  config.shaderConfig.tevStages[0].colorOp.clamp = clamped;
  config.shaderConfig.ext.fwdEnv = 1;
  return config;
}

detail::FrameState ao_frame(Quality ao, uint16_t view = 0) {
  detail::FrameState state;
  state.settings.ao = ao;
  state.settings.debugView = view;
  state.debugView = view;
  return state;
}

gx::RenderExtConfig derived_ext(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_NE(hooks, nullptr);
  gx::render_ext::StochasticLanes lanes{};
  return hooks != nullptr ? hooks->derive(hooks->userData, config, gx::ShaderInfo{}, layout, lanes)
                          : gx::RenderExtConfig{};
}

TEST_F(RenderGxExt, FwdEnvOnlyForTranslucentMainColourWriters) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  const gfx::RenderTargetLayout gbuffer = scene_layout(true);
  const gfx::RenderTargetLayout legacy = scene_layout(false);
  ASSERT_TRUE(records::in_forward_window(forward_class()));
  records::on_route(forward_class());
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::ForwardWindow);
  EXPECT_EQ(derived_ext(writer(false, true), gbuffer).fwdEnv, 1u);
  EXPECT_EQ(derived_ext(writer(false, true), gbuffer).output, 0u);
  EXPECT_EQ(derived_ext(layer_config(false), gbuffer).fwdEnv, 1u);
  EXPECT_EQ(derived_ext(writer(true, false), gbuffer).fwdEnv, 0u);
  EXPECT_EQ(derived_ext(writer(false, true), legacy).fwdEnv, 0u);

  scene::DrawClass notMain = forward_class();
  notMain.mainClass = false;
  scene::DrawClass offscreen = forward_class();
  offscreen.offscreen = true;
  scene::DrawClass noColour = forward_class();
  noColour.colorWrite = false;
  std::vector<scene::DrawClass> others{notMain, offscreen, noColour};
  for (const scene::Window window : {scene::Window::PreScene, scene::Window::Opaque, scene::Window::Post,
                                     scene::Window::SceneEnd, scene::Window::Hud}) {
    scene::DrawClass cls = forward_class();
    cls.key.window = window;
    others.push_back(cls);
  }
  for (const scene::DrawClass& cls : others) {
    EXPECT_FALSE(records::in_forward_window(cls)) << static_cast<int>(cls.key.window);
    records::on_route(cls);
    EXPECT_EQ(gx::render_ext::g_drawInputs & records::ForwardWindow, 0u);
    EXPECT_EQ(derived_ext(writer(false, true), gbuffer).fwdEnv, 0u) << static_cast<int>(cls.key.window);
  }
  EXPECT_EQ(fwd_env_for(writer(false, true), true), 1u);
  EXPECT_EQ(fwd_env_for(writer(true, false), true), 0u);
  EXPECT_EQ(fwd_env_for(writer(false, true), false), 0u);
}

TEST_F(RenderGxExt, AoToggleKeepsPipelineKeys) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  const gfx::RenderTargetLayout gbuffer = scene_layout(true);
  scene::DrawClass opaque = draw_class();
  opaque.mainClass = true;
  const std::array classes{forward_class(), forward_class(0), opaque};
  const std::array configs{layer_config(), layer_config(false), writer(true, true), writer(false, true)};
  const auto keys = [&](const detail::FrameState& state) {
    apply_frame_state(state);
    std::vector<gx::RenderExtConfig> out;
    for (const scene::DrawClass& cls : classes) {
      records::on_route(cls);
      for (const gx::PipelineConfig& config : configs) {
        out.push_back(derived_ext(config, gbuffer));
      }
    }
    return out;
  };
  const std::vector<gx::RenderExtConfig> off = keys(ao_frame(Quality::Off));
  EXPECT_TRUE(std::ranges::any_of(off, [](const gx::RenderExtConfig& ext) { return ext.fwdEnv != 0; }));
  EXPECT_TRUE(keys(ao_frame(Quality::High)) == off);
  EXPECT_TRUE(keys(ao_frame(Quality::Ultra, 1)) == off);
  fwd_env::testing::set_identity_override(true);
  EXPECT_TRUE(keys(ao_frame(Quality::High)) == off);
}

TEST_F(RenderGxExt, ScreenAoReceiverRule) {
  const gx::PipelineConfig base = layer_config();
  const auto receiver = [](const scene::DrawClass& cls, const gx::PipelineConfig& config) {
    return fwd_env::screen_ao_receiver(cls, config);
  };
  EXPECT_TRUE(receiver(forward_class(cat::Terrain), base));
  EXPECT_TRUE(receiver(forward_class(cat::Decal), base));
  EXPECT_TRUE(receiver(forward_class(cat::Static | cat::Terrain | cat::NoShadowReceive), base));
  EXPECT_FALSE(receiver(forward_class(0), base));
  EXPECT_FALSE(receiver(forward_class(cat::Static | cat::Foliage), base));
  for (const uint32_t excluded : {cat::Water, cat::NoAo, cat::Emissive, cat::Particle, cat::Sky, cat::Ui, cat::Overlay,
                                  cat::FakeShadow, cat::ScreenRead}) {
    EXPECT_FALSE(receiver(forward_class(cat::Terrain | excluded), base)) << excluded;
    EXPECT_FALSE(receiver(forward_class(cat::Decal | excluded), base)) << excluded;
  }

  struct Blend {
    GXBlendMode mode;
    GXBlendFactor src;
    GXBlendFactor dst;
  };
  for (const Blend& blend :
       {Blend{GX_BM_NONE, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA}, Blend{GX_BM_BLEND, GX_BL_ONE, GX_BL_ONE},
        Blend{GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_ONE}, Blend{GX_BM_BLEND, GX_BL_ONE, GX_BL_ZERO},
        Blend{GX_BM_SUBTRACT, GX_BL_ONE, GX_BL_ONE}, Blend{GX_BM_LOGIC, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA}}) {
    gx::PipelineConfig config = base;
    config.blendMode = blend.mode;
    config.blendFacSrc = blend.src;
    config.blendFacDst = blend.dst;
    EXPECT_FALSE(receiver(forward_class(), config)) << int{blend.mode} << " " << int{blend.src};
  }
  gx::PipelineConfig noTest = base;
  noTest.depthCompare = false;
  gx::PipelineConfig depthWrite = base;
  depthWrite.depthUpdate = true;
  gx::PipelineConfig noColour = base;
  noColour.colorUpdate = false;
  for (const gx::PipelineConfig& config : {noTest, depthWrite, noColour, layer_config(false)}) {
    EXPECT_FALSE(receiver(forward_class(), config));
  }

  scene::DrawClass notMain = forward_class();
  notMain.mainClass = false;
  scene::DrawClass offscreen = forward_class();
  offscreen.offscreen = true;
  EXPECT_FALSE(receiver(notMain, base));
  EXPECT_FALSE(receiver(offscreen, base));
  for (const scene::Window window : {scene::Window::PreScene, scene::Window::Opaque, scene::Window::Post}) {
    scene::DrawClass cls = forward_class();
    cls.key.window = window;
    EXPECT_FALSE(receiver(cls, base)) << static_cast<int>(window);
  }
}

TEST_F(RenderGxExt, ScreenAoFlagFollowsAoAndTier) {
  const ScopedTargets targets;
  const ScopedHints hints;
  records::testing::set_push(&FakeStorage::push);
  webgpu::g_graphicsConfig.gbuffer = true;
  const auto stamp_forward = [](const gx::PipelineConfig& config, uint32_t categories = cat::Terrain) {
    gx::DrawData draw{};
    draw.instanceCount = 1;
    records::stamp(draw, config, gx::ShaderInfo{}, forward_class(categories));
    return draw.immediateData._pad;
  };
  const auto flagged = [&](const gx::PipelineConfig& config, uint32_t categories = cat::Terrain) {
    const uint32_t locator = stamp_forward(config, categories);
    EXPECT_NE(locator, 0u);
    return (FakeStorage::at<records::DrawRecord>(locator).flags & records::flag::ScreenAo) != 0;
  };

  apply_frame_state(ao_frame(Quality::High));
  ASSERT_TRUE(records::gbuffer());
  EXPECT_TRUE(fwd_env::testing::screen_ao_active());
  EXPECT_TRUE(flagged(layer_config()));
  EXPECT_FALSE(flagged(layer_config(false)));
  EXPECT_FALSE(flagged(layer_config(), 0));
  gx::PipelineConfig noEnv = layer_config();
  noEnv.shaderConfig.ext.fwdEnv = 0;
  EXPECT_FALSE(flagged(noEnv));
  const uint32_t a = stamp_forward(layer_config());
  EXPECT_EQ(stamp_forward(layer_config()), a);
  EXPECT_NE(stamp_forward(layer_config(false)), a);
  apply_frame_state(ao_frame(Quality::Off));
  EXPECT_EQ(records::stats().screenAo, 3u);

  const auto expect_off = [&](const detail::FrameState& state, std::string_view label) {
    apply_frame_state(state);
    EXPECT_FALSE(fwd_env::testing::screen_ao_active()) << label;
    if (records::gbuffer()) {
      EXPECT_FALSE(flagged(layer_config())) << label;
    } else {
      EXPECT_EQ(stamp_forward(layer_config()), 0u) << label;
    }
  };
  expect_off(ao_frame(Quality::Off), "ao off");
  expect_off(ao_frame(Quality::High, 1), "class view");
  detail::FrameState overridden = ao_frame(Quality::High);
  overridden.debugView = 1;
  expect_off(overridden, "inspector class view");
  detail::FrameState pbrOnly = ao_frame(Quality::Off);
  pbrOnly.settings.pbr = true;
  expect_off(pbrOnly, "pbr without ao");
  fwd_env::testing::set_identity_override(true);
  expect_off(ao_frame(Quality::High), "identity");
  fwd_env::testing::set_identity_override(std::nullopt);
  webgpu::g_graphicsConfig.gbuffer = false;
  expect_off(ao_frame(Quality::High), "tier off");
  webgpu::g_graphicsConfig.gbuffer = true;
  detail::remove_hints();
  expect_off(ao_frame(Quality::High), "hints off");
  apply_frame_state(ao_frame(Quality::Off));
  EXPECT_EQ(records::stats().screenAo, 3u);
}

detail::GraphEnvironment tier_b_env() {
  detail::GraphEnvironment env;
  env.efbSize = {64, 32};
  env.outputSize = {64, 32};
  env.sceneSize = {64, 32};
  const auto set = [&](detail::GxResource resource, wgpu::TextureFormat format, wgpu::TextureUsage usage) {
    env.gxFormats[static_cast<size_t>(resource)] = format;
    env.gxUsage[static_cast<size_t>(resource)] = usage;
  };
  constexpr wgpu::TextureUsage Target =
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  set(detail::GxResource::Color, wgpu::TextureFormat::BGRA8Unorm, Target | wgpu::TextureUsage::CopyDst);
  set(detail::GxResource::Depth, wgpu::TextureFormat::Depth32Float,
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
  set(detail::GxResource::Normal, wgpu::TextureFormat::RGB10A2Unorm, Target);
  set(detail::GxResource::Motion, wgpu::TextureFormat::RG16Float, Target);
  set(detail::GxResource::Reactive, wgpu::TextureFormat::R8Unorm, Target);
  set(detail::GxResource::Fog, wgpu::TextureFormat::RGBA8Uint, Target);
  return env;
}

void declare_stub_ao(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.after(phase::Ao);
  builder.create(resource::AoFinal,
                 {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb, .divisor = 1},
                 Access::CopyDst);
}

void encode_nothing(detail::PassContext&, const void*) {}

struct PublisherGraph {
  std::vector<detail::PassDesc> passes;
  std::vector<detail::GxReadDesc> reads;
  std::shared_ptr<detail::CompiledGraph> graph;
};

PublisherGraph compile_publisher(const Settings& settings, const detail::GraphEnvironment& env = tier_b_env()) {
  static const Caps caps;
  PublisherGraph out;
  out.passes = {detail::composite::copy_pass_desc(), detail::composite::pass_desc(), fwd_env::pass_desc(),
                detail::PassDesc{.name = "test.ao",
                                 .stage = Stage::OpaqueDone,
                                 .kind = PassKind::Copy,
                                 .declare = declare_stub_ao,
                                 .encode = encode_nothing}};
  out.reads = {fwd_env::read_desc()};
  out.graph = detail::compile_graph(detail::GraphInputs{
      .passes = out.passes,
      .gxReads = out.reads,
      .settings = &settings,
      .caps = &caps,
      .env = env,
      .sceneDepthSource = DepthSource::Direct,
  });
  return out;
}

detail::CullReason publisher_status(const detail::CompiledGraph& graph) {
  const detail::NodeId id = graph.find_node(fwd_env::Pass);
  EXPECT_NE(id, detail::InvalidNode);
  return id != detail::InvalidNode ? graph.nodes[id].status : detail::CullReason::None;
}

bool reads_ao(const detail::CompiledGraph& graph) {
  const detail::NodeId chain = graph.find_node(fwd_env::ReadNode);
  if (chain == detail::InvalidNode) {
    return false;
  }
  return std::ranges::any_of(graph.node_uses(chain), [&](const detail::CompiledUse& use) {
    return graph.resources[use.resource].name == resource::AoFinal;
  });
}

TEST_F(RenderGxExt, FwdEnvPublisherSkipsWithoutAoOrTier) {
  Settings settings;
  const PublisherGraph off = compile_publisher(settings);
  ASSERT_NE(off.graph, nullptr);
  EXPECT_EQ(publisher_status(*off.graph), detail::CullReason::Skipped);
  EXPECT_FALSE(reads_ao(*off.graph));
  Settings pbrOnly;
  pbrOnly.pbr = true;
  const PublisherGraph pbr = compile_publisher(pbrOnly);
  EXPECT_EQ(publisher_status(*pbr.graph), detail::CullReason::Skipped);
  EXPECT_FALSE(reads_ao(*pbr.graph));
  settings.ao = Quality::High;
  for (const detail::GxResource missing : {detail::GxResource::Fog, detail::GxResource::Normal}) {
    detail::GraphEnvironment env = tier_b_env();
    env.gxFormats[static_cast<size_t>(missing)] = wgpu::TextureFormat::Undefined;
    const PublisherGraph noTier = compile_publisher(settings, env);
    EXPECT_EQ(publisher_status(*noTier.graph), detail::CullReason::Skipped);
    EXPECT_FALSE(reads_ao(*noTier.graph));
  }
  detail::GraphEnvironment msaa = tier_b_env();
  msaa.gxSampleCount = 4;
  EXPECT_EQ(publisher_status(*compile_publisher(settings, msaa).graph), detail::CullReason::Skipped);
  const PublisherGraph on = compile_publisher(settings);
  EXPECT_EQ(publisher_status(*on.graph), detail::CullReason::None);
  EXPECT_TRUE(reads_ao(*on.graph));
}

TEST_F(RenderGxExt, FwdEnvPublisherSkipsUnderIdentity) {
  Settings settings;
  settings.ao = Quality::High;
  fwd_env::testing::set_identity_override(true);
  const PublisherGraph graph = compile_publisher(settings);
  EXPECT_EQ(publisher_status(*graph.graph), detail::CullReason::Skipped);
  EXPECT_FALSE(reads_ao(*graph.graph));
  EXPECT_FALSE(fwd_env::forward_ao_wanted(settings, 0));
  fwd_env::testing::set_identity_override(false);
  EXPECT_TRUE(fwd_env::forward_ao_wanted(settings, 0));
  EXPECT_FALSE(fwd_env::forward_ao_wanted(settings, 1));
  EXPECT_TRUE(fwd_env::forward_ao_wanted(settings, 16));
}

TEST_F(RenderGxExt, ForwardAoIgnoresPbr) {
  fwd_env::testing::set_identity_override(false);
  Settings settings;
  settings.pbr = true;
  EXPECT_TRUE(detail::composite::wanted(settings));
  EXPECT_FALSE(detail::composite::ao_wanted(settings));
  EXPECT_FALSE(fwd_env::forward_ao_wanted(settings, 0));
  settings.ao = Quality::High;
  EXPECT_TRUE(fwd_env::forward_ao_wanted(settings, 0));
  settings.ao = Quality::Off;
  settings.legacyBlend = 1.f;
  EXPECT_FALSE(fwd_env::forward_ao_wanted(settings, 0));
  fwd_env::testing::set_identity_override(std::nullopt);
}

TEST_F(RenderGxExt, FwdEnvPublisherSortsAfterComposite) {
  Settings settings;
  settings.ao = Quality::High;
  const PublisherGraph built = compile_publisher(settings);
  const detail::CompiledGraph& graph = *built.graph;
  EXPECT_EQ(publisher_status(graph), detail::CullReason::None);
  const std::vector<detail::NodeId>& program = graph.stages[static_cast<size_t>(Stage::OpaqueDone)].nodes;
  const auto at = [&](std::string_view name) {
    return std::ranges::find(program, graph.find_node(name)) - program.begin();
  };
  ASSERT_LT(at(detail::composite::Pass), static_cast<std::ptrdiff_t>(program.size()));
  EXPECT_LT(at(detail::composite::Pass), at(fwd_env::Pass));
  EXPECT_LT(at(fwd_env::Pass), at(phase::Overlay));
  EXPECT_TRUE(graph.nodes[graph.find_node(fwd_env::Pass)].sink);
  EXPECT_TRUE(reads_ao(graph));
  const std::optional<std::string> contract = detail::composite::check_contract(graph);
  EXPECT_FALSE(contract.has_value()) << contract.value_or("");
}

gfx::RenderTargetLayout pbr_layout() {
  return gfx::compose_scene_layout({.gbuffer = true, .surface = true}, {.color = wgpu::TextureFormat::BGRA8Unorm,
                                                                        .depth = wgpu::TextureFormat::Depth32Float,
                                                                        .size = {64, 64, 1},
                                                                        .samples = 1});
}

gx::PipelineConfig pbr_writer() {
  gx::PipelineConfig config = writer(true, true);
  config.blendMode = GX_BM_NONE;
  gx::ShaderConfig& shader = config.shaderConfig;
  shader.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  shader.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  shader.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
  shader.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0};
  shader.colorChannels[GX_COLOR0] = {
      .matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .diffFn = GX_DF_CLAMP, .attnFn = GX_AF_SPOT, .lightingEnabled = true};
  shader.colorChannels[GX_ALPHA0] = {.matSrc = GX_SRC_REG};
  gx::TevStage& stage = shader.tevStages[0];
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  stage.texMapId = GX_TEXMAP0;
  stage.texCoordId = GX_TEXCOORD0;
  stage.channelId = GX_COLOR0A0;
  shader.tevStageCount = 1;
  return config;
}

scene::DrawClass main_opaque(uint32_t categories = cat::Static | cat::Terrain, uint64_t object = 1) {
  scene::DrawClass cls = draw_class(scene::Window::Opaque, object);
  cls.kind = scene::Kind::MainOpaque;
  cls.mainClass = true;
  cls.key.categories = categories;
  return cls;
}

uint8_t pbr_output(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_NE(hooks, nullptr);
  gx::render_ext::StochasticLanes lanes{};
  return hooks != nullptr
             ? hooks->derive(hooks->userData, config, gx::build_shader_info(config.shaderConfig), layout, lanes).output
             : 0xFF;
}

void pbr_frame() {
  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  apply_frame_state(frame(false, false));
}

records::DrawRecord stamped(const gx::PipelineConfig& config, const scene::DrawClass& cls) {
  gx::DrawData draw{};
  draw.instanceCount = 1;
  records::stamp(draw, config, gx::build_shader_info(config.shaderConfig), cls);
  return FakeStorage::at<records::DrawRecord>(draw.immediateData._pad);
}

gx::PipelineConfig with_output(gx::PipelineConfig config, uint8_t output) {
  config.shaderConfig.ext.output = output;
  return config;
}

TEST_F(RenderGxExt, Output2TruthTable) {
  const ScopedTargets targets;
  const ScopedHints hints;
  pbr_frame();
  ASSERT_TRUE(records::gbuffer());
  const gfx::RenderTargetLayout pbr = pbr_layout();
  const gfx::RenderTargetLayout base = scene_layout(true);
  records::on_route(main_opaque());
  ASSERT_NE(gx::render_ext::g_drawInputs & records::PbrEligible, 0u);
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 2u);
  EXPECT_EQ(pbr_output(pbr_writer(), base), 1u);
  EXPECT_EQ(pbr_output(pbr_writer(), scene_layout(false)), 0u);

  gx::PipelineConfig blendedWriter = pbr_writer();
  blendedWriter.blendMode = GX_BM_BLEND;
  blendedWriter.blendFacSrc = GX_BL_SRCALPHA;
  blendedWriter.blendFacDst = GX_BL_INVSRCALPHA;
  gx::PipelineConfig noDepth = pbr_writer();
  noDepth.depthUpdate = false;
  gx::PipelineConfig noColor = pbr_writer();
  noColor.colorUpdate = false;
  gx::PipelineConfig noNormal = pbr_writer();
  noNormal.shaderConfig.attrs[GX_VA_NRM] = {};
  gx::PipelineConfig failing = pbr_writer();
  failing.shaderConfig.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO};
  failing.shaderConfig.colorChannels[GX_ALPHA0] = failing.shaderConfig.colorChannels[GX_COLOR0];
  gx::PipelineConfig lines = pbr_writer();
  lines.shaderConfig.lineMode = 1;
  for (const gx::PipelineConfig& config : {blendedWriter, noDepth, noColor, noNormal, failing, lines}) {
    EXPECT_EQ(pbr_output(config, pbr), 1u);
  }

  gx::g_gxState.textures[0].material = {.id = 1, .flags = material::NoPbr};
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 1u);
  gx::g_gxState.textures[0].material = {};
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 2u);

  for (const uint32_t excluded :
       {cat::NoPbr, cat::Water, cat::ScreenRead, cat::Particle, cat::Ui, cat::Sky, cat::FakeShadow, cat::Overlay}) {
    records::on_route(main_opaque(cat::Static | excluded));
    EXPECT_EQ(gx::render_ext::g_drawInputs & records::PbrEligible, 0u) << excluded;
    EXPECT_EQ(pbr_output(pbr_writer(), pbr), 1u) << excluded;
  }
  scene::DrawClass blended = main_opaque();
  blended.kind = scene::Kind::MainBlended;
  records::on_route(blended);
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 1u);

  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_UNLIT", "off", true);
  unlit::testing::reset();
  apply_frame_state(frame(false, false));
  records::on_route(main_opaque());
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 1u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_UNLIT");
  unlit::testing::reset();
  apply_frame_state(frame(false, false));
  records::on_route(main_opaque());
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 2u);
}

TEST_F(RenderGxExt, PbrEligibleOnlyWithSurfaceLayout) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  records::on_route(main_opaque());
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
  EXPECT_EQ(pbr_output(pbr_writer(), pbr_layout()), 1u);

  pbr_frame();
  records::on_route(main_opaque());
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow | records::PbrEligible);

  webgpu::g_graphicsConfig.surfaceBuffer = false;
  webgpu::g_graphicsConfig.gbuffer = false;
  apply_frame_state(frame(false, false));
  records::on_route(main_opaque());
  EXPECT_EQ(gx::render_ext::g_drawInputs & records::PbrEligible, 0u);
}

TEST_F(RenderGxExt, PbrEligibleNeverInTranslucentWindow) {
  const ScopedTargets targets;
  const ScopedHints hints;
  pbr_frame();
  for (const scene::Window window : {scene::Window::PreScene, scene::Window::Translucent, scene::Window::Post}) {
    scene::DrawClass cls = main_opaque();
    cls.key.window = window;
    records::on_route(cls);
    EXPECT_EQ(gx::render_ext::g_drawInputs & records::PbrEligible, 0u) << static_cast<int>(window);
    EXPECT_NE(pbr_output(pbr_writer(), pbr_layout()), 2u) << static_cast<int>(window);
  }
  scene::DrawClass offscreen = main_opaque();
  offscreen.offscreen = true;
  records::on_route(offscreen);
  EXPECT_EQ(gx::render_ext::g_drawInputs & records::PbrEligible, 0u);
  records::on_route(forward_class());
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::ForwardWindow);
}

TEST_F(RenderGxExt, RecordBitsOnlyOnOutput2) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  scene::DrawClass cls = main_opaque(cat::Static | cat::Emissive, 5);
  cls.key.flags = scene::object_flag::BakedVertexLight;
  cls.key.tevPolicy = 0x10000;
  for (const uint8_t output : {uint8_t{0}, uint8_t{1}}) {
    const records::DrawRecord record = stamped(with_output(pbr_writer(), output), cls);
    EXPECT_EQ(record.flags & (records::flag::VertexColorMask | records::flag::Unlit), 0u) << int{output};
    EXPECT_EQ(record.flags >> 14, 0u) << int{output};
    EXPECT_EQ(record.tevPolicyOff, 0u) << int{output};
    EXPECT_EQ(record.material, 0u) << int{output};
  }
  const records::DrawRecord record = stamped(with_output(pbr_writer(), 2), cls);
  EXPECT_EQ((record.flags & records::flag::VertexColorMask) >> records::flag::VertexColorShift, 1u);
  EXPECT_NE(record.flags & records::flag::Unlit, 0u);
  EXPECT_EQ(record.flags & (records::flag::SurfaceSlotMask | records::flag::BackFaceFlip), 0u);
  EXPECT_NE(record.tevPolicyOff, 0u);
  EXPECT_EQ(record.material, material::classes::default_word(true) | gx::render_ext::MaterialWordTwoSided);
  EXPECT_EQ(record.materialOff1, 0u);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().pbrRecords, 1u);
  EXPECT_EQ(records::stats().policyBlocks, 1u);
}

uint32_t surface_slot(const records::DrawRecord& record) {
  return (record.flags & records::flag::SurfaceSlotMask) >> records::flag::SurfaceSlotShift;
}

TEST_F(RenderGxExt, SurfaceTableDedupesEntries) {
  surface_table::testing::reset();
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  surface_table::begin_frame(true);
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const records::DrawRecord first = stamped(config, main_opaque(cat::Static, 5));
  EXPECT_EQ(surface_slot(first), surface_table::StaticSlots);
  EXPECT_EQ(surface_slot(stamped(config, main_opaque(cat::Static, 6))), surface_table::StaticSlots);
  scene::DrawClass baked = main_opaque(cat::Static | cat::Emissive, 7);
  baked.key.flags = scene::object_flag::BakedVertexLight;
  EXPECT_EQ(surface_slot(stamped(config, baked)), surface_table::StaticSlots + 1);
  EXPECT_EQ(surface_slot(stamped(config, main_opaque(cat::Static, 8))), surface_table::StaticSlots);
  EXPECT_EQ(surface_slot(stamped(with_output(pbr_writer(), 1), main_opaque(cat::Static, 9))), 0u);
  EXPECT_EQ(first.flags & ~(records::flag::SurfaceSlotMask | 0xFFFFu), 0u);

  const std::shared_ptr<const surface_table::Table> table = surface_table::snapshot();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->size(), surface_table::StaticSlots + 2);
  EXPECT_EQ((*table)[0], surface_table::Entry{});
  constexpr uint32_t StaticByte = 0x01u << 16;
  constexpr uint32_t TwoSided = uint32_t{gx::surface::TwoSided} << 8;
  EXPECT_EQ((*table)[surface_table::StaticSlots],
            (surface_table::Entry{gx::surface::ClassDefault | TwoSided | StaticByte, 0, 0, 0}));
  const uint32_t flags = gx::surface::Unlit | gx::surface::BakedLight | gx::surface::TwoSided;
  EXPECT_EQ((*table)[surface_table::StaticSlots + 1],
            (surface_table::Entry{gx::surface::ClassDefault | (flags << 8) | StaticByte, 0, 0, 0}));
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().surfaceSlots, 4u);
}

TEST_F(RenderGxExt, SurfaceTableEntryFollowsReceiverAndKey) {
  constexpr uint32_t Default = gx::surface::ClassDefault;
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  using records::flag::ReceiverShift;
  constexpr uint32_t Word = material::classes::default_word(true);
  EXPECT_EQ(records::surface_entry(config, info, 2u << ReceiverShift, 0, Word)[0],
            Default | (gx::surface::CloudReceiver << 8));
  EXPECT_EQ(records::surface_entry(config, info, 1u << ReceiverShift, 0, Word)[0],
            Default | (gx::surface::NoShadowReceive << 8));
  EXPECT_EQ(records::surface_entry(config, info, 3u << ReceiverShift, 0, Word)[0], Default);
  EXPECT_EQ(records::surface_entry(config, info, records::flag::Unlit, 0, Word)[0],
            Default | (gx::surface::Unlit << 8));
  EXPECT_EQ(records::surface_entry(config, info, 2u << records::flag::VertexColorShift, 0, Word)[0], Default);
  gx::PipelineConfig keyed = config;
  keyed.shaderConfig.ext.surfaceExt = 1;
  EXPECT_EQ(records::surface_entry(keyed, info, 0, 0, Word)[0],
            Default | gx::surface::EntryExtAmbient | gx::surface::EntryExtEmissive);
  EXPECT_EQ(records::surface_entry(config, info, 0, cat::Dynamic | cat::Player, Word)[0], Default | (0x82u << 16));
  EXPECT_EQ(records::surface_entry(config, info, 0, cat::Static | cat::Terrain | cat::Emissive, Word)[0],
            Default | (0x09u << 16));
}

TEST_F(RenderGxExt, SurfaceTableStampsCategoryOnOutput2Only) {
  surface_table::testing::reset();
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  surface_table::begin_frame(true);
  const scene::DrawClass link = main_opaque(cat::Dynamic | cat::Player | cat::Skinned, 5);
  const scene::DrawClass room = main_opaque(cat::Static | cat::Terrain, 6);
  for (const uint8_t output : {uint8_t{0}, uint8_t{1}}) {
    const records::DrawRecord record = stamped(with_output(pbr_writer(), output), link);
    EXPECT_EQ(record.flags >> 16, 0u) << int{output};
  }
  const records::DrawRecord player = stamped(with_output(pbr_writer(), 2), link);
  const records::DrawRecord terrain = stamped(with_output(pbr_writer(), 2), room);
  ASSERT_NE(surface_slot(player), 0u);
  ASSERT_NE(surface_slot(terrain), 0u);
  EXPECT_NE(surface_slot(player), surface_slot(terrain));
  for (const records::DrawRecord& record : {player, terrain}) {
    EXPECT_EQ(record.flags & ~(records::flag::SurfaceSlotMask | 0xFFFFu), 0u);
  }
  const std::shared_ptr<const surface_table::Table> table = surface_table::snapshot();
  ASSERT_NE(table, nullptr);
  ASSERT_EQ(table->size(), surface_table::StaticSlots + 2);
  const gx::surface::Words record{};
  EXPECT_EQ(gx::surface::decode(record, (*table)[surface_slot(player)], {}).category, 0x86u);
  EXPECT_EQ(gx::surface::decode(record, (*table)[surface_slot(terrain)], {}).category, 0x09u);
  EXPECT_EQ(((*table)[surface_slot(player)][0] >> 16) & 0xFFu, 0x86u);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().categoryRecords, 2u);
  EXPECT_EQ(records::stats().surfaceSlots, 2u);
}

TEST_F(RenderGxExt, PostShadowEntryOnlyWhenConsumed) {
  surface_table::testing::reset();
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  surface_table::begin_frame(true);
  scene::DrawClass before = main_opaque(cat::Static | cat::Terrain, 5);
  scene::DrawClass after = before;
  after.afterFakeShadow = true;
  fakeshadow::lsm::testing::reset();
  fakeshadow::lsm::apply_frame_state();
  const records::DrawRecord off = stamped(with_output(pbr_writer(), 2), after);
  const records::DrawRecord plain = stamped(with_output(pbr_writer(), 2), before);
  EXPECT_EQ(surface_slot(off), surface_slot(plain));
  fakeshadow::lsm::testing::set_consumed(true);
  fakeshadow::lsm::apply_frame_state();
  const records::DrawRecord pre = stamped(with_output(pbr_writer(), 2), before);
  const records::DrawRecord post = stamped(with_output(pbr_writer(), 2), after);
  EXPECT_EQ(surface_slot(pre), surface_slot(plain));
  ASSERT_NE(surface_slot(post), surface_slot(plain));
  const std::shared_ptr<const surface_table::Table> table = surface_table::snapshot();
  ASSERT_NE(table, nullptr);
  EXPECT_EQ((*table)[surface_slot(post)][0], (*table)[surface_slot(plain)][0] | gx::surface::EntryPostShadow);
  for (size_t i = 1; i < 4; ++i) {
    EXPECT_EQ((*table)[surface_slot(post)][i], (*table)[surface_slot(plain)][i]) << i;
  }
  EXPECT_EQ(fakeshadow::lsm::stats().postShadowDraws, 1u);
  fakeshadow::lsm::testing::reset();
  fakeshadow::lsm::apply_frame_state();
}

TEST_F(RenderGxExt, SurfaceTableOverflowTakesSlotZero) {
  surface_table::testing::reset();
  surface_table::begin_frame(true);
  constexpr uint32_t Fifo = surface_table::Capacity - surface_table::StaticSlots;
  for (uint32_t i = 0; i < Fifo; ++i) {
    ASSERT_EQ(surface_table::slot_for({gx::surface::ClassDefault, i, 0, 0}), surface_table::StaticSlots + i);
  }
  EXPECT_EQ(surface_table::slot_for({gx::surface::ClassDefault, Fifo, 0, 0}), 0u);
  EXPECT_EQ(surface_table::slot_for({gx::surface::ClassDefault, 3, 0, 0}), surface_table::StaticSlots + 3);
  EXPECT_EQ(surface_table::snapshot()->size(), surface_table::Capacity);
  surface_table::begin_frame(true);
  EXPECT_EQ(surface_table::stats().overflow, 1u);
  EXPECT_EQ(surface_table::stats().maxEntries, Fifo);
  EXPECT_EQ(surface_table::slot_for({gx::surface::ClassDefault, Fifo, 0, 0}), surface_table::StaticSlots);
}

TEST_F(RenderGxExt, SurfaceTableNeverSlotsClassZero) {
  surface_table::testing::reset();
  surface_table::begin_frame(true);
  EXPECT_EQ(surface_table::slot_for({0, 0x123456, 0, 0}), 0u);
  EXPECT_EQ(surface_table::slot_for(surface_table::Entry{}), 0u);
  surface_table::begin_frame(false);
  EXPECT_EQ(surface_table::slot_for({gx::surface::ClassDefault, 0, 0, 0}), 0u);
  EXPECT_EQ(surface_table::snapshot(), nullptr);
}

TEST_F(RenderGxExt, SurfaceTableStaticSlotsComeFirst) {
  surface_table::testing::reset();
  const surface_table::Entry grass{2, 0, 0, 0};
  const uint32_t slot = surface_table::reserve(grass);
  EXPECT_EQ(slot, 1u);
  EXPECT_EQ(surface_table::reserve({3, 0, 0, 0}), 2u);
  surface_table::release(2);
  EXPECT_EQ(surface_table::reserve({4, 0, 0, 0}), 2u);
  surface_table::begin_frame(true);
  EXPECT_EQ(surface_table::slot_for({gx::surface::ClassDefault, 0, 0, 0}), surface_table::StaticSlots);
  const std::shared_ptr<const surface_table::Table> table = surface_table::snapshot();
  ASSERT_NE(table, nullptr);
  EXPECT_EQ((*table)[slot], grass);
  EXPECT_EQ((*table)[2], (surface_table::Entry{4, 0, 0, 0}));
  EXPECT_EQ((*table)[3], surface_table::Entry{});
  surface_table::update(slot, {5, 0, 0, 0});
  surface_table::begin_frame(true);
  EXPECT_EQ((*surface_table::snapshot())[slot], (surface_table::Entry{5, 0, 0, 0}));
  surface_table::testing::reset();
}

TEST_F(RenderGxExt, BakedVertexFlagReachesDrawInputs) {
  const ScopedTargets targets;
  const ScopedHints hints;
  pbr_frame();
  scene::DrawClass cls = main_opaque();
  records::on_route(cls);
  EXPECT_EQ(gx::render_ext::g_drawInputs & records::BakedVertex, 0u);
  cls.key.flags = scene::object_flag::BakedVertexLight;
  records::on_route(cls);
  EXPECT_NE(gx::render_ext::g_drawInputs & records::BakedVertex, 0u);
}

TEST_F(RenderGxExt, TevPolicyBlockCarriesNeutralsAndDedupes) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  scene::DrawClass cls = main_opaque(cat::Static, 5);
  cls.key.tevPolicy = 0x08010070;
  cls.key.tevNeutral[4 * 4 + 0] = 255;
  cls.key.tevNeutral[1 * 4 + 1] = 17;
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const records::DrawRecord first = stamped(config, cls);
  ASSERT_NE(first.tevPolicyOff, 0u);
  const auto block = FakeStorage::at<AuroraTevPolicy>(first.tevPolicyOff);
  EXPECT_EQ(block.mask, 0x08010070u);
  EXPECT_EQ(block.neutral[4][0], 255u);
  EXPECT_EQ(block.neutral[1][1], 17u);
  EXPECT_EQ(block.neutral[6][3], 0u);

  scene::DrawClass other = cls;
  other.key.object.id = 6;
  EXPECT_EQ(stamped(config, other).tevPolicyOff, first.tevPolicyOff);
  scene::DrawClass changed = cls;
  changed.key.tevNeutral[4 * 4 + 0] = 128;
  const records::DrawRecord third = stamped(config, changed);
  EXPECT_NE(third.tevPolicyOff, first.tevPolicyOff);
  EXPECT_EQ(FakeStorage::at<AuroraTevPolicy>(third.tevPolicyOff).neutral[4][0], 128u);
  scene::DrawClass none = cls;
  none.key.tevPolicy = 0;
  EXPECT_EQ(stamped(config, none).tevPolicyOff, 0u);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().policyBlocks, 2u);

  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_UNLIT_POLICY", "off", true);
  unlit::testing::reset();
  apply_frame_state(frame(false, false));
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  EXPECT_EQ(stamped(config, cls).tevPolicyOff, 0u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_UNLIT_POLICY");
  unlit::testing::reset();
}

TEST_F(RenderGxExt, VertexPolicyPrecedence) {
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  ASSERT_EQ(info.unlitLayerStages, 1u);
  scene::DrawClass cls = main_opaque();
  EXPECT_EQ(unlit::vertex_policy(config, info, cls), 0u);
  cls.key.flags = scene::object_flag::BakedVertexLight;
  EXPECT_EQ(unlit::vertex_policy(config, info, cls), 1u);
  for (const uint32_t vc : {1u, 2u, 3u}) {
    gx::g_gxState.textures[0].material = {.id = 1, .flags = vc << material::VertexColorShift};
    EXPECT_EQ(unlit::vertex_policy(config, info, cls), vc - 1) << vc;
  }
  gx::g_gxState.textures[1].material = {.id = 2, .flags = 3u << material::VertexColorShift};
  gx::g_gxState.textures[0].material = {};
  EXPECT_EQ(unlit::vertex_policy(config, info, cls), 1u);
  cls.key.flags = 0;
  EXPECT_EQ(unlit::vertex_policy(config, info, cls), 0u);

  EXPECT_FALSE(unlit::unlit_flag(config, info, cls));
  gx::g_gxState.textures[0].material = {.id = 1, .flags = material::Unlit};
  EXPECT_TRUE(unlit::unlit_flag(config, info, cls));
  gx::g_gxState.textures[0].material = {};
  cls.key.categories |= cat::Emissive;
  EXPECT_TRUE(unlit::unlit_flag(config, info, cls));

  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_VERTEX_COLOR", "ignore", true);
  unlit::testing::reset();
  unlit::apply_frame_state(frame(false, false));
  EXPECT_EQ(unlit::vertex_policy(config, info, cls), 2u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_VERTEX_COLOR");
  unlit::testing::reset();
}

TEST_F(RenderGxExt, PbrOffRecordsByteIdentical) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  scene::DrawClass cls = main_opaque(cat::Static | cat::Emissive, 9);
  cls.key.tevPolicy = 0x10000;
  const records::DrawRecord plain = stamped(with_output(pbr_writer(), 0), cls);
  const records::DrawRecord marked = stamped(with_output(pbr_writer(), 1), cls);
  EXPECT_EQ(std::memcmp(&plain, &marked, sizeof(plain)), 0);
  EXPECT_EQ(plain.tevPolicyOff, 0u);
  EXPECT_EQ(plain.flags & (records::flag::VertexColorMask | records::flag::Unlit), 0u);

  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  apply_frame_state(frame(false, false));
  records::on_route(main_opaque());
  EXPECT_EQ(gx::render_ext::g_drawInputs, records::InWindow);
  EXPECT_EQ(pbr_output(pbr_writer(), scene_layout(true)), 1u);
}

constexpr uint8_t NrmRole = 1u << 2;
constexpr uint8_t MatRole = 1u << 6;

detail::FrameState pbr_state(float legacyBlend, bool textureMaterials = true) {
  detail::FrameState state;
  state.settings.pbr = true;
  state.settings.textureMaterials = textureMaterials;
  state.settings.legacyBlend = legacyBlend;
  return state;
}

uint8_t pbr_layer(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_NE(hooks, nullptr);
  gx::render_ext::StochasticLanes lanes{};
  return hooks != nullptr
             ? hooks->derive(hooks->userData, config, gx::build_shader_info(config.shaderConfig), layout, lanes).layer0
             : 0xFF;
}

gx::PipelineConfig cloud_writer() {
  gx::PipelineConfig config = pbr_writer();
  gx::ShaderConfig& shader = config.shaderConfig;
  shader.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_TEXMTX1};
  gx::TevStage& cloud = shader.tevStages[1];
  cloud.colorPass = {GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO};
  cloud.texMapId = GX_TEXMAP1;
  cloud.texCoordId = GX_TEXCOORD1;
  cloud.channelId = GX_COLOR_ZERO;
  shader.tevStageCount = 2;
  return config;
}

struct CompanionScope {
  CompanionScope() {
    testing::set_companion_support(true);
    pbr_frame();
    records::on_route(main_opaque());
  }
  ~CompanionScope() { testing::set_companion_support(std::nullopt); }
  ScopedTargets targets;
  ScopedHints hints;
};

TEST_F(RenderGxExt, Layer0Rules) {
  const CompanionScope scope;
  const gfx::RenderTargetLayout pbr = pbr_layout();
  ASSERT_EQ(pbr_output(pbr_writer(), pbr), 2u);
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 0u);
  gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = NrmRole};
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 1u);
  EXPECT_EQ(pbr_layer(pbr_writer(), scene_layout(true)), 0u);
  gx::g_gxState.textures[0].material.companionRoles = 1u << 5;
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 0u);
  gx::g_gxState.textures[0].material.companionRoles = MatRole;
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 1u);
  testing::set_companion_support(false);
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 0u);
  testing::set_companion_support(true);

  const gx::PipelineConfig clouded = cloud_writer();
  ASSERT_EQ(pbr_output(clouded, pbr), 2u);
  gx::g_gxState.textures[0].material = {};
  gx::g_gxState.textures[1].material = {.companionSet = 4, .companionRoles = NrmRole};
  EXPECT_EQ(pbr_layer(clouded, pbr), 0u);
  gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = NrmRole};
  EXPECT_EQ(pbr_layer(clouded, pbr), 1u);

  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_COMPANIONS", "off", true);
  testing::reset();
  testing::set_companion_support(true);
  pbr_frame();
  records::on_route(main_opaque());
  EXPECT_EQ(pbr_output(pbr_writer(), pbr), 2u);
  EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 0u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_COMPANIONS");
  testing::reset();
}

TEST_F(RenderGxExt, Layer0SkipsFakeShadowSlots) {
  const CompanionScope scope;
  const gfx::RenderTargetLayout pbr = pbr_layout();
  const gx::PipelineConfig clouded = cloud_writer();
  gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = NrmRole};
  ASSERT_EQ(pbr_layer(clouded, pbr), 1u);
  gx::g_gxState.textures[1].material = {.id = 2, .flags = material::FakeShadow};
  EXPECT_EQ(pbr_layer(clouded, pbr), 0u);
  gx::g_gxState.textures[1].material = {};
  gx::g_gxState.textures[0].material.flags = material::FakeShadow;
  EXPECT_EQ(pbr_layer(clouded, pbr), 0u);
}

TEST_F(RenderGxExt, Layer0IndependentOfResidencyAndSettings) {
  const CompanionScope scope;
  const gfx::RenderTargetLayout pbr = pbr_layout();
  gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = NrmRole | MatRole};
  const uint8_t expected = pbr_layer(pbr_writer(), pbr);
  ASSERT_EQ(expected, 1u);
  for (const detail::FrameState& state :
       {pbr_state(0.f), pbr_state(1.f), pbr_state(0.5f, false), frame(false, false)}) {
    apply_frame_state(state);
    records::on_route(main_opaque());
    for (const gx::render_ext::CompanionGates gates :
         {gx::render_ext::CompanionGates{}, gx::render_ext::CompanionGates{true, true, true}}) {
      gx::render_ext::set_companion_gates(gates);
      gx::render_ext::store_companion_lane({.v = {15.f, 1.f, -1.f, 0.f}});
      EXPECT_EQ(pbr_layer(pbr_writer(), pbr), expected);
    }
  }
  gx::render_ext::set_companion_gates({});
  gx::render_ext::store_companion_lane({});
}

TEST_F(RenderGxExt, GateFlipInvalidatesBindings) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  apply_frame_state(frame(false, true));
  EXPECT_EQ(gx::render_ext::g_companionGates, gx::render_ext::CompanionGates{});
  uint64_t generation = gx::texture::current_bind_generation();
  apply_frame_state(pbr_state(0.f));
  EXPECT_EQ(gx::render_ext::g_companionGates, (gx::render_ext::CompanionGates{true, false, true}));
  EXPECT_NE(gx::texture::current_bind_generation(), generation);
  generation = gx::texture::current_bind_generation();
  apply_frame_state(pbr_state(0.f));
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);
  apply_frame_state(pbr_state(0.f, false));
  EXPECT_EQ(gx::render_ext::g_companionGates, (gx::render_ext::CompanionGates{false, false, true}));
  EXPECT_NE(gx::texture::current_bind_generation(), generation);
  generation = gx::texture::current_bind_generation();
  apply_frame_state(frame(false, true));
  EXPECT_EQ(gx::render_ext::g_companionGates, gx::render_ext::CompanionGates{});
  EXPECT_NE(gx::texture::current_bind_generation(), generation);
}

TEST_F(RenderGxExt, NormalsOffAtBlendOne) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  apply_frame_state(pbr_state(0.f));
  EXPECT_TRUE(gx::render_ext::g_companionGates.normals);
  apply_frame_state(pbr_state(0.99f));
  EXPECT_TRUE(gx::render_ext::g_companionGates.normals);
  apply_frame_state(pbr_state(1.f));
  EXPECT_FALSE(gx::render_ext::g_companionGates.normals);
  EXPECT_TRUE(gx::render_ext::g_companionGates.materials);
  webgpu::g_graphicsConfig.surfaceBuffer = false;
  apply_frame_state(pbr_state(0.f));
  EXPECT_FALSE(gx::render_ext::g_companionGates.normals);
  EXPECT_FALSE(gx::render_ext::g_companionGates.materials);
}

constexpr uint8_t AlbedoRole = 1u << 0;
constexpr uint8_t DelightRole = 1u << 1;
constexpr uint8_t EmitRole = 1u << 4;

gx::RenderExtConfig pbr_ext(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_NE(hooks, nullptr);
  gx::render_ext::StochasticLanes lanes{};
  return hooks != nullptr
             ? hooks->derive(hooks->userData, config, gx::build_shader_info(config.shaderConfig), layout, lanes)
             : gx::RenderExtConfig{};
}

gx::render_ext::CompanionLane params_lane(const gfx::TextureMaterialRef& material, u32 bits) {
  const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
  EXPECT_TRUE(hooks != nullptr && hooks->companion_params != nullptr);
  const gx::render_ext::CompanionLane lane = gx::render_ext::companion_lane(bits, gx::render_ext::g_companionGates);
  return hooks != nullptr && hooks->companion_params != nullptr
             ? hooks->companion_params(hooks->userData, material, lane)
             : lane;
}

void consumed_frame(float legacyBlend = 0.f) {
  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  apply_frame_state(pbr_state(legacyBlend));
}

TEST_F(RenderGxExt, Layer0PicksEmitRole) {
  const CompanionScope scope;
  const gfx::RenderTargetLayout pbr = pbr_layout();
  ASSERT_EQ(pbr_ext(pbr_writer(), pbr).surfaceExt, 0u);
  gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = EmitRole};
  gx::RenderExtConfig ext = pbr_ext(pbr_writer(), pbr);
  EXPECT_EQ(ext.layer0, 1u);
  EXPECT_EQ(ext.surfaceExt, 1u);
  gx::g_gxState.textures[0].material.companionRoles = NrmRole;
  ext = pbr_ext(pbr_writer(), pbr);
  EXPECT_EQ(ext.layer0, 1u);
  EXPECT_EQ(ext.surfaceExt, 0u);
  gx::g_gxState.textures[0].material.companionRoles = NrmRole | EmitRole;
  for (const detail::FrameState& state : {pbr_state(0.f), pbr_state(1.f), pbr_state(0.5f, false)}) {
    apply_frame_state(state);
    records::on_route(main_opaque());
    EXPECT_EQ(pbr_ext(pbr_writer(), pbr).surfaceExt, 1u);
  }
  EXPECT_EQ(pbr_ext(pbr_writer(), scene_layout(true)).surfaceExt, 0u);
}

TEST_F(RenderGxExt, DelightRoleOnlyUnderDevEnv) {
  const gfx::RenderTargetLayout pbr = pbr_layout();
  {
    const CompanionScope scope;
    gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = DelightRole};
    EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 0u);
    consumed_frame();
    EXPECT_TRUE(gx::render_ext::g_companionGates.materials);
    EXPECT_FALSE(gx::render_ext::g_companionGates.delight);
  }
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_DELIGHT", "on", true);
  testing::reset();
  {
    const CompanionScope scope;
    gx::g_gxState.textures[0].material = {.companionSet = 3, .companionRoles = DelightRole};
    EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 1u);
    EXPECT_EQ(pbr_ext(pbr_writer(), pbr).surfaceExt, 0u);
    consumed_frame();
    EXPECT_TRUE(gx::render_ext::g_companionGates.delight);
    apply_frame_state(pbr_state(0.f, false));
    EXPECT_FALSE(gx::render_ext::g_companionGates.delight);
    records::on_route(main_opaque());
    gx::g_gxState.textures[0].material.companionRoles = DelightRole | AlbedoRole;
    EXPECT_EQ(pbr_layer(pbr_writer(), pbr), 1u);
  }
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_DELIGHT");
  testing::reset();
}

TEST_F(RenderGxExt, ScalarEditRebuildsLaneOnGeneration) {
  const ScopedTargets targets;
  const ScopedHints hints;
  add(R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "normalStrength": 2 } } })");
  consumed_frame();
  const texture::TextureSourceKey key{.textureHash = 0xa001, .width = 16, .height = 16, .format = 14};
  const gfx::TextureMaterialRef ref = material::testing::gx_find(key);
  ASSERT_NE(ref.id, 0u);
  EXPECT_FLOAT_EQ(params_lane(ref, gx::render_ext::LaneNrm).v[1], 2.f);
  const uint64_t generation = gx::texture::current_bind_generation();
  consumed_frame();
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);

  const uint64_t bindEpoch = material::snapshot()->bindEpoch;
  add(R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "normalStrength": 3 } } })");
  ASSERT_EQ(material::snapshot()->bindEpoch, bindEpoch);
  EXPECT_EQ(material::testing::gx_find(key), ref);
  consumed_frame();
  EXPECT_EQ(gx::texture::current_bind_generation(), generation + 1);
  EXPECT_FLOAT_EQ(params_lane(ref, gx::render_ext::LaneNrm).v[1], 3.f);

  add(R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "normalStrength": 1 } } })");
  webgpu::g_graphicsConfig.surfaceBuffer = false;
  const uint64_t unconsumed = gx::texture::current_bind_generation();
  apply_frame_state(pbr_state(0.f));
  apply_frame_state(pbr_state(0.f));
  const uint64_t settled = gx::texture::current_bind_generation();
  add(R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "normalStrength": 4 } } })");
  apply_frame_state(pbr_state(0.f));
  EXPECT_EQ(gx::texture::current_bind_generation(), settled);
  EXPECT_GE(settled, unconsumed);
}

TEST_F(RenderGxExt, NormalScalarsReplaceRoleDefault) {
  const ScopedTargets targets;
  const ScopedHints hints;
  consumed_frame();
  using gx::render_ext::LaneAlbedo;
  using gx::render_ext::LaneMat;
  using gx::render_ext::LaneNrm;
  const gfx::TextureMaterialRef none{};
  EXPECT_EQ(params_lane(none, LaneNrm).v, (Vec4<float>{2.f, 1.f, 1.f, 0.f}));
  EXPECT_EQ(params_lane(none, LaneMat).v, (Vec4<float>{4.f, 0.5f, -1.f, 0.f}));
  EXPECT_EQ(params_lane(none, LaneAlbedo).v[1], 0.f);

  add(R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "normalStrength": 2, )"
      R"("normalFlipY": true }, "tex1_16x16_000000000000a002_14": { "normalFlipY": false } } })");
  consumed_frame();
  const gfx::TextureMaterialRef scaled =
      material::testing::gx_find({.textureHash = 0xa001, .width = 16, .height = 16, .format = 14});
  const gfx::TextureMaterialRef unflipped =
      material::testing::gx_find({.textureHash = 0xa002, .width = 16, .height = 16, .format = 14});
  ASSERT_NE(scaled.id, 0u);
  ASSERT_NE(unflipped.id, 0u);
  EXPECT_EQ(params_lane(scaled, LaneNrm | gx::render_ext::LaneOrm).v, (Vec4<float>{10.f, 2.f, -1.f, 0.f}));
  EXPECT_EQ(params_lane(scaled, LaneMat).v, (Vec4<float>{4.f, 2.f, 1.f, 0.f}));
  EXPECT_EQ(params_lane(scaled, LaneAlbedo).v[1], 0.f);
  EXPECT_EQ(params_lane(unflipped, LaneMat).v, (Vec4<float>{4.f, 0.5f, -1.f, 0.f}));

  consumed_frame(1.f);
  ASSERT_FALSE(gx::render_ext::g_companionGates.normals);
  EXPECT_EQ(params_lane(scaled, LaneNrm).v[1], 0.f);
  EXPECT_EQ(params_lane(none, LaneMat).v[1], 0.f);
}

constexpr uint32_t DefaultWord = material::classes::default_word(true);
constexpr uint32_t TwoSidedBit = gx::render_ext::MaterialWordTwoSided;

gx::PipelineConfig culled(gx::PipelineConfig config, GXCullMode cull) {
  config.cullMode = cull;
  return config;
}

unlit::DrawMaterial material_of(const gx::PipelineConfig& config, const scene::DrawClass& cls) {
  return unlit::draw_material(config, gx::build_shader_info(config.shaderConfig), cls);
}

scene::DrawClass hinted(uint8_t hint, uint32_t flags = 0) {
  scene::DrawClass cls = main_opaque();
  cls.key.materialClass = hint;
  cls.key.flags = flags;
  return cls;
}

void flip_frame() {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  records::set_back_face_flip(true);
}

bool flipped(const records::DrawRecord& record) { return (record.flags & records::flag::BackFaceFlip) != 0; }

gx::PipelineConfig ma00_writer() {
  gx::PipelineConfig config = pbr_writer();
  config.shaderConfig.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  config.shaderConfig.colorChannels[GX_COLOR0].matSrc = GX_SRC_VTX;
  return config;
}

TEST_F(RenderGxExt, MaterialWordOnlyOnOutput2) {
  records::testing::set_push(&FakeStorage::push);
  records::testing::set_active(true, true, false);
  records::testing::set_surface(true);
  for (const uint8_t output : {uint8_t{0}, uint8_t{1}}) {
    EXPECT_EQ(stamped(with_output(pbr_writer(), output), main_opaque()).material, 0u) << int{output};
  }
  EXPECT_EQ(stamped(with_output(pbr_writer(), 2), main_opaque()).material, DefaultWord | TwoSidedBit);
  EXPECT_EQ(stamped(culled(with_output(pbr_writer(), 2), GX_CULL_BACK), main_opaque()).material, DefaultWord);
  EXPECT_EQ(DefaultWord, 0x800100D9u);
  EXPECT_EQ(material::classes::default_word(false), 0x000100D9u);
}

TEST_F(RenderGxExt, TwoSidedFromCullNoneObjectOrClass) {
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  EXPECT_NE(material_of(config, main_opaque()).word & TwoSidedBit, 0u);
  const gx::PipelineConfig back = culled(config, GX_CULL_BACK);
  EXPECT_EQ(material_of(back, main_opaque()).word, DefaultWord);
  EXPECT_EQ(material_of(back, hinted(0, scene::object_flag::DoubleSided)).word, DefaultWord | TwoSidedBit);
  const uint32_t foliage = material_of(back, hinted(7)).word;
  EXPECT_EQ((foliage >> 16) & 0xFFu, 7u);
  EXPECT_EQ(foliage & gx::render_ext::MaterialWordFlagMask,
            uint32_t{gx::surface::ThinTranslucent | gx::surface::TwoSided | gx::surface::NoSsr} << 24);
  EXPECT_EQ(foliage & 0xFFu, gx::surface::unorm8(0.55f));
}

TEST_F(RenderGxExt, CullFrontWithTwoSidedClassNotFlagged) {
  flip_frame();
  const gx::PipelineConfig front = culled(with_output(pbr_writer(), 2), GX_CULL_FRONT);
  const scene::DrawClass cls = hinted(7, scene::object_flag::DoubleSided);
  const uint32_t word = material_of(front, cls).word;
  EXPECT_EQ(word & TwoSidedBit, 0u);
  EXPECT_NE(word & (uint32_t{gx::surface::ThinTranslucent} << 24), 0u);
  const records::DrawRecord record = stamped(front, cls);
  EXPECT_FALSE(flipped(record));
  EXPECT_EQ(record.material, word);
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().cullFront, 1u);
  EXPECT_EQ(records::stats().twoSided, 0u);
}

TEST_F(RenderGxExt, BackFaceFlipOnlyCullNone) {
  flip_frame();
  EXPECT_TRUE(flipped(stamped(with_output(pbr_writer(), 2), main_opaque())));
  const records::DrawRecord back =
      stamped(culled(with_output(pbr_writer(), 2), GX_CULL_BACK), hinted(0, scene::object_flag::DoubleSided));
  EXPECT_NE(back.material & TwoSidedBit, 0u);
  EXPECT_FALSE(flipped(back));
  for (const uint8_t output : {uint8_t{0}, uint8_t{1}}) {
    EXPECT_FALSE(flipped(stamped(with_output(pbr_writer(), output), main_opaque()))) << int{output};
  }
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().twoSided, 2u);
}

TEST_F(RenderGxExt, MirroredCullNoneNotFlipped) {
  flip_frame();
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  gx::g_gxState.proj[0][0] = 1.f;
  gx::g_gxState.proj[1][1] = 1.f;
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  EXPECT_TRUE(flipped(stamped(config, main_opaque())));
  gx::g_gxState.pnMtx[0].pos.m0[0] = -1.f;
  EXPECT_FALSE(flipped(stamped(config, main_opaque())));
  gx::g_gxState.pnMtx[0].pos.m0[0] = 1.f;
  gx::g_gxState.proj[1][1] = -1.f;
  EXPECT_FALSE(flipped(stamped(config, main_opaque())));
  gx::g_gxState.proj[0][0] = -1.f;
  EXPECT_TRUE(flipped(stamped(config, main_opaque())));
  gx::DrawData draw{};
  draw.immediateData.currentPnMtx = gx::MaxPnMtx + gx::MaxTexMtx;
  EXPECT_TRUE(records::mirrored(draw, config));
  draw.immediateData.currentPnMtx = gx::MaxPnMtx;
  gx::g_gxState.texMtxs[0] = {{-1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  EXPECT_TRUE(records::mirrored(draw, config));
  gx::g_gxState.proj[0][0] = 1.f;
  EXPECT_FALSE(records::mirrored(draw, config));
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().mirrored, 4u);
}

TEST_F(RenderGxExt, PaletteTwoSidedCounted) {
  flip_frame();
  gx::PipelineConfig config = with_output(pbr_writer(), 2);
  config.shaderConfig.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
  gx::g_gxState.proj[0][0] = -1.f;
  gx::g_gxState.proj[1][1] = 1.f;
  EXPECT_TRUE(flipped(stamped(config, main_opaque())));
  apply_frame_state(frame(false, false));
  EXPECT_EQ(records::stats().paletteTwoSided, 1u);
  EXPECT_EQ(records::stats().mirrored, 0u);
}

TEST_F(RenderGxExt, BackFaceFlipOnlyUnderNormalsGate) {
  const ScopedTargets targets;
  const ScopedHints hints;
  webgpu::g_graphicsConfig.gbuffer = true;
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  records::testing::set_push(&FakeStorage::push);
  const auto flips = [] {
    records::testing::set_active(true, true, false);
    records::testing::set_surface(true);
    return flipped(stamped(with_output(pbr_writer(), 2), main_opaque()));
  };
  apply_frame_state(pbr_state(0.f));
  EXPECT_TRUE(flips());
  apply_frame_state(pbr_state(1.f));
  EXPECT_FALSE(flips());
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(flips());
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES", "off", true);
  unlit::testing::reset();
  apply_frame_state(pbr_state(0.f));
  EXPECT_FALSE(flips());
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES");
  unlit::testing::reset();
  apply_frame_state(pbr_state(0.f));
  EXPECT_TRUE(flips());
}

constexpr std::string_view ScalarKey = "tex1_16x16_000000000000a001_14";

TEST_F(RenderGxExt, ManifestFactorBitsOnlyWhenSet) {
  add(R"({"version":1,"materials":{"tex1_16x16_000000000000a001_14":{"class":"ground/rock","roughness":0.25,)"
      R"("metalness":1},"tex1_16x16_000000000000a004_14":{"class":"ground/rock"}}})");
  apply_frame_state(frame(false, false));
  const gx::PipelineConfig config = culled(with_output(pbr_writer(), 2), GX_CULL_BACK);
  gx::g_gxState.textures[0].material = material_ref(ScalarKey);
  const unlit::DrawMaterial scalars = material_of(config, main_opaque());
  EXPECT_EQ(scalars.source, material::classes::Source::Manifest);
  EXPECT_EQ(scalars.word, 64u | 255u << 8 | 4u << 16 | gx::render_ext::MaterialWordRoughnessFactor |
                              gx::render_ext::MaterialWordMetalnessFactor | gx::render_ext::MaterialWordAmbient);
  gx::g_gxState.textures[0].material = material_ref(PlainKey);
  EXPECT_EQ(material_of(config, main_opaque()).word, 204u | 4u << 16 | gx::render_ext::MaterialWordAmbient);
  gx::g_gxState.textures[0].material = {};
  EXPECT_EQ(material_of(config, main_opaque()).word, DefaultWord);
}

TEST_F(RenderGxExt, ClassesOffMatchesW1Word) {
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES", "off", true);
  unlit::testing::reset();
  apply_frame_state(frame(false, false));
  surface_table::testing::reset();
  flip_frame();
  surface_table::begin_frame(true);
  gx::g_gxState.colorChannelState[0].ambColor = {0.25f, 0.5f, 0.75f, 1.f};
  for (const scene::DrawClass& cls : {main_opaque(), hinted(7, scene::object_flag::DoubleSided)}) {
    const records::DrawRecord record = stamped(with_output(pbr_writer(), 2), cls);
    EXPECT_EQ(record.material, material::classes::default_word(false));
    EXPECT_FALSE(flipped(record));
    EXPECT_EQ((*surface_table::snapshot())[surface_slot(record)],
              (surface_table::Entry{gx::surface::ClassDefault | (0x09u << 16), 0, 0, 0}));
  }
  EXPECT_FALSE(unlit::surface_ext(with_output(ma00_writer(), 2), gx::build_shader_info(ma00_writer().shaderConfig),
                                  records::BakedVertex));
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES");
  unlit::testing::reset();
}

TEST_F(RenderGxExt, ForceClassOverridesAll) {
  add(Plain);
  ASSERT_TRUE(material::classes::register_class(200, "test/coat", {}));
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES", "force:ground/rock", true);
  unlit::testing::reset();
  apply_frame_state(frame(false, false));
  const gx::PipelineConfig config = culled(with_output(pbr_writer(), 2), GX_CULL_BACK);
  gx::g_gxState.textures[0].material = material_ref(PlainKey);
  for (const scene::DrawClass& cls : {main_opaque(), hinted(200), hinted(0)}) {
    const unlit::DrawMaterial dm = material_of(config, cls);
    EXPECT_EQ((dm.word >> 16) & 0xFFu, 4u);
    EXPECT_EQ(dm.source, material::classes::Source::Force);
  }
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MATERIAL_CLASSES");
  testing::reset();
  add(Plain);
  ASSERT_TRUE(material::classes::register_class(200, "test/coat", {}));
  apply_frame_state(frame(false, false));
  gx::g_gxState.textures[0].material = material_ref(PlainKey);
  EXPECT_EQ((material_of(config, hinted(200)).word >> 16) & 0xFFu, 2u);
  gx::g_gxState.textures[0].material = {};
  EXPECT_EQ((material_of(config, hinted(200)).word >> 16) & 0xFFu, 200u);
  EXPECT_EQ((material_of(config, hinted(201)).word >> 16) & 0xFFu, 1u);
  EXPECT_EQ(material_of(config, hinted(201)).source, material::classes::Source::Hint);
}

TEST_F(RenderGxExt, SurfaceTableReadsMaterialWord) {
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  const uint32_t foliage = 0x8u | 7u << 16 | uint32_t{0x1Cu} << 24 | gx::render_ext::MaterialWordAmbient;
  const surface_table::Entry entry = records::surface_entry(config, info, records::flag::Unlit, 0, foliage);
  EXPECT_EQ(entry[0] & 0xFFu, 7u);
  EXPECT_EQ((entry[0] >> 8) & 0xFFu, 0x1Cu | gx::surface::Unlit);
  EXPECT_EQ(records::surface_entry(config, info, 0, 0, 0)[0] & 0xFFu, 0u);
  surface_table::testing::reset();
  surface_table::begin_frame(true);
  EXPECT_EQ(surface_table::slot_for(records::surface_entry(config, info, 0, 0, 0)), 0u);
  EXPECT_EQ(surface_table::slot_for(entry), surface_table::StaticSlots);
}

TEST_F(RenderGxExt, SurfaceTableAmbientFromChannelRegister) {
  const gx::PipelineConfig config = with_output(pbr_writer(), 2);
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  gx::g_gxState.colorChannelState[0].ambColor = {64.f / 255.f, 128.f / 255.f, 191.f / 255.f, 1.f};
  gx::g_gxState.colorChannelState[1].ambColor = {1.f, 1.f, 1.f, 1.f};
  EXPECT_EQ(records::surface_entry(config, info, 0, 0, DefaultWord)[1], 0xBF8040u);
  EXPECT_EQ(records::surface_entry(config, info, 0, 0, material::classes::default_word(false))[1], 0u);
  gx::PipelineConfig keyedConfig = config;
  keyedConfig.shaderConfig.ext.surfaceExt = 1;
  EXPECT_EQ(records::surface_entry(keyedConfig, info, 0, 0, DefaultWord)[1], 0u);
  gx::PipelineConfig unknown = config;
  unknown.shaderConfig.colorChannels[GX_COLOR0].ambSrc = GX_SRC_VTX;
  EXPECT_EQ(records::surface_entry(unknown, gx::build_shader_info(unknown.shaderConfig), 0, 0, DefaultWord)[1], 0u);
}

TEST_F(RenderGxExt, SurfaceTableAmbientSkipsUnlitChannel) {
  gx::PipelineConfig config = with_output(pbr_writer(), 2);
  gx::ShaderConfig& shader = config.shaderConfig;
  shader.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  shader.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_VTX};
  shader.colorChannels[GX_COLOR1] = shader.colorChannels[GX_ALPHA0];
  shader.colorChannels[GX_COLOR1] = {
      .matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .diffFn = GX_DF_CLAMP, .attnFn = GX_AF_SPOT, .lightingEnabled = true};
  gx::TevStage& lit = shader.tevStages[1];
  lit.colorPass = {GX_CC_ZERO, GX_CC_CPREV, GX_CC_RASC, GX_CC_ZERO};
  lit.texMapId = GX_TEXMAP_NULL;
  lit.texCoordId = GX_TEXCOORD_NULL;
  lit.channelId = GX_COLOR1A1;
  shader.tevStageCount = 2;
  const gx::ShaderInfo info = gx::build_shader_info(shader);
  ASSERT_TRUE(info.unlitAdmitted);
  const gx::unlit::Ambient ambient = gx::unlit::ambient_of(shader, info);
  ASSERT_EQ(ambient.source, gx::unlit::AmbientSource::Register);
  EXPECT_EQ(ambient.channel, 1u);
  gx::g_gxState.colorChannelState[0].ambColor = {1.f, 1.f, 1.f, 1.f};
  gx::g_gxState.colorChannelState[1].ambColor = {64.f / 255.f, 128.f / 255.f, 191.f / 255.f, 1.f};
  EXPECT_EQ(records::surface_entry(config, info, 0, 0, DefaultWord)[1], 0xBF8040u);
}

TEST_F(RenderGxExt, SurfaceTableBakedLightAmbientTimesVertex) {
  const ScopedTargets targets;
  const ScopedHints hints;
  pbr_frame();
  const gx::PipelineConfig config = ma00_writer();
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  const gfx::RenderTargetLayout layout = pbr_layout();
  const auto derived = [&](const scene::DrawClass& cls) {
    records::on_route(cls);
    gx::render_ext::StochasticLanes lanes{};
    const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
    return hooks->derive(hooks->userData, config, info, layout, lanes);
  };
  scene::DrawClass tint = main_opaque();
  scene::DrawClass baked = main_opaque();
  baked.key.flags = scene::object_flag::BakedVertexLight;
  const gx::RenderExtConfig tintExt = derived(tint);
  const gx::RenderExtConfig bakedExt = derived(baked);
  ASSERT_EQ(tintExt.output, 2u);
  ASSERT_EQ(bakedExt.output, 2u);
  EXPECT_EQ(tintExt.surfaceExt, 0u);
  EXPECT_EQ(bakedExt.surfaceExt, 1u);
  gx::PipelineConfig modulateWriter = pbr_writer();
  EXPECT_EQ(derived(baked).surfaceExt, 1u);
  records::on_route(baked);
  EXPECT_FALSE(unlit::surface_ext(modulateWriter, gx::build_shader_info(modulateWriter.shaderConfig),
                                  gx::render_ext::g_drawInputs));

  surface_table::testing::reset();
  surface_table::begin_frame(true);
  records::testing::set_push(&FakeStorage::push);
  gx::g_gxState.colorChannelState[0].ambColor = {64.f / 255.f, 128.f / 255.f, 191.f / 255.f, 1.f};
  gx::PipelineConfig tintConfig = config;
  tintConfig.shaderConfig.ext = tintExt;
  gx::PipelineConfig bakedConfig = config;
  bakedConfig.shaderConfig.ext = bakedExt;
  const records::DrawRecord tintRecord = stamped(tintConfig, tint);
  const records::DrawRecord bakedRecord = stamped(bakedConfig, baked);
  const std::shared_ptr<const surface_table::Table> table = surface_table::snapshot();
  ASSERT_NE(table, nullptr);
  const surface_table::Entry& tintEntry = (*table)[surface_slot(tintRecord)];
  const surface_table::Entry& bakedEntry = (*table)[surface_slot(bakedRecord)];
  EXPECT_EQ(tintEntry[1], 0xBF8040u);
  EXPECT_EQ(tintEntry[0] & gx::surface::EntryExtAmbient, 0u);
  EXPECT_EQ(bakedEntry[1], 0u);
  EXPECT_NE(bakedEntry[0] & gx::surface::EntryExtAmbient, 0u);
  EXPECT_NE((bakedEntry[0] >> 8) & gx::surface::BakedLight, 0u);
}
} // namespace
} // namespace aurora::render::gx_ext

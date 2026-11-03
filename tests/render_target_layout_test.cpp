#include <gtest/gtest.h>

#include <aurora/gfx.hpp>

#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/hash.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/scene_targets.hpp"
#include "gx/gx.hpp"
#include "gx/surface_pack.hpp"
#if AURORA_ENABLE_RENDER
#include "render/graph.hpp"
#endif
#include "webgpu/gpu.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace aurora::gfx {
namespace {

RenderTargetLayout base_layout() {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = wgpu::TextureFormat::RGBA8Unorm,
          .width = 128,
          .height = 128,
      }}},
      .depthStencilFormat = wgpu::TextureFormat::Depth24Plus,
      .sampleCount = 4,
  };
  detail::finalize_render_target_layout(layout);
  return layout;
}

TEST(RenderTargetLayoutTest, KeyChangesWithPipelineCompatibility) {
  const auto base = base_layout();

  auto colorFormat = base;
  colorFormat.colorAttachments[SceneColorAttachmentIndex].format = wgpu::TextureFormat::BGRA8Unorm;
  detail::finalize_render_target_layout(colorFormat);
  EXPECT_NE(colorFormat.key, base.key);

  auto colorCount = base;
  colorCount.colorAttachmentCount = 2;
  colorCount.colorAttachments[1] = {ColorAttachmentSemantic::Normal, wgpu::TextureFormat::RGBA16Float};
  detail::finalize_render_target_layout(colorCount);
  EXPECT_NE(colorCount.key, base.key);

  auto depthFormat = base;
  depthFormat.depthStencilFormat = wgpu::TextureFormat::Depth32Float;
  detail::finalize_render_target_layout(depthFormat);
  EXPECT_NE(depthFormat.key, base.key);

  auto sampleCount = base;
  sampleCount.sampleCount = 1;
  detail::finalize_render_target_layout(sampleCount);
  EXPECT_NE(sampleCount.key, base.key);
}

TEST(RenderTargetLayoutTest, RenderPassDerivesLayoutFromAttachmentState) {
  detail::RenderPass pass{
      .colorAttachmentCount = 2,
      .depthStencilFormat = wgpu::TextureFormat::Depth24Plus,
      .msaaSamples = 4,
  };
  pass.colorAttachments[SceneColorAttachmentIndex].semantic = ColorAttachmentSemantic::SceneColor;
  pass.colorAttachments[SceneColorAttachmentIndex].format = wgpu::TextureFormat::RGBA8Unorm;
  pass.colorAttachments[1].semantic = ColorAttachmentSemantic::Normal;
  pass.colorAttachments[1].format = wgpu::TextureFormat::RGBA16Float;

  const auto initial = pass.target_layout();
  EXPECT_EQ(initial.colorAttachmentCount, 2u);
  EXPECT_EQ(initial.colorAttachments[SceneColorAttachmentIndex].semantic, ColorAttachmentSemantic::SceneColor);
  EXPECT_EQ(initial.colorAttachments[1].format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_EQ(initial.depthStencilFormat, wgpu::TextureFormat::Depth24Plus);
  EXPECT_EQ(initial.sampleCount, 4u);

  pass.colorAttachments[1].format = wgpu::TextureFormat::RG16Float;
  EXPECT_NE(pass.target_layout().key, initial.key);
}

uint64_t reference_key(const RenderTargetLayout& layout) {
  Hasher hasher{};
  hasher.update(layout.colorAttachmentCount);
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    hasher.update(layout.colorAttachments[i].semantic);
    hasher.update(layout.colorAttachments[i].format);
  }
  hasher.update(static_cast<uint32_t>(layout.depthStencilFormat));
  hasher.update(layout.sampleCount);
  return hasher.digest();
}

TEST(RenderTargetLayoutTest, KeyMemoMatchesUncachedHash) {
  const auto base = base_layout();
  auto format = base;
  format.colorAttachments[SceneColorAttachmentIndex].format = wgpu::TextureFormat::BGRA8Unorm;
  auto count = base;
  count.colorAttachmentCount = 2;
  count.colorAttachments[1] = {ColorAttachmentSemantic::Normal, wgpu::TextureFormat::RGBA16Float};
  auto depth = base;
  depth.depthStencilFormat = wgpu::TextureFormat::Depth32Float;
  auto samples = base;
  samples.sampleCount = 1;
  auto size = base;
  size.colorAttachments[SceneColorAttachmentIndex].width = 64;
  size.colorAttachments[SceneColorAttachmentIndex].height = 32;
  auto secondSemantic = count;
  secondSemantic.colorAttachments[1].semantic = ColorAttachmentSemantic::Auxiliary;
  auto secondFormat = count;
  secondFormat.colorAttachments[1].format = wgpu::TextureFormat::RG16Float;

  // Each variant runs right after the layout it differs from, so the memo still holds that layout
  const std::array sequence{base, format, base, count, base,           depth, base,         samples,
                            base, size,   base, count, secondSemantic, count, secondFormat, count};
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < sequence.size(); ++i) {
    auto layout = sequence[i];
    layout.key = 0;
    detail::finalize_render_target_layout(layout);
    EXPECT_EQ(layout.key, reference_key(layout)) << i;
    keys.push_back(layout.key);
  }
  EXPECT_EQ(keys[9], keys[0]);
  for (const size_t variant : {1u, 3u, 5u, 7u}) {
    EXPECT_NE(keys[variant], keys[0]) << variant;
  }
  EXPECT_NE(keys[12], keys[11]);
  EXPECT_NE(keys[14], keys[11]);
}

TEST(RenderTargetLayoutTest, DebugIdTierCosts12Bytes) {
  auto& config = webgpu::g_graphicsConfig;
  const webgpu::GraphicsConfig savedConfig = config;
  const webgpu::TextureWithSampler savedFrame = webgpu::g_frameBuffer;
  config.surfaceConfiguration.format = wgpu::TextureFormat::RGBA8Unorm;
  config.depthFormat = wgpu::TextureFormat::Depth32Float;
  config.msaaSamples = 1;
  config.normalBuffer = false;
  config.debugIdBuffer = false;
  webgpu::g_frameBuffer.size = {640, 480, 1};
  webgpu::g_frameBuffer.format = wgpu::TextureFormat::RGBA8Unorm;
  webgpu::g_normalBuffer.size = {640, 480, 1};
  webgpu::g_normalBuffer.format = webgpu::NormalBufferFormat;
  webgpu::g_debugIdBuffer.size = {640, 480, 1};
  webgpu::g_debugIdBuffer.format = webgpu::DebugIdBufferFormat;
#if AURORA_ENABLE_RENDER
  const auto cost = [](const RenderTargetLayout& layout) {
    std::vector<wgpu::TextureFormat> formats;
    for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
      formats.push_back(layout.colorAttachments[i].format);
    }
    return render::detail::color_bytes_per_sample(formats);
  };
#endif

  const auto off = scene_render_target_layout();
  ASSERT_EQ(off.colorAttachmentCount, 1u);
  EXPECT_EQ(off.key, reference_key(off));
  EXPECT_EQ(pipeline_base_layout(off).key, off.key);
#if AURORA_ENABLE_RENDER
  EXPECT_EQ(cost(off), 8u);
#endif

  config.debugIdBuffer = true;
  const auto on = scene_render_target_layout();
  ASSERT_EQ(on.colorAttachmentCount, 2u);
  EXPECT_EQ(on.colorAttachments[0].semantic, ColorAttachmentSemantic::SceneColor);
  EXPECT_EQ(on.colorAttachments[1].semantic, ColorAttachmentSemantic::DebugId);
  EXPECT_EQ(on.colorAttachments[1].format, wgpu::TextureFormat::R32Uint);
  EXPECT_NE(on.key, off.key);
  EXPECT_EQ(pipeline_base_layout(on).key, off.key);
#if AURORA_ENABLE_RENDER
  EXPECT_EQ(cost(on), 12u);
#endif

  config.normalBuffer = true;
  const auto both = scene_render_target_layout();
  ASSERT_EQ(both.colorAttachmentCount, 3u);
  EXPECT_EQ(both.colorAttachments[1].semantic, ColorAttachmentSemantic::Normal);
  EXPECT_EQ(both.colorAttachments[2].semantic, ColorAttachmentSemantic::DebugId);
#if AURORA_ENABLE_RENDER
  EXPECT_EQ(cost(both), 20u);
#endif
  config.debugIdBuffer = false;
  EXPECT_EQ(pipeline_base_layout(both).key, scene_render_target_layout().key);

  config = savedConfig;
  webgpu::g_frameBuffer = savedFrame;
  webgpu::g_normalBuffer = {};
  webgpu::g_debugIdBuffer = {};
}

constexpr SceneLayoutBase TestBase{
    .color = wgpu::TextureFormat::RGBA8Unorm,
    .depth = wgpu::TextureFormat::Depth32Float,
    .size = {640, 480, 1},
    .samples = 1,
};

RenderTargetLayout legacy_scene_layout(bool normal) {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = TestBase.color,
          .width = TestBase.size.width,
          .height = TestBase.size.height,
      }}},
      .depthStencilFormat = TestBase.depth,
      .sampleCount = TestBase.samples,
  };
  if (normal) {
    layout.colorAttachments[layout.colorAttachmentCount++] = {
        .semantic = ColorAttachmentSemantic::Normal,
        .format = webgpu::NormalBufferFormat,
        .width = TestBase.size.width,
        .height = TestBase.size.height,
    };
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}

void expect_same_layout(const RenderTargetLayout& actual, const RenderTargetLayout& expected) {
  EXPECT_EQ(actual.key, expected.key);
  ASSERT_EQ(actual.colorAttachmentCount, expected.colorAttachmentCount);
  for (uint32_t i = 0; i < actual.colorAttachmentCount; ++i) {
    EXPECT_EQ(actual.colorAttachments[i].semantic, expected.colorAttachments[i].semantic) << i;
    EXPECT_EQ(actual.colorAttachments[i].format, expected.colorAttachments[i].format) << i;
    EXPECT_EQ(actual.colorAttachments[i].width, expected.colorAttachments[i].width) << i;
    EXPECT_EQ(actual.colorAttachments[i].height, expected.colorAttachments[i].height) << i;
  }
  EXPECT_EQ(actual.depthStencilFormat, expected.depthStencilFormat);
  EXPECT_EQ(actual.sampleCount, expected.sampleCount);
}

TEST(RenderTargetLayoutTest, SceneLayoutComposesBaseGBuffer) {
  const RenderTargetLayout gbuffer = compose_scene_layout({.gbuffer = true}, TestBase);
  ASSERT_EQ(gbuffer.colorAttachmentCount, 5u);
  constexpr std::array Semantics{ColorAttachmentSemantic::SceneColor, ColorAttachmentSemantic::Normal,
                                 ColorAttachmentSemantic::Motion, ColorAttachmentSemantic::Reactive,
                                 ColorAttachmentSemantic::Fog};
  constexpr std::array Formats{wgpu::TextureFormat::RGBA8Unorm, wgpu::TextureFormat::RGB10A2Unorm,
                               wgpu::TextureFormat::RG16Float, wgpu::TextureFormat::R8Unorm,
                               wgpu::TextureFormat::RGBA8Uint};
  for (uint32_t i = 0; i < Semantics.size(); ++i) {
    EXPECT_EQ(gbuffer.colorAttachments[i].semantic, Semantics[i]) << i;
    EXPECT_EQ(gbuffer.colorAttachments[i].format, Formats[i]) << i;
    EXPECT_EQ(gbuffer.colorAttachments[i].width, 640u) << i;
    EXPECT_EQ(gbuffer.colorAttachments[i].height, 480u) << i;
  }
  EXPECT_EQ(gbuffer.key, reference_key(gbuffer));
  EXPECT_TRUE(layout_has_gbuffer(gbuffer));
  EXPECT_EQ(compose_scene_layout({.normal = true, .gbuffer = true}, TestBase).key, gbuffer.key);

  const RenderTargetLayout legacy = compose_scene_layout({}, TestBase);
  const RenderTargetLayout modNormal = compose_scene_layout({.normal = true}, TestBase);
  expect_same_layout(legacy, legacy_scene_layout(false));
  expect_same_layout(modNormal, legacy_scene_layout(true));
  EXPECT_FALSE(layout_has_gbuffer(legacy));
  EXPECT_FALSE(layout_has_gbuffer(modNormal));

  const RenderTargetLayout ids = compose_scene_layout({.gbuffer = true, .debugId = true}, TestBase);
  ASSERT_EQ(ids.colorAttachmentCount, 6u);
  EXPECT_EQ(ids.colorAttachments[4].semantic, ColorAttachmentSemantic::Fog);
  EXPECT_EQ(ids.colorAttachments[5].semantic, ColorAttachmentSemantic::DebugId);
  EXPECT_EQ(pipeline_base_layout(ids).key, gbuffer.key);
  EXPECT_FALSE(layout_has_gbuffer(compose_scene_layout({.debugId = true}, TestBase)));

  EXPECT_TRUE(clears_with_depth(ColorAttachmentSemantic::Normal));
  EXPECT_TRUE(clears_with_depth(ColorAttachmentSemantic::Motion));
  EXPECT_TRUE(clears_with_color(ColorAttachmentSemantic::Reactive));
  EXPECT_TRUE(clears_with_color(ColorAttachmentSemantic::Fog));
  for (const auto semantic : {ColorAttachmentSemantic::SceneColor, ColorAttachmentSemantic::DebugId}) {
    EXPECT_FALSE(clears_with_depth(semantic));
    EXPECT_FALSE(clears_with_color(semantic));
  }

  auto& config = webgpu::g_graphicsConfig;
  const webgpu::GraphicsConfig savedConfig = config;
  const webgpu::TextureWithSampler savedFrame = webgpu::g_frameBuffer;
  config.depthFormat = TestBase.depth;
  config.msaaSamples = 1;
  config.normalBuffer = false;
  config.debugIdBuffer = false;
  config.gbuffer = true;
  webgpu::g_frameBuffer.size = TestBase.size;
  webgpu::g_frameBuffer.format = TestBase.color;
  EXPECT_EQ(live_scene_targets(), (SceneTargetFlags{.normal = true, .gbuffer = true}));
  expect_same_layout(scene_render_target_layout(), gbuffer);
  config = savedConfig;
  webgpu::g_frameBuffer = savedFrame;
}

TEST(RenderTargetLayoutTest, SceneLayoutComposesPbrSurface) {
  static_assert(static_cast<uint8_t>(ColorAttachmentSemantic::Fog) == 6);
  static_assert(static_cast<uint8_t>(ColorAttachmentSemantic::Surface) == 7);
  static_assert(static_cast<uint8_t>(ColorAttachmentSemantic::SurfaceExt) == 8);
  const RenderTargetLayout pbr = compose_scene_layout({.gbuffer = true, .surface = true}, TestBase);
  ASSERT_EQ(pbr.colorAttachmentCount, 7u);
  constexpr std::array Semantics{ColorAttachmentSemantic::SceneColor, ColorAttachmentSemantic::Normal,
                                 ColorAttachmentSemantic::Motion,     ColorAttachmentSemantic::Reactive,
                                 ColorAttachmentSemantic::Fog,        ColorAttachmentSemantic::Surface,
                                 ColorAttachmentSemantic::SurfaceExt};
  constexpr std::array Formats{wgpu::TextureFormat::RGBA8Unorm, wgpu::TextureFormat::RGB10A2Unorm,
                               wgpu::TextureFormat::RG16Float,  wgpu::TextureFormat::R8Unorm,
                               wgpu::TextureFormat::RGBA8Uint,  wgpu::TextureFormat::RGBA16Uint,
                               wgpu::TextureFormat::RGBA16Uint};
  for (uint32_t i = 0; i < Semantics.size(); ++i) {
    EXPECT_EQ(pbr.colorAttachments[i].semantic, Semantics[i]) << i;
    EXPECT_EQ(pbr.colorAttachments[i].format, Formats[i]) << i;
    EXPECT_EQ(pbr.colorAttachments[i].width, 640u) << i;
    EXPECT_EQ(pbr.colorAttachments[i].height, 480u) << i;
  }
  EXPECT_EQ(pbr.key, reference_key(pbr));
  EXPECT_TRUE(layout_has_gbuffer(pbr));
  EXPECT_TRUE(layout_has_surface(pbr));
  EXPECT_EQ(compose_scene_layout({.surface = true}, TestBase).key, pbr.key);
  EXPECT_EQ(compose_scene_layout({.normal = true, .gbuffer = true, .surface = true}, TestBase).key, pbr.key);

  const RenderTargetLayout ids = compose_scene_layout({.gbuffer = true, .surface = true, .debugId = true}, TestBase);
  ASSERT_EQ(ids.colorAttachmentCount, 8u);
  EXPECT_EQ(ids.colorAttachments[5].semantic, ColorAttachmentSemantic::Surface);
  EXPECT_EQ(ids.colorAttachments[6].semantic, ColorAttachmentSemantic::SurfaceExt);
  EXPECT_EQ(ids.colorAttachments[7].semantic, ColorAttachmentSemantic::DebugId);
  EXPECT_EQ(pipeline_base_layout(ids).key, pbr.key);

  const RenderTargetLayout gbuffer = compose_scene_layout({.gbuffer = true}, TestBase);
  EXPECT_FALSE(layout_has_surface(gbuffer));
  EXPECT_NE(gbuffer.key, pbr.key);
  expect_same_layout(compose_scene_layout({}, TestBase), legacy_scene_layout(false));
  expect_same_layout(compose_scene_layout({.normal = true}, TestBase), legacy_scene_layout(true));
  EXPECT_FALSE(layout_has_surface(compose_scene_layout({.normal = true, .debugId = true}, TestBase)));

  EXPECT_TRUE(clears_with_either(ColorAttachmentSemantic::Surface));
  EXPECT_FALSE(clears_with_depth(ColorAttachmentSemantic::Surface));
  EXPECT_FALSE(clears_with_color(ColorAttachmentSemantic::Surface));
  for (const auto semantic :
       {ColorAttachmentSemantic::SceneColor, ColorAttachmentSemantic::Normal, ColorAttachmentSemantic::Motion,
        ColorAttachmentSemantic::Reactive, ColorAttachmentSemantic::Fog, ColorAttachmentSemantic::DebugId}) {
    EXPECT_FALSE(clears_with_either(semantic)) << static_cast<int>(semantic);
  }

  auto& config = webgpu::g_graphicsConfig;
  const webgpu::GraphicsConfig savedConfig = config;
  const webgpu::TextureWithSampler savedFrame = webgpu::g_frameBuffer;
  config.depthFormat = TestBase.depth;
  config.msaaSamples = 1;
  config.normalBuffer = false;
  config.debugIdBuffer = false;
  config.gbuffer = true;
  config.surfaceBuffer = true;
  webgpu::g_frameBuffer.size = TestBase.size;
  webgpu::g_frameBuffer.format = TestBase.color;
  EXPECT_EQ(webgpu::scene_tier(), SceneTier::Pbr);
  EXPECT_EQ(live_scene_targets(), (SceneTargetFlags{.normal = true, .gbuffer = true, .surface = true}));
  expect_same_layout(scene_render_target_layout(), pbr);
  config.surfaceBuffer = false;
  EXPECT_EQ(webgpu::scene_tier(), SceneTier::Base);
  config.gbuffer = false;
  EXPECT_EQ(webgpu::scene_tier(), SceneTier::Legacy);
  config = savedConfig;
  webgpu::g_frameBuffer = savedFrame;
}

TEST(RenderTargetLayoutTest, PbrShaderTargetsMatchBase) {
  const gx::ShaderTargets base = gx::shader_targets(compose_scene_layout({.gbuffer = true}, TestBase));
  const gx::ShaderTargets pbr = gx::shader_targets(compose_scene_layout({.gbuffer = true, .surface = true}, TestBase));
  EXPECT_EQ(pbr.normal, base.normal);
  EXPECT_EQ(pbr.gbufferNormal, base.gbufferNormal);
  EXPECT_EQ(pbr.motion, base.motion);
  EXPECT_EQ(pbr.reactive, base.reactive);
  EXPECT_EQ(pbr.fog, base.fog);
  EXPECT_EQ(pbr.debugId, base.debugId);
  EXPECT_EQ(pbr.fwdEnv, base.fwdEnv);
  EXPECT_TRUE(pbr.fwdEnv);
  EXPECT_EQ(gx::shader_targets(compose_scene_layout({.gbuffer = true, .debugId = true}, TestBase)).debugId, 5u);
  EXPECT_EQ(
      gx::shader_targets(compose_scene_layout({.gbuffer = true, .surface = true, .debugId = true}, TestBase)).debugId,
      7u);
  EXPECT_EQ(pbr.surface, 5u);
  EXPECT_EQ(pbr.surfaceExt, 6u);
  EXPECT_EQ(base.surfaceExt, UINT32_MAX);
}

uint32_t ufloat_code(uint32_t word, uint32_t channel) noexcept {
  return channel == 2 ? word >> 22 : (word >> (11 * channel)) & 0x7FFu;
}

TEST(SurfaceV2, MirrorRoundTrips) {
  namespace s = gx::surface;
  static_assert(s::LayoutVersion == 2);
  std::mt19937 rng{0x2u};
  std::uniform_int_distribution<uint32_t> byte{0, 255};
  std::uniform_int_distribution<uint32_t> word;
  for (uint32_t i = 0; i < 4160; ++i) {
    const auto u = [&] { return s::unorm8_decode(byte(rng)); };
    s::Fields f{.albedo = {u(), u(), u()},
                .ao = u(),
                .roughness = u(),
                .metalness = u(),
                .classId = static_cast<uint8_t>(byte(rng)),
                .flags = static_cast<uint8_t>(byte(rng)),
                .ambient = {u(), u(), u()},
                .category = static_cast<uint8_t>(byte(rng)),
                .emissive = s::unpack_r11g11b10(word(rng))};
    const uint32_t slot = word(rng) & s::SlotMax;
    const uint32_t extBits = (i & 1 ? s::EntryExtAmbient : 0u) | (i & 2 ? s::EntryExtEmissive : 0u);
    const s::Words record = s::pack_record(f, slot);
    EXPECT_EQ(s::join(s::split(record)), record);
    EXPECT_EQ(s::record_slot(record), slot);
    const s::Fields d = s::decode(record, s::pack_entry(f, extBits), s::pack_ext(f));
    EXPECT_EQ(s::pack(d), s::pack(f)) << i;
    EXPECT_EQ(s::expand_v1(record, s::pack_entry(f, extBits), s::pack_ext(f)), s::pack(f)) << i;
  }
}

TEST(SurfaceV2, EntryBitsAreFrozen) {
  namespace s = gx::surface;
  const s::Fields f{
      .classId = 0x12, .flags = 0x34, .ambient = {1.f, 0.f, 1.f}, .category = 0x56, .emissive = {1.f, 0.f, 0.f}};
  EXPECT_EQ(s::pack_entry(f, s::EntryExtAmbient), (s::Entry{0x01563412u, 0x00FF00FFu, 0x3C0u, 0}));
  EXPECT_EQ(s::pack_entry(f, ~0u)[0], 0x03563412u);
  EXPECT_EQ(s::split({0x44332211u, 0x88776655u}), (s::Channels{0x2211, 0x4433, 0x6655, 0x8877}));
  EXPECT_EQ(s::pack_record({.roughness = 1.f, .metalness = 0.f}, 0x1234)[1], 0x123400FFu);
  const s::Words ext{0x00102030u, 0x3C0u};
  const s::Entry withAmbient{s::ClassDefault | s::EntryExtAmbient, 0x00ABCDEFu, 0, 0};
  EXPECT_EQ(s::pack(s::decode({}, withAmbient, ext))[2], 0x00102030u);
  EXPECT_EQ(s::pack(s::decode({}, {s::ClassDefault, 0x00ABCDEFu, 0, 0}, ext))[2], 0x00ABCDEFu);
}

TEST(RenderTargetLayoutTest, SurfaceMirrorRoundTrips) {
  namespace s = gx::surface;
  EXPECT_EQ(std::bit_cast<uint32_t>(s::InvUnorm8), 0x3B808081u);
  for (uint32_t b = 0; b < 256; ++b) {
    EXPECT_EQ(std::bit_cast<uint32_t>(s::unorm8_decode(b)),
              std::bit_cast<uint32_t>(static_cast<float>(b) * s::InvUnorm8))
        << b;
    EXPECT_EQ(s::unorm8(s::unorm8_decode(b)), b) << b;
  }
  for (const uint32_t bits : {0x7FC00000u, 0x7F800001u, 0xFFC00000u, 0x80000000u, 0xBF800000u, 0xFF800000u}) {
    EXPECT_EQ(s::unorm8(std::bit_cast<float>(bits)), 0u) << std::hex << bits;
  }
  for (const float value : {2.f, 1e6f, std::numeric_limits<float>::infinity()}) {
    EXPECT_EQ(s::unorm8(value), 255u) << value;
  }

  for (uint32_t code = 0x040; code <= s::RgMax; ++code) {
    for (uint32_t channel = 0; channel < 2; ++channel) {
      const uint32_t word = code << (11 * channel);
      EXPECT_EQ(s::pack_r11g11b10(s::unpack_r11g11b10(word)), word) << code << " " << channel;
    }
  }
  for (uint32_t code = 0x020; code <= s::BMax; ++code) {
    EXPECT_EQ(s::pack_r11g11b10(s::unpack_r11g11b10(code << 22)), code << 22) << code;
  }
  const auto code_of = [](float value, uint32_t channel) {
    std::array<float, 3> v{};
    v[channel] = value;
    return ufloat_code(s::pack_r11g11b10(v), channel);
  };
  for (const uint32_t bits : {0u, 0x80000000u, 0xBF800000u, 0x7FC00000u, 0xFF800000u, 0x38000000u}) {
    for (uint32_t channel = 0; channel < 3; ++channel) {
      EXPECT_EQ(code_of(std::bit_cast<float>(bits), channel), 0u) << std::hex << bits << " " << channel;
    }
  }
  const float minNormal = std::bit_cast<float>(0x38800000u);
  EXPECT_EQ(code_of(minNormal, 0), 0x040u);
  EXPECT_EQ(code_of(minNormal, 2), 0x020u);
  EXPECT_EQ(code_of(1.f, 1), 0x3C0u);
  EXPECT_EQ(code_of(1.f, 2), 0x1E0u);
  EXPECT_EQ(code_of(1.9921875f, 0), 0x400u);
  EXPECT_EQ(code_of(65024.f, 0), s::RgMax);
  for (uint32_t value = 65025; value <= 65536; ++value) {
    EXPECT_EQ(code_of(static_cast<float>(value), 1), s::RgMax) << value;
  }
  EXPECT_EQ(code_of(std::numeric_limits<float>::infinity(), 0), s::RgMax);
  EXPECT_EQ(code_of(64512.f, 2), s::BMax);
  for (const float value : {64513.f, 65024.f, 1e9f, std::numeric_limits<float>::infinity()}) {
    EXPECT_EQ(code_of(value, 2), s::BMax) << value;
  }

  std::mt19937 rng{0x5u};
  std::uniform_int_distribution<uint32_t> word;
  const auto legal_code = [&](uint32_t low, uint32_t high) {
    const uint32_t code = std::uniform_int_distribution<uint32_t>{low - 1, high}(rng);
    return code == low - 1 ? 0u : code;
  };
  for (int i = 0; i < 4096; ++i) {
    const std::array<uint32_t, 4> words{word(rng), word(rng), word(rng),
                                        legal_code(0x040, s::RgMax) | (legal_code(0x040, s::RgMax) << 11) |
                                            (legal_code(0x020, s::BMax) << 22)};
    EXPECT_EQ(s::pack(s::unpack(words)), words) << i;
  }
  const s::Fields fields{.albedo = {1.f, 0.5f, 0.f},
                         .ao = 1.f,
                         .roughness = 0.85f,
                         .classId = s::ClassDefault,
                         .flags = s::Unlit | s::BakedLight,
                         .category = 0x81,
                         .emissive = {1.f, 0.f, 2.f}};
  const std::array<uint32_t, 4> packed = s::pack(fields);
  EXPECT_EQ(packed[0], 0xFF0080FFu);
  EXPECT_EQ(packed[1], 0x600100D9u);
  EXPECT_EQ(packed[2], 0x81000000u);
  EXPECT_EQ(packed[3], 0x3C0u | (0x200u << 22));
}

#if AURORA_ENABLE_RENDER
uint32_t layout_cost(const RenderTargetLayout& layout) {
  std::vector<wgpu::TextureFormat> formats;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    formats.push_back(layout.colorAttachments[i].format);
  }
  return render::detail::color_bytes_per_sample(formats);
}

TEST(RenderTargetLayoutTest, SceneTargetCostsFollowTiers) {
  EXPECT_EQ(layout_cost(compose_scene_layout({}, TestBase)), 8u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.normal = true}, TestBase)), 16u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.gbuffer = true}, TestBase)), webgpu::BaseSceneColorBytes);
  EXPECT_EQ(layout_cost(compose_scene_layout({.debugId = true}, TestBase)), 12u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.normal = true, .debugId = true}, TestBase)), 20u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.gbuffer = true, .debugId = true}, TestBase)), 32u);
  const RenderTargetLayout pbr = compose_scene_layout({.gbuffer = true, .surface = true}, TestBase);
  EXPECT_EQ(layout_cost(pbr), webgpu::PbrSceneColorBytes);
  EXPECT_EQ(layout_cost(compose_scene_layout({.gbuffer = true, .surface = true, .debugId = true}, TestBase)), 48u);
  EXPECT_EQ(compose_scene_layout({.surface = true}, TestBase).key, pbr.key);
  EXPECT_EQ(compose_scene_layout({.normal = true, .gbuffer = true, .surface = true}, TestBase).key, pbr.key);
}
#endif

RenderTargetLayout clear_layout(bool normal) {
  RenderTargetLayout layout = base_layout();
  if (normal) {
    layout.colorAttachments[layout.colorAttachmentCount++] = {ColorAttachmentSemantic::Normal,
                                                              webgpu::NormalBufferFormat};
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}

std::string legacy_clear_source(const clear::PipelineConfig& config, bool normal) {
  std::string source{R"""(
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, 1.0),
    vec2(-1.0, -3.0),
    vec2(3.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vtxIdx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[vtxIdx], 0.0, 1.0);
    return out;
}
)"""};
  std::string outputs;
  std::string values;
  if (config.clearColor || config.clearAlpha) {
    outputs += "    @location(0) color: vec4f,\n";
    values += "    out.color = vec4f(1.0);\n";
  }
  if (normal && config.clearDepth) {
    outputs += "    @location(1) normal1: vec4f,\n";
    values += "    out.normal1 = vec4f(0.0);\n";
  }
  if (outputs.empty()) {
    return source + "\n@fragment\nfn fs_main() {\n}\n";
  }
  return source + "\nstruct FragmentOutput {\n" + outputs +
         "};\n@fragment\nfn fs_main() -> FragmentOutput {\n    var out: FragmentOutput;\n" + values +
         "    return out;\n}\n";
}

clear::PipelineConfig clear_config(uint32_t bits) {
  return {.clearColor = (bits & 1) != 0, .clearAlpha = (bits & 2) != 0, .clearDepth = (bits & 4) != 0};
}

TEST(RenderTargetLayoutTest, LegacyClearShaderTextUnchanged) {
  for (const bool normal : {false, true}) {
    for (uint32_t bits = 0; bits < 8; ++bits) {
      const clear::PipelineConfig config = clear_config(bits);
      EXPECT_EQ(clear::detail::shader_source(config, clear_layout(normal)), legacy_clear_source(config, normal))
          << normal << " " << bits;
    }
  }
}

TEST(RenderTargetLayoutTest, ClearShaderWritesSurfaceOnEitherClear) {
  const RenderTargetLayout layout = compose_scene_layout({.gbuffer = true, .surface = true}, TestBase);
  for (const uint32_t bits : {4u, 1u, 3u, 5u, 7u}) {
    const std::string source = clear::detail::shader_source(clear_config(bits), layout);
    EXPECT_NE(source.find("@location(5) t5: vec4u,"), std::string::npos) << bits;
    EXPECT_NE(source.find("out.t5 = vec4u(0u);"), std::string::npos) << bits;
  }
  for (const uint32_t bits : {0u, 2u}) {
    EXPECT_EQ(clear::detail::shader_source(clear_config(bits), layout).find("t5"), std::string::npos) << bits;
  }
  const std::string depth = clear::detail::shader_source(clear_config(4), layout);
  EXPECT_EQ(depth.find("t4"), std::string::npos);
  const std::string color = clear::detail::shader_source(clear_config(1), layout);
  EXPECT_NE(color.find("@location(4) t4: vec4u,"), std::string::npos);
  EXPECT_NE(color.find("out.t4 = vec4u(0u);"), std::string::npos);
  EXPECT_EQ(color.find("normal1"), std::string::npos);
  const RenderTargetLayout gbuffer = compose_scene_layout({.gbuffer = true}, TestBase);
  for (uint32_t bits = 0; bits < 8; ++bits) {
    EXPECT_EQ(clear::detail::shader_source(clear_config(bits), gbuffer).find("t5"), std::string::npos) << bits;
  }
}

TEST(RenderTargetLayoutTest, ClearShaderWritesGBufferTargets) {
  const RenderTargetLayout layout = compose_scene_layout({.gbuffer = true}, TestBase);
  const std::string depth = clear::detail::shader_source(clear_config(4), layout);
  EXPECT_NE(depth.find("@location(1) normal1: vec4f,"), std::string::npos);
  EXPECT_NE(depth.find("@location(2) t2: vec4f,"), std::string::npos);
  EXPECT_NE(depth.find("out.t2 = vec4f(0.0);"), std::string::npos);
  EXPECT_EQ(depth.find("t3"), std::string::npos);
  EXPECT_EQ(depth.find("t4"), std::string::npos);
  EXPECT_EQ(depth.find("color"), std::string::npos);

  const std::string color = clear::detail::shader_source(clear_config(1), layout);
  EXPECT_NE(color.find("@location(0) color: vec4f,"), std::string::npos);
  EXPECT_NE(color.find("@location(3) t3: vec4f,"), std::string::npos);
  EXPECT_NE(color.find("out.t3 = vec4f(0.0);"), std::string::npos);
  EXPECT_NE(color.find("@location(4) t4: vec4u,"), std::string::npos);
  EXPECT_NE(color.find("out.t4 = vec4u(0u);"), std::string::npos);
  EXPECT_EQ(color.find("normal1"), std::string::npos);
  EXPECT_EQ(color.find("t2"), std::string::npos);

  const std::string alpha = clear::detail::shader_source(clear_config(2), layout);
  EXPECT_NE(alpha.find("@location(0) color: vec4f,"), std::string::npos);
  EXPECT_EQ(alpha.find("t3"), std::string::npos);
  EXPECT_EQ(alpha.find("t4"), std::string::npos);

  const std::string neither = clear::detail::shader_source(clear_config(0), layout);
  EXPECT_EQ(neither.find("FragmentOutput"), std::string::npos);
  EXPECT_NE(neither.find("fn fs_main() {"), std::string::npos);
}

} // namespace
} // namespace aurora::gfx

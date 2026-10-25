#include <gtest/gtest.h>

#include <aurora/gfx.hpp>

#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/hash.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/scene_targets.hpp"
#if AURORA_ENABLE_RENDER
#include "render/graph.hpp"
#endif
#include "webgpu/gpu.hpp"

#include <array>
#include <cstdint>
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
  EXPECT_EQ(layout_cost(compose_scene_layout({.gbuffer = true}, TestBase)), 25u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.debugId = true}, TestBase)), 12u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.normal = true, .debugId = true}, TestBase)), 20u);
  EXPECT_EQ(layout_cost(compose_scene_layout({.gbuffer = true, .debugId = true}, TestBase)), 32u);

  std::vector<wgpu::TextureFormat> pbr{wgpu::TextureFormat::RGBA8Unorm, wgpu::TextureFormat::RGB10A2Unorm,
                                       wgpu::TextureFormat::RG16Float,  wgpu::TextureFormat::R8Unorm,
                                       wgpu::TextureFormat::RGBA8Uint,  wgpu::TextureFormat::RGBA32Uint};
  EXPECT_EQ(render::detail::color_bytes_per_sample(pbr), 44u);
  pbr.push_back(wgpu::TextureFormat::R32Uint);
  EXPECT_EQ(render::detail::color_bytes_per_sample(pbr), 48u);
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

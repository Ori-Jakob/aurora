#include <gtest/gtest.h>

#include <aurora/gfx.hpp>

#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/hash.hpp"
#include "gfx/pipeline_cache.hpp"
#if AURORA_ENABLE_RENDER
#include "render/graph.hpp"
#endif
#include "webgpu/gpu.hpp"

#include <array>
#include <cstdint>
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

} // namespace
} // namespace aurora::gfx

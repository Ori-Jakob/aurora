#include <gtest/gtest.h>

#include "render/scene_depth.hpp"

namespace aurora::render::detail {
namespace {
static_assert(select_scene_depth_source(true, 1, true) == DepthSource::Direct);

TEST(SceneDepth, SelectsDirectOnCoreSingleSample) {
  EXPECT_EQ(select_scene_depth_source(true, 1, true), DepthSource::Direct);
  EXPECT_EQ(select_scene_depth_source(true, 1, false), DepthSource::Direct);
}

TEST(SceneDepth, SelectsCopyUnderMsaa) { EXPECT_EQ(select_scene_depth_source(true, 4, true), DepthSource::Copy); }

TEST(SceneDepth, UnavailableUnderMsaaWithoutSnapshot) {
  EXPECT_EQ(select_scene_depth_source(true, 4, false), DepthSource::Unavailable);
}

TEST(SceneDepth, UnavailableWithoutCore) {
  EXPECT_EQ(select_scene_depth_source(false, 1, true), DepthSource::Unavailable);
  EXPECT_EQ(select_scene_depth_source(false, 4, true), DepthSource::Unavailable);
}

TEST(SceneDepth, NamesAndChoiceText) {
  EXPECT_EQ(scene_depth_source_name(DepthSource::Unavailable), "unavailable");
  EXPECT_EQ(scene_depth_source_name(DepthSource::Direct), "direct");
  EXPECT_EQ(scene_depth_source_name(DepthSource::Copy), "copy");
  EXPECT_EQ(depth_format_name(wgpu::TextureFormat::Depth32Float), "Depth32Float");
  EXPECT_EQ(depth_format_name(wgpu::TextureFormat::Depth24Plus), "Depth24Plus");
  EXPECT_EQ(depth_format_name(wgpu::TextureFormat::Depth16Unorm), "Depth16Unorm");
  EXPECT_EQ(depth_format_name(wgpu::TextureFormat::Depth24PlusStencil8), "unknown");

  using wgpu::TextureFormat;
  EXPECT_EQ(scene_depth_choice_text(DepthSource::Direct, 1, true, TextureFormat::Depth32Float),
            "Scene depth for render passes: direct (Depth32Float, 1 sample)");
  EXPECT_EQ(scene_depth_choice_text(DepthSource::Copy, 4, true, TextureFormat::Depth32Float),
            "Scene depth for render passes: copy (R32Float snapshot of sample 0, 4x MSAA)");
  EXPECT_EQ(scene_depth_choice_text(DepthSource::Unavailable, 1, false, TextureFormat::Depth32Float),
            "Scene depth for render passes: unavailable (compatibility mode)");
  EXPECT_EQ(scene_depth_choice_text(DepthSource::Unavailable, 4, true, TextureFormat::Depth32Float),
            "Scene depth for render passes: unavailable (no depth snapshot pipeline)");
  EXPECT_EQ(scene_depth_choice_text(DepthSource::Direct, 1, true, TextureFormat::Depth24Plus),
            "Scene depth for render passes: direct (Depth24Plus, 1 sample)");
}

TEST(SceneDepth, StaticSelectionIsConstexpr) {
  constexpr DepthSource source = select_scene_depth_source(true, 1, true);
  static_assert(source == DepthSource::Direct);
  EXPECT_EQ(source, DepthSource::Direct);
}
} // namespace
} // namespace aurora::render::detail

#include <gtest/gtest.h>

#include "gfx/frame.hpp"
#include "gfx/texture_replacement.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/module.hpp"
#include "render/settings.hpp"
#include "render_test_caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <thread>

// gx/texture.hpp needs absl, which render-off builds of this target do not link
namespace aurora::gx::texture {
bool generate_missing_mips_requested() noexcept;
} // namespace aurora::gx::texture

namespace aurora::render::test {
namespace {
using detail::current_frame_state;
using detail::latch_frame;
using detail::testing::settings_generation;

template <typename E>
E random_level(std::mt19937& rng, E last) {
  return static_cast<E>(std::uniform_int_distribution<int>{0, static_cast<int>(last)}(rng));
}

Settings random_request(std::mt19937& rng) {
  std::bernoulli_distribution coin;
  std::uniform_real_distribution<float> value{-1.f, 4.f};
  Settings s;
  s.textureMaterials = coin(rng);
  s.legacyMatMaps = coin(rng);
  s.pbr = coin(rng);
  s.taa = coin(rng);
  s.ssil = coin(rng);
  s.ao = random_level(rng, Quality::Ultra);
  s.shadows.quality = random_level(rng, Quality::Ultra);
  s.shadows.game = random_level(rng, GameShadows::Remove);
  s.rt = random_level(rng, RtMode::PathTraceReference);
  s.rtBackend = random_level(rng, RtBackend::Dxr);
  s.reflections = random_level(rng, Reflections::SsrRt);
  s.fog.volumetric = random_level(rng, Quality::Ultra);
  s.fog.godRays = random_level(rng, GodRays::Volumetric);
  s.sky.clouds = random_level(rng, Quality::Ultra);
  s.sky.cloudShadows = coin(rng);
  s.grass.quality = random_level(rng, Quality::Ultra);
  s.terrain.mode = random_level(rng, TerrainDetail::Tessellated);
  // Out-of-range tuning values make the preview depend on sanitising as well as gating
  s.legacyBlend = value(rng);
  s.exposureBias = value(rng) * 4.f;
  s.grass.distance = value(rng) * 50.f;
  return s;
}

class RenderLatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    detail::testing::reset();
    detail::testing::set_caps(full_caps());
    latch_frame();
  }
  void TearDown() override { detail::testing::reset(); }
};

TEST_F(RenderLatchTest, ApplyLandsAtNextLatch) {
  apply_settings({.legacyBlend = 0.5f});
  EXPECT_EQ(effective_settings().legacyBlend, 0.f);
  EXPECT_EQ(current_frame_state()->settings.legacyBlend, 0.f);
  latch_frame();
  EXPECT_EQ(effective_settings().legacyBlend, 0.5f);
}

TEST_F(RenderLatchTest, CurrentFrameStateIsImmutable) {
  const auto s0 = latch_frame();
  apply_settings({.legacyBlend = 0.5f});
  const auto s1 = latch_frame();
  EXPECT_EQ(s0->settings.legacyBlend, 0.f);
  EXPECT_EQ(s1->settings.legacyBlend, 0.5f);
  EXPECT_NE(s0, s1);
  EXPECT_EQ(current_frame_state(), s1);
}

TEST_F(RenderLatchTest, LastApplyBeforeLatchWins) {
  const uint64_t generation = settings_generation();
  apply_settings({.legacyBlend = 0.25f});
  apply_settings({.legacyBlend = 0.75f});
  EXPECT_EQ(latch_frame()->settings.legacyBlend, 0.75f);
  EXPECT_EQ(settings_generation(), generation + 1);
}

TEST_F(RenderLatchTest, IdenticalApplyKeepsFrameState) {
  const auto state = current_frame_state();
  const uint64_t generation = settings_generation();
  apply_settings(effective_settings());
  EXPECT_EQ(latch_frame(), state);
  EXPECT_EQ(settings_generation(), generation);
}

TEST_F(RenderLatchTest, NormalisedApplyKeepsGeneration) {
  apply_settings({.legacyBlend = 1.f});
  latch_frame();
  ASSERT_EQ(effective_settings().legacyBlend, 1.f);
  const auto state = current_frame_state();
  const uint64_t generation = settings_generation();
  apply_settings({.legacyBlend = 2.f});
  EXPECT_EQ(latch_frame(), state);
  EXPECT_EQ(settings_generation(), generation);
}

TEST_F(RenderLatchTest, ApplyPreviewMatchesLatch) {
  enable_all_built();
  std::mt19937 rng{1};
  for (int i = 0; i < 50; ++i) {
    const Settings preview = apply_settings(random_request(rng));
    EXPECT_EQ(preview, latch_frame()->settings) << "request " << i;
  }
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderLatchTest, CapsChangeReclampsLastRequest) {
  enable_all_built();
  detail::testing::set_caps(compat_caps());
  apply_settings({.ao = Quality::High});
  EXPECT_EQ(latch_frame()->settings.ao, Quality::Off);
  EXPECT_EQ(effective_settings().ao, Quality::Off);
  detail::testing::set_caps(full_caps());
  EXPECT_EQ(latch_frame()->settings.ao, Quality::High);
  EXPECT_EQ(effective_settings().ao, Quality::High);
}

TEST_F(RenderLatchTest, OnlyMaterialFeaturesKeepLegacyStorageSize) {
  constexpr std::array MaterialFeatures{Feature::TextureMaterials, Feature::StochasticTiling, Feature::LegacyMatMaps};
  const auto is_material = [&](Feature feature) {
    return std::ranges::find(MaterialFeatures, feature) != MaterialFeatures.end();
  };
  size_t requests = 0;
  // Building one feature at a time makes the clamp turn every other feature off
  for (size_t i = 0; i < FeatureCount; ++i) {
    const auto feature = static_cast<Feature>(i);
    detail::testing::reset();
    detail::testing::set_caps(full_caps());
    gfx::detail::testing::reset_streaming_request();
    if (is_material(feature)) {
      for (const Feature material : MaterialFeatures) {
        detail::testing::set_built_options(material, ~0u);
      }
    } else {
      detail::testing::set_built_options(feature, ~0u);
    }
    apply_settings(everything_on());
    latch_frame();
    bool moduleFeature = false;
    for (size_t j = 0; j < FeatureCount; ++j) {
      const auto enabled = static_cast<Feature>(j);
      moduleFeature = moduleFeature || (feature_enabled(effective_settings(), enabled) && !is_material(enabled));
    }
    if (is_material(feature)) {
      EXPECT_TRUE(feature_enabled(effective_settings(), feature)) << feature_name(feature);
      EXPECT_FALSE(moduleFeature) << feature_name(feature);
    }
    const uint64_t storage = gfx::detail::testing::requested_streaming_sizes().storage;
    EXPECT_EQ(storage, moduleFeature ? 32 * MiB : gfx::StorageBufferSize) << feature_name(feature);
    requests += moduleFeature ? 1 : 0;
  }
  EXPECT_GT(requests, 0u);
  gfx::detail::testing::reset_streaming_request();
}
#endif

TEST_F(RenderLatchTest, MipGenerationReachesTexturesWithoutStorage) {
#if AURORA_ENABLE_RENDER
  gfx::detail::testing::reset_streaming_request();
#endif
  apply_settings({.generateMissingMips = true});
  latch_frame();
  EXPECT_TRUE(effective_settings().generateMissingMips);
  EXPECT_TRUE(gx::texture::generate_missing_mips_requested());
  EXPECT_TRUE(gfx::texture_replacement::testing::mip_generation());
#if AURORA_ENABLE_RENDER
  EXPECT_EQ(gfx::detail::testing::requested_streaming_sizes().storage, gfx::StorageBufferSize);
#endif

  apply_settings({});
  latch_frame();
  EXPECT_FALSE(effective_settings().generateMissingMips);
  EXPECT_FALSE(gx::texture::generate_missing_mips_requested());
  EXPECT_FALSE(gfx::texture_replacement::testing::mip_generation());
}

TEST_F(RenderLatchTest, ApplyFromAnotherThread) {
  constexpr int Iterations = 1000;
  constexpr std::array Values{0.25f, 0.75f};
  std::thread writer{[&] {
    for (int i = 0; i < Iterations; ++i) {
      apply_settings({.legacyBlend = Values[i % 2]});
    }
  }};
  for (int i = 0; i < Iterations; ++i) {
    const float value = latch_frame()->settings.legacyBlend;
    EXPECT_TRUE(value == 0.f || value == Values[0] || value == Values[1]) << value;
  }
  writer.join();
  EXPECT_EQ(latch_frame()->settings.legacyBlend, Values[(Iterations - 1) % 2]);
}

TEST_F(RenderLatchTest, FirstLatchAfterInitializeKeepsGeneration) {
  detail::shutdown();
  webgpu::g_graphicsConfig.msaaSamples = 1;
  apply_settings({.legacyBlend = 0.5f});
  detail::initialize();
  EXPECT_EQ(effective_settings().legacyBlend, 0.5f);
  const uint64_t generation = settings_generation();
  EXPECT_EQ(latch_frame()->settings.legacyBlend, 0.5f);
  EXPECT_EQ(settings_generation(), generation);
  detail::shutdown();
}

TEST_F(RenderLatchTest, ShutdownDropsFrameState) {
  ASSERT_NE(current_frame_state(), nullptr);
  detail::shutdown();
  EXPECT_EQ(current_frame_state(), nullptr);
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderLatchTest, GraphHandedThroughFrameState) {
  const auto first = latch_frame();
  ASSERT_NE(first->graph, nullptr);
  EXPECT_EQ(first->graph, detail::empty_graph());
  EXPECT_EQ(first->graph->graphHash, 0u);
  EXPECT_EQ(latch_frame()->graph, first->graph);

  apply_settings({.legacyBlend = 0.5f});
  const auto changed = latch_frame();
  EXPECT_NE(changed, first);
  EXPECT_EQ(changed->graph, first->graph);
}
#else
TEST_F(RenderLatchTest, ModuleOffHasNoGraph) { EXPECT_EQ(latch_frame()->graph, nullptr); }
#endif

TEST_F(RenderLatchTest, GBufferWantNeedsBaseTier) {
  using detail::ForceMode;
  using detail::gbuffer_wanted;
  Settings request;
  request.taa = true;
  const Settings none;
  const Caps full = full_caps();
  Caps budget24 = full;
  budget24.limits.maxColorAttachmentBytesPerSample = 24;
  Caps metal32 = full;
  metal32.limits.maxColorAttachmentBytesPerSample = 32;

  EXPECT_FALSE(gbuffer_wanted(request, compat_caps(), 1, true, ForceMode::On));
  EXPECT_FALSE(gbuffer_wanted(request, full, 4, true, ForceMode::On));
  EXPECT_FALSE(gbuffer_wanted(request, budget24, 1, true, ForceMode::On));
  EXPECT_TRUE(gbuffer_wanted(none, metal32, 1, true, ForceMode::On));
  EXPECT_TRUE(gbuffer_wanted(request, metal32, 1, true, ForceMode::Unset));
  EXPECT_TRUE(gbuffer_wanted(request, full, 1, true, ForceMode::Unset));
  EXPECT_FALSE(gbuffer_wanted(none, full, 1, true, ForceMode::Unset));
  EXPECT_FALSE(gbuffer_wanted(request, full, 1, false, ForceMode::On));
  EXPECT_FALSE(gbuffer_wanted(request, full, 1, true, ForceMode::Off));
}

TEST_F(RenderLatchTest, CycleForceTogglesEvery600Latches) {
  using detail::forced_gbuffer;
  using detail::ForceMode;
  constexpr uint64_t Period = detail::GBufferCycleLatches;
  ASSERT_EQ(Period, 600u);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, 0), ForceMode::Off);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, Period - 1), ForceMode::Off);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, Period), ForceMode::On);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, 2 * Period - 1), ForceMode::On);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, 2 * Period), ForceMode::Off);
  EXPECT_EQ(forced_gbuffer(ForceMode::Cycle, 5 * Period + 7), ForceMode::On);
  for (const ForceMode mode : {ForceMode::Unset, ForceMode::On, ForceMode::Off}) {
    EXPECT_EQ(forced_gbuffer(mode, Period), mode);
  }
}
} // namespace
} // namespace aurora::render::test

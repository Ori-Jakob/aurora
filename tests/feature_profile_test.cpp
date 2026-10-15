#include <gtest/gtest.h>

#include "webgpu/feature_profile.hpp"

#include <array>
#include <cstddef>

using namespace aurora::webgpu;

static_assert(FEATURE_PROFILE_AUTO == 0);
static_assert(offsetof(AuroraConfig, featureProfile) > offsetof(AuroraConfig, mem2Size));

namespace {
using wgpu::FeatureName;

constexpr std::array MetalMasked{
    FeatureName::MultiDrawIndirect,
    FeatureName::ChromiumExperimentalSamplingResourceTable,
    FeatureName::PrimitiveIndex,
    FeatureName::AtomicVec2uMinMax,
    FeatureName::SharedBufferMemoryD3D12Resource,
    FeatureName::SharedFenceDXGISharedHandle,
};

constexpr std::array CommonFeatures{
    FeatureName::TextureCompressionBC,
    FeatureName::TextureCompressionASTC,
    FeatureName::TextureComponentSwizzle,
    FeatureName::DualSourceBlending,
    FeatureName::TimestampQuery,
    FeatureName::Subgroups,
    FeatureName::ShaderF16,
};

constexpr std::array CurrentWhitelist{
    FeatureName::CoreFeaturesAndLimits,   FeatureName::TextureCompressionBC, FeatureName::TextureCompressionASTC,
    FeatureName::TextureComponentSwizzle, FeatureName::DualSourceBlending,   FeatureName::TimestampQuery,
};

void expect_resolved(ResolvedFeatureProfile actual, AuroraFeatureProfile profile, FeatureProfileSource source) {
  EXPECT_EQ(actual.profile, profile);
  EXPECT_EQ(actual.source, source);
}

wgpu::Limits limits_with_color_budget(uint32_t bytes) {
  wgpu::Limits limits{};
  limits.maxColorAttachmentBytesPerSample = bytes;
  return limits;
}
} // namespace

TEST(FeatureProfileTest, ParseKnownNames) {
  EXPECT_EQ(parse_feature_profile("auto"), FEATURE_PROFILE_AUTO);
  EXPECT_EQ(parse_feature_profile("full"), FEATURE_PROFILE_FULL);
  EXPECT_EQ(parse_feature_profile("compat"), FEATURE_PROFILE_COMPAT);
  EXPECT_EQ(parse_feature_profile("metal"), FEATURE_PROFILE_METAL);
  EXPECT_EQ(parse_feature_profile("metal32"), FEATURE_PROFILE_METAL32);
}

TEST(FeatureProfileTest, ParseIsCaseInsensitive) {
  EXPECT_EQ(parse_feature_profile("COMPAT"), FEATURE_PROFILE_COMPAT);
  EXPECT_EQ(parse_feature_profile("Metal32"), FEATURE_PROFILE_METAL32);
}

TEST(FeatureProfileTest, ParseRejectsUnknown) {
  EXPECT_FALSE(parse_feature_profile(""));
  EXPECT_FALSE(parse_feature_profile("metal16"));
  EXPECT_FALSE(parse_feature_profile("core"));
  EXPECT_FALSE(parse_feature_profile(" compat"));
  EXPECT_FALSE(parse_feature_profile("compat "));
}

TEST(FeatureProfileTest, ConfigOverridesEnvironment) {
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_METAL, "compat"), FEATURE_PROFILE_METAL,
                  FeatureProfileSource::Config);
}

TEST(FeatureProfileTest, AutoReadsEnvironment) {
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_AUTO, "metal32"), FEATURE_PROFILE_METAL32,
                  FeatureProfileSource::Environment);
}

TEST(FeatureProfileTest, AutoWithoutEnvironmentIsFull) {
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_AUTO, nullptr), FEATURE_PROFILE_FULL,
                  FeatureProfileSource::Default);
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_AUTO, ""), FEATURE_PROFILE_FULL,
                  FeatureProfileSource::Default);
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_AUTO, "auto"), FEATURE_PROFILE_FULL,
                  FeatureProfileSource::Default);
}

TEST(FeatureProfileTest, InvalidEnvironmentFallsBackToFull) {
  expect_resolved(resolve_feature_profile(FEATURE_PROFILE_AUTO, "bogus"), FEATURE_PROFILE_FULL,
                  FeatureProfileSource::InvalidEnvironment);
}

TEST(FeatureProfileTest, OutOfRangeConfigIsTreatedAsAuto) {
  expect_resolved(resolve_feature_profile(static_cast<AuroraFeatureProfile>(5), "compat"), FEATURE_PROFILE_COMPAT,
                  FeatureProfileSource::Environment);
}

TEST(FeatureProfileTest, FullAllowsEveryFeature) {
  EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_FULL, FeatureName::CoreFeaturesAndLimits));
  for (const auto feature : CommonFeatures) {
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_FULL, feature));
  }
  for (const auto feature : MetalMasked) {
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_FULL, feature));
  }
}

TEST(FeatureProfileTest, CompatWithholdsOnlyCore) {
  EXPECT_FALSE(feature_profile_allows(FEATURE_PROFILE_COMPAT, FeatureName::CoreFeaturesAndLimits));
  for (const auto feature : CommonFeatures) {
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_COMPAT, feature));
  }
  for (const auto feature : MetalMasked) {
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_COMPAT, feature));
  }
}

TEST(FeatureProfileTest, MetalProfilesMaskAppleFeatures) {
  for (const auto profile : {FEATURE_PROFILE_METAL, FEATURE_PROFILE_METAL32}) {
    for (const auto feature : MetalMasked) {
      EXPECT_FALSE(feature_profile_allows(profile, feature));
    }
    EXPECT_TRUE(feature_profile_allows(profile, FeatureName::CoreFeaturesAndLimits));
    for (const auto feature : CommonFeatures) {
      EXPECT_TRUE(feature_profile_allows(profile, feature));
    }
  }
}

TEST(FeatureProfileTest, CurrentWhitelistUnaffected) {
  for (const auto feature : CurrentWhitelist) {
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_FULL, feature));
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_METAL, feature));
    EXPECT_TRUE(feature_profile_allows(FEATURE_PROFILE_METAL32, feature));
    EXPECT_EQ(feature_profile_allows(FEATURE_PROFILE_COMPAT, feature), feature != FeatureName::CoreFeaturesAndLimits);
  }
}

TEST(FeatureProfileTest, MaskedFeatureNames) {
  const auto compat = feature_profile_masked_features(FEATURE_PROFILE_COMPAT);
  ASSERT_EQ(compat.size(), 1u);
  EXPECT_EQ(compat[0].feature, FeatureName::CoreFeaturesAndLimits);
  EXPECT_EQ(compat[0].name, "CoreFeaturesAndLimits");

  const auto metal = feature_profile_masked_features(FEATURE_PROFILE_METAL);
  ASSERT_EQ(metal.size(), 6u);
  EXPECT_EQ(metal[0].name, "MultiDrawIndirect");
  EXPECT_EQ(metal[1].name, "ChromiumExperimentalSamplingResourceTable");
  EXPECT_EQ(metal[2].name, "PrimitiveIndex");
  EXPECT_EQ(metal[3].name, "AtomicVec2uMinMax");
  EXPECT_EQ(metal[4].name, "SharedBufferMemoryD3D12Resource");
  EXPECT_EQ(metal[5].name, "SharedFenceDXGISharedHandle");
  for (size_t i = 0; i < metal.size(); ++i) {
    EXPECT_FALSE(metal[i].name.empty());
    EXPECT_EQ(metal[i].feature, MetalMasked[i]);
  }

  EXPECT_TRUE(feature_profile_masked_features(FEATURE_PROFILE_FULL).empty());
  EXPECT_TRUE(feature_profile_masked_features(FEATURE_PROFILE_AUTO).empty());
}

TEST(FeatureProfileTest, ColorBudgets) {
  EXPECT_EQ(feature_profile_color_budget(FEATURE_PROFILE_FULL), 0u);
  EXPECT_EQ(feature_profile_color_budget(FEATURE_PROFILE_COMPAT), 0u);
  EXPECT_EQ(feature_profile_color_budget(FEATURE_PROFILE_METAL), 64u);
  EXPECT_EQ(feature_profile_color_budget(FEATURE_PROFILE_METAL32), 32u);
}

TEST(FeatureProfileTest, ClampLeavesUndefinedLimit) {
  wgpu::Limits limits{};
  clamp_limits_to_feature_profile(FEATURE_PROFILE_METAL, limits);
  EXPECT_EQ(limits.maxColorAttachmentBytesPerSample, WGPU_LIMIT_U32_UNDEFINED);
}

TEST(FeatureProfileTest, ClampOnlyLowers) {
  auto metal = limits_with_color_budget(128);
  clamp_limits_to_feature_profile(FEATURE_PROFILE_METAL, metal);
  EXPECT_EQ(metal.maxColorAttachmentBytesPerSample, 64u);

  auto metal32 = limits_with_color_budget(128);
  clamp_limits_to_feature_profile(FEATURE_PROFILE_METAL32, metal32);
  EXPECT_EQ(metal32.maxColorAttachmentBytesPerSample, 32u);

  auto alreadyLow = limits_with_color_budget(32);
  clamp_limits_to_feature_profile(FEATURE_PROFILE_METAL, alreadyLow);
  EXPECT_EQ(alreadyLow.maxColorAttachmentBytesPerSample, 32u);

  auto full = limits_with_color_budget(128);
  clamp_limits_to_feature_profile(FEATURE_PROFILE_FULL, full);
  EXPECT_EQ(full.maxColorAttachmentBytesPerSample, 128u);
}

TEST(FeatureProfileTest, ClampTouchesNothingElse) {
  auto limits = limits_with_color_budget(128);
  limits.maxTextureDimension2D = 8192;
  limits.maxStorageBuffersPerShaderStage = 2;
  limits.maxColorAttachments = 8;
  clamp_limits_to_feature_profile(FEATURE_PROFILE_METAL32, limits);
  EXPECT_EQ(limits.maxTextureDimension2D, 8192u);
  EXPECT_EQ(limits.maxStorageBuffersPerShaderStage, 2u);
  EXPECT_EQ(limits.maxColorAttachments, 8u);
}

TEST(FeatureProfileTest, ForcesValidation) {
  EXPECT_FALSE(feature_profile_forces_validation(FEATURE_PROFILE_FULL));
  EXPECT_TRUE(feature_profile_forces_validation(FEATURE_PROFILE_COMPAT));
  EXPECT_TRUE(feature_profile_forces_validation(FEATURE_PROFILE_METAL));
  EXPECT_TRUE(feature_profile_forces_validation(FEATURE_PROFILE_METAL32));
}

TEST(FeatureProfileTest, Names) {
  EXPECT_EQ(feature_profile_name(FEATURE_PROFILE_AUTO), "auto");
  EXPECT_EQ(feature_profile_name(FEATURE_PROFILE_FULL), "full");
  EXPECT_EQ(feature_profile_name(FEATURE_PROFILE_COMPAT), "compat");
  EXPECT_EQ(feature_profile_name(FEATURE_PROFILE_METAL), "metal");
  EXPECT_EQ(feature_profile_name(FEATURE_PROFILE_METAL32), "metal32");
  EXPECT_EQ(feature_profile_name(static_cast<AuroraFeatureProfile>(9)), "unknown");
  EXPECT_EQ(feature_profile_source_name(FeatureProfileSource::Default), "default");
  EXPECT_EQ(feature_profile_source_name(FeatureProfileSource::Config), "config");
  EXPECT_EQ(feature_profile_source_name(FeatureProfileSource::Environment), "environment");
  EXPECT_EQ(feature_profile_source_name(FeatureProfileSource::InvalidEnvironment), "invalid environment");
}

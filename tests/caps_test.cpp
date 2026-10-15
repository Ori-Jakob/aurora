#include <gtest/gtest.h>

#include "webgpu/caps.hpp"
#include "webgpu/feature_profile.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

using namespace aurora::webgpu;
using aurora::render::AdapterKind;
using aurora::render::Caps;
using aurora::render::ShaderStage;
using aurora::render::VramSource;

namespace {
using wgpu::FeatureName;
using wgpu::HeapProperty;

constexpr uint64_t MiB = uint64_t{1} << 20;
constexpr uint64_t GiB = uint64_t{1} << 30;

struct FeatureFlag {
  FeatureName feature;
  bool Caps::* flag;
};

constexpr std::array FeatureFlags{
    FeatureFlag{FeatureName::CoreFeaturesAndLimits, &Caps::core},
    FeatureFlag{FeatureName::TextureCompressionBC, &Caps::bcTextures},
    FeatureFlag{FeatureName::TextureCompressionASTC, &Caps::astcTextures},
    FeatureFlag{FeatureName::DualSourceBlending, &Caps::dualSourceBlending},
    FeatureFlag{FeatureName::TextureComponentSwizzle, &Caps::textureComponentSwizzle},
    FeatureFlag{FeatureName::TimestampQuery, &Caps::timestampQuery},
    FeatureFlag{FeatureName::Subgroups, &Caps::subgroups},
    FeatureFlag{FeatureName::ShaderF16, &Caps::shaderF16},
    FeatureFlag{FeatureName::MultiDrawIndirect, &Caps::multiDrawIndirect},
    FeatureFlag{FeatureName::IndirectFirstInstance, &Caps::indirectFirstInstance},
    FeatureFlag{FeatureName::PrimitiveIndex, &Caps::primitiveIndex},
    FeatureFlag{FeatureName::AtomicVec2uMinMax, &Caps::atomic64},
    FeatureFlag{FeatureName::ClipDistances, &Caps::clipDistances},
    FeatureFlag{FeatureName::TextureFormatsTier1, &Caps::textureFormatsTier1},
    FeatureFlag{FeatureName::TextureFormatsTier2, &Caps::textureFormatsTier2},
    FeatureFlag{FeatureName::Float32Filterable, &Caps::float32Filterable},
    FeatureFlag{FeatureName::RG11B10UfloatRenderable, &Caps::rg11b10Renderable},
    FeatureFlag{FeatureName::SharedBufferMemoryD3D12Resource, &Caps::d3d12SharedBuffer},
    FeatureFlag{FeatureName::SharedFenceDXGISharedHandle, &Caps::d3d12SharedFence},
};

std::vector<FeatureName> table_features() {
  std::vector<FeatureName> features;
  for (const auto& entry : requested_features()) {
    features.push_back(entry.feature);
  }
  return features;
}

std::vector<FeatureName> without(std::vector<FeatureName> features, std::initializer_list<FeatureName> removed) {
  std::erase_if(features, [&](FeatureName f) { return std::ranges::find(removed, f) != removed.end(); });
  return features;
}

bool contains(const std::vector<FeatureName>& features, FeatureName feature) {
  return std::ranges::find(features, feature) != features.end();
}

wgpu::Limits d3d12_like() {
  wgpu::Limits limits{};
  limits.maxTextureDimension1D = 16384;
  limits.maxTextureDimension2D = 16384;
  limits.maxTextureDimension3D = 2048;
  limits.maxTextureArrayLayers = 2048;
  limits.maxStorageBuffersPerShaderStage = 16;
  limits.maxStorageTexturesPerShaderStage = 16;
  limits.maxSampledTexturesPerShaderStage = 48;
  limits.maxColorAttachments = 8;
  limits.maxColorAttachmentBytesPerSample = 128;
  limits.maxInterStageShaderVariables = 30;
  limits.maxBufferSize = uint64_t{1} << 52;
  limits.maxStorageBufferBindingSize = uint64_t{1} << 52;
  limits.maxComputeWorkgroupStorageSize = 32768;
  limits.minUniformBufferOffsetAlignment = 256;
  limits.minStorageBufferOffsetAlignment = 32;
  return limits;
}

wgpu::MemoryHeapInfo heap(HeapProperty properties, uint64_t size) {
  return wgpu::MemoryHeapInfo{.properties = properties, .size = size};
}

// The pre-caps request, kept literal so the expectation does not reuse the code under test
wgpu::Limits m1_required_limits(const wgpu::Limits& supportedLimits) {
  return wgpu::Limits{
      .nextInChain = nullptr,
      .maxTextureDimension1D =
          supportedLimits.maxTextureDimension1D == 0 ? WGPU_LIMIT_U32_UNDEFINED : supportedLimits.maxTextureDimension1D,
      .maxTextureDimension2D =
          supportedLimits.maxTextureDimension2D == 0 ? WGPU_LIMIT_U32_UNDEFINED : supportedLimits.maxTextureDimension2D,
      .maxTextureDimension3D =
          supportedLimits.maxTextureDimension3D == 0 ? WGPU_LIMIT_U32_UNDEFINED : supportedLimits.maxTextureDimension3D,
      .maxTextureArrayLayers =
          supportedLimits.maxTextureArrayLayers == 0 ? WGPU_LIMIT_U32_UNDEFINED : supportedLimits.maxTextureArrayLayers,
      .maxStorageBuffersPerShaderStage = 2,
      .minUniformBufferOffsetAlignment =
          supportedLimits.minUniformBufferOffsetAlignment < 64 ? 64 : supportedLimits.minUniformBufferOffsetAlignment,
      .minStorageBufferOffsetAlignment =
          supportedLimits.minStorageBufferOffsetAlignment < 16 ? 16 : supportedLimits.minStorageBufferOffsetAlignment,
      .maxImmediateSize = 64,
  };
}

void expect_limits_eq(const wgpu::Limits& actual, const wgpu::Limits& expected) {
  EXPECT_EQ(actual.nextInChain, expected.nextInChain);
  EXPECT_EQ(actual.maxTextureDimension1D, expected.maxTextureDimension1D);
  EXPECT_EQ(actual.maxTextureDimension2D, expected.maxTextureDimension2D);
  EXPECT_EQ(actual.maxTextureDimension3D, expected.maxTextureDimension3D);
  EXPECT_EQ(actual.maxTextureArrayLayers, expected.maxTextureArrayLayers);
  EXPECT_EQ(actual.maxBindGroups, expected.maxBindGroups);
  EXPECT_EQ(actual.maxBindGroupsPlusVertexBuffers, expected.maxBindGroupsPlusVertexBuffers);
  EXPECT_EQ(actual.maxBindingsPerBindGroup, expected.maxBindingsPerBindGroup);
  EXPECT_EQ(actual.maxDynamicUniformBuffersPerPipelineLayout, expected.maxDynamicUniformBuffersPerPipelineLayout);
  EXPECT_EQ(actual.maxDynamicStorageBuffersPerPipelineLayout, expected.maxDynamicStorageBuffersPerPipelineLayout);
  EXPECT_EQ(actual.maxSampledTexturesPerShaderStage, expected.maxSampledTexturesPerShaderStage);
  EXPECT_EQ(actual.maxSamplersPerShaderStage, expected.maxSamplersPerShaderStage);
  EXPECT_EQ(actual.maxStorageBuffersPerShaderStage, expected.maxStorageBuffersPerShaderStage);
  EXPECT_EQ(actual.maxStorageTexturesPerShaderStage, expected.maxStorageTexturesPerShaderStage);
  EXPECT_EQ(actual.maxUniformBuffersPerShaderStage, expected.maxUniformBuffersPerShaderStage);
  EXPECT_EQ(actual.maxUniformBufferBindingSize, expected.maxUniformBufferBindingSize);
  EXPECT_EQ(actual.maxStorageBufferBindingSize, expected.maxStorageBufferBindingSize);
  EXPECT_EQ(actual.minUniformBufferOffsetAlignment, expected.minUniformBufferOffsetAlignment);
  EXPECT_EQ(actual.minStorageBufferOffsetAlignment, expected.minStorageBufferOffsetAlignment);
  EXPECT_EQ(actual.maxVertexBuffers, expected.maxVertexBuffers);
  EXPECT_EQ(actual.maxBufferSize, expected.maxBufferSize);
  EXPECT_EQ(actual.maxVertexAttributes, expected.maxVertexAttributes);
  EXPECT_EQ(actual.maxVertexBufferArrayStride, expected.maxVertexBufferArrayStride);
  EXPECT_EQ(actual.maxInterStageShaderVariables, expected.maxInterStageShaderVariables);
  EXPECT_EQ(actual.maxColorAttachments, expected.maxColorAttachments);
  EXPECT_EQ(actual.maxColorAttachmentBytesPerSample, expected.maxColorAttachmentBytesPerSample);
  EXPECT_EQ(actual.maxComputeWorkgroupStorageSize, expected.maxComputeWorkgroupStorageSize);
  EXPECT_EQ(actual.maxComputeInvocationsPerWorkgroup, expected.maxComputeInvocationsPerWorkgroup);
  EXPECT_EQ(actual.maxComputeWorkgroupSizeX, expected.maxComputeWorkgroupSizeX);
  EXPECT_EQ(actual.maxComputeWorkgroupSizeY, expected.maxComputeWorkgroupSizeY);
  EXPECT_EQ(actual.maxComputeWorkgroupSizeZ, expected.maxComputeWorkgroupSizeZ);
  EXPECT_EQ(actual.maxComputeWorkgroupsPerDimension, expected.maxComputeWorkgroupsPerDimension);
  EXPECT_EQ(actual.maxImmediateSize, expected.maxImmediateSize);
}

Caps caps_with(std::span<const FeatureName> features, wgpu::BackendType backend = wgpu::BackendType::D3D12) {
  CapsSource source{};
  source.backend = backend;
  source.features = features;
  return build_caps(source);
}
} // namespace

TEST(CapsTest, FeatureTableNamesAndOrder) {
  const auto table = requested_features();
  ASSERT_EQ(table.size(), 21u);
  std::set<std::string_view> names;
  std::set<FeatureName> features;
  for (const auto& entry : table) {
    EXPECT_FALSE(entry.name.empty());
    names.insert(entry.name);
    features.insert(entry.feature);
  }
  EXPECT_EQ(names.size(), table.size());
  EXPECT_EQ(features.size(), table.size());
  const std::array<std::pair<FeatureName, std::string_view>, 21> expected{{
      {FeatureName::CoreFeaturesAndLimits, "CoreFeaturesAndLimits"},
      {FeatureName::TextureCompressionBC, "TextureCompressionBC"},
      {FeatureName::TextureCompressionASTC, "TextureCompressionASTC"},
      {FeatureName::DualSourceBlending, "DualSourceBlending"},
      {FeatureName::TextureComponentSwizzle, "TextureComponentSwizzle"},
      {FeatureName::TimestampQuery, "TimestampQuery"},
      {FeatureName::Subgroups, "Subgroups"},
      {FeatureName::ShaderF16, "ShaderF16"},
      {FeatureName::MultiDrawIndirect, "MultiDrawIndirect"},
      {FeatureName::IndirectFirstInstance, "IndirectFirstInstance"},
      {FeatureName::PrimitiveIndex, "PrimitiveIndex"},
      {FeatureName::AtomicVec2uMinMax, "AtomicVec2uMinMax"},
      {FeatureName::ClipDistances, "ClipDistances"},
      {FeatureName::TextureFormatsTier1, "TextureFormatsTier1"},
      {FeatureName::TextureFormatsTier2, "TextureFormatsTier2"},
      {FeatureName::Float32Filterable, "Float32Filterable"},
      {FeatureName::RG11B10UfloatRenderable, "RG11B10UfloatRenderable"},
      {FeatureName::AdapterPropertiesMemoryHeaps, "AdapterPropertiesMemoryHeaps"},
      {FeatureName::AdapterPropertiesD3D, "AdapterPropertiesD3D"},
      {FeatureName::SharedBufferMemoryD3D12Resource, "SharedBufferMemoryD3D12Resource"},
      {FeatureName::SharedFenceDXGISharedHandle, "SharedFenceDXGISharedHandle"},
  }};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(table[i].feature, expected[i].first) << i;
    EXPECT_EQ(table[i].name, expected[i].second) << i;
  }
  EXPECT_FALSE(features.contains(FeatureName::SharedTextureMemoryD3D12Resource));
  EXPECT_FALSE(features.contains(FeatureName::SubgroupSizeControl));
}

TEST(CapsTest, FeatureNameLookup) {
  EXPECT_EQ(requested_feature_name(FeatureName::MultiDrawIndirect), "MultiDrawIndirect");
  EXPECT_EQ(requested_feature_name(FeatureName::CoreFeaturesAndLimits), "CoreFeaturesAndLimits");
  EXPECT_EQ(requested_feature_name(FeatureName::DepthClipControl), "");
}

TEST(CapsTest, SelectFollowsTableOrder) {
  auto adapter = table_features();
  std::ranges::reverse(adapter);
  EXPECT_EQ(select_device_features(adapter, FEATURE_PROFILE_FULL), table_features());
}

TEST(CapsTest, SelectSkipsUnsupported) {
  const auto adapter = without(table_features(), {FeatureName::Subgroups, FeatureName::ShaderF16});
  const auto selected = select_device_features(adapter, FEATURE_PROFILE_FULL);
  EXPECT_FALSE(contains(selected, FeatureName::Subgroups));
  EXPECT_FALSE(contains(selected, FeatureName::ShaderF16));
  EXPECT_EQ(selected.size(), 19u);
  EXPECT_EQ(selected, adapter);
}

TEST(CapsTest, SelectIgnoresUnlistedFeatures) {
  const std::array adapter{
      FeatureName::DepthClipControl,
      FeatureName::ChromiumExperimentalSamplingResourceTable,
      FeatureName::SubgroupSizeControl,
  };
  EXPECT_TRUE(select_device_features(adapter, FEATURE_PROFILE_FULL).empty());
}

TEST(CapsTest, SelectHasNoDuplicates) {
  const std::array adapter{
      FeatureName::TimestampQuery,
      FeatureName::TimestampQuery,
      FeatureName::TimestampQuery,
  };
  const auto selected = select_device_features(adapter, FEATURE_PROFILE_FULL);
  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0], FeatureName::TimestampQuery);
}

TEST(CapsTest, CompatProfileWithholdsOnlyCore) {
  EXPECT_EQ(select_device_features(table_features(), FEATURE_PROFILE_COMPAT),
            without(table_features(), {FeatureName::CoreFeaturesAndLimits}));
}

TEST(CapsTest, MetalProfilesWithholdAppleFeatures) {
  for (const auto profile : {FEATURE_PROFILE_METAL, FEATURE_PROFILE_METAL32}) {
    const auto selected = select_device_features(table_features(), profile);
    EXPECT_FALSE(contains(selected, FeatureName::MultiDrawIndirect));
    EXPECT_FALSE(contains(selected, FeatureName::PrimitiveIndex));
    EXPECT_FALSE(contains(selected, FeatureName::AtomicVec2uMinMax));
    EXPECT_FALSE(contains(selected, FeatureName::SharedBufferMemoryD3D12Resource));
    EXPECT_FALSE(contains(selected, FeatureName::SharedFenceDXGISharedHandle));
    EXPECT_TRUE(contains(selected, FeatureName::CoreFeaturesAndLimits));
    EXPECT_TRUE(contains(selected, FeatureName::TimestampQuery));
    EXPECT_TRUE(contains(selected, FeatureName::Subgroups));
    EXPECT_TRUE(contains(selected, FeatureName::AdapterPropertiesMemoryHeaps));
  }
}

TEST(CapsTest, RequestsCoreOnlyWhenAllowed) {
  const std::array withCore{FeatureName::TextureCompressionBC, FeatureName::CoreFeaturesAndLimits};
  const std::array withoutCore{FeatureName::TextureCompressionBC};
  EXPECT_TRUE(requests_core_features(withCore, FEATURE_PROFILE_FULL));
  EXPECT_FALSE(requests_core_features(withCore, FEATURE_PROFILE_COMPAT));
  EXPECT_FALSE(requests_core_features(withoutCore, FEATURE_PROFILE_FULL));
}

TEST(CapsTest, CompatRequestMatchesM1) {
  const auto adapter = d3d12_like();
  for (const auto profile :
       {FEATURE_PROFILE_FULL, FEATURE_PROFILE_COMPAT, FEATURE_PROFILE_METAL, FEATURE_PROFILE_METAL32}) {
    SCOPED_TRACE(std::string{feature_profile_name(profile)});
    const auto limits = build_required_limits(adapter, false, profile, 64);
    expect_limits_eq(limits, m1_required_limits(adapter));
    EXPECT_EQ(limits.maxTextureDimension1D, 16384u);
    EXPECT_EQ(limits.maxTextureDimension2D, 16384u);
    EXPECT_EQ(limits.maxTextureDimension3D, 2048u);
    EXPECT_EQ(limits.maxTextureArrayLayers, 2048u);
    EXPECT_EQ(limits.maxStorageBuffersPerShaderStage, 2u);
    EXPECT_EQ(limits.minUniformBufferOffsetAlignment, 256u);
    EXPECT_EQ(limits.minStorageBufferOffsetAlignment, 32u);
    EXPECT_EQ(limits.maxImmediateSize, 64u);
    EXPECT_EQ(limits.maxColorAttachments, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(limits.maxColorAttachmentBytesPerSample, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(limits.maxInterStageShaderVariables, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(limits.maxStorageTexturesPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(limits.maxSampledTexturesPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(limits.maxBufferSize, WGPU_LIMIT_U64_UNDEFINED);
    EXPECT_EQ(limits.maxStorageBufferBindingSize, WGPU_LIMIT_U64_UNDEFINED);
    EXPECT_EQ(limits.maxComputeWorkgroupStorageSize, WGPU_LIMIT_U32_UNDEFINED);
  }
}

TEST(CapsTest, CoreRequestRaisesTargets) {
  const auto adapter = d3d12_like();
  auto expected = m1_required_limits(adapter);
  expected.maxColorAttachments = 8;
  expected.maxColorAttachmentBytesPerSample = 128;
  expected.maxInterStageShaderVariables = 30;
  expected.maxStorageBuffersPerShaderStage = 10;
  expected.maxStorageTexturesPerShaderStage = 8;
  expected.maxSampledTexturesPerShaderStage = 32;
  expected.maxBufferSize = GiB;
  expected.maxStorageBufferBindingSize = GiB;
  expected.maxComputeWorkgroupStorageSize = 32768;
  expect_limits_eq(build_required_limits(adapter, true, FEATURE_PROFILE_FULL, 64), expected);
}

TEST(CapsTest, CoreRequestNeverExceedsAdapter) {
  auto adapter = d3d12_like();
  adapter.maxColorAttachments = 4;
  adapter.maxColorAttachmentBytesPerSample = 64;
  adapter.maxStorageBuffersPerShaderStage = 8;
  adapter.maxStorageTexturesPerShaderStage = 4;
  adapter.maxSampledTexturesPerShaderStage = 16;
  adapter.maxBufferSize = 256 * MiB;
  adapter.maxStorageBufferBindingSize = 128 * MiB;
  adapter.maxComputeWorkgroupStorageSize = 16384;
  adapter.maxInterStageShaderVariables = 16;
  const auto limits = build_required_limits(adapter, true, FEATURE_PROFILE_FULL, 64);
  EXPECT_EQ(limits.maxColorAttachments, 4u);
  EXPECT_EQ(limits.maxColorAttachmentBytesPerSample, 64u);
  EXPECT_EQ(limits.maxStorageBuffersPerShaderStage, 8u);
  EXPECT_EQ(limits.maxStorageTexturesPerShaderStage, 4u);
  EXPECT_EQ(limits.maxSampledTexturesPerShaderStage, 16u);
  EXPECT_EQ(limits.maxBufferSize, 256 * MiB);
  EXPECT_EQ(limits.maxStorageBufferBindingSize, 128 * MiB);
  EXPECT_EQ(limits.maxComputeWorkgroupStorageSize, 16384u);
  EXPECT_EQ(limits.maxInterStageShaderVariables, 16u);
}

TEST(CapsTest, CoreRequestSkipsUndefinedAdapterValues) {
  wgpu::Limits adapter{};
  adapter.maxTextureDimension2D = 0;
  const auto limits = build_required_limits(adapter, true, FEATURE_PROFILE_FULL, 64);
  EXPECT_EQ(limits.maxTextureDimension2D, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxStorageBuffersPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxColorAttachments, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxColorAttachmentBytesPerSample, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxInterStageShaderVariables, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxStorageTexturesPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxSampledTexturesPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(limits.maxBufferSize, WGPU_LIMIT_U64_UNDEFINED);
  EXPECT_EQ(limits.maxStorageBufferBindingSize, WGPU_LIMIT_U64_UNDEFINED);
  EXPECT_EQ(limits.maxComputeWorkgroupStorageSize, WGPU_LIMIT_U32_UNDEFINED);

  wgpu::Limits zeros = adapter;
  zeros.maxStorageBuffersPerShaderStage = 0;
  zeros.maxBufferSize = 0;
  const auto zeroLimits = build_required_limits(zeros, true, FEATURE_PROFILE_FULL, 64);
  EXPECT_EQ(zeroLimits.maxStorageBuffersPerShaderStage, WGPU_LIMIT_U32_UNDEFINED);
  EXPECT_EQ(zeroLimits.maxBufferSize, WGPU_LIMIT_U64_UNDEFINED);
}

TEST(CapsTest, StorageBindingCappedByBufferSize) {
  auto adapter = d3d12_like();
  adapter.maxBufferSize = 512 * MiB;
  adapter.maxStorageBufferBindingSize = 2 * GiB;
  const auto limits = build_required_limits(adapter, true, FEATURE_PROFILE_FULL, 64);
  EXPECT_EQ(limits.maxBufferSize, 512 * MiB);
  EXPECT_EQ(limits.maxStorageBufferBindingSize, 512 * MiB);
}

TEST(CapsTest, MetalProfilesClampColorBudget) {
  const auto metal = build_required_limits(d3d12_like(), true, FEATURE_PROFILE_METAL, 64);
  EXPECT_EQ(metal.maxColorAttachmentBytesPerSample, 64u);
  EXPECT_EQ(metal.maxStorageBuffersPerShaderStage, 10u);
  const auto metal32 = build_required_limits(d3d12_like(), true, FEATURE_PROFILE_METAL32, 64);
  EXPECT_EQ(metal32.maxColorAttachmentBytesPerSample, 32u);
  EXPECT_EQ(metal32.maxStorageBuffersPerShaderStage, 10u);
}

TEST(CapsTest, AlignmentFloors) {
  for (const bool core : {false, true}) {
    auto low = d3d12_like();
    low.minUniformBufferOffsetAlignment = 32;
    low.minStorageBufferOffsetAlignment = 8;
    const auto lowLimits = build_required_limits(low, core, FEATURE_PROFILE_FULL, 64);
    EXPECT_EQ(lowLimits.minUniformBufferOffsetAlignment, 64u);
    EXPECT_EQ(lowLimits.minStorageBufferOffsetAlignment, 16u);

    auto high = d3d12_like();
    high.minUniformBufferOffsetAlignment = 256;
    high.minStorageBufferOffsetAlignment = 256;
    const auto highLimits = build_required_limits(high, core, FEATURE_PROFILE_FULL, 64);
    EXPECT_EQ(highLimits.minUniformBufferOffsetAlignment, 256u);
    EXPECT_EQ(highLimits.minStorageBufferOffsetAlignment, 256u);

    const auto undefinedLimits = build_required_limits(wgpu::Limits{}, core, FEATURE_PROFILE_FULL, 64);
    EXPECT_EQ(undefinedLimits.minUniformBufferOffsetAlignment, WGPU_LIMIT_U32_UNDEFINED);
    EXPECT_EQ(undefinedLimits.minStorageBufferOffsetAlignment, WGPU_LIMIT_U32_UNDEFINED);
  }
}

TEST(CapsTest, DescribeFeatureStates) {
  const auto adapter = without(table_features(), {FeatureName::Subgroups});
  std::vector<FeatureName> device;
  for (const auto feature : adapter) {
    if (feature_profile_allows(FEATURE_PROFILE_METAL, feature)) {
      device.push_back(feature);
    }
  }
  const auto results = describe_feature_requests(adapter, device, FEATURE_PROFILE_METAL);
  const auto table = requested_features();
  ASSERT_EQ(results.size(), table.size());
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_EQ(results[i].feature, table[i].feature);
    EXPECT_EQ(results[i].name, table[i].name);
    if (results[i].feature == FeatureName::MultiDrawIndirect) {
      EXPECT_EQ(results[i].state, FeatureRequestState::MaskedByProfile);
    } else if (results[i].feature == FeatureName::Subgroups) {
      EXPECT_EQ(results[i].state, FeatureRequestState::Unsupported);
    } else if (results[i].feature == FeatureName::TimestampQuery) {
      EXPECT_EQ(results[i].state, FeatureRequestState::Enabled);
    }
  }
}

TEST(CapsTest, BuildCapsMapsFeatures) {
  for (const auto& flag : FeatureFlags) {
    EXPECT_FALSE(requested_feature_name(flag.feature).empty());
  }
  for (const auto& entry : requested_features()) {
    SCOPED_TRACE(std::string{entry.name});
    const std::array features{entry.feature};
    const auto caps = caps_with(features);
    for (const auto& flag : FeatureFlags) {
      EXPECT_EQ(caps.*flag.flag, flag.feature == entry.feature) << requested_feature_name(flag.feature);
    }
    EXPECT_FALSE(caps.d3d12Interop);
  }
}

TEST(CapsTest, ShaderModelText) {
  EXPECT_EQ(shader_model_text(0), "n/a");
  EXPECT_EQ(shader_model_text(51), "5.1");
  EXPECT_EQ(shader_model_text(62), "6.2");
  EXPECT_EQ(shader_model_text(66), "6.6");
  EXPECT_EQ(shader_model_text(69), "6.9");
  EXPECT_EQ(shader_model_text(70), "6.10");
  EXPECT_EQ(shader_model_text(71), "6.11");
}

TEST(CapsTest, BuildCapsCopiesLimits) {
  CapsSource source{};
  auto& l = source.limits;
  l.maxTextureDimension1D = 1;
  l.maxTextureDimension2D = 2;
  l.maxTextureDimension3D = 3;
  l.maxTextureArrayLayers = 4;
  l.maxBindGroups = 5;
  l.maxBindGroupsPlusVertexBuffers = 6;
  l.maxBindingsPerBindGroup = 7;
  l.maxDynamicUniformBuffersPerPipelineLayout = 8;
  l.maxDynamicStorageBuffersPerPipelineLayout = 9;
  l.maxSampledTexturesPerShaderStage = 10;
  l.maxSamplersPerShaderStage = 11;
  l.maxStorageBuffersPerShaderStage = 12;
  l.maxStorageTexturesPerShaderStage = 13;
  l.maxUniformBuffersPerShaderStage = 14;
  l.maxUniformBufferBindingSize = 15;
  l.maxStorageBufferBindingSize = 16;
  l.minUniformBufferOffsetAlignment = 17;
  l.minStorageBufferOffsetAlignment = 18;
  l.maxVertexBuffers = 19;
  l.maxBufferSize = 20;
  l.maxVertexAttributes = 21;
  l.maxVertexBufferArrayStride = 22;
  l.maxInterStageShaderVariables = 23;
  l.maxColorAttachments = 24;
  l.maxColorAttachmentBytesPerSample = 25;
  l.maxComputeWorkgroupStorageSize = 26;
  l.maxComputeInvocationsPerWorkgroup = 27;
  l.maxComputeWorkgroupSizeX = 28;
  l.maxComputeWorkgroupSizeY = 29;
  l.maxComputeWorkgroupSizeZ = 30;
  l.maxComputeWorkgroupsPerDimension = 31;
  l.maxImmediateSize = 32;
  source.compatLimits.maxStorageBuffersInVertexStage = 33;
  source.compatLimits.maxStorageTexturesInVertexStage = 34;
  source.compatLimits.maxStorageBuffersInFragmentStage = 35;
  source.compatLimits.maxStorageTexturesInFragmentStage = 36;
  const auto d = build_caps(source).limits;
  EXPECT_EQ(d.maxTextureDimension2D, 2u);
  EXPECT_EQ(d.maxTextureDimension3D, 3u);
  EXPECT_EQ(d.maxTextureArrayLayers, 4u);
  EXPECT_EQ(d.maxBindGroups, 5u);
  EXPECT_EQ(d.maxBindingsPerBindGroup, 7u);
  EXPECT_EQ(d.maxDynamicUniformBuffersPerPipelineLayout, 8u);
  EXPECT_EQ(d.maxDynamicStorageBuffersPerPipelineLayout, 9u);
  EXPECT_EQ(d.maxSampledTexturesPerShaderStage, 10u);
  EXPECT_EQ(d.maxSamplersPerShaderStage, 11u);
  EXPECT_EQ(d.maxStorageBuffersPerShaderStage, 12u);
  EXPECT_EQ(d.maxStorageTexturesPerShaderStage, 13u);
  EXPECT_EQ(d.maxUniformBuffersPerShaderStage, 14u);
  EXPECT_EQ(d.maxUniformBufferBindingSize, 15u);
  EXPECT_EQ(d.maxStorageBufferBindingSize, 16u);
  EXPECT_EQ(d.minUniformBufferOffsetAlignment, 17u);
  EXPECT_EQ(d.minStorageBufferOffsetAlignment, 18u);
  EXPECT_EQ(d.maxVertexBuffers, 19u);
  EXPECT_EQ(d.maxBufferSize, 20u);
  EXPECT_EQ(d.maxVertexAttributes, 21u);
  EXPECT_EQ(d.maxInterStageShaderVariables, 23u);
  EXPECT_EQ(d.maxColorAttachments, 24u);
  EXPECT_EQ(d.maxColorAttachmentBytesPerSample, 25u);
  EXPECT_EQ(d.maxComputeWorkgroupStorageSize, 26u);
  EXPECT_EQ(d.maxComputeInvocationsPerWorkgroup, 27u);
  EXPECT_EQ(d.maxComputeWorkgroupSizeX, 28u);
  EXPECT_EQ(d.maxComputeWorkgroupSizeY, 29u);
  EXPECT_EQ(d.maxComputeWorkgroupSizeZ, 30u);
  EXPECT_EQ(d.maxComputeWorkgroupsPerDimension, 31u);
  EXPECT_EQ(d.maxImmediateSize, 32u);
  EXPECT_EQ(d.maxStorageBuffersInVertexStage, 33u);
  EXPECT_EQ(d.maxStorageTexturesInVertexStage, 34u);
  EXPECT_EQ(d.maxStorageBuffersInFragmentStage, 35u);
  EXPECT_EQ(d.maxStorageTexturesInFragmentStage, 36u);
}

TEST(CapsTest, BuildCapsSubgroupSizesNeedFeature) {
  CapsSource source{};
  source.subgroupMinSize = 32;
  source.subgroupMaxSize = 32;
  const std::array features{FeatureName::Subgroups};
  source.features = features;
  const auto with = build_caps(source);
  EXPECT_TRUE(with.subgroups);
  EXPECT_EQ(with.subgroupMinSize, 32u);
  EXPECT_EQ(with.subgroupMaxSize, 32u);

  source.features = {};
  const auto withoutFeature = build_caps(source);
  EXPECT_FALSE(withoutFeature.subgroups);
  EXPECT_EQ(withoutFeature.subgroupMinSize, 0u);
  EXPECT_EQ(withoutFeature.subgroupMaxSize, 0u);
}

TEST(CapsTest, D3D12InteropNeedsBackendAndBothFeatures) {
  const std::array both{FeatureName::SharedBufferMemoryD3D12Resource, FeatureName::SharedFenceDXGISharedHandle};
  const std::array bufferOnly{FeatureName::SharedBufferMemoryD3D12Resource};
  EXPECT_TRUE(caps_with(both, wgpu::BackendType::D3D12).d3d12Interop);
  EXPECT_FALSE(caps_with(bufferOnly, wgpu::BackendType::D3D12).d3d12Interop);
  EXPECT_FALSE(caps_with(both, wgpu::BackendType::Vulkan).d3d12Interop);
  EXPECT_FALSE(caps_with(both, wgpu::BackendType::Null).d3d12Interop);
}

TEST(CapsTest, NativeD3D12StaysOff) {
  const std::array both{FeatureName::SharedBufferMemoryD3D12Resource, FeatureName::SharedFenceDXGISharedHandle};
  CapsSource source{};
  source.backend = wgpu::BackendType::D3D12;
  source.features = both;
  source.d3dShaderModel = 66;
  const auto caps = build_caps(source);
  EXPECT_TRUE(caps.d3d12Interop);
  EXPECT_FALSE(caps.nativeD3D12);
  EXPECT_EQ(caps.d3dShaderModel, 66u);
}

TEST(CapsTest, BuildCapsIdentity) {
  EXPECT_EQ(aurora_backend(wgpu::BackendType::Undefined), BACKEND_AUTO);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::Null), BACKEND_NULL);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::WebGPU), BACKEND_WEBGPU);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::D3D11), BACKEND_D3D11);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::D3D12), BACKEND_D3D12);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::Metal), BACKEND_METAL);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::Vulkan), BACKEND_VULKAN);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::OpenGL), BACKEND_OPENGL);
  EXPECT_EQ(aurora_backend(wgpu::BackendType::OpenGLES), BACKEND_OPENGLES);

  EXPECT_EQ(adapter_kind(wgpu::AdapterType::DiscreteGPU), AdapterKind::Discrete);
  EXPECT_EQ(adapter_kind(wgpu::AdapterType::IntegratedGPU), AdapterKind::Integrated);
  EXPECT_EQ(adapter_kind(wgpu::AdapterType::CPU), AdapterKind::Cpu);
  EXPECT_EQ(adapter_kind(wgpu::AdapterType::Unknown), AdapterKind::Unknown);

  CapsSource source{};
  source.backend = wgpu::BackendType::Vulkan;
  source.adapterType = wgpu::AdapterType::IntegratedGPU;
  source.profile = FEATURE_PROFILE_METAL32;
  source.vendorId = 0x10de;
  source.deviceId = 0x2820;
  source.adapterName = "Test Adapter";
  const auto caps = build_caps(source);
  EXPECT_TRUE(caps.valid);
  EXPECT_EQ(caps.backend, BACKEND_VULKAN);
  EXPECT_EQ(caps.adapterKind, AdapterKind::Integrated);
  EXPECT_EQ(caps.profile, FEATURE_PROFILE_METAL32);
  EXPECT_EQ(caps.vendorId, 0x10deu);
  EXPECT_EQ(caps.deviceId, 0x2820u);
  EXPECT_EQ(caps.adapterName, "Test Adapter");
}

TEST(CapsTest, VramDiscreteD3D12) {
  const std::array heaps{
      heap(HeapProperty::DeviceLocal, 8 * GiB),
      heap(HeapProperty::HostVisible | HeapProperty::HostCoherent, 16 * GiB),
  };
  const auto vram = vram_from_heaps(heaps, wgpu::AdapterType::DiscreteGPU);
  EXPECT_EQ(vram.bytes, 8 * GiB);
  EXPECT_EQ(vram.source, VramSource::MemoryHeaps);
  EXPECT_FALSE(vram.unified);

  CapsSource source{};
  source.adapterType = wgpu::AdapterType::DiscreteGPU;
  source.memoryHeaps = heaps;
  const auto caps = build_caps(source);
  EXPECT_EQ(caps.vramBytes, 8 * GiB);
  EXPECT_EQ(caps.vramSource, VramSource::MemoryHeaps);
  EXPECT_FALSE(caps.unifiedMemory);
}

TEST(CapsTest, VramPicksLargestDeviceLocalHeap) {
  const std::array heaps{
      heap(HeapProperty::DeviceLocal, 8 * GiB),
      heap(HeapProperty::DeviceLocal | HeapProperty::HostVisible, 256 * MiB),
      heap(HeapProperty::HostVisible, 32 * GiB),
  };
  EXPECT_EQ(vram_from_heaps(heaps, wgpu::AdapterType::DiscreteGPU).bytes, 8 * GiB);
}

TEST(CapsTest, VramRebarHeapCounts) {
  const std::array heaps{
      heap(HeapProperty::DeviceLocal | HeapProperty::HostVisible, 8 * GiB),
      heap(HeapProperty::HostVisible, 32 * GiB),
  };
  const auto vram = vram_from_heaps(heaps, wgpu::AdapterType::DiscreteGPU);
  EXPECT_EQ(vram.bytes, 8 * GiB);
  EXPECT_EQ(vram.source, VramSource::MemoryHeaps);
  EXPECT_FALSE(vram.unified);
}

TEST(CapsTest, VramIntegratedIsUnified) {
  const std::array heaps{heap(HeapProperty::DeviceLocal | HeapProperty::HostVisible, 16 * GiB)};
  const auto integrated = vram_from_heaps(heaps, wgpu::AdapterType::IntegratedGPU);
  EXPECT_EQ(integrated.bytes, 16 * GiB);
  EXPECT_EQ(integrated.source, VramSource::MemoryHeaps);
  EXPECT_TRUE(integrated.unified);
  EXPECT_TRUE(vram_from_heaps(heaps, wgpu::AdapterType::CPU).unified);
}

TEST(CapsTest, VramIntegratedCarveOutUsesSharedHeap) {
  const std::array heaps{
      heap(HeapProperty::DeviceLocal, 256 * MiB),
      heap(HeapProperty::HostVisible | HeapProperty::HostCoherent, 15 * GiB),
  };
  const auto integrated = vram_from_heaps(heaps, wgpu::AdapterType::IntegratedGPU);
  EXPECT_EQ(integrated.bytes, 15 * GiB);
  EXPECT_EQ(integrated.source, VramSource::MemoryHeaps);
  EXPECT_TRUE(integrated.unified);
  EXPECT_EQ(vram_from_heaps(heaps, wgpu::AdapterType::DiscreteGPU).bytes, 256 * MiB);
}

TEST(CapsTest, VramZeroSizeDeviceLocalHeapIsUnknown) {
  const std::array heaps{heap(HeapProperty::DeviceLocal, 0), heap(HeapProperty::HostVisible, 16 * GiB)};
  const auto vram = vram_from_heaps(heaps, wgpu::AdapterType::DiscreteGPU);
  EXPECT_EQ(vram.bytes, 0u);
  EXPECT_EQ(vram.source, VramSource::Unknown);
}

TEST(CapsTest, VramUnknownWithoutDeviceLocalHeap) {
  const auto empty = vram_from_heaps({}, wgpu::AdapterType::DiscreteGPU);
  EXPECT_EQ(empty.bytes, 0u);
  EXPECT_EQ(empty.source, VramSource::Unknown);
  EXPECT_FALSE(empty.unified);

  const std::array heaps{heap(HeapProperty::HostVisible, 16 * GiB)};
  const auto hostOnly = vram_from_heaps(heaps, wgpu::AdapterType::IntegratedGPU);
  EXPECT_EQ(hostOnly.bytes, 0u);
  EXPECT_EQ(hostOnly.source, VramSource::Unknown);
  EXPECT_FALSE(hostOnly.unified);
}

TEST(CapsTest, StageBindingLimitsPerStage) {
  Caps caps{};
  caps.limits.maxStorageBuffersPerShaderStage = 10;
  caps.limits.maxStorageTexturesPerShaderStage = 8;
  caps.limits.maxStorageBuffersInVertexStage = 2;
  caps.limits.maxStorageTexturesInVertexStage = 1;
  caps.limits.maxStorageBuffersInFragmentStage = 4;
  caps.limits.maxStorageTexturesInFragmentStage = 3;
  caps.limits.maxSampledTexturesPerShaderStage = 32;
  caps.limits.maxSamplersPerShaderStage = 16;
  caps.limits.maxUniformBuffersPerShaderStage = 12;

  const auto vertex = aurora::render::stage_binding_limits(caps, ShaderStage::Vertex);
  EXPECT_EQ(vertex.storageBuffers, 2u);
  EXPECT_EQ(vertex.storageTextures, 1u);
  const auto fragment = aurora::render::stage_binding_limits(caps, ShaderStage::Fragment);
  EXPECT_EQ(fragment.storageBuffers, 4u);
  EXPECT_EQ(fragment.storageTextures, 3u);
  const auto compute = aurora::render::stage_binding_limits(caps, ShaderStage::Compute);
  EXPECT_EQ(compute.storageBuffers, 10u);
  EXPECT_EQ(compute.storageTextures, 8u);
  for (const auto& stage : {vertex, fragment, compute}) {
    EXPECT_EQ(stage.sampledTextures, 32u);
    EXPECT_EQ(stage.samplers, 16u);
    EXPECT_EQ(stage.uniformBuffers, 12u);
  }
}

TEST(CapsTest, GlobalCapsLifecycle) {
  EXPECT_FALSE(aurora::render::caps().valid);
  EXPECT_EQ(aurora::render::caps(), Caps{});
  Caps caps{};
  caps.valid = true;
  caps.adapterName = "Global Adapter";
  set_caps(caps);
  EXPECT_TRUE(aurora::render::caps().valid);
  EXPECT_EQ(aurora::render::caps().adapterName, "Global Adapter");
  EXPECT_EQ(aurora::render::caps(), caps);
  reset_caps();
  EXPECT_FALSE(aurora::render::caps().valid);
  EXPECT_EQ(aurora::render::caps(), Caps{});
}

TEST(CapsTest, Names) {
  EXPECT_EQ(aurora::render::adapter_kind_name(AdapterKind::Unknown), "unknown");
  EXPECT_EQ(aurora::render::adapter_kind_name(AdapterKind::Discrete), "discrete");
  EXPECT_EQ(aurora::render::adapter_kind_name(AdapterKind::Integrated), "integrated");
  EXPECT_EQ(aurora::render::adapter_kind_name(AdapterKind::Cpu), "cpu");
  EXPECT_EQ(aurora::render::vram_source_name(VramSource::Unknown), "unknown");
  EXPECT_EQ(aurora::render::vram_source_name(VramSource::MemoryHeaps), "memory heaps");
  EXPECT_EQ(aurora::render::vram_source_name(VramSource::Dxgi), "dxgi");
}

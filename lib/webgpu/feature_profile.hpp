#pragma once

#include <aurora/aurora.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::webgpu {
inline constexpr const char* FeatureProfileEnvVar = "AURORA_FEATURE_PROFILE";

enum class FeatureProfileSource : uint8_t { Default, Config, Environment, InvalidEnvironment };

struct ResolvedFeatureProfile {
  AuroraFeatureProfile profile;
  FeatureProfileSource source;
};

struct MaskedFeature {
  wgpu::FeatureName feature;
  std::string_view name;
};

std::string_view feature_profile_name(AuroraFeatureProfile profile) noexcept;
std::string_view feature_profile_source_name(FeatureProfileSource source) noexcept;
std::optional<AuroraFeatureProfile> parse_feature_profile(std::string_view text) noexcept;
ResolvedFeatureProfile resolve_feature_profile(AuroraFeatureProfile configured, const char* envValue) noexcept;
std::span<const MaskedFeature> feature_profile_masked_features(AuroraFeatureProfile profile) noexcept;
bool feature_profile_allows(AuroraFeatureProfile profile, wgpu::FeatureName feature) noexcept;
uint32_t feature_profile_color_budget(AuroraFeatureProfile profile) noexcept;
bool feature_profile_forces_validation(AuroraFeatureProfile profile) noexcept;
void clamp_limits_to_feature_profile(AuroraFeatureProfile profile, wgpu::Limits& limits) noexcept;
} // namespace aurora::webgpu

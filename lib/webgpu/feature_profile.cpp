#include "feature_profile.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace aurora::webgpu {
namespace {
constexpr std::array<MaskedFeature, 1> CompatMaskedFeatures{{
    {wgpu::FeatureName::CoreFeaturesAndLimits, "CoreFeaturesAndLimits"},
}};

// Worst-case Apple GPU: Dawn Metal has no MDI or resource table; PrimitiveIndex needs Apple7, atomic64 Apple8+
constexpr std::array<MaskedFeature, 4> MetalMaskedFeatures{{
    {wgpu::FeatureName::MultiDrawIndirect, "MultiDrawIndirect"},
    {wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable, "ChromiumExperimentalSamplingResourceTable"},
    {wgpu::FeatureName::PrimitiveIndex, "PrimitiveIndex"},
    {wgpu::FeatureName::AtomicVec2uMinMax, "AtomicVec2uMinMax"},
}};

constexpr std::array<std::pair<std::string_view, AuroraFeatureProfile>, 5> ProfileNames{{
    {"auto", FEATURE_PROFILE_AUTO},
    {"full", FEATURE_PROFILE_FULL},
    {"compat", FEATURE_PROFILE_COMPAT},
    {"metal", FEATURE_PROFILE_METAL},
    {"metal32", FEATURE_PROFILE_METAL32},
}};

constexpr char ascii_lower(char c) noexcept { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

bool equals_ignore_case(std::string_view a, std::string_view b) noexcept {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return ascii_lower(x) == ascii_lower(y); });
}
} // namespace

std::string_view feature_profile_name(AuroraFeatureProfile profile) noexcept {
  for (const auto& [name, value] : ProfileNames) {
    if (value == profile) {
      return name;
    }
  }
  return "unknown";
}

std::string_view feature_profile_source_name(FeatureProfileSource source) noexcept {
  switch (source) {
  case FeatureProfileSource::Default:
    return "default";
  case FeatureProfileSource::Config:
    return "config";
  case FeatureProfileSource::Environment:
    return "environment";
  case FeatureProfileSource::InvalidEnvironment:
    return "invalid environment";
  }
  return "unknown";
}

std::optional<AuroraFeatureProfile> parse_feature_profile(std::string_view text) noexcept {
  for (const auto& [name, value] : ProfileNames) {
    if (equals_ignore_case(text, name)) {
      return value;
    }
  }
  return std::nullopt;
}

ResolvedFeatureProfile resolve_feature_profile(AuroraFeatureProfile configured, const char* envValue) noexcept {
  const int configuredValue = static_cast<int>(configured);
  if (configuredValue >= FEATURE_PROFILE_FULL && configuredValue <= FEATURE_PROFILE_METAL32) {
    return {configured, FeatureProfileSource::Config};
  }
  if (envValue == nullptr || *envValue == '\0') {
    return {FEATURE_PROFILE_FULL, FeatureProfileSource::Default};
  }
  const auto parsed = parse_feature_profile(envValue);
  if (!parsed) {
    return {FEATURE_PROFILE_FULL, FeatureProfileSource::InvalidEnvironment};
  }
  if (*parsed == FEATURE_PROFILE_AUTO) {
    return {FEATURE_PROFILE_FULL, FeatureProfileSource::Default};
  }
  return {*parsed, FeatureProfileSource::Environment};
}

std::span<const MaskedFeature> feature_profile_masked_features(AuroraFeatureProfile profile) noexcept {
  switch (profile) {
  case FEATURE_PROFILE_COMPAT:
    return CompatMaskedFeatures;
  case FEATURE_PROFILE_METAL:
  case FEATURE_PROFILE_METAL32:
    return MetalMaskedFeatures;
  default:
    return {};
  }
}

bool feature_profile_allows(AuroraFeatureProfile profile, wgpu::FeatureName feature) noexcept {
  const auto masked = feature_profile_masked_features(profile);
  return std::none_of(masked.begin(), masked.end(), [feature](const MaskedFeature& m) { return m.feature == feature; });
}

uint32_t feature_profile_color_budget(AuroraFeatureProfile profile) noexcept {
  switch (profile) {
  case FEATURE_PROFILE_METAL:
    return 64;
  case FEATURE_PROFILE_METAL32:
    return 32;
  default:
    return 0;
  }
}

bool feature_profile_forces_validation(AuroraFeatureProfile profile) noexcept {
  return profile != FEATURE_PROFILE_FULL;
}

void clamp_limits_to_feature_profile(AuroraFeatureProfile profile, wgpu::Limits& limits) noexcept {
  const uint32_t budget = feature_profile_color_budget(profile);
  // Only lower an explicit request; writing the budget into an undefined limit would raise the device default
  if (budget != 0 && limits.maxColorAttachmentBytesPerSample != WGPU_LIMIT_U32_UNDEFINED &&
      limits.maxColorAttachmentBytesPerSample > budget) {
    limits.maxColorAttachmentBytesPerSample = budget;
  }
}
} // namespace aurora::webgpu

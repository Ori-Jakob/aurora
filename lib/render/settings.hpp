#pragma once

#include <aurora/render.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace aurora::render::detail {
inline constexpr uint32_t BaseTierColorBytes = 25;
inline constexpr uint32_t PbrTierColorBytes = 44;
inline constexpr uint64_t HighVramBytes = 6ull << 30;

enum class Tier : uint8_t { None, Base, Pbr };
enum class ForceMode : uint8_t { Unset, On, Off, Cycle };
inline constexpr uint64_t GBufferCycleLatches = 600;

struct FeatureInfo {
  std::string_view name;
  Tier tier = Tier::None;
  uint8_t optionCount = 1;
};

using BuiltOptions = std::array<uint32_t, FeatureCount>;

struct SupportEnv {
  const Caps& caps;
  uint32_t msaaSamples = 1;
  const BuiltOptions& built;
  bool moduleEnabled = false;
};

const FeatureInfo& feature_info(Feature feature) noexcept;
const BuiltOptions& default_built_options() noexcept;
uint8_t requested_option(const Settings& settings, Feature feature) noexcept;
bool sun_shadows(const Settings& settings) noexcept;
Tier device_tier(const Caps& caps, uint32_t msaaSamples) noexcept;
bool gbuffer_wanted(const Settings& settings, const Caps& caps, uint32_t msaaSamples, bool moduleEnabled,
                    ForceMode forced) noexcept;
ForceMode forced_gbuffer(ForceMode mode, uint64_t latchIndex) noexcept;
FeatureSupport evaluate_support(Feature feature, uint8_t option, const Settings& context,
                                const SupportEnv& env) noexcept;
Settings sanitize_settings(const Settings& requested) noexcept;
Settings clamp_settings(const Settings& requested, const SupportEnv& env) noexcept;

Settings submit_request(const Settings& requested);
FeatureSupport current_support(Feature feature, uint8_t option, const Settings* context) noexcept;
const Settings& latched_settings() noexcept;
void reset_settings_state(uint32_t msaaSamples) noexcept;
void shutdown_settings_state() noexcept;

namespace testing {
void reset() noexcept;
void set_caps(const Caps& caps, uint32_t msaaSamples = 1) noexcept;
void set_built_options(Feature feature, uint32_t optionMask) noexcept;
} // namespace testing
} // namespace aurora::render::detail

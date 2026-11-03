#pragma once

#include <aurora/render.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::detail {
inline constexpr uint32_t BaseTierColorBytes = 25;
inline constexpr uint32_t PbrTierColorBytes = 42;
inline constexpr uint64_t HighVramBytes = 6ull << 30;

enum class Tier : uint8_t { None, Base, Pbr };
enum class ForceMode : uint8_t { Unset, On, Off, Cycle, Pbr, PbrCycle };
inline constexpr uint64_t GBufferCycleLatches = 600;

struct FeatureInfo {
  std::string_view name;
  Tier tier = Tier::None;
  uint8_t optionCount = 1;
};

using BuiltOptions = std::array<uint32_t, FeatureCount>;

struct ProbeEntry {
  bool known = true;
  FeatureSupport support{false, Reason::BuildDisabled};

  bool operator==(const ProbeEntry&) const = default;
};

struct UpscalerProbe {
  std::array<ProbeEntry, 2> native;
  uint32_t vendorId = 0;
  bool xessXeFx = false;

  bool operator==(const UpscalerProbe&) const = default;
};

struct UpscalerOverride {
  Upscaler upscaler = Upscaler::Off;
  std::optional<UpscalerQuality> quality;

  bool operator==(const UpscalerOverride&) const = default;
};

struct SupportEnv {
  const Caps& caps;
  uint32_t msaaSamples = 1;
  const BuiltOptions& built;
  bool moduleEnabled = false;
  const UpscalerProbe* upscalers = nullptr;
};

const FeatureInfo& feature_info(Feature feature) noexcept;
const BuiltOptions& default_built_options() noexcept;
uint8_t requested_option(const Settings& settings, Feature feature) noexcept;
bool sun_shadows(const Settings& settings) noexcept;
Tier device_tier(const Caps& caps, uint32_t msaaSamples) noexcept;
Tier feature_tier(Feature feature, uint8_t level) noexcept;
Tier requested_scene_tier(const Settings& settings) noexcept;
Tier scene_tier_wanted(const Settings& settings, const Caps& caps, uint32_t msaaSamples, bool moduleEnabled,
                       ForceMode forced) noexcept;
bool gbuffer_wanted(const Settings& settings, const Caps& caps, uint32_t msaaSamples, bool moduleEnabled,
                    ForceMode forced) noexcept;
ForceMode forced_gbuffer(ForceMode mode, uint64_t latchIndex) noexcept;
BuiltOptions with_pbr_dev_built(BuiltOptions built, bool requested) noexcept;
std::optional<float> parse_legacy_blend(std::string_view text) noexcept;
bool mat_maps_env() noexcept;
FeatureSupport evaluate_support(Feature feature, uint8_t option, const Settings& context,
                                const SupportEnv& env) noexcept;
Settings sanitize_settings(const Settings& requested) noexcept;
std::optional<UpscalerOverride> parse_upscaler_override(std::string_view text) noexcept;
Settings clamp_settings(const Settings& requested, const SupportEnv& env) noexcept;

Settings submit_request(const Settings& requested);
FeatureSupport current_support(Feature feature, uint8_t option, const Settings* context) noexcept;
const Settings& latched_settings() noexcept;
void reset_settings_state(uint32_t msaaSamples) noexcept;
void shutdown_settings_state() noexcept;
void set_upscaler_probe(const UpscalerProbe& probe) noexcept;
void set_probe_requester(void (*requester)() noexcept) noexcept;
Upscaler requested_upscaler() noexcept;

namespace testing {
using Clock = std::chrono::steady_clock;

void reset() noexcept;
void set_caps(const Caps& caps, uint32_t msaaSamples = 1) noexcept;
void set_built_options(Feature feature, uint32_t optionMask) noexcept;
void set_clock(Clock::time_point (*now)() noexcept) noexcept;
} // namespace testing
} // namespace aurora::render::detail

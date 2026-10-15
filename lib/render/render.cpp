#include "module.hpp"

#include "gpu_heap.hpp"
#include "settings.hpp"

#include "../logging.hpp"
#include "../webgpu/gpu.hpp"

#include <array>

namespace aurora::render {
namespace {
constexpr Module Log{"aurora::render"};

constexpr std::array<std::string_view, 22> ReasonNames{
    "None",
    "NeedsCore",
    "ColorBudget",
    "NoFragmentStorage",
    "NoVertexStorage",
    "NoAtomic64",
    "NoMdi",
    "NeedsD3D12",
    "NativeUnavailable",
    "NativeDisabledAfterCrash",
    "MsaaActive",
    "NeedsRetainedScene",
    "NeedsCompanions",
    "NeedsSunShadows",
    "BindingLimit",
    "LowVram",
    "BuildDisabled",
    "NotImplemented",
    "NeedsPbr",
    "RequiredByShadows",
    "NeedsAo",
    "NeedsRt",
};
static_assert(ReasonNames.size() == static_cast<size_t>(Reason::NeedsRt) + 1);

std::string_view yes_no(bool value) noexcept { return value ? "yes" : "no"; }
} // namespace

namespace detail {
bool module_enabled() noexcept { return AURORA_ENABLE_RENDER != 0; }

void initialize() {
  Log.info("Render module: {} (API {})", module_enabled() ? "enabled" : "disabled at build time", ApiVersion);
  const uint32_t msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  reset_settings_state(msaaSamples);
  const Caps& deviceCaps = caps();
  const Tier tier = device_tier(deviceCaps, msaaSamples);
  Log.info("Render tiers: base {}, PBR {} (core {}, msaa {}, colour budget {} B)", yes_no(tier != Tier::None),
           yes_no(tier == Tier::Pbr), yes_no(deviceCaps.core), msaaSamples,
           deviceCaps.limits.maxColorAttachmentBytesPerSample);
}

void shutdown() noexcept {
#if AURORA_ENABLE_RENDER
  gpu_heap::shutdown();
#endif
  shutdown_settings_state();
}

void on_worker_frame_begin([[maybe_unused]] uint64_t frameId, [[maybe_unused]] const FrameState* frame) {
#if AURORA_ENABLE_RENDER
  // Heap generations bump here, before anything later in the frame compares them
  gpu_heap::begin_frame_worker();
#endif
}
} // namespace detail

Settings apply_settings(const Settings& requested) { return detail::submit_request(requested); }

const Settings& effective_settings() noexcept { return detail::latched_settings(); }

bool feature_enabled(const Settings& settings, Feature feature) noexcept {
  return detail::requested_option(settings, feature) != 0;
}

bool feature_active(Feature feature) noexcept {
  return feature_enabled(effective_settings(), feature) && feature_ready(feature);
}

FeatureSupport feature_support(Feature feature, uint8_t option) noexcept {
  return detail::current_support(feature, option, nullptr);
}

FeatureSupport feature_support(Feature feature, uint8_t option, const Settings& context) noexcept {
  return detail::current_support(feature, option, &context);
}

bool feature_ready(Feature) noexcept { return true; }

std::string_view feature_name(Feature feature) noexcept { return detail::feature_info(feature).name; }

std::string_view reason_name(Reason reason) noexcept {
  const auto index = static_cast<size_t>(reason);
  return index < ReasonNames.size() ? ReasonNames[index] : "Unknown";
}
} // namespace aurora::render

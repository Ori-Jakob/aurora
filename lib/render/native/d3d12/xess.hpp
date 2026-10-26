#pragma once

#include "device.hpp"

#include "../../post/upscale/native_upscale.hpp"

#include <xess/xess_d3d12.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace aurora::render::detail::native::d3d12::xess {
inline constexpr std::string_view RuntimeFile = "libxess.dll";
inline constexpr uint32_t IntelVendorId = 0x8086;
inline constexpr std::string_view DisplayName = "XeSS-SR";
inline constexpr std::string_view XeFxPath = "XeFX";
inline constexpr std::string_view CrossVendorPath = "cross-vendor";

// Types come from Intel's headers in the build tree; their text must never be copied into this repo
struct Api {
  decltype(&xessGetVersion) getVersion = nullptr;
  decltype(&xessGetIntelXeFXVersion) getIntelXeFxVersion = nullptr;
  decltype(&xessGetProperties) getProperties = nullptr;
  decltype(&xessGetOptimalInputResolution) getOptimalInputResolution = nullptr;
  decltype(&xessGetPipelineBuildStatus) getPipelineBuildStatus = nullptr;
  decltype(&xessSetVelocityScale) setVelocityScale = nullptr;
  decltype(&xessSetLoggingCallback) setLoggingCallback = nullptr;
  decltype(&xessIsOptimalDriver) isOptimalDriver = nullptr;
  decltype(&xessDestroyContext) destroyContext = nullptr;
  decltype(&xessD3D12CreateContext) d3d12CreateContext = nullptr;
  decltype(&xessD3D12BuildPipelines) d3d12BuildPipelines = nullptr;
  decltype(&xessD3D12Init) d3d12Init = nullptr;
  decltype(&xessD3D12Execute) d3d12Execute = nullptr;
};

struct Path {
  std::string_view name;
  bool xefx = false;
};

class XessUpscaler final : public upscale::native_upscale::NativeUpscaler {
public:
  [[nodiscard]] Upscaler option() const noexcept override { return Upscaler::Xess; }
  [[nodiscard]] std::string_view name() const noexcept override { return "xess"; }
  [[nodiscard]] std::string_view runtime_file() const noexcept override { return RuntimeFile; }
  upscale::native_upscale::ProbeResult probe(const upscale::native_upscale::Adapter& adapter) noexcept override;
  [[nodiscard]] std::string_view display_name(UpscalerQuality) const noexcept override { return DisplayName; }
  [[nodiscard]] ColorEncoding color_encoding() const noexcept override { return ColorEncoding::Linear; }
  [[nodiscard]] bool wants_reactive() const noexcept override { return false; }
  std::optional<GraphExtent> prepare(const upscale::native_upscale::Adapter& adapter, GraphExtent output,
                                     UpscalerQuality quality) noexcept override;
  bool evaluate(const upscale::native_upscale::Inputs& inputs) noexcept override;
  void destroy() noexcept override {}
  [[nodiscard]] uint64_t vram_bytes() const noexcept override { return 0; }
  [[nodiscard]] uint32_t jitter_phases(float ratio) const noexcept override;
  [[nodiscard]] float mip_bias(float ratio) const noexcept override;
};

[[nodiscard]] FeatureSupport support_from_result(xess_result_t result) noexcept;
[[nodiscard]] Path path_for(uint32_t vendorId, const xess_version_t& xefx) noexcept;
const Api* load() noexcept;
[[nodiscard]] bool loaded() noexcept;
[[nodiscard]] XessUpscaler& upscaler() noexcept;

namespace testing {
void set_library(std::optional<std::filesystem::path> path) noexcept;
[[nodiscard]] uint32_t loads() noexcept;
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::xess

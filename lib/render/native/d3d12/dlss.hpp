#pragma once

#include "device.hpp"

#include "../native.hpp"
#include "../../post/upscale/native_upscale.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::detail::native::d3d12::dlss {
inline constexpr std::string_view RuntimeFile = "nvngx_dlss.dll";
inline constexpr uint32_t NvidiaVendorId = 0x10DE;
inline constexpr std::string_view DisplayName = "DLSS";
inline constexpr std::string_view NativeAaName = "DLAA";
inline constexpr std::string_view DataFolder = "ngx";
inline constexpr GraphExtent MaxOutput{7680, 4320};
inline constexpr GraphExtent MinOutput{64, 32};
inline constexpr size_t ModeCount = 5;

enum class OutputFit : uint8_t { Fits, TooLarge, TooSmall };
enum class InitResult : uint8_t { Ok, Denied, OutOfDate, Failed };

struct Requirements {
  bool driverTooOld = false;
  bool adapterUnsupported = false;
  bool osTooOld = false;
};

struct Capability {
  bool available = false;
  bool needsUpdatedDriver = false;
  uint32_t minDriverMajor = 0;
  uint32_t minDriverMinor = 0;
  InitResult initResult = InitResult::Ok;
};

struct ModeSize {
  GraphExtent optimal;
  GraphExtent minimum;
  GraphExtent maximum;

  bool operator==(const ModeSize&) const = default;
};

using Modes = std::array<std::optional<ModeSize>, ModeCount>;

// NGX calls in aurora types, so this file and its tests build without NVIDIA's headers; dlss_ngx.cpp binds the SDK
struct Api {
  std::optional<Requirements> (*requirements)(IDXGIAdapter* adapter, const Identity& identity,
                                              const std::filesystem::path& data) noexcept = nullptr;
  bool (*init)(ID3D12Device* device, const Identity& identity, const std::filesystem::path& data) noexcept = nullptr;
  std::optional<Capability> (*capability)() noexcept = nullptr;
  std::optional<ModeSize> (*optimal)(GraphExtent output, UpscalerQuality quality) noexcept = nullptr;
  void (*shutdown)(ID3D12Device* device) noexcept = nullptr;
};

class DlssUpscaler final : public upscale::native_upscale::NativeUpscaler {
public:
  [[nodiscard]] Upscaler option() const noexcept override { return Upscaler::Dlss; }
  [[nodiscard]] std::string_view name() const noexcept override { return "dlss"; }
  [[nodiscard]] std::string_view runtime_file() const noexcept override { return RuntimeFile; }
  [[nodiscard]] bool configured() const noexcept override;
  upscale::native_upscale::ProbeResult probe(const upscale::native_upscale::Adapter& adapter) noexcept override;
  [[nodiscard]] std::string_view display_name(UpscalerQuality quality) const noexcept override;
  [[nodiscard]] ColorEncoding color_encoding() const noexcept override { return ColorEncoding::Stored; }
  [[nodiscard]] bool wants_reactive() const noexcept override { return false; }
  std::optional<GraphExtent> prepare(const upscale::native_upscale::Adapter& adapter, GraphExtent output,
                                     UpscalerQuality quality) noexcept override;
  [[nodiscard]] Reason prepare_failure() const noexcept override { return m_failure; }
  upscale::native_upscale::Warmup warmup() noexcept override;
  bool evaluate(const upscale::native_upscale::Inputs& inputs) noexcept override;
  void destroy() noexcept override;
  [[nodiscard]] uint64_t vram_bytes() const noexcept override { return 0; }
  [[nodiscard]] uint32_t jitter_phases(float ratio) const noexcept override;
  [[nodiscard]] float mip_bias(float ratio) const noexcept override;
  [[nodiscard]] bool session() const noexcept { return m_session; }
  [[nodiscard]] const Modes& modes() const noexcept { return m_modes; }

private:
  upscale::native_upscale::ProbeResult probe_steps(const upscale::native_upscale::Adapter& adapter) noexcept;
  FeatureSupport check_capability() noexcept;
  bool begin_session(ID3D12Device* device, const Identity& identity) noexcept;
  void end_session() noexcept;

  const Api* m_api = nullptr;
  ID3D12Device* m_device = nullptr;
  Modes m_modes{};
  GraphExtent m_modesOutput;
  Reason m_failure = Reason::NativeUnavailable;
  bool m_session = false;
  bool m_waiting = false;
};

using FenceWait = void (*)() noexcept;

[[nodiscard]] std::optional<FeatureSupport> refusal_from_requirements(const Requirements& requirements) noexcept;
[[nodiscard]] FeatureSupport support_from_capability(const Capability& capability) noexcept;
[[nodiscard]] OutputFit output_fit(GraphExtent output) noexcept;
[[nodiscard]] bool valid_project_id(std::string_view id) noexcept;
[[nodiscard]] std::optional<Identity> resolve_identity(const std::optional<Identity>& configured,
                                                       const std::optional<std::string>& env);
[[nodiscard]] std::optional<Identity> identity() noexcept;
[[nodiscard]] std::optional<GraphExtent> render_size(const Modes& modes, UpscalerQuality quality,
                                                     GraphExtent output) noexcept;
[[nodiscard]] std::string_view mode_name(UpscalerQuality quality) noexcept;
[[nodiscard]] std::filesystem::path data_directory();
void wait_last_use() noexcept;
void set_api(const Api* api) noexcept;
[[nodiscard]] const Api& ngx_api() noexcept;
[[nodiscard]] DlssUpscaler& upscaler() noexcept;

namespace testing {
void set_project_env(std::optional<std::string> value) noexcept;
void set_data_directory(std::optional<std::filesystem::path> directory) noexcept;
void set_fence_wait(FenceWait wait) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::dlss

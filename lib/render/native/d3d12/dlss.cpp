#include "dlss.hpp"

#include "teardown.hpp"
#include "upscale_bridge.hpp"

#include "../../settings.hpp"
#include "../../../internal.hpp"
#include "../../../io.hpp"
#include "../../../logging.hpp"

#include <SDL3/SDL_stdinc.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <system_error>
#include <utility>

namespace aurora::render::detail::native::d3d12::dlss {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr std::string_view ProjectIdEnv = "AURORA_RENDER_NGX_PROJECT_ID";

using upscale::native_upscale::Adapter;
using upscale::native_upscale::Backend;
using upscale::native_upscale::ProbeResult;
using upscale::native_upscale::Warmup;

void wait_bridge() noexcept { upscale_bridge::wait_idle(); }

const Api* g_api = nullptr;
FenceWait g_fenceWait = &wait_bridge;
DlssUpscaler g_upscaler;
std::atomic_bool g_envWarned{false};
bool g_envOverridden = false;
std::optional<std::string> g_envOverride;
std::optional<std::filesystem::path> g_dataOverride;

std::optional<std::string> project_env() {
  if (g_envOverridden) {
    return g_envOverride;
  }
  static const std::optional<std::string> value = []() -> std::optional<std::string> {
    const char* env = SDL_getenv(ProjectIdEnv.data());
    if (env == nullptr || *env == '\0') {
      return std::nullopt;
    }
    return std::string{env};
  }();
  return value;
}

bool is_hex(char c) noexcept { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

double ms_since(std::chrono::steady_clock::time_point start) noexcept {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

ID3D12Device* d3d12_device(const Adapter& adapter) noexcept {
  return adapter.backend == Backend::D3D12 ? static_cast<ID3D12Device*>(adapter.device) : nullptr;
}

void teardown() noexcept { g_upscaler.destroy(); }
} // namespace

std::optional<FeatureSupport> refusal_from_requirements(const Requirements& requirements) noexcept {
  if (requirements.osTooOld) {
    return FeatureSupport{false, Reason::NativeUnavailable};
  }
  if (requirements.driverTooOld) {
    return FeatureSupport{false, Reason::DriverTooOld};
  }
  if (requirements.adapterUnsupported) {
    return FeatureSupport{false, Reason::UnsupportedGpu};
  }
  return std::nullopt;
}

FeatureSupport support_from_capability(const Capability& capability) noexcept {
  if (capability.initResult == InitResult::Denied) {
    return {false, Reason::NativeUnavailable};
  }
  if (capability.needsUpdatedDriver) {
    return {false, Reason::DriverTooOld};
  }
  if (capability.available) {
    return {true, Reason::None};
  }
  if (capability.initResult == InitResult::OutOfDate) {
    return {false, Reason::DriverTooOld};
  }
  return {false, Reason::UnsupportedGpu};
}

OutputFit output_fit(GraphExtent output) noexcept {
  if (output.width > MaxOutput.width || output.height > MaxOutput.height) {
    return OutputFit::TooLarge;
  }
  if (output.width < MinOutput.width || output.height < MinOutput.height) {
    return OutputFit::TooSmall;
  }
  return OutputFit::Fits;
}

bool valid_project_id(std::string_view id) noexcept {
  constexpr std::array<size_t, 4> Dashes{8, 13, 18, 23};
  if (id.size() != 36) {
    return false;
  }
  for (size_t i = 0; i < id.size(); ++i) {
    const bool dash = std::ranges::find(Dashes, i) != Dashes.end();
    if (dash ? id[i] != '-' : !is_hex(id[i])) {
      return false;
    }
  }
  return true;
}

std::optional<Identity> resolve_identity(const std::optional<Identity>& configured,
                                         const std::optional<std::string>& env) {
  const std::string engineVersion = std::to_string(ApiVersion);
  if (configured) {
    if (configured->projectId.empty()) {
      return std::nullopt;
    }
    Identity out = *configured;
    if (out.engineVersion.empty()) {
      out.engineVersion = engineVersion;
    }
    return out;
  }
  if (env && !env->empty()) {
    return Identity{.projectId = *env, .engineVersion = engineVersion};
  }
  return std::nullopt;
}

std::optional<Identity> identity() noexcept {
  const std::optional<Identity> configured = app_identity();
  std::optional<Identity> out = resolve_identity(configured, configured ? std::nullopt : project_env());
  if (!configured && out && !g_envWarned.exchange(true)) {
    Log.warn("{}: DLSS runs under a development NGX project id", ProjectIdEnv);
  }
  return out;
}

std::optional<GraphExtent> render_size(const Modes& modes, UpscalerQuality quality, GraphExtent output) noexcept {
  const auto index = static_cast<size_t>(quality);
  if (index >= modes.size()) {
    return std::nullopt;
  }
  const std::optional<ModeSize>& mode = modes[index];
  const bool sized = mode && mode->optimal.width != 0 && mode->optimal.height != 0;
  if (!sized) {
    return quality == UpscalerQuality::NativeAa ? std::optional{output} : std::nullopt;
  }
  return GraphExtent{std::min(mode->optimal.width, output.width), std::min(mode->optimal.height, output.height)};
}

std::string_view mode_name(UpscalerQuality quality) noexcept {
  switch (quality) {
  case UpscalerQuality::NativeAa:
    return NativeAaName;
  case UpscalerQuality::Quality:
    return "Quality";
  case UpscalerQuality::Balanced:
    return "Balanced";
  case UpscalerQuality::Performance:
    return "Performance";
  case UpscalerQuality::UltraPerformance:
    return "Ultra Performance";
  }
  return "Unknown";
}

std::filesystem::path data_directory() {
  if (g_dataOverride) {
    return *g_dataOverride;
  }
  if (g_config.cachePath != nullptr) {
    return io::fs_path_from_string(g_config.cachePath) / DataFolder;
  }
  std::error_code ec;
  const std::filesystem::path temp = std::filesystem::temp_directory_path(ec);
  return ec ? std::filesystem::path{} : temp / "aurora" / DataFolder;
}

void wait_last_use() noexcept {
  if (g_fenceWait != nullptr) {
    g_fenceWait();
  }
}

void set_api(const Api* api) noexcept { g_api = api; }

DlssUpscaler& upscaler() noexcept { return g_upscaler; }

bool DlssUpscaler::configured() const noexcept {
  const std::optional<Identity> id = identity();
  if (!id) {
    return false;
  }
  if (!valid_project_id(id->projectId)) {
    Log.warn("{}: the NGX project id '{}' is not a GUID", DisplayName, id->projectId);
    return false;
  }
  return true;
}

ProbeResult DlssUpscaler::probe(const Adapter& adapter) noexcept {
  native::assert_worker();
  const auto start = std::chrono::steady_clock::now();
  const ProbeResult result = probe_steps(adapter);
  Log.info("{}: {} (probe {:.1f} ms)", DisplayName,
           result.support.supported ? std::string_view{"available"} : reason_name(result.support.reason),
           ms_since(start));
  return result;
}

ProbeResult DlssUpscaler::probe_steps(const Adapter& adapter) noexcept {
  if (adapter.vendorId != NvidiaVendorId) {
    Log.info("{}: adapter vendor {:#06x} (LUID {:016x}) is not NVIDIA", DisplayName, adapter.vendorId, adapter.luid);
    return {.support = {false, Reason::UnsupportedGpu}, .autoPick = false};
  }
  ID3D12Device* device = d3d12_device(adapter);
  const std::optional<Identity> id = identity();
  if (g_api == nullptr || device == nullptr || !id || !valid_project_id(id->projectId)) {
    return {.support = {false, Reason::NativeUnavailable}, .autoPick = false};
  }
  const std::filesystem::path data = data_directory();
  std::error_code ec;
  std::filesystem::create_directories(data, ec);
  const Upscaler requested = requested_upscaler();
  const bool selected = requested == Upscaler::Dlss || requested == Upscaler::Auto;
  auto step = std::chrono::steady_clock::now();
  // NGX initialises only for users who pick DLSS or Auto; everyone else gets NVIDIA's pre-init check
  if (!selected && !m_session) {
    const std::optional<Requirements> requirements =
        g_api->requirements(static_cast<IDXGIAdapter*>(adapter.physical), *id, data);
    Log.debug("{} probe: feature requirements {:.1f} ms", DisplayName, ms_since(step));
    if (requirements) {
      const std::optional<FeatureSupport> refusal = refusal_from_requirements(*requirements);
      return {.support = refusal.value_or(FeatureSupport{true, Reason::None}), .autoPick = !refusal};
    }
  }
  if (!begin_session(device, *id)) {
    return {.support = {false, Reason::NativeUnavailable}, .autoPick = false};
  }
  step = std::chrono::steady_clock::now();
  const FeatureSupport support = check_capability();
  Log.debug("{} probe: capabilities {:.1f} ms", DisplayName, ms_since(step));
  if (!support.supported || !selected) {
    end_session();
  }
  return {.support = support, .autoPick = support.supported};
}

FeatureSupport DlssUpscaler::check_capability() noexcept {
  const std::optional<Capability> capability = m_api->capability();
  if (!capability) {
    return {false, Reason::NativeUnavailable};
  }
  const FeatureSupport support = support_from_capability(*capability);
  if (support.reason == Reason::DriverTooOld) {
    Log.warn("{} needs NVIDIA driver {}.{} or newer", DisplayName, capability->minDriverMajor,
             capability->minDriverMinor);
  }
  return support;
}

std::optional<GraphExtent> DlssUpscaler::prepare(const Adapter& adapter, GraphExtent output,
                                                 UpscalerQuality quality) noexcept {
  native::assert_worker();
  m_failure = Reason::NativeUnavailable;
  m_waiting = false;
  switch (output_fit(output)) {
  case OutputFit::TooLarge:
    Log.warn("{}: a {}x{} output is above its {}x{} limit", DisplayName, output.width, output.height, MaxOutput.width,
             MaxOutput.height);
    m_failure = Reason::OutputTooLarge;
    return std::nullopt;
  case OutputFit::TooSmall:
    m_waiting = true;
    return output;
  case OutputFit::Fits:
    break;
  }
  ID3D12Device* device = d3d12_device(adapter);
  const std::optional<Identity> id = identity();
  const bool fresh = !m_session;
  if (g_api == nullptr || device == nullptr || !id || !begin_session(device, *id)) {
    return std::nullopt;
  }
  if (fresh) {
    const FeatureSupport support = check_capability();
    if (!support.supported) {
      Log.warn("{}: NGX reports it cannot run here ({})", DisplayName, reason_name(support.reason));
      m_failure = support.reason;
      end_session();
      return std::nullopt;
    }
  }
  if (m_modesOutput != output) {
    m_modes = {};
    std::string sizes;
    for (size_t i = 0; i < m_modes.size(); ++i) {
      const auto mode = static_cast<UpscalerQuality>(i);
      m_modes[i] = m_api->optimal(output, mode);
      if (const std::optional<GraphExtent> size = render_size(m_modes, mode, output)) {
        fmt::format_to(std::back_inserter(sizes), "{}{} {}x{}", sizes.empty() ? "" : ", ", mode_name(mode), size->width,
                       size->height);
      }
    }
    m_modesOutput = output;
    Log.info("{} render sizes for {}x{}: {}", DisplayName, output.width, output.height, sizes);
  }
  const std::optional<GraphExtent> render = render_size(m_modes, quality, output);
  if (!render) {
    Log.warn("{}: NGX gave no {} render size for {}x{}", DisplayName, mode_name(quality), output.width, output.height);
    return std::nullopt;
  }
  want_feature({.render = *render, .output = output, .quality = quality, .flags = feature_flags()});
  return render;
}

Warmup DlssUpscaler::warmup() noexcept { return m_waiting ? Warmup::Pending : Warmup::Ready; }

bool DlssUpscaler::begin_session(ID3D12Device* device, const Identity& identity) noexcept {
  if (m_session) {
    return true;
  }
  const std::filesystem::path data = data_directory();
  std::error_code ec;
  if (data.empty() || (!std::filesystem::create_directories(data, ec) && ec)) {
    Log.warn("{}: no writable NGX data folder", DisplayName);
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  if (!g_api->init(device, identity, data)) {
    Log.warn("{}: NGX did not initialise", DisplayName);
    return false;
  }
  Log.info("{}: NGX initialised ({:.1f} ms)", DisplayName, ms_since(start));
  m_api = g_api;
  m_device = device;
  m_session = true;
  register_teardown(&teardown);
  return true;
}

void DlssUpscaler::end_session() noexcept {
  if (!m_session) {
    return;
  }
  native::assert_worker();
  wait_last_use();
  drop_feature();
  m_api->shutdown(m_device);
  m_api = nullptr;
  m_device = nullptr;
  m_session = false;
}

namespace testing {
void set_project_env(std::optional<std::string> value) noexcept {
  g_envOverridden = true;
  g_envOverride = std::move(value);
}

void set_data_directory(std::optional<std::filesystem::path> directory) noexcept {
  g_dataOverride = std::move(directory);
}

void set_fence_wait(FenceWait wait) noexcept { g_fenceWait = wait != nullptr ? wait : &wait_bridge; }

void reset() noexcept {
  g_envOverridden = false;
  g_envOverride.reset();
  g_dataOverride.reset();
  g_fenceWait = &wait_bridge;
}
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::dlss

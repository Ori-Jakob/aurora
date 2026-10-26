#include "xess.hpp"

#include "teardown.hpp"

#include "../native.hpp"
#include "../../../logging.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <utility>

namespace aurora::render::detail::native::d3d12::xess {
namespace {
constexpr Module Log{"aurora::render::native"};

using upscale::native_upscale::Adapter;
using upscale::native_upscale::Backend;
using upscale::native_upscale::ProbeResult;

constexpr uint32_t MinJitterPhases = 8;

HMODULE g_module = nullptr;
Api g_api;

std::atomic_bool g_loaded{false};
std::atomic_uint32_t g_loads{0};
std::optional<std::filesystem::path> g_library;
const Api* g_apiOverride = nullptr;
XessUpscaler g_upscaler;

template <typename T>
bool resolve(HMODULE module, const char* name, T& out) noexcept {
  out = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
  if (out == nullptr) {
    Log.warn("{} has no export {}", RuntimeFile, name);
    return false;
  }
  return true;
}

bool resolve_api(HMODULE module, Api& api) noexcept {
  return resolve(module, "xessGetVersion", api.getVersion) &&
         resolve(module, "xessGetIntelXeFXVersion", api.getIntelXeFxVersion) &&
         resolve(module, "xessGetProperties", api.getProperties) &&
         resolve(module, "xessGetOptimalInputResolution", api.getOptimalInputResolution) &&
         resolve(module, "xessGetPipelineBuildStatus", api.getPipelineBuildStatus) &&
         resolve(module, "xessSetVelocityScale", api.setVelocityScale) &&
         resolve(module, "xessSetLoggingCallback", api.setLoggingCallback) &&
         resolve(module, "xessIsOptimalDriver", api.isOptimalDriver) &&
         resolve(module, "xessDestroyContext", api.destroyContext) &&
         resolve(module, "xessD3D12CreateContext", api.d3d12CreateContext) &&
         resolve(module, "xessD3D12BuildPipelines", api.d3d12BuildPipelines) &&
         resolve(module, "xessD3D12Init", api.d3d12Init) && resolve(module, "xessD3D12Execute", api.d3d12Execute);
}

HMODULE open_library() noexcept {
  if (g_library) {
    return LoadLibraryExW(g_library->c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  }
  return LoadLibraryExW(L"libxess.dll", nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
}

void unload() noexcept {
  g_upscaler.destroy();
  if (g_module == nullptr) {
    return;
  }
  FreeLibrary(g_module);
  g_module = nullptr;
  g_api = {};
  g_loaded.store(false, std::memory_order_release);
}

void forward_log(const char* message, xess_logging_level_t level) {
  const std::string_view text = message != nullptr ? std::string_view{message} : std::string_view{};
  switch (level) {
  case XESS_LOGGING_LEVEL_ERROR:
    Log.error("XeSS-SR: {}", text);
    break;
  case XESS_LOGGING_LEVEL_WARNING:
    Log.warn("XeSS-SR: {}", text);
    break;
  case XESS_LOGGING_LEVEL_INFO:
    Log.info("XeSS-SR: {}", text);
    break;
  default:
    Log.debug("XeSS-SR: {}", text);
    break;
  }
}
} // namespace

FeatureSupport support_from_result(xess_result_t result) noexcept {
  switch (result) {
  case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE:
    return {false, Reason::UnsupportedGpu};
  case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER:
    return {false, Reason::DriverTooOld};
  case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY:
    return {false, Reason::RuntimeMissing};
  default:
    break;
  }
  if (result >= XESS_RESULT_SUCCESS) {
    return {true, Reason::None};
  }
  return {false, Reason::NativeUnavailable};
}

Path path_for(uint32_t vendorId, const xess_version_t& xefx) noexcept {
  const bool present = xefx.major != 0 || xefx.minor != 0 || xefx.patch != 0;
  if (vendorId == IntelVendorId && present) {
    return {.name = XeFxPath, .xefx = true};
  }
  return {.name = CrossVendorPath, .xefx = false};
}

xess_result_t create_context(const Api& api, ID3D12Device* device, xess_context_handle_t& context) noexcept {
  context = nullptr;
  const xess_result_t created = api.d3d12CreateContext(device, &context);
  if (created >= XESS_RESULT_SUCCESS && context != nullptr) {
    api.setLoggingCallback(context, XESS_LOGGING_LEVEL_WARNING, &forward_log);
  }
  return created;
}

const Api* load() noexcept {
  native::assert_worker();
  if (g_apiOverride != nullptr) {
    return g_apiOverride;
  }
  if (g_module != nullptr) {
    return &g_api;
  }
  HMODULE module = open_library();
  if (module == nullptr) {
    Log.info("{} could not be loaded (error {})", RuntimeFile, GetLastError());
    return nullptr;
  }
  Api api;
  if (!resolve_api(module, api)) {
    FreeLibrary(module);
    return nullptr;
  }
  g_module = module;
  g_api = api;
  g_loaded.store(true, std::memory_order_release);
  g_loads.fetch_add(1, std::memory_order_relaxed);
  register_teardown(&unload);
  xess_version_t version{};
  if (g_api.getVersion(&version) == XESS_RESULT_SUCCESS) {
    Log.info("Loaded {} {}.{}.{}", RuntimeFile, version.major, version.minor, version.patch);
  }
  return &g_api;
}

bool loaded() noexcept { return g_loaded.load(std::memory_order_acquire); }

XessUpscaler& upscaler() noexcept { return g_upscaler; }

ProbeResult XessUpscaler::probe(const Adapter& adapter) noexcept {
  native::assert_worker();
  const Api* api = load();
  if (api == nullptr) {
    return {.support = {false, Reason::RuntimeMissing}, .autoPick = false};
  }
  auto* device = adapter.backend == Backend::D3D12 ? static_cast<ID3D12Device*>(adapter.device) : nullptr;
  if (device == nullptr) {
    return {.support = {false, Reason::NativeUnavailable}, .autoPick = false};
  }
  xess_context_handle_t context = nullptr;
  const xess_result_t created = create_context(*api, device, context);
  FeatureSupport support = support_from_result(created);
  if (support.supported && context == nullptr) {
    support = {false, Reason::NativeUnavailable};
  }
  if (!support.supported) {
    if (context != nullptr) {
      api->destroyContext(context);
    }
    Log.info("{}: no context on this device (result {}, {})", DisplayName, static_cast<int>(created),
             reason_name(support.reason));
    return {.support = support, .autoPick = false};
  }
  if (api->isOptimalDriver(context) == XESS_RESULT_WARNING_OLD_DRIVER) {
    Log.warn("{}: the installed GPU driver may give degraded performance or quality", DisplayName);
  }
  xess_version_t xefx{};
  if (api->getIntelXeFxVersion(context, &xefx) != XESS_RESULT_SUCCESS) {
    xefx = {};
  }
  api->destroyContext(context);
  const Path path = path_for(adapter.vendorId, xefx);
  if (path.xefx) {
    Log.info("{} ({} {}.{}.{})", DisplayName, path.name, xefx.major, xefx.minor, xefx.patch);
  } else {
    Log.info("{} ({})", DisplayName, path.name);
  }
  return {.support = {true, Reason::None}, .path = std::string{path.name}, .autoPick = path.xefx};
}

uint32_t XessUpscaler::jitter_phases(float ratio) const noexcept {
  return std::max(MinJitterPhases, static_cast<uint32_t>(std::ceil(8.f * ratio * ratio)));
}

float XessUpscaler::mip_bias(float ratio) const noexcept { return std::log2(1.f / ratio); }

namespace testing {
void set_library(std::optional<std::filesystem::path> path) noexcept { g_library = std::move(path); }

uint32_t loads() noexcept { return g_loads.load(std::memory_order_relaxed); }

void set_api(const Api* api) noexcept { g_apiOverride = api; }
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::xess

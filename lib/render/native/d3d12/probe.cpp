#include "probe.hpp"

#include "device.hpp"
#include "sentinel.hpp"
#include "upscale_bridge.hpp"

#include "../../json.hpp"
#include "../../settings.hpp"
#include "../../../io.hpp"
#include "../../../logging.hpp"

#include <aurora/render_caps.hpp>

#include <winver.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace aurora::render::detail::native::d3d12::probe {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr size_t MaxModulePath = 32768;

using upscale::native_upscale::NativeUpscaler;
using upscale::native_upscale::ProbeResult;

constexpr std::array<Upscaler, 2> Options{Upscaler::Dlss, Upscaler::Xess};
constexpr std::array<const char*, 2> Keys{"dlss", "xess"};

struct Runtime {
  bool present = true;
  uint64_t version = 0;
  uint64_t size = 0;

  bool operator==(const Runtime&) const = default;
};

struct Cached {
  bool valid = false;
  Runtime runtime;
  FeatureSupport support;
  std::string_view path;
  bool autoPick = false;
};

struct State {
  std::filesystem::path file;
  std::optional<Identity> identity;
  bool usable = false;
  UpscalerProbe probe;
  std::array<Runtime, 2> runtimes;
  std::array<Cached, 2> cache;
  std::array<std::string_view, 2> paths;
};

std::mutex g_mutex;
State g_state;
std::deque<std::string> g_strings;
std::atomic_bool g_requested{false};
std::atomic_uint32_t g_sdkCalls{0};
bool g_identitySet = false;
std::optional<Identity> g_identityOverride;
std::optional<std::filesystem::path> g_runtimeOverride;

constexpr size_t index_of(Upscaler option) noexcept { return option == Upscaler::Dlss ? 0 : 1; }

std::string_view intern_locked(std::string_view text) {
  if (text.empty()) {
    return {};
  }
  for (const std::string& known : g_strings) {
    if (known == text) {
      return known;
    }
  }
  return g_strings.emplace_back(text);
}

std::optional<Identity> device_identity() noexcept {
  if (g_identitySet) {
    return g_identityOverride;
  }
  if (!available()) {
    return std::nullopt;
  }
  const DXGI_ADAPTER_DESC1& desc = adapter_desc();
  return Identity{desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision, driver_version()};
}

std::filesystem::path runtime_directory() {
  if (g_runtimeOverride) {
    return *g_runtimeOverride;
  }
  std::wstring buffer(MAX_PATH, L'\0');
  while (buffer.size() <= MaxModulePath) {
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return {};
    }
    if (length < buffer.size()) {
      buffer.resize(length);
      return std::filesystem::path{buffer}.parent_path();
    }
    buffer.resize(buffer.size() * 2);
  }
  return {};
}

Runtime runtime_info(std::string_view file) {
  if (file.empty()) {
    return {};
  }
  const std::filesystem::path path = runtime_directory() / std::filesystem::path{std::string{file}};
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    return {.present = false};
  }
  Runtime out{.present = true, .size = size};
  DWORD handle = 0;
  const DWORD bytes = GetFileVersionInfoSizeW(path.c_str(), &handle);
  std::vector<std::byte> info(bytes);
  VS_FIXEDFILEINFO* fixed = nullptr;
  UINT fixedSize = 0;
  if (bytes != 0 && GetFileVersionInfoW(path.c_str(), 0, bytes, info.data()) != FALSE &&
      VerQueryValueW(info.data(), L"\\", reinterpret_cast<void**>(&fixed), &fixedSize) != FALSE && fixed != nullptr &&
      fixedSize >= sizeof(VS_FIXEDFILEINFO)) {
    out.version = (uint64_t{fixed->dwFileVersionMS} << 32) | fixed->dwFileVersionLS;
  }
  return out;
}

std::array<Cached, 2> read_cache_locked(const Identity& identity) {
  std::array<Cached, 2> out;
  if (g_state.file.empty()) {
    return out;
  }
  const auto bytes = io::read_file(g_state.file);
  if (!bytes) {
    return out;
  }
  const json::Doc doc = json::read({reinterpret_cast<const char*>(bytes->data()), bytes->size()});
  yyjson_val* root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  const auto number = [](yyjson_val* object, const char* key) { return yyjson_get_uint(yyjson_obj_get(object, key)); };
  if (!yyjson_is_obj(root) || number(root, "schema") != SchemaVersion ||
      number(root, "vendorId") != identity.vendorId || number(root, "deviceId") != identity.deviceId ||
      number(root, "subSysId") != identity.subSysId || number(root, "revision") != identity.revision ||
      number(root, "driver") != identity.driverVersion) {
    return out;
  }
  yyjson_val* entries = yyjson_obj_get(root, "entries");
  for (size_t i = 0; i < Keys.size(); ++i) {
    yyjson_val* entry = yyjson_obj_get(entries, Keys[i]);
    const uint64_t reason = number(entry, "reason");
    if (!yyjson_is_obj(entry) || reason > static_cast<uint64_t>(Reason::Probing)) {
      continue;
    }
    const char* path = yyjson_get_str(yyjson_obj_get(entry, "path"));
    out[i] = {
        .valid = true,
        .runtime = {.present = true, .version = number(entry, "runtimeVersion"), .size = number(entry, "runtimeSize")},
        .support = {yyjson_get_bool(yyjson_obj_get(entry, "supported")), static_cast<Reason>(reason)},
        .path = intern_locked(path != nullptr ? std::string_view{path} : std::string_view{}),
        .autoPick = yyjson_get_bool(yyjson_obj_get(entry, "autoPick")),
    };
  }
  return out;
}

void write_cache_locked() {
  if (g_state.file.empty() || !g_state.identity) {
    return;
  }
  const Identity& identity = *g_state.identity;
  const json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  yyjson_mut_doc* d = doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(d);
  yyjson_mut_doc_set_root(d, root);
  yyjson_mut_obj_add_uint(d, root, "schema", SchemaVersion);
  yyjson_mut_obj_add_uint(d, root, "vendorId", identity.vendorId);
  yyjson_mut_obj_add_uint(d, root, "deviceId", identity.deviceId);
  yyjson_mut_obj_add_uint(d, root, "subSysId", identity.subSysId);
  yyjson_mut_obj_add_uint(d, root, "revision", identity.revision);
  yyjson_mut_obj_add_uint(d, root, "driver", identity.driverVersion);
  yyjson_mut_val* entries = yyjson_mut_obj_add_obj(d, root, "entries");
  for (size_t i = 0; i < Keys.size(); ++i) {
    const Cached& cached = g_state.cache[i];
    if (!cached.valid) {
      continue;
    }
    yyjson_mut_val* entry = yyjson_mut_obj_add_obj(d, entries, Keys[i]);
    yyjson_mut_obj_add_uint(d, entry, "runtimeVersion", cached.runtime.version);
    yyjson_mut_obj_add_uint(d, entry, "runtimeSize", cached.runtime.size);
    yyjson_mut_obj_add_bool(d, entry, "supported", cached.support.supported);
    yyjson_mut_obj_add_uint(d, entry, "reason", static_cast<uint64_t>(cached.support.reason));
    yyjson_mut_obj_add_strncpy(d, entry, "path", cached.path.data(), cached.path.size());
    yyjson_mut_obj_add_bool(d, entry, "autoPick", cached.autoPick);
  }
  const std::string text = json::write(d, true);
  std::error_code ec;
  std::filesystem::create_directories(g_state.file.parent_path(), ec);
  const auto* data = reinterpret_cast<const uint8_t*>(text.data());
  if (!io::write_file_atomic(g_state.file, std::span{data, text.size()})) {
    Log.warn("Failed to write {}", io::fs_path_to_string(g_state.file));
  }
}

ProbeEntry entry_locked(size_t index, uint8_t stale) noexcept {
  const NativeUpscaler* upscaler = upscale_bridge::upscaler(Options[index]);
  if (upscaler == nullptr) {
    return {.known = true, .support = {false, Reason::BuildDisabled}};
  }
  if (!g_state.usable) {
    return {.known = true, .support = {false, Reason::NativeUnavailable}};
  }
  if (stale != 0) {
    return {.known = true, .support = {false, Reason::NativeDisabledAfterCrash}};
  }
  if (!g_state.runtimes[index].present) {
    return {.known = true, .support = {false, Reason::RuntimeMissing}};
  }
  const Cached& cached = g_state.cache[index];
  if (cached.valid && cached.runtime == g_state.runtimes[index]) {
    g_state.paths[index] = cached.path;
    return {.known = true, .support = cached.support};
  }
  return {.known = false, .support = {false, Reason::Probing}};
}

void rebuild_entries_locked() noexcept {
  const uint8_t stale = sentinel::stale_at_startup();
  g_state.paths = {};
  for (size_t i = 0; i < Options.size(); ++i) {
    g_state.probe.native[i] = entry_locked(i, stale);
  }
  const ProbeEntry& xess = g_state.probe.native[index_of(Upscaler::Xess)];
  g_state.probe.xessXeFx = xess.known && xess.support.supported && g_state.cache[index_of(Upscaler::Xess)].autoPick;
  g_state.probe.vendorId = g_state.identity ? g_state.identity->vendorId : 0;
}
} // namespace

void initialize(const std::filesystem::path& directory) noexcept {
  {
    std::lock_guard lock{g_mutex};
    g_state = {};
    g_state.file = directory.empty() ? std::filesystem::path{} : directory / CacheFile;
    g_state.identity = device_identity();
    const Caps& deviceCaps = caps();
    g_state.usable =
        g_state.identity.has_value() && !killed() && deviceCaps.d3d12SharedTexture && deviceCaps.d3d12SharedFence;
    if (g_state.usable) {
      for (size_t i = 0; i < Options.size(); ++i) {
        if (const NativeUpscaler* upscaler = upscale_bridge::upscaler(Options[i])) {
          g_state.runtimes[i] = runtime_info(upscaler->runtime_file());
        }
      }
      g_state.cache = read_cache_locked(*g_state.identity);
    }
    rebuild_entries_locked();
    g_requested.store(false, std::memory_order_relaxed);
    set_upscaler_probe(g_state.probe);
  }
  set_probe_requester(&request);
}

void shutdown() noexcept {
  set_probe_requester(nullptr);
  std::lock_guard lock{g_mutex};
  g_state = {};
  g_requested.store(false, std::memory_order_relaxed);
}

void request() noexcept { g_requested.store(true, std::memory_order_release); }

void run_pending() noexcept {
  if (!g_requested.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  std::lock_guard lock{g_mutex};
  if (!g_state.usable) {
    return;
  }
  bool probed = false;
  for (size_t i = 0; i < Options.size(); ++i) {
    NativeUpscaler* upscaler = upscale_bridge::upscaler(Options[i]);
    if (g_state.probe.native[i].known || upscaler == nullptr) {
      continue;
    }
    sentinel::arm(sentinel::Probe);
    count_sdk_call();
    const ProbeResult result = upscaler->probe(upscale_bridge::adapter());
    sentinel::disarm(sentinel::Probe);
    g_state.probe.native[i] = {.known = true, .support = result.support};
    g_state.paths[i] = intern_locked(result.path);
    g_state.cache[i] = {
        .valid = true,
        .runtime = g_state.runtimes[i],
        .support = result.support,
        .path = g_state.paths[i],
        .autoPick = result.autoPick,
    };
    if (Options[i] == Upscaler::Xess) {
      g_state.probe.xessXeFx = result.support.supported && result.autoPick;
    }
    Log.info("Native upscaler {}: {}{}{}", upscaler->name(),
             result.support.supported ? std::string_view{"supported"} : reason_name(result.support.reason),
             result.path.empty() ? "" : ", ", result.path);
    probed = true;
  }
  if (probed) {
    write_cache_locked();
    set_upscaler_probe(g_state.probe);
  }
}

void fail(Upscaler option) noexcept {
  if (option != Upscaler::Dlss && option != Upscaler::Xess) {
    return;
  }
  std::lock_guard lock{g_mutex};
  const size_t index = index_of(option);
  g_state.probe.native[index] = {.known = true, .support = {false, Reason::NativeUnavailable}};
  if (option == Upscaler::Xess) {
    g_state.probe.xessXeFx = false;
  }
  g_state.cache[index].valid = false;
  write_cache_locked();
  set_upscaler_probe(g_state.probe);
  Log.warn("Native upscaler {} failed; it stays off until the next launch probes it again", Keys[index]);
}

void clear() noexcept {
  std::lock_guard lock{g_mutex};
  g_state.cache = {};
  if (!g_state.file.empty()) {
    std::error_code ec;
    std::filesystem::remove(g_state.file, ec);
  }
  rebuild_entries_locked();
  set_upscaler_probe(g_state.probe);
}

std::string_view path(Upscaler option) noexcept {
  if (option != Upscaler::Dlss && option != Upscaler::Xess) {
    return {};
  }
  std::lock_guard lock{g_mutex};
  return g_state.paths[index_of(option)];
}

void count_sdk_call() noexcept { g_sdkCalls.fetch_add(1, std::memory_order_relaxed); }

uint32_t sdk_calls() noexcept { return g_sdkCalls.load(std::memory_order_relaxed); }

namespace testing {
void set_identity(std::optional<Identity> identity) noexcept {
  g_identitySet = true;
  g_identityOverride = identity;
}

void set_runtime_directory(std::optional<std::filesystem::path> directory) noexcept {
  g_runtimeOverride = std::move(directory);
}

void reset() noexcept {
  set_probe_requester(nullptr);
  g_identitySet = false;
  g_identityOverride.reset();
  g_runtimeOverride.reset();
  g_sdkCalls.store(0, std::memory_order_relaxed);
  g_requested.store(false, std::memory_order_relaxed);
  std::lock_guard lock{g_mutex};
  g_state = {};
}
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::probe

#include "sentinel.hpp"

#include "../../../io.hpp"
#include "../../../logging.hpp"

#include <array>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace aurora::render::detail::native::d3d12::sentinel {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr std::string_view FileName = "native_upscaler.sentinel";
constexpr std::array<std::pair<Phase, std::string_view>, 2> PhaseNames{{{Probe, "probe"}, {Session, "session"}}};

std::mutex g_mutex;
std::filesystem::path g_path;
uint8_t g_stale = 0;
uint8_t g_armed = 0;

uint8_t parse(std::string_view text) noexcept {
  uint8_t phases = 0;
  for (const auto& [phase, name] : PhaseNames) {
    if (text.find(name) != std::string_view::npos) {
      phases |= phase;
    }
  }
  return phases != 0 ? phases : uint8_t{Session};
}

std::string describe(uint8_t phases, std::string_view separator) {
  std::string text;
  for (const auto& [phase, name] : PhaseNames) {
    if ((phases & phase) != 0) {
      if (!text.empty()) {
        text += separator;
      }
      text += name;
    }
  }
  return text;
}

void write_locked() noexcept {
  if (g_path.empty()) {
    return;
  }
  const uint8_t phases = g_stale | g_armed;
  if (phases == 0) {
    std::error_code ec;
    std::filesystem::remove(g_path, ec);
    return;
  }
  const std::string text = describe(phases, "\n") + "\n";
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  if (!io::write_file(g_path, std::span{bytes, text.size()})) {
    Log.warn("Failed to write {}", io::fs_path_to_string(g_path));
  }
}
} // namespace

void open(const std::filesystem::path& directory) noexcept {
  std::lock_guard lock{g_mutex};
  g_stale = 0;
  g_armed = 0;
  g_path = directory.empty() ? std::filesystem::path{} : directory / FileName;
  std::error_code ec;
  if (g_path.empty() || !std::filesystem::exists(g_path, ec)) {
    return;
  }
  const auto bytes = io::read_file(g_path);
  g_stale = bytes ? parse({reinterpret_cast<const char*>(bytes->data()), bytes->size()}) : uint8_t{Session};
  Log.warn("The previous run stopped during a native upscaler {}; native upscalers stay off until re-enabled",
           describe(g_stale, " and "));
}

uint8_t stale_at_startup() noexcept {
  std::lock_guard lock{g_mutex};
  return g_stale;
}

void arm(Phase phase) noexcept {
  std::lock_guard lock{g_mutex};
  if ((g_armed & phase) != 0) {
    return;
  }
  g_armed |= phase;
  write_locked();
}

void disarm(Phase phase) noexcept {
  std::lock_guard lock{g_mutex};
  if ((g_armed & phase) == 0) {
    return;
  }
  g_armed &= static_cast<uint8_t>(~phase);
  write_locked();
}

void clear() noexcept {
  std::lock_guard lock{g_mutex};
  g_stale = 0;
  write_locked();
}
} // namespace aurora::render::detail::native::d3d12::sentinel

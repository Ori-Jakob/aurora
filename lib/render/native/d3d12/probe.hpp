#pragma once

#include <aurora/render.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace aurora::render::detail::native::d3d12::probe {
inline constexpr uint32_t SchemaVersion = 1;
inline constexpr std::string_view CacheFile = "native_upscaler_probe.json";

struct Identity {
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  uint32_t subSysId = 0;
  uint32_t revision = 0;
  uint64_t driverVersion = 0;

  bool operator==(const Identity&) const = default;
};

void initialize(const std::filesystem::path& directory) noexcept;
void shutdown() noexcept;
void request() noexcept;
void run_pending() noexcept;
void fail(Upscaler option) noexcept;
void clear() noexcept;
[[nodiscard]] std::string_view path(Upscaler option) noexcept;
void count_sdk_call() noexcept;
[[nodiscard]] uint32_t sdk_calls() noexcept;

namespace testing {
void set_identity(std::optional<Identity> identity) noexcept;
void set_runtime_directory(std::optional<std::filesystem::path> directory) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::probe

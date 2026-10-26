#pragma once

#include <cstdint>
#include <filesystem>

namespace aurora::render::detail::native::d3d12::sentinel {
enum Phase : uint8_t {
  Probe = 1u << 0,
  Session = 1u << 1,
};

void open(const std::filesystem::path& directory) noexcept;
uint8_t stale_at_startup() noexcept;
void arm(Phase phase) noexcept;
void disarm(Phase phase) noexcept;
void clear() noexcept;
} // namespace aurora::render::detail::native::d3d12::sentinel

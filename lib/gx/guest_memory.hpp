#pragma once

#include "../internal.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aurora::gx::guest_memory {
inline constexpr Module Log{"aurora::gx::guest_memory"};

enum class Kind : uint8_t {
  VertexArray = 0,
  IndexedLoad = 1,
  Texture = 2,
  Tlut = 3,
};

struct Hooks {
  // Replay serves recorded bytes because recorded guest addresses are not mapped in the replaying process.
  std::span<const uint8_t> (*resolve)(void* userData, Kind kind, uint64_t address, size_t size) noexcept = nullptr;
  // Recording observes bytes at FIFO time because the game may rewrite the memory after issuing the GX call.
  void (*observe)(void* userData, Kind kind, uint64_t address, std::span<const uint8_t> bytes) noexcept = nullptr;
  void* userData = nullptr;
};

inline std::atomic<const Hooks*> g_hooks{nullptr};

// Swapped only while the FIFO is drained so one frame never mixes hooked and unhooked reads.
inline void set_hooks(const Hooks* hooks) noexcept { g_hooks.store(hooks, std::memory_order_release); }

inline std::span<const uint8_t> read_hooked(const Hooks& hooks, Kind kind, const void* ptr, size_t size) noexcept {
  const auto address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  std::span<const uint8_t> bytes{static_cast<const uint8_t*>(ptr), size};
  if (hooks.resolve != nullptr) {
    bytes = hooks.resolve(hooks.userData, kind, address, size);
    AURORA_ASSERT(bytes.size() == size, "resolver returned {} bytes for a {}-byte read (kind {}) at 0x{:X}",
                  bytes.size(), size, static_cast<unsigned>(kind), address);
  }
  if (hooks.observe != nullptr && size != 0) {
    hooks.observe(hooks.userData, kind, address, bytes);
  }
  return bytes;
}

[[nodiscard]] inline std::span<const uint8_t> read(Kind kind, const void* ptr, size_t size) noexcept {
  const Hooks* hooks = g_hooks.load(std::memory_order_acquire);
  if (hooks == nullptr)
    LIKELY { return {static_cast<const uint8_t*>(ptr), size}; }
  return read_hooked(*hooks, kind, ptr, size);
}
} // namespace aurora::gx::guest_memory

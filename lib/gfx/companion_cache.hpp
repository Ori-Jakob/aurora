#pragma once

#include "texture_replacement.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>

namespace aurora::gfx::companion_cache {
constexpr uint32_t ViewSlotCount = 4;
constexpr uint64_t EvictAfterFrames = 1800;

constexpr int32_t view_slot(texture_replacement::CompanionRole role) noexcept {
  switch (role) {
  case texture_replacement::CompanionRole::Albedo:
  // Delight loads only without an albedo map, so the two share a slot
  case texture_replacement::CompanionRole::Delight:
    return 0;
  case texture_replacement::CompanionRole::Nrm:
  case texture_replacement::CompanionRole::Mat:
    return 1;
  case texture_replacement::CompanionRole::Orm:
    return 2;
  case texture_replacement::CompanionRole::Emit:
    return 3;
  default:
    return -1;
  }
}

constexpr uint8_t role_bit(texture_replacement::CompanionRole role) noexcept {
  return static_cast<uint8_t>(1u << static_cast<uint32_t>(role));
}

uint8_t views(uint32_t set, uint8_t wanted, std::array<WGPUTextureView, ViewSlotCount>& out) noexcept;
void process() noexcept;
uint64_t epoch() noexcept;
void shutdown() noexcept;

namespace testing {
struct Stats {
  uint64_t sets = 0;
  uint64_t loaded = 0;
  uint64_t loadedBytes = 0;
  uint64_t loadedEmit = 0;
  uint64_t loadedDelight = 0;
  uint64_t rejected = 0;
  uint64_t rejectedFormat = 0;
  uint64_t rejectedBc = 0;
  uint64_t evicted = 0;
  uint64_t syncLoads = 0;
  uint64_t repaired = 0;
  uint64_t repairedMat = 0;
  uint64_t repairedNrm = 0;
  uint64_t repairedOrm = 0;
  uint64_t repairedColour = 0;
  uint64_t truncated = 0;
};

void set_deviceless(bool deviceless) noexcept;
bool wait_idle(uint32_t timeoutMs) noexcept;
Stats stats() noexcept;
size_t resident_sets() noexcept;
} // namespace testing
} // namespace aurora::gfx::companion_cache

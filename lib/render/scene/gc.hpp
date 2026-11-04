#pragma once

#include "identity.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace aurora::render::scene::retained::gc {
inline constexpr uint32_t MaxSweeps = 16;
inline constexpr uint64_t SweepDelay = 120;
inline constexpr uint64_t DynamicFrames = 2;
inline constexpr uint32_t MaxReleases = 1024;
inline constexpr uint32_t RecentKeys = 4096;
inline constexpr uint64_t RecentFrames = 120;
inline constexpr uint64_t QuarantineSeals = 2;

// Identity keeps one per slot; the collector reads it only while the FIFO is idle
struct Life {
  uint64_t lastSeen = 0;
  uint64_t posSeq = 0;
  uintptr_t posBegin = 0;
  bool statics = false;
  bool released = false;
};

struct Seal {
  uint64_t sceneFrame = 0;
  bool scene = false;
  bool epochKnown = false;
  uint32_t epoch = 0;
};

struct Eviction {
  uint32_t slot = 0;
  EvictReason reason = EvictReason::None;
};

struct SweepResult {
  uint32_t releases = 0;
  uint32_t releaseOverflow = 0;
  bool epochChanged = false;
};

struct Recreation {
  EvictReason reason = EvictReason::None;
  uint64_t age = 0;
};

// Game thread while the FIFO is idle
void allocate();
void release() noexcept;
void reset() noexcept;
SweepResult sweep(const Seal& seal, std::span<const uint32_t> live, std::span<Life> lives,
                  std::vector<Eviction>& out) noexcept;
void quarantine(uint32_t slot) noexcept;
void free_quarantined(std::vector<uint32_t>& freeList) noexcept;
void remember(const Key& key, EvictReason reason, uint64_t sceneFrame) noexcept;
[[nodiscard]] uint32_t pending_sweeps() noexcept;

// FIFO thread
void on_release(uintptr_t begin, uint32_t size) noexcept;
[[nodiscard]] uint64_t release_seq() noexcept;
// Consumes the key's eviction record, so one return counts once
[[nodiscard]] Recreation recreated(const Key& key, uint64_t sceneFrame) noexcept;
} // namespace aurora::render::scene::retained::gc

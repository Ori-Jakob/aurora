#pragma once

#include "../internal.hpp"

#include <array>
#include <cstdint>

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

struct DrawRouteStats {
  uint64_t routed = 0;
  uint64_t suppressed = 0;
  // Merges actually prevented; classify.json's mergeBreaks counts every requested break
  uint64_t mergeBreaks = 0;
  uint64_t substituted = 0;
};

enum CpuZone : uint8_t {
  CpuZoneProcess,
  CpuZonePushDraw,
  CpuZoneArrays,
  CpuZonePipelineBuild,
  CpuZoneBindGroupBuild,
  CpuZoneUniformBuild,
  CpuZoneFogLut,
  CpuZoneCount,
};
enum CpuZoneTiming : uint32_t { CpuZoneTimingFifo = 1u << 0, CpuZoneTimingDraws = 1u << 1 };
struct CpuZoneStats {
  std::array<uint64_t, CpuZoneCount> calls{};
  std::array<uint64_t, CpuZoneCount> ns{};
  uint64_t arrayBytes = 0;
  uint64_t fogLutBytes = 0;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// Recording thread only: the next draw starts a new command instead of merging into the last one
void break_draw_merge() noexcept;
// Recording thread only: the next draw rebuilds its pipeline, bindings and uniform and cannot merge
void invalidate_draw_cache() noexcept;
DrawRouteStats draw_route_stats() noexcept;
// Totals since startup; readers take deltas, so nothing resets them
CpuZoneStats cpu_zone_stats() noexcept;
// Fifo times only process(), so the yardstick reads the clock a few times per frame instead of per draw
void set_cpu_zone_timing(uint32_t timing) noexcept;

namespace testing {
void reset_draw_route_stats() noexcept;
} // namespace testing

} // namespace aurora::gx::fifo

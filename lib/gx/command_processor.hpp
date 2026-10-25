#pragma once

#include "../internal.hpp"

#include <array>
#include <cstdint>
#include <vector>

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

struct ArrayUploadStats {
  uint64_t reused = 0;
  uint64_t reusedBytes = 0;
  // Rebinds whose bytes no longer matched the upload they would have reused
  uint64_t changed = 0;
};

enum class ImmutableArrayMode : uint8_t { Off, On, Validate };

struct ImmutableArrayStats {
  uint64_t entries = 0;
  uint64_t hits = 0;
  uint64_t stages = 0;
  uint64_t stageBytes = 0;
  uint64_t resets = 0;
  uint64_t epochs = 0;
  uint64_t staleReleases = 0;
  uint64_t overlapEvictions = 0;
  uint64_t sizeMismatches = 0;
  uint64_t arenaFull = 0;
  uint64_t violations = 0;
  uint64_t used = 0;
  uint64_t live = 0;
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
ArrayUploadStats array_upload_stats() noexcept;
// Totals since startup; readers take deltas, so nothing resets them
CpuZoneStats cpu_zone_stats() noexcept;
// Fifo times only process(), so the yardstick reads the clock a few times per frame instead of per draw
void set_cpu_zone_timing(uint32_t timing) noexcept;
void set_array_immutable(uint64_t address, uint32_t size, bool le, uint32_t generation) noexcept;
void release_array_immutable(uint64_t address, uint32_t generation) noexcept;
// Only while the FIFO is drained
void reset_immutable_arrays() noexcept;
void init_immutable_arrays() noexcept;
ImmutableArrayStats immutable_array_stats() noexcept;
void load_array_gpu(uint8_t attr, uint64_t handle, uint32_t byteOffset) noexcept;
// Only while the FIFO is drained
void end_gpu_frame() noexcept;

namespace testing {
struct FogRangeLutProbe {
  float rangeCenter;
  float renderWidth;
  std::vector<float> factors;
};
FogRangeLutProbe fog_range_lut(uint32_t targetWidth) noexcept;
void reset_draw_route_stats() noexcept;
void reset_array_upload_stats() noexcept;
void set_immutable_array_mode(ImmutableArrayMode mode) noexcept;
void reset_immutable_array_stats() noexcept;
} // namespace testing

} // namespace aurora::gx::fifo

#pragma once

#include "../internal.hpp"

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

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// Recording thread only: the next draw starts a new command instead of merging into the last one
void break_draw_merge() noexcept;
// Recording thread only: the next draw rebuilds its pipeline, bindings and uniform and cannot merge
void invalidate_draw_cache() noexcept;
DrawRouteStats draw_route_stats() noexcept;

namespace testing {
void reset_draw_route_stats() noexcept;
} // namespace testing

} // namespace aurora::gx::fifo

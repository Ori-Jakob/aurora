#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug::dual_source {
inline constexpr std::string_view Schema = "aurora.render.dual_source";

struct Counts {
  uint64_t dstAlphaDraws = 0;
  uint64_t prepassDraws = 0;
  uint64_t dualSourcePipelines = 0;
  uint64_t prepassPipelines = 0;
};

struct LayoutCounts {
  uint32_t colorAttachments = 0;
  uint64_t frames = 0;
  uint64_t dstAlphaDraws = 0;
  uint64_t prepassDraws = 0;
};

struct Report {
  bool dualSourceBlending = false;
  std::string backend;
  uint64_t frames = 0;
  uint64_t dualSourcePipelines = 0;
  uint64_t prepassPipelines = 0;
  std::vector<LayoutCounts> layouts;
};

void accumulate(Report& report, const Counts& delta, uint32_t colorAttachments);
[[nodiscard]] uint64_t lost_dual_source_draws(const Report& report) noexcept;
[[nodiscard]] std::string format_json(const Report& report);

void initialize();
void shutdown() noexcept;
void on_latch();
[[nodiscard]] bool active() noexcept;
} // namespace aurora::render::debug::dual_source

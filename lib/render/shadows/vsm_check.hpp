#pragma once

#include "vsm_pages.hpp"

#include "../graph.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace aurora::render::shadows::vsm::check {
inline constexpr uint64_t DefaultFirst = 30;
inline constexpr uint64_t DefaultLast = 300;
inline constexpr uint64_t CheckEvery = 60;

struct Config {
  bool on = false;
  uint64_t first = DefaultFirst;
  uint64_t last = DefaultLast;
  std::optional<uint64_t> full;
};

struct Result {
  uint64_t readbacks = 0;
  uint64_t frames = 0;
  uint64_t failures = 0;
  uint64_t entries = 0;
  uint64_t fulls = 0;
  uint64_t skipped = 0;
  uint64_t referenced = 0;
  uint64_t changed = 0;
  uint64_t footprint = 0;
  uint64_t statsSkipped = 0;
  bool busy = false;
};

struct ScanReadback {
  std::span<const uint32_t> prevHistory;
  std::span<const uint32_t> records;
  std::span<const uint32_t> materials;
  std::span<const uint32_t> history;
  std::span<const uint32_t> castBits;
  std::span<const uint32_t> dirtyPages;
  std::span<const uint32_t> table;
  std::span<const uint32_t> stats;
};

struct ScanCounts {
  bool referenced = false;
  uint64_t changed = 0;
  uint64_t footprint = 0;
};

[[nodiscard]] std::optional<Config> parse(std::string_view text) noexcept;
[[nodiscard]] const Config& config() noexcept;
[[nodiscard]] bool check_frame(uint64_t hintFrame) noexcept;
[[nodiscard]] bool force_full(uint64_t hintFrame) noexcept;
void record_views(const Views& views, uint64_t hintFrame) noexcept;
[[nodiscard]] std::string verify(const Views& expected, std::span<const uint32_t> views,
                                 std::span<const uint32_t> table, std::span<const uint32_t> phys);
// prevHistory empty skips the dirty reference and checks only the cast bits against the history
[[nodiscard]] std::string verify_scan(const Views& expected, const ScanReadback& in, ScanCounts& counts);
[[nodiscard]] detail::PassDesc pass_desc();
void shutdown() noexcept;

namespace testing {
void reset() noexcept;
void set_config(const std::optional<Config>& config) noexcept;
[[nodiscard]] Result result() noexcept;
} // namespace testing
} // namespace aurora::render::shadows::vsm::check

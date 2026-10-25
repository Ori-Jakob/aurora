#pragma once

#include "../fakeshadow/texture_gate.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::material {
struct Snapshot;
} // namespace aurora::render::material

namespace aurora::render::debug::fake_shadows {
inline constexpr std::string_view TabName = "Fake shadows";

struct Row {
  uint16_t materialId = 0;
  std::string key;
  fakeshadow::Neutral neutral = fakeshadow::Neutral::None;
  uint8_t gateBit = 0;
  uint16_t scopeMask = 0;
  bool scoped = false;
  bool registered = false;
  bool unseen = false;
  fakeshadow::OutcomeCounts counts{};
};

struct Rows {
  std::vector<Row> rows;
  fakeshadow::OutcomeCounts totals{};
  uint32_t unscoped = 0;
  uint32_t unseen = 0;
};

void initialize();
void shutdown() noexcept;
void on_latch() noexcept;
void draw_tab();
void queue_force(fakeshadow::Gate gate, fakeshadow::Force force) noexcept;
[[nodiscard]] fakeshadow::Force shown_force(fakeshadow::Gate gate) noexcept;

[[nodiscard]] Rows build_rows(const fakeshadow::TextureReport* report, const material::Snapshot* snapshot);
[[nodiscard]] std::string scope_text(uint16_t scopeMask, bool scoped);
[[nodiscard]] std::string report_json(const fakeshadow::TextureReport* report, const Rows& rows);

namespace testing {
void reset() noexcept;
void force_visible(int value) noexcept;
} // namespace testing
} // namespace aurora::render::debug::fake_shadows

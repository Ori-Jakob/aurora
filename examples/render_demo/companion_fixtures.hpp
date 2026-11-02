#pragma once

#include <dolphin/types.h>

#include <cstdint>
#include <span>
#include <string>

namespace render_demo {
struct CompanionCounts {
  uint64_t loaded = 0;
  uint64_t rejected = 0;
  uint64_t companionSkips = 0;
};

void prepare_companion_loads() noexcept;
bool load_companion_fixtures(bool mat, std::span<const u8> groundBase, std::string& error);
void release_companion_fixtures() noexcept;
CompanionCounts companion_counts() noexcept;
} // namespace render_demo

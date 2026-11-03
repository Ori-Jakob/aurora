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
  uint64_t emit = 0;
  uint64_t delight = 0;
};

enum class CompanionFixture { Nrm, Mat, Emit };

void prepare_companion_loads() noexcept;
bool load_companion_fixtures(CompanionFixture fixture, std::span<const u8> groundBase, std::string& error);
// Later calls rewrite and reload the same file, so the registration and its map directory stay put
bool set_companion_manifest(bool maps, std::span<const u8> groundBase, std::string& error);
void release_companion_fixtures() noexcept;
CompanionCounts companion_counts() noexcept;
} // namespace render_demo

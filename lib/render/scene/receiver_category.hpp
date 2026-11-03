#pragma once

#include "classifier.hpp"

#include <dolphin/gx/GXAurora.h>

#include <cstdint>

// One table for the record stamp and the light upload, so a masked light and its receivers always agree
namespace aurora::render::scene::receiver_category {
inline constexpr uint32_t Covered = 0x53F;

struct LightByte {
  uint8_t byte = 0;
  bool widened = false;
};

[[nodiscard]] constexpr uint8_t compress(uint32_t c) noexcept {
  return static_cast<uint8_t>((c & 0x3Fu) | ((c >> 8) & 1u) << 6 | ((c >> 10) & 1u) << 7);
}

[[nodiscard]] constexpr LightByte light_byte(uint32_t mask) noexcept {
  if (mask == 0) {
    return {};
  }
  if ((mask & ~Covered) != 0) {
    return {.byte = 0, .widened = true};
  }
  return {.byte = compress(mask), .widened = false};
}

static_assert(compress(AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER) == 0x82);
static_assert(compress(AURORA_CAT_STATIC | AURORA_CAT_TERRAIN) == 0x09);
static_assert(Covered == (category::Static | category::Dynamic | category::Skinned | category::Terrain |
                          category::Foliage | category::Water | category::Decal | category::Player));
static_assert(category::Static == AURORA_CAT_STATIC && compress(category::Static) == 1u << 0);
static_assert(category::Dynamic == AURORA_CAT_DYNAMIC && compress(category::Dynamic) == 1u << 1);
static_assert(category::Skinned == AURORA_CAT_SKINNED && compress(category::Skinned) == 1u << 2);
static_assert(category::Terrain == AURORA_CAT_TERRAIN && compress(category::Terrain) == 1u << 3);
static_assert(category::Foliage == AURORA_CAT_FOLIAGE && compress(category::Foliage) == 1u << 4);
static_assert(category::Water == AURORA_CAT_WATER && compress(category::Water) == 1u << 5);
static_assert(category::Decal == AURORA_CAT_DECAL && compress(category::Decal) == 1u << 6);
static_assert(category::Player == AURORA_CAT_PLAYER && compress(category::Player) == 1u << 7);
} // namespace aurora::render::scene::receiver_category

#pragma once

#include "../../gx/gx.hpp"

#include <cstdint>

namespace aurora::render::scene::map_pick {
inline constexpr uint8_t NoStage = 0xFF;
inline constexpr uint8_t NoTex = 0xFF;

struct MapPick {
  GXTexMapID map = GX_TEXMAP_NULL;
  GXTexCoordID coord = GX_TEXCOORD_NULL;
  uint8_t stage = NoStage;
  bool texcoordSource = false;
  uint8_t srcTex = NoTex;
  uint8_t channel = GX_CH_ALPHA;

  [[nodiscard]] bool valid() const noexcept { return map != GX_TEXMAP_NULL; }
  bool operator==(const MapPick&) const = default;
};

struct MapSelection {
  MapPick albedo;
  MapPick alpha;
  bool alphaTest = false;
};

[[nodiscard]] MapSelection select_maps(const gx::GXState& state) noexcept;
} // namespace aurora::render::scene::map_pick

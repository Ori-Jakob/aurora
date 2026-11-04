#include "map_pick.hpp"

#include <algorithm>
#include <array>

namespace aurora::render::scene::map_pick {
namespace {
bool reads_texture_colour(const gx::TevStage& stage) noexcept {
  const auto& pass = stage.colorPass;
  return std::ranges::any_of(std::array{pass.a, pass.b, pass.c, pass.d},
                             [](GXTevColorArg arg) { return arg == GX_CC_TEXC || arg == GX_CC_TEXA; });
}

bool reads_texture_alpha(const gx::TevStage& stage) noexcept {
  const auto& pass = stage.alphaPass;
  return std::ranges::any_of(std::array{pass.a, pass.b, pass.c, pass.d},
                             [](GXTevAlphaArg arg) { return arg == GX_CA_TEXA; });
}

bool sampled_map(const gx::GXState& state, const gx::TevStage& stage) noexcept {
  if (stage.texMapId == GX_TEXMAP_NULL || stage.texMapId >= gx::MaxTextures || stage.texCoordId == GX_TEXCOORD_NULL ||
      stage.texCoordId >= gx::MaxTexCoord) {
    return false;
  }
  const void* data = state.loadedTextures[stage.texMapId].data;
  return data == nullptr || !state.copyTextures.contains(data);
}

MapPick make_pick(const gx::GXState& state, const gx::TevStage& stage, uint8_t index) noexcept {
  MapPick pick{
      .map = stage.texMapId,
      .coord = stage.texCoordId,
      .stage = index,
  };
  const gx::TcgConfig& tcg = state.tcgs[stage.texCoordId];
  const bool matrix = tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4;
  if (matrix && tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7) {
    pick.texcoordSource = true;
    pick.srcTex = static_cast<uint8_t>(tcg.src - GX_TG_TEX0);
  }
  const size_t swap = std::min<size_t>(stage.tevSwapTex, state.tevSwapTable.size() - 1);
  pick.channel = static_cast<uint8_t>(state.tevSwapTable[swap].alpha);
  return pick;
}
} // namespace

MapSelection select_maps(const gx::GXState& state) noexcept {
  MapSelection out;
  out.alphaTest = static_cast<bool>(state.alphaCompare);
  const uint32_t stages = std::min<uint32_t>(state.numTevStages, gx::MaxTevStages);
  for (uint32_t i = 0; i < stages; ++i) {
    const gx::TevStage& stage = state.tevStages[i];
    if (!sampled_map(state, stage)) {
      continue;
    }
    const auto index = static_cast<uint8_t>(i);
    if (!out.albedo.valid() && reads_texture_colour(stage)) {
      out.albedo = make_pick(state, stage, index);
    }
    if (out.alphaTest && reads_texture_alpha(stage)) {
      out.alpha = make_pick(state, stage, index);
    }
  }
  return out;
}
} // namespace aurora::render::scene::map_pick

#pragma once

#include <dolphin/types.h>

#include <span>
#include <vector>

namespace render_demo {
struct SkinnedVertex {
  u8 matrixIndex;
  f32 position[3];
  f32 normal[3];
  u8 color[4];
};

u32 rgba8_tiled_size(u32 width, u32 height) noexcept;
void encode_rgba8_tiled(std::span<const u8> rgba, u32 width, u32 height, std::span<u8> out) noexcept;
std::vector<u8> make_ground_texture(u32 size, u32 mipCount);
std::vector<std::vector<SkinnedVertex>> make_tentacle(u8 firstMatrixId);
} // namespace render_demo

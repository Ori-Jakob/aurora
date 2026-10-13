#include "procedural.hpp"

#include <algorithm>
#include <cmath>

namespace render_demo {
namespace {
constexpr u32 TentacleRings = 9;
constexpr u32 TentacleSides = 10;
constexpr u32 TentacleBones = 4;
constexpr f32 RingSpacing = 0.75f;
constexpr f32 BoneLength = 1.5f;

u8 lerp_u8(u8 a, u8 b, f32 t) noexcept { return static_cast<u8>(std::lround(a + (b - a) * t)); }

std::vector<u8> downsample_rgba(const std::vector<u8>& src, u32 width, u32 height) {
  const u32 dstWidth = std::max(width / 2, 1u);
  const u32 dstHeight = std::max(height / 2, 1u);
  std::vector<u8> dst(dstWidth * dstHeight * 4);
  for (u32 y = 0; y < dstHeight; ++y) {
    const u32 y0 = std::min(y * 2, height - 1);
    const u32 y1 = std::min(y * 2 + 1, height - 1);
    for (u32 x = 0; x < dstWidth; ++x) {
      const u32 x0 = std::min(x * 2, width - 1);
      const u32 x1 = std::min(x * 2 + 1, width - 1);
      for (u32 c = 0; c < 4; ++c) {
        const u32 sum = src[(y0 * width + x0) * 4 + c] + src[(y0 * width + x1) * 4 + c] +
                        src[(y1 * width + x0) * 4 + c] + src[(y1 * width + x1) * 4 + c];
        dst[(y * dstWidth + x) * 4 + c] = static_cast<u8>((sum + 2) / 4);
      }
    }
  }
  return dst;
}

SkinnedVertex ring_vertex(u32 ring, u32 side, u8 firstMatrixId) noexcept {
  const u32 bone = std::min(ring / 2, TentacleBones - 1);
  const f32 theta = 2.f * 3.14159265358979f * static_cast<f32>(side) / static_cast<f32>(TentacleSides);
  const f32 radius = 0.7f - 0.06f * static_cast<f32>(ring);
  const f32 t = static_cast<f32>(ring) / static_cast<f32>(TentacleRings - 1);
  return SkinnedVertex{
      .matrixIndex = static_cast<u8>(firstMatrixId + 3 * bone),
      .position = {std::cos(theta) * radius, RingSpacing * static_cast<f32>(ring) - BoneLength * static_cast<f32>(bone),
                   std::sin(theta) * radius},
      .normal = {std::cos(theta), 0.f, std::sin(theta)},
      .color = {lerp_u8(170, 255, t), lerp_u8(60, 210, t), lerp_u8(150, 110, t), 255},
  };
}
} // namespace

u32 rgba8_tiled_size(u32 width, u32 height) noexcept { return ((width + 3) / 4) * ((height + 3) / 4) * 64; }

void encode_rgba8_tiled(std::span<const u8> rgba, u32 width, u32 height, std::span<u8> out) noexcept {
  const u32 blocksX = (width + 3) / 4;
  const u32 blocksY = (height + 3) / 4;
  std::fill_n(out.begin(), rgba8_tiled_size(width, height), u8{0});
  for (u32 by = 0; by < blocksY; ++by) {
    for (u32 bx = 0; bx < blocksX; ++bx) {
      u8* block = out.data() + (by * blocksX + bx) * 64;
      for (u32 ty = 0; ty < 4; ++ty) {
        for (u32 tx = 0; tx < 4; ++tx) {
          const u32 x = bx * 4 + tx;
          const u32 y = by * 4 + ty;
          if (x >= width || y >= height) {
            continue;
          }
          const u8* texel = rgba.data() + (y * width + x) * 4;
          const u32 i = ty * 4 + tx;
          block[i * 2] = texel[3];
          block[i * 2 + 1] = texel[0];
          block[32 + i * 2] = texel[1];
          block[32 + i * 2 + 1] = texel[2];
        }
      }
    }
  }
}

std::vector<u8> make_ground_texture(u32 size, u32 mipCount) {
  constexpr u8 CellA[3] = {96, 148, 72};
  constexpr u8 CellB[3] = {72, 120, 56};
  constexpr u8 Grout[3] = {44, 64, 40};
  std::vector<u8> level(size * size * 4);
  for (u32 y = 0; y < size; ++y) {
    for (u32 x = 0; x < size; ++x) {
      const u8* color = (x % 16 == 0 || y % 16 == 0) ? Grout : (((x / 16) + (y / 16)) & 1) ? CellB : CellA;
      u8* texel = level.data() + (y * size + x) * 4;
      texel[0] = color[0];
      texel[1] = color[1];
      texel[2] = color[2];
      texel[3] = 255;
    }
  }

  u32 total = 0;
  for (u32 i = 0, w = size, h = size; i < mipCount; ++i, w = std::max(w / 2, 1u), h = std::max(h / 2, 1u)) {
    total += rgba8_tiled_size(w, h);
  }
  std::vector<u8> out(total);
  u32 offset = 0;
  u32 width = size;
  u32 height = size;
  for (u32 i = 0; i < mipCount; ++i) {
    const u32 levelSize = rgba8_tiled_size(width, height);
    encode_rgba8_tiled(level, width, height, std::span{out}.subspan(offset, levelSize));
    offset += levelSize;
    if (i + 1 < mipCount) {
      level = downsample_rgba(level, width, height);
      width = std::max(width / 2, 1u);
      height = std::max(height / 2, 1u);
    }
  }
  return out;
}

std::vector<std::vector<SkinnedVertex>> make_tentacle(u8 firstMatrixId) {
  std::vector<std::vector<SkinnedVertex>> strips;
  strips.reserve(TentacleRings - 1);
  for (u32 ring = 0; ring + 1 < TentacleRings; ++ring) {
    auto& strip = strips.emplace_back();
    strip.reserve((TentacleSides + 1) * 2);
    for (u32 side = 0; side <= TentacleSides; ++side) {
      strip.push_back(ring_vertex(ring + 1, side, firstMatrixId));
      strip.push_back(ring_vertex(ring, side, firstMatrixId));
    }
  }
  return strips;
}
} // namespace render_demo

#pragma once

#include <array>
#include <cfloat>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::test::hzb {
struct CpuPyramid {
  std::vector<uint32_t> widths;
  std::vector<uint32_t> heights;
  std::vector<std::vector<float>> levels;

  [[nodiscard]] uint32_t level_count() const noexcept { return static_cast<uint32_t>(levels.size()); }
  [[nodiscard]] float at(uint32_t level, uint32_t x, uint32_t y) const {
    return levels[level][size_t{y} * widths[level] + x];
  }
};

struct Rect {
  uint32_t x0 = 0;
  uint32_t y0 = 0;
  uint32_t x1 = 0;
  uint32_t y1 = 0;
};

struct Bounds {
  std::array<float, 2> ndcMin{FLT_MAX, FLT_MAX};
  std::array<float, 2> ndcMax{-FLT_MAX, -FLT_MAX};
  float nearest = -FLT_MAX;
  bool valid = true;
};

struct Viewport {
  float x = 0.f;
  float y = 0.f;
  float width = 0.f;
  float height = 0.f;
  float minDepth = 0.f;
  float maxDepth = 1.f;
};

using Mat4 = std::array<float, 16>;
using Vec3 = std::array<float, 3>;

CpuPyramid build_pyramid(std::span<const float> depth, uint32_t width, uint32_t height);
bool occluded_rect(const CpuPyramid& pyramid, Rect rect, float nearest);
Bounds bounds_add(Bounds bounds, std::array<float, 4> clip);
Bounds bounds_aabb(const Mat4& clipFromLocal, Vec3 bmin, Vec3 bmax);
bool bounds_rect(const Bounds& bounds, const Viewport& viewport, uint32_t efbWidth, uint32_t efbHeight, Rect& rect,
                 float& nearest);
bool occluded_bounds(const CpuPyramid& pyramid, const Bounds& bounds, const Viewport& viewport, uint32_t efbWidth,
                     uint32_t efbHeight);
std::array<float, 4> transform(const Mat4& m, Vec3 point);
float brute_min(std::span<const float> depth, uint32_t width, Rect rect);

class RangeMin {
public:
  RangeMin(std::span<const float> depth, uint32_t width, uint32_t height);
  [[nodiscard]] float min(Rect rect) const;

private:
  uint32_t m_width = 0;
  std::vector<std::vector<float>> m_levels;
};
} // namespace aurora::test::hzb

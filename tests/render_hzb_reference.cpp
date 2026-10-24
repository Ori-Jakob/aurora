#include "render_hzb_reference.hpp"

#include "render/geometry/hzb.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace aurora::test::hzb {
namespace rules = render::detail::hzb;

namespace {
constexpr float RelMargin = 0x1p-18f;
constexpr float AbsMargin = 0x1p-30f;
constexpr float MinW = 1e-6f;

std::vector<float> reduce(std::span<const float> source, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth,
                          uint32_t outHeight) {
  std::vector<float> out(size_t{outWidth} * outHeight, 0.f);
  for (uint32_t y = 0; y < outHeight; ++y) {
    const rules::Footprint fy = rules::footprint(y, outHeight, inHeight);
    for (uint32_t x = 0; x < outWidth; ++x) {
      const rules::Footprint fx = rules::footprint(x, outWidth, inWidth);
      if (!rules::reducible(fx, fy)) {
        continue;
      }
      float farthest = source[size_t{fy.first} * inWidth + fx.first];
      for (uint32_t sy = fy.first; sy <= fy.last; ++sy) {
        for (uint32_t sx = fx.first; sx <= fx.last; ++sx) {
          farthest = std::min(farthest, source[size_t{sy} * inWidth + sx]);
        }
      }
      out[size_t{y} * outWidth + x] = farthest;
    }
  }
  return out;
}
} // namespace

CpuPyramid build_pyramid(std::span<const float> depth, uint32_t width, uint32_t height) {
  CpuPyramid out;
  const uint32_t mip0Width = (width + 1) / 2;
  const uint32_t mip0Height = (height + 1) / 2;
  const auto levels = static_cast<uint32_t>(std::bit_width(std::max(mip0Width, mip0Height)));
  std::span<const float> source = depth;
  uint32_t inWidth = width;
  uint32_t inHeight = height;
  for (uint32_t level = 0; level < levels; ++level) {
    const uint32_t outWidth = rules::level_size(mip0Width, level);
    const uint32_t outHeight = rules::level_size(mip0Height, level);
    out.levels.push_back(reduce(source, inWidth, inHeight, outWidth, outHeight));
    out.widths.push_back(outWidth);
    out.heights.push_back(outHeight);
    source = out.levels.back();
    inWidth = outWidth;
    inHeight = outHeight;
  }
  return out;
}

bool occluded_rect(const CpuPyramid& pyramid, Rect rect, float nearest) {
  if (rect.x0 > rect.x1 || rect.y0 > rect.y1 || pyramid.level_count() == 0) {
    return false;
  }
  const uint32_t extent = std::max(rect.x1 - rect.x0, rect.y1 - rect.y0) + 1;
  uint32_t level = rules::query_level(extent, pyramid.level_count());
  if (level > 0) {
    const uint32_t finerWidth = pyramid.widths[level - 1];
    const uint32_t finerHeight = pyramid.heights[level - 1];
    if (rules::texel_of(rect.x1, level - 1, finerWidth) - rules::texel_of(rect.x0, level - 1, finerWidth) <= 1 &&
        rules::texel_of(rect.y1, level - 1, finerHeight) - rules::texel_of(rect.y0, level - 1, finerHeight) <= 1) {
      --level;
    }
  }
  const uint32_t width = pyramid.widths[level];
  const uint32_t height = pyramid.heights[level];
  const uint32_t loX = rules::texel_of(rect.x0, level, width);
  const uint32_t loY = rules::texel_of(rect.y0, level, height);
  const uint32_t hiX = rules::texel_of(rect.x1, level, width);
  const uint32_t hiY = rules::texel_of(rect.y1, level, height);
  if (hiX - loX > 1 || hiY - loY > 1) {
    return false;
  }
  const float farthest = std::min(std::min(pyramid.at(level, loX, loY), pyramid.at(level, hiX, loY)),
                                  std::min(pyramid.at(level, loX, hiY), pyramid.at(level, hiX, hiY)));
  return nearest < farthest;
}

Bounds bounds_add(Bounds bounds, std::array<float, 4> clip) {
  const bool finite = std::abs(clip[0]) <= FLT_MAX && std::abs(clip[1]) <= FLT_MAX && std::abs(clip[2]) <= FLT_MAX;
  if (!(clip[3] > MinW && finite)) {
    bounds.valid = false;
    return bounds;
  }
  const float x = clip[0] / clip[3];
  const float y = clip[1] / clip[3];
  const float z = clip[2] / clip[3];
  bounds.ndcMin = {std::min(bounds.ndcMin[0], x), std::min(bounds.ndcMin[1], y)};
  bounds.ndcMax = {std::max(bounds.ndcMax[0], x), std::max(bounds.ndcMax[1], y)};
  bounds.nearest = std::max(bounds.nearest, z);
  return bounds;
}

std::array<float, 4> transform(const Mat4& m, Vec3 point) {
  std::array<float, 4> out{};
  for (size_t row = 0; row < 4; ++row) {
    out[row] = m[row] * point[0] + m[4 + row] * point[1] + m[8 + row] * point[2] + m[12 + row];
  }
  return out;
}

Bounds bounds_aabb(const Mat4& clipFromLocal, Vec3 bmin, Vec3 bmax) {
  Bounds bounds;
  for (uint32_t i = 0; i < 8; ++i) {
    const Vec3 corner{(i & 1) != 0 ? bmax[0] : bmin[0], (i & 2) != 0 ? bmax[1] : bmin[1],
                      (i & 4) != 0 ? bmax[2] : bmin[2]};
    bounds = bounds_add(bounds, transform(clipFromLocal, corner));
  }
  return bounds;
}

bool bounds_rect(const Bounds& bounds, const Viewport& viewport, uint32_t efbWidth, uint32_t efbHeight, Rect& rect,
                 float& nearest) {
  if (!bounds.valid || bounds.ndcMin[0] > bounds.ndcMax[0] || bounds.ndcMin[1] > bounds.ndcMax[1] || efbWidth == 0 ||
      efbHeight == 0) {
    return false;
  }
  const float x0 = viewport.x + (bounds.ndcMin[0] * 0.5f + 0.5f) * viewport.width;
  const float x1 = viewport.x + (bounds.ndcMax[0] * 0.5f + 0.5f) * viewport.width;
  const float y0 = viewport.y + (0.5f - bounds.ndcMax[1] * 0.5f) * viewport.height;
  const float y1 = viewport.y + (0.5f - bounds.ndcMin[1] * 0.5f) * viewport.height;
  const auto clamp_to = [](float value, uint32_t size) {
    return static_cast<uint32_t>(std::clamp(value, 0.f, static_cast<float>(size - 1)));
  };
  rect = {
      clamp_to(std::floor(x0) - 1.f, efbWidth),
      clamp_to(std::floor(y0) - 1.f, efbHeight),
      clamp_to(std::floor(x1) + 1.f, efbWidth),
      clamp_to(std::floor(y1) + 1.f, efbHeight),
  };
  const float window = viewport.minDepth + std::max(bounds.nearest, 0.f) * (viewport.maxDepth - viewport.minDepth);
  nearest = window * (1.f + RelMargin) + AbsMargin;
  return true;
}

bool occluded_bounds(const CpuPyramid& pyramid, const Bounds& bounds, const Viewport& viewport, uint32_t efbWidth,
                     uint32_t efbHeight) {
  Rect rect;
  float nearest = 0.f;
  return bounds_rect(bounds, viewport, efbWidth, efbHeight, rect, nearest) && occluded_rect(pyramid, rect, nearest);
}

float brute_min(std::span<const float> depth, uint32_t width, Rect rect) {
  float out = FLT_MAX;
  for (uint32_t y = rect.y0; y <= rect.y1; ++y) {
    for (uint32_t x = rect.x0; x <= rect.x1; ++x) {
      out = std::min(out, depth[size_t{y} * width + x]);
    }
  }
  return out;
}

RangeMin::RangeMin(std::span<const float> depth, uint32_t width, uint32_t height) : m_width(width) {
  m_levels.emplace_back(depth.begin(), depth.end());
  for (uint32_t span = 2; span <= width; span *= 2) {
    const std::vector<float>& previous = m_levels.back();
    std::vector<float> next(previous.size(), FLT_MAX);
    const uint32_t half = span / 2;
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x + span <= width; ++x) {
        const size_t row = size_t{y} * width;
        next[row + x] = std::min(previous[row + x], previous[row + x + half]);
      }
    }
    m_levels.push_back(std::move(next));
  }
}

float RangeMin::min(Rect rect) const {
  const uint32_t count = rect.x1 - rect.x0 + 1;
  const auto level = static_cast<uint32_t>(std::bit_width(count)) - 1;
  const uint32_t span = 1u << level;
  const std::vector<float>& table = m_levels[level];
  float out = FLT_MAX;
  for (uint32_t y = rect.y0; y <= rect.y1; ++y) {
    const size_t row = size_t{y} * m_width;
    out = std::min({out, table[row + rect.x0], table[row + rect.x1 + 1 - span]});
  }
  return out;
}
} // namespace aurora::test::hzb

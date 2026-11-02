#include "render_ao_reference.hpp"

#include "render/ao/ao.hpp"
#include "render/ao/hiz.hpp"

#include <algorithm>
#include <cmath>

namespace aurora::test::ao {
namespace rules = render::detail::ao;

double view_depth(const render::detail::StageCamera& camera, float depth) {
  const double sky = rules::SkyDepth;
  if (depth <= 0.f) {
    return sky;
  }
  return std::clamp(double{camera.depthMul} / (double{camera.depthAdd} - double{depth}), 0.0, sky);
}

double depth_mip_filter(const std::array<double, 4>& depths, double effectRadius) {
  const double maxDepth = std::max(std::max(depths[0], depths[1]), std::max(depths[2], depths[3]));
  const double falloff = rules::FalloffRange;
  const double radius = std::max(0.75 * effectRadius, 1e-6);
  const double falloffRange = falloff * radius;
  const double falloffFrom = radius * (1.0 - falloff);
  const double falloffMul = -1.0 / falloffRange;
  const double falloffAdd = falloffFrom / falloffRange + 1.0;
  double weighted = 0.0;
  double total = 0.0;
  for (const double depth : depths) {
    const double weight = std::clamp((maxDepth - depth) * falloffMul + falloffAdd, 0.0, 1.0);
    weighted += weight * depth;
    total += weight;
  }
  return weighted / total;
}

std::array<double, 3> view_position_fwd(const render::detail::StageCamera& camera, double u, double v, double z,
                                        double width, double height) {
  const double cx = double{camera.p02} - 2.0 * camera.jitterPx[0] / width;
  const double cy = double{camera.p12} + 2.0 * camera.jitterPx[1] / height;
  return {(2.0 * u - 1.0 + cx) * z / camera.p00, (1.0 - 2.0 * v + cy) * z / camera.p11, z};
}

CpuPyramid build_pyramid(std::span<const float> depth, uint32_t width, uint32_t height, uint32_t divisor,
                         const render::detail::StageCamera& camera, double effectRadius, uint32_t levels) {
  CpuPyramid out;
  const uint32_t gridWidth = rules::grid_size(width, divisor);
  const uint32_t gridHeight = rules::grid_size(height, divisor);
  std::vector<double> mip0(size_t{gridWidth} * gridHeight);
  for (uint32_t y = 0; y < gridHeight; ++y) {
    const uint32_t qy = rules::rep_texel(y, divisor, height);
    for (uint32_t x = 0; x < gridWidth; ++x) {
      const uint32_t qx = rules::rep_texel(x, divisor, width);
      mip0[size_t{y} * gridWidth + x] = view_depth(camera, depth[size_t{qy} * width + qx]);
    }
  }
  out.widths.push_back(gridWidth);
  out.heights.push_back(gridHeight);
  out.levels.push_back(std::move(mip0));
  for (uint32_t level = 1; level < levels; ++level) {
    const std::vector<double>& prev = out.levels.back();
    const uint32_t prevWidth = out.widths.back();
    const uint32_t prevHeight = out.heights.back();
    const uint32_t levelWidth = rules::hiz::level_size(gridWidth, level);
    const uint32_t levelHeight = rules::hiz::level_size(gridHeight, level);
    std::vector<double> next(size_t{levelWidth} * levelHeight);
    const auto at = [&](uint32_t x, uint32_t y) {
      return prev[size_t{std::min(y, prevHeight - 1)} * prevWidth + std::min(x, prevWidth - 1)];
    };
    for (uint32_t y = 0; y < levelHeight; ++y) {
      for (uint32_t x = 0; x < levelWidth; ++x) {
        const std::array taps{at(2 * x, 2 * y), at(2 * x + 1, 2 * y), at(2 * x, 2 * y + 1), at(2 * x + 1, 2 * y + 1)};
        next[size_t{y} * levelWidth + x] = depth_mip_filter(taps, effectRadius);
      }
    }
    out.widths.push_back(levelWidth);
    out.heights.push_back(levelHeight);
    out.levels.push_back(std::move(next));
  }
  return out;
}
} // namespace aurora::test::ao

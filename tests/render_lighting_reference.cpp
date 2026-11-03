#include "render_lighting_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aurora::test::lighting {
namespace clusters = render::detail::lighting::clusters;

Vec3d view_position_fwd(const StageCamera& camera, double u, double v, double z, double width, double height) noexcept {
  const double cx = double{camera.p02} - 2.0 * double{camera.jitterPx[0]} / width;
  const double cy = double{camera.p12} + 2.0 * double{camera.jitterPx[1]} / height;
  return {(2.0 * u - 1.0 + cx) * z / double{camera.p00}, (1.0 - 2.0 * v + cy) * z / double{camera.p11}, z};
}

float view_depth(const StageCamera& camera, float stored) noexcept {
  constexpr float Far = 1e30f;
  if (stored <= 0.f) {
    return Far;
  }
  return std::clamp(camera.depthMul / (camera.depthAdd - stored), 0.f, Far);
}

std::array<float, 2> cluster_depths(const StageCamera& camera, const LightListHeader& header, uint32_t slice) noexcept {
  constexpr float Far = 1e30f;
  const float near = std::clamp(camera.depthMul / (camera.depthAdd - 1.f), 0.f, Far);
  const float far = camera.depthAdd != 0.f ? std::clamp(camera.depthMul / camera.depthAdd, 0.f, Far) : Far;
  const float lo = slice == 0 ? near : header.sliceNear * clusters::SliceScale[slice];
  const float hi = slice == clusters::Slices - 1 ? far : header.sliceNear * clusters::SliceScale[slice + 1];
  return {lo, hi};
}

Box cluster_box(const StageCamera& camera, const LightListHeader& header, uint32_t cluster, uint32_t width,
                uint32_t height, double depthInflate) noexcept {
  const uint32_t tx = cluster % clusters::TilesX;
  const uint32_t ty = (cluster / clusters::TilesX) % clusters::TilesY;
  const uint32_t slice = cluster / (clusters::TilesX * clusters::TilesY);
  const std::array<float, 2> depths = cluster_depths(camera, header, slice);
  const std::array<float, 2> u{static_cast<float>(tx) * (1.f / 16.f), static_cast<float>(tx + 1) * (1.f / 16.f)};
  const std::array<float, 2> v{static_cast<float>(ty) * (1.f / 9.f), static_cast<float>(ty + 1) * (1.f / 9.f)};
  Box box{.lo = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                 std::numeric_limits<double>::max()},
          .hi = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
                 std::numeric_limits<double>::lowest()}};
  const double w = std::max(width, 1u);
  const double h = std::max(height, 1u);
  for (uint32_t i = 0; i < 8; ++i) {
    const Vec3d corner = view_position_fwd(camera, u[i & 1], v[(i >> 1) & 1], depths[(i >> 2) & 1], w, h);
    for (size_t a = 0; a < 3; ++a) {
      box.lo[a] = std::min(box.lo[a], corner[a]);
      box.hi[a] = std::max(box.hi[a], corner[a]);
    }
  }
  const double inflate = depthInflate * std::max(std::abs(double{depths[0]}), std::abs(double{depths[1]}));
  for (size_t a = 0; a < 3; ++a) {
    box.lo[a] -= inflate;
    box.hi[a] += inflate;
  }
  return box;
}

std::vector<Mask> assign_reference(const LightListHeader& header, std::span<const GpuLight> lights,
                                   const StageCamera& camera, uint32_t width, uint32_t height, double radiusScale,
                                   double depthInflate) {
  std::vector<Mask> masks(clusters::ClusterCount);
  const uint32_t dir = std::min(header.dirCount, clusters::MaxDirectional);
  const uint32_t local = std::min(header.localCount, clusters::MaxLights - dir);
  for (uint32_t cluster = 0; cluster < clusters::ClusterCount; ++cluster) {
    const Box box = cluster_box(camera, header, cluster, width, height, depthInflate);
    for (uint32_t i = 0; i < local && dir + i < lights.size(); ++i) {
      const GpuLight& light = lights[dir + i];
      double d2 = 0.0;
      for (size_t a = 0; a < 3; ++a) {
        const double c = light.v0[a];
        const double d = std::max({box.lo[a] - c, c - box.hi[a], 0.0});
        d2 += d * d;
      }
      const double r = double{light.v0[3]} * radiusScale;
      if (d2 <= r * r) {
        masks[cluster][i / 32] |= 1u << (i % 32);
      }
    }
  }
  return masks;
}

std::array<uint8_t, 3> heatmap_rgb(uint32_t count, bool border, bool sky) noexcept {
  std::array<uint8_t, 3> code{230, 20, 20};
  if (count == 0) {
    code = {40, 40, 40};
  } else if (count <= 2) {
    code = {30, 90, 255};
  } else if (count <= 4) {
    code = {40, 200, 60};
  } else if (count <= 8) {
    code = {255, 230, 40};
  } else if (count <= 16) {
    code = {255, 140, 20};
  } else if (count < 32) {
    code = {255, 80, 20};
  }
  const double scale = (border ? 0.5 : 1.0) * (sky ? 0.35 : 1.0);
  std::array<uint8_t, 3> out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = static_cast<uint8_t>(std::lround(code[i] * scale));
  }
  return out;
}
} // namespace aurora::test::lighting

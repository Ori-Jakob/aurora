#include "render_ao_reference.hpp"

#include "render/ao/ao.hpp"
#include "render/ao/hiz.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

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

namespace {
using Vec3 = std::array<double, 3>;
constexpr double Pi = std::numbers::pi;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double length(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 normalize(const Vec3& a) {
  const double len = length(a);
  return len > 0.0 ? scale(a, 1.0 / len) : a;
}
Vec3 safe_normalize(const Vec3& a, const Vec3& fallback) {
  const double len = length(a);
  return len > 1e-6 ? scale(a, 1.0 / len) : fallback;
}
double saturate(double v) { return std::clamp(v, 0.0, 1.0); }
double acos_clamped(double v) { return std::acos(std::clamp(v, -1.0, 1.0)); }
double sign(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
double round_even(double v) { return std::nearbyint(v); }

Vec3 decode_normal(uint32_t packed) {
  const Vec3 unorm{static_cast<double>(packed & 1023u) / 1023.0, static_cast<double>((packed >> 10) & 1023u) / 1023.0,
                   static_cast<double>((packed >> 20) & 1023u) / 1023.0};
  return add(scale(unorm, 2.0), {-1.0, -1.0, -1.0});
}

double normal_angle(double ux, double uy, double width, double height, const render::detail::StageCamera& camera,
                    const Vec3& view, const Vec3& normal) {
  const Vec3 direction = normalize({ux / (width * camera.p00), -uy / (height * camera.p11), 0.0});
  const Vec3 ortho = sub(direction, scale(view, dot(direction, view)));
  const Vec3 axis = normalize(cross(ortho, view));
  const Vec3 projected = sub(normal, scale(axis, dot(normal, axis)));
  const double cosN = saturate(dot(projected, view) / std::max(length(projected), 1e-6));
  return sign(dot(ortho, projected)) * std::acos(cosN);
}
} // namespace

uint32_t pack_normal(const std::array<double, 3>& gxNormal, uint32_t code) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    const auto unorm = static_cast<uint32_t>(std::lround(std::clamp(gxNormal[i] * 0.5 + 0.5, 0.0, 1.0) * 1023.0));
    out |= unorm << (10 * i);
  }
  return out | (std::min(code, 3u) << 30);
}

uint32_t normal_code(uint32_t packed) { return packed >> 30; }

std::array<double, 4> calculate_edges(double z, double left, double right, double top, double bottom) {
  std::array<double, 4> edges{left - z, right - z, top - z, bottom - z};
  const double slopeLr = (edges[1] - edges[0]) * 0.5;
  const double slopeTb = (edges[3] - edges[2]) * 0.5;
  const std::array<double, 4> adjusted{edges[0] + slopeLr, edges[1] - slopeLr, edges[2] + slopeTb, edges[3] - slopeTb};
  for (size_t i = 0; i < edges.size(); ++i) {
    edges[i] = saturate(1.25 - std::min(std::abs(edges[i]), std::abs(adjusted[i])) / (z * 0.011));
  }
  return edges;
}

uint8_t pack_edges(const std::array<double, 4>& edges) {
  uint32_t out = 0;
  for (size_t i = 0; i < edges.size(); ++i) {
    out |= static_cast<uint32_t>(round_even(saturate(edges[i]) * 2.9)) << (6 - 2 * i);
  }
  return static_cast<uint8_t>(out);
}

std::array<double, 4> unpack_edges(uint8_t packed) {
  std::array<double, 4> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<double>((packed >> (6 - 2 * i)) & 3u) / 3.0;
  }
  return out;
}

double interval_union(std::vector<std::array<double, 2>> intervals) {
  for (auto& [lo, hi] : intervals) {
    lo = saturate(lo);
    hi = saturate(hi);
  }
  std::erase_if(intervals, [](const std::array<double, 2>& interval) { return interval[1] <= interval[0]; });
  std::ranges::sort(intervals);
  double total = 0.0;
  double start = 0.0;
  double end = 0.0;
  for (const auto& [lo, hi] : intervals) {
    if (lo > end) {
      total += end - start;
      start = lo;
      end = hi;
    } else {
      end = std::max(end, hi);
    }
  }
  return total + (end - start);
}

std::vector<GtaoTexel> estimate_gtao(const GtaoSetup& setup, const CpuPyramid& pyramid) {
  const render::detail::ao::Params& params = setup.params;
  const render::detail::StageCamera& camera = setup.camera;
  const uint32_t width = setup.width;
  const uint32_t height = setup.height;
  const uint32_t d = setup.divisor;
  const uint32_t gridWidth = rules::grid_size(width, d);
  const uint32_t gridHeight = rules::grid_size(height, d);
  const std::vector<double>& hiz = pyramid.levels[0];
  const auto depth_at = [&](int64_t x, int64_t y) {
    x = std::clamp<int64_t>(x, 0, gridWidth - 1);
    y = std::clamp<int64_t>(y, 0, gridHeight - 1);
    return hiz[static_cast<size_t>(y) * gridWidth + static_cast<size_t>(x)];
  };
  const auto position = [&](double px, double py, double z) {
    return view_position_fwd(camera, px / width, py / height, z, width, height);
  };
  const auto sky = [](double z) { return z >= 0.5 * rules::SkyDepth; };
  const bool ceilMode = (params.flags & rules::FlagSectorCeil) != 0;
  const uint32_t maxMip = std::min(pyramid.level_count(), rules::MipLevels) - 1;
  std::vector<GtaoTexel> out(size_t{gridWidth} * gridHeight);
  for (uint32_t py = 0; py < gridHeight; ++py) {
    for (uint32_t px = 0; px < gridWidth; ++px) {
      GtaoTexel& texel = out[size_t{py} * gridWidth + px];
      const double z = depth_at(px, py);
      const std::array<double, 4> around{depth_at(int64_t{px} - 1, py), depth_at(int64_t{px} + 1, py),
                                         depth_at(px, int64_t{py} - 1), depth_at(px, int64_t{py} + 1)};
      const std::array<double, 4> edges = calculate_edges(z, around[0], around[1], around[2], around[3]);
      texel.edges = pack_edges(edges);
      const uint32_t qx = rules::rep_texel(px, d, width);
      const uint32_t qy = rules::rep_texel(py, d, height);
      const uint32_t packedNormal = setup.normals[size_t{qy} * width + qx];
      if (sky(z) || normal_code(packedNormal) == 0) {
        continue;
      }
      texel.receiver = true;
      if (params.sliceCount == 0) {
        continue;
      }
      const double cx = qx + 0.5;
      const double cy = qy + 0.5;
      std::array<double, 4> nearAround = around;
      for (double& value : nearAround) {
        value = sky(value) ? z : value;
      }
      const Vec3 centre = position(cx, cy, z);
      const auto toward = [&](double dx, double dy, double value) {
        return normalize(sub(position(cx + dx, cy + dy, value), centre));
      };
      const double step = d;
      const Vec3 dl = toward(-step, 0.0, nearAround[0]);
      const Vec3 dr = toward(step, 0.0, nearAround[1]);
      const Vec3 dt = toward(0.0, -step, nearAround[2]);
      const Vec3 db = toward(0.0, step, nearAround[3]);
      const std::array<double, 4> accepted{saturate(edges[0] * edges[2] + 0.01), saturate(edges[2] * edges[1] + 0.01),
                                           saturate(edges[1] * edges[3] + 0.01), saturate(edges[3] * edges[0] + 0.01)};
      const Vec3 surface = add(add(scale(cross(dl, dt), accepted[0]), scale(cross(dt, dr), accepted[1])),
                               add(scale(cross(dr, db), accepted[2]), scale(cross(db, dl), accepted[3])));
      const double pulled = z * 0.99999;
      const Vec3 pixCentre = position(cx, cy, pulled);
      const Vec3 view = normalize(scale(pixCentre, -1.0));
      const Vec3 depthNormal = safe_normalize(surface, scale(view, -1.0));
      Vec3 normal = depthNormal;
      if ((params.flags & rules::FlagNormalFromDepth) == 0) {
        const Vec3 gx = safe_normalize(decode_normal(packedNormal), {0.0, 0.0, 1.0});
        normal = {gx[0], gx[1], -gx[2]};
        if (dot(normal, depthNormal) < 0.0) {
          normal = scale(normal, -1.0);
        }
      }
      normal = safe_normalize(add(normal, scale(view, std::max(0.0, -dot(normal, view)))), depthNormal);
      const double pixelSize = 2.0 * d * pulled / (camera.p00 * width);
      const double screenRadius = params.effectRadius / pixelSize;
      if (screenRadius < rules::PixelTooClose) {
        continue;
      }
      const std::array<float, 2> noise = rules::noise(px, py, params.noiseIndex);
      const double falloffRange = params.falloffRange * params.effectRadius;
      const double falloffFrom = params.effectRadius * (1.0 - params.falloffRange);
      const double falloffMul = -1.0 / falloffRange;
      const double falloffAdd = falloffFrom / falloffRange + 1.0;
      const double minS = rules::PixelTooClose / screenRadius;
      double visibility = 0.0;
      double continuous = 0.0;
      for (uint32_t slice = 0; slice < params.sliceCount; ++slice) {
        const double phi = (slice + double{noise[0]}) / params.sliceCount * Pi;
        const double omegaX = std::cos(phi) * screenRadius;
        const double omegaY = -std::sin(phi) * screenRadius;
        uint32_t mask = 0;
        std::vector<std::array<double, 2>> intervals;
        for (uint32_t index = 0; index < params.stepsPerSlice; ++index) {
          const double raw = noise[1] + (slice + index * params.stepsPerSlice) * 0.618033988749895;
          const double stepNoise = raw - std::floor(raw);
          const double s =
              std::pow(std::max((index + stepNoise) / params.stepsPerSlice, 1e-6), double{params.distributionPower}) +
              minS;
          const double offsetX = s * omegaX;
          const double offsetY = s * omegaY;
          const double mipLevel =
              std::clamp(std::log2(std::hypot(offsetX, offsetY)) - params.mipOffset, 0.0, double{rules::MipLevels});
          const uint32_t mip = std::min(static_cast<uint32_t>(round_even(mipLevel)), maxMip);
          const double snappedX = round_even(offsetX);
          const double snappedY = round_even(offsetY);
          for (uint32_t side = 0; side < 2; ++side) {
            const double sideSign = side == 0 ? 1.0 : -1.0;
            const int64_t gx = std::clamp<int64_t>(px + static_cast<int64_t>(sideSign * snappedX), 0, gridWidth - 1);
            const int64_t gy = std::clamp<int64_t>(py + static_cast<int64_t>(sideSign * snappedY), 0, gridHeight - 1);
            const uint32_t mipWidth = pyramid.widths[mip];
            const uint32_t mipHeight = pyramid.heights[mip];
            const uint32_t tx = std::min(static_cast<uint32_t>(gx) >> mip, mipWidth - 1);
            const uint32_t ty = std::min(static_cast<uint32_t>(gy) >> mip, mipHeight - 1);
            const double sz = pyramid.levels[mip][size_t{ty} * mipWidth + tx];
            if (sky(sz)) {
              continue;
            }
            const double cellX = (tx + 0.5) * double(1u << mip) - 0.5;
            const double cellY = (ty + 0.5) * double(1u << mip) - 0.5;
            const double towardX = cellX - px;
            const double towardY = cellY - py;
            const double half = double(d / 2) + 0.5;
            const Vec3 delta = sub(position(cellX * d + half, cellY * d + half, sz), pixCentre);
            const double dist = length(delta);
            const double weight = saturate(dist * falloffMul + falloffAdd);
            if (weight <= 0.0 || dist <= 0.0 || towardX * towardX + towardY * towardY < 0.25) {
              continue;
            }
            const double n = normal_angle(sideSign * towardX, sideSign * towardY, width, height, camera, view, normal);
            const double front = acos_clamped(dot(scale(delta, 1.0 / dist), view));
            const double back = acos_clamped(dot(normalize(sub(delta, scale(view, params.thickness))), view));
            double lo = 0.0;
            double hi = 0.0;
            if (side == 0) {
              lo = 1.0 - weight * (1.0 - (front - n + Pi / 2) / Pi);
              hi = 1.0 - weight * (1.0 - (back - n + Pi / 2) / Pi);
            } else {
              lo = weight * ((Pi / 2 - back - n) / Pi);
              hi = weight * ((Pi / 2 - front - n) / Pi);
            }
            mask |= rules::sector_bits(static_cast<float>(lo), static_cast<float>(hi), ceilMode);
            intervals.push_back({lo, hi});
          }
        }
        visibility += 1.0 - std::popcount(mask) / double{rules::SectorCount};
        continuous += 1.0 - interval_union(std::move(intervals));
      }
      texel.visibility = std::max(double{params.minVisibility}, visibility / params.sliceCount);
      texel.continuous = std::max(double{params.minVisibility}, continuous / params.sliceCount);
    }
  }
  return out;
}

std::vector<double> upsample(const UpsampleSetup& setup) {
  const render::detail::ao::Params& params = setup.params;
  const uint32_t width = setup.width;
  const uint32_t height = setup.height;
  const uint32_t d =
      std::max(rules::divisor_from_dims(width, setup.gridWidth), rules::divisor_from_dims(height, setup.gridHeight));
  struct Tap {
    double a = 1.0;
    bool receiver = false;
    double z = 0.0;
    std::array<double, 4> edges{};
  };
  std::vector<double> out(size_t{width} * height, 1.0);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const float dz = setup.depth[size_t{y} * width + x];
      if ((params.flags & rules::FlagIdentityForced) != 0 || dz <= 0.f) {
        continue;
      }
      const double z = view_depth(setup.camera, dz);
      const double gx = (double(x) - double(d / 2)) / d;
      const double gy = (double(y) - double(d / 2)) / d;
      const double baseX = std::floor(gx);
      const double baseY = std::floor(gy);
      const double fx = gx - baseX;
      const double fy = gy - baseY;
      std::array<Tap, 4> taps{};
      uint32_t home = 4;
      double homeGap = 0.0;
      for (uint32_t i = 0; i < 4; ++i) {
        const auto tx = static_cast<uint32_t>(std::clamp<double>(baseX + (i & 1), 0.0, setup.gridWidth - 1.0));
        const auto ty = static_cast<uint32_t>(std::clamp<double>(baseY + (i >> 1), 0.0, setup.gridHeight - 1.0));
        const uint32_t raw = setup.raw[size_t{ty} * setup.gridWidth + tx];
        taps[i] = {
            .a = saturate((raw & 0xFFu) / 255.0 * rules::RawScale),
            .receiver = ((raw >> 16) & 0xFFu) / 255.0 > 0.5,
            .z = setup.hiz[size_t{ty} * setup.gridWidth + tx],
            .edges = unpack_edges(static_cast<uint8_t>((raw >> 8) & 0xFFu)),
        };
        const double gap = std::abs(taps[i].z - z);
        if (taps[i].receiver && (home == 4 || gap < homeGap)) {
          home = i;
          homeGap = gap;
        }
      }
      const double reject = params.depthReject * z;
      if (home == 4 || homeGap > reject) {
        continue;
      }
      double sum = 0.0;
      double weights = 0.0;
      bool allOpen = true;
      for (uint32_t i = 0; i < 4; ++i) {
        const double wx = (i & 1) != 0 ? fx : 1.0 - fx;
        const double wy = (i >> 1) != 0 ? fy : 1.0 - fy;
        double edge = 1.0;
        if (i != home) {
          const std::array<double, 4>& e = taps[home].edges;
          const bool right = (i & 1) > (home & 1);
          const bool below = (i >> 1) > (home >> 1);
          const double horizontal = right ? e[1] : e[0];
          const double vertical = below ? e[3] : e[2];
          if ((i >> 1) == (home >> 1)) {
            edge = horizontal;
          } else if ((i & 1) == (home & 1)) {
            edge = vertical;
          } else {
            const std::array<double, 4>& across = taps[(i & 1) | (home & 2)].edges;
            const std::array<double, 4>& down = taps[(home & 1) | (i & 2)].edges;
            edge = 0.5 * (horizontal * (below ? across[3] : across[2]) + vertical * (right ? down[1] : down[0]));
          }
        }
        const double w =
            taps[i].receiver ? (wx * wy + 1e-3) * edge * saturate(1.0 - std::abs(taps[i].z - z) / reject) : 0.0;
        if (w > 0.0) {
          sum += w * taps[i].a;
          weights += w;
          allOpen = allOpen && taps[i].a >= 1.0;
        }
      }
      if (allOpen || weights <= 0.0) {
        continue;
      }
      const double v = rules::apply_strength(static_cast<float>(sum / weights), params.strength, params.minVisibility);
      if (v < 1.0 && normal_code(setup.normals[size_t{y} * width + x]) == 0) {
        continue;
      }
      out[size_t{y} * width + x] = v;
    }
  }
  return out;
}
} // namespace aurora::test::ao

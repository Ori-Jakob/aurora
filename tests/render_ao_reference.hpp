#pragma once

#include "render/ao/ao.hpp"
#include "render/stage_camera.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::test::ao {
struct CpuPyramid {
  std::vector<uint32_t> widths;
  std::vector<uint32_t> heights;
  std::vector<std::vector<double>> levels;

  [[nodiscard]] uint32_t level_count() const noexcept { return static_cast<uint32_t>(levels.size()); }
};

double view_depth(const render::detail::StageCamera& camera, float depth);
double depth_mip_filter(const std::array<double, 4>& depths, double effectRadius);
std::array<double, 3> view_position_fwd(const render::detail::StageCamera& camera, double u, double v, double z,
                                        double width, double height);
CpuPyramid build_pyramid(std::span<const float> depth, uint32_t width, uint32_t height, uint32_t divisor,
                         const render::detail::StageCamera& camera, double effectRadius, uint32_t levels);

uint32_t pack_normal(const std::array<double, 3>& gxNormal, uint32_t code);
uint32_t normal_code(uint32_t packed);
std::array<double, 4> calculate_edges(double z, double left, double right, double top, double bottom);
uint8_t pack_edges(const std::array<double, 4>& edges);
std::array<double, 4> unpack_edges(uint8_t packed);
double interval_union(std::vector<std::array<double, 2>> intervals);

using Rgba = std::array<double, 4>;
using Normal3 = std::array<double, 3>;

struct SsilPyramids {
  std::vector<uint32_t> widths;
  std::vector<uint32_t> heights;
  std::vector<std::vector<Rgba>> radiance;
  std::vector<std::vector<Normal3>> normals;
};

std::array<double, 4> ssil_depth_mip_weights(const std::array<double, 4>& depths, double effectRadius);
double ssil_facing(const Normal3& meanNormal, const Normal3& toEmitter);
Rgba ssil_clamp_radiance(const Rgba& premultiplied);
std::array<double, 4> ssil_prev_depth_weights(double fx, double fy, const std::array<double, 4>& logDepths, double z,
                                              double reject);
SsilPyramids build_ssil_pyramids(std::vector<Rgba> radiance, std::vector<Normal3> normals, const CpuPyramid& depth,
                                 double effectRadius);

struct GtaoTexel {
  double visibility = 1.0;
  double continuous = 1.0;
  bool receiver = false;
  uint8_t edges = 0;
  Rgba ssil{};
  double occluded = 0.0;
};

struct GtaoSetup {
  std::span<const float> depth;
  std::span<const uint32_t> normals;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t divisor = 2;
  render::detail::StageCamera camera;
  render::detail::ao::Params params;
  const SsilPyramids* ssil = nullptr;
};

std::vector<GtaoTexel> estimate_gtao(const GtaoSetup& setup, const CpuPyramid& pyramid);

struct UpsampleSetup {
  std::span<const float> depth;
  std::span<const uint32_t> normals;
  std::span<const uint32_t> raw;
  std::span<const double> hiz;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t gridWidth = 0;
  uint32_t gridHeight = 0;
  render::detail::StageCamera camera;
  render::detail::ao::Params params;
};

std::vector<double> upsample(const UpsampleSetup& setup);
} // namespace aurora::test::ao

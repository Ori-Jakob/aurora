#pragma once

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
} // namespace aurora::test::ao

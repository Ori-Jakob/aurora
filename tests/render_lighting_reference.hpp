#pragma once

#include "render/lighting/clusters.hpp"
#include "render/stage_camera.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::test::lighting {
using render::detail::StageCamera;
using render::detail::lighting::clusters::GpuLight;
using render::detail::lighting::clusters::LightListHeader;
using render::detail::lighting::clusters::MaskWords;

using Mask = std::array<uint32_t, MaskWords>;
using Vec3d = std::array<double, 3>;

struct Box {
  Vec3d lo{};
  Vec3d hi{};
};

[[nodiscard]] Vec3d view_position_fwd(const StageCamera& camera, double u, double v, double z, double width,
                                      double height) noexcept;
[[nodiscard]] float view_depth(const StageCamera& camera, float stored) noexcept;
[[nodiscard]] std::array<float, 2> cluster_depths(const StageCamera& camera, const LightListHeader& header,
                                                  uint32_t slice) noexcept;
[[nodiscard]] Box cluster_box(const StageCamera& camera, const LightListHeader& header, uint32_t cluster,
                              uint32_t width, uint32_t height, double depthInflate) noexcept;
[[nodiscard]] std::vector<Mask> assign_reference(const LightListHeader& header, std::span<const GpuLight> lights,
                                                 const StageCamera& camera, uint32_t width, uint32_t height,
                                                 double radiusScale, double depthInflate);
[[nodiscard]] std::array<uint8_t, 3> heatmap_rgb(uint32_t count, bool border, bool sky) noexcept;
} // namespace aurora::test::lighting

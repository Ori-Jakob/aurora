#pragma once

#include "hints.hpp"

#include <aurora/gfx.hpp>
#include <dolphin/mtx.h>
#include <dolphin/types.h>

#include <array>
#include <optional>

namespace render_demo {
struct SceneOptions {
  bool hints = true;
  bool captureOnlyBox = false;
  aurora::gfx::DrawTypeId customDrawProbe = aurora::gfx::InvalidDrawType;
  bool gpuArrays = false;
  uint32_t gpuPadMiB = 0;
  uint64_t gpuCopySystem = 0;
  bool fogBands = false;
};

struct SceneCamera {
  Mtx44 projection;
  Mtx view;
  Point3d eye;
  Point3d target;
};

enum class MotionProbe : u32 { Ground, Box, Tentacle, Water };

SceneCamera scene_camera(u32 frame) noexcept;
std::array<std::array<f32, 2>, 4> motion_probe_points(u32 frame) noexcept;
std::optional<std::array<f32, 2>> motion_probe_expected(MotionProbe probe, u32 frame, std::array<f32, 2> ndc) noexcept;
const std::array<FogParams, 2>& fog_bands() noexcept;
uint32_t gpu_uploads_per_frame(const SceneOptions& options) noexcept;
bool init_scene();
void draw_scene(u32 frame, const SceneOptions& options);
void shutdown_scene();
} // namespace render_demo

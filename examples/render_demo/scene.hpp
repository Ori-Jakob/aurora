#pragma once

#include <aurora/gfx.hpp>
#include <dolphin/mtx.h>
#include <dolphin/types.h>

namespace render_demo {
struct SceneOptions {
  bool hints = true;
  bool captureOnlyBox = false;
  aurora::gfx::DrawTypeId customDrawProbe = aurora::gfx::InvalidDrawType;
};

struct SceneCamera {
  Mtx44 projection;
  Mtx view;
  Point3d eye;
  Point3d target;
};

SceneCamera scene_camera(u32 frame) noexcept;
bool init_scene();
void draw_scene(u32 frame, const SceneOptions& options);
void shutdown_scene();
} // namespace render_demo

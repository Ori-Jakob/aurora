#pragma once

#include <aurora/gfx.hpp>
#include <dolphin/types.h>

namespace render_demo {
struct SceneOptions {
  bool hints = true;
  bool captureOnlyBox = false;
  aurora::gfx::DrawTypeId customDrawProbe = aurora::gfx::InvalidDrawType;
};

bool init_scene();
void draw_scene(u32 frame, const SceneOptions& options);
void shutdown_scene();
} // namespace render_demo

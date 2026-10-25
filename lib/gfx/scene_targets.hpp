#pragma once

#include <aurora/gfx.hpp>

#include <cstdint>

namespace aurora::gfx {
struct SceneTargetFlags {
  bool normal = false, gbuffer = false, debugId = false;
  bool operator==(const SceneTargetFlags&) const = default;
};

struct SceneLayoutBase {
  wgpu::TextureFormat color = wgpu::TextureFormat::Undefined;
  wgpu::TextureFormat depth = wgpu::TextureFormat::Undefined;
  wgpu::Extent3D size{};
  uint32_t samples = 1;
};

RenderTargetLayout compose_scene_layout(const SceneTargetFlags& flags, const SceneLayoutBase& base) noexcept;
SceneTargetFlags live_scene_targets() noexcept;
SceneLayoutBase live_layout_base() noexcept;

inline bool layout_has_gbuffer(const RenderTargetLayout& layout) noexcept {
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    const ColorAttachmentSemantic semantic = layout.colorAttachments[i].semantic;
    if (semantic == ColorAttachmentSemantic::Motion || semantic == ColorAttachmentSemantic::Reactive ||
        semantic == ColorAttachmentSemantic::Fog) {
      return true;
    }
  }
  return false;
}

inline bool clears_with_depth(ColorAttachmentSemantic semantic) noexcept {
  return semantic == ColorAttachmentSemantic::Normal || semantic == ColorAttachmentSemantic::Motion;
}

inline bool clears_with_color(ColorAttachmentSemantic semantic) noexcept {
  return semantic == ColorAttachmentSemantic::Reactive || semantic == ColorAttachmentSemantic::Fog;
}

namespace detail {
void latch_scene_layout(bool wantGBuffer);
} // namespace detail
} // namespace aurora::gfx

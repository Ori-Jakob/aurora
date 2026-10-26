#pragma once

#include "pipeline_cache.hpp"

#include <aurora/gfx.hpp>

#include <cstdint>

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

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

inline constexpr uint32_t SceneLayoutMatchedFrames = 2;
inline constexpr uint32_t MaxSceneLayoutWarmConfigs = 4096;

struct LayoutWarm {
  RenderTargetLayout pending;
  uint64_t liveKey = 0;
};

enum class SceneLayoutAction : uint8_t { None, BeginWarm, Restart, Switch, Refuse, Cancel };

struct SceneLayoutInput {
  bool want = false;
  bool live = false;
  bool warming = false;
  bool pendingKeyChanged = false;
  bool firstLatch = false;
  bool blocking = false;
  bool supported = false;
  WarmProgress progress;
  uint32_t hardFailed = 0;
  uint32_t matchedFrames = 0;
  uint64_t elapsedMs = 0;
};

SceneLayoutAction next_scene_layout_action(const SceneLayoutInput& input) noexcept;
RenderTargetLayout pending_scene_layout(bool modNormal, bool gbuffer, const SceneLayoutBase& base) noexcept;

const LayoutWarm* layout_warm() noexcept;
void warm_scene_variant(const gx::PipelineConfig& config);
bool scene_layout_transitioning() noexcept;
bool gbuffer_live_or_pending() noexcept;

namespace detail {
void latch_scene_layout(bool wantGBuffer);
void refresh_layout_warm() noexcept;
void reset_scene_layout();
} // namespace detail

namespace testing {
struct WarmTally {
  WarmProgress progress;
  uint32_t hardFailed = 0;
  uint32_t pairs = 0;
  uint32_t overflow = 0;
};

void begin_warm(const RenderTargetLayout& pending, uint64_t liveKey);
uint32_t latch_warm() noexcept;
WarmTally warm_tally();
} // namespace testing
} // namespace aurora::gfx

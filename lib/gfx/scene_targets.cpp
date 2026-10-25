#include "scene_targets.hpp"

#include "frame_packet.hpp"
#include "../logging.hpp"
#include "../webgpu/gpu.hpp"

#include <utility>

namespace aurora::gfx {
namespace {
constexpr Module Log{"aurora::gfx"};

bool g_lastWant = false;
bool g_loggedUnsupported = false;

void append_target(RenderTargetLayout& layout, ColorAttachmentSemantic semantic, wgpu::TextureFormat format,
                   const wgpu::Extent3D& size) noexcept {
  layout.colorAttachments[layout.colorAttachmentCount++] = {
      .semantic = semantic,
      .format = format,
      .width = size.width,
      .height = size.height,
  };
}
} // namespace

RenderTargetLayout compose_scene_layout(const SceneTargetFlags& flags, const SceneLayoutBase& base) noexcept {
  RenderTargetLayout layout{
      .depthStencilFormat = base.depth,
      .sampleCount = base.samples,
  };
  append_target(layout, ColorAttachmentSemantic::SceneColor, base.color, base.size);
  if (flags.normal || flags.gbuffer) {
    append_target(layout, ColorAttachmentSemantic::Normal, webgpu::NormalBufferFormat, base.size);
  }
  if (flags.gbuffer) {
    // Fixed order keeps the WebGPU byte cost at 8, 16, 20, 21, 25
    append_target(layout, ColorAttachmentSemantic::Motion, webgpu::MotionBufferFormat, base.size);
    append_target(layout, ColorAttachmentSemantic::Reactive, webgpu::ReactiveBufferFormat, base.size);
    append_target(layout, ColorAttachmentSemantic::Fog, webgpu::FogRecordFormat, base.size);
  }
  // Last, so later targets go before it and the base layout stays a prefix
  if (flags.debugId) {
    append_target(layout, ColorAttachmentSemantic::DebugId, webgpu::DebugIdBufferFormat, base.size);
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}

SceneTargetFlags live_scene_targets() noexcept {
  const auto& config = webgpu::g_graphicsConfig;
  return {.normal = webgpu::normal_present(), .gbuffer = config.gbuffer, .debugId = config.debugIdBuffer};
}

SceneLayoutBase live_layout_base() noexcept {
  return {
      .color = webgpu::g_frameBuffer.format,
      .depth = webgpu::g_graphicsConfig.depthFormat,
      .size = webgpu::g_frameBuffer.size,
      .samples = webgpu::g_graphicsConfig.msaaSamples,
  };
}

namespace detail {
void latch_scene_layout(bool wantGBuffer) {
  if (std::exchange(g_lastWant, wantGBuffer) != wantGBuffer) {
    g_loggedUnsupported = false;
  }
  if (wantGBuffer == webgpu::g_graphicsConfig.gbuffer) {
    return;
  }
  if (wantGBuffer && !webgpu::scene_targets_supported()) {
    if (!std::exchange(g_loggedUnsupported, true)) {
      Log.warn("Scene targets: base G-buffer unavailable on this device");
    }
    return;
  }
  if (webgpu::set_gbuffer(wantGBuffer)) {
    Log.info("Scene targets: base G-buffer {}", wantGBuffer ? "on" : "off");
  }
}
} // namespace detail
} // namespace aurora::gfx

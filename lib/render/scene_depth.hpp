#pragma once

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace aurora::render::detail {
struct SceneDepth {
  DepthSource source = DepthSource::Unavailable;
  wgpu::Texture sampledTexture;
  wgpu::TextureView sampled;
  wgpu::TextureFormat sampledFormat = wgpu::TextureFormat::Undefined;
  wgpu::Texture attachmentTexture;
  wgpu::TextureView attachment;
  wgpu::TextureFormat attachmentFormat = wgpu::TextureFormat::Undefined;
  uint32_t attachmentSamples = 1;
  wgpu::Extent3D size{};
};

constexpr DepthSource select_scene_depth_source(bool coreFeatures, uint32_t msaaSamples,
                                                bool snapshotSupported) noexcept {
  if (!coreFeatures) {
    return DepthSource::Unavailable;
  }
  if (msaaSamples > 1) {
    return snapshotSupported ? DepthSource::Copy : DepthSource::Unavailable;
  }
  return DepthSource::Direct;
}

constexpr std::string_view scene_depth_source_name(DepthSource source) noexcept {
  switch (source) {
  case DepthSource::Direct:
    return "direct";
  case DepthSource::Copy:
    return "copy";
  default:
    return "unavailable";
  }
}

constexpr std::string_view depth_format_name(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::Depth32Float:
    return "Depth32Float";
  case wgpu::TextureFormat::Depth24Plus:
    return "Depth24Plus";
  case wgpu::TextureFormat::Depth16Unorm:
    return "Depth16Unorm";
  default:
    return "unknown";
  }
}

inline std::string scene_depth_choice_text(DepthSource source, uint32_t msaaSamples, bool coreFeatures,
                                           wgpu::TextureFormat depthFormat) {
  std::string text{"Scene depth for render passes: "};
  switch (source) {
  case DepthSource::Direct:
    text += "direct (";
    text += depth_format_name(depthFormat);
    text += ", 1 sample)";
    break;
  case DepthSource::Copy:
    text += "copy (R32Float snapshot of sample 0, ";
    text += std::to_string(msaaSamples);
    text += "x MSAA)";
    break;
  default:
    text += coreFeatures ? "unavailable (no depth snapshot pipeline)" : "unavailable (compatibility mode)";
    break;
  }
  return text;
}
} // namespace aurora::render::detail

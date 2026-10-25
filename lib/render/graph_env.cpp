#include "graph.hpp"
#include "debug/debug.hpp"

#include "../webgpu/gpu.hpp"

namespace aurora::render::detail {
namespace {
// Game thread only, like the latch that gathers the environment
WGPUDevice g_limitsDevice = nullptr;
GraphLimits g_deviceLimits;

const GraphLimits& device_limits() {
  const auto& device = webgpu::g_device;
  if (device.Get() == g_limitsDevice) {
    return g_deviceLimits;
  }
  wgpu::CompatibilityModeLimits compat{};
  wgpu::Limits limits{.nextInChain = &compat};
  device.GetLimits(&limits);
  const GraphFeatures features{
      .core = device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits),
      .tier1Formats = device.HasFeature(wgpu::FeatureName::TextureFormatsTier1),
      .tier2Formats = device.HasFeature(wgpu::FeatureName::TextureFormatsTier2),
      .bgra8Storage = device.HasFeature(wgpu::FeatureName::BGRA8UnormStorage),
      .rg11b10Renderable = device.HasFeature(wgpu::FeatureName::RG11B10UfloatRenderable),
      .float32Filterable = device.HasFeature(wgpu::FeatureName::Float32Filterable),
  };
  g_deviceLimits = graph_limits(limits, compat, features);
  g_limitsDevice = device.Get();
  return g_deviceLimits;
}

constexpr size_t gx_index(GxResource resource) noexcept { return static_cast<size_t>(resource); }
} // namespace

GraphEnvironment gather_graph_environment() {
  if (!webgpu::g_device) {
    g_limitsDevice = nullptr;
    return {};
  }
  const auto& config = webgpu::g_graphicsConfig;
  GraphEnvironment env;
  env.limits = device_limits();
  env.efbSize = {webgpu::g_frameBuffer.size.width, webgpu::g_frameBuffer.size.height};
  env.outputSize = {config.surfaceConfiguration.width, config.surfaceConfiguration.height};
  // The usages gpu.cpp creates the EFB textures with; the graph never widens them
  env.gxFormats[gx_index(GxResource::Color)] = config.surfaceConfiguration.format;
  env.gxUsage[gx_index(GxResource::Color)] = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                                             wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
  env.gxFormats[gx_index(GxResource::Depth)] = config.depthFormat;
  env.gxUsage[gx_index(GxResource::Depth)] = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
  constexpr wgpu::TextureUsage SceneTargetUsage =
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  if (webgpu::normal_present()) {
    env.gxFormats[gx_index(GxResource::Normal)] = webgpu::NormalBufferFormat;
    env.gxUsage[gx_index(GxResource::Normal)] = SceneTargetUsage;
  }
  if (config.gbuffer) {
    env.gxFormats[gx_index(GxResource::Motion)] = webgpu::MotionBufferFormat;
    env.gxUsage[gx_index(GxResource::Motion)] = SceneTargetUsage;
    env.gxFormats[gx_index(GxResource::Reactive)] = webgpu::ReactiveBufferFormat;
    env.gxUsage[gx_index(GxResource::Reactive)] = SceneTargetUsage;
    env.gxFormats[gx_index(GxResource::Fog)] = webgpu::FogRecordFormat;
    env.gxUsage[gx_index(GxResource::Fog)] = SceneTargetUsage;
  }
  if (config.debugIdBuffer) {
    env.gxFormats[gx_index(GxResource::DebugId)] = webgpu::DebugIdBufferFormat;
    env.gxUsage[gx_index(GxResource::DebugId)] =
        wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;
  }
  env.gxSampleCount = config.msaaSamples;
  env.profile = webgpu::g_featureProfile;
  env.inspectorOpen = debug::inspector_open();
  return env;
}
} // namespace aurora::render::detail

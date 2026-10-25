#include "gpu.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <aurora/aurora.h>
#include <aurora/webgpu.hpp>
#include <aurora/gfx.h>
#include <magic_enum.hpp>
#include <webgpu/webgpu_cpp.h>
#include <SDL3/SDL_stdinc.h>

#include "../gx/gx.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/render_worker.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#include "feature_profile.hpp"
#include "caps.hpp"
#include "gpu_prof.hpp"

#ifdef WEBGPU_DAWN
#include "../dawn/TracyPlatform.hpp"
#include <dawn/native/DawnNative.h>
#endif

namespace aurora::webgpu {
static Module Log("aurora::gpu");

wgpu::Device g_device;
wgpu::Queue g_queue;
wgpu::Surface g_surface;
wgpu::BackendType g_backendType;
GraphicsConfig g_graphicsConfig;
TextureWithSampler g_frameBuffer;
TextureWithSampler g_frameBufferResolved;
TextureWithSampler g_depthBuffer;
TextureWithSampler g_normalBuffer;
TextureWithSampler g_debugIdBuffer;

// EFB -> XFB copy pipeline
static wgpu::BindGroupLayout g_CopyBindGroupLayout;
wgpu::RenderPipeline g_CopyPipeline;
wgpu::RenderPipeline g_CopyPremultipliedAlphaPipeline;
wgpu::BindGroup g_CopyBindGroup;
static AuroraSampler g_Resampler = SAMPLER_BILINEAR;
static wgpu::BindGroupLayout g_ResampleBindGroupLayout;
static wgpu::RenderPipeline g_ResamplePipeline;
static wgpu::Buffer g_ResampleUniformBuffer;
static TextureWithSampler g_resampledFrameBuffer;

static wgpu::Adapter g_adapter;
wgpu::Instance g_instance;
wgpu::AdapterInfo g_adapterInfo;
static wgpu::SurfaceCapabilities g_surfaceCapabilities;
bool g_hasCoreFeatures = false;
bool g_dualSourceBlendingSupported = false;
bool g_bcTexturesSupported = false;
bool g_astcTexturesSupported = false;
bool g_textureComponentSwizzleSupported = false;
AuroraFeatureProfile g_featureProfile = FEATURE_PROFILE_FULL;
static std::atomic_bool g_initialized = false;
static std::atomic_bool g_vsyncEnabled = true;

namespace {

AuroraLogLevel wgpu_log_level(wgpu::LoggingType type) {
  switch (type) {
  case wgpu::LoggingType::Verbose:
    return LOG_DEBUG;
  case wgpu::LoggingType::Info:
    return LOG_INFO;
  case wgpu::LoggingType::Warning:
    return LOG_WARNING;
  case wgpu::LoggingType::Error:
    return LOG_ERROR;
  default:
    return LOG_FATAL;
  }
}

void wgpu_log(wgpu::LoggingType type, wgpu::StringView message) {
  Log.report(wgpu_log_level(type), "WebGPU message: {}", message);
}

void log_feature_profile(const ResolvedFeatureProfile& resolved, const wgpu::SupportedFeatures& adapterFeatures,
                         const wgpu::Limits& adapterLimits) {
  if (resolved.profile == FEATURE_PROFILE_FULL && resolved.source == FeatureProfileSource::Default) {
    Log.info("Feature profile: full");
    return;
  }
  wgpu::Limits deviceLimits{};
  g_device.GetLimits(&deviceLimits);
  std::string maskedList;
  for (const auto& masked : feature_profile_masked_features(resolved.profile)) {
    const auto begin = adapterFeatures.features;
    const auto end = begin + adapterFeatures.featureCount;
    const bool adapterHas = std::find(begin, end, masked.feature) != end;
    maskedList += fmt::format("\n    {} (adapter: {})", masked.name, adapterHas ? "yes" : "no");
  }
  const uint32_t budget = feature_profile_color_budget(resolved.profile);
  Log.info(
      "Feature profile: {} (source: {})"
      "\n  Core features: {}"
      "\n  Validation: {}"
      "\n  Masked features:{}"
      "\n  maxColorAttachmentBytesPerSample: adapter {}, cap {}, device {}",
      feature_profile_name(resolved.profile), feature_profile_source_name(resolved.source),
      g_device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits) ? "enabled" : "disabled (compatibility validation)",
      feature_profile_forces_validation(resolved.profile) ? "forced on" : "build default",
      maskedList.empty() ? " none" : maskedList, adapterLimits.maxColorAttachmentBytesPerSample,
      budget == 0 ? std::string{"none"} : std::to_string(budget), deviceLimits.maxColorAttachmentBytesPerSample);
}

std::string limit_text(uint32_t value) {
  return value == WGPU_LIMIT_U32_UNDEFINED ? std::string{"default"} : std::to_string(value);
}

std::string limit_text(uint64_t value) {
  return value == WGPU_LIMIT_U64_UNDEFINED ? std::string{"default"} : std::to_string(value);
}

void log_required_limits(const wgpu::Limits& limits, bool core) {
  Log.info(
      "Using limits ({}):"
      "\n  maxTextureDimension1D: {}"
      "\n  maxTextureDimension2D: {}"
      "\n  maxTextureDimension3D: {}"
      "\n  maxTextureArrayLayers: {}"
      "\n  maxStorageBuffersPerShaderStage: {}"
      "\n  minUniformBufferOffsetAlignment: {}"
      "\n  minStorageBufferOffsetAlignment: {}"
      "\n  maxImmediateSize: {}"
      "\n  maxColorAttachments: {}"
      "\n  maxColorAttachmentBytesPerSample: {}"
      "\n  maxInterStageShaderVariables: {}"
      "\n  maxStorageTexturesPerShaderStage: {}"
      "\n  maxSampledTexturesPerShaderStage: {}"
      "\n  maxBufferSize: {}"
      "\n  maxStorageBufferBindingSize: {}"
      "\n  maxComputeWorkgroupStorageSize: {}",
      core ? "core" : "compatibility", limit_text(limits.maxTextureDimension1D),
      limit_text(limits.maxTextureDimension2D), limit_text(limits.maxTextureDimension3D),
      limit_text(limits.maxTextureArrayLayers), limit_text(limits.maxStorageBuffersPerShaderStage),
      limit_text(limits.minUniformBufferOffsetAlignment), limit_text(limits.minStorageBufferOffsetAlignment),
      limit_text(limits.maxImmediateSize), limit_text(limits.maxColorAttachments),
      limit_text(limits.maxColorAttachmentBytesPerSample), limit_text(limits.maxInterStageShaderVariables),
      limit_text(limits.maxStorageTexturesPerShaderStage), limit_text(limits.maxSampledTexturesPerShaderStage),
      limit_text(limits.maxBufferSize), limit_text(limits.maxStorageBufferBindingSize),
      limit_text(limits.maxComputeWorkgroupStorageSize));
}

std::string feature_state_list(const std::vector<FeatureRequestResult>& requests, FeatureRequestState state) {
  std::string list;
  for (const auto& request : requests) {
    if (request.state != state) {
      continue;
    }
    if (!list.empty()) {
      list += ", ";
    }
    list += request.name;
  }
  return list.empty() ? std::string{"none"} : list;
}

void log_caps(const render::Caps& caps, const std::vector<FeatureRequestResult>& requests) {
  constexpr uint64_t MiB = uint64_t{1} << 20;
  const auto& limits = caps.limits;
  const auto vertex = render::stage_binding_limits(caps, render::ShaderStage::Vertex);
  const auto fragment = render::stage_binding_limits(caps, render::ShaderStage::Fragment);
  const auto subgroups =
      caps.subgroups ? fmt::format("{}-{}", caps.subgroupMinSize, caps.subgroupMaxSize) : std::string{"no"};
  const auto vram = caps.vramSource == render::VramSource::Unknown
                        ? std::string{"unknown"}
                        : fmt::format("{} MiB ({}{})", caps.vramBytes / MiB, render::vram_source_name(caps.vramSource),
                                      caps.unifiedMemory ? ", unified" : "");
  Log.info(
      "Device caps:"
      "\n  Backend: {} ({}), profile {}, core features: {}"
      "\n  Masked by profile: {}"
      "\n  Not supported by adapter: {}"
      "\n  Subgroups: {}"
      "\n  Color attachments: {}, {} bytes per sample; inter-stage variables: {}"
      "\n  Storage buffers per stage: {} (vertex {}, fragment {}); storage textures: {} (vertex {}, fragment {})"
      "\n  Sampled textures: {}; samplers: {}; uniform buffers: {}"
      "\n  Max buffer: {} MiB; storage binding: {} MiB; workgroup storage: {} bytes"
      "\n  VRAM: {}"
      "\n  D3D12 interop: {}; shader model: {}",
      magic_enum::enum_name(g_backendType), render::adapter_kind_name(caps.adapterKind),
      feature_profile_name(caps.profile), caps.core ? "yes" : "no",
      feature_state_list(requests, FeatureRequestState::MaskedByProfile),
      feature_state_list(requests, FeatureRequestState::Unsupported), subgroups, limits.maxColorAttachments,
      limits.maxColorAttachmentBytesPerSample, limits.maxInterStageShaderVariables,
      limits.maxStorageBuffersPerShaderStage, vertex.storageBuffers, fragment.storageBuffers,
      limits.maxStorageTexturesPerShaderStage, vertex.storageTextures, fragment.storageTextures,
      limits.maxSampledTexturesPerShaderStage, limits.maxSamplersPerShaderStage, limits.maxUniformBuffersPerShaderStage,
      limits.maxBufferSize / MiB, limits.maxStorageBufferBindingSize / MiB, limits.maxComputeWorkgroupStorageSize, vram,
      caps.d3d12Interop ? "yes" : "no", shader_model_text(caps.d3dShaderModel));
}

render::Caps query_device_caps(std::span<const wgpu::FeatureName> adapterFeatures) {
  wgpu::SupportedFeatures deviceFeatures;
  g_device.GetFeatures(&deviceFeatures);
  wgpu::CompatibilityModeLimits compatLimits{};
  wgpu::Limits limits{.nextInChain = &compatLimits};
  g_device.GetLimits(&limits);
  limits.nextInChain = nullptr;
  const bool hasHeaps = g_device.HasFeature(wgpu::FeatureName::AdapterPropertiesMemoryHeaps);
  const bool hasD3D = g_device.HasFeature(wgpu::FeatureName::AdapterPropertiesD3D);
  wgpu::AdapterPropertiesMemoryHeaps memoryHeaps{};
  wgpu::AdapterPropertiesD3D d3dProperties{};
  wgpu::AdapterInfo info{};
  // Chaining a property struct the adapter lacks fails the whole query, so only chain what the device has
  wgpu::ChainedStructOut** next = &info.nextInChain;
  if (hasHeaps) {
    *next = &memoryHeaps;
    next = &memoryHeaps.nextInChain;
  }
  if (hasD3D) {
    *next = &d3dProperties;
  }
  const bool infoValid = g_device.GetAdapterInfo(&info) == wgpu::Status::Success;
  if (!infoValid) {
    Log.warn("Failed to query adapter info from the device; using the adapter's");
  }
  // The adapter query at init describes the same adapter, so backend and identity survive a failed device query
  const wgpu::AdapterInfo& ids = infoValid ? info : g_adapterInfo;
  const std::span<const wgpu::FeatureName> features{deviceFeatures.features, deviceFeatures.featureCount};
  auto deviceCaps = build_caps({
      .backend = ids.backendType,
      .adapterType = ids.adapterType,
      .profile = g_featureProfile,
      .vendorId = ids.vendorID,
      .deviceId = ids.deviceID,
      .adapterName = ids.device.IsUndefined() ? std::string_view{"Unknown"} : std::string_view{ids.device},
      .features = features,
      .limits = limits,
      .compatLimits = compatLimits,
      .subgroupMinSize = ids.subgroupMinSize,
      .subgroupMaxSize = ids.subgroupMaxSize,
      .memoryHeaps = infoValid && hasHeaps
                         ? std::span<const wgpu::MemoryHeapInfo>{memoryHeaps.heapInfo, memoryHeaps.heapCount}
                         : std::span<const wgpu::MemoryHeapInfo>{},
      .d3dShaderModel = infoValid && hasD3D ? d3dProperties.shaderModel : 0,
  });
  log_caps(deviceCaps, describe_feature_requests(adapterFeatures, features, g_featureProfile));
  return deviceCaps;
}

struct ResampleUniformBlock {
  uint32_t samplerMode = 0;
  float frameWidth = 0.f;
  float frameHeight = 0.f;
};

constexpr std::string_view resampleShaderSource = R"(
struct Uniforms {
    sampler_mode: u32,
    frame_width: f32,
    frame_height: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var t: texture_2d<f32>;

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, 1.0),
    vec2(-1.0, -3.0),
    vec2(3.0, 1.0),
);
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(0.0, 0.0),
    vec2(0.0, 2.0),
    vec2(2.0, 0.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vtxIdx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vtxIdx], 0.0, 1.0);
    out.uv = uvs[vtxIdx];
    return out;
}

// BEGIN AREA SAMPLER REGION

fn sample_by_pixel(pixel: vec2<i32>) -> vec4<f32> {
    let source_dims = textureDimensions(t);
    let max_coord = vec2<i32>(source_dims) - vec2<i32>(1, 1);
    let coord = clamp(pixel, vec2<i32>(0, 0), max_coord);
    return textureLoad(t, coord, 0);
}

fn sample_area(frag_position: vec4<f32>) -> vec4<f32> {
    let source_size = vec2<f32>(textureDimensions(t));
    let target_size = max(vec2<f32>(uniforms.frame_width, uniforms.frame_height), vec2<f32>(1.0, 1.0));

    let source_min = clamp((frag_position.xy - vec2<f32>(0.5, 0.5)) / target_size,
                           vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)) * source_size;
    let source_max = clamp((frag_position.xy + vec2<f32>(0.5, 0.5)) / target_size,
                           vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)) * source_size;

    let first_pixel = vec2<i32>(floor(source_min));
    let last_pixel = vec2<i32>(ceil(source_max));
    let max_iterations: i32 = 16;

    var avg_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var total_weight = 0.0;

    for (var iy: i32 = 0; iy < max_iterations; iy = iy + 1) {
        let source_y = first_pixel.y + iy;
        if (source_y < last_pixel.y) {
            let y0 = f32(source_y);
            let weight_y = max(min(source_max.y, y0 + 1.0) - max(source_min.y, y0), 0.0);

            for (var ix: i32 = 0; ix < max_iterations; ix = ix + 1) {
                let source_x = first_pixel.x + ix;
                if (source_x < last_pixel.x) {
                    let x0 = f32(source_x);
                    let weight_x = max(min(source_max.x, x0 + 1.0) - max(source_min.x, x0), 0.0);
                    let weight = weight_x * weight_y;
                    avg_color += weight * sample_by_pixel(vec2<i32>(source_x, source_y));
                    total_weight += weight;
                }
            }
        }
    }

    return avg_color / max(total_weight, 0.000001);
}

// END AREA SAMPLER REGION

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(t, s, in.uv);
    if (uniforms.sampler_mode == 1u) {
        color = sample_area(in.position);
    }
    return vec4(color.rgb, 1.0);
}
)"sv;

wgpu::TextureFormat to_linear(wgpu::TextureFormat format) {
  if (format == wgpu::TextureFormat::RGBA8UnormSrgb) {
    return wgpu::TextureFormat::RGBA8Unorm;
  }
  if (format == wgpu::TextureFormat::BGRA8UnormSrgb) {
    return wgpu::TextureFormat::BGRA8Unorm;
  }
  return format;
}

wgpu::TextureFormat best_surface_format() {
  if (g_surfaceCapabilities.formatCount == 0) {
    return wgpu::TextureFormat::Undefined;
  }
  for (size_t i = 0; i < g_surfaceCapabilities.formatCount; ++i) {
    const auto format = to_linear(g_surfaceCapabilities.formats[i]);
    if (format == wgpu::TextureFormat::RGBA8Unorm || format == wgpu::TextureFormat::BGRA8Unorm) {
      return format;
    }
  }
  return g_surfaceCapabilities.formats[0];
}

uint32_t sampler_mode(AuroraSampler sampler) noexcept {
  switch (sampler) {
  case SAMPLER_AREA:
    return 1;
  case SAMPLER_BILINEAR:
  default:
    return 0;
  }
}

uint32_t viewport_extent(float value) noexcept {
  return std::max(1u, static_cast<uint32_t>(std::lround(std::max(value, 1.f))));
}

} // namespace

bool vsync_enabled() noexcept { return g_vsyncEnabled.load(std::memory_order_acquire); }

wgpu::PresentMode select_present_mode(const wgpu::SurfaceCapabilities& capabilities) noexcept {
  const auto supports = [&capabilities](const wgpu::PresentMode candidate) {
    for (size_t i = 0; i < capabilities.presentModeCount; ++i) {
      if (capabilities.presentModes[i] == candidate) {
        return true;
      }
    }
    return false;
  };
  if (vsync_enabled()) {
    if (supports(wgpu::PresentMode::FifoRelaxed)) {
      return wgpu::PresentMode::FifoRelaxed;
    }
  } else {
    // Dawn only disables CAMetalLayer displaySyncEnabled for Immediate on Metal
    if (g_backendType != wgpu::BackendType::Metal && supports(wgpu::PresentMode::Mailbox)) {
      return wgpu::PresentMode::Mailbox;
    }
    if (supports(wgpu::PresentMode::Immediate)) {
      return wgpu::PresentMode::Immediate;
    }
  }
  return wgpu::PresentMode::Fifo;
}

TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled) {
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const auto format = g_graphicsConfig.surfaceConfiguration.format;
  uint32_t sampleCount = 1;
  if (multisampled) {
    sampleCount = g_graphicsConfig.msaaSamples;
  }
  if (width == 0 || height == 0) {
    Log.fatal("Invalid render texture size! {}x{}, multisampled {}, format {}", width, height,
              static_cast<uint32_t>(format), multisampled);
  }
  const wgpu::TextureDescriptor textureDescriptor{
      .label = "Render texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
               wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = sampleCount,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  constexpr wgpu::TextureViewDescriptor viewDescriptor{
      .label = "Render texture view",
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto view = texture.CreateView(&viewDescriptor);

  constexpr wgpu::SamplerDescriptor samplerDescriptor{
      .label = "Render sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.f,
      .lodMaxClamp = 1000.f,
      .maxAnisotropy = 1,
  };
  auto sampler = g_device.CreateSampler(&samplerDescriptor);

  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = size,
      .format = format,
      .sampler = std::move(sampler),
  };
}

const TextureWithSampler& present_source() noexcept {
  return g_graphicsConfig.msaaSamples > 1 ? g_frameBufferResolved : g_frameBuffer;
}

void set_resampler(AuroraSampler sampler) noexcept {
  switch (sampler) {
  case SAMPLER_AREA:
  case SAMPLER_BILINEAR:
    g_Resampler = sampler;
    return;
  default:
    g_Resampler = SAMPLER_BILINEAR;
    return;
  }
}

AuroraSampler get_resampler() noexcept { return g_Resampler; }

Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept {
  if (surface_width == 0 || surface_height == 0 || content_width == 0 || content_height == 0) {
    return {};
  }

  uint32_t viewport_width = surface_width;
  uint32_t viewport_height = std::min<uint32_t>(
      surface_height, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_width) *
                                                                               static_cast<double>(content_height) /
                                                                               static_cast<double>(content_width)))));
  if (viewport_height == surface_height) {
    viewport_width = std::min<uint32_t>(
        surface_width, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_height) *
                                                                                static_cast<double>(content_width) /
                                                                                static_cast<double>(content_height)))));
  }

  return {
      .left = static_cast<float>((surface_width - viewport_width) / 2),
      .top = static_cast<float>((surface_height - viewport_height) / 2),
      .width = static_cast<float>(viewport_width),
      .height = static_cast<float>(viewport_height),
      .znear = 0.f,
      .zfar = 1.f,
  };
}

static TextureWithSampler create_depth_texture(uint32_t width, uint32_t height) {
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const auto format = g_graphicsConfig.depthFormat;
  const wgpu::TextureDescriptor textureDescriptor{
      .label = "Depth texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = g_graphicsConfig.msaaSamples,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  const wgpu::TextureViewDescriptor viewDescriptor{
      .label = "Depth texture view",
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto view = texture.CreateView(&viewDescriptor);

  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "Depth sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.f,
      .lodMaxClamp = 1000.f,
      .maxAnisotropy = 1,
  };
  auto sampler = g_device.CreateSampler(&samplerDescriptor);

  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = size,
      .format = format,
      .sampler = std::move(sampler),
  };
}

static TextureWithSampler create_normal_texture(uint32_t width, uint32_t height) {
  const wgpu::Extent3D size{width, height, 1};
  const wgpu::TextureDescriptor desc{
      .label = "Scene normals",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
      .size = size,
      .format = NormalBufferFormat,
  };
  auto texture = g_device.CreateTexture(&desc);
  auto view = texture.CreateView();
  return {.texture = std::move(texture), .view = std::move(view), .size = size, .format = NormalBufferFormat};
}

static TextureWithSampler create_debug_id_texture(uint32_t width, uint32_t height) {
  const wgpu::Extent3D size{width, height, 1};
  const wgpu::TextureDescriptor desc{
      .label = "Scene debug ids",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding,
      .size = size,
      .format = DebugIdBufferFormat,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&desc);
  auto view = texture.CreateView();
  return {.texture = std::move(texture), .view = std::move(view), .size = size, .format = DebugIdBufferFormat};
}

bool enable_normal_buffer() {
  if (g_graphicsConfig.normalBuffer) {
    return true;
  }
  if (!g_hasCoreFeatures || g_graphicsConfig.msaaSamples != 1 || !g_device || g_frameBuffer.size.width == 0 ||
      g_frameBuffer.size.height == 0) {
    return false;
  }
  g_normalBuffer = create_normal_texture(g_frameBuffer.size.width, g_frameBuffer.size.height);
  g_graphicsConfig.normalBuffer = true;
  return true;
}

bool set_debug_id_buffer(bool enabled) {
  if (g_graphicsConfig.debugIdBuffer == enabled) {
    return true;
  }
  if (enabled && (!g_hasCoreFeatures || g_graphicsConfig.msaaSamples != 1 || !g_device ||
                  g_frameBuffer.size.width == 0 || g_frameBuffer.size.height == 0)) {
    return false;
  }
  // The render worker resolves gbuf.debug_id from this global, so it must be idle before the swap
  gfx::gpu_synchronize();
  g_debugIdBuffer =
      enabled ? create_debug_id_texture(g_frameBuffer.size.width, g_frameBuffer.size.height) : TextureWithSampler{};
  g_graphicsConfig.debugIdBuffer = enabled;
  return true;
}

void create_copy_pipeline() {
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = R"""(
@group(0) @binding(0)
var efb_sampler: sampler;
@group(0) @binding(1)
var efb_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, 1.0),
    vec2(-1.0, -3.0),
    vec2(3.0, 1.0),
);
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(0.0, 0.0),
    vec2(0.0, 2.0),
    vec2(2.0, 0.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vtxIdx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[vtxIdx], 0.0, 1.0);
    out.uv = uvs[vtxIdx];
    return out;
}

@fragment
fn fs_opaque(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(efb_texture, efb_sampler, in.uv);
    return vec4(color.rgb, 1.0);
}

@fragment
fn fs_premultiplied_alpha(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(efb_texture, efb_sampler, in.uv);
}
)""";
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "XFB Copy Module",
  };
  auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler =
              wgpu::SamplerBindingLayout{
                  .type = wgpu::SamplerBindingType::Filtering,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  g_CopyBindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_CopyBindGroupLayout,
  };
  auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  const auto make_copy_pipeline = [&](const char* label, const char* fragmentEntryPoint,
                                      const wgpu::BlendState* blend) {
    const std::array colorTargets{wgpu::ColorTargetState{
        .format = g_graphicsConfig.surfaceConfiguration.format,
        .blend = blend,
        .writeMask = wgpu::ColorWriteMask::All,
    }};
    const wgpu::FragmentState fragmentState{
        .module = module,
        .entryPoint = fragmentEntryPoint,
        .targetCount = colorTargets.size(),
        .targets = colorTargets.data(),
    };
    const wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .label = label,
        .layout = pipelineLayout,
        .vertex =
            wgpu::VertexState{
                .module = module,
                .entryPoint = "vs_main",
            },
        .primitive =
            wgpu::PrimitiveState{
                .topology = wgpu::PrimitiveTopology::TriangleList,
            },
        .multisample =
            wgpu::MultisampleState{
                .count = 1,
                .mask = UINT32_MAX,
            },
        .fragment = &fragmentState,
    };
    return g_device.CreateRenderPipeline(&pipelineDescriptor);
  };
  g_CopyPipeline = make_copy_pipeline("XFB Copy Pipeline", "fs_opaque", nullptr);

  const wgpu::BlendState premultipliedAlphaBlend{
      .color =
          {
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::One,
              .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha,
          },
      .alpha =
          {
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::One,
              .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha,
          },
  };
  g_CopyPremultipliedAlphaPipeline =
      make_copy_pipeline("XFB Premultiplied Alpha Copy Pipeline", "fs_premultiplied_alpha", &premultipliedAlphaBlend);
}

void create_resample_pipeline() {
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = resampleShaderSource;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "Present Resample Module",
  };
  auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array colorTargets{wgpu::ColorTargetState{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .writeMask = wgpu::ColorWriteMask::All,
  }};
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler =
              wgpu::SamplerBindingLayout{
                  .type = wgpu::SamplerBindingType::Filtering,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  g_ResampleBindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_ResampleBindGroupLayout,
  };
  auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "Present Resample Pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
          },
      .multisample =
          wgpu::MultisampleState{
              .count = 1,
              .mask = UINT32_MAX,
          },
      .fragment = &fragmentState,
  };
  g_ResamplePipeline = g_device.CreateRenderPipeline(&pipelineDescriptor);

  const wgpu::BufferDescriptor uniformBufferDescriptor{
      .label = "Present Resample Uniform Buffer",
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
      .size = AURORA_ALIGN(sizeof(ResampleUniformBlock), 16),
  };
  g_ResampleUniformBuffer = g_device.CreateBuffer(&uniformBufferDescriptor);
}

wgpu::BindGroup create_copy_bind_group(const TextureWithSampler& source) {
  const std::array bindGroupEntries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .sampler = source.sampler,
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .textureView = source.view,
      },
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_CopyBindGroupLayout,
      .entryCount = bindGroupEntries.size(),
      .entries = bindGroupEntries.data(),
  };
  return g_device.CreateBindGroup(&bindGroupDescriptor);
}

const TextureWithSampler& resample_present_source(const wgpu::CommandEncoder& encoder, const Viewport& viewport) {
  const auto& source = present_source();
  const uint32_t width = viewport_extent(viewport.width);
  const uint32_t height = viewport_extent(viewport.height);
  if (!g_resampledFrameBuffer.view || g_resampledFrameBuffer.size.width != width ||
      g_resampledFrameBuffer.size.height != height || g_resampledFrameBuffer.format != source.format) {
    g_resampledFrameBuffer = create_render_texture(width, height, false);
  }

  const ResampleUniformBlock uniform{
      .samplerMode = sampler_mode(g_Resampler),
      .frameWidth = static_cast<float>(width),
      .frameHeight = static_cast<float>(height),
  };
  AURORA_ASSERT(gfx::render_worker::is_worker_thread(), "Present resample queue write must run on the render worker");
  g_queue.WriteBuffer(g_ResampleUniformBuffer, 0, &uniform, sizeof(uniform));

  const std::array bindGroupEntries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .buffer = g_ResampleUniformBuffer,
          .size = AURORA_ALIGN(sizeof(ResampleUniformBlock), 16),
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .sampler = source.sampler,
      },
      wgpu::BindGroupEntry{
          .binding = 2,
          .textureView = source.view,
      },
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_ResampleBindGroupLayout,
      .entryCount = bindGroupEntries.size(),
      .entries = bindGroupEntries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);

  const std::array attachments{
      wgpu::RenderPassColorAttachment{
          .view = g_resampledFrameBuffer.view,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
      },
  };
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = "Present resample render pass",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = gpu_prof::pass_writes("Present resample"),
  };
  const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
  pass.SetPipeline(g_ResamplePipeline);
  pass.SetBindGroup(0, bindGroup, 0, nullptr);
  pass.SetViewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f);
  pass.Draw(3);
  pass.End();

  return g_resampledFrameBuffer;
}

static wgpu::BackendType to_wgpu_backend(AuroraBackend backend) {
  switch (backend) {
  case BACKEND_WEBGPU:
    return wgpu::BackendType::WebGPU;
  case BACKEND_D3D11:
    return wgpu::BackendType::D3D11;
  case BACKEND_D3D12:
    return wgpu::BackendType::D3D12;
  case BACKEND_METAL:
    return wgpu::BackendType::Metal;
  case BACKEND_VULKAN:
    return wgpu::BackendType::Vulkan;
  case BACKEND_OPENGL:
    return wgpu::BackendType::OpenGL;
  case BACKEND_OPENGLES:
    return wgpu::BackendType::OpenGLES;
  default:
    return wgpu::BackendType::Null;
  }
}

static void release_surface_locked() noexcept { g_surface = {}; }

static bool create_surface() {
  SDL_Window* window = window::get_sdl_window();
  if (window == nullptr) {
    Log.error("Failed to create surface: no window");
    return false;
  }
  window::SurfaceLock surfaceLock;
  release_surface_locked();
  g_surface = create_window_surface(g_instance, window, "Surface");
  if (!g_surface) {
    Log.error("Failed to create surface");
    return false;
  }
  return true;
}

bool initialize(AuroraBackend auroraBackend, bool allowCpu) {
  if (!g_instance) {
    Log.info("Creating WebGPU instance");
    const std::array requiredInstanceFeatures{
        wgpu::InstanceFeatureName::TimedWaitAny,
    };
    wgpu::InstanceDescriptor instanceDescriptor{
        .requiredFeatureCount = requiredInstanceFeatures.size(),
        .requiredFeatures = requiredInstanceFeatures.data(),
    };
#ifdef WEBGPU_DAWN
    constexpr std::array instanceToggles{
        "allow_unsafe_apis",
    };
    wgpu::DawnTogglesDescriptor instanceTogglesDescriptor{wgpu::DawnTogglesDescriptor::Init{
        .enabledToggleCount = instanceToggles.size(),
        .enabledToggles = instanceToggles.data(),
    }};
    dawn::native::DawnInstanceDescriptor dawnInstanceDescriptor{};
    dawnInstanceDescriptor.nextInChain = &instanceTogglesDescriptor;
#ifdef AURORA_GFX_DEBUG_GROUPS
    dawnInstanceDescriptor.backendValidationLevel = auroraBackend == BACKEND_VULKAN
                                                        ? dawn::native::BackendValidationLevel::Partial
                                                        : dawn::native::BackendValidationLevel::Disabled;
#else
    dawnInstanceDescriptor.backendValidationLevel = dawn::native::BackendValidationLevel::Disabled;
#endif
    dawnInstanceDescriptor.SetLoggingCallback(wgpu_log);
#ifdef TRACY_ENABLE
    dawnInstanceDescriptor.platform = tracy_dawn_platform();
#endif
    instanceDescriptor.nextInChain = &dawnInstanceDescriptor;
#endif
    g_instance = wgpu::CreateInstance(&instanceDescriptor);
    if (!g_instance) {
      Log.error("Failed to create WebGPU instance");
      return false;
    }
  }
  const wgpu::BackendType backend = to_wgpu_backend(auroraBackend);
  Log.info("Attempting to initialize {}", magic_enum::enum_name(backend));
#if 0
  // D3D12's debug layer is very slow
  g_dawnInstance->EnableBackendValidation(backend != WGPUBackendType::D3D12);
#endif

  if (!create_surface()) {
    return false;
  }
  {
    const wgpu::RequestAdapterOptions options{
        .featureLevel = wgpu::FeatureLevel::Compatibility,
        .powerPreference = wgpu::PowerPreference::HighPerformance,
        .backendType = backend,
        .compatibleSurface = g_surface,
    };
    Log.info("Requesting adapter\n  Feature level: {}\n  Power preference: {}\n  Backend: {}\n  Compatible surface: {}",
             magic_enum::enum_name(options.featureLevel), magic_enum::enum_name(options.powerPreference),
             magic_enum::enum_name(options.backendType), static_cast<bool>(options.compatibleSurface));
    bool requestAdapterCallbackCompleted = false;
    wgpu::RequestAdapterStatus requestAdapterStatus = wgpu::RequestAdapterStatus::CallbackCancelled;
    std::string requestAdapterMessage;
    const auto future = g_instance.RequestAdapter(
        &options, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
          requestAdapterCallbackCompleted = true;
          requestAdapterStatus = status;
          requestAdapterMessage = std::string{std::string_view{message}};
          if (status == wgpu::RequestAdapterStatus::Success) {
            g_adapter = std::move(adapter);
          } else {
            Log.warn("Adapter request failed: {}: {}", magic_enum::enum_name(status), message);
          }
        });
    const auto status = g_instance.WaitAny(future, 5000000000);
    if (status != wgpu::WaitStatus::Success) {
      if (requestAdapterCallbackCompleted) {
        Log.error("Failed to create adapter: wait status {}, request status {}, message: {}",
                  magic_enum::enum_name(status), magic_enum::enum_name(requestAdapterStatus), requestAdapterMessage);
      } else {
        Log.error("Failed to create adapter: wait status {}, request callback did not complete",
                  magic_enum::enum_name(status));
      }
      return false;
    }
    if (!g_adapter) {
      if (requestAdapterCallbackCompleted) {
        Log.error("Failed to create adapter: request status {}, message: {}",
                  magic_enum::enum_name(requestAdapterStatus), requestAdapterMessage);
      } else {
        Log.error("Failed to create adapter: request callback did not complete");
      }
      return false;
    }
  }
  g_adapter.GetInfo(&g_adapterInfo);
  auto adapterName = g_adapterInfo.device;
  if (adapterName.IsUndefined()) {
    adapterName = wgpu::StringView("Unknown");
  }
  if (!allowCpu && g_adapterInfo.adapterType == wgpu::AdapterType::CPU && backend != wgpu::BackendType::Null) {
    Log.warn("Ignoring CPU adapter: {}", adapterName);
    g_adapterInfo = {};
    g_adapter = {};
    return false;
  }
  auto description = g_adapterInfo.description;
  if (description.IsUndefined()) {
    description = wgpu::StringView("Unknown");
  }
  g_backendType = g_adapterInfo.backendType;
  const auto backendName = magic_enum::enum_name(g_backendType);
  Log.info("Graphics adapter information\n  API: {}\n  Device: {} ({})\n  Driver: {}", backendName, adapterName,
           magic_enum::enum_name(g_adapterInfo.adapterType), description);

  const char* profileEnv = SDL_getenv(FeatureProfileEnvVar);
  const auto resolvedProfile = resolve_feature_profile(g_config.featureProfile, profileEnv);
  if (resolvedProfile.source == FeatureProfileSource::InvalidEnvironment) {
    Log.warn("Ignoring {}={}; expected full, compat, metal or metal32", FeatureProfileEnvVar, profileEnv);
  }
  g_featureProfile = resolvedProfile.profile;

  render::Caps deviceCaps;
  {
    wgpu::Limits supportedLimits{};
    g_adapter.GetLimits(&supportedLimits);
    wgpu::SupportedFeatures supportedFeatures;
    g_adapter.GetFeatures(&supportedFeatures);
    const std::span<const wgpu::FeatureName> adapterFeatures{supportedFeatures.features,
                                                             supportedFeatures.featureCount};
    const bool requestCore = requests_core_features(adapterFeatures, g_featureProfile);
    wgpu::CompatibilityModeLimits compatibilityModeLimits{wgpu::CompatibilityModeLimits::Init{
        .maxStorageBuffersInVertexStage = 2,
        .maxStorageBuffersInFragmentStage = 2,
    }};
    wgpu::Limits requiredLimits =
        build_required_limits(supportedLimits, requestCore, g_featureProfile, sizeof(gx::DrawImmediateData));
    requiredLimits.nextInChain = &compatibilityModeLimits;
    log_required_limits(requiredLimits, requestCore);
    const auto requiredFeatures = select_device_features(adapterFeatures, g_featureProfile);
    const auto requested = [&](wgpu::FeatureName feature) {
      return std::ranges::find(requiredFeatures, feature) != requiredFeatures.end();
    };
    g_hasCoreFeatures = requested(wgpu::FeatureName::CoreFeaturesAndLimits);
    g_bcTexturesSupported = requested(wgpu::FeatureName::TextureCompressionBC);
    g_astcTexturesSupported = requested(wgpu::FeatureName::TextureCompressionASTC);
    g_textureComponentSwizzleSupported = requested(wgpu::FeatureName::TextureComponentSwizzle);
    g_dualSourceBlendingSupported = requested(wgpu::FeatureName::DualSourceBlending);
    std::string featureList;
    for (auto featureName : requiredFeatures) {
      featureList += "\n  ";
      featureList += requested_feature_name(featureName);
    }
    Log.info("Enabling features: {}", featureList);
#ifdef WEBGPU_DAWN
    wgpu::DawnCacheDeviceDescriptor cacheDescriptor({.nextInChain = nullptr});
    cacheDescriptor.SetDawnLoadCacheDataCallback(
        [](std::span<const std::byte> key, std::span<std::byte> value) noexcept -> size_t {
          return load_from_cache(key.data(), key.size(), value.data(), value.size(), nullptr);
        });
    cacheDescriptor.SetDawnStoreCacheDataCallback(
        [](std::span<const std::byte> key, std::span<const std::byte> value) noexcept {
          store_to_cache(key.data(), key.size(), value.data(), value.size(), nullptr);
        });

    constexpr std::array enableToggles{
#if _WIN32
        "use_dxc",
#ifndef NDEBUG
        "emit_hlsl_debug_symbols",
#endif
#endif
#ifdef NDEBUG
        "skip_validation",
        "disable_robustness",
#endif
#ifndef ANDROID
        "use_user_defined_labels_in_backend",
#endif
        "disable_symbol_renaming",
        "enable_immediate_error_handling",
        "gl_allow_context_on_multi_threads",
    };
    constexpr std::array disableToggles{
        "timestamp_quantization",
        // Adreno strikes again!
        // https://github.com/TwilitRealm/dusklight/issues/2563
        "use_spirv_reconvergence_mode",
    };
    std::vector<const char*> enabledToggles;
    enabledToggles.reserve(enableToggles.size());
    for (const char* toggle : enableToggles) {
      // Profiles exist to surface validation errors, so they keep validation on in release builds
      if (feature_profile_forces_validation(g_featureProfile) && std::string_view{toggle} == "skip_validation") {
        continue;
      }
      enabledToggles.push_back(toggle);
    }
    wgpu::DawnTogglesDescriptor togglesDescriptor(wgpu::DawnTogglesDescriptor::Init{
        .nextInChain = &cacheDescriptor,
        .enabledToggleCount = enabledToggles.size(),
        .enabledToggles = enabledToggles.data(),
        .disabledToggleCount = disableToggles.size(),
        .disabledToggles = disableToggles.data(),
    });
#endif
    wgpu::DeviceDescriptor deviceDescriptor({
#ifdef WEBGPU_DAWN
        .nextInChain = &togglesDescriptor,
#endif
        .requiredFeatureCount = requiredFeatures.size(),
        .requiredFeatures = requiredFeatures.data(),
        .requiredLimits = &requiredLimits,
    });
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device& device, wgpu::ErrorType type, wgpu::StringView message) {
          if (g_initialized) {
            FATAL("WebGPU error {}: {}", underlying(type), message);
          } else {
            Log.warn("WebGPU error {}: {}", underlying(type), message);
          }
        });
    deviceDescriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device& device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
          if (g_initialized) {
            FATAL("Device lost: {}", message);
          } else {
            Log.warn("Device lost: {}", message);
          }
        });
    const auto future =
        g_adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::WaitAnyOnly,
                                [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                                  if (status == wgpu::RequestDeviceStatus::Success) {
                                    g_device = std::move(device);
                                  } else {
                                    Log.warn("Device request failed: {}", message);
                                  }
                                });
    const auto status = g_instance.WaitAny(future, 5000000000);
    if (status != wgpu::WaitStatus::Success) {
      Log.error("Failed to create device: {}", magic_enum::enum_name(status));
      return false;
    }
    if (!g_device) {
      return false;
    }
    g_device.SetLoggingCallback(wgpu_log);
    log_feature_profile(resolvedProfile, supportedFeatures, supportedLimits);
    deviceCaps = query_device_caps(adapterFeatures);
  }
  g_queue = g_device.GetQueue();

  const wgpu::Status status = g_surface.GetCapabilities(g_adapter, &g_surfaceCapabilities);
  if (status != wgpu::Status::Success) {
    Log.error("Failed to get surface capabilities: {}", magic_enum::enum_name(status));
    return false;
  }
  if (g_surfaceCapabilities.formatCount == 0) {
    Log.error("Surface has no formats");
    return false;
  }
  if (g_surfaceCapabilities.presentModeCount == 0) {
    Log.error("Surface has no present modes");
    return false;
  }
  auto surfaceFormat = best_surface_format();
  g_vsyncEnabled.store(g_config.vsync, std::memory_order_release);
  auto presentMode = select_present_mode(g_surfaceCapabilities);
  Log.info("Using surface format {}, present mode {}", magic_enum::enum_name(surfaceFormat),
           magic_enum::enum_name(presentMode));
  const auto size = window::get_window_size();
  g_graphicsConfig = GraphicsConfig{
      .surfaceConfiguration =
          wgpu::SurfaceConfiguration{
              .format = surfaceFormat,
              .usage = wgpu::TextureUsage::RenderAttachment,
              .width = size.native_fb_width,
              .height = size.native_fb_height,
              .presentMode = presentMode,
          },
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .msaaSamples = g_config.msaa,
      .textureAnisotropy = g_config.maxTextureAnisotropy,
  };
  create_copy_pipeline();
  create_resample_pipeline();
  gpu_prof::initialize();
  resize_swapchain(size.fb_width, size.fb_height, size.native_fb_width, size.native_fb_height, true);
  set_caps(std::move(deviceCaps));
  g_initialized = true;
  return true;
}

void shutdown() {
  g_initialized = false;
  gfx::gpu_synchronize();
  gpu_prof::shutdown();
  g_CopyBindGroupLayout = {};
  g_CopyPipeline = {};
  g_CopyPremultipliedAlphaPipeline = {};
  g_CopyBindGroup = {};
  g_ResampleBindGroupLayout = {};
  g_ResamplePipeline = {};
  g_ResampleUniformBuffer = {};
  g_resampledFrameBuffer = {};
  g_frameBuffer = {};
  g_frameBufferResolved = {};
  g_depthBuffer = {};
  g_normalBuffer = {};
  g_graphicsConfig.normalBuffer = false;
  g_debugIdBuffer = {};
  g_graphicsConfig.debugIdBuffer = false;
  g_queue = {};
  g_surface = {};
  g_device = {};
  reset_caps();
  g_adapter = {};
  g_instance = {};
  cache_shutdown();
}

void release_surface() noexcept {
  gfx::gpu_synchronize();
  {
    window::SurfaceLock surfaceLock;
    release_surface_locked();
  }
}

static void resize_swapchain_internal(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight,
                                      bool force) {
  if (!g_surface || !g_device || width == 0 || height == 0 || nativeHeight == 0 || nativeWidth == 0) {
    return;
  }
  const bool sizeChanged = g_graphicsConfig.surfaceConfiguration.width != nativeWidth ||
                           g_graphicsConfig.surfaceConfiguration.height != nativeHeight ||
                           g_frameBuffer.size.width != width || g_frameBuffer.size.height != height;
  if (!force && !sizeChanged) {
    return;
  }
  g_graphicsConfig.surfaceConfiguration.width = nativeWidth;
  g_graphicsConfig.surfaceConfiguration.height = nativeHeight;
  auto surfaceConfiguration = g_graphicsConfig.surfaceConfiguration;
  surfaceConfiguration.device = g_device;
  {
    window::SurfaceLock surfaceLock;
    g_surface.Configure(&surfaceConfiguration);
  }
  g_frameBuffer = create_render_texture(width, height, true);
  g_frameBufferResolved = create_render_texture(width, height, false);
  g_depthBuffer = create_depth_texture(width, height);
  if (g_graphicsConfig.normalBuffer) {
    g_normalBuffer = create_normal_texture(width, height);
  }
  if (g_graphicsConfig.debugIdBuffer) {
    g_debugIdBuffer = create_debug_id_texture(width, height);
  }
  g_CopyBindGroup = create_copy_bind_group(present_source());
}

bool refresh_surface(bool recreate) {
  gfx::gpu_synchronize();
  if (!g_instance || !g_device) {
    return false;
  }
  if (!window::is_presentable()) {
    {
      window::SurfaceLock surfaceLock;
      release_surface_locked();
    }
    return false;
  }
  if ((!g_surface || recreate) && !create_surface()) {
    return false;
  }
  uint32_t width = g_graphicsConfig.surfaceConfiguration.width;
  uint32_t height = g_graphicsConfig.surfaceConfiguration.height;
  uint32_t nativeWidth = width;
  uint32_t nativeHeight = height;
  if (window::get_sdl_window() != nullptr) {
    const auto size = window::get_window_size();
    width = size.fb_width;
    height = size.fb_height;
    nativeWidth = size.native_fb_width;
    nativeHeight = size.native_fb_height;
  }
  if (width != 0 && height != 0) {
    resize_swapchain_internal(width, height, nativeWidth, nativeHeight, true);
  }
  return true;
}

void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force) {
  gfx::gpu_synchronize();
  resize_swapchain_internal(width, height, nativeWidth, nativeHeight, force);
}
} // namespace aurora::webgpu

void aurora_enable_vsync(const bool enabled) {
  aurora::webgpu::g_vsyncEnabled.store(enabled, std::memory_order_release);
  aurora::webgpu::g_graphicsConfig.surfaceConfiguration.presentMode =
      aurora::webgpu::select_present_mode(aurora::webgpu::g_surfaceCapabilities);
  aurora::window::push_custom_event(aurora::window::CustomEvent::RefreshSurface);
}

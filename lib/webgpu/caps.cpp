#include "caps.hpp"

#include "feature_profile.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace aurora::webgpu {
namespace {
constexpr std::array RequestedFeatureTable{
    RequestedFeature{wgpu::FeatureName::CoreFeaturesAndLimits, "CoreFeaturesAndLimits"},
    RequestedFeature{wgpu::FeatureName::TextureCompressionBC, "TextureCompressionBC"},
    RequestedFeature{wgpu::FeatureName::TextureCompressionASTC, "TextureCompressionASTC"},
    RequestedFeature{wgpu::FeatureName::DualSourceBlending, "DualSourceBlending"},
    RequestedFeature{wgpu::FeatureName::TextureComponentSwizzle, "TextureComponentSwizzle"},
    RequestedFeature{wgpu::FeatureName::TimestampQuery, "TimestampQuery"},
    RequestedFeature{wgpu::FeatureName::Subgroups, "Subgroups"},
    RequestedFeature{wgpu::FeatureName::ShaderF16, "ShaderF16"},
    RequestedFeature{wgpu::FeatureName::MultiDrawIndirect, "MultiDrawIndirect"},
    RequestedFeature{wgpu::FeatureName::IndirectFirstInstance, "IndirectFirstInstance"},
    RequestedFeature{wgpu::FeatureName::PrimitiveIndex, "PrimitiveIndex"},
    RequestedFeature{wgpu::FeatureName::AtomicVec2uMinMax, "AtomicVec2uMinMax"},
    RequestedFeature{wgpu::FeatureName::ClipDistances, "ClipDistances"},
    RequestedFeature{wgpu::FeatureName::TextureFormatsTier1, "TextureFormatsTier1"},
    RequestedFeature{wgpu::FeatureName::TextureFormatsTier2, "TextureFormatsTier2"},
    RequestedFeature{wgpu::FeatureName::Float32Filterable, "Float32Filterable"},
    RequestedFeature{wgpu::FeatureName::RG11B10UfloatRenderable, "RG11B10UfloatRenderable"},
    RequestedFeature{wgpu::FeatureName::AdapterPropertiesMemoryHeaps, "AdapterPropertiesMemoryHeaps"},
    RequestedFeature{wgpu::FeatureName::AdapterPropertiesD3D, "AdapterPropertiesD3D"},
    RequestedFeature{wgpu::FeatureName::SharedBufferMemoryD3D12Resource, "SharedBufferMemoryD3D12Resource"},
    RequestedFeature{wgpu::FeatureName::SharedFenceDXGISharedHandle, "SharedFenceDXGISharedHandle"},
};

// Written on the main thread inside initialize, before any other thread reads it
render::Caps g_caps;

bool contains(std::span<const wgpu::FeatureName> features, wgpu::FeatureName feature) noexcept {
  return std::find(features.begin(), features.end(), feature) != features.end();
}

uint32_t best(uint32_t adapter) noexcept { return adapter == 0 ? WGPU_LIMIT_U32_UNDEFINED : adapter; }

uint32_t up_to(uint32_t adapter, uint32_t target) noexcept {
  return adapter == 0 || adapter == WGPU_LIMIT_U32_UNDEFINED ? WGPU_LIMIT_U32_UNDEFINED : std::min(adapter, target);
}

uint64_t up_to(uint64_t adapter, uint64_t target) noexcept {
  return adapter == 0 || adapter == WGPU_LIMIT_U64_UNDEFINED ? WGPU_LIMIT_U64_UNDEFINED : std::min(adapter, target);
}

render::DeviceLimits device_limits(const wgpu::Limits& limits, const wgpu::CompatibilityModeLimits& compat) noexcept {
  return {
      .maxTextureDimension2D = limits.maxTextureDimension2D,
      .maxTextureDimension3D = limits.maxTextureDimension3D,
      .maxTextureArrayLayers = limits.maxTextureArrayLayers,
      .maxBindGroups = limits.maxBindGroups,
      .maxBindingsPerBindGroup = limits.maxBindingsPerBindGroup,
      .maxDynamicUniformBuffersPerPipelineLayout = limits.maxDynamicUniformBuffersPerPipelineLayout,
      .maxDynamicStorageBuffersPerPipelineLayout = limits.maxDynamicStorageBuffersPerPipelineLayout,
      .maxSampledTexturesPerShaderStage = limits.maxSampledTexturesPerShaderStage,
      .maxSamplersPerShaderStage = limits.maxSamplersPerShaderStage,
      .maxStorageBuffersPerShaderStage = limits.maxStorageBuffersPerShaderStage,
      .maxStorageTexturesPerShaderStage = limits.maxStorageTexturesPerShaderStage,
      .maxStorageBuffersInVertexStage = compat.maxStorageBuffersInVertexStage,
      .maxStorageTexturesInVertexStage = compat.maxStorageTexturesInVertexStage,
      .maxStorageBuffersInFragmentStage = compat.maxStorageBuffersInFragmentStage,
      .maxStorageTexturesInFragmentStage = compat.maxStorageTexturesInFragmentStage,
      .maxUniformBuffersPerShaderStage = limits.maxUniformBuffersPerShaderStage,
      .maxUniformBufferBindingSize = limits.maxUniformBufferBindingSize,
      .maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize,
      .maxBufferSize = limits.maxBufferSize,
      .minUniformBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment,
      .minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment,
      .maxVertexBuffers = limits.maxVertexBuffers,
      .maxVertexAttributes = limits.maxVertexAttributes,
      .maxInterStageShaderVariables = limits.maxInterStageShaderVariables,
      .maxColorAttachments = limits.maxColorAttachments,
      .maxColorAttachmentBytesPerSample = limits.maxColorAttachmentBytesPerSample,
      .maxComputeWorkgroupStorageSize = limits.maxComputeWorkgroupStorageSize,
      .maxComputeInvocationsPerWorkgroup = limits.maxComputeInvocationsPerWorkgroup,
      .maxComputeWorkgroupSizeX = limits.maxComputeWorkgroupSizeX,
      .maxComputeWorkgroupSizeY = limits.maxComputeWorkgroupSizeY,
      .maxComputeWorkgroupSizeZ = limits.maxComputeWorkgroupSizeZ,
      .maxComputeWorkgroupsPerDimension = limits.maxComputeWorkgroupsPerDimension,
      .maxImmediateSize = limits.maxImmediateSize,
  };
}
} // namespace

std::span<const RequestedFeature> requested_features() noexcept { return RequestedFeatureTable; }

std::string_view requested_feature_name(wgpu::FeatureName feature) noexcept {
  for (const auto& entry : RequestedFeatureTable) {
    if (entry.feature == feature) {
      return entry.name;
    }
  }
  return {};
}

bool requests_core_features(std::span<const wgpu::FeatureName> adapterFeatures, AuroraFeatureProfile profile) noexcept {
  return contains(adapterFeatures, wgpu::FeatureName::CoreFeaturesAndLimits) &&
         feature_profile_allows(profile, wgpu::FeatureName::CoreFeaturesAndLimits);
}

std::vector<wgpu::FeatureName> select_device_features(std::span<const wgpu::FeatureName> adapterFeatures,
                                                      AuroraFeatureProfile profile) {
  std::vector<wgpu::FeatureName> features;
  for (const auto& entry : RequestedFeatureTable) {
    if (contains(adapterFeatures, entry.feature) && feature_profile_allows(profile, entry.feature)) {
      features.push_back(entry.feature);
    }
  }
  return features;
}

std::vector<FeatureRequestResult> describe_feature_requests(std::span<const wgpu::FeatureName> adapterFeatures,
                                                            std::span<const wgpu::FeatureName> deviceFeatures,
                                                            AuroraFeatureProfile profile) {
  std::vector<FeatureRequestResult> results;
  results.reserve(RequestedFeatureTable.size());
  for (const auto& entry : RequestedFeatureTable) {
    auto state = FeatureRequestState::Unsupported;
    if (contains(deviceFeatures, entry.feature)) {
      state = FeatureRequestState::Enabled;
    } else if (contains(adapterFeatures, entry.feature) && !feature_profile_allows(profile, entry.feature)) {
      state = FeatureRequestState::MaskedByProfile;
    }
    results.push_back({entry.feature, entry.name, state});
  }
  return results;
}

wgpu::Limits build_required_limits(const wgpu::Limits& adapter, bool core, AuroraFeatureProfile profile,
                                   uint32_t immediateSize) noexcept {
  wgpu::Limits limits{
      .maxTextureDimension1D = best(adapter.maxTextureDimension1D),
      .maxTextureDimension2D = best(adapter.maxTextureDimension2D),
      .maxTextureDimension3D = best(adapter.maxTextureDimension3D),
      .maxTextureArrayLayers = best(adapter.maxTextureArrayLayers),
      .maxStorageBuffersPerShaderStage = LegacyStorageBuffersPerShaderStage,
      .minUniformBufferOffsetAlignment =
          std::max(adapter.minUniformBufferOffsetAlignment, MinUniformBufferOffsetAlignment),
      .minStorageBufferOffsetAlignment =
          std::max(adapter.minStorageBufferOffsetAlignment, MinStorageBufferOffsetAlignment),
      .maxImmediateSize = immediateSize,
  };
  if (core) {
    limits.maxStorageBuffersPerShaderStage =
        up_to(adapter.maxStorageBuffersPerShaderStage, CoreStorageBuffersPerShaderStage);
    limits.maxColorAttachments = up_to(adapter.maxColorAttachments, CoreColorAttachments);
    limits.maxColorAttachmentBytesPerSample =
        up_to(adapter.maxColorAttachmentBytesPerSample, CoreColorAttachmentBytesPerSample);
    limits.maxInterStageShaderVariables = best(adapter.maxInterStageShaderVariables);
    limits.maxStorageTexturesPerShaderStage =
        up_to(adapter.maxStorageTexturesPerShaderStage, CoreStorageTexturesPerShaderStage);
    limits.maxSampledTexturesPerShaderStage =
        up_to(adapter.maxSampledTexturesPerShaderStage, CoreSampledTexturesPerShaderStage);
    limits.maxBufferSize = up_to(adapter.maxBufferSize, CoreMaxBufferSize);
    limits.maxStorageBufferBindingSize = up_to(adapter.maxStorageBufferBindingSize, CoreStorageBufferBindingSize);
    // A binding can never be larger than the buffer that holds it
    if (limits.maxBufferSize != WGPU_LIMIT_U64_UNDEFINED &&
        limits.maxStorageBufferBindingSize != WGPU_LIMIT_U64_UNDEFINED) {
      limits.maxStorageBufferBindingSize = std::min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    }
    limits.maxComputeWorkgroupStorageSize =
        up_to(adapter.maxComputeWorkgroupStorageSize, CoreComputeWorkgroupStorageSize);
  }
  clamp_limits_to_feature_profile(profile, limits);
  return limits;
}

VramInfo vram_from_heaps(std::span<const wgpu::MemoryHeapInfo> heaps, wgpu::AdapterType adapterType) noexcept {
  const bool unified = adapterType == wgpu::AdapterType::IntegratedGPU || adapterType == wgpu::AdapterType::CPU;
  uint64_t deviceLocal = 0;
  uint64_t largest = 0;
  for (const auto& heap : heaps) {
    largest = std::max(largest, heap.size);
    if (heap.properties & wgpu::HeapProperty::DeviceLocal) {
      deviceLocal = std::max(deviceLocal, heap.size);
    }
  }
  if (deviceLocal == 0) {
    return {};
  }
  return {
      // Integrated Vulkan drivers expose only the BIOS carve-out as device local; shared memory is the real budget
      .bytes = unified ? largest : deviceLocal,
      .source = render::VramSource::MemoryHeaps,
      .unified = unified,
  };
}

render::AdapterKind adapter_kind(wgpu::AdapterType type) noexcept {
  switch (type) {
  case wgpu::AdapterType::DiscreteGPU:
    return render::AdapterKind::Discrete;
  case wgpu::AdapterType::IntegratedGPU:
    return render::AdapterKind::Integrated;
  case wgpu::AdapterType::CPU:
    return render::AdapterKind::Cpu;
  default:
    return render::AdapterKind::Unknown;
  }
}

AuroraBackend aurora_backend(wgpu::BackendType type) noexcept {
  switch (type) {
  case wgpu::BackendType::D3D11:
    return BACKEND_D3D11;
  case wgpu::BackendType::D3D12:
    return BACKEND_D3D12;
  case wgpu::BackendType::Metal:
    return BACKEND_METAL;
  case wgpu::BackendType::Vulkan:
    return BACKEND_VULKAN;
  case wgpu::BackendType::OpenGL:
    return BACKEND_OPENGL;
  case wgpu::BackendType::OpenGLES:
    return BACKEND_OPENGLES;
  case wgpu::BackendType::WebGPU:
    return BACKEND_WEBGPU;
  case wgpu::BackendType::Null:
    return BACKEND_NULL;
  default:
    return BACKEND_AUTO;
  }
}

render::Caps build_caps(const CapsSource& source) {
  const auto has = [&](wgpu::FeatureName feature) { return contains(source.features, feature); };
  const auto backend = aurora_backend(source.backend);
  const bool subgroups = has(wgpu::FeatureName::Subgroups);
  const bool sharedBuffer = has(wgpu::FeatureName::SharedBufferMemoryD3D12Resource);
  const bool sharedFence = has(wgpu::FeatureName::SharedFenceDXGISharedHandle);
  const auto vram = vram_from_heaps(source.memoryHeaps, source.adapterType);
  return {
      .valid = true,
      .backend = backend,
      .profile = source.profile,
      .adapterKind = adapter_kind(source.adapterType),
      .vendorId = source.vendorId,
      .deviceId = source.deviceId,
      .adapterName = std::string{source.adapterName},
      .core = has(wgpu::FeatureName::CoreFeaturesAndLimits),
      .bcTextures = has(wgpu::FeatureName::TextureCompressionBC),
      .astcTextures = has(wgpu::FeatureName::TextureCompressionASTC),
      .textureComponentSwizzle = has(wgpu::FeatureName::TextureComponentSwizzle),
      .dualSourceBlending = has(wgpu::FeatureName::DualSourceBlending),
      .timestampQuery = has(wgpu::FeatureName::TimestampQuery),
      .subgroups = subgroups,
      .subgroupMinSize = subgroups ? source.subgroupMinSize : 0,
      .subgroupMaxSize = subgroups ? source.subgroupMaxSize : 0,
      .shaderF16 = has(wgpu::FeatureName::ShaderF16),
      .multiDrawIndirect = has(wgpu::FeatureName::MultiDrawIndirect),
      .indirectFirstInstance = has(wgpu::FeatureName::IndirectFirstInstance),
      .primitiveIndex = has(wgpu::FeatureName::PrimitiveIndex),
      .atomic64 = has(wgpu::FeatureName::AtomicVec2uMinMax),
      .clipDistances = has(wgpu::FeatureName::ClipDistances),
      .textureFormatsTier1 = has(wgpu::FeatureName::TextureFormatsTier1),
      .textureFormatsTier2 = has(wgpu::FeatureName::TextureFormatsTier2),
      .float32Filterable = has(wgpu::FeatureName::Float32Filterable),
      .rg11b10Renderable = has(wgpu::FeatureName::RG11B10UfloatRenderable),
      .d3d12SharedBuffer = sharedBuffer,
      .d3d12SharedFence = sharedFence,
      // The Null backend reports every feature, so interop also needs a real D3D12 device
      .d3d12Interop = backend == BACKEND_D3D12 && sharedBuffer && sharedFence,
      .nativeD3D12 = false,
      .d3dShaderModel = source.d3dShaderModel,
      .vramBytes = vram.bytes,
      .vramSource = vram.source,
      .unifiedMemory = vram.unified,
      .limits = device_limits(source.limits, source.compatLimits),
  };
}

std::string shader_model_text(uint32_t shaderModel) {
  if (shaderModel == 0) {
    return "n/a";
  }
  // Dawn encodes SM 6.10 and later as 60 + minor, so 70 is 6.10 rather than 7.0
  if (shaderModel >= 70) {
    return "6." + std::to_string(shaderModel - 60);
  }
  return std::to_string(shaderModel / 10) + "." + std::to_string(shaderModel % 10);
}

void set_caps(render::Caps caps) noexcept { g_caps = std::move(caps); }

void reset_caps() noexcept { g_caps = {}; }
} // namespace aurora::webgpu

namespace aurora::render {
const Caps& caps() noexcept { return webgpu::g_caps; }

StageBindingLimits stage_binding_limits(const Caps& caps, ShaderStage stage) noexcept {
  const auto& limits = caps.limits;
  StageBindingLimits result{
      .storageBuffers = limits.maxStorageBuffersPerShaderStage,
      .storageTextures = limits.maxStorageTexturesPerShaderStage,
      .sampledTextures = limits.maxSampledTexturesPerShaderStage,
      .samplers = limits.maxSamplersPerShaderStage,
      .uniformBuffers = limits.maxUniformBuffersPerShaderStage,
  };
  switch (stage) {
  case ShaderStage::Vertex:
    result.storageBuffers = limits.maxStorageBuffersInVertexStage;
    result.storageTextures = limits.maxStorageTexturesInVertexStage;
    break;
  case ShaderStage::Fragment:
    result.storageBuffers = limits.maxStorageBuffersInFragmentStage;
    result.storageTextures = limits.maxStorageTexturesInFragmentStage;
    break;
  case ShaderStage::Compute:
    break;
  }
  return result;
}

std::string_view adapter_kind_name(AdapterKind kind) noexcept {
  switch (kind) {
  case AdapterKind::Unknown:
    return "unknown";
  case AdapterKind::Discrete:
    return "discrete";
  case AdapterKind::Integrated:
    return "integrated";
  case AdapterKind::Cpu:
    return "cpu";
  }
  return "unknown";
}

std::string_view vram_source_name(VramSource source) noexcept {
  switch (source) {
  case VramSource::Unknown:
    return "unknown";
  case VramSource::MemoryHeaps:
    return "memory heaps";
  case VramSource::Dxgi:
    return "dxgi";
  }
  return "unknown";
}
} // namespace aurora::render

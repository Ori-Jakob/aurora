#pragma once

#include <aurora/aurora.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace aurora::render {
enum class AdapterKind : uint8_t { Unknown, Discrete, Integrated, Cpu };
enum class VramSource : uint8_t { Unknown, MemoryHeaps, Dxgi };
enum class ShaderStage : uint8_t { Vertex, Fragment, Compute };

struct DeviceLimits {
  uint32_t maxTextureDimension2D = 0;
  uint32_t maxTextureDimension3D = 0;
  uint32_t maxTextureArrayLayers = 0;
  uint32_t maxBindGroups = 0;
  uint32_t maxBindingsPerBindGroup = 0;
  uint32_t maxDynamicUniformBuffersPerPipelineLayout = 0;
  uint32_t maxDynamicStorageBuffersPerPipelineLayout = 0;
  uint32_t maxSampledTexturesPerShaderStage = 0;
  uint32_t maxSamplersPerShaderStage = 0;
  uint32_t maxStorageBuffersPerShaderStage = 0;
  uint32_t maxStorageTexturesPerShaderStage = 0;
  uint32_t maxStorageBuffersInVertexStage = 0;
  uint32_t maxStorageTexturesInVertexStage = 0;
  uint32_t maxStorageBuffersInFragmentStage = 0;
  uint32_t maxStorageTexturesInFragmentStage = 0;
  uint32_t maxUniformBuffersPerShaderStage = 0;
  uint64_t maxUniformBufferBindingSize = 0;
  uint64_t maxStorageBufferBindingSize = 0;
  uint64_t maxBufferSize = 0;
  uint32_t minUniformBufferOffsetAlignment = 0;
  uint32_t minStorageBufferOffsetAlignment = 0;
  uint32_t maxVertexBuffers = 0;
  uint32_t maxVertexAttributes = 0;
  uint32_t maxInterStageShaderVariables = 0;
  uint32_t maxColorAttachments = 0;
  uint32_t maxColorAttachmentBytesPerSample = 0;
  uint32_t maxComputeWorkgroupStorageSize = 0;
  uint32_t maxComputeInvocationsPerWorkgroup = 0;
  uint32_t maxComputeWorkgroupSizeX = 0;
  uint32_t maxComputeWorkgroupSizeY = 0;
  uint32_t maxComputeWorkgroupSizeZ = 0;
  uint32_t maxComputeWorkgroupsPerDimension = 0;
  uint32_t maxImmediateSize = 0;

  bool operator==(const DeviceLimits&) const = default;
};

struct StageBindingLimits {
  uint32_t storageBuffers = 0;
  uint32_t storageTextures = 0;
  uint32_t sampledTextures = 0;
  uint32_t samplers = 0;
  uint32_t uniformBuffers = 0;
};

struct Caps {
  bool valid = false;
  AuroraBackend backend = BACKEND_AUTO;
  AuroraFeatureProfile profile = FEATURE_PROFILE_FULL;
  AdapterKind adapterKind = AdapterKind::Unknown;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  std::string adapterName;
  bool core = false;
  bool bcTextures = false;
  bool astcTextures = false;
  bool textureComponentSwizzle = false;
  bool dualSourceBlending = false;
  bool timestampQuery = false;
  bool subgroups = false;
  uint32_t subgroupMinSize = 0;
  uint32_t subgroupMaxSize = 0;
  bool shaderF16 = false;
  bool multiDrawIndirect = false;
  bool indirectFirstInstance = false;
  bool primitiveIndex = false;
  bool atomic64 = false;
  bool clipDistances = false;
  bool textureFormatsTier1 = false;
  bool textureFormatsTier2 = false;
  bool float32Filterable = false;
  bool rg11b10Renderable = false;
  bool d3d12SharedBuffer = false;
  bool d3d12SharedFence = false;
  bool d3d12Interop = false;
  bool nativeD3D12 = false;
  uint32_t d3dShaderModel = 0;
  uint64_t vramBytes = 0;
  VramSource vramSource = VramSource::Unknown;
  bool unifiedMemory = false;
  DeviceLimits limits;

  bool operator==(const Caps&) const = default;
};

const Caps& caps() noexcept;
StageBindingLimits stage_binding_limits(const Caps& caps, ShaderStage stage) noexcept;
std::string_view adapter_kind_name(AdapterKind kind) noexcept;
std::string_view vram_source_name(VramSource source) noexcept;
} // namespace aurora::render

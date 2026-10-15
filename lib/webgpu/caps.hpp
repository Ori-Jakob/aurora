#pragma once

#include <aurora/aurora.h>
#include <aurora/render_caps.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::webgpu {
inline constexpr uint32_t LegacyStorageBuffersPerShaderStage = 2;
inline constexpr uint32_t MinUniformBufferOffsetAlignment = 64;
inline constexpr uint32_t MinStorageBufferOffsetAlignment = 16;
inline constexpr uint32_t CoreColorAttachments = 8;
inline constexpr uint32_t CoreColorAttachmentBytesPerSample = 128;
inline constexpr uint32_t CoreStorageBuffersPerShaderStage = 10;
inline constexpr uint32_t CoreStorageTexturesPerShaderStage = 8;
inline constexpr uint32_t CoreSampledTexturesPerShaderStage = 32;
inline constexpr uint64_t CoreMaxBufferSize = uint64_t{1} << 30;
inline constexpr uint64_t CoreStorageBufferBindingSize = uint64_t{1} << 30;
inline constexpr uint32_t CoreComputeWorkgroupStorageSize = 32768;

struct RequestedFeature {
  wgpu::FeatureName feature;
  std::string_view name;
};

enum class FeatureRequestState : uint8_t { Enabled, MaskedByProfile, Unsupported };

struct FeatureRequestResult {
  wgpu::FeatureName feature;
  std::string_view name;
  FeatureRequestState state;
};

struct VramInfo {
  uint64_t bytes = 0;
  render::VramSource source = render::VramSource::Unknown;
  bool unified = false;
};

struct CapsSource {
  wgpu::BackendType backend = wgpu::BackendType::Undefined;
  wgpu::AdapterType adapterType = wgpu::AdapterType::Unknown;
  AuroraFeatureProfile profile = FEATURE_PROFILE_FULL;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  std::string_view adapterName;
  std::span<const wgpu::FeatureName> features;
  wgpu::Limits limits;
  wgpu::CompatibilityModeLimits compatLimits;
  uint32_t subgroupMinSize = 0;
  uint32_t subgroupMaxSize = 0;
  std::span<const wgpu::MemoryHeapInfo> memoryHeaps;
  uint32_t d3dShaderModel = 0;
};

std::span<const RequestedFeature> requested_features() noexcept;
std::string_view requested_feature_name(wgpu::FeatureName feature) noexcept;
bool requests_core_features(std::span<const wgpu::FeatureName> adapterFeatures, AuroraFeatureProfile profile) noexcept;
std::vector<wgpu::FeatureName> select_device_features(std::span<const wgpu::FeatureName> adapterFeatures,
                                                      AuroraFeatureProfile profile);
std::vector<FeatureRequestResult> describe_feature_requests(std::span<const wgpu::FeatureName> adapterFeatures,
                                                            std::span<const wgpu::FeatureName> deviceFeatures,
                                                            AuroraFeatureProfile profile);
wgpu::Limits build_required_limits(const wgpu::Limits& adapter, bool core, AuroraFeatureProfile profile,
                                   uint32_t immediateSize) noexcept;
VramInfo vram_from_heaps(std::span<const wgpu::MemoryHeapInfo> heaps, wgpu::AdapterType adapterType) noexcept;
render::AdapterKind adapter_kind(wgpu::AdapterType type) noexcept;
AuroraBackend aurora_backend(wgpu::BackendType type) noexcept;
render::Caps build_caps(const CapsSource& source);
std::string shader_model_text(uint32_t shaderModel);
void set_caps(render::Caps caps) noexcept;
void reset_caps() noexcept;
} // namespace aurora::webgpu

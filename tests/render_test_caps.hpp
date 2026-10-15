#pragma once

#include "render/settings.hpp"

#include <aurora/render.hpp>

#include <cstddef>
#include <cstdint>
#include <ostream>

namespace aurora::render {
inline void PrintTo(const FeatureSupport& support, std::ostream* os) {
  *os << '{' << (support.supported ? "true" : "false") << ", " << reason_name(support.reason) << '}';
}
} // namespace aurora::render

namespace aurora::render::test {
inline constexpr uint64_t MiB = uint64_t{1} << 20;
inline constexpr uint64_t GiB = uint64_t{1} << 30;

inline Caps full_caps() {
  Caps result;
  result.valid = true;
  result.backend = BACKEND_D3D12;
  result.profile = FEATURE_PROFILE_FULL;
  result.adapterKind = AdapterKind::Discrete;
  result.adapterName = "Test adapter";
  result.core = true;
  result.bcTextures = true;
  result.astcTextures = true;
  result.textureComponentSwizzle = true;
  result.dualSourceBlending = true;
  result.timestampQuery = true;
  result.subgroups = true;
  result.shaderF16 = true;
  result.multiDrawIndirect = true;
  result.indirectFirstInstance = true;
  result.primitiveIndex = true;
  result.atomic64 = true;
  result.clipDistances = true;
  result.textureFormatsTier1 = true;
  result.textureFormatsTier2 = true;
  result.float32Filterable = true;
  result.rg11b10Renderable = true;
  result.d3d12SharedBuffer = true;
  result.d3d12SharedFence = true;
  result.d3d12Interop = true;
  result.nativeD3D12 = false;
  result.vramBytes = 8 * GiB;
  result.vramSource = VramSource::MemoryHeaps;
  result.unifiedMemory = false;

  auto& limits = result.limits;
  limits.maxColorAttachments = 8;
  limits.maxColorAttachmentBytesPerSample = 128;
  limits.maxInterStageShaderVariables = 30;
  limits.maxBindGroups = 4;
  limits.maxStorageBuffersPerShaderStage = 10;
  limits.maxStorageBuffersInVertexStage = 10;
  limits.maxStorageBuffersInFragmentStage = 10;
  limits.maxStorageTexturesPerShaderStage = 8;
  limits.maxStorageTexturesInVertexStage = 8;
  limits.maxStorageTexturesInFragmentStage = 8;
  limits.maxSampledTexturesPerShaderStage = 32;
  limits.maxSamplersPerShaderStage = 16;
  limits.maxUniformBuffersPerShaderStage = 12;
  limits.maxBufferSize = GiB;
  limits.maxStorageBufferBindingSize = GiB;
  limits.maxComputeWorkgroupStorageSize = 32768;
  limits.maxComputeInvocationsPerWorkgroup = 256;
  limits.maxImmediateSize = 64;
  return result;
}

inline Caps compat_caps() {
  Caps result = full_caps();
  result.core = false;
  result.profile = FEATURE_PROFILE_COMPAT;
  auto& limits = result.limits;
  limits.maxColorAttachments = 4;
  limits.maxColorAttachmentBytesPerSample = 32;
  limits.maxInterStageShaderVariables = 15;
  limits.maxStorageBuffersPerShaderStage = 8;
  limits.maxStorageBuffersInVertexStage = 2;
  limits.maxStorageBuffersInFragmentStage = 4;
  limits.maxStorageTexturesPerShaderStage = 4;
  limits.maxStorageTexturesInVertexStage = 0;
  limits.maxStorageTexturesInFragmentStage = 4;
  limits.maxSampledTexturesPerShaderStage = 16;
  limits.maxBufferSize = 256 * MiB;
  limits.maxStorageBufferBindingSize = 128 * MiB;
  return result;
}

inline Caps metal32_caps() {
  Caps result = full_caps();
  result.profile = FEATURE_PROFILE_METAL32;
  result.limits.maxColorAttachmentBytesPerSample = 32;
  result.multiDrawIndirect = false;
  result.primitiveIndex = false;
  result.atomic64 = false;
  result.d3d12SharedBuffer = false;
  result.d3d12SharedFence = false;
  result.d3d12Interop = false;
  return result;
}

inline detail::BuiltOptions built_all() {
  detail::BuiltOptions built;
  built.fill(~0u);
  return built;
}

inline detail::BuiltOptions built_none() { return {}; }

inline void enable_all_built() {
  for (size_t i = 0; i < FeatureCount; ++i) {
    detail::testing::set_built_options(static_cast<Feature>(i), ~0u);
  }
}

inline detail::SupportEnv env(const Caps& caps, const detail::BuiltOptions& built, uint32_t msaaSamples = 1) {
  return {caps, msaaSamples, built, true};
}

inline Settings everything_on() {
  Settings s;
  s.preset = Preset::Ultra;
  s.textureMaterials = true;
  s.stochasticTiling = true;
  s.legacyMatMaps = true;
  s.pbr = true;
  s.taa = true;
  s.ao = Quality::Ultra;
  s.ssil = true;
  s.shadows.quality = Quality::Ultra;
  s.shadows.game = GameShadows::Remove;
  s.shadows.keepCues = true;
  s.gi = GiMode::Ddgi;
  s.rt = RtMode::PathTraceReference;
  s.rtBackend = RtBackend::Dxr;
  s.reflections = Reflections::SsrRt;
  s.fog.volumetric = Quality::Ultra;
  s.fog.godRays = GodRays::Volumetric;
  s.fog.legacyMatch = true;
  s.fog.translucents = true;
  s.sky.clouds = Quality::Ultra;
  s.sky.cloudShadows = true;
  s.sky.keepCloudDome = true;
  s.grass.quality = Quality::Ultra;
  s.grass.replaceGamePatches = true;
  s.grass.terrainGrass = true;
  s.grass.shadows = GrassShadows::Full;
  s.grass.interaction = true;
  s.grass.legacyShadowTransfer = true;
  s.terrain.mode = TerrainDetail::Tessellated;
  s.terrain.quality = Quality::Ultra;
  s.terrain.flattenVolumes = true;
  s.gpuFoliage = true;
  s.gpuWeather = true;
  s.gpuParticles = true;
  s.gpuDeform = true;
  s.meshReplacement = true;
  s.staticWorld = StaticWorld::On;
  return s;
}
} // namespace aurora::render::test

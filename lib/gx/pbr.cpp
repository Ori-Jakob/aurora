#include "pbr.hpp"

#include "../gfx/dds_io.hpp"
#include "../gfx/texture_replacement.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static aurora::Module PbrLog("aurora::gx::pbr");

namespace aurora::gx {
using webgpu::g_device;
using webgpu::g_queue;

bool enablePbrMaterialOverride = false;
bool pbrMaterialOverrideActive = false;
Vec4<float> pbrParams{0.30f, 0.04f, 0.20f, 0.0f};
Vec4<float> pbrScales{1.0f, 1.0f, static_cast<float>(AURORA_PBR_DEBUG_OFF), 0.0f};
Vec4<float> pbrNormalParams{1.0f, 1.0f, 1.0f, 0.0f};
Vec4<float> pbrAmbientGradient{1.15f, 0.45f, 0.80f, 1.0f};
Vec4<float> pbrIndirectOcclusion{0.35f, 0.45f, 0.60f, 0.0f};
Vec4<float> pbrDynamicGiParams{0.0f, 0.15f, 0.35f, 0.35f};
Vec4<float> pbrIblParams{1.0f, 1.0f, 1.0f, 5.0f};
Vec4<float> pbrIblBlendParams{1.0f, 5.0f, 0.0f, 0.0f};
Vec4<float> pbrFillDir{0.39f, -0.44f, -0.81f, 0.0f};
Vec4<float> pbrMaterialFactors{0.5f, 0.0f, 1.0f, 0.5f};
Vec4<float> pbrMaterialEmissive{0.0f, 0.0f, 0.0f, 0.0f};
bool pbrEnhancedLightsEnabled = false;
bool pbrEnhancedLightsDebugEnabled = false;
AuroraPbrEnhancedLightFalloff pbrEnhancedLightFalloff = AURORA_PBR_ENHANCED_LIGHT_FALLOFF_LEGACY_RADIUS;
u32 pbrEnhancedLightMaxCount = 4;
u32 pbrEnhancedLightCount = 0;
u32 pbrSubmittedSceneLightCount = 0;
bool pbrSceneLightsApiBacked = false;
float pbrEnhancedLightIntensityScale = 1.0f;
u32 pbrEnhancedLightStorageOffset = 0;
u32 pbrEnhancedLightStorageSize = 0;
std::array<PbrEnhancedLight, PbrMaxEnhancedLights> pbrEnhancedLights{};
gfx::TextureHandle pbrMaterialRmaos;
gfx::TextureHandle pbrMaterialRoughness;
gfx::TextureHandle pbrMaterialMetallic;
gfx::TextureHandle pbrMaterialAo;
gfx::TextureHandle pbrMaterialSpecular;
gfx::TextureHandle pbrMaterialNormal;
gfx::TextureHandle pbrMaterialEmissiveMap;

namespace pbr_internal {
constexpr u32 PbrIrradianceCubeSize = 16;
constexpr u32 PbrPrefilterCubeSize = 64;
constexpr u32 PbrPrefilterMipCount = 6;
constexpr u32 PbrRuntimeProbeCubeSize = 32;
constexpr u32 PbrRuntimeProbePrefilterMipCount = 6;
constexpr u32 PbrProbeCaptureFramesPerFace = 12;
constexpr u32 PbrProbePeriodicRefreshFrames = 900;
constexpr float PbrProbeDirtyDistance = 48.0f;
constexpr float PbrProbeDirtyRotationSum = 0.08f;
constexpr u32 PbrBrdfLutSize = 128;
constexpr u32 PbrShadowMapMinSize = 256;
constexpr u32 PbrShadowMapMaxSize = 4096;
constexpr u32 PbrShadowMapDefaultSize = 1024;
constexpr u32 PbrShadowAtlasSlotCount = 4;
constexpr u32 PbrShadowRequestMaxCount = PbrShadowAtlasSlotCount;
constexpr float PbrShadowSourceRefreshDistance = 64.0f;
constexpr float PbrShadowTargetRefreshDistance = 512.0f;
constexpr float PbrShadowRadiusRefreshDistance = 128.0f;
constexpr float PbrShadowLocalProjectionFade = 0.10f;
constexpr float PbrShadowDirectionalProjectionFade = 0.02f;
constexpr u32 PbrShadowFailedCaptureRetryFrames = 30;
constexpr float PbrPi = 3.14159265358979323846f;
constexpr std::array<std::string_view, 6> PbrCubeFaceNames{"px", "nx", "py", "ny", "pz", "nz"};
constexpr u32 PbrProbeIrradianceFilterPassCount = 6;
constexpr u32 PbrProbePrefilterPassCount = 6 * PbrRuntimeProbePrefilterMipCount;
constexpr u32 PbrProbeFilterPassCount = PbrProbeIrradianceFilterPassCount + PbrProbePrefilterPassCount;
constexpr u32 PbrProbeCacheSlotCount = 4;

wgpu::Sampler sPbrMaterialSampler;
wgpu::Sampler sPbrIblSampler;
wgpu::Sampler sPbrBrdfLutSampler;

struct PbrIblTextureSet {
  wgpu::Texture irradianceCubeTexture;
  wgpu::TextureView irradianceCubeView;
  wgpu::Texture prefilterCubeTexture;
  wgpu::TextureView prefilterCubeView;
  wgpu::Texture brdfLutTexture;
  wgpu::TextureView brdfLutView;
  u32 prefilterMipCount = 1;
  bool available = false;
};

struct PbrProbeTargetViews {
  std::array<wgpu::TextureView, 6> irradianceFaceViews;
  std::array<std::array<wgpu::TextureView, 6>, PbrRuntimeProbePrefilterMipCount> prefilterFaceViews;
};

struct PbrProbeCamera {
  Mat3x4<float> sourceView;
  Mat3x4<float> sourceInvView;
  Mat4x4<float> projection;
  std::array<Mat3x4<float>, 6> viewFromSource;
  std::array<Mat3x4<float>, 6> normalFromSource;
  bool valid = false;
};

struct PbrProbeCacheSlot {
  PbrIblTextureSet ibl;
  PbrProbeTargetViews targetViews;
  PbrProbeCamera camera;
  std::string sceneKey;
  u32 lastUsedSerial = 0;
  bool resourcesReady = false;
};

PbrIblTextureSet sFallbackPbrIbl;
PbrIblTextureSet sAuthoredPbrIbl;
PbrIblTextureSet sProbePbrIbl;
PbrIblTextureSet sProbeCapturePbrIbl;
PbrIblTextureSet* sActivePbrIbl = &sFallbackPbrIbl;
AuroraPbrIblSource sRequestedPbrIblSource = AURORA_PBR_IBL_SOURCE_PROBE;
std::string sActivePbrIblSceneKey;
std::string sAuthoredPbrIblRootPath;
std::string sAuthoredPbrIblStage;
int sAuthoredPbrIblRoom = -1;
bool sAuthoredPbrIblGlobalLoaded = false;
bool sAuthoredPbrIblStageLoaded = false;
bool sAuthoredPbrIblRoomLoaded = false;
PbrProbeTargetViews sProbeTargetViews;
std::array<wgpu::TextureView, 6> sProbeCaptureFaceViews;
wgpu::Texture sProbeCaptureMsaaColorTexture;
wgpu::TextureView sProbeCaptureMsaaColorView;
wgpu::Texture sProbeCaptureDepthTexture;
wgpu::TextureView sProbeCaptureDepthView;
wgpu::BindGroupLayout sProbeFilterBindGroupLayout;
wgpu::RenderPipeline sProbeFilterPipeline;
std::array<wgpu::Buffer, PbrProbeFilterPassCount> sProbeFilterUniforms;
std::array<wgpu::BindGroup, PbrProbeFilterPassCount> sProbeFilterBindGroups;
wgpu::TextureFormat sProbeCaptureColorFormat = wgpu::TextureFormat::Undefined;
wgpu::TextureFormat sProbeCaptureDepthFormat = wgpu::TextureFormat::Undefined;
u32 sProbeCaptureMsaaSamples = 0;
u32 sProbeCaptureFace = 0;
u32 sProbeCaptureDelayFrames = 0;
bool sProbeCaptureResourcesReady = false;
bool sProbeFilterPending = false;

PbrProbeCamera sProbeCamera;
PbrProbeCamera sProbeCaptureCamera;
PbrProbeCamera sProbeLastCompletedCamera;
u32 sProbeFramesSinceRefresh = PbrProbePeriodicRefreshFrames;
bool sProbeRefreshPending = true;
bool sProbeAutoRefresh = false;
bool sProbeLocalGi = false;
bool sProbeSceneStale = false;
bool sProbeCacheEnabled = true;
bool sProbeCacheLastHit = false;
bool sProbeNearestCacheEnabled = true;
bool sProbeNearestCacheActive = false;
float sProbeNearestCacheDistance = 0.0f;
float sProbeNearestCacheMaxDistance = 6000.0f;
bool sProbeSpatialBlendEnabled = true;
bool sProbeSpatialBlendActive = false;
float sProbeSpatialBlendFactor = 1.0f;
float sProbeSpatialBlendDistance = 0.0f;
float sProbeSpatialBlendMaxDistance = 8000.0f;
bool sProbeBlendEnabled = true;
PbrIblTextureSet* sPbrIblBlendFrom = nullptr;
PbrIblTextureSet* sPbrIblSpatialFrom = nullptr;
u32 sProbeBlendFrames = 45;
u32 sProbeBlendFrame = 45;
float sProbeBlendFactor = 1.0f;
bool sProbeReplayPbrVisible = false;
u32 sProbeReplayDrawCount = 0;
u32 sProbeCacheSerial = 1;
std::array<PbrProbeCacheSlot, PbrProbeCacheSlotCount> sProbeCacheSlots;
PbrProbeCacheSlot* sActiveProbeCacheSlot = nullptr;
PbrProbeCacheSlot* sSpatialProbeCacheSlot = nullptr;
PbrProbeCacheSlot* sProbeCaptureCacheSlot = nullptr;
PbrProbeCacheSlot* sProbeFilterCacheSlot = nullptr;
u32 sPbrShadowMapSize = PbrShadowMapDefaultSize;
float sPbrShadowMapStrength = 1.0f;
float sPbrShadowMapBias = 0.002f;
bool sPbrShadowMapEnabled = false;
bool sPbrShadowMapAvailable = false;
AuroraPbrShadowLightRequest sPbrShadowLightRequest{};
std::array<AuroraPbrShadowLightRequest, PbrShadowRequestMaxCount> sPbrShadowLightRequests{};
u32 sPbrShadowLightRequestCount = 0;
wgpu::Texture sPbrShadowMapTexture;
wgpu::TextureView sPbrShadowMapView;
std::array<wgpu::TextureView, PbrShadowAtlasSlotCount> sPbrShadowMapSlotViews;
wgpu::Sampler sPbrShadowMapSampler;
wgpu::Texture sPbrShadowFallbackTexture;
wgpu::TextureView sPbrShadowFallbackView;
wgpu::TextureFormat sPbrShadowMapDepthFormat = wgpu::TextureFormat::Undefined;
wgpu::TextureFormat sPbrShadowFallbackDepthFormat = wgpu::TextureFormat::Undefined;
u32 sPbrShadowMapResourceSize = 0;
u32 sPbrShadowMapDrawCount = 0;
std::array<u32, PbrShadowAtlasSlotCount> sPbrShadowMapSlotDrawCounts{};
u32 sPbrShadowMapCapturedSlotCount = 0;
u32 sPbrShadowMapCapturedSlotMask = 0;
bool sPbrShadowMapResourcesReady = false;
bool sPbrShadowMapMatrixValid = false;
bool sPbrShadowCasterPassReady = false;
bool sPbrShadowReceiverSamplingReady = false;
bool sPbrShadowMapRefreshPending = false;
u32 sPbrShadowMapMaxActiveSlots = 2;
u32 sPbrShadowMapSlotsPerFrame = 1;
u32 sPbrShadowMapPendingSlotMask = 0;
u32 sPbrShadowFailedCaptureFrames = 0;
std::array<float, 16> sPbrShadowLightViewProjection{};
Mat3x4<float> sPbrShadowViewFromWorld{};
Mat4x4<float> sPbrShadowProjection{};
std::array<Mat3x4<float>, PbrShadowAtlasSlotCount> sPbrShadowSlotViewFromWorld{};
std::array<Mat4x4<float>, PbrShadowAtlasSlotCount> sPbrShadowSlotProjection{};
u32 sPbrShadowDescriptorStorageFrame = UINT32_MAX;
u32 sPbrShadowDescriptorStorageOffset = 0;
u32 sPbrShadowDescriptorStorageSize = 0;

struct PbrProbeFilterParams {
  u32 face = 0;
  u32 mode = 0;
  u32 mip = 0;
  u32 sampleCount = 0;
  float roughness = 0.0f;
  float sourceMip = 0.0f;
  float padding0 = 0.0f;
  float padding1 = 0.0f;
};

struct PbrVec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PbrShadowDescriptor {
  Vec4<float> viewFromSource0{};
  Vec4<float> viewFromSource1{};
  Vec4<float> viewFromSource2{};
  Vec4<float> projection0{};
  Vec4<float> projection1{};
  Vec4<float> projection2{};
  Vec4<float> projection3{};
  Vec4<float> params{};
};
static_assert(sizeof(PbrShadowDescriptor) == 128);

float pbr_saturate(float v) noexcept { return std::clamp(v, 0.0f, 1.0f); }

bool pbr_finite(float v) noexcept { return std::isfinite(v); }

float pbr_dot(PbrVec3 a, PbrVec3 b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }

PbrVec3 pbr_normalize(PbrVec3 v) noexcept {
  const float len = std::sqrt(std::max(pbr_dot(v, v), 1e-12f));
  return {v.x / len, v.y / len, v.z / len};
}

PbrVec3 pbr_cross(PbrVec3 a, PbrVec3 b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

PbrVec3 pbr_probe_eye(const PbrProbeCamera& camera) noexcept {
  return {camera.sourceInvView.m0[3], camera.sourceInvView.m1[3], camera.sourceInvView.m2[3]};
}

float pbr_distance_sq(PbrVec3 a, PbrVec3 b) noexcept {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

PbrVec3 pbr_vec3_from_array(const float (&values)[3]) noexcept { return {values[0], values[1], values[2]}; }

float pbr_length(PbrVec3 v) noexcept { return std::sqrt(std::max(pbr_dot(v, v), 0.0f)); }

void pbr_set_shadow_identity_matrix() noexcept {
  sPbrShadowLightViewProjection = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
  };
  sPbrShadowViewFromWorld = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };
  sPbrShadowProjection = Mat4x4_Identity;
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    sPbrShadowSlotViewFromWorld[slot] = sPbrShadowViewFromWorld;
    sPbrShadowSlotProjection[slot] = sPbrShadowProjection;
  }
  sPbrShadowDescriptorStorageFrame = UINT32_MAX;
}

std::array<float, 16> pbr_mul_row_major_mtx4(const std::array<float, 16>& lhs,
                                             const std::array<float, 16>& rhs) noexcept {
  std::array<float, 16> out{};
  for (u32 row = 0; row < 4; ++row) {
    for (u32 col = 0; col < 4; ++col) {
      out[row * 4 + col] = lhs[row * 4 + 0] * rhs[0 * 4 + col] +
                           lhs[row * 4 + 1] * rhs[1 * 4 + col] +
                           lhs[row * 4 + 2] * rhs[2 * 4 + col] +
                           lhs[row * 4 + 3] * rhs[3 * 4 + col];
    }
  }
  return out;
}

bool pbr_shadow_request_finite(const AuroraPbrShadowLightRequest& request) noexcept {
  for (float value : request.position) {
    if (!pbr_finite(value)) {
      return false;
    }
  }
  for (float value : request.target) {
    if (!pbr_finite(value)) {
      return false;
    }
  }
  return pbr_finite(request.radius);
}

bool pbr_shadow_request_same_light(const AuroraPbrShadowLightRequest& lhs,
                                   const AuroraPbrShadowLightRequest& rhs) noexcept {
  return lhs.valid && rhs.valid && lhs.stableId == rhs.stableId && lhs.source == rhs.source &&
         lhs.sourceIndex == rhs.sourceIndex && lhs.shadowType == rhs.shadowType;
}

u32 pbr_shadow_active_slot_limit() noexcept {
  return std::clamp<u32>(sPbrShadowMapMaxActiveSlots, 1u, PbrShadowAtlasSlotCount);
}

u32 pbr_shadow_slot_mask(u32 slotCount) noexcept {
  slotCount = std::min(slotCount, PbrShadowAtlasSlotCount);
  return slotCount == 0 ? 0 : ((1u << slotCount) - 1u);
}

u32 pbr_shadow_active_slot_count() noexcept {
  return std::min({sPbrShadowLightRequestCount, PbrShadowAtlasSlotCount, pbr_shadow_active_slot_limit()});
}

u32 pbr_shadow_active_slot_mask() noexcept {
  return pbr_shadow_slot_mask(pbr_shadow_active_slot_count());
}

u32 pbr_count_bits(u32 value) noexcept {
  u32 count = 0;
  while (value != 0) {
    count += value & 1u;
    value >>= 1u;
  }
  return count;
}

const AuroraPbrShadowLightRequest* pbr_active_shadow_request() noexcept {
  const u32 slotCount = pbr_shadow_active_slot_count();
  for (u32 i = 0; i < slotCount; ++i) {
    const AuroraPbrShadowLightRequest& request = sPbrShadowLightRequests[i];
    if (request.valid && request.shadowType != AURORA_PBR_SHADOW_TYPE_NONE &&
        request.shadowType != AURORA_PBR_SHADOW_TYPE_POINT && pbr_shadow_request_finite(request)) {
      return &request;
    }
  }
  return nullptr;
}

bool pbr_scene_light_matches_shadow_request(const AuroraSceneLight& light) noexcept {
  return sPbrShadowLightRequest.valid && light.stableId == sPbrShadowLightRequest.stableId &&
         light.source == sPbrShadowLightRequest.source && light.sourceIndex == sPbrShadowLightRequest.sourceIndex;
}

Vec4<float> pbr_scene_light_shadow_params(const AuroraSceneLight& light) noexcept {
  if (!sPbrShadowMapAvailable || sPbrShadowMapCapturedSlotMask == 0) {
    return {};
  }
  const u32 slotCount = pbr_shadow_active_slot_count();
  for (u32 i = 0; i < slotCount; ++i) {
    const AuroraPbrShadowLightRequest& request = sPbrShadowLightRequests[i];
    if (!request.valid || request.shadowType == AURORA_PBR_SHADOW_TYPE_NONE ||
        request.shadowType == AURORA_PBR_SHADOW_TYPE_POINT || request.atlasSlot >= PbrShadowAtlasSlotCount) {
      continue;
    }
    if ((sPbrShadowMapCapturedSlotMask & (1u << request.atlasSlot)) == 0) {
      continue;
    }
    if (light.stableId == request.stableId && light.source == request.source &&
        light.sourceIndex == request.sourceIndex) {
      return {1.0f, static_cast<float>(request.atlasSlot), 0.0f, 0.0f};
    }
  }
  return {};
}

float pbr_shadow_vec3_distance_sq(const float* a, const float* b) noexcept {
  const float dx = a[0] - b[0];
  const float dy = a[1] - b[1];
  const float dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

bool pbr_shadow_request_needs_refresh(const AuroraPbrShadowLightRequest& request) noexcept {
  if (!sPbrShadowLightRequest.valid) {
    return true;
  }
  if (request.stableId != sPbrShadowLightRequest.stableId || request.source != sPbrShadowLightRequest.source ||
      request.sourceIndex != sPbrShadowLightRequest.sourceIndex ||
      request.shadowType != sPbrShadowLightRequest.shadowType) {
    return true;
  }

  const float sourceThresholdSq = PbrShadowSourceRefreshDistance * PbrShadowSourceRefreshDistance;
  const float targetThresholdSq = PbrShadowTargetRefreshDistance * PbrShadowTargetRefreshDistance;
  if (pbr_shadow_vec3_distance_sq(request.position, sPbrShadowLightRequest.position) > sourceThresholdSq ||
      pbr_shadow_vec3_distance_sq(request.target, sPbrShadowLightRequest.target) > targetThresholdSq) {
    return true;
  }

  return std::abs(request.radius - sPbrShadowLightRequest.radius) > PbrShadowRadiusRefreshDistance;
}

bool pbr_shadow_request_needs_refresh(const AuroraPbrShadowLightRequest& request,
                                      const AuroraPbrShadowLightRequest& cached) noexcept {
  if (!cached.valid) {
    return true;
  }
  if (request.stableId != cached.stableId || request.source != cached.source ||
      request.sourceIndex != cached.sourceIndex || request.shadowType != cached.shadowType) {
    return true;
  }

  const float sourceThresholdSq = PbrShadowSourceRefreshDistance * PbrShadowSourceRefreshDistance;
  const float targetThresholdSq = PbrShadowTargetRefreshDistance * PbrShadowTargetRefreshDistance;
  if (pbr_shadow_vec3_distance_sq(request.position, cached.position) > sourceThresholdSq ||
      pbr_shadow_vec3_distance_sq(request.target, cached.target) > targetThresholdSq) {
    return true;
  }

  return std::abs(request.radius - cached.radius) > PbrShadowRadiusRefreshDistance;
}

bool pbr_shadow_request_list_needs_refresh(
    const std::array<AuroraPbrShadowLightRequest, PbrShadowRequestMaxCount>& requests, u32 requestCount) noexcept {
  const u32 activeRequestCount = std::min(requestCount, pbr_shadow_active_slot_limit());
  if (activeRequestCount != pbr_shadow_active_slot_count()) {
    return true;
  }
  for (u32 i = 0; i < activeRequestCount && i < PbrShadowRequestMaxCount; ++i) {
    if (pbr_shadow_request_needs_refresh(requests[i], sPbrShadowLightRequests[i])) {
      return true;
    }
  }
  return false;
}

bool ensure_pbr_shadow_sampler() noexcept {
  if (sPbrShadowMapSampler) {
    return true;
  }
  if (!g_device) {
    return false;
  }

  const wgpu::SamplerDescriptor samplerDesc{
      .label = "PBR shadow map comparison sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Nearest,
      .lodMinClamp = 0.0f,
      .lodMaxClamp = 1.0f,
      .compare = UseReversedZ ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual,
      .maxAnisotropy = 1,
  };
  sPbrShadowMapSampler = g_device.CreateSampler(&samplerDesc);
  return static_cast<bool>(sPbrShadowMapSampler);
}

bool ensure_pbr_shadow_fallback_resources() noexcept {
  if (!g_device) {
    return false;
  }

  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  if (depthFormat == wgpu::TextureFormat::Undefined) {
    return false;
  }

  if (sPbrShadowFallbackView && sPbrShadowFallbackDepthFormat == depthFormat && ensure_pbr_shadow_sampler()) {
    return true;
  }

  const wgpu::TextureDescriptor depthDesc{
      .label = "PBR fallback shadow depth texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = 1, .height = 1, .depthOrArrayLayers = 1},
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  sPbrShadowFallbackTexture = g_device.CreateTexture(&depthDesc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = "PBR fallback shadow depth view",
      .format = depthFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
      .arrayLayerCount = 1,
  };
  sPbrShadowFallbackView = sPbrShadowFallbackTexture.CreateView(&viewDesc);
  sPbrShadowFallbackDepthFormat = depthFormat;
  return static_cast<bool>(sPbrShadowFallbackView) && ensure_pbr_shadow_sampler();
}

bool pbr_build_shadow_matrix_from_request(const AuroraPbrShadowLightRequest& request) noexcept {
  if (!request.valid || !pbr_shadow_request_finite(request)) {
    pbr_set_shadow_identity_matrix();
    sPbrShadowMapMatrixValid = false;
    return false;
  }

  const PbrVec3 eye = pbr_vec3_from_array(request.position);
  const PbrVec3 target = pbr_vec3_from_array(request.target);
  PbrVec3 forward{target.x - eye.x, target.y - eye.y, target.z - eye.z};
  const float lightDistance = pbr_length(forward);
  if (lightDistance < 0.001f) {
    forward = {0.0f, -1.0f, 0.0f};
  }

  PbrVec3 up{0.0f, 1.0f, 0.0f};
  const PbrVec3 normalizedForward = pbr_normalize(forward);
  if (std::abs(pbr_dot(normalizedForward, up)) > 0.95f) {
    up = {0.0f, 0.0f, 1.0f};
  }

  const PbrVec3 f = normalizedForward;
  const PbrVec3 s = pbr_normalize(pbr_cross(f, up));
  const PbrVec3 u = pbr_cross(s, f);
  sPbrShadowViewFromWorld = {
      {s.x, s.y, s.z, -pbr_dot(s, eye)},
      {u.x, u.y, u.z, -pbr_dot(u, eye)},
      {-f.x, -f.y, -f.z, pbr_dot(f, eye)},
  };
  const std::array<float, 16> view{
      s.x, s.y, s.z, -pbr_dot(s, eye),
      u.x, u.y, u.z, -pbr_dot(u, eye),
      -f.x, -f.y, -f.z, pbr_dot(f, eye),
      0.0f, 0.0f, 0.0f, 1.0f,
  };

  const float requestedRadius = pbr_finite(request.radius) ? std::max(request.radius, 1.0f) : 1.0f;
  const float halfExtent = std::clamp(requestedRadius * 0.65f, 512.0f, 30000.0f);
  const float nearPlane = 1.0f;
  const float farPlane = std::clamp(std::max(lightDistance + requestedRadius, requestedRadius * 2.0f), 1024.0f,
                                    60000.0f);
  sPbrShadowProjection = {};
  sPbrShadowProjection.m0[0] = 1.0f / halfExtent;
  sPbrShadowProjection.m1[1] = 1.0f / halfExtent;
  sPbrShadowProjection.m2[2] = 1.0f / (nearPlane - farPlane);
  sPbrShadowProjection.m2[3] = nearPlane / (nearPlane - farPlane);
  sPbrShadowProjection.m3[3] = 1.0f;
  const std::array<float, 16> projection{
      1.0f / halfExtent, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f / halfExtent, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f / (nearPlane - farPlane), nearPlane / (nearPlane - farPlane),
      0.0f, 0.0f, 0.0f, 1.0f,
  };

  sPbrShadowLightViewProjection = pbr_mul_row_major_mtx4(projection, view);
  if (request.atlasSlot < PbrShadowAtlasSlotCount) {
    sPbrShadowSlotViewFromWorld[request.atlasSlot] = sPbrShadowViewFromWorld;
    sPbrShadowSlotProjection[request.atlasSlot] = sPbrShadowProjection;
  }
  sPbrShadowMapMatrixValid = true;
  sPbrShadowDescriptorStorageFrame = UINT32_MAX;
  return true;
}

bool ensure_pbr_shadow_map_resources() noexcept {
  if (!sPbrShadowMapEnabled || !g_device) {
    if (!sPbrShadowMapEnabled) {
      sPbrShadowMapResourcesReady = false;
    }
    return false;
  }

  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  if (depthFormat == wgpu::TextureFormat::Undefined) {
    sPbrShadowMapResourcesReady = false;
    return false;
  }

  ensure_pbr_shadow_sampler();

  bool slotViewsReady = true;
  for (const auto& view : sPbrShadowMapSlotViews) {
    slotViewsReady = slotViewsReady && static_cast<bool>(view);
  }
  if (sPbrShadowMapResourcesReady && sPbrShadowMapResourceSize == sPbrShadowMapSize &&
      sPbrShadowMapDepthFormat == depthFormat && sPbrShadowMapTexture && sPbrShadowMapView &&
      sPbrShadowMapSampler && slotViewsReady) {
    return true;
  }

  const wgpu::TextureDescriptor depthDesc{
      .label = "PBR shadow map depth texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = sPbrShadowMapSize, .height = sPbrShadowMapSize,
               .depthOrArrayLayers = PbrShadowAtlasSlotCount},
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  sPbrShadowMapTexture = g_device.CreateTexture(&depthDesc);
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    const wgpu::TextureViewDescriptor depthViewDesc{
        .label = "PBR shadow map depth slot view",
        .format = depthFormat,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = slot,
        .arrayLayerCount = 1,
    };
    sPbrShadowMapSlotViews[slot] = sPbrShadowMapTexture.CreateView(&depthViewDesc);
  }
  sPbrShadowMapView = sPbrShadowMapSlotViews[0];

  sPbrShadowMapDepthFormat = depthFormat;
  sPbrShadowMapResourceSize = sPbrShadowMapSize;
  slotViewsReady = true;
  for (const auto& view : sPbrShadowMapSlotViews) {
    slotViewsReady = slotViewsReady && static_cast<bool>(view);
  }
  sPbrShadowMapResourcesReady = sPbrShadowMapTexture && sPbrShadowMapView && sPbrShadowMapSampler && slotViewsReady;
  sPbrShadowMapAvailable = false;
  sPbrShadowMapDrawCount = 0;
  sPbrShadowMapSlotDrawCounts = {};
  sPbrShadowMapCapturedSlotCount = 0;
  sPbrShadowMapCapturedSlotMask = 0;
  sPbrShadowMapPendingSlotMask = pbr_shadow_active_slot_mask();
  return sPbrShadowMapResourcesReady;
}

float pbr_mtx(const Mat3x4<float>& mtx, u32 row, u32 col) noexcept {
  return (&mtx.m0)[row][col];
}

float& pbr_mtx(Mat3x4<float>& mtx, u32 row, u32 col) noexcept {
  return (&mtx.m0)[row][col];
}

Mat3x4<float> pbr_mul_affine(const Mat3x4<float>& lhs, const Mat3x4<float>& rhs) noexcept {
  Mat3x4<float> out;
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 4; ++col) {
      pbr_mtx(out, row, col) = pbr_mtx(lhs, row, 0) * pbr_mtx(rhs, 0, col) +
                               pbr_mtx(lhs, row, 1) * pbr_mtx(rhs, 1, col) +
                               pbr_mtx(lhs, row, 2) * pbr_mtx(rhs, 2, col) +
                               (col == 3 ? pbr_mtx(lhs, row, 3) : 0.0f);
    }
  }
  return out;
}

Mat3x4<float> pbr_mul_linear(const Mat3x4<float>& lhs, const Mat3x4<float>& rhs) noexcept {
  Mat3x4<float> out{};
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 3; ++col) {
      pbr_mtx(out, row, col) = pbr_mtx(lhs, row, 0) * pbr_mtx(rhs, 0, col) +
                               pbr_mtx(lhs, row, 1) * pbr_mtx(rhs, 1, col) +
                               pbr_mtx(lhs, row, 2) * pbr_mtx(rhs, 2, col);
    }
  }
  return out;
}

Mat3x4<float> pbr_make_look_at(PbrVec3 eye, PbrVec3 forward, PbrVec3 up) noexcept {
  const PbrVec3 f = pbr_normalize(forward);
  const PbrVec3 s = pbr_normalize(pbr_cross(f, up));
  const PbrVec3 u = pbr_cross(s, f);
  return {
      {s.x, s.y, s.z, -pbr_dot(s, eye)},
      {u.x, u.y, u.z, -pbr_dot(u, eye)},
      {-f.x, -f.y, -f.z, pbr_dot(f, eye)},
  };
}

bool pbr_probe_camera_changed(const PbrProbeCamera& reference, const PbrProbeCamera& camera) noexcept {
  if (!reference.valid || !camera.valid) {
    return true;
  }

  const float dx = reference.sourceInvView.m0[3] - camera.sourceInvView.m0[3];
  const float dy = reference.sourceInvView.m1[3] - camera.sourceInvView.m1[3];
  const float dz = reference.sourceInvView.m2[3] - camera.sourceInvView.m2[3];
  if (dx * dx + dy * dy + dz * dz >= PbrProbeDirtyDistance * PbrProbeDirtyDistance) {
    return true;
  }

  float rotationDiff = 0.0f;
  for (u32 row = 0; row < 3; ++row) {
    for (u32 col = 0; col < 3; ++col) {
      rotationDiff += std::abs(pbr_mtx(reference.sourceInvView, row, col) - pbr_mtx(camera.sourceInvView, row, col));
    }
  }
  return rotationDiff >= PbrProbeDirtyRotationSum;
}

PbrVec3 pbr_cube_direction(u32 face, u32 x, u32 y, u32 size) noexcept {
  const float u = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 1.0f;
  const float v = 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 1.0f;
  switch (face) {
  case 0:
    return pbr_normalize({1.0f, -v, -u});
  case 1:
    return pbr_normalize({-1.0f, -v, u});
  case 2:
    return pbr_normalize({u, 1.0f, v});
  case 3:
    return pbr_normalize({u, -1.0f, -v});
  case 4:
    return pbr_normalize({u, -v, 1.0f});
  default:
    return pbr_normalize({-u, -v, -1.0f});
  }
}

float pbr_smoothstep(float a, float b, float v) noexcept {
  const float t = pbr_saturate((v - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}

float pbr_fallback_environment_luma(PbrVec3 dir) noexcept {
  const float sky = pbr_smoothstep(0.0f, 1.0f, std::max(dir.y, 0.0f));
  const float ground = pbr_smoothstep(0.0f, 1.0f, std::max(-dir.y, 0.0f));
  const float horizon = std::max(1.0f - std::max(sky, ground), 0.0f);
  return sky * 1.15f + ground * 0.45f + horizon * 0.80f;
}

uint8_t pbr_to_unorm8(float v) noexcept { return static_cast<uint8_t>(std::round(pbr_saturate(v) * 255.0f)); }

wgpu::Extent3D pbr_physical_size(wgpu::Extent3D size, gfx::TextureFormatInfo info) noexcept {
  return {
      .width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth,
      .height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight,
      .depthOrArrayLayers = size.depthOrArrayLayers,
  };
}

void pbr_write_texture_layer_mip(wgpu::Texture& texture, u32 mip, u32 layer,
                                 const gfx::ConvertedTexture& converted) noexcept {
  const wgpu::Extent3D logicalSize{
      .width = converted.width,
      .height = converted.height,
      .depthOrArrayLayers = 1,
  };
  const auto info = gfx::format_info(converted.format);
  const auto physicalSize = pbr_physical_size(logicalSize, info);
  const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
  const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
  const uint32_t bytesPerRow = widthBlocks * info.blockSize;
  const uint32_t dataSize = bytesPerRow * heightBlocks;
  if (converted.data.size() < dataSize) {
    PbrLog.warn("PBR IBL texture upload expected {} bytes, got {}", dataSize, converted.data.size());
    return;
  }

  const wgpu::TexelCopyTextureInfo dst{
      .texture = texture,
      .mipLevel = mip,
      .origin = wgpu::Origin3D{.z = layer},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferLayout layout{
      .bytesPerRow = bytesPerRow,
      .rowsPerImage = heightBlocks,
  };
  g_queue.WriteTexture(&dst, converted.data.data(), dataSize, &layout, &physicalSize);
}

void pbr_write_texture_layer_mip(wgpu::Texture& texture, u32 mip, u32 layer, u32 width, u32 height,
                                 const std::vector<uint8_t>& data) noexcept {
  const wgpu::TexelCopyTextureInfo dst{
      .texture = texture,
      .mipLevel = mip,
      .origin = wgpu::Origin3D{.z = layer},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::TexelCopyBufferLayout layout{
      .bytesPerRow = width * 4,
      .rowsPerImage = height,
  };
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  g_queue.WriteTexture(&dst, data.data(), data.size(), &layout, &size);
}

void pbr_fill_cube_texture(wgpu::Texture& texture, u32 baseSize, u32 mipCount, bool prefiltered) noexcept {
  for (u32 mip = 0; mip < mipCount; ++mip) {
    const u32 size = std::max(baseSize >> mip, 1u);
    const float roughness = mipCount > 1 ? static_cast<float>(mip) / static_cast<float>(mipCount - 1) : 0.0f;
    for (u32 face = 0; face < 6; ++face) {
      std::vector<uint8_t> pixels(static_cast<size_t>(size) * static_cast<size_t>(size) * 4);
      for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
          const PbrVec3 dir = pbr_cube_direction(face, x, y, size);
          float luma = pbr_fallback_environment_luma(dir);
          if (prefiltered) {
            luma = luma * (1.0f - roughness) + 0.80f * roughness;
          }
          const uint8_t value = pbr_to_unorm8(luma);
          const size_t offset = (static_cast<size_t>(y) * size + x) * 4;
          pixels[offset + 0] = value;
          pixels[offset + 1] = value;
          pixels[offset + 2] = value;
          pixels[offset + 3] = 255;
        }
      }
      pbr_write_texture_layer_mip(texture, mip, face, size, size, pixels);
    }
  }
}

float pbr_radical_inverse_vdc(u32 bits) noexcept {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

PbrVec3 pbr_importance_sample_ggx(float xi0, float xi1, float roughness) noexcept {
  const float a = roughness * roughness;
  const float phi = 2.0f * PbrPi * xi0;
  const float cosTheta = std::sqrt((1.0f - xi1) / (1.0f + (a * a - 1.0f) * xi1));
  const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
  return {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

float pbr_geometry_schlick_ggx(float nDotV, float roughness) noexcept {
  const float a = roughness;
  const float k = (a * a) * 0.5f;
  return nDotV / (nDotV * (1.0f - k) + k);
}

std::array<float, 2> pbr_integrate_brdf(float nDotV, float roughness) noexcept {
  constexpr u32 SampleCount = 128;
  const PbrVec3 v{std::sqrt(std::max(1.0f - nDotV * nDotV, 0.0f)), 0.0f, nDotV};
  float a = 0.0f;
  float b = 0.0f;

  for (u32 i = 0; i < SampleCount; ++i) {
    const float xi0 = static_cast<float>(i) / static_cast<float>(SampleCount);
    const float xi1 = pbr_radical_inverse_vdc(i);
    const PbrVec3 h = pbr_importance_sample_ggx(xi0, xi1, roughness);
    const float vDotH = std::max(pbr_dot(v, h), 0.0f);
    const PbrVec3 l =
        pbr_normalize({2.0f * vDotH * h.x - v.x, 2.0f * vDotH * h.y - v.y, 2.0f * vDotH * h.z - v.z});
    const float nDotL = std::max(l.z, 0.0f);
    const float nDotH = std::max(h.z, 0.0f);

    if (nDotL > 0.0f) {
      const float g = pbr_geometry_schlick_ggx(nDotL, roughness) * pbr_geometry_schlick_ggx(nDotV, roughness);
      const float gVis = (g * vDotH) / std::max(nDotH * nDotV, 1e-6f);
      const float fc = std::pow(1.0f - vDotH, 5.0f);
      a += (1.0f - fc) * gVis;
      b += fc * gVis;
    }
  }

  return {a / static_cast<float>(SampleCount), b / static_cast<float>(SampleCount)};
}

std::vector<uint8_t> pbr_generate_brdf_lut() {
  std::vector<uint8_t> pixels(static_cast<size_t>(PbrBrdfLutSize) * static_cast<size_t>(PbrBrdfLutSize) * 4);
  for (u32 y = 0; y < PbrBrdfLutSize; ++y) {
    const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(PbrBrdfLutSize);
    for (u32 x = 0; x < PbrBrdfLutSize; ++x) {
      const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(PbrBrdfLutSize);
      const auto integrated = pbr_integrate_brdf(nDotV, roughness);
      const size_t offset = (static_cast<size_t>(y) * PbrBrdfLutSize + x) * 4;
      pixels[offset + 0] = pbr_to_unorm8(integrated[0]);
      pixels[offset + 1] = pbr_to_unorm8(integrated[1]);
      pixels[offset + 2] = 0;
      pixels[offset + 3] = 255;
    }
  }
  return pixels;
}

std::filesystem::path pbr_ibl_root_path() {
  if (g_config.userPath == nullptr) {
    return {};
  }
  return std::filesystem::path{reinterpret_cast<const char8_t*>(g_config.userPath)} / "texture_replacements" /
         "pbr_ibl";
}

bool pbr_is_safe_scene_component(std::string_view value) noexcept {
  if (value.empty()) {
    return false;
  }
  for (const char ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                    ch == '_' || ch == '-';
    if (!ok) {
      return false;
    }
  }
  return true;
}

std::optional<std::filesystem::path> pbr_find_existing_file(
    const std::filesystem::path& dir, std::initializer_list<std::filesystem::path> names) noexcept {
  std::error_code ec;
  for (const auto& name : names) {
    const auto path = dir / name;
    if (std::filesystem::is_regular_file(path, ec)) {
      return path;
    }
  }
  return std::nullopt;
}

std::optional<gfx::ConvertedTexture> pbr_load_face_dds(const std::filesystem::path& path) noexcept {
  auto loaded = gfx::dds::load_dds_file(path);
  if (!loaded.has_value()) {
    PbrLog.warn("Failed to load PBR IBL DDS {}", path.string());
    return std::nullopt;
  }
  return loaded;
}

std::optional<std::array<gfx::ConvertedTexture, 6>> pbr_load_cube_mip_faces(
    const std::filesystem::path& dir, std::string_view stem, u32 mip, bool allowPlainMip0) noexcept {
  std::array<gfx::ConvertedTexture, 6> faces;
  for (u32 face = 0; face < faces.size(); ++face) {
    const auto faceName = PbrCubeFaceNames[face];
    std::optional<std::filesystem::path> path;
    if (mip == 0 && allowPlainMip0) {
      path = pbr_find_existing_file(dir, {fmt::format("{}_{}.dds", stem, faceName),
                                          fmt::format("{}_mip0_{}.dds", stem, faceName),
                                          fmt::format("{}_{}_mip0.dds", stem, faceName)});
    } else {
      path = pbr_find_existing_file(dir, {fmt::format("{}_mip{}_{}.dds", stem, mip, faceName),
                                          fmt::format("{}_{}_mip{}.dds", stem, faceName, mip)});
    }
    if (!path.has_value()) {
      return std::nullopt;
    }

    auto loaded = pbr_load_face_dds(*path);
    if (!loaded.has_value()) {
      return std::nullopt;
    }
    faces[face] = std::move(*loaded);
  }

  const auto format = faces[0].format;
  const auto width = faces[0].width;
  const auto height = faces[0].height;
  for (u32 face = 1; face < faces.size(); ++face) {
    if (faces[face].format != format || faces[face].width != width || faces[face].height != height) {
      PbrLog.warn("PBR IBL cube faces for {} mip {} must have matching dimensions and format", stem, mip);
      return std::nullopt;
    }
  }
  return faces;
}

bool pbr_load_cube_from_directory(const std::filesystem::path& dir, std::string_view stem, bool prefiltered,
                                  wgpu::Texture& texture, wgpu::TextureView& view, u32& loadedMipCount) noexcept {
  std::vector<std::array<gfx::ConvertedTexture, 6>> mips;
  for (u32 mip = 0; mip < 12; ++mip) {
    auto faces = pbr_load_cube_mip_faces(dir, stem, mip, true);
    if (!faces.has_value()) {
      break;
    }
    if (!mips.empty()) {
      const uint32_t expectedWidth = std::max(mips[0][0].width >> mip, 1u);
      const uint32_t expectedHeight = std::max(mips[0][0].height >> mip, 1u);
      if ((*faces)[0].width != expectedWidth || (*faces)[0].height != expectedHeight) {
        PbrLog.warn("PBR IBL {} mip {} expected {}x{}, got {}x{}", stem, mip, expectedWidth, expectedHeight,
                    (*faces)[0].width, (*faces)[0].height);
        break;
      }
    }
    mips.push_back(std::move(*faces));
    if (!prefiltered) {
      break;
    }
  }

  if (mips.empty()) {
    return false;
  }

  const auto& first = mips[0][0];
  const wgpu::TextureDescriptor desc{
      .label = prefiltered ? "PBR authored prefiltered specular cube" : "PBR authored irradiance cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = first.width, .height = first.height, .depthOrArrayLayers = 6},
      .format = first.format,
      .mipLevelCount = static_cast<uint32_t>(mips.size()),
      .sampleCount = 1,
  };
  texture = g_device.CreateTexture(&desc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = prefiltered ? "PBR authored prefiltered specular cube view" : "PBR authored irradiance cube view",
      .format = first.format,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = static_cast<uint32_t>(mips.size()),
      .arrayLayerCount = 6,
  };
  view = texture.CreateView(&viewDesc);

  for (u32 mip = 0; mip < mips.size(); ++mip) {
    for (u32 face = 0; face < 6; ++face) {
      pbr_write_texture_layer_mip(texture, mip, face, mips[mip][face]);
    }
  }

  loadedMipCount = static_cast<uint32_t>(mips.size());
  return true;
}

bool pbr_load_brdf_lut_from_directory(const std::filesystem::path& dir, PbrIblTextureSet& set) noexcept {
  const auto path = pbr_find_existing_file(dir, {"brdf_lut.dds", "brdf.dds"});
  if (!path.has_value()) {
    return false;
  }
  auto lut = pbr_load_face_dds(*path);
  if (!lut.has_value()) {
    return false;
  }

  const wgpu::TextureDescriptor desc{
      .label = "PBR authored BRDF LUT",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = lut->width, .height = lut->height, .depthOrArrayLayers = 1},
      .format = lut->format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.brdfLutTexture = g_device.CreateTexture(&desc);
  const wgpu::TextureViewDescriptor viewDesc{
      .label = "PBR authored BRDF LUT view",
      .format = lut->format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  set.brdfLutView = set.brdfLutTexture.CreateView(&viewDesc);
  pbr_write_texture_layer_mip(set.brdfLutTexture, 0, 0, *lut);
  return true;
}

void pbr_update_ibl_max_mip(const PbrIblTextureSet& set) noexcept {
  const float maxMip = static_cast<float>(std::max(set.prefilterMipCount, 1u) - 1u);
  pbrIblParams = {pbrIblParams.x(), pbrIblParams.y(), pbrIblParams.z(), maxMip};
}

void load_fallback_pbr_ibl_textures(PbrIblTextureSet& set) noexcept {
  const wgpu::TextureDescriptor irradianceDesc{
      .label = "PBR fallback irradiance cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrIrradianceCubeSize, .height = PbrIrradianceCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.irradianceCubeTexture = g_device.CreateTexture(&irradianceDesc);
  const wgpu::TextureViewDescriptor irradianceViewDesc{
      .label = "PBR fallback irradiance cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.irradianceCubeView = set.irradianceCubeTexture.CreateView(&irradianceViewDesc);
  pbr_fill_cube_texture(set.irradianceCubeTexture, PbrIrradianceCubeSize, 1, false);

  const wgpu::TextureDescriptor prefilterDesc{
      .label = "PBR fallback prefiltered specular cube",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrPrefilterCubeSize, .height = PbrPrefilterCubeSize, .depthOrArrayLayers = 6},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = PbrPrefilterMipCount,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&prefilterDesc);
  const wgpu::TextureViewDescriptor prefilterViewDesc{
      .label = "PBR fallback prefiltered specular cube view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = PbrPrefilterMipCount,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&prefilterViewDesc);
  pbr_fill_cube_texture(set.prefilterCubeTexture, PbrPrefilterCubeSize, PbrPrefilterMipCount, true);
  set.prefilterMipCount = PbrPrefilterMipCount;

  const wgpu::TextureDescriptor brdfDesc{
      .label = "PBR BRDF LUT",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrBrdfLutSize, .height = PbrBrdfLutSize, .depthOrArrayLayers = 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.brdfLutTexture = g_device.CreateTexture(&brdfDesc);
  const wgpu::TextureViewDescriptor brdfViewDesc{
      .label = "PBR BRDF LUT view",
      .format = wgpu::TextureFormat::RGBA8Unorm,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = 1,
  };
  set.brdfLutView = set.brdfLutTexture.CreateView(&brdfViewDesc);

  const auto brdfLut = pbr_generate_brdf_lut();
  pbr_write_texture_layer_mip(set.brdfLutTexture, 0, 0, PbrBrdfLutSize, PbrBrdfLutSize, brdfLut);
  set.available = true;
}

void select_active_pbr_ibl_set() noexcept;

void create_probe_capture_target(PbrIblTextureSet& set, std::array<wgpu::TextureView, 6>& faceViews,
                                 wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::TextureDescriptor cubeDesc{
      .label = "PBR runtime probe cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&cubeDesc);
  set.irradianceCubeTexture = set.prefilterCubeTexture;

  const wgpu::TextureViewDescriptor cubeViewDesc{
      .label = "PBR runtime probe cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&cubeViewDesc);
  set.irradianceCubeView = set.prefilterCubeTexture.CreateView(&cubeViewDesc);

  for (u32 face = 0; face < faceViews.size(); ++face) {
    const wgpu::TextureViewDescriptor faceViewDesc{
        .label = "PBR runtime probe face view",
        .format = colorFormat,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = face,
        .arrayLayerCount = 1,
    };
    faceViews[face] = set.prefilterCubeTexture.CreateView(&faceViewDesc);
  }

  set.brdfLutTexture = sFallbackPbrIbl.brdfLutTexture;
  set.brdfLutView = sFallbackPbrIbl.brdfLutView;
  set.prefilterMipCount = 1;
  set.available = false;
}

void create_processed_probe_target(PbrIblTextureSet& set, PbrProbeTargetViews& targetViews,
                                   wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::TextureDescriptor irradianceDesc{
      .label = "PBR runtime probe irradiance cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrIrradianceCubeSize, .height = PbrIrradianceCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  set.irradianceCubeTexture = g_device.CreateTexture(&irradianceDesc);
  const wgpu::TextureViewDescriptor irradianceViewDesc{
      .label = "PBR runtime probe irradiance cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = 1,
      .arrayLayerCount = 6,
  };
  set.irradianceCubeView = set.irradianceCubeTexture.CreateView(&irradianceViewDesc);
  for (u32 face = 0; face < targetViews.irradianceFaceViews.size(); ++face) {
    const wgpu::TextureViewDescriptor faceViewDesc{
        .label = "PBR runtime probe irradiance face view",
        .format = colorFormat,
        .dimension = wgpu::TextureViewDimension::e2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = face,
        .arrayLayerCount = 1,
    };
    targetViews.irradianceFaceViews[face] = set.irradianceCubeTexture.CreateView(&faceViewDesc);
  }

  const wgpu::TextureDescriptor prefilterDesc{
      .label = "PBR runtime probe prefiltered specular cube",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 6},
      .format = colorFormat,
      .mipLevelCount = PbrRuntimeProbePrefilterMipCount,
      .sampleCount = 1,
  };
  set.prefilterCubeTexture = g_device.CreateTexture(&prefilterDesc);
  const wgpu::TextureViewDescriptor prefilterViewDesc{
      .label = "PBR runtime probe prefiltered specular cube view",
      .format = colorFormat,
      .dimension = wgpu::TextureViewDimension::Cube,
      .mipLevelCount = PbrRuntimeProbePrefilterMipCount,
      .arrayLayerCount = 6,
  };
  set.prefilterCubeView = set.prefilterCubeTexture.CreateView(&prefilterViewDesc);
  for (u32 mip = 0; mip < targetViews.prefilterFaceViews.size(); ++mip) {
    for (u32 face = 0; face < targetViews.prefilterFaceViews[mip].size(); ++face) {
      const wgpu::TextureViewDescriptor faceViewDesc{
          .label = "PBR runtime probe prefiltered specular face view",
          .format = colorFormat,
          .dimension = wgpu::TextureViewDimension::e2D,
          .baseMipLevel = mip,
          .mipLevelCount = 1,
          .baseArrayLayer = face,
          .arrayLayerCount = 1,
      };
      targetViews.prefilterFaceViews[mip][face] = set.prefilterCubeTexture.CreateView(&faceViewDesc);
    }
  }

  set.brdfLutTexture = sFallbackPbrIbl.brdfLutTexture;
  set.brdfLutView = sFallbackPbrIbl.brdfLutView;
  set.prefilterMipCount = PbrRuntimeProbePrefilterMipCount;
  set.available = false;
}

wgpu::RenderPipeline create_probe_filter_pipeline(wgpu::TextureFormat colorFormat) noexcept {
  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{
      .code = R"""(
struct Params {
  face: u32,
  mode: u32,
  mip: u32,
  sample_count: u32,
  roughness: f32,
  source_mip: f32,
  padding0: f32,
  padding1: f32,
};

@group(0) @binding(0) var source_sampler: sampler;
@group(0) @binding(1) var source_cube: texture_cube<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, -3.0),
  vec2<f32>(3.0, 1.0)
);
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
  vec2<f32>(0.0, 0.0),
  vec2<f32>(0.0, 2.0),
  vec2<f32>(2.0, 0.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var out: VertexOutput;
  out.pos = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  out.uv = uvs[vertex_index];
  return out;
}

fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
  let p = uv * 2.0 - vec2<f32>(1.0);
  let u = p.x;
  let v = p.y;
  switch face {
  case 0u: { return normalize(vec3<f32>(1.0, -v, -u)); }
  case 1u: { return normalize(vec3<f32>(-1.0, -v, u)); }
  case 2u: { return normalize(vec3<f32>(u, 1.0, v)); }
  case 3u: { return normalize(vec3<f32>(u, -1.0, -v)); }
  case 4u: { return normalize(vec3<f32>(u, -v, 1.0)); }
  default: { return normalize(vec3<f32>(-u, -v, -1.0)); }
  }
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn build_tangent(n: vec3<f32>) -> mat3x3<f32> {
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) {
    up = vec3<f32>(0.0, 0.0, 1.0);
  }
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  return mat3x3<f32>(tangent, bitangent, n);
}

fn sample_env(dir: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(source_cube, source_sampler, dir, 0.0).rgb;
}

fn filter_irradiance(n: vec3<f32>) -> vec3<f32> {
  let basis = build_tangent(n);
  var color = vec3<f32>(0.0);
  let count = max(params.sample_count, 1u);
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= count) {
      break;
    }
    let xi = vec2<f32>((f32(i) + 0.5) / f32(count), radical_inverse_vdc(i));
    let phi = 6.28318530718 * xi.x;
    let cos_theta = sqrt(1.0 - xi.y);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    color += sample_env(normalize(basis * local));
  }
  return color / f32(count);
}

fn filter_prefilter(n: vec3<f32>) -> vec3<f32> {
  if (params.roughness <= 0.001) {
    return sample_env(n);
  }

  let basis = build_tangent(n);
  let count = max(params.sample_count, 1u);
  let a = max(params.roughness * params.roughness, 0.001);
  var color = vec3<f32>(0.0);
  var total_weight = 0.0;
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= count) {
      break;
    }
    let xi = vec2<f32>((f32(i) + 0.5) / f32(count), radical_inverse_vdc(i));
    let phi = 6.28318530718 * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = normalize(basis * vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
    let l = normalize(2.0 * dot(n, h) * h - n);
    let ndotl = max(dot(n, l), 0.0);
    if (ndotl > 0.0) {
      color += sample_env(l) * ndotl;
      total_weight += ndotl;
    }
  }
  return color / max(total_weight, 0.0001);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let n = cube_dir(params.face, in.uv);
  var color: vec3<f32>;
  if (params.mode == 0u) {
    color = filter_irradiance(n);
  } else {
    color = filter_prefilter(n);
  }
  return vec4<f32>(color, 1.0);
}
)""",
  }};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &wgslSource,
      .label = "PBR runtime probe filter shader",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  const std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::Cube,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform},
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{
      .label = "PBR runtime probe filter bind group layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  sProbeFilterBindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDesc);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{
      .label = "PBR runtime probe filter pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &sProbeFilterBindGroupLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDesc);
  const std::array colorTargets{
      wgpu::ColorTargetState{
          .format = colorFormat,
          .writeMask = wgpu::ColorWriteMask::All,
      },
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor pipelineDesc{
      .label = "PBR runtime probe filter pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = wgpu::MultisampleState{.count = 1, .mask = UINT32_MAX},
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&pipelineDesc);
}

void create_probe_filter_bind_groups() noexcept {
  for (u32 pass = 0; pass < PbrProbeFilterPassCount; ++pass) {
    const wgpu::BufferDescriptor bufferDesc{
        .label = "PBR runtime probe filter params",
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(PbrProbeFilterParams),
    };
    sProbeFilterUniforms[pass] = g_device.CreateBuffer(&bufferDesc);

    const bool irradiance = pass < PbrProbeIrradianceFilterPassCount;
    const u32 localPass = irradiance ? pass : pass - PbrProbeIrradianceFilterPassCount;
    const u32 face = irradiance ? localPass : localPass % 6;
    const u32 mip = irradiance ? 0 : localPass / 6;
    const float roughness = PbrRuntimeProbePrefilterMipCount > 1
                                ? static_cast<float>(mip) / static_cast<float>(PbrRuntimeProbePrefilterMipCount - 1)
                                : 0.0f;
    const PbrProbeFilterParams params{
        .face = face,
        .mode = irradiance ? 0u : 1u,
        .mip = mip,
        .sampleCount = irradiance ? 24u : 32u,
        .roughness = roughness,
        .sourceMip = 0.0f,
    };
    g_queue.WriteBuffer(sProbeFilterUniforms[pass], 0, &params, sizeof(params));

    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .sampler = sPbrIblSampler,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .textureView = sProbeCapturePbrIbl.prefilterCubeView,
        },
        wgpu::BindGroupEntry{
            .binding = 2,
            .buffer = sProbeFilterUniforms[pass],
            .offset = 0,
            .size = sizeof(PbrProbeFilterParams),
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDesc{
        .label = "PBR runtime probe filter bind group",
        .layout = sProbeFilterBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    sProbeFilterBindGroups[pass] = g_device.CreateBindGroup(&bindGroupDesc);
  }
}

void ensure_probe_capture_resources() noexcept {
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  const u32 msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  if (sProbeCaptureResourcesReady && sProbeCaptureColorFormat == colorFormat &&
      sProbeCaptureDepthFormat == depthFormat && sProbeCaptureMsaaSamples == msaaSamples) {
    return;
  }

  create_processed_probe_target(sProbePbrIbl, sProbeTargetViews, colorFormat);
  for (auto& slot : sProbeCacheSlots) {
    create_processed_probe_target(slot.ibl, slot.targetViews, colorFormat);
    slot.resourcesReady = true;
    slot.ibl.available = false;
    slot.sceneKey.clear();
    slot.camera.valid = false;
    slot.lastUsedSerial = 0;
  }
  create_probe_capture_target(sProbeCapturePbrIbl, sProbeCaptureFaceViews, colorFormat);
  sProbeFilterPipeline = create_probe_filter_pipeline(colorFormat);
  create_probe_filter_bind_groups();

  if (msaaSamples > 1) {
    const wgpu::TextureDescriptor msaaDesc{
        .label = "PBR runtime probe capture MSAA color",
        .usage = wgpu::TextureUsage::RenderAttachment,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 1},
        .format = colorFormat,
        .mipLevelCount = 1,
        .sampleCount = msaaSamples,
    };
    sProbeCaptureMsaaColorTexture = g_device.CreateTexture(&msaaDesc);
    sProbeCaptureMsaaColorView = sProbeCaptureMsaaColorTexture.CreateView();
  } else {
    sProbeCaptureMsaaColorTexture = {};
    sProbeCaptureMsaaColorView = {};
  }

  const wgpu::TextureDescriptor depthDesc{
      .label = "PBR runtime probe capture depth",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = PbrRuntimeProbeCubeSize, .height = PbrRuntimeProbeCubeSize, .depthOrArrayLayers = 1},
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = msaaSamples,
  };
  sProbeCaptureDepthTexture = g_device.CreateTexture(&depthDesc);
  sProbeCaptureDepthView = sProbeCaptureDepthTexture.CreateView();

  sProbeCaptureColorFormat = colorFormat;
  sProbeCaptureDepthFormat = depthFormat;
  sProbeCaptureMsaaSamples = msaaSamples;
  sProbeCaptureFace = 0;
  sProbeCaptureDelayFrames = 0;
  sProbeFramesSinceRefresh = PbrProbePeriodicRefreshFrames;
  sProbeRefreshPending = true;
  sProbeCaptureCamera.valid = false;
  sProbeLastCompletedCamera.valid = false;
  sActiveProbeCacheSlot = nullptr;
  sProbeCaptureCacheSlot = nullptr;
  sProbeFilterCacheSlot = nullptr;
  sProbeCacheLastHit = false;
  sProbeCaptureResourcesReady = true;
  select_active_pbr_ibl_set();
}

bool load_authored_pbr_ibl_from_directory(const std::filesystem::path& dir, PbrIblTextureSet& set) noexcept {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return false;
  }

  bool loadedAny = false;
  u32 prefilterMipCount = set.prefilterMipCount;
  loadedAny |= pbr_load_cube_from_directory(dir, "irradiance", false, set.irradianceCubeTexture,
                                            set.irradianceCubeView, prefilterMipCount);
  if (pbr_load_cube_from_directory(dir, "prefilter", true, set.prefilterCubeTexture, set.prefilterCubeView,
                                   prefilterMipCount)) {
    set.prefilterMipCount = prefilterMipCount;
    loadedAny = true;
  }
  loadedAny |= pbr_load_brdf_lut_from_directory(dir, set);
  return loadedAny;
}

PbrIblTextureSet* active_probe_ibl_set() noexcept;

PbrIblTextureSet& active_pbr_ibl_set() noexcept { return sActivePbrIbl != nullptr ? *sActivePbrIbl : sFallbackPbrIbl; }

bool pbr_transition_blend_active() noexcept {
  return sPbrIblBlendFrom != nullptr && sPbrIblBlendFrom->available && sProbeBlendFactor < 1.0f;
}

bool pbr_spatial_blend_active() noexcept {
  return sPbrIblSpatialFrom != nullptr && sPbrIblSpatialFrom->available && sProbeSpatialBlendFactor < 0.999f;
}

PbrIblTextureSet& active_pbr_ibl_blend_from_set() noexcept {
  if (pbr_transition_blend_active()) {
    return *sPbrIblBlendFrom;
  }
  if (pbr_spatial_blend_active()) {
    return *sPbrIblSpatialFrom;
  }
  return active_pbr_ibl_set();
}

void update_pbr_ibl_blend_uniform() noexcept {
  const auto& blendFrom = active_pbr_ibl_blend_from_set();
  const float maxMip = static_cast<float>(std::max<u32>(blendFrom.prefilterMipCount, 1) - 1);
  const float blendFactor = pbr_transition_blend_active() ? sProbeBlendFactor
                                                          : (pbr_spatial_blend_active() ? sProbeSpatialBlendFactor
                                                                                        : 1.0f);
  pbrIblBlendParams = {blendFactor, maxMip, 0.0f, 0.0f};
}

void clear_pbr_probe_spatial_blend() noexcept {
  sPbrIblSpatialFrom = nullptr;
  sSpatialProbeCacheSlot = nullptr;
  sProbeSpatialBlendActive = false;
  sProbeSpatialBlendFactor = 1.0f;
  sProbeSpatialBlendDistance = 0.0f;
  update_pbr_ibl_blend_uniform();
}

void clear_pbr_ibl_blend() noexcept {
  sPbrIblBlendFrom = nullptr;
  sProbeBlendFrame = sProbeBlendFrames;
  sProbeBlendFactor = 1.0f;
  update_pbr_ibl_blend_uniform();
  g_gxState.stateDirty = true;
}

void start_pbr_ibl_blend(PbrIblTextureSet* from) noexcept {
  if (!sProbeBlendEnabled || sProbeBlendFrames == 0 || from == nullptr || !from->available || from == sActivePbrIbl) {
    clear_pbr_ibl_blend();
    return;
  }

  sPbrIblBlendFrom = from;
  sProbeBlendFrame = 0;
  sProbeBlendFactor = 0.0f;
  update_pbr_ibl_blend_uniform();
  g_gxState.stateDirty = true;
}

void advance_pbr_ibl_blend() noexcept {
  if (sPbrIblBlendFrom == nullptr || sProbeBlendFactor >= 1.0f) {
    return;
  }

  if (!sProbeBlendEnabled || sProbeBlendFrames == 0 || !sPbrIblBlendFrom->available) {
    clear_pbr_ibl_blend();
    return;
  }

  sProbeBlendFrame = std::min(sProbeBlendFrame + 1, sProbeBlendFrames);
  sProbeBlendFactor = static_cast<float>(sProbeBlendFrame) / static_cast<float>(std::max<u32>(sProbeBlendFrames, 1));
  if (sProbeBlendFactor >= 1.0f) {
    clear_pbr_ibl_blend();
    return;
  }

  update_pbr_ibl_blend_uniform();
  g_gxState.stateDirty = true;
}

AuroraPbrIblSource active_pbr_ibl_source() noexcept {
  if (sActivePbrIbl == active_probe_ibl_set()) {
    return AURORA_PBR_IBL_SOURCE_PROBE;
  }
  if (sActivePbrIbl == &sAuthoredPbrIbl) {
    return AURORA_PBR_IBL_SOURCE_AUTHORED;
  }
  return AURORA_PBR_IBL_SOURCE_FALLBACK;
}

template <size_t N>
void copy_status_string(char (&dst)[N], std::string_view value) noexcept {
  static_assert(N > 0);
  const size_t len = std::min(value.size(), N - 1);
  std::memcpy(dst, value.data(), len);
  dst[len] = '\0';
}

PbrIblTextureSet* active_probe_ibl_set() noexcept {
  if (sActiveProbeCacheSlot != nullptr && sActiveProbeCacheSlot->ibl.available) {
    return &sActiveProbeCacheSlot->ibl;
  }
  return sProbePbrIbl.available ? &sProbePbrIbl : nullptr;
}

PbrProbeCacheSlot* find_probe_cache_slot(std::string_view sceneKey) noexcept {
  if (sceneKey.empty()) {
    return nullptr;
  }
  for (auto& slot : sProbeCacheSlots) {
    if (slot.resourcesReady && slot.ibl.available && slot.sceneKey == sceneKey) {
      return &slot;
    }
  }
  return nullptr;
}

std::string_view pbr_scene_stage(std::string_view sceneKey) noexcept {
  const size_t separator = sceneKey.find(':');
  if (separator == std::string_view::npos) {
    return sceneKey;
  }
  return sceneKey.substr(0, separator);
}

PbrProbeCacheSlot* find_nearest_probe_cache_slot(std::string_view sceneKey, const PbrProbeCacheSlot* excluded,
                                                 float maxDistance, float* distanceOut) noexcept {
  if (sceneKey.empty() || !sProbeCamera.valid) {
    return nullptr;
  }

  const std::string_view stage = pbr_scene_stage(sceneKey);
  if (stage.empty()) {
    return nullptr;
  }

  PbrProbeCacheSlot* nearest = nullptr;
  float nearestDistanceSq = std::numeric_limits<float>::max();
  const PbrVec3 cameraEye = pbr_probe_eye(sProbeCamera);
  for (auto& slot : sProbeCacheSlots) {
    if (&slot == excluded || !slot.resourcesReady || !slot.ibl.available || !slot.camera.valid ||
        slot.sceneKey == sceneKey ||
        pbr_scene_stage(slot.sceneKey) != stage) {
      continue;
    }

    const float distanceSq = pbr_distance_sq(cameraEye, pbr_probe_eye(slot.camera));
    if (distanceSq < nearestDistanceSq) {
      nearestDistanceSq = distanceSq;
      nearest = &slot;
    }
  }

  if (nearest == nullptr) {
    return nullptr;
  }

  const float distance = std::sqrt(nearestDistanceSq);
  if (maxDistance > 0.0f && distance > maxDistance) {
    return nullptr;
  }

  if (distanceOut != nullptr) {
    *distanceOut = distance;
  }
  return nearest;
}

PbrProbeCacheSlot* find_nearest_probe_cache_slot(std::string_view sceneKey) noexcept {
  if (!sProbeNearestCacheEnabled) {
    return nullptr;
  }
  return find_nearest_probe_cache_slot(sceneKey, nullptr, sProbeNearestCacheMaxDistance, &sProbeNearestCacheDistance);
}

bool activate_nearest_probe_cache_slot(std::string_view sceneKey, PbrIblTextureSet* previousIbl, bool blend) noexcept {
  auto* nearest = find_nearest_probe_cache_slot(sceneKey);
  if (nearest == nullptr) {
    return false;
  }

  sActiveProbeCacheSlot = nearest;
  nearest->lastUsedSerial = ++sProbeCacheSerial;
  sProbeLastCompletedCamera = nearest->camera;
  sProbeSceneStale = true;
  sProbeNearestCacheActive = true;
  select_active_pbr_ibl_set();
  if (blend) {
    start_pbr_ibl_blend(previousIbl);
  }
  return true;
}

void update_pbr_probe_spatial_blend() noexcept {
  if (!sProbeSpatialBlendEnabled || !sProbeCacheEnabled || sRequestedPbrIblSource != AURORA_PBR_IBL_SOURCE_PROBE ||
      !sProbeCamera.valid || sActiveProbeCacheSlot == nullptr || !sActiveProbeCacheSlot->ibl.available ||
      !sActiveProbeCacheSlot->camera.valid) {
    clear_pbr_probe_spatial_blend();
    return;
  }

  float nearestDistance = 0.0f;
  auto* nearest = find_nearest_probe_cache_slot(sActivePbrIblSceneKey, sActiveProbeCacheSlot,
                                                sProbeSpatialBlendMaxDistance, &nearestDistance);
  if (nearest == nullptr) {
    clear_pbr_probe_spatial_blend();
    return;
  }

  const PbrVec3 cameraEye = pbr_probe_eye(sProbeCamera);
  const float activeDistance = std::sqrt(pbr_distance_sq(cameraEye, pbr_probe_eye(sActiveProbeCacheSlot->camera)));
  constexpr float MinProbeDistance = 128.0f;
  const float activeWeightDistance = std::max(activeDistance, MinProbeDistance);
  const float nearestWeightDistance = std::max(nearestDistance, MinProbeDistance);
  const float activeWeight = 1.0f / (activeWeightDistance * activeWeightDistance);
  const float nearestWeight = 1.0f / (nearestWeightDistance * nearestWeightDistance);
  const float totalWeight = activeWeight + nearestWeight;
  const float activeFactor = totalWeight > 0.0f ? activeWeight / totalWeight : 1.0f;
  const bool meaningfulBlend = activeFactor < 0.95f;
  if (!meaningfulBlend) {
    clear_pbr_probe_spatial_blend();
    return;
  }

  const bool slotChanged = nearest != sSpatialProbeCacheSlot;
  sSpatialProbeCacheSlot = nearest;
  sPbrIblSpatialFrom = &nearest->ibl;
  sProbeSpatialBlendActive = true;
  sProbeSpatialBlendFactor = std::clamp(activeFactor, 0.0f, 1.0f);
  sProbeSpatialBlendDistance = nearestDistance;
  nearest->lastUsedSerial = ++sProbeCacheSerial;
  update_pbr_ibl_blend_uniform();
  if (slotChanged) {
    g_gxState.stateDirty = true;
  }
}

PbrProbeCacheSlot* select_probe_cache_slot_for_capture(std::string_view sceneKey) noexcept {
  if (!sProbeCacheEnabled || sceneKey.empty()) {
    return nullptr;
  }

  if (auto* existing = find_probe_cache_slot(sceneKey); existing != nullptr) {
    return existing;
  }

  PbrProbeCacheSlot* oldest = nullptr;
  for (auto& slot : sProbeCacheSlots) {
    if (!slot.resourcesReady) {
      continue;
    }
    if (!slot.ibl.available) {
      return &slot;
    }
    if (oldest == nullptr || slot.lastUsedSerial < oldest->lastUsedSerial) {
      oldest = &slot;
    }
  }

  return oldest;
}

u32 used_probe_cache_slot_count() noexcept {
  u32 count = 0;
  for (const auto& slot : sProbeCacheSlots) {
    if (slot.resourcesReady && slot.ibl.available) {
      ++count;
    }
  }
  return count;
}

void select_active_pbr_ibl_set() noexcept {
  PbrIblTextureSet* selected = &sFallbackPbrIbl;
  switch (sRequestedPbrIblSource) {
  case AURORA_PBR_IBL_SOURCE_PROBE:
    if (auto* probe = active_probe_ibl_set(); probe != nullptr) {
      selected = probe;
    }
    break;
  case AURORA_PBR_IBL_SOURCE_AUTHORED:
    if (sAuthoredPbrIbl.available) {
      selected = &sAuthoredPbrIbl;
    }
    break;
  case AURORA_PBR_IBL_SOURCE_FALLBACK:
  default:
    break;
  }

  sActivePbrIbl = selected;
  if (sRequestedPbrIblSource != AURORA_PBR_IBL_SOURCE_PROBE || selected != active_probe_ibl_set()) {
    clear_pbr_probe_spatial_blend();
  }
  pbr_update_ibl_max_mip(*sActivePbrIbl);
  update_pbr_ibl_blend_uniform();
  g_gxState.stateDirty = true;
}

Mat3x4<float> pbr_load_mtx3x4(const float* data) noexcept {
  if (data == nullptr) {
    return {};
  }
  return {
      {data[0], data[1], data[2], data[3]},
      {data[4], data[5], data[6], data[7]},
      {data[8], data[9], data[10], data[11]},
  };
}

Mat4x4<float> pbr_projection_from_gx_mtx(const float* data) noexcept {
  if (data == nullptr) {
    return {};
  }
  const float p0 = data[0];
  const float p1 = data[2];
  const float p2 = data[5];
  const float p3 = data[6];
  const float p4 = data[10];
  const float p5 = data[11];
  Mat4x4<float> proj{};
  proj.m0[0] = p0;
  proj.m0[2] = p1;
  proj.m1[1] = p2;
  proj.m1[2] = p3;
  proj.m2[2] = p4;
  proj.m2[3] = p5;
  proj.m3[2] = -1.0f;
  return proj;
}

void pbr_update_probe_face_matrices() noexcept {
  const PbrVec3 eye{sProbeCamera.sourceInvView.m0[3], sProbeCamera.sourceInvView.m1[3],
                    sProbeCamera.sourceInvView.m2[3]};
  constexpr std::array<PbrVec3, 6> directions{{
      {1.0f, 0.0f, 0.0f},
      {-1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, -1.0f},
  }};
  constexpr std::array<PbrVec3, 6> ups{{
      {0.0f, -1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, -1.0f},
      {0.0f, -1.0f, 0.0f},
      {0.0f, -1.0f, 0.0f},
  }};

  for (u32 face = 0; face < 6; ++face) {
    const Mat3x4<float> faceView = pbr_make_look_at(eye, directions[face], ups[face]);
    sProbeCamera.viewFromSource[face] = pbr_mul_affine(faceView, sProbeCamera.sourceInvView);
    sProbeCamera.normalFromSource[face] = pbr_mul_linear(faceView, sProbeCamera.sourceInvView);
  }
}

gfx::TextureHandle find_named_pbr_sidecar(std::string_view materialName,
                                          std::initializer_list<std::string_view> suffixes) {
  if (materialName.empty()) {
    return {};
  }

  for (std::string_view suffix : suffixes) {
    std::string candidate{materialName};
    candidate += suffix;
    if (auto texture = gfx::texture_replacement::find_named_pbr_texture(candidate); texture.has_value()) {
      return *texture;
    }
  }

  return {};
}

void clear_pbr_material_maps() {
  pbrMaterialRmaos.reset();
  pbrMaterialRoughness.reset();
  pbrMaterialMetallic.reset();
  pbrMaterialAo.reset();
  pbrMaterialSpecular.reset();
  pbrMaterialNormal.reset();
  pbrMaterialEmissiveMap.reset();
}

} // namespace pbr_internal

using namespace pbr_internal;

void initialize_pbr_resources() noexcept {
  const wgpu::SamplerDescriptor materialSamplerDesc{
      .label = "PBR material sampler",
      .addressModeU = wgpu::AddressMode::Repeat,
      .addressModeV = wgpu::AddressMode::Repeat,
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.0f,
      .lodMaxClamp = 1000.0f,
  };
  sPbrMaterialSampler = gfx::sampler_ref(materialSamplerDesc);

  const wgpu::SamplerDescriptor iblSamplerDesc{
      .label = "PBR IBL sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
      .lodMinClamp = 0.0f,
      .lodMaxClamp = static_cast<float>(PbrPrefilterMipCount - 1),
  };
  sPbrIblSampler = gfx::sampler_ref(iblSamplerDesc);

  const wgpu::SamplerDescriptor brdfSamplerDesc{
      .label = "PBR BRDF LUT sampler",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Undefined,
  };
  sPbrBrdfLutSampler = gfx::sampler_ref(brdfSamplerDesc);
  pbr_set_shadow_identity_matrix();
  ensure_pbr_shadow_fallback_resources();

  load_fallback_pbr_ibl_textures(sFallbackPbrIbl);
  sActivePbrIbl = &sFallbackPbrIbl;
  select_active_pbr_ibl_set();
}

void add_pbr_texture_layout_entries(
    std::array<wgpu::BindGroupLayoutEntry, TextureBindGroupEntryCount>& textureEntries) noexcept {
  const auto addPbrTextureLayout = [&](u32 textureBinding,
                                       wgpu::TextureViewDimension viewDimension = wgpu::TextureViewDimension::e2D) {
    textureEntries[textureBinding] = {
        .binding = textureBinding,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            {
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = viewDimension,
            },
    };
  };
  const auto addPbrSamplerLayout = [&](u32 samplerBinding) {
    textureEntries[samplerBinding] = {
        .binding = samplerBinding,
        .visibility = wgpu::ShaderStage::Fragment,
        .sampler = {.type = wgpu::SamplerBindingType::Filtering},
    };
  };
  const auto addPbrShadowTextureLayout = [&](u32 textureBinding) {
    textureEntries[textureBinding] = {
        .binding = textureBinding,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            {
                .sampleType = wgpu::TextureSampleType::Depth,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
    };
  };
  const auto addPbrShadowSamplerLayout = [&](u32 samplerBinding) {
    textureEntries[samplerBinding] = {
        .binding = samplerBinding,
        .visibility = wgpu::ShaderStage::Fragment,
      .sampler = {.type = wgpu::SamplerBindingType::Comparison},
    };
  };
  addPbrTextureLayout(pbr_rmaos_texture_binding(0));
  addPbrTextureLayout(pbr_roughness_texture_binding(0));
  addPbrTextureLayout(pbr_metallic_texture_binding(0));
  addPbrTextureLayout(pbr_ao_texture_binding(0));
  addPbrTextureLayout(pbr_specular_texture_binding(0));
  addPbrTextureLayout(pbr_normal_texture_binding(0));
  addPbrTextureLayout(pbr_emissive_texture_binding(0));
  addPbrSamplerLayout(pbr_material_sampler_binding(0));
  addPbrTextureLayout(pbr_ibl_irradiance_texture_binding(0), wgpu::TextureViewDimension::Cube);
  addPbrTextureLayout(pbr_ibl_prefilter_texture_binding(0), wgpu::TextureViewDimension::Cube);
  addPbrTextureLayout(pbr_ibl_brdf_lut_texture_binding(0));
  addPbrTextureLayout(pbr_ibl_blend_irradiance_texture_binding(0), wgpu::TextureViewDimension::Cube);
  addPbrTextureLayout(pbr_ibl_blend_prefilter_texture_binding(0), wgpu::TextureViewDimension::Cube);
  addPbrSamplerLayout(pbr_ibl_sampler_binding(0));
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    addPbrShadowTextureLayout(pbr_shadow_texture_binding(0, slot));
  }
  addPbrShadowSamplerLayout(pbr_shadow_sampler_binding(0));
}

void add_pbr_empty_bind_group_entries(std::array<wgpu::BindGroupEntry, TextureBindGroupEntryCount>& entries,
                                      const wgpu::TextureView& emptyTextureView,
                                      const wgpu::Sampler& emptySampler) noexcept {
  const auto addPbrEmptyTexture = [&](u32 textureBinding) {
    entries[textureBinding] = {
        .binding = textureBinding,
        .textureView = emptyTextureView,
    };
  };

  addPbrEmptyTexture(pbr_rmaos_texture_binding(0));
  addPbrEmptyTexture(pbr_roughness_texture_binding(0));
  addPbrEmptyTexture(pbr_metallic_texture_binding(0));
  addPbrEmptyTexture(pbr_ao_texture_binding(0));
  addPbrEmptyTexture(pbr_specular_texture_binding(0));
  addPbrEmptyTexture(pbr_normal_texture_binding(0));
  addPbrEmptyTexture(pbr_emissive_texture_binding(0));
  entries[pbr_material_sampler_binding(0)] = {
      .binding = pbr_material_sampler_binding(0),
      .sampler = emptySampler,
  };
  const auto& ibl = active_pbr_ibl_set();
  entries[pbr_ibl_irradiance_texture_binding(0)] = {
      .binding = pbr_ibl_irradiance_texture_binding(0),
      .textureView = ibl.irradianceCubeView,
  };
  entries[pbr_ibl_prefilter_texture_binding(0)] = {
      .binding = pbr_ibl_prefilter_texture_binding(0),
      .textureView = ibl.prefilterCubeView,
  };
  entries[pbr_ibl_brdf_lut_texture_binding(0)] = {
      .binding = pbr_ibl_brdf_lut_texture_binding(0),
      .textureView = ibl.brdfLutView,
  };
  const auto& blendFrom = active_pbr_ibl_blend_from_set();
  entries[pbr_ibl_blend_irradiance_texture_binding(0)] = {
      .binding = pbr_ibl_blend_irradiance_texture_binding(0),
      .textureView = blendFrom.irradianceCubeView,
  };
  entries[pbr_ibl_blend_prefilter_texture_binding(0)] = {
      .binding = pbr_ibl_blend_prefilter_texture_binding(0),
      .textureView = blendFrom.prefilterCubeView,
  };
  entries[pbr_ibl_sampler_binding(0)] = {
      .binding = pbr_ibl_sampler_binding(0),
      .sampler = sPbrIblSampler,
  };
  ensure_pbr_shadow_fallback_resources();
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    entries[pbr_shadow_texture_binding(0, slot)] = {
        .binding = pbr_shadow_texture_binding(0, slot),
        .textureView = sPbrShadowFallbackView,
    };
  }
  entries[pbr_shadow_sampler_binding(0)] = {
      .binding = pbr_shadow_sampler_binding(0),
      .sampler = sPbrShadowMapSampler,
  };
}

void bind_pbr_texture_entries(std::array<WGPUBindGroupEntry, TextureBindGroupEntryCount>& textureEntries,
                              const ShaderInfo& info, const wgpu::TextureView& emptyTextureView,
                              const wgpu::Sampler& emptySampler, bool bindShadowReceivers) noexcept {
  (void)emptySampler;
  const auto bindPbrTexture = [&](u32 textureBinding, const gfx::TextureHandle& handle) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    pbrTextureEntry.binding = textureBinding;
    if (handle) {
      pbrTextureEntry.textureView = handle->sampleTextureView.Get();
    } else {
      pbrTextureEntry.textureView = emptyTextureView.Get();
    }
  };
  const auto bindPbrView = [&](u32 textureBinding, const wgpu::TextureView& textureView) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    pbrTextureEntry.binding = textureBinding;
    pbrTextureEntry.textureView = textureView.Get();
  };
  const auto bindPbrStandaloneTexture = [&](u32 textureBinding, const gfx::TextureHandle& handle) {
    WGPUBindGroupEntry& pbrTextureEntry = textureEntries[textureBinding];
    pbrTextureEntry.binding = textureBinding;
    if (handle) {
      pbrTextureEntry.textureView = handle->sampleTextureView.Get();
    } else {
      pbrTextureEntry.textureView = emptyTextureView.Get();
    }
  };
  WGPUBindGroupEntry& materialSamplerEntry = textureEntries[pbr_material_sampler_binding(0)];
  materialSamplerEntry.binding = pbr_material_sampler_binding(0);
  materialSamplerEntry.sampler = sPbrMaterialSampler.Get();
  WGPUBindGroupEntry& iblSamplerEntry = textureEntries[pbr_ibl_sampler_binding(0)];
  iblSamplerEntry.binding = pbr_ibl_sampler_binding(0);
  iblSamplerEntry.sampler = sPbrIblSampler.Get();

  const gfx::TextureBind* pbrTex = nullptr;
  if ((info.pbrFlags & PbrMaterialEnabled) != 0 && info.pbrTexMapId < MaxTextures) {
    pbrTex = &g_gxState.textures[info.pbrTexMapId];
  }
  const bool useGlobalMaps = (info.pbrFlags & PbrMaterialUseGlobalMaps) != 0;
  if (useGlobalMaps) {
    bindPbrStandaloneTexture(pbr_rmaos_texture_binding(0), pbrMaterialRmaos);
    bindPbrStandaloneTexture(pbr_roughness_texture_binding(0), pbrMaterialRoughness);
    bindPbrStandaloneTexture(pbr_metallic_texture_binding(0), pbrMaterialMetallic);
    bindPbrStandaloneTexture(pbr_ao_texture_binding(0), pbrMaterialAo);
    bindPbrStandaloneTexture(pbr_specular_texture_binding(0), pbrMaterialSpecular);
    bindPbrStandaloneTexture(pbr_normal_texture_binding(0), pbrMaterialNormal);
    bindPbrStandaloneTexture(pbr_emissive_texture_binding(0), pbrMaterialEmissiveMap);
  } else {
    bindPbrTexture(pbr_rmaos_texture_binding(0), pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrRmaos : nullptr);
    bindPbrTexture(pbr_roughness_texture_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrRoughness : nullptr);
    bindPbrTexture(pbr_metallic_texture_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrMetallic : nullptr);
    bindPbrTexture(pbr_ao_texture_binding(0), pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrAo : nullptr);
    bindPbrTexture(pbr_specular_texture_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrSpecular : nullptr);
    bindPbrTexture(pbr_normal_texture_binding(0), pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrNormal : nullptr);
    bindPbrTexture(pbr_emissive_texture_binding(0),
                   pbrTex != nullptr && *pbrTex ? pbrTex->ref->pbrEmissive : nullptr);
  }
  const auto& ibl = active_pbr_ibl_set();
  const auto& blendFrom = active_pbr_ibl_blend_from_set();
  bindPbrView(pbr_ibl_irradiance_texture_binding(0), ibl.irradianceCubeView);
  bindPbrView(pbr_ibl_prefilter_texture_binding(0), ibl.prefilterCubeView);
  bindPbrView(pbr_ibl_brdf_lut_texture_binding(0), ibl.brdfLutView);
  bindPbrView(pbr_ibl_blend_irradiance_texture_binding(0), blendFrom.irradianceCubeView);
  bindPbrView(pbr_ibl_blend_prefilter_texture_binding(0), blendFrom.prefilterCubeView);
  if (!bindShadowReceivers) {
    ensure_pbr_shadow_fallback_resources();
  }
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    WGPUBindGroupEntry& pbrShadowEntry = textureEntries[pbr_shadow_texture_binding(0, slot)];
    pbrShadowEntry.binding = pbr_shadow_texture_binding(0, slot);
    pbrShadowEntry.textureView =
        (bindShadowReceivers ? pbr_shadow_receiver_view(slot) : sPbrShadowFallbackView).Get();
  }
  WGPUBindGroupEntry& pbrShadowSamplerEntry = textureEntries[pbr_shadow_sampler_binding(0)];
  pbrShadowSamplerEntry.binding = pbr_shadow_sampler_binding(0);
  pbrShadowSamplerEntry.sampler = pbr_shadow_receiver_sampler().Get();
}

bool pbr_probe_capture_requested() noexcept {
  using namespace pbr_internal;
  advance_pbr_ibl_blend();
  if (!enablePbrMaterialOverride || pbrIblParams.x() <= 0.0f ||
      sRequestedPbrIblSource != AURORA_PBR_IBL_SOURCE_PROBE || !sProbeCamera.valid) {
    sProbeCaptureDelayFrames = 0;
    sProbeReplayDrawCount = 0;
    sProbeReplayPbrVisible = false;
    return false;
  }

  if (sProbeCaptureDelayFrames > 0) {
    --sProbeCaptureDelayFrames;
    return false;
  }

  if (sProbeCaptureFace != 0 || active_probe_ibl_set() == nullptr || sProbeRefreshPending) {
    ensure_probe_capture_resources();
    return sProbeCaptureResourcesReady;
  }

  if (sProbeAutoRefresh && sProbeFramesSinceRefresh >= PbrProbePeriodicRefreshFrames) {
    sProbeRefreshPending = true;
    ensure_probe_capture_resources();
    return sProbeCaptureResourcesReady;
  }

  ++sProbeFramesSinceRefresh;
  return false;
}

void set_pbr_probe_replay_status(uint32_t eligible_draws, bool pbr_material_visible) noexcept {
  using namespace pbr_internal;
  sProbeReplayDrawCount = eligible_draws;
  sProbeReplayPbrVisible = pbr_material_visible;
}

uint32_t pbr_probe_cube_size() noexcept { return pbr_internal::PbrRuntimeProbeCubeSize; }

uint32_t pbr_probe_capture_face() noexcept { return pbr_internal::sProbeCaptureFace; }

void begin_pbr_probe_capture() noexcept {
  using namespace pbr_internal;
  ensure_probe_capture_resources();
  if (sProbeCaptureFace == 0) {
    sProbeCaptureCamera = sProbeCamera;
    sProbeCaptureCacheSlot = select_probe_cache_slot_for_capture(sActivePbrIblSceneKey);
    if (sProbeCaptureCacheSlot != nullptr) {
      sProbeCaptureCacheSlot->sceneKey = sActivePbrIblSceneKey;
    }
    sProbeRefreshPending = false;
  }
}

void finish_pbr_probe_capture() noexcept {
  using namespace pbr_internal;
  if (!sProbeCaptureResourcesReady) {
    return;
  }

  sProbeCaptureDelayFrames = PbrProbeCaptureFramesPerFace > 0 ? PbrProbeCaptureFramesPerFace - 1 : 0;
  sProbeCaptureFace = (sProbeCaptureFace + 1) % 6;
  if (sProbeCaptureFace != 0) {
    return;
  }

  PbrIblTextureSet* previousIbl = sActivePbrIbl;
  PbrIblTextureSet* completedProbe = &sProbePbrIbl;
  if (sProbeCaptureCacheSlot != nullptr) {
    completedProbe = &sProbeCaptureCacheSlot->ibl;
    sProbeCaptureCacheSlot->camera = sProbeCaptureCamera;
    sProbeCaptureCacheSlot->lastUsedSerial = ++sProbeCacheSerial;
    sActiveProbeCacheSlot = sProbeCaptureCacheSlot;
  } else {
    sActiveProbeCacheSlot = nullptr;
  }

  completedProbe->available = true;
  sProbeSceneStale = false;
  sProbeNearestCacheActive = false;
  sProbeNearestCacheDistance = 0.0f;
  sProbeFilterPending = true;
  sProbeFilterCacheSlot = sProbeCaptureCacheSlot;
  sProbeCaptureCacheSlot = nullptr;
  sProbeFramesSinceRefresh = 0;
  sProbeRefreshPending = false;
  sProbeLastCompletedCamera = sProbeCaptureCamera;
  sProbeCaptureCamera.valid = false;
  select_active_pbr_ibl_set();
  update_pbr_probe_spatial_blend();
  start_pbr_ibl_blend(previousIbl);
}

const wgpu::TextureView& pbr_probe_capture_face_view(uint32_t face) noexcept {
  return pbr_internal::sProbeCaptureFaceViews[std::min<uint32_t>(face, 5)];
}

const wgpu::TextureView& pbr_probe_capture_color_view(uint32_t face) noexcept {
  if (webgpu::g_graphicsConfig.msaaSamples > 1) {
    return pbr_internal::sProbeCaptureMsaaColorView;
  }
  return pbr_probe_capture_face_view(face);
}

const wgpu::TextureView& pbr_probe_capture_depth_view() noexcept { return pbr_internal::sProbeCaptureDepthView; }

void run_pbr_probe_filter(const wgpu::CommandEncoder& cmd) noexcept {
  using namespace pbr_internal;
  if (!sProbeFilterPending || !sProbeCaptureResourcesReady || !sProbeFilterPipeline) {
    return;
  }
  const auto& targetViews = sProbeFilterCacheSlot != nullptr ? sProbeFilterCacheSlot->targetViews : sProbeTargetViews;

  const auto runPass = [&](u32 passIndex, const wgpu::TextureView& targetView, u32 size, std::string_view label) {
    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = targetView,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = {0.0, 0.0, 0.0, 1.0},
        },
    };
    const wgpu::RenderPassDescriptor passDesc{
        .label = label.data(),
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    auto pass = cmd.BeginRenderPass(&passDesc);
    pass.SetPipeline(sProbeFilterPipeline);
    pass.SetBindGroup(0, sProbeFilterBindGroups[passIndex]);
    pass.SetViewport(0.0f, 0.0f, static_cast<float>(size), static_cast<float>(size), 0.0f, 1.0f);
    pass.SetScissorRect(0, 0, size, size);
    pass.Draw(3);
    pass.End();
  };

  for (u32 face = 0; face < 6; ++face) {
    runPass(face, targetViews.irradianceFaceViews[face], PbrIrradianceCubeSize, "PBR probe irradiance filter");
  }

  for (u32 mip = 0; mip < PbrRuntimeProbePrefilterMipCount; ++mip) {
    const u32 size = std::max(PbrRuntimeProbeCubeSize >> mip, 1u);
    for (u32 face = 0; face < 6; ++face) {
      const u32 passIndex = PbrProbeIrradianceFilterPassCount + mip * 6 + face;
      runPass(passIndex, targetViews.prefilterFaceViews[mip][face], size, "PBR probe prefilter");
    }
  }

  sProbeFilterPending = false;
  sProbeFilterCacheSlot = nullptr;
  select_active_pbr_ibl_set();
}

void patch_pbr_probe_uniform(uint8_t* uniformData, const ProbeUniformPatchInfo& patch, uint32_t face) noexcept {
  using namespace pbr_internal;
  const auto& probeCamera = sProbeCaptureCamera.valid ? sProbeCaptureCamera : sProbeCamera;
  if (uniformData == nullptr || !patch.eligible || !probeCamera.valid || face >= 6) {
    return;
  }
  if (patch.nrmMtxOffset + sizeof(Mat3x4<float>) * MaxPnMtx > patch.uniformSize ||
      patch.projectionOffset + sizeof(Mat4x4<float>) > patch.uniformSize) {
    return;
  }

  auto* viewport = reinterpret_cast<float*>(uniformData + 8);
  const float size = static_cast<float>(PbrRuntimeProbeCubeSize);
  viewport[0] = size;
  viewport[1] = size;
  viewport[2] = size;
  viewport[3] = size;
  std::memcpy(uniformData + patch.projectionOffset, &probeCamera.projection, sizeof(probeCamera.projection));

  auto* posMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.pnMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    posMtx[i] = pbr_mul_affine(probeCamera.viewFromSource[face], posMtx[i]);
  }

  auto* nrmMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.nrmMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    nrmMtx[i] = pbr_mul_linear(probeCamera.normalFromSource[face], nrmMtx[i]);
  }
}

bool pbr_shadow_map_capture_requested() noexcept {
  using namespace pbr_internal;
  return sPbrShadowMapRefreshPending && sPbrShadowMapEnabled && sPbrShadowLightRequest.valid &&
         sPbrShadowMapMatrixValid && sProbeCamera.valid && sPbrShadowMapPendingSlotMask != 0 &&
         ensure_pbr_shadow_map_resources();
}

uint32_t pbr_shadow_map_capture_slot_count() noexcept {
  using namespace pbr_internal;
  if (!pbr_shadow_map_capture_requested()) {
    return 0;
  }
  return pbr_shadow_active_slot_count();
}

uint32_t pbr_shadow_map_capture_slots_per_frame() noexcept {
  using namespace pbr_internal;
  return std::clamp<u32>(sPbrShadowMapSlotsPerFrame, 1u, PbrShadowAtlasSlotCount);
}

bool pbr_shadow_map_capture_slot_requested(uint32_t slot) noexcept {
  using namespace pbr_internal;
  return slot < PbrShadowAtlasSlotCount && (sPbrShadowMapPendingSlotMask & (1u << slot)) != 0;
}

uint32_t pbr_shadow_map_size() noexcept { return pbr_internal::sPbrShadowMapSize; }

const wgpu::TextureView& pbr_shadow_map_depth_view() noexcept {
  pbr_internal::ensure_pbr_shadow_map_resources();
  return pbr_internal::sPbrShadowMapView;
}

const wgpu::TextureView& pbr_shadow_map_depth_slot_view(uint32_t slot) noexcept {
  using namespace pbr_internal;
  ensure_pbr_shadow_map_resources();
  if (slot < PbrShadowAtlasSlotCount && sPbrShadowMapSlotViews[slot]) {
    return sPbrShadowMapSlotViews[slot];
  }
  return sPbrShadowMapView;
}

bool begin_pbr_shadow_map_capture_slot(uint32_t slot) noexcept {
  using namespace pbr_internal;
  sPbrShadowCasterPassReady = true;
  if (!pbr_shadow_map_capture_requested() || !pbr_shadow_map_capture_slot_requested(slot) ||
      slot >= pbr_shadow_active_slot_count() ||
      slot >= PbrShadowAtlasSlotCount) {
    return false;
  }
  const AuroraPbrShadowLightRequest& request = sPbrShadowLightRequests[slot];
  if (!request.valid || request.shadowType == AURORA_PBR_SHADOW_TYPE_POINT) {
    return false;
  }

  sPbrShadowLightRequest = request;
  sPbrShadowLightRequest.atlasSlot = slot;
  return pbr_build_shadow_matrix_from_request(sPbrShadowLightRequest);
}

void begin_pbr_shadow_map_capture() noexcept {
  begin_pbr_shadow_map_capture_slot(0);
}

void finish_pbr_shadow_map_capture(uint32_t drawCount, uint32_t capturedSlotCount, uint32_t capturedSlotMask) noexcept {
  using namespace pbr_internal;
  const bool previousAvailable = sPbrShadowMapAvailable;
  const u32 previousDrawCount = sPbrShadowMapDrawCount;
  sPbrShadowMapDrawCount = drawCount;
  if (drawCount == 0 || capturedSlotCount == 0 || capturedSlotMask == 0) {
    sPbrShadowMapAvailable = false;
    sPbrShadowMapCapturedSlotCount = 0;
    sPbrShadowMapCapturedSlotMask = 0;
    sPbrShadowMapSlotDrawCounts = {};
    sPbrShadowMapPendingSlotMask = 0;
    sPbrShadowMapRefreshPending = false;
    sPbrShadowFailedCaptureFrames = 0;
    if (previousAvailable != sPbrShadowMapAvailable || previousDrawCount != sPbrShadowMapDrawCount) {
      g_gxState.stateDirty = true;
    }
    return;
  }

  const u32 activeMask = pbr_shadow_active_slot_mask();
  const u32 perCapturedSlotDrawCount = capturedSlotCount == 0 ? 0 : drawCount / capturedSlotCount;
  for (u32 slot = 0; slot < PbrShadowAtlasSlotCount; ++slot) {
    const u32 slotBit = 1u << slot;
    if ((activeMask & slotBit) == 0) {
      sPbrShadowMapSlotDrawCounts[slot] = 0;
    } else if ((capturedSlotMask & slotBit) != 0) {
      sPbrShadowMapSlotDrawCounts[slot] = perCapturedSlotDrawCount;
    }
  }
  sPbrShadowMapCapturedSlotMask = (sPbrShadowMapCapturedSlotMask | capturedSlotMask) & activeMask;
  sPbrShadowMapCapturedSlotCount = pbr_count_bits(sPbrShadowMapCapturedSlotMask);
  sPbrShadowMapPendingSlotMask &= ~capturedSlotMask;
  sPbrShadowMapPendingSlotMask &= activeMask;
  if (const auto* activeRequest = pbr_active_shadow_request()) {
    sPbrShadowLightRequest = *activeRequest;
    pbr_build_shadow_matrix_from_request(sPbrShadowLightRequest);
  }
  sPbrShadowMapAvailable = sPbrShadowMapEnabled && sPbrShadowMapResourcesReady && sPbrShadowMapMatrixValid &&
                            sPbrShadowLightRequest.valid && sPbrShadowMapCapturedSlotMask != 0;
  sPbrShadowMapRefreshPending = sPbrShadowMapPendingSlotMask != 0;
  sPbrShadowFailedCaptureFrames = 0;
  if (previousAvailable != sPbrShadowMapAvailable || previousDrawCount != sPbrShadowMapDrawCount) {
    g_gxState.stateDirty = true;
  }
}

void finish_pbr_shadow_map_capture(uint32_t drawCount, uint32_t capturedSlotCount) noexcept {
  const uint32_t capturedSlotMask =
      capturedSlotCount >= pbr_internal::PbrShadowAtlasSlotCount
          ? ((1u << pbr_internal::PbrShadowAtlasSlotCount) - 1u)
          : ((1u << capturedSlotCount) - 1u);
  finish_pbr_shadow_map_capture(drawCount, capturedSlotCount, capturedSlotMask);
}

void finish_pbr_shadow_map_capture(uint32_t drawCount) noexcept {
  finish_pbr_shadow_map_capture(drawCount, drawCount > 0 ? 1u : 0u);
}

void patch_pbr_shadow_uniform(uint8_t* uniformData, const ProbeUniformPatchInfo& patch) noexcept {
  using namespace pbr_internal;
  if (uniformData == nullptr || !patch.eligible || !sPbrShadowMapMatrixValid || !sProbeCamera.valid) {
    return;
  }
  if (patch.projectionOffset + sizeof(Mat4x4<float>) > patch.uniformSize ||
      patch.pnMtxOffset + sizeof(Mat3x4<float>) * (MaxPnMtx + MaxTexMtx) > patch.uniformSize ||
      patch.nrmMtxOffset + sizeof(Mat3x4<float>) * MaxPnMtx > patch.uniformSize) {
    return;
  }

  auto* viewport = reinterpret_cast<float*>(uniformData + 8);
  const float size = static_cast<float>(sPbrShadowMapSize);
  viewport[0] = size;
  viewport[1] = size;
  viewport[2] = size;
  viewport[3] = size;
  std::memcpy(uniformData + patch.projectionOffset, &sPbrShadowProjection, sizeof(sPbrShadowProjection));

  const Mat3x4<float> shadowViewFromSource = pbr_mul_affine(sPbrShadowViewFromWorld, sProbeCamera.sourceInvView);
  auto* posMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.pnMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    posMtx[i] = pbr_mul_affine(shadowViewFromSource, posMtx[i]);
  }

  auto* nrmMtx = reinterpret_cast<Mat3x4<float>*>(uniformData + patch.nrmMtxOffset);
  for (u32 i = 0; i < MaxPnMtx; ++i) {
    nrmMtx[i] = pbr_mul_linear(shadowViewFromSource, nrmMtx[i]);
  }
}

Vec4<float> pbr_shadow_params() noexcept {
  using namespace pbr_internal;
  const bool receiverReady = ensure_pbr_shadow_fallback_resources();
  sPbrShadowReceiverSamplingReady = receiverReady;
  const bool enabled = receiverReady && sPbrShadowMapEnabled && sPbrShadowMapAvailable && sPbrShadowMapResourcesReady &&
                       sPbrShadowMapMatrixValid && sPbrShadowLightRequest.valid && sProbeCamera.valid &&
                       sPbrShadowMapCapturedSlotMask != 0;
  return {
      enabled ? 1.0f : 0.0f,
      sPbrShadowMapStrength,
      UseReversedZ ? sPbrShadowMapBias : -sPbrShadowMapBias,
      static_cast<float>(std::max(sPbrShadowMapSize, 1u)),
  };
}

Vec4<float> pbr_shadow_storage_params() noexcept {
  using namespace pbr_internal;

  const u32 slotCount = pbr_shadow_active_slot_count();
  if (!sPbrShadowMapEnabled || !sProbeCamera.valid || slotCount == 0) {
    sPbrShadowDescriptorStorageOffset = 0;
    sPbrShadowDescriptorStorageSize = 0;
    return {};
  }

  const u32 frame = gfx::current_frame();
  if (sPbrShadowDescriptorStorageFrame != frame || sPbrShadowDescriptorStorageSize == 0) {
    std::array<PbrShadowDescriptor, PbrShadowAtlasSlotCount> descriptors{};
    const Mat3x4<float> identityView{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };

    for (u32 slot = 0; slot < slotCount; ++slot) {
      const bool slotCaptured = (sPbrShadowMapCapturedSlotMask & (1u << slot)) != 0;
      const bool requestValid =
          slot < sPbrShadowLightRequestCount && sPbrShadowLightRequests[slot].valid &&
          sPbrShadowLightRequests[slot].shadowType != AURORA_PBR_SHADOW_TYPE_NONE &&
          sPbrShadowLightRequests[slot].shadowType != AURORA_PBR_SHADOW_TYPE_POINT;
      const Mat3x4<float> viewFromSource =
          sProbeCamera.valid ? pbr_mul_affine(sPbrShadowSlotViewFromWorld[slot], sProbeCamera.sourceInvView)
                             : identityView;
      const Mat4x4<float>& projection = sPbrShadowSlotProjection[slot];

      auto& descriptor = descriptors[slot];
      descriptor.viewFromSource0 = {viewFromSource.m0[0], viewFromSource.m0[1], viewFromSource.m0[2],
                                    viewFromSource.m0[3]};
      descriptor.viewFromSource1 = {viewFromSource.m1[0], viewFromSource.m1[1], viewFromSource.m1[2],
                                    viewFromSource.m1[3]};
      descriptor.viewFromSource2 = {viewFromSource.m2[0], viewFromSource.m2[1], viewFromSource.m2[2],
                                    viewFromSource.m2[3]};
      descriptor.projection0 = {projection.m0[0], projection.m0[1], projection.m0[2], projection.m0[3]};
      descriptor.projection1 = {projection.m1[0], projection.m1[1], projection.m1[2], projection.m1[3]};
      descriptor.projection2 = {projection.m2[0], projection.m2[1], projection.m2[2], projection.m2[3]};
      descriptor.projection3 = {projection.m3[0], projection.m3[1], projection.m3[2], projection.m3[3]};
      const bool localProjected =
          requestValid && sPbrShadowLightRequests[slot].shadowType == AURORA_PBR_SHADOW_TYPE_LOCAL_PROJECTED;
      const float projectionFade = localProjected ? PbrShadowLocalProjectionFade : PbrShadowDirectionalProjectionFade;
      descriptor.params = {requestValid && slotCaptured ? 1.0f : 0.0f, static_cast<float>(slot), projectionFade,
                           static_cast<float>(requestValid ? sPbrShadowLightRequests[slot].shadowType
                                                           : AURORA_PBR_SHADOW_TYPE_NONE)};
    }

    const u32 byteSize = slotCount * static_cast<u32>(sizeof(PbrShadowDescriptor));
    const auto range = gfx::push_storage(reinterpret_cast<const uint8_t*>(descriptors.data()), byteSize);
    sPbrShadowDescriptorStorageFrame = frame;
    sPbrShadowDescriptorStorageOffset = range.offset;
    sPbrShadowDescriptorStorageSize = byteSize;
  }

  return {
      static_cast<float>(sPbrShadowDescriptorStorageOffset),
      static_cast<float>(sPbrShadowDescriptorStorageSize),
      static_cast<float>(sizeof(PbrShadowDescriptor)),
      static_cast<float>(slotCount),
  };
}

Mat3x4<float> pbr_shadow_view_from_source() noexcept {
  using namespace pbr_internal;
  if (!sPbrShadowMapMatrixValid || !sProbeCamera.valid) {
    return {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
  }
  return pbr_mul_affine(sPbrShadowViewFromWorld, sProbeCamera.sourceInvView);
}

Mat4x4<float> pbr_shadow_projection() noexcept { return pbr_internal::sPbrShadowProjection; }

std::array<Mat3x4<float>, 4> pbr_shadow_view_from_source_slots() noexcept {
  using namespace pbr_internal;
  std::array<Mat3x4<float>, 4> slots{};
  const Mat3x4<float> identity{
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
  };
  for (u32 slot = 0; slot < slots.size(); ++slot) {
    slots[slot] = sProbeCamera.valid ? pbr_mul_affine(sPbrShadowSlotViewFromWorld[slot], sProbeCamera.sourceInvView)
                                     : identity;
  }
  return slots;
}

std::array<Mat4x4<float>, 4> pbr_shadow_projection_slots() noexcept {
  std::array<Mat4x4<float>, 4> slots{};
  for (u32 slot = 0; slot < slots.size(); ++slot) {
    slots[slot] = pbr_internal::sPbrShadowSlotProjection[slot];
  }
  return slots;
}

const wgpu::TextureView& pbr_shadow_receiver_view(uint32_t slot) noexcept {
  using namespace pbr_internal;
  ensure_pbr_shadow_fallback_resources();
  if (sPbrShadowMapAvailable && sPbrShadowMapResourcesReady && slot < PbrShadowAtlasSlotCount &&
      sPbrShadowMapSlotViews[slot]) {
    return sPbrShadowMapSlotViews[slot];
  }
  return sPbrShadowFallbackView;
}

const wgpu::TextureView& pbr_shadow_receiver_view() noexcept {
  using namespace pbr_internal;
  const u32 slot = sPbrShadowLightRequest.valid ? std::min(sPbrShadowLightRequest.atlasSlot, PbrShadowAtlasSlotCount - 1)
                                                : 0;
  return pbr_shadow_receiver_view(slot);
}

const wgpu::Sampler& pbr_shadow_receiver_sampler() noexcept {
  using namespace pbr_internal;
  ensure_pbr_shadow_fallback_resources();
  return sPbrShadowMapSampler;
}

void configure_pbr_material_override(ShaderConfig& config, const ShaderInfo& info) noexcept {
  config.pbrFlags = 0;
  config.pbrTexMapId = GX_TEXMAP_NULL;
  config.pbrTexCoordId = GX_TEXCOORD_NULL;
  config.pbrChannelId = GX_COLOR_NULL;

  if (!enablePbrMaterialOverride || config.lineMode != 0) {
    return;
  }

  const auto choosePbrChannel = [&]() -> GXChannelID {
    for (u32 i = 0; i < config.tevStageCount; ++i) {
      const auto channel = config.tevStages[i].channelId;
      if (channel != GX_COLOR_NULL && channel != GX_COLOR_ZERO && channel != GX_ALPHA_BUMP &&
          channel != GX_ALPHA_BUMPN) {
        return channel;
      }
    }
    if (config.colorChannels[GX_COLOR0].lightingEnabled || config.colorChannels[GX_ALPHA0].lightingEnabled) {
      return GX_COLOR0A0;
    }
    if (config.colorChannels[GX_COLOR1].lightingEnabled || config.colorChannels[GX_ALPHA1].lightingEnabled) {
      return GX_COLOR1A1;
    }
    return GX_COLOR0A0;
  };

  const auto choosePbrTexCoord = [&]() -> GXTexCoordID {
    for (u32 i = 0; i < config.tcgs.size(); ++i) {
      if (config.tcgs[i].src != GX_MAX_TEXGENSRC) {
        return static_cast<GXTexCoordID>(i);
      }
    }
    return GX_TEXCOORD_NULL;
  };

  if (pbrMaterialOverrideActive) {
    config.pbrFlags =
        PbrMaterialEnabled | PbrMaterialUsePrevAlbedo | PbrMaterialPrevAlbedoIsLit | PbrMaterialUseGlobalMaps;
    if (pbrMaterialEmissive.w() > 0.0f) {
      config.pbrFlags |= PbrMaterialHasConstantEmissive;
    }
    if (pbrMaterialRmaos) {
      config.pbrFlags |= PbrMaterialHasRmaos;
    }
    if (pbrMaterialRoughness) {
      config.pbrFlags |= PbrMaterialHasRoughness;
    }
    if (pbrMaterialMetallic) {
      config.pbrFlags |= PbrMaterialHasMetallic;
    }
    if (pbrMaterialAo) {
      config.pbrFlags |= PbrMaterialHasAo;
    }
    if (pbrMaterialSpecular) {
      config.pbrFlags |= PbrMaterialHasSpecular;
    }
    if (pbrMaterialNormal) {
      config.pbrFlags |= PbrMaterialHasNormal;
    }
    if (pbrMaterialEmissiveMap) {
      config.pbrFlags |= PbrMaterialHasEmissive;
    }
    config.pbrTexMapId = 0;
    config.pbrTexCoordId = underlying(choosePbrTexCoord());
    config.pbrChannelId = underlying(choosePbrChannel());
    return;
  }

  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    if (stage.texMapId == GX_TEXMAP_NULL || stage.texCoordId == GX_TEXCOORD_NULL) {
      continue;
    }

    const u32 texMap = underlying(stage.texMapId);
    if (texMap >= MaxTextures || !info.sampledTextures.test(texMap)) {
      continue;
    }

    const auto& tex = g_gxState.textures[texMap];
    if (!tex) {
      continue;
    }

    const auto& ref = *tex.ref;
    if (!ref.pbrRmaos && !ref.pbrRoughness && !ref.pbrMetallic && !ref.pbrAo && !ref.pbrSpecular && !ref.pbrNormal &&
        !ref.pbrEmissive) {
      continue;
    }

    config.pbrFlags = PbrMaterialEnabled | PbrMaterialUsePrevAlbedo | PbrMaterialPrevAlbedoIsLit;
    if (ref.pbrRmaos) {
      config.pbrFlags |= PbrMaterialHasRmaos;
    }
    if (ref.pbrRoughness) {
      config.pbrFlags |= PbrMaterialHasRoughness;
    }
    if (ref.pbrMetallic) {
      config.pbrFlags |= PbrMaterialHasMetallic;
    }
    if (ref.pbrAo) {
      config.pbrFlags |= PbrMaterialHasAo;
    }
    if (ref.pbrSpecular) {
      config.pbrFlags |= PbrMaterialHasSpecular;
    }
    if (tex.ref->pbrNormal) {
      config.pbrFlags |= PbrMaterialHasNormal;
    }
    if (tex.ref->pbrEmissive) {
      config.pbrFlags |= PbrMaterialHasEmissive;
    }
    config.pbrTexMapId = texMap;
    config.pbrTexCoordId = underlying(stage.texCoordId);
    config.pbrChannelId = underlying(stage.channelId);
    return;
  }
}
} // namespace aurora::gx

namespace {

void upload_pbr_enhanced_light_storage() {
  using namespace aurora::gx;

  if (pbrEnhancedLightCount > 0) {
    const auto range =
        aurora::gfx::push_storage(reinterpret_cast<const uint8_t*>(pbrEnhancedLights.data()),
                                  pbrEnhancedLightCount * sizeof(PbrEnhancedLight));
    pbrEnhancedLightStorageOffset = range.offset;
    pbrEnhancedLightStorageSize = pbrEnhancedLightCount * sizeof(PbrEnhancedLight);
  } else {
    pbrEnhancedLightStorageOffset = 0;
    pbrEnhancedLightStorageSize = 0;
  }
}

} // namespace

void aurora_enable_pbr(bool enabled) {
  aurora::gx::enablePbrMaterialOverride = enabled;
  aurora::gx::g_gxState.stateDirty = true;
}

bool aurora_pbr_enabled() { return aurora::gx::enablePbrMaterialOverride; }

void aurora_set_pbr_light_params(float ambient, float ambient_specular, float fill_intensity) {
  aurora::gx::pbrParams = {ambient, ambient_specular, fill_intensity, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_material_params(float diffuse_scale, float specular_scale) {
  aurora::gx::pbrScales = {diffuse_scale, specular_scale, aurora::gx::pbrScales.z(), 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_debug_mode(AuroraPbrDebugMode mode) {
  const int clampedMode =
      std::clamp(static_cast<int>(mode), static_cast<int>(AURORA_PBR_DEBUG_OFF),
                 static_cast<int>(AURORA_PBR_DEBUG_DYNAMIC_GI));
  aurora::gx::pbrScales = {aurora::gx::pbrScales.x(), aurora::gx::pbrScales.y(), static_cast<float>(clampedMode),
                           0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_normal_params(float strength, bool flip_y, bool invert_handedness) {
  const float normalStrength = strength < 0.0f ? 0.0f : strength;
  aurora::gx::pbrNormalParams = {normalStrength, flip_y ? -1.0f : 1.0f, invert_handedness ? -1.0f : 1.0f, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ambient_gradient_params(float sky, float ground, float horizon, float environment_tint) {
  aurora::gx::pbrAmbientGradient = {sky, ground, horizon, environment_tint};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_indirect_occlusion_params(float strength, float horizon, float specular) {
  aurora::gx::pbrIndirectOcclusion = {std::clamp(strength, 0.0f, 1.0f), std::clamp(horizon, 0.0f, 1.0f),
                                      std::clamp(specular, 0.0f, 1.0f), 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_dynamic_gi_params(bool enabled, float strength, float normal_wrap, float albedo_influence) {
  aurora::gx::pbrDynamicGiParams = {enabled ? 1.0f : 0.0f, std::max(strength, 0.0f),
                                    std::clamp(normal_wrap, 0.0f, 1.0f),
                                    std::clamp(albedo_influence, 0.0f, 1.0f)};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ibl_params(bool enabled, float diffuse_strength, float specular_strength) {
  aurora::gx::pbrIblParams = {enabled ? 1.0f : 0.0f, std::max(diffuse_strength, 0.0f),
                              std::max(specular_strength, 0.0f), aurora::gx::pbrIblParams.w()};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_ibl_source(AuroraPbrIblSource source) {
  switch (source) {
  case AURORA_PBR_IBL_SOURCE_PROBE:
  case AURORA_PBR_IBL_SOURCE_AUTHORED:
  case AURORA_PBR_IBL_SOURCE_FALLBACK:
    aurora::gx::pbr_internal::sRequestedPbrIblSource = source;
    break;
  default:
    aurora::gx::pbr_internal::sRequestedPbrIblSource = AURORA_PBR_IBL_SOURCE_PROBE;
    break;
  }

  aurora::gx::pbr_internal::select_active_pbr_ibl_set();
  aurora::gx::pbr_internal::clear_pbr_ibl_blend();
}

void aurora_set_pbr_enhanced_lighting(bool enabled, AuroraPbrEnhancedLightFalloff falloff, uint32_t max_light_count,
                                      float intensity_scale, bool debug_enabled) {
  using namespace aurora::gx;

  pbrEnhancedLightsEnabled = enabled;
  pbrEnhancedLightsDebugEnabled = debug_enabled;
  pbrEnhancedLightFalloff = falloff == AURORA_PBR_ENHANCED_LIGHT_FALLOFF_INVERSE_SQUARE
                                ? AURORA_PBR_ENHANCED_LIGHT_FALLOFF_INVERSE_SQUARE
                                : AURORA_PBR_ENHANCED_LIGHT_FALLOFF_LEGACY_RADIUS;
  pbrEnhancedLightMaxCount = std::clamp<u32>(max_light_count, 1, PbrMaxEnhancedLights);
  pbrEnhancedLightIntensityScale = std::max(intensity_scale, 0.0f);
  if (!enabled) {
    pbrEnhancedLightCount = 0;
    pbrSubmittedSceneLightCount = 0;
    pbrSceneLightsApiBacked = false;
    upload_pbr_enhanced_light_storage();
  } else if (pbrEnhancedLightCount > pbrEnhancedLightMaxCount) {
    pbrEnhancedLightCount = pbrEnhancedLightMaxCount;
    upload_pbr_enhanced_light_storage();
  }
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_enhanced_lights(const AuroraPbrEnhancedLight* lights, uint32_t light_count) {
  using namespace aurora::gx;

  pbrSceneLightsApiBacked = false;
  pbrSubmittedSceneLightCount = lights != nullptr ? light_count : 0;
  pbrEnhancedLightCount = lights != nullptr && pbrEnhancedLightsEnabled
                              ? std::min<u32>(light_count, std::min(pbrEnhancedLightMaxCount, PbrMaxEnhancedLights))
                              : 0;
  for (u32 i = 0; i < PbrMaxEnhancedLights; ++i) {
    if (i < pbrEnhancedLightCount) {
      const auto& light = lights[i];
      pbrEnhancedLights[i].posRadius = {light.position[0], light.position[1], light.position[2],
                                        std::max(light.radius, 1.0f)};
      pbrEnhancedLights[i].colorIntensity = {std::max(light.color[0], 0.0f), std::max(light.color[1], 0.0f),
                                             std::max(light.color[2], 0.0f), std::max(light.intensity, 0.0f)};
      pbrEnhancedLights[i].dirType = {0.0f, 0.0f, 0.0f, static_cast<float>(AURORA_SCENE_LIGHT_POINT)};
      pbrEnhancedLights[i].shadowParams = {};
    } else {
      pbrEnhancedLights[i] = {};
    }
  }
  upload_pbr_enhanced_light_storage();
  g_gxState.stateDirty = true;
}

void aurora_set_scene_lights(const AuroraSceneLight* lights, uint32_t light_count) {
  using namespace aurora::gx;

  pbrSceneLightsApiBacked = true;
  pbrSubmittedSceneLightCount = lights != nullptr ? light_count : 0;
  pbrEnhancedLightCount = lights != nullptr && pbrEnhancedLightsEnabled
                              ? std::min<u32>(light_count, std::min(pbrEnhancedLightMaxCount, PbrMaxEnhancedLights))
                              : 0;
  for (u32 i = 0; i < PbrMaxEnhancedLights; ++i) {
    if (i < pbrEnhancedLightCount) {
      const auto& light = lights[i];
      pbrEnhancedLights[i].posRadius = {light.position[0], light.position[1], light.position[2],
                                        std::max(light.radius, 1.0f)};
      pbrEnhancedLights[i].colorIntensity = {std::max(light.color[0], 0.0f), std::max(light.color[1], 0.0f),
                                             std::max(light.color[2], 0.0f), std::max(light.intensity, 0.0f)};
      pbrEnhancedLights[i].dirType = {light.direction[0], light.direction[1], light.direction[2],
                                      static_cast<float>(light.type)};
      pbrEnhancedLights[i].shadowParams = pbr_internal::pbr_scene_light_shadow_params(light);
    } else {
      pbrEnhancedLights[i] = {};
    }
  }
  upload_pbr_enhanced_light_storage();
  g_gxState.stateDirty = true;
}

const AuroraPbrEnhancedLightingStatus* aurora_get_pbr_enhanced_lighting_status() {
  using namespace aurora::gx;

  static AuroraPbrEnhancedLightingStatus status{};
  status = {};
  status.enabled = pbrEnhancedLightsEnabled;
  status.debugEnabled = pbrEnhancedLightsDebugEnabled;
  status.lightCount = pbrEnhancedLightCount;
  status.maxLightCount = pbrEnhancedLightMaxCount;
  status.submittedSceneLightCount = pbrSubmittedSceneLightCount;
  status.sceneLightApiBacked = pbrSceneLightsApiBacked;
  status.storageBacked = pbrEnhancedLightStorageSize > 0;
  status.storageByteSize = pbrEnhancedLightStorageSize;
  status.falloff = pbrEnhancedLightFalloff;
  status.intensityScale = pbrEnhancedLightIntensityScale;
  status.dynamicGiEnabled = pbrDynamicGiParams.x() > 0.5f;
  status.dynamicGiStrength = pbrDynamicGiParams.y();
  status.dynamicGiNormalWrap = pbrDynamicGiParams.z();
  status.dynamicGiAlbedoInfluence = pbrDynamicGiParams.w();
  if (pbrEnhancedLightCount > 0) {
    const auto& light = pbrEnhancedLights[0];
    status.strongestLight.position[0] = light.posRadius.x();
    status.strongestLight.position[1] = light.posRadius.y();
    status.strongestLight.position[2] = light.posRadius.z();
    status.strongestLight.radius = light.posRadius.w();
    status.strongestLight.color[0] = light.colorIntensity.x();
    status.strongestLight.color[1] = light.colorIntensity.y();
    status.strongestLight.color[2] = light.colorIntensity.z();
    status.strongestLight.intensity = light.colorIntensity.w();
  }
  return &status;
}

bool aurora_pbr_probe_ibl_available() { return aurora::gx::pbr_internal::active_probe_ibl_set() != nullptr; }

const AuroraPbrIblStatus* aurora_get_pbr_ibl_status() {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  static AuroraPbrIblStatus status{};
  status = {};
  status.pbrEnabled = enablePbrMaterialOverride;
  status.iblEnabled = pbrIblParams.x() > 0.0f;
  status.requestedSource = sRequestedPbrIblSource;
  status.activeSource = active_pbr_ibl_source();
  status.fallbackAvailable = sFallbackPbrIbl.available;
  status.authoredAvailable = sAuthoredPbrIbl.available;
  status.probeAvailable = active_probe_ibl_set() != nullptr;
  status.probeCameraValid = sProbeCamera.valid;
  status.probeCaptureResourcesReady = sProbeCaptureResourcesReady;
  status.probeCaptureInProgress = sProbeCaptureCamera.valid || sProbeCaptureFace != 0;
  status.probeRefreshPending = sProbeRefreshPending;
  status.probeFilterPending = sProbeFilterPending;
  status.probeAutoRefresh = sProbeAutoRefresh;
  status.probeLocalGi = sProbeLocalGi;
  status.probeSceneStale = sProbeSceneStale;
  status.probeCacheEnabled = sProbeCacheEnabled;
  status.probeCacheHit = sProbeCacheLastHit;
  status.probeNearestCacheEnabled = sProbeNearestCacheEnabled;
  status.probeNearestCacheActive = sProbeNearestCacheActive;
  status.probeSpatialBlendEnabled = sProbeSpatialBlendEnabled;
  status.probeSpatialBlendActive = pbr_spatial_blend_active();
  status.probeBlendEnabled = sProbeBlendEnabled;
  status.probeBlendActive = pbr_transition_blend_active();
  status.probeReplayPbrVisible = sProbeReplayPbrVisible;
  status.probeCaptureFace = sProbeCaptureFace;
  status.probeCaptureDelayFrames = sProbeCaptureDelayFrames;
  status.probeFramesSinceRefresh = sProbeFramesSinceRefresh;
  status.probeReplayDraws = sProbeReplayDrawCount;
  status.probeCubeSize = PbrRuntimeProbeCubeSize;
  status.probeIrradianceSize = PbrIrradianceCubeSize;
  status.probePrefilterMipCount = PbrRuntimeProbePrefilterMipCount;
  status.probeCacheSlots = PbrProbeCacheSlotCount;
  status.probeCacheUsedSlots = used_probe_cache_slot_count();
  status.probeBlendFrames = sProbeBlendFrames;
  status.probeBlendFactor = sProbeBlendFactor;
  status.probeNearestCacheDistance = sProbeNearestCacheDistance;
  status.probeNearestCacheMaxDistance = sProbeNearestCacheMaxDistance;
  status.probeSpatialBlendFactor = sProbeSpatialBlendFactor;
  status.probeSpatialBlendDistance = sProbeSpatialBlendDistance;
  status.probeSpatialBlendMaxDistance = sProbeSpatialBlendMaxDistance;
  status.activePrefilterMipCount = active_pbr_ibl_set().prefilterMipCount;
  status.authoredGlobalLoaded = sAuthoredPbrIblGlobalLoaded;
  status.authoredStageLoaded = sAuthoredPbrIblStageLoaded;
  status.authoredRoomLoaded = sAuthoredPbrIblRoomLoaded;
  status.authoredRoom = sAuthoredPbrIblRoom;
  copy_status_string(status.authoredRoot, sAuthoredPbrIblRootPath);
  copy_status_string(status.authoredStage, sAuthoredPbrIblStage);
  copy_status_string(status.authoredSceneKey, sActivePbrIblSceneKey);
  if (sActiveProbeCacheSlot != nullptr && sActiveProbeCacheSlot->ibl.available) {
    copy_status_string(status.activeProbeSceneKey, sActiveProbeCacheSlot->sceneKey);
  } else if (active_probe_ibl_set() != nullptr) {
    copy_status_string(status.activeProbeSceneKey, sActivePbrIblSceneKey);
  }
  if (sSpatialProbeCacheSlot != nullptr && sSpatialProbeCacheSlot->ibl.available) {
    copy_status_string(status.spatialProbeSceneKey, sSpatialProbeCacheSlot->sceneKey);
  }
  return &status;
}

void aurora_set_pbr_shadow_map_params(bool enabled, uint32_t size, float strength, float bias) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  const u32 clampedSize = std::clamp<u32>(size, PbrShadowMapMinSize, PbrShadowMapMaxSize);
  if (sPbrShadowMapSize != clampedSize) {
    sPbrShadowMapSize = clampedSize;
    sPbrShadowMapResourcesReady = false;
    sPbrShadowMapTexture = {};
    sPbrShadowMapView = {};
    sPbrShadowMapSlotViews = {};
    sPbrShadowMapAvailable = false;
    sPbrShadowMapDrawCount = 0;
    sPbrShadowMapSlotDrawCounts = {};
    sPbrShadowMapCapturedSlotCount = 0;
    sPbrShadowMapCapturedSlotMask = 0;
    sPbrShadowMapPendingSlotMask = pbr_shadow_active_slot_mask();
    sPbrShadowMapRefreshPending = true;
    sPbrShadowFailedCaptureFrames = 0;
  }

  const bool wasEnabled = sPbrShadowMapEnabled;
  sPbrShadowMapEnabled = enabled;
  if (!enabled) {
    sPbrShadowMapAvailable = false;
    sPbrShadowMapDrawCount = 0;
    sPbrShadowMapSlotDrawCounts = {};
    sPbrShadowMapCapturedSlotCount = 0;
    sPbrShadowMapCapturedSlotMask = 0;
    sPbrShadowMapPendingSlotMask = 0;
    sPbrShadowMapRefreshPending = false;
    sPbrShadowFailedCaptureFrames = 0;
  } else if (!wasEnabled) {
    sPbrShadowMapPendingSlotMask = pbr_shadow_active_slot_mask();
    sPbrShadowMapRefreshPending = true;
    sPbrShadowFailedCaptureFrames = 0;
  }
  sPbrShadowMapStrength = std::clamp(strength, 0.0f, 1.0f);
  sPbrShadowMapBias = std::clamp(bias, 0.0f, 0.05f);
  ensure_pbr_shadow_map_resources();
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_shadow_map_budget(uint32_t max_active_maps, uint32_t maps_per_frame) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  const u32 clampedMaxActive = std::clamp<u32>(max_active_maps, 1u, PbrShadowAtlasSlotCount);
  const u32 clampedPerFrame = std::clamp<u32>(maps_per_frame, 1u, PbrShadowAtlasSlotCount);
  if (sPbrShadowMapMaxActiveSlots == clampedMaxActive && sPbrShadowMapSlotsPerFrame == clampedPerFrame) {
    return;
  }

  sPbrShadowMapMaxActiveSlots = clampedMaxActive;
  sPbrShadowMapSlotsPerFrame = clampedPerFrame;
  if (sPbrShadowLightRequestCount > clampedMaxActive) {
    sPbrShadowLightRequestCount = clampedMaxActive;
  }

  const u32 activeMask = pbr_shadow_active_slot_mask();
  sPbrShadowMapCapturedSlotMask &= activeMask;
  sPbrShadowMapCapturedSlotCount = pbr_count_bits(sPbrShadowMapCapturedSlotMask);
  if (sPbrShadowMapEnabled && sPbrShadowLightRequestCount > 0) {
    sPbrShadowMapPendingSlotMask = activeMask & ~sPbrShadowMapCapturedSlotMask;
    sPbrShadowMapRefreshPending = sPbrShadowMapPendingSlotMask != 0;
  } else {
    sPbrShadowMapPendingSlotMask = 0;
    sPbrShadowMapRefreshPending = false;
  }
  sPbrShadowDescriptorStorageFrame = UINT32_MAX;
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_shadow_light_request(const AuroraPbrShadowLightRequest* request) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  if (request == nullptr || !request->valid || !pbr_shadow_request_finite(*request)) {
    sPbrShadowLightRequestCount = 0;
    sPbrShadowLightRequests = {};
    if (sPbrShadowLightRequest.valid || sPbrShadowMapAvailable || sPbrShadowMapRefreshPending ||
        sPbrShadowMapDrawCount != 0 || sPbrShadowMapCapturedSlotCount != 0 || sPbrShadowMapCapturedSlotMask != 0) {
      sPbrShadowLightRequest = {};
      sPbrShadowMapAvailable = false;
      sPbrShadowMapDrawCount = 0;
      sPbrShadowMapSlotDrawCounts = {};
      sPbrShadowMapCapturedSlotCount = 0;
      sPbrShadowMapCapturedSlotMask = 0;
      sPbrShadowMapPendingSlotMask = 0;
      sPbrShadowMapRefreshPending = false;
      sPbrShadowFailedCaptureFrames = 0;
      pbr_set_shadow_identity_matrix();
      g_gxState.stateDirty = true;
    }
    return;
  }

  if (!pbr_shadow_request_needs_refresh(*request)) {
    if (!sPbrShadowMapAvailable && !sPbrShadowMapRefreshPending) {
      ++sPbrShadowFailedCaptureFrames;
      if (sPbrShadowFailedCaptureFrames >= PbrShadowFailedCaptureRetryFrames) {
        sPbrShadowFailedCaptureFrames = 0;
        sPbrShadowMapPendingSlotMask = pbr_shadow_active_slot_mask();
        sPbrShadowMapRefreshPending = true;
      }
    }
    return;
  }

  sPbrShadowLightRequest = *request;
  sPbrShadowLightRequest.valid = true;
  sPbrShadowLightRequest.atlasSlot = 0;
  sPbrShadowLightRequestCount = 1;
  sPbrShadowLightRequests = {};
  sPbrShadowLightRequests[0] = sPbrShadowLightRequest;
  sPbrShadowMapAvailable = false;
  sPbrShadowMapRefreshPending = true;
  sPbrShadowMapCapturedSlotCount = 0;
  sPbrShadowMapCapturedSlotMask = 0;
  sPbrShadowMapSlotDrawCounts = {};
  sPbrShadowMapPendingSlotMask = 1u;
  sPbrShadowFailedCaptureFrames = 0;
  pbr_build_shadow_matrix_from_request(sPbrShadowLightRequest);
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_shadow_light_requests(const AuroraPbrShadowLightRequest* requests, uint32_t request_count) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  std::array<AuroraPbrShadowLightRequest, PbrShadowRequestMaxCount> incomingRequests{};
  std::array<bool, PbrShadowRequestMaxCount> incomingUsed{};
  u32 incomingCount = 0;
  const u32 requestLimit = pbr_shadow_active_slot_limit();
  if (requests != nullptr) {
    for (u32 i = 0; i < request_count && incomingCount < PbrShadowRequestMaxCount && incomingCount < requestLimit; ++i) {
      AuroraPbrShadowLightRequest request = requests[i];
      if (!request.valid || !pbr_shadow_request_finite(request) ||
          request.shadowType == AURORA_PBR_SHADOW_TYPE_NONE) {
        continue;
      }

      request.atlasSlot = incomingCount;
      incomingRequests[incomingCount++] = request;
    }
  }

  std::array<AuroraPbrShadowLightRequest, PbrShadowRequestMaxCount> validRequests{};
  u32 validCount = 0;
  const auto appendRequest = [&](AuroraPbrShadowLightRequest request) {
    if (validCount >= requestLimit || validCount >= PbrShadowRequestMaxCount) {
      return false;
    }
    request.atlasSlot = validCount;
    validRequests[validCount++] = request;
    return true;
  };

  const u32 previousCount = pbr_shadow_active_slot_count();
  for (u32 previousSlot = 0; previousSlot < previousCount && validCount < requestLimit; ++previousSlot) {
    const AuroraPbrShadowLightRequest& previousRequest = sPbrShadowLightRequests[previousSlot];
    if (!previousRequest.valid) {
      continue;
    }
    for (u32 incomingSlot = 0; incomingSlot < incomingCount; ++incomingSlot) {
      if (incomingUsed[incomingSlot]) {
        continue;
      }
      if (!pbr_shadow_request_same_light(previousRequest, incomingRequests[incomingSlot])) {
        continue;
      }
      incomingUsed[incomingSlot] = appendRequest(incomingRequests[incomingSlot]);
      break;
    }
  }

  for (u32 incomingSlot = 0; incomingSlot < incomingCount && validCount < requestLimit; ++incomingSlot) {
    if (incomingUsed[incomingSlot]) {
      continue;
    }
    appendRequest(incomingRequests[incomingSlot]);
  }

  const bool needsRefresh = pbr_shadow_request_list_needs_refresh(validRequests, validCount);
  sPbrShadowLightRequests = validRequests;
  sPbrShadowLightRequestCount = validCount;

  const AuroraPbrShadowLightRequest* activeRequest = pbr_active_shadow_request();
  if (activeRequest == nullptr) {
    aurora_set_pbr_shadow_light_request(nullptr);
    return;
  }

  if (!needsRefresh) {
    if (!sPbrShadowMapAvailable && !sPbrShadowMapRefreshPending) {
      ++sPbrShadowFailedCaptureFrames;
      if (sPbrShadowFailedCaptureFrames >= PbrShadowFailedCaptureRetryFrames) {
        sPbrShadowFailedCaptureFrames = 0;
        sPbrShadowMapPendingSlotMask = pbr_shadow_active_slot_mask();
        sPbrShadowMapRefreshPending = true;
      }
    }
    return;
  }

  const AuroraPbrShadowLightRequest activeRequestCopy = *activeRequest;
  for (u32 i = 0; i < validCount; ++i) {
    pbr_build_shadow_matrix_from_request(sPbrShadowLightRequests[i]);
  }
  sPbrShadowLightRequest = activeRequestCopy;
  sPbrShadowMapAvailable = false;
  sPbrShadowMapRefreshPending = true;
  sPbrShadowMapCapturedSlotCount = 0;
  sPbrShadowMapCapturedSlotMask = 0;
  sPbrShadowMapSlotDrawCounts = {};
  sPbrShadowMapPendingSlotMask = pbr_shadow_slot_mask(validCount);
  sPbrShadowFailedCaptureFrames = 0;
  pbr_build_shadow_matrix_from_request(sPbrShadowLightRequest);
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_shadow_map_matrix(const float* light_view_projection_4x4) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  if (light_view_projection_4x4 == nullptr) {
    pbr_build_shadow_matrix_from_request(sPbrShadowLightRequest);
    return;
  }

  bool finite = true;
  for (u32 i = 0; i < sPbrShadowLightViewProjection.size(); ++i) {
    const float value = light_view_projection_4x4[i];
    sPbrShadowLightViewProjection[i] = value;
    finite = finite && pbr_finite(value);
  }
  sPbrShadowMapMatrixValid = finite;
  if (!finite) {
    pbr_set_shadow_identity_matrix();
  }
  g_gxState.stateDirty = true;
}

const AuroraPbrShadowMapStatus* aurora_get_pbr_shadow_map_status() {
  using namespace aurora::gx::pbr_internal;

  static AuroraPbrShadowMapStatus status{};
  ensure_pbr_shadow_map_resources();
  status = {};
  status.enabled = sPbrShadowMapEnabled;
  status.resourcesReady = sPbrShadowMapResourcesReady;
  status.mapAvailable = sPbrShadowMapAvailable;
  status.lightRequestValid = sPbrShadowLightRequest.valid;
  status.matrixValid = sPbrShadowMapMatrixValid;
  status.casterPassReady = sPbrShadowMapEnabled && sPbrShadowCasterPassReady;
  status.receiverSamplingReady = sPbrShadowMapEnabled && sPbrShadowReceiverSamplingReady;
  status.refreshPending = sPbrShadowMapRefreshPending;
  status.size = sPbrShadowMapSize;
  status.drawCount = sPbrShadowMapDrawCount;
  status.requestCount = sPbrShadowLightRequestCount;
  status.requestMask = pbr_shadow_slot_mask(sPbrShadowLightRequestCount);
  status.atlasSlotCount = PbrShadowAtlasSlotCount;
  status.maxActiveSlots = sPbrShadowMapMaxActiveSlots;
  status.slotsPerFrame = sPbrShadowMapSlotsPerFrame;
  status.activeAtlasSlot = sPbrShadowLightRequest.valid ? sPbrShadowLightRequest.atlasSlot : 0;
  status.activeShadowType = sPbrShadowLightRequest.valid ? sPbrShadowLightRequest.shadowType
                                                         : AURORA_PBR_SHADOW_TYPE_NONE;
  status.capturedSlotCount = sPbrShadowMapCapturedSlotCount;
  status.capturedSlotMask = sPbrShadowMapCapturedSlotMask;
  status.pendingSlotMask = sPbrShadowMapPendingSlotMask;
  std::memcpy(status.slotDrawCounts, sPbrShadowMapSlotDrawCounts.data(), sizeof(status.slotDrawCounts));
  status.strength = sPbrShadowMapStrength;
  status.bias = sPbrShadowMapBias;
  std::memcpy(status.lightViewProjection, sPbrShadowLightViewProjection.data(),
              sizeof(status.lightViewProjection));
  status.lightRequest = sPbrShadowLightRequest;
  std::memcpy(status.requests, sPbrShadowLightRequests.data(), sizeof(status.requests));
  return &status;
}

void aurora_request_pbr_shadow_map_refresh() {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  if (!sPbrShadowMapEnabled || sPbrShadowLightRequestCount == 0) {
    return;
  }

  const u32 activeMask = pbr_shadow_active_slot_mask();
  sPbrShadowMapPendingSlotMask = activeMask;
  sPbrShadowMapRefreshPending = sPbrShadowMapPendingSlotMask != 0;
  sPbrShadowFailedCaptureFrames = 0;
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_probe_auto_refresh(bool enabled) {
  aurora::gx::pbr_internal::sProbeAutoRefresh = enabled;
  if (enabled && aurora::gx::pbr_internal::active_probe_ibl_set() == nullptr) {
    aurora::gx::pbr_internal::sProbeRefreshPending = true;
  }
}

void aurora_set_pbr_local_probe_gi(bool enabled) {
  aurora::gx::pbr_internal::sProbeLocalGi = enabled;
  if (!enabled) {
    aurora::gx::pbr_internal::sProbeSceneStale = false;
  }
}

void aurora_set_pbr_probe_cache(bool enabled) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  sProbeCacheEnabled = enabled;
  if (!enabled) {
    sActiveProbeCacheSlot = nullptr;
    sProbeCaptureCacheSlot = nullptr;
    sProbeFilterCacheSlot = nullptr;
    sProbeCacheLastHit = false;
    sProbeNearestCacheActive = false;
    sProbeNearestCacheDistance = 0.0f;
    clear_pbr_probe_spatial_blend();
    sProbeSceneStale = false;
  }
  select_active_pbr_ibl_set();
  clear_pbr_ibl_blend();
}

void aurora_set_pbr_probe_nearest_cache(bool enabled, float max_distance) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  sProbeNearestCacheEnabled = enabled;
  sProbeNearestCacheMaxDistance = std::max(max_distance, 0.0f);
  if (!enabled && sProbeNearestCacheActive) {
    sActiveProbeCacheSlot = nullptr;
    sProbeNearestCacheActive = false;
    sProbeNearestCacheDistance = 0.0f;
    select_active_pbr_ibl_set();
    clear_pbr_ibl_blend();
  } else {
    g_gxState.stateDirty = true;
  }
}

void aurora_set_pbr_probe_spatial_blending(bool enabled, float max_distance) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;
  sProbeSpatialBlendEnabled = enabled;
  sProbeSpatialBlendMaxDistance = std::max(max_distance, 0.0f);
  if (!enabled) {
    clear_pbr_probe_spatial_blend();
  } else {
    update_pbr_probe_spatial_blend();
  }
  g_gxState.stateDirty = true;
}

void aurora_set_pbr_probe_blending(bool enabled, uint32_t frames) {
  using namespace aurora::gx;
  using namespace aurora::gx::pbr_internal;

  sProbeBlendEnabled = enabled;
  sProbeBlendFrames = std::clamp<u32>(frames, 1, 600);
  if (!enabled) {
    clear_pbr_ibl_blend();
  } else {
    update_pbr_ibl_blend_uniform();
    g_gxState.stateDirty = true;
  }
}

void aurora_request_pbr_probe_refresh() {
  aurora::gx::pbr_internal::sProbeCaptureFace = 0;
  aurora::gx::pbr_internal::sProbeCaptureDelayFrames = 0;
  aurora::gx::pbr_internal::sProbeRefreshPending = true;
}

void aurora_set_pbr_probe_camera_matrices(const float* source_view_3x4, const float* source_inv_view_3x4,
                                          const float* projection_4x4) {
  auto& camera = aurora::gx::pbr_internal::sProbeCamera;
  camera.sourceView = aurora::gx::pbr_internal::pbr_load_mtx3x4(source_view_3x4);
  camera.sourceInvView = aurora::gx::pbr_internal::pbr_load_mtx3x4(source_inv_view_3x4);
  camera.projection = aurora::gx::pbr_internal::pbr_projection_from_gx_mtx(projection_4x4);
  camera.valid = source_view_3x4 != nullptr && source_inv_view_3x4 != nullptr && projection_4x4 != nullptr;
  if (camera.valid) {
    aurora::gx::pbr_internal::pbr_update_probe_face_matrices();
    if (aurora::gx::pbr_internal::active_probe_ibl_set() == nullptr) {
      aurora::gx::pbr_internal::sProbeRefreshPending = true;
    }
    if (aurora::gx::pbr_internal::sProbeRefreshPending &&
        (aurora::gx::pbr_internal::sActiveProbeCacheSlot == nullptr ||
         aurora::gx::pbr_internal::sActiveProbeCacheSlot->sceneKey !=
             aurora::gx::pbr_internal::sActivePbrIblSceneKey)) {
      auto* nearest =
          aurora::gx::pbr_internal::find_nearest_probe_cache_slot(aurora::gx::pbr_internal::sActivePbrIblSceneKey);
      if (nearest != nullptr && nearest != aurora::gx::pbr_internal::sActiveProbeCacheSlot) {
        auto* previousIbl = aurora::gx::pbr_internal::sActivePbrIbl;
        aurora::gx::pbr_internal::activate_nearest_probe_cache_slot(
            aurora::gx::pbr_internal::sActivePbrIblSceneKey, previousIbl, true);
      }
    }
    aurora::gx::pbr_internal::update_pbr_probe_spatial_blend();
  }
}

void aurora_set_pbr_ibl_scene(const char* stage_name, int room_no) {
  const auto root = aurora::gx::pbr_internal::pbr_ibl_root_path();
  auto* previousIbl = aurora::gx::pbr_internal::sActivePbrIbl;
  std::string stage = stage_name != nullptr ? stage_name : "";
  if (!stage.empty() && !aurora::gx::pbr_internal::pbr_is_safe_scene_component(stage)) {
    stage.clear();
  }

  const std::string key = fmt::format("{}:{}", stage, room_no);
  if (key == aurora::gx::pbr_internal::sActivePbrIblSceneKey) {
    return;
  }
  aurora::gx::pbr_internal::sActivePbrIblSceneKey = key;
  aurora::gx::pbr_internal::sAuthoredPbrIblRootPath = root.empty() ? std::string{} : root.string();
  aurora::gx::pbr_internal::sAuthoredPbrIblStage = stage;
  aurora::gx::pbr_internal::sAuthoredPbrIblRoom = room_no;
  aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded = false;
  aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded = false;
  aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded = false;

  aurora::gx::pbr_internal::load_fallback_pbr_ibl_textures(aurora::gx::pbr_internal::sAuthoredPbrIbl);
  aurora::gx::pbr_internal::sAuthoredPbrIbl.available = false;
  bool loaded = false;
  if (!root.empty()) {
    aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded =
        aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / "global",
                                                                       aurora::gx::pbr_internal::sAuthoredPbrIbl);
    loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblGlobalLoaded;
    if (!stage.empty()) {
      aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded =
          aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(root / stage,
                                                                         aurora::gx::pbr_internal::sAuthoredPbrIbl);
      loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblStageLoaded;
      if (room_no >= 0 &&
          (aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded =
               aurora::gx::pbr_internal::load_authored_pbr_ibl_from_directory(
                   root / stage / fmt::format("room_{}", room_no), aurora::gx::pbr_internal::sAuthoredPbrIbl))) {
        loaded = loaded || aurora::gx::pbr_internal::sAuthoredPbrIblRoomLoaded;
      }
    }
  }

  aurora::gx::pbr_internal::sAuthoredPbrIbl.available = loaded;
  auto* cachedProbe = aurora::gx::pbr_internal::sProbeCacheEnabled
                          ? aurora::gx::pbr_internal::find_probe_cache_slot(key)
                          : nullptr;
  bool nearestProbeSelected = false;
  aurora::gx::pbr_internal::sProbeCacheLastHit = cachedProbe != nullptr;
  aurora::gx::pbr_internal::sProbeNearestCacheActive = false;
  aurora::gx::pbr_internal::sProbeNearestCacheDistance = 0.0f;
  if (cachedProbe != nullptr) {
    aurora::gx::pbr_internal::sActiveProbeCacheSlot = cachedProbe;
    cachedProbe->lastUsedSerial = ++aurora::gx::pbr_internal::sProbeCacheSerial;
    aurora::gx::pbr_internal::sProbeLastCompletedCamera = cachedProbe->camera;
    aurora::gx::pbr_internal::sProbeSceneStale = false;
  } else if (aurora::gx::pbr_internal::sProbeCacheEnabled &&
             aurora::gx::pbr_internal::activate_nearest_probe_cache_slot(key, previousIbl, false)) {
    nearestProbeSelected = true;
  } else if (aurora::gx::pbr_internal::sProbeLocalGi) {
    aurora::gx::pbr_internal::sProbeSceneStale = aurora::gx::pbr_internal::active_probe_ibl_set() != nullptr;
  } else {
    aurora::gx::pbr_internal::sProbePbrIbl.available = false;
    aurora::gx::pbr_internal::sActiveProbeCacheSlot = nullptr;
    aurora::gx::pbr_internal::sProbeSceneStale = false;
  }
  aurora::gx::pbr_internal::sProbeCaptureFace = 0;
  aurora::gx::pbr_internal::sProbeCaptureDelayFrames = 0;
  if (cachedProbe == nullptr && !nearestProbeSelected) {
    aurora::gx::pbr_internal::sProbeLastCompletedCamera.valid = false;
  }
  aurora::gx::pbr_internal::sProbeRefreshPending = true;
  if (loaded) {
    PbrLog.info("Loaded authored PBR IBL set for stage '{}' room {}", stage, room_no);
  }
  aurora::gx::pbr_internal::select_active_pbr_ibl_set();
  aurora::gx::pbr_internal::update_pbr_probe_spatial_blend();
  if (cachedProbe != nullptr || nearestProbeSelected) {
    aurora::gx::pbr_internal::start_pbr_ibl_blend(previousIbl);
  } else {
    aurora::gx::pbr_internal::clear_pbr_ibl_blend();
  }
}

void aurora_set_pbr_fill_dir(float x, float y, float z) {
  const float len = std::sqrt(x * x + y * y + z * z);
  const float inv = len > 1e-6f ? 1.0f / len : 1.0f;
  aurora::gx::pbrFillDir = {x * inv, y * inv, z * inv, 0.0f};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override(float roughness, float metallic, float ao, float specular,
                                               float emissive_r, float emissive_g, float emissive_b,
                                               float emissive_strength) {
  aurora::gx::pbrMaterialOverrideActive = true;
  aurora::gx::pbrMaterialFactors = {std::clamp(roughness, 0.04f, 1.0f), std::clamp(metallic, 0.0f, 1.0f),
                                    std::clamp(ao, 0.0f, 1.0f), std::clamp(specular, 0.0f, 1.0f)};
  aurora::gx::pbrMaterialEmissive = {std::max(emissive_r, 0.0f), std::max(emissive_g, 0.0f),
                                     std::max(emissive_b, 0.0f), std::max(emissive_strength, 0.0f)};
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override_normal_map(const char* normal_map_name) {
  if (normal_map_name == nullptr || normal_map_name[0] == '\0') {
    aurora::gx::pbrMaterialNormal.reset();
    aurora::gx::g_gxState.stateDirty = true;
    return;
  }

  const auto normal = aurora::gfx::texture_replacement::find_named_pbr_texture(normal_map_name);
  aurora::gx::pbrMaterialNormal = normal.value_or(nullptr);
  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_set_pbr_constant_material_override_maps(const char* material_name, bool use_rmaos, bool use_loose_maps,
                                                    bool use_normal, bool use_emissive) {
  aurora::gx::pbr_internal::clear_pbr_material_maps();

  if (material_name == nullptr || material_name[0] == '\0') {
    aurora::gx::g_gxState.stateDirty = true;
    return;
  }

  const std::string_view materialName{material_name};
  if (use_rmaos) {
    aurora::gx::pbrMaterialRmaos = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_rmaos"});
  }
  if (use_loose_maps) {
    aurora::gx::pbrMaterialRoughness =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_roughness", "_rough"});
    aurora::gx::pbrMaterialMetallic =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_metallic", "_metal"});
    aurora::gx::pbrMaterialAo = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_ao"});
    aurora::gx::pbrMaterialSpecular =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_specular", "_spec"});
  }
  if (use_normal) {
    aurora::gx::pbrMaterialNormal = aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_normal", "_n"});
  }
  if (use_emissive) {
    aurora::gx::pbrMaterialEmissiveMap =
        aurora::gx::pbr_internal::find_named_pbr_sidecar(materialName, {"_emissive", "_e"});
  }

  aurora::gx::g_gxState.stateDirty = true;
}

void aurora_clear_pbr_constant_material_override() {
  if (!aurora::gx::pbrMaterialOverrideActive) {
    return;
  }
  aurora::gx::pbrMaterialOverrideActive = false;
  aurora::gx::pbrMaterialFactors = {0.5f, 0.0f, 1.0f, 0.5f};
  aurora::gx::pbrMaterialEmissive = {0.0f, 0.0f, 0.0f, 0.0f};
  aurora::gx::pbr_internal::clear_pbr_material_maps();
  aurora::gx::g_gxState.stateDirty = true;
}

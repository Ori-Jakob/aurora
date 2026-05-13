#ifndef AURORA_GFX_H
#define AURORA_GFX_H

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stdint.h"
#endif

#ifndef NDEBUG
#define AURORA_GFX_DEBUG_GROUPS
#endif

void push_debug_group(const char* label);
void pop_debug_group();

typedef struct {
  uint32_t queuedPipelines;
  uint32_t createdPipelines;
  uint32_t drawCallCount;
  uint32_t mergedDrawCallCount;
  uint32_t lastVertSize;
  uint32_t lastUniformSize;
  uint32_t lastIndexSize;
  uint32_t lastStorageSize;
  uint32_t lastTextureUploadSize;
} AuroraStats;

const AuroraStats* aurora_get_stats();

void aurora_enable_vsync(bool enabled);
void aurora_set_fog_override(bool enabled, float exposure, float opacity, float color_r, float color_g, float color_b);
void aurora_enable_pbr(bool enabled);
bool aurora_pbr_enabled();
void aurora_set_pbr_light_params(float ambient, float ambient_specular, float fill_intensity);
void aurora_set_pbr_fill_dir(float x, float y, float z);
void aurora_set_pbr_material_params(float diffuse_scale, float specular_scale);
void aurora_set_pbr_normal_params(float strength, bool flip_y, bool invert_handedness);
void aurora_set_pbr_ambient_gradient_params(float sky, float ground, float horizon, float environment_tint);
void aurora_set_pbr_indirect_occlusion_params(float strength, float horizon, float specular);
void aurora_set_pbr_dynamic_gi_params(bool enabled, float strength, float normal_wrap, float albedo_influence);
void aurora_set_pbr_ibl_params(bool enabled, float diffuse_strength, float specular_strength);
typedef enum {
  AURORA_PBR_ENHANCED_LIGHT_FALLOFF_LEGACY_RADIUS = 0,
  AURORA_PBR_ENHANCED_LIGHT_FALLOFF_INVERSE_SQUARE = 1,
} AuroraPbrEnhancedLightFalloff;
typedef struct {
  float position[3];
  float radius;
  float color[3];
  float intensity;
} AuroraPbrEnhancedLight;
typedef enum {
  AURORA_SCENE_LIGHT_POINT = 0,
  AURORA_SCENE_LIGHT_DIRECTIONAL = 1,
  AURORA_SCENE_LIGHT_SPOT = 2,
} AuroraSceneLightType;
typedef enum {
  AURORA_SCENE_LIGHT_SOURCE_UNKNOWN = 0,
  AURORA_SCENE_LIGHT_SOURCE_GAME_POINT = 1,
  AURORA_SCENE_LIGHT_SOURCE_GAME_EFFECT = 2,
  AURORA_SCENE_LIGHT_SOURCE_AUTHORED = 3,
} AuroraSceneLightSource;
typedef struct {
  float position[3];
  float radius;
  float color[3];
  float intensity;
  float direction[3];
  float innerConeCos;
  float outerConeCos;
  float score;
  uint32_t type;
  uint32_t source;
  uint32_t sourceIndex;
  uint32_t flags;
  uint64_t stableId;
} AuroraSceneLight;
typedef struct {
  bool enabled;
  bool debugEnabled;
  uint32_t lightCount;
  uint32_t maxLightCount;
  uint32_t submittedSceneLightCount;
  bool sceneLightApiBacked;
  bool storageBacked;
  uint32_t storageByteSize;
  AuroraPbrEnhancedLightFalloff falloff;
  float intensityScale;
  bool dynamicGiEnabled;
  float dynamicGiStrength;
  float dynamicGiNormalWrap;
  float dynamicGiAlbedoInfluence;
  AuroraPbrEnhancedLight strongestLight;
} AuroraPbrEnhancedLightingStatus;
typedef enum {
  AURORA_PBR_IBL_SOURCE_PROBE = 0,
  AURORA_PBR_IBL_SOURCE_AUTHORED = 1,
  AURORA_PBR_IBL_SOURCE_FALLBACK = 2,
} AuroraPbrIblSource;
typedef struct {
  bool pbrEnabled;
  bool iblEnabled;
  AuroraPbrIblSource requestedSource;
  AuroraPbrIblSource activeSource;
  bool fallbackAvailable;
  bool authoredAvailable;
  bool probeAvailable;
  bool probeCameraValid;
  bool probeCaptureResourcesReady;
  bool probeCaptureInProgress;
  bool probeRefreshPending;
  bool probeFilterPending;
  bool probeAutoRefresh;
  bool probeLocalGi;
  bool probeSceneStale;
  bool probeCacheEnabled;
  bool probeCacheHit;
  bool probeNearestCacheEnabled;
  bool probeNearestCacheActive;
  bool probeSpatialBlendEnabled;
  bool probeSpatialBlendActive;
  bool probeBlendEnabled;
  bool probeBlendActive;
  bool probeReplayPbrVisible;
  uint32_t probeCaptureFace;
  uint32_t probeCaptureDelayFrames;
  uint32_t probeFramesSinceRefresh;
  uint32_t probeReplayDraws;
  uint32_t probeCubeSize;
  uint32_t probeIrradianceSize;
  uint32_t probePrefilterMipCount;
  uint32_t probeCacheSlots;
  uint32_t probeCacheUsedSlots;
  uint32_t probeBlendFrames;
  float probeBlendFactor;
  float probeNearestCacheDistance;
  float probeNearestCacheMaxDistance;
  float probeSpatialBlendFactor;
  float probeSpatialBlendDistance;
  float probeSpatialBlendMaxDistance;
  uint32_t activePrefilterMipCount;
  bool authoredGlobalLoaded;
  bool authoredStageLoaded;
  bool authoredRoomLoaded;
  int32_t authoredRoom;
  char authoredRoot[260];
  char authoredStage[64];
  char authoredSceneKey[96];
  char activeProbeSceneKey[96];
  char spatialProbeSceneKey[96];
} AuroraPbrIblStatus;
typedef struct {
  bool valid;
  uint32_t source;
  uint32_t sourceIndex;
  uint64_t stableId;
  float position[3];
  float target[3];
  float color[3];
  float radius;
  float score;
  float priority;
} AuroraPbrShadowLightRequest;
typedef struct {
  bool enabled;
  bool resourcesReady;
  bool mapAvailable;
  bool lightRequestValid;
  uint32_t size;
  float strength;
  float bias;
  AuroraPbrShadowLightRequest lightRequest;
} AuroraPbrShadowMapStatus;
typedef enum {
  AURORA_PBR_DEBUG_OFF = 0,
  AURORA_PBR_DEBUG_ALBEDO = 1,
  AURORA_PBR_DEBUG_ROUGHNESS = 2,
  AURORA_PBR_DEBUG_METALLIC = 3,
  AURORA_PBR_DEBUG_AO = 4,
  AURORA_PBR_DEBUG_SPECULAR = 5,
  AURORA_PBR_DEBUG_NORMAL = 6,
  AURORA_PBR_DEBUG_GX_LIGHT_TINT = 7,
  AURORA_PBR_DEBUG_DIRECT_DIFFUSE = 8,
  AURORA_PBR_DEBUG_DIRECT_SPECULAR = 9,
  AURORA_PBR_DEBUG_IBL_DIFFUSE = 10,
  AURORA_PBR_DEBUG_IBL_SPECULAR = 11,
  AURORA_PBR_DEBUG_INDIRECT_OCCLUSION = 12,
  AURORA_PBR_DEBUG_DYNAMIC_GI = 13,
} AuroraPbrDebugMode;
void aurora_set_pbr_ibl_source(AuroraPbrIblSource source);
void aurora_set_pbr_debug_mode(AuroraPbrDebugMode mode);
void aurora_set_pbr_enhanced_lighting(bool enabled, AuroraPbrEnhancedLightFalloff falloff, uint32_t max_light_count,
                                      float intensity_scale, bool debug_enabled);
void aurora_set_pbr_enhanced_lights(const AuroraPbrEnhancedLight* lights, uint32_t light_count);
void aurora_set_scene_lights(const AuroraSceneLight* lights, uint32_t light_count);
const AuroraPbrEnhancedLightingStatus* aurora_get_pbr_enhanced_lighting_status();
bool aurora_pbr_probe_ibl_available();
const AuroraPbrIblStatus* aurora_get_pbr_ibl_status();
void aurora_set_pbr_shadow_map_params(bool enabled, uint32_t size, float strength, float bias);
void aurora_set_pbr_shadow_light_request(const AuroraPbrShadowLightRequest* request);
void aurora_set_pbr_shadow_map_matrix(const float* light_view_projection_4x4);
const AuroraPbrShadowMapStatus* aurora_get_pbr_shadow_map_status();
void aurora_set_pbr_probe_auto_refresh(bool enabled);
void aurora_set_pbr_local_probe_gi(bool enabled);
void aurora_set_pbr_probe_cache(bool enabled);
void aurora_set_pbr_probe_nearest_cache(bool enabled, float max_distance);
void aurora_set_pbr_probe_spatial_blending(bool enabled, float max_distance);
void aurora_set_pbr_probe_blending(bool enabled, uint32_t frames);
void aurora_request_pbr_probe_refresh();
void aurora_set_pbr_probe_capture_enabled(bool enabled);
void aurora_set_pbr_probe_camera_matrices(const float* source_view_3x4, const float* source_inv_view_3x4,
                                          const float* projection_4x4);
void aurora_set_pbr_ibl_scene(const char* stage_name, int room_no);
void aurora_set_pbr_constant_material_override(float roughness, float metallic, float ao, float specular,
                                               float emissive_r, float emissive_g, float emissive_b,
                                               float emissive_strength);
void aurora_set_pbr_constant_material_override_normal_map(const char* normal_map_name);
void aurora_set_pbr_constant_material_override_maps(const char* material_name, bool use_rmaos, bool use_loose_maps,
                                                    bool use_normal, bool use_emissive);
void aurora_clear_pbr_constant_material_override();

#ifdef __cplusplus
}
#endif

#endif

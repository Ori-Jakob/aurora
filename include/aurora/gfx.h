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
void aurora_set_pbr_ibl_params(bool enabled, float diffuse_strength, float specular_strength);
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
  uint32_t probeCaptureFace;
  uint32_t probeCaptureDelayFrames;
  uint32_t probeFramesSinceRefresh;
  uint32_t probeCubeSize;
  uint32_t probeIrradianceSize;
  uint32_t probePrefilterMipCount;
  uint32_t activePrefilterMipCount;
  bool authoredGlobalLoaded;
  bool authoredStageLoaded;
  bool authoredRoomLoaded;
  int32_t authoredRoom;
  char authoredRoot[260];
  char authoredStage[64];
  char authoredSceneKey[96];
} AuroraPbrIblStatus;
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
} AuroraPbrDebugMode;
void aurora_set_pbr_ibl_source(AuroraPbrIblSource source);
void aurora_set_pbr_debug_mode(AuroraPbrDebugMode mode);
bool aurora_pbr_probe_ibl_available();
const AuroraPbrIblStatus* aurora_get_pbr_ibl_status();
void aurora_set_pbr_probe_auto_refresh(bool enabled);
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

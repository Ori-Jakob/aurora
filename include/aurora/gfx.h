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
void aurora_enable_pbr(bool enabled);
bool aurora_pbr_enabled();
void aurora_set_pbr_light_params(float ambient, float ambient_specular, float fill_intensity);
void aurora_set_pbr_fill_dir(float x, float y, float z);
void aurora_set_pbr_material_params(float diffuse_scale, float specular_scale);
void aurora_set_pbr_normal_params(float strength, bool flip_y, bool invert_handedness);
void aurora_set_pbr_ambient_gradient_params(float sky, float ground, float horizon, float environment_tint);
void aurora_set_pbr_ibl_params(bool enabled, float diffuse_strength, float specular_strength);
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

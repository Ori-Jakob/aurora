#pragma once

#include "gx.hpp"

#include <aurora/gfx.h>

#include <array>

namespace aurora::gx {

extern bool enablePbrMaterialOverride;
extern bool pbrMaterialOverrideActive;
extern Vec4<float> pbrParams;           // x=ambient, y=ambient_specular, z=fill_intensity, w=unused
extern Vec4<float> pbrScales;           // x=diffuse_scale, y=specular_scale, z=debug_mode, w=unused
extern Vec4<float> pbrNormalParams;     // x=strength, y=normal_y_sign, z=handedness_sign, w=unused
extern Vec4<float> pbrAmbientGradient;  // x=sky, y=ground, z=horizon, w=environment_tint
extern Vec4<float> pbrIndirectOcclusion; // x=strength, y=horizon, z=specular, w=unused
extern Vec4<float> pbrDynamicGiParams;  // x=enabled, y=strength, z=normal_wrap, w=albedo_influence
extern Vec4<float> pbrIblParams;        // x=enabled, y=diffuse_strength, z=specular_strength, w=max_prefilter_mip
extern Vec4<float> pbrIblBlendParams;   // x=blend_to_active/spatial_active_weight, y=blend_from_max_mip, zw=unused
extern Vec4<float> pbrFillDir;          // xyz=fill light direction (view space), w=unused
extern Vec4<float> pbrMaterialFactors;  // x=roughness, y=metallic, z=ao, w=specular
extern Vec4<float> pbrMaterialEmissive; // rgb=color, a=strength
constexpr u32 PbrMaxEnhancedLights = 64;
struct PbrEnhancedLight {
  Vec4<float> posRadius{0.0f, 0.0f, 0.0f, 1.0f};      // xyz=view-space position, w=radius
  Vec4<float> colorIntensity{0.0f, 0.0f, 0.0f, 0.0f}; // rgb=color, a=intensity
  Vec4<float> dirType{0.0f, 0.0f, 0.0f, 0.0f};        // xyz=view-space direction, w=type
  Vec4<float> shadowParams{0.0f, 0.0f, 0.0f, 0.0f};   // x=has shadow map, y=atlas slot
};
extern bool pbrEnhancedLightsEnabled;
extern bool pbrEnhancedLightsDebugEnabled;
extern AuroraPbrEnhancedLightFalloff pbrEnhancedLightFalloff;
extern u32 pbrEnhancedLightMaxCount;
extern u32 pbrEnhancedLightCount;
extern u32 pbrSubmittedSceneLightCount;
extern bool pbrSceneLightsApiBacked;
extern float pbrEnhancedLightIntensityScale;
extern u32 pbrEnhancedLightStorageOffset;
extern u32 pbrEnhancedLightStorageSize;
extern std::array<PbrEnhancedLight, PbrMaxEnhancedLights> pbrEnhancedLights;
extern gfx::TextureHandle pbrMaterialRmaos;
extern gfx::TextureHandle pbrMaterialRoughness;
extern gfx::TextureHandle pbrMaterialMetallic;
extern gfx::TextureHandle pbrMaterialAo;
extern gfx::TextureHandle pbrMaterialSpecular;
extern gfx::TextureHandle pbrMaterialNormal;
extern gfx::TextureHandle pbrMaterialEmissiveMap;

void configure_pbr_material_override(ShaderConfig& config, const ShaderInfo& info) noexcept;
void initialize_pbr_resources() noexcept;
void add_pbr_texture_layout_entries(
    std::array<wgpu::BindGroupLayoutEntry, TextureBindGroupEntryCount>& textureEntries) noexcept;
void add_pbr_empty_bind_group_entries(std::array<wgpu::BindGroupEntry, TextureBindGroupEntryCount>& entries,
                                      const wgpu::TextureView& emptyTextureView,
                                      const wgpu::Sampler& emptySampler) noexcept;
void bind_pbr_texture_entries(std::array<WGPUBindGroupEntry, TextureBindGroupEntryCount>& textureEntries,
                              const ShaderInfo& info, const wgpu::TextureView& emptyTextureView,
                              const wgpu::Sampler& emptySampler, bool bindShadowReceivers = true) noexcept;
bool pbr_probe_capture_requested() noexcept;
void set_pbr_probe_replay_status(uint32_t eligible_draws, bool pbr_material_visible) noexcept;
uint32_t pbr_probe_cube_size() noexcept;
uint32_t pbr_probe_capture_face() noexcept;
void begin_pbr_probe_capture() noexcept;
void finish_pbr_probe_capture() noexcept;
const wgpu::TextureView& pbr_probe_capture_color_view(uint32_t face) noexcept;
const wgpu::TextureView& pbr_probe_capture_face_view(uint32_t face) noexcept;
const wgpu::TextureView& pbr_probe_capture_depth_view() noexcept;
void run_pbr_probe_filter(const wgpu::CommandEncoder& cmd) noexcept;
void patch_pbr_probe_uniform(uint8_t* uniformData, const ProbeUniformPatchInfo& patch, uint32_t face) noexcept;
bool pbr_shadow_map_capture_requested() noexcept;
uint32_t pbr_shadow_map_capture_slot_count() noexcept;
uint32_t pbr_shadow_map_capture_slots_per_frame() noexcept;
bool pbr_shadow_map_capture_slot_requested(uint32_t slot) noexcept;
uint32_t pbr_shadow_map_size() noexcept;
const wgpu::TextureView& pbr_shadow_map_depth_view() noexcept;
const wgpu::TextureView& pbr_shadow_map_depth_slot_view(uint32_t slot) noexcept;
void begin_pbr_shadow_map_capture() noexcept;
bool begin_pbr_shadow_map_capture_slot(uint32_t slot) noexcept;
void finish_pbr_shadow_map_capture(uint32_t drawCount) noexcept;
void finish_pbr_shadow_map_capture(uint32_t drawCount, uint32_t capturedSlotCount) noexcept;
void finish_pbr_shadow_map_capture(uint32_t drawCount, uint32_t capturedSlotCount, uint32_t capturedSlotMask) noexcept;
void patch_pbr_shadow_uniform(uint8_t* uniformData, const ProbeUniformPatchInfo& patch) noexcept;
Vec4<float> pbr_shadow_params() noexcept;
Vec4<float> pbr_shadow_storage_params() noexcept;
Mat3x4<float> pbr_shadow_view_from_source() noexcept;
Mat4x4<float> pbr_shadow_projection() noexcept;
std::array<Mat3x4<float>, 4> pbr_shadow_view_from_source_slots() noexcept;
std::array<Mat4x4<float>, 4> pbr_shadow_projection_slots() noexcept;
const wgpu::TextureView& pbr_shadow_receiver_view() noexcept;
const wgpu::TextureView& pbr_shadow_receiver_view(uint32_t slot) noexcept;
const wgpu::Sampler& pbr_shadow_receiver_sampler() noexcept;

} // namespace aurora::gx

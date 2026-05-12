#pragma once

#include "gx.hpp"

#include <array>

namespace aurora::gx {

extern bool enablePbrMaterialOverride;
extern bool pbrMaterialOverrideActive;
extern Vec4<float> pbrParams;           // x=ambient, y=ambient_specular, z=fill_intensity, w=unused
extern Vec4<float> pbrScales;           // x=diffuse_scale, y=specular_scale, z/w=unused
extern Vec4<float> pbrNormalParams;     // x=strength, y=normal_y_sign, z=handedness_sign, w=unused
extern Vec4<float> pbrAmbientGradient;  // x=sky, y=ground, z=horizon, w=environment_tint
extern Vec4<float> pbrIblParams;        // x=enabled, y=diffuse_strength, z=specular_strength, w=max_prefilter_mip
extern Vec4<float> pbrFillDir;          // xyz=fill light direction (view space), w=unused
extern Vec4<float> pbrMaterialFactors;  // x=roughness, y=metallic, z=ao, w=specular
extern Vec4<float> pbrMaterialEmissive; // rgb=color, a=strength
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
                              const wgpu::Sampler& emptySampler) noexcept;

} // namespace aurora::gx

#pragma once

#include "texture.hpp"

namespace aurora::gfx {
constexpr u8 TextureFlagStochasticSampling = 0x40;
constexpr u8 TextureFlagNoCache = 0x80;

bool stochastic_sampling_enabled() noexcept;
void set_stochastic_sampling_enabled(bool enabled) noexcept;
void set_stochastic_sampling_params(float cellScale, float jitter, float blendWidth) noexcept;
Vec4<float> stochastic_sampling_params() noexcept;
bool uses_stochastic_sampling(const GXTexObj_& obj) noexcept;
void set_stochastic_sampling(GXTexObj_& obj, bool enabled) noexcept;
void set_stochastic_sampling_params_override(const GXTexObj_& obj, float cellScale, float jitter,
                                             float blendWidth) noexcept;
void clear_stochastic_sampling_params_override(const GXTexObj_& obj) noexcept;
bool stochastic_sampling_params_override(const GXTexObj_& obj, Vec4<float>& params) noexcept;
void erase_stochastic_sampling_params_override(u32 texObjId) noexcept;
Vec4<float> texture_size_bias_sampling_params(const TextureBind& tex) noexcept;
Vec4<float> texture_stochastic_sampling_params(const TextureBind& tex) noexcept;
} // namespace aurora::gfx

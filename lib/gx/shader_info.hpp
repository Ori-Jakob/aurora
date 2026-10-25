#pragma once

#include "gx.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aurora::gx {
bool uses_texture_sample(const TevStage& stage) noexcept;
bool stochastic_fits(u32 unalignedUniformSize, u32 maps) noexcept;
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
u32 varying_count(const ShaderConfig& config, const ShaderTargets& targets) noexcept;
gfx::Range build_uniform(const ShaderInfo& info) noexcept;
inline constexpr size_t DrawUniformReuseDepth = 1;
gfx::Range build_draw_uniform(const ShaderInfo& info) noexcept;
void clear_draw_uniforms() noexcept;
u8 color_channel(GXChannelID id) noexcept;

// Matches AuroraGxFog in aurora/gx_fog.wgsl
struct FogParams {
  Fog fog;
  u32 type;
  u32 rangeEnabled;
  f32 renderWidth;
  f32 viewportZNear;
  f32 viewportZFar;
  u32 pad[3] = {};
};
static_assert(offsetof(FogParams, type) == 80 && offsetof(FogParams, viewportZFar) == 96 && sizeof(FogParams) == 112);
Fog make_fog_uniform() noexcept;
FogParams current_fog_params() noexcept; // FIFO thread

namespace testing {
// The exact bytes fill_uniform produces; tests compare them with the WGSL member order
std::vector<uint8_t> uniform_bytes(const ShaderInfo& info) noexcept;
} // namespace testing
}; // namespace aurora::gx

#pragma once

#include "gx.hpp"

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

namespace testing {
// The exact bytes fill_uniform produces; tests compare them with the WGSL member order
std::vector<uint8_t> uniform_bytes(const ShaderInfo& info) noexcept;
} // namespace testing
}; // namespace aurora::gx

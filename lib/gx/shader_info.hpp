#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
gfx::Range build_uniform(const ShaderInfo& info) noexcept;
inline constexpr size_t DrawUniformReuseDepth = 1;
gfx::Range build_draw_uniform(const ShaderInfo& info) noexcept;
void clear_draw_uniforms() noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx

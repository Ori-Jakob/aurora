#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
gfx::Range build_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges) noexcept;
ProbeUniformPatchInfo probe_uniform_patch_info(const ShaderInfo& info, bool projectionIsPerspective) noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx

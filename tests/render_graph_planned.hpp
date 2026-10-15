#pragma once

#include "render/graph.hpp"

#include <array>
#include <span>
#include <vector>

namespace aurora::render::detail::planned {
std::vector<PassDesc> planned_passes(bool reversed = false);
std::vector<GxReadDesc> planned_gx_reads();
GraphLimits full_limits();
GraphLimits metal32_limits();
GraphLimits compat_limits();
GraphInputs planned_inputs(const GraphLimits& limits, std::span<const PassDesc> passes,
                           std::span<const GxReadDesc> gxReads);
std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_tier_p();
std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_tier_b();
std::array<wgpu::TextureFormat, GxResourceCount> gx_formats_legacy();
std::array<wgpu::TextureUsage, GxResourceCount> gx_usage_future();
std::array<wgpu::TextureUsage, GxResourceCount> gx_usage_today();
} // namespace aurora::render::detail::planned

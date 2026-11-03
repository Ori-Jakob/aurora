#pragma once

#include "../graph.hpp"

#include <string_view>

namespace aurora::render::debug::surface_expand {
inline constexpr std::string_view Pass = "debug.surface_expand";
inline constexpr std::string_view Shader = "debug/surface_expand";
inline constexpr uint32_t WorkgroupSize = 8;

detail::PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
} // namespace aurora::render::debug::surface_expand

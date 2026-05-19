#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace aurora::gx {
std::string_view stochastic_sampling_wgsl() noexcept;
std::string sampled_texture_expr(uint32_t texMapId, std::string_view uvExpr);
} // namespace aurora::gx

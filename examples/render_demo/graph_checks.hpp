#pragma once

#include <aurora/aurora.h>
#include <dolphin/types.h>

#include <string_view>

namespace render_demo {
struct VignetteParams {
  f32 strength;
  f32 inner;
  f32 outer;
  f32 pad;
};
static_assert(sizeof(VignetteParams) == 16);

VignetteParams vignette_params(u32 frame) noexcept;
bool is_unknown_anchor_warning(AuroraLogLevel level, std::string_view module, std::string_view message) noexcept;
} // namespace render_demo

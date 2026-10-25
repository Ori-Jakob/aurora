#pragma once

#include <aurora/aurora.h>
#include <dolphin/types.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace render_demo {
struct VignetteParams {
  f32 strength;
  f32 inner;
  f32 outer;
  f32 pad;
};
static_assert(sizeof(VignetteParams) == 16);

inline constexpr u32 DepthCheckFrame = 30;
inline constexpr std::array<f32, 3> DepthProbeRows{0.04f, 0.80f, 0.96f};

inline constexpr u32 MotionCheckFrame = 30;
inline constexpr u32 MotionProbeCount = 4;
inline constexpr u32 MotionProbeStride = 8;
inline constexpr u32 MotionProbeFloats = MotionProbeCount * MotionProbeStride + 4;

struct MotionProbeParams {
  f32 ndc[MotionProbeCount * 2];
  u32 frame;
  u32 pad[3];
};
static_assert(sizeof(MotionProbeParams) == 48);

VignetteParams vignette_params(u32 frame) noexcept;
bool is_unknown_anchor_warning(AuroraLogLevel level, std::string_view module, std::string_view message) noexcept;
std::optional<std::string> check_depth_probe(std::span<const f32, 4> values, bool reversedZ);
} // namespace render_demo

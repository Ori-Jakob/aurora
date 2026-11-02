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

enum class AoProbeClass : u32 { Crease, Open, Strip, Interior };
inline constexpr u32 AoProbeClassCount = 4;

inline constexpr double AoCreaseMax = 0.75;
inline constexpr double AoOpenMin = 0.97;
inline constexpr double AoOpenShare = 0.99;
inline constexpr double AoStripMin = 0.9;
inline constexpr double AoStripInteriorGap = 0.05;
inline constexpr double AoCoverageMin = 0.1;

struct AoProbeStat {
  AoProbeClass cls;
  double visibility;
  bool receiver;
};

struct AoCheckSummary {
  std::array<double, AoProbeClassCount> mean{};
  std::array<u32, AoProbeClassCount> receivers{};
  std::array<u32, AoProbeClassCount> probes{};
  double openShare = 0.0;
  double coverage = 0.0;
};

VignetteParams vignette_params(u32 frame) noexcept;
bool is_unknown_anchor_warning(AuroraLogLevel level, std::string_view module, std::string_view message) noexcept;
std::optional<std::string> check_depth_probe(std::span<const f32, 4> values, bool reversedZ);
AoCheckSummary summarize_ao_check(std::span<const AoProbeStat> probes, double coverage);
std::optional<std::string> check_ao(const AoCheckSummary& summary);
} // namespace render_demo

#pragma once

#include <aurora/aurora.h>
#include <dolphin/types.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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

enum class AoProbeClass : u32 { Crease, Open, Strip, Interior, Contact };
inline constexpr u32 AoProbeClassCount = 4;

inline constexpr double AoCreaseMax = 0.8;
inline constexpr double AoOpenMin = 0.97;
inline constexpr double AoOpenShare = 0.99;
inline constexpr double AoStripMin = 0.9;
inline constexpr double AoStripInteriorGap = 0.05;
inline constexpr double AoCoverageMin = 0.1;

inline constexpr u32 AoCheckFrames = 240;
inline constexpr u32 AoSigmaFirst = 40;
inline constexpr u32 AoSigmaLast = 100;
inline constexpr u32 AoSliderFirst = 110;
inline constexpr u32 AoSliderRemoved = 170;
inline constexpr u32 AoTrailFirst = 130;
inline constexpr u32 AoTrailLast = 170;
inline constexpr u32 AoSettleFrame = 174;
inline constexpr u32 AoSteadyFrame = 230;
inline constexpr std::array<u32, 3> AoContactFrames{AoSliderRemoved - 1, AoSettleFrame, AoSteadyFrame};
inline constexpr f32 AoSliderStep = 0.05f;
inline constexpr f32 AoTrailNear = 0.98f;
inline constexpr f32 AoTrailFar = 1.48f;
inline constexpr f32 AoContactNear = 0.05f;
inline constexpr f32 AoContactFar = 0.6f;
inline constexpr u32 AoTrailProbeCount = 10;
inline constexpr double AoTrailMargin = 0.05;
inline constexpr double AoRemovalMax = 0.05;
inline constexpr double AoSliderShadowMin = 0.02;

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

struct AoTrailFrame {
  u32 frame = 0;
  double open = 0.0;
  u32 openReceivers = 0;
  double trail = 0.0;
  u32 trailReceivers = 0;
};

struct AoSigma {
  u32 frames = 0;
  u32 pixels = 0;
  u32 covered = 0;
  double sigma = 0.0;
  double mean = 0.0;
};

struct AoTemporalSummary {
  AoSigma sigma;
  std::vector<AoTrailFrame> trail;
  std::array<std::vector<f32>, AoContactFrames.size()> contact;
};

struct AoTemporalOutcome {
  std::optional<std::string> failure;
  double trailMargin = 0.0;
  u32 trailFrames = 0;
  double removal = 0.0;
  u32 removalProbes = 0;
  double shadow = 0.0;
};

inline constexpr u32 FwdEnvCheckFrame = 30;
inline constexpr u32 FwdEnvSettleFrames = 16;

enum class FwdEnvProbeClass : u32 { StripA, StripB, Card };

struct FwdEnvSample {
  FwdEnvProbeClass cls = FwdEnvProbeClass::StripA;
  u32 color = 0;
  u32 fog = 0;
  u32 ao = 0;
  u32 normal = 0;
  u32 depthBits = 0;
};

struct FwdEnvExpect {
  std::array<std::array<u8, 3>, 3> colors{};
  std::array<f32, 3> groundNormal{};
};

struct FwdEnvGate {
  u32 checked = 0;
  u32 sensitive = 0;
  u32 failures = 0;
  double maxError = 0.0;
};

struct FwdEnvSummary {
  FwdEnvGate far;
  FwdEnvGate near;
  FwdEnvGate plain;
  FwdEnvGate card;
  u32 alphaMismatches = 0;
  u32 exactOneViolations = 0;
  u32 samples = 0;
};

inline constexpr u32 FwdEnvFarSensitive = 8;
inline constexpr u32 FwdEnvNearSensitive = 16;
inline constexpr u32 FwdEnvCardSensitive = 16;

inline constexpr u32 UnlitCheckFirstFrame = 18;
inline constexpr u32 UnlitCheckSecondFrame = 38;
inline constexpr u32 UnlitSunFrame = 20;
inline constexpr u32 UnlitPipelineWaitFrame = 4;
inline constexpr u32 UnlitColourMove = 8;
inline constexpr u32 UnlitRoughness = 217;

enum class UnlitProbeClass : u32 { Ground, Control };

struct UnlitSample {
  UnlitProbeClass cls = UnlitProbeClass::Ground;
  u32 color = 0;
  std::array<u32, 4> surface{};
};

struct UnlitCheckSummary {
  u32 ground = 0;
  u32 control = 0;
  u32 classMismatches = 0;
  u32 paramMismatches = 0;
  u32 albedoChanges = 0;
  u32 minColourMove = 0;
  u32 maxColourMove = 0;
  u32 controlPbr = 0;
};

VignetteParams vignette_params(u32 frame) noexcept;
bool is_unknown_anchor_warning(AuroraLogLevel level, std::string_view module, std::string_view message) noexcept;
std::optional<std::string> check_depth_probe(std::span<const f32, 4> values, bool reversedZ);
AoCheckSummary summarize_ao_check(std::span<const AoProbeStat> probes, double coverage);
std::optional<std::string> check_ao(const AoCheckSummary& summary);
AoTemporalOutcome check_ao_temporal(const AoTemporalSummary& summary);
FwdEnvSummary summarize_fwd_env(std::span<const FwdEnvSample> samples, const FwdEnvExpect& expect);
std::optional<std::string> check_fwd_env(const FwdEnvSummary& summary);
UnlitCheckSummary summarize_unlit(std::span<const UnlitSample> before, std::span<const UnlitSample> after);
std::optional<std::string> check_unlit(const UnlitCheckSummary& summary);
} // namespace render_demo

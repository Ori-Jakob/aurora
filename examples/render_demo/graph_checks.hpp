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

inline constexpr u32 SsilCheckFrame = 40;
inline constexpr u32 SsilSettleFrames = 24;
inline constexpr u32 SsilBandProbes = 6;
inline constexpr u32 SsilBouncedTexels = 1000;

enum class SsilProbeClass : u32 { WallFoot, FinBack, FinFront, Open, Count };

struct SsilSample {
  SsilProbeClass cls = SsilProbeClass::WallFoot;
  std::array<f32, 4> irr{};
  u32 code = 0;
};

struct SsilBand {
  u32 probes = 0;
  u32 receivers = 0;
  std::array<double, 3> mean{};
  double coverage = 0.0;
};

struct SsilCheckSummary {
  std::array<SsilBand, static_cast<size_t>(SsilProbeClass::Count)> bands{};
  u32 texels = 0;
  u32 bounced = 0;
  u32 outOfRange = 0;
};

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

inline constexpr u32 CompanionPipelineWaitFrame = 4;
inline constexpr u32 CompanionBareFrame = 16;
inline constexpr u32 CompanionLoadFrame = 18;
inline constexpr u32 CompanionMappedWaitFrame = 22;
inline constexpr u32 CompanionMappedFrame = 30;
inline constexpr u32 CompanionBlendOneFrame = 31;
inline constexpr u32 CompanionBlendOneProbeFrame = 38;
inline constexpr u32 CompanionMatFrame = 39;
inline constexpr u32 CompanionMatWaitFrame = 43;
inline constexpr u32 CompanionMatProbeFrame = 52;
inline constexpr u32 CompanionScalarFrame = 53;
inline constexpr u32 CompanionScalarWaitFrame = 56;
inline constexpr u32 CompanionScalarProbeFrame = 62;
inline constexpr u32 CompanionEmitFrame = 63;
inline constexpr u32 CompanionEmitWaitFrame = 67;
inline constexpr u32 CompanionEmitProbeFrame = 76;
inline constexpr u32 CompanionManifestFrame = 77;
inline constexpr u32 CompanionManifestWaitFrame = 81;
inline constexpr u32 CompanionManifestProbeFrame = 90;
inline constexpr u32 CompanionPhaseCount = 7;
inline constexpr u32 CompanionNrmRoles = 3;
inline constexpr u32 CompanionMatRoles = 1;
inline constexpr f32 CompanionTiltMin = 0.05f;
inline constexpr f32 CompanionBareTolerance = 2e-3f;
inline constexpr u8 CompanionDelightCode = 144;
inline constexpr u8 CompanionTiltNrmCode = 224;
inline constexpr f32 CompanionNormalStrength = 2.f;
inline constexpr f32 CompanionManifestRoughness = 0.5f;
inline constexpr int CompanionDelightLsb = 2;

enum class CompanionPhase : u32 { Bare, Mapped, BlendOne, Mat, Scalars, Emit, Manifest };

struct CompanionSample {
  u32 quad = 0;
  u32 cell = 0;
  std::array<f32, 4> normal{};
  std::array<u32, 4> surface{};
};

struct CompanionAxes {
  std::array<f32, 3> x{};
  std::array<f32, 3> z{};
  f32 matYSign = -1.f;
};

struct CompanionCheckSummary {
  u32 probes = 0;
  u32 bareMismatches = 0;
  u32 tiltFailures = 0;
  u32 matFailures = 0;
  u32 blendOneMismatches = 0;
  u32 ormFailures = 0;
  u32 albedoFailures = 0;
  f32 minTilt = 0.f;
  u32 scalarFailures = 0;
  u32 emitFailures = 0;
  u32 emitLeaks = 0;
  u32 delightFailures = 0;
  int maxDelightError = 0;
  u32 manifestNrmFailures = 0;
  u32 manifestRoughFailures = 0;
};

inline constexpr u32 ClassPipelineWaitFrame = 4;
inline constexpr u32 ClassCheckFirstFrame = 18;
inline constexpr u32 ClassEditFrame = 20;
inline constexpr u32 ClassCheckSecondFrame = 30;
inline constexpr u32 ClassBlendOneFrame = 31;
inline constexpr u32 ClassCheckThirdFrame = 38;
inline constexpr u32 ClassPhaseCount = 3;
inline constexpr u8 ClassHintRegistered = 200;
inline constexpr u8 ClassHintUnregistered = 210;
inline constexpr u8 ClassHintFoliage = 7;
inline constexpr std::array<u8, 3> ClassAmbient{64, 128, 191};
inline constexpr std::array<u8, 3> ClassVertex{128, 128, 128};
inline constexpr std::array<u8, 3> ClassChannel1Vertex{200, 100, 50};
inline constexpr u32 ClassEditedRoughness = 128;

enum class ClassQuad : u32 {
  Plain,
  Rock,
  Scalars,
  FakeShadow,
  Lava,
  Dry,
  Hint,
  UnregisteredHint,
  RockHint,
  BackFace,
  CullFront,
  Mirrored,
  Foliage,
  Player,
  Ambient,
  Channel1,
  Count,
};

enum class ClassPhase : u32 { First, Edited, BlendOne };

struct ClassSample {
  ClassQuad quad = ClassQuad::Plain;
  std::array<f32, 4> normal{};
  std::array<u32, 4> surface{};
};

struct ClassCheckSummary {
  u32 probes = 0;
  u32 failures = 0;
  std::string first;
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
CompanionCheckSummary summarize_companions(std::span<const std::vector<CompanionSample>, CompanionPhaseCount> phases,
                                           const CompanionAxes& axes, bool delight);
std::array<int, 3> companion_delight_expected(u32 albedo, u8 delight) noexcept;
std::optional<std::string> check_companions(const CompanionCheckSummary& summary);
ClassCheckSummary summarize_classes(std::span<const std::vector<ClassSample>, ClassPhaseCount> phases,
                                    bool bakedVertex);
std::optional<std::string> check_classes(const ClassCheckSummary& summary);
SsilCheckSummary summarize_ssil(std::span<const SsilSample> samples, u32 texels, u32 bounced, u32 outOfRange);
std::optional<std::string> check_ssil(const SsilCheckSummary& summary);
} // namespace render_demo

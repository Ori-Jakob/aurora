#pragma once

#include "cull_pass.hpp"
#include "graph_checks.hpp"
#include "scene.hpp"

#include <aurora/render_graph.hpp>
#include <dolphin/types.h>

#include <array>
#include <string>
#include <vector>

namespace render_demo {
inline constexpr u32 FogCheckFrame = 30;

enum FogCheckWord : u32 {
  FogGeometry,
  FogSkyNonZero,
  FogZeroOnGeometry,
  FogRgbMismatch,
  FogQMismatch,
  FogBand0,
  FogBand1,
  // 127 minus the smallest band-0 q, so a zeroed buffer means no sample
  FogBand0QMinInv,
  FogBand0QMax,
  FogMaxErrorBits,
  FogWidth,
  FogHeight,
  FogWritten,
  FogCheckWords = 16,
};

inline constexpr u32 AoCheckFrame = 30;
inline constexpr u32 AoCheckMaxProbes = 512;

struct AoCheckReport {
  u32 encodes = 0;
  u32 frames = 0;
  u32 failedMaps = 0;
  u32 receivers = 0;
  u32 pixels = 0;
  std::vector<AoProbe> probes;
  std::vector<double> sums;
  std::vector<u32> receiverFrames;
  AoTemporalSummary temporal;
};

struct FwdEnvCheckReport {
  u32 encodes = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  std::vector<FwdEnvSample> samples;
};

struct SsilCheckReport {
  u32 encodes = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  u32 texels = 0;
  u32 outOfRange = 0;
  u32 bounced = 0;
  std::vector<SsilSample> samples;
};

struct UnlitCheckReport {
  u32 encodes = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  std::vector<UnlitSample> before;
  std::vector<UnlitSample> after;
};

struct CompanionCheckReport {
  u32 encodes = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  std::array<std::vector<CompanionSample>, CompanionPhaseCount> phases;
};

struct ClassCheckReport {
  u32 encodes = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  std::array<std::vector<ClassSample>, ClassPhaseCount> phases;
};

enum OverlaySample : u32 { OverlayOpaqueDone30, OverlayPrePost30, OverlayPrePost32, OverlaySampleCount };

struct OverlayCheckReport {
  u32 encodesA = 0;
  u32 encodesB = 0;
  bool ready = false;
  u32 width = 0;
  u32 height = 0;
  // Packed RGBA8 at the quad centre, in OverlaySample order
  std::array<u32, OverlaySampleCount> colors{};
};

struct GraphPassOptions {
  bool vignette = false;
  std::string vignetteAnchor;
  bool depthCheck = false;
  u32 gpuCull = 0;
  bool cullCheck = false;
  bool cullTimings = false;
  bool motionCheck = false;
  bool motionView = false;
  bool fogCheck = false;
  bool aoCheck = false;
  bool fwdEnvCheck = false;
  bool unlitCheck = false;
  bool companionCheck = false;
  bool overlayCheck = false;
  bool classCheck = false;
  bool ssilCheck = false;
  bool captureListCheck = false;
  bool retainedCheck = false;
};

struct GraphPassReport {
  u32 vignetteEncodes = 0;
  u32 probeEncodes = 0;
  bool probeReady = false;
  std::array<f32, 4> probe{};
  aurora::render::DepthSource source = aurora::render::DepthSource::Unavailable;
  CullReport cull;
  u32 motionEncodes = 0;
  bool motionReady = false;
  std::array<f32, MotionProbeFloats> motion{};
  u32 fogEncodes = 0;
  bool fogReady = false;
  std::array<u32, FogCheckWords> fog{};
  AoCheckReport ao;
  FwdEnvCheckReport fwdEnv;
  SsilCheckReport ssil;
  UnlitCheckReport unlit;
  CompanionCheckReport companion;
  OverlayCheckReport overlay;
  ClassCheckReport classes;
};

bool pipelines_compiling() noexcept;

bool register_graph_passes(const GraphPassOptions& options, std::string& error);
void update_graph_passes(u32 frame);
void finish_graph_passes();
GraphPassReport graph_pass_report();
void unregister_graph_passes();
} // namespace render_demo

#pragma once

#include "cull_pass.hpp"
#include "graph_checks.hpp"

#include <aurora/render_graph.hpp>
#include <dolphin/types.h>

#include <array>
#include <string>

namespace render_demo {
struct GraphPassOptions {
  bool vignette = false;
  std::string vignetteAnchor;
  bool depthCheck = false;
  u32 gpuCull = 0;
  bool cullCheck = false;
  bool cullTimings = false;
  bool motionCheck = false;
  bool motionView = false;
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
};

bool pipelines_compiling() noexcept;

bool register_graph_passes(const GraphPassOptions& options, std::string& error);
void update_graph_passes(u32 frame);
void finish_graph_passes();
GraphPassReport graph_pass_report();
void unregister_graph_passes();
} // namespace render_demo

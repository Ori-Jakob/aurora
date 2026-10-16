#pragma once

#include <aurora/render_graph.hpp>
#include <dolphin/types.h>

#include <array>
#include <string>

namespace render_demo {
struct GraphPassOptions {
  bool vignette = false;
  std::string vignetteAnchor;
  bool depthCheck = false;
};

struct GraphPassReport {
  u32 vignetteEncodes = 0;
  u32 probeEncodes = 0;
  bool probeReady = false;
  std::array<f32, 4> probe{};
  aurora::render::DepthSource source = aurora::render::DepthSource::Unavailable;
};

bool register_graph_passes(const GraphPassOptions& options, std::string& error);
void update_graph_passes(u32 frame);
void finish_graph_passes();
GraphPassReport graph_pass_report();
void unregister_graph_passes();
} // namespace render_demo

#pragma once

#include <dolphin/types.h>

#include <string>

namespace render_demo {
struct GraphPassOptions {
  bool vignette = false;
  std::string vignetteAnchor;
};

struct GraphPassReport {
  u32 vignetteEncodes = 0;
};

bool register_graph_passes(const GraphPassOptions& options, std::string& error);
void update_graph_passes(u32 frame);
void finish_graph_passes();
GraphPassReport graph_pass_report();
void unregister_graph_passes();
} // namespace render_demo

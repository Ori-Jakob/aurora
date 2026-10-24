#pragma once

#include <dolphin/types.h>

#include <optional>
#include <string>

namespace render_demo {
struct CullPassOptions {
  u32 instances = 0;
  bool check = false;
  bool timings = false;
};

enum class CullCheck : u8 { NotRequested, SkippedNull, Passed, Failed, NotReady };

struct CullReport {
  bool registered = false;
  std::string clamp;
  u32 requested = 0, instances = 0, cullEncodes = 0, drawEncodes = 0, checkFrame = 0, visible = 0, ambiguous = 0;
  CullCheck check = CullCheck::NotRequested;
  std::string failure;
  std::optional<f32> cullMs, drawMs;
  std::optional<f32> cullP90Ms, drawP90Ms;
  u32 timingSamples = 0;
};

bool register_cull_passes(const CullPassOptions& options, std::string& error);
void update_cull_passes(u32 frame);
void finish_cull_passes();
CullReport cull_report();
bool unregister_cull_passes();
void release_cull_passes();
} // namespace render_demo

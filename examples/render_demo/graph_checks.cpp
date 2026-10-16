#include "graph_checks.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

namespace render_demo {
namespace {
constexpr u32 VignettePeriodFrames = 240;
} // namespace

VignetteParams vignette_params(u32 frame) noexcept {
  // Driven by the frame number rather than the clock, so every run renders the same frames
  const f32 phase = static_cast<f32>(frame % VignettePeriodFrames) *
                    (2.f * std::numbers::pi_v<f32> / static_cast<f32>(VignettePeriodFrames));
  return {
      .strength = 0.45f + 0.15f * std::sin(phase),
      .inner = 0.55f,
      .outer = 1.f,
      .pad = 0.f,
  };
}

bool is_unknown_anchor_warning(AuroraLogLevel level, std::string_view module, std::string_view message) noexcept {
  return level == LOG_WARNING && module == "aurora::render::graph" &&
         message.find("' has unknown anchor '") != std::string_view::npos;
}

std::optional<std::string> check_depth_probe(std::span<const f32, 4> values, bool reversedZ) {
  const f32 clearDepth = reversedZ ? 0.f : 1.f;
  char text[128];
  // The shader stores the texture width last, so anything below one means it never wrote
  if (values[3] < 1.f) {
    return std::string{"probe shader did not run"};
  }
  if (values[0] != clearDepth) {
    std::snprintf(text, sizeof(text), "sky probe %.6g is not the clear depth %.6g", values[0], clearDepth);
    return std::string{text};
  }
  for (u32 i = 1; i < 3; ++i) {
    if (!(values[i] > 0.f && values[i] < 1.f)) {
      std::snprintf(text, sizeof(text), "ground probe %u = %.6g is outside (0, 1)", i, values[i]);
      return std::string{text};
    }
  }
  const bool closer = reversedZ ? values[2] > values[1] : values[2] < values[1];
  if (!closer) {
    std::snprintf(text, sizeof(text), "near probe %.6g is not closer than mid probe %.6g", values[2], values[1]);
    return std::string{text};
  }
  return std::nullopt;
}
} // namespace render_demo

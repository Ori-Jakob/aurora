#include "graph_checks.hpp"

#include <cmath>
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
} // namespace render_demo

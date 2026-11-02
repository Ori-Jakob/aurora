#include "graph_checks.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string_view>

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

AoCheckSummary summarize_ao_check(std::span<const AoProbeStat> probes, double coverage) {
  AoCheckSummary summary{.coverage = coverage};
  u32 open = 0;
  for (const AoProbeStat& probe : probes) {
    const auto cls = static_cast<size_t>(probe.cls);
    ++summary.probes[cls];
    if (!probe.receiver) {
      continue;
    }
    ++summary.receivers[cls];
    summary.mean[cls] += probe.visibility;
    open += probe.cls == AoProbeClass::Open && probe.visibility >= AoOpenMin ? 1 : 0;
  }
  for (size_t cls = 0; cls < AoProbeClassCount; ++cls) {
    summary.mean[cls] = summary.receivers[cls] != 0 ? summary.mean[cls] / summary.receivers[cls] : 0.0;
  }
  const u32 openReceivers = summary.receivers[static_cast<size_t>(AoProbeClass::Open)];
  summary.openShare = openReceivers != 0 ? static_cast<double>(open) / openReceivers : 0.0;
  return summary;
}

std::optional<std::string> check_ao(const AoCheckSummary& summary) {
  constexpr std::array<std::string_view, AoProbeClassCount> Names{"crease", "open-ground", "strip", "interior"};
  char text[160];
  if (summary.coverage < AoCoverageMin) {
    std::snprintf(text, sizeof(text), "receiver coverage %.3f is below %.2f", summary.coverage, AoCoverageMin);
    return std::string{text};
  }
  for (size_t cls = 0; cls < AoProbeClassCount; ++cls) {
    if (summary.receivers[cls] * 2 < summary.probes[cls] || summary.receivers[cls] == 0) {
      std::snprintf(text, sizeof(text), "only %u of %u %.*s probes landed on receivers", summary.receivers[cls],
                    summary.probes[cls], static_cast<int>(Names[cls].size()), Names[cls].data());
      return std::string{text};
    }
  }
  const auto mean = [&](AoProbeClass cls) { return summary.mean[static_cast<size_t>(cls)]; };
  if (mean(AoProbeClass::Crease) > AoCreaseMax) {
    std::snprintf(text, sizeof(text), "crease mean %.3f is above %.2f", mean(AoProbeClass::Crease), AoCreaseMax);
    return std::string{text};
  }
  if (summary.openShare < AoOpenShare) {
    std::snprintf(text, sizeof(text), "%.1f%% of open-ground probes reach %.2f", summary.openShare * 100.0, AoOpenMin);
    return std::string{text};
  }
  const double strip = mean(AoProbeClass::Strip);
  const double interior = mean(AoProbeClass::Interior);
  if (strip < AoStripMin || std::abs(strip - interior) > AoStripInteriorGap) {
    std::snprintf(text, sizeof(text), "strip mean %.3f against interior %.3f", strip, interior);
    return std::string{text};
  }
  return std::nullopt;
}
} // namespace render_demo

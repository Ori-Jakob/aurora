#include "graph_checks.hpp"

#include <algorithm>
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
    if (cls >= AoProbeClassCount) {
      continue;
    }
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
AoTemporalOutcome check_ao_temporal(const AoTemporalSummary& summary) {
  AoTemporalOutcome outcome{.trailMargin = 1.0};
  char text[160];
  const AoSigma& sigma = summary.sigma;
  if (sigma.frames == 0 || sigma.covered == 0) {
    std::snprintf(text, sizeof(text), "the sigma window read %u frames over %u receiver pixels", sigma.frames,
                  sigma.covered);
    outcome.failure = std::string{text};
    return outcome;
  }
  for (u32 frame = AoTrailFirst; frame <= AoTrailLast; ++frame) {
    const auto it = std::ranges::find(summary.trail, frame, &AoTrailFrame::frame);
    if (it == summary.trail.end()) {
      std::snprintf(text, sizeof(text), "frame %u was not read back", frame);
      outcome.failure = std::string{text};
      return outcome;
    }
    if (it->trailReceivers * 2 < AoTrailProbeCount || it->openReceivers == 0) {
      std::snprintf(text, sizeof(text), "frame %u: only %u of %u trail probes landed on receivers", frame,
                    it->trailReceivers, AoTrailProbeCount);
      outcome.failure = std::string{text};
      return outcome;
    }
    ++outcome.trailFrames;
    outcome.trailMargin = std::min(outcome.trailMargin, it->trail - (it->open - AoTrailMargin));
    if (it->trail < it->open - AoTrailMargin) {
      std::snprintf(text, sizeof(text), "frame %u: trail mean %.3f is below open-ground %.3f - %.2f", frame, it->trail,
                    it->open, AoTrailMargin);
      outcome.failure = std::string{text};
      return outcome;
    }
  }
  const auto& [before, settle, steady] = summary.contact;
  if (before.empty() || settle.empty() || steady.empty() || settle.size() != steady.size() ||
      before.size() != steady.size()) {
    std::snprintf(text, sizeof(text), "the contact band was not read back on frames %u, %u and %u", AoContactFrames[0],
                  AoContactFrames[1], AoContactFrames[2]);
    outcome.failure = std::string{text};
    return outcome;
  }
  double removal = 0.0;
  double shadow = 0.0;
  u32 shadowProbes = 0;
  for (size_t i = 0; i < steady.size(); ++i) {
    if (std::isnan(steady[i])) {
      continue;
    }
    if (!std::isnan(settle[i])) {
      removal += std::abs(double{settle[i]} - steady[i]);
      ++outcome.removalProbes;
    }
    if (!std::isnan(before[i])) {
      shadow += double{steady[i]} - before[i];
      ++shadowProbes;
    }
  }
  if (outcome.removalProbes * 2 < steady.size() || shadowProbes == 0) {
    std::snprintf(text, sizeof(text), "only %u of %zu contact probes landed on receivers", outcome.removalProbes,
                  steady.size());
    outcome.failure = std::string{text};
    return outcome;
  }
  outcome.removal = removal / outcome.removalProbes;
  outcome.shadow = shadow / shadowProbes;
  if (outcome.shadow < AoSliderShadowMin) {
    std::snprintf(text, sizeof(text), "the slider darkened its contact band by only %.3f", outcome.shadow);
    outcome.failure = std::string{text};
  } else if (outcome.removal > AoRemovalMax) {
    std::snprintf(text, sizeof(text), "contact band changed by %.3f between frames %u and %u", outcome.removal,
                  AoSettleFrame, AoSteadyFrame);
    outcome.failure = std::string{text};
  }
  return outcome;
}
} // namespace render_demo

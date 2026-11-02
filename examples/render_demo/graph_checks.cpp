#include "graph_checks.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
namespace {
std::array<u8, 4> word_bytes(u32 word) noexcept {
  return {static_cast<u8>(word), static_cast<u8>(word >> 8), static_cast<u8>(word >> 16), static_cast<u8>(word >> 24)};
}

void judge(FwdEnvGate& gate, const std::array<u8, 4>& colour, const std::array<double, 3>& expected, double limit,
           bool sensitive) {
  double error = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    error = std::max(error, std::abs(static_cast<double>(colour[i]) - expected[i]));
  }
  ++gate.checked;
  gate.sensitive += sensitive ? 1 : 0;
  gate.failures += error > limit ? 1 : 0;
  gate.maxError = std::max(gate.maxError, error);
}

std::optional<std::string> gate_failure(const FwdEnvGate& gate, std::string_view name, u32 needed, double aoLimit,
                                        double limit) {
  char text[192];
  if (gate.sensitive < needed) {
    std::snprintf(text, sizeof(text), "no occlusion at probe: %.*s has %u of %u pixels at ao <= %.2f (needs %u)",
                  static_cast<int>(name.size()), name.data(), gate.sensitive, gate.checked, aoLimit, needed);
    return std::string{text};
  }
  if (gate.failures != 0) {
    std::snprintf(text, sizeof(text), "%.*s: %u of %u pixels off by up to %.2f LSB (limit %.0f)",
                  static_cast<int>(name.size()), name.data(), gate.failures, gate.checked, gate.maxError, limit);
    return std::string{text};
  }
  return std::nullopt;
}
} // namespace

FwdEnvSummary summarize_fwd_env(std::span<const FwdEnvSample> samples, const FwdEnvExpect& expect) {
  FwdEnvSummary summary{.samples = static_cast<u32>(samples.size())};
  std::optional<u8> alpha;
  for (const FwdEnvSample& sample : samples) {
    const std::array<u8, 4> colour = word_bytes(sample.color);
    const std::array<u8, 4> record = word_bytes(sample.fog);
    const std::array<u8, 4> ao = word_bytes(sample.ao);
    const std::array<u8, 4> normal = word_bytes(sample.normal);
    const f32 depth = std::bit_cast<f32>(sample.depthBits);
    const u32 code = normal[3];
    if ((code == 0 || depth == 0.f) && ao[0] != 255) {
      ++summary.exactOneViolations;
    }
    const double a = ao[0] / 255.0;
    const u32 q7 = std::min<u32>(record[3] >> 1, 127);
    const double q = q7 / 127.0;
    const auto& p = expect.colors[static_cast<size_t>(sample.cls)];
    if (sample.cls == FwdEnvProbeClass::Card) {
      judge(summary.card, colour, {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])},
            1.0, a <= 0.9);
      continue;
    }
    double facing = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      facing += (normal[i] / 255.0 * 2.0 - 1.0) * expect.groundNormal[i];
    }
    // Only pixels whose receiver is the ground the strips were cut from can hold the strip
    if (code == 0 || depth == 0.f || facing < 0.98) {
      continue;
    }
    const auto expected = [&](double factor) {
      return std::array<double, 3>{p[0] * factor * (1.0 - q) + record[0] * q, p[1] * factor * (1.0 - q) + record[1] * q,
                                   p[2] * factor * (1.0 - q) + record[2] * q};
    };
    if (alpha && *alpha != colour[3]) {
      ++summary.alphaMismatches;
    }
    alpha = colour[3];
    if (sample.cls == FwdEnvProbeClass::StripB) {
      judge(summary.plain, colour, expected(1.0), 2.0, a <= 0.9);
    } else if (q7 == 127) {
      judge(summary.far, colour,
            {static_cast<double>(record[0]), static_cast<double>(record[1]), static_cast<double>(record[2])}, 1.0,
            a <= 0.95);
    } else if (q7 <= 15) {
      judge(summary.near, colour, expected(a), 2.0, a <= 0.9);
    }
  }
  return summary;
}

UnlitCheckSummary summarize_unlit(std::span<const UnlitSample> before, std::span<const UnlitSample> after) {
  UnlitCheckSummary summary{.minColourMove = UINT32_MAX};
  const auto surface_class = [](const UnlitSample& s) { return (s.surface[1] >> 16) & 0xFFu; };
  for (size_t i = 0; i < before.size() && i < after.size(); ++i) {
    const UnlitSample& a = before[i];
    const UnlitSample& b = after[i];
    if (a.cls == UnlitProbeClass::Control) {
      ++summary.control;
      summary.controlPbr += surface_class(a) != 0 || surface_class(b) != 0 ? 1 : 0;
      continue;
    }
    ++summary.ground;
    summary.classMismatches += surface_class(a) != 1 || surface_class(b) != 1 ? 1 : 0;
    for (const UnlitSample* s : {&a, &b}) {
      const u32 roughness = s->surface[1] & 0xFFu;
      const u32 metalness = (s->surface[1] >> 8) & 0xFFu;
      summary.paramMismatches += roughness != UnlitRoughness || metalness != 0 ? 1 : 0;
    }
    summary.albedoChanges += (a.surface[0] & 0xFFFFFFu) != (b.surface[0] & 0xFFFFFFu) ? 1 : 0;
    u32 move = 0;
    for (u32 c = 0; c < 3; ++c) {
      const int x = static_cast<int>((a.color >> (8 * c)) & 0xFFu);
      const int y = static_cast<int>((b.color >> (8 * c)) & 0xFFu);
      move = std::max(move, static_cast<u32>(std::abs(x - y)));
    }
    summary.minColourMove = std::min(summary.minColourMove, move);
    summary.maxColourMove = std::max(summary.maxColourMove, move);
  }
  if (summary.ground == 0) {
    summary.minColourMove = 0;
  }
  return summary;
}

std::optional<std::string> check_unlit(const UnlitCheckSummary& summary) {
  char text[160];
  if (summary.ground == 0 || summary.control == 0) {
    return std::string{"no ground or control probe on screen"};
  }
  if (summary.classMismatches != 0 || summary.paramMismatches != 0) {
    std::snprintf(text, sizeof(text), "%u ground probes off class 1 and %u reads off roughness %u or metalness 0",
                  summary.classMismatches, summary.paramMismatches, UnlitRoughness);
    return std::string{text};
  }
  if (summary.albedoChanges != 0) {
    std::snprintf(text, sizeof(text), "%u ground probes changed base colour with the sun", summary.albedoChanges);
    return std::string{text};
  }
  if (summary.minColourMove < UnlitColourMove) {
    std::snprintf(text, sizeof(text), "the sun moved gx.color by only %u at a ground probe (needs %u)",
                  summary.minColourMove, UnlitColourMove);
    return std::string{text};
  }
  if (summary.controlPbr != 0) {
    std::snprintf(text, sizeof(text), "%u NoPbr probes carry a PBR class", summary.controlPbr);
    return std::string{text};
  }
  return std::nullopt;
}

std::optional<std::string> check_fwd_env(const FwdEnvSummary& summary) {
  char text[160];
  if (summary.exactOneViolations != 0) {
    std::snprintf(text, sizeof(text), "%u sampled pixels with code 0 or clear depth have ao.final below 1",
                  summary.exactOneViolations);
    return std::string{text};
  }
  if (summary.alphaMismatches != 0) {
    std::snprintf(text, sizeof(text), "%u strip pixels changed the EFB alpha", summary.alphaMismatches);
    return std::string{text};
  }
  if (auto failure = gate_failure(summary.far, "strip A at q7 = 127", FwdEnvFarSensitive, 0.95, 1.0)) {
    return failure;
  }
  if (auto failure = gate_failure(summary.near, "strip A at q7 <= 15", FwdEnvNearSensitive, 0.9, 2.0)) {
    return failure;
  }
  if (auto failure = gate_failure(summary.plain, "strip B", FwdEnvNearSensitive, 0.9, 2.0)) {
    return failure;
  }
  return gate_failure(summary.card, "card D", FwdEnvCardSensitive, 0.9, 1.0);
}
} // namespace render_demo

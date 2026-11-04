#include "bench_report.hpp"
#include "companion_fixtures.hpp"
#include "gfx/recording.hpp"
#include "graph_checks.hpp"
#include "graph_passes.hpp"
#include "hint_checks.hpp"
#include "hints.hpp"
#include "indirect_bench.hpp"
#include "options.hpp"
#include "scene.hpp"

#include <aurora/aurora.h>
#include <aurora/event.h>
#include <aurora/gfx.hpp>
#include <aurora/gpu_systems.hpp>
#include <aurora/render.hpp>
#include <aurora/render_caps.hpp>
#include <aurora/texture.hpp>
#include <aurora/trace.hpp>
#include <dolphin/gx.h>
#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

// Last because it redefines main
#include <aurora/main.h>

namespace {
enum ExitCode : int {
  ExitSuccess = 0,
  ExitUsage = 1,
  ExitTimeout = 2,
  ExitLoggedError = 3,
  ExitBackendMismatch = 4,
  ExitSceneInit = 5,
  ExitTraceFailed = 6,
  ExitRenderGraphCheck = 7,
  ExitHintCheck = 8,
  ExitCullCheck = 9,
  ExitBench = 10,
  ExitMotionCheck = 11,
  ExitFogCheck = 12,
  ExitAoCheck = 13,
  ExitFwdEnvCheck = 14,
  ExitUnlitCheck = 15,
  ExitCompanionCheck = 16,
  ExitOverlayCheck = 17,
  ExitClassCheck = 18,
  ExitSsilCheck = 19,
};

std::atomic_uint32_t g_errorCount = 0;
std::atomic_uint32_t g_anchorWarnings = 0;
std::atomic_uint32_t g_unknownClassLines = 0;

void log_callback(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  if (render_demo::is_unknown_anchor_warning(level, module != nullptr ? module : "", {message, len})) {
    g_anchorWarnings.fetch_add(1, std::memory_order_relaxed);
  }
  if (level == LOG_WARNING && module != nullptr && std::string_view{module} == "aurora::render::material" &&
      std::string_view{message, len}.find("\"ground/lava\"") != std::string_view::npos) {
    g_unknownClassLines.fetch_add(1, std::memory_order_relaxed);
  }
  const char* levelStr = "INFO";
  FILE* out = stdout;
  switch (level) {
  case LOG_DEBUG:
    levelStr = "DEBUG";
    break;
  case LOG_INFO:
    break;
  case LOG_WARNING:
    levelStr = "WARNING";
    break;
  case LOG_ERROR:
    levelStr = "ERROR";
    out = stderr;
    g_errorCount.fetch_add(1, std::memory_order_relaxed);
    break;
  case LOG_FATAL:
    levelStr = "FATAL";
    out = stderr;
    break;
  }
  std::fprintf(out, "[%s: %s;%.*s]\n", levelStr, module, static_cast<int>(len), message);
  if (level == LOG_FATAL) {
    std::fflush(out);
    std::abort();
  }
}

std::string_view depth_source_name(aurora::render::DepthSource source) noexcept {
  switch (source) {
  case aurora::render::DepthSource::Direct:
    return "direct";
  case aurora::render::DepthSource::Copy:
    return "copy";
  default:
    return "unavailable";
  }
}

std::optional<std::string> depth_check_failure(const render_demo::GraphPassReport& report, bool nullBackend,
                                               bool reversedZ) {
  // A probe that runs without scene depth, or never runs with it, means DisablePass did not follow the source
  const bool available = report.source != aurora::render::DepthSource::Unavailable;
  if (available == (report.probeEncodes == 0)) {
    char text[96];
    std::snprintf(text, sizeof(text), "probe encoded %u times with scene depth %.*s", report.probeEncodes,
                  static_cast<int>(depth_source_name(report.source).size()), depth_source_name(report.source).data());
    return std::string{text};
  }
  // The null backend runs no shaders, so its values say nothing
  if (!available || nullBackend) {
    return std::nullopt;
  }
  if (!report.probeReady) {
    return std::string{"probe readback did not complete"};
  }
  return render_demo::check_depth_probe(report.probe, reversedZ);
}

constexpr std::array<f32, 3> MotionTolerancePx{0.05f, 0.05f, 0.25f};

std::optional<std::string> motion_check_failure(const render_demo::GraphPassReport& report, bool nullBackend) {
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return std::nullopt;
  }
  if (report.motionEncodes == 0) {
    return std::string{"gbuf.motion unavailable"};
  }
  if (!report.motionReady) {
    return std::string{"probe readback did not complete"};
  }
  const auto& values = report.motion;
  constexpr u32 Header = render_demo::MotionProbeCount * render_demo::MotionProbeStride;
  if (values[Header + 3] < 1.f) {
    return std::string{"probe shader did not run"};
  }
  const auto frame = static_cast<u32>(values[Header]);
  const f32 width = values[Header + 1];
  const f32 height = values[Header + 2];
  char text[192];
  for (u32 i = 0; i < MotionTolerancePx.size(); ++i) {
    const f32* probe = values.data() + i * render_demo::MotionProbeStride;
    const std::array<f32, 2> ndc{(probe[4] + 0.5f) / width * 2.f - 1.f, 1.f - (probe[5] + 0.5f) / height * 2.f};
    const auto expected = render_demo::motion_probe_expected(static_cast<render_demo::MotionProbe>(i), frame, ndc);
    if (!expected) {
      std::snprintf(text, sizeof(text), "probe %u has no expected motion at frame %u", i, frame);
      return std::string{text};
    }
    const f32 dx = (probe[0] - (*expected)[0]) * 0.5f * width;
    const f32 dy = (probe[1] - (*expected)[1]) * 0.5f * height;
    const f32 error = std::sqrt(dx * dx + dy * dy);
    if (!(error <= MotionTolerancePx[i])) {
      std::snprintf(text, sizeof(text),
                    "probe %u at (%g, %g) frame %u read motion (%.6g, %.6g), expected (%.6g, %.6g): %.4g px off", i,
                    probe[4], probe[5], frame, probe[0], probe[1], (*expected)[0], (*expected)[1], error);
      return std::string{text};
    }
  }
  const f32 groundReactive = values[2];
  const f32 waterReactive =
      values[static_cast<u32>(render_demo::MotionProbe::Water) * render_demo::MotionProbeStride + 2];
  if (groundReactive != 0.f || !(waterReactive > 0.f)) {
    std::snprintf(text, sizeof(text), "reactive reads %.6g on the ground and %.6g on the water", groundReactive,
                  waterReactive);
    return std::string{text};
  }
  return std::nullopt;
}

std::optional<std::string> fog_check_failure(const render_demo::GraphPassReport& report,
                                             const render_demo::FogResolveCounts& resolves, bool nullBackend) {
  using namespace render_demo;
  char text[160];
  if (resolves.checked == 0 || resolves.mismatches != 0) {
    std::snprintf(text, sizeof(text), "%u of %u fog resolves disagreed with the scene layout", resolves.mismatches,
                  resolves.checked);
    return std::string{text};
  }
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return std::nullopt;
  }
  if (report.fogEncodes == 0) {
    return std::string{"gbuf.fog or gx.depth unavailable"};
  }
  if (!report.fogReady) {
    return std::string{"check readback did not complete"};
  }
  const auto& w = report.fog;
  if (w[FogWritten] != 1) {
    return std::string{"check shader did not run"};
  }
  const double pixels = static_cast<double>(w[FogWidth]) * w[FogHeight];
  const double geometry = w[FogGeometry];
  if (w[FogRgbMismatch] != 0 || w[FogQMismatch] != 0 || w[FogSkyNonZero] != 0) {
    std::snprintf(text, sizeof(text), "%u colour and %u q mismatches, %u records on sky pixels", w[FogRgbMismatch],
                  w[FogQMismatch], w[FogSkyNonZero]);
    return std::string{text};
  }
  if (geometry == 0.0 || (geometry - w[FogZeroOnGeometry]) / geometry < 0.99) {
    std::snprintf(text, sizeof(text), "%u of %u geometry pixels have no record", w[FogZeroOnGeometry], w[FogGeometry]);
    return std::string{text};
  }
  if (w[FogBand0] < 0.05 * pixels || w[FogBand1] < 0.002 * pixels) {
    std::snprintf(text, sizeof(text), "band coverage %u and %u of %.0f pixels", w[FogBand0], w[FogBand1], pixels);
    return std::string{text};
  }
  const u32 minQ = 127 - w[FogBand0QMinInv];
  if (minQ > 15 || w[FogBand0QMax] != 127) {
    std::snprintf(text, sizeof(text), "band 0 q spans %u-%u, not 15 or less to 127", minQ, w[FogBand0QMax]);
    return std::string{text};
  }
  return std::nullopt;
}

struct AoCheckOutcome {
  render_demo::AoCheckSummary summary;
  render_demo::AoTemporalOutcome temporal;
  std::optional<std::string> failure;
};

AoCheckOutcome ao_check_outcome(const render_demo::AoCheckReport& report, bool nullBackend) {
  using namespace render_demo;
  std::vector<AoProbeStat> probes;
  probes.reserve(report.probes.size());
  for (size_t i = 0; i < report.probes.size() && i < report.sums.size(); ++i) {
    const bool receiver = report.frames != 0 && report.receiverFrames[i] == report.frames;
    const double visibility = report.frames != 0 ? report.sums[i] / report.frames : 0.0;
    probes.push_back({.cls = report.probes[i].cls, .visibility = visibility, .receiver = receiver});
  }
  const double coverage = report.pixels != 0 ? static_cast<double>(report.receivers) / report.pixels : 0.0;
  AoCheckOutcome outcome{.summary = summarize_ao_check(probes, coverage)};
  if (nullBackend) {
    return outcome;
  }
  outcome.temporal = check_ao_temporal(report.temporal);
  if (report.encodes == 0) {
    outcome.failure = "ao.final or gx.normal unavailable";
  } else if (report.frames == 0) {
    outcome.failure = "no frame was read back";
  } else if (const std::optional<std::string> failure = check_ao(outcome.summary)) {
    outcome.failure = failure;
  } else {
    outcome.failure = outcome.temporal.failure;
  }
  return outcome;
}

struct FwdEnvOutcome {
  render_demo::FwdEnvSummary summary;
  std::optional<std::string> failure;
};

FwdEnvOutcome fwd_env_outcome(const render_demo::FwdEnvCheckReport& report, bool nullBackend) {
  FwdEnvOutcome outcome{.summary = render_demo::summarize_fwd_env(report.samples, render_demo::fwd_env_expect())};
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return outcome;
  }
  if (report.encodes == 0) {
    outcome.failure = "ao.final or a scene target never reached demo.fwd_env_check";
  } else if (!report.ready || report.samples.empty()) {
    outcome.failure = "check readback did not complete";
  } else {
    outcome.failure = render_demo::check_fwd_env(outcome.summary);
  }
  return outcome;
}

struct UnlitOutcome {
  render_demo::UnlitCheckSummary summary;
  std::optional<std::string> failure;
};

UnlitOutcome unlit_outcome(const render_demo::UnlitCheckReport& report, bool nullBackend) {
  UnlitOutcome outcome{.summary = render_demo::summarize_unlit(report.before, report.after)};
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return outcome;
  }
  if (report.encodes == 0) {
    outcome.failure = "debug.surface_v1 never reached demo.unlit_check";
  } else if (!report.ready || report.before.empty() || report.after.empty()) {
    outcome.failure = "check readback did not complete";
  } else {
    outcome.failure = render_demo::check_unlit(outcome.summary);
  }
  return outcome;
}

struct ClassRun {
  std::string setupError;
  aurora::texture::MaterialManifestRegistration manifest;
  uint32_t createdAtEdit = 0;
  uint32_t createdAfterEdit = 0;
};

aurora::render::OpenPbrParams class_params(int kind) {
  aurora::render::OpenPbrParams params;
  switch (kind) {
  case 0:
    params.coatWeight = 1.f;
    params.coatRoughness = 0.1f;
    break;
  case 1:
    params.baseMetalness = 1.f;
    params.specularRoughness = 0.3f;
    break;
  case 2:
    params.fuzzWeight = 1.f;
    break;
  default:
    params.geometryThinWalled = true;
    params.thinWalledWrap = 0.5f;
    params.twoSided = true;
    break;
  }
  return params;
}

void start_class_check(ClassRun& run) {
  constexpr std::array<std::string_view, 4> Names{"test/coat", "test/metal", "test/fuzz", "test/thin"};
  for (int i = 0; i < 4; ++i) {
    aurora::render::register_material_class(static_cast<uint8_t>(render_demo::ClassHintRegistered + i), Names[i],
                                            class_params(i));
  }
  const std::string text = render_demo::class_manifest(false);
  run.manifest = aurora::texture::register_material_manifest(
      text, {.layer = aurora::texture::MaterialLayer::User, .label = "render_demo class check"});
  if (text.empty() || run.manifest.status != aurora::texture::MaterialManifestStatus::Accepted) {
    run.setupError = "the class check manifest was not accepted";
  }
}

void edit_class_check(ClassRun& run) {
  run.createdAtEdit = aurora_get_stats()->createdPipelines;
  aurora::render::OpenPbrParams coat = class_params(0);
  coat.coatWeight = 0.5f;
  aurora::render::register_material_class(render_demo::ClassHintRegistered, "test/coat", coat);
  if (aurora::texture::update_material_manifest(run.manifest, render_demo::class_manifest(true)) !=
      aurora::texture::MaterialManifestStatus::Accepted) {
    run.setupError = "the edited class check manifest was not accepted";
  }
}

struct ClassOutcome {
  render_demo::ClassCheckSummary summary;
  std::optional<std::string> failure;
};

ClassOutcome class_outcome(const render_demo::ClassCheckReport& report, const ClassRun& run, bool nullBackend) {
  const char* vertexColor = SDL_getenv("AURORA_RENDER_VERTEX_COLOR");
  const bool baked = vertexColor != nullptr && std::string_view{vertexColor} == "baked";
  ClassOutcome outcome{.summary = render_demo::summarize_classes(report.phases, baked)};
  const uint32_t compiles = run.createdAfterEdit - run.createdAtEdit;
  const uint32_t unknownLines = g_unknownClassLines.load(std::memory_order_relaxed);
  if (!run.setupError.empty()) {
    outcome.failure = run.setupError;
  } else if (unknownLines != 1) {
    outcome.failure = "ground/lava logged " + std::to_string(unknownLines) + " unknown-class lines, expected 1";
  } else if (nullBackend) {
    return outcome;
  } else if (report.encodes == 0) {
    outcome.failure = "debug.surface_v1 or gx.normal never reached demo.class_check";
  } else if (!report.ready) {
    outcome.failure = "check readback did not complete";
  } else if (compiles != 0) {
    outcome.failure = "the class and manifest edit compiled " + std::to_string(compiles) + " pipelines";
  } else {
    outcome.failure = render_demo::check_classes(outcome.summary);
  }
  return outcome;
}

struct SsilOutcome {
  render_demo::SsilCheckSummary summary;
  std::optional<std::string> failure;
};

SsilOutcome ssil_outcome(const render_demo::SsilCheckReport& report, bool nullBackend) {
  SsilOutcome outcome{
      .summary = render_demo::summarize_ssil(report.samples, report.texels, report.bounced, report.outOfRange)};
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return outcome;
  }
  if (report.encodes == 0) {
    outcome.failure = "ssil.irr never reached demo.ssil_check";
  } else if (!report.ready || report.samples.empty()) {
    outcome.failure = "check readback did not complete";
  } else {
    outcome.failure = render_demo::check_ssil(outcome.summary);
  }
  return outcome;
}

struct CompanionRun {
  std::string setupError;
  uint64_t loadedMapped = 0;
  uint64_t loadedMat = 0;
  uint64_t companionSkips = 0;
  uint32_t createdAtScalars = 0;
  uint32_t createdAfterScalars = 0;
  render_demo::CompanionCounts emit;
  uint64_t loadedManifest = 0;
};

bool delight_env() {
  const char* value = SDL_getenv("AURORA_RENDER_DELIGHT");
  return value != nullptr && std::string_view{value} == "on";
}

struct CompanionOutcome {
  render_demo::CompanionCheckSummary summary;
  std::optional<std::string> failure;
};

CompanionOutcome companion_outcome(const render_demo::CompanionCheckReport& report, const CompanionRun& run,
                                   bool nullBackend) {
  CompanionOutcome outcome{
      .summary = render_demo::summarize_companions(report.phases, render_demo::companion_axes(), delight_env())};
  if (!run.setupError.empty()) {
    outcome.failure = run.setupError;
    return outcome;
  }
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return outcome;
  }
  const uint32_t compiles = run.createdAfterScalars - run.createdAtScalars;
  if (run.loadedMapped < render_demo::CompanionNrmRoles ||
      run.loadedMat < run.loadedMapped + render_demo::CompanionMatRoles || run.emit.emit == 0 ||
      (run.emit.delight != 0) != delight_env() ||
      run.loadedManifest < run.emit.loaded + render_demo::CompanionNrmRoles) {
    outcome.failure = "companion maps were not resident at their probe frames";
  } else if (compiles != 0) {
    outcome.failure = "the manifest normal scalars compiled " + std::to_string(compiles) + " pipelines";
  } else if (run.companionSkips != 0) {
    outcome.failure = "companion draws were skipped for want of a companion group";
  } else if (report.encodes == 0) {
    outcome.failure = "gx.normal and debug.surface_v1 never reached demo.companion_probe";
  } else if (!report.ready) {
    outcome.failure = "check readback did not complete";
  } else {
    outcome.failure = render_demo::check_companions(outcome.summary);
  }
  return outcome;
}

struct OverlayShape {
  uint64_t ops = 0;
  uint64_t passes = 0;
  uint64_t commands = 0;
  std::string checkFrame;
  std::string lateFrame;
};

// From frame 20 on, so the run-to-run timing of the G-buffer switch never shows in the counts
constexpr uint32_t OverlayShapeFirstFrame = 20;

OverlayShape overlay_shape(const std::vector<aurora::gfx::detail::testing::RecordedShape>& shapes) {
  OverlayShape shape;
  const auto list = [](const std::vector<uint32_t>& counts) {
    std::string text;
    for (const uint32_t count : counts) {
      text += (text.empty() ? "" : ",") + std::to_string(count);
    }
    return text;
  };
  for (const auto& frame : shapes) {
    if (frame.frameIndex < OverlayShapeFirstFrame) {
      continue;
    }
    shape.ops += frame.ops;
    shape.passes += frame.passCommands.size();
    for (const uint32_t count : frame.passCommands) {
      shape.commands += count;
    }
    if (frame.frameIndex == render_demo::OverlayCheckFrame) {
      shape.checkFrame = list(frame.passCommands);
    } else if (frame.frameIndex == render_demo::OverlayLateFrame) {
      shape.lateFrame = list(frame.passCommands);
    }
  }
  return shape;
}

uint32_t pack_rgb(const std::array<u8, 4>& rgba) { return rgba[0] | (rgba[1] << 8) | (rgba[2] << 16); }

uint32_t overlay_blend_rgb() {
  const render_demo::OverlayColors colors = render_demo::overlay_colors();
  const double alpha = colors.quad[3] / 255.0;
  std::array<u8, 4> blend{};
  for (size_t i = 0; i < 3; ++i) {
    blend[i] = static_cast<u8>(std::lround(colors.quad[i] * alpha + colors.box[i] * (1.0 - alpha)));
  }
  return pack_rgb(blend);
}

uint32_t rgb_distance(uint32_t a, uint32_t b) {
  uint32_t worst = 0;
  for (uint32_t shift = 0; shift < 24; shift += 8) {
    const int delta = static_cast<int>((a >> shift) & 0xff) - static_cast<int>((b >> shift) & 0xff);
    worst = std::max<uint32_t>(worst, static_cast<uint32_t>(std::abs(delta)));
  }
  return worst;
}

std::optional<std::string> overlay_failure(const render_demo::OverlayCheckReport& report,
                                           const aurora::gfx::OverlayStats& stats, std::string_view mode,
                                           bool nullBackend) {
  const bool force = mode == "force";
  const bool deferring = force || mode == "drop" || mode == "restore";
  char text[160];
  if (deferring ? stats.draws < 2 : stats.draws != 0) {
    std::snprintf(text, sizeof(text), "%llu draws deferred under %.*s", static_cast<unsigned long long>(stats.draws),
                  static_cast<int>(mode.size()), mode.data());
    return std::string{text};
  }
  if (stats.refused != 0 || stats.dropped != 0) {
    return std::string{"deferred draws were refused or dropped"};
  }
  // Frame 32 has no OPAQUE_DONE marker, so it alone restores late; every other frame replays
  if (force && (stats.restoredLate != 1 || stats.restoredSeal != 0 || stats.restoredUnapplied != 0)) {
    return std::string{"restores under force were not exactly the late frame"};
  }
  // The null backend runs no shaders, so its values say nothing
  if (nullBackend) {
    return std::nullopt;
  }
  if (report.encodesA == 0 || report.encodesB == 0) {
    return std::string{"gx.color never reached demo.overlay_probe_a and _b"};
  }
  if (!report.ready) {
    return std::string{"probe readback did not complete"};
  }
  const uint32_t box = pack_rgb(render_demo::overlay_colors().box);
  const auto rgb = [&](render_demo::OverlaySample sample) { return report.colors[sample] & 0xffffffu; };
  if (rgb(render_demo::OverlayOpaqueDone30) != box || rgb(render_demo::OverlayPrePost32) != box) {
    return std::string{"the box was not exactly in place before gx.overlay on frame 30 or after frame 32's restore"};
  }
  if (force ? rgb_distance(rgb(render_demo::OverlayPrePost30), overlay_blend_rgb()) > 1
            : rgb(render_demo::OverlayPrePost30) != box) {
    return std::string{force ? "the replayed quad did not blend over the box" : "the quad showed through the box"};
  }
  return std::nullopt;
}

struct HintCounterSnapshot {
  std::array<render_demo::ObservedCounters, render_demo::LastHintId + 1 - render_demo::FirstHintId> ids{};
  uint64_t suppressedDraws = 0;
  uint64_t capturedDraws = 0;
  aurora::render::RetainedStats retained;
};

// aurora_end_frame drained every frame's FIFO, so the core counters already cover each hint the demo emitted
HintCounterSnapshot read_hint_counters() {
  HintCounterSnapshot snapshot;
  for (uint32_t id = render_demo::FirstHintId; id <= render_demo::LastHintId; ++id) {
    const aurora::render::HintCounters counters = aurora::render::hint_counters(static_cast<uint16_t>(id));
    snapshot.ids[id - render_demo::FirstHintId] = {
        .decoded = counters.decoded,
        .skipped = counters.skipped,
        .malformed = counters.malformed,
        .late = counters.late,
        .bytes = counters.bytes,
    };
  }
  const aurora::render::DrawRouteCounters route = aurora::render::draw_route_counters();
  // A captured CAPTURE_ONLY draw stays hidden too, so the hint check counts it with the suppressed ones
  snapshot.suppressedDraws = route.suppressed + route.captured;
  snapshot.capturedDraws = route.captured;
  snapshot.retained = aurora::render::retained_stats();
  return snapshot;
}

render_demo::ObservedCounters snapshot_counters(u16 id, void* userData) {
  return static_cast<const HintCounterSnapshot*>(userData)->ids[id - render_demo::FirstHintId];
}

void print_hint_summary(const render_demo::HintTally& tally, const HintCounterSnapshot& snapshot) {
  render_demo::ObservedCounters total;
  for (uint32_t id = render_demo::FirstHintId; id <= render_demo::LastHintId; ++id) {
    if (id >= GX_AURORA_GPU_UPLOAD && id <= GX_AURORA_GPU_DESTROY_SYSTEM) {
      continue;
    }
    const auto& counters = snapshot.ids[id - render_demo::FirstHintId];
    total.decoded += counters.decoded;
    total.skipped += counters.skipped;
    total.malformed += counters.malformed;
    total.late += counters.late;
    total.bytes += counters.bytes;
  }
  std::printf(
      "render_demo: hints: emitted %llu, decoded %llu, skipped %llu, malformed %llu, late %llu, bytes %llu; "
      "suppressed draws %llu\n",
      static_cast<unsigned long long>(tally.total()), static_cast<unsigned long long>(total.decoded),
      static_cast<unsigned long long>(total.skipped), static_cast<unsigned long long>(total.malformed),
      static_cast<unsigned long long>(total.late), static_cast<unsigned long long>(total.bytes),
      static_cast<unsigned long long>(snapshot.suppressedDraws));
  const aurora::render::RetainedStats& retained = snapshot.retained;
  if (retained.active) {
    std::printf(
        "render_demo: retained: instances %u, statics %u, undelivered statics %u, created %llu, evicted %llu, "
        "captured %llu, capture-only %llu, captured draws %llu, lost captures %llu, coverage %s\n",
        retained.instances, retained.staticInstances, retained.undeliveredStatics,
        static_cast<unsigned long long>(retained.created), static_cast<unsigned long long>(retained.evicted),
        static_cast<unsigned long long>(retained.captured), static_cast<unsigned long long>(retained.capturedOnly),
        static_cast<unsigned long long>(snapshot.capturedDraws), static_cast<unsigned long long>(retained.lostCaptures),
        retained.coverageComplete ? "complete" : "open");
  }
}

constexpr uint64_t CustomDrawProbeUniformBytes = 256;

constexpr std::string_view NoopSystemWgsl = R"(@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z)
fn noop(@builtin(global_invocation_id) id: vec3<u32>) {
  if (stream_active(id)) {
    state_store(0u, state_load(0u));
  }
}
)";
constexpr std::array<aurora::render::gpu::KernelDesc, 1> NoopKernels{{{.entryPoint = "noop"}}};
constexpr uint32_t NoopStateBytes = 256;
constexpr std::array<aurora::render::gpu::KernelDesc, 1> CopyKernels{
    {{.entryPoint = "main", .workgroupSize = {64, 1, 1}}}};

struct GpuDemo {
  aurora::render::gpu::SystemId system = aurora::render::gpu::InvalidSystem;
  aurora::render::gpu::SystemId copy = aurora::render::gpu::InvalidSystem;
  bool available = false;
};

GpuDemo create_gpu_demo(const render_demo::Options& options) {
  if (!options.gpuArrays) {
    return {};
  }
  GpuDemo demo{.available = aurora::render::gpu::systems_available()};
  if (demo.available) {
    demo.system = aurora::render::gpu::create_system({
        .name = "demo_noop",
        .wgsl = NoopSystemWgsl,
        .kernels = NoopKernels,
        .persistentBytes = NoopStateBytes,
    });
    demo.copy = aurora::render::gpu::create_system({
        .name = "demo_copy",
        .library = "systems/copy",
        .kernels = CopyKernels,
    });
  }
  return demo;
}

void update_gpu_demo(const GpuDemo& demo, uint32_t frame) {
  if (demo.system != aurora::render::gpu::InvalidSystem &&
      aurora::render::gpu::system_status(demo.system) == aurora::render::gpu::SystemStatus::Ready) {
    aurora::render::gpu::update_persistent(demo.system, 0, &frame, sizeof(frame));
  }
}

aurora::render::gpu::SystemStatus settled_status(aurora::render::gpu::SystemId system) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  auto status = aurora::render::gpu::system_status(system);
  while (status == aurora::render::gpu::SystemStatus::Pending && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    status = aurora::render::gpu::system_status(system);
  }
  return status;
}

// The copy kernel compiles asynchronously, so frame 0 waits for it and every frame then dispatches
void settle_gpu_demo(const GpuDemo& demo) {
  if (demo.copy == aurora::render::gpu::InvalidSystem) {
    return;
  }
  GXDrawDone();
  settled_status(demo.copy);
}

struct GpuReport {
  aurora::render::gpu::Stats stats;
  std::string systems;
};

GpuReport finish_gpu_demo(const GpuDemo& demo) {
  GpuReport report{.stats = aurora::render::gpu::stats()};
  if (!demo.available) {
    report.systems = "unavailable";
  } else {
    uint32_t ready = 0;
    for (const auto system : {demo.system, demo.copy}) {
      if (system != aurora::render::gpu::InvalidSystem &&
          settled_status(system) == aurora::render::gpu::SystemStatus::Ready) {
        ++ready;
      }
    }
    report.systems = "ready " + std::to_string(ready) + "/2";
  }
  for (const auto system : {demo.system, demo.copy}) {
    if (system != aurora::render::gpu::InvalidSystem) {
      aurora::render::gpu::destroy_system(system);
    }
  }
  return report;
}

bool gpu_failed(const GpuReport& report, uint64_t expectedUploads) {
  const auto& stats = report.stats;
  return stats.suppressedDraws != 0 || stats.failedHandles != 0 || stats.rejected != 0 || stats.refused != 0 ||
         stats.staleBinds != 0 || stats.uploads != expectedUploads || stats.dispatchesRejected != 0 ||
         stats.dispatchesRefused != 0;
}

struct CustomDrawProbe {
  aurora::gfx::DrawTypeId type = aurora::gfx::InvalidDrawType;
  wgpu::BindGroupLayout layout;
  wgpu::Buffer uniformBuffer;
  wgpu::BindGroup group;
};
CustomDrawProbe g_customDrawProbe;

// Leaves foreign bindings behind, so a GX draw after it fails validation unless the encoder rebinds
void draw_custom_draw_probe(const aurora::gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void*,
                            size_t, void* userdata) {
  auto& probe = *static_cast<CustomDrawProbe*>(userdata);
  if (!probe.layout) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
        .buffer = {.type = wgpu::BufferBindingType::Uniform},
    };
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "render_demo custom draw probe",
        .entryCount = 1,
        .entries = &entry,
    };
    probe.layout = ctx.device.CreateBindGroupLayout(&descriptor);
  }
  if (probe.uniformBuffer.Get() != ctx.uniformBuffer.Get()) {
    const wgpu::BindGroupEntry entry{.binding = 0, .buffer = ctx.uniformBuffer, .size = CustomDrawProbeUniformBytes};
    const wgpu::BindGroupDescriptor descriptor{
        .label = "render_demo custom draw probe",
        .layout = probe.layout,
        .entryCount = 1,
        .entries = &entry,
    };
    probe.group = ctx.device.CreateBindGroup(&descriptor);
    probe.uniformBuffer = ctx.uniformBuffer;
  }
  pass.SetIndexBuffer(ctx.indexBuffer, wgpu::IndexFormat::Uint32, 0, 4);
  pass.SetBindGroup(1, probe.group);
}

std::string_view profile_name(AuroraFeatureProfile profile) noexcept {
  switch (profile) {
  case FEATURE_PROFILE_COMPAT:
    return "compat";
  case FEATURE_PROFILE_METAL:
    return "metal";
  case FEATURE_PROFILE_METAL32:
    return "metal32";
  default:
    return "full";
  }
}

bool pump_events(void*) {
  const AuroraEvent* event = aurora_update();
  while (event != nullptr && event->type != AURORA_NONE) {
    if (event->type == AURORA_EXIT) {
      return false;
    }
    ++event;
  }
  return true;
}

bool write_text(const std::string& path, const std::string& text) {
  std::error_code ec;
  const std::filesystem::path target{path};
  if (target.has_parent_path()) {
    std::filesystem::create_directories(target.parent_path(), ec);
  }
  std::ofstream out{target, std::ios::binary | std::ios::trunc};
  out << text;
  return static_cast<bool>(out);
}

// Measurement-only exception to principle 13: own command buffers on the main thread, before the first frame
int run_bench(const render_demo::Options& options, const AuroraInfo& info) {
  aurora::gfx::synchronize();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (render_demo::pipelines_compiling() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }
  std::string error;
  if (render_demo::pipelines_compiling()) {
    error = "pipelines were still compiling after 10 s";
  } else if (aurora::gfx::current_frame() != 0) {
    error = "a frame has already begun";
  }
  render_demo::BenchRun run;
  bool ok = error.empty();
  if (ok) {
    const aurora::render::Caps& caps = aurora::render::caps();
    const wgpu::Device device = aurora::gfx::device();
    const render_demo::BenchDevice bench{
        .instance = device.GetAdapter().GetInstance(),
        .device = device,
        .queue = aurora::gfx::queue(),
        .colorFormat = aurora::gfx::color_format(),
        .sampleCount = aurora::gfx::sample_count(),
        .minUniformBufferOffsetAlignment = caps.limits.minUniformBufferOffsetAlignment,
        .multiDrawIndirect = caps.multiDrawIndirect,
        .timestampQuery = caps.timestampQuery,
        .nullBackend = caps.backend == BACKEND_NULL,
    };
    run.device = "aurora";
    run.backend = render_demo::backend_name(info.backend);
    run.profile = profile_name(caps.profile);
    run.adapter = caps.adapterName;
    run.adapterKind = aurora::render::adapter_kind_name(caps.adapterKind);
    run.vendorId = caps.vendorId;
    run.deviceId = caps.deviceId;
    run.maxImmediateSize = caps.limits.maxImmediateSize;
    run.vertexStorageBuffers =
        aurora::render::stage_binding_limits(caps, aurora::render::ShaderStage::Vertex).storageBuffers;
    run.indirectFirstInstance = caps.indirectFirstInstance;
    const render_demo::BenchOptions benchOptions{
        .blocks = options.benchBlocks,
        .width = options.width,
        .height = options.height,
        .uniformPermille = options.benchUniformPermille,
        .texturePermille = options.benchTexturePermille,
    };
    ok = render_demo::run_indirect_bench(bench, benchOptions, pump_events, nullptr, run, error);
    if (ok && !write_text(options.benchPath, render_demo::bench_json(run))) {
      ok = false;
      error = "failed to write " + options.benchPath;
    }
  }
  aurora_shutdown();
  if (!ok) {
    std::fprintf(stderr, "render_demo: bench: %s\n", error.c_str());
    return ExitBench;
  }
  std::printf("%s\n", render_demo::bench_summary(run, options.benchPath).c_str());
  if (options.failOnError && g_errorCount.load(std::memory_order_relaxed) != 0) {
    return ExitLoggedError;
  }
  return ExitSuccess;
}

std::string cull_status(const render_demo::CullReport& report) {
  switch (report.check) {
  case render_demo::CullCheck::Passed:
    return "passed";
  case render_demo::CullCheck::SkippedNull:
    return "skipped (null backend)";
  case render_demo::CullCheck::NotRequested:
    return "skipped (not requested)";
  default:
    return "FAILED: " + report.failure;
  }
}

void print_cull_report(const render_demo::CullReport& report, uint32_t frames, bool timings) {
  if (report.clamp.starts_with("unavailable")) {
    std::printf("render_demo: cull: %s\n", report.clamp.c_str());
    return;
  }
  char timing[128] = "gpu timings off";
  if (timings && report.cullMs && report.drawMs) {
    std::snprintf(timing, sizeof(timing), "gpu cull %.3f ms, draw %.3f ms (median of %u frames, profiler on)",
                  *report.cullMs, *report.drawMs, report.timingSamples);
  } else if (timings) {
    std::snprintf(timing, sizeof(timing), "gpu cull n/a, draw n/a (median of 0 frames, profiler on)");
  }
  std::printf(
      "render_demo: cull: %u instances (requested %u), visible %u at frame %u, encodes %u/%u of %u, check %s, "
      "ambiguous %u; %s\n",
      report.instances, report.requested, report.visible, report.checkFrame, report.cullEncodes, report.drawEncodes,
      frames, cull_status(report).c_str(), report.ambiguous, timing);
  if (timings && report.cullP90Ms && report.drawP90Ms) {
    std::printf("render_demo: cull: p90 gpu cull %.3f ms, draw %.3f ms (profiler on)\n", *report.cullP90Ms,
                *report.drawP90Ms);
  }
}

bool cull_failed(const render_demo::CullReport& report, uint32_t requestedFrames, uint32_t frames, bool nullBackend) {
  if (report.registered && requestedFrames != 0 && (report.cullEncodes != frames || report.drawEncodes != frames)) {
    return true;
  }
  return report.check == render_demo::CullCheck::Failed ||
         (report.check == render_demo::CullCheck::NotReady && !nullBackend);
}
} // namespace

int main(int argc, char* argv[]) {
  std::string error;
  const auto parsed = render_demo::parse_options(argc, argv, SDL_getenv("AURORA_BACKEND"), error);
  if (!parsed) {
    std::fprintf(stderr, "render_demo: %s\n%.*s", error.c_str(), static_cast<int>(render_demo::usage().size()),
                 render_demo::usage().data());
    return ExitUsage;
  }
  const render_demo::Options& options = *parsed;
  if (options.help) {
    std::printf("%.*s", static_cast<int>(render_demo::usage().size()), render_demo::usage().data());
    return ExitSuccess;
  }
  if (!options.dataDir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(options.dataDir, ec);
    if (ec) {
      std::fprintf(stderr, "render_demo: failed to create %s: %s\n", options.dataDir.c_str(), ec.message().c_str());
      return ExitUsage;
    }
  }

  if (options.aoCheck || options.fwdEnvCheck || options.ssilCheck) {
    // The checks run at AO High whatever the saved settings say
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_AO", "high", false);
  }
  if (options.ssilCheck) {
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_SSIL", "on", false);
  }
  if (options.unlitCheck || options.companionCheck) {
    // PBR ships no built bit in M9, so only the verification hook can reach the Surface layout
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_PBR", "on", false);
  }
  if (options.unlitCheck) {
    // Below a blend of 1 the composite shows the light-free preview, so dimming the sun would move nothing
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_LEGACY_BLEND", "1", true);
  }
  if (options.companionCheck) {
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_MAT_MAPS", "on", false);
  }
  if (options.overlayCheck) {
    // Deferral arms only on a G-buffer layout, which the demo's own settings never request
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_GBUFFER", "on", false);
  }
  if (options.classCheck) {
    SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_PBR", "on", false);
  }

  AuroraConfig config{};
  config.appName = "render_demo";
  config.userPath = options.dataDir.empty() ? nullptr : options.dataDir.c_str();
  config.cachePath = config.userPath;
  config.desiredBackend = options.backend;
  config.featureProfile = options.profile;
  config.windowWidth = options.width;
  config.windowHeight = options.height;
  config.windowPosX = -1;
  config.windowPosY = -1;
  config.logCallback = &log_callback;
  config.logLevel = LOG_INFO;
  config.msaa = options.msaa;
  const AuroraInfo info = aurora_initialize(argc, argv, &config);
  if (options.backend != BACKEND_AUTO && info.backend != options.backend) {
    std::fprintf(stderr, "render_demo: requested %.*s but got %.*s\n",
                 static_cast<int>(render_demo::backend_name(options.backend).size()),
                 render_demo::backend_name(options.backend).data(),
                 static_cast<int>(render_demo::backend_name(info.backend).size()),
                 render_demo::backend_name(info.backend).data());
    aurora_shutdown();
    return ExitBackendMismatch;
  }
  if (!options.benchPath.empty()) {
    return run_bench(options, info);
  }
  if (options.customDrawProbe) {
    g_customDrawProbe.type = aurora::gfx::register_draw_type(
        {.label = "render_demo custom draw probe", .draw = draw_custom_draw_probe, .userdata = &g_customDrawProbe});
    if (g_customDrawProbe.type == aurora::gfx::InvalidDrawType) {
      std::fprintf(stderr, "render_demo: custom draw probe registration failed\n");
      aurora_shutdown();
      return ExitSceneInit;
    }
  }
  // The receiver is installed once at initialisation, so the answer is fixed for the whole run
  const bool decoding = aurora::render::hints_enabled();
  render_demo::SceneOptions sceneOptions{.hints = options.hints,
                                         .captureOnlyBox = options.hints && decoding,
                                         .customDrawProbe = g_customDrawProbe.type,
                                         .gpuArrays = options.gpuArrays,
                                         .gpuPadMiB = options.gpuPadMiB,
                                         .fogBands = options.fogCheck || options.fwdEnvCheck,
                                         .aoCheck = options.aoCheck,
                                         .fwdEnvCheck = options.fwdEnvCheck,
                                         .unlitCheck = options.unlitCheck,
                                         .companionCheck = options.companionCheck,
                                         .overlayCheck = options.overlayCheck,
                                         .classCheck = options.classCheck,
                                         .ssilCheck = options.ssilCheck};
  bool decodingChanged = false;

  GXInit(nullptr, 0);
  AuroraSetViewportPolicy(AURORA_VIEWPORT_FIT);
  if (!render_demo::init_scene()) {
    std::fprintf(stderr, "render_demo: scene initialisation failed\n");
    aurora_shutdown();
    return ExitSceneInit;
  }
  if (std::string graphError; !render_demo::register_graph_passes({.vignette = options.vignette,
                                                                   .vignetteAnchor = options.vignetteAnchor,
                                                                   .depthCheck = options.depthCheck,
                                                                   .gpuCull = options.gpuCull,
                                                                   .cullCheck = options.cullCheck,
                                                                   .cullTimings = options.cullTimings,
                                                                   .motionCheck = options.motionCheck,
                                                                   .motionView = options.motionView,
                                                                   .fogCheck = options.fogCheck,
                                                                   .aoCheck = options.aoCheck,
                                                                   .fwdEnvCheck = options.fwdEnvCheck,
                                                                   .unlitCheck = options.unlitCheck,
                                                                   .companionCheck = options.companionCheck,
                                                                   .overlayCheck = options.overlayCheck,
                                                                   .classCheck = options.classCheck,
                                                                   .ssilCheck = options.ssilCheck},
                                                                  graphError)) {
    std::fprintf(stderr, "render_demo: render graph setup failed: %s\n", graphError.c_str());
    render_demo::shutdown_scene();
    aurora_shutdown();
    return ExitRenderGraphCheck;
  }
  const GpuDemo gpuDemo = create_gpu_demo(options);
  sceneOptions.gpuCopySystem = gpuDemo.copy;
  CompanionRun companionRun;
  if (options.companionCheck) {
    render_demo::prepare_companion_loads();
  }
  ClassRun classRun;
  if (options.classCheck) {
    start_class_check(classRun);
    aurora::render::Settings settings = aurora::render::effective_settings();
    settings.legacyBlend = 0.f;
    aurora::render::apply_settings(settings);
  }
  if (options.overlayCheck) {
    aurora::gfx::detail::testing::keep_frame_shapes(true);
  }

  const auto start = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds{options.timeoutSeconds};
  uint32_t frame = 0;
  bool exiting = false;
  bool paused = false;
  bool timedOut = false;
  bool traceRequested = false;
  while (!exiting) {
    const AuroraEvent* event = aurora_update();
    while (event != nullptr && event->type != AURORA_NONE) {
      switch (event->type) {
      case AURORA_EXIT:
        exiting = true;
        break;
      case AURORA_PAUSED:
        paused = true;
        break;
      case AURORA_UNPAUSED:
        paused = false;
        break;
      default:
        break;
      }
      ++event;
    }
    if (options.timeoutSeconds != 0 && std::chrono::steady_clock::now() - start >= timeout) {
      timedOut = true;
      break;
    }
    if (!options.tracePath.empty() && !traceRequested && frame == options.traceStart) {
      traceRequested = true;
      if (!aurora::trace::start_capture(
              {.path = options.tracePath, .frameCount = options.traceFrames, .label = "render_demo"})) {
        const auto status = aurora::trace::capture_status();
        render_demo::shutdown_scene();
        aurora_shutdown();
        std::fprintf(stderr, "render_demo: failed to start trace capture: %s\n", status.error.c_str());
        return ExitTraceFailed;
      }
    }
    if (options.companionCheck && companionRun.setupError.empty()) {
      const auto set_blend = [](float blend) {
        aurora::render::Settings settings = aurora::render::effective_settings();
        settings.legacyBlend = blend;
        aurora::render::apply_settings(settings);
      };
      const std::span<const u8> ground = render_demo::ground_texture_base();
      if (frame == render_demo::CompanionLoadFrame || frame == render_demo::CompanionMatFrame) {
        const bool mat = frame == render_demo::CompanionMatFrame;
        if (mat) {
          set_blend(0.f);
        }
        render_demo::load_companion_fixtures(mat ? render_demo::CompanionFixture::Mat
                                                 : render_demo::CompanionFixture::Nrm,
                                             ground, companionRun.setupError);
      } else if (frame == render_demo::CompanionBlendOneFrame) {
        set_blend(1.f);
      } else if (frame == render_demo::CompanionMappedFrame) {
        companionRun.loadedMapped = render_demo::companion_counts().loaded;
      } else if (frame == render_demo::CompanionMatProbeFrame) {
        companionRun.loadedMat = render_demo::companion_counts().loaded;
      } else if (frame == render_demo::CompanionScalarFrame) {
        companionRun.createdAtScalars = aurora_get_stats()->createdPipelines;
        render_demo::set_companion_manifest(false, ground, companionRun.setupError);
      } else if (frame == render_demo::CompanionScalarProbeFrame) {
        companionRun.createdAfterScalars = aurora_get_stats()->createdPipelines;
      } else if (frame == render_demo::CompanionEmitFrame) {
        render_demo::load_companion_fixtures(render_demo::CompanionFixture::Emit, ground, companionRun.setupError);
      } else if (frame == render_demo::CompanionEmitProbeFrame) {
        companionRun.emit = render_demo::companion_counts();
      } else if (frame == render_demo::CompanionManifestFrame) {
        if (render_demo::load_companion_fixtures(render_demo::CompanionFixture::Nrm, ground, companionRun.setupError)) {
          render_demo::set_companion_manifest(true, ground, companionRun.setupError);
        }
      } else if (frame == render_demo::CompanionManifestProbeFrame) {
        companionRun.loadedManifest = render_demo::companion_counts().loaded;
      }
    }
    if (options.classCheck && classRun.setupError.empty()) {
      if (frame == render_demo::ClassEditFrame) {
        edit_class_check(classRun);
      } else if (frame == render_demo::ClassCheckSecondFrame) {
        classRun.createdAfterEdit = aurora_get_stats()->createdPipelines;
      } else if (frame == render_demo::ClassBlendOneFrame) {
        aurora::render::Settings settings = aurora::render::effective_settings();
        settings.legacyBlend = 1.f;
        aurora::render::apply_settings(settings);
      }
    }
    // Params set before aurora_begin_frame apply to the frame it begins
    render_demo::update_graph_passes(frame);
    if (exiting || paused || !aurora_begin_frame()) {
      continue;
    }
    decodingChanged |= aurora::render::hints_enabled() != decoding;
    if (frame == 0) {
      settle_gpu_demo(gpuDemo);
    }
    update_gpu_demo(gpuDemo, frame);
    render_demo::draw_scene(frame, sceneOptions);
    aurora_end_frame();
    ++frame;
    if (options.frames != 0 && frame >= options.frames) {
      break;
    }
  }

  // Every recorded frame has been encoded and its after-submit callbacks have run once this returns
  aurora::gfx::synchronize();
  render_demo::finish_graph_passes();
  const auto graphReport = render_demo::graph_pass_report();
  if (options.companionCheck) {
    companionRun.companionSkips = render_demo::companion_counts().companionSkips;
    render_demo::release_companion_fixtures();
  }
  if (options.classCheck) {
    aurora::texture::unregister_material_manifest(classRun.manifest);
  }
  const aurora::gfx::OverlayStats overlayStats = aurora::gfx::overlay_stats();
  const OverlayShape overlayShape =
      options.overlayCheck ? overlay_shape(aurora::gfx::detail::testing::take_frame_shapes()) : OverlayShape{};
  if (options.overlayCheck) {
    aurora::gfx::detail::testing::keep_frame_shapes(false);
  }
  HintCounterSnapshot hintCounters = read_hint_counters();
  const bool gpuChannel = options.gpuArrays || options.gpuPadMiB != 0;
  const GpuReport gpuReport = gpuChannel ? finish_gpu_demo(gpuDemo) : GpuReport{};
  render_demo::unregister_graph_passes();
  if (g_customDrawProbe.type != aurora::gfx::InvalidDrawType) {
    aurora::gfx::unregister_draw_type(g_customDrawProbe.type);
    g_customDrawProbe = {};
  }
  render_demo::shutdown_scene();
  aurora_shutdown();
  const auto traceStatus = aurora::trace::capture_status();

  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  const uint32_t errors = g_errorCount.load(std::memory_order_relaxed);
  const auto backend = render_demo::backend_name(info.backend);
  std::printf("render_demo: %u frames on %.*s in %lld ms, %u errors\n", frame, static_cast<int>(backend.size()),
              backend.data(), static_cast<long long>(elapsed), errors);
  const uint32_t anchorWarnings = g_anchorWarnings.load(std::memory_order_relaxed);
  if (options.vignette) {
    std::printf("render_demo: graph: vignette %u/%u encodes, %u unknown-anchor warnings\n", graphReport.vignetteEncodes,
                frame, anchorWarnings);
  }
  const std::optional<std::string> depthFailure =
      options.depthCheck
          ? depth_check_failure(graphReport, info.backend == BACKEND_NULL, aurora::gfx::uses_reversed_z())
          : std::nullopt;
  if (options.depthCheck) {
    std::string status;
    if (depthFailure) {
      status = "FAILED: " + *depthFailure;
    } else if (graphReport.source == aurora::render::DepthSource::Unavailable) {
      status = "skipped (scene depth unavailable)";
    } else if (info.backend == BACKEND_NULL) {
      status = "skipped (null backend)";
    } else {
      status = "passed";
    }
    const std::string_view source = depth_source_name(graphReport.source);
    std::printf("render_demo: depth check %s (%.*s): sky %.6g, mid %.6g, near %.6g\n", status.c_str(),
                static_cast<int>(source.size()), source.data(), graphReport.probe[0], graphReport.probe[1],
                graphReport.probe[2]);
  }
  const std::optional<std::string> motionFailure =
      options.motionCheck ? motion_check_failure(graphReport, info.backend == BACKEND_NULL) : std::nullopt;
  if (options.motionCheck) {
    const std::string status = motionFailure                  ? "FAILED: " + *motionFailure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& m = graphReport.motion;
    std::printf(
        "render_demo: motion check %s: frame %g, ground (%.6g, %.6g), box (%.6g, %.6g), tentacle (%.6g, "
        "%.6g), reactive ground %.3g water %.3g\n",
        status.c_str(), m[32], m[0], m[1], m[8], m[9], m[16], m[17], m[2], m[26]);
  }
  const render_demo::FogResolveCounts fogResolves = render_demo::fog_resolve_counts();
  const std::optional<std::string> fogFailure =
      options.fogCheck ? fog_check_failure(graphReport, fogResolves, info.backend == BACKEND_NULL) : std::nullopt;
  if (options.fogCheck) {
    const std::string status = fogFailure                     ? "FAILED: " + *fogFailure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& f = graphReport.fog;
    std::printf(
        "render_demo: fog check %s: geometry %u, no record %u, sky records %u, band 0 %u (q %u-%u), band 1 "
        "%u, colour mismatches %u, q mismatches %u, max error %.4g/127, resolves %u (%u live, %u wrong)\n",
        status.c_str(), f[render_demo::FogGeometry], f[render_demo::FogZeroOnGeometry], f[render_demo::FogSkyNonZero],
        f[render_demo::FogBand0], 127 - f[render_demo::FogBand0QMinInv], f[render_demo::FogBand0QMax],
        f[render_demo::FogBand1], f[render_demo::FogRgbMismatch], f[render_demo::FogQMismatch],
        std::bit_cast<float>(f[render_demo::FogMaxErrorBits]) * 127.f, fogResolves.checked, fogResolves.live,
        fogResolves.mismatches);
  }
  const AoCheckOutcome aoOutcome =
      options.aoCheck ? ao_check_outcome(graphReport.ao, info.backend == BACKEND_NULL) : AoCheckOutcome{};
  if (options.aoCheck) {
    const std::string status = aoOutcome.failure              ? "FAILED: " + *aoOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& a = aoOutcome.summary;
    std::printf(
        "render_demo: ao check %s: frames %u, coverage %.3f, crease %.3f (%u/%u), open %.3f at %.1f%% "
        "(%u/%u), strip %.3f (%u/%u), interior %.3f (%u/%u)\n",
        status.c_str(), graphReport.ao.frames, a.coverage, a.mean[0], a.receivers[0], a.probes[0], a.mean[1],
        a.openShare * 100.0, a.receivers[1], a.probes[1], a.mean[2], a.receivers[2], a.probes[2], a.mean[3],
        a.receivers[3], a.probes[3]);
    const auto& sigma = graphReport.ao.temporal.sigma;
    const auto& t = aoOutcome.temporal;
    std::printf(
        "render_demo: ao temporal: sigma %.5f over %u frames (coverage %.3f, mean %.4f), trail margin %+.3f over %u "
        "frames, removal %.4f (%u probes), slider shadow %.3f\n",
        sigma.sigma, sigma.frames, sigma.pixels != 0 ? static_cast<double>(sigma.covered) / sigma.pixels : 0.0,
        sigma.mean, t.trailMargin, t.trailFrames, t.removal, t.removalProbes, t.shadow);
  }
  const FwdEnvOutcome fwdEnvOutcome =
      options.fwdEnvCheck ? fwd_env_outcome(graphReport.fwdEnv, info.backend == BACKEND_NULL) : FwdEnvOutcome{};
  if (options.fwdEnvCheck) {
    const std::string status = fwdEnvOutcome.failure          ? "FAILED: " + *fwdEnvOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& s = fwdEnvOutcome.summary;
    std::printf(
        "render_demo: fwd-env check %s: %u samples at %ux%u; far A %u (%u at ao<=0.95, max %.2f), near A %u (%u at "
        "ao<=0.9, max %.2f), B %u (%u, max %.2f), card %u (%u, max %.2f), alpha mismatches %u, exact-1 "
        "violations %u\n",
        status.c_str(), s.samples, graphReport.fwdEnv.width, graphReport.fwdEnv.height, s.far.checked, s.far.sensitive,
        s.far.maxError, s.near.checked, s.near.sensitive, s.near.maxError, s.plain.checked, s.plain.sensitive,
        s.plain.maxError, s.card.checked, s.card.sensitive, s.card.maxError, s.alphaMismatches, s.exactOneViolations);
  }
  const UnlitOutcome unlitOutcome =
      options.unlitCheck ? unlit_outcome(graphReport.unlit, info.backend == BACKEND_NULL) : UnlitOutcome{};
  if (options.unlitCheck) {
    const std::string status = unlitOutcome.failure           ? "FAILED: " + *unlitOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& u = unlitOutcome.summary;
    std::printf(
        "render_demo: unlit check %s: %u ground probes (class off %u, params off %u, base colour changed "
        "%u, gx.color moved %u-%u) and %u NoPbr probes (%u with a PBR class) at %ux%u\n",
        status.c_str(), u.ground, u.classMismatches, u.paramMismatches, u.albedoChanges, u.minColourMove,
        u.maxColourMove, u.control, u.controlPbr, graphReport.unlit.width, graphReport.unlit.height);
  }
  const CompanionOutcome companionOutcome =
      options.companionCheck ? companion_outcome(graphReport.companion, companionRun, info.backend == BACKEND_NULL)
                             : CompanionOutcome{};
  if (options.companionCheck) {
    const std::string status = companionOutcome.failure       ? "FAILED: " + *companionOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& c = companionOutcome.summary;
    std::printf(
        "render_demo: companion check %s: %u probes, loaded %llu / wanted %u, then %llu / wanted %u, "
        "companion skips %llu; off-geometric %u, C1 tilt off %u (min %.3f), .mat tilt off %u, blend-one "
        "changes %u, orm off %u, albedo off %u; scalars off %u (compiled %u), emit off %u (leaks %u, loaded "
        "%llu), delight %s off %u (worst %d LSB, loaded %llu), manifest nrm off %u, roughness off %u at %ux%u\n",
        status.c_str(), c.probes, static_cast<unsigned long long>(companionRun.loadedMapped),
        render_demo::CompanionNrmRoles, static_cast<unsigned long long>(companionRun.loadedMat),
        render_demo::CompanionNrmRoles + render_demo::CompanionMatRoles,
        static_cast<unsigned long long>(companionRun.companionSkips), c.bareMismatches, c.tiltFailures, c.minTilt,
        c.matFailures, c.blendOneMismatches, c.ormFailures, c.albedoFailures, c.scalarFailures,
        companionRun.createdAfterScalars - companionRun.createdAtScalars, c.emitFailures, c.emitLeaks,
        static_cast<unsigned long long>(companionRun.emit.emit), delight_env() ? "on" : "off", c.delightFailures,
        c.maxDelightError, static_cast<unsigned long long>(companionRun.emit.delight), c.manifestNrmFailures,
        c.manifestRoughFailures, graphReport.companion.width, graphReport.companion.height);
  }
  const ClassOutcome classOutcome =
      options.classCheck ? class_outcome(graphReport.classes, classRun, info.backend == BACKEND_NULL) : ClassOutcome{};
  if (options.classCheck) {
    const std::string status = classOutcome.failure           ? "FAILED: " + *classOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    std::printf(
        "render_demo: class check %s: %u probes over %u phases, %u failed, edit compiled %u pipelines, "
        "unknown-class lines %u at %ux%u\n",
        status.c_str(), classOutcome.summary.probes, render_demo::ClassPhaseCount, classOutcome.summary.failures,
        classRun.createdAfterEdit - classRun.createdAtEdit, g_unknownClassLines.load(std::memory_order_relaxed),
        graphReport.classes.width, graphReport.classes.height);
  }
  const SsilOutcome ssilOutcome =
      options.ssilCheck ? ssil_outcome(graphReport.ssil, info.backend == BACKEND_NULL) : SsilOutcome{};
  if (options.ssilCheck) {
    const std::string status = ssilOutcome.failure            ? "FAILED: " + *ssilOutcome.failure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& b = ssilOutcome.summary.bands;
    const auto band = [&](render_demo::SsilProbeClass cls) { return b[static_cast<size_t>(cls)]; };
    const auto wall = band(render_demo::SsilProbeClass::WallFoot);
    const auto back = band(render_demo::SsilProbeClass::FinBack);
    const auto front = band(render_demo::SsilProbeClass::FinFront);
    const auto open = band(render_demo::SsilProbeClass::Open);
    std::printf(
        "render_demo: ssil check %s: %u of %u texels bounced, %u out of range at %ux%u; wall foot %u (%.4f, %.4f, "
        "%.4f), behind fin %u (%.4f, %.4f, %.4f), before fin %u (%.4f, %.4f, %.4f), open %u coverage %.4f\n",
        status.c_str(), ssilOutcome.summary.bounced, ssilOutcome.summary.texels, ssilOutcome.summary.outOfRange,
        graphReport.ssil.width, graphReport.ssil.height, wall.receivers, wall.mean[0], wall.mean[1], wall.mean[2],
        back.receivers, back.mean[0], back.mean[1], back.mean[2], front.receivers, front.mean[0], front.mean[1],
        front.mean[2], open.receivers, open.coverage);
  }
  const char* overlayEnv = SDL_getenv("AURORA_RENDER_OVERLAY_DEFER");
  const std::string_view overlayMode = overlayEnv != nullptr ? overlayEnv : "unset";
  const std::optional<std::string> overlayFailure =
      options.overlayCheck
          ? overlay_failure(graphReport.overlay, overlayStats, overlayMode, info.backend == BACKEND_NULL)
          : std::nullopt;
  if (options.overlayCheck) {
    const std::string status = overlayFailure                 ? "FAILED: " + *overlayFailure
                               : info.backend == BACKEND_NULL ? std::string{"skipped (null backend)"}
                                                              : std::string{"passed"};
    const auto& s = overlayStats;
    const auto& o = graphReport.overlay;
    std::printf(
        "render_demo: overlay check %s (%.*s): deferred %llu, replayed frames %llu, restored %llu seal / %llu late / "
        "%llu unapplied, discarded %llu, refused %llu, dropped %llu, in-place after %llu; A30 %06x, B30 %06x, B32 "
        "%06x, box %06x, blend %06x at %ux%u\n",
        status.c_str(), static_cast<int>(overlayMode.size()), overlayMode.data(),
        static_cast<unsigned long long>(s.draws), static_cast<unsigned long long>(s.replayFrames),
        static_cast<unsigned long long>(s.restoredSeal), static_cast<unsigned long long>(s.restoredLate),
        static_cast<unsigned long long>(s.restoredUnapplied), static_cast<unsigned long long>(s.discarded),
        static_cast<unsigned long long>(s.refused), static_cast<unsigned long long>(s.dropped),
        static_cast<unsigned long long>(s.inPlaceAfter), o.colors[render_demo::OverlayOpaqueDone30] & 0xffffffu,
        o.colors[render_demo::OverlayPrePost30] & 0xffffffu, o.colors[render_demo::OverlayPrePost32] & 0xffffffu,
        pack_rgb(render_demo::overlay_colors().box), overlay_blend_rgb(), o.width, o.height);
    std::printf(
        "render_demo: overlay shape from frame %u: ops %llu, passes %llu, commands %llu; frame %u [%s]; "
        "frame %u [%s]\n",
        OverlayShapeFirstFrame, static_cast<unsigned long long>(overlayShape.ops),
        static_cast<unsigned long long>(overlayShape.passes), static_cast<unsigned long long>(overlayShape.commands),
        render_demo::OverlayCheckFrame, overlayShape.checkFrame.c_str(), render_demo::OverlayLateFrame,
        overlayShape.lateFrame.c_str());
  }
  if (options.gpuCull != 0) {
    print_cull_report(graphReport.cull, frame, options.cullTimings);
  }
  const uint64_t expectedUploads = uint64_t{frame} * render_demo::gpu_uploads_per_frame(sceneOptions);
  if (gpuChannel) {
    const auto& stats = gpuReport.stats;
    std::printf(
        "render_demo: gpu: uploads %llu, binds %llu, suppressed %llu, failed %llu, rejected %llu, refused %llu, "
        "stale %llu, dispatches %llu, systems %s\n",
        static_cast<unsigned long long>(stats.uploads), static_cast<unsigned long long>(stats.binds),
        static_cast<unsigned long long>(stats.suppressedDraws), static_cast<unsigned long long>(stats.failedHandles),
        static_cast<unsigned long long>(stats.rejected), static_cast<unsigned long long>(stats.refused),
        static_cast<unsigned long long>(stats.staleBinds), static_cast<unsigned long long>(stats.dispatches),
        gpuReport.systems.c_str());
  }
  const bool cullFailure =
      options.gpuCull != 0 && cull_failed(graphReport.cull, options.frames, frame, info.backend == BACKEND_NULL);
  const render_demo::HintTally& tally = render_demo::hint_tally();
  print_hint_summary(tally, hintCounters);
  std::optional<std::string> hintFailure;
  if (options.checkHints) {
    hintFailure = decodingChanged ? std::optional<std::string>{"hints_enabled() changed during the run"}
                                  : render_demo::check_hints(tally, decoding, snapshot_counters, &hintCounters,
                                                             hintCounters.suppressedDraws);
    std::printf("render_demo: hint check %s%s (decoding %s)\n", hintFailure ? "FAILED: " : "passed",
                hintFailure ? hintFailure->c_str() : "", decoding ? "on" : "off");
  }
  if (timedOut) {
    return ExitTimeout;
  }
  if (options.vignette) {
    // Registered before frame 0's latch and counted after the final synchronize, so every frame encodes it
    if (options.frames != 0 && graphReport.vignetteEncodes != frame) {
      std::fprintf(stderr, "render_demo: demo.vignette encoded %u times in %u frames\n", graphReport.vignetteEncodes,
                   frame);
      return ExitRenderGraphCheck;
    }
    if (anchorWarnings != options.expectAnchorWarnings) {
      std::fprintf(stderr, "render_demo: expected %u unknown-anchor warnings, got %u\n", options.expectAnchorWarnings,
                   anchorWarnings);
      return ExitRenderGraphCheck;
    }
  }
  if (depthFailure) {
    return ExitRenderGraphCheck;
  }
  if (motionFailure) {
    return ExitMotionCheck;
  }
  if (fogFailure) {
    return ExitFogCheck;
  }
  if (aoOutcome.failure) {
    return ExitAoCheck;
  }
  if (fwdEnvOutcome.failure) {
    return ExitFwdEnvCheck;
  }
  if (unlitOutcome.failure) {
    return ExitUnlitCheck;
  }
  if (companionOutcome.failure) {
    return ExitCompanionCheck;
  }
  if (overlayFailure) {
    return ExitOverlayCheck;
  }
  if (classOutcome.failure) {
    return ExitClassCheck;
  }
  if (ssilOutcome.failure) {
    return ExitSsilCheck;
  }

  if (cullFailure) {
    return ExitCullCheck;
  }
  if (hintFailure) {
    return ExitHintCheck;
  }
  if (!options.tracePath.empty() && traceStatus.state != aurora::trace::CaptureState::Finished) {
    std::fprintf(stderr, "render_demo: trace capture did not finish: %s\n",
                 traceStatus.error.empty() ? "the capture window was never reached" : traceStatus.error.c_str());
    return ExitTraceFailed;
  }
  if (options.failOnError && (errors != 0 || (gpuChannel && gpu_failed(gpuReport, expectedUploads)))) {
    return ExitLoggedError;
  }
  return ExitSuccess;
}

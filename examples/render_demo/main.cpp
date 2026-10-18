#include "graph_checks.hpp"
#include "graph_passes.hpp"
#include "hint_checks.hpp"
#include "hints.hpp"
#include "options.hpp"
#include "scene.hpp"

#include <aurora/aurora.h>
#include <aurora/event.h>
#include <aurora/gfx.hpp>
#include <aurora/render.hpp>
#include <aurora/trace.hpp>
#include <dolphin/gx.h>
#include <SDL3/SDL_stdinc.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

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
};

std::atomic_uint32_t g_errorCount = 0;
std::atomic_uint32_t g_anchorWarnings = 0;

void log_callback(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  if (render_demo::is_unknown_anchor_warning(level, module != nullptr ? module : "", {message, len})) {
    g_anchorWarnings.fetch_add(1, std::memory_order_relaxed);
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

struct HintCounterSnapshot {
  std::array<render_demo::ObservedCounters, render_demo::LastHintId + 1 - render_demo::FirstHintId> ids{};
  uint64_t suppressedDraws = 0;
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
  snapshot.suppressedDraws = aurora::render::draw_route_counters().suppressed;
  return snapshot;
}

render_demo::ObservedCounters snapshot_counters(u16 id, void* userData) {
  return static_cast<const HintCounterSnapshot*>(userData)->ids[id - render_demo::FirstHintId];
}

void print_hint_summary(const render_demo::HintTally& tally, const HintCounterSnapshot& snapshot) {
  render_demo::ObservedCounters total;
  for (const auto& counters : snapshot.ids) {
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
  // The receiver is installed once at initialisation, so the answer is fixed for the whole run
  const bool decoding = aurora::render::hints_enabled();
  const render_demo::SceneOptions sceneOptions{.hints = options.hints, .captureOnlyBox = options.hints && decoding};
  bool decodingChanged = false;

  GXInit(nullptr, 0);
  AuroraSetViewportPolicy(AURORA_VIEWPORT_FIT);
  if (!render_demo::init_scene()) {
    std::fprintf(stderr, "render_demo: scene initialisation failed\n");
    aurora_shutdown();
    return ExitSceneInit;
  }
  if (std::string graphError; !render_demo::register_graph_passes(
          {.vignette = options.vignette, .vignetteAnchor = options.vignetteAnchor, .depthCheck = options.depthCheck},
          graphError)) {
    std::fprintf(stderr, "render_demo: render graph setup failed: %s\n", graphError.c_str());
    render_demo::shutdown_scene();
    aurora_shutdown();
    return ExitRenderGraphCheck;
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
    // Params set before aurora_begin_frame apply to the frame it begins
    render_demo::update_graph_passes(frame);
    if (exiting || paused || !aurora_begin_frame()) {
      continue;
    }
    decodingChanged |= aurora::render::hints_enabled() != decoding;
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
  HintCounterSnapshot hintCounters = read_hint_counters();
  render_demo::unregister_graph_passes();
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
  if (hintFailure) {
    return ExitHintCheck;
  }
  if (!options.tracePath.empty() && traceStatus.state != aurora::trace::CaptureState::Finished) {
    std::fprintf(stderr, "render_demo: trace capture did not finish: %s\n",
                 traceStatus.error.empty() ? "the capture window was never reached" : traceStatus.error.c_str());
    return ExitTraceFailed;
  }
  if (options.failOnError && errors != 0) {
    return ExitLoggedError;
  }
  return ExitSuccess;
}

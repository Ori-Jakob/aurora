#include "options.hpp"
#include "scene.hpp"

#include <aurora/aurora.h>
#include <aurora/event.h>
#include <aurora/trace.hpp>
#include <dolphin/gx.h>
#include <SDL3/SDL_stdinc.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
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
};

std::atomic_uint32_t g_errorCount = 0;

void log_callback(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
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

  GXInit(nullptr, 0);
  AuroraSetViewportPolicy(AURORA_VIEWPORT_FIT);
  if (!render_demo::init_scene()) {
    std::fprintf(stderr, "render_demo: scene initialisation failed\n");
    aurora_shutdown();
    return ExitSceneInit;
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
    if (exiting || paused || !aurora_begin_frame()) {
      continue;
    }
    render_demo::draw_scene(frame);
    aurora_end_frame();
    ++frame;
    if (options.frames != 0 && frame >= options.frames) {
      break;
    }
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
  if (timedOut) {
    return ExitTimeout;
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

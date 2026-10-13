#include "capture.hpp"

#include <aurora/trace.hpp>

#include "capture_session.hpp"
#include "state_preamble.hpp"

#include "../dolphin/vi/vi_internal.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/texture_replacement.hpp"
#include "../gx/fifo.hpp"
#include "../gx/guest_memory.hpp"
#include "../gx/gx.hpp"
#include "../internal.hpp"
#include "../io.hpp"
#include "../webgpu/gpu.hpp"

#include <fmt/format.h>

#include <atomic>
#include <mutex>
#include <string_view>

namespace aurora::trace {
namespace {
constexpr Module Log{"aurora::trace::capture"};

enum class Phase : uint8_t { Idle, Pending, Capturing };

std::atomic<Phase> g_phase{Phase::Idle};
std::mutex g_mutex;
CaptureSession g_session;
CaptureOptions g_options;
CaptureStatus g_status;
bool g_stopRequested = false;
std::string g_abortReason;
uint32_t g_endFlags = 0;
Vec2<uint32_t> g_frameViSize;

void observe_guest_read(void*, gx::guest_memory::Kind kind, uint64_t address, std::span<const uint8_t> bytes) noexcept {
  std::lock_guard lock{g_mutex};
  g_session.observe(static_cast<uint8_t>(kind), address, bytes);
}

void observe_processed(void*, const uint8_t* data, uint32_t size) noexcept {
  std::lock_guard lock{g_mutex};
  g_session.processed({data, size});
}

constexpr gx::guest_memory::Hooks RecordHooks{.observe = observe_guest_read};
constexpr gx::fifo::ProcessObserver RecordObserver{.processed = observe_processed};

uint32_t device_features() noexcept {
  uint32_t features = 0;
  if (webgpu::g_hasCoreFeatures) {
    features |= DeviceCoreFeatures;
  }
  if (webgpu::g_dualSourceBlendingSupported) {
    features |= DeviceDualSourceBlending;
  }
  if (webgpu::g_bcTexturesSupported) {
    features |= DeviceBcTextures;
  }
  if (webgpu::g_astcTexturesSupported) {
    features |= DeviceAstcTextures;
  }
  if (webgpu::g_textureComponentSwizzleSupported) {
    features |= DeviceTextureSwizzle;
  }
  return features;
}

TraceInfo make_info() {
  const auto& graphics = webgpu::g_graphicsConfig;
  TraceInfo info{
      .captureFlags = CaptureReplacementsBypassed | CaptureBlockingPipelines | CaptureSynchronousPublish,
      .backend = g_config.desiredBackend,
      .msaaSamples = graphics.msaaSamples,
      .maxAnisotropy = graphics.textureAnisotropy,
      .efbColorFormat = static_cast<uint32_t>(graphics.surfaceConfiguration.format),
      .depthFormat = static_cast<uint32_t>(graphics.depthFormat),
      .deviceFeatures = device_features(),
      .featureProfile = webgpu::g_featureProfile,
      .resampler = webgpu::get_resampler(),
      .requestedFrames = g_options.frameCount,
      .label = g_options.label,
  };
  if (!webgpu::g_adapterInfo.device.IsUndefined()) {
    info.adapter = std::string_view{webgpu::g_adapterInfo.device};
  }
  return info;
}

RenderConfig current_render_config() noexcept {
  const auto viSize = vi::configured_fb_size();
  return {
      .efbWidth = webgpu::g_frameBuffer.size.width,
      .efbHeight = webgpu::g_frameBuffer.size.height,
      .viFbWidth = viSize.x,
      .viEfbHeight = viSize.y,
      .viewportPolicy = static_cast<uint8_t>(gx::g_gxState.viewportPolicy),
      .normalBuffer = webgpu::g_graphicsConfig.normalBuffer,
  };
}

// Runs with g_mutex held after a drain, so no decode can see a half-removed capture
void teardown() noexcept {
  gx::fifo::set_synchronous_publish(false);
  gx::fifo::set_process_observer(nullptr);
  gx::guest_memory::set_hooks(nullptr);
  gfx::texture_replacement::set_bypass(false);
  gfx::set_blocking_pipelines(false);
  gx::clear_static_texture_cache();
  g_stopRequested = false;
  g_abortReason.clear();
  g_endFlags = 0;
  g_phase.store(Phase::Idle, std::memory_order_relaxed);
}

void fail(std::string error) noexcept {
  g_session.discard();
  g_status.state = CaptureState::Failed;
  g_status.error = std::move(error);
  Log.error("Trace capture failed: {}", g_status.error);
}
} // namespace

bool start_capture(const CaptureOptions& options) {
  std::lock_guard lock{g_mutex};
  if (g_phase.load(std::memory_order_relaxed) != Phase::Idle) {
    Log.warn("Ignoring a trace capture request while another capture is active");
    return false;
  }

  std::string error;
  if (options.frameCount == 0) {
    error = "frameCount must be at least 1";
  } else if (options.compressionLevel < 0 || options.compressionLevel > 19) {
    error = fmt::format("compression level {} is outside [0, 19]", options.compressionLevel);
  } else if (options.path.empty()) {
    error = "no trace path was given";
  }
  g_status = {
      .state = CaptureState::Pending,
      .path = options.path,
      .framesRequested = options.frameCount,
  };
  if (error.empty()) {
    g_session.open(options.path, options.compressionLevel, 0, error);
  }
  if (!error.empty()) {
    fail(std::move(error));
    return false;
  }

  g_options = options;
  g_stopRequested = false;
  g_abortReason.clear();
  g_endFlags = 0;
  g_phase.store(Phase::Pending, std::memory_order_relaxed);
  Log.info("Capturing {} frames to {} from the next frame", options.frameCount, io::fs_path_to_string(options.path));
  return true;
}

void stop_capture() {
  std::lock_guard lock{g_mutex};
  switch (g_phase.load(std::memory_order_relaxed)) {
  case Phase::Pending:
    g_session.discard();
    g_status.state = CaptureState::Idle;
    g_phase.store(Phase::Idle, std::memory_order_relaxed);
    break;
  case Phase::Capturing:
    g_stopRequested = true;
    break;
  case Phase::Idle:
    break;
  }
}

CaptureStatus capture_status() {
  std::lock_guard lock{g_mutex};
  return g_status;
}

bool is_capturing() noexcept { return g_phase.load(std::memory_order_relaxed) != Phase::Idle; }

namespace detail {
void on_frame_begin() noexcept {
  if (g_phase.load(std::memory_order_relaxed) == Phase::Idle) {
    return;
  }
  std::lock_guard lock{g_mutex};
  if (g_phase.load(std::memory_order_relaxed) == Phase::Pending) {
    gx::reset_texture_caches();
    gfx::texture_replacement::set_bypass(true);
    std::string error;
    if (!g_session.write_header_chunks(make_info(), build_state_preamble(gx::g_gxState), error)) {
      gfx::texture_replacement::set_bypass(false);
      g_phase.store(Phase::Idle, std::memory_order_relaxed);
      fail(std::move(error));
      return;
    }
    gfx::set_blocking_pipelines(true);
    gx::guest_memory::set_hooks(&RecordHooks);
    gx::fifo::set_process_observer(&RecordObserver);
    gx::fifo::set_synchronous_publish(true);
    g_phase.store(Phase::Capturing, std::memory_order_relaxed);
    g_status.state = CaptureState::Capturing;
  }

  const auto config = current_render_config();
  g_frameViSize = {config.viFbWidth, config.viEfbHeight};
  g_session.begin_frame(gfx::current_frame(), config);
}

void on_frame_end() noexcept {
  if (g_phase.load(std::memory_order_relaxed) != Phase::Capturing) {
    return;
  }
  std::lock_guard lock{g_mutex};
  if (!g_session.in_frame()) {
    return;
  }

  std::string error;
  if (!g_session.end_frame(error)) {
    teardown();
    fail(std::move(error));
    return;
  }
  ++g_status.framesCaptured;
  g_status.bytesWritten = g_session.bytes_written();
  if (const auto viSize = vi::configured_fb_size(); viSize != g_frameViSize) {
    g_endFlags |= EndViChangedMidFrame;
    Log.warn("VI configuration changed from {}x{} to {}x{} during captured frame {}", g_frameViSize.x, g_frameViSize.y,
             viSize.x, viSize.y, g_status.framesCaptured - 1);
  }

  if (!g_abortReason.empty()) {
    auto reason = std::move(g_abortReason);
    teardown();
    fail(std::move(reason));
    return;
  }
  if (!g_stopRequested && g_status.framesCaptured < g_status.framesRequested) {
    return;
  }

  const uint32_t endFlags = g_endFlags;
  const std::string reason =
      g_stopRequested ? fmt::format("stopped after {} of {} frames", g_status.framesCaptured, g_status.framesRequested)
                      : std::string{};
  teardown();
  if (!g_session.finalize(endFlags, reason, error)) {
    fail(std::move(error));
    return;
  }
  g_status.bytesWritten = g_session.bytes_written();
  g_status.state = CaptureState::Finished;
  Log.info("Captured {} frames to {} ({} bytes)", g_status.framesCaptured, io::fs_path_to_string(g_status.path),
           g_status.bytesWritten);
}

void note_untraceable(const char* api) noexcept {
  if (g_phase.load(std::memory_order_relaxed) != Phase::Capturing) {
    return;
  }
  std::lock_guard lock{g_mutex};
  if (g_phase.load(std::memory_order_relaxed) != Phase::Capturing || !g_abortReason.empty()) {
    return;
  }
  g_abortReason = fmt::format("{} records GPU work outside the GX FIFO; trace discarded", api);
  g_status.error = g_abortReason;
  Log.warn("{}", g_abortReason);
}

void shutdown() noexcept {
  const auto phase = g_phase.load(std::memory_order_relaxed);
  if (phase == Phase::Idle) {
    return;
  }
  if (phase == Phase::Capturing) {
    gx::fifo::drain();
  }

  std::lock_guard lock{g_mutex};
  if (g_phase.load(std::memory_order_relaxed) == Phase::Pending) {
    g_session.discard();
    g_status.state = CaptureState::Idle;
    g_phase.store(Phase::Idle, std::memory_order_relaxed);
    return;
  }

  std::string error;
  if (g_session.in_frame()) {
    if (!g_session.end_frame(error)) {
      teardown();
      fail(std::move(error));
      return;
    }
    ++g_status.framesCaptured;
  }
  const uint32_t endFlags = g_endFlags | EndAborted;
  teardown();
  if (!g_session.finalize(endFlags, "aurora shutdown", error)) {
    fail(std::move(error));
    return;
  }
  g_status.bytesWritten = g_session.bytes_written();
  g_status.state = CaptureState::Failed;
  g_status.error = fmt::format("aurora shut down after {} of {} frames; the partial trace was kept",
                               g_status.framesCaptured, g_status.framesRequested);
  Log.warn("{}", g_status.error);
}
} // namespace detail
} // namespace aurora::trace

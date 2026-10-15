#include "replayer.hpp"

#include "dolphin/vi/vi_internal.hpp"
#include "gfx/frame.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/png_io.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/guest_memory.hpp"
#include "gx/gx.hpp"
#include "gx/texture.hpp"
#include "io.hpp"
#if AURORA_ENABLE_RENDER
#include "render/jobs.hpp"
#endif
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include "window.hpp"

#include <aurora/gfx.h>
#include <aurora/texture.hpp>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace aurora::gxreplay {
namespace {
constexpr Module Log{"aurora::gxreplay"};
using Clock = std::chrono::steady_clock;
constexpr uint64_t GpuTimingWaitNs = 1'000'000'000;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>{end - start}.count();
}

void log_callback(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  const char* name = "INFO";
  FILE* out = stdout;
  switch (level) {
  case LOG_DEBUG:
    name = "DEBUG";
    break;
  case LOG_INFO:
    break;
  case LOG_WARNING:
    name = "WARNING";
    break;
  case LOG_ERROR:
    name = "ERROR";
    out = stderr;
    break;
  case LOG_FATAL:
    name = "FATAL";
    out = stderr;
    break;
  }
  std::fprintf(out, "[%s: %s;%.*s]\n", name, module, static_cast<int>(len), message);
  if (level == LOG_FATAL) {
    std::fflush(out);
    std::abort();
  }
}

uint32_t device_features() noexcept {
  uint32_t features = 0;
  if (webgpu::g_hasCoreFeatures) {
    features |= trace::DeviceCoreFeatures;
  }
  if (webgpu::g_dualSourceBlendingSupported) {
    features |= trace::DeviceDualSourceBlending;
  }
  if (webgpu::g_bcTexturesSupported) {
    features |= trace::DeviceBcTextures;
  }
  if (webgpu::g_astcTexturesSupported) {
    features |= trace::DeviceAstcTextures;
  }
  if (webgpu::g_textureComponentSwizzleSupported) {
    features |= trace::DeviceTextureSwizzle;
  }
  return features;
}

class ScopedStoreHooks {
public:
  explicit ScopedStoreHooks(const gx::guest_memory::Hooks* hooks) noexcept { gx::guest_memory::set_hooks(hooks); }
  ~ScopedStoreHooks() { gx::guest_memory::set_hooks(nullptr); }
  ScopedStoreHooks(const ScopedStoreHooks&) = delete;
  ScopedStoreHooks& operator=(const ScopedStoreHooks&) = delete;
};
} // namespace

ExitCode bring_up_aurora(int argc, char* argv[], const Options& options, const trace::TraceInfo& info, bool synthetic,
                         AuroraInfo& auroraInfo, std::string& error) {
  window::set_start_hidden(!options.showWindow);
  const auto dataDir = options.dataDir.empty() ? options.outDir / ".aurora" : options.dataDir;
  if (!io::create_directories(dataDir)) {
    error = fmt::format("failed to create {}: {}", io::fs_path_to_string(dataDir), SDL_GetError());
    return ExitUsage;
  }
  const std::string dataPath = io::fs_path_to_string(dataDir);

  AuroraConfig config{};
  config.appName = "aurora_gxreplay";
  config.userPath = dataPath.c_str();
  config.cachePath = dataPath.c_str();
  config.desiredBackend = options.backend;
  config.featureProfile = options.profile != FEATURE_PROFILE_AUTO ? options.profile : info.featureProfile;
  config.msaa = info.msaaSamples;
  config.maxTextureAnisotropy = static_cast<uint16_t>(info.maxAnisotropy);
  config.vsync = false;
  config.windowPosX = -1;
  config.windowPosY = -1;
  config.windowWidth = 640;
  config.windowHeight = 480;
  config.allowTextureDumps = false;
  config.allowCpuAdapter = options.allowCpu;
  config.mem1Size = 0;
  config.mem2Size = 0;
  config.logCallback = &log_callback;
  config.logLevel = options.verbose ? LOG_DEBUG : LOG_INFO;
  auroraInfo = aurora_initialize(argc, argv, &config);

  if (options.backend != BACKEND_AUTO && auroraInfo.backend != options.backend) {
    error =
        fmt::format("requested backend {} but got {}", backend_name(options.backend), backend_name(auroraInfo.backend));
    aurora_shutdown();
    return ExitDeviceMismatch;
  }
  if (!synthetic && !options.allowDeviceMismatch) {
    const auto& graphics = webgpu::g_graphicsConfig;
    const uint32_t features = device_features();
    if (graphics.msaaSamples != info.msaaSamples || graphics.textureAnisotropy != info.maxAnisotropy ||
        features != info.deviceFeatures) {
      error = fmt::format(
          "device differs from the capture: MSAA {} (trace {}), anisotropy {} (trace {}), features "
          "0x{:X} (trace 0x{:X}); pass --allow-device-mismatch to replay anyway",
          graphics.msaaSamples, info.msaaSamples, graphics.textureAnisotropy, info.maxAnisotropy, features,
          info.deviceFeatures);
      aurora_shutdown();
      return ExitDeviceMismatch;
    }
  }

  gfx::set_blocking_pipelines(true);
#if AURORA_ENABLE_RENDER
  render::jobs::set_synchronous(true);
#endif
  if (options.gpuTimings) {
    webgpu::gpu_prof::set_always(true);
  }
  if (!options.replacementsDir.empty()) {
    aurora::texture::load_replacement_directory(options.replacementsDir);
    gfx::texture_replacement::set_synchronous_loads(true);
  }
  return ExitOk;
}

Replayer::Replayer(const Options& options, trace::TraceReader& reader, bool keepImages)
: m_options(options), m_reader(reader), m_keepImages(keepImages) {}

bool Replayer::fail(ExitCode code, std::string message, std::string& error) {
  m_exitCode = code;
  error = std::move(message);
  return false;
}

bool Replayer::replay(uint32_t run, ReplayResult& out, std::string& error) {
  m_exitCode = ExitOk;
  gfx::gpu_synchronize();
  gx::guest_memory::set_hooks(nullptr);
  gx::g_gxState = gx::GXState{};
  gx::fifo::clear_draw_cache();
  gx::reset_texture_caches();
  m_store.reset();
  if (!m_reader.rewind(error)) {
    return fail(ExitTrace, error, error);
  }

  const trace::BlobSink sink = [this](const trace::Hash128& hash, std::span<const uint8_t> bytes,
                                      std::string& sinkError) { return m_store.add_blob(hash, bytes, sinkError); };
  trace::FrameView frame;
  auto status = m_reader.next(frame, sink, error);
  if (status == trace::TraceReader::Status::Error) {
    return fail(ExitTrace, error, error);
  }
  if (status == trace::TraceReader::Status::End) {
    return fail(ExitTrace, "the trace contains no frames", error);
  }

  const ScopedStoreHooks hooks{m_store.hooks()};
  std::vector<uint8_t> rgba;
  uint32_t width = 0;
  uint32_t height = 0;
  // Frame 0 begins with the clear color and viewport the preamble restores, so the preamble decodes a frame earlier
  if (!apply_render_config(frame.config, error)) {
    return false;
  }
  if (!gfx::begin_frame()) {
    return fail(ExitUsage, "gfx::begin_frame failed", error);
  }
  gx::fifo::begin_frame();
  const bool preambleOk = process_fifo(m_reader.state_preamble(), error);
  if (!end_frame(false, rgba, width, height, error) || !preambleOk) {
    return false;
  }

  for (uint32_t index = 0; index < m_options.frames.last;) {
    SDL_PumpEvents();
    SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
    const auto frameStart = Clock::now();
    const bool output = m_options.frames.contains(index);
    if (!apply_render_config(frame.config, error)) {
      return false;
    }
    if (!gfx::begin_frame()) {
      return fail(ExitUsage, "gfx::begin_frame failed", error);
    }
    gx::fifo::begin_frame();

    const uint64_t missesBefore = m_store.misses();
    uint64_t fifoBytes = 0;
    uint32_t mapRecords = 0;
    bool recordsOk = true;
    for (const auto& record : frame.records) {
      switch (record.type) {
      case trace::RecordType::MemoryMap: {
        trace::MemoryMapRecord map;
        if (!trace::decode_memory_map(record.payload, map) || !m_store.apply(map, error)) {
          recordsOk = fail(ExitTrace, error.empty() ? "unreadable MEMORY_MAP record" : error, error);
        }
        ++mapRecords;
        break;
      }
      case trace::RecordType::Fifo:
        fifoBytes += record.payload.size();
        recordsOk = process_fifo(record.payload, error);
        break;
      case trace::RecordType::RenderConfig: {
        trace::RenderConfig config;
        if (!trace::decode_render_config(record.payload, config)) {
          recordsOk = fail(ExitTrace, "unreadable RENDER_CONFIG record", error);
          break;
        }
        if (config.efbWidth != frame.config.efbWidth || config.efbHeight != frame.config.efbHeight ||
            config.normalBuffer != frame.config.normalBuffer) {
          Log.warn("Frame {} changes its EFB size or normal buffer mid-frame; only VI and viewport policy apply",
                   index);
        }
        const auto viSize = vi::configured_fb_size();
        if (config.viFbWidth != viSize.x || config.viEfbHeight != viSize.y) {
          GXRenderModeObj renderMode{};
          renderMode.fbWidth = static_cast<u16>(config.viFbWidth);
          renderMode.efbHeight = static_cast<u16>(config.viEfbHeight);
          vi::configure(&renderMode);
        }
        gx::set_viewport_policy(static_cast<AuroraViewportPolicy>(config.viewportPolicy));
        gx::update();
        break;
      }
      }
      if (!recordsOk) {
        break;
      }
    }
    const auto processEnd = Clock::now();

    const bool frameOk = end_frame(output && recordsOk, rgba, width, height, error);
    if (!frameOk || !recordsOk) {
      return false;
    }
    const uint64_t frameMisses = m_store.misses() - missesBefore;
    out.misses += frameMisses;
    if (output) {
      const auto frameEnd = Clock::now();
      out.frames.push_back({
          .frame = index,
          .sourceFrame = frame.sourceFrame,
          .width = width,
          .height = height,
          .rgbaHash = hash_rgba(rgba),
          .fifoBytes = fifoBytes,
          .mapRecords = mapRecords,
          .misses = frameMisses,
          .draws = aurora_get_stats()->drawCallCount,
      });
      out.timings.push_back({
          .run = run,
          .frame = index,
          .processMs = elapsed_ms(frameStart, processEnd),
          .finishMs = m_finishMs,
          .gpuWaitMs = m_gpuWaitMs,
          .totalMs = elapsed_ms(frameStart, frameEnd),
      });
      if (m_options.gpuTimings) {
        collect_gpu_timing(run, index, out);
      }
      if (run == 0 && m_options.writePng) {
        const auto path = m_options.outDir / fmt::format("frame_{:05}.png", index);
        if (!gfx::png::write_rgba8_png(path, width, height, {rgba.data(), rgba.size()})) {
          return fail(ExitUsage, fmt::format("failed to write {}", io::fs_path_to_string(path)), error);
        }
      }
      if (m_keepImages) {
        out.images.push_back(std::move(rgba));
        rgba = {};
      }
    }

    if (++index >= m_options.frames.last) {
      break;
    }
    status = m_reader.next(frame, sink, error);
    if (status == trace::TraceReader::Status::Error) {
      return fail(ExitTrace, error, error);
    }
    if (status == trace::TraceReader::Status::End) {
      break;
    }
  }
  return true;
}

// Only output frames come here, after their CPU timings; their readback already waited for the GPU, so nothing stalls
void Replayer::collect_gpu_timing(uint32_t run, uint32_t frame, ReplayResult& out) {
  webgpu::gpu_prof::wait_idle(GpuTimingWaitNs);
  if (!webgpu::gpu_prof::latest(m_gpuTimes) || m_gpuTimes.sequence != webgpu::gpu_prof::last_frame_sequence()) {
    return;
  }
  GpuFrameTiming gpu{.run = run, .frame = frame, .frameMs = m_gpuTimes.frameMs, .dropped = m_gpuTimes.dropped};
  for (const auto& pass : m_gpuTimes.passes) {
    gpu.passes.push_back({.name = pass.name != nullptr ? pass.name : "", .ms = pass.ms});
  }
  out.gpuTimings.push_back(std::move(gpu));
}

bool Replayer::apply_render_config(const trace::RenderConfig& config, std::string& error) {
  const auto viSize = vi::configured_fb_size();
  if (config.viFbWidth != viSize.x || config.viEfbHeight != viSize.y) {
    GXRenderModeObj renderMode{};
    renderMode.fbWidth = static_cast<u16>(config.viFbWidth);
    renderMode.efbHeight = static_cast<u16>(config.viEfbHeight);
    vi::configure(&renderMode);
  }
  gx::set_viewport_policy(static_cast<AuroraViewportPolicy>(config.viewportPolicy));
  gx::update();

  const auto& efb = webgpu::g_frameBuffer.size;
  if (config.efbWidth != 0 && config.efbHeight != 0 &&
      (config.efbWidth != efb.width || config.efbHeight != efb.height)) {
    window::set_frame_buffer_size_override(config.efbWidth, config.efbHeight);
    const auto size = window::get_window_size();
    webgpu::resize_swapchain(config.efbWidth, config.efbHeight, size.native_fb_width, size.native_fb_height, false);
  }
  if (config.normalBuffer && !webgpu::g_graphicsConfig.normalBuffer && !webgpu::enable_normal_buffer()) {
    return fail(ExitDeviceMismatch, "the trace uses a normal buffer this device cannot enable", error);
  }
  return true;
}

bool Replayer::process_fifo(std::span<const uint8_t> bytes, std::string& error) {
  m_store.begin_record();
  size_t offset = 0;
  while (offset < bytes.size()) {
    const auto result = gx::fifo::process(bytes.data() + offset, static_cast<uint32_t>(bytes.size() - offset));
    if (result.bytesProcessed == 0) {
      return fail(ExitTrace, fmt::format("FIFO record stalled at byte {} of {}", offset, bytes.size()), error);
    }
    offset += result.bytesProcessed;
  }
  return true;
}

bool Replayer::end_frame(bool readback, std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height,
                         std::string& error) {
  const auto start = Clock::now();
  gx::fifo::drain();
  gx::fifo::end_frame();
  gx::texture::end_frame();
  gfx::finish();
  gfx::end_frame([this, readback](wgpu::CommandEncoder& encoder, std::vector<gfx::AfterSubmitCallback> callbacks) {
    webgpu::gpu_prof::frame_end(encoder);
    if (readback) {
      m_readback.encode(encoder);
    }
    const wgpu::CommandBufferDescriptor descriptor{.label = "gxreplay frame"};
    const auto buffer = encoder.Finish(&descriptor);
    gfx::before_submit();
    webgpu::g_queue.Submit(1, &buffer);
    webgpu::gpu_prof::after_submit();
    gfx::after_frame();
    for (auto& callback : callbacks) {
      if (callback) {
        callback();
      }
    }
    gfx::after_submit();
  });
  const auto submitted = Clock::now();
  gfx::gpu_synchronize();
  bool ok = true;
  if (readback) {
    ok = m_readback.map(rgba, width, height, error);
  }
  const auto done = Clock::now();
  m_finishMs = elapsed_ms(start, submitted);
  m_gpuWaitMs = elapsed_ms(submitted, done);
  if (!ok) {
    return fail(ExitUsage, error, error);
  }
  return true;
}
} // namespace aurora::gxreplay

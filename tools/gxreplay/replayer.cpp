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
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "gx/texture.hpp"
#include "io.hpp"
#if AURORA_ENABLE_RENDER
#include "render/debug/views.hpp"
#include "render/jobs.hpp"
#include "render/scene/classify.hpp"
#endif
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include "window.hpp"

#include <aurora/gfx.h>
#include <aurora/render_caps.hpp>
#include <aurora/texture.hpp>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace aurora::gxreplay {
namespace {
constexpr Module Log{"aurora::gxreplay"};
using Clock = std::chrono::steady_clock;
constexpr uint64_t GpuTimingWaitNs = 1'000'000'000;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>{end - start}.count();
}

CpuFrameZones frame_zones(uint32_t run, uint32_t frame, const gx::fifo::CpuZoneStats& before,
                          const gx::fifo::CpuZoneStats& after, const gx::fifo::ImmutableArrayStats& residentBefore,
                          const gx::fifo::ImmutableArrayStats& residentAfter, const gx::BindCounts& bindsBefore,
                          const gx::BindCounts& bindsAfter, uint64_t resolves, uint64_t uniqueArrayBytes,
                          const AuroraStats& stats) {
  const auto ms = [&](gx::fifo::CpuZone zone) { return static_cast<double>(after.ns[zone] - before.ns[zone]) / 1e6; };
  const auto calls = [&](gx::fifo::CpuZone zone) { return after.calls[zone] - before.calls[zone]; };
  return {
      .run = run,
      .frame = frame,
      .fifoMs = ms(gx::fifo::CpuZoneProcess),
      .pushDrawMs = ms(gx::fifo::CpuZonePushDraw),
      .arraysMs = ms(gx::fifo::CpuZoneArrays),
      .pipelineBuildMs = ms(gx::fifo::CpuZonePipelineBuild),
      .bindGroupBuildMs = ms(gx::fifo::CpuZoneBindGroupBuild),
      .uniformBuildMs = ms(gx::fifo::CpuZoneUniformBuild),
      .fogLutMs = ms(gx::fifo::CpuZoneFogLut),
      .fifoCalls = calls(gx::fifo::CpuZoneProcess),
      .pushDraws = calls(gx::fifo::CpuZonePushDraw),
      .arrayMisses = calls(gx::fifo::CpuZoneArrays),
      .arrayBytes = after.arrayBytes - before.arrayBytes,
      .pipelineBuilds = calls(gx::fifo::CpuZonePipelineBuild),
      .bindGroupBuilds = calls(gx::fifo::CpuZoneBindGroupBuild),
      .uniformBuilds = calls(gx::fifo::CpuZoneUniformBuild),
      .fogLutPushes = calls(gx::fifo::CpuZoneFogLut),
      .fogLutBytes = after.fogLutBytes - before.fogLutBytes,
      .resolves = resolves,
      .uniqueArrayBytes = uniqueArrayBytes,
      .vertBytes = stats.lastVertSize,
      .indexBytes = stats.lastIndexSize,
      .uniformBytes = stats.lastUniformSize,
      .storageBytes = stats.lastStorageSize,
      .residentHits = residentAfter.hits - residentBefore.hits,
      .residentStages = residentAfter.stages - residentBefore.stages,
      .encodedDraws = bindsAfter.draws - bindsBefore.draws,
      .texturedDraws = bindsAfter.texturedDraws - bindsBefore.texturedDraws,
      .uniformBinds = bindsAfter.uniformBinds - bindsBefore.uniformBinds,
      .textureBinds = bindsAfter.textureBinds - bindsBefore.textureBinds,
      .indexBinds = bindsAfter.indexBinds - bindsBefore.indexBinds,
      .encodeMs = static_cast<double>(bindsAfter.encodeNs - bindsBefore.encodeNs) / 1e6,
  };
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
  if (render::caps().subgroups) {
    features |= trace::DeviceSubgroups;
  }
  if (render::caps().shaderF16) {
    features |= trace::DeviceShaderF16;
  }
  return features;
}

uint64_t malformed_hints() noexcept {
  uint64_t total = 0;
  for (uint16_t id = GX_AURORA_RENDER_CAMERA; id <= GX_AURORA_RENDER_WIND_SOURCES; ++id) {
    total += gx::hints::counters(id).malformed;
  }
  for (uint16_t id = GX_AURORA_GPU_UPLOAD; id <= GX_AURORA_GPU_DESTROY_SYSTEM; ++id) {
    total += gx::hints::counters(id).malformed;
  }
  return total;
}

render::gpu::Stats gpu_delta(const render::gpu::Stats& before, const render::gpu::Stats& after) noexcept {
  return {
      .refused = after.refused - before.refused,
      .refusedBytes = after.refusedBytes - before.refusedBytes,
      .staleBinds = after.staleBinds - before.staleBinds,
      .uploads = after.uploads - before.uploads,
      .uploadBytes = after.uploadBytes - before.uploadBytes,
      .reserves = after.reserves - before.reserves,
      .reserveBytes = after.reserveBytes - before.reserveBytes,
      .failedHandles = after.failedHandles - before.failedHandles,
      .storageMisses = after.storageMisses - before.storageMisses,
      .rejected = after.rejected - before.rejected,
      .binds = after.binds - before.binds,
      .invalidBinds = after.invalidBinds - before.invalidBinds,
      .suppressedDraws = after.suppressedDraws - before.suppressedDraws,
      .created = after.created - before.created,
      .destroyed = after.destroyed - before.destroyed,
      .updates = after.updates - before.updates,
      .updateBytes = after.updateBytes - before.updateBytes,
      .liveSystems = after.liveSystems,
  };
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
#if !AURORA_ENABLE_RENDER
  if (options.classifies()) {
    error = "--debug-view and --classify-* need the render module";
    return ExitUsage;
  }
#endif
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
    const uint32_t compared = trace::compared_device_features(info.captureFlags);
    const uint32_t features = device_features() & compared;
    if (graphics.msaaSamples != info.msaaSamples || graphics.textureAnisotropy != info.maxAnisotropy ||
        features != (info.deviceFeatures & compared)) {
      error = fmt::format(
          "device differs from the capture: MSAA {} (trace {}), anisotropy {} (trace {}), features "
          "0x{:X} (trace 0x{:X}); pass --allow-device-mismatch to replay anyway",
          graphics.msaaSamples, info.msaaSamples, graphics.textureAnisotropy, info.maxAnisotropy, features,
          info.deviceFeatures & compared);
      aurora_shutdown();
      return ExitDeviceMismatch;
    }
  }

  gfx::set_blocking_pipelines(true);
#if AURORA_ENABLE_RENDER
  render::jobs::set_synchronous(true);
  // The request outlives the run-start reset, so every repeat classifies the same way
  render::scene::request_report({.records = options.classifyDraws, .objects = options.classifyObjects});
  if (options.debugView != 0) {
    render::apply_settings(render::Settings{.debugView = options.debugView});
    if (options.debugViewBit != 0) {
      render::debug::set_view_bit(options.debugViewBit);
    }
  }
#endif
  if (options.gpuTimings) {
    webgpu::gpu_prof::set_always(true);
  }
  if (options.cpuZones) {
    gx::fifo::set_cpu_zone_timing(gx::fifo::CpuZoneTimingFifo |
                                  (options.cpuDrawZones ? gx::fifo::CpuZoneTimingDraws : 0u));
  }
  if (!options.replacementsDir.empty()) {
    aurora::texture::load_replacement_directory(options.replacementsDir);
    gfx::texture_replacement::set_synchronous_loads(true);
  }
  return ExitOk;
}

Replayer::Replayer(const Options& options, trace::TraceReader& reader, bool keepImages, bool keepRecords)
: m_options(options), m_reader(reader), m_keepImages(keepImages), m_keepRecords(keepRecords) {
  m_countingHooks = {.resolve = &Replayer::resolve_counting, .userData = this};
}

std::span<const uint8_t> Replayer::resolve_counting(void* userData, gx::guest_memory::Kind kind, uint64_t address,
                                                    size_t size) noexcept {
  auto& self = *static_cast<Replayer*>(userData);
  if (kind == gx::guest_memory::Kind::VertexArray && size != 0 && self.m_frameArrays.emplace(address, size).second) {
    self.m_uniqueArrayBytes += size;
  }
  const auto* hooks = self.m_store.hooks();
  return hooks->resolve(hooks->userData, kind, address, size);
}

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
  gx::hints::reset_state();
  gx::reset_texture_caches();
  m_store.reset();
  const auto residentStart = gx::fifo::immutable_array_stats();
  const auto gpuStart = render::gpu::stats();
  if (!m_reader.rewind(error)) {
    return fail(ExitTrace, error, error);
  }
  const uint64_t malformedBefore = malformed_hints();
  // One line per draw is written as each frame ends, so long traces never hold their records
  std::ofstream drawLines;
  const auto drawLinesPath = m_options.outDir / "classify_draws.jsonl";
  if (run == 0 && m_options.classifyDraws) {
    drawLines.open(drawLinesPath, std::ios::binary | std::ios::trunc);
    if (!drawLines) {
      return fail(ExitUsage, fmt::format("failed to write {}", io::fs_path_to_string(drawLinesPath)), error);
    }
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

  // Unique bytes cost a set insert per array read, so only the diagnostic mode counts them
  const ScopedStoreHooks hooks{m_options.cpuDrawZones ? &m_countingHooks : m_store.hooks()};
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
    const auto bindsBefore = gx::bind_counts();
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
    const auto zonesBefore = gx::fifo::cpu_zone_stats();
    const auto residentBefore = gx::fifo::immutable_array_stats();
    const uint64_t resolvesBefore = m_store.resolves();
    m_frameArrays.clear();
    m_uniqueArrayBytes = 0;
    const uint64_t hintBytesBefore = gx::hints::total_bytes();
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
    const uint64_t hintBytes = gx::hints::total_bytes() - hintBytesBefore;
    const auto zonesAfter = gx::fifo::cpu_zone_stats();
    const auto residentAfter = gx::fifo::immutable_array_stats();
    const uint64_t resolvesAfter = m_store.resolves();

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
          .hintBytes = hintBytes,
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
      if (m_options.cpuZones) {
        out.cpuZones.push_back(frame_zones(run, index, zonesBefore, zonesAfter, residentBefore, residentAfter,
                                           bindsBefore, gx::bind_counts(), resolvesAfter - resolvesBefore,
                                           m_uniqueArrayBytes, *aurora_get_stats()));
      }
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
#if AURORA_ENABLE_RENDER
      if (run == 0 && m_options.classifies()) {
        // The next latch would seal this frame too; sealing now keeps the report next to its frame
        render::scene::seal_frame();
        if (const auto report = render::scene::latest_report()) {
          if (drawLines.is_open()) {
            for (const auto& draw : report->draws) {
              drawLines << format_classify_draw_line(index, draw);
            }
            if (!drawLines) {
              return fail(ExitUsage, fmt::format("failed to write {}", io::fs_path_to_string(drawLinesPath)), error);
            }
          }
          out.classify.push_back({
              .frame = index,
              .report =
                  {
                      .stats = report->stats,
                      .mainCamera = report->mainCamera,
                      .cutEpoch = report->cutEpoch,
                      .sceneEpochKnown = report->sceneEpochKnown,
                      .sceneEpoch = report->sceneEpoch,
                      .draws = m_keepRecords ? report->draws : std::vector<render::scene::ClassifiedDraw>{},
                      .objects = report->objects,
                  },
              .uniformBytes = gfx::last_recorded_uniform_size(),
          });
        }
      }
#endif
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
  if (drawLines.is_open()) {
    drawLines.close();
    if (!drawLines) {
      return fail(ExitUsage, fmt::format("failed to write {}", io::fs_path_to_string(drawLinesPath)), error);
    }
  }
  out.malformedHints = malformed_hints() - malformedBefore;
  const auto residentEnd = gx::fifo::immutable_array_stats();
  out.residentRuns.push_back({
      .run = run,
      .stages = residentEnd.stages - residentStart.stages,
      .hits = residentEnd.hits - residentStart.hits,
      .resets = residentEnd.resets - residentStart.resets,
      .epochs = residentEnd.epochs - residentStart.epochs,
      .mismatches = residentEnd.sizeMismatches - residentStart.sizeMismatches,
      .violations = residentEnd.violations - residentStart.violations,
      .arenaFull = residentEnd.arenaFull - residentStart.arenaFull,
  });
  out.gpuRuns.push_back({.run = run, .stats = gpu_delta(gpuStart, render::gpu::stats())});
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
  if (config.streamingStorageMiB != 0) {
    // Live accepted its GPU-channel commands against this size, so replay must be at least as large
    gfx::detail::grow_streaming_storage_now(uint64_t{config.streamingStorageMiB} * 1024 * 1024);
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

#include "options.hpp"
#include "replayer.hpp"
#include "report.hpp"
#include "self_test.hpp"

#include "gx/fifo.hpp"
#include "io.hpp"
#include "trace/trace_reader.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/aurora.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_stdinc.h>
#include <fmt/format.h>

#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// Last because it redefines main
#include <aurora/main.h>

namespace aurora::gxreplay {
namespace {
void print_error(std::string_view message) {
  std::fprintf(stderr, "aurora_gxreplay: %.*s\n", static_cast<int>(message.size()), message.data());
}

std::string adapter_name() {
  if (webgpu::g_adapterInfo.device.IsUndefined()) {
    return {};
  }
  return std::string{std::string_view{webgpu::g_adapterInfo.device}};
}

bool write_text(const std::filesystem::path& path, const std::string& text) {
  if (!io::write_file_atomic(path, {reinterpret_cast<const uint8_t*>(text.data()), text.size()})) {
    print_error(fmt::format("failed to write {}: {}", io::fs_path_to_string(path), SDL_GetError()));
    return false;
  }
  return true;
}

bool write_reports(const Options& options, const ReportContext& context, std::span<const FrameResult> frames,
                   std::span<const FrameTiming> timings) {
  return write_text(options.outDir / "frames.json", format_frames_json(context, frames)) &&
         write_text(options.outDir / "timing.json", format_timing_json(timings));
}

std::string flag_names(uint32_t flags) {
  std::string names;
  const auto add = [&](uint32_t bit, std::string_view name) {
    if ((flags & bit) != 0) {
      names += names.empty() ? "" : " ";
      names += name;
    }
  };
  add(trace::FileFinalized, "finalized");
  add(trace::FileAborted, "aborted");
  add(trace::FileSynthetic, "synthetic");
  return names.empty() ? "none" : names;
}

int print_info(trace::TraceReader& reader, const std::filesystem::path& path) {
  const auto& header = reader.header();
  const auto& info = reader.info();
  fmt::print(
      "trace: {}\n"
      "version: {}.{}, flags: {} ({}), frames: {}, blobs: {}, chunks: {}\n",
      io::fs_path_to_string(path), header.versionMajor, header.versionMinor, header.flags, flag_names(header.flags),
      header.frameCount, header.blobCount, header.chunkCount);
  fmt::print(
      "info: label \"{}\", adapter \"{}\", backend {}, capture flags 0x{:X}, MSAA {}, "
      "anisotropy {}, EFB format {}, depth format {}, features 0x{:X}, profile {}, "
      "resampler {}, requested frames {}\n",
      info.label, info.adapter, backend_name(info.backend), info.captureFlags, info.msaaSamples, info.maxAnisotropy,
      info.efbColorFormat, info.depthFormat, info.deviceFeatures, static_cast<int>(info.featureProfile),
      static_cast<int>(info.resampler), info.requestedFrames);
  std::printf("state preamble: %zu bytes\n", reader.state_preamble().size());

  uint32_t blobs = 0;
  uint64_t blobBytes = 0;
  const trace::BlobSink sink = [&](const trace::Hash128&, std::span<const uint8_t> bytes, std::string&) {
    ++blobs;
    blobBytes += bytes.size();
    return true;
  };
  std::string error;
  trace::FrameView frame;
  trace::TraceReader::Status status;
  while ((status = reader.next(frame, sink, error)) == trace::TraceReader::Status::Frame) {
    uint32_t fifoRecords = 0;
    uint32_t mapRecords = 0;
    uint64_t fifoBytes = 0;
    for (const auto& record : frame.records) {
      if (record.type == trace::RecordType::Fifo) {
        ++fifoRecords;
        fifoBytes += record.payload.size();
      } else if (record.type == trace::RecordType::MemoryMap) {
        ++mapRecords;
      }
    }
    const auto& config = frame.config;
    fmt::print(
        "frame {} (source {}): EFB {}x{}, VI {}x{}, policy {}, normal buffer {}, "
        "{} FIFO records ({} bytes), {} maps, {} new blobs ({} bytes)\n",
        frame.frameIndex, frame.sourceFrame, config.efbWidth, config.efbHeight, config.viFbWidth, config.viEfbHeight,
        config.viewportPolicy, config.normalBuffer, fifoRecords, fifoBytes, mapRecords, blobs, blobBytes);
    blobs = 0;
    blobBytes = 0;
  }
  if (status == trace::TraceReader::Status::Error) {
    print_error(error);
    return ExitTrace;
  }
  if (const auto& trailer = reader.trailer()) {
    fmt::print(
        "trailer: frames {}, blobs {} ({} bytes), FIFO bytes {}, maps {}, carried records "
        "{}, end flags 0x{:X}, reason \"{}\"\n",
        trailer->frameCount, trailer->blobCount, trailer->blobBytes, trailer->fifoBytes, trailer->mapRecords,
        trailer->carriedRecords, trailer->endFlags, trailer->reason);
  } else {
    std::printf("trailer: missing (incomplete trace)\n");
  }
  return ExitOk;
}

int run_write_self_test(const Options& options) {
  std::string error;
  gx::fifo::init();
  const bool ok = write_self_test_trace(options.writeSelfTest, true, error);
  gx::fifo::shutdown();
  if (!ok) {
    print_error(error);
    return ExitUsage;
  }
  std::printf("aurora_gxreplay: wrote %s\n", io::fs_path_to_string(options.writeSelfTest).c_str());
  return ExitOk;
}

bool same_frames(std::span<const FrameResult> lhs, std::span<const FrameResult> rhs, std::string& mismatch) {
  if (lhs.size() != rhs.size()) {
    mismatch = fmt::format("{} frames instead of {}", rhs.size(), lhs.size());
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].rgbaHash != rhs[i].rgbaHash || lhs[i].width != rhs[i].width || lhs[i].height != rhs[i].height) {
      mismatch =
          fmt::format("frame {} hashes {} instead of {}", lhs[i].frame, hex64(rhs[i].rgbaHash), hex64(lhs[i].rgbaHash));
      return false;
    }
  }
  return true;
}

int run_self_test(int argc, char* argv[], const Options& options) {
  std::string error;
  if (!io::create_directories(options.outDir)) {
    print_error(fmt::format("failed to create {}: {}", io::fs_path_to_string(options.outDir), SDL_GetError()));
    return ExitUsage;
  }
  AuroraInfo auroraInfo{};
  if (const auto code = bring_up_aurora(argc, argv, options, self_test_info(), true, auroraInfo, error);
      code != ExitOk) {
    print_error(error);
    return code;
  }

  const auto hintsPath = options.outDir / "self_test_hints.gxtrace";
  const auto plainPath = options.outDir / "self_test_plain.gxtrace";
  trace::TraceReader hintsReader;
  trace::TraceReader plainReader;
  if (!write_self_test_trace(hintsPath, true, error) || !write_self_test_trace(plainPath, false, error) ||
      !hintsReader.open(hintsPath, error) || !plainReader.open(plainPath, error)) {
    print_error(error);
    aurora_shutdown();
    return ExitUsage;
  }

  Options hintsOptions = options;
  hintsOptions.frames = {};
  Options plainOptions = hintsOptions;
  plainOptions.writePng = false;
  Replayer hints{hintsOptions, hintsReader, true};
  Replayer plain{plainOptions, plainReader, true};
  ReplayResult first;
  ReplayResult second;
  ReplayResult plainResult;
  if (!hints.replay(0, first, error) || !hints.replay(1, second, error)) {
    print_error(error);
    aurora_shutdown();
    return hints.exit_code();
  }
  if (!plain.replay(0, plainResult, error)) {
    print_error(error);
    aurora_shutdown();
    return plain.exit_code();
  }

  std::string failure;
  std::string mismatch;
  const bool deterministic = same_frames(first.frames, second.frames, mismatch);
  if (!deterministic) {
    failure = "run 1 differs from run 0: " + mismatch;
  } else if (first.frames.size() != SelfTestFrameCount) {
    failure = fmt::format("replayed {} frames instead of {}", first.frames.size(), SelfTestFrameCount);
  } else if (!same_frames(first.frames, plainResult.frames, mismatch)) {
    failure = "the plain trace differs from the hint trace: " + mismatch;
  } else if (const uint64_t misses = first.misses + second.misses + plainResult.misses; misses != 0) {
    failure = fmt::format("{} guest reads were not resolved", misses);
  } else {
    check_self_test_frames(first.images, plainResult.images, auroraInfo.backend != BACKEND_NULL, failure);
  }

  std::vector<FrameTiming> timings = first.timings;
  timings.insert(timings.end(), second.timings.begin(), second.timings.end());
  const ReportContext context{
      .traceFile = io::fs_path_to_string(hintsPath),
      .traceVersion = fmt::format("{}.{}", hintsReader.header().versionMajor, hintsReader.header().versionMinor),
      .backend = std::string{backend_name(auroraInfo.backend)},
      .adapter = adapter_name(),
      .synthetic = true,
      .runs = 2,
      .deterministic = deterministic,
  };
  const bool reportsWritten = write_reports(options, context, first.frames, timings);
  aurora_shutdown();
  if (!failure.empty()) {
    print_error("self-test failed: " + failure);
    return ExitSelfTestFailed;
  }
  if (!reportsWritten) {
    return ExitUsage;
  }
  std::printf("aurora_gxreplay: self-test passed on %s\n", context.backend.c_str());
  return ExitOk;
}

int run_replay(int argc, char* argv[], const Options& options) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(options.trace, ec)) {
    print_error(fmt::format("{} is not a readable file", io::fs_path_to_string(options.trace)));
    return ExitUsage;
  }
  std::string error;
  trace::TraceReader reader;
  if (!reader.open(options.trace, error)) {
    print_error(error);
    return ExitTrace;
  }
  if (options.info) {
    return print_info(reader, options.trace);
  }
  if (!reader.complete() && !options.allowIncomplete) {
    print_error("the trace was aborted or never finalized; pass --allow-incomplete to replay its complete frames");
    return ExitTrace;
  }
  if (!io::create_directories(options.outDir)) {
    print_error(fmt::format("failed to create {}: {}", io::fs_path_to_string(options.outDir), SDL_GetError()));
    return ExitUsage;
  }

  AuroraInfo auroraInfo{};
  const bool synthetic = (reader.header().flags & trace::FileSynthetic) != 0;
  if (const auto code = bring_up_aurora(argc, argv, options, reader.info(), synthetic, auroraInfo, error);
      code != ExitOk) {
    print_error(error);
    return code;
  }

  Replayer replayer{options, reader};
  std::vector<ReplayResult> results(options.repeat);
  std::string mismatch;
  bool deterministic = true;
  for (uint32_t run = 0; run < options.repeat; ++run) {
    if (!replayer.replay(run, results[run], error)) {
      print_error(error);
      aurora_shutdown();
      return replayer.exit_code();
    }
    if (run > 0 && deterministic && !same_frames(results[0].frames, results[run].frames, mismatch)) {
      deterministic = false;
      print_error(fmt::format("run {} is not bit-identical to run 0: {}", run, mismatch));
    }
  }

  const auto& frames = results[0].frames;
  if (frames.empty()) {
    print_error(fmt::format("--frames selects no frame of this {}-frame trace", reader.header().frameCount));
  } else if (options.frames.last != UINT32_MAX && frames.back().frame + 1 < options.frames.last) {
    print_error(fmt::format("--frames clamped to the trace's last frame, {}", frames.back().frame));
  }
  std::vector<FrameTiming> timings;
  for (const auto& result : results) {
    timings.insert(timings.end(), result.timings.begin(), result.timings.end());
  }
  const ReportContext context{
      .traceFile = io::fs_path_to_string(options.trace),
      .traceVersion = fmt::format("{}.{}", reader.header().versionMajor, reader.header().versionMinor),
      .backend = std::string{backend_name(auroraInfo.backend)},
      .adapter = adapter_name(),
      .synthetic = synthetic,
      .runs = options.repeat,
      .deterministic = deterministic,
  };
  const bool reportsWritten = write_reports(options, context, frames, timings);
  aurora_shutdown();

  // Frames before --frames are still decoded, so their unresolved reads count too
  const uint64_t misses = results[0].misses;
  std::printf("aurora_gxreplay: %zu frames x %u runs on %s, %llu unresolved reads\n", frames.size(), options.repeat,
              context.backend.c_str(), static_cast<unsigned long long>(misses));
  if (!deterministic) {
    return ExitNotDeterministic;
  }
  if (misses != 0 && !options.allowMisses) {
    print_error("guest reads were not resolved by the trace; pass --allow-misses to accept them");
    return ExitUnresolvedReads;
  }
  return reportsWritten ? ExitOk : ExitUsage;
}
} // namespace
} // namespace aurora::gxreplay

int main(int argc, char* argv[]) {
  using namespace aurora::gxreplay;
  std::string error;
  const auto parsed = parse_options(argc, argv, SDL_getenv("AURORA_BACKEND"), error);
  if (!parsed) {
    std::fprintf(stderr, "aurora_gxreplay: %s\n%.*s", error.c_str(), static_cast<int>(usage().size()), usage().data());
    return ExitUsage;
  }
  const Options& options = *parsed;
  if (options.help) {
    std::printf("%.*s", static_cast<int>(usage().size()), usage().data());
    return ExitOk;
  }
  if (!options.writeSelfTest.empty()) {
    return run_write_self_test(options);
  }
  if (options.selfTest) {
    return run_self_test(argc, argv, options);
  }
  return run_replay(argc, argv, options);
}

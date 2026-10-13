#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace aurora::gxreplay {
struct FrameResult {
  uint32_t frame = 0;
  uint32_t sourceFrame = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t rgbaHash = 0;
  uint64_t fifoBytes = 0;
  uint32_t mapRecords = 0;
  uint64_t misses = 0;
  uint32_t draws = 0;
};

struct FrameTiming {
  uint32_t run = 0;
  uint32_t frame = 0;
  double processMs = 0.0;
  double finishMs = 0.0;
  double gpuWaitMs = 0.0;
  double totalMs = 0.0;
};

struct ReportContext {
  std::string traceFile;
  std::string traceVersion;
  std::string backend;
  std::string adapter;
  bool synthetic = false;
  uint32_t runs = 1;
  bool deterministic = true;
};

std::string format_frames_json(const ReportContext& context, std::span<const FrameResult> frames);
std::string format_timing_json(std::span<const FrameTiming> timings);
std::string json_escape(std::string_view text);
std::string hex64(uint64_t value);
} // namespace aurora::gxreplay

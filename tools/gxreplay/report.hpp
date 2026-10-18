#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::gxreplay {
struct FrameResult {
  uint32_t frame = 0;
  uint32_t sourceFrame = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t rgbaHash = 0;
  uint64_t fifoBytes = 0;
  uint64_t hintBytes = 0;
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

struct GpuPassTiming {
  std::string name;
  double ms = 0.0;
};

struct GpuFrameTiming {
  uint32_t run = 0;
  uint32_t frame = 0;
  double frameMs = 0.0;
  uint32_t dropped = 0;
  std::vector<GpuPassTiming> passes;
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
std::string format_gpu_timing_json(bool supported, std::span<const GpuFrameTiming> timings);
std::string json_escape(std::string_view text);
std::string hex64(uint64_t value);
} // namespace aurora::gxreplay

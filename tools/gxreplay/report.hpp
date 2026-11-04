#pragma once

#include "render/scene/classify.hpp"

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

struct CpuFrameZones {
  uint32_t run = 0;
  uint32_t frame = 0;
  double fifoMs = 0.0;
  double pushDrawMs = 0.0;
  double arraysMs = 0.0;
  double pipelineBuildMs = 0.0;
  double bindGroupBuildMs = 0.0;
  double uniformBuildMs = 0.0;
  double fogLutMs = 0.0;
  uint64_t fifoCalls = 0;
  uint64_t pushDraws = 0;
  uint64_t arrayMisses = 0;
  uint64_t arrayBytes = 0;
  uint64_t pipelineBuilds = 0;
  uint64_t bindGroupBuilds = 0;
  uint64_t uniformBuilds = 0;
  uint64_t fogLutPushes = 0;
  uint64_t fogLutBytes = 0;
  uint64_t resolves = 0;
  uint64_t uniqueArrayBytes = 0;
  uint32_t vertBytes = 0;
  uint32_t indexBytes = 0;
  uint32_t uniformBytes = 0;
  uint32_t storageBytes = 0;
  uint64_t residentHits = 0;
  uint64_t residentStages = 0;
  uint64_t encodedDraws = 0;
  uint64_t texturedDraws = 0;
  uint64_t uniformBinds = 0;
  uint64_t textureBinds = 0;
  uint64_t indexBinds = 0;
  double encodeMs = 0.0;
  double retainedMs = 0.0;
  uint64_t retainedCalls = 0;
  uint64_t routed = 0;
  uint64_t suppressed = 0;
  uint64_t mergeBreaks = 0;
  uint64_t substituted = 0;
  uint64_t deferred = 0;
  uint64_t merged = 0;
};

struct ResidentRun {
  uint32_t run = 0;
  uint64_t stages = 0;
  uint64_t hits = 0;
  uint64_t resets = 0;
  uint64_t epochs = 0;
  uint64_t mismatches = 0;
  uint64_t violations = 0;
  uint64_t arenaFull = 0;
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

struct RetainedFrame {
  uint32_t frame = 0;
  bool active = false;
  uint64_t sceneFrame = 0;
  uint32_t epoch = 0;
  uint32_t entries = 0;
  uint32_t captureOnly = 0;
  uint32_t transient = 0;
  uint32_t late = 0;
  uint32_t instances = 0;
  uint32_t statics = 0;
  uint32_t created = 0;
  uint32_t rekeyed = 0;
  uint32_t newSubId = 0;
  uint32_t newOccurrence = 0;
  uint32_t relatched = 0;
  uint32_t staticRestream = 0;
  uint32_t signatureMisses = 0;
  uint32_t reorigins = 0;
  uint32_t overflow = 0;
  uint32_t droppedEntries = 0;
  uint32_t undeliveredStatics = 0;
  uint32_t lostCaptures = 0;
  uint32_t resyncs = 0;
  uint32_t eventsPending = 0;
};

inline constexpr uint32_t RetainedGateFirstFrame = 30;

struct ClassifyFrame {
  uint32_t frame = 0;
  render::scene::FrameReport report;
  uint32_t uniformBytes = 0;
};

std::string format_frames_json(const ReportContext& context, std::span<const FrameResult> frames);
// The schema writes null for no view or bit, so a run without one never names an empty category
std::string format_classify_json(const ReportContext& context, std::string_view view, std::string_view viewBit,
                                 std::span<const ClassifyFrame> frames, bool objects);
std::string format_classify_draw_line(uint32_t frame, const render::scene::ClassifiedDraw& draw);
std::string format_classify_summary(std::span<const ClassifyFrame> frames, uint64_t malformedHints);
std::string format_retained_line(const RetainedFrame& frame);
std::string format_retained_summary(std::span<const RetainedFrame> frames);
std::string format_timing_json(std::span<const FrameTiming> timings);
std::string format_gpu_timing_json(bool supported, std::span<const GpuFrameTiming> timings);
std::string format_cpu_zones_json(bool drawZones, std::span<const CpuFrameZones> frames);
bool check_cpu_zones(std::span<const CpuFrameZones> frames, size_t framesPerRun, bool drawZones, std::string& error);
std::string json_escape(std::string_view text);
std::string hex64(uint64_t value);
} // namespace aurora::gxreplay

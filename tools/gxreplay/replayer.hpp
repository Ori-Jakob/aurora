#pragma once

#include "options.hpp"
#include "readback.hpp"
#include "report.hpp"

#include "gx/guest_memory.hpp"
#include "trace/snapshot_store.hpp"
#include "trace/trace_reader.hpp"
#include "webgpu/gpu_prof.hpp"

#include <absl/container/flat_hash_set.h>
#include <aurora/aurora.h>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aurora::gxreplay {
enum ExitCode : int {
  ExitOk = 0,
  ExitUsage = 1,
  ExitTrace = 2,
  ExitNotDeterministic = 3,
  ExitDeviceMismatch = 4,
  ExitSelfTestFailed = 5,
  ExitUnresolvedReads = 6,
};

struct ReplayResult {
  std::vector<FrameResult> frames;
  std::vector<FrameTiming> timings;
  std::vector<GpuFrameTiming> gpuTimings;
  std::vector<CpuFrameZones> cpuZones;
  std::vector<ResidentRun> residentRuns;
  uint64_t misses = 0;
  std::vector<std::vector<uint8_t>> images;
  std::vector<ClassifyFrame> classify;
  uint64_t malformedHints = 0;
};

ExitCode bring_up_aurora(int argc, char* argv[], const Options& options, const trace::TraceInfo& info, bool synthetic,
                         AuroraInfo& auroraInfo, std::string& error);

class Replayer {
public:
  Replayer(const Options& options, trace::TraceReader& reader, bool keepImages = false, bool keepRecords = false);
  bool replay(uint32_t run, ReplayResult& out, std::string& error);
  [[nodiscard]] ExitCode exit_code() const noexcept { return m_exitCode; }

private:
  bool apply_render_config(const trace::RenderConfig& config, std::string& error);
  bool process_fifo(std::span<const uint8_t> bytes, std::string& error);
  bool end_frame(bool readback, std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height, std::string& error);
  bool fail(ExitCode code, std::string message, std::string& error);
  void collect_gpu_timing(uint32_t run, uint32_t frame, ReplayResult& out);
  static std::span<const uint8_t> resolve_counting(void* userData, gx::guest_memory::Kind kind, uint64_t address,
                                                   size_t size) noexcept;

  const Options& m_options;
  trace::TraceReader& m_reader;
  trace::SnapshotStore m_store;
  gx::guest_memory::Hooks m_countingHooks;
  absl::flat_hash_set<std::pair<uint64_t, uint64_t>> m_frameArrays;
  uint64_t m_uniqueArrayBytes = 0;
  EfbReadback m_readback;
  bool m_keepImages = false;
  bool m_keepRecords = false;
  ExitCode m_exitCode = ExitOk;
  double m_finishMs = 0.0;
  double m_gpuWaitMs = 0.0;
  webgpu::gpu_prof::FrameTimes m_gpuTimes;
};
} // namespace aurora::gxreplay

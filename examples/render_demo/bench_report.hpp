#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace render_demo {
inline constexpr uint32_t BenchMaxDraws = 2000;
inline constexpr uint32_t BenchMaxPasses = 32;

enum class BenchMode : uint8_t { Direct, Indirect, Zero, Mdi, Bundle };
enum class BenchPipeline : uint8_t { VertexIndex, Plain };
enum class BenchState : uint8_t { Gx, None, Subst };
enum class BenchArgs : uint8_t { Gpu, Cpu };
enum class BenchIndex : uint8_t { U16, U32 };

struct BenchArm {
  BenchMode mode;
  BenchPipeline pipeline;
  BenchState state;
  BenchArgs args;
  BenchIndex index;
  bool operator==(const BenchArm&) const = default;
};

struct BenchPair {
  BenchArm a, b;
  uint32_t draws, passes;
};

std::vector<BenchPair> bench_pairs();
std::string arm_name(const BenchArm& arm);
std::string pair_name(const BenchPair& pair);

struct SampleStats {
  double p50 = 0, p90 = 0, mean = 0, min = 0, max = 0;
  uint32_t count = 0;
};

SampleStats sample_stats(std::vector<double> samples);
bool unstable(const SampleStats& s) noexcept;

struct ArmSamples {
  std::vector<double> encodeUs, finishUs, submitUs, totalUs, gpuTotalUs, gpuInPassUs, gpuArgsUs;
  uint32_t gpuInvalid = 0;
};

struct ArmResult {
  SampleStats encodeUs, finishUs, submitUs, totalUs;
  std::optional<SampleStats> gpuTotalUs, gpuInPassUs, gpuArgsUs;
  uint32_t gpuInvalid = 0;
};

struct PairResult {
  BenchPair pair;
  std::string clamp;
  ArmResult a, b;
  SampleStats deltaCpuUs;
  std::optional<SampleStats> deltaGpuUs;
  double drift = 0;
  bool unstable = false;
};

PairResult reduce_pair(const BenchPair& pair, const ArmSamples& a, const ArmSamples& b);
bool rebinds(uint32_t drawInPass, uint32_t globalDraw, uint32_t permille) noexcept;

struct DeltaFit {
  double alpha = 0, beta = 0, gamma = 0, rmsResidual = 0;
};

struct BenchRun {
  std::string device, backend, profile, adapter, adapterKind, build, validation;
  uint32_t vendorId = 0, deviceId = 0, msaa = 1, width = 0, height = 0, warmupBlocks = 0, blocks = 0;
  uint32_t uniformPermille = 1000, texturePermille = 500, erroredPairs = 0;
  uint32_t maxImmediateSize = 0, vertexStorageBuffers = 0;
  bool mdi = false, timestamps = false, indirectFirstInstance = false;
  std::vector<PairResult> pairs;
};

struct BenchDerived {
  std::optional<DeltaFit> cpuFit, gpuFit;
  std::optional<double> gate500CpuMs, gate500GpuMs, substBetaCpuUs, substBetaGpuUs, u32BetaCpuUs, u32BetaGpuUs,
      bundleBetaCpuUs, bundleBetaGpuUs, zeroBetaCpuUs, zeroBetaGpuUs, argsBarrierCpuUs, argsBarrierGpuUs;
};

std::optional<DeltaFit> fit_delta(const std::vector<double>& draws, const std::vector<double>& passes,
                                  const std::vector<double>& deltas);
BenchDerived bench_derived(const BenchRun& run, BenchPipeline pipeline);
std::string bench_json(const BenchRun& run);
std::string bench_summary(const BenchRun& run, std::string_view path);
} // namespace render_demo

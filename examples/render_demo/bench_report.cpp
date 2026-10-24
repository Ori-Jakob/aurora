#include "bench_report.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <utility>

namespace render_demo {
namespace {
using M = BenchMode;
using P = BenchPipeline;
using S = BenchState;
using A = BenchArgs;
using I = BenchIndex;

constexpr double DriftLimit = 0.10;

constexpr std::array<std::pair<uint32_t, uint32_t>, 9> FitCells{
    {{50, 1}, {500, 1}, {2000, 1}, {500, 8}, {500, 32}, {50, 8}, {2000, 8}, {50, 32}, {2000, 32}}};

constexpr BenchArm arm(M mode, P pipeline, S state, A args, I index) noexcept {
  return {.mode = mode, .pipeline = pipeline, .state = state, .args = args, .index = index};
}

std::string_view mode_name(M mode) noexcept {
  switch (mode) {
  case M::Direct:
    return "direct";
  case M::Indirect:
    return "indirect";
  case M::Zero:
    return "zero";
  case M::Mdi:
    return "mdi";
  case M::Bundle:
    return "bundle";
  }
  return "unknown";
}

std::string_view pipeline_name(P pipeline) noexcept { return pipeline == P::VertexIndex ? "vi" : "plain"; }

std::string_view state_name(S state) noexcept {
  switch (state) {
  case S::Gx:
    return "gx";
  case S::None:
    return "none";
  case S::Subst:
    return "subst";
  }
  return "unknown";
}

std::string_view args_name(A args) noexcept { return args == A::Gpu ? "gpu" : "cpu"; }

std::string_view index_name(I index) noexcept { return index == I::U16 ? "u16" : "u32"; }

double median(std::vector<double> values) { return sample_stats(std::move(values)).p50; }

std::vector<double> finite(const std::vector<double>& values) {
  std::vector<double> out;
  out.reserve(values.size());
  std::ranges::copy_if(values, std::back_inserter(out), [](double v) { return std::isfinite(v); });
  return out;
}

std::optional<SampleStats> optional_stats(const std::vector<double>& values) {
  std::vector<double> kept = finite(values);
  if (kept.empty()) {
    return std::nullopt;
  }
  return sample_stats(std::move(kept));
}

ArmResult reduce_arm(const ArmSamples& samples) {
  return {
      .encodeUs = sample_stats(samples.encodeUs),
      .finishUs = sample_stats(samples.finishUs),
      .submitUs = sample_stats(samples.submitUs),
      .totalUs = sample_stats(samples.totalUs),
      .gpuTotalUs = optional_stats(samples.gpuTotalUs),
      .gpuInPassUs = optional_stats(samples.gpuInPassUs),
      .gpuArgsUs = optional_stats(samples.gpuArgsUs),
      .gpuInvalid = samples.gpuInvalid,
  };
}

// Block k holds a[2k], b[2k], b[2k + 1], a[2k + 1] in time order, so a linear drift cancels in the difference
std::vector<double> block_deltas(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> deltas;
  const size_t blocks = std::min(a.size(), b.size()) / 2;
  for (size_t k = 0; k < blocks; ++k) {
    const double d = (b[2 * k] + b[2 * k + 1] - a[2 * k] - a[2 * k + 1]) / 2.0;
    if (std::isfinite(d)) {
      deltas.push_back(d);
    }
  }
  return deltas;
}

double drift_of(const std::vector<double>& a, double medianA) {
  const size_t blocks = a.size() / 2;
  if (blocks == 0 || medianA <= 0.0) {
    return 0.0;
  }
  std::vector<double> means(blocks);
  for (size_t k = 0; k < blocks; ++k) {
    means[k] = (a[2 * k] + a[2 * k + 1]) / 2.0;
  }
  const size_t quarter = std::max<size_t>(1, blocks / 4);
  const auto span = static_cast<std::ptrdiff_t>(quarter);
  const double first = median(std::vector<double>(means.begin(), means.begin() + span));
  const double last = median(std::vector<double>(means.end() - span, means.end()));
  return (last - first) / medianA;
}

const PairResult* find_pair(const BenchRun& run, const BenchArm& a, const BenchArm& b, uint32_t draws,
                            uint32_t passes) {
  for (const PairResult& result : run.pairs) {
    if (result.pair.a == a && result.pair.b == b && result.pair.draws == draws && result.pair.passes == passes) {
      return result.clamp.empty() ? &result : nullptr;
    }
  }
  return nullptr;
}

struct SlopePair {
  std::optional<double> cpu, gpu;
};

SlopePair slope(const BenchRun& run, const BenchArm& a, const BenchArm& b) {
  const PairResult* lo = find_pair(run, a, b, 500, 1);
  const PairResult* hi = find_pair(run, a, b, 2000, 1);
  SlopePair out;
  if (lo == nullptr || hi == nullptr) {
    return out;
  }
  out.cpu = (hi->deltaCpuUs.p50 - lo->deltaCpuUs.p50) / 1500.0;
  if (lo->deltaGpuUs && hi->deltaGpuUs) {
    out.gpu = (hi->deltaGpuUs->p50 - lo->deltaGpuUs->p50) / 1500.0;
  }
  return out;
}

void append(std::string& out, const char* format, auto... args) {
  char text[256];
  const int n = std::snprintf(text, sizeof(text), format, args...);
  if (n > 0) {
    out.append(text, std::min<size_t>(static_cast<size_t>(n), sizeof(text) - 1));
  }
}

void append_string(std::string& out, std::string_view value) {
  out += '"';
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      append(out, "\\u%04x", static_cast<unsigned>(c));
    } else {
      out += c;
    }
  }
  out += '"';
}

void append_stats(std::string& out, const SampleStats& s) {
  append(out, "{\"p50\":%.3f,\"p90\":%.3f,\"mean\":%.3f,\"min\":%.3f,\"max\":%.3f,\"count\":%u}", s.p50, s.p90, s.mean,
         s.min, s.max, s.count);
}

void append_optional_stats(std::string& out, const std::optional<SampleStats>& s) {
  if (s) {
    append_stats(out, *s);
  } else {
    out += "null";
  }
}

void append_number(std::string& out, const std::optional<double>& value, const char* format) {
  if (value) {
    append(out, format, *value);
  } else {
    out += "null";
  }
}

void append_arm_desc(std::string& out, const BenchArm& a) {
  out += "{\"mode\":";
  append_string(out, mode_name(a.mode));
  out += ",\"pipeline\":";
  append_string(out, pipeline_name(a.pipeline));
  out += ",\"state\":";
  append_string(out, state_name(a.state));
  out += ",\"args\":";
  append_string(out, args_name(a.args));
  out += ",\"index\":";
  append_string(out, index_name(a.index));
  out += '}';
}

void append_arm_result(std::string& out, const ArmResult& r) {
  out += "{\"cpuUs\":{\"encode\":";
  append_stats(out, r.encodeUs);
  out += ",\"finish\":";
  append_stats(out, r.finishUs);
  out += ",\"submit\":";
  append_stats(out, r.submitUs);
  out += ",\"total\":";
  append_stats(out, r.totalUs);
  out += "},\"gpuUs\":";
  if (r.gpuTotalUs) {
    out += "{\"total\":";
    append_stats(out, *r.gpuTotalUs);
    out += ",\"inPass\":";
    append_optional_stats(out, r.gpuInPassUs);
    out += ",\"args\":";
    append_optional_stats(out, r.gpuArgsUs);
    out += '}';
  } else {
    out += "null";
  }
  append(out, ",\"gpuInvalid\":%u}", r.gpuInvalid);
}

void append_fit(std::string& out, const std::optional<DeltaFit>& fit) {
  if (fit) {
    append(out, "{\"alpha\":%.6f,\"beta\":%.6f,\"gamma\":%.6f,\"rmsResidual\":%.6f}", fit->alpha, fit->beta, fit->gamma,
           fit->rmsResidual);
  } else {
    out += "null";
  }
}

void append_betas(std::string& out, const char* name, const std::optional<double>& cpu,
                  const std::optional<double>& gpu) {
  append(out, ",\"%s\":{\"cpuBetaUs\":", name);
  append_number(out, cpu, "%.6f");
  out += ",\"gpuBetaUs\":";
  append_number(out, gpu, "%.6f");
  out += '}';
}
} // namespace

std::vector<BenchPair> bench_pairs() {
  std::vector<BenchPair> pairs;
  for (const P pipeline : {P::VertexIndex, P::Plain}) {
    for (const auto& [draws, passes] : FitCells) {
      pairs.push_back({arm(M::Direct, pipeline, S::Gx, A::Gpu, I::U16),
                       arm(M::Indirect, pipeline, S::Gx, A::Gpu, I::U16), draws, passes});
    }
  }
  for (const P pipeline : {P::VertexIndex, P::Plain}) {
    for (const uint32_t draws : {500u, 2000u}) {
      pairs.push_back({arm(M::Direct, pipeline, S::None, A::Gpu, I::U16),
                       arm(M::Indirect, pipeline, S::None, A::Gpu, I::U16), draws, 1});
    }
    for (const uint32_t draws : {500u, 2000u}) {
      pairs.push_back({arm(M::Indirect, pipeline, S::None, A::Gpu, I::U16),
                       arm(M::Mdi, pipeline, S::None, A::Gpu, I::U16), draws, 1});
    }
  }
  for (const auto& [draws, passes] :
       {std::pair{500u, 1u}, std::pair{2000u, 1u}, std::pair{500u, 8u}, std::pair{2000u, 8u}}) {
    pairs.push_back({arm(M::Indirect, P::VertexIndex, S::Gx, A::Gpu, I::U16),
                     arm(M::Bundle, P::VertexIndex, S::Gx, A::Gpu, I::U16), draws, passes});
  }
  pairs.push_back({arm(M::Indirect, P::VertexIndex, S::Gx, A::Gpu, I::U16),
                   arm(M::Indirect, P::VertexIndex, S::Gx, A::Cpu, I::U16), 500, 1});
  for (const uint32_t draws : {500u, 2000u}) {
    pairs.push_back({arm(M::Indirect, P::VertexIndex, S::Gx, A::Gpu, I::U16),
                     arm(M::Zero, P::VertexIndex, S::Gx, A::Gpu, I::U16), draws, 1});
  }
  for (const uint32_t draws : {500u, 2000u}) {
    pairs.push_back({arm(M::Direct, P::VertexIndex, S::Gx, A::Gpu, I::U32),
                     arm(M::Indirect, P::VertexIndex, S::Gx, A::Gpu, I::U32), draws, 1});
  }
  for (const uint32_t draws : {500u, 2000u}) {
    pairs.push_back({arm(M::Direct, P::VertexIndex, S::Gx, A::Gpu, I::U16),
                     arm(M::Indirect, P::VertexIndex, S::Subst, A::Gpu, I::U32), draws, 1});
  }
  return pairs;
}

std::string arm_name(const BenchArm& a) {
  std::string out;
  for (const std::string_view part :
       {mode_name(a.mode), pipeline_name(a.pipeline), state_name(a.state), args_name(a.args), index_name(a.index)}) {
    if (!out.empty()) {
      out += '.';
    }
    out += part;
  }
  return out;
}

std::string pair_name(const BenchPair& pair) {
  std::string out = arm_name(pair.a) + "~" + arm_name(pair.b);
  append(out, ".n%u.k%u", pair.draws, pair.passes);
  return out;
}

SampleStats sample_stats(std::vector<double> samples) {
  SampleStats s;
  s.count = static_cast<uint32_t>(samples.size());
  if (samples.empty()) {
    return s;
  }
  std::ranges::sort(samples);
  const auto at = [&](double q) {
    const double pos = q * static_cast<double>(samples.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, samples.size() - 1);
    return samples[lo] + (samples[hi] - samples[lo]) * (pos - static_cast<double>(lo));
  };
  s.p50 = at(0.5);
  s.p90 = at(0.9);
  s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  s.min = samples.front();
  s.max = samples.back();
  return s;
}

bool unstable(const SampleStats& s) noexcept { return s.p50 > 0.0 && s.p90 > 1.25 * s.p50; }

PairResult reduce_pair(const BenchPair& pair, const ArmSamples& a, const ArmSamples& b) {
  PairResult result{.pair = pair, .a = reduce_arm(a), .b = reduce_arm(b)};
  result.deltaCpuUs = sample_stats(block_deltas(a.totalUs, b.totalUs));
  if (result.a.gpuTotalUs && result.b.gpuTotalUs) {
    if (std::vector<double> deltas = block_deltas(a.gpuTotalUs, b.gpuTotalUs); !deltas.empty()) {
      result.deltaGpuUs = sample_stats(std::move(deltas));
    }
  }
  result.drift = drift_of(a.totalUs, result.a.totalUs.p50);
  result.unstable = unstable(result.a.totalUs) || unstable(result.b.totalUs) || std::abs(result.drift) > DriftLimit;
  if (result.a.gpuTotalUs && result.b.gpuTotalUs) {
    result.unstable = result.unstable || unstable(*result.a.gpuTotalUs) || unstable(*result.b.gpuTotalUs);
  }
  return result;
}

bool rebinds(uint32_t drawInPass, uint32_t globalDraw, uint32_t permille) noexcept {
  const uint64_t draw = globalDraw;
  return drawInPass == 0 || (draw + 1) * permille / 1000 > draw * permille / 1000;
}

std::optional<DeltaFit> fit_delta(const std::vector<double>& draws, const std::vector<double>& passes,
                                  const std::vector<double>& deltas) {
  const size_t n = deltas.size();
  if (n < 3 || draws.size() != n || passes.size() != n) {
    return std::nullopt;
  }
  std::array<std::array<double, 4>, 3> m{};
  for (size_t i = 0; i < n; ++i) {
    const std::array<double, 3> x{1.0, draws[i], passes[i]};
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 3; ++c) {
        m[r][c] += x[r] * x[c];
      }
      m[r][3] += x[r] * deltas[i];
    }
  }
  for (size_t col = 0; col < 3; ++col) {
    size_t pivot = col;
    for (size_t r = col + 1; r < 3; ++r) {
      if (std::abs(m[r][col]) > std::abs(m[pivot][col])) {
        pivot = r;
      }
    }
    if (std::abs(m[pivot][col]) < 1e-12) {
      return std::nullopt;
    }
    std::swap(m[col], m[pivot]);
    for (size_t r = 0; r < 3; ++r) {
      if (r == col) {
        continue;
      }
      const double factor = m[r][col] / m[col][col];
      for (size_t c = col; c < 4; ++c) {
        m[r][c] -= factor * m[col][c];
      }
    }
  }
  DeltaFit fit{.alpha = m[0][3] / m[0][0], .beta = m[1][3] / m[1][1], .gamma = m[2][3] / m[2][2]};
  double squares = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double residual = fit.alpha + fit.beta * draws[i] + fit.gamma * passes[i] - deltas[i];
    squares += residual * residual;
  }
  fit.rmsResidual = std::sqrt(squares / static_cast<double>(n));
  return fit;
}

BenchDerived bench_derived(const BenchRun& run, BenchPipeline pipeline) {
  BenchDerived derived;
  const BenchArm direct = arm(M::Direct, pipeline, S::Gx, A::Gpu, I::U16);
  const BenchArm indirect = arm(M::Indirect, pipeline, S::Gx, A::Gpu, I::U16);
  std::vector<double> draws, passes, cpu, gpu;
  bool gpuComplete = true;
  for (const auto& [n, k] : FitCells) {
    const PairResult* result = find_pair(run, direct, indirect, n, k);
    if (result == nullptr) {
      return derived;
    }
    draws.push_back(n);
    passes.push_back(k);
    cpu.push_back(result->deltaCpuUs.p50);
    gpuComplete = gpuComplete && result->deltaGpuUs.has_value();
    gpu.push_back(result->deltaGpuUs ? result->deltaGpuUs->p50 : 0.0);
  }
  derived.cpuFit = fit_delta(draws, passes, cpu);
  if (gpuComplete) {
    derived.gpuFit = fit_delta(draws, passes, gpu);
  }
  if (const PairResult* gate = find_pair(run, direct, indirect, 500, 1)) {
    derived.gate500CpuMs = gate->deltaCpuUs.p50 / 1000.0;
    if (gate->deltaGpuUs) {
      derived.gate500GpuMs = gate->deltaGpuUs->p50 / 1000.0;
    }
  }
  if (pipeline != P::VertexIndex) {
    return derived;
  }
  const SlopePair subst = slope(run, direct, arm(M::Indirect, P::VertexIndex, S::Subst, A::Gpu, I::U32));
  derived.substBetaCpuUs = subst.cpu;
  derived.substBetaGpuUs = subst.gpu;
  const SlopePair wide = slope(run, arm(M::Direct, P::VertexIndex, S::Gx, A::Gpu, I::U32),
                               arm(M::Indirect, P::VertexIndex, S::Gx, A::Gpu, I::U32));
  derived.u32BetaCpuUs = wide.cpu;
  derived.u32BetaGpuUs = wide.gpu;
  const SlopePair bundle = slope(run, indirect, arm(M::Bundle, P::VertexIndex, S::Gx, A::Gpu, I::U16));
  derived.bundleBetaCpuUs = bundle.cpu;
  derived.bundleBetaGpuUs = bundle.gpu;
  const SlopePair zero = slope(run, indirect, arm(M::Zero, P::VertexIndex, S::Gx, A::Gpu, I::U16));
  derived.zeroBetaCpuUs = zero.cpu;
  derived.zeroBetaGpuUs = zero.gpu;
  // The pair runs GPU-written args first, so the barrier's price is the negated paired delta
  if (const PairResult* args =
          find_pair(run, indirect, arm(M::Indirect, P::VertexIndex, S::Gx, A::Cpu, I::U16), 500, 1)) {
    derived.argsBarrierCpuUs = -args->deltaCpuUs.p50;
    if (args->deltaGpuUs) {
      derived.argsBarrierGpuUs = -args->deltaGpuUs->p50;
    }
  }
  return derived;
}

std::string bench_json(const BenchRun& run) {
  std::string out = "{\"schema\":\"aurora.render_demo.indirect_bench/1\",\"device\":";
  append_string(out, run.device);
  out += ",\"backend\":";
  append_string(out, run.backend);
  out += ",\"profile\":";
  append_string(out, run.profile);
  out += ",\"adapter\":";
  append_string(out, run.adapter);
  out += ",\"adapterKind\":";
  append_string(out, run.adapterKind);
  append(out, ",\"vendorId\":%u,\"deviceId\":%u,\"build\":", run.vendorId, run.deviceId);
  append_string(out, run.build);
  out += ",\"validation\":";
  append_string(out, run.validation);
  append(out, ",\"msaa\":%u,\"target\":{\"width\":%u,\"height\":%u}", run.msaa, run.width, run.height);
  append(out,
         ",\"caps\":{\"multiDrawIndirect\":%s,\"indirectFirstInstance\":%s,\"timestampQuery\":%s,"
         "\"maxImmediateSize\":%u,\"vertexStorageBuffers\":%u}",
         run.mdi ? "true" : "false", run.indirectFirstInstance ? "true" : "false", run.timestamps ? "true" : "false",
         run.maxImmediateSize, run.vertexStorageBuffers);
  append(out, ",\"rebind\":{\"uniformPermille\":%u,\"texturePermille\":%u}", run.uniformPermille, run.texturePermille);
  append(out, ",\"wait\":\"spin\",\"warmupBlocks\":%u,\"blocks\":%u,\"erroredPairs\":%u,\"pairs\":[", run.warmupBlocks,
         run.blocks, run.erroredPairs);
  for (size_t i = 0; i < run.pairs.size(); ++i) {
    const PairResult& result = run.pairs[i];
    out += i == 0 ? "{\"name\":" : ",{\"name\":";
    append_string(out, pair_name(result.pair));
    append(out, ",\"draws\":%u,\"passes\":%u,\"a\":", result.pair.draws, result.pair.passes);
    append_arm_desc(out, result.pair.a);
    out += ",\"b\":";
    append_arm_desc(out, result.pair.b);
    out += ",\"clamp\":";
    append_string(out, result.clamp);
    append(out, ",\"unstable\":%s,\"drift\":%.4f,\"arms\":", result.unstable ? "true" : "false", result.drift);
    if (!result.clamp.empty()) {
      out += "null,\"deltaUs\":null}";
      continue;
    }
    out += "{\"a\":";
    append_arm_result(out, result.a);
    out += ",\"b\":";
    append_arm_result(out, result.b);
    out += "},\"deltaUs\":{\"cpu\":";
    append_stats(out, result.deltaCpuUs);
    out += ",\"gpu\":";
    append_optional_stats(out, result.deltaGpuUs);
    out += "}}";
  }
  out += "],\"derived\":{";
  const BenchDerived vi = bench_derived(run, P::VertexIndex);
  const BenchDerived plain = bench_derived(run, P::Plain);
  for (const auto& [name, derived] : {std::pair{"vi", &vi}, std::pair{"plain", &plain}}) {
    append(out, "%s\"%s\":{\"cpuFit\":", derived == &vi ? "" : ",", name);
    append_fit(out, derived->cpuFit);
    out += ",\"gpuFit\":";
    append_fit(out, derived->gpuFit);
    out += ",\"gate500\":{\"cpuMs\":";
    append_number(out, derived->gate500CpuMs, "%.6f");
    out += ",\"gpuMs\":";
    append_number(out, derived->gate500GpuMs, "%.6f");
    out += "}}";
  }
  append_betas(out, "subst", vi.substBetaCpuUs, vi.substBetaGpuUs);
  append_betas(out, "u32", vi.u32BetaCpuUs, vi.u32BetaGpuUs);
  append_betas(out, "bundle", vi.bundleBetaCpuUs, vi.bundleBetaGpuUs);
  append_betas(out, "zero", vi.zeroBetaCpuUs, vi.zeroBetaGpuUs);
  out += ",\"argsBarrier\":{\"cpuUs\":";
  append_number(out, vi.argsBarrierCpuUs, "%.3f");
  out += ",\"gpuUs\":";
  append_number(out, vi.argsBarrierGpuUs, "%.3f");
  out += "}}}\n";
  return out;
}

std::string bench_summary(const BenchRun& run, std::string_view path) {
  const BenchDerived vi = bench_derived(run, P::VertexIndex);
  std::string out;
  append(out, "render_demo: bench: %zu pairs on %s/%s (device %s, validation %s, build %s, msaa %u, %ux%u), ",
         run.pairs.size(), run.backend.c_str(), run.profile.c_str(), run.device.c_str(),
         run.validation == "skipped" ? "skipped" : "on", run.build.c_str(), run.msaa, run.width, run.height);
  append(out, "mdi %s, timestamps %s, errored %u; 500 vi gx draws: indirect-direct cpu ", run.mdi ? "yes" : "no",
         run.timestamps ? "yes" : "no", run.erroredPairs);
  if (vi.gate500CpuMs) {
    append(out, "%+.3f ms", *vi.gate500CpuMs);
  } else {
    out += "n/a";
  }
  out += ", gpu ";
  if (vi.gate500GpuMs) {
    append(out, "%+.3f ms", *vi.gate500GpuMs);
  } else {
    out += "n/a";
  }
  out += " (paired p50); wrote ";
  out += path;
  return out;
}
} // namespace render_demo

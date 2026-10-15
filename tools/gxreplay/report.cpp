#include "report.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace aurora::gxreplay {
namespace {
struct Summary {
  double sum = 0.0;
  double mean = 0.0;
  double p95 = 0.0;
};

Summary summarize(std::vector<double> values) {
  Summary out;
  for (const double value : values) {
    out.sum += value;
  }
  if (values.empty()) {
    return out;
  }
  std::sort(values.begin(), values.end());
  out.mean = out.sum / static_cast<double>(values.size());
  const auto rank = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(values.size())));
  out.p95 = values[std::max<size_t>(rank, 1) - 1];
  return out;
}

template <typename T>
std::vector<uint32_t> run_ids(std::span<const T> timings) {
  std::vector<uint32_t> runs;
  for (const auto& timing : timings) {
    if (std::find(runs.begin(), runs.end(), timing.run) == runs.end()) {
      runs.push_back(timing.run);
    }
  }
  return runs;
}
} // namespace

std::string json_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        out += fmt::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
      } else {
        out += c;
      }
      break;
    }
  }
  return out;
}

std::string hex64(uint64_t value) { return fmt::format("{:016x}", value); }

std::string format_frames_json(const ReportContext& context, std::span<const FrameResult> frames) {
  std::string out;
  out += "{\n";
  out += "  \"tool\": \"aurora_gxreplay\",\n";
  out += fmt::format("  \"traceFile\": \"{}\",\n", json_escape(context.traceFile));
  out += fmt::format("  \"traceVersion\": \"{}\",\n", json_escape(context.traceVersion));
  out += fmt::format("  \"synthetic\": {},\n", context.synthetic);
  out += fmt::format("  \"backend\": \"{}\",\n", json_escape(context.backend));
  out += fmt::format("  \"adapter\": \"{}\",\n", json_escape(context.adapter));
  out += fmt::format("  \"runs\": {},\n", context.runs);
  out += fmt::format("  \"deterministic\": {},\n", context.deterministic);
  if (frames.empty()) {
    out += "  \"frames\": []\n";
  } else {
    out += "  \"frames\": [\n";
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& frame = frames[i];
      out += "    {\n";
      out += fmt::format("      \"frame\": {},\n", frame.frame);
      out += fmt::format("      \"sourceFrame\": {},\n", frame.sourceFrame);
      out += fmt::format("      \"width\": {},\n", frame.width);
      out += fmt::format("      \"height\": {},\n", frame.height);
      out += fmt::format("      \"rgbaXxh3\": \"{}\",\n", hex64(frame.rgbaHash));
      out += fmt::format("      \"fifoBytes\": {},\n", frame.fifoBytes);
      out += fmt::format("      \"mapRecords\": {},\n", frame.mapRecords);
      out += fmt::format("      \"misses\": {},\n", frame.misses);
      out += fmt::format("      \"draws\": {}\n", frame.draws);
      out += i + 1 == frames.size() ? "    }\n" : "    },\n";
    }
    out += "  ]\n";
  }
  out += "}\n";
  return out;
}

std::string format_timing_json(std::span<const FrameTiming> timings) {
  const std::vector<uint32_t> runs = run_ids(timings);

  std::string out;
  out += "{\n";
  out += runs.empty() ? "  \"runs\": []\n" : "  \"runs\": [\n";
  for (size_t r = 0; r < runs.size(); ++r) {
    std::vector<double> totals;
    out += "    {\n";
    out += fmt::format("      \"run\": {},\n", runs[r]);
    out += "      \"frames\": [\n";
    std::vector<const FrameTiming*> frames;
    for (const auto& timing : timings) {
      if (timing.run == runs[r]) {
        frames.push_back(&timing);
        totals.push_back(timing.totalMs);
      }
    }
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& frame = *frames[i];
      out += fmt::format(
          "        {{\"frame\": {}, \"processMs\": {:.3f}, \"finishMs\": {:.3f}, \"gpuWaitMs\": {:.3f}, "
          "\"totalMs\": {:.3f}}}{}\n",
          frame.frame, frame.processMs, frame.finishMs, frame.gpuWaitMs, frame.totalMs,
          i + 1 == frames.size() ? "" : ",");
    }
    out += "      ],\n";
    const Summary summary = summarize(std::move(totals));
    out += fmt::format("      \"meanMs\": {:.3f},\n", summary.mean);
    out += fmt::format("      \"p95Ms\": {:.3f},\n", summary.p95);
    out += fmt::format("      \"totalMs\": {:.3f}\n", summary.sum);
    out += r + 1 == runs.size() ? "    }\n" : "    },\n";
  }
  if (!runs.empty()) {
    out += "  ]\n";
  }
  out += "}\n";
  return out;
}

std::string format_gpu_timing_json(bool supported, std::span<const GpuFrameTiming> timings) {
  const std::vector<uint32_t> runs = supported ? run_ids(timings) : std::vector<uint32_t>{};
  if (runs.empty()) {
    return fmt::format("{{\"supported\": {}, \"runs\": []}}\n", supported);
  }

  std::string out;
  out += "{\n";
  out += "  \"supported\": true,\n";
  out += "  \"runs\": [\n";
  for (size_t r = 0; r < runs.size(); ++r) {
    std::vector<const GpuFrameTiming*> frames;
    for (const auto& timing : timings) {
      if (timing.run == runs[r]) {
        frames.push_back(&timing);
      }
    }
    out += "    {\n";
    out += fmt::format("      \"run\": {},\n", runs[r]);
    out += "      \"frames\": [\n";
    std::vector<double> frameMs;
    std::vector<std::string_view> names;
    std::vector<std::vector<double>> nameMs;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& frame = *frames[i];
      frameMs.push_back(frame.frameMs);
      std::string passes;
      for (size_t p = 0; p < frame.passes.size(); ++p) {
        const auto& pass = frame.passes[p];
        passes +=
            fmt::format("{}{{\"name\": \"{}\", \"ms\": {:.3f}}}", p == 0 ? "" : ", ", json_escape(pass.name), pass.ms);
        const auto it = std::find(names.begin(), names.end(), std::string_view{pass.name});
        if (it == names.end()) {
          names.push_back(pass.name);
          nameMs.push_back({pass.ms});
        } else {
          nameMs[static_cast<size_t>(it - names.begin())].push_back(pass.ms);
        }
      }
      out += fmt::format("        {{\"frame\": {}, \"gpuFrameMs\": {:.3f}, \"dropped\": {}, \"passes\": [{}]}}{}\n",
                         frame.frame, frame.frameMs, frame.dropped, passes, i + 1 == frames.size() ? "" : ",");
    }
    out += "      ],\n";
    const Summary frameSummary = summarize(std::move(frameMs));
    out += fmt::format("      \"meanGpuFrameMs\": {:.3f},\n", frameSummary.mean);
    out += fmt::format("      \"p95GpuFrameMs\": {:.3f},\n", frameSummary.p95);
    out += names.empty() ? "      \"passes\": []\n" : "      \"passes\": [\n";
    for (size_t n = 0; n < names.size(); ++n) {
      const size_t count = nameMs[n].size();
      const Summary summary = summarize(std::move(nameMs[n]));
      out += fmt::format("        {{\"name\": \"{}\", \"meanMs\": {:.3f}, \"p95Ms\": {:.3f}, \"frames\": {}}}{}\n",
                         json_escape(names[n]), summary.mean, summary.p95, count, n + 1 == names.size() ? "" : ",");
    }
    if (!names.empty()) {
      out += "      ]\n";
    }
    out += r + 1 == runs.size() ? "    }\n" : "    },\n";
  }
  out += "  ]\n";
  out += "}\n";
  return out;
}
} // namespace aurora::gxreplay

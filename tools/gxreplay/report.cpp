#include "report.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace aurora::gxreplay {
namespace {
namespace scene = render::scene;

constexpr std::array<std::string_view, 6> MarkerNames{"sceneBegin", "opaqueDone", "prePost",
                                                      "sceneEnd",   "beforeHud",  "frameEnd"};

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
      out += fmt::format("      \"hintBytes\": {},\n", frame.hintBytes);
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

namespace {
std::string json_string_or_null(std::string_view value) {
  return value.empty() ? std::string{"null"} : fmt::format("\"{}\"", json_escape(value));
}

std::string object_id(uint64_t id) { return fmt::format("\"0x{:016x}\"", id); }

template <typename Enum, size_t N, typename NameFn>
std::string named_counts(const std::array<uint32_t, N>& counts, NameFn name) {
  std::string out = "{";
  for (size_t i = 0; i < N; ++i) {
    out += fmt::format("{}\"{}\": {}", i == 0 ? "" : ", ", name(static_cast<Enum>(i)), counts[i]);
  }
  return out + "}";
}

std::string category_counts(const std::array<uint32_t, 32>& counts) {
  std::string out = "{";
  bool first = true;
  for (uint32_t bit = 0; bit < counts.size(); ++bit) {
    const std::string_view name = scene::category_name(bit);
    if (name.empty()) {
      continue;
    }
    out += fmt::format("{}\"{}\": {}", first ? "" : ", ", name, counts[bit]);
    first = false;
  }
  return out + "}";
}

std::string under_scope(const std::array<scene::RoleScopeCounts, scene::RoleCount>& scopes) {
  std::string out = "{";
  for (size_t role = 0; role < scopes.size(); ++role) {
    const scene::RoleScopeCounts& counts = scopes[role];
    out += fmt::format(
        "{}\"{}\": {{\"draws\": {}, \"asRole\": {}, \"main\": {}, \"roleOnly\": {}, \"captures\": {}, \"skyKind\": "
        "{}}}",
        role == 0 ? "" : ", ", scene::role_name(static_cast<scene::Role>(role)), counts.draws, counts.asRole,
        counts.main, counts.roleOnly, counts.captures, counts.skyKind);
  }
  return out + "}";
}

std::string markers(const std::array<int32_t, render::StageCount>& markerDraw) {
  std::string out = "{";
  for (size_t i = 0; i < MarkerNames.size(); ++i) {
    out += fmt::format("{}\"{}\": {}", i == 0 ? "" : ", ", MarkerNames[i], markerDraw[i]);
  }
  return out + "}";
}

std::string hint_counts(const scene::HintCounts& hints) {
  return fmt::format(
      "{{\"cameras\": {}, \"mainCameras\": {}, \"scopePushes\": {}, \"scopePops\": {}, \"objectBegins\": {}, "
      "\"objectEnds\": {}, \"markers\": {}, \"staleMarkers\": {}, \"scopeOverflow\": {}, \"scopeUnderflow\": {}, "
      "\"nestedObjects\": {}, \"objectUnderflow\": {}, \"openScopes\": {}, \"openObjects\": {}, \"errors\": {}}}",
      hints.cameras, hints.mainCameras, hints.scopePushes, hints.scopePops, hints.objectBegins, hints.objectEnds,
      hints.markers, hints.staleMarkers, hints.scopeOverflow, hints.scopeUnderflow, hints.nestedObjects,
      hints.objectUnderflow, hints.openScopes, hints.openObjects, hints.errors());
}

std::string role_list(uint8_t roles) {
  std::string out = "[";
  bool first = true;
  for (size_t role = 0; role < scene::RoleCount; ++role) {
    if ((roles & (1u << role)) != 0) {
      out += fmt::format("{}\"{}\"", first ? "" : ", ", scene::role_name(static_cast<scene::Role>(role)));
      first = false;
    }
  }
  return out + "]";
}

uint32_t main_class_draws(const scene::FrameStats& stats) {
  const auto kind = [&stats](scene::Kind value) { return stats.kinds[static_cast<size_t>(value)]; };
  return kind(scene::Kind::MainOpaque) + kind(scene::Kind::MainBlended) + kind(scene::Kind::Overlay) +
         kind(scene::Kind::MainDepthOnly);
}
} // namespace

std::string format_classify_json(const ReportContext& context, std::string_view view, std::string_view viewBit,
                                 std::span<const ClassifyFrame> frames, bool objects) {
  std::string out;
  out += "{\n";
  out += "  \"schema\": \"aurora.classify\",\n";
  out += "  \"version\": 1,\n";
  out += fmt::format("  \"trace\": \"{}\",\n", json_escape(context.traceFile));
  out += fmt::format("  \"backend\": \"{}\",\n", json_escape(context.backend));
  out += fmt::format("  \"adapter\": \"{}\",\n", json_escape(context.adapter));
  out += fmt::format("  \"view\": {},\n", json_string_or_null(view));
  out += fmt::format("  \"viewBit\": {},\n", json_string_or_null(viewBit));
  if (frames.empty()) {
    out += "  \"frames\": []\n";
    out += "}\n";
    return out;
  }
  out += "  \"frames\": [\n";
  for (size_t i = 0; i < frames.size(); ++i) {
    const ClassifyFrame& frame = frames[i];
    const scene::FrameReport& report = frame.report;
    const scene::FrameStats& stats = report.stats;
    const scene::ViewCounts& view = stats.view;
    out += "    {\n";
    out += fmt::format("      \"frame\": {},\n", frame.frame);
    out += fmt::format("      \"draws\": {},\n", stats.draws);
    out += fmt::format("      \"suppressed\": {},\n", stats.suppressed);
    out += fmt::format("      \"mergeBreaks\": {},\n", stats.mergeBreaks);
    out += fmt::format("      \"captureCandidates\": {},\n", stats.captureCandidates);
    out += fmt::format("      \"fakeShadow\": {},\n", stats.fakeShadow);
    out += fmt::format("      \"fakeShadowOverlay\": {},\n", stats.fakeShadowOverlay);
    out += fmt::format("      \"unclassified\": {},\n", stats.unclassified);
    out += fmt::format("      \"tevPolicyDraws\": {},\n", stats.tevPolicyDraws);
    out += fmt::format("      \"skyMisses\": {},\n", stats.skyMisses);
    out += fmt::format("      \"droppedRecords\": {},\n", stats.droppedRecords);
    out += fmt::format("      \"droppedObjects\": {},\n", stats.droppedObjects);
    out += fmt::format("      \"kinds\": {},\n", named_counts<scene::Kind>(stats.kinds, scene::kind_name));
    out += fmt::format("      \"roles\": {},\n", named_counts<scene::Role>(stats.roles, scene::role_name));
    out += fmt::format("      \"checks\": {},\n", named_counts<scene::MainCheck>(stats.checks, scene::check_name));
    out += fmt::format("      \"windows\": {},\n", named_counts<scene::Window>(stats.windows, scene::window_name));
    out += fmt::format("      \"categories\": {},\n", category_counts(stats.categories));
    out += fmt::format("      \"underScope\": {},\n", under_scope(stats.underScope));
    out += fmt::format("      \"markers\": {},\n", markers(stats.markerDraw));
    out += fmt::format("      \"hints\": {},\n", hint_counts(stats.hints));
    out += fmt::format("      \"camera\": {{\"main\": {}, \"cutEpoch\": {}}},\n", report.mainCamera, report.cutEpoch);
    out += fmt::format("      \"sceneEpoch\": {},\n",
                       report.sceneEpochKnown ? std::to_string(report.sceneEpoch) : std::string{"null"});
    out += fmt::format(
        "      \"viewCounts\": {{\"substituted\": {}, \"untouched\": {}, \"pending\": {}, "
        "\"overBudget\": {}, \"uniformBytes\": {}}},\n",
        view.substituted, view.untouched, view.pending, view.overBudget, view.uniformBytes);
    out += fmt::format("      \"uniformBytes\": {}{}\n", frame.uniformBytes, objects ? "," : "");
    if (objects) {
      if (report.objects.empty()) {
        out += "      \"objects\": []\n";
      } else {
        out += "      \"objects\": [\n";
        for (size_t o = 0; o < report.objects.size(); ++o) {
          const scene::ObjectStats& object = report.objects[o];
          out += fmt::format(
              "        {{\"id\": {}, \"subId\": {}, \"generation\": {}, \"categories\": {}, "
              "\"flags\": {}, \"roles\": {}, \"draws\": {}, \"mainDraws\": {}}}{}\n",
              object_id(object.ref.id), object.ref.subId, object.ref.generation, object.categories, object.flags,
              role_list(object.roles), object.draws, object.mainDraws, o + 1 == report.objects.size() ? "" : ",");
        }
        out += "      ]\n";
      }
    }
    out += i + 1 == frames.size() ? "    }\n" : "    },\n";
  }
  out += "  ]\n";
  out += "}\n";
  return out;
}

std::string format_classify_draw_line(uint32_t frame, const scene::ClassifiedDraw& draw) {
  const scene::DrawClass& cls = draw.cls;
  const scene::ClassKey& key = cls.key;
  return fmt::format(
      "{{\"frame\": {}, \"index\": {}, \"kind\": \"{}\", \"role\": \"{}\", \"check\": \"{}\", "
      "\"window\": \"{}\", \"object\": {}, \"subId\": {}, \"generation\": {}, \"categories\": {}, "
      "\"flags\": {}, \"material\": {}, \"tevPolicy\": {}, \"scope\": {}, \"vertices\": {}, "
      "\"indexed\": {}, \"suppressed\": {}}}\n",
      frame, cls.index, scene::kind_name(cls.kind), scene::role_name(key.role), scene::check_name(cls.check),
      scene::window_name(key.window), key.object.valid ? object_id(key.object.id) : std::string{"null"},
      key.object.subId, key.object.generation, key.categories, key.flags, key.materialClass, key.tevPolicy, key.scope,
      draw.vertices, draw.indexed, cls.suppressed);
}

std::string format_classify_summary(std::span<const ClassifyFrame> frames, uint64_t malformedHints) {
  uint64_t draws = 0;
  uint64_t main = 0;
  uint64_t sky = 0;
  uint64_t skyMisses = 0;
  uint64_t suppressed = 0;
  uint64_t hintErrors = 0;
  uint64_t substituted = 0;
  for (const ClassifyFrame& frame : frames) {
    const scene::FrameStats& stats = frame.report.stats;
    draws += stats.draws;
    main += main_class_draws(stats);
    sky += stats.kinds[static_cast<size_t>(scene::Kind::Sky)];
    skyMisses += stats.skyMisses;
    suppressed += stats.suppressed;
    hintErrors += stats.hints.errors();
    substituted += stats.view.substituted;
  }
  return fmt::format(
      "aurora_gxreplay: classification: frames {}, draws {}, main {}, sky {}, sky misses {}, "
      "suppressed {}, hint errors {}, malformed hints {}, view substituted {}\n",
      frames.size(), draws, main, sky, skyMisses, suppressed, hintErrors, malformedHints, substituted);
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

namespace {
constexpr std::array<std::string_view, 7> CpuZoneMsNames{
    "fifoMs", "pushDrawMs", "arraysMs", "pipelineBuildMs", "bindGroupBuildMs", "uniformBuildMs", "fogLutMs",
};
constexpr std::array<std::string_view, 5> CpuSectionNames{"arrays", "pipelineBuild", "bindGroupBuild", "uniformBuild",
                                                          "fogLut"};
constexpr size_t FirstSectionMs = 2;
constexpr std::array<std::string_view, 15> CpuZoneCountNames{
    "fifoCalls",        "pushDraws",     "arrayMisses",  "arrayBytes",   "pipelineBuilds",
    "bindGroupBuilds",  "uniformBuilds", "fogLutPushes", "fogLutBytes",  "resolves",
    "uniqueArrayBytes", "vertBytes",     "indexBytes",   "uniformBytes", "storageBytes",
};

std::array<double, CpuZoneMsNames.size()> cpu_zone_ms(const CpuFrameZones& frame) {
  return {frame.fifoMs,           frame.pushDrawMs,     frame.arraysMs, frame.pipelineBuildMs,
          frame.bindGroupBuildMs, frame.uniformBuildMs, frame.fogLutMs};
}

std::array<uint64_t, CpuZoneCountNames.size()> cpu_zone_counts(const CpuFrameZones& frame) {
  return {frame.fifoCalls,        frame.pushDraws,     frame.arrayMisses,  frame.arrayBytes,   frame.pipelineBuilds,
          frame.bindGroupBuilds,  frame.uniformBuilds, frame.fogLutPushes, frame.fogLutBytes,  frame.resolves,
          frame.uniqueArrayBytes, frame.vertBytes,     frame.indexBytes,   frame.uniformBytes, frame.storageBytes};
}

std::string cpu_zone_row(const CpuFrameZones& frame) {
  std::string out = fmt::format("{{\"frame\": {}", frame.frame);
  const auto ms = cpu_zone_ms(frame);
  for (size_t i = 0; i < ms.size(); ++i) {
    out += fmt::format(", \"{}\": {:.3f}", CpuZoneMsNames[i], ms[i]);
  }
  const auto counts = cpu_zone_counts(frame);
  for (size_t i = 0; i < counts.size(); ++i) {
    out += fmt::format(", \"{}\": {}", CpuZoneCountNames[i], counts[i]);
  }
  return out + "}";
}

std::string warm_cpu_zones(bool drawZones, std::span<const CpuFrameZones> frames, const std::vector<uint32_t>& runs) {
  // Run 0 pays blocking pipeline compiles and frame 0 re-decodes every texture, so warm leaves both out
  const uint32_t coldRun = *std::min_element(runs.begin(), runs.end());
  std::vector<const CpuFrameZones*> warm;
  std::vector<uint32_t> warmRuns;
  for (const auto& frame : frames) {
    if (frame.run != coldRun && frame.frame != 0) {
      warm.push_back(&frame);
      if (std::find(warmRuns.begin(), warmRuns.end(), frame.run) == warmRuns.end()) {
        warmRuns.push_back(frame.run);
      }
    }
  }
  if (warm.empty()) {
    return "null";
  }

  const auto count = static_cast<double>(warm.size());
  std::vector<double> fifoMs;
  std::vector<double> pushDrawMs;
  std::array<double, CpuZoneMsNames.size()> msSums{};
  std::array<double, CpuZoneCountNames.size()> countSums{};
  double ratioSum = 0.0;
  double ratioMax = 0.0;
  size_t ratioRows = 0;
  for (const CpuFrameZones* frame : warm) {
    fifoMs.push_back(frame->fifoMs);
    pushDrawMs.push_back(frame->pushDrawMs);
    const auto ms = cpu_zone_ms(*frame);
    for (size_t i = 0; i < ms.size(); ++i) {
      msSums[i] += ms[i];
    }
    const auto counts = cpu_zone_counts(*frame);
    for (size_t i = 0; i < counts.size(); ++i) {
      countSums[i] += static_cast<double>(counts[i]);
    }
    if (frame->uniqueArrayBytes != 0) {
      const double ratio = (static_cast<double>(frame->storageBytes) - static_cast<double>(frame->fogLutBytes)) /
                           static_cast<double>(frame->uniqueArrayBytes);
      ratioSum += ratio;
      ratioMax = ratioRows == 0 ? ratio : std::max(ratioMax, ratio);
      ++ratioRows;
    }
  }
  const Summary fifo = summarize(std::move(fifoMs));
  const Summary pushDraw = summarize(std::move(pushDrawMs));

  std::string sections = "{";
  for (size_t i = 0; i < CpuSectionNames.size(); ++i) {
    sections +=
        fmt::format("{}\"{}\": {:.3f}", i == 0 ? "" : ", ", CpuSectionNames[i], msSums[FirstSectionMs + i] / count);
  }
  sections += "}";
  std::string means = "{";
  for (size_t i = 0; i < CpuZoneCountNames.size(); ++i) {
    means += fmt::format("{}\"{}\": {:.1f}", i == 0 ? "" : ", ", CpuZoneCountNames[i], countSums[i] / count);
  }
  means += "}";
  const std::string storageRatio =
      drawZones && ratioRows != 0
          ? fmt::format("{{\"mean\": {:.3f}, \"max\": {:.3f}}}", ratioSum / static_cast<double>(ratioRows), ratioMax)
          : std::string{"null"};

  std::string out = "{\n";
  out += fmt::format("    \"firstRun\": {},\n", *std::min_element(warmRuns.begin(), warmRuns.end()));
  out += fmt::format("    \"runs\": {},\n", warmRuns.size());
  out += fmt::format("    \"frames\": {},\n", warm.size());
  out += fmt::format("    \"meanFifoMs\": {:.3f},\n", fifo.mean);
  out += fmt::format("    \"p95FifoMs\": {:.3f},\n", fifo.p95);
  out += fmt::format("    \"meanPushDrawMs\": {:.3f},\n", pushDraw.mean);
  out += fmt::format("    \"p95PushDrawMs\": {:.3f},\n", pushDraw.p95);
  out += fmt::format("    \"sectionMeanMs\": {},\n", sections);
  out += fmt::format("    \"storageRatio\": {},\n", storageRatio);
  out += fmt::format("    \"means\": {}\n", means);
  out += "  }";
  return out;
}
} // namespace

std::string format_cpu_zones_json(bool drawZones, std::span<const CpuFrameZones> frames) {
  const std::vector<uint32_t> runs = run_ids(frames);
  if (runs.empty()) {
    return fmt::format("{{\"drawZones\": {}, \"runs\": [], \"warm\": null}}\n", drawZones);
  }

  std::string out;
  out += "{\n";
  out += fmt::format("  \"drawZones\": {},\n", drawZones);
  out += "  \"runs\": [\n";
  for (size_t r = 0; r < runs.size(); ++r) {
    std::vector<const CpuFrameZones*> rows;
    std::vector<double> fifoMs;
    std::vector<double> pushDrawMs;
    for (const auto& frame : frames) {
      if (frame.run == runs[r]) {
        rows.push_back(&frame);
        fifoMs.push_back(frame.fifoMs);
        pushDrawMs.push_back(frame.pushDrawMs);
      }
    }
    out += "    {\n";
    out += fmt::format("      \"run\": {},\n", runs[r]);
    out += "      \"frames\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      out += fmt::format("        {}{}\n", cpu_zone_row(*rows[i]), i + 1 == rows.size() ? "" : ",");
    }
    out += "      ],\n";
    const Summary fifo = summarize(std::move(fifoMs));
    const Summary pushDraw = summarize(std::move(pushDrawMs));
    out += fmt::format("      \"meanFifoMs\": {:.3f},\n", fifo.mean);
    out += fmt::format("      \"p95FifoMs\": {:.3f},\n", fifo.p95);
    out += fmt::format("      \"meanPushDrawMs\": {:.3f},\n", pushDraw.mean);
    out += fmt::format("      \"p95PushDrawMs\": {:.3f}\n", pushDraw.p95);
    out += r + 1 == runs.size() ? "    }\n" : "    },\n";
  }
  out += "  ],\n";
  out += fmt::format("  \"warm\": {}\n", warm_cpu_zones(drawZones, frames, runs));
  out += "}\n";
  return out;
}

bool check_cpu_zones(std::span<const CpuFrameZones> frames, size_t framesPerRun, bool drawZones, std::string& error) {
  if (frames.empty()) {
    error = "cpu zones: no frame rows";
    return false;
  }
  for (const uint32_t run : run_ids(frames)) {
    const auto rows = static_cast<size_t>(
        std::count_if(frames.begin(), frames.end(), [run](const CpuFrameZones& frame) { return frame.run == run; }));
    if (rows != framesPerRun) {
      error = fmt::format("cpu zones: run {} has {} rows instead of {}", run, rows, framesPerRun);
      return false;
    }
  }
  double fifoMs = 0.0;
  double pushDrawMs = 0.0;
  for (const auto& frame : frames) {
    if (frame.fifoCalls == 0) {
      error = fmt::format("cpu zones: run {} frame {} has no FIFO calls", frame.run, frame.frame);
      return false;
    }
    fifoMs += frame.fifoMs;
    pushDrawMs += frame.pushDrawMs;
  }
  if (fifoMs == 0.0) {
    error = "cpu zones: no FIFO time was measured";
    return false;
  }
  if (drawZones != (pushDrawMs != 0.0)) {
    error = drawZones ? "cpu zones: no draw time was measured" : "cpu zones: draw time was measured without draw zones";
    return false;
  }
  return true;
}
} // namespace aurora::gxreplay

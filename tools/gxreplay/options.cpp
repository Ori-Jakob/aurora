#include "options.hpp"

#include "render/debug/views.hpp"

#include <array>
#include <charconv>
#include <utility>

namespace aurora::gxreplay {
namespace {
constexpr std::array<std::pair<std::string_view, AuroraBackend>, 9> BackendNames{{
    {"auto", BACKEND_AUTO},
    {"d3d12", BACKEND_D3D12},
    {"vulkan", BACKEND_VULKAN},
    {"metal", BACKEND_METAL},
    {"d3d11", BACKEND_D3D11},
    {"opengl", BACKEND_OPENGL},
    {"opengles", BACKEND_OPENGLES},
    {"webgpu", BACKEND_WEBGPU},
    {"null", BACKEND_NULL},
}};

constexpr std::array<std::pair<std::string_view, AuroraFeatureProfile>, 5> ProfileNames{{
    {"auto", FEATURE_PROFILE_AUTO},
    {"full", FEATURE_PROFILE_FULL},
    {"compat", FEATURE_PROFILE_COMPAT},
    {"metal", FEATURE_PROFILE_METAL},
    {"metal32", FEATURE_PROFILE_METAL32},
}};

std::optional<uint32_t> parse_u32(std::string_view text) noexcept {
  uint32_t value = 0;
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (text.empty() || ec != std::errc{} || ptr != end) {
    return std::nullopt;
  }
  return value;
}

bool apply_path(std::filesystem::path& target, std::string_view value) {
  if (value.empty()) {
    return false;
  }
  target = std::filesystem::path{std::u8string{value.begin(), value.end()}};
  return true;
}

bool apply_out(Options& options, std::string_view value) { return apply_path(options.outDir, value); }
bool apply_data_dir(Options& options, std::string_view value) { return apply_path(options.dataDir, value); }
bool apply_replacements(Options& options, std::string_view value) { return apply_path(options.replacementsDir, value); }
bool apply_write_self_test(Options& options, std::string_view value) {
  return apply_path(options.writeSelfTest, value);
}

bool apply_backend(Options& options, std::string_view value) {
  const auto backend = parse_backend(value);
  if (backend) {
    options.backend = *backend;
  }
  return backend.has_value();
}

bool apply_profile(Options& options, std::string_view value) {
  const auto profile = parse_profile(value);
  if (profile) {
    options.profile = *profile;
  }
  return profile.has_value();
}

bool apply_frames(Options& options, std::string_view value) {
  const auto range = parse_frame_range(value);
  if (range) {
    options.frames = *range;
  }
  return range.has_value();
}

bool apply_repeat(Options& options, std::string_view value) {
  const auto repeat = parse_u32(value);
  if (!repeat || *repeat == 0) {
    return false;
  }
  options.repeat = *repeat;
  return true;
}

bool apply_no_png(Options& options, std::string_view) {
  options.writePng = false;
  return true;
}

bool apply_gpu_timings(Options& options, std::string_view) {
  options.gpuTimings = true;
  return true;
}

bool apply_cpu_zones(Options& options, std::string_view) {
  options.cpuZones = true;
  return true;
}

bool apply_cpu_draw_zones(Options& options, std::string_view) {
  options.cpuZones = true;
  options.cpuDrawZones = true;
  return true;
}

bool apply_allow_incomplete(Options& options, std::string_view) {
  options.allowIncomplete = true;
  return true;
}

bool apply_allow_device_mismatch(Options& options, std::string_view) {
  options.allowDeviceMismatch = true;
  return true;
}

bool apply_allow_misses(Options& options, std::string_view) {
  options.allowMisses = true;
  return true;
}

bool apply_allow_cpu(Options& options, std::string_view) {
  options.allowCpu = true;
  return true;
}

bool apply_show_window(Options& options, std::string_view) {
  options.showWindow = true;
  return true;
}

bool apply_self_test(Options& options, std::string_view) {
  options.selfTest = true;
  return true;
}

bool apply_info(Options& options, std::string_view) {
  options.info = true;
  return true;
}

bool apply_verbose(Options& options, std::string_view) {
  options.verbose = true;
  return true;
}

bool apply_help(Options& options, std::string_view) {
  options.help = true;
  return true;
}

bool apply_debug_view(Options& options, std::string_view value) {
  const auto selection = render::debug::parse_view(value);
  if (!selection) {
    return false;
  }
  options.debugView = selection->id;
  options.debugViewBit = selection->bit;
  const auto* view = render::debug::find_view(selection->id);
  options.debugViewName = view != nullptr ? std::string{view->name} : std::string{};
  return true;
}

bool apply_classify_json(Options& options, std::string_view) {
  options.classifyJson = true;
  return true;
}

bool apply_classify_objects(Options& options, std::string_view) {
  options.classifyObjects = true;
  options.classifyJson = true;
  return true;
}

bool apply_classify_draws(Options& options, std::string_view) {
  options.classifyDraws = true;
  return true;
}

struct Flag {
  std::string_view name;
  bool takesValue;
  bool (*apply)(Options& options, std::string_view value);
};

constexpr std::array<Flag, 27> Flags{{
    {"--out", true, apply_out},
    {"--backend", true, apply_backend},
    {"--profile", true, apply_profile},
    {"--frames", true, apply_frames},
    {"--repeat", true, apply_repeat},
    {"--no-png", false, apply_no_png},
    {"--gpu-timings", false, apply_gpu_timings},
    {"--cpu-zones", false, apply_cpu_zones},
    {"--cpu-draw-zones", false, apply_cpu_draw_zones},
    {"--replacements", true, apply_replacements},
    {"--data-dir", true, apply_data_dir},
    {"--allow-incomplete", false, apply_allow_incomplete},
    {"--allow-device-mismatch", false, apply_allow_device_mismatch},
    {"--allow-misses", false, apply_allow_misses},
    {"--allow-cpu", false, apply_allow_cpu},
    {"--show-window", false, apply_show_window},
    {"--self-test", false, apply_self_test},
    {"--write-self-test", true, apply_write_self_test},
    {"--info", false, apply_info},
    {"--debug-view", true, apply_debug_view},
    {"--classify-json", false, apply_classify_json},
    {"--classify-objects", false, apply_classify_objects},
    {"--classify-draws", false, apply_classify_draws},
    {"-v", false, apply_verbose},
    {"--verbose", false, apply_verbose},
    {"-h", false, apply_help},
    {"--help", false, apply_help},
}};
} // namespace

std::optional<AuroraBackend> parse_backend(std::string_view name) noexcept {
  for (const auto& [text, backend] : BackendNames) {
    if (text == name) {
      return backend;
    }
  }
  return std::nullopt;
}

std::string_view backend_name(AuroraBackend backend) noexcept {
  for (const auto& [text, value] : BackendNames) {
    if (value == backend) {
      return text;
    }
  }
  return "unknown";
}

std::optional<AuroraFeatureProfile> parse_profile(std::string_view name) noexcept {
  for (const auto& [text, profile] : ProfileNames) {
    if (text == name) {
      return profile;
    }
  }
  return std::nullopt;
}

std::optional<FrameRange> parse_frame_range(std::string_view text) noexcept {
  const auto split = text.find(':');
  if (split == std::string_view::npos) {
    const auto frame = parse_u32(text);
    if (!frame || *frame == UINT32_MAX) {
      return std::nullopt;
    }
    return FrameRange{*frame, *frame + 1};
  }
  const auto first = parse_u32(text.substr(0, split));
  if (!first) {
    return std::nullopt;
  }
  const auto lastText = text.substr(split + 1);
  if (lastText.empty()) {
    return FrameRange{*first, UINT32_MAX};
  }
  const auto last = parse_u32(lastText);
  if (!last || *first >= *last) {
    return std::nullopt;
  }
  return FrameRange{*first, *last};
}

std::optional<Options> parse_options(int argc, const char* const* argv, const char* envBackend, std::string& error) {
  Options options;
  bool backendFlag = false;
  bool haveTrace = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (!arg.starts_with('-')) {
      if (haveTrace) {
        error = std::string{"unexpected argument: "} + std::string{arg};
        return std::nullopt;
      }
      haveTrace = true;
      options.trace = std::filesystem::path{std::u8string{arg.begin(), arg.end()}};
      continue;
    }
    const Flag* flag = nullptr;
    for (const auto& candidate : Flags) {
      if (candidate.name == arg) {
        flag = &candidate;
        break;
      }
    }
    if (flag == nullptr) {
      error = std::string{"unknown option: "} + std::string{arg};
      return std::nullopt;
    }
    std::string_view value;
    if (flag->takesValue) {
      if (i + 1 >= argc) {
        error = std::string{"missing value for "} + std::string{arg};
        return std::nullopt;
      }
      value = argv[++i];
    }
    if (!flag->apply(options, value)) {
      error = flag->name == "--debug-view"
                  ? std::string{"unknown debug view: "} + std::string{value}
                  : std::string{"invalid value for "} + std::string{arg} + ": " + std::string{value};
      return std::nullopt;
    }
    backendFlag |= flag->name == "--backend";
  }
  if (!backendFlag && !options.help && envBackend != nullptr && *envBackend != '\0') {
    const auto backend = parse_backend(envBackend);
    if (!backend) {
      error = std::string{"invalid AURORA_BACKEND value: "} + envBackend;
      return std::nullopt;
    }
    options.backend = *backend;
  }
  if (options.help) {
    return options;
  }

  const int modes = (haveTrace ? 1 : 0) + (options.selfTest ? 1 : 0) + (options.writeSelfTest.empty() ? 0 : 1);
  if (modes != 1) {
    error = "give exactly one of a trace file, --self-test or --write-self-test";
    return std::nullopt;
  }
  if (options.info && !haveTrace) {
    error = "--info needs a trace file";
    return std::nullopt;
  }
  if (options.outDir.empty() && !options.info && options.writeSelfTest.empty()) {
    error = "--out is required";
    return std::nullopt;
  }
  return options;
}

std::string_view usage() noexcept {
  return "usage: aurora_gxreplay <trace> --out DIR [options]\n"
         "       aurora_gxreplay <trace> --info\n"
         "       aurora_gxreplay --self-test --out DIR [options]\n"
         "       aurora_gxreplay --write-self-test PATH\n"
         "  --out DIR                 output directory for PNGs, frames.json and timing.json\n"
         "  --backend NAME            auto, d3d12, vulkan, metal, d3d11, opengl, opengles, webgpu or null\n"
         "                            (defaults to AURORA_BACKEND)\n"
         "  --profile NAME            auto, full, compat, metal or metal32 (defaults to the trace's profile)\n"
         "  --frames N|A:B|A:         frames to output; B is exclusive\n"
         "  --repeat N                replay N times and require identical frames\n"
         "  --no-png                  skip PNG output\n"
         "  --gpu-timings             write gpu_timing.json with per-pass GPU times (needs TimestampQuery)\n"
         "  --cpu-zones               write cpu_zones.json with per-frame FIFO decode time and draw-work counts\n"
         "  --cpu-draw-zones          also time each draw and its rebuild steps, and count unique array bytes\n"
         "  --replacements DIR        load a texture replacement directory\n"
         "  --data-dir DIR            user and cache directory (defaults to <out>/.aurora)\n"
         "  --allow-incomplete        replay the complete frames of an aborted or unfinished trace\n"
         "  --allow-device-mismatch   replay on a device whose features differ from the capture\n"
         "  --allow-misses            do not fail on guest reads the trace cannot resolve\n"
         "  --allow-cpu               allow a CPU adapter\n"
         "  --show-window             show the window while replaying\n"
         "  --self-test               replay the synthetic self-test traces\n"
         "  --write-self-test PATH    write the synthetic self-test trace and exit\n"
         "  --info                    print the trace header, INFO and per-frame counts\n"
         "  --debug-view VIEW         render a classification view (role, kind, main, window, category, object,\n"
         "                            scope, bit[:category] or an id)\n"
         "  --classify-json           write classify.json with per-frame classification counters\n"
         "  --classify-objects        also list each frame's objects in classify.json\n"
         "  --classify-draws          write classify_draws.jsonl with one line per routed draw\n"
         "  -v, --verbose             debug logging\n"
         "  -h, --help                show this message\n";
}
} // namespace aurora::gxreplay

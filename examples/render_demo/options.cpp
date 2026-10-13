#include "options.hpp"

#include <array>
#include <charconv>
#include <utility>

namespace render_demo {
namespace {
constexpr std::array<std::pair<std::string_view, AuroraBackend>, 9> BackendNames{{
    {"auto", BACKEND_AUTO},
    {"d3d11", BACKEND_D3D11},
    {"d3d12", BACKEND_D3D12},
    {"metal", BACKEND_METAL},
    {"vulkan", BACKEND_VULKAN},
    {"opengl", BACKEND_OPENGL},
    {"opengles", BACKEND_OPENGLES},
    {"webgpu", BACKEND_WEBGPU},
    {"null", BACKEND_NULL},
}};

constexpr std::array<std::pair<std::string_view, AuroraFeatureProfile>, 4> ProfileNames{{
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

bool apply_frames(Options& options, std::string_view value) {
  const auto frames = parse_u32(value);
  if (frames) {
    options.frames = *frames;
  }
  return frames.has_value();
}

bool apply_timeout(Options& options, std::string_view value) {
  const auto seconds = parse_u32(value);
  if (seconds) {
    options.timeoutSeconds = *seconds;
  }
  return seconds.has_value();
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

bool apply_size(Options& options, std::string_view value) {
  const auto split = value.find('x');
  if (split == std::string_view::npos) {
    return false;
  }
  const auto width = parse_u32(value.substr(0, split));
  const auto height = parse_u32(value.substr(split + 1));
  if (!width || !height || *width == 0 || *height == 0) {
    return false;
  }
  options.width = *width;
  options.height = *height;
  return true;
}

bool apply_data_dir(Options& options, std::string_view value) {
  if (value.empty()) {
    return false;
  }
  options.dataDir = value;
  return true;
}

bool apply_trace(Options& options, std::string_view value) {
  if (value.empty()) {
    return false;
  }
  options.tracePath = value;
  return true;
}

bool apply_trace_start(Options& options, std::string_view value) {
  const auto start = parse_u32(value);
  if (start) {
    options.traceStart = *start;
  }
  return start.has_value();
}

bool apply_trace_frames(Options& options, std::string_view value) {
  const auto frames = parse_u32(value);
  if (!frames || *frames == 0) {
    return false;
  }
  options.traceFrames = *frames;
  return true;
}

bool apply_fail_on_error(Options& options, std::string_view) {
  options.failOnError = true;
  return true;
}

bool apply_help(Options& options, std::string_view) {
  options.help = true;
  return true;
}

struct Flag {
  std::string_view name;
  bool takesValue;
  bool (*apply)(Options& options, std::string_view value);
};

constexpr std::array<Flag, 12> Flags{{
    {"--frames", true, apply_frames},
    {"--timeout", true, apply_timeout},
    {"--backend", true, apply_backend},
    {"--profile", true, apply_profile},
    {"--size", true, apply_size},
    {"--data-dir", true, apply_data_dir},
    {"--trace", true, apply_trace},
    {"--trace-start", true, apply_trace_start},
    {"--trace-frames", true, apply_trace_frames},
    {"--fail-on-error", false, apply_fail_on_error},
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

std::optional<AuroraFeatureProfile> parse_profile(std::string_view name) noexcept {
  for (const auto& [text, profile] : ProfileNames) {
    if (text == name) {
      return profile;
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

std::optional<Options> parse_options(int argc, const char* const* argv, const char* envBackend, std::string& error) {
  Options options;
  if (envBackend != nullptr && *envBackend != '\0') {
    const auto backend = parse_backend(envBackend);
    if (!backend) {
      error = std::string{"invalid AURORA_BACKEND value: "} + envBackend;
      return std::nullopt;
    }
    options.backend = *backend;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
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
      error = std::string{"invalid value for "} + std::string{arg} + ": " + std::string{value};
      return std::nullopt;
    }
  }
  if (!options.tracePath.empty() && options.frames != 0 &&
      options.frames < static_cast<uint64_t>(options.traceStart) + options.traceFrames) {
    error = "--frames must cover the trace window";
    return std::nullopt;
  }
  return options;
}

std::string_view usage() noexcept {
  return "usage: render_demo [options]\n"
         "  --frames N          exit after N frames (0 runs until the window closes)\n"
         "  --timeout S         fail with exit code 2 if N frames take longer than S seconds\n"
         "  --backend NAME      auto, d3d11, d3d12, metal, vulkan, opengl, opengles, webgpu or null\n"
         "                      (defaults to AURORA_BACKEND)\n"
         "  --profile NAME      full, compat, metal or metal32 (defaults to AURORA_FEATURE_PROFILE)\n"
         "  --size WxH          window size\n"
         "  --data-dir PATH     user and cache directory\n"
         "  --trace PATH        capture a GX trace to PATH\n"
         "  --trace-start N     first frame to capture (default 2)\n"
         "  --trace-frames N    number of frames to capture (default 4)\n"
         "  --fail-on-error     exit with code 3 if any error is logged\n"
         "  -h, --help          show this message\n";
}
} // namespace render_demo

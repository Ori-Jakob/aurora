#include "options.hpp"

#include "cull_reference.hpp"

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

constexpr uint32_t DepthCheckMinFrames = 40;
constexpr uint32_t MotionCheckMinFrames = 40;
constexpr uint32_t MaxGpuPadMiB = 1024;

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

bool apply_vignette(Options& options, std::string_view) {
  options.vignette = true;
  return true;
}

bool apply_vignette_anchor(Options& options, std::string_view value) {
  if (value.empty()) {
    return false;
  }
  options.vignette = true;
  options.vignetteAnchor = value;
  return true;
}

bool apply_expect_anchor_warnings(Options& options, std::string_view value) {
  const auto count = parse_u32(value);
  if (count) {
    options.expectAnchorWarnings = *count;
    options.expectAnchorWarningsSet = true;
  }
  return count.has_value();
}

bool apply_msaa(Options& options, std::string_view value) {
  // WebGPU allows only 1 and 4 samples
  const auto samples = parse_u32(value);
  if (!samples || (*samples != 1 && *samples != 4)) {
    return false;
  }
  options.msaa = *samples;
  return true;
}

bool apply_depth_check(Options& options, std::string_view) {
  options.depthCheck = true;
  return true;
}

bool apply_motion_check(Options& options, std::string_view) {
  options.motionCheck = true;
  return true;
}

bool apply_motion_view(Options& options, std::string_view) {
  options.motionView = true;
  return true;
}

bool apply_no_hints(Options& options, std::string_view) {
  options.hints = false;
  return true;
}

bool apply_check_hints(Options& options, std::string_view) {
  options.checkHints = true;
  return true;
}

bool apply_custom_draw_probe(Options& options, std::string_view) {
  options.customDrawProbe = true;
  return true;
}

bool apply_gpu_cull(Options& options, std::string_view value) {
  const auto instances = parse_u32(value);
  if (!instances || *instances == 0 || *instances > MaxCullInstances) {
    return false;
  }
  options.gpuCull = *instances;
  return true;
}

bool apply_cull_check(Options& options, std::string_view) {
  options.cullCheck = true;
  return true;
}

bool apply_cull_timings(Options& options, std::string_view) {
  options.cullTimings = true;
  return true;
}

bool apply_bench_indirect(Options& options, std::string_view value) {
  if (value.empty()) {
    return false;
  }
  options.benchPath = value;
  return true;
}

bool apply_bench_blocks(Options& options, std::string_view value) {
  const auto blocks = parse_u32(value);
  if (!blocks || *blocks == 0) {
    return false;
  }
  options.benchBlocks = *blocks;
  options.benchBlocksSet = true;
  return true;
}

bool apply_bench_rebind(Options& options, std::string_view value) {
  const auto split = value.find(',');
  if (split == std::string_view::npos) {
    return false;
  }
  const auto uniform = parse_u32(value.substr(0, split));
  const auto texture = parse_u32(value.substr(split + 1));
  if (!uniform || !texture || *uniform > 1000 || *texture > 1000) {
    return false;
  }
  options.benchUniformPermille = *uniform;
  options.benchTexturePermille = *texture;
  options.benchRebindSet = true;
  return true;
}

bool apply_gpu_arrays(Options& options, std::string_view) {
  options.gpuArrays = true;
  return true;
}

bool apply_gpu_pad_mib(Options& options, std::string_view value) {
  const auto mib = parse_u32(value);
  if (!mib || *mib == 0 || *mib > MaxGpuPadMiB) {
    return false;
  }
  options.gpuPadMiB = *mib;
  return true;
}

struct Flag {
  std::string_view name;
  bool takesValue;
  bool (*apply)(Options& options, std::string_view value);
  // The bench runs before GXInit and never enters the frame loop, so these flags would have nothing to act on
  bool frameLoop = false;
};

constexpr std::array<Flag, 30> Flags{{
    {"--frames", true, apply_frames, true},
    {"--timeout", true, apply_timeout, true},
    {"--backend", true, apply_backend},
    {"--profile", true, apply_profile},
    {"--size", true, apply_size},
    {"--data-dir", true, apply_data_dir},
    {"--trace", true, apply_trace, true},
    {"--trace-start", true, apply_trace_start, true},
    {"--trace-frames", true, apply_trace_frames, true},
    {"--fail-on-error", false, apply_fail_on_error},
    {"--vignette", false, apply_vignette, true},
    {"--vignette-anchor", true, apply_vignette_anchor, true},
    {"--expect-anchor-warnings", true, apply_expect_anchor_warnings, true},
    {"--msaa", true, apply_msaa},
    {"--depth-check", false, apply_depth_check, true},
    {"--motion-check", false, apply_motion_check, true},
    {"--motion-view", false, apply_motion_view, true},
    {"--no-hints", false, apply_no_hints, true},
    {"--check-hints", false, apply_check_hints, true},
    {"--custom-draw-probe", false, apply_custom_draw_probe, true},
    {"--gpu-cull", true, apply_gpu_cull, true},
    {"--cull-check", false, apply_cull_check, true},
    {"--cull-timings", false, apply_cull_timings, true},
    {"--bench-indirect", true, apply_bench_indirect},
    {"--bench-blocks", true, apply_bench_blocks},
    {"--bench-rebind", true, apply_bench_rebind},
    {"--gpu-arrays", false, apply_gpu_arrays, true},
    {"--gpu-pad-mib", true, apply_gpu_pad_mib, true},
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
  bool backendFlag = false;
  std::string_view frameLoopFlag;
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
    backendFlag |= flag->name == "--backend";
    if (flag->frameLoop && frameLoopFlag.empty()) {
      frameLoopFlag = flag->name;
    }
  }
  if (!backendFlag && !options.help && envBackend != nullptr && *envBackend != '\0') {
    const auto backend = parse_backend(envBackend);
    if (!backend) {
      error = std::string{"invalid AURORA_BACKEND value: "} + envBackend;
      return std::nullopt;
    }
    options.backend = *backend;
  }
  if (!options.benchPath.empty() && !frameLoopFlag.empty()) {
    error = "--bench-indirect cannot be combined with " + std::string{frameLoopFlag};
    return std::nullopt;
  }
  if (!options.tracePath.empty() && options.frames != 0 &&
      options.frames < static_cast<uint64_t>(options.traceStart) + options.traceFrames) {
    error = "--frames must cover the trace window";
    return std::nullopt;
  }
  if (options.expectAnchorWarningsSet && !options.vignette) {
    error = "--expect-anchor-warnings needs --vignette";
    return std::nullopt;
  }
  // Graph passes and custom draws record GPU work outside the GX FIFO, so the capture would be discarded
  if ((options.vignette || options.depthCheck || options.motionCheck || options.motionView || options.gpuCull != 0) &&
      !options.tracePath.empty()) {
    error = "render graph passes cannot be traced";
    return std::nullopt;
  }
  if (options.customDrawProbe && !options.tracePath.empty()) {
    error = "custom draws cannot be traced";
    return std::nullopt;
  }
  // The probe reads frame 30 at the earliest, and the rest leave room for its readback to land
  if (options.depthCheck && options.frames != 0 && options.frames < DepthCheckMinFrames) {
    error = "--depth-check needs at least 40 frames";
    return std::nullopt;
  }
  if (options.motionCheck && options.frames != 0 && options.frames < MotionCheckMinFrames) {
    error = "--motion-check needs at least 40 frames";
    return std::nullopt;
  }
  if ((options.cullCheck || options.cullTimings) && options.gpuCull == 0) {
    error = options.cullCheck ? "--cull-check needs --gpu-cull" : "--cull-timings needs --gpu-cull";
    return std::nullopt;
  }
  if (options.cullCheck && options.frames != 0 && options.frames < CullCheckMinFrames) {
    error = "--cull-check needs at least 40 frames";
    return std::nullopt;
  }
  if ((options.benchBlocksSet || options.benchRebindSet) && options.benchPath.empty()) {
    error = options.benchBlocksSet ? "--bench-blocks needs --bench-indirect" : "--bench-rebind needs --bench-indirect";
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
         "  --vignette          register the demo.vignette render graph pass\n"
         "  --vignette-anchor NAME\n"
         "                      anchor demo.vignette after NAME (implies --vignette)\n"
         "  --expect-anchor-warnings N\n"
         "                      exit with code 7 unless N unknown-anchor warnings are logged (default 0)\n"
         "  --msaa N            EFB sample count, 1 or 4 (default 1)\n"
         "  --depth-check       register demo.depth_probe and exit with code 7 if scene depth reads wrong\n"
         "                      (needs at least 40 frames)\n"
         "  --motion-check      register demo.motion_probe and exit with code 11 unless gbuf.motion and\n"
         "                      gbuf.reactive match the scene's own motion (needs at least 40 frames)\n"
         "  --motion-view       show gbuf.motion in the scene colour before the HUD\n"
         "  --no-hints          emit no render hints and no capture-only draw\n"
         "  --check-hints       exit with code 8 unless aurora decoded or skipped every emitted hint as expected\n"
         "  --custom-draw-probe record a custom draw that leaves foreign bindings behind between GX draws\n"
         "  --gpu-cull N        frustum-cull N instances (1 to 4194304) in demo.cull and draw them indirectly\n"
         "  --cull-check        exit with code 9 unless the frame-30 cull output matches the CPU reference\n"
         "                      (needs --gpu-cull and at least 40 frames)\n"
         "  --cull-timings      report GPU times of the cull passes from frame 60 (turns the profiler on)\n"
         "  --bench-indirect PATH\n"
         "                      measure indirect draw overhead before the frame loop, write JSON to PATH and exit\n"
         "                      (exit code 10 on failure; combines only with device, size and bench options)\n"
         "  --bench-blocks N    measured ABBA blocks per bench pair (default 150)\n"
         "  --bench-rebind U,T  uniform and texture rebind rates in permille (default 1000,500)\n"
         "  --gpu-arrays        upload the box arrays through the GPU channel each frame and bind them with\n"
         "                      GXSetArrayGPU, and run a no-op GPU system\n"
         "  --gpu-pad-mib N     upload N MiB (1 to 1024) of zeros through the GPU channel each frame\n"
         "  -h, --help          show this message\n";
}
} // namespace render_demo

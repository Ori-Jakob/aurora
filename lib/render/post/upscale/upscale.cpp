#include "upscale.hpp"

#include "compose.hpp"
#include "native_upscale.hpp"
#include "fsr/fsr.hpp"
#include "fsr/fsr_pass.hpp"
#include "../taa.hpp"
#include "../temporal_resolve.hpp"
#include "../../module.hpp"
#include "../../native/native.hpp"

#include "../../../gfx/frame.hpp"
#include "../../../gx/gx.hpp"
#include "../../../logging.hpp"
#include "../../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <utility>

namespace aurora::render::detail::upscale {
namespace {
constexpr Module Log{"aurora::render::upscale"};

template <typename T>
struct EnvOverride {
  bool set = false;
  std::optional<T> value;
};

template <UpscalerQuality Quality>
GraphExtent fsr_scene_size(GraphExtent output) noexcept {
  const GraphExtent size = fsr::render_size(output, Quality);
  return {std::min(size.width, output.width), std::min(size.height, output.height)};
}

bool fsr_wanted(const Settings& settings) noexcept { return settings.upscaler == Upscaler::Fsr; }

template <UpscalerQuality Quality>
const temporal_resolve::Resolver FsrResolver{
    .pass = fsr_pass::Pass,
    .jitterPhases = fsr::jitter_phases(fsr::mode_ratio(Quality)),
    .wanted = &fsr_wanted,
    .jitter_ready = &fsr_pass::jitter_ready,
    .scene_size = &fsr_scene_size<Quality>,
};

const temporal_resolve::Resolver* fsr_resolver(UpscalerQuality quality) noexcept {
  switch (quality) {
  case UpscalerQuality::NativeAa:
    return &FsrResolver<UpscalerQuality::NativeAa>;
  case UpscalerQuality::Quality:
    return &FsrResolver<UpscalerQuality::Quality>;
  case UpscalerQuality::Balanced:
    return &FsrResolver<UpscalerQuality::Balanced>;
  case UpscalerQuality::Performance:
    return &FsrResolver<UpscalerQuality::Performance>;
  case UpscalerQuality::UltraPerformance:
    return &FsrResolver<UpscalerQuality::UltraPerformance>;
  }
  return nullptr;
}

// Game thread
EnvOverride<float> g_envOverride;
EnvOverride<float> g_mipBiasOverride;
EnvOverride<Signs> g_signsOverride;
EnvOverride<ColorEncoding> g_colorOverride;
const temporal_resolve::Resolver* g_installed = nullptr;
const CompiledGraph* g_checkedGraph = nullptr;
bool g_warnedLateWriter = false;
bool g_warnedLinearFsr = false;

std::atomic_bool g_splitLive{false};
std::atomic_bool g_nativeInPlace{false};
std::atomic<Upscaler> g_active{Upscaler::Off};
std::atomic<UpscalerQuality> g_quality{UpscalerQuality::Quality};
std::atomic<float> g_rcasStrength{0.f};
std::atomic_uint32_t g_sceneWidth{0};
std::atomic_uint32_t g_sceneHeight{0};
std::atomic_uint32_t g_outputWidth{0};
std::atomic_uint32_t g_outputHeight{0};

std::optional<float> split_env() {
  if (g_envOverride.set) {
    return g_envOverride.value;
  }
  static const std::optional<float> env = []() -> std::optional<float> {
    const char* value = SDL_getenv("AURORA_RENDER_SPLIT");
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::optional<float> ratio = parse_split_ratio(value);
    if (ratio) {
      Log.warn("AURORA_RENDER_SPLIT={}: rendering the scene at 1/{} of the output size", value, *ratio);
    } else {
      Log.warn("AURORA_RENDER_SPLIT={} is not a ratio above 1 and at most {}; ignored", value, MaxSplitRatio);
    }
    return ratio;
  }();
  return env;
}

float mip_bias_env() {
  if (g_mipBiasOverride.set) {
    return g_mipBiasOverride.value.value_or(0.f);
  }
  static const float env = []() -> float {
    const char* value = SDL_getenv("AURORA_RENDER_UPSCALER_MIP_BIAS");
    if (value == nullptr) {
      return 0.f;
    }
    const std::optional<float> extra = parse_mip_bias(value);
    if (extra) {
      Log.warn("AURORA_RENDER_UPSCALER_MIP_BIAS={}: adding {} to the scene mip bias while upscaling", value, *extra);
    } else {
      Log.warn("AURORA_RENDER_UPSCALER_MIP_BIAS={} is not a bias between -{} and {}; ignored", value, MaxExtraMipBias,
               MaxExtraMipBias);
    }
    return extra.value_or(0.f);
  }();
  return env;
}

Signs signs_env() {
  if (g_signsOverride.set) {
    return g_signsOverride.value.value_or(Signs{});
  }
  static const Signs env = []() -> Signs {
    const char* value = SDL_getenv("AURORA_RENDER_UPSCALER_SIGNS");
    if (value == nullptr) {
      return {};
    }
    const std::optional<Signs> signs = parse_signs(value);
    if (signs) {
      Log.warn("AURORA_RENDER_UPSCALER_SIGNS={}: flipping the upscaler's jitter and motion inputs", value);
    } else {
      Log.warn("AURORA_RENDER_UPSCALER_SIGNS={} is not four comma-separated 1 or -1 values; ignored", value);
    }
    return signs.value_or(Signs{});
  }();
  return env;
}

std::optional<ColorEncoding> color_env() {
  if (g_colorOverride.set) {
    return g_colorOverride.value;
  }
  static const std::optional<ColorEncoding> env = []() -> std::optional<ColorEncoding> {
    const char* value = SDL_getenv("AURORA_RENDER_UPSCALER_COLOR");
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::optional<ColorEncoding> color = parse_color(value);
    if (color) {
      Log.warn("AURORA_RENDER_UPSCALER_COLOR={}: overriding the upscaler's colour input", value);
    } else {
      Log.warn("AURORA_RENDER_UPSCALER_COLOR={} is not stored or linear; ignored", value);
    }
    return color;
  }();
  return env;
}

void publish_sampling() {
  if (!compose_wanted()) {
    gx::g_upscaleSampling = {};
    return;
  }
  const float outputPerRender = ratio();
  const temporal_resolve::Resolver& resolver = temporal_resolve::installed();
  const float bias =
      resolver.mip_bias != nullptr ? resolver.mip_bias(outputPerRender) : std::log2(1.f / outputPerRender);
  gx::g_upscaleSampling = {.active = true, .outputPerRender = outputPerRender, .mipBias = bias + mip_bias_env()};
}

void publish() noexcept {
  const wgpu::Extent3D scene = webgpu::scene_extent();
  const wgpu::Extent3D output = webgpu::g_frameBuffer.size;
  g_sceneWidth.store(scene.width, std::memory_order_relaxed);
  g_sceneHeight.store(scene.height, std::memory_order_relaxed);
  g_outputWidth.store(output.width, std::memory_order_relaxed);
  g_outputHeight.store(output.height, std::memory_order_relaxed);
  const temporal_resolve::Resolver& installed = temporal_resolve::installed();
  const bool fsr = installed.pass == fsr_pass::Pass;
  const bool native = native_upscale::is_native(installed);
  g_splitLive.store(webgpu::scene_split_active(), std::memory_order_release);
  g_nativeInPlace.store(native && !webgpu::scene_split_active(), std::memory_order_release);
  g_active.store(fsr ? Upscaler::Fsr : (native ? native_upscale::active() : Upscaler::Off), std::memory_order_relaxed);
  publish_sampling();
}

bool writes_color(const CompiledGraph& graph, NodeId node) noexcept {
  return std::ranges::any_of(graph.node_uses(node), [&graph](const CompiledUse& use) {
    return use.resource < graph.resources.size() && graph.resources[use.resource].name == resource::GxColor &&
           (use.access == Access::ColorWrite || use.access == Access::ColorRMW);
  });
}
} // namespace

std::optional<float> parse_mip_bias(std::string_view text) noexcept {
  float value = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
      std::abs(value) > MaxExtraMipBias) {
    return std::nullopt;
  }
  return value;
}

std::optional<float> parse_split_ratio(std::string_view text) noexcept {
  float value = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) || value <= 1.f ||
      value > MaxSplitRatio) {
    return std::nullopt;
  }
  return value;
}

std::optional<Signs> parse_signs(std::string_view text) noexcept {
  std::array<float, 4> values{};
  for (size_t i = 0; i < values.size(); ++i) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    if (item == "1" || item == "+1") {
      values[i] = 1.f;
    } else if (item == "-1") {
      values[i] = -1.f;
    } else {
      return std::nullopt;
    }
    if ((comma == std::string_view::npos) != (i + 1 == values.size())) {
      return std::nullopt;
    }
    text.remove_prefix(comma == std::string_view::npos ? text.size() : comma + 1);
  }
  return Signs{.jitter = {values[0], values[1]}, .motion = {values[2], values[3]}};
}

std::optional<ColorEncoding> parse_color(std::string_view text) noexcept {
  if (text == "stored") {
    return ColorEncoding::Stored;
  }
  if (text == "linear") {
    return ColorEncoding::Linear;
  }
  return std::nullopt;
}

GraphExtent split_size(GraphExtent output, float ratio) noexcept {
  if (!(ratio > 0.f)) {
    return output;
  }
  const auto axis = [ratio](uint32_t size) {
    return std::max(1u, static_cast<uint32_t>(std::floor(static_cast<double>(size) / static_cast<double>(ratio))));
  };
  return {axis(output.width), axis(output.height)};
}

GraphExtent wanted_scene_size(const SplitInputs& inputs) noexcept {
  if (!inputs.moduleEnabled || !inputs.sceneTargets || !(inputs.ratio > 1.f) || inputs.output.width == 0 ||
      inputs.output.height == 0) {
    return {};
  }
  const GraphExtent size = split_size(inputs.output, inputs.ratio);
  return size == inputs.output ? GraphExtent{} : size;
}

void initialize() {
  fsr_pass::initialize();
  native_upscale::initialize();
}

bool on_latch(const Settings& settings) {
  const wgpu::Extent3D efb = webgpu::g_frameBuffer.size;
  const GraphExtent output{efb.width, efb.height};
  const bool sceneTargets = webgpu::scene_targets_supported();
  const bool usable = module_enabled() && sceneTargets && output.width != 0 && output.height != 0;
  const bool fsr = settings.upscaler == Upscaler::Fsr && usable;
  native_upscale::NativeUpscaler* vendor = nullptr;
  if (usable && (settings.upscaler == Upscaler::Dlss || settings.upscaler == Upscaler::Xess)) {
    vendor = native::upscaler(settings.upscaler);
  } else if (usable && settings.upscaler == Upscaler::Off) {
    vendor = native_upscale::stand_in(split_env().value_or(1.f));
  }
  const temporal_resolve::Resolver* resolver = fsr ? fsr_resolver(settings.upscalerQuality) : nullptr;
  ColorEncoding encoding = ColorEncoding::Stored;
  GraphExtent wanted;
  if (resolver != nullptr) {
    const GraphExtent size = resolver->scene_size(output);
    wanted = size == output ? GraphExtent{} : size;
  } else if (vendor != nullptr) {
    const std::optional<native_upscale::Selection> selection =
        native_upscale::select(*vendor, settings.upscalerQuality, output);
    if (!selection) {
      g_quality.store(settings.upscalerQuality, std::memory_order_relaxed);
      g_rcasStrength.store(taa::rcas_strength(settings.taaSharpness), std::memory_order_relaxed);
      publish();
      return false;
    }
    resolver = selection->resolver;
    wanted = selection->scene == output ? GraphExtent{} : selection->scene;
    encoding = selection->encoding;
  } else {
    wanted = wanted_scene_size({
        .output = output,
        .ratio = split_env().value_or(0.f),
        .sceneTargets = sceneTargets,
        .moduleEnabled = module_enabled(),
    });
  }
  const wgpu::Extent3D scene = webgpu::scene_extent();
  const GraphExtent current = webgpu::scene_split_active() ? GraphExtent{scene.width, scene.height} : GraphExtent{};
  const bool resolverChanged = resolver != g_installed;
  if (fsr && color_env() == ColorEncoding::Linear && !std::exchange(g_warnedLinearFsr, true)) {
    Log.warn("AURORA_RENDER_UPSCALER_COLOR=linear: FSR keeps stored colour");
  }
  bool switched = false;
  if (wanted != current || resolverChanged) {
    const wgpu::TextureFormat colorFormat = webgpu::g_frameBuffer.format;
    const wgpu::TextureFormat depthFormat = webgpu::g_graphicsConfig.depthFormat;
    bool warm = wanted.width == 0 || compose::prepare_pipelines(ComposeMode::Split, colorFormat, depthFormat);
    if (fsr) {
      warm = fsr_pass::prepare(wanted.width == 0, colorFormat) && warm;
    } else if (vendor != nullptr) {
      const ComposeMode mode = wanted.width == 0 ? ComposeMode::InPlace : ComposeMode::Split;
      warm = compose::prepare_pipelines(mode, colorFormat, depthFormat) &&
             compose::prepare_pipelines(mode, colorFormat, depthFormat, encoding) &&
             native_upscale::prepare_pipelines(encoding) && warm;
    }
    if (warm) {
      gfx::gpu_synchronize();
      if (wanted != current) {
        webgpu::set_scene_render_size(wanted.width, wanted.height);
      }
      temporal_resolve::install(resolver);
      g_installed = resolver;
      native_upscale::installed(resolver);
      switched = true;
      std::string_view name;
      if (fsr) {
        name = "FSR";
      } else if (vendor != nullptr) {
        name = vendor->name();
      }
      const std::string_view separator = name.empty() ? "" : ", ";
      if (webgpu::scene_split_active()) {
        Log.info("Scene split: {}x{} scene, {}x{} output{}{}", wanted.width, wanted.height, efb.width, efb.height,
                 separator, name);
      } else {
        Log.info("Scene split: off{}{}{}", separator, name, name.empty() ? "" : " in place");
      }
    }
  }
  if (vendor == nullptr && !native_upscale::is_native(temporal_resolve::installed())) {
    native_upscale::release();
  }
  g_quality.store(settings.upscalerQuality, std::memory_order_relaxed);
  g_rcasStrength.store(taa::rcas_strength(settings.taaSharpness), std::memory_order_relaxed);
  publish();
  return switched;
}

void latch_graph(const Settings& settings, const CompiledGraph* graph) {
  fsr_pass::on_latch(settings, graph, signs_env());
  native_upscale::on_latch(settings, graph, signs_env());
  if (graph == g_checkedGraph) {
    return;
  }
  g_checkedGraph = graph;
  const temporal_resolve::Resolver& resolver = temporal_resolve::installed();
  const bool composes = native_upscale::is_native(resolver) || (split_live() && resolver.scene_size != nullptr);
  if (graph == nullptr || !composes || g_warnedLateWriter) {
    return;
  }
  const std::span<const NodeId> passes = graph->stages[static_cast<size_t>(Stage::PrePost)].passes;
  const auto at = std::ranges::find_if(passes, [&](NodeId node) { return graph->nodes[node].name == resolver.pass; });
  if (at == passes.end()) {
    return;
  }
  for (auto it = std::next(at); it != passes.end(); ++it) {
    if (graph->nodes[*it].status == CullReason::None && writes_color(*graph, *it)) {
      g_warnedLateWriter = true;
      Log.warn("Pass {} writes gx.color after {}, but the upscaled output replaces it", graph->nodes[*it].name,
               resolver.pass);
      return;
    }
  }
}

void shutdown() noexcept {
  fsr_pass::shutdown();
  native_upscale::shutdown();
  compose::shutdown();
  temporal_resolve::install(nullptr);
  g_installed = nullptr;
  gx::g_upscaleSampling = {};
  g_checkedGraph = nullptr;
  g_splitLive.store(false, std::memory_order_release);
  g_nativeInPlace.store(false, std::memory_order_release);
  g_active.store(Upscaler::Off, std::memory_order_relaxed);
  g_sceneWidth.store(0, std::memory_order_relaxed);
  g_sceneHeight.store(0, std::memory_order_relaxed);
  g_outputWidth.store(0, std::memory_order_relaxed);
  g_outputHeight.store(0, std::memory_order_relaxed);
}

bool split_live() noexcept { return g_splitLive.load(std::memory_order_acquire); }

bool compose_wanted() noexcept { return split_live() || g_nativeInPlace.load(std::memory_order_acquire); }

GraphExtent scene_size() noexcept {
  return {g_sceneWidth.load(std::memory_order_relaxed), g_sceneHeight.load(std::memory_order_relaxed)};
}

GraphExtent output_size() noexcept {
  return {g_outputWidth.load(std::memory_order_relaxed), g_outputHeight.load(std::memory_order_relaxed)};
}

float ratio() noexcept {
  const GraphExtent scene = scene_size();
  const GraphExtent output = output_size();
  return scene.width != 0 ? static_cast<float>(output.width) / static_cast<float>(scene.width) : 1.f;
}

float rcas_strength() noexcept { return g_rcasStrength.load(std::memory_order_relaxed); }

bool ready(const Settings& settings) noexcept {
  const temporal_resolve::Resolver& installed = temporal_resolve::installed();
  switch (settings.upscaler) {
  case Upscaler::Fsr:
    return installed.pass == fsr_pass::Pass && fsr_pass::ready();
  case Upscaler::Dlss:
  case Upscaler::Xess:
    return native_upscale::is_native(installed) && native_upscale::active() == settings.upscaler &&
           native_upscale::ready();
  case Upscaler::Off:
  case Upscaler::Auto:
    break;
  }
  return true;
}

UpscaleStatus status() noexcept {
  const GraphExtent scene = scene_size();
  const GraphExtent output = output_size();
  const Upscaler active = g_active.load(std::memory_order_relaxed);
  const UpscalerQuality quality = g_quality.load(std::memory_order_relaxed);
  const bool fsr = active == Upscaler::Fsr;
  const bool native = active == Upscaler::Dlss || active == Upscaler::Xess;
  std::string_view displayName;
  if (fsr) {
    displayName = "FSR";
  } else if (native) {
    displayName = native_upscale::display_name(quality);
  }
  return {
      .active = active,
      .quality = quality,
      .sceneSize = {scene.width, scene.height},
      .outputSize = {output.width, output.height},
      .displayName = displayName,
      .path = native ? native_upscale::path() : std::string_view{},
      .ready = fsr ? fsr_pass::ready() : native && native_upscale::ready(),
  };
}

namespace testing {
void set_split_env(std::optional<float> ratio) noexcept { g_envOverride = {.set = true, .value = ratio}; }

void set_mip_bias_env(std::optional<float> extra) noexcept { g_mipBiasOverride = {.set = true, .value = extra}; }

void set_signs_env(std::optional<Signs> signs) noexcept { g_signsOverride = {.set = true, .value = signs}; }

void set_color_env(std::optional<ColorEncoding> color) noexcept { g_colorOverride = {.set = true, .value = color}; }

void reset() noexcept {
  g_envOverride = {};
  g_mipBiasOverride = {};
  g_signsOverride = {};
  g_colorOverride = {};
  g_warnedLateWriter = false;
  g_warnedLinearFsr = false;
  g_checkedGraph = nullptr;
  g_rcasStrength.store(0.f, std::memory_order_relaxed);
  compose::testing::reset();
  fsr_pass::testing::reset();
  native_upscale::testing::reset();
  publish();
}
} // namespace testing
} // namespace aurora::render::detail::upscale

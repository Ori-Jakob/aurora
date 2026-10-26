#include "upscale.hpp"

#include "compose.hpp"
#include "../temporal_resolve.hpp"
#include "../../module.hpp"

#include "../../../gfx/frame.hpp"
#include "../../../gx/gx.hpp"
#include "../../../logging.hpp"
#include "../../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <utility>

namespace aurora::render::detail::upscale {
namespace {
constexpr Module Log{"aurora::render::upscale"};

struct EnvOverride {
  bool set = false;
  std::optional<float> value;
};

// Game thread
EnvOverride g_envOverride;
EnvOverride g_mipBiasOverride;
const CompiledGraph* g_checkedGraph = nullptr;
bool g_warnedLateWriter = false;

std::atomic_bool g_splitLive{false};
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
  g_splitLive.store(webgpu::scene_split_active(), std::memory_order_release);
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

bool on_latch(const Settings&) {
  const wgpu::Extent3D efb = webgpu::g_frameBuffer.size;
  const SplitInputs inputs{
      .output = {efb.width, efb.height},
      .ratio = split_env().value_or(0.f),
      .sceneTargets = webgpu::scene_targets_supported(),
      .moduleEnabled = module_enabled(),
  };
  const GraphExtent wanted = wanted_scene_size(inputs);
  const wgpu::Extent3D scene = webgpu::scene_extent();
  const GraphExtent current = webgpu::scene_split_active() ? GraphExtent{scene.width, scene.height} : GraphExtent{};
  bool switched = false;
  if (wanted != current) {
    const bool warm = wanted.width == 0 || compose::prepare_pipelines(ComposeMode::Split, webgpu::g_frameBuffer.format,
                                                                      webgpu::g_graphicsConfig.depthFormat);
    if (warm) {
      gfx::gpu_synchronize();
      webgpu::set_scene_render_size(wanted.width, wanted.height);
      switched = true;
      if (webgpu::scene_split_active()) {
        Log.info("Scene split: {}x{} scene, {}x{} output", wanted.width, wanted.height, efb.width, efb.height);
      } else {
        Log.info("Scene split: off");
      }
    }
  }
  publish();
  return switched;
}

void check_graph(const CompiledGraph* graph) {
  if (graph == g_checkedGraph) {
    return;
  }
  g_checkedGraph = graph;
  const temporal_resolve::Resolver& resolver = temporal_resolve::installed();
  if (graph == nullptr || !split_live() || resolver.scene_size == nullptr || g_warnedLateWriter) {
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
  compose::shutdown();
  gx::g_upscaleSampling = {};
  g_checkedGraph = nullptr;
  g_splitLive.store(false, std::memory_order_release);
  g_sceneWidth.store(0, std::memory_order_relaxed);
  g_sceneHeight.store(0, std::memory_order_relaxed);
  g_outputWidth.store(0, std::memory_order_relaxed);
  g_outputHeight.store(0, std::memory_order_relaxed);
}

bool split_live() noexcept { return g_splitLive.load(std::memory_order_acquire); }

bool compose_wanted() noexcept { return split_live(); }

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

UpscaleStatus status() noexcept {
  const GraphExtent scene = scene_size();
  const GraphExtent output = output_size();
  return {
      .active = Upscaler::Off,
      .sceneSize = {scene.width, scene.height},
      .outputSize = {output.width, output.height},
  };
}

namespace testing {
void set_split_env(std::optional<float> ratio) noexcept { g_envOverride = {.set = true, .value = ratio}; }

void set_mip_bias_env(std::optional<float> extra) noexcept { g_mipBiasOverride = {.set = true, .value = extra}; }

void reset() noexcept {
  g_envOverride = {};
  g_mipBiasOverride = {};
  g_warnedLateWriter = false;
  g_checkedGraph = nullptr;
  compose::testing::reset();
  publish();
}
} // namespace testing
} // namespace aurora::render::detail::upscale

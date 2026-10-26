#include "module.hpp"

#include "frame_state.hpp"
#include "gpu_heap.hpp"
#include "graph_api.hpp"
#include "graph_bind.hpp"
#include "graph_pool.hpp"
#include "hints.hpp"
#include "jobs.hpp"
#include "settings.hpp"
#include "shaders.hpp"
#include "debug/debug.hpp"
#include "debug/views.hpp"
#include "geometry/compact.hpp"
#include "geometry/hzb.hpp"
#include "gx_ext/ext_provider.hpp"
#include "material/gx_lookup.hpp"
#include "post/taa.hpp"
#include "scene/motion.hpp"
#include "systems/dispatch.hpp"
#include "systems/systems.hpp"

#include "../gfx/scene_targets.hpp"
#include "../gx/command_processor.hpp"
#include "../logging.hpp"
#include "../webgpu/gpu.hpp"

#if !AURORA_ENABLE_RENDER
#include <imgui.h>
#endif

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <array>

namespace aurora::render {
namespace {
constexpr Module Log{"aurora::render"};

constexpr std::array<std::string_view, 22> ReasonNames{
    "None",
    "NeedsCore",
    "ColorBudget",
    "NoFragmentStorage",
    "NoVertexStorage",
    "NoAtomic64",
    "NoMdi",
    "NeedsD3D12",
    "NativeUnavailable",
    "NativeDisabledAfterCrash",
    "MsaaActive",
    "NeedsRetainedScene",
    "NeedsCompanions",
    "NeedsSunShadows",
    "BindingLimit",
    "LowVram",
    "BuildDisabled",
    "NotImplemented",
    "NeedsPbr",
    "RequiredByShadows",
    "NeedsAo",
    "NeedsRt",
};
static_assert(ReasonNames.size() == static_cast<size_t>(Reason::NeedsRt) + 1);

std::string_view yes_no(bool value) noexcept { return value ? "yes" : "no"; }
} // namespace

namespace detail {
bool module_enabled() noexcept { return AURORA_ENABLE_RENDER != 0; }

void initialize() {
  Log.info("Render module: {} (API {})", module_enabled() ? "enabled" : "disabled at build time", ApiVersion);
  const uint32_t msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  reset_settings_state(msaaSamples);
  const Caps& deviceCaps = caps();
  const Tier tier = device_tier(deviceCaps, msaaSamples);
  Log.info("Render tiers: base {}, PBR {} (core {}, msaa {}, colour budget {} B)", yes_no(tier != Tier::None),
           yes_no(tier == Tier::Pbr), yes_no(deviceCaps.core), msaaSamples,
           deviceCaps.limits.maxColorAttachmentBytesPerSample);
#if AURORA_ENABLE_RENDER
  transient_pool().set_release_listener(&bind_cache_on_release);
  register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
  jobs::initialize(webgpu::g_backendType == wgpu::BackendType::WebGPU ? 0 : jobs::default_worker_count());
  shaders::initialize();
  compact::initialize();
  hzb::initialize();
  taa::initialize();
  systems::initialize();
  debug::initialize();
  debug::initialize_classification();
  install_hints();
  material::install_gx_lookup();
#endif
}

void quiesce() noexcept {
#if AURORA_ENABLE_RENDER
  shaders::quiesce();
#endif
}

void shutdown() noexcept {
#if AURORA_ENABLE_RENDER
  // Game registrations are dropped first so no callback can reach anything torn down below
  shutdown_external_passes();
  texture::clear_material_manifests();
  material::remove_gx_lookup();
  clear_after_frame_callbacks();
  remove_hints();
  gx_ext::shutdown();
  scene::motion::shutdown();
  systems::shutdown();
  debug::shutdown_classification();
  debug::shutdown();
  compact::shutdown();
  // Compile jobs use the device, so they finish before the job pool and the device go away
  shaders::shutdown();
  jobs::shutdown();
  hzb::shutdown();
  taa::shutdown();
  // Unhooks the release listener before the pool clears, so its last releases have no cache left to evict from
  shutdown_bindings();
  transient_pool().clear();
  gpu_heap::shutdown();
#endif
  shutdown_settings_state();
}

void on_worker_frame_begin([[maybe_unused]] uint64_t frameId, [[maybe_unused]] const FrameState* frame) {
#if AURORA_ENABLE_RENDER
  // Heap generations bump here, before anything later in the frame compares them
  gpu_heap::begin_frame_worker(frameId);
  retire_heap_blocks();
  transient_pool_tick(frame != nullptr ? frame->graph : nullptr, TransientPool::Clock::now());
  bind_cache_tick(frameId);
  compact::worker_frame_begin(frameId);
  hzb::worker_tick(frameId);
  taa::worker_tick(frameId);
  systems::tick_worker(frameId);
#endif
}

void on_frame_recording([[maybe_unused]] uint64_t frameId) noexcept {
#if AURORA_ENABLE_RENDER
  gpu_heap::set_recording_frame(frameId);
#endif
}
} // namespace detail

Settings apply_settings(const Settings& requested) { return detail::submit_request(requested); }

const Settings& effective_settings() noexcept { return detail::latched_settings(); }

bool feature_enabled(const Settings& settings, Feature feature) noexcept {
  return detail::requested_option(settings, feature) != 0;
}

bool feature_active(Feature feature) noexcept {
  return feature_enabled(effective_settings(), feature) && feature_ready(feature);
}

FeatureSupport feature_support(Feature feature, uint8_t option) noexcept {
  return detail::current_support(feature, option, nullptr);
}

FeatureSupport feature_support(Feature feature, uint8_t option, const Settings& context) noexcept {
  return detail::current_support(feature, option, &context);
}

bool feature_ready(Feature feature) noexcept {
  // feature_active calls this, and inactive features stay ready
  if (detail::feature_info(feature).tier == detail::Tier::None || !feature_enabled(effective_settings(), feature)) {
    return true;
  }
  if (gfx::scene_layout_transitioning()) {
    return false;
  }
#if AURORA_ENABLE_RENDER
  if (feature == Feature::Taa && !detail::taa::ready(effective_settings())) {
    return false;
  }
#endif
  // DeferredFog mirrors the gbuffer request but has no composite until its built bit is set
  return feature != Feature::DeferredFog || feature_support(Feature::DeferredFog, 1).supported;
}

std::string_view feature_name(Feature feature) noexcept { return detail::feature_info(feature).name; }

std::string_view reason_name(Reason reason) noexcept {
  const auto index = static_cast<size_t>(reason);
  return index < ReasonNames.size() ? ReasonNames[index] : "Unknown";
}

bool hints_enabled() noexcept {
#if AURORA_ENABLE_RENDER
  return detail::hints_enabled();
#else
  return false;
#endif
}

HintCounters hint_counters(uint16_t id) noexcept {
  const gx::hints::Counters counters = gx::hints::counters(id);
  return {
      .decoded = counters.decoded,
      .skipped = counters.skipped,
      .malformed = counters.malformed,
      .late = counters.late,
      .bytes = counters.bytes,
  };
}

DrawRouteCounters draw_route_counters() noexcept {
  const gx::fifo::DrawRouteStats stats = gx::fifo::draw_route_stats();
  return {
      .routed = stats.routed,
      .suppressed = stats.suppressed,
      .mergeBreaks = stats.mergeBreaks,
      .substituted = stats.substituted,
  };
}

#if AURORA_ENABLE_RENDER
void draw_debug_window(bool* open) { debug::draw_window(open); }

std::span<const PassTiming> pass_timings() noexcept { return debug::pass_timings(); }

std::string graph_dump_json(bool includeFrameState) { return debug::graph_dump_json(includeFrameState); }
#else
void draw_debug_window(bool* open) {
  if (open == nullptr || !*open) {
    return;
  }
  if (ImGui::Begin(debug::RendererWindowId, open)) {
    ImGui::TextUnformatted("Render module disabled at build time");
  }
  ImGui::End();
}

std::span<const PassTiming> pass_timings() noexcept { return {}; }

std::string graph_dump_json(bool) { return std::string{debug::NotCompiledGraphJson}; }
#endif
} // namespace aurora::render

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
#include "ao/ao.hpp"
#include "debug/debug.hpp"
#include "debug/surface_expand.hpp"
#include "debug/views.hpp"
#include "geometry/compact.hpp"
#include "geometry/hzb.hpp"
#include "gx_ext/ext_provider.hpp"
#include "gx_ext/fwd_env.hpp"
#include "gx_ext/surface_table.hpp"
#include "lighting/clusters.hpp"
#include "lighting/deferred.hpp"
#include "lighting/exposure.hpp"
#include "lighting/openpbr.hpp"
#include "lighting/sky_probe.hpp"
#include "material/gx_lookup.hpp"
#include "native/native.hpp"
#include "post/composite.hpp"
#include "post/taa.hpp"
#include "post/temporal_resolve.hpp"
#include "post/upscale/upscale.hpp"
#include "scene/capture_list.hpp"
#include "scene/identity.hpp"
#include "scene/meshlets.hpp"
#include "scene/motion.hpp"
#include "scene/streamout.hpp"
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

constexpr std::array<std::string_view, 28> ReasonNames{
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
    "NeedsTextureFormats",
    "RuntimeMissing",
    "DriverTooOld",
    "UnsupportedGpu",
    "OutputTooLarge",
    "Probing",
};
static_assert(ReasonNames.size() == static_cast<size_t>(Reason::Probing) + 1);

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
  native::initialize();
  transient_pool().set_release_listener(&bind_cache_on_release);
  register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
  jobs::initialize(webgpu::g_backendType == wgpu::BackendType::WebGPU ? 0 : jobs::default_worker_count());
  shaders::initialize();
  compact::initialize();
  scene::streamout::initialize();
  scene::meshlets::initialize();
  hzb::initialize();
  taa::initialize();
  ao::initialize();
  composite::initialize();
  lighting::clusters::initialize();
  lighting::openpbr::initialize();
  lighting::deferred::initialize();
  lighting::sky_probe::initialize();
  lighting::exposure::initialize();
  gx_ext::fwd_env::initialize();
  gx_ext::surface_table::initialize();
  debug::surface_expand::initialize();
  upscale::initialize();
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
  debug::surface_expand::shutdown();
  gx_ext::surface_table::unregister_passes();
  gx_ext::fwd_env::shutdown();
  gx_ext::shutdown();
  scene::motion::shutdown();
  systems::shutdown();
  debug::shutdown_classification();
  debug::shutdown();
  scene::capture_list::shutdown();
  scene::streamout::shutdown();
  scene::meshlets::shutdown();
  compact::shutdown();
  // Compile jobs use the device, so they finish before the job pool and the device go away
  shaders::shutdown();
  jobs::shutdown();
  hzb::shutdown();
  lighting::exposure::shutdown();
  lighting::sky_probe::shutdown();
  lighting::deferred::shutdown();
  lighting::openpbr::shutdown();
  lighting::clusters::shutdown();
  composite::shutdown();
  ao::shutdown();
  taa::shutdown();
  upscale::shutdown();
  native::shutdown();
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
  scene::meshlets::worker_frame_begin(frameId);
  transient_pool_tick(frame != nullptr ? frame->graph : nullptr, TransientPool::Clock::now());
  bind_cache_tick(frameId);
  compact::worker_frame_begin(frameId);
  scene::streamout::worker_frame_begin(frameId, frame);
  hzb::worker_tick(frameId);
  // Before any pass of the frame publishes, so no frame reuses last frame's textures
  gx_ext::fwd_env::worker_frame_begin(frameId);
  ao::worker_tick(frameId);
  lighting::sky_probe::worker_tick(frameId);
  lighting::exposure::worker_tick(frameId);
  taa::worker_tick(frameId);
  systems::tick_worker(frameId);
  native_worker_frame_begin();
#endif
}

void on_frame_recording([[maybe_unused]] uint64_t frameId) noexcept {
#if AURORA_ENABLE_RENDER
  gpu_heap::set_recording_frame(frameId);
  scene::meshlets::set_recording_frame(frameId);
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
  if (feature == Feature::Taa && detail::temporal_resolve::installed().pass == detail::taa::Pass &&
      !detail::taa::ready(effective_settings())) {
    return false;
  }
  if (feature == Feature::Upscaler && !detail::upscale::ready(effective_settings())) {
    return false;
  }
  if (feature == Feature::Ao && !detail::ao::ready(effective_settings())) {
    return false;
  }
  // AO is visible only through the composite, so its pipeline and input contract gate Applying...
  if ((feature == Feature::Ao || feature == Feature::DeferredFog) && !detail::composite::ready(effective_settings())) {
    return false;
  }
  // With PBR the composite can be live while ao.final is a dummy, so AO readiness needs a produced ao.final
  if (feature == Feature::Ao && !detail::composite::ao_ready(effective_settings())) {
    return false;
  }
  // Bounce shows only through the composite and light.deferred, so their pipelines gate Applying...
  if (feature == Feature::Ssil &&
      (!detail::ao::ssil_ready(effective_settings()) || !detail::composite::ready(effective_settings()))) {
    return false;
  }
  // PBR shows only through the PBR tier and the composite, so a refused switch or a pending pipeline keeps Applying
  if (feature == Feature::Pbr && webgpu::scene_tier() != webgpu::SceneTier::Pbr) {
    return false;
  }
  // Without its lighting passes the composite keeps the original shading, so Applying... waits for them
  if (feature == Feature::Pbr &&
      !(detail::composite::ready(effective_settings()) && detail::lighting::ready(effective_settings()))) {
    return false;
  }
#endif
  return true;
}

std::string_view feature_name(Feature feature) noexcept { return detail::feature_info(feature).name; }

std::string_view reason_name(Reason reason) noexcept {
  const auto index = static_cast<size_t>(reason);
  return index < ReasonNames.size() ? ReasonNames[index] : "Unknown";
}

UpscaleStatus upscale_status() noexcept {
#if AURORA_ENABLE_RENDER
  return detail::upscale::status();
#else
  return {.active = Upscaler::Off};
#endif
}

void reenable_native_upscalers() noexcept { detail::native::reenable_upscalers(); }

void set_app_identity(const AppIdentity& identity) noexcept { detail::native::set_app_identity(identity); }

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
      .captured = stats.captured,
      .captureResets = stats.captureResets,
  };
}

RetainedStats retained_stats() noexcept {
#if AURORA_ENABLE_RENDER
  return scene::retained::stats();
#else
  return {};
#endif
}

bool retained_capture_active() noexcept {
#if AURORA_ENABLE_RENDER
  return scene::retained::capture_active();
#else
  return false;
#endif
}

bool retained_coverage_complete() noexcept {
#if AURORA_ENABLE_RENDER
  return scene::retained::coverage_complete();
#else
  return false;
#endif
}

void retained_hold_coverage([[maybe_unused]] bool hold) noexcept {
#if AURORA_ENABLE_RENDER
  scene::retained::hold_coverage(hold);
#endif
}

std::span<const DebugViewInfo> debug_views() noexcept {
#if AURORA_ENABLE_RENDER
  return debug::registry();
#else
  return {};
#endif
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

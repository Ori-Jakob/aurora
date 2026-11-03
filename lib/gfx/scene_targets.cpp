#include "scene_targets.hpp"

#include "clear.hpp"
#include "frame.hpp"
#include "frame_packet.hpp"
#include "hash.hpp"
#include "recording.hpp"
#include "../gx/render_ext.hpp"
#include "../logging.hpp"
#include "../webgpu/gpu.hpp"

#include <absl/container/flat_hash_set.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aurora::gfx {
namespace {
constexpr Module Log{"aurora::gfx"};
constexpr uint32_t SwitchTrackLatches = 600;

using Clock = std::chrono::steady_clock;

struct WarmPair {
  PipelineRef variant = 0;
  PipelineRef fallback = 0;
};

// Written on the FIFO while a frame records and read at the latch, when the FIFO is idle
struct WarmSet {
  absl::flat_hash_set<HashType> seen;
  std::vector<WarmPair> pairs;
  uint32_t overflow = 0;
  bool frameMatched = false;
};

struct WarmTotals {
  WarmProgress progress;
  uint32_t hardFailed = 0;
};

// Game thread
struct Controller {
  bool firstLatch = true;
  bool warming = false;
  SceneTier target = SceneTier::Legacy;
  std::optional<SceneTier> held;
  bool modNormal = false;
  Clock::time_point start;
  uint64_t skippedAtStart = 0;
  uint32_t matchedFrames = 0;
  std::vector<PipelineRef> clearRefs;
  bool tracking = false;
  SceneTier trackedTier = SceneTier::Legacy;
  uint64_t skippedAtSwitch = 0;
  uint32_t framesSinceSwitch = 0;
};

SceneTier g_lastWant = SceneTier::Legacy;
bool g_loggedUnsupported = false;
std::optional<bool> g_tierSupported;
Controller g_controller;
LayoutWarm g_layoutWarm;
const LayoutWarm* g_activeWarm = nullptr;
WarmSet g_warmSet;
std::atomic_bool g_transitioning{false};
std::atomic_bool g_liveOrPending{false};

void append_target(RenderTargetLayout& layout, ColorAttachmentSemantic semantic, wgpu::TextureFormat format,
                   const wgpu::Extent3D& size) noexcept {
  layout.colorAttachments[layout.colorAttachmentCount++] = {
      .semantic = semantic,
      .format = format,
      .width = size.width,
      .height = size.height,
  };
}
} // namespace

RenderTargetLayout compose_scene_layout(const SceneTargetFlags& flags, const SceneLayoutBase& base) noexcept {
  RenderTargetLayout layout{
      .depthStencilFormat = base.depth,
      .sampleCount = base.samples,
  };
  const bool gbuffer = flags.gbuffer || flags.surface;
  append_target(layout, ColorAttachmentSemantic::SceneColor, base.color, base.size);
  if (flags.normal || gbuffer) {
    append_target(layout, ColorAttachmentSemantic::Normal, webgpu::NormalBufferFormat, base.size);
  }
  if (gbuffer) {
    // Fixed order keeps the WebGPU byte cost at 8, 16, 20, 21, 25, and 42 with Surface
    append_target(layout, ColorAttachmentSemantic::Motion, webgpu::MotionBufferFormat, base.size);
    append_target(layout, ColorAttachmentSemantic::Reactive, webgpu::ReactiveBufferFormat, base.size);
    append_target(layout, ColorAttachmentSemantic::Fog, webgpu::FogRecordFormat, base.size);
  }
  if (flags.surface) {
    append_target(layout, ColorAttachmentSemantic::Surface, webgpu::SurfaceBufferFormat, base.size);
    append_target(layout, ColorAttachmentSemantic::SurfaceExt, webgpu::SurfaceExtBufferFormat, base.size);
  }
  // Last, so later targets go before it and the base layout stays a prefix
  if (flags.debugId) {
    append_target(layout, ColorAttachmentSemantic::DebugId, webgpu::DebugIdBufferFormat, base.size);
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}

SceneTargetFlags live_scene_targets() noexcept {
  const auto& config = webgpu::g_graphicsConfig;
  return {.normal = webgpu::normal_present(),
          .gbuffer = config.gbuffer,
          .surface = config.surfaceBuffer,
          .debugId = config.debugIdBuffer};
}

SceneLayoutBase live_layout_base() noexcept {
  return {
      .color = webgpu::g_frameBuffer.format,
      .depth = webgpu::g_graphicsConfig.depthFormat,
      .size = webgpu::g_frameBuffer.size,
      .samples = webgpu::g_graphicsConfig.msaaSamples,
  };
}

RenderTargetLayout pending_scene_layout(bool modNormal, SceneTier tier, const SceneLayoutBase& base) noexcept {
  // The inspector drops its target on a base change and warms it again, so it never joins the pending layout
  return compose_scene_layout(
      {.normal = modNormal, .gbuffer = tier != SceneTier::Legacy, .surface = tier == SceneTier::Pbr, .debugId = false},
      base);
}

std::array<uint64_t, 4> switchable_scene_layout_keys(const SceneLayoutBase& base) noexcept {
  return {
      pending_scene_layout(false, SceneTier::Legacy, base).key, pending_scene_layout(true, SceneTier::Legacy, base).key,
      pending_scene_layout(false, SceneTier::Base, base).key, pending_scene_layout(false, SceneTier::Pbr, base).key};
}

// A refused PBR layout fails on its own targets, so legacy still tries the base tier before giving up
SceneTier scene_tier_after_refusal(SceneTier refused, SceneTier live) noexcept {
  return refused == SceneTier::Pbr && live == SceneTier::Legacy ? SceneTier::Base : live;
}

SceneLayoutAction next_scene_layout_action(const SceneLayoutInput& input) noexcept {
  if (input.want == input.live || (input.want != SceneTier::Legacy && !input.supported)) {
    return input.warming ? SceneLayoutAction::Cancel : SceneLayoutAction::None;
  }
  if (input.firstLatch || input.blocking) {
    return SceneLayoutAction::Switch;
  }
  if (!input.warming) {
    return SceneLayoutAction::BeginWarm;
  }
  if (input.pendingKeyChanged) {
    return SceneLayoutAction::Restart;
  }
  const WarmProgress& progress = input.progress;
  // No timeout: switching early skips every draw whose pipeline is still compiling
  if (input.matchedFrames < SceneLayoutMatchedFrames || progress.ready + progress.failed != progress.total) {
    return SceneLayoutAction::None;
  }
  // Every G-buffer layout needs its own variants, so a switch into one can be refused; legacy is the floor and never is
  return input.want != SceneTier::Legacy && input.hardFailed != 0 ? SceneLayoutAction::Refuse
                                                                  : SceneLayoutAction::Switch;
}

namespace {
const char* tier_name(SceneTier tier) noexcept {
  switch (tier) {
  case SceneTier::Base:
    return "base G-buffer";
  case SceneTier::Pbr:
    return "PBR G-buffer";
  case SceneTier::Legacy:
    break;
  }
  return "legacy";
}

bool tier_supported(SceneTier tier) noexcept {
  return g_tierSupported ? *g_tierSupported : webgpu::scene_tier_supported(tier);
}

WarmProgress add_progress(const WarmProgress& a, const WarmProgress& b) noexcept {
  return {.ready = a.ready + b.ready, .total = a.total + b.total, .failed = a.failed + b.failed};
}

void publish_scene_layout() noexcept {
  const Controller& c = g_controller;
  g_transitioning.store(c.warming, std::memory_order_relaxed);
  g_liveOrPending.store(webgpu::g_graphicsConfig.gbuffer || (c.warming && c.target != SceneTier::Legacy),
                        std::memory_order_relaxed);
}

void open_warm(const RenderTargetLayout& pending, uint64_t liveKey) {
  g_layoutWarm = {.pending = pending, .liveKey = liveKey};
  g_warmSet = {};
  g_activeWarm = &g_layoutWarm;
}

void end_warm() {
  g_activeWarm = nullptr;
  g_warmSet = {};
  g_controller.warming = false;
  g_controller.clearRefs = {};
}

WarmTotals warm_totals() {
  std::vector<PipelineRef> refs;
  refs.reserve(g_warmSet.pairs.size() * 2);
  for (const WarmPair& pair : g_warmSet.pairs) {
    refs.push_back(pair.variant);
    if (pair.fallback != 0) {
      refs.push_back(pair.fallback);
    }
  }
  WarmTotals totals{.progress = count_resolved(refs)};
  if (totals.progress.failed == 0) {
    return totals;
  }
  for (const WarmPair& pair : g_warmSet.pairs) {
    if (pipeline_state(pair.variant) == PipelineState::Failed &&
        (pair.fallback == 0 || pipeline_state(pair.fallback) == PipelineState::Failed)) {
      ++totals.hardFailed;
    }
  }
  return totals;
}

// The mod arms its normal only after leaving the G-buffer, so its request there decides the legacy layout
bool pending_normal() noexcept { return webgpu::g_graphicsConfig.normalBuffer || g_controller.modNormal; }

void begin_warm(SceneTier target) {
  Controller& c = g_controller;
  const RenderTargetLayout pending = pending_scene_layout(pending_normal(), target, live_layout_base());
  open_warm(pending, scene_render_target_layout().key);
  c.warming = true;
  c.target = target;
  c.start = Clock::now();
  c.skippedAtStart = skipped_draws();
  c.matchedFrames = 0;
  c.clearRefs.clear();
  for (uint32_t bits = 0; bits < 8; ++bits) {
    const clear::PipelineConfig config{
        .clearColor = (bits & 1u) != 0, .clearAlpha = (bits & 2u) != 0, .clearDepth = (bits & 4u) != 0};
    c.clearRefs.push_back(warm_pipeline(config, pending));
  }
  warm_known_pipelines(pending, ShaderType::GX);
  warm_known_pipelines(pending, ShaderType::Clear);
}

void report_switch_skips() {
  Controller& c = g_controller;
  Log.info("Scene layout {}: {} draws skipped after the switch over {} frames", tier_name(c.trackedTier),
           skipped_draws() - c.skippedAtSwitch, c.framesSinceSwitch);
  c.tracking = false;
}

void step_switch_tracking() {
  Controller& c = g_controller;
  if (!c.tracking) {
    return;
  }
  ++c.framesSinceSwitch;
  // Skips are counted at encode, which trails the latch by the frames in flight
  if (c.framesSinceSwitch >= SwitchTrackLatches ||
      (c.framesSinceSwitch > detail::FrameSlotCount && background_pipelines_drained())) {
    report_switch_skips();
  }
}

void switch_scene_layout(SceneTier want, const SceneLayoutInput& input) {
  Controller& c = g_controller;
  if (c.tracking) {
    report_switch_skips();
  }
  const bool warmed = c.warming;
  if (warmed) {
    const WarmProgress& progress = input.progress;
    const std::string notes = g_warmSet.overflow != 0 ? fmt::format(", overflow {}", g_warmSet.overflow) : "";
    Log.info("Scene layout {} after {} ms: {}/{} ready, {} failed{}", tier_name(want), input.elapsedMs, progress.ready,
             progress.total, progress.failed, notes);
    Log.info("Scene layout {}: {} draws skipped while warming", tier_name(want), skipped_draws() - c.skippedAtStart);
    end_warm();
  }
  // Armed while the G-buffer still owns the texture, so the switch lands on the layout the warm prepared
  if (want == SceneTier::Legacy && c.modNormal) {
    webgpu::enable_normal_buffer();
  }
  const SceneTier live = webgpu::scene_tier();
  if (!webgpu::set_scene_tier(want)) {
    return;
  }
  if (warmed) {
    c.tracking = true;
    c.trackedTier = want;
    c.skippedAtSwitch = skipped_draws();
    c.framesSinceSwitch = 0;
  } else {
    Log.info("Scene targets: {} {}", tier_name(std::max(want, live)), want > live ? "on" : "off");
  }
}
} // namespace

const LayoutWarm* layout_warm() noexcept { return g_activeWarm; }

void warm_scene_variant(const gx::PipelineConfig& config) {
  const LayoutWarm* warm = g_activeWarm;
  if (warm == nullptr) {
    return;
  }
  g_warmSet.frameMatched = true;
  const HashType key = xxh3_hash(config, static_cast<HashType>(ShaderType::GX));
  if (g_warmSet.seen.contains(key)) {
    return;
  }
  if (g_warmSet.seen.size() >= MaxSceneLayoutWarmConfigs) {
    ++g_warmSet.overflow;
    return;
  }
  g_warmSet.seen.insert(key);
  WarmPair pair{.variant = warm_pipeline(config, warm->pending)};
  // The output-1 twin clears the surface class, so it is the better stand-in and costs no extra compile
  const gx::PipelineConfig fallback = config.shaderConfig.ext.output == 2
                                          ? gx::render_ext::output1_config(config)
                                          : gx::render_ext::fallback_config(config, warm->pending);
  if (std::memcmp(&fallback, &config, sizeof(config)) != 0) {
    pair.fallback = warm_pipeline(fallback, warm->pending);
  }
  g_warmSet.pairs.push_back(pair);
}

bool scene_layout_transitioning() noexcept { return g_transitioning.load(std::memory_order_relaxed); }

bool gbuffer_live_or_pending() noexcept { return g_liveOrPending.load(std::memory_order_relaxed); }

namespace detail {
void latch_scene_layout(SceneTier want) {
  Controller& c = g_controller;
  if (std::exchange(g_lastWant, want) != want) {
    g_loggedUnsupported = false;
    c.held.reset();
  }
  const SceneTier live = webgpu::scene_tier();
  SceneLayoutInput input{
      .want = c.held.value_or(want),
      .live = live,
      .warming = c.warming,
      .firstLatch = c.firstLatch,
      .blocking = blocking_pipelines(),
  };
  input.supported = tier_supported(input.want);
  c.firstLatch = false;
  // Taken every latch so an old request never counts, then held while the switch to legacy is pending
  const bool normalRequested = take_mod_normal_request();
  c.modNormal = live != SceneTier::Legacy && input.want == SceneTier::Legacy && (c.modNormal || normalRequested);
  if (input.want != SceneTier::Legacy && input.want != live && !input.supported &&
      !std::exchange(g_loggedUnsupported, true)) {
    Log.warn("Scene targets: {} unavailable on this device", tier_name(input.want));
  }
  step_switch_tracking();
  if (c.warming) {
    // Built from the want, so a tier change mid-warm restarts instead of switching onto a layout nobody warmed
    const RenderTargetLayout pending = pending_scene_layout(pending_normal(), input.want, live_layout_base());
    input.pendingKeyChanged = pending.key != g_layoutWarm.pending.key;
    if (std::exchange(g_warmSet.frameMatched, false)) {
      ++c.matchedFrames;
    }
    const WarmTotals totals = warm_totals();
    WarmProgress progress = add_progress(totals.progress, count_resolved(c.clearRefs));
    progress = add_progress(progress, warm_known_pipelines(g_layoutWarm.pending, ShaderType::GX));
    input.progress = add_progress(progress, warm_known_pipelines(g_layoutWarm.pending, ShaderType::Clear));
    input.hardFailed = totals.hardFailed;
    input.matchedFrames = c.matchedFrames;
    input.elapsedMs =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - c.start).count());
  }
  switch (next_scene_layout_action(input)) {
  case SceneLayoutAction::None:
    if (c.warming) {
      g_layoutWarm.liveKey = scene_render_target_layout().key;
    }
    break;
  case SceneLayoutAction::BeginWarm:
  case SceneLayoutAction::Restart:
    begin_warm(input.want);
    break;
  case SceneLayoutAction::Switch:
    switch_scene_layout(input.want, input);
    break;
  case SceneLayoutAction::Refuse:
    Log.warn("Scene layout {} refused: {} pipelines failed with no fallback", tier_name(input.want), input.hardFailed);
    end_warm();
    c.held = scene_tier_after_refusal(input.want, live);
    if (*c.held != live) {
      Log.info("Scene layout {}: trying the {} instead", tier_name(input.want), tier_name(*c.held));
    }
    break;
  case SceneLayoutAction::Cancel:
    Log.info("Scene layout {} cancelled after {} ms: {}/{} ready, {} failed", tier_name(c.target), input.elapsedMs,
             input.progress.ready, input.progress.total, input.progress.failed);
    end_warm();
    break;
  }
  publish_scene_layout();
}

void refresh_layout_warm() noexcept {
  if (g_activeWarm != nullptr) {
    g_layoutWarm.liveKey = scene_render_target_layout().key;
  }
}

void reset_scene_layout() {
  end_warm();
  g_controller = {};
  g_lastWant = SceneTier::Legacy;
  g_loggedUnsupported = false;
  g_tierSupported.reset();
  publish_scene_layout();
}
} // namespace detail

namespace testing {
void begin_warm(const RenderTargetLayout& pending, uint64_t liveKey) {
  open_warm(pending, liveKey);
  g_controller.matchedFrames = 0;
}

uint32_t latch_warm() noexcept {
  if (std::exchange(g_warmSet.frameMatched, false)) {
    ++g_controller.matchedFrames;
  }
  return g_controller.matchedFrames;
}

void set_scene_tier_supported(std::optional<bool> supported) noexcept { g_tierSupported = supported; }

WarmTally warm_tally() {
  const WarmTotals totals = warm_totals();
  return {.progress = totals.progress,
          .hardFailed = totals.hardFailed,
          .pairs = static_cast<uint32_t>(g_warmSet.pairs.size()),
          .overflow = g_warmSet.overflow};
}
} // namespace testing
} // namespace aurora::gfx

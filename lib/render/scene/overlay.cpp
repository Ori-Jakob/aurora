#include "overlay.hpp"

#include "classifier.hpp"

#include "../debug/views.hpp"
#include "../frame_state.hpp"
#include "../hints.hpp"
#include "../post/composite.hpp"
#include "../../gfx/recording.hpp"
#include "../../gfx/scene_targets.hpp"
#include "../../logging.hpp"

#include <aurora/gfx.hpp>
#include <SDL3/SDL_stdinc.h>

#include <atomic>
#include <string_view>
#include <utility>

namespace aurora::render::scene::overlay {
namespace {
constexpr Module Log{"aurora::render::overlay"};

// Game thread while the FIFO is idle
bool g_envRead = false;
Env g_env = Env::Unset;
bool g_installed = false;
uint64_t g_armedFrames = 0;
uint64_t g_blockedFrames = 0;
bool g_blockWarned = false;
gfx::OverlayStats g_totals;

// Set while the FIFO is idle, read on the FIFO thread
std::atomic<PbrAppliedFn> g_source{nullptr};
bool g_armed = false;

void read_env() noexcept {
  if (std::exchange(g_envRead, true)) {
    return;
  }
  const char* value = SDL_getenv("AURORA_RENDER_OVERLAY_DEFER");
  if (value == nullptr) {
    return;
  }
  const std::string_view text{value};
  if (text == "off" || text == "0") {
    g_env = Env::Off;
    return;
  }
  if (text == "force") {
    g_env = Env::Force;
  } else if (text == "drop") {
    g_env = Env::Drop;
  } else if (text == "restore") {
    g_env = Env::Restore;
  } else {
    Log.warn("AURORA_RENDER_OVERLAY_DEFER={} is not off, force, drop or restore; ignored", text);
    return;
  }
  Log.warn("AURORA_RENDER_OVERLAY_DEFER={}: verification only", text);
}
} // namespace

Env env() noexcept {
  read_env();
  return g_env;
}

ArmInputs arm_inputs(const detail::FrameState& state) noexcept {
  const gfx::RenderTargetLayout layout = gfx::scene_render_target_layout();
  return {
      .hints = detail::hints_enabled(),
      .gbufferLayout = gfx::layout_has_gbuffer(layout),
      .surfaceLayout = gfx::layout_has_surface(layout),
      .pbr = state.settings.pbr,
      .legacyBlend = state.settings.legacyBlend,
      .identityForced = detail::composite::identity_forced(),
      .classView = debug::find_view(state.debugView) != nullptr,
      .passAfterOverlay = state.graph != nullptr && pass_after_overlay(*state.graph),
      .env = env(),
  };
}

void set_pbr_applied_source(PbrAppliedFn source) noexcept { g_source.store(source, std::memory_order_release); }

gfx::OverlayDisposition disposition(const detail::StageData* opaqueDone) noexcept {
  switch (g_env) {
  case Env::Drop:
    return gfx::OverlayDisposition::Discard;
  case Env::Restore:
    return gfx::OverlayDisposition::Restore;
  case Env::Force:
    return gfx::OverlayDisposition::Replay;
  default:
    break;
  }
  const PbrAppliedFn source = g_source.load(std::memory_order_acquire);
  return source != nullptr && opaqueDone != nullptr && source(*opaqueDone) ? gfx::OverlayDisposition::Replay
                                                                           : gfx::OverlayDisposition::Restore;
}

void install() noexcept {
  read_env();
  gfx::set_overlay_policy(&disposition);
  g_installed = true;
}

void remove() noexcept {
  if (!std::exchange(g_installed, false)) {
    return;
  }
  gfx::set_overlay_policy(nullptr);
  g_armed = false;
  g_totals = gfx::overlay_stats();
  const gfx::OverlayStats& t = g_totals;
  constexpr std::string_view Line =
      "Render overlays: {} draws deferred over {} armed frames ({} blocked by a pass after gx.overlay); {} replayed "
      "frames, restored {} seal / {} late / {} unapplied, {} discarded, {} refused, {} dropped, {} in-place after";
  if (t.draws != 0) {
    Log.info(Line, t.draws, g_armedFrames, g_blockedFrames, t.replayFrames, t.restoredSeal, t.restoredLate,
             t.restoredUnapplied, t.discarded, t.refused, t.dropped, t.inPlaceAfter);
  } else {
    Log.debug(Line, t.draws, g_armedFrames, g_blockedFrames, t.replayFrames, t.restoredSeal, t.restoredLate,
              t.restoredUnapplied, t.discarded, t.refused, t.dropped, t.inPlaceAfter);
  }
}

void apply_frame_state(const detail::FrameState& state) noexcept {
  g_totals = gfx::overlay_stats();
  const ArmInputs inputs = arm_inputs(state);
  g_armed = armed_for(inputs);
  if (g_armed) {
    ++g_armedFrames;
  }
  ArmInputs unblocked = inputs;
  unblocked.passAfterOverlay = false;
  if (inputs.passAfterOverlay && armed_for(unblocked)) {
    ++g_blockedFrames;
    if (!std::exchange(g_blockWarned, true)) {
      const detail::NodeId pass = first_pass_after_overlay(*state.graph);
      Log.warn("Render overlays: '{}' runs after gx.overlay, so overlays draw in place", state.graph->nodes[pass].name);
    }
  }
}

bool defers(const DrawClass& cls, bool depthWrite) noexcept {
  // Depth writers reset the surface class below them, so only see-through overlays would be mixed away
  return g_armed && cls.kind == Kind::Overlay && cls.colorWrite && !depthWrite &&
         (cls.key.categories & category::FakeShadow) == 0;
}

namespace testing {
void reset() noexcept {
  if (g_installed) {
    gfx::set_overlay_policy(nullptr);
  }
  g_envRead = false;
  g_env = Env::Unset;
  g_installed = false;
  g_armedFrames = 0;
  g_blockedFrames = 0;
  g_blockWarned = false;
  g_totals = {};
  g_source.store(nullptr, std::memory_order_release);
  g_armed = false;
}

void set_armed(bool armed) noexcept { g_armed = armed; }

void set_env(Env env) noexcept {
  g_envRead = true;
  g_env = env;
}
} // namespace testing
} // namespace aurora::render::scene::overlay

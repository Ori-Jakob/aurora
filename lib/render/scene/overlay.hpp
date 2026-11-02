#pragma once

#include "../graph.hpp"

#include <cstdint>

namespace aurora::gfx {
enum class OverlayDisposition : uint8_t;
} // namespace aurora::gfx

namespace aurora::render::detail {
struct FrameState;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::overlay {
enum class Env : uint8_t { Unset, Off, Force, Drop, Restore };

struct ArmInputs {
  bool hints = false;
  bool gbufferLayout = false;
  bool surfaceLayout = false;
  bool pbr = false;
  float legacyBlend = 1.f;
  bool identityForced = false;
  bool classView = false;
  bool passAfterOverlay = false;
  Env env = Env::Unset;
};

[[nodiscard]] constexpr bool pbr_can_change_pixels(const ArmInputs& in) noexcept {
  return in.surfaceLayout && in.pbr && in.legacyBlend < 1.f;
}

[[nodiscard]] constexpr bool armed_for(const ArmInputs& in) noexcept {
  if (!in.hints || !in.gbufferLayout || in.identityForced || in.classView || in.passAfterOverlay ||
      in.env == Env::Off) {
    return false;
  }
  return in.env != Env::Unset || pbr_can_change_pixels(in);
}

// Graphs are replaced on recompile, so nothing is cached by address
[[nodiscard]] inline detail::NodeId first_pass_after_overlay(const detail::CompiledGraph& graph) noexcept {
  bool after = false;
  for (const detail::NodeId id : graph.stages[static_cast<size_t>(Stage::OpaqueDone)].nodes) {
    const detail::NodeKind kind = graph.nodes[id].kind;
    if (kind == detail::NodeKind::Overlay) {
      after = true;
    } else if (after && kind == detail::NodeKind::Pass) {
      return id;
    }
  }
  return detail::InvalidNode;
}

[[nodiscard]] inline bool pass_after_overlay(const detail::CompiledGraph& graph) noexcept {
  return first_pass_after_overlay(graph) != detail::InvalidNode;
}

using PbrAppliedFn = bool (*)(const detail::StageData& opaqueDone) noexcept;

// Game thread while the FIFO is idle
[[nodiscard]] ArmInputs arm_inputs(const detail::FrameState& state) noexcept;
[[nodiscard]] Env env() noexcept;
void set_pbr_applied_source(PbrAppliedFn source) noexcept;
void install() noexcept;
void remove() noexcept;
void apply_frame_state(const detail::FrameState& state) noexcept;

// FIFO thread
[[nodiscard]] gfx::OverlayDisposition disposition(const detail::StageData* opaqueDone) noexcept;
[[nodiscard]] bool defers(const DrawClass& cls, bool depthWrite) noexcept;

namespace testing {
void reset() noexcept;
void set_armed(bool armed) noexcept;
void set_env(Env env) noexcept;
} // namespace testing
} // namespace aurora::render::scene::overlay

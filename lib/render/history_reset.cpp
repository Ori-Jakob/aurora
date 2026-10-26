#include "history_reset.hpp"

#include "graph.hpp"
#include "../logging.hpp"

#include <cmath>
#include <optional>

namespace aurora::render::detail::history_reset {
namespace {
constexpr Module Log{"aurora::render::history"};

Camera g_camera;
Stats g_stats;

std::optional<Camera> camera_of(const gx::hints::Camera& camera) noexcept {
  const Mat3x4<float>& view = camera.view;
  const double m[3][3] = {
      {view.m0[0], view.m0[1], view.m0[2]},
      {view.m1[0], view.m1[1], view.m1[2]},
      {view.m2[0], view.m2[1], view.m2[2]},
  };
  const double t[3] = {view.m0[3], view.m1[3], view.m2[3]};
  const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  if (!std::isfinite(det) || std::abs(det) < 1e-12) {
    return std::nullopt;
  }
  const double inv[3][3] = {
      {c00 / det, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det},
      {c01 / det, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det},
      {c02 / det, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det},
  };
  double eye[3];
  double forward[3];
  for (int r = 0; r < 3; ++r) {
    eye[r] = -(inv[r][0] * t[0] + inv[r][1] * t[1] + inv[r][2] * t[2]);
    forward[r] = -inv[r][2];
  }
  const double length = std::sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
  if (!std::isfinite(length) || length == 0.0 || !std::isfinite(eye[0] + eye[1] + eye[2])) {
    return std::nullopt;
  }
  return Camera{
      .valid = true,
      .cutEpoch = camera.cutEpoch,
      .eye = {static_cast<float>(eye[0]), static_cast<float>(eye[1]), static_cast<float>(eye[2])},
      .forward = {static_cast<float>(forward[0] / length), static_cast<float>(forward[1] / length),
                  static_cast<float>(forward[2] / length)},
  };
}

float units_per_meter(const SceneHints& hints) noexcept {
  const float value = hints.environment.value.unitsPerMeter;
  return hints.has(SceneHasEnvironment) && std::isfinite(value) && value > 0.f ? value : DefaultUnitsPerMeter;
}
} // namespace

std::string_view reason_name(Reason reason) noexcept {
  switch (reason) {
  case Reason::None:
    return "None";
  case Reason::CameraReacquired:
    return "CameraReacquired";
  case Reason::CutEpoch:
    return "CutEpoch";
  case Reason::SceneEpoch:
    return "SceneEpoch";
  case Reason::CameraJump:
    return "CameraJump";
  }
  return "Unknown";
}

Reason classify(const Camera& prev, const SceneHints& hints, Camera& next) noexcept {
  next = {};
  if (!hints.has(SceneHasMainCamera)) {
    return Reason::None;
  }
  const std::optional<Camera> camera = camera_of(hints.mainCamera);
  if (!camera) {
    return Reason::None;
  }
  next = *camera;
  if (!prev.valid) {
    return Reason::CameraReacquired;
  }
  if (next.cutEpoch != prev.cutEpoch) {
    return Reason::CutEpoch;
  }
  if (hints.sceneEpochChanged) {
    return Reason::SceneEpoch;
  }
  const double dx = double{next.eye.x} - prev.eye.x;
  const double dy = double{next.eye.y} - prev.eye.y;
  const double dz = double{next.eye.z} - prev.eye.z;
  const double limit = double{JumpMetres} * units_per_meter(hints);
  const double turn = double{next.forward.x} * prev.forward.x + double{next.forward.y} * prev.forward.y +
                      double{next.forward.z} * prev.forward.z;
  if (dx * dx + dy * dy + dz * dz > limit * limit || turn < MinForwardCos) {
    return Reason::CameraJump;
  }
  return Reason::None;
}

Reason begin_frame(const CompiledGraph& graph, uint64_t frameId) {
  if (graph.histories.empty()) {
    g_camera = {};
    return Reason::None;
  }
  const std::shared_ptr<const SceneHints> hints = scene_hints();
  Camera next;
  const Reason reason = classify(g_camera, *hints, next);
  g_camera = next;
  ++g_stats.frames;
  ++g_stats.reasons[static_cast<size_t>(reason)];
  if (reason != Reason::None) {
    Log.debug("History reset at frame {}: {}", frameId, reason_name(reason));
  }
  return reason;
}

Stats stats() noexcept { return g_stats; }

void reset() noexcept {
  g_camera = {};
  g_stats = {};
}
} // namespace aurora::render::detail::history_reset

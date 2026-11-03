#include "stage_camera.hpp"

#include "hints.hpp"

#include <cmath>

namespace aurora::render::detail {
bool fill_stage_camera(const SceneHints* hints, StageCamera& out) noexcept {
  out = {};
  if (hints == nullptr || !hints->has(SceneHasMainCamera)) {
    return false;
  }
  const Mat4x4<float>& proj = hints->mainCamera.gxProj;
  if (!gx::hints::main_camera_usable(proj)) {
    return false;
  }
  out.depthMul = proj[2][3];
  out.depthAdd = proj[2][2];
  out.p00 = proj[0][0];
  out.p11 = proj[1][1];
  out.p02 = proj[0][2];
  out.p12 = proj[1][2];
  out.jitterPx[0] = hints->jitterPx[0];
  out.jitterPx[1] = hints->jitterPx[1];
  out.unitsPerMeter = units_per_metre(*hints);
  out.flags = CameraValid;
  return true;
}

bool fill_view_rotation(const SceneHints* hints, ViewRotation& out) noexcept {
  out = {};
  if (hints == nullptr || !hints->has(SceneHasMainCamera)) {
    return false;
  }
  const Mat3x4<float>& view = hints->mainCamera.view;
  double m[3][3] = {
      {view.m0[0], view.m0[1], view.m0[2]},
      {view.m1[0], view.m1[1], view.m1[2]},
      {view.m2[0], view.m2[1], view.m2[2]},
  };
  // Game views can carry scale or shear, and the probe needs pure directions
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < r; ++k) {
      const double d = m[r][0] * m[k][0] + m[r][1] * m[k][1] + m[r][2] * m[k][2];
      for (int c = 0; c < 3; ++c) {
        m[r][c] -= d * m[k][c];
      }
    }
    const double length = std::sqrt(m[r][0] * m[r][0] + m[r][1] * m[r][1] + m[r][2] * m[r][2]);
    if (!std::isfinite(length) || length < 1e-9) {
      return false;
    }
    for (int c = 0; c < 3; ++c) {
      m[r][c] /= length;
    }
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out.rows[r][c] = static_cast<float>(m[r][c]);
    }
  }
  return true;
}

float units_per_metre(const SceneHints& hints) noexcept {
  const float value = hints.environment.value.unitsPerMeter;
  if (!hints.has(SceneHasEnvironment) || !std::isfinite(value) || value <= 0.f) {
    return DefaultUnitsPerMetre;
  }
  return value;
}
} // namespace aurora::render::detail

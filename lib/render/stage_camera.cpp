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

float units_per_metre(const SceneHints& hints) noexcept {
  const float value = hints.environment.value.unitsPerMeter;
  if (!hints.has(SceneHasEnvironment) || !std::isfinite(value) || value <= 0.f) {
    return DefaultUnitsPerMetre;
  }
  return value;
}
} // namespace aurora::render::detail

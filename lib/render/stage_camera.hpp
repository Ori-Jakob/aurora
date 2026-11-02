#pragma once

#include <cstdint>

namespace aurora::render::detail {
struct SceneHints;

inline constexpr float DefaultUnitsPerMetre = 100.f;

enum StageCameraFlags : uint32_t {
  CameraValid = 1u << 0,
};

// Later passes embed this block in their own params, so frame uniforms stay unchanged
struct StageCamera {
  float depthMul = 0.f;
  float depthAdd = 0.f;
  float p00 = 0.f;
  float p11 = 0.f;
  float p02 = 0.f;
  float p12 = 0.f;
  float jitterPx[2]{};
  float unitsPerMeter = 0.f;
  uint32_t flags = 0;
  uint32_t pad[2]{};
};
static_assert(sizeof(StageCamera) == 48);

bool fill_stage_camera(const SceneHints* hints, StageCamera& out) noexcept;
float units_per_metre(const SceneHints& hints) noexcept;
} // namespace aurora::render::detail

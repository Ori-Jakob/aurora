#pragma once

#include "../gx/hints.hpp"

#include <memory>
#include <vector>

namespace aurora::render::detail {
enum SceneHintBits : uint32_t {
  SceneHasMainCamera = 1u << 0,
  SceneHasLights = 1u << 1,
  SceneHasEnvironment = 1u << 2,
  SceneHasInfluencers = 1u << 3,
  SceneHasWindSources = 1u << 4,
  SceneHasEpoch = 1u << 5,
};

enum class MotionReason : uint8_t { Valid, NoCamera, NoPrevious, Unusable, Cut, SceneEpoch, StorageFull, Inactive };

// Frame-level hint values as of the frame's accepted SCENE_BEGIN; immutable once published
struct SceneHints {
  uint64_t hintFrame = 0;
  uint32_t present = 0;
  gx::hints::Camera mainCamera{};
  std::vector<AuroraLight> lights;
  gx::hints::Environment environment{};
  std::vector<AuroraInfluencer> influencers;
  std::vector<AuroraWindSource> windSources;
  uint32_t sceneEpoch = 0;
  bool sceneEpochChanged = false;
  // Decided at the last pre-freeze MAIN camera; TAA resets history when false
  bool motionValid = false;
  MotionReason motionReason = MotionReason::NoCamera;
  float jitterPx[2] = {0.f, 0.f};

  [[nodiscard]] bool has(uint32_t bits) const noexcept { return (present & bits) == bits; }
};

bool hints_enabled() noexcept;
void install_hints() noexcept;
void remove_hints() noexcept;
void begin_hint_frame() noexcept;
std::shared_ptr<const SceneHints> scene_hints();
const SceneHints& live_scene_hints() noexcept;
bool scene_hints_frozen() noexcept;

namespace testing {
void set_hints_override(int value) noexcept;
void reset_hints() noexcept;
} // namespace testing
} // namespace aurora::render::detail

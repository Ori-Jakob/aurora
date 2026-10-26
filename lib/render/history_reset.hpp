#pragma once

#include "hints.hpp"

#include <aurora/math.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace aurora::render::detail {
struct CompiledGraph;
} // namespace aurora::render::detail

namespace aurora::render::detail::history_reset {
enum class Reason : uint8_t { None, CameraReacquired, CutEpoch, SceneEpoch, CameraJump };
inline constexpr size_t ReasonCount = 5;

struct Camera {
  bool valid = false;
  uint32_t cutEpoch = 0;
  Vec3<float> eye{};
  Vec3<float> forward{};
};

// Matches dusk's always-on jump limits; rotation itself reprojects exactly
inline constexpr float JumpMetres = 10.f;
inline constexpr float MinForwardCos = 0.5f;
inline constexpr float DefaultUnitsPerMeter = 100.f;

struct Stats {
  uint64_t frames = 0;
  std::array<uint64_t, ReasonCount> reasons{};
};

[[nodiscard]] std::string_view reason_name(Reason reason) noexcept;
[[nodiscard]] Reason classify(const Camera& prev, const SceneHints& hints, Camera& next) noexcept;
Reason begin_frame(const CompiledGraph& graph, uint64_t frameId);
[[nodiscard]] Stats stats() noexcept;
void reset() noexcept;
} // namespace aurora::render::detail::history_reset

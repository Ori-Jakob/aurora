#pragma once

#include "../hints.hpp"
#include "../../gx/render_ext.hpp"

#include <aurora/math.hpp>
#include <aurora/render.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::motion {
struct MotionFrame {
  float jitterPx[2] = {0.f, 0.f};
  uint32_t valid = 0;
  uint32_t pad = 0;
  Vec4<float> prevProj[3]{};
  Mat3x4<float> reproject{};
};
static_assert(sizeof(MotionFrame) == gx::render_ext::FrameMotionWords * sizeof(uint32_t));
static_assert(offsetof(MotionFrame, jitterPx) == gx::render_ext::FrameMotionJitterWord * sizeof(uint32_t));
static_assert(offsetof(MotionFrame, valid) == gx::render_ext::FrameMotionValidWord * sizeof(uint32_t));
static_assert(offsetof(MotionFrame, prevProj) == gx::render_ext::FrameMotionPrevProjWord * sizeof(uint32_t));
static_assert(offsetof(MotionFrame, reproject) == gx::render_ext::FrameMotionReprojectWord * sizeof(uint32_t));

struct DrawMotion {
  uint32_t flags = 0;
  uint32_t prevMtxOff = 0;
};

struct Validity {
  bool valid = false;
  render::detail::MotionReason reason = render::detail::MotionReason::Inactive;
  float jitterPx[2] = {0.f, 0.f};
};

enum class PlayerKind : uint8_t { Object, Palette, CpuDeformed };
inline constexpr size_t PlayerKindCount = 3;

struct PlayerCounts {
  uint64_t eligible = 0;
  uint64_t exact = 0;
  uint64_t fallback = 0;
  uint64_t miss = 0;
};

struct Counts {
  uint64_t none = 0;
  uint64_t camera = 0;
  uint64_t object = 0;
  uint64_t palette = 0;
  uint64_t jittered = 0;
  uint64_t exact = 0;
  uint64_t fallback = 0;
  uint64_t miss = 0;
  uint64_t ambiguous = 0;
  uint64_t overflow = 0;
  uint64_t pushRefused = 0;
  uint64_t badIndex = 0;
  std::array<PlayerCounts, PlayerKindCount> player{};
  uint64_t framesInWindow = 0;
  uint64_t framesWithoutPlayer = 0;
  uint64_t immobileSamples = 0;
  uint64_t immobileSkipped = 0;
  float immobileMaxPx = 0.f;
};

struct Stats {
  uint64_t frames = 0;
  Counts frame;
  Counts total;
  Counts check;
};

using PushFn = gfx::Range (*)(std::span<const uint8_t> bytes) noexcept;

// Game thread, while the FIFO is idle
void latch(const Settings& settings, bool motionLayoutLive) noexcept;
[[nodiscard]] std::array<float, 2> frame_jitter() noexcept;
[[nodiscard]] const gx::hints::Camera* previous_camera() noexcept;
[[nodiscard]] Stats latest_stats() noexcept;
void reset() noexcept;
void shutdown() noexcept;

// FIFO thread
Validity on_main_camera(const gx::hints::Camera& camera, bool usable) noexcept;
DrawMotion draw_motion(const gx::DrawData& draw, const gx::PipelineConfig& config, const DrawClass& cls) noexcept;

namespace detail {
[[nodiscard]] std::array<float, 2> halton_jitter(uint32_t phase) noexcept;
[[nodiscard]] bool rows_match(const Mat4x4<float>& proj, const Mat4x4<float>& mainProj) noexcept;
[[nodiscard]] Mat3x4<float> reproject_matrix(const Mat3x4<float>& prevView, const Mat3x4<float>& curView) noexcept;
[[nodiscard]] std::optional<float> immobile_residual_px(const Mat3x4<float>& cur, const Mat3x4<float>& prev,
                                                        const Mat3x4<float>& reproject, const Mat4x4<float>& curProj,
                                                        const Mat4x4<float>& prevProj, float width,
                                                        float height) noexcept;
[[nodiscard]] std::optional<std::string_view> check_failure(const Counts& check) noexcept;
} // namespace detail

namespace testing {
enum class JitterEnv : uint8_t { Unset, On, Off };
void set_jitter_env(JitterEnv env) noexcept;
void set_check_range(uint32_t first, uint32_t last) noexcept;
void set_push(PushFn push) noexcept;
[[nodiscard]] bool active() noexcept;
[[nodiscard]] bool jitter_active() noexcept;
[[nodiscard]] uint64_t storage_fingerprint() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::scene::motion

#pragma once

#include <aurora/render.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
struct GxReadDesc;
struct PassDesc;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::gx_ext::fwd_env {
inline constexpr std::string_view Pass = "fwd_env.publish";
inline constexpr std::string_view ReadNode = "gx@PrePost";

[[nodiscard]] bool screen_ao_receiver(const scene::DrawClass& cls, const gx::PipelineConfig& config) noexcept;
[[nodiscard]] bool forward_ao_wanted(const Settings& settings) noexcept;

// Game thread, while the FIFO is idle
void apply_frame_state(const detail::FrameState& state) noexcept;

// FIFO thread
[[nodiscard]] uint32_t record_flags(const scene::DrawClass& cls, const gx::PipelineConfig& config) noexcept;

[[nodiscard]] detail::PassDesc pass_desc();
[[nodiscard]] detail::GxReadDesc read_desc() noexcept;
void initialize();
void shutdown() noexcept;
void worker_frame_begin(uint64_t frameId) noexcept;

namespace testing {
struct Counters {
  uint64_t frames = 0;
  uint64_t published = 0;
  uint64_t neutral = 0;
  uint64_t dummy = 0;
  uint64_t mismatch = 0;
  uint64_t neutralAfterFirstPublish = 0;
  uint32_t cacheEntries = 0;
};
[[nodiscard]] Counters counters() noexcept;
[[nodiscard]] bool screen_ao_active() noexcept;
void set_identity_override(std::optional<bool> identity) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext::fwd_env

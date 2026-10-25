#pragma once

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::gx_ext {
void apply_frame_state(const detail::FrameState& state) noexcept;
void shutdown() noexcept;

namespace testing {
void set_force(bool force) noexcept;
bool active() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext

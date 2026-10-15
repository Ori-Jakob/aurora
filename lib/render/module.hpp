#pragma once

#include <cstdint>

namespace aurora::render::detail {
struct FrameState;

bool module_enabled() noexcept;
void initialize();
void quiesce() noexcept;
void shutdown() noexcept;
void on_worker_frame_begin(uint64_t frameId, const FrameState* frame);
} // namespace aurora::render::detail

#pragma once

namespace aurora::trace::detail {
void on_frame_begin() noexcept;
void on_frame_end() noexcept;
void note_untraceable(const char* api) noexcept;
void shutdown() noexcept;
} // namespace aurora::trace::detail

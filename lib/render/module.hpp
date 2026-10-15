#pragma once

namespace aurora::render::detail {
bool module_enabled() noexcept;
void initialize();
void shutdown() noexcept;
} // namespace aurora::render::detail

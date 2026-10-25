#pragma once

#include <mutex>
#include <optional>
#include <string>

namespace aurora::webgpu {
// Dawn may keep one error-scope stack per device, so every scope user and the render module's device calls share it
std::mutex& error_scope_mutex() noexcept;
void push_error_scopes() noexcept;
std::optional<std::string> pop_error_scopes() noexcept;
} // namespace aurora::webgpu

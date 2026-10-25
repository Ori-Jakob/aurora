#include "error_scope.hpp"

#include "gpu.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace aurora::webgpu {
namespace {
constexpr uint64_t DawnWaitNs = 5'000'000'000;

struct WaitState {
  std::mutex mutex;
  std::optional<std::string> error;
};

// Requires error_scope_mutex held. State is shared with the callback, which Dawn may still run after a timeout
std::optional<std::string> pop_error_scope() {
  auto state = std::make_shared<WaitState>();
  const wgpu::Future future =
      g_device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
                             [state](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message) {
                               std::lock_guard lock{state->mutex};
                               if (status != wgpu::PopErrorScopeStatus::Success) {
                                 state->error = "error scope lost";
                               } else if (type != wgpu::ErrorType::NoError) {
                                 state->error = std::string{std::string_view{message}};
                               }
                             });
  if (g_instance.WaitAny(future, DawnWaitNs) != wgpu::WaitStatus::Success) {
    return "timed out";
  }
  std::lock_guard lock{state->mutex};
  return state->error;
}
} // namespace

std::mutex& error_scope_mutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

void push_error_scopes() noexcept {
  g_device.PushErrorScope(wgpu::ErrorFilter::Validation);
  g_device.PushErrorScope(wgpu::ErrorFilter::Internal);
}

std::optional<std::string> pop_error_scopes() noexcept {
  const std::optional<std::string> internal = pop_error_scope();
  const std::optional<std::string> validation = pop_error_scope();
  if (internal && validation) {
    return *validation + "\n" + *internal;
  }
  return validation ? validation : internal;
}
} // namespace aurora::webgpu

#include "teardown.hpp"

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

namespace aurora::render::detail::native::d3d12 {
namespace {
std::mutex g_mutex;
std::vector<Teardown> g_teardowns;
} // namespace

void register_teardown(Teardown teardown) noexcept {
  if (teardown == nullptr) {
    return;
  }
  std::lock_guard lock{g_mutex};
  if (std::ranges::find(g_teardowns, teardown) == g_teardowns.end()) {
    g_teardowns.push_back(teardown);
  }
}

void run_teardowns() noexcept {
  std::vector<Teardown> teardowns;
  {
    std::lock_guard lock{g_mutex};
    teardowns = std::exchange(g_teardowns, {});
  }
  for (auto it = teardowns.rbegin(); it != teardowns.rend(); ++it) {
    (*it)();
  }
}
} // namespace aurora::render::detail::native::d3d12

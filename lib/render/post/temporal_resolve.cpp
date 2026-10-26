#include "temporal_resolve.hpp"

#include "taa.hpp"

#include <algorithm>
#include <atomic>

namespace aurora::render::detail::temporal_resolve {
namespace {
constexpr Resolver TaaResolver{
    .pass = taa::Pass,
    .jitterPhases = 8,
    .wanted = &taa::wanted,
    .jitter_ready = &taa::jitter_enabled,
};

// One resolver owns jitter and the scene colour resolve, so an upscaler replaces TAA instead of running beside it
std::atomic<const Resolver*> g_installed{&TaaResolver};
} // namespace

void install(const Resolver* resolver) noexcept {
  g_installed.store(resolver != nullptr ? resolver : &TaaResolver, std::memory_order_relaxed);
}

const Resolver& installed() noexcept { return *g_installed.load(std::memory_order_relaxed); }

bool jitter_enabled() noexcept {
  const Resolver& resolver = installed();
  return resolver.jitter_ready != nullptr && resolver.jitter_ready();
}

uint32_t jitter_phases() noexcept { return std::max(installed().jitterPhases, 1u); }
} // namespace aurora::render::detail::temporal_resolve

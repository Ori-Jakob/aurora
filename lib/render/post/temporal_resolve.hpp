#pragma once

#include "../graph.hpp"

#include <aurora/render.hpp>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::temporal_resolve {
struct Resolver {
  std::string_view pass;
  uint32_t jitterPhases = 0;
  bool (*wanted)(const Settings& settings) noexcept = nullptr;
  bool (*jitter_ready)() noexcept = nullptr;
  GraphExtent (*scene_size)(GraphExtent output) noexcept = nullptr;
  float (*mip_bias)(float ratio) noexcept = nullptr;
};

void install(const Resolver* resolver) noexcept;
[[nodiscard]] const Resolver& installed() noexcept;
[[nodiscard]] bool jitter_enabled() noexcept;
[[nodiscard]] uint32_t jitter_phases() noexcept;
} // namespace aurora::render::detail::temporal_resolve

#pragma once

#include "ao.hpp"

#include "../graph.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace aurora::render::detail::ao::hiz {
inline constexpr std::string_view Pass = "depth.pyramid";
inline constexpr std::string_view Resource = "ao.hiz";
inline constexpr std::string_view Shader = "ao/hiz";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::R32Float;
inline constexpr uint32_t WorkgroupSize = 8;
inline constexpr uint32_t MaxCachedLevels = MipLevels;
inline constexpr uint64_t RetainFrames = 8;

constexpr TexDesc pyramid_desc(uint32_t divisor) noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
      .mips = 0,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr uint32_t written_levels(uint32_t mipCount) noexcept { return std::min(mipCount, MipLevels); }

constexpr uint32_t level_size(uint32_t mip0, uint32_t level) noexcept {
  return level >= 32 ? 1 : std::max(1u, mip0 >> level);
}

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

PassDesc pass_desc();
void worker_tick(uint64_t frameId) noexcept;
void clear_cache() noexcept;

namespace testing {
struct Counters {
  uint32_t dispatches = 0;
  uint32_t groupsCreated = 0;
  uint32_t cacheEntries = 0;
};
Counters counters() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::ao::hiz

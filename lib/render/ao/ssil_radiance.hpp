#pragma once

#include "ao.hpp"
#include "hiz.hpp"

#include "../graph.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::ao::ssil_radiance {
inline constexpr std::string_view Pass = "ssil.radiance";
inline constexpr std::string_view Resource = "ssil.radiance";
inline constexpr std::string_view Normal = "ssil.normal";
inline constexpr std::string_view Shader = "ao/ssil_radiance";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RGBA16Float;
inline constexpr uint32_t WorkgroupSize = 8;
inline constexpr uint64_t RetainFrames = 8;
inline constexpr size_t CacheCapacity = 4;

constexpr TexDesc radiance_desc(uint32_t divisor) noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
      .mips = 0,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr TexDesc normal_desc(uint32_t divisor) noexcept {
  return {
      .format = SsilNormalFormat,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
      .mips = 0,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr uint32_t written_levels(uint32_t mipCount) noexcept { return hiz::written_levels(mipCount); }

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

PassDesc pass_desc();
void worker_tick(uint64_t frameId) noexcept;
void clear_cache() noexcept;

namespace testing {
struct Counters {
  uint32_t dispatches = 0;
  uint32_t groupsCreated = 0;
  uint32_t cacheEntries = 0;
  uint32_t reprojected = 0;
  uint32_t cleared = 0;
  uint32_t clearedNoRadiance = 0;
  uint32_t clearedNoHistory = 0;
  uint32_t clearedMotion = 0;
  uint32_t clearedUnits = 0;
};
Counters counters() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::ao::ssil_radiance

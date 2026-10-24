#pragma once

#include "../graph.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <string_view>

namespace aurora::render::detail::hzb {
inline constexpr std::string_view Resource = "hzb.depth";
inline constexpr std::string_view OpaquePass = "hzb.opaque";
inline constexpr std::string_view Shader = "geometry/hzb";
inline constexpr std::string_view Library = "aurora/hzb";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::R32Float;
inline constexpr uint32_t WorkgroupSize = 8;
inline constexpr uint32_t MaxLevels = 16;
inline constexpr uint64_t RetainFrames = 8;

struct Footprint {
  uint32_t first = 0;
  uint32_t last = 0;
};

constexpr Footprint footprint(uint32_t texel, uint32_t outSize, uint32_t inSize) noexcept {
  if (inSize == 0) {
    return {1, 0};
  }
  const uint32_t last = texel + 1 == outSize ? inSize - 1 : 2 * texel + 1;
  return {2 * texel, std::min(last, inSize - 1)};
}

constexpr bool reducible(Footprint x, Footprint y) noexcept {
  return x.first <= x.last && y.first <= y.last && x.last - x.first < 3 && y.last - y.first < 3;
}

constexpr uint32_t level_size(uint32_t mip0, uint32_t level) noexcept {
  return level >= 32 ? 1 : std::max(1u, mip0 >> level);
}

constexpr uint32_t texel_of(uint32_t pixel, uint32_t level, uint32_t size) noexcept {
  const uint32_t shifted = level + 1 >= 32 ? 0 : pixel >> (level + 1);
  return std::min(shifted, size - 1);
}

constexpr uint32_t query_level(uint32_t extent, uint32_t levels) noexcept {
  if (extent <= 1 || levels == 0) {
    return 0;
  }
  return std::min(static_cast<uint32_t>(std::bit_width(extent - 1)) - 1, levels - 1);
}

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

constexpr TexDesc pyramid_desc() noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = 2,
      .mips = 0,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

TexRef use(InternalPassBuilder& builder, Missing missing = Missing::BindDummyBlack);
void declare_build(InternalPassBuilder& builder, std::string_view source, std::string_view target);
void request_pipelines(const PipelineSetup& setup, const void* state, std::span<uint32_t, MaxPassPipelines> ids);
void encode_build(PassContext& ctx, const void* state);
PassDesc opaque_pass_desc();
void initialize();
void worker_tick(uint64_t frameId) noexcept;
void shutdown() noexcept;

namespace testing {
struct Counters {
  uint32_t dispatches = 0;
  uint32_t groupsCreated = 0;
  uint32_t cacheEntries = 0;
};
Counters counters() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::hzb

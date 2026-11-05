#pragma once

#include "vsm_cull.hpp"
#include "vsm_pages.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::shadows::vsm {
inline constexpr std::string_view RasterPass = "vsm.raster";
inline constexpr std::string_view RasterShader = "shadows/vsm_raster";
inline constexpr std::string_view CasterShader = "shadows/vsm_caster";
inline constexpr uint32_t MaxClusterTriangles = ChunkTriangles;
inline constexpr float ClearDepth = 0.f;
inline constexpr wgpu::CompareFunction CasterCompare = wgpu::CompareFunction::Greater;
inline constexpr uint32_t ChunkBit = PairChunk;
inline constexpr uint32_t SlotBits = PairSlotBits;
inline constexpr uint32_t RasterFragmentStorage = 1;
inline constexpr uint64_t RasterLayoutKey = 0x76736D7261737401ull;

namespace raster_binding {
inline constexpr uint32_t Vertices = 0;
inline constexpr uint32_t Indices = 1;
inline constexpr uint32_t Instances = 2;
inline constexpr uint32_t Clusters = 3;
inline constexpr uint32_t RenderList = 4;
inline constexpr uint32_t Pairs = 5;
inline constexpr uint32_t Views = 6;
inline constexpr uint32_t Materials = 7;
inline constexpr uint32_t Atlas = 8;
inline constexpr uint32_t Sampler = 9;
inline constexpr uint32_t Count = 10;
inline constexpr uint32_t Buffers = Atlas;
} // namespace raster_binding

struct AtlasLayout {
  uint32_t pagesX = 0;
  uint32_t pagesY = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};
[[nodiscard]] AtlasLayout atlas_layout(uint32_t pages, uint32_t maxDimension) noexcept;

struct CasterRefs {
  uint16_t pool = UINT16_MAX;
  uint16_t views = UINT16_MAX;
  uint16_t renderList = UINT16_MAX;
};
CasterRefs declare_caster_pass(detail::InternalPassBuilder& builder, const detail::PassSetup& setup);
void caster_pipeline(shaders::RenderPipelineDesc& desc, bool cutout) noexcept;
void caster_variant(shaders::RenderPipelineDesc& desc, bool cutout, bool clipDistances, bool unclipped) noexcept;

[[nodiscard]] detail::PassDesc raster_pass_desc();
void raster_shutdown() noexcept;

struct RasterCounts {
  uint64_t ran = 0;
  uint64_t clipVariant = 0;
  uint64_t unclipped = 0;
};

namespace testing {
void set_atlas_handle(std::optional<detail::PersistentHandle> handle) noexcept;
[[nodiscard]] wgpu::BindGroupLayout raster_group_layout();
[[nodiscard]] wgpu::PipelineLayout raster_layout();
[[nodiscard]] RasterCounts raster_counts() noexcept;
} // namespace testing
} // namespace aurora::render::shadows::vsm

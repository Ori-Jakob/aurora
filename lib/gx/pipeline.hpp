#pragma once

#include "../gfx/pipeline_cache.hpp"
#include "gx.hpp"

#include <optional>
#include <span>

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  GXBindGroups bindGroups;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  DrawImmediateData immediateData;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t dstAlpha;
};

constexpr uint32_t GXPipelineConfigVersion = 14;
// Bumped when a stored ext != 0 config would mean or normalize differently; WGSL text changes never need it
constexpr uint32_t RenderShaderRev = 1;
static_assert(GXPipelineConfigVersion < 0x10000);
constexpr uint32_t GXExtRowVersion = (RenderShaderRev << 16) | GXPipelineConfigVersion;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1; // deprecated
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);
// Replaces the colour it covers; reads the remapped config factors, unlike scene::blended's raw state
constexpr bool effectively_opaque(const PipelineConfig& c) noexcept {
  switch (c.blendMode) {
  case GX_BM_NONE:
    return true;
  case GX_BM_BLEND:
    return c.blendFacSrc == GX_BL_ONE && c.blendFacDst == GX_BL_ZERO;
  case GX_BM_LOGIC:
    return c.blendOp == GX_LO_COPY;
  default:
    return false;
  }
}

constexpr uint32_t persisted_version(const PipelineConfig& config) noexcept {
  return config.shaderConfig.ext == RenderExtConfig{} ? config.version : GXExtRowVersion;
}
std::optional<PipelineConfig> migrate_v13_config(std::span<const uint8_t> bytes) noexcept;

struct PipelineOptions {
  DstAlphaMode dstAlphaMode = DstAlphaMode::None;
  bool colorUpdate = false;
  bool alphaUpdate = false;
  bool depthUpdate = false;
};
// The record belongs to the last draw that replaced RGB, so blended writes keep the record below them
constexpr bool writes_fog_record(const PipelineConfig& c, const PipelineOptions& o) noexcept {
  return c.shaderConfig.ext.output != 0 && o.colorUpdate && effectively_opaque(c);
}
enum class ReactiveWrite : uint8_t { None, Clear, Raise };
// A colour writer that replaces the pixel or writes depth leaves something TAA can track, so it clears coverage
constexpr ReactiveWrite reactive_write(const PipelineConfig& c, const PipelineOptions& o) noexcept {
  if (!o.colorUpdate) {
    return ReactiveWrite::None;
  }
  return effectively_opaque(c) || (c.depthCompare && o.depthUpdate) ? ReactiveWrite::Clear : ReactiveWrite::Raise;
}
constexpr ReactiveWrite reactive_write(const PipelineConfig& c) noexcept {
  return reactive_write(c, {.colorUpdate = c.colorUpdate, .alphaUpdate = c.alphaUpdate, .depthUpdate = c.depthUpdate});
}

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout);
uint64_t gbuffer_pipeline_failures() noexcept;
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);
struct BindCounts {
  uint64_t draws, texturedDraws, uniformBinds, textureBinds, indexBinds, encodeNs;
};
void reset_pass_bindings() noexcept;
void add_encode_ns(uint64_t ns) noexcept;
BindCounts bind_counts() noexcept;
struct DstAlphaCounts {
  uint64_t dstAlphaDraws, prepassDraws, dualSourcePipelines, prepassPipelines;
};
DstAlphaCounts dst_alpha_counts() noexcept;

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx

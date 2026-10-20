#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <atomic>

#include <tracy/Tracy.hpp>

namespace aurora::gx {
namespace {
constexpr Module Log{"aurora::gx"};

// Render worker only; render_pass resets it whenever other code may have rebound these slots
struct BoundState {
  uint32_t uniformOffset = UINT32_MAX;
  gfx::BindGroupRef textureGroup = 0;
  bool indexBuffer = false;
};
BoundState g_bound;

struct Counters {
  std::atomic_uint64_t draws, texturedDraws, uniformBinds, textureBinds, indexBinds, encodeNs;
} g_counts;

inline void count(std::atomic_uint64_t& c, uint64_t n = 1) noexcept {
  c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}
} // namespace

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  ZoneScoped;
  const auto hash = xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)));
  const auto create = [&](const PipelineOptions& options, const char* passName) {
    const auto shader = build_shader(config.shaderConfig, layout, options.dstAlphaMode);
    const auto label = fmt::format("GX Pipeline {:x} {}", hash, passName);
    return build_pipeline(config, layout, options, {}, shader, label.c_str());
  };

  PipelineOptions options{
      .colorUpdate = config.colorUpdate,
      .alphaUpdate = config.alphaUpdate,
      .depthUpdate = config.depthUpdate,
  };
  gfx::CompiledPipeline pipeline;
  if (config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0) {
    const auto usesSourceAlpha = [](GXBlendFactor factor) {
      return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
    };
    const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                  (usesSourceAlpha(config.blendFacSrc) || usesSourceAlpha(config.blendFacDst));
    if (!needsSourceAlpha) {
      options.dstAlphaMode = DstAlphaMode::Replace;
    } else if (webgpu::g_dualSourceBlendingSupported && layout.colorAttachmentCount == 1) {
      options.dstAlphaMode = DstAlphaMode::DualSource;
    } else {
      // Write alpha before RGB, no depth write
      auto alpha = options;
      alpha.dstAlphaMode = DstAlphaMode::Replace;
      alpha.colorUpdate = false;
      alpha.depthUpdate = false;
      pipeline.prepass = create(alpha, "alpha");
      if (!pipeline.prepass) {
        return {};
      }
      options.alphaUpdate = false;
    }
  }
  pipeline.main = create(options, "main");
  if (!pipeline.main) {
    return {};
  }
  return pipeline;
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  gfx::CompiledPipeline pipeline;
  if (!gfx::get_pipeline(data.pipeline, pipeline)) {
    return;
  }
  const auto& resources = gfx::detail::resources();
  count(g_counts.draws);
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  if (data.uniformRange.offset != g_bound.uniformOffset) {
    const std::array offsets{data.uniformRange.offset};
    pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
    g_bound.uniformOffset = data.uniformRange.offset;
    count(g_counts.uniformBinds);
  }
  if (const auto group = data.bindGroups.textureBindGroup; group != 0) {
    count(g_counts.texturedDraws);
    if (group != g_bound.textureGroup) {
      pass.SetBindGroup(2, gfx::find_bind_group(group));
      g_bound.textureGroup = group;
      count(g_counts.textureBinds);
    }
  }
  if (data.indexCount != 0) {
    // The whole buffer is bound, so nothing else bounds the read in builds that skip validation
    AURORA_ASSERT(data.idxRange.offset % sizeof(uint16_t) == 0 &&
                      data.indexCount <= data.idxRange.size / sizeof(uint16_t),
                  "GX draw index range offset {} size {} does not hold {} indices", data.idxRange.offset,
                  data.idxRange.size, data.indexCount);
    if (!g_bound.indexBuffer) {
      pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16);
      g_bound.indexBuffer = true;
      count(g_counts.indexBinds);
    }
  }
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  const auto draw = [&] {
    if (data.indexCount == 0) {
      pass.Draw(data.vtxCount, data.instanceCount);
    } else {
      pass.DrawIndexed(data.indexCount, data.instanceCount,
                       static_cast<uint32_t>(data.idxRange.offset / sizeof(uint16_t)));
    }
  };
  if (pipeline.prepass) {
    gfx::bind_pipeline(pipeline.prepass, pass);
    draw();
  }
  gfx::bind_pipeline(pipeline.main, pass);
  draw();
}

void reset_pass_bindings() noexcept { g_bound = {}; }

void add_encode_ns(uint64_t ns) noexcept { count(g_counts.encodeNs, ns); }

BindCounts bind_counts() noexcept {
  return {
      .draws = g_counts.draws.load(std::memory_order_relaxed),
      .texturedDraws = g_counts.texturedDraws.load(std::memory_order_relaxed),
      .uniformBinds = g_counts.uniformBinds.load(std::memory_order_relaxed),
      .textureBinds = g_counts.textureBinds.load(std::memory_order_relaxed),
      .indexBinds = g_counts.indexBinds.load(std::memory_order_relaxed),
      .encodeNs = g_counts.encodeNs.load(std::memory_order_relaxed),
  };
}

} // namespace aurora::gx

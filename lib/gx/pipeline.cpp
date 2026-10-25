#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"
#include "../gfx/scene_targets.hpp"
#include "../webgpu/error_scope.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

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
  std::atomic_uint64_t dstAlphaDraws, prepassDraws;
} g_counts;
// MRT turns dual-source into an alpha prepass; these counts measure that cost
std::atomic_uint64_t g_dualSourcePipelines{0};
std::atomic_uint64_t g_prepassPipelines{0};
std::atomic_uint64_t g_gbufferFailures{0};
std::atomic_bool g_gbufferFailureLogged{false};

inline void count(std::atomic_uint64_t& c, uint64_t n = 1) noexcept {
  c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

gfx::CompiledPipeline create_unchecked(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                                       std::optional<std::string>* moduleError = nullptr) {
  const auto hash = xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)));
  const auto create = [&](const PipelineOptions& options, const char* passName) -> wgpu::RenderPipeline {
    if (moduleError != nullptr) {
      webgpu::push_error_scopes();
    }
    const auto shader = build_shader(config.shaderConfig, layout, options.dstAlphaMode);
    // Release devices skip validation, so no pipeline may be built from a module that failed to compile
    if (moduleError != nullptr && (*moduleError = webgpu::pop_error_scopes())) {
      return {};
    }
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
      g_dualSourcePipelines.fetch_add(1, std::memory_order_relaxed);
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
      g_prepassPipelines.fetch_add(1, std::memory_order_relaxed);
      options.alphaUpdate = false;
    }
  }
  pipeline.main = create(options, "main");
  if (!pipeline.main) {
    return {};
  }
  return pipeline;
}
} // namespace

// The v13 splice depends on these offsets
static_assert(sizeof(PipelineConfig) == 2776);
static_assert(offsetof(PipelineConfig, depthFunc) == 2732);

std::optional<PipelineConfig> migrate_v13_config(std::span<const uint8_t> bytes) noexcept {
  constexpr size_t V13Size = 2772;
  constexpr size_t ExtOffset = offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, ext);
  static_assert(ExtOffset == 2728);
  uint32_t version = 0;
  if (bytes.size() != V13Size) {
    return std::nullopt;
  }
  std::memcpy(&version, bytes.data(), sizeof(version));
  if (version != 13) {
    return std::nullopt;
  }
  std::array<uint8_t, sizeof(PipelineConfig)> spliced{};
  std::memcpy(spliced.data(), bytes.data(), ExtOffset);
  std::memcpy(spliced.data() + ExtOffset + sizeof(RenderExtConfig), bytes.data() + ExtOffset, V13Size - ExtOffset);
  PipelineConfig config;
  std::memcpy(&config, spliced.data(), sizeof(config));
  config.version = GXPipelineConfigVersion;
  return config;
}

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  ZoneScoped;
  // Scene-target text is not golden, so a backend failure must drop the draw instead of ending the process
  if (config.shaderConfig.ext == RenderExtConfig{} && !gfx::layout_has_gbuffer(layout)) {
    return create_unchecked(config, layout);
  }
  // A failed ext variant leaves an entry with no main pipeline, which render() skips and the FIFO replaces with legacy
  std::lock_guard lock{webgpu::error_scope_mutex()};
  webgpu::push_error_scopes();
  std::optional<std::string> moduleError;
  gfx::CompiledPipeline pipeline = create_unchecked(config, layout, &moduleError);
  const std::optional<std::string> error = webgpu::pop_error_scopes();
  if (moduleError || error) {
    const auto hash = xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)));
    const std::string& message = moduleError ? *moduleError : *error;
    if (gfx::layout_has_gbuffer(layout)) {
      g_gbufferFailures.fetch_add(1, std::memory_order_relaxed);
    }
    if (config.shaderConfig.ext != RenderExtConfig{}) {
      Log.warn("GX ext pipeline {:x} failed: {}", hash, message);
    } else if (!g_gbufferFailureLogged.exchange(true, std::memory_order_relaxed)) {
      Log.warn("GX scene-target pipeline {:x} failed: {}", hash, message);
    } else {
      Log.debug("GX scene-target pipeline {:x} failed: {}", hash, message);
    }
    return {};
  }
  return pipeline;
}

uint64_t gbuffer_pipeline_failures() noexcept { return g_gbufferFailures.load(std::memory_order_relaxed); }

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  gfx::CompiledPipeline pipeline;
  if (!gfx::get_pipeline(data.pipeline, pipeline)) {
    // Empty draws (M6-W13) never count
    if (data.instanceCount != 0) {
      gfx::note_skipped_draw();
    }
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
    count(g_counts.dstAlphaDraws);
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
    count(g_counts.prepassDraws);
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

DstAlphaCounts dst_alpha_counts() noexcept {
  return {
      .dstAlphaDraws = g_counts.dstAlphaDraws.load(std::memory_order_relaxed),
      .prepassDraws = g_counts.prepassDraws.load(std::memory_order_relaxed),
      .dualSourcePipelines = g_dualSourcePipelines.load(std::memory_order_relaxed),
      .prepassPipelines = g_prepassPipelines.load(std::memory_order_relaxed),
  };
}

} // namespace aurora::gx

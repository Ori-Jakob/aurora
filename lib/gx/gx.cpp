#include "gx.hpp"

#include "pipeline.hpp"
#include "shader_info.hpp"
#include "texture.hpp"
#include "../dolphin/vi/vi_internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resource_cache.hpp"
#include "../gfx/texture.hpp"
#include "gx_fmt.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <atomic>
#include <bit>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <utility>

static aurora::Module Log("aurora::gx");

namespace aurora::gx {
using webgpu::g_device;
using webgpu::g_graphicsConfig;

GXState g_gxState{};
wgpu::BindGroup g_emptyTextureBindGroup;

namespace {
wgpu::Sampler sEmptySampler;
wgpu::Texture sEmptyTexture;
wgpu::TextureView sEmptyTextureView;
wgpu::Texture sNeutralWhite;
wgpu::Texture sNeutralClear;
wgpu::TextureView sNeutralWhiteView;
wgpu::TextureView sNeutralClearView;
wgpu::BindGroupLayout sTextureBindGroupLayout;
wgpu::PipelineLayout sPipelineLayout;
wgpu::BindGroupLayout sFwdEnvBindGroupLayout;
wgpu::PipelineLayout sFwdEnvPipelineLayout;
wgpu::BindGroup sNeutralFwdEnvGroup;
wgpu::BindGroupLayout sCompanionBindGroupLayout;
wgpu::PipelineLayout sCompanionPipelineLayout;
constexpr uint32_t CompanionFirstBinding = MaxTextures * 2;
constexpr uint32_t CompanionBindings = 8;

std::atomic<int> sPendingViewportPolicy{-1};

template <typename T>
T round_away_from_zero(float value) noexcept {
  return static_cast<T>(value < 0.0f ? std::floor(value) : std::ceil(value));
}

std::pair<f32, f32> polygon_offset_for_cull_mode(GXCullMode cullMode) noexcept {
  if (cullMode == GX_CULL_FRONT) {
    return {g_gxState.backOffset, g_gxState.backScale};
  }
  return {g_gxState.frontOffset, g_gxState.frontScale};
}

wgpu::BlendFactor to_blend_factor(GXBlendFactor fac, bool isDst) {
  switch (fac) {
    DEFAULT_FATAL("invalid blend factor {}", underlying(fac));
  case GX_BL_ZERO:
    return wgpu::BlendFactor::Zero;
  case GX_BL_ONE:
    return wgpu::BlendFactor::One;
  case GX_BL_SRCCLR: // + GX_BL_DSTCLR
    if (isDst) {
      return wgpu::BlendFactor::Src;
    } else {
      return wgpu::BlendFactor::Dst;
    }
  case GX_BL_INVSRCCLR: // + GX_BL_INVDSTCLR
    if (isDst) {
      return wgpu::BlendFactor::OneMinusSrc;
    } else {
      return wgpu::BlendFactor::OneMinusDst;
    }
  case GX_BL_SRCALPHA:
    return wgpu::BlendFactor::SrcAlpha;
  case GX_BL_INVSRCALPHA:
    return wgpu::BlendFactor::OneMinusSrcAlpha;
  case GX_BL_DSTALPHA:
    return wgpu::BlendFactor::DstAlpha;
  case GX_BL_INVDSTALPHA:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

wgpu::CompareFunction to_compare_function(GXCompare func) {
  switch (func) {
    DEFAULT_FATAL("invalid depth fn {}", underlying(func));
  case GX_NEVER:
    return wgpu::CompareFunction::Never;
  case GX_LESS:
    return UseReversedZ ? wgpu::CompareFunction::Greater : wgpu::CompareFunction::Less;
  case GX_EQUAL:
    return wgpu::CompareFunction::Equal;
  case GX_LEQUAL:
    return UseReversedZ ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual;
  case GX_GREATER:
    return UseReversedZ ? wgpu::CompareFunction::Less : wgpu::CompareFunction::Greater;
  case GX_NEQUAL:
    return wgpu::CompareFunction::NotEqual;
  case GX_GEQUAL:
    return UseReversedZ ? wgpu::CompareFunction::LessEqual : wgpu::CompareFunction::GreaterEqual;
  case GX_ALWAYS:
    return wgpu::CompareFunction::Always;
  }
}

wgpu::BlendState to_blend_state(GXBlendMode mode, GXBlendFactor srcFac, GXBlendFactor dstFac, GXLogicOp op,
                                u32 dstAlpha, bool dualSource) {
  wgpu::BlendComponent colorBlendComponent;
  switch (mode) {
    DEFAULT_FATAL("unsupported blend mode {}", underlying(mode));
  case GX_BM_NONE:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::Zero,
    };
    break;
  case GX_BM_BLEND:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor(srcFac, false),
        .dstFactor = to_blend_factor(dstFac, true),
    };
    break;
  case GX_BM_SUBTRACT:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::ReverseSubtract,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::One,
    };
    break;
  case GX_BM_LOGIC:
    switch (op) {
      DEFAULT_FATAL("unsupported logic op {}", underlying(op));
    case GX_LO_CLEAR:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::Zero,
          .dstFactor = wgpu::BlendFactor::Zero,
      };
      break;
    case GX_LO_COPY:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::One,
          .dstFactor = wgpu::BlendFactor::Zero,
      };
      break;
    case GX_LO_NOOP:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::Zero,
          .dstFactor = wgpu::BlendFactor::One,
      };
      break;
    }
    break;
  }
  if (dualSource) {
    const auto remap = [](wgpu::BlendFactor factor) {
      if (factor == wgpu::BlendFactor::SrcAlpha) {
        return wgpu::BlendFactor::Src1Alpha;
      }
      if (factor == wgpu::BlendFactor::OneMinusSrcAlpha) {
        return wgpu::BlendFactor::OneMinusSrc1Alpha;
      }
      return factor;
    };
    colorBlendComponent.srcFactor = remap(colorBlendComponent.srcFactor);
    colorBlendComponent.dstFactor = remap(colorBlendComponent.dstFactor);
  }
  wgpu::BlendComponent alphaBlendComponent;
  if (dstAlpha != UINT32_MAX) {
    alphaBlendComponent = wgpu::BlendComponent{
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::Constant,
        .dstFactor = wgpu::BlendFactor::Zero,
    };
  } else {
    alphaBlendComponent = colorBlendComponent;
  }
  return {
      .color = colorBlendComponent,
      .alpha = alphaBlendComponent,
  };
}

wgpu::ColorWriteMask to_write_mask(bool colorUpdate, bool alphaUpdate) {
  wgpu::ColorWriteMask writeMask = wgpu::ColorWriteMask::None;
  if (colorUpdate) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
  }
  if (alphaUpdate) {
    writeMask |= wgpu::ColorWriteMask::Alpha;
  }
  return writeMask;
}

wgpu::PrimitiveState to_primitive_state(GXCullMode gx_cullMode) {
  auto cullMode = wgpu::CullMode::None;
  switch (gx_cullMode) {
    DEFAULT_FATAL("unsupported cull mode {}", underlying(gx_cullMode));
  case GX_CULL_FRONT:
    cullMode = wgpu::CullMode::Front;
    break;
  case GX_CULL_BACK:
    cullMode = wgpu::CullMode::Back;
    break;
  case GX_CULL_NONE:
    break;
  }
  return {
      .topology = wgpu::PrimitiveTopology::TriangleList,
      .stripIndexFormat = wgpu::IndexFormat::Undefined,
      .frontFace = wgpu::FrontFace::CW,
      .cullMode = cullMode,
  };
}
} // namespace

void set_viewport_policy(AuroraViewportPolicy policy) noexcept {
  sPendingViewportPolicy.store(policy, std::memory_order_release);
}

void update() noexcept {
  if (const int pending = sPendingViewportPolicy.exchange(-1, std::memory_order_acq_rel); pending != -1) {
    const auto policy = static_cast<AuroraViewportPolicy>(pending);
    g_gxState.viewportPolicy = policy;
    window::set_frame_buffer_aspect_fit(policy == AURORA_VIEWPORT_FIT);
  }
}

Vec2<uint32_t> logical_fb_size() noexcept {
  return gfx::is_offscreen() ? gfx::get_render_target_size() : vi::configured_fb_size();
}

namespace {
gfx::Viewport scale_viewport(const gfx::Viewport& viewport, float scaleX, float scaleY) noexcept {
  return {
      .left = viewport.left * scaleX,
      .top = viewport.top * scaleY,
      .width = viewport.width * scaleX,
      .height = viewport.height * scaleY,
      .znear = viewport.znear,
      .zfar = viewport.zfar,
  };
}

gfx::ClipRect scale_scissor(const gfx::ClipRect& scissor, float scaleX, float scaleY, uint32_t targetWidth,
                            uint32_t targetHeight) noexcept {
  const float left = static_cast<float>(scissor.x) * scaleX;
  const float top = static_cast<float>(scissor.y) * scaleY;
  const float right = static_cast<float>(scissor.x + scissor.width) * scaleX;
  const float bottom = static_cast<float>(scissor.y + scissor.height) * scaleY;

  const auto mappedLeft = std::clamp(static_cast<int32_t>(std::floor(left)), 0, static_cast<int32_t>(targetWidth));
  const auto mappedTop = std::clamp(static_cast<int32_t>(std::floor(top)), 0, static_cast<int32_t>(targetHeight));
  const auto mappedRight =
      std::clamp(static_cast<int32_t>(std::ceil(right)), mappedLeft, static_cast<int32_t>(targetWidth));
  const auto mappedBottom =
      std::clamp(static_cast<int32_t>(std::ceil(bottom)), mappedTop, static_cast<int32_t>(targetHeight));

  return {
      .x = mappedLeft,
      .y = mappedTop,
      .width = mappedRight - mappedLeft,
      .height = mappedBottom - mappedTop,
  };
}
} // namespace

Vec2<float> output_pixel_scale(Vec2<uint32_t> target, Vec2<uint32_t> output) noexcept {
  if (target.x == 0 || target.y == 0 || output.x == 0 || output.y == 0 || target == output) {
    return {1.f, 1.f};
  }
  return {static_cast<float>(target.x) / static_cast<float>(output.x),
          static_cast<float>(target.y) / static_cast<float>(output.y)};
}

Vec2<float> output_pixel_scale() noexcept {
  if (gfx::is_offscreen() || !webgpu::scene_split_active()) {
    return {1.f, 1.f};
  }
  const wgpu::Extent3D output = webgpu::g_frameBuffer.size;
  return output_pixel_scale(gfx::get_render_target_size(), {output.width, output.height});
}

gfx::Viewport map_render_viewport(const gfx::Viewport& viewport) noexcept {
  const auto [scaleX, scaleY] = output_pixel_scale();
  return scaleX == 1.f && scaleY == 1.f ? viewport : scale_viewport(viewport, scaleX, scaleY);
}

gfx::ClipRect map_render_scissor(const gfx::ClipRect& scissor) noexcept {
  const auto [scaleX, scaleY] = output_pixel_scale();
  if (scaleX == 1.f && scaleY == 1.f) {
    return scissor;
  }
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  return scale_scissor(scissor, scaleX, scaleY, targetWidth, targetHeight);
}

gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return map_render_viewport(logicalViewport);
  }

  const auto [logicalFbWidth, logicalFbHeight] = logical_fb_size();
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return logicalViewport;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  return scale_viewport(logicalViewport, scaleX, scaleY);
}

gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return map_render_scissor(logicalScissor);
  }

  const auto [logicalFbWidth, logicalFbHeight] = logical_fb_size();
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return logicalScissor;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  return scale_scissor(logicalScissor, scaleX, scaleY, targetWidth, targetHeight);
}

void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  if (viewport.left != g_gxState.logicalViewport.left || viewport.width != g_gxState.logicalViewport.width ||
      viewport.height != g_gxState.logicalViewport.height) {
    g_gxState.dirty |= DirtyUniform;
  }
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}

void set_render_viewport(const gfx::Viewport& viewport) noexcept {
  if (viewport.left != g_gxState.renderViewport.left || viewport.width != g_gxState.renderViewport.width ||
      viewport.height != g_gxState.renderViewport.height) {
    g_gxState.dirty |= DirtyUniform;
  }
  g_gxState.renderViewport = viewport;
  gfx::set_viewport(viewport);
}

void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(g_gxState.logicalScissor));
}

void set_render_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.renderScissor = scissor;
  gfx::set_scissor(scissor);
}

const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[static_cast<size_t>(id)]; }

std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments>
color_target_states(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                    const PipelineOptions& options) noexcept {
  // Coverage only grows, so overlapping translucents keep the strongest
  static const wgpu::BlendState ReactiveBlend{
      .color = {.operation = wgpu::BlendOperation::Max,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Max,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const bool writesDepth = config.depthCompare && options.depthUpdate;
  const ShaderTargets targets = shader_targets(layout);
  const uint32_t normalTarget = normal_target(config.shaderConfig, targets);
  const bool opaque = effectively_opaque(config);
  const bool writesFog = writes_fog_record(config, options);
  const ReactiveWrite reactive = reactive_write(config, options);
  std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> colorTargets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    bool writes = i == normalTarget && writesDepth;
    if (i == targets.debugId) {
      writes = options.colorUpdate;
    } else if (i == targets.motion) {
      writes = writesDepth || (options.colorUpdate && opaque);
    } else if (i == targets.reactive) {
      writes = reactive != ReactiveWrite::None;
    } else if (i == targets.fog) {
      writes = writesFog;
    } else if (i == targets.surface) {
      // Motion's rule; every non-PBR winner writes class 0, so it is never shaded as PBR
      writes = writesDepth || (options.colorUpdate && opaque);
    }
    colorTargets[i] = {
        .format = layout.colorAttachments[i].format,
        .blend = i == targets.reactive && reactive == ReactiveWrite::Raise ? &ReactiveBlend : nullptr,
        .writeMask = writes ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None,
    };
  }
  return colorTargets;
}

wgpu::RenderPipeline build_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                                    const PipelineOptions& options, ArrayRef<wgpu::VertexBufferLayout> vtxBuffers,
                                    wgpu::ShaderModule shader, const char* label) noexcept {
  ZoneScoped;
  const float depthBias = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetBits);
  const float depthBiasSlopeScale = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetScaleBits);
  const float depthBiasClamp = webgpu::g_hasCoreFeatures ? std::bit_cast<float>(config.polygonOffsetClampBits) : 0.0f;
  const bool writesDepth = config.depthCompare && options.depthUpdate;
  const wgpu::DepthStencilState depthStencil{
      .format = layout.depthStencilFormat,
      .depthWriteEnabled = writesDepth,
      .depthCompare = config.depthCompare ? to_compare_function(config.depthFunc) : wgpu::CompareFunction::Always,
      .depthBias = round_away_from_zero<int32_t>(depthBias),
      .depthBiasSlopeScale = depthBiasSlopeScale,
      .depthBiasClamp = depthBiasClamp,
  };
  const auto blendState = to_blend_state(config.blendMode, config.blendFacSrc, config.blendFacDst, config.blendOp,
                                         config.dstAlpha, options.dstAlphaMode == DstAlphaMode::DualSource);
  std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> colorTargets =
      color_target_states(config, layout, options);
  // Integer targets such as the fog record can never blend
  colorTargets[gfx::SceneColorAttachmentIndex].blend = &blendState;
  colorTargets[gfx::SceneColorAttachmentIndex].writeMask = to_write_mask(options.colorUpdate, options.alphaUpdate);
  const wgpu::FragmentState fragmentState{
      .module = shader,
      .entryPoint = "fs_main",
      .targetCount = layout.colorAttachmentCount,
      .targets = colorTargets.data(),
  };
  const ShaderTargets targets = shader_targets(layout);
  CHECK(!(config.shaderConfig.ext.fwdEnv && config.shaderConfig.ext.tessFetch),
        "GX pipeline asks for both the forward environment and tessellation fetch");
  CHECK(!(config.shaderConfig.ext.layer0 && config.shaderConfig.ext.fwdEnv),
        "GX pipeline asks for both companion maps and the forward environment");
  const bool companions = config.shaderConfig.ext.layer0 != 0 &&
                          uses_companions(config.shaderConfig, build_shader_info(config.shaderConfig), targets);
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      // Must follow the generator's test exactly, or pipeline creation fails
      .layout = companions                                       ? sCompanionPipelineLayout
                : uses_forward_env(config.shaderConfig, targets) ? sFwdEnvPipelineLayout
                                                                 : sPipelineLayout,
      .vertex =
          {
              .module = shader,
              .entryPoint = "vs_main",
              .bufferCount = static_cast<uint32_t>(vtxBuffers.size()),
              .buffers = vtxBuffers.data(),
          },
      .primitive = to_primitive_state(config.cullMode),
      .depthStencil = layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &depthStencil : nullptr,
      .multisample = wgpu::MultisampleState{.count = layout.sampleCount},
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt) noexcept {
  ZoneScoped;

  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  config.shaderConfig = {};
  config.shaderConfig.fogType = g_gxState.fog.type;
  config.shaderConfig.fogRangeEnabled = g_gxState.fog.rangeEnabled;
  u8 vtxOffset = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = g_gxState.vtxDesc[i];
    auto& mapping = config.shaderConfig.attrs[i];
    if (type == GX_NONE) {
      mapping = {};
      continue;
    }
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto cnt = comp_cnt_count(attr, attrFmt.cnt);
    const bool nbt3 = attr == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3;
    mapping = AttrConfig{
        .attrType = static_cast<u8>(type),
        .cnt = cnt,
        .compType = static_cast<u8>(attrFmt.type),
        .offset = vtxOffset,
        .stride = 0,
        .frac = attrFmt.frac,
        .le = false,
        .nbt3 = nbt3,
    };
    switch (type) {
    case GX_DIRECT: {
      vtxOffset += comp_type_size(attr, attrFmt.type) * cnt;
      break;
    }
    case GX_INDEX8:
      mapping.stride = g_gxState.arrays[i].stride;
      mapping.le = g_gxState.arrays[i].le;
      vtxOffset += nbt3 ? 3 : 1;
      break;
    case GX_INDEX16:
      mapping.stride = g_gxState.arrays[i].stride;
      mapping.le = g_gxState.arrays[i].le;
      vtxOffset += nbt3 ? 6 : 2;
      break;
    default:
      Log.fatal("populate_pipeline_config: Invalid vertex type {}", type);
    }
  }
  config.shaderConfig.vtxStride = vtxOffset;
  if (primitive == GX_LINES) {
    config.shaderConfig.lineMode = 1;
  } else if (primitive == GX_LINESTRIP) {
    config.shaderConfig.lineMode = 2;
  } else if (primitive == GX_POINTS) {
    config.shaderConfig.lineMode = 3;
  } else {
    config.shaderConfig.lineMode = 0;
  }
  config.shaderConfig.tevSwapTable = g_gxState.tevSwapTable;
  for (u8 i = 0; i < g_gxState.numTevStages; ++i) {
    config.shaderConfig.tevStages[i] = g_gxState.tevStages[i];
  }
  config.shaderConfig.tevStageCount = g_gxState.numTevStages;
  for (u8 i = 0; i < g_gxState.numIndStages; ++i) {
    config.shaderConfig.indStages[i] = g_gxState.indStages[i];
  }
  config.shaderConfig.numIndStages = g_gxState.numIndStages;
  for (u8 i = 0; i < MaxColorChannels; ++i) {
    const auto& cc = g_gxState.colorChannelConfig[i];
    if (cc.lightingEnabled) {
      config.shaderConfig.colorChannels[i] = cc;
    } else {
      // Only matSrc matters when lighting disabled
      config.shaderConfig.colorChannels[i] = {
          .matSrc = cc.matSrc,
      };
    }
  }
  for (u8 i = 0; i < g_gxState.numTexGens; ++i) {
    config.shaderConfig.tcgs[i] = g_gxState.tcgs[i];
  }
  if (g_gxState.alphaCompare) {
    config.shaderConfig.alphaCompare = g_gxState.alphaCompare;
  }
  const auto cullMode = config.shaderConfig.lineMode == 0 ? g_gxState.cullMode : GX_CULL_NONE;
  const auto [polygonOffset, polygonOffsetScale] = polygon_offset_for_cull_mode(cullMode);
  const bool hasAlpha = efb_has_alpha(g_gxState.pixelFmt);
  const bool alphaUpdate = hasAlpha && g_gxState.alphaUpdate;
  const auto blendFactor = [hasAlpha](GXBlendFactor factor) {
    if (!hasAlpha) {
      if (factor == GX_BL_DSTALPHA) {
        return GX_BL_ONE;
      }
      if (factor == GX_BL_INVDSTALPHA) {
        return GX_BL_ZERO;
      }
    }
    return factor;
  };
  config = {
      .shaderConfig = config.shaderConfig,
      .depthFunc = g_gxState.depthFunc,
      .cullMode = cullMode,
      .blendMode = g_gxState.blendMode,
      .blendFacSrc = blendFactor(g_gxState.blendFacSrc),
      .blendFacDst = blendFactor(g_gxState.blendFacDst),
      .blendOp = g_gxState.blendOp,
      .dstAlpha = alphaUpdate ? g_gxState.dstAlpha : UINT32_MAX,
      .polygonOffsetBits = std::bit_cast<uint32_t>(polygonOffset),
      .polygonOffsetScaleBits = std::bit_cast<uint32_t>(polygonOffsetScale),
      .polygonOffsetClampBits = std::bit_cast<uint32_t>(g_gxState.clamp),
      .depthCompare = g_gxState.depthCompare,
      .depthUpdate = g_gxState.depthUpdate,
      .alphaUpdate = alphaUpdate,
      .colorUpdate = g_gxState.colorUpdate,
  };
}

namespace detail {
void fill_texture_entries(const ShaderInfo& info, u8 whiteMask, u8 clearMask, bool baseLevel,
                          const TextureFillViews& views,
                          std::span<WGPUBindGroupEntry, MaxTextures * 2> entries) noexcept {
  for (u32 i = 0; i < MaxTextures; ++i) {
    const auto& tex = g_gxState.textures[i];
    WGPUBindGroupEntry& textureEntry = entries[i * 2];
    WGPUBindGroupEntry& samplerEntry = entries[i * 2 + 1];
    textureEntry.binding = i * 2;
    samplerEntry.binding = i * 2 + 1;
    if (tex && (info.sampledTextures[i] || info.sampledIndTextures[i])) {
      if (((whiteMask | clearMask) >> i & 1u) != 0) {
        textureEntry.textureView = (whiteMask >> i & 1u) != 0 ? views.white : views.clear;
        // A 1x1 constant samples the same at any coordinate or lod, so the default sampler is enough
        samplerEntry.sampler = views.emptySampler;
        continue;
      }
      textureEntry.textureView = tex.ref->sampleTextureView.Get();
      // Indirect offsets keep level 0 too, so generated mips never filter them
      samplerEntry.sampler = gfx::sampler_ref(tex.get_descriptor(baseLevel || info.sampledIndTextures[i])).Get();
    } else {
      textureEntry.textureView = views.empty;
      samplerEntry.sampler = views.emptySampler;
    }
  }
}
} // namespace detail

static GXBindGroups texture_bind_groups(const ShaderInfo& info, u8 whiteMask, u8 clearMask, bool baseLevel,
                                        const char* label) noexcept {
  if (!info.sampledTextures.any() && !info.sampledIndTextures.any()) {
    // Don't bother re-binding anything
    return {};
  }

  // Using C WGPU types instead of C++ wrappers to avoid destructor overhead
  std::array<WGPUBindGroupEntry, MaxTextures * 2> textureEntries{};
  detail::fill_texture_entries(info, whiteMask, clearMask, baseLevel,
                               {
                                   .empty = sEmptyTextureView.Get(),
                                   .white = sNeutralWhiteView.Get(),
                                   .clear = sNeutralClearView.Get(),
                                   .emptySampler = sEmptySampler.Get(),
                               },
                               textureEntries);
  const WGPUBindGroupDescriptor textureBindGroupDescriptor{
      .label = {label, WGPU_STRLEN},
      .layout = sTextureBindGroupLayout.Get(),
      .entryCount = textureEntries.size(),
      .entries = textureEntries.data(),
  };
  return {
      .textureBindGroup = gfx::bind_group_ref(textureBindGroupDescriptor),
  };
}

GXBindGroups build_bind_groups(const ShaderInfo& info, bool baseLevel) noexcept {
  ZoneScoped;
  return texture_bind_groups(info, 0, 0, baseLevel, "GX Texture Bind Group");
}

GXBindGroups build_neutral_bind_groups(const ShaderInfo& info, u8 whiteMask, u8 clearMask, bool baseLevel) noexcept {
  ZoneScoped;
  return texture_bind_groups(info, whiteMask, clearMask, baseLevel, "GX Neutral Texture Bind Group");
}

GXBindGroups build_companion_bind_group(const ShaderInfo& info, bool baseLevel, u8 layerMap,
                                        const std::array<WGPUTextureView, CompanionViewCount>& views) noexcept {
  ZoneScoped;
  if (layerMap >= MaxTextures || !sCompanionBindGroupLayout) {
    return {};
  }
  std::array<WGPUBindGroupEntry, CompanionFirstBinding + CompanionBindings> entries{};
  detail::fill_texture_entries(info, 0, 0, baseLevel,
                               {
                                   .empty = sEmptyTextureView.Get(),
                                   .white = sNeutralWhiteView.Get(),
                                   .clear = sNeutralClearView.Get(),
                                   .emptySampler = sEmptySampler.Get(),
                               },
                               std::span<WGPUBindGroupEntry, MaxTextures * 2>{entries.data(), MaxTextures * 2});
  for (uint32_t i = 0; i < CompanionBindings; ++i) {
    WGPUBindGroupEntry& entry = entries[CompanionFirstBinding + i];
    entry.binding = CompanionFirstBinding + i;
    entry.textureView = i < CompanionViewCount && views[i] != nullptr ? views[i] : sNeutralWhiteView.Get();
  }
  const WGPUBindGroupDescriptor descriptor{
      .label = {"GX Companion Texture Bind Group", WGPU_STRLEN},
      .layout = sCompanionBindGroupLayout.Get(),
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return {.companionBindGroup = gfx::bind_group_ref(descriptor)};
}

void initialize() noexcept {
  {
    std::array<wgpu::BindGroupLayoutEntry, MaxTextures * 2> textureEntries;
    for (u32 i = 0; i < MaxTextures; ++i) {
      textureEntries[i * 2] = {
          .binding = i * 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              {
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      };
      textureEntries[i * 2 + 1] = {
          .binding = i * 2 + 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "GX Texture Bind Group Layout",
        .entryCount = textureEntries.size(),
        .entries = textureEntries.data(),
    };
    sTextureBindGroupLayout = g_device.CreateBindGroupLayout(&descriptor);
  }
  {
    constexpr wgpu::SamplerDescriptor descriptor{.label = "Empty sampler"};
    sEmptySampler = gfx::sampler_ref(descriptor);
  }
  {
    constexpr wgpu::TextureDescriptor descriptor{
        .label = "Empty texture",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    sEmptyTexture = g_device.CreateTexture(&descriptor);
    sEmptyTextureView = sEmptyTexture.CreateView();
  }
  {
    // Constant stand-ins for fake shadow textures; clear keeps RGB at 1 so alpha-blended decals vanish
    const auto create_neutral = [](const char* label, std::array<u8, 4> texel, wgpu::Texture& texture,
                                   wgpu::TextureView& view) {
      const wgpu::TextureDescriptor descriptor{
          .label = label,
          .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
          .size = {1, 1},
          .format = wgpu::TextureFormat::RGBA8Unorm,
      };
      texture = g_device.CreateTexture(&descriptor);
      const wgpu::TexelCopyTextureInfo destination{.texture = texture};
      const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = 4, .rowsPerImage = 1};
      const wgpu::Extent3D size{1, 1};
      webgpu::g_queue.WriteTexture(&destination, texel.data(), texel.size(), &layout, &size);
      view = texture.CreateView();
    };
    create_neutral("GX Neutral White", {255, 255, 255, 255}, sNeutralWhite, sNeutralWhiteView);
    create_neutral("GX Neutral Clear", {255, 255, 255, 0}, sNeutralClear, sNeutralClearView);
  }
  {
    std::array<wgpu::BindGroupEntry, MaxTextures * 2> entries;
    for (u32 i = 0; i < MaxTextures; ++i) {
      entries[i * 2] = {
          .binding = i * 2,
          .textureView = sEmptyTextureView,
      };
      entries[i * 2 + 1] = {
          .binding = i * 2 + 1,
          .sampler = sEmptySampler,
      };
    }
    const wgpu::BindGroupDescriptor desc{
        .label = "GX Empty Texture Bind Group",
        .layout = sTextureBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_emptyTextureBindGroup = g_device.CreateBindGroup(&desc);
  }
  {
    const std::array layouts{
        gfx::detail::resources().staticBindGroupLayout,
        gfx::detail::resources().uniformBindGroupLayout,
        sTextureBindGroupLayout,
    };
    const wgpu::PipelineLayoutDescriptor desc{
        .label = "GX Pipeline Layout",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
        .immediateSize = sizeof(DrawImmediateData),
    };
    sPipelineLayout = g_device.CreatePipelineLayout(&desc);
  }
  {
    // Binding numbers are append-only; later screen factors bind after 0
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            {
                .sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
    };
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "GX Forward Environment Bind Group Layout",
        .entryCount = 1,
        .entries = &entry,
    };
    sFwdEnvBindGroupLayout = g_device.CreateBindGroupLayout(&descriptor);
    const std::array layouts{
        gfx::detail::resources().staticBindGroupLayout,
        gfx::detail::resources().uniformBindGroupLayout,
        sTextureBindGroupLayout,
        sFwdEnvBindGroupLayout,
    };
    const wgpu::PipelineLayoutDescriptor desc{
        .label = "GX Forward Environment Pipeline Layout",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
        .immediateSize = sizeof(DrawImmediateData),
    };
    sFwdEnvPipelineLayout = g_device.CreatePipelineLayout(&desc);
    const wgpu::BindGroupEntry groupEntry{.binding = 0, .textureView = sNeutralWhiteView};
    const wgpu::BindGroupDescriptor group{
        .label = "GX Neutral Forward Environment Bind Group",
        .layout = sFwdEnvBindGroupLayout,
        .entryCount = 1,
        .entries = &groupEntry,
    };
    sNeutralFwdEnvGroup = g_device.CreateBindGroup(&group);
  }
  {
    // 16 albedo, 17 nrm or .mat, 18 orm, 19 emit, 20-23 layer 1; append-only
    std::array<wgpu::BindGroupLayoutEntry, CompanionFirstBinding + CompanionBindings> entries;
    for (u32 i = 0; i < MaxTextures; ++i) {
      entries[i * 2] = {
          .binding = i * 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              {
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      };
      entries[i * 2 + 1] = {
          .binding = i * 2 + 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      };
    }
    for (u32 i = 0; i < CompanionBindings; ++i) {
      entries[CompanionFirstBinding + i] = {
          .binding = CompanionFirstBinding + i,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              {
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "GX Companion Texture Bind Group Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    sCompanionBindGroupLayout = g_device.CreateBindGroupLayout(&descriptor);
    const std::array layouts{
        gfx::detail::resources().staticBindGroupLayout,
        gfx::detail::resources().uniformBindGroupLayout,
        sCompanionBindGroupLayout,
    };
    // render() sets immediates on every draw, so every GX layout needs them
    const wgpu::PipelineLayoutDescriptor desc{
        .label = "GX Companion Pipeline Layout",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
        .immediateSize = sizeof(DrawImmediateData),
    };
    sCompanionPipelineLayout = g_device.CreatePipelineLayout(&desc);
  }
}

const wgpu::BindGroupLayout& forward_environment_layout() noexcept { return sFwdEnvBindGroupLayout; }

const wgpu::BindGroup& detail::neutral_forward_environment_group() noexcept { return sNeutralFwdEnvGroup; }

void shutdown() noexcept {
  // Backstop: the last published group must not outlive the device
  set_forward_environment_group({});
  // TODO we should probably store this all in g_state.gx instead
  sTextureBindGroupLayout = {};
  sNeutralFwdEnvGroup = {};
  sFwdEnvPipelineLayout = {};
  sFwdEnvBindGroupLayout = {};
  sCompanionPipelineLayout = {};
  sCompanionBindGroupLayout = {};
  sNeutralWhiteView = {};
  sNeutralClearView = {};
  sNeutralWhite = {};
  sNeutralClear = {};
  for (auto& item : g_gxState.textures) {
    item.ref.reset();
  }
  g_gxState.loadedTextures.fill({});
  g_gxState.loadedTluts.fill({});
  texture::shutdown();
}
} // namespace aurora::gx

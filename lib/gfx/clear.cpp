#include "clear.hpp"

#include "encoding.hpp"
#include "scene_targets.hpp"
#include "../webgpu/gpu.hpp"
#include "tracy/Tracy.hpp"

namespace aurora::gfx::clear {
using webgpu::g_device;

namespace {
wgpu::ColorWriteMask clear_write_mask(bool clearColor, bool clearAlpha) {
  auto writeMask = wgpu::ColorWriteMask::None;
  if (clearColor) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
  }
  if (clearAlpha) {
    writeMask |= wgpu::ColorWriteMask::Alpha;
  }
  return writeMask;
}

bool clears_target(const PipelineConfig& config, ColorAttachmentSemantic semantic) {
  return (config.clearDepth && clears_with_depth(semantic)) || (config.clearColor && clears_with_color(semantic));
}
} // namespace

namespace detail {
std::string shader_source(const PipelineConfig& config, const RenderTargetLayout& layout) {
  std::string source{R"""(
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, 1.0),
    vec2(-1.0, -3.0),
    vec2(3.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vtxIdx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[vtxIdx], 0.0, 1.0);
    return out;
}
)"""};

  std::string outputs;
  std::string values;
  if (config.clearColor || config.clearAlpha) {
    outputs += fmt::format("    @location({}) color: vec4f,\n", SceneColorAttachmentIndex);
    values += "    out.color = vec4f(1.0);\n";
  }
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    const ColorAttachmentSemantic semantic = layout.colorAttachments[i].semantic;
    if (!clears_target(config, semantic)) {
      continue;
    }
    if (semantic == ColorAttachmentSemantic::Normal) {
      outputs += fmt::format("    @location({0}) normal{0}: vec4f,\n", i);
      values += fmt::format("    out.normal{} = vec4f(0.0);\n", i);
    } else if (semantic == ColorAttachmentSemantic::Fog) {
      outputs += fmt::format("    @location({0}) t{0}: vec4u,\n", i);
      values += fmt::format("    out.t{} = vec4u(0u);\n", i);
    } else {
      outputs += fmt::format("    @location({0}) t{0}: vec4f,\n", i);
      values += fmt::format("    out.t{} = vec4f(0.0);\n", i);
    }
  }
  if (!outputs.empty()) {
    source += fmt::format(R"""(
struct FragmentOutput {{
{0}}};
@fragment
fn fs_main() -> FragmentOutput {{
    var out: FragmentOutput;
{1}    return out;
}}
)""",
                          outputs, values);
  } else {
    source += R"""(
@fragment
fn fs_main() {
}
)""";
  }
  return source;
}
} // namespace detail

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config, const RenderTargetLayout& layout) {
  ZoneScoped;
  const auto source = detail::shader_source(config, layout);
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = source.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "EFB Clear Module",
  };
  auto module = g_device.CreateShaderModule(&moduleDescriptor);
  constexpr wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 0,
      .bindGroupLayouts = nullptr,
  };
  auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);
  constexpr wgpu::BlendState blendState{
      .color =
          wgpu::BlendComponent{
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::Constant,
              .dstFactor = wgpu::BlendFactor::Zero,
          },
      .alpha =
          wgpu::BlendComponent{
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::Constant,
              .dstFactor = wgpu::BlendFactor::Zero,
          },
  };
  std::array<wgpu::ColorTargetState, MaxColorAttachments> colorTargets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    colorTargets[i] = {
        .format = layout.colorAttachments[i].format,
        .writeMask = clears_target(config, layout.colorAttachments[i].semantic) ? wgpu::ColorWriteMask::All
                                                                                : wgpu::ColorWriteMask::None,
    };
  }
  colorTargets[SceneColorAttachmentIndex].blend = &blendState;
  colorTargets[SceneColorAttachmentIndex].writeMask = clear_write_mask(config.clearColor, config.clearAlpha);
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = layout.colorAttachmentCount,
      .targets = colorTargets.data(),
  };
  const wgpu::DepthStencilState depthStencil{
      .format = layout.depthStencilFormat,
      .depthWriteEnabled = config.clearDepth,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  const auto label = fmt::format("EFB Clear Pipeline (color {}, alpha {}, depth {})", config.clearColor,
                                 config.clearAlpha, config.clearDepth);
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = label.c_str(),
      .layout = pipelineLayout,
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &depthStencil : nullptr,
      .multisample = wgpu::MultisampleState{.count = layout.sampleCount},
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&pipelineDescriptor);
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass, const wgpu::Extent3D& targetSize) {
  if (!bind_pipeline(data.pipeline, pass)) {
    return;
  }

  pass.SetBlendConstant(&data.color);
  pass.SetViewport(0.f, 0.f, static_cast<float>(targetSize.width), static_cast<float>(targetSize.height), data.depth,
                   data.depth);
  pass.SetScissorRect(data.rect.x, data.rect.y, data.rect.width, data.rect.height);
  pass.Draw(3);
}
} // namespace aurora::gfx::clear

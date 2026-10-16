#include "graph_passes.hpp"

#include "graph_checks.hpp"

#include <aurora/gfx.hpp>
#include <aurora/render_graph.hpp>

#include <atomic>
#include <string>
#include <string_view>

namespace render_demo {
namespace {
using aurora::render::Access;
using aurora::render::PassBuilder;
using aurora::render::PassContext;
using aurora::render::PassId;

constexpr std::string_view VignetteWgsl = R"(
struct VignetteParams {
  strength: f32,
  inner: f32,
  outer: f32,
  pad: f32,
};
@group(0) @binding(1) var<uniform> params: VignetteParams;

var<private> positions: array<vec2f, 3> = array(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(positions[vi], 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * aurora_frame.inv_efb_size;
  let d = length(uv - vec2f(0.5)) * 1.41421356;
  let f = 1.0 - params.strength * smoothstep(params.inner, params.outer, d);
  return vec4f(f, f, f, 1.0);
}
)";

struct VignettePass {
  std::string anchor;
  PassId id = aurora::render::InvalidPass;
  std::atomic_uint32_t encodes = 0;
  // Render worker only
  wgpu::RenderPipeline pipeline;
  wgpu::PipelineLayout layout;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  u32 samples = 0;
};

VignettePass sVignette;

wgpu::RenderPipeline create_vignette_pipeline(const PassContext& ctx, wgpu::ColorTargetState target, u32 samples) {
  const std::string code = std::string{aurora::render::FrameWgsl} + std::string{VignetteWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl, .label = "demo.vignette"};
  const wgpu::ShaderModule module = ctx.device().CreateShaderModule(&moduleDescriptor);
  // Multiplies the scene by the shader's factor, so the pass never has to sample gx.color
  const wgpu::BlendState blend{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  target.blend = &blend;
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "demo.vignette",
      .layout = ctx.pipeline_layout(),
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples},
      .fragment = &fragment,
  };
  return ctx.device().CreateRenderPipeline(&descriptor);
}

void declare_vignette(PassBuilder& builder, void* userdata) {
  const auto& pass = *static_cast<const VignettePass*>(userdata);
  builder.use(aurora::render::resource::GxColor, Access::ColorRMW);
  if (!pass.anchor.empty()) {
    builder.after(pass.anchor);
  }
}

void encode_vignette(PassContext& ctx, void* userdata) {
  auto& pass = *static_cast<VignettePass*>(userdata);
  constexpr aurora::render::TexRef Color{0};
  const wgpu::ColorTargetState target = ctx.color_target(Color);
  const u32 samples = ctx.sample_count(Color);
  if (!pass.pipeline || pass.format != target.format || pass.samples != samples ||
      pass.layout.Get() != ctx.pipeline_layout().Get()) {
    pass.pipeline = create_vignette_pipeline(ctx, target, samples);
    pass.layout = ctx.pipeline_layout();
    pass.format = target.format;
    pass.samples = samples;
  }
  const wgpu::RenderPassEncoder render = ctx.begin_render();
  if (!render) {
    return;
  }
  render.SetPipeline(pass.pipeline);
  render.Draw(3);
  render.End();
  // Counted only once the pass began and ended, so N/N proves every frame drew the vignette
  pass.encodes.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

bool register_graph_passes(const GraphPassOptions& options, std::string& error) {
  if (!options.vignette) {
    return true;
  }
  sVignette.anchor = options.vignetteAnchor;
  sVignette.id = aurora::render::register_pass({
      .name = "demo.vignette",
      .stage = aurora::render::Stage::BeforeHud,
      .kind = aurora::render::PassKind::Render,
      .paramsSize = sizeof(VignetteParams),
      .declare = declare_vignette,
      .encode = encode_vignette,
      .userdata = &sVignette,
  });
  if (sVignette.id == aurora::render::InvalidPass) {
    error = "register_pass refused demo.vignette";
    return false;
  }
  return true;
}

void update_graph_passes(u32 frame) {
  if (sVignette.id == aurora::render::InvalidPass) {
    return;
  }
  const VignetteParams params = vignette_params(frame);
  aurora::render::set_pass_params(sVignette.id, &params, sizeof(params));
}

void finish_graph_passes() {}

GraphPassReport graph_pass_report() { return {.vignetteEncodes = sVignette.encodes.load(std::memory_order_relaxed)}; }

void unregister_graph_passes() {
  if (sVignette.id == aurora::render::InvalidPass) {
    return;
  }
  aurora::render::unregister_pass(sVignette.id);
  sVignette.id = aurora::render::InvalidPass;
  // The worker may still be inside encode, so the pipeline outlives it until the queue drains
  aurora::gfx::synchronize();
  sVignette.pipeline = {};
  sVignette.layout = {};
}
} // namespace render_demo

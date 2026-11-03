#include "surface_expand.hpp"

#include "../pass_context.hpp"
#include "../shaders.hpp"

#include <aurora/render_graph.hpp>

#include <string>

namespace aurora::render::debug::surface_expand {
namespace {
constexpr TexRef Target{3};

detail::PassId g_pass = detail::InvalidPassId;

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

// Only views, probes and captures read v1 words, so without one of them the pass is culled
void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!setup.gx_available(detail::GxResource::Surface) || !setup.gx_available(detail::GxResource::SurfaceExt)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.use(resource::Surface, Access::Sampled,
              detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use(resource::SurfaceExt, Access::Sampled,
              detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use_buffer(resource::SurfaceTable, Access::StorageRead, detail::UseOptions{.missing = Missing::DisablePass});
  builder.create(resource::SurfaceV1, {.format = wgpu::TextureFormat::RGBA32Uint, .sizeClass = SizeClass::Efb},
                 Access::StorageWrite);
}

void request_pipelines(const detail::PipelineSetup& setup, const void*,
                       std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "debug.surface_expand",
                                        .shader = Shader,
                                        .entryPoint = "expand",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode(detail::PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::Extent3D extent = ctx.extent(Target);
  if (!pipeline || extent.width == 0 || extent.height == 0) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(groups(extent.width), groups(extent.height));
  pass.End();
}
} // namespace

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .declare = declare,
      .encode = encode,
      .pipelines = request_pipelines,
  };
}

void initialize() {
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
}

void shutdown() noexcept {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
}
} // namespace aurora::render::debug::surface_expand

#include "upsample.hpp"

#include "denoise.hpp"
#include "hiz.hpp"

#include "../pass_context.hpp"

#include <atomic>
#include <span>
#include <string>

namespace aurora::render::detail::ao::upsample {
namespace {
std::atomic_uint32_t g_draws{0};

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Ao);
  constexpr UseOptions Unfilterable{.missing = Missing::DisablePass,
                                    .sampleType = wgpu::TextureSampleType::UnfilterableFloat};
  builder.use(denoise::History, Access::Sampled, Missing::DisablePass);
  builder.use(hiz::Resource, Access::Sampled, Unfilterable);
  builder.use(resource::GxDepth, Access::Sampled, Unfilterable);
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  builder.create(resource::AoFinal, final_desc(), Access::ColorWrite);
  if (sink_forced()) {
    // Dev timing only: keeps the chain alive before the composite consumes ao.final
    builder.side_effect();
  }
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  shaders::RenderPipelineDesc desc = testing::pipeline_desc();
  desc.layout = setup.layout;
  desc.layoutKey = setup.layoutHash;
  ids[0] = shaders::request_render(desc).value;
}

void encode(PassContext& ctx, const void*) {
  const wgpu::RenderPipeline pipeline = shaders::render_pipeline(ctx.pipeline(0));
  if (!pipeline) {
    return;
  }
  const wgpu::RenderPassEncoder pass = ctx.begin_render();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.Draw(3);
  pass.End();
  g_draws.store(g_draws.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Render,
      .cls = PassClass::Required,
      .order = 2,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = scene_ready,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_camera,
  };
}

namespace testing {
Counters counters() noexcept { return {.draws = g_draws.load(std::memory_order_relaxed)}; }

void reset() noexcept { g_draws.store(0, std::memory_order_relaxed); }

shaders::RenderPipelineDesc pipeline_desc() {
  shaders::RenderPipelineDesc desc{
      .label = "ao.upsample",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .colorCount = 1,
  };
  desc.colors[0] = {.format = Format};
  return desc;
}
} // namespace testing
} // namespace aurora::render::detail::ao::upsample

#include "upsample.hpp"

#include "denoise.hpp"
#include "hiz.hpp"

#include "../pass_context.hpp"

#include <array>
#include <atomic>
#include <span>
#include <string>

namespace aurora::render::detail::ao::upsample {
namespace {
constexpr std::array<std::string_view, 1> SsilDefines{"AO_SSIL"};

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
  if (ssil_wanted(setup)) {
    builder.use(denoise::SsilHistory, Access::Sampled, Missing::BindDummyBlack);
    builder.create(resource::SsilIrradiance, irradiance_desc(), Access::ColorWrite);
  }
  if (sink_forced()) {
    // Dev timing only: keeps the chain alive before the composite consumes ao.final
    builder.side_effect();
  }
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  shaders::RenderPipelineDesc desc = testing::pipeline_desc(setup.pass != nullptr && ssil_wanted(*setup.pass));
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

shaders::RenderPipelineDesc pipeline_desc(bool ssil) {
  shaders::RenderPipelineDesc desc{
      .label = "ao.upsample",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .colorCount = ssil ? 2u : 1u,
  };
  desc.colors[0] = {.format = Format};
  if (ssil) {
    desc.defines = SsilDefines;
    desc.colors[1] = {.format = SsilFormat};
  }
  return desc;
}
} // namespace testing
} // namespace aurora::render::detail::ao::upsample

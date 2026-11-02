#include "gtao.hpp"

#include "hiz.hpp"

#include "../pass_context.hpp"
#include "../shaders.hpp"

#include "../../gx/gx.hpp"

#include <atomic>
#include <span>
#include <string>

namespace aurora::render::detail::ao::gtao {
namespace {
static_assert(gx::UseReversedZ);

constexpr TexRef Target{2};

std::atomic_uint32_t g_dispatches{0};

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Ao);
  builder.use(hiz::Resource, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  builder.create(Resource, raw_desc(working_divisor(setup.settings->ao)), Access::StorageWrite);
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "ao.gtao",
                                        .shader = Shader,
                                        .entryPoint = "main",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  if (!pipeline) {
    return;
  }
  const wgpu::Extent3D size = ctx.extent(Target);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(groups(size.width), groups(size.height));
  pass.End();
  g_dispatches.store(g_dispatches.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 0,
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
Counters counters() noexcept { return {.dispatches = g_dispatches.load(std::memory_order_relaxed)}; }

void reset() noexcept { g_dispatches.store(0, std::memory_order_relaxed); }
} // namespace testing
} // namespace aurora::render::detail::ao::gtao

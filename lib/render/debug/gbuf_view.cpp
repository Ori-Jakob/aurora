#include "gbuf_view.hpp"

#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <optional>
#include <span>
#include <string>

namespace aurora::render::debug::gbuf_view {
namespace {
constexpr std::array<std::string_view, 1> UintDefines{"SOURCE_UINT"};
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
constexpr std::array<shaders::Constant, 7> ModeConstants{{
    {"VIEW_MODE", 0},
    {"VIEW_MODE", 1},
    {"VIEW_MODE", 2},
    {"VIEW_MODE", 3},
    {"VIEW_MODE", 4},
    {"VIEW_MODE", 5},
    {"VIEW_MODE", 6},
}};

std::atomic_uint64_t g_generation{0};
const GbufView* g_view = nullptr;
uint16_t g_selected = 0;
detail::PassId g_snapshot = detail::InvalidPassId;
detail::PassId g_blit = detail::InvalidPassId;

void store_generation(detail::InternalPassBuilder& builder) {
  const uint64_t generation = g_generation.load(std::memory_order_relaxed);
  std::memcpy(builder.local().data(), &generation, sizeof(generation));
}

bool current(const detail::PassContext& ctx) noexcept {
  uint64_t generation = 0;
  std::memcpy(&generation, ctx.local().data(), sizeof(generation));
  return generation == g_generation.load(std::memory_order_relaxed);
}

bool source_available(const GbufView& view, const detail::PassSetup& setup) noexcept {
  const std::optional<detail::GxResource> gx = detail::find_gx_resource(view.resource);
  return gx && setup.gx_available(*gx);
}

void declare_snapshot(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void* state) {
  const auto& view = *static_cast<const GbufView*>(state);
  if (!source_available(view, setup)) {
    builder.skip();
    return;
  }
  store_generation(builder);
  builder.after(view.anchor);
  builder.use(
      view.resource, Access::Sampled,
      detail::UseOptions{
          .missing = Missing::DisablePass,
          .sampleType = view.uintSource ? wgpu::TextureSampleType::Uint : wgpu::TextureSampleType::UnfilterableFloat,
      });
  builder.create(SnapshotResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                 Access::ColorWrite);
}

bool snapshot_enabled(const detail::StageData& stage, const void*) {
  return stage.markerSeen && stage.sceneHints != nullptr && stage.sceneHints->has(detail::SceneHasMainCamera);
}

void snapshot_pipelines(const detail::PipelineSetup& setup, const void* state,
                        std::span<uint32_t, detail::MaxPassPipelines> ids) {
  shaders::RenderPipelineDesc desc = snapshot_pipeline_desc(*static_cast<const GbufView*>(state));
  desc.layout = setup.layout;
  desc.layoutKey = setup.layoutHash;
  ids[0] = shaders::request_render(desc).value;
}

void draw_fullscreen(detail::PassContext& ctx) {
  if (!current(ctx)) {
    return;
  }
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
}

void encode_snapshot(detail::PassContext& ctx, const void*) { draw_fullscreen(ctx); }

void declare_blit(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void* state) {
  if (!source_available(*static_cast<const GbufView*>(state), setup)) {
    builder.skip();
    return;
  }
  store_generation(builder);
  builder.after(phase::BeforeHud);
  builder.use(
      SnapshotResource, Access::Sampled,
      detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::GxColor, Access::ColorRMW);
}

void blit_pipelines(const detail::PipelineSetup& setup, const void*,
                    std::span<uint32_t, detail::MaxPassPipelines> ids) {
  if (setup.pass == nullptr || setup.pass->env == nullptr) {
    return;
  }
  shaders::RenderPipelineDesc desc = blit_pipeline_desc(*setup.pass->env);
  desc.layout = setup.layout;
  desc.layoutKey = setup.layoutHash;
  ids[0] = shaders::request_render(desc).value;
}

void encode_blit(detail::PassContext& ctx, const void*) { draw_fullscreen(ctx); }

void unregister() noexcept {
  if (g_snapshot != detail::InvalidPassId) {
    detail::unregister_pass(g_snapshot);
    g_snapshot = detail::InvalidPassId;
  }
  if (g_blit != detail::InvalidPassId) {
    detail::unregister_pass(g_blit);
    g_blit = detail::InvalidPassId;
  }
}
} // namespace

shaders::RenderPipelineDesc snapshot_pipeline_desc(const GbufView& view) {
  shaders::RenderPipelineDesc desc{
      .label = SnapshotPass,
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_view",
      .defines = view.uintSource ? std::span<const std::string_view>{UintDefines} : std::span<const std::string_view>{},
      .constants = std::span{&ModeConstants[std::min<size_t>(view.mode, ModeConstants.size() - 1)], 1},
      .colorCount = 1,
  };
  desc.colors[0] = {.format = wgpu::TextureFormat::RGBA8Unorm};
  return desc;
}

shaders::RenderPipelineDesc blit_pipeline_desc(const detail::GraphEnvironment& env) {
  shaders::RenderPipelineDesc desc{
      .label = BlitPass,
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_blit",
      .colorCount = 1,
      .sampleCount = env.gxSampleCount,
  };
  // EFB alpha is the game's shadow stencil, so the view replaces RGB only
  desc.colors[0] = {.format = env.gxFormats[static_cast<size_t>(detail::GxResource::Color)], .writeMask = RgbMask};
  return desc;
}

void on_latch(uint16_t selected) {
  if (selected == g_selected) {
    return;
  }
  g_selected = selected;
  unregister();
  g_generation.fetch_add(1, std::memory_order_relaxed);
  g_view = find_gbuf_view(selected);
  if (g_view == nullptr) {
    return;
  }
  g_snapshot = detail::register_pass({
      .name = std::string{SnapshotPass},
      .stage = g_view->stage,
      .kind = PassKind::Render,
      .cls = detail::PassClass::Optional,
      .order = LastPassOrder,
      .declare = declare_snapshot,
      .enabled = snapshot_enabled,
      .encode = encode_snapshot,
      .pipelines = snapshot_pipelines,
      .state = g_view,
  });
  g_blit = detail::register_pass({
      .name = std::string{BlitPass},
      .stage = Stage::BeforeHud,
      .kind = PassKind::Render,
      .cls = detail::PassClass::Optional,
      .order = LastPassOrder,
      .declare = declare_blit,
      .encode = encode_blit,
      .pipelines = blit_pipelines,
      .state = g_view,
  });
}

void shutdown() noexcept {
  unregister();
  g_generation.fetch_add(1, std::memory_order_relaxed);
  g_view = nullptr;
  g_selected = 0;
}

const GbufView* active_view() noexcept { return g_view; }

uint64_t generation() noexcept { return g_generation.load(std::memory_order_relaxed); }
} // namespace aurora::render::debug::gbuf_view

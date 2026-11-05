#include "vsm_raster.hpp"

#include "vsm_alloc.hpp"
#include "vsm_invalidate.hpp"

#include "../graph_bind.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../scene/atlas.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>

namespace aurora::render::shadows::vsm {
namespace {
using detail::PassSetup;
using detail::PersistentHandle;
using detail::StageData;

constexpr Module Log{"aurora::render::shadows"};
constexpr uint16_t NoRef = UINT16_MAX;
constexpr std::string_view ClipDefine = "VSM_CLIP_DISTANCES";
constexpr std::string_view CutoutDefine = "VSM_CUTOUT";
constexpr std::string_view UnclippedDefine = "VSM_UNCLIPPED";

struct Variant {
  std::array<std::string_view, 3> defines;
  uint32_t count = 0;
};

constexpr std::array<Variant, 8> Variants{{
    {{}, 0},
    {{UnclippedDefine}, 1},
    {{ClipDefine}, 1},
    {{ClipDefine, UnclippedDefine}, 2},
    {{CutoutDefine}, 1},
    {{CutoutDefine, UnclippedDefine}, 2},
    {{ClipDefine, CutoutDefine}, 2},
    {{ClipDefine, CutoutDefine, UnclippedDefine}, 3},
}};

struct RasterLocal {
  CasterRefs caster;
  std::array<uint16_t, 6> buffers{NoRef, NoRef, NoRef, NoRef, NoRef, NoRef};
  uint16_t args = NoRef;
  uint16_t atlas = NoRef;
};
static_assert(sizeof(RasterLocal) <= detail::PassLocalSize);

enum BufferRef : uint32_t { VerticesRef, IndicesRef, InstancesRef, ClustersRef, PairsRef, MaterialsRef };

struct RasterBinding {
  wgpu::BindGroupLayout groupLayout;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::Sampler sampler;
  wgpu::BindGroup group;
  std::array<WGPUBuffer, raster_binding::Buffers> buffers{};
  std::array<uint64_t, raster_binding::Buffers> offsets{};
  std::array<uint64_t, raster_binding::Buffers> sizes{};
  WGPUTextureView atlas = nullptr;
};
RasterBinding g_raster;
std::optional<PersistentHandle> g_atlasOverride;
std::atomic_uint64_t g_ran{0};
std::atomic_uint64_t g_clipRuns{0};
std::atomic_uint64_t g_unclippedRuns{0};

PersistentHandle atlas_handle() noexcept { return g_atlasOverride.value_or(scene::atlas::page_handle()); }

bool raster_layouts() {
  if (g_raster.pipelineLayout) {
    return true;
  }
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return false;
  }
  std::array<wgpu::BindGroupLayoutEntry, raster_binding::Count> entries{};
  for (uint32_t i = 0; i < raster_binding::Materials; ++i) {
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Vertex,
        .buffer = {.type = i == raster_binding::Views ? wgpu::BufferBindingType::Uniform
                                                      : wgpu::BufferBindingType::ReadOnlyStorage},
    };
  }
  entries[raster_binding::Materials] = {
      .binding = raster_binding::Materials,
      .visibility = wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
  };
  entries[raster_binding::Atlas] = {
      .binding = raster_binding::Atlas,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
  };
  entries[raster_binding::Sampler] = {
      .binding = raster_binding::Sampler,
      .visibility = wgpu::ShaderStage::Fragment,
      .sampler = {.type = wgpu::SamplerBindingType::Filtering},
  };
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{
      .label = "vsm.raster", .entryCount = entries.size(), .entries = entries.data()};
  g_raster.groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "vsm.raster", .bindGroupLayoutCount = 1, .bindGroupLayouts = &g_raster.groupLayout};
  g_raster.pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  const wgpu::SamplerDescriptor sampler{
      .label = "vsm.raster",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_raster.sampler = device.CreateSampler(&sampler);
  return static_cast<bool>(g_raster.pipelineLayout);
}

const wgpu::BindGroup& raster_group(const std::array<BufferView, raster_binding::Buffers>& views,
                                    const wgpu::TextureView& atlas) {
  bool same = static_cast<bool>(g_raster.group) && g_raster.atlas == atlas.Get();
  for (size_t i = 0; i < views.size(); ++i) {
    same = same && g_raster.buffers[i] == views[i].buffer.Get() && g_raster.offsets[i] == views[i].offset &&
           g_raster.sizes[i] == views[i].size;
  }
  if (same) {
    return g_raster.group;
  }
  std::array<wgpu::BindGroupEntry, raster_binding::Count> entries{};
  for (uint32_t i = 0; i < views.size(); ++i) {
    entries[i] = {.binding = i, .buffer = views[i].buffer, .offset = views[i].offset, .size = views[i].size};
    g_raster.buffers[i] = views[i].buffer.Get();
    g_raster.offsets[i] = views[i].offset;
    g_raster.sizes[i] = views[i].size;
  }
  entries[raster_binding::Atlas] = {.binding = raster_binding::Atlas, .textureView = atlas};
  entries[raster_binding::Sampler] = {.binding = raster_binding::Sampler, .sampler = g_raster.sampler};
  g_raster.atlas = atlas.Get();
  const wgpu::BindGroupDescriptor descriptor{
      .label = "vsm.raster",
      .layout = g_raster.groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_raster.group = webgpu::g_device.CreateBindGroup(&descriptor);
  return g_raster.group;
}

bool declarable(const PassSetup& setup) noexcept {
  const std::array<PersistentHandle, 3> geometry = geometry_handles();
  const std::array<PersistentHandle, 2> scene = scene_handles();
  return active() && setup.env != nullptr && setup.env->retained && handle(Pool).id != 0 && handle(ViewsName).id != 0 &&
         geometry[0].id != 0 && geometry[1].id != 0 && geometry[2].id != 0 && scene[0].id != 0 && scene[1].id != 0 &&
         atlas_handle().id != 0;
}

void declare_raster(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  RasterLocal local;
  local.caster = declare_caster_pass(builder, setup);
  const std::array<PersistentHandle, 3> geometry = geometry_handles();
  const std::array<PersistentHandle, 2> scene = scene_handles();
  const auto pool = [&](std::string_view name, PersistentHandle h) {
    return builder.import_persistent(name, h, Access::StorageRead).index;
  };
  local.buffers[VerticesRef] = pool("scene.vertices", geometry[1]);
  local.buffers[IndicesRef] = pool("scene.indices", geometry[0]);
  local.buffers[InstancesRef] = pool("scene.instances", scene[0]);
  local.buffers[ClustersRef] = pool("scene.clusters", geometry[2]);
  local.buffers[PairsRef] = builder.use_buffer(Pairs, Access::StorageRead).index;
  local.buffers[MaterialsRef] = pool("scene.materials", scene[1]);
  local.args = builder.use_buffer(RasterArgs, Access::Indirect).index;
  local.atlas =
      builder.import_persistent_texture("scene.atlas_page", atlas_handle(), scene::atlas::page_desc(), Access::Sampled)
          .index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

bool raster_enabled(const StageData& stage, const void*) { return chain_enabled(stage); }

void raster_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  if (!raster_layouts()) {
    ids[0] = shaders::NeverReady.value;
    ids[1] = shaders::NeverReady.value;
    return;
  }
  for (uint32_t i = 0; i < 2; ++i) {
    shaders::RenderPipelineDesc desc{
        .label = RasterPass,
        .shader = RasterShader,
        .layout = g_raster.pipelineLayout,
        .layoutKey = RasterLayoutKey,
    };
    caster_pipeline(desc, i == 1);
    const shaders::PipelineId id = shaders::request_render(desc);
    ids[i] = id.value;
    register_chain_pipeline(id);
  }
}

void encode_raster(detail::PassContext& ctx, const void*) {
  RasterLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const std::array pipelines{shaders::render_pipeline(ctx.pipeline(0)), shaders::render_pipeline(ctx.pipeline(1))};
  constexpr uint32_t Need = chain_bit::Clear | chain_bit::CullEmit;
  if (!pipelines[0] || !pipelines[1] || !g_raster.groupLayout || (encoded_mask() & Need) != Need) {
    return;
  }
  std::array<BufferView, raster_binding::Buffers> views;
  const std::array<uint16_t, raster_binding::Buffers> refs{
      local.buffers[VerticesRef], local.buffers[IndicesRef], local.buffers[InstancesRef], local.buffers[ClustersRef],
      local.caster.renderList,    local.buffers[PairsRef],   local.caster.views,          local.buffers[MaterialsRef],
  };
  for (uint32_t i = 0; i < refs.size(); ++i) {
    views[i] = ctx.buffer(BufRef{refs[i]});
    if (!views[i].buffer) {
      return;
    }
  }
  views[raster_binding::Views].size = std::min(views[raster_binding::Views].size, ViewsBytes);
  const BufferView args = ctx.buffer(BufRef{local.args});
  const wgpu::TextureView atlas = ctx.view(TexRef{local.atlas});
  if (!args.buffer || !atlas) {
    return;
  }
  const wgpu::BindGroup& group = raster_group(views, atlas);
  const detail::DepthTarget depth{.ref = TexRef{local.caster.pool}};
  const wgpu::RenderPassEncoder pass = ctx.begin_render({}, &depth, "vsm.raster");
  if (!pass) {
    return;
  }
  pass.SetBindGroup(0, group);
  pass.SetPipeline(pipelines[0]);
  pass.DrawIndirect(args.buffer, args.offset);
  pass.SetPipeline(pipelines[1]);
  pass.DrawIndirect(args.buffer, args.offset + sizeof(uint32_t) * 4);
  pass.End();
  mark_encoded(chain_bit::Raster);
  g_ran.fetch_add(1, std::memory_order_relaxed);
  g_clipRuns.fetch_add(caps().clipDistances ? 1 : 0, std::memory_order_relaxed);
  g_unclippedRuns.fetch_add(caps().depthClipControl ? 1 : 0, std::memory_order_relaxed);
}
} // namespace

AtlasLayout atlas_layout(uint32_t pages, uint32_t maxDimension) noexcept {
  const PoolExtent extent = pool_extent(pages, maxDimension);
  if (!extent.fits) {
    return {};
  }
  return {
      .pagesX = extent.pagesX,
      .pagesY = extent.height / PageTexels,
      .width = extent.width,
      .height = extent.height,
  };
}

CasterRefs declare_caster_pass(detail::InternalPassBuilder& builder, const PassSetup&) {
  builder.after(phase::Shadows);
  builder.manual_bindings();
  CasterRefs refs;
  refs.pool = builder.import_persistent_texture(Pool, handle(Pool), pool_desc(), Access::DepthWrite).index;
  refs.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::Uniform).index;
  refs.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  return refs;
}

void caster_variant(shaders::RenderPipelineDesc& desc, bool cutout, bool clipDistances, bool unclipped) noexcept {
  const Variant& v = Variants[(cutout ? 4u : 0u) | (clipDistances ? 2u : 0u) | (unclipped ? 1u : 0u)];
  desc.defines = std::span{v.defines}.first(v.count);
  desc.vertexEntry = "vs_caster";
  desc.fragmentEntry = cutout ? "fs_cutout" : clipDistances ? std::string_view{} : "fs_rect";
  desc.cullMode = wgpu::CullMode::None;
  desc.colorCount = 0;
  desc.depthFormat = wgpu::TextureFormat::Depth32Float;
  desc.depthWrite = true;
  desc.depthCompare = CasterCompare;
  desc.unclippedDepth = unclipped;
}

void caster_pipeline(shaders::RenderPipelineDesc& desc, bool cutout) noexcept {
  caster_variant(desc, cutout, caps().clipDistances, caps().depthClipControl);
}

detail::PassDesc raster_pass_desc() {
  return {
      .name = std::string{RasterPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .cls = detail::PassClass::Required,
      .order = 50,
      .declare = declare_raster,
      .enabled = raster_enabled,
      .encode = encode_raster,
      .pipelines = raster_pipelines,
  };
}

void raster_shutdown() noexcept {
  if (const uint64_t ran = g_ran.exchange(0, std::memory_order_relaxed); ran != 0) {
    Log.info("VSM: raster ran on {} frames ({} with clip distances, {} with unclipped depth)", ran,
             g_clipRuns.exchange(0, std::memory_order_relaxed), g_unclippedRuns.exchange(0, std::memory_order_relaxed));
  }
  g_raster = {};
}

namespace testing {
void set_atlas_handle(std::optional<PersistentHandle> handle) noexcept { g_atlasOverride = handle; }

wgpu::BindGroupLayout raster_group_layout() {
  (void)raster_layouts();
  return g_raster.groupLayout;
}

wgpu::PipelineLayout raster_layout() {
  (void)raster_layouts();
  return g_raster.pipelineLayout;
}

RasterCounts raster_counts() noexcept {
  return {
      .ran = g_ran.load(std::memory_order_relaxed),
      .clipVariant = g_clipRuns.load(std::memory_order_relaxed),
      .unclipped = g_unclippedRuns.load(std::memory_order_relaxed),
  };
}
} // namespace testing
} // namespace aurora::render::shadows::vsm

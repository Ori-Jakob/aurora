#include "vsm_alloc.hpp"

#include "vsm_check.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../geometry/compact.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstring>

namespace aurora::render::shadows::vsm {
namespace {
using detail::PassSetup;
using detail::StageData;

constexpr uint16_t NoRef = UINT16_MAX;
constexpr uint64_t ClearLayoutKey = 0x76736D636C720001ull;
constexpr float CoordSlack = 1e-2f;
constexpr double LevelSlack = 1e-4;
constexpr std::array<std::string_view, 1> MarkDefines{"VSM_MARK"};
constexpr std::array<std::string_view, 2> MarkCheckDefines{"VSM_MARK", "VSM_MARK_CHECK"};
constexpr std::array<std::string_view, 1> ClassifyDefines{"VSM_CLASSIFY"};
constexpr std::array<std::string_view, 1> AssignDefines{"VSM_ASSIGN"};
constexpr std::array<std::string_view, 1> CommitDefines{"VSM_COMMIT"};
constexpr std::array<std::string_view, 1> ClearDefines{"VSM_CLEAR"};

struct MarkLocal {
  uint16_t depth = NoRef;
  uint16_t normal = NoRef;
  uint16_t views = NoRef;
  uint16_t requests = NoRef;
  uint16_t receivers = NoRef;
};
static_assert(sizeof(MarkLocal) <= detail::PassLocalSize);

struct ClassifyLocal {
  uint16_t requests = NoRef;
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t phys = NoRef;
  uint16_t stats = NoRef;
  uint16_t need = NoRef;
  uint16_t physWords = NoRef;
};
static_assert(sizeof(ClassifyLocal) <= detail::PassLocalSize);

struct ListsLocal {
  uint16_t need = NoRef;
  uint16_t physWords = NoRef;
  uint16_t needList = NoRef;
  uint16_t needScratch = NoRef;
  uint16_t freeList = NoRef;
  uint16_t freeScratch = NoRef;
};
static_assert(sizeof(ListsLocal) <= detail::PassLocalSize);

struct AssignLocal {
  uint16_t needList = NoRef;
  uint16_t freeList = NoRef;
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t phys = NoRef;
  uint16_t stats = NoRef;
  uint16_t renderList = NoRef;
  uint16_t clearArgs = NoRef;
};
static_assert(sizeof(AssignLocal) <= detail::PassLocalSize);

struct ClearLocal {
  uint16_t renderList = NoRef;
  uint16_t table = NoRef;
  uint16_t args = NoRef;
  uint16_t pool = NoRef;
};
static_assert(sizeof(ClearLocal) <= detail::PassLocalSize);

struct CommitLocal {
  uint16_t renderList = NoRef;
  uint16_t table = NoRef;
  uint16_t stats = NoRef;
};
static_assert(sizeof(CommitLocal) <= detail::PassLocalSize);

struct ClearBinding {
  wgpu::BindGroupLayout groupLayout;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::BindGroup group;
  std::array<WGPUBuffer, 2> buffers{};
  std::array<uint64_t, 2> offsets{};
  std::array<uint64_t, 2> sizes{};
};
ClearBinding g_clear;

template <typename T>
T read_local(const detail::PassContext& ctx) {
  T local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  return local;
}

template <typename T>
void write_local(detail::InternalPassBuilder& builder, const T& local) {
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

bool declarable(const PassSetup& setup) noexcept {
  return active() && setup.env != nullptr && setup.env->retained && handle(Table).id != 0 && handle(Pool).id != 0;
}

bool chain_pass_enabled(const StageData& stage, const void*) { return chain_enabled(stage); }

bool updated() noexcept { return (encoded_mask() & chain_bit::Update) != 0; }

shaders::PipelineId compute(const detail::PipelineSetup& setup, std::string_view label, std::string_view entry,
                            std::span<const std::string_view> defines) {
  const shaders::PipelineId id = shaders::request_compute({
      .label = label,
      .shader = AllocShader,
      .entryPoint = entry,
      .defines = defines,
      .layout = setup.layout,
      .layoutKey = setup.layoutHash,
  });
  register_chain_pipeline(id);
  return id;
}

void dispatch(detail::PassContext& ctx, std::span<const wgpu::ComputePipeline> pipelines,
              std::span<const uint32_t> groups) {
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  for (size_t i = 0; i < pipelines.size(); ++i) {
    pass.SetPipeline(pipelines[i]);
    pass.DispatchWorkgroups(groups[i]);
  }
  pass.End();
}

void declare_mark(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  MarkLocal local;
  local.depth = builder
                    .use(resource::GxDepth, Access::Sampled,
                         detail::UseOptions{.missing = Missing::DisablePass,
                                            .sampleType = wgpu::TextureSampleType::UnfilterableFloat})
                    .index;
  local.normal = builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.requests = builder
                       .create_buffer(Requests, {.size = RequestsBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                      Access::StorageWrite)
                       .index;
  if (check::config().on) {
    local.receivers =
        builder
            .create_buffer(MarkReceivers, {.sizeClass = SizeClass::Efb, .bytesPerTexel = ReceiverWords * 4},
                           Access::StorageWrite)
            .index;
  }
  write_local(builder, local);
}

void mark_pipelines(const detail::PipelineSetup& setup, const void*,
                    std::span<uint32_t, detail::MaxPassPipelines> ids) {
  MarkLocal local;
  std::memcpy(&local, setup.local.data(), sizeof(local));
  const std::span<const std::string_view> defines =
      local.receivers != NoRef ? std::span<const std::string_view>{MarkCheckDefines} : MarkDefines;
  ids[0] = compute(setup, MarkPass, "vsm_mark", defines).value;
}

void mark_stage_params(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(MarkParams)) {
    return;
  }
  MarkParams params;
  const detail::SceneHints* hints = stage.sceneHints.get();
  (void)detail::fill_stage_camera(hints, params.camera);
  (void)detail::fill_view_rotation(hints, params.rotation);
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode_mark(detail::PassContext& ctx, const void*) {
  const auto local = read_local<MarkLocal>(ctx);
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView requests = ctx.buffer(BufRef{local.requests});
  const wgpu::Extent3D extent = ctx.extent(TexRef{local.depth});
  if (!pipeline || !requests.buffer || extent.width == 0 || extent.height == 0 || !updated()) {
    return;
  }
  ctx.encoder().ClearBuffer(requests.buffer, requests.offset, std::min(requests.size, RequestsBytes));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups((extent.width + MarkWorkgroup - 1) / MarkWorkgroup,
                          (extent.height + MarkWorkgroup - 1) / MarkWorkgroup);
  pass.End();
  mark_encoded(chain_bit::Mark);
  if (local.receivers != NoRef) {
    MarkParams params;
    std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
    params.width = extent.width;
    params.height = extent.height;
    const StageData& stage = ctx.stage_data();
    check::record_mark(params, stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0);
  }
}

void declare_classify(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  ClassifyLocal local;
  local.requests = builder.use_buffer(Requests, Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRW).index;
  local.phys = builder.import_persistent(Phys, handle(Phys), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.need = builder.create_buffer(NeedWords, {.size = NeedWordsBytes}, Access::StorageWrite).index;
  local.physWords = builder.create_buffer(PhysWordsName, {.size = PhysWordsBytes}, Access::StorageWrite).index;
  write_local(builder, local);
}

void classify_pipelines(const detail::PipelineSetup& setup, const void*,
                        std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, ClassifyPass, "vsm_classify", ClassifyDefines).value;
  ids[1] = compute(setup, ClassifyPass, "vsm_phys_classify", ClassifyDefines).value;
}

void encode_classify(detail::PassContext& ctx, const void*) {
  const std::array pipelines{shaders::compute_pipeline(ctx.pipeline(0)), shaders::compute_pipeline(ctx.pipeline(1))};
  if (!pipelines[0] || !pipelines[1] || (encoded_mask() & chain_bit::Mark) == 0) {
    return;
  }
  constexpr std::array<uint32_t, 2> Groups{MaxVirtualPages / AllocWorkgroup, MaxPoolPages / AllocWorkgroup};
  dispatch(ctx, pipelines, Groups);
  mark_encoded(chain_bit::Classify);
}

void declare_lists(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup) || setup.caps == nullptr || !compact::supported(*setup.caps)) {
    builder.skip();
    return;
  }
  const compact::Plan needPlan = compact::make_plan(compact::Op::Compact, NeedItems, *setup.caps);
  const compact::Plan physPlan = compact::make_plan(compact::Op::Compact, PhysItems, *setup.caps);
  if (!needPlan.valid() || !physPlan.valid()) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.manual_bindings();
  ListsLocal local;
  local.need = builder.use_buffer(NeedWords, Access::StorageRead).index;
  local.physWords = builder.use_buffer(PhysWordsName, Access::StorageRead).index;
  const auto create = [&](std::string_view name, uint64_t size) {
    return builder.create_buffer(name, {.size = size, .extraUsage = wgpu::BufferUsage::CopyDst}, Access::StorageWrite)
        .index;
  };
  local.needList = create(NeedList, uint64_t{NeedItems} * 4);
  local.needScratch = create(NeedScratch, needPlan.scratchBytes);
  local.freeList = create(FreeList, uint64_t{PhysItems} * 4);
  local.freeScratch = create(FreeScratch, physPlan.scratchBytes);
  write_local(builder, local);
}

void encode_lists(detail::PassContext& ctx, const void*) {
  const auto local = read_local<ListsLocal>(ctx);
  if ((encoded_mask() & chain_bit::Classify) == 0) {
    return;
  }
  const auto view = [&](uint16_t ref) { return ctx.buffer(BufRef{ref}); };
  const std::array jobs{
      compact::Job{
          .plan = compact::make_plan(compact::Op::Compact, NeedItems, caps()),
          .input = view(local.need),
          .output = view(local.needList),
          .scratch = view(local.needScratch),
          .keepMask = 1,
      },
      compact::Job{
          .plan = compact::make_plan(compact::Op::Compact, PhysItems, caps()),
          .input = view(local.physWords),
          .output = view(local.freeList),
          .scratch = view(local.freeScratch),
          .keepMask = 1,
      },
  };
  if (compact::encode(ctx, jobs)) {
    mark_encoded(chain_bit::AllocLists);
  }
}

void declare_assign(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  AssignLocal local;
  local.needList = builder.use_buffer(NeedList, Access::StorageRead).index;
  local.freeList = builder.use_buffer(FreeList, Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRW).index;
  local.phys = builder.import_persistent(Phys, handle(Phys), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.renderList =
      builder
          .create_buffer(RenderList, {.size = render_list::Bytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                         Access::StorageWrite)
          .index;
  local.clearArgs = builder
                        .create_buffer(ClearArgs,
                                       {.size = sizeof(uint32_t) * 4,
                                        .extraUsage = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst},
                                       Access::StorageWrite)
                        .index;
  write_local(builder, local);
}

void assign_pipelines(const detail::PipelineSetup& setup, const void*,
                      std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, AssignPass, "vsm_assign", AssignDefines).value;
}

void encode_assign(detail::PassContext& ctx, const void*) {
  const auto local = read_local<AssignLocal>(ctx);
  const std::array pipelines{shaders::compute_pipeline(ctx.pipeline(0))};
  const BufferView renderList = ctx.buffer(BufRef{local.renderList});
  if (!pipelines[0] || !renderList.buffer || (encoded_mask() & chain_bit::AllocLists) == 0) {
    return;
  }
  ctx.encoder().ClearBuffer(renderList.buffer, renderList.offset, std::min(renderList.size, render_list::Bytes));
  constexpr std::array<uint32_t, 1> Groups{MaxRenderPages / AllocWorkgroup};
  dispatch(ctx, pipelines, Groups);
  mark_encoded(chain_bit::Assign);
}

bool clear_layouts() {
  if (g_clear.pipelineLayout) {
    return true;
  }
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return false;
  }
  std::array<wgpu::BindGroupLayoutEntry, 2> entries{};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Vertex,
        .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
    };
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{
      .label = "vsm.clear", .entryCount = entries.size(), .entries = entries.data()};
  g_clear.groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "vsm.clear", .bindGroupLayoutCount = 1, .bindGroupLayouts = &g_clear.groupLayout};
  g_clear.pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  return static_cast<bool>(g_clear.pipelineLayout);
}

const wgpu::BindGroup& clear_group(const std::array<BufferView, 2>& views) {
  bool same = static_cast<bool>(g_clear.group);
  for (size_t i = 0; i < views.size(); ++i) {
    same = same && g_clear.buffers[i] == views[i].buffer.Get() && g_clear.offsets[i] == views[i].offset &&
           g_clear.sizes[i] == views[i].size;
  }
  if (same) {
    return g_clear.group;
  }
  std::array<wgpu::BindGroupEntry, 2> entries{};
  for (uint32_t i = 0; i < views.size(); ++i) {
    entries[i] = {.binding = i, .buffer = views[i].buffer, .offset = views[i].offset, .size = views[i].size};
    g_clear.buffers[i] = views[i].buffer.Get();
    g_clear.offsets[i] = views[i].offset;
    g_clear.sizes[i] = views[i].size;
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = "vsm.clear",
      .layout = g_clear.groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_clear.group = webgpu::g_device.CreateBindGroup(&descriptor);
  return g_clear.group;
}

void declare_clear(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.manual_bindings();
  ClearLocal local;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRead).index;
  local.args = builder.use_buffer(ClearArgs, Access::Indirect).index;
  local.pool = builder.import_persistent_texture(Pool, handle(Pool), pool_desc(), Access::DepthWrite).index;
  write_local(builder, local);
}

void clear_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  if (!clear_layouts()) {
    ids[0] = shaders::NeverReady.value;
    return;
  }
  const shaders::PipelineId id = shaders::request_render({
      .label = ClearPass,
      .shader = AllocShader,
      .vertexEntry = "vs_clear",
      .fragmentEntry = {},
      .defines = ClearDefines,
      .layout = g_clear.pipelineLayout,
      .layoutKey = ClearLayoutKey,
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .depthWrite = true,
      .depthCompare = wgpu::CompareFunction::Always,
  });
  ids[0] = id.value;
  register_chain_pipeline(id);
}

void encode_clear(detail::PassContext& ctx, const void*) {
  const auto local = read_local<ClearLocal>(ctx);
  const wgpu::RenderPipeline pipeline = shaders::render_pipeline(ctx.pipeline(0));
  const std::array views{ctx.buffer(BufRef{local.renderList}), ctx.buffer(BufRef{local.table})};
  const BufferView args = ctx.buffer(BufRef{local.args});
  if (!pipeline || !views[0].buffer || !views[1].buffer || !args.buffer || !g_clear.groupLayout ||
      (encoded_mask() & chain_bit::Assign) == 0) {
    return;
  }
  const wgpu::BindGroup& group = clear_group(views);
  const detail::DepthTarget depth{.ref = TexRef{local.pool}};
  const wgpu::RenderPassEncoder pass = ctx.begin_render({}, &depth, "vsm.clear");
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group);
  pass.DrawIndirect(args.buffer, args.offset);
  pass.End();
  mark_encoded(chain_bit::Clear);
}

void declare_commit(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  CommitLocal local;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  write_local(builder, local);
}

void commit_pipelines(const detail::PipelineSetup& setup, const void*,
                      std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, CommitPass, "vsm_commit", CommitDefines).value;
}

void encode_commit(detail::PassContext& ctx, const void*) {
  const std::array pipelines{shaders::compute_pipeline(ctx.pipeline(0))};
  const uint32_t want = expected_chain() & ~chain_bit::Commit;
  if (!pipelines[0] || (encoded_mask() & want) != want) {
    return;
  }
  constexpr std::array<uint32_t, 1> Groups{MaxRenderPages / AllocWorkgroup};
  dispatch(ctx, pipelines, Groups);
  mark_encoded(chain_bit::Commit);
}

detail::PassDesc compute_desc(std::string_view name, uint16_t order, detail::PassDeclareFn declare,
                              detail::PassEncodeFn encode, detail::PassPipelinesFn pipelines) {
  return {
      .name = std::string{name},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = order,
      .declare = declare,
      .enabled = chain_pass_enabled,
      .encode = encode,
      .pipelines = pipelines,
  };
}

float saturate(float x) noexcept { return std::clamp(x, 0.f, 1.f); }

int32_t page_floor(float texel) noexcept {
  const float page = std::floor(texel / PageTexels);
  return static_cast<int32_t>(std::clamp(page, -1.f, static_cast<float>(WindowPages)));
}

template <typename Add>
void level_pages(const ViewRecord& view, const std::array<float, 3>& pRel, uint32_t bits, float slack, const Add& add) {
  const std::array<float, 3> c = view_coords(view, pRel);
  constexpr float Margin = FilterMarginTexels;
  std::array<int32_t, 2> lo{};
  std::array<int32_t, 2> hi{};
  for (size_t k = 0; k < 2; ++k) {
    lo[k] = std::max(page_floor(c[k] - Margin - slack), 0);
    hi[k] = std::min(page_floor(c[k] + Margin + slack), static_cast<int32_t>(WindowPages) - 1);
  }
  for (int32_t y = lo[1]; y <= hi[1]; ++y) {
    for (int32_t x = lo[0]; x <= hi[0]; ++x) {
      add(request_key(page_index(view.table_offset(), view.origin[0] + x, view.origin[1] + y), bits));
    }
  }
}

void set_key(std::vector<uint32_t>& words, uint32_t key) noexcept {
  const uint32_t index = key & 0xFFFFu;
  if (index < words.size()) {
    words[index] |= key >> 16;
  }
}

uint32_t clamp_level(double wanted) noexcept {
  return static_cast<uint32_t>(std::clamp(wanted, 0.0, static_cast<double>(SunLevels - 1)));
}
} // namespace

uint32_t level_wanted(float footprint, float texel0, float bias) noexcept {
  const float wanted = std::ceil(std::log2(std::max(footprint, 1e-30f) / texel0) + bias);
  return std::isfinite(wanted) ? static_cast<uint32_t>(std::clamp(wanted, 0.f, SunLevels - 1.f)) : 0u;
}

MarkKeys mark_keys(const Views& views, const std::array<float, 3>& pRel, float footprint) noexcept {
  MarkKeys out;
  const uint32_t level = select_level(views, footprint, pRel, views.header.levelBias);
  if (level >= SunLevels) {
    return out;
  }
  const auto add = [&](uint32_t key) {
    if (out.count < out.keys.size()) {
      out.keys[out.count++] = key;
    }
  };
  level_pages(views.views[level_view(level)], pRel, request_bit::Pixel, 0.f, add);
  if (level + 1 < SunLevels) {
    level_pages(views.views[level_view(level + 1)], pRel, request_bit::Fallback, 0.f, add);
  }
  return out;
}

Receiver mark_receiver(const MarkParams& params, uint32_t x, uint32_t y, float depth, float normalAlpha) noexcept {
  Receiver out;
  const detail::StageCamera& cam = params.camera;
  const auto code = static_cast<uint32_t>(std::nearbyint(saturate(normalAlpha) * 3.f));
  if (!(depth > 0.f) || code < 2 || (cam.flags & detail::CameraValid) == 0 || params.width == 0 || params.height == 0) {
    return out;
  }
  const auto w = static_cast<float>(params.width);
  const auto h = static_cast<float>(params.height);
  const float z = std::clamp(cam.depthMul / (cam.depthAdd - depth), 0.f, 1e30f);
  const float u = (static_cast<float>(x) + 0.5f) / w;
  const float v = (static_cast<float>(y) + 0.5f) / h;
  const float cx = cam.p02 + -2.f * cam.jitterPx[0] / w;
  const float cy = cam.p12 + 2.f * cam.jitterPx[1] / h;
  const float px = (2.f * u - 1.f + cx) * z / cam.p00;
  const float py = (1.f - 2.f * v + cy) * z / cam.p11;
  const auto& r = params.rotation.rows;
  for (size_t i = 0; i < 3; ++i) {
    out.pRel[i] = r[0][i] * px + r[1][i] * py - r[2][i] * z;
  }
  out.footprint = 2.f * z / (h * cam.p11);
  out.valid = true;
  return out;
}

MarkReference mark_reference(const Views& views, const MarkParams& params, std::span<const uint32_t> receivers) {
  MarkReference out;
  out.must.assign(MaxVirtualPages, 0);
  out.may.assign(MaxVirtualPages, 0);
  const float texel0 = views.header.texel0;
  if (!(texel0 > 0.f)) {
    return out;
  }
  const uint64_t pixels = std::min<uint64_t>(uint64_t{params.width} * params.height, receivers.size() / ReceiverWords);
  constexpr float Margin = FilterMarginTexels;
  constexpr float Edge = static_cast<float>(WindowTexels) - Margin;
  for (uint64_t i = 0; i < pixels; ++i) {
    const auto x = static_cast<uint32_t>(i % params.width);
    const auto y = static_cast<uint32_t>(i / params.width);
    const Receiver r = mark_receiver(params, x, y, std::bit_cast<float>(receivers[i * ReceiverWords]),
                                     std::bit_cast<float>(receivers[i * ReceiverWords + 1]));
    if (!r.valid) {
      continue;
    }
    ++out.receivers;
    const double w = std::log2(static_cast<double>(std::max(r.footprint, 1e-30f)) / texel0) + views.header.levelBias;
    std::bitset<SunLevels + 1> possible;
    for (const uint32_t wanted : {clamp_level(std::ceil(w - LevelSlack)), clamp_level(std::ceil(w + LevelSlack))}) {
      bool decided = false;
      for (uint32_t level = wanted; level < SunLevels && !decided; ++level) {
        const std::array<float, 3> c = view_coords(views.views[level_view(level)], r.pRel);
        const bool surely = c[0] >= Margin + CoordSlack && c[0] < Edge - CoordSlack && c[1] >= Margin + CoordSlack &&
                            c[1] < Edge - CoordSlack;
        const bool maybe = c[0] >= Margin - CoordSlack && c[0] < Edge + CoordSlack && c[1] >= Margin - CoordSlack &&
                           c[1] < Edge + CoordSlack;
        possible[level] = possible[level] || maybe;
        decided = surely;
      }
      possible[SunLevels] = possible[SunLevels] || !decided;
    }
    const bool definite = possible.count() == 1 && !possible[SunLevels];
    out.ambiguous += definite ? 0 : 1;
    for (uint32_t level = 0; level < SunLevels; ++level) {
      if (!possible[level]) {
        continue;
      }
      const auto add_may = [&](uint32_t key) { set_key(out.may, key); };
      const auto add_must = [&](uint32_t key) { set_key(out.must, key); };
      level_pages(views.views[level_view(level)], r.pRel, request_bit::Pixel, CoordSlack, add_may);
      if (definite) {
        level_pages(views.views[level_view(level)], r.pRel, request_bit::Pixel, -CoordSlack, add_must);
      }
      if (level + 1 < SunLevels) {
        level_pages(views.views[level_view(level + 1)], r.pRel, request_bit::Fallback, CoordSlack, add_may);
        if (definite) {
          level_pages(views.views[level_view(level + 1)], r.pRel, request_bit::Fallback, -CoordSlack, add_must);
        }
      }
    }
  }
  return out;
}

uint32_t phys_band_of(uint32_t owner, uint32_t lastUsed, uint32_t clock) noexcept {
  if (owner == 0) {
    return phys_band::Free;
  }
  const uint32_t age = clock - lastUsed;
  if (age >= OldAge) {
    return phys_band::Old;
  }
  if (age >= MidAge) {
    return phys_band::Mid;
  }
  return age >= 1 ? phys_band::Young : phys_band::Count;
}

AllocBudget alloc_budget(uint32_t priority, uint32_t fresh, uint32_t reuse, uint32_t evictable, uint32_t budget,
                         bool held) noexcept {
  AllocBudget b;
  if (held) {
    return b;
  }
  const uint32_t within = std::min(budget, fresh + reuse);
  b.count = std::min(priority + within, MaxRenderPages);
  const uint32_t pIn = std::min(priority, b.count);
  b.fresh = std::min(std::min(fresh, within), b.count - pIn);
  b.reuse = std::min(within - std::min(fresh, within), b.count - pIn - b.fresh);
  b.deferredNew = fresh - b.fresh;
  b.deferredReuse = reuse - b.reuse;
  b.overflowPool = b.fresh - std::min(b.fresh, evictable);
  b.playerDeferred = priority - pIn;
  return b;
}

AllocCounts allocate(const Views& views, std::span<const uint32_t> requests, std::span<uint32_t> table,
                     std::span<uint32_t> phys, std::span<uint32_t> renderList) noexcept {
  AllocCounts out;
  const ViewsHeader& h = views.header;
  const uint32_t clock = h.clock;
  const bool held = (h.flags & header_flag::Held) != 0;
  if (table.size() < MaxVirtualPages || phys.size() < size_t{MaxPoolPages} * PhysWords ||
      renderList.size() < render_list::Words) {
    return out;
  }
  std::array<std::vector<uint32_t>, need_class::Count> need;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    const uint32_t req = v < requests.size() ? requests[v] & request_bit::Mask : 0u;
    const uint32_t e = table[v];
    const bool allocated = (e & entry::Allocated) != 0;
    const size_t p = size_t{e & entry::PhysMask} * PhysWords;
    const bool stale = (e & entry::Dirty) != 0 || (e & entry::Cached) == 0;
    if (req != 0) {
      ++out.requested;
      if (allocated) {
        phys[p + 1] = clock;
        if (stale) {
          need[(e & entry::Priority) != 0 ? need_class::Priority : need_class::Reuse].push_back(v);
        } else {
          ++out.hits;
        }
      } else {
        need[need_class::New].push_back(v);
      }
    } else if (allocated && (e & entry::Dirty) != 0) {
      table[v] = 0;
      phys[p] = 0;
      phys[p + 1] = 0;
      ++out.freedDirty;
    }
  }
  std::array<std::vector<uint32_t>, phys_band::Count> bands;
  for (uint32_t p = 0; p < std::min(h.poolPages, MaxPoolPages); ++p) {
    const uint32_t band = phys_band_of(phys[size_t{p} * PhysWords], phys[size_t{p} * PhysWords + 1], clock);
    if (band < phys_band::Count) {
      bands[band].push_back(p);
    }
  }
  std::vector<uint32_t> freeList;
  for (const std::vector<uint32_t>& band : bands) {
    freeList.insert(freeList.end(), band.begin(), band.end());
  }
  out.priority = static_cast<uint32_t>(need[need_class::Priority].size());
  out.fresh = static_cast<uint32_t>(need[need_class::New].size());
  out.reuse = static_cast<uint32_t>(need[need_class::Reuse].size());
  out.evictable = static_cast<uint32_t>(freeList.size());
  out.freePages = static_cast<uint32_t>(bands[phys_band::Free].size());
  const AllocBudget b = alloc_budget(out.priority, out.fresh, out.reuse, out.evictable, h.budget, held);
  out.rendered = b.count;
  out.deferredNew = b.deferredNew;
  out.deferredReuse = b.deferredReuse;
  out.overflowPool = b.overflowPool;
  out.playerDeferred = b.playerDeferred;
  std::vector<std::pair<uint32_t, uint32_t>> list;
  for (uint32_t c = 0; c < need_class::Count; ++c) {
    for (const uint32_t v : need[c]) {
      list.emplace_back(c, v);
    }
  }
  std::ranges::fill(renderList, 0u);
  const uint32_t pagesX = std::max(h.poolPagesX, 1u);
  const uint32_t pIn = b.count - b.fresh - b.reuse;
  renderList[render_list::Count] = b.count;
  renderList[render_list::Priority] = pIn;
  renderList[render_list::New] = b.fresh;
  renderList[render_list::Reuse] = b.reuse;
  renderList[render_list::Budget] = h.budget;
  renderList[render_list::PagesX] = pagesX;
  renderList[render_list::Rows] = (std::max(h.poolPages, 1u) + pagesX - 1) / pagesX;
  renderList[render_list::Flags] = held ? header_flag::Held : 0u;
  for (uint32_t i = 0; i < b.count && i < list.size(); ++i) {
    const auto [cls, v] = list[i];
    uint32_t page = NoPage;
    if (cls == need_class::New) {
      const uint32_t j = i - pIn;
      if (j < freeList.size()) {
        page = freeList[j];
        const uint32_t old = phys[size_t{page} * PhysWords];
        if (old != 0 && old - 1 < MaxVirtualPages) {
          table[old - 1] = 0;
          ++out.evicted;
        }
        table[v] = page | entry::Allocated | entry::RenderNow | entry::Fresh | i << entry::RenderIndexShift;
        phys[size_t{page} * PhysWords] = v + 1;
        phys[size_t{page} * PhysWords + 1] = clock;
      }
    } else {
      const uint32_t e = table[v];
      page = e & entry::PhysMask;
      table[v] = (e & (entry::Cached | entry::Dirty | entry::Priority)) | page | entry::Allocated | entry::RenderNow |
                 i << entry::RenderIndexShift;
    }
    renderList[render_list::Pages + 2 * i] = page == NoPage ? NoPage : v;
    renderList[render_list::Pages + 2 * i + 1] = page;
    if (page == NoPage) {
      continue;
    }
    const uint32_t view = v / PagesPerView;
    const uint32_t slot = v % PagesPerView;
    const ViewRecord& record = views.views[view];
    const uint32_t lx = (slot % WindowPages - static_cast<uint32_t>(record.origin[0])) & (WindowPages - 1);
    const uint32_t ly = (slot / WindowPages - static_cast<uint32_t>(record.origin[1])) & (WindowPages - 1);
    renderList[render_list::RowMasks + view * WindowPages + ly] |= 1u << lx;
  }
  return out;
}

uint32_t commit(std::span<const uint32_t> renderList, std::span<uint32_t> table) noexcept {
  if (renderList.size() < render_list::Words || table.size() < MaxVirtualPages) {
    return 0;
  }
  uint32_t committed = 0;
  const uint32_t count = std::min(renderList[render_list::Count], MaxRenderPages);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t v = renderList[render_list::Pages + 2 * i];
    if (v >= MaxVirtualPages || (table[v] & entry::RenderNow) == 0) {
      continue;
    }
    const uint32_t e = table[v];
    const uint32_t keep =
        e & ~(entry::RenderNow | entry::Skip | entry::Fresh | entry::RenderIndexMask << entry::RenderIndexShift);
    if ((e & entry::Skip) != 0) {
      table[v] = keep;
    } else {
      table[v] = (keep | entry::Cached) & ~(entry::Dirty | entry::Priority);
      ++committed;
    }
  }
  return committed;
}

detail::PassDesc mark_pass_desc() {
  detail::PassDesc desc = compute_desc(MarkPass, 10, declare_mark, encode_mark, mark_pipelines);
  desc.paramsSize = sizeof(MarkParams);
  desc.stageParams = mark_stage_params;
  return desc;
}

detail::PassDesc classify_pass_desc() {
  return compute_desc(ClassifyPass, 20, declare_classify, encode_classify, classify_pipelines);
}

detail::PassDesc alloc_lists_pass_desc() {
  return compute_desc(AllocListsPass, 21, declare_lists, encode_lists, compact::pass_pipelines);
}

detail::PassDesc assign_pass_desc() {
  return compute_desc(AssignPass, 22, declare_assign, encode_assign, assign_pipelines);
}

detail::PassDesc clear_pass_desc() {
  detail::PassDesc desc = compute_desc(ClearPass, 40, declare_clear, encode_clear, clear_pipelines);
  desc.kind = PassKind::Encoder;
  return desc;
}

detail::PassDesc commit_pass_desc() {
  return compute_desc(CommitPass, 58, declare_commit, encode_commit, commit_pipelines);
}

void alloc_shutdown() noexcept { g_clear = {}; }
} // namespace aurora::render::shadows::vsm

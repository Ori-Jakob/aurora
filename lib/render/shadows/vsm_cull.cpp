#include "vsm_cull.hpp"

#include "../graph_bind.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../geometry/compact.hpp"
#include "../scene/meshlets.hpp"
#include "../scene/streamout.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>

namespace aurora::render::shadows::vsm {
namespace {
using detail::PassSetup;
using detail::PersistentHandle;
using detail::StageData;

constexpr uint16_t NoRef = UINT16_MAX;
constexpr std::array<std::string_view, 1> ChunkDefines{"VSM_CHUNK_BOUNDS"};
constexpr std::array<std::string_view, 1> CandidateDefines{"VSM_CULL_CANDIDATES"};
constexpr std::array<std::string_view, 1> CountDefines{"VSM_CULL_PAIRS"};
constexpr std::array<std::string_view, 2> EmitDefines{"VSM_CULL_EMIT", "VSM_CULL_PAIRS"};
constexpr std::array<std::string_view, 1> SkipDefines{"VSM_CULL_SKIP"};

const compact::GraphJob ListJob{
    .op = compact::Op::Compact,
    .capacity = Slots,
    .input = CullWords,
    .output = CullSlots,
    .scratch = CullScratch,
    .keepMask = 0xFFFFu,
    .args = CullArgs,
    .argsSpec = {.kind = compact::ArgsKind::Dispatch, .scale = 1},
};

const compact::GraphJob ScanJob{
    .op = compact::Op::Scan,
    .capacity = PageCountWords,
    .input = PageCounts,
    .output = PageOffsets,
    .scratch = OffsetScratch,
};

struct ChunkLocal {
  uint16_t list = NoRef;
  uint16_t history = NoRef;
  uint16_t indices = NoRef;
  uint16_t vertices = NoRef;
  uint16_t boxes = NoRef;
  uint16_t stats = NoRef;
  uint16_t args = NoRef;
};
static_assert(sizeof(ChunkLocal) <= detail::PassLocalSize);

struct CandidateLocal {
  uint16_t history = NoRef;
  uint16_t views = NoRef;
  uint16_t renderList = NoRef;
  uint16_t words = NoRef;
  uint16_t stats = NoRef;
};
static_assert(sizeof(CandidateLocal) <= detail::PassLocalSize);

struct PairsLocal {
  uint16_t list = NoRef;
  uint16_t history = NoRef;
  uint16_t clusters = NoRef;
  uint16_t boxes = NoRef;
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t renderList = NoRef;
  uint16_t target = NoRef;
  uint16_t pairs = NoRef;
  uint16_t args = NoRef;
};
static_assert(sizeof(PairsLocal) <= detail::PassLocalSize);

struct SkipLocal {
  uint16_t counts = NoRef;
  uint16_t offsets = NoRef;
  uint16_t renderList = NoRef;
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t stats = NoRef;
  uint16_t pairs = NoRef;
  uint16_t args = NoRef;
};
static_assert(sizeof(SkipLocal) <= detail::PassLocalSize);

std::optional<std::array<PersistentHandle, 3>> g_geometryOverride;

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
  const std::array<PersistentHandle, 3> geometry = geometry_handles();
  return active() && setup.env != nullptr && setup.env->retained && handle(History).id != 0 &&
         handle(ChunkBoxes).id != 0 && geometry[0].id != 0 && geometry[1].id != 0 && geometry[2].id != 0;
}

bool chain_pass_enabled(const StageData& stage, const void*) { return chain_enabled(stage); }

bool has(uint32_t bits) noexcept { return (encoded_mask() & bits) == bits; }

shaders::PipelineId compute(const detail::PipelineSetup& setup, std::string_view label, std::string_view entry,
                            std::span<const std::string_view> defines) {
  const shaders::PipelineId id = shaders::request_compute({
      .label = label,
      .shader = CullShader,
      .entryPoint = entry,
      .defines = defines,
      .layout = setup.layout,
      .layoutKey = setup.layoutHash,
  });
  register_chain_pipeline(id);
  return id;
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

void declare_chunks(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  const std::array<PersistentHandle, 3> geometry = geometry_handles();
  ChunkLocal local;
  local.list = builder.use_buffer(ListedSlots, Access::StorageRead).index;
  local.history = builder.import_persistent(History, handle(History), Access::StorageRead).index;
  local.indices = builder.import_persistent("scene.indices", geometry[0], Access::StorageRead).index;
  local.vertices = builder.import_persistent("scene.vertices", geometry[1], Access::StorageRead).index;
  local.boxes = builder.import_persistent(ChunkBoxes, handle(ChunkBoxes), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.args = builder.use_buffer(DirtyArgs, Access::Indirect).index;
  write_local(builder, local);
}

void chunk_pipelines(const detail::PipelineSetup& setup, const void*,
                     std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, ChunkBoundsPass, "vsm_chunk_bounds", ChunkDefines).value;
}

void encode_chunks(detail::PassContext& ctx, const void*) {
  const auto local = read_local<ChunkLocal>(ctx);
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView args = ctx.buffer(BufRef{local.args});
  if (!pipeline || !args.buffer || !has(chain_bit::Scan | chain_bit::ScanList)) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroupsIndirect(args.buffer, args.offset);
  pass.End();
  mark_encoded(chain_bit::ChunkBounds);
}

void declare_candidates(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  CandidateLocal local;
  local.history = builder.import_persistent(History, handle(History), Access::StorageRW).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  local.words = builder
                    .create_buffer(CullWords, {.size = CullWordsBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                   Access::StorageWrite)
                    .index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  write_local(builder, local);
}

void candidate_pipelines(const detail::PipelineSetup& setup, const void*,
                         std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, CullCandidatesPass, "vsm_cull_candidates", CandidateDefines).value;
}

void encode_candidates(detail::PassContext& ctx, const void*) {
  const auto local = read_local<CandidateLocal>(ctx);
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView words = ctx.buffer(BufRef{local.words});
  if (!pipeline || !words.buffer || !has(chain_bit::Assign)) {
    return;
  }
  if (frame_held()) {
    ctx.encoder().ClearBuffer(words.buffer, words.offset, std::min(words.size, CullWordsBytes));
    mark_encoded(chain_bit::CullCandidates);
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(Slots / CullWorkgroup);
  pass.End();
  mark_encoded(chain_bit::CullCandidates);
}

void declare_list(detail::InternalPassBuilder& builder, const PassSetup& setup, const void* state) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  compact::declare(builder, setup, *static_cast<const compact::GraphJob*>(state));
}

void encode_list(detail::PassContext& ctx, const void* state) {
  if (!has(chain_bit::CullCandidates)) {
    return;
  }
  compact::encode_pass(ctx, state);
  mark_encoded(chain_bit::CullList);
}

void encode_scan(detail::PassContext& ctx, const void* state) {
  if (!has(chain_bit::CullCount)) {
    return;
  }
  compact::encode_pass(ctx, state);
  mark_encoded(chain_bit::CullScan);
}

void declare_pairs(detail::InternalPassBuilder& builder, const PassSetup& setup, bool emit) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  const std::array<PersistentHandle, 3> geometry = geometry_handles();
  PairsLocal local;
  local.list = builder.use_buffer(CullSlots, Access::StorageRead).index;
  local.history = builder.import_persistent(History, handle(History), Access::StorageRead).index;
  local.clusters = builder.import_persistent("scene.clusters", geometry[2], Access::StorageRead).index;
  local.boxes = builder.import_persistent(ChunkBoxes, handle(ChunkBoxes), Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::Uniform).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRead).index;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  if (emit) {
    local.target = builder.use_buffer(PageOffsets, Access::StorageRW).index;
    local.pairs = builder.use_buffer(Pairs, Access::StorageRW).index;
  } else {
    local.target = builder
                       .create_buffer(PageCounts, {.size = PageCountBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                      Access::StorageWrite)
                       .index;
  }
  local.args = builder.use_buffer(CullArgs, Access::Indirect).index;
  write_local(builder, local);
}

void declare_count(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  declare_pairs(builder, setup, false);
}

void declare_emit(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  declare_pairs(builder, setup, true);
}

void count_pipelines(const detail::PipelineSetup& setup, const void*,
                     std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, CullCountPass, "vsm_cull_pairs", CountDefines).value;
}

void emit_pipelines(const detail::PipelineSetup& setup, const void*,
                    std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, CullEmitPass, "vsm_cull_pairs", EmitDefines).value;
}

void encode_pairs(detail::PassContext& ctx, uint32_t need, uint32_t bit, bool clear) {
  const auto local = read_local<PairsLocal>(ctx);
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView target = ctx.buffer(BufRef{local.target});
  const BufferView args = ctx.buffer(BufRef{local.args});
  if (!pipeline || !target.buffer || !args.buffer || !has(need)) {
    return;
  }
  if (clear) {
    ctx.encoder().ClearBuffer(target.buffer, target.offset, std::min(target.size, PageCountBytes));
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroupsIndirect(args.buffer, args.offset);
  pass.End();
  mark_encoded(bit);
}

void encode_count(detail::PassContext& ctx, const void*) {
  encode_pairs(ctx, chain_bit::CullList, chain_bit::CullCount, true);
}

void encode_emit(detail::PassContext& ctx, const void*) {
  encode_pairs(ctx, chain_bit::CullSkip, chain_bit::CullEmit, false);
}

void declare_skip(detail::InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  SkipLocal local;
  local.counts = builder.use_buffer(PageCounts, Access::StorageRead).index;
  local.offsets = builder.use_buffer(PageOffsets, Access::StorageRead).index;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.pairs = builder
                    .create_buffer(Pairs, {.size = pairs_header::Bytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                   Access::StorageWrite)
                    .index;
  local.args = builder
                   .create_buffer(RasterArgs,
                                  {.size = RasterArgsBytes,
                                   .extraUsage = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst},
                                  Access::StorageWrite)
                   .index;
  write_local(builder, local);
}

void skip_pipelines(const detail::PipelineSetup& setup, const void*,
                    std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = compute(setup, CullSkipPass, "vsm_cull_skip", SkipDefines).value;
  ids[1] = compute(setup, CullSkipPass, "vsm_cull_args", SkipDefines).value;
}

void encode_skip(detail::PassContext& ctx, const void*) {
  const auto local = read_local<SkipLocal>(ctx);
  const std::array pipelines{shaders::compute_pipeline(ctx.pipeline(0)), shaders::compute_pipeline(ctx.pipeline(1))};
  const BufferView pairs = ctx.buffer(BufRef{local.pairs});
  if (!pipelines[0] || !pipelines[1] || !pairs.buffer || !has(chain_bit::CullScan)) {
    return;
  }
  ctx.encoder().ClearBuffer(pairs.buffer, pairs.offset, uint64_t{pairs_header::Words} * 4);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipelines[0]);
  pass.DispatchWorkgroups(MaxRenderPages / CullWorkgroup);
  pass.SetPipeline(pipelines[1]);
  pass.DispatchWorkgroups(1);
  pass.End();
  mark_encoded(chain_bit::CullSkip);
}

float word_f32(uint32_t w) noexcept { return std::bit_cast<float>(w); }

Box history_bounds(std::span<const uint32_t, HistoryWords> history) noexcept { return history_box(history); }
} // namespace

std::array<PersistentHandle, 3> geometry_handles() noexcept {
  if (g_geometryOverride) {
    return *g_geometryOverride;
  }
  return {scene::streamout::pool_handle(scene::streamout::binding::Indices),
          scene::streamout::pool_handle(scene::streamout::binding::Vertices), scene::meshlets::clusters_handle()};
}

std::array<uint32_t, 2> chunk_box_pack(const Box& box, const Box& bounds) noexcept {
  std::array<uint32_t, 2> out{};
  constexpr auto Quanta = static_cast<float>(ChunkQuanta);
  for (uint32_t i = 0; i < 3; ++i) {
    const float extent = std::max(bounds.hi[i] - bounds.lo[i], 1e-30f);
    const float lo = std::clamp(std::floor((box.lo[i] - bounds.lo[i]) / extent * Quanta), 0.f, Quanta);
    const float hi = std::clamp(std::ceil((box.hi[i] - bounds.lo[i]) / extent * Quanta), 0.f, Quanta);
    out[0] |= static_cast<uint32_t>(lo) << (10 * i);
    out[1] |= static_cast<uint32_t>(hi) << (10 * i);
  }
  return out;
}

Box chunk_box_unpack(const std::array<uint32_t, 2>& words, const Box& bounds) noexcept {
  Box out;
  constexpr auto Quanta = static_cast<float>(ChunkQuanta);
  for (uint32_t i = 0; i < 3; ++i) {
    const float extent = bounds.hi[i] - bounds.lo[i];
    out.lo[i] = bounds.lo[i] + static_cast<float>((words[0] >> (10 * i)) & 1023u) / Quanta * extent;
    out.hi[i] = bounds.lo[i] + static_cast<float>((words[1] >> (10 * i)) & 1023u) / Quanta * extent;
  }
  return out;
}

Box box_through(const Box& local, std::span<const uint32_t, 12> xform) noexcept {
  Box out{.lo = {3.4e38f, 3.4e38f, 3.4e38f}, .hi = {-3.4e38f, -3.4e38f, -3.4e38f}};
  for (uint32_t i = 0; i < 8; ++i) {
    const std::array<float, 3> c{(i & 1) != 0 ? local.hi[0] : local.lo[0], (i & 2) != 0 ? local.hi[1] : local.lo[1],
                                 (i & 4) != 0 ? local.hi[2] : local.lo[2]};
    for (size_t r = 0; r < 3; ++r) {
      const float p = c[0] * word_f32(xform[r * 4]) + c[1] * word_f32(xform[r * 4 + 1]) +
                      c[2] * word_f32(xform[r * 4 + 2]) + word_f32(xform[r * 4 + 3]);
      out.lo[r] = std::min(out.lo[r], p);
      out.hi[r] = std::max(out.hi[r], p);
    }
  }
  return out;
}

TexelRect texel_rect(const ViewRecord& view, const Box& box) noexcept {
  TexelRect out{.lo = {3.4e38f, 3.4e38f}, .hi = {-3.4e38f, -3.4e38f}};
  for (uint32_t i = 0; i < 8; ++i) {
    const std::array<float, 3> c{(i & 1) != 0 ? box.hi[0] : box.lo[0], (i & 2) != 0 ? box.hi[1] : box.lo[1],
                                 (i & 4) != 0 ? box.hi[2] : box.lo[2]};
    const std::array<float, 3> t = view_coords_anchor(view, c);
    for (size_t k = 0; k < 2; ++k) {
      out.lo[k] = std::min(out.lo[k], t[k]);
      out.hi[k] = std::max(out.hi[k], t[k]);
    }
  }
  return out;
}

bool rect_holds_centre(const TexelRect& rect, float slack) noexcept {
  for (size_t k = 0; k < 2; ++k) {
    if (std::ceil(rect.lo[k] - slack - 0.5f) > std::floor(rect.hi[k] + slack - 0.5f)) {
      return false;
    }
  }
  return true;
}

std::array<int32_t, 4> page_rect(const ViewRecord& view, const TexelRect& rect, float slack) noexcept {
  const std::array<float, 2> limit{static_cast<float>(view.pages_x()) - 1.f, static_cast<float>(view.pages_y()) - 1.f};
  std::array<int32_t, 4> out{};
  for (size_t k = 0; k < 2; ++k) {
    const float p0 = std::clamp(std::floor((rect.lo[k] - slack) / PageTexels), -1.f, limit[k] + 1.f);
    const float p1 = std::clamp(std::floor((rect.hi[k] + slack) / PageTexels), -1.f, limit[k] + 1.f);
    out[k] = static_cast<int32_t>(std::max(p0, 0.f));
    out[2 + k] = static_cast<int32_t>(std::min(p1, limit[k]));
  }
  return out;
}

uint32_t row_bits(uint32_t x0, uint32_t x1) noexcept {
  const uint32_t upper = x1 >= 31 ? 0xFFFFFFFFu : (1u << (x1 + 1)) - 1;
  return upper & ~((1u << x0) - 1);
}

uint32_t skip_first(std::span<const uint32_t> counts, uint32_t maxPairs) noexcept {
  if (counts.size() < PageCountWords) {
    return MaxRenderPages;
  }
  uint64_t prefix = 0;
  for (uint32_t r = 0; r < MaxRenderPages; ++r) {
    const uint64_t n = uint64_t{counts[r]} + counts[MaxRenderPages + r];
    if (n != 0 && prefix + n > maxPairs) {
      return r;
    }
    prefix += n;
  }
  return MaxRenderPages;
}

std::vector<CullItem> cull_items(std::span<const uint32_t, HistoryWords> history, std::span<const uint32_t> clusters,
                                 std::span<const uint32_t> chunkBoxes) noexcept {
  std::vector<CullItem> out;
  const Box bounds = history_bounds(history);
  const uint32_t clusterCount = history[hist::ClusterCount] & 0x00FFFFFFu;
  const std::span<const uint32_t, 12> xform = history.subspan<hist::Xform, 12>();
  if (clusterCount != 0) {
    out.reserve(clusterCount);
    for (uint32_t k = 0; k < clusterCount; ++k) {
      const uint32_t index = history[hist::ClusterFirst] + k;
      CullItem item{.box = bounds, .word = index};
      const size_t base = size_t{index} * scene::meshlets::ClusterWords;
      if (base + scene::meshlets::ClusterWords <= clusters.size()) {
        const uint32_t counts = clusters[base + scene::meshlets::cluster::Counts];
        item.posed = ((counts >> scene::meshlets::cluster::BoundsSpaceShift) & 0xFFu) != 0;
        if (!item.posed) {
          Box local;
          for (uint32_t c = 0; c < 3; ++c) {
            local.lo[c] = word_f32(clusters[base + scene::meshlets::cluster::AabbMin + c]);
            local.hi[c] = word_f32(clusters[base + scene::meshlets::cluster::AabbMax + c]);
          }
          item.box = box_through(local, xform);
        }
      }
      out.push_back(item);
    }
    return out;
  }
  const uint32_t chunks = chunk_count(history[hist::IndexCount]);
  out.reserve(chunks);
  for (uint32_t c = 0; c < chunks; ++c) {
    CullItem item{.box = bounds, .word = PairChunk | c, .chunk = true};
    const uint32_t slot = chunk_slot(history[hist::IndexFirst], c);
    if (slot != NoChunk && size_t{slot} * 2 + 1 < chunkBoxes.size()) {
      item.box = chunk_box_unpack({chunkBoxes[size_t{slot} * 2], chunkBoxes[size_t{slot} * 2 + 1]}, bounds);
    }
    out.push_back(item);
  }
  return out;
}

CullReference cull_reference(const Views& views, std::span<const uint32_t> history, std::span<const uint32_t> clusters,
                             std::span<const uint32_t> chunkBoxes, std::span<const uint32_t> renderList, float margin) {
  CullReference out;
  if (renderList.size() < render_list::Words || history.size() < size_t{Slots} * HistoryWords) {
    return out;
  }
  std::vector<uint32_t> rendered(MaxVirtualPages, NoPage);
  const uint32_t count = render_entries(renderList);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t v = render_list::virtual_page(renderList[render_list::Pages + 2 * i]);
    if (v < MaxVirtualPages) {
      rendered[v] = i;
    }
  }
  const uint32_t viewCount = std::min(views.header.viewCount, MaxViews);
  for (uint32_t s = 0; s < Slots; ++s) {
    const auto h = history.subspan(size_t{s} * HistoryWords).first<HistoryWords>();
    constexpr uint32_t Need = hist::Caster | hist::BoxValid;
    if ((h[hist::Flags] & Need) != Need) {
      continue;
    }
    const std::vector<CullItem> items = cull_items(h, clusters, chunkBoxes);
    for (uint32_t i = 0; i < viewCount; ++i) {
      const ViewRecord& view = views.views[i];
      if ((view.view_flags() & view_flag::Active) == 0) {
        continue;
      }
      const auto add = [&](std::vector<uint64_t>& keys, const CullItem& item, const TexelRect& t, float slack) {
        if (!rect_holds_centre(t, slack)) {
          return;
        }
        const std::array<int32_t, 4> r = page_rect(view, t, slack);
        for (int32_t y = r[1]; y <= r[3]; ++y) {
          for (int32_t x = r[0]; x <= r[2]; ++x) {
            const uint32_t v = page_index(view.table_offset(), view.origin[0] + x, view.origin[1] + y);
            if (rendered[v] != NoPage) {
              keys.push_back(pair_key(rendered[v], item.word, s));
            }
          }
        }
      };
      const TexelRect whole = texel_rect(view, history_bounds(h));
      const std::array<int32_t, 4> wr = page_rect(view, whole, margin);
      bool any = false;
      for (int32_t y = wr[1]; y <= wr[3] && !any; ++y) {
        for (int32_t x = wr[0]; x <= wr[2] && !any; ++x) {
          any = rendered[page_index(view.table_offset(), view.origin[0] + x, view.origin[1] + y)] != NoPage;
        }
      }
      if (!any) {
        continue;
      }
      for (const CullItem& item : items) {
        const TexelRect t = texel_rect(view, item.box);
        add(out.exact, item, t, -10.f * CentreSlack);
        add(out.loose, item, t, margin);
      }
    }
  }
  for (std::vector<uint64_t>* keys : {&out.exact, &out.loose}) {
    std::ranges::sort(*keys);
    keys->erase(std::ranges::unique(*keys).begin(), keys->end());
  }
  return out;
}

detail::PassDesc chunk_bounds_pass_desc() {
  return compute_desc(ChunkBoundsPass, 3, declare_chunks, encode_chunks, chunk_pipelines);
}

detail::PassDesc cull_candidates_pass_desc() {
  return compute_desc(CullCandidatesPass, 30, declare_candidates, encode_candidates, candidate_pipelines);
}

detail::PassDesc cull_list_pass_desc() {
  detail::PassDesc desc = compact::graph_pass(std::string{CullListPass}, Stage::OpaqueDone, declare_list, &ListJob);
  desc.cls = detail::PassClass::Required;
  desc.order = 31;
  desc.enabled = chain_pass_enabled;
  desc.encode = encode_list;
  return desc;
}

detail::PassDesc cull_count_pass_desc() {
  return compute_desc(CullCountPass, 32, declare_count, encode_count, count_pipelines);
}

detail::PassDesc cull_scan_pass_desc() {
  detail::PassDesc desc = compact::graph_pass(std::string{CullScanPass}, Stage::OpaqueDone, declare_list, &ScanJob);
  desc.cls = detail::PassClass::Required;
  desc.order = 33;
  desc.enabled = chain_pass_enabled;
  desc.encode = encode_scan;
  return desc;
}

detail::PassDesc cull_skip_pass_desc() {
  return compute_desc(CullSkipPass, 34, declare_skip, encode_skip, skip_pipelines);
}

detail::PassDesc cull_emit_pass_desc() {
  return compute_desc(CullEmitPass, 35, declare_emit, encode_emit, emit_pipelines);
}

namespace testing {
void set_geometry_handles(std::optional<std::array<PersistentHandle, 3>> handles) noexcept {
  g_geometryOverride = handles;
}
} // namespace testing
} // namespace aurora::render::shadows::vsm

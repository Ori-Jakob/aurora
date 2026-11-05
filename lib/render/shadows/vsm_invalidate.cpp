#include "vsm_invalidate.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../geometry/compact.hpp"
#include "../scene/atlas.hpp"
#include "../scene/streamout.hpp"
#include "../../gx/gx.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace aurora::render::shadows::vsm {
namespace {
using detail::PersistentHandle;
using detail::StageData;
namespace record = scene::streamout::record;
namespace instance_flag = scene::streamout::instance_flag;
namespace class_bit = scene::retained::class_bit;

constexpr uint32_t NoRef = UINT16_MAX;
constexpr uint32_t ExcludedClasses =
    class_bit::CpuDeformed | class_bit::Billboard | class_bit::Water | class_bit::Blended;
constexpr uint32_t ExcludedFlags =
    instance_flag::CpuDeformed | instance_flag::Billboard | instance_flag::Water | instance_flag::Blended;
constexpr uint32_t ExcludedCategories = AURORA_CAT_NO_SHADOW_CAST | AURORA_CAT_SKY | AURORA_CAT_DECAL |
                                        AURORA_CAT_FAKE_SHADOW | AURORA_CAT_UI | AURORA_CAT_OVERLAY |
                                        AURORA_CAT_PARTICLE;
constexpr uint32_t MaterialAlphaTest = 1u;
constexpr uint32_t MaterialAlphaPending = 16u;
constexpr uint32_t MaterialAlphaMissing = 64u;
constexpr uint32_t MaterialWords = 8;
constexpr uint32_t ClusterCountMask = 0x00FFFFFFu;
constexpr std::array<std::string_view, 1> DirtyDefines{"VSM_DIRTY"};

struct ScanLocal {
  uint16_t instances = NoRef;
  uint16_t materials = NoRef;
  uint16_t views = NoRef;
  uint16_t history = NoRef;
  uint16_t stats = NoRef;
  uint16_t castBits = NoRef;
  uint16_t slotWords = NoRef;
  uint16_t boxes = NoRef;
};
static_assert(sizeof(ScanLocal) <= detail::PassLocalSize);

struct DirtyLocal {
  uint16_t list = NoRef;
  uint16_t boxes = NoRef;
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t stats = NoRef;
  uint16_t pages = NoRef;
  uint16_t args = NoRef;
};
static_assert(sizeof(DirtyLocal) <= detail::PassLocalSize);

const compact::GraphJob ListJob{
    .op = compact::Op::Compact,
    .capacity = Slots,
    .input = SlotWordsName,
    .output = ListedSlots,
    .scratch = ListScratch,
    .keepMask = slot_word::Changed | slot_word::ChunkNeed,
    .args = DirtyArgs,
    .argsSpec = {.kind = compact::ArgsKind::Dispatch, .scale = 1},
};

std::optional<std::array<PersistentHandle, 2>> g_sceneOverride;

float word_f32(uint32_t w) noexcept { return std::bit_cast<float>(w); }

uint32_t f32_word(float f) noexcept { return std::bit_cast<uint32_t>(f); }

float ordered_to_f32(uint32_t u) noexcept { return word_f32((u & 0x80000000u) != 0 ? (u & 0x7FFFFFFFu) : ~u); }

bool declarable(const detail::PassSetup& setup) noexcept {
  const std::array<PersistentHandle, 2> scene = scene_handles();
  return active() && setup.env != nullptr && setup.env->retained && handle(History).id != 0 && handle(Table).id != 0 &&
         scene[0].id != 0 && scene[1].id != 0;
}

bool chain_pass_enabled(const StageData& stage, const void*) { return chain_enabled(stage); }

void declare_scan(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  const std::array<PersistentHandle, 2> scene = scene_handles();
  ScanLocal local;
  local.instances = builder.import_persistent("scene.instances", scene[0], Access::StorageRead).index;
  local.materials = builder.import_persistent("scene.materials", scene[1], Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.history = builder.import_persistent(History, handle(History), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.castBits = builder.import_persistent(CastBits, handle(CastBits), Access::StorageRW).index;
  local.slotWords =
      builder
          .create_buffer(SlotWordsName, {.size = SlotWordsBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                         Access::StorageWrite)
          .index;
  local.boxes = builder.create_buffer(DirtyBoxesName, {.size = DirtyBoxesBytes}, Access::StorageWrite).index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

void scan_pipelines(const detail::PipelineSetup& setup, const void*,
                    std::span<uint32_t, detail::MaxPassPipelines> ids) {
  const shaders::PipelineId id = shaders::request_compute({
      .label = ScanPass,
      .shader = InvalidateShader,
      .entryPoint = "vsm_scan",
      .layout = setup.layout,
      .layoutKey = setup.layoutHash,
  });
  ids[0] = id.value;
  register_chain_pipeline(id);
}

void encode_scan(detail::PassContext& ctx, const void*) {
  ScanLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView slotWords = ctx.buffer(BufRef{local.slotWords});
  if (!pipeline || !slotWords.buffer) {
    return;
  }
  const bool updated = (encoded_mask() & chain_bit::Update) != 0;
  if (!updated || frame_held()) {
    // A held frame lists no slot, so the dirty pass dispatches nothing and the history keeps what it last saw
    ctx.encoder().ClearBuffer(slotWords.buffer, slotWords.offset, std::min(slotWords.size, SlotWordsBytes));
    if (updated) {
      mark_encoded(chain_bit::Scan);
    }
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(Slots / ScanWorkgroup);
  pass.End();
  mark_encoded(chain_bit::Scan);
}

void declare_list(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void* state) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  compact::declare(builder, setup, *static_cast<const compact::GraphJob*>(state));
}

void encode_list(detail::PassContext& ctx, const void* state) {
  // A rejected compaction zeroes its args, so the dirty pass still reads a valid empty dispatch
  compact::encode_pass(ctx, state);
  mark_encoded(chain_bit::ScanList);
}

void declare_dirty(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  DirtyLocal local;
  local.list = builder.use_buffer(ListedSlots, Access::StorageRead).index;
  local.boxes = builder.use_buffer(DirtyBoxesName, Access::StorageRead).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.pages = builder
                    .create_buffer(DirtyPagesName, {.size = DirtyPagesBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                   Access::StorageWrite)
                    .index;
  local.args = builder.use_buffer(DirtyArgs, Access::Indirect).index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

void dirty_pipelines(const detail::PipelineSetup& setup, const void*,
                     std::span<uint32_t, detail::MaxPassPipelines> ids) {
  const shaders::PipelineId id = shaders::request_compute({
      .label = DirtyPass,
      .shader = InvalidateShader,
      .entryPoint = "vsm_dirty",
      .defines = DirtyDefines,
      .layout = setup.layout,
      .layoutKey = setup.layoutHash,
  });
  ids[0] = id.value;
  register_chain_pipeline(id);
}

void encode_dirty(detail::PassContext& ctx, const void*) {
  DirtyLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView pages = ctx.buffer(BufRef{local.pages});
  const BufferView args = ctx.buffer(BufRef{local.args});
  if (!pipeline || !pages.buffer || !args.buffer) {
    return;
  }
  ctx.encoder().ClearBuffer(pages.buffer, pages.offset, std::min(pages.size, DirtyPagesBytes));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroupsIndirect(args.buffer, args.offset);
  pass.End();
  mark_encoded(chain_bit::Dirty);
}

std::array<float, 3> sub3(const std::array<float, 3>& a, const std::array<float, 3>& b) noexcept {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

void put_box(std::span<uint32_t> out, const Box& box) noexcept {
  for (size_t i = 0; i < 3; ++i) {
    out[i] = f32_word(box.lo[i]);
    out[3 + i] = f32_word(box.hi[i]);
  }
}

Box read_box(std::span<const uint32_t> words) noexcept {
  Box box;
  for (size_t i = 0; i < 3; ++i) {
    box.lo[i] = word_f32(words[i]);
    box.hi[i] = word_f32(words[3 + i]);
  }
  return box;
}
} // namespace

std::array<PersistentHandle, 2> scene_handles() noexcept {
  if (g_sceneOverride) {
    return *g_sceneOverride;
  }
  return {scene::streamout::pool_handle(scene::streamout::binding::Instances), scene::atlas::materials_handle()};
}

bool is_caster(const scene::retained::InstanceInfo& info) noexcept {
  return info.slot != scene::retained::TransientSlot && (info.classBits & ExcludedClasses) == 0 &&
         (info.categories & ExcludedCategories) == 0;
}

bool record_is_caster(uint32_t flags, uint32_t categories) noexcept {
  return (flags & instance_flag::Valid) != 0 && (flags & ExcludedFlags) == 0 && (categories & ExcludedCategories) == 0;
}

FrameSignals frame_signals(const StageData& stage) noexcept {
  FrameSignals out;
  if (const scene::retained::Frame* frame = stage.retained.get(); frame != nullptr) {
    out.retainedReset = std::ranges::any_of(frame->events, [](const scene::retained::Event& event) {
      return event.type == scene::retained::EventType::Reset;
    });
    const scene::streamout::Plan* plan = frame->plan.get();
    out.clearInstances = plan != nullptr && plan->clearInstances;
    out.unapplied = plan == nullptr || !scene::streamout::applied(plan->serial);
  }
  out.sceneEpoch = stage.sceneHints != nullptr && stage.sceneHints->sceneEpochChanged;
  return out;
}

SlotScan scan_slot(std::span<const uint32_t, HistoryWords> prev, std::span<const uint32_t, RecordWords> rec,
                   std::span<const uint32_t> materials, const ViewsHeader& header) noexcept {
  SlotScan out;
  const bool full = (header.flags & header_flag::Full) != 0;
  const uint32_t w0 = rec[record::Flags];
  const uint32_t w1 = rec[record::Categories];
  const uint32_t hash = rec[record::IdentityHash];
  const uint32_t streamFrame = rec[record::StreamFrame];
  const uint32_t hflags = prev[hist::Flags];
  const bool valid = (w0 & instance_flag::Valid) != 0;
  const bool casterClass = record_is_caster(w0, w1);
  const bool sameStream = streamFrame == prev[hist::StreamFrame];
  const bool occupant =
      (hflags & hist::Seen) != 0 && valid && ((hflags & hist::Valid) == 0 || hash != prev[hist::Hash]);
  out.pending = valid && sameStream && (occupant || (hflags & hist::PendingStream) != 0);
  out.pendingCaster = out.pending && casterClass;
  out.castNow = casterClass && !out.pending;
  const bool wasCaster = (hflags & hist::Caster) != 0;

  bool cutout = false;
  bool alphaPending = false;
  const uint32_t material = rec[record::Material] & 0xFFFFu;
  if (material != scene::streamout::NoMaterial && size_t{material + 1} * MaterialWords <= materials.size()) {
    const uint32_t flags = materials[size_t{material} * MaterialWords + 1] >> 16;
    const bool tested = (flags & MaterialAlphaTest) != 0;
    cutout = tested && (flags & (MaterialAlphaPending | MaterialAlphaMissing)) == 0;
    alphaPending = tested && (flags & MaterialAlphaPending) != 0;
  }
  const bool wasCutout = (hflags & hist::Cutout) != 0;
  const bool player = (w1 & AURORA_CAT_PLAYER) != 0;

  std::array<std::array<float, 4>, 3> xform{};
  const std::array<float, 3> anchorHi{header.anchorHi[0], header.anchorHi[1], header.anchorHi[2]};
  const std::array<float, 3> anchorLo{header.anchorLo[0], header.anchorLo[1], header.anchorLo[2]};
  const std::array<float, 3> worldHi{word_f32(rec[record::World + 3]), word_f32(rec[record::World + 7]),
                                     word_f32(rec[record::World + 11])};
  const std::array<float, 3> worldLo{word_f32(rec[record::World + 12]), word_f32(rec[record::World + 13]),
                                     word_f32(rec[record::World + 14])};
  const std::array<float, 3> hiRel = sub3(worldHi, anchorHi);
  const std::array<float, 3> loRel = sub3(worldLo, anchorLo);
  for (size_t r = 0; r < 3; ++r) {
    for (size_t c = 0; c < 3; ++c) {
      xform[r][c] = word_f32(rec[record::World + r * 4 + c]);
    }
    xform[r][3] = hiRel[r] + loRel[r];
  }
  bool boxValid = true;
  for (size_t i = 0; i < 3; ++i) {
    const uint32_t lo = rec[record::CurAabb + i];
    const uint32_t hi = rec[record::CurAabb + 3 + i];
    boxValid = boxValid && lo <= hi && lo != 0xFFFFFFFFu && hi != 0;
  }
  Box box;
  if (boxValid) {
    std::array<float, 3> lo{};
    std::array<float, 3> hi{};
    for (size_t i = 0; i < 3; ++i) {
      lo[i] = ordered_to_f32(rec[record::CurAabb + i]);
      hi[i] = ordered_to_f32(rec[record::CurAabb + 3 + i]);
    }
    box.lo = {3.4e38f, 3.4e38f, 3.4e38f};
    box.hi = {-3.4e38f, -3.4e38f, -3.4e38f};
    for (uint32_t i = 0; i < 8; ++i) {
      const std::array<float, 3> c{(i & 1) != 0 ? hi[0] : lo[0], (i & 2) != 0 ? hi[1] : lo[1],
                                   (i & 4) != 0 ? hi[2] : lo[2]};
      for (size_t r = 0; r < 3; ++r) {
        const float p = c[0] * xform[r][0] + c[1] * xform[r][1] + c[2] * xform[r][2] + xform[r][3];
        box.lo[r] = std::min(box.lo[r], p);
        box.hi[r] = std::max(box.hi[r], p);
      }
    }
    for (size_t r = 0; r < 3; ++r) {
      box.lo[r] -= header.texel0;
      box.hi[r] += header.texel0;
    }
  }

  uint32_t reasons = 0;
  reasons |= out.castNow != wasCaster ? reason::Caster : 0u;
  reasons |= hash != prev[hist::Hash] ? reason::Hash : 0u;
  reasons |= !sameStream ? reason::Stream : 0u;
  reasons |= rec[record::WorldFrame] != prev[hist::WorldFrame] ? reason::World : 0u;
  reasons |= rec[record::MaterialFrame] != prev[hist::MaterialFrame] && (wasCutout || cutout) ? reason::Material : 0u;
  reasons |= rec[record::UvFrame] != prev[hist::UvFrame] && cutout ? reason::Uv : 0u;
  reasons |= (hflags & hist::AlphaPending) != 0 && !alphaPending ? reason::Alpha : 0u;
  reasons |= wasCutout != cutout ? reason::Cutout : 0u;
  reasons |= out.castNow && (w0 & instance_flag::GpuArrays) != 0 ? reason::Arrays : 0u;
  out.reasons = reasons;
  out.changed = !full && (out.castNow || wasCaster) && reasons != 0;
  const bool chunked = out.castNow && (rec[record::ClusterCount] & ClusterCountMask) == 0;
  out.chunkNeed = chunked && (out.changed || full || (hflags & hist::Chunked) == 0);

  if (out.changed || out.chunkNeed) {
    uint32_t boxFlags = 0;
    if (out.changed && wasCaster && (hflags & hist::BoxValid) != 0) {
      std::copy_n(prev.begin() + hist::Box, 6, out.box.begin() + dirty_box::Old);
      boxFlags |= dirty_box::HasOld | ((hflags & hist::Player) != 0 ? dirty_box::OldPlayer : 0u);
    }
    if (out.changed && out.castNow && boxValid) {
      put_box(std::span{out.box}.subspan(dirty_box::New, 6), box);
      boxFlags |= dirty_box::HasNew | (player ? dirty_box::NewPlayer : 0u);
    }
    out.box[dirty_box::Flags] = boxFlags;
  }

  uint32_t nf = hist::Seen;
  nf |= valid ? hist::Valid : 0u;
  nf |= out.castNow ? hist::Caster : 0u;
  nf |= cutout ? hist::Cutout : 0u;
  nf |= alphaPending ? hist::AlphaPending : 0u;
  nf |= player ? hist::Player : 0u;
  nf |= chunked ? hist::Chunked : 0u;
  nf |= out.pending ? hist::PendingStream : 0u;
  nf |= boxValid ? hist::BoxValid : 0u;
  std::array<uint32_t, HistoryWords>& h = out.history;
  h[hist::Flags] = nf;
  h[hist::Hash] = hash;
  h[hist::StreamFrame] = streamFrame;
  h[hist::WorldFrame] = rec[record::WorldFrame];
  h[hist::MaterialFrame] = rec[record::MaterialFrame];
  h[hist::UvFrame] = rec[record::UvFrame];
  h[hist::ClusterFirst] = rec[record::ClusterFirst];
  h[hist::ClusterCount] = rec[record::ClusterCount];
  h[hist::IndexFirst] = rec[record::IndexFirst];
  h[hist::IndexCount] = rec[record::IndexCount];
  h[hist::VertexFirst] = rec[record::VertexFirst];
  if (boxValid) {
    put_box(std::span{h}.subspan(hist::Box, 6), box);
  }
  for (size_t r = 0; r < 3; ++r) {
    for (size_t c = 0; c < 4; ++c) {
      h[hist::Xform + r * 4 + c] = f32_word(xform[r][c]);
    }
  }
  return out;
}

Box history_box(std::span<const uint32_t, HistoryWords> history) noexcept {
  return read_box(history.subspan(hist::Box, 6));
}

std::array<int32_t, 4> box_rect(const ViewRecord& view, const Box& box, float grow) noexcept {
  std::array<float, 2> lo{3.4e38f, 3.4e38f};
  std::array<float, 2> hi{-3.4e38f, -3.4e38f};
  for (uint32_t i = 0; i < 8; ++i) {
    const std::array<float, 3> c{(i & 1) != 0 ? box.hi[0] : box.lo[0], (i & 2) != 0 ? box.hi[1] : box.lo[1],
                                 (i & 4) != 0 ? box.hi[2] : box.lo[2]};
    const std::array<float, 3> t = view_coords_anchor(view, c);
    for (size_t k = 0; k < 2; ++k) {
      lo[k] = std::min(lo[k], t[k]);
      hi[k] = std::max(hi[k], t[k]);
    }
  }
  const std::array<float, 2> limit{static_cast<float>(view.pages_x()) - 1.f, static_cast<float>(view.pages_y()) - 1.f};
  std::array<int32_t, 4> out{};
  for (size_t k = 0; k < 2; ++k) {
    const float p0 = std::clamp(std::floor((lo[k] - grow) / PageTexels), -1.f, limit[k] + 1.f);
    const float p1 = std::clamp(std::floor((hi[k] + grow) / PageTexels), -1.f, limit[k] + 1.f);
    out[k] = static_cast<int32_t>(std::max(p0, 0.f));
    out[2 + k] = static_cast<int32_t>(std::min(p1, limit[k]));
  }
  return out;
}

void dirty_footprint(const Views& views, std::span<const uint32_t, DirtyBoxWords> box, std::span<uint32_t> planes,
                     float grow) noexcept {
  if (planes.size() < DirtyPagesWords) {
    return;
  }
  const uint32_t flags = box[dirty_box::Flags];
  const uint32_t count = std::min(views.header.viewCount, MaxViews);
  for (uint32_t b = 0; b < 2; ++b) {
    if ((flags & (dirty_box::HasOld << b)) == 0) {
      continue;
    }
    const Box bounds = read_box(box.subspan(b == 0 ? dirty_box::Old : dirty_box::New, 6));
    const bool priority = (flags & (dirty_box::OldPlayer << b)) != 0;
    for (uint32_t v = 0; v < count; ++v) {
      const ViewRecord& view = views.views[v];
      if ((view.view_flags() & view_flag::Active) == 0) {
        continue;
      }
      const std::array<int32_t, 4> r = box_rect(view, bounds, grow);
      for (int32_t y = r[1]; y <= r[3]; ++y) {
        for (int32_t x = r[0]; x <= r[2]; ++x) {
          const uint32_t index = page_index(view.table_offset(), view.origin[0] + x, view.origin[1] + y);
          const uint32_t bit = 1u << (index & 31u);
          planes[index >> 5] |= bit;
          if (priority) {
            planes[DirtyPlaneWords + (index >> 5)] |= bit;
          }
        }
      }
    }
  }
}

CasterObject CasterObjects::find(uint64_t id) const noexcept {
  if (m_objects == nullptr) {
    return {};
  }
  const std::vector<scene::retained::FilteredObject>& objects = m_objects->objects;
  const auto it = std::ranges::lower_bound(objects, id, {}, &scene::retained::FilteredObject::id);
  if (it == objects.end() || it->id != id) {
    return {};
  }
  CasterObject out;
  out.cover = it->passing == 0   ? CasterCover::ExcludedOnly
              : it->failing == 0 ? CasterCover::Full
                                 : CasterCover::Partial;
  if (size_t{it->firstSlot} + it->slotCount <= m_objects->slots.size()) {
    out.slots = std::span{m_objects->slots}.subspan(it->firstSlot, it->slotCount);
  }
  return out;
}

CasterObjects caster_objects() noexcept { return CasterObjects{scene::retained::covered_filtered()}; }

detail::PassDesc scan_pass_desc() {
  return {
      .name = std::string{ScanPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 1,
      .declare = declare_scan,
      .enabled = chain_pass_enabled,
      .encode = encode_scan,
      .pipelines = scan_pipelines,
  };
}

detail::PassDesc scan_list_pass_desc() {
  detail::PassDesc desc = compact::graph_pass(std::string{ScanListPass}, Stage::OpaqueDone, declare_list, &ListJob);
  desc.cls = detail::PassClass::Required;
  desc.order = 2;
  desc.enabled = chain_pass_enabled;
  desc.encode = encode_list;
  return desc;
}

detail::PassDesc dirty_pass_desc() {
  return {
      .name = std::string{DirtyPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 8,
      .declare = declare_dirty,
      .enabled = chain_pass_enabled,
      .encode = encode_dirty,
      .pipelines = dirty_pipelines,
  };
}

namespace testing {
void set_scene_handles(std::optional<std::array<detail::PersistentHandle, 2>> handles) noexcept {
  g_sceneOverride = handles;
}
} // namespace testing
} // namespace aurora::render::shadows::vsm

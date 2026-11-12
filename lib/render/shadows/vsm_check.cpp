#include "vsm_check.hpp"

#include "vsm_cull.hpp"
#include "vsm_invalidate.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../scene/atlas.hpp"
#include "../scene/meshlets.hpp"
#include "../scene/streamout_layout.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace aurora::render::shadows::vsm::check {
namespace {
constexpr Module Log{"aurora::render::shadows"};
constexpr uint32_t ShutdownWaitMs = 2000;
constexpr uint16_t NoRef = UINT16_MAX;
constexpr uint32_t StatsRing = 8;
constexpr uint64_t NoFrame = UINT64_MAX;
constexpr float RectSlack = 0.01f;
constexpr uint64_t RecordsBytes = scene::streamout::InstancePoolBytes;
constexpr uint64_t MaterialsBytes = scene::atlas::MaterialsBytes;
constexpr uint64_t ViewsOffset = 0;
constexpr uint64_t TableOffset = ViewsOffset + ViewsBytes;
constexpr uint64_t PhysOffset = TableOffset + TableBytes;
constexpr uint64_t StatsOffset = PhysOffset + PhysBytes;
constexpr uint64_t RecordsOffset = StatsOffset + StatsBytes;
constexpr uint64_t MaterialsOffset = RecordsOffset + RecordsBytes;
constexpr uint64_t HistoryOffset = MaterialsOffset + MaterialsBytes;
constexpr uint64_t PrevHistoryOffset = HistoryOffset + HistoryBytes;
constexpr uint64_t CastBitsOffset = PrevHistoryOffset + HistoryBytes;
constexpr uint64_t DirtyOffset = CastBitsOffset + CastBitsBytes;
constexpr uint64_t RequestsOffset = DirtyOffset + DirtyPagesBytes;
constexpr uint64_t RenderListOffset = RequestsOffset + RequestsBytes;
constexpr uint64_t ClustersOffset = RenderListOffset + render_list::Bytes;
constexpr uint64_t ChunkBoxesOffset = ClustersOffset + scene::meshlets::PoolBytes;
constexpr uint64_t PairsOffset = ChunkBoxesOffset + ChunkBoxesBytes;
constexpr uint64_t PageCountsOffset = PairsOffset + pairs_header::Bytes;
constexpr uint64_t ReadbackBytes = PageCountsOffset + PageCountBytes;

enum Ref : uint32_t {
  ViewsRef,
  TableRef,
  PhysRef,
  StatsRef,
  RecordsRef,
  MaterialsRef,
  HistoryRef,
  CastRef,
  DirtyRef,
  RequestsRef,
  RenderListRef,
  ReceiversRef,
  ClustersRef,
  ChunkBoxesRef,
  PairsRef,
  PageCountsRef,
  RefCount
};

struct CheckLocal {
  std::array<uint16_t, RefCount> refs{};
};
static_assert(sizeof(CheckLocal) <= detail::PassLocalSize);

struct CheckParams {
  uint64_t hintFrame = 0;
  uint32_t set = 0;
  uint32_t reserved = 0;
};

struct FrameStats {
  uint64_t seq = 0;
  uint64_t hintFrame = 0;
  uint32_t flags = 0;
  uint32_t reasons = 0;
  std::array<uint32_t, StatCount> counters{};
};

struct RingSlot {
  wgpu::Buffer buffer;
  bool busy = false;
};

std::optional<Config> g_override;
std::mutex g_mutex;
Result g_result;
wgpu::Buffer g_buffer;
Views g_lastViews;
uint64_t g_lastFrame = NoFrame;
uint64_t g_prevFrame = NoFrame;
bool g_warnedViews = false;
std::array<RingSlot, StatsRing> g_ring;
std::vector<FrameStats> g_frames;
uint64_t g_seq = 0;
MarkParams g_markParams;
uint64_t g_markFrame = NoFrame;
std::vector<std::array<double, 3>> g_probes;

struct Capture {
  Views views;
  uint64_t hintFrame = 0;
  bool withPrev = false;
  bool marked = false;
  MarkParams mark;
  uint64_t receiverBytes = 0;
};

bool parse_u64(std::string_view text, uint64_t& out) noexcept {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
  return error == std::errc{} && end == text.data() + text.size() && !text.empty();
}

const Config& env_config() noexcept {
  static const Config config = [] {
    const char* value = SDL_getenv("AURORA_RENDER_VSM_CHECK");
    if (value == nullptr) {
      return Config{};
    }
    if (const std::optional<Config> parsed = parse(value)) {
      Log.warn("AURORA_RENDER_VSM_CHECK={}: checking the virtual shadow page tables on frames {} to {}", value,
               parsed->first, parsed->last);
      return *parsed;
    }
    Log.warn("AURORA_RENDER_VSM_CHECK={} is not 1, 1,<first>,<last> or 1,<first>,<last>,full=<frame>; ignored", value);
    return Config{};
  }();
  return config;
}

bool in_window(uint64_t hintFrame) noexcept {
  const Config& c = config();
  return c.on && hintFrame >= c.first && hintFrame <= c.last;
}

uint32_t allocated_entries(std::span<const uint32_t> table) noexcept {
  uint32_t count = 0;
  for (const uint32_t e : table) {
    count += (e & entry::Allocated) != 0 ? 1 : 0;
  }
  return count;
}

wgpu::Buffer readback_buffer(const char* label, uint64_t size) {
  const wgpu::BufferDescriptor descriptor{
      .label = label,
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = size,
  };
  return webgpu::g_device.CreateBuffer(&descriptor);
}

std::string level_list(const std::array<uint32_t, SunLevels>& counts) {
  std::string out;
  for (uint32_t level = 0; level < SunLevels; ++level) {
    out += fmt::format("{}{}", level == 0 ? "" : " ", counts[level]);
  }
  return out;
}

void finish_readback(const wgpu::Buffer& buffer, const Capture& capture, std::span<const uint32_t> receivers) {
  const Views& expected = capture.views;
  const uint64_t hintFrame = capture.hintFrame;
  const auto* bytes = static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, ReadbackBytes));
  std::string error = "the readback could not be mapped";
  uint32_t allocated = 0;
  ScanCounts scan;
  AllocSummary alloc;
  CullSummary cull;
  std::array<uint32_t, StatCount> stats{};
  if (bytes != nullptr) {
    std::vector<uint32_t> words(ReadbackBytes / 4);
    std::memcpy(words.data(), bytes, ReadbackBytes);
    const std::span<const uint32_t> all{words};
    const auto region = [&](uint64_t offset, uint64_t size) { return all.subspan(offset / 4, size / 4); };
    const std::span<const uint32_t> table = region(TableOffset, TableBytes);
    const std::span<const uint32_t> renderList = region(RenderListOffset, render_list::Bytes);
    std::ranges::copy(region(StatsOffset, StatsBytes), stats.begin());
    error = verify(expected, region(ViewsOffset, ViewsBytes), table, region(PhysOffset, PhysBytes));
    if (error.empty()) {
      const ScanReadback scanned{
          .prevHistory = capture.withPrev ? region(PrevHistoryOffset, HistoryBytes) : std::span<const uint32_t>{},
          .records = region(RecordsOffset, RecordsBytes),
          .materials = region(MaterialsOffset, MaterialsBytes),
          .history = region(HistoryOffset, HistoryBytes),
          .castBits = region(CastBitsOffset, CastBitsBytes),
          .dirtyPages = region(DirtyOffset, DirtyPagesBytes),
          .table = table,
          .stats = region(StatsOffset, StatsBytes),
          .renderList = renderList,
      };
      error = verify_scan(expected, scanned, scan);
    }
    if (error.empty()) {
      const AllocReadback readback{
          .requests = region(RequestsOffset, RequestsBytes),
          .table = table,
          .phys = region(PhysOffset, PhysBytes),
          .renderList = renderList,
          .stats = region(StatsOffset, StatsBytes),
          .receivers = receivers,
          .params = capture.marked ? &capture.mark : nullptr,
      };
      error = verify_alloc(expected, readback, alloc);
    }
    if (error.empty()) {
      const CullReadback culled{
          .history = region(HistoryOffset, HistoryBytes),
          .clusters = region(ClustersOffset, scene::meshlets::PoolBytes),
          .chunkBoxes = region(ChunkBoxesOffset, ChunkBoxesBytes),
          .renderList = renderList,
          .pairs = region(PairsOffset, pairs_header::Bytes),
          .pageCounts = region(PageCountsOffset, PageCountBytes),
      };
      error = verify_cull(expected, culled, cull);
    }
    allocated = allocated_entries(table);
  }
  Log.info(
      "VSM: alloc frame {}: requested {} [{}], cached {} [{}], rendered {} [{}] (priority {}, new {}, reuse {}; "
      "budget {}{}), deferred new {} reuse {}, overflow {}, player deferred {}, evicted {}, freed dirty {}, "
      "hits {}; receivers {} ({} near a boundary); pt and phys hash {:016x}",
      hintFrame, alloc.requested, level_list(alloc.requestedLevels), alloc.cached, level_list(alloc.cachedLevels),
      alloc.rendered, level_list(alloc.renderedLevels), stats[stat::NeedPriority], stats[stat::NeedNew],
      stats[stat::NeedReuse], expected.header.budget,
      (expected.header.flags & header_flag::Exempt) != 0 ? ", exempt" : "", stats[stat::DeferredNew],
      stats[stat::DeferredReuse], stats[stat::OverflowPool], stats[stat::PlayerDeferred], stats[stat::Evicted],
      stats[stat::FreedDirty], stats[stat::Hits], alloc.receivers, alloc.ambiguous, alloc.hash);
  if (cull.checked) {
    const auto share = [&](uint64_t part) {
      return cull.gpuPairs != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(cull.gpuPairs) : 0.0;
    };
    Log.info(
        "VSM: cull frame {}: {} pairs on {} pages (max {} per page, {} on in-place check pages), {} exact, "
        "amplification {:.2f}, loose {}; {} skipped pages; cutout {:.1f}%, chunk {:.1f}%, posed {:.1f}%; {} caster "
        "slots",
        hintFrame, cull.gpuPairs, cull.pages, cull.maxPerPage, cull.checkPairs, cull.exactPairs,
        cull.exactPairs != 0 ? static_cast<double>(cull.gpuPairs) / static_cast<double>(cull.exactPairs) : 0.0,
        cull.loosePairs, cull.skipped, share(cull.cutout), share(cull.chunk), share(cull.posed),
        stats[stat::CullSlots]);
  }
  std::lock_guard lock{g_mutex};
  ++g_result.readbacks;
  g_result.entries += allocated;
  g_result.culled += cull.checked ? 1 : 0;
  g_result.pairs += cull.gpuPairs;
  g_result.exactPairs += cull.exactPairs;
  g_result.skippedPages += cull.skipped;
  g_result.fulls += (expected.header.flags & header_flag::Full) != 0 ? 1 : 0;
  g_result.referenced += scan.referenced ? 1 : 0;
  g_result.changed += scan.changed;
  g_result.footprint += scan.footprint;
  g_result.marked += alloc.marked ? 1 : 0;
  g_result.receivers += alloc.receivers;
  g_result.ambiguous += alloc.ambiguous;
  g_result.requested += alloc.requested;
  g_result.rendered += alloc.rendered;
  g_result.uncommitted += alloc.committed || alloc.rendered == 0 ? 0 : 1;
  if (!error.empty()) {
    ++g_result.failures;
    Log.error("VSM: check frame {} FAILED: {}", hintFrame, error);
  }
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  const std::array<detail::PersistentHandle, 2> scene = scene_handles();
  if (!config().on || !active() || setup.env == nullptr || !setup.env->retained || handle(Table).id == 0 ||
      scene[0].id == 0 || scene[1].id == 0 || geometry_handles()[2].id == 0) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.side_effect();
  CheckLocal local;
  local.refs.fill(NoRef);
  const auto copy = [&](std::string_view name, detail::PersistentHandle h) {
    return builder.import_persistent(name, h, Access::CopySrc).index;
  };
  local.refs[ViewsRef] = copy(ViewsName, handle(ViewsName));
  local.refs[TableRef] = copy(Table, handle(Table));
  local.refs[PhysRef] = copy(Phys, handle(Phys));
  local.refs[StatsRef] = copy(StatsName, handle(StatsName));
  local.refs[RecordsRef] = copy("scene.instances", scene[0]);
  local.refs[MaterialsRef] = copy("scene.materials", scene[1]);
  local.refs[HistoryRef] = copy(History, handle(History));
  local.refs[CastRef] = copy(CastBits, handle(CastBits));
  local.refs[DirtyRef] = builder.use_buffer(DirtyPagesName, Access::CopySrc).index;
  local.refs[RequestsRef] = builder.use_buffer(Requests, Access::CopySrc).index;
  local.refs[RenderListRef] = builder.use_buffer(RenderList, Access::CopySrc).index;
  local.refs[ReceiversRef] = builder.use_buffer(MarkReceivers, Access::CopySrc).index;
  local.refs[ClustersRef] = copy("scene.clusters", geometry_handles()[2]);
  local.refs[ChunkBoxesRef] = copy(ChunkBoxes, handle(ChunkBoxes));
  local.refs[PairsRef] = builder.use_buffer(Pairs, Access::CopySrc).index;
  local.refs[PageCountsRef] = builder.use_buffer(PageCounts, Access::CopySrc).index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

bool enabled(const detail::StageData& stage, const void*) {
  if (!config().on || webgpu::g_backendType == wgpu::BackendType::Null || stage.sceneHints == nullptr) {
    return false;
  }
  const uint64_t h = stage.sceneHints->hintFrame;
  return (in_window(h) || check_frame(h) || check_frame(h + 1)) && chain_enabled(stage);
}

void stage_params(const detail::StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(CheckParams)) {
    return;
  }
  const CheckParams params{.hintFrame = stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0, .set = 1};
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void copy_stats(const wgpu::CommandEncoder& encoder, const BufferView& stats, uint64_t hintFrame) {
  FrameStats frame{.hintFrame = hintFrame};
  if (g_lastFrame == hintFrame) {
    frame.flags = g_lastViews.header.flags;
    frame.reasons = g_lastViews.header.fullReason;
  }
  wgpu::Buffer buffer;
  size_t index = 0;
  {
    std::lock_guard lock{g_mutex};
    const auto it = std::ranges::find_if(g_ring, [](const RingSlot& slot) { return !slot.busy; });
    if (it == g_ring.end()) {
      ++g_result.statsSkipped;
      return;
    }
    if (!it->buffer) {
      it->buffer = readback_buffer("VSM check stats", StatsBytes);
    }
    it->busy = true;
    buffer = it->buffer;
    index = static_cast<size_t>(it - g_ring.begin());
    frame.seq = g_seq++;
  }
  encoder.CopyBufferToBuffer(stats.buffer, stats.offset, buffer, 0, StatsBytes);
  detail::queue_after_frame([buffer, index, frame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, StatsBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, index, frame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      FrameStats out = frame;
                      const bool mapped = status == wgpu::MapAsyncStatus::Success;
                      if (mapped) {
                        if (const void* bytes = buffer.GetConstMappedRange(0, StatsBytes); bytes != nullptr) {
                          std::memcpy(out.counters.data(), bytes, StatsBytes);
                        }
                        buffer.Unmap();
                      }
                      std::lock_guard lock{g_mutex};
                      if (mapped) {
                        g_frames.push_back(out);
                      }
                      g_ring[index].busy = false;
                    });
  });
}

void encode(detail::PassContext& ctx, const void*) {
  CheckParams params;
  std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
  CheckLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  std::array<BufferView, RefCount> views;
  for (uint32_t i = 0; i < RefCount; ++i) {
    views[i] = local.refs[i] != NoRef ? ctx.buffer(BufRef{local.refs[i]}) : BufferView{};
  }
  if (params.set == 0 || std::ranges::any_of(views, [](const BufferView& v) { return !v.buffer; })) {
    return;
  }
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  const uint64_t hintFrame = params.hintFrame;
  if (in_window(hintFrame)) {
    copy_stats(encoder, views[StatsRef], hintFrame);
  }
  const auto copy = [&](Ref ref, uint64_t offset, uint64_t size, const wgpu::Buffer& target) {
    encoder.CopyBufferToBuffer(views[ref].buffer, views[ref].offset, target, offset, std::min(size, views[ref].size));
  };
  if (check_frame(hintFrame + 1)) {
    // The scan overwrites the history it compares against, so the frame before each check frame keeps a copy
    std::lock_guard lock{g_mutex};
    g_prevFrame = NoFrame;
    if (!g_result.busy) {
      const uint64_t need = ReadbackBytes + (views[ReceiversRef].buffer ? views[ReceiversRef].size & ~uint64_t{3} : 0);
      if (!g_buffer || g_buffer.GetSize() < need) {
        g_buffer = readback_buffer("VSM check readback", need);
      }
      copy(HistoryRef, PrevHistoryOffset, HistoryBytes, g_buffer);
      g_prevFrame = hintFrame;
    }
  }
  if (!check_frame(hintFrame)) {
    return;
  }
  auto capture = std::make_shared<Capture>();
  const bool hasViews = g_lastFrame == hintFrame;
  if (hasViews) {
    capture->views = g_lastViews;
  }
  capture->hintFrame = hintFrame;
  capture->marked = g_markFrame == hintFrame && views[ReceiversRef].buffer;
  capture->mark = g_markParams;
  capture->receiverBytes = capture->marked ? views[ReceiversRef].size & ~uint64_t{3} : 0;
  const uint64_t total = ReadbackBytes + capture->receiverBytes;
  wgpu::Buffer buffer;
  {
    std::lock_guard lock{g_mutex};
    ++g_result.frames;
    if (!hasViews) {
      ++g_result.failures;
      if (!std::exchange(g_warnedViews, true)) {
        Log.error("VSM: check frame {} FAILED: vsm.update wrote no views this frame", hintFrame);
      }
      return;
    }
    if (g_result.busy) {
      ++g_result.skipped;
      return;
    }
    if (!g_buffer || g_buffer.GetSize() < total) {
      g_buffer = readback_buffer("VSM check readback", total);
      g_prevFrame = NoFrame;
    }
    capture->withPrev = g_prevFrame != NoFrame && g_prevFrame + 1 == hintFrame;
    g_prevFrame = NoFrame;
    g_result.busy = true;
    buffer = g_buffer;
  }
  copy(ViewsRef, ViewsOffset, ViewsBytes, buffer);
  copy(TableRef, TableOffset, TableBytes, buffer);
  copy(PhysRef, PhysOffset, PhysBytes, buffer);
  copy(StatsRef, StatsOffset, StatsBytes, buffer);
  copy(RecordsRef, RecordsOffset, RecordsBytes, buffer);
  copy(MaterialsRef, MaterialsOffset, MaterialsBytes, buffer);
  copy(HistoryRef, HistoryOffset, HistoryBytes, buffer);
  copy(CastRef, CastBitsOffset, CastBitsBytes, buffer);
  copy(DirtyRef, DirtyOffset, DirtyPagesBytes, buffer);
  copy(RequestsRef, RequestsOffset, RequestsBytes, buffer);
  copy(RenderListRef, RenderListOffset, render_list::Bytes, buffer);
  copy(ClustersRef, ClustersOffset, scene::meshlets::PoolBytes, buffer);
  copy(ChunkBoxesRef, ChunkBoxesOffset, ChunkBoxesBytes, buffer);
  copy(PairsRef, PairsOffset, pairs_header::Bytes, buffer);
  copy(PageCountsRef, PageCountsOffset, PageCountBytes, buffer);
  if (capture->receiverBytes != 0) {
    copy(ReceiversRef, ReadbackBytes, capture->receiverBytes, buffer);
  }
  detail::queue_after_frame([buffer, capture, total] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, total, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, capture, total](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      if (status == wgpu::MapAsyncStatus::Success) {
                        std::vector<uint32_t> receivers;
                        if (capture->receiverBytes != 0) {
                          if (const auto* bytes = static_cast<const uint32_t*>(
                                  buffer.GetConstMappedRange(ReadbackBytes, capture->receiverBytes));
                              bytes != nullptr) {
                            receivers.assign(bytes, bytes + capture->receiverBytes / 4);
                          }
                        }
                        finish_readback(buffer, *capture, receivers);
                        buffer.Unmap();
                      }
                      std::lock_guard lock{g_mutex};
                      g_result.busy = false;
                    });
  });
}

uint64_t fnv(uint64_t hash, uint64_t value) noexcept {
  for (int i = 0; i < 8; ++i) {
    hash = (hash ^ ((value >> (8 * i)) & 0xFF)) * 0x100000001B3ull;
  }
  return hash;
}

uint32_t median(std::vector<uint32_t> values) noexcept {
  if (values.empty()) {
    return 0;
  }
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  return *mid;
}

uint32_t peak(const std::vector<uint32_t>& values) noexcept {
  return values.empty() ? 0u : *std::ranges::max_element(values);
}

void log_run(size_t run, std::span<const FrameStats> frames) {
  std::array<uint64_t, StatCount> sums{};
  std::array<uint64_t, StatCount> frameCounts{};
  uint64_t fulls = 0;
  uint64_t held = 0;
  uint64_t hash = 0xCBF29CE484222325ull;
  std::vector<uint32_t> footprint;
  std::vector<uint32_t> player;
  for (const FrameStats& f : frames) {
    fulls += (f.flags & header_flag::Full) != 0 ? 1 : 0;
    held += (f.flags & header_flag::Held) != 0 ? 1 : 0;
    hash = fnv(fnv(fnv(hash, f.hintFrame), f.flags), f.reasons);
    for (uint32_t i = 0; i < StatCount; ++i) {
      sums[i] += f.counters[i];
      frameCounts[i] += f.counters[i] != 0 ? 1 : 0;
      hash = fnv(hash, f.counters[i]);
    }
    if ((f.flags & (header_flag::Full | header_flag::Held)) == 0) {
      footprint.push_back(f.counters[stat::FootprintPages]);
      player.push_back(f.counters[stat::FootprintPriority]);
    }
  }
  const auto by = [&](uint32_t bit) { return sums[stat::ReasonFirst + std::countr_zero(bit)]; };
  const auto on = [&](uint32_t bit) { return frameCounts[stat::ReasonFirst + std::countr_zero(bit)]; };
  Log.info(
      "VSM: invalidation run {}: {} frames ({}-{}), {} full, {} held; changed {} on {} frames (static {} on {}, "
      "immobile {} on {}, player {} on {}, palette {} on {}); world changes on {} frames, stream changes on {}; "
      "pending-stream {} (casters {}), chunk need {}; by caster {} hash {} stream {} world {} material {} uv {} "
      "alpha {} cutout {} arrays {}; non-player other {}",
      run, frames.size(), frames.front().hintFrame, frames.back().hintFrame, fulls, held, sums[stat::Changed],
      frameCounts[stat::Changed], sums[stat::ChangedStatic], frameCounts[stat::ChangedStatic],
      sums[stat::ChangedImmobile], frameCounts[stat::ChangedImmobile], sums[stat::ChangedPlayer],
      frameCounts[stat::ChangedPlayer], sums[stat::ChangedPalette], frameCounts[stat::ChangedPalette],
      on(reason::World), on(reason::Stream), sums[stat::PendingStream], sums[stat::PendingCasters],
      sums[stat::ChunkNeed], by(reason::Caster), by(reason::Hash), by(reason::Stream), by(reason::World),
      by(reason::Material), by(reason::Uv), by(reason::Alpha), by(reason::Cutout), by(reason::Arrays),
      sums[stat::NonPlayerOther]);
  Log.info(
      "VSM: invalidation run {}: footprint pages {} (median {}, max {} per steady frame), player footprint {} "
      "(median {}, max {}); dirtied entries {} (priority {}); stats hash {:016x}",
      run, sums[stat::FootprintPages], median(footprint), peak(footprint), sums[stat::FootprintPriority],
      median(player), peak(player), sums[stat::DirtiedPages], sums[stat::DirtiedPriority], hash);
}

void log_runs(std::vector<FrameStats> frames) {
  std::ranges::sort(frames, {}, &FrameStats::seq);
  size_t first = 0;
  size_t run = 0;
  for (size_t i = 1; i <= frames.size(); ++i) {
    // A replay repeat restarts hint frames, so each drop opens a new run
    if (i == frames.size() || frames[i].hintFrame <= frames[i - 1].hintFrame) {
      log_run(run++, std::span{frames}.subspan(first, i - first));
      first = i;
    }
  }
}

bool near(uint32_t a, uint32_t b) noexcept {
  const float x = std::bit_cast<float>(a);
  const float y = std::bit_cast<float>(b);
  return a == b || std::abs(x - y) <= 1e-3f + 1e-5f * std::max(std::abs(x), std::abs(y));
}

bool test_bit(std::span<const uint32_t> words, uint32_t index) noexcept {
  return ((words[index >> 5] >> (index & 31)) & 1u) != 0;
}
} // namespace

std::optional<Config> parse(std::string_view text) noexcept {
  Config out{.on = true};
  if (text == "1") {
    return out;
  }
  if (!text.starts_with("1,")) {
    return std::nullopt;
  }
  text.remove_prefix(2);
  std::array<std::string_view, 3> parts{};
  size_t count = 0;
  while (count < parts.size()) {
    const size_t comma = text.find(',');
    parts[count++] = text.substr(0, comma);
    if (comma == std::string_view::npos) {
      text = {};
      break;
    }
    text.remove_prefix(comma + 1);
  }
  if (!text.empty() || count < 2 || !parse_u64(parts[0], out.first) || !parse_u64(parts[1], out.last) ||
      out.first > out.last) {
    return std::nullopt;
  }
  if (count == 3) {
    constexpr std::string_view Full = "full=";
    uint64_t frame = 0;
    if (!parts[2].starts_with(Full) || !parse_u64(parts[2].substr(Full.size()), frame)) {
      return std::nullopt;
    }
    out.full = frame;
  }
  return out;
}

const Config& config() noexcept { return g_override ? *g_override : env_config(); }

Result result() noexcept {
  std::lock_guard lock{g_mutex};
  return g_result;
}

bool check_frame(uint64_t hintFrame) noexcept {
  const Config& c = config();
  return c.on && hintFrame >= c.first && hintFrame <= c.last && (hintFrame - c.first) % CheckEvery == 0;
}

bool force_full(uint64_t hintFrame) noexcept {
  const Config& c = config();
  return c.on && c.full && *c.full == hintFrame;
}

void record_views(const Views& views, uint64_t hintFrame) noexcept {
  if (!config().on) {
    return;
  }
  g_lastViews = views;
  g_lastFrame = hintFrame;
}

void record_mark(const MarkParams& params, uint64_t hintFrame) noexcept {
  if (!config().on) {
    return;
  }
  g_markParams = params;
  g_markFrame = hintFrame;
}

std::string verify(const Views& expected, std::span<const uint32_t> views, std::span<const uint32_t> table,
                   std::span<const uint32_t> phys) {
  if (views.size() * 4 < ViewsBytes || table.size() < MaxVirtualPages || phys.size() < MaxPoolPages * PhysWords) {
    return "the readback is short";
  }
  std::array<uint32_t, ViewsBytes / 4> words{};
  std::memcpy(words.data(), &expected, ViewsBytes);
  for (size_t i = 0; i < words.size(); ++i) {
    if (words[i] != views[i]) {
      return fmt::format("views word {} is {:#x}, the worker wrote {:#x}", i, views[i], words[i]);
    }
  }
  const uint32_t poolPages = expected.header.poolPages;
  const bool full = (expected.header.flags & header_flag::Full) != 0;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    const uint32_t e = table[v];
    if (e == 0) {
      continue;
    }
    if ((expected.views[v / PagesPerView].view_flags() & view_flag::Active) == 0) {
      return fmt::format("entry {} of inactive view {} is {:#x}", v, v / PagesPerView, e);
    }
    if ((e & entry::Allocated) == 0) {
      return fmt::format("entry {} is {:#x} without Allocated", v, e);
    }
    const uint32_t p = e & entry::PhysMask;
    if (p >= poolPages) {
      return fmt::format("entry {} names page {} of a {} page pool", v, p, poolPages);
    }
    if (phys[size_t{p} * PhysWords] != v + 1) {
      return fmt::format("entry {} names page {}, whose owner is {}", v, p, phys[size_t{p} * PhysWords]);
    }
  }
  for (uint32_t p = 0; p < MaxPoolPages; ++p) {
    const uint32_t owner = phys[size_t{p} * PhysWords];
    const uint32_t lastUsed = phys[size_t{p} * PhysWords + 1];
    if (owner == 0) {
      if (full && lastUsed != 0) {
        return fmt::format("free page {} keeps lastUsed {} after a full reset", p, lastUsed);
      }
      continue;
    }
    if (p >= poolPages || owner > MaxVirtualPages) {
      return fmt::format("page {} has owner {} in a {} page pool", p, owner, poolPages);
    }
    if (full && lastUsed != expected.header.clock) {
      return fmt::format("page {} was last used at {} on a full frame at clock {}", p, lastUsed, expected.header.clock);
    }
    const uint32_t e = table[owner - 1];
    if ((e & entry::Allocated) == 0 || (e & entry::PhysMask) != p) {
      return fmt::format("page {} names entry {}, which is {:#x}", p, owner - 1, e);
    }
  }
  return {};
}

std::string verify_scan(const Views& expected, const ScanReadback& in, ScanCounts& counts) {
  if (in.records.size() < size_t{Slots} * RecordWords || in.history.size() < size_t{Slots} * HistoryWords ||
      in.castBits.size() < CastBitsWords || in.dirtyPages.size() < DirtyPagesWords ||
      in.table.size() < MaxVirtualPages || in.stats.size() < StatCount ||
      (!in.prevHistory.empty() && in.prevHistory.size() < size_t{Slots} * HistoryWords)) {
    return "the scan readback is short";
  }
  const bool held = (expected.header.flags & header_flag::Held) != 0;
  for (uint32_t s = 0; s < Slots; ++s) {
    const bool caster = (in.history[size_t{s} * HistoryWords + hist::Flags] & hist::Caster) != 0;
    if (caster != test_bit(in.castBits, s)) {
      return fmt::format("slot {} has cast bit {} but history Caster {}", s, !caster, caster);
    }
    const bool valid = (in.records[size_t{s} * RecordWords] & scene::streamout::instance_flag::Valid) != 0;
    if (caster && !valid && !held) {
      return fmt::format("slot {} has history Caster while its record lacks Valid", s);
    }
  }
  if (in.prevHistory.empty()) {
    return {};
  }
  counts.referenced = true;
  if (held) {
    if (!std::ranges::equal(in.history, in.prevHistory)) {
      return "a held frame rewrote the history";
    }
    if (!std::ranges::all_of(in.dirtyPages, [](uint32_t w) { return w == 0; })) {
      return "a held frame dirtied pages";
    }
    return {};
  }
  std::vector<uint32_t> inner(DirtyPagesWords, 0);
  std::vector<uint32_t> outer(DirtyPagesWords, 0);
  std::array<uint32_t, StatCount> want{};
  for (uint32_t s = 0; s < Slots; ++s) {
    const auto prev = in.prevHistory.subspan(size_t{s} * HistoryWords).first<HistoryWords>();
    const auto rec = in.records.subspan(size_t{s} * RecordWords).first<RecordWords>();
    const auto got = in.history.subspan(size_t{s} * HistoryWords).first<HistoryWords>();
    SlotScan twin = scan_slot(prev, rec, in.materials, expected.header);
    for (uint32_t k = 0; k < HistoryWords; ++k) {
      if (k == hist::CullMask) {
        continue;
      }
      const bool approx = k >= hist::Box && k < hist::Xform + 12;
      if (approx ? !near(got[k], twin.history[k]) : got[k] != twin.history[k]) {
        return fmt::format("slot {} history word {} is {:#x}, the twin has {:#x}", s, k, got[k], twin.history[k]);
      }
    }
    want[stat::ScanValid] += (rec[0] & scene::streamout::instance_flag::Valid) != 0 ? 1 : 0;
    want[stat::ScanCasters] += twin.castNow ? 1 : 0;
    want[stat::Changed] += twin.changed ? 1 : 0;
    want[stat::PendingStream] += twin.pending ? 1 : 0;
    want[stat::PendingCasters] += twin.pendingCaster ? 1 : 0;
    want[stat::ChunkNeed] += twin.chunkNeed ? 1 : 0;
    if (!twin.changed) {
      continue;
    }
    ++counts.changed;
    // The dirty pass reads the box the scan wrote, so the reference takes the GPU's box and checks only the rect
    if ((twin.box[dirty_box::Flags] & dirty_box::HasNew) != 0) {
      std::copy_n(got.begin() + hist::Box, 6, twin.box.begin() + dirty_box::New);
    }
    dirty_footprint(expected, twin.box, inner, -RectSlack);
    dirty_footprint(expected, twin.box, outer, RectSlack);
  }
  for (const uint32_t k : {stat::ScanValid, stat::ScanCasters, stat::Changed, stat::PendingStream, stat::PendingCasters,
                           stat::ChunkNeed}) {
    if (in.stats[k] != want[k]) {
      return fmt::format("stat {} is {}, the twin counts {}", k, in.stats[k], want[k]);
    }
  }
  for (uint32_t w = 0; w < DirtyPagesWords; ++w) {
    const uint32_t gpu = in.dirtyPages[w];
    if ((inner[w] & ~gpu) != 0 || (gpu & ~outer[w]) != 0) {
      return fmt::format("dirty plane {} word {} is {:#x}, the reference spans {:#x} to {:#x}", w / DirtyPlaneWords,
                         w % DirtyPlaneWords, gpu, inner[w], outer[w]);
    }
    counts.footprint += w < DirtyPlaneWords ? static_cast<uint64_t>(std::popcount(gpu)) : 0;
  }
  const std::span<const uint32_t> innerPriority = std::span{inner}.subspan(DirtyPlaneWords);
  std::vector<bool> rendered(MaxVirtualPages, false);
  if (in.renderList.size() >= render_list::Words) {
    const uint32_t count = std::min(in.renderList[render_list::Count], MaxRenderPages);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t v = in.renderList[render_list::Pages + 2 * i];
      if (v < MaxVirtualPages) {
        rendered[v] = true;
      }
    }
  }
  for (uint32_t index = 0; index < MaxVirtualPages; ++index) {
    const uint32_t e = in.table[index];
    if ((e & entry::Allocated) == 0 || rendered[index]) {
      continue;
    }
    if (test_bit(inner, index) && (e & entry::Dirty) == 0) {
      return fmt::format("entry {} lies in a dirty footprint but is {:#x}", index, e);
    }
    if (test_bit(innerPriority, index) && (e & entry::Priority) == 0) {
      return fmt::format("entry {} lies in a player footprint but is {:#x}", index, e);
    }
  }
  return {};
}

std::string verify_alloc(const Views& expected, const AllocReadback& in, AllocSummary& out) {
  if (in.requests.size() < MaxVirtualPages || in.table.size() < MaxVirtualPages ||
      in.phys.size() < size_t{MaxPoolPages} * PhysWords || in.renderList.size() < render_list::Words ||
      in.stats.size() < StatCount) {
    return "the alloc readback is short";
  }
  const ViewsHeader& h = expected.header;
  const std::span<const uint32_t> rl = in.renderList;
  const std::span<const uint32_t> st = in.stats;
  if (in.params != nullptr && !in.receivers.empty()) {
    const MarkReference ref = mark_reference(expected, *in.params, in.receivers);
    out.marked = true;
    out.receivers = ref.receivers;
    out.ambiguous = ref.ambiguous;
    for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
      const uint32_t gpu = in.requests[v] & (request_bit::Pixel | request_bit::Fallback);
      if ((ref.must[v] & ~gpu) != 0) {
        return fmt::format("page {} has requests {:#x} but the mark twin needs {:#x}", v, gpu, ref.must[v]);
      }
      if ((gpu & ~(ref.must[v] | ref.may[v])) != 0) {
        return fmt::format("page {} has requests {:#x}, more than the mark twin's {:#x}", v, gpu,
                           ref.must[v] | ref.may[v]);
      }
    }
  }
  const bool held = (h.flags & header_flag::Held) != 0;
  const AllocBudget budget =
      alloc_budget(st[stat::NeedPriority], st[stat::NeedNew], st[stat::NeedReuse], st[stat::Evictable], h.budget, held);
  const uint32_t count = rl[render_list::Count];
  if (count != budget.count || st[stat::Rendered] != count) {
    return fmt::format("the render list holds {} pages (stat {}), the budget allows {}", count, st[stat::Rendered],
                       budget.count);
  }
  if (st[stat::PlayerDeferred] != 0 || budget.playerDeferred != 0) {
    return fmt::format("{} player pages were deferred", st[stat::PlayerDeferred]);
  }
  if (st[stat::DeferredNew] != budget.deferredNew || st[stat::DeferredReuse] != budget.deferredReuse ||
      st[stat::OverflowPool] != budget.overflowPool) {
    return "the deferred and overflow counts disagree with the budget";
  }
  std::vector<bool> seenPage(MaxVirtualPages, false);
  std::vector<bool> seenPhys(MaxPoolPages, false);
  std::vector<uint32_t> rows(size_t{MaxViews} * WindowPages, 0);
  uint32_t sentinels = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t v = rl[render_list::Pages + 2 * i];
    const uint32_t p = rl[render_list::Pages + 2 * i + 1];
    if (v == NoPage) {
      ++sentinels;
      continue;
    }
    if (v >= MaxVirtualPages || p >= h.poolPages || seenPage[v] || seenPhys[p]) {
      return fmt::format("render entry {} names page {} and physical page {} twice or out of range", i, v, p);
    }
    seenPage[v] = true;
    seenPhys[p] = true;
    const uint32_t e = in.table[v];
    if ((e & entry::Allocated) == 0 || (e & entry::PhysMask) != p || in.phys[size_t{p} * PhysWords] != v + 1 ||
        in.phys[size_t{p} * PhysWords + 1] != h.clock) {
      return fmt::format("render entry {} names page {} on {}, but the entry is {:#x}", i, v, p, e);
    }
    const ViewRecord& view = expected.views[v / PagesPerView];
    const uint32_t slot = v % PagesPerView;
    const uint32_t lx = (slot % WindowPages - static_cast<uint32_t>(view.origin[0])) & (WindowPages - 1);
    const uint32_t ly = (slot / WindowPages - static_cast<uint32_t>(view.origin[1])) & (WindowPages - 1);
    rows[size_t{v / PagesPerView} * WindowPages + ly] |= 1u << lx;
    ++out.renderedLevels[SunLevels - 1 - std::min(v / PagesPerView, SunLevels - 1)];
  }
  if (sentinels != budget.overflowPool) {
    return fmt::format("{} render entries overflowed the pool, the stats say {}", sentinels, budget.overflowPool);
  }
  const uint32_t checks = rl[render_list::CheckCount];
  if (checks != 0 && (h.flags & header_flag::Check) == 0) {
    return fmt::format("{} check entries on a frame without the check flag", checks);
  }
  for (uint32_t i = count; i < std::min(count + checks, MaxRenderPages); ++i) {
    const uint32_t word = rl[render_list::Pages + 2 * i];
    if (word == NoPage) {
      continue;
    }
    const uint32_t v = render_list::virtual_page(word);
    const uint32_t p = rl[render_list::Pages + 2 * i + 1];
    if ((word & render_list::CheckOnly) == 0 || v >= MaxVirtualPages || p >= h.poolPages || seenPage[v] ||
        seenPhys[p]) {
      return fmt::format("check entry {} names page {:#x} and physical page {} twice or out of range", i, word, p);
    }
    seenPage[v] = true;
    seenPhys[p] = true;
    const uint32_t e = in.table[v];
    if ((e & (entry::Allocated | entry::Cached | entry::Dirty)) != (entry::Allocated | entry::Cached) ||
        (e & entry::PhysMask) != p) {
      return fmt::format("check entry {} names page {} on {}, but the entry is {:#x}", i, v, p, e);
    }
    const ViewRecord& view = expected.views[v / PagesPerView];
    const uint32_t slot = v % PagesPerView;
    const uint32_t lx = (slot % WindowPages - static_cast<uint32_t>(view.origin[0])) & (WindowPages - 1);
    const uint32_t ly = (slot / WindowPages - static_cast<uint32_t>(view.origin[1])) & (WindowPages - 1);
    rows[size_t{v / PagesPerView} * WindowPages + ly] |= 1u << lx;
  }
  for (uint32_t w = 0; w < rows.size(); ++w) {
    if (rl[render_list::RowMasks + w] != rows[w]) {
      return fmt::format("row mask {} is {:#x}, the render list gives {:#x}", w, rl[render_list::RowMasks + w],
                         rows[w]);
    }
  }
  const uint32_t valid = count - sentinels;
  out.committed = st[stat::Committed] + st[stat::CommitSkipped] == valid;
  uint32_t unallocated = 0;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    const uint32_t e = in.table[v];
    const uint32_t level = SunLevels - 1 - std::min(v / PagesPerView, SunLevels - 1);
    if (out.committed && (e & (entry::RenderNow | entry::Check)) != 0) {
      return fmt::format("entry {} is {:#x} after commit", v, e);
    }
    if ((e & (entry::Allocated | entry::Cached | entry::RenderNow)) == (entry::Allocated | entry::Cached)) {
      ++out.cached;
      ++out.cachedLevels[level];
    }
    if ((in.requests[v] & request_bit::Mask) == 0) {
      continue;
    }
    ++out.requested;
    ++out.requestedLevels[level];
    unallocated += (e & entry::Allocated) == 0 ? 1 : 0;
  }
  if (out.requested != st[stat::Requested]) {
    return fmt::format("{} pages carry requests, the classify counted {}", out.requested, st[stat::Requested]);
  }
  if (!held && unallocated != budget.deferredNew + budget.overflowPool) {
    return fmt::format("{} requested pages hold no physical page, but {} were deferred and {} overflowed", unallocated,
                       budget.deferredNew, budget.overflowPool);
  }
  out.rendered = valid;
  uint64_t hash = 0xCBF29CE484222325ull;
  for (const uint32_t w : in.table) {
    hash = fnv(hash, w);
  }
  for (const uint32_t w : in.phys) {
    hash = fnv(hash, w);
  }
  out.hash = hash;
  return {};
}

std::string verify_cull(const Views& expected, const CullReadback& in, CullSummary& out) {
  if (in.history.size() < size_t{Slots} * HistoryWords || in.renderList.size() < render_list::Words ||
      in.pairs.size() < pairs_header::PairsWords || in.pageCounts.size() < PageCountWords) {
    return "the cull readback is short";
  }
  out.checked = true;
  const std::span<const uint32_t> hdr = in.pairs.first(pairs_header::Words);
  const std::span<const uint32_t> counts = in.pageCounts;
  const uint32_t maxPairs = std::min(expected.header.maxPairs, MaxPairs);
  const uint32_t first = skip_first(counts, maxPairs);
  const uint32_t key = hdr[pairs_header::SkipKey];
  const uint32_t gpuFirst = key != 0 ? MaxRenderPages - std::min(key, MaxRenderPages) : MaxRenderPages;
  if (gpuFirst != first) {
    return fmt::format("the first skipped page is {}, the counts give {}", gpuFirst, first);
  }
  uint32_t opaque = 0;
  uint32_t cutout = 0;
  for (uint32_t r = 0; r < MaxRenderPages; ++r) {
    const uint32_t n = counts[r] + counts[MaxRenderPages + r];
    if (r >= first) {
      out.skipped += n != 0 ? 1 : 0;
      continue;
    }
    opaque += counts[r];
    cutout += counts[MaxRenderPages + r];
  }
  if (hdr[pairs_header::Skipped] != out.skipped) {
    return fmt::format("{} pages skipped, the counts give {}", hdr[pairs_header::Skipped], out.skipped);
  }
  if (hdr[pairs_header::Dropped] != 0) {
    return fmt::format("the emit dropped {} pairs its count had", hdr[pairs_header::Dropped]);
  }
  if (hdr[pairs_header::OpaqueCount] != opaque || hdr[pairs_header::CutoutBase] != opaque ||
      hdr[pairs_header::CutoutCount] != cutout) {
    return fmt::format("the pair header says {} opaque from 0 and {} cutout from {}, the counts give {} and {}",
                       hdr[pairs_header::OpaqueCount], hdr[pairs_header::CutoutCount], hdr[pairs_header::CutoutBase],
                       opaque, cutout);
  }
  std::vector<uint64_t> gpu;
  gpu.reserve(size_t{opaque} + cutout);
  std::vector<uint32_t> perPage(MaxRenderPages, 0);
  for (uint32_t i = 0; i < opaque + cutout; ++i) {
    const std::array<uint32_t, 2> w{in.pairs[pairs_header::Words + 2 * i], in.pairs[pairs_header::Words + 2 * i + 1]};
    const Pair pair = pair_unpack(w);
    const bool wantCutout = i >= opaque;
    const uint32_t flags = in.history[size_t{pair.slot} * HistoryWords + hist::Flags];
    if (pair.renderIndex >= first || ((flags & hist::Cutout) != 0) != wantCutout) {
      return fmt::format("pair {} names slot {} on page {} in the {} list", i, pair.slot, pair.renderIndex,
                         wantCutout ? "cutout" : "opaque");
    }
    ++perPage[pair.renderIndex];
    gpu.push_back(pair_key(pair.renderIndex, w[0], pair.slot));
    out.cutout += wantCutout ? 1 : 0;
    out.chunk += pair.chunk ? 1 : 0;
    out.checkPairs += pair.renderIndex >= in.renderList[render_list::Count] ? 1 : 0;
  }
  for (uint32_t r = 0; r < first; ++r) {
    if (perPage[r] != counts[r] + counts[MaxRenderPages + r]) {
      return fmt::format("page {} holds {} pairs, its count is {}", r, perPage[r],
                         counts[r] + counts[MaxRenderPages + r]);
    }
    out.pages += perPage[r] != 0 ? 1 : 0;
    out.maxPerPage = std::max(out.maxPerPage, perPage[r]);
  }
  out.gpuPairs = gpu.size();
  out.posed = hdr[pairs_header::PosedPairs];
  if (hdr[pairs_header::ChunkPairs] != out.chunk) {
    return fmt::format("the header counts {} chunk pairs, the list holds {}", hdr[pairs_header::ChunkPairs], out.chunk);
  }
  std::ranges::sort(gpu);
  if (std::ranges::adjacent_find(gpu) != gpu.end()) {
    return "a pair appears twice";
  }
  constexpr float Margin = 2.f;
  const CullReference ref = cull_reference(expected, in.history, in.clusters, in.chunkBoxes, in.renderList, Margin);
  out.exactPairs = ref.exact.size();
  out.loosePairs = ref.loose.size();
  for (const uint64_t k : gpu) {
    if (!std::ranges::binary_search(ref.loose, k)) {
      return fmt::format("page {} pairs item {:#x} of slot {}, which lies more than {} texels outside it", k >> 46,
                         (k >> 14) & 0xFFFFFFFFu, k & 0x3FFFu, Margin);
    }
  }
  for (const uint64_t k : ref.exact) {
    if ((k >> 46) < first && !std::ranges::binary_search(gpu, k)) {
      return fmt::format("page {} misses item {:#x} of slot {}", k >> 46, (k >> 14) & 0xFFFFFFFFu, k & 0x3FFFu);
    }
  }
  return {};
}

RasterSummary raster_summary(std::span<const std::array<uint32_t, StatCount>> frames,
                             std::span<const uint64_t> hintFrames) noexcept {
  RasterSummary out;
  for (size_t i = 0; i < frames.size(); ++i) {
    const std::array<uint32_t, StatCount>& c = frames[i];
    if (c[stat::CheckListed] == 0 && c[stat::CheckCompared] == 0) {
      continue;
    }
    ++out.frames;
    out.pages += c[stat::CheckCompared];
    out.mismatched += c[stat::CheckMismatched];
    out.texels += c[stat::CheckTexels];
    out.skipped += c[stat::CheckSkipped];
    const uint64_t frame = i < hintFrames.size() ? hintFrames[i] : i;
    if (c[stat::CheckMismatched] != 0 && (out.firstFrame == UINT64_MAX || frame < out.firstFrame)) {
      out.firstFrame = frame;
      out.firstKey = ~c[stat::CheckFirst];
    }
  }
  return out;
}

std::string raster_mismatch(uint32_t key) {
  const uint32_t slot = (key >> 12) & (PagesPerView - 1);
  return fmt::format("level {} slot ({}, {}) physical page {}", key >> 22, slot % WindowPages, slot / WindowPages,
                     key & entry::PhysMask);
}

namespace {
struct CopyLocal {
  uint16_t pool = NoRef;
  uint16_t copy = NoRef;
};
static_assert(sizeof(CopyLocal) <= detail::PassLocalSize);

struct CompareLocal {
  uint16_t pool = NoRef;
  uint16_t copy = NoRef;
  uint16_t renderList = NoRef;
  uint16_t table = NoRef;
  uint16_t stats = NoRef;
  uint16_t views = NoRef;
  uint16_t probes = NoRef;
};

constexpr uint64_t ProbesBytes = uint64_t{MaxProbes + 1} * 16;
static_assert(sizeof(CompareLocal) <= detail::PassLocalSize);

bool raster_declarable(const detail::PassSetup& setup) noexcept {
  return config().on && active() && setup.env != nullptr && setup.env->retained && handle(Table).id != 0 &&
         handle(Pool).id != 0;
}

bool chain_pass_enabled(const detail::StageData& stage, const void*) {
  return config().on && webgpu::g_backendType != wgpu::BackendType::Null && chain_enabled(stage);
}

void declare_copy(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!raster_declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  CopyLocal local;
  local.pool = builder.import_persistent_texture(Pool, handle(Pool), pool_desc(), Access::CopySrc).index;
  local.copy = builder.create(CopyName, pool_desc(), Access::CopyDst).index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

void encode_copy(detail::PassContext& ctx, const void*) {
  CopyLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const wgpu::Texture pool = ctx.texture(TexRef{local.pool});
  const wgpu::Texture copy = ctx.texture(TexRef{local.copy});
  if (!pool || !copy || pool.GetWidth() != copy.GetWidth() || pool.GetHeight() != copy.GetHeight() ||
      (encoded_mask() & chain_bit::Assign) == 0) {
    return;
  }
  const wgpu::TexelCopyTextureInfo source{.texture = pool};
  const wgpu::TexelCopyTextureInfo destination{.texture = copy};
  const wgpu::Extent3D extent{pool.GetWidth(), pool.GetHeight(), 1};
  ctx.encoder().CopyTextureToTexture(&source, &destination, &extent);
  mark_encoded(chain_bit::CheckCopy);
}

void declare_compare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!raster_declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  CompareLocal local;
  local.pool = builder.import_persistent_texture(Pool, handle(Pool), pool_desc(), Access::Sampled).index;
  local.copy = builder.use(CopyName, Access::Sampled).index;
  local.renderList = builder.use_buffer(RenderList, Access::StorageRead).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::StorageRead).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::StorageRW).index;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::StorageRead).index;
  local.probes = builder
                     .create_buffer(ProbesName, {.size = ProbesBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                                    Access::StorageWrite)
                     .index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

void compare_pipelines(const detail::PipelineSetup& setup, const void*,
                       std::span<uint32_t, detail::MaxPassPipelines> ids) {
  for (uint32_t i = 0; i < 2; ++i) {
    const shaders::PipelineId id = shaders::request_compute({
        .label = ComparePass,
        .shader = CheckShader,
        .entryPoint = i == 0 ? "vsm_check_cmp" : "vsm_check_probe",
        .layout = setup.layout,
        .layoutKey = setup.layoutHash,
    });
    ids[i] = id.value;
    register_chain_pipeline(id);
  }
}

void encode_compare(detail::PassContext& ctx, const void*) {
  CompareLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const std::array pipelines{shaders::compute_pipeline(ctx.pipeline(0)), shaders::compute_pipeline(ctx.pipeline(1))};
  const BufferView probes = ctx.buffer(BufRef{local.probes});
  constexpr uint32_t Need = chain_bit::CheckCopy | chain_bit::Raster;
  if (!pipelines[0] || !pipelines[1] || !probes.buffer || (encoded_mask() & Need) != Need) {
    return;
  }
  std::array<std::array<float, 4>, MaxProbes + 1> points{};
  {
    std::lock_guard lock{g_mutex};
    const ViewsHeader& h = g_lastViews.header;
    const size_t count = std::min<size_t>(g_probes.size(), MaxProbes);
    points[0][0] = static_cast<float>(count);
    for (size_t i = 0; i < count; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        const double anchor = double{h.anchorHi[k]} + double{h.anchorLo[k]};
        points[i + 1][k] = static_cast<float>(g_probes[i][k] - anchor);
      }
    }
  }
  ctx.encoder().WriteBuffer(probes.buffer, probes.offset, reinterpret_cast<const uint8_t*>(points.data()),
                            std::min(probes.size, ProbesBytes));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipelines[0]);
  pass.DispatchWorkgroups(MaxRenderPages);
  pass.SetPipeline(pipelines[1]);
  pass.DispatchWorkgroups(1);
  pass.End();
  mark_encoded(chain_bit::CheckCompare);
}
} // namespace

detail::PassDesc copy_pass_desc() {
  return {
      .name = std::string{CopyPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .cls = detail::PassClass::Required,
      .order = 38,
      .declare = declare_copy,
      .enabled = chain_pass_enabled,
      .encode = encode_copy,
  };
}

detail::PassDesc compare_pass_desc() {
  return {
      .name = std::string{ComparePass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 57,
      .declare = declare_compare,
      .enabled = chain_pass_enabled,
      .encode = encode_compare,
      .pipelines = compare_pipelines,
  };
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{CheckPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .cls = detail::PassClass::Required,
      .order = 95,
      .paramsSize = sizeof(CheckParams),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .stageParams = stage_params,
  };
}

void shutdown() noexcept {
  std::unique_lock lock{g_mutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{ShutdownWaitMs};
  const auto pending = [] {
    return g_result.busy || std::ranges::any_of(g_ring, [](const RingSlot& slot) { return slot.busy; });
  };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  Result r = g_result;
  std::vector<FrameStats> frames = std::move(g_frames);
  g_frames.clear();
  g_buffer = {};
  g_ring = {};
  lock.unlock();
  std::vector<std::array<uint32_t, StatCount>> counters;
  std::vector<uint64_t> hintFrames;
  for (const FrameStats& f : frames) {
    counters.push_back(f.counters);
    hintFrames.push_back(f.hintFrame);
  }
  const RasterSummary raster = raster_summary(counters, hintFrames);
  if (!frames.empty()) {
    log_runs(std::move(frames));
  }
  r.rasterFrames = raster.frames;
  r.rasterPages = raster.pages;
  r.rasterMismatched = raster.mismatched;
  r.rasterTexels = raster.texels;
  r.rasterSkipped = raster.skipped;
  if (raster.frames != 0) {
    Log.info(
        "VSM: raster check on {} frames: {} cached pages re-rendered in place and compared, {} mismatched ({} "
        "texels), {} skipped past the pair cap",
        raster.frames, raster.pages, raster.mismatched, raster.texels, raster.skipped);
  }
  if (raster.mismatched != 0) {
    ++r.failures;
    Log.error("VSM: check frame {} FAILED: a cached page differs from its in-place re-render at {}", raster.firstFrame,
              raster_mismatch(raster.firstKey));
  }
  {
    std::lock_guard relock{g_mutex};
    g_result.rasterFrames = r.rasterFrames;
    g_result.rasterPages = r.rasterPages;
    g_result.rasterMismatched = r.rasterMismatched;
    g_result.rasterTexels = r.rasterTexels;
    g_result.rasterSkipped = r.rasterSkipped;
    g_result.failures = r.failures;
  }
  if (!config().on && r.readbacks == 0) {
    return;
  }
  if (r.failures != 0 || r.busy) {
    Log.error("VSM: check FAILED ({} failures, {} readbacks over {} check frames{})", r.failures, r.readbacks, r.frames,
              r.busy ? ", a readback never finished" : "");
  } else if (r.readbacks == 0 && r.frames != 0) {
    Log.error("VSM: check FAILED (no readback over {} check frames)", r.frames);
  } else if (r.readbacks == 0) {
    Log.warn("VSM: check ran no readback (no check frame ran the shadow chain)");
  } else {
    Log.info(
        "VSM: check passed ({} readbacks, {} allocated entries, {} full frames, {} skipped; dirty reference on {} "
        "frames, {} changed slots, {} footprint pages; mark twin on {} frames, {} receivers, {} near a boundary; "
        "{} requested and {} rendered pages, {} frames uncommitted; cull reference on {} frames, {} pairs, {} "
        "exact, {} skipped pages; raster check on {} frames, {} pages compared, {} mismatched; {} stats frames "
        "skipped)",
        r.readbacks, r.entries, r.fulls, r.skipped, r.referenced, r.changed, r.footprint, r.marked, r.receivers,
        r.ambiguous, r.requested, r.rendered, r.uncommitted, r.culled, r.pairs, r.exactPairs, r.skippedPages,
        r.rasterFrames, r.rasterPages, r.rasterMismatched, r.statsSkipped);
  }
}

namespace testing {
void reset() noexcept {
  std::lock_guard lock{g_mutex};
  g_override.reset();
  g_result = {};
  g_buffer = {};
  g_lastViews = {};
  g_lastFrame = NoFrame;
  g_prevFrame = NoFrame;
  g_warnedViews = false;
  g_ring = {};
  g_frames.clear();
  g_seq = 0;
  g_markParams = {};
  g_markFrame = NoFrame;
  g_probes.clear();
}

void set_config(const std::optional<Config>& config) noexcept { g_override = config; }

void set_probes(std::span<const std::array<double, 3>> worldPoints) noexcept {
  std::lock_guard lock{g_mutex};
  g_probes.assign(worldPoints.begin(), worldPoints.begin() + std::min<size_t>(worldPoints.size(), MaxProbes));
}

std::vector<FrameCounters> frame_stats() {
  std::unique_lock lock{g_mutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{ShutdownWaitMs};
  while (std::ranges::any_of(g_ring, [](const RingSlot& slot) { return slot.busy; }) &&
         std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  std::vector<FrameStats> frames = g_frames;
  lock.unlock();
  std::ranges::sort(frames, {}, &FrameStats::seq);
  std::vector<FrameCounters> out;
  out.reserve(frames.size());
  for (const FrameStats& f : frames) {
    out.push_back({.hintFrame = f.hintFrame, .counters = f.counters});
  }
  return out;
}

Result result() noexcept { return check::result(); }
} // namespace testing
} // namespace aurora::render::shadows::vsm::check

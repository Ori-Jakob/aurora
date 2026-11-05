#include "vsm_check.hpp"

#include "vsm_invalidate.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../scene/atlas.hpp"
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
constexpr uint64_t ReadbackBytes = RenderListOffset + render_list::Bytes;

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
  RefCount
};

struct CheckLocal {
  std::array<uint16_t, RefCount> refs{NoRef, NoRef, NoRef, NoRef, NoRef, NoRef,
                                      NoRef, NoRef, NoRef, NoRef, NoRef, NoRef};
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
  std::lock_guard lock{g_mutex};
  ++g_result.readbacks;
  g_result.entries += allocated;
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
      scene[0].id == 0 || scene[1].id == 0) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.side_effect();
  CheckLocal local;
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
    if (out.committed && (e & entry::RenderNow) != 0) {
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
  const Result r = g_result;
  std::vector<FrameStats> frames = std::move(g_frames);
  g_frames.clear();
  g_buffer = {};
  g_ring = {};
  lock.unlock();
  if (!frames.empty()) {
    log_runs(std::move(frames));
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
        "{} requested and {} rendered pages, {} frames uncommitted; {} stats frames skipped)",
        r.readbacks, r.entries, r.fulls, r.skipped, r.referenced, r.changed, r.footprint, r.marked, r.receivers,
        r.ambiguous, r.requested, r.rendered, r.uncommitted, r.statsSkipped);
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
}

void set_config(const std::optional<Config>& config) noexcept { g_override = config; }

Result result() noexcept {
  std::lock_guard lock{g_mutex};
  return g_result;
}
} // namespace testing
} // namespace aurora::render::shadows::vsm::check

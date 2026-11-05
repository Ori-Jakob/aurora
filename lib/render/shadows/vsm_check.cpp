#include "vsm_check.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>

#include <charconv>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace aurora::render::shadows::vsm::check {
namespace {
constexpr Module Log{"aurora::render::shadows"};
constexpr uint32_t ShutdownWaitMs = 2000;
constexpr uint32_t NoRef = UINT16_MAX;
constexpr uint64_t ViewsOffset = 0;
constexpr uint64_t TableOffset = ViewsOffset + ViewsBytes;
constexpr uint64_t PhysOffset = TableOffset + TableBytes;
constexpr uint64_t StatsOffset = PhysOffset + PhysBytes;
constexpr uint64_t ReadbackBytes = StatsOffset + StatsBytes;

struct CheckLocal {
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t phys = NoRef;
  uint16_t stats = NoRef;
};

struct CheckParams {
  uint64_t hintFrame = 0;
  uint32_t set = 0;
  uint32_t reserved = 0;
};

std::optional<Config> g_override;
std::mutex g_mutex;
Result g_result;
wgpu::Buffer g_buffer;
Views g_lastViews;
uint64_t g_lastFrame = UINT64_MAX;
bool g_warnedViews = false;

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

uint32_t allocated_entries(std::span<const uint32_t> table) noexcept {
  uint32_t count = 0;
  for (const uint32_t e : table) {
    count += (e & entry::Allocated) != 0 ? 1 : 0;
  }
  return count;
}

void finish_readback(const wgpu::Buffer& buffer, const Views& expected, uint64_t hintFrame) {
  const auto* bytes = static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, ReadbackBytes));
  std::string error = "the readback could not be mapped";
  uint32_t allocated = 0;
  if (bytes != nullptr) {
    std::vector<uint32_t> words(ReadbackBytes / 4);
    std::memcpy(words.data(), bytes, ReadbackBytes);
    const std::span<const uint32_t> all{words};
    const std::span<const uint32_t> table = all.subspan(TableOffset / 4, TableBytes / 4);
    error = verify(expected, all.subspan(ViewsOffset / 4, ViewsBytes / 4), table,
                   all.subspan(PhysOffset / 4, PhysBytes / 4));
    allocated = allocated_entries(table);
  }
  std::lock_guard lock{g_mutex};
  ++g_result.readbacks;
  g_result.entries += allocated;
  g_result.fulls += (expected.header.flags & header_flag::Full) != 0 ? 1 : 0;
  if (!error.empty()) {
    ++g_result.failures;
    Log.error("VSM: check frame {} FAILED: {}", hintFrame, error);
  }
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!config().on || !active() || setup.env == nullptr || !setup.env->retained || handle(Table).id == 0) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.side_effect();
  CheckLocal local;
  local.views = builder.import_persistent(ViewsName, handle(ViewsName), Access::CopySrc).index;
  local.table = builder.import_persistent(Table, handle(Table), Access::CopySrc).index;
  local.phys = builder.import_persistent(Phys, handle(Phys), Access::CopySrc).index;
  local.stats = builder.import_persistent(StatsName, handle(StatsName), Access::CopySrc).index;
  const std::span<std::byte, detail::PassLocalSize> out = builder.local();
  std::memcpy(out.data(), &local, sizeof(local));
}

bool enabled(const detail::StageData& stage, const void*) {
  return config().on && webgpu::g_backendType != wgpu::BackendType::Null && stage.sceneHints != nullptr &&
         check_frame(stage.sceneHints->hintFrame) && chain_enabled(stage);
}

void stage_params(const detail::StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(CheckParams)) {
    return;
  }
  const CheckParams params{.hintFrame = stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0, .set = 1};
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode(detail::PassContext& ctx, const void*) {
  CheckParams params;
  std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
  CheckLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const BufferView views = ctx.buffer(BufRef{local.views});
  const BufferView table = ctx.buffer(BufRef{local.table});
  const BufferView phys = ctx.buffer(BufRef{local.phys});
  const BufferView stats = ctx.buffer(BufRef{local.stats});
  if (params.set == 0 || !views.buffer || !table.buffer || !phys.buffer || !stats.buffer) {
    return;
  }
  std::shared_ptr<const Views> expected;
  if (g_lastFrame == params.hintFrame) {
    expected = std::make_shared<const Views>(g_lastViews);
  }
  wgpu::Buffer buffer;
  {
    std::lock_guard lock{g_mutex};
    ++g_result.frames;
    if (expected == nullptr) {
      ++g_result.failures;
      if (!std::exchange(g_warnedViews, true)) {
        Log.error("VSM: check frame {} FAILED: vsm.update wrote no views this frame", params.hintFrame);
      }
      return;
    }
    if (g_result.busy) {
      ++g_result.skipped;
      return;
    }
    if (!g_buffer) {
      const wgpu::BufferDescriptor descriptor{
          .label = "VSM check readback",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = ReadbackBytes,
      };
      g_buffer = webgpu::g_device.CreateBuffer(&descriptor);
    }
    g_result.busy = true;
    buffer = g_buffer;
  }
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  encoder.CopyBufferToBuffer(views.buffer, views.offset, buffer, ViewsOffset, ViewsBytes);
  encoder.CopyBufferToBuffer(table.buffer, table.offset, buffer, TableOffset, TableBytes);
  encoder.CopyBufferToBuffer(phys.buffer, phys.offset, buffer, PhysOffset, PhysBytes);
  encoder.CopyBufferToBuffer(stats.buffer, stats.offset, buffer, StatsOffset, StatsBytes);
  const uint64_t hintFrame = params.hintFrame;
  detail::queue_after_frame([buffer, expected, hintFrame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, ReadbackBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, expected, hintFrame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      if (status == wgpu::MapAsyncStatus::Success) {
                        finish_readback(buffer, *expected, hintFrame);
                        buffer.Unmap();
                      }
                      std::lock_guard lock{g_mutex};
                      g_result.busy = false;
                    });
  });
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
    if (full) {
      return fmt::format("entry {} is {:#x} after a full reset", v, e);
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
    if (full || p >= poolPages || owner > MaxVirtualPages) {
      return fmt::format("page {} has owner {} in a {} page pool{}", p, owner, poolPages, full ? " after a reset" : "");
    }
    const uint32_t e = table[owner - 1];
    if ((e & entry::Allocated) == 0 || (e & entry::PhysMask) != p) {
      return fmt::format("page {} names entry {}, which is {:#x}", p, owner - 1, e);
    }
  }
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
  while (g_result.busy && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  const Result r = g_result;
  g_buffer = {};
  lock.unlock();
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
    Log.info("VSM: check passed ({} readbacks, {} allocated entries, {} full frames, {} skipped)", r.readbacks,
             r.entries, r.fulls, r.skipped);
  }
}

namespace testing {
void reset() noexcept {
  std::lock_guard lock{g_mutex};
  g_override.reset();
  g_result = {};
  g_buffer = {};
  g_lastViews = {};
  g_lastFrame = UINT64_MAX;
  g_warnedViews = false;
}

void set_config(const std::optional<Config>& config) noexcept { g_override = config; }

Result result() noexcept {
  std::lock_guard lock{g_mutex};
  return g_result;
}
} // namespace testing
} // namespace aurora::render::shadows::vsm::check

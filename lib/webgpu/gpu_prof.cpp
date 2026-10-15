#include "gpu_prof.hpp"

#include "../internal.hpp"
#include "gpu.hpp"

#include <tracy/Tracy.hpp>

#ifdef TRACY_ENABLE
#include <tracy/TracyC.h>

#include <bitset>
#include <magic_enum.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>

namespace aurora::webgpu::gpu_prof {
namespace {
Module Log("aurora::webgpu::gpu_prof");

using detail::Event;
using detail::EventKind;

// Each zone consumes a begin/end timestamp pair.
// The last pair is reserved for the frame zone.
constexpr uint32_t MaxZones = 511;
constexpr uint32_t QueryCount = MaxZones * 2 + 2;
constexpr uint32_t FrameBeginQuery = MaxZones * 2;
constexpr uint32_t FrameEndQuery = MaxZones * 2 + 1;
constexpr uint64_t ReadbackSize = QueryCount * sizeof(uint64_t);
constexpr size_t RingDepth = 4;
#ifdef TRACY_ENABLE
constexpr bool TracyBuild = true;
constexpr uint8_t ContextId = 0;
constexpr uint64_t SaneFrameGapNs = UINT64_C(5'000'000'000);
#else
constexpr bool TracyBuild = false;
#endif
constexpr auto IdlePollInterval = std::chrono::microseconds{100};

enum class SlotState : uint8_t {
  Free,
  Recording,
  InFlight,
  Mapped,
  Failed,
};

struct Slot {
  wgpu::Buffer readback;
  std::vector<Event> events;
  uint32_t passCount = 0;
  uint64_t sequence = 0;
  uint32_t dropped = 0;
  int64_t submitNs = 0;
  std::atomic<SlotState> state{SlotState::Free};
};

struct TimestampBounds {
  uint64_t begin = 0;
  uint64_t end = 0;

  bool valid() const { return begin != 0 && end > begin; }
};

bool g_supported = false;
bool g_resourcesCreated = false;
bool g_encoderTimestamps = false;
std::atomic_bool g_always{TracyBuild};
std::atomic_uint64_t g_requestedUntil{0};
std::atomic_uint64_t g_sequence{0};
std::atomic_uint64_t g_resolveCount{0};
wgpu::QuerySet g_querySet;
wgpu::Buffer g_resolveBuffer;
std::array<Slot, RingDepth> g_slots;
size_t g_recordSlot = 0;
size_t g_emitSlot = 0;
bool g_frameActive = false;
bool g_framePending = false;
uint32_t g_zoneCount = 0;

std::mutex g_latestMutex;
FrameTimes g_latest;
// Worker only, or the wait_idle caller while the worker is idle
FrameTimes g_scratch;

#ifdef TRACY_ENABLE
bool g_contextEmitted = false;
uint16_t g_queryId = 0;
uint64_t g_lastEmittedTs = 0;
uint64_t g_lastFrameEnd = 0;
#endif

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Tracy and the published timings keep references to zone names; intern them for static lifetime.
const char* intern_name(std::string_view name) {
  static absl::flat_hash_map<std::string, const char*> names;
  const auto it = names.find(name);
  if (it != names.end()) {
    return it->second;
  }
  char* stable = new char[name.size() + 1];
  std::memcpy(stable, name.data(), name.size());
  stable[name.size()] = '\0';
  names.emplace(name, stable);
  return stable;
}

#ifdef TRACY_ENABLE
// tracy::GpuContextType not exposed through TracyC.h
uint8_t tracy_context_type(wgpu::BackendType backend) {
  switch (backend) {
  case wgpu::BackendType::OpenGL:
  case wgpu::BackendType::OpenGLES:
    return 1; // OpenGl
  case wgpu::BackendType::Vulkan:
    return 2; // Vulkan
  case wgpu::BackendType::D3D12:
    return 4; // Direct3D12
  case wgpu::BackendType::D3D11:
    return 5; // Direct3D11
  case wgpu::BackendType::Metal:
    return 6; // Metal
  default:
    return 7; // Custom
  }
}

void emit_context(const Slot& slot, uint64_t frameBegin) {
  // Tracy has no notion of WebGPU's opaque timestamp epoch, so the context
  // anchor pairs a GPU timestamp with "now" at emission. Shift the timestamp
  // by the time elapsed since this frame was submitted, so the frame zone
  // lands at the submit point on the timeline instead of trailing it by the
  // readback latency. Residual error is the GPU's submit-to-execute delay.
  const int64_t anchor = static_cast<int64_t>(frameBegin) + (now_ns() - slot.submitNs);
  ___tracy_emit_gpu_new_context_serial({
      .gpuTime = anchor,
      .period = 1.0f,
      .context = ContextId,
      .flags = 0,
      .type = tracy_context_type(g_backendType),
  });
  const std::string name =
      fmt::format("{} ({})", std::string_view{g_adapterInfo.device}, magic_enum::enum_name(g_backendType));
  ___tracy_emit_gpu_context_name_serial({
      .context = ContextId,
      .name = name.c_str(),
      .len = static_cast<uint16_t>(std::min<size_t>(name.size(), UINT16_MAX)),
  });
}

void emit_zone_begin(const char* name, uint64_t gpuNs) {
  const uint64_t srcloc = ___tracy_alloc_srcloc_name(0, "aurora", 6, "gpu_prof", 8, name, std::strlen(name), 0);
  const uint16_t queryId = g_queryId++;
  ___tracy_emit_gpu_zone_begin_alloc_serial({.srcloc = srcloc, .queryId = queryId, .context = ContextId});
  ___tracy_emit_gpu_time_serial({.gpuTime = int64_t(gpuNs), .queryId = queryId, .context = ContextId});
}

void emit_zone_end(uint64_t gpuNs) {
  const uint16_t queryId = g_queryId++;
  ___tracy_emit_gpu_zone_end_serial({.queryId = queryId, .context = ContextId});
  ___tracy_emit_gpu_time_serial({.gpuTime = int64_t(gpuNs), .queryId = queryId, .context = ContextId});
}
#endif

TimestampBounds event_bounds(const Slot& slot, const uint64_t* ts) {
  TimestampBounds bounds;
  for (const auto& event : slot.events) {
    const uint64_t t = ts[event.query];
    if (t == 0) {
      continue;
    }
    if (bounds.begin == 0 || t < bounds.begin) {
      bounds.begin = t;
    }
    if (t > bounds.end) {
      bounds.end = t;
    }
  }
  return bounds;
}

const uint64_t* mapped_timestamps(const Slot& slot) {
  return static_cast<const uint64_t*>(slot.readback.GetConstMappedRange(0, ReadbackSize));
}

TimestampBounds frame_bounds(const Slot& slot, const uint64_t* ts) {
  // Prefer recorded work bounds. Some backends expose encoder timestamps but
  // report startup/epoch values that are not stable enough to anchor Tracy's
  // GPU context, while pass timestamps are already what the emitted zones use.
  TimestampBounds bounds = event_bounds(slot, ts);
  if (!bounds.valid() && g_encoderTimestamps) {
    bounds = {ts[FrameBeginQuery], ts[FrameEndQuery]};
  }
  return bounds;
}

void publish(const Slot& slot) {
  const uint64_t* ts = mapped_timestamps(slot);
  if (ts == nullptr) {
    return;
  }
  const TimestampBounds bounds = frame_bounds(slot, ts);
  detail::aggregate(slot.events, {ts, QueryCount}, bounds.begin, bounds.end, g_scratch);
  g_scratch.sequence = slot.sequence;
  g_scratch.dropped = slot.dropped;
  std::lock_guard lock{g_latestMutex};
  std::swap(g_latest, g_scratch);
}

#ifdef TRACY_ENABLE
void emit_frame(Slot& slot) {
  const uint64_t* ts = mapped_timestamps(slot);
  if (ts == nullptr) {
    return;
  }

  const TimestampBounds bounds = frame_bounds(slot, ts);
  uint64_t frameBegin = bounds.begin;
  uint64_t frameEnd = bounds.end;
  if (frameBegin == 0 || frameEnd <= frameBegin) {
    return;
  }

  const uint64_t lastFrameEnd = std::exchange(g_lastFrameEnd, frameEnd);
  if (lastFrameEnd != 0) {
    if (frameBegin < lastFrameEnd) {
      if (frameEnd <= lastFrameEnd || frameEnd - lastFrameEnd >= SaneFrameGapNs) {
        return;
      }
      // Some Dawn/Metal timestamp query begins can remain pinned to an old
      // value while end timestamps advance normally. Use consecutive frame
      // ends to keep Tracy's timeline monotonic in that case.
      frameBegin = lastFrameEnd;
    } else if (frameBegin - lastFrameEnd >= SaneFrameGapNs) {
      return;
    }
  }

  if (!TracyIsConnected) {
    return;
  }

  if (!g_contextEmitted) {
    if (lastFrameEnd == 0) {
      return;
    }
    emit_context(slot, frameBegin);
    g_contextEmitted = true;
  }

  // Tracy requires GPU zones within a context to be properly nested in
  // time, and treats large backward jumps as timer wraparound. The recorded
  // events mirror encode order, which matches GPU execution order. Clamp
  // to keep the stream monotonic.
  uint64_t prev = std::max(g_lastEmittedTs, frameBegin);
  const uint64_t endBound = std::max(frameEnd, prev);
  const auto clamped = [&prev, endBound](uint64_t t) {
    prev = std::min(std::max(t, prev), endBound);
    return prev;
  };

  emit_zone_begin(intern_name("Frame"), clamped(frameBegin));
  uint32_t depth = 0;
  uint64_t topLevelEnd = prev;
  uint64_t idleNs = 0;
  // Zones whose timestamps were never written resolve to 0 (e.g. Metal
  // cannot sample encoder-level timestamps); drop those to keep the stream
  // balanced.
  std::bitset<MaxZones> dropped;
  for (const auto& event : slot.events) {
    if (event.kind == EventKind::End) {
      if (dropped[event.query / 2]) {
        continue;
      }
      const uint64_t t = clamped(ts[event.query]);
      emit_zone_end(t);
      if (--depth == 0) {
        topLevelEnd = t;
      }
    } else {
      if (ts[event.query] == 0 && ts[event.query + 1] == 0) {
        dropped[event.query / 2] = true;
        continue;
      }
      const uint64_t t = clamped(ts[event.query]);
      if (depth++ == 0 && t > topLevelEnd) {
        idleNs += t - topLevelEnd;
      }
      emit_zone_begin(event.name, t);
    }
  }
  const uint64_t end = clamped(frameEnd);
  if (end > topLevelEnd) {
    idleNs += end - topLevelEnd;
  }
  emit_zone_end(end);
  g_lastEmittedTs = prev;

  TracyPlot("aurora: gpuFrameMs", double(frameEnd - frameBegin) * 1e-6);
  TracyPlot("aurora: gpuIdleMs", double(idleNs) * 1e-6);
  TracyPlot("aurora: gpuPasses", int64_t(slot.passCount));
}
#endif

Slot& record_slot() { return g_slots[g_recordSlot]; }

uint32_t alloc_zone() {
  if (!g_frameActive) {
    return UINT32_MAX;
  }
  if (g_zoneCount >= MaxZones) {
    ++record_slot().dropped;
    return UINT32_MAX;
  }
  return g_zoneCount++;
}

bool collecting(uint64_t sequence) noexcept {
  return g_always.load(std::memory_order_relaxed) || sequence <= g_requestedUntil.load(std::memory_order_relaxed);
}

void ensure_resources() {
  if (g_resourcesCreated) {
    return;
  }
  constexpr wgpu::QuerySetDescriptor querySetDescriptor{
      .label = "GPU profiler timestamps",
      .type = wgpu::QueryType::Timestamp,
      .count = QueryCount,
  };
  g_querySet = g_device.CreateQuerySet(&querySetDescriptor);
  constexpr wgpu::BufferDescriptor resolveDescriptor{
      .label = "GPU profiler resolve",
      .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
      .size = ReadbackSize,
  };
  g_resolveBuffer = g_device.CreateBuffer(&resolveDescriptor);
  constexpr wgpu::BufferDescriptor readbackDescriptor{
      .label = "GPU profiler readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = ReadbackSize,
  };
  for (auto& slot : g_slots) {
    slot.readback = g_device.CreateBuffer(&readbackDescriptor);
    slot.events.reserve(MaxZones * 2);
    slot.state = SlotState::Free;
  }
  g_recordSlot = 0;
  g_emitSlot = 0;
  g_framePending = false;
  g_resourcesCreated = true;
}

void release_resources() {
  g_querySet = {};
  g_resolveBuffer = {};
  for (auto& slot : g_slots) {
    slot.readback = {};
  }
  g_resourcesCreated = false;
}

bool ring_idle() {
  return std::ranges::all_of(
      g_slots, [](const Slot& slot) { return slot.state.load(std::memory_order_acquire) == SlotState::Free; });
}

void drain_mapped() {
  while (true) {
    auto& slot = g_slots[g_emitSlot];
    const auto state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::Mapped) {
      publish(slot);
#ifdef TRACY_ENABLE
      emit_frame(slot);
#endif
      slot.readback.Unmap();
    } else if (state != SlotState::Failed) {
      break;
    }
    slot.state.store(SlotState::Free, std::memory_order_release);
    g_emitSlot = (g_emitSlot + 1) % RingDepth;
  }
}
} // namespace

namespace detail {
void aggregate(std::span<const Event> events, std::span<const uint64_t> timestamps, uint64_t frameBegin,
               uint64_t frameEnd, FrameTimes& out) {
  struct Open {
    const char* name;
    uint32_t endQuery;
    bool counted;
  };
  std::vector<Open> open;
  out.passes.clear();
  out.frameMs = frameEnd > frameBegin ? static_cast<float>(static_cast<double>(frameEnd - frameBegin) * 1e-6) : 0.f;
  for (const Event& event : events) {
    if (event.kind == EventKind::End) {
      const auto it = std::ranges::find(open, event.query, &Open::endQuery);
      if (it != open.end()) {
        open.erase(it);
      }
      continue;
    }
    const uint64_t begin = event.query + 1 < timestamps.size() ? timestamps[event.query] : 0;
    const uint64_t end = event.query + 1 < timestamps.size() ? timestamps[event.query + 1] : 0;
    const bool valid = begin != 0 && end != 0 && end >= begin;
    // An encoder zone and the passes it opens share the graph pass name, so only the outermost counts
    const bool nested =
        std::ranges::any_of(open, [&](const Open& outer) { return outer.counted && outer.name == event.name; });
    open.push_back({event.name, event.query + 1, valid && !nested});
    if (!valid || nested) {
      continue;
    }
    const auto ms = static_cast<float>(static_cast<double>(end - begin) * 1e-6);
    const auto it = std::ranges::find(out.passes, event.name, &PassTime::name);
    if (it != out.passes.end()) {
      it->ms += ms;
    } else {
      out.passes.push_back({event.name, ms});
    }
  }
}
} // namespace detail

void initialize() {
  g_supported = g_device.HasFeature(wgpu::FeatureName::TimestampQuery);
  // Browsers have no encoder timestamps; native devices inherit allow_unsafe_apis from the instance
  g_encoderTimestamps = g_supported && g_backendType != wgpu::BackendType::WebGPU;
  if (!g_supported) {
    Log.info("Timestamp queries unsupported; GPU timings disabled");
    return;
  }
#ifdef TRACY_ENABLE
  TracyPlotConfig("aurora: gpuFrameMs", tracy::PlotFormatType::Number, false, true, 0);
  TracyPlotConfig("aurora: gpuIdleMs", tracy::PlotFormatType::Number, false, true, 0);
  TracyPlotConfig("aurora: gpuPasses", tracy::PlotFormatType::Number, true, true, 0);
#endif
  Log.info("GPU timestamps available ({} zones max)", MaxZones);
}

void shutdown() {
  g_querySet = {};
  g_resolveBuffer = {};
  for (auto& slot : g_slots) {
    slot.readback = {};
    slot.events.clear();
    slot.passCount = 0;
    slot.sequence = 0;
    slot.dropped = 0;
    slot.state = SlotState::Free;
  }
  g_supported = false;
  g_resourcesCreated = false;
  g_encoderTimestamps = false;
  g_frameActive = false;
  g_framePending = false;
  g_sequence = 0;
  g_requestedUntil = 0;
  std::lock_guard lock{g_latestMutex};
  g_latest = {};
}

void frame_begin(const wgpu::CommandEncoder& encoder) {
  const uint64_t sequence = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
  if (!g_supported || !collecting(sequence)) {
    g_frameActive = false;
    // Once the last requested frame is read back, a frame without timings costs nothing again
    if (g_resourcesCreated && ring_idle()) {
      release_resources();
    }
    return;
  }
  ensure_resources();
  auto& slot = record_slot();
  if (slot.state != SlotState::Free) {
    g_frameActive = false;
    return;
  }
  slot.state = SlotState::Recording;
  slot.events.clear();
  slot.passCount = 0;
  slot.sequence = sequence;
  slot.dropped = 0;
  g_zoneCount = 0;
  g_frameActive = true;
  if (g_encoderTimestamps) {
    encoder.WriteTimestamp(g_querySet, FrameBeginQuery);
  }
}

void frame_end(const wgpu::CommandEncoder& encoder) {
  if (!g_frameActive) {
    return;
  }
  g_frameActive = false;
  auto& slot = record_slot();
  if (slot.events.empty() && !g_encoderTimestamps) {
    slot.state = SlotState::Free;
    return;
  }
  if (g_encoderTimestamps) {
    encoder.WriteTimestamp(g_querySet, FrameEndQuery);
  }
  encoder.ResolveQuerySet(g_querySet, 0, QueryCount, g_resolveBuffer, 0);
  encoder.CopyBufferToBuffer(g_resolveBuffer, 0, slot.readback, 0, ReadbackSize);
  g_resolveCount.fetch_add(1, std::memory_order_relaxed);
  g_framePending = true;
}

void after_submit() {
  if (!g_resourcesCreated) {
    return;
  }
  if (g_framePending) {
    g_framePending = false;
    auto& slot = record_slot();
    slot.submitNs = now_ns();
    slot.state = SlotState::InFlight;
    slot.readback.MapAsync(wgpu::MapMode::Read, 0, ReadbackSize, wgpu::CallbackMode::AllowSpontaneous,
                           [&slot](wgpu::MapAsyncStatus status, wgpu::StringView) {
                             slot.state =
                                 status == wgpu::MapAsyncStatus::Success ? SlotState::Mapped : SlotState::Failed;
                           });
    g_recordSlot = (g_recordSlot + 1) % RingDepth;
  }
#ifdef TRACY_ENABLE
  {
    ZoneScopedN("ProcessEvents");
    g_instance.ProcessEvents();
  }
#endif
  drain_mapped();
}

const wgpu::PassTimestampWrites* pass_writes(std::string_view name) {
  const uint32_t index = alloc_zone();
  if (index == UINT32_MAX) {
    return nullptr;
  }
  auto& slot = record_slot();
  slot.events.push_back({intern_name(name), index * 2, EventKind::PassBegin});
  slot.events.push_back({nullptr, index * 2 + 1, EventKind::End});
  ++slot.passCount;

  static std::array<wgpu::PassTimestampWrites, 4> writes;
  static size_t writesIndex = 0;
  auto& out = writes[writesIndex++ % writes.size()];
  out = {
      .querySet = g_querySet,
      .beginningOfPassWriteIndex = index * 2,
      .endOfPassWriteIndex = index * 2 + 1,
  };
  return &out;
}

bool supported() noexcept { return g_supported; }

void request(uint32_t frames) noexcept {
  const uint64_t until = g_sequence.load(std::memory_order_relaxed) + frames;
  uint64_t current = g_requestedUntil.load(std::memory_order_relaxed);
  while (current < until && !g_requestedUntil.compare_exchange_weak(current, until, std::memory_order_relaxed)) {}
}

void set_always(bool always) noexcept { g_always.store(always, std::memory_order_relaxed); }

bool latest(FrameTimes& out) {
  std::lock_guard lock{g_latestMutex};
  if (g_latest.sequence == 0 || g_latest.sequence == out.sequence) {
    return false;
  }
  out = g_latest;
  return true;
}

uint64_t last_frame_sequence() noexcept { return g_sequence.load(std::memory_order_relaxed); }

bool wait_idle(uint64_t timeoutNs) {
  if (!g_resourcesCreated) {
    return true;
  }
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    g_instance.ProcessEvents();
    drain_mapped();
    if (std::ranges::none_of(g_slots, [](const Slot& slot) { return slot.state == SlotState::InFlight; })) {
      return true;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    if (static_cast<uint64_t>(elapsed.count()) >= timeoutNs) {
      return false;
    }
    std::this_thread::sleep_for(IdlePollInterval);
  }
}

Zone::Zone(const wgpu::CommandEncoder& encoder, std::string_view name) {
  if (!g_frameActive || !g_encoderTimestamps) {
    return;
  }
  const uint32_t index = alloc_zone();
  if (index == UINT32_MAX) {
    return;
  }
  record_slot().events.push_back({intern_name(name), index * 2, EventKind::ZoneBegin});
  encoder.WriteTimestamp(g_querySet, index * 2);
  m_encoder = &encoder;
  m_endQuery = index * 2 + 1;
}

Zone::~Zone() {
  if (m_encoder == nullptr) {
    return;
  }
  record_slot().events.push_back({nullptr, m_endQuery, EventKind::End});
  m_encoder->WriteTimestamp(g_querySet, m_endQuery);
}

namespace testing {
uint64_t resolve_count() noexcept { return g_resolveCount.load(std::memory_order_relaxed); }

bool resources_created() noexcept { return g_resourcesCreated; }

std::span<const detail::Event> recording_events() noexcept { return record_slot().events; }
} // namespace testing

} // namespace aurora::webgpu::gpu_prof

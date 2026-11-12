#include "gate.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../settings.hpp"
#include "../shadows/vsm_invalidate.hpp"
#include "../shadows/vsm_pages.hpp"
#include "../../webgpu/gpu.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace aurora::render::fakeshadow::gate {
namespace {
using shadows::vsm::CasterObjects;

constexpr uint64_t CastBitBytes = uint64_t{CastBitWords} * 4;
constexpr uint32_t FifoNotes = 16;
constexpr int FlushWaitMs = 500;
constexpr uint32_t MaxReplayFrames = 64;
constexpr size_t MaxReplayEntries = 256;

struct Slot {
  uint64_t key = 0;
  uint32_t hold = 0;
  uint32_t returns = 0;
  uint64_t lastChange = 0;
  bool used = false;
  bool skipped = false;
  bool changed = false;
  ShadowDecision decision = ShadowDecision::None;
  ShadowSource source = ShadowSource::Unknown;
  ShadowClass cls = ShadowClass::Real;
  std::vector<CasterRow> casters;
};
using Table = std::array<Slot, TableSlots>;

struct RingCaster {
  uint64_t id = 0;
  bool provisional = false;
  std::vector<uint32_t> slots;
};

struct RingFrame {
  uint64_t frame = 0;
  bool used = false;
  bool latchChecked = false;
  bool castChecked = false;
  bool sunChecked = false;
  std::vector<RingCaster> casters;
};

struct CasterSet {
  CasterObjects live;
  std::vector<std::pair<uint64_t, CasterCover>> covers;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> slots;
  bool fake = false;

  [[nodiscard]] CasterCover cover(uint64_t id) const noexcept {
    if (fake) {
      const auto it = std::ranges::find(covers, id, &std::pair<uint64_t, CasterCover>::first);
      return it != covers.end() ? it->second : CasterCover::Missing;
    }
    return static_cast<CasterCover>(live.find(id).cover);
  }

  [[nodiscard]] std::vector<uint32_t> slot_list(uint64_t id) const {
    if (fake) {
      const auto it = std::ranges::find(slots, id, &std::pair<uint64_t, std::vector<uint32_t>>::first);
      return it != slots.end() ? it->second : std::vector<uint32_t>{};
    }
    const std::span<const uint32_t> found = live.find(id).slots;
    return {found.begin(), found.end()};
  }
};

struct Snapshot {
  GateState state;
  bool available = false;
  bool lightKeyRoom = false;
};

// Game thread
bool g_captured = false;
HintInputs g_hints;
Snapshot g_snap;
CasterSet g_casters;
uint64_t g_frame = 0;
ConsumerForce g_force = ConsumerForce::Auto;
AppliedFn g_applied = nullptr;
SealedFn g_sealed = nullptr;
Table g_prev;
Table g_cur;
GameShadowCounts g_current;
GameShadowCounts g_total;
std::array<RingFrame, RingFrames> g_ring;
std::atomic_bool g_castWanted{false};

// FIFO thread writes, the game thread reads at the latch
std::array<std::atomic_uint64_t, FifoNotes> g_fifo{};

// Counter readbacks add these from map callbacks
struct DdPixels {
  std::atomic_uint64_t pbr{0}, legacy{0}, lastPbr{0}, lastLegacy{0};
} g_dd;

// Render worker and map callbacks
struct ReadbackSlot {
  wgpu::Buffer buffer;
  bool busy = false;
};
std::mutex g_bitsMutex;
std::array<ReadbackSlot, CastBitSlots> g_readback;
std::deque<std::pair<uint64_t, std::vector<uint32_t>>> g_arrived;

// FIFO thread
std::atomic_bool g_observe{false};
ReplayFrame g_replay;
std::vector<float> g_alphas;
std::mutex g_replayMutex;
std::deque<ReplayFrame> g_replayDone;

uint32_t slot_hash(uint64_t key) noexcept {
  uint64_t x = key + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return static_cast<uint32_t>((x ^ (x >> 31)) & (TableSlots - 1));
}

const Slot* find(const Table& table, uint64_t key) noexcept {
  const uint32_t start = slot_hash(key);
  for (uint32_t i = 0; i < TableSlots; ++i) {
    const Slot& slot = table[(start + i) & (TableSlots - 1)];
    if (!slot.used) {
      return nullptr;
    }
    if (slot.key == key) {
      return &slot;
    }
  }
  return nullptr;
}

Slot* claim(Table& table, uint64_t key) noexcept {
  const uint32_t start = slot_hash(key);
  for (uint32_t i = 0; i < TableSlots; ++i) {
    Slot& slot = table[(start + i) & (TableSlots - 1)];
    if (!slot.used) {
      slot = Slot{.key = key, .used = true};
      return &slot;
    }
  }
  return nullptr;
}

template <typename F>
void count(F&& f) noexcept {
  f(g_current);
  f(g_total);
}

bool caster_present(CasterCover cover) noexcept { return cover == CasterCover::Full || cover == CasterCover::Partial; }

RingFrame& ring_frame(uint64_t frame) noexcept { return g_ring[frame % RingFrames]; }

bool bit_set(std::span<const uint32_t> words, uint32_t slot) noexcept {
  return slot / 32 < words.size() && ((words[slot / 32] >> (slot % 32)) & 1u) != 0;
}

void check_cast_bits(RingFrame& ring, std::span<const uint32_t> words) noexcept {
  for (const RingCaster& caster : ring.casters) {
    if (caster.slots.empty()) {
      continue;
    }
    const bool cast = std::ranges::any_of(caster.slots, [&](uint32_t slot) { return bit_set(words, slot); });
    if (!cast) {
      count([](GameShadowCounts& c) { ++c.missingCasterCast; });
    }
  }
  ring.castChecked = true;
}

void drain_arrivals() {
  std::deque<std::pair<uint64_t, std::vector<uint32_t>>> arrived;
  {
    std::lock_guard lock{g_bitsMutex};
    arrived.swap(g_arrived);
  }
  std::deque<std::pair<uint64_t, std::vector<uint32_t>>> keep;
  for (auto& [frame, words] : arrived) {
    RingFrame& ring = ring_frame(frame);
    if (!ring.used || ring.frame != frame || ring.castChecked) {
      continue;
    }
    // Slots resolve at the next latch, so bits that arrive earlier wait for it
    if (!ring.latchChecked) {
      keep.emplace_back(frame, std::move(words));
      continue;
    }
    check_cast_bits(ring, words);
  }
  if (!keep.empty()) {
    std::lock_guard lock{g_bitsMutex};
    for (auto& entry : keep) {
      g_arrived.push_back(std::move(entry));
    }
  }
}

void check_latch(RingFrame& ring) {
  for (RingCaster& caster : ring.casters) {
    if (!caster_present(g_casters.cover(caster.id))) {
      count([&](GameShadowCounts& c) {
        ++c.missingCasterLatch;
        c.provisionalFailed += caster.provisional ? 1 : 0;
      });
      continue;
    }
    caster.slots = g_casters.slot_list(caster.id);
  }
  ring.latchChecked = true;
}

void check_fifo_notes() noexcept {
  for (const std::atomic_uint64_t& note : g_fifo) {
    const uint64_t value = note.load(std::memory_order_acquire);
    if (value == 0) {
      continue;
    }
    const uint64_t frame = value >> 1;
    RingFrame& ring = ring_frame(frame);
    if (!ring.used || ring.frame != frame || ring.sunChecked || ring.casters.empty()) {
      continue;
    }
    ring.sunChecked = true;
    if ((value & 1u) == 0) {
      count([](GameShadowCounts& c) { ++c.sunLossFrames; });
    }
  }
}

void latch_with(const Snapshot& snap) {
  g_snap = snap;
  g_frame = snap.state.frame;
  for (RingFrame& ring : g_ring) {
    if (ring.used && !ring.latchChecked && ring.frame < g_frame) {
      check_latch(ring);
    }
  }
  check_fifo_notes();
  drain_arrivals();
  RingFrame& next = ring_frame(g_frame);
  if (next.used && !next.castChecked) {
    const auto pending = static_cast<uint64_t>(
        std::ranges::count_if(next.casters, [](const RingCaster& caster) { return !caster.slots.empty(); }));
    g_total.castBitsPending += pending;
  }
  next = RingFrame{.frame = g_frame, .used = true};
  std::swap(g_prev, g_cur);
  g_cur.fill(Slot{});
  g_current = {};
  g_castWanted.store(snap.available && snap.state.consumer, std::memory_order_relaxed);
}

bool forced_consumer(bool& prevResolveRan) noexcept {
  switch (g_force) {
  case ConsumerForce::Off:
    prevResolveRan = false;
    return false;
  case ConsumerForce::On:
    prevResolveRan = true;
    return true;
  case ConsumerForce::Flip30:
    prevResolveRan = (g_hints.hintFrame / HoldFrames) % 2 == 0;
    return prevResolveRan;
  case ConsumerForce::Auto:
    break;
  }
  const bool applied = g_applied != nullptr && g_applied();
  prevResolveRan = g_sealed != nullptr && g_sealed(g_hints.hintFrame);
  return applied;
}

bool double_dark(const GameShadowEntry& entry, ShadowDecision decision) noexcept {
  switch (decision) {
  case ShadowDecision::Skip:
  case ShadowDecision::KeepHold:
  case ShadowDecision::KeepCoverage:
  case ShadowDecision::KeepClass:
  case ShadowDecision::KeepInterior:
  case ShadowDecision::KeepSetting:
    return false;
  default:
    break;
  }
  const GateState& s = g_snap.state;
  if (!g_snap.available || !s.consumer || !covered(s, g_snap.lightKeyRoom, entry.source, entry.cls)) {
    return false;
  }
  if (marker_class(entry.cls) || (entry.cls == ShadowClass::Cue && s.keepCues) ||
      ((entry.source == ShadowSource::SunMoon || entry.cls == ShadowClass::Blob) && s.interior)) {
    return false;
  }
  return std::ranges::any_of(entry.casters,
                             [](const GameShadowCaster& caster) { return caster_present(g_casters.cover(caster.id)); });
}

struct Outcome {
  ShadowDecision decision = ShadowDecision::None;
  uint32_t hold = 0;
  bool hasHold = false;
  uint32_t provisional = 0;
};

Outcome evaluate(const GameShadowEntry& entry, const Slot* prev) noexcept {
  const GateState& s = g_snap.state;
  if (const std::optional<ShadowDecision> rule = setting_decision(s.game, entry.cls)) {
    return {.decision = *rule};
  }
  if (!g_snap.available || !s.consumer || !s.prevResolveRan) {
    return {.decision = ShadowDecision::KeepNoConsumer};
  }
  if (entry.source == ShadowSource::Vertical || entry.source == ShadowSource::Unknown || marker_class(entry.cls) ||
      (entry.cls == ShadowClass::Cue && s.keepCues)) {
    return {.decision = ShadowDecision::KeepClass};
  }
  if ((entry.source == ShadowSource::SunMoon || entry.cls == ShadowClass::Blob) && s.interior) {
    return {.decision = ShadowDecision::KeepInterior};
  }
  if (!covered(s, g_snap.lightKeyRoom, entry.source, entry.cls)) {
    return {.decision = ShadowDecision::KeepSource};
  }
  if (entry.casters.empty() || entry.castersTruncated) {
    return {.decision = ShadowDecision::KeepCasterUnknown};
  }
  uint32_t passing = 0;
  uint32_t provisional = 0;
  uint32_t excluded = 0;
  bool missing = false;
  for (const GameShadowCaster& caster : entry.casters) {
    switch (g_casters.cover(caster.id)) {
    case CasterCover::Full:
    case CasterCover::Partial:
      ++passing;
      break;
    case CasterCover::ExcludedOnly:
      ++excluded;
      break;
    case CasterCover::Missing:
      if (caster.enteredNow) {
        ++passing;
        ++provisional;
      } else {
        missing = true;
      }
      break;
    }
  }
  count([&](GameShadowCounts& c) { c.partialCaster += excluded; });
  if (missing || passing == 0) {
    return {.decision = ShadowDecision::KeepCaster};
  }
  const uint32_t prevHold = prev != nullptr ? prev->hold : 0;
  // A coverage gap freezes the hold, so a load never restarts the 30 frames already earned
  if (!s.coverageComplete) {
    return {.decision = prev != nullptr && prev->skipped ? ShadowDecision::Skip : ShadowDecision::KeepCoverage,
            .hold = prevHold,
            .hasHold = true,
            .provisional = provisional};
  }
  const uint32_t hold = prevHold + 1;
  return {.decision = hold >= HoldFrames ? ShadowDecision::Skip : ShadowDecision::KeepHold,
          .hold = hold,
          .hasHold = true,
          .provisional = provisional};
}

void record_skip(const GameShadowEntry& entry) {
  RingFrame& ring = ring_frame(g_frame);
  if (!ring.used || ring.frame != g_frame) {
    return;
  }
  for (const GameShadowCaster& caster : entry.casters) {
    if (g_casters.cover(caster.id) == CasterCover::ExcludedOnly) {
      continue;
    }
    ring.casters.push_back({.id = caster.id, .provisional = !caster_present(g_casters.cover(caster.id))});
  }
}
} // namespace

HintInputs hint_inputs(const detail::SceneHints& hints) noexcept {
  HintInputs out;
  out.hintFrame = hints.hintFrame;
  if (hints.has(detail::SceneHasLights)) {
    out.sunDeclared = shadows::vsm::select_sun(hints).found;
    for (const AuroraLight& light : hints.lights) {
      out.keyRoomDeclared =
          out.keyRoomDeclared || (light.type == AURORA_LIGHT_DIRECTIONAL && (light.flags & AURORA_LIGHT_KEY_ROOM) != 0);
      out.lanternDeclared = out.lanternDeclared || (light.flags & AURORA_LIGHT_LANTERN) != 0;
    }
  }
  if (hints.has(detail::SceneHasEnvironment)) {
    const gx::hints::Environment& env = hints.environment;
    out.interior = (env.value.flags & AURORA_ENV_INTERIOR) != 0;
    out.skyVisibility = env.value.skyVisibility;
    out.shadowSun = env.value.shadowSun;
    out.thunder = env.value.shadowThunderAdd;
  }
  return out;
}

bool covered(const GateState& state, bool lightKeyRoom, ShadowSource source, ShadowClass cls) noexcept {
  if (marker_class(cls) || source == ShadowSource::Vertical || source == ShadowSource::Unknown) {
    return false;
  }
  const bool sunMoon =
      state.sunDeclared && state.skyVisibility > 0.f && state.sunGeometric && !state.interior && !lightKeyRoom;
  if (cls == ShadowClass::Blob || cls == ShadowClass::Cue) {
    return state.skyVisibility > 0.f ? sunMoon : state.localCovered;
  }
  switch (source) {
  case ShadowSource::SunMoon:
    return sunMoon;
  case ShadowSource::RoomKey:
    return state.keyRoomDeclared && state.sunGeometric && lightKeyRoom;
  default:
    return state.localCovered;
  }
}

void shutdown() noexcept {
  std::unique_lock lock{g_bitsMutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{FlushWaitMs};
  const auto pending = [] { return std::ranges::any_of(g_readback, [](const ReadbackSlot& s) { return s.busy; }); };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  g_readback = {};
  g_arrived.clear();
  lock.unlock();
  testing::reset();
}

void capture_inputs() noexcept {
  g_hints = hint_inputs(detail::live_scene_hints());
  g_captured = true;
}

void on_latch(const Settings& settings) noexcept {
  Snapshot snap;
  GateState& s = snap.state;
  s.consumer = forced_consumer(s.prevResolveRan);
  s.sunDeclared = g_hints.sunDeclared;
  s.keyRoomDeclared = g_hints.keyRoomDeclared;
  s.lanternDeclared = g_hints.lanternDeclared;
  s.interior = g_hints.interior;
  s.skyVisibility = g_hints.skyVisibility;
  const shadows::vsm::SunPlan& plan = shadows::vsm::plan();
  s.sunGeometric = plan.producing;
  snap.lightKeyRoom = (plan.lightFlags & AURORA_LIGHT_KEY_ROOM) != 0;
  s.sunCovered = covered(s, snap.lightKeyRoom, ShadowSource::SunMoon, ShadowClass::Real);
  s.keyRoomCovered = covered(s, snap.lightKeyRoom, ShadowSource::RoomKey, ShadowClass::Real);
  s.localCovered = false;
  s.cloudShadows = settings.sky.cloudShadows;
  s.clouds = settings.sky.clouds != Quality::Off;
  s.ao = settings.ao != Quality::Off;
  s.coverageComplete = retained_coverage_complete();
  s.keepCues = settings.shadows.keepCues;
  s.game = settings.shadows.game;
  s.frame = static_cast<uint32_t>(g_hints.hintFrame + 1);
  snap.available = g_captured && hints_enabled();
  g_casters = CasterSet{.live = shadows::vsm::caster_objects()};
  latch_with(snap);
}

void register_consumer(AppliedFn applied, SealedFn sealed) noexcept {
  g_applied = applied;
  g_sealed = sealed;
}

void set_force(ConsumerForce force) noexcept { g_force = force; }

ConsumerForce force() noexcept { return g_force; }

GateState state() noexcept { return g_snap.state; }

uint64_t decision_frame() noexcept { return g_frame; }

ShadowDecision decide(const GameShadowEntry& entry) noexcept {
  if (const Slot* repeat = find(g_cur, entry.key)) {
    return repeat->decision;
  }
  const Slot* prev = find(g_prev, entry.key);
  Outcome out = evaluate(entry, prev);
  Slot* slot = claim(g_cur, entry.key);
  if (slot == nullptr) {
    count([](GameShadowCounts& c) { ++c.overflow; });
    if (out.decision == ShadowDecision::Skip || out.decision == ShadowDecision::KeepHold) {
      out.decision = ShadowDecision::KeepOverflow;
    }
  }
  const bool skipped = out.decision == ShadowDecision::Skip;
  const bool wasSkipped = prev != nullptr && prev->skipped;
  count([&](GameShadowCounts& c) {
    ++c.entries;
    ++c.decisions[static_cast<size_t>(out.decision)];
    c.userKept += out.decision == ShadowDecision::KeepSetting ? 1 : 0;
    c.provisional += skipped || out.decision == ShadowDecision::KeepHold ? out.provisional : 0;
  });
  if (double_dark(entry, out.decision)) {
    count([](GameShadowCounts& c) { ++c.doubleDarkEntries; });
  }
  uint32_t returns = prev != nullptr ? prev->returns : 0;
  bool changed = prev != nullptr && prev->changed;
  uint64_t lastChange = prev != nullptr ? prev->lastChange : 0;
  if (skipped != wasSkipped) {
    const bool violation = changed && g_frame - lastChange < HoldFrames;
    returns += wasSkipped ? 1 : 0;
    count([&](GameShadowCounts& c) {
      ++c.transitions;
      c.flipViolations += violation ? 1 : 0;
      if (wasSkipped) {
        ++c.safetyReturns[static_cast<size_t>(out.decision)];
        c.safetyReturnsPerKeyMax = std::max<uint64_t>(c.safetyReturnsPerKeyMax, returns);
      }
    });
    changed = true;
    lastChange = g_frame;
  }
  if (slot != nullptr) {
    slot->hold = out.hasHold ? out.hold : 0;
    slot->returns = returns;
    slot->lastChange = lastChange;
    slot->changed = changed;
    slot->skipped = skipped;
    slot->decision = out.decision;
    slot->source = entry.source;
    slot->cls = entry.cls;
    slot->casters.reserve(entry.casters.size());
    for (const GameShadowCaster& caster : entry.casters) {
      slot->casters.push_back({.id = caster.id, .cover = g_casters.cover(caster.id), .enteredNow = caster.enteredNow});
    }
  }
  // Remove skips whatever casts, so only coverage skips owe their casters
  if (skipped && g_snap.state.game == GameShadows::Auto) {
    record_skip(entry);
  }
  return out.decision;
}

GameShadowStats stats(bool flush) noexcept {
  if (flush) {
    std::unique_lock lock{g_bitsMutex};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{FlushWaitMs};
    const auto pending = [] { return std::ranges::any_of(g_readback, [](const ReadbackSlot& s) { return s.busy; }); };
    while (pending() && std::chrono::steady_clock::now() < deadline) {
      lock.unlock();
      if (webgpu::g_instance) {
        webgpu::g_instance.ProcessEvents();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
      lock.lock();
    }
    lock.unlock();
    for (RingFrame& ring : g_ring) {
      if (ring.used && !ring.latchChecked && ring.frame < g_frame) {
        check_latch(ring);
      }
    }
    drain_arrivals();
  }
  GameShadowStats out{.frame = static_cast<uint32_t>(g_frame), .current = g_current, .total = g_total};
  out.current.ddPixelsPbr = g_dd.lastPbr.load(std::memory_order_relaxed);
  out.current.ddPixelsLegacy = g_dd.lastLegacy.load(std::memory_order_relaxed);
  out.total.ddPixelsPbr = g_dd.pbr.load(std::memory_order_relaxed);
  out.total.ddPixelsLegacy = g_dd.legacy.load(std::memory_order_relaxed);
  uint64_t pending = 0;
  for (const RingFrame& ring : g_ring) {
    if (ring.used && ring.latchChecked && !ring.castChecked) {
      pending += static_cast<uint64_t>(
          std::ranges::count_if(ring.casters, [](const RingCaster& caster) { return !caster.slots.empty(); }));
    }
  }
  out.current.castBitsPending = pending;
  return out;
}

std::vector<EntryRow> entry_rows() {
  std::vector<EntryRow> rows;
  for (const Table* table : {&g_cur, &g_prev}) {
    if (!rows.empty()) {
      break;
    }
    for (const Slot& slot : *table) {
      if (slot.used) {
        rows.push_back({.key = slot.key,
                        .source = slot.source,
                        .cls = slot.cls,
                        .decision = slot.decision,
                        .hold = slot.hold,
                        .casters = slot.casters});
      }
    }
  }
  std::ranges::sort(rows, {}, &EntryRow::key);
  return rows;
}

bool cast_bits_wanted() noexcept { return g_castWanted.load(std::memory_order_relaxed); }

void note_fifo_frame(uint64_t hintFrame, bool sunEnabled) noexcept {
  g_fifo[hintFrame % FifoNotes].store(hintFrame << 1 | (sunEnabled ? 1u : 0u), std::memory_order_release);
}

void note_dd_pixels(uint64_t hintFrame, uint64_t pbr, uint64_t legacy) noexcept {
  g_dd.pbr.fetch_add(pbr, std::memory_order_relaxed);
  g_dd.legacy.fetch_add(legacy, std::memory_order_relaxed);
  g_dd.lastPbr.store(pbr, std::memory_order_relaxed);
  g_dd.lastLegacy.store(legacy, std::memory_order_relaxed);
  std::lock_guard lock{g_replayMutex};
  for (ReplayFrame& frame : g_replayDone) {
    if (frame.frame == hintFrame) {
      frame.ddPixelsPbr = pbr;
      frame.ddPixelsLegacy = legacy;
    }
  }
}

void copy_cast_bits(const wgpu::CommandEncoder& encoder, const BufferView& source, uint64_t hintFrame) noexcept {
  if (!source.buffer || source.size < CastBitBytes) {
    return;
  }
  wgpu::Buffer buffer;
  size_t index = 0;
  {
    std::lock_guard lock{g_bitsMutex};
    const auto it = std::ranges::find_if(g_readback, [](const ReadbackSlot& slot) { return !slot.busy; });
    if (it == g_readback.end()) {
      return;
    }
    if (!it->buffer) {
      const wgpu::BufferDescriptor descriptor{
          .label = "fakeshadow.cast_bits",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = CastBitBytes,
      };
      it->buffer = webgpu::g_device.CreateBuffer(&descriptor);
    }
    it->busy = true;
    buffer = it->buffer;
    index = static_cast<size_t>(it - g_readback.begin());
  }
  encoder.CopyBufferToBuffer(source.buffer, source.offset, buffer, 0, CastBitBytes);
  detail::queue_after_frame([buffer, index, hintFrame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, CastBitBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, index, hintFrame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      std::vector<uint32_t> words;
                      if (status == wgpu::MapAsyncStatus::Success) {
                        if (const void* bytes = buffer.GetConstMappedRange(0, CastBitBytes); bytes != nullptr) {
                          words.resize(CastBitWords);
                          std::memcpy(words.data(), bytes, CastBitBytes);
                        }
                        buffer.Unmap();
                      }
                      std::lock_guard lock{g_bitsMutex};
                      if (!words.empty()) {
                        g_arrived.emplace_back(hintFrame, std::move(words));
                        while (g_arrived.size() > RingFrames * 2) {
                          g_arrived.pop_front();
                        }
                      }
                      g_readback[index].busy = false;
                    });
  });
}

void set_replay_observer(bool on) noexcept { g_observe.store(on, std::memory_order_relaxed); }

bool replay_observer() noexcept { return g_observe.load(std::memory_order_relaxed); }

bool drop_active() noexcept {
  static const bool drop = detail::game_shadows_env() == detail::GameShadowsEnv::Drop;
  return drop;
}

void observe_object(uint32_t categories, uint32_t subId) noexcept {
  if ((categories & AURORA_CAT_FAKE_SHADOW) == 0) {
    return;
  }
  ++g_replay.objects[static_cast<size_t>(tag_class(subId))][static_cast<size_t>(tag_decision(subId))];
}

void observe_draw(uint64_t object, uint32_t subId, float tevReg0Alpha) noexcept {
  ReplayEntry* entry = nullptr;
  for (auto it = g_replay.entries.rbegin(); it != g_replay.entries.rend(); ++it) {
    if (it->object == object && it->subId == subId) {
      entry = &*it;
      break;
    }
  }
  if (entry == nullptr && g_replay.entries.size() < MaxReplayEntries) {
    entry = &g_replay.entries.emplace_back(
        ReplayEntry{.object = object, .subId = subId, .alphaMin = tevReg0Alpha, .alphaMax = tevReg0Alpha});
  }
  if (entry != nullptr) {
    ++entry->draws;
    entry->alphaMin = std::min(entry->alphaMin, tevReg0Alpha);
    entry->alphaMax = std::max(entry->alphaMax, tevReg0Alpha);
  }
  if (tag_class(subId) != ShadowClass::Real) {
    return;
  }
  ++g_replay.realDraws;
  g_alphas.push_back(tevReg0Alpha);
}

void count_drop() noexcept { ++g_replay.drops; }

void seal_replay_frame(uint64_t hintFrame, const detail::SceneHints& hints) noexcept {
  ReplayFrame frame = std::exchange(g_replay, ReplayFrame{});
  frame.frame = hintFrame;
  if (!g_alphas.empty()) {
    std::ranges::sort(g_alphas);
    frame.realAlphaMedian = g_alphas[g_alphas.size() / 2];
    frame.realAlphaMax = g_alphas.back();
  }
  g_alphas.clear();
  const HintInputs in = hint_inputs(hints);
  frame.shadowSun = in.shadowSun;
  frame.thunder = in.thunder;
  std::lock_guard lock{g_replayMutex};
  g_replayDone.push_back(frame);
  while (g_replayDone.size() > MaxReplayFrames) {
    g_replayDone.pop_front();
  }
}

std::optional<ReplayFrame> take_replay_frame() noexcept {
  std::lock_guard lock{g_replayMutex};
  if (g_replayDone.empty()) {
    return std::nullopt;
  }
  ReplayFrame frame = g_replayDone.front();
  g_replayDone.pop_front();
  return frame;
}

namespace testing {
void reset() noexcept {
  g_captured = false;
  g_hints = {};
  g_snap = {};
  g_casters = {};
  g_frame = 0;
  g_force = ConsumerForce::Auto;
  g_applied = nullptr;
  g_sealed = nullptr;
  g_prev.fill(Slot{});
  g_cur.fill(Slot{});
  g_current = {};
  g_total = {};
  g_ring = {};
  g_castWanted.store(false, std::memory_order_relaxed);
  for (std::atomic_uint64_t& note : g_fifo) {
    note.store(0, std::memory_order_relaxed);
  }
  {
    std::lock_guard lock{g_bitsMutex};
    g_arrived.clear();
  }
  g_observe.store(false, std::memory_order_relaxed);
  g_replay = {};
  g_alphas.clear();
  for (std::atomic_uint64_t* value : {&g_dd.pbr, &g_dd.legacy, &g_dd.lastPbr, &g_dd.lastLegacy}) {
    value->store(0, std::memory_order_relaxed);
  }
  std::lock_guard lock{g_replayMutex};
  g_replayDone.clear();
}

void latch(const Inputs& inputs) noexcept {
  g_casters = CasterSet{.covers = inputs.casters, .slots = inputs.slots, .fake = true};
  Snapshot snap{.state = inputs.state, .available = inputs.available, .lightKeyRoom = inputs.lightKeyRoom};
  if (snap.state.frame == 0) {
    snap.state.frame = static_cast<uint32_t>(g_frame + 1);
  }
  latch_with(snap);
}

void deliver_cast_bits(uint64_t hintFrame, std::span<const uint32_t> words) noexcept {
  std::lock_guard lock{g_bitsMutex};
  g_arrived.emplace_back(hintFrame, std::vector<uint32_t>{words.begin(), words.end()});
}

uint32_t table_slot(uint64_t key) noexcept { return slot_hash(key); }
} // namespace testing
} // namespace aurora::render::fakeshadow::gate

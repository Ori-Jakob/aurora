#include "identity.hpp"

#include "classify.hpp"
#include "gc.hpp"
#include "streamout.hpp"
#include "../hints.hpp"
#include "../stage_data.hpp"
#include "../../gfx/hash.hpp"
#include "../../gfx/recording.hpp"
#include "../../gx/command_processor.hpp"
#include "../../gx/draw_route.hpp"
#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"

#include <aurora/render_caps.hpp>
#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string_view>
#include <utility>

namespace aurora::render::scene::retained {
namespace {
constexpr Module Log{"aurora::render::retained"};

constexpr uint32_t IndexMask = IndexSlots - 1;
constexpr uint32_t ObjectSlots = 32768;
constexpr uint32_t ObjectMask = ObjectSlots - 1;
constexpr uint32_t MaxObjects = ObjectSlots / 4 * 3;
constexpr uint32_t ModelSlots = 16384;
constexpr uint32_t ModelMask = ModelSlots - 1;
constexpr uint32_t MaxModels = ModelSlots / 4 * 3;
constexpr uint32_t ArrayHashSlots = 4096;
constexpr uint32_t ArrayHashMask = ArrayHashSlots - 1;
constexpr uint32_t ArrayHashProbes = 8;
constexpr uint32_t MaxMatrices = MaxEntries + 4096 * (PaletteSlots - 1);
constexpr uint32_t MaxLayouts = 1024;
constexpr uint32_t FramePoolSize = 8;
constexpr uint32_t PositionSlots = gx::MaxPnMtx + gx::MaxTexMtx;
constexpr uint32_t StaticBits = class_bit::Static | class_bit::Immobile;
constexpr uint32_t NoIndex = UINT32_MAX;
constexpr uint64_t Never = UINT64_MAX;
constexpr uint8_t CancelledEvent = 0xFF;
constexpr uint16_t ResetSlot = 0xFFFF;
static_assert((IndexSlots & IndexMask) == 0 && (ObjectSlots & ObjectMask) == 0 && (ModelSlots & ModelMask) == 0);
static_assert(MaxSlots <= ResetSlot && MaxLive <= MaxSlots && MaxSlots < IndexSlots);

struct PendingEvent {
  uint8_t type = 0;
  uint8_t reason = 0;
  uint16_t slot = 0;
  uint32_t generation = 0;
};
static_assert(sizeof(PendingEvent) == 8);

struct Slot {
  InstanceInfo info;
  Affine world{};
  uint64_t seenHintFrame = Never;
  uint64_t signature = 0;
  uint32_t generation = 0;
  uint32_t createdEvent = NoIndex;
  uint32_t object = NoIndex;
  uint32_t livePos = NoIndex;
  bool live = false;
  bool delivered = false;
  bool createdPending = false;
  bool latched = false;
  bool captureOnlySeen = false;
  bool visibleSeen = false;
};

struct ObjectEntry {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  uint64_t last = Never;
  uint64_t prev = Never;
  uint32_t statics = 0;
  bool used = false;
};

struct ModelEntry {
  uint64_t id = 0;
  uint32_t generation = 0;
  bool used = false;
};

struct ArrayHash {
  const void* data = nullptr;
  uint32_t size = 0;
  uint32_t generation = 0;
  uint64_t frame = Never;
  uint64_t hash = 0;
};

struct Pending {
  bool valid = false;
  uint16_t flags = 0;
  ObjectRef object;
  uint32_t categories = 0;
  uint32_t objectFlags = 0;
  Kind kind = Kind::Foreign;
  uint16_t ordinal = 0;
  GXPrimitive prim = GX_TRIANGLES;
  GXVtxFmt fmt = GX_VTXFMT0;
  uint32_t vtxCount = 0;
  bool indexed = false;
  std::span<const uint8_t> vertices;
  std::span<const uint8_t> indices;
  bool hasWorld = false;
  Mat3x4<float> world{};
};

struct DrawRef {
  gfx::Range vertRange{};
  gfx::Range idxRange{};
  uint32_t indexCount = 0;
  gx::DrawImmediateData immediates{};
  uint32_t record = 0;
  const gx::AttrConfig* attrs = nullptr;
  uint8_t stride = 0;
};

struct FifoFrame {
  uint64_t hintFrame = 0;
  uint64_t sceneFrame = 0;
  bool viewReady = false;
  Affine invView{};
  double eye = 0.0;
  std::array<double, 3> transientOrigin{};
  bool opaquePrepared = false;
  bool staticCreated = false;
  FrameCounts counts;
};

struct Totals {
  uint64_t created = 0;
  uint64_t evicted = 0;
  uint64_t captured = 0;
  uint64_t churn = 0;
  uint64_t lostCaptures = 0;
  uint64_t resyncs = 0;
  uint64_t capturedOnly = 0;
};

class Zone {
public:
  Zone() noexcept : m_timed(gx::fifo::cpu_zone_timed(gx::fifo::CpuZoneRetained)) {
    if (m_timed) {
      m_start = std::chrono::steady_clock::now();
    }
  }
  ~Zone() {
    uint64_t ns = 0;
    if (m_timed) {
      ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count());
    }
    gx::fifo::cpu_zone_record(gx::fifo::CpuZoneRetained, ns);
  }
  Zone(const Zone&) = delete;
  Zone& operator=(const Zone&) = delete;

private:
  std::chrono::steady_clock::time_point m_start{};
  bool m_timed;
};

bool g_envRead = false;
uint8_t g_demand = 0;
bool g_checkMode = false;
uint64_t g_checkFirst = 0;
uint64_t g_checkLast = UINT64_MAX;
int g_overrun = -1;
bool g_active = false;
bool g_allocated = false;
bool g_wantCovered = false;
uint64_t g_sceneClock = 0;
bool g_producerCalled = false;
bool g_holdCalled = true;
bool g_producer = false;
bool g_held = true;
bool g_coverage = false;
uint64_t g_lastDisturb = 0;
std::array<SettlePredicate, MaxSettlePredicates> g_settle{};
uint32_t g_settleCount = 0;
std::atomic_uint64_t g_reportedLost{0};
std::atomic_bool g_captureActive{false};
std::atomic_bool g_coverageComplete{false};
Totals g_totals;
uint32_t g_peakLive = 0;

std::vector<Slot> g_slots;
std::vector<gc::Life> g_lives;
std::vector<uint32_t> g_liveList;
std::vector<gc::Eviction> g_evictions;
std::vector<uint32_t> g_index;
std::vector<uint32_t> g_free;
std::vector<ObjectEntry> g_objects;
std::vector<ModelEntry> g_models;
std::vector<ArrayHash> g_arrayHashes;
uint32_t g_objectCount = 0;
uint32_t g_modelCount = 0;
uint32_t g_lastObject = NoIndex;
uint32_t g_live = 0;
uint32_t g_liveStatics = 0;
uint32_t g_undeliveredStatics = 0;
uint32_t g_staticObjects = 0;
bool g_lifecycleChanged = true;
std::vector<Entry> g_log;
std::vector<gx::PnMtx> g_matrices;
std::vector<Layout> g_layouts;
std::vector<PendingEvent> g_events;
bool g_resync = false;
bool g_resetPending = false;
bool g_warnedOverflow = false;
bool g_warnedEntries = false;
Pending g_pending;
bool g_newGeometry = false;
FifoFrame g_frame;
std::array<std::shared_ptr<Frame>, FramePoolSize> g_pool;
std::shared_ptr<Frame> g_tentative;

std::mutex g_publishMutex;
RetainedStats g_published;
FrameCounts g_publishedCounts;
std::shared_ptr<const std::vector<uint64_t>> g_covered;

constexpr uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint32_t key_home(const Key& key) noexcept {
  const uint64_t mixed = splitmix64(key.id ^ (uint64_t{key.subId} << 32) ^ (uint64_t{key.generation} << 16) ^
                                    (uint64_t{key.ordinal} << 48) ^ (uint64_t{key.occurrence} << 40));
  return static_cast<uint32_t>(mixed) & IndexMask;
}

uint32_t identity_hash(const Key& key) noexcept {
  Hasher hasher;
  hasher.update(key.id);
  hasher.update(key.subId);
  hasher.update(key.generation);
  hasher.update(key.ordinal);
  hasher.update(key.occurrence);
  return static_cast<uint32_t>(hasher.digest());
}

bool read_check_env(std::string_view text) noexcept {
  if (text == "1") {
    g_checkFirst = 0;
    g_checkLast = UINT64_MAX;
    return true;
  }
  if (!text.starts_with("1,")) {
    return false;
  }
  const std::string_view range = text.substr(2);
  const size_t comma = range.find(',');
  if (comma == std::string_view::npos) {
    return false;
  }
  uint64_t first = 0;
  uint64_t last = 0;
  const auto a = std::from_chars(range.data(), range.data() + comma, first);
  const auto b = std::from_chars(range.data() + comma + 1, range.data() + range.size(), last);
  if (a.ec != std::errc{} || a.ptr != range.data() + comma || b.ec != std::errc{} ||
      b.ptr != range.data() + range.size() || first > last) {
    return false;
  }
  g_checkFirst = first;
  g_checkLast = last;
  return true;
}

void read_env() noexcept {
  if (std::exchange(g_envRead, true)) {
    return;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_RETAINED")) {
    const std::string_view text{value};
    if (text == "1") {
      g_demand |= static_cast<uint8_t>(Demand::Env);
      Log.warn("Retained scene: on by AURORA_RENDER_RETAINED=1");
    } else if (text != "0") {
      Log.warn("Retained scene: AURORA_RENDER_RETAINED={} is not 0 or 1; ignored", text);
    }
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_RETAINED_CHECK")) {
    const std::string_view text{value};
    if (read_check_env(text)) {
      g_checkMode = true;
      g_demand |= static_cast<uint8_t>(Demand::Check);
      Log.warn("Retained scene: check mode on by AURORA_RENDER_RETAINED_CHECK={}", text);
    } else {
      Log.warn("Retained scene: AURORA_RENDER_RETAINED_CHECK={} is not 1 or 1,<first>,<last>; ignored", text);
    }
  }
}

void clear_tables() noexcept {
  for (Slot& slot : g_slots) {
    const uint32_t generation = slot.generation;
    slot = {};
    slot.generation = generation;
  }
  std::fill(g_lives.begin(), g_lives.end(), gc::Life{});
  g_liveList.clear();
  g_evictions.clear();
  gc::reset();
  std::fill(g_index.begin(), g_index.end(), 0u);
  g_free.clear();
  for (uint32_t i = MaxSlots; i-- > 0;) {
    g_free.push_back(i);
  }
  std::fill(g_objects.begin(), g_objects.end(), ObjectEntry{});
  std::fill(g_models.begin(), g_models.end(), ModelEntry{});
  std::fill(g_arrayHashes.begin(), g_arrayHashes.end(), ArrayHash{});
  g_objectCount = 0;
  g_modelCount = 0;
  g_lastObject = NoIndex;
  g_live = 0;
  g_liveStatics = 0;
  g_undeliveredStatics = 0;
  g_staticObjects = 0;
  g_lifecycleChanged = true;
  g_log.clear();
  g_matrices.clear();
  g_layouts.clear();
  g_events.clear();
  g_resync = false;
  g_pending = {};
  g_tentative.reset();
}

void allocate() {
  if (g_allocated) {
    return;
  }
  g_slots.assign(MaxSlots, Slot{});
  g_lives.assign(MaxSlots, gc::Life{});
  g_liveList.reserve(MaxSlots);
  g_evictions.reserve(MaxSlots);
  gc::allocate();
  g_index.assign(IndexSlots, 0u);
  g_free.reserve(MaxSlots);
  g_objects.assign(ObjectSlots, ObjectEntry{});
  g_models.assign(ModelSlots, ModelEntry{});
  g_arrayHashes.assign(ArrayHashSlots, ArrayHash{});
  g_log.reserve(MaxEntries);
  g_matrices.reserve(MaxMatrices);
  g_layouts.reserve(MaxLayouts);
  g_events.reserve(MaxPendingEvents);
  g_allocated = true;
  clear_tables();
}

void release() noexcept {
  gx::fifo::set_release_listener(nullptr);
  g_slots = {};
  g_lives = {};
  g_liveList = {};
  g_evictions = {};
  gc::release();
  g_index = {};
  g_free = {};
  g_objects = {};
  g_models = {};
  g_arrayHashes = {};
  g_log = {};
  g_matrices = {};
  g_layouts = {};
  g_events = {};
  g_pool = {};
  g_tentative.reset();
  g_pending = {};
  g_allocated = false;
  g_live = 0;
  g_liveStatics = 0;
  g_undeliveredStatics = 0;
  g_staticObjects = 0;
  g_objectCount = 0;
  g_modelCount = 0;
  g_lastObject = NoIndex;
}

bool frame_overrun() noexcept { return g_overrun >= 0 ? g_overrun != 0 : gfx::frame_overrun(); }

bool check_active() noexcept {
  return g_checkMode && g_frame.hintFrame >= g_checkFirst && g_frame.hintFrame <= g_checkLast;
}

bool retained_stage(const detail::StageData& stage) noexcept {
  return stage.stage == Stage::OpaqueDone || stage.stage == Stage::PrePost;
}

const Mat3x4<float>& position_matrix(uint32_t slot) noexcept {
  return slot < gx::MaxPnMtx ? gx::g_gxState.pnMtx[slot].pos : gx::g_gxState.texMtxs[slot - gx::MaxPnMtx];
}

bool indexed_attr(uint32_t attr) noexcept {
  const GXAttrType type = gx::g_gxState.vtxDesc[attr];
  return type == GX_INDEX8 || type == GX_INDEX16;
}

bool gpu_handle_draw() noexcept {
  for (uint32_t mask = gx::g_gxState.gpuArrayMask; mask != 0; mask &= mask - 1) {
    if (indexed_attr(GX_VA_POS + std::countr_zero(mask))) {
      return true;
    }
  }
  return false;
}

bool candidate(const DrawClass& cls, const gx::draw_route::Query& query) noexcept {
  if (!cls.mainClass || cls.offscreen || (cls.key.categories & category::NoCapture) != 0) {
    return false;
  }
  if (cls.kind == Kind::MainBlended) {
    if (!cls.key.object.valid || (cls.key.categories & category::Particle) != 0) {
      return false;
    }
  } else if (cls.kind != Kind::MainOpaque) {
    return false;
  }
  if (query.prim == GX_LINES || query.prim == GX_LINESTRIP || query.prim == GX_POINTS) {
    return false;
  }
  const bool palette = gx::g_gxState.vtxDesc[GX_VA_PNMTXIDX] != GX_NONE;
  if (!palette && gx::g_gxState.currentPnMtx >= PositionSlots) {
    return false;
  }
  return gx::g_gxState.gpuArrayMask == 0 || !gpu_handle_draw();
}

void set_pending(const DrawClass& cls, const gx::draw_route::Query& query, uint16_t flags) noexcept {
  g_pending.valid = true;
  g_pending.flags = flags;
  g_pending.object = cls.key.object;
  g_pending.categories = cls.key.categories;
  g_pending.objectFlags = cls.key.flags;
  g_pending.kind = cls.kind;
  g_pending.ordinal = cls.ordinal;
  g_pending.prim = query.prim;
  g_pending.fmt = query.fmt;
  g_pending.vtxCount = query.vtxCount;
  g_pending.indexed = query.indexed;
  g_pending.vertices = query.vertices;
  g_pending.indices = query.indexed ? query.indices : std::span<const uint8_t>{};
  const Mat3x4<float>* world = object_world();
  g_pending.hasWorld = world != nullptr;
  if (world != nullptr) {
    g_pending.world = *world;
  }
}

uint32_t class_bits(const Pending& pending, bool palette) noexcept {
  const uint32_t cat = pending.categories;
  const uint32_t flags = pending.objectFlags;
  uint32_t bits = 0;
  bits |= (cat & category::Static) != 0 ? class_bit::Static : 0u;
  bits |= (flags & object_flag::Immobile) != 0 ? class_bit::Immobile : 0u;
  bits |= (cat & category::Dynamic) != 0 ? class_bit::Dynamic : 0u;
  bits |= (flags & object_flag::CpuDeformed) != 0 ? class_bit::CpuDeformed : 0u;
  bits |= palette ? class_bit::Palette : 0u;
  bits |= (flags & object_flag::Billboard) != 0 ? class_bit::Billboard : 0u;
  bits |= (cat & category::Water) != 0 ? class_bit::Water : 0u;
  bits |= (cat & category::Terrain) != 0 ? class_bit::Terrain : 0u;
  bits |= (cat & category::Foliage) != 0 ? class_bit::Foliage : 0u;
  bits |= (cat & category::NoShadowCast) != 0 ? class_bit::NoShadowCast : 0u;
  bits |= (cat & category::NoRt) != 0 ? class_bit::NoRt : 0u;
  bits |= pending.kind == Kind::MainBlended ? class_bit::Blended : 0u;
  bits |= (cat & category::Player) != 0 ? class_bit::Player : 0u;
  return bits;
}

void ensure_view() noexcept {
  if (g_frame.viewReady) {
    return;
  }
  g_frame.viewReady = true;
  g_frame.invView = inverse(to_affine(detail::live_scene_hints().mainCamera.view));
  const auto& m = g_frame.invView.m;
  g_frame.eye = std::sqrt(m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]);
  for (int r = 0; r < 3; ++r) {
    g_frame.transientOrigin[r] = std::round(m[r][3] / TransientGrid) * TransientGrid;
  }
}

uint16_t layout_index(const gx::AttrConfig* attrs, uint8_t stride) noexcept {
  constexpr size_t AttrBytes = sizeof(Layout::attrs);
  if (!g_layouts.empty()) {
    const Layout& last = g_layouts.back();
    if (last.stride == stride && std::memcmp(last.attrs.data(), attrs, AttrBytes) == 0) {
      return static_cast<uint16_t>(g_layouts.size() - 1);
    }
  }
  Hasher hasher;
  hasher.update(attrs, AttrBytes);
  hasher.update(stride);
  const uint64_t key = hasher.digest();
  for (size_t i = g_layouts.size(); i-- > 0;) {
    if (g_layouts[i].key == key) {
      return static_cast<uint16_t>(i);
    }
  }
  if (g_layouts.size() >= MaxLayouts) {
    return UINT16_MAX;
  }
  Layout& layout = g_layouts.emplace_back();
  std::memcpy(layout.attrs.data(), attrs, AttrBytes);
  layout.stride = stride;
  layout.key = key;
  return static_cast<uint16_t>(g_layouts.size() - 1);
}

uint64_t array_hash(const gx::AttrArray& array) noexcept {
  const uint32_t generation = gx::fifo::immutable_generation(array.data, array.size);
  const uint64_t frame = generation != 0 ? Never : g_frame.hintFrame;
  const uint64_t mixed =
      splitmix64(reinterpret_cast<uintptr_t>(array.data) ^ (uint64_t{array.size} << 40) ^ (uint64_t{generation} << 8));
  const uint32_t home = static_cast<uint32_t>(mixed) & ArrayHashMask;
  uint32_t victim = home;
  for (uint32_t probe = 0; probe < ArrayHashProbes; ++probe) {
    ArrayHash& entry = g_arrayHashes[(home + probe) & ArrayHashMask];
    if (entry.data == array.data && entry.size == array.size && entry.generation == generation &&
        entry.frame == frame) {
      return entry.hash;
    }
    const bool stale = entry.data == nullptr || (entry.generation == 0 && entry.frame != g_frame.hintFrame);
    if (stale) {
      victim = (home + probe) & ArrayHashMask;
      break;
    }
  }
  const auto bytes = gx::fifo::array_bytes(array, 0, array.size);
  const uint64_t hash = xxh3_hash_s(bytes.data(), bytes.size());
  g_arrayHashes[victim] = {
      .data = array.data, .size = array.size, .generation = generation, .frame = frame, .hash = hash};
  return hash;
}

uint64_t topology_hash(const Pending& pending, uint64_t layoutKey) noexcept {
  Hasher hasher;
  hasher.update(layoutKey);
  hasher.update(static_cast<uint32_t>(pending.prim));
  hasher.update(pending.vtxCount);
  hasher.update(pending.vertices.data(), pending.vertices.size());
  if (pending.indexed) {
    hasher.update(pending.indices.data(), pending.indices.size());
  }
  return hasher.digest();
}

uint64_t geometry_hash(uint64_t topology, uint32_t classBits) noexcept {
  if ((classBits & class_bit::CpuDeformed) != 0) {
    return 0;
  }
  Hasher hasher;
  hasher.update(topology);
  for (uint32_t attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
    if (!indexed_attr(attr)) {
      continue;
    }
    hasher.update(attr);
    hasher.update(array_hash(gx::g_gxState.arrays[attr]));
  }
  return hasher.digest();
}

uint64_t signature(const Pending& pending) noexcept {
  std::array<uint8_t, 512> bytes{};
  size_t size = 0;
  const auto put = [&](const void* data, size_t length) {
    std::memcpy(bytes.data() + size, data, length);
    size += length;
  };
  const uint32_t head[4] = {static_cast<uint32_t>(pending.prim), static_cast<uint32_t>(pending.fmt), pending.vtxCount,
                            static_cast<uint32_t>(pending.vertices.size())};
  put(head, sizeof(head));
  const size_t edge = std::min<size_t>(16, pending.vertices.size());
  put(pending.vertices.data(), edge);
  put(pending.vertices.data() + pending.vertices.size() - edge, edge);
  for (uint32_t attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
    if (!indexed_attr(attr)) {
      continue;
    }
    const gx::AttrArray& array = gx::g_gxState.arrays[attr];
    const uintptr_t data = reinterpret_cast<uintptr_t>(array.data);
    const uint32_t tail[3] = {attr, array.size, gx::fifo::immutable_generation(array.data, array.size)};
    put(&data, sizeof(data));
    put(tail, sizeof(tail));
  }
  return xxh3_hash_s(bytes.data(), size);
}

bool event_room() noexcept {
  if (g_events.size() < MaxPendingEvents) {
    return true;
  }
  size_t kept = 0;
  for (const PendingEvent& event : g_events) {
    if (event.type == CancelledEvent) {
      continue;
    }
    if (event.type == static_cast<uint8_t>(EventType::Created)) {
      g_slots[event.slot].createdEvent = static_cast<uint32_t>(kept);
    }
    g_events[kept++] = event;
  }
  g_events.resize(kept);
  if (kept < MaxPendingEvents) {
    return true;
  }
  g_events.clear();
  g_resync = true;
  ++g_frame.counts.resyncs;
  ++g_totals.resyncs;
  return false;
}

void queue_event(EventType type, uint32_t slot, EvictReason reason = EvictReason::None) noexcept {
  if (g_resync) {
    return;
  }
  Slot& entry = g_slots[slot];
  if ((type != EventType::Created && entry.createdPending) || !event_room()) {
    return;
  }
  if (type == EventType::Created) {
    entry.createdEvent = static_cast<uint32_t>(g_events.size());
  }
  g_events.push_back({.type = static_cast<uint8_t>(type),
                      .reason = static_cast<uint8_t>(reason),
                      .slot = static_cast<uint16_t>(slot),
                      .generation = entry.generation});
}

uint32_t index_find(const Key& key) noexcept {
  for (uint32_t pos = key_home(key);; pos = (pos + 1) & IndexMask) {
    const uint32_t value = g_index[pos];
    if (value == 0) {
      return NoIndex;
    }
    if (g_slots[value - 1].info.key == key) {
      return pos;
    }
  }
}

void index_insert(const Key& key, uint32_t slot) noexcept {
  uint32_t pos = key_home(key);
  while (g_index[pos] != 0) {
    pos = (pos + 1) & IndexMask;
  }
  g_index[pos] = slot + 1;
}

void index_erase(uint32_t pos) noexcept {
  g_index[pos] = 0;
  uint32_t hole = pos;
  for (uint32_t next = (pos + 1) & IndexMask; g_index[next] != 0; next = (next + 1) & IndexMask) {
    const uint32_t home = key_home(g_slots[g_index[next] - 1].info.key);
    const bool movable = hole <= next ? (home <= hole || home > next) : (home <= hole && home > next);
    if (movable) {
      g_index[hole] = g_index[next];
      g_index[next] = 0;
      hole = next;
    }
  }
}

uint32_t object_lookup(const ObjectRef& ref, bool& created) noexcept {
  created = false;
  if (g_lastObject != NoIndex) {
    const ObjectEntry& last = g_objects[g_lastObject];
    if (last.id == ref.id && last.subId == ref.subId && last.generation == ref.generation) {
      return g_lastObject;
    }
  }
  const uint64_t mixed = splitmix64(ref.id ^ (uint64_t{ref.subId} << 32) ^ (uint64_t{ref.generation} << 16));
  for (uint32_t pos = static_cast<uint32_t>(mixed) & ObjectMask;; pos = (pos + 1) & ObjectMask) {
    ObjectEntry& entry = g_objects[pos];
    if (!entry.used) {
      if (g_objectCount >= MaxObjects) {
        return NoIndex;
      }
      entry = {.id = ref.id, .subId = ref.subId, .generation = ref.generation, .used = true};
      ++g_objectCount;
      created = true;
      g_lastObject = pos;
      return pos;
    }
    if (entry.id == ref.id && entry.subId == ref.subId && entry.generation == ref.generation) {
      g_lastObject = pos;
      return pos;
    }
  }
}

bool model_known(const ObjectRef& ref) noexcept {
  const uint64_t mixed = splitmix64(ref.id ^ (uint64_t{ref.generation} << 24));
  for (uint32_t pos = static_cast<uint32_t>(mixed) & ModelMask;; pos = (pos + 1) & ModelMask) {
    ModelEntry& entry = g_models[pos];
    if (!entry.used) {
      if (g_modelCount < MaxModels) {
        entry = {.id = ref.id, .generation = ref.generation, .used = true};
        ++g_modelCount;
      }
      return false;
    }
    if (entry.id == ref.id && entry.generation == ref.generation) {
      return true;
    }
  }
}

bool seen_last_scene_frame(const ObjectEntry& entry) noexcept {
  const uint64_t frame = g_frame.sceneFrame;
  if (frame == 0) {
    return false;
  }
  return entry.last == frame - 1 || (entry.last == frame && entry.prev == frame - 1);
}

void touch_object(uint32_t object) noexcept {
  if (object == NoIndex) {
    return;
  }
  ObjectEntry& entry = g_objects[object];
  if (entry.last != g_frame.sceneFrame) {
    entry.prev = entry.last;
    entry.last = g_frame.sceneFrame;
  }
}

std::array<double, 3> translation(const Affine& a) noexcept { return {a.m[0][3], a.m[1][3], a.m[2][3]}; }

double distance(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool latchable(uint32_t classBits) noexcept {
  return (classBits & class_bit::Immobile) != 0 && (classBits & (class_bit::Billboard | class_bit::Palette)) == 0;
}

bool rigid_dynamic(uint32_t classBits) noexcept { return (classBits & (StaticBits | class_bit::Palette)) == 0; }

// Rebound at every sighting, so a release earlier in the frame cannot mark an array declared again after it
void bind_pos(gc::Life& life) noexcept {
  life.posBegin = indexed_attr(GX_VA_POS) ? reinterpret_cast<uintptr_t>(gx::g_gxState.arrays[GX_VA_POS].data) : 0;
  life.posSeq = gc::release_seq();
}

uint32_t create(const Pending& pending, const Key& key, uint32_t classBits, uint64_t layoutKey,
                const Affine& world) noexcept {
  FrameCounts& counts = g_frame.counts;
  if (g_live >= MaxLive || g_free.empty()) {
    ++counts.overflow;
    if (!std::exchange(g_warnedOverflow, true)) {
      Log.warn("Retained scene: {} live instances; further draws are not retained", g_live);
    }
    return TransientSlot;
  }
  bool objectCreated = false;
  const uint32_t object = object_lookup(pending.object, objectCreated);
  const gc::Recreation prior = gc::recreated(key, g_frame.sceneFrame);
  // A key the collector evicted was seen before, so its return is a re-creation and never a new key
  if (prior.reason != EvictReason::None) {
    if (prior.age <= gc::RecentFrames) {
      counts.recreatedStatic += prior.reason == EvictReason::Epoch ? 1 : 0;
      counts.recreatedDynamic += prior.reason == EvictReason::Dynamic ? 1 : 0;
      g_totals.churn += prior.reason == EvictReason::Epoch ? 1 : 0;
    }
  } else if (key.occurrence != 0) {
    ++counts.newOccurrence;
  } else if (object != NoIndex && !objectCreated && seen_last_scene_frame(g_objects[object])) {
    ++counts.rekeyed;
  }
  if (model_known(pending.object) && objectCreated) {
    ++counts.newSubId;
  }
  touch_object(object);

  const uint32_t index = g_free.back();
  g_free.pop_back();
  Slot& slot = g_slots[index];
  const uint32_t generation = slot.generation + 1;
  slot = {};
  slot.generation = generation;
  slot.live = true;
  slot.object = object;
  slot.seenHintFrame = g_frame.hintFrame;
  slot.livePos = static_cast<uint32_t>(g_liveList.size());
  g_liveList.push_back(index);
  gc::Life& life = g_lives[index];
  life = {.lastSeen = g_frame.sceneFrame, .statics = (classBits & StaticBits) != 0};
  bind_pos(life);
  const uint64_t topology = topology_hash(pending, layoutKey);
  slot.info = {
      .slot = index,
      .key = key,
      .classBits = classBits,
      .categories = pending.categories,
      .flags = pending.objectFlags,
      .topologyHash = topology,
      .geometryHash = geometry_hash(topology, classBits),
      .origin = pending.hasWorld ? std::array<double, 3>{pending.world.m0[3], pending.world.m1[3], pending.world.m2[3]}
                                 : translation(world),
  };
  if ((classBits & StaticBits) != 0) {
    slot.signature = signature(pending);
  }
  if (latchable(classBits)) {
    slot.world = world;
    slot.latched = true;
  }
  index_insert(key, index);
  ++g_live;
  g_peakLive = std::max(g_peakLive, g_live);
  if ((classBits & StaticBits) != 0) {
    g_frame.staticCreated = true;
    ++g_liveStatics;
    ++g_undeliveredStatics;
    if (object != NoIndex && g_objects[object].statics++ == 0) {
      ++g_staticObjects;
    }
  }
  slot.createdPending = true;
  g_newGeometry = true;
  queue_event(EventType::Created, index);
  ++counts.created;
  ++g_totals.created;
  g_lifecycleChanged = true;
  return index;
}

void check_static(Slot& slot, const Pending& pending, uint64_t layoutKey) noexcept {
  FrameCounts& counts = g_frame.counts;
  const uint64_t sig = signature(pending);
  const bool check = check_active();
  if (sig == slot.signature && !check) {
    return;
  }
  const uint64_t topology = topology_hash(pending, layoutKey);
  const uint64_t geometry = geometry_hash(topology, slot.info.classBits);
  if (topology != slot.info.topologyHash || geometry != slot.info.geometryHash) {
    if (sig == slot.signature) {
      ++counts.signatureMisses;
    }
    ++counts.staticRestream;
    ++g_totals.churn;
    slot.info.topologyHash = topology;
    slot.info.geometryHash = geometry;
    g_newGeometry = true;
    queue_event(EventType::GeometryChanged, slot.info.slot);
  }
  slot.signature = sig;
}

uint32_t find_or_create(const Pending& pending, uint32_t classBits, uint64_t layoutKey, const Affine& world) noexcept {
  Key key{.id = pending.object.id,
          .subId = pending.object.subId,
          .generation = pending.object.generation,
          .ordinal = pending.ordinal};
  for (;;) {
    const uint32_t pos = index_find(key);
    if (pos == NoIndex) {
      break;
    }
    const uint32_t index = g_index[pos] - 1;
    Slot& slot = g_slots[index];
    if (slot.seenHintFrame != g_frame.hintFrame) {
      slot.seenHintFrame = g_frame.hintFrame;
      g_lives[index].lastSeen = g_frame.sceneFrame;
      bind_pos(g_lives[index]);
      touch_object(slot.object);
      if ((slot.info.classBits & StaticBits) != 0) {
        check_static(slot, pending, layoutKey);
      }
      return index;
    }
    if (++key.occurrence >= MaxOccurrence) {
      ++g_frame.counts.overflow;
      return TransientSlot;
    }
  }
  return create(pending, key, classBits, layoutKey, world);
}

Affine instance_world(uint32_t index, const Affine& world, std::array<double, 3>& origin) noexcept {
  Slot& slot = g_slots[index];
  FrameCounts& counts = g_frame.counts;
  if (slot.latched) {
    if (world_moved(slot.world, world, g_frame.eye)) {
      slot.world = world;
      ++counts.relatched;
      ++g_totals.churn;
      queue_event(EventType::Relatched, index);
    }
    origin = slot.info.origin;
    return slot.world;
  }
  if (rigid_dynamic(slot.info.classBits) && distance(translation(world), slot.info.origin) > ReoriginDistance) {
    slot.info.origin = translation(world);
    ++counts.reorigins;
    queue_event(EventType::Reorigin, index);
  }
  origin = slot.info.origin;
  return world;
}

Mat3x4<float> relative(const Affine& world, const std::array<double, 3>& origin) noexcept {
  Mat3x4<float> out{};
  Vec4<float>* rows[3] = {&out.m0, &out.m1, &out.m2};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      (*rows[r])[c] = static_cast<float>(world.m[r][c]);
    }
    (*rows[r])[3] = static_cast<float>(world.m[r][3] - origin[r]);
  }
  return out;
}

void drop(uint16_t flags) noexcept {
  if ((flags & entry_flag::CaptureOnly) != 0) {
    ++g_frame.counts.lostCaptures;
    ++g_totals.lostCaptures;
  }
}

void record(const DrawRef& draw) noexcept {
  const Zone zone;
  const Pending pending = g_pending;
  g_pending.valid = false;
  FrameCounts& counts = g_frame.counts;
  const gx::GXState& state = gx::g_gxState;
  const bool palette = draw.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE;
  const uint32_t matrixCount = palette ? PaletteSlots : 1;
  if (g_log.size() >= MaxEntries || g_matrices.size() + matrixCount > MaxMatrices) {
    ++counts.droppedEntries;
    drop(pending.flags);
    if (!std::exchange(g_warnedEntries, true)) {
      Log.warn("Retained scene: more than {} entries in a frame; the rest are dropped", MaxEntries);
    }
    return;
  }
  const uint16_t layout = layout_index(draw.attrs, draw.stride);
  if (layout == UINT16_MAX) {
    ++counts.droppedEntries;
    drop(pending.flags);
    return;
  }
  ensure_view();
  const uint32_t pnSlot = draw.immediates.currentPnMtx;
  Affine world = multiply(g_frame.invView, to_affine(position_matrix(pnSlot)));
  const uint32_t classBits = class_bits(pending, palette);
  uint32_t slot = TransientSlot;
  g_newGeometry = false;
  if (pending.object.valid) {
    slot = find_or_create(pending, classBits, g_layouts[layout].key, world);
  }
  std::array<double, 3> origin = g_frame.transientOrigin;
  if (slot != TransientSlot) {
    Slot& seen = g_slots[slot];
    seen.captureOnlySeen = seen.captureOnlySeen || (pending.flags & entry_flag::CaptureOnly) != 0;
    seen.visibleSeen = seen.visibleSeen || (pending.flags & entry_flag::CaptureOnly) == 0;
    world = instance_world(slot, world, origin);
    if ((pending.objectFlags & object_flag::Teleport) != 0) {
      queue_event(EventType::Teleported, slot);
    }
  }

  uint16_t flags = pending.flags;
  flags |= palette ? entry_flag::Palette : 0;
  flags |= pending.indexed ? entry_flag::Indexed : 0;
  flags |= g_frame.opaquePrepared ? entry_flag::Late : 0;
  flags |= slot == TransientSlot ? entry_flag::Transient : 0;
  Entry& entry = g_log.emplace_back();
  entry.slot = slot;
  entry.flags = flags;
  entry.layout = layout;
  entry.categories = pending.categories;
  entry.prim = static_cast<uint8_t>(pending.prim);
  entry.matrixCount = static_cast<uint8_t>(matrixCount);
  entry.vtxCount = pending.vtxCount;
  entry.vertRange = draw.vertRange;
  if (pending.indexed) {
    entry.idxRange = draw.idxRange;
    entry.indexCount = draw.indexCount;
  }
  entry.immediates = draw.immediates;
  entry.immediates.vtxStart = draw.vertRange.offset;
  entry.record = draw.record;
  entry.matrix = static_cast<uint32_t>(g_matrices.size());
  entry.worldRel = relative(world, origin);
  if (palette) {
    g_matrices.insert(g_matrices.end(), state.pnMtx.begin(), state.pnMtx.end());
  } else {
    g_matrices.push_back(
        {.pos = position_matrix(pnSlot), .nrm = pnSlot < gx::MaxPnMtx ? state.pnMtx[pnSlot].nrm : Mat3x4<float>{}});
  }
  const Slot* seen = slot != TransientSlot ? &g_slots[slot] : nullptr;
  streamout::on_capture({
      .slot = slot,
      .slotGeneration = seen != nullptr ? seen->generation : 0,
      .identityHash = seen != nullptr ? identity_hash(seen->info.key) : 0,
      .newGeometry = g_newGeometry,
      .latched = seen != nullptr && seen->latched,
      .teleport = (pending.objectFlags & object_flag::Teleport) != 0,
      .gpuArrays = state.gpuArrayMask != 0,
      .entryFlags = flags,
      .classBits = classBits,
      .categories = pending.categories,
      .objectFlags = pending.objectFlags,
      .attrs = draw.attrs,
      .stride = draw.stride,
      .immediates = &entry.immediates,
      .prim = entry.prim,
      .vtxCount = entry.vtxCount,
      .idxRange = entry.idxRange,
      .indexCount = entry.indexCount,
      .pnSlot = pnSlot,
      .world = &world,
      .invView = &g_frame.invView,
      .eye = g_frame.eye,
  });
  ++counts.entries;
  counts.captureOnly += (flags & entry_flag::CaptureOnly) != 0 ? 1 : 0;
  counts.transient += slot == TransientSlot ? 1 : 0;
  counts.late += (flags & entry_flag::Late) != 0 ? 1 : 0;
}

void drop_log() noexcept {
  streamout::drop_pending();
  for (const Entry& entry : g_log) {
    drop(entry.flags);
  }
  g_log.clear();
  g_matrices.clear();
  g_layouts.clear();
}

std::shared_ptr<Frame> take_frame() {
  for (auto& frame : g_pool) {
    if (!frame) {
      frame = std::make_shared<Frame>();
      return frame;
    }
    if (frame.use_count() == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      return frame;
    }
  }
  return std::make_shared<Frame>();
}

void build_events(std::vector<Event>& events) {
  events.clear();
  if (g_resetPending || g_resync) {
    events.push_back({.type = EventType::Reset, .slot = TransientSlot});
  }
  if (g_resync) {
    for (const Slot& slot : g_slots) {
      if (slot.live) {
        events.push_back(
            {.type = EventType::Created, .slot = slot.info.slot, .slotGeneration = slot.generation, .info = slot.info});
      }
    }
    return;
  }
  for (const PendingEvent& pending : g_events) {
    if (pending.type == CancelledEvent) {
      continue;
    }
    const auto type = static_cast<EventType>(pending.type);
    Event& event = events.emplace_back(Event{.type = type,
                                             .reason = static_cast<EvictReason>(pending.reason),
                                             .slot = pending.slot == ResetSlot ? TransientSlot : pending.slot,
                                             .slotGeneration = pending.generation});
    if (type == EventType::Created) {
      event.info = g_slots[pending.slot].info;
    }
  }
}

void commit_events() noexcept {
  if (g_resync) {
    for (Slot& slot : g_slots) {
      slot.createdPending = false;
      slot.createdEvent = NoIndex;
    }
  } else {
    for (const PendingEvent& pending : g_events) {
      if (pending.type == static_cast<uint8_t>(EventType::Created)) {
        Slot& slot = g_slots[pending.slot];
        slot.createdPending = false;
        slot.createdEvent = NoIndex;
      }
    }
  }
  g_events.clear();
  g_resync = false;
  g_resetPending = false;
}

void evict_slot(uint32_t index, EvictReason reason) noexcept {
  Slot& slot = g_slots[index];
  if (!slot.live) {
    return;
  }
  if (!g_resync) {
    if (slot.createdPending && slot.createdEvent != NoIndex) {
      g_events[slot.createdEvent].type = CancelledEvent;
    } else if (!slot.createdPending && event_room()) {
      g_events.push_back({.type = static_cast<uint8_t>(EventType::Evicted),
                          .reason = static_cast<uint8_t>(reason),
                          .slot = static_cast<uint16_t>(index),
                          .generation = slot.generation});
    }
  }
  if (const uint32_t pos = index_find(slot.info.key); pos != NoIndex) {
    index_erase(pos);
  }
  const uint32_t moved = g_liveList.back();
  g_liveList[slot.livePos] = moved;
  g_slots[moved].livePos = slot.livePos;
  g_liveList.pop_back();
  FrameCounts& counts = g_frame.counts;
  counts.evictedDynamic += reason == EvictReason::Dynamic ? 1 : 0;
  counts.evictedStatic += reason == EvictReason::Epoch ? 1 : 0;
  counts.evictedReleased += reason == EvictReason::Released ? 1 : 0;
  gc::remember(slot.info.key, reason, g_sceneClock);
  --g_live;
  if ((slot.info.classBits & StaticBits) != 0) {
    --g_liveStatics;
    if (!slot.delivered) {
      --g_undeliveredStatics;
    }
    if (slot.object != NoIndex && --g_objects[slot.object].statics == 0) {
      --g_staticObjects;
    }
  }
  const uint32_t generation = slot.generation;
  slot = {};
  slot.generation = generation;
  g_lives[index] = {};
  streamout::on_freed(index);
  // Frames sealed before the eviction may still name the slot on the worker
  gc::quarantine(index);
  ++g_totals.evicted;
  g_lifecycleChanged = true;
}

bool collect(bool scene) noexcept {
  const detail::SceneHints& hints = detail::live_scene_hints();
  g_evictions.clear();
  const gc::SweepResult result = gc::sweep({.sceneFrame = g_sceneClock,
                                            .scene = scene,
                                            .epochKnown = hints.has(detail::SceneHasEpoch),
                                            .epoch = hints.sceneEpoch},
                                           g_liveList, g_lives, g_evictions);
  g_frame.counts.releaseOverflow += result.releaseOverflow;
  for (const gc::Eviction& eviction : g_evictions) {
    evict_slot(eviction.slot, eviction.reason);
  }
  gc::free_quarantined(g_free);
  return result.epochChanged;
}

bool settled(bool epochChanged) noexcept {
  const FrameCounts& counts = g_frame.counts;
  if (g_frame.staticCreated || epochChanged || counts.lostCaptures != 0 || counts.resyncs != 0 ||
      counts.overflow != 0 || counts.droppedEntries != 0) {
    g_lastDisturb = g_sceneClock;
  }
  // aurora cannot know about geometry it was never sent, so only a producer that stopped holding can settle
  if (!g_producer || g_held || g_undeliveredStatics != 0 || g_sceneClock - g_lastDisturb < QuietFrames) {
    return false;
  }
  for (uint32_t i = 0; i < g_settleCount; ++i) {
    if (!g_settle[i]()) {
      return false;
    }
  }
  return true;
}

void publish() {
  FrameCounts counts = g_frame.counts;
  counts.hintFrame = g_frame.hintFrame;
  counts.sceneFrame = g_frame.sceneFrame;
  counts.epoch = detail::live_scene_hints().sceneEpoch;
  counts.active = g_active;
  counts.instances = g_live;
  counts.statics = g_liveStatics;
  counts.undeliveredStatics = g_undeliveredStatics;
  counts.eventsPending = static_cast<uint32_t>(g_events.size());
  counts.pendingSweeps = g_active ? gc::pending_sweeps() : 0;
  counts.coverageComplete = g_coverage;
  const RetainedStats stats{
      .active = g_active,
      .coverageComplete = g_coverage,
      .instances = g_live,
      .staticInstances = g_liveStatics,
      .staticObjects = g_staticObjects,
      .undeliveredStatics = g_undeliveredStatics,
      .created = g_totals.created,
      .evicted = g_totals.evicted,
      .captured = g_totals.captured,
      .churn = g_totals.churn,
      .lostCaptures = g_totals.lostCaptures,
      .resyncs = g_totals.resyncs,
      .capturedOnly = g_totals.capturedOnly,
  };
  std::shared_ptr<const std::vector<uint64_t>> covered;
  const bool rebuild = g_wantCovered && g_active && std::exchange(g_lifecycleChanged, false);
  if (rebuild) {
    auto ids = std::make_shared<std::vector<uint64_t>>();
    for (const Slot& slot : g_slots) {
      if (slot.live && slot.delivered) {
        ids->push_back(slot.info.key.id);
      }
    }
    std::sort(ids->begin(), ids->end());
    ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
    covered = std::move(ids);
  }
  std::lock_guard lock{g_publishMutex};
  g_published = stats;
  g_publishedCounts = counts;
  if (rebuild) {
    g_covered = std::move(covered);
  } else if (!g_wantCovered || !g_active) {
    g_covered.reset();
  }
}
} // namespace

void latch_gate(const Settings&, const Caps& caps) noexcept {
  read_env();
  const bool next = detail::hints_enabled() && caps.core && streamout::supported(caps) && g_demand != 0;
  if (next != g_active) {
    streamout::reset();
    if (next) {
      allocate();
      g_resetPending = true;
      Log.info("Retained scene: active");
    } else {
      release();
      Log.info("Retained scene: inactive");
    }
    g_active = next;
    g_frame = {};
    g_lastDisturb = g_sceneClock;
  }
  g_producer = g_producerCalled;
  g_held = g_holdCalled;
  g_captureActive.store(next, std::memory_order_relaxed);
  latch_resources(next);
}

void latch_resources(bool active) noexcept { streamout::latch_resources(active); }

void set_demand(Demand demand, bool on) noexcept {
  const auto bit = static_cast<uint8_t>(demand);
  g_demand = on ? static_cast<uint8_t>(g_demand | bit) : static_cast<uint8_t>(g_demand & ~bit);
}

void apply_frame_state(const detail::FrameState&) noexcept {
  gx::fifo::set_release_listener(g_active ? &gc::on_release : nullptr);
  const uint64_t hintFrame = detail::live_scene_hints().hintFrame;
  const bool resynced = streamout::begin_frame(g_active, hintFrame);
  if (!g_active) {
    return;
  }
  g_frame = {};
  g_frame.hintFrame = hintFrame;
  g_frame.sceneFrame = g_sceneClock;
  g_pending = {};
  if (resynced) {
    ++g_frame.counts.resyncs;
    ++g_totals.resyncs;
  }
}

void end_frame() noexcept {
  const bool scene = detail::live_scene_hints().has(detail::SceneHasMainCamera);
  const uint64_t reported = g_reportedLost.exchange(0, std::memory_order_relaxed);
  g_coverage = false;
  if (g_active) {
    if (!g_log.empty()) {
      drop_log();
    }
    g_pending = {};
    g_tentative.reset();
    streamout::end_frame();
    const bool epochChanged = collect(scene);
    g_frame.counts.lostCaptures += static_cast<uint32_t>(reported);
    g_totals.lostCaptures += reported;
    g_coverage = settled(epochChanged);
  }
  g_coverageComplete.store(g_coverage, std::memory_order_relaxed);
  publish();
  if (g_active && scene) {
    ++g_sceneClock;
  }
  const uint64_t hintFrame = g_frame.hintFrame;
  g_frame = {};
  g_frame.hintFrame = hintFrame;
  g_frame.sceneFrame = g_sceneClock;
}

void reset() noexcept {
  if (g_allocated) {
    clear_tables();
  }
  streamout::reset();
  g_resetPending = g_active;
  g_sceneClock = 0;
  g_lastDisturb = 0;
  g_coverage = false;
  g_coverageComplete.store(false, std::memory_order_relaxed);
  g_totals = {};
  g_peakLive = 0;
  g_frame = {};
  g_warnedOverflow = false;
  g_warnedEntries = false;
  std::lock_guard lock{g_publishMutex};
  g_published = {.active = g_active};
  g_publishedCounts = {};
  g_covered.reset();
}

void want_covered_objects(bool want) noexcept {
  g_wantCovered = want;
  g_lifecycleChanged = true;
}

void hold_coverage(bool hold) noexcept {
  g_producerCalled = true;
  g_holdCalled = hold;
}

void add_settle_predicate(SettlePredicate predicate) noexcept {
  for (uint32_t i = 0; i < g_settleCount; ++i) {
    if (g_settle[i] == predicate) {
      return;
    }
  }
  if (predicate != nullptr && g_settleCount < MaxSettlePredicates) {
    g_settle[g_settleCount++] = predicate;
  }
}

bool active() noexcept { return g_active; }

bool check_window(uint64_t& first, uint64_t& last) noexcept {
  first = g_checkFirst;
  last = g_checkLast;
  return g_checkMode;
}

void report_lost(uint32_t count) noexcept { g_reportedLost.fetch_add(count, std::memory_order_relaxed); }

bool capture_active() noexcept { return g_captureActive.load(std::memory_order_relaxed); }

bool coverage_complete() noexcept { return g_coverageComplete.load(std::memory_order_relaxed); }

bool captures(const DrawClass& cls, const gx::draw_route::Query& query) noexcept {
  return g_active && candidate(cls, query);
}

void route(const DrawClass& cls, const gx::draw_route::Query& query) noexcept {
  if (!g_active) {
    return;
  }
  g_pending.valid = false;
  if (!candidate(cls, query)) {
    return;
  }
  const Zone zone;
  set_pending(cls, query, cls.suppressed ? entry_flag::CaptureOnly : 0);
}

void on_push(const gx::DrawData& draw, const gx::PipelineConfig& config) noexcept {
  if (!g_active || !g_pending.valid) {
    return;
  }
  record({
      .vertRange = draw.vertRange,
      .idxRange = draw.idxRange,
      .indexCount = draw.indexCount,
      .immediates = draw.immediateData,
      .record = draw.immediateData._pad,
      .attrs = config.shaderConfig.attrs.data(),
      .stride = config.shaderConfig.vtxStride,
  });
}

void on_merge(const gx::draw_route::MergedDraw& draw) noexcept {
  if (!g_active || !g_pending.valid) {
    return;
  }
  record({
      .vertRange = {draw.vertOffset, draw.vertSize},
      .immediates = draw.command->immediateData,
      .record = draw.command->immediateData._pad,
      .attrs = draw.config->shaderConfig.attrs.data(),
      .stride = draw.config->shaderConfig.vtxStride,
  });
}

void on_capture(const gx::draw_route::CapturedDraw& draw) noexcept {
  if (!g_active || !g_pending.valid) {
    return;
  }
  record({
      .vertRange = {draw.vertOffset, draw.vertSize},
      .idxRange = {draw.idxOffset, draw.idxSize},
      .indexCount = draw.indexCount,
      .immediates = *draw.immediates,
      .attrs = draw.attrs,
      .stride = draw.vtxStride,
  });
}

void prepare(detail::StageData& stage) noexcept {
  if (!g_active || !retained_stage(stage)) {
    return;
  }
  const Zone zone;
  if (stage.stage == Stage::OpaqueDone) {
    g_frame.opaquePrepared = true;
  }
  g_tentative.reset();
  if (frame_overrun()) {
    drop_log();
    return;
  }
  ensure_view();
  std::shared_ptr<Frame> frame = take_frame();
  const detail::SceneHints& hints = detail::live_scene_hints();
  frame->stage = stage.stage;
  frame->hintFrame = g_frame.hintFrame;
  frame->sceneFrame = g_frame.sceneFrame;
  frame->epoch = hints.sceneEpoch;
  frame->invView = g_frame.invView;
  frame->transientOrigin = g_frame.transientOrigin;
  frame->entries.assign(g_log.begin(), g_log.end());
  frame->matrices.assign(g_matrices.begin(), g_matrices.end());
  frame->layouts.assign(g_layouts.begin(), g_layouts.end());
  build_events(frame->events);
  frame->plan.reset();
  streamout::seal(*frame, stage);
  g_tentative = frame;
  stage.retained = std::move(frame);
}

void skip(detail::StageData& stage) noexcept {
  if (!g_active || !retained_stage(stage)) {
    return;
  }
  const Zone zone;
  streamout::skip(stage);
  stage.retained.reset();
  g_tentative.reset();
  if (stage.stage == Stage::PrePost) {
    drop_log();
  }
}

void seal(detail::StageData& stage) noexcept {
  if (!g_active || !retained_stage(stage) || stage.retained == nullptr || g_tentative == nullptr) {
    return;
  }
  const Zone zone;
  if (frame_overrun()) {
    stage.retained.reset();
    g_tentative.reset();
    drop_log();
    return;
  }
  for (const Entry& entry : g_tentative->entries) {
    if (entry.slot == TransientSlot) {
      continue;
    }
    Slot& slot = g_slots[entry.slot];
    if (slot.live && !std::exchange(slot.delivered, true)) {
      g_lifecycleChanged = true;
      if ((slot.info.classBits & StaticBits) != 0) {
        --g_undeliveredStatics;
      }
    }
  }
  const auto delivered = static_cast<uint32_t>(g_tentative->entries.size());
  g_frame.counts.delivered += delivered;
  g_totals.captured += delivered;
  g_totals.capturedOnly += static_cast<uint64_t>(std::ranges::count_if(
      g_tentative->entries, [](const Entry& entry) { return (entry.flags & entry_flag::CaptureOnly) != 0; }));
  commit_events();
  streamout::sealed(stage);
  g_log.clear();
  g_matrices.clear();
  g_layouts.clear();
  g_tentative.reset();
}

RetainedStats stats() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_published;
}

FrameCounts frame_counts() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_publishedCounts;
}

std::shared_ptr<const std::vector<uint64_t>> covered_objects() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_covered;
}

namespace testing {
void reset() noexcept {
  release();
  g_active = false;
  g_envRead = true;
  g_demand = 0;
  g_checkMode = false;
  g_checkFirst = 0;
  g_checkLast = UINT64_MAX;
  g_overrun = -1;
  g_wantCovered = false;
  g_producerCalled = false;
  g_holdCalled = true;
  g_producer = false;
  g_held = true;
  g_settle = {};
  g_settleCount = 0;
  g_reportedLost.store(0, std::memory_order_relaxed);
  g_captureActive.store(false, std::memory_order_relaxed);
  streamout::testing::reset();
  retained::reset();
}

void set_check(bool on, uint64_t first, uint64_t last) noexcept {
  g_checkMode = on;
  g_checkFirst = first;
  g_checkLast = last;
}

void set_overrun(int overrun) noexcept { g_overrun = overrun; }

Instance instance(uint32_t slot) noexcept {
  if (!g_allocated || slot >= MaxSlots) {
    return {};
  }
  const Slot& entry = g_slots[slot];
  return {.live = entry.live,
          .delivered = entry.delivered,
          .latched = entry.latched,
          .released = g_lives[slot].released,
          .captureOnlySeen = entry.captureOnlySeen,
          .visibleSeen = entry.visibleSeen,
          .lastSeen = g_lives[slot].lastSeen,
          .slotGeneration = entry.generation,
          .info = entry.info,
          .world = entry.world};
}

uint32_t find(const Key& key) noexcept {
  if (!g_allocated) {
    return TransientSlot;
  }
  const uint32_t pos = index_find(key);
  return pos == NoIndex ? TransientSlot : g_index[pos] - 1;
}

void evict(uint32_t slot, EvictReason reason) noexcept {
  if (g_allocated && slot < MaxSlots) {
    evict_slot(slot, reason);
  }
}
} // namespace testing
} // namespace aurora::render::scene::retained

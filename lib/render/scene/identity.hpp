#pragma once

#include "affine.hpp"

#include "../../gx/gx.hpp"

#include <aurora/render.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::gx::draw_route {
struct Query;
struct MergedDraw;
struct CapturedDraw;
} // namespace aurora::gx::draw_route

namespace aurora::render {
struct Caps;
struct Settings;
} // namespace aurora::render

namespace aurora::render::detail {
struct FrameState;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::streamout {
struct Plan;
} // namespace aurora::render::scene::streamout

namespace aurora::render::scene::atlas {
struct Snapshot;
} // namespace aurora::render::scene::atlas

namespace aurora::render::scene::retained {
inline constexpr uint32_t MaxSlots = 16384;
inline constexpr uint32_t IndexSlots = 32768;
inline constexpr uint32_t MaxLive = 12288;
inline constexpr uint32_t MaxEntries = 32768;
inline constexpr uint32_t MaxPendingEvents = 24576;
inline constexpr uint32_t MaxOccurrence = 8;
inline constexpr uint32_t PaletteSlots = gx::MaxPnMtx;
inline constexpr uint32_t TransientSlot = 0xFFFFFFFF;
inline constexpr uint16_t NoGroup = 0xFFFF;
inline constexpr double ReoriginDistance = 2048.0;
inline constexpr double TransientGrid = 64.0;
inline constexpr uint64_t QuietFrames = 8;
inline constexpr uint32_t MaxSettlePredicates = 8;

enum class Demand : uint8_t {
  Env = 1u << 0,
  Tool = 1u << 1,
  View = 1u << 2,
  Check = 1u << 3,
  Consumer = 1u << 4,
};

struct Key {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  uint16_t ordinal = 0;
  uint8_t occurrence = 0;
  bool operator==(const Key&) const = default;
};

namespace class_bit {
inline constexpr uint32_t Static = 1u << 0;
inline constexpr uint32_t Immobile = 1u << 1;
inline constexpr uint32_t Dynamic = 1u << 2;
inline constexpr uint32_t CpuDeformed = 1u << 3;
inline constexpr uint32_t Palette = 1u << 4;
inline constexpr uint32_t Billboard = 1u << 5;
inline constexpr uint32_t Water = 1u << 6;
inline constexpr uint32_t Terrain = 1u << 7;
inline constexpr uint32_t Foliage = 1u << 8;
inline constexpr uint32_t NoShadowCast = 1u << 9;
inline constexpr uint32_t NoRt = 1u << 10;
inline constexpr uint32_t Blended = 1u << 11;
inline constexpr uint32_t Player = 1u << 12;
} // namespace class_bit

namespace entry_flag {
inline constexpr uint16_t CaptureOnly = 1u << 0;
inline constexpr uint16_t Palette = 1u << 1;
inline constexpr uint16_t Indexed = 1u << 2;
inline constexpr uint16_t Late = 1u << 3;
inline constexpr uint16_t Transient = 1u << 4;
} // namespace entry_flag

enum class EventType : uint8_t { Created, Evicted, Relatched, GeometryChanged, Teleported, Reorigin, Reset };
enum class EvictReason : uint8_t { None, Dynamic, Epoch, Released, Reset, Deactivate };

struct InstanceInfo {
  uint32_t slot = TransientSlot;
  Key key;
  uint32_t classBits = 0;
  uint32_t categories = 0;
  uint32_t flags = 0;
  uint64_t topologyHash = 0;
  uint64_t geometryHash = 0;
  std::array<double, 3> origin{};
  uint16_t group = NoGroup;
};

struct Event {
  EventType type = EventType::Reset;
  EvictReason reason = EvictReason::None;
  uint32_t slot = TransientSlot;
  uint32_t slotGeneration = 0;
  InstanceInfo info;
};

struct Layout {
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  uint8_t stride = 0;
  uint64_t key = 0;
};

struct Entry {
  uint32_t slot = TransientSlot;
  uint16_t flags = 0;
  uint16_t layout = 0;
  uint32_t categories = 0;
  uint8_t prim = 0;
  uint8_t matrixCount = 1;
  uint32_t vtxCount = 0;
  gfx::Range vertRange{};
  gfx::Range idxRange{};
  uint32_t indexCount = 0;
  gx::DrawImmediateData immediates{};
  uint32_t record = 0;
  uint32_t matrix = 0;
  Mat3x4<float> worldRel{};
};

struct Frame {
  Stage stage = Stage::OpaqueDone;
  uint64_t hintFrame = 0;
  uint64_t sceneFrame = 0;
  uint32_t epoch = 0;
  Affine invView{};
  std::array<double, 3> transientOrigin{};
  std::vector<Entry> entries;
  std::vector<gx::PnMtx> matrices;
  std::vector<Layout> layouts;
  std::vector<Event> events;
  std::shared_ptr<const streamout::Plan> plan;
  std::shared_ptr<const atlas::Snapshot> atlas;
};

struct FrameCounts {
  uint64_t hintFrame = 0;
  uint64_t sceneFrame = 0;
  uint32_t epoch = 0;
  bool active = false;
  uint32_t entries = 0;
  uint32_t captureOnly = 0;
  uint32_t transient = 0;
  uint32_t late = 0;
  uint32_t instances = 0;
  uint32_t statics = 0;
  uint32_t created = 0;
  uint32_t rekeyed = 0;
  uint32_t newSubId = 0;
  uint32_t newOccurrence = 0;
  uint32_t relatched = 0;
  uint32_t staticRestream = 0;
  uint32_t signatureMisses = 0;
  uint32_t reorigins = 0;
  uint32_t overflow = 0;
  uint32_t droppedEntries = 0;
  uint32_t undeliveredStatics = 0;
  uint32_t lostCaptures = 0;
  uint32_t resyncs = 0;
  uint32_t eventsPending = 0;
  uint32_t delivered = 0;
  uint32_t evictedDynamic = 0;
  uint32_t evictedStatic = 0;
  uint32_t evictedReleased = 0;
  uint32_t recreatedStatic = 0;
  uint32_t recreatedDynamic = 0;
  uint32_t pendingSweeps = 0;
  uint32_t releaseOverflow = 0;
  bool coverageComplete = false;
};

struct FilteredObject {
  uint64_t id = 0;
  uint32_t passing = 0;
  uint32_t failing = 0;
  uint32_t firstSlot = 0;
  uint32_t slotCount = 0;
};

struct FilteredObjects {
  std::vector<FilteredObject> objects;
  std::vector<uint32_t> slots;
};

// Game thread at end_frame; false holds coverage back
using SettlePredicate = bool (*)() noexcept;
using CoverFilter = bool (*)(const InstanceInfo& info) noexcept;

[[nodiscard]] bool available(const Caps& caps) noexcept;
// Game thread while the FIFO is idle
void latch_gate(const Settings& settings, const Caps& caps) noexcept;
void latch_resources(bool active) noexcept;
void set_demand(Demand demand, bool on) noexcept;
void apply_frame_state(const detail::FrameState& state) noexcept;
void end_frame() noexcept;
void reset() noexcept;
void want_covered_objects(bool want) noexcept;
// While set, end_frame also publishes covered_filtered(), each id's delivered instances split by the filter
void set_cover_filter(CoverFilter filter) noexcept;
// The first call registers the caller as the producer; latched with the gate
void hold_coverage(bool hold) noexcept;
void add_settle_predicate(SettlePredicate predicate) noexcept;

[[nodiscard]] bool active() noexcept;
[[nodiscard]] bool check_window(uint64_t& first, uint64_t& last) noexcept;
// Any thread
void report_lost(uint32_t count) noexcept;
[[nodiscard]] bool capture_active() noexcept;
[[nodiscard]] bool coverage_complete() noexcept;

// FIFO thread
[[nodiscard]] bool captures(const DrawClass& cls, const gx::draw_route::Query& query) noexcept;
void route(const DrawClass& cls, const gx::draw_route::Query& query) noexcept;
void on_push(const gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
             const DrawClass& cls) noexcept;
void on_merge(const gx::draw_route::MergedDraw& draw, const DrawClass& cls) noexcept;
void on_capture(const gx::draw_route::CapturedDraw& draw) noexcept;
void prepare(detail::StageData& stage) noexcept;
void skip(detail::StageData& stage) noexcept;
void seal(detail::StageData& stage) noexcept;

// Published at end_frame and read from the next frame on
[[nodiscard]] RetainedStats stats() noexcept;
[[nodiscard]] FrameCounts frame_counts() noexcept;
[[nodiscard]] std::shared_ptr<const std::vector<uint64_t>> covered_objects() noexcept;
[[nodiscard]] std::shared_ptr<const FilteredObjects> covered_filtered() noexcept;

namespace testing {
struct Instance {
  bool live = false;
  bool delivered = false;
  bool latched = false;
  bool released = false;
  bool captureOnlySeen = false;
  bool visibleSeen = false;
  uint64_t lastSeen = 0;
  uint32_t slotGeneration = 0;
  InstanceInfo info;
  Affine world{};
};
void reset() noexcept;
[[nodiscard]] uint8_t demand() noexcept;
void set_check(bool on, uint64_t first = 0, uint64_t last = UINT64_MAX) noexcept;
void set_overrun(int overrun) noexcept;
[[nodiscard]] Instance instance(uint32_t slot) noexcept;
[[nodiscard]] uint32_t find(const Key& key) noexcept;
void evict(uint32_t slot, EvictReason reason) noexcept;
} // namespace testing
} // namespace aurora::render::scene::retained

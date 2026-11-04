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
};

// Game thread while the FIFO is idle
void latch_gate(const Settings& settings, const Caps& caps) noexcept;
void latch_resources(bool active) noexcept;
void set_demand(Demand demand, bool on) noexcept;
void apply_frame_state(const detail::FrameState& state) noexcept;
void end_frame() noexcept;
void reset() noexcept;
void want_covered_objects(bool want) noexcept;

[[nodiscard]] bool active() noexcept;

// FIFO thread
void route(const DrawClass& cls, const gx::draw_route::Query& query) noexcept;
void on_push(const gx::DrawData& draw, const gx::PipelineConfig& config) noexcept;
void on_merge(const gx::draw_route::MergedDraw& draw) noexcept;
void prepare(detail::StageData& stage) noexcept;
void skip(detail::StageData& stage) noexcept;
void seal(detail::StageData& stage) noexcept;

// Published at end_frame and read from the next frame on
[[nodiscard]] RetainedStats stats() noexcept;
[[nodiscard]] FrameCounts frame_counts() noexcept;
[[nodiscard]] std::shared_ptr<const std::vector<uint64_t>> covered_objects() noexcept;

namespace testing {
struct Instance {
  bool live = false;
  bool delivered = false;
  bool latched = false;
  uint32_t slotGeneration = 0;
  InstanceInfo info;
  Affine world{};
};
void reset() noexcept;
void set_check(bool on, uint64_t first = 0, uint64_t last = UINT64_MAX) noexcept;
void set_overrun(int overrun) noexcept;
[[nodiscard]] Instance instance(uint32_t slot) noexcept;
[[nodiscard]] uint32_t find(const Key& key) noexcept;
void evict(uint32_t slot, EvictReason reason) noexcept;
void route_capture_only(const DrawClass& cls, const gx::draw_route::Query& query) noexcept;
} // namespace testing
} // namespace aurora::render::scene::retained

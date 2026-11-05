#pragma once

#include "vsm_pages.hpp"

#include "../graph.hpp"
#include "../scene/identity.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::shadows::vsm {
inline constexpr uint32_t Slots = scene::retained::MaxSlots;
inline constexpr uint32_t HistoryWords = 32;
inline constexpr uint64_t HistoryBytes = uint64_t{Slots} * HistoryWords * 4;
inline constexpr uint32_t CastBitsWords = Slots / 32;
inline constexpr uint64_t CastBitsBytes = uint64_t{CastBitsWords} * 4;
inline constexpr uint64_t SlotWordsBytes = uint64_t{Slots} * 4;
inline constexpr uint32_t DirtyBoxWords = 16;
inline constexpr uint64_t DirtyBoxesBytes = uint64_t{Slots} * DirtyBoxWords * 4;
inline constexpr uint32_t DirtyPlaneWords = MaxVirtualPages / 32;
inline constexpr uint32_t DirtyPagesWords = 2 * DirtyPlaneWords;
inline constexpr uint64_t DirtyPagesBytes = uint64_t{DirtyPagesWords} * 4;
inline constexpr uint32_t ScanWorkgroup = 64;
inline constexpr uint32_t DirtyWorkgroup = 64;
inline constexpr uint32_t RecordWords = 96;

inline constexpr std::string_view ScanPass = "vsm.scan";
inline constexpr std::string_view ScanListPass = "vsm.scan_list";
inline constexpr std::string_view DirtyPass = "vsm.dirty";
inline constexpr std::string_view History = "vsm.history";
inline constexpr std::string_view CastBits = "vsm.cast_bits";
inline constexpr std::string_view SlotWordsName = "vsm.slot_words";
inline constexpr std::string_view DirtyBoxesName = "vsm.dirty_boxes";
inline constexpr std::string_view ListedSlots = "vsm.listed_slots";
inline constexpr std::string_view ListScratch = "vsm.list_scratch";
inline constexpr std::string_view DirtyArgs = "vsm.dirty_args";
inline constexpr std::string_view DirtyPagesName = "vsm.dirty_pages";
inline constexpr std::string_view InvalidateShader = "shadows/vsm_invalidate";

namespace hist {
inline constexpr uint32_t Flags = 0;
inline constexpr uint32_t Hash = 1;
inline constexpr uint32_t StreamFrame = 2;
inline constexpr uint32_t WorldFrame = 3;
inline constexpr uint32_t MaterialFrame = 4;
inline constexpr uint32_t UvFrame = 5;
inline constexpr uint32_t ClusterFirst = 6;
inline constexpr uint32_t ClusterCount = 7;
inline constexpr uint32_t IndexFirst = 8;
inline constexpr uint32_t IndexCount = 9;
inline constexpr uint32_t Box = 10;
inline constexpr uint32_t Xform = 16;
inline constexpr uint32_t VertexFirst = 28;
inline constexpr uint32_t CullMask = 29;

inline constexpr uint32_t Valid = 1u << 0;
inline constexpr uint32_t Caster = 1u << 1;
inline constexpr uint32_t Cutout = 1u << 2;
inline constexpr uint32_t AlphaPending = 1u << 3;
inline constexpr uint32_t Player = 1u << 4;
inline constexpr uint32_t Chunked = 1u << 5;
inline constexpr uint32_t PendingStream = 1u << 6;
inline constexpr uint32_t Seen = 1u << 7;
inline constexpr uint32_t BoxValid = 1u << 8;
} // namespace hist

namespace slot_word {
inline constexpr uint32_t Changed = 1u << 0;
inline constexpr uint32_t ChunkNeed = 1u << 1;
} // namespace slot_word

namespace dirty_box {
inline constexpr uint32_t Old = 0;
inline constexpr uint32_t New = 6;
inline constexpr uint32_t Flags = 12;
inline constexpr uint32_t HasOld = 1u << 0;
inline constexpr uint32_t HasNew = 1u << 1;
inline constexpr uint32_t OldPlayer = 1u << 2;
inline constexpr uint32_t NewPlayer = 1u << 3;
} // namespace dirty_box

namespace reason {
inline constexpr uint32_t Caster = 1u << 0;
inline constexpr uint32_t Hash = 1u << 1;
inline constexpr uint32_t Stream = 1u << 2;
inline constexpr uint32_t World = 1u << 3;
inline constexpr uint32_t Material = 1u << 4;
inline constexpr uint32_t Uv = 1u << 5;
inline constexpr uint32_t Alpha = 1u << 6;
inline constexpr uint32_t Cutout = 1u << 7;
inline constexpr uint32_t Arrays = 1u << 8;
inline constexpr uint32_t Count = 9;
} // namespace reason

[[nodiscard]] std::array<detail::PersistentHandle, 2> scene_handles() noexcept;
[[nodiscard]] bool is_caster(const scene::retained::InstanceInfo& info) noexcept;
[[nodiscard]] bool record_is_caster(uint32_t flags, uint32_t categories) noexcept;

// Worker, read from the stage's retained Frame before any VSM pass encodes
[[nodiscard]] FrameSignals frame_signals(const detail::StageData& stage) noexcept;

struct Box {
  std::array<float, 3> lo{};
  std::array<float, 3> hi{};
};

struct SlotScan {
  std::array<uint32_t, HistoryWords> history{};
  bool castNow = false;
  bool changed = false;
  bool chunkNeed = false;
  bool pending = false;
  bool pendingCaster = false;
  uint32_t reasons = 0;
  std::array<uint32_t, DirtyBoxWords> box{};
};

// The check and the tests hold the GPU scan to this; empty materials read every material as absent
[[nodiscard]] SlotScan scan_slot(std::span<const uint32_t, HistoryWords> prev,
                                 std::span<const uint32_t, RecordWords> rec, std::span<const uint32_t> materials,
                                 const ViewsHeader& header) noexcept;
[[nodiscard]] Box history_box(std::span<const uint32_t, HistoryWords> history) noexcept;
// grow widens the rect in texels, so a reference can bracket the GPU result against rounding at page edges
[[nodiscard]] std::array<int32_t, 4> box_rect(const ViewRecord& view, const Box& box, float grow = 0.f) noexcept;
void dirty_footprint(const Views& views, std::span<const uint32_t, DirtyBoxWords> box, std::span<uint32_t> planes,
                     float grow = 0.f) noexcept;

enum class CasterCover : uint8_t { Missing, Full, Partial, ExcludedOnly };
struct CasterObject {
  CasterCover cover = CasterCover::Missing;
  std::span<const uint32_t> slots;
};

class CasterObjects {
public:
  CasterObjects() = default;
  explicit CasterObjects(std::shared_ptr<const scene::retained::FilteredObjects> objects) noexcept
  : m_objects(std::move(objects)) {}
  [[nodiscard]] CasterObject find(uint64_t id) const noexcept;
  [[nodiscard]] bool valid() const noexcept { return m_objects != nullptr; }
  [[nodiscard]] size_t size() const noexcept { return m_objects != nullptr ? m_objects->objects.size() : 0; }

private:
  std::shared_ptr<const scene::retained::FilteredObjects> m_objects;
};

// Latched N -> N+1 with retained::covered_objects(); empty until the filter is registered and a frame published
[[nodiscard]] CasterObjects caster_objects() noexcept;

[[nodiscard]] detail::PassDesc scan_pass_desc();
[[nodiscard]] detail::PassDesc scan_list_pass_desc();
[[nodiscard]] detail::PassDesc dirty_pass_desc();

namespace testing {
void set_scene_handles(std::optional<std::array<detail::PersistentHandle, 2>> handles) noexcept;
} // namespace testing
} // namespace aurora::render::shadows::vsm

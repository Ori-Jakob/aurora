#pragma once

#include "vsm_alloc.hpp"
#include "vsm_invalidate.hpp"
#include "vsm_pages.hpp"

#include "../graph.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::shadows::vsm {
inline constexpr std::string_view ChunkBoundsPass = "vsm.chunk_bounds";
inline constexpr std::string_view CullCandidatesPass = "vsm.cull_candidates";
inline constexpr std::string_view CullListPass = "vsm.cull_list";
inline constexpr std::string_view CullCountPass = "vsm.cull_count";
inline constexpr std::string_view CullScanPass = "vsm.cull_scan";
inline constexpr std::string_view CullSkipPass = "vsm.cull_skip";
inline constexpr std::string_view CullEmitPass = "vsm.cull_emit";
inline constexpr std::string_view ChunkBoxes = "vsm.chunk_boxes";
inline constexpr std::string_view CullWords = "vsm.cull_words";
inline constexpr std::string_view CullSlots = "vsm.cull_slots";
inline constexpr std::string_view CullScratch = "vsm.cull_scratch";
inline constexpr std::string_view CullArgs = "vsm.cull_args";
inline constexpr std::string_view PageCounts = "vsm.page_counts";
inline constexpr std::string_view PageOffsets = "vsm.page_offsets";
inline constexpr std::string_view OffsetScratch = "vsm.offset_scratch";
inline constexpr std::string_view Pairs = "vsm.pairs";
inline constexpr std::string_view RasterArgs = "vsm.raster_args";
inline constexpr std::string_view CullShader = "shadows/vsm_cull";

inline constexpr uint32_t ChunkTriangles = 124;
inline constexpr uint32_t ChunkIndices = 3 * ChunkTriangles;
inline constexpr uint32_t VerticesPerPair = ChunkIndices;
inline constexpr uint32_t IndexGranule = 64;
inline constexpr uint32_t ChunkBoxCount = 65536;
inline constexpr uint64_t ChunkBoxesBytes = uint64_t{ChunkBoxCount} * 8;
inline constexpr uint32_t ChunkQuanta = 1023;
inline constexpr uint32_t NoChunk = 0xFFFFFFFFu;
inline constexpr uint32_t PairChunk = 0x80000000u;
inline constexpr uint32_t PairSlotBits = 14;
inline constexpr uint32_t CullWorkgroup = 64;
inline constexpr uint32_t ChunkWorkgroup = 128;
inline constexpr float CentreSlack = 1e-3f;
inline constexpr uint32_t PageCountWords = 2 * MaxRenderPages;
inline constexpr uint64_t PageCountBytes = uint64_t{PageCountWords} * 4;
inline constexpr uint64_t CullWordsBytes = uint64_t{Slots} * 4;
inline constexpr uint64_t RasterArgsBytes = 8 * 4;

namespace pairs_header {
inline constexpr uint32_t OpaqueCount = 0;
inline constexpr uint32_t CutoutBase = 1;
inline constexpr uint32_t CutoutCount = 2;
inline constexpr uint32_t Skipped = 3;
inline constexpr uint32_t ChunkPairs = 4;
inline constexpr uint32_t PosedPairs = 5;
inline constexpr uint32_t TotalOpaque = 6;
inline constexpr uint32_t SkipKey = 7;
inline constexpr uint32_t Dropped = 8;
inline constexpr uint32_t Max = 9;
inline constexpr uint32_t Words = 16;
inline constexpr uint32_t Ends = Words + 2 * MaxPairs;
inline constexpr uint32_t PairsWords = Ends + 2 * MaxRenderPages;
inline constexpr uint64_t Bytes = uint64_t{PairsWords} * 4;
} // namespace pairs_header

namespace stat {
inline constexpr uint32_t CullSlots = 44;
inline constexpr uint32_t SkippedPages = 45;
inline constexpr uint32_t OpaquePairs = 46;
inline constexpr uint32_t CutoutPairs = 47;
inline constexpr uint32_t ChunkBoxes = 48;
inline constexpr uint32_t ChunkMisaligned = 49;
} // namespace stat

struct Pair {
  uint32_t item = 0;
  bool chunk = false;
  uint32_t slot = 0;
  uint32_t renderIndex = 0;
};
[[nodiscard]] constexpr std::array<uint32_t, 2> pair_pack(uint32_t item, uint32_t slot, uint32_t renderIndex) noexcept {
  return {item, (slot & ((1u << PairSlotBits) - 1)) | renderIndex << PairSlotBits};
}
[[nodiscard]] constexpr Pair pair_unpack(const std::array<uint32_t, 2>& w) noexcept {
  return {.item = w[0] & ~PairChunk,
          .chunk = (w[0] & PairChunk) != 0,
          .slot = w[1] & ((1u << PairSlotBits) - 1),
          .renderIndex = w[1] >> PairSlotBits};
}
[[nodiscard]] constexpr uint32_t chunk_count(uint32_t indexCount) noexcept {
  return (indexCount / 3 + ChunkTriangles - 1) / ChunkTriangles;
}
[[nodiscard]] constexpr uint32_t chunk_slot(uint32_t indexFirst, uint32_t chunk) noexcept {
  return indexFirst % IndexGranule != 0 ? NoChunk : indexFirst / IndexGranule + chunk;
}
[[nodiscard]] std::array<uint32_t, 2> chunk_box_pack(const Box& box, const Box& bounds) noexcept;
[[nodiscard]] Box chunk_box_unpack(const std::array<uint32_t, 2>& words, const Box& bounds) noexcept;
[[nodiscard]] Box box_through(const Box& local, std::span<const uint32_t, 12> xform) noexcept;

struct TexelRect {
  std::array<float, 2> lo{};
  std::array<float, 2> hi{};
};
[[nodiscard]] TexelRect texel_rect(const ViewRecord& view, const Box& box) noexcept;
[[nodiscard]] bool rect_holds_centre(const TexelRect& rect, float slack = CentreSlack) noexcept;
[[nodiscard]] std::array<int32_t, 4> page_rect(const ViewRecord& view, const TexelRect& rect,
                                               float slack = CentreSlack) noexcept;
[[nodiscard]] uint32_t row_bits(uint32_t x0, uint32_t x1) noexcept;
[[nodiscard]] uint32_t skip_first(std::span<const uint32_t> counts, uint32_t maxPairs) noexcept;

struct CullItem {
  Box box;
  uint32_t word = 0;
  bool chunk = false;
  bool posed = false;
};
[[nodiscard]] std::vector<CullItem> cull_items(std::span<const uint32_t, HistoryWords> history,
                                               std::span<const uint32_t> clusters,
                                               std::span<const uint32_t> chunkBoxes) noexcept;

struct CullReference {
  std::vector<uint64_t> exact;
  std::vector<uint64_t> loose;
};
[[nodiscard]] constexpr uint64_t pair_key(uint32_t renderIndex, uint32_t item, uint32_t slot) noexcept {
  return uint64_t{renderIndex} << 46 | uint64_t{item} << 14 | slot;
}
[[nodiscard]] CullReference cull_reference(const Views& views, std::span<const uint32_t> history,
                                           std::span<const uint32_t> clusters, std::span<const uint32_t> chunkBoxes,
                                           std::span<const uint32_t> renderList, float margin);

[[nodiscard]] std::array<detail::PersistentHandle, 3> geometry_handles() noexcept;
[[nodiscard]] detail::PassDesc chunk_bounds_pass_desc();
[[nodiscard]] detail::PassDesc cull_candidates_pass_desc();
[[nodiscard]] detail::PassDesc cull_list_pass_desc();
[[nodiscard]] detail::PassDesc cull_count_pass_desc();
[[nodiscard]] detail::PassDesc cull_scan_pass_desc();
[[nodiscard]] detail::PassDesc cull_skip_pass_desc();
[[nodiscard]] detail::PassDesc cull_emit_pass_desc();

namespace testing {
void set_geometry_handles(std::optional<std::array<detail::PersistentHandle, 3>> handles) noexcept;
} // namespace testing
} // namespace aurora::render::shadows::vsm

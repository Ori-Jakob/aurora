#pragma once

#include "vsm_pages.hpp"

#include "../graph.hpp"
#include "../stage_camera.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::shadows::vsm {
inline constexpr std::string_view MarkPass = "vsm.mark";
inline constexpr std::string_view ClassifyPass = "vsm.classify";
inline constexpr std::string_view AllocListsPass = "vsm.alloc_lists";
inline constexpr std::string_view AssignPass = "vsm.assign";
inline constexpr std::string_view ClearPass = "vsm.clear";
inline constexpr std::string_view CommitPass = "vsm.commit";
inline constexpr std::string_view Requests = "vsm.requests";
inline constexpr std::string_view NeedWords = "vsm.need_words";
inline constexpr std::string_view PhysWordsName = "vsm.phys_words";
inline constexpr std::string_view NeedList = "vsm.need_list";
inline constexpr std::string_view NeedScratch = "vsm.need_scratch";
inline constexpr std::string_view FreeList = "vsm.free_list";
inline constexpr std::string_view FreeScratch = "vsm.free_scratch";
inline constexpr std::string_view CheckScratch = "vsm.check_scratch";
inline constexpr std::string_view RenderList = "vsm.render_list";
inline constexpr std::string_view ClearArgs = "vsm.clear_args";
inline constexpr std::string_view MarkReceivers = "vsm.mark_receivers";
inline constexpr std::string_view AllocShader = "shadows/vsm_alloc";

inline constexpr uint16_t RequestOrderFirst = 11;
inline constexpr uint16_t RequestOrderLast = 19;

namespace request_bit {
inline constexpr uint32_t Pixel = 1u << 0;
inline constexpr uint32_t Fallback = 1u << 1;
inline constexpr uint32_t Volume = 1u << 2;
inline constexpr uint32_t Priority = 1u << 3;
inline constexpr uint32_t Mask = 0xFu;
} // namespace request_bit

namespace need_class {
inline constexpr uint32_t Priority = 0;
inline constexpr uint32_t New = 1;
inline constexpr uint32_t Reuse = 2;
inline constexpr uint32_t Count = 3;
inline constexpr uint32_t Check = 3;
} // namespace need_class

namespace phys_band {
inline constexpr uint32_t Free = 0;
inline constexpr uint32_t Old = 1;
inline constexpr uint32_t Mid = 2;
inline constexpr uint32_t Young = 3;
inline constexpr uint32_t Count = 4;
} // namespace phys_band

inline constexpr uint32_t OldAge = 64;
inline constexpr uint32_t MidAge = 8;
inline constexpr uint32_t NeedItems = need_class::Count * MaxVirtualPages;
inline constexpr uint32_t PhysItems = phys_band::Count * MaxPoolPages;
inline constexpr uint64_t RequestsBytes = uint64_t{MaxVirtualPages} * 4;
inline constexpr uint64_t NeedWordsBytes = uint64_t{NeedItems} * 4;
inline constexpr uint64_t CheckWordsBytes = uint64_t{MaxVirtualPages} * 4;
inline constexpr uint64_t NeedCheckBytes = NeedWordsBytes + CheckWordsBytes;
inline constexpr uint64_t PhysWordsBytes = uint64_t{PhysItems} * 4;
inline constexpr uint32_t MaxRenderPages = MaxPoolPages;
inline constexpr uint32_t NoPage = 0xFFFFFFFFu;
inline constexpr uint32_t MarkWorkgroup = 8;
inline constexpr uint32_t AllocWorkgroup = 64;
inline constexpr uint32_t MaxMarkKeys = 8;
inline constexpr uint32_t ReceiverWords = 2;

namespace render_list {
inline constexpr uint32_t Count = 0;
inline constexpr uint32_t Priority = 1;
inline constexpr uint32_t New = 2;
inline constexpr uint32_t Reuse = 3;
inline constexpr uint32_t Budget = 4;
inline constexpr uint32_t PagesX = 5;
inline constexpr uint32_t Rows = 6;
inline constexpr uint32_t Flags = 7;
inline constexpr uint32_t CheckCount = 8;
inline constexpr uint32_t HeaderWords = 16;
inline constexpr uint32_t Pages = HeaderWords;
inline constexpr uint32_t RowMasks = Pages + 2 * MaxRenderPages;
inline constexpr uint32_t Words = RowMasks + MaxViews * WindowPages;
inline constexpr uint64_t Bytes = uint64_t{Words} * 4;
inline constexpr uint32_t CheckOnly = 1u << 31;
[[nodiscard]] constexpr uint32_t virtual_page(uint32_t word) noexcept {
  return word == NoPage ? NoPage : word & ~CheckOnly;
}
} // namespace render_list

namespace stat {
inline constexpr uint32_t CheckPages = 50;
inline constexpr uint32_t CheckListed = 51;
inline constexpr uint32_t CheckCompared = 52;
inline constexpr uint32_t CheckTexels = 53;
inline constexpr uint32_t CheckMismatched = 54;
inline constexpr uint32_t CheckFirst = 55;
inline constexpr uint32_t CheckSkipped = 56;
inline constexpr uint32_t ProbeFound = 60;
inline constexpr uint32_t ProbeOccluded = 61;
} // namespace stat

struct MarkParams {
  detail::StageCamera camera;
  detail::ViewRotation rotation;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t flags = 0;
  uint32_t reserved = 0;
};
static_assert(sizeof(MarkParams) == 112 && sizeof(MarkParams) <= MaxPassParamsSize);

struct MarkKeys {
  std::array<uint32_t, MaxMarkKeys> keys{};
  uint32_t count = 0;
};
[[nodiscard]] constexpr uint32_t request_key(uint32_t index, uint32_t bits) noexcept { return index | bits << 16; }
[[nodiscard]] uint32_t level_wanted(float footprint, float texel0, float bias) noexcept;
[[nodiscard]] MarkKeys mark_keys(const Views& views, const std::array<float, 3>& pRel, float footprint) noexcept;

struct Receiver {
  bool valid = false;
  std::array<float, 3> pRel{};
  float footprint = 0.f;
};
[[nodiscard]] Receiver mark_receiver(const MarkParams& params, uint32_t x, uint32_t y, float depth,
                                     float normalAlpha) noexcept;

struct MarkReference {
  std::vector<uint32_t> must;
  std::vector<uint32_t> may;
  uint64_t receivers = 0;
  uint64_t ambiguous = 0;
};
[[nodiscard]] MarkReference mark_reference(const Views& views, const MarkParams& params,
                                           std::span<const uint32_t> receivers);

struct AllocCounts {
  uint32_t requested = 0;
  uint32_t priority = 0;
  uint32_t fresh = 0;
  uint32_t reuse = 0;
  uint32_t hits = 0;
  uint32_t freedDirty = 0;
  uint32_t evictable = 0;
  uint32_t freePages = 0;
  uint32_t rendered = 0;
  uint32_t deferredNew = 0;
  uint32_t deferredReuse = 0;
  uint32_t overflowPool = 0;
  uint32_t playerDeferred = 0;
  uint32_t evicted = 0;
};

struct AllocBudget {
  uint32_t count = 0;
  uint32_t fresh = 0;
  uint32_t reuse = 0;
  uint32_t deferredNew = 0;
  uint32_t deferredReuse = 0;
  uint32_t overflowPool = 0;
  uint32_t playerDeferred = 0;
};
[[nodiscard]] AllocBudget alloc_budget(uint32_t priority, uint32_t fresh, uint32_t reuse, uint32_t evictable,
                                       uint32_t budget, bool held) noexcept;

[[nodiscard]] uint32_t render_entries(std::span<const uint32_t> renderList) noexcept;
[[nodiscard]] AllocCounts allocate(const Views& views, std::span<const uint32_t> requests, std::span<uint32_t> table,
                                   std::span<uint32_t> phys, std::span<uint32_t> renderList) noexcept;
uint32_t commit(std::span<const uint32_t> renderList, std::span<uint32_t> table) noexcept;
[[nodiscard]] uint32_t phys_band_of(uint32_t owner, uint32_t lastUsed, uint32_t clock) noexcept;

[[nodiscard]] detail::PassDesc mark_pass_desc();
[[nodiscard]] detail::PassDesc classify_pass_desc();
[[nodiscard]] detail::PassDesc alloc_lists_pass_desc();
[[nodiscard]] detail::PassDesc assign_pass_desc();
[[nodiscard]] detail::PassDesc clear_pass_desc();
[[nodiscard]] detail::PassDesc commit_pass_desc();
void alloc_shutdown() noexcept;
} // namespace aurora::render::shadows::vsm

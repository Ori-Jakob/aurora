#pragma once

#include "../../gx/render_ext.hpp"
#include "../../gx/surface_pack.hpp"
#include "../graph.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace aurora::render::gx_ext::surface_table {
inline constexpr std::string_view UploadPass = "scene.surface_upload";
inline constexpr uint32_t Capacity = 4096;
// Slots 1-63 belong to Surface writers that are not GX draws; the FIFO hands out the rest
inline constexpr uint32_t StaticSlots = 64;
inline constexpr uint64_t TableBytes = uint64_t{Capacity} * sizeof(gx::surface::Entry);
static_assert(Capacity <= (1u << gx::render_ext::RecordSurfaceSlotBits));
static_assert(Capacity - 1 <= gx::surface::SlotMax);

using Entry = gx::surface::Entry;
using Table = std::vector<Entry>;

struct Stats {
  uint64_t frames = 0;
  uint64_t slots = 0;
  uint64_t overflow = 0;
  uint32_t maxEntries = 0;
};

// Game thread; graph passes and GPU-driven draws have no FIFO record, so they keep a fixed slot
[[nodiscard]] uint32_t reserve(const Entry& entry) noexcept;
void update(uint32_t slot, const Entry& entry) noexcept;
void release(uint32_t slot) noexcept;

// Game thread, while the FIFO is idle
void begin_frame(bool active) noexcept;
void shutdown() noexcept;
[[nodiscard]] Stats stats() noexcept;

// FIFO thread
[[nodiscard]] uint32_t slot_for(const Entry& entry) noexcept;
[[nodiscard]] std::shared_ptr<const Table> snapshot();

detail::PassDesc upload_pass_desc();
void initialize();
void unregister_passes() noexcept;

namespace testing {
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext::surface_table

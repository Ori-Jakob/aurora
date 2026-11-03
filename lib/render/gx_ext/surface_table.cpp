#include "surface_table.hpp"

#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../../logging.hpp"

#include <absl/container/flat_hash_map.h>
#include <aurora/render_graph.hpp>

#include <algorithm>
#include <mutex>
#include <utility>

namespace aurora::render::gx_ext::surface_table {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};
constexpr BufRef TableRef{0};

std::mutex g_staticMutex;
std::array<Entry, StaticSlots> g_static{};
std::array<bool, StaticSlots> g_reserved{};

detail::PassId g_upload = detail::InvalidPassId;

// FIFO thread, reset while the FIFO is idle
bool g_active = false;
Table g_entries;
absl::flat_hash_map<Entry, uint32_t> g_slots;
Entry g_lastEntry{};
uint32_t g_lastSlot = 0;
uint64_t g_frameOverflow = 0;
Stats g_totals;
bool g_overflowWarned = false;

void fold() noexcept {
  if (!g_active) {
    return;
  }
  ++g_totals.frames;
  const auto entries = static_cast<uint32_t>(g_entries.size() > StaticSlots ? g_entries.size() - StaticSlots : 0);
  g_totals.slots += entries;
  g_totals.maxEntries = std::max(g_totals.maxEntries, entries);
  g_totals.overflow += g_frameOverflow;
  if (g_frameOverflow != 0 && !std::exchange(g_overflowWarned, true)) {
    Log.warn("Surface table: {} draws read as legacy because the frame used all {} slots", g_frameOverflow, Capacity);
  }
}

void encode_upload(detail::PassContext& ctx, const void*) {
  const std::shared_ptr<const Table>& table = ctx.stage_data().surfaceTable;
  const BufferView view = ctx.buffer(TableRef);
  if (table == nullptr || !view.buffer) {
    return;
  }
  const uint64_t bytes = std::min<uint64_t>(table->size() * sizeof(Entry), view.size);
  ctx.encoder().WriteBuffer(view.buffer, view.offset, reinterpret_cast<const uint8_t*>(table->data()), bytes);
}

void declare_upload(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!setup.gx_available(detail::GxResource::Surface)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.create_buffer(resource::SurfaceTable, {.size = TableBytes}, Access::CopyDst);
}

bool upload_enabled(const detail::StageData& stage, const void*) { return stage.surfaceTable != nullptr; }
} // namespace

uint32_t reserve(const Entry& entry) noexcept {
  std::lock_guard lock{g_staticMutex};
  for (uint32_t slot = 1; slot < StaticSlots; ++slot) {
    if (!g_reserved[slot]) {
      g_reserved[slot] = true;
      g_static[slot] = entry;
      return slot;
    }
  }
  return 0;
}

void update(uint32_t slot, const Entry& entry) noexcept {
  std::lock_guard lock{g_staticMutex};
  if (slot != 0 && slot < StaticSlots && g_reserved[slot]) {
    g_static[slot] = entry;
  }
}

void release(uint32_t slot) noexcept {
  std::lock_guard lock{g_staticMutex};
  if (slot != 0 && slot < StaticSlots) {
    g_reserved[slot] = false;
    g_static[slot] = {};
  }
}

void begin_frame(bool active) noexcept {
  fold();
  g_active = active;
  g_slots.clear();
  g_lastEntry = {};
  g_lastSlot = 0;
  g_frameOverflow = 0;
  std::lock_guard lock{g_staticMutex};
  g_entries.assign(g_static.begin(), g_static.end());
}

uint32_t slot_for(const Entry& entry) noexcept {
  if (!g_active || (entry[0] & 0xFFu) == gx::surface::ClassNone) {
    return 0;
  }
  if (g_lastSlot != 0 && entry == g_lastEntry) {
    return g_lastSlot;
  }
  uint32_t slot = 0;
  if (const auto it = g_slots.find(entry); it != g_slots.end()) {
    slot = it->second;
  } else if (g_entries.size() < Capacity) {
    slot = static_cast<uint32_t>(g_entries.size());
    g_entries.push_back(entry);
    g_slots.emplace(entry, slot);
  } else {
    ++g_frameOverflow;
    return 0;
  }
  g_lastEntry = entry;
  g_lastSlot = slot;
  return slot;
}

std::shared_ptr<const Table> snapshot() {
  if (!g_active) {
    return nullptr;
  }
  return std::make_shared<const Table>(g_entries);
}

Stats stats() noexcept { return g_totals; }

void shutdown() noexcept {
  fold();
  if (g_totals.frames != 0) {
    Log.info("Surface table: {} frames, up to {} entries, {} overflow", g_totals.frames, g_totals.maxEntries,
             g_totals.overflow);
  }
  g_active = false;
  g_entries.clear();
  g_slots.clear();
  g_lastEntry = {};
  g_lastSlot = 0;
  g_frameOverflow = 0;
  g_totals = {};
  g_overflowWarned = false;
}

detail::PassDesc upload_pass_desc() {
  return {
      .name = std::string{UploadPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .cls = detail::PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .declare = declare_upload,
      .enabled = upload_enabled,
      .encode = encode_upload,
  };
}

void initialize() {
  if (g_upload == detail::InvalidPassId) {
    g_upload = detail::register_pass(upload_pass_desc());
  }
}

void unregister_passes() noexcept {
  if (g_upload != detail::InvalidPassId) {
    detail::unregister_pass(g_upload);
    g_upload = detail::InvalidPassId;
  }
}

namespace testing {
void reset() noexcept {
  shutdown();
  std::lock_guard lock{g_staticMutex};
  g_static = {};
  g_reserved = {};
}
} // namespace testing
} // namespace aurora::render::gx_ext::surface_table

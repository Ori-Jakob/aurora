#include "command_processor.hpp"

#include "../gfx/depth_peek.hpp"
#include "../gfx/hash.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "draw_route.hpp"
#include "guest_memory.hpp"
#include "gx.hpp"
#include "hints.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "shader_info.hpp"
#include "texture.hpp"

#include <SDL3/SDL_stdinc.h>
#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};

u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

static_assert(GX_AURORA_DRAW_INDEXED < GX_AURORA_HINT_FIRST && GX2_SET_POLYGON_OFFSET > GX_AURORA_HINT_LAST,
              "length-prefixed aurora hints must not overlap legacy subcommands");

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
};
DrawCache sDrawCache;

// FIFO thread only; copies run at the end of the frame that staged them, so a freed block can be reused at once
struct ImmutableArray {
  u32 size = 0;
  bool le = false;
  u32 generation = 0;
  u64 setFrame = 0;
  u32 offset = 0;
  u32 blockSize = 0;
  u32 epoch = 0;
  u64 stagedFrame = 0;
  HashType hash = 0;
  bool warned = false;
};
absl::btree_map<uintptr_t, ImmutableArray> g_immutableArrays;
struct Arena {
  u64 frameId = 0;
  u32 gfxEpoch = 0;
  u32 epoch = 1;
  u32 base = 0;
  u32 capacity = 0;
  u32 limit = 0;
  u32 blockAlign = 0;
  u32 used = 0;
  u32 live = 0;
  absl::btree_map<u32, u32> free;
  bool bypass = false;
  bool reset = false;
} g_arena;
ImmutableArrayMode g_immutableMode = ImmutableArrayMode::On;
bool g_warnedOverlap = false;
bool g_warnedStaleRelease = false;

std::atomic<uint64_t> g_routed{0};
std::atomic<uint64_t> g_suppressed{0};
std::atomic<uint64_t> g_mergeBreaks{0};
std::atomic<uint64_t> g_substituted{0};
std::atomic_uint32_t g_cpuZoneTimed{0};
std::array<std::atomic_uint64_t, CpuZoneCount> g_cpuZoneCalls{};
std::array<std::atomic_uint64_t, CpuZoneCount> g_cpuZoneNs{};
std::atomic_uint64_t g_cpuArrayBytes{0};
std::atomic_uint64_t g_cpuFogLutBytes{0};
std::atomic_uint64_t g_arrayReused{0};
std::atomic_uint64_t g_arrayReusedBytes{0};
std::atomic_uint64_t g_arrayChanged{0};
std::atomic_uint64_t g_residentHits{0};
std::atomic_uint64_t g_residentStages{0};
std::atomic_uint64_t g_residentStageBytes{0};
std::atomic_uint64_t g_residentResets{0};
std::atomic_uint64_t g_residentEpochs{0};
std::atomic_uint64_t g_staleReleases{0};
std::atomic_uint64_t g_overlapEvictions{0};
std::atomic_uint64_t g_sizeMismatches{0};
std::atomic_uint64_t g_arenaFull{0};
std::atomic_uint64_t g_residentViolations{0};

// Only the decoding thread writes these, so a plain add keeps lock prefixes off the per-draw path
inline void bump(std::atomic_uint64_t& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

class CpuZoneScope {
public:
  explicit CpuZoneScope(CpuZone zone) noexcept
  : m_zone(zone), m_timed(((g_cpuZoneTimed.load(std::memory_order_relaxed) >> zone) & 1u) != 0) {
    if (m_timed) {
      m_start = std::chrono::steady_clock::now();
    }
  }
  ~CpuZoneScope() {
    bump(g_cpuZoneCalls[m_zone]);
    if (m_timed) {
      const auto elapsed = std::chrono::steady_clock::now() - m_start;
      bump(g_cpuZoneNs[m_zone],
           static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
    }
  }
  CpuZoneScope(const CpuZoneScope&) = delete;
  CpuZoneScope& operator=(const CpuZoneScope&) = delete;

private:
  std::chrono::steady_clock::time_point m_start{};
  CpuZone m_zone;
  bool m_timed;
};

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

struct ArrayUpload {
  gfx::Range range;
  size_t shadowOffset = 0;
};
// Ranges index this frame's storage buffer, so they are dropped with the rest of the draw cache
absl::flat_hash_map<std::pair<const void*, u32>, ArrayUpload> sArrayUploads;
// Staging memory is write-combined, so reuse is checked against a CPU copy of the pushed bytes
std::vector<u8> sArrayShadow;

gfx::Range push_array(const AttrArray& array, std::span<const u8>* pushed = nullptr) noexcept {
  const auto bytes = guest_memory::read(guest_memory::Kind::VertexArray, array.data, array.size);
  if (bytes.empty()) {
    return gfx::push_storage(bytes.data(), bytes.size());
  }
  auto [it, inserted] = sArrayUploads.try_emplace(std::pair<const void*, u32>{array.data, array.size});
  auto& upload = it->second;
  if (inserted) {
    upload.shadowOffset = sArrayShadow.size();
    sArrayShadow.insert(sArrayShadow.end(), bytes.begin(), bytes.end());
  } else {
    u8* shadow = sArrayShadow.data() + upload.shadowOffset;
    // The game may rewrite an array between binds, so only identical bytes reuse the earlier upload
    if (std::memcmp(shadow, bytes.data(), bytes.size()) == 0) {
      bump(g_arrayReused);
      bump(g_arrayReusedBytes, bytes.size());
      return upload.range;
    }
    bump(g_arrayChanged);
    std::memcpy(shadow, bytes.data(), bytes.size());
  }
  // Live reads alias game memory, so the upload copies the shadow to stay identical to what later binds compare
  const auto range = gfx::push_storage(sArrayShadow.data() + upload.shadowOffset, bytes.size());
  bump(g_cpuArrayBytes, bytes.size());
  if (range.size == 0) {
    if (inserted) {
      sArrayShadow.resize(upload.shadowOffset);
    }
    sArrayUploads.erase(it);
    return range;
  }
  upload.range = range;
  if (pushed != nullptr) {
    *pushed = {sArrayShadow.data() + upload.shadowOffset, bytes.size()};
  }
  return range;
}

// A cached upload may hold an earlier generation's bytes, so a stage always takes a fresh read
gfx::Range push_array_fresh(const AttrArray& array, std::span<const u8>& pushed) noexcept {
  sArrayUploads.erase(std::pair<const void*, u32>{array.data, array.size});
  return push_array(array, &pushed);
}

void erase_uploads_overlapping(uintptr_t begin, uintptr_t end) noexcept {
  absl::erase_if(sArrayUploads, [begin, end](const auto& upload) {
    const auto data = reinterpret_cast<uintptr_t>(upload.first.first);
    return data < end && begin < data + upload.first.second;
  });
}

std::optional<u32> arena_alloc(u32 size) noexcept {
  auto best = g_arena.free.end();
  for (auto it = g_arena.free.begin(); it != g_arena.free.end(); ++it) {
    if (it->second >= size && (best == g_arena.free.end() || it->second < best->second)) {
      best = it;
    }
  }
  if (best != g_arena.free.end()) {
    const u32 offset = best->first;
    const u32 remainder = best->second - size;
    g_arena.free.erase(best);
    if (remainder != 0) {
      g_arena.free.emplace(offset + size, remainder);
    }
    return offset;
  }
  if (u64{g_arena.used} + size > g_arena.capacity) {
    return std::nullopt;
  }
  const u32 offset = g_arena.used;
  g_arena.used += size;
  return offset;
}

void arena_free(u32 offset, u32 size) noexcept {
  g_arena.live -= size;
  auto next = g_arena.free.lower_bound(offset);
  if (next != g_arena.free.end() && offset + size == next->first) {
    size += next->second;
    next = g_arena.free.erase(next);
  }
  if (next != g_arena.free.begin()) {
    const auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      offset = prev->first;
      size += prev->second;
      g_arena.free.erase(prev);
    }
  }
  if (offset + size == g_arena.used) {
    g_arena.used = offset;
  } else {
    g_arena.free.emplace(offset, size);
  }
}

using ImmutableArrayIt = absl::btree_map<uintptr_t, ImmutableArray>::iterator;

ImmutableArrayIt evict_immutable(ImmutableArrayIt it) noexcept {
  const ImmutableArray& entry = it->second;
  if (entry.epoch == g_arena.epoch) {
    arena_free(entry.offset, entry.blockSize);
  }
  erase_uploads_overlapping(it->first, it->first + entry.size);
  return g_immutableArrays.erase(it);
}

void sync_arena() noexcept {
  const auto storage = gfx::resident_storage();
  if (storage.frameId == g_arena.frameId) {
    return;
  }
  g_arena.frameId = storage.frameId;
  if (storage.frameId == 0) {
    return;
  }
  const auto* hooks = guest_memory::g_hooks.load(std::memory_order_acquire);
  const bool bypass = hooks != nullptr && hooks->observe != nullptr;
  if (storage.epoch != g_arena.gfxEpoch || g_arena.reset || bypass != g_arena.bypass) {
    ++g_arena.epoch;
    g_arena.used = 0;
    g_arena.live = 0;
    g_arena.free.clear();
    g_arena.reset = false;
    bump(g_residentEpochs);
  }
  g_arena.gfxEpoch = storage.epoch;
  g_arena.bypass = bypass;
  g_arena.base = storage.base;
  g_arena.capacity = storage.capacity;
  g_arena.limit = storage.limit;
  g_arena.blockAlign = storage.blockAlign;
}

void stage(ImmutableArray& entry, gfx::Range range, std::span<const u8> bytes) noexcept {
  const u32 blockSize = AURORA_ALIGN(entry.size, g_arena.blockAlign);
  if (const auto offset = arena_alloc(blockSize)) {
    gfx::copy_to_resident(range, *offset, blockSize);
    entry.offset = *offset;
    entry.blockSize = blockSize;
    entry.epoch = g_arena.epoch;
    entry.stagedFrame = g_arena.frameId;
    bump(g_residentStages);
    bump(g_residentStageBytes, entry.size);
    g_arena.live += blockSize;
    if (g_immutableMode == ImmutableArrayMode::Validate) {
      entry.hash = xxh3_hash_s(bytes.data(), bytes.size());
      entry.warned = false;
    }
    return;
  }
  const u64 need = u64{g_arena.live} + blockSize;
  if (2 * need > g_arena.capacity && g_arena.capacity < g_arena.limit) {
    gfx::request_resident_capacity(need);
  } else if (need <= g_arena.capacity) {
    g_arena.reset = true;
    bump(g_residentResets);
  } else {
    bump(g_arenaFull);
  }
}

void validate_resident(const AttrArray& array, ImmutableArray& entry) noexcept {
  const auto bytes = guest_memory::read(guest_memory::Kind::VertexArray, array.data, array.size);
  if (xxh3_hash_s(bytes.data(), bytes.size()) == entry.hash) {
    return;
  }
  bump(g_residentViolations);
  if (!std::exchange(entry.warned, true)) {
    Log.error("immutable array 0x{:X} ({} bytes, generation {}) changed without a new generation",
              reinterpret_cast<uintptr_t>(array.data), array.size, entry.generation);
  }
}

gfx::Range upload_array(const AttrArray& array) noexcept {
  if (g_immutableMode == ImmutableArrayMode::Off || g_immutableArrays.empty()) {
    return push_array(array);
  }
  sync_arena();
  const auto it =
      g_arena.frameId == 0 ? g_immutableArrays.end() : g_immutableArrays.find(reinterpret_cast<uintptr_t>(array.data));
  // A declaration is trusted only in a frame that sent it, so a missed release can never serve freed memory
  if (it == g_immutableArrays.end() || it->second.setFrame != g_arena.frameId) {
    return push_array(array);
  }
  auto& entry = it->second;
  if (entry.size != array.size || entry.le != array.le) {
    bump(g_sizeMismatches);
    return push_array(array);
  }
  if (g_arena.bypass) {
    return push_array(array);
  }
  if (entry.epoch == g_arena.epoch) {
    if (entry.stagedFrame != g_arena.frameId) {
      bump(g_residentHits);
      if (g_immutableMode == ImmutableArrayMode::Validate) {
        validate_resident(array, entry);
      }
      return {g_arena.base + entry.offset, array.size};
    }
    return push_array(array);
  }
  std::span<const u8> pushed;
  const auto range = push_array_fresh(array, pushed);
  if (range.size != 0) {
    stage(entry, range, pushed);
  }
  return range;
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  const CpuZoneScope zone{CpuZoneProcess};
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      const auto srcData =
          guest_memory::read(guest_memory::Kind::IndexedLoad, static_cast<const u8*>(array.data) + srcOffset, srcSize);
      if (!copy_xf_data(dstAddr, srcData.data(), len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      handle_aurora(reader);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) noexcept {
  ZoneScoped;
  const CpuZoneScope zone{CpuZonePushDraw};
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = state.arrays[i];
    if (array.cachedRange.size == 0) {
      ZoneScopedN("upload array");
      const CpuZoneScope arrays{CpuZoneArrays};
      array.cachedRange = upload_array(array);
    }
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout().key;
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  if (!pipelineValid) {
    ZoneScopedN("build pipeline");
    const CpuZoneScope build{CpuZonePipelineBuild};
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    populate_pipeline_config(cache.config, prim, fmt);
    cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
    cache.pipelineRef = gfx::pipeline_ref(cache.config);
    cache.targetLayoutKey = targetLayoutKey;
    cache.fmt = fmt;
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    ZoneScopedN("build bind groups");
    const CpuZoneScope build{CpuZoneBindGroupBuild};
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    resolve_sampled_textures(cache.shaderInfo);
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For texture_size_bias uniform
    if (cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyUniform;
    }
  }

  const bool uniformValid = (state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0;
  if (!uniformValid) {
    ZoneScopedN("build uniform");
    const CpuZoneScope build{CpuZoneUniformBuild};
    cache.uniformRange = build_draw_uniform(cache.shaderInfo);
    state.dirty &= ~DirtyUniform;
  }
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      ZoneScopedN("push fog range lut");
      const CpuZoneScope build{CpuZoneFogLut};
      cache.fogRange = push_fog_range_lut(key);
      bump(g_cpuFogLutBytes, cache.fogRange.size);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  cache.lastDrawFmt = fmt;
  DrawData draw{
      .pipeline = cache.pipelineRef,
      .bindGroups = cache.bindGroups,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .dstAlpha = state.dstAlpha,
  };
  const auto* router = draw_route::router();
  if (router != nullptr && router->push != nullptr &&
      router->push(router->userData, draw, cache.config, cache.shaderInfo)) {
    // The command no longer matches the cached state, so a later clean draw must not merge into it
    break_draw_merge();
    g_substituted.fetch_add(1, std::memory_order_relaxed);
  }
  gfx::push_draw_command(draw);
}

// draw_prim's merge test; it has no side effects, so routing can tell whether a break prevents a real merge
static DrawData* merge_target(GXPrimitive prim, GXVtxFmt fmt) noexcept {
  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 &&
                          prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  return lastDraw != nullptr && lastDraw->instanceCount == 1 ? lastDraw : nullptr;
}

// A diverted draw ends the merge chain, so no later draw can merge across what was left out
static bool route_draw(const draw_route::Query& query) noexcept {
  const auto* router = draw_route::router();
  if (router == nullptr || router->route == nullptr) {
    return false;
  }
  g_routed.fetch_add(1, std::memory_order_relaxed);
  const draw_route::Decision decision = router->route(router->userData, query);
  if (decision.route != draw_route::Route::Push) {
    break_draw_merge();
    g_suppressed.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  // Only a break that prevents a merge counts; an indexed draw always starts its own command
  if (decision.breakMerge && !query.indexed && merge_target(query.prim, query.fmt) != nullptr) {
    break_draw_merge();
    g_mergeBreaks.fetch_add(1, std::memory_order_relaxed);
  }
  return false;
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }

  if (route_draw({.prim = prim, .fmt = fmt, .vtxCount = vtxCount})) {
    reader.skip(totalVtxBytes);
    return;
  }

  auto* lastDraw = merge_target(prim, fmt);
  const bool canMerge = lastDraw != nullptr;

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
  gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

static void handle_draw_indexed(ByteReader& reader) noexcept {
  ZoneScopedN("DRAW_INDEXED");
  const u8 cmd = reader.read<u8>();
  const u16 vtxCount = reader.read<u16>();
  const u32 indexCount = reader.read<u32>();
  const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                static_cast<u32>(prim));
  const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt) {
    vtxSize = g_gxState.lastVtxSize;
  } else {
    vtxSize = calc_vtx_size(fmt);
  }
  const u32 totalVtxBytes = vtxCount * vtxSize;
  if (indexCount != 0 && route_draw({.prim = prim, .fmt = fmt, .vtxCount = vtxCount, .indexed = true})) {
    reader.skip(idxBytes + totalVtxBytes);
    return;
  }
  // Index data is always host-endian; push it to the GPU buffer as-is
  const auto indexData = reader.take(idxBytes);
  const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
  const auto vertexData = reader.take(totalVtxBytes);
  const gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), 4);
  if (indexCount != 0) {
    push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
  }
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    handle_draw_indexed(reader);
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  } else if (subCmd >= GX_AURORA_HINT_FIRST && subCmd <= GX_AURORA_HINT_LAST) {
    hints::handle(subCmd, reader);
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  clear_draw_uniforms();
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
  // clear() frees large tables, so reserving last frame's size regrows them once
  const size_t uploads = sArrayUploads.size();
  sArrayUploads.clear();
  sArrayUploads.reserve(uploads);
  sArrayShadow.clear();
}

// Failing the format check fails the whole merge test, and only push_gx_draw re-arms it
void break_draw_merge() noexcept { sDrawCache.lastDrawFmt = GX_MAX_VTXFMT; }

void invalidate_draw_cache() noexcept {
  clear_draw_cache();
  sDrawCache.hasPipeline = false;
  break_draw_merge();
}

DrawRouteStats draw_route_stats() noexcept {
  return {
      .routed = g_routed.load(std::memory_order_relaxed),
      .suppressed = g_suppressed.load(std::memory_order_relaxed),
      .mergeBreaks = g_mergeBreaks.load(std::memory_order_relaxed),
      .substituted = g_substituted.load(std::memory_order_relaxed),
  };
}

ArrayUploadStats array_upload_stats() noexcept {
  return {
      .reused = g_arrayReused.load(std::memory_order_relaxed),
      .reusedBytes = g_arrayReusedBytes.load(std::memory_order_relaxed),
      .changed = g_arrayChanged.load(std::memory_order_relaxed),
  };
}

CpuZoneStats cpu_zone_stats() noexcept {
  CpuZoneStats stats;
  for (size_t zone = 0; zone < CpuZoneCount; ++zone) {
    stats.calls[zone] = g_cpuZoneCalls[zone].load(std::memory_order_relaxed);
    stats.ns[zone] = g_cpuZoneNs[zone].load(std::memory_order_relaxed);
  }
  stats.arrayBytes = g_cpuArrayBytes.load(std::memory_order_relaxed);
  stats.fogLutBytes = g_cpuFogLutBytes.load(std::memory_order_relaxed);
  return stats;
}

void set_cpu_zone_timing(uint32_t timing) noexcept {
  constexpr uint32_t drawZones = ((1u << CpuZoneCount) - 1u) & ~(1u << CpuZoneProcess);
  g_cpuZoneTimed.store(((timing & CpuZoneTimingFifo) != 0 ? 1u << CpuZoneProcess : 0u) |
                           ((timing & CpuZoneTimingDraws) != 0 ? drawZones : 0u),
                       std::memory_order_relaxed);
}

void set_array_immutable(uint64_t address, uint32_t size, bool le, uint32_t generation) noexcept {
  const auto begin = static_cast<uintptr_t>(address);
  const u64 frameId = gfx::resident_storage().frameId;
  if (const auto it = g_immutableArrays.find(begin); it != g_immutableArrays.end()) {
    auto& entry = it->second;
    if (entry.size == size && entry.le == le && entry.generation == generation) {
      entry.setFrame = frameId;
      return;
    }
    evict_immutable(it);
  }
  const uintptr_t end = begin + size;
  auto it = g_immutableArrays.lower_bound(begin);
  if (it != g_immutableArrays.begin() && std::prev(it)->first + std::prev(it)->second.size > begin) {
    it = std::prev(it);
  }
  while (it != g_immutableArrays.end() && it->first < end) {
    bump(g_overlapEvictions);
    if (!std::exchange(g_warnedOverlap, true)) {
      Log.warn("Immutable array 0x{:X} ({} bytes) overlaps the one declared at 0x{:X} ({} bytes); evicting it", begin,
               size, it->first, it->second.size);
    }
    it = evict_immutable(it);
  }
  erase_uploads_overlapping(begin, end);
  g_immutableArrays.emplace(begin,
                            ImmutableArray{.size = size, .le = le, .generation = generation, .setFrame = frameId});
}

void release_array_immutable(uint64_t address, uint32_t generation) noexcept {
  const auto at = static_cast<uintptr_t>(address);
  auto it = g_immutableArrays.upper_bound(at);
  if (it == g_immutableArrays.begin() || at >= std::prev(it)->first + std::prev(it)->second.size) {
    bump(g_staleReleases);
    return;
  }
  --it;
  if (it->second.generation != generation) {
    bump(g_staleReleases);
    if (!std::exchange(g_warnedStaleRelease, true)) {
      Log.warn("Immutable array 0x{:X} released as generation {} while generation {} is declared; evicting it", at,
               generation, it->second.generation);
    }
  }
  evict_immutable(it);
}

void reset_immutable_arrays() noexcept {
  g_immutableArrays.clear();
  g_arena.reset = true;
  g_arena.frameId = 0;
}

void init_immutable_arrays() noexcept {
  const char* value = SDL_getenv("AURORA_IMMUTABLE_ARRAYS");
  const std::string_view mode = value != nullptr ? value : "";
  if (mode == "0") {
    g_immutableMode = ImmutableArrayMode::Off;
  } else if (mode == "validate") {
    g_immutableMode = ImmutableArrayMode::Validate;
  } else {
    g_immutableMode = ImmutableArrayMode::On;
  }
}

ImmutableArrayStats immutable_array_stats() noexcept {
  return {
      .entries = g_immutableArrays.size(),
      .hits = g_residentHits.load(std::memory_order_relaxed),
      .stages = g_residentStages.load(std::memory_order_relaxed),
      .stageBytes = g_residentStageBytes.load(std::memory_order_relaxed),
      .resets = g_residentResets.load(std::memory_order_relaxed),
      .epochs = g_residentEpochs.load(std::memory_order_relaxed),
      .staleReleases = g_staleReleases.load(std::memory_order_relaxed),
      .overlapEvictions = g_overlapEvictions.load(std::memory_order_relaxed),
      .sizeMismatches = g_sizeMismatches.load(std::memory_order_relaxed),
      .arenaFull = g_arenaFull.load(std::memory_order_relaxed),
      .violations = g_residentViolations.load(std::memory_order_relaxed),
      .used = g_arena.used,
      .live = g_arena.live,
  };
}

namespace testing {
void reset_draw_route_stats() noexcept {
  g_routed.store(0, std::memory_order_relaxed);
  g_suppressed.store(0, std::memory_order_relaxed);
  g_mergeBreaks.store(0, std::memory_order_relaxed);
  g_substituted.store(0, std::memory_order_relaxed);
}

void reset_array_upload_stats() noexcept {
  g_arrayReused.store(0, std::memory_order_relaxed);
  g_arrayReusedBytes.store(0, std::memory_order_relaxed);
  g_arrayChanged.store(0, std::memory_order_relaxed);
}

void set_immutable_array_mode(ImmutableArrayMode mode) noexcept { g_immutableMode = mode; }

void reset_immutable_array_stats() noexcept {
  for (auto* counter :
       {&g_residentHits, &g_residentStages, &g_residentStageBytes, &g_residentResets, &g_residentEpochs,
        &g_staleReleases, &g_overlapEvictions, &g_sizeMismatches, &g_arenaFull, &g_residentViolations}) {
    counter->store(0, std::memory_order_relaxed);
  }
  g_warnedOverlap = false;
  g_warnedStaleRelease = false;
}
} // namespace testing

} // namespace aurora::gx::fifo

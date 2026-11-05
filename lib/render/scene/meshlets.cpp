#include "meshlets.hpp"

#include "../graph_bind.hpp"
#include "../jobs.hpp"
#include "../tlsf.hpp"
#include "../../gfx/hash.hpp"
#include "../../gx/command_processor.hpp"
#include "../../gx/vertex_fetch.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/dl.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace aurora::render::scene::meshlets {
namespace {
constexpr Module Log{"aurora::render::retained"};
constexpr uint32_t PoolGranule = ClusterBytes;
constexpr Tlsf::BlockId NoBlock = Tlsf::InvalidBlock;
constexpr uint32_t MaxPieceVertices = 65535;
constexpr uint32_t StateShift = 24;
constexpr uint32_t ReasonShift = 28;

struct KeyHash {
  size_t operator()(const Key& key) const noexcept {
    return static_cast<size_t>(key.lo ^ (key.hi * 0x9E3779B97F4A7C15ull));
  }
};

struct Entry {
  State state = State::None;
  Reason reason = Reason::None;
  uint32_t refs = 0;
  bool statics = false;
  uint64_t ticket = 0;
  Tlsf::BlockId block = NoBlock;
  uint32_t first = 0;
  uint32_t count = 0;
  std::shared_ptr<const CpuGeometry> geometry;
};

struct Job {
  Key key;
  uint64_t ticket = 0;
  uint32_t epoch = 0;
  bool dynamic = false;
  GXPrimitive prim = GX_TRIANGLES;
  uint32_t vtxCount = 0;
  uint8_t stride = 0;
  uint8_t flags = 0;
  bool indexed = false;
  gx::AttrConfig pos{};
  gx::AttrConfig pnmtx{};
  std::vector<uint8_t> records;
  std::vector<uint8_t> indices;
  std::vector<uint8_t> slice;
  uint32_t sliceFirst = 0;
  std::array<Mat3x4<float>, gx::MaxPnMtx> palette{};
  uint32_t currentPnMtx = 0;
};

struct Result {
  Key key;
  uint64_t ticket = 0;
  uint32_t epoch = 0;
  bool dynamic = false;
  State state = State::Ready;
  Reason reason = Reason::None;
  std::vector<uint32_t> block;
  uint32_t clusters = 0;
  std::shared_ptr<const CpuGeometry> geometry;
  uint32_t degenerate = 0;
  uint32_t maxVertices = 0;
  uint32_t maxTriangles = 0;
  bool inlineBuild = false;
};

struct Staged {
  uint64_t stamp = 0;
  uint64_t offset = 0;
  std::vector<uint32_t> words;
};

struct Cached {
  std::shared_ptr<const CpuGeometry> geometry;
  uint64_t used = 0;
};

bool g_active = false;
std::unordered_map<Key, Entry, KeyHash> g_entries;
std::optional<Tlsf> g_pool;
uint64_t g_poolBytes = PoolBytes;
std::vector<Tlsf::BlockId> g_releasing;
std::vector<Tlsf::BlockId> g_released;
std::vector<Result> g_backlog;
jobs::ResultQueue<Result> g_results;
std::unique_ptr<jobs::JobGroup> g_group;
std::unordered_map<Key, Cached, KeyHash> g_cache;
uint32_t g_epoch = 0;
uint64_t g_nextTicket = 1;
uint32_t g_inFlight = 0;
uint32_t g_inFlightDynamic = 0;
uint64_t g_windowBytes = 0;
uint64_t g_windowDynamic = 0;
uint32_t g_pending = 0;
uint32_t g_pendingStatic = 0;
uint32_t g_ready = 0;
uint64_t g_clusters = 0;
uint64_t g_geometryBytes = 0;
uint64_t g_latches = 0;
Stats g_stats;
std::atomic_uint64_t g_recordingFrame{0};

std::mutex g_uploadMutex;
std::vector<Staged> g_staged;
wgpu::Buffer g_buffer;
testing::WriteFn g_write = nullptr;
detail::PersistentHandle g_handle{};
bool g_resourcesLive = false;

std::mutex g_publishMutex;
Stats g_published;

bool waiting(State state) noexcept { return state == State::Deferred || state == State::Building; }

void set_state(Entry& entry, State state, Reason reason = Reason::None) noexcept {
  if (waiting(entry.state)) {
    --g_pending;
    g_pendingStatic -= entry.statics ? 1 : 0;
  }
  if (entry.state == State::Ready) {
    --g_ready;
    g_clusters -= entry.count;
  }
  entry.state = state;
  entry.reason = reason;
  if (waiting(state)) {
    ++g_pending;
    g_pendingStatic += entry.statics ? 1 : 0;
  }
  if (state == State::Ready) {
    ++g_ready;
    g_clusters += entry.count;
  }
}

void mark_static(Entry& entry) noexcept {
  if (!std::exchange(entry.statics, true) && waiting(entry.state)) {
    ++g_pendingStatic;
  }
}

uint64_t geometry_bytes(const CpuGeometry& geometry) noexcept {
  return (geometry.positions.size() + geometry.triangles.size()) * sizeof(uint32_t);
}

ClusterRange range_of(const Entry& entry) noexcept {
  const bool ready = entry.state == State::Ready;
  return {
      .first = ready ? entry.first : 0, .count = ready ? entry.count : 0, .state = entry.state, .reason = entry.reason};
}

void remove_entry(std::unordered_map<Key, Entry, KeyHash>::iterator it) noexcept {
  Entry& entry = it->second;
  set_state(entry, State::None);
  if (entry.block != NoBlock) {
    g_releasing.push_back(entry.block);
  }
  if (entry.geometry) {
    g_cache[it->first] = {.geometry = std::move(entry.geometry), .used = g_latches};
  }
  g_entries.erase(it);
}

bool line_prim(GXPrimitive prim) noexcept { return prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS; }

bool indexed_type(uint8_t type) noexcept { return type == GX_INDEX8 || type == GX_INDEX16; }

uint32_t comp_bytes(uint8_t compType) noexcept {
  switch (compType) {
  case GX_U8:
  case GX_S8:
    return 1;
  case GX_U16:
  case GX_S16:
    return 2;
  default:
    return 4;
  }
}

uint32_t triangle_count(const Source& source) noexcept {
  if (source.indexed) {
    return static_cast<uint32_t>(source.indices.size() / sizeof(uint16_t) / 3);
  }
  const uint32_t n = source.vtxCount;
  switch (source.prim) {
  case GX_TRIANGLES:
    return n / 3;
  case GX_QUADS:
    return n / 4 * 2;
  case GX_TRIANGLESTRIP:
  case GX_TRIANGLEFAN:
    return n >= 3 ? n - 2 : 0;
  default:
    return 0;
  }
}

uint32_t record_index(const uint8_t* at, uint8_t type) noexcept {
  return type == GX_INDEX8 ? at[0] : (uint32_t{at[0]} << 8) | at[1];
}

Key content_key(const Source& source, std::span<const uint8_t> slice) noexcept {
  std::array<uint8_t, 64> head{};
  size_t size = 0;
  const auto put = [&](const void* data, size_t length) {
    std::memcpy(head.data() + size, data, length);
    size += length;
  };
  const uint32_t fields[5] = {static_cast<uint32_t>(source.prim), source.vtxCount, source.vtxStride,
                              static_cast<uint32_t>(source.flags & (piece_flag::Posed | piece_flag::Deformed)),
                              static_cast<uint32_t>(source.indices.size())};
  put(fields, sizeof(fields));
  put(&source.pos, sizeof(source.pos));
  put(&source.pnmtx, sizeof(source.pnmtx));
  const bool indexedPos = indexed_type(source.pos.attrType) && source.posArray != nullptr;
  const uint32_t posLayout[2] = {indexedPos ? source.posArray->stride : 0u,
                                 indexedPos && source.posArray->le ? 1u : 0u};
  put(posLayout, sizeof(posLayout));
  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_128bits_reset(&state);
  XXH3_128bits_update(&state, head.data(), size);
  XXH3_128bits_update(&state, source.records.data(), size_t{source.vtxCount} * source.vtxStride);
  if (!source.indices.empty()) {
    XXH3_128bits_update(&state, source.indices.data(), source.indices.size());
  }
  if (!slice.empty()) {
    XXH3_128bits_update(&state, slice.data(), slice.size());
  }
  const XXH128_hash_t hash = XXH3_128bits_digest(&state);
  Key key{.lo = hash.low64, .hi = hash.high64};
  if (!key.valid()) {
    key.lo = 1;
  }
  return key;
}

std::vector<uint32_t> triangle_list(const Job& in, bool& valid) {
  std::vector<uint32_t> out;
  valid = true;
  if (in.indexed) {
    const size_t count = in.indices.size() / sizeof(uint16_t) / 3 * 3;
    out.resize(count);
    for (size_t i = 0; i < count; ++i) {
      uint16_t index = 0;
      std::memcpy(&index, in.indices.data() + i * sizeof(uint16_t), sizeof(index));
      valid = valid && index < in.vtxCount;
      out[i] = index;
    }
    return out;
  }
  uint32_t n = in.vtxCount;
  n -= in.prim == GX_TRIANGLES ? n % 3 : in.prim == GX_QUADS ? n % 4 : 0;
  out.reserve(size_t{triangle_count({.prim = in.prim, .vtxCount = n})} * 3);
  gx::dl::expand_triangles(in.prim, static_cast<u16>(n),
                           [&](u16 a, u16 b, u16 c) { out.insert(out.end(), {a, b, c}); });
  return out;
}

Result build(const Job& in) noexcept {
  Result out{.key = in.key,
             .ticket = in.ticket,
             .epoch = in.epoch,
             .dynamic = in.dynamic,
             .inlineBuild = !jobs::is_job_thread()};
  const auto fail = [&](State state, Reason reason) {
    out.state = state;
    out.reason = reason;
    return std::move(out);
  };
  gx::ShaderConfig config{};
  config.vtxStride = in.stride;
  config.attrs[GX_VA_POS] = in.pos;
  config.attrs[GX_VA_PNMTXIDX] = in.pnmtx;
  gx::vertex_fetch::CpuVertexSource source{.records = in.records, .currentPnMtx = in.currentPnMtx};
  source.arrays[0] = {.bytes = in.slice, .first = in.sliceFirst};
  const bool posed = (in.flags & piece_flag::Posed) != 0;
  std::vector<float> positions(size_t{in.vtxCount} * 3);
  gx::vertex_fetch::CpuVertex vertex;
  for (uint32_t v = 0; v < in.vtxCount; ++v) {
    if (!gx::vertex_fetch::decode_vertex(config, source, v, vertex)) {
      return fail(State::Failed, Reason::Decode);
    }
    std::array<float, 3> p = vertex.pos;
    if (posed) {
      const Mat3x4<float>& m = in.palette[std::min(vertex.pnMtx, gx::MaxPnMtx - 1)];
      p = {m.m0[0] * vertex.pos[0] + m.m0[1] * vertex.pos[1] + m.m0[2] * vertex.pos[2] + m.m0[3],
           m.m1[0] * vertex.pos[0] + m.m1[1] * vertex.pos[1] + m.m1[2] * vertex.pos[2] + m.m1[3],
           m.m2[0] * vertex.pos[0] + m.m2[1] * vertex.pos[1] + m.m2[2] * vertex.pos[2] + m.m2[3]};
    }
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
      return fail(State::Failed, Reason::Decode);
    }
    std::copy(p.begin(), p.end(), positions.begin() + size_t{v} * 3);
  }

  bool valid = true;
  std::vector<uint32_t> indices = triangle_list(in, valid);
  if (!valid) {
    return fail(State::Failed, Reason::Decode);
  }
  std::vector<uint32_t> remap(in.vtxCount);
  // Identical vertex records decode identically, so strips that share an edge share its vertices
  const size_t unique = meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), in.records.data(),
                                                    in.vtxCount, in.stride);
  std::vector<uint32_t> rep(unique, UINT32_MAX);
  std::vector<float> uniquePositions(unique * 3);
  for (uint32_t v = 0; v < in.vtxCount; ++v) {
    const uint32_t u = remap[v];
    if (u != UINT32_MAX && rep[u] == UINT32_MAX) {
      rep[u] = v;
      std::copy_n(positions.begin() + size_t{v} * 3, 3, uniquePositions.begin() + size_t{u} * 3);
    }
  }
  size_t kept = 0;
  for (size_t t = 0; t + 2 < indices.size(); t += 3) {
    const uint32_t a = remap[indices[t]];
    const uint32_t b = remap[indices[t + 1]];
    const uint32_t c = remap[indices[t + 2]];
    if (a == b || b == c || a == c) {
      ++out.degenerate;
      continue;
    }
    indices[kept++] = a;
    indices[kept++] = b;
    indices[kept++] = c;
  }
  indices.resize(kept);
  if (indices.empty()) {
    return fail(State::NoTriangles, Reason::None);
  }

  const size_t bound = meshopt_buildMeshletsBound(indices.size(), MaxVertices, MaxTriangles);
  std::vector<meshopt_Meshlet> meshlets(bound);
  std::vector<uint32_t> meshletVertices(bound * MaxVertices);
  std::vector<uint8_t> meshletTriangles(bound * MaxTriangles * 3);
  const size_t count = meshopt_buildMeshlets(meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
                                             indices.data(), indices.size(), uniquePositions.data(), unique,
                                             sizeof(float) * 3, MaxVertices, MaxTriangles, ConeWeight);
  meshlets.resize(count);
  size_t refs = 0;
  size_t triangles = 0;
  for (const meshopt_Meshlet& m : meshlets) {
    refs += m.vertex_count;
    triangles += m.triangle_count;
  }
  const uint32_t boundsSpace = (in.flags & (piece_flag::Posed | piece_flag::Deformed)) != 0 ? 1u : 0u;
  std::vector<uint32_t>& block = out.block;
  block.assign(count * ClusterWords + refs + triangles, 0);
  auto refWord = static_cast<uint32_t>(count * ClusterWords);
  auto triWord = static_cast<uint32_t>(refWord + refs);
  for (size_t i = 0; i < count; ++i) {
    const meshopt_Meshlet& m = meshlets[i];
    uint32_t* mv = &meshletVertices[m.vertex_offset];
    uint8_t* mt = &meshletTriangles[m.triangle_offset];
    meshopt_optimizeMeshlet(mv, mt, m.triangle_count, m.vertex_count);
    const meshopt_Bounds bounds =
        meshopt_computeMeshletBounds(mv, mt, m.triangle_count, uniquePositions.data(), unique, sizeof(float) * 3);
    std::array<float, 3> lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
    std::array<float, 3> hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                            -std::numeric_limits<float>::max()};
    for (uint32_t k = 0; k < m.vertex_count; ++k) {
      const float* p = &uniquePositions[size_t{mv[k]} * 3];
      for (int c = 0; c < 3; ++c) {
        lo[c] = std::min(lo[c], p[c]);
        hi[c] = std::max(hi[c], p[c]);
      }
    }
    const auto base = static_cast<uint32_t>(i * ClusterWords);
    uint32_t* rec = &block[base];
    // Offsets are relative to each cluster's own record, so a block uploads anywhere without patching
    rec[cluster::RefRel] = refWord - base;
    rec[cluster::TriRel] = triWord - base;
    rec[cluster::Counts] = m.vertex_count | m.triangle_count << 8 | boundsSpace << cluster::BoundsSpaceShift;
    rec[cluster::Cone] = static_cast<uint8_t>(bounds.cone_axis_s8[0]) |
                         static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[1])) << 8 |
                         static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_axis_s8[2])) << 16 |
                         static_cast<uint32_t>(static_cast<uint8_t>(bounds.cone_cutoff_s8)) << 24;
    for (int c = 0; c < 3; ++c) {
      rec[cluster::Sphere + c] = std::bit_cast<uint32_t>(bounds.center[c]);
      rec[cluster::AabbMin + c] = std::bit_cast<uint32_t>(lo[c]);
      rec[cluster::AabbMax + c] = std::bit_cast<uint32_t>(hi[c]);
    }
    rec[cluster::Sphere + 3] = std::bit_cast<uint32_t>(bounds.radius);
    for (uint32_t k = 0; k < m.vertex_count; ++k) {
      block[refWord + k] = rep[mv[k]];
    }
    for (uint32_t t = 0; t < m.triangle_count; ++t) {
      block[triWord + t] = uint32_t{mt[t * 3]} | uint32_t{mt[t * 3 + 1]} << 8 | uint32_t{mt[t * 3 + 2]} << 16;
    }
    refWord += m.vertex_count;
    triWord += m.triangle_count;
    out.maxVertices = std::max(out.maxVertices, m.vertex_count);
    out.maxTriangles = std::max(out.maxTriangles, m.triangle_count);
  }
  out.clusters = static_cast<uint32_t>(count);

  if ((in.flags & piece_flag::Static) != 0) {
    auto geometry = std::make_shared<CpuGeometry>();
    geometry->positions = std::move(positions);
    geometry->triangles.reserve(indices.size());
    geometry->aabb = {std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),  -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (const uint32_t u : indices) {
      const uint32_t v = rep[u];
      geometry->triangles.push_back(v);
      for (int c = 0; c < 3; ++c) {
        geometry->aabb[c] = std::min(geometry->aabb[c], geometry->positions[size_t{v} * 3 + c]);
        geometry->aabb[3 + c] = std::max(geometry->aabb[3 + c], geometry->positions[size_t{v} * 3 + c]);
      }
    }
    out.geometry = std::move(geometry);
  }
  return out;
}

ClusterRange keyless(Key* inOut, State state, Reason reason, uint64_t& counter) noexcept {
  ++counter;
  if (inOut->valid()) {
    release(*inOut);
    *inOut = {};
  }
  return {.state = state, .reason = reason};
}

void publish() noexcept {
  g_stats.active = g_active;
  g_stats.entries = static_cast<uint32_t>(g_entries.size());
  g_stats.ready = g_ready;
  g_stats.pending = g_pending;
  g_stats.pendingStatic = g_pendingStatic;
  g_stats.inFlight = g_inFlight;
  g_stats.clusters = g_clusters;
  g_stats.poolUsed = g_pool ? g_pool->used() : 0;
  g_stats.cpuGeometryBytes = g_geometryBytes;
  std::lock_guard lock{g_publishMutex};
  g_published = g_stats;
}

void log_stats() {
  const Stats s = published_stats();
  if (s.requests == 0) {
    return;
  }
  Log.info(
      "Retained: meshlets requests {} built {} shared {} deferred {} pending {} ready {} clusters {} pool {} KiB "
      "max {} vertices {} triangles, inline {}, degenerate {}, stale {}",
      s.requests, s.built, s.shared, s.deferred, s.pending, s.ready, s.clusters, s.poolUsed >> 10, s.maxVertices,
      s.maxTriangles, s.inlineBuilds, s.degenerate, s.stale);
  Log.info(
      "Retained: meshlets skipped deformed {} gpu {} churn {}; failed past-array {} decode {} pool {} (retries "
      "{}); no triangles {}; unavailable {}",
      s.skippedDeformedDirect, s.skippedGpuArray, s.skippedChurn, s.failedIndexPastArray, s.failedDecode,
      s.failedPoolFull, s.poolFullRetries, s.noTriangles, s.unavailable);
}
} // namespace

uint32_t count_word(const ClusterRange& range) noexcept {
  return (range.count & 0xFFFFFFu) | static_cast<uint32_t>(range.state) << StateShift |
         static_cast<uint32_t>(range.reason) << ReasonShift;
}

ClusterRange request(const Source& source, Key* inOut, uint8_t* churn) noexcept {
  if (!g_active) {
    return {};
  }
  ++g_stats.requests;
  const bool indexedPos = indexed_type(source.pos.attrType);
  if (line_prim(source.prim) || source.pos.attrType == GX_NONE) {
    return keyless(inOut, State::NoTriangles, Reason::None, g_stats.noTriangles);
  }
  if (indexedPos && source.posArray != nullptr && source.posArray->gpu) {
    return keyless(inOut, State::Skipped, Reason::GpuArray, g_stats.skippedGpuArray);
  }
  // CPU-deformed direct positions are new bytes every frame, and the vertex count moves with them
  if ((source.flags & piece_flag::Deformed) != 0 && source.pos.attrType == GX_DIRECT) {
    return keyless(inOut, State::Skipped, Reason::DeformedDirect, g_stats.skippedDeformedDirect);
  }
  if (*churn >= ChurnLimit) {
    return keyless(inOut, State::Skipped, Reason::Churn, g_stats.skippedChurn);
  }
  if (jobs::worker_count() == 0) {
    return keyless(inOut, State::Unavailable, Reason::None, g_stats.unavailable);
  }
  const size_t recordBytes = size_t{source.vtxCount} * source.vtxStride;
  if (triangle_count(source) == 0 || source.vtxStride == 0 || source.vtxCount > MaxPieceVertices ||
      source.records.size() < recordBytes) {
    return keyless(inOut, State::NoTriangles, Reason::None, g_stats.noTriangles);
  }

  std::span<const uint8_t> slice;
  uint32_t sliceFirst = 0;
  if (indexedPos) {
    if (source.posArray == nullptr || (source.pos.cnt != 2 && source.pos.cnt != 3)) {
      return keyless(inOut, State::Failed, Reason::Decode, g_stats.failedDecode);
    }
    uint32_t lo = UINT32_MAX;
    uint32_t hi = 0;
    const uint32_t idxBytes = source.pos.attrType == GX_INDEX8 ? 1 : 2;
    for (uint32_t v = 0; v < source.vtxCount; ++v) {
      const size_t at = size_t{v} * source.vtxStride + source.pos.offset;
      if (at + idxBytes > source.records.size()) {
        return keyless(inOut, State::Failed, Reason::Decode, g_stats.failedDecode);
      }
      const uint32_t index = record_index(source.records.data() + at, source.pos.attrType);
      lo = std::min(lo, index);
      hi = std::max(hi, index);
    }
    const uint64_t end = uint64_t{hi} * source.pos.stride + uint64_t{comp_bytes(source.pos.compType)} * source.pos.cnt;
    if (end > source.posArray->size) {
      return keyless(inOut, State::Failed, Reason::IndexPastArray, g_stats.failedIndexPastArray);
    }
    sliceFirst = lo * source.pos.stride;
    const auto size = static_cast<uint32_t>(end - sliceFirst);
    slice = gx::fifo::array_bytes(*source.posArray, sliceFirst, size);
    if (slice.size() != size) {
      return keyless(inOut, State::Failed, Reason::Decode, g_stats.failedDecode);
    }
  }

  const bool positionKeyed = (source.flags & (piece_flag::Posed | piece_flag::Deformed)) == 0;
  const Key key = content_key(source, positionKeyed ? slice : std::span<const uint8_t>{});
  if (inOut->valid() && *inOut != key && ++*churn >= ChurnLimit) {
    return keyless(inOut, State::Skipped, Reason::Churn, g_stats.skippedChurn);
  }
  if (*inOut != key) {
    if (inOut->valid()) {
      release(*inOut);
    }
    acquire(key);
    *inOut = key;
  }
  Entry& entry = g_entries[key];
  if ((source.flags & piece_flag::Static) != 0) {
    mark_static(entry);
  }
  if (entry.state != State::None && entry.state != State::Deferred) {
    ++g_stats.shared;
    return range_of(entry);
  }

  const bool dynamic = (source.flags & piece_flag::Static) == 0;
  const uint64_t bytes = recordBytes + source.indices.size() + slice.size();
  const bool first = g_windowBytes == 0;
  const bool jobRoom = g_inFlight < MaxJobsInFlight && (!dynamic || g_inFlightDynamic < MaxJobsNonStatic);
  const bool copyRoom = first || (g_windowBytes + bytes <= CopyBytesPerWindow &&
                                  (!dynamic || g_windowDynamic + bytes <= CopyBytesNonStatic));
  if (!jobRoom || !copyRoom) {
    ++g_stats.deferred;
    set_state(entry, State::Deferred);
    return range_of(entry);
  }

  auto job = std::make_shared<Job>();
  job->key = key;
  job->ticket = g_nextTicket++;
  job->epoch = g_epoch;
  job->dynamic = dynamic;
  job->prim = source.prim;
  job->vtxCount = source.vtxCount;
  job->stride = source.vtxStride;
  job->flags = source.flags;
  job->indexed = source.indexed;
  job->pos = source.pos;
  job->pnmtx = source.pnmtx;
  job->records.assign(source.records.begin(), source.records.begin() + static_cast<ptrdiff_t>(recordBytes));
  job->indices.assign(source.indices.begin(), source.indices.end());
  job->slice.assign(slice.begin(), slice.end());
  job->sliceFirst = sliceFirst;
  job->currentPnMtx = source.currentPnMtx;
  for (size_t i = 0; i < std::min<size_t>(source.palette.size(), gx::MaxPnMtx); ++i) {
    job->palette[i] = source.palette[i].pos;
  }
  entry.ticket = job->ticket;
  set_state(entry, State::Building);
  g_windowBytes += bytes;
  g_windowDynamic += dynamic ? bytes : 0;
  ++g_inFlight;
  g_inFlightDynamic += dynamic ? 1 : 0;
  ++g_stats.submitted;
  if (!g_group) {
    g_group = std::make_unique<jobs::JobGroup>();
  }
  g_group->submit([job](const jobs::JobContext&) { g_results.push(build(*job)); }, jobs::Priority::Low,
                  "retained meshlets");
  return range_of(entry);
}

ClusterRange lookup(const Key& key) noexcept {
  const auto it = g_entries.find(key);
  return it != g_entries.end() ? range_of(it->second) : ClusterRange{};
}

void acquire(const Key& key) noexcept {
  if (!g_active || !key.valid()) {
    return;
  }
  ++g_entries[key].refs;
}

void release(const Key& key) noexcept {
  const auto it = g_entries.find(key);
  if (it == g_entries.end()) {
    return;
  }
  if (it->second.refs > 0 && --it->second.refs == 0) {
    remove_entry(it);
  }
}

std::shared_ptr<const CpuGeometry> cpu_geometry(const Key& key) noexcept {
  if (const auto it = g_entries.find(key); it != g_entries.end() && it->second.geometry) {
    return it->second.geometry;
  }
  if (const auto it = g_cache.find(key); it != g_cache.end()) {
    it->second.used = g_latches;
    return it->second.geometry;
  }
  return nullptr;
}

uint32_t pending_static() noexcept { return g_pendingStatic; }

void initialize() {
  if (webgpu::g_device && g_handle.id == 0) {
    g_handle = detail::create_persistent("scene.clusters", detail::PersistentKind::Buffer);
  }
}

void shutdown() {
  if (g_group) {
    g_group->cancel();
    g_group->wait();
  }
  log_stats();
  latch_resources(false);
  if (g_handle.id != 0) {
    detail::destroy_persistent(g_handle);
    g_handle = {};
  }
  set_active(false);
  g_group.reset();
}

void set_active(bool active) noexcept {
  if (active == g_active) {
    return;
  }
  g_active = active;
  reset();
  if (active) {
    g_pool.emplace(g_poolBytes, PoolGranule);
  } else {
    g_pool.reset();
    g_results.drain([](Result&&) {});
  }
}

void latch_resources(bool active) noexcept {
  std::lock_guard lock{g_uploadMutex};
  if (active == g_resourcesLive) {
    return;
  }
  if (!active) {
    if (g_handle.id != 0) {
      detail::set_persistent_buffer(g_handle, {}, 0, 0);
    }
    g_buffer = {};
    g_staged.clear();
    g_resourcesLive = false;
    return;
  }
  if (!webgpu::g_device || g_handle.id == 0) {
    return;
  }
  const wgpu::BufferDescriptor descriptor{
      .label = "scene.clusters",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
      .size = PoolBytes,
  };
  g_buffer = webgpu::g_device.CreateBuffer(&descriptor);
  detail::set_persistent_buffer(g_handle, g_buffer, 0, PoolBytes);
  g_resourcesLive = true;
}

bool latch() noexcept {
  if (!g_active) {
    return false;
  }
  ++g_latches;
  bool changed = false;
  g_results.drain([&](Result&& result) {
    if (result.epoch != g_epoch) {
      return;
    }
    --g_inFlight;
    g_inFlightDynamic -= result.dynamic ? 1 : 0;
    g_stats.inlineBuilds += result.inlineBuild ? 1 : 0;
    const auto it = g_entries.find(result.key);
    if (it == g_entries.end() || it->second.ticket != result.ticket || it->second.state != State::Building) {
      ++g_stats.stale;
      return;
    }
    Entry& entry = it->second;
    if (result.state != State::Ready) {
      g_stats.failedDecode += result.reason == Reason::Decode ? 1 : 0;
      g_stats.noTriangles += result.state == State::NoTriangles ? 1 : 0;
      set_state(entry, result.state, result.reason);
      changed = true;
      return;
    }
    ++g_stats.built;
    g_stats.degenerate += result.degenerate;
    g_stats.maxVertices = std::max(g_stats.maxVertices, result.maxVertices);
    g_stats.maxTriangles = std::max(g_stats.maxTriangles, result.maxTriangles);
    if (result.geometry) {
      g_geometryBytes += geometry_bytes(*result.geometry);
      entry.geometry = result.geometry;
    }
    g_backlog.push_back(std::move(result));
  });

  const uint64_t stamp = g_recordingFrame.load(std::memory_order_acquire);
  std::vector<Staged> staged;
  uint64_t stagedBytes = 0;
  size_t kept = 0;
  for (size_t i = 0; i < g_backlog.size(); ++i) {
    Result& result = g_backlog[i];
    const auto it = g_entries.find(result.key);
    if (it == g_entries.end() || it->second.ticket != result.ticket || it->second.state != State::Building) {
      ++g_stats.stale;
      continue;
    }
    const uint64_t bytes = result.block.size() * sizeof(uint32_t);
    if (!staged.empty() && stagedBytes + bytes > UploadBytesPerLatch) {
      if (kept != i) {
        g_backlog[kept] = std::move(result);
      }
      ++kept;
      continue;
    }
    Entry& entry = it->second;
    const auto allocation = g_pool->allocate(bytes);
    if (!allocation) {
      ++g_stats.failedPoolFull;
      set_state(entry, State::Failed, Reason::PoolFull);
      changed = true;
      continue;
    }
    entry.block = allocation->block;
    entry.first = static_cast<uint32_t>(allocation->offset / ClusterBytes);
    entry.count = result.clusters;
    set_state(entry, State::Ready);
    changed = true;
    stagedBytes += bytes;
    ++g_stats.uploads;
    g_stats.uploadBytes += bytes;
    staged.push_back({.stamp = stamp, .offset = allocation->offset, .words = std::move(result.block)});
  }
  g_backlog.resize(kept);
  if (!staged.empty()) {
    std::lock_guard lock{g_uploadMutex};
    std::ranges::move(staged, std::back_inserter(g_staged));
  }

  const bool returned = !g_released.empty();
  for (const Tlsf::BlockId block : g_released) {
    g_pool->free(block);
  }
  // A block freed this latch may still be read by the frame in flight, so it returns to the allocator one latch later
  g_released = std::exchange(g_releasing, {});
  if (returned) {
    for (auto& [key, entry] : g_entries) {
      if (entry.state == State::Failed && entry.reason == Reason::PoolFull) {
        set_state(entry, State::None);
        ++g_stats.poolFullRetries;
        changed = true;
      }
    }
  }
  while (g_geometryBytes > CpuGeometryBytes && !g_cache.empty()) {
    const auto oldest = std::ranges::min_element(g_cache, {}, [](const auto& item) { return item.second.used; });
    g_geometryBytes -= geometry_bytes(*oldest->second.geometry);
    g_cache.erase(oldest);
  }
  g_windowBytes = 0;
  g_windowDynamic = 0;
  publish();
  return changed;
}

void reset() noexcept {
  ++g_epoch;
  g_entries.clear();
  g_cache.clear();
  g_backlog.clear();
  g_releasing.clear();
  g_released.clear();
  if (g_pool) {
    g_pool.emplace(g_poolBytes, PoolGranule);
  }
  g_inFlight = 0;
  g_inFlightDynamic = 0;
  g_windowBytes = 0;
  g_windowDynamic = 0;
  g_pending = 0;
  g_pendingStatic = 0;
  g_ready = 0;
  g_clusters = 0;
  g_geometryBytes = 0;
  std::lock_guard lock{g_uploadMutex};
  g_staged.clear();
}

void set_recording_frame(uint64_t frameId) noexcept { g_recordingFrame.store(frameId, std::memory_order_release); }

Stats published_stats() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_published;
}

detail::PersistentHandle clusters_handle() noexcept { return g_handle; }

void worker_frame_begin(uint64_t frameId) {
  std::vector<Staged> due;
  wgpu::Buffer buffer;
  testing::WriteFn write = nullptr;
  {
    std::lock_guard lock{g_uploadMutex};
    if (g_staged.empty()) {
      return;
    }
    const auto split = std::stable_partition(g_staged.begin(), g_staged.end(),
                                             [&](const Staged& upload) { return upload.stamp < frameId; });
    due.assign(std::make_move_iterator(g_staged.begin()), std::make_move_iterator(split));
    g_staged.erase(g_staged.begin(), split);
    buffer = g_buffer;
    write = g_write;
  }
  for (const Staged& upload : due) {
    if (write != nullptr) {
      write(upload.offset, upload.words);
    } else if (buffer) {
      webgpu::g_queue.WriteBuffer(buffer, upload.offset, upload.words.data(), upload.words.size() * sizeof(uint32_t));
    }
  }
}

namespace testing {
void reset() noexcept {
  if (g_group) {
    g_group->wait();
  }
  set_active(false);
  g_group.reset();
  g_stats = {};
  g_poolBytes = PoolBytes;
  g_latches = 0;
  g_nextTicket = 1;
  g_recordingFrame.store(0, std::memory_order_relaxed);
  {
    std::lock_guard lock{g_uploadMutex};
    g_write = nullptr;
  }
  std::lock_guard lock{g_publishMutex};
  g_published = {};
}

void set_write(WriteFn write) noexcept {
  std::lock_guard lock{g_uploadMutex};
  g_write = write;
}

void set_pool_bytes(uint64_t bytes) noexcept {
  g_poolBytes = bytes;
  if (g_pool) {
    g_pool.emplace(g_poolBytes, PoolGranule);
  }
}

std::vector<Upload> staged() noexcept {
  std::lock_guard lock{g_uploadMutex};
  std::vector<Upload> out;
  for (const Staged& upload : g_staged) {
    out.push_back(
        {.stamp = upload.stamp, .offset = upload.offset, .words = static_cast<uint32_t>(upload.words.size())});
  }
  return out;
}

std::vector<uint32_t> block(const Key& key) noexcept {
  const auto it = g_entries.find(key);
  if (it == g_entries.end() || it->second.state != State::Ready) {
    return {};
  }
  std::lock_guard lock{g_uploadMutex};
  for (const Staged& upload : g_staged) {
    if (upload.offset == uint64_t{it->second.first} * ClusterBytes) {
      return upload.words;
    }
  }
  return {};
}

uint32_t refs(const Key& key) noexcept {
  const auto it = g_entries.find(key);
  return it != g_entries.end() ? it->second.refs : 0;
}
} // namespace testing
} // namespace aurora::render::scene::meshlets

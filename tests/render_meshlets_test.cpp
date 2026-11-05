#include <gtest/gtest.h>

#include "gx/guest_memory.hpp"
#include "gx/gx.hpp"
#include "render/jobs.hpp"
#include "render/scene/meshlets.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <span>
#include <thread>
#include <vector>

namespace aurora::render::scene::meshlets {
namespace {
constexpr gx::AttrConfig Index16Pos{
    .attrType = GX_INDEX16, .cnt = 3, .compType = GX_F32, .offset = 0, .stride = 12, .le = false};
constexpr gx::AttrConfig DirectPos{.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32, .offset = 0};

void put_be16(std::vector<uint8_t>& out, size_t at, uint16_t value) {
  out[at] = static_cast<uint8_t>(value >> 8);
  out[at + 1] = static_cast<uint8_t>(value);
}

void put_be32(std::vector<uint8_t>& out, size_t at, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out[at + i] = static_cast<uint8_t>(value >> (24 - 8 * i));
  }
}

struct Piece {
  GXPrimitive prim = GX_TRIANGLES;
  uint32_t vtxCount = 0;
  uint8_t stride = 2;
  uint8_t flags = 0;
  gx::AttrConfig pos = Index16Pos;
  gx::AttrConfig pnmtx{};
  std::vector<uint8_t> records;
  std::vector<uint8_t> positions;
  std::vector<std::array<float, 3>> points;
  std::vector<uint32_t> refs;
  std::vector<uint8_t> indices;
  std::array<gx::PnMtx, gx::MaxPnMtx> palette{};
  gx::AttrArray array{};
  const void* address = nullptr;

  void set_points(std::vector<std::array<float, 3>> values) {
    points = std::move(values);
    positions.assign(points.size() * 12, 0);
    for (size_t i = 0; i < points.size(); ++i) {
      for (int c = 0; c < 3; ++c) {
        put_be32(positions, i * 12 + c * 4, std::bit_cast<uint32_t>(points[i][c]));
      }
    }
  }

  void set_refs(std::vector<uint32_t> values) {
    refs = std::move(values);
    vtxCount = static_cast<uint32_t>(refs.size());
    records.assign(refs.size() * stride, 0);
    for (size_t v = 0; v < refs.size(); ++v) {
      put_be16(records, v * stride + pos.offset, static_cast<uint16_t>(refs[v]));
    }
  }

  Source source() {
    array = {.data = address != nullptr ? address : positions.data(),
             .size = static_cast<u32>(positions.size()),
             .stride = 12,
             .le = false};
    for (auto& m : palette) {
      if (m.pos.m0[0] == 0.f && m.pos.m1[1] == 0.f && m.pos.m2[2] == 0.f) {
        m.pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
      }
    }
    return {.prim = prim,
            .vtxCount = vtxCount,
            .vtxStride = stride,
            .records = records,
            .pos = pos,
            .pnmtx = pnmtx,
            .posArray = &array,
            .palette = palette,
            .flags = flags,
            .indexed = !indices.empty(),
            .indices = indices};
  }
};

Piece grid(uint32_t n, float y = 0.f) {
  Piece piece;
  piece.prim = GX_QUADS;
  std::vector<std::array<float, 3>> points;
  for (uint32_t z = 0; z <= n; ++z) {
    for (uint32_t x = 0; x <= n; ++x) {
      points.push_back({static_cast<float>(x), y, static_cast<float>(z)});
    }
  }
  piece.set_points(std::move(points));
  std::vector<uint32_t> refs;
  for (uint32_t z = 0; z < n; ++z) {
    for (uint32_t x = 0; x < n; ++x) {
      const uint32_t v0 = z * (n + 1) + x;
      refs.insert(refs.end(), {v0, v0 + 1, v0 + n + 2, v0 + n + 1});
    }
  }
  piece.set_refs(std::move(refs));
  return piece;
}

Piece strip(uint32_t count, GXPrimitive prim = GX_TRIANGLESTRIP, float offset = 0.f) {
  Piece piece;
  piece.prim = prim;
  std::vector<std::array<float, 3>> points;
  std::vector<uint32_t> refs;
  for (uint32_t k = 0; k < count; ++k) {
    const float angle = static_cast<float>(k) * 0.3f;
    points.push_back({static_cast<float>(k / 2) + offset + std::cos(angle) * 0.1f, static_cast<float>(k % 2),
                      std::sin(angle) * 0.25f});
    refs.push_back(k);
  }
  piece.set_points(std::move(points));
  piece.set_refs(std::move(refs));
  return piece;
}

struct Cluster {
  uint32_t boundsSpace = 0;
  std::vector<uint32_t> refs;
  std::vector<std::array<uint32_t, 3>> triangles;
  std::array<float, 3> lo{};
  std::array<float, 3> hi{};
  std::array<float, 4> sphere{};
};

std::vector<Cluster> decode(const std::vector<uint32_t>& block, uint32_t count, uint32_t baseWord = 0) {
  std::vector<Cluster> out;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t base = baseWord + i * ClusterWords;
    const uint32_t refWord = base + block[base + cluster::RefRel];
    const uint32_t triWord = base + block[base + cluster::TriRel];
    const uint32_t counts = block[base + cluster::Counts];
    Cluster& c = out.emplace_back();
    c.boundsSpace = counts >> cluster::BoundsSpaceShift & 0xFF;
    for (uint32_t k = 0; k < (counts & 0xFF); ++k) {
      c.refs.push_back(block[refWord + k]);
    }
    for (uint32_t t = 0; t < (counts >> 8 & 0xFF); ++t) {
      const uint32_t w = block[triWord + t];
      c.triangles.push_back({c.refs.at(w & 0xFF), c.refs.at(w >> 8 & 0xFF), c.refs.at(w >> 16 & 0xFF)});
    }
    for (int k = 0; k < 3; ++k) {
      c.lo[k] = std::bit_cast<float>(block[base + cluster::AabbMin + k]);
      c.hi[k] = std::bit_cast<float>(block[base + cluster::AabbMax + k]);
    }
    for (int k = 0; k < 4; ++k) {
      c.sphere[k] = std::bit_cast<float>(block[base + cluster::Sphere + k]);
    }
  }
  return out;
}

std::array<uint32_t, 3> canonical(std::array<uint32_t, 3> t) {
  while (t[0] > t[1] || t[0] > t[2]) {
    t = {t[1], t[2], t[0]};
  }
  return t;
}

std::vector<std::array<uint32_t, 3>> raster_triangles(GXPrimitive prim, uint32_t n) {
  std::vector<std::array<uint32_t, 3>> out;
  if (prim == GX_TRIANGLES) {
    for (uint32_t v = 0; v + 2 < n; v += 3) {
      out.push_back({v, v + 1, v + 2});
    }
  } else if (prim == GX_QUADS) {
    for (uint32_t v = 0; v + 3 < n; v += 4) {
      out.push_back({v, v + 1, v + 2});
      out.push_back({v + 2, v + 3, v});
    }
  } else {
    for (uint32_t v = 2; v < n; ++v) {
      if (prim == GX_TRIANGLEFAN) {
        out.push_back({0, v - 1, v});
      } else if ((v & 1) == 0) {
        out.push_back({v - 2, v - 1, v});
      } else {
        out.push_back({v - 1, v - 2, v});
      }
    }
  }
  return out;
}

class RetainedMeshlets : public ::testing::Test {
protected:
  void SetUp() override {
    jobs::initialize(1);
    jobs::set_synchronous(true);
    testing::reset();
    set_active(true);
  }

  void TearDown() override {
    testing::reset();
    jobs::shutdown();
  }

  struct Built {
    Key key;
    ClusterRange range;
    std::vector<uint32_t> block;
  };

  Built build(Piece& piece) {
    Built out;
    uint8_t churn = 0;
    out.range = request(piece.source(), &out.key, &churn);
    (void)latch();
    out.range = lookup(out.key);
    out.block = testing::block(out.key);
    return out;
  }
};

TEST_F(RetainedMeshlets, StripMatchesRasterTopology) {
  for (const auto& [prim, count] : std::vector<std::pair<GXPrimitive, uint32_t>>{
           {GX_TRIANGLESTRIP, 30}, {GX_TRIANGLEFAN, 17}, {GX_TRIANGLES, 10}, {GX_QUADS, 14}}) {
    Piece piece = strip(count, prim);
    const Built built = build(piece);
    ASSERT_EQ(built.range.state, State::Ready) << prim;
    std::multiset<std::array<uint32_t, 3>> got;
    for (const Cluster& c : decode(built.block, built.range.count)) {
      for (const auto& t : c.triangles) {
        got.insert(canonical(t));
      }
    }
    std::multiset<std::array<uint32_t, 3>> want;
    for (const auto& t : raster_triangles(prim, count)) {
      want.insert(canonical(t));
    }
    EXPECT_EQ(got, want) << prim;
    testing::reset();
    set_active(true);
  }
}

TEST_F(RetainedMeshlets, EveryTriangleOnce) {
  Piece piece = grid(20);
  const Built built = build(piece);
  ASSERT_EQ(built.range.state, State::Ready);
  ASSERT_GT(built.range.count, 1u);
  std::map<std::array<uint32_t, 3>, int> got;
  for (const Cluster& c : decode(built.block, built.range.count)) {
    EXPECT_LE(c.refs.size(), MaxVertices);
    EXPECT_LE(c.triangles.size(), MaxTriangles);
    for (const auto& t : c.triangles) {
      ++got[canonical({piece.refs[t[0]], piece.refs[t[1]], piece.refs[t[2]]})];
    }
  }
  size_t triangles = 0;
  for (const auto& t : raster_triangles(GX_QUADS, piece.vtxCount)) {
    const auto key = canonical({piece.refs[t[0]], piece.refs[t[1]], piece.refs[t[2]]});
    EXPECT_EQ(got[key], 1) << key[0] << " " << key[1] << " " << key[2];
    ++triangles;
  }
  EXPECT_EQ(triangles, 800u);
  EXPECT_EQ(got.size(), 800u);
  const Stats stats = published_stats();
  EXPECT_LE(stats.maxVertices, MaxVertices);
  EXPECT_LE(stats.maxTriangles, MaxTriangles);
  EXPECT_EQ(stats.degenerate, 0u);
}

TEST_F(RetainedMeshlets, BoundsContainVertices) {
  Piece piece = grid(16);
  for (auto& p : piece.points) {
    p[1] = std::sin(p[0] * 0.7f) * 3.f + std::cos(p[2] * 0.4f);
  }
  piece.set_points(piece.points);
  const Built built = build(piece);
  ASSERT_EQ(built.range.state, State::Ready);
  for (const Cluster& c : decode(built.block, built.range.count)) {
    EXPECT_EQ(c.boundsSpace, 0u);
    for (const uint32_t ref : c.refs) {
      const auto& p = piece.points[piece.refs[ref]];
      double d2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        EXPECT_LE(c.lo[k], p[k]);
        EXPECT_GE(c.hi[k], p[k]);
        d2 += (p[k] - c.sphere[k]) * (p[k] - c.sphere[k]);
      }
      EXPECT_LE(std::sqrt(d2), c.sphere[3] * (1.0 + 1e-5) + 1e-5);
    }
  }
}

TEST_F(RetainedMeshlets, SharedRecordsShareVertices) {
  Piece pair = strip(4);
  pair.prim = GX_TRIANGLES;
  pair.set_refs({0, 1, 2, 2, 1, 3});
  const Built built = build(pair);
  ASSERT_EQ(built.range.state, State::Ready);
  const auto clusters = decode(built.block, built.range.count);
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].refs.size(), 4u);
  EXPECT_EQ(clusters[0].triangles.size(), 2u);

  Piece mesh = grid(12);
  const Built gridBuilt = build(mesh);
  ASSERT_EQ(gridBuilt.range.state, State::Ready);
  size_t refs = 0;
  for (const Cluster& c : decode(gridBuilt.block, gridBuilt.range.count)) {
    std::set<uint32_t> records;
    for (const uint32_t ref : c.refs) {
      EXPECT_TRUE(records.insert(mesh.refs[ref]).second);
    }
    refs += c.refs.size();
  }
  EXPECT_LT(refs, mesh.vtxCount);
}

TEST_F(RetainedMeshlets, BlockIsDeterministicAndRelocatable) {
  Piece first = grid(9);
  Piece other = strip(40);
  const Built a = build(first);
  const Built b = build(other);
  ASSERT_EQ(a.range.state, State::Ready);
  ASSERT_EQ(b.range.state, State::Ready);
  const Key key = a.key;
  const std::vector<uint32_t> block = a.block;
  const uint32_t at = a.range.first;

  testing::reset();
  set_active(true);
  const Built otherAgain = build(other);
  const Built again = build(first);
  ASSERT_EQ(again.range.state, State::Ready);
  EXPECT_EQ(again.key, key);
  EXPECT_NE(again.range.first, at);
  EXPECT_EQ(again.block, block);
  EXPECT_EQ(otherAgain.block, b.block);

  std::vector<uint32_t> pool(uint64_t{again.range.first + again.range.count} * ClusterWords + block.size(), 0);
  std::copy(block.begin(), block.end(), pool.begin() + uint64_t{again.range.first} * ClusterWords);
  const auto placed = decode(pool, again.range.count, again.range.first * ClusterWords);
  const auto original = decode(block, again.range.count);
  ASSERT_EQ(placed.size(), original.size());
  for (size_t i = 0; i < placed.size(); ++i) {
    EXPECT_EQ(placed[i].refs, original[i].refs);
    EXPECT_EQ(placed[i].triangles, original[i].triangles);
  }
}

TEST_F(RetainedMeshlets, PositionKeysCoverPositions) {
  const auto key_of = [](Piece& piece) {
    Key key;
    uint8_t churn = 0;
    (void)request(piece.source(), &key, &churn);
    return key;
  };
  for (const uint8_t flags : {uint8_t{0}, piece_flag::Static, piece_flag::Posed, piece_flag::Deformed}) {
    Piece a = grid(4);
    Piece b = grid(4, 1.f);
    a.flags = flags;
    b.flags = flags;
    const Key ka = key_of(a);
    const Key kb = key_of(b);
    ASSERT_TRUE(ka.valid());
    const bool positionKeyed = (flags & (piece_flag::Posed | piece_flag::Deformed)) == 0;
    EXPECT_EQ(ka != kb, positionKeyed) << int{flags};
    (void)latch();
    for (const Cluster& c : decode(testing::block(ka), lookup(ka).count)) {
      EXPECT_EQ(c.boundsSpace, positionKeyed ? 0u : 1u) << int{flags};
    }
  }

  const auto direct = [](float y) {
    Piece piece;
    piece.prim = GX_TRIANGLES;
    piece.pos = DirectPos;
    piece.stride = 12;
    piece.vtxCount = 3;
    piece.records.assign(36, 0);
    for (uint32_t v = 0; v < 3; ++v) {
      const std::array<float, 3> p{static_cast<float>(v), y, static_cast<float>(v * v)};
      for (int c = 0; c < 3; ++c) {
        put_be32(piece.records, v * 12 + c * 4, std::bit_cast<uint32_t>(p[c]));
      }
    }
    return piece;
  };
  Piece da = direct(0.f);
  Piece db = direct(2.f);
  Piece dc = direct(0.f);
  const Key ka = key_of(da);
  EXPECT_NE(ka, key_of(db));
  EXPECT_EQ(ka, key_of(dc));
  (void)latch();
  const auto clusters = decode(testing::block(ka), lookup(ka).count);
  ASSERT_EQ(clusters.size(), 1u);
  EXPECT_EQ(clusters[0].boundsSpace, 0u);
  EXPECT_EQ(clusters[0].hi[2], 4.f);
}

TEST_F(RetainedMeshlets, RigidPiecesWithSharedIndicesGetTheirOwnBlocks) {
  Piece a = grid(6);
  Piece b = grid(6, 5.f);
  ASSERT_EQ(a.records, b.records);
  const Built ba = build(a);
  const Built bb = build(b);
  ASSERT_EQ(ba.range.state, State::Ready);
  ASSERT_EQ(bb.range.state, State::Ready);
  EXPECT_NE(ba.key, bb.key);
  EXPECT_NE(ba.range.first, bb.range.first);
  EXPECT_EQ(testing::refs(ba.key), 1u);
  EXPECT_EQ(decode(ba.block, ba.range.count)[0].lo[1], 0.f);
  EXPECT_EQ(decode(bb.block, bb.range.count)[0].lo[1], 5.f);
}

TEST_F(RetainedMeshlets, DeformedDirectPositionsAreSkipped) {
  Piece piece;
  piece.prim = GX_TRIANGLES;
  piece.pos = DirectPos;
  piece.stride = 12;
  piece.vtxCount = 3;
  piece.records.assign(36, 0);
  piece.flags = piece_flag::Deformed | piece_flag::Static;
  Key key;
  uint8_t churn = 0;
  const ClusterRange range = request(piece.source(), &key, &churn);
  EXPECT_EQ(range.state, State::Skipped);
  EXPECT_EQ(range.reason, Reason::DeformedDirect);
  EXPECT_FALSE(key.valid());
  EXPECT_EQ(pending_static(), 0u);
  (void)latch();
  EXPECT_TRUE(testing::staged().empty());
  EXPECT_EQ(published_stats().skippedDeformedDirect, 1u);
  EXPECT_EQ(published_stats().submitted, 0u);

  Piece indexed = grid(3);
  indexed.flags = piece_flag::Deformed;
  EXPECT_EQ(build(indexed).range.state, State::Ready);
}

TEST_F(RetainedMeshlets, GpuArraysAreSkipped) {
  Piece piece = grid(3);
  Source source = piece.source();
  piece.array.gpu = true;
  Key key;
  uint8_t churn = 0;
  const ClusterRange range = request(source, &key, &churn);
  EXPECT_EQ(range.state, State::Skipped);
  EXPECT_EQ(range.reason, Reason::GpuArray);
  EXPECT_FALSE(key.valid());
  EXPECT_EQ(published_stats().submitted, 0u);
}

TEST_F(RetainedMeshlets, ChurningPiecesStopBuilding) {
  Key key;
  uint8_t churn = 0;
  Piece a = grid(3);
  Piece b = grid(3, 1.f);
  Piece c = grid(3, 2.f);
  EXPECT_EQ(request(a.source(), &key, &churn).state, State::Building);
  const Key first = key;
  EXPECT_EQ(request(b.source(), &key, &churn).state, State::Building);
  EXPECT_EQ(churn, 1u);
  EXPECT_EQ(testing::refs(first), 0u);
  const ClusterRange stopped = request(c.source(), &key, &churn);
  EXPECT_EQ(stopped.state, State::Skipped);
  EXPECT_EQ(stopped.reason, Reason::Churn);
  EXPECT_FALSE(key.valid());
  EXPECT_EQ(request(a.source(), &key, &churn).reason, Reason::Churn);
  (void)latch();
  EXPECT_EQ(published_stats().submitted, 2u);
  EXPECT_EQ(published_stats().entries, 0u);
}

struct GuestRead {
  gx::guest_memory::Kind kind;
  uint64_t address = 0;
  size_t size = 0;
  std::thread::id thread;
};

std::vector<GuestRead> g_reads;
std::vector<uint8_t> g_guest;
constexpr uint64_t FakeAddress = 0x7ABC00000000ull;

std::span<const uint8_t> resolve_guest(void*, gx::guest_memory::Kind kind, uint64_t address, size_t size) noexcept {
  g_reads.push_back({kind, address, size, std::this_thread::get_id()});
  if (address < FakeAddress || address + size > FakeAddress + g_guest.size()) {
    return {};
  }
  return {g_guest.data() + (address - FakeAddress), size};
}

TEST_F(RetainedMeshlets, PositionsReadThroughGuestMemory) {
  Piece real = grid(5);
  real.set_refs({7, 8, 14, 13, 8, 9, 15, 14});
  const Built reference = build(real);
  ASSERT_EQ(reference.range.state, State::Ready);
  testing::reset();
  set_active(true);

  Piece guest = grid(5);
  guest.set_refs({7, 8, 14, 13, 8, 9, 15, 14});
  g_guest = guest.positions;
  guest.address = reinterpret_cast<const void*>(static_cast<uintptr_t>(FakeAddress));
  g_reads.clear();
  const gx::guest_memory::Hooks hooks{.resolve = &resolve_guest};
  gx::guest_memory::set_hooks(&hooks);
  const Built hooked = build(guest);
  gx::guest_memory::set_hooks(nullptr);
  ASSERT_EQ(hooked.range.state, State::Ready);
  EXPECT_EQ(hooked.key, reference.key);
  EXPECT_EQ(hooked.block, reference.block);
  ASSERT_EQ(g_reads.size(), 1u);
  EXPECT_EQ(g_reads[0].kind, gx::guest_memory::Kind::VertexArray);
  EXPECT_EQ(g_reads[0].address, FakeAddress + 7 * 12);
  EXPECT_EQ(g_reads[0].size, (15 - 7) * 12 + 12);
  EXPECT_EQ(g_reads[0].thread, std::this_thread::get_id());
}

TEST_F(RetainedMeshlets, PalettePiecesArePosed) {
  Piece piece = grid(4);
  piece.stride = 3;
  piece.pos.offset = 1;
  piece.pnmtx = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 0};
  piece.flags = piece_flag::Posed;
  piece.set_refs(piece.refs);
  for (uint32_t v = 0; v < piece.vtxCount; ++v) {
    piece.records[v * 3] = piece.points[piece.refs[v]][0] >= 2.f ? 3 : 0;
  }
  piece.palette[1].pos = {{1.f, 0.f, 0.f, 100.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, -7.f}};
  const Built built = build(piece);
  ASSERT_EQ(built.range.state, State::Ready);
  std::array<float, 3> lo{INFINITY, INFINITY, INFINITY};
  std::array<float, 3> hi{-INFINITY, -INFINITY, -INFINITY};
  for (const Cluster& c : decode(built.block, built.range.count)) {
    EXPECT_EQ(c.boundsSpace, 1u);
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], c.lo[k]);
      hi[k] = std::max(hi[k], c.hi[k]);
    }
  }
  EXPECT_EQ(lo[0], 0.f);
  EXPECT_EQ(hi[0], 104.f);
  EXPECT_EQ(lo[2], -7.f);
  EXPECT_EQ(hi[2], 4.f);
  EXPECT_EQ(cpu_geometry(built.key), nullptr);
}

TEST_F(RetainedMeshlets, LinesAndPointsGetNoClusters) {
  for (const GXPrimitive prim : {GX_LINES, GX_LINESTRIP, GX_POINTS}) {
    Piece piece = strip(6, prim);
    piece.flags = piece_flag::Static;
    Key key;
    uint8_t churn = 0;
    const ClusterRange range = request(piece.source(), &key, &churn);
    EXPECT_EQ(range.state, State::NoTriangles) << prim;
    EXPECT_FALSE(key.valid());
  }
  (void)latch();
  EXPECT_EQ(published_stats().submitted, 0u);
  EXPECT_EQ(published_stats().noTriangles, 3u);
  EXPECT_EQ(pending_static(), 0u);
}

TEST_F(RetainedMeshlets, CapsDeferAndPoolFullRetries) {
  std::vector<Piece> pieces;
  pieces.reserve(MaxJobsNonStatic + 2);
  for (uint32_t i = 0; i < MaxJobsNonStatic + 2; ++i) {
    pieces.push_back(strip(8, GX_TRIANGLESTRIP, static_cast<float>(i)));
  }
  std::vector<Key> keys(pieces.size());
  std::vector<uint8_t> churn(pieces.size());
  for (uint32_t i = 0; i < MaxJobsNonStatic; ++i) {
    EXPECT_EQ(request(pieces[i].source(), &keys[i], &churn[i]).state, State::Building) << i;
  }
  EXPECT_EQ(request(pieces[MaxJobsNonStatic].source(), &keys[MaxJobsNonStatic], &churn[MaxJobsNonStatic]).state,
            State::Deferred);
  pieces[MaxJobsNonStatic + 1].flags = piece_flag::Static;
  EXPECT_EQ(
      request(pieces[MaxJobsNonStatic + 1].source(), &keys[MaxJobsNonStatic + 1], &churn[MaxJobsNonStatic + 1]).state,
      State::Building);
  (void)latch();
  EXPECT_EQ(request(pieces[MaxJobsNonStatic].source(), &keys[MaxJobsNonStatic], &churn[MaxJobsNonStatic]).state,
            State::Building);

  Piece huge = strip(30000);
  Piece next = strip(12000, GX_TRIANGLESTRIP, 1.f);
  Key hugeKey;
  Key nextKey;
  uint8_t c0 = 0;
  uint8_t c1 = 0;
  (void)latch();
  EXPECT_EQ(request(huge.source(), &hugeKey, &c0).state, State::Building);
  EXPECT_EQ(request(next.source(), &nextKey, &c1).state, State::Deferred);
  next.flags = piece_flag::Static;
  EXPECT_EQ(request(next.source(), &nextKey, &c1).state, State::Building);
  (void)latch();
  EXPECT_EQ(published_stats().deferred, 2u);
  EXPECT_EQ(lookup(hugeKey).state, State::Ready);

  Piece small = strip(20);
  Piece large = grid(6);
  const auto granules = [](const Built& built) { return (built.block.size() * 4 + ClusterBytes - 1) / ClusterBytes; };
  testing::reset();
  set_active(true);
  const uint64_t smallBytes = granules(build(small)) * ClusterBytes;
  const uint64_t largeBytes = granules(build(large)) * ClusterBytes;
  testing::reset();
  set_active(true);
  testing::set_pool_bytes(smallBytes + largeBytes - ClusterBytes);
  const Built a = build(small);
  ASSERT_EQ(a.range.state, State::Ready);
  Key largeKey;
  uint8_t largeChurn = 0;
  (void)request(large.source(), &largeKey, &largeChurn);
  (void)latch();
  EXPECT_EQ(lookup(largeKey).state, State::Failed);
  EXPECT_EQ(lookup(largeKey).reason, Reason::PoolFull);
  release(a.key);
  (void)latch();
  EXPECT_EQ(lookup(largeKey).reason, Reason::PoolFull);
  EXPECT_TRUE(latch());
  EXPECT_EQ(lookup(largeKey).state, State::None);
  EXPECT_EQ(published_stats().poolFullRetries, 1u);
  EXPECT_EQ(request(large.source(), &largeKey, &largeChurn).state, State::Building);
  (void)latch();
  EXPECT_EQ(lookup(largeKey).state, State::Ready);
}

TEST_F(RetainedMeshlets, PendingStaticCountsOnlyWaitingStatic) {
  Piece dynamicPiece = grid(3);
  Piece staticPiece = grid(3, 1.f);
  staticPiece.flags = piece_flag::Static;
  Key dk;
  Key sk;
  uint8_t dc = 0;
  uint8_t sc = 0;
  EXPECT_EQ(request(dynamicPiece.source(), &dk, &dc).state, State::Building);
  EXPECT_EQ(pending_static(), 0u);
  EXPECT_EQ(request(staticPiece.source(), &sk, &sc).state, State::Building);
  EXPECT_EQ(pending_static(), 1u);
  (void)latch();
  EXPECT_EQ(lookup(sk).state, State::Ready);
  EXPECT_EQ(pending_static(), 0u);

  Piece shared = grid(5);
  Key a;
  Key b;
  uint8_t ca = 0;
  uint8_t cb = 0;
  EXPECT_EQ(request(shared.source(), &a, &ca).state, State::Building);
  EXPECT_EQ(pending_static(), 0u);
  shared.flags = piece_flag::Static;
  EXPECT_EQ(request(shared.source(), &b, &cb).state, State::Building);
  EXPECT_EQ(a, b);
  EXPECT_EQ(testing::refs(a), 2u);
  EXPECT_EQ(pending_static(), 1u);
  release(b);
  release(a);
  EXPECT_EQ(pending_static(), 0u);
  (void)latch();
  EXPECT_EQ(published_stats().stale, 1u);
}

TEST_F(RetainedMeshlets, FreedBlocksWaitOneLatch) {
  testing::set_pool_bytes(256);
  Piece first = strip(20);
  Piece second = strip(20, GX_TRIANGLESTRIP, 3.f);
  const Built a = build(first);
  ASSERT_EQ(a.range.state, State::Ready);
  release(a.key);
  Key key;
  uint8_t churn = 0;
  EXPECT_EQ(request(second.source(), &key, &churn).state, State::Building);
  (void)latch();
  EXPECT_EQ(lookup(key).reason, Reason::PoolFull);
  (void)latch();
  EXPECT_EQ(lookup(key).state, State::None);
  EXPECT_EQ(request(second.source(), &key, &churn).state, State::Building);
  (void)latch();
  EXPECT_EQ(lookup(key).state, State::Ready);
  EXPECT_EQ(lookup(key).first, a.range.first);
}

std::vector<std::pair<uint64_t, std::vector<uint32_t>>> g_writes;

void capture_write(uint64_t offset, std::span<const uint32_t> words) noexcept {
  g_writes.emplace_back(offset, std::vector<uint32_t>(words.begin(), words.end()));
}

TEST_F(RetainedMeshlets, UploadsLandAtTheirFrame) {
  g_writes.clear();
  testing::set_write(&capture_write);
  set_recording_frame(7);
  Piece piece = grid(4);
  piece.flags = piece_flag::Static;
  const Built built = build(piece);
  ASSERT_EQ(built.range.state, State::Ready);
  const auto staged = testing::staged();
  ASSERT_EQ(staged.size(), 1u);
  EXPECT_EQ(staged[0].stamp, 7u);
  EXPECT_EQ(staged[0].offset, uint64_t{built.range.first} * ClusterBytes);
  worker_frame_begin(7);
  EXPECT_TRUE(g_writes.empty());
  worker_frame_begin(8);
  ASSERT_EQ(g_writes.size(), 1u);
  EXPECT_EQ(g_writes[0].first, uint64_t{built.range.first} * ClusterBytes);
  EXPECT_EQ(g_writes[0].second, built.block);
  EXPECT_TRUE(testing::staged().empty());
  const auto geometry = cpu_geometry(built.key);
  ASSERT_NE(geometry, nullptr);
  EXPECT_EQ(geometry->positions.size(), size_t{piece.vtxCount} * 3);
  EXPECT_EQ(geometry->triangles.size(), size_t{16} * 2 * 3);
  EXPECT_EQ(geometry->aabb[3], 4.f);
}

TEST(RetainedMeshletsNoWorkers, NoWorkersMeansUnavailable) {
  jobs::initialize(0);
  testing::reset();
  set_active(true);
  Piece piece = grid(3);
  piece.flags = piece_flag::Static;
  Key key;
  uint8_t churn = 0;
  const ClusterRange range = request(piece.source(), &key, &churn);
  EXPECT_EQ(range.state, State::Unavailable);
  EXPECT_FALSE(key.valid());
  EXPECT_EQ(pending_static(), 0u);
  (void)latch();
  EXPECT_EQ(published_stats().unavailable, 1u);
  EXPECT_EQ(published_stats().inlineBuilds, 0u);
  testing::reset();
  jobs::shutdown();
}
} // namespace
} // namespace aurora::render::scene::meshlets

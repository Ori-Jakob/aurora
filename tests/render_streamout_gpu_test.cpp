#include <gtest/gtest.h>

#include "render_graph_planned.hpp"
#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gx/gx.hpp"
#include "render/graph.hpp"
#include "render/jobs.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/streamout.hpp"
#include "render/shaders.hpp"
#include "render/stage_data.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::scene::streamout {
namespace {
constexpr uint32_t BlockOffset = 256;
constexpr uint32_t PosArray = 8192;
constexpr uint32_t NrmArray = 12288;
constexpr uint32_t TexArray = 16384;
constexpr uint32_t ArrayCount = 32;
constexpr uint32_t Stride = 9;
constexpr uint32_t FrameIndex = 7;

std::vector<uint8_t> g_block;

gfx::Range capture_block(std::span<const uint8_t> bytes) noexcept {
  g_block.assign(bytes.begin(), bytes.end());
  return {BlockOffset, static_cast<uint32_t>(bytes.size())};
}

void put_be16(std::vector<uint8_t>& out, size_t at, uint16_t value) {
  out[at] = static_cast<uint8_t>(value >> 8);
  out[at + 1] = static_cast<uint8_t>(value);
}

void put_be32(std::vector<uint8_t>& out, size_t at, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out[at + i] = static_cast<uint8_t>(value >> (24 - 8 * i));
  }
}

float ordered_to_f32(uint32_t u) { return std::bit_cast<float>((u & 0x80000000u) != 0 ? (u & 0x7FFFFFFFu) : ~u); }

std::array<float, 3> oct_decode(uint32_t w) {
  const auto unpack = [](uint32_t v) {
    return std::max(static_cast<float>(static_cast<int16_t>(v & 0xFFFF)) / 32767.f, -1.f);
  };
  std::array<float, 3> n{unpack(w), unpack(w >> 16), 0.f};
  n[2] = 1.f - std::abs(n[0]) - std::abs(n[1]);
  const float t = std::max(-n[2], 0.f);
  n[0] += n[0] >= 0.f ? -t : t;
  n[1] += n[1] >= 0.f ? -t : t;
  const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  return {n[0] / len, n[1] / len, n[2] / len};
}

std::vector<uint32_t> reference_indices(GXPrimitive prim, uint32_t vtxCount) {
  std::vector<uint32_t> out;
  if (prim == GX_QUADS) {
    for (uint32_t v = 0; v + 3 < vtxCount; v += 4) {
      out.insert(out.end(), {v, v + 1, v + 2, v + 2, v + 3, v});
    }
  } else if (prim == GX_TRIANGLES) {
    for (uint32_t v = 0; v + 2 < vtxCount; v += 3) {
      out.insert(out.end(), {v, v + 1, v + 2});
    }
  } else {
    for (uint32_t v = 0; v < vtxCount; ++v) {
      if (v < 3) {
        out.push_back(v);
      } else if (prim == GX_TRIANGLEFAN) {
        out.insert(out.end(), {0u, v - 1, v});
      } else if ((v & 1) == 0) {
        out.insert(out.end(), {v - 2, v - 1, v});
      } else {
        out.insert(out.end(), {v - 1, v - 2, v});
      }
    }
  }
  return out;
}

struct Draw {
  GXPrimitive prim = GX_TRIANGLES;
  uint32_t vtxCount = 0;
  uint32_t slot = 0;
  std::vector<uint32_t> indices;
  uint32_t vtxStart = 0;
  uint32_t idxStart = 0;
};

class RetainedStreamoutGpu : public ::testing::Test {
protected:
  void SetUp() override {
    m_options = {
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined)};
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    jobs::initialize(2);
    shaders::initialize();
    gfx::set_blocking_pipelines(true);
    retained::testing::reset();
    initialize();
    testing::set_push(&capture_block);
    testing::set_room(&m_room);
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      testing::reset();
      shutdown();
      gfx::set_blocking_pipelines(false);
      shaders::shutdown();
      jobs::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  [[nodiscard]] wgpu::Buffer buffer(uint64_t size, wgpu::BufferUsage extra = wgpu::BufferUsage::None) const {
    const wgpu::BufferDescriptor descriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | extra,
        .size = std::max<uint64_t>((size + 3) / 4 * 4, 4),
    };
    return m_device->device.CreateBuffer(&descriptor);
  }

  [[nodiscard]] std::vector<uint32_t> read(const wgpu::Buffer& source, uint64_t offset, uint64_t size) const {
    return test::read_buffer(*m_device, source, offset, size).value_or(std::vector<uint32_t>{});
  }

  void write(const wgpu::Buffer& target, uint64_t offset, std::span<const uint8_t> bytes) const {
    std::vector<uint8_t> padded(bytes.begin(), bytes.end());
    padded.resize((padded.size() + 3) / 4 * 4, 0);
    m_device->queue.WriteBuffer(target, offset, padded.data(), padded.size());
  }

  std::shared_ptr<const Plan> seal_and_commit(std::shared_ptr<retained::Frame>& frame) {
    detail::StageData data{.stage = Stage::OpaqueDone, .frameIndex = FrameIndex};
    frame = std::make_shared<retained::Frame>();
    seal(*frame, data);
    data.retained = frame;
    std::array<std::byte, ParamsSize> params{};
    commit_stage_params(data, nullptr, params);
    sealed(data);
    return frame->plan;
  }

  struct Pools {
    wgpu::Buffer vbuf, abuf, ibuf, vertices, indices, instances, palettes, frame;
  };

  Pools make_pools(std::span<const uint8_t> vertexBytes, std::span<const uint8_t> storageBytes,
                   std::span<const uint8_t> indexBytes) {
    Pools pools{
        .vbuf = buffer(std::max<size_t>(vertexBytes.size(), 256)),
        .abuf = buffer(std::max<size_t>(storageBytes.size(), BlockOffset + g_block.size())),
        .ibuf = buffer(std::max<size_t>(indexBytes.size(), 256), wgpu::BufferUsage::Index),
        .vertices = buffer(1 << 20),
        .indices = buffer(1 << 18),
        .instances = buffer(InstancePoolBytes),
        .palettes = buffer(PalettePoolBytes),
        .frame = buffer(FramePoolBytes),
    };
    write(pools.vbuf, 0, vertexBytes);
    std::vector<uint8_t> storage(storageBytes.begin(), storageBytes.end());
    storage.resize(std::max(storage.size(), BlockOffset + g_block.size()), 0);
    std::copy(g_block.begin(), g_block.end(), storage.begin() + BlockOffset);
    write(pools.abuf, 0, storage);
    write(pools.ibuf, 0, indexBytes);
    return pools;
  }

  bool run(const Pools& pools, const Plan& plan) {
    if (!testing::wait_variants(plan)) {
      return false;
    }
    const auto view = [](const wgpu::Buffer& b) { return BufferView{.buffer = b, .offset = 0, .size = b.GetSize()}; };
    const std::array<BufferView, 8> views{view(pools.vbuf),     view(pools.abuf),    view(pools.ibuf),
                                          view(pools.vertices), view(pools.indices), view(pools.instances),
                                          view(pools.palettes), view(pools.frame)};
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    if (!testing::encode_plan(encoder, views, plan)) {
      return false;
    }
    return test::submit_and_wait(*m_device, encoder.Finish());
  }

  [[nodiscard]] std::vector<uint32_t> block_words() const {
    std::vector<uint32_t> words(g_block.size() / 4);
    std::memcpy(words.data(), g_block.data(), words.size() * 4);
    return words;
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  gfx::detail::StreamRoom m_room{.storageCapacity = 64ull << 20};
};

TEST_F(RetainedStreamoutGpu, RigidPiecesMatchReference) {
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  attrs[GX_VA_POS] = {.attrType = GX_INDEX16, .cnt = 3, .compType = GX_F32, .offset = 0, .stride = 12, .le = false};
  attrs[GX_VA_NRM] = {
      .attrType = GX_INDEX8, .cnt = 3, .compType = GX_S16, .offset = 2, .stride = 6, .frac = 14, .le = false};
  attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8, .offset = 3};
  attrs[GX_VA_TEX0] = {
      .attrType = GX_INDEX16, .cnt = 2, .compType = GX_S16, .offset = 7, .stride = 4, .frac = 8, .le = false};

  std::vector<uint8_t> storage(TexArray + ArrayCount * 4, 0);
  std::vector<std::array<float, 3>> positions(ArrayCount);
  std::vector<std::array<int16_t, 3>> normals(ArrayCount);
  std::vector<std::array<int16_t, 2>> uvs(ArrayCount);
  for (uint32_t i = 0; i < ArrayCount; ++i) {
    positions[i] = {static_cast<float>(i) * 0.37f - 3.f, std::sin(static_cast<float>(i)) * 5.f, 1e3f + i * 0.125f};
    normals[i] = {static_cast<int16_t>(i * 977 % 16384 - 8192), static_cast<int16_t>(4096 - i * 311 % 8192),
                  static_cast<int16_t>(i == 5 ? 0 : 16384 - i * 200)};
    if (i == 6) {
      normals[i] = {0, 0, 16384};
    }
    uvs[i] = {static_cast<int16_t>(i * 64 - 900), static_cast<int16_t>(static_cast<int32_t>(i) * -37 + 300)};
    for (int c = 0; c < 3; ++c) {
      put_be32(storage, PosArray + i * 12 + c * 4, std::bit_cast<uint32_t>(positions[i][c]));
      put_be16(storage, NrmArray + i * 6 + c * 2, static_cast<uint16_t>(normals[i][c]));
    }
    put_be16(storage, TexArray + i * 4, static_cast<uint16_t>(uvs[i][0]));
    put_be16(storage, TexArray + i * 4 + 2, static_cast<uint16_t>(uvs[i][1]));
  }

  std::vector<Draw> draws{
      {.prim = GX_QUADS, .vtxCount = 8, .slot = 1},
      {.prim = GX_TRIANGLESTRIP, .vtxCount = 7, .slot = 2},
      {.prim = GX_TRIANGLES, .vtxCount = 9, .slot = 3},
      {.prim = GX_TRIANGLEFAN, .vtxCount = 6, .slot = 5},
      {.prim = GX_TRIANGLES, .vtxCount = 6, .slot = 4, .indices = {0, 1, 2, 2, 3, 4, 5, 4, 3, 0, 5, 1}},
  };
  std::vector<uint8_t> vertexBytes;
  std::vector<uint8_t> indexBytes;
  std::vector<std::vector<uint32_t>> refs(draws.size());
  for (size_t d = 0; d < draws.size(); ++d) {
    Draw& draw = draws[d];
    draw.vtxStart = static_cast<uint32_t>(vertexBytes.size());
    vertexBytes.resize(vertexBytes.size() + draw.vtxCount * Stride, 0);
    for (uint32_t v = 0; v < draw.vtxCount; ++v) {
      const uint32_t ref = (v * 7 + static_cast<uint32_t>(d) * 5) % ArrayCount;
      refs[d].push_back(ref);
      const size_t at = draw.vtxStart + v * Stride;
      put_be16(vertexBytes, at, static_cast<uint16_t>(ref));
      vertexBytes[at + 2] = static_cast<uint8_t>(ref);
      vertexBytes[at + 3] = 0xFF;
      put_be16(vertexBytes, at + 7, static_cast<uint16_t>(ref));
    }
    vertexBytes.resize((vertexBytes.size() + 3) / 4 * 4, 0);
    if (!draw.indices.empty()) {
      draw.idxStart = static_cast<uint32_t>(indexBytes.size());
      for (const uint32_t index : draw.indices) {
        const auto value = static_cast<uint16_t>(index);
        indexBytes.push_back(static_cast<uint8_t>(value));
        indexBytes.push_back(static_cast<uint8_t>(value >> 8));
      }
    }
  }

  gx::g_gxState = {};
  gx::g_gxState.numTevStages = 1;
  gx::g_gxState.tevStages[0].colorPass.a = GX_CC_TEXC;
  gx::g_gxState.tevStages[0].texMapId = GX_TEXMAP0;
  gx::g_gxState.tevStages[0].texCoordId = GX_TEXCOORD0;
  gx::g_gxState.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY, .postMtx = GX_PTIDENTITY};
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, -10.f}};
  gx::g_gxState.pnMtx[0].nrm = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  const Affine invView = to_affine(Mat3x4<float>{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}});
  const Affine world = to_affine(gx::g_gxState.pnMtx[0].pos);

  ASSERT_FALSE(begin_frame(true, 1));
  for (const Draw& draw : draws) {
    gx::DrawImmediateData immediates{.vtxStart = draw.vtxStart};
    immediates.arrayStart[GX_VA_POS - GX_VA_POS] = PosArray;
    immediates.arrayStart[GX_VA_NRM - GX_VA_POS] = NrmArray;
    immediates.arrayStart[GX_VA_TEX0 - GX_VA_POS] = TexArray;
    const bool indexed = !draw.indices.empty();
    on_capture({
        .slot = draw.slot,
        .slotGeneration = 1,
        .identityHash = 0x1000 + draw.slot,
        .newGeometry = true,
        .entryFlags = indexed ? retained::entry_flag::Indexed : uint16_t{0},
        .classBits = retained::class_bit::Dynamic,
        .attrs = attrs.data(),
        .stride = Stride,
        .immediates = &immediates,
        .prim = static_cast<uint8_t>(draw.prim),
        .vtxCount = draw.vtxCount,
        .idxRange = {draw.idxStart, static_cast<uint32_t>(draw.indices.size() * 2)},
        .indexCount = static_cast<uint32_t>(draw.indices.size()),
        .world = &world,
        .invView = &invView,
    });
  }
  std::shared_ptr<retained::Frame> frame;
  const auto plan = seal_and_commit(frame);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->variants.size(), 1u);
  const std::vector<uint32_t> blockWords = block_words();
  ASSERT_EQ(blockWords[block::PieceCount], draws.size());
  const Pools pools = make_pools(vertexBytes, storage, indexBytes);
  ASSERT_TRUE(run(pools, *plan));

  const std::vector<uint32_t> instances = read(pools.instances, 0, 8 * record::Bytes);
  const std::vector<uint32_t> header = read(pools.frame, 0, FramePoolBytes);
  ASSERT_EQ(instances.size(), 8 * record::Words);
  ASSERT_EQ(header.size(), frame::Words);
  const uint32_t headerSrc = blockWords[blockWords[block::WriteJobsWord] + 2];
  for (uint32_t w = 0; w < frame::LateFrame; ++w) {
    if (w == frame::BreakCount) {
      EXPECT_EQ(header[w], 0u);
      continue;
    }
    EXPECT_EQ(header[w], blockWords[headerSrc + w]) << "header word " << w;
  }
  EXPECT_EQ(header[frame::FrameIndex], FrameIndex);
  EXPECT_NE(header[frame::Flags] & frame::Reset, 0u);

  for (size_t d = 0; d < draws.size(); ++d) {
    const Draw& draw = draws[d];
    const uint32_t* rec = &instances[draw.slot * record::Words];
    ASSERT_NE(rec[record::Flags] & instance_flag::Valid, 0u) << d;
    for (uint32_t i = 0; i < blockWords[block::WriteJobCount]; ++i) {
      const uint32_t* job = &blockWords[blockWords[block::WriteJobsWord] + i * block::WriteJobWords];
      if (job[0] == block::TargetInstances && job[1] == draw.slot * record::Words && job[3] == record::CpuWords) {
        for (uint32_t w = 0; w < record::CpuWords; ++w) {
          EXPECT_EQ(rec[w], blockWords[job[2] + w]) << "draw " << d << " record word " << w;
        }
      }
    }
    EXPECT_EQ(rec[record::StreamFrame], FrameIndex);
    EXPECT_EQ(rec[record::PrevStreamFrame], 0u);
    const uint32_t vertexFirst = rec[record::VertexFirst];
    const uint32_t indexFirst = rec[record::IndexFirst];
    const uint32_t indexCount = rec[record::IndexCount];
    const std::vector<uint32_t> want =
        draw.indices.empty() ? reference_indices(draw.prim, draw.vtxCount) : draw.indices;
    ASSERT_EQ(indexCount, want.size()) << d;
    const std::vector<uint32_t> gotIndices = read(pools.indices, uint64_t{indexFirst} * 4, uint64_t{indexCount} * 4);
    EXPECT_EQ(gotIndices, want) << d;
    const std::vector<uint32_t> vertices =
        read(pools.vertices, uint64_t{vertexFirst} * vertex::Bytes, uint64_t{draw.vtxCount} * vertex::Bytes);
    ASSERT_EQ(vertices.size(), draw.vtxCount * vertex::Words);
    std::array<float, 3> lo{INFINITY, INFINITY, INFINITY};
    std::array<float, 3> hi{-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t v = 0; v < draw.vtxCount; ++v) {
      const uint32_t* out = &vertices[v * vertex::Words];
      const uint32_t ref = refs[d][v];
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(out[c], std::bit_cast<uint32_t>(positions[ref][c])) << d << " vertex " << v;
        lo[c] = std::min(lo[c], positions[ref][c]);
        hi[c] = std::max(hi[c], positions[ref][c]);
      }
      ASSERT_NE(out[vertex::Aux] & vertex::HasNormal, 0u);
      std::array<double, 3> n{normals[ref][0] / 16384.0, normals[ref][1] / 16384.0, normals[ref][2] / 16384.0};
      const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      const std::array<float, 3> decoded = oct_decode(out[vertex::Normal]);
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(decoded[c], n[c] / len, 1e-4) << d << " vertex " << v << " normal " << c;
      }
      EXPECT_EQ(std::bit_cast<float>(out[vertex::Uv0]), uvs[ref][0] / 256.f);
      EXPECT_EQ(std::bit_cast<float>(out[vertex::Uv0 + 1]), uvs[ref][1] / 256.f);
    }
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(ordered_to_f32(rec[record::CurAabb + c]), lo[c]) << d;
      EXPECT_EQ(ordered_to_f32(rec[record::CurAabb + 3 + c]), hi[c]) << d;
    }
  }
  end_frame();
}

TEST_F(RetainedStreamoutGpu, AlphaUvFixedPointAndClampFlag) {
  constexpr uint32_t TexStride = 20;
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  attrs[GX_VA_TEX1] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  const std::array<std::array<float, 2>, 3> alpha{{{0.5f, -1.25f}, {9.f, 3.f}, {-7.99f, 0.000244140625f}}};
  std::vector<uint8_t> vertexBytes(3 * TexStride, 0);
  for (uint32_t v = 0; v < 3; ++v) {
    const std::array<float, 5> words{static_cast<float>(v), 1.f, 2.f, alpha[v][0], alpha[v][1]};
    for (size_t i = 0; i < words.size(); ++i) {
      put_be32(vertexBytes, v * TexStride + i * 4, std::bit_cast<uint32_t>(words[i]));
    }
  }
  gx::g_gxState = {};
  gx::g_gxState.numTevStages = 2;
  gx::g_gxState.tevStages[0].colorPass.a = GX_CC_TEXC;
  gx::g_gxState.tevStages[0].texMapId = GX_TEXMAP0;
  gx::g_gxState.tevStages[0].texCoordId = GX_TEXCOORD0;
  gx::g_gxState.tevStages[1].alphaPass.a = GX_CA_TEXA;
  gx::g_gxState.tevStages[1].texMapId = GX_TEXMAP1;
  gx::g_gxState.tevStages[1].texCoordId = GX_TEXCOORD1;
  gx::g_gxState.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_POS, .mtx = GX_IDENTITY, .postMtx = GX_PTIDENTITY};
  gx::g_gxState.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_IDENTITY, .postMtx = GX_PTIDENTITY};
  gx::g_gxState.alphaCompare.comp0 = GX_GREATER;
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, -10.f}};
  const Affine invView = to_affine(Mat3x4<float>{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}});
  const Affine world = to_affine(gx::g_gxState.pnMtx[0].pos);

  ASSERT_FALSE(begin_frame(true, 1));
  const gx::DrawImmediateData immediates{.vtxStart = 0};
  on_capture({
      .slot = 9,
      .slotGeneration = 1,
      .identityHash = 0x99,
      .newGeometry = true,
      .classBits = retained::class_bit::Dynamic,
      .attrs = attrs.data(),
      .stride = TexStride,
      .immediates = &immediates,
      .prim = GX_TRIANGLES,
      .vtxCount = 3,
      .world = &world,
      .invView = &invView,
  });
  std::shared_ptr<retained::Frame> frame;
  const auto plan = seal_and_commit(frame);
  ASSERT_NE(plan, nullptr);
  const Pools pools = make_pools(vertexBytes, {}, {});
  ASSERT_TRUE(run(pools, *plan));
  const std::vector<uint32_t> rec = read(pools.instances, 9 * record::Bytes, record::Bytes);
  ASSERT_EQ(rec.size(), record::Words);
  EXPECT_NE(rec[record::Flags] & instance_flag::UvNonTex, 0u);
  EXPECT_EQ(rec[record::Flags] & instance_flag::AlphaUvIsUv0, 0u);
  EXPECT_EQ(rec[record::Material] >> 16 & 0xFF, uint32_t{GX_TEXCOORD1});
  EXPECT_EQ(rec[record::GpuFlags] & gpu_flag::AlphaUvClamped, gpu_flag::AlphaUvClamped);
  const std::vector<uint32_t> vertices =
      read(pools.vertices, uint64_t{rec[record::VertexFirst]} * vertex::Bytes, 3 * vertex::Bytes);
  ASSERT_EQ(vertices.size(), 3 * vertex::Words);
  for (uint32_t v = 0; v < 3; ++v) {
    const uint32_t word = vertices[v * vertex::Words + vertex::AlphaUv];
    for (int c = 0; c < 2; ++c) {
      const auto fixed = static_cast<int16_t>(c == 0 ? word & 0xFFFF : word >> 16);
      const float want = std::clamp(std::round(alpha[v][c] * 4096.f), -32767.f, 32767.f);
      EXPECT_EQ(fixed, static_cast<int16_t>(want)) << v << " " << c;
    }
    EXPECT_EQ(vertices[v * vertex::Words + vertex::Uv0], 0u);
    EXPECT_EQ(vertices[v * vertex::Words + vertex::Aux] & vertex::HasNormal, 0u);
  }
  end_frame();
}

TEST_F(RetainedStreamoutGpu, DeclarationCompilesInGraph) {
  const std::array passes{pass_desc()};
  const auto compile = [&](bool retained) {
    detail::GraphInputs inputs = detail::planned::planned_inputs(detail::planned::full_limits(), passes, {});
    inputs.caps = &caps();
    inputs.env.retained = retained;
    return detail::compile_graph(inputs);
  };
  const auto graph = compile(true);
  ASSERT_NE(graph, nullptr);
  const detail::NodeId node = graph->find_node("scene.streamout");
  ASSERT_NE(node, detail::InvalidNode);
  const detail::CompiledNode& compiled = graph->nodes[node];
  EXPECT_EQ(compiled.status, detail::CullReason::None);
  EXPECT_EQ(compiled.stage, Stage::OpaqueDone);
  EXPECT_TRUE(compiled.runsWithoutMarker);
  EXPECT_TRUE(compiled.manualBindings);
  EXPECT_EQ(compiled.paramsSize, ParamsSize);
  const auto& program = graph->stages[static_cast<size_t>(Stage::OpaqueDone)].passes;
  EXPECT_NE(std::ranges::find(program, node), program.end());
  for (const std::string_view name : {"scene.gx_vertex", "scene.gx_storage", "scene.gx_index", "scene.vertices",
                                      "scene.indices", "scene.instances", "scene.palettes", "scene.frame"}) {
    const detail::ResourceId resource = graph->find_resource(name);
    ASSERT_NE(resource, detail::InvalidResource) << name;
    EXPECT_EQ(graph->resources[resource].kind, detail::ResourceKind::Persistent) << name;
  }
  const auto off = compile(false);
  ASSERT_NE(off, nullptr);
  const detail::NodeId skipped = off->find_node("scene.streamout");
  ASSERT_NE(skipped, detail::InvalidNode);
  EXPECT_NE(off->nodes[skipped].status, detail::CullReason::None);
}

TEST_F(RetainedStreamoutGpu, VariantRerequestedAfterShaderSession) {
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  gx::g_gxState = {};
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, -10.f}};
  const Affine invView = to_affine(Mat3x4<float>{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}});
  const Affine world = to_affine(gx::g_gxState.pnMtx[0].pos);
  const gx::DrawImmediateData immediates{};
  const auto frame_plan = [&](uint64_t frame) {
    EXPECT_FALSE(begin_frame(true, frame));
    on_capture({
        .slot = 3,
        .slotGeneration = 1,
        .identityHash = 0x33,
        .newGeometry = true,
        .classBits = retained::class_bit::Dynamic,
        .attrs = attrs.data(),
        .stride = 12,
        .immediates = &immediates,
        .prim = GX_TRIANGLES,
        .vtxCount = 3,
        .world = &world,
        .invView = &invView,
    });
    std::shared_ptr<retained::Frame> sealedFrame;
    auto plan = seal_and_commit(sealedFrame);
    end_frame();
    return plan;
  };
  const auto first = frame_plan(1);
  ASSERT_NE(first, nullptr);
  ASSERT_EQ(first->variants.size(), 1u);
  EXPECT_EQ(first->variants[0].session, shaders::session());
  EXPECT_TRUE(shaders::ready(first->variants[0].pipeline));

  shaders::shutdown();
  shaders::initialize();
  on_freed(3);
  const auto second = frame_plan(2);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(second->variants.size(), 1u);
  EXPECT_EQ(second->variants[0].session, shaders::session());
  EXPECT_NE(second->variants[0].session, first->variants[0].session);
  EXPECT_TRUE(shaders::ready(second->variants[0].pipeline));
  EXPECT_EQ(testing::variant_count(), 1u);
}
} // namespace
} // namespace aurora::render::scene::streamout

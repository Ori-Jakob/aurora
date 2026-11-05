#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"
#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gx/gx.hpp"
#include "gx/vertex_fetch.hpp"
#include "render/jobs.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/streamout.hpp"
#include "render/shaders.hpp"
#include "render/stage_data.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::scene::streamout {
namespace {
constexpr uint32_t BlockOffset = 256;
constexpr uint32_t ArrayBase = 1u << 20;
constexpr uint32_t Elements = 8;
constexpr uint32_t Vertices = 12;
constexpr uint32_t FrameIndex = 3;
constexpr size_t MaxKeys = 200;

std::vector<uint8_t> g_block;

gfx::Range capture_block(std::span<const uint8_t> bytes) noexcept {
  g_block.assign(bytes.begin(), bytes.end());
  return {BlockOffset, static_cast<uint32_t>(bytes.size())};
}

uint32_t comp_size(uint8_t compType) {
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

uint32_t color_size(uint8_t compType) {
  switch (compType) {
  case GX_RGB565:
  case GX_RGBA4:
    return 2;
  case GX_RGB8:
  case GX_RGBA6:
    return 3;
  default:
    return 4;
  }
}

uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}

void put(std::vector<uint8_t>& out, size_t at, uint32_t value, uint32_t size, bool le) {
  if (out.size() < at + size) {
    out.resize(at + size, 0);
  }
  for (uint32_t i = 0; i < size; ++i) {
    out[at + i] = static_cast<uint8_t>(value >> (8 * (le ? i : size - 1 - i)));
  }
}

void fill_value(std::vector<uint8_t>& out, size_t at, const gx::AttrConfig& attr, bool color, bool le, uint32_t seed) {
  if (color) {
    const uint32_t size = color_size(attr.compType);
    put(out, at, mix(seed), size, le);
    return;
  }
  const uint32_t size = comp_size(attr.compType);
  for (uint32_t c = 0; c < attr.cnt; ++c) {
    const uint32_t r = mix(seed * 31 + c);
    if (attr.compType == GX_F32) {
      const float value = static_cast<float>(static_cast<int32_t>(r % 200001) - 100000) / 1024.f;
      put(out, at + c * 4, std::bit_cast<uint32_t>(value), 4, le);
    } else {
      put(out, at + c * size, r, size, le);
    }
  }
}

float ulp(float a) { return std::nextafter(std::abs(a), INFINITY) - std::abs(a); }

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

struct Format {
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  uint8_t stride = 0;
  uint8_t uvTex = 0xFF;
  std::string label;
};

struct Draw {
  uint32_t slot = 0;
  uint32_t vtxStart = 0;
  gx::DrawImmediateData immediates{};
  std::vector<uint8_t> records;
};

void select_uv(uint8_t tex) {
  gx::g_gxState = {};
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  gx::g_gxState.pnMtx[0].nrm = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  if (tex == 0xFF) {
    return;
  }
  gx::g_gxState.numTevStages = 1;
  gx::g_gxState.tevStages[0].colorPass.a = GX_CC_TEXC;
  gx::g_gxState.tevStages[0].texMapId = GX_TEXMAP0;
  gx::g_gxState.tevStages[0].texCoordId = GX_TEXCOORD0;
  gx::g_gxState.tcgs[0] = {.type = GX_TG_MTX2x4,
                           .src = static_cast<GXTexGenSrc>(GX_TG_TEX0 + tex),
                           .mtx = GX_IDENTITY,
                           .postMtx = GX_PTIDENTITY};
}

class VertexFetchCpu : public ::testing::Test {
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

  void write(const wgpu::Buffer& target, std::span<const uint8_t> bytes) const {
    std::vector<uint8_t> padded(bytes.begin(), bytes.end());
    padded.resize((padded.size() + 3) / 4 * 4, 0);
    m_device->queue.WriteBuffer(target, 0, padded.data(), padded.size());
  }

  [[nodiscard]] std::vector<uint32_t> read(const wgpu::Buffer& source, uint64_t offset, uint64_t size) const {
    return test::read_buffer(*m_device, source, offset, size).value_or(std::vector<uint32_t>{});
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  gfx::detail::StreamRoom m_room{.storageCapacity = 64ull << 20};
};

TEST_F(VertexFetchCpu, MatchesKernelForCorpusFormats) {
  std::vector<Format> formats;
  std::set<std::string> seen;
  for (const auto& entry : gx::corpus::generate_synthetic()) {
    const gx::ShaderConfig config = gx::corpus::decode(entry.record);
    const auto& pos = config.attrs[GX_VA_POS];
    if (pos.attrType == GX_NONE || config.lineMode != 0 || (pos.cnt != 2 && pos.cnt != 3) || config.vtxStride == 0) {
      continue;
    }
    Format format{.stride = config.vtxStride, .label = entry.label};
    for (const GXAttr attr : {GX_VA_POS, GX_VA_NRM, GX_VA_CLR0}) {
      format.attrs[attr] = config.attrs[attr];
    }
    for (uint8_t t = 0; t < 8; ++t) {
      if (config.attrs[GX_VA_TEX0 + t].attrType != GX_NONE && config.attrs[GX_VA_TEX0 + t].cnt <= 2) {
        format.uvTex = t;
        format.attrs[GX_VA_TEX0 + t] = config.attrs[GX_VA_TEX0 + t];
        break;
      }
    }
    std::string key(reinterpret_cast<const char*>(format.attrs.data()), sizeof(format.attrs));
    key += static_cast<char>(format.stride);
    if (seen.insert(key).second && formats.size() < MaxKeys) {
      formats.push_back(std::move(format));
    }
  }
  ASSERT_GE(formats.size(), 4u);
  RecordProperty("formats", static_cast<int>(formats.size()));

  std::vector<uint8_t> vertexBytes;
  std::vector<uint8_t> storage(ArrayBase, 0);
  std::vector<Draw> draws(formats.size());
  const Affine identity = to_affine(Mat3x4<float>{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}});
  ASSERT_FALSE(begin_frame(true, 1));
  for (size_t f = 0; f < formats.size(); ++f) {
    const Format& format = formats[f];
    Draw& draw = draws[f];
    draw.slot = static_cast<uint32_t>(f + 1);
    draw.vtxStart = static_cast<uint32_t>(vertexBytes.size());
    draw.immediates.vtxStart = draw.vtxStart;
    draw.records.assign(size_t{Vertices} * format.stride, 0);
    for (uint32_t a = GX_VA_POS; a <= GX_VA_TEX7; ++a) {
      const gx::AttrConfig& attr = format.attrs[a];
      if (attr.attrType == GX_NONE) {
        continue;
      }
      const bool color = a == GX_VA_CLR0;
      if (attr.attrType == GX_DIRECT) {
        for (uint32_t v = 0; v < Vertices; ++v) {
          fill_value(draw.records, size_t{v} * format.stride + attr.offset, attr, color, false,
                     static_cast<uint32_t>(f * 1000 + v * 16 + a));
        }
        continue;
      }
      const auto arrayStart = static_cast<uint32_t>(storage.size());
      draw.immediates.arrayStart[a - GX_VA_POS] = arrayStart;
      for (uint32_t e = 0; e < Elements; ++e) {
        fill_value(storage, arrayStart + size_t{e} * attr.stride, attr, color, attr.le,
                   static_cast<uint32_t>(f * 7919 + e * 16 + a + 0x10000));
      }
      storage.resize((storage.size() + size_t{Elements} * std::max<uint32_t>(attr.stride, 1) + 15) / 16 * 16, 0);
      const uint32_t idxSize = attr.attrType == GX_INDEX8 ? 1 : 2;
      for (uint32_t v = 0; v < Vertices; ++v) {
        put(draw.records, size_t{v} * format.stride + attr.offset, (v * 5 + a) % Elements, idxSize, false);
      }
    }
    draw.records.resize(size_t{Vertices} * format.stride);
    vertexBytes.insert(vertexBytes.end(), draw.records.begin(), draw.records.end());
    vertexBytes.resize((vertexBytes.size() + 3) / 4 * 4, 0);
    select_uv(format.uvTex);
    on_capture({
        .slot = draw.slot,
        .slotGeneration = 1,
        .identityHash = draw.slot,
        .newGeometry = true,
        .classBits = retained::class_bit::Dynamic,
        .attrs = format.attrs.data(),
        .stride = format.stride,
        .immediates = &draw.immediates,
        .prim = GX_TRIANGLES,
        .vtxCount = Vertices,
        .world = &identity,
        .invView = &identity,
    });
  }
  detail::StageData data{.stage = Stage::OpaqueDone, .frameIndex = FrameIndex};
  auto frame = std::make_shared<retained::Frame>();
  seal(*frame, data);
  data.retained = frame;
  std::array<std::byte, ParamsSize> params{};
  commit_stage_params(data, nullptr, params);
  sealed(data);
  ASSERT_NE(frame->plan, nullptr);
  ASSERT_TRUE(testing::wait_variants(*frame->plan));

  storage.resize(std::max(storage.size(), BlockOffset + g_block.size()), 0);
  ASSERT_LE(BlockOffset + g_block.size(), ArrayBase);
  std::copy(g_block.begin(), g_block.end(), storage.begin() + BlockOffset);
  const wgpu::Buffer vbuf = buffer(std::max<size_t>(vertexBytes.size(), 256));
  const wgpu::Buffer abuf = buffer(storage.size());
  const wgpu::Buffer ibuf = buffer(256, wgpu::BufferUsage::Index);
  const wgpu::Buffer vertices = buffer(4 << 20);
  const wgpu::Buffer indices = buffer(1 << 20);
  const wgpu::Buffer instances = buffer(InstancePoolBytes);
  const wgpu::Buffer palettes = buffer(PalettePoolBytes);
  const wgpu::Buffer header = buffer(FramePoolBytes);
  write(vbuf, vertexBytes);
  write(abuf, storage);
  const auto view = [](const wgpu::Buffer& b) { return BufferView{.buffer = b, .offset = 0, .size = b.GetSize()}; };
  const std::array<BufferView, 8> views{view(vbuf),    view(abuf),      view(ibuf),     view(vertices),
                                        view(indices), view(instances), view(palettes), view(header)};
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  ASSERT_TRUE(testing::encode_plan(encoder, views, *frame->plan));
  ASSERT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  const std::vector<uint32_t> records = read(instances, 0, (formats.size() + 1) * record::Bytes);
  ASSERT_EQ(records.size(), (formats.size() + 1) * record::Words);

  size_t compared = 0;
  for (size_t f = 0; f < formats.size(); ++f) {
    const Format& format = formats[f];
    const Draw& draw = draws[f];
    const uint32_t* rec = &records[draw.slot * record::Words];
    ASSERT_NE(rec[record::Flags] & instance_flag::Valid, 0u) << format.label;
    const std::vector<uint32_t> out =
        read(vertices, uint64_t{rec[record::VertexFirst]} * vertex::Bytes, uint64_t{Vertices} * vertex::Bytes);
    ASSERT_EQ(out.size(), Vertices * vertex::Words);
    gx::ShaderConfig config{};
    config.vtxStride = format.stride;
    config.attrs = format.attrs;
    gx::vertex_fetch::CpuVertexSource source{.records = draw.records};
    for (uint32_t i = 0; i < gx::MaxIndexAttr; ++i) {
      const uint32_t start = draw.immediates.arrayStart[i];
      source.arrays[i] = {.bytes = std::span<const uint8_t>{storage}.subspan(start)};
    }
    const auto close = [](float got, float want, bool exact) {
      return exact ? std::bit_cast<uint32_t>(got) == std::bit_cast<uint32_t>(want)
                   : std::abs(got - want) <= ulp(std::max(std::abs(got), std::abs(want)));
    };
    const auto& posAttr = format.attrs[GX_VA_POS];
    const auto& nrmAttr = format.attrs[GX_VA_NRM];
    const auto& clrAttr = format.attrs[GX_VA_CLR0];
    for (uint32_t v = 0; v < Vertices; ++v) {
      gx::vertex_fetch::CpuVertex cpu;
      ASSERT_TRUE(gx::vertex_fetch::decode_vertex(config, source, v, cpu)) << format.label << " vertex " << v;
      const uint32_t* gpu = &out[v * vertex::Words];
      for (int c = 0; c < 3; ++c) {
        EXPECT_TRUE(close(std::bit_cast<float>(gpu[vertex::Position + c]), cpu.pos[c], posAttr.compType == GX_F32))
            << format.label << " vertex " << v << " pos " << c << ": " << std::bit_cast<float>(gpu[c]) << " vs "
            << cpu.pos[c];
      }
      if (format.uvTex != 0xFF) {
        const bool exact = format.attrs[GX_VA_TEX0 + format.uvTex].compType == GX_F32;
        for (int c = 0; c < 2; ++c) {
          EXPECT_TRUE(close(std::bit_cast<float>(gpu[vertex::Uv0 + c]), cpu.tex[format.uvTex][c], exact))
              << format.label << " vertex " << v << " uv " << c;
        }
      }
      if (nrmAttr.attrType != GX_NONE) {
        const auto& n = cpu.nrm;
        const double len = std::sqrt(double{n[0]} * n[0] + double{n[1]} * n[1] + double{n[2]} * n[2]);
        ASSERT_NE(gpu[vertex::Aux] & vertex::HasNormal, 0u) << format.label;
        if (len > 1e-3) {
          const std::array<float, 3> decoded = oct_decode(gpu[vertex::Normal]);
          for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(decoded[c], n[c] / len, 1e-4) << format.label << " vertex " << v << " normal " << c;
          }
        }
      }
      if (clrAttr.attrType != GX_NONE) {
        const auto& clr = cpu.clr[0];
        const auto alpha = static_cast<uint32_t>(std::round(std::clamp(clr[3], 0.f, 1.f) * 255.f));
        const auto luma = static_cast<int32_t>(
            std::round(std::clamp(clr[0] * 0.2126f + clr[1] * 0.7152f + clr[2] * 0.0722f, 0.f, 1.f) * 255.f));
        EXPECT_EQ(gpu[vertex::Aux] & 0xFF, alpha) << format.label << " vertex " << v;
        EXPECT_LE(std::abs(static_cast<int32_t>(gpu[vertex::Aux] >> 8 & 0xFF) - luma), 1)
            << format.label << " vertex " << v;
      }
    }
    ++compared;
  }
  EXPECT_EQ(compared, formats.size());
  end_frame();
}
} // namespace
} // namespace aurora::render::scene::streamout

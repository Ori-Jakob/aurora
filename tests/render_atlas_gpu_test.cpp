#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/texture.hpp"
#include "render/jobs.hpp"
#include "render/scene/atlas.hpp"
#include "render/shaders.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace aurora::render::scene::atlas {
namespace {
constexpr uint32_t CheckerSize = 256;
constexpr uint32_t SmallSize = 64;
constexpr uint32_t CheckerTile = 0;
constexpr uint32_t SmallTile = 33;

using Rgba = std::array<uint8_t, 4>;

Rgba checker(uint32_t x, uint32_t y) {
  const bool odd = ((x + y) & 1u) != 0;
  return {static_cast<uint8_t>(odd ? 200 : 40), static_cast<uint8_t>(odd ? 10 : 250),
          static_cast<uint8_t>(x * 255 / (CheckerSize - 1)), static_cast<uint8_t>(odd ? 255 : 128)};
}

Rgba gradient(uint32_t x, uint32_t y) {
  return {static_cast<uint8_t>(x * 4), static_cast<uint8_t>(y * 4), static_cast<uint8_t>((x + y) * 2), 255};
}

class RenderAtlasGpu : public ::testing::Test {
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
    testing::reset();
    initialize();
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      shutdown();
      testing::reset();
      shaders::shutdown();
      jobs::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  gfx::TextureHandle source(uint32_t size, Rgba (*texel)(uint32_t, uint32_t)) const {
    const wgpu::TextureDescriptor descriptor{
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .size = {size, size, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    std::vector<uint8_t> bytes(size_t{size} * size * 4);
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        const Rgba c = texel(x, y);
        std::copy(c.begin(), c.end(), bytes.begin() + (size_t{y} * size + x) * 4);
      }
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = size * 4, .rowsPerImage = size};
    const wgpu::Extent3D extent{size, size, 1};
    m_device->queue.WriteTexture(&destination, bytes.data(), bytes.size(), &layout, &extent);
    return std::make_shared<gfx::TextureRef>(texture, texture.CreateView(), wgpu::TextureView{},
                                             wgpu::Extent3D{size, size, 1}, wgpu::TextureFormat::RGBA8Unorm, 1,
                                             GX_TF_RGBA8);
  }

  std::vector<Rgba> read_tile(const wgpu::Texture& page, uint32_t tile) const {
    constexpr uint32_t RowBytes = TileSize * 4;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{RowBytes} * TileSize};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::TexelCopyTextureInfo from{.texture = page,
                                          .origin = {tile % TilesPerRow * TileSize, tile / TilesPerRow * TileSize, 0}};
    const wgpu::TexelCopyBufferInfo to{.layout = {.bytesPerRow = RowBytes, .rowsPerImage = TileSize}, .buffer = buffer};
    const wgpu::Extent3D extent{TileSize, TileSize, 1};
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&from, &to, &extent);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    const std::vector<uint32_t> words =
        test::read_buffer(*m_device, buffer, 0, descriptor.size).value_or(std::vector<uint32_t>{});
    std::vector<Rgba> out(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
      out[i] = {static_cast<uint8_t>(words[i]), static_cast<uint8_t>(words[i] >> 8),
                static_cast<uint8_t>(words[i] >> 16), static_cast<uint8_t>(words[i] >> 24)};
    }
    return out;
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
};

TEST_F(RenderAtlasGpu, BlitAndMean) {
  const wgpu::TextureDescriptor pageDescriptor{
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .size = {PageSize, PageSize, 1},
      .format = PageFormat,
  };
  const wgpu::Texture page = m_device->device.CreateTexture(&pageDescriptor);
  const wgpu::BufferDescriptor meansDescriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                               .size = MeansBytes};
  const wgpu::Buffer means = m_device->device.CreateBuffer(&meansDescriptor);
  Snapshot snapshot;
  snapshot.blits.push_back(blit_for(CheckerTile, source(CheckerSize, &checker)));
  snapshot.blits.push_back(blit_for(SmallTile, source(SmallSize, &gradient)));
  EXPECT_EQ(snapshot.blits[0].taps, 2u);
  EXPECT_EQ(snapshot.blits[1].taps, 1u);
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  ASSERT_TRUE(testing::encode_blits(encoder, page.CreateView(), {means, 0, MeansBytes}, snapshot));
  const wgpu::CommandBuffer commands = encoder.Finish();
  ASSERT_TRUE(test::submit_and_wait(*m_device, commands));

  const std::vector<Rgba> boxed = read_tile(page, CheckerTile);
  ASSERT_EQ(boxed.size(), size_t{TileSize} * TileSize);
  int worstBox = 0;
  std::array<double, 4> sum{};
  for (uint32_t y = 0; y < TileSize; ++y) {
    for (uint32_t x = 0; x < TileSize; ++x) {
      const Rgba got = boxed[y * TileSize + x];
      for (uint32_t c = 0; c < 4; ++c) {
        double box = 0.0;
        for (uint32_t dy = 0; dy < 2; ++dy) {
          for (uint32_t dx = 0; dx < 2; ++dx) {
            box += checker(2 * x + dx, 2 * y + dy)[c];
          }
        }
        worstBox = std::max(worstBox, static_cast<int>(std::abs(got[c] - box / 4.0) + 0.5));
        sum[c] += got[c];
      }
    }
  }
  EXPECT_LE(worstBox, 1);

  const std::vector<Rgba> small = read_tile(page, SmallTile);
  ASSERT_EQ(small.size(), size_t{TileSize} * TileSize);
  int worstUp = 0;
  std::array<double, 4> smallSum{};
  for (uint32_t y = 0; y < TileSize; ++y) {
    for (uint32_t x = 0; x < TileSize; ++x) {
      const double sx = std::clamp((x + 0.5) / 2.0 - 0.5, 0.0, SmallSize - 1.0);
      const double sy = std::clamp((y + 0.5) / 2.0 - 0.5, 0.0, SmallSize - 1.0);
      const auto x0 = static_cast<uint32_t>(sx);
      const auto y0 = static_cast<uint32_t>(sy);
      const uint32_t x1 = std::min(x0 + 1, SmallSize - 1);
      const uint32_t y1 = std::min(y0 + 1, SmallSize - 1);
      const double fx = sx - x0;
      const double fy = sy - y0;
      const Rgba got = small[y * TileSize + x];
      for (uint32_t c = 0; c < 4; ++c) {
        const double top = gradient(x0, y0)[c] * (1.0 - fx) + gradient(x1, y0)[c] * fx;
        const double bottom = gradient(x0, y1)[c] * (1.0 - fx) + gradient(x1, y1)[c] * fx;
        const double expected = top * (1.0 - fy) + bottom * fy;
        worstUp = std::max(worstUp, static_cast<int>(std::abs(got[c] - expected) + 0.5));
        smallSum[c] += got[c];
      }
    }
  }
  EXPECT_LE(worstUp, 1);

  const std::vector<uint32_t> meanWords =
      test::read_buffer(*m_device, means, 0, MeansBytes).value_or(std::vector<uint32_t>{});
  ASSERT_EQ(meanWords.size(), MaxTiles);
  constexpr double Texels = double{TileSize} * TileSize;
  for (uint32_t c = 0; c < 4; ++c) {
    const int checkerMean = static_cast<int>((meanWords[CheckerTile] >> (8 * c)) & 0xFFu);
    const int smallMean = static_cast<int>((meanWords[SmallTile] >> (8 * c)) & 0xFFu);
    EXPECT_LE(std::abs(checkerMean - sum[c] / Texels), 1.0) << c;
    EXPECT_LE(std::abs(smallMean - smallSum[c] / Texels), 1.0) << c;
  }
  EXPECT_EQ(meanWords[1], 0u);
}
} // namespace
} // namespace aurora::render::scene::atlas

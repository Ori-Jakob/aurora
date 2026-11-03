#include "gfx/texture.hpp"
#include "gfx/texture_convert.hpp"
#include "gfx/texture_replacement.hpp"
#include "internal.hpp"
#include "render/material/mip_repair.hpp"
#include "webgpu/gpu.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <thread>
#include <vector>

namespace aurora::render::material {
namespace {
using Format = wgpu::TextureFormat;

constexpr std::array AllFilters{MipFilter::Colour, MipFilter::Linear, MipFilter::MatNormal, MipFilter::NormalRg};

std::vector<uint8_t> noise_bytes(size_t size, uint32_t seed) {
  std::vector<uint8_t> bytes(size);
  uint32_t state = seed * 2654435761u + 1u;
  for (auto& value : bytes) {
    state = state * 1664525u + 1013904223u;
    value = static_cast<uint8_t>(state >> 24);
  }
  return bytes;
}

std::optional<std::vector<uint8_t>> tail_of(Format format, uint32_t width, uint32_t height,
                                            const std::vector<uint8_t>& level, MipFilter filter) {
  const auto tail = generate_mip_tail({
      .format = format,
      .width = width,
      .height = height,
      .level = {level.data(), level.size()},
      .filter = filter,
  });
  if (!tail) {
    return std::nullopt;
  }
  return std::vector<uint8_t>(tail->data(), tail->data() + tail->size());
}

std::vector<uint8_t> decoded_tail(Format format, uint32_t width, uint32_t height, const std::vector<uint8_t>& tail) {
  std::vector<uint8_t> out;
  size_t offset = 0;
  while (width > 1 || height > 1) {
    width = std::max(width >> 1, 1u);
    height = std::max(height >> 1, 1u);
    const size_t size = gfx::calc_texture_size(format, width, height, 1);
    const auto rgba = decode_level(format, width, height, {tail.data() + offset, size});
    if (!rgba) {
      return {};
    }
    out.insert(out.end(), rgba->data(), rgba->data() + rgba->size());
    offset += size;
  }
  return out;
}

uint64_t squared_error(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  uint64_t sum = 0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    const int64_t d = int64_t{a[i]} - int64_t{b[i]};
    sum += static_cast<uint64_t>(d * d);
  }
  return sum;
}

struct Normal {
  double x, y, z;
};

Normal unit_of(double x, double y) {
  const double z = std::sqrt(std::max(0.0, 1.0 - x * x - y * y));
  const double length = std::sqrt(x * x + y * y + z * z);
  return {x / length, y / length, z / length};
}

Normal mean_direction(std::initializer_list<Normal> normals) {
  Normal sum{0, 0, 0};
  for (const Normal& n : normals) {
    sum = {sum.x + n.x, sum.y + n.y, sum.z + n.z};
  }
  const double length = std::sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
  return {sum.x / length, sum.y / length, sum.z / length};
}

double mat_decode(uint8_t code) { return (code - 128.0) / 127.0; }
int mat_encode(double v) { return static_cast<int>(std::lround(128.0 + v * 127.0)); }
double rg_decode(uint8_t code) { return code / 127.5 - 1.0; }
int rg_encode(double v) { return static_cast<int>(std::lround((v + 1.0) * 127.5)); }

TEST(MaterialMipRepair, LinearFilterUsesLinearWeights) {
  constexpr uint32_t Size = 64;
  const auto level = noise_bytes(gfx::calc_texture_size(Format::BC7RGBAUnorm, Size, Size, 1), 7);
  const auto colour = tail_of(Format::BC7RGBAUnorm, Size, Size, level, MipFilter::Colour);
  const auto linear = tail_of(Format::BC7RGBAUnorm, Size, Size, level, MipFilter::Linear);
  ASSERT_TRUE(colour.has_value());
  ASSERT_TRUE(linear.has_value());
  EXPECT_NE(*colour, *linear);

  const auto source = decode_level(Format::BC7RGBAUnorm, Size, Size, {level.data(), level.size()});
  ASSERT_TRUE(source.has_value());
  const auto box = gfx::box_mip_tail_rgba8(Size, Size, *source);
  const std::vector<uint8_t> reference(box.data(), box.data() + box.size());
  const auto colourRgba = decoded_tail(Format::BC7RGBAUnorm, Size, Size, *colour);
  const auto linearRgba = decoded_tail(Format::BC7RGBAUnorm, Size, Size, *linear);
  ASSERT_EQ(colourRgba.size(), reference.size());
  ASSERT_EQ(linearRgba.size(), reference.size());
  EXPECT_LT(squared_error(linearRgba, reference), squared_error(colourRgba, reference));

  const auto rgba = noise_bytes(size_t{Size} * Size * 4, 8);
  EXPECT_EQ(tail_of(Format::RGBA8Unorm, Size, Size, rgba, MipFilter::Linear),
            tail_of(Format::RGBA8Unorm, Size, Size, rgba, MipFilter::Colour));
}

TEST(MaterialMipRepair, MatNormalFilterRenormalises) {
  constexpr uint32_t Size = 4;
  std::vector<uint8_t> level(Size * Size * 4);
  for (uint32_t y = 0; y < Size; ++y) {
    for (uint32_t x = 0; x < Size; ++x) {
      uint8_t* texel = level.data() + (y * Size + x) * 4;
      const bool odd = ((x + y) & 1) != 0;
      texel[0] = odd ? 10 : 250;
      texel[1] = odd ? 204 : 128;
      texel[2] = odd ? 40 : 100;
      texel[3] = odd ? 128 : 204;
    }
  }
  const auto tail = tail_of(Format::RGBA8Unorm, Size, Size, level, MipFilter::MatNormal);
  ASSERT_TRUE(tail.has_value());
  ASSERT_EQ(tail->size(), (4u + 1u) * 4u);

  const Normal expected =
      mean_direction({unit_of(mat_decode(204), mat_decode(128)), unit_of(mat_decode(128), mat_decode(204)),
                      unit_of(mat_decode(204), mat_decode(128)), unit_of(mat_decode(128), mat_decode(204))});
  for (uint32_t i = 0; i < 5; ++i) {
    SCOPED_TRACE(i);
    const uint8_t* texel = tail->data() + i * 4;
    EXPECT_NEAR(texel[3], mat_encode(expected.x), 1);
    EXPECT_NEAR(texel[1], mat_encode(expected.y), 1);
    EXPECT_NEAR(texel[0], 130, 1);
    EXPECT_NEAR(texel[2], 70, 1);
    // A box over the codes would give 166 and a reconstructed normal pointing elsewhere
    EXPECT_GE(std::abs(int{texel[3]} - 166), 3);
    const double x = mat_decode(texel[3]);
    const double y = mat_decode(texel[1]);
    EXPECT_LE(x * x + y * y, 1.0);
  }

  const std::vector<uint8_t> outside(2 * 2 * 4, 255);
  const auto clamped = tail_of(Format::RGBA8Unorm, 2, 2, outside, MipFilter::MatNormal);
  ASSERT_TRUE(clamped.has_value());
  const double x = mat_decode((*clamped)[3]);
  const double y = mat_decode((*clamped)[1]);
  EXPECT_LE(x * x + y * y, 1.0 + 2.0 / 127.0);
  EXPECT_NEAR((*clamped)[3], mat_encode(std::sqrt(0.5)), 1);

  for (const Format format : {Format::RGBA8Unorm, Format::BGRA8Unorm}) {
    SCOPED_TRACE(static_cast<uint32_t>(format));
    const uint32_t xChannel = format == Format::BGRA8Unorm ? 2 : 0;
    std::vector<uint8_t> rg(2 * 2 * 4, 77);
    for (uint32_t i = 0; i < 4; ++i) {
      rg[i * 4 + xChannel] = (i & 1) != 0 ? 204 : 128;
      rg[i * 4 + 1] = (i & 1) != 0 ? 128 : 204;
    }
    const auto rgTail = tail_of(format, 2, 2, rg, MipFilter::NormalRg);
    ASSERT_TRUE(rgTail.has_value());
    const Normal rgExpected =
        mean_direction({unit_of(rg_decode(204), rg_decode(128)), unit_of(rg_decode(128), rg_decode(204))});
    EXPECT_NEAR((*rgTail)[xChannel], rg_encode(rgExpected.x), 1);
    EXPECT_NEAR((*rgTail)[1], rg_encode(rgExpected.y), 1);
    EXPECT_EQ((*rgTail)[3], 77);
  }
}

TEST(MaterialMipRepair, Bc4Bc5StillTruncate) {
  for (const Format format : {Format::BC4RUnorm, Format::BC5RGUnorm, Format::BC1RGBAUnorm, Format::BC3RGBAUnorm}) {
    SCOPED_TRACE(static_cast<uint32_t>(format));
    EXPECT_FALSE(can_repair(format));
    const auto level = noise_bytes(gfx::calc_texture_size(format, 16, 16, 1), 3);
    for (const MipFilter filter : AllFilters) {
      EXPECT_FALSE(tail_of(format, 16, 16, level, filter).has_value());
    }
  }
}

TEST(MaterialMipRepair, SameBuildDeterministic) {
  struct Case {
    Format format;
    uint32_t size;
  };
  for (const auto& [format, size] : {Case{Format::BC7RGBAUnorm, 64}, Case{Format::RGBA8Unorm, 32}}) {
    const auto level = noise_bytes(gfx::calc_texture_size(format, size, size, 1), 11);
    for (const MipFilter filter : AllFilters) {
      SCOPED_TRACE(fmt::format("{} {}", static_cast<uint32_t>(format), static_cast<uint32_t>(filter)));
      const auto first = tail_of(format, size, size, level, filter);
      ASSERT_TRUE(first.has_value());
      std::array<std::optional<std::vector<uint8_t>>, 2> threaded;
      std::thread a([&] { threaded[0] = tail_of(format, size, size, level, filter); });
      std::thread b([&] { threaded[1] = tail_of(format, size, size, level, filter); });
      a.join();
      b.join();
      EXPECT_EQ(tail_of(format, size, size, level, filter), first);
      EXPECT_EQ(threaded[0], first);
      EXPECT_EQ(threaded[1], first);
    }
  }
}

TEST(MaterialMipRepair, C001MatChainComesBackFull) {
  namespace tr = gfx::texture_replacement;
  const bool savedBc = webgpu::g_bcTexturesSupported;
  webgpu::g_bcTexturesSupported = true;
  const auto path =
      std::filesystem::path{AURORA_TEST_FIXTURES_DIR} / "materials" / "dds" / "tex1_16x16_000000000000c001_14.mat.dds";
  tr::CompanionLoadError error = tr::CompanionLoadError::Failed;
  tr::ChainReport chain;
  const auto loaded = tr::load_companion(tr::CompanionRole::Mat, path, &error, &chain);
  webgpu::g_bcTexturesSupported = savedBc;
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(error, tr::CompanionLoadError::None);
  EXPECT_EQ(chain.verdict, tr::ChainVerdict::Clamped);
  EXPECT_TRUE(chain.repaired);
  EXPECT_EQ(chain.keptLevels, 2u);
  EXPECT_EQ(chain.generatedLevels, 4u);
  EXPECT_EQ(loaded->mips, 6u);
  EXPECT_EQ(loaded->data.size(), gfx::calc_texture_size(loaded->format, 32, 32, 6));
}
} // namespace
} // namespace aurora::render::material

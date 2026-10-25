#include "gfx/dds_io.hpp"
#include "gfx/texture_replacement.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/texture.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gfx::texture_replacement {
namespace {
using namespace std::chrono_literals;

constexpr const char* VirtualPathA = "tex1_4x4_0000000000000001_6.dds";
constexpr const char* VirtualPathB = "tex1_4x4_0000000000000002_6.dds";
constexpr const char* VirtualPngPath = "tex1_4x4_0000000000000001_6.png";

struct BlockingSource {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  std::atomic<uint32_t> reads = 0;

  static bool read(void* userData, const char*, std::vector<uint8_t>&) {
    auto& source = *static_cast<BlockingSource*>(userData);
    ++source.reads;
    std::unique_lock lock{source.mutex};
    source.entered = true;
    source.cv.notify_all();
    source.cv.wait(lock, [&] { return source.release; });
    return false;
  }
};

struct RecordingSource {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::string> paths;

  static bool read(void* userData, const char* path, std::vector<uint8_t>&) {
    auto& source = *static_cast<RecordingSource*>(userData);
    {
      std::lock_guard lock{source.mutex};
      source.paths.emplace_back(path);
    }
    source.cv.notify_all();
    return false;
  }
};

class ReplacementStreamingTest : public ::testing::Test {
protected:
  void SetUp() override {
    shutdown();
    testing::set_worker_count(1);
  }

  void TearDown() override { shutdown(); }
};

TEST_F(ReplacementStreamingTest, UnregisterWaitsForInFlightVirtualRead) {
  BlockingSource source;
  const auto registration =
      texture::register_virtual_replacement(VirtualPathA, {.read = &BlockingSource::read, .userData = &source});
  ASSERT_NE(registration.id, 0u);
  {
    std::unique_lock lock{source.mutex};
    ASSERT_TRUE(source.cv.wait_for(lock, 2s, [&] { return source.entered; }));
  }

  auto unregister = std::async(std::launch::async, [&] { texture::unregister_replacement(registration); });
  EXPECT_EQ(unregister.wait_for(50ms), std::future_status::timeout);
  {
    std::lock_guard lock{source.mutex};
    source.release = true;
  }
  source.cv.notify_all();

  EXPECT_EQ(unregister.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(source.reads.load(), 1u);
}

TEST_F(ReplacementStreamingTest, CancelledQueuedVirtualJobNeverReads) {
  RecordingSource source;
  testing::set_workers_paused(true);
  const auto registration =
      texture::register_virtual_replacement(VirtualPathA, {.read = &RecordingSource::read, .userData = &source});
  ASSERT_NE(registration.id, 0u);

  texture::unregister_replacement(registration);
  testing::set_workers_paused(false);
  shutdown();

  EXPECT_TRUE(source.paths.empty());
}

TEST_F(ReplacementStreamingTest, HighPriorityFullLoadRunsBeforeThumbnailQueue) {
  RecordingSource source;
  testing::set_workers_paused(true);
  const auto low =
      texture::register_virtual_replacement(VirtualPathA, {.read = &RecordingSource::read, .userData = &source});
  const auto high =
      texture::register_virtual_replacement(VirtualPathB, {.read = &RecordingSource::read, .userData = &source});
  ASSERT_NE(low.id, 0u);
  ASSERT_NE(high.id, 0u);

  GXTexObj_ obj{};
  const auto key = std::get<texture::TextureSourceKey>(high.key);
  ASSERT_TRUE(find_source_replacement(obj, key).has_value());
  testing::set_workers_paused(false);
  {
    std::unique_lock lock{source.mutex};
    ASSERT_TRUE(source.cv.wait_for(lock, 2s, [&] { return !source.paths.empty(); }));
    EXPECT_EQ(source.paths.front(), VirtualPathB);
  }
  const std::array registrations{low, high};
  texture::unregister_replacements(registrations);
}

TEST_F(ReplacementStreamingTest, StaleFailureDoesNotPoisonNewRegistration) {
  RecordingSource source;
  testing::set_workers_paused(true);
  const auto oldRegistration =
      texture::register_virtual_replacement(VirtualPngPath, {.read = &RecordingSource::read, .userData = &source});
  ASSERT_NE(oldRegistration.id, 0u);
  GXTexObj_ obj{};
  const auto key = std::get<texture::TextureSourceKey>(oldRegistration.key);
  ASSERT_TRUE(find_source_replacement(obj, key).has_value());
  testing::set_workers_paused(false);
  ASSERT_TRUE(testing::wait_for_completions(oldRegistration.id, 1, 2000));
  texture::unregister_replacement(oldRegistration);

  testing::set_workers_paused(true);
  const auto newRegistration =
      texture::register_virtual_replacement(VirtualPngPath, {.read = &RecordingSource::read, .userData = &source});
  process_streaming();

  ASSERT_TRUE(find_source_replacement(obj, key).has_value());
  testing::set_workers_paused(false);
  EXPECT_TRUE(testing::wait_for_completions(newRegistration.id, 1, 2000));
  texture::unregister_replacement(newRegistration);
}

TEST_F(ReplacementStreamingTest, GameThreadKeyMatchesComputeSourceKey) {
  const std::array<uint8_t, 4> pixel{1, 2, 3, 4};
  const texture::RawTextureReplacement raw{.bytes = pixel, .width = 1, .height = 1, .gxFormat = GX_TF_RGBA8_PC};
  const auto make_obj = [](const void* data, u32 format, u32 width, u32 height) {
    GXTexObj_ obj{};
    obj.data = data;
    obj.mWidth = width;
    obj.mHeight = height;
    obj.mFormat = format;
    obj.tlut = GX_TLUT0;
    return obj;
  };
  const auto make_tlut = [](const std::array<uint8_t, 512>& palette) {
    GXTlutObj_ tlut{};
    tlut.data = palette.data();
    tlut.numEntries = 256;
    tlut.format = GX_TL_RGB5A3;
    return tlut;
  };

  std::array<uint8_t, 32> cmpr{};
  for (size_t i = 0; i < cmpr.size(); ++i) {
    cmpr[i] = static_cast<uint8_t>(i * 11 + 3);
  }
  const auto cmprObj = make_obj(cmpr.data(), GX_TF_CMPR, 8, 8);
  const auto cmprKey = texture::compute_source_key(GX_TF_CMPR, 8, 8, cmpr);
  ASSERT_TRUE(cmprKey.has_value());
  EXPECT_FALSE(has_replacement(cmprObj));
  ASSERT_NE(texture::register_replacement(*cmprKey, raw).id, 0u);
  EXPECT_TRUE(has_replacement(cmprObj));
  EXPECT_EQ(build_texture_replacement_name(cmprObj), build_texture_replacement_name(*cmprKey));

  std::array<uint8_t, 256> indices{};
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<uint8_t>(5 + i % 5);
  }
  std::array<uint8_t, 512> paletteA{};
  for (size_t i = 0; i < paletteA.size(); ++i) {
    paletteA[i] = static_cast<uint8_t>(i * 3);
  }
  auto paletteB = paletteA;
  paletteB[7 * sizeof(uint16_t)] ^= 0xff;
  auto paletteC = paletteA;
  paletteC[100 * sizeof(uint16_t)] ^= 0xff;
  const auto c8Obj = make_obj(indices.data(), GX_TF_C8, 16, 16);
  const auto c8Key = texture::compute_source_key(GX_TF_C8, 16, 16, indices, paletteA);
  ASSERT_TRUE(c8Key.has_value());
  ASSERT_NE(texture::register_replacement(*c8Key, raw).id, 0u);
  EXPECT_TRUE(has_replacement(c8Obj, make_tlut(paletteA)));
  EXPECT_TRUE(has_replacement(c8Obj, make_tlut(paletteC)));
  EXPECT_FALSE(has_replacement(c8Obj, make_tlut(paletteB)));
  EXPECT_FALSE(has_replacement(c8Obj));

  auto anyTlut = *c8Key;
  anyTlut.tlutHash = texture::kWildcardTlutHash;
  ASSERT_NE(texture::register_replacement(anyTlut, raw).id, 0u);
  EXPECT_TRUE(has_replacement(c8Obj, make_tlut(paletteB)));
  EXPECT_TRUE(has_replacement(c8Obj));

  const auto emptyObj = make_obj(nullptr, GX_TF_RGBA8_PC, 4, 4);
  EXPECT_FALSE(has_replacement(emptyObj));
  ASSERT_NE(texture::register_replacement(
                texture::TextureSourceKey{
                    .textureHash = texture::kWildcardTextureHash, .width = 4, .height = 4, .format = GX_TF_RGBA8_PC},
                raw)
                .id,
            0u);
  EXPECT_TRUE(has_replacement(emptyObj));
}

template <typename T>
void write_u32(std::vector<uint8_t>& bytes, size_t offset, T value) {
  const uint32_t word = static_cast<uint32_t>(value);
  std::memcpy(bytes.data() + offset, &word, sizeof(word));
}

TEST(DdsMipTailTest, SlicesTheSameTailAsFullDecode) {
  std::vector<uint8_t> base(8 * 8 * 4);
  auto encoded = dds::encode_rgba8_dds(8, 8, base);
  std::vector<uint8_t> bytes(encoded.data(), encoded.data() + encoded.size());
  bytes.resize(128 + (8 * 8 + 4 * 4 + 2 * 2 + 1) * 4);
  for (size_t i = 128; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i);
  }
  write_u32(bytes, 8, 0x1u | 0x2u | 0x4u | 0x8u | 0x1000u | 0x20000u);
  write_u32(bytes, 28, 4u);
  write_u32(bytes, 108, 0x1000u | 0x8u | 0x400000u);

  const auto full = dds::parse_dds_bytes({bytes.data(), bytes.size()});
  const auto tail = dds::parse_dds_mip_tail({bytes.data(), bytes.size()}, 2);
  const auto path =
      std::filesystem::temp_directory_path() /
      ("aurora-dds-tail-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dds");
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  const auto fileTail = dds::load_dds_mip_tail(path, 2);
  std::error_code removeError;
  std::filesystem::remove(path, removeError);
  ASSERT_TRUE(full.has_value());
  ASSERT_TRUE(tail.has_value());
  ASSERT_TRUE(fileTail.has_value());
  EXPECT_FALSE(tail->includesBase);
  EXPECT_EQ(tail->texture.width, 2u);
  EXPECT_EQ(tail->texture.height, 2u);
  EXPECT_EQ(tail->texture.mips, 2u);
  ASSERT_EQ(tail->texture.data.size(), 20u);
  EXPECT_TRUE(std::equal(tail->texture.data.data(), tail->texture.data.data() + tail->texture.data.size(),
                         full->data.data() + full->data.size() - 20));
  ASSERT_EQ(fileTail->texture.data.size(), tail->texture.data.size());
  EXPECT_TRUE(std::equal(fileTail->texture.data.data(), fileTail->texture.data.data() + fileTail->texture.data.size(),
                         tail->texture.data.data()));
}

TEST(DdsMipTailTest, RejectsMiplessLargeTextureButKeepsSmallWholeTexture) {
  std::vector<uint8_t> largePixels(128 * 128 * 4);
  const auto large = dds::encode_rgba8_dds(128, 128, largePixels);
  EXPECT_FALSE(dds::parse_dds_mip_tail(large, 64).has_value());

  std::vector<uint8_t> smallPixels(32 * 16 * 4);
  const auto small = dds::encode_rgba8_dds(32, 16, smallPixels);
  const auto tail = dds::parse_dds_mip_tail(small, 64);
  ASSERT_TRUE(tail.has_value());
  EXPECT_TRUE(tail->includesBase);
  ASSERT_EQ(tail->texture.data.size(), smallPixels.size());
  EXPECT_TRUE(
      std::equal(tail->texture.data.data(), tail->texture.data.data() + tail->texture.data.size(), smallPixels.data()));
}

using Format = wgpu::TextureFormat;

std::vector<uint8_t> make_dx10_dds(uint32_t dxgi, uint32_t width, uint32_t height, uint32_t mips, size_t dataSize,
                                   uint8_t seed = 0) {
  std::vector<uint8_t> bytes(148 + dataSize);
  write_u32(bytes, 0, 0x20534444u);
  write_u32(bytes, 4, 124u);
  write_u32(bytes, 8, 0x81007u | (mips > 1 ? 0x20000u : 0u));
  write_u32(bytes, 12, height);
  write_u32(bytes, 16, width);
  write_u32(bytes, 28, mips);
  write_u32(bytes, 76, 32u);
  write_u32(bytes, 80, 0x4u);
  write_u32(bytes, 84, 0x30315844u);
  write_u32(bytes, 108, 0x1000u | (mips > 1 ? 0x400008u : 0u));
  write_u32(bytes, 128, dxgi);
  write_u32(bytes, 132, 3u);
  write_u32(bytes, 140, 1u);
  for (size_t i = 0; i < dataSize; ++i) {
    bytes[148 + i] = static_cast<uint8_t>(i * 7 + seed);
  }
  return bytes;
}

std::vector<uint8_t> make_dds(uint32_t dxgi, Format format, uint32_t width, uint32_t height, uint32_t mips = 1,
                              uint8_t seed = 0) {
  return make_dx10_dds(dxgi, width, height, mips, calc_texture_size(format, width, height, mips), seed);
}

std::string virtual_name(uint32_t hash) { return fmt::format("tex1_16x16_{:016x}_14.dds", hash); }

std::string virtual_mip_name(uint32_t hash, uint32_t mip) {
  return fmt::format("tex1_16x16_{:016x}_14_mip{}.dds", hash, mip);
}

struct MapSource {
  std::map<std::string, std::vector<uint8_t>, std::less<>> files;
  std::string gatePath;
  std::mutex mutex;
  std::condition_variable cv;
  bool gateEntered = false;
  bool gateReleased = false;
  std::vector<std::string> reads;

  static bool read(void* userData, const char* path, std::vector<uint8_t>& outBytes) {
    auto& source = *static_cast<MapSource*>(userData);
    std::unique_lock lock{source.mutex};
    source.reads.emplace_back(path);
    if (!source.gateEntered && !source.gatePath.empty() && source.gatePath == path) {
      source.gateEntered = true;
      source.cv.notify_all();
      source.cv.wait(lock, [&] { return source.gateReleased; });
    }
    const auto it = source.files.find(std::string_view{path});
    if (it == source.files.end()) {
      return false;
    }
    outBytes = it->second;
    return true;
  }

  bool wait_for_gate() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 2s, [&] { return gateEntered; });
  }

  void release_gate() {
    {
      std::lock_guard lock{mutex};
      gateReleased = true;
    }
    cv.notify_all();
  }

  bool was_read(std::string_view path) {
    std::lock_guard lock{mutex};
    return std::find(reads.begin(), reads.end(), path) != reads.end();
  }

  texture::VirtualFileSource callbacks() { return {.read = &MapSource::read, .userData = this}; }
};

class BcCapsGuard {
public:
  explicit BcCapsGuard(bool supported) : m_saved(webgpu::g_bcTexturesSupported) {
    webgpu::g_bcTexturesSupported = supported;
  }
  ~BcCapsGuard() { webgpu::g_bcTexturesSupported = m_saved; }
  BcCapsGuard(const BcCapsGuard&) = delete;
  BcCapsGuard& operator=(const BcCapsGuard&) = delete;

private:
  bool m_saved;
};

struct CodeFormat {
  uint32_t code;
  Format format;
};

constexpr std::array<CodeFormat, 10> AddedCodes{{
    {29, Format::RGBA8UnormSrgb},
    {72, Format::BC1RGBAUnormSrgb},
    {78, Format::BC3RGBAUnormSrgb},
    {80, Format::BC4RUnorm},
    {81, Format::BC4RSnorm},
    {84, Format::BC5RGSnorm},
    {91, Format::BGRA8UnormSrgb},
    {95, Format::BC6HRGBUfloat},
    {96, Format::BC6HRGBFloat},
    {99, Format::BC7RGBAUnormSrgb},
}};

TEST(DdsFormatTest, Dx10CodeSweep) {
  std::set<uint32_t> expected{28, 29, 71, 72, 77, 78, 80, 81, 83, 84, 87, 91, 95, 96, 98, 99};
  for (uint32_t astc = 134; astc <= 186; astc += 4) {
    expected.insert(astc);
    expected.insert(astc + 1);
  }
  std::set<uint32_t> accepted;
  for (uint32_t code = 0; code < 256; ++code) {
    SCOPED_TRACE(code);
    const auto bytes = make_dx10_dds(code, 4, 4, 1, 64);
    const auto converted = dds::parse_dds_bytes({bytes.data(), bytes.size()});
    if (!converted) {
      continue;
    }
    accepted.insert(code);
    uint64_t blockBytes = 16;
    if (code == 28 || code == 29 || code == 87 || code == 91) {
      blockBytes = 64;
    } else if (code == 71 || code == 72 || code == 80 || code == 81) {
      blockBytes = 8;
    }
    EXPECT_EQ(calc_texture_size(converted->format, 4, 4, 1), blockBytes);
    EXPECT_EQ(converted->data.size(), blockBytes);
  }
  EXPECT_EQ(accepted, expected);
}

TEST(DdsFormatTest, MapsToExactFormatAndTailKeepsIt) {
  for (const auto& [code, format] : AddedCodes) {
    SCOPED_TRACE(code);
    const auto bytes = make_dds(code, format, 128, 128, 8, static_cast<uint8_t>(code));
    const auto full = dds::parse_dds_bytes({bytes.data(), bytes.size()});
    ASSERT_TRUE(full.has_value());
    EXPECT_EQ(full->format, format);
    EXPECT_EQ(full->mips, 8u);
    const auto tail = dds::parse_dds_mip_tail({bytes.data(), bytes.size()}, 64);
    ASSERT_TRUE(tail.has_value());
    EXPECT_FALSE(tail->includesBase);
    EXPECT_EQ(tail->texture.format, format);
    EXPECT_EQ(tail->texture.width, 64u);
    EXPECT_EQ(tail->texture.mips, 7u);
    EXPECT_EQ(tail->texture.data.size(), calc_texture_size(format, 64, 64, 7));
  }
}

constexpr bool keeps_pre_w5_formats() {
  constexpr std::array Formats{Format::RGBA8Unorm,        Format::BGRA8Unorm,       Format::BC1RGBAUnorm,
                               Format::BC3RGBAUnorm,      Format::BC5RGUnorm,       Format::BC7RGBAUnorm,
                               Format::ASTC4x4Unorm,      Format::ASTC4x4UnormSrgb, Format::ASTC8x8UnormSrgb,
                               Format::ASTC12x12UnormSrgb};
  for (const auto format : Formats) {
    if (legacy_storage_format(format) != format || !legacy_base_accepts(format)) {
      return false;
    }
  }
  return true;
}

TEST(DdsFormatTest, LegacyStorageFormatKeepsRawBits) {
  static_assert(legacy_storage_format(Format::RGBA8UnormSrgb) == Format::RGBA8Unorm);
  static_assert(legacy_storage_format(Format::BGRA8UnormSrgb) == Format::BGRA8Unorm);
  static_assert(legacy_storage_format(Format::BC1RGBAUnormSrgb) == Format::BC1RGBAUnorm);
  static_assert(legacy_storage_format(Format::BC3RGBAUnormSrgb) == Format::BC3RGBAUnorm);
  static_assert(legacy_storage_format(Format::BC7RGBAUnormSrgb) == Format::BC7RGBAUnorm);
  static_assert(legacy_storage_format(Format::BC4RUnorm) == Format::BC4RUnorm);
  static_assert(legacy_storage_format(Format::BC6HRGBFloat) == Format::BC6HRGBFloat);
  static_assert(keeps_pre_w5_formats());
  static_assert(!legacy_base_accepts(Format::BC4RUnorm) && !legacy_base_accepts(Format::BC4RSnorm) &&
                !legacy_base_accepts(Format::BC5RGSnorm) && !legacy_base_accepts(Format::BC6HRGBUfloat) &&
                !legacy_base_accepts(Format::BC6HRGBFloat));
  static_assert(legacy_base_accepts(Format::BC7RGBAUnormSrgb) && legacy_base_accepts(Format::RGBA8UnormSrgb) &&
                legacy_base_accepts(Format::BGRA8UnormSrgb) && legacy_base_accepts(Format::BC1RGBAUnormSrgb) &&
                legacy_base_accepts(Format::BC3RGBAUnormSrgb));
  SUCCEED();
}

TEST_F(ReplacementStreamingTest, SrgbBaseLoadsWithExactFormat) {
  BcCapsGuard caps{true};
  testing::set_workers_paused(true);
  constexpr std::array<CodeFormat, 5> Srgb{{
      {29, Format::RGBA8UnormSrgb},
      {72, Format::BC1RGBAUnormSrgb},
      {78, Format::BC3RGBAUnormSrgb},
      {91, Format::BGRA8UnormSrgb},
      {99, Format::BC7RGBAUnormSrgb},
  }};
  for (const auto& [code, format] : Srgb) {
    SCOPED_TRACE(code);
    MapSource source;
    const auto base = make_dds(code, format, 16, 16, 1, 1);
    const auto mip1 = make_dds(code, format, 8, 8, 1, 2);
    source.files[virtual_name(code)] = base;
    source.files[virtual_mip_name(code, 1)] = mip1;
    const auto registration = texture::register_virtual_replacement(virtual_name(code), source.callbacks());
    ASSERT_NE(registration.id, 0u);
    const auto loaded = testing::load_replacement_texture(registration);
    texture::unregister_replacement(registration);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->format, format);
    EXPECT_EQ(loaded->width, 16u);
    EXPECT_EQ(loaded->mips, 2u);
    std::vector<uint8_t> expected(base.begin() + 148, base.end());
    expected.insert(expected.end(), mip1.begin() + 148, mip1.end());
    ASSERT_EQ(loaded->data.size(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), loaded->data.data()));
  }
}

TEST_F(ReplacementStreamingTest, MixedSrgbChainStopsAtFormatChange) {
  BcCapsGuard caps{true};
  testing::set_workers_paused(true);
  MapSource source;
  source.files[virtual_name(0x99)] = make_dds(99, Format::BC7RGBAUnormSrgb, 16, 16);
  source.files[virtual_mip_name(0x99, 1)] = make_dds(98, Format::BC7RGBAUnorm, 8, 8);
  source.files[virtual_mip_name(0x99, 2)] = make_dds(99, Format::BC7RGBAUnormSrgb, 4, 4);
  const auto registration = texture::register_virtual_replacement(virtual_name(0x99), source.callbacks());
  ASSERT_NE(registration.id, 0u);
  const auto loaded = testing::load_replacement_texture(registration);
  texture::unregister_replacement(registration);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->format, Format::BC7RGBAUnormSrgb);
  EXPECT_EQ(loaded->mips, 1u);
  EXPECT_EQ(loaded->data.size(), calc_texture_size(Format::BC7RGBAUnormSrgb, 16, 16, 1));
}

TEST_F(ReplacementStreamingTest, MaterialOnlyFormatsAreRejectedAsBase) {
  BcCapsGuard caps{true};
  testing::set_workers_paused(true);
  constexpr std::array<CodeFormat, 6> Bases{{
      {80, Format::BC4RUnorm},
      {81, Format::BC4RSnorm},
      {84, Format::BC5RGSnorm},
      {95, Format::BC6HRGBUfloat},
      {96, Format::BC6HRGBFloat},
      {83, Format::BC5RGUnorm},
  }};
  for (const auto& [code, format] : Bases) {
    SCOPED_TRACE(code);
    MapSource source;
    source.files[virtual_name(code)] = make_dds(code, format, 8, 8);
    const auto registration = texture::register_virtual_replacement(virtual_name(code), source.callbacks());
    ASSERT_NE(registration.id, 0u);
    const auto loaded = testing::load_replacement_texture(registration);
    texture::unregister_replacement(registration);
    EXPECT_EQ(loaded.has_value(), code == 83);
    if (loaded) {
      EXPECT_EQ(loaded->format, format);
    }
  }
}

struct CapturedLogs {
  std::mutex mutex;
  std::vector<std::pair<AuroraLogLevel, std::string>> lines;
};

CapturedLogs g_capturedLogs;

void capture_log(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  std::lock_guard lock{g_capturedLogs.mutex};
  g_capturedLogs.lines.emplace_back(level, std::string{message, len});
}

// Restores the previous log callback even when an assertion leaves the test early
class ScopedWarningCapture {
public:
  ScopedWarningCapture() noexcept : m_previous(g_config.logCallback) {
    {
      std::lock_guard lock{g_capturedLogs.mutex};
      g_capturedLogs.lines.clear();
    }
    g_config.logCallback = &capture_log;
  }
  ~ScopedWarningCapture() { g_config.logCallback = m_previous; }
  ScopedWarningCapture(const ScopedWarningCapture&) = delete;
  ScopedWarningCapture& operator=(const ScopedWarningCapture&) = delete;

  [[nodiscard]] std::vector<std::string> lines(AuroraLogLevel level, std::string_view text = {}) const {
    std::lock_guard lock{g_capturedLogs.mutex};
    std::vector<std::string> out;
    for (const auto& [lineLevel, line] : g_capturedLogs.lines) {
      if (lineLevel == level && line.find(text) != std::string::npos) {
        out.push_back(line);
      }
    }
    return out;
  }
  [[nodiscard]] size_t warnings(std::string_view text = {}) const { return lines(LOG_WARNING, text).size(); }
  [[nodiscard]] size_t debugs(std::string_view text) const { return lines(LOG_DEBUG, text).size(); }

private:
  AuroraLogCallback m_previous;
};

TEST_F(ReplacementStreamingTest, BcRejectedWithoutCaps) {
  BcCapsGuard caps{false};
  testing::set_workers_paused(true);
  constexpr std::array<CodeFormat, 4> Bases{{
      {98, Format::BC7RGBAUnorm},
      {99, Format::BC7RGBAUnormSrgb},
      {80, Format::BC4RUnorm},
      {29, Format::RGBA8UnormSrgb},
  }};
  for (const auto& [code, format] : Bases) {
    SCOPED_TRACE(code);
    MapSource source;
    source.files[virtual_name(code)] = make_dds(code, format, 8, 8);
    const auto registration = texture::register_virtual_replacement(virtual_name(code), source.callbacks());
    ASSERT_NE(registration.id, 0u);
    ScopedWarningCapture capture;
    const auto loaded = testing::load_replacement_texture(registration);
    texture::unregister_replacement(registration);
    EXPECT_EQ(loaded.has_value(), code == 29);
    // The material-map check runs before the BC caps check, so code 80 never reaches the format warning
    EXPECT_EQ(capture.warnings("only valid for material maps"), code == 80 ? 1u : 0u);
    EXPECT_EQ(capture.warnings("unsupported format"), code == 98 || code == 99 ? 1u : 0u);
  }
}

TEST_F(ReplacementStreamingTest, DescriptorsUseStorageTwin) {
  constexpr std::array<std::pair<Format, Format>, 7> Cases{{
      {Format::BC7RGBAUnormSrgb, Format::BC7RGBAUnorm},
      {Format::BC1RGBAUnormSrgb, Format::BC1RGBAUnorm},
      {Format::BC3RGBAUnormSrgb, Format::BC3RGBAUnorm},
      {Format::RGBA8UnormSrgb, Format::RGBA8Unorm},
      {Format::BGRA8UnormSrgb, Format::BGRA8Unorm},
      {Format::BC7RGBAUnorm, Format::BC7RGBAUnorm},
      {Format::ASTC4x4UnormSrgb, Format::ASTC4x4UnormSrgb},
  }};
  for (const auto& [format, storage] : Cases) {
    SCOPED_TRACE(static_cast<uint32_t>(format));
    const ConvertedTexture texture{.format = format, .width = 32, .height = 16, .mips = 3};
    const auto descriptors = replacement_texture_descriptors(texture, "replacement");
    EXPECT_EQ(descriptors.texture.format, storage);
    EXPECT_EQ(descriptors.view.format, storage);
    EXPECT_EQ(descriptors.texture.viewFormatCount, 0u);
    EXPECT_EQ(descriptors.texture.viewFormats, nullptr);
    EXPECT_EQ(descriptors.texture.usage, wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst);
    EXPECT_EQ(descriptors.texture.size.width, 32u);
    EXPECT_EQ(descriptors.texture.size.height, 16u);
    EXPECT_EQ(descriptors.texture.size.depthOrArrayLayers, 1u);
    EXPECT_EQ(descriptors.texture.mipLevelCount, 3u);
    EXPECT_EQ(descriptors.view.mipLevelCount, 3u);
    EXPECT_EQ(descriptors.view.dimension, wgpu::TextureViewDimension::e2D);
    EXPECT_EQ(std::string_view{descriptors.texture.label}, "replacement");
  }
}

TEST_F(ReplacementStreamingTest, MaterialOnlyBaseNeverStoresThumbnail) {
  BcCapsGuard caps{true};
  MapSource source;
  source.files[virtual_name(0x80)] = make_dds(80, Format::BC4RUnorm, 8, 8);
  source.files[virtual_name(0x98)] = make_dds(98, Format::BC7RGBAUnorm, 8, 8);
  const auto bc4 = texture::register_virtual_replacement(virtual_name(0x80), source.callbacks());
  const auto control = texture::register_virtual_replacement(virtual_name(0x98), source.callbacks());
  ASSERT_NE(bc4.id, 0u);
  ASSERT_NE(control.id, 0u);
  ASSERT_TRUE(testing::wait_for_completions(bc4.id, 1, 2000));
  ASSERT_TRUE(testing::wait_for_completions(control.id, 1, 2000));
  EXPECT_FALSE(testing::has_thumbnail(bc4.id));
  EXPECT_TRUE(testing::has_thumbnail(control.id));
  const std::array registrations{bc4, control};
  texture::unregister_replacements(registrations);
}

constexpr std::string_view StemA = "tex1_16x16_000000000000a001_14";
constexpr texture::TextureSourceKey KeyA{.textureHash = 0xa001, .width = 16, .height = 16, .format = 14};

constexpr uint32_t role_bit(CompanionRole role) { return 1u << static_cast<uint32_t>(role); }

std::string stem_a(std::string_view suffix) { return std::string{StemA} + std::string{suffix}; }

class CompanionScanTest : public ::testing::Test {
protected:
  void SetUp() override {
    shutdown();
    testing::set_worker_count(1);
    testing::set_workers_paused(true);
    set_bypass(false);
    m_base = std::filesystem::temp_directory_path() /
             ("aurora-companions-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    m_root = m_base / "pack";
    std::filesystem::create_directories(m_root);
    m_cachePath = (m_base / "cache").string();
    m_savedCachePath = g_config.cachePath;
    g_config.cachePath = m_cachePath.c_str();
  }

  void TearDown() override {
    set_bypass(false);
    g_config.cachePath = m_savedCachePath;
    texture::clear_material_manifests();
    shutdown();
    std::error_code ec;
    std::filesystem::remove_all(m_base, ec);
  }

  std::filesystem::path write_file(const std::filesystem::path& relative, std::string_view text = {}) const {
    const auto path = m_root / relative;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    return path;
  }

  std::filesystem::path write_dds(const std::filesystem::path& relative) const {
    std::vector<uint8_t> pixels(8 * 8 * 4, 0x80);
    const auto encoded = dds::encode_rgba8_dds(8, 8, pixels);
    return write_file(relative, {reinterpret_cast<const char*>(encoded.data()), encoded.size()});
  }

  std::filesystem::path m_base;
  std::filesystem::path m_root;
  std::string m_cachePath;
  const char* m_savedCachePath = nullptr;
};

TEST_F(CompanionScanTest, PairsByKeyAcrossFoldersAndKeepsBaseRegistrations) {
  write_file(std::filesystem::path{"BASE"} / stem_a(".dds"));
  const auto nrm = write_file(std::filesystem::path{"MATERIALMAPS"} / stem_a(".nrm.dds"));
  const auto mat = write_file(std::filesystem::path{"MATERIALMAPS"} / stem_a(".mat.dds"));
  const auto orm = write_file(std::filesystem::path{"other"} / stem_a(".orm.dds"));
  write_file(std::filesystem::path{"other"} / stem_a("_mip1.orm.dds"));
  write_file(std::filesystem::path{"other"} / stem_a(".height.dds"));
  write_file(std::filesystem::path{"other"} / "notes.txt");
  const uint64_t generation = companion_generation();

  const auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.registrations.size(), 1u);
  EXPECT_EQ(std::get<texture::TextureSourceKey>(group.registrations[0].key), KeyA);
  EXPECT_TRUE(should_build_source_key());
  ASSERT_EQ(group.companions.size(), 3u);
  EXPECT_TRUE(group.manifests.empty());
  EXPECT_EQ(companion_generation(), generation + 1);
  for (const auto& companion : group.companions) {
    EXPECT_NE(companion.id, 0u);
    EXPECT_NE(companion.id, group.registrations[0].id);
    EXPECT_EQ(std::get<texture::TextureSourceKey>(companion.key), KeyA);
  }

  EXPECT_EQ(companion_roles(KeyA),
            role_bit(CompanionRole::Nrm) | role_bit(CompanionRole::Orm) | role_bit(CompanionRole::Mat));
  const auto foundNrm = find_companion(KeyA, CompanionRole::Nrm);
  const auto foundOrm = find_companion(KeyA, CompanionRole::Orm);
  const auto foundMat = find_companion(KeyA, CompanionRole::Mat);
  ASSERT_TRUE(foundNrm.has_value());
  ASSERT_TRUE(foundOrm.has_value());
  ASSERT_TRUE(foundMat.has_value());
  EXPECT_EQ(foundNrm->path, nrm);
  EXPECT_EQ(foundOrm->path, orm);
  EXPECT_EQ(foundMat->path, mat);
  EXPECT_FALSE(find_companion(KeyA, CompanionRole::Albedo).has_value());

  texture::unregister_replacements(group);
  EXPECT_FALSE(should_build_source_key());
  EXPECT_EQ(companion_roles(KeyA), 0u);
  EXPECT_EQ(companion_generation(), generation + 2);
}

TEST_F(CompanionScanTest, ResolvesTlutWildcardSet) {
  write_file("tex1_16x16_000000000000d001_$_9.nrm.dds");
  const auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.companions.size(), 1u);

  const texture::TextureSourceKey key{
      .textureHash = 0xd001, .tlutHash = 0x1234, .width = 16, .height = 16, .format = 9, .hasTlut = true};
  EXPECT_EQ(companion_roles(key), role_bit(CompanionRole::Nrm));
  const auto found = find_companion(key, CompanionRole::Nrm);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->id, group.companions[0].id);
  auto wildcard = key;
  wildcard.tlutHash = texture::kWildcardTlutHash;
  EXPECT_EQ(companion_roles(wildcard), role_bit(CompanionRole::Nrm));
  auto otherTexture = key;
  otherTexture.textureHash = 0xd002;
  EXPECT_EQ(companion_roles(otherTexture), 0u);
}

TEST_F(CompanionScanTest, ExactKeySetWinsOverTlutWildcard) {
  write_file("tex1_16x16_000000000000d001_$_9.nrm.dds");
  write_file("tex1_16x16_000000000000d001_000000000000d0f1_9.orm.dds");
  const auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.companions.size(), 2u);

  const texture::TextureSourceKey exact{
      .textureHash = 0xd001, .tlutHash = 0xd0f1, .width = 16, .height = 16, .format = 9, .hasTlut = true};
  EXPECT_EQ(companion_roles(exact), role_bit(CompanionRole::Orm));
  EXPECT_TRUE(find_companion(exact, CompanionRole::Orm).has_value());
  EXPECT_FALSE(find_companion(exact, CompanionRole::Nrm).has_value());

  auto other = exact;
  other.tlutHash = 0xd0f2;
  EXPECT_EQ(companion_roles(other), role_bit(CompanionRole::Nrm));
  EXPECT_FALSE(find_companion(other, CompanionRole::Orm).has_value());
}

TEST_F(CompanionScanTest, HigherPriorityWinsAndGroupRemovalRestores) {
  const auto high = write_file(std::filesystem::path{"A"} / stem_a(".nrm.dds"));
  const auto low = write_file(std::filesystem::path{"B"} / stem_a(".nrm.dds"));
  const auto later = write_file(std::filesystem::path{"C"} / stem_a(".nrm.dds"));

  const auto highGroup = texture::load_replacement_directory(m_root / "A", {.priority = 5});
  const auto lowGroup = texture::load_replacement_directory(m_root / "B", {.priority = 0});
  ASSERT_TRUE(find_companion(KeyA, CompanionRole::Nrm).has_value());
  EXPECT_EQ(find_companion(KeyA, CompanionRole::Nrm)->path, high);

  texture::unregister_replacements(highGroup);
  ASSERT_TRUE(find_companion(KeyA, CompanionRole::Nrm).has_value());
  EXPECT_EQ(find_companion(KeyA, CompanionRole::Nrm)->path, low);

  const auto laterGroup = texture::load_replacement_directory(m_root / "C", {.priority = 0});
  ASSERT_TRUE(find_companion(KeyA, CompanionRole::Nrm).has_value());
  EXPECT_EQ(find_companion(KeyA, CompanionRole::Nrm)->path, later);

  texture::unregister_replacements(laterGroup);
  ASSERT_TRUE(find_companion(KeyA, CompanionRole::Nrm).has_value());
  EXPECT_EQ(find_companion(KeyA, CompanionRole::Nrm)->path, low);

  texture::unregister_replacements(lowGroup);
  EXPECT_FALSE(find_companion(KeyA, CompanionRole::Nrm).has_value());
}

TEST_F(CompanionScanTest, BypassHidesCompanions) {
  write_file(stem_a(".albedo.dds"));
  const auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.companions.size(), 1u);

  set_bypass(true);
  EXPECT_EQ(companion_roles(KeyA), 0u);
  EXPECT_FALSE(find_companion(KeyA, CompanionRole::Albedo).has_value());
  set_bypass(false);
  EXPECT_EQ(companion_roles(KeyA), role_bit(CompanionRole::Albedo));
  EXPECT_TRUE(find_companion(KeyA, CompanionRole::Albedo).has_value());
}

TEST_F(CompanionScanTest, GroupUnregisterIgnoresPointerAndStaleIds) {
  write_file(stem_a(".nrm.dds"));
  write_file(std::filesystem::path{"A"} / "materials.json",
             R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "roughness": 0.5 } } })");
  auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.companions.size(), 1u);
  const auto stale = group;
  const int pointee = 0;
  group.companions.push_back({.id = group.companions[0].id, .key = texture::TexturePointerKey{.data = &pointee}});
  group.companions.push_back({.id = 0, .key = KeyA});

  const uint64_t generation = companion_generation();
  const uint64_t materialGeneration = texture::material_report().generation;
  texture::unregister_replacements(group);
  EXPECT_EQ(companion_generation(), generation + 1);
  EXPECT_EQ(companion_roles(KeyA), 0u);
  if (!group.manifests.empty()) {
    EXPECT_EQ(texture::material_report().generation, materialGeneration + 1);
    EXPECT_TRUE(texture::material_report().manifests.empty());
  }

  const uint64_t removedGeneration = companion_generation();
  const uint64_t removedMaterialGeneration = texture::material_report().generation;
  texture::unregister_replacements(group);
  EXPECT_EQ(companion_generation(), removedGeneration);
  EXPECT_EQ(texture::material_report().generation, removedMaterialGeneration);

  const auto fresh = texture::load_replacement_directory(m_root);
  ASSERT_EQ(fresh.companions.size(), 1u);
  EXPECT_NE(fresh.companions[0].id, stale.companions[0].id);
  const uint64_t freshGeneration = companion_generation();
  const uint64_t freshMaterialGeneration = texture::material_report().generation;
  texture::unregister_replacements(stale);
  EXPECT_EQ(companion_generation(), freshGeneration);
  EXPECT_EQ(texture::material_report().generation, freshMaterialGeneration);
  EXPECT_EQ(companion_roles(KeyA), role_bit(CompanionRole::Nrm));
  texture::unregister_replacements(fresh);
  EXPECT_EQ(companion_roles(KeyA), 0u);
}

TEST_F(CompanionScanTest, CompanionMipNameKeepsRoleLast) {
  EXPECT_EQ(companion_mip_filename("MATERIALMAPS/" + stem_a(".nrm.dds"), 3), "MATERIALMAPS/" + stem_a("_mip3.nrm.dds"));
  EXPECT_EQ(companion_mip_filename("C:\\pack.v2\\" + stem_a(".mat.dds"), 1), "C:\\pack.v2\\" + stem_a("_mip1.mat.dds"));

  const auto name = parse_companion_filename(companion_mip_filename(stem_a(".nrm.dds"), 3));
  ASSERT_TRUE(name.has_value());
  EXPECT_EQ(name->key, KeyA);
  EXPECT_EQ(name->role, CompanionRole::Nrm);
  EXPECT_EQ(name->mip, 3u);
  EXPECT_EQ(companion_role_name(name->role), "nrm");
  EXPECT_FALSE(parse_companion_filename(stem_a(".nrm_mip3.dds")).has_value());
  EXPECT_FALSE(parse_companion_filename(stem_a("_mip03.nrm.dds")).has_value());
  EXPECT_FALSE(parse_companion_filename(stem_a("_mip16.nrm.dds")).has_value());

  write_file(stem_a(".nrm.dds"));
  write_file(stem_a("_mip1.nrm.dds"));
  write_file(stem_a("_mip2.nrm.dds"));
  const auto group = texture::load_replacement_directory(m_root);
  ASSERT_EQ(group.companions.size(), 1u);
  ASSERT_TRUE(find_companion(KeyA, CompanionRole::Nrm).has_value());
  EXPECT_EQ(find_companion(KeyA, CompanionRole::Nrm)->path.filename(), std::filesystem::path{stem_a(".nrm.dds")});
}

TEST_F(CompanionScanTest, CompanionOnlyPackLeavesKeyGateAlone) {
  testing::set_workers_paused(false);
  write_dds(std::filesystem::path{"maps"} / stem_a(".nrm.dds"));
  write_dds(std::filesystem::path{"maps"} / stem_a(".orm.dds"));
  const bool gate = should_build_source_key();
  const auto companionsOnly = texture::load_replacement_directory(m_root / "maps");
  EXPECT_TRUE(companionsOnly.registrations.empty());
  ASSERT_EQ(companionsOnly.companions.size(), 2u);
  EXPECT_EQ(should_build_source_key(), gate);

  write_dds(std::filesystem::path{"full"} / stem_a(".dds"));
  write_dds(std::filesystem::path{"full"} / stem_a(".albedo.dds"));
  const auto full = texture::load_replacement_directory(m_root / "full");
  ASSERT_EQ(full.registrations.size(), 1u);
  ASSERT_EQ(full.companions.size(), 1u);
  EXPECT_TRUE(testing::wait_for_completions(full.registrations[0].id, 1, 2000));
  EXPECT_FALSE(testing::wait_for_completions(full.companions[0].id, 1, 200));
  EXPECT_FALSE(testing::wait_for_completions(companionsOnly.companions[0].id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(companionsOnly.companions[1].id, 1, 0));
}

TEST_F(CompanionScanTest, PackManifestsFollowGroup) {
  write_file(std::filesystem::path{"A"} / "materials.json",
             R"({ "version": 1, "materials": { "tex1_16x16_000000000000a001_14": { "roughness": 0.5 } } })");
  write_file(std::filesystem::path{"B"} / "materials.json",
             R"({ "version": 1, "materials": { "tex1_16x16_000000000000b001_14": { "roughness": 0.25 } } })");
  write_file(std::filesystem::path{"A"} / "x" / "materials.json", R"({ "version": 1 })");

  const auto group = texture::load_replacement_directory(m_root, {.priority = 7});
  if (group.manifests.empty()) {
    GTEST_SKIP() << "render module disabled";
  }
  ASSERT_EQ(group.manifests.size(), 2u);
  const auto first = texture::material_manifest_info(group.manifests[0]);
  const auto second = texture::material_manifest_info(group.manifests[1]);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->path, m_root / "B" / "materials.json");
  EXPECT_EQ(second->path, m_root / "A" / "materials.json");
  EXPECT_GT(second->sequence, first->sequence);

  const auto report = texture::material_report();
  ASSERT_EQ(report.manifests.size(), 2u);
  for (const auto& info : report.manifests) {
    EXPECT_EQ(info.layer, texture::MaterialLayer::Pack);
    EXPECT_EQ(info.priority, 7);
    EXPECT_EQ(info.status, texture::MaterialManifestStatus::Accepted);
  }
  EXPECT_EQ(report.materialCount, 2u);

  texture::unregister_replacements(group);
  const auto removed = texture::material_report();
  EXPECT_TRUE(removed.manifests.empty());
  EXPECT_EQ(removed.generation, report.generation + 1);
}

// load_replacement_directory builds the dump root from cachePath even with dumps off
class ScopedCachePath {
public:
  explicit ScopedCachePath(const std::filesystem::path& path) : m_path(path.string()), m_previous(g_config.cachePath) {
    EXPECT_FALSE(g_config.allowTextureDumps);
    g_config.cachePath = m_path.c_str();
  }
  ~ScopedCachePath() { g_config.cachePath = m_previous; }
  ScopedCachePath(const ScopedCachePath&) = delete;
  ScopedCachePath& operator=(const ScopedCachePath&) = delete;

private:
  std::string m_path;
  const char* m_previous;
};

std::vector<uint8_t> test_pixels(uint32_t width, uint32_t height, uint32_t seed) {
  std::vector<uint8_t> pixels(size_t{width} * height * 4);
  uint32_t state = seed * 2654435761u + 1u;
  for (auto& value : pixels) {
    state = state * 1664525u + 1013904223u;
    value = static_cast<uint8_t>(state >> 24);
  }
  return pixels;
}

std::vector<uint8_t> rgba8_dds(uint32_t width, uint32_t height, const std::vector<uint8_t>& pixels) {
  const auto encoded = dds::encode_rgba8_dds(width, height, pixels);
  return {encoded.data(), encoded.data() + encoded.size()};
}

std::vector<uint32_t> box_taps(uint32_t size, uint32_t outSize, uint32_t index) {
  if (size == 1) {
    return {0};
  }
  std::vector<uint32_t> taps{2 * index, 2 * index + 1};
  if (size % 2 == 1 && index == outSize - 1) {
    taps.push_back(2 * index + 2);
  }
  return taps;
}

std::vector<uint8_t> reference_box_tail(const std::vector<uint8_t>& level, uint32_t width, uint32_t height) {
  std::vector<uint32_t> current(level.begin(), level.end());
  for (auto& value : current) {
    value <<= 8;
  }
  std::vector<uint8_t> out;
  while (width > 1 || height > 1) {
    const uint32_t outWidth = std::max(width / 2, 1u);
    const uint32_t outHeight = std::max(height / 2, 1u);
    std::vector<uint32_t> next(size_t{outWidth} * outHeight * 4);
    for (uint32_t y = 0; y < outHeight; ++y) {
      const auto ys = box_taps(height, outHeight, y);
      for (uint32_t x = 0; x < outWidth; ++x) {
        const auto xs = box_taps(width, outWidth, x);
        const uint32_t count = static_cast<uint32_t>(xs.size() * ys.size());
        for (uint32_t c = 0; c < 4; ++c) {
          uint32_t sum = 0;
          for (const uint32_t sy : ys) {
            for (const uint32_t sx : xs) {
              sum += current[(size_t{sy} * width + sx) * 4 + c];
            }
          }
          next[(size_t{y} * outWidth + x) * 4 + c] = (sum + count / 2) / count;
        }
      }
    }
    for (const uint32_t value : next) {
      out.push_back(static_cast<uint8_t>(std::min((value + 128) >> 8, 255u)));
    }
    current = std::move(next);
    width = outWidth;
    height = outHeight;
  }
  return out;
}

std::vector<uint8_t> joined(std::initializer_list<const std::vector<uint8_t>*> parts) {
  std::vector<uint8_t> out;
  for (const auto* part : parts) {
    out.insert(out.end(), part->begin(), part->end());
  }
  return out;
}

bool holds_bytes(const ConvertedTexture& texture, const std::vector<uint8_t>& expected) {
  return texture.data.size() == expected.size() && std::equal(expected.begin(), expected.end(), texture.data.data());
}

TEST(MipChainTest, ClassifiesLikeContract) {
  static_assert(full_mip_count(1, 1) == 1);
  static_assert(full_mip_count(64, 64) == 7);
  static_assert(full_mip_count(64, 16) == 7);
  static_assert(full_mip_count(5, 3) == 3);
  static_assert(full_mip_count(2688, 2688) == 12);
  // The c001 fixture chains: 64, 32, 16, 16, 16 and the .mat chain 32, 16, 16, 16
  static_assert(classify_mip_level(64, 64, 1, 64, 64, 32, 32) == MipFit::Standard);
  static_assert(classify_mip_level(64, 64, 2, 32, 32, 16, 16) == MipFit::Standard);
  static_assert(classify_mip_level(64, 64, 3, 16, 16, 16, 16) == MipFit::Clamped);
  static_assert(classify_mip_level(64, 64, 4, 16, 16, 16, 16) == MipFit::Clamped);
  static_assert(classify_mip_level(32, 32, 2, 16, 16, 16, 16) == MipFit::Clamped);
  EXPECT_EQ(classify_mip_level(64, 64, 1, 64, 64, 64, 64), MipFit::Clamped);
  EXPECT_EQ(classify_mip_level(128, 64, 2, 64, 32, 64, 32), MipFit::Clamped);
  EXPECT_EQ(classify_mip_level(64, 64, 2, 32, 32, 24, 24), MipFit::Invalid);
  EXPECT_EQ(classify_mip_level(64, 64, 3, 16, 16, 4, 4), MipFit::Invalid);
  EXPECT_EQ(classify_mip_level(64, 64, 3, 16, 16, 12, 12), MipFit::Invalid);
  EXPECT_EQ(classify_mip_level(64, 32, 3, 8, 4, 8, 8), MipFit::Invalid);
  EXPECT_EQ(classify_mip_level(64, 64, 2, 32, 32, 32, 16), MipFit::Invalid);
  EXPECT_EQ(classify_mip_level(4, 4, 3, 1, 1, 1, 1), MipFit::Standard);
  EXPECT_EQ(classify_mip_level(64, 64, 40, 1, 1, 1, 1), MipFit::Standard);
  EXPECT_EQ(classify_mip_level(64, 64, 40, 2, 2, 2, 2), MipFit::Clamped);
  EXPECT_EQ(classify_mip_level(64, 64, 40, 2, 2, 2, 1), MipFit::Invalid);
}

class MipRepairTest : public ReplacementStreamingTest {
protected:
  void SetUp() override {
    ReplacementStreamingTest::SetUp();
    m_savedBc = webgpu::g_bcTexturesSupported;
    m_dir = std::filesystem::temp_directory_path() /
            ("aurora-mip-repair-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(m_dir);
  }

  void TearDown() override {
    ReplacementStreamingTest::TearDown();
    set_mip_repair(false);
    set_synchronous_loads(false);
    webgpu::g_bcTexturesSupported = m_savedBc;
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
  }

  struct Chain {
    std::filesystem::path path;
    std::vector<std::vector<uint8_t>> levels;
  };

  Chain write_chain(std::string_view stem, std::initializer_list<std::pair<uint32_t, uint32_t>> sizes,
                    uint32_t seed = 1) const {
    Chain chain;
    uint32_t level = 0;
    for (const auto& [width, height] : sizes) {
      auto pixels = test_pixels(width, height, seed + level);
      const auto name = level == 0 ? fmt::format("{}.dds", stem) : fmt::format("{}_mip{}.dds", stem, level);
      const auto path = m_dir / name;
      const auto bytes = rgba8_dds(width, height, pixels);
      std::ofstream file(path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      if (level == 0) {
        chain.path = path;
      }
      chain.levels.push_back(std::move(pixels));
      ++level;
    }
    return chain;
  }

  std::filesystem::path m_dir;
  bool m_savedBc = false;
};

TEST_F(MipRepairTest, RepairOffKeepsLegacyPrefix) {
  const auto chain = write_chain("tex1_16x16_000000000000e001_6", {{16, 16}, {8, 8}, {8, 8}, {8, 8}});
  ScopedWarningCapture capture;
  ChainReport report;
  const auto loaded = testing::load_replacement_file(chain.path, false, &report);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->format, wgpu::TextureFormat::RGBA8Unorm);
  EXPECT_EQ(loaded->mips, 2u);
  EXPECT_TRUE(holds_bytes(*loaded, joined({&chain.levels[0], &chain.levels[1]})));
  EXPECT_EQ(report.verdict, ChainVerdict::Clamped);
  EXPECT_EQ(report.format, wgpu::TextureFormat::RGBA8Unorm);
  EXPECT_EQ(report.keptLevels, 2u);
  EXPECT_EQ(report.fullLevels, 5u);
  EXPECT_EQ(report.repairable, mip_repair_built());
  EXPECT_FALSE(report.repaired);
  EXPECT_EQ(report.generatedLevels, 0u);
  EXPECT_EQ(capture.warnings(), 0u);
  EXPECT_EQ(capture.debugs("clamps at 8x8 from mip 2"), 1u);
}

TEST_F(MipRepairTest, Rgba8TailIsExactBoxFilter) {
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  const auto even = write_chain("tex1_24x12_000000000000e002_6", {{24, 12}, {12, 6}, {12, 6}});
  const auto odd = write_chain("tex1_10x6_000000000000e003_6", {{10, 6}, {5, 3}, {5, 3}}, 7);
  ScopedWarningCapture capture;
  for (const auto* chain : {&even, &odd}) {
    SCOPED_TRACE(chain->path.filename().string());
    ChainReport report;
    const auto loaded = testing::load_replacement_file(chain->path, true, &report);
    ASSERT_TRUE(loaded.has_value());
    const auto [width, height] = std::pair{loaded->width, loaded->height};
    EXPECT_EQ(loaded->mips, full_mip_count(width, height));
    const auto tail = reference_box_tail(chain->levels[1], std::max(width / 2, 1u), std::max(height / 2, 1u));
    EXPECT_TRUE(holds_bytes(*loaded, joined({&chain->levels[0], &chain->levels[1], &tail})));
    EXPECT_EQ(report.verdict, ChainVerdict::Clamped);
    EXPECT_TRUE(report.repairable);
    EXPECT_TRUE(report.repaired);
    EXPECT_EQ(report.keptLevels, 2u);
    EXPECT_EQ(report.generatedLevels, full_mip_count(width, height) - 2);
  }
  EXPECT_EQ(capture.warnings(), 0u);
  EXPECT_EQ(capture.debugs("rebuilt 3 mips for"), 1u);
  EXPECT_EQ(capture.debugs("rebuilt 2 mips for"), 1u);
}

TEST_F(MipRepairTest, InvalidChainRegeneratesFromLastStandardLevel) {
  testing::set_workers_paused(true);
  MapSource source;
  const auto base = test_pixels(16, 16, 1);
  const auto mip1 = test_pixels(8, 8, 2);
  source.files[virtual_name(0xe004)] = rgba8_dds(16, 16, base);
  source.files[virtual_mip_name(0xe004, 1)] = rgba8_dds(8, 8, mip1);
  source.files[virtual_mip_name(0xe004, 2)] = rgba8_dds(6, 6, test_pixels(6, 6, 3));
  source.files[virtual_mip_name(0xe004, 3)] = rgba8_dds(2, 2, test_pixels(2, 2, 4));
  const auto registration = texture::register_virtual_replacement(virtual_name(0xe004), source.callbacks());
  ASSERT_NE(registration.id, 0u);

  ScopedWarningCapture capture;
  ChainReport off;
  const auto legacy = testing::load_replacement_texture(registration, false, &off);
  ASSERT_TRUE(legacy.has_value());
  EXPECT_EQ(legacy->mips, 2u);
  EXPECT_TRUE(holds_bytes(*legacy, joined({&base, &mip1})));
  EXPECT_EQ(off.verdict, ChainVerdict::Invalid);
  EXPECT_EQ(capture.warnings(fmt::format("expected 4x4 for mip {}, got 6x6", virtual_mip_name(0xe004, 2))), 1u);

  if (mip_repair_built()) {
    ChainReport on;
    const auto repaired = testing::load_replacement_texture(registration, true, &on);
    ASSERT_TRUE(repaired.has_value());
    EXPECT_EQ(repaired->mips, 5u);
    const auto tail = reference_box_tail(mip1, 8, 8);
    EXPECT_TRUE(holds_bytes(*repaired, joined({&base, &mip1, &tail})));
    EXPECT_EQ(on.verdict, ChainVerdict::Invalid);
    EXPECT_TRUE(on.repaired);
    EXPECT_EQ(on.generatedLevels, 3u);
    EXPECT_EQ(capture.warnings(), 1u);
  }
  EXPECT_FALSE(source.was_read(virtual_mip_name(0xe004, 3)));
  texture::unregister_replacement(registration);
}

TEST_F(MipRepairTest, ArbChainKeepsPrefix) {
  const auto chain = write_chain("tex1_16x16_000000000000e005_6_arb", {{16, 16}, {8, 8}, {8, 8}});
  ScopedWarningCapture capture;
  for (const bool repair : {false, true}) {
    SCOPED_TRACE(repair);
    ChainReport report;
    const auto loaded = testing::load_replacement_file(chain.path, repair, &report);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->mips, 2u);
    EXPECT_TRUE(holds_bytes(*loaded, joined({&chain.levels[0], &chain.levels[1]})));
    EXPECT_EQ(report.verdict, ChainVerdict::Clamped);
    EXPECT_FALSE(report.repairable);
    EXPECT_FALSE(report.repaired);
  }
  EXPECT_EQ(capture.warnings(), 0u);
}

TEST_F(MipRepairTest, UntouchedChainsAreByteIdentical) {
  webgpu::g_bcTexturesSupported = true;
  const auto complete = write_chain("tex1_8x8_000000000000e006_6", {{8, 8}, {4, 4}, {2, 2}, {1, 1}});
  const auto shortChain = write_chain("tex1_16x16_000000000000e007_6", {{16, 16}, {8, 8}});
  const auto mipless = write_chain("tex1_16x16_000000000000e008_6", {{16, 16}});
  const auto internalPath = m_dir / "tex1_16x16_000000000000e009_6.dds";
  {
    const auto bytes = make_dds(28, Format::RGBA8Unorm, 16, 16, 5);
    std::ofstream file(internalPath, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  const auto bc7Path = m_dir / "tex1_16x16_000000000000e00a_14.dds";
  const auto bc7Mip = m_dir / "tex1_16x16_000000000000e00a_14_mip1.dds";
  for (const auto& [path, width] : {std::pair{bc7Path, 16u}, std::pair{bc7Mip, 8u}}) {
    const auto bytes = make_dds(98, Format::BC7RGBAUnorm, width, width, 1, static_cast<uint8_t>(width));
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }

  struct Case {
    std::filesystem::path path;
    ChainVerdict verdict;
    uint32_t mips;
  };
  const std::array cases{
      Case{complete.path, ChainVerdict::Complete, 4}, Case{shortChain.path, ChainVerdict::Short, 2},
      Case{mipless.path, ChainVerdict::Mipless, 1},   Case{internalPath, ChainVerdict::Internal, 5},
      Case{bc7Path, ChainVerdict::Short, 2},
  };
  ScopedWarningCapture capture;
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.path.filename().string());
    ChainReport offReport;
    ChainReport onReport;
    const auto off = testing::load_replacement_file(entry.path, false, &offReport);
    const auto on = testing::load_replacement_file(entry.path, true, &onReport);
    ASSERT_TRUE(off.has_value());
    ASSERT_TRUE(on.has_value());
    EXPECT_EQ(off->format, on->format);
    EXPECT_EQ(off->mips, entry.mips);
    EXPECT_EQ(on->mips, entry.mips);
    EXPECT_TRUE(holds_bytes(*on, std::vector<uint8_t>(off->data.data(), off->data.data() + off->data.size())));
    EXPECT_EQ(offReport.verdict, entry.verdict);
    EXPECT_EQ(onReport.verdict, entry.verdict);
    EXPECT_EQ(onReport.keptLevels, entry.mips);
    EXPECT_FALSE(onReport.repairable);
    EXPECT_FALSE(onReport.repaired);
  }
  EXPECT_TRUE(
      holds_bytes(*testing::load_replacement_file(complete.path, true),
                  joined({&complete.levels[0], &complete.levels[1], &complete.levels[2], &complete.levels[3]})));
  EXPECT_EQ(capture.warnings(), 0u);
}

TEST_F(MipRepairTest, BrokenSidecarStillWarns) {
  testing::set_workers_paused(true);
  MapSource source;
  source.files[virtual_name(0xe00b)] = make_dds(28, Format::RGBA8Unorm, 16, 16);
  source.files[virtual_mip_name(0xe00b, 1)] = make_dds(87, Format::BGRA8Unorm, 8, 8);
  source.files[virtual_name(0xe00c)] = make_dds(28, Format::RGBA8Unorm, 16, 16);
  source.files[virtual_mip_name(0xe00c, 1)] = {1, 2, 3, 4};
  source.files[virtual_name(0xe00d)] = make_dds(28, Format::RGBA8Unorm, 16, 16);
  source.files[virtual_mip_name(0xe00d, 1)] = make_dds(28, Format::RGBA8Unorm, 8, 8);
  source.files[virtual_mip_name(0xe00d, 2)] = {1, 2, 3, 4};
  source.files[virtual_mip_name(0xe00d, 3)] = make_dds(28, Format::RGBA8Unorm, 4, 4);
  struct Case {
    uint32_t hash;
    std::string warning;
    uint32_t mips;
  };
  const std::array cases{
      Case{0xe00b, fmt::format("mip {} has format", virtual_mip_name(0xe00b, 1)), 1},
      Case{0xe00c, fmt::format("could not load mip {}", virtual_mip_name(0xe00c, 1)), 1},
      Case{0xe00d, fmt::format("could not load mip {}", virtual_mip_name(0xe00d, 2)), 2},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.hash);
    const auto registration = texture::register_virtual_replacement(virtual_name(entry.hash), source.callbacks());
    ASSERT_NE(registration.id, 0u);
    for (const bool repair : {false, true}) {
      ScopedWarningCapture capture;
      ChainReport report;
      const auto loaded = testing::load_replacement_texture(registration, repair, &report);
      ASSERT_TRUE(loaded.has_value());
      EXPECT_EQ(loaded->mips, entry.mips);
      EXPECT_EQ(report.verdict, ChainVerdict::Broken);
      EXPECT_FALSE(report.repairable);
      EXPECT_FALSE(report.repaired);
      EXPECT_EQ(capture.warnings(), 1u);
      EXPECT_EQ(capture.warnings(entry.warning), 1u);
    }
    texture::unregister_replacement(registration);
  }
  EXPECT_FALSE(source.was_read(virtual_mip_name(0xe00d, 3)));
}

TEST_F(MipRepairTest, StaleCompletionIsRequeued) {
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  testing::set_workers_paused(true);
  MapSource source;
  source.files[virtual_name(0xe00e)] = rgba8_dds(128, 128, test_pixels(128, 128, 1));
  source.files[virtual_mip_name(0xe00e, 1)] = rgba8_dds(64, 64, test_pixels(64, 64, 2));
  source.files[virtual_mip_name(0xe00e, 2)] = rgba8_dds(64, 64, test_pixels(64, 64, 3));
  source.gatePath = virtual_name(0xe00e);
  const auto registration = texture::register_virtual_replacement(virtual_name(0xe00e), source.callbacks());
  ASSERT_NE(registration.id, 0u);
  GXTexObj_ obj{};
  const auto key = std::get<texture::TextureSourceKey>(registration.key);
  ASSERT_TRUE(find_source_replacement(obj, key).has_value());

  testing::set_workers_paused(false);
  ASSERT_TRUE(source.wait_for_gate());
  set_mip_repair(true);
  source.release_gate();
  // The full load and the thumbnail probe of the mipless base
  ASSERT_TRUE(testing::wait_for_completions(registration.id, 2, 2000));
  const auto stats = process_streaming();
  EXPECT_EQ(stats.publishes, 0u);
  EXPECT_EQ(stats.pendingLoads, 1u);
  EXPECT_FALSE(testing::resident_chain(registration.key).has_value());
  EXPECT_TRUE(testing::wait_for_completions(registration.id, 1, 2000));
}

TEST_F(MipRepairTest, ToggleRequeuesOnlyRepairableResidentChains) {
  RecordingSource source;
  testing::set_workers_paused(true);
  const auto png_name = [](uint32_t hash) { return fmt::format("tex1_16x16_{:016x}_6.png", hash); };
  const auto repairable =
      texture::register_virtual_replacement(png_name(0xe010), {.read = &RecordingSource::read, .userData = &source});
  const auto complete =
      texture::register_virtual_replacement(png_name(0xe011), {.read = &RecordingSource::read, .userData = &source});
  const auto shadowed =
      texture::register_virtual_replacement(png_name(0xe012), {.read = &RecordingSource::read, .userData = &source});
  const auto selected =
      texture::register_virtual_replacement(png_name(0xe012), {.read = &RecordingSource::read, .userData = &source});
  ASSERT_NE(repairable.id, 0u);
  ASSERT_NE(complete.id, 0u);
  ASSERT_NE(shadowed.id, 0u);
  ASSERT_NE(selected.id, 0u);
  const ChainReport clamped{.verdict = ChainVerdict::Clamped, .repairable = true};
  const ChainReport whole{.verdict = ChainVerdict::Complete};
  ASSERT_TRUE(testing::seed_resident_full(repairable, Format::RGBA8Unorm, 16, 16, 2, clamped));
  ASSERT_TRUE(testing::seed_resident_full(complete, Format::RGBA8Unorm, 16, 16, 5, whole));
  ASSERT_TRUE(testing::seed_resident_full(shadowed, Format::RGBA8Unorm, 16, 16, 2, clamped));

  set_mip_repair(true);
  set_mip_repair(true);
  testing::set_workers_paused(false);
  EXPECT_TRUE(testing::wait_for_completions(repairable.id, 1, 2000));
  EXPECT_FALSE(testing::wait_for_completions(complete.id, 1, 200));
  EXPECT_FALSE(testing::wait_for_completions(shadowed.id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(selected.id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(repairable.id, 2, 0));
  {
    std::lock_guard lock{source.mutex};
    EXPECT_EQ(source.paths, std::vector<std::string>{png_name(0xe010)});
  }
  const auto resident = testing::resident_chain(repairable.key);
  ASSERT_TRUE(resident.has_value());
  EXPECT_TRUE(resident->repairable);
  EXPECT_TRUE(testing::resident_chain(complete.key).has_value());
  EXPECT_TRUE(testing::resident_chain(shadowed.key).has_value());
}

TEST_F(MipRepairTest, SynchronousToggleErasesRepairableResidentChains) {
  RecordingSource source;
  set_synchronous_loads(true);
  const auto png_name = [](uint32_t hash) { return fmt::format("tex1_16x16_{:016x}_6.png", hash); };
  const auto repairable =
      texture::register_virtual_replacement(png_name(0xe013), {.read = &RecordingSource::read, .userData = &source});
  const auto complete =
      texture::register_virtual_replacement(png_name(0xe014), {.read = &RecordingSource::read, .userData = &source});
  ASSERT_TRUE(testing::seed_resident_full(repairable, Format::RGBA8Unorm, 16, 16, 2,
                                          {.verdict = ChainVerdict::Invalid, .repairable = true}));
  ASSERT_TRUE(
      testing::seed_resident_full(complete, Format::RGBA8Unorm, 16, 16, 5, {.verdict = ChainVerdict::Complete}));

  set_mip_repair(true);
  EXPECT_FALSE(testing::resident_chain(repairable.key).has_value());
  const auto kept = testing::resident_chain(complete.key);
  ASSERT_TRUE(kept.has_value());
  EXPECT_EQ(kept->verdict, ChainVerdict::Complete);
  EXPECT_FALSE(testing::wait_for_completions(repairable.id, 1, 200));
  std::lock_guard lock{source.mutex};
  EXPECT_TRUE(source.paths.empty());
}

TEST(MipTailTest, BoxTailMatchesReferenceFilter) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 5> Sizes{{{5, 3}, {7, 1}, {1, 9}, {13, 8}, {16, 16}}};
  for (const auto& [width, height] : Sizes) {
    SCOPED_TRACE(fmt::format("{}x{}", width, height));
    const auto level = test_pixels(width, height, width * 31 + height);
    const auto tail = box_mip_tail_rgba8(width, height, level);
    const auto expected = reference_box_tail(level, width, height);
    EXPECT_EQ(tail.size(), calc_texture_size(Format::RGBA8Unorm, width, height, full_mip_count(width, height)) -
                               calc_texture_size(Format::RGBA8Unorm, width, height, 1));
    EXPECT_TRUE(tail.size() == expected.size() && std::equal(expected.begin(), expected.end(), tail.data()));
  }
  EXPECT_TRUE(box_mip_tail_rgba8(1, 1, test_pixels(1, 1, 1)).empty());
  const auto pixels = test_pixels(4, 4, 2);
  EXPECT_TRUE(box_mip_tail_rgba8(4, 4, {pixels.data(), pixels.size() - 4}).empty());
  EXPECT_TRUE(box_mip_tail_rgba8(0, 4, {}).empty());
}

TEST(MipTailTest, ReportsCostAt1024) {
  constexpr uint32_t Size = 1024;
  const auto level = test_pixels(Size, Size, 5);
  int64_t best = INT64_MAX;
  for (int run = 0; run < 3; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const auto tail = box_mip_tail_rgba8(Size, Size, level);
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    best = std::min<int64_t>(best, micros);
    ASSERT_EQ(tail.size(), calc_texture_size(Format::RGBA8Unorm, Size, Size, full_mip_count(Size, Size)) -
                               calc_texture_size(Format::RGBA8Unorm, Size, Size, 1));
  }
  std::printf("box mip tail at %ux%u: %lld us (best of 3)\n", Size, Size, static_cast<long long>(best));
}

constexpr u32 HwRepeat = GX_REPEAT | GX_REPEAT << 2;
constexpr u32 HwLinearMin = 4u << 5;
constexpr u32 HwLinMipLinMin = 6u << 5;
constexpr u32 HwLinearMag = 1u << 4;
constexpr u32 HwAniso4 = 2u << 19;

TextureBind sampler_bind(u32 mode0, bool generatedMips, bool replacement = false) {
  TextureBind bind;
  bind.ref =
      std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, wgpu::Extent3D{16, 16, 1},
                                   Format::RGBA8Unorm, full_mip_count(16, 16), GX_TF_I8);
  bind.ref->generatedMips = generatedMips;
  bind.ref->isReplacement = replacement;
  bind.texObj.mode0 = mode0;
  return bind;
}

bool same_sampling(const wgpu::SamplerDescriptor& lhs, const wgpu::SamplerDescriptor& rhs) {
  return lhs.addressModeU == rhs.addressModeU && lhs.addressModeV == rhs.addressModeV &&
         lhs.addressModeW == rhs.addressModeW && lhs.magFilter == rhs.magFilter && lhs.minFilter == rhs.minFilter &&
         lhs.mipmapFilter == rhs.mipmapFilter && lhs.lodMinClamp == rhs.lodMinClamp &&
         lhs.lodMaxClamp == rhs.lodMaxClamp && lhs.compare == rhs.compare && lhs.maxAnisotropy == rhs.maxAnisotropy;
}

class GeneratedMipSamplerTest : public ::testing::Test {
protected:
  void SetUp() override { webgpu::g_graphicsConfig.textureAnisotropy = 16; }
  void TearDown() override { webgpu::g_graphicsConfig.textureAnisotropy = m_savedAnisotropy; }

private:
  decltype(webgpu::g_graphicsConfig.textureAnisotropy) m_savedAnisotropy = webgpu::g_graphicsConfig.textureAnisotropy;
};

TEST_F(GeneratedMipSamplerTest, LinearMiplessWidensOnlyWithGeneratedMips) {
  constexpr u32 Mode0 = HwRepeat | HwLinearMag | HwLinearMin | HwAniso4;
  const auto legacy = sampler_bind(Mode0, false).get_descriptor();
  EXPECT_EQ(legacy.lodMinClamp, 0.f);
  EXPECT_EQ(legacy.lodMaxClamp, 0.f);
  EXPECT_EQ(legacy.mipmapFilter, wgpu::MipmapFilterMode::Undefined);
  EXPECT_EQ(legacy.maxAnisotropy, 1u);

  const auto generated = sampler_bind(Mode0, true);
  const auto widened = generated.get_descriptor();
  EXPECT_EQ(widened.lodMinClamp, 0.f);
  EXPECT_EQ(widened.lodMaxClamp, 1000.f);
  EXPECT_EQ(widened.mipmapFilter, wgpu::MipmapFilterMode::Linear);
  EXPECT_EQ(widened.maxAnisotropy, 1u);
  EXPECT_EQ(widened.minFilter, legacy.minFilter);
  EXPECT_EQ(widened.magFilter, legacy.magFilter);
  EXPECT_TRUE(same_sampling(generated.get_descriptor(true), legacy));
}

TEST_F(GeneratedMipSamplerTest, OtherFiltersKeepLegacyDescriptor) {
  struct Case {
    u32 mode0;
    bool replacement;
  };
  constexpr std::array Cases{
      Case{HwRepeat | HwAniso4, false},
      Case{HwRepeat | HwLinearMag | HwLinMipLinMin | HwAniso4, false},
      Case{HwRepeat | HwLinearMag | HwLinearMin | HwAniso4, true},
  };
  for (const auto& entry : Cases) {
    SCOPED_TRACE(entry.mode0);
    const auto plain = sampler_bind(entry.mode0, false, entry.replacement).get_descriptor();
    const auto generated = sampler_bind(entry.mode0, true, entry.replacement).get_descriptor();
    EXPECT_TRUE(same_sampling(plain, generated));
  }
}

TEST_F(GeneratedMipSamplerTest, GeneratedMipsIgnoreLodBias) {
  constexpr u32 HalfBias = 16u << 9;
  EXPECT_EQ(sampler_bind(HwRepeat | HwLinearMin | HalfBias, true).sample_lod_bias(), 0.f);
  EXPECT_EQ(sampler_bind(HwRepeat | HwLinearMin | HalfBias, false).sample_lod_bias(), 0.5f);
  EXPECT_EQ(sampler_bind(HwRepeat | HalfBias, true).sample_lod_bias(), 0.5f);
}

class MipGenerationTest : public MipRepairTest {
protected:
  // After shutdown, so no resident chain re-queues a read from a source the test body already destroyed
  void TearDown() override {
    MipRepairTest::TearDown();
    set_mip_generation(false);
  }

  void write_base(const std::filesystem::path& path, uint32_t seed) const {
    std::filesystem::create_directories(path.parent_path());
    const auto bytes = rgba8_dds(16, 16, test_pixels(16, 16, seed));
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
};

std::vector<uint8_t> generated_tail(uint32_t width, uint32_t height, const std::vector<uint8_t>& level) {
  const auto tail = box_mip_tail_rgba8(width, height, level);
  return std::vector<uint8_t>(tail.data(), tail.data() + tail.size());
}

TEST_F(MipGenerationTest, MiplessChainGeneratesFromBase) {
  const auto chain = write_chain("tex1_16x8_000000000000f001_6", {{16, 8}});
  ScopedWarningCapture capture;
  ChainReport off;
  const auto legacy = testing::load_replacement_file(chain.path, false, &off);
  ASSERT_TRUE(legacy.has_value());
  EXPECT_EQ(legacy->mips, 1u);
  EXPECT_TRUE(holds_bytes(*legacy, chain.levels[0]));
  EXPECT_EQ(off.verdict, ChainVerdict::Mipless);
  EXPECT_EQ(off.generatable, mip_repair_built());
  EXPECT_FALSE(off.generated);
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }

  ChainReport on;
  const auto generated = testing::load_replacement_file(chain.path, false, &on, true);
  ASSERT_TRUE(generated.has_value());
  EXPECT_EQ(generated->mips, 5u);
  const auto tail = generated_tail(16, 8, chain.levels[0]);
  EXPECT_TRUE(holds_bytes(*generated, joined({&chain.levels[0], &tail})));
  EXPECT_EQ(on.verdict, ChainVerdict::Mipless);
  EXPECT_TRUE(on.generatable);
  EXPECT_TRUE(on.generated);
  EXPECT_FALSE(on.repaired);
  EXPECT_EQ(on.generatedLevels, 4u);
  EXPECT_EQ(capture.warnings(), 0u);
  EXPECT_EQ(capture.debugs("generated 4 mips for"), 1u);
}

TEST_F(MipGenerationTest, ShortChainGeneratesFromLastLevel) {
  const auto chain = write_chain("tex1_16x16_000000000000f002_6", {{16, 16}, {8, 8}});
  ChainReport off;
  const auto legacy = testing::load_replacement_file(chain.path, false, &off);
  ASSERT_TRUE(legacy.has_value());
  EXPECT_EQ(legacy->mips, 2u);
  EXPECT_EQ(off.verdict, ChainVerdict::Short);
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }

  ChainReport on;
  const auto generated = testing::load_replacement_file(chain.path, false, &on, true);
  ASSERT_TRUE(generated.has_value());
  EXPECT_EQ(generated->mips, 5u);
  const auto tail = generated_tail(8, 8, chain.levels[1]);
  EXPECT_TRUE(holds_bytes(*generated, joined({&chain.levels[0], &chain.levels[1], &tail})));
  EXPECT_EQ(on.verdict, ChainVerdict::Short);
  EXPECT_EQ(on.keptLevels, 2u);
  EXPECT_EQ(on.generatedLevels, 3u);
  EXPECT_TRUE(on.generated);
  EXPECT_FALSE(on.repaired);
}

TEST_F(MipGenerationTest, RepairAndGenerationGatesAreIndependent) {
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  const auto clamped = write_chain("tex1_16x16_000000000000f003_6", {{16, 16}, {8, 8}, {8, 8}});
  const auto mipless = write_chain("tex1_16x16_000000000000f004_6", {{16, 16}});
  struct Case {
    const Chain* chain;
    bool repair;
    bool generate;
    uint32_t mips;
  };
  const std::array cases{
      Case{&clamped, false, true, 2},
      Case{&clamped, true, false, 5},
      Case{&mipless, true, false, 1},
      Case{&mipless, false, true, 5},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(
        fmt::format("{} repair {} generate {}", entry.chain->path.filename().string(), entry.repair, entry.generate));
    ChainReport report;
    const auto loaded = testing::load_replacement_file(entry.chain->path, entry.repair, &report, entry.generate);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->mips, entry.mips);
    EXPECT_EQ(report.repaired, entry.chain == &clamped && entry.repair);
    EXPECT_EQ(report.generated, entry.chain == &mipless && entry.generate);
  }
}

TEST_F(MipGenerationTest, ExcludedDirectoriesNeverGenerate) {
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  testing::set_workers_paused(true);
  constexpr std::array<std::string_view, 1> Excluded{"ui"};
  const auto root = m_dir / "pack";
  write_base(root / "UI" / "tex1_16x16_000000000000f005_6.dds", 1);
  write_base(root / "env" / "tex1_16x16_000000000000f006_6.dds", 2);
  ScopedCachePath cachePath{m_dir / "cache"};
  const auto group = texture::load_replacement_directory(root, {.noGeneratedMipDirs = Excluded});
  ASSERT_EQ(group.registrations.size(), 2u);
  for (const auto& registration : group.registrations) {
    const bool ui = std::get<texture::TextureSourceKey>(registration.key).textureHash == 0xf005;
    SCOPED_TRACE(ui);
    ChainReport report;
    const auto loaded = testing::load_replacement_texture(registration, false, &report, true);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(report.verdict, ChainVerdict::Mipless);
    EXPECT_EQ(report.generatable, !ui);
    EXPECT_EQ(report.generated, !ui);
    EXPECT_EQ(loaded->mips, ui ? 1u : 5u);
  }
  texture::unregister_replacements(group);

  MapSource source;
  const std::string uiPath = "mod/UI/tex1_16x16_000000000000f007_6.dds";
  const std::string envPath = "mod/env/tex1_16x16_000000000000f008_6.dds";
  source.files[uiPath] = rgba8_dds(16, 16, test_pixels(16, 16, 3));
  source.files[envPath] = rgba8_dds(16, 16, test_pixels(16, 16, 4));
  for (const auto& [path, ui] : {std::pair{uiPath, true}, std::pair{envPath, false}}) {
    SCOPED_TRACE(path);
    const auto registration =
        texture::register_virtual_replacement(path, source.callbacks(), {.noGeneratedMipDirs = Excluded});
    ASSERT_NE(registration.id, 0u);
    ChainReport report;
    const auto loaded = testing::load_replacement_texture(registration, false, &report, true);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(report.generatable, !ui);
    EXPECT_EQ(loaded->mips, ui ? 1u : 5u);
    texture::unregister_replacement(registration);
  }
}

TEST_F(MipGenerationTest, ArbBaseStaysMipless) {
  const auto chain = write_chain("tex1_16x16_000000000000f009_6_arb", {{16, 16}});
  ChainReport report;
  const auto loaded = testing::load_replacement_file(chain.path, true, &report, true);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->mips, 1u);
  EXPECT_TRUE(holds_bytes(*loaded, chain.levels[0]));
  EXPECT_EQ(report.verdict, ChainVerdict::Mipless);
  EXPECT_FALSE(report.generatable);
  EXPECT_FALSE(report.generated);
}

TEST_F(MipGenerationTest, StaleCompletionIsRequeued) {
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  testing::set_workers_paused(true);
  MapSource source;
  source.files[virtual_name(0xf00a)] = rgba8_dds(128, 128, test_pixels(128, 128, 1));
  source.gatePath = virtual_name(0xf00a);
  const auto registration = texture::register_virtual_replacement(virtual_name(0xf00a), source.callbacks());
  ASSERT_NE(registration.id, 0u);
  GXTexObj_ obj{};
  const auto key = std::get<texture::TextureSourceKey>(registration.key);
  ASSERT_TRUE(find_source_replacement(obj, key).has_value());

  testing::set_workers_paused(false);
  ASSERT_TRUE(source.wait_for_gate());
  set_mip_generation(true);
  source.release_gate();
  // The full load and the thumbnail probe of the mipless base
  ASSERT_TRUE(testing::wait_for_completions(registration.id, 2, 2000));
  const auto stats = process_streaming();
  EXPECT_EQ(stats.publishes, 0u);
  EXPECT_EQ(stats.pendingLoads, 1u);
  EXPECT_FALSE(testing::resident_chain(registration.key).has_value());
  EXPECT_TRUE(testing::wait_for_completions(registration.id, 1, 2000));
}

TEST_F(MipGenerationTest, ToggleRequeuesOnlyGeneratableResidentChains) {
  RecordingSource source;
  testing::set_workers_paused(true);
  const auto png_name = [](uint32_t hash) { return fmt::format("tex1_16x16_{:016x}_6.png", hash); };
  const auto register_png = [&](uint32_t hash) {
    return texture::register_virtual_replacement(png_name(hash), {.read = &RecordingSource::read, .userData = &source});
  };
  const auto generatable = register_png(0xf010);
  const auto repairable = register_png(0xf011);
  const auto complete = register_png(0xf012);
  const auto shadowed = register_png(0xf013);
  const auto selected = register_png(0xf013);
  const ChainReport mipless{.verdict = ChainVerdict::Mipless, .generatable = true};
  ASSERT_TRUE(testing::seed_resident_full(generatable, Format::RGBA8Unorm, 16, 16, 1, mipless));
  ASSERT_TRUE(testing::seed_resident_full(repairable, Format::RGBA8Unorm, 16, 16, 2,
                                          {.verdict = ChainVerdict::Clamped, .repairable = true}));
  ASSERT_TRUE(
      testing::seed_resident_full(complete, Format::RGBA8Unorm, 16, 16, 5, {.verdict = ChainVerdict::Complete}));
  ASSERT_TRUE(testing::seed_resident_full(shadowed, Format::RGBA8Unorm, 16, 16, 1, mipless));

  set_mip_generation(true);
  set_mip_generation(true);
  testing::set_workers_paused(false);
  EXPECT_TRUE(testing::wait_for_completions(generatable.id, 1, 2000));
  EXPECT_FALSE(testing::wait_for_completions(repairable.id, 1, 200));
  EXPECT_FALSE(testing::wait_for_completions(complete.id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(shadowed.id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(selected.id, 1, 0));
  EXPECT_FALSE(testing::wait_for_completions(generatable.id, 2, 0));
  {
    std::lock_guard lock{source.mutex};
    EXPECT_EQ(source.paths, std::vector<std::string>{png_name(0xf010)});
  }
  const auto resident = testing::resident_chain(generatable.key);
  ASSERT_TRUE(resident.has_value());
  EXPECT_TRUE(resident->generatable);
  EXPECT_TRUE(testing::resident_chain(repairable.key).has_value());
  EXPECT_TRUE(testing::resident_chain(complete.key).has_value());
}

TEST_F(MipGenerationTest, SynchronousToggleErasesGeneratableResidentChains) {
  RecordingSource source;
  set_synchronous_loads(true);
  const auto png_name = [](uint32_t hash) { return fmt::format("tex1_16x16_{:016x}_6.png", hash); };
  const auto generatable =
      texture::register_virtual_replacement(png_name(0xf014), {.read = &RecordingSource::read, .userData = &source});
  const auto complete =
      texture::register_virtual_replacement(png_name(0xf015), {.read = &RecordingSource::read, .userData = &source});
  ASSERT_TRUE(testing::seed_resident_full(generatable, Format::RGBA8Unorm, 16, 16, 2,
                                          {.verdict = ChainVerdict::Short, .generatable = true}));
  ASSERT_TRUE(
      testing::seed_resident_full(complete, Format::RGBA8Unorm, 16, 16, 5, {.verdict = ChainVerdict::Complete}));

  set_mip_generation(true);
  EXPECT_FALSE(testing::resident_chain(generatable.key).has_value());
  const auto kept = testing::resident_chain(complete.key);
  ASSERT_TRUE(kept.has_value());
  EXPECT_EQ(kept->verdict, ChainVerdict::Complete);
  EXPECT_FALSE(testing::wait_for_completions(generatable.id, 1, 200));
  std::lock_guard lock{source.mutex};
  EXPECT_TRUE(source.paths.empty());
}

class ReplacementPackSweep : public MipRepairTest {};

double srgb_to_linear(double value) {
  return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
}

double linear_to_srgb(double value) {
  return value <= 0.0031308 ? value * 12.92 : 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
}

std::vector<uint8_t> linear_box_level(const std::vector<uint8_t>& level, uint32_t width, uint32_t height) {
  const uint32_t outWidth = std::max(width / 2, 1u);
  const uint32_t outHeight = std::max(height / 2, 1u);
  std::vector<uint8_t> out(size_t{outWidth} * outHeight * 4);
  for (uint32_t y = 0; y < outHeight; ++y) {
    const auto ys = box_taps(height, outHeight, y);
    for (uint32_t x = 0; x < outWidth; ++x) {
      const auto xs = box_taps(width, outWidth, x);
      for (uint32_t c = 0; c < 4; ++c) {
        double sum = 0;
        for (const uint32_t sy : ys) {
          for (const uint32_t sx : xs) {
            const double value = level[(size_t{sy} * width + sx) * 4 + c] / 255.0;
            sum += c == 3 ? value : srgb_to_linear(value);
          }
        }
        const double mean = sum / static_cast<double>(xs.size() * ys.size());
        out[(size_t{y} * outWidth + x) * 4 + c] =
            static_cast<uint8_t>(std::lround(std::clamp(c == 3 ? mean : linear_to_srgb(mean), 0.0, 1.0) * 255.0));
      }
    }
  }
  return out;
}

double mean_abs_difference(const uint8_t* lhs, const uint8_t* rhs, size_t count, bool colorOnly) {
  double sum = 0;
  size_t samples = 0;
  for (size_t i = 0; i < count; ++i) {
    if (colorOnly && i % 4 == 3) {
      continue;
    }
    sum += std::abs(static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]));
    ++samples;
  }
  return samples == 0 ? 0.0 : sum / static_cast<double>(samples);
}

std::vector<uint8_t> decoded(const ConvertedTexture& chain, uint32_t level) {
  const uint32_t width = std::max(chain.width >> level, 1u);
  const uint32_t height = std::max(chain.height >> level, 1u);
  const uint64_t offset = calc_texture_size(chain.format, chain.width, chain.height, level);
  const uint64_t size = calc_texture_size(chain.format, width, height, 1);
  if (offset + size > chain.data.size()) {
    return {};
  }
  const auto rgba = testing::decode_replacement_level(chain.format, width, height, {chain.data.data() + offset, size});
  return rgba ? std::vector<uint8_t>(rgba->data(), rgba->data() + rgba->size()) : std::vector<uint8_t>{};
}

std::string summarize(std::vector<double> values, double scale, std::string_view unit) {
  if (values.empty()) {
    return "n 0";
  }
  std::sort(values.begin(), values.end());
  double sum = 0;
  for (const double value : values) {
    sum += value;
  }
  const auto at = [&](size_t percent) { return values[std::min(values.size() - 1, values.size() * percent / 100)]; };
  return fmt::format("n {}, mean {:.3f}{}, p50 {:.3f}{}, p95 {:.3f}{}, max {:.3f}{}", values.size(),
                     sum / static_cast<double>(values.size()) * scale, unit, at(50) * scale, unit, at(95) * scale, unit,
                     values.back() * scale, unit);
}

std::map<std::string, std::filesystem::path> base_paths_by_name(const std::filesystem::path& root) {
  std::vector<std::pair<std::vector<std::string>, std::filesystem::path>> candidates;
  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file()) {
      continue;
    }
    std::vector<std::string> components;
    for (const auto& component : it->path().lexically_relative(root)) {
      components.push_back(component.string());
    }
    candidates.emplace_back(std::move(components), it->path());
  }
  const auto lower = [](std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
  };
  std::sort(candidates.begin(), candidates.end(), [&](const auto& lhs, const auto& rhs) {
    const size_t count = std::min(lhs.first.size(), rhs.first.size());
    for (size_t i = 0; i < count; ++i) {
      const auto a = lower(lhs.first[i]);
      const auto b = lower(rhs.first[i]);
      if (a != b) {
        return a < b;
      }
      if (lhs.first[i] != rhs.first[i]) {
        return lhs.first[i] < rhs.first[i];
      }
    }
    return lhs.first.size() < rhs.first.size();
  });
  std::map<std::string, std::filesystem::path> out;
  for (const auto& [components, path] : candidates) {
    if (const auto key = texture::parse_replacement_filename(path.filename().string()); key.has_value()) {
      out.try_emplace(build_texture_replacement_name(*key), path);
    }
  }
  return out;
}

TEST_F(ReplacementPackSweep, HasNoMipWarnings) {
  const char* pack = std::getenv("AURORA_TEST_REPLACEMENT_PACK");
  if (pack == nullptr || *pack == '\0') {
    GTEST_SKIP() << "AURORA_TEST_REPLACEMENT_PACK is not set";
  }
  if (!mip_repair_built()) {
    GTEST_SKIP() << "render module disabled";
  }
  const std::filesystem::path root{pack};
  ScopedCachePath cachePath{m_dir / "cache"};
  BcCapsGuard caps{true};
  testing::set_workers_paused(true);
  ScopedWarningCapture capture;
  constexpr std::array<std::string_view, 1> NoGeneratedMipDirs{"UI"};
  const auto group = texture::load_replacement_directory(root, {.noGeneratedMipDirs = NoGeneratedMipDirs});
  ASSERT_FALSE(group.registrations.empty());
  const auto paths = base_paths_by_name(root);
  const auto in_ui_dir = [&](const std::string& name) {
    const auto path = paths.find(name);
    if (path == paths.end()) {
      return false;
    }
    const auto relative = path->second.parent_path().lexically_relative(root);
    return std::any_of(relative.begin(), relative.end(), [](const std::filesystem::path& component) {
      auto text = component.string();
      std::transform(text.begin(), text.end(), text.begin(),
                     [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      return text == "ui";
    });
  };

  std::map<ChainVerdict, size_t> verdicts;
  std::vector<double> repairMicros;
  std::vector<std::pair<uint32_t, std::string>> slowest;
  std::vector<double> resampleDelta;
  std::vector<double> linearDelta;
  std::vector<double> encodeDelta;
  size_t failures = 0;
  size_t uiCandidates = 0;
  size_t uiGeneratable = 0;
  std::vector<std::string> generatedNames;
  for (const auto& registration : group.registrations) {
    ChainReport report;
    const auto loaded = testing::load_replacement_texture(registration, true, &report, true);
    ++verdicts[report.verdict];
    if (!loaded.has_value()) {
      ++failures;
      continue;
    }
    const auto& key = std::get<texture::TextureSourceKey>(registration.key);
    const auto name = build_texture_replacement_name(key);
    const bool ui = in_ui_dir(name);
    if (ui && (report.verdict == ChainVerdict::Mipless || report.verdict == ChainVerdict::Short)) {
      ++uiCandidates;
    }
    if (report.generatable) {
      EXPECT_TRUE(report.generated) << name;
      EXPECT_EQ(loaded->mips, full_mip_count(loaded->width, loaded->height)) << name;
      uiGeneratable += ui ? 1 : 0;
      const auto path = paths.find(name);
      generatedNames.push_back(path != paths.end() ? path->second.lexically_relative(root).generic_string() : name);
    }
    if (report.verdict != ChainVerdict::Clamped && report.verdict != ChainVerdict::Invalid) {
      continue;
    }
    EXPECT_TRUE(report.repaired) << name;
    EXPECT_EQ(loaded->mips, full_mip_count(loaded->width, loaded->height)) << name;
    EXPECT_EQ(report.keptLevels + report.generatedLevels, report.fullLevels) << name;
    repairMicros.push_back(report.repairMicros);

    const uint32_t level = report.keptLevels;
    const uint32_t lastWidth = std::max(loaded->width >> (level - 1), 1u);
    const uint32_t lastHeight = std::max(loaded->height >> (level - 1), 1u);
    slowest.emplace_back(report.repairMicros, fmt::format("{} from {}x{}", name, lastWidth, lastHeight));
    const auto last = decoded(*loaded, level - 1);
    const auto generated = decoded(*loaded, level);
    if (last.empty() || generated.empty()) {
      continue;
    }
    const auto stored = reference_box_tail(last, lastWidth, lastHeight);
    const auto linear = linear_box_level(last, lastWidth, lastHeight);
    linearDelta.push_back(mean_abs_difference(stored.data(), linear.data(), linear.size(), true));
    encodeDelta.push_back(mean_abs_difference(stored.data(), generated.data(), generated.size(), false));

    const auto path = paths.find(name);
    if (report.verdict != ChainVerdict::Clamped || path == paths.end()) {
      continue;
    }
    const auto& base = path->second;
    const auto clampedPath =
        base.parent_path() / fmt::format("{}_mip{}{}", base.stem().string(), level, base.extension().string());
    const auto authored = dds::load_dds_file(clampedPath);
    if (!authored || authored->format != loaded->format || authored->width != lastWidth ||
        authored->height != lastHeight) {
      continue;
    }
    const auto authoredRgba = testing::decode_replacement_level(
        authored->format, lastWidth, lastHeight,
        {authored->data.data(), calc_texture_size(authored->format, lastWidth, lastHeight, 1)});
    if (!authoredRgba) {
      continue;
    }
    const uint32_t width = std::max(lastWidth / 2, 1u);
    const uint32_t height = std::max(lastHeight / 2, 1u);
    const uint32_t factorX = lastWidth / width;
    const uint32_t factorY = lastHeight / height;
    std::vector<uint8_t> resampled(size_t{width} * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        for (uint32_t c = 0; c < 4; ++c) {
          uint32_t sum = 0;
          for (uint32_t sy = 0; sy < factorY; ++sy) {
            for (uint32_t sx = 0; sx < factorX; ++sx) {
              sum += authoredRgba->data()[((size_t{y} * factorY + sy) * lastWidth + x * factorX + sx) * 4 + c];
            }
          }
          resampled[(size_t{y} * width + x) * 4 + c] =
              static_cast<uint8_t>((sum + factorX * factorY / 2) / (factorX * factorY));
        }
      }
    }
    resampleDelta.push_back(mean_abs_difference(resampled.data(), generated.data(), resampled.size(), false));
  }

  const auto warnings = capture.lines(LOG_WARNING, "texture_replacement:");
  for (size_t i = 0; i < std::min<size_t>(warnings.size(), 20); ++i) {
    ADD_FAILURE() << warnings[i];
  }
  EXPECT_EQ(warnings.size(), 0u);
  EXPECT_EQ(failures, 0u);
  EXPECT_EQ(uiGeneratable, 0u);
  constexpr std::array<std::string_view, 8> VerdictNames{"Unprobed", "Mipless", "Internal", "Complete",
                                                         "Short",    "Clamped", "Invalid",  "Broken"};
  std::string histogram;
  for (const auto& [verdict, count] : verdicts) {
    histogram += fmt::format(" {} {}", VerdictNames[static_cast<size_t>(verdict)], count);
  }
  std::printf("registrations %zu, failures %zu, verdicts:%s\n", group.registrations.size(), failures,
              histogram.c_str());
  std::printf("repair time: %s\n", summarize(repairMicros, 0.001, " ms").c_str());
  std::printf("generated %zu, UI generatable %zu (UI mipless or short %zu)\n", generatedNames.size(), uiGeneratable,
              uiCandidates);
  for (size_t i = 0; i < std::min<size_t>(generatedNames.size(), 5); ++i) {
    std::printf("  generated %s\n", generatedNames[i].c_str());
  }
  std::sort(slowest.begin(), slowest.end(), std::greater<>{});
  for (size_t i = 0; i < std::min<size_t>(slowest.size(), 3); ++i) {
    std::printf("  %.3f ms %s\n", slowest[i].first * 0.001, slowest[i].second.c_str());
  }
  std::printf("resample vs regenerate, first generated level, mean |d| per chain (RGBA): %s\n",
              summarize(resampleDelta, 1.0, "").c_str());
  std::printf("stored vs linear box, first generated level, mean |d| per chain (RGB): %s\n",
              summarize(linearDelta, 1.0, "").c_str());
  std::printf("encoded vs box reference, first generated level, mean |d| per chain (RGBA): %s\n",
              summarize(encodeDelta, 1.0, "").c_str());
  texture::unregister_replacements(group);
}
} // namespace
} // namespace aurora::gfx::texture_replacement

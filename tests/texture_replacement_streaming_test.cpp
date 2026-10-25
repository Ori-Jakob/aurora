#include "gfx/dds_io.hpp"
#include "gfx/texture_replacement.hpp"

#include <aurora/texture.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
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
} // namespace
} // namespace aurora::gfx::texture_replacement

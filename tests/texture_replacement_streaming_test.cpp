#include "gfx/dds_io.hpp"
#include "gfx/texture_replacement.hpp"
#include "internal.hpp"

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
} // namespace
} // namespace aurora::gfx::texture_replacement

#include "render/material/registry.hpp"
#include "render/material/user_layer.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

namespace aurora::render::material {
namespace {
using texture::MaterialLayer;
using texture::MaterialManifestStatus;

constexpr std::string_view KeyText = "tex1_16x16_000000000000b001_14";
constexpr std::string_view OtherText = "tex1_16x16_000000000000b004_14";
constexpr std::string_view PaletteText = "tex1_16x16_000000000000b002_000000000000c003_9";
constexpr std::string_view Empty = R"({"version":1,"materials":{}})";
constexpr std::array<std::string_view, 1> StochasticFlag{"stochastic"};

texture::TextureSourceKey key(std::string_view text) { return *texture::parse_material_key(text); }

std::string read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void write(const std::filesystem::path& path, std::string_view text) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

MaterialRecord record_for(const texture::TextureSourceKey& lookup) {
  latch();
  const auto snapshot = latched_snapshot();
  return snapshot->record(snapshot->find(lookup));
}

std::string class_of(const texture::TextureSourceKey& lookup) {
  latch();
  const auto snapshot = latched_snapshot();
  const uint16_t id = snapshot->find(lookup);
  const uint16_t classId = snapshot->record(id).classId;
  return id != 0 && classId < snapshot->classes.size() ? snapshot->classes[classId] : std::string{};
}

size_t session_registrations() {
  size_t count = 0;
  for (const texture::MaterialManifestInfo& info : texture::material_report().manifests) {
    count += info.label == UserLayer::SessionLabel ? 1 : 0;
  }
  return count;
}

uint64_t g_hookRegistration = 0;
bool g_hookArmed = false;

// A reload started inside another one supersedes it, which is how a racing writer looks to apply()
void supersede_hook(uint64_t) {
  if (!std::exchange(g_hookArmed, false)) {
    return;
  }
  texture::update_material_manifest({g_hookRegistration, MaterialManifestStatus::Accepted},
                                    R"({"version":1,"materials":{"bad":{}}})");
}

class UserLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
    texture::clear_material_manifests();
    const std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    m_dir = std::filesystem::temp_directory_path() / fmt::format("aurora_user_layer_{}", name);
    std::filesystem::remove_all(m_dir);
    std::filesystem::create_directories(m_dir);
    m_path = m_dir / "texture_materials.user.json";
    m_backup = m_path;
    m_backup += ".bak";
  }

  void TearDown() override {
    testing::set_check_hook(nullptr);
    texture::clear_material_manifests();
    latch();
    std::error_code error;
    std::filesystem::remove_all(m_dir, error);
  }

  texture::MaterialManifestRegistration register_user() {
    return texture::load_material_manifest(m_path, {.layer = MaterialLayer::User, .writable = true});
  }

  std::filesystem::path m_dir;
  std::filesystem::path m_path;
  std::filesystem::path m_backup;
};

TEST_F(UserLayerTest, RoundTrip) {
  const std::string base = fmt::format(
      R"({{"$schema":"materials.schema.json","version":1,"materials":{{"{}":{{"_source":"hand"}}}}}})", OtherText);
  write(m_path, base);
  ASSERT_EQ(register_user().status, MaterialManifestStatus::Accepted);
  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.set_stochastic(key(KeyText), "scale", 2.0));
  EXPECT_TRUE(layer.set_flags(key(KeyText), StochasticFlag));
  EXPECT_FALSE(layer.set_stochastic(key(KeyText), "size", 2.0));
  EXPECT_TRUE(layer.dirty());
  const UserApplyResult result = layer.apply();
  ASSERT_TRUE(result.applied) << result.message;
  EXPECT_FALSE(layer.dirty());

  const std::string written = read(m_path);
  EXPECT_EQ(written, layer.text());
  EXPECT_LT(written.find("$schema"), written.find("\"version\""));
  EXPECT_LT(written.find(OtherText), written.find(KeyText));
  EXPECT_NE(written.find("\"_source\": \"hand\""), std::string::npos);
  EXPECT_EQ(read(m_backup), base);

  const MaterialRecord record = record_for(key(KeyText));
  EXPECT_NE(record.id, 0u);
  EXPECT_FLOAT_EQ(record.scale, 2.f);
  EXPECT_NE(record.summary & Stochastic, 0u);

  UserLayer again;
  ASSERT_EQ(again.open(), UserLayerState::Loaded);
  const UserEntry entry = again.read_entry(key(KeyText));
  ASSERT_TRUE(entry.exists);
  ASSERT_TRUE(entry.stochastic[0].has_value());
  EXPECT_DOUBLE_EQ(*entry.stochastic[0], 2.0);
  EXPECT_EQ(entry.flags, std::vector<std::string>{"stochastic"});
}

TEST_F(UserLayerTest, ResetRestoresInherited) {
  ASSERT_EQ(texture::register_material_manifest(
                fmt::format(R"({{"version":1,"materials":{{"{}":{{"stochastic":{{"scale":4}}}}}}}})", KeyText),
                {.layer = MaterialLayer::Pack})
                .status,
            MaterialManifestStatus::Accepted);
  write(m_path, fmt::format(R"({{"version":1,"materials":{{"{}":{{"stochastic":{{"scale":2}}}}}}}})", KeyText));
  ASSERT_EQ(register_user().status, MaterialManifestStatus::Accepted);
  EXPECT_FLOAT_EQ(record_for(key(KeyText)).scale, 2.f);

  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.reset_field(key(KeyText), "stochastic/scale"));
  EXPECT_FALSE(layer.has_entry(key(KeyText)));
  ASSERT_TRUE(layer.apply().applied);
  EXPECT_FLOAT_EQ(record_for(key(KeyText)).scale, 4.f);
  EXPECT_EQ(read(m_path).find(KeyText), std::string::npos);
}

TEST_F(UserLayerTest, InvalidOrChangedFileIsNeverOverwritten) {
  write(m_path, "{not json");
  const texture::MaterialManifestRegistration registration = register_user();
  UserLayer layer;
  EXPECT_EQ(layer.open(), UserLayerState::Invalid);
  EXPECT_FALSE(layer.editable());
  EXPECT_FALSE(layer.set_class(key(KeyText), "ground"));
  EXPECT_FALSE(layer.apply().applied);
  EXPECT_EQ(read(m_path), "{not json");

  write(m_path, Empty);
  ASSERT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Accepted);
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.set_class(key(KeyText), "ground"));
  const std::string external = fmt::format(R"({{"version":1,"materials":{{"{}":{{}}}}}})", OtherText);
  write(m_path, external);
  const UserApplyResult changed = layer.apply();
  EXPECT_FALSE(changed.applied);
  EXPECT_TRUE(changed.changedOnDisk);
  EXPECT_EQ(read(m_path), external);
  EXPECT_FALSE(std::filesystem::exists(m_backup));

  ASSERT_EQ(layer.revert(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.has_entry(key(OtherText)));
  EXPECT_TRUE(layer.set_stochastic(key(KeyText), "scale", 20.0));
  const UserApplyResult rejected = layer.apply();
  EXPECT_FALSE(rejected.applied);
  ASSERT_TRUE(rejected.diagnostic.has_value());
  EXPECT_FALSE(rejected.diagnostic->rule.empty());
  EXPECT_EQ(read(m_path), external);
}

TEST_F(UserLayerTest, MissingFileIsCreated) {
  ASSERT_EQ(register_user().status, MaterialManifestStatus::Missing);
  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Missing);
  EXPECT_TRUE(layer.set_flags(key(KeyText), StochasticFlag));
  const UserApplyResult result = layer.apply();
  ASSERT_TRUE(result.applied) << result.message;
  EXPECT_EQ(result.status, MaterialManifestStatus::Accepted);
  EXPECT_TRUE(std::filesystem::exists(m_path));
  EXPECT_FALSE(std::filesystem::exists(m_backup));
  EXPECT_EQ(layer.state(), UserLayerState::Loaded);
  EXPECT_NE(record_for(key(KeyText)).summary & Stochastic, 0u);
}

TEST_F(UserLayerTest, PaletteKeyUsesTlutWildcard) {
  write(m_path, Empty);
  ASSERT_EQ(register_user().status, MaterialManifestStatus::Accepted);
  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.set_class(key(PaletteText), "ground"));
  const std::string text = layer.text();
  EXPECT_NE(text.find("tex1_16x16_000000000000b002_$_9"), std::string::npos);
  EXPECT_EQ(text.find("c003"), std::string::npos);
  ASSERT_TRUE(layer.apply().applied);
  texture::TextureSourceKey otherPalette = key(PaletteText);
  otherPalette.tlutHash = 0x1234;
  EXPECT_EQ(class_of(otherPalette), "ground");
  EXPECT_TRUE(layer.has_entry(otherPalette));
}

TEST_F(UserLayerTest, SessionOnlyWithoutPath) {
  {
    UserLayer layer;
    ASSERT_EQ(layer.open(), UserLayerState::NoPath);
    EXPECT_TRUE(layer.set_flags(key(KeyText), StochasticFlag));
    ASSERT_TRUE(layer.apply().applied);
    const uint64_t first = layer.session_registration();
    EXPECT_NE(first, 0u);
    EXPECT_EQ(session_registrations(), 1u);
    EXPECT_NE(record_for(key(KeyText)).summary & Stochastic, 0u);

    EXPECT_TRUE(layer.set_class(key(KeyText), "ground"));
    ASSERT_TRUE(layer.apply().applied);
    EXPECT_NE(layer.session_registration(), first);
    EXPECT_EQ(session_registrations(), 1u);
    EXPECT_EQ(class_of(key(KeyText)), "ground");
    EXPECT_EQ(layer.revert(), UserLayerState::NoPath);
    EXPECT_TRUE(layer.has_entry(key(KeyText)));
  }
  EXPECT_EQ(session_registrations(), 0u);
}

TEST_F(UserLayerTest, InvalidAfterReloadRestoresBase) {
  const std::string base = fmt::format(R"({{"version":1,"materials":{{"{}":{{"class":"base"}}}}}})", KeyText);
  write(m_path, base);
  const texture::MaterialManifestRegistration registration = register_user();
  ASSERT_EQ(registration.status, MaterialManifestStatus::Accepted);
  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.set_class(key(KeyText), "edited"));
  g_hookRegistration = registration.id;
  g_hookArmed = true;
  testing::set_check_hook(&supersede_hook);
  const UserApplyResult result = layer.apply();
  testing::set_check_hook(nullptr);
  EXPECT_FALSE(result.applied);
  EXPECT_EQ(result.status, MaterialManifestStatus::Superseded);
  EXPECT_FALSE(result.message.empty());
  EXPECT_EQ(read(m_path), base);
  const auto info = texture::material_manifest_info(registration);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->status, MaterialManifestStatus::Accepted);
  EXPECT_FALSE(info->overridden);
  EXPECT_EQ(class_of(key(KeyText)), "base");
}

TEST_F(UserLayerTest, ReplacedRegistrationIsRefused) {
  write(m_path, Empty);
  const texture::MaterialManifestRegistration registration = register_user();
  UserLayer layer;
  ASSERT_EQ(layer.open(), UserLayerState::Loaded);
  EXPECT_TRUE(layer.set_class(key(KeyText), "ground"));
  texture::unregister_material_manifest(registration);
  ASSERT_EQ(register_user().status, MaterialManifestStatus::Accepted);
  const UserApplyResult result = layer.apply();
  EXPECT_FALSE(result.applied);
  EXPECT_FALSE(result.message.empty());
  EXPECT_EQ(read(m_path), Empty);
  EXPECT_FALSE(std::filesystem::exists(m_backup));
}
} // namespace
} // namespace aurora::render::material

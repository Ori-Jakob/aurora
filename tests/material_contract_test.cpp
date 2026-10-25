#include "material_contract_check.hpp"

#include "gfx/dds_io.hpp"
#include "gfx/texture_replacement.hpp"
#include "internal.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aurora::test::c1 {
void PrintTo(const MergeWarning& warning, std::ostream* os) {
  *os << "{" << warning.file << ", " << warning.rule << ", " << warning.pointer << "}";
}

namespace {
constexpr std::string_view BaseSchema = "materials.schema.json";
constexpr std::string_view Layer1Schema = "materials.layer1.schema.json";
constexpr std::string_view Draft202012 = "https://json-schema.org/draft/2020-12/schema";

std::string text_of(const Json& object, std::string_view key) {
  const Json* value = object.find(key);
  return value != nullptr && value->type == Json::Type::String ? value->text : std::string{};
}

int64_t int_of(const Json& object, std::string_view key, int64_t fallback = 0) {
  const Json* value = object.find(key);
  return value != nullptr && value->type == Json::Type::Number ? static_cast<int64_t>(value->number) : fallback;
}

bool bool_of(const Json& object, std::string_view key) {
  const Json* value = object.find(key);
  return value != nullptr && value->type == Json::Type::Bool && value->boolean;
}

const std::vector<Json>& array_of(const Json& object, std::string_view key) {
  static const std::vector<Json> Empty;
  const Json* value = object.find(key);
  return value != nullptr && value->type == Json::Type::Array ? value->items : Empty;
}

std::string describe(const std::vector<Violation>& violations) {
  std::string out;
  for (const Violation& violation : violations) {
    out += fmt::format("\n  {} at '{}': {}", violation.rule, violation.pointer, violation.detail);
  }
  return out;
}

std::string join(const std::vector<std::string>& values) {
  std::string out;
  for (const std::string& value : values) {
    out += out.empty() ? value : ", " + value;
  }
  return out;
}

std::string dump(const Json& value) {
  switch (value.type) {
  case Json::Type::Null:
    return "null";
  case Json::Type::Bool:
    return value.boolean ? "true" : "false";
  case Json::Type::Number:
    return fmt::format("{}", value.number);
  case Json::Type::String:
    return fmt::format("\"{}\"", value.text);
  case Json::Type::Array: {
    std::string out = "[";
    for (size_t i = 0; i < value.items.size(); ++i) {
      out += (i == 0 ? "" : ", ") + dump(value.items[i]);
    }
    return out + "]";
  }
  case Json::Type::Object: {
    std::string out = "{";
    for (size_t i = 0; i < value.items.size(); ++i) {
      out += fmt::format("{}\"{}\": {}", i == 0 ? "" : ", ", value.keys[i], dump(value.items[i]));
    }
    return out + "}";
  }
  }
  return {};
}

void collect_rule_annotations(const Json& value, std::vector<std::string>& out) {
  for (size_t i = 0; i < value.items.size(); ++i) {
    if (value.type == Json::Type::Object && value.keys[i] == "x-c1-rule") {
      out.push_back(value.items[i].text);
    }
    collect_rule_annotations(value.items[i], out);
  }
}

bool is_known_rule(std::string_view rule) {
  const auto rules = known_rules();
  return std::find(rules.begin(), rules.end(), rule) != rules.end();
}

std::string_view kind_name(NameKind kind) {
  switch (kind) {
  case NameKind::Base:
    return "base";
  case NameKind::BaseMip:
    return "base-mip";
  case NameKind::Companion:
    return "companion";
  case NameKind::CompanionMip:
    return "companion-mip";
  case NameKind::Ignored:
    return "ignored";
  case NameKind::Rejected:
    return "rejected";
  }
  return {};
}

class MaterialContractTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    s_root = std::filesystem::path{AURORA_TEST_FIXTURES_DIR} / "materials";
    s_schemas.dir = s_root;
    const JsonParse parsed = parse_json(read_text(s_root / "index.json"));
    ASSERT_TRUE(parsed.value.has_value()) << parsed.error.detail;
    s_index = *parsed.value;
  }

  static std::string read_text(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    return bytes ? std::string{bytes->begin(), bytes->end()} : std::string{};
  }

  static Json load_json(const std::string& relative) {
    const JsonParse parsed = parse_json(read_text(s_root / relative));
    EXPECT_TRUE(parsed.value.has_value()) << relative << ": " << parsed.error.detail;
    return parsed.value.value_or(Json{});
  }

  static std::vector<DdsFile> load_dds_files() {
    std::vector<DdsFile> files;
    for (const auto& entry : std::filesystem::directory_iterator{s_root / "dds"}) {
      auto bytes = read_file(entry.path());
      EXPECT_TRUE(bytes.has_value()) << entry.path().string();
      files.push_back({entry.path().filename().string(), bytes.value_or(std::vector<uint8_t>{})});
    }
    return files;
  }

  static const std::vector<Json>& index_array(std::string_view key) { return array_of(s_index, key); }

  static inline std::filesystem::path s_root;
  static inline Json s_index;
  static inline SchemaSet s_schemas;
};

TEST_F(MaterialContractTest, SchemasLoadWithSupportedKeywordsOnly) {
  for (const std::string_view file : {BaseSchema, Layer1Schema}) {
    SCOPED_TRACE(file);
    const JsonParse parsed = parse_json(read_text(s_root / std::string{file}));
    ASSERT_TRUE(parsed.value.has_value()) << parsed.error.detail;
    EXPECT_EQ(text_of(*parsed.value, "$schema"), Draft202012);
    const auto unsupported = unsupported_schema_keywords(s_schemas, file);
    EXPECT_TRUE(unsupported.empty()) << join(unsupported);
    std::vector<std::string> rules;
    collect_rule_annotations(*parsed.value, rules);
    EXPECT_FALSE(rules.empty());
    for (const std::string& rule : rules) {
      EXPECT_TRUE(is_known_rule(rule)) << rule;
    }
  }
}

TEST_F(MaterialContractTest, IndexCoversEveryFixtureFile) {
  std::set<std::string> referenced;
  for (const Json& entry : index_array("manifests")) {
    referenced.insert(text_of(entry, "file"));
  }
  for (const Json& entry : index_array("merges")) {
    const std::string dir = text_of(entry, "dir");
    for (const Json& file : array_of(entry, "files")) {
      referenced.insert(dir + "/" + text_of(file, "file"));
    }
    referenced.insert(dir + "/" + text_of(entry, "result"));
  }
  for (const Json& entry : index_array("lookups")) {
    referenced.insert(text_of(entry, "manifest"));
  }
  for (const Json& entry : index_array("dds")) {
    referenced.insert("dds/" + text_of(entry, "file"));
  }
  ASSERT_FALSE(referenced.empty());
  for (const std::string& path : referenced) {
    EXPECT_TRUE(std::filesystem::is_regular_file(s_root / path)) << path;
  }

  const std::set<std::string> exempt{"materials.schema.json", "materials.layer1.schema.json", "index.json",
                                     "gen_dds_fixtures.py", ".gitattributes"};
  for (const auto& entry : std::filesystem::recursive_directory_iterator{s_root}) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string relative = entry.path().lexically_relative(s_root).generic_string();
    if (!exempt.contains(relative)) {
      EXPECT_TRUE(referenced.contains(relative)) << relative;
    }
  }
}

TEST_F(MaterialContractTest, IndexRulesAreKnownAndCovered) {
  std::set<std::string> used;
  const auto add = [&](const Json& entry) {
    if (std::string rule = text_of(entry, "rule"); !rule.empty()) {
      used.insert(std::move(rule));
    }
  };
  for (const Json& entry : index_array("manifests")) {
    add(entry);
  }
  for (const Json& entry : index_array("merges")) {
    for (const Json& warning : array_of(entry, "warnings")) {
      add(warning);
    }
  }
  for (const Json& entry : index_array("names")) {
    add(entry);
  }
  for (const Json& entry : index_array("dds")) {
    add(entry);
  }
  for (const std::string& rule : used) {
    EXPECT_TRUE(is_known_rule(rule)) << rule;
  }

  std::set<std::string> expected;
  for (const std::string_view rule : known_rules()) {
    // Every schema "not" carries a more specific rule, so nothing can raise the generic one.
    if (rule != "MANIFEST-NOT") {
      expected.emplace(rule);
    }
  }
  EXPECT_EQ(used, expected);
}

TEST_F(MaterialContractTest, ManifestFixturesMatchExpectedVerdicts) {
  ASSERT_FALSE(index_array("manifests").empty());
  for (const Json& entry : index_array("manifests")) {
    const std::string file = text_of(entry, "file");
    const auto layer = static_cast<int>(int_of(entry, "layer"));
    SCOPED_TRACE(fmt::format("{} at layer {}", file, layer));
    const auto violations = check_manifest(s_schemas, read_text(s_root / file), layer);
    if (text_of(entry, "expect") == "valid") {
      EXPECT_TRUE(violations.empty()) << describe(violations);
      continue;
    }
    EXPECT_EQ(violations.size(), 1u) << describe(violations);
    if (!violations.empty()) {
      EXPECT_EQ(violations.front().rule, text_of(entry, "rule"));
      EXPECT_EQ(violations.front().pointer, text_of(entry, "pointer"));
    }
  }
}

TEST_F(MaterialContractTest, ManifestKeysRoundTripThroughReplacementParser) {
  std::set<std::string> files;
  for (const Json& entry : index_array("manifests")) {
    if (text_of(entry, "expect") == "valid") {
      files.insert(text_of(entry, "file"));
    }
  }
  for (const Json& entry : index_array("merges")) {
    files.insert(text_of(entry, "dir") + "/" + text_of(entry, "result"));
  }

  size_t checked = 0;
  for (const std::string& file : files) {
    SCOPED_TRACE(file);
    const Json manifest = load_json(file);
    std::vector<std::string> keys;
    if (const Json* materials = manifest.find("materials"); materials != nullptr) {
      keys = materials->keys;
    }
    if (const Json* types = manifest.find("grassTypes"); types != nullptr) {
      for (const Json& type : types->items) {
        if (std::string colorFrom = text_of(type, "colorFrom"); !colorFrom.empty()) {
          keys.push_back(std::move(colorFrom));
        }
      }
    }
    for (const std::string& key : keys) {
      SCOPED_TRACE(key);
      const auto parsed = texture::parse_replacement_filename(key + ".dds");
      EXPECT_TRUE(parsed.has_value());
      if (!parsed) {
        continue;
      }
      EXPECT_NE(parsed->textureHash, texture::kWildcardTextureHash);
      EXPECT_EQ(canonical_key(*parsed), key);
      ++checked;
    }
  }
  EXPECT_GT(checked, 0u);
}

TEST_F(MaterialContractTest, MergeFixturesProduceExpectedResult) {
  ASSERT_FALSE(index_array("merges").empty());
  for (const Json& entry : index_array("merges")) {
    SCOPED_TRACE(text_of(entry, "name"));
    const std::string dir = text_of(entry, "dir");
    std::vector<LayerFile> files;
    for (const Json& file : array_of(entry, "files")) {
      const std::string name = text_of(file, "file");
      files.push_back({name, static_cast<int>(int_of(file, "layer")), static_cast<int32_t>(int_of(file, "priority")),
                       read_text(s_root / dir / name)});
    }
    const MergeResult result = merge_layers(s_schemas, files, bool_of(entry, "replacements"));
    EXPECT_TRUE(json_equal(result.manifest, load_json(dir + "/" + text_of(entry, "result")))) << dump(result.manifest);

    std::vector<MergeWarning> expected;
    for (const Json& warning : array_of(entry, "warnings")) {
      expected.push_back({text_of(warning, "file"), text_of(warning, "rule"), text_of(warning, "pointer")});
    }
    std::vector<MergeWarning> warnings = result.warnings;
    std::sort(warnings.begin(), warnings.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(warnings, expected);
  }
}

TEST_F(MaterialContractTest, MergeResultsAreStable) {
  std::set<std::string> results;
  for (const Json& entry : index_array("merges")) {
    results.insert(text_of(entry, "dir") + "/" + text_of(entry, "result"));
  }
  ASSERT_FALSE(results.empty());
  for (const std::string& file : results) {
    SCOPED_TRACE(file);
    const std::string text = read_text(s_root / file);
    const auto violations = check_manifest(s_schemas, text, 2);
    EXPECT_TRUE(violations.empty()) << describe(violations);

    const LayerFile single{file, 4, 0, text};
    const MergeResult result = merge_layers(s_schemas, std::span<const LayerFile>{&single, 1}, true);
    EXPECT_TRUE(result.warnings.empty());
    EXPECT_TRUE(json_equal(result.manifest, load_json(file))) << dump(result.manifest);
  }
}

TEST_F(MaterialContractTest, LookupsResolveMostSpecificKey) {
  ASSERT_FALSE(index_array("lookups").empty());
  for (const Json& entry : index_array("lookups")) {
    const std::string query = text_of(entry, "query");
    SCOPED_TRACE(query);
    const auto resolved = lookup_material(load_json(text_of(entry, "manifest")), query);
    const Json* expected = entry.find("resolved");
    ASSERT_NE(expected, nullptr);
    if (expected->type == Json::Type::Null) {
      EXPECT_FALSE(resolved.has_value()) << resolved.value_or("");
    } else {
      EXPECT_EQ(resolved.value_or("<none>"), expected->text);
    }
  }
}

TEST_F(MaterialContractTest, NameFixturesClassify) {
  ASSERT_FALSE(index_array("names").empty());
  for (const Json& entry : index_array("names")) {
    const std::string name = text_of(entry, "name");
    SCOPED_TRACE(name);
    const NameInfo info = classify_name(name);
    EXPECT_EQ(kind_name(info.kind), text_of(entry, "kind"));
    EXPECT_EQ(info.key, text_of(entry, "key"));
    EXPECT_EQ(info.role, text_of(entry, "role"));
    EXPECT_EQ(info.mip, static_cast<uint32_t>(int_of(entry, "mip")));
    EXPECT_EQ(info.rule, text_of(entry, "rule"));
    EXPECT_EQ(texture::parse_replacement_filename(name).has_value(), bool_of(entry, "legacyParse"));
  }
}

TEST_F(MaterialContractTest, CompanionFilesAreInvisibleToLegacyScan) {
  size_t companions = 0;
  for (const auto& entry : std::filesystem::directory_iterator{s_root / "dds"}) {
    const std::string name = entry.path().filename().string();
    const NameKind kind = classify_name(name).kind;
    if (kind != NameKind::Companion && kind != NameKind::CompanionMip) {
      continue;
    }
    ++companions;
    EXPECT_FALSE(texture::parse_replacement_filename(name).has_value()) << name;
  }
  EXPECT_GT(companions, 0u);
}

TEST_F(MaterialContractTest, DdsHeadersMatchIndex) {
  ASSERT_FALSE(index_array("dds").empty());
  for (const Json& entry : index_array("dds")) {
    const std::string file = text_of(entry, "file");
    SCOPED_TRACE(file);
    const auto bytes = read_file(s_root / "dds" / file);
    ASSERT_TRUE(bytes.has_value());
    Violation error;
    const auto header = read_dds_header(*bytes, &error);
    if (const std::string defect = text_of(entry, "defect"); !defect.empty()) {
      EXPECT_FALSE(header.has_value());
      EXPECT_EQ(error.rule, defect == "truncate" ? "DDS-SIZE" : "DDS-HEADER");
      continue;
    }
    ASSERT_TRUE(header.has_value()) << error.detail;
    const std::string style = text_of(entry, "header");
    const DdsStyle expectedStyle = style == "gz2" ? DdsStyle::Gz2 : style == "ati2" ? DdsStyle::FourCC : DdsStyle::Dx10;
    EXPECT_EQ(header->style, expectedStyle);
    EXPECT_EQ(header->code, static_cast<uint32_t>(int_of(entry, "dxgi", 83)));
    EXPECT_EQ(header->width, static_cast<uint32_t>(int_of(entry, "width")));
    EXPECT_EQ(header->height, static_cast<uint32_t>(int_of(entry, "height")));
    EXPECT_EQ(header->mips, static_cast<uint32_t>(int_of(entry, "mips")));
    EXPECT_EQ(bytes->size(), header->dataOffset + header->dataSize);
  }
}

TEST_F(MaterialContractTest, DdsIoParsesSupportedFormats) {
  ASSERT_FALSE(index_array("dds").empty());
  for (const Json& entry : index_array("dds")) {
    const std::string file = text_of(entry, "file");
    SCOPED_TRACE(file);
    const auto bytes = read_file(s_root / "dds" / file);
    ASSERT_TRUE(bytes.has_value());
    const auto converted = gfx::dds::parse_dds_bytes({bytes->data(), bytes->size()});
    EXPECT_EQ(converted.has_value(), text_of(entry, "ddsIo") == "parse");
    if (!converted) {
      continue;
    }
    EXPECT_EQ(converted->width, static_cast<uint32_t>(int_of(entry, "width")));
    EXPECT_EQ(converted->height, static_cast<uint32_t>(int_of(entry, "height")));
    EXPECT_EQ(converted->mips, static_cast<uint32_t>(int_of(entry, "mips")));
    const int64_t code = int_of(entry, "dxgi", 83);
    constexpr std::array<std::pair<int64_t, wgpu::TextureFormat>, 5> Formats{{
        {98, wgpu::TextureFormat::BC7RGBAUnorm},
        {99, wgpu::TextureFormat::BC7RGBAUnormSrgb},
        {83, wgpu::TextureFormat::BC5RGUnorm},
        {80, wgpu::TextureFormat::BC4RUnorm},
        {95, wgpu::TextureFormat::BC6HRGBUfloat},
    }};
    const auto it = std::find_if(Formats.begin(), Formats.end(), [code](const auto& row) { return row.first == code; });
    if (it == Formats.end()) {
      ADD_FAILURE() << "parsed unexpected DXGI format " << code;
      continue;
    }
    EXPECT_EQ(converted->format, it->second);
  }
}

TEST_F(MaterialContractTest, DdsCompanionRulesMatchIndex) {
  const auto verdicts = check_dds_files(load_dds_files());
  EXPECT_EQ(verdicts.size(), index_array("dds").size());
  for (const Json& entry : index_array("dds")) {
    const std::string file = text_of(entry, "file");
    SCOPED_TRACE(file);
    const auto it = verdicts.find(file);
    ASSERT_TRUE(it != verdicts.end());
    EXPECT_EQ(it->second.verdict, text_of(entry, "expect"));
    EXPECT_EQ(it->second.rule, text_of(entry, "rule"));
    EXPECT_EQ(it->second.chain, text_of(entry, "chain"));
  }
}

TEST_F(MaterialContractTest, ClampedChainMirrorsPackLayout) {
  const std::string stem = "tex1_16x16_000000000000c001_14";
  const auto verdicts = check_dds_files(load_dds_files());
  const auto check_chain = [&](std::string_view role, const std::vector<uint32_t>& expected) {
    SCOPED_TRACE(role.empty() ? std::string_view{"base"} : role);
    const std::string suffix = role.empty() ? std::string{".dds"} : fmt::format(".{}.dds", role);
    std::vector<std::string> names{stem + suffix};
    while (std::filesystem::is_regular_file(s_root / "dds" / fmt::format("{}_mip{}{}", stem, names.size(), suffix))) {
      names.push_back(fmt::format("{}_mip{}{}", stem, names.size(), suffix));
    }
    std::vector<uint32_t> sizes;
    for (const std::string& name : names) {
      const auto bytes = read_file(s_root / "dds" / name);
      ASSERT_TRUE(bytes.has_value()) << name;
      const auto header = read_dds_header(*bytes, nullptr);
      ASSERT_TRUE(header.has_value()) << name;
      EXPECT_EQ(header->height, header->width) << name;
      sizes.push_back(header->width);
    }
    EXPECT_EQ(sizes, expected);
    ASSERT_GE(names.size(), 2u);
    for (size_t i = names.size() - 2; i < names.size(); ++i) {
      const auto it = verdicts.find(names[i]);
      ASSERT_TRUE(it != verdicts.end()) << names[i];
      EXPECT_EQ(it->second.chain, "clamped") << names[i];
    }
  };
  // GZ2 packs stop downscaling at the 16x16 native size, so their tails repeat it.
  check_chain({}, {64, 32, 16, 16, 16});
  check_chain("mat", {32, 16, 16, 16});
}

TEST_F(MaterialContractTest, CompanionParserMatchesC1Names) {
  std::vector<std::string> names;
  for (const Json& entry : index_array("names")) {
    names.push_back(text_of(entry, "name"));
  }
  EXPECT_EQ(names.size(), 30u);
  for (const Json& entry : index_array("dds")) {
    names.push_back(text_of(entry, "file"));
  }
  EXPECT_EQ(names.size(), 77u);
  size_t companions = 0;
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const NameInfo info = classify_name(name);
    const auto parsed = gfx::texture_replacement::parse_companion_filename(name);
    ASSERT_EQ(parsed.has_value(), info.kind == NameKind::Companion || info.kind == NameKind::CompanionMip);
    if (!parsed) {
      continue;
    }
    ++companions;
    EXPECT_EQ(canonical_key(parsed->key), info.key);
    EXPECT_EQ(gfx::texture_replacement::companion_role_name(parsed->role), info.role);
    EXPECT_EQ(parsed->mip, info.mip);
  }
  EXPECT_GT(companions, 0u);
}

class MaterialDiscoveryTest : public MaterialContractTest {
protected:
  void SetUp() override {
    gfx::texture_replacement::shutdown();
    gfx::texture_replacement::testing::set_worker_count(1);
    gfx::texture_replacement::testing::set_workers_paused(true);
    m_cachePath = (std::filesystem::temp_directory_path() /
                   ("aurora-discovery-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
                      .string();
    m_savedCachePath = g_config.cachePath;
    g_config.cachePath = m_cachePath.c_str();
  }

  void TearDown() override {
    g_config.cachePath = m_savedCachePath;
    gfx::texture_replacement::shutdown();
    std::error_code ec;
    std::filesystem::remove_all(m_cachePath, ec);
  }

  std::string m_cachePath;
  const char* m_savedCachePath = nullptr;
};

TEST_F(MaterialDiscoveryTest, DiscoveryOverDdsFixtures) {
  using gfx::texture_replacement::CompanionRole;
  const auto group = texture::load_replacement_directory(s_root / "dds");
  std::set<std::string> registrations;
  for (const auto& registration : group.registrations) {
    registrations.insert(canonical_key(std::get<texture::TextureSourceKey>(registration.key)));
  }
  EXPECT_EQ(registrations, (std::set<std::string>{"tex1_16x16_000000000000a001_14", "tex1_16x16_000000000000c001_14"}));
  EXPECT_EQ(group.companions.size(), 24u);
  EXPECT_TRUE(group.manifests.empty());

  const auto roles = companion_roles();
  size_t levelZero = 0;
  for (const auto& entry : std::filesystem::directory_iterator{s_root / "dds"}) {
    const std::string name = entry.path().filename().string();
    const NameInfo info = classify_name(name);
    if (info.kind != NameKind::Companion) {
      continue;
    }
    SCOPED_TRACE(name);
    ++levelZero;
    const auto key = texture::parse_replacement_filename(info.key + ".dds");
    ASSERT_TRUE(key.has_value());
    const auto role = static_cast<CompanionRole>(std::find(roles.begin(), roles.end(), info.role) - roles.begin());
    const auto file = gfx::texture_replacement::find_companion(*key, role);
    ASSERT_TRUE(file.has_value());
    EXPECT_EQ(file->path.filename().string(), name);
  }
  EXPECT_EQ(levelZero, 24u);

  const texture::TextureSourceKey d0f1{
      .textureHash = 0xd001, .tlutHash = 0xd0f1, .width = 16, .height = 16, .format = 9, .hasTlut = true};
  EXPECT_EQ(gfx::texture_replacement::companion_roles(d0f1), 1u << static_cast<uint32_t>(CompanionRole::Orm));

  texture::unregister_replacements(group);
  EXPECT_EQ(gfx::texture_replacement::companion_roles(d0f1), 0u);
}

struct SplitMix64 {
  uint64_t state = 0;

  uint64_t next() noexcept {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
};

std::vector<uint8_t> splitmix64_bytes(uint64_t seed, size_t size) {
  SplitMix64 rng{seed};
  std::vector<uint8_t> out;
  out.reserve(size + sizeof(uint64_t));
  while (out.size() < size) {
    const uint64_t value = rng.next();
    for (size_t i = 0; i < sizeof(value); ++i) {
      out.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
  }
  out.resize(size);
  return out;
}

std::vector<uint8_t> palette_bytes(uint32_t format, uint64_t seed, size_t size, uint32_t lo, uint32_t hi) {
  const size_t count = format == GX_TF_C4 ? size * 2 : format == GX_TF_C8 ? size : size / 2;
  SplitMix64 rng{seed};
  std::vector<uint8_t> out;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t r = rng.next();
    const uint32_t index = i == 0 ? lo : i == count - 1 ? hi : lo + static_cast<uint32_t>(r % (hi - lo + 1));
    if (format == GX_TF_C4) {
      if (i % 2 == 0) {
        out.push_back(static_cast<uint8_t>(index << 4));
      } else {
        out.back() |= static_cast<uint8_t>(index);
      }
    } else if (format == GX_TF_C8) {
      out.push_back(static_cast<uint8_t>(index));
    } else {
      const uint32_t value = static_cast<uint32_t>(r >> 62) << 14 | index;
      out.push_back(static_cast<uint8_t>(value >> 8));
      out.push_back(static_cast<uint8_t>(value));
    }
  }
  return out;
}

Json load_source_key_fixture() {
  const auto bytes = read_file(std::filesystem::path{AURORA_TEST_FIXTURES_DIR} / "source_keys" / "source_keys.json");
  EXPECT_TRUE(bytes.has_value());
  const std::string text = bytes ? std::string{bytes->begin(), bytes->end()} : std::string{};
  const JsonParse parsed = parse_json(text);
  EXPECT_TRUE(parsed.value.has_value()) << parsed.error.detail;
  EXPECT_EQ(parsed.value ? int_of(*parsed.value, "version") : 0, 1);
  return parsed.value.value_or(Json{});
}

std::string hex_of(uint64_t value) { return fmt::format("{:016x}", value); }

TEST(SourceKeyParity, KeysMatchPythonFixture) {
  const Json fixture = load_source_key_fixture();
  const auto& cases = array_of(fixture, "cases");
  ASSERT_FALSE(cases.empty());
  for (const Json& entry : cases) {
    SCOPED_TRACE(text_of(entry, "name"));
    const auto format = static_cast<uint32_t>(int_of(entry, "format"));
    const auto width = static_cast<uint32_t>(int_of(entry, "width"));
    const auto height = static_cast<uint32_t>(int_of(entry, "height"));
    const Json* textureSpec = entry.find("texture");
    ASSERT_NE(textureSpec, nullptr);
    const auto seed = static_cast<uint64_t>(int_of(*textureSpec, "seed"));
    const auto size = static_cast<size_t>(int_of(*textureSpec, "size"));
    std::vector<uint8_t> texture;
    if (const auto& indices = array_of(*textureSpec, "indices"); !indices.empty()) {
      ASSERT_EQ(indices.size(), 2u);
      texture = palette_bytes(format, seed, size, static_cast<uint32_t>(indices[0].number),
                              static_cast<uint32_t>(indices[1].number));
    } else {
      texture = splitmix64_bytes(seed, size);
    }
    ASSERT_EQ(texture.size(), size);
    std::vector<uint8_t> tlut;
    if (const Json* tlutSpec = entry.find("tlut"); tlutSpec != nullptr && tlutSpec->type == Json::Type::Object) {
      tlut = splitmix64_bytes(static_cast<uint64_t>(int_of(*tlutSpec, "seed")),
                              static_cast<size_t>(int_of(*tlutSpec, "entries")) * sizeof(uint16_t));
    }

    const auto key = texture::compute_source_key(format, width, height, texture, tlut);
    const Json* expected = entry.find("key");
    ASSERT_NE(expected, nullptr);
    if (expected->type == Json::Type::Null) {
      EXPECT_FALSE(key.has_value());
      continue;
    }
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(hex_of(key->textureHash), text_of(*expected, "textureHash"));
    EXPECT_EQ(hex_of(key->tlutHash), text_of(*expected, "tlutHash"));
    EXPECT_EQ(key->hasTlut, bool_of(*expected, "hasTlut"));
    EXPECT_EQ(key->width, width);
    EXPECT_EQ(key->height, height);
    EXPECT_EQ(key->format, format);

    const auto baseSize = static_cast<size_t>(int_of(entry, "baseSize"));
    ASSERT_GT(baseSize, 0u);
    ASSERT_LE(baseSize, texture.size());
    const std::span<const uint8_t> bytes{texture};
    EXPECT_EQ(texture::compute_source_key(format, width, height, bytes.first(baseSize), tlut), key);
    EXPECT_FALSE(texture::compute_source_key(format, width, height, bytes.first(baseSize - 1), tlut).has_value());
  }
}

TEST(SourceKeyParity, NamesMatchPythonFixture) {
  const Json fixture = load_source_key_fixture();
  const auto& names = array_of(fixture, "names");
  ASSERT_FALSE(names.empty());
  for (const Json& entry : names) {
    const std::string name = text_of(entry, "name");
    SCOPED_TRACE(name);
    const auto parsed = texture::parse_replacement_filename(name);
    const Json* expected = entry.find("key");
    ASSERT_NE(expected, nullptr);
    if (expected->type == Json::Type::Null) {
      EXPECT_FALSE(parsed.has_value());
      continue;
    }
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(hex_of(parsed->textureHash), text_of(*expected, "textureHash"));
    EXPECT_EQ(hex_of(parsed->tlutHash), text_of(*expected, "tlutHash"));
    EXPECT_EQ(parsed->width, static_cast<uint32_t>(int_of(*expected, "width")));
    EXPECT_EQ(parsed->height, static_cast<uint32_t>(int_of(*expected, "height")));
    EXPECT_EQ(parsed->format, static_cast<uint32_t>(int_of(*expected, "format")));
    EXPECT_EQ(parsed->hasTlut, bool_of(*expected, "hasTlut"));
    EXPECT_EQ(canonical_key(*parsed), text_of(entry, "canonical"));
  }
}
} // namespace
} // namespace aurora::test::c1

#include "material_contract_check.hpp"

#include "io.hpp"
#include "render/material/gx_lookup.hpp"
#include "render/material/manifest.hpp"
#include "render/material/registry.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace aurora::test {
namespace {
namespace material = aurora::render::material;
using texture::MaterialLayer;
using texture::MaterialManifestStatus;
using namespace std::chrono_literals;

constexpr std::string_view KeyA = "tex1_16x16_000000000000a001_14";
constexpr std::string_view KeyB = "tex1_16x16_000000000000a002_14";
constexpr std::string_view KeyPalette = "tex1_16x16_000000000000d001_$_9";

std::string text_of(const c1::Json& object, std::string_view key) {
  const c1::Json* value = object.find(key);
  return value != nullptr && value->type == c1::Json::Type::String ? value->text : std::string{};
}

int64_t int_of(const c1::Json& object, std::string_view key, int64_t fallback = 0) {
  const c1::Json* value = object.find(key);
  return value != nullptr && value->type == c1::Json::Type::Number ? static_cast<int64_t>(value->number) : fallback;
}

bool bool_of(const c1::Json& object, std::string_view key) {
  const c1::Json* value = object.find(key);
  return value != nullptr && value->type == c1::Json::Type::Bool && value->boolean;
}

const std::vector<c1::Json>& array_of(const c1::Json& object, std::string_view key) {
  static const std::vector<c1::Json> Empty;
  const c1::Json* value = object.find(key);
  return value != nullptr && value->type == c1::Json::Type::Array ? value->items : Empty;
}

size_t members_of(const c1::Json& object, std::string_view key) {
  const c1::Json* value = object.find(key);
  return value != nullptr && value->type == c1::Json::Type::Object ? value->keys.size() : 0;
}

std::string read_text(const std::filesystem::path& path) {
  const auto bytes = c1::read_file(path);
  return bytes ? std::string{bytes->begin(), bytes->end()} : std::string{};
}

c1::Json parse(std::string_view text) {
  c1::JsonParse parsed = c1::parse_json(text);
  EXPECT_TRUE(parsed.value.has_value()) << parsed.error.detail << " in " << text;
  return parsed.value.value_or(c1::Json{});
}

void dump_string(std::string_view text, std::string& out) {
  out += '"';
  for (const char ch : text) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (static_cast<unsigned char>(ch) < 0x20) {
      out += fmt::format("\\u{:04x}", static_cast<unsigned char>(ch));
    } else {
      out += ch;
    }
  }
  out += '"';
}

void dump_json(const c1::Json& value, std::string& out) {
  switch (value.type) {
  case c1::Json::Type::Null:
    out += "null";
    break;
  case c1::Json::Type::Bool:
    out += value.boolean ? "true" : "false";
    break;
  case c1::Json::Type::Number:
    out += fmt::format("{}", value.number);
    break;
  case c1::Json::Type::String:
    dump_string(value.text, out);
    break;
  case c1::Json::Type::Array:
    out += '[';
    for (size_t i = 0; i < value.items.size(); ++i) {
      out += i == 0 ? "" : ",";
      dump_json(value.items[i], out);
    }
    out += ']';
    break;
  case c1::Json::Type::Object:
    out += '{';
    for (size_t i = 0; i < value.items.size(); ++i) {
      out += i == 0 ? "" : ",";
      dump_string(value.keys[i], out);
      out += ':';
      dump_json(value.items[i], out);
    }
    out += '}';
    break;
  }
}

std::string dump_json(const c1::Json& value) {
  std::string out;
  dump_json(value, out);
  return out;
}

texture::TextureSourceKey key_of(std::string_view text) {
  const auto key = texture::parse_material_key(text);
  EXPECT_TRUE(key.has_value()) << text;
  return key.value_or(texture::TextureSourceKey{});
}

uint16_t find_id(std::string_view text) { return material::snapshot()->find(key_of(text)); }

std::string manifest_with(std::string_view materials, std::string_view grassTypes = {}) {
  std::string out = R"({"version":1)";
  if (!grassTypes.empty()) {
    out += fmt::format(R"(,"grassTypes":{{{}}})", grassTypes);
  }
  out += fmt::format(R"(,"materials":{{{}}}}})", materials);
  return out;
}

texture::MaterialManifestRegistration add(std::string_view text, MaterialLayer layer = MaterialLayer::Pack,
                                          std::string_view label = {}, int32_t priority = 0,
                                          std::string_view base = {}) {
  return texture::register_material_manifest(text,
                                             {.layer = layer, .priority = priority, .label = label, .base = base});
}

std::vector<c1::MergeWarning> report_warnings() {
  std::vector<c1::MergeWarning> out;
  for (const texture::MaterialDiagnostic& diagnostic : texture::material_report().diagnostics) {
    out.push_back({diagnostic.label, diagnostic.rule, diagnostic.pointer});
  }
  std::sort(out.begin(), out.end());
  return out;
}

void reset_registry() {
  material::set_publish_observer(nullptr);
  material::testing::set_check_hook(nullptr);
  texture::clear_material_manifests();
  texture::set_material_layer_enabled(MaterialLayer::Pack, true);
  texture::set_material_layer_enabled(MaterialLayer::Mod, true);
}

std::atomic<int> g_observed{0};
std::atomic<uint64_t> g_observedGeneration{0};
std::atomic<uint64_t> g_observedBindEpoch{0};
std::atomic<uint32_t> g_observedLive{0};

void observe(const std::shared_ptr<const material::Snapshot>& snapshot) noexcept {
  g_observedGeneration.store(snapshot->generation);
  g_observedBindEpoch.store(snapshot->bindEpoch);
  g_observedLive.store(snapshot->liveCount);
  g_observed.fetch_add(1);
}

struct HookGate {
  std::mutex mutex;
  std::condition_variable changed;
  int arrivals = 0;
  std::array<bool, 4> open{};

  bool wait_arrivals(int count) {
    std::unique_lock lock{mutex};
    return changed.wait_for(lock, 10s, [&] { return arrivals >= count; });
  }

  void release(int ticket) {
    std::lock_guard lock{mutex};
    open[ticket] = true;
    changed.notify_all();
  }
};

HookGate* g_gate = nullptr;

void gated_check(uint64_t) {
  std::unique_lock lock{g_gate->mutex};
  const int ticket = g_gate->arrivals++;
  g_gate->changed.notify_all();
  if (ticket < static_cast<int>(g_gate->open.size())) {
    g_gate->changed.wait_for(lock, 10s, [&] { return g_gate->open[ticket]; });
  }
}

class MaterialRegistryTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    s_root = std::filesystem::path{AURORA_TEST_FIXTURES_DIR} / "materials";
    s_schemas.dir = s_root;
    s_index = parse(read_text(s_root / "index.json"));
  }

  void SetUp() override {
    reset_registry();
    g_observed = 0;
  }

  void TearDown() override { reset_registry(); }

  static std::string fixture(const std::string& relative) { return read_text(s_root / relative); }

  static const std::vector<c1::Json>& index_array(std::string_view key) { return array_of(s_index, key); }

  static c1::Json exported() { return parse(texture::material_manifest_json(false)); }

  static inline std::filesystem::path s_root;
  static inline c1::Json s_index;
  static inline c1::SchemaSet s_schemas;
};

TEST_F(MaterialRegistryTest, ManifestVerdictsMatchIndex) {
  ASSERT_FALSE(index_array("manifests").empty());
  for (const c1::Json& entry : index_array("manifests")) {
    const std::string file = text_of(entry, "file");
    const auto layer = static_cast<int>(int_of(entry, "layer"));
    SCOPED_TRACE(fmt::format("{} at layer {}", file, layer));
    const std::string text = fixture(file);
    const auto oracle = c1::check_manifest(s_schemas, text, layer);
    const auto registration = add(text, static_cast<MaterialLayer>(layer), file);
    ASSERT_NE(registration.id, 0u);
    const auto info = texture::material_manifest_info(registration);
    ASSERT_TRUE(info.has_value());
    const auto check = texture::check_material_manifest(text, static_cast<MaterialLayer>(layer));
    if (text_of(entry, "expect") == "valid") {
      EXPECT_TRUE(oracle.empty());
      EXPECT_EQ(registration.status, MaterialManifestStatus::Accepted);
      EXPECT_EQ(info->status, MaterialManifestStatus::Accepted);
      EXPECT_FALSE(info->violation.has_value());
      EXPECT_FALSE(check.has_value());
      const c1::Json manifest = parse(text);
      EXPECT_EQ(info->materialCount, members_of(manifest, "materials"));
      EXPECT_EQ(info->grassTypeCount, members_of(manifest, "grassTypes"));
    } else {
      ASSERT_FALSE(oracle.empty());
      EXPECT_EQ(oracle.front().rule, text_of(entry, "rule"));
      EXPECT_EQ(oracle.front().pointer, text_of(entry, "pointer"));
      EXPECT_EQ(registration.status, MaterialManifestStatus::Rejected);
      ASSERT_TRUE(info->violation.has_value());
      EXPECT_EQ(info->violation->label, file);
      EXPECT_EQ(info->violation->rule, text_of(entry, "rule"));
      EXPECT_EQ(info->violation->pointer, text_of(entry, "pointer"));
      EXPECT_EQ(info->materialCount, 0u);
      ASSERT_TRUE(check.has_value());
      EXPECT_EQ(check->rule, text_of(entry, "rule"));
      EXPECT_EQ(check->pointer, text_of(entry, "pointer"));
    }
    texture::unregister_material_manifest(registration);
  }
}

TEST_F(MaterialRegistryTest, LayerOneAllowListEnforced) {
  std::set<std::string> files;
  for (const c1::Json& entry : index_array("manifests")) {
    if (const std::string file = text_of(entry, "file"); file.find("/layer1_") != std::string::npos) {
      files.insert(file);
    }
  }
  EXPECT_EQ(files.size(), 11u);
  for (const std::string& file : files) {
    SCOPED_TRACE(file);
    const std::string text = fixture(file);
    EXPECT_EQ(add(text, MaterialLayer::Game, file).status, MaterialManifestStatus::Rejected);
    EXPECT_EQ(add(text, MaterialLayer::Pack, file).status, MaterialManifestStatus::Accepted);
    texture::clear_material_manifests();
  }
  const std::string layer1 = fixture("manifests/valid/layer1.json");
  EXPECT_EQ(add(layer1, MaterialLayer::Game).status, MaterialManifestStatus::Accepted);
  EXPECT_EQ(add(layer1, MaterialLayer::Pack).status, MaterialManifestStatus::Accepted);
}

TEST_F(MaterialRegistryTest, MergesMatchIndex) {
  ASSERT_FALSE(index_array("merges").empty());
  for (const c1::Json& entry : index_array("merges")) {
    SCOPED_TRACE(text_of(entry, "name"));
    texture::clear_material_manifests();
    const bool replacements = bool_of(entry, "replacements");
    texture::set_material_layer_enabled(MaterialLayer::Pack, replacements);
    texture::set_material_layer_enabled(MaterialLayer::Mod, replacements);
    const std::string dir = text_of(entry, "dir");
    for (const c1::Json& file : array_of(entry, "files")) {
      const std::string name = text_of(file, "file");
      const auto registration = add(fixture(dir + "/" + name), static_cast<MaterialLayer>(int_of(file, "layer")), name,
                                    static_cast<int32_t>(int_of(file, "priority")));
      EXPECT_NE(registration.id, 0u);
    }
    const c1::Json expected = parse(fixture(dir + "/" + text_of(entry, "result")));
    const c1::Json merged = exported();
    EXPECT_TRUE(c1::json_equal(merged, expected)) << texture::material_manifest_json(true);
    EXPECT_TRUE(c1::json_equal(parse(texture::material_manifest_json(true)), expected));

    std::vector<c1::MergeWarning> warnings;
    for (const c1::Json& warning : array_of(entry, "warnings")) {
      warnings.push_back({text_of(warning, "file"), text_of(warning, "rule"), text_of(warning, "pointer")});
    }
    std::sort(warnings.begin(), warnings.end());
    EXPECT_EQ(report_warnings(), warnings);
  }
}

TEST_F(MaterialRegistryTest, MergeResultsAreStable) {
  std::set<std::string> results;
  for (const c1::Json& entry : index_array("merges")) {
    results.insert(text_of(entry, "dir") + "/" + text_of(entry, "result"));
  }
  ASSERT_FALSE(results.empty());
  for (const std::string& file : results) {
    SCOPED_TRACE(file);
    texture::clear_material_manifests();
    const std::string text = fixture(file);
    EXPECT_EQ(add(text, MaterialLayer::User, file).status, MaterialManifestStatus::Accepted);
    EXPECT_TRUE(c1::json_equal(exported(), parse(text)));
    EXPECT_TRUE(texture::material_report().diagnostics.empty());

    const std::string canonical = texture::material_manifest_json(false);
    texture::clear_material_manifests();
    EXPECT_EQ(add(canonical, MaterialLayer::Pack).status, MaterialManifestStatus::Accepted);
    EXPECT_EQ(texture::material_manifest_json(false), canonical);
  }
}

TEST_F(MaterialRegistryTest, LookupsMatchIndex) {
  ASSERT_FALSE(index_array("lookups").empty());
  for (const c1::Json& entry : index_array("lookups")) {
    const std::string query = text_of(entry, "query");
    SCOPED_TRACE(query);
    texture::clear_material_manifests();
    ASSERT_EQ(add(fixture(text_of(entry, "manifest"))).status, MaterialManifestStatus::Accepted);
    const auto snap = material::snapshot();
    const uint16_t id = snap->find(key_of(query));
    const c1::Json* expected = entry.find("resolved");
    ASSERT_NE(expected, nullptr);
    if (expected->type == c1::Json::Type::Null) {
      EXPECT_EQ(id, 0u);
      continue;
    }
    ASSERT_NE(id, 0u);
    ASSERT_NE(snap->key(id), nullptr);
    EXPECT_EQ(c1::canonical_key(*snap->key(id)), expected->text);
    EXPECT_EQ(snap->key_name(id), expected->text);
    EXPECT_TRUE(snap->live(id));
  }
}

TEST_F(MaterialRegistryTest, EmbeddedSchemasMatchFixtures) {
  EXPECT_TRUE(material::schema_ready());
  for (const std::string_view name :
       {std::string_view{"materials.schema.json"}, std::string_view{"materials.layer1.schema.json"}}) {
    SCOPED_TRACE(name);
    const std::string expected = fixture(std::string{name});
    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(material::embedded_schema(name), expected);
  }
  EXPECT_TRUE(material::embedded_schema("index.json").empty());
}

struct Rng {
  uint64_t state = 0;

  uint64_t next() noexcept {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }

  size_t below(size_t count) noexcept { return static_cast<size_t>(next() % count); }

  char pick(std::string_view chars) noexcept { return chars[below(chars.size())]; }

  template <typename List>
  const auto& choose(const List& list) noexcept {
    return list[below(list.size())];
  }

  std::string run(std::string_view chars, size_t lo, size_t hi) {
    std::string out;
    const size_t count = lo + below(hi - lo + 1);
    for (size_t i = 0; i < count; ++i) {
      out += pick(chars);
    }
    return out;
  }
};

void collect_strings(const c1::Json& value, std::set<std::string>& out) {
  if (value.type == c1::Json::Type::String) {
    out.insert(value.text);
  }
  for (const std::string& key : value.keys) {
    out.insert(key);
  }
  for (const c1::Json& item : value.items) {
    collect_strings(item, out);
  }
}

void collect_patterns(const c1::Json& value, std::set<std::string>& out) {
  if (const c1::Json* pattern = value.find("pattern"); pattern != nullptr && pattern->type == c1::Json::Type::String) {
    out.insert(pattern->text);
  }
  for (const c1::Json& item : value.items) {
    collect_patterns(item, out);
  }
}

std::string generated_key(Rng& rng) {
  static const std::array<std::string_view, 15> Dimensions{"0",    "1",     "9",  "16",  "99",    "100", "999", "1000",
                                                           "9999", "10000", "01", "001", "12345", "",    "16"};
  static const std::array<std::string_view, 5> Markers{"", "", "_m", "_M", "_mm"};
  static const std::array<std::string_view, 24> Formats{"0",  "1",  "2",   "3",  "4",  "5",   "6",  "7",
                                                        "8",  "9",  "10",  "11", "14", "65",  "67", "70",
                                                        "78", "79", "100", "08", "",   " 14", "14", "9"};
  static const std::array<std::string_view, 6> Suffixes{"", "", "", "_arb", ".dds", "x"};
  const auto hash = [&rng] {
    switch (rng.below(8)) {
    case 0:
      return rng.run("0123456789abcdef", 15, 15);
    case 1:
      return rng.run("0123456789abcdef", 17, 17);
    case 2:
      return rng.run("0123456789abcdefABCDEF", 16, 16);
    case 3:
      return std::string{"$"};
    default:
      return rng.run("0123456789abcdef", 16, 16);
    }
  };
  std::string out = rng.below(12) == 0 ? "tex2_" : "tex1_";
  out += rng.choose(Dimensions);
  out += rng.below(10) == 0 ? "X" : "x";
  out += rng.choose(Dimensions);
  out += rng.choose(Markers);
  out += '_';
  out += hash();
  switch (rng.below(4)) {
  case 0:
    out += '_';
    out += hash();
    break;
  case 1:
    out += "_$";
    break;
  default:
    break;
  }
  out += '_';
  out += rng.choose(Formats);
  out += rng.choose(Suffixes);
  return out;
}

std::string generated_path(Rng& rng) {
  static const std::array<std::string_view, 9> Extensions{".dds", ".dds", ".DDS",  ".dDs", ".png",
                                                          ".dd",  "dds",  ".ddss", ""};
  static const std::array<std::string_view, 6> Prefixes{"", "", "", "/", "./", "../"};
  std::string out{rng.choose(Prefixes)};
  const size_t segments = 1 + rng.below(3);
  for (size_t i = 0; i < segments; ++i) {
    out += rng.below(10) == 0 ? rng.pick(". :\\") : rng.pick("aZ09_$-");
    out += rng.run(rng.below(10) == 0 ? "aZ09_.$- \\:" : "aZ09_.$-", 0, 6);
    out += i + 1 < segments ? (rng.below(12) == 0 ? "//" : "/") : "";
  }
  out += rng.choose(Extensions);
  return out;
}

std::string generated_source(Rng& rng) {
  static const std::array<std::string_view, 8> Names{"mat", "blend", "texmap", "room", "stage", "Mat", "rooms", ""};
  std::string out;
  const size_t segments = 1 + rng.below(3);
  for (size_t i = 0; i < segments; ++i) {
    out += rng.run(rng.below(10) == 0 ? "aZ09_.- /" : "aZ09_.-", rng.below(10) == 0 ? 0 : 1, 6);
    out += i + 1 < segments ? "/" : "";
  }
  if (rng.below(3) == 0) {
    out += ':';
    out += rng.run(rng.below(8) == 0 ? "aZ09_.-:," : "aZ09_.-", 0, 5);
  }
  const size_t qualifiers = rng.below(4);
  for (size_t i = 0; i < qualifiers; ++i) {
    out += rng.below(10) == 0 ? "  " : " ";
    out += rng.choose(Names);
    out += rng.below(10) == 0 ? "" : ":";
    out += rng.run(rng.below(8) == 0 ? "aZ09_.,-: " : "aZ09_.,-", rng.below(8) == 0 ? 0 : 1, 6);
  }
  return out;
}

std::string generated_class(Rng& rng) {
  std::string out;
  const size_t segments = 1 + rng.below(3);
  for (size_t i = 0; i < segments; ++i) {
    out += rng.pick(rng.below(8) == 0 ? "aZ0_" : "az");
    out += rng.run(rng.below(8) == 0 ? "az09_A-" : "az09_", 0, 6);
    out += i + 1 < segments ? (rng.below(10) == 0 ? "//" : "/") : (rng.below(15) == 0 ? "/" : "");
  }
  return out;
}

std::string mutated(Rng& rng, std::string text) {
  static constexpr std::string_view Edits = "0aAfFgxz9_$.-/:, #m\n\r\x01";
  const size_t pos = text.empty() ? 0 : rng.below(text.size());
  switch (text.empty() ? 2 : rng.below(4)) {
  case 0:
    text.erase(pos, 1);
    break;
  case 1:
    text[pos] = rng.pick(Edits);
    break;
  case 2:
    text.insert(text.begin() + static_cast<ptrdiff_t>(pos), rng.pick(Edits));
    break;
  default:
    text.insert(pos, text.substr(pos, 1));
    break;
  }
  return text;
}

TEST_F(MaterialRegistryTest, PatternMatchersMatchSchemaRegex) {
  std::set<std::string> patterns;
  collect_patterns(parse(fixture("materials.schema.json")), patterns);
  collect_patterns(parse(fixture("materials.layer1.schema.json")), patterns);
  ASSERT_EQ(patterns.size(), 9u);

  std::set<std::string> pool;
  collect_strings(s_index, pool);
  for (const auto& entry : std::filesystem::recursive_directory_iterator{s_root}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      if (auto parsed = c1::parse_json(read_text(entry.path())); parsed.value) {
        collect_strings(*parsed.value, pool);
      }
    }
  }
  const std::vector<std::string> exemplars{"#2f4a1c",
                                           "#ABCDEF",
                                           "#abcde",
                                           "#abcdeg",
                                           "lawn",
                                           "a",
                                           "a_b9",
                                           std::string(32, 'a'),
                                           std::string(33, 'a'),
                                           "tex1_16x16_$_14",
                                           "tex1_16x16_m_$_9",
                                           "tex1_0x0_$",
                                           "tex1_1x1_$_$_9",
                                           "tex1_16x16_$x",
                                           "tex1_16x16_m_$",
                                           std::string{KeyA},
                                           std::string{KeyPalette},
                                           "tex1_16x16_000000000000d001_000000000000d0f1_9",
                                           "tex1_9999x1_0123456789abcdef_78",
                                           "tex1_16x16_000000000000a001_000000000000d0f1_14",
                                           "tex1_16x16_000000000000d001_9",
                                           "tex1_16x16_000000000000d001_$_14",
                                           "a.dds",
                                           "maps/ground/field01.nrm.dds",
                                           "x/y.DdS",
                                           "$-_.x/a.b.c.dds",
                                           "../a.dds",
                                           "/a.dds",
                                           "a//b.dds",
                                           "a/.dds",
                                           ".dds",
                                           "ground/grass",
                                           "a/b_c9/d",
                                           "a/",
                                           std::string(64, '0'),
                                           std::string(64, 'f'),
                                           std::string(63, 'a'),
                                           "stage01/room00/model.bdl:field01",
                                           "dusk/d_a:tex stage:F_SP121 room:0,1 texmap:0",
                                           "a b",
                                           "a:b:c",
                                           "a mat:x",
                                           "a mat:",
                                           "a  mat:x",
                                           "a mat:x ",
                                           "a/b:c blend:ZERO,SRCCLR",
                                           "caf\xc3\xa9",
                                           std::string{"a\0b", 3},
                                           "",
                                           " ",
                                           "\n",
                                           "#000000\n",
                                           "lawn\r"};
  Rng rng{0x6d617470};
  for (const std::string& text : exemplars) {
    pool.insert(text);
    for (int i = 0; i < 12; ++i) {
      pool.insert(mutated(rng, text));
    }
  }
  for (int i = 0; i < 700; ++i) {
    const std::array<std::string, 7> generated{
        "#" + rng.run("0123456789abcdefABCDEFgG#", 5, 7),
        rng.run("abcz09_A-", 0, 34),
        generated_key(rng),
        generated_path(rng),
        generated_class(rng),
        rng.below(3) == 0 ? rng.run("0123456789abcdefA", 62, 66) : rng.run("0123456789abcdef", 64, 64),
        generated_source(rng),
    };
    for (const std::string& text : generated) {
      pool.insert(text);
      pool.insert(mutated(rng, text));
    }
    pool.insert(rng.run("0123456789abcdefABCDEFgxzm_$.-/:, #\n\r\x01", 0, 40));
  }
  for (const std::string_view dimension : {"0", "1", "9", "10", "99", "100", "999", "1000", "9999", "10000", "01"}) {
    pool.insert(fmt::format("tex1_{}x16_000000000000a001_14", dimension));
    pool.insert(fmt::format("tex1_16x{}_000000000000d001_$_9", dimension));
    pool.insert(fmt::format("tex1_{}x{}_$_14", dimension, dimension));
  }
  for (int format = 0; format <= 80; ++format) {
    pool.insert(fmt::format("tex1_16x16_000000000000a001_{}", format));
    pool.insert(fmt::format("tex1_16x16_000000000000a001_000000000000d0f1_{}", format));
    pool.insert(fmt::format("tex1_16x16_000000000000a001_$_{}", format));
  }

  const auto start = std::chrono::steady_clock::now();
  for (const std::string& source : patterns) {
    SCOPED_TRACE(source);
    const std::regex regex{source, std::regex::ECMAScript};
    size_t matched = 0;
    for (const std::string& text : pool) {
      std::string mapped = text;
      std::replace(mapped.begin(), mapped.end(), '\n', '\x01');
      std::replace(mapped.begin(), mapped.end(), '\r', '\x01');
      const bool expected = std::regex_search(mapped, regex);
      bool bound = false;
      EXPECT_EQ(material::testing::match_schema_pattern(source, text, &bound), expected) << '"' << text << '"';
      EXPECT_TRUE(bound);
      matched += expected ? 1 : 0;
    }
    EXPECT_GE(matched, 10u);
  }
  const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  fmt::print("pattern parity: {} strings x {} patterns in {:.0f} ms\n", pool.size(), patterns.size(), ms);
  bool bound = true;
  EXPECT_FALSE(material::testing::match_schema_pattern("^a$", "a", &bound));
  EXPECT_FALSE(bound);
}

void collect_paths(const c1::Json& value, std::vector<size_t>& path, std::vector<std::vector<size_t>>& out) {
  for (size_t i = 0; i < value.items.size(); ++i) {
    path.push_back(i);
    out.push_back(path);
    collect_paths(value.items[i], path, out);
    path.pop_back();
  }
}

template <typename Json>
Json& node_at(Json& root, std::span<const size_t> path) {
  Json* node = &root;
  for (const size_t index : path) {
    node = &node->items[index];
  }
  return *node;
}

TEST_F(MaterialRegistryTest, MutationsMatchOracle) {
  const std::vector<std::string> files{
      "manifests/valid/all_fields.json", "manifests/valid/layer1.json",      "manifests/valid/tlut_wildcard.json",
      "manifests/valid/bounds.json",     "manifests/valid/enum_values.json", "merge/layers/expected.json",
      "merge/post_merge/user.json",
  };
  std::vector<c1::Json> replacements;
  for (const std::string_view text :
       {"null", "true", "false", "0", "-1", "0.5", "1", "2", "17", "1000000000", R"("")", R"("x")",
        R"("tex1_16x16_000000000000a001_14")", R"("#12345g")", "[]", "[1]", R"(["a","a"])", "{}", R"({"zz":1})"}) {
    replacements.push_back(parse(text));
  }
  c1::Json extra = parse("1");
  size_t compared = 0;
  const auto compare = [&](const c1::Json& manifest) {
    const std::string text = dump_json(manifest);
    for (const int layer : {1, 2}) {
      const auto oracle = c1::check_manifest(s_schemas, text, layer);
      const auto check = material::check_manifest(text, static_cast<MaterialLayer>(layer));
      ++compared;
      if (oracle.empty()) {
        EXPECT_FALSE(check.violation.has_value())
            << text << " at " << layer << ": " << check.violation->rule << " " << check.violation->pointer;
        EXPECT_NE(check.doc, nullptr);
        continue;
      }
      ASSERT_TRUE(check.violation.has_value()) << text << " at " << layer;
      EXPECT_EQ(check.violation->rule, oracle.front().rule) << text << " at " << layer;
      EXPECT_EQ(check.violation->pointer, oracle.front().pointer) << text << " at " << layer;
    }
  };
  for (const std::string& file : files) {
    SCOPED_TRACE(file);
    const c1::Json original = parse(fixture(file));
    compare(original);
    std::vector<std::vector<size_t>> paths;
    std::vector<size_t> path;
    collect_paths(original, path, paths);
    c1::Json widenedRoot = original;
    widenedRoot.keys.emplace_back("zz");
    widenedRoot.items.push_back(extra);
    compare(widenedRoot);
    for (const auto& target : paths) {
      const std::span<const size_t> parentPath{target.data(), target.size() - 1};
      const size_t index = target.back();
      for (const c1::Json& replacement : replacements) {
        c1::Json copy = original;
        node_at(copy, target) = replacement;
        compare(copy);
      }
      c1::Json removed = original;
      c1::Json& parent = node_at(removed, parentPath);
      parent.items.erase(parent.items.begin() + static_cast<ptrdiff_t>(index));
      if (parent.type == c1::Json::Type::Object) {
        parent.keys.erase(parent.keys.begin() + static_cast<ptrdiff_t>(index));
        compare(removed);
        c1::Json renamed = original;
        node_at(renamed, parentPath).keys[index] += "X";
        compare(renamed);
      } else {
        compare(removed);
        c1::Json doubled = original;
        c1::Json& items = node_at(doubled, parentPath);
        items.items.insert(items.items.begin() + static_cast<ptrdiff_t>(index), items.items[index]);
        compare(doubled);
      }
      if (node_at(original, target).type == c1::Json::Type::Object) {
        c1::Json widened = original;
        c1::Json& object = node_at(widened, target);
        object.keys.emplace_back("zz");
        object.items.push_back(extra);
        compare(widened);
      }
    }
  }
  EXPECT_GT(compared, 4000u);
}

TEST_F(MaterialRegistryTest, JsonStrictnessMatchesOracle) {
  const std::string material = std::string{R"({"version":1,"materials":{")"} + std::string{KeyA} + R"(":)";
  const auto nested = [](int depth) {
    return R"({"version":1,"x":)" + std::string(static_cast<size_t>(depth - 1), '[') +
           std::string(static_cast<size_t>(depth - 1), ']') + "}";
  };
  const std::vector<std::string> texts{
      R"({"version":1})",
      R"({"version":1.0})",
      R"({"version":1.5})",
      R"({"version":true})",
      "\xEF\xBB\xBF{\"version\":1}",
      R"({"version":NaN})",
      R"({"version":1e400})",
      R"({"version":1,"x":-1e400})",
      material + R"({"roughness":1e-400}}})",
      material + R"({"roughness":123456789012345678901234567890}}})",
      material + R"({"roughness":-0}}})",
      nested(64),
      nested(65),
      R"({"version":1,"version":2,"x":)" + std::string(70, '[') + std::string(70, ']') + "}",
      R"({"version":1,"materials":{"k":{"a":1,"a":2}}})",
      R"({"version":1,"version":1})",
      R"({"version":1,"version":1})",
      R"({"version":1,"x":[{"a":1,"a":2}]})",
      R"({"version":1,"a/b~c":1})",
      R"({"version":1,"a\u0000b":1})",
      R"({"version":1,})",
      R"({"version":1 /* c */})",
      R"({'version':1})",
      R"({"version":01})",
      "{\"version\":1,\"x\":\"a\tb\"}",
      "{\"version\":1,\"x\":\"\xC3\x28\"}",
      "{\"version\":1,\"x\":\"\xC0\xAF\"}",
      "{\"version\":1,\"x\":\"\xED\xA0\x80\"}",
      R"({"version":1,"x":"\ud800"})",
      R"({"version":1,"x":"\udc00\ud800"})",
      material + R"({"_source":"😀"}}})",
      "",
      "  ",
      "null",
      "[]",
      "\f{\"version\":1}",
      R"({"version":1} x)",
      R"({"version":1}  )",
  };
  for (const std::string& text : texts) {
    SCOPED_TRACE(text);
    const auto oracle = c1::check_manifest(s_schemas, text, 2);
    const auto check = material::check_manifest(text, MaterialLayer::Pack);
    if (oracle.empty()) {
      EXPECT_FALSE(check.violation.has_value()) << check.violation->rule << " " << check.violation->pointer;
      continue;
    }
    ASSERT_TRUE(check.violation.has_value());
    EXPECT_EQ(check.violation->rule, oracle.front().rule);
    EXPECT_EQ(check.violation->pointer, oracle.front().pointer);
  }
  EXPECT_FALSE(material::check_manifest(R"({"version":1.0})", MaterialLayer::Pack).violation.has_value());
  const auto fraction = material::check_manifest(R"({"version":1.5})", MaterialLayer::Pack);
  ASSERT_TRUE(fraction.violation.has_value());
  EXPECT_EQ(fraction.violation->rule, "MANIFEST-VERSION");
  EXPECT_EQ(fraction.violation->pointer, "/version");
  const auto deep = material::check_manifest(nested(65), MaterialLayer::Pack);
  ASSERT_TRUE(deep.violation.has_value());
  EXPECT_EQ(deep.violation->rule, "JSON-SYNTAX");
  const auto shallow = material::check_manifest(nested(64), MaterialLayer::Pack);
  ASSERT_TRUE(shallow.violation.has_value());
  EXPECT_EQ(shallow.violation->rule, "MANIFEST-UNKNOWN-MEMBER");
  const auto bom = material::check_manifest("\xEF\xBB\xBF{\"version\":1}", MaterialLayer::Pack);
  ASSERT_TRUE(bom.violation.has_value());
  EXPECT_EQ(bom.violation->rule, "JSON-SYNTAX");
}

TEST_F(MaterialRegistryTest, RecordsDecodeHotFields) {
  const std::string text = manifest_with(
      R"("tex1_16x16_000000000000a001_14":{"flags":["stochastic","cutout"],"class":"ground/grass",)"
      R"("vertexColor":"bakedLight","stochastic":{"contrast":0.25,"scale":4},)"
      R"("fakeShadow":{"neutral":"clear","gate":"ao","scope":["TERRAIN","DECAL"]},"grass":{"type":"lawn"},)"
      R"("displacement":{"amplitude":2},"maps":{"nrm":"a.nrm.dds","orm":"a.orm.dds"}},)"
      R"("tex1_16x16_000000000000a002_14":{"flags":["noPbr","unlit"],"class":"rock","vertexColor":"ignore",)"
      R"("stochastic":{"rotation":0.5,"exponent":9},"fakeShadow":{"neutral":"white","gate":"light"}},)"
      R"("tex1_16x16_000000000000d001_$_9":{"vertexColor":"tint"})",
      R"("lawn":{"height":[0.1,0.2],"width":[0.01,0.02],"densityPerM2":100,"rootColor":"#000000","tipColor":"#ffffff"})");
  ASSERT_EQ(add(text, MaterialLayer::Pack, "records", 0, "packbase").status, MaterialManifestStatus::Accepted);
  const auto snap = material::snapshot();
  EXPECT_EQ(snap->liveCount, 3u);
  EXPECT_EQ(snap->materialCount, 3u);
  EXPECT_EQ(snap->grassTypeCount, 1u);

  const uint16_t first = snap->find(key_of(KeyA));
  ASSERT_NE(first, 0u);
  const material::MaterialRecord& a = snap->record(first);
  EXPECT_EQ(a.id, first);
  EXPECT_EQ(a.summary, material::Stochastic | material::Cutout | material::FakeShadow | material::Grass |
                           material::Displacement | material::ShadowScope | material::StochasticParams |
                           material::Class | material::map_role_bit(2) | material::map_role_bit(3) |
                           2u << material::VertexColorShift | 2u << material::NeutralShift | 3u << material::GateShift);
  EXPECT_EQ(a.shadowScope, (1u << 3) | (1u << 8));
  EXPECT_EQ(a.stochasticSet, static_cast<uint8_t>(material::StochasticContrast | material::StochasticScale));
  EXPECT_EQ(a.contrast, 0.25f);
  EXPECT_EQ(a.scale, 4.f);
  EXPECT_EQ(a.rotation, 0.f);
  EXPECT_EQ(a.exponent, 0.f);
  EXPECT_TRUE(material::has_map_role(a.summary, 2));
  EXPECT_FALSE(material::has_map_role(a.summary, 0));
  EXPECT_EQ(material::vertex_color(a.summary), 2u);
  EXPECT_EQ(material::shadow_neutral(a.summary), 2u);
  EXPECT_EQ(material::shadow_gate(a.summary), 3u);
  ASSERT_EQ(snap->classes.size(), 3u);
  EXPECT_EQ(snap->classes[a.classId], "ground/grass");
  EXPECT_EQ(snap->map_base(first, 2), "packbase");
  EXPECT_EQ(snap->map_base(first, 3), "packbase");
  EXPECT_EQ(snap->map_base(first, 0), "");
  EXPECT_EQ(snap->mapSupplier[first][2], 1u);
  ASSERT_NE(snap->entries[first], nullptr);
  EXPECT_EQ(yyjson_obj_size(snap->entries[first]), 8u);

  const uint16_t second = snap->find(key_of(KeyB));
  ASSERT_NE(second, 0u);
  const material::MaterialRecord& b = snap->record(second);
  EXPECT_EQ(b.summary, material::NoPbr | material::Unlit | material::FakeShadow | material::StochasticParams |
                           material::Class | 3u << material::VertexColorShift | 1u << material::NeutralShift |
                           1u << material::GateShift);
  EXPECT_EQ(b.shadowScope, 0u);
  EXPECT_EQ(b.stochasticSet, static_cast<uint8_t>(material::StochasticRotation | material::StochasticExponent));
  EXPECT_EQ(b.rotation, 0.5f);
  EXPECT_EQ(b.exponent, 9.f);
  EXPECT_EQ(snap->classes[b.classId], "rock");

  texture::TextureSourceKey paletteKey = key_of(KeyPalette);
  paletteKey.tlutHash = 0x1234;
  const uint16_t third = snap->find(paletteKey);
  ASSERT_NE(third, 0u);
  EXPECT_EQ(snap->key_name(third), KeyPalette);
  EXPECT_EQ(snap->record(third).summary, 1u << material::VertexColorShift);
  texture::TextureSourceKey wildcard = key_of(KeyA);
  wildcard.textureHash = texture::kWildcardTextureHash;
  EXPECT_EQ(snap->find(wildcard), 0u);

  EXPECT_EQ(snap->summaryCount[0], 1u);
  EXPECT_EQ(snap->summaryCount[8], 2u);
  EXPECT_EQ(snap->summaryCount[24], 2u);
  EXPECT_EQ(snap->summaryCount[25], 2u);
  EXPECT_EQ(snap->record(0).summary, 0u);
  EXPECT_EQ(snap->record(70000).id, 0u);
  EXPECT_EQ(snap->key(0), nullptr);
  EXPECT_EQ(snap->key(70000), nullptr);
  EXPECT_FALSE(snap->live(0));
  EXPECT_EQ(snap->map_base(70000, 2), "");
}

TEST_F(MaterialRegistryTest, OriginsFollowMergeOrder) {
  const std::string key{KeyA};
  add(manifest_with(fmt::format(R"("{}":{{"roughness":0.2,"class":"a","stochastic":{{"scale":2,"contrast":0.1}},)"
                                R"("maps":{{"nrm":"low.nrm.dds"}}}})",
                                key)),
      MaterialLayer::Pack, "low", 0);
  add(manifest_with(fmt::format(R"("{}":{{"stochastic":{{"contrast":0.9}},"maps":{{"orm":"high.orm.dds"}}}})", key)),
      MaterialLayer::Pack, "high", 5);
  add(manifest_with(fmt::format(R"("{}":{{"roughness":0.7,"fakeShadow":{{"neutral":"white"}}}})", key)),
      MaterialLayer::User, "user");

  const c1::Json found = parse(texture::find_material_json(key_of(key)));
  EXPECT_EQ(text_of(found, "key"), key);
  EXPECT_EQ(int_of(found, "id"), find_id(key));
  const c1::Json all = exported();
  const c1::Json* merged = all.find("materials")->find(key);
  ASSERT_NE(merged, nullptr);
  ASSERT_NE(found.find("material"), nullptr);
  EXPECT_TRUE(c1::json_equal(*found.find("material"), *merged));
  const c1::Json* origins = found.find("origins");
  ASSERT_NE(origins, nullptr);
  const std::map<std::string, std::string> expected{
      {"/class", "low"},
      {"/maps/nrm", "low"},
      {"/maps/orm", "high"},
      {"/roughness", "user"},
      {"/stochastic/contrast", "high"},
      {"/stochastic/scale", "low"},
  };
  std::map<std::string, std::string> actual;
  for (size_t i = 0; i < origins->keys.size(); ++i) {
    actual.emplace(origins->keys[i], origins->items[i].text);
  }
  EXPECT_EQ(actual, expected);
  EXPECT_TRUE(c1::json_equal(parse(texture::find_material_json(key_of(key), true)), found));
  EXPECT_EQ(texture::find_material_json(key_of(KeyB)), "");
}

TEST_F(MaterialRegistryTest, SupplierOnlyChangesAreTracked) {
  const std::string maps = manifest_with(fmt::format(R"("{}":{{"maps":{{"nrm":"n.nrm.dds"}}}})", KeyA));
  add(maps, MaterialLayer::Pack, "a", 0, "baseA");
  const auto high = add(maps, MaterialLayer::Pack, "b", 1, "baseB");
  const uint16_t id = find_id(KeyA);
  EXPECT_EQ(material::snapshot()->map_base(id, 2), "baseB");
  material::set_publish_observer(&observe);
  EXPECT_EQ(g_observed.load(), 1);
  const std::string before = texture::material_manifest_json(false);
  const uint64_t generation = material::generation();
  const uint64_t bindEpoch = material::snapshot()->bindEpoch;
  texture::unregister_material_manifest(high);
  EXPECT_EQ(texture::material_manifest_json(false), before);
  EXPECT_EQ(material::generation(), generation + 1);
  EXPECT_EQ(material::snapshot()->bindEpoch, bindEpoch);
  EXPECT_EQ(g_observed.load(), 2);
  EXPECT_EQ(material::snapshot()->map_base(id, 2), "baseA");

  const auto roughness = [](std::string_view key) {
    const c1::Json found = parse(texture::find_material_json(key_of(key)));
    const c1::Json* origins = found.find("origins");
    return origins != nullptr ? text_of(*origins, "/roughness") : std::string{};
  };
  const std::string shared = manifest_with(fmt::format(R"("{}":{{"roughness":0.5}})", KeyB));
  add(shared, MaterialLayer::Pack, "c", 0);
  const auto top = add(shared, MaterialLayer::Pack, "d", 1);
  EXPECT_EQ(roughness(KeyB), "d");
  const uint64_t sameGeneration = material::generation();
  const int observed = g_observed.load();
  EXPECT_EQ(texture::update_material_manifest(top, manifest_with(fmt::format(R"("{}":{{}})", KeyB))),
            MaterialManifestStatus::Accepted);
  EXPECT_EQ(material::generation(), sameGeneration);
  EXPECT_EQ(g_observed.load(), observed + 1);
  EXPECT_EQ(roughness(KeyB), "c");

  const std::string twin = manifest_with(R"("tex1_16x16_000000000000a003_14":{"roughness":0.3})");
  const auto first = add(twin, MaterialLayer::Pack, "e", 2);
  add(twin, MaterialLayer::Pack, "f", 2);
  const uint64_t reorderGeneration = material::generation();
  const uint64_t reorderEpoch = material::snapshot()->bindEpoch;
  const int reorderObserved = g_observed.load();
  texture::unregister_material_manifest(first);
  add(twin, MaterialLayer::Pack, "e", 2);
  EXPECT_EQ(material::generation(), reorderGeneration);
  EXPECT_EQ(material::snapshot()->bindEpoch, reorderEpoch);
  EXPECT_EQ(g_observed.load(), reorderObserved + 2);
  ASSERT_EQ(material::snapshot()->contributors.size(), 5u);
  EXPECT_EQ(material::snapshot()->contributors.back().label, "e");
}

TEST_F(MaterialRegistryTest, IdsAndGenerationAreMonotonic) {
  constexpr std::string_view First = "tex1_16x16_00000000000000a1_14";
  constexpr std::string_view Second = "tex1_16x16_00000000000000a2_14";
  constexpr std::string_view Third = "tex1_16x16_00000000000000a3_14";
  std::vector<uint64_t> generations{material::generation()};
  const auto record_generation = [&] { generations.push_back(material::generation()); };

  const auto one = add(manifest_with(fmt::format(R"("{}":{{}},"{}":{{}})", Third, First)));
  record_generation();
  const uint16_t firstId = find_id(First);
  const uint16_t thirdId = find_id(Third);
  ASSERT_NE(firstId, 0u);
  EXPECT_LT(firstId, thirdId);

  texture::unregister_material_manifest(one);
  record_generation();
  EXPECT_EQ(find_id(First), 0u);
  EXPECT_FALSE(material::snapshot()->live(firstId));
  EXPECT_EQ(material::snapshot()->key_name(firstId), First);

  const auto two = add(manifest_with(fmt::format(R"("{}":{{}},"{}":{{}})", Second, First)));
  record_generation();
  EXPECT_GT(two.id, one.id);
  EXPECT_EQ(find_id(First), firstId);
  EXPECT_GT(find_id(Second), thirdId);

  texture::clear_material_manifests();
  record_generation();
  const auto three = add(manifest_with(fmt::format(R"("{}":{{}})", First)));
  record_generation();
  EXPECT_GT(three.id, two.id);
  EXPECT_EQ(find_id(First), firstId);
  EXPECT_TRUE(std::is_sorted(generations.begin(), generations.end()));
  EXPECT_EQ(std::set<uint64_t>(generations.begin(), generations.end()).size(), generations.size());
}

TEST_F(MaterialRegistryTest, StatusReloadAndOverride) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      fmt::format("aurora_material_registry_{}", std::chrono::steady_clock::now().time_since_epoch().count());
  ASSERT_TRUE(std::filesystem::create_directories(dir));
  const std::filesystem::path path = dir / "materials.json";
  const auto write = [&](std::string_view text) {
    ASSERT_TRUE(io::write_file(path, {reinterpret_cast<const uint8_t*>(text.data()), text.size()}));
  };

  const auto registration =
      texture::load_material_manifest(path, {.layer = MaterialLayer::User, .priority = 3, .writable = true});
  EXPECT_NE(registration.id, 0u);
  EXPECT_EQ(registration.status, MaterialManifestStatus::Missing);
  auto info = texture::material_manifest_info(registration);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->label, io::fs_path_to_string(path));
  EXPECT_TRUE(info->path == path);
  EXPECT_EQ(info->base, io::fs_path_to_string(dir));
  EXPECT_EQ(info->layer, MaterialLayer::User);
  EXPECT_EQ(info->priority, 3);
  EXPECT_TRUE(info->writable);
  EXPECT_TRUE(info->layerEnabled);
  EXPECT_FALSE(info->overridden);

  write(manifest_with(fmt::format(R"("{}":{{"roughness":0.1}})", KeyA)));
  EXPECT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Accepted);
  EXPECT_NE(find_id(KeyA), 0u);

  EXPECT_EQ(texture::update_material_manifest(registration, manifest_with(fmt::format(R"("{}":{{}})", KeyB))),
            MaterialManifestStatus::Accepted);
  EXPECT_TRUE(texture::material_manifest_info(registration)->overridden);
  EXPECT_EQ(find_id(KeyA), 0u);
  EXPECT_NE(find_id(KeyB), 0u);

  EXPECT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Accepted);
  EXPECT_FALSE(texture::material_manifest_info(registration)->overridden);
  EXPECT_NE(find_id(KeyA), 0u);
  EXPECT_EQ(find_id(KeyB), 0u);

  write(R"({"version":2})");
  EXPECT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Rejected);
  info = texture::material_manifest_info(registration);
  ASSERT_TRUE(info->violation.has_value());
  EXPECT_EQ(info->violation->rule, "MANIFEST-VERSION");
  EXPECT_EQ(info->violation->pointer, "/version");
  EXPECT_EQ(find_id(KeyA), 0u);
  const auto diagnostics = texture::material_report().diagnostics;
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].label, io::fs_path_to_string(path));

  std::filesystem::remove(path);
  EXPECT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Missing);
  const auto directory = texture::load_material_manifest(dir);
  EXPECT_NE(directory.id, 0u);
  EXPECT_EQ(directory.status, MaterialManifestStatus::Unreadable);

  const std::string huge(size_t{16} << 20 | 1, ' ');
  const auto oversized = add(huge);
  EXPECT_EQ(oversized.id, 0u);
  EXPECT_EQ(oversized.status, MaterialManifestStatus::Unreadable);
  EXPECT_EQ(texture::update_material_manifest(registration, huge), MaterialManifestStatus::Unreadable);
  EXPECT_EQ(add(R"({"version":1})", static_cast<MaterialLayer>(7)).id, 0u);
  EXPECT_FALSE(texture::check_material_manifest(R"({"version":1})", MaterialLayer::Game).has_value());
  EXPECT_TRUE(texture::check_material_manifest(R"({"version":1})", static_cast<MaterialLayer>(0)).has_value());

  const auto text = add(manifest_with(fmt::format(R"("{}":{{}})", KeyB)));
  EXPECT_EQ(texture::material_manifest_info(text)->label, fmt::format("manifest#{}", text.id));
  EXPECT_EQ(texture::update_material_manifest(text, R"({"version":1})"), MaterialManifestStatus::Accepted);
  EXPECT_EQ(find_id(KeyB), 0u);
  EXPECT_EQ(texture::reload_material_manifest(text), MaterialManifestStatus::Accepted);
  EXPECT_NE(find_id(KeyB), 0u);

  texture::unregister_material_manifest(registration);
  EXPECT_FALSE(texture::material_manifest_info(registration).has_value());
  EXPECT_EQ(texture::reload_material_manifest(registration), MaterialManifestStatus::Superseded);
  EXPECT_EQ(texture::update_material_manifest({}, R"({"version":1})"), MaterialManifestStatus::Superseded);
  std::error_code error;
  std::filesystem::remove_all(dir, error);
}

TEST_F(MaterialRegistryTest, ConcurrentEditsNewestWins) {
  const auto registration = add(manifest_with(fmt::format(R"("{}":{{"roughness":0.1}})", KeyA)));
  const std::string older = manifest_with(fmt::format(R"("{}":{{"roughness":0.2}})", KeyA));
  const std::string newer = manifest_with(fmt::format(R"("{}":{{"roughness":0.3}})", KeyA));
  const auto roughness = [] {
    const c1::Json all = exported();
    const c1::Json* entry = all.find("materials")->find(KeyA);
    return entry != nullptr && entry->find("roughness") != nullptr ? entry->find("roughness")->number : -1.0;
  };

  for (const bool newerFinishesFirst : {true, false}) {
    SCOPED_TRACE(newerFinishesFirst ? "newer finishes first" : "older finishes first");
    ASSERT_EQ(texture::update_material_manifest(registration,
                                                manifest_with(fmt::format(R"("{}":{{"roughness":0.1}})", KeyA))),
              MaterialManifestStatus::Accepted);
    HookGate gate;
    g_gate = &gate;
    material::testing::set_check_hook(&gated_check);
    MaterialManifestStatus olderStatus = MaterialManifestStatus::Unreadable;
    MaterialManifestStatus newerStatus = MaterialManifestStatus::Unreadable;
    std::thread first{[&] { olderStatus = texture::update_material_manifest(registration, older); }};
    ASSERT_TRUE(gate.wait_arrivals(1));
    if (newerFinishesFirst) {
      gate.release(1);
      newerStatus = texture::update_material_manifest(registration, newer);
      EXPECT_EQ(roughness(), 0.3);
      gate.release(0);
      first.join();
    } else {
      std::thread second{[&] { newerStatus = texture::update_material_manifest(registration, newer); }};
      ASSERT_TRUE(gate.wait_arrivals(2));
      gate.release(0);
      first.join();
      EXPECT_EQ(roughness(), 0.1);
      gate.release(1);
      second.join();
    }
    EXPECT_EQ(olderStatus, MaterialManifestStatus::Superseded);
    EXPECT_EQ(newerStatus, MaterialManifestStatus::Accepted);
    EXPECT_EQ(roughness(), 0.3);
    material::testing::set_check_hook(nullptr);
  }

  HookGate gate;
  g_gate = &gate;
  material::testing::set_check_hook(&gated_check);
  MaterialManifestStatus status = MaterialManifestStatus::Unreadable;
  std::thread editor{[&] { status = texture::update_material_manifest(registration, older); }};
  ASSERT_TRUE(gate.wait_arrivals(1));
  texture::unregister_material_manifest(registration);
  gate.release(0);
  editor.join();
  EXPECT_EQ(status, MaterialManifestStatus::Superseded);
  EXPECT_FALSE(texture::material_manifest_info(registration).has_value());
  EXPECT_EQ(find_id(KeyA), 0u);
  material::testing::set_check_hook(nullptr);
  g_gate = nullptr;
}

TEST_F(MaterialRegistryTest, LayerSwitchesAndGeneration) {
  const auto pack = add(manifest_with(R"("tex1_16x16_00000000000000b1_14":{})"), MaterialLayer::Pack, "pack");
  add(manifest_with(R"("tex1_16x16_00000000000000b2_14":{})"), MaterialLayer::Mod, "mod");
  add(manifest_with(R"("tex1_16x16_00000000000000b3_14":{})"), MaterialLayer::User, "user");
  add(manifest_with(R"("tex1_16x16_00000000000000b4_14":{"flags":["stochastic"]})"), MaterialLayer::Game, "game");
  add(R"({"version":2})", MaterialLayer::Pack, "bad");
  const std::array<uint16_t, 4> ids{
      find_id("tex1_16x16_00000000000000b1_14"), find_id("tex1_16x16_00000000000000b2_14"),
      find_id("tex1_16x16_00000000000000b3_14"), find_id("tex1_16x16_00000000000000b4_14")};
  for (const uint16_t id : ids) {
    EXPECT_NE(id, 0u);
  }
  const auto has_bad = [] {
    const auto diagnostics = texture::material_report().diagnostics;
    return std::any_of(diagnostics.begin(), diagnostics.end(),
                       [](const texture::MaterialDiagnostic& diagnostic) { return diagnostic.label == "bad"; });
  };
  EXPECT_TRUE(has_bad());

  uint64_t generation = material::generation();
  texture::set_material_layer_enabled(MaterialLayer::Pack, false);
  EXPECT_EQ(material::generation(), generation + 1);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b1_14"), 0u);
  EXPECT_FALSE(texture::material_manifest_info(pack)->layerEnabled);
  EXPECT_FALSE(has_bad());
  texture::set_material_layer_enabled(MaterialLayer::Mod, false);
  EXPECT_EQ(material::generation(), generation + 2);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b2_14"), 0u);

  generation = material::generation();
  texture::set_material_layer_enabled(MaterialLayer::Pack, false);
  texture::set_material_layer_enabled(MaterialLayer::Game, false);
  texture::set_material_layer_enabled(MaterialLayer::User, false);
  EXPECT_EQ(material::generation(), generation);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b3_14"), ids[2]);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b4_14"), ids[3]);

  texture::set_material_layer_enabled(MaterialLayer::Pack, true);
  texture::set_material_layer_enabled(MaterialLayer::Mod, true);
  EXPECT_EQ(material::generation(), generation + 2);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b1_14"), ids[0]);
  EXPECT_EQ(find_id("tex1_16x16_00000000000000b2_14"), ids[1]);
  EXPECT_TRUE(has_bad());
}

TEST_F(MaterialRegistryTest, ObserverAndBindEpoch) {
  material::set_publish_observer(&observe);
  EXPECT_EQ(g_observed.load(), 1);
  const auto registration = add(manifest_with(fmt::format(R"("{}":{{"roughness":0.5}})", KeyA)));
  EXPECT_EQ(g_observed.load(), 2);
  EXPECT_EQ(g_observedLive.load(), 1u);
  uint64_t generation = g_observedGeneration.load();
  uint64_t bindEpoch = g_observedBindEpoch.load();
  EXPECT_EQ(generation, material::generation());

  int observed = g_observed.load();
  for (const std::string_view fields :
       {std::string_view{R"("roughness":0.6)"}, std::string_view{R"("normalStrength":2)"},
        std::string_view{R"("_source":"edited")"}}) {
    SCOPED_TRACE(fields);
    texture::update_material_manifest(registration, manifest_with(fmt::format(R"("{}":{{{}}})", KeyA, fields)));
    EXPECT_EQ(g_observed.load(), ++observed);
    EXPECT_EQ(g_observedGeneration.load(), ++generation);
    EXPECT_EQ(g_observedBindEpoch.load(), bindEpoch);
  }

  texture::update_material_manifest(registration,
                                    manifest_with(fmt::format(R"("{}":{{"flags":["stochastic"]}})", KeyA)));
  EXPECT_EQ(g_observed.load(), ++observed);
  EXPECT_EQ(g_observedBindEpoch.load(), ++bindEpoch);
  texture::update_material_manifest(
      registration, manifest_with(fmt::format(R"("{}":{{"flags":["stochastic"]}},"{}":{{}})", KeyA, KeyB)));
  EXPECT_EQ(g_observed.load(), ++observed);
  EXPECT_EQ(g_observedBindEpoch.load(), ++bindEpoch);
  EXPECT_EQ(g_observedLive.load(), 2u);

  add(R"({"version":2})");
  texture::set_material_layer_enabled(MaterialLayer::Mod, false);
  add(manifest_with(R"("tex1_16x16_00000000000000c1_14":{})"), MaterialLayer::Mod);
  texture::unregister_material_manifest({});
  EXPECT_EQ(g_observed.load(), observed);

  texture::clear_material_manifests();
  EXPECT_EQ(g_observed.load(), ++observed);
  EXPECT_EQ(g_observedLive.load(), 0u);
  EXPECT_EQ(g_observedBindEpoch.load(), ++bindEpoch);
  EXPECT_EQ(material::snapshot()->liveCount, 0u);
  EXPECT_EQ(texture::material_manifest_json(false), R"({"grassTypes":{},"materials":{},"version":1})");
}

TEST_F(MaterialRegistryTest, LatchSeparatesLatestFromLatched) {
  material::latch();
  EXPECT_EQ(material::latched_raw(), material::latched_snapshot().get());
  EXPECT_EQ(material::latched_snapshot(), material::snapshot());

  const auto registration = add(manifest_with(fmt::format(R"("{}":{{"roughness":0.5}})", KeyA)));
  EXPECT_GT(material::snapshot()->generation, material::latched_snapshot()->generation);
  EXPECT_EQ(material::latched_raw()->find(key_of(KeyA)), 0u);
  EXPECT_TRUE(material::latch());
  EXPECT_EQ(material::latched_snapshot(), material::snapshot());
  EXPECT_EQ(material::latched_raw(), material::snapshot().get());
  EXPECT_NE(material::latched_raw()->find(key_of(KeyA)), 0u);

  texture::update_material_manifest(registration, manifest_with(fmt::format(R"("{}":{{"roughness":0.6}})", KeyA)));
  EXPECT_EQ(material::snapshot()->generation, material::latched_snapshot()->generation + 1);
  EXPECT_FALSE(material::latch());
  EXPECT_EQ(material::latched_snapshot()->generation, material::snapshot()->generation);
  EXPECT_FALSE(material::latch());

  texture::update_material_manifest(registration, manifest_with(fmt::format(R"("{}":{{"flags":["unlit"]}})", KeyA)));
  EXPECT_TRUE(material::latch());
}

TEST_F(MaterialRegistryTest, MaterialKeyTextMatchesOracle) {
  ASSERT_FALSE(index_array("names").empty());
  size_t expected = 0;
  size_t checked = 0;
  for (const c1::Json& entry : index_array("names")) {
    expected += (bool_of(entry, "legacyParse") ? 1 : 0) + (text_of(entry, "key").empty() ? 0 : 1);
    const std::string name = text_of(entry, "name");
    SCOPED_TRACE(name);
    if (const auto legacy = texture::parse_replacement_filename(name)) {
      EXPECT_EQ(texture::format_material_key(*legacy), c1::canonical_key(*legacy));
      ++checked;
    }
    const std::string key = c1::classify_name(name).key;
    if (key.empty()) {
      continue;
    }
    const auto parsed = texture::parse_material_key(key);
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(texture::format_material_key(*parsed), key);
    EXPECT_EQ(c1::canonical_key(*parsed), key);
    ++checked;
  }
  for (const c1::Json& entry : index_array("lookups")) {
    const auto parsed = texture::parse_material_key(text_of(entry, "query"));
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(texture::format_material_key(*parsed), c1::canonical_key(*parsed));
    EXPECT_EQ(texture::format_material_key(*parsed), text_of(entry, "query"));
  }
  EXPECT_EQ(checked, expected);
  EXPECT_GE(checked, 15u);
  EXPECT_EQ(texture::parse_material_key("tex1_16x16_m_000000000000a001_14"), texture::parse_material_key(KeyA));
  EXPECT_FALSE(texture::parse_material_key("tex1_16x16_000000000000a001").has_value());
  EXPECT_FALSE(texture::parse_material_key(std::string{KeyA} + ".dds").has_value());
}

TEST_F(MaterialRegistryTest, SpanUnregisterPublishesOnce) {
  const auto first = add(manifest_with(fmt::format(R"("{}":{{}})", KeyA)));
  const auto second = add(manifest_with(fmt::format(R"("{}":{{}})", KeyB)));
  material::set_publish_observer(&observe);
  const uint64_t generation = material::generation();
  const std::array<texture::MaterialManifestRegistration, 2> both{first, second};
  texture::unregister_material_manifests(both);
  EXPECT_EQ(material::generation(), generation + 1);
  EXPECT_EQ(g_observed.load(), 2);
  EXPECT_EQ(material::snapshot()->liveCount, 0u);

  const std::array<texture::MaterialManifestRegistration, 3> stale{first, second, {}};
  texture::unregister_material_manifests(stale);
  EXPECT_EQ(material::generation(), generation + 1);
  EXPECT_EQ(g_observed.load(), 2);
}

TEST_F(MaterialRegistryTest, GxLookupFindsRegisteredKey) {
  std::array<uint8_t, 32> pixels{};
  std::array<uint8_t, 64> indices{};
  std::array<uint8_t, 32> tlutA{};
  std::array<uint8_t, 32> tlutB{};
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(i * 11 + 3);
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<uint8_t>(2 + i % 3);
  }
  tlutB[5] = 1;
  const auto plain = texture::compute_source_key(GX_TF_CMPR, 8, 8, pixels);
  const auto palettedA = texture::compute_source_key(GX_TF_C8, 8, 8, indices, tlutA);
  const auto palettedB = texture::compute_source_key(GX_TF_C8, 8, 8, indices, tlutB);
  ASSERT_TRUE(plain.has_value() && palettedA.has_value() && palettedB.has_value());
  ASSERT_NE(palettedA->tlutHash, palettedB->tlutHash);
  texture::TextureSourceKey anyTlut = *palettedA;
  anyTlut.tlutHash = texture::kWildcardTlutHash;
  const std::string plainKey = texture::format_material_key(*plain);
  const std::string paletteKey = texture::format_material_key(anyTlut);

  add(manifest_with(
      fmt::format(R"("{}":{{"flags":["stochastic"]}},"{}":{{"flags":["unlit"]}})", plainKey, paletteKey)));
  const auto wildcard = add(manifest_with(fmt::format(R"("tex1_8x8_$_{}":{{"flags":["cutout"]}})", +GX_TF_CMPR)));
  EXPECT_EQ(wildcard.status, MaterialManifestStatus::Rejected);
  material::latch();
  const auto snap = material::snapshot();
  const uint16_t plainId = snap->find(*plain);
  const uint16_t paletteId = snap->find(anyTlut);
  ASSERT_NE(plainId, 0u);
  ASSERT_NE(paletteId, 0u);
  EXPECT_NE(snap->record(plainId).summary & material::Stochastic, 0u);
  EXPECT_NE(snap->record(paletteId).summary & material::Unlit, 0u);
  const texture::MaterialUsage before = texture::material_usage();
  const uint64_t plainHits = texture::material_hits(plainKey);
  const uint64_t paletteHits = texture::material_hits(paletteKey);

  EXPECT_EQ(material::testing::gx_find(*plain), (gfx::TextureMaterialRef{plainId, snap->record(plainId).summary}));
  for (const texture::TextureSourceKey& key : {*palettedA, *palettedB}) {
    EXPECT_EQ(material::testing::gx_find(key), (gfx::TextureMaterialRef{paletteId, snap->record(paletteId).summary}));
  }
  texture::TextureSourceKey anyTexture = *plain;
  anyTexture.textureHash = texture::kWildcardTextureHash;
  EXPECT_EQ(material::testing::gx_find(anyTexture), gfx::TextureMaterialRef{});
  texture::TextureSourceKey otherTexture = *plain;
  otherTexture.textureHash ^= 1;
  EXPECT_EQ(material::testing::gx_find(otherTexture), gfx::TextureMaterialRef{});

  const texture::MaterialUsage after = texture::material_usage();
  EXPECT_EQ(after.materialCount, 2u);
  EXPECT_EQ(after.materialLookups, before.materialLookups + 5);
  EXPECT_EQ(after.materialHits, before.materialHits + 3);
  EXPECT_EQ(texture::material_hits(plainKey), plainHits + 1);
  EXPECT_EQ(texture::material_hits(paletteKey), paletteHits + 2);
  EXPECT_EQ(texture::material_hits(texture::format_material_key(*palettedB)), paletteHits + 2);
  EXPECT_EQ(texture::material_hits("tex1_8x8_not_a_key"), 0u);
}
} // namespace
} // namespace aurora::test

#pragma once

#include <dolphin/gx.h>

#include <aurora/texture.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::test::c1 {
struct Json {
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<Json> items;
  std::vector<std::string> keys;

  const Json* find(std::string_view key) const noexcept;
  bool is_integer() const noexcept;
};

struct Violation {
  std::string rule;
  std::string pointer;
  std::string detail;
};

struct JsonParse {
  std::optional<Json> value;
  Violation error;
};

struct SchemaSet {
  std::filesystem::path dir;
  std::map<std::string, Json, std::less<>> documents;
  std::map<std::string, std::regex, std::less<>> patterns;
};

struct LayerFile {
  std::string file;
  int layer = 0;
  int32_t priority = 0;
  std::string text;
};

struct MergeWarning {
  std::string file;
  std::string rule;
  std::string pointer;

  bool operator==(const MergeWarning&) const = default;
  auto operator<=>(const MergeWarning&) const = default;
};

struct MergeResult {
  Json manifest;
  std::vector<MergeWarning> warnings;
};

enum class NameKind : uint8_t { Base, BaseMip, Companion, CompanionMip, Ignored, Rejected };

struct NameInfo {
  NameKind kind = NameKind::Ignored;
  std::string key;
  std::string role;
  uint32_t mip = 0;
  std::string rule;
};

enum class DdsStyle : uint8_t { Dx10, Gz2, FourCC, Rgb };

struct DdsHeader {
  DdsStyle style = DdsStyle::Dx10;
  uint32_t code = 0;
  uint32_t fourCC = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mips = 1;
  size_t dataOffset = 0;
  size_t dataSize = 0;
};

struct DdsFile {
  std::string name;
  std::vector<uint8_t> bytes;
};

struct DdsVerdict {
  std::string verdict;
  std::string rule;
  std::string chain;
};

std::span<const std::string_view> known_rules() noexcept;
std::span<const std::string_view> companion_roles() noexcept;
std::optional<std::vector<uint8_t>> read_file(const std::filesystem::path& path);
JsonParse parse_json(std::string_view text);
bool json_equal(const Json& a, const Json& b) noexcept;
std::string pointer_join(std::string_view parent, std::string_view token);
std::vector<Violation> validate_schema(SchemaSet& schemas, std::string_view file, const Json& instance);
std::vector<std::string> unsupported_schema_keywords(SchemaSet& schemas, std::string_view file);
std::string canonical_key(const texture::TextureSourceKey& key);
std::vector<Violation> semantic_rules(const Json& manifest);
std::vector<Violation> check_manifest(SchemaSet& schemas, std::string_view text, int layer, Json* parsed = nullptr);
MergeResult merge_layers(SchemaSet& schemas, std::span<const LayerFile> files, bool replacementsEnabled);
std::optional<std::string> lookup_material(const Json& merged, std::string_view queryKey);
NameInfo classify_name(std::string_view filename);
std::optional<DdsHeader> read_dds_header(std::span<const uint8_t> bytes, Violation* error);
std::map<std::string, DdsVerdict> check_dds_files(std::span<const DdsFile> files);
} // namespace aurora::test::c1

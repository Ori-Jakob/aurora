#include "material_contract_check.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <tuple>
#include <utility>

namespace aurora::test::c1 {
namespace {
constexpr std::array<std::string_view, 40> KnownRules{
    "JSON-SYNTAX",
    "JSON-DUPLICATE-MEMBER",
    "MANIFEST-TYPE",
    "MANIFEST-REQUIRED",
    "MANIFEST-VERSION",
    "MANIFEST-UNKNOWN-MEMBER",
    "MANIFEST-ENUM",
    "MANIFEST-RANGE",
    "MANIFEST-PATTERN",
    "MANIFEST-ARRAY-LENGTH",
    "MANIFEST-DUPLICATE-ITEM",
    "MANIFEST-NOT",
    "KEY-SYNTAX",
    "KEY-TEXTURE-WILDCARD",
    "KEY-TLUT-FORMAT",
    "KEY-RESERVED-HASH",
    "MAP-PATH",
    "RANGE-ORDER",
    "LODVERTS-ORDER",
    "LAYER1-FORBIDDEN-MEMBER",
    "LAYER1-SOURCE",
    "LAYER1-FAKESHADOW-INCOMPLETE",
    "MERGE-GRASS-TYPE-INCOMPLETE",
    "MERGE-GRASS-TYPE-UNRESOLVED",
    "MERGE-FAKESHADOW-INCOMPLETE",
    "MERGE-DISPLACEMENT-INCOMPLETE",
    "NAME-TEXTURE-WILDCARD",
    "IGNORE-NOT-DDS",
    "IGNORE-UNKNOWN-ROLE",
    "IGNORE-BAD-MIP",
    "IGNORE-BAD-STEM",
    "IGNORE-ORPHAN-SIDECAR",
    "IGNORE-INTERNAL-MIPS",
    "IGNORE-AFTER-INVALID",
    "DDS-HEADER",
    "DDS-SIZE",
    "DDS-ROLE-FORMAT",
    "DDS-DELIGHT-RES",
    "DDS-MIP-FORMAT",
    "DDS-MIP-SIZE",
};

constexpr std::array<std::string_view, 7> CompanionRoles{"albedo", "delight", "nrm", "orm", "emit", "grassmask", "mat"};

constexpr std::array<std::string_view, 29> SchemaKeywords{
    "$ref",
    "$defs",
    "allOf",
    "not",
    "properties",
    "additionalProperties",
    "propertyNames",
    "items",
    "type",
    "const",
    "enum",
    "required",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "$schema",
    "$id",
    "title",
    "description",
    "default",
    "$comment",
    "x-c1-rule",
};

constexpr std::array<std::string_view, 6> NestedMaterialFields{"stochastic",   "maps",       "grass",
                                                               "displacement", "fakeShadow", "_provenance"};
constexpr std::array<std::string_view, 1> NestedGrassTypeFields{"clump"};
constexpr std::array<std::string_view, 5> RequiredGrassTypeFields{"height", "width", "densityPerM2", "rootColor",
                                                                  "tipColor"};

constexpr std::string_view BaseSchema = "materials.schema.json";
constexpr std::string_view Layer1Schema = "materials.layer1.schema.json";
constexpr int MaxJsonDepth = 64;

constexpr uint32_t DdsMagic = 0x20534444;
constexpr uint32_t FourCCDx10 = 0x30315844;
constexpr uint32_t FourCCDxt1 = 0x31545844;
constexpr uint32_t FourCCDxt5 = 0x35545844;
constexpr uint32_t FourCCAti2 = 0x32495441;
constexpr uint32_t PixelFormatFourCC = 0x4;
constexpr uint32_t PixelFormatRgb = 0x40;
constexpr uint32_t FlagMipMapCount = 0x00020000;
constexpr uint32_t CapsComplex = 0x8;
constexpr uint32_t CapsMipMap = 0x00400000;
constexpr uint32_t Caps2Cubemap = 0x200;
constexpr uint32_t Caps2Volume = 0x00200000;
constexpr uint32_t MiscTextureCube = 0x4;
constexpr size_t DdsHeaderSize = 128;
constexpr size_t Dx10HeaderSize = 148;

template <typename List>
bool contains(const List& list, std::string_view value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

std::string_view as_text(const std::vector<uint8_t>& bytes) noexcept {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

char ascii_lower(char ch) noexcept { return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch - 'A' + 'a') : ch; }

bool iequals_ascii(std::string_view a, std::string_view b) noexcept {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return ascii_lower(x) == ascii_lower(y); });
}

Json make_object() {
  Json out;
  out.type = Json::Type::Object;
  return out;
}

Json make_string(std::string_view text) {
  Json out;
  out.type = Json::Type::String;
  out.text = text;
  return out;
}

Json* find_member(Json& object, std::string_view key) noexcept {
  for (size_t i = 0; i < object.keys.size(); ++i) {
    if (object.keys[i] == key) {
      return &object.items[i];
    }
  }
  return nullptr;
}

Json& member(Json& object, std::string_view key) {
  if (Json* existing = find_member(object, key)) {
    return *existing;
  }
  object.keys.emplace_back(key);
  return object.items.emplace_back();
}

void erase_member(Json& object, std::string_view key) {
  for (size_t i = 0; i < object.keys.size(); ++i) {
    if (object.keys[i] == key) {
      object.keys.erase(object.keys.begin() + static_cast<ptrdiff_t>(i));
      object.items.erase(object.items.begin() + static_cast<ptrdiff_t>(i));
      return;
    }
  }
}

size_t valid_utf8_length(std::string_view text) noexcept {
  const auto lead = static_cast<unsigned char>(text[0]);
  if (lead < 0x80) {
    return 1;
  }
  size_t length = 0;
  uint32_t codePoint = 0;
  uint32_t minimum = 0;
  if ((lead & 0xE0) == 0xC0) {
    length = 2;
    codePoint = lead & 0x1F;
    minimum = 0x80;
  } else if ((lead & 0xF0) == 0xE0) {
    length = 3;
    codePoint = lead & 0x0F;
    minimum = 0x800;
  } else if ((lead & 0xF8) == 0xF0) {
    length = 4;
    codePoint = lead & 0x07;
    minimum = 0x10000;
  } else {
    return 0;
  }
  if (text.size() < length) {
    return 0;
  }
  for (size_t i = 1; i < length; ++i) {
    const auto next = static_cast<unsigned char>(text[i]);
    if ((next & 0xC0) != 0x80) {
      return 0;
    }
    codePoint = codePoint << 6 | (next & 0x3F);
  }
  if (codePoint < minimum || codePoint > 0x10FFFF || (codePoint >= 0xD800 && codePoint <= 0xDFFF)) {
    return 0;
  }
  return length;
}

void append_utf8(std::string& out, uint32_t codePoint) {
  if (codePoint < 0x80) {
    out += static_cast<char>(codePoint);
  } else if (codePoint < 0x800) {
    out += static_cast<char>(0xC0 | codePoint >> 6);
    out += static_cast<char>(0x80 | (codePoint & 0x3F));
  } else if (codePoint < 0x10000) {
    out += static_cast<char>(0xE0 | codePoint >> 12);
    out += static_cast<char>(0x80 | (codePoint >> 6 & 0x3F));
    out += static_cast<char>(0x80 | (codePoint & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | codePoint >> 18);
    out += static_cast<char>(0x80 | (codePoint >> 12 & 0x3F));
    out += static_cast<char>(0x80 | (codePoint >> 6 & 0x3F));
    out += static_cast<char>(0x80 | (codePoint & 0x3F));
  }
}

size_t code_points(std::string_view text) noexcept {
  return static_cast<size_t>(
      std::count_if(text.begin(), text.end(), [](char ch) { return (static_cast<unsigned char>(ch) & 0xC0) != 0x80; }));
}

struct JsonParser {
  std::string_view text;
  size_t pos = 0;
  int depth = 0;
  std::string pointer;
  std::optional<Violation> error;

  bool fail(std::string_view what) {
    error = Violation{"JSON-SYNTAX", "", fmt::format("{} at byte {}", what, pos)};
    return false;
  }

  bool at(char ch) const noexcept { return pos < text.size() && text[pos] == ch; }
  bool at_digit() const noexcept { return pos < text.size() && text[pos] >= '0' && text[pos] <= '9'; }

  void skip_space() noexcept {
    while (at(' ') || at('\t') || at('\n') || at('\r')) {
      ++pos;
    }
  }

  bool parse_literal(std::string_view word) {
    if (text.substr(pos, word.size()) != word) {
      return fail("invalid literal");
    }
    pos += word.size();
    return true;
  }

  bool parse_value(Json& out) {
    skip_space();
    if (pos >= text.size()) {
      return fail("unexpected end of input");
    }
    switch (text[pos]) {
    case '{':
      return parse_object(out);
    case '[':
      return parse_array(out);
    case '"':
      out.type = Json::Type::String;
      return parse_string(out.text);
    case 't':
      out.type = Json::Type::Bool;
      out.boolean = true;
      return parse_literal("true");
    case 'f':
      out.type = Json::Type::Bool;
      return parse_literal("false");
    case 'n':
      return parse_literal("null");
    default:
      if (at('-') || at_digit()) {
        return parse_number(out);
      }
      return fail("unexpected character");
    }
  }

  bool parse_number(Json& out) {
    const size_t start = pos;
    if (at('-')) {
      ++pos;
    }
    if (at('0')) {
      ++pos;
      if (at_digit()) {
        return fail("leading zero");
      }
    } else if (at_digit()) {
      while (at_digit()) {
        ++pos;
      }
    } else {
      return fail("invalid number");
    }
    if (at('.')) {
      ++pos;
      if (!at_digit()) {
        return fail("invalid fraction");
      }
      while (at_digit()) {
        ++pos;
      }
    }
    if (at('e') || at('E')) {
      ++pos;
      if (at('+') || at('-')) {
        ++pos;
      }
      if (!at_digit()) {
        return fail("invalid exponent");
      }
      while (at_digit()) {
        ++pos;
      }
    }
    const std::string token{text.substr(start, pos - start)};
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size() || !std::isfinite(value)) {
      pos = start;
      return fail("number out of range");
    }
    out.type = Json::Type::Number;
    out.number = value;
    return true;
  }

  bool parse_hex4(uint32_t& out) {
    if (text.size() - pos < 4) {
      return fail("truncated escape");
    }
    out = 0;
    for (int i = 0; i < 4; ++i) {
      const char ch = text[pos++];
      out <<= 4;
      if (ch >= '0' && ch <= '9') {
        out |= static_cast<uint32_t>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        out |= static_cast<uint32_t>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        out |= static_cast<uint32_t>(ch - 'A' + 10);
      } else {
        return fail("invalid escape");
      }
    }
    return true;
  }

  bool parse_string(std::string& out) {
    ++pos;
    out.clear();
    while (true) {
      if (pos >= text.size()) {
        return fail("unterminated string");
      }
      const auto ch = static_cast<unsigned char>(text[pos]);
      if (ch == '"') {
        ++pos;
        return true;
      }
      if (ch < 0x20) {
        return fail("unescaped control character");
      }
      if (ch != '\\') {
        const size_t length = valid_utf8_length(text.substr(pos));
        if (length == 0) {
          return fail("invalid UTF-8");
        }
        out.append(text.substr(pos, length));
        pos += length;
        continue;
      }
      ++pos;
      if (pos >= text.size()) {
        return fail("unterminated string");
      }
      const char escape = text[pos++];
      switch (escape) {
      case '"':
      case '\\':
      case '/':
        out += escape;
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'u': {
        uint32_t codePoint = 0;
        if (!parse_hex4(codePoint)) {
          return false;
        }
        if (codePoint >= 0xDC00 && codePoint <= 0xDFFF) {
          return fail("lone surrogate");
        }
        if (codePoint >= 0xD800 && codePoint <= 0xDBFF) {
          uint32_t low = 0;
          if (text.substr(pos, 2) != "\\u") {
            return fail("lone surrogate");
          }
          pos += 2;
          if (!parse_hex4(low)) {
            return false;
          }
          if (low < 0xDC00 || low > 0xDFFF) {
            return fail("lone surrogate");
          }
          codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
        }
        append_utf8(out, codePoint);
        break;
      }
      default:
        return fail("invalid escape");
      }
    }
  }

  bool parse_array(Json& out) {
    if (++depth > MaxJsonDepth) {
      return fail("nesting deeper than 64");
    }
    out.type = Json::Type::Array;
    ++pos;
    skip_space();
    if (at(']')) {
      ++pos;
      --depth;
      return true;
    }
    const size_t base = pointer.size();
    while (true) {
      pointer.resize(base);
      pointer += fmt::format("/{}", out.items.size());
      if (!parse_value(out.items.emplace_back())) {
        return false;
      }
      skip_space();
      if (at(',')) {
        ++pos;
        continue;
      }
      if (at(']')) {
        ++pos;
        break;
      }
      return fail("expected ',' or ']'");
    }
    pointer.resize(base);
    --depth;
    return true;
  }

  bool parse_object(Json& out) {
    if (++depth > MaxJsonDepth) {
      return fail("nesting deeper than 64");
    }
    out.type = Json::Type::Object;
    ++pos;
    skip_space();
    if (at('}')) {
      ++pos;
      --depth;
      return true;
    }
    const size_t base = pointer.size();
    while (true) {
      skip_space();
      if (!at('"')) {
        return fail("expected member name");
      }
      const size_t keyStart = pos;
      std::string key;
      if (!parse_string(key)) {
        return false;
      }
      pointer = pointer_join(std::string_view{pointer}.substr(0, base), key);
      if (contains(out.keys, key)) {
        error = Violation{"JSON-DUPLICATE-MEMBER", pointer, fmt::format("duplicate member at byte {}", keyStart)};
        return false;
      }
      skip_space();
      if (!at(':')) {
        return fail("expected ':'");
      }
      ++pos;
      out.keys.push_back(std::move(key));
      if (!parse_value(out.items.emplace_back())) {
        return false;
      }
      skip_space();
      if (at(',')) {
        ++pos;
        continue;
      }
      if (at('}')) {
        ++pos;
        break;
      }
      return fail("expected ',' or '}'");
    }
    pointer.resize(base);
    --depth;
    return true;
  }
};

std::string unescape_pointer_token(std::string_view token) {
  std::string out;
  for (size_t i = 0; i < token.size(); ++i) {
    if (token[i] == '~' && i + 1 < token.size() && (token[i + 1] == '0' || token[i + 1] == '1')) {
      out += token[i + 1] == '0' ? '~' : '/';
      ++i;
    } else {
      out += token[i];
    }
  }
  return out;
}

const Json* load_schema(SchemaSet& schemas, std::string_view file) {
  if (const auto it = schemas.documents.find(file); it != schemas.documents.end()) {
    return &it->second;
  }
  const auto bytes = read_file(schemas.dir / std::string{file});
  if (!bytes) {
    return nullptr;
  }
  JsonParse parsed = parse_json(as_text(*bytes));
  if (!parsed.value) {
    return nullptr;
  }
  return &schemas.documents.emplace(std::string{file}, std::move(*parsed.value)).first->second;
}

struct SchemaNode {
  const Json* schema = nullptr;
  const Json* document = nullptr;
};

std::optional<SchemaNode> resolve_ref(SchemaSet& schemas, const Json& document, std::string_view ref) {
  const size_t hash = ref.find('#');
  const std::string_view location = ref.substr(0, hash);
  const std::string_view fragment = hash == std::string_view::npos ? std::string_view{} : ref.substr(hash + 1);
  const Json* root = &document;
  if (!location.empty()) {
    const size_t slash = location.rfind('/');
    root = load_schema(schemas, slash == std::string_view::npos ? location : location.substr(slash + 1));
    if (root == nullptr) {
      return std::nullopt;
    }
  }
  const Json* node = root;
  if (!fragment.empty()) {
    if (fragment.front() != '/') {
      return std::nullopt;
    }
    size_t start = 1;
    while (node != nullptr) {
      const size_t end = fragment.find('/', start);
      node = node->find(unescape_pointer_token(fragment.substr(start, end - start)));
      if (end == std::string_view::npos) {
        break;
      }
      start = end + 1;
    }
    if (node == nullptr) {
      return std::nullopt;
    }
  }
  return SchemaNode{node, root};
}

bool matches_type_name(std::string_view name, const Json& value) noexcept {
  if (name == "integer") {
    return value.is_integer();
  }
  switch (value.type) {
  case Json::Type::Null:
    return name == "null";
  case Json::Type::Bool:
    return name == "boolean";
  case Json::Type::Number:
    return name == "number";
  case Json::Type::String:
    return name == "string";
  case Json::Type::Array:
    return name == "array";
  case Json::Type::Object:
    return name == "object";
  }
  return false;
}

bool matches_type(const Json& type, const Json& value) noexcept {
  if (type.type == Json::Type::String) {
    return matches_type_name(type.text, value);
  }
  return std::any_of(type.items.begin(), type.items.end(), [&](const Json& name) {
    return name.type == Json::Type::String && matches_type_name(name.text, value);
  });
}

bool has_duplicate_items(const Json& array) noexcept {
  for (size_t i = 0; i < array.items.size(); ++i) {
    for (size_t j = i + 1; j < array.items.size(); ++j) {
      if (json_equal(array.items[i], array.items[j])) {
        return true;
      }
    }
  }
  return false;
}

struct SchemaEvaluator {
  SchemaSet& schemas;

  const std::regex& regex_for(const std::string& source) {
    auto it = schemas.patterns.find(source);
    if (it == schemas.patterns.end()) {
      it = schemas.patterns.emplace(source, std::regex{source, std::regex::ECMAScript}).first;
    }
    return it->second;
  }

  void eval(const Json& schema, const Json& document, const Json& instance, const std::string& pointer,
            std::vector<Violation>& out) {
    if (schema.type == Json::Type::Bool) {
      if (!schema.boolean) {
        out.push_back({"MANIFEST-NOT", pointer, "false schema"});
      }
      return;
    }
    if (schema.type != Json::Type::Object) {
      return;
    }
    // A failing keyword takes the annotation of its own subschema only, never an ancestor's.
    const Json* annotation = schema.find("x-c1-rule");
    const auto fail = [&](std::string_view fallback, std::string at, std::string_view keyword) {
      const bool annotated = annotation != nullptr && annotation->type == Json::Type::String;
      out.push_back({std::string{annotated ? std::string_view{annotation->text} : fallback}, std::move(at),
                     std::string{keyword}});
    };

    if (const Json* type = schema.find("type"); type != nullptr && !matches_type(*type, instance)) {
      fail("MANIFEST-TYPE", pointer, "type");
      return;
    }
    if (const Json* value = schema.find("const"); value != nullptr && !json_equal(*value, instance)) {
      fail("MANIFEST-ENUM", pointer, "const");
    }
    if (const Json* values = schema.find("enum");
        values != nullptr && std::none_of(values->items.begin(), values->items.end(),
                                          [&](const Json& v) { return json_equal(v, instance); })) {
      fail("MANIFEST-ENUM", pointer, "enum");
    }
    if (const Json* ref = schema.find("$ref"); ref != nullptr) {
      const auto target = resolve_ref(schemas, document, ref->text);
      if (!target) {
        out.push_back({"", pointer, fmt::format("unresolved $ref {}", ref->text)});
      } else {
        eval(*target->schema, *target->document, instance, pointer, out);
      }
    }
    if (const Json* branches = schema.find("allOf"); branches != nullptr) {
      for (const Json& branch : branches->items) {
        std::vector<Violation> branchOut;
        eval(branch, document, instance, pointer, branchOut);
        if (!branchOut.empty()) {
          out.insert(out.end(), std::make_move_iterator(branchOut.begin()), std::make_move_iterator(branchOut.end()));
          break;
        }
      }
    }
    if (const Json* negated = schema.find("not"); negated != nullptr) {
      std::vector<Violation> inner;
      eval(*negated, document, instance, pointer, inner);
      if (inner.empty()) {
        fail("MANIFEST-NOT", pointer, "not");
      }
    }

    if (instance.type == Json::Type::String) {
      const auto length = static_cast<double>(code_points(instance.text));
      if (const Json* limit = schema.find("minLength"); limit != nullptr && length < limit->number) {
        fail("MANIFEST-PATTERN", pointer, "minLength");
      }
      if (const Json* limit = schema.find("maxLength"); limit != nullptr && length > limit->number) {
        fail("MANIFEST-PATTERN", pointer, "maxLength");
      }
      if (const Json* source = schema.find("pattern");
          source != nullptr && !std::regex_search(instance.text, regex_for(source->text))) {
        fail("MANIFEST-PATTERN", pointer, "pattern");
      }
    }
    if (instance.type == Json::Type::Number) {
      const double value = instance.number;
      if (const Json* limit = schema.find("minimum"); limit != nullptr && value < limit->number) {
        fail("MANIFEST-RANGE", pointer, "minimum");
      }
      if (const Json* limit = schema.find("maximum"); limit != nullptr && value > limit->number) {
        fail("MANIFEST-RANGE", pointer, "maximum");
      }
      if (const Json* limit = schema.find("exclusiveMinimum"); limit != nullptr && value <= limit->number) {
        fail("MANIFEST-RANGE", pointer, "exclusiveMinimum");
      }
      if (const Json* limit = schema.find("exclusiveMaximum"); limit != nullptr && value >= limit->number) {
        fail("MANIFEST-RANGE", pointer, "exclusiveMaximum");
      }
    }
    if (instance.type == Json::Type::Array) {
      const auto count = static_cast<double>(instance.items.size());
      if (const Json* limit = schema.find("minItems"); limit != nullptr && count < limit->number) {
        fail("MANIFEST-ARRAY-LENGTH", pointer, "minItems");
      }
      if (const Json* limit = schema.find("maxItems"); limit != nullptr && count > limit->number) {
        fail("MANIFEST-ARRAY-LENGTH", pointer, "maxItems");
      }
      if (const Json* unique = schema.find("uniqueItems");
          unique != nullptr && unique->boolean && has_duplicate_items(instance)) {
        fail("MANIFEST-DUPLICATE-ITEM", pointer, "uniqueItems");
      }
      if (const Json* items = schema.find("items"); items != nullptr) {
        for (size_t i = 0; i < instance.items.size(); ++i) {
          eval(*items, document, instance.items[i], pointer_join(pointer, std::to_string(i)), out);
        }
      }
    }
    if (instance.type != Json::Type::Object) {
      return;
    }
    if (const Json* required = schema.find("required"); required != nullptr) {
      for (const Json& name : required->items) {
        if (instance.find(name.text) == nullptr) {
          fail("MANIFEST-REQUIRED", pointer_join(pointer, name.text), "required");
        }
      }
    }
    if (const Json* names = schema.find("propertyNames"); names != nullptr) {
      for (const std::string& key : instance.keys) {
        eval(*names, document, make_string(key), pointer_join(pointer, key), out);
      }
    }
    const Json* properties = schema.find("properties");
    if (properties != nullptr) {
      for (size_t i = 0; i < instance.keys.size(); ++i) {
        if (const Json* property = properties->find(instance.keys[i]); property != nullptr) {
          eval(*property, document, instance.items[i], pointer_join(pointer, instance.keys[i]), out);
        }
      }
    }
    if (const Json* additional = schema.find("additionalProperties"); additional != nullptr) {
      for (size_t i = 0; i < instance.keys.size(); ++i) {
        if (properties != nullptr && properties->find(instance.keys[i]) != nullptr) {
          continue;
        }
        std::string at = pointer_join(pointer, instance.keys[i]);
        if (additional->type != Json::Type::Bool) {
          eval(*additional, document, instance.items[i], at, out);
        } else if (!additional->boolean) {
          fail("MANIFEST-UNKNOWN-MEMBER", std::move(at), "additionalProperties");
        }
      }
    }
  }
};

void collect_unsupported(SchemaSet& schemas, const Json& document, const Json& schema, const std::string& pointer,
                         std::vector<std::string>& out) {
  if (schema.type == Json::Type::Bool) {
    return;
  }
  if (schema.type != Json::Type::Object) {
    out.push_back(pointer);
    return;
  }
  for (size_t i = 0; i < schema.keys.size(); ++i) {
    const std::string& keyword = schema.keys[i];
    const Json& value = schema.items[i];
    const std::string at = pointer_join(pointer, keyword);
    if (!contains(SchemaKeywords, keyword)) {
      out.push_back(at);
    } else if (keyword == "$defs" || keyword == "properties") {
      for (size_t j = 0; j < value.keys.size(); ++j) {
        collect_unsupported(schemas, document, value.items[j], pointer_join(at, value.keys[j]), out);
      }
    } else if (keyword == "additionalProperties" || keyword == "propertyNames" || keyword == "items" ||
               keyword == "not") {
      collect_unsupported(schemas, document, value, at, out);
    } else if (keyword == "allOf") {
      for (size_t j = 0; j < value.items.size(); ++j) {
        collect_unsupported(schemas, document, value.items[j], pointer_join(at, std::to_string(j)), out);
      }
    } else if (keyword == "$ref" && (value.type != Json::Type::String || !resolve_ref(schemas, document, value.text))) {
      out.push_back(at);
    }
  }
}

std::optional<texture::TextureSourceKey> parse_key(std::string_view key) {
  return texture::parse_replacement_filename(fmt::format("{}.dds", key));
}

bool is_reserved_key(std::string_view key) {
  const auto parsed = parse_key(key);
  if (!parsed) {
    return false;
  }
  if (parsed->textureHash == texture::kWildcardTextureHash) {
    return true;
  }
  // The parser turns a literal fffffffffffffffe into the TLUT wildcard, so only a real "$" may produce it.
  return parsed->hasTlut && parsed->tlutHash == texture::kWildcardTlutHash && key.find("_$_") == std::string_view::npos;
}

void merge_fields(Json& target, const Json& source, std::span<const std::string_view> nested) {
  for (size_t i = 0; i < source.keys.size(); ++i) {
    const Json& value = source.items[i];
    Json& slot = member(target, source.keys[i]);
    const bool mergeable =
        contains(nested, source.keys[i]) && slot.type == Json::Type::Object && value.type == Json::Type::Object;
    if (!mergeable) {
      slot = value;
      continue;
    }
    for (size_t j = 0; j < value.keys.size(); ++j) {
      member(slot, value.keys[j]) = value.items[j];
    }
  }
}

void merge_section(Json& target, const Json* source, std::span<const std::string_view> nested) {
  if (source == nullptr) {
    return;
  }
  for (size_t i = 0; i < source->keys.size(); ++i) {
    Json& entry = member(target, source->keys[i]);
    if (entry.type != Json::Type::Object) {
      entry = make_object();
    }
    merge_fields(entry, source->items[i], nested);
  }
}

void check_merged(Json& grassTypes, Json& materials, std::vector<MergeWarning>& warnings) {
  std::vector<std::string> names = grassTypes.keys;
  std::sort(names.begin(), names.end());
  for (const std::string& name : names) {
    const Json& type = *grassTypes.find(name);
    const bool complete = std::all_of(RequiredGrassTypeFields.begin(), RequiredGrassTypeFields.end(),
                                      [&](std::string_view field) { return type.find(field) != nullptr; });
    if (!complete) {
      warnings.push_back({"", "MERGE-GRASS-TYPE-INCOMPLETE", pointer_join("/grassTypes", name)});
      erase_member(grassTypes, name);
    }
  }

  std::vector<std::string> keys = materials.keys;
  std::sort(keys.begin(), keys.end());
  for (const std::string& key : keys) {
    Json& entry = *find_member(materials, key);
    const std::string at = pointer_join("/materials", key);
    if (const Json* grass = entry.find("grass"); grass != nullptr) {
      const Json* type = grass->find("type");
      if (type == nullptr || grassTypes.find(type->text) == nullptr) {
        warnings.push_back({"", "MERGE-GRASS-TYPE-UNRESOLVED", pointer_join(at, "grass")});
        erase_member(entry, "grass");
      }
    }
    if (const Json* shadow = entry.find("fakeShadow");
        shadow != nullptr && (shadow->find("neutral") == nullptr || shadow->find("gate") == nullptr)) {
      warnings.push_back({"", "MERGE-FAKESHADOW-INCOMPLETE", pointer_join(at, "fakeShadow")});
      erase_member(entry, "fakeShadow");
    }
    if (const Json* displacement = entry.find("displacement");
        displacement != nullptr && displacement->find("amplitude") == nullptr) {
      warnings.push_back({"", "MERGE-DISPLACEMENT-INCOMPLETE", pointer_join(at, "displacement")});
      erase_member(entry, "displacement");
    }
  }
}

NameInfo ignored_name(std::string_view rule) {
  NameInfo info;
  info.kind = NameKind::Ignored;
  info.rule = rule;
  return info;
}

struct BlockFormat {
  uint32_t bytes = 0;
  bool compressed = false;
};

std::optional<BlockFormat> block_format(uint32_t code) noexcept {
  switch (code) {
  case 71:
  case 72:
  case 80:
  case 81:
    return BlockFormat{8, true};
  case 77:
  case 78:
  case 83:
  case 84:
  case 95:
  case 96:
  case 98:
  case 99:
    return BlockFormat{16, true};
  case 28:
  case 29:
  case 87:
  case 91:
    return BlockFormat{4, false};
  default:
    return std::nullopt;
  }
}

size_t level_bytes(const BlockFormat& format, uint32_t width, uint32_t height) noexcept {
  if (!format.compressed) {
    return size_t{width} * height * format.bytes;
  }
  return size_t{(width + 3) / 4} * ((height + 3) / 4) * format.bytes;
}

uint32_t full_chain_length(uint32_t width, uint32_t height) noexcept {
  uint32_t count = 1;
  while (width > 1 || height > 1) {
    width = std::max(width >> 1, 1u);
    height = std::max(height >> 1, 1u);
    ++count;
  }
  return count;
}

uint32_t read_u32(std::span<const uint8_t> bytes, size_t offset) noexcept {
  return uint32_t{bytes[offset]} | uint32_t{bytes[offset + 1]} << 8 | uint32_t{bytes[offset + 2]} << 16 |
         uint32_t{bytes[offset + 3]} << 24;
}

bool role_accepts(std::string_view role, uint32_t code) noexcept {
  if (role == "albedo" || role == "emit") {
    return code == 99 || code == 98;
  }
  if (role == "delight" || role == "grassmask") {
    return code == 80;
  }
  if (role == "nrm") {
    return code == 83;
  }
  if (role == "orm") {
    return code == 98;
  }
  if (role == "mat") {
    return code == 71 || code == 77 || code == 83 || code == 98 || code == 28 || code == 87;
  }
  return false;
}
} // namespace

const Json* Json::find(std::string_view key) const noexcept {
  if (type != Type::Object) {
    return nullptr;
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] == key) {
      return &items[i];
    }
  }
  return nullptr;
}

bool Json::is_integer() const noexcept {
  return type == Type::Number && std::isfinite(number) && std::floor(number) == number;
}

std::span<const std::string_view> known_rules() noexcept { return KnownRules; }

std::span<const std::string_view> companion_roles() noexcept { return CompanionRoles; }

std::optional<std::vector<uint8_t>> read_file(const std::filesystem::path& path) {
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    return std::nullopt;
  }
  std::vector<uint8_t> bytes{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
  if (stream.bad()) {
    return std::nullopt;
  }
  return bytes;
}

JsonParse parse_json(std::string_view text) {
  JsonParser parser{.text = text};
  Json root;
  if (parser.parse_value(root)) {
    parser.skip_space();
    if (parser.pos != text.size()) {
      parser.fail("trailing data");
    }
  }
  if (parser.error) {
    return {std::nullopt, *parser.error};
  }
  return {std::move(root), {}};
}

bool json_equal(const Json& a, const Json& b) noexcept {
  if (a.type != b.type) {
    return false;
  }
  switch (a.type) {
  case Json::Type::Null:
    return true;
  case Json::Type::Bool:
    return a.boolean == b.boolean;
  case Json::Type::Number:
    return a.number == b.number;
  case Json::Type::String:
    return a.text == b.text;
  case Json::Type::Array:
    return a.items.size() == b.items.size() &&
           std::equal(a.items.begin(), a.items.end(), b.items.begin(),
                      [](const Json& x, const Json& y) { return json_equal(x, y); });
  case Json::Type::Object:
    if (a.keys.size() != b.keys.size()) {
      return false;
    }
    for (size_t i = 0; i < a.keys.size(); ++i) {
      const Json* other = b.find(a.keys[i]);
      if (other == nullptr || !json_equal(a.items[i], *other)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string pointer_join(std::string_view parent, std::string_view token) {
  std::string out{parent};
  out += '/';
  for (const char ch : token) {
    if (ch == '~') {
      out += "~0";
    } else if (ch == '/') {
      out += "~1";
    } else {
      out += ch;
    }
  }
  return out;
}

std::vector<Violation> validate_schema(SchemaSet& schemas, std::string_view file, const Json& instance) {
  std::vector<Violation> out;
  const Json* document = load_schema(schemas, file);
  if (document == nullptr) {
    out.push_back({"", "", fmt::format("cannot load schema {}", file)});
    return out;
  }
  SchemaEvaluator{schemas}.eval(*document, *document, instance, "", out);
  return out;
}

std::vector<std::string> unsupported_schema_keywords(SchemaSet& schemas, std::string_view file) {
  std::vector<std::string> out;
  const Json* document = load_schema(schemas, file);
  if (document == nullptr) {
    out.emplace_back(file);
    return out;
  }
  collect_unsupported(schemas, *document, *document, "", out);
  return out;
}

std::string canonical_key(const texture::TextureSourceKey& key) {
  std::string out = fmt::format("tex1_{}x{}_", key.width, key.height);
  out += key.textureHash == texture::kWildcardTextureHash ? std::string{"$"} : fmt::format("{:016x}", key.textureHash);
  if (key.hasTlut) {
    out += '_';
    out += key.tlutHash == texture::kWildcardTlutHash ? std::string{"$"} : fmt::format("{:016x}", key.tlutHash);
  }
  out += fmt::format("_{}", key.format);
  return out;
}

std::vector<Violation> semantic_rules(const Json& manifest) {
  std::vector<Violation> out;
  if (const Json* types = manifest.find("grassTypes"); types != nullptr) {
    for (size_t i = 0; i < types->keys.size(); ++i) {
      const Json& type = types->items[i];
      const std::string at = pointer_join("/grassTypes", types->keys[i]);
      for (const std::string_view field : {std::string_view{"height"}, std::string_view{"width"}}) {
        const Json* range = type.find(field);
        if (range != nullptr && range->items.size() == 2 && range->items[0].number > range->items[1].number) {
          out.push_back({"RANGE-ORDER", pointer_join(at, field), "lower bound above upper bound"});
        }
      }
      if (const Json* lods = type.find("lodVerts"); lods != nullptr) {
        for (size_t j = 1; j < lods->items.size(); ++j) {
          if (lods->items[j].number >= lods->items[j - 1].number) {
            out.push_back({"LODVERTS-ORDER", pointer_join(at, "lodVerts"), "not strictly decreasing"});
            break;
          }
        }
      }
      if (const Json* colorFrom = type.find("colorFrom"); colorFrom != nullptr && is_reserved_key(colorFrom->text)) {
        out.push_back({"KEY-RESERVED-HASH", pointer_join(at, "colorFrom"), colorFrom->text});
      }
    }
  }
  if (const Json* materials = manifest.find("materials"); materials != nullptr) {
    for (const std::string& key : materials->keys) {
      if (is_reserved_key(key)) {
        out.push_back({"KEY-RESERVED-HASH", pointer_join("/materials", key), key});
      }
    }
  }
  return out;
}

std::vector<Violation> check_manifest(SchemaSet& schemas, std::string_view text, int layer, Json* parsed) {
  JsonParse result = parse_json(text);
  if (!result.value) {
    return {result.error};
  }
  std::vector<Violation> violations = validate_schema(schemas, layer == 1 ? Layer1Schema : BaseSchema, *result.value);
  if (violations.empty()) {
    violations = semantic_rules(*result.value);
  }
  if (parsed != nullptr) {
    *parsed = std::move(*result.value);
  }
  return violations;
}

MergeResult merge_layers(SchemaSet& schemas, std::span<const LayerFile> files, bool replacementsEnabled) {
  std::vector<size_t> order;
  for (size_t i = 0; i < files.size(); ++i) {
    // Pack and mod layers ride on the replacements toggle; the game and user layers always apply.
    if (replacementsEnabled || (files[i].layer != 2 && files[i].layer != 3)) {
      order.push_back(i);
    }
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::tie(files[a].layer, files[a].priority) < std::tie(files[b].layer, files[b].priority);
  });

  MergeResult result;
  Json grassTypes = make_object();
  Json materials = make_object();
  for (const size_t i : order) {
    Json manifest;
    const auto violations = check_manifest(schemas, files[i].text, files[i].layer, &manifest);
    if (!violations.empty()) {
      result.warnings.push_back({files[i].file, violations.front().rule, violations.front().pointer});
      continue;
    }
    merge_section(grassTypes, manifest.find("grassTypes"), NestedGrassTypeFields);
    merge_section(materials, manifest.find("materials"), NestedMaterialFields);
  }
  check_merged(grassTypes, materials, result.warnings);

  result.manifest = make_object();
  Json& version = member(result.manifest, "version");
  version.type = Json::Type::Number;
  version.number = 1.0;
  member(result.manifest, "grassTypes") = std::move(grassTypes);
  member(result.manifest, "materials") = std::move(materials);
  return result;
}

std::optional<std::string> lookup_material(const Json& merged, std::string_view queryKey) {
  const Json* materials = merged.find("materials");
  auto parsed = parse_key(queryKey);
  if (materials == nullptr || !parsed) {
    return std::nullopt;
  }
  std::string key = canonical_key(*parsed);
  if (materials->find(key) != nullptr) {
    return key;
  }
  if (parsed->hasTlut) {
    parsed->tlutHash = texture::kWildcardTlutHash;
    key = canonical_key(*parsed);
    if (materials->find(key) != nullptr) {
      return key;
    }
  }
  return std::nullopt;
}

NameInfo classify_name(std::string_view filename) {
  constexpr std::string_view Extension = ".dds";
  if (filename.size() < Extension.size() ||
      !iequals_ascii(filename.substr(filename.size() - Extension.size()), Extension)) {
    return ignored_name("IGNORE-NOT-DDS");
  }
  std::string_view body = filename.substr(0, filename.size() - Extension.size());
  std::string_view role;
  if (const size_t dot = body.rfind('.'); dot != std::string_view::npos) {
    role = body.substr(dot + 1);
    if (!contains(CompanionRoles, role)) {
      return ignored_name("IGNORE-UNKNOWN-ROLE");
    }
    body = body.substr(0, dot);
  }

  uint32_t mip = 0;
  size_t digits = body.size();
  while (digits > 0 && body[digits - 1] >= '0' && body[digits - 1] <= '9') {
    --digits;
  }
  if (digits < body.size() && body.substr(0, digits).ends_with("_mip")) {
    const std::string_view number = body.substr(digits);
    if (number.front() == '0' || number.size() > 2) {
      return ignored_name("IGNORE-BAD-MIP");
    }
    for (const char ch : number) {
      mip = mip * 10 + static_cast<uint32_t>(ch - '0');
    }
    if (mip > 15) {
      return ignored_name("IGNORE-BAD-MIP");
    }
    body = body.substr(0, digits - 4);
  }

  const auto parsed = parse_key(body);
  if (!parsed) {
    return ignored_name("IGNORE-BAD-STEM");
  }
  if (!role.empty()) {
    // Legacy parse_hex keeps only the low 64 bits, so a longer hash would alias another texture.
    size_t start = 0;
    while (true) {
      const size_t end = body.find('_', start);
      const size_t length = (end == std::string_view::npos ? body.size() : end) - start;
      if (length > 16) {
        return ignored_name("IGNORE-BAD-STEM");
      }
      if (end == std::string_view::npos) {
        break;
      }
      start = end + 1;
    }
    if (parsed->textureHash == texture::kWildcardTextureHash) {
      NameInfo info;
      info.kind = NameKind::Rejected;
      info.rule = "NAME-TEXTURE-WILDCARD";
      return info;
    }
  }

  NameInfo info;
  if (role.empty()) {
    info.kind = mip == 0 ? NameKind::Base : NameKind::BaseMip;
  } else {
    info.kind = mip == 0 ? NameKind::Companion : NameKind::CompanionMip;
  }
  info.key = canonical_key(*parsed);
  info.role = role;
  info.mip = mip;
  return info;
}

std::optional<DdsHeader> read_dds_header(std::span<const uint8_t> bytes, Violation* error) {
  const auto reject = [error](std::string_view rule, std::string detail) -> std::optional<DdsHeader> {
    if (error != nullptr) {
      *error = {std::string{rule}, "", std::move(detail)};
    }
    return std::nullopt;
  };
  if (bytes.size() < DdsHeaderSize) {
    return reject("DDS-HEADER", "shorter than the header");
  }
  if (read_u32(bytes, 0x00) != DdsMagic) {
    return reject("DDS-HEADER", "bad magic");
  }
  if (read_u32(bytes, 0x04) != 124) {
    return reject("DDS-HEADER", "header size is not 124");
  }
  if (read_u32(bytes, 0x4C) != 32) {
    return reject("DDS-HEADER", "pixel format size is not 32");
  }

  DdsHeader header;
  header.height = read_u32(bytes, 0x0C);
  header.width = read_u32(bytes, 0x10);
  if (header.width == 0 || header.height == 0) {
    return reject("DDS-HEADER", "zero dimension");
  }
  if ((read_u32(bytes, 0x70) & (Caps2Cubemap | Caps2Volume)) != 0) {
    return reject("DDS-HEADER", "cubemap or volume");
  }

  const uint32_t flags = read_u32(bytes, 0x08);
  const uint32_t mipCount = read_u32(bytes, 0x1C);
  const uint32_t pixelFlags = read_u32(bytes, 0x50);
  const uint32_t fourCC = read_u32(bytes, 0x54);
  const uint32_t caps = read_u32(bytes, 0x6C);
  header.dataOffset = DdsHeaderSize;
  if ((pixelFlags & PixelFormatFourCC) != 0) {
    header.fourCC = fourCC;
    header.style = DdsStyle::FourCC;
    if (fourCC == FourCCDx10) {
      if (bytes.size() < Dx10HeaderSize) {
        return reject("DDS-HEADER", "shorter than the DX10 header");
      }
      if (read_u32(bytes, 0x84) != 3 || read_u32(bytes, 0x8C) != 1 || (read_u32(bytes, 0x88) & MiscTextureCube) != 0) {
        return reject("DDS-HEADER", "not a single 2D texture");
      }
      header.code = read_u32(bytes, 0x80);
      header.style = pixelFlags == 0x44 && caps == 0 && mipCount == 0 ? DdsStyle::Gz2 : DdsStyle::Dx10;
      header.dataOffset = Dx10HeaderSize;
    } else if (fourCC == FourCCDxt1) {
      header.code = 71;
    } else if (fourCC == FourCCDxt5) {
      header.code = 77;
    } else if (fourCC == FourCCAti2) {
      header.code = 83;
    } else {
      return reject("DDS-HEADER", fmt::format("unknown FourCC {:08x}", fourCC));
    }
  } else if ((pixelFlags & PixelFormatRgb) != 0 && read_u32(bytes, 0x58) == 32) {
    const uint32_t r = read_u32(bytes, 0x5C);
    const uint32_t g = read_u32(bytes, 0x60);
    const uint32_t b = read_u32(bytes, 0x64);
    const uint32_t a = read_u32(bytes, 0x68);
    header.style = DdsStyle::Rgb;
    if (r == 0x000000FF && g == 0x0000FF00 && b == 0x00FF0000 && a == 0xFF000000) {
      header.code = 28;
    } else if (r == 0x00FF0000 && g == 0x0000FF00 && b == 0x000000FF && a == 0xFF000000) {
      header.code = 87;
    } else {
      return reject("DDS-HEADER", "unsupported RGB masks");
    }
  } else {
    return reject("DDS-HEADER", "unsupported pixel format");
  }

  const auto format = block_format(header.code);
  if (!format) {
    return reject("DDS-HEADER", fmt::format("unknown DXGI format {}", header.code));
  }
  if (mipCount > 1) {
    const bool hasMipFlags = (flags & FlagMipMapCount) != 0 && (caps & CapsComplex) != 0 && (caps & CapsMipMap) != 0;
    if (!hasMipFlags || mipCount > full_chain_length(header.width, header.height)) {
      return reject("DDS-HEADER", "invalid mip count");
    }
    header.mips = mipCount;
  }
  if (format->compressed && (header.width % 4 != 0 || header.height % 4 != 0)) {
    return reject("DDS-HEADER", "compressed level 0 is not a multiple of 4");
  }
  for (uint32_t level = 0; level < header.mips; ++level) {
    header.dataSize += level_bytes(*format, std::max(header.width >> level, 1u), std::max(header.height >> level, 1u));
  }
  if (bytes.size() < header.dataOffset + header.dataSize) {
    return reject("DDS-SIZE", fmt::format("{} bytes, expected {}", bytes.size(), header.dataOffset + header.dataSize));
  }
  return header;
}

std::map<std::string, DdsVerdict> check_dds_files(std::span<const DdsFile> files) {
  struct Member {
    const DdsFile* file = nullptr;
    uint32_t mip = 0;
  };
  std::map<std::string, DdsVerdict> verdicts;
  std::map<std::pair<std::string, std::string>, std::vector<Member>> groups;
  for (const DdsFile& file : files) {
    const NameInfo info = classify_name(file.name);
    if (info.kind == NameKind::Ignored) {
      verdicts[file.name] = {"ignored", info.rule, ""};
    } else if (info.kind == NameKind::Rejected) {
      verdicts[file.name] = {"invalid", info.rule, ""};
    } else {
      groups[{info.key, info.role}].push_back({&file, info.mip});
    }
  }

  const auto ignore_all = [&](std::span<const Member> members, std::string_view rule) {
    for (const Member& entry : members) {
      verdicts[entry.file->name] = {"ignored", std::string{rule}, ""};
    }
  };
  const auto base_size = [&](const std::string& key) -> std::pair<uint32_t, uint32_t> {
    if (const auto it = groups.find({key, ""}); it != groups.end()) {
      for (const Member& entry : it->second) {
        if (entry.mip != 0) {
          continue;
        }
        if (const auto header = read_dds_header(entry.file->bytes, nullptr)) {
          return {header->width, header->height};
        }
      }
    }
    const auto parsed = parse_key(key);
    return parsed ? std::pair{parsed->width, parsed->height} : std::pair{0u, 0u};
  };

  for (auto& [id, members] : groups) {
    const auto& [key, role] = id;
    std::stable_sort(members.begin(), members.end(), [](const Member& a, const Member& b) { return a.mip < b.mip; });
    if (members.front().mip != 0) {
      ignore_all(members, "IGNORE-ORPHAN-SIDECAR");
      continue;
    }

    const DdsFile& levelZero = *members.front().file;
    const std::span<const Member> sidecars = std::span<const Member>{members}.subspan(1);
    Violation error;
    const auto head = read_dds_header(levelZero.bytes, &error);
    std::string problem;
    if (!head) {
      problem = error.rule;
    } else if (!role.empty() && !role_accepts(role, head->code)) {
      problem = "DDS-ROLE-FORMAT";
    } else if (role == "delight") {
      const auto [baseWidth, baseHeight] = base_size(key);
      if (uint64_t{head->width} * 8 < baseWidth || uint64_t{head->height} * 8 < baseHeight) {
        problem = "DDS-DELIGHT-RES";
      }
    }
    if (!problem.empty()) {
      verdicts[levelZero.name] = {"invalid", problem, ""};
      ignore_all(sidecars, "IGNORE-AFTER-INVALID");
      continue;
    }
    verdicts[levelZero.name] = {"valid", "", ""};
    if (head->mips > 1) {
      ignore_all(sidecars, "IGNORE-INTERNAL-MIPS");
      continue;
    }

    uint32_t previousWidth = head->width;
    uint32_t previousHeight = head->height;
    uint32_t expectedMip = 1;
    bool ended = false;
    bool gap = false;
    for (const Member& sidecar : sidecars) {
      DdsVerdict& verdict = verdicts[sidecar.file->name];
      if (ended) {
        verdict = {"ignored", "IGNORE-AFTER-INVALID", ""};
        continue;
      }
      if (gap || sidecar.mip != expectedMip) {
        gap = true;
        verdict = {"ignored", "IGNORE-ORPHAN-SIDECAR", ""};
        continue;
      }
      const auto level = read_dds_header(sidecar.file->bytes, &error);
      std::string chain;
      if (!level) {
        problem = error.rule;
      } else if (level->code != head->code) {
        problem = "DDS-MIP-FORMAT";
      } else {
        const uint32_t expectedWidth = std::max(head->width >> sidecar.mip, 1u);
        const uint32_t expectedHeight = std::max(head->height >> sidecar.mip, 1u);
        if (level->width == expectedWidth && level->height == expectedHeight) {
          chain = "standard";
        } else if (level->width == previousWidth && level->height == previousHeight && level->width >= expectedWidth &&
                   level->height >= expectedHeight) {
          // Packs that stop downscaling at the native size repeat the last level instead.
          chain = "clamped";
        } else {
          problem = "DDS-MIP-SIZE";
        }
      }
      if (!problem.empty()) {
        verdict = {"invalid", problem, ""};
        ended = true;
        continue;
      }
      verdict = {"valid", "", chain};
      previousWidth = level->width;
      previousHeight = level->height;
      ++expectedMip;
    }
  }
  return verdicts;
}
} // namespace aurora::test::c1

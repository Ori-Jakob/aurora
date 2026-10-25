#include "manifest.hpp"

#include "key_text.hpp"

#include "../../logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aurora::render::material {
namespace {
constexpr Module Log{"aurora::render::material"};

constexpr int MaxJsonDepth = 64;
constexpr size_t LinearMemberLimit = 16;
constexpr std::string_view BaseSchemaName = "materials.schema.json";
constexpr std::string_view Layer1SchemaName = "materials.layer1.schema.json";

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

bool is_digit(char ch) noexcept { return ch >= '0' && ch <= '9'; }
bool is_lower(char ch) noexcept { return ch >= 'a' && ch <= 'z'; }
bool is_upper(char ch) noexcept { return ch >= 'A' && ch <= 'Z'; }
bool is_alnum(char ch) noexcept { return is_digit(ch) || is_lower(ch) || is_upper(ch); }
bool is_lower_hex(char ch) noexcept { return is_digit(ch) || (ch >= 'a' && ch <= 'f'); }
bool is_hex(char ch) noexcept { return is_lower_hex(ch) || (ch >= 'A' && ch <= 'F'); }

template <typename Pred>
bool all_of(std::string_view text, Pred pred) noexcept {
  return std::all_of(text.begin(), text.end(), pred);
}

bool match_color(std::string_view text) noexcept {
  return text.size() == 7 && text[0] == '#' && all_of(text.substr(1), is_hex);
}

bool match_grass_type_name(std::string_view text) noexcept {
  return !text.empty() && text.size() <= 32 && is_lower(text[0]) &&
         all_of(text.substr(1), [](char ch) { return is_lower(ch) || is_digit(ch) || ch == '_'; });
}

bool match_sha256(std::string_view text) noexcept { return text.size() == 64 && all_of(text, is_lower_hex); }

bool match_class(std::string_view text) noexcept {
  size_t start = 0;
  while (true) {
    const size_t slash = text.find('/', start);
    const std::string_view segment = text.substr(start, slash == std::string_view::npos ? slash : slash - start);
    if (segment.empty() || !is_lower(segment[0]) ||
        !all_of(segment.substr(1), [](char ch) { return is_lower(ch) || is_digit(ch) || ch == '_'; })) {
      return false;
    }
    if (slash == std::string_view::npos) {
      return true;
    }
    start = slash + 1;
  }
}

bool match_texture_wildcard(std::string_view text) noexcept {
  if (!text.starts_with("tex1_")) {
    return false;
  }
  size_t pos = 5;
  for (int dimension = 0; dimension < 2; ++dimension) {
    const size_t start = pos;
    while (pos < text.size() && is_digit(text[pos])) {
      ++pos;
    }
    if (pos == start) {
      return false;
    }
    if (dimension == 0) {
      if (pos >= text.size() || text[pos] != 'x') {
        return false;
      }
      ++pos;
    }
  }
  const auto wildcard = [](std::string_view rest) {
    return rest.starts_with("_$") && (rest.size() == 2 || rest[2] == '_');
  };
  const std::string_view rest = text.substr(pos);
  return wildcard(rest) || (rest.starts_with("_m") && wildcard(rest.substr(2)));
}

bool parse_dimension(std::string_view text, size_t& pos) noexcept {
  if (pos >= text.size() || text[pos] < '1' || text[pos] > '9') {
    return false;
  }
  ++pos;
  for (int digits = 1; digits < 4 && pos < text.size() && is_digit(text[pos]); ++digits) {
    ++pos;
  }
  return true;
}

std::optional<std::string_view> key_suffix(std::string_view text) noexcept {
  if (!text.starts_with("tex1_")) {
    return std::nullopt;
  }
  size_t pos = 5;
  if (!parse_dimension(text, pos) || pos >= text.size() || text[pos++] != 'x' || !parse_dimension(text, pos) ||
      pos >= text.size() || text[pos++] != '_' || text.size() - pos < 17 ||
      !all_of(text.substr(pos, 16), is_lower_hex) || text[pos + 16] != '_') {
    return std::nullopt;
  }
  return text.substr(pos + 17);
}

bool is_direct_format(std::string_view text) noexcept {
  return (text.size() == 1 && text[0] >= '0' && text[0] <= '6') || text == "14" || text == "65" || text == "67" ||
         text == "70" || text == "78";
}

bool is_palette_format(std::string_view text) noexcept { return text == "8" || text == "9" || text == "10"; }

bool is_tlut_part(std::string_view text) noexcept {
  return text == "$" || (text.size() == 16 && all_of(text, is_lower_hex));
}

bool match_key_syntax(std::string_view text) noexcept {
  const auto suffix = key_suffix(text);
  if (!suffix) {
    return false;
  }
  if (is_direct_format(*suffix)) {
    return true;
  }
  const size_t split = suffix->find('_');
  return split != std::string_view::npos && is_tlut_part(suffix->substr(0, split)) &&
         is_palette_format(suffix->substr(split + 1));
}

bool match_tlut_format(std::string_view text) noexcept {
  const auto suffix = key_suffix(text);
  if (!suffix) {
    return false;
  }
  if (is_palette_format(*suffix)) {
    return true;
  }
  const size_t split = suffix->find('_');
  return split != std::string_view::npos && is_tlut_part(suffix->substr(0, split)) &&
         is_direct_format(suffix->substr(split + 1));
}

bool is_path_lead(char ch) noexcept { return is_alnum(ch) || ch == '_' || ch == '$' || ch == '-'; }
bool is_path_char(char ch) noexcept { return is_path_lead(ch) || ch == '.'; }
char ascii_lower(char ch) noexcept { return is_upper(ch) ? static_cast<char>(ch - 'A' + 'a') : ch; }

bool match_map_path(std::string_view text) noexcept {
  size_t start = 0;
  while (true) {
    const size_t slash = text.find('/', start);
    const std::string_view segment = text.substr(start, slash == std::string_view::npos ? slash : slash - start);
    if (segment.empty() || !is_path_lead(segment[0])) {
      return false;
    }
    if (slash != std::string_view::npos) {
      if (!all_of(segment.substr(1), is_path_char)) {
        return false;
      }
      start = slash + 1;
      continue;
    }
    if (segment.size() < 5) {
      return false;
    }
    const std::string_view extension = segment.substr(segment.size() - 4);
    return extension[0] == '.' && ascii_lower(extension[1]) == 'd' && ascii_lower(extension[2]) == 'd' &&
           ascii_lower(extension[3]) == 's' && all_of(segment.substr(1, segment.size() - 5), is_path_char);
  }
}

bool is_source_char(char ch) noexcept { return is_alnum(ch) || ch == '_' || ch == '.' || ch == '-'; }

bool match_layer1_source(std::string_view text) noexcept {
  size_t space = text.find(' ');
  const std::string_view head = text.substr(0, space);
  const size_t colon = head.find(':');
  const std::string_view path = head.substr(0, colon);
  size_t start = 0;
  while (true) {
    const size_t slash = path.find('/', start);
    const std::string_view segment = path.substr(start, slash == std::string_view::npos ? slash : slash - start);
    if (segment.empty() || !all_of(segment, is_source_char)) {
      return false;
    }
    if (slash == std::string_view::npos) {
      break;
    }
    start = slash + 1;
  }
  if (colon != std::string_view::npos) {
    const std::string_view tag = head.substr(colon + 1);
    if (tag.empty() || !all_of(tag, is_source_char)) {
      return false;
    }
  }
  while (space != std::string_view::npos) {
    const size_t next = text.find(' ', space + 1);
    const std::string_view qualifier = text.substr(space + 1, next == std::string_view::npos ? next : next - space - 1);
    const size_t split = qualifier.find(':');
    if (split == std::string_view::npos) {
      return false;
    }
    const std::string_view name = qualifier.substr(0, split);
    const std::string_view value = qualifier.substr(split + 1);
    if ((name != "mat" && name != "blend" && name != "texmap" && name != "room" && name != "stage") || value.empty() ||
        !all_of(value, [](char ch) { return is_source_char(ch) || ch == ','; })) {
      return false;
    }
    space = next;
  }
  return true;
}

using Matcher = bool (*)(std::string_view) noexcept;

struct PatternMatcher {
  std::string_view source;
  Matcher match;
};

constexpr std::array<PatternMatcher, 9> PatternMatchers{{
    {"^#[0-9A-Fa-f]{6}$", &match_color},
    {"^[a-z][a-z0-9_]{0,31}$", &match_grass_type_name},
    {"^tex1_[0-9]+x[0-9]+(_m)?_\\$(_|$)", &match_texture_wildcard},
    {"^tex1_[1-9][0-9]{0,3}x[1-9][0-9]{0,3}_[0-9a-f]{16}(_(8|9|10)|_([0-9a-f]{16}|\\$)_([0-6]|14|65|67|70|78))$",
     &match_tlut_format},
    {"^tex1_[1-9][0-9]{0,3}x[1-9][0-9]{0,3}_[0-9a-f]{16}(_([0-9a-f]{16}|\\$)_(8|9|10)|_([0-6]|14|65|67|70|78))$",
     &match_key_syntax},
    {"^([A-Za-z0-9_$-][A-Za-z0-9_.$-]*/)*[A-Za-z0-9_$-][A-Za-z0-9_.$-]*\\.[Dd][Dd][Ss]$", &match_map_path},
    {"^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*$", &match_class},
    {"^[0-9a-f]{64}$", &match_sha256},
    {"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*(:[A-Za-z0-9_.-]+)?( (mat|blend|texmap|room|stage):[A-Za-z0-9_.,-]+)*$",
     &match_layer1_source},
}};

Matcher find_matcher(std::string_view source) noexcept {
  for (const PatternMatcher& entry : PatternMatchers) {
    if (entry.source == source) {
      return entry.match;
    }
  }
  return nullptr;
}

std::string_view text_of(yyjson_val* val) noexcept { return {unsafe_yyjson_get_str(val), unsafe_yyjson_get_len(val)}; }

yyjson_val* member(yyjson_val* object, std::string_view key) noexcept {
  return yyjson_obj_getn(object, key.data(), key.size());
}

size_t code_points(std::string_view text) noexcept {
  return static_cast<size_t>(
      std::count_if(text.begin(), text.end(), [](char ch) { return (static_cast<unsigned char>(ch) & 0xC0) != 0x80; }));
}

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

void append_token(std::string& pointer, std::string_view token) {
  pointer += '/';
  for (const char ch : token) {
    if (ch == '~') {
      pointer += "~0";
    } else if (ch == '/') {
      pointer += "~1";
    } else {
      pointer += ch;
    }
  }
}

struct SchemaNode {
  yyjson_val* schema = nullptr;
  yyjson_val* document = nullptr;
};

struct SchemaSet {
  json::Doc base;
  json::Doc layer1;
  yyjson_val* baseRoot = nullptr;
  yyjson_val* layer1Root = nullptr;
  std::vector<std::pair<yyjson_val*, Matcher>> patterns;
  bool ready = false;

  yyjson_val* root(std::string_view name) const noexcept {
    if (name == BaseSchemaName) {
      return baseRoot;
    }
    if (name == Layer1SchemaName) {
      return layer1Root;
    }
    return nullptr;
  }

  std::optional<SchemaNode> resolve(yyjson_val* document, std::string_view ref) const {
    const size_t hash = ref.find('#');
    const std::string_view location = ref.substr(0, hash);
    const std::string_view fragment = hash == std::string_view::npos ? std::string_view{} : ref.substr(hash + 1);
    yyjson_val* top = document;
    if (!location.empty()) {
      const size_t slash = location.rfind('/');
      top = root(slash == std::string_view::npos ? location : location.substr(slash + 1));
      if (top == nullptr) {
        return std::nullopt;
      }
    }
    yyjson_val* node = top;
    if (!fragment.empty()) {
      if (fragment.front() != '/') {
        return std::nullopt;
      }
      size_t start = 1;
      while (node != nullptr) {
        const size_t end = fragment.find('/', start);
        node = member(node, unescape_pointer_token(fragment.substr(start, end - start)));
        if (end == std::string_view::npos) {
          break;
        }
        start = end + 1;
      }
      if (node == nullptr) {
        return std::nullopt;
      }
    }
    return SchemaNode{node, top};
  }

  Matcher matcher(yyjson_val* pattern) const noexcept {
    for (const auto& [source, match] : patterns) {
      if (source == pattern) {
        return match;
      }
    }
    return nullptr;
  }

  bool self_check(yyjson_val* document, yyjson_val* schema) {
    if (yyjson_is_bool(schema)) {
      return true;
    }
    if (!yyjson_is_obj(schema)) {
      return false;
    }
    size_t index = 0;
    size_t count = 0;
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(schema);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      yyjson_val* value = yyjson_obj_iter_get_val(key);
      const std::string_view keyword = text_of(key);
      if (std::find(SchemaKeywords.begin(), SchemaKeywords.end(), keyword) == SchemaKeywords.end()) {
        return false;
      }
      if (keyword == "$defs" || keyword == "properties") {
        yyjson_val* child = nullptr;
        yyjson_obj_iter children = yyjson_obj_iter_with(value);
        while ((child = yyjson_obj_iter_next(&children)) != nullptr) {
          if (!self_check(document, yyjson_obj_iter_get_val(child))) {
            return false;
          }
        }
      } else if (keyword == "additionalProperties" || keyword == "propertyNames" || keyword == "items" ||
                 keyword == "not") {
        if (!self_check(document, value)) {
          return false;
        }
      } else if (keyword == "allOf") {
        yyjson_val* child = nullptr;
        yyjson_arr_foreach(value, index, count, child) {
          if (!self_check(document, child)) {
            return false;
          }
        }
      } else if (keyword == "$ref") {
        if (!yyjson_is_str(value) || !resolve(document, text_of(value))) {
          return false;
        }
      } else if (keyword == "pattern") {
        const Matcher match = yyjson_is_str(value) ? find_matcher(text_of(value)) : nullptr;
        if (match == nullptr) {
          return false;
        }
        patterns.emplace_back(value, match);
      }
    }
    return true;
  }
};

const SchemaSet& schemas() {
  static SchemaSet s_schemas;
  static std::once_flag s_once;
  std::call_once(s_once, [] {
    s_schemas.base = json::read(embedded_schema(BaseSchemaName));
    s_schemas.layer1 = json::read(embedded_schema(Layer1SchemaName));
    if (s_schemas.base && s_schemas.layer1) {
      s_schemas.baseRoot = yyjson_doc_get_root(s_schemas.base.get());
      s_schemas.layer1Root = yyjson_doc_get_root(s_schemas.layer1.get());
      s_schemas.ready = s_schemas.self_check(s_schemas.baseRoot, s_schemas.baseRoot) &&
                        s_schemas.self_check(s_schemas.layer1Root, s_schemas.layer1Root);
    }
    if (!s_schemas.ready) {
      Log.error("Texture material schemas failed their self-check; every manifest will be rejected");
    }
  });
  return s_schemas;
}

bool is_integer(yyjson_val* value) noexcept {
  if (yyjson_is_int(value)) {
    return true;
  }
  if (!yyjson_is_real(value)) {
    return false;
  }
  const double number = yyjson_get_real(value);
  return std::isfinite(number) && std::floor(number) == number;
}

bool matches_type_name(std::string_view name, yyjson_val* value) noexcept {
  if (name == "integer") {
    return is_integer(value);
  }
  switch (yyjson_get_type(value)) {
  case YYJSON_TYPE_NULL:
    return name == "null";
  case YYJSON_TYPE_BOOL:
    return name == "boolean";
  case YYJSON_TYPE_NUM:
    return name == "number";
  case YYJSON_TYPE_STR:
    return name == "string";
  case YYJSON_TYPE_ARR:
    return name == "array";
  case YYJSON_TYPE_OBJ:
    return name == "object";
  default:
    return false;
  }
}

bool matches_type(yyjson_val* type, yyjson_val* value) noexcept {
  if (yyjson_is_str(type)) {
    return matches_type_name(text_of(type), value);
  }
  size_t index = 0;
  size_t count = 0;
  yyjson_val* name = nullptr;
  yyjson_arr_foreach(type, index, count, name) {
    if (yyjson_is_str(name) && matches_type_name(text_of(name), value)) {
      return true;
    }
  }
  return false;
}

bool has_duplicate_items(yyjson_val* array) {
  std::vector<yyjson_val*> items;
  items.reserve(yyjson_arr_size(array));
  size_t index = 0;
  size_t count = 0;
  yyjson_val* item = nullptr;
  yyjson_arr_foreach(array, index, count, item) { items.push_back(item); }
  for (size_t i = 0; i < items.size(); ++i) {
    for (size_t j = i + 1; j < items.size(); ++j) {
      if (json_equal(items[i], items[j])) {
        return true;
      }
    }
  }
  return false;
}

double number_of(yyjson_val* value) noexcept { return yyjson_get_num(value); }

struct Evaluator {
  const SchemaSet& schemas;
  std::string pointer;
  Violation violation;

  bool fail(yyjson_val* annotation, std::string_view fallback, std::string_view keyword) {
    violation.rule = yyjson_is_str(annotation) ? text_of(annotation) : fallback;
    violation.pointer = pointer;
    violation.detail = keyword;
    return false;
  }

  bool fail_at(yyjson_val* annotation, std::string_view fallback, std::string_view token, std::string_view keyword) {
    const size_t base = pointer.size();
    append_token(pointer, token);
    fail(annotation, fallback, keyword);
    pointer.resize(base);
    return false;
  }

  bool eval_child(yyjson_val* schema, yyjson_val* document, yyjson_val* instance, std::string_view token) {
    const size_t base = pointer.size();
    append_token(pointer, token);
    const bool ok = eval(schema, document, instance);
    pointer.resize(base);
    return ok;
  }

  bool eval(yyjson_val* schema, yyjson_val* document, yyjson_val* instance) {
    if (yyjson_is_bool(schema)) {
      if (!yyjson_get_bool(schema)) {
        violation = {"MANIFEST-NOT", pointer, "false schema"};
        return false;
      }
      return true;
    }
    if (!yyjson_is_obj(schema)) {
      return true;
    }
    yyjson_val* annotation = member(schema, "x-c1-rule");

    if (yyjson_val* type = member(schema, "type"); type != nullptr && !matches_type(type, instance)) {
      return fail(annotation, "MANIFEST-TYPE", "type");
    }
    if (yyjson_val* value = member(schema, "const"); value != nullptr && !json_equal(value, instance)) {
      return fail(annotation, "MANIFEST-ENUM", "const");
    }
    if (yyjson_val* values = member(schema, "enum"); values != nullptr) {
      bool found = false;
      size_t index = 0;
      size_t count = 0;
      yyjson_val* value = nullptr;
      yyjson_arr_foreach(values, index, count, value) {
        if (json_equal(value, instance)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return fail(annotation, "MANIFEST-ENUM", "enum");
      }
    }
    if (yyjson_val* ref = member(schema, "$ref"); ref != nullptr) {
      const auto target = yyjson_is_str(ref) ? schemas.resolve(document, text_of(ref)) : std::nullopt;
      if (!target) {
        violation = {"", pointer, "unresolved $ref"};
        return false;
      }
      if (!eval(target->schema, target->document, instance)) {
        return false;
      }
    }
    if (yyjson_val* branches = member(schema, "allOf"); branches != nullptr) {
      size_t index = 0;
      size_t count = 0;
      yyjson_val* branch = nullptr;
      yyjson_arr_foreach(branches, index, count, branch) {
        if (!eval(branch, document, instance)) {
          return false;
        }
      }
    }
    if (yyjson_val* negated = member(schema, "not"); negated != nullptr && eval(negated, document, instance)) {
      return fail(annotation, "MANIFEST-NOT", "not");
    }

    if (yyjson_is_str(instance)) {
      const auto length = static_cast<double>(code_points(text_of(instance)));
      if (yyjson_val* limit = member(schema, "minLength"); limit != nullptr && length < number_of(limit)) {
        return fail(annotation, "MANIFEST-PATTERN", "minLength");
      }
      if (yyjson_val* limit = member(schema, "maxLength"); limit != nullptr && length > number_of(limit)) {
        return fail(annotation, "MANIFEST-PATTERN", "maxLength");
      }
      if (yyjson_val* source = member(schema, "pattern"); source != nullptr) {
        const Matcher match = schemas.matcher(source);
        if (match == nullptr || !match(text_of(instance))) {
          return fail(annotation, "MANIFEST-PATTERN", "pattern");
        }
      }
    }
    if (yyjson_is_num(instance)) {
      const double value = yyjson_get_num(instance);
      if (yyjson_val* limit = member(schema, "minimum"); limit != nullptr && value < number_of(limit)) {
        return fail(annotation, "MANIFEST-RANGE", "minimum");
      }
      if (yyjson_val* limit = member(schema, "maximum"); limit != nullptr && value > number_of(limit)) {
        return fail(annotation, "MANIFEST-RANGE", "maximum");
      }
      if (yyjson_val* limit = member(schema, "exclusiveMinimum"); limit != nullptr && value <= number_of(limit)) {
        return fail(annotation, "MANIFEST-RANGE", "exclusiveMinimum");
      }
      if (yyjson_val* limit = member(schema, "exclusiveMaximum"); limit != nullptr && value >= number_of(limit)) {
        return fail(annotation, "MANIFEST-RANGE", "exclusiveMaximum");
      }
    }
    if (yyjson_is_arr(instance)) {
      const auto count = static_cast<double>(yyjson_arr_size(instance));
      if (yyjson_val* limit = member(schema, "minItems"); limit != nullptr && count < number_of(limit)) {
        return fail(annotation, "MANIFEST-ARRAY-LENGTH", "minItems");
      }
      if (yyjson_val* limit = member(schema, "maxItems"); limit != nullptr && count > number_of(limit)) {
        return fail(annotation, "MANIFEST-ARRAY-LENGTH", "maxItems");
      }
      if (yyjson_val* unique = member(schema, "uniqueItems");
          unique != nullptr && yyjson_is_true(unique) && has_duplicate_items(instance)) {
        return fail(annotation, "MANIFEST-DUPLICATE-ITEM", "uniqueItems");
      }
      if (yyjson_val* items = member(schema, "items"); items != nullptr) {
        size_t index = 0;
        size_t total = 0;
        yyjson_val* item = nullptr;
        yyjson_arr_foreach(instance, index, total, item) {
          if (!eval_child(items, document, item, std::to_string(index))) {
            return false;
          }
        }
      }
    }
    if (!yyjson_is_obj(instance)) {
      return true;
    }
    if (yyjson_val* required = member(schema, "required"); required != nullptr) {
      size_t index = 0;
      size_t count = 0;
      yyjson_val* name = nullptr;
      yyjson_arr_foreach(required, index, count, name) {
        if (yyjson_is_str(name) && member(instance, text_of(name)) == nullptr) {
          return fail_at(annotation, "MANIFEST-REQUIRED", text_of(name), "required");
        }
      }
    }
    yyjson_val* key = nullptr;
    if (yyjson_val* names = member(schema, "propertyNames"); names != nullptr) {
      yyjson_obj_iter iter = yyjson_obj_iter_with(instance);
      while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
        if (!eval_child(names, document, key, text_of(key))) {
          return false;
        }
      }
    }
    yyjson_val* properties = member(schema, "properties");
    if (properties != nullptr) {
      yyjson_obj_iter iter = yyjson_obj_iter_with(instance);
      while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
        if (yyjson_val* property = member(properties, text_of(key)); property != nullptr) {
          if (!eval_child(property, document, yyjson_obj_iter_get_val(key), text_of(key))) {
            return false;
          }
        }
      }
    }
    if (yyjson_val* additional = member(schema, "additionalProperties"); additional != nullptr) {
      yyjson_obj_iter iter = yyjson_obj_iter_with(instance);
      while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
        if (properties != nullptr && member(properties, text_of(key)) != nullptr) {
          continue;
        }
        if (!yyjson_is_bool(additional)) {
          if (!eval_child(additional, document, yyjson_obj_iter_get_val(key), text_of(key))) {
            return false;
          }
        } else if (!yyjson_get_bool(additional)) {
          return fail_at(annotation, "MANIFEST-UNKNOWN-MEMBER", text_of(key), "additionalProperties");
        }
      }
    }
    return true;
  }
};

struct StructureWalk {
  std::string pointer;
  Violation violation;

  bool walk(yyjson_val* value, int depth) {
    const bool array = yyjson_is_arr(value);
    if (!array && !yyjson_is_obj(value)) {
      return true;
    }
    if (depth > MaxJsonDepth) {
      violation = {"JSON-SYNTAX", "", "nesting deeper than 64"};
      return false;
    }
    const size_t base = pointer.size();
    if (array) {
      size_t index = 0;
      size_t count = 0;
      yyjson_val* item = nullptr;
      yyjson_arr_foreach(value, index, count, item) {
        append_token(pointer, std::to_string(index));
        if (!walk(item, depth + 1)) {
          return false;
        }
        pointer.resize(base);
      }
      return true;
    }
    const size_t size = yyjson_obj_size(value);
    std::vector<std::string_view> earlier;
    std::unordered_set<std::string_view> seen;
    if (size <= LinearMemberLimit) {
      earlier.reserve(size);
    } else {
      seen.reserve(size);
    }
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(value);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      const std::string_view name = text_of(key);
      append_token(pointer, name);
      bool repeated = false;
      if (size <= LinearMemberLimit) {
        repeated = std::find(earlier.begin(), earlier.end(), name) != earlier.end();
        earlier.push_back(name);
      } else {
        repeated = !seen.insert(name).second;
      }
      if (repeated) {
        violation = {"JSON-DUPLICATE-MEMBER", pointer, "duplicate member"};
        return false;
      }
      if (!walk(yyjson_obj_iter_get_val(key), depth + 1)) {
        return false;
      }
      pointer.resize(base);
    }
    return true;
  }
};

bool is_reserved_key(std::string_view key) {
  const auto parsed = parse_key_text(key);
  if (!parsed) {
    return false;
  }
  if (parsed->textureHash == texture::kWildcardTextureHash) {
    return true;
  }
  return parsed->hasTlut && parsed->tlutHash == texture::kWildcardTlutHash && key.find("_$_") == std::string_view::npos;
}

std::optional<Violation> semantic_violation(yyjson_val* manifest) {
  yyjson_val* key = nullptr;
  if (yyjson_val* types = member(manifest, "grassTypes"); types != nullptr) {
    yyjson_obj_iter iter = yyjson_obj_iter_with(types);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      yyjson_val* type = yyjson_obj_iter_get_val(key);
      const std::string at = pointer_join("/grassTypes", text_of(key));
      for (const std::string_view field : {std::string_view{"height"}, std::string_view{"width"}}) {
        yyjson_val* range = member(type, field);
        if (range != nullptr && yyjson_arr_size(range) == 2 &&
            number_of(yyjson_arr_get(range, 0)) > number_of(yyjson_arr_get(range, 1))) {
          return Violation{"RANGE-ORDER", pointer_join(at, field), "lower bound above upper bound"};
        }
      }
      if (yyjson_val* lods = member(type, "lodVerts"); lods != nullptr) {
        yyjson_val* previous = nullptr;
        size_t index = 0;
        size_t count = 0;
        yyjson_val* lod = nullptr;
        yyjson_arr_foreach(lods, index, count, lod) {
          if (previous != nullptr && number_of(lod) >= number_of(previous)) {
            return Violation{"LODVERTS-ORDER", pointer_join(at, "lodVerts"), "not strictly decreasing"};
          }
          previous = lod;
        }
      }
      if (yyjson_val* colorFrom = member(type, "colorFrom");
          colorFrom != nullptr && is_reserved_key(text_of(colorFrom))) {
        return Violation{"KEY-RESERVED-HASH", pointer_join(at, "colorFrom"), std::string{text_of(colorFrom)}};
      }
    }
  }
  if (yyjson_val* materials = member(manifest, "materials"); materials != nullptr) {
    yyjson_obj_iter iter = yyjson_obj_iter_with(materials);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      if (is_reserved_key(text_of(key))) {
        return Violation{"KEY-RESERVED-HASH", pointer_join("/materials", text_of(key)), std::string{text_of(key)}};
      }
    }
  }
  return std::nullopt;
}
} // namespace

bool json_equal(yyjson_val* a, yyjson_val* b) noexcept {
  const yyjson_type type = yyjson_get_type(a);
  if (type != yyjson_get_type(b)) {
    return false;
  }
  switch (type) {
  case YYJSON_TYPE_NULL:
    return true;
  case YYJSON_TYPE_BOOL:
    return yyjson_get_bool(a) == yyjson_get_bool(b);
  case YYJSON_TYPE_NUM:
    return yyjson_get_num(a) == yyjson_get_num(b);
  case YYJSON_TYPE_STR:
    return text_of(a) == text_of(b);
  case YYJSON_TYPE_ARR: {
    if (yyjson_arr_size(a) != yyjson_arr_size(b)) {
      return false;
    }
    yyjson_arr_iter left = yyjson_arr_iter_with(a);
    yyjson_arr_iter right = yyjson_arr_iter_with(b);
    yyjson_val* x = nullptr;
    while ((x = yyjson_arr_iter_next(&left)) != nullptr) {
      if (!json_equal(x, yyjson_arr_iter_next(&right))) {
        return false;
      }
    }
    return true;
  }
  case YYJSON_TYPE_OBJ: {
    if (yyjson_obj_size(a) != yyjson_obj_size(b)) {
      return false;
    }
    yyjson_obj_iter iter = yyjson_obj_iter_with(a);
    yyjson_val* key = nullptr;
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      yyjson_val* other = member(b, text_of(key));
      if (other == nullptr || !json_equal(yyjson_obj_iter_get_val(key), other)) {
        return false;
      }
    }
    return true;
  }
  default:
    return false;
  }
}

std::string pointer_join(std::string_view parent, std::string_view token) {
  std::string out{parent};
  append_token(out, token);
  return out;
}

bool schema_ready() noexcept { return schemas().ready; }

ManifestCheck check_manifest(std::string_view text, texture::MaterialLayer layer) noexcept {
  const SchemaSet& set = schemas();
  if (!set.ready) {
    return {nullptr, Violation{"", "", "schema self-check failed"}};
  }
  std::string error;
  json::Doc doc = json::read(text, &error);
  if (!doc) {
    return {nullptr, Violation{"JSON-SYNTAX", "", std::move(error)}};
  }
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  StructureWalk structure;
  if (!structure.walk(root, 1)) {
    return {nullptr, std::move(structure.violation)};
  }
  yyjson_val* schema = layer == texture::MaterialLayer::Game ? set.layer1Root : set.baseRoot;
  Evaluator evaluator{set};
  if (!evaluator.eval(schema, schema, root)) {
    return {nullptr, std::move(evaluator.violation)};
  }
  if (auto violation = semantic_violation(root)) {
    return {nullptr, std::move(violation)};
  }
  return {std::make_shared<const json::Doc>(std::move(doc)), std::nullopt};
}

namespace testing {
bool match_schema_pattern(std::string_view source, std::string_view text, bool* bound) noexcept {
  const Matcher match = find_matcher(source);
  if (bound != nullptr) {
    *bound = match != nullptr;
  }
  return match != nullptr && match(text);
}
} // namespace testing
} // namespace aurora::render::material

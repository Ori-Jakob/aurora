#pragma once

#include "../json.hpp"

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::material {
struct Violation {
  std::string rule;
  std::string pointer;
  std::string detail;
};

struct ManifestCheck {
  std::shared_ptr<const json::Doc> doc;
  std::optional<Violation> violation;
};

ManifestCheck check_manifest(std::string_view text, texture::MaterialLayer layer) noexcept;
bool json_equal(yyjson_val* a, yyjson_val* b) noexcept;
std::string pointer_join(std::string_view parent, std::string_view token);
std::string_view embedded_schema(std::string_view name) noexcept;
bool schema_ready() noexcept;

namespace testing {
bool match_schema_pattern(std::string_view source, std::string_view text, bool* bound) noexcept;
} // namespace testing
} // namespace aurora::render::material

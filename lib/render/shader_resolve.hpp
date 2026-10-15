#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::shaders {
struct LineOrigin {
  uint32_t file = 0;
  uint32_t line = 0;
};

struct ResolvedSource {
  std::string text;
  std::vector<std::string> files;
  std::vector<LineOrigin> lines;
};

using SourceLookup = std::function<std::optional<std::string_view>(std::string_view id)>;

bool is_valid_shader_id(std::string_view id) noexcept;
bool is_valid_define(std::string_view name) noexcept;
std::string normalize_source(std::string_view source);
bool resolve_source(std::string_view root, std::span<const std::string_view> defines, const SourceLookup& lookup,
                    ResolvedSource& out, std::string& error);
std::string describe_line(const ResolvedSource& source, uint32_t outputLine, uint32_t column);
} // namespace aurora::render::shaders

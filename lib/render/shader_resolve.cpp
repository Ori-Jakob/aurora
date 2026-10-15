#include "shader_resolve.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace aurora::render::shaders {
namespace {
constexpr size_t MaxImportDepth = 32;
constexpr size_t MaxBranchDepth = 16;
constexpr std::string_view Bom = "\xEF\xBB\xBF";
constexpr std::string_view DirectivePrefix = "//#";

constexpr bool is_blank(char c) noexcept { return c == ' ' || c == '\t'; }

std::string_view trim_left(std::string_view text) noexcept {
  size_t start = 0;
  while (start < text.size() && is_blank(text[start])) {
    ++start;
  }
  return text.substr(start);
}

std::string_view trim(std::string_view text) noexcept {
  text = trim_left(text);
  size_t end = text.size();
  while (end > 0 && is_blank(text[end - 1])) {
    --end;
  }
  return text.substr(0, end);
}

std::string_view take_token(std::string_view& text) noexcept {
  text = trim_left(text);
  size_t end = 0;
  while (end < text.size() && !is_blank(text[end])) {
    ++end;
  }
  const std::string_view token = text.substr(0, end);
  text.remove_prefix(end);
  return token;
}

bool only_comment(std::string_view rest) noexcept {
  rest = trim_left(rest);
  return rest.empty() || rest.starts_with("//");
}

// WGSL rejects these after the first declaration, and imported files may carry them
bool is_global_directive(std::string_view trimmed) noexcept {
  return trimmed.ends_with(';') && (trimmed.starts_with("enable ") || trimmed.starts_with("requires ") ||
                                    trimmed.starts_with("diagnostic(") || trimmed.starts_with("diagnostic ("));
}

struct Branch {
  uint32_t line = 0;
  bool parentActive = true;
  bool condition = false;
  bool active = true;
  bool seenElse = false;
};

struct Frame {
  uint32_t file = 0;
  uint32_t line = 0;
};

class Resolver {
public:
  Resolver(std::span<const std::string_view> defines, const SourceLookup& lookup, ResolvedSource& out) noexcept
  : m_defines(defines), m_lookup(lookup), m_out(out) {}

  bool include(std::string_view id, std::string_view text, std::string& error) {
    const auto file = static_cast<uint32_t>(m_out.files.size());
    m_out.files.emplace_back(id);
    m_stack.push_back({file, 0});
    std::vector<Branch> branches;
    size_t position = 0;
    uint32_t lineNumber = 0;
    while (position < text.size()) {
      size_t end = text.find('\n', position);
      if (end == std::string_view::npos) {
        end = text.size();
      }
      std::string_view line = text.substr(position, end - position);
      position = end + 1;
      if (line.ends_with('\r')) {
        line.remove_suffix(1);
      }
      m_stack.back().line = ++lineNumber;
      const bool active = branches.empty() || branches.back().active;
      const std::string_view lead = trim_left(line);
      if (lead.starts_with(DirectivePrefix)) {
        if (!directive(lead.substr(DirectivePrefix.size()), active, branches, error)) {
          return false;
        }
        continue;
      }
      if (active) {
        emit(line, file, lineNumber);
      }
    }
    if (!branches.empty()) {
      m_stack.back().line = branches.back().line;
      return fail("unterminated #if", error);
    }
    m_stack.pop_back();
    return true;
  }

  void finish() {
    while (!m_lines.empty() && (m_body.size() == 1 || m_body[m_body.size() - 2] == '\n')) {
      m_body.pop_back();
      m_lines.pop_back();
    }
    m_out.text.clear();
    m_out.lines.clear();
    for (const auto& [text, origin] : m_hoisted) {
      m_out.text.append(text);
      m_out.text.push_back('\n');
      m_out.lines.push_back(origin);
    }
    m_out.text.append(m_body);
    m_out.lines.insert(m_out.lines.end(), m_lines.begin(), m_lines.end());
  }

private:
  bool fail(std::string_view message, std::string& error) const {
    const Frame& frame = m_stack.back();
    error = fmt::format("{}.wgsl:{}: {}", m_out.files[frame.file], frame.line, message);
    return false;
  }

  [[nodiscard]] bool defined(std::string_view name) const noexcept {
    return std::ranges::binary_search(m_defines, name);
  }

  bool directive(std::string_view rest, bool active, std::vector<Branch>& branches, std::string& error) {
    size_t nameEnd = 0;
    while (nameEnd < rest.size() && !is_blank(rest[nameEnd])) {
      ++nameEnd;
    }
    const std::string_view name = rest.substr(0, nameEnd);
    std::string_view args = rest.substr(nameEnd);
    if (name == "import") {
      const std::string_view target = take_token(args);
      if (!only_comment(args)) {
        return fail("unexpected text after directive", error);
      }
      return !active || import(target, error);
    }
    if (name == "if") {
      std::string_view define = take_token(args);
      if (!only_comment(args)) {
        return fail("unexpected text after directive", error);
      }
      const bool negated = define.starts_with('!');
      if (negated) {
        define.remove_prefix(1);
      }
      if (!is_valid_define(define)) {
        return fail(fmt::format("invalid define '{}'", define), error);
      }
      if (branches.size() >= MaxBranchDepth) {
        return fail(fmt::format("#if nested deeper than {}", MaxBranchDepth), error);
      }
      const bool condition = defined(define) != negated;
      branches.push_back({
          .line = m_stack.back().line,
          .parentActive = active,
          .condition = condition,
          .active = active && condition,
      });
      return true;
    }
    if (name == "else" || name == "endif") {
      if (!only_comment(args)) {
        return fail("unexpected text after directive", error);
      }
      if (branches.empty()) {
        return fail(name == "else" ? "#else without #if" : "#endif without #if", error);
      }
      if (name == "endif") {
        branches.pop_back();
        return true;
      }
      Branch& branch = branches.back();
      if (branch.seenElse) {
        return fail("duplicate #else", error);
      }
      branch.seenElse = true;
      branch.active = branch.parentActive && !branch.condition;
      return true;
    }
    return fail(fmt::format("unknown directive '{}{}'", DirectivePrefix, name), error);
  }

  bool import(std::string_view id, std::string& error) {
    if (!is_valid_shader_id(id)) {
      return fail(fmt::format("invalid shader id '{}'", id), error);
    }
    for (size_t i = 0; i < m_stack.size(); ++i) {
      if (m_out.files[m_stack[i].file] != id) {
        continue;
      }
      std::string chain;
      for (size_t j = i; j < m_stack.size(); ++j) {
        chain += m_out.files[m_stack[j].file];
        chain += " -> ";
      }
      chain += id;
      const Frame& start = m_stack[i];
      error = fmt::format("{}.wgsl:{}: import cycle: {}", m_out.files[start.file], start.line, chain);
      return false;
    }
    if (std::ranges::find(m_out.files, id) != m_out.files.end()) {
      return true;
    }
    if (m_stack.size() > MaxImportDepth) {
      return fail(fmt::format("imports nested deeper than {}", MaxImportDepth), error);
    }
    const std::optional<std::string_view> text = m_lookup(id);
    if (!text) {
      return fail(fmt::format("unknown shader '{}'", id), error);
    }
    return include(id, *text, error);
  }

  void emit(std::string_view line, uint32_t file, uint32_t lineNumber) {
    const std::string_view trimmed = trim(line);
    if (is_global_directive(trimmed)) {
      const bool seen = std::ranges::any_of(m_hoisted, [&](const auto& entry) { return entry.first == trimmed; });
      if (!seen) {
        m_hoisted.emplace_back(std::string{trimmed}, LineOrigin{file, lineNumber});
      }
      return;
    }
    m_body.append(line);
    m_body.push_back('\n');
    m_lines.push_back({file, lineNumber});
  }

  std::span<const std::string_view> m_defines;
  const SourceLookup& m_lookup;
  ResolvedSource& m_out;
  std::vector<Frame> m_stack;
  std::vector<std::pair<std::string, LineOrigin>> m_hoisted;
  std::string m_body;
  std::vector<LineOrigin> m_lines;
};
} // namespace

bool is_valid_shader_id(std::string_view id) noexcept {
  if (id.empty() || id.front() == '/' || id.back() == '/') {
    return false;
  }
  char previous = '\0';
  for (const char c : id) {
    const bool word = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    if (!word && (c != '/' || previous == '/')) {
      return false;
    }
    previous = c;
  }
  return true;
}

bool is_valid_define(std::string_view name) noexcept {
  if (name.empty() || name.front() < 'A' || name.front() > 'Z') {
    return false;
  }
  return std::ranges::all_of(name, [](char c) { return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; });
}

std::string normalize_source(std::string_view source) {
  if (source.starts_with(Bom)) {
    source.remove_prefix(Bom.size());
  }
  std::string out;
  out.reserve(source.size());
  for (const char c : source) {
    if (c != '\r') {
      out.push_back(c);
    }
  }
  return out;
}

bool resolve_source(std::string_view root, std::span<const std::string_view> defines, const SourceLookup& lookup,
                    ResolvedSource& out, std::string& error) {
  out = {};
  error.clear();
  std::vector<std::string_view> sorted{defines.begin(), defines.end()};
  std::ranges::sort(sorted);
  const auto duplicates = std::ranges::unique(sorted);
  sorted.erase(duplicates.begin(), duplicates.end());
  for (const std::string_view define : sorted) {
    if (!is_valid_define(define)) {
      error = fmt::format("{}.wgsl: invalid define '{}'", root, define);
      return false;
    }
  }
  if (!is_valid_shader_id(root)) {
    error = fmt::format("invalid shader id '{}'", root);
    return false;
  }
  const std::optional<std::string_view> text = lookup(root);
  if (!text) {
    error = fmt::format("unknown shader '{}'", root);
    return false;
  }
  Resolver resolver{sorted, lookup, out};
  if (!resolver.include(root, *text, error)) {
    out = {};
    return false;
  }
  resolver.finish();
  return true;
}

std::string describe_line(const ResolvedSource& source, uint32_t outputLine, uint32_t column) {
  if (outputLine == 0 || outputLine > source.lines.size()) {
    return fmt::format("<generated>:{}:{}", outputLine, column);
  }
  const LineOrigin& origin = source.lines[outputLine - 1];
  return fmt::format("{}.wgsl:{}:{}", source.files[origin.file], origin.line, column);
}
} // namespace aurora::render::shaders

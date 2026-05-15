#include "texture_replacement.hpp"

#include "../internal.hpp"
#include "../gx/gx.hpp"
#include "../webgpu/gpu.hpp"
#include "dds_io.hpp"
#include "texture_convert.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/format.h>
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "png_io.hpp"
#include "../fs_helper.hpp"

using namespace aurora::gx;
using aurora::webgpu::g_device;

namespace aurora::gfx::texture_replacement {
Module Log("aurora::gfx::texture_replacement");

struct RuntimeTextureKey {
  uint64_t textureHash = 0;
  uint64_t tlutHash = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  bool hasTlut = false;
  uint32_t format = 0;

  bool operator==(const RuntimeTextureKey& rhs) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const RuntimeTextureKey& key) {
    return H::combine(std::move(h), key.textureHash, key.tlutHash, key.width, key.height, key.hasTlut, key.format);
  }
};

struct TlutMetadata {
  uint32_t size = 0;
  uint32_t format = 0;
  uint16_t entries = 0;
  bool valid = false;
  ByteBuffer data;
};

struct CachedReplacement {
  gfx::TextureHandle handle;
  uint64_t bytes = 0;
  std::list<RuntimeTextureKey>::iterator lruIt;
};

struct ParsedReplacementFilename {
  RuntimeTextureKey key;
  bool hasMips = false;
};

struct ReplacementPaths {
  std::filesystem::path base;
  bool hasMips = false;
  std::filesystem::path rmaos;
  std::filesystem::path roughness;
  std::filesystem::path metallic;
  std::filesystem::path ao;
  std::filesystem::path specular;
  std::filesystem::path normal;
  std::filesystem::path emissive;
};

struct DirectorySignature {
  uint64_t hash = 0;
  uint32_t fileCount = 0;
  bool valid = false;

  bool operator==(const DirectorySignature& rhs) const = default;
};

absl::flat_hash_map<RuntimeTextureKey, ReplacementPaths> s_replacementIndex;
absl::flat_hash_map<RuntimeTextureKey, CachedReplacement> s_replacementCache;
absl::flat_hash_map<std::string, std::filesystem::path> s_namedPbrTextureIndex;
absl::flat_hash_map<std::string, gfx::TextureHandle> s_namedPbrTextureCache;
absl::flat_hash_map<std::string, gfx::TextureHandle> s_debugPreviewTextureCache;
absl::flat_hash_set<RuntimeTextureKey> s_failedKeys;
absl::flat_hash_set<std::string> s_failedNamedPbrTextures;
absl::flat_hash_set<RuntimeTextureKey> s_reportedMisses;
absl::flat_hash_map<const GXTlutObj*, TlutMetadata> s_pendingTluts;
std::array<TlutMetadata, MaxTluts> s_loadedTluts{};
std::list<RuntimeTextureKey> s_replacementLru;
std::filesystem::path s_replacementRoot;
std::filesystem::path s_dumpRoot;
uint64_t s_replacementCacheBytes = 0;
DirectorySignature s_lastDirectorySignature;
uint32_t s_lastAutoRefreshFrame = UINT32_MAX;
bool s_autoRefreshEnabled = false;
constexpr uint64_t kReplacementCacheBudgetBytes = 4294967296; // 4GB, reasonable for modern hardware?
constexpr uint32_t kAutoRefreshPollFrames = 30;
constexpr uint64_t kReplacementWildcardTextureHash = 0xFFFFFFFFFFFFFFFFull;
constexpr uint64_t kReplacementWildcardTlutHash = 0xFFFFFFFFFFFFFFFEull;

bool iequals_ascii(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(lhs[i])) != std::tolower(static_cast<unsigned char>(rhs[i]))) {
      return false;
    }
  }
  return true;
}

bool is_relative_to(const std::filesystem::path& path, const std::filesystem::path& root) noexcept {
  if (root.empty()) {
    return false;
  }
  auto pathIt = path.begin();
  auto rootIt = root.begin();
  for (; rootIt != root.end(); ++rootIt, ++pathIt) {
    if (pathIt == path.end() || !iequals_ascii(fs_path_to_string(*pathIt), fs_path_to_string(*rootIt))) {
      return false;
    }
  }
  return true;
}

bool is_sidecar_mip(std::string_view stem) noexcept {
  constexpr std::string_view tag = "_mip";
  size_t i = stem.size();
  while (i > 0 && stem[i - 1] >= '0' && stem[i - 1] <= '9') {
    --i;
  }

  if (i == stem.size() || i < tag.size()) {
    return false;
  }

  return stem.substr(i - tag.size(), tag.size()) == tag;
}

bool ends_with_ascii_ci(std::string_view text, std::string_view suffix) noexcept {
  if (text.size() < suffix.size()) {
    return false;
  }
  return iequals_ascii(text.substr(text.size() - suffix.size()), suffix);
}

bool is_pbr_sidecar(std::string_view stem) noexcept {
  return ends_with_ascii_ci(stem, "_rmaos") || ends_with_ascii_ci(stem, "_roughness") ||
         ends_with_ascii_ci(stem, "_rough") || ends_with_ascii_ci(stem, "_metallic") ||
         ends_with_ascii_ci(stem, "_metal") || ends_with_ascii_ci(stem, "_ao") ||
         ends_with_ascii_ci(stem, "_specular") || ends_with_ascii_ci(stem, "_spec") ||
         ends_with_ascii_ci(stem, "_normal") || ends_with_ascii_ci(stem, "_n") ||
         ends_with_ascii_ci(stem, "_emissive") || ends_with_ascii_ci(stem, "_e");
}

bool is_supported_replacement_extension(const std::filesystem::path& path) noexcept {
  const auto extension = fs_path_to_string(path.extension());
  return iequals_ascii(extension, ".dds") || iequals_ascii(extension, ".png");
}

std::string to_lower_ascii(std::string_view text) {
  std::string out{text};
  for (char& ch : out) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return out;
}

void mix_signature_value(uint64_t& hash, uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
}

DirectorySignature compute_directory_signature() noexcept {
  DirectorySignature signature{.hash = 0xcbf29ce484222325ull, .fileCount = 0, .valid = false};
  if (s_replacementRoot.empty()) {
    return signature;
  }

  std::error_code ec;
  if (!std::filesystem::is_directory(s_replacementRoot, ec)) {
    return signature;
  }

  signature.valid = true;
  for (std::filesystem::recursive_directory_iterator it(
           s_replacementRoot,
           std::filesystem::directory_options::skip_permission_denied |
               std::filesystem::directory_options::follow_directory_symlink,
           ec);
       !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (!it->is_regular_file(ec)) {
      continue;
    }

    const auto& path = it->path();
    if (is_relative_to(path, s_dumpRoot) || !is_supported_replacement_extension(path)) {
      continue;
    }

    const auto pathString = fs_path_to_string(path.lexically_relative(s_replacementRoot));
    const uint64_t pathHash = XXH64(pathString.data(), pathString.size(), 0);
    const uint64_t fileSize = std::filesystem::file_size(path, ec);
    if (ec) {
      ec.clear();
      continue;
    }
    const auto writeTime = std::filesystem::last_write_time(path, ec);
    if (ec) {
      ec.clear();
      continue;
    }
    const uint64_t writeTicks = static_cast<uint64_t>(writeTime.time_since_epoch().count());

    mix_signature_value(signature.hash, pathHash);
    mix_signature_value(signature.hash, fileSize);
    mix_signature_value(signature.hash, writeTicks);
    ++signature.fileCount;
  }

  return signature;
}

void clear_replacement_index_and_caches() noexcept {
  s_replacementIndex.clear();
  s_replacementCache.clear();
  s_namedPbrTextureIndex.clear();
  s_namedPbrTextureCache.clear();
  s_debugPreviewTextureCache.clear();
  s_failedKeys.clear();
  s_failedNamedPbrTextures.clear();
  s_reportedMisses.clear();
  s_replacementLru.clear();
  s_replacementCacheBytes = 0;
}

std::optional<std::filesystem::path> find_sibling_sidecar(
    const std::filesystem::path& base, std::initializer_list<std::string_view> suffixes) noexcept {
  std::error_code ec;
  const auto parent = base.parent_path();
  const auto stem = base.stem().string();
  const auto extension = base.extension().string();

  for (const std::string_view suffix : suffixes) {
    auto candidate = parent / fmt::format("{}{}{}", stem, suffix, extension);
    if (std::filesystem::is_regular_file(candidate, ec)) {
      return candidate;
    }
  }

  for (std::filesystem::directory_iterator it(parent, std::filesystem::directory_options::skip_permission_denied, ec);
       !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
    if (!it->is_regular_file(ec)) {
      continue;
    }

    const auto filename = it->path().filename().string();
    for (const std::string_view suffix : suffixes) {
      const auto wanted = fmt::format("{}{}{}", stem, suffix, extension);
      if (iequals_ascii(filename, wanted)) {
        return it->path();
      }
    }
  }

  return std::nullopt;
}

std::optional<uint64_t> parse_hex(std::string_view text) noexcept {
  if (text.empty()) {
    return std::nullopt;
  }
  uint64_t value = 0;
  for (const char ch : text) {
    value <<= 4;
    if (ch >= '0' && ch <= '9') {
      value |= static_cast<uint64_t>(ch - '0');
    } else if (ch >= 'a' && ch <= 'f') {
      value |= static_cast<uint64_t>(ch - 'a' + 10);
    } else if (ch >= 'A' && ch <= 'F') {
      value |= static_cast<uint64_t>(ch - 'A' + 10);
    } else {
      return std::nullopt;
    }
  }
  return value;
}

std::optional<uint32_t> parse_u32(std::string_view text, int base = 10) noexcept {
  if (text.empty()) {
    return std::nullopt;
  }

  uint32_t value = 0;
  const auto* begin = text.data();
  const auto* end = begin + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value, base);
  if (ec != std::errc{} || ptr != end) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::pair<uint32_t, uint32_t>> parse_dimensions(std::string_view text) noexcept {
  const size_t sep = text.find('x');
  if (sep == std::string_view::npos) {
    return std::nullopt;
  }

  const auto width = parse_u32(text.substr(0, sep));
  const auto height = parse_u32(text.substr(sep + 1));
  if (!width.has_value() || !height.has_value()) {
    return std::nullopt;
  }
  return std::pair{*width, *height};
}

uint32_t texture_base_level_size(const GXTexObj_& obj) noexcept {
  switch (obj.format()) {
  case GX_TF_R8_PC:
    return obj.width() * obj.height();
  case GX_TF_RGBA8_PC:
    return obj.width() * obj.height() * 4;
  default:
    return GXGetTexBufferSize(obj.width(), obj.height(), obj.format(), false, 0);
  }
}

std::optional<uint64_t> compute_referenced_tlut_hash(const GXTexObj_& obj) noexcept {
  if (!is_palette_format(obj.format()) || obj.tlut >= s_loadedTluts.size()) {
    return std::nullopt;
  }

  const auto& tlut = s_loadedTluts[obj.tlut];
  const uint32_t textureSize = texture_base_level_size(obj);
  const auto* textureData = static_cast<const uint8_t*>(obj.data);
  if (!tlut.valid || textureData == nullptr || textureSize == 0) {
    return std::nullopt;
  }

  uint32_t minIndex = 0xffff;
  uint32_t maxIndex = 0;
  switch (obj.format()) {
  case GX_TF_C4:
    for (uint32_t i = 0; i < textureSize; ++i) {
      const uint32_t lowNibble = textureData[i] & 0xf;
      const uint32_t highNibble = textureData[i] >> 4;
      minIndex = std::min({minIndex, lowNibble, highNibble});
      maxIndex = std::max({maxIndex, lowNibble, highNibble});
    }
    break;
  case GX_TF_C8:
    for (uint32_t i = 0; i < textureSize; ++i) {
      const uint32_t index = textureData[i];
      minIndex = std::min(minIndex, index);
      maxIndex = std::max(maxIndex, index);
    }
    break;
  case GX_TF_C14X2:
    for (uint32_t i = 0; i + sizeof(uint16_t) <= textureSize; i += sizeof(uint16_t)) {
      uint16_t value = 0;
      std::memcpy(&value, textureData + i, sizeof(value));
      const uint32_t index = bswap(value) & 0x3fff;
      minIndex = std::min(minIndex, index);
      maxIndex = std::max(maxIndex, index);
    }
    break;
  default:
    return std::nullopt;
  }

  size_t tlutSize = 2 * (static_cast<size_t>(maxIndex) + 1 - minIndex);
  const size_t tlutOffset = 2 * static_cast<size_t>(minIndex);
  if (tlutOffset + tlutSize > tlut.data.size()) {
    return std::nullopt;
  }
  return XXH64(tlut.data.data() + tlutOffset, tlutSize, 0);
}

const TlutMetadata* get_loaded_tlut(const GXTexObj_& obj) noexcept {
  if (!is_palette_format(obj.format()) || obj.tlut >= s_loadedTluts.size()) {
    return nullptr;
  }

  const auto& tlut = s_loadedTluts[obj.tlut];
  return tlut.valid ? &tlut : nullptr;
}

std::optional<uint32_t> tlut_to_texture_format(uint32_t tlutFormat) noexcept {
  switch (tlutFormat) {
  case GX_TL_IA8:
    return GX_TF_IA8;
  case GX_TL_RGB565:
    return GX_TF_RGB565;
  case GX_TL_RGB5A3:
    return GX_TF_RGB5A3;
  default:
    return std::nullopt;
  }
}

bool ensure_directory(const std::filesystem::path& dir) noexcept {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return !ec;
}

RuntimeTextureKey build_runtime_key(const GXTexObj_& obj) noexcept {
  RuntimeTextureKey key{
      .width = obj.width(),
      .height = obj.height(),
      .hasTlut = is_palette_format(obj.format()),
      .format = obj.format(),
  };

  const uint32_t textureSize = texture_base_level_size(obj);
  if (obj.data != nullptr && textureSize != 0) {
    key.textureHash = XXH64(obj.data, textureSize, 0);
  }
  if (key.hasTlut) {
    key.tlutHash = compute_referenced_tlut_hash(obj).value_or(0);
  }
  return key;
}

std::string format_replacement_filename(const RuntimeTextureKey& key) {
  if (key.hasTlut) {
    return fmt::format("tex1_{}x{}_{:016x}_{:016x}_{}.dds", key.width, key.height,
                       key.textureHash, key.tlutHash, key.format);
  }
  return fmt::format("tex1_{}x{}_{:016x}_{}.dds", key.width, key.height, key.textureHash, key.format);
}

std::optional<ParsedReplacementFilename> parse_replacement_filename(std::string_view filename) noexcept {
  const size_t dot = filename.rfind('.');
  if (dot == std::string_view::npos) {
    return std::nullopt;
  }

  if (!iequals_ascii(filename.substr(dot), ".dds") && !iequals_ascii(filename.substr(dot), ".png")) {
    return std::nullopt;
  }

  const std::string_view stem = filename.substr(0, dot);
  constexpr std::string_view prefix = "tex1_";
  if (!stem.starts_with(prefix)) {
    return std::nullopt;
  }

  std::array<std::string_view, 6> parts{};
  size_t partCount = 0;
  size_t offset = 0;
  bool consumedAll = false;
  while (offset <= stem.size() && partCount < parts.size()) {
    const size_t next = stem.find('_', offset);
    parts[partCount++] = stem.substr(offset, next == std::string_view::npos ? stem.size() - offset : next - offset);
    if (next == std::string_view::npos) {
      consumedAll = true;
      break;
    }
    offset = next + 1;
  }
  if (!consumedAll || partCount < 4 || partCount > 6 || parts[0] != "tex1") {
    return std::nullopt;
  }

  const auto dimensions = parse_dimensions(parts[1]);
  if (!dimensions.has_value()) {
    return std::nullopt;
  }

  size_t index = 2;
  bool hasMips = false;
  if (parts[index] == "m") {
    hasMips = true;
    ++index;
  }

  size_t remaining = partCount - index;
  if (remaining != 2 && remaining != 3) {
    return std::nullopt;
  }

  uint64_t textureHash = 0;
  if (parts[index] == "$") {
    textureHash = kReplacementWildcardTextureHash;
  } else {
    const auto parsedTex = parse_hex(parts[index]);
    if (!parsedTex.has_value()) {
      return std::nullopt;
    }
    textureHash = *parsedTex;
  }

  auto formatPart = parts[partCount - 1];
  if (formatPart == "arb") {
    formatPart = parts[partCount - 2];
    remaining -= 1;
  }
  const auto format = parse_u32(formatPart);
  if (!format.has_value()) {
    return std::nullopt;
  }

  uint64_t tlutHash = 0;
  const bool hasTlut = remaining == 3;
  if (hasTlut) {
    const std::string_view tlutPart = parts[index + 1];
    if (tlutPart == "$") {
      tlutHash = kReplacementWildcardTlutHash;
    } else {
      const auto parsedTlutHash = parse_hex(tlutPart);
      if (!parsedTlutHash.has_value()) {
        return std::nullopt;
      }
      tlutHash = *parsedTlutHash;
    }
  }

  return ParsedReplacementFilename{
      .key =
          RuntimeTextureKey{
              .textureHash = textureHash,
              .tlutHash = tlutHash,
              .width = dimensions->first,
              .height = dimensions->second,
              .hasTlut = hasTlut,
              .format = *format,
          },
      .hasMips = hasMips,
  };
}

static std::optional<ConvertedTexture> load_texture_file(const std::filesystem::path& path) {
  if (iequals_ascii(fs_path_to_string(path.extension()), ".png")) {
    return png::load_png_file(path);
  } else {
    return dds::load_dds_file(path);
  }
}

std::optional<ConvertedTexture> load_replacement(const std::filesystem::path& path, bool hasMips) noexcept {
  auto base = load_texture_file(path);
  if (!base.has_value()) {
    Log.warn("texture_replacement: failed to load texture {}", fs_path_to_string(path));
    return std::nullopt;
  }
  if (!hasMips) {
    return base;
  }

  std::vector<ConvertedTexture> more;
  std::error_code ec;
  for (uint32_t mipLevel = 1;; ++mipLevel) {
    const auto mipPath =
        path.parent_path() /
        fmt::format("{}_mip{}{}", fs_path_to_string(path.stem()), mipLevel, fs_path_to_string(path.extension()));
    if (!std::filesystem::is_regular_file(mipPath, ec)) {
      break;
    }

    auto lvl = load_texture_file(mipPath);
    const uint32_t ew = std::max(base->width >> mipLevel, 1u);
    const uint32_t eh = std::max(base->height >> mipLevel, 1u);
    const bool ok = lvl.has_value() && lvl->format == base->format && lvl->width == ew && lvl->height == eh;
    if (!ok) {
      if (!lvl.has_value()) {
        Log.warn("texture_replacement: could not load mip {}", fs_path_to_string(mipPath));
      } else {
        Log.warn("texture_replacement: expected {}x{} for mip {}, got {}x{}", ew, eh, fs_path_to_string(mipPath),
                 lvl->width, lvl->height);
      }

      break;
    }
    more.push_back(std::move(*lvl));
  }

  if (more.empty()) {
    return std::nullopt;
  }

  const uint32_t mips = 1u + static_cast<uint32_t>(more.size());
  const uint64_t n = calc_texture_size(base->format, base->width, base->height, mips);
  if (n == 0) {
    return std::nullopt;
  }

  ByteBuffer blob{static_cast<size_t>(n)};
  uint8_t* const dst = blob.data();
  uint64_t o = 0;
  const auto append = [&](const ByteBuffer& d) noexcept -> bool {
    if (o + d.size() > n) {
      return false;
    }
    std::memcpy(dst + o, d.data(), d.size());
    o += d.size();
    return true;
  };
  if (!append(base->data)) {
    return std::nullopt;
  }
  for (const auto& mip : more) {
    if (!append(mip.data)) {
      return std::nullopt;
    }
  }
  if (o != n) {
    return std::nullopt;
  }

  return ConvertedTexture{
      .format = base->format,
      .width = base->width,
      .height = base->height,
      .mips = mips,
      .data = std::move(blob),
  };
}

void touch_cached_replacement(decltype(s_replacementCache)::iterator it) noexcept {
  if (it->second.lruIt != s_replacementLru.begin()) {
    s_replacementLru.splice(s_replacementLru.begin(), s_replacementLru, it->second.lruIt);
    it->second.lruIt = s_replacementLru.begin();
  }
}

void evict_replacement_cache_if_needed() noexcept {
  while (s_replacementCacheBytes > kReplacementCacheBudgetBytes && !s_replacementLru.empty()) {
    const RuntimeTextureKey key = s_replacementLru.back();
    s_replacementLru.pop_back();

    const auto it = s_replacementCache.find(key);
    if (it == s_replacementCache.end()) {
      continue;
    }

    const uint64_t entryBytes = it->second.bytes;
    s_replacementCache.erase(it);
    s_replacementCacheBytes -= std::min(s_replacementCacheBytes, entryBytes);
  }
}

void build_index() noexcept {
  if (!g_config.allowTextureReplacements) {
    return;
  }

  auto userPath = std::filesystem::path{reinterpret_cast<const char8_t*>(g_config.userPath)};
  auto cachePath = std::filesystem::path{reinterpret_cast<const char8_t*>(g_config.cachePath)};

  s_replacementRoot = userPath / "texture_replacements";
  s_dumpRoot = cachePath / "texture_dumps";

  if (!ensure_directory(s_replacementRoot)) {
    return;
  }
  if (g_config.allowTextureDumps && !ensure_directory(s_dumpRoot)) {
    return;
  }

  // Single recursive scan: index PBR sidecars by (parent, lowercase_stem) for O(1) matching,
  // avoiding per-texture directory re-scans in find_sibling_sidecar.
  absl::flat_hash_map<std::pair<std::string, std::string>, std::filesystem::path> sidecarIndex;
  std::vector<std::pair<ParsedReplacementFilename, std::filesystem::path>> baseTextures;

  std::error_code ec;
  for (std::filesystem::recursive_directory_iterator it(
           s_replacementRoot,
           std::filesystem::directory_options::skip_permission_denied |
               std::filesystem::directory_options::follow_directory_symlink,
           ec);
       it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) {
      break;
    }

    if (!it->is_regular_file()) {
      continue;
    }

    const auto& path = it->path();

    if (is_relative_to(path, s_dumpRoot)) {
      continue;
    }

    if (!is_supported_replacement_extension(path)) {
      continue;
    }

    const auto stem = path.stem().string();

    if (is_sidecar_mip(stem)) {
      continue;
    }

    if (is_pbr_sidecar(stem)) {
      s_namedPbrTextureIndex.try_emplace(to_lower_ascii(path.filename().string()), path);
      s_namedPbrTextureIndex.try_emplace(to_lower_ascii(stem), path);
      sidecarIndex.try_emplace(std::pair{path.parent_path().string(), to_lower_ascii(stem)}, path);
      continue;
    }

    const auto parsed = parse_replacement_filename(fs_path_to_string(path.filename()));
    if (!parsed.has_value()) {
      continue;
    }

    baseTextures.emplace_back(*parsed, path);
  }

  for (auto& [parsed, basePath] : baseTextures) {
    const auto parentStr = basePath.parent_path().string();
    const auto stemLower = to_lower_ascii(basePath.stem().string());

    const auto findSidecar = [&](std::initializer_list<std::string_view> suffixes) -> std::filesystem::path {
      for (const std::string_view suffix : suffixes) {
        if (const auto sit = sidecarIndex.find(std::pair{parentStr, stemLower + std::string(suffix)});
            sit != sidecarIndex.end()) {
          return sit->second;
        }
      }
      return {};
    };

    ReplacementPaths replacement{.base = basePath, .hasMips = parsed.hasMips};
    replacement.rmaos = findSidecar({"_rmaos"});
    replacement.roughness = findSidecar({"_roughness", "_rough"});
    replacement.metallic = findSidecar({"_metallic", "_metal"});
    replacement.ao = findSidecar({"_ao"});
    replacement.specular = findSidecar({"_specular", "_spec"});
    replacement.normal = findSidecar({"_normal", "_n"});
    replacement.emissive = findSidecar({"_emissive", "_e"});

    s_replacementIndex.try_emplace(parsed.key, std::move(replacement));
  }

  Log.info("Indexed {} texture replacements", s_replacementIndex.size());
}

const ReplacementPaths* find_replacement_paths(const RuntimeTextureKey& key) noexcept {
  if (const auto it = s_replacementIndex.find(key); it != s_replacementIndex.end()) {
    return &it->second;
  }

  if (key.hasTlut) {
    RuntimeTextureKey tlutWildcardKey = key;
    tlutWildcardKey.tlutHash = kReplacementWildcardTlutHash;
    if (const auto it = s_replacementIndex.find(tlutWildcardKey); it != s_replacementIndex.end()) {
      return &it->second;
    }
  }

  RuntimeTextureKey textureWildcardKey = key;
  textureWildcardKey.textureHash = kReplacementWildcardTextureHash;
  if (const auto it = s_replacementIndex.find(textureWildcardKey); it != s_replacementIndex.end()) {
    return &it->second;
  }

  return nullptr;
}

const gfx::TextureHandle* find_cached_replacement(const RuntimeTextureKey& key) noexcept {
  const auto cached = s_replacementCache.find(key);
  if (cached == s_replacementCache.end()) {
    return nullptr;
  }

  touch_cached_replacement(cached);
  return &cached->second.handle;
}

std::string texture_format_name(wgpu::TextureFormat format) {
  switch (format) {
  case wgpu::TextureFormat::RGBA8Unorm:
    return "RGBA8Unorm";
  case wgpu::TextureFormat::BGRA8Unorm:
    return "BGRA8Unorm";
  case wgpu::TextureFormat::BC1RGBAUnorm:
    return "BC1RGBAUnorm";
  case wgpu::TextureFormat::BC3RGBAUnorm:
    return "BC3RGBAUnorm";
  case wgpu::TextureFormat::BC5RGUnorm:
    return "BC5RGUnorm";
  case wgpu::TextureFormat::BC7RGBAUnorm:
    return "BC7RGBAUnorm";
  case wgpu::TextureFormat::Undefined:
    return "Undefined";
  default:
    return fmt::format("TextureFormat({})", static_cast<uint32_t>(format));
  }
}

gfx::TextureHandle upload_converted_texture(const ConvertedTexture& replacement, std::string_view label) noexcept {
  const wgpu::Extent3D size{
      .width = replacement.width,
      .height = replacement.height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label.data(),
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = replacement.format,
      .mipLevelCount = replacement.mips,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);
  const auto viewLabel = fmt::format("{} view", label);
  const wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = replacement.format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = replacement.mips,
  };
  auto textureView = texture.CreateView(&textureViewDescriptor);
  auto handle = std::make_shared<gfx::TextureRef>(std::move(texture), std::move(textureView), wgpu::TextureView{}, size,
                                                  replacement.format, replacement.mips, gfx::InvalidTextureFormat);
  gfx::write_texture(*handle, replacement.data);
  return handle;
}

gfx::TextureHandle load_texture_file(const RuntimeTextureKey& key, const std::filesystem::path& path, bool hasMips,
                                     std::string_view labelPrefix, bool reportFailure) noexcept {
  const auto replacement = load_replacement(path, hasMips);
  if (!replacement.has_value()) {
    if (reportFailure) {
      s_failedKeys.insert(key);
    }
    return {};
  }

  const auto label = fmt::format("{} {}", labelPrefix, fs_path_to_string(path.filename().string()));
  return upload_converted_texture(*replacement, label);
}

gfx::TextureHandle load_named_texture_file(const std::filesystem::path& path) noexcept {
  const auto replacement = load_replacement(path, false);
  if (!replacement.has_value()) {
    Log.warn("texture_replacement: failed to load named PBR texture {}", fs_path_to_string(path.string()));
    return {};
  }

  const auto label = fmt::format("TextureReplacement Named PBR {}", fs_path_to_string(path.filename().string()));
  return upload_converted_texture(*replacement, label);
}

gfx::TextureHandle load_replacement_texture(const RuntimeTextureKey& key, const ReplacementPaths& paths) noexcept {
  auto handle = load_texture_file(key, paths.base, paths.hasMips, "TextureReplacement", true);
  if (!handle) {
    return {};
  }

  if (!paths.rmaos.empty()) {
    handle->pbrRmaos = load_texture_file(key, paths.rmaos, false, "TextureReplacement RMAOS", false);
  }
  if (!paths.roughness.empty()) {
    handle->pbrRoughness = load_texture_file(key, paths.roughness, false, "TextureReplacement Roughness", false);
  }
  if (!paths.metallic.empty()) {
    handle->pbrMetallic = load_texture_file(key, paths.metallic, false, "TextureReplacement Metallic", false);
  }
  if (!paths.ao.empty()) {
    handle->pbrAo = load_texture_file(key, paths.ao, false, "TextureReplacement AO", false);
  }
  if (!paths.specular.empty()) {
    handle->pbrSpecular = load_texture_file(key, paths.specular, false, "TextureReplacement Specular", false);
  }
  if (!paths.normal.empty()) {
    handle->pbrNormal = load_texture_file(key, paths.normal, false, "TextureReplacement Normal", false);
  }
  if (!paths.emissive.empty()) {
    handle->pbrEmissive = load_texture_file(key, paths.emissive, false, "TextureReplacement Emissive", false);
  }

  return handle;
}

uint64_t calc_texture_handle_bytes(const gfx::TextureHandle& handle) noexcept {
  if (!handle) {
    return 0;
  }

  return calc_texture_size(handle->format, handle->size.width, handle->size.height, handle->mipCount);
}

void cache_replacement(const RuntimeTextureKey& key, const gfx::TextureHandle& handle) noexcept {
  const uint64_t replacementBytes = calc_texture_handle_bytes(handle) + calc_texture_handle_bytes(handle->pbrRmaos) +
                                    calc_texture_handle_bytes(handle->pbrRoughness) +
                                    calc_texture_handle_bytes(handle->pbrMetallic) +
                                    calc_texture_handle_bytes(handle->pbrAo) +
                                    calc_texture_handle_bytes(handle->pbrSpecular) +
                                    calc_texture_handle_bytes(handle->pbrNormal) +
                                    calc_texture_handle_bytes(handle->pbrEmissive);
  s_replacementLru.push_front(key);
  s_replacementCache.emplace(
      key, CachedReplacement{.handle = handle, .bytes = replacementBytes, .lruIt = s_replacementLru.begin()});
  s_replacementCacheBytes += replacementBytes;
  evict_replacement_cache_if_needed();
}

void bind_replacement(GXTexObj_& obj, GXTexMapID id, const gfx::TextureHandle& handle) noexcept {
  GXTexObj_ out = obj;
  out.mWidth = handle->size.width;
  out.mHeight = handle->size.height;
  out.mFormat = GX_TF_RGBA8_PC;
  g_gxState.textures[id] = gfx::TextureBind(out, handle);
  g_gxState.stateDirty = true;
}

bool dump_editable_texture_dds(const RuntimeTextureKey& key, const GXTexObj_& obj) noexcept {
  const ArrayRef<uint8_t> texData{static_cast<const uint8_t*>(obj.data), UINT32_MAX};
  const uint32_t texWidth = obj.width();
  const uint32_t texHeight = obj.height();

  ConvertedTexture pixels;
  if (is_palette_format(obj.format())) {
    const TlutMetadata* tlut = get_loaded_tlut(obj);
    if (tlut == nullptr) {
      return false;
    }
    pixels =
        convert_texture_palette(obj.format(), texWidth, texHeight, 1, texData, static_cast<GXTlutFmt>(tlut->format),
                                tlut->entries, {tlut->data.data(), tlut->data.size()});
  } else {
    pixels = convert_texture(obj.format(), texWidth, texHeight, 1, texData);
  }

  const uint64_t rgbaBytes = calc_texture_size(wgpu::TextureFormat::RGBA8Unorm, texWidth, texHeight, 1);

  if (pixels.data.empty() || pixels.format != wgpu::TextureFormat::RGBA8Unorm || pixels.data.size() != rgbaBytes) {
    return false;
  }

  const auto path = s_dumpRoot / format_replacement_filename(key);
  return dds::write_rgba8_dds(path, texWidth, texHeight, pixels.data);
}

bool report_missing_key(const RuntimeTextureKey& key, const GXTexObj_& obj) noexcept {
  if (!s_reportedMisses.insert(key).second) {
    return false;
  }

  if (g_config.allowTextureDumps) {
    dump_editable_texture_dds(key, obj);
  }
  return true;
}

constexpr size_t debug_map_slot_index(DebugMapSlot slot) noexcept { return static_cast<size_t>(slot); }

std::filesystem::path debug_make_relative_path(const std::filesystem::path& path) {
  if (!s_replacementRoot.empty() && is_relative_to(path, s_replacementRoot)) {
    return path.lexically_relative(s_replacementRoot);
  }
  return path;
}

std::filesystem::path debug_directory_for_path(const std::filesystem::path& path) {
  const auto relative = debug_make_relative_path(path.parent_path());
  return relative.empty() ? std::filesystem::path{"."} : relative;
}

std::string debug_path_key(const std::filesystem::path& path) {
  return to_lower_ascii(fs_path_to_string(path.lexically_normal()));
}

bool debug_has_any_pbr_map(const ReplacementPaths& paths) noexcept {
  return !paths.rmaos.empty() || !paths.roughness.empty() || !paths.metallic.empty() || !paths.ao.empty() ||
         !paths.specular.empty() || !paths.normal.empty() || !paths.emissive.empty();
}

void debug_set_map(DebugMaterialInfo& info, DebugMapSlot slot, const std::filesystem::path& path) {
  info.maps[debug_map_slot_index(slot)] = path;
}

void debug_add_attached_sidecar_paths(absl::flat_hash_set<std::string>& out) {
  const auto add = [&](const std::filesystem::path& path) {
    if (!path.empty()) {
      out.insert(debug_path_key(path));
    }
  };

  for (const auto& [key, paths] : s_replacementIndex) {
    (void)key;
    add(paths.rmaos);
    add(paths.roughness);
    add(paths.metallic);
    add(paths.ao);
    add(paths.specular);
    add(paths.normal);
    add(paths.emissive);
  }
}

std::optional<std::pair<std::string, DebugMapSlot>> debug_split_sidecar_stem(std::string_view stem) {
  constexpr std::array<std::pair<std::string_view, DebugMapSlot>, 12> suffixes{{
      {"_roughness", DebugMapSlot::Roughness},
      {"_metallic", DebugMapSlot::Metallic},
      {"_specular", DebugMapSlot::Specular},
      {"_emissive", DebugMapSlot::Emissive},
      {"_rmaos", DebugMapSlot::Rmaos},
      {"_rough", DebugMapSlot::Roughness},
      {"_metal", DebugMapSlot::Metallic},
      {"_normal", DebugMapSlot::Normal},
      {"_spec", DebugMapSlot::Specular},
      {"_ao", DebugMapSlot::Ao},
      {"_n", DebugMapSlot::Normal},
      {"_e", DebugMapSlot::Emissive},
  }};

  for (const auto& [suffix, slot] : suffixes) {
    if (ends_with_ascii_ci(stem, suffix)) {
      return std::pair{std::string{stem.substr(0, stem.size() - suffix.size())}, slot};
    }
  }
  return std::nullopt;
}

std::vector<DebugMaterialInfo> debug_collect_named_pbr_materials(
    const absl::flat_hash_set<std::string>& attachedSidecars) {
  absl::flat_hash_set<std::string> uniquePaths;
  absl::flat_hash_map<std::string, DebugMaterialInfo> groups;

  for (const auto& [name, path] : s_namedPbrTextureIndex) {
    (void)name;
    const auto pathKey = debug_path_key(path);
    if (!uniquePaths.insert(pathKey).second || attachedSidecars.contains(pathKey)) {
      continue;
    }

    const auto split = debug_split_sidecar_stem(to_lower_ascii(path.stem().string()));
    if (!split.has_value() || split->first.empty()) {
      continue;
    }

    const std::string groupKey = fmt::format("{}|{}", debug_path_key(path.parent_path()), split->first);
    auto [it, inserted] = groups.try_emplace(groupKey);
    if (inserted) {
      auto& info = it->second;
      info.kind = DebugMaterialKind::NamedPbr;
      info.name = split->first;
      info.directory = debug_directory_for_path(path);
    }
    debug_set_map(it->second, split->second, path);
  }

  std::vector<DebugMaterialInfo> out;
  out.reserve(groups.size());
  for (auto& [key, info] : groups) {
    (void)key;
    out.push_back(std::move(info));
  }
  return out;
}

uint32_t debug_unique_named_pbr_texture_count() {
  absl::flat_hash_set<std::string> uniquePaths;
  for (const auto& [name, path] : s_namedPbrTextureIndex) {
    (void)name;
    uniquePaths.insert(debug_path_key(path));
  }
  return static_cast<uint32_t>(uniquePaths.size());
}

std::vector<DebugMaterialInfo> debug_collect_materials() {
  std::vector<DebugMaterialInfo> out;
  out.reserve(s_replacementIndex.size());

  for (const auto& [key, paths] : s_replacementIndex) {
    DebugMaterialInfo info;
    info.kind = DebugMaterialKind::Replacement;
    info.name = fs_path_to_string(paths.base.filename());
    info.directory = debug_directory_for_path(paths.base);
    debug_set_map(info, DebugMapSlot::Base, paths.base);
    debug_set_map(info, DebugMapSlot::Rmaos, paths.rmaos);
    debug_set_map(info, DebugMapSlot::Roughness, paths.roughness);
    debug_set_map(info, DebugMapSlot::Metallic, paths.metallic);
    debug_set_map(info, DebugMapSlot::Ao, paths.ao);
    debug_set_map(info, DebugMapSlot::Specular, paths.specular);
    debug_set_map(info, DebugMapSlot::Normal, paths.normal);
    debug_set_map(info, DebugMapSlot::Emissive, paths.emissive);
    info.textureHash = key.textureHash;
    info.tlutHash = key.tlutHash;
    info.width = key.width;
    info.height = key.height;
    info.format = key.format;
    info.hasTlut = key.hasTlut;
    info.hasMips = paths.hasMips;
    info.cached = s_replacementCache.contains(key);
    info.failed = s_failedKeys.contains(key);
    info.reportedMissing = s_reportedMisses.contains(key);
    info.textureHashWildcard = key.textureHash == kReplacementWildcardTextureHash;
    info.tlutHashWildcard = key.tlutHash == kReplacementWildcardTlutHash;
    out.push_back(std::move(info));
  }

  absl::flat_hash_set<std::string> attachedSidecars;
  debug_add_attached_sidecar_paths(attachedSidecars);
  auto namedMaterials = debug_collect_named_pbr_materials(attachedSidecars);
  out.insert(out.end(), std::make_move_iterator(namedMaterials.begin()), std::make_move_iterator(namedMaterials.end()));

  std::sort(out.begin(), out.end(), [](const DebugMaterialInfo& lhs, const DebugMaterialInfo& rhs) {
    const auto lhsDir = fs_path_to_string(lhs.directory);
    const auto rhsDir = fs_path_to_string(rhs.directory);
    if (lhsDir != rhsDir) {
      return lhsDir < rhsDir;
    }
    if (lhs.name != rhs.name) {
      return lhs.name < rhs.name;
    }
    return static_cast<uint8_t>(lhs.kind) < static_cast<uint8_t>(rhs.kind);
  });
  return out;
}

DebugInventoryStats debug_inventory_stats() {
  DebugInventoryStats stats;
  stats.replacementRoot = s_replacementRoot;
  stats.dumpRoot = s_dumpRoot;
  stats.indexedReplacementCount = static_cast<uint32_t>(s_replacementIndex.size());
  stats.cachedReplacementCount = static_cast<uint32_t>(s_replacementCache.size());
  stats.failedReplacementCount = static_cast<uint32_t>(s_failedKeys.size());
  stats.reportedMissingCount = static_cast<uint32_t>(s_reportedMisses.size());
  stats.cachedReplacementBytes = s_replacementCacheBytes;
  stats.autoRefreshEnabled = s_autoRefreshEnabled;
  stats.directorySignatureValid = s_lastDirectorySignature.valid;
  stats.watchedFileCount = s_lastDirectorySignature.fileCount;
  stats.namedPbrTextureCount = debug_unique_named_pbr_texture_count();

  for (const auto& [key, paths] : s_replacementIndex) {
    (void)key;
    if (debug_has_any_pbr_map(paths)) {
      ++stats.pbrReplacementCount;
    }
  }

  absl::flat_hash_set<std::string> attachedSidecars;
  debug_add_attached_sidecar_paths(attachedSidecars);
  stats.namedPbrMaterialCount = static_cast<uint32_t>(debug_collect_named_pbr_materials(attachedSidecars).size());
  return stats;
}

DebugTexturePreview debug_load_texture_preview(const std::filesystem::path& path) {
  DebugTexturePreview preview;
  if (path.empty()) {
    preview.error = "No texture path";
    return preview;
  }
  if (!is_supported_replacement_extension(path)) {
    preview.error = "Unsupported texture extension";
    return preview;
  }

  const std::string cacheKey = debug_path_key(path);
  if (const auto it = s_debugPreviewTextureCache.find(cacheKey); it != s_debugPreviewTextureCache.end() && it->second) {
    const auto& handle = it->second;
    preview.width = handle->size.width;
    preview.height = handle->size.height;
    preview.mipCount = handle->mipCount;
    preview.byteSize = calc_texture_size(handle->format, handle->size.width, handle->size.height, handle->mipCount);
    preview.formatName = texture_format_name(handle->format);
    preview.textureId = reinterpret_cast<ImTextureID>(handle->sampleTextureView.Get());
    return preview;
  }

  const auto converted = load_replacement(path, false);
  if (!converted.has_value()) {
    preview.error = "Failed to load texture";
    return preview;
  }

  const auto label = fmt::format("TextureReplacement Debug {}", fs_path_to_string(path.filename()));
  preview.width = converted->width;
  preview.height = converted->height;
  preview.mipCount = converted->mips;
  preview.byteSize = calc_texture_size(converted->format, converted->width, converted->height, converted->mips);
  preview.formatName = texture_format_name(converted->format);
  auto texture = upload_converted_texture(*converted, label);
  if (!texture) {
    preview.error = "Failed to upload texture preview";
    return preview;
  }
  preview.textureId = reinterpret_cast<ImTextureID>(texture->sampleTextureView.Get());
  s_debugPreviewTextureCache[cacheKey] = std::move(texture);
  return preview;
}

void debug_clear_preview_cache() { s_debugPreviewTextureCache.clear(); }

void initialize() noexcept {
  build_index();
  s_lastDirectorySignature = compute_directory_signature();
  s_lastAutoRefreshFrame = UINT32_MAX;
}

void refresh() noexcept {
  if (!g_config.allowTextureReplacements) {
    clear_replacement_index_and_caches();
    s_lastDirectorySignature = {};
    return;
  }

  clear_replacement_index_and_caches();
  build_index();
  s_lastDirectorySignature = compute_directory_signature();
  Log.info("Refreshed texture replacements");
}

void set_auto_refresh(bool enabled) noexcept { s_autoRefreshEnabled = enabled; }

bool auto_refresh_enabled() noexcept { return s_autoRefreshEnabled; }

bool update_auto_refresh(uint32_t frame) noexcept {
  if (!s_autoRefreshEnabled || !g_config.allowTextureReplacements) {
    return false;
  }

  if (s_lastAutoRefreshFrame != UINT32_MAX && frame - s_lastAutoRefreshFrame < kAutoRefreshPollFrames) {
    return false;
  }
  s_lastAutoRefreshFrame = frame;

  const DirectorySignature currentSignature = compute_directory_signature();
  if (!currentSignature.valid) {
    return false;
  }
  if (!s_lastDirectorySignature.valid) {
    s_lastDirectorySignature = currentSignature;
    return false;
  }
  if (currentSignature == s_lastDirectorySignature) {
    return false;
  }

  refresh();
  return true;
}

void shutdown() noexcept {
  s_replacementIndex.clear();
  s_replacementCache.clear();
  s_namedPbrTextureIndex.clear();
  s_namedPbrTextureCache.clear();
  s_debugPreviewTextureCache.clear();
  s_failedKeys.clear();
  s_failedNamedPbrTextures.clear();
  s_reportedMisses.clear();
  s_pendingTluts.clear();
  for (auto& tlut : s_loadedTluts) {
    tlut = {};
  }
  s_replacementLru.clear();
  s_replacementCacheBytes = 0;
  s_lastDirectorySignature = {};
  s_lastAutoRefreshFrame = UINT32_MAX;
  s_replacementRoot.clear();
  s_dumpRoot.clear();
}

void register_tlut(const GXTlutObj* obj, const void* data, GXTlutFmt format, uint16_t entries) noexcept {
  if (obj == nullptr || data == nullptr) {
    return;
  }

  const size_t sz = static_cast<size_t>(entries) * 2;
  ByteBuffer buffer{sz};
  std::memcpy(buffer.data(), static_cast<const uint8_t*>(data), sz);
  s_pendingTluts[obj] = {
      .size = static_cast<uint32_t>(entries) * 2,
      .format = static_cast<uint32_t>(format),
      .entries = entries,
      .valid = true,
      .data = std::move(buffer),
  };
}

void load_tlut(const GXTlutObj* obj, uint32_t idx) noexcept {
  if (idx >= s_loadedTluts.size()) {
    return;
  }

  const auto it = s_pendingTluts.find(obj);
  if (it == s_pendingTluts.end()) {
    s_loadedTluts[idx] = {};
    return;
  }

  const auto& pending = it->second;
  s_loadedTluts[idx] = {
      .size = pending.size,
      .format = pending.format,
      .entries = pending.entries,
      .valid = pending.valid,
      .data = pending.data.clone(),
  };
}

bool try_bind_replacement(GXTexObj_& obj, GXTexMapID id) noexcept {
  if (!g_config.allowTextureReplacements) {
    return false;
  }

  const auto handle = find_replacement(obj);
  if (!handle.has_value()) {
    return false;
  }
  bind_replacement(obj, id, *handle);
  return true;
}

std::optional<TextureHandle> find_named_pbr_texture(std::string_view name) noexcept {
  if (!g_config.allowTextureReplacements || name.empty()) {
    return std::nullopt;
  }

  auto key = to_lower_ascii(std::filesystem::path{std::string{name}}.filename().string());
  if (key.empty()) {
    return std::nullopt;
  }

  auto pathIt = s_namedPbrTextureIndex.find(key);
  if (pathIt == s_namedPbrTextureIndex.end() &&
      (ends_with_ascii_ci(key, ".dds") || ends_with_ascii_ci(key, ".png"))) {
    pathIt = s_namedPbrTextureIndex.find(key.substr(0, key.size() - 4));
  }
  if (pathIt == s_namedPbrTextureIndex.end() && !ends_with_ascii_ci(key, ".dds") &&
      !ends_with_ascii_ci(key, ".png")) {
    if (pathIt = s_namedPbrTextureIndex.find(key + ".dds"); pathIt == s_namedPbrTextureIndex.end()) {
      pathIt = s_namedPbrTextureIndex.find(key + ".png");
    }
  }
  if (pathIt == s_namedPbrTextureIndex.end()) {
    return std::nullopt;
  }

  key = pathIt->first;
  if (const auto cached = s_namedPbrTextureCache.find(key); cached != s_namedPbrTextureCache.end()) {
    return cached->second;
  }
  if (s_failedNamedPbrTextures.contains(key)) {
    return std::nullopt;
  }

  auto handle = load_named_texture_file(pathIt->second);
  if (!handle) {
    s_failedNamedPbrTextures.insert(key);
    return std::nullopt;
  }

  s_namedPbrTextureCache.emplace(key, handle);
  return handle;
}

std::optional<TextureHandle> find_replacement(const GXTexObj_& obj) noexcept {
  ZoneScoped;

  if (!g_config.allowTextureReplacements) {
    return std::nullopt;
  }

  const RuntimeTextureKey key = build_runtime_key(obj);
  const auto* paths = find_replacement_paths(key);
  if (paths == nullptr) {
    report_missing_key(key, obj);
    return std::nullopt;
  }

  if (const auto* cached = find_cached_replacement(key); cached != nullptr) {
    return *cached;
  }

  if (s_failedKeys.contains(key)) {
    return std::nullopt;
  }

  auto handle = load_replacement_texture(key, *paths);
  if (!handle) {
    return std::nullopt;
  }

  cache_replacement(key, handle);
  return handle;
}

std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept {
  const RuntimeTextureKey key = build_runtime_key(obj);
  return format_replacement_filename(key);
}

} // namespace aurora::gfx::texture_replacement

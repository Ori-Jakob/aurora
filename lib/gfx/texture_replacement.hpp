#pragma once

#include "texture.hpp"
#include "texture_convert.hpp"
#include <aurora/texture.hpp>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::gfx::texture_replacement {
struct ReplacementResult {
  TextureHandle handle;
  uint64_t id = 0;
};

struct StreamingStats {
  uint64_t pendingLoads = 0;
  uint64_t publishes = 0;
  uint64_t publishBytes = 0;
};

constexpr wgpu::TextureFormat legacy_storage_format(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::RGBA8UnormSrgb:
    return wgpu::TextureFormat::RGBA8Unorm;
  case wgpu::TextureFormat::BGRA8UnormSrgb:
    return wgpu::TextureFormat::BGRA8Unorm;
  case wgpu::TextureFormat::BC1RGBAUnormSrgb:
    return wgpu::TextureFormat::BC1RGBAUnorm;
  case wgpu::TextureFormat::BC3RGBAUnormSrgb:
    return wgpu::TextureFormat::BC3RGBAUnorm;
  case wgpu::TextureFormat::BC7RGBAUnormSrgb:
    return wgpu::TextureFormat::BC7RGBAUnorm;
  default:
    return format;
  }
}

constexpr bool legacy_base_accepts(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::BC4RUnorm:
  case wgpu::TextureFormat::BC4RSnorm:
  case wgpu::TextureFormat::BC5RGSnorm:
  case wgpu::TextureFormat::BC6HRGBUfloat:
  case wgpu::TextureFormat::BC6HRGBFloat:
    return false;
  default:
    return true;
  }
}

enum class MipFit { Standard, Clamped, Invalid };

// C1 D8 (spec-w8.md:61): exact size, or a repeat of the previous level that is not smaller than expected
constexpr MipFit classify_mip_level(uint32_t baseWidth, uint32_t baseHeight, uint32_t level, uint32_t prevWidth,
                                    uint32_t prevHeight, uint32_t width, uint32_t height) noexcept {
  const uint32_t expectedWidth = level >= 32 ? 1u : std::max(baseWidth >> level, 1u);
  const uint32_t expectedHeight = level >= 32 ? 1u : std::max(baseHeight >> level, 1u);
  if (width == expectedWidth && height == expectedHeight) {
    return MipFit::Standard;
  }
  if (width == prevWidth && height == prevHeight && width >= expectedWidth && height >= expectedHeight) {
    return MipFit::Clamped;
  }
  return MipFit::Invalid;
}

// Unprobed: the handle came from a raw entry or the thumbnail tier, not a full encoded load
enum class ChainVerdict : uint8_t { Unprobed, Mipless, Internal, Complete, Short, Clamped, Invalid, Broken };

struct ChainReport {
  ChainVerdict verdict = ChainVerdict::Unprobed;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint8_t keptLevels = 0, generatedLevels = 0, fullLevels = 0;
  bool repairable = false, repaired = false;
  uint32_t repairMicros = 0;
  bool generatable = false, generated = false;
};

// Latched from Settings::textureMaterials; a flip reloads only chains whose bytes depend on it.
void set_mip_repair(bool enabled) noexcept;
// Latched from Settings::generateMissingMips; a flip reloads only chains whose bytes depend on it
void set_mip_generation(bool enabled) noexcept;
bool mip_repair_built() noexcept;

struct ReplacementTextureDescriptors {
  wgpu::TextureDescriptor texture;
  wgpu::TextureViewDescriptor view;
};

ReplacementTextureDescriptors replacement_texture_descriptors(const ConvertedTexture& replacement,
                                                              std::string_view label) noexcept;

void shutdown() noexcept;
StreamingStats process_streaming() noexcept;
std::optional<ReplacementResult> find_pointer_replacement(const GXTexObj_& obj) noexcept;
std::optional<ReplacementResult> find_source_replacement(const GXTexObj_& obj,
                                                         const texture::TextureSourceKey& sourceKey) noexcept;
bool should_build_source_key() noexcept;
// Trace capture bypasses replacements so every texture is read from guest memory and replays without the registry.
void set_bypass(bool bypass) noexcept;
// Replay needs replacement pixels on first use, not after a timing-dependent thumbnail stage.
void set_synchronous_loads(bool synchronous) noexcept;
bool has_replacement(const GXTexObj_& obj) noexcept;
bool has_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut) noexcept;
std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept;
std::string build_texture_replacement_name(const texture::TextureSourceKey& sourceKey) noexcept;

enum class CompanionRole : uint8_t { Albedo, Delight, Nrm, Orm, Emit, GrassMask, Mat };
inline constexpr uint32_t CompanionRoleCount = 7;

struct CompanionName {
  texture::TextureSourceKey key;
  CompanionRole role = CompanionRole::Albedo;
  uint32_t mip = 0;
};

struct CompanionFile {
  uint64_t id = 0;
  std::filesystem::path path;
};

std::string_view companion_role_name(CompanionRole role) noexcept;
std::optional<CompanionName> parse_companion_filename(std::string_view filename) noexcept;
std::string companion_mip_filename(std::string_view path, uint32_t mip);
std::optional<CompanionFile> find_companion(const texture::TextureSourceKey& key, CompanionRole role);
uint32_t companion_roles(const texture::TextureSourceKey& key) noexcept;
uint64_t companion_generation() noexcept;

namespace testing {
void set_workers_paused(bool paused) noexcept;
void set_worker_count(uint32_t count) noexcept;
bool wait_for_completions(uint64_t id, uint32_t count, uint32_t timeoutMs) noexcept;
std::optional<ConvertedTexture> load_replacement_texture(const texture::ReplacementRegistration& registration) noexcept;
std::optional<ConvertedTexture> load_replacement_texture(const texture::ReplacementRegistration& registration,
                                                         bool repair, ChainReport* report,
                                                         bool generate = false) noexcept;
std::optional<ConvertedTexture> load_replacement_file(const std::filesystem::path& path, bool repair,
                                                      ChainReport* report = nullptr, bool generate = false) noexcept;
std::optional<ByteBuffer> decode_replacement_level(wgpu::TextureFormat format, uint32_t width, uint32_t height,
                                                   ArrayRef<uint8_t> level) noexcept;
bool seed_resident_full(const texture::ReplacementRegistration& registration, wgpu::TextureFormat format,
                        uint32_t width, uint32_t height, uint32_t mips, const ChainReport& chain) noexcept;
std::optional<ChainReport> resident_chain(const texture::ReplacementKey& key) noexcept;
bool mip_generation() noexcept;
bool has_thumbnail(uint64_t id) noexcept;
} // namespace testing
} // namespace aurora::gfx::texture_replacement

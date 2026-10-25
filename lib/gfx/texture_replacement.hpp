#pragma once

#include "texture.hpp"
#include <aurora/texture.hpp>
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
} // namespace testing
} // namespace aurora::gfx::texture_replacement

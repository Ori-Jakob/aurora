#include "companion_fixtures.hpp"

#include "gfx/companion_cache.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/pipeline.hpp"

#include <aurora/texture.hpp>
#include <dolphin/gx.h>
#include <fmt/format.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <string_view>
#include <system_error>
#include <utility>

namespace render_demo {
namespace {
constexpr u32 GroundSize = 64;
constexpr std::string_view NrmFixture = "tex1_16x16_000000000000a001_14";
constexpr std::string_view MatFixture = "tex1_16x16_000000000000c001_14";
constexpr std::array<std::string_view, 3> NrmRoles{".nrm.dds", ".orm.dds", ".albedo.dds"};
constexpr std::array<std::string_view, 4> MatLevels{"", "_mip1", "_mip2", "_mip3"};

std::filesystem::path s_root;
aurora::texture::ReplacementGroup s_group;
bool s_registered = false;

bool copy_fixture(const std::filesystem::path& from, const std::filesystem::path& to, std::string& error) {
  std::error_code ec;
  std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    error = fmt::format("copying {} failed: {}", from.string(), ec.message());
    return false;
  }
  return true;
}
} // namespace

void prepare_companion_loads() noexcept { aurora::gfx::texture_replacement::set_synchronous_loads(true); }

bool load_companion_fixtures(bool mat, std::span<const u8> groundBase, std::string& error) {
  const auto key = aurora::texture::compute_source_key(GX_TF_RGBA8, GroundSize, GroundSize, groundBase);
  if (!key) {
    error = "the ground texture has no source key";
    return false;
  }
  const std::string stem = fmt::format("tex1_{}x{}_{:016x}_{}", key->width, key->height, key->textureHash, key->format);
  std::error_code ec;
  if (s_root.empty()) {
    s_root = std::filesystem::temp_directory_path(ec) /
             fmt::format("render_demo_companions_{}", std::chrono::steady_clock::now().time_since_epoch().count());
  }
  const std::filesystem::path dir = s_root / (mat ? "mat" : "nrm");
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    error = fmt::format("creating {} failed: {}", dir.string(), ec.message());
    return false;
  }
  const std::filesystem::path fixtures{RENDER_DEMO_MATERIAL_FIXTURES};
  if (mat) {
    for (const std::string_view level : MatLevels) {
      const std::string from = fmt::format("{}{}.mat.dds", MatFixture, level);
      if (!copy_fixture(fixtures / from, dir / fmt::format("{}{}.mat.dds", stem, level), error)) {
        return false;
      }
    }
  } else {
    for (const std::string_view role : NrmRoles) {
      if (!copy_fixture(fixtures / fmt::format("{}{}", NrmFixture, role), dir / fmt::format("{}{}", stem, role),
                        error)) {
        return false;
      }
    }
  }
  if (std::exchange(s_registered, false)) {
    aurora::texture::unregister_replacements(s_group);
  }
  s_group = aurora::texture::load_replacement_directory(dir);
  s_registered = true;
  if (s_group.companions.empty()) {
    error = fmt::format("{} indexed no companion", dir.string());
    return false;
  }
  return true;
}

void release_companion_fixtures() noexcept {
  if (std::exchange(s_registered, false)) {
    aurora::texture::unregister_replacements(s_group);
    s_group = {};
  }
  if (!s_root.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(s_root, ec);
    s_root.clear();
  }
}

CompanionCounts companion_counts() noexcept {
  const auto stats = aurora::gfx::companion_cache::testing::stats();
  return {
      .loaded = stats.loaded,
      .rejected = stats.rejected,
      .companionSkips = aurora::gx::bind_counts().companionSkips,
  };
}
} // namespace render_demo

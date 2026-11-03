#include "companion_fixtures.hpp"

#include "graph_checks.hpp"

#include "gfx/companion_cache.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/pipeline.hpp"

#include <aurora/texture.hpp>
#include <dolphin/gx.h>
#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace render_demo {
namespace {
constexpr u32 GroundSize = 64;
constexpr std::string_view NrmFixture = "tex1_16x16_000000000000a001_14";
constexpr std::string_view MatFixture = "tex1_16x16_000000000000c001_14";
constexpr std::array<std::string_view, 3> NrmRoles{".nrm.dds", ".orm.dds", ".albedo.dds"};
constexpr std::array<std::string_view, 4> MatLevels{"", "_mip1", "_mip2", "_mip3"};
constexpr uint32_t DxgiBc4 = 80;
constexpr uint32_t DxgiBc5 = 83;
constexpr std::string_view TiltNrm = "tilt.nrm.dds";

std::filesystem::path s_root;
aurora::texture::ReplacementGroup s_group;
bool s_registered = false;
aurora::texture::MaterialManifestRegistration s_manifest;

bool copy_fixture(const std::filesystem::path& from, const std::filesystem::path& to, std::string& error) {
  std::error_code ec;
  std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    error = fmt::format("copying {} failed: {}", from.string(), ec.message());
    return false;
  }
  return true;
}

void put_u32(std::vector<u8>& out, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    out.push_back(static_cast<u8>(value >> (8 * i)));
  }
}

// One 4x4 block per channel with equal endpoints and index 0 decodes to exactly that code everywhere
bool write_flat_dds(const std::filesystem::path& path, uint32_t dxgi, std::span<const u8> codes, std::string& error) {
  std::vector<u8> bytes;
  put_u32(bytes, 0x20534444u);
  const std::array<uint32_t, 18> header{124, 0x81007, 4, 4, static_cast<uint32_t>(8 * codes.size()), 0, 1};
  for (const uint32_t word : header) {
    put_u32(bytes, word);
  }
  put_u32(bytes, 32);
  put_u32(bytes, 0x4);
  put_u32(bytes, 0x30315844u);
  for (uint32_t i = 0; i < 5; ++i) {
    put_u32(bytes, 0);
  }
  put_u32(bytes, 0x1000);
  for (uint32_t i = 0; i < 4; ++i) {
    put_u32(bytes, 0);
  }
  for (const uint32_t word : {dxgi, 3u, 0u, 1u, 0u}) {
    put_u32(bytes, word);
  }
  for (const u8 code : codes) {
    bytes.insert(bytes.end(), {code, code, 0, 0, 0, 0, 0, 0});
  }
  std::ofstream file{path, std::ios::binary};
  file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    error = fmt::format("writing {} failed", path.string());
    return false;
  }
  return true;
}

std::optional<std::string> ground_stem(std::span<const u8> groundBase) {
  const auto key = aurora::texture::compute_source_key(GX_TF_RGBA8, GroundSize, GroundSize, groundBase);
  if (!key) {
    return std::nullopt;
  }
  return fmt::format("tex1_{}x{}_{:016x}_{}", key->width, key->height, key->textureHash, key->format);
}

bool make_dir(const std::filesystem::path& dir, std::string& error) {
  std::error_code ec;
  if (s_root.empty()) {
    s_root = std::filesystem::temp_directory_path(ec) /
             fmt::format("render_demo_companions_{}", std::chrono::steady_clock::now().time_since_epoch().count());
  }
  std::filesystem::create_directories(s_root / dir, ec);
  if (ec) {
    error = fmt::format("creating {} failed: {}", (s_root / dir).string(), ec.message());
    return false;
  }
  return true;
}
} // namespace

void prepare_companion_loads() noexcept { aurora::gfx::texture_replacement::set_synchronous_loads(true); }

bool load_companion_fixtures(CompanionFixture fixture, std::span<const u8> groundBase, std::string& error) {
  const std::optional<std::string> stem = ground_stem(groundBase);
  if (!stem) {
    error = "the ground texture has no source key";
    return false;
  }
  constexpr std::array<std::string_view, 3> Dirs{"nrm", "mat", "emit"};
  const std::string_view name = Dirs[static_cast<size_t>(fixture)];
  if (!make_dir(name, error)) {
    return false;
  }
  const std::filesystem::path dir = s_root / name;
  const std::filesystem::path fixtures{RENDER_DEMO_MATERIAL_FIXTURES};
  if (fixture == CompanionFixture::Mat) {
    for (const std::string_view level : MatLevels) {
      const std::string from = fmt::format("{}{}.mat.dds", MatFixture, level);
      if (!copy_fixture(fixtures / from, dir / fmt::format("{}{}.mat.dds", *stem, level), error)) {
        return false;
      }
    }
  } else if (fixture == CompanionFixture::Emit) {
    const std::array delight{CompanionDelightCode};
    if (!copy_fixture(fixtures / fmt::format("{}.emit.dds", NrmFixture), dir / fmt::format("{}.emit.dds", *stem),
                      error) ||
        !write_flat_dds(dir / fmt::format("{}.delight.dds", *stem), DxgiBc4, delight, error)) {
      return false;
    }
  } else {
    for (const std::string_view role : NrmRoles) {
      if (!copy_fixture(fixtures / fmt::format("{}{}", NrmFixture, role), dir / fmt::format("{}{}", *stem, role),
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

bool set_companion_manifest(bool maps, std::span<const u8> groundBase, std::string& error) {
  const std::optional<std::string> stem = ground_stem(groundBase);
  if (!stem || !make_dir("manifest", error)) {
    error = error.empty() ? std::string{"the ground texture has no source key"} : error;
    return false;
  }
  const std::filesystem::path dir = s_root / "manifest";
  std::string fields =
      fmt::format(R"("normalStrength": {}, "normalFlipY": true)", static_cast<double>(CompanionNormalStrength));
  if (maps) {
    const std::array tilt{CompanionTiltNrmCode, u8{128}};
    if (!write_flat_dds(dir / TiltNrm, DxgiBc5, tilt, error)) {
      return false;
    }
    fields = fmt::format(R"("maps": {{ "nrm": "{}" }}, "roughness": {})", TiltNrm,
                         static_cast<double>(CompanionManifestRoughness));
  }
  const std::filesystem::path path = dir / "materials.json";
  const std::string text = fmt::format(R"({{ "version": 1, "materials": {{ "{}": {{ {} }} }} }})", *stem, fields);
  {
    std::ofstream file{path, std::ios::binary};
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  const auto status =
      s_manifest.id == 0
          ? (s_manifest = aurora::texture::load_material_manifest(
                 path, {.layer = aurora::texture::MaterialLayer::Pack, .label = "render_demo companions"}))
                .status
          : aurora::texture::reload_material_manifest(s_manifest);
  if (status != aurora::texture::MaterialManifestStatus::Accepted) {
    error = fmt::format("the companion manifest {} was not accepted", path.string());
    return false;
  }
  return true;
}

void release_companion_fixtures() noexcept {
  aurora::texture::unregister_material_manifest(std::exchange(s_manifest, {}));
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
      .emit = stats.loadedEmit,
      .delight = stats.loadedDelight,
  };
}
} // namespace render_demo

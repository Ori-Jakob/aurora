#include "gx_lookup.hpp"

#include "registry.hpp"

#include "../../gx/texture.hpp"
#include "../../io.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace aurora::render::material {
namespace {
constexpr Module Log{"aurora::render::material"};
constexpr std::array<std::string_view, 6> StatusNames{"accepted",   "rejected",    "missing",
                                                      "unreadable", "unavailable", "superseded"};

std::atomic<uint64_t> g_lookups{0};
std::atomic<uint64_t> g_totalHits{0};
std::array<std::atomic<uint32_t>, 0x10000> g_hits{};
texture::MaterialManifestRegistration g_envRegistration;

gfx::TextureMaterialRef find_material(void*, const texture::TextureSourceKey& key) noexcept {
  const Snapshot* snap = latched_raw();
  const uint16_t id = snap->find(key);
  g_lookups.fetch_add(1, std::memory_order_relaxed);
  if (id != 0) {
    g_totalHits.fetch_add(1, std::memory_order_relaxed);
    g_hits[id].fetch_add(1, std::memory_order_relaxed);
  }
  return {id, snap->record(id).summary};
}

bool latch_material(void*) noexcept { return latch(); }

const gx::texture::MaterialLookup kLookup{.find = &find_material, .latch = &latch_material};

void on_publish(const std::shared_ptr<const Snapshot>& snapshot) noexcept {
  gx::texture::set_material_lookup(snapshot->liveCount != 0 ? &kLookup : nullptr);
}
} // namespace

void install_gx_lookup() noexcept {
  set_publish_observer(&on_publish);
  const char* path = SDL_getenv("AURORA_RENDER_MATERIALS");
  if (path == nullptr || *path == '\0') {
    return;
  }
  // Verification hook: replays get materials without a host
  g_envRegistration = texture::load_material_manifest(
      io::fs_path_from_string(path), {.layer = texture::MaterialLayer::User, .label = "AURORA_RENDER_MATERIALS"});
  Log.warn("AURORA_RENDER_MATERIALS manifest {}: {}", path, StatusNames[static_cast<size_t>(g_envRegistration.status)]);
}

void remove_gx_lookup() noexcept {
  texture::unregister_material_manifest(std::exchange(g_envRegistration, {}));
  set_publish_observer(nullptr);
  gx::texture::set_material_lookup(nullptr);
}

namespace testing {
gfx::TextureMaterialRef gx_find(const texture::TextureSourceKey& key) noexcept { return find_material(nullptr, key); }
} // namespace testing
} // namespace aurora::render::material

namespace aurora::texture {
MaterialUsage material_usage() {
  return {
      .materialCount = render::material::snapshot()->materialCount,
      .sourceKeyBuilds = gx::texture::source_key_builds_total(),
      .materialLookups = render::material::g_lookups.load(std::memory_order_relaxed),
      .materialHits = render::material::g_totalHits.load(std::memory_order_relaxed),
  };
}

uint64_t material_hits(std::string_view canonicalKey) {
  const auto key = parse_material_key(canonicalKey);
  if (!key) {
    return 0;
  }
  const uint16_t id = render::material::snapshot()->find(*key);
  return id != 0 ? render::material::g_hits[id].load(std::memory_order_relaxed) : 0;
}
} // namespace aurora::texture

#include "key_text.hpp"

#include "../../logging.hpp"

#include <atomic>

namespace aurora::texture {
namespace {
constexpr Module Log{"aurora::render::material"};

std::atomic_bool g_warned{false};

MaterialManifestRegistration unavailable() {
  if (!g_warned.exchange(true)) {
    Log.warn("Texture material manifests need aurora built with AURORA_ENABLE_RENDER");
  }
  return {0, MaterialManifestStatus::Unavailable};
}
} // namespace

MaterialManifestRegistration load_material_manifest(const std::filesystem::path&, MaterialManifestOptions) {
  return unavailable();
}

MaterialManifestRegistration register_material_manifest(std::string_view, MaterialManifestOptions) {
  return unavailable();
}

std::optional<MaterialManifestInfo> material_manifest_info(const MaterialManifestRegistration&) { return std::nullopt; }

MaterialManifestStatus reload_material_manifest(const MaterialManifestRegistration&) {
  return MaterialManifestStatus::Unavailable;
}

MaterialManifestStatus update_material_manifest(const MaterialManifestRegistration&, std::string_view) {
  return MaterialManifestStatus::Unavailable;
}

void unregister_material_manifest(const MaterialManifestRegistration&) {}

void unregister_material_manifests(std::span<const MaterialManifestRegistration>) {}

void clear_material_manifests() {}

void set_material_layer_enabled(MaterialLayer, bool) {}

std::optional<MaterialDiagnostic> check_material_manifest(std::string_view, MaterialLayer) {
  return MaterialDiagnostic{.detail = "render module disabled"};
}

MaterialReport material_report() { return {}; }

std::string material_manifest_json(bool) { return R"({"grassTypes":{},"materials":{},"version":1})"; }

std::string find_material_json(const TextureSourceKey&, bool) { return {}; }

std::string format_material_key(const TextureSourceKey& key) { return render::material::format_key_text(key); }

std::optional<TextureSourceKey> parse_material_key(std::string_view text) {
  return render::material::parse_key_text(text);
}
} // namespace aurora::texture

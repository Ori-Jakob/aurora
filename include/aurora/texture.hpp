#ifndef AURORA_TEXTURE_HPP
#define AURORA_TEXTURE_HPP

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace aurora::texture {

/// Wildcard hash values used by the replacement filename convention ("$" fields).
inline constexpr uint64_t kWildcardTextureHash = 0xFFFFFFFFFFFFFFFFull;
inline constexpr uint64_t kWildcardTlutHash = 0xFFFFFFFFFFFFFFFEull;

struct TextureSourceKey {
  uint64_t textureHash = 0;
  uint64_t tlutHash = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t format = 0;
  bool hasTlut = false;

  bool operator==(const TextureSourceKey&) const = default;
};

struct TexturePointerKey {
  const void* data = nullptr;

  bool operator==(const TexturePointerKey&) const = default;
};

using ReplacementKey = std::variant<TexturePointerKey, TextureSourceKey>;

struct RawTextureReplacement {
  std::span<const uint8_t> bytes;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mipCount = 1;
  uint32_t gxFormat = 0;
  std::string_view label = {};
};

struct ReplacementOptions {
  int32_t priority = 0;
};

struct ReplacementRegistration {
  uint64_t id = 0;
  ReplacementKey key;
};

enum class MaterialLayer : uint8_t { Game = 1, Pack = 2, Mod = 3, User = 4 };
enum class MaterialManifestStatus : uint8_t { Accepted, Rejected, Missing, Unreadable, Unavailable, Superseded };

struct MaterialManifestRegistration {
  uint64_t id = 0;
  MaterialManifestStatus status = MaterialManifestStatus::Unreadable;
};

struct ReplacementGroup {
  std::vector<ReplacementRegistration> registrations;
  // Companion maps and pack manifests found by load_replacement_directory; removed with the group
  std::vector<ReplacementRegistration> companions;
  std::vector<MaterialManifestRegistration> manifests;
};

/// Parses a replacement filename of the form "tex1_{w}x{h}[_m]_{texhash}[_{tluthash}]_{fmt}[_arb].dds|.png"
/// into the source key it addresses. Hash fields may be "$" (see the wildcard constants above).
/// Returns nullopt if the filename is invalid.
std::optional<TextureSourceKey> parse_replacement_filename(std::string_view filename) noexcept;

/// Computes the source key the draw path builds for any texture GX can describe, from its base level and TLUT bytes.
/// Returns nullopt for an unlisted format, a zero or over 0xFFFF dimension, a short base level, or a palette format
/// without a TLUT.
std::optional<TextureSourceKey> compute_source_key(uint32_t format, uint32_t width, uint32_t height,
                                                   std::span<const uint8_t> texture,
                                                   std::span<const uint8_t> tlut = {}) noexcept;

/// Callback source for replacement files that do not live on the filesystem.
///
/// read() loads the entire encoded file at `path` into outBytes, returning false if the file is
/// unavailable. It must be thread safe: it is invoked lazily from Aurora worker threads without
/// internal registry locks held. `userData` must remain valid until unregistration returns;
/// unregistration may wait for one in-flight read callback to finish.
///
/// The callback must not call back into aurora::texture.
struct VirtualFileSource {
  bool (*read)(void* userData, const char* path, std::vector<uint8_t>& outBytes) = nullptr;
  void* userData = nullptr;
};

/// Registers an encoded (.dds/.png) replacement served through callbacks instead of the filesystem.
/// `path` is a relative '/'-separated virtual path whose final component follows the replacement
/// filename convention and determines the key. Mip sidecars are probed lazily by deriving
/// "{stem}_mipN{ext}" from `path` and calling source.read for each level until it returns false.
///
/// Returns ID 0 (invalid) if the filename does not parse or source.read is null.
ReplacementRegistration register_virtual_replacement(std::string_view path, VirtualFileSource source,
                                                     ReplacementOptions options = {});

ReplacementRegistration register_replacement(ReplacementKey key, RawTextureReplacement replacement,
                                             ReplacementOptions options = {});
void unregister_replacement(const ReplacementRegistration& registration);
void unregister_replacements(std::span<const ReplacementRegistration> registrations);
void unregister_replacements(const ReplacementGroup& group);
void unregister_replacements(const ReplacementKey& key);
void clear_replacements();

ReplacementGroup load_replacement_directory(const std::filesystem::path& root, ReplacementOptions options = {});
void reload_replacement_directory(const std::filesystem::path& root, ReplacementGroup& group,
                                  ReplacementOptions options = {});

bool has_replacement(const GXTexObj* obj, const GXTlutObj* tlut = nullptr);

struct MaterialManifestOptions {
  MaterialLayer layer = MaterialLayer::Pack;
  int32_t priority = 0;
  std::string_view label = {};
  std::string_view base = {};
  bool writable = false;
};

struct MaterialDiagnostic {
  std::string label;
  std::string rule;
  std::string pointer;
  std::string detail;
};

struct MaterialManifestInfo {
  uint64_t id = 0;
  std::string label;
  std::filesystem::path path;
  std::string base;
  MaterialLayer layer = MaterialLayer::Pack;
  int32_t priority = 0;
  uint64_t sequence = 0;
  MaterialManifestStatus status = MaterialManifestStatus::Unreadable;
  bool writable = false;
  bool overridden = false;
  bool layerEnabled = false;
  uint32_t materialCount = 0;
  uint32_t grassTypeCount = 0;
  std::optional<MaterialDiagnostic> violation;
};

struct MaterialReport {
  std::vector<MaterialManifestInfo> manifests;
  std::vector<MaterialDiagnostic> diagnostics;
  uint32_t materialCount = 0;
  uint32_t grassTypeCount = 0;
  uint64_t generation = 0;
};

/// Texture material manifests merge in ascending (layer, priority, sequence) order, where sequence is registration
/// order; the last manifest to set a field wins. Every function is thread safe and may be called before aurora
/// initializes and after it shuts down.
/// `base` names the directory a manifest's map paths are relative to; a file defaults to its own directory.
/// Registration ids are never reused.
/// A missing or unreadable file keeps its registration; id 0 means the render module is disabled, the layer is
/// invalid, or the text is over 16 MiB.
MaterialManifestRegistration load_material_manifest(const std::filesystem::path& path,
                                                    MaterialManifestOptions options = {});
MaterialManifestRegistration register_material_manifest(std::string_view text, MaterialManifestOptions options = {});
std::optional<MaterialManifestInfo> material_manifest_info(const MaterialManifestRegistration& registration);
/// Reload re-reads a file (or re-checks registered text) and drops any override; update overrides the content for
/// this session without touching the file. When reloads or updates of one registration overlap, the one started last
/// wins and the others return Superseded and apply nothing.
MaterialManifestStatus reload_material_manifest(const MaterialManifestRegistration& registration);
MaterialManifestStatus update_material_manifest(const MaterialManifestRegistration& registration,
                                                std::string_view text);
void unregister_material_manifest(const MaterialManifestRegistration& registration);
void unregister_material_manifests(std::span<const MaterialManifestRegistration> registrations);
void clear_material_manifests();
void set_material_layer_enabled(MaterialLayer layer, bool enabled);
std::optional<MaterialDiagnostic> check_material_manifest(std::string_view text, MaterialLayer layer);
MaterialReport material_report();
std::string material_manifest_json(bool pretty = true);
std::string find_material_json(const TextureSourceKey& key, bool pretty = false);
std::string format_material_key(const TextureSourceKey& key);
std::optional<TextureSourceKey> parse_material_key(std::string_view text);

struct MaterialUsage {
  uint32_t materialCount = 0;
  uint64_t sourceKeyBuilds = 0;
  uint64_t materialLookups = 0;
  uint64_t materialHits = 0;
};

MaterialUsage material_usage();
uint64_t material_hits(std::string_view canonicalKey);

} // namespace aurora::texture

#endif

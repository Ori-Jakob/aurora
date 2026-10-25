#pragma once

#include "../json.hpp"

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::material {
enum class UserLayerState : uint8_t { Loaded, Missing, Invalid, NoPath };

struct UserApplyResult {
  bool applied = false;
  bool changedOnDisk = false;
  texture::MaterialManifestStatus status = texture::MaterialManifestStatus::Unavailable;
  std::optional<texture::MaterialDiagnostic> diagnostic;
  std::string message;
};

inline constexpr std::array<std::string_view, 4> StochasticFieldNames{"scale", "rotation", "contrast", "exponent"};

struct UserEntry {
  bool exists = false;
  std::vector<std::string> flags;
  std::string className;
  std::array<std::optional<double>, StochasticFieldNames.size()> stochastic;
  std::string neutral;
  std::string gate;
  std::vector<std::string> scope;
};

class UserLayer {
public:
  static constexpr std::string_view SessionLabel = "inspector session";

  UserLayer() = default;
  ~UserLayer();
  UserLayer(const UserLayer&) = delete;
  UserLayer& operator=(const UserLayer&) = delete;

  UserLayerState open();
  UserLayerState revert();
  UserApplyResult apply();

  bool set_flags(const texture::TextureSourceKey& key, std::span<const std::string_view> flags);
  bool set_class(const texture::TextureSourceKey& key, std::string_view name);
  bool set_stochastic(const texture::TextureSourceKey& key, std::string_view field, double value);
  bool set_fake_shadow(const texture::TextureSourceKey& key, std::string_view neutral, std::string_view gate,
                       std::span<const std::string_view> scope);
  bool add_entry(const texture::TextureSourceKey& key);
  bool reset_field(const texture::TextureSourceKey& key, std::string_view field);
  bool remove_entry(const texture::TextureSourceKey& key);

  [[nodiscard]] UserLayerState state() const noexcept { return m_state; }
  [[nodiscard]] bool opened() const noexcept { return m_opened; }
  [[nodiscard]] bool editable() const noexcept { return m_state != UserLayerState::Invalid && m_doc != nullptr; }
  [[nodiscard]] bool dirty() const noexcept { return m_dirty; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }
  [[nodiscard]] uint64_t session_registration() const noexcept { return m_sessionId; }
  [[nodiscard]] std::string text() const;
  [[nodiscard]] bool has_entry(const texture::TextureSourceKey& key) const;
  [[nodiscard]] std::string entry_json(const texture::TextureSourceKey& key) const;
  [[nodiscard]] UserEntry read_entry(const texture::TextureSourceKey& key) const;

  static texture::TextureSourceKey entry_key(texture::TextureSourceKey key) noexcept;

private:
  yyjson_mut_val* entry(const texture::TextureSourceKey& key, bool create) const;
  bool load(std::string_view text);
  bool registration_alive() const;
  UserApplyResult apply_session(const std::string& text);
  void prune(const texture::TextureSourceKey& key);

  UserLayerState m_state = UserLayerState::NoPath;
  json::MutDoc m_doc;
  std::optional<std::string> m_base;
  std::filesystem::path m_path;
  uint64_t m_registration = 0;
  uint64_t m_sessionId = 0;
  std::optional<std::string> m_sessionText;
  bool m_opened = false;
  bool m_dirty = false;
  bool m_backedUp = false;
};
} // namespace aurora::render::material

#include "user_layer.hpp"

#include "../../io.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <system_error>
#include <utility>

namespace aurora::render::material {
namespace {
using texture::MaterialLayer;
using texture::MaterialManifestStatus;

constexpr std::string_view EmptyManifest = "{\"version\":1,\"materials\":{}}";

std::optional<std::string> read_text(const std::filesystem::path& path) {
  std::error_code error;
  if (!std::filesystem::exists(path, error)) {
    return std::nullopt;
  }
  auto bytes = io::read_file(path);
  if (!bytes) {
    return std::string{};
  }
  return std::string{reinterpret_cast<const char*>(bytes->data()), bytes->size()};
}

bool write_text(const std::filesystem::path& path, std::string_view text) {
  return io::write_file_atomic(path, {reinterpret_cast<const uint8_t*>(text.data()), text.size()});
}

void set_member(yyjson_mut_doc* doc, yyjson_mut_val* object, std::string_view name, yyjson_mut_val* value) {
  yyjson_mut_val* key = yyjson_mut_strncpy(doc, name.data(), name.size());
  if (yyjson_mut_obj_getn(object, name.data(), name.size()) != nullptr) {
    yyjson_mut_obj_replace(object, key, value);
  } else {
    yyjson_mut_obj_add(object, key, value);
  }
}

yyjson_mut_val* child_object(yyjson_mut_doc* doc, yyjson_mut_val* object, std::string_view name) {
  yyjson_mut_val* child = yyjson_mut_obj_getn(object, name.data(), name.size());
  if (child == nullptr) {
    child = yyjson_mut_obj(doc);
    set_member(doc, object, name, child);
  }
  return yyjson_mut_is_obj(child) ? child : nullptr;
}

yyjson_mut_val* string_array(yyjson_mut_doc* doc, std::span<const std::string_view> values) {
  yyjson_mut_val* array = yyjson_mut_arr(doc);
  for (const std::string_view value : values) {
    yyjson_mut_arr_append(array, yyjson_mut_strncpy(doc, value.data(), value.size()));
  }
  return array;
}

std::string_view status_name(MaterialManifestStatus status) noexcept {
  switch (status) {
  case MaterialManifestStatus::Accepted:
    return "accepted";
  case MaterialManifestStatus::Rejected:
    return "rejected";
  case MaterialManifestStatus::Missing:
    return "missing";
  case MaterialManifestStatus::Unreadable:
    return "unreadable";
  case MaterialManifestStatus::Unavailable:
    return "unavailable";
  case MaterialManifestStatus::Superseded:
    return "superseded";
  }
  return "unknown";
}
} // namespace

UserLayer::~UserLayer() {
  if (m_sessionId != 0) {
    texture::unregister_material_manifest({m_sessionId, MaterialManifestStatus::Accepted});
  }
}

texture::TextureSourceKey UserLayer::entry_key(texture::TextureSourceKey key) noexcept {
  if (key.hasTlut) {
    key.tlutHash = texture::kWildcardTlutHash;
  }
  return key;
}

bool UserLayer::load(std::string_view text) {
  const json::Doc parsed = json::read(text);
  if (parsed == nullptr || !yyjson_is_obj(yyjson_doc_get_root(parsed.get()))) {
    m_doc.reset();
    return false;
  }
  m_doc.reset(yyjson_doc_mut_copy(parsed.get(), nullptr));
  return m_doc != nullptr;
}

UserLayerState UserLayer::open() {
  m_opened = true;
  m_dirty = false;
  m_doc.reset();
  m_base.reset();
  m_registration = 0;
  const std::filesystem::path previous = std::exchange(m_path, {});
  const texture::MaterialReport report = texture::material_report();
  const texture::MaterialManifestInfo* winner = nullptr;
  for (const texture::MaterialManifestInfo& info : report.manifests) {
    if (info.layer == MaterialLayer::User && !info.path.empty()) {
      winner = &info;
    }
  }
  if (winner == nullptr) {
    m_state = UserLayerState::NoPath;
    m_base = m_sessionText;
    load(m_sessionText.value_or(std::string{EmptyManifest}));
    return m_state;
  }
  m_registration = winner->id;
  m_path = winner->path;
  if (m_path != previous) {
    m_backedUp = false;
  }
  if (winner->status == MaterialManifestStatus::Missing) {
    m_state = load(EmptyManifest) ? UserLayerState::Missing : UserLayerState::Invalid;
    return m_state;
  }
  m_base = read_text(m_path);
  const bool valid = winner->status == MaterialManifestStatus::Accepted && m_base && load(*m_base);
  m_state = valid ? UserLayerState::Loaded : UserLayerState::Invalid;
  if (!valid) {
    m_doc.reset();
  }
  return m_state;
}

UserLayerState UserLayer::revert() { return open(); }

std::string UserLayer::text() const { return m_doc != nullptr ? json::write(m_doc.get(), true) : std::string{}; }

yyjson_mut_val* UserLayer::entry(const texture::TextureSourceKey& key, bool create) const {
  if (m_doc == nullptr) {
    return nullptr;
  }
  yyjson_mut_doc* doc = m_doc.get();
  yyjson_mut_val* root = yyjson_mut_doc_get_root(doc);
  if (!yyjson_mut_is_obj(root)) {
    return nullptr;
  }
  yyjson_mut_val* materials = yyjson_mut_obj_get(root, "materials");
  if (materials == nullptr) {
    if (!create) {
      return nullptr;
    }
    materials = child_object(doc, root, "materials");
  }
  if (!yyjson_mut_is_obj(materials)) {
    return nullptr;
  }
  const std::string name = texture::format_material_key(entry_key(key));
  yyjson_mut_val* value = yyjson_mut_obj_getn(materials, name.data(), name.size());
  if (value == nullptr && create) {
    value = child_object(doc, materials, name);
  }
  return yyjson_mut_is_obj(value) ? value : nullptr;
}

bool UserLayer::has_entry(const texture::TextureSourceKey& key) const { return entry(key, false) != nullptr; }

UserEntry UserLayer::read_entry(const texture::TextureSourceKey& key) const {
  UserEntry out;
  yyjson_mut_val* value = entry(key, false);
  if (value == nullptr) {
    return out;
  }
  out.exists = true;
  const auto text_of = [](yyjson_mut_val* item) {
    return yyjson_mut_is_str(item) ? std::string{yyjson_mut_get_str(item), yyjson_mut_get_len(item)} : std::string{};
  };
  const auto strings = [&](yyjson_mut_val* array, std::vector<std::string>& into) {
    yyjson_mut_val* item = nullptr;
    yyjson_mut_arr_iter iter = yyjson_mut_arr_iter_with(array);
    while ((item = yyjson_mut_arr_iter_next(&iter)) != nullptr) {
      into.push_back(text_of(item));
    }
  };
  strings(yyjson_mut_obj_get(value, "flags"), out.flags);
  out.className = text_of(yyjson_mut_obj_get(value, "class"));
  if (yyjson_mut_val* stochastic = yyjson_mut_obj_get(value, "stochastic"); yyjson_mut_is_obj(stochastic)) {
    for (size_t i = 0; i < StochasticFieldNames.size(); ++i) {
      const std::string_view name = StochasticFieldNames[i];
      yyjson_mut_val* number = yyjson_mut_obj_getn(stochastic, name.data(), name.size());
      if (yyjson_mut_is_num(number)) {
        out.stochastic[i] = yyjson_mut_get_num(number);
      }
    }
  }
  if (yyjson_mut_val* shadow = yyjson_mut_obj_get(value, "fakeShadow"); yyjson_mut_is_obj(shadow)) {
    out.neutral = text_of(yyjson_mut_obj_get(shadow, "neutral"));
    out.gate = text_of(yyjson_mut_obj_get(shadow, "gate"));
    strings(yyjson_mut_obj_get(shadow, "scope"), out.scope);
  }
  return out;
}

bool UserLayer::add_entry(const texture::TextureSourceKey& key) {
  if (!editable() || entry(key, true) == nullptr) {
    return false;
  }
  m_dirty = true;
  return true;
}

void UserLayer::prune(const texture::TextureSourceKey& key) {
  yyjson_mut_val* value = entry(key, false);
  if (value == nullptr) {
    return;
  }
  for (const std::string_view nested : {std::string_view{"stochastic"}, std::string_view{"fakeShadow"}}) {
    yyjson_mut_val* child = yyjson_mut_obj_getn(value, nested.data(), nested.size());
    if (yyjson_mut_is_obj(child) && yyjson_mut_obj_size(child) == 0) {
      yyjson_mut_obj_remove_keyn(value, nested.data(), nested.size());
    }
  }
  if (yyjson_mut_obj_size(value) == 0) {
    remove_entry(key);
  }
}

bool UserLayer::set_flags(const texture::TextureSourceKey& key, std::span<const std::string_view> flags) {
  if (!editable()) {
    return false;
  }
  if (flags.empty()) {
    return reset_field(key, "flags");
  }
  yyjson_mut_val* value = entry(key, true);
  if (value == nullptr) {
    return false;
  }
  set_member(m_doc.get(), value, "flags", string_array(m_doc.get(), flags));
  m_dirty = true;
  return true;
}

bool UserLayer::set_class(const texture::TextureSourceKey& key, std::string_view name) {
  if (!editable()) {
    return false;
  }
  if (name.empty()) {
    return reset_field(key, "class");
  }
  yyjson_mut_val* value = entry(key, true);
  if (value == nullptr) {
    return false;
  }
  set_member(m_doc.get(), value, "class", yyjson_mut_strncpy(m_doc.get(), name.data(), name.size()));
  m_dirty = true;
  return true;
}

bool UserLayer::set_stochastic(const texture::TextureSourceKey& key, std::string_view field, double value) {
  if (!editable() || std::ranges::find(StochasticFieldNames, field) == StochasticFieldNames.end()) {
    return false;
  }
  yyjson_mut_val* material = entry(key, true);
  yyjson_mut_val* stochastic = material != nullptr ? child_object(m_doc.get(), material, "stochastic") : nullptr;
  if (stochastic == nullptr) {
    return false;
  }
  set_member(m_doc.get(), stochastic, field, yyjson_mut_real(m_doc.get(), value));
  m_dirty = true;
  return true;
}

bool UserLayer::set_fake_shadow(const texture::TextureSourceKey& key, std::string_view neutral, std::string_view gate,
                                std::span<const std::string_view> scope) {
  if (!editable()) {
    return false;
  }
  if (neutral.empty() && gate.empty() && scope.empty()) {
    return reset_field(key, "fakeShadow");
  }
  yyjson_mut_val* material = entry(key, true);
  if (material == nullptr) {
    return false;
  }
  yyjson_mut_doc* doc = m_doc.get();
  yyjson_mut_val* shadow = yyjson_mut_obj(doc);
  if (!neutral.empty()) {
    yyjson_mut_obj_add(shadow, yyjson_mut_str(doc, "neutral"), yyjson_mut_strncpy(doc, neutral.data(), neutral.size()));
  }
  if (!gate.empty()) {
    yyjson_mut_obj_add(shadow, yyjson_mut_str(doc, "gate"), yyjson_mut_strncpy(doc, gate.data(), gate.size()));
  }
  if (!scope.empty()) {
    yyjson_mut_obj_add(shadow, yyjson_mut_str(doc, "scope"), string_array(doc, scope));
  }
  set_member(doc, material, "fakeShadow", shadow);
  m_dirty = true;
  return true;
}

bool UserLayer::reset_field(const texture::TextureSourceKey& key, std::string_view field) {
  if (!editable()) {
    return false;
  }
  yyjson_mut_val* value = entry(key, false);
  if (value == nullptr) {
    return true;
  }
  const size_t slash = field.find('/');
  if (slash == std::string_view::npos) {
    yyjson_mut_obj_remove_keyn(value, field.data(), field.size());
  } else {
    const std::string_view parent = field.substr(0, slash);
    const std::string_view child = field.substr(slash + 1);
    yyjson_mut_val* object = yyjson_mut_obj_getn(value, parent.data(), parent.size());
    if (yyjson_mut_is_obj(object)) {
      yyjson_mut_obj_remove_keyn(object, child.data(), child.size());
    }
  }
  prune(key);
  m_dirty = true;
  return true;
}

bool UserLayer::remove_entry(const texture::TextureSourceKey& key) {
  if (!editable()) {
    return false;
  }
  yyjson_mut_val* materials = yyjson_mut_obj_get(yyjson_mut_doc_get_root(m_doc.get()), "materials");
  if (!yyjson_mut_is_obj(materials)) {
    return true;
  }
  const std::string name = texture::format_material_key(entry_key(key));
  yyjson_mut_obj_remove_keyn(materials, name.data(), name.size());
  m_dirty = true;
  return true;
}

bool UserLayer::registration_alive() const {
  const auto info = texture::material_manifest_info({m_registration, MaterialManifestStatus::Accepted});
  return info && info->layer == MaterialLayer::User && info->path == m_path;
}

UserApplyResult UserLayer::apply_session(const std::string& text) {
  UserApplyResult result;
  if (auto diagnostic = texture::check_material_manifest(text, MaterialLayer::User)) {
    result.message = fmt::format("{} at '{}'", diagnostic->rule, diagnostic->pointer);
    result.diagnostic = std::move(diagnostic);
    return result;
  }
  const texture::MaterialManifestRegistration next =
      texture::register_material_manifest(text, {.layer = MaterialLayer::User, .priority = 0, .label = SessionLabel});
  result.status = next.status;
  if (next.status != MaterialManifestStatus::Accepted) {
    if (const auto info = texture::material_manifest_info(next)) {
      result.diagnostic = info->violation;
    }
    texture::unregister_material_manifest(next);
    result.message = fmt::format("session manifest {}", status_name(next.status));
    return result;
  }
  if (m_sessionId != 0) {
    texture::unregister_material_manifest({m_sessionId, MaterialManifestStatus::Accepted});
  }
  m_sessionId = next.id;
  m_sessionText = text;
  m_base = text;
  m_dirty = false;
  result.applied = true;
  return result;
}

UserApplyResult UserLayer::apply() {
  UserApplyResult result;
  if (!editable()) {
    result.message = "the user manifest is not editable";
    return result;
  }
  const std::string text = this->text();
  if (m_state == UserLayerState::NoPath) {
    return apply_session(text);
  }
  if (!registration_alive()) {
    result.message = "the user manifest registration changed; reopen it";
    return result;
  }
  if (auto diagnostic = texture::check_material_manifest(text, MaterialLayer::User)) {
    result.message = fmt::format("{} at '{}'", diagnostic->rule, diagnostic->pointer);
    result.diagnostic = std::move(diagnostic);
    return result;
  }
  if (read_text(m_path) != m_base) {
    result.changedOnDisk = true;
    result.message = "the file changed on disk; revert to reload it";
    return result;
  }
  if (m_base && !m_backedUp) {
    std::filesystem::path backup = m_path;
    backup += ".bak";
    if (!write_text(backup, *m_base)) {
      result.message = "could not write the backup";
      return result;
    }
    m_backedUp = true;
  }
  if (!write_text(m_path, text)) {
    result.message = "could not write the file";
    return result;
  }
  const texture::MaterialManifestRegistration registration{m_registration, MaterialManifestStatus::Accepted};
  result.status = texture::reload_material_manifest(registration);
  if (result.status != MaterialManifestStatus::Accepted) {
    if (const auto info = texture::material_manifest_info(registration)) {
      result.diagnostic = info->violation;
    }
    if (m_base) {
      write_text(m_path, *m_base);
    } else {
      std::error_code error;
      std::filesystem::remove(m_path, error);
    }
    texture::reload_material_manifest(registration);
    result.message = fmt::format("reload {}; restored the previous file", status_name(result.status));
    return result;
  }
  m_base = text;
  m_state = UserLayerState::Loaded;
  m_dirty = false;
  result.applied = true;
  return result;
}
} // namespace aurora::render::material

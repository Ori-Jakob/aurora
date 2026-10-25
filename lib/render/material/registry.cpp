#include "registry.hpp"

#include "key_text.hpp"
#include "manifest.hpp"

#include "../../io.hpp"
#include "../../logging.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <tuple>
#include <utility>

namespace aurora::render::material {
struct KeyHash {
  size_t operator()(const texture::TextureSourceKey& key) const noexcept {
    return absl::HashOf(key.textureHash, key.tlutHash, key.width, key.height, key.format, key.hasTlut);
  }
};

struct KeyIndex {
  absl::flat_hash_map<texture::TextureSourceKey, uint16_t, KeyHash> ids;
};

namespace {
using texture::MaterialLayer;
using texture::MaterialManifestStatus;

constexpr Module Log{"aurora::render::material"};
constexpr size_t MaxManifestBytes = size_t{16} << 20;
constexpr size_t MaxMaterialId = 0xFFFF;

constexpr std::array<std::string_view, 6> NestedMaterialFields{"stochastic",   "maps",       "grass",
                                                               "displacement", "fakeShadow", "_provenance"};
constexpr std::array<std::string_view, 1> NestedGrassTypeFields{"clump"};
constexpr std::array<std::string_view, 5> RequiredGrassTypeFields{"height", "width", "densityPerM2", "rootColor",
                                                                  "tipColor"};
constexpr std::array<std::string_view, 4> FlagNames{"stochastic", "noPbr", "cutout", "unlit"};
constexpr std::array<std::string_view, 3> VertexColorNames{"tint", "bakedLight", "ignore"};
constexpr std::array<std::string_view, 3> NeutralNames{"white", "clear", "skip"};
constexpr std::array<std::string_view, 3> GateNames{"light", "cloud", "ao"};
constexpr std::array<std::string_view, 13> ScopeNames{"STATIC", "DYNAMIC",  "SKINNED",  "TERRAIN", "FOLIAGE",
                                                      "WATER",  "SKY",      "PARTICLE", "DECAL",   "UI",
                                                      "PLAYER", "EMISSIVE", "OVERLAY"};

template <typename List>
size_t index_of(const List& list, std::string_view value) noexcept {
  return static_cast<size_t>(std::find(list.begin(), list.end(), value) - list.begin());
}

template <typename List>
bool contains(const List& list, std::string_view value) noexcept {
  return index_of(list, value) < list.size();
}

std::string_view text_of(yyjson_val* val) noexcept {
  return yyjson_is_str(val) ? std::string_view{unsafe_yyjson_get_str(val), unsafe_yyjson_get_len(val)}
                            : std::string_view{};
}

yyjson_val* member(yyjson_val* object, std::string_view key) noexcept {
  return yyjson_obj_getn(object, key.data(), key.size());
}

yyjson_val* root_of(const std::shared_ptr<const json::Doc>& doc) noexcept {
  return doc != nullptr && *doc != nullptr ? yyjson_doc_get_root(doc->get()) : nullptr;
}

bool valid_layer(MaterialLayer layer) noexcept { return layer >= MaterialLayer::Game && layer <= MaterialLayer::User; }

std::string write_doc(yyjson_doc* doc, bool pretty) {
  const yyjson_write_flag flags = pretty ? YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_NEWLINE_AT_END : 0;
  size_t length = 0;
  char* text = yyjson_write_opts(doc, flags, nullptr, &length, nullptr);
  if (text == nullptr) {
    return {};
  }
  std::string out{text, length};
  std::free(text);
  return out;
}

enum class Source : uint8_t { File, Text };

struct Registration {
  uint64_t id = 0;
  MaterialLayer layer = MaterialLayer::Pack;
  int32_t priority = 0;
  uint64_t sequence = 0;
  std::string label;
  std::string base;
  bool writable = false;
  Source source = Source::Text;
  std::filesystem::path path;
  std::string text;
  std::optional<std::string> overrideText;
  MaterialManifestStatus status = MaterialManifestStatus::Unreadable;
  std::optional<Violation> violation;
  std::shared_ptr<const json::Doc> doc;
  uint32_t materialCount = 0;
  uint32_t grassTypeCount = 0;
  uint64_t latestTicket = 0;
};

struct Outcome {
  MaterialManifestStatus status = MaterialManifestStatus::Unreadable;
  std::optional<Violation> violation;
  std::shared_ptr<const json::Doc> doc;
};

struct MergeWarning {
  std::string rule;
  std::string pointer;

  bool operator==(const MergeWarning&) const = default;
};

struct Slot {
  yyjson_val* value = nullptr;
  uint32_t supplier = 0;
  bool split = false;
  std::map<std::string, std::pair<yyjson_val*, uint32_t>, std::less<>> members;

  bool is_object() const noexcept { return split || yyjson_is_obj(value); }

  yyjson_val* find(std::string_view key) const noexcept {
    if (!split) {
      return member(value, key);
    }
    const auto it = members.find(key);
    return it != members.end() ? it->second.first : nullptr;
  }
};

using Entry = std::map<std::string, Slot, std::less<>>;
using Section = std::map<std::string, Entry, std::less<>>;

struct MergeOutput {
  json::MutDoc doc;
  std::string text;
  std::string mapBaseText;
  Section materials;
  uint32_t grassTypeCount = 0;
  std::vector<MergeWarning> warnings;
};

struct State {
  std::mutex mutex;
  std::mutex snapshotMutex;
  std::vector<Registration> registrations;
  bool packEnabled = true;
  bool modEnabled = true;
  uint64_t nextId = 0;
  uint64_t nextSequence = 0;
  uint64_t generationCounter = 0;
  uint64_t bindEpochCounter = 0;
  absl::flat_hash_map<texture::TextureSourceKey, uint16_t, KeyHash> interned;
  std::shared_ptr<const KeyTable> keyTable;
  std::string mapBaseText;
  std::vector<MergeWarning> mergeWarnings;
  std::set<std::tuple<std::string, std::string, std::string>> warnedRejections;
  bool warnedLayerSwitch = false;
  PublishObserver observer = nullptr;
  std::shared_ptr<const Snapshot> current;
  std::shared_ptr<const Snapshot> latched;
  std::atomic<const Snapshot*> latchedRaw{nullptr};
  std::atomic<uint64_t> generation{0};

  State();

  bool layer_enabled(MaterialLayer layer) const noexcept {
    return (layer != MaterialLayer::Pack || packEnabled) && (layer != MaterialLayer::Mod || modEnabled);
  }

  Registration* find(uint64_t id) noexcept {
    const auto it = std::find_if(registrations.begin(), registrations.end(),
                                 [id](const Registration& entry) { return entry.id == id; });
    return it != registrations.end() ? &*it : nullptr;
  }
};

std::atomic<void (*)(uint64_t)> g_checkHook{nullptr};

void merge_fields(Entry& entry, yyjson_val* source, std::span<const std::string_view> nested, uint32_t supplier) {
  yyjson_val* key = nullptr;
  yyjson_obj_iter iter = yyjson_obj_iter_with(source);
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    yyjson_val* value = yyjson_obj_iter_get_val(key);
    const auto [it, inserted] = entry.try_emplace(std::string{text_of(key)});
    Slot& slot = it->second;
    const bool mergeable = contains(nested, it->first) && !inserted && slot.is_object() && yyjson_is_obj(value);
    if (!mergeable) {
      slot = Slot{.value = value, .supplier = supplier};
      continue;
    }
    if (!slot.split) {
      yyjson_val* child = nullptr;
      yyjson_obj_iter children = yyjson_obj_iter_with(slot.value);
      while ((child = yyjson_obj_iter_next(&children)) != nullptr) {
        slot.members.insert_or_assign(std::string{text_of(child)},
                                      std::pair{yyjson_obj_iter_get_val(child), slot.supplier});
      }
      slot.split = true;
      slot.value = nullptr;
    }
    yyjson_val* child = nullptr;
    yyjson_obj_iter children = yyjson_obj_iter_with(value);
    while ((child = yyjson_obj_iter_next(&children)) != nullptr) {
      slot.members.insert_or_assign(std::string{text_of(child)}, std::pair{yyjson_obj_iter_get_val(child), supplier});
    }
  }
}

void merge_section(Section& target, yyjson_val* source, std::span<const std::string_view> nested, uint32_t supplier) {
  yyjson_val* key = nullptr;
  yyjson_obj_iter iter = yyjson_obj_iter_with(source);
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    merge_fields(target[std::string{text_of(key)}], yyjson_obj_iter_get_val(key), nested, supplier);
  }
}

void check_merged(Section& grassTypes, Section& materials, std::vector<MergeWarning>& warnings) {
  for (auto it = grassTypes.begin(); it != grassTypes.end();) {
    const bool complete = std::all_of(RequiredGrassTypeFields.begin(), RequiredGrassTypeFields.end(),
                                      [&](std::string_view field) { return it->second.contains(field); });
    if (complete) {
      ++it;
      continue;
    }
    warnings.push_back({"MERGE-GRASS-TYPE-INCOMPLETE", pointer_join("/grassTypes", it->first)});
    it = grassTypes.erase(it);
  }
  for (auto& [key, entry] : materials) {
    const std::string at = pointer_join("/materials", key);
    if (const auto grass = entry.find("grass"); grass != entry.end()) {
      yyjson_val* type = grass->second.find("type");
      if (type == nullptr || !yyjson_is_str(type) || !grassTypes.contains(text_of(type))) {
        warnings.push_back({"MERGE-GRASS-TYPE-UNRESOLVED", pointer_join(at, "grass")});
        entry.erase(grass);
      }
    }
    if (const auto shadow = entry.find("fakeShadow");
        shadow != entry.end() &&
        (shadow->second.find("neutral") == nullptr || shadow->second.find("gate") == nullptr)) {
      warnings.push_back({"MERGE-FAKESHADOW-INCOMPLETE", pointer_join(at, "fakeShadow")});
      entry.erase(shadow);
    }
    if (const auto displacement = entry.find("displacement");
        displacement != entry.end() && displacement->second.find("amplitude") == nullptr) {
      warnings.push_back({"MERGE-DISPLACEMENT-INCOMPLETE", pointer_join(at, "displacement")});
      entry.erase(displacement);
    }
  }
}

yyjson_mut_val* copy_sorted(yyjson_mut_doc* doc, yyjson_val* value) {
  if (yyjson_is_obj(value)) {
    std::vector<std::pair<std::string_view, yyjson_val*>> members;
    members.reserve(yyjson_obj_size(value));
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(value);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      members.emplace_back(text_of(key), yyjson_obj_iter_get_val(key));
    }
    std::sort(members.begin(), members.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    yyjson_mut_val* out = yyjson_mut_obj(doc);
    for (const auto& [name, child] : members) {
      yyjson_mut_obj_add(out, yyjson_mut_strncpy(doc, name.data(), name.size()), copy_sorted(doc, child));
    }
    return out;
  }
  if (yyjson_is_arr(value)) {
    yyjson_mut_val* out = yyjson_mut_arr(doc);
    size_t index = 0;
    size_t count = 0;
    yyjson_val* item = nullptr;
    yyjson_arr_foreach(value, index, count, item) { yyjson_mut_arr_append(out, copy_sorted(doc, item)); }
    return out;
  }
  return yyjson_val_mut_copy(doc, value);
}

yyjson_mut_val* emit_section(yyjson_mut_doc* doc, const Section& section) {
  yyjson_mut_val* out = yyjson_mut_obj(doc);
  for (const auto& [name, entry] : section) {
    yyjson_mut_val* fields = yyjson_mut_obj(doc);
    for (const auto& [field, slot] : entry) {
      yyjson_mut_val* value = nullptr;
      if (slot.split) {
        value = yyjson_mut_obj(doc);
        for (const auto& [child, supplied] : slot.members) {
          yyjson_mut_obj_add(value, yyjson_mut_strncpy(doc, child.data(), child.size()),
                             copy_sorted(doc, supplied.first));
        }
      } else {
        value = copy_sorted(doc, slot.value);
      }
      yyjson_mut_obj_add(fields, yyjson_mut_strncpy(doc, field.data(), field.size()), value);
    }
    yyjson_mut_obj_add(out, yyjson_mut_strncpy(doc, name.data(), name.size()), fields);
  }
  return out;
}

std::map<std::string_view, uint32_t> map_suppliers(const Entry& entry) {
  std::map<std::string_view, uint32_t> out;
  const auto maps = entry.find("maps");
  if (maps == entry.end()) {
    return out;
  }
  const Slot& slot = maps->second;
  if (slot.split) {
    for (const auto& [role, supplied] : slot.members) {
      out.emplace(role, supplied.second);
    }
  } else {
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(slot.value);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      out.emplace(text_of(key), slot.supplier);
    }
  }
  return out;
}

MergeOutput merge(const std::vector<Contributor>& contributors) {
  MergeOutput out;
  Section grassTypes;
  for (size_t i = 0; i < contributors.size(); ++i) {
    yyjson_val* manifest = root_of(contributors[i].doc);
    const auto supplier = static_cast<uint32_t>(i + 1);
    merge_section(grassTypes, member(manifest, "grassTypes"), NestedGrassTypeFields, supplier);
    merge_section(out.materials, member(manifest, "materials"), NestedMaterialFields, supplier);
  }
  check_merged(grassTypes, out.materials, out.warnings);
  out.grassTypeCount = static_cast<uint32_t>(grassTypes.size());

  out.doc.reset(yyjson_mut_doc_new(nullptr));
  yyjson_mut_doc* doc = out.doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(doc);
  yyjson_mut_obj_add(root, yyjson_mut_str(doc, "grassTypes"), emit_section(doc, grassTypes));
  yyjson_mut_obj_add(root, yyjson_mut_str(doc, "materials"), emit_section(doc, out.materials));
  yyjson_mut_obj_add(root, yyjson_mut_str(doc, "version"), yyjson_mut_uint(doc, 1));
  yyjson_mut_doc_set_root(doc, root);
  out.text = json::write(doc, false);

  for (const auto& [key, entry] : out.materials) {
    for (const auto& [role, supplier] : map_suppliers(entry)) {
      out.mapBaseText += key;
      out.mapBaseText += '\0';
      out.mapBaseText += role;
      out.mapBaseText += '\0';
      out.mapBaseText += contributors[supplier - 1].base;
      out.mapBaseText += '\n';
    }
  }
  return out;
}

std::vector<std::array<uint32_t, MapRoleCount>> suppliers_by_id(const State& s, const Section& materials,
                                                                size_t idCount) {
  std::vector<std::array<uint32_t, MapRoleCount>> out(idCount);
  for (const auto& [key, entry] : materials) {
    const auto parsed = parse_key_text(key);
    const auto id = parsed ? s.interned.find(*parsed) : s.interned.end();
    if (id == s.interned.end() || id->second >= idCount) {
      continue;
    }
    for (const auto& [role, supplier] : map_suppliers(entry)) {
      if (const size_t index = index_of(MapRoleNames, role); index < MapRoleCount) {
        out[id->second][index] = supplier;
      }
    }
  }
  return out;
}

MaterialRecord decode_record(yyjson_val* entry, uint16_t id, const std::map<std::string_view, uint16_t>& classIds) {
  MaterialRecord record;
  record.id = id;
  size_t index = 0;
  size_t count = 0;
  yyjson_val* item = nullptr;
  yyjson_arr_foreach(member(entry, "flags"), index, count, item) {
    if (const size_t bit = index_of(FlagNames, text_of(item)); bit < FlagNames.size()) {
      record.summary |= 1u << bit;
    }
  }
  if (yyjson_val* shadow = member(entry, "fakeShadow"); yyjson_is_obj(shadow)) {
    record.summary |= FakeShadow;
    if (const size_t neutral = index_of(NeutralNames, text_of(member(shadow, "neutral")));
        neutral < NeutralNames.size()) {
      record.summary |= static_cast<uint32_t>(neutral + 1) << NeutralShift;
    }
    if (const size_t gate = index_of(GateNames, text_of(member(shadow, "gate"))); gate < GateNames.size()) {
      record.summary |= static_cast<uint32_t>(gate + 1) << GateShift;
    }
    if (yyjson_val* scope = member(shadow, "scope"); yyjson_is_arr(scope)) {
      record.summary |= ShadowScope;
      yyjson_arr_foreach(scope, index, count, item) {
        if (const size_t bit = index_of(ScopeNames, text_of(item)); bit < ScopeNames.size()) {
          record.shadowScope |= static_cast<uint16_t>(1u << bit);
        }
      }
    }
  }
  if (yyjson_is_obj(member(entry, "grass"))) {
    record.summary |= Grass;
  }
  if (yyjson_is_obj(member(entry, "displacement"))) {
    record.summary |= Displacement;
  }
  if (yyjson_val* stochastic = member(entry, "stochastic"); yyjson_is_obj(stochastic)) {
    const auto read = [&](std::string_view field, StochasticField bit, float& out) {
      if (yyjson_val* value = member(stochastic, field); yyjson_is_num(value)) {
        out = static_cast<float>(yyjson_get_num(value));
        record.stochasticSet |= bit;
      }
    };
    read("contrast", StochasticContrast, record.contrast);
    read("rotation", StochasticRotation, record.rotation);
    read("scale", StochasticScale, record.scale);
    read("exponent", StochasticExponent, record.exponent);
    if (record.stochasticSet != 0) {
      record.summary |= StochasticParams;
    }
  }
  if (yyjson_val* name = member(entry, "class"); yyjson_is_str(name)) {
    record.summary |= Class;
    if (const auto it = classIds.find(text_of(name)); it != classIds.end()) {
      record.classId = it->second;
    }
  }
  if (yyjson_val* maps = member(entry, "maps"); yyjson_is_obj(maps)) {
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(maps);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      if (const size_t role = index_of(MapRoleNames, text_of(key)); role < MapRoleCount) {
        record.summary |= map_role_bit(static_cast<uint32_t>(role));
      }
    }
  }
  if (const size_t mode = index_of(VertexColorNames, text_of(member(entry, "vertexColor")));
      mode < VertexColorNames.size()) {
    record.summary |= static_cast<uint32_t>(mode + 1) << VertexColorShift;
  }
  return record;
}

struct Built {
  std::shared_ptr<Snapshot> snapshot;
  bool bindChanged = false;
};

Built build_full(State& s, MergeOutput& merged, std::vector<Contributor> contributors, const Snapshot* previous) {
  auto next = std::make_shared<Snapshot>();
  size_t overflow = 0;
  std::shared_ptr<KeyTable> grown;
  for (const auto& [key, entry] : merged.materials) {
    const auto parsed = parse_key_text(key);
    if (!parsed || s.interned.contains(*parsed)) {
      continue;
    }
    const size_t tableSize = grown != nullptr ? grown->keys.size() : s.keyTable->keys.size();
    if (tableSize > MaxMaterialId) {
      ++overflow;
      continue;
    }
    if (grown == nullptr) {
      grown = std::make_shared<KeyTable>(*s.keyTable);
    }
    s.interned.emplace(*parsed, static_cast<uint16_t>(tableSize));
    grown->keys.push_back(*parsed);
    grown->names.push_back(key);
  }
  if (grown != nullptr) {
    s.keyTable = std::move(grown);
  }
  if (overflow != 0) {
    Log.warn("{} texture material key(s) did not get an id: only {} distinct keys fit", overflow, MaxMaterialId);
  }

  const size_t idCount = s.keyTable->keys.size();
  next->keyTable = s.keyTable;
  next->records.resize(idCount);
  next->entries.resize(idCount);
  next->mapSupplier = suppliers_by_id(s, merged.materials, idCount);
  next->mergedText = std::move(merged.text);
  next->merged = std::make_shared<const json::Doc>(yyjson_mut_doc_imut_copy(merged.doc.get(), nullptr));
  next->contributors = std::move(contributors);
  next->materialCount = static_cast<uint32_t>(merged.materials.size());
  next->grassTypeCount = merged.grassTypeCount;

  yyjson_val* materials = member(root_of(next->merged), "materials");
  std::vector<std::pair<uint16_t, yyjson_val*>> live;
  std::set<std::string_view> classNames;
  yyjson_val* key = nullptr;
  yyjson_obj_iter iter = yyjson_obj_iter_with(materials);
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    const auto parsed = parse_key_text(text_of(key));
    const auto id = parsed ? s.interned.find(*parsed) : s.interned.end();
    if (id == s.interned.end()) {
      continue;
    }
    yyjson_val* entry = yyjson_obj_iter_get_val(key);
    live.emplace_back(id->second, entry);
    if (yyjson_val* name = member(entry, "class"); yyjson_is_str(name)) {
      classNames.insert(text_of(name));
    }
  }
  std::map<std::string_view, uint16_t> classIds;
  next->classes.emplace_back();
  for (const std::string_view name : classNames) {
    classIds.emplace(name, static_cast<uint16_t>(next->classes.size()));
    next->classes.emplace_back(name);
  }

  auto index = std::make_shared<KeyIndex>();
  index->ids.reserve(live.size());
  for (const auto& [id, entry] : live) {
    const MaterialRecord record = decode_record(entry, id, classIds);
    next->records[id] = record;
    next->entries[id] = entry;
    index->ids.emplace(s.keyTable->keys[id], id);
    for (uint32_t bit = 0; bit < 32; ++bit) {
      next->summaryCount[bit] += record.summary >> bit & 1u;
    }
  }
  next->liveCount = static_cast<uint32_t>(live.size());
  next->index = std::move(index);

  bool bindChanged = false;
  if (previous != nullptr) {
    for (size_t id = 1; id < idCount && !bindChanged; ++id) {
      const MaterialRecord& before = previous->record(static_cast<uint32_t>(id));
      const MaterialRecord& after = next->records[id];
      bindChanged = (before.id != 0) != (after.id != 0) || before.summary != after.summary;
    }
  }
  return {std::move(next), bindChanged};
}

std::vector<Contributor> collect_contributors(const State& s) {
  std::vector<const Registration*> order;
  for (const Registration& entry : s.registrations) {
    if (entry.status == MaterialManifestStatus::Accepted && entry.doc != nullptr && s.layer_enabled(entry.layer)) {
      order.push_back(&entry);
    }
  }
  std::sort(order.begin(), order.end(), [](const Registration* a, const Registration* b) {
    return std::tie(a->layer, a->priority, a->sequence) < std::tie(b->layer, b->priority, b->sequence);
  });
  std::vector<Contributor> out;
  out.reserve(order.size());
  for (const Registration* entry : order) {
    out.push_back({entry->id, entry->label, entry->base, entry->layer, entry->priority, entry->sequence, entry->doc});
  }
  return out;
}

bool same_contributors(const std::vector<Contributor>& a, const std::vector<Contributor>& b) noexcept {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const Contributor& x, const Contributor& y) {
    return x.registration == y.registration && x.doc == y.doc;
  });
}

void swap_snapshot(State& s, std::shared_ptr<const Snapshot> next) {
  std::shared_ptr<const Snapshot> previous;
  {
    std::lock_guard lock{s.snapshotMutex};
    previous = std::exchange(s.current, next);
  }
  s.generation.store(next->generation, std::memory_order_release);
  if (s.observer != nullptr) {
    s.observer(next);
  }
}

void publish_locked(State& s) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<Contributor> contributors = collect_contributors(s);
  MergeOutput merged = merge(contributors);
  if (merged.warnings != s.mergeWarnings) {
    for (const MergeWarning& warning : merged.warnings) {
      Log.warn("Texture material merge dropped {}: {}", warning.pointer, warning.rule);
    }
    s.mergeWarnings = merged.warnings;
  }

  const Snapshot& previous = *s.current;
  if (merged.text == previous.mergedText && merged.mapBaseText == s.mapBaseText) {
    if (same_contributors(contributors, previous.contributors)) {
      return;
    }
    auto next = std::make_shared<Snapshot>(previous);
    next->mapSupplier = suppliers_by_id(s, merged.materials, next->records.size());
    next->contributors = std::move(contributors);
    swap_snapshot(s, std::move(next));
    return;
  }

  const size_t manifestCount = contributors.size();
  s.mapBaseText = std::move(merged.mapBaseText);
  Built built = build_full(s, merged, std::move(contributors), &previous);
  built.snapshot->generation = ++s.generationCounter;
  built.snapshot->bindEpoch = built.bindChanged ? ++s.bindEpochCounter : previous.bindEpoch;
  const Snapshot& next = *built.snapshot;
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  Log.info("Texture materials generation {}: {} material(s), {} grass type(s) from {} manifest(s) in {:.2f} ms",
           next.generation, next.materialCount, next.grassTypeCount, manifestCount, ms);
  swap_snapshot(s, std::move(built.snapshot));
}

State::State() {
  keyTable = std::make_shared<const KeyTable>(KeyTable{{texture::TextureSourceKey{}}, {std::string{}}});
  MergeOutput empty = merge({});
  mapBaseText = std::move(empty.mapBaseText);
  current = build_full(*this, empty, {}, nullptr).snapshot;
  latched = current;
  latchedRaw.store(current.get(), std::memory_order_release);
}

State& state() {
  static State s_state;
  return s_state;
}

Outcome evaluate(std::string_view text, MaterialLayer layer) {
  if (text.size() > MaxManifestBytes) {
    return {MaterialManifestStatus::Unreadable, {}, {}};
  }
  ManifestCheck check = check_manifest(text, layer);
  if (check.violation) {
    return {MaterialManifestStatus::Rejected, std::move(check.violation), {}};
  }
  return {MaterialManifestStatus::Accepted, {}, std::move(check.doc)};
}

Outcome evaluate_file(const std::filesystem::path& path, MaterialLayer layer) {
  std::error_code error;
  const uintmax_t size = std::filesystem::file_size(path, error);
  const auto missing_or_unreadable = [&] {
    std::error_code ignored;
    return std::filesystem::exists(path, ignored) ? MaterialManifestStatus::Unreadable
                                                  : MaterialManifestStatus::Missing;
  };
  if (error) {
    return {missing_or_unreadable(), {}, {}};
  }
  if (size > MaxManifestBytes) {
    return {MaterialManifestStatus::Unreadable, {}, {}};
  }
  const auto bytes = io::read_file(path);
  if (!bytes) {
    return {missing_or_unreadable(), {}, {}};
  }
  return evaluate({reinterpret_cast<const char*>(bytes->data()), bytes->size()}, layer);
}

uint32_t member_count(const std::shared_ptr<const json::Doc>& doc, std::string_view section) noexcept {
  return static_cast<uint32_t>(yyjson_obj_size(member(root_of(doc), section)));
}

bool apply_outcome(State& s, Registration& entry, Outcome&& outcome) {
  entry.status = outcome.status;
  entry.violation = std::move(outcome.violation);
  entry.doc = std::move(outcome.doc);
  entry.materialCount = member_count(entry.doc, "materials");
  entry.grassTypeCount = member_count(entry.doc, "grassTypes");
  return entry.violation &&
         s.warnedRejections.emplace(entry.label, entry.violation->rule, entry.violation->pointer).second;
}

void log_outcome(std::string_view label, MaterialManifestStatus status, const std::optional<Violation>& violation,
                 bool warnRejection) {
  switch (status) {
  case MaterialManifestStatus::Accepted:
    Log.info("Texture material manifest {} accepted", label);
    break;
  case MaterialManifestStatus::Missing:
    Log.info("Texture material manifest {} is missing", label);
    break;
  case MaterialManifestStatus::Unreadable:
    Log.warn("Texture material manifest {} could not be read", label);
    break;
  case MaterialManifestStatus::Rejected:
    if (warnRejection && violation) {
      Log.warn("Texture material manifest {} rejected: {} at '{}' ({})", label, violation->rule, violation->pointer,
               violation->detail);
    }
    break;
  default:
    break;
  }
}

texture::MaterialManifestRegistration add_registration(Registration&& entry, Outcome&& outcome) {
  State& s = state();
  std::string label;
  texture::MaterialManifestRegistration registration;
  bool warnRejection = false;
  std::optional<Violation> violation;
  {
    std::lock_guard lock{s.mutex};
    entry.id = ++s.nextId;
    entry.sequence = ++s.nextSequence;
    if (entry.label.empty()) {
      entry.label =
          entry.source == Source::File ? io::fs_path_to_string(entry.path) : fmt::format("manifest#{}", entry.id);
    }
    warnRejection = apply_outcome(s, entry, std::move(outcome));
    registration = {entry.id, entry.status};
    label = entry.label;
    violation = entry.violation;
    s.registrations.push_back(std::move(entry));
    publish_locked(s);
  }
  log_outcome(label, registration.status, violation, warnRejection);
  return registration;
}

Registration make_registration(const texture::MaterialManifestOptions& options, Source source) {
  Registration entry;
  entry.layer = options.layer;
  entry.priority = options.priority;
  entry.label = options.label;
  entry.base = options.base;
  entry.writable = options.writable;
  entry.source = source;
  return entry;
}

MaterialManifestStatus mutate(uint64_t id, std::optional<std::string_view> text) {
  if (text && text->size() > MaxManifestBytes) {
    return MaterialManifestStatus::Unreadable;
  }
  State& s = state();
  uint64_t ticket = 0;
  MaterialLayer layer = MaterialLayer::Pack;
  Source source = Source::Text;
  std::filesystem::path path;
  std::string stored;
  {
    std::lock_guard lock{s.mutex};
    Registration* entry = s.find(id);
    if (entry == nullptr) {
      return MaterialManifestStatus::Superseded;
    }
    ticket = ++entry->latestTicket;
    layer = entry->layer;
    source = entry->source;
    path = entry->path;
    if (!text && source == Source::Text) {
      stored = entry->text;
    }
  }
  Outcome outcome;
  if (text) {
    outcome = evaluate(*text, layer);
  } else if (source == Source::File) {
    outcome = evaluate_file(path, layer);
  } else {
    outcome = evaluate(stored, layer);
  }
  if (const auto hook = g_checkHook.load(std::memory_order_acquire); hook != nullptr) {
    hook(id);
  }
  std::string label;
  MaterialManifestStatus status = MaterialManifestStatus::Superseded;
  bool warnRejection = false;
  std::optional<Violation> violation;
  {
    std::lock_guard lock{s.mutex};
    Registration* entry = s.find(id);
    if (entry == nullptr || entry->latestTicket != ticket) {
      return MaterialManifestStatus::Superseded;
    }
    if (text) {
      entry->overrideText = std::string{*text};
    } else {
      entry->overrideText.reset();
    }
    warnRejection = apply_outcome(s, *entry, std::move(outcome));
    status = entry->status;
    label = entry->label;
    violation = entry->violation;
    publish_locked(s);
  }
  log_outcome(label, status, violation, warnRejection);
  return status;
}

texture::MaterialDiagnostic diagnostic_of(std::string_view label, const Violation& violation) {
  return {std::string{label}, violation.rule, violation.pointer, violation.detail};
}

texture::MaterialManifestInfo info_of(const State& s, const Registration& entry) {
  texture::MaterialManifestInfo info;
  info.id = entry.id;
  info.label = entry.label;
  info.path = entry.path;
  info.base = entry.base;
  info.layer = entry.layer;
  info.priority = entry.priority;
  info.sequence = entry.sequence;
  info.status = entry.status;
  info.writable = entry.writable;
  info.overridden = entry.overrideText.has_value();
  info.layerEnabled = s.layer_enabled(entry.layer);
  info.materialCount = entry.materialCount;
  info.grassTypeCount = entry.grassTypeCount;
  if (entry.violation) {
    info.violation = diagnostic_of(entry.label, *entry.violation);
  }
  return info;
}

void unregister_ids(std::span<const uint64_t> ids) {
  State& s = state();
  std::lock_guard lock{s.mutex};
  const auto removed = std::erase_if(s.registrations, [&](const Registration& entry) {
    return std::find(ids.begin(), ids.end(), entry.id) != ids.end();
  });
  if (removed != 0) {
    publish_locked(s);
  }
}

std::map<std::string, std::string, std::less<>> merge_origins(const Snapshot& snap, std::string_view keyText) {
  std::map<std::string, std::string, std::less<>> origins;
  std::map<std::string, bool, std::less<>> objectSlots;
  for (const Contributor& contributor : snap.contributors) {
    yyjson_val* entry = member(member(root_of(contributor.doc), "materials"), keyText);
    yyjson_val* key = nullptr;
    yyjson_obj_iter iter = yyjson_obj_iter_with(entry);
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
      yyjson_val* value = yyjson_obj_iter_get_val(key);
      const std::string field = pointer_join("", text_of(key));
      const bool nested = contains(NestedMaterialFields, text_of(key)) && yyjson_is_obj(value);
      const auto slot = objectSlots.find(field);
      if (!nested || slot == objectSlots.end() || !slot->second) {
        std::erase_if(origins, [&](const auto& origin) {
          return origin.first == field || origin.first.starts_with(field + '/');
        });
      }
      if (nested) {
        yyjson_val* child = nullptr;
        yyjson_obj_iter children = yyjson_obj_iter_with(value);
        while ((child = yyjson_obj_iter_next(&children)) != nullptr) {
          origins.insert_or_assign(pointer_join(field, text_of(child)), contributor.label);
        }
      } else {
        origins.insert_or_assign(field, contributor.label);
      }
      objectSlots.insert_or_assign(field, yyjson_is_obj(value));
    }
  }
  return origins;
}
} // namespace

uint16_t Snapshot::find(const texture::TextureSourceKey& key) const noexcept {
  if (index == nullptr || index->ids.empty()) {
    return 0;
  }
  texture::TextureSourceKey query = key;
  if (!query.hasTlut) {
    query.tlutHash = 0;
  }
  if (const auto it = index->ids.find(query); it != index->ids.end()) {
    return it->second;
  }
  if (query.hasTlut && query.tlutHash != texture::kWildcardTlutHash) {
    query.tlutHash = texture::kWildcardTlutHash;
    if (const auto it = index->ids.find(query); it != index->ids.end()) {
      return it->second;
    }
  }
  return 0;
}

const MaterialRecord& Snapshot::record(uint32_t id) const noexcept {
  static constexpr MaterialRecord Empty{};
  return id < records.size() ? records[id] : Empty;
}

bool Snapshot::live(uint32_t id) const noexcept { return id != 0 && id < records.size() && records[id].id != 0; }

const texture::TextureSourceKey* Snapshot::key(uint32_t id) const noexcept {
  return keyTable != nullptr && id != 0 && id < keyTable->keys.size() ? &keyTable->keys[id] : nullptr;
}

std::string_view Snapshot::key_name(uint32_t id) const noexcept {
  return keyTable != nullptr && id != 0 && id < keyTable->names.size() ? std::string_view{keyTable->names[id]}
                                                                       : std::string_view{};
}

std::string_view Snapshot::map_base(uint32_t id, uint32_t role) const noexcept {
  if (!live(id) || role >= MapRoleCount || id >= mapSupplier.size()) {
    return {};
  }
  const uint32_t supplier = mapSupplier[id][role];
  return supplier != 0 && supplier <= contributors.size() ? std::string_view{contributors[supplier - 1].base}
                                                          : std::string_view{};
}

std::shared_ptr<const Snapshot> snapshot() noexcept {
  State& s = state();
  std::lock_guard lock{s.snapshotMutex};
  return s.current;
}

uint64_t generation() noexcept { return state().generation.load(std::memory_order_acquire); }

void set_publish_observer(PublishObserver observer) noexcept {
  State& s = state();
  std::lock_guard lock{s.mutex};
  s.observer = observer;
  if (observer != nullptr) {
    observer(s.current);
  }
}

bool latch() noexcept {
  State& s = state();
  std::shared_ptr<const Snapshot> previous;
  bool changed = false;
  {
    std::lock_guard lock{s.snapshotMutex};
    changed = s.current->bindEpoch != s.latched->bindEpoch;
    previous = std::exchange(s.latched, s.current);
    s.latchedRaw.store(s.latched.get(), std::memory_order_release);
  }
  return changed;
}

std::shared_ptr<const Snapshot> latched_snapshot() noexcept {
  State& s = state();
  std::lock_guard lock{s.snapshotMutex};
  return s.latched;
}

const Snapshot* latched_raw() noexcept { return state().latchedRaw.load(std::memory_order_acquire); }

namespace testing {
void set_check_hook(void (*hook)(uint64_t)) noexcept { g_checkHook.store(hook, std::memory_order_release); }
} // namespace testing
} // namespace aurora::render::material

namespace aurora::texture {
using render::material::Source;

MaterialManifestRegistration load_material_manifest(const std::filesystem::path& path,
                                                    MaterialManifestOptions options) {
  if (!render::material::valid_layer(options.layer)) {
    render::material::Log.warn("Texture material manifest {} has an invalid layer", io::fs_path_to_string(path));
    return {0, MaterialManifestStatus::Rejected};
  }
  render::material::Registration entry = render::material::make_registration(options, Source::File);
  entry.path = path;
  if (entry.base.empty()) {
    entry.base = io::fs_path_to_string(path.parent_path());
  }
  return render::material::add_registration(std::move(entry), render::material::evaluate_file(path, options.layer));
}

MaterialManifestRegistration register_material_manifest(std::string_view text, MaterialManifestOptions options) {
  if (!render::material::valid_layer(options.layer)) {
    render::material::Log.warn("Texture material manifest {} has an invalid layer", options.label);
    return {0, MaterialManifestStatus::Rejected};
  }
  if (text.size() > render::material::MaxManifestBytes) {
    render::material::Log.warn("Texture material manifest {} is larger than 16 MiB", options.label);
    return {0, MaterialManifestStatus::Unreadable};
  }
  render::material::Registration entry = render::material::make_registration(options, Source::Text);
  entry.text = text;
  return render::material::add_registration(std::move(entry), render::material::evaluate(text, options.layer));
}

std::optional<MaterialManifestInfo> material_manifest_info(const MaterialManifestRegistration& registration) {
  render::material::State& s = render::material::state();
  std::lock_guard lock{s.mutex};
  const render::material::Registration* entry = s.find(registration.id);
  if (registration.id == 0 || entry == nullptr) {
    return std::nullopt;
  }
  return render::material::info_of(s, *entry);
}

MaterialManifestStatus reload_material_manifest(const MaterialManifestRegistration& registration) {
  return render::material::mutate(registration.id, std::nullopt);
}

MaterialManifestStatus update_material_manifest(const MaterialManifestRegistration& registration,
                                                std::string_view text) {
  return render::material::mutate(registration.id, text);
}

void unregister_material_manifest(const MaterialManifestRegistration& registration) {
  const uint64_t id = registration.id;
  render::material::unregister_ids({&id, 1});
}

void unregister_material_manifests(std::span<const MaterialManifestRegistration> registrations) {
  std::vector<uint64_t> ids;
  ids.reserve(registrations.size());
  for (const MaterialManifestRegistration& registration : registrations) {
    if (registration.id != 0) {
      ids.push_back(registration.id);
    }
  }
  render::material::unregister_ids(ids);
}

void clear_material_manifests() {
  render::material::State& s = render::material::state();
  std::lock_guard lock{s.mutex};
  s.registrations.clear();
  render::material::publish_locked(s);
}

void set_material_layer_enabled(MaterialLayer layer, bool enabled) {
  render::material::State& s = render::material::state();
  std::lock_guard lock{s.mutex};
  if (layer != MaterialLayer::Pack && layer != MaterialLayer::Mod) {
    if (!std::exchange(s.warnedLayerSwitch, true)) {
      render::material::Log.warn("Only the pack and mod texture material layers can be switched");
    }
    return;
  }
  bool& flag = layer == MaterialLayer::Pack ? s.packEnabled : s.modEnabled;
  if (std::exchange(flag, enabled) != enabled) {
    render::material::publish_locked(s);
  }
}

std::optional<MaterialDiagnostic> check_material_manifest(std::string_view text, MaterialLayer layer) {
  if (!render::material::valid_layer(layer)) {
    return MaterialDiagnostic{.detail = "invalid layer"};
  }
  if (text.size() > render::material::MaxManifestBytes) {
    return MaterialDiagnostic{.detail = "larger than 16 MiB"};
  }
  const auto check = render::material::check_manifest(text, layer);
  if (!check.violation) {
    return std::nullopt;
  }
  return render::material::diagnostic_of({}, *check.violation);
}

MaterialReport material_report() {
  render::material::State& s = render::material::state();
  std::lock_guard lock{s.mutex};
  MaterialReport report;
  std::vector<const render::material::Registration*> order;
  for (const render::material::Registration& entry : s.registrations) {
    order.push_back(&entry);
  }
  std::sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
    return std::tie(a->layer, a->priority, a->sequence) < std::tie(b->layer, b->priority, b->sequence);
  });
  for (const render::material::Registration* entry : order) {
    report.manifests.push_back(render::material::info_of(s, *entry));
    if (entry->status == MaterialManifestStatus::Rejected && entry->violation && s.layer_enabled(entry->layer)) {
      report.diagnostics.push_back(render::material::diagnostic_of(entry->label, *entry->violation));
    }
  }
  for (const render::material::MergeWarning& warning : s.mergeWarnings) {
    report.diagnostics.push_back({{}, warning.rule, warning.pointer, {}});
  }
  report.materialCount = s.current->materialCount;
  report.grassTypeCount = s.current->grassTypeCount;
  report.generation = s.current->generation;
  return report;
}

std::string material_manifest_json(bool pretty) {
  const auto snap = render::material::snapshot();
  if (!pretty) {
    return snap->mergedText;
  }
  return render::material::write_doc(snap->merged->get(), true);
}

std::string find_material_json(const TextureSourceKey& key, bool pretty) {
  const auto snap = render::material::snapshot();
  const uint16_t id = snap->find(key);
  if (id == 0) {
    return {};
  }
  const std::string_view name = snap->key_name(id);
  yyjson_val* material = snap->entries[id];
  const auto origins = render::material::merge_origins(*snap, name);

  render::json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  yyjson_mut_doc* out = doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(out);
  yyjson_mut_obj_add(root, yyjson_mut_str(out, "key"), yyjson_mut_strncpy(out, name.data(), name.size()));
  yyjson_mut_obj_add(root, yyjson_mut_str(out, "id"), yyjson_mut_uint(out, id));
  yyjson_mut_obj_add(root, yyjson_mut_str(out, "material"), yyjson_val_mut_copy(out, material));
  yyjson_mut_val* fields = yyjson_mut_obj(out);
  yyjson_val* field = nullptr;
  yyjson_obj_iter iter = yyjson_obj_iter_with(material);
  while ((field = yyjson_obj_iter_next(&iter)) != nullptr) {
    const std::string path = render::material::pointer_join("", render::material::text_of(field));
    const auto add = [&](std::string_view at) {
      if (const auto it = origins.find(at); it != origins.end()) {
        yyjson_mut_obj_add(fields, yyjson_mut_strncpy(out, at.data(), at.size()),
                           yyjson_mut_strncpy(out, it->second.data(), it->second.size()));
      }
    };
    yyjson_val* value = yyjson_obj_iter_get_val(field);
    if (render::material::contains(render::material::NestedMaterialFields, render::material::text_of(field)) &&
        yyjson_is_obj(value)) {
      yyjson_val* child = nullptr;
      yyjson_obj_iter children = yyjson_obj_iter_with(value);
      while ((child = yyjson_obj_iter_next(&children)) != nullptr) {
        add(render::material::pointer_join(path, render::material::text_of(child)));
      }
    } else {
      add(path);
    }
  }
  yyjson_mut_obj_add(root, yyjson_mut_str(out, "origins"), fields);
  yyjson_mut_doc_set_root(out, root);
  return render::json::write(out, pretty);
}

std::string format_material_key(const TextureSourceKey& key) { return render::material::format_key_text(key); }

std::optional<TextureSourceKey> parse_material_key(std::string_view text) {
  return render::material::parse_key_text(text);
}
} // namespace aurora::texture

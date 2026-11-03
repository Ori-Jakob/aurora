#include "classes.hpp"

#include "manifest.hpp"
#include "registry.hpp"

#include "../../gx/render_ext.hpp"
#include "../../gx/surface_pack.hpp"
#include "../../logging.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::material::classes {
namespace {
constexpr Module Log{"aurora::render::material"};

constexpr OpenPbrParams ground(float roughness) noexcept {
  OpenPbrParams params;
  params.specularRoughness = roughness;
  params.baseDiffuseRoughness = 0.5f;
  return params;
}

constexpr OpenPbrParams foliage() noexcept {
  OpenPbrParams params;
  params.specularRoughness = 0.55f;
  params.geometryThinWalled = true;
  params.thinWalledWrap = 0.5f;
  params.twoSided = true;
  params.noSsr = true;
  return params;
}

constexpr OpenPbrParams water_edge() noexcept {
  OpenPbrParams params;
  params.specularRoughness = 0.3f;
  return params;
}

constexpr std::array<BuiltIn, 8> kBuiltIn{{
    {1, "default", OpenPbrParams{}},
    {2, "ground/grass", ground(0.9f)},
    {3, "ground/dirt", ground(0.9f)},
    {4, "ground/rock", ground(0.8f)},
    {5, "ground/sand", ground(0.95f)},
    {6, "ground/snow", ground(0.6f)},
    {7, "foliage/grass", foliage()},
    {8, "water/edge", water_edge()},
}};

struct ClassEntry {
  bool present = false;
  std::string name;
  OpenPbrParams params;
};
using Table = std::array<ClassEntry, 256>;

struct IdEntry {
  uint8_t fields = 0;
  uint8_t classId = Default;
  uint8_t rough = 0;
  uint8_t metal = 0;
};

struct Latched {
  std::shared_ptr<const Table> table;
  std::array<uint32_t, 256> words{};
  std::bitset<256> twoSided;
};

std::shared_ptr<const Table> built_in_table() {
  auto table = std::make_shared<Table>();
  for (const BuiltIn& entry : kBuiltIn) {
    (*table)[entry.id] = {.present = true, .name = std::string{entry.name}, .params = entry.params};
  }
  return table;
}

const std::shared_ptr<const Table>& default_table() {
  static const std::shared_ptr<const Table> table = built_in_table();
  return table;
}

uint32_t flags_byte(const OpenPbrParams& params) noexcept {
  return (params.geometryThinWalled ? gx::surface::ThinTranslucent : 0u) | (params.noSsr ? gx::surface::NoSsr : 0u);
}

Latched latch_table(std::shared_ptr<const Table> table) {
  Latched latched{.table = std::move(table)};
  for (uint32_t id = 1; id < Reserved; ++id) {
    const ClassEntry& entry = (*latched.table)[id];
    if (!entry.present) {
      continue;
    }
    latched.words[id] = uint32_t{gx::surface::unorm8(entry.params.specularRoughness)} |
                        uint32_t{gx::surface::unorm8(entry.params.baseMetalness)} << 8 | id << 16 |
                        flags_byte(entry.params) << 24 | gx::render_ext::MaterialWordAmbient;
    latched.twoSided[id] = entry.params.twoSided;
  }
  return latched;
}

GpuRows build_rows(const Table& table) noexcept {
  GpuRows rows{};
  const GpuRow fallback = gpu_row(table[Default].params);
  for (uint32_t id = 0; id < rows.size(); ++id) {
    rows[id] = id != None && id != Reserved && table[id].present ? gpu_row(table[id].params) : fallback;
  }
  return rows;
}

std::mutex g_mutex;
std::shared_ptr<const Table> g_pending;
std::shared_ptr<const GpuRows> g_rows;
uint32_t g_registered = 0;
uint32_t g_rejected = 0;
std::atomic_uint64_t g_epoch{0};

// Game thread, while the FIFO is idle
Latched g_latched = latch_table(default_table());
std::vector<IdEntry> g_ids;
uint64_t g_generation = 0;
bool g_idsBuilt = false;
std::string g_forceName;
int g_forceId = -1;
std::set<std::string, std::less<>> g_warnedNames;

// FIFO thread
std::bitset<256> g_warnedHints;
uint32_t g_unregisteredHints = 0;
uint64_t g_words = 0;

uint8_t resolve_in(const Table& table, std::string_view name, bool& fallback) noexcept {
  fallback = false;
  std::string_view probe = name;
  while (!probe.empty()) {
    if (probe == FakeShadowName) {
      return None;
    }
    for (uint32_t id = 1; id < Reserved; ++id) {
      if (table[id].present && table[id].name == probe) {
        return static_cast<uint8_t>(id);
      }
    }
    const size_t slash = probe.rfind('/');
    probe = slash == std::string_view::npos ? std::string_view{} : probe.substr(0, slash);
  }
  fallback = true;
  return Default;
}

uint8_t resolve_logged(std::string_view name) {
  bool fallback = false;
  const uint8_t id = resolve_in(*g_latched.table, name, fallback);
  if (fallback && g_warnedNames.emplace(name).second) {
    Log.warn("Material class \"{}\" is not a known class; it shades as default", name);
  }
  return id;
}

float finite_or(float value, float fallback, float lo, float hi) noexcept {
  return std::isfinite(value) ? std::clamp(value, lo, hi) : fallback;
}

std::array<float, 3> colour_or(const std::array<float, 3>& value, const std::array<float, 3>& fallback) noexcept {
  return {finite_or(value[0], fallback[0], 0.f, 1.f), finite_or(value[1], fallback[1], 0.f, 1.f),
          finite_or(value[2], fallback[2], 0.f, 1.f)};
}

void rebuild_ids(const Snapshot& snapshot) {
  std::vector<uint8_t> registryClass(snapshot.classes.size(), Default);
  for (size_t i = 1; i < snapshot.classes.size(); ++i) {
    registryClass[i] = resolve_logged(snapshot.classes[i]);
  }
  g_ids.assign(snapshot.records.size(), IdEntry{});
  for (size_t id = 1; id < snapshot.records.size(); ++id) {
    const MaterialRecord& record = snapshot.records[id];
    if (record.id == 0) {
      continue;
    }
    IdEntry& entry = g_ids[id];
    if ((record.summary & Class) != 0) {
      entry.fields |= FieldClass;
      entry.classId =
          record.classId != 0 && record.classId < registryClass.size() ? registryClass[record.classId] : Default;
    }
    const MaterialScalars& scalars = snapshot.scalars_of(static_cast<uint32_t>(id));
    if ((scalars.set & ScalarRoughness) != 0) {
      entry.fields |= FieldRoughness;
      entry.rough = gx::surface::unorm8(scalars.roughness);
    }
    if ((scalars.set & ScalarMetalness) != 0) {
      entry.fields |= FieldMetalness;
      entry.metal = gx::surface::unorm8(scalars.metalness);
    }
  }
  g_forceId = g_forceName.empty() ? -1 : resolve_logged(g_forceName);
}

uint8_t unregistered_hint(uint8_t hint) noexcept {
  if (!g_warnedHints.test(hint)) {
    g_warnedHints.set(hint);
    ++g_unregisteredHints;
    Log.warn("MATERIAL hint class {} is not registered; it shades as default", hint);
  }
  return Default;
}
} // namespace

std::span<const BuiltIn> built_ins() noexcept { return kBuiltIn; }

GpuRow gpu_row(const OpenPbrParams& p) noexcept {
  const uint32_t flags =
      (p.geometryThinWalled ? RowThin : 0u) | (p.twoSided ? RowTwoSided : 0u) | (p.noSsr ? RowNoSsr : 0u);
  return {{
      p.specularColor[0],
      p.specularColor[1],
      p.specularColor[2],
      p.specularWeight,
      p.coatColor[0],
      p.coatColor[1],
      p.coatColor[2],
      p.coatWeight,
      p.fuzzColor[0],
      p.fuzzColor[1],
      p.fuzzColor[2],
      p.fuzzWeight,
      p.subsurfaceColor[0],
      p.subsurfaceColor[1],
      p.subsurfaceColor[2],
      p.subsurfaceWeight,
      p.emissionColor[0],
      p.emissionColor[1],
      p.emissionColor[2],
      p.emissionScale,
      p.specularIor,
      p.coatIor,
      p.coatRoughness,
      p.coatDarkening,
      p.baseDiffuseRoughness,
      p.fuzzRoughness,
      p.subsurfaceScatterAnisotropy,
      p.baseWeight,
      p.specularRoughness,
      p.baseMetalness,
      std::bit_cast<float>(flags),
      p.thinWalledWrap,
  }};
}

OpenPbrParams sanitised(const OpenPbrParams& p) noexcept {
  const OpenPbrParams d{};
  OpenPbrParams out = p;
  out.baseWeight = finite_or(p.baseWeight, d.baseWeight, 0.f, 1.f);
  out.baseMetalness = finite_or(p.baseMetalness, d.baseMetalness, 0.f, 1.f);
  out.baseDiffuseRoughness = finite_or(p.baseDiffuseRoughness, d.baseDiffuseRoughness, 0.f, 1.f);
  out.specularWeight = finite_or(p.specularWeight, d.specularWeight, 0.f, 1.f);
  out.specularColor = colour_or(p.specularColor, d.specularColor);
  out.specularRoughness = finite_or(p.specularRoughness, d.specularRoughness, 0.f, 1.f);
  out.specularIor = finite_or(p.specularIor, d.specularIor, 1.f, 3.f);
  out.coatWeight = finite_or(p.coatWeight, d.coatWeight, 0.f, 1.f);
  out.coatColor = colour_or(p.coatColor, d.coatColor);
  out.coatRoughness = finite_or(p.coatRoughness, d.coatRoughness, 0.f, 1.f);
  out.coatIor = finite_or(p.coatIor, d.coatIor, 1.f, 3.f);
  out.coatDarkening = finite_or(p.coatDarkening, d.coatDarkening, 0.f, 1.f);
  out.fuzzWeight = finite_or(p.fuzzWeight, d.fuzzWeight, 0.f, 1.f);
  out.fuzzColor = colour_or(p.fuzzColor, d.fuzzColor);
  out.fuzzRoughness = finite_or(p.fuzzRoughness, d.fuzzRoughness, 0.f, 1.f);
  out.subsurfaceWeight = finite_or(p.subsurfaceWeight, d.subsurfaceWeight, 0.f, 1.f);
  out.subsurfaceColor = colour_or(p.subsurfaceColor, d.subsurfaceColor);
  out.subsurfaceScatterAnisotropy = finite_or(p.subsurfaceScatterAnisotropy, d.subsurfaceScatterAnisotropy, -1.f, 1.f);
  out.emissionColor = colour_or(p.emissionColor, d.emissionColor);
  out.emissionScale = finite_or(p.emissionScale, d.emissionScale, 0.f, 65504.f);
  out.thinWalledWrap = finite_or(p.thinWalledWrap, d.thinWalledWrap, 0.f, 1.f);
  return out;
}

bool register_class(uint8_t id, std::string_view name, const OpenPbrParams& params) noexcept {
  std::lock_guard lock{g_mutex};
  const auto reject = [&](std::string_view reason) {
    ++g_rejected;
    Log.warn("register_material_class({}, \"{}\") refused: {}", id, name, reason);
    return false;
  };
  if (id == None || id == Default || id == Reserved) {
    return reject("ids 0, 1 and 255 are reserved");
  }
  if (name.size() > MaxNameLength || !match_class(name) || name == FakeShadowName) {
    return reject("not a usable class name");
  }
  if (id < FirstGame) {
    const auto builtIn = std::ranges::find(kBuiltIn, id, &BuiltIn::id);
    if (builtIn == kBuiltIn.end() || builtIn->name != name) {
      return reject("ids below 128 only retune the built-in of the same name");
    }
  }
  const Table& base = g_pending != nullptr ? *g_pending : *g_latched.table;
  for (uint32_t other = 1; other < Reserved; ++other) {
    if (other != id && base[other].present && base[other].name == name) {
      return reject("the name already belongs to another id");
    }
  }
  if (base[id].present && base[id].name != name) {
    return reject("the id already belongs to another name");
  }
  auto next = std::make_shared<Table>(base);
  (*next)[id] = {.present = true, .name = std::string{name}, .params = sanitised(params)};
  g_pending = std::move(next);
  ++g_registered;
  return true;
}

void set_force(std::string_view name) noexcept {
  g_forceName = std::string{name};
  g_idsBuilt = false;
}

void apply_frame_state(const Snapshot& snapshot) noexcept {
  std::shared_ptr<const Table> pending;
  {
    std::lock_guard lock{g_mutex};
    pending = std::exchange(g_pending, nullptr);
  }
  const bool tableChanged = pending != nullptr;
  if (tableChanged) {
    g_latched = latch_table(std::move(pending));
    auto rows = std::make_shared<const GpuRows>(build_rows(*g_latched.table));
    {
      std::lock_guard lock{g_mutex};
      g_rows = std::move(rows);
    }
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  if (tableChanged || !g_idsBuilt || snapshot.generation != g_generation) {
    rebuild_ids(snapshot);
    g_generation = snapshot.generation;
    g_idsBuilt = true;
  }
}

uint64_t epoch() noexcept { return g_epoch.load(std::memory_order_acquire); }

std::shared_ptr<const GpuRows> gpu_rows() noexcept {
  std::lock_guard lock{g_mutex};
  if (g_rows == nullptr) {
    g_rows = std::make_shared<const GpuRows>(build_rows(*g_latched.table));
  }
  return g_rows;
}

Stats stats() noexcept {
  std::lock_guard lock{g_mutex};
  return {.builtIn = static_cast<uint32_t>(kBuiltIn.size()),
          .registered = g_registered,
          .rejected = g_rejected,
          .unknownNames = static_cast<uint32_t>(g_warnedNames.size()),
          .unregisteredHints = g_unregisteredHints};
}

uint8_t resolve(std::string_view name) noexcept {
  bool fallback = false;
  return resolve_in(*g_latched.table, name, fallback);
}

bool present(uint8_t id) noexcept { return id != None && id != Reserved && (*g_latched.table)[id].present; }

const OpenPbrParams& params(uint8_t id) noexcept { return (*g_latched.table)[present(id) ? id : Default].params; }

uint8_t supplied(uint32_t materialId) noexcept { return materialId < g_ids.size() ? g_ids[materialId].fields : 0; }

uint8_t manifest_class(uint32_t materialId) noexcept {
  return materialId < g_ids.size() ? g_ids[materialId].classId : Default;
}

uint32_t material_word(const uint16_t idPerField[FieldCount], uint8_t hint, [[maybe_unused]] uint32_t categories,
                       GXCullMode cull, bool doubleSided, Source& source) noexcept {
  ++g_words;
  uint8_t cls = Default;
  source = Source::Default;
  if (g_forceId >= 0) {
    cls = static_cast<uint8_t>(g_forceId);
    source = Source::Force;
  } else if (idPerField[0] != 0) {
    cls = manifest_class(idPerField[0]);
    source = Source::Manifest;
  } else if (hint != 0) {
    cls = present(hint) ? hint : unregistered_hint(hint);
    source = Source::Hint;
  }
  if (cls == None) {
    return 0;
  }
  uint32_t word = g_latched.words[cls];
  if (idPerField[1] != 0) {
    word = (word & ~0xFFu) | g_ids[idPerField[1]].rough | gx::render_ext::MaterialWordRoughnessFactor;
  }
  if (idPerField[2] != 0) {
    word = (word & ~0xFF00u) | uint32_t{g_ids[idPerField[2]].metal} << 8 | gx::render_ext::MaterialWordMetalnessFactor;
  }
  const bool twoSided = cull != GX_CULL_FRONT && (cull == GX_CULL_NONE || doubleSided || g_latched.twoSided[cls]);
  return word | (twoSided ? gx::render_ext::MaterialWordTwoSided : 0u);
}

void shutdown() noexcept {
  const Stats s = stats();
  if (g_words != 0 || s.registered != 0 || s.rejected != 0) {
    Log.info("Material classes: built-in {}, registered {}, rejected {}, unknown names {}, unregistered hints {}",
             s.builtIn, s.registered, s.rejected, s.unknownNames, s.unregisteredHints);
  }
  const bool customised = g_latched.table != default_table();
  {
    std::lock_guard lock{g_mutex};
    g_pending.reset();
    g_rows.reset();
    g_registered = 0;
    g_rejected = 0;
  }
  g_latched = latch_table(default_table());
  if (customised) {
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  g_ids.clear();
  g_generation = 0;
  g_idsBuilt = false;
  g_forceName.clear();
  g_forceId = -1;
  g_warnedNames.clear();
  g_warnedHints.reset();
  g_unregisteredHints = 0;
  g_words = 0;
}

namespace testing {
void reset() noexcept { shutdown(); }
} // namespace testing
} // namespace aurora::render::material::classes

namespace aurora::render {
void register_material_class(uint8_t id, std::string_view name, const OpenPbrParams& params) {
  material::classes::register_class(id, name, params);
}
} // namespace aurora::render

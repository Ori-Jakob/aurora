#include "classifier.hpp"

#include <bit>

namespace aurora::render::scene {
namespace {
constexpr int32_t EmptySlot = -1;
constexpr int32_t FirstDroppedSlot = -2;
// A dropped key stays in the table so it counts once per frame; the bound keeps the table at most 75% full
constexpr uint32_t MaxDroppedKeys = ObjectSlots / 4;
static_assert((ObjectSlots & (ObjectSlots - 1)) == 0);
static_assert(MaxObjectEntries + MaxDroppedKeys < ObjectSlots);

constexpr uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// FNV-1a over the little-endian declaration bytes; tools recompute it, so the byte order is part of the contract
uint32_t scope_hash(uint32_t depth, uint8_t role, uint32_t categories, uint32_t flags) noexcept {
  uint32_t hash = 2166136261u;
  const auto mix = [&hash](uint8_t byte) { hash = (hash ^ byte) * 16777619u; };
  const auto mix32 = [&mix](uint32_t value) {
    for (uint32_t shift = 0; shift < 32; shift += 8) {
      mix(static_cast<uint8_t>(value >> shift));
    }
  };
  mix32(depth);
  mix(role);
  mix32(categories);
  mix32(flags);
  return hash == 0 ? 1 : hash;
}

constexpr Window window_after(int32_t marker) noexcept {
  return marker < 0 ? Window::PreScene : static_cast<Window>(marker + 1);
}

constexpr uint8_t role_bit(Role role) noexcept { return static_cast<uint8_t>(1u << static_cast<uint32_t>(role)); }

bool same_key(const ClassKey& lhs, const ClassKey& rhs, MergeKey mode) noexcept {
  if (mode == MergeKey::Full) {
    return lhs == rhs;
  }
  ClassKey other = rhs;
  other.object.subId = lhs.object.subId;
  other.object.generation = lhs.object.generation;
  return lhs == other;
}

bool same_object(const ObjectRef& lhs, const ObjectRef& rhs) noexcept {
  return lhs.id == rhs.id && lhs.subId == rhs.subId;
}
} // namespace

void Classifier::reserve() {
  clear_frame();
  m_records.reserve(MaxRecordedDraws);
  m_objectSlots.assign(ObjectSlots, EmptySlot);
  m_objectList.reserve(MaxObjectEntries);
  m_usedSlots.reserve(MaxObjectEntries + MaxDroppedKeys);
  m_droppedKeys.reserve(MaxDroppedKeys);
}

void Classifier::clear_frame() noexcept {
  m_root = {};
  m_scopes = {};
  m_object = {};
  m_objectOpen = false;
  m_scopeDepth = 0;
  m_scopeSwallow = 0;
  m_objectSwallow = 0;
  m_nextOrder = 1;
  m_lastMarker = -1;
  m_mainCameraSeen = false;
  m_key = {};
  m_keyDirty = true;
  m_last = {};
  m_lastPushedKey = {};
  m_hasPushed = false;
  m_ended = false;
  m_records.clear();
  for (const uint32_t slot : m_usedSlots) {
    m_objectSlots[slot] = EmptySlot;
  }
  m_usedSlots.clear();
  m_objectList.clear();
  m_droppedKeys.clear();
}

void Classifier::reset() noexcept {
  clear_frame();
  m_stats = {};
  m_stats.markerDraw.fill(-1);
  m_sequence = 0;
  m_record = false;
  m_objects = false;
  m_mergeKey = MergeKey::Full;
}

void Classifier::begin_frame() noexcept {
  clear_frame();
  m_stats = {};
  m_stats.frame = ++m_sequence;
  m_stats.markerDraw.fill(-1);
}

void Classifier::set_record(bool record) noexcept { m_record = record; }

void Classifier::set_objects(bool enabled) noexcept { m_objects = enabled; }

void Classifier::set_merge_key(MergeKey key) noexcept { m_mergeKey = key; }

void Classifier::end_frame() noexcept {
  if (m_ended) {
    return;
  }
  m_ended = true;
  m_stats.hints.openScopes = m_scopeDepth;
  m_stats.hints.openObjects = m_objectOpen ? 1 : 0;
}

void Classifier::camera(Role role) noexcept {
  ++m_stats.hints.cameras;
  if (role >= Role::None) {
    return;
  }
  if (role == Role::Main) {
    ++m_stats.hints.mainCameras;
    m_mainCameraSeen = true;
  }
  Entry& entry = innermost();
  entry.hasCamera = true;
  entry.camera = role;
  m_keyDirty = true;
}

void Classifier::object_begin(const ObjectHint& hint) noexcept {
  ++m_stats.hints.objectBegins;
  if (m_objectOpen) {
    // Objects never nest, so the open one ends here and its own END is swallowed later
    ++m_stats.hints.nestedObjects;
    ++m_objectSwallow;
  }
  m_object = Entry{
      .order = m_nextOrder++,
      .categories = hint.categories,
      .flags = hint.flags,
      .object = {.id = hint.id, .subId = hint.subId, .generation = hint.generation, .valid = true},
  };
  m_objectOpen = true;
  m_keyDirty = true;
}

void Classifier::object_end() noexcept {
  ++m_stats.hints.objectEnds;
  if (m_objectOpen) {
    m_object = {};
    m_objectOpen = false;
    m_keyDirty = true;
  } else if (m_objectSwallow > 0) {
    --m_objectSwallow;
  } else {
    ++m_stats.hints.objectUnderflow;
  }
}

void Classifier::scope_push(const ScopeHint& hint) noexcept {
  ++m_stats.hints.scopePushes;
  if (m_scopeDepth >= MaxScopeDepth) {
    ++m_stats.hints.scopeOverflow;
    ++m_scopeSwallow;
    return;
  }
  const uint8_t role = hint.role <= static_cast<uint8_t>(Role::Ui) ? hint.role : InheritRole;
  m_scopes[m_scopeDepth] = Entry{
      .order = m_nextOrder++,
      .categories = hint.categories,
      .flags = hint.flags,
      .scopeHash = scope_hash(m_scopeDepth + 1, role, hint.categories, hint.flags),
      .role = role,
  };
  ++m_scopeDepth;
  m_keyDirty = true;
}

void Classifier::scope_pop() noexcept {
  ++m_stats.hints.scopePops;
  if (m_scopeSwallow > 0) {
    --m_scopeSwallow;
  } else if (m_scopeDepth > 0) {
    m_scopes[--m_scopeDepth] = {};
    m_keyDirty = true;
  } else {
    ++m_stats.hints.scopeUnderflow;
  }
}

bool Classifier::stage(Stage stage) noexcept {
  const auto index = static_cast<int32_t>(stage);
  if (index > static_cast<int32_t>(Stage::FrameEnd) || index <= m_lastMarker) {
    ++m_stats.hints.staleMarkers;
    return false;
  }
  m_lastMarker = index;
  m_stats.markerDraw[static_cast<size_t>(index)] = static_cast<int32_t>(m_stats.draws);
  ++m_stats.hints.markers;
  m_keyDirty = true;
  return true;
}

void Classifier::material(uint32_t materialClass) noexcept {
  Entry& entry = innermost();
  entry.hasMaterial = true;
  entry.materialClass = static_cast<uint8_t>(materialClass & 0xFF);
  m_keyDirty = true;
}

void Classifier::tev_policy(uint32_t mask) noexcept {
  Entry& entry = innermost();
  entry.hasTevPolicy = true;
  entry.tevPolicy = mask;
  m_keyDirty = true;
}

Classifier::Entry& Classifier::innermost() noexcept {
  Entry* entry = m_scopeDepth > 0 ? &m_scopes[m_scopeDepth - 1] : &m_root;
  if (m_objectOpen && (m_scopeDepth == 0 || m_object.order > entry->order)) {
    entry = &m_object;
  }
  return *entry;
}

const ClassKey& Classifier::key() noexcept {
  if (!m_keyDirty) {
    return m_key;
  }
  // The object can open between any two scopes, so it joins the innermost-first walk by its open order
  std::array<const Entry*, MaxScopeDepth + 2> chain{};
  size_t count = 0;
  bool objectPlaced = !m_objectOpen;
  for (uint32_t i = m_scopeDepth; i-- > 0;) {
    if (!objectPlaced && m_object.order > m_scopes[i].order) {
      chain[count++] = &m_object;
      objectPlaced = true;
    }
    chain[count++] = &m_scopes[i];
  }
  if (!objectPlaced) {
    chain[count++] = &m_object;
  }
  chain[count++] = &m_root;

  ClassKey key;
  bool scopeRole = false;
  bool camera = false;
  bool material = false;
  bool tevPolicy = false;
  Role cameraRole = Role::None;
  for (size_t i = 0; i < count; ++i) {
    const Entry& entry = *chain[i];
    key.categories |= entry.categories;
    key.flags |= entry.flags;
    if (entry.role != InheritRole) {
      const auto role = static_cast<Role>(entry.role);
      key.roleScopes |= role_bit(role);
      if (!scopeRole) {
        key.role = role;
        scopeRole = true;
      }
    }
    if (entry.hasCamera && !camera) {
      cameraRole = entry.camera;
      camera = true;
    }
    if (entry.hasMaterial && !material) {
      key.materialClass = entry.materialClass;
      material = true;
    }
    if (entry.hasTevPolicy && !tevPolicy) {
      key.tevPolicy = entry.tevPolicy;
      tevPolicy = true;
    }
  }
  if (!scopeRole) {
    key.role = camera ? cameraRole : Role::None;
  }
  key.window = window_after(m_lastMarker);
  key.scope = m_scopeDepth > 0 ? m_scopes[m_scopeDepth - 1].scopeHash : 0;
  if (m_objectOpen) {
    key.object = m_object.object;
  }
  m_key = key;
  m_keyDirty = false;
  return m_key;
}

const DrawClass& Classifier::route(const DrawInputs& inputs) noexcept {
  DrawClass cls;
  cls.key = key();
  cls.key.mainCamera = inputs.mainCamera;
  cls.offscreen = inputs.offscreen;
  cls.colorWrite = inputs.colorWrite;
  cls.suppressed = (cls.key.flags & object_flag::CaptureOnly) != 0;
  cls.index = m_stats.draws;

  MainCheck rest = MainCheck::Passed;
  if (inputs.offscreen) {
    rest = MainCheck::Target;
  } else if (cls.key.window != Window::Opaque && cls.key.window != Window::Translucent) {
    rest = MainCheck::Window;
  } else if (!inputs.perspective) {
    rest = MainCheck::Ortho;
  } else if (!inputs.mainCamera) {
    rest = MainCheck::NoCamera;
  } else if (!inputs.projectionMatch) {
    rest = MainCheck::Projection;
  }
  const bool mainRole = cls.key.role == Role::Main;
  cls.check = mainRole ? rest : MainCheck::Role;
  cls.mainClass = mainRole && rest == MainCheck::Passed;
  cls.roleOnly = !mainRole && rest == MainCheck::Passed;
  if (cls.mainClass) {
    if ((cls.key.categories & category::Overlay) != 0 && cls.key.window == Window::Opaque) {
      cls.kind = Kind::Overlay;
    } else if (!inputs.colorWrite) {
      cls.kind = Kind::MainDepthOnly;
    } else if (inputs.depthWrite && !inputs.blended) {
      cls.kind = Kind::MainOpaque;
    } else {
      cls.kind = Kind::MainBlended;
    }
  } else {
    cls.kind = cls.key.role == Role::Sky && !inputs.offscreen ? Kind::Sky : Kind::Foreign;
  }

  cls.keyChanged = m_hasPushed && !same_key(cls.key, m_lastPushedKey, m_mergeKey);
  if (!cls.suppressed) {
    m_lastPushedKey = cls.key;
    m_hasPushed = true;
  }
  count(cls, inputs);
  if (m_record) {
    // Never grows past the reservation, so the FIFO thread does not allocate
    if (m_records.size() < MaxRecordedDraws && m_records.size() < m_records.capacity()) {
      m_records.push_back({.cls = cls, .vertices = inputs.vertices, .indexed = inputs.indexed});
    } else {
      ++m_stats.droppedRecords;
    }
  }
  m_last = cls;
  return m_last;
}

void Classifier::count(const DrawClass& cls, const DrawInputs& inputs) noexcept {
  ++m_stats.draws;
  if (cls.suppressed) {
    ++m_stats.suppressed;
    return;
  }
  const ClassKey& key = cls.key;
  ++m_stats.kinds[static_cast<size_t>(cls.kind)];
  ++m_stats.roles[static_cast<size_t>(key.role)];
  ++m_stats.checks[static_cast<size_t>(cls.check)];
  ++m_stats.windows[static_cast<size_t>(key.window)];
  for (uint32_t bits = key.categories; bits != 0; bits &= bits - 1) {
    ++m_stats.categories[static_cast<size_t>(std::countr_zero(bits))];
  }
  const bool capture = would_capture(cls);
  if (capture) {
    ++m_stats.captureCandidates;
  }
  if ((key.categories & category::FakeShadow) != 0) {
    ++m_stats.fakeShadow;
    if ((key.categories & category::Overlay) != 0) {
      ++m_stats.fakeShadowOverlay;
    }
  }
  if (!m_objectOpen && m_scopeDepth == 0) {
    ++m_stats.unclassified;
  }
  if (key.tevPolicy != 0) {
    ++m_stats.tevPolicyDraws;
  }
  // A main-projection scene draw before SCENE_BEGIN that no SKY scope claimed
  if (key.window == Window::PreScene && m_mainCameraSeen && inputs.mainCamera && !inputs.offscreen &&
      inputs.colorWrite && inputs.perspective && inputs.projectionMatch && key.role != Role::Sky) {
    ++m_stats.skyMisses;
  }
  for (size_t role = 0; role < RoleCount; ++role) {
    if ((key.roleScopes & (1u << role)) == 0) {
      continue;
    }
    RoleScopeCounts& counts = m_stats.underScope[role];
    ++counts.draws;
    counts.asRole += static_cast<size_t>(key.role) == role ? 1 : 0;
    counts.main += cls.mainClass ? 1 : 0;
    counts.roleOnly += cls.roleOnly ? 1 : 0;
    counts.captures += capture ? 1 : 0;
    counts.skyKind += cls.kind == Kind::Sky ? 1 : 0;
  }
  note_object(cls);
}

void Classifier::note_object(const DrawClass& cls) noexcept {
  const ObjectRef& ref = cls.key.object;
  if (!m_objects || !ref.valid || m_objectSlots.size() != ObjectSlots) {
    return;
  }
  constexpr uint32_t Mask = ObjectSlots - 1;
  auto slot = static_cast<uint32_t>(splitmix64(ref.id ^ (uint64_t{ref.subId} << 48))) & Mask;
  ObjectStats* entry = nullptr;
  for (;; slot = (slot + 1) & Mask) {
    const int32_t value = m_objectSlots[slot];
    if (value == EmptySlot) {
      break;
    }
    if (value >= 0) {
      if (ObjectStats& candidate = m_objectList[static_cast<size_t>(value)]; same_object(candidate.ref, ref)) {
        entry = &candidate;
        break;
      }
    } else if (same_object(m_droppedKeys[static_cast<size_t>(FirstDroppedSlot - value)], ref)) {
      return;
    }
  }
  if (entry == nullptr) {
    if (m_objectList.size() >= MaxObjectEntries) {
      ++m_stats.droppedObjects;
      // Past the dropped-key cap a key is not remembered, so each further draw of it counts again
      if (m_droppedKeys.size() < MaxDroppedKeys) {
        m_objectSlots[slot] = FirstDroppedSlot - static_cast<int32_t>(m_droppedKeys.size());
        m_usedSlots.push_back(slot);
        m_droppedKeys.push_back(ref);
      }
      return;
    }
    m_objectSlots[slot] = static_cast<int32_t>(m_objectList.size());
    m_usedSlots.push_back(slot);
    entry = &m_objectList.emplace_back();
  }
  entry->ref = ref;
  entry->categories |= cls.key.categories;
  entry->flags |= cls.key.flags;
  entry->roles |= role_bit(cls.key.role);
  ++entry->draws;
  entry->mainDraws += cls.mainClass ? 1 : 0;
}
} // namespace aurora::render::scene

#pragma once

#include <aurora/render.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::scene {
// Values equal AuroraCameraRole for the six hint roles; None means no camera and no role scope
enum class Role : uint8_t { Main, Sky, Reflection, Shadow, Offscreen, Ui, None };
inline constexpr size_t RoleCount = 7;
inline constexpr uint8_t InheritRole = 0xFF;

enum class Window : uint8_t { PreScene, Opaque, Translucent, Post, SceneEnd, Hud, Final };
inline constexpr size_t WindowCount = 7;

enum class Kind : uint8_t { MainOpaque, MainBlended, Overlay, Sky, Foreign, MainDepthOnly };
inline constexpr size_t KindCount = 6;

enum class MainCheck : uint8_t { Passed, Role, Target, Window, Ortho, NoCamera, Projection };
inline constexpr size_t MainCheckCount = 7;

enum class MergeKey : uint8_t { Full, ObjectOnly };

namespace category {
inline constexpr uint32_t Static = 1u << 0;
inline constexpr uint32_t Dynamic = 1u << 1;
inline constexpr uint32_t Skinned = 1u << 2;
inline constexpr uint32_t Terrain = 1u << 3;
inline constexpr uint32_t Foliage = 1u << 4;
inline constexpr uint32_t Water = 1u << 5;
inline constexpr uint32_t Sky = 1u << 6;
inline constexpr uint32_t Particle = 1u << 7;
inline constexpr uint32_t Decal = 1u << 8;
inline constexpr uint32_t Ui = 1u << 9;
inline constexpr uint32_t Player = 1u << 10;
inline constexpr uint32_t Emissive = 1u << 11;
inline constexpr uint32_t Overlay = 1u << 12;
inline constexpr uint32_t NoShadowCast = 1u << 16;
inline constexpr uint32_t NoShadowReceive = 1u << 17;
inline constexpr uint32_t NoPbr = 1u << 18;
inline constexpr uint32_t NoCapture = 1u << 19;
inline constexpr uint32_t NoRt = 1u << 20;
inline constexpr uint32_t ScreenRead = 1u << 21;
inline constexpr uint32_t FakeShadow = 1u << 22;
inline constexpr uint32_t CloudReceiver = 1u << 23;
inline constexpr uint32_t NoAo = 1u << 24;
} // namespace category

namespace object_flag {
inline constexpr uint32_t CaptureOnly = 1u << 0;
inline constexpr uint32_t Teleport = 1u << 1;
inline constexpr uint32_t CpuDeformed = 1u << 2;
inline constexpr uint32_t BakedVertexLight = 1u << 3;
inline constexpr uint32_t DoubleSided = 1u << 4;
inline constexpr uint32_t Billboard = 1u << 5;
inline constexpr uint32_t Immobile = 1u << 6;
} // namespace object_flag

inline constexpr uint32_t MaxScopeDepth = 16;
inline constexpr uint32_t MaxObjectDepth = 1;
inline constexpr uint32_t MaxRecordedDraws = 16384;
inline constexpr uint32_t MaxObjectEntries = 8192;
inline constexpr uint32_t ObjectSlots = 16384;

struct ObjectHint {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  uint32_t categories = 0;
  uint32_t flags = 0;
};

struct ScopeHint {
  uint32_t categories = 0;
  uint32_t flags = 0;
  uint8_t role = InheritRole;
};

struct ObjectRef {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  bool valid = false;
  bool operator==(const ObjectRef&) const = default;
};

struct ClassKey {
  Role role = Role::None;
  Window window = Window::PreScene;
  bool mainCamera = false;
  uint8_t roleScopes = 0;
  uint8_t materialClass = 0;
  uint32_t tevPolicy = 0;
  uint32_t categories = 0;
  uint32_t flags = 0;
  uint32_t scope = 0;
  ObjectRef object;
  bool operator==(const ClassKey&) const = default;
};

struct DrawInputs {
  bool mainCamera = false;
  bool projectionMatch = false;
  bool perspective = false;
  bool offscreen = false;
  bool depthWrite = false;
  bool blended = false;
  bool colorWrite = true;
  bool indexed = false;
  uint32_t vertices = 0;
};

struct DrawClass {
  ClassKey key;
  Kind kind = Kind::Foreign;
  MainCheck check = MainCheck::Role;
  bool mainClass = false;
  bool roleOnly = false;
  bool offscreen = false;
  bool colorWrite = true;
  bool suppressed = false;
  bool keyChanged = false;
  uint32_t index = 0;
};

struct ClassifiedDraw {
  DrawClass cls;
  uint32_t vertices = 0;
  bool indexed = false;
};

struct ObjectStats {
  ObjectRef ref;
  uint32_t categories = 0;
  uint32_t flags = 0;
  uint8_t roles = 0;
  uint32_t draws = 0;
  uint32_t mainDraws = 0;
};

struct RoleScopeCounts {
  uint32_t draws = 0;
  uint32_t asRole = 0;
  uint32_t main = 0;
  uint32_t roleOnly = 0;
  uint32_t captures = 0;
  uint32_t skyKind = 0;
};

struct HintCounts {
  uint32_t cameras = 0;
  uint32_t mainCameras = 0;
  uint32_t scopePushes = 0;
  uint32_t scopePops = 0;
  uint32_t objectBegins = 0;
  uint32_t objectEnds = 0;
  uint32_t markers = 0;
  uint32_t staleMarkers = 0;
  uint32_t scopeOverflow = 0;
  uint32_t scopeUnderflow = 0;
  uint32_t nestedObjects = 0;
  uint32_t objectUnderflow = 0;
  uint32_t openScopes = 0;
  uint32_t openObjects = 0;

  [[nodiscard]] uint32_t errors() const noexcept {
    return staleMarkers + scopeOverflow + scopeUnderflow + nestedObjects + objectUnderflow + openScopes + openObjects;
  }
};

// The view hook updates these on the FIFO thread
struct ViewCounts {
  uint32_t substituted = 0;
  uint32_t untouched = 0;
  uint32_t pending = 0;
  uint32_t overBudget = 0;
  uint32_t uniformBytes = 0;
};

struct FrameStats {
  uint64_t frame = 0;
  uint32_t draws = 0;
  uint32_t suppressed = 0;
  uint32_t mergeBreaks = 0;
  uint32_t captureCandidates = 0;
  uint32_t fakeShadow = 0;
  uint32_t droppedRecords = 0;
  std::array<uint32_t, KindCount> kinds{};
  std::array<uint32_t, RoleCount> roles{};
  std::array<uint32_t, MainCheckCount> checks{};
  std::array<uint32_t, WindowCount> windows{};
  std::array<RoleScopeCounts, RoleCount> underScope{};
  std::array<int32_t, StageCount> markerDraw{};
  HintCounts hints;
  std::array<uint32_t, 32> categories{};
  uint32_t unclassified = 0;
  uint32_t tevPolicyDraws = 0;
  uint32_t fakeShadowOverlay = 0;
  uint32_t skyMisses = 0;
  uint32_t droppedObjects = 0;
  ViewCounts view;
};

[[nodiscard]] constexpr bool would_capture(const DrawClass& cls) noexcept {
  return cls.kind == Kind::MainOpaque && (cls.key.categories & category::NoCapture) == 0;
}

[[nodiscard]] constexpr std::string_view role_name(Role role) noexcept {
  constexpr std::array<std::string_view, RoleCount> Names{"main",      "sky", "reflection", "shadow",
                                                          "offscreen", "ui",  "none"};
  const auto i = static_cast<size_t>(role);
  return i < Names.size() ? Names[i] : "unknown";
}
[[nodiscard]] constexpr std::string_view kind_name(Kind kind) noexcept {
  constexpr std::array<std::string_view, KindCount> Names{"mainOpaque", "mainBlended", "overlay",
                                                          "sky",        "foreign",     "mainDepthOnly"};
  const auto i = static_cast<size_t>(kind);
  return i < Names.size() ? Names[i] : "unknown";
}
[[nodiscard]] constexpr std::string_view check_name(MainCheck check) noexcept {
  constexpr std::array<std::string_view, MainCheckCount> Names{"passed", "role",     "target",    "window",
                                                               "ortho",  "noCamera", "projection"};
  const auto i = static_cast<size_t>(check);
  return i < Names.size() ? Names[i] : "unknown";
}
[[nodiscard]] constexpr std::string_view window_name(Window window) noexcept {
  constexpr std::array<std::string_view, WindowCount> Names{"preScene", "opaque", "translucent", "post",
                                                            "sceneEnd", "hud",    "final"};
  const auto i = static_cast<size_t>(window);
  return i < Names.size() ? Names[i] : "unknown";
}
// Bit index to the classify.json name; empty for undefined bits
[[nodiscard]] constexpr std::string_view category_name(uint32_t bit) noexcept {
  constexpr std::array<std::string_view, 25> Names{
      "static",  "dynamic",   "skinned", "terrain",    "foliage",      "water",
      "sky",     "particle",  "decal",   "ui",         "player",       "emissive",
      "overlay", "",          "",        "",           "noShadowCast", "noShadowReceive",
      "noPbr",   "noCapture", "noRt",    "screenRead", "fakeShadow",   "cloudReceiver",
      "noAo"};
  return bit < Names.size() ? Names[bit] : std::string_view{};
}

class Classifier {
public:
  void reserve();
  void reset() noexcept;
  void begin_frame() noexcept;
  void set_record(bool record) noexcept;
  void set_objects(bool enabled) noexcept;
  void set_merge_key(MergeKey key) noexcept;
  void end_frame() noexcept;

  void camera(Role role) noexcept;
  void object_begin(const ObjectHint& hint) noexcept;
  void object_end() noexcept;
  void scope_push(const ScopeHint& hint) noexcept;
  void scope_pop() noexcept;
  bool stage(Stage stage) noexcept;
  void material(uint32_t materialClass) noexcept;
  void tev_policy(uint32_t mask) noexcept;

  const DrawClass& route(const DrawInputs& inputs) noexcept;
  void count_merge_break() noexcept { ++m_stats.mergeBreaks; }

  [[nodiscard]] const DrawClass& last() const noexcept { return m_last; }
  [[nodiscard]] const FrameStats& stats() const noexcept { return m_stats; }
  [[nodiscard]] FrameStats& stats() noexcept { return m_stats; }
  [[nodiscard]] std::span<const ClassifiedDraw> draws() const noexcept { return m_records; }
  [[nodiscard]] std::span<const ObjectStats> objects() const noexcept { return m_objectList; }

private:
  struct Entry {
    uint32_t order = 0;
    uint32_t categories = 0;
    uint32_t flags = 0;
    uint32_t scopeHash = 0;
    ObjectRef object;
    uint8_t role = InheritRole;
    bool hasCamera = false;
    Role camera = Role::None;
    bool hasMaterial = false;
    uint8_t materialClass = 0;
    bool hasTevPolicy = false;
    uint32_t tevPolicy = 0;
  };

  Entry& innermost() noexcept;
  const ClassKey& key() noexcept;
  void clear_frame() noexcept;
  void count(const DrawClass& cls, const DrawInputs& inputs) noexcept;
  void note_object(const DrawClass& cls) noexcept;

  FrameStats m_stats;
  Entry m_root;
  std::array<Entry, MaxScopeDepth> m_scopes{};
  Entry m_object;
  bool m_objectOpen = false;
  uint32_t m_scopeDepth = 0;
  uint32_t m_scopeSwallow = 0;
  uint32_t m_objectSwallow = 0;
  uint32_t m_nextOrder = 1;
  int32_t m_lastMarker = -1;
  bool m_mainCameraSeen = false;
  ClassKey m_key;
  bool m_keyDirty = true;
  DrawClass m_last;
  ClassKey m_lastPushedKey;
  bool m_hasPushed = false;
  bool m_record = false;
  bool m_objects = false;
  MergeKey m_mergeKey = MergeKey::Full;
  bool m_ended = false;
  uint64_t m_sequence = 0;
  std::vector<ClassifiedDraw> m_records;
  std::vector<int32_t> m_objectSlots;
  std::vector<ObjectStats> m_objectList;
  // Slots written this frame, so begin_frame clears only those
  std::vector<uint32_t> m_usedSlots;
  std::vector<ObjectRef> m_droppedKeys;
};
} // namespace aurora::render::scene

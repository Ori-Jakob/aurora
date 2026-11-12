#pragma once

#include <aurora/render.hpp>
#include <aurora/render_graph.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::detail {
struct PassSetup;
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::fakeshadow::gate {
inline constexpr uint32_t HoldFrames = 30;
inline constexpr uint32_t TableSlots = 512;
inline constexpr uint32_t RingFrames = 8;
inline constexpr uint32_t CastBitSlots = 3;
inline constexpr uint32_t CastBitWords = 512;
inline constexpr std::string_view CastBitsPass = "fakeshadow.cast_bits";
inline constexpr uint16_t CastBitsOrder = 96;

inline constexpr uint32_t TagClassShift = 3;
inline constexpr uint32_t TagDecisionShift = 6;
inline constexpr size_t ShadowClassCount = static_cast<size_t>(ShadowClass::Bridge) + 1;

enum class ConsumerForce : uint8_t { Auto, Off, On, Flip30 };
inline constexpr size_t ConsumerForceCount = 4;
enum class CasterCover : uint8_t { Missing, Full, Partial, ExcludedOnly };

[[nodiscard]] constexpr ShadowClass tag_class(uint32_t subId) noexcept {
  const uint32_t cls = (subId >> TagClassShift) & 7u;
  return cls < ShadowClassCount ? static_cast<ShadowClass>(cls) : ShadowClass::Real;
}

[[nodiscard]] constexpr ShadowDecision tag_decision(uint32_t subId) noexcept {
  const uint32_t decision = (subId >> TagDecisionShift) & 15u;
  return decision < ShadowDecisionCount ? static_cast<ShadowDecision>(decision) : ShadowDecision::None;
}

[[nodiscard]] constexpr bool marker_class(ShadowClass cls) noexcept {
  return cls == ShadowClass::GroundMark || cls == ShadowClass::Telegraph;
}

// Steps 1 and 2 need no producer, so they also answer with render or hints off
[[nodiscard]] constexpr std::optional<ShadowDecision> setting_decision(GameShadows game, ShadowClass cls) noexcept {
  if (game == GameShadows::Remove) {
    return marker_class(cls) ? ShadowDecision::KeepClass : ShadowDecision::Skip;
  }
  if (game == GameShadows::Keep) {
    return ShadowDecision::KeepSetting;
  }
  return std::nullopt;
}

[[nodiscard]] constexpr std::string_view decision_name(ShadowDecision decision) noexcept {
  constexpr std::array<std::string_view, ShadowDecisionCount> Names{
      "none",         "skip",         "keepSetting", "keepNoConsumer",
      "keepCoverage", "keepSource",   "keepCaster",  "keepCasterUnknown",
      "keepClass",    "keepInterior", "keepHold",    "keepOverflow"};
  const auto i = static_cast<size_t>(decision);
  return i < Names.size() ? Names[i] : "unknown";
}

[[nodiscard]] constexpr std::string_view source_name(ShadowSource source) noexcept {
  constexpr std::array<std::string_view, 8> Names{"unknown",    "vertical",  "sunMoon", "roomKey",
                                                  "pointLight", "roomLight", "lantern", "effectLight"};
  const auto i = static_cast<size_t>(source);
  return i < Names.size() ? Names[i] : "unknown";
}

[[nodiscard]] constexpr std::string_view class_name(ShadowClass cls) noexcept {
  constexpr std::array<std::string_view, ShadowClassCount> Names{"real",      "blob",       "cue",
                                                                 "telegraph", "groundMark", "bridge"};
  const auto i = static_cast<size_t>(cls);
  return i < Names.size() ? Names[i] : "unknown";
}

struct HintInputs {
  bool sunDeclared = false;
  bool keyRoomDeclared = false;
  bool lanternDeclared = false;
  bool interior = false;
  float skyVisibility = 0.f;
  float shadowSun = 0.f;
  float thunder = 0.f;
  uint64_t hintFrame = 0;
};
[[nodiscard]] HintInputs hint_inputs(const detail::SceneHints& hints) noexcept;

// Locals stay uncovered until M14 gives them shadows of their own
[[nodiscard]] bool covered(const GateState& state, bool lightKeyRoom, ShadowSource source, ShadowClass cls) noexcept;

struct CasterRow {
  uint64_t id = 0;
  CasterCover cover = CasterCover::Missing;
  bool enteredNow = false;
};

struct EntryRow {
  uint64_t key = 0;
  ShadowSource source = ShadowSource::Unknown;
  ShadowClass cls = ShadowClass::Real;
  ShadowDecision decision = ShadowDecision::None;
  uint32_t hold = 0;
  std::vector<CasterRow> casters;
};

using AppliedFn = bool (*)() noexcept;
using SealedFn = bool (*)(uint64_t hintFrame) noexcept;

// Game thread while the FIFO is idle
void shutdown() noexcept;
void capture_inputs() noexcept;
void on_latch(const Settings& settings) noexcept;
void register_consumer(AppliedFn applied, SealedFn sealed) noexcept;
void set_force(ConsumerForce force) noexcept;
[[nodiscard]] ConsumerForce force() noexcept;

// Game thread only, between latches
[[nodiscard]] GateState state() noexcept;
[[nodiscard]] ShadowDecision decide(const GameShadowEntry& entry) noexcept;
[[nodiscard]] GameShadowStats stats(bool flush) noexcept;
[[nodiscard]] std::vector<EntryRow> entry_rows();
[[nodiscard]] uint64_t decision_frame() noexcept;

// Latched; the cast-bit copy runs only while a consumer can make skips
[[nodiscard]] bool cast_bits_wanted() noexcept;
// FIFO thread; frame_enabled is the sun term the frame really carried
void note_fifo_frame(uint64_t hintFrame, bool sunEnabled) noexcept;
// Any thread; the counter's readback for that frame, never a decision input
void note_dd_pixels(uint64_t hintFrame, uint64_t pbr, uint64_t legacy) noexcept;
// Render worker
void copy_cast_bits(const wgpu::CommandEncoder& encoder, const BufferView& source, uint64_t hintFrame) noexcept;

struct ReplayEntry {
  uint64_t object = 0;
  uint32_t subId = 0;
  uint32_t draws = 0;
  float alphaMin = 0.f;
  float alphaMax = 0.f;
};

struct ReplayFrame {
  uint64_t frame = 0;
  std::array<std::array<uint32_t, ShadowDecisionCount>, ShadowClassCount> objects{};
  uint32_t realDraws = 0;
  float realAlphaMedian = 0.f;
  float realAlphaMax = 0.f;
  float shadowSun = 0.f;
  float thunder = 0.f;
  uint32_t drops = 0;
  uint64_t ddPixelsPbr = 0;
  uint64_t ddPixelsLegacy = 0;
  // An untagged trace gives every fake subId 0, so only the object id tells a real shadow from the blobs
  std::vector<ReplayEntry> entries;
};

// FIFO thread; off unless a replay asks, so a frame pays one branch
void set_replay_observer(bool on) noexcept;
[[nodiscard]] bool replay_observer() noexcept;
[[nodiscard]] bool drop_active() noexcept;
void observe_object(uint32_t categories, uint32_t subId) noexcept;
void observe_draw(uint64_t object, uint32_t subId, float tevReg0Alpha) noexcept;
void count_drop() noexcept;
void seal_replay_frame(uint64_t hintFrame, const detail::SceneHints& hints) noexcept;
[[nodiscard]] std::optional<ReplayFrame> take_replay_frame() noexcept;

namespace testing {
struct Inputs {
  GateState state;
  bool available = true;
  bool lightKeyRoom = false;
  std::vector<std::pair<uint64_t, CasterCover>> casters;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> slots;
};
void reset() noexcept;
// Runs the latch bookkeeping with these inputs instead of the live producers
void latch(const Inputs& inputs) noexcept;
void deliver_cast_bits(uint64_t hintFrame, std::span<const uint32_t> words) noexcept;
[[nodiscard]] uint32_t table_slot(uint64_t key) noexcept;
} // namespace testing
} // namespace aurora::render::fakeshadow::gate

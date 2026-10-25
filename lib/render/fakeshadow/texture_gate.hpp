#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::gx {
struct DrawData;
struct GXBindGroups;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::fakeshadow {
enum class Gate : uint8_t { Light, Cloud, Ao };
inline constexpr size_t GateCount = 3;

enum class Force : uint8_t { Auto, Off, On };
enum class Neutral : uint8_t { None, White, Clear, Skip };
enum class Edit : uint8_t { Untouched, Substituted, Skipped };

using GateSource = bool (*)(const detail::FrameState& state) noexcept;

enum class Outcome : uint8_t {
  Bound,
  InScope,
  White,
  Clear,
  Skipped,
  GateOff,
  OutOfScope,
  OutOfRole,
  EntryOwned,
  RefusedIndirect,
  RefusedCoverage,
  RefusedWhite,
  RefusedClear,
  RefusedWrites,
  RefusedShared,
  Count,
};
inline constexpr size_t OutcomeCount = static_cast<size_t>(Outcome::Count);
using OutcomeCounts = std::array<uint64_t, OutcomeCount>;

struct Rule {
  uint16_t materialId = 0;
  Neutral neutral = Neutral::None;
  uint8_t gateBit = 0;
  uint16_t scopeMask = 0;
  bool scoped = false;
};

struct Slot {
  uint8_t index = 0;
  uint16_t rule = 0;
  bool indirect = false;
};

struct DrawFacts {
  bool writes = false;
  bool alphaConsumed = false;
  bool alphaTest = false;
  bool srcAlphaBlend = false;
  uint8_t texAlphaMask = 0;
  uint8_t boundMask = 0;
};

struct Decision {
  uint8_t white = 0;
  uint8_t clear = 0;
  uint8_t skip = 0;
};

struct DrawEdit {
  Edit edit = Edit::Untouched;
  bool applied = false;
  uint8_t flagged = 0;
  uint8_t white = 0;
  uint8_t clear = 0;
  uint8_t skip = 0;
};

struct GateView {
  bool live = false;
  bool effective = false;
  Force force = Force::Auto;

  bool operator==(const GateView&) const = default;
};

struct RuleInfo {
  Rule rule;
  std::string key;
};

struct RuleTable {
  uint64_t generation = 0;
  std::vector<RuleInfo> rules;
  uint32_t unscoped = 0;
};

struct TextureReport {
  std::array<GateView, GateCount> gates{};
  bool active = false;
  bool observing = false;
  bool hintsOff = false;
  std::shared_ptr<const RuleTable> table;
  std::vector<OutcomeCounts> counts;
};

// Game thread while the FIFO is idle
void install() noexcept;
void remove() noexcept;
void begin_frame(const detail::FrameState& state) noexcept;
void seal() noexcept;
void set_gate_source(Gate gate, GateSource source) noexcept;
void reset_frame() noexcept;

// Any thread
void set_force(Gate gate, Force force) noexcept;
[[nodiscard]] Force force(Gate gate) noexcept;
void set_stats_enabled(bool enabled) noexcept;
[[nodiscard]] std::shared_ptr<const TextureReport> latest_report() noexcept;

// FIFO thread
Edit push(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
          const scene::DrawClass& cls) noexcept;
[[nodiscard]] bool lead_flagged() noexcept;
// The current draw's slot masks, for hooks that run after the gate
[[nodiscard]] const DrawEdit& last_edit() noexcept;

[[nodiscard]] DrawFacts facts(const gx::PipelineConfig& config) noexcept;
Decision decide(std::span<const Slot> slots, const scene::DrawClass& cls, const DrawFacts& drawFacts, uint8_t gates,
                std::span<const Rule> rules, std::span<Outcome> results) noexcept;

namespace testing {
using NeutralBuilder = gx::GXBindGroups (*)(const gx::ShaderInfo& info, uint8_t whiteMask, uint8_t clearMask,
                                            bool baseLevel) noexcept;

struct GateList {
  uint8_t mask = 0;
  bool unknown = false;
};

void set_rules(std::span<const Rule> rules) noexcept;
void set_neutral_builder(NeutralBuilder builder) noexcept;
[[nodiscard]] GateList parse_gate_list(std::string_view text) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::fakeshadow

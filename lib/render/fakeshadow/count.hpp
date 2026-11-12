#pragma once

#include "../graph.hpp"
#include "../post/composite.hpp"

#include <aurora/render.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::fakeshadow::count {
inline constexpr std::string_view Pass = "fakeshadow.count";
inline constexpr std::string_view Buffer = "fakeshadow.counts";
inline constexpr std::string_view Shader = "fakeshadow/count";
inline constexpr uint16_t Order = 40;
inline constexpr uint32_t ReadbackSlots = 4;
inline constexpr uint32_t GroupSize = 8;

enum class Mode : uint8_t { Off, Rev3, All };

enum Word : uint32_t {
  WordDdPbr = 0,
  WordDdLegacy = 1,
  WordTouched = 2,
  WordShadowed = 3,
  WordCount = 4,
};
inline constexpr uint64_t WordBytes = uint64_t{WordCount} * 4;

inline constexpr uint32_t FlagSun = 1u << 0;

struct Params {
  uint32_t flags = 0;
  uint32_t pad[3]{};
  detail::composite::Sun sun;
};
static_assert(sizeof(Params) == 64 && offsetof(Params, sun) == 16 && sizeof(Params) <= MaxPassParamsSize);

struct Counts {
  uint64_t frames = 0;
  uint64_t ddPbr = 0;
  uint64_t ddLegacy = 0;
  uint64_t touched = 0;
  uint64_t shadowed = 0;
};

// AURORA_RENDER_GAME_SHADOWS=count or count=all; dev only, so image runs never pay for it
[[nodiscard]] Mode mode() noexcept;
[[nodiscard]] bool active() noexcept;
// rev3's double-darkening classes: entries kept although their light is covered and a caster is present
[[nodiscard]] bool selects(Mode mode, ShadowDecision decision) noexcept;
[[nodiscard]] detail::PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
// Waits for the readbacks in flight, so a replay reports each frame's counts on that frame's line
void flush() noexcept;
[[nodiscard]] Counts totals() noexcept;

namespace testing {
void set_mode(std::optional<Mode> mode) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::fakeshadow::count

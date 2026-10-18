#pragma once

#include <dolphin/gx/GXAurora.h>
#include <dolphin/gx/GXEnum.h>
#include <dolphin/types.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace render_demo {
inline constexpr u16 FirstHintId = GX_AURORA_HINT_FIRST;
inline constexpr u16 LastHintId = GX_AURORA_HINT_LAST;

struct HintTally {
  std::array<u32, LastHintId + 1 - FirstHintId> emitted{};
  u32 captureOnlyDraws = 0;

  void add(u16 id, u32 count = 1) noexcept;
  [[nodiscard]] u32 at(u16 id) const noexcept;
  [[nodiscard]] u64 total() const noexcept;
};

struct ObservedCounters {
  u64 decoded = 0;
  u64 skipped = 0;
  u64 malformed = 0;
  u64 late = 0;
  u64 bytes = 0;
};

using CounterLookup = ObservedCounters (*)(u16 id, void* userData);

struct DistanceFalloff {
  f32 radius = 0.f;
  f32 exponent = 0.f;
};

std::string_view hint_name(u16 id) noexcept;
std::optional<std::string> check_hints(const HintTally& tally, bool decoding, CounterLookup lookup, void* userData,
                                       u64 suppressedDraws);
DistanceFalloff distance_falloff(f32 refDistance, f32 refBrightness, GXDistAttnFn fn) noexcept;
} // namespace render_demo

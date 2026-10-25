#pragma once

#include "../../gx/render_ext.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aurora::gx {
struct DrawData;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::gx_ext::records {
struct DrawRecord {
  uint32_t frameOff = 0;
  uint32_t flags = 0;
  uint32_t objectLo = 0;
  uint32_t objectHi = 0;
  uint32_t prevMtxOff = 0;
  uint32_t materialOff0 = 0;
  uint32_t materialOff1 = 0;
  uint32_t tevPolicyOff = 0;
};
static_assert(sizeof(DrawRecord) == gx::render_ext::DrawRecordWords * sizeof(uint32_t));
static_assert(offsetof(DrawRecord, flags) == gx::render_ext::DrawRecordFlagsWord * sizeof(uint32_t));

namespace flag {
inline constexpr uint32_t MotionSourceMask = 3u << 0;
inline constexpr uint32_t Jitter = 1u << 2;
inline constexpr uint32_t VertexColorShift = 3;
inline constexpr uint32_t VertexColorMask = 3u << VertexColorShift;
inline constexpr uint32_t Unlit = 1u << 5;
inline constexpr uint32_t ReceiverShift = gx::render_ext::DrawRecordReceiverShift;
inline constexpr uint32_t ReceiverMask = 3u << ReceiverShift;
inline constexpr uint32_t VolFog = 1u << 8;
inline constexpr uint32_t CloudTap = 1u << 9;
inline constexpr uint32_t ScreenShadow = 1u << 10;
inline constexpr uint32_t ScreenAo = 1u << 11;
inline constexpr uint32_t PrevProjFrame = 1u << 12;
inline constexpr uint32_t CategoryShift = 16;
inline constexpr uint32_t CategoryMask = 0xFFu << CategoryShift;
} // namespace flag

struct FrameRecord {
  uint32_t frameSeq = 0;
  uint32_t flags = 0;
  uint32_t pad0 = 0;
  uint32_t pad1 = 0;
};
static_assert(sizeof(FrameRecord) == 16);

struct Stats {
  uint64_t frames = 0;
  uint64_t records = 0;
  uint64_t reused = 0;
  uint64_t dropped = 0;
  std::array<uint64_t, 4> codes{};
};

using PushFn = gfx::Range (*)(std::span<const uint8_t> bytes) noexcept;

// Game thread, while the FIFO is idle
void apply_frame_state(const detail::FrameState& state) noexcept;
[[nodiscard]] Stats stats() noexcept;
void shutdown() noexcept;

// FIFO thread
bool on_route(const scene::DrawClass& cls) noexcept;
void stamp(gx::DrawData& draw, const scene::DrawClass& cls) noexcept;

namespace testing {
void set_active(bool active, bool gbuffer, bool debugId) noexcept;
void set_push(PushFn push) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext::records

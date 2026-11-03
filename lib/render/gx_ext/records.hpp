#pragma once

#include "../../gx/render_ext.hpp"
#include "../../gx/surface_pack.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::motion {
struct MotionFrame;
} // namespace aurora::render::scene::motion

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
static_assert(offsetof(DrawRecord, tevPolicyOff) == gx::render_ext::DrawRecordTevPolicyWord * sizeof(uint32_t));

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
inline constexpr uint32_t Reactive = gx::render_ext::RecordFlagReactive;
inline constexpr uint32_t SurfaceSlotShift = gx::render_ext::RecordSurfaceSlotShift;
inline constexpr uint32_t SurfaceSlotMask = gx::render_ext::RecordSurfaceSlotMask;
} // namespace flag
static_assert(flag::ScreenAo == gx::render_ext::RecordFlagScreenAo);
static_assert(flag::VertexColorShift == gx::render_ext::RecordVertexColorShift);
static_assert(flag::Unlit == gx::render_ext::RecordFlagUnlit);

struct FrameRecord {
  uint32_t frameSeq = 0;
  uint32_t flags = 0;
  uint32_t pad0 = 0;
  uint32_t pad1 = 0;
};
static_assert(sizeof(FrameRecord) == 16);
static_assert(sizeof(FrameRecord) == gx::render_ext::FrameMotionWord * sizeof(uint32_t));

namespace frame_flag {
inline constexpr uint32_t HasCamera = 1u << 0;
} // namespace frame_flag

struct Stats {
  uint64_t frames = 0;
  uint64_t records = 0;
  uint64_t reused = 0;
  uint64_t dropped = 0;
  uint64_t screenAo = 0;
  uint64_t pbrRecords = 0;
  uint64_t policyBlocks = 0;
  uint64_t blendedInWindow = 0;
  uint64_t surfaceSlots = 0;
  std::array<uint64_t, 4> codes{};
};

// Bit 1 stays reserved for a later velocity window
enum DrawInput : uint32_t {
  InWindow = 1u << 0,
  // Main colour writers between OPAQUE_DONE and PRE_POST get the group-3 forward environment
  ForwardWindow = 1u << 2,
  PbrEligible = 1u << 3,
  // The vertex policy can follow this object flag, so a change must re-derive under the same GX state
  BakedVertex = 1u << 4,
};

using PushFn = gfx::Range (*)(std::span<const uint8_t> bytes) noexcept;

// Game thread, while the FIFO is idle
void apply_frame_state(const detail::FrameState& state) noexcept;
[[nodiscard]] Stats stats() noexcept;
[[nodiscard]] bool gbuffer() noexcept;
void shutdown() noexcept;

[[nodiscard]] bool in_forward_window(const scene::DrawClass& cls) noexcept;

// FIFO thread
bool on_route(const scene::DrawClass& cls) noexcept;
void stamp(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
           const scene::DrawClass& cls) noexcept;
bool set_camera_frame(const scene::motion::MotionFrame& frame) noexcept;

// Built from the sources v1's text used, so M9 content decodes to the same fields
[[nodiscard]] gx::surface::Entry surface_entry(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                                               uint32_t recordFlags) noexcept;

namespace testing {
void set_active(bool active, bool gbuffer, bool debugId) noexcept;
void set_surface(bool surface) noexcept;
void set_push(PushFn push) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext::records

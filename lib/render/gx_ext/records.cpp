#include "records.hpp"

#include "../hints.hpp"
#include "../scene/classifier.hpp"
#include "../scene/motion.hpp"
#include "../../gfx/recording.hpp"
#include "../../gfx/scene_targets.hpp"
#include "../../logging.hpp"

#include <cstring>
#include <utility>

namespace aurora::render::gx_ext::records {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};

struct Counters {
  uint64_t records = 0;
  uint64_t reused = 0;
  uint64_t dropped = 0;
  std::array<uint64_t, 4> codes{};
};

// Game thread, while the FIFO is idle
bool g_active = false;
bool g_gbuffer = false;
bool g_debugId = false;
Stats g_totals;
bool g_dropWarned = false;

// FIFO thread
Counters g_frame;
uint32_t g_frameLoc = 0;
DrawRecord g_last;
uint32_t g_lastLoc = 0;

gfx::Range default_push(std::span<const uint8_t> bytes) noexcept { return gfx::push_storage_packed(bytes); }

PushFn g_push = &default_push;

template <typename T>
std::span<const uint8_t> bytes_of(const T& value) noexcept {
  return {reinterpret_cast<const uint8_t*>(&value), sizeof(T)};
}

bool in_window(const scene::DrawClass& cls) noexcept {
  return cls.key.window == scene::Window::Opaque && !cls.offscreen;
}

uint32_t draw_inputs(const scene::DrawClass& cls) noexcept { return in_window(cls) ? InWindow : 0u; }

uint32_t receiver_code(const scene::DrawClass& cls) noexcept {
  constexpr uint32_t NonReceivers = scene::category::Overlay | scene::category::Particle | scene::category::NoAo |
                                    scene::category::Sky | scene::category::Ui | scene::category::FakeShadow;
  const uint32_t categories = cls.key.categories;
  if (cls.kind != scene::Kind::MainOpaque || !in_window(cls) || (categories & NonReceivers) != 0) {
    return 0;
  }
  if ((categories & scene::category::NoShadowReceive) != 0) {
    return 1;
  }
  return (categories & scene::category::CloudReceiver) != 0 ? 2 : 3;
}

// Sky and pre-scene translucents lie under the whole scene, so only in-window translucents ask TAA for the frame
bool raises_reactive(const gx::PipelineConfig& config, const scene::DrawClass& cls) noexcept {
  const bool window = cls.key.window == scene::Window::Opaque || cls.key.window == scene::Window::Translucent;
  return window && cls.key.role != scene::Role::Sky && gx::reactive_write(config) == gx::ReactiveWrite::Raise;
}

bool has_debug_id(const gfx::RenderTargetLayout& layout) noexcept {
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::DebugId) {
      return true;
    }
  }
  return false;
}

uint32_t locator(gfx::Range range) noexcept { return range.size == 0 ? 0 : range.offset / 4u; }

bool ensure_frame() noexcept {
  if (g_frameLoc != 0) {
    return true;
  }
  const FrameRecord header{.frameSeq = static_cast<uint32_t>(detail::live_scene_hints().hintFrame)};
  g_frameLoc = locator(g_push(bytes_of(header)));
  return g_frameLoc != 0;
}

void count_code(Counters& counters, const DrawRecord& record) noexcept {
  ++counters.codes[(record.flags & flag::ReceiverMask) >> flag::ReceiverShift];
}

void fold() noexcept {
  const Counters frame = std::exchange(g_frame, {});
  if (g_active) {
    ++g_totals.frames;
  }
  g_totals.records += frame.records;
  g_totals.reused += frame.reused;
  g_totals.dropped += frame.dropped;
  for (size_t i = 0; i < frame.codes.size(); ++i) {
    g_totals.codes[i] += frame.codes[i];
  }
  if (frame.dropped != 0 && !std::exchange(g_dropWarned, true)) {
    Log.warn("Render records: {} draws went without a record because the frame's storage was full", frame.dropped);
  }
}

void reset_frame() noexcept {
  g_frameLoc = 0;
  g_last = {};
  g_lastLoc = 0;
  gx::render_ext::g_drawInputs = 0;
}
} // namespace

void apply_frame_state(const detail::FrameState&) noexcept {
  // Published while the FIFO is idle, so the debug UI never reads live counters
  fold();
  const gfx::RenderTargetLayout layout = gfx::scene_render_target_layout();
  const bool hints = detail::hints_enabled();
  g_debugId = has_debug_id(layout);
  g_gbuffer = hints && gfx::layout_has_gbuffer(layout);
  g_active = hints && (g_gbuffer || g_debugId);
  reset_frame();
}

Stats stats() noexcept { return g_totals; }

bool gbuffer() noexcept { return g_gbuffer; }

void shutdown() noexcept {
  fold();
  if (g_totals.records != 0) {
    const auto& codes = g_totals.codes;
    Log.info("Render records: {} draws, {} reused, {} dropped over {} frames, codes {}/{}/{}/{}", g_totals.records,
             g_totals.reused, g_totals.dropped, g_totals.frames, codes[0], codes[1], codes[2], codes[3]);
  }
  g_active = false;
  g_gbuffer = false;
  g_debugId = false;
  g_totals = {};
  g_dropWarned = false;
  reset_frame();
}

bool on_route(const scene::DrawClass& cls) noexcept {
  gx::render_ext::g_drawInputs = g_gbuffer || gfx::gbuffer_live_or_pending() ? draw_inputs(cls) : 0;
  return g_active;
}

void stamp(gx::DrawData& draw, const gx::PipelineConfig& config, const scene::DrawClass& cls) noexcept {
  if (!g_active) {
    return;
  }
  const scene::motion::DrawMotion motion = scene::motion::draw_motion(draw, config, cls);
  if (draw.instanceCount == 0 || cls.offscreen || (cls.key.window >= scene::Window::Post && !g_debugId)) {
    return;
  }
  if (!ensure_frame()) {
    ++g_frame.dropped;
    return;
  }
  DrawRecord record{.frameOff = g_frameLoc,
                    .flags = motion.flags | (receiver_code(cls) << flag::ReceiverShift) |
                             (raises_reactive(config, cls) ? flag::Reactive : 0u),
                    .prevMtxOff = motion.prevMtxOff};
  if (cls.key.object.valid) {
    record.objectLo = static_cast<uint32_t>(cls.key.object.id);
    record.objectHi = static_cast<uint32_t>(cls.key.object.id >> 32);
  }
  if (!g_debugId && g_lastLoc != 0 && std::memcmp(&record, &g_last, sizeof(DrawRecord)) == 0) {
    draw.immediateData._pad = g_lastLoc;
    ++g_frame.reused;
    count_code(g_frame, record);
    return;
  }
  const uint32_t loc = locator(g_push(bytes_of(record)));
  if (loc == 0) {
    ++g_frame.dropped;
    return;
  }
  g_last = record;
  g_lastLoc = loc;
  draw.immediateData._pad = loc;
  ++g_frame.records;
  count_code(g_frame, record);
}

bool set_camera_frame(const scene::motion::MotionFrame& frame) noexcept {
  if (!g_active) {
    return false;
  }
  struct CameraFrame {
    FrameRecord header;
    scene::motion::MotionFrame tail;
  };
  static_assert(sizeof(CameraFrame) == 128 && offsetof(CameraFrame, tail) == sizeof(FrameRecord));
  const CameraFrame block{
      .header = {.frameSeq = static_cast<uint32_t>(detail::live_scene_hints().hintFrame),
                 .flags = frame_flag::HasCamera},
      .tail = frame,
  };
  const uint32_t loc = locator(g_push(bytes_of(block)));
  if (loc == 0) {
    return false;
  }
  g_frameLoc = loc;
  return true;
}

namespace testing {
void set_active(bool active, bool gbuffer, bool debugId) noexcept {
  g_active = active;
  g_gbuffer = gbuffer;
  g_debugId = debugId;
  reset_frame();
}

void set_push(PushFn push) noexcept { g_push = push != nullptr ? push : &default_push; }

void reset() noexcept {
  shutdown();
  g_push = &default_push;
}
} // namespace testing
} // namespace aurora::render::gx_ext::records

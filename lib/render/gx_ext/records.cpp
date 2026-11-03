#include "records.hpp"
#include "fwd_env.hpp"
#include "surface_table.hpp"
#include "unlit.hpp"

#include "../hints.hpp"
#include "../scene/classifier.hpp"
#include "../scene/motion.hpp"
#include "../../gfx/recording.hpp"
#include "../../gfx/scene_targets.hpp"
#include "../../gx/gx.hpp"
#include "../../logging.hpp"

#include <dolphin/gx/GXAurora.h>

#include <cstring>
#include <utility>

namespace aurora::render::gx_ext::records {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};

struct Counters {
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

// Game thread, while the FIFO is idle
bool g_active = false;
bool g_gbuffer = false;
bool g_debugId = false;
bool g_surface = false;
Stats g_totals;
bool g_dropWarned = false;

// FIFO thread
Counters g_frame;
uint32_t g_frameLoc = 0;
DrawRecord g_last;
uint32_t g_lastLoc = 0;
AuroraTevPolicy g_lastPolicy{};
uint32_t g_lastPolicyLoc = 0;

gfx::Range default_push(std::span<const uint8_t> bytes) noexcept { return gfx::push_storage_packed(bytes); }

PushFn g_push = &default_push;

template <typename T>
std::span<const uint8_t> bytes_of(const T& value) noexcept {
  return {reinterpret_cast<const uint8_t*>(&value), sizeof(T)};
}

uint32_t locator(gfx::Range range) noexcept { return range.size == 0 ? 0 : range.offset / 4u; }

bool in_window(const scene::DrawClass& cls) noexcept {
  return cls.key.window == scene::Window::Opaque && !cls.offscreen;
}

bool pbr_eligible(const scene::DrawClass& cls) noexcept {
  constexpr uint32_t NotPbr = scene::category::NoPbr | scene::category::Water | scene::category::ScreenRead |
                              scene::category::Particle | scene::category::Ui | scene::category::Sky |
                              scene::category::FakeShadow | scene::category::Overlay;
  return g_surface && cls.kind == scene::Kind::MainOpaque && in_window(cls) && (cls.key.categories & NotPbr) == 0;
}

uint32_t draw_inputs(const scene::DrawClass& cls) noexcept {
  return (in_window(cls) ? InWindow : 0u) | (in_forward_window(cls) ? ForwardWindow : 0u) |
         (pbr_eligible(cls) ? PbrEligible : 0u) |
         ((cls.key.flags & scene::object_flag::BakedVertexLight) != 0 ? BakedVertex : 0u);
}

uint32_t policy_block(const scene::DrawClass& cls) noexcept {
  if (cls.key.tevPolicy == 0 || !unlit::policy_enabled()) {
    return 0;
  }
  AuroraTevPolicy policy{.mask = cls.key.tevPolicy};
  static_assert(sizeof(policy.neutral) == sizeof(cls.key.tevNeutral));
  std::memcpy(policy.neutral, cls.key.tevNeutral.data(), sizeof(policy.neutral));
  if (g_lastPolicyLoc != 0 && std::memcmp(&policy, &g_lastPolicy, sizeof(policy)) == 0) {
    return g_lastPolicyLoc;
  }
  const uint32_t loc = locator(g_push(bytes_of(policy)));
  if (loc != 0) {
    g_lastPolicy = policy;
    g_lastPolicyLoc = loc;
    ++g_frame.policyBlocks;
  }
  return loc;
}

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
  // Terrain layers and decals blend over the depth they lie on, so the motion below them is their own
  constexpr uint32_t Layers = scene::category::Terrain | scene::category::Decal;
  const bool window = cls.key.window == scene::Window::Opaque || cls.key.window == scene::Window::Translucent;
  return window && cls.key.role != scene::Role::Sky && (cls.key.categories & Layers) == 0 &&
         gx::reactive_write(config) == gx::ReactiveWrite::Raise;
}

bool has_debug_id(const gfx::RenderTargetLayout& layout) noexcept {
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::DebugId) {
      return true;
    }
  }
  return false;
}


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
  counters.screenAo += (record.flags & flag::ScreenAo) != 0 ? 1 : 0;
}

void fold() noexcept {
  const Counters frame = std::exchange(g_frame, {});
  if (g_active) {
    ++g_totals.frames;
  }
  g_totals.records += frame.records;
  g_totals.reused += frame.reused;
  g_totals.dropped += frame.dropped;
  g_totals.screenAo += frame.screenAo;
  g_totals.pbrRecords += frame.pbrRecords;
  g_totals.policyBlocks += frame.policyBlocks;
  g_totals.blendedInWindow += frame.blendedInWindow;
  g_totals.surfaceSlots += frame.surfaceSlots;
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
  g_lastPolicy = {};
  g_lastPolicyLoc = 0;
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
  const gfx::LayoutWarm* warm = gfx::layout_warm();
  g_surface = hints && (gfx::layout_has_surface(layout) || (warm != nullptr && gfx::layout_has_surface(warm->pending)));
  surface_table::begin_frame(g_active && g_surface);
  reset_frame();
}

Stats stats() noexcept { return g_totals; }

bool in_forward_window(const scene::DrawClass& cls) noexcept {
  return cls.key.window == scene::Window::Translucent && cls.mainClass && cls.colorWrite && !cls.offscreen;
}

bool gbuffer() noexcept { return g_gbuffer; }

void shutdown() noexcept {
  fold();
  if (g_totals.records != 0) {
    const auto& codes = g_totals.codes;
    Log.info(
        "Render records: {} draws, {} reused, {} dropped over {} frames, codes {}/{}/{}/{}, screen AO {}, "
        "surface slots {}",
        g_totals.records, g_totals.reused, g_totals.dropped, g_totals.frames, codes[0], codes[1], codes[2], codes[3],
        g_totals.screenAo, g_totals.surfaceSlots);
  }
  surface_table::shutdown();
  g_active = false;
  g_gbuffer = false;
  g_debugId = false;
  g_surface = false;
  g_totals = {};
  g_dropWarned = false;
  reset_frame();
}

bool on_route(const scene::DrawClass& cls) noexcept {
  gx::render_ext::g_drawInputs = g_gbuffer || gfx::gbuffer_live_or_pending() ? draw_inputs(cls) : 0;
  return g_active;
}

void stamp(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
           const scene::DrawClass& cls) noexcept {
  if (!g_active) {
    return;
  }
  if (g_surface && cls.kind == scene::Kind::MainBlended && in_window(cls)) {
    ++g_frame.blendedInWindow;
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
                             (raises_reactive(config, cls) ? flag::Reactive : 0u) | fwd_env::record_flags(cls, config),
                    .prevMtxOff = motion.prevMtxOff};
  if (cls.key.object.valid) {
    record.objectLo = static_cast<uint32_t>(cls.key.object.id);
    record.objectHi = static_cast<uint32_t>(cls.key.object.id >> 32);
  }
  if (config.shaderConfig.ext.output == 2) {
    record.flags |= unlit::vertex_policy(config, info, cls) << flag::VertexColorShift |
                    (unlit::unlit_flag(config, info, cls) ? flag::Unlit : 0u);
    record.tevPolicyOff = policy_block(cls);
    // Only admitted text writes the record; any other output-2 pipeline writes slot 0
    if (info.unlitAdmitted) {
      const uint32_t slot = surface_table::slot_for(surface_entry(config, info, record.flags));
      record.flags |= slot << flag::SurfaceSlotShift;
      g_frame.surfaceSlots += slot != 0 ? 1 : 0;
    }
    ++g_frame.pbrRecords;
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

gx::surface::Entry surface_entry(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                                 uint32_t recordFlags) noexcept {
  namespace surface = gx::surface;
  const uint32_t code = (recordFlags & flag::ReceiverMask) >> flag::ReceiverShift;
  const bool passthrough = static_cast<gx::unlit::Verdict>(info.unlitVerdict) == gx::unlit::Verdict::Passthrough;
  uint32_t flags = (code == 2 ? surface::CloudReceiver : 0u) | (code == 1 ? surface::NoShadowReceive : 0u);
  flags |= passthrough || (recordFlags & flag::Unlit) != 0 ? surface::Unlit : 0u;
  flags |= ((recordFlags & flag::VertexColorMask) >> flag::VertexColorShift) == 1 ? surface::BakedLight : 0u;
  surface::Fields fields{.classId = surface::ClassDefault, .flags = static_cast<uint8_t>(flags)};
  const uint32_t extBits =
      config.shaderConfig.ext.surfaceExt != 0 ? surface::EntryExtAmbient | surface::EntryExtEmissive : 0u;
  return surface::pack_entry(fields, extBits);
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

void set_surface(bool surface) noexcept { g_surface = surface; }

void set_push(PushFn push) noexcept { g_push = push != nullptr ? push : &default_push; }

void reset() noexcept {
  shutdown();
  g_push = &default_push;
}
} // namespace testing
} // namespace aurora::render::gx_ext::records

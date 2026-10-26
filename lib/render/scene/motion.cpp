#include "motion.hpp"

#include "classifier.hpp"
#include "../gx_ext/records.hpp"
#include "../post/temporal_resolve.hpp"
#include "../../gfx/frame.hpp"
#include "../../gfx/recording.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::scene::motion {
namespace {
constexpr Module Log{"aurora::render::motion"};
namespace rx = gx::render_ext;
namespace temporal_resolve = render::detail::temporal_resolve;

constexpr uint32_t TableSlots = 16384;
constexpr uint32_t TableMask = TableSlots - 1;
constexpr uint32_t MaxEntries = TableSlots / 4 * 3;
constexpr uint32_t ArenaMatrices = 16384;
constexpr uint32_t OrdinalFallbacks = 3;
constexpr uint32_t PositionSlots = gx::MaxPnMtx + gx::MaxTexMtx;
constexpr uint32_t ImmobileSamplesPerFrame = 64;
constexpr float ImmobileMaxPx = 0.05f;
constexpr uint32_t StatsLogFrames = 60;
constexpr uint32_t DefaultCheckFirst = 30;
constexpr uint32_t DefaultCheckLast = 300;
constexpr uint32_t MatrixWords = sizeof(Mat3x4<float>) / sizeof(uint32_t);
static_assert((TableSlots & TableMask) == 0);
static_assert(rx::MotionPaletteSlots == gx::MaxPnMtx);

struct Key {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  uint16_t ordinal = 0;
  uint8_t role = 0;
  bool operator==(const Key&) const = default;
};

struct Slot {
  uint32_t tag = 0;
  Key key;
  uint32_t first = 0;
  uint8_t count = 0;
  bool ambiguous = false;
};

struct Table {
  std::vector<Slot> slots;
  std::vector<Mat3x4<float>> arena;
  uint32_t tag = 0;
  uint32_t entries = 0;
  uint32_t used = 0;
};

struct Hit {
  const Slot* slot = nullptr;
  bool exact = false;
};

struct CameraState {
  bool valid = false;
  gx::hints::Camera camera;
};

struct Frame {
  uint32_t index = 0;
  bool usable = false;
  bool cameraLive = false;
  bool valid = false;
  bool jitter = false;
  gx::hints::Camera camera;
  Mat3x4<float> reproject{};
  uint32_t prevBase = 0;
  bool prevPushTried = false;
  uint32_t immobileSamples = 0;
  uint64_t refusedBytes = 0;
  Counts counts;
};

gfx::Range default_push(std::span<const uint8_t> bytes) noexcept { return gfx::push_storage_packed(bytes); }

// Game thread, while the FIFO is idle
bool g_envRead = false;
testing::JitterEnv g_jitterEnv = testing::JitterEnv::Unset;
uint32_t g_checkFirst = 0;
uint32_t g_checkLast = 0;
bool g_active = false;
bool g_jitterActive = false;
std::array<float, 2> g_jitter{};
uint32_t g_frames = 0;
uint32_t g_nextTag = 0;
std::array<Table, 2> g_tables;
uint32_t g_current = 0;
CameraState g_previous;
Stats g_stats;
PushFn g_push = &default_push;

// FIFO thread during the frame
Frame g_frame;

constexpr uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint32_t slot_of(const Key& key) noexcept {
  const uint64_t mixed = splitmix64(key.id ^ (uint64_t{key.subId} << 32) ^ (uint64_t{key.generation} << 16) ^
                                    (uint64_t{key.ordinal} << 48) ^ (uint64_t{key.role} << 56));
  return static_cast<uint32_t>(mixed) & TableMask;
}

bool check_mode() noexcept { return g_checkLast != 0; }

void read_env() noexcept {
  if (std::exchange(g_envRead, true)) {
    return;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_JITTER")) {
    const std::string_view text{value};
    if (text == "on" || text == "off") {
      g_jitterEnv = text == "on" ? testing::JitterEnv::On : testing::JitterEnv::Off;
      Log.warn("Motion: jitter forced {} by AURORA_RENDER_JITTER", text);
    } else {
      Log.warn("Motion: AURORA_RENDER_JITTER={} is not on or off; ignored", text);
    }
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_MOTION_CHECK")) {
    const std::string_view text{value};
    uint32_t first = DefaultCheckFirst;
    uint32_t last = DefaultCheckLast;
    bool parsed = text == "1";
    if (const size_t dash = text.find('-'); !parsed && dash != std::string_view::npos) {
      const auto a = std::from_chars(text.data(), text.data() + dash, first);
      const auto b = std::from_chars(text.data() + dash + 1, text.data() + text.size(), last);
      parsed = a.ec == std::errc{} && a.ptr == text.data() + dash && b.ec == std::errc{} &&
               b.ptr == text.data() + text.size() && first <= last && last != 0;
    }
    if (parsed) {
      g_checkFirst = first;
      g_checkLast = last;
      Log.warn("Motion: check mode on frames {}-{} by AURORA_RENDER_MOTION_CHECK", first, last);
    } else {
      Log.warn("Motion: AURORA_RENDER_MOTION_CHECK={} is not 1 or <first>-<last>; ignored", text);
    }
  }
}

void allocate() {
  for (Table& table : g_tables) {
    if (table.slots.size() != TableSlots) {
      table.slots.assign(TableSlots, Slot{});
      table.arena.assign(ArenaMatrices, Mat3x4<float>{});
    }
  }
}

void clear_history() noexcept {
  for (Table& table : g_tables) {
    table.tag = 0;
    table.entries = 0;
    table.used = 0;
  }
  g_previous = {};
}

uint32_t next_tag() noexcept {
  if (++g_nextTag == 0) {
    ++g_nextTag;
  }
  return g_nextTag;
}

void add(Counts& total, const Counts& frame) noexcept {
  total.none += frame.none;
  total.camera += frame.camera;
  total.object += frame.object;
  total.palette += frame.palette;
  total.jittered += frame.jittered;
  total.exact += frame.exact;
  total.fallback += frame.fallback;
  total.miss += frame.miss;
  total.ambiguous += frame.ambiguous;
  total.overflow += frame.overflow;
  total.pushRefused += frame.pushRefused;
  total.badIndex += frame.badIndex;
  for (size_t i = 0; i < PlayerKindCount; ++i) {
    total.player[i].eligible += frame.player[i].eligible;
    total.player[i].exact += frame.player[i].exact;
    total.player[i].fallback += frame.player[i].fallback;
    total.player[i].miss += frame.player[i].miss;
  }
  total.framesInWindow += frame.framesInWindow;
  total.framesWithoutPlayer += frame.framesWithoutPlayer;
  total.immobileSamples += frame.immobileSamples;
  total.immobileSkipped += frame.immobileSkipped;
  total.immobileMaxPx = std::max(total.immobileMaxPx, frame.immobileMaxPx);
}

uint64_t player_eligible(const Counts& counts) noexcept {
  uint64_t eligible = 0;
  for (const PlayerCounts& player : counts.player) {
    eligible += player.eligible;
  }
  return eligible;
}

void log_counts(std::string_view label, const Counts& c) noexcept {
  const auto& p = c.player;
  Log.info(
      "Motion {}: none {} camera {} object {} palette {} jittered {}; exact {} fallback {} miss {} ambiguous {} "
      "overflow {} pushRefused {} badIndex {}; PLAYER object {}/{}/{}/{} palette {}/{}/{}/{} cpuDeformed "
      "{}/{}/{}/{} (eligible/exact/fallback/miss); frames {} without PLAYER {}; IMMOBILE samples {} skipped {} "
      "max {:.6f} px",
      label, c.none, c.camera, c.object, c.palette, c.jittered, c.exact, c.fallback, c.miss, c.ambiguous, c.overflow,
      c.pushRefused, c.badIndex, p[0].eligible, p[0].exact, p[0].fallback, p[0].miss, p[1].eligible, p[1].exact,
      p[1].fallback, p[1].miss, p[2].eligible, p[2].exact, p[2].fallback, p[2].miss, c.framesInWindow,
      c.framesWithoutPlayer, c.immobileSamples, c.immobileSkipped, c.immobileMaxPx);
}

void fold() noexcept {
  if (!g_active) {
    return;
  }
  Counts& frame = g_frame.counts;
  const bool inCheck = check_mode() && g_frame.valid && g_frame.index >= g_checkFirst && g_frame.index <= g_checkLast;
  if (inCheck) {
    frame.framesInWindow = 1;
    frame.framesWithoutPlayer = player_eligible(frame) == 0 ? 1 : 0;
    add(g_stats.check, frame);
  }
  ++g_stats.frames;
  g_stats.frame = frame;
  add(g_stats.total, frame);
  if (check_mode() && g_stats.frames % StatsLogFrames == 0) {
    log_counts("totals", g_stats.total);
  }
  if (g_frame.refusedBytes != 0) {
    gfx::detail::request_streaming_sizes({.storage = gfx::detail::streaming_sizes().storage + g_frame.refusedBytes});
  }
}

const Mat3x4<float>& position_matrix(uint32_t slot) noexcept {
  return slot < gx::MaxPnMtx ? gx::g_gxState.pnMtx[slot].pos : gx::g_gxState.texMtxs[slot - gx::MaxPnMtx];
}

void store(const Key& key, bool palette, uint32_t slot) noexcept {
  Table& table = g_tables[g_current];
  const uint32_t count = palette ? rx::MotionPaletteSlots : 1;
  if (table.tag == 0 || table.entries >= MaxEntries || table.used + count > ArenaMatrices) {
    ++g_frame.counts.overflow;
    return;
  }
  uint32_t index = slot_of(key);
  for (;; index = (index + 1) & TableMask) {
    Slot& entry = table.slots[index];
    if (entry.tag != table.tag) {
      break;
    }
    if (entry.key == key) {
      ++g_frame.counts.ambiguous;
      entry.ambiguous = true;
      return;
    }
  }
  if (palette) {
    for (uint32_t i = 0; i < count; ++i) {
      table.arena[table.used + i] = gx::g_gxState.pnMtx[i].pos;
    }
  } else {
    table.arena[table.used] = position_matrix(slot);
  }
  table.slots[index] = {.tag = table.tag, .key = key, .first = table.used, .count = static_cast<uint8_t>(count)};
  table.used += count;
  ++table.entries;
}

const Slot* find(const Table& table, const Key& key) noexcept {
  if (table.tag == 0) {
    return nullptr;
  }
  for (uint32_t index = slot_of(key);; index = (index + 1) & TableMask) {
    const Slot& entry = table.slots[index];
    if (entry.tag != table.tag) {
      return nullptr;
    }
    if (entry.key == key) {
      return entry.ambiguous ? nullptr : &entry;
    }
  }
}

Hit lookup(const Key& key, uint8_t count) noexcept {
  const Table& previous = g_tables[g_current ^ 1];
  if (const Slot* exact = find(previous, key); exact != nullptr && exact->count == count) {
    return {exact, true};
  }
  for (uint32_t back = 1; back <= OrdinalFallbacks && back <= key.ordinal; ++back) {
    Key earlier = key;
    earlier.ordinal = static_cast<uint16_t>(key.ordinal - back);
    if (const Slot* slot = find(previous, earlier); slot != nullptr && slot->count == count) {
      return {slot, false};
    }
  }
  return {};
}

bool previous_pushed() noexcept {
  if (g_frame.prevBase != 0) {
    return true;
  }
  if (std::exchange(g_frame.prevPushTried, true)) {
    return false;
  }
  const Table& previous = g_tables[g_current ^ 1];
  const std::span<const uint8_t> bytes{reinterpret_cast<const uint8_t*>(previous.arena.data()),
                                       size_t{previous.used} * sizeof(Mat3x4<float>)};
  const gfx::Range range = g_push(bytes);
  if (range.size == 0 || range.offset == 0) {
    ++g_frame.counts.pushRefused;
    g_frame.refusedBytes += bytes.size();
    return false;
  }
  g_frame.prevBase = range.offset / 4;
  return true;
}

void sample_immobile(const Key& key, uint32_t slot) noexcept {
  if (g_frame.immobileSamples >= ImmobileSamplesPerFrame || !g_previous.valid) {
    return;
  }
  const Slot* previous = find(g_tables[g_current ^ 1], key);
  if (previous == nullptr || previous->count != 1) {
    return;
  }
  ++g_frame.immobileSamples;
  const auto& viewport = gx::g_gxState.renderViewport;
  const std::optional<float> residual = detail::immobile_residual_px(
      position_matrix(slot), g_tables[g_current ^ 1].arena[previous->first], g_frame.reproject, g_frame.camera.gxProj,
      g_previous.camera.gxProj, viewport.width, viewport.height);
  if (!residual) {
    ++g_frame.counts.immobileSkipped;
    return;
  }
  ++g_frame.counts.immobileSamples;
  g_frame.counts.immobileMaxPx = std::max(g_frame.counts.immobileMaxPx, *residual);
}

struct Affine {
  double m[3][4];
};

Affine to_affine(const Mat3x4<float>& in) noexcept {
  Affine out{};
  const Vec4<float>* rows[3] = {&in.m0, &in.m1, &in.m2};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out.m[r][c] = (*rows[r])[c];
    }
  }
  return out;
}

Affine inverse(const Affine& a) noexcept {
  const auto& m = a.m;
  const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  Affine out{};
  if (det == 0.0 || !std::isfinite(det)) {
    return out;
  }
  const double inv = 1.0 / det;
  out.m[0][0] = c00 * inv;
  out.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
  out.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
  out.m[1][0] = c01 * inv;
  out.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
  out.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
  out.m[2][0] = c02 * inv;
  out.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
  out.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
  for (int r = 0; r < 3; ++r) {
    out.m[r][3] = -(out.m[r][0] * m[0][3] + out.m[r][1] * m[1][3] + out.m[r][2] * m[2][3]);
  }
  return out;
}

Affine multiply(const Affine& a, const Affine& b) noexcept {
  Affine out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c] + (c == 3 ? a.m[r][3] : 0.0);
    }
  }
  return out;
}

std::array<double, 3> transform(const Affine& a, const std::array<double, 3>& p) noexcept {
  std::array<double, 3> out{};
  for (int r = 0; r < 3; ++r) {
    out[r] = a.m[r][0] * p[0] + a.m[r][1] * p[1] + a.m[r][2] * p[2] + a.m[r][3];
  }
  return out;
}

float halton(uint32_t index, uint32_t base) noexcept {
  float fraction = 1.f;
  float result = 0.f;
  while (index > 0) {
    fraction /= static_cast<float>(base);
    result += fraction * static_cast<float>(index % base);
    index /= base;
  }
  return result;
}
} // namespace

namespace detail {
std::array<float, 2> halton_jitter(uint32_t phase) noexcept {
  return {halton(phase, 2) - 0.5f, halton(phase, 3) - 0.5f};
}

bool rows_match(const Mat4x4<float>& proj, const Mat4x4<float>& mainProj) noexcept {
  return std::memcmp(&proj.m0, &mainProj.m0, sizeof(proj.m0)) == 0 &&
         std::memcmp(&proj.m1, &mainProj.m1, sizeof(proj.m1)) == 0 &&
         std::memcmp(&proj.m3, &mainProj.m3, sizeof(proj.m3)) == 0;
}

Mat3x4<float> reproject_matrix(const Mat3x4<float>& prevView, const Mat3x4<float>& curView) noexcept {
  const Affine product = multiply(to_affine(prevView), inverse(to_affine(curView)));
  const auto row = [&](int r) {
    return Vec4<float>{static_cast<float>(product.m[r][0]), static_cast<float>(product.m[r][1]),
                       static_cast<float>(product.m[r][2]), static_cast<float>(product.m[r][3])};
  };
  return {row(0), row(1), row(2)};
}

std::optional<float> immobile_residual_px(const Mat3x4<float>& cur, const Mat3x4<float>& prev,
                                          const Mat3x4<float>& reproject, const Mat4x4<float>& curProj,
                                          const Mat4x4<float>& prevProj, float width, float height) noexcept {
  const double p22 = curProj.m2[2];
  const double p23 = curProj.m2[3];
  const double near = p23 / (p22 - 1.0);
  const double far = p22 != 0.0 ? p23 / p22 : std::numeric_limits<double>::infinity();
  const double depth = std::isfinite(far) && far > 0.0 ? std::sqrt(near * far) : 64.0 * near;
  if (!(near > 0.0) || !std::isfinite(depth)) {
    return std::nullopt;
  }
  const std::array<double, 3> view{0.0, 0.0, -depth};
  const std::array<double, 3> local = transform(inverse(to_affine(cur)), view);
  const std::array<double, 3> object = transform(to_affine(prev), local);
  const std::array<double, 3> camera = transform(to_affine(reproject), view);
  const auto project = [&](const std::array<double, 3>& p, double& x, double& y) {
    const auto dot = [&](const Vec4<float>& r) { return r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]; };
    const double w = dot(prevProj.m3);
    x = dot(prevProj.m0) / w;
    y = dot(prevProj.m1) / w;
    return w;
  };
  double ox = 0.0;
  double oy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  if (project(object, ox, oy) <= near || project(camera, cx, cy) <= near) {
    return std::nullopt;
  }
  const double dx = (ox - cx) * 0.5 * width;
  const double dy = (oy - cy) * 0.5 * height;
  return static_cast<float>(std::sqrt(dx * dx + dy * dy));
}

std::optional<std::string_view> check_failure(const Counts& check) noexcept {
  uint64_t eligible = 0;
  uint64_t hits = 0;
  uint64_t fallbacks = 0;
  for (const PlayerCounts& player : check.player) {
    eligible += player.eligible;
    hits += player.exact + player.fallback;
    fallbacks += player.fallback;
  }
  if (eligible == 0) {
    return "no PLAYER draws were eligible";
  }
  if (hits * 100 < eligible * 99) {
    return "PLAYER draws found their previous matrices below 99%";
  }
  if (fallbacks * 100 > eligible * 2) {
    return "PLAYER draws used the ordinal fallback above 2%";
  }
  if (check.framesWithoutPlayer * 20 > check.framesInWindow) {
    return "frames without PLAYER draws above 5%";
  }
  if (check.immobileSamples == 0) {
    return "no IMMOBILE samples";
  }
  if (check.immobileMaxPx > ImmobileMaxPx) {
    return "IMMOBILE residual above 0.05 px";
  }
  if (check.pushRefused != 0) {
    return "refused storage pushes";
  }
  return std::nullopt;
}
} // namespace detail

void latch(const Settings& settings, bool motionLayoutLive) noexcept {
  read_env();
  fold();
  const bool active = render::detail::hints_enabled() && motionLayoutLive;
  if (!active) {
    if (g_active) {
      clear_history();
    }
    g_active = false;
    g_jitterActive = false;
    g_jitter = {};
    g_frame = {};
    return;
  }
  allocate();
  const bool first = !g_active;
  g_active = true;
  const temporal_resolve::Resolver& resolver = temporal_resolve::installed();
  g_jitterActive = g_jitterEnv == testing::JitterEnv::On ||
                   (g_jitterEnv != testing::JitterEnv::Off && resolver.wanted != nullptr && resolver.wanted(settings));
  // The hint frame restarts with the hint stream, so a replayed run walks the same phases as the first one
  const uint64_t phase = render::detail::live_scene_hints().hintFrame % temporal_resolve::jitter_phases() + 1;
  g_jitter = g_jitterActive ? detail::halton_jitter(static_cast<uint32_t>(phase)) : std::array<float, 2>{};
  if (first) {
    clear_history();
  } else if (g_frame.usable) {
    g_previous = {true, g_frame.camera};
  } else {
    g_previous = {};
  }
  g_current ^= 1;
  Table& table = g_tables[g_current];
  table.tag = next_tag();
  table.entries = 0;
  table.used = 0;
  g_frame = {};
  g_frame.index = g_frames++;
}

std::array<float, 2> frame_jitter() noexcept { return g_jitter; }

const gx::hints::Camera* previous_camera() noexcept { return g_previous.valid ? &g_previous.camera : nullptr; }

Stats latest_stats() noexcept { return g_stats; }

void reset() noexcept {
  clear_history();
  g_frame = {};
}

void shutdown() noexcept {
  fold();
  if (g_stats.frames != 0) {
    log_counts("totals", g_stats.total);
    if (check_mode()) {
      log_counts("check", g_stats.check);
      if (const auto failure = detail::check_failure(g_stats.check)) {
        Log.warn("Motion: check FAILED: {}", *failure);
      } else {
        const uint64_t eligible = player_eligible(g_stats.check);
        uint64_t exact = 0;
        uint64_t fallback = 0;
        for (const PlayerCounts& player : g_stats.check.player) {
          exact += player.exact;
          fallback += player.fallback;
        }
        Log.info(
            "Motion: check passed (PLAYER {:.2f}% found, {:.2f}% fallback over {} draws, {} of {} frames "
            "without PLAYER, IMMOBILE {} samples max {:.6f} px)",
            100.0 * static_cast<double>(exact + fallback) / static_cast<double>(eligible),
            100.0 * static_cast<double>(fallback) / static_cast<double>(eligible), eligible,
            g_stats.check.framesWithoutPlayer, g_stats.check.framesInWindow, g_stats.check.immobileSamples,
            g_stats.check.immobileMaxPx);
      }
    }
  }
  g_active = false;
  g_jitterActive = false;
  g_jitter = {};
  g_frames = 0;
  g_stats = {};
  g_frame = {};
  clear_history();
  for (Table& table : g_tables) {
    table.slots = {};
    table.arena = {};
  }
}

Validity on_main_camera(const gx::hints::Camera& camera, bool usable) noexcept {
  if (!g_active) {
    return {};
  }
  g_frame.camera = camera;
  g_frame.usable = usable;
  g_frame.cameraLive = false;
  using render::detail::MotionReason;
  MotionReason reason = MotionReason::Valid;
  if (!usable) {
    reason = MotionReason::Unusable;
  } else if (!g_previous.valid) {
    reason = MotionReason::NoPrevious;
  } else if (camera.cutEpoch != g_previous.camera.cutEpoch) {
    reason = MotionReason::Cut;
  } else if (render::detail::live_scene_hints().sceneEpochChanged) {
    reason = MotionReason::SceneEpoch;
  }
  MotionFrame frame;
  if (g_previous.valid) {
    const Mat4x4<float>& prevProj = g_previous.camera.gxProj;
    frame.prevProj[0] = prevProj.m0;
    frame.prevProj[1] = prevProj.m1;
    frame.prevProj[2] = prevProj.m3;
    frame.reproject = detail::reproject_matrix(g_previous.camera.view, camera.view);
  }
  bool jitter = g_jitterActive && reason == MotionReason::Valid &&
                (g_jitterEnv == testing::JitterEnv::On || temporal_resolve::jitter_enabled());
  if (jitter) {
    frame.jitterPx[0] = g_jitter[0];
    frame.jitterPx[1] = g_jitter[1];
  }
  frame.valid = reason == MotionReason::Valid ? 1u : 0u;
  if (gx_ext::records::set_camera_frame(frame)) {
    g_frame.cameraLive = true;
  } else {
    reason = MotionReason::StorageFull;
    jitter = false;
    g_frame.refusedBytes += sizeof(gx_ext::records::FrameRecord) + sizeof(MotionFrame);
  }
  g_frame.reproject = frame.reproject;
  g_frame.valid = reason == MotionReason::Valid;
  g_frame.jitter = jitter;
  Validity validity{.valid = g_frame.valid, .reason = reason};
  if (jitter) {
    validity.jitterPx[0] = g_jitter[0];
    validity.jitterPx[1] = g_jitter[1];
  }
  return validity;
}

DrawMotion draw_motion(const gx::DrawData& draw, const gx::PipelineConfig& config, const DrawClass& cls) noexcept {
  if (!g_active || cls.offscreen) {
    return {};
  }
  Counts& counts = g_frame.counts;
  const gx::GXState& state = gx::g_gxState;
  const ClassKey& key = cls.key;
  const bool perspective = state.proj[3][3] == 0.f;
  const bool rowsMatch = g_frame.cameraLive && detail::rows_match(state.proj, g_frame.camera.gxProj);
  const bool inWindow = key.window == Window::Opaque || key.window == Window::Translucent ||
                        (cls.kind == Kind::Sky && key.window == Window::PreScene);
  const bool role = key.role == Role::Main || key.role == Role::Sky;
  DrawMotion motion;
  if (g_frame.jitter && perspective && inWindow && key.mainCamera && role) {
    motion.flags |= rx::RecordFlagJitter;
    ++counts.jittered;
  }
  const bool palette = config.shaderConfig.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE;
  const uint32_t slot = draw.immediateData.currentPnMtx;
  const bool badIndex = !palette && slot >= PositionSlots;
  const bool immobile = (key.flags & object_flag::Immobile) != 0;
  const bool teleport = (key.flags & object_flag::Teleport) != 0;
  const Key tableKey{
      .id = key.object.id,
      .subId = key.object.subId,
      .generation = key.object.generation,
      .ordinal = cls.ordinal,
      .role = static_cast<uint8_t>(key.role),
  };
  if (key.object.valid && role && key.window <= Window::Translucent) {
    if (badIndex) {
      ++counts.badIndex;
    } else if (!immobile || check_mode()) {
      store(tableKey, palette, slot);
    }
  }
  if (!g_frame.valid || !inWindow || !perspective || !role) {
    ++counts.none;
    return motion;
  }
  const bool objectPath = key.object.valid && !immobile && !teleport && !badIndex;
  const bool player = check_mode() && objectPath && key.role == Role::Main && (key.categories & category::Player) != 0;
  PlayerCounts* playerCounts = nullptr;
  if (player) {
    const PlayerKind kind = (key.flags & object_flag::CpuDeformed) != 0 ? PlayerKind::CpuDeformed
                            : palette                                   ? PlayerKind::Palette
                                                                        : PlayerKind::Object;
    playerCounts = &counts.player[static_cast<size_t>(kind)];
    ++playerCounts->eligible;
  }
  if (objectPath) {
    const Hit hit = lookup(tableKey, static_cast<uint8_t>(palette ? rx::MotionPaletteSlots : 1));
    if (hit.slot != nullptr && previous_pushed()) {
      motion.flags |= palette ? rx::MotionSourcePalette : rx::MotionSourceObject;
      if (rowsMatch) {
        motion.flags |= rx::RecordFlagPrevProjFrame;
      }
      motion.prevMtxOff = g_frame.prevBase + hit.slot->first * MatrixWords;
      ++(palette ? counts.palette : counts.object);
      ++(hit.exact ? counts.exact : counts.fallback);
      if (playerCounts != nullptr) {
        ++(hit.exact ? playerCounts->exact : playerCounts->fallback);
      }
      return motion;
    }
    ++counts.miss;
    if (playerCounts != nullptr) {
      ++playerCounts->miss;
    }
  }
  if (immobile && check_mode() && key.object.valid && !palette && !badIndex && rowsMatch) {
    sample_immobile(tableKey, slot);
  }
  if (rowsMatch) {
    motion.flags |= rx::MotionSourceCamera | rx::RecordFlagPrevProjFrame;
    ++counts.camera;
  } else {
    ++counts.none;
  }
  return motion;
}

namespace testing {
void set_jitter_env(JitterEnv env) noexcept {
  g_envRead = true;
  g_jitterEnv = env;
}

void set_check_range(uint32_t first, uint32_t last) noexcept {
  g_envRead = true;
  g_checkFirst = first;
  g_checkLast = last;
}

void set_push(PushFn push) noexcept { g_push = push != nullptr ? push : &default_push; }

bool active() noexcept { return g_active; }

bool jitter_active() noexcept { return g_jitterActive; }

uint64_t storage_fingerprint() noexcept {
  uint64_t hash = 0;
  for (const Table& table : g_tables) {
    hash = splitmix64(hash ^ reinterpret_cast<uintptr_t>(table.slots.data()) ^ table.slots.capacity());
    hash = splitmix64(hash ^ reinterpret_cast<uintptr_t>(table.arena.data()) ^ table.arena.capacity());
  }
  return hash;
}

void reset() noexcept {
  shutdown();
  g_push = &default_push;
  g_envRead = true;
  g_jitterEnv = JitterEnv::Unset;
  g_checkFirst = 0;
  g_checkLast = 0;
}
} // namespace testing
} // namespace aurora::render::scene::motion

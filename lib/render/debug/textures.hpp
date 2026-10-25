#pragma once

#include "pick.hpp"

#include "../scene/classifier.hpp"
#include "../../gfx/texture.hpp"

#include <aurora/texture.hpp>

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::render::material {
struct Snapshot;
} // namespace aurora::render::material

namespace aurora::render::debug::textures {
inline constexpr std::string_view TabName = "Textures";
inline constexpr uint32_t MaxDraws = 16384;
inline constexpr uint32_t MaxSlots = 65536;
inline constexpr uint32_t MaxKeyRows = 4096;
inline constexpr uint32_t MaxWarmConfigs = 1024;
inline constexpr size_t KeyMemoCapacity = 16384;
inline constexpr uint32_t KeyMemoInsertsPerFrame = 4096;
inline constexpr uint64_t HashBudgetBytes = 4u << 20;
inline constexpr size_t ReportRing = 8;
inline constexpr uint32_t WarmLookupsPerLatch = 64;
inline constexpr uint32_t NoRow = UINT32_MAX;

enum class SlotKind : uint8_t { Empty, Static, Palette, Copy, DynamicPalette };
enum class KeyState : uint8_t { None, Ready, Pending, Paused, Failed };

struct SlotRecord {
  uint32_t row = NoRow;
  gfx::TextureMaterialRef material;
  uint16_t width = 0;
  uint16_t height = 0;
  uint8_t texmap = 0;
  uint8_t format = 0;
  uint8_t wrapS = 0;
  uint8_t wrapT = 0;
  SlotKind kind = SlotKind::Empty;
  bool replaced = false;
  bool stochastic = false;
  bool indirect = false;
  bool pointerReplacement = false;
  bool neutral = false;
};

struct DrawRecord {
  uint32_t index = 0;
  uint32_t locator = 0;
  uint32_t categories = 0;
  uint64_t objectId = 0;
  uint32_t slotBegin = 0;
  uint8_t slotCount = 0;
  scene::Role role = scene::Role::None;
  scene::Window window = scene::Window::PreScene;
  bool mainClass = false;
  bool inSpan = false;
  bool skipped = false;
};

struct KeyRow {
  texture::TextureSourceKey key;
  gfx::TextureMaterialRef material;
  uint32_t draws = 0;
  uint32_t mainDraws = 0;
  uint32_t neutralDraws = 0;
  uint16_t width = 0;
  uint16_t height = 0;
  uint8_t format = 0;
  SlotKind kind = SlotKind::Empty;
  KeyState state = KeyState::None;
  bool replaced = false;
  bool stochastic = false;
  bool pointerReplacement = false;
};

struct Overflow {
  uint32_t draws = 0;
  uint32_t slots = 0;
  uint32_t keys = 0;
  uint32_t configs = 0;
};

struct TexturesReport {
  uint64_t sequence = 0;
  std::vector<DrawRecord> draws;
  std::vector<uint32_t> stamped;
  std::vector<SlotRecord> slots;
  std::vector<KeyRow> keys;
  Overflow overflow;
  uint32_t pendingKeys = 0;
  uint32_t pausedKeys = 0;
  uint64_t hashedBytes = 0;
};

using SeenSet = std::bitset<65536>;

// Game thread
void initialize();
void shutdown() noexcept;
void on_latch() noexcept;
void draw_tab();
void set_pick_enabled(bool enabled) noexcept;
[[nodiscard]] pick::ArmState arm_state() noexcept;
[[nodiscard]] std::shared_ptr<const TexturesReport> latest_report() noexcept;
[[nodiscard]] std::shared_ptr<const TexturesReport> report_for(uint32_t sequence) noexcept;
[[nodiscard]] uint32_t flagged_seen(const SeenSet& seen, const material::Snapshot* snapshot) noexcept;

// FIFO thread
void observe(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
             const scene::DrawClass& cls) noexcept;

namespace testing {
struct Capacities {
  size_t draws = 0;
  size_t slots = 0;
  size_t keys = 0;
};

struct HoverState {
  uint64_t sequence = 0;
  uint32_t record = 0;
  bool valid = false;
};

void reset() noexcept;
void force_visible(int value) noexcept;
void set_hash_budget(uint64_t bytes) noexcept;
void set_arm_state(pick::ArmState state) noexcept;
void set_pointer_probe(bool (*probe)(const GXTexObj_& obj) noexcept) noexcept;
void set_hover_pinned(bool pinned) noexcept;
[[nodiscard]] bool observer_installed() noexcept;
[[nodiscard]] size_t memo_size() noexcept;
[[nodiscard]] uint64_t sequence() noexcept;
[[nodiscard]] Capacities live_capacities() noexcept;
[[nodiscard]] HoverState hover() noexcept;
} // namespace testing
} // namespace aurora::render::debug::textures

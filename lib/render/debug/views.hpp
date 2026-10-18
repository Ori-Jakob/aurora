#pragma once

#include "../scene/classifier.hpp"

#include "../../gfx/types.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::debug {
enum class ClassView : uint16_t {
  Off = 0,
  Role = 1,
  Kind = 2,
  MainCheck = 3,
  Window = 4,
  Category = 5,
  Object = 6,
  Scope = 7,
  Bit = 8,
};
// Ids up to this one belong to classification; later ids are other views and select nothing here
inline constexpr uint16_t LastReservedClassView = 15;

struct DebugView {
  uint16_t id = 0;
  std::string_view name;
  std::string_view label;
};

inline constexpr std::array<DebugView, 8> ClassViews{{
    {1, "class.role", "Classification: camera role"},
    {2, "class.kind", "Classification: draw class"},
    {3, "class.main", "Classification: main-class test"},
    {4, "class.window", "Classification: stage window"},
    {5, "class.category", "Classification: category"},
    {6, "class.object", "Classification: object"},
    {7, "class.scope", "Classification: scope"},
    {8, "class.bit", "Classification: one category"},
}};

struct ViewSelection {
  uint16_t id = 0;
  uint32_t bit = 0;
};

struct Rgb8 {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  bool operator==(const Rgb8&) const = default;
};

inline constexpr Rgb8 UnsetColor{64, 64, 64};
inline constexpr Rgb8 BitOnColor{255, 80, 255};
inline constexpr std::array<Rgb8, scene::RoleCount> RoleColors{
    {{60, 200, 60}, {80, 160, 255}, {255, 80, 255}, {150, 90, 40}, {255, 160, 0}, {255, 240, 80}, UnsetColor}};
inline constexpr std::array<Rgb8, scene::KindCount> KindColors{
    {{60, 200, 60}, {170, 255, 170}, {255, 200, 0}, {80, 160, 255}, {210, 50, 50}, {0, 110, 0}}};
inline constexpr std::array<Rgb8, scene::MainCheckCount> CheckColors{
    {{60, 200, 60}, {210, 50, 50}, {255, 160, 0}, {160, 90, 255}, {255, 240, 80}, {128, 128, 128}, {0, 220, 220}}};
inline constexpr std::array<Rgb8, scene::WindowCount> WindowColors{
    {{128, 128, 128}, {60, 200, 60}, {80, 160, 255}, {255, 160, 0}, {210, 50, 50}, {255, 240, 80}, {255, 80, 255}}};

struct CategoryColor {
  uint32_t bit = 0;
  Rgb8 color;
};

// Highest priority first; view 8 shows the bits this order hides
inline constexpr std::array<CategoryColor, 14> CategoryColors{{
    {scene::category::FakeShadow, {90, 0, 140}},
    {scene::category::Overlay, {255, 200, 0}},
    {scene::category::Ui, {255, 240, 80}},
    {scene::category::Sky, {80, 160, 255}},
    {scene::category::Particle, {255, 120, 200}},
    {scene::category::Decal, {150, 90, 40}},
    {scene::category::Water, {0, 120, 255}},
    {scene::category::Foliage, {40, 160, 40}},
    {scene::category::Terrain, {160, 130, 80}},
    {scene::category::Player, {255, 80, 80}},
    {scene::category::Skinned, {255, 160, 0}},
    {scene::category::Emissive, {255, 255, 255}},
    {scene::category::Dynamic, {0, 220, 220}},
    {scene::category::Static, {180, 180, 180}},
}};

[[nodiscard]] constexpr const DebugView* find_view(uint16_t id) noexcept {
  for (const DebugView& view : ClassViews) {
    if (view.id == id) {
      return &view;
    }
  }
  return nullptr;
}

[[nodiscard]] constexpr std::optional<uint32_t> category_bit(std::string_view name) noexcept {
  for (uint32_t bit = 0; bit < 32; ++bit) {
    const std::string_view candidate = scene::category_name(bit);
    if (!candidate.empty() && candidate == name) {
      return 1u << bit;
    }
  }
  return std::nullopt;
}

// Bit views take classify.json's category names, so a selected bit reads the same as the counts it explains
[[nodiscard]] constexpr std::optional<ViewSelection> parse_view(std::string_view text) noexcept {
  constexpr std::string_view Prefix = "class.";
  constexpr std::string_view BitPrefix = "bit:";
  const auto select = [](uint16_t id) {
    return ViewSelection{id, id == static_cast<uint16_t>(ClassView::Bit) ? scene::category::Static : 0u};
  };
  if (text.starts_with(BitPrefix)) {
    const std::optional<uint32_t> bit = category_bit(text.substr(BitPrefix.size()));
    if (!bit) {
      return std::nullopt;
    }
    return ViewSelection{static_cast<uint16_t>(ClassView::Bit), *bit};
  }
  for (const DebugView& view : ClassViews) {
    if (text == view.name || text == view.name.substr(Prefix.size())) {
      return select(view.id);
    }
  }
  if (text.empty()) {
    return std::nullopt;
  }
  uint32_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    value = value * 10 + static_cast<uint32_t>(c - '0');
    if (value > UINT16_MAX) {
      return std::nullopt;
    }
  }
  if (value == 0) {
    return std::nullopt;
  }
  return select(static_cast<uint16_t>(value));
}

// splitmix64, with every channel kept above the unset grey's darkness
[[nodiscard]] constexpr Rgb8 hash_color(uint64_t key) noexcept {
  uint64_t x = key + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  const auto channel = [x](uint32_t byte) { return static_cast<uint8_t>(64 + ((x >> (8 * byte)) & 0xFF) * 191 / 255); };
  return {channel(0), channel(1), channel(2)};
}

[[nodiscard]] constexpr Rgb8 lighten(Rgb8 color) noexcept {
  const auto half = [](uint8_t c) { return static_cast<uint8_t>(c + (255 - c) / 2); };
  return {half(color.r), half(color.g), half(color.b)};
}

[[nodiscard]] constexpr Rgb8 view_color(ClassView view, uint32_t bit, const scene::DrawClass& cls) noexcept {
  const scene::ClassKey& key = cls.key;
  switch (view) {
  case ClassView::Role:
    return RoleColors[static_cast<size_t>(key.role)];
  case ClassView::Kind:
    return KindColors[static_cast<size_t>(cls.kind)];
  case ClassView::MainCheck:
    return CheckColors[static_cast<size_t>(cls.check)];
  case ClassView::Window:
    return WindowColors[static_cast<size_t>(key.window)];
  case ClassView::Category:
    for (const CategoryColor& entry : CategoryColors) {
      if ((key.categories & entry.bit) != 0) {
        return entry.color;
      }
    }
    return UnsetColor;
  case ClassView::Object:
    if (key.object.valid) {
      return hash_color(key.object.id ^ (uint64_t{key.object.subId} << 48));
    }
    // Scope-only draws still get a colour of their own, lighter than any object's
    return key.scope != 0 ? lighten(hash_color(key.scope)) : UnsetColor;
  case ClassView::Scope:
    return key.scope != 0 ? hash_color(key.scope) : UnsetColor;
  case ClassView::Bit:
    return (key.categories & bit) != 0 ? BitOnColor : UnsetColor;
  case ClassView::Off:
    break;
  }
  return UnsetColor;
}

// Post and HUD draws keep their look, so the screen stays readable under a view
[[nodiscard]] constexpr bool in_view_span(const scene::DrawClass& cls) noexcept {
  return cls.key.mainCamera && cls.key.window <= scene::Window::Translucent && !cls.offscreen && cls.colorWrite;
}

void set_view_override(uint16_t id) noexcept;
[[nodiscard]] uint16_t view_override() noexcept;
void set_view_bit(uint32_t bit) noexcept;
[[nodiscard]] uint32_t view_bit() noexcept;
void initialize_classification();
void shutdown_classification() noexcept;
void draw_scene_tab();

// Only colour may change, because alpha feeds the alpha test, blending and EFB alpha; such variants are refused
[[nodiscard]] bool make_view_variant(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                                     gx::PipelineConfig& out) noexcept;
void begin_view_frame(const detail::FrameState& state) noexcept;
bool push_view(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
               const scene::DrawClass& cls) noexcept;

namespace testing {
void set_variant_lookup(gfx::PipelineRef (*lookup)(const gx::PipelineConfig&, const gfx::RenderTargetLayout&)) noexcept;
void assume_variants_ready(bool ready) noexcept;
void set_view_uniform_budget(uint32_t bytes) noexcept;
void reset_views() noexcept;
[[nodiscard]] ClassView active_view() noexcept;
[[nodiscard]] uint32_t view_uniform_budget() noexcept;
} // namespace testing
} // namespace aurora::render::debug

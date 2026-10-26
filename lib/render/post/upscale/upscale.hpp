#pragma once

#include "../../graph.hpp"
#include "../../stage.hpp"

#include <aurora/render.hpp>

#include <array>
#include <optional>
#include <string_view>

namespace aurora::render::detail::upscale {
inline constexpr float MaxSplitRatio = 3.f;
inline constexpr float MaxExtraMipBias = 4.f;

struct SplitInputs {
  GraphExtent output;
  float ratio = 0.f;
  bool sceneTargets = false;
  bool moduleEnabled = true;
};

struct Signs {
  std::array<float, 2> jitter{1.f, 1.f};
  std::array<float, 2> motion{1.f, 1.f};
  bool operator==(const Signs&) const = default;
};

[[nodiscard]] std::optional<float> parse_split_ratio(std::string_view text) noexcept;
[[nodiscard]] std::optional<float> parse_mip_bias(std::string_view text) noexcept;
[[nodiscard]] std::optional<Signs> parse_signs(std::string_view text) noexcept;
[[nodiscard]] std::optional<ColorEncoding> parse_color(std::string_view text) noexcept;
[[nodiscard]] GraphExtent split_size(GraphExtent output, float ratio) noexcept;
[[nodiscard]] GraphExtent wanted_scene_size(const SplitInputs& inputs) noexcept;

void initialize();
bool on_latch(const Settings& settings);
void latch_graph(const Settings& settings, const CompiledGraph* graph);
void shutdown() noexcept;
[[nodiscard]] bool split_live() noexcept;
[[nodiscard]] bool compose_wanted() noexcept;
[[nodiscard]] GraphExtent scene_size() noexcept;
[[nodiscard]] GraphExtent output_size() noexcept;
[[nodiscard]] float ratio() noexcept;
[[nodiscard]] float rcas_strength() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] UpscaleStatus status() noexcept;

namespace testing {
void set_split_env(std::optional<float> ratio) noexcept;
void set_mip_bias_env(std::optional<float> extra) noexcept;
void set_signs_env(std::optional<Signs> signs) noexcept;
void set_color_env(std::optional<ColorEncoding> color) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::upscale

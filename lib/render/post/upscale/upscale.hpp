#pragma once

#include "../../graph.hpp"

#include <aurora/render.hpp>

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

[[nodiscard]] std::optional<float> parse_split_ratio(std::string_view text) noexcept;
[[nodiscard]] std::optional<float> parse_mip_bias(std::string_view text) noexcept;
[[nodiscard]] GraphExtent split_size(GraphExtent output, float ratio) noexcept;
[[nodiscard]] GraphExtent wanted_scene_size(const SplitInputs& inputs) noexcept;

bool on_latch(const Settings& settings);
void check_graph(const CompiledGraph* graph);
void shutdown() noexcept;
[[nodiscard]] bool split_live() noexcept;
[[nodiscard]] bool compose_wanted() noexcept;
[[nodiscard]] GraphExtent scene_size() noexcept;
[[nodiscard]] GraphExtent output_size() noexcept;
[[nodiscard]] float ratio() noexcept;
[[nodiscard]] UpscaleStatus status() noexcept;

namespace testing {
void set_split_env(std::optional<float> ratio) noexcept;
void set_mip_bias_env(std::optional<float> extra) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::upscale

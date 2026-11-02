#pragma once

#include "views.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace aurora::render::debug::decomp_view {
inline constexpr std::string_view Shader = "debug/decomp_view";

using Texel = std::array<uint8_t, 4>;

[[nodiscard]] bool declarable(const GbufView& row, const detail::PassSetup& setup) noexcept;
void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GbufView& row);
[[nodiscard]] shaders::RenderPipelineDesc pipeline_desc(DecompMode mode);
void pipelines(const detail::PipelineSetup& setup, const GbufView& row,
               std::span<uint32_t, detail::MaxPassPipelines> ids);
[[nodiscard]] bool available(const GbufView& row, const Settings& settings, const detail::CompiledGraph* graph);

[[nodiscard]] Rgb8 identity_color(Texel legacy, Texel out, Texel rec, uint32_t code) noexcept;
[[nodiscard]] Rgb8 coverage_color(Texel rec, uint32_t code, Texel legacy, uint8_t reactive) noexcept;
[[nodiscard]] Rgb8 defog_color(Texel legacy, Texel rec) noexcept;
[[nodiscard]] Rgb8 multiply_color(Texel legacy, uint32_t code, uint8_t ao) noexcept;
} // namespace aurora::render::debug::decomp_view

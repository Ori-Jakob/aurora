#pragma once

#include "views.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"

#include <cstdint>
#include <string_view>

namespace aurora::render::debug::gbuf_view {
inline constexpr std::string_view SnapshotPass = "debug.view";
inline constexpr std::string_view BlitPass = "debug.view_blit";
inline constexpr std::string_view SnapshotResource = "debug.view";
inline constexpr std::string_view Shader = "debug/gbuf_view";
inline constexpr uint16_t LastPassOrder = 0xFFFF;

[[nodiscard]] shaders::RenderPipelineDesc snapshot_pipeline_desc(const GbufView& view);
[[nodiscard]] shaders::RenderPipelineDesc blit_pipeline_desc(const detail::GraphEnvironment& env);

void on_latch(uint16_t selected);
void shutdown() noexcept;
[[nodiscard]] const GbufView* active_view() noexcept;
[[nodiscard]] uint64_t generation() noexcept;
} // namespace aurora::render::debug::gbuf_view

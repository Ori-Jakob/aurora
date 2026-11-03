#pragma once

#include "../material/classes.hpp"

#include <cstdint>

namespace aurora::gx {
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::gx_ext::unlit {
struct DrawMaterial {
  uint32_t vertexPolicy = 0;
  bool unlit = false;
  uint32_t word = 0;
  material::classes::Source source = material::classes::Source::Default;
};

// FIFO thread
[[nodiscard]] bool output2(const gx::PipelineConfig& config, const gx::ShaderInfo& info, uint32_t inputs) noexcept;
[[nodiscard]] DrawMaterial draw_material(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                                         const scene::DrawClass& cls) noexcept;
[[nodiscard]] bool surface_ext(const gx::PipelineConfig& config, const gx::ShaderInfo& info, uint32_t inputs) noexcept;
[[nodiscard]] uint32_t vertex_policy(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                                     const scene::DrawClass& cls) noexcept;
[[nodiscard]] bool unlit_flag(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                              const scene::DrawClass& cls) noexcept;
[[nodiscard]] bool policy_enabled() noexcept;
[[nodiscard]] bool classes_enabled() noexcept;

// Game thread, while the FIFO is idle
void apply_frame_state(const detail::FrameState& state) noexcept;
void shutdown() noexcept;

namespace testing {
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::gx_ext::unlit

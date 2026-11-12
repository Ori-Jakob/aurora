#pragma once

#include "../graph.hpp"

#include <aurora/gfx.hpp>
#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::detail {
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::fakeshadow::lsm {
inline constexpr std::string_view Pass = "lsm.mask";
inline constexpr std::string_view Resource = "lsm.mask";
inline constexpr std::string_view ConsumerName = "lsm";
inline constexpr uint16_t Order = 3;
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::BGRA8Unorm;
// scene.hdr alpha for PBR pixels drawn after the game's fake-shadow pass; readers test >= 0.5 for PBR
inline constexpr float PostShadowAlpha = 0.75f;
inline constexpr uint64_t WarmUpLatches = 60;

struct Params {
  float clearAlpha = 0.f;
  uint32_t pad[3]{};
};
static_assert(sizeof(Params) == 16);

struct Stats {
  uint64_t latches = 0;
  uint64_t declaredLatches = 0;
  uint64_t consumedLatches = 0;
  uint64_t selected = 0;
  uint64_t depthRefused = 0;
  uint64_t frames = 0;
  uint64_t replayed = 0;
  uint64_t pending = 0;
  uint64_t pendingAfterWarmUp = 0;
  uint64_t postShadowDraws = 0;
  uint64_t graphHash = 0;
};

// Game thread
void initialize();
void shutdown() noexcept;
// After the composite's latch, which decides FlagPbr
void on_latch(const detail::PassSetup& latched, const detail::CompiledGraph* graph);
// Game thread while the FIFO is idle, before the capture list applies its own state
void apply_frame_state() noexcept;
[[nodiscard]] bool declared(const detail::PassSetup& setup) noexcept;
[[nodiscard]] bool consumed() noexcept;
[[nodiscard]] gfx::RenderTargetLayout layout(wgpu::TextureFormat depth) noexcept;
[[nodiscard]] bool select_draw(const scene::DrawClass& cls) noexcept;
[[nodiscard]] detail::PassDesc pass_desc();

// FIFO thread
[[nodiscard]] bool consumed_fifo() noexcept;
void note_post_shadow() noexcept;
// FIFO at stage build, once the stage's passes are enabled
[[nodiscard]] bool frame_enabled(const detail::StageData& stage) noexcept;

// Any thread
[[nodiscard]] Stats stats() noexcept;

// The single mask decode: GX fog the clones carried is divided back out, and white or fog-drowned texels are exact 1
[[nodiscard]] std::array<float, 3> term(std::array<float, 3> mask, float q, std::array<float, 3> fc) noexcept;

namespace testing {
void reset() noexcept;
void set_consumed(std::optional<bool> consumed) noexcept;
[[nodiscard]] uint32_t consumer() noexcept;
} // namespace testing
} // namespace aurora::render::fakeshadow::lsm

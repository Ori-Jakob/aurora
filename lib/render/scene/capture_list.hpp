#pragma once

#include "../../gfx/recording.hpp"

#include <aurora/gfx.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::gx::draw_route {
struct Query;
} // namespace aurora::gx::draw_route

namespace aurora::render::detail {
class PassContext;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::capture_list {
inline constexpr uint32_t MaxConsumers = gfx::MaxCaptureLists;
inline constexpr std::string_view CheckPass = "capture.check";
inline constexpr std::string_view CheckColor = "capture.check.color";
inline constexpr std::string_view CheckDepth = "capture.check.depth";

using ConsumerId = uint32_t;
inline constexpr ConsumerId InvalidConsumer = UINT32_MAX;

struct Consumer {
  const char* name = nullptr;
  // Runs at route and must select every draw accept would take
  bool (*select)(const DrawClass& cls, void* user) noexcept = nullptr;
  bool (*accept)(const gx::DrawData& draw, const DrawClass& cls, void* user) noexcept = nullptr;
  // Reserved: it may only swap in a uniform block the consumer pushed itself, never patch a pushed one
  bool (*edit)(gx::DrawData& draw, const DrawClass& cls, void* user) noexcept = nullptr;
  gfx::RenderTargetLayout layout;
  void* user = nullptr;
};

struct Stats {
  uint64_t selected = 0;
  uint64_t cloned = 0;
  uint64_t pending = 0;
  uint64_t lateRefused = 0;
  uint64_t gpuRefused = 0;
  uint64_t discarded = 0;
  uint64_t replayed = 0;
};

// Game thread; registrations and the enabled flags take effect at the next apply_frame_state
ConsumerId register_consumer(const Consumer& consumer) noexcept;
void unregister_consumer(ConsumerId id) noexcept;
void set_enabled(ConsumerId id, bool enabled) noexcept;
// FIFO idle
void apply_frame_state() noexcept;
// FIFO thread; the mask each push receives is the one its own route returned
[[nodiscard]] uint8_t select(const DrawClass& cls, const gx::draw_route::Query& query) noexcept;
void on_push(const gx::DrawData& draw, const gx::PipelineConfig& config, const DrawClass& cls, uint8_t mask) noexcept;
// Render worker, inside the consumer's own phase.capture pass on a target of the consumer's layout
void encode(const detail::PassContext& ctx, const wgpu::RenderPassEncoder& pass, ConsumerId id);
[[nodiscard]] Stats stats() noexcept;
void shutdown() noexcept;

namespace testing {
struct PipelineHooks {
  gfx::PipelineRef (*ready)(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) noexcept = nullptr;
  void (*warm)(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) noexcept = nullptr;
};
// Null restores the pipeline cache
void set_pipeline_hooks(const PipelineHooks* hooks) noexcept;

enum class CheckState : uint8_t { Idle, Armed, Ready, Failed };
struct CheckResult {
  CheckState state = CheckState::Idle;
  uint32_t frame = 0;
  uint32_t mismatches = 0;
  // One past the last mismatching texel index, or 0
  uint32_t lastMismatch = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t draws = 0;
};
// Game thread before frames start; selects every non-offscreen draw up to the opaque window into capture.check
ConsumerId install_check(const wgpu::Color& clearColor, float clearDepth);
void remove_check() noexcept;
void arm_check(uint32_t frameIndex) noexcept;
[[nodiscard]] CheckResult check_result() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::scene::capture_list

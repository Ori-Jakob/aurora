#pragma once

#include "types.hpp"

#include <optional>
#include <span>

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

struct CompiledPipeline {
  wgpu::RenderPipeline main;
  wgpu::RenderPipeline prepass;

  [[nodiscard]] uint32_t pipeline_count() const {
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();
// Replay must never skip a draw because its pipeline is still compiling.
void set_blocking_pipelines(bool blocking) noexcept;
bool blocking_pipelines() noexcept;

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);
// Debug variants are built like any GX pipeline but never remembered, so they never reach the cache file
PipelineRef find_debug_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);

bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);

enum class PipelineState : uint8_t { Pending, Ready, Failed };
// Failed means the entry exists with no main pipeline, which render() skips
PipelineState pipeline_state(PipelineRef ref) noexcept;
uint64_t pipeline_epoch() noexcept;
// The ref only if that GX pipeline is already compiled; never queues, remembers or persists
PipelineRef ready_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept;

// The inspector's pick target comes and goes, so warm-up tracks the layout without it
RenderTargetLayout pipeline_base_layout(RenderTargetLayout layout) noexcept;

PipelineRef warm_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef warm_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);

struct WarmProgress {
  uint32_t ready = 0;
  uint32_t total = 0;
  uint32_t failed = 0;
};
WarmProgress warm_known_pipelines(const RenderTargetLayout& layout, ShaderType type);
WarmProgress count_resolved(std::span<const PipelineRef> refs) noexcept;
bool background_pipelines_drained() noexcept;

void note_skipped_draw() noexcept;
uint64_t skipped_draws() noexcept;

namespace detail {
enum class PipelinePriority : uint8_t {
  Background, // cache warmup
  Warm,       // pending scene layout
  Normal,     // async skip draw
  Blocking,   // block until compiled
};

std::optional<PipelinePriority> next_queue(bool normal, bool warm, bool background) noexcept;
PipelinePriority promoted(PipelinePriority current, PipelinePriority requested) noexcept;
} // namespace detail

} // namespace aurora::gfx

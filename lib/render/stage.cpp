#include "stage.hpp"

#include "frame_state.hpp"
#include "graph.hpp"
#include "graph_bind.hpp"
#include "graph_pool.hpp"
#include "../logging.hpp"

#if AURORA_ENABLE_RENDER
#include "shaders.hpp"
#endif

#include <aurora/gfx.h>

#include <array>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render"};

// Recording thread only; a frame's stages are built in order and never concurrently
FrameCullState g_frameCull;

const CompiledGraph* stage_graph(const StageData& stage) noexcept {
  return stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
}

bool has_native_passes(const FrameState* frame) noexcept {
  return frame != nullptr && frame->graph != nullptr && frame->graph->hasNativePasses;
}

void warn_missing_marker(Stage stage) {
  // Stages are only built on the recording thread
  static std::array<bool, StageCount> warned{};
  if (std::exchange(warned[static_cast<size_t>(stage)], true)) {
    return;
  }
  Log.warn("Render stage {} marker missing; passes that need it are skipped", stage_name(stage));
}
} // namespace

bool build_stage_data(StageData& stage) {
  const CompiledGraph* graph = stage_graph(stage);
  if (graph == nullptr) {
    return false;
  }
#if AURORA_ENABLE_RENDER
  // Runs before the empty-program return so every frame resets the cull state
  if (stage.stage == Stage::SceneBegin) {
    begin_frame_cull(*graph, g_frameCull);
  }
#endif
  const StageProgram& program = graph->stages[static_cast<size_t>(stage.stage)];
  if (program.passes.empty()) {
    return false;
  }
  const auto passCount = static_cast<uint32_t>(program.passes.size());
  stage.passCount = passCount;
  stage.enabled.assign((passCount + 63) / 64, 0);
  const bool needsMarker = !stage.markerSeen && stage.stage <= Stage::OpaqueDone;
  for (uint32_t i = 0; i < passCount; ++i) {
    const CompiledNode& node = graph->nodes[program.passes[i]];
    if (needsMarker && !node.runsWithoutMarker) {
      warn_missing_marker(stage.stage);
      continue;
    }
    if (node.enabled != nullptr && !node.enabled(stage, node.state)) {
      continue;
    }
#if AURORA_ENABLE_RENDER
    // A pass still compiling drops out like a disabled one, so the cull treats it as missing for its consumers
    if (!shaders::pipelines_ready(node.pipelineIds)) {
      continue;
    }
#endif
    stage.enabled[i / 64] |= uint64_t{1} << (i % 64);
  }
#if AURORA_ENABLE_RENDER
  StageMask mask;
  cull_stage(*graph, stage.stage, stage.enabled, g_frameCull, mask);
  stage.enabled = std::move(mask.passes);
  stage.dummies = std::move(mask.dummies);
  stage.prevValid = std::move(mask.prevValid);
  if (!stage.any_enabled()) {
    return false;
  }
  build_stage_bindings(stage);
  return true;
#else
  return stage.any_enabled();
#endif
}

void run_stage(const wgpu::CommandEncoder& encoder, const StageData& stage) {
  const CompiledGraph* graph = stage_graph(stage);
  if (graph == nullptr) {
    return;
  }
#if AURORA_ENABLE_RENDER
  // The worker hook already did this for the frame's graph; without an encoder nothing could use the result
  if (encoder) {
    transient_pool().materialize(stage.frame->graph, TransientPool::Clock::now());
  }
  const StageProgram& program = graph->stages[static_cast<size_t>(stage.stage)];
  for (uint32_t i = 0; i < stage.passCount; ++i) {
    if (!stage.is_enabled(i)) {
      continue;
    }
    const NodeId nodeId = program.passes[i];
    poison_before(encoder, stage, nodeId);
    // A pass skipped at bind time wrote nothing, so its histories must not flip
    if (encode_pass(encoder, stage, nodeId)) {
      after_pass(stage, nodeId);
    }
  }
#endif
}

// Native passes are rejected at registration, so no graph has a submit bracket to open yet
void before_submit(const FrameState* frame) noexcept {
  if (!has_native_passes(frame)) {
    return;
  }
}

void after_frame(const FrameState* frame) noexcept {
  if (!has_native_passes(frame)) {
    return;
  }
}

void reset_frame_cull() noexcept { g_frameCull = {}; }

void invalidate_frame_histories() noexcept {
#if AURORA_ENABLE_RENDER
  invalidate_histories(g_frameCull);
#endif
}
} // namespace aurora::render::detail

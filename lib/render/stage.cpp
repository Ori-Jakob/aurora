#include "stage.hpp"

#include "frame_state.hpp"
#include "graph.hpp"
#include "graph_bind.hpp"
#include "graph_pool.hpp"
#include "../logging.hpp"

#if AURORA_ENABLE_RENDER
#include "debug/debug.hpp"
#include "hints.hpp"
#include "shaders.hpp"
#include "../webgpu/gpu_prof.hpp"
#endif

#include <aurora/gfx.h>

#include <array>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render"};

// Recording thread only; a frame's stages are built in order and never concurrently
FrameCullState g_frameCull;
std::array<bool, StageCount> g_warnedMissingMarker{};
#if AURORA_ENABLE_RENDER
std::vector<FrameCullReason> g_cullReasons;
#endif

const CompiledGraph* stage_graph(const StageData& stage) noexcept {
  return stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
}

bool has_native_passes(const FrameState* frame) noexcept {
  return frame != nullptr && frame->graph != nullptr && frame->graph->hasNativePasses;
}

void warn_missing_marker(Stage stage) {
  if (std::exchange(g_warnedMissingMarker[static_cast<size_t>(stage)], true)) {
    return;
  }
  Log.warn("Render stage {} marker missing; passes that need it are skipped", stage_name(stage));
}

#if AURORA_ENABLE_RENDER
bool encode_timed(const wgpu::CommandEncoder& encoder, const StageData& stage, const CompiledGraph& graph,
                  NodeId nodeId) {
  const CompiledNode& node = graph.nodes[nodeId];
  // Compute and render passes carry their own pass timestamps
  if (!encoder || (node.passKind != PassKind::Copy && node.passKind != PassKind::Encoder)) {
    return encode_pass(encoder, stage, nodeId);
  }
  const webgpu::gpu_prof::Zone zone{encoder, node.name};
  return encode_pass(encoder, stage, nodeId);
}
#endif
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
#if AURORA_ENABLE_RENDER
    // FrameEnd publishes even without passes, so a frame whose passes all sit in other stages still pairs
    if (stage.stage == Stage::FrameEnd && stage.frame->inspectorOpen) {
      debug::record_stage_cull(*graph, stage.stage, stage.frameId, {});
      debug::publish_frame_cull(stage.frameId, graph->graphHash);
    }
#endif
    return false;
  }
  const auto passCount = static_cast<uint32_t>(program.passes.size());
  stage.passCount = passCount;
#if AURORA_ENABLE_RENDER
  stage.sceneHints = scene_hints();
  // Frames without a scene camera (logos, loads) legitimately send no scene markers
  const bool sceneFrame = stage.sceneHints->has(SceneHasMainCamera);
#else
  constexpr bool sceneFrame = false;
#endif
  stage.enabled.assign((passCount + 63) / 64, 0);
  const bool needsMarker = !stage.markerSeen && stage.stage <= Stage::OpaqueDone;
  for (uint32_t i = 0; i < passCount; ++i) {
    const CompiledNode& node = graph->nodes[program.passes[i]];
    if (needsMarker && !node.runsWithoutMarker) {
      if (sceneFrame) {
        warn_missing_marker(stage.stage);
      }
      continue;
    }
    if (node.enabled != nullptr && !node.enabled(stage, node.state)) {
      continue;
    }
    // A pass still compiling, or whose hook never got a pipeline, drops out like a disabled one for its consumers
    if (!pass_pipelines_ready(node)) {
      continue;
    }
    stage.enabled[i / 64] |= uint64_t{1} << (i % 64);
  }
#if AURORA_ENABLE_RENDER
  StageMask mask;
  if (stage.frame->inspectorOpen) {
    g_cullReasons.assign(passCount, FrameCullReason::Ran);
    cull_stage(*graph, stage.stage, stage.enabled, g_frameCull, mask, g_cullReasons);
    debug::record_stage_cull(*graph, stage.stage, stage.frameId, g_cullReasons);
    if (stage.stage == Stage::FrameEnd) {
      debug::publish_frame_cull(stage.frameId, graph->graphHash);
    }
  } else {
    cull_stage(*graph, stage.stage, stage.enabled, g_frameCull, mask);
  }
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
    if (encode_timed(encoder, stage, *graph, nodeId)) {
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
#if AURORA_ENABLE_RENDER
  // The stage that queued these has no frame packet, so this is its only after-submit point
  run_after_frame_callbacks();
#endif
  if (!has_native_passes(frame)) {
    return;
  }
}

bool pass_pipelines_ready(const CompiledNode& node) noexcept {
  if (node.pipelines != nullptr && node.pipelineIds[0] == 0) {
    return false;
  }
#if AURORA_ENABLE_RENDER
  return shaders::pipelines_ready(node.pipelineIds);
#else
  return true;
#endif
}

void reset_frame_cull() noexcept { g_frameCull = {}; }

namespace testing {
void reset_missing_marker_warnings() noexcept { g_warnedMissingMarker = {}; }
} // namespace testing

void invalidate_frame_histories() noexcept {
#if AURORA_ENABLE_RENDER
  invalidate_histories(g_frameCull);
#endif
}
} // namespace aurora::render::detail

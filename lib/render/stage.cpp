#include "stage.hpp"

#include "frame_state.hpp"
#include "graph.hpp"
#include "pass_context.hpp"
#include "../logging.hpp"

#include <aurora/gfx.h>

#include <array>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render"};

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
    stage.enabled[i / 64] |= uint64_t{1} << (i % 64);
  }
  return stage.any_enabled();
}

void run_stage(const wgpu::CommandEncoder& encoder, const StageData& stage) {
  const CompiledGraph* graph = stage_graph(stage);
  if (graph == nullptr) {
    return;
  }
  const StageProgram& program = graph->stages[static_cast<size_t>(stage.stage)];
  PassContext ctx{encoder, stage};
  for (uint32_t i = 0; i < stage.passCount; ++i) {
    if (!stage.is_enabled(i)) {
      continue;
    }
    const CompiledNode& node = graph->nodes[program.passes[i]];
    ctx.set_pass(node);
#ifdef AURORA_GFX_DEBUG_GROUPS
    if (encoder) {
      encoder.PushDebugGroup(node.name.c_str());
    }
#endif
    if (node.encode != nullptr) {
      node.encode(ctx, node.state);
    }
#ifdef AURORA_GFX_DEBUG_GROUPS
    if (encoder) {
      encoder.PopDebugGroup();
    }
#endif
  }
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
} // namespace aurora::render::detail

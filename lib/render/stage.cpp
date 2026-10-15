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
  const StageRange& range = graph->range(stage.stage);
  if (range.count == 0) {
    return false;
  }
  stage.passCount = range.count;
  stage.enabled.assign((range.count + 63) / 64, 0);
  const bool needsMarker = !stage.markerSeen && stage.stage <= Stage::OpaqueDone;
  for (uint32_t i = 0; i < range.count; ++i) {
    const PassDesc& desc = graph->passes[range.first + i].desc;
    if (needsMarker && !desc.runsWithoutMarker) {
      warn_missing_marker(stage.stage);
      continue;
    }
    if (desc.enabled != nullptr && !desc.enabled(stage, desc.state)) {
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
  const StageRange& range = graph->range(stage.stage);
  PassContext ctx{encoder, stage};
  for (uint32_t i = 0; i < stage.passCount; ++i) {
    if (!stage.is_enabled(i)) {
      continue;
    }
    const CompiledPass& pass = graph->passes[range.first + i];
    ctx.set_pass(pass);
#ifdef AURORA_GFX_DEBUG_GROUPS
    if (encoder) {
      encoder.PushDebugGroup(pass.desc.name.c_str());
    }
#endif
    pass.desc.encode(ctx, pass.desc.state);
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

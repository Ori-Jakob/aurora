#pragma once

#include "stage_data.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

namespace aurora::render::detail {
struct CompiledNode;

bool build_stage_data(StageData& stage);
void run_stage(const wgpu::CommandEncoder& encoder, const StageData& stage);
void before_submit(const FrameState* frame) noexcept;
void after_frame(const FrameState* frame) noexcept;
void reset_frame_cull() noexcept;
void invalidate_frame_histories() noexcept;
bool stage_samples_scene_depth(const StageData& stage) noexcept;
DepthSource stage_scene_depth_source(const StageData& stage) noexcept;
bool pass_pipelines_ready(const CompiledNode& node) noexcept;

namespace testing {
void reset_missing_marker_warnings() noexcept;
} // namespace testing
} // namespace aurora::render::detail

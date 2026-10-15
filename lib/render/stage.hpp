#pragma once

#include "stage_data.hpp"

#include <webgpu/webgpu_cpp.h>

namespace aurora::render::detail {
bool build_stage_data(StageData& stage);
void run_stage(const wgpu::CommandEncoder& encoder, const StageData& stage);
void before_submit(const FrameState* frame) noexcept;
void after_frame(const FrameState* frame) noexcept;
void reset_frame_cull() noexcept;
void invalidate_frame_histories() noexcept;
} // namespace aurora::render::detail

#pragma once

#include "stage_data.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <vector>

namespace aurora::gfx::detail {
struct NativeStep;
struct FrameSegment;
} // namespace aurora::gfx::detail

namespace aurora::render::detail {
struct CompiledNode;

enum class ComposeMode : uint8_t { Split, InPlace };
enum class ColorEncoding : uint8_t { Stored, Linear };

struct SceneComposeData {
  ComposeMode mode = ComposeMode::Split;
  float ratio[2] = {1.f, 1.f};
  float jitterPx[2] = {0.f, 0.f};
  float rcasStrength = 0.f;
  ColorEncoding outEncoding = ColorEncoding::Stored;
  uint32_t uniformOffset = 0;
  uint64_t frameId = 0;
};

bool build_stage_data(StageData& stage);
void run_stage(const wgpu::CommandEncoder& encoder, const StageData& stage);
void before_submit(const FrameState* frame) noexcept;
void after_frame(const FrameState* frame) noexcept;
void reset_frame_cull() noexcept;
void invalidate_frame_histories() noexcept;
bool stage_samples_scene_depth(const StageData& stage) noexcept;
DepthSource stage_scene_depth_source(const StageData& stage) noexcept;
bool pass_pipelines_ready(const CompiledNode& node) noexcept;
bool scene_compose_wanted() noexcept;
void prepare_scene_compose(SceneComposeData& data);
void scene_compose(wgpu::CommandEncoder& cmd, const SceneComposeData& data);
bool take_native_step(wgpu::CommandEncoder& cmd, const SceneComposeData& data, gfx::detail::NativeStep& step) noexcept;
void submit_segments(std::vector<gfx::detail::FrameSegment>& segments) noexcept;
void end_segments(std::vector<gfx::detail::FrameSegment>& segments) noexcept;

namespace testing {
void reset_missing_marker_warnings() noexcept;
uint64_t segmented_frames() noexcept;
} // namespace testing
} // namespace aurora::render::detail

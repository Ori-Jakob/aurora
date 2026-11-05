#pragma once

#include "frame_packet.hpp"

#include <span>

namespace aurora::gfx {

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass);
void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass);

namespace detail {
void encode_op(wgpu::CommandEncoder& encoder, FramePacket& frame, const FrameOp& op);
void release_compute_encoding() noexcept;
void reset_encoder_state() noexcept;
// Viewport, scissor and draw commands only; the list starts and ends with no encoder state carried over
void encode_command_list(const wgpu::RenderPassEncoder& pass, std::span<const Command> commands,
                         const RenderPass& passInfo);
// Draw encoders read only the scene colour size from their pass, so a replay needs nothing else
RenderPass replay_pass_info(wgpu::Extent3D extent);
namespace testing {
uint32_t compute_segments_encoded() noexcept;
WGPURenderPipeline current_pipeline() noexcept;
void set_current_pipeline(WGPURenderPipeline pipeline) noexcept;
// Runs where a replay's first command would, so a test sees the state that command inherits
void set_replay_hook(void (*hook)()) noexcept;
} // namespace testing
} // namespace detail

} // namespace aurora::gfx

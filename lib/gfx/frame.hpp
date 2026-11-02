#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"

#include <optional>

namespace aurora::gfx::detail {

inline constexpr size_t FrameSlotCount = 2;
inline constexpr size_t StagingBufferCount = FrameSlotCount + 3;

const wgpu::Buffer& staging_buffer(size_t slot);
bool spill_frame_stream(FramePacket& frame, FrameStream stream, uint64_t needed) noexcept;
bool grow_device_stream(FrameStream stream, uint64_t size) noexcept;
void refuse_frame_stream(FrameStream stream, uint64_t needed, uint64_t capacity) noexcept;
void grow_uniform_buffer(uint64_t size) noexcept;

void request_streaming_sizes(const StreamingSizes& minimum) noexcept;
StreamingSizes streaming_sizes() noexcept;
struct ChannelBudget {
  uint64_t bytes = 0;
  uint32_t alignment = 0;
};
ChannelBudget channel_budget() noexcept;
void request_channel_storage(uint64_t bytesPerFrame) noexcept;
void grow_streaming_storage_now(uint64_t storage) noexcept;
wgpu::BindGroupLayout compute_empty_layout() noexcept;
wgpu::BindGroup compute_empty_group() noexcept;
wgpu::BindGroupLayout compute_stream_layout() noexcept;
struct ComputeFailures {
  uint64_t binds = 0;
  uint64_t copies = 0;
};
ComputeFailures compute_failures() noexcept;
void set_compute_scratch(wgpu::Buffer buffer) noexcept;

namespace testing {
StreamingSizes requested_streaming_sizes() noexcept;
void reset_streaming_request(const StreamingSizes& sizes = {}) noexcept;
void create_compute_layouts() noexcept;
void release_compute_layouts() noexcept;
void set_staging_buffer(size_t slot, wgpu::Buffer buffer) noexcept;
void reset_stream_warnings() noexcept;
} // namespace testing

struct RegisteredDrawType {
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
};

struct RegisteredEncoderTaskType {
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
};

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id);
std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id);
void stop_render_worker();

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void initialize();
void shutdown();
bool begin_frame();
void end_frame(EndFrameCallback callback);
uint32_t current_frame() noexcept;
void after_submit() noexcept;
void before_submit() noexcept;
void after_frame() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx

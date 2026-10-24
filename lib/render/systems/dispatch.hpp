#pragma once

#include "../gpu_heap.hpp"
#include "../shaders.hpp"
#include "../../gfx/recording.hpp"
#include "../../gx/gpu_channel.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <span>

namespace aurora::render::systems {
struct KernelRef {
  shaders::PipelineId pipeline;
  gpu_heap::BlockHandle state = gpu_heap::BlockHandle::Invalid;
  uint32_t stateBytes = 0;
};
struct StateRef {
  gpu_heap::BlockHandle state = gpu_heap::BlockHandle::Invalid;
  uint32_t stateBytes = 0;
};

bool bind_kernel(const KernelRef& ref, const wgpu::ComputePassEncoder& pass);
gfx::detail::CopyDestination state_target(const StateRef& ref);
void initialize_dispatch();
void shutdown_dispatch() noexcept;
void tick_worker(uint64_t frameId);
bool decode_dispatch(void*, const gx::gpu_channel::DispatchDecode& desc) noexcept;
bool apply_update(void*, uint64_t id, uint32_t byteOffset, std::span<const uint8_t> bytes) noexcept;
} // namespace aurora::render::systems

#pragma once

#include "../gpu_heap.hpp"
#include "../shaders.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>

namespace aurora::render::systems {
inline constexpr uint64_t KernelLayoutKey = 0x6770757379737401;

struct KernelSlot {
  shaders::PipelineId pipeline;
  std::array<uint32_t, 3> workgroupSize{};
  gpu_heap::BlockHandle state = gpu_heap::BlockHandle::Invalid;
  uint32_t stateBytes = 0;
};

void initialize();
void shutdown();
wgpu::BindGroupLayout state_layout() noexcept;
wgpu::PipelineLayout kernel_layout() noexcept;
shaders::PipelineId kernel_pipeline(uint64_t id, uint32_t kernel) noexcept;
bool kernel_slot(uint64_t id, uint32_t kernel, KernelSlot& out) noexcept;
bool wait_ready(uint64_t id) noexcept;
} // namespace aurora::render::systems

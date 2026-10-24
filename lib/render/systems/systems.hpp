#pragma once

#include "../shaders.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>

namespace aurora::render::systems {
inline constexpr uint64_t KernelLayoutKey = 0x6770757379737401;

void initialize();
void shutdown();
wgpu::BindGroupLayout state_layout() noexcept;
wgpu::PipelineLayout kernel_layout() noexcept;
shaders::PipelineId kernel_pipeline(uint64_t id, uint32_t kernel) noexcept;
bool wait_ready(uint64_t id) noexcept;
} // namespace aurora::render::systems

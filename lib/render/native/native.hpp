#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::render::detail {
void native_worker_frame_begin() noexcept;
void shutdown_native() noexcept;
} // namespace aurora::render::detail

namespace aurora::render::detail::native {
bool interop_enabled() noexcept;
bool initialize_device(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept;
void initialize() noexcept;
void assert_worker() noexcept;
void reenable_upscalers() noexcept;
void encode_test_step(wgpu::CommandEncoder& cmd) noexcept;
} // namespace aurora::render::detail::native

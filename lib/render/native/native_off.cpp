#include "native.hpp"

namespace aurora::render::detail {
void native_worker_frame_begin() noexcept {}

void shutdown_native() noexcept {}
} // namespace aurora::render::detail

namespace aurora::render::detail::native {
bool interop_enabled() noexcept { return false; }

bool initialize_device(const wgpu::Device&, const wgpu::Adapter&) noexcept { return false; }

void initialize() noexcept {}

void assert_worker() noexcept {}

void reenable_upscalers() noexcept {}

void encode_test_step(wgpu::CommandEncoder&) noexcept {}
} // namespace aurora::render::detail::native

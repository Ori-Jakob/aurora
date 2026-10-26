#include "native.hpp"

namespace aurora::render::detail {
void native_worker_frame_begin() noexcept {}

void shutdown_native() noexcept {}
} // namespace aurora::render::detail

namespace aurora::render::detail::native {
bool initialize_device(const wgpu::Device&, const wgpu::Adapter&) noexcept { return false; }

void initialize() noexcept {}

void assert_worker() noexcept {}

void reenable_upscalers() noexcept {}
} // namespace aurora::render::detail::native

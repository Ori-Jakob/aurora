#include "native.hpp"

namespace aurora::render::detail {
void native_worker_frame_begin() noexcept {}

void shutdown_native() noexcept {}
} // namespace aurora::render::detail

namespace aurora::render::detail::native {
bool interop_enabled() noexcept { return false; }

bool initialize_device(const wgpu::Device&, const wgpu::Adapter&) noexcept { return false; }

void initialize() noexcept {}

void shutdown() noexcept {}

void assert_worker() noexcept {}

void reenable_upscalers() noexcept {}

upscale::native_upscale::NativeUpscaler* upscaler(Upscaler) noexcept { return nullptr; }

upscale::native_upscale::NativeUpscaler* test_upscaler(float) noexcept { return nullptr; }

bool prepare_upscaler(const upscale::native_upscale::Plan&, GraphExtent&) noexcept { return false; }

void release_upscaler() noexcept {}

bool begin_upscale(const upscale::native_upscale::Frame&, upscale::native_upscale::Targets&) noexcept { return false; }

std::string_view upscaler_path(Upscaler) noexcept { return {}; }

void set_app_identity(const AppIdentity&) noexcept {}

std::optional<Identity> app_identity() noexcept { return std::nullopt; }

namespace testing {
uint32_t sdk_calls() noexcept { return 0; }

void clear_app_identity() noexcept {}
} // namespace testing
} // namespace aurora::render::detail::native

#include "native.hpp"

#include "d3d12/device.hpp"
#include "d3d12/sentinel.hpp"
#include "d3d12/teardown.hpp"

#include "../../gfx/render_worker.hpp"
#include "../../internal.hpp"
#include "../../io.hpp"

#include <filesystem>

namespace aurora::render::detail {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::native"};
} // namespace

void native_worker_frame_begin() noexcept {}

void shutdown_native() noexcept {
  native::d3d12::run_teardowns();
  native::d3d12::sentinel::disarm(native::d3d12::sentinel::Session);
  native::d3d12::shutdown();
}

namespace native {
bool initialize_device(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept {
  return d3d12::initialize(device, adapter);
}

void initialize() noexcept {
  d3d12::sentinel::open(g_config.cachePath != nullptr ? io::fs_path_from_string(g_config.cachePath)
                                                      : std::filesystem::path{});
}

void assert_worker() noexcept {
  CHECK(gfx::render_worker::is_worker_thread(), "Native D3D12 call off the render worker");
}

void reenable_upscalers() noexcept { d3d12::sentinel::clear(); }
} // namespace native
} // namespace aurora::render::detail

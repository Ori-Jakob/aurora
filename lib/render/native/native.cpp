#include "native.hpp"

#include "d3d12/device.hpp"
#include "d3d12/probe.hpp"
#include "d3d12/sentinel.hpp"
#include "d3d12/teardown.hpp"
#include "d3d12/upscale_bridge.hpp"

#if AURORA_ENABLE_XESS
#include "d3d12/xess.hpp"
#endif

#include "../../gfx/render_worker.hpp"
#include "../../internal.hpp"
#include "../../io.hpp"

#include <filesystem>

namespace aurora::render::detail {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::native"};

native::d3d12::upscale_bridge::PassthroughUpscaler g_testUpscaler;
} // namespace

void native_worker_frame_begin() noexcept {
  native::d3d12::probe::run_pending();
  native::d3d12::upscale_bridge::worker_frame_begin();
}

void shutdown_native() noexcept {
  native::d3d12::run_teardowns();
  native::d3d12::sentinel::disarm(native::d3d12::sentinel::Session);
  native::d3d12::shutdown();
}

namespace native {
bool interop_enabled() noexcept { return !d3d12::killed(); }

bool initialize_device(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept {
  return d3d12::initialize(device, adapter);
}

void initialize() noexcept {
  const std::filesystem::path directory =
      g_config.cachePath != nullptr ? io::fs_path_from_string(g_config.cachePath) : std::filesystem::path{};
  d3d12::sentinel::open(directory);
#if AURORA_ENABLE_XESS
  d3d12::upscale_bridge::register_upscaler(d3d12::xess::upscaler());
#endif
  d3d12::probe::initialize(directory);
}

void shutdown() noexcept { d3d12::probe::shutdown(); }

void assert_worker() noexcept {
  CHECK(gfx::render_worker::is_worker_thread(), "Native D3D12 call off the render worker");
}

void reenable_upscalers() noexcept {
  d3d12::sentinel::clear();
  d3d12::probe::clear();
}

upscale::native_upscale::NativeUpscaler* upscaler(Upscaler option) noexcept {
  return d3d12::upscale_bridge::upscaler(option);
}

upscale::native_upscale::NativeUpscaler* test_upscaler(float ratio) noexcept {
  g_testUpscaler.set_ratio(ratio);
  return &g_testUpscaler;
}

bool prepare_upscaler(const upscale::native_upscale::Plan& plan, GraphExtent& render) noexcept {
  return d3d12::upscale_bridge::prepare(plan, render);
}

void release_upscaler() noexcept { d3d12::upscale_bridge::release(); }

bool begin_upscale(const upscale::native_upscale::Frame& frame, upscale::native_upscale::Targets& targets) noexcept {
  return d3d12::upscale_bridge::begin(frame, targets);
}

std::string_view upscaler_path(Upscaler option) noexcept { return d3d12::probe::path(option); }

namespace testing {
uint32_t sdk_calls() noexcept { return d3d12::probe::sdk_calls(); }
} // namespace testing
} // namespace native
} // namespace aurora::render::detail

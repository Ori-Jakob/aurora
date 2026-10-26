#pragma once

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail {
struct GraphExtent;
} // namespace aurora::render::detail

namespace aurora::render::detail::upscale::native_upscale {
class NativeUpscaler;
struct Plan;
struct Frame;
struct Targets;
} // namespace aurora::render::detail::upscale::native_upscale

namespace aurora::render::detail {
void native_worker_frame_begin() noexcept;
void shutdown_native() noexcept;
} // namespace aurora::render::detail

namespace aurora::render::detail::native {
bool interop_enabled() noexcept;
bool initialize_device(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept;
void initialize() noexcept;
void shutdown() noexcept;
void assert_worker() noexcept;
void reenable_upscalers() noexcept;

upscale::native_upscale::NativeUpscaler* upscaler(Upscaler option) noexcept;
upscale::native_upscale::NativeUpscaler* test_upscaler(float ratio) noexcept;
bool prepare_upscaler(const upscale::native_upscale::Plan& plan, GraphExtent& render) noexcept;
void release_upscaler() noexcept;
bool begin_upscale(const upscale::native_upscale::Frame& frame, upscale::native_upscale::Targets& targets) noexcept;
std::string_view upscaler_path(Upscaler option) noexcept;

namespace testing {
uint32_t sdk_calls() noexcept;
} // namespace testing
} // namespace aurora::render::detail::native

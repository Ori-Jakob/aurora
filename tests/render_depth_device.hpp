#pragma once

#include "render_test_device.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <span>

namespace aurora::render::detail {
struct SceneDepth;
} // namespace aurora::render::detail

namespace aurora::test {
[[nodiscard]] wgpu::Texture make_depth(TestDevice& device, uint32_t width, uint32_t height);
std::shared_ptr<const render::detail::SceneDepth> scene_depth(const wgpu::Texture& texture);
void fill_depth(TestDevice& device, const wgpu::Texture& texture, std::span<const float> values);
} // namespace aurora::test

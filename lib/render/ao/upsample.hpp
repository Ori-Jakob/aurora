#pragma once

#include "ao.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::ao::upsample {
inline constexpr std::string_view Pass = "ao.upsample";
inline constexpr std::string_view Shader = "ao/upsample";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RGBA8Unorm;
inline constexpr wgpu::TextureFormat SsilFormat = wgpu::TextureFormat::RGBA16Float;

constexpr TexDesc final_desc() noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = 1,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr TexDesc irradiance_desc() noexcept {
  return {
      .format = SsilFormat,
      .sizeClass = SizeClass::Efb,
      .divisor = 1,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

PassDesc pass_desc();

namespace testing {
struct Counters {
  uint32_t draws = 0;
};
Counters counters() noexcept;
void reset() noexcept;
shaders::RenderPipelineDesc pipeline_desc(bool ssil = false);
} // namespace testing
} // namespace aurora::render::detail::ao::upsample

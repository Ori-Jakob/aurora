#pragma once

#include "ao.hpp"

#include "../graph.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::ao::gtao {
inline constexpr std::string_view Pass = "ao.gtao";
inline constexpr std::string_view Resource = "ao.raw";
inline constexpr std::string_view SsilRaw = "ssil.raw";
inline constexpr wgpu::TextureFormat SsilFormat = wgpu::TextureFormat::RGBA16Float;
inline constexpr std::string_view Shader = "ao/gtao";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RGBA8Unorm;
inline constexpr uint32_t WorkgroupSize = 8;

constexpr TexDesc raw_desc(uint32_t divisor) noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
  };
}

constexpr TexDesc ssil_raw_desc(uint32_t divisor) noexcept {
  return {
      .format = SsilFormat,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
  };
}

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

PassDesc pass_desc();

namespace testing {
struct Counters {
  uint32_t dispatches = 0;
};
Counters counters() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::ao::gtao

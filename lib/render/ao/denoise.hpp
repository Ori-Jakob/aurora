#pragma once

#include "ao.hpp"

#include "../graph.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::ao::denoise {
inline constexpr std::string_view Pass = "ao.denoise";
inline constexpr std::string_view History = "ao.history";
inline constexpr std::string_view SsilHistory = "ssil.history";
inline constexpr std::string_view Shader = "ao/denoise";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RGBA16Float;
inline constexpr uint32_t WorkgroupSize = 8;
inline constexpr uint64_t RetainFrames = 8;

constexpr TexDesc history_desc(uint32_t divisor) noexcept {
  return {
      .format = Format,
      .sizeClass = SizeClass::Efb,
      .divisor = static_cast<uint16_t>(divisor),
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr TexDesc ssil_history_desc(uint32_t divisor) noexcept { return history_desc(divisor); }

constexpr uint32_t groups(uint32_t size) noexcept { return (size + WorkgroupSize - 1) / WorkgroupSize; }

PassDesc pass_desc();
void worker_tick(uint64_t frameId) noexcept;
void shutdown() noexcept;

namespace testing {
struct Counters {
  uint32_t dispatches = 0;
  uint32_t temporalValid = 0;
  uint32_t temporalResets = 0;
  uint32_t groupsCreated = 0;
  uint32_t ssilResets = 0;
};
Counters counters() noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::ao::denoise

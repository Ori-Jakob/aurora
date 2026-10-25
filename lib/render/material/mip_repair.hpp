#pragma once

#include "../../internal.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>

namespace aurora::render::material {
struct MipTailRequest {
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  // the last standard level, kept verbatim by the caller
  uint32_t width = 0, height = 0;
  ArrayRef<uint8_t> level;
};

// covers the GZ2 maximum of 7,225,344
inline constexpr uint64_t MaxRepairSourceTexels = 1ull << 23;

bool can_repair(wgpu::TextureFormat format) noexcept;
// Every level below width x height down to 1x1, concatenated in upload order; nullopt on any failure.
std::optional<ByteBuffer> generate_mip_tail(const MipTailRequest& request) noexcept;
std::optional<ByteBuffer> decode_level(wgpu::TextureFormat format, uint32_t width, uint32_t height,
                                       ArrayRef<uint8_t> level) noexcept;
} // namespace aurora::render::material

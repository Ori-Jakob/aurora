#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace aurora::gxreplay {
class EfbReadback {
public:
  void encode(const wgpu::CommandEncoder& encoder);
  bool map(std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height, std::string& error);

private:
  wgpu::Buffer m_buffer;
  wgpu::TextureFormat m_format = wgpu::TextureFormat::Undefined;
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  uint32_t m_bytesPerRow = 0;
  bool m_pending = false;
};

uint64_t hash_rgba(std::span<const uint8_t> rgba) noexcept;
} // namespace aurora::gxreplay

#include "readback.hpp"

#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <fmt/format.h>
#include <xxhash.h>

#include <cstring>
#include <string_view>

namespace aurora::gxreplay {
void EfbReadback::encode(const wgpu::CommandEncoder& encoder) {
  const auto& source = webgpu::present_source();
  const uint32_t width = source.size.width;
  const uint32_t height = source.size.height;
  if (!m_buffer || width != m_width || height != m_height || source.format != m_format) {
    m_width = width;
    m_height = height;
    m_format = source.format;
    m_bytesPerRow = AURORA_ALIGN(width * 4, 256);
    const wgpu::BufferDescriptor descriptor{
        .label = "gxreplay EFB readback",
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
        .size = static_cast<uint64_t>(m_bytesPerRow) * height,
    };
    m_buffer = webgpu::g_device.CreateBuffer(&descriptor);
  }

  const wgpu::TexelCopyTextureInfo src{
      .texture = source.texture,
      .mipLevel = 0,
  };
  const wgpu::TexelCopyBufferInfo dst{
      .layout =
          {
              .offset = 0,
              .bytesPerRow = m_bytesPerRow,
              .rowsPerImage = height,
          },
      .buffer = m_buffer,
  };
  const wgpu::Extent3D size{width, height, 1};
  encoder.CopyTextureToBuffer(&src, &dst, &size);
  m_pending = true;
}

bool EfbReadback::map(std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height, std::string& error) {
  if (!m_pending) {
    error = "no EFB readback was recorded for this frame";
    return false;
  }
  m_pending = false;
  if (m_format != wgpu::TextureFormat::RGBA8Unorm && m_format != wgpu::TextureFormat::BGRA8Unorm) {
    error = fmt::format("cannot read back EFB format {}", static_cast<uint32_t>(m_format));
    return false;
  }

  const size_t bufferSize = static_cast<size_t>(m_bytesPerRow) * m_height;
  wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
  std::string message;
  const auto future = m_buffer.MapAsync(wgpu::MapMode::Read, 0, bufferSize, wgpu::CallbackMode::WaitAnyOnly,
                                        [&](wgpu::MapAsyncStatus mapStatus, wgpu::StringView mapMessage) {
                                          status = mapStatus;
                                          message = std::string_view{mapMessage};
                                        });
  const auto waitStatus = webgpu::g_instance.WaitAny(future, UINT64_MAX);
  if (waitStatus != wgpu::WaitStatus::Success || status != wgpu::MapAsyncStatus::Success) {
    error = fmt::format("EFB readback failed to map: {}", message);
    return false;
  }

  const auto* mapped = static_cast<const uint8_t*>(m_buffer.GetConstMappedRange(0, bufferSize));
  width = m_width;
  height = m_height;
  const size_t rowBytes = static_cast<size_t>(m_width) * 4;
  rgba.resize(rowBytes * m_height);
  for (uint32_t y = 0; y < m_height; ++y) {
    const uint8_t* srcRow = mapped + static_cast<size_t>(y) * m_bytesPerRow;
    uint8_t* dstRow = rgba.data() + y * rowBytes;
    if (m_format == wgpu::TextureFormat::RGBA8Unorm) {
      std::memcpy(dstRow, srcRow, rowBytes);
      continue;
    }
    for (uint32_t x = 0; x < m_width; ++x) {
      dstRow[x * 4 + 0] = srcRow[x * 4 + 2];
      dstRow[x * 4 + 1] = srcRow[x * 4 + 1];
      dstRow[x * 4 + 2] = srcRow[x * 4 + 0];
      dstRow[x * 4 + 3] = srcRow[x * 4 + 3];
    }
  }
  m_buffer.Unmap();
  return true;
}

uint64_t hash_rgba(std::span<const uint8_t> rgba) noexcept { return XXH3_64bits(rgba.data(), rgba.size()); }
} // namespace aurora::gxreplay

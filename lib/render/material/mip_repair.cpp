#include "mip_repair.hpp"

#include "../../gfx/texture.hpp"

#include <tracy/Tracy.hpp>

#define BCDEC_IMPLEMENTATION
#include <bcdec.h>
#include <bc7enc.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::render::material {
namespace {
enum class Codec : uint8_t { None, Rgba8, Bc7 };

constexpr uint32_t Bc7BlockBytes = 16;

Codec codec_of(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::RGBA8UnormSrgb:
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::BGRA8UnormSrgb:
    return Codec::Rgba8;
  case wgpu::TextureFormat::BC7RGBAUnorm:
  case wgpu::TextureFormat::BC7RGBAUnormSrgb:
    return Codec::Bc7;
  default:
    return Codec::None;
  }
}

uint64_t level_bytes(Codec codec, uint32_t width, uint32_t height) noexcept {
  if (codec == Codec::Bc7) {
    return uint64_t{(width + 3) / 4} * ((height + 3) / 4) * Bc7BlockBytes;
  }
  return uint64_t{width} * height * 4;
}

struct Taps {
  uint32_t first;
  uint32_t count;
};

Taps source_taps(uint32_t size, uint32_t outSize, uint32_t index) noexcept {
  if (size == 1) {
    return {0, 1};
  }
  return {2 * index, (size & 1) != 0 && index == outSize - 1 ? 3u : 2u};
}

void halve(const std::vector<uint16_t>& src, uint32_t width, uint32_t height, std::vector<uint16_t>& dst,
           uint32_t outWidth, uint32_t outHeight) {
  dst.resize(size_t{outWidth} * outHeight * 4);
  for (uint32_t y = 0; y < outHeight; ++y) {
    const Taps ty = source_taps(height, outHeight, y);
    for (uint32_t x = 0; x < outWidth; ++x) {
      const Taps tx = source_taps(width, outWidth, x);
      const uint32_t count = tx.count * ty.count;
      std::array<uint32_t, 4> sum{};
      for (uint32_t sy = ty.first; sy < ty.first + ty.count; ++sy) {
        const uint16_t* row = src.data() + (size_t{sy} * width + tx.first) * 4;
        for (uint32_t sx = 0; sx < tx.count; ++sx) {
          for (uint32_t c = 0; c < 4; ++c) {
            sum[c] += row[sx * 4 + c];
          }
        }
      }
      uint16_t* out = dst.data() + (size_t{y} * outWidth + x) * 4;
      for (uint32_t c = 0; c < 4; ++c) {
        out[c] = static_cast<uint16_t>((sum[c] + count / 2) / count);
      }
    }
  }
}

void to_u8(const std::vector<uint16_t>& src, uint8_t* dst) noexcept {
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = static_cast<uint8_t>(std::min((src[i] + 128u) >> 8, 255u));
  }
}

std::once_flag s_bc7Init;

void encode_bc7(const uint8_t* rgba, uint32_t width, uint32_t height, uint8_t* dst) noexcept {
  std::call_once(s_bc7Init, [] { bc7enc_compress_block_init(); });
  bc7enc_compress_block_params params;
  params.clear();
  bc7enc_compress_block_params_init(&params);
  // Halves the GZ2 sweep's encode time for about 5% more error on distant levels
  params.m_max_partitions = 8;
  params.m_try_least_squares = false;
  std::array<uint8_t, 64> block{};
  for (uint32_t by = 0; by < height; by += 4) {
    for (uint32_t bx = 0; bx < width; bx += 4) {
      for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t x = std::min(bx + i % 4, width - 1);
        const uint32_t y = std::min(by + i / 4, height - 1);
        std::memcpy(block.data() + i * 4, rgba + (size_t{y} * width + x) * 4, 4);
      }
      bc7enc_compress_block(dst, block.data(), &params);
      dst += Bc7BlockBytes;
    }
  }
}
} // namespace

bool can_repair(wgpu::TextureFormat format) noexcept { return codec_of(format) != Codec::None; }

std::optional<ByteBuffer> decode_level(wgpu::TextureFormat format, uint32_t width, uint32_t height,
                                       ArrayRef<uint8_t> level) noexcept {
  const Codec codec = codec_of(format);
  if (codec == Codec::None || width == 0 || height == 0 || level.size() != level_bytes(codec, width, height)) {
    return std::nullopt;
  }
  ByteBuffer out{size_t{width} * height * 4};
  if (codec == Codec::Rgba8) {
    std::memcpy(out.data(), level.data(), level.size());
    return out;
  }
  std::array<uint8_t, 64> block{};
  const uint8_t* src = level.data();
  for (uint32_t by = 0; by < height; by += 4) {
    for (uint32_t bx = 0; bx < width; bx += 4) {
      bcdec_bc7(src, block.data(), 16);
      src += Bc7BlockBytes;
      const uint32_t rows = std::min(4u, height - by);
      const uint32_t columns = std::min(4u, width - bx);
      for (uint32_t row = 0; row < rows; ++row) {
        std::memcpy(out.data() + ((size_t{by} + row) * width + bx) * 4, block.data() + row * 16, columns * 4);
      }
    }
  }
  return out;
}

std::optional<ByteBuffer> generate_mip_tail(const MipTailRequest& request) noexcept {
  ZoneScopedN("texture mip repair");
  const Codec codec = codec_of(request.format);
  const uint64_t texels = uint64_t{request.width} * request.height;
  if (codec == Codec::None || texels <= 1 || texels > MaxRepairSourceTexels) {
    return std::nullopt;
  }
  const auto decoded = decode_level(request.format, request.width, request.height, request.level);
  if (!decoded) {
    return std::nullopt;
  }

  uint32_t levels = 1;
  uint64_t tailBytes = 0;
  for (uint32_t w = request.width, h = request.height; w > 1 || h > 1; ++levels) {
    w = std::max(w >> 1, 1u);
    h = std::max(h >> 1, 1u);
    tailBytes += level_bytes(codec, w, h);
  }
  const uint64_t expected = gfx::calc_texture_size(request.format, request.width, request.height, levels) -
                            gfx::calc_texture_size(request.format, request.width, request.height, 1);
  if (tailBytes != expected) {
    return std::nullopt;
  }

  ByteBuffer tail{static_cast<size_t>(tailBytes)};
  std::vector<uint16_t> current(decoded->size());
  for (size_t i = 0; i < current.size(); ++i) {
    current[i] = static_cast<uint16_t>(decoded->data()[i] << 8);
  }
  std::vector<uint16_t> next;
  std::vector<uint8_t> rgba;
  uint8_t* dst = tail.data();
  uint32_t width = request.width;
  uint32_t height = request.height;
  while (width > 1 || height > 1) {
    const uint32_t outWidth = std::max(width >> 1, 1u);
    const uint32_t outHeight = std::max(height >> 1, 1u);
    halve(current, width, height, next, outWidth, outHeight);
    if (codec == Codec::Rgba8) {
      to_u8(next, dst);
    } else {
      rgba.resize(next.size());
      to_u8(next, rgba.data());
      encode_bc7(rgba.data(), outWidth, outHeight, dst);
    }
    dst += level_bytes(codec, outWidth, outHeight);
    current.swap(next);
    width = outWidth;
    height = outHeight;
  }
  return tail;
}
} // namespace aurora::render::material

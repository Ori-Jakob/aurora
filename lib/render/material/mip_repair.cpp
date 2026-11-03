#include "mip_repair.hpp"

#include "../../gfx/texture.hpp"
#include "../../gfx/texture_convert.hpp"

#include <tracy/Tracy.hpp>

#define BCDEC_IMPLEMENTATION
#include <bcdec.h>
#include <bc7enc.h>

#include <algorithm>
#include <array>
#include <cmath>
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

bool is_bgra(wgpu::TextureFormat format) noexcept {
  return format == wgpu::TextureFormat::BGRA8Unorm || format == wgpu::TextureFormat::BGRA8UnormSrgb;
}

uint64_t level_bytes(Codec codec, uint32_t width, uint32_t height) noexcept {
  if (codec == Codec::Bc7) {
    return uint64_t{(width + 3) / 4} * ((height + 3) / 4) * Bc7BlockBytes;
  }
  return uint64_t{width} * height * 4;
}

std::once_flag s_bc7Init;

void encode_bc7(const uint8_t* rgba, uint32_t width, uint32_t height, uint8_t* dst, bool perceptual) noexcept {
  std::call_once(s_bc7Init, [] { bc7enc_compress_block_init(); });
  bc7enc_compress_block_params params;
  params.clear();
  bc7enc_compress_block_params_init(&params);
  if (!perceptual) {
    bc7enc_compress_block_params_init_linear_weights(&params);
  }
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

struct NormalCodec {
  uint32_t x = 0, y = 0;
  bool mat = false;

  float decode(uint8_t code) const noexcept {
    return mat ? (static_cast<float>(code) - 128.f) / 127.f : static_cast<float>(code) / 127.5f - 1.f;
  }
  uint8_t encode(float v) const noexcept {
    return static_cast<uint8_t>(std::clamp(std::lround(mat ? 128.f + v * 127.f : (v + 1.f) * 127.5f), 0l, 255l));
  }
};

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

using NormalTexel = std::array<float, 5>;

ByteBuffer normal_mip_tail(uint32_t width, uint32_t height, ArrayRef<uint8_t> level, NormalCodec codec) noexcept {
  if (width == 0 || height == 0 || (width == 1 && height == 1) || level.size() != size_t{width} * height * 4) {
    return {};
  }
  std::array<uint32_t, 2> rest{};
  for (uint32_t c = 0, i = 0; c < 4; ++c) {
    if (c != codec.x && c != codec.y) {
      rest[i++] = c;
    }
  }

  std::vector<NormalTexel> current(size_t{width} * height);
  for (size_t i = 0; i < current.size(); ++i) {
    const uint8_t* texel = level.data() + i * 4;
    const float x = codec.decode(texel[codec.x]);
    const float y = codec.decode(texel[codec.y]);
    const float z = std::sqrt(std::max(0.f, 1.f - x * x - y * y));
    const float length = std::sqrt(x * x + y * y + z * z);
    current[i] = {x / length, y / length, z / length, static_cast<float>(texel[rest[0]]),
                  static_cast<float>(texel[rest[1]])};
  }

  size_t tailBytes = 0;
  for (uint32_t w = width, h = height; w > 1 || h > 1;) {
    w = std::max(w >> 1, 1u);
    h = std::max(h >> 1, 1u);
    tailBytes += size_t{w} * h * 4;
  }
  ByteBuffer tail{tailBytes};
  uint8_t* dst = tail.data();
  std::vector<NormalTexel> next;
  while (width > 1 || height > 1) {
    const uint32_t outWidth = std::max(width >> 1, 1u);
    const uint32_t outHeight = std::max(height >> 1, 1u);
    next.assign(size_t{outWidth} * outHeight, NormalTexel{});
    for (uint32_t y = 0; y < outHeight; ++y) {
      const Taps ty = source_taps(height, outHeight, y);
      for (uint32_t x = 0; x < outWidth; ++x) {
        const Taps tx = source_taps(width, outWidth, x);
        NormalTexel sum{};
        for (uint32_t sy = ty.first; sy < ty.first + ty.count; ++sy) {
          for (uint32_t sx = tx.first; sx < tx.first + tx.count; ++sx) {
            const NormalTexel& texel = current[size_t{sy} * width + sx];
            for (size_t c = 0; c < sum.size(); ++c) {
              sum[c] += texel[c];
            }
          }
        }
        const float length = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
        const float count = static_cast<float>(tx.count * ty.count);
        NormalTexel& out = next[size_t{y} * outWidth + x];
        // Opposed normals cancel, and flat is the only direction that favours neither
        out = length > 1e-6f ? NormalTexel{sum[0] / length, sum[1] / length, sum[2] / length, 0.f, 0.f}
                             : NormalTexel{0.f, 0.f, 1.f, 0.f, 0.f};
        out[3] = sum[3] / count;
        out[4] = sum[4] / count;
        uint8_t* texel = dst + (size_t{y} * outWidth + x) * 4;
        texel[codec.x] = codec.encode(out[0]);
        texel[codec.y] = codec.encode(out[1]);
        texel[rest[0]] = static_cast<uint8_t>(std::clamp(std::lround(out[3]), 0l, 255l));
        texel[rest[1]] = static_cast<uint8_t>(std::clamp(std::lround(out[4]), 0l, 255l));
      }
    }
    dst += next.size() * 4;
    current.swap(next);
    width = outWidth;
    height = outHeight;
  }
  return tail;
}

ByteBuffer rgba_mip_tail(const MipTailRequest& request, ArrayRef<uint8_t> level) noexcept {
  switch (request.filter) {
  case MipFilter::MatNormal:
    return normal_mip_tail(request.width, request.height, level, {.x = 3, .y = 1, .mat = true});
  case MipFilter::NormalRg:
    return normal_mip_tail(request.width, request.height, level, {.x = is_bgra(request.format) ? 2u : 0u, .y = 1});
  case MipFilter::Colour:
  case MipFilter::Linear:
    break;
  }
  return gfx::box_mip_tail_rgba8(request.width, request.height, level);
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

  if (codec == Codec::Rgba8) {
    auto tail = rgba_mip_tail(request, request.level);
    if (tail.size() != tailBytes) {
      return std::nullopt;
    }
    return tail;
  }
  const auto decoded = decode_level(request.format, request.width, request.height, request.level);
  if (!decoded) {
    return std::nullopt;
  }
  const ByteBuffer rgba = rgba_mip_tail(request, *decoded);
  if (rgba.empty()) {
    return std::nullopt;
  }
  // Perceptual weights score luma and chroma, which only colour data has
  const bool perceptual = request.filter == MipFilter::Colour;
  ByteBuffer tail{static_cast<size_t>(tailBytes)};
  const uint8_t* src = rgba.data();
  uint8_t* dst = tail.data();
  uint32_t width = request.width;
  uint32_t height = request.height;
  while (width > 1 || height > 1) {
    width = std::max(width >> 1, 1u);
    height = std::max(height >> 1, 1u);
    encode_bc7(src, width, height, dst, perceptual);
    src += size_t{width} * height * 4;
    dst += level_bytes(codec, width, height);
  }
  return tail;
}
} // namespace aurora::render::material

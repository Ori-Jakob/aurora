#pragma once

#include "types.hpp"

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
inline constexpr uint64_t UniformBufferSize = 25165824; // 24 MiB
inline constexpr uint64_t VertexBufferSize = 5242880;   // 5 MiB
inline constexpr uint64_t IndexBufferSize = 2097152;    // 2 MiB
inline constexpr uint64_t StorageBufferSize = 8388608;  // 8 MiB
inline constexpr uint64_t TextureUploadSize = 25165824; // 24 MiB

namespace detail {
struct StreamingSizes {
  uint64_t vertex = VertexBufferSize;
  uint64_t index = IndexBufferSize;
  uint64_t storage = StorageBufferSize;

  bool operator==(const StreamingSizes&) const = default;
};

struct StagingLayout {
  uint64_t vertex = 0;
  uint64_t uniform = 0;
  uint64_t index = 0;
  uint64_t storage = 0;
  uint64_t textureUpload = 0;
  uint64_t total = 0;
};

constexpr StagingLayout staging_layout(const StreamingSizes& sizes) noexcept {
  StagingLayout layout{};
  layout.uniform = layout.vertex + sizes.vertex;
  layout.index = layout.uniform + UniformBufferSize;
  layout.storage = layout.index + sizes.index;
  layout.textureUpload = layout.storage + sizes.storage;
  layout.total = layout.textureUpload + (UseTextureBuffer ? TextureUploadSize : 0);
  return layout;
}

static_assert(staging_layout({}).uniform == VertexBufferSize);
static_assert(staging_layout({}).index == VertexBufferSize + UniformBufferSize);
static_assert(staging_layout({}).storage == VertexBufferSize + UniformBufferSize + IndexBufferSize);
static_assert(staging_layout({}).textureUpload ==
              VertexBufferSize + UniformBufferSize + IndexBufferSize + StorageBufferSize);
static_assert(staging_layout({}).total == UniformBufferSize + VertexBufferSize + IndexBufferSize + StorageBufferSize +
                                              (UseTextureBuffer ? TextureUploadSize : 0));

struct StreamingLimits {
  uint64_t vertex = 0;
  uint64_t index = 0;
  uint64_t storage = 0;
  uint64_t staging = 0;
};

struct StreamingGrowth {
  StreamingSizes sizes;
  bool stagingCapped = false;
};

StreamingGrowth next_streaming_sizes(const StreamingSizes& current, const AuroraStats& lastFrame,
                                     const StreamingSizes& requested, const StreamingLimits& limits) noexcept;

struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::Limits limits;
  AuroraStats stats{};
  StreamingSizes deviceSizes;
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx

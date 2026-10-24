#pragma once

#include "types.hpp"

#include <algorithm>
#include <bit>

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

inline constexpr uint64_t StreamingGranule = 1024 * 1024;

constexpr uint64_t round_streaming_request(uint64_t bytes) noexcept {
  return bytes == 0 ? 0 : std::bit_ceil(std::max(bytes, StreamingGranule));
}

constexpr uint64_t round_streaming_limit(uint64_t bytes) noexcept { return bytes & ~(StreamingGranule - 1); }

// Twice last frame's legacy covers the 4/3 that the growth rule alone guarantees
constexpr uint64_t legacy_storage_reserve(uint64_t storage, uint64_t legacy, bool history) noexcept {
  if (!history) {
    return storage / 2;
  }
  return std::min(storage, std::max(storage / 4, AURORA_ALIGN(2 * legacy, StreamingGranule)));
}

constexpr uint64_t channel_storage_budget(uint64_t storage, uint64_t legacy, bool history, uint64_t cap) noexcept {
  return std::min(storage - legacy_storage_reserve(storage, legacy, history), cap);
}

constexpr uint64_t channel_storage_request(uint64_t demand, uint64_t legacy) noexcept {
  return 2 * demand + AURORA_ALIGN(2 * legacy, StreamingGranule);
}

static_assert(round_streaming_request(StorageBufferSize) == StorageBufferSize);
static_assert(round_streaming_request(32 * StreamingGranule) == 32 * StreamingGranule);
static_assert(round_streaming_request(VertexBufferSize) % StreamingGranule == 0);

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
uint32_t next_resident_capacity(uint32_t current, uint64_t request, uint64_t streamingStorage, uint64_t bound) noexcept;

struct ResidentLayout {
  uint64_t streamingStorage = 0;
  uint32_t capacity = 0;
  uint32_t epoch = 0;
};

ResidentLayout next_resident_layout(const ResidentLayout& last, uint64_t streamingStorage, uint32_t capacity) noexcept;
uint64_t streaming_storage_limit(uint64_t bound, uint32_t residentCapacity) noexcept;

struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::BindGroupLayout computeEmptyLayout;
  wgpu::BindGroup computeEmptyGroup;
  wgpu::BindGroupLayout computeStreamLayout;
  wgpu::Limits limits;
  AuroraStats stats{};
  StreamingSizes deviceSizes;
  uint32_t residentCapacity = 0;
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx

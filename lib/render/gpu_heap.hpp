#pragma once

#include "tlsf.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>

namespace aurora::render::gpu_heap {

enum class PoolId : uint32_t { Invalid = 0 };
inline constexpr uint32_t MaxPools = 64;

enum class Backing : uint8_t {
  Dawn,
  NativeShared,
};

struct PoolDesc {
  const char* name = nullptr;
  wgpu::BufferUsage usage = wgpu::BufferUsage::None;
  uint64_t initialSize = 0;
  uint64_t maxSize = 0;
  uint32_t granularity = 256;
  bool gpuWrittenOnly = false;
  Backing backing = Backing::Dawn;
};

struct Allocation {
  PoolId pool = PoolId::Invalid;
  Tlsf::BlockId block = Tlsf::InvalidBlock;
  uint64_t offset = 0;
  uint64_t size = 0;

  explicit operator bool() const noexcept { return pool != PoolId::Invalid; }
};

struct PoolBinding {
  wgpu::Buffer buffer;
  uint64_t size = 0;
  uint32_t generation = 0;
};

struct PoolStats {
  uint64_t capacity = 0;
  uint64_t used = 0;
  uint64_t largestFree = 0;
  uint64_t pendingFreeBytes = 0;
  uint64_t pendingCapacity = 0;
  uint32_t allocations = 0;
  uint32_t generation = 0;
  uint32_t growths = 0;
  uint32_t rejectedUploads = 0;
  bool atLimit = false;
};

PoolId create_pool(const PoolDesc& desc);
void destroy_pool(PoolId pool);
Allocation allocate(PoolId pool, uint64_t size, uint64_t alignment = 0);
void reserve(PoolId pool, uint64_t capacity);
void free(const Allocation& allocation);
void upload(const Allocation& allocation, uint64_t offset, const void* data, uint64_t size);
PoolBinding binding(PoolId pool);
PoolStats stats(PoolId pool) noexcept;
uint64_t total_bytes() noexcept;
uint32_t growth_submits() noexcept;
void shutdown();

enum class Region : uint8_t { DrawArgs, DispatchArgs, Index32 };
inline constexpr uint32_t RegionCount = 3;
inline constexpr uint32_t RegionGranularity = 256;
inline constexpr uint32_t MaxBlocks = 1024;
inline constexpr uint64_t AllRequests = UINT64_MAX;
enum class BlockHandle : uint32_t { Invalid = 0 };
enum class BlockState : uint8_t { Invalid, Pending, Ready, Failed };

struct BlockView {
  wgpu::Buffer buffer;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint32_t generation = 0;

  explicit operator bool() const noexcept { return generation != 0; }
};

struct RegionStats {
  PoolStats pool;
  bool created = false;
  uint32_t blocks = 0;
  uint32_t pending = 0;
  uint32_t failed = 0;
  uint32_t releases = 0;
  uint64_t clearedBytes = 0;
};

void set_recording_frame(uint64_t frameId) noexcept;
BlockHandle create_block(Region region, uint64_t size);
void destroy_block(BlockHandle block);
BlockState block_state(BlockHandle block) noexcept;
BlockView block_view(BlockHandle block);
RegionStats region_stats(Region region) noexcept;
uint32_t boundary_submits() noexcept;
void begin_frame_worker(uint64_t frameId = AllRequests);

} // namespace aurora::render::gpu_heap

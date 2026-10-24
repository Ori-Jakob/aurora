#include "gpu_heap.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::render::gpu_heap {
namespace {
constexpr Module Log{"aurora::render"};
constexpr double MiB = 1024.0 * 1024.0;
constexpr uint32_t SlotBits = 8;
constexpr uint32_t SlotMask = (1u << SlotBits) - 1;
constexpr uint32_t MaxSlotGeneration = UINT32_MAX >> SlotBits;
static_assert(MaxPools <= SlotMask);

enum class SlotState : uint8_t { Free, Live, Destroying };

struct PublishedStats {
  std::atomic_uint64_t capacity{0};
  std::atomic_uint64_t used{0};
  std::atomic_uint64_t largestFree{0};
  std::atomic_uint64_t pendingFreeBytes{0};
  std::atomic_uint64_t pendingCapacity{0};
  std::atomic_uint32_t allocations{0};
  std::atomic_uint32_t generation{0};
  std::atomic_uint32_t growths{0};
  std::atomic_uint32_t rejectedUploads{0};
  std::atomic_bool atLimit{false};
};

struct PoolSlot {
  std::atomic<SlotState> state{SlotState::Free};
  std::atomic_uint32_t id{0};
  // Changed under g_mutex each time the slot is freed, so a PoolId is never handed out twice
  uint32_t slotGeneration = 1;
  // Written by create_pool before the slot is published, then owned by the owner thread
  std::string name;
  wgpu::BufferUsage usage = wgpu::BufferUsage::None;
  uint64_t maxSize = 0;
  bool gpuWrittenOnly = false;
  std::unique_ptr<Tlsf> tlsf;
  wgpu::Buffer buffer;
  std::vector<Tlsf::BlockId> pendingFrees;
  uint64_t pendingFreeBytes = 0;
  uint64_t pendingCapacity = 0;
  uint32_t generation = 0;
  uint32_t growths = 0;
  uint32_t rejectedUploads = 0;
  bool atLimit = false;
  bool loggedLimit = false;
  PublishedStats published;
};

std::array<PoolSlot, MaxPools> g_slots;
std::mutex g_mutex;
std::atomic_uint32_t g_activePools{0};
std::atomic_uint32_t g_growthSubmits{0};
std::atomic<std::thread::id> g_owner;

struct RegionInfo {
  const char* name;
  wgpu::BufferUsage usage;
  uint64_t initialSize;
};

constexpr std::array<RegionInfo, RegionCount> Regions{{
    {"heap.draw_args", wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage, 64 * 1024},
    {"heap.dispatch_args", wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage, 16 * 1024},
    {"heap.idx32", wgpu::BufferUsage::Index | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Storage, 1024 * 1024},
    {"heap.system_state", wgpu::BufferUsage::Storage, 1024 * 1024},
}};

constexpr uint32_t BlockIndexBits = 12;
constexpr uint32_t BlockIndexMask = (1u << BlockIndexBits) - 1;
constexpr uint32_t MaxBlockGeneration = UINT32_MAX >> BlockIndexBits;
static_assert(MaxBlocks < 1u << BlockIndexBits);

struct BlockSlot {
  std::atomic<BlockState> state{BlockState::Invalid};
  std::atomic_uint32_t id{0};
  uint32_t slotGeneration = 1;
  Region region = Region::DrawArgs;
  uint64_t size = 0;
  bool destroyQueued = false;
  Tlsf::BlockId tlsfBlock = Tlsf::InvalidBlock;
  uint64_t offset = 0;
  uint32_t generation = 0;
};

struct Request {
  enum Kind : uint8_t { Create, Destroy } kind;
  uint32_t handle;
  uint64_t stamp;
};

struct RegionCounters {
  std::atomic_uint64_t clearedBytes{0};
  std::atomic_uint32_t releases{0};
  bool loggedCreateFailure = false;
  bool loggedLimit = false;
};

std::array<BlockSlot, MaxBlocks> g_blocks;
std::mutex g_blockMutex;
std::vector<Request> g_requests;
std::vector<uint32_t> g_retries;
std::atomic_bool g_blockWork{false};
std::atomic_uint64_t g_recordingFrame{0};
std::atomic<std::thread::id> g_workerThread;
std::atomic_uint64_t g_workerFrame{0};
std::array<std::atomic<PoolId>, RegionCount> g_regions{};
std::array<RegionCounters, RegionCount> g_regionCounters;
std::atomic_uint32_t g_boundarySubmits{0};

void check_owner() noexcept {
  const auto self = std::this_thread::get_id();
  auto owner = std::thread::id{};
  if (!g_owner.compare_exchange_strong(owner, self, std::memory_order_relaxed)) {
    CHECK(owner == self, "gpu_heap: pool calls must stay on the thread that made the first one");
  }
}

PoolSlot* find_slot(PoolId pool) noexcept {
  const auto raw = static_cast<uint32_t>(pool);
  const uint32_t index = (raw & SlotMask) - 1;
  if (index >= MaxPools) {
    return nullptr;
  }
  PoolSlot& slot = g_slots[index];
  if (slot.state.load(std::memory_order_acquire) == SlotState::Free || slot.id.load(std::memory_order_relaxed) != raw) {
    return nullptr;
  }
  return &slot;
}

PoolSlot* find_live_slot(PoolId pool) noexcept {
  PoolSlot* slot = find_slot(pool);
  return slot != nullptr && slot->state.load(std::memory_order_acquire) == SlotState::Live ? slot : nullptr;
}

void publish(PoolSlot& slot) noexcept {
  const Tlsf& tlsf = *slot.tlsf;
  auto& out = slot.published;
  out.capacity.store(tlsf.capacity(), std::memory_order_relaxed);
  out.used.store(tlsf.used(), std::memory_order_relaxed);
  out.pendingFreeBytes.store(slot.pendingFreeBytes, std::memory_order_relaxed);
  out.pendingCapacity.store(slot.pendingCapacity, std::memory_order_relaxed);
  out.allocations.store(tlsf.allocations(), std::memory_order_relaxed);
  out.generation.store(slot.generation, std::memory_order_relaxed);
  out.growths.store(slot.growths, std::memory_order_relaxed);
  out.rejectedUploads.store(slot.rejectedUploads, std::memory_order_relaxed);
  out.atLimit.store(slot.atLimit, std::memory_order_relaxed);
}

// Finding the largest free block walks a free list, so it is refreshed once per frame instead of on every call
void publish_largest_free(PoolSlot& slot) noexcept {
  slot.published.largestFree.store(slot.tlsf->stats().largestFree, std::memory_order_relaxed);
}

// Requires g_mutex held.
void reset_slot(PoolSlot& slot) {
  slot.id.store(0, std::memory_order_relaxed);
  auto& out = slot.published;
  out.capacity.store(0, std::memory_order_relaxed);
  out.used.store(0, std::memory_order_relaxed);
  out.largestFree.store(0, std::memory_order_relaxed);
  out.pendingFreeBytes.store(0, std::memory_order_relaxed);
  out.pendingCapacity.store(0, std::memory_order_relaxed);
  out.allocations.store(0, std::memory_order_relaxed);
  out.generation.store(0, std::memory_order_relaxed);
  out.growths.store(0, std::memory_order_relaxed);
  out.rejectedUploads.store(0, std::memory_order_relaxed);
  out.atLimit.store(false, std::memory_order_relaxed);
  slot.name.clear();
  slot.usage = wgpu::BufferUsage::None;
  slot.maxSize = 0;
  slot.gpuWrittenOnly = false;
  slot.tlsf.reset();
  slot.buffer = {};
  slot.pendingFrees.clear();
  slot.pendingFreeBytes = 0;
  slot.pendingCapacity = 0;
  slot.generation = 0;
  slot.growths = 0;
  slot.rejectedUploads = 0;
  slot.atLimit = false;
  slot.loggedLimit = false;
  slot.slotGeneration = slot.slotGeneration % MaxSlotGeneration + 1;
  slot.state.store(SlotState::Free, std::memory_order_release);
}

uint64_t growth_target(const PoolSlot& slot, uint64_t wanted) noexcept {
  return wanted >= slot.maxSize ? slot.maxSize : std::min(std::bit_ceil(wanted), slot.maxSize);
}

// The worst case pads the front of a block by the alignment less one granule, as Tlsf::allocate searches
bool fits_in_limit(const PoolSlot& slot, uint64_t size, uint64_t alignment) noexcept {
  if (size > slot.maxSize || alignment > slot.maxSize) {
    return false;
  }
  const uint64_t granularity = slot.tlsf->granularity();
  const uint64_t rounded = (size + granularity - 1) & ~(granularity - 1);
  const uint64_t padding = alignment > granularity ? alignment - granularity : 0;
  return rounded + padding <= slot.maxSize;
}

void grow(PoolSlot& slot, const wgpu::CommandEncoder* shared = nullptr) {
  const uint64_t capacity = slot.tlsf->capacity();
  const wgpu::BufferDescriptor descriptor{
      .label = slot.name.c_str(),
      .usage = slot.usage,
      .size = slot.pendingCapacity,
  };
  wgpu::Buffer buffer = webgpu::g_device.CreateBuffer(&descriptor);
  if (slot.tlsf->used() > 0 && shared != nullptr) {
    shared->CopyBufferToBuffer(slot.buffer, 0, buffer, 0, capacity);
  } else if (slot.tlsf->used() > 0) {
    // Its own submit, so the copy lands before this frame's WriteBuffer uploads instead of after them
    constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "gpu_heap growth"};
    const wgpu::CommandEncoder encoder = webgpu::g_device.CreateCommandEncoder(&EncoderDescriptor);
    encoder.CopyBufferToBuffer(slot.buffer, 0, buffer, 0, capacity);
    const wgpu::CommandBuffer commands = encoder.Finish();
    webgpu::g_queue.Submit(1, &commands);
    g_growthSubmits.fetch_add(1, std::memory_order_relaxed);
  }
  slot.buffer = std::move(buffer);
  slot.tlsf->grow(slot.pendingCapacity);
  ++slot.generation;
  ++slot.growths;
  Log.info("gpu_heap {}: grew to {:.2f} MiB", slot.name, static_cast<double>(slot.pendingCapacity) / MiB);
}

BlockSlot* find_block(BlockHandle block) noexcept {
  const auto raw = static_cast<uint32_t>(block);
  const uint32_t index = (raw & BlockIndexMask) - 1;
  if (index >= MaxBlocks) {
    return nullptr;
  }
  BlockSlot& slot = g_blocks[index];
  return slot.id.load(std::memory_order_acquire) == raw ? &slot : nullptr;
}

BlockSlot& block_slot(uint32_t handle) noexcept { return g_blocks[(handle & BlockIndexMask) - 1]; }

void reset_block(BlockSlot& slot) noexcept {
  slot.state.store(BlockState::Invalid, std::memory_order_release);
  slot.id.store(0, std::memory_order_release);
  slot.slotGeneration = slot.slotGeneration % MaxBlockGeneration + 1;
  slot.region = Region::DrawArgs;
  slot.size = 0;
  slot.destroyQueued = false;
  slot.tlsfBlock = Tlsf::InvalidBlock;
  slot.offset = 0;
  slot.generation = 0;
}

uint64_t request_stamp() noexcept {
  return std::this_thread::get_id() == g_workerThread.load(std::memory_order_relaxed)
             ? g_workerFrame.load(std::memory_order_relaxed)
             : g_recordingFrame.load(std::memory_order_acquire);
}

PoolSlot* region_pool(size_t region) noexcept {
  return find_live_slot(g_regions[region].load(std::memory_order_acquire));
}

bool place_block(BlockSlot& slot, PoolSlot& pool) {
  const auto block = pool.tlsf->allocate(slot.size);
  if (!block) {
    return false;
  }
  slot.tlsfBlock = block->block;
  slot.offset = block->offset;
  slot.generation = 1;
  return true;
}

void apply_blocks(uint64_t frameId) {
  std::vector<Request> due;
  {
    std::lock_guard lock{g_blockMutex};
    const auto split = std::stable_partition(g_requests.begin(), g_requests.end(),
                                             [&](const Request& request) { return request.stamp < frameId; });
    due.assign(g_requests.begin(), split);
    g_requests.erase(g_requests.begin(), split);
  }

  for (const Request& request : due) {
    BlockSlot* slot = request.kind == Request::Destroy ? find_block(static_cast<BlockHandle>(request.handle)) : nullptr;
    if (slot == nullptr) {
      continue;
    }
    if (PoolSlot* pool = region_pool(static_cast<size_t>(slot->region));
        pool != nullptr && slot->tlsfBlock != Tlsf::InvalidBlock) {
      pool->tlsf->free(slot->tlsfBlock);
    }
    std::lock_guard lock{g_blockMutex};
    reset_block(*slot);
  }

  std::vector<uint32_t> placing = std::exchange(g_retries, {});
  for (const Request& request : due) {
    if (request.kind == Request::Create) {
      placing.push_back(request.handle);
    }
  }
  std::erase_if(placing, [](uint32_t handle) { return find_block(static_cast<BlockHandle>(handle)) == nullptr; });

  std::array<uint64_t, RegionCount> wanted{};
  for (const uint32_t handle : placing) {
    const BlockSlot& slot = block_slot(handle);
    wanted[static_cast<size_t>(slot.region)] += slot.size;
  }
  for (size_t region = 0; region < RegionCount; ++region) {
    if (wanted[region] == 0 || g_regions[region].load(std::memory_order_relaxed) != PoolId::Invalid) {
      continue;
    }
    const RegionInfo& info = Regions[region];
    const PoolId pool = create_pool({
        .name = info.name,
        .usage = info.usage,
        .initialSize = std::max(info.initialSize, std::bit_ceil(wanted[region] + wanted[region] / 16)),
        .granularity = RegionGranularity,
        .gpuWrittenOnly = true,
    });
    if (pool == PoolId::Invalid) {
      if (!std::exchange(g_regionCounters[region].loggedCreateFailure, true)) {
        Log.error("gpu_heap {}: the region could not be created, so its blocks failed", info.name);
      }
      continue;
    }
    g_regions[region].store(pool, std::memory_order_release);
  }

  std::vector<uint32_t> placed;
  std::vector<uint32_t> failed;
  std::array<std::vector<uint32_t>, RegionCount> missed;
  for (const uint32_t handle : placing) {
    BlockSlot& slot = block_slot(handle);
    const auto region = static_cast<size_t>(slot.region);
    PoolSlot* pool = region_pool(region);
    if (pool == nullptr) {
      failed.push_back(handle);
    } else if (place_block(slot, *pool)) {
      placed.push_back(handle);
    } else {
      missed[region].push_back(handle);
    }
  }

  wgpu::CommandEncoder encoder;
  const auto shared_encoder = [&]() -> const wgpu::CommandEncoder& {
    if (!encoder) {
      constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "gpu_heap blocks"};
      encoder = webgpu::g_device.CreateCommandEncoder(&EncoderDescriptor);
    }
    return encoder;
  };
  bool copied = false;
  for (size_t region = 0; region < RegionCount; ++region) {
    if (missed[region].empty()) {
      continue;
    }
    PoolSlot& pool = *region_pool(region);
    const uint64_t capacity = pool.tlsf->capacity();
    if (capacity < pool.maxSize) {
      uint64_t miss = 0;
      for (const uint32_t handle : missed[region]) {
        miss += block_slot(handle).size;
      }
      const uint64_t target = growth_target(pool, std::max(2 * capacity, capacity + miss + miss / 16));
      pool.pendingCapacity = std::max(pool.pendingCapacity, target);
      copied = copied || pool.tlsf->used() > 0;
      grow(pool, &shared_encoder());
      for (BlockSlot& slot : g_blocks) {
        if (slot.state.load(std::memory_order_relaxed) == BlockState::Ready &&
            static_cast<size_t>(slot.region) == region) {
          slot.generation = slot.generation == UINT32_MAX ? 1 : slot.generation + 1;
        }
      }
      std::erase_if(missed[region], [&](uint32_t handle) {
        if (!place_block(block_slot(handle), pool)) {
          return false;
        }
        placed.push_back(handle);
        return true;
      });
    }
    const bool atLimit = pool.tlsf->capacity() >= pool.maxSize;
    for (const uint32_t handle : missed[region]) {
      (atLimit ? failed : g_retries).push_back(handle);
    }
    if (atLimit && !missed[region].empty() && !std::exchange(g_regionCounters[region].loggedLimit, true)) {
      Log.warn("gpu_heap {}: block of {} bytes failed at the {:.2f} MiB limit", pool.name,
               block_slot(missed[region].front()).size, static_cast<double>(pool.maxSize) / MiB);
    }
  }

  // After the growth copy, so a reused range is zero the first time a frame sees it
  for (const uint32_t handle : placed) {
    const BlockSlot& slot = block_slot(handle);
    const auto region = static_cast<size_t>(slot.region);
    shared_encoder().ClearBuffer(region_pool(region)->buffer, slot.offset, slot.size);
    g_regionCounters[region].clearedBytes.fetch_add(slot.size, std::memory_order_relaxed);
  }
  if (encoder) {
    const wgpu::CommandBuffer commands = encoder.Finish();
    webgpu::g_queue.Submit(1, &commands);
    g_boundarySubmits.fetch_add(1, std::memory_order_relaxed);
    if (copied) {
      g_growthSubmits.fetch_add(1, std::memory_order_relaxed);
    }
  }

  std::array<bool, RegionCount> live{};
  {
    std::lock_guard lock{g_blockMutex};
    for (const uint32_t handle : placed) {
      block_slot(handle).state.store(BlockState::Ready, std::memory_order_release);
    }
    for (const uint32_t handle : failed) {
      block_slot(handle).state.store(BlockState::Failed, std::memory_order_release);
    }
    for (const BlockSlot& slot : g_blocks) {
      const BlockState state = slot.state.load(std::memory_order_relaxed);
      if (slot.id.load(std::memory_order_relaxed) != 0 &&
          (state == BlockState::Ready || state == BlockState::Pending)) {
        live[static_cast<size_t>(slot.region)] = true;
      }
    }
    g_blockWork.store(!g_requests.empty() || !g_retries.empty(), std::memory_order_release);
  }
  for (size_t region = 0; region < RegionCount; ++region) {
    const PoolId pool = g_regions[region].load(std::memory_order_relaxed);
    if (pool == PoolId::Invalid) {
      continue;
    }
    if (live[region]) {
      if (PoolSlot* slot = region_pool(region)) {
        publish(*slot);
        publish_largest_free(*slot);
      }
      continue;
    }
    destroy_pool(pool);
    g_regions[region].store(PoolId::Invalid, std::memory_order_release);
    g_regionCounters[region].releases.fetch_add(1, std::memory_order_relaxed);
  }
}
} // namespace

PoolId create_pool(const PoolDesc& desc) {
  const char* name = desc.name != nullptr ? desc.name : "unnamed";
  if (desc.backing != Backing::Dawn) {
    Log.error("gpu_heap {}: native shared pools are not available", name);
    return PoolId::Invalid;
  }
  if (desc.granularity < 4 || !std::has_single_bit(desc.granularity)) {
    Log.error("gpu_heap {}: granularity {} is not a power of two of at least 4", name, desc.granularity);
    return PoolId::Invalid;
  }
  if (desc.usage & (wgpu::BufferUsage::MapRead | wgpu::BufferUsage::MapWrite)) {
    Log.error("gpu_heap {}: mappable pools cannot grow", name);
    return PoolId::Invalid;
  }
  if (!webgpu::g_device) {
    Log.error("gpu_heap {}: no device", name);
    return PoolId::Invalid;
  }
  const DeviceLimits& limits = caps().limits;
  uint64_t limit = limits.maxBufferSize;
  uint32_t offsetAlignment = 0;
  if (desc.usage & wgpu::BufferUsage::Storage) {
    limit = std::min(limit, limits.maxStorageBufferBindingSize);
    offsetAlignment = std::max(offsetAlignment, limits.minStorageBufferOffsetAlignment);
  }
  if (desc.usage & wgpu::BufferUsage::Uniform) {
    limit = std::min(limit, limits.maxUniformBufferBindingSize);
    offsetAlignment = std::max(offsetAlignment, limits.minUniformBufferOffsetAlignment);
  }
  if (desc.granularity < offsetAlignment) {
    Log.error("gpu_heap {}: granularity {} is below the {} byte bind offset alignment", name, desc.granularity,
              offsetAlignment);
    return PoolId::Invalid;
  }
  const uint64_t granularityMask = desc.granularity - 1;
  const uint64_t maxSize = (desc.maxSize == 0 ? limit : std::min(desc.maxSize, limit)) & ~granularityMask;
  if (maxSize < desc.granularity) {
    Log.error("gpu_heap {}: the size limit is below the {} byte granularity", name, desc.granularity);
    return PoolId::Invalid;
  }
  const uint64_t initial = (std::min(desc.initialSize, maxSize) + granularityMask) & ~granularityMask;
  const uint64_t size = std::max<uint64_t>(desc.granularity, initial);
  const wgpu::BufferUsage usage = desc.usage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;

  std::lock_guard lock{g_mutex};
  uint32_t index = 0;
  while (index < MaxPools && g_slots[index].state.load(std::memory_order_acquire) != SlotState::Free) {
    ++index;
  }
  if (index == MaxPools) {
    Log.error("gpu_heap {}: all {} pools are in use", name, MaxPools);
    return PoolId::Invalid;
  }
  PoolSlot& slot = g_slots[index];
  const wgpu::BufferDescriptor descriptor{
      .label = name,
      .usage = usage,
      .size = size,
  };
  slot.buffer = webgpu::g_device.CreateBuffer(&descriptor);
  slot.name = name;
  slot.usage = usage;
  slot.maxSize = maxSize;
  slot.gpuWrittenOnly = desc.gpuWrittenOnly;
  slot.tlsf = std::make_unique<Tlsf>(size, desc.granularity);
  slot.pendingCapacity = size;
  slot.generation = 1;
  publish(slot);
  publish_largest_free(slot);
  const uint32_t id = (slot.slotGeneration << SlotBits) | (index + 1);
  slot.id.store(id, std::memory_order_relaxed);
  slot.state.store(SlotState::Live, std::memory_order_release);
  g_activePools.fetch_add(1, std::memory_order_release);
  return static_cast<PoolId>(id);
}

void destroy_pool(PoolId pool) {
  std::lock_guard lock{g_mutex};
  if (PoolSlot* slot = find_slot(pool)) {
    auto expected = SlotState::Live;
    slot->state.compare_exchange_strong(expected, SlotState::Destroying, std::memory_order_acq_rel);
  }
}

Allocation allocate(PoolId pool, uint64_t size, uint64_t alignment) {
  check_owner();
  PoolSlot* slot = find_live_slot(pool);
  if (slot == nullptr || size == 0 || (alignment != 0 && !std::has_single_bit(alignment))) {
    return {};
  }
  if (const auto block = slot->tlsf->allocate(size, alignment)) {
    slot->atLimit = false;
    publish(*slot);
    return {.pool = pool, .block = block->block, .offset = block->offset, .size = block->size};
  }
  const uint64_t capacity = slot->tlsf->capacity();
  // A request that could never fit leaves the pool alone, so one bad size cannot grow it to the device limit
  if (capacity >= slot->maxSize || !fits_in_limit(*slot, size, alignment)) {
    slot->atLimit = true;
    if (!std::exchange(slot->loggedLimit, true)) {
      Log.warn("gpu_heap {}: allocation of {} bytes failed at the {:.2f} MiB limit", slot->name, size,
               static_cast<double>(slot->maxSize) / MiB);
    }
  } else {
    // Growth waits for the next frame boundary; the caller retries then
    const uint64_t wanted = std::max(2 * capacity, capacity + size + alignment);
    slot->pendingCapacity = std::max(slot->pendingCapacity, growth_target(*slot, wanted));
  }
  publish(*slot);
  return {};
}

void reserve(PoolId pool, uint64_t capacity) {
  check_owner();
  PoolSlot* slot = find_live_slot(pool);
  if (slot == nullptr || capacity <= slot->pendingCapacity) {
    return;
  }
  slot->pendingCapacity = std::max(slot->pendingCapacity, growth_target(*slot, capacity));
  publish(*slot);
}

void free(const Allocation& allocation) {
  check_owner();
  if (!allocation) {
    return;
  }
  PoolSlot* slot = find_slot(allocation.pool);
  CHECK(slot != nullptr, "gpu_heap: free of {} bytes from a pool that no longer exists", allocation.size);
  // A destroying pool drops every block at the boundary, so its frees need no queueing
  if (slot == nullptr || allocation.block == Tlsf::InvalidBlock ||
      slot->state.load(std::memory_order_acquire) != SlotState::Live) {
    return;
  }
  slot->pendingFrees.push_back(allocation.block);
  slot->pendingFreeBytes += allocation.size;
  publish(*slot);
}

void upload(const Allocation& allocation, uint64_t offset, const void* data, uint64_t size) {
  check_owner();
  PoolSlot* slot = find_slot(allocation.pool);
  CHECK(slot != nullptr, "gpu_heap: upload of {} bytes to a pool that no longer exists", size);
  if (slot == nullptr) {
    return;
  }
  const bool valid = !slot->gpuWrittenOnly && offset <= allocation.size && size <= allocation.size - offset &&
                     offset % 4 == 0 && size % 4 == 0 && (data != nullptr || size == 0);
  CHECK(valid, "gpu_heap {}: rejected upload of {} bytes at offset {}", slot->name, size, offset);
  if (!valid) {
    ++slot->rejectedUploads;
    publish(*slot);
    return;
  }
  if (size == 0) {
    return;
  }
  webgpu::g_queue.WriteBuffer(slot->buffer, allocation.offset + offset, data, size);
}

PoolBinding binding(PoolId pool) {
  check_owner();
  const PoolSlot* slot = find_slot(pool);
  if (slot == nullptr) {
    return {};
  }
  return {.buffer = slot->buffer, .size = slot->tlsf->capacity(), .generation = slot->generation};
}

PoolStats stats(PoolId pool) noexcept {
  const PoolSlot* slot = find_slot(pool);
  if (slot == nullptr) {
    return {};
  }
  const auto& in = slot->published;
  return {
      .capacity = in.capacity.load(std::memory_order_relaxed),
      .used = in.used.load(std::memory_order_relaxed),
      .largestFree = in.largestFree.load(std::memory_order_relaxed),
      .pendingFreeBytes = in.pendingFreeBytes.load(std::memory_order_relaxed),
      .pendingCapacity = in.pendingCapacity.load(std::memory_order_relaxed),
      .allocations = in.allocations.load(std::memory_order_relaxed),
      .generation = in.generation.load(std::memory_order_relaxed),
      .growths = in.growths.load(std::memory_order_relaxed),
      .rejectedUploads = in.rejectedUploads.load(std::memory_order_relaxed),
      .atLimit = in.atLimit.load(std::memory_order_relaxed),
  };
}

uint64_t total_bytes() noexcept {
  uint64_t total = 0;
  for (const auto& slot : g_slots) {
    total += slot.published.capacity.load(std::memory_order_relaxed);
  }
  return total;
}

uint32_t growth_submits() noexcept { return g_growthSubmits.load(std::memory_order_relaxed); }

void set_recording_frame(uint64_t frameId) noexcept { g_recordingFrame.store(frameId, std::memory_order_release); }

BlockHandle create_block(Region region, uint64_t size) {
  const DeviceLimits& limits = caps().limits;
  const uint64_t limit =
      std::min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) & ~uint64_t{RegionGranularity - 1};
  if (static_cast<size_t>(region) >= RegionCount || size == 0 || size > limit) {
    Log.error("gpu_heap: block of {} bytes is outside the {} byte region limit", size, limit);
    return BlockHandle::Invalid;
  }
  const uint64_t rounded = (size + RegionGranularity - 1) & ~uint64_t{RegionGranularity - 1};
  std::lock_guard lock{g_blockMutex};
  const auto it = std::ranges::find_if(
      g_blocks, [](const BlockSlot& slot) { return slot.id.load(std::memory_order_relaxed) == 0; });
  if (it == g_blocks.end()) {
    Log.error("gpu_heap: all {} blocks are in use", MaxBlocks);
    return BlockHandle::Invalid;
  }
  BlockSlot& slot = *it;
  const auto index = static_cast<uint32_t>(it - g_blocks.begin());
  const uint32_t id = slot.slotGeneration << BlockIndexBits | (index + 1);
  slot.region = region;
  slot.size = rounded;
  slot.destroyQueued = false;
  slot.state.store(BlockState::Pending, std::memory_order_release);
  slot.id.store(id, std::memory_order_release);
  g_requests.push_back({.kind = Request::Create, .handle = id, .stamp = request_stamp()});
  g_blockWork.store(true, std::memory_order_release);
  return static_cast<BlockHandle>(id);
}

void destroy_block(BlockHandle block) {
  std::lock_guard lock{g_blockMutex};
  BlockSlot* slot = find_block(block);
  if (slot == nullptr || slot->destroyQueued) {
    return;
  }
  slot->destroyQueued = true;
  g_requests.push_back({.kind = Request::Destroy, .handle = static_cast<uint32_t>(block), .stamp = request_stamp()});
  g_blockWork.store(true, std::memory_order_release);
}

BlockState block_state(BlockHandle block) noexcept {
  const BlockSlot* slot = find_block(block);
  if (slot == nullptr) {
    return BlockState::Invalid;
  }
  const BlockState state = slot->state.load(std::memory_order_acquire);
  return slot->id.load(std::memory_order_acquire) == static_cast<uint32_t>(block) ? state : BlockState::Invalid;
}

BlockView block_view(BlockHandle block) {
  check_owner();
  const BlockSlot* slot = find_block(block);
  if (slot == nullptr || slot->state.load(std::memory_order_acquire) != BlockState::Ready) {
    return {};
  }
  const PoolSlot* pool = region_pool(static_cast<size_t>(slot->region));
  if (pool == nullptr) {
    return {};
  }
  return {.buffer = pool->buffer, .offset = slot->offset, .size = slot->size, .generation = slot->generation};
}

RegionStats region_stats(Region region) noexcept {
  const auto index = static_cast<size_t>(region);
  if (index >= RegionCount) {
    return {};
  }
  const PoolId pool = g_regions[index].load(std::memory_order_acquire);
  RegionStats out{
      .pool = stats(pool),
      .created = pool != PoolId::Invalid,
      .releases = g_regionCounters[index].releases.load(std::memory_order_relaxed),
      .clearedBytes = g_regionCounters[index].clearedBytes.load(std::memory_order_relaxed),
  };
  std::lock_guard lock{g_blockMutex};
  for (const BlockSlot& slot : g_blocks) {
    if (slot.id.load(std::memory_order_relaxed) == 0 || slot.region != region) {
      continue;
    }
    switch (slot.state.load(std::memory_order_relaxed)) {
    case BlockState::Ready:
      ++out.blocks;
      break;
    case BlockState::Pending:
      ++out.pending;
      break;
    case BlockState::Failed:
      ++out.failed;
      break;
    default:
      break;
    }
  }
  return out;
}

uint32_t boundary_submits() noexcept { return g_boundarySubmits.load(std::memory_order_relaxed); }

void begin_frame_worker(uint64_t frameId) {
  if (frameId != AllRequests) {
    g_workerThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    g_workerFrame.store(frameId, std::memory_order_relaxed);
  }
  if (g_activePools.load(std::memory_order_acquire) == 0 && !g_blockWork.load(std::memory_order_acquire)) {
    return;
  }
  check_owner();
  if (g_blockWork.load(std::memory_order_acquire)) {
    apply_blocks(frameId);
  }
  for (auto& slot : g_slots) {
    const SlotState state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::Free) {
      continue;
    }
    if (state == SlotState::Destroying) {
      std::lock_guard lock{g_mutex};
      reset_slot(slot);
      g_activePools.fetch_sub(1, std::memory_order_acq_rel);
      continue;
    }
    // Frame N's commands were submitted before this boundary, so its freed ranges are safe to reuse
    for (const Tlsf::BlockId block : slot.pendingFrees) {
      slot.tlsf->free(block);
    }
    slot.pendingFrees.clear();
    slot.pendingFreeBytes = 0;
    if (slot.pendingCapacity > slot.tlsf->capacity()) {
      grow(slot);
    }
    publish(slot);
    publish_largest_free(slot);
  }
}

void shutdown() {
  {
    std::lock_guard lock{g_blockMutex};
    for (auto& slot : g_blocks) {
      if (slot.id.load(std::memory_order_relaxed) != 0) {
        reset_block(slot);
      }
    }
    g_requests.clear();
  }
  g_retries.clear();
  for (size_t region = 0; region < RegionCount; ++region) {
    g_regions[region].store(PoolId::Invalid, std::memory_order_release);
    RegionCounters& counters = g_regionCounters[region];
    counters.clearedBytes.store(0, std::memory_order_relaxed);
    counters.releases.store(0, std::memory_order_relaxed);
    counters.loggedCreateFailure = false;
    counters.loggedLimit = false;
  }
  g_boundarySubmits.store(0, std::memory_order_relaxed);
  g_blockWork.store(false, std::memory_order_release);
  g_recordingFrame.store(0, std::memory_order_release);
  g_workerFrame.store(0, std::memory_order_relaxed);
  g_workerThread.store(std::thread::id{}, std::memory_order_relaxed);
  std::lock_guard lock{g_mutex};
  for (auto& slot : g_slots) {
    if (slot.state.load(std::memory_order_acquire) != SlotState::Free) {
      reset_slot(slot);
    }
  }
  g_activePools.store(0, std::memory_order_release);
  g_growthSubmits.store(0, std::memory_order_relaxed);
  g_owner.store(std::thread::id{}, std::memory_order_relaxed);
}

} // namespace aurora::render::gpu_heap

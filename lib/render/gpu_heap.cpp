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

void grow(PoolSlot& slot) {
  const uint64_t capacity = slot.tlsf->capacity();
  const wgpu::BufferDescriptor descriptor{
      .label = slot.name.c_str(),
      .usage = slot.usage,
      .size = slot.pendingCapacity,
  };
  wgpu::Buffer buffer = webgpu::g_device.CreateBuffer(&descriptor);
  if (slot.tlsf->used() > 0) {
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

void begin_frame_worker() {
  if (g_activePools.load(std::memory_order_acquire) == 0) {
    return;
  }
  check_owner();
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

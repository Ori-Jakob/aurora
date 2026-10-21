#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "logging.hpp"
#include "render/gpu_heap.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::render {
namespace {
using gpu_heap::Allocation;
using gpu_heap::PoolId;
using gpu_heap::PoolStats;

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;

struct LogLine {
  AuroraLogLevel level;
  std::string message;
};

std::mutex g_logMutex;
std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  std::lock_guard lock{g_logMutex};
  g_logLines.push_back({level, std::string{message, len}});
}

size_t count_logs(AuroraLogLevel level, std::string_view text) {
  std::lock_guard lock{g_logMutex};
  size_t count = 0;
  for (const auto& line : g_logLines) {
    if (line.level == level && line.message.find(text) != std::string::npos) {
      ++count;
    }
  }
  return count;
}

uint32_t slot_bits(PoolId pool) { return static_cast<uint32_t>(pool) & 0xFFu; }

class GpuHeapTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    m_device = test::create_test_device({}, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, test::TestDeviceOptions{});
    gpu_heap::shutdown();
    {
      std::lock_guard lock{g_logMutex};
      g_logLines.clear();
    }
    g_config.logCallback = capture_log;
  }

  void TearDown() override {
    gpu_heap::shutdown();
    g_config.logCallback = nullptr;
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static PoolId storage_pool(uint64_t initialSize, uint64_t maxSize = 0, bool gpuWrittenOnly = false) {
    return gpu_heap::create_pool({
        .name = "test.storage",
        .usage = wgpu::BufferUsage::Storage,
        .initialSize = initialSize,
        .maxSize = maxSize,
        .gpuWrittenOnly = gpuWrittenOnly,
    });
  }

  void push_validation_scope() const { m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation); }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(GpuHeapTest, CreatePoolAddsCopyUsages) {
  const PoolId pool = storage_pool(4 * KiB);
  ASSERT_NE(pool, PoolId::Invalid);
  const auto binding = gpu_heap::binding(pool);
  ASSERT_TRUE(binding.buffer);
  EXPECT_EQ(binding.buffer.GetUsage(),
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
  EXPECT_EQ(binding.buffer.GetSize(), 4 * KiB);
  EXPECT_EQ(binding.size, 4 * KiB);
  EXPECT_EQ(binding.generation, 1u);
  EXPECT_EQ(gpu_heap::stats(pool).generation, 1u);
  EXPECT_EQ(gpu_heap::total_bytes(), 4 * KiB);
}

TEST_F(GpuHeapTest, AllocateRespectsGranularity) {
  const PoolId pool = storage_pool(64 * KiB);
  for (int i = 0; i < 8; ++i) {
    const Allocation allocation = gpu_heap::allocate(pool, 1);
    ASSERT_TRUE(allocation);
    EXPECT_EQ(allocation.offset % 256, 0u);
    EXPECT_EQ(allocation.size, 256u);
  }
  EXPECT_EQ(gpu_heap::stats(pool).used, 8 * 256u);
}

TEST_F(GpuHeapTest, ExhaustionSchedulesGrowth) {
  const PoolId pool = storage_pool(4 * KiB, 1 * MiB);
  const wgpu::Buffer before = gpu_heap::binding(pool).buffer;
  ASSERT_TRUE(gpu_heap::allocate(pool, 4 * KiB));
  EXPECT_FALSE(gpu_heap::allocate(pool, 256));
  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_GE(stats.pendingCapacity, 8 * KiB);
  EXPECT_EQ(stats.capacity, 4 * KiB);
  EXPECT_FALSE(stats.atLimit);
  const auto binding = gpu_heap::binding(pool);
  EXPECT_EQ(binding.buffer.Get(), before.Get());
  EXPECT_EQ(binding.generation, 1u);
}

TEST_F(GpuHeapTest, GrowthAppliesAtFrameBoundary) {
  const PoolId pool = storage_pool(4 * KiB, 1 * MiB);
  const Allocation first = gpu_heap::allocate(pool, 1 * KiB);
  const Allocation second = gpu_heap::allocate(pool, 3 * KiB);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const std::vector<uint32_t> data(256, 0x12345678u);
  gpu_heap::upload(first, 0, data.data(), data.size() * sizeof(uint32_t));
  ASSERT_FALSE(gpu_heap::allocate(pool, 2 * KiB));
  const uint64_t pending = gpu_heap::stats(pool).pendingCapacity;
  const wgpu::Buffer before = gpu_heap::binding(pool).buffer;

  push_validation_scope();
  gpu_heap::begin_frame_worker();
  const auto binding = gpu_heap::binding(pool);
  EXPECT_NE(binding.buffer.Get(), before.Get());
  EXPECT_GE(binding.buffer.GetSize(), pending);
  EXPECT_EQ(binding.size, binding.buffer.GetSize());
  EXPECT_EQ(binding.generation, 2u);
  EXPECT_EQ(gpu_heap::growth_submits(), 1u);
  const Allocation retry = gpu_heap::allocate(pool, 2 * KiB);
  ASSERT_TRUE(retry);
  // The copied range keeps the earlier offsets, so new space only comes from the tail
  EXPECT_GE(retry.offset, second.offset + second.size);
  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.used, 6 * KiB);
  EXPECT_EQ(stats.allocations, 3u);
  EXPECT_EQ(stats.growths, 1u);

  // The allocator kept the pre-growth block, so its range comes back once freed
  gpu_heap::free(first);
  gpu_heap::begin_frame_worker();
  const Allocation reused = gpu_heap::allocate(pool, 1 * KiB);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused.offset, first.offset);
  EXPECT_EQ(gpu_heap::binding(pool).generation, 2u);
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
}

TEST_F(GpuHeapTest, GrowthWithoutDataSkipsCopy) {
  const PoolId pool = storage_pool(4 * KiB, 1 * MiB);
  gpu_heap::reserve(pool, 64 * KiB);
  EXPECT_EQ(gpu_heap::stats(pool).pendingCapacity, 64 * KiB);
  gpu_heap::begin_frame_worker();
  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.capacity, 64 * KiB);
  EXPECT_EQ(stats.largestFree, 64 * KiB);
  EXPECT_EQ(stats.generation, 2u);
  EXPECT_EQ(stats.growths, 1u);
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);
  EXPECT_EQ(gpu_heap::binding(pool).buffer.GetSize(), 64 * KiB);
}

TEST_F(GpuHeapTest, FreeIsDeferred) {
  const PoolId pool = storage_pool(4 * KiB);
  const Allocation allocation = gpu_heap::allocate(pool, 256);
  ASSERT_TRUE(allocation);
  gpu_heap::free(allocation);
  EXPECT_EQ(gpu_heap::stats(pool).pendingFreeBytes, 256u);
  const Allocation sameFrame = gpu_heap::allocate(pool, allocation.size);
  ASSERT_TRUE(sameFrame);
  EXPECT_NE(sameFrame.offset, allocation.offset);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::stats(pool).pendingFreeBytes, 0u);
  const Allocation nextFrame = gpu_heap::allocate(pool, allocation.size);
  ASSERT_TRUE(nextFrame);
  EXPECT_EQ(nextFrame.offset, allocation.offset);
}

TEST_F(GpuHeapTest, GrowthStopsAtLimit) {
  const PoolId pool = storage_pool(8 * KiB, 8 * KiB);
  const Allocation full = gpu_heap::allocate(pool, 8 * KiB);
  ASSERT_TRUE(full);
  EXPECT_FALSE(gpu_heap::allocate(pool, 256));
  EXPECT_FALSE(gpu_heap::allocate(pool, 512));
  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_TRUE(stats.atLimit);
  EXPECT_EQ(stats.pendingCapacity, stats.capacity);
  EXPECT_EQ(count_logs(LOG_WARNING, "failed at the"), 1u);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::stats(pool).generation, 1u);
  EXPECT_EQ(gpu_heap::binding(pool).generation, 1u);
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);

  gpu_heap::free(full);
  gpu_heap::begin_frame_worker();
  ASSERT_TRUE(gpu_heap::allocate(pool, 256));
  EXPECT_FALSE(gpu_heap::stats(pool).atLimit);
}

TEST_F(GpuHeapTest, UnfittableRequestDoesNotGrow) {
  const PoolId pool = storage_pool(4 * KiB, 64 * KiB);
  EXPECT_FALSE(gpu_heap::allocate(pool, 128 * KiB));
  // 48 KiB fits the limit on its own, but not with up to 32 KiB less one granule of alignment padding
  EXPECT_FALSE(gpu_heap::allocate(pool, 48 * KiB, 32 * KiB));
  EXPECT_FALSE(gpu_heap::allocate(pool, 4 * KiB, 128 * KiB));
  PoolStats stats = gpu_heap::stats(pool);
  EXPECT_TRUE(stats.atLimit);
  EXPECT_EQ(stats.pendingCapacity, 4 * KiB);
  EXPECT_EQ(count_logs(LOG_WARNING, "failed at the"), 1u);
  gpu_heap::begin_frame_worker();
  stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.capacity, 4 * KiB);
  EXPECT_EQ(stats.generation, 1u);
  EXPECT_EQ(stats.growths, 0u);

  // A request exactly at the worst-case bound still fits, so it grows the pool to its limit and then succeeds
  EXPECT_FALSE(gpu_heap::allocate(pool, 32 * KiB, 32 * KiB));
  EXPECT_EQ(gpu_heap::stats(pool).pendingCapacity, 64 * KiB);
  gpu_heap::begin_frame_worker();
  const Allocation aligned = gpu_heap::allocate(pool, 32 * KiB, 32 * KiB);
  ASSERT_TRUE(aligned);
  EXPECT_EQ(aligned.offset % (32 * KiB), 0u);
  EXPECT_FALSE(gpu_heap::stats(pool).atLimit);
}

TEST_F(GpuHeapTest, UploadRejectedOnGpuWrittenPool) {
  const PoolId pool = storage_pool(4 * KiB, 0, true);
  const Allocation allocation = gpu_heap::allocate(pool, 256);
  ASSERT_TRUE(allocation);
  constexpr std::array<uint32_t, 4> Data{1, 2, 3, 4};
#ifndef NDEBUG
  EXPECT_DEATH(gpu_heap::upload(allocation, 0, Data.data(), sizeof(Data)), "");
#else
  const PoolStats before = gpu_heap::stats(pool);
  push_validation_scope();
  gpu_heap::upload(allocation, 0, Data.data(), sizeof(Data));
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  const PoolStats after = gpu_heap::stats(pool);
  EXPECT_EQ(after.rejectedUploads, before.rejectedUploads + 1);
  EXPECT_EQ(after.used, before.used);
  EXPECT_EQ(after.allocations, before.allocations);
  EXPECT_EQ(after.generation, before.generation);
#endif
}

TEST_F(GpuHeapTest, UploadRejectsUnalignedRange) {
  const PoolId pool = storage_pool(4 * KiB);
  const Allocation allocation = gpu_heap::allocate(pool, 256);
  ASSERT_TRUE(allocation);
  constexpr std::array<uint8_t, 8> Data{1, 2, 3, 4, 5, 6, 7, 8};
#ifndef NDEBUG
  EXPECT_DEATH(gpu_heap::upload(allocation, 0, Data.data(), 6), "");
#else
  // WriteBuffer rejects both, so no error means neither reached the queue
  push_validation_scope();
  gpu_heap::upload(allocation, 0, Data.data(), 6);
  gpu_heap::upload(allocation, 2, Data.data(), 4);
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  EXPECT_EQ(gpu_heap::stats(pool).rejectedUploads, 2u);
#endif
}

TEST_F(GpuHeapTest, FreeAndUploadWhileDestroying) {
  const PoolId pool = storage_pool(4 * KiB);
  const Allocation first = gpu_heap::allocate(pool, 256);
  const Allocation second = gpu_heap::allocate(pool, 256);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  gpu_heap::destroy_pool(pool);

  // Allocations stay valid until the boundary, so owners can still write and release them in any order
  constexpr std::array<uint32_t, 4> Data{1, 2, 3, 4};
  push_validation_scope();
  gpu_heap::upload(second, 0, Data.data(), sizeof(Data));
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  gpu_heap::free(first);
  gpu_heap::free(second);
  PoolStats stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.rejectedUploads, 0u);
  EXPECT_EQ(stats.pendingFreeBytes, 0u);
  EXPECT_EQ(stats.allocations, 2u);

  gpu_heap::begin_frame_worker();
  stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.capacity, 0u);
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
}

TEST_F(GpuHeapTest, DestroyAppliesAtBoundary) {
  const PoolId pool = storage_pool(4 * KiB);
  const Allocation allocation = gpu_heap::allocate(pool, 256);
  ASSERT_TRUE(allocation);
  const uint64_t totalBefore = gpu_heap::total_bytes();
  gpu_heap::destroy_pool(pool);
  EXPECT_TRUE(gpu_heap::binding(pool).buffer);
  EXPECT_FALSE(gpu_heap::allocate(pool, 256));

  gpu_heap::begin_frame_worker();
  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.capacity, 0u);
  EXPECT_EQ(stats.used, 0u);
  EXPECT_EQ(stats.allocations, 0u);
  EXPECT_EQ(stats.generation, 0u);
  EXPECT_LT(gpu_heap::total_bytes(), totalBefore);
  EXPECT_FALSE(gpu_heap::binding(pool).buffer);

  const PoolId reused = storage_pool(4 * KiB);
  ASSERT_NE(reused, PoolId::Invalid);
  EXPECT_NE(reused, pool);
  EXPECT_EQ(slot_bits(reused), slot_bits(pool));
  EXPECT_EQ(gpu_heap::stats(reused).used, 0u);
  EXPECT_EQ(gpu_heap::stats(pool).capacity, 0u);
#ifndef NDEBUG
  EXPECT_DEATH(gpu_heap::free(allocation), "");
#else
  // The stale id names the reused slot, so a free that ignored it would land in the new pool
  gpu_heap::free(allocation);
  EXPECT_EQ(gpu_heap::stats(reused).pendingFreeBytes, 0u);
#endif
}

TEST_F(GpuHeapTest, StatsReadableFromAnotherThread) {
  const PoolId pool = storage_pool(4 * KiB, 1 * MiB);
  std::atomic_bool done = false;
  std::thread reader([&] {
    uint64_t checksum = 0;
    for (int reads = 0; reads < 10000 || !done.load(std::memory_order_acquire); ++reads) {
      checksum += gpu_heap::total_bytes();
      const PoolStats stats = gpu_heap::stats(pool);
      checksum += stats.used + stats.capacity + stats.generation;
    }
    EXPECT_GT(checksum, 0u);
  });
  for (int frame = 0; frame < 64; ++frame) {
    std::vector<Allocation> allocations;
    for (int i = 0; i < 16; ++i) {
      if (const Allocation allocation = gpu_heap::allocate(pool, 1 * KiB)) {
        allocations.push_back(allocation);
      }
    }
    const PoolId other = storage_pool(4 * KiB);
    gpu_heap::begin_frame_worker();
    for (const Allocation& allocation : allocations) {
      gpu_heap::free(allocation);
    }
    gpu_heap::destroy_pool(other);
    gpu_heap::begin_frame_worker();
  }
  done.store(true, std::memory_order_release);
  reader.join();

  const PoolStats stats = gpu_heap::stats(pool);
  EXPECT_EQ(stats.capacity, gpu_heap::binding(pool).size);
  EXPECT_EQ(stats.capacity, 16 * KiB);
  EXPECT_EQ(stats.used, 0u);
  EXPECT_EQ(stats.allocations, 0u);
  EXPECT_EQ(stats.pendingFreeBytes, 0u);
  EXPECT_EQ(gpu_heap::total_bytes(), stats.capacity);
}

TEST_F(GpuHeapTest, CreatePoolFromAnotherThread) {
  std::atomic_uint32_t created = 0;
  std::thread creator([&] { created.store(static_cast<uint32_t>(storage_pool(4 * KiB)), std::memory_order_release); });
  for (int i = 0; i < 1'000'000 && created.load(std::memory_order_acquire) == 0; ++i) {
    gpu_heap::begin_frame_worker();
    std::this_thread::yield();
  }
  creator.join();
  const auto pool = static_cast<PoolId>(created.load(std::memory_order_acquire));
  ASSERT_NE(pool, PoolId::Invalid);
  gpu_heap::begin_frame_worker();
  EXPECT_TRUE(gpu_heap::allocate(pool, 256));
  EXPECT_TRUE(gpu_heap::binding(pool).buffer);
  EXPECT_EQ(gpu_heap::stats(pool).allocations, 1u);
}

struct RejectedDesc {
  const char* label;
  gpu_heap::PoolDesc desc;
  const char* log;
  bool withoutDevice = false;
};

void PrintTo(const RejectedDesc& param, std::ostream* os) { *os << param.label; }

class GpuHeapRejectTest : public GpuHeapTest, public ::testing::WithParamInterface<RejectedDesc> {};

TEST_P(GpuHeapRejectTest, CreatePoolRejectsInvalidDesc) {
  const RejectedDesc& param = GetParam();
  wgpu::Device device;
  if (param.withoutDevice) {
    device = std::exchange(webgpu::g_device, {});
  }
  const PoolId pool = gpu_heap::create_pool(param.desc);
  if (param.withoutDevice) {
    webgpu::g_device = std::move(device);
  }
  EXPECT_EQ(pool, PoolId::Invalid);
  EXPECT_EQ(count_logs(LOG_ERROR, param.log), 1u);
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
}

// The test device keeps the default limits, so both bind offset alignments are 256
INSTANTIATE_TEST_SUITE_P(
    GpuHeap, GpuHeapRejectTest,
    ::testing::Values(
        RejectedDesc{"NativeShared",
                     {.usage = wgpu::BufferUsage::Storage, .backing = gpu_heap::Backing::NativeShared},
                     "native shared pools"},
        RejectedDesc{"GranularityNotPowerOfTwo",
                     {.usage = wgpu::BufferUsage::Storage, .granularity = 768},
                     "is not a power of two"},
        RejectedDesc{
            "GranularityBelowFour", {.usage = wgpu::BufferUsage::Storage, .granularity = 2}, "is not a power of two"},
        RejectedDesc{"Mappable", {.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc}, "mappable pools"},
        RejectedDesc{"BelowStorageOffsetAlignment",
                     {.usage = wgpu::BufferUsage::Storage, .granularity = 16},
                     "bind offset alignment"},
        RejectedDesc{"BelowUniformOffsetAlignment",
                     {.usage = wgpu::BufferUsage::Uniform, .granularity = 64},
                     "bind offset alignment"},
        RejectedDesc{
            "LimitBelowGranularity", {.usage = wgpu::BufferUsage::Storage, .maxSize = 128}, "size limit is below"},
        RejectedDesc{"NoDevice", {.usage = wgpu::BufferUsage::Storage}, "no device", true}),
    [](const ::testing::TestParamInfo<RejectedDesc>& info) { return std::string{info.param.label}; });

TEST_F(GpuHeapTest, NoPoolsNoWork) {
  push_validation_scope();
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
}

TEST_F(GpuHeapTest, ShutdownReleasesPools) {
  const PoolId small = storage_pool(4 * KiB);
  const PoolId large = storage_pool(64 * KiB);
  ASSERT_NE(small, PoolId::Invalid);
  ASSERT_NE(large, PoolId::Invalid);
  EXPECT_EQ(gpu_heap::total_bytes(), 68 * KiB);
  gpu_heap::shutdown();
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
  EXPECT_EQ(gpu_heap::stats(large).capacity, 0u);
  EXPECT_FALSE(gpu_heap::binding(small).buffer);
}

using gpu_heap::BlockHandle;
using gpu_heap::BlockState;
using gpu_heap::BlockView;
using gpu_heap::Region;
using gpu_heap::RegionStats;

constexpr std::array AllRegions{Region::DrawArgs, Region::DispatchArgs, Region::Index32};
constexpr auto WorkerTimeout = std::chrono::seconds{10};

void on_game_thread(const std::function<void()>& work) {
  std::thread thread(work);
  thread.join();
}

class TestWorker {
public:
  TestWorker() : m_thread([this] { run(); }) {}

  ~TestWorker() {
    {
      std::lock_guard lock{m_mutex};
      m_stop = true;
    }
    m_wake.notify_all();
    m_thread.join();
  }

  TestWorker(const TestWorker&) = delete;
  TestWorker& operator=(const TestWorker&) = delete;

  void post(std::function<void()> job) {
    {
      std::lock_guard lock{m_mutex};
      m_jobs.push_back(std::move(job));
    }
    m_wake.notify_all();
  }

  bool drain() {
    std::unique_lock lock{m_mutex};
    return m_wake.wait_for(lock, WorkerTimeout, [&] { return m_jobs.empty() && !m_busy; });
  }

private:
  void run() {
    std::unique_lock lock{m_mutex};
    while (true) {
      m_wake.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
      if (m_jobs.empty()) {
        return;
      }
      std::function<void()> job = std::move(m_jobs.front());
      m_jobs.pop_front();
      m_busy = true;
      lock.unlock();
      job();
      lock.lock();
      m_busy = false;
      m_wake.notify_all();
    }
  }

  std::mutex m_mutex;
  std::condition_variable m_wake;
  std::deque<std::function<void()>> m_jobs;
  bool m_busy = false;
  bool m_stop = false;
  std::thread m_thread;
};

uint64_t block_limit() {
  const DeviceLimits& limits = caps().limits;
  return std::min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) & ~uint64_t{255};
}

TEST_F(GpuHeapTest, RegionsStayUncreatedWithoutBlocks) {
  push_validation_scope();
  for (uint64_t frame = 1; frame <= 10; ++frame) {
    gpu_heap::set_recording_frame(frame);
    gpu_heap::begin_frame_worker(frame);
  }
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  for (const Region region : AllRegions) {
    const RegionStats stats = gpu_heap::region_stats(region);
    EXPECT_FALSE(stats.created);
    EXPECT_EQ(stats.pool.capacity, 0u);
  }
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);
  EXPECT_EQ(gpu_heap::boundary_submits(), 0u);
}

TEST_F(GpuHeapTest, BlockMaterializesAtBoundary) {
  const BlockHandle block = gpu_heap::create_block(Region::DrawArgs, 100);
  ASSERT_NE(block, BlockHandle::Invalid);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Pending);
  EXPECT_FALSE(gpu_heap::block_view(block));
  EXPECT_FALSE(gpu_heap::region_stats(Region::DrawArgs).created);

  push_validation_scope();
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);
  const BlockView view = gpu_heap::block_view(block);
  ASSERT_TRUE(view);
  ASSERT_TRUE(view.buffer);
  EXPECT_EQ(view.size, 256u);
  EXPECT_EQ(view.offset % 256, 0u);
  EXPECT_EQ(view.generation, 1u);
  EXPECT_EQ(view.buffer.GetUsage(), wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage |
                                        wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
  const RegionStats stats = gpu_heap::region_stats(Region::DrawArgs);
  EXPECT_TRUE(stats.created);
  EXPECT_EQ(stats.blocks, 1u);
  EXPECT_EQ(stats.pending, 0u);
  EXPECT_EQ(stats.clearedBytes, 256u);
  EXPECT_EQ(stats.pool.capacity, 64 * KiB);
  EXPECT_EQ(gpu_heap::boundary_submits(), 1u);
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);
}

TEST_F(GpuHeapTest, RegionUsagesMatchRole) {
  std::array<BlockHandle, AllRegions.size()> blocks{};
  for (size_t i = 0; i < AllRegions.size(); ++i) {
    blocks[i] = gpu_heap::create_block(AllRegions[i], 256);
  }
  gpu_heap::begin_frame_worker();
  std::array<BlockView, AllRegions.size()> views;
  for (size_t i = 0; i < AllRegions.size(); ++i) {
    views[i] = gpu_heap::block_view(blocks[i]);
    ASSERT_TRUE(views[i]) << i;
  }
  EXPECT_NE(views[0].buffer.Get(), views[1].buffer.Get());
  EXPECT_NE(views[0].buffer.Get(), views[2].buffer.Get());
  EXPECT_NE(views[1].buffer.Get(), views[2].buffer.Get());
  const auto args = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                    wgpu::BufferUsage::CopyDst;
  EXPECT_EQ(views[0].buffer.GetUsage(), args);
  EXPECT_EQ(views[1].buffer.GetUsage(), args);
  EXPECT_EQ(views[2].buffer.GetUsage(), wgpu::BufferUsage::Index | wgpu::BufferUsage::Vertex |
                                            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                            wgpu::BufferUsage::CopyDst);
  EXPECT_EQ(gpu_heap::region_stats(Region::DispatchArgs).pool.capacity, 16 * KiB);
  EXPECT_EQ(gpu_heap::region_stats(Region::Index32).pool.capacity, 1 * MiB);
}

TEST_F(GpuHeapTest, BlockSizeRules) {
  EXPECT_EQ(gpu_heap::create_block(Region::DrawArgs, 0), BlockHandle::Invalid);
  EXPECT_EQ(count_logs(LOG_ERROR, "outside the"), 1u);
  EXPECT_EQ(gpu_heap::create_block(Region::DrawArgs, block_limit() + 1), BlockHandle::Invalid);
  EXPECT_EQ(count_logs(LOG_ERROR, "outside the"), 2u);

  const BlockHandle one = gpu_heap::create_block(Region::Index32, 1);
  const BlockHandle odd = gpu_heap::create_block(Region::Index32, 257);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_view(one).size, 256u);
  EXPECT_EQ(gpu_heap::block_view(odd).size, 512u);

  for (uint32_t live = 2; live < gpu_heap::MaxBlocks; ++live) {
    ASSERT_NE(gpu_heap::create_block(Region::DrawArgs, 256), BlockHandle::Invalid) << live;
  }
  EXPECT_EQ(count_logs(LOG_ERROR, "blocks are in use"), 0u);
  EXPECT_EQ(gpu_heap::create_block(Region::DrawArgs, 256), BlockHandle::Invalid);
  EXPECT_EQ(count_logs(LOG_ERROR, "blocks are in use"), 1u);
  EXPECT_EQ(count_logs(LOG_ERROR, "outside the"), 2u);
}

TEST_F(GpuHeapTest, BurstCreatesRegionWithoutGrowth) {
  std::vector<BlockHandle> blocks;
  for (int i = 0; i < 40; ++i) {
    blocks.push_back(gpu_heap::create_block(Region::DrawArgs, 4 * KiB));
  }
  push_validation_scope();
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  for (const BlockHandle block : blocks) {
    EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);
  }
  const RegionStats stats = gpu_heap::region_stats(Region::DrawArgs);
  EXPECT_EQ(stats.pool.capacity, 256 * KiB);
  EXPECT_EQ(stats.pool.growths, 0u);
  EXPECT_EQ(stats.blocks, 40u);
  EXPECT_EQ(stats.clearedBytes, 160 * KiB);
  EXPECT_EQ(gpu_heap::boundary_submits(), 1u);
  EXPECT_EQ(gpu_heap::growth_submits(), 0u);
}

TEST_F(GpuHeapTest, GrowthAndClearsShareOneSubmit) {
  const BlockHandle first = gpu_heap::create_block(Region::DrawArgs, 256);
  gpu_heap::begin_frame_worker();
  const BlockView before = gpu_heap::block_view(first);
  ASSERT_TRUE(before);
  EXPECT_EQ(gpu_heap::boundary_submits(), 1u);

  std::vector<BlockHandle> large;
  for (int i = 0; i < 3; ++i) {
    large.push_back(gpu_heap::create_block(Region::DrawArgs, 32 * KiB));
  }
  push_validation_scope();
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  const BlockView after = gpu_heap::block_view(first);
  ASSERT_TRUE(after);
  EXPECT_EQ(after.offset, before.offset);
  EXPECT_NE(after.buffer.Get(), before.buffer.Get());
  EXPECT_EQ(after.generation, 2u);
  for (const BlockHandle block : large) {
    const BlockView view = gpu_heap::block_view(block);
    ASSERT_TRUE(view);
    EXPECT_EQ(view.buffer.Get(), after.buffer.Get());
    EXPECT_EQ(view.generation, 1u);
  }
  const RegionStats stats = gpu_heap::region_stats(Region::DrawArgs);
  EXPECT_EQ(stats.pool.growths, 1u);
  EXPECT_EQ(stats.pool.capacity, 256 * KiB);
  EXPECT_EQ(gpu_heap::growth_submits(), 1u);
  EXPECT_EQ(gpu_heap::boundary_submits(), 2u);
}

TEST_F(GpuHeapTest, DestroyedBlockFreesAtBoundary) {
  const BlockHandle keep = gpu_heap::create_block(Region::DrawArgs, 256);
  const BlockHandle block = gpu_heap::create_block(Region::DrawArgs, 256);
  gpu_heap::begin_frame_worker();
  const BlockView view = gpu_heap::block_view(block);
  ASSERT_TRUE(view);

  gpu_heap::destroy_block(block);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);
  EXPECT_EQ(gpu_heap::block_view(block).offset, view.offset);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Invalid);
  const BlockView gone = gpu_heap::block_view(block);
  EXPECT_FALSE(gone);
  EXPECT_FALSE(gone.buffer);
  EXPECT_EQ(gone.size, 0u);

  const BlockHandle reused = gpu_heap::create_block(Region::DrawArgs, 256);
  EXPECT_NE(reused, block);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_view(reused).offset, view.offset);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Invalid);
  EXPECT_EQ(gpu_heap::block_state(keep), BlockState::Ready);
}

TEST_F(GpuHeapTest, EmptyRegionIsReleased) {
  const BlockHandle block = gpu_heap::create_block(Region::Index32, 256);
  gpu_heap::begin_frame_worker();
  ASSERT_TRUE(gpu_heap::region_stats(Region::Index32).created);
  EXPECT_GT(gpu_heap::total_bytes(), 0u);

  gpu_heap::destroy_block(block);
  gpu_heap::begin_frame_worker();
  RegionStats stats = gpu_heap::region_stats(Region::Index32);
  EXPECT_FALSE(stats.created);
  EXPECT_EQ(stats.releases, 1u);
  EXPECT_EQ(stats.blocks, 0u);
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);

  const BlockHandle again = gpu_heap::create_block(Region::Index32, 256);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(again), BlockState::Ready);
  stats = gpu_heap::region_stats(Region::Index32);
  EXPECT_TRUE(stats.created);
  EXPECT_EQ(stats.releases, 1u);
  EXPECT_EQ(gpu_heap::total_bytes(), 1 * MiB);
}

TEST_F(GpuHeapTest, RegionAtLimitFailsBlock) {
  Caps limited = caps();
  limited.limits.maxStorageBufferBindingSize = 64 * KiB;
  webgpu::set_caps(std::move(limited));
  const BlockHandle fits = gpu_heap::create_block(Region::DrawArgs, 48 * KiB);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(fits), BlockState::Ready);

  const BlockHandle failed = gpu_heap::create_block(Region::DrawArgs, 48 * KiB);
  ASSERT_NE(failed, BlockHandle::Invalid);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(failed), BlockState::Failed);
  EXPECT_FALSE(gpu_heap::block_view(failed));
  EXPECT_EQ(count_logs(LOG_WARNING, "failed at the"), 1u);
  RegionStats stats = gpu_heap::region_stats(Region::DrawArgs);
  EXPECT_EQ(stats.failed, 1u);
  EXPECT_EQ(stats.blocks, 1u);
  EXPECT_EQ(stats.pool.capacity, 64 * KiB);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(failed), BlockState::Failed);

  gpu_heap::destroy_block(failed);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(failed), BlockState::Invalid);
  stats = gpu_heap::region_stats(Region::DrawArgs);
  EXPECT_EQ(stats.failed, 0u);
  EXPECT_EQ(stats.blocks, 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "failed at the"), 1u);
}

TEST_F(GpuHeapTest, RequestsApplyByRecordingFrame) {
  BlockHandle block = BlockHandle::Invalid;
  gpu_heap::set_recording_frame(5);
  on_game_thread([&] { block = gpu_heap::create_block(Region::DrawArgs, 256); });
  gpu_heap::begin_frame_worker(5);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Pending);
  gpu_heap::begin_frame_worker(6);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);

  gpu_heap::set_recording_frame(7);
  on_game_thread([&] { gpu_heap::destroy_block(block); });
  gpu_heap::begin_frame_worker(7);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);
  gpu_heap::begin_frame_worker(8);
  EXPECT_EQ(gpu_heap::block_state(block), BlockState::Invalid);

  gpu_heap::set_recording_frame(8);
  BlockHandle late = BlockHandle::Invalid;
  on_game_thread([&] { late = gpu_heap::create_block(Region::DrawArgs, 256); });
  gpu_heap::set_recording_frame(9);
  gpu_heap::begin_frame_worker(9);
  EXPECT_EQ(gpu_heap::block_state(late), BlockState::Ready);
}

TEST_F(GpuHeapTest, LaggingWorkerSeesSameFrames) {
  for (const bool held : {false, true}) {
    SCOPED_TRACE(held ? "held worker" : "eager worker");
    gpu_heap::shutdown();
    std::atomic_uint32_t created{0};
    std::array<bool, 7> visible{};
    std::mutex gateMutex;
    std::condition_variable gate;
    uint64_t requested = 0;
    {
      TestWorker worker;
      for (uint64_t frame = 1; frame <= 6; ++frame) {
        gpu_heap::set_recording_frame(frame);
        worker.post([&, frame] {
          if (held) {
            std::unique_lock lock{gateMutex};
            gate.wait_for(lock, WorkerTimeout, [&] { return requested >= frame; });
          }
          gpu_heap::begin_frame_worker(frame);
          const auto block = static_cast<BlockHandle>(created.load(std::memory_order_acquire));
          visible[frame] = static_cast<bool>(gpu_heap::block_view(block));
        });
        if (!held) {
          ASSERT_TRUE(worker.drain());
        }
        if (frame == 2) {
          const BlockHandle block = gpu_heap::create_block(Region::DrawArgs, 256);
          created.store(static_cast<uint32_t>(block), std::memory_order_release);
        } else if (frame == 4) {
          gpu_heap::destroy_block(static_cast<BlockHandle>(created.load(std::memory_order_acquire)));
        }
        {
          std::lock_guard lock{gateMutex};
          requested = frame;
        }
        gate.notify_all();
      }
      ASSERT_TRUE(worker.drain());
    }
    EXPECT_EQ(visible, (std::array<bool, 7>{false, false, false, true, true, false, false}));
  }
}

TEST_F(GpuHeapTest, WorkerRequestsUseItsOwnFrame) {
  TestWorker worker;
  worker.post([] { gpu_heap::begin_frame_worker(5); });
  ASSERT_TRUE(worker.drain());
  gpu_heap::set_recording_frame(6);
  BlockState afterBoundary = BlockState::Invalid;
  worker.post([&] {
    const BlockHandle block = gpu_heap::create_block(Region::DrawArgs, 256);
    gpu_heap::begin_frame_worker(6);
    afterBoundary = gpu_heap::block_state(block);
  });
  ASSERT_TRUE(worker.drain());
  EXPECT_EQ(afterBoundary, BlockState::Ready);
}

TEST_F(GpuHeapTest, BlocksCreatedFromAnotherThread) {
  std::vector<BlockHandle> blocks(64, BlockHandle::Invalid);
  std::thread creator([&] {
    for (BlockHandle& block : blocks) {
      block = gpu_heap::create_block(Region::Index32, 1 * KiB);
    }
  });
  creator.join();
  gpu_heap::begin_frame_worker();
  for (const BlockHandle block : blocks) {
    EXPECT_EQ(gpu_heap::block_state(block), BlockState::Ready);
  }
  EXPECT_EQ(gpu_heap::region_stats(Region::Index32).blocks, 64u);
}

TEST_F(GpuHeapTest, ShutdownResetsBlocks) {
  const BlockHandle draw = gpu_heap::create_block(Region::DrawArgs, 256);
  const BlockHandle index = gpu_heap::create_block(Region::Index32, 256);
  gpu_heap::begin_frame_worker();
  const BlockHandle pending = gpu_heap::create_block(Region::DispatchArgs, 256);
  ASSERT_EQ(gpu_heap::block_state(draw), BlockState::Ready);

  gpu_heap::shutdown();
  for (const BlockHandle block : {draw, index, pending}) {
    EXPECT_EQ(gpu_heap::block_state(block), BlockState::Invalid);
  }
  for (const Region region : AllRegions) {
    const RegionStats stats = gpu_heap::region_stats(region);
    EXPECT_FALSE(stats.created);
    EXPECT_EQ(stats.blocks + stats.pending + stats.failed, 0u);
    EXPECT_EQ(stats.clearedBytes, 0u);
  }
  EXPECT_EQ(gpu_heap::total_bytes(), 0u);
  EXPECT_EQ(gpu_heap::boundary_submits(), 0u);
  const BlockHandle fresh = gpu_heap::create_block(Region::DrawArgs, 256);
  EXPECT_NE(fresh, draw);
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(fresh), BlockState::Ready);
  EXPECT_EQ(gpu_heap::block_state(draw), BlockState::Invalid);
}

constexpr std::string_view FillBlockWgsl = R"(
@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&words)) {
    words[id.x] = 0xA5000000u | id.x;
  }
}
)";

class GpuHeapGpu : public ::testing::Test {
protected:
  void SetUp() override {
    const test::TestDeviceOptions options{
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
    };
    std::string error;
    m_device = test::create_test_device(options, error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(m_device->adapterInfo.backendType));
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, options);
    gpu_heap::shutdown();
  }

  void TearDown() override {
    gpu_heap::shutdown();
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  void fill(const BlockView& view) const {
    const std::string code{FillBlockWgsl};
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    const wgpu::ComputePipelineDescriptor pipelineDescriptor{
        .compute = {.module = m_device->device.CreateShaderModule(&moduleDescriptor), .entryPoint = "main"},
    };
    const wgpu::ComputePipeline pipeline = m_device->device.CreateComputePipeline(&pipelineDescriptor);
    const wgpu::BindGroupEntry entry{.binding = 0, .buffer = view.buffer, .offset = view.offset, .size = view.size};
    const wgpu::BindGroupDescriptor groupDescriptor{
        .layout = pipeline.GetBindGroupLayout(0),
        .entryCount = 1,
        .entries = &entry,
    };
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, m_device->device.CreateBindGroup(&groupDescriptor));
    pass.DispatchWorkgroups(static_cast<uint32_t>(view.size / sizeof(uint32_t) / 64));
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(GpuHeapGpu, BlockZeroedAndPreservedAcrossGrowth) {
  const BlockHandle block = gpu_heap::create_block(Region::DrawArgs, 256);
  gpu_heap::begin_frame_worker();
  const BlockView before = gpu_heap::block_view(block);
  ASSERT_TRUE(before);
  fill(before);

  ASSERT_NE(gpu_heap::create_block(Region::DrawArgs, 64 * KiB), BlockHandle::Invalid);
  gpu_heap::begin_frame_worker();
  const BlockView after = gpu_heap::block_view(block);
  ASSERT_TRUE(after);
  ASSERT_NE(after.buffer.Get(), before.buffer.Get());
  EXPECT_EQ(after.offset, before.offset);
  const auto kept = test::read_buffer(*m_device, after.buffer, after.offset, after.size);
  ASSERT_TRUE(kept);
  ASSERT_EQ(kept->size(), 64u);
  for (uint32_t i = 0; i < kept->size(); ++i) {
    EXPECT_EQ((*kept)[i], 0xA5000000u | i) << i;
  }

  gpu_heap::destroy_block(block);
  const BlockHandle reused = gpu_heap::create_block(Region::DrawArgs, 256);
  gpu_heap::begin_frame_worker();
  const BlockView view = gpu_heap::block_view(reused);
  ASSERT_TRUE(view);
  ASSERT_EQ(view.offset, before.offset);
  const auto cleared = test::read_buffer(*m_device, view.buffer, view.offset, view.size);
  ASSERT_TRUE(cleared);
  for (uint32_t i = 0; i < cleared->size(); ++i) {
    EXPECT_EQ((*cleared)[i], 0u) << i;
  }
}
} // namespace
} // namespace aurora::render

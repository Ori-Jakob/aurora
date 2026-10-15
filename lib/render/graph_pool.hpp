#pragma once

#include "graph.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
class PhysicalAllocator;

namespace testing {
void reset_transient_pool(std::unique_ptr<PhysicalAllocator> allocator);
} // namespace testing

class PhysicalTexture {
public:
  uint64_t serial = 0;
  wgpu::Texture texture;
  TexKey key;
  uint64_t bytes = 0;

  const wgpu::TextureView& binding_view(wgpu::TextureViewDimension dimension) const;
  const wgpu::TextureView& mip_view(uint32_t mip, wgpu::TextureViewDimension dimension) const;
  const wgpu::TextureView& attachment_view(uint32_t mip, uint32_t layerOrSlice) const;

private:
  struct CachedView {
    uint8_t kind = 0;
    uint32_t mip = 0;
    uint32_t layer = 0;
    wgpu::TextureViewDimension dimension = wgpu::TextureViewDimension::Undefined;
    wgpu::TextureView view;
  };
  const wgpu::TextureView& cached_view(uint8_t kind, uint32_t mip, uint32_t layer,
                                       wgpu::TextureViewDimension dimension) const;
  // A deque, so a view handed out stays put when later views are added
  mutable std::deque<CachedView> m_views;
};

struct PhysicalBuffer {
  uint64_t serial = 0;
  wgpu::Buffer buffer;
  BufKey key;
  uint64_t bytes = 0;
};

class PhysicalAllocator {
public:
  virtual ~PhysicalAllocator() = default;
  virtual PhysicalTexture create_texture(const TexKey& key, std::string_view label) = 0;
  virtual PhysicalBuffer create_buffer(const BufKey& key, std::string_view label) = 0;
};

std::unique_ptr<PhysicalAllocator> make_dawn_allocator();
uint64_t next_resource_serial() noexcept;

struct PoolStats {
  uint64_t claimedBytes = 0;
  uint64_t idleBytes = 0;
  uint64_t historyBytes = 0;
  uint64_t logicalBytes = 0;
  uint64_t slotBytes = 0;
  uint32_t textures = 0;
  uint32_t buffers = 0;
  uint32_t idle = 0;
  uint64_t created = 0;
  uint64_t released = 0;
  uint32_t unpoisonable = 0;
};

using ReleaseListener = void (*)(uint64_t serial);

class TransientPool {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::seconds IdleRelease{2};

  explicit TransientPool(std::unique_ptr<PhysicalAllocator> allocator);
  ~TransientPool();
  TransientPool(const TransientPool&) = delete;
  TransientPool& operator=(const TransientPool&) = delete;

  void materialize(const std::shared_ptr<const CompiledGraph>& graph, Clock::time_point now);
  const PhysicalTexture& texture(SlotId slot) const;
  const PhysicalBuffer& buffer(SlotId slot) const;
  const PhysicalTexture& history_write(uint16_t historySlot) const;
  const PhysicalTexture& history_prev(uint16_t historySlot) const;
  void mark_history_written(uint16_t historySlot) noexcept;
  void apply_history_swaps() noexcept;
  [[nodiscard]] bool history_fresh(uint16_t historySlot) const noexcept;
  [[nodiscard]] std::vector<uint64_t> serials() const;
  void collect(Clock::time_point now, bool pressure);
  void set_release_listener(ReleaseListener listener) noexcept;
  [[nodiscard]] PoolStats stats() const noexcept;
  [[nodiscard]] uint64_t total_bytes() const noexcept;
  [[nodiscard]] const CompiledGraph* materialized_graph() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  void clear();

private:
  friend void testing::reset_transient_pool(std::unique_ptr<PhysicalAllocator> allocator);

  struct Physical {
    bool isBuffer = false;
    PhysicalTexture texture;
    PhysicalBuffer buffer;
    std::string anchor;
    bool claimed = false;
    Clock::time_point idleSince{};
  };

  struct HistoryPair {
    std::string name;
    std::array<PhysicalTexture, 2> textures;
    uint8_t read = 0;
    bool pendingSwap = false;
    bool fresh = false;
    Clock::time_point idleSince{};
  };

  struct PublishedStats {
    std::atomic_uint64_t claimedBytes{0};
    std::atomic_uint64_t idleBytes{0};
    std::atomic_uint64_t historyBytes{0};
    std::atomic_uint64_t logicalBytes{0};
    std::atomic_uint64_t slotBytes{0};
    std::atomic_uint32_t textures{0};
    std::atomic_uint32_t buffers{0};
    std::atomic_uint32_t idle{0};
    std::atomic_uint64_t created{0};
    std::atomic_uint64_t released{0};
  };

  Physical& create_physical(const SlotPlan::Slot& slot);
  void release(const Physical& physical, std::vector<uint64_t>& released);
  void release(const HistoryPair& pair, std::vector<uint64_t>& released);
  void notify(const std::vector<uint64_t>& released) const;
  void publish() noexcept;

  std::unique_ptr<PhysicalAllocator> m_allocator;
  std::shared_ptr<const CompiledGraph> m_graph;
  std::vector<std::unique_ptr<Physical>> m_physicals;
  std::vector<Physical*> m_slots;
  std::vector<HistoryPair> m_histories;
  std::vector<HistoryPair> m_idleHistories;
  GraphExtent m_efbSize;
  GraphExtent m_outputSize;
  std::atomic<ReleaseListener> m_listener{nullptr};
  uint64_t m_created = 0;
  uint64_t m_released = 0;
  PublishedStats m_published;
};

TransientPool& transient_pool();
void transient_pool_tick(const std::shared_ptr<const CompiledGraph>& graph, TransientPool::Clock::time_point now);
uint64_t transient_pool_bytes() noexcept;
uint64_t memory_budget() noexcept;
bool memory_pressure(uint64_t rendererBytes) noexcept;
void force_memory_pressure() noexcept;

bool debug_poison_enabled() noexcept;
void set_debug_poison(bool enabled) noexcept;

class Poisoner {
public:
  void encode(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture);
  void encode(const wgpu::CommandEncoder& cmd, const PhysicalBuffer& buffer, uint64_t size);

private:
  struct ClearKernel {
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    wgpu::TextureViewDimension dimension = wgpu::TextureViewDimension::Undefined;
    wgpu::BindGroupLayout layout;
    wgpu::ComputePipeline pipeline;
  };

  const wgpu::Buffer& source(uint64_t size);
  const ClearKernel* clear_kernel(wgpu::TextureFormat format, wgpu::TextureViewDimension dimension);
  void render_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture) const;
  void compute_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture);
  void copy_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture);
  void fill(const wgpu::CommandEncoder& cmd, const wgpu::Buffer& buffer, uint64_t size);

  wgpu::Buffer m_source;
  uint64_t m_sourceSize = 0;
  std::vector<ClearKernel> m_kernels;
  wgpu::BindGroupLayout m_fillLayout;
  wgpu::ComputePipeline m_fill;
};

void poison_before(const wgpu::CommandEncoder& cmd, const StageData& stage, NodeId node);
void after_pass(const StageData& stage, NodeId node);
} // namespace aurora::render::detail

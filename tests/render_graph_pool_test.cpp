#include <gtest/gtest.h>

#include "render_graph_planned.hpp"
#include "render_test_device.hpp"

#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/graph_pool.hpp"
#include "render/graph_slots.hpp"
#include "render/module.hpp"
#include "render/stage.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
using namespace std::chrono_literals;
using wgpu::BufferUsage;
using wgpu::TextureFormat;
using wgpu::TextureUsage;
using Clock = TransientPool::Clock;

constexpr uint64_t MiB = 1024 * 1024;

struct AllocatorLog {
  std::vector<TexKey> textures;
  std::vector<BufKey> buffers;

  [[nodiscard]] size_t calls() const noexcept { return textures.size() + buffers.size(); }
};

// Logs every request, and forwards it to a device allocator when it has one
class LoggingAllocator final : public PhysicalAllocator {
public:
  explicit LoggingAllocator(AllocatorLog& log, std::unique_ptr<PhysicalAllocator> device = nullptr)
  : m_log(log), m_device(std::move(device)) {}

  PhysicalTexture create_texture(const TexKey& key, std::string_view label) override {
    m_log.textures.push_back(key);
    if (m_device) {
      return m_device->create_texture(key, label);
    }
    PhysicalTexture physical;
    physical.serial = next_resource_serial();
    physical.key = key;
    physical.bytes = key_bytes(key);
    return physical;
  }

  PhysicalBuffer create_buffer(const BufKey& key, std::string_view label) override {
    m_log.buffers.push_back(key);
    if (m_device) {
      return m_device->create_buffer(key, label);
    }
    return {.serial = next_resource_serial(), .key = key, .bytes = key.bucketSize};
  }

private:
  AllocatorLog& m_log;
  std::unique_ptr<PhysicalAllocator> m_device;
};

std::vector<uint64_t> g_released;

void record_release(uint64_t serial) { g_released.push_back(serial); }

bool was_released(uint64_t serial) { return std::ranges::find(g_released, serial) != g_released.end(); }

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  uint16_t order = 0;
  std::function<void(InternalPassBuilder&)> body;
  bool sink = false;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  if (pass.body) {
    pass.body(builder);
  }
  if (pass.sink) {
    builder.side_effect();
  }
}

GraphInputs test_inputs(GraphExtent efb = {1280, 720}, GraphExtent output = {1280, 720}) {
  GraphInputs inputs;
  inputs.env.limits = planned::full_limits();
  inputs.env.efbSize = efb;
  inputs.env.outputSize = output;
  inputs.env.gxFormats = planned::gx_formats_tier_p();
  inputs.env.gxUsage = planned::gx_usage_future();
  return inputs;
}

std::shared_ptr<const CompiledGraph> compile(const std::vector<TestPass>& passes, GraphInputs inputs = test_inputs()) {
  std::vector<PassDesc> descs;
  for (const TestPass& pass : passes) {
    descs.push_back({
        .name = pass.name,
        .stage = pass.stage,
        .kind = pass.kind,
        .order = pass.order,
        .declare = declare_test_pass,
        .state = &pass,
    });
  }
  inputs.passes = descs;
  return compile_graph(inputs);
}

void append_chain(std::vector<TestPass>& passes, const std::string& name, const TexDesc& desc) {
  const auto order = static_cast<uint16_t>(passes.size());
  passes.push_back({
      .name = name + ".make",
      .order = order,
      .body = [name, desc](InternalPassBuilder& b) { b.create(name, desc, Access::StorageWrite); },
  });
  passes.push_back({
      .name = name + ".read",
      .order = static_cast<uint16_t>(order + 1),
      .body = [name](InternalPassBuilder& b) { b.use(name, Access::Sampled); },
      .sink = true,
  });
}

void append_history(std::vector<TestPass>& passes, const std::string& name, const TexDesc& desc) {
  const auto order = static_cast<uint16_t>(passes.size());
  passes.push_back({
      .name = name + ".own",
      .order = order,
      .body = [name, desc](InternalPassBuilder& b) { b.history(name, desc, Access::StorageWrite); },
  });
  passes.push_back({
      .name = name + ".prev",
      .order = static_cast<uint16_t>(order + 1),
      .body = [name](InternalPassBuilder& b) { b.use_prev(name); },
      .sink = true,
  });
}

TexDesc efb_tex(TextureFormat format) { return {.format = format, .sizeClass = SizeClass::Efb}; }

TexDesc output_tex(TextureFormat format) { return {.format = format, .sizeClass = SizeClass::Output}; }

TexDesc absolute_tex(TextureFormat format, uint32_t width, uint32_t height, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

SlotId slot_of(const CompiledGraph& graph, std::string_view name) {
  const ResourceId id = graph.find_resource(name);
  EXPECT_NE(id, InvalidResource) << name;
  return id != InvalidResource ? graph.slots.resourceSlot[id] : InvalidSlot;
}

uint16_t history_of(const CompiledGraph& graph, std::string_view name) {
  const ResourceId id = graph.find_resource(name);
  EXPECT_NE(id, InvalidResource) << name;
  return id != InvalidResource ? graph.resources[id].historySlot : InvalidResource;
}

uint64_t serial_of(const CompiledGraph& graph, std::string_view name) {
  return transient_pool().texture(slot_of(graph, name)).serial;
}

std::map<std::string, uint64_t> serials_by_name(const CompiledGraph& graph) {
  std::map<std::string, uint64_t> out;
  const TransientPool& pool = transient_pool();
  for (size_t r = 0; r < graph.resources.size(); ++r) {
    const SlotId slot = graph.slots.resourceSlot[r];
    if (slot == InvalidSlot) {
      continue;
    }
    const bool buffer = graph.slots.slots[slot].isBuffer;
    out.emplace(graph.resources[r].name, buffer ? pool.buffer(slot).serial : pool.texture(slot).serial);
  }
  return out;
}

const TexDesc HdrDesc = efb_tex(TextureFormat::RGBA16Float);
const TexDesc MaskDesc = absolute_tex(TextureFormat::R32Float, 256, 256);

std::vector<TestPass> both_chains() {
  std::vector<TestPass> passes;
  append_chain(passes, "x", HdrDesc);
  append_chain(passes, "y", MaskDesc);
  return passes;
}

std::vector<TestPass> x_chain() {
  std::vector<TestPass> passes;
  append_chain(passes, "x", HdrDesc);
  return passes;
}

class GraphPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset_transient_pool(std::make_unique<LoggingAllocator>(m_log));
    g_released.clear();
    transient_pool().set_release_listener(record_release);
  }

  void TearDown() override { testing::reset_transient_pool(nullptr); }

  AllocatorLog m_log;
  const Clock::time_point t0 = Clock::time_point{} + 1h;
};

TEST_F(GraphPoolTest, ToggleBackAllocatesNothing) {
  auto passesA = both_chains();
  append_history(passesA, "h", HdrDesc);
  const auto passesB = x_chain();
  const auto a = compile(passesA);
  const auto b = compile(passesB);
  transient_pool_tick(a, t0);
  const uint64_t created = transient_pool().stats().created;
  EXPECT_EQ(created, 4u);
  const uint16_t h = history_of(*a, "h");
  const uint64_t prev = transient_pool().history_prev(h).serial;
  const uint64_t write = transient_pool().history_write(h).serial;
  transient_pool_tick(b, t0 + 100ms);
  EXPECT_EQ(transient_pool().stats().idle, 3u);
  EXPECT_EQ(transient_pool().stats().historyBytes, 0u);
  transient_pool_tick(a, t0 + 900ms);
  EXPECT_EQ(transient_pool().stats().created, created);
  EXPECT_EQ(transient_pool().stats().idle, 0u);
  EXPECT_EQ(m_log.calls(), created);
  EXPECT_TRUE(g_released.empty());
  EXPECT_EQ(transient_pool().history_prev(h).serial, prev);
  EXPECT_EQ(transient_pool().history_write(h).serial, write);
}

TEST_F(GraphPoolTest, ToggleBackAfterIdleReleaseAllocates) {
  const auto passesA = both_chains();
  const auto passesB = x_chain();
  const auto a = compile(passesA);
  const auto b = compile(passesB);
  transient_pool_tick(a, t0);
  const uint64_t created = transient_pool().stats().created;
  transient_pool_tick(b, t0);
  transient_pool_tick(b, t0 + 3s);
  EXPECT_EQ(g_released.size(), 1u);
  transient_pool_tick(a, t0 + 3s);
  EXPECT_GT(transient_pool().stats().created, created);
}

TEST_F(GraphPoolTest, IdleReleasedAfterTwoSeconds) {
  const auto passesA = both_chains();
  const auto passesB = x_chain();
  const auto a = compile(passesA);
  const auto b = compile(passesB);
  TransientPool& pool = transient_pool();
  pool.materialize(a, t0);
  const uint64_t y = serial_of(*a, "y");
  pool.materialize(b, t0);
  pool.collect(t0 + 1900ms, false);
  EXPECT_TRUE(g_released.empty());
  EXPECT_EQ(pool.stats().idle, 1u);
  pool.collect(t0 + 2s, false);
  EXPECT_EQ(g_released, (std::vector<uint64_t>{y}));
  EXPECT_EQ(pool.stats().idle, 0u);
  EXPECT_EQ(pool.stats().released, 1u);
}

TEST_F(GraphPoolTest, PressureReleasesIdleAtOnce) {
  const auto passesA = both_chains();
  const auto passesB = x_chain();
  const auto a = compile(passesA);
  const auto b = compile(passesB);
  TransientPool& pool = transient_pool();
  pool.materialize(a, t0);
  const uint64_t x = serial_of(*a, "x");
  const uint64_t y = serial_of(*a, "y");
  pool.materialize(b, t0);
  pool.collect(t0, true);
  EXPECT_EQ(g_released, (std::vector<uint64_t>{y}));
  EXPECT_FALSE(was_released(x));
  EXPECT_EQ(serial_of(*b, "x"), x);
  const PoolStats stats = pool.stats();
  EXPECT_EQ(stats.idle, 0u);
  EXPECT_EQ(stats.textures, 1u);
  EXPECT_EQ(stats.idleBytes, 0u);
  EXPECT_EQ(stats.claimedBytes, key_bytes(b->slots.slots[slot_of(*b, "x")].tex));
}

TEST_F(GraphPoolTest, EfbResizeRematerializesEfbSlots) {
  const auto passes = both_chains();
  const auto small = compile(passes, test_inputs({1280, 720}));
  const auto large = compile(passes, test_inputs({1920, 1080}));
  TransientPool& pool = transient_pool();
  pool.materialize(small, t0);
  const uint64_t x = serial_of(*small, "x");
  const uint64_t y = serial_of(*small, "y");
  pool.materialize(large, t0 + 16ms);
  EXPECT_EQ(g_released, (std::vector<uint64_t>{x}));
  EXPECT_NE(serial_of(*large, "x"), x);
  EXPECT_EQ(transient_pool().texture(slot_of(*large, "x")).key.width, 1920u);
  EXPECT_EQ(serial_of(*large, "y"), y);
  EXPECT_EQ(pool.stats().idle, 0u);
}

TEST_F(GraphPoolTest, OutputResizeOnlyTouchesOutputClass) {
  auto passes = both_chains();
  append_chain(passes, "o", output_tex(TextureFormat::RGBA8Unorm));
  const auto before = compile(passes, test_inputs({1280, 720}, {1280, 720}));
  const auto after = compile(passes, test_inputs({1280, 720}, {2560, 1440}));
  TransientPool& pool = transient_pool();
  pool.materialize(before, t0);
  const auto serials = serials_by_name(*before);
  pool.materialize(after, t0 + 16ms);
  EXPECT_EQ(serial_of(*after, "x"), serials.at("x"));
  EXPECT_EQ(serial_of(*after, "y"), serials.at("y"));
  EXPECT_NE(serial_of(*after, "o"), serials.at("o"));
  EXPECT_EQ(g_released, (std::vector<uint64_t>{serials.at("o")}));
}

TEST_F(GraphPoolTest, SameGraphIsNoOp) {
  const auto passes = both_chains();
  const auto graph = compile(passes);
  TransientPool& pool = transient_pool();
  pool.materialize(graph, t0);
  const size_t calls = m_log.calls();
  pool.materialize(graph, t0 + 16ms);
  EXPECT_EQ(m_log.calls(), calls);
  EXPECT_EQ(pool.stats().created, calls);
  EXPECT_TRUE(g_released.empty());
  EXPECT_EQ(pool.materialized_graph(), graph.get());
}

TEST_F(GraphPoolTest, EqualHashRecompileKeepsPhysicals) {
  auto passes = both_chains();
  append_history(passes, "h", HdrDesc);
  const auto first = compile(passes);
  const auto second = compile(passes);
  ASSERT_NE(first, second);
  ASSERT_EQ(first->graphHash, second->graphHash);
  TransientPool& pool = transient_pool();
  pool.materialize(first, t0);
  const auto serials = serials_by_name(*first);
  const uint64_t prev = pool.history_prev(history_of(*first, "h")).serial;
  const size_t calls = m_log.calls();
  pool.materialize(second, t0 + 16ms);
  EXPECT_EQ(pool.materialized_graph(), second.get());
  EXPECT_EQ(m_log.calls(), calls);
  EXPECT_EQ(serials_by_name(*second), serials);
  EXPECT_EQ(pool.history_prev(history_of(*second, "h")).serial, prev);
  EXPECT_TRUE(g_released.empty());
}

TEST_F(GraphPoolTest, EmptyGraphReleasesEverything) {
  auto passes = both_chains();
  append_history(passes, "h", HdrDesc);
  const auto graph = compile(passes);
  transient_pool_tick(graph, t0);
  const PoolStats live = transient_pool().stats();
  EXPECT_EQ(live.created, 4u);
  EXPECT_EQ(live.textures, 4u);
  EXPECT_GT(live.historyBytes, 0u);

  transient_pool_tick(empty_graph(), t0);
  PoolStats stats = transient_pool().stats();
  EXPECT_EQ(stats.idle, 4u);
  EXPECT_EQ(stats.claimedBytes, 0u);
  EXPECT_EQ(stats.historyBytes, 0u);
  EXPECT_EQ(stats.idleBytes, live.claimedBytes + live.historyBytes);
  EXPECT_TRUE(g_released.empty());

  transient_pool_tick(empty_graph(), t0 + 2s);
  stats = transient_pool().stats();
  EXPECT_EQ(stats.released, stats.created);
  EXPECT_EQ(g_released.size(), stats.created);
  EXPECT_EQ(stats.claimedBytes, 0u);
  EXPECT_EQ(stats.idleBytes, 0u);
  EXPECT_EQ(stats.historyBytes, 0u);
  EXPECT_EQ(stats.logicalBytes, 0u);
  EXPECT_EQ(stats.slotBytes, 0u);
  EXPECT_EQ(stats.textures, 0u);
  EXPECT_EQ(stats.buffers, 0u);
  EXPECT_EQ(stats.idle, 0u);
  EXPECT_TRUE(transient_pool().empty());
  EXPECT_EQ(transient_pool_bytes(), 0u);
}

TEST_F(GraphPoolTest, EmptyGraphKeepsLastExtents) {
  const auto passes = both_chains();
  const auto small = compile(passes, test_inputs({1280, 720}));
  const auto large = compile(passes, test_inputs({1920, 1080}));
  TransientPool& pool = transient_pool();
  pool.materialize(small, t0);
  const uint64_t x = serial_of(*small, "x");
  const uint64_t y = serial_of(*small, "y");
  pool.materialize(empty_graph(), t0 + 16ms);
  EXPECT_TRUE(g_released.empty());
  EXPECT_EQ(pool.stats().idle, 2u);
  pool.materialize(large, t0 + 1s);
  EXPECT_EQ(g_released, (std::vector<uint64_t>{x}));
  EXPECT_NE(serial_of(*large, "x"), x);
  EXPECT_EQ(serial_of(*large, "y"), y);
  EXPECT_EQ(pool.stats().idle, 0u);
}

TEST_F(GraphPoolTest, NeverMaterializedEmptyGraphDoesNothing) {
  for (int i = 0; i < 100; ++i) {
    transient_pool_tick(nullptr, t0 + i * 16ms);
  }
  EXPECT_EQ(transient_pool().materialized_graph(), nullptr);
  for (int i = 0; i < 100; ++i) {
    transient_pool_tick(empty_graph(), t0 + i * 16ms);
  }
  EXPECT_EQ(m_log.calls(), 0u);
  EXPECT_TRUE(transient_pool().empty());
  EXPECT_EQ(transient_pool().stats().created, 0u);

  // A graph whose passes were all culled has no plan either
  std::vector<TestPass> culled;
  append_chain(culled, "x", HdrDesc);
  culled.back().sink = false;
  const auto graph = compile(culled);
  ASSERT_TRUE(graph->slots.slots.empty());
  transient_pool_tick(graph, t0);
  EXPECT_EQ(m_log.calls(), 0u);
}

TEST_F(GraphPoolTest, HistorySwapsOnlyWhenOwnerRan) {
  std::vector<TestPass> passes;
  append_history(passes, "h", HdrDesc);
  const auto graph = compile(passes);
  TransientPool& pool = transient_pool();
  pool.materialize(graph, t0);
  const uint16_t h = history_of(*graph, "h");
  ASSERT_NE(h, InvalidResource);
  const uint64_t first = pool.history_prev(h).serial;
  const uint64_t second = pool.history_write(h).serial;
  ASSERT_NE(first, second);

  pool.mark_history_written(h);
  pool.apply_history_swaps();
  EXPECT_EQ(pool.history_prev(h).serial, second);
  EXPECT_EQ(pool.history_write(h).serial, first);

  pool.apply_history_swaps();
  EXPECT_EQ(pool.history_prev(h).serial, second);

  pool.mark_history_written(h);
  pool.apply_history_swaps();
  EXPECT_EQ(pool.history_prev(h).serial, first);
  EXPECT_EQ(pool.stats().historyBytes, 2 * key_bytes(graph->slots.histories[h].key));
}

TEST_F(GraphPoolTest, HistoryPairsSurviveEqualKeyRecompile) {
  std::vector<TestPass> passes;
  append_history(passes, "h", HdrDesc);
  const auto graph = compile(passes);
  auto wider = passes;
  append_chain(wider, "x", MaskDesc);
  const auto other = compile(wider);
  ASSERT_NE(graph->graphHash, other->graphHash);
  TransientPool& pool = transient_pool();
  pool.materialize(graph, t0);
  const uint16_t h = history_of(*graph, "h");
  pool.mark_history_written(h);
  pool.apply_history_swaps();
  const uint64_t prev = pool.history_prev(h).serial;
  const uint64_t write = pool.history_write(h).serial;
  pool.materialize(other, t0 + 16ms);
  const uint16_t moved = history_of(*other, "h");
  EXPECT_EQ(pool.history_prev(moved).serial, prev);
  EXPECT_EQ(pool.history_write(moved).serial, write);
  EXPECT_TRUE(g_released.empty());
}

TEST_F(GraphPoolTest, HistoryKeyChangeRecreatesPair) {
  std::vector<TestPass> passes;
  append_history(passes, "h", HdrDesc);
  const auto small = compile(passes, test_inputs({1280, 720}));
  const auto large = compile(passes, test_inputs({1920, 1080}));
  TransientPool& pool = transient_pool();
  pool.materialize(small, t0);
  const uint16_t h = history_of(*small, "h");
  pool.mark_history_written(h);
  pool.apply_history_swaps();
  const uint64_t oldPrev = pool.history_prev(h).serial;
  const uint64_t oldWrite = pool.history_write(h).serial;
  pool.materialize(large, t0 + 16ms);
  const uint16_t resized = history_of(*large, "h");
  const uint64_t prev = pool.history_prev(resized).serial;
  const uint64_t write = pool.history_write(resized).serial;
  EXPECT_NE(prev, oldPrev);
  EXPECT_NE(prev, oldWrite);
  EXPECT_NE(write, oldPrev);
  EXPECT_NE(write, oldWrite);
  // The pair is created #0 then #1, so prev being the older serial means the read index is 0
  EXPECT_LT(prev, write);
  EXPECT_EQ(pool.history_prev(resized).key.width, 1920u);
  EXPECT_TRUE(was_released(oldPrev));
  EXPECT_TRUE(was_released(oldWrite));
}

TEST_F(GraphPoolTest, StatsReadableFromAnotherThread) {
  auto passesA = both_chains();
  append_history(passesA, "h", HdrDesc);
  const auto passesB = x_chain();
  const auto a = compile(passesA);
  const auto b = compile(passesB);
  std::atomic_bool done = false;
  std::atomic_uint64_t reads = 0;
  std::thread reader([&] {
    while (!done.load(std::memory_order_relaxed)) {
      const PoolStats stats = transient_pool().stats();
      const uint64_t total = transient_pool_bytes();
      reads.fetch_add(1 + (stats.created > 1000000 ? total : 0), std::memory_order_relaxed);
    }
  });
  // The reader may not be scheduled before the ticks finish, so only start them once it has read
  while (reads.load(std::memory_order_relaxed) == 0) {
    std::this_thread::yield();
  }
  for (int i = 0; i < 300; ++i) {
    const auto now = t0 + i * 1s;
    transient_pool_tick(i % 2 == 0 ? a : b, now);
    if (i % 7 == 0) {
      force_memory_pressure();
    }
  }
  done.store(true, std::memory_order_relaxed);
  reader.join();
  EXPECT_GT(reads.load(), 0u);
  const PoolStats stats = transient_pool().stats();
  EXPECT_EQ(stats.claimedBytes + stats.idleBytes + stats.historyBytes, transient_pool_bytes());
  EXPECT_EQ(stats.created - stats.released, static_cast<uint64_t>(stats.textures + stats.buffers));
}

TEST_F(GraphPoolTest, BudgetFollowsVram) {
  webgpu::reset_caps();
  EXPECT_EQ(memory_budget(), 1024 * MiB);
  Caps deviceCaps;
  deviceCaps.vramSource = VramSource::MemoryHeaps;
  deviceCaps.vramBytes = 8192 * MiB;
  webgpu::set_caps(deviceCaps);
  EXPECT_EQ(memory_budget(), 2048 * MiB);
  deviceCaps.vramBytes = 1024 * MiB;
  webgpu::set_caps(deviceCaps);
  EXPECT_EQ(memory_budget(), 512 * MiB);
  deviceCaps.vramBytes = 4096 * MiB;
  webgpu::set_caps(deviceCaps);
  EXPECT_EQ(memory_budget(), 1024 * MiB);
  EXPECT_FALSE(memory_pressure(1024 * MiB));
  EXPECT_TRUE(memory_pressure(1024 * MiB + 1));
  webgpu::reset_caps();

  force_memory_pressure();
  EXPECT_TRUE(memory_pressure(0));
  EXPECT_FALSE(memory_pressure(0));
}

class GraphPoolDeviceTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    m_device = test::create_test_device({}, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, test::TestDeviceOptions{});
    testing::reset_transient_pool(nullptr);
  }

  void TearDown() override {
    testing::reset_transient_pool(nullptr);
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  void push_validation_scope() const { m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation); }

  [[nodiscard]] std::optional<std::string> submit(const wgpu::CommandEncoder& encoder) const {
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    return test::pop_validation_error(*m_device);
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TexKey texture_key(TextureFormat format, wgpu::TextureUsage usage, uint32_t width = 64, uint32_t height = 64,
                   uint32_t depthOrLayers = 1, uint32_t mips = 1,
                   wgpu::TextureDimension dimension = wgpu::TextureDimension::e2D) {
  auto bindingView = wgpu::TextureViewDimension::e2D;
  if (dimension == wgpu::TextureDimension::e3D) {
    bindingView = wgpu::TextureViewDimension::e3D;
  } else if (depthOrLayers > 1) {
    bindingView = wgpu::TextureViewDimension::e2DArray;
  }
  return {
      .format = format,
      .dimension = dimension,
      .bindingView = bindingView,
      .width = width,
      .height = height,
      .depthOrLayers = depthOrLayers,
      .mips = mips,
      .usage = usage,
  };
}

TEST_F(GraphPoolDeviceTest, DawnAllocatorUsesExactUsage) {
  const auto allocator = make_dawn_allocator();
  const std::array keys{
      texture_key(TextureFormat::RGBA8Unorm, TextureUsage::RenderAttachment | TextureUsage::TextureBinding),
      texture_key(TextureFormat::RGBA16Float, TextureUsage::StorageBinding | TextureUsage::TextureBinding),
      texture_key(TextureFormat::RGBA8Unorm, TextureUsage::CopyDst | TextureUsage::TextureBinding),
      texture_key(TextureFormat::Depth32Float, TextureUsage::RenderAttachment | TextureUsage::TextureBinding),
      texture_key(TextureFormat::RGBA16Float, TextureUsage::StorageBinding, 32, 32, 8, 2, wgpu::TextureDimension::e3D),
  };
  push_validation_scope();
  for (const TexKey& key : keys) {
    const PhysicalTexture physical = allocator->create_texture(key, "graph:test");
    ASSERT_TRUE(physical.texture);
    EXPECT_EQ(physical.texture.GetUsage(), key.usage);
    EXPECT_EQ(physical.texture.GetFormat(), key.format);
    EXPECT_EQ(physical.texture.GetMipLevelCount(), key.mips);
    EXPECT_EQ(physical.texture.GetDepthOrArrayLayers(), key.depthOrLayers);
    EXPECT_EQ(physical.texture.GetSampleCount(), 1u);
    EXPECT_EQ(physical.bytes, key_bytes(key));
    EXPECT_NE(physical.serial, 0u);
  }
  const std::array buffers{
      BufKey{.bucketSize = 4096, .usage = BufferUsage::CopyDst | BufferUsage::Storage},
      BufKey{.bucketSize = 65536, .usage = BufferUsage::Storage | BufferUsage::Indirect},
      BufKey{.bucketSize = 256, .usage = BufferUsage::Uniform | BufferUsage::CopyDst},
  };
  uint64_t lastSerial = 0;
  for (const BufKey& key : buffers) {
    const PhysicalBuffer physical = allocator->create_buffer(key, "graph:test");
    ASSERT_TRUE(physical.buffer);
    EXPECT_EQ(physical.buffer.GetUsage(), key.usage);
    EXPECT_EQ(physical.buffer.GetSize(), key.bucketSize);
    EXPECT_GT(physical.serial, lastSerial);
    lastSerial = physical.serial;
  }
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
}

TEST_F(GraphPoolDeviceTest, PoisonEncodesWithoutErrors) {
  constexpr auto Attachment = TextureUsage::RenderAttachment | TextureUsage::TextureBinding;
  constexpr auto Storage = TextureUsage::StorageBinding | TextureUsage::TextureBinding;
  constexpr auto Copy = TextureUsage::CopyDst | TextureUsage::TextureBinding;
  constexpr auto Volume = wgpu::TextureDimension::e3D;
  std::vector<TexKey> keys{
      texture_key(TextureFormat::RGBA8Unorm, Attachment),
      texture_key(TextureFormat::RGBA8Uint, TextureUsage::RenderAttachment),
      texture_key(TextureFormat::R32Sint, TextureUsage::RenderAttachment),
      texture_key(TextureFormat::RGB10A2Uint, TextureUsage::RenderAttachment),
      texture_key(TextureFormat::Depth32Float, Attachment),
      texture_key(TextureFormat::Depth24PlusStencil8, Attachment),
      texture_key(TextureFormat::Stencil8, TextureUsage::RenderAttachment),
      texture_key(TextureFormat::RGBA16Float, Attachment, 64, 64, 4, 3),
      texture_key(TextureFormat::RGBA16Float, Attachment, 32, 32, 8, 2, Volume),
      texture_key(TextureFormat::RGBA16Float, Storage),
      texture_key(TextureFormat::R32Uint, Storage),
      texture_key(TextureFormat::R32Sint, Storage),
      texture_key(TextureFormat::RGBA8Unorm, Storage, 100, 60, 1, 3),
      texture_key(TextureFormat::RGBA16Float, Storage, 64, 64, 4, 3),
      texture_key(TextureFormat::RGBA16Float, Storage, 32, 32, 8, 2, Volume),
      texture_key(TextureFormat::RGBA8Unorm, Copy, 100, 60, 1, 3),
      texture_key(TextureFormat::RGBA16Float, Copy, 64, 64, 3, 2),
      texture_key(TextureFormat::RGBA16Float, Copy, 32, 32, 8, 2, Volume),
  };
  if (m_device->device.HasFeature(wgpu::FeatureName::TextureCompressionBC)) {
    keys.push_back(texture_key(TextureFormat::BC7RGBAUnorm, Copy, 256, 256, 1, 2));
  }
  const auto allocator = make_dawn_allocator();
  std::vector<PhysicalTexture> textures;
  for (const TexKey& key : keys) {
    EXPECT_NE(poison_method(key), PoisonMethod::None) << static_cast<uint32_t>(key.format);
    textures.push_back(allocator->create_texture(key, "graph:poison"));
  }
  const std::array buffers{
      allocator->create_buffer({.bucketSize = 4096, .usage = BufferUsage::CopyDst | BufferUsage::Storage},
                               "graph:copy"),
      // Over 64 x 65,535 words, so the fill needs a second dispatch dimension
      allocator->create_buffer({.bucketSize = 32 * MiB, .usage = BufferUsage::Storage}, "graph:fill"),
  };
  EXPECT_EQ(poison_method(buffers[0].key), PoisonMethod::BufferCopy);
  EXPECT_EQ(poison_method(buffers[1].key), PoisonMethod::BufferFill);

  push_validation_scope();
  Poisoner poisoner;
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  for (const PhysicalTexture& texture : textures) {
    poisoner.encode(encoder, texture);
  }
  for (const PhysicalBuffer& buffer : buffers) {
    poisoner.encode(encoder, buffer, buffer.bytes);
  }
  EXPECT_EQ(submit(encoder), std::nullopt);
}

TEST_F(GraphPoolDeviceTest, PoisonKeepsUsageUnion) {
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  const auto run_planned_frame = [&](bool poison) {
    AllocatorLog log;
    testing::reset_transient_pool(std::make_unique<LoggingAllocator>(log, make_dawn_allocator()));
    set_debug_poison(poison);
    const std::shared_ptr<const CompiledGraph> graph =
        compile_graph(planned::planned_inputs(planned::full_limits(), passes, reads));
    const FrameState frame{.graph = graph};
    on_worker_frame_begin(1, &frame);
    push_validation_scope();
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    for (size_t s = 0; s < StageCount; ++s) {
      const StageProgram& program = graph->stages[s];
      const auto passCount = static_cast<uint32_t>(program.passes.size());
      const StageData stage{
          .stage = static_cast<Stage>(s),
          .frameId = 1,
          .frame = &frame,
          .passCount = passCount,
          .enabled = std::vector<uint64_t>((passCount + 63) / 64, ~uint64_t{0}),
          .prevValid = std::vector<uint64_t>((graph->histories.size() + 63) / 64, 0),
      };
      for (const NodeId node : program.passes) {
        poison_before(encoder, stage, node);
      }
    }
    EXPECT_EQ(submit(encoder), std::nullopt) << "poison " << poison;
    EXPECT_EQ(transient_pool().stats().unpoisonable, 0u);
    testing::reset_transient_pool(nullptr);
    return log;
  };
  const AllocatorLog on = run_planned_frame(true);
  const AllocatorLog off = run_planned_frame(false);
  EXPECT_EQ(on.textures.size(), 21u + 2 * 7u);
  EXPECT_EQ(on.buffers.size(), 14u);
  EXPECT_EQ(on.textures, off.textures);
  EXPECT_EQ(on.buffers, off.buffers);
}

TEST_F(GraphPoolDeviceTest, AttachmentViewsCoverEveryAspect) {
  const auto allocator = make_dawn_allocator();
  const PhysicalTexture depth = allocator->create_texture(
      texture_key(TextureFormat::Depth24PlusStencil8, TextureUsage::RenderAttachment | TextureUsage::TextureBinding),
      "graph:depth");
  const PhysicalTexture color = allocator->create_texture(
      texture_key(TextureFormat::RGBA8Unorm, TextureUsage::RenderAttachment | TextureUsage::TextureBinding, 64, 64, 1,
                  3),
      "graph:color");
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.entryCount = 1, .entries = &entry};
  const wgpu::BindGroupLayout layout = m_device->device.CreateBindGroupLayout(&layoutDescriptor);
  const auto bind = [&](const wgpu::TextureView& view) {
    const wgpu::BindGroupEntry binding{.binding = 0, .textureView = view};
    const wgpu::BindGroupDescriptor descriptor{.layout = layout, .entryCount = 1, .entries = &binding};
    return m_device->device.CreateBindGroup(&descriptor);
  };

  // A depth-stencil texture binds through one aspect only, and attaches through both
  push_validation_scope();
  bind(depth.binding_view(wgpu::TextureViewDimension::e2D));
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  push_validation_scope();
  bind(depth.attachment_view(0, 0));
  EXPECT_NE(test::pop_validation_error(*m_device), std::nullopt);

  push_validation_scope();
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::RenderPassDepthStencilAttachment depthAttachment{
      .view = depth.attachment_view(0, 0),
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 0.0f,
      .stencilLoadOp = wgpu::LoadOp::Clear,
      .stencilStoreOp = wgpu::StoreOp::Store,
  };
  const wgpu::RenderPassDescriptor depthPass{.colorAttachmentCount = 0, .depthStencilAttachment = &depthAttachment};
  encoder.BeginRenderPass(&depthPass).End();
  const wgpu::RenderPassColorAttachment colorAttachment{
      .view = color.attachment_view(1, 0),
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
  };
  const wgpu::RenderPassDescriptor colorPass{.colorAttachmentCount = 1, .colorAttachments = &colorAttachment};
  encoder.BeginRenderPass(&colorPass).End();
  EXPECT_EQ(submit(encoder), std::nullopt);

  // Cached views are handed out once per shape
  EXPECT_EQ(color.attachment_view(1, 0).Get(), color.attachment_view(1, 0).Get());
  EXPECT_NE(color.attachment_view(1, 0).Get(), color.attachment_view(0, 0).Get());
}

TEST_F(GraphPoolDeviceTest, CompatSetsBindingViewDimension) {
  std::string error;
  const test::TestDeviceOptions options{.core = false};
  const auto device = test::create_test_device(options, error);
  if (!device || device->core) {
    GTEST_SKIP() << "no compat null device: " << error;
  }
  {
    const test::ScopedGlobalDevice globals{*device, options};
    ASSERT_FALSE(webgpu::g_hasCoreFeatures);
    TexKey key =
        texture_key(TextureFormat::RGBA8Unorm, TextureUsage::TextureBinding | TextureUsage::CopyDst, 64, 64, 6);
    key.bindingView = wgpu::TextureViewDimension::Cube;
    device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const PhysicalTexture physical = make_dawn_allocator()->create_texture(key, "graph:cube");
    EXPECT_EQ(physical.texture.GetTextureBindingViewDimension(), wgpu::TextureViewDimension::Cube);
    EXPECT_TRUE(physical.binding_view(wgpu::TextureViewDimension::Cube));
    EXPECT_EQ(test::pop_validation_error(*device), std::nullopt);
  }
  EXPECT_TRUE(test::take_uncaptured_errors(*device).empty());
}

TEST_F(GraphPoolDeviceTest, RunStagePoisonsAndSwaps) {
  const TexDesc desc = absolute_tex(TextureFormat::RGBA16Float, 64, 64);
  const std::vector<TestPass> passes{
      {.name = "own",
       .order = 0,
       .body =
           [desc](InternalPassBuilder& b) {
             b.history("hist", desc, Access::StorageWrite);
             b.create("tmp", desc, Access::StorageWrite);
           }},
      {.name = "use",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.use("tmp", Access::Sampled);
             b.use_prev("hist");
           },
       .sink = true},
  };
  const auto graph = compile(passes);
  const StageProgram& program = graph->stages[static_cast<size_t>(Stage::OpaqueDone)];
  ASSERT_EQ(program.passes.size(), 2u);
  const FrameState frame{.graph = graph};
  StageData stage{
      .stage = Stage::OpaqueDone,
      .frameId = 1,
      .frame = &frame,
      .passCount = static_cast<uint32_t>(program.passes.size()),
      .enabled = {0b11},
      .prevValid = {0},
  };
  set_debug_poison(true);
  on_worker_frame_begin(1, &frame);
  TransientPool& pool = transient_pool();
  ASSERT_EQ(pool.materialized_graph(), graph.get());
  const uint16_t h = history_of(*graph, "hist");
  const uint64_t written = pool.history_write(h).serial;

  push_validation_scope();
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  run_stage(encoder, stage);
  EXPECT_EQ(submit(encoder), std::nullopt);
  EXPECT_EQ(pool.stats().unpoisonable, 0u);

  on_worker_frame_begin(2, &frame);
  EXPECT_EQ(pool.history_prev(h).serial, written);

  // A frame whose owner did not run leaves the pair alone
  stage.frameId = 2;
  stage.enabled = {0b10};
  stage.prevValid = {1};
  push_validation_scope();
  const wgpu::CommandEncoder skipped = m_device->device.CreateCommandEncoder();
  run_stage(skipped, stage);
  EXPECT_EQ(submit(skipped), std::nullopt);
  on_worker_frame_begin(3, &frame);
  EXPECT_EQ(pool.history_prev(h).serial, written);
}
} // namespace
} // namespace aurora::render::detail

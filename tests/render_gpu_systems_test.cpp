#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/frame.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gpu_channel.hpp"
#include "gx/hints.hpp"
#include "render/jobs.hpp"
#include "render/shaders.hpp"
#include "render/systems/systems.hpp"

#include <aurora/gpu_systems.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace aurora::render {
namespace {
using gpu::SystemStatus;
namespace channel = gx::gpu_channel;
namespace wire = gx::hints::wire;

constexpr std::string_view NoopSource = R"(@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (stream_active(id)) {
    stream_store(stream_index(id), stream_load(0u, stream_index(id)));
  }
}
)";
constexpr std::string_view AtomicSource = R"(@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
  if (stream_active(id)) {
    stream_store(stream_index(id), state_add(0u, 1u));
  }
}
)";
constexpr std::string_view RedeclaringSource = R"(@group(1) @binding(0) var<storage, read_write> mine: array<u32>;
@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z)
fn main() {
  mine[0] = 1u;
  stream_store(0u, 1u);
}
)";
constexpr std::array<gpu::KernelDesc, 1> MainKernel{{{.entryPoint = "main"}}};

void decode_pending() {
  const uint32_t size = gx::fifo::get_buffer_size();
  if (size != 0) {
    gx::fifo::process(gx::fifo::get_buffer_data(), size);
  }
  gx::fifo::clear_buffer();
}

std::vector<uint8_t> hint(uint16_t id, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out(GX_AURORA_HINT_HEADER_SIZE + payload.size());
  wire::Writer writer{out};
  writer.header(id, static_cast<uint32_t>(payload.size()));
  writer.bytes(payload);
  return out;
}

std::vector<uint8_t> create_bytes(uint64_t id, std::string_view source) {
  constexpr std::string_view Name = "race";
  constexpr std::string_view Entry = "main";
  std::vector<uint8_t> payload(wire::GpuCreateHeader + Name.size() + wire::GpuCreateKernelHeader + Entry.size() +
                               source.size());
  wire::Writer out{payload};
  const auto text = [](std::string_view value) {
    return std::span<const uint8_t>{reinterpret_cast<const uint8_t*>(value.data()), value.size()};
  };
  out.u16(1);
  out.u64(id);
  out.u32(0);
  out.u32(0);
  out.u32(0);
  out.u8(static_cast<uint8_t>(Name.size()));
  out.u8(1);
  out.u32(static_cast<uint32_t>(source.size()));
  out.bytes(text(Name));
  out.u8(static_cast<uint8_t>(Entry.size()));
  out.bytes(text(Entry));
  out.u16(64);
  out.u16(1);
  out.u16(1);
  out.u8(0);
  out.bytes(text(source));
  return hint(GX_AURORA_GPU_CREATE_SYSTEM, payload);
}

std::vector<uint8_t> destroy_bytes(uint64_t id) {
  std::vector<uint8_t> payload(wire::GpuDestroyV1);
  wire::Writer out{payload};
  out.u16(1);
  out.u64(id);
  return hint(GX_AURORA_GPU_DESTROY_SYSTEM, payload);
}

class RenderGpuSystems : public ::testing::Test {
protected:
  void SetUp() override { start({}); }

  void TearDown() override { stop(); }

  void start(const test::TestDeviceOptions& options) {
    std::string error;
    m_device = test::create_test_device(options, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, options);
    jobs::initialize(2);
    shaders::initialize();
    gfx::detail::testing::create_compute_layouts();
    systems::initialize();
    channel::reset();
    channel::testing::reset_stats();
    gfx::set_blocking_pipelines(true);
  }

  void stop() {
    if (!m_device) {
      return;
    }
    gfx::set_blocking_pipelines(false);
    gx::fifo::clear_buffer();
    systems::shutdown();
    gfx::detail::testing::release_compute_layouts();
    shaders::shutdown();
    jobs::shutdown();
    channel::reset();
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  void restart(const test::TestDeviceOptions& options) {
    stop();
    start(options);
  }

  static SystemStatus create_and_decode(const gpu::SystemDesc& desc, gpu::SystemId& id) {
    id = gpu::create_system(desc);
    decode_pending();
    return gpu::system_status(id);
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(RenderGpuSystems, NoopBecomesReady) {
  ASSERT_TRUE(gpu::systems_available());
  ASSERT_TRUE(systems::kernel_layout());
  const auto id = gpu::create_system({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel});
  ASSERT_NE(id, gpu::InvalidSystem);
  EXPECT_EQ(gpu::system_status(id), SystemStatus::Pending);
  decode_pending();
  EXPECT_EQ(gpu::system_status(id), SystemStatus::Ready);
  EXPECT_TRUE(systems::kernel_pipeline(id, 0));
  EXPECT_FALSE(systems::kernel_pipeline(id, 1));
  EXPECT_TRUE(systems::wait_ready(id));
}

TEST_F(RenderGpuSystems, BrokenKernelsFail) {
  gpu::SystemId id = gpu::InvalidSystem;
  EXPECT_EQ(create_and_decode({.name = "broken", .wgsl = "this is not wgsl", .kernels = MainKernel}, id),
            SystemStatus::Failed);
  const std::array<gpu::KernelDesc, 1> absent{{{.entryPoint = "absent"}}};
  EXPECT_EQ(create_and_decode({.name = "absent", .wgsl = NoopSource, .kernels = absent}, id), SystemStatus::Failed);
  EXPECT_EQ(create_and_decode({.name = "redeclares", .wgsl = RedeclaringSource, .kernels = MainKernel}, id),
            SystemStatus::Failed);
  EXPECT_EQ(create_and_decode({.name = "missing", .library = "systems/missing", .kernels = MainKernel}, id),
            SystemStatus::Failed);
  EXPECT_EQ(create_and_decode({.name = "gpu_only",
                               .wgsl = NoopSource,
                               .kernels = MainKernel,
                               .persistentBytes = 64,
                               .flags = gpu::SystemGpuOnlyState},
                              id),
            SystemStatus::Failed);
}

TEST_F(RenderGpuSystems, UnmetRequirementIsUnavailable) {
  restart({.optionalFeatures = false});
  ASSERT_FALSE(caps().subgroups);
  const auto before = shaders::stats();
  gpu::SystemId id = gpu::InvalidSystem;
  EXPECT_EQ(
      create_and_decode(
          {.name = "subgroups", .wgsl = NoopSource, .kernels = MainKernel, .requirements = gpu::RequiresSubgroups}, id),
      SystemStatus::Unavailable);
  const auto after = shaders::stats();
  EXPECT_EQ(after.pending + after.ready + after.failed, before.pending + before.ready + before.failed);
  EXPECT_FALSE(systems::kernel_pipeline(id, 0));
}

TEST_F(RenderGpuSystems, DestroyAndRecreateReusesIds) {
  gpu::SystemId first = gpu::InvalidSystem;
  ASSERT_EQ(create_and_decode({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel}, first), SystemStatus::Ready);
  const auto pipeline = systems::kernel_pipeline(first, 0);
  gpu::destroy_system(first);
  decode_pending();
  EXPECT_EQ(gpu::system_status(first), SystemStatus::Invalid);
  EXPECT_FALSE(systems::kernel_pipeline(first, 0));
  gpu::SystemId second = gpu::InvalidSystem;
  ASSERT_EQ(create_and_decode({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel}, second),
            SystemStatus::Ready);
  EXPECT_EQ(static_cast<uint32_t>(second), static_cast<uint32_t>(first));
  EXPECT_EQ(second >> 32, (first >> 32) + 1);
  EXPECT_EQ(systems::kernel_pipeline(second, 0), pipeline);
}

TEST_F(RenderGpuSystems, KernelLayoutFitsCompat) {
  restart({.core = false, .optionalFeatures = false});
  ASSERT_FALSE(caps().core);
  gpu::SystemId copy = gpu::InvalidSystem;
  const std::array<gpu::KernelDesc, 1> wide{{{.entryPoint = "main", .workgroupSize = {128, 1, 1}}}};
  EXPECT_EQ(create_and_decode({.name = "copy", .library = "systems/copy", .kernels = wide}, copy), SystemStatus::Ready);
  const std::array<gpu::KernelDesc, 1> atomic{
      {{.entryPoint = "count", .workgroupSize = {128, 1, 1}, .flags = gpu::KernelStateAtomic}}};
  gpu::SystemId counter = gpu::InvalidSystem;
  create_and_decode({.name = "counter", .wgsl = AtomicSource, .kernels = atomic, .persistentBytes = 64}, counter);
  EXPECT_TRUE(systems::wait_ready(counter));
  const std::array<gpu::KernelDesc, 1> tooWide{{{.entryPoint = "main", .workgroupSize = {256, 1, 1}}}};
  EXPECT_EQ(gpu::create_system({.name = "too_wide", .library = "systems/copy", .kernels = tooWide}),
            gpu::InvalidSystem);
}

TEST_F(RenderGpuSystems, StatusRace) {
  gpu::SystemId id = gpu::InvalidSystem;
  ASSERT_EQ(create_and_decode({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel}, id), SystemStatus::Ready);
  std::atomic_bool done{false};
  std::atomic_uint32_t notReady{0};
  std::thread poller{[&] {
    while (!done.load(std::memory_order_acquire)) {
      if (gpu::system_status(id) != SystemStatus::Ready || !systems::kernel_pipeline(id, 0)) {
        notReady.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }};
  for (uint32_t generation = 1; generation <= 1000; ++generation) {
    const uint64_t churn = uint64_t{generation} << 32 | 2;
    const auto create = create_bytes(churn, NoopSource);
    const auto destroy = destroy_bytes(churn);
    gx::fifo::process(create.data(), static_cast<uint32_t>(create.size()));
    gx::fifo::process(destroy.data(), static_cast<uint32_t>(destroy.size()));
  }
  done.store(true, std::memory_order_release);
  poller.join();
  EXPECT_EQ(notReady.load(), 0u);
  EXPECT_EQ(gpu::stats().created, 1001u);
  EXPECT_EQ(gpu::stats().destroyed, 1000u);
}

TEST_F(RenderGpuSystems, WaitSystems) {
  gfx::set_blocking_pipelines(false);
  const auto id = gpu::create_system({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel});
  decode_pending();
  channel::wait_systems();
  EXPECT_EQ(gpu::system_status(id), SystemStatus::Ready);
}

TEST_F(RenderGpuSystems, StatefulSystemReadyFromNextFrame) {
  gpu::SystemId id = gpu::InvalidSystem;
  EXPECT_EQ(create_and_decode({.name = "noop", .wgsl = NoopSource, .kernels = MainKernel, .persistentBytes = 16}, id),
            SystemStatus::Pending);
  channel::end_frame();
  EXPECT_EQ(gpu::system_status(id), SystemStatus::Ready);
}
} // namespace
} // namespace aurora::render

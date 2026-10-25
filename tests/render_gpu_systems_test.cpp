#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/frame.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gpu_channel.hpp"
#include "gx/hints.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "render/gpu_heap.hpp"
#include "render/jobs.hpp"
#include "render/shaders.hpp"
#include "render/systems/dispatch.hpp"
#include "render/systems/systems.hpp"
#include "webgpu/caps.hpp"

#include <aurora/gpu_systems.hpp>
#include <dolphin/gx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

extern "C" void __GXSetDirtyState();

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
    gpu_heap::shutdown();
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
    gpu_heap::shutdown();
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
}

TEST_F(RenderGpuSystems, GpuOnlyStateGetsABlock) {
  gpu::SystemId id = gpu::InvalidSystem;
  EXPECT_EQ(create_and_decode({.name = "gpu_only",
                               .wgsl = NoopSource,
                               .kernels = MainKernel,
                               .persistentBytes = 300,
                               .flags = gpu::SystemGpuOnlyState},
                              id),
            SystemStatus::Pending);
  channel::end_frame();
  EXPECT_EQ(gpu::system_status(id), SystemStatus::Ready);
  systems::KernelSlot slot;
  ASSERT_TRUE(systems::kernel_slot(id, 0, slot));
  EXPECT_NE(slot.state, gpu_heap::BlockHandle::Invalid);
  EXPECT_EQ(slot.stateBytes, 300u);
  EXPECT_EQ(slot.workgroupSize, (std::array<uint32_t, 3>{64, 1, 1}));
  EXPECT_EQ(gpu_heap::block_state(slot.state), gpu_heap::BlockState::Pending);
  gpu_heap::begin_frame_worker();
  ASSERT_EQ(gpu_heap::block_state(slot.state), gpu_heap::BlockState::Ready);
  EXPECT_EQ(gpu_heap::block_view(slot.state).size, 512u);
  gpu::destroy_system(id);
  decode_pending();
  gpu_heap::begin_frame_worker();
  EXPECT_EQ(gpu_heap::block_state(slot.state), gpu_heap::BlockState::Invalid);
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

TEST_F(RenderGpuSystems, FailedStateBlockFailsStatus) {
  Caps limited = caps();
  limited.limits.maxStorageBufferBindingSize = 64 * 1024;
  webgpu::set_caps(std::move(limited));
  constexpr uint32_t StateBytes = 48 * 1024;
  gpu::SystemId placed = gpu::InvalidSystem;
  gpu::SystemId failed = gpu::InvalidSystem;
  create_and_decode({.name = "placed", .wgsl = NoopSource, .kernels = MainKernel, .persistentBytes = StateBytes},
                    placed);
  create_and_decode({.name = "failed", .wgsl = NoopSource, .kernels = MainKernel, .persistentBytes = StateBytes},
                    failed);
  channel::end_frame();
  EXPECT_EQ(gpu::system_status(failed), SystemStatus::Ready);
  gpu_heap::begin_frame_worker();
  systems::KernelSlot slot;
  ASSERT_TRUE(systems::kernel_slot(failed, 0, slot));
  ASSERT_EQ(gpu_heap::block_state(slot.state), gpu_heap::BlockState::Failed);
  EXPECT_EQ(gpu::system_status(placed), SystemStatus::Ready);
  EXPECT_EQ(gpu::system_status(failed), SystemStatus::Failed);
}

std::vector<uint8_t> upload_bytes(uint64_t handle, std::span<const uint8_t> data) {
  std::vector<uint8_t> payload(wire::GpuUploadHeader + data.size());
  wire::Writer out{payload};
  out.u16(1);
  out.u64(handle);
  out.bytes(data);
  return hint(GX_AURORA_GPU_UPLOAD, payload);
}

std::vector<uint8_t> reserve_bytes(uint64_t handle, uint32_t bytes, uint32_t flags = 0) {
  std::vector<uint8_t> payload(wire::GpuReserveV1);
  wire::Writer out{payload};
  out.u16(1);
  out.u64(handle);
  out.u32(bytes);
  out.u32(flags);
  return hint(GX_AURORA_GPU_RESERVE, payload);
}

std::vector<uint8_t> update_bytes(uint64_t system, uint32_t offset, std::span<const uint8_t> data) {
  std::vector<uint8_t> payload(wire::GpuUpdateHeader + data.size());
  wire::Writer out{payload};
  out.u16(1);
  out.u64(system);
  out.u32(offset);
  out.bytes(data);
  return hint(GX_AURORA_GPU_UPDATE_PERSISTENT, payload);
}

std::vector<uint8_t> dispatch_bytes(uint64_t system, uint32_t kernel, uint64_t output, std::array<uint32_t, 3> elements,
                                    std::span<const uint64_t> inputs, std::span<const uint8_t> params = {}) {
  std::vector<uint8_t> payload(wire::GpuDispatchHeaderV1 + inputs.size() * 8 + params.size());
  wire::Writer out{payload};
  out.u16(1);
  out.u64(system);
  out.u32(kernel);
  out.u64(output);
  for (const uint32_t count : elements) {
    out.u32(count);
  }
  out.u8(static_cast<uint8_t>(inputs.size()));
  out.u8(static_cast<uint8_t>(params.size() / 4));
  for (const uint64_t input : inputs) {
    out.u64(input);
  }
  out.bytes(params);
  return hint(GX_AURORA_GPU_DISPATCH, payload);
}

void process(const std::vector<uint8_t>& bytes) {
  gx::fifo::process(bytes.data(), static_cast<uint32_t>(bytes.size()));
}

uint64_t handle_at(uint32_t index) { return uint64_t{channel::current_serial()} << 32 | index; }

constexpr std::array<gpu::KernelDesc, 1> CopyKernel{{{.entryPoint = "main", .workgroupSize = {64, 1, 1}}}};

class RenderSystemsDispatch : public RenderGpuSystems {
protected:
  void SetUp() override {
    RenderGpuSystems::SetUp();
    if (!m_device) {
      return;
    }
    gfx::detail::testing::suppress_render_worker(true);
    gfx::detail::begin_recording(m_packet, 0);
    m_recording = true;
  }

  void TearDown() override {
    if (m_recording) {
      gfx::finish();
      gfx::detail::end_recording();
      gfx::detail::shutdown_recording();
      gfx::detail::testing::suppress_render_worker(false);
    }
    RenderGpuSystems::TearDown();
  }

  void next_frame() {
    gfx::finish();
    gfx::detail::end_recording();
    m_packet = {};
    channel::end_frame();
    gfx::detail::begin_recording(m_packet, 0);
  }

  const std::vector<gfx::detail::ComputeEntry>& pending_entries() const {
    static const std::vector<gfx::detail::ComputeEntry> None;
    return m_packet.pendingCompute < m_packet.computeBatches.size()
               ? m_packet.computeBatches[m_packet.pendingCompute].entries
               : None;
  }

  gpu::SystemId copy_system() {
    const auto id = gpu::create_system({.name = "copy", .library = "systems/copy", .kernels = CopyKernel});
    decode_pending();
    EXPECT_EQ(gpu::system_status(id), SystemStatus::Ready);
    return id;
  }

  gfx::detail::FramePacket m_packet;
  bool m_recording = false;
};

TEST_F(RenderSystemsDispatch, RecordsGroupsAndParams) {
  const auto id = copy_system();
  std::vector<uint8_t> data(260);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i);
  }
  process(upload_bytes(handle_at(1), data));
  process(reserve_bytes(handle_at(2), 260));
  const std::array<uint64_t, 1> inputs{handle_at(1)};
  const std::array<uint8_t, 4> user{7, 0, 0, 0};
  process(dispatch_bytes(id, 0, handle_at(2), {65, 1, 1}, inputs, user));

  const auto src = channel::lookup(handle_at(1));
  const auto dst = channel::lookup(handle_at(2));
  EXPECT_EQ(dst.state, channel::HandleState::Written);
  EXPECT_EQ(gpu::stats().dispatches, 1u);
  EXPECT_EQ(gfx::detail::testing::pending_compute_entries(), 1u);
  const auto& entries = pending_entries();
  ASSERT_EQ(entries.size(), 1u);
  const auto& entry = entries[0];
  EXPECT_EQ(entry.kind, gfx::detail::ComputeEntry::Kind::Dispatch);
  EXPECT_EQ(entry.groups, (std::array<uint32_t, 3>{2, 1, 1}));
  EXPECT_EQ(entry.range.offset, dst.range.offset);
  EXPECT_EQ(entry.range.size, 260u);
  EXPECT_EQ(entry.params % 256, 0u);
  EXPECT_EQ(m_packet.computeParamsBytes, gfx::DispatchParamsSize);
  ASSERT_LE(entry.params + sizeof(gfx::DispatchParams), m_packet.uniforms.size());
  gfx::DispatchParams params;
  std::memcpy(&params, m_packet.uniforms.data() + entry.params, sizeof(params));
  EXPECT_EQ(params.output, (std::array<uint32_t, 4>{dst.range.offset / 4, 65, 0, 0}));
  EXPECT_EQ(params.inputs[0], src.range.offset / 4);
  EXPECT_EQ(params.inputs[1], 65u);
  for (size_t i = 2; i < params.inputs.size(); ++i) {
    EXPECT_EQ(params.inputs[i], 0u) << i;
  }
  EXPECT_EQ(params.groups, (std::array<uint32_t, 4>{2, 1, 1, 1}));
  EXPECT_EQ(params.elements, (std::array<uint32_t, 4>{65, 1, 1, 0}));
  EXPECT_EQ(params.reserved, (std::array<uint32_t, 4>{}));
  EXPECT_EQ(params.user[0], 7u);
  EXPECT_TRUE(std::all_of(params.user.begin() + 1, params.user.end(), [](uint8_t b) { return b == 0; }));

  gfx::finish();
  ASSERT_EQ(m_packet.ops.size(), 2u);
  EXPECT_EQ(m_packet.ops[0].type, gfx::detail::FrameOpType::Compute);
  EXPECT_EQ(m_packet.ops[0].computeBatch, &m_packet.computeBatches[0]);
  EXPECT_EQ(m_packet.ops[1].type, gfx::detail::FrameOpType::RenderPass);
}

TEST_F(RenderSystemsDispatch, ZeroElementsPreClearOnly) {
  const auto id = copy_system();
  process(reserve_bytes(handle_at(1), 64));
  const size_t uniforms = m_packet.uniforms.size();
  process(dispatch_bytes(id, 0, handle_at(1), {0, 1, 1}, {}));
  EXPECT_EQ(channel::lookup(handle_at(1)).state, channel::HandleState::Written);
  ASSERT_EQ(pending_entries().size(), 1u);
  EXPECT_EQ(pending_entries()[0].groups[0], 0u);
  EXPECT_EQ(pending_entries()[0].range.size, 64u);
  EXPECT_EQ(m_packet.uniforms.size(), uniforms);
  EXPECT_EQ(m_packet.computeParamsBytes, 0u);
}

TEST_F(RenderSystemsDispatch, RejectionsFailTheOutput) {
  const auto id = copy_system();
  const auto state =
      gpu::create_system({.name = "state", .wgsl = NoopSource, .kernels = MainKernel, .persistentBytes = 16});
  decode_pending();
  const std::array<uint8_t, 16> data{};
  process(upload_bytes(handle_at(1), data));
  for (uint32_t index = 2; index <= 6; ++index) {
    process(reserve_bytes(handle_at(index), 16));
  }
  process(reserve_bytes(handle_at(7), 16, gpu::ReserveZero));
  const std::array<uint64_t, 1> uploaded{handle_at(1)};
  const std::array<uint64_t, 1> unwritten{handle_at(4)};
  process(dispatch_bytes(0x77ull << 32 | 5, 0, handle_at(2), {1, 1, 1}, uploaded));
  process(dispatch_bytes(id, 1, handle_at(3), {1, 1, 1}, uploaded));
  process(dispatch_bytes(id, 0, handle_at(7), {1, 1, 1}, uploaded));
  process(dispatch_bytes(id, 0, handle_at(5), {1, 1, 1}, unwritten));
  process(dispatch_bytes(state, 0, handle_at(6), {1, 1, 1}, {}));
  for (const uint32_t index : {2u, 3u, 5u, 6u, 7u}) {
    EXPECT_EQ(channel::lookup(handle_at(index)).state, channel::HandleState::Failed) << index;
  }
  EXPECT_EQ(channel::lookup(handle_at(4)).state, channel::HandleState::Reserved);
  EXPECT_EQ(gpu::stats().dispatchesRejected, 5u);
  EXPECT_EQ(gpu::stats().dispatches, 0u);
  EXPECT_TRUE(pending_entries().empty());
}

TEST_F(RenderSystemsDispatch, CreationFrameUpdateFlushesBeforeFirstDispatch) {
  const auto state =
      gpu::create_system({.name = "state", .wgsl = NoopSource, .kernels = MainKernel, .persistentBytes = 16});
  decode_pending();
  const std::array<uint8_t, 8> words{1, 2, 3, 4, 5, 6, 7, 8};
  process(update_bytes(state, 4, words));
  EXPECT_TRUE(pending_entries().empty());
  EXPECT_FALSE(channel::uncaptured_gpu_state());

  next_frame();
  EXPECT_EQ(gpu::system_status(state), SystemStatus::Ready);
  process(dispatch_bytes(state, 0, 0, {1, 1, 1}, {}));
  const auto& entries = pending_entries();
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].kind, gfx::detail::ComputeEntry::Kind::Copy);
  EXPECT_EQ(entries[0].dstOffset, 0u);
  ASSERT_EQ(entries[0].range.size, 16u);
  const std::vector<uint8_t> staged(m_packet.storage.data() + entries[0].range.offset,
                                    m_packet.storage.data() + entries[0].range.offset + 16);
  EXPECT_EQ(staged, (std::vector<uint8_t>{0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0}));
  EXPECT_EQ(entries[1].kind, gfx::detail::ComputeEntry::Kind::Dispatch);
  EXPECT_EQ(entries[1].range.size, 0u);
  EXPECT_EQ(gpu::stats().persistentCopies, 1u);
  EXPECT_EQ(gpu::stats().dispatches, 1u);
  EXPECT_TRUE(channel::uncaptured_gpu_state());

  process(update_bytes(state, 8, std::span{words}.first(4)));
  ASSERT_EQ(entries.size(), 3u);
  EXPECT_EQ(entries[2].kind, gfx::detail::ComputeEntry::Kind::Copy);
  EXPECT_EQ(entries[2].dstOffset, 8u);
  EXPECT_EQ(entries[2].range.size, 4u);
  EXPECT_EQ(gpu::stats().persistentCopies, 2u);
}

std::vector<uint8_t> load_array_bytes(uint8_t attr, uint64_t handle) {
  std::vector<uint8_t> payload(wire::LoadArrayGpuV1);
  wire::Writer out{payload};
  out.u16(1);
  out.u8(attr);
  out.u64(handle);
  out.u32(0);
  return hint(GX_AURORA_LOAD_ARRAY_GPU, payload);
}

// Holds both job workers so kernel compiles stay queued until release is set
class WorkerGate {
public:
  WorkerGate() {
    for (jobs::JobHandle& job : m_jobs) {
      job = jobs::submit([this](const jobs::JobContext& context) {
        m_started.fetch_add(1, std::memory_order_relaxed);
        while (!m_release.load(std::memory_order_acquire) && !context.cancelled()) {
          std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
      });
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (m_started.load(std::memory_order_relaxed) < m_jobs.size() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  }
  ~WorkerGate() {
    release();
    for (const jobs::JobHandle& job : m_jobs) {
      job.wait();
    }
  }
  WorkerGate(const WorkerGate&) = delete;
  WorkerGate& operator=(const WorkerGate&) = delete;

  [[nodiscard]] bool holding() const noexcept { return m_started.load(std::memory_order_relaxed) == m_jobs.size(); }
  void release() noexcept { m_release.store(true, std::memory_order_release); }

private:
  std::array<jobs::JobHandle, 2> m_jobs;
  std::atomic_uint32_t m_started{0};
  std::atomic_bool m_release{false};
};

TEST_F(RenderSystemsDispatch, PendingKernelRejectsWithoutBlocking) {
  gfx::set_blocking_pipelines(false);
  WorkerGate gate;
  ASSERT_TRUE(gate.holding());
  const auto id = gpu::create_system({.name = "copy", .library = "systems/copy", .kernels = CopyKernel});
  decode_pending();
  ASSERT_EQ(gpu::system_status(id), SystemStatus::Pending);
  const std::array<uint8_t, 16> data{};
  process(upload_bytes(handle_at(1), data));
  process(reserve_bytes(handle_at(2), 16));
  const std::array<uint64_t, 1> inputs{handle_at(1)};
  process(dispatch_bytes(id, 0, handle_at(2), {4, 1, 1}, inputs));
  EXPECT_EQ(gpu::stats().dispatchesRejected, 1u);
  EXPECT_EQ(gpu::stats().dispatches, 0u);
  EXPECT_EQ(channel::lookup(handle_at(2)).state, channel::HandleState::Failed);
  EXPECT_TRUE(pending_entries().empty());
  EXPECT_EQ(gfx::detail::testing::pending_compute_entries(), 0u);

  GXInit(nullptr, 0);
  gx::fifo::clear_buffer();
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  __GXSetDirtyState();
  decode_pending();
  process(load_array_bytes(GX_VA_POS, handle_at(2)));
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  GXPosition1x8(0);
  GXPosition1x8(1);
  GXPosition1x8(2);
  GXEnd();
  decode_pending();
  EXPECT_EQ(gpu::stats().suppressedDraws, 1u);
}

TEST_F(RenderSystemsDispatch, PendingKernelWaitsUnderBlocking) {
  gfx::set_blocking_pipelines(false);
  WorkerGate gate;
  ASSERT_TRUE(gate.holding());
  const auto id = gpu::create_system({.name = "copy", .library = "systems/copy", .kernels = CopyKernel});
  decode_pending();
  ASSERT_EQ(gpu::system_status(id), SystemStatus::Pending);
  const std::array<uint8_t, 16> data{};
  process(upload_bytes(handle_at(1), data));
  process(reserve_bytes(handle_at(2), 16));
  gfx::set_blocking_pipelines(true);
  std::atomic_bool waited{false};
  std::thread releaser{[&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (shaders::testing::waiters() == 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    waited = shaders::testing::waiters() != 0;
    gate.release();
  }};
  const std::array<uint64_t, 1> inputs{handle_at(1)};
  process(dispatch_bytes(id, 0, handle_at(2), {4, 1, 1}, inputs));
  releaser.join();
  EXPECT_TRUE(waited);
  EXPECT_EQ(gpu::stats().dispatchesRejected, 0u);
  EXPECT_EQ(gpu::stats().dispatches, 1u);
  EXPECT_EQ(channel::lookup(handle_at(2)).state, channel::HandleState::Written);
  EXPECT_EQ(pending_entries().size(), 1u);
}
} // namespace
} // namespace aurora::render

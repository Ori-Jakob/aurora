#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "render/jobs.hpp"
#include "render/shaders.hpp"
#include "render/shadows/vsm_pages.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace aurora::render::shadows::vsm {
namespace {
constexpr uint64_t LayoutKey = 0x76736D7570640001ull;

class RenderVsmGpu : public ::testing::Test {
protected:
  void SetUp() override {
    m_options = {
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined)};
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    jobs::initialize(2);
    shaders::initialize();
    m_started = true;
  }

  void TearDown() override {
    if (m_started) {
      shaders::shutdown();
      jobs::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  wgpu::Buffer buffer(uint64_t size) const {
    const wgpu::BufferDescriptor descriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
        .size = size,
    };
    return m_device->device.CreateBuffer(&descriptor);
  }

  std::vector<uint32_t> read(const wgpu::Buffer& source, uint64_t size) const {
    return test::read_buffer(*m_device, source, 0, size).value_or(std::vector<uint32_t>{});
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
};

std::array<double, 3> sun_dir(double elevationDeg, double azimuthDeg) {
  const double e = elevationDeg * std::numbers::pi / 180.0;
  const double a = azimuthDeg * std::numbers::pi / 180.0;
  return {std::cos(e) * std::sin(a), -std::sin(e), std::cos(e) * std::cos(a)};
}

TEST_F(RenderVsmGpu, MaintenanceMatchesCpuTwin) {
  const wgpu::Device& device = m_device->device;
  std::array<wgpu::BindGroupLayoutEntry, 4> entries{};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Storage},
    };
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{.entryCount = entries.size(), .entries = entries.data()};
  const wgpu::BindGroupLayout groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::BindGroupLayoutDescriptor emptyDescriptor{};
  const wgpu::BindGroupLayout emptyLayout = device.CreateBindGroupLayout(&emptyDescriptor);
  const std::array layouts{emptyLayout, groupLayout};
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{.bindGroupLayoutCount = layouts.size(),
                                                          .bindGroupLayouts = layouts.data()};
  const wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  const shaders::PipelineId id = shaders::request_compute({
      .label = "vsm.update test",
      .shader = Shader,
      .entryPoint = "vsm_update",
      .layout = pipelineLayout,
      .layoutKey = LayoutKey,
  });
  ASSERT_EQ(shaders::wait(id), shaders::PipelineStatus::Ready);
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(id);

  const wgpu::Buffer views = buffer(ViewsBytes);
  const wgpu::Buffer table = buffer(TableBytes);
  const wgpu::Buffer phys = buffer(PhysBytes);
  const wgpu::Buffer stats = buffer(StatsBytes);
  const std::array groupEntries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = views, .size = ViewsBytes},
      wgpu::BindGroupEntry{.binding = 1, .buffer = table, .size = TableBytes},
      wgpu::BindGroupEntry{.binding = 2, .buffer = phys, .size = PhysBytes},
      wgpu::BindGroupEntry{.binding = 3, .buffer = stats, .size = StatsBytes},
  };
  const wgpu::BindGroupDescriptor bindDescriptor{
      .layout = groupLayout, .entryCount = groupEntries.size(), .entries = groupEntries.data()};
  const wgpu::BindGroup group = device.CreateBindGroup(&bindDescriptor);
  const wgpu::BindGroupDescriptor emptyGroupDescriptor{.layout = emptyLayout};
  const wgpu::BindGroup emptyGroup = device.CreateBindGroup(&emptyGroupDescriptor);

  SunClipmap clipmap;
  (void)clipmap.update({.eye = {1200.0, 340.0, -9100.0}, .dir = sun_dir(38.0, 140.0), .lightId = 2, .texel0 = 0.25});
  Views shifted{};
  clipmap.fill(shifted);
  shifted.header.poolPages = MaxPoolPages;
  for (uint32_t k = 0; k < SunLevels; ++k) {
    ViewRecord& v = shifted.views[level_view(k)];
    v.flags &= ~(view_flag::Freed | view_flag::Recentred | view_flag::Wrap);
    v.prevOrigin[0] = v.origin[0];
    v.prevOrigin[1] = v.origin[1];
    v.origin[0] += 3;
    v.origin[1] -= 5;
    v.flags |= view_flag::Wrap;
  }
  shifted.views[level_view(2)].flags |= view_flag::Freed;
  shifted.views[level_view(5)].flags |= view_flag::Recentred;

  std::vector<uint32_t> cpuTable(MaxVirtualPages, 0);
  std::vector<uint32_t> cpuPhys(MaxPoolPages * PhysWords, 0);
  uint32_t page = 0;
  for (uint32_t v = 0; v < SunLevels * PagesPerView; v += 3) {
    if (page == MaxPoolPages) {
      break;
    }
    cpuTable[v] = page | entry::Allocated | entry::Cached | (v % 7 == 0 ? entry::Dirty : 0u);
    cpuPhys[page * PhysWords] = v + 1;
    cpuPhys[page * PhysWords + 1] = 1 + v % 9;
    ++page;
  }
  ASSERT_EQ(page, MaxPoolPages);

  const wgpu::Queue& queue = m_device->queue;
  const auto run = [&](const Views& in) {
    queue.WriteBuffer(views, 0, &in, ViewsBytes);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.ClearBuffer(stats, 0, StatsBytes);
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, emptyGroup);
    pass.SetBindGroup(1, group);
    pass.DispatchWorkgroups(MaxVirtualPages / UpdateWorkgroup);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
  };
  const auto expect_twin = [&](const Views& in, const char* what) {
    const UpdateStats twin = apply_update(in, cpuTable, cpuPhys);
    EXPECT_EQ(read(table, TableBytes), cpuTable) << what;
    EXPECT_EQ(read(phys, PhysBytes), cpuPhys) << what;
    const std::vector<uint32_t> counters = read(stats, StatsBytes);
    EXPECT_EQ(counters.size(), StatCount) << what;
    if (counters.size() < StatCount) {
      return twin;
    }
    EXPECT_EQ(counters[stat::FreedWrap], twin.freedWrap) << what;
    EXPECT_EQ(counters[stat::FreedRelatch], twin.freedRelatch) << what;
    EXPECT_EQ(counters[stat::FreedRecentre], twin.freedRecentre) << what;
    EXPECT_EQ(counters[stat::FreedFull], twin.freedFull) << what;
    std::vector<uint32_t> written(ViewsBytes / 4);
    std::memcpy(written.data(), &in, ViewsBytes);
    EXPECT_EQ(read(views, ViewsBytes), written) << what;
    return twin;
  };

  queue.WriteBuffer(table, 0, cpuTable.data(), TableBytes);
  queue.WriteBuffer(phys, 0, cpuPhys.data(), PhysBytes);
  run(shifted);
  const UpdateStats moved = expect_twin(shifted, "shift (3, -5), relatch level 2, re-centre level 5");
  EXPECT_GT(moved.freedWrap, 0u);
  EXPECT_GT(moved.freedRelatch, 0u);
  EXPECT_GT(moved.freedRecentre, 0u);
  uint32_t kept = 0;
  for (const uint32_t e : cpuTable) {
    kept += (e & entry::Allocated) != 0 ? 1 : 0;
  }
  EXPECT_GT(kept, 0u);

  Views still = shifted;
  for (uint32_t k = 0; k < SunLevels; ++k) {
    ViewRecord& v = still.views[level_view(k)];
    v.flags &= ~(view_flag::Freed | view_flag::Recentred | view_flag::Wrap);
    v.prevOrigin[0] = v.origin[0];
    v.prevOrigin[1] = v.origin[1];
  }
  run(still);
  const UpdateStats quiet = expect_twin(still, "steady");
  EXPECT_EQ(quiet.freedWrap + quiet.freedRelatch + quiet.freedRecentre + quiet.freedFull, 0u);

  Views full = still;
  full.header.flags |= header_flag::Full;
  run(full);
  const UpdateStats reset = expect_twin(full, "full");
  EXPECT_EQ(reset.freedFull, kept);
  EXPECT_TRUE(std::ranges::all_of(cpuTable, [](uint32_t e) { return e == 0; }));
}
} // namespace
} // namespace aurora::render::shadows::vsm

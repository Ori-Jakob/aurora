#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gfx/recording.hpp"
#include "gx/gx.hpp"
#include "render/jobs.hpp"
#include "render/scene/atlas.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/streamout.hpp"
#include "render/shaders.hpp"
#include "render/shadows/vsm_alloc.hpp"
#include "render/shadows/vsm_check.hpp"
#include "render/shadows/vsm_cull.hpp"
#include "render/shadows/vsm_invalidate.hpp"
#include "render/shadows/vsm_pages.hpp"
#include "render/shadows/vsm_raster.hpp"
#include "render/stage_data.hpp"
#include "webgpu/gpu.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace aurora::render::shadows::vsm {
namespace {
constexpr uint64_t LayoutKey = 0x76736D7570640001ull;

class RenderVsmGpuDevice : public ::testing::Test {
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

namespace record = scene::streamout::record;
namespace instance_flag = scene::streamout::instance_flag;

constexpr uint64_t ScanLayoutKey = 0x76736D7363616E01ull;
constexpr uint64_t DirtyLayoutKey = 0x76736D6469727401ull;
constexpr uint32_t TestSlots = 64;
constexpr uint64_t MaterialBytes = 4096ull * 32;
constexpr double TestTexel0 = 0.25;
constexpr std::array<double, 3> TestEye{1200.0, 340.0, -9100.0};

uint32_t ordered(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  return (u & 0x80000000u) != 0 ? ~u : u | 0x80000000u;
}

using Record = std::array<uint32_t, RecordWords>;

void place(std::span<uint32_t, RecordWords> rec, const std::array<double, 3>& t) {
  for (uint32_t r = 0; r < 3; ++r) {
    const auto hi = static_cast<float>(t[r]);
    const auto lo = static_cast<float>(t[r] - static_cast<double>(hi));
    for (uint32_t c = 0; c < 3; ++c) {
      rec[record::World + r * 4 + c] = std::bit_cast<uint32_t>(r == c ? 1.f : 0.f);
    }
    rec[record::World + r * 4 + 3] = std::bit_cast<uint32_t>(hi);
    rec[record::World + 12 + r] = std::bit_cast<uint32_t>(lo);
  }
}

Record make_record(uint32_t hash, const std::array<double, 3>& t, float half, uint32_t categories = AURORA_CAT_DYNAMIC,
                   uint32_t flags = 0) {
  Record rec{};
  rec[record::Flags] = instance_flag::Valid | flags;
  rec[record::Categories] = categories;
  rec[record::IdentityHash] = hash;
  rec[record::Material] = scene::streamout::NoMaterial | 0xFF0000u;
  rec[record::WorldFrame] = 1;
  rec[record::UvFrame] = 1;
  rec[record::MaterialFrame] = 1;
  rec[record::StreamFrame] = 1;
  rec[record::ClusterFirst] = 4;
  rec[record::ClusterCount] = 2;
  rec[record::IndexCount] = 36;
  place(std::span{rec}, t);
  for (uint32_t c = 0; c < 3; ++c) {
    rec[record::CurAabb + c] = ordered(-half);
    rec[record::CurAabb + 3 + c] = ordered(half);
  }
  return rec;
}

std::array<double, 3> offset(double x, double y, double z) { return {TestEye[0] + x, TestEye[1] + y, TestEye[2] + z}; }

Views test_views(bool full) {
  SunClipmap clipmap;
  (void)clipmap.update({.eye = TestEye, .dir = sun_dir(38.0, 140.0), .lightId = 2, .texel0 = TestTexel0});
  Views views{};
  clipmap.fill(views);
  views.header.flags = full ? header_flag::Full : 0u;
  views.header.poolPages = MaxPoolPages;
  return views;
}

bool test_bit(std::span<const uint32_t> words, uint32_t index) {
  return ((words[index >> 5] >> (index & 31)) & 1u) != 0;
}

uint32_t popcount(std::span<const uint32_t> words) {
  uint32_t n = 0;
  for (const uint32_t w : words) {
    n += static_cast<uint32_t>(std::popcount(w));
  }
  return n;
}

struct ScanOut {
  std::vector<uint32_t> history;
  std::vector<uint32_t> castBits;
  std::vector<uint32_t> slotWords;
  std::vector<uint32_t> dirty;
  std::vector<uint32_t> stats;
  std::vector<uint32_t> table;
  std::vector<uint32_t> listed;
};

class RenderVsmGpu : public RenderVsmGpuDevice {
protected:
  void SetUp() override {
    RenderVsmGpuDevice::SetUp();
    if (IsSkipped()) {
      return;
    }
    const wgpu::Device& device = m_device->device;
    const auto layout = [&](uint32_t count, uint32_t readOnly) {
      std::vector<wgpu::BindGroupLayoutEntry> entries(count);
      for (uint32_t i = 0; i < count; ++i) {
        entries[i] = {
            .binding = i,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = {.type =
                           i < readOnly ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage},
        };
      }
      const wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = entries.size(), .entries = entries.data()};
      return device.CreateBindGroupLayout(&descriptor);
    };
    m_scanGroup = layout(8, 3);
    m_dirtyGroup = layout(6, 3);
    const wgpu::BindGroupLayoutDescriptor emptyDescriptor{};
    m_emptyLayout = device.CreateBindGroupLayout(&emptyDescriptor);
    const wgpu::BindGroupDescriptor emptyGroup{.layout = m_emptyLayout};
    m_empty = device.CreateBindGroup(&emptyGroup);
    const auto pipeline = [&](const wgpu::BindGroupLayout& group, std::string_view entry, uint64_t key,
                              std::span<const std::string_view> defines) {
      const std::array layouts{m_emptyLayout, group};
      const wgpu::PipelineLayoutDescriptor descriptor{.bindGroupLayoutCount = layouts.size(),
                                                      .bindGroupLayouts = layouts.data()};
      const shaders::PipelineId id = shaders::request_compute({
          .label = entry,
          .shader = InvalidateShader,
          .entryPoint = entry,
          .defines = defines,
          .layout = device.CreatePipelineLayout(&descriptor),
          .layoutKey = key,
      });
      EXPECT_EQ(shaders::wait(id), shaders::PipelineStatus::Ready) << entry;
      return shaders::compute_pipeline(id);
    };
    static constexpr std::array<std::string_view, 1> Dirty{"VSM_DIRTY"};
    m_scan = pipeline(m_scanGroup, "vsm_scan", ScanLayoutKey, {});
    m_dirty = pipeline(m_dirtyGroup, "vsm_dirty", DirtyLayoutKey, Dirty);
    m_instances = buffer(scene::streamout::InstancePoolBytes);
    m_materials = buffer(MaterialBytes);
    m_views = buffer(ViewsBytes);
    m_history = buffer(HistoryBytes);
    m_stats = buffer(StatsBytes);
    m_castBits = buffer(CastBitsBytes);
    m_slotWords = buffer(SlotWordsBytes);
    m_boxes = buffer(DirtyBoxesBytes);
    m_list = buffer(SlotWordsBytes);
    m_table = buffer(TableBytes);
    m_dirtyPages = buffer(DirtyPagesBytes);
    m_records.assign(size_t{TestSlots} * RecordWords, 0);
    m_materialWords.assign(MaterialBytes / 4, 0);
    m_twinHistory.assign(size_t{Slots} * HistoryWords, 0);
  }

  wgpu::BindGroup group(const wgpu::BindGroupLayout& layout, std::span<const wgpu::Buffer> buffers) const {
    std::vector<wgpu::BindGroupEntry> entries(buffers.size());
    for (uint32_t i = 0; i < buffers.size(); ++i) {
      entries[i] = {.binding = i, .buffer = buffers[i], .size = buffers[i].GetSize()};
    }
    const wgpu::BindGroupDescriptor descriptor{
        .layout = layout, .entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroup(&descriptor);
  }

  std::span<uint32_t, RecordWords> rec(uint32_t slot) {
    return std::span{m_records}.subspan(size_t{slot} * RecordWords).first<RecordWords>();
  }

  void put(uint32_t slot, const Record& words) { std::ranges::copy(words, rec(slot).begin()); }

  void set_material(uint32_t id, uint32_t flags) { m_materialWords[size_t{id} * 8 + 1] = flags << 16; }

  ScanOut run(const Views& views, const wgpu::Buffer* instances = nullptr) {
    const wgpu::Queue& queue = m_device->queue;
    if (instances == nullptr) {
      queue.WriteBuffer(m_instances, 0, m_records.data(), m_records.size() * 4);
    }
    queue.WriteBuffer(m_materials, 0, m_materialWords.data(), MaterialBytes);
    queue.WriteBuffer(m_views, 0, &views, ViewsBytes);
    const wgpu::Buffer& source = instances != nullptr ? *instances : m_instances;
    const std::array scanBuffers{source, m_materials, m_views, m_history, m_stats, m_castBits, m_slotWords, m_boxes};
    const wgpu::BindGroup scanGroup = group(m_scanGroup, scanBuffers);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    encoder.ClearBuffer(m_stats, 0, StatsBytes);
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(m_scan);
    pass.SetBindGroup(0, m_empty);
    pass.SetBindGroup(1, scanGroup);
    pass.DispatchWorkgroups(Slots / ScanWorkgroup);
    pass.End();
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));

    ScanOut out;
    out.slotWords = read(m_slotWords, SlotWordsBytes);
    for (uint32_t s = 0; s < out.slotWords.size(); ++s) {
      if (out.slotWords[s] != 0) {
        out.listed.push_back(s);
      }
    }
    if (!out.listed.empty()) {
      queue.WriteBuffer(m_list, 0, out.listed.data(), out.listed.size() * 4);
    }
    const std::array dirtyBuffers{m_list, m_boxes, m_views, m_table, m_stats, m_dirtyPages};
    const wgpu::BindGroup dirtyGroup = group(m_dirtyGroup, dirtyBuffers);
    const wgpu::CommandEncoder second = m_device->device.CreateCommandEncoder();
    second.ClearBuffer(m_dirtyPages, 0, DirtyPagesBytes);
    const wgpu::ComputePassEncoder dirty = second.BeginComputePass();
    dirty.SetPipeline(m_dirty);
    dirty.SetBindGroup(0, m_empty);
    dirty.SetBindGroup(1, dirtyGroup);
    dirty.DispatchWorkgroups(static_cast<uint32_t>(out.listed.size()));
    dirty.End();
    EXPECT_TRUE(test::submit_and_wait(*m_device, second.Finish()));
    out.history = read(m_history, HistoryBytes);
    out.castBits = read(m_castBits, CastBitsBytes);
    out.dirty = read(m_dirtyPages, DirtyPagesBytes);
    out.stats = read(m_stats, StatsBytes);
    out.table = read(m_table, TableBytes);
    return out;
  }

  // The twin keeps its own history, so a GPU error cannot hide by feeding the reference
  void expect_twin(const ScanOut& gpu, std::span<const uint32_t> records, const Views& views, const char* what) {
    ASSERT_EQ(gpu.history.size(), size_t{Slots} * HistoryWords) << what;
    ASSERT_EQ(gpu.dirty.size(), DirtyPagesWords) << what;
    std::vector<uint32_t> inner(DirtyPagesWords, 0);
    std::vector<uint32_t> outer(DirtyPagesWords, 0);
    uint32_t mismatches = 0;
    const Record empty{};
    for (uint32_t s = 0; s < Slots; ++s) {
      const auto prev = std::span{m_twinHistory}.subspan(size_t{s} * HistoryWords).first<HistoryWords>();
      const std::span<const uint32_t, RecordWords> r =
          size_t{s + 1} * RecordWords <= records.size() ? records.subspan(size_t{s} * RecordWords).first<RecordWords>()
                                                        : std::span<const uint32_t, RecordWords>{empty};
      const SlotScan twin = scan_slot(prev, r, m_materialWords, views.header);
      const uint32_t* got = &gpu.history[size_t{s} * HistoryWords];
      for (uint32_t k = 0; k < HistoryWords && mismatches < 8; ++k) {
        const bool approx = k >= hist::Box && k < hist::Xform + 12;
        const float a = std::bit_cast<float>(got[k]);
        const float b = std::bit_cast<float>(twin.history[k]);
        const bool same = approx ? std::abs(a - b) <= 1e-3f + 1e-5f * std::abs(b) : got[k] == twin.history[k];
        if (!same) {
          ADD_FAILURE() << what << ": slot " << s << " history word " << k << " is " << got[k] << ", twin "
                        << twin.history[k];
          ++mismatches;
        }
      }
      const uint32_t wantWord = (twin.changed ? slot_word::Changed : 0u) | (twin.chunkNeed ? slot_word::ChunkNeed : 0u);
      if (gpu.slotWords[s] != wantWord && mismatches++ < 8) {
        ADD_FAILURE() << what << ": slot " << s << " slot word " << gpu.slotWords[s] << ", twin " << wantWord;
      }
      if (test_bit(gpu.castBits, s) != twin.castNow && mismatches++ < 8) {
        ADD_FAILURE() << what << ": slot " << s << " cast bit differs from the twin's " << twin.castNow;
      }
      if (twin.changed) {
        dirty_footprint(views, twin.box, inner, -0.01f);
        dirty_footprint(views, twin.box, outer, 0.01f);
      }
      std::copy(twin.history.begin(), twin.history.end(), prev.begin());
    }
    for (uint32_t w = 0; w < DirtyPagesWords; ++w) {
      EXPECT_EQ(inner[w] & ~gpu.dirty[w], 0u) << what << ": dirty word " << w;
      EXPECT_EQ(gpu.dirty[w] & ~outer[w], 0u) << what << ": dirty word " << w;
    }
  }

  std::vector<uint32_t> footprint(const Views& views, const Box& box, bool player = false) const {
    std::array<uint32_t, DirtyBoxWords> words{};
    for (uint32_t i = 0; i < 3; ++i) {
      words[dirty_box::New + i] = std::bit_cast<uint32_t>(box.lo[i]);
      words[dirty_box::New + 3 + i] = std::bit_cast<uint32_t>(box.hi[i]);
    }
    words[dirty_box::Flags] = dirty_box::HasNew | (player ? dirty_box::NewPlayer : 0u);
    std::vector<uint32_t> planes(DirtyPagesWords, 0);
    dirty_footprint(views, words, planes, -0.01f);
    return planes;
  }

  Box world_box(const std::array<double, 3>& t, float half, const Views& views) const {
    Box box;
    for (uint32_t i = 0; i < 3; ++i) {
      const double anchor = double{views.header.anchorHi[i]} + double{views.header.anchorLo[i]};
      box.lo[i] = static_cast<float>(t[i] - anchor - half - TestTexel0);
      box.hi[i] = static_cast<float>(t[i] - anchor + half + TestTexel0);
    }
    return box;
  }

  wgpu::BindGroupLayout m_scanGroup;
  wgpu::BindGroupLayout m_dirtyGroup;
  wgpu::BindGroupLayout m_emptyLayout;
  wgpu::BindGroup m_empty;
  wgpu::ComputePipeline m_scan;
  wgpu::ComputePipeline m_dirty;
  wgpu::Buffer m_instances, m_materials, m_views, m_history, m_stats, m_castBits, m_slotWords, m_boxes, m_list, m_table,
      m_dirtyPages;
  std::vector<uint32_t> m_records;
  std::vector<uint32_t> m_materialWords;
  std::vector<uint32_t> m_twinHistory;
};

bool covers(std::span<const uint32_t> got, std::span<const uint32_t> want, uint32_t firstWord, uint32_t words) {
  for (uint32_t w = firstWord; w < firstWord + words; ++w) {
    if ((want[w] & ~got[w]) != 0) {
      return false;
    }
  }
  return true;
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

TEST_F(RenderVsmGpu, ScanDirtiesOldAndNewFootprint) {
  constexpr uint32_t Mover = 1, Still = 2, Freed = 3, Reused = 4, Skinned = 5, Opaque = 6, Cutout = 7, Pending = 8,
                     Player = 9, Clusters = 10;
  constexpr uint32_t CutoutMaterial = 3, PendingMaterial = 4;
  constexpr uint32_t AlphaTest = 1, AlphaPending = 16;
  const std::array<std::array<double, 3>, 11> at{{
      {},
      offset(-300.0, 0.0, 0.0),
      offset(-150.0, 0.0, 250.0),
      offset(0.0, 0.0, -300.0),
      offset(200.0, 0.0, -200.0),
      offset(300.0, 0.0, 100.0),
      offset(-250.0, 0.0, -150.0),
      offset(120.0, 0.0, 300.0),
      offset(-50.0, 0.0, 120.0),
      offset(60.0, 0.0, -60.0),
      offset(250.0, 0.0, 250.0),
  }};
  for (uint32_t s = Mover; s <= Clusters; ++s) {
    put(s,
        make_record(0x900 + s, at[s], 20.f, s == Player ? AURORA_CAT_PLAYER | AURORA_CAT_DYNAMIC : AURORA_CAT_DYNAMIC));
  }
  rec(Still)[record::Categories] = AURORA_CAT_STATIC;
  rec(Skinned)[record::Flags] |= instance_flag::Palette;
  rec(Cutout)[record::Material] = CutoutMaterial;
  rec(Pending)[record::Material] = PendingMaterial;
  set_material(CutoutMaterial, AlphaTest);
  set_material(PendingMaterial, AlphaTest | AlphaPending);
  std::vector<uint32_t> table(MaxVirtualPages, 0);
  for (uint32_t v = 0; v < SunLevels * PagesPerView; ++v) {
    table[v] = (v % MaxPoolPages) | entry::Allocated | entry::Cached;
  }
  m_device->queue.WriteBuffer(m_table, 0, table.data(), TableBytes);

  const Views full = test_views(true);
  const Views steady = test_views(false);
  ScanOut out = run(full);
  expect_twin(out, m_records, full, "full");
  EXPECT_TRUE(out.listed.empty());
  EXPECT_EQ(popcount(out.dirty), 0u);
  for (uint32_t s = Mover; s <= Clusters; ++s) {
    EXPECT_TRUE(test_bit(out.castBits, s)) << s;
  }

  out = run(steady);
  expect_twin(out, m_records, steady, "steady");
  EXPECT_TRUE(out.listed.empty());
  EXPECT_EQ(popcount(out.dirty), 0u);
  EXPECT_EQ(out.stats[stat::Changed], 0u);

  const Box moverOld = world_box(at[Mover], 20.f, steady);
  const Box playerOld = world_box(at[Player], 20.f, steady);
  const std::array<double, 3> moved{at[Mover][0] + 100.0, at[Mover][1], at[Mover][2]};
  place(rec(Mover), moved);
  rec(Mover)[record::WorldFrame] = 2;
  const Box freedOld = world_box(at[Freed], 20.f, steady);
  rec(Freed)[record::Flags] = 0;
  const Box reusedOld = world_box(at[Reused], 20.f, steady);
  rec(Reused)[record::IdentityHash] = 0xABCD;
  rec(Skinned)[record::StreamFrame] = 2;
  rec(Opaque)[record::UvFrame] = 2;
  rec(Cutout)[record::UvFrame] = 2;
  set_material(PendingMaterial, AlphaTest);
  const std::array<double, 3> playerMoved{at[Player][0], at[Player][1], at[Player][2] + 40.0};
  place(rec(Player), playerMoved);
  rec(Player)[record::WorldFrame] = 2;
  rec(Clusters)[record::ClusterFirst] = 40;
  rec(Clusters)[record::ClusterCount] = 3;
  out = run(steady);
  expect_twin(out, m_records, steady, "changes");
  const std::vector<uint32_t> listed{Mover, Freed, Reused, Skinned, Cutout, Pending, Player};
  EXPECT_EQ(out.listed, listed);
  const auto dirtied = [&](const Box& box) { return covers(out.dirty, footprint(steady, box), 0, DirtyPlaneWords); };
  EXPECT_TRUE(dirtied(moverOld)) << "the old footprint of a 100 u move";
  EXPECT_TRUE(dirtied(world_box(moved, 20.f, steady))) << "the new footprint of a 100 u move";
  EXPECT_FALSE(dirtied(world_box(at[Still], 20.f, steady))) << "a static box";
  EXPECT_TRUE(dirtied(freedOld)) << "a freed slot's old footprint";
  EXPECT_FALSE(test_bit(out.castBits, Freed));
  EXPECT_TRUE(dirtied(reusedOld)) << "the old occupant stops casting";
  EXPECT_FALSE(test_bit(out.castBits, Reused)) << "a reused slot waits for its stream";
  EXPECT_NE(out.history[Reused * HistoryWords + hist::Flags] & hist::PendingStream, 0u);
  EXPECT_TRUE(dirtied(world_box(at[Skinned], 20.f, steady))) << "a palette re-stream";
  EXPECT_FALSE(dirtied(world_box(at[Opaque], 20.f, steady))) << "a uv change on an opaque caster";
  EXPECT_TRUE(dirtied(world_box(at[Cutout], 20.f, steady))) << "a uv change on a cutout caster";
  EXPECT_TRUE(dirtied(world_box(at[Pending], 20.f, steady))) << "an alpha tile that turned ready";
  EXPECT_FALSE(dirtied(world_box(at[Clusters], 20.f, steady))) << "a cluster range change";
  EXPECT_EQ(out.history[Clusters * HistoryWords + hist::ClusterFirst], 40u);
  std::vector<uint32_t> playerPages = footprint(steady, world_box(playerMoved, 20.f, steady), true);
  const std::vector<uint32_t> playerOldPages = footprint(steady, playerOld, true);
  for (uint32_t w = 0; w < DirtyPagesWords; ++w) {
    playerPages[w] |= playerOldPages[w];
  }
  EXPECT_TRUE(covers(out.dirty, playerPages, DirtyPlaneWords, DirtyPlaneWords)) << "PLAYER sets Priority";
  EXPECT_GT(out.stats[stat::FootprintPriority], 0u);
  uint32_t priority = 0;
  for (uint32_t index = 0; index < MaxVirtualPages; ++index) {
    if (test_bit(std::span{playerPages}.subspan(DirtyPlaneWords), index)) {
      EXPECT_EQ(out.table[index] & (entry::Dirty | entry::Priority), entry::Dirty | entry::Priority) << index;
      ++priority;
    }
    if (test_bit(out.dirty, index)) {
      EXPECT_NE(out.table[index] & entry::Dirty, 0u) << index;
    }
  }
  EXPECT_GT(priority, 0u);
  EXPECT_GE(out.stats[stat::DirtiedPriority], priority);
  EXPECT_EQ(out.stats[stat::PendingStream], 1u);
  EXPECT_EQ(out.stats[stat::PendingCasters], 1u);

  rec(Reused)[record::StreamFrame] = 3;
  const std::array<double, 3> reusedAt = offset(220.0, 10.0, -180.0);
  place(rec(Reused), reusedAt);
  out = run(steady);
  expect_twin(out, m_records, steady, "stream lands");
  EXPECT_EQ(out.listed, std::vector<uint32_t>{Reused});
  EXPECT_TRUE(test_bit(out.castBits, Reused));
  EXPECT_TRUE(dirtied(world_box(reusedAt, 20.f, steady))) << "the new occupant's footprint once w92 moves";
  EXPECT_FALSE(test_bit(out.castBits, Freed));

  out = run(steady);
  expect_twin(out, m_records, steady, "settled");
  EXPECT_TRUE(out.listed.empty()) << "an alpha tile dirties once";
  EXPECT_EQ(popcount(out.dirty), 0u);
}

TEST_F(RenderVsmGpu, ExcludedClassesNeverDirty) {
  const std::array<std::pair<uint32_t, uint32_t>, 11> excluded{{
      {instance_flag::CpuDeformed, AURORA_CAT_DYNAMIC},
      {instance_flag::Billboard, AURORA_CAT_DYNAMIC},
      {instance_flag::Water, AURORA_CAT_WATER},
      {instance_flag::Blended, AURORA_CAT_DYNAMIC},
      {0, AURORA_CAT_NO_SHADOW_CAST},
      {0, AURORA_CAT_SKY},
      {0, AURORA_CAT_DECAL},
      {0, AURORA_CAT_FAKE_SHADOW},
      {0, AURORA_CAT_UI},
      {0, AURORA_CAT_OVERLAY},
      {0, AURORA_CAT_PARTICLE},
  }};
  for (uint32_t i = 0; i < excluded.size(); ++i) {
    put(i + 1, make_record(0x300 + i, offset(-400.0 + 70.0 * i, 0.0, 0.0), 25.f,
                           excluded[i].second | AURORA_CAT_DYNAMIC, excluded[i].first));
  }
  const Views full = test_views(true);
  ScanOut out = run(full);
  expect_twin(out, m_records, full, "full");
  const Views steady = test_views(false);
  for (uint32_t frame = 2; frame < 6; ++frame) {
    for (uint32_t i = 0; i < excluded.size(); ++i) {
      place(rec(i + 1), offset(-400.0 + 70.0 * i, 10.0 * frame, 5.0 * frame));
      rec(i + 1)[record::WorldFrame] = frame;
      rec(i + 1)[record::StreamFrame] = frame;
    }
    out = run(steady);
    expect_twin(out, m_records, steady, "excluded");
    EXPECT_TRUE(out.listed.empty()) << frame;
    EXPECT_EQ(popcount(out.dirty), 0u) << frame;
    EXPECT_EQ(popcount(out.castBits), 0u) << frame;
    EXPECT_EQ(out.stats[stat::ScanCasters], 0u) << frame;
    EXPECT_EQ(out.stats[stat::ScanValid], excluded.size()) << frame;
  }
}

TEST_F(RenderVsmGpu, LateChangeSeenNextFrame) {
  constexpr uint32_t Late = 5;
  const std::array<double, 3> start = offset(100.0, 0.0, 100.0);
  put(Late, make_record(0x55, start, 30.f, AURORA_CAT_DYNAMIC, instance_flag::Late));
  const Views full = test_views(true);
  ScanOut out = run(full);
  expect_twin(out, m_records, full, "full");
  const Views steady = test_views(false);
  out = run(steady);
  expect_twin(out, m_records, steady, "before");
  EXPECT_TRUE(out.listed.empty());
  // The late plan streams after phase.shadows, so this frame's scan has already read the record
  const std::array<double, 3> later = offset(180.0, 0.0, 100.0);
  place(rec(Late), later);
  rec(Late)[record::WorldFrame] = 3;
  out = run(steady);
  expect_twin(out, m_records, steady, "next frame");
  EXPECT_EQ(out.listed, std::vector<uint32_t>{Late});
  EXPECT_TRUE(covers(out.dirty, footprint(steady, world_box(start, 30.f, steady)), 0, DirtyPlaneWords));
  EXPECT_TRUE(covers(out.dirty, footprint(steady, world_box(later, 30.f, steady)), 0, DirtyPlaneWords));
}

TEST_F(RenderVsmGpu, CastBitsFollowCaster) {
  constexpr std::array<uint32_t, 4> Watched{31, 32, 33, 63};
  for (const uint32_t s : Watched) {
    put(s, make_record(0x700 + s, offset(-200.0 + 3.0 * s, 0.0, 50.0), 10.f));
  }
  const Views steady = test_views(false);
  const Views full = test_views(true);
  ScanOut out = run(full);
  expect_twin(out, m_records, full, "full");
  for (const uint32_t s : Watched) {
    EXPECT_TRUE(test_bit(out.castBits, s)) << s;
  }
  EXPECT_EQ(popcount(out.castBits), Watched.size());
  rec(32)[record::Flags] = 0;
  out = run(steady);
  expect_twin(out, m_records, steady, "freed");
  EXPECT_FALSE(test_bit(out.castBits, 32));
  EXPECT_TRUE(test_bit(out.castBits, 31));
  EXPECT_TRUE(test_bit(out.castBits, 33));
  rec(32)[record::Flags] = instance_flag::Valid;
  rec(32)[record::IdentityHash] = 0x1234;
  out = run(steady);
  expect_twin(out, m_records, steady, "reused before its stream");
  EXPECT_FALSE(test_bit(out.castBits, 32));
  rec(32)[record::StreamFrame] = 9;
  out = run(steady);
  expect_twin(out, m_records, steady, "reused after its stream");
  EXPECT_TRUE(test_bit(out.castBits, 32));
  rec(63)[record::Categories] = AURORA_CAT_DYNAMIC | AURORA_CAT_NO_SHADOW_CAST;
  out = run(steady);
  EXPECT_FALSE(test_bit(out.castBits, 63));
  EXPECT_EQ(popcount(out.castBits), 3u);
}

namespace streamout = scene::streamout;

std::vector<uint8_t> g_block;

gfx::Range capture_block(std::span<const uint8_t> bytes) noexcept {
  g_block.assign(bytes.begin(), bytes.end());
  return {256, static_cast<uint32_t>(bytes.size())};
}

TEST_F(RenderVsmGpu, CasterPredicateMatchesCpu) {
  namespace class_bit = scene::retained::class_bit;
  constexpr uint32_t QuadStride = 12;
  constexpr std::array<uint32_t, 6> Classes{
      class_bit::Dynamic,
      class_bit::Static | class_bit::Immobile,
      class_bit::Dynamic | class_bit::CpuDeformed,
      class_bit::Dynamic | class_bit::Billboard,
      class_bit::Dynamic | class_bit::Water,
      class_bit::Dynamic | class_bit::Blended,
  };
  constexpr std::array<uint32_t, 11> Categories{
      AURORA_CAT_DYNAMIC,
      AURORA_CAT_STATIC | AURORA_CAT_TERRAIN,
      AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER,
      AURORA_CAT_DYNAMIC | AURORA_CAT_FOLIAGE,
      AURORA_CAT_DYNAMIC | AURORA_CAT_NO_SHADOW_CAST,
      AURORA_CAT_SKY,
      AURORA_CAT_DECAL,
      AURORA_CAT_DYNAMIC | AURORA_CAT_FAKE_SHADOW,
      AURORA_CAT_UI,
      AURORA_CAT_OVERLAY,
      AURORA_CAT_PARTICLE,
  };
  constexpr std::array<std::array<float, 3>, 4> Quad{{
      {-4.f, 0.f, -4.f},
      {4.f, 0.f, -4.f},
      {4.f, 2.f, 4.f},
      {-4.f, 2.f, 4.f},
  }};
  std::vector<uint8_t> vertexBytes(Quad.size() * QuadStride, 0);
  for (size_t v = 0; v < Quad.size(); ++v) {
    for (size_t c = 0; c < 3; ++c) {
      const uint32_t bits = std::bit_cast<uint32_t>(Quad[v][c]);
      for (int b = 0; b < 4; ++b) {
        vertexBytes[v * QuadStride + c * 4 + b] = static_cast<uint8_t>(bits >> (24 - 8 * b));
      }
    }
  }
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  gx::g_gxState = {};
  gx::g_gxState.pnMtx[0].pos = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  const scene::Affine invView = scene::to_affine(gx::g_gxState.pnMtx[0].pos);
  gfx::set_blocking_pipelines(true);
  scene::retained::testing::reset();
  streamout::initialize();
  gfx::detail::StreamRoom room{.storageCapacity = 64ull << 20};
  streamout::testing::set_push(&capture_block);
  streamout::testing::set_room(&room);

  std::vector<scene::retained::InstanceInfo> infos;
  std::vector<scene::Affine> worlds;
  worlds.reserve(Classes.size() * Categories.size());
  ASSERT_FALSE(streamout::begin_frame(true, 1));
  const gx::DrawImmediateData immediates{};
  for (const uint32_t classBits : Classes) {
    for (const uint32_t categories : Categories) {
      const auto slot = static_cast<uint32_t>(infos.size() + 1);
      infos.push_back({.slot = slot, .classBits = classBits, .categories = categories});
      scene::Affine& world = worlds.emplace_back(invView);
      world.m[0][3] = TestEye[0] + 12.0 * (slot % 8);
      world.m[1][3] = TestEye[1];
      world.m[2][3] = TestEye[2] + 12.0 * (slot / 8);
      streamout::on_capture({
          .slot = slot,
          .slotGeneration = 1,
          .identityHash = 0x4000 + slot,
          .newGeometry = true,
          .classBits = classBits,
          .categories = categories,
          .attrs = attrs.data(),
          .stride = QuadStride,
          .immediates = &immediates,
          .prim = GX_QUADS,
          .vtxCount = static_cast<uint32_t>(Quad.size()),
          .world = &world,
          .invView = &invView,
      });
    }
  }
  detail::StageData data{.stage = Stage::OpaqueDone, .frameIndex = 7};
  auto frame = std::make_shared<scene::retained::Frame>();
  streamout::seal(*frame, data);
  data.retained = frame;
  std::array<std::byte, streamout::ParamsSize> params{};
  streamout::commit_stage_params(data, nullptr, params);
  streamout::sealed(data);
  const std::shared_ptr<const streamout::Plan> plan = frame->plan;
  ASSERT_NE(plan, nullptr);
  ASSERT_TRUE(streamout::testing::wait_variants(*plan));

  std::vector<uint8_t> storage(256 + g_block.size(), 0);
  std::copy(g_block.begin(), g_block.end(), storage.begin() + 256);
  storage.resize((storage.size() + 3) / 4 * 4, 0);
  const wgpu::Buffer vbuf = buffer(std::max<size_t>(vertexBytes.size(), 256));
  const wgpu::Buffer abuf = buffer(storage.size());
  const wgpu::BufferDescriptor indexDescriptor{
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index, .size = 256};
  const wgpu::Buffer ibuf = m_device->device.CreateBuffer(&indexDescriptor);
  const wgpu::Buffer vertices = buffer(1 << 20);
  const wgpu::Buffer indices = buffer(1 << 18);
  const wgpu::Buffer palettes = buffer(streamout::PalettePoolBytes);
  const wgpu::Buffer header = buffer(streamout::FramePoolBytes);
  vertexBytes.resize(std::max<size_t>(vertexBytes.size(), 256), 0);
  m_device->queue.WriteBuffer(vbuf, 0, vertexBytes.data(), vertexBytes.size());
  m_device->queue.WriteBuffer(abuf, 0, storage.data(), storage.size());
  const auto view = [](const wgpu::Buffer& b) { return BufferView{.buffer = b, .offset = 0, .size = b.GetSize()}; };
  const std::array<BufferView, 8> pools{view(vbuf),    view(abuf),        view(ibuf),     view(vertices),
                                        view(indices), view(m_instances), view(palettes), view(header)};
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  ASSERT_TRUE(streamout::testing::encode_plan(encoder, pools, *plan));
  ASSERT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  streamout::end_frame();

  ScanOut out = run(test_views(true), &m_instances);
  const std::vector<uint32_t> records = read(m_instances, uint64_t{TestSlots + 16} * record::Bytes);
  ASSERT_EQ(records.size(), size_t{TestSlots + 16} * RecordWords);
  uint32_t casters = 0;
  for (const scene::retained::InstanceInfo& info : infos) {
    const uint32_t* r = &records[size_t{info.slot} * RecordWords];
    ASSERT_NE(r[record::Flags] & instance_flag::Valid, 0u) << info.slot;
    EXPECT_EQ(record_is_caster(r[record::Flags], r[record::Categories]), is_caster(info))
        << "class " << info.classBits << " categories " << info.categories;
    EXPECT_EQ(test_bit(out.castBits, info.slot), is_caster(info))
        << "class " << info.classBits << " categories " << info.categories;
    casters += is_caster(info) ? 1 : 0;
  }
  EXPECT_EQ(popcount(out.castBits), casters);
  EXPECT_EQ(casters, 8u);

  const uint32_t freed = infos.front().slot;
  const uint32_t zero = 0;
  m_device->queue.WriteBuffer(m_instances, uint64_t{freed} * record::Bytes, &zero, 4);
  out = run(test_views(false), &m_instances);
  EXPECT_FALSE(test_bit(out.castBits, freed)) << "a freed record is never a caster";
  EXPECT_EQ(popcount(out.castBits), casters - 1);
  streamout::testing::reset();
  streamout::shutdown();
  scene::retained::testing::reset();
  gfx::set_blocking_pipelines(false);
}
constexpr uint64_t MarkLayoutKey = 0x76736D6D61726B01ull;
constexpr uint64_t MarkCheckLayoutKey = 0x76736D6D61726B02ull;
constexpr uint64_t ClassifyLayoutKey = 0x76736D636C617301ull;
constexpr uint64_t AssignLayoutKey = 0x76736D6173736701ull;
constexpr uint64_t CommitLayoutKey = 0x76736D636F6D6D01ull;
constexpr uint64_t ClearTestLayoutKey = 0x76736D636C720101ull;
constexpr uint64_t HookLayoutKey = 0x76736D686F6F6B01ull;
constexpr uint32_t MarkWidth = 96;
constexpr uint32_t MarkHeight = 64;

struct AllocOut {
  std::vector<uint32_t> table;
  std::vector<uint32_t> phys;
  std::vector<uint32_t> renderList;
  std::vector<uint32_t> stats;
  std::vector<uint32_t> clearArgs;
};

std::vector<uint32_t> compact_on_cpu(std::span<const uint32_t> words) {
  std::vector<uint32_t> out;
  for (uint32_t i = 0; i < words.size(); ++i) {
    if (words[i] != 0) {
      out.push_back(i);
    }
  }
  return out;
}

Views alloc_test_views(uint32_t poolPages, uint32_t budget, uint32_t clock) {
  Views views = test_views(false);
  views.header.poolPages = poolPages;
  views.header.poolPagesX = 32;
  views.header.budget = budget;
  views.header.clock = clock;
  return views;
}

class RenderVsmAllocGpu : public RenderVsmGpuDevice {
protected:
  void SetUp() override {
    RenderVsmGpuDevice::SetUp();
    if (IsSkipped()) {
      return;
    }
    const wgpu::Device& device = m_device->device;
    const wgpu::BindGroupLayoutDescriptor emptyDescriptor{};
    m_emptyLayout = device.CreateBindGroupLayout(&emptyDescriptor);
    const wgpu::BindGroupDescriptor emptyGroup{.layout = m_emptyLayout};
    m_empty = device.CreateBindGroup(&emptyGroup);
    m_classifyGroup = storage_layout(7, 2);
    m_assignGroup = storage_layout(8, 3);
    m_commitGroup = storage_layout(3, 1);
    m_hookGroup = storage_layout(3, 1);
    m_markGroup = mark_layout(false);
    m_markCheckGroup = mark_layout(true);
    const wgpu::BindGroupLayoutEntry uniform{
        .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform}};
    const wgpu::BindGroupLayoutDescriptor uniformDescriptor{.entryCount = 1, .entries = &uniform};
    m_paramsLayout = device.CreateBindGroupLayout(&uniformDescriptor);
    static constexpr std::array<std::string_view, 1> Classify{"VSM_CLASSIFY"};
    static constexpr std::array<std::string_view, 1> Assign{"VSM_ASSIGN"};
    static constexpr std::array<std::string_view, 1> Commit{"VSM_COMMIT"};
    static constexpr std::array<std::string_view, 1> Mark{"VSM_MARK"};
    static constexpr std::array<std::string_view, 2> MarkCheck{"VSM_MARK", "VSM_MARK_CHECK"};
    m_classify = pipeline(m_emptyLayout, m_classifyGroup, AllocShader, "vsm_classify", ClassifyLayoutKey, Classify);
    m_physClassify =
        pipeline(m_emptyLayout, m_classifyGroup, AllocShader, "vsm_phys_classify", ClassifyLayoutKey, Classify);
    m_assign = pipeline(m_emptyLayout, m_assignGroup, AllocShader, "vsm_assign", AssignLayoutKey, Assign);
    m_commit = pipeline(m_emptyLayout, m_commitGroup, AllocShader, "vsm_commit", CommitLayoutKey, Commit);
    m_mark = pipeline(m_paramsLayout, m_markGroup, AllocShader, "vsm_mark", MarkLayoutKey, Mark);
    m_markCheck = pipeline(m_paramsLayout, m_markCheckGroup, AllocShader, "vsm_mark", MarkCheckLayoutKey, MarkCheck);
    m_views = buffer(ViewsBytes);
    m_requests = buffer(RequestsBytes);
    m_table = buffer(TableBytes);
    m_phys = buffer(PhysBytes);
    m_stats = buffer(StatsBytes);
    m_need = buffer(NeedWordsBytes);
    m_physWords = buffer(PhysWordsBytes);
    m_needList = buffer(NeedWordsBytes);
    m_freeList = buffer(PhysWordsBytes);
    m_renderList = buffer(render_list::Bytes);
    m_clearArgs = buffer(16);
  }

  wgpu::BindGroupLayout storage_layout(uint32_t count, uint32_t readOnly) const {
    std::vector<wgpu::BindGroupLayoutEntry> entries(count);
    for (uint32_t i = 0; i < count; ++i) {
      entries[i] = {
          .binding = i,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type =
                         i < readOnly ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage},
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroupLayout(&descriptor);
  }

  wgpu::BindGroupLayout mark_layout(bool check) const {
    std::vector<wgpu::BindGroupLayoutEntry> entries;
    for (uint32_t i = 0; i < 2; ++i) {
      entries.push_back({
          .binding = i,
          .visibility = wgpu::ShaderStage::Compute,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      });
    }
    for (uint32_t i = 2; i < (check ? 5u : 4u); ++i) {
      entries.push_back({
          .binding = i,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = i == 2 ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage},
      });
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroupLayout(&descriptor);
  }

  wgpu::ComputePipeline pipeline(const wgpu::BindGroupLayout& first, const wgpu::BindGroupLayout& second,
                                 std::string_view shader, std::string_view entry, uint64_t key,
                                 std::span<const std::string_view> defines) const {
    const std::array layouts{first, second};
    const wgpu::PipelineLayoutDescriptor descriptor{.bindGroupLayoutCount = layouts.size(),
                                                    .bindGroupLayouts = layouts.data()};
    const shaders::PipelineId id = shaders::request_compute({
        .label = entry,
        .shader = shader,
        .entryPoint = entry,
        .defines = defines,
        .layout = m_device->device.CreatePipelineLayout(&descriptor),
        .layoutKey = key,
    });
    EXPECT_EQ(shaders::wait(id), shaders::PipelineStatus::Ready) << entry;
    return shaders::compute_pipeline(id);
  }

  wgpu::BindGroup group(const wgpu::BindGroupLayout& layout, std::span<const wgpu::Buffer> buffers) const {
    std::vector<wgpu::BindGroupEntry> entries(buffers.size());
    for (uint32_t i = 0; i < buffers.size(); ++i) {
      entries[i] = {.binding = i, .buffer = buffers[i], .size = buffers[i].GetSize()};
    }
    const wgpu::BindGroupDescriptor descriptor{
        .layout = layout, .entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroup(&descriptor);
  }

  void dispatch(const wgpu::ComputePipeline& pipe, const wgpu::BindGroup& first, const wgpu::BindGroup& second,
                uint32_t x, uint32_t y = 1) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipe);
    pass.SetBindGroup(0, first);
    pass.SetBindGroup(1, second);
    pass.DispatchWorkgroups(x, y);
    pass.End();
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  }

  void upload(const wgpu::Buffer& target, std::span<const uint32_t> words) const {
    if (!words.empty()) {
      m_device->queue.WriteBuffer(target, 0, words.data(), words.size_bytes());
    }
  }

  AllocOut run_alloc(const Views& views, std::span<const uint32_t> requests, std::span<const uint32_t> table,
                     std::span<const uint32_t> phys, bool commit = true) {
    upload(m_views, std::span{reinterpret_cast<const uint32_t*>(&views), ViewsBytes / 4});
    upload(m_requests, requests);
    upload(m_table, table);
    upload(m_phys, phys);
    const std::vector<uint32_t> zero(StatCount, 0);
    upload(m_stats, zero);
    const std::array classifyBuffers{m_requests, m_views, m_table, m_phys, m_stats, m_need, m_physWords};
    const wgpu::BindGroup classifyGroup = group(m_classifyGroup, classifyBuffers);
    dispatch(m_classify, m_empty, classifyGroup, MaxVirtualPages / AllocWorkgroup);
    dispatch(m_physClassify, m_empty, classifyGroup, MaxPoolPages / AllocWorkgroup);
    upload(m_needList, compact_on_cpu(read(m_need, NeedWordsBytes)));
    upload(m_freeList, compact_on_cpu(read(m_physWords, PhysWordsBytes)));
    const std::vector<uint32_t> blank(render_list::Words, 0);
    upload(m_renderList, blank);
    const std::array assignBuffers{m_needList, m_freeList, m_views,      m_table,
                                   m_phys,     m_stats,    m_renderList, m_clearArgs};
    dispatch(m_assign, m_empty, group(m_assignGroup, assignBuffers), MaxRenderPages / AllocWorkgroup);
    if (commit) {
      const std::array commitBuffers{m_renderList, m_table, m_stats};
      dispatch(m_commit, m_empty, group(m_commitGroup, commitBuffers), MaxRenderPages / AllocWorkgroup);
    }
    return {
        .table = read(m_table, TableBytes),
        .phys = read(m_phys, PhysBytes),
        .renderList = read(m_renderList, render_list::Bytes),
        .stats = read(m_stats, StatsBytes),
        .clearArgs = read(m_clearArgs, 16),
    };
  }

  wgpu::Texture texture(wgpu::TextureFormat format, uint32_t width, uint32_t height, wgpu::TextureUsage usage) const {
    const wgpu::TextureDescriptor descriptor{.usage = usage, .size = {width, height, 1}, .format = format};
    return m_device->device.CreateTexture(&descriptor);
  }

  std::vector<uint32_t> run_mark(const MarkParams& params, std::span<const float> depth, std::span<const float> alpha,
                                 std::vector<uint32_t>* receivers = nullptr) {
    const wgpu::Texture depthTexture = texture(wgpu::TextureFormat::R32Float, MarkWidth, MarkHeight,
                                               wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst);
    const wgpu::Texture normalTexture = texture(wgpu::TextureFormat::RGBA32Float, MarkWidth, MarkHeight,
                                                wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst);
    std::vector<float> normal(size_t{MarkWidth} * MarkHeight * 4, 0.5f);
    for (size_t i = 0; i < alpha.size(); ++i) {
      normal[i * 4 + 3] = alpha[i];
    }
    const auto write = [&](const wgpu::Texture& t, const void* data, uint32_t texelBytes) {
      const wgpu::TexelCopyTextureInfo destination{.texture = t};
      const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = MarkWidth * texelBytes, .rowsPerImage = MarkHeight};
      const wgpu::Extent3D extent{MarkWidth, MarkHeight, 1};
      m_device->queue.WriteTexture(&destination, data, size_t{MarkWidth} * MarkHeight * texelBytes, &layout, &extent);
    };
    write(depthTexture, depth.data(), 4);
    write(normalTexture, normal.data(), 16);
    const wgpu::Buffer uniform = [&] {
      const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                              .size = sizeof(MarkParams)};
      return m_device->device.CreateBuffer(&descriptor);
    }();
    m_device->queue.WriteBuffer(uniform, 0, &params, sizeof(params));
    const wgpu::BindGroupEntry paramsEntry{.binding = 1, .buffer = uniform, .size = sizeof(MarkParams)};
    const wgpu::BindGroupDescriptor paramsDescriptor{
        .layout = m_paramsLayout, .entryCount = 1, .entries = &paramsEntry};
    const wgpu::BindGroup paramsGroup = m_device->device.CreateBindGroup(&paramsDescriptor);
    const std::vector<uint32_t> zero(MaxVirtualPages, 0);
    upload(m_requests, zero);
    const wgpu::Buffer receiverBuffer = buffer(uint64_t{MarkWidth} * MarkHeight * ReceiverWords * 4);
    std::vector<wgpu::BindGroupEntry> entries{
        {.binding = 0, .textureView = depthTexture.CreateView()},
        {.binding = 1, .textureView = normalTexture.CreateView()},
        {.binding = 2, .buffer = m_views, .size = ViewsBytes},
        {.binding = 3, .buffer = m_requests, .size = RequestsBytes},
    };
    if (receivers != nullptr) {
      entries.push_back({.binding = 4, .buffer = receiverBuffer, .size = receiverBuffer.GetSize()});
    }
    const wgpu::BindGroupDescriptor descriptor{.layout = receivers != nullptr ? m_markCheckGroup : m_markGroup,
                                               .entryCount = entries.size(),
                                               .entries = entries.data()};
    const wgpu::BindGroup markGroup = m_device->device.CreateBindGroup(&descriptor);
    dispatch(receivers != nullptr ? m_markCheck : m_mark, paramsGroup, markGroup,
             (MarkWidth + MarkWorkgroup - 1) / MarkWorkgroup, (MarkHeight + MarkWorkgroup - 1) / MarkWorkgroup);
    if (receivers != nullptr) {
      *receivers = read(receiverBuffer, receiverBuffer.GetSize());
    }
    return read(m_requests, RequestsBytes);
  }

  wgpu::BindGroupLayout m_emptyLayout, m_paramsLayout, m_classifyGroup, m_assignGroup, m_commitGroup, m_hookGroup,
      m_markGroup, m_markCheckGroup;
  wgpu::BindGroup m_empty;
  wgpu::ComputePipeline m_classify, m_physClassify, m_assign, m_commit, m_mark, m_markCheck;
  wgpu::Buffer m_views, m_requests, m_table, m_phys, m_stats, m_need, m_physWords, m_needList, m_freeList, m_renderList,
      m_clearArgs;
};

MarkParams mark_params() {
  MarkParams params;
  params.camera = {
      .depthMul = -10.f,
      .depthAdd = 0.f,
      .p00 = 1.2f,
      .p11 = 1.8f,
      .p02 = 0.02f,
      .p12 = -0.01f,
      .jitterPx = {0.25f, -0.25f},
      .unitsPerMeter = 100.f,
      .flags = detail::CameraValid,
  };
  const double yaw = 0.6;
  params.rotation.rows[0][0] = static_cast<float>(std::cos(yaw));
  params.rotation.rows[0][2] = static_cast<float>(-std::sin(yaw));
  params.rotation.rows[1][1] = 1.f;
  params.rotation.rows[2][0] = static_cast<float>(std::sin(yaw));
  params.rotation.rows[2][2] = static_cast<float>(std::cos(yaw));
  params.width = MarkWidth;
  params.height = MarkHeight;
  return params;
}

TEST_F(RenderVsmAllocGpu, MarkMatchesCpuTwin) {
  Views views = test_views(false);
  views.header.levelBias = 0.5f;
  upload(m_views, std::span{reinterpret_cast<const uint32_t*>(&views), ViewsBytes / 4});
  const MarkParams params = mark_params();
  std::vector<float> depth(size_t{MarkWidth} * MarkHeight);
  std::vector<float> alpha(depth.size());
  for (uint32_t y = 0; y < MarkHeight; ++y) {
    for (uint32_t x = 0; x < MarkWidth; ++x) {
      const size_t i = size_t{y} * MarkWidth + x;
      const double z = 20.0 * std::exp2(0.12 * x + 0.05 * y);
      depth[i] = (x + y) % 17 == 0 ? 0.f : static_cast<float>(10.0 / z);
      alpha[i] = static_cast<float>((x * 7 + y * 3) % 4) / 3.f;
    }
  }
  std::vector<uint32_t> receivers;
  const std::vector<uint32_t> gpu = run_mark(params, depth, alpha, &receivers);
  ASSERT_EQ(gpu.size(), MaxVirtualPages);
  ASSERT_EQ(receivers.size(), size_t{MarkWidth} * MarkHeight * ReceiverWords);
  for (size_t i = 0; i < depth.size(); ++i) {
    EXPECT_EQ(receivers[i * 2], std::bit_cast<uint32_t>(depth[i])) << i;
    EXPECT_EQ(receivers[i * 2 + 1], std::bit_cast<uint32_t>(alpha[i])) << i;
  }
  const MarkReference ref = mark_reference(views, params, receivers);
  uint32_t mustPages = 0;
  uint32_t gpuPages = 0;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    EXPECT_EQ(ref.must[v] & ~gpu[v], 0u) << v;
    EXPECT_EQ(gpu[v] & ~(ref.must[v] | ref.may[v]), 0u) << v;
    mustPages += ref.must[v] != 0 ? 1 : 0;
    gpuPages += gpu[v] != 0 ? 1 : 0;
  }
  EXPECT_GT(ref.receivers, depth.size() / 3);
  EXPECT_LT(ref.receivers, depth.size());
  EXPECT_LT(ref.ambiguous, ref.receivers / 20);
  EXPECT_GT(mustPages, 20u);
  EXPECT_GE(gpuPages, mustPages);
  const std::vector<uint32_t> plain = run_mark(params, depth, alpha);
  EXPECT_EQ(plain, gpu);
  std::vector<float> sky(depth.size(), 0.f);
  const std::vector<uint32_t> none = run_mark(params, sky, alpha);
  EXPECT_TRUE(std::ranges::all_of(none, [](uint32_t w) { return w == 0; }));
  std::vector<float> codeOne(alpha.size(), 1.f / 3.f);
  const std::vector<uint32_t> unlit = run_mark(params, depth, codeOne);
  EXPECT_TRUE(std::ranges::all_of(unlit, [](uint32_t w) { return w == 0; }));
}

struct AllocState {
  std::vector<uint32_t> requests = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> table = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> phys = std::vector<uint32_t>(size_t{MaxPoolPages} * PhysWords, 0);
  uint32_t nextPhys = 0;

  void own(uint32_t v, uint32_t bits, uint32_t lastUsed) {
    const uint32_t p = nextPhys++;
    table[v] = p | entry::Allocated | bits;
    phys[size_t{p} * PhysWords] = v + 1;
    phys[size_t{p} * PhysWords + 1] = lastUsed;
  }
};

AllocState mixed_state(uint32_t clock) {
  AllocState s;
  std::mt19937 rng{77};
  std::uniform_int_distribution<uint32_t> pick(0, 11);
  for (uint32_t v = 0; v < SunLevels * PagesPerView; v += 7) {
    const uint32_t r = pick(rng);
    if (r < 4 && s.nextPhys < 700) {
      const uint32_t bits = r == 0   ? entry::Cached | entry::Dirty | entry::Priority
                            : r == 1 ? entry::Cached | entry::Dirty
                            : r == 2 ? entry::Cached
                                     : 0u;
      s.own(v, bits, clock - std::min(clock, pick(rng) * 9));
    }
    if (pick(rng) < 6) {
      s.requests[v] = pick(rng) < 6 ? request_bit::Pixel : request_bit::Fallback;
    }
  }
  return s;
}

TEST_F(RenderVsmAllocGpu, AllocIsDeterministic) {
  const Views views = alloc_test_views(1024, 64, 120);
  const AllocState s = mixed_state(120);
  std::array<AllocOut, 3> runs;
  for (AllocOut& run : runs) {
    run = run_alloc(views, s.requests, s.table, s.phys);
  }
  for (size_t i = 1; i < runs.size(); ++i) {
    EXPECT_EQ(runs[i].table, runs[0].table);
    EXPECT_EQ(runs[i].phys, runs[0].phys);
    EXPECT_EQ(runs[i].renderList, runs[0].renderList);
  }
  std::vector<uint32_t> table = s.table;
  std::vector<uint32_t> phys = s.phys;
  std::vector<uint32_t> list(render_list::Words, 0);
  const AllocCounts twin = allocate(views, s.requests, table, phys, list);
  const uint32_t committed = commit(list, table);
  EXPECT_EQ(runs[0].renderList, list);
  EXPECT_EQ(runs[0].table, table);
  EXPECT_EQ(runs[0].phys, phys);
  const std::vector<uint32_t>& st = runs[0].stats;
  EXPECT_EQ(st[stat::Requested], twin.requested);
  EXPECT_EQ(st[stat::NeedPriority], twin.priority);
  EXPECT_EQ(st[stat::NeedNew], twin.fresh);
  EXPECT_EQ(st[stat::NeedReuse], twin.reuse);
  EXPECT_EQ(st[stat::Hits], twin.hits);
  EXPECT_EQ(st[stat::FreedDirty], twin.freedDirty);
  EXPECT_EQ(st[stat::Evictable], twin.evictable);
  EXPECT_EQ(st[stat::Rendered], twin.rendered);
  EXPECT_EQ(st[stat::DeferredNew], twin.deferredNew);
  EXPECT_EQ(st[stat::DeferredReuse], twin.deferredReuse);
  EXPECT_EQ(st[stat::Evicted], twin.evicted);
  EXPECT_EQ(st[stat::Committed], committed);
  EXPECT_EQ(st[stat::PlayerDeferred], 0u);
  EXPECT_GT(twin.priority, 0u);
  EXPECT_GT(twin.deferredNew, 0u);
  EXPECT_EQ(runs[0].clearArgs, (std::vector<uint32_t>{6, twin.rendered, 0, 0}));
  std::vector<uint32_t> viewsWords(ViewsBytes / 4);
  std::memcpy(viewsWords.data(), &views, ViewsBytes);
  EXPECT_EQ(check::verify(views, viewsWords, runs[0].table, runs[0].phys), "");
}

TEST_F(RenderVsmAllocGpu, BudgetDefersAndFallsBack) {
  const Views views = alloc_test_views(1024, 8, 30);
  AllocState s;
  const uint32_t fine = level_view(2) * PagesPerView;
  const uint32_t coarse = level_view(3) * PagesPerView;
  for (uint32_t i = 0; i < 5; ++i) {
    s.own(fine + 900 + i, entry::Cached | entry::Dirty | entry::Priority, 29);
    s.requests[fine + 900 + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 40; ++i) {
    s.requests[fine + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 20; ++i) {
    s.own(coarse + i, entry::Cached, 29);
    s.requests[coarse + i] = request_bit::Fallback;
  }
  const AllocOut out = run_alloc(views, s.requests, s.table, s.phys);
  EXPECT_EQ(out.renderList[render_list::Count], 13u);
  EXPECT_EQ(out.stats[stat::DeferredNew], 32u);
  EXPECT_EQ(out.stats[stat::PlayerDeferred], 0u);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(out.renderList[render_list::Pages + 2 * i], fine + 900 + i);
  }
  uint32_t deferred = 0;
  for (uint32_t i = 0; i < 40; ++i) {
    const uint32_t e = out.table[fine + i];
    if ((e & entry::Allocated) == 0) {
      ++deferred;
    } else {
      EXPECT_NE(e & entry::Cached, 0u);
    }
  }
  EXPECT_EQ(deferred, 32u);
  for (uint32_t i = 0; i < 20; ++i) {
    const uint32_t e = out.table[coarse + i];
    EXPECT_EQ(e & (entry::Allocated | entry::Cached | entry::RenderNow), entry::Allocated | entry::Cached) << i;
  }
  EXPECT_EQ(out.stats[stat::Hits], 20u);
}

TEST_F(RenderVsmAllocGpu, PoolOverflowCounts) {
  const Views views = alloc_test_views(MinPoolPages, MinPoolPages, 10);
  AllocState s;
  const uint32_t base = level_view(5) * PagesPerView;
  for (uint32_t i = 0; i < 200; ++i) {
    s.own(base + i, entry::Cached, 10);
    s.requests[base + i] = request_bit::Pixel;
  }
  for (uint32_t i = 0; i < 90; ++i) {
    s.requests[base + 300 + i] = request_bit::Pixel;
  }
  const AllocOut out = run_alloc(views, s.requests, s.table, s.phys);
  EXPECT_EQ(out.stats[stat::Evictable], MinPoolPages - 200);
  EXPECT_EQ(out.stats[stat::OverflowPool], 90 - (MinPoolPages - 200));
  EXPECT_EQ(out.renderList[render_list::Count], 90u);
  uint32_t sentinels = 0;
  for (uint32_t i = 0; i < 90; ++i) {
    sentinels += out.renderList[render_list::Pages + 2 * i] == NoPage ? 1 : 0;
  }
  EXPECT_EQ(sentinels, 90 - (MinPoolPages - 200));
  for (uint32_t i = 0; i < 200; ++i) {
    EXPECT_EQ(out.table[base + i], s.table[base + i]) << "a requested page was evicted";
  }
}

TEST_F(RenderVsmAllocGpu, FreshPoolHasNoOwners) {
  const Views views = alloc_test_views(1024, 1024, 1);
  AllocState s;
  for (uint32_t v = 0; v < SunLevels * PagesPerView; v += 13) {
    s.requests[v] = request_bit::Pixel;
  }
  const AllocOut out = run_alloc(views, s.requests, s.table, s.phys);
  std::vector<bool> listed(MaxPoolPages, false);
  const uint32_t count = out.renderList[render_list::Count];
  EXPECT_EQ(count, out.stats[stat::NeedNew]);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t p = out.renderList[render_list::Pages + 2 * i + 1];
    ASSERT_LT(p, 1024u);
    EXPECT_EQ(p, i) << "a fresh pool hands out pages in order";
    listed[p] = true;
  }
  for (uint32_t p = 0; p < MaxPoolPages; ++p) {
    EXPECT_EQ(out.phys[size_t{p} * PhysWords] != 0, bool{listed[p]}) << p;
  }
  std::vector<uint32_t> viewsWords(ViewsBytes / 4);
  std::memcpy(viewsWords.data(), &views, ViewsBytes);
  EXPECT_EQ(check::verify(views, viewsWords, out.table, out.phys), "");
}

TEST_F(RenderVsmAllocGpu, HookRequestAllocates) {
  constexpr std::string_view Hook = R"(//#import aurora/vsm
@group(1) @binding(0) var<storage, read> views: array<u32>;
@group(1) @binding(1) var<storage, read_write> requests: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> out_levels: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec3f(f32(id.x) * 40.0 - 60.0, -15.0, f32(id.x) * 25.0);
  let f = 0.8 * f32(1u << id.x);
  vsm_request_world(&requests, &views, p, f, VSM_REQUEST_VOLUME);
  out_levels[id.x] = vsm_select_level(&views, f, p, vsm_f32(views[VSM_HDR_LEVEL_BIAS]));
  let v = vsm_view(&views, vsm_level_view(9u));
  vsm_request(&requests, v, v.origin + vec2i(3, 4), VSM_REQUEST_VOLUME | VSM_REQUEST_PRIORITY);
  vsm_request(&requests, v, v.origin + vec2i(-1, 4), VSM_REQUEST_VOLUME);
}
)";
  ASSERT_TRUE(shaders::register_source("test/vsm_hook", Hook));
  const wgpu::ComputePipeline hook = pipeline(m_emptyLayout, m_hookGroup, "test/vsm_hook", "main", HookLayoutKey, {});
  const Views views = alloc_test_views(1024, 64, 3);
  upload(m_views, std::span{reinterpret_cast<const uint32_t*>(&views), ViewsBytes / 4});
  const std::vector<uint32_t> zero(MaxVirtualPages, 0);
  upload(m_requests, zero);
  const wgpu::Buffer levels = buffer(16);
  const std::array hookBuffers{m_views, m_requests, levels};
  dispatch(hook, m_empty, group(m_hookGroup, hookBuffers), 1);
  const std::vector<uint32_t> requests = read(m_requests, RequestsBytes);
  const std::vector<uint32_t> gotLevels = read(levels, 16);
  std::vector<uint32_t> want(MaxVirtualPages, 0);
  for (uint32_t i = 0; i < 4; ++i) {
    const std::array<float, 3> p{static_cast<float>(i) * 40.f - 60.f, -15.f, static_cast<float>(i) * 25.f};
    const float f = 0.8f * static_cast<float>(1u << i);
    EXPECT_EQ(gotLevels[i], select_level(views, f, p, views.header.levelBias)) << i;
    const MarkKeys keys = mark_keys(views, p, f);
    for (uint32_t k = 0; k < keys.count; ++k) {
      const uint32_t bits = keys.keys[k] >> 16;
      want[keys.keys[k] & 0xFFFFu] |= bits == request_bit::Pixel ? request_bit::Volume : bits;
    }
  }
  const ViewRecord& v9 = views.views[level_view(9)];
  want[page_index(v9.table_offset(), v9.origin[0] + 3, v9.origin[1] + 4)] |=
      request_bit::Volume | request_bit::Priority;
  EXPECT_EQ(requests, want);
  AllocState s;
  s.requests = requests;
  const AllocOut out = run_alloc(views, s.requests, s.table, s.phys);
  uint32_t requested = 0;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    if (want[v] == 0) {
      continue;
    }
    ++requested;
    EXPECT_EQ(out.table[v] & (entry::Allocated | entry::Cached), entry::Allocated | entry::Cached) << v;
  }
  EXPECT_EQ(out.renderList[render_list::Count], requested);
  EXPECT_GE(requested, 5u);
}

TEST_F(RenderVsmAllocGpu, ClearWritesOnlyListedPages) {
  const wgpu::Device& device = m_device->device;
  std::array<wgpu::BindGroupLayoutEntry, 2> entries{};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i] = {.binding = i,
                  .visibility = wgpu::ShaderStage::Vertex,
                  .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}};
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{.entryCount = entries.size(), .entries = entries.data()};
  const wgpu::BindGroupLayout clearGroup = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{.bindGroupLayoutCount = 1, .bindGroupLayouts = &clearGroup};
  static constexpr std::array<std::string_view, 1> Clear{"VSM_CLEAR"};
  const shaders::PipelineId id = shaders::request_render({
      .label = "vsm.clear test",
      .shader = AllocShader,
      .vertexEntry = "vs_clear",
      .fragmentEntry = {},
      .defines = Clear,
      .layout = device.CreatePipelineLayout(&layoutDescriptor),
      .layoutKey = ClearTestLayoutKey,
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .depthWrite = true,
      .depthCompare = wgpu::CompareFunction::Always,
  });
  ASSERT_EQ(shaders::wait(id), shaders::PipelineStatus::Ready);
  constexpr uint32_t PagesX = 4;
  constexpr uint32_t Rows = 3;
  constexpr uint32_t Width = PagesX * PageTexels;
  constexpr uint32_t Height = Rows * PageTexels;
  const wgpu::Texture atlas = texture(wgpu::TextureFormat::Depth32Float, Width, Height,
                                      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc);
  std::vector<uint32_t> list(render_list::Words, 0);
  list[render_list::Count] = 4;
  list[render_list::PagesX] = PagesX;
  list[render_list::Rows] = Rows;
  const std::array<std::array<uint32_t, 2>, 4> pages{{{100, 1}, {200, 6}, {NoPage, NoPage}, {300, 11}}};
  for (uint32_t i = 0; i < pages.size(); ++i) {
    list[render_list::Pages + 2 * i] = pages[i][0];
    list[render_list::Pages + 2 * i + 1] = pages[i][1];
  }
  std::vector<uint32_t> table(MaxVirtualPages, 0);
  table[200] = entry::Skip;
  upload(m_renderList, list);
  upload(m_table, table);
  const std::array clearBuffers{m_renderList, m_table};
  const wgpu::BindGroup bind = group(clearGroup, clearBuffers);
  const std::array<uint32_t, 4> args{6, 4, 0, 0};
  m_device->queue.WriteBuffer(m_clearArgs, 0, args.data(), sizeof(args));
  const wgpu::Buffer indirect = [&] {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst,
                                            .size = sizeof(args)};
    return device.CreateBuffer(&descriptor);
  }();
  m_device->queue.WriteBuffer(indirect, 0, args.data(), sizeof(args));
  const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  {
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = atlas.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 0.5f,
    };
    const wgpu::RenderPassDescriptor descriptor{.depthStencilAttachment = &depth};
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&descriptor);
    pass.End();
  }
  {
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = atlas.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Load,
        .depthStoreOp = wgpu::StoreOp::Store,
    };
    const wgpu::RenderPassDescriptor descriptor{.depthStencilAttachment = &depth};
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&descriptor);
    pass.SetPipeline(shaders::render_pipeline(id));
    pass.SetBindGroup(0, bind);
    pass.DrawIndirect(indirect, 0);
    pass.End();
  }
  const wgpu::Buffer readback = buffer(uint64_t{Width} * Height * 4);
  const wgpu::TexelCopyTextureInfo source{.texture = atlas, .aspect = wgpu::TextureAspect::DepthOnly};
  const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = Width * 4, .rowsPerImage = Height},
                                              .buffer = readback};
  const wgpu::Extent3D extent{Width, Height, 1};
  encoder.CopyTextureToBuffer(&source, &destination, &extent);
  ASSERT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  const std::vector<uint32_t> texels = read(readback, readback.GetSize());
  ASSERT_EQ(texels.size(), size_t{Width} * Height);
  const std::array<bool, PagesX * Rows> cleared{false, true,  false, false, false, false,
                                                false, false, false, false, false, true};
  for (uint32_t y = 0; y < Height; ++y) {
    for (uint32_t x = 0; x < Width; ++x) {
      const uint32_t page = (y / PageTexels) * PagesX + x / PageTexels;
      const float d = std::bit_cast<float>(texels[size_t{y} * Width + x]);
      EXPECT_EQ(d, cleared[page] ? 0.f : 0.5f) << x << "," << y;
      if (d != (cleared[page] ? 0.f : 0.5f)) {
        return;
      }
    }
  }
}

TEST_F(RenderVsmAllocGpu, CommitNeedsWholeChain) {
  const Views views = alloc_test_views(1024, 64, 4);
  AllocState s;
  const uint32_t base = level_view(1) * PagesPerView;
  for (uint32_t i = 0; i < 12; ++i) {
    s.requests[base + i] = request_bit::Pixel;
  }
  s.own(base + 20, entry::Cached | entry::Dirty, 3);
  s.requests[base + 20] = request_bit::Pixel;
  const AllocOut held = run_alloc(views, s.requests, s.table, s.phys, false);
  for (uint32_t i = 0; i < 12; ++i) {
    const uint32_t e = held.table[base + i];
    EXPECT_NE(e & entry::RenderNow, 0u) << i;
    EXPECT_EQ(e & entry::Cached, 0u) << i;
  }
  const uint32_t stale = held.table[base + 20];
  EXPECT_NE(stale & entry::RenderNow, 0u);
  EXPECT_NE(stale & entry::Cached, 0u) << "an unfinished re-render keeps the old Cached bit but is not usable";
  EXPECT_EQ(held.stats[stat::Committed], 0u);
  testing::reset();
  FrameInput in{.frameIndex = 1,
                .hintFrame = 1,
                .eye = TestEye,
                .planDir = sun_dir(38.0, 140.0),
                .lightId = 2,
                .texel0 = TestTexel0,
                .producing = true,
                .coverageComplete = true,
                .pages = 1024,
                .pagesX = 32,
                .budget = 64,
                .generation = 1,
                .chainMask = chain_bit::Assign | chain_bit::Commit};
  (void)decide_frame(in);
  mark_encoded(chain_bit::Assign);
  in.frameIndex = in.hintFrame = 2;
  const FrameDecision next = decide_frame(in);
  EXPECT_TRUE(next.full);
  EXPECT_EQ(next.reasons, full_reason::ChainIncomplete);
  testing::reset();
}
constexpr uint64_t ChunkLayoutKey = 0x76736D63686B0001ull;
constexpr uint64_t CandidateLayoutKey = 0x76736D63616E0001ull;
constexpr uint64_t CountLayoutKey = 0x76736D636E740001ull;
constexpr uint64_t EmitLayoutKey = 0x76736D656D740001ull;
constexpr uint64_t SkipLayoutKey = 0x76736D736B700001ull;
constexpr uint32_t TestClusters = 4096;

struct CullScene {
  Views views;
  std::vector<uint32_t> history = std::vector<uint32_t>(size_t{Slots} * HistoryWords, 0);
  std::vector<uint32_t> clusters = std::vector<uint32_t>(size_t{TestClusters} * 16, 0);
  std::vector<uint32_t> chunkBoxes = std::vector<uint32_t>(size_t{ChunkBoxCount} * 2, 0);
  std::vector<uint32_t> materials = std::vector<uint32_t>(MaterialBytes / 4, 0);
  std::vector<uint32_t> requests = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> table = std::vector<uint32_t>(MaxVirtualPages, 0);
  std::vector<uint32_t> phys = std::vector<uint32_t>(size_t{MaxPoolPages} * PhysWords, 0);
  std::vector<uint32_t> renderList = std::vector<uint32_t>(render_list::Words, 0);

  std::span<uint32_t, HistoryWords> hist(uint32_t slot) {
    return std::span{history}.subspan(size_t{slot} * HistoryWords).first<HistoryWords>();
  }
  void add(uint32_t slot, const Record& rec) {
    const std::array<uint32_t, HistoryWords> prev{};
    const SlotScan scan = scan_slot(prev, rec, materials, views.header);
    std::ranges::copy(scan.history, hist(slot).begin());
  }
  void cluster(uint32_t index, const Box& local, bool posed) {
    uint32_t* c = &clusters[size_t{index} * 16];
    c[2] = 64u | 124u << 8 | (posed ? 1u : 0u) << 16;
    for (uint32_t k = 0; k < 3; ++k) {
      c[8 + k] = std::bit_cast<uint32_t>(local.lo[k]);
      c[12 + k] = std::bit_cast<uint32_t>(local.hi[k]);
    }
  }
  void request_level(uint32_t level) {
    for (uint32_t i = 0; i < PagesPerView; ++i) {
      requests[level_view(level) * PagesPerView + i] = request_bit::Pixel;
    }
  }
  void allocate_pages() { (void)allocate(views, requests, table, phys, renderList); }
};

struct CullOut {
  std::vector<uint32_t> pairs;
  std::vector<uint32_t> args;
  std::vector<uint32_t> counts;
  std::vector<uint32_t> table;
  std::vector<uint32_t> stats;
  uint32_t listed = 0;
};

Views cull_views() {
  Views views = test_views(false);
  views.header.poolPages = MaxPoolPages;
  views.header.poolPagesX = 64;
  views.header.budget = MaxPoolPages;
  views.header.clock = 2;
  return views;
}

Record cluster_record(uint32_t hash, const std::array<double, 3>& t, float half, uint32_t first, uint32_t count,
                      uint32_t categories = AURORA_CAT_STATIC, uint32_t flags = 0) {
  Record rec = make_record(hash, t, half, categories, flags);
  rec[record::ClusterFirst] = first;
  rec[record::ClusterCount] = count;
  return rec;
}

class RenderVsmCullGpu : public RenderVsmAllocGpu {
protected:
  void SetUp() override {
    RenderVsmAllocGpu::SetUp();
    if (IsSkipped()) {
      return;
    }
    m_chunkGroup = pattern_layout("rrrrww");
    m_candidateGroup = pattern_layout("wrrww");
    m_countGroup = pattern_layout("rrrrrrw");
    m_emitGroup = pattern_layout("rrrrrrww");
    m_skipGroup = pattern_layout("rrrrwwww");
    static constexpr std::array<std::string_view, 1> Chunk{"VSM_CHUNK_BOUNDS"};
    static constexpr std::array<std::string_view, 1> Candidates{"VSM_CULL_CANDIDATES"};
    static constexpr std::array<std::string_view, 1> Count{"VSM_CULL_PAIRS"};
    static constexpr std::array<std::string_view, 2> Emit{"VSM_CULL_EMIT", "VSM_CULL_PAIRS"};
    static constexpr std::array<std::string_view, 1> Skip{"VSM_CULL_SKIP"};
    m_chunkBounds = pipeline(m_emptyLayout, m_chunkGroup, CullShader, "vsm_chunk_bounds", ChunkLayoutKey, Chunk);
    m_candidates =
        pipeline(m_emptyLayout, m_candidateGroup, CullShader, "vsm_cull_candidates", CandidateLayoutKey, Candidates);
    m_count = pipeline(m_emptyLayout, m_countGroup, CullShader, "vsm_cull_pairs", CountLayoutKey, Count);
    m_emit = pipeline(m_emptyLayout, m_emitGroup, CullShader, "vsm_cull_pairs", EmitLayoutKey, Emit);
    m_skip = pipeline(m_emptyLayout, m_skipGroup, CullShader, "vsm_cull_skip", SkipLayoutKey, Skip);
    m_skipArgs = pipeline(m_emptyLayout, m_skipGroup, CullShader, "vsm_cull_args", SkipLayoutKey, Skip);
    m_history = buffer(HistoryBytes);
    m_clusters = buffer(uint64_t{TestClusters} * 64);
    m_chunkBoxes = buffer(ChunkBoxesBytes);
    m_cullWords = buffer(CullWordsBytes);
    m_cullList = buffer(CullWordsBytes);
    m_counts = buffer(PageCountBytes);
    m_offsets = buffer(PageCountBytes);
    m_pairs = buffer(pairs_header::Bytes);
    m_rasterArgs = buffer(RasterArgsBytes);
  }

  wgpu::BindGroupLayout pattern_layout(std::string_view pattern) const {
    std::vector<wgpu::BindGroupLayoutEntry> entries(pattern.size());
    for (uint32_t i = 0; i < pattern.size(); ++i) {
      entries[i] = {
          .binding = i,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = pattern[i] == 'r' ? wgpu::BufferBindingType::ReadOnlyStorage
                                               : wgpu::BufferBindingType::Storage},
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{.entryCount = entries.size(), .entries = entries.data()};
    return m_device->device.CreateBindGroupLayout(&descriptor);
  }

  CullOut run_cull(const CullScene& sc, uint32_t maxPairs = MaxPairs) {
    Views views = sc.views;
    views.header.maxPairs = maxPairs;
    upload(m_views, std::span{reinterpret_cast<const uint32_t*>(&views), ViewsBytes / 4});
    upload(m_history, sc.history);
    upload(m_table, sc.table);
    upload(m_renderList, sc.renderList);
    upload(m_clusters, sc.clusters);
    upload(m_chunkBoxes, sc.chunkBoxes);
    upload(m_stats, std::vector<uint32_t>(StatCount, 0));
    const std::array candidateBuffers{m_history, m_views, m_renderList, m_cullWords, m_stats};
    dispatch(m_candidates, m_empty, group(m_candidateGroup, candidateBuffers), Slots / CullWorkgroup);
    const std::vector<uint32_t> list = compact_on_cpu(read(m_cullWords, CullWordsBytes));
    upload(m_cullList, list);
    CullOut out;
    out.listed = static_cast<uint32_t>(list.size());
    upload(m_counts, std::vector<uint32_t>(PageCountWords, 0));
    const std::array countBuffers{m_cullList, m_history, m_clusters, m_chunkBoxes, m_views, m_table, m_counts};
    dispatch(m_count, m_empty, group(m_countGroup, countBuffers), out.listed);
    out.counts = read(m_counts, PageCountBytes);
    std::vector<uint32_t> offsets(PageCountWords, 0);
    uint32_t running = 0;
    for (uint32_t i = 0; i < PageCountWords; ++i) {
      offsets[i] = running;
      running += out.counts[i];
    }
    upload(m_offsets, offsets);
    upload(m_pairs, std::vector<uint32_t>(pairs_header::Words, 0));
    const std::array skipBuffers{m_counts, m_offsets, m_renderList, m_views, m_table, m_stats, m_pairs, m_rasterArgs};
    const wgpu::BindGroup skipGroup = group(m_skipGroup, skipBuffers);
    dispatch(m_skip, m_empty, skipGroup, MaxRenderPages / CullWorkgroup);
    dispatch(m_skipArgs, m_empty, skipGroup, 1);
    const std::array emitBuffers{m_cullList, m_history, m_clusters, m_chunkBoxes, m_views, m_table, m_offsets, m_pairs};
    dispatch(m_emit, m_empty, group(m_emitGroup, emitBuffers), out.listed);
    out.pairs = read(m_pairs, pairs_header::Bytes);
    out.args = read(m_rasterArgs, RasterArgsBytes);
    out.table = read(m_table, TableBytes);
    out.stats = read(m_stats, StatsBytes);
    return out;
  }

  static std::vector<uint64_t> pair_keys(const CullOut& out) {
    std::vector<uint64_t> keys;
    const uint32_t total = out.pairs[pairs_header::OpaqueCount] + out.pairs[pairs_header::CutoutCount];
    for (uint32_t i = 0; i < total; ++i) {
      const std::array<uint32_t, 2> w{out.pairs[pairs_header::Words + 2 * i],
                                      out.pairs[pairs_header::Words + 2 * i + 1]};
      const Pair p = pair_unpack(w);
      keys.push_back(pair_key(p.renderIndex, w[0], p.slot));
    }
    std::ranges::sort(keys);
    return keys;
  }

  CullScene mixed_scene() {
    CullScene sc{.views = cull_views()};
    sc.materials[5 * 8 + 1] = 1u << 16;
    sc.materials[6 * 8 + 1] = (1u | 16u) << 16;
    sc.add(1, cluster_record(11, offset(0.0, 0.0, 0.0), 150.f, 100, 4));
    for (uint32_t k = 0; k < 4; ++k) {
      const float x0 = (k & 1) != 0 ? 0.f : -150.f;
      const float z0 = (k & 2) != 0 ? 0.f : -150.f;
      sc.cluster(100 + k, {.lo = {x0, -150.f, z0}, .hi = {x0 + 150.f, 150.f, z0 + 150.f}}, false);
    }
    sc.add(2, cluster_record(12, offset(400.0, 0.0, 200.0), 60.f, 200, 3, AURORA_CAT_PLAYER, instance_flag::Palette));
    for (uint32_t k = 0; k < 3; ++k) {
      sc.cluster(200 + k, {.lo = {-60.f, -60.f, -60.f}, .hi = {60.f, 60.f, 60.f}}, true);
    }
    Record chunked = cluster_record(13, offset(-500.0, 20.0, -300.0), 80.f, 0, 0, AURORA_CAT_DYNAMIC);
    chunked[record::IndexFirst] = 640;
    chunked[record::IndexCount] = 600;
    sc.add(3, chunked);
    const std::span<const uint32_t, 12> xform = sc.hist(3).subspan<hist::Xform, 12>();
    const Box bounds = history_box(sc.hist(3));
    for (uint32_t c = 0; c < 2; ++c) {
      const Box local{.lo = {-80.f, -80.f + 80.f * static_cast<float>(c), -80.f},
                      .hi = {80.f, static_cast<float>(c) * 80.f, 80.f}};
      const std::array<uint32_t, 2> words = chunk_box_pack(box_through(local, xform), bounds);
      sc.chunkBoxes[size_t{chunk_slot(640, c)} * 2] = words[0];
      sc.chunkBoxes[size_t{chunk_slot(640, c)} * 2 + 1] = words[1];
    }
    Record cutout = cluster_record(14, offset(200.0, 0.0, -400.0), 40.f, 300, 1, AURORA_CAT_DYNAMIC);
    cutout[record::Material] = 5 | 0xFF0000u;
    sc.add(4, cutout);
    Record pending = cluster_record(15, offset(-200.0, 0.0, 400.0), 40.f, 301, 1, AURORA_CAT_DYNAMIC);
    pending[record::Material] = 6 | 0xFF0000u;
    sc.add(5, pending);
    sc.add(6, cluster_record(16, offset(0.0, 0.0, 600.0), 40.f, 302, 1, AURORA_CAT_NO_SHADOW_CAST));
    Record freed = cluster_record(17, offset(0.0, 0.0, -700.0), 40.f, 303, 1);
    freed[record::Flags] = 0;
    sc.add(7, freed);
    for (uint32_t k = 300; k < 304; ++k) {
      sc.cluster(k, {.lo = {-40.f, -40.f, -40.f}, .hi = {40.f, 40.f, 40.f}}, false);
    }
    sc.request_level(2);
    sc.request_level(3);
    sc.allocate_pages();
    return sc;
  }

  wgpu::BindGroupLayout m_chunkGroup, m_candidateGroup, m_countGroup, m_emitGroup, m_skipGroup;
  wgpu::ComputePipeline m_chunkBounds, m_candidates, m_count, m_emit, m_skip, m_skipArgs;
  wgpu::Buffer m_history, m_clusters, m_chunkBoxes, m_cullWords, m_cullList, m_counts, m_offsets, m_pairs, m_rasterArgs;
};

TEST_F(RenderVsmCullGpu, CullMatchesCpuReference) {
  const CullScene sc = mixed_scene();
  ASSERT_EQ(sc.renderList[render_list::Count], 2 * PagesPerView);
  const CullOut out = run_cull(sc);
  const std::vector<uint64_t> gpu = pair_keys(out);
  ASSERT_FALSE(gpu.empty());
  const CullReference ref = cull_reference(sc.views, sc.history, sc.clusters, sc.chunkBoxes, sc.renderList, 2.f);
  for (const uint64_t k : ref.exact) {
    EXPECT_TRUE(std::ranges::binary_search(gpu, k)) << "missing " << std::hex << k;
  }
  for (const uint64_t k : gpu) {
    EXPECT_TRUE(std::ranges::binary_search(ref.loose, k)) << "beyond the margin " << std::hex << k;
  }
  EXPECT_GE(gpu.size(), ref.exact.size());
  const uint32_t opaque = out.pairs[pairs_header::OpaqueCount];
  const uint32_t cutout = out.pairs[pairs_header::CutoutCount];
  std::array<uint32_t, 8> perSlot{};
  uint32_t chunkPairs = 0;
  for (uint32_t i = 0; i < opaque + cutout; ++i) {
    const Pair p = pair_unpack({out.pairs[pairs_header::Words + 2 * i], out.pairs[pairs_header::Words + 2 * i + 1]});
    ASSERT_LT(p.slot, perSlot.size());
    ++perSlot[p.slot];
    EXPECT_EQ(p.slot == 4, i >= opaque) << i;
    EXPECT_EQ(p.chunk, p.slot == 3) << i;
    chunkPairs += p.chunk ? 1 : 0;
  }
  for (const uint32_t slot : {1u, 2u, 3u, 4u, 5u}) {
    EXPECT_GT(perSlot[slot], 0u) << slot;
  }
  EXPECT_EQ(perSlot[6], 0u);
  EXPECT_EQ(perSlot[7], 0u);
  EXPECT_EQ(out.pairs[pairs_header::ChunkPairs], chunkPairs);
  EXPECT_EQ(out.pairs[pairs_header::PosedPairs], perSlot[2]);
  EXPECT_EQ(out.pairs[pairs_header::Skipped], 0u);
  EXPECT_EQ(out.pairs[pairs_header::Dropped], 0u);
  EXPECT_EQ(out.args, (std::vector<uint32_t>{VerticesPerPair, opaque, 0, 0, VerticesPerPair, cutout, 0, 0}));
  uint64_t sum = 0;
  for (const uint32_t c : out.counts) {
    sum += c;
  }
  EXPECT_EQ(sum, opaque + cutout);
  EXPECT_EQ(out.stats[stat::OpaquePairs], opaque);
  EXPECT_EQ(out.stats[stat::CutoutPairs], cutout);
  EXPECT_EQ(out.stats[stat::CullSlots], 5u);
  check::CullSummary summary;
  const check::CullReadback readback{.history = sc.history,
                                     .clusters = sc.clusters,
                                     .chunkBoxes = sc.chunkBoxes,
                                     .renderList = sc.renderList,
                                     .pairs = out.pairs,
                                     .pageCounts = out.counts};
  Views views = sc.views;
  views.header.maxPairs = MaxPairs;
  EXPECT_EQ(check::verify_cull(views, readback, summary), "");
  EXPECT_EQ(summary.gpuPairs, gpu.size());
}

TEST_F(RenderVsmCullGpu, SmallClustersCast) {
  CullScene sc{.views = cull_views()};
  SunClipmap overhead;
  (void)overhead.update({.eye = TestEye, .dir = {0.0, -1.0, 0.0}, .lightId = 2, .texel0 = TestTexel0});
  overhead.fill(sc.views);
  const ViewRecord& view = sc.views.views[level_view(0)];
  std::array<std::array<double, 3>, 3> basis{};
  for (uint32_t i = 0; i < 3; ++i) {
    basis[0][i] = double{view.anchorRel[i]} * view.texel;
    basis[1][i] = double{view.anchorRel[4 + i]} * view.texel;
    basis[2][i] = -double{view.anchorRel[8 + i]} / view.depthScale;
  }
  const double a = (2048.0 - view.anchorRel[3]) * view.texel;
  const double b = (2048.0 - view.anchorRel[7]) * view.texel;
  std::array<double, 3> origin{};
  for (uint32_t i = 0; i < 3; ++i) {
    const double anchor = double{sc.views.header.anchorHi[i]} + double{sc.views.header.anchorLo[i]};
    origin[i] = anchor + a * basis[0][i] + b * basis[1][i];
  }
  Record rec = cluster_record(21, origin, 1.f, 1000, 1024);
  for (uint32_t r = 0; r < 3; ++r) {
    for (uint32_t c = 0; c < 3; ++c) {
      rec[record::World + r * 4 + c] = std::bit_cast<uint32_t>(static_cast<float>(basis[c][r]));
    }
  }
  const float t = view.texel;
  for (uint32_t c = 0; c < 3; ++c) {
    rec[record::CurAabb + c] = ordered(0.f);
    rec[record::CurAabb + 3 + c] = ordered(c == 2 ? t : 128.f * t);
  }
  sc.add(1, rec);
  constexpr std::array<float, 4> Fractions{0.15f, 0.35f, 0.6f, 0.65f};
  std::vector<uint32_t> want;
  for (uint32_t j = 0; j < 32; ++j) {
    for (uint32_t i = 0; i < 32; ++i) {
      const float fx = Fractions[(i * 3 + j) % 4];
      const float fy = Fractions[(j * 5 + i) % 4];
      const float x = 4.f * static_cast<float>(i) + fx;
      const float y = 4.f * static_cast<float>(j) + fy;
      const uint32_t index = 1000 + j * 32 + i;
      sc.cluster(index, {.lo = {x * t, y * t, 0.f}, .hi = {(x + 0.4f) * t, (y + 0.4f) * t, t}}, false);
      if (fx < 0.5f && fy < 0.5f) {
        want.push_back(index);
      }
    }
  }
  const uint32_t page = page_index(view.table_offset(), view.origin[0] + 16, view.origin[1] + 16);
  sc.requests[page] = request_bit::Pixel;
  sc.allocate_pages();
  ASSERT_EQ(sc.renderList[render_list::Count], 1u);
  const CullOut out = run_cull(sc);
  std::vector<uint32_t> got;
  for (uint32_t i = 0; i < out.pairs[pairs_header::OpaqueCount]; ++i) {
    const Pair p = pair_unpack({out.pairs[pairs_header::Words + 2 * i], out.pairs[pairs_header::Words + 2 * i + 1]});
    EXPECT_EQ(p.renderIndex, 0u);
    got.push_back(p.item);
  }
  std::ranges::sort(got);
  EXPECT_EQ(got, want);
  EXPECT_EQ(want.size(), 256u);
}

TEST_F(RenderVsmCullGpu, ChunkBoundsMatchCpu) {
  CullScene sc{.views = cull_views()};
  constexpr uint32_t VertexFirst = 40;
  constexpr uint32_t Vertices = 300;
  std::mt19937 rng{3};
  std::uniform_real_distribution<float> coord(-70.f, 70.f);
  std::vector<uint32_t> vertexWords(size_t{1024} * scene::streamout::vertex::Words, 0);
  std::vector<std::array<float, 3>> local(Vertices);
  for (uint32_t v = 0; v < Vertices; ++v) {
    local[v] = {coord(rng), coord(rng) * 0.5f, coord(rng)};
    for (uint32_t k = 0; k < 3; ++k) {
      vertexWords[size_t{VertexFirst + v} * scene::streamout::vertex::Words + k] = std::bit_cast<uint32_t>(local[v][k]);
    }
  }
  std::vector<uint32_t> indexWords(2048, 0);
  std::uniform_int_distribution<uint32_t> pick(0, Vertices - 1);
  for (uint32_t i = 0; i < 900; ++i) {
    indexWords[128 + i] = pick(rng);
    indexWords[1130 + i % 300] = pick(rng);
  }
  const double yaw = 0.7;
  Record rec = cluster_record(31, offset(100.0, 0.0, 50.0), 70.f, 0, 0, AURORA_CAT_DYNAMIC);
  rec[record::World + 0] = std::bit_cast<uint32_t>(static_cast<float>(std::cos(yaw)));
  rec[record::World + 2] = std::bit_cast<uint32_t>(static_cast<float>(std::sin(yaw)));
  rec[record::World + 8] = std::bit_cast<uint32_t>(static_cast<float>(-std::sin(yaw)));
  rec[record::World + 10] = std::bit_cast<uint32_t>(static_cast<float>(std::cos(yaw)));
  rec[record::IndexFirst] = 128;
  rec[record::IndexCount] = 900;
  rec[record::VertexFirst] = VertexFirst;
  sc.add(1, rec);
  Record misaligned = rec;
  misaligned[record::IndexFirst] = 1130;
  misaligned[record::IndexCount] = 300;
  sc.add(2, misaligned);
  sc.add(3, cluster_record(32, offset(0.0, 0.0, 0.0), 10.f, 100, 1));
  ASSERT_NE(sc.hist(1)[hist::Flags] & hist::Chunked, 0u);
  EXPECT_EQ(sc.hist(1)[hist::VertexFirst], VertexFirst);
  const wgpu::Buffer indices = buffer(indexWords.size() * 4);
  const wgpu::Buffer vertices = buffer(vertexWords.size() * 4);
  upload(indices, indexWords);
  upload(vertices, vertexWords);
  upload(m_history, sc.history);
  upload(m_chunkBoxes, sc.chunkBoxes);
  upload(m_stats, std::vector<uint32_t>(StatCount, 0));
  const std::vector<uint32_t> list{1, 2, 3};
  upload(m_cullList, list);
  const std::array chunkBuffers{m_cullList, m_history, indices, vertices, m_chunkBoxes, m_stats};
  dispatch(m_chunkBounds, m_empty, group(m_chunkGroup, chunkBuffers), 3);
  const std::vector<uint32_t> boxes = read(m_chunkBoxes, ChunkBoxesBytes);
  const std::vector<uint32_t> stats = read(m_stats, StatsBytes);
  EXPECT_EQ(stats[stat::ChunkBoxes], 3u);
  EXPECT_EQ(stats[stat::ChunkMisaligned], 1u);
  const std::span<const uint32_t, 12> xform = sc.hist(1).subspan<hist::Xform, 12>();
  const Box bounds = history_box(sc.hist(1));
  for (uint32_t c = 0; c < 3; ++c) {
    Box cpu{.lo = {3.4e38f, 3.4e38f, 3.4e38f}, .hi = {-3.4e38f, -3.4e38f, -3.4e38f}};
    std::vector<std::array<float, 3>> points;
    for (uint32_t i = c * ChunkIndices; i < std::min((c + 1) * ChunkIndices, 900u); ++i) {
      const std::array<float, 3>& p = local[indexWords[128 + i]];
      const Box one = box_through({.lo = p, .hi = p}, xform);
      points.push_back(one.lo);
      for (uint32_t k = 0; k < 3; ++k) {
        cpu.lo[k] = std::min(cpu.lo[k], one.lo[k]);
        cpu.hi[k] = std::max(cpu.hi[k], one.hi[k]);
      }
    }
    const uint32_t slot = chunk_slot(128, c);
    const std::array<uint32_t, 2> gpuWords{boxes[size_t{slot} * 2], boxes[size_t{slot} * 2 + 1]};
    const std::array<uint32_t, 2> cpuWords = chunk_box_pack(cpu, bounds);
    for (uint32_t w = 0; w < 2; ++w) {
      for (uint32_t k = 0; k < 3; ++k) {
        const auto g = static_cast<int32_t>((gpuWords[w] >> (10 * k)) & 1023u);
        const auto e = static_cast<int32_t>((cpuWords[w] >> (10 * k)) & 1023u);
        EXPECT_LE(std::abs(g - e), 1) << "chunk " << c << " word " << w << " axis " << k;
      }
    }
    const Box decoded = chunk_box_unpack(gpuWords, bounds);
    for (const std::array<float, 3>& p : points) {
      for (uint32_t k = 0; k < 3; ++k) {
        EXPECT_GE(p[k], decoded.lo[k] - 1e-3f) << c;
        EXPECT_LE(p[k], decoded.hi[k] + 1e-3f) << c;
      }
    }
  }
  for (uint32_t s = 0; s < ChunkBoxCount * 2; ++s) {
    const bool written = s >= chunk_slot(128, 0) * 2 && s < chunk_slot(128, 3) * 2;
    if (!written) {
      EXPECT_EQ(boxes[s], 0u) << s;
    }
  }
}

TEST_F(RenderVsmCullGpu, PairOverflowSkipsWholePages) {
  CullScene sc = mixed_scene();
  constexpr uint32_t Max = 16;
  std::vector<uint32_t> firstTable;
  for (int run = 0; run < 3; ++run) {
    const CullOut out = run_cull(sc, Max);
    const uint32_t first = skip_first(out.counts, Max);
    ASSERT_LT(first, MaxRenderPages);
    EXPECT_EQ(out.pairs[pairs_header::SkipKey], MaxRenderPages - first);
    EXPECT_LE(out.pairs[pairs_header::OpaqueCount] + out.pairs[pairs_header::CutoutCount], Max);
    EXPECT_EQ(out.pairs[pairs_header::Dropped], 0u);
    uint32_t skipped = 0;
    for (uint32_t r = 0; r < sc.renderList[render_list::Count]; ++r) {
      const uint32_t v = sc.renderList[render_list::Pages + 2 * r];
      const bool skip = r >= first && out.counts[r] + out.counts[MaxRenderPages + r] != 0;
      const uint32_t e = out.table[v];
      EXPECT_EQ((e & entry::Skip) != 0, skip) << r;
      if (skip) {
        EXPECT_EQ(e & (entry::Dirty | entry::Priority), entry::Dirty | entry::Priority) << r;
        ++skipped;
      }
    }
    EXPECT_GT(skipped, 0u);
    EXPECT_EQ(out.pairs[pairs_header::Skipped], skipped);
    for (uint32_t i = 0; i < out.pairs[pairs_header::OpaqueCount] + out.pairs[pairs_header::CutoutCount]; ++i) {
      EXPECT_LT(
          pair_unpack({out.pairs[pairs_header::Words + 2 * i], out.pairs[pairs_header::Words + 2 * i + 1]}).renderIndex,
          first);
    }
    if (run == 0) {
      firstTable = out.table;
    } else {
      EXPECT_EQ(out.table, firstTable) << run;
    }
  }
  std::vector<uint32_t> table = firstTable;
  (void)commit(sc.renderList, table);
  std::vector<uint32_t> skippedPages;
  for (uint32_t v = 0; v < MaxVirtualPages; ++v) {
    if ((table[v] & (entry::Allocated | entry::Priority)) == (entry::Allocated | entry::Priority)) {
      EXPECT_EQ(table[v] & (entry::Cached | entry::Skip | entry::RenderNow), 0u) << v;
      skippedPages.push_back(v);
    }
  }
  ASSERT_FALSE(skippedPages.empty());
  Views next = sc.views;
  next.header.clock = 3;
  next.header.budget = 8;
  std::vector<uint32_t> list(render_list::Words, 0);
  const AllocCounts counts = allocate(next, sc.requests, table, sc.phys, list);
  EXPECT_EQ(counts.priority, skippedPages.size());
  for (uint32_t i = 0; i < skippedPages.size(); ++i) {
    EXPECT_EQ(list[render_list::Pages + 2 * i], skippedPages[i]) << i;
  }
}

constexpr uint64_t CompareLayoutKey = 0x76736D636D700001ull;
constexpr uint32_t AtlasPagesX = 8;
constexpr uint32_t AtlasRows = 8;
constexpr uint32_t AtlasWidth = AtlasPagesX * PageTexels;
constexpr uint32_t AtlasHeight = AtlasRows * PageTexels;
constexpr uint32_t PayloadBase = TestClusters * 16;
constexpr int32_t TestPage = 16;
constexpr uint32_t CutoutMaterial = 5;

struct Vtx {
  std::array<float, 3> pos{};
  std::array<float, 2> uv{};
};

using Tri = std::array<uint32_t, 3>;

struct RasterScene {
  CullScene cull;
  std::vector<uint32_t> records = std::vector<uint32_t>(size_t{TestSlots} * RecordWords, 0);
  std::vector<uint32_t> vertices;
  std::vector<uint32_t> indices;
  std::vector<uint32_t> payload;

  explicit RasterScene(const Views& views) : cull{.views = views} {
    cull.views.header.poolPages = AtlasPagesX * AtlasRows;
    cull.views.header.poolPagesX = AtlasPagesX;
  }

  [[nodiscard]] std::vector<uint32_t> cluster_words() const {
    std::vector<uint32_t> out = cull.clusters;
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
  }

  [[nodiscard]] const ViewRecord& view(uint32_t level = 0) const { return cull.views.views[level_view(level)]; }

  [[nodiscard]] std::array<double, 3> anchor() const {
    const ViewsHeader& h = cull.views.header;
    return {double{h.anchorHi[0]} + h.anchorLo[0], double{h.anchorHi[1]} + h.anchorLo[1],
            double{h.anchorHi[2]} + h.anchorLo[2]};
  }

  // Inverts the level's anchor-space rows, which are orthogonal, so a test can author casters in texel space
  [[nodiscard]] std::array<float, 3> at(double x, double y, double z, uint32_t level = 0) const {
    const ViewRecord& v = view(level);
    const std::array<double, 3> c{x, y, z};
    std::array<double, 3> p{};
    for (uint32_t r = 0; r < 3; ++r) {
      const float* row = v.anchorRel + r * 4;
      const double norm = double{row[0]} * row[0] + double{row[1]} * row[1] + double{row[2]} * row[2];
      for (uint32_t k = 0; k < 3; ++k) {
        p[k] += row[k] * (c[r] - row[3]) / norm;
      }
    }
    return {static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
  }

  [[nodiscard]] uint32_t page_index_of(int32_t px, int32_t py, uint32_t level = 0) const {
    const ViewRecord& v = view(level);
    return page_index(v.table_offset(), v.origin[0] + px, v.origin[1] + py);
  }

  Record base_record(uint32_t hash) const {
    Record rec{};
    rec[record::Flags] = instance_flag::Valid;
    rec[record::Categories] = AURORA_CAT_STATIC;
    rec[record::IdentityHash] = hash;
    rec[record::Material] = scene::streamout::NoMaterial | 0xFF0000u;
    rec[record::WorldFrame] = 1;
    rec[record::UvFrame] = 1;
    rec[record::MaterialFrame] = 1;
    rec[record::StreamFrame] = 1;
    place(std::span{rec}, anchor());
    for (const uint32_t base : {record::UvBase, record::UvAlpha}) {
      rec[base + 0] = std::bit_cast<uint32_t>(1.f);
      rec[base + 4] = std::bit_cast<uint32_t>(1.f);
    }
    return rec;
  }

  static Box box_of(const std::vector<Vtx>& verts) {
    Box box{.lo = {3.4e38f, 3.4e38f, 3.4e38f}, .hi = {-3.4e38f, -3.4e38f, -3.4e38f}};
    for (const Vtx& v : verts) {
      for (uint32_t k = 0; k < 3; ++k) {
        box.lo[k] = std::min(box.lo[k], v.pos[k]);
        box.hi[k] = std::max(box.hi[k], v.pos[k]);
      }
    }
    return box;
  }

  void add_cluster(uint32_t index, uint32_t firstVertex, const std::vector<Vtx>& verts, const std::vector<Tri>& tris) {
    ASSERT_LE(verts.size(), 64u);
    ASSERT_LE(tris.size(), size_t{ChunkTriangles});
    ASSERT_LT(index, TestClusters);
    uint32_t* c = &cull.clusters[size_t{index} * 16];
    const uint32_t at = index * 16;
    c[0] = PayloadBase + static_cast<uint32_t>(payload.size()) - at;
    for (uint32_t i = 0; i < verts.size(); ++i) {
      payload.push_back(firstVertex + i);
    }
    c[1] = PayloadBase + static_cast<uint32_t>(payload.size()) - at;
    for (const Tri& t : tris) {
      payload.push_back(t[0] | t[1] << 8 | t[2] << 16);
    }
    c[2] = static_cast<uint32_t>(verts.size()) | static_cast<uint32_t>(tris.size()) << 8;
    const Box box = box_of(verts);
    for (uint32_t k = 0; k < 3; ++k) {
      c[8 + k] = std::bit_cast<uint32_t>(box.lo[k]);
      c[12 + k] = std::bit_cast<uint32_t>(box.hi[k]);
    }
  }

  void push_vertices(const std::vector<Vtx>& verts) {
    for (const Vtx& v : verts) {
      for (uint32_t k = 0; k < 3; ++k) {
        vertices.push_back(std::bit_cast<uint32_t>(v.pos[k]));
      }
      vertices.push_back(0);
      vertices.push_back(std::bit_cast<uint32_t>(v.uv[0]));
      vertices.push_back(std::bit_cast<uint32_t>(v.uv[1]));
      vertices.push_back(0);
      vertices.push_back(0xFFu);
    }
  }

  void finish(uint32_t slot, Record& rec, const Box& box) {
    for (uint32_t k = 0; k < 3; ++k) {
      rec[record::CurAabb + k] = ordered(box.lo[k]);
      rec[record::CurAabb + 3 + k] = ordered(box.hi[k]);
    }
    cull.add(slot, rec);
    std::ranges::copy(rec, records.begin() + size_t{slot} * RecordWords);
  }

  void add_mesh(uint32_t slot, Record rec, const std::vector<Vtx>& verts, const std::vector<Tri>& tris, bool clustered,
                uint32_t clusterIndex = 0) {
    const auto base = static_cast<uint32_t>(vertices.size() / scene::streamout::vertex::Words);
    rec[record::VertexFirst] = base;
    rec[record::VertexCount] = static_cast<uint32_t>(verts.size());
    push_vertices(verts);
    const Box box = box_of(verts);
    if (clustered) {
      add_cluster(clusterIndex, 0, verts, tris);
      rec[record::ClusterFirst] = clusterIndex;
      rec[record::ClusterCount] = 1;
      finish(slot, rec, box);
      return;
    }
    {
      indices.resize((indices.size() + IndexGranule - 1) / IndexGranule * IndexGranule, 0);
      rec[record::IndexFirst] = static_cast<uint32_t>(indices.size());
      rec[record::IndexCount] = static_cast<uint32_t>(tris.size() * 3);
      for (const Tri& t : tris) {
        indices.insert(indices.end(), t.begin(), t.end());
      }
      rec[record::ClusterFirst] = 0;
      rec[record::ClusterCount] = 0;
    }
    finish(slot, rec, box);
    {
      const Box bounds = history_box(cull.hist(slot));
      for (uint32_t ch = 0; ch < chunk_count(rec[record::IndexCount]); ++ch) {
        Box cb{.lo = {3.4e38f, 3.4e38f, 3.4e38f}, .hi = {-3.4e38f, -3.4e38f, -3.4e38f}};
        for (uint32_t t = ch * ChunkTriangles; t < std::min<uint32_t>((ch + 1) * ChunkTriangles, tris.size()); ++t) {
          for (const uint32_t i : tris[t]) {
            for (uint32_t k = 0; k < 3; ++k) {
              cb.lo[k] = std::min(cb.lo[k], verts[i].pos[k]);
              cb.hi[k] = std::max(cb.hi[k], verts[i].pos[k]);
            }
          }
        }
        const std::array<uint32_t, 2> words = chunk_box_pack(cb, bounds);
        const uint32_t cs = chunk_slot(rec[record::IndexFirst], ch);
        cull.chunkBoxes[size_t{cs} * 2] = words[0];
        cull.chunkBoxes[size_t{cs} * 2 + 1] = words[1];
      }
    }
  }

  // A quad over window texels [x0, x1) x [y0, y1) of level 0, its depth linear from z0 at x0 to z1 at x1
  void add_quad(uint32_t slot, Record rec, double x0, double y0, double x1, double y1, double z0, double z1,
                bool clustered, uint32_t clusterIndex) {
    const std::vector<Vtx> verts{
        {.pos = at(x0, y0, z0), .uv = {0.f, 0.f}},
        {.pos = at(x1, y0, z1), .uv = {1.f, 0.f}},
        {.pos = at(x0, y1, z0), .uv = {0.f, 1.f}},
        {.pos = at(x1, y1, z1), .uv = {1.f, 1.f}},
    };
    add_mesh(slot, rec, verts, {{0, 1, 2}, {2, 1, 3}}, clustered, clusterIndex);
  }

  void list(const std::vector<std::pair<uint32_t, uint32_t>>& pages, bool check = false) {
    std::ranges::fill(cull.renderList, 0u);
    std::ranges::fill(cull.table, 0u);
    const auto n = static_cast<uint32_t>(pages.size());
    cull.renderList[render_list::Count] = check ? 0 : n;
    cull.renderList[render_list::CheckCount] = check ? n : 0;
    cull.renderList[render_list::PagesX] = AtlasPagesX;
    cull.renderList[render_list::Rows] = AtlasRows;
    for (uint32_t i = 0; i < n; ++i) {
      const auto [v, phys] = pages[i];
      cull.renderList[render_list::Pages + 2 * i] = v | (check ? render_list::CheckOnly : 0u);
      cull.renderList[render_list::Pages + 2 * i + 1] = phys;
      const ViewRecord& record = cull.views.views[v / PagesPerView];
      const uint32_t s = v % PagesPerView;
      const uint32_t lx = (s % WindowPages - static_cast<uint32_t>(record.origin[0])) & (WindowPages - 1);
      const uint32_t ly = (s / WindowPages - static_cast<uint32_t>(record.origin[1])) & (WindowPages - 1);
      cull.renderList[render_list::RowMasks + (v / PagesPerView) * WindowPages + ly] |= 1u << lx;
      cull.table[v] = phys | entry::Allocated | i << entry::RenderIndexShift |
                      (check ? entry::Cached | entry::Check : entry::RenderNow | entry::Fresh);
    }
  }
};

Views raster_views() {
  SunClipmap clipmap;
  (void)clipmap.update({.eye = TestEye, .dir = sun_dir(40.0, 30.0), .lightId = 2, .texel0 = TestTexel0});
  Views views{};
  clipmap.fill(views);
  views.header.clock = 2;
  views.header.budget = MaxPoolPages;
  return views;
}

struct Variant {
  bool cutout = false;
  bool clip = false;
  bool unclipped = false;
};

class RenderVsmRasterGpu : public RenderVsmCullGpu {
protected:
  void SetUp() override {
    RenderVsmCullGpu::SetUp();
    if (IsSkipped()) {
      return;
    }
    const wgpu::Device& device = m_device->device;
    m_hasClip = device.HasFeature(wgpu::FeatureName::ClipDistances);
    m_hasUnclipped = device.HasFeature(wgpu::FeatureName::DepthClipControl);
    std::array<wgpu::BindGroupLayoutEntry, 2> clear{};
    for (uint32_t i = 0; i < clear.size(); ++i) {
      clear[i] = {.binding = i,
                  .visibility = wgpu::ShaderStage::Vertex,
                  .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}};
    }
    const wgpu::BindGroupLayoutDescriptor clearDescriptor{.entryCount = clear.size(), .entries = clear.data()};
    m_clearGroup = device.CreateBindGroupLayout(&clearDescriptor);
    const wgpu::PipelineLayoutDescriptor clearLayout{.bindGroupLayoutCount = 1, .bindGroupLayouts = &m_clearGroup};
    static constexpr std::array<std::string_view, 1> Clear{"VSM_CLEAR"};
    const shaders::PipelineId clearId = shaders::request_render({
        .label = "vsm.clear raster test",
        .shader = AllocShader,
        .vertexEntry = "vs_clear",
        .fragmentEntry = {},
        .defines = Clear,
        .layout = device.CreatePipelineLayout(&clearLayout),
        .layoutKey = ClearTestLayoutKey,
        .depthFormat = wgpu::TextureFormat::Depth32Float,
        .depthWrite = true,
        .depthCompare = wgpu::CompareFunction::Always,
    });
    ASSERT_EQ(shaders::wait(clearId), shaders::PipelineStatus::Ready);
    m_clear = shaders::render_pipeline(clearId);
    std::vector<wgpu::BindGroupLayoutEntry> compare(7);
    for (uint32_t i = 0; i < compare.size(); ++i) {
      compare[i] = {.binding = i, .visibility = wgpu::ShaderStage::Compute};
      if (i < 2) {
        compare[i].texture = {.sampleType = wgpu::TextureSampleType::Depth,
                              .viewDimension = wgpu::TextureViewDimension::e2D};
      } else {
        compare[i].buffer = {.type = i == 4 || i == 6 ? wgpu::BufferBindingType::Storage
                                                      : wgpu::BufferBindingType::ReadOnlyStorage};
      }
    }
    const wgpu::BindGroupLayoutDescriptor compareDescriptor{.entryCount = compare.size(), .entries = compare.data()};
    m_compareGroup = device.CreateBindGroupLayout(&compareDescriptor);
    m_compare = pipeline(m_emptyLayout, m_compareGroup, check::CheckShader, "vsm_check_cmp", CompareLayoutKey, {});
    const auto make = [&](uint64_t size, wgpu::BufferUsage usage) {
      const wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst, .size = size};
      return device.CreateBuffer(&descriptor);
    };
    m_uniformViews = make(ViewsBytes, wgpu::BufferUsage::Uniform);
    m_indirect = make(RasterArgsBytes, wgpu::BufferUsage::Indirect);
    m_clearIndirect = make(16, wgpu::BufferUsage::Indirect);
    m_instancesBuffer = buffer(uint64_t{TestSlots} * RecordWords * 4);
    m_materialsBuffer = buffer(MaterialBytes);
    m_colorAtlas = texture(wgpu::TextureFormat::RGBA8Unorm, 1, 1,
                           wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst);
  }

  void TearDown() override {
    raster_shutdown();
    RenderVsmCullGpu::TearDown();
  }

  wgpu::RenderPipeline raster_pipeline(const Variant& v) const {
    shaders::RenderPipelineDesc desc{
        .label = "vsm.raster test",
        .shader = RasterShader,
        .layout = testing::raster_layout(),
        .layoutKey = RasterLayoutKey,
    };
    caster_variant(desc, v.cutout, v.clip, v.unclipped);
    const shaders::PipelineId id = shaders::request_render(desc);
    EXPECT_EQ(shaders::wait(id), shaders::PipelineStatus::Ready) << v.cutout << v.clip << v.unclipped;
    return shaders::render_pipeline(id);
  }

  wgpu::Texture depth_atlas() const {
    return texture(wgpu::TextureFormat::Depth32Float, AtlasWidth, AtlasHeight,
                   wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst |
                       wgpu::TextureUsage::TextureBinding);
  }

  void fill(const wgpu::Texture& atlas, float value) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = atlas.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = value,
    };
    const wgpu::RenderPassDescriptor descriptor{.depthStencilAttachment = &depth};
    encoder.BeginRenderPass(&descriptor).End();
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  }

  std::vector<float> read_atlas(const wgpu::Texture& atlas) const {
    const wgpu::Buffer readback = buffer(uint64_t{AtlasWidth} * AtlasHeight * 4);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = atlas, .aspect = wgpu::TextureAspect::DepthOnly};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = AtlasWidth * 4, .rowsPerImage = AtlasHeight},
                                                .buffer = readback};
    const wgpu::Extent3D extent{AtlasWidth, AtlasHeight, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &extent);
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
    const std::vector<uint32_t> words = read(readback, readback.GetSize());
    std::vector<float> out(words.size());
    std::ranges::transform(words, out.begin(), [](uint32_t w) { return std::bit_cast<float>(w); });
    return out;
  }

  // Culls, clears the listed pages and rasterises them into the atlas
  CullOut draw(RasterScene& rs, const wgpu::Texture& atlas, const Variant& opaque, bool withCutout = false) {
    const CullOut out = run_cull(rs.cull);
    const wgpu::Device& device = m_device->device;
    const auto upload_or = [&](const wgpu::Buffer& target, std::span<const uint32_t> words) {
      if (!words.empty()) {
        m_device->queue.WriteBuffer(target, 0, words.data(), words.size_bytes());
      }
    };
    std::vector<uint32_t> vertices = rs.vertices;
    vertices.resize(std::max<size_t>(vertices.size(), 8), 0);
    std::vector<uint32_t> indices = rs.indices;
    indices.resize(std::max<size_t>(indices.size(), 4), 0);
    const wgpu::Buffer vertexBuffer = buffer(vertices.size() * 4);
    const wgpu::Buffer indexBuffer = buffer(indices.size() * 4);
    upload_or(vertexBuffer, vertices);
    upload_or(indexBuffer, indices);
    upload_or(m_instancesBuffer, rs.records);
    upload_or(m_materialsBuffer, rs.cull.materials);
    const std::vector<uint32_t> clusterWords = rs.cluster_words();
    const wgpu::Buffer clusters = buffer(clusterWords.size() * 4);
    upload_or(clusters, clusterWords);
    Views views = rs.cull.views;
    views.header.maxPairs = MaxPairs;
    m_device->queue.WriteBuffer(m_uniformViews, 0, &views, ViewsBytes);
    m_device->queue.WriteBuffer(m_indirect, 0, out.args.data(), RasterArgsBytes);
    const std::array<uint32_t, 4> clearArgs{6, render_entries(rs.cull.renderList), 0, 0};
    m_device->queue.WriteBuffer(m_clearIndirect, 0, clearArgs.data(), sizeof(clearArgs));
    const std::array clearBuffers{m_renderList, m_table};
    const wgpu::BindGroup clearBind = group(m_clearGroup, clearBuffers);
    std::array<wgpu::BindGroupEntry, raster_binding::Count> entries{};
    const std::array<wgpu::Buffer, raster_binding::Buffers> buffers{vertexBuffer,   indexBuffer,      m_instancesBuffer,
                                                                    clusters,       m_renderList,     m_pairs,
                                                                    m_uniformViews, m_materialsBuffer};
    for (uint32_t i = 0; i < buffers.size(); ++i) {
      entries[i] = {.binding = i, .buffer = buffers[i], .size = buffers[i].GetSize()};
    }
    entries[raster_binding::Atlas] = {.binding = raster_binding::Atlas, .textureView = m_colorAtlas.CreateView()};
    const wgpu::SamplerDescriptor samplerDescriptor{.magFilter = wgpu::FilterMode::Linear,
                                                    .minFilter = wgpu::FilterMode::Linear};
    entries[raster_binding::Sampler] = {.binding = raster_binding::Sampler,
                                        .sampler = device.CreateSampler(&samplerDescriptor)};
    const wgpu::BindGroupDescriptor descriptor{
        .layout = testing::raster_group_layout(), .entryCount = entries.size(), .entries = entries.data()};
    const wgpu::BindGroup rasterBind = device.CreateBindGroup(&descriptor);
    const wgpu::RenderPipeline opaquePipe = raster_pipeline(opaque);
    const wgpu::RenderPipeline cutoutPipe =
        raster_pipeline({.cutout = true, .clip = opaque.clip, .unclipped = opaque.unclipped});
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::RenderPassDepthStencilAttachment depth{
        .view = atlas.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Load,
        .depthStoreOp = wgpu::StoreOp::Store,
    };
    const wgpu::RenderPassDescriptor passDescriptor{.depthStencilAttachment = &depth};
    {
      const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
      pass.SetPipeline(m_clear);
      pass.SetBindGroup(0, clearBind);
      pass.DrawIndirect(m_clearIndirect, 0);
      pass.End();
    }
    {
      const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
      pass.SetBindGroup(0, rasterBind);
      pass.SetPipeline(opaquePipe);
      pass.DrawIndirect(m_indirect, 0);
      if (withCutout) {
        pass.SetPipeline(cutoutPipe);
        pass.DrawIndirect(m_indirect, 16);
      }
      pass.End();
    }
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
    return out;
  }

  std::vector<uint32_t> compare(const RasterScene& rs, const wgpu::Texture& atlas, const wgpu::Texture& copy) {
    const std::vector<uint32_t> zero(StatCount, 0);
    upload(m_stats, zero);
    const wgpu::Buffer probes = buffer(64);
    std::vector<wgpu::BindGroupEntry> entries{
        {.binding = 0, .textureView = atlas.CreateView()},
        {.binding = 1, .textureView = copy.CreateView()},
        {.binding = 2, .buffer = m_renderList, .size = m_renderList.GetSize()},
        {.binding = 3, .buffer = m_table, .size = m_table.GetSize()},
        {.binding = 4, .buffer = m_stats, .size = m_stats.GetSize()},
        {.binding = 5, .buffer = m_views, .size = m_views.GetSize()},
        {.binding = 6, .buffer = probes, .size = probes.GetSize()},
    };
    (void)rs;
    const wgpu::BindGroupDescriptor descriptor{
        .layout = m_compareGroup, .entryCount = entries.size(), .entries = entries.data()};
    dispatch(m_compare, m_empty, m_device->device.CreateBindGroup(&descriptor), MaxRenderPages);
    return read(m_stats, StatsBytes);
  }

  void copy_texture(const wgpu::Texture& from, const wgpu::Texture& to) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = from};
    const wgpu::TexelCopyTextureInfo destination{.texture = to};
    const wgpu::Extent3D extent{AtlasWidth, AtlasHeight, 1};
    encoder.CopyTextureToTexture(&source, &destination, &extent);
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
  }

  [[nodiscard]] std::vector<Variant> opaque_variants() const {
    std::vector<Variant> out;
    for (const bool unclipped : {false, true}) {
      for (const bool clip : {false, true}) {
        if ((!clip || m_hasClip) && (!unclipped || m_hasUnclipped)) {
          out.push_back({.clip = clip, .unclipped = unclipped});
        }
      }
    }
    return out;
  }

  static float texel(const std::vector<float>& atlas, uint32_t phys, uint32_t x, uint32_t y) {
    const uint32_t ax = (phys % AtlasPagesX) * PageTexels + x;
    const uint32_t ay = (phys / AtlasPagesX) * PageTexels + y;
    return atlas[size_t{ay} * AtlasWidth + ax];
  }

  // The level-0 depth of a plane through the quad corners, in double at a texel centre
  static double plane_depth(double x0, double x1, double z0, double z1, double x) {
    return z0 + (z1 - z0) * (x - x0) / (x1 - x0);
  }

  wgpu::BindGroupLayout m_clearGroup, m_compareGroup;
  wgpu::RenderPipeline m_clear;
  wgpu::ComputePipeline m_compare;
  wgpu::Buffer m_uniformViews, m_indirect, m_clearIndirect, m_instancesBuffer, m_materialsBuffer;
  wgpu::Texture m_colorAtlas;
  bool m_hasClip = false;
  bool m_hasUnclipped = false;
};

int64_t ulp_distance(float a, float b) {
  const auto ia = static_cast<int64_t>(std::bit_cast<int32_t>(a));
  const auto ib = static_cast<int64_t>(std::bit_cast<int32_t>(b));
  return std::abs(ia - ib);
}

TEST_F(RenderVsmRasterGpu, RasterStaysInsideListedPages) {
  const auto x0 = static_cast<double>((TestPage - 1) * PageTexels) + 7.3;
  const auto x1 = static_cast<double>((TestPage + 2) * PageTexels) - 5.1;
  for (const Variant& variant : opaque_variants()) {
    RasterScene rs{raster_views()};
    rs.add_quad(1, rs.base_record(41), x0, x0, x1, x1, 0.62, 0.64, true, 100);
    constexpr uint32_t Inside = 9;
    constexpr uint32_t Far = 10;
    rs.list({{rs.page_index_of(TestPage, TestPage), Inside}, {rs.page_index_of(TestPage + 6, TestPage + 6), Far}});
    const wgpu::Texture atlas = depth_atlas();
    constexpr float Sentinel = 0.25f;
    fill(atlas, Sentinel);
    const CullOut out = draw(rs, atlas, variant);
    EXPECT_GT(out.pairs[pairs_header::OpaqueCount], 0u);
    const std::vector<float> texels = read_atlas(atlas);
    uint32_t leaks = 0;
    uint32_t holes = 0;
    for (uint32_t p = 0; p < AtlasPagesX * AtlasRows; ++p) {
      for (uint32_t y = 0; y < PageTexels; ++y) {
        for (uint32_t x = 0; x < PageTexels; ++x) {
          const float d = texel(texels, p, x, y);
          if (p == Inside) {
            holes += d > 0.6f && d < 0.65f ? 0 : 1;
          } else {
            leaks += d == (p == Far ? 0.f : Sentinel) ? 0 : 1;
          }
        }
      }
    }
    EXPECT_EQ(holes, 0u) << "clip " << variant.clip << " unclipped " << variant.unclipped;
    EXPECT_EQ(leaks, 0u) << "clip " << variant.clip << " unclipped " << variant.unclipped;
  }
}

TEST_F(RenderVsmRasterGpu, ClipAndDiscardAgree) {
  if (!m_hasClip) {
    GTEST_SKIP() << "the device has no clip distances";
  }
  std::mt19937 rng{7};
  std::uniform_real_distribution<double> coord(static_cast<double>((TestPage - 1) * PageTexels),
                                               static_cast<double>((TestPage + 2) * PageTexels));
  std::uniform_real_distribution<double> depth(0.499, 0.501);
  RasterScene rs{raster_views()};
  std::vector<Vtx> verts;
  std::vector<Tri> tris;
  for (uint32_t t = 0; t < 20; ++t) {
    for (uint32_t k = 0; k < 3; ++k) {
      verts.push_back({.pos = rs.at(coord(rng), coord(rng), depth(rng))});
    }
    tris.push_back({3 * t, 3 * t + 1, 3 * t + 2});
  }
  rs.add_mesh(1, rs.base_record(42), verts, tris, true, 100);
  rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
  std::array<std::vector<float>, 2> results;
  for (const bool clip : {false, true}) {
    const wgpu::Texture atlas = depth_atlas();
    fill(atlas, 0.f);
    (void)draw(rs, atlas, {.clip = clip});
    results[clip ? 1 : 0] = read_atlas(atlas);
  }
  uint32_t covered = 0;
  uint32_t bitwise = 0;
  int64_t worst = 0;
  for (size_t i = 0; i < results[0].size(); ++i) {
    const float a = results[0][i];
    const float b = results[1][i];
    ASSERT_EQ(a > 0.f, b > 0.f) << "coverage differs at texel " << i;
    if (a > 0.f) {
      ++covered;
      bitwise += a == b ? 1 : 0;
      worst = std::max(worst, ulp_distance(a, b));
    }
  }
  EXPECT_GT(covered, 1000u);
  EXPECT_LE(worst, 4);
  std::printf("clip and discard: %u covered texels, %.2f%% bitwise equal, worst %lld ulp\n", covered,
              100.0 * bitwise / std::max(covered, 1u), static_cast<long long>(worst));
}

TEST_F(RenderVsmRasterGpu, CutoutCastsHoles) {
  m_colorAtlas = texture(wgpu::TextureFormat::RGBA8Unorm, scene::atlas::PageSize, scene::atlas::PageSize,
                         wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst);
  constexpr uint32_t Tile = scene::atlas::TileSize;
  std::vector<uint32_t> tile(size_t{Tile} * Tile, 0);
  for (uint32_t y = 0; y < Tile; ++y) {
    for (uint32_t x = 0; x < Tile; ++x) {
      const bool open = ((x / 16) + (y / 16)) % 2 != 0;
      tile[size_t{y} * Tile + x] = open ? 0x00FFFFFFu : 0xFFFFFFFFu;
    }
  }
  const wgpu::TexelCopyTextureInfo destination{.texture = m_colorAtlas};
  const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = Tile * 4, .rowsPerImage = Tile};
  const wgpu::Extent3D extent{Tile, Tile, 1};
  m_device->queue.WriteTexture(&destination, tile.data(), tile.size() * 4, &layout, &extent);
  RasterScene rs{raster_views()};
  uint32_t* m = &rs.cull.materials[size_t{CutoutMaterial} * 8];
  m[0] = scene::atlas::NoTile | 0u << 16;
  m[1] = 3u << 8 | scene::atlas::material_flag::AlphaTest << 16;
  m[2] = 4u | 7u << 3 | 0u << 6 | 128u << 8;
  Record rec = rs.base_record(43);
  rec[record::Material] = CutoutMaterial | 0xFF0000u;
  rec[record::Flags] |= instance_flag::AlphaUvIsUv0;
  const auto x0 = static_cast<double>(TestPage * PageTexels);
  const double x1 = x0 + PageTexels;
  rs.add_quad(1, rec, x0, x0, x1, x1, 0.55, 0.55, true, 100);
  ASSERT_NE(rs.cull.hist(1)[hist::Flags] & hist::Cutout, 0u);
  rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
  for (const Variant& variant : opaque_variants()) {
    const wgpu::Texture atlas = depth_atlas();
    fill(atlas, 0.f);
    const CullOut out = draw(rs, atlas, variant, true);
    EXPECT_EQ(out.pairs[pairs_header::OpaqueCount], 0u);
    EXPECT_GT(out.pairs[pairs_header::CutoutCount], 0u);
    const std::vector<float> texels = read_atlas(atlas);
    uint32_t wrong = 0;
    uint32_t open = 0;
    for (uint32_t y = 0; y < PageTexels; ++y) {
      for (uint32_t x = 0; x < PageTexels; ++x) {
        const bool hole = ((x / 16) + (y / 16)) % 2 != 0;
        const float d = texel(texels, 9, x, y);
        open += hole ? 1 : 0;
        wrong += (hole ? d == 0.f : std::abs(d - 0.55f) < 1e-4f) ? 0 : 1;
      }
    }
    EXPECT_EQ(open, PageTexels * PageTexels / 2);
    EXPECT_EQ(wrong, 0u) << "clip " << variant.clip << " unclipped " << variant.unclipped;
  }
}

TEST_F(RenderVsmRasterGpu, PancakeKeepsCastersBeyondTheLightSide) {
  const auto x0 = static_cast<double>(TestPage * PageTexels) - 3.0;
  const double x1 = x0 + PageTexels + 6.0;
  for (const Variant& variant : opaque_variants()) {
    RasterScene rs{raster_views()};
    rs.add_quad(1, rs.base_record(44), x0, x0, x1, x1, 1.3, 1.4, true, 100);
    rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
    const wgpu::Texture atlas = depth_atlas();
    fill(atlas, 0.f);
    (void)draw(rs, atlas, variant);
    const std::vector<float> texels = read_atlas(atlas);
    uint32_t wrong = 0;
    for (uint32_t y = 0; y < PageTexels; ++y) {
      for (uint32_t x = 0; x < PageTexels; ++x) {
        wrong += texel(texels, 9, x, y) == 1.f ? 0 : 1;
      }
    }
    EXPECT_EQ(wrong, 0u) << "clip " << variant.clip << " unclipped " << variant.unclipped;
  }
}

TEST_F(RenderVsmRasterGpu, PancakeStraddlingTriangleKeepsPlane) {
  const auto x0 = static_cast<double>(TestPage * PageTexels);
  const double x1 = x0 + PageTexels;
  struct Case {
    double z0;
    double z1;
    bool lightSide;
  };
  for (const Case c : {Case{0.6, 1.4, true}, Case{-0.4, 0.4, false}}) {
    for (const Variant& variant : opaque_variants()) {
      RasterScene rs{raster_views()};
      rs.add_quad(1, rs.base_record(45), x0, x0, x1, x1, c.z0, c.z1, true, 100);
      rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
      const wgpu::Texture atlas = depth_atlas();
      fill(atlas, 0.f);
      (void)draw(rs, atlas, variant);
      const std::vector<float> texels = read_atlas(atlas);
      int64_t worst = 0;
      double residual = 0.0;
      uint32_t outside = 0;
      for (uint32_t y = 0; y < PageTexels; ++y) {
        for (uint32_t x = 2; x + 2 < PageTexels; ++x) {
          const double plane = plane_depth(x0, x1, c.z0, c.z1, x0 + x + 0.5);
          const float d = texel(texels, 9, x, y);
          if (plane > 1.0 + 1e-3) {
            outside += d == 1.f ? 0 : 1;
          } else if (plane < -1e-3) {
            outside += d == 0.f ? 0 : 1;
          } else if (plane > 1e-3 && plane < 1.0 - 1e-3) {
            residual = std::max(residual, std::abs(d - plane));
            if (plane > 0.25) {
              worst = std::max(worst, ulp_distance(d, static_cast<float>(plane)));
            }
          }
        }
      }
      const bool exact = variant.unclipped || !c.lightSide;
      const double slope = std::abs(c.z1 - c.z0) / (x1 - x0);
      if (exact && !variant.clip) {
        EXPECT_LE(worst, 4) << "light side " << c.lightSide << " unclipped " << variant.unclipped;
      }
      if (exact && variant.clip) {
        EXPECT_LE(residual, slope / 64.0 + 2.5e-7)
            << "light side " << c.lightSide << " unclipped " << variant.unclipped;
      }
      if (exact) {
        EXPECT_EQ(outside, 0u) << "light side " << c.lightSide << " unclipped " << variant.unclipped;
      }
      std::printf("straddling the %s bound, clip %d unclipped %d: worst %lld ulp, residual %.3g\n",
                  c.lightSide ? "light" : "far", variant.clip, variant.unclipped, static_cast<long long>(worst),
                  residual);
    }
  }
}

TEST_F(RenderVsmRasterGpu, DegenerateTailDrawsNothing) {
  const auto x0 = static_cast<double>(TestPage * PageTexels);
  RasterScene rs{raster_views()};
  const std::vector<Vtx> one{{.pos = rs.at(x0 + 10, x0 + 10, 0.5)},
                             {.pos = rs.at(x0 + 30, x0 + 10, 0.5)},
                             {.pos = rs.at(x0 + 10, x0 + 30, 0.5)}};
  rs.add_mesh(1, rs.base_record(46), one, {{0, 1, 2}}, true, 100);
  const std::vector<Vtx> chunk{{.pos = rs.at(x0 + 60, x0 + 60, 0.5)},
                               {.pos = rs.at(x0 + 90, x0 + 60, 0.5)},
                               {.pos = rs.at(x0 + 60, x0 + 90, 0.5)}};
  rs.add_mesh(2, rs.base_record(47), chunk, {{0, 1, 2}}, false);
  const std::vector<Vtx> gone{{.pos = rs.at(x0 + 10, x0 + 80, 0.5)},
                              {.pos = rs.at(x0 + 40, x0 + 80, 0.5)},
                              {.pos = rs.at(x0 + 10, x0 + 110, 0.5)}};
  rs.add_mesh(3, rs.base_record(48), gone, {{0, 1, 2}}, true, 101);
  rs.records[size_t{3} * RecordWords + record::Flags] = 0;
  rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
  for (const Variant& variant : opaque_variants()) {
    const wgpu::Texture atlas = depth_atlas();
    fill(atlas, 0.f);
    const CullOut out = draw(rs, atlas, variant);
    EXPECT_EQ(out.pairs[pairs_header::OpaqueCount], 3u);
    const std::vector<float> texels = read_atlas(atlas);
    uint32_t stray = 0;
    uint32_t drawn = 0;
    for (uint32_t y = 0; y < PageTexels; ++y) {
      for (uint32_t x = 0; x < PageTexels; ++x) {
        const double fx = x + 0.5;
        const double fy = y + 0.5;
        const bool first = fx > 10 && fy > 10 && fx + fy < 40;
        const bool second = fx > 60 && fy > 60 && fx + fy < 150;
        const bool edge = std::abs(fx + fy - 40) < 1 || std::abs(fx + fy - 150) < 1;
        const float d = texel(texels, 9, x, y);
        if (first || second) {
          drawn += d > 0.f ? 1 : 0;
        } else if (!edge) {
          stray += d != 0.f ? 1 : 0;
        }
      }
    }
    EXPECT_EQ(stray, 0u) << "clip " << variant.clip;
    EXPECT_GT(drawn, 300u) << "clip " << variant.clip;
  }
}

TEST_F(RenderVsmRasterGpu, InPlaceRerenderMatches) {
  RasterScene rs{raster_views()};
  std::vector<Vtx> verts;
  std::vector<Tri> tris;
  const auto x0 = static_cast<double>(TestPage * PageTexels);
  std::mt19937 rng{11};
  std::uniform_real_distribution<double> coord(x0 - 40.0, x0 + PageTexels + 40.0);
  std::uniform_real_distribution<double> depth(0.4, 0.6);
  for (uint32_t t = 0; t < 21; ++t) {
    for (uint32_t k = 0; k < 3; ++k) {
      verts.push_back({.pos = rs.at(coord(rng), coord(rng), depth(rng))});
    }
    tris.push_back({3 * t, 3 * t + 1, 3 * t + 2});
  }
  rs.add_mesh(1, rs.base_record(49), verts, tris, true, 100);
  const uint32_t v = rs.page_index_of(TestPage, TestPage);
  const Variant variant = opaque_variants().back();
  const wgpu::Texture atlas = depth_atlas();
  const wgpu::Texture copy = depth_atlas();
  fill(atlas, 0.f);
  rs.list({{v, 9}});
  (void)draw(rs, atlas, variant);
  const auto rerender = [&] {
    copy_texture(atlas, copy);
    rs.list({{v, 9}}, true);
    const CullOut out = draw(rs, atlas, variant);
    EXPECT_GT(out.pairs[pairs_header::OpaqueCount], 0u);
    return compare(rs, atlas, copy);
  };
  const std::vector<uint32_t> same = rerender();
  EXPECT_EQ(same[stat::CheckCompared], 1u);
  EXPECT_EQ(same[stat::CheckMismatched], 0u);
  EXPECT_EQ(same[stat::CheckTexels], 0u);
  const ViewRecord& view = rs.view();
  for (uint32_t k = 0; k < 3; ++k) {
    const float shift = 0.3f * view.anchorRel[k] * view.texel * view.texel;
    const uint32_t w = size_t{1} * RecordWords + record::World + k * 4 + 3;
    rs.records[w] = std::bit_cast<uint32_t>(std::bit_cast<float>(rs.records[w]) + shift);
  }
  const std::vector<uint32_t> moved = rerender();
  EXPECT_EQ(moved[stat::CheckCompared], 1u);
  EXPECT_EQ(moved[stat::CheckMismatched], 1u);
  EXPECT_GT(moved[stat::CheckTexels], 0u);
  const uint32_t key = ~moved[stat::CheckFirst];
  EXPECT_EQ(key & entry::PhysMask, 9u);
  EXPECT_EQ(key >> 22, 0u);
}

TEST_F(RenderVsmRasterGpu, SmallClustersCastDepth) {
  RasterScene rs{raster_views()};
  const auto x0 = static_cast<double>(TestPage * PageTexels);
  constexpr std::array<double, 4> Fractions{0.15, 0.35, 0.6, 0.65};
  constexpr uint32_t First = 100;
  Record rec = rs.base_record(21);
  std::vector<Vtx> all;
  std::vector<std::pair<uint32_t, uint32_t>> centres;
  for (uint32_t j = 0; j < 32; ++j) {
    for (uint32_t i = 0; i < 32; ++i) {
      const double fx = Fractions[(i * 3 + j) % 4];
      const double fy = Fractions[(j * 5 + i) % 4];
      const double x = x0 + 4.0 * i + fx;
      const double y = x0 + 4.0 * j + fy;
      const std::vector<Vtx> quad{{.pos = rs.at(x, y, 0.5)},
                                  {.pos = rs.at(x + 0.4, y, 0.5)},
                                  {.pos = rs.at(x, y + 0.4, 0.5)},
                                  {.pos = rs.at(x + 0.4, y + 0.4, 0.5)}};
      rs.add_cluster(First + j * 32 + i, static_cast<uint32_t>(all.size()), quad, {{0, 1, 2}, {2, 1, 3}});
      all.insert(all.end(), quad.begin(), quad.end());
      if (fx < 0.5 && fy < 0.5) {
        centres.emplace_back(4 * i, 4 * j);
      }
    }
  }
  rec[record::VertexFirst] = 0;
  rec[record::VertexCount] = static_cast<uint32_t>(all.size());
  rec[record::ClusterFirst] = First;
  rec[record::ClusterCount] = 1024;
  rs.push_vertices(all);
  rs.finish(1, rec, RasterScene::box_of(all));
  rs.list({{rs.page_index_of(TestPage, TestPage), 9}});
  for (const Variant& variant : opaque_variants()) {
    const wgpu::Texture atlas = depth_atlas();
    fill(atlas, 0.f);
    const CullOut out = draw(rs, atlas, variant);
    EXPECT_GE(out.pairs[pairs_header::OpaqueCount], centres.size());
    const std::vector<float> texels = read_atlas(atlas);
    uint32_t written = 0;
    for (uint32_t y = 0; y < PageTexels; ++y) {
      for (uint32_t x = 0; x < PageTexels; ++x) {
        written += texel(texels, 9, x, y) > 0.f ? 1 : 0;
      }
    }
    uint32_t hit = 0;
    for (const auto& [cx, cy] : centres) {
      hit += texel(texels, 9, cx, cy) > 0.f ? 1 : 0;
    }
    EXPECT_EQ(centres.size(), 256u);
    EXPECT_EQ(hit, centres.size()) << "clip " << variant.clip;
    EXPECT_EQ(written, centres.size()) << "clip " << variant.clip;
  }
}
} // namespace
} // namespace aurora::render::shadows::vsm

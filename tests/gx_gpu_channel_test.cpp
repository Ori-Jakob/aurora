// GPU-system commands through the FIFO: wire bytes, handles, suppression, order, budgets and the core registry

#include "gx_test_common.hpp"
#include "__gx.h"
#include "gfx/frame.hpp"
#include "gx/guest_memory.hpp"
#include "gx/pipeline.hpp"

#include <aurora/gpu_systems.hpp>

#include <array>
#include <atomic>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace gpu = aurora::render::gpu;
namespace channel = aurora::gx::gpu_channel;
namespace hints = aurora::gx::hints;
namespace wire = aurora::gx::hints::wire;
using aurora::gx::g_gxState;

namespace aurora::gx {
extern uint32_t g_testPipelineBuildCount;
} // namespace aurora::gx

namespace aurora::gfx {
extern uint32_t g_testDrawCount;
extern gx::DrawData g_testLastDraw;
extern bool g_testMergeDraws;
extern uint32_t g_testMergedDrawCount;
extern uint32_t g_testStoragePushes;
extern bool g_testStorageRanges;
extern std::vector<uint8_t> g_testStorage;
extern bool g_testChannelStorageFull;
namespace testing {
void reset_stub_ranges() noexcept;
} // namespace testing
namespace detail {
extern ChannelBudget g_testChannelBudget;
extern uint64_t g_testChannelDemand;
} // namespace detail
} // namespace aurora::gfx

namespace {
using Positions = std::array<f32, 9>;
constexpr Positions kTriangle{0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
constexpr std::array<u8, 5> kRealBpWrite{GX_LOAD_BP_REG, 0x41, 0x12, 0x34, 0x56};
constexpr std::string_view kSource = "@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z) fn main() {}";

std::vector<u8> bytes_of(const void* data, size_t size) {
  const auto* begin = static_cast<const u8*>(data);
  return {begin, begin + size};
}

std::vector<u8> storage_at(size_t offset, size_t size) {
  const auto& storage = aurora::gfx::g_testStorage;
  return offset + size <= storage.size() ? bytes_of(storage.data() + offset, size) : std::vector<u8>{};
}

u32 array_start(GXAttr attr) { return aurora::gfx::g_testLastDraw.immediateData.arrayStart[attr - GX_VA_POS]; }

void triangle() {
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  GXPosition1x8(0);
  GXPosition1x8(1);
  GXPosition1x8(2);
  GXEnd();
}

void write_bytes(const std::vector<u8>& bytes) {
  aurora::gx::fifo::write_data(bytes.data(), static_cast<u32>(bytes.size()));
}

std::vector<u8> raw_hint(u16 id, const std::vector<u8>& payload) {
  std::vector<u8> out(GX_AURORA_HINT_HEADER_SIZE + payload.size());
  wire::Writer writer{out};
  writer.header(id, static_cast<u32>(payload.size()));
  writer.bytes(payload);
  return out;
}

template <typename Fill>
std::vector<u8> payload(size_t size, Fill&& fill) {
  std::vector<u8> out(size);
  wire::Writer writer{out};
  fill(writer);
  return out;
}

std::vector<u8> load_array_gpu(u8 attr, u64 handle, u32 offset, u16 version = 1) {
  return raw_hint(GX_AURORA_LOAD_ARRAY_GPU, payload(wire::LoadArrayGpuV1, [&](wire::Writer& out) {
                    out.u16(version);
                    out.u8(attr);
                    out.u64(handle);
                    out.u32(offset);
                  }));
}

std::vector<u8> upload_command(u64 handle, std::vector<u8> data, u16 version = 1) {
  auto bytes = payload(wire::GpuUploadHeader, [&](wire::Writer& out) {
    out.u16(version);
    out.u64(handle);
  });
  bytes.insert(bytes.end(), data.begin(), data.end());
  return raw_hint(GX_AURORA_GPU_UPLOAD, bytes);
}

std::vector<u8> update_command(u64 system, u32 offset, size_t size) {
  auto bytes = payload(wire::GpuUpdateHeader, [&](wire::Writer& out) {
    out.u16(1);
    out.u64(system);
    out.u32(offset);
  });
  bytes.resize(bytes.size() + size, 0x5A);
  return raw_hint(GX_AURORA_GPU_UPDATE_PERSISTENT, bytes);
}

struct KernelBytes {
  std::string entry = "main";
  u16 x = 64;
  u16 y = 1;
  u16 z = 1;
  u8 flags = 0;
};

struct CreateBytes {
  u64 id = 1ull << 32 | 1;
  u32 persistentBytes = 0;
  u32 flags = 0;
  u32 requirements = 0;
  std::string name = "sys";
  std::vector<KernelBytes> kernels{KernelBytes{}};
  std::string source{kSource};
  int sourceLengthDelta = 0;
  int kernelCountOverride = -1;
};

std::vector<u8> create_command(const CreateBytes& create) {
  size_t size = wire::GpuCreateHeader + create.name.size() + create.source.size();
  for (const auto& kernel : create.kernels) {
    size += wire::GpuCreateKernelHeader + kernel.entry.size();
  }
  auto bytes = payload(size, [&](wire::Writer& out) {
    out.u16(1);
    out.u64(create.id);
    out.u32(create.persistentBytes);
    out.u32(create.flags);
    out.u32(create.requirements);
    out.u8(static_cast<u8>(create.name.size()));
    out.u8(static_cast<u8>(create.kernelCountOverride >= 0 ? create.kernelCountOverride : create.kernels.size()));
    out.u32(static_cast<u32>(static_cast<int>(create.source.size()) + create.sourceLengthDelta));
    out.bytes(bytes_of(create.name.data(), create.name.size()));
    for (const auto& kernel : create.kernels) {
      out.u8(static_cast<u8>(kernel.entry.size()));
      out.bytes(bytes_of(kernel.entry.data(), kernel.entry.size()));
      out.u16(kernel.x);
      out.u16(kernel.y);
      out.u16(kernel.z);
      out.u8(kernel.flags);
    }
    out.bytes(bytes_of(create.source.data(), create.source.size()));
  });
  return raw_hint(GX_AURORA_GPU_CREATE_SYSTEM, bytes);
}

u64 handle_for(u32 index) { return u64{channel::current_serial()} << 32 | index; }

struct FakeBackend {
  struct Event {
    std::string what;
    u64 id;
    u32 draws;
  };
  std::vector<Event> events;
  gpu::SystemStatus status = gpu::SystemStatus::Ready;
  u32 waits = 0;
  u32 resets = 0;

  channel::SystemBackend backend() {
    return {
        .userData = this,
        .create =
            [](void* self, const channel::SystemCreate& desc) noexcept {
              get(self).events.push_back({"create " + std::string{desc.name}, desc.id, aurora::gfx::g_testDrawCount});
            },
        .destroy = [](void* self,
                      u64 id) noexcept { get(self).events.push_back({"destroy", id, aurora::gfx::g_testDrawCount}); },
        .reset = [](void* self) noexcept { ++get(self).resets; },
        .status = [](void* self, u64) noexcept { return get(self).status; },
        .wait_all = [](void* self) noexcept { ++get(self).waits; },
    };
  }

private:
  static FakeBackend& get(void* self) { return *static_cast<FakeBackend*>(self); }
};

struct VertexReadCounter {
  std::atomic_uint32_t reads{0};
  aurora::gx::guest_memory::Hooks hooks{
      .resolve =
          [](void* self, aurora::gx::guest_memory::Kind kind, uint64_t address, size_t size) noexcept {
            if (kind == aurora::gx::guest_memory::Kind::VertexArray) {
              static_cast<VertexReadCounter*>(self)->reads.fetch_add(1, std::memory_order_relaxed);
            }
            return std::span<const uint8_t>{reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address)), size};
          },
      .userData = this,
  };
  VertexReadCounter() { aurora::gx::guest_memory::set_hooks(&hooks); }
  ~VertexReadCounter() { aurora::gx::guest_memory::set_hooks(nullptr); }
  VertexReadCounter(const VertexReadCounter&) = delete;
  VertexReadCounter& operator=(const VertexReadCounter&) = delete;
};
} // namespace

class GXGpuChannelTest : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    aurora::gfx::testing::reset_stub_ranges();
    aurora::gfx::g_testStorageRanges = true;
    aurora::gfx::g_testChannelStorageFull = false;
    aurora::gfx::g_testMergeDraws = false;
    aurora::gfx::detail::g_testChannelBudget = {.bytes = UINT64_MAX / 4, .alignment = 256};
    aurora::gfx::detail::g_testChannelDemand = 0;
    aurora::gx::fifo::invalidate_draw_cache();
    aurora::gx::fifo::testing::reset_draw_route_stats();
    aurora::gx::fifo::testing::reset_array_upload_stats();
    hints::testing::reset_counters();
    hints::set_sink(nullptr);

    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_INDEX8);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
    decode_fifo(flush_and_capture());
    aurora::gfx::g_testDrawCount = 0;
    aurora::gfx::g_testStoragePushes = 0;
  }

  void TearDown() override {
    channel::set_system_backend(nullptr);
    aurora::gfx::g_testStorageRanges = false;
    aurora::gfx::g_testChannelStorageFull = false;
    aurora::gfx::g_testMergeDraws = false;
    aurora::gfx::detail::g_testChannelBudget = {.bytes = UINT64_MAX / 4, .alignment = 256};
    aurora::gx::fifo::invalidate_draw_cache();
    GXFifoTest::TearDown();
  }

  void decode() { decode_fifo(capture_fifo()); }

  void expect_suppressed_then_bp() {
    const uint32_t draws = aurora::gfx::g_testDrawCount;
    const uint64_t suppressed = gpu::stats().suppressedDraws;
    write_bytes({kRealBpWrite.begin(), kRealBpWrite.end()});
    g_gxState.bpRegCache[0x41] = 0;
    decode();
    EXPECT_EQ(aurora::gfx::g_testDrawCount, draws);
    EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
    EXPECT_EQ(gpu::stats().suppressedDraws, suppressed + 1);
    EXPECT_EQ(aurora::gx::fifo::draw_route_stats().suppressed, 0u);
  }

  void next_frame() {
    capture_fifo();
    aurora::gx::fifo::end_frame();
  }
};

TEST_F(GXGpuChannelTest, WireBytes) {
  channel::testing::set_serial(0x01020304);
  const std::array<u8, 3> data{1, 2, 3};
  const auto handle = gpu::upload(data.data(), static_cast<u32>(data.size()));
  EXPECT_EQ(handle, 0x0102030400000001ull);
  EXPECT_EQ(capture_fifo(), (std::vector<u8>{0x50, 0x01, 0x20, 0x00, 0x00, 0x00, 0x0D, 0x00, 0x01, 0x01,
                                             0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x01, 1,    2,    3}));

  const auto reserved = gpu::reserve(10, gpu::ReserveZero);
  EXPECT_EQ(reserved, 0x0102030400000002ull);
  EXPECT_EQ(capture_fifo(),
            (std::vector<u8>{0x50, 0x01, 0x21, 0x00, 0x00, 0x00, 0x12, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04,
                             0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x01}));

  GXSetArrayGPU(GX_VA_CLR0, handle, 8, 4);
  const std::vector<u8> clr0{0x50, 0x01, 0x23, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x01, 0x0B, 0x01, 0x02, 0x03, 0x04,
                             0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x08, 0xB2, 0x00, 0x00, 0x00, 0x04};
  EXPECT_EQ(capture_fifo(), clr0);

  GXSetArrayGPU(GX_VA_NBT, handle, 0, 12);
  const auto nbt = capture_fifo();
  ASSERT_EQ(nbt.size(), 28u);
  EXPECT_EQ(nbt[9], GX_VA_NRM);
  EXPECT_EQ(nbt[23], 0xB1);

  GXSetArrayGPU(GX_VA_PNMTXIDX, handle, 0, 12);
  GXSetArrayGPU(GX_VA_TEX0MTXIDX, handle, 0, 12);
  EXPECT_TRUE(capture_fifo().empty());

  write_bytes(load_array_gpu(21, handle, 0));
  decode();
  EXPECT_EQ(hints::counters(GX_AURORA_LOAD_ARRAY_GPU).malformed, 1u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 0u);
}

TEST_F(GXGpuChannelTest, HandleSerial) {
  channel::testing::set_serial(7);
  const std::array<u8, 4> data{};
  EXPECT_EQ(gpu::upload(data.data(), 4), 7ull << 32 | 1);
  EXPECT_EQ(gpu::upload(data.data(), 4), 7ull << 32 | 2);
  EXPECT_EQ(gpu::reserve(4), 7ull << 32 | 3);
  EXPECT_TRUE(channel::handle_is_current(7ull << 32 | 3));
  EXPECT_FALSE(channel::handle_is_current(7ull << 32 | 4));
  next_frame();
  EXPECT_FALSE(channel::handle_is_current(7ull << 32 | 1));
  EXPECT_EQ(gpu::upload(data.data(), 4), 8ull << 32 | 1);
  channel::testing::set_serial(UINT32_MAX);
  next_frame();
  EXPECT_EQ(channel::current_serial(), 1u);
  EXPECT_EQ(gpu::upload(data.data(), 4), 1ull << 32 | 1);
}

TEST_F(GXGpuChannelTest, HeldHandleNeverResolves) {
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  triangle();
  decode();
  ASSERT_EQ(aurora::gfx::g_testDrawCount, 1u);
  next_frame();
  for (int frame = 0; frame < 300; ++frame) {
    GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
    triangle();
    decode();
    next_frame();
  }
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gpu::stats().staleBinds, 300u);
  EXPECT_EQ(gpu::stats().suppressedDraws, 300u);

  write_bytes(load_array_gpu(GX_VA_POS, handle, 0));
  triangle();
  expect_suppressed_then_bp();
}

TEST_F(GXGpuChannelTest, GapsAndDuplicates) {
  const auto data = bytes_of(kTriangle.data(), sizeof(kTriangle));
  write_bytes(upload_command(handle_for(1), data));
  write_bytes(upload_command(handle_for(2), {}));
  write_bytes(upload_command(handle_for(3), data));
  decode();
  EXPECT_EQ(hints::counters(GX_AURORA_GPU_UPLOAD).malformed, 1u);
  EXPECT_EQ(channel::lookup(handle_for(1)).state, channel::HandleState::Uploaded);
  EXPECT_EQ(channel::lookup(handle_for(2)).state, channel::HandleState::Failed);
  EXPECT_EQ(channel::lookup(handle_for(3)).state, channel::HandleState::Uploaded);
  EXPECT_EQ(storage_at(channel::lookup(handle_for(3)).range.offset, data.size()), data);

  const auto stats = gpu::stats();
  write_bytes(upload_command(handle_for(3 + 257 + 1), data));
  write_bytes(upload_command(handle_for(3), data));
  write_bytes(upload_command(handle_for(0), data));
  write_bytes(upload_command(u64{channel::current_serial() + 1} << 32 | 4, data));
  decode();
  EXPECT_EQ(gpu::stats().rejected, stats.rejected + 4);
  EXPECT_EQ(gpu::stats().uploads, stats.uploads);
  EXPECT_EQ(channel::lookup(handle_for(4)).state, channel::HandleState::Unknown);
  write_bytes(upload_command(handle_for(4), data));
  decode();
  EXPECT_EQ(channel::lookup(handle_for(4)).state, channel::HandleState::Uploaded);
}

TEST_F(GXGpuChannelTest, UploadBindDraw) {
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  ASSERT_NE(handle, gpu::InvalidHandle);
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  triangle();
  const VertexReadCounter reads;
  const CpuZoneDelta zones;
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(array_start(GX_VA_POS), 256u);
  EXPECT_EQ(storage_at(256, sizeof(kTriangle)), bytes_of(kTriangle.data(), sizeof(kTriangle)));
  EXPECT_EQ(reads.reads.load(), 0u);
  EXPECT_EQ(zones.calls(aurora::gx::fifo::CpuZoneArrays), 0u);
  EXPECT_EQ(aurora::gfx::g_testStoragePushes, 0u);
  EXPECT_EQ(aurora::gx::fifo::array_upload_stats().reused, 0u);
  EXPECT_EQ(aurora::gx::fifo::immutable_array_stats().hits, 0u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 1u);
  EXPECT_TRUE(g_gxState.arrays[GX_VA_POS].gpu);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].data, nullptr);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].stride, 12u);
  const auto stats = gpu::stats();
  EXPECT_EQ(stats.uploads, 1u);
  EXPECT_EQ(stats.uploadBytes, sizeof(kTriangle));
  EXPECT_EQ(stats.binds, 1u);
  EXPECT_EQ(stats.suppressedDraws, 0u);
}

TEST_F(GXGpuChannelTest, ByteOffsetSharesOneHandle) {
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  std::array<u8, sizeof(kTriangle) + 12> data{};
  std::memcpy(data.data(), kTriangle.data(), sizeof(kTriangle));
  const auto handle = gpu::upload(data.data(), static_cast<u32>(data.size()));
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  GXSetArrayGPU(GX_VA_CLR0, handle, 36, 4);
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  for (u8 i = 0; i < 3; ++i) {
    GXPosition1x8(i);
    GXColor1x8(i);
  }
  GXEnd();
  decode_fifo(flush_and_capture());
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(array_start(GX_VA_POS), 256u);
  EXPECT_EQ(array_start(GX_VA_CLR0), 256u + 36u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 1u << (GX_VA_CLR0 - GX_VA_POS) | 1u);
}

TEST_F(GXGpuChannelTest, MergesAndDirtyBits) {
  aurora::gfx::g_testMergeDraws = true;
  GXSetArray(GX_VA_POS, kTriangle.data(), sizeof(kTriangle), 12, false);
  triangle();
  decode();
  const uint32_t builds = aurora::gx::g_testPipelineBuildCount;

  const auto first = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  const auto second = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  GXSetArrayGPU(GX_VA_POS, first, 0, 12);
  decode();
  EXPECT_EQ(g_gxState.dirty & (aurora::gx::DirtyPipeline | aurora::gx::DirtyImmediates),
            aurora::gx::DirtyPipeline | aurora::gx::DirtyImmediates);
  triangle();
  decode();
  EXPECT_EQ(aurora::gx::g_testPipelineBuildCount, builds + 1);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);

  const uint32_t merged = aurora::gfx::g_testMergedDrawCount;
  GXSetArrayGPU(GX_VA_POS, first, 0, 12);
  triangle();
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(aurora::gfx::g_testMergedDrawCount, merged + 1);

  GXSetArrayGPU(GX_VA_POS, second, 0, 12);
  decode();
  EXPECT_EQ(g_gxState.dirty & aurora::gx::DirtyPipeline, 0);
  triangle();
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 3u);
  EXPECT_EQ(array_start(GX_VA_POS), channel::lookup(second).range.offset);

  GXSetArrayGPU(GX_VA_POS, second, 12, 12);
  triangle();
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 4u);
  EXPECT_EQ(array_start(GX_VA_POS), channel::lookup(second).range.offset + 12);
  EXPECT_EQ(aurora::gx::g_testPipelineBuildCount, builds + 1);
}

TEST_F(GXGpuChannelTest, InvalVtxAndArraybase) {
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  triangle();
  decode();
  const VertexReadCounter reads;
  GXInvalidateVtxCache();
  triangle();
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(array_start(GX_VA_POS), 256u);
  EXPECT_EQ(reads.reads.load(), 0u);

  GXSetArray(GX_VA_POS, nullptr, 0, 12, true);
  decode();
  EXPECT_FALSE(g_gxState.arrays[GX_VA_POS].gpu);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].gpuHandle, 0u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 0u);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].stride, 12u);

  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  GXSetArray(GX_VA_POS, kTriangle.data(), sizeof(kTriangle), 12, true);
  triangle();
  decode();
  EXPECT_EQ(g_gxState.gpuArrayMask, 0u);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].data, kTriangle.data());
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].stride, 12u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 3u);
}

TEST_F(GXGpuChannelTest, Suppression) {
  {
    SCOPED_TRACE("handle 0");
    GXSetArrayGPU(GX_VA_POS, gpu::InvalidHandle, 0, 12);
    triangle();
    expect_suppressed_then_bp();
    EXPECT_EQ(gpu::stats().invalidBinds, 1u);
  }
  {
    SCOPED_TRACE("unknown index");
    write_bytes(load_array_gpu(GX_VA_POS, handle_for(5), 0));
    triangle();
    expect_suppressed_then_bp();
  }
  {
    SCOPED_TRACE("reserve without zero");
    GXSetArrayGPU(GX_VA_POS, gpu::reserve(sizeof(kTriangle)), 0, 12);
    triangle();
    expect_suppressed_then_bp();
  }
  {
    SCOPED_TRACE("offset past the range");
    GXSetArrayGPU(GX_VA_POS, gpu::upload(kTriangle.data(), sizeof(kTriangle)), sizeof(kTriangle), 12);
    triangle();
    expect_suppressed_then_bp();
  }
  {
    SCOPED_TRACE("storage miss");
    aurora::gfx::g_testChannelStorageFull = true;
    GXSetArrayGPU(GX_VA_POS, gpu::upload(kTriangle.data(), sizeof(kTriangle)), 0, 12);
    triangle();
    expect_suppressed_then_bp();
    aurora::gfx::g_testChannelStorageFull = false;
    EXPECT_EQ(gpu::stats().storageMisses, 1u);
    EXPECT_EQ(gpu::stats().failedHandles, 1u);
  }
  {
    SCOPED_TRACE("stale raw handle");
    const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
    next_frame();
    write_bytes(load_array_gpu(GX_VA_POS, handle, 0));
    triangle();
    expect_suppressed_then_bp();
  }
  {
    SCOPED_TRACE("zero reserve");
    const auto handle = gpu::reserve(sizeof(kTriangle), gpu::ReserveZero);
    GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
    triangle();
    decode();
    EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
    EXPECT_EQ(storage_at(array_start(GX_VA_POS), sizeof(kTriangle)), std::vector<u8>(sizeof(kTriangle), 0));
    EXPECT_EQ(channel::lookup(handle).state, channel::HandleState::Written);
  }
}

TEST_F(GXGpuChannelTest, Order) {
  FakeBackend fake;
  const auto backend = fake.backend();
  channel::set_system_backend(&backend);
  write_bytes(load_array_gpu(GX_VA_POS, handle_for(1), 0));
  triangle();
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  const std::array<gpu::KernelDesc, 1> kernels{{{.entryPoint = "main"}}};
  const auto system = gpu::create_system({.name = "sys", .wgsl = kSource, .kernels = kernels, .persistentBytes = 8});
  const std::array<u32, 2> words{11, 22};
  gpu::update_persistent(system, 0, words.data(), sizeof(words));
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  triangle();
  const auto zeroed = gpu::reserve(16, gpu::ReserveZero);
  gpu::destroy_system(system);
  GXSetArrayGPU(GX_VA_POS, zeroed, 0, 12);
  triangle();
  decode();

  EXPECT_EQ(gpu::stats().suppressedDraws, 1u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(array_start(GX_VA_POS), channel::lookup(zeroed).range.offset);
  EXPECT_GT(channel::lookup(zeroed).range.offset, channel::lookup(handle).range.offset);
  ASSERT_EQ(fake.events.size(), 2u);
  EXPECT_EQ(fake.events[0].what, "create sys");
  EXPECT_EQ(fake.events[0].id, system);
  EXPECT_EQ(fake.events[0].draws, 0u);
  EXPECT_EQ(fake.events[1].what, "destroy");
  EXPECT_EQ(fake.events[1].draws, 1u);
  EXPECT_EQ(gpu::stats().updates, 1u);
}

TEST_F(GXGpuChannelTest, BudgetRefusesSynchronously) {
  aurora::gfx::detail::g_testChannelBudget = {.bytes = 4096, .alignment = 256};
  const std::vector<u8> data(3000, 7);
  EXPECT_NE(gpu::upload(data.data(), 3000), gpu::InvalidHandle);
  const u32 size = aurora::gx::fifo::get_buffer_size();
  EXPECT_EQ(gpu::upload(data.data(), 1000), gpu::InvalidHandle);
  EXPECT_EQ(aurora::gx::fifo::get_buffer_size(), size);
  EXPECT_EQ(gpu::stats().refused, 1u);
  EXPECT_EQ(gpu::stats().refusedBytes, 1000u);
  EXPECT_EQ(gpu::reserve(1000), gpu::InvalidHandle);
  next_frame();
  EXPECT_EQ(aurora::gfx::detail::g_testChannelDemand, 3328u + 1280u + 1280u);

  aurora::gfx::detail::g_testChannelDemand = 0;
  gpu::request_capacity(1u << 20);
  next_frame();
  EXPECT_GE(aurora::gfx::detail::g_testChannelDemand, 1u << 20);
  gpu::request_capacity(0);
}

TEST_F(GXGpuChannelTest, FrameCaps) {
  const std::array<u8, 64> data{};
  channel::testing::set_frame_caps(UINT64_MAX, UINT64_MAX, 2);
  EXPECT_NE(gpu::upload(data.data(), 4), gpu::InvalidHandle);
  EXPECT_NE(gpu::reserve(4), gpu::InvalidHandle);
  EXPECT_EQ(gpu::upload(data.data(), 4), gpu::InvalidHandle);
  next_frame();
  EXPECT_NE(gpu::upload(data.data(), 4), gpu::InvalidHandle);

  channel::testing::set_frame_caps(UINT64_MAX, 100, gpu::MaxFrameHandles);
  next_frame();
  EXPECT_NE(gpu::upload(data.data(), 60), gpu::InvalidHandle);
  EXPECT_EQ(gpu::upload(data.data(), 60), gpu::InvalidHandle);
  EXPECT_NE(gpu::reserve(60), gpu::InvalidHandle);
  next_frame();
  EXPECT_NE(gpu::upload(data.data(), 60), gpu::InvalidHandle);
  EXPECT_EQ(gpu::stats().refused, 2u);
}

TEST_F(GXGpuChannelTest, StatsHaveOneWriter) {
  aurora::gx::fifo::init();
  aurora::gx::fifo::begin_frame();
  aurora::gfx::detail::g_testChannelBudget = {.bytes = 0, .alignment = 256};
  EXPECT_EQ(gpu::upload(kTriangle.data(), sizeof(kTriangle)), gpu::InvalidHandle);
  write_bytes(upload_command(0, bytes_of(kTriangle.data(), sizeof(kTriangle))));
  aurora::gx::fifo::publish();
  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::shutdown();
  EXPECT_EQ(gpu::stats().refused, 1u);
  EXPECT_EQ(gpu::stats().rejected, 1u);
  EXPECT_EQ(gpu::stats().uploads, 0u);
}

TEST_F(GXGpuChannelTest, FrameEndInvalidatesBindings) {
  GXSetArrayGPU(GX_VA_POS, gpu::upload(kTriangle.data(), sizeof(kTriangle)), 0, 12);
  triangle();
  decode();
  ASSERT_EQ(aurora::gfx::g_testDrawCount, 1u);
  next_frame();
  EXPECT_TRUE(g_gxState.arrays[GX_VA_POS].gpu);
  EXPECT_EQ(g_gxState.arrays[GX_VA_POS].gpuHandle, 0u);
  triangle();
  expect_suppressed_then_bp();
  GXSetArrayGPU(GX_VA_POS, gpu::upload(kTriangle.data(), sizeof(kTriangle)), 0, 12);
  triangle();
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
}

TEST_F(GXGpuChannelTest, DisplayListsRefuse) {
  FakeBackend fake;
  const auto backend = fake.backend();
  channel::set_system_backend(&backend);
  const std::array<gpu::KernelDesc, 1> kernels{{{.entryPoint = "main"}}};
  const auto system = gpu::create_system({.name = "sys", .wgsl = kSource, .kernels = kernels, .persistentBytes = 8});
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  decode();
  const auto before = gpu::stats();

  alignas(32) std::array<u8, 256> list{};
  GXBeginDisplayList(list.data(), static_cast<u32>(list.size()));
  EXPECT_EQ(gpu::upload(kTriangle.data(), sizeof(kTriangle)), gpu::InvalidHandle);
  EXPECT_EQ(gpu::reserve(16), gpu::InvalidHandle);
  EXPECT_EQ(gpu::create_system({.name = "list", .wgsl = kSource, .kernels = kernels}), gpu::InvalidSystem);
  const u32 word = 1;
  gpu::update_persistent(system, 0, &word, 4);
  gpu::destroy_system(system);
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  triangle();
  const u32 size = GXEndDisplayList();
  EXPECT_EQ(gpu::stats().refused, before.refused + 5);
  EXPECT_TRUE(channel::handle_is_current(handle));

  GXCallDisplayList(list.data(), size);
  GXCallDisplayList(list.data(), size);
  decode();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 0u);
  EXPECT_EQ(gpu::stats().suppressedDraws, before.suppressedDraws + 2);
  EXPECT_EQ(gpu::stats().invalidBinds, before.invalidBinds + 2);
  EXPECT_EQ(gpu::stats().liveSystems, 1u);
}

TEST_F(GXGpuChannelTest, MalformedInputLeavesStateUnchanged) {
  FakeBackend fake;
  const auto backend = fake.backend();
  channel::set_system_backend(&backend);
  const auto data = bytes_of(kTriangle.data(), sizeof(kTriangle));
  const auto reserve = [](u32 bytes, u32 flags, size_t size = wire::GpuReserveV1) {
    auto bytesOut = payload(size, [&](wire::Writer& out) {
      out.u16(1);
      out.u64(handle_for(1));
      if (size >= wire::GpuReserveV1) {
        out.u32(bytes);
        out.u32(flags);
      }
    });
    return raw_hint(GX_AURORA_GPU_RESERVE, bytesOut);
  };
  struct Case {
    std::string label;
    u16 id;
    std::vector<u8> bytes;
  };
  CreateBytes misaligned;
  misaligned.persistentBytes = 10;
  CreateBytes noKernels;
  noKernels.kernels.clear();
  CreateBytes nineKernels;
  nineKernels.kernels.assign(9, KernelBytes{});
  CreateBytes longName;
  longName.name = std::string(32, 'n');
  CreateBytes badEntry;
  badEntry.kernels[0].entry = "1main";
  CreateBytes wideGroup;
  wideGroup.kernels[0].x = 129;
  CreateBytes shortSource;
  shortSource.sourceLengthDelta = 1;
  const std::vector<Case> cases{
      {"upload version 2", GX_AURORA_GPU_UPLOAD, upload_command(handle_for(1), data, 2)},
      {"upload without data", GX_AURORA_GPU_UPLOAD, upload_command(handle_for(1), {})},
      {"reserve size", GX_AURORA_GPU_RESERVE, reserve(16, 0, wire::GpuReserveV1 - 1)},
      {"reserve flag bit 1", GX_AURORA_GPU_RESERVE, reserve(16, 2)},
      {"reserve zero bytes", GX_AURORA_GPU_RESERVE, reserve(0, 0)},
      {"misaligned update", GX_AURORA_GPU_UPDATE_PERSISTENT, update_command(1ull << 32 | 1, 2, 4)},
      {"short update", GX_AURORA_GPU_UPDATE_PERSISTENT, update_command(1ull << 32 | 1, 0, 2)},
      {"persistentBytes 10", GX_AURORA_GPU_CREATE_SYSTEM, create_command(misaligned)},
      {"kernelCount 0", GX_AURORA_GPU_CREATE_SYSTEM, create_command(noKernels)},
      {"kernelCount 9", GX_AURORA_GPU_CREATE_SYSTEM, create_command(nineKernels)},
      {"32-byte name", GX_AURORA_GPU_CREATE_SYSTEM, create_command(longName)},
      {"bad identifier", GX_AURORA_GPU_CREATE_SYSTEM, create_command(badEntry)},
      {"workgroup product 129", GX_AURORA_GPU_CREATE_SYSTEM, create_command(wideGroup)},
      {"length mismatch", GX_AURORA_GPU_CREATE_SYSTEM, create_command(shortSource)},
      {"attribute 21", GX_AURORA_LOAD_ARRAY_GPU, load_array_gpu(21, handle_for(1), 0)},
      {"destroy size", GX_AURORA_GPU_DESTROY_SYSTEM, raw_hint(GX_AURORA_GPU_DESTROY_SYSTEM, {0x00, 0x01, 0x02})},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.label);
    const uint64_t malformed = hints::counters(c.id).malformed;
    write_bytes(c.bytes);
    write_bytes({kRealBpWrite.begin(), kRealBpWrite.end()});
    g_gxState.bpRegCache[0x41] = 0;
    decode();
    EXPECT_EQ(hints::counters(c.id).malformed, malformed + 1);
    EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
  }
  const auto stats = gpu::stats();
  EXPECT_EQ(stats.uploads, 0u);
  EXPECT_EQ(stats.reserves, 0u);
  EXPECT_EQ(stats.created, 0u);
  EXPECT_EQ(stats.updates, 0u);
  EXPECT_EQ(stats.rejected, 0u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 0u);
  EXPECT_TRUE(fake.events.empty());
  EXPECT_EQ(channel::lookup(handle_for(1)).state, channel::HandleState::Unknown);
}

TEST_F(GXGpuChannelTest, DecodesWithoutSink) {
  const uint64_t hintBytes = hints::total_bytes();
  const uint64_t gpuBytes = hints::gpu_command_bytes();
  const auto handle = gpu::upload(kTriangle.data(), sizeof(kTriangle));
  GXSetArrayGPU(GX_VA_POS, handle, 0, 12);
  decode();
  EXPECT_EQ(hints::counters(GX_AURORA_GPU_UPLOAD).decoded, 1u);
  EXPECT_EQ(hints::counters(GX_AURORA_LOAD_ARRAY_GPU).decoded, 1u);
  EXPECT_EQ(hints::total_bytes(), hintBytes);
  EXPECT_EQ(hints::gpu_command_bytes(), gpuBytes + 17 + sizeof(kTriangle) + 22);
}

TEST_F(GXGpuChannelTest, SystemsInCore) {
  const std::array<gpu::KernelDesc, 1> kernels{{{.entryPoint = "main"}}};
  const gpu::SystemDesc stateful{.name = "sys", .wgsl = kSource, .kernels = kernels, .persistentBytes = 16};
  EXPECT_FALSE(gpu::systems_available());
  EXPECT_EQ(gpu::create_system(stateful), gpu::InvalidSystem);

  FakeBackend fake;
  const auto backend = fake.backend();
  channel::set_system_backend(&backend);
  const auto system = gpu::create_system(stateful);
  ASSERT_NE(system, gpu::InvalidSystem);
  EXPECT_EQ(system >> 32, 1u);
  EXPECT_EQ(gpu::system_status(system), gpu::SystemStatus::Pending);
  const std::array<u32, 2> words{0xAABBCCDD, 0x11223344};
  gpu::update_persistent(system, 4, words.data(), sizeof(words));
  gpu::update_persistent(system, 12, words.data(), sizeof(words));
  decode();
  EXPECT_EQ(gpu::stats().created, 1u);
  EXPECT_EQ(gpu::stats().liveSystems, 1u);
  EXPECT_EQ(gpu::stats().refused, 2u);
  const auto shadow = channel::shadow(system);
  ASSERT_EQ(shadow.size(), 16u);
  EXPECT_EQ(std::vector<u8>(shadow.begin() + 4, shadow.begin() + 12), bytes_of(words.data(), sizeof(words)));
  EXPECT_EQ(std::vector<u8>(shadow.begin(), shadow.begin() + 4), std::vector<u8>(4, 0));

  write_bytes(update_command(system, 12, 8));
  decode();
  EXPECT_EQ(gpu::stats().rejected, 1u);

  const auto gpuOnly = gpu::create_system({.name = "gpu_only",
                                           .wgsl = kSource,
                                           .kernels = kernels,
                                           .persistentBytes = 64,
                                           .flags = gpu::SystemGpuOnlyState});
  ASSERT_NE(gpuOnly, gpu::InvalidSystem);
  gpu::update_persistent(gpuOnly, 0, words.data(), sizeof(words));
  write_bytes(update_command(gpuOnly, 0, 8));
  decode();
  EXPECT_EQ(gpu::stats().refused, 3u);
  EXPECT_EQ(gpu::stats().rejected, 2u);
  EXPECT_TRUE(channel::shadow(gpuOnly).empty());
  EXPECT_TRUE(channel::uncaptured_gpu_state());

  next_frame();
  EXPECT_EQ(gpu::system_status(system), gpu::SystemStatus::Ready);
  fake.status = gpu::SystemStatus::Failed;
  EXPECT_EQ(gpu::system_status(system), gpu::SystemStatus::Failed);
  fake.status = gpu::SystemStatus::Ready;

  gpu::destroy_system(system);
  gpu::destroy_system(gpuOnly);
  decode();
  EXPECT_EQ(gpu::system_status(system), gpu::SystemStatus::Invalid);
  EXPECT_FALSE(channel::uncaptured_gpu_state());
  const u64 refused = gpu::stats().refused;
  gpu::update_persistent(system, 0, words.data(), sizeof(words));
  EXPECT_EQ(gpu::stats().refused, refused + 1);
  write_bytes(update_command(system, 0, 8));
  decode();
  EXPECT_EQ(gpu::stats().rejected, 3u);
  const auto reused = gpu::create_system(stateful);
  EXPECT_EQ(static_cast<u32>(reused), static_cast<u32>(system));
  EXPECT_EQ(reused >> 32, 2u);
  decode();
  EXPECT_EQ(gpu::stats().created, 3u);
  EXPECT_EQ(gpu::stats().destroyed, 2u);
  gpu::destroy_system(reused);
  decode();
}

TEST_F(GXGpuChannelTest, SystemAndShadowCaps) {
  FakeBackend fake;
  const auto backend = fake.backend();
  channel::set_system_backend(&backend);
  const std::array<gpu::KernelDesc, 1> kernels{{{.entryPoint = "main"}}};
  std::vector<gpu::SystemId> systems;
  for (u32 i = 0; i < gpu::MaxSystems; ++i) {
    systems.push_back(gpu::create_system({.name = "sys", .wgsl = kSource, .kernels = kernels}));
    ASSERT_NE(systems.back(), gpu::InvalidSystem) << i;
  }
  EXPECT_EQ(gpu::create_system({.name = "sys", .wgsl = kSource, .kernels = kernels}), gpu::InvalidSystem);
  decode();
  EXPECT_EQ(gpu::stats().liveSystems, gpu::MaxSystems);
  CreateBytes extra;
  extra.id = 1ull << 32 | (gpu::MaxSystems + 1);
  write_bytes(create_command(extra));
  decode();
  EXPECT_EQ(gpu::stats().rejected, 1u);
  for (const auto id : systems) {
    gpu::destroy_system(id);
  }
  decode();
  EXPECT_EQ(gpu::stats().liveSystems, 0u);

  const u32 quarter = gpu::MaxPersistentBytes;
  for (u32 i = 0; i < 4; ++i) {
    ASSERT_NE(gpu::create_system({.name = "big", .wgsl = kSource, .kernels = kernels, .persistentBytes = quarter}),
              gpu::InvalidSystem);
  }
  EXPECT_EQ(gpu::create_system({.name = "big", .wgsl = kSource, .kernels = kernels, .persistentBytes = 4}),
            gpu::InvalidSystem);
  decode();
  CreateBytes fifth;
  fifth.id = 9ull << 32 | 200;
  fifth.persistentBytes = 4;
  write_bytes(create_command(fifth));
  decode();
  EXPECT_EQ(gpu::stats().rejected, 2u);
  EXPECT_EQ(gpu::stats().liveSystems, 4u);
}

TEST_F(GXGpuChannelTest, MatchesLegacyStream) {
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  const std::array<u8, 12> colors{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255};
  GXSetArray(GX_VA_POS, kTriangle.data(), sizeof(kTriangle), 12, true);
  GXSetArray(GX_VA_CLR0, colors.data(), sizeof(colors), 4, true);
  decode_fifo(flush_and_capture());
  for (int i = 0; i < 2; ++i) {
    GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
    for (u8 v = 0; v < 3; ++v) {
      GXPosition1x8(v);
      GXColor1x8(v);
    }
    GXEnd();
  }
  decode();
  EXPECT_EQ(g_gxState.dirty, 0);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(aurora::gfx::g_testStoragePushes, 2u);
  EXPECT_EQ(array_start(GX_VA_POS), 256u);
  EXPECT_EQ(array_start(GX_VA_CLR0), 512u);
  EXPECT_EQ(g_gxState.gpuArrayMask, 0u);
  const auto stats = gpu::stats();
  EXPECT_EQ(stats.uploads + stats.binds + stats.suppressedDraws + stats.invalidBinds, 0u);
}

TEST_F(GXGpuChannelTest, LargeUploadThroughThreadedFifo) {
  std::vector<u8> data(70000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<u8>(i * 7);
  }
  aurora::gx::fifo::init();
  aurora::gx::fifo::begin_frame();
  const auto handle = gpu::upload(data.data(), static_cast<u32>(data.size()));
  write_bytes({kRealBpWrite.begin(), kRealBpWrite.end()});
  aurora::gx::fifo::publish();
  aurora::gx::fifo::drain();
  const auto info = channel::lookup(handle);
  EXPECT_EQ(info.state, channel::HandleState::Uploaded);
  EXPECT_EQ(info.range.size, data.size());
  EXPECT_EQ(storage_at(info.range.offset, data.size()), data);
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::shutdown();
  EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
}

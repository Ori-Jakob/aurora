#include "gx_test_common.hpp"
#include "__gx.h"
#include "../tools/gxreplay/self_test.hpp"
#include "gx/guest_memory.hpp"
#include "trace/capture_session.hpp"
#include "trace/snapshot_store.hpp"
#include "trace/state_preamble.hpp"
#include "trace/trace_reader.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <vector>

using aurora::gx::g_gxState;
using aurora::gx::GXState;

namespace aurora::gfx {
extern uint32_t g_testDrawCount;
namespace testing {
extern std::atomic<uint32_t> resolvePassCount;
} // namespace testing
} // namespace aurora::gfx

namespace {
struct PreambleCommand {
  u8 opcode;
  u32 reg;
};

bool parse_preamble(std::span<const u8> bytes, std::vector<PreambleCommand>& out) {
  size_t pos = 0;
  const auto be32 = [&](size_t at) {
    return static_cast<u32>(bytes[at]) << 24 | static_cast<u32>(bytes[at + 1]) << 16 |
           static_cast<u32>(bytes[at + 2]) << 8 | bytes[at + 3];
  };
  while (pos < bytes.size()) {
    const u8 opcode = bytes[pos];
    if (opcode == GX_LOAD_BP_REG) {
      out.push_back({opcode, bytes[pos + 1]});
      pos += 5;
    } else if (opcode == GX_LOAD_CP_REG) {
      out.push_back({opcode, bytes[pos + 1]});
      pos += 6;
    } else if (opcode == GX_LOAD_XF_REG) {
      const u32 header = be32(pos + 1);
      out.push_back({opcode, header & 0xFFFF});
      pos += 5 + ((header >> 16) + 1) * 4;
    } else if (opcode == GX_AURORA) {
      const u32 subCmd = static_cast<u32>(bytes[pos + 1]) << 8 | bytes[pos + 2];
      size_t payload = 0;
      if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
        payload = 24;
      } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER || subCmd == GX_AURORA_LOAD_COPY_SRC) {
        payload = 16;
      } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
        payload = 64;
      } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0F)) {
        payload = 13;
      } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
        payload = 34;
      } else if (subCmd == GX_AURORA_LOAD_TLUT) {
        payload = 23;
      } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
        payload = 20;
      } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
        payload = 13;
      } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
        payload = 8;
      } else {
        return false;
      }
      out.push_back({opcode, subCmd});
      pos += 3 + payload;
    } else {
      return false;
    }
  }
  return pos == bytes.size();
}

bool contains(const std::vector<PreambleCommand>& commands, u8 opcode, u32 reg) {
  return std::ranges::any_of(commands, [&](const PreambleCommand& c) { return c.opcode == opcode && c.reg == reg; });
}

const void* fake_pointer(uintptr_t address) { return reinterpret_cast<const void*>(address); }

void load_identity_like(f32 (&mtx)[3][4], f32 seed) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      mtx[r][c] = seed + static_cast<f32>(r * 4 + c) * 0.25f;
    }
  }
}

void build_full_state() {
  GXSetNumTevStages(3);
  GXSetNumTexGens(2);
  GXSetCullMode(GX_CULL_FRONT);

  GXSetTevColor(GX_TEVREG0, {10, 20, 30, 40});
  GXSetTevKColor(GX_KCOLOR1, {50, 60, 70, 80});
  GXSetTevColorS10(GX_TEVREG1, {-200, 300, -1024, 1023});
  GXSetTevKColor(GX_KCOLOR2, {1, 2, 3, 4});
  GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
  GXSetTevOrder(GX_TEVSTAGE1, GX_TEXCOORD1, GX_TEXMAP1, GX_COLOR1A1);
  GXSetTevOp(GX_TEVSTAGE0, GX_MODULATE);
  GXSetTevOp(GX_TEVSTAGE1, GX_DECAL);
  GXSetTevSwapModeTable(GX_TEV_SWAP1, GX_CH_GREEN, GX_CH_BLUE, GX_CH_RED, GX_CH_ALPHA);
  GXSetTevSwapMode(GX_TEVSTAGE1, GX_TEV_SWAP1, GX_TEV_SWAP2);

  GXSetChanCtrl(GX_COLOR0A0, GX_ENABLE, GX_SRC_REG, GX_SRC_VTX, GX_LIGHT0 | GX_LIGHT3, GX_DF_CLAMP, GX_AF_SPOT);
  GXSetChanAmbColor(GX_COLOR0A0, {11, 22, 33, 44});
  GXSetChanMatColor(GX_COLOR1A1, {55, 66, 77, 88});

  f32 pos[3][4];
  load_identity_like(pos, 1.f);
  GXLoadPosMtxImm(pos, GX_PNMTX3);
  f32 nrm[3][4];
  load_identity_like(nrm, -2.f);
  GXLoadNrmMtxImm(nrm, GX_PNMTX3);
  f32 tex[3][4];
  load_identity_like(tex, 3.f);
  GXLoadTexMtxImm(tex, GX_TEXMTX1, GX_MTX3x4);
  f32 post[3][4];
  load_identity_like(post, 0.5f);
  GXLoadTexMtxImm(post, GX_PTTEXMTX2, GX_MTX3x4);
  GXSetTexCoordGen2(GX_TEXCOORD1, GX_TG_MTX3x4, GX_TG_POS, GX_TEXMTX1, GX_TRUE, GX_PTTEXMTX2);
  GXSetCurrentMtx(GX_PNMTX3);

  GXLightObj light0;
  GXInitLightColor(&light0, {255, 128, 64, 32});
  GXInitLightPos(&light0, 1.f, 2.f, 3.f);
  GXInitLightDir(&light0, 0.f, -1.f, 0.f);
  GXInitLightAttn(&light0, 1.f, 0.5f, 0.25f, 2.f, 0.1f, 0.01f);
  GXLoadLightObjImm(&light0, GX_LIGHT0);
  GXLightObj light3;
  GXInitLightColor(&light3, {1, 2, 3, 4});
  GXInitLightPos(&light3, -5.f, 6.f, -7.f);
  GXInitLightAttn(&light3, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f);
  GXLoadLightObjImm(&light3, GX_LIGHT3);

  GXSetFog(GX_FOG_PERSP_LIN, 40.f, 160.f, 0.5f, 400.f, {112, 160, 208, 255});
  GXFogAdjTable table{};
  for (u16 i = 0; i < 10; ++i) {
    table.r[i] = static_cast<u16>(0x100 + i * 37);
  }
  GXSetFogRangeAdj(GX_TRUE, 320, &table);

  GXSetNumIndStages(1);
  GXSetIndTexOrder(GX_INDTEXSTAGE0, GX_TEXCOORD1, GX_TEXMAP2);
  GXSetIndTexCoordScale(GX_INDTEXSTAGE0, GX_ITS_2, GX_ITS_4);
  f32 ind[2][3] = {{0.5f, -0.25f, 0.125f}, {-0.5f, 0.75f, 0.f}};
  GXSetIndTexMtx(GX_ITM_0, ind, 3);
  GXSetTevIndirect(GX_TEVSTAGE1, GX_INDTEXSTAGE0, GX_ITF_8, GX_ITB_STU, GX_ITM_0, GX_ITW_16, GX_ITW_OFF, GX_FALSE,
                   GX_TRUE, GX_ITBA_OFF);

  GXSetAlphaCompare(GX_GREATER, 32, GX_AOP_OR, GX_LESS, 200);
  GXSetZMode(GX_TRUE, GX_GEQUAL, GX_FALSE);
  GXSetBlendMode(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_ONE, GX_LO_CLEAR);
  GXSetDstAlpha(GX_TRUE, 0x40);
  GXSetPixelFmt(GX_PF_RGBA6_Z24, GX_ZC_LINEAR);
  GXSetZCompLoc(GX_FALSE);
  GXSetLineWidth(12, GX_TO_ONE);
  GXSetPointSize(6, GX_TO_ONE);
  GXSetCopyClear({1, 2, 3, 4}, 0x123456);

  GXSetScissor(10, 20, 300, 200);
  GXSetScissorBoxOffset(4, 6);
  GXSetViewport(8.f, 16.f, 600.f, 440.f, 0.1f, 0.9f);
  f32 proj[4][4]{};
  proj[0][0] = 1.5f;
  proj[0][2] = 0.1f;
  proj[1][1] = 2.f;
  proj[1][2] = -0.2f;
  proj[2][2] = -1.001f;
  proj[2][3] = -0.5f;
  proj[3][2] = -1.f;
  GXSetProjection(proj, GX_PERSPECTIVE);
  GXSetViewportRender(4.f, 8.f, 1200.f, 880.f, 0.f, 1.f);

  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX16);
  GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
  GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT2, GX_VA_POS, GX_POS_XYZ, GX_S16, 7);
  GXSetVtxAttrFmt(GX_VTXFMT2, GX_VA_TEX5, GX_TEX_ST, GX_U8, 3);
  GXSetArray(GX_VA_POS, fake_pointer(0x80200000), 144, 12, false);
  GXSetArray(GX_VA_CLR0, fake_pointer(0x80200100), 48, 4, true);
  GXSetArray(GX_POS_MTX_ARRAY, fake_pointer(0x80210000), 480, 48, false);

  GXTexObj tex0;
  GXInitTexObj(&tex0, fake_pointer(0x80100000), 64, 32, GX_TF_RGBA8, GX_REPEAT, GX_CLAMP, GX_TRUE);
  GXInitTexObjLOD(&tex0, GX_LIN_MIP_LIN, GX_LINEAR, 0.f, 5.f, 0.5f, GX_FALSE, GX_TRUE, GX_ANISO_4);
  GXLoadTexObj(&tex0, GX_TEXMAP0);
  GXTlutObj tlut;
  GXInitTlutObj(&tlut, fake_pointer(0x80110000), GX_TL_RGB5A3, 256);
  GXLoadTlut(&tlut, GX_TLUT2);
  GXTexObj tex1;
  GXInitTexObjCI(&tex1, fake_pointer(0x80120000), 32, 32, GX_TF_C8, GX_CLAMP, GX_REPEAT, GX_FALSE, GX_TLUT2);
  GXLoadTexObj(&tex1, GX_TEXMAP1);

  GXSetTexCopySrc(0, 0, 320, 240);
  GXSetTexCopyDst(160, 120, GX_TF_RGBA8, GX_FALSE);
  GXCopyTex(const_cast<void*>(fake_pointer(0x80300000)), GX_FALSE);
  GX2SetPolygonOffset(1.f, 2.f, -1.f, -2.f, 0.5f);
  __GXSetDirtyState();
  // Written after the last genmode flush, so numChans comes from XF and the genmode valid bit is cleared
  GX_WRITE_XF_REG(0x09, 2);
}

std::filesystem::path temp_trace(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "aurora_trace_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / (name + ".gxtrace");
  std::filesystem::remove(path);
  return path;
}

void decode_all(const std::vector<u8>& bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    const auto result = aurora::gx::fifo::process(bytes.data() + offset, static_cast<u32>(bytes.size() - offset));
    ASSERT_GT(result.bytesProcessed, 0u);
    offset += result.bytesProcessed;
  }
}
} // namespace

TEST_F(GXFifoTest, StatePreamble_RestoresDecodedState) {
  build_full_state();
  const auto bytes = capture_fifo();
  reset_gx_state();
  decode_all(bytes);
  const GXState snapshot = g_gxState;
  ASSERT_TRUE(snapshot.xfRegValid.test(0x09));
  ASSERT_FALSE(snapshot.bpRegValid.test(0x00));
  ASSERT_TRUE(snapshot.xfRegValid.test(0x18));
  ASSERT_NE(snapshot.texCopyDest, nullptr);

  const auto preamble = aurora::trace::build_state_preamble(snapshot);
  reset_gx_state();
  decode_all(preamble);
  expect_state_eq(snapshot, g_gxState);
}

TEST_F(GXFifoTest, StatePreamble_ProjectionFullPath) {
  f32 proj[4][4]{};
  proj[0][0] = 1.f;
  proj[1][1] = 1.f;
  proj[2][2] = -1.f;
  proj[3][3] = 1.f;
  GXSetProjection(proj, GX_ORTHOGRAPHIC);
  f32 full[4][4];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      full[r][c] = static_cast<f32>(r * 4 + c) * -0.375f + 0.1f;
    }
  }
  GXSetProjectionFull(full);
  const auto bytes = capture_fifo();
  reset_gx_state();
  decode_all(bytes);
  const GXState snapshot = g_gxState;
  ASSERT_FALSE(snapshot.xfRegValid.test(0x20));

  const auto preamble = aurora::trace::build_state_preamble(snapshot);
  std::vector<PreambleCommand> commands;
  ASSERT_TRUE(parse_preamble(preamble, commands));
  EXPECT_TRUE(contains(commands, GX_AURORA, GX_AURORA_LOAD_PROJECTION_FULL));
  EXPECT_FALSE(contains(commands, GX_LOAD_XF_REG, 0x1020));
  reset_gx_state();
  decode_all(preamble);
  EXPECT_TRUE(same_bits(snapshot.proj, g_gxState.proj));
  expect_state_eq(snapshot, g_gxState);
}

TEST_F(GXFifoTest, StatePreamble_IsIdempotentAndSideEffectFree) {
  build_full_state();
  const auto bytes = capture_fifo();
  reset_gx_state();
  decode_all(bytes);
  const auto preamble = aurora::trace::build_state_preamble(g_gxState);
  std::vector<PreambleCommand> commands;
  ASSERT_TRUE(parse_preamble(preamble, commands));
  EXPECT_FALSE(contains(commands, GX_LOAD_BP_REG, 0x45));
  EXPECT_FALSE(contains(commands, GX_LOAD_BP_REG, 0x52));

  reset_gx_state();
  aurora::gfx::g_testDrawCount = 0;
  const uint32_t resolves = aurora::gfx::testing::resolvePassCount.load();
  const auto first = aurora::gx::fifo::process(preamble.data(), static_cast<u32>(preamble.size()));
  EXPECT_EQ(first.bytesProcessed, preamble.size());
  EXPECT_FALSE(first.drawDone);
  const GXState once = g_gxState;
  const auto second = aurora::gx::fifo::process(preamble.data(), static_cast<u32>(preamble.size()));
  EXPECT_EQ(second.bytesProcessed, preamble.size());
  EXPECT_FALSE(second.drawDone);
  expect_state_eq(once, g_gxState);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 0u);
  EXPECT_EQ(aurora::gfx::testing::resolvePassCount.load(), resolves);
}

TEST_F(GXFifoTest, StatePreamble_DefaultStateRoundTrips) {
  const GXState defaults{};
  const auto preamble = aurora::trace::build_state_preamble(defaults);
  std::vector<PreambleCommand> commands;
  ASSERT_TRUE(parse_preamble(preamble, commands));
  reset_gx_state();
  decode_all(preamble);
  expect_state_eq(defaults, g_gxState);
}

namespace {
std::mutex sObservedMutex;
std::vector<u8> sObservedBytes;

void append_processed(void*, const uint8_t* data, uint32_t size) noexcept {
  std::lock_guard lock{sObservedMutex};
  sObservedBytes.insert(sObservedBytes.end(), data, data + size);
}

void write_bytes(std::initializer_list<u8> bytes, std::vector<u8>& expected) {
  for (const u8 byte : bytes) {
    GX_WRITE_U8(byte);
  }
  expected.insert(expected.end(), bytes);
}
} // namespace

TEST_F(GXFifoTest, ProcessObserver_SeesEveryProcessedByteOnce) {
  sObservedBytes.clear();
  const aurora::gx::fifo::ProcessObserver observer{.processed = append_processed};
  std::vector<u8> expected;
  aurora::gx::fifo::init();
  aurora::gx::fifo::set_process_observer(&observer);
  aurora::gx::fifo::begin_frame();
  write_bytes({0x61, 0x41, 0x12, 0x34, 0x56}, expected);
  aurora::gx::fifo::publish();
  write_bytes({0x08, 0x50, 0x00, 0x00, 0x02, 0x00}, expected);
  aurora::gx::fifo::publish();
  write_bytes({0x61, 0x45, 0x00, 0x00, 0x02, 0x61, 0x42, 0x00, 0x00, 0x01}, expected);
  aurora::gx::fifo::publish();
  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::set_process_observer(nullptr);

  std::lock_guard lock{sObservedMutex};
  EXPECT_EQ(sObservedBytes, expected);
  EXPECT_EQ(g_gxState.bpRegCache[0x42], 0x42000001u);
}

TEST_F(GXFifoTest, SynchronousPublish_DecodesBeforeReturning) {
  aurora::gx::fifo::init();
  aurora::gx::fifo::set_synchronous_publish(true);
  aurora::gx::fifo::begin_frame();
  GX_WRITE_RAS_REG(0x41123456);
  aurora::gx::fifo::publish();
  EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
  EXPECT_TRUE(g_gxState.bpRegValid.test(0x41));
  aurora::gx::fifo::set_synchronous_publish(false);
  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
}

namespace {
struct ObservedRead {
  aurora::gx::guest_memory::Kind kind;
  uint64_t address;
  std::vector<u8> bytes;
};
std::vector<ObservedRead> sReplayReads;
aurora::trace::SnapshotStore* sReplayStore = nullptr;

void record_read(void* session, aurora::gx::guest_memory::Kind kind, uint64_t address,
                 std::span<const uint8_t> bytes) noexcept {
  static_cast<aurora::trace::CaptureSession*>(session)->observe(static_cast<uint8_t>(kind), address, bytes);
}

void record_processed(void* session, const uint8_t* data, uint32_t size) noexcept {
  static_cast<aurora::trace::CaptureSession*>(session)->processed({data, size});
}

std::span<const uint8_t> replay_resolve(void*, aurora::gx::guest_memory::Kind kind, uint64_t address,
                                        size_t size) noexcept {
  return sReplayStore->hooks()->resolve(sReplayStore->hooks()->userData, kind, address, size);
}

void replay_observe(void*, aurora::gx::guest_memory::Kind kind, uint64_t address,
                    std::span<const uint8_t> bytes) noexcept {
  sReplayReads.push_back({kind, address, {bytes.begin(), bytes.end()}});
}

std::array<u8, 48> big_endian_matrix(const std::array<f32, 12>& values) {
  std::array<u8, 48> bytes{};
  for (size_t i = 0; i < values.size(); ++i) {
    const u32 bits = std::bit_cast<u32>(values[i]);
    bytes[i * 4] = static_cast<u8>(bits >> 24);
    bytes[i * 4 + 1] = static_cast<u8>(bits >> 16);
    bytes[i * 4 + 2] = static_cast<u8>(bits >> 8);
    bytes[i * 4 + 3] = static_cast<u8>(bits);
  }
  return bytes;
}

void draw_indexed_triangle() {
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  GXPosition1x8(0);
  GXPosition1x8(1);
  GXPosition1x8(2);
  GXEnd();
}
} // namespace

TEST_F(GXFifoTest, CapturedStreamReplaysAgainstZeroedMemory) {
  const std::array<f32, 12> matrixValues{1.f, 2.f, 3.f, 10.f, 4.f, 5.f, 6.f, 20.f, 7.f, 8.f, 9.f, 30.f};
  auto matrix = big_endian_matrix(matrixValues);
  std::array<f32, 9> positions{0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
  std::array<f32, 9> mutated{2.f, 2.f, 2.f, 3.f, 2.f, 2.f, 2.f, 3.f, 2.f};
  const auto path = temp_trace("captured_stream");
  const auto as_bytes = [](const auto& array) {
    const auto* data = reinterpret_cast<const u8*>(array.data());
    return std::vector<u8>(data, data + sizeof(array));
  };
  const auto v1 = as_bytes(positions);
  const auto v2 = as_bytes(mutated);

  aurora::trace::CaptureSession session;
  std::string error;
  ASSERT_TRUE(session.open(path, 0, 0, error)) << error;
  ASSERT_TRUE(session.write_header_chunks({}, {}, error)) << error;
  const aurora::gx::guest_memory::Hooks recordHooks{.observe = record_read, .userData = &session};
  const aurora::gx::fifo::ProcessObserver recordObserver{.processed = record_processed, .userData = &session};

  aurora::gx::fifo::init();
  aurora::gx::guest_memory::set_hooks(&recordHooks);
  aurora::gx::fifo::set_process_observer(&recordObserver);
  aurora::gx::fifo::set_synchronous_publish(true);
  aurora::gx::fifo::begin_frame();
  session.begin_frame(0, {.efbWidth = 640, .efbHeight = 480});

  GXSetArray(GX_POS_MTX_ARRAY, matrix.data(), matrix.size(), matrix.size(), false);
  GXLoadPosMtxIndx(0, GX_PNMTX0);
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_INDEX8);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetArray(GX_VA_POS, positions.data(), sizeof(positions), 12, false);
  draw_indexed_triangle();
  positions = mutated;
  GXInvalidateVtxCache();
  draw_indexed_triangle();

  aurora::gx::fifo::drain();
  aurora::gx::fifo::end_frame();
  aurora::gx::fifo::set_synchronous_publish(false);
  aurora::gx::fifo::set_process_observer(nullptr);
  aurora::gx::guest_memory::set_hooks(nullptr);
  ASSERT_TRUE(session.end_frame(error)) << error;
  ASSERT_TRUE(session.finalize(0, {}, error)) << error;

  aurora::trace::TraceReader reader;
  ASSERT_TRUE(reader.open(path, error)) << error;
  aurora::trace::SnapshotStore store;
  const aurora::trace::BlobSink sink = [&](const aurora::trace::Hash128& hash, std::span<const uint8_t> bytes,
                                           std::string& sinkError) { return store.add_blob(hash, bytes, sinkError); };
  aurora::trace::FrameView frame;
  ASSERT_EQ(reader.next(frame, sink, error), aurora::trace::TraceReader::Status::Frame) << error;

  std::vector<aurora::trace::RecordType> types;
  std::vector<aurora::trace::MemoryMapRecord> maps;
  for (const auto& record : frame.records) {
    types.push_back(record.type);
    if (record.type == aurora::trace::RecordType::MemoryMap) {
      aurora::trace::MemoryMapRecord map;
      ASSERT_TRUE(aurora::trace::decode_memory_map(record.payload, map));
      maps.push_back(map);
    }
  }
  using aurora::trace::RecordType;
  EXPECT_EQ(types, (std::vector{RecordType::MemoryMap, RecordType::MemoryMap, RecordType::Fifo, RecordType::MemoryMap,
                                RecordType::Fifo}));
  ASSERT_EQ(maps.size(), 3u);
  EXPECT_EQ(maps[0].kind, static_cast<uint8_t>(aurora::gx::guest_memory::Kind::IndexedLoad));
  EXPECT_EQ(maps[0].address, reinterpret_cast<uintptr_t>(matrix.data()));
  EXPECT_EQ(maps[1].kind, static_cast<uint8_t>(aurora::gx::guest_memory::Kind::VertexArray));
  EXPECT_EQ(maps[1].blob, aurora::trace::hash_bytes(v1));
  EXPECT_EQ(maps[2].address, reinterpret_cast<uintptr_t>(positions.data()));
  EXPECT_EQ(maps[2].blob, aurora::trace::hash_bytes(v2));

  matrix.fill(0);
  positions.fill(0.f);
  reset_gx_state();
  sReplayReads.clear();
  sReplayStore = &store;
  const aurora::gx::guest_memory::Hooks replayHooks{.resolve = replay_resolve, .observe = replay_observe};
  aurora::gx::guest_memory::set_hooks(&replayHooks);
  for (const auto& record : frame.records) {
    if (record.type == aurora::trace::RecordType::MemoryMap) {
      aurora::trace::MemoryMapRecord map;
      ASSERT_TRUE(aurora::trace::decode_memory_map(record.payload, map));
      ASSERT_TRUE(store.apply(map, error)) << error;
    } else if (record.type == aurora::trace::RecordType::Fifo) {
      store.begin_record();
      decode_all({record.payload.begin(), record.payload.end()});
    }
  }
  aurora::gx::guest_memory::set_hooks(nullptr);
  sReplayStore = nullptr;

  const auto* decoded = reinterpret_cast<const f32*>(&g_gxState.pnMtx[0].pos);
  for (size_t i = 0; i < matrixValues.size(); ++i) {
    EXPECT_EQ(decoded[i], matrixValues[i]) << i;
  }
  std::vector<std::vector<u8>> vertexReads;
  for (const auto& read : sReplayReads) {
    if (read.kind == aurora::gx::guest_memory::Kind::VertexArray) {
      vertexReads.push_back(read.bytes);
    }
  }
  ASSERT_EQ(vertexReads.size(), 2u);
  EXPECT_EQ(vertexReads[0], v1);
  EXPECT_EQ(vertexReads[1], v2);
  EXPECT_EQ(store.misses(), 0u);
}

namespace {
struct DecodedFrame {
  GXState state;
  uint32_t draws = 0;
  uint32_t resolves = 0;
  std::vector<u8> colors;
};

std::vector<DecodedFrame> decode_trace(const std::filesystem::path& path, uint64_t& misses) {
  std::vector<DecodedFrame> frames;
  aurora::trace::TraceReader reader;
  std::string error;
  EXPECT_TRUE(reader.open(path, error)) << error;
  aurora::trace::SnapshotStore store;
  const aurora::trace::BlobSink sink = [&](const aurora::trace::Hash128& hash, std::span<const uint8_t> bytes,
                                           std::string& sinkError) { return store.add_blob(hash, bytes, sinkError); };
  g_gxState = GXState{};
  sReplayStore = &store;
  const aurora::gx::guest_memory::Hooks hooks{.resolve = replay_resolve, .observe = replay_observe};
  aurora::gx::guest_memory::set_hooks(&hooks);
  aurora::trace::FrameView frame;
  while (reader.next(frame, sink, error) == aurora::trace::TraceReader::Status::Frame) {
    sReplayReads.clear();
    const uint32_t draws = aurora::gfx::g_testDrawCount;
    const uint32_t resolves = aurora::gfx::testing::resolvePassCount.load();
    for (const auto& record : frame.records) {
      if (record.type == aurora::trace::RecordType::MemoryMap) {
        aurora::trace::MemoryMapRecord map;
        EXPECT_TRUE(aurora::trace::decode_memory_map(record.payload, map));
        EXPECT_TRUE(store.apply(map, error)) << error;
      } else if (record.type == aurora::trace::RecordType::Fifo) {
        store.begin_record();
        decode_all({record.payload.begin(), record.payload.end()});
      }
    }
    DecodedFrame decoded{
        g_gxState, aurora::gfx::g_testDrawCount - draws, aurora::gfx::testing::resolvePassCount.load() - resolves, {}};
    for (const auto& read : sReplayReads) {
      if (read.address == aurora::gxreplay::SelfTestColorAddress) {
        decoded.colors = read.bytes;
        break;
      }
    }
    frames.push_back(std::move(decoded));
  }
  EXPECT_TRUE(error.empty()) << error;
  aurora::gx::guest_memory::set_hooks(nullptr);
  sReplayStore = nullptr;
  misses = store.misses();
  return frames;
}
} // namespace

TEST_F(GXFifoTest, SelfTestTrace_HintsDoNotChangeDecodedState) {
  const auto hintsPath = temp_trace("self_test_hints");
  const auto plainPath = temp_trace("self_test_plain");
  std::string error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(hintsPath, true, error)) << error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(plainPath, false, error)) << error;

  uint64_t hintMisses = 0;
  uint64_t plainMisses = 0;
  const auto hints = decode_trace(hintsPath, hintMisses);
  const auto plain = decode_trace(plainPath, plainMisses);
  EXPECT_EQ(hintMisses, 0u);
  EXPECT_EQ(plainMisses, 0u);
  ASSERT_EQ(hints.size(), aurora::gxreplay::SelfTestFrameCount);
  ASSERT_EQ(plain.size(), aurora::gxreplay::SelfTestFrameCount);
  for (size_t i = 0; i < hints.size(); ++i) {
    SCOPED_TRACE(i);
    expect_state_eq(hints[i].state, plain[i].state);
    EXPECT_EQ(hints[i].draws, plain[i].draws);
    EXPECT_GT(hints[i].draws, 0u);
    EXPECT_EQ(hints[i].resolves, plain[i].resolves);
    EXPECT_EQ(hints[i].resolves, 1u);
    EXPECT_EQ(hints[i].colors, plain[i].colors);
    EXPECT_EQ(hints[i].colors.size(), 48u);
  }
  EXPECT_EQ(hints[0].colors, hints[1].colors);
  EXPECT_NE(hints[2].colors, hints[1].colors);
}

TEST_F(GXFifoTest, SelfTestTrace_DecodedHintsDoNotChangeDecodedState) {
  const auto hintsPath = temp_trace("self_test_decoded_hints");
  const auto plainPath = temp_trace("self_test_decoded_plain");
  std::string error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(hintsPath, true, error)) << error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(plainPath, false, error)) << error;

  HintRecorder recorder;
  const auto sink = recorder.sink();
  aurora::gx::hints::set_sink(&sink);
  uint64_t hintMisses = 0;
  const auto hints = decode_trace(hintsPath, hintMisses);
  aurora::gx::hints::set_sink(nullptr);
  uint64_t plainMisses = 0;
  const auto plain = decode_trace(plainPath, plainMisses);
  EXPECT_FALSE(recorder.events.empty());
  EXPECT_EQ(hintMisses, 0u);
  ASSERT_EQ(hints.size(), aurora::gxreplay::SelfTestFrameCount);
  ASSERT_EQ(plain.size(), aurora::gxreplay::SelfTestFrameCount);
  for (size_t i = 0; i < hints.size(); ++i) {
    SCOPED_TRACE(i);
    expect_state_eq(hints[i].state, plain[i].state);
    EXPECT_EQ(hints[i].draws, plain[i].draws);
    EXPECT_EQ(hints[i].resolves, plain[i].resolves);
    EXPECT_EQ(hints[i].colors, plain[i].colors);
  }
  aurora::gx::hints::testing::reset_counters();
}

TEST_F(GXFifoTest, SelfTestTrace_ValidHintsDecode) {
  const auto path = temp_trace("self_test_valid_hints");
  std::string error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(path, true, error)) << error;

  HintRecorder recorder;
  const auto sink = recorder.sink();
  aurora::gx::hints::set_sink(&sink);
  aurora::gx::hints::testing::reset_counters();
  const uint32_t drawsBefore = aurora::gfx::g_testDrawCount;
  uint64_t misses = 0;
  const auto frames = decode_trace(path, misses);
  aurora::gx::hints::set_sink(nullptr);
  ASSERT_EQ(frames.size(), aurora::gxreplay::SelfTestFrameCount);

  constexpr size_t Frames = aurora::gxreplay::SelfTestFrameCount;
  const std::array<std::pair<u16, size_t>, 14> perFrame{{
      {GX_AURORA_RENDER_CAMERA, 1},
      {GX_AURORA_RENDER_OBJECT_BEGIN, 3},
      {GX_AURORA_RENDER_OBJECT_END, 3},
      {GX_AURORA_RENDER_SCOPE_PUSH, 1},
      {GX_AURORA_RENDER_SCOPE_POP, 1},
      {GX_AURORA_RENDER_STAGE, 6},
      {GX_AURORA_RENDER_LIGHTS, 1},
      {GX_AURORA_RENDER_ENVIRONMENT, 1},
      {GX_AURORA_RENDER_MATERIAL, 1},
      {GX_AURORA_RENDER_SKELETON, 1},
      {GX_AURORA_RENDER_SCENE_EPOCH, 1},
      {GX_AURORA_RENDER_TEV_POLICY, 1},
      {GX_AURORA_RENDER_INFLUENCERS, 1},
      {GX_AURORA_RENDER_WIND_SOURCES, 1},
  }};
  size_t total = 0;
  for (const auto& [id, count] : perFrame) {
    EXPECT_EQ(recorder.count(id), count * Frames) << "hint 0x" << std::hex << id;
    EXPECT_EQ(aurora::gx::hints::counters(id).decoded, count * Frames) << "hint 0x" << std::hex << id;
    total += count * Frames;
  }
  EXPECT_EQ(recorder.events.size(), total);
  // The self-test's garbage payloads under CAMERA are rejected, and every skip-only ID is skipped
  EXPECT_EQ(aurora::gx::hints::counters(GX_AURORA_RENDER_CAMERA).malformed, 4 * Frames);
  for (const u16 id :
       {u16{GX_AURORA_RENDER_GRASS_PATCHES}, u16{GX_AURORA_RENDER_SKY}, u16{GX_AURORA_RENDER_FOG_VOLUMES}}) {
    EXPECT_EQ(aurora::gx::hints::counters(id).skipped, Frames) << "hint 0x" << std::hex << id;
  }
  EXPECT_EQ(aurora::gx::hints::counters(GX_AURORA_SET_ARRAY_IMMUTABLE).skipped, 5 * Frames);

  ASSERT_EQ(recorder.objects.size(), 3 * Frames);
  ASSERT_EQ(recorder.stages.size(), 6 * Frames);
  std::vector<HintRecorder::Event> stageEvents;
  std::vector<HintRecorder::Event> endEvents;
  for (const auto& event : recorder.events) {
    if (event.id == GX_AURORA_RENDER_STAGE) {
      stageEvents.push_back(event);
    } else if (event.id == GX_AURORA_RENDER_OBJECT_END) {
      endEvents.push_back(event);
    }
  }
  for (size_t frame = 0; frame < Frames; ++frame) {
    SCOPED_TRACE(frame);
    for (size_t quad = 0; quad < 3; ++quad) {
      const auto& object = recorder.objects[frame * 3 + quad];
      EXPECT_EQ(object.id, aurora::gxreplay::SelfTestObjects[quad].id);
      EXPECT_EQ(object.categories, aurora::gxreplay::SelfTestObjects[quad].categories);
      EXPECT_EQ(object.generation, 1u);
    }
    const uint32_t frameStart = stageEvents[frame * 6].draws;
    EXPECT_EQ(frameStart, drawsBefore + frame * frames[0].draws);
    EXPECT_EQ(recorder.stages[frame * 6], aurora::render::Stage::SceneBegin);
    const uint32_t afterQuad3 = endEvents[frame * 3 + 2].draws;
    EXPECT_EQ(afterQuad3, frameStart + frames[frame].draws);
    for (size_t marker = 1; marker < 6; ++marker) {
      EXPECT_EQ(recorder.stages[frame * 6 + marker], static_cast<aurora::render::Stage>(marker));
      EXPECT_EQ(stageEvents[frame * 6 + marker].draws, afterQuad3);
    }
  }
  ASSERT_EQ(recorder.cameras.size(), Frames);
  EXPECT_TRUE(aurora::gx::hints::main_camera_usable(recorder.cameras[0].gxProj));
  ASSERT_EQ(recorder.scopes.size(), Frames);
  EXPECT_EQ(recorder.scopes[0].categories, AURORA_CAT_EMISSIVE);
  EXPECT_EQ(recorder.scopes[0].role, AURORA_CAMERA_INHERIT);
  aurora::gx::hints::testing::reset_counters();
}

TEST_F(GXFifoTest, SelfTestTrace_ContainsEveryHintIdAndSize) {
  const auto path = temp_trace("self_test_scan");
  std::string error;
  ASSERT_TRUE(aurora::gxreplay::write_self_test_trace(path, true, error)) << error;

  aurora::trace::TraceReader reader;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_NE(reader.header().flags & aurora::trace::FileSynthetic, 0u);
  aurora::trace::FrameView frame;
  uint32_t frames = 0;
  while (reader.next(frame, {}, error) == aurora::trace::TraceReader::Status::Frame) {
    std::vector<u8> fifo;
    for (const auto& record : frame.records) {
      if (record.type == aurora::trace::RecordType::Fifo) {
        fifo.insert(fifo.end(), record.payload.begin(), record.payload.end());
      }
    }
    for (const uint16_t id : aurora::gxreplay::SelfTestHintIds) {
      for (const uint32_t size : aurora::gxreplay::SelfTestPayloadSizes) {
        std::vector<u8> hint{GX_AURORA,
                             static_cast<u8>(id >> 8),
                             static_cast<u8>(id),
                             static_cast<u8>(size >> 24),
                             static_cast<u8>(size >> 16),
                             static_cast<u8>(size >> 8),
                             static_cast<u8>(size)};
        for (uint32_t i = 0; i < size; ++i) {
          hint.push_back(aurora::gxreplay::self_test_hint_byte(i));
        }
        uint32_t matches = 0;
        for (auto it = std::search(fifo.begin(), fifo.end(), hint.begin(), hint.end()); it != fifo.end();
             it = std::search(it + 1, fifo.end(), hint.begin(), hint.end())) {
          ++matches;
        }
        EXPECT_EQ(matches, 1u) << "frame " << frames << " hint 0x" << std::hex << id << std::dec << " size " << size;
      }
    }
    ++frames;
  }
  EXPECT_TRUE(error.empty()) << error;
  EXPECT_EQ(frames, aurora::gxreplay::SelfTestFrameCount);
}

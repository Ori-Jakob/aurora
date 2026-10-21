// Draw routing: suppressed draws, merge breaks and push-time substitution against emulated merging

#include "gx_test_common.hpp"
#include "gx/draw_route.hpp"
#include "gx/pipeline.hpp"

#include <array>
#include <vector>

namespace aurora::gx {
extern uint32_t g_testPipelineBuildCount;
extern uint32_t g_testShaderInfoCount;
extern uint32_t g_testUniformBuildCount;
extern u32 g_testLastUniformInfoSize;
extern GXCompare g_testPopulateDepthFunc;
extern u32 g_testPopulateTevStageCount;
} // namespace aurora::gx

namespace aurora::gfx {
extern gx::DrawData g_testLastDraw;
extern bool g_testMergeDraws;
extern bool g_testNormalAttachment;
extern uint32_t g_testPipelineRefCount;
extern PipelineRef g_testLastPipelineRef;
extern uint32_t g_testMergedDrawCount;
extern uint64_t g_testPushedVertexBytes;
extern uint64_t g_testPushedIndexBytes;
namespace testing {
void reset_stub_ranges() noexcept;
} // namespace testing
} // namespace aurora::gfx

namespace fifo = aurora::gx::fifo;
namespace gfx = aurora::gfx;
namespace hints = aurora::gx::hints;
namespace route = aurora::gx::draw_route;
using aurora::gx::g_gxState;
using aurora::gx::g_testLastUniformInfoSize;
using aurora::gx::g_testPipelineBuildCount;
using aurora::gx::g_testPopulateDepthFunc;
using aurora::gx::g_testPopulateTevStageCount;
using aurora::gx::g_testShaderInfoCount;
using aurora::gx::g_testUniformBuildCount;

namespace {
// Direct F32 position and RGBA8 colour
constexpr u32 kVtxSize = 16;
constexpr std::array<u8, 5> kBpWrite{GX_LOAD_BP_REG, 0x41, 0x12, 0x34, 0x56};

void set_vertex_format(GXVtxFmt fmt) {
  GXSetVtxAttrFmt(fmt, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(fmt, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
}

void vertices(u16 count) {
  for (u16 i = 0; i < count; ++i) {
    GXPosition3f32(static_cast<f32>(i), static_cast<f32>(i & 1), 0.f);
    GXColor4u8(255, 128, 64, 255);
  }
}

void draw(GXPrimitive prim, u16 count, GXVtxFmt fmt = GX_VTXFMT0) {
  GXBegin(prim, fmt, count);
  vertices(count);
  GXEnd();
}

void triangle() { draw(GX_TRIANGLES, 3); }

void sized_draw(GXPrimitive prim, u16 count) {
  GXBegin(prim, GX_VTXFMT0, GX_AUTO);
  vertices(count);
  GXEnd();
}

void indexed_triangle() {
  const u16 indices[] = {0, 1, 2};
  GXBeginIndexed(GX_VTXFMT0, 3, indices, 3);
  vertices(3);
  GXEnd();
}

void push_capture_only_scope() { GXAuroraPushScope(0, AURORA_OBJ_CAPTURE_ONLY, AURORA_CAMERA_INHERIT); }
} // namespace

class GXDrawRouteTest : public GXFifoTest {
protected:
  enum class Diversion : uint8_t { Scope, Object };

  void SetUp() override {
    GXFifoTest::SetUp();
    hints::testing::reset_counters();
    fifo::testing::reset_draw_route_stats();
    // The draw cache outlives a test, so nothing an earlier test pushed may become a merge target here
    fifo::invalidate_draw_cache();
    gfx::testing::reset_stub_ranges();
    gfx::g_testMergeDraws = true;
    sink = recorder.sink();
    hints::set_sink(&sink);

    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxDesc(GX_VA_CLR0, GX_DIRECT);
    set_vertex_format(GX_VTXFMT0);
    GXSetTevOp(GX_TEVSTAGE0, GX_PASSCLR);
    decode_fifo(flush_and_capture());

    gfx::g_testDrawCount = 0;
    gfx::g_testMergedDrawCount = 0;
    gfx::g_testPushedVertexBytes = 0;
    gfx::g_testPushedIndexBytes = 0;
    gfx::g_testPipelineRefCount = 0;
    gfx::g_testLastPipelineRef = 0;
    g_testPipelineBuildCount = 0;
    g_testShaderInfoCount = 0;
    g_testUniformBuildCount = 0;
    g_testLastUniformInfoSize = 0;
  }

  void TearDown() override {
    route::set_router(nullptr);
    hints::set_sink(nullptr);
    hints::testing::reset_counters();
    gfx::g_testMergeDraws = false;
    gfx::g_testNormalAttachment = false;
    gfx::set_blocking_pipelines(false);
    g_testPopulateDepthFunc = GX_NEVER;
    g_testPopulateTevStageCount = 0;
    GXFifoTest::TearDown();
  }

  // The first draw looks the pipeline up; the second rebuilds after the change and must look up again
  void expect_rebuild_looks_up(void (*change)()) {
    gfx::g_testMergeDraws = false;
    triangle();
    decode();
    change();
    triangle();
    decode();
    EXPECT_EQ(g_testPipelineBuildCount, 2u);
    EXPECT_EQ(gfx::g_testPipelineRefCount, 2u);
    EXPECT_EQ(g_testShaderInfoCount, 2u);
    EXPECT_EQ(gfx::g_testLastDraw.pipeline, gfx::g_testLastPipelineRef);
    EXPECT_EQ(g_testLastUniformInfoSize, 2u);
    EXPECT_EQ(g_gxState.dirty, 0u);
  }

  void install_router(bool withPush = false) {
    router = {.route = &on_route, .push = withPush ? &on_push : nullptr, .userData = this};
    route::set_router(&router);
  }

  void decode() { decode_fifo(capture_fifo()); }

  // A BP write after the stream lands only if every diverted draw consumed exactly its own bytes
  void decode_then_bp() {
    auto bytes = capture_fifo();
    bytes.insert(bytes.end(), kBpWrite.begin(), kBpWrite.end());
    decode_fifo(bytes);
    EXPECT_EQ(g_gxState.bpRegCache[0x41], 0x41123456u);
  }

  // A, then B1 and B2 inside a scope or object carrying the flags, then C1 and C2; all five are clean
  void emit_diverted_pair(Diversion diversion, uint32_t flags = AURORA_OBJ_CAPTURE_ONLY) {
    triangle();
    if (diversion == Diversion::Scope) {
      GXAuroraPushScope(0, flags, AURORA_CAMERA_INHERIT);
    } else {
      GXAuroraBeginObject(0x20, 0, 1, AURORA_CAT_DYNAMIC, flags);
    }
    triangle();
    triangle();
    if (diversion == Diversion::Scope) {
      GXAuroraPopScope();
    } else {
      GXAuroraEndObject();
    }
    triangle();
    triangle();
  }

  void expect_diverted_pair_split() {
    // A alone, then C1 with C2 merged into it; the suppressed vertices never reach the buffer
    EXPECT_EQ(gfx::g_testDrawCount, 2u);
    EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
    EXPECT_EQ(gfx::g_testLastDraw.vtxCount, 6u);
    EXPECT_EQ(gfx::g_testLastDraw.vertRange.offset, 3 * kVtxSize);
    EXPECT_EQ(gfx::g_testLastDraw.vertRange.size, 6 * kVtxSize);
    EXPECT_EQ(gfx::g_testPushedVertexBytes, 9 * kVtxSize);
    const fifo::DrawRouteStats stats = fifo::draw_route_stats();
    EXPECT_EQ(stats.routed, 5u);
    EXPECT_EQ(stats.suppressed, 2u);
    EXPECT_EQ(stats.mergeBreaks, 0u);
    EXPECT_EQ(stats.substituted, 0u);
    EXPECT_EQ(g_gxState.dirty, 0u);
  }

  // Replays the hints decoded so far, which are exactly the ones that precede the draw being routed
  [[nodiscard]] bool capture_only() const {
    std::vector<uint32_t> scopeFlags;
    uint32_t objectFlags = 0;
    size_t nextScope = 0;
    size_t nextObject = 0;
    for (const auto& event : recorder.events) {
      switch (event.id) {
      case GX_AURORA_RENDER_SCOPE_PUSH:
        scopeFlags.push_back(recorder.scopes[nextScope++].flags);
        break;
      case GX_AURORA_RENDER_SCOPE_POP:
        if (!scopeFlags.empty()) {
          scopeFlags.pop_back();
        }
        break;
      case GX_AURORA_RENDER_OBJECT_BEGIN:
        objectFlags = recorder.objects[nextObject++].flags;
        break;
      case GX_AURORA_RENDER_OBJECT_END:
        objectFlags = 0;
        break;
      default:
        break;
      }
    }
    uint32_t flags = objectFlags;
    for (const uint32_t scope : scopeFlags) {
      flags |= scope;
    }
    return (flags & AURORA_OBJ_CAPTURE_ONLY) != 0;
  }

  static route::Decision on_route(void* userData, const route::Query& query) noexcept {
    auto& test = *static_cast<GXDrawRouteTest*>(userData);
    test.queries.push_back(query);
    if (test.capture_only()) {
      return {.route = route::Route::Suppress};
    }
    return {.route = route::Route::Push, .breakMerge = test.queries.size() == test.breakAt};
  }

  static bool on_push(void* userData, aurora::gx::DrawData&, const aurora::gx::PipelineConfig&,
                      const aurora::gx::ShaderInfo&) noexcept {
    auto& test = *static_cast<GXDrawRouteTest*>(userData);
    return ++test.pushes == test.substituteAt;
  }

  HintRecorder recorder;
  hints::Sink sink;
  route::Router router;
  std::vector<route::Query> queries;
  // 1-based query and push numbers; 0 never matches
  size_t breakAt = 0;
  size_t substituteAt = 0;
  size_t pushes = 0;
};

TEST_F(GXDrawRouteTest, WithoutRouterCleanDrawsMerge) {
  // The hazard a router has to close: the capture-only draws join the visible chain like any other clean draw
  emit_diverted_pair(Diversion::Scope);
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 4u);
  EXPECT_EQ(gfx::g_testLastDraw.vtxCount, 15u);
  EXPECT_EQ(fifo::draw_route_stats().routed, 0u);
}

TEST_F(GXDrawRouteTest, PushOnlyRouterMergesLikeNoRouter) {
  install_router();
  emit_diverted_pair(Diversion::Scope, 0);
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 4u);
  EXPECT_EQ(gfx::g_testLastDraw.vtxCount, 15u);
  EXPECT_EQ(gfx::g_testPushedVertexBytes, 15 * kVtxSize);
  const fifo::DrawRouteStats stats = fifo::draw_route_stats();
  EXPECT_EQ(stats.routed, 5u);
  EXPECT_EQ(stats.suppressed, 0u);
  EXPECT_EQ(stats.mergeBreaks, 0u);
}

TEST_F(GXDrawRouteTest, CaptureOnlyScopeResetsMerge) {
  install_router();
  emit_diverted_pair(Diversion::Scope);
  decode();
  expect_diverted_pair_split();
}

TEST_F(GXDrawRouteTest, CaptureOnlyObjectResetsMerge) {
  install_router();
  emit_diverted_pair(Diversion::Object);
  decode();
  expect_diverted_pair_split();
}

TEST_F(GXDrawRouteTest, SuppressedStateChangeStaysDirty) {
  install_router();
  triangle();
  push_capture_only_scope();
  GXSetTevOp(GX_TEVSTAGE0, GX_REPLACE);
  triangle();
  GXAuroraPopScope();
  triangle();
  decode();
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 1u);
  // The suppressed draw never consumed the TEV change, so the next draw still rebuilds for it
  EXPECT_EQ(g_testPipelineBuildCount, 2u);
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 0u);
  EXPECT_EQ(g_gxState.dirty, 0u);
}

TEST_F(GXDrawRouteTest, BreakMergeSplitsCleanDraws) {
  breakAt = 2;
  install_router();
  triangle();
  triangle();
  triangle();
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
  EXPECT_EQ(gfx::g_testLastDraw.vtxCount, 6u);
  const fifo::DrawRouteStats stats = fifo::draw_route_stats();
  EXPECT_EQ(stats.routed, 3u);
  EXPECT_EQ(stats.mergeBreaks, 1u);
  EXPECT_EQ(stats.suppressed, 0u);
}

TEST_F(GXDrawRouteTest, BreakMergeWithoutMergeTargetNotCounted) {
  breakAt = 2;
  install_router();
  triangle();
  GXSetTevOp(GX_TEVSTAGE0, GX_REPLACE);
  triangle();
  decode();
  // The state change already ends the chain, so the break prevents nothing
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(fifo::draw_route_stats().mergeBreaks, 0u);

  breakAt = 3;
  draw(GX_LINES, 2);
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 3u);
  EXPECT_EQ(fifo::draw_route_stats().mergeBreaks, 0u);
}

TEST_F(GXDrawRouteTest, BreakMergeOnIndexedDrawNotCounted) {
  breakAt = 2;
  install_router();
  triangle();
  indexed_triangle();
  decode();
  ASSERT_EQ(queries.size(), 2u);
  EXPECT_TRUE(queries[1].indexed);
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(gfx::g_testLastDraw.indexCount, 3u);
  EXPECT_EQ(fifo::draw_route_stats().mergeBreaks, 0u);
}

TEST_F(GXDrawRouteTest, SuppressedIndexedDrawConsumesPayload) {
  install_router();
  push_capture_only_scope();
  indexed_triangle();
  GXAuroraPopScope();
  decode_then_bp();
  EXPECT_EQ(gfx::g_testDrawCount, 0u);
  EXPECT_EQ(gfx::g_testPushedVertexBytes, 0u);
  EXPECT_EQ(gfx::g_testPushedIndexBytes, 0u);
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 1u);
}

TEST_F(GXDrawRouteTest, SuppressedSizedDraw) {
  install_router();
  push_capture_only_scope();
  sized_draw(GX_TRIANGLES, 3);
  GXAuroraPopScope();
  sized_draw(GX_TRIANGLES, 3);
  decode_then_bp();
  EXPECT_EQ(fifo::draw_route_stats().routed, 2u);
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 1u);
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testLastDraw.vtxCount, 3u);
  EXPECT_EQ(gfx::g_testPushedVertexBytes, 3 * kVtxSize);
}

TEST_F(GXDrawRouteTest, SuppressedDrawInsideDisplayList) {
  alignas(32) std::array<u8, 128> storage{};
  GXBeginDisplayList(storage.data(), static_cast<u32>(storage.size()));
  triangle();
  const u32 size = GXEndDisplayList();

  install_router();
  push_capture_only_scope();
  GXCallDisplayList(storage.data(), size);
  GXAuroraPopScope();
  triangle();
  decode_then_bp();
  EXPECT_EQ(fifo::draw_route_stats().routed, 2u);
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 1u);
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testPushedVertexBytes, 3 * kVtxSize);
}

TEST_F(GXDrawRouteTest, SuppressedLinesAndPoints) {
  install_router();
  push_capture_only_scope();
  draw(GX_LINES, 2);
  draw(GX_LINESTRIP, 3);
  draw(GX_POINTS, 1);
  GXAuroraPopScope();
  decode_then_bp();
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 3u);
  EXPECT_EQ(gfx::g_testDrawCount, 0u);
  EXPECT_EQ(gfx::g_testPushedVertexBytes, 0u);
  EXPECT_EQ(gfx::g_testPushedIndexBytes, 0u);
}

TEST_F(GXDrawRouteTest, SuppressedZeroVertexDraw) {
  install_router();
  push_capture_only_scope();
  draw(GX_TRIANGLES, 0);
  GXAuroraPopScope();
  decode_then_bp();
  EXPECT_EQ(fifo::draw_route_stats().routed, 1u);
  EXPECT_EQ(fifo::draw_route_stats().suppressed, 1u);
  EXPECT_EQ(gfx::g_testDrawCount, 0u);
}

TEST_F(GXDrawRouteTest, PushHookSubstitutionBreaksMerge) {
  substituteAt = 1;
  install_router(true);
  triangle();
  triangle();
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 0u);
  EXPECT_EQ(pushes, 2u);
  EXPECT_EQ(fifo::draw_route_stats().substituted, 1u);
}

TEST_F(GXDrawRouteTest, PushHookOnIndexedDrawBreaksMerge) {
  install_router(true);
  indexed_triangle();
  triangle();
  decode();
  // Without a substitution the clean draw merges into the indexed one
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);

  substituteAt = pushes + 1;
  indexed_triangle();
  triangle();
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 3u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
  EXPECT_EQ(fifo::draw_route_stats().substituted, 1u);
}

TEST_F(GXDrawRouteTest, PushHookNotCalledForMergedFollowers) {
  install_router(true);
  triangle();
  triangle();
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 1u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
  EXPECT_EQ(pushes, 1u);
  EXPECT_EQ(fifo::draw_route_stats().substituted, 0u);
}

TEST_F(GXDrawRouteTest, InvalidateDrawCacheForcesRebuild) {
  triangle();
  decode();
  EXPECT_EQ(g_testPipelineBuildCount, 1u);
  EXPECT_EQ(g_testUniformBuildCount, 1u);

  fifo::invalidate_draw_cache();
  triangle();
  decode();
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 0u);
  EXPECT_EQ(g_testPipelineBuildCount, 2u);
  EXPECT_EQ(g_testUniformBuildCount, 2u);
}

TEST_F(GXDrawRouteTest, PipelineRebuildWithSameConfigSkipsLookup) {
  const CpuZoneDelta zones;
  triangle();
  decode();
  const auto ref = gfx::g_testLastDraw.pipeline;
  EXPECT_NE(ref, 0u);
  g_gxState.dirty |= aurora::gx::DirtyPipeline;
  triangle();
  decode();
  EXPECT_EQ(zones.calls(fifo::CpuZonePipelineBuild), 2u);
  EXPECT_EQ(g_testPipelineBuildCount, 2u);
  EXPECT_EQ(gfx::g_testPipelineRefCount, 1u);
  EXPECT_EQ(g_testShaderInfoCount, 1u);
  // The skip must keep what the first lookup returned, not just avoid calling it again
  EXPECT_EQ(gfx::g_testLastDraw.pipeline, ref);
  EXPECT_EQ(g_testLastUniformInfoSize, 1u);
  // A rebuild still forces the uniform, as a lookup does
  EXPECT_EQ(zones.calls(fifo::CpuZoneUniformBuild), 2u);
  EXPECT_EQ(g_testUniformBuildCount, 2u);
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(g_gxState.dirty, 0u);
}

TEST_F(GXDrawRouteTest, PipelineRebuildWithChangedConfigLooksUp) {
  expect_rebuild_looks_up([] {
    g_testPopulateDepthFunc = GX_LEQUAL;
    g_gxState.dirty |= aurora::gx::DirtyPipeline;
  });
}

TEST_F(GXDrawRouteTest, PipelineRebuildWithChangedShaderConfigLooksUp) {
  expect_rebuild_looks_up([] {
    g_testPopulateTevStageCount = 2;
    g_gxState.dirty |= aurora::gx::DirtyPipeline;
  });
}

TEST_F(GXDrawRouteTest, PipelineRebuildAfterLayoutChangeLooksUp) {
  expect_rebuild_looks_up([] { gfx::g_testNormalAttachment = true; });
}

TEST_F(GXDrawRouteTest, PipelineRebuildAfterBlockingFlipLooksUp) {
  expect_rebuild_looks_up([] {
    gfx::set_blocking_pipelines(true);
    g_gxState.dirty |= aurora::gx::DirtyPipeline;
  });
}

TEST_F(GXDrawRouteTest, PipelineRebuildAfterInvalidateDrawCacheLooksUp) {
  expect_rebuild_looks_up([] { fifo::invalidate_draw_cache(); });
}

TEST_F(GXDrawRouteTest, CpuZonesCountDrawWork) {
  const CpuZoneDelta zones;
  triangle();
  decode();
  fifo::invalidate_draw_cache();
  triangle();
  decode();
  EXPECT_EQ(zones.calls(fifo::CpuZoneProcess), 2u);
  EXPECT_EQ(zones.calls(fifo::CpuZonePushDraw), 2u);
  EXPECT_EQ(gfx::g_testDrawCount, 2u);
  EXPECT_EQ(zones.calls(fifo::CpuZonePipelineBuild), 2u);
  EXPECT_EQ(g_testPipelineBuildCount, 2u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneUniformBuild), 2u);
  EXPECT_EQ(g_testUniformBuildCount, 2u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneBindGroupBuild), 2u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneArrays), 0u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneFogLut), 0u);
  for (uint8_t zone = 0; zone < fifo::CpuZoneCount; ++zone) {
    EXPECT_EQ(zones.ns(static_cast<fifo::CpuZone>(zone)), 0u) << static_cast<int>(zone);
  }
}

TEST_F(GXDrawRouteTest, CpuZonesSkipCachedDrawState) {
  gfx::g_testMergeDraws = false;
  const CpuZoneDelta zones;
  triangle();
  triangle();
  triangle();
  decode();
  EXPECT_EQ(zones.calls(fifo::CpuZonePushDraw), 3u);
  EXPECT_EQ(zones.calls(fifo::CpuZonePipelineBuild), 1u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneBindGroupBuild), 1u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneUniformBuild), 1u);
  EXPECT_EQ(g_gxState.dirty, 0u);
}

TEST_F(GXDrawRouteTest, CpuZoneTimingNests) {
  const ScopedCpuZoneTiming timing{fifo::CpuZoneTimingFifo | fifo::CpuZoneTimingDraws};
  const CpuZoneDelta zones;
  uint64_t iterations = 0;
  while (iterations < 100000 && (zones.ns(fifo::CpuZoneProcess) == 0 || zones.ns(fifo::CpuZonePushDraw) == 0)) {
    fifo::invalidate_draw_cache();
    triangle();
    decode();
    ++iterations;
  }
  EXPECT_GT(zones.ns(fifo::CpuZoneProcess), 0u);
  EXPECT_GT(zones.ns(fifo::CpuZonePushDraw), 0u);
  EXPECT_EQ(zones.calls(fifo::CpuZoneProcess), iterations);
  EXPECT_EQ(zones.calls(fifo::CpuZonePushDraw), iterations);
  const uint64_t builds = zones.ns(fifo::CpuZonePipelineBuild) + zones.ns(fifo::CpuZoneBindGroupBuild) +
                          zones.ns(fifo::CpuZoneUniformBuild);
  EXPECT_LE(builds, zones.ns(fifo::CpuZonePushDraw));
  EXPECT_LE(zones.ns(fifo::CpuZonePushDraw), zones.ns(fifo::CpuZoneProcess));
}

TEST_F(GXDrawRouteTest, CpuZoneTimingFifoOnly) {
  const ScopedCpuZoneTiming timing{fifo::CpuZoneTimingFifo};
  const CpuZoneDelta zones;
  for (uint32_t i = 0; i < 100000 && zones.ns(fifo::CpuZoneProcess) == 0; ++i) {
    fifo::invalidate_draw_cache();
    triangle();
    decode();
  }
  EXPECT_GT(zones.ns(fifo::CpuZoneProcess), 0u);
  for (uint8_t zone = fifo::CpuZonePushDraw; zone < fifo::CpuZoneCount; ++zone) {
    EXPECT_EQ(zones.ns(static_cast<fifo::CpuZone>(zone)), 0u) << static_cast<int>(zone);
  }
  EXPECT_EQ(zones.calls(fifo::CpuZonePushDraw), gfx::g_testDrawCount);
}

TEST_F(GXDrawRouteTest, RouterSeesEveryDrawOnce) {
  set_vertex_format(GX_VTXFMT1);
  alignas(32) std::array<u8, 128> storage{};
  GXBeginDisplayList(storage.data(), static_cast<u32>(storage.size()));
  draw(GX_TRIANGLESTRIP, 4);
  const u32 size = GXEndDisplayList();

  install_router();
  draw(GX_QUADS, 4, GX_VTXFMT1);
  GXCallDisplayList(storage.data(), size);
  sized_draw(GX_TRIANGLEFAN, 5);
  indexed_triangle();
  draw(GX_LINES, 2);
  decode();

  const std::array<route::Query, 5> expected{{
      {.prim = GX_QUADS, .fmt = GX_VTXFMT1, .vtxCount = 4},
      {.prim = GX_TRIANGLESTRIP, .fmt = GX_VTXFMT0, .vtxCount = 4},
      {.prim = GX_TRIANGLEFAN, .fmt = GX_VTXFMT0, .vtxCount = 5},
      {.prim = GX_TRIANGLES, .fmt = GX_VTXFMT0, .vtxCount = 3, .indexed = true},
      {.prim = GX_LINES, .fmt = GX_VTXFMT0, .vtxCount = 2},
  }};
  ASSERT_EQ(queries.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(queries[i].prim, expected[i].prim) << i;
    EXPECT_EQ(queries[i].fmt, expected[i].fmt) << i;
    EXPECT_EQ(queries[i].vtxCount, expected[i].vtxCount) << i;
    EXPECT_EQ(queries[i].indexed, expected[i].indexed) << i;
  }
  EXPECT_EQ(fifo::draw_route_stats().routed, expected.size());
  // The fan merges into the strip; every other draw is its own command
  EXPECT_EQ(gfx::g_testDrawCount, 4u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
}

TEST_F(GXDrawRouteTest, RecordedCommandsEndTheMergeChain) {
  triangle();
  GXSetViewport(0.f, 0.f, 320.f, 240.f, 0.f, 1.f);
  triangle();
  GXSetViewport(0.f, 0.f, 320.f, 240.f, 0.f, 1.f);
  triangle();
  GXSetScissor(0, 0, 320, 240);
  triangle();
  GXCopyTex(reinterpret_cast<void*>(0x1234), GX_FALSE);
  triangle();
  decode();
  // A viewport that did not change appends nothing, so only the third draw merges
  EXPECT_EQ(gfx::g_testDrawCount, 4u);
  EXPECT_EQ(gfx::g_testMergedDrawCount, 1u);
}

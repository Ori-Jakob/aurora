#include <gtest/gtest.h>

#include "gx/command_processor.hpp"
#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "render/frame_state.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/scene/classify.hpp"
#include "render/scene/identity.hpp"
#include "render/scene/map_pick.hpp"
#include "render/scene/meshlets.hpp"
#include "render/scene/streamout.hpp"
#include "render/settings.hpp"
#include "render/stage_data.hpp"
#include "render_test_caps.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

namespace aurora::render::scene::streamout {
namespace {
using retained::Demand;

std::vector<uint8_t> g_pushed;
uint32_t g_pushes = 0;

gfx::Range fake_push(std::span<const uint8_t> bytes) noexcept {
  g_pushed.assign(bytes.begin(), bytes.end());
  ++g_pushes;
  return {256, static_cast<uint32_t>(bytes.size())};
}

Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

Mat3x4<float> affine(float x = 0.f, float y = 0.f, float z = 0.f) {
  return {{1.f, 0.f, 0.f, x}, {0.f, 1.f, 0.f, y}, {0.f, 0.f, 1.f, z}};
}

Mat3x4<float> concat(const Mat3x4<float>& a, const Mat3x4<float>& b) {
  Mat3x4<float> out{};
  const Vec4<float>* ar[3] = {&a.m0, &a.m1, &a.m2};
  const Vec4<float>* br[3] = {&b.m0, &b.m1, &b.m2};
  Vec4<float>* orow[3] = {&out.m0, &out.m1, &out.m2};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      float value = (*ar[r])[0] * (*br[0])[c] + (*ar[r])[1] * (*br[1])[c] + (*ar[r])[2] * (*br[2])[c];
      if (c == 3) {
        value += (*ar[r])[3];
      }
      (*orow[r])[c] = value;
    }
  }
  return out;
}

Mat3x4<float> look_at(const std::array<double, 3>& eye, const std::array<double, 3>& target) {
  std::array<double, 3> f{target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
  const double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (double& v : f) {
    v /= fl;
  }
  std::array<double, 3> s{-f[2], 0.0, f[0]};
  const double sl = std::sqrt(s[0] * s[0] + s[2] * s[2]);
  s[0] /= sl;
  s[2] /= sl;
  const std::array<double, 3> u{s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
  const auto row = [&](const std::array<double, 3>& axis, double sign) {
    const double t = -(axis[0] * eye[0] + axis[1] * eye[1] + axis[2] * eye[2]) * sign;
    return Vec4<float>{static_cast<float>(axis[0] * sign), static_cast<float>(axis[1] * sign),
                       static_cast<float>(axis[2] * sign), static_cast<float>(t)};
  };
  return {row(s, 1.0), row(u, 1.0), row(f, -1.0)};
}

struct BlockView {
  std::vector<uint32_t> words;

  [[nodiscard]] uint32_t at(uint32_t word) const { return words.at(word); }
  [[nodiscard]] uint32_t pieces() const { return at(block::PieceCount); }
  [[nodiscard]] uint32_t piece(uint32_t index, uint32_t field) const {
    return at(at(block::PiecesWord) + index * block::PieceWords + field);
  }
  [[nodiscard]] uint32_t jobs() const { return at(block::WriteJobCount); }
  [[nodiscard]] std::array<uint32_t, 4> job(uint32_t index) const {
    const uint32_t base = at(block::WriteJobsWord) + index * block::WriteJobWords;
    return {at(base), at(base + 1), at(base + 2), at(base + 3)};
  }
  [[nodiscard]] const uint32_t* data(uint32_t src) const { return &words.at(src); }
};

class RetainedStreamout : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
    gx::g_gxState.pnMtx[0].pos = affine(0.f, 0.f, -5.f);
    retained::testing::reset();
    retained::set_demand(Demand::Tool, true);
    testing::set_push(&fake_push);
    m_room = {.storageUsed = 0, .storageCapacity = 64ull << 20};
    testing::set_room(&m_room);
    testing::set_variants_ready(true);
    g_pushed.clear();
    g_pushes = 0;
    m_fifo.assign(4 << 20, 0);
    use_format(12, false);
  }

  void TearDown() override {
    retained::testing::reset();
    gx::fifo::reset_immutable_arrays();
    detail::remove_hints();
    detail::testing::reset_hints();
    scene::testing::reset();
    detail::testing::reset();
    gx::g_gxState = {};
  }

  static const gx::hints::Sink& sink() { return *gx::hints::sink(); }

  void use_format(uint8_t stride, bool normal, bool tex = false) {
    m_config.shaderConfig = {};
    m_config.shaderConfig.vtxStride = stride;
    m_config.shaderConfig.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
    if (normal) {
      m_config.shaderConfig.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32, .offset = 12};
    }
    if (tex) {
      m_config.shaderConfig.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
    }
  }

  void use_palette(std::vector<uint8_t> slots) {
    constexpr uint8_t Stride = 16;
    m_config.shaderConfig = {};
    m_config.shaderConfig.vtxStride = Stride;
    m_config.shaderConfig.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
    m_config.shaderConfig.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 12};
    gx::g_gxState.vtxDesc[GX_VA_PNMTXIDX] = GX_DIRECT;
    m_paletteSlots = std::move(slots);
  }

  void begin_frame(const Mat3x4<float>& view = affine(), uint32_t cutEpoch = 0) {
    detail::begin_hint_frame();
    retained::latch_gate(Settings{}, test::full_caps());
    scene::apply_frame_state(detail::FrameState{});
    const gx::hints::Camera main{
        .role = AURORA_CAMERA_MAIN,
        .view = view,
        .proj = perspective(),
        .type = GX_PERSPECTIVE,
        .cutEpoch = cutEpoch,
        .gxProj = gx::hints::gx_projection(perspective(), GX_PERSPECTIVE),
    };
    sink().camera(sink().userData, main);
    sink().stage(sink().userData, Stage::SceneBegin);
    gx::g_gxState.proj = main.gxProj;
    m_offset = 0;
    ++m_frameIndex;
  }

  static void begin_object(uint64_t id, uint32_t categories = AURORA_CAT_DYNAMIC, uint32_t flags = 0) {
    sink().objectBegin(sink().userData, {.id = id, .generation = 1, .categories = categories, .flags = flags});
  }
  static void end_object() { sink().objectEnd(sink().userData); }

  uint32_t draw(uint32_t vtxCount = 3, bool merge = false, GXPrimitive prim = GX_TRIANGLES) {
    const uint32_t stride = m_config.shaderConfig.vtxStride;
    const uint32_t size = vtxCount * stride;
    for (uint32_t v = 0; v < vtxCount; ++v) {
      const std::array<float, 3> p{static_cast<float>(v % 7), static_cast<float>(v % 5), static_cast<float>(v % 3)};
      std::memcpy(m_fifo.data() + m_offset + v * stride, p.data(), sizeof(p));
      if (!m_paletteSlots.empty()) {
        m_fifo[m_offset + v * stride + 12] = static_cast<uint8_t>(m_paletteSlots[v % m_paletteSlots.size()] * 3);
      }
    }
    const gx::draw_route::Query query{
        .prim = prim,
        .fmt = GX_VTXFMT0,
        .vtxCount = vtxCount,
        .vertices = std::span<const uint8_t>{m_fifo.data() + m_offset, size},
    };
    const gx::draw_route::Router* router = gx::draw_route::router();
    const gx::draw_route::Route route = router->route(router->userData, query).route;
    const uint32_t offset = m_offset;
    m_offset += size;
    if (route == gx::draw_route::Route::Capture) {
      const gx::DrawImmediateData immediates{.vtxStart = offset, .currentPnMtx = gx::g_gxState.currentPnMtx};
      router->capture(router->userData, {.prim = prim,
                                         .vertOffset = offset,
                                         .vertSize = size,
                                         .vtxCount = vtxCount,
                                         .immediates = &immediates,
                                         .attrs = m_config.shaderConfig.attrs.data(),
                                         .vtxStride = m_config.shaderConfig.vtxStride});
      return offset;
    }
    if (route == gx::draw_route::Route::Suppress) {
      return offset;
    }
    if (merge) {
      m_command.vertRange.size += size;
      m_command.vtxCount += vtxCount;
      router->merge(router->userData,
                    {.vertOffset = offset, .vertSize = size, .command = &m_command, .config = &m_config});
      return offset;
    }
    m_command = {};
    m_command.vertRange = {offset, size};
    m_command.vtxCount = vtxCount;
    m_command.instanceCount = 1;
    m_command.immediateData.vtxStart = offset;
    m_command.immediateData.currentPnMtx = gx::g_gxState.currentPnMtx;
    m_command.immediateData._pad = ++m_record;
    router->push(router->userData, m_command, m_config, m_info);
    return offset;
  }

  void object_draw(uint64_t id, uint32_t vtxCount = 3, uint32_t categories = AURORA_CAT_DYNAMIC, uint32_t flags = 0) {
    begin_object(id, categories, flags);
    draw(vtxCount);
    end_object();
  }

  std::shared_ptr<const Plan> stage(bool commit = true) { return stage_at(Stage::OpaqueDone, commit); }

  std::shared_ptr<const Plan> stage_at(Stage which, bool commit = true, bool marker = true) {
    detail::StageData data{.stage = which, .frameIndex = m_frameIndex, .markerSeen = marker};
    retained::prepare(data);
    if (commit) {
      std::array<std::byte, ParamsSize> params{};
      commit_stage_params(data, nullptr, params);
      std::memcpy(m_params.data(), params.data(), sizeof(m_params));
    }
    retained::seal(data);
    return data.retained != nullptr ? data.retained->plan : nullptr;
  }

  static retained::FrameCounts end() {
    scene::seal_frame();
    return retained::frame_counts();
  }

  [[nodiscard]] static BlockView block() {
    BlockView view;
    view.words.resize(g_pushed.size() / 4);
    std::memcpy(view.words.data(), g_pushed.data(), view.words.size() * 4);
    return view;
  }

  static uint32_t slot_of(uint64_t id) { return retained::testing::find({.id = id, .generation = 1}); }

  static bool has_event(const Plan& plan, EventType type, uint32_t slot) {
    return std::ranges::any_of(plan.events,
                               [&](const PlanEvent& event) { return event.type == type && event.slot == slot; });
  }

  static uint32_t count_events(const Plan& plan, EventType type) {
    return static_cast<uint32_t>(
        std::ranges::count_if(plan.events, [&](const PlanEvent& event) { return event.type == type; }));
  }

  [[nodiscard]] static uint32_t roll_flags(const BlockView& view, uint32_t job) {
    return view.at(view.at(block::RollJobsWord) + job * block::RollJobWords + block::RollFlags);
  }

  static bool writes(const BlockView& view, uint32_t slot, uint32_t first, uint32_t count) {
    for (uint32_t i = 0; i < view.jobs(); ++i) {
      const auto job = view.job(i);
      if (job[0] == block::TargetInstances && job[1] == slot * record::Words + first && job[3] == count) {
        return true;
      }
    }
    return false;
  }

  gfx::detail::StreamRoom m_room{};
  std::vector<uint8_t> m_fifo;
  uint32_t m_offset = 0;
  uint32_t m_record = 0;
  uint32_t m_frameIndex = 0;
  std::array<uint32_t, 4> m_params{};
  gx::DrawData m_command{};
  gx::PipelineConfig m_config{};
  gx::ShaderInfo m_info{};
  std::vector<uint8_t> m_paletteSlots;
};

TEST_F(RetainedStreamout, PlanGroupsChunksByVariant) {
  begin_frame();
  begin_object(0x10);
  draw(3);
  end_object();
  use_format(24, true);
  begin_object(0x11);
  draw(3);
  end_object();
  use_format(12, false);
  begin_object(0x12);
  draw(130);
  end_object();
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  const BlockView view = block();
  EXPECT_EQ(view.pieces(), 3u);
  ASSERT_EQ(plan->variants.size(), 2u);
  EXPECT_EQ(plan->variants[0].chunkFirst, 0u);
  EXPECT_EQ(plan->variants[0].chunkCount, 1u + 3u);
  EXPECT_EQ(plan->variants[1].chunkFirst, 4u);
  EXPECT_EQ(plan->variants[1].chunkCount, 1u);
  EXPECT_EQ(view.at(block::VtxChunkCount), 5u);
  EXPECT_EQ(view.at(block::IdxChunkCount), 1u + 1u + 1u);
  EXPECT_EQ(view.piece(0, block::PieceVtxCount), 3u);
  EXPECT_EQ(view.piece(1, block::PieceVtxCount), 130u);
  EXPECT_EQ(view.piece(2, block::PieceFlags) & block::PieceHasNormal, block::PieceHasNormal);
  EXPECT_EQ(testing::variant_count(), 2u);
  EXPECT_EQ(m_params[0], plan->serial);
  EXPECT_TRUE(plan->clearInstances);
  end();
  EXPECT_TRUE(testing::slot(slot_of(0x12)).streamed);
}

TEST_F(RetainedStreamout, NewVertexCapDefersStaticsNotDynamics) {
  constexpr uint32_t Large = 39999;
  begin_frame();
  object_draw(0x20, Large, AURORA_CAT_STATIC);
  object_draw(0x21, Large, AURORA_CAT_STATIC);
  object_draw(0x22, Large, AURORA_CAT_DYNAMIC);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(testing::slot(slot_of(0x20)).streamed);
  EXPECT_FALSE(testing::slot(slot_of(0x21)).streamed);
  EXPECT_TRUE(testing::slot(slot_of(0x22)).streamed);
  EXPECT_FALSE(settled());
  EXPECT_EQ(stats().capDeferred, 1u);
  EXPECT_EQ(coverage().pending, 1u);

  begin_frame();
  object_draw(0x20, Large, AURORA_CAT_STATIC);
  object_draw(0x21, Large, AURORA_CAT_STATIC);
  object_draw(0x22, Large, AURORA_CAT_DYNAMIC);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(testing::slot(slot_of(0x21)).streamed);
  EXPECT_EQ(testing::slot(slot_of(0x20)).streams, 1u);
  EXPECT_TRUE(settled());
  EXPECT_EQ(coverage().pending, 0u);
}

TEST_F(RetainedStreamout, CaptureOnlyPiecesAreNeverCapDeferred) {
  constexpr uint32_t Large = 39999;
  begin_frame();
  object_draw(0x30, Large, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  object_draw(0x31, Large, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  object_draw(0x32, Large, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  ASSERT_NE(stage(), nullptr);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.captureOnly, 3u);
  EXPECT_EQ(counts.lostCaptures, 0u);
  EXPECT_TRUE(testing::slot(slot_of(0x30)).streamed);
  EXPECT_TRUE(testing::slot(slot_of(0x31)).streamed);
  EXPECT_TRUE(testing::slot(slot_of(0x32)).streamed);
  EXPECT_EQ(stats().capDeferred, 0u);
}

TEST_F(RetainedStreamout, HeadroomRefusalCommitsNothing) {
  m_room = {.storageUsed = 512, .storageCapacity = 1024};
  begin_frame();
  object_draw(0x40, 300, AURORA_CAT_STATIC);
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(plan->header_only());
  EXPECT_TRUE(plan->variants.empty());
  EXPECT_EQ(block().pieces(), 0u);
  end();
  EXPECT_FALSE(testing::slot(slot_of(0x40)).streamed);
  EXPECT_EQ(stats().headroomDeferred, 1u);

  m_room = {.storageUsed = 0, .storageCapacity = 64ull << 20};
  begin_frame();
  object_draw(0x40, 300, AURORA_CAT_STATIC);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(testing::slot(slot_of(0x40)).streamed);
}

TEST_F(RetainedStreamout, OverrunFrameCommitsNothing) {
  retained::testing::set_overrun(1);
  begin_frame();
  object_draw(0x50, 3, AURORA_CAT_STATIC);
  EXPECT_EQ(stage(), nullptr);
  end();
  EXPECT_FALSE(testing::slot(slot_of(0x50)).streamed);
  EXPECT_EQ(g_pushes, 0u);

  retained::testing::set_overrun(0);
  m_room.overrun = true;
  begin_frame();
  object_draw(0x50, 3, AURORA_CAT_STATIC);
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(plan->header_only());
  end();
  EXPECT_FALSE(testing::slot(slot_of(0x50)).streamed);
  EXPECT_EQ(stats().overrunDeferred, 1u);

  m_room.overrun = false;
  begin_frame();
  object_draw(0x50, 3, AURORA_CAT_STATIC);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(testing::slot(slot_of(0x50)).streamed);
}

TEST_F(RetainedStreamout, UncommittedPlanRollsBack) {
  begin_frame();
  object_draw(0x60, 30);
  const auto first = stage(false);
  ASSERT_NE(first, nullptr);
  const uint32_t vertexFirst = block().piece(0, block::PieceOutVertexFirst);
  end();
  const uint32_t slot = slot_of(0x60);
  EXPECT_FALSE(testing::slot(slot).streamed);
  EXPECT_EQ(stats().uncommitted, 1u);

  begin_frame();
  object_draw(0x60, 30);
  const auto second = stage();
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->baseSerial, 0u);
  EXPECT_TRUE(second->clearInstances);
  EXPECT_GT(second->serial, first->serial);
  EXPECT_EQ(block().piece(0, block::PieceOutVertexFirst), vertexFirst);
  end();
  EXPECT_TRUE(testing::slot(slot).streamed);
}

TEST_F(RetainedStreamout, DroppedPlanStopsLaterDeltas) {
  testing::ChainState state;
  EXPECT_EQ(testing::chain_step(state, 1, 0, true), testing::Chain::Accept);
  EXPECT_EQ(testing::chain_step(state, 2, 1, false), testing::Chain::Accept);
  EXPECT_EQ(testing::chain_step(state, 4, 3, false), testing::Chain::Break);
  EXPECT_EQ(testing::chain_step(state, 5, 4, false), testing::Chain::Skip);
  EXPECT_EQ(testing::chain_step(state, 6, 0, true), testing::Chain::Accept);
  EXPECT_EQ(testing::chain_step(state, 7, 6, false), testing::Chain::Accept);

  begin_frame();
  object_draw(0x70);
  ASSERT_NE(stage(), nullptr);
  end();
  begin_frame();
  object_draw(0x70);
  const auto delta = stage();
  ASSERT_NE(delta, nullptr);
  EXPECT_FALSE(delta->clearInstances);
  EXPECT_NE(delta->baseSerial, 0u);
  end();

  testing::break_chain();
  begin_frame();
  object_draw(0x70);
  const auto reset = stage();
  ASSERT_NE(reset, nullptr);
  EXPECT_TRUE(reset->clearInstances);
  EXPECT_EQ(reset->baseSerial, 0u);
  EXPECT_EQ(block().pieces(), 1u);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.resyncs, 1u);
  EXPECT_EQ(stats().resyncs, 1u);
  EXPECT_EQ(testing::slot(slot_of(0x70)).streams, 1u);
}

TEST_F(RetainedStreamout, ResyncKeepsUndrawnStaticsPending) {
  begin_frame();
  object_draw(0x78, 3, AURORA_CAT_STATIC);
  object_draw(0x79, 3, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  object_draw(0x7A);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(settled());
  EXPECT_EQ(coverage().pending, 0u);

  testing::break_chain();
  begin_frame();
  object_draw(0x7A);
  ASSERT_NE(stage(), nullptr);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.resyncs, 1u);
  EXPECT_EQ(counts.lostCaptures, 2u);
  EXPECT_FALSE(settled());
  EXPECT_EQ(coverage().pending, 2u);
  EXPECT_TRUE(testing::slot(slot_of(0x7A)).streamed);

  begin_frame();
  object_draw(0x78, 3, AURORA_CAT_STATIC);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_FALSE(settled());
  EXPECT_EQ(coverage().pending, 1u);

  begin_frame();
  object_draw(0x79, 3, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_TRUE(settled());
  EXPECT_EQ(coverage().pending, 0u);
  EXPECT_TRUE(testing::slot(slot_of(0x78)).streamed);
  EXPECT_TRUE(testing::slot(slot_of(0x79)).streamed);
}

TEST_F(RetainedStreamout, FreedSlotWritesInvalidationInSamePlan) {
  begin_frame();
  object_draw(0x80);
  ASSERT_NE(stage(), nullptr);
  end();
  const uint32_t slot = slot_of(0x80);
  ASSERT_NE(slot, retained::TransientSlot);
  ASSERT_TRUE(testing::slot(slot).streamed);
  retained::testing::evict(slot, retained::EvictReason::Dynamic);
  EXPECT_FALSE(testing::slot(slot).streamed);

  begin_frame();
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(has_event(*plan, EventType::Freed, slot));
  const BlockView view = block();
  bool cleared = false;
  for (uint32_t i = 0; i < view.jobs(); ++i) {
    const auto job = view.job(i);
    if (job[0] == block::TargetInstances && job[1] == slot * record::Words && job[3] == 1) {
      cleared = *view.data(job[2]) == 0;
    }
  }
  EXPECT_TRUE(cleared);
  end();
}

TEST_F(RetainedStreamout, RigidDynamicMovesWithoutRestream) {
  begin_frame();
  object_draw(0x90);
  ASSERT_NE(stage(), nullptr);
  end();
  const uint32_t slot = slot_of(0x90);

  gx::g_gxState.pnMtx[0].pos = affine(1.f, 0.f, -5.f);
  begin_frame();
  object_draw(0x90);
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(has_event(*plan, EventType::Moved, slot));
  EXPECT_FALSE(has_event(*plan, EventType::Restreamed, slot));
  EXPECT_EQ(block().pieces(), 0u);
  EXPECT_TRUE(writes(block(), slot, record::MovedFirst, record::MovedLast - record::MovedFirst + 1));
  end();
  EXPECT_EQ(testing::slot(slot).streams, 1u);
  EXPECT_DOUBLE_EQ(testing::slot(slot).lastSent.m[0][3], 1.0);

  begin_frame();
  object_draw(0x90);
  const auto still = stage();
  ASSERT_NE(still, nullptr);
  EXPECT_TRUE(still->events.empty());
  end();
}

TEST_F(RetainedStreamout, CameraOrbitKeepsStationaryWorld) {
  const Mat3x4<float> nearModel = affine(10.f, 0.f, 0.f);
  const Mat3x4<float> farModel = affine(1.0e5f, 0.f, 0.f);
  const std::array<double, 3> centre{1.0e5, 0.0, 0.0};
  uint32_t moved = 0;
  for (uint32_t frame = 0; frame < 600; ++frame) {
    const double angle = 2.0 * std::numbers::pi * frame / 600.0;
    const std::array<double, 3> eye{centre[0] + 50.0 * std::cos(angle), 8.0, centre[2] + 50.0 * std::sin(angle)};
    const Mat3x4<float> view = look_at(eye, centre);
    begin_frame(view);
    gx::g_gxState.pnMtx[0].pos = concat(view, nearModel);
    object_draw(0xA0);
    gx::g_gxState.pnMtx[0].pos = concat(view, farModel);
    object_draw(0xA1);
    const auto plan = stage();
    ASSERT_NE(plan, nullptr);
    if (frame != 0) {
      moved += static_cast<uint32_t>(
          std::ranges::count_if(plan->events, [](const PlanEvent& event) { return event.type == EventType::Moved; }));
    }
    end();
  }
  EXPECT_EQ(moved, 0u);
  EXPECT_EQ(testing::slot(slot_of(0xA0)).recordWrites, 1u);
  EXPECT_EQ(testing::slot(slot_of(0xA1)).recordWrites, 1u);
  EXPECT_EQ(testing::slot(slot_of(0xA1)).streams, 1u);
}

TEST_F(RetainedStreamout, UvChangesEmitEvents) {
  use_format(20, false, true);
  gx::g_gxState.numTevStages = 1;
  gx::g_gxState.tevStages[0].colorPass.a = GX_CC_TEXC;
  gx::g_gxState.tevStages[0].texMapId = GX_TEXMAP0;
  gx::g_gxState.tevStages[0].texCoordId = GX_TEXCOORD0;
  gx::g_gxState.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX0, .postMtx = GX_PTIDENTITY};
  gx::g_gxState.texMtxs[0] = affine();
  begin_frame();
  object_draw(0xB0);
  ASSERT_NE(stage(), nullptr);
  end();
  const uint32_t slot = slot_of(0xB0);
  EXPECT_EQ(testing::slot(slot).flags & instance_flag::UvNonTex, 0u);

  gx::g_gxState.texMtxs[0] = affine(0.5f, 0.f, 0.f);
  begin_frame();
  object_draw(0xB0);
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(has_event(*plan, EventType::UvChanged, slot));
  EXPECT_FALSE(has_event(*plan, EventType::Moved, slot));
  EXPECT_TRUE(writes(block(), slot, record::UvFirst, record::UvLast - record::UvFirst + 1));
  end();

  begin_frame();
  object_draw(0xB0);
  const auto still = stage();
  ASSERT_NE(still, nullptr);
  EXPECT_TRUE(still->events.empty());
  end();
}

TEST_F(RetainedStreamout, ReadyClustersReachTheRecord) {
  jobs::initialize(1);
  jobs::set_synchronous(true);
  constexpr auto Building = static_cast<uint32_t>(meshlets::State::Building);
  constexpr auto Ready = static_cast<uint32_t>(meshlets::State::Ready);
  begin_frame();
  object_draw(0xD0, 3, AURORA_CAT_STATIC);
  ASSERT_NE(stage(), nullptr);
  const uint32_t slot = slot_of(0xD0);
  EXPECT_EQ(testing::slot(slot).clusterWord >> 24, Building);
  end();

  begin_frame();
  const auto plan = stage();
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(has_event(*plan, EventType::ClustersChanged, slot));
  EXPECT_TRUE(writes(block(), slot, record::ClusterFirst, 3));
  EXPECT_EQ(testing::slot(slot).clusterWord, Ready << 24 | 1u);
  end();

  begin_frame();
  object_draw(0xD0, 3, AURORA_CAT_STATIC);
  const auto again = stage();
  ASSERT_NE(again, nullptr);
  EXPECT_FALSE(has_event(*again, EventType::ClustersChanged, slot));
  end();
  EXPECT_EQ(meshlets::pending_static(), 0u);
  retained::testing::reset();
  jobs::shutdown();
}

TEST_F(RetainedStreamout, UvTransformComposesTexMtxAndPostMtx) {
  use_format(20, false, true);
  gx::g_gxState.numTevStages = 1;
  gx::g_gxState.tevStages[0].colorPass.a = GX_CC_TEXC;
  gx::g_gxState.tevStages[0].texMapId = GX_TEXMAP0;
  gx::g_gxState.tevStages[0].texCoordId = GX_TEXCOORD0;
  gx::g_gxState.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX0, .postMtx = GX_PTTEXMTX0};
  gx::g_gxState.texMtxs[0] = {{2.f, 0.f, 0.f, 0.25f}, {0.f, 3.f, 0.f, 0.5f}, {0.f, 0.f, 1.f, 0.f}};
  gx::g_gxState.ptTexMtxs[0] = {{1.f, 0.f, 0.f, 0.1f}, {0.f, 1.f, 0.f, 0.2f}, {0.f, 0.f, 1.f, 0.f}};
  begin_frame();
  object_draw(0xC0);
  ASSERT_NE(stage(), nullptr);
  const uint32_t slot = slot_of(0xC0);
  const BlockView view = block();
  const uint32_t* uv = nullptr;
  for (uint32_t i = 0; i < view.jobs(); ++i) {
    const auto job = view.job(i);
    if (job[0] == block::TargetInstances && job[1] == slot * record::Words && job[3] == record::CpuWords) {
      uv = view.data(job[2] + record::UvBase);
    }
  }
  ASSERT_NE(uv, nullptr);
  const std::array<float, 6> want{2.f, 0.f, 0.35f, 0.f, 3.f, 0.7f};
  for (size_t i = 0; i < want.size(); ++i) {
    EXPECT_NEAR(std::bit_cast<float>(uv[i]), want[i], 1e-6f) << i;
  }
  end();
  EXPECT_EQ(testing::slot(slot).flags & (instance_flag::UvNonTex | instance_flag::UvNonAffine), 0u);
}

TEST_F(RetainedStreamout, LinesAndPointsAreSkipped) {
  begin_frame();
  begin_object(0xD1);
  draw(4, false, GX_LINES);
  end_object();
  ASSERT_NE(stage(), nullptr);
  end();
  EXPECT_EQ(slot_of(0xD1), retained::TransientSlot);
}

TEST_F(RetainedStreamout, FollowerIndicesAreInstanceLocal) {
  begin_frame();
  begin_object(0xE0);
  draw(3);
  const uint32_t follower = draw(6, true);
  end_object();
  ASSERT_NE(stage(), nullptr);
  const BlockView view = block();
  ASSERT_EQ(view.pieces(), 2u);
  EXPECT_EQ(view.piece(1, block::PieceImmediates), follower);
  EXPECT_EQ(view.piece(1, block::PieceIndexCount), 6u);
  EXPECT_EQ(view.piece(1, block::PiecePrim), uint32_t{GX_TRIANGLES});
  EXPECT_NE(view.piece(0, block::PieceOutIndexFirst), view.piece(1, block::PieceOutIndexFirst));
  EXPECT_NE(view.piece(0, block::PieceInstance), view.piece(1, block::PieceInstance));
  EXPECT_EQ(view.at(block::IdxChunkCount), 2u);
  end();
}

TEST_F(RetainedStreamout, LatePiecesSealAtPrePost) {
  constexpr uint32_t Water = AURORA_CAT_DYNAMIC | AURORA_CAT_WATER;
  begin_frame();
  object_draw(0x100);
  const auto opaque = stage();
  ASSERT_NE(opaque, nullptr);
  EXPECT_FALSE(opaque->late);
  EXPECT_EQ(block().job(0)[3], frame::LateFrame);
  object_draw(0x101, 6, Water);
  const auto late = stage_at(Stage::PrePost);
  ASSERT_NE(late, nullptr);
  EXPECT_TRUE(late->late);
  EXPECT_FALSE(late->clearInstances);
  EXPECT_EQ(late->baseSerial, opaque->serial);
  const BlockView view = block();
  ASSERT_EQ(view.pieces(), 1u);
  EXPECT_EQ(view.piece(0, block::PieceInstance), slot_of(0x101));
  const auto header = view.job(0);
  EXPECT_EQ(header[0], block::TargetFrame);
  EXPECT_EQ(header[1], 0u);
  ASSERT_EQ(header[3], frame::LateFrame + 1);
  EXPECT_EQ(*view.data(header[2] + frame::LateFrame), m_frameIndex);
  EXPECT_EQ(*view.data(header[2] + frame::FrameIndex), m_frameIndex);
  EXPECT_NE(*view.data(header[2] + frame::Flags) & frame::Reset, 0u);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.late, 1u);
  const uint32_t lateSlot = slot_of(0x101);
  EXPECT_TRUE(testing::slot(lateSlot).streamed);
  EXPECT_EQ(testing::slot(lateSlot).flags & (instance_flag::Late | instance_flag::Water),
            instance_flag::Late | instance_flag::Water);
  EXPECT_EQ(testing::slot(slot_of(0x100)).flags & instance_flag::Late, 0u);
  EXPECT_EQ(stats().latePlans, 1u);
  EXPECT_EQ(stats().latePieces, 1u);

  begin_frame();
  object_draw(0x100);
  ASSERT_NE(stage(), nullptr);
  object_draw(0x101, 6, Water);
  EXPECT_EQ(stage_at(Stage::PrePost), nullptr);
  end();
  EXPECT_EQ(testing::slot(lateSlot).streams, 1u);
  EXPECT_EQ(stats().latePlans, 1u);
}

TEST_F(RetainedStreamout, SkippedOpaqueDoneStreamsAtPrePost) {
  begin_frame();
  object_draw(0x110, 3, AURORA_CAT_STATIC);
  object_draw(0x111, 30, AURORA_CAT_STATIC, AURORA_OBJ_CAPTURE_ONLY);
  detail::StageData opaque{.stage = Stage::OpaqueDone, .frameIndex = m_frameIndex, .markerSeen = true};
  retained::prepare(opaque);
  ASSERT_NE(opaque.retained, nullptr);
  ASSERT_NE(opaque.retained->plan, nullptr);
  const uint32_t opaqueFirst = block().piece(0, block::PieceOutVertexFirst);
  retained::skip(opaque);
  EXPECT_FALSE(testing::slot(slot_of(0x110)).streamed);
  const auto late = stage_at(Stage::PrePost);
  ASSERT_NE(late, nullptr);
  EXPECT_TRUE(late->late);
  EXPECT_TRUE(late->clearInstances);
  EXPECT_EQ(late->baseSerial, 0u);
  EXPECT_EQ(block().pieces(), 2u);
  EXPECT_EQ(block().piece(0, block::PieceOutVertexFirst), opaqueFirst);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.late, 0u);
  EXPECT_EQ(counts.lostCaptures, 0u);
  EXPECT_EQ(counts.undeliveredStatics, 0u);
  for (const uint64_t id : {uint64_t{0x110}, uint64_t{0x111}}) {
    EXPECT_TRUE(testing::slot(slot_of(id)).streamed) << id;
    EXPECT_EQ(testing::slot(slot_of(id)).flags & instance_flag::Late, 0u) << id;
  }
  EXPECT_TRUE(settled());
  EXPECT_EQ(stats().uncommitted, 1u);
  EXPECT_EQ(stats().latePieces, 2u);
}

TEST_F(RetainedStreamout, OpaqueDoneMissingMarkerKeepsPiecesOpaque) {
  EXPECT_TRUE(pass_desc().runsWithoutMarker);
  const detail::PassDesc lateDesc = late_pass_desc();
  EXPECT_EQ(lateDesc.stage, Stage::PrePost);
  EXPECT_EQ(lateDesc.paramsSize, ParamsSize);
  EXPECT_EQ(lateDesc.stageParams, pass_desc().stageParams);
  begin_frame();
  object_draw(0x120);
  object_draw(0x121, 3, AURORA_CAT_DYNAMIC | AURORA_CAT_WATER);
  const auto opaque = stage_at(Stage::OpaqueDone, true, false);
  ASSERT_NE(opaque, nullptr);
  EXPECT_FALSE(opaque->late);
  EXPECT_EQ(block().pieces(), 2u);
  EXPECT_EQ(stage_at(Stage::PrePost), nullptr);
  const retained::FrameCounts counts = end();
  EXPECT_EQ(counts.late, 0u);
  EXPECT_TRUE(testing::slot(slot_of(0x121)).streamed);
  EXPECT_EQ(testing::slot(slot_of(0x121)).flags & instance_flag::Late, 0u);
  EXPECT_NE(testing::slot(slot_of(0x121)).flags & instance_flag::Water, 0u);
  EXPECT_EQ(stats().latePlans, 0u);
}

TEST_F(RetainedStreamout, LatePlanChainsAfterOpaquePlan) {
  testing::ChainState chain;
  uint32_t previousLate = 0;
  for (uint32_t frame = 0; frame < 3; ++frame) {
    begin_frame();
    gx::g_gxState.pnMtx[0].pos = affine(0.f, 0.f, -5.f);
    object_draw(0x130, 3, AURORA_CAT_STATIC);
    const auto opaque = stage();
    gx::g_gxState.pnMtx[0].pos = affine(static_cast<float>(frame), 0.f, -5.f);
    object_draw(0x131);
    const auto late = stage_at(Stage::PrePost);
    ASSERT_NE(opaque, nullptr);
    ASSERT_NE(late, nullptr);
    EXPECT_TRUE(late->late);
    EXPECT_FALSE(late->clearInstances);
    EXPECT_EQ(late->baseSerial, opaque->serial);
    EXPECT_GT(late->serial, opaque->serial);
    if (frame > 0) {
      EXPECT_EQ(opaque->baseSerial, previousLate);
      EXPECT_TRUE(has_event(*late, EventType::Moved, slot_of(0x131)));
    }
    EXPECT_EQ(testing::chain_step(chain, opaque->serial, opaque->baseSerial, opaque->clearInstances),
              testing::Chain::Accept);
    EXPECT_EQ(testing::chain_step(chain, late->serial, late->baseSerial, late->clearInstances), testing::Chain::Accept);
    previousLate = late->serial;
    end();
  }
  EXPECT_EQ(testing::slot(slot_of(0x131)).streams, 1u);
  EXPECT_EQ(stats().latePlans, 3u);
}

TEST_F(RetainedStreamout, UndeliveredLateEntriesOpenPrePost) {
  begin_frame();
  ASSERT_NE(stage(), nullptr);
  object_draw(0x140, 2, AURORA_CAT_STATIC);
  const auto late = stage_at(Stage::PrePost);
  ASSERT_NE(late, nullptr);
  EXPECT_TRUE(late->header_only());
  EXPECT_EQ(end().undeliveredStatics, 0u);
  EXPECT_FALSE(testing::slot(slot_of(0x140)).streamed);

  begin_frame();
  ASSERT_NE(stage(), nullptr);
  object_draw(0x140, 2, AURORA_CAT_STATIC);
  EXPECT_EQ(stage_at(Stage::PrePost), nullptr);
  end();
  EXPECT_EQ(stats().largeSkipped, 2u);
}

TEST_F(RetainedStreamout, PaletteStationaryUnderOrbit) {
  use_palette({1, 2});
  const std::array<double, 3> centre{1.0e5, 0.0, 0.0};
  const std::array<Mat3x4<float>, 2> bones{affine(1.0e5f, 0.f, 0.f), affine(1.0e5f, 1.5f, 0.f)};
  uint32_t changes = 0;
  for (uint32_t frame = 0; frame < 600; ++frame) {
    const double angle = 2.0 * std::numbers::pi * frame / 600.0;
    const std::array<double, 3> eye{centre[0] + 50.0 * std::cos(angle), 8.0, centre[2] + 50.0 * std::sin(angle)};
    const Mat3x4<float> view = look_at(eye, centre);
    begin_frame(view);
    gx::g_gxState.pnMtx[0].pos = concat(view, affine(static_cast<float>(frame), 0.f, 0.f));
    gx::g_gxState.pnMtx[1].pos = concat(view, bones[0]);
    gx::g_gxState.pnMtx[2].pos = concat(view, bones[1]);
    object_draw(0x150, 6);
    const auto plan = stage();
    ASSERT_NE(plan, nullptr);
    if (frame != 0) {
      changes += count_events(*plan, EventType::Moved) + count_events(*plan, EventType::Restreamed);
      changes += block().pieces();
    }
    end();
  }
  EXPECT_EQ(changes, 0u);
  const testing::SlotState st = testing::slot(slot_of(0x150));
  EXPECT_NE(st.flags & instance_flag::Palette, 0u);
  EXPECT_EQ(st.streams, 1u);
  EXPECT_EQ(st.recordWrites, 1u);
  EXPECT_EQ(st.usedMask, 0b110u);
  EXPECT_EQ(st.origin[0], 99840.0);
  EXPECT_EQ(stats().paletteStreamed, 1u);
  EXPECT_EQ(stats().paletteSkipped, 599u);
}

TEST_F(RetainedStreamout, ImmobilePaletteLatches) {
  use_palette({0});
  for (uint32_t frame = 0; frame < 10; ++frame) {
    begin_frame();
    gx::g_gxState.pnMtx[0].pos = affine(static_cast<float>(frame) * 4.f, 0.f, -5.f);
    object_draw(0x160, 3, AURORA_CAT_DYNAMIC, AURORA_OBJ_IMMOBILE);
    ASSERT_NE(stage(), nullptr);
    end();
  }
  const testing::SlotState st = testing::slot(slot_of(0x160));
  EXPECT_EQ(st.streams, 1u);
  EXPECT_NE(st.flags & instance_flag::Latched, 0u);
  EXPECT_EQ(stats().paletteLatched, 9u);
  EXPECT_EQ(stats().paletteSkipped, 0u);
}

TEST_F(RetainedStreamout, UsedSlotMaskIgnoresUnreferencedSlots) {
  use_palette({1, 3});
  const auto frame = [&](float unused, float used) {
    begin_frame();
    gx::g_gxState.pnMtx[0].pos = affine(unused, 9.f, 9.f);
    gx::g_gxState.pnMtx[1].pos = affine(1.f, 0.f, -5.f);
    gx::g_gxState.pnMtx[2].pos = affine(unused, 7.f, 7.f);
    gx::g_gxState.pnMtx[3].pos = affine(used, 0.f, -5.f);
    object_draw(0x170, 6);
    return stage();
  };
  const auto first = frame(0.f, 2.f);
  ASSERT_NE(first, nullptr);
  const BlockView view = block();
  ASSERT_EQ(view.pieces(), 1u);
  const uint32_t matrices = view.at(block::MatricesWord) + view.piece(0, block::PieceMatrixWord);
  EXPECT_EQ(view.at(block::WriteJobsWord) - view.at(block::MatricesWord), palette::BankWords);
  for (uint32_t slot = 0; slot < palette::Matrices; ++slot) {
    const bool used = slot == 1 || slot == 3;
    bool zero = true;
    for (uint32_t w = 0; w < palette::MatrixWords; ++w) {
      zero = zero && view.at(matrices + slot * palette::MatrixWords + w) == 0 &&
             view.at(matrices + palette::NormalWord + slot * palette::MatrixWords + w) == 0;
    }
    EXPECT_EQ(zero, !used) << slot;
  }
  EXPECT_EQ(std::bit_cast<float>(view.at(matrices + 3 * palette::MatrixWords + 3)), 2.f);
  end();
  const uint32_t slot = slot_of(0x170);
  EXPECT_EQ(testing::slot(slot).usedMask, 0b1010u);
  EXPECT_EQ(testing::slot(slot).seenMask, 0b1010u);

  const auto still = frame(5.f, 2.f);
  ASSERT_NE(still, nullptr);
  EXPECT_EQ(block().pieces(), 0u);
  EXPECT_FALSE(has_event(*still, EventType::Restreamed, slot));
  end();

  const auto moved = frame(5.f, 3.f);
  ASSERT_NE(moved, nullptr);
  EXPECT_EQ(block().pieces(), 1u);
  EXPECT_TRUE(has_event(*moved, EventType::Restreamed, slot));
  EXPECT_FALSE(has_event(*moved, EventType::Moved, slot));
  end();
  EXPECT_EQ(testing::slot(slot).streams, 2u);
}

TEST_F(RetainedStreamout, CutMarksFresh) {
  use_palette({0});
  const auto frame = [&](float x, uint32_t cut) {
    begin_frame(affine(), cut);
    gx::g_gxState.pnMtx[0].pos = affine(x, 0.f, -5.f);
    object_draw(0x180);
    EXPECT_NE(stage(), nullptr);
    const BlockView view = block();
    end();
    EXPECT_EQ(view.at(block::RollJobCount), 1u);
    return roll_flags(view, 0);
  };
  EXPECT_EQ(frame(0.f, 0), block::RollPalette | block::RollFresh);
  EXPECT_EQ(frame(1.f, 0), block::RollPalette);
  EXPECT_EQ(frame(2.f, 1), block::RollPalette | block::RollFresh);
  EXPECT_EQ(frame(3.f, 1), block::RollPalette);
}

TEST_F(RetainedStreamout, UnseenFrameMarksFresh) {
  use_palette({0});
  const auto frame = [&](float x, bool drawn) {
    begin_frame();
    gx::g_gxState.pnMtx[0].pos = affine(x, 0.f, -5.f);
    if (drawn) {
      object_draw(0x190);
    }
    EXPECT_NE(stage(), nullptr);
    const BlockView view = block();
    end();
    return view.at(block::RollJobCount) == 0 ? 0u : roll_flags(view, 0);
  };
  EXPECT_EQ(frame(0.f, true), block::RollPalette | block::RollFresh);
  EXPECT_EQ(frame(1.f, true), block::RollPalette);
  EXPECT_EQ(frame(2.f, false), 0u);
  EXPECT_EQ(frame(3.f, true), block::RollPalette | block::RollFresh);
  EXPECT_EQ(frame(4.f, true), block::RollPalette);
}

TEST_F(RetainedStreamout, PaletteBytesShareWithPreviousPiece) {
  use_palette({0, 1});
  gx::g_gxState.pnMtx[1].pos = affine(1.f, 2.f, -5.f);
  begin_frame();
  begin_object(0x1A0);
  draw(3);
  draw(4);
  end_object();
  ASSERT_NE(stage(), nullptr);
  const BlockView view = block();
  ASSERT_EQ(view.pieces(), 2u);
  EXPECT_NE(view.piece(0, block::PieceInstance), view.piece(1, block::PieceInstance));
  EXPECT_EQ(view.piece(0, block::PieceMatrixWord), 0u);
  EXPECT_EQ(view.piece(1, block::PieceMatrixWord), 0u);
  EXPECT_EQ(view.at(block::WriteJobsWord) - view.at(block::MatricesWord), palette::BankWords);
  end();
  EXPECT_EQ(stats().paletteShared, 1u);
  EXPECT_EQ(stats().paletteStreamed, 2u);
}

TEST(RetainedMapPick, SelectsFirstColourMapAndLastAlphaMap) {
  gx::GXState state{};
  state.numTevStages = 3;
  for (uint8_t i = 0; i < 3; ++i) {
    state.tevStages[i].texMapId = static_cast<GXTexMapID>(GX_TEXMAP0 + i);
    state.tevStages[i].texCoordId = static_cast<GXTexCoordID>(GX_TEXCOORD0 + i);
  }
  state.tevStages[0].colorPass.b = GX_CC_TEXC;
  state.tevStages[1].colorPass.a = GX_CC_TEXA;
  state.tevStages[1].alphaPass.a = GX_CA_TEXA;
  state.tevStages[2].alphaPass.d = GX_CA_TEXA;
  state.tevStages[2].tevSwapTex = GX_TEV_SWAP1;
  state.tevSwapTable[1].alpha = GX_CH_RED;
  state.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1};
  state.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_TEX0};
  state.tcgs[2] = {.type = GX_TG_MTX2x4, .src = GX_TG_POS};

  map_pick::MapSelection maps = map_pick::select_maps(state);
  EXPECT_FALSE(maps.alphaTest);
  EXPECT_FALSE(maps.alpha.valid());
  ASSERT_TRUE(maps.albedo.valid());
  EXPECT_EQ(maps.albedo.stage, 0u);
  EXPECT_EQ(maps.albedo.map, GX_TEXMAP0);
  EXPECT_TRUE(maps.albedo.texcoordSource);
  EXPECT_EQ(maps.albedo.srcTex, 1u);

  state.alphaCompare.comp0 = GX_GREATER;
  maps = map_pick::select_maps(state);
  EXPECT_TRUE(maps.alphaTest);
  ASSERT_TRUE(maps.alpha.valid());
  EXPECT_EQ(maps.alpha.stage, 2u);
  EXPECT_EQ(maps.alpha.coord, GX_TEXCOORD2);
  EXPECT_FALSE(maps.alpha.texcoordSource);
  EXPECT_EQ(maps.alpha.channel, uint8_t{GX_CH_RED});

  const int copy = 0;
  state.loadedTextures[0].data = &copy;
  state.copyTextures[&copy] = {};
  maps = map_pick::select_maps(state);
  ASSERT_TRUE(maps.albedo.valid());
  EXPECT_EQ(maps.albedo.stage, 1u);
  EXPECT_EQ(maps.albedo.srcTex, 0u);
}
} // namespace
} // namespace aurora::render::scene::streamout

#include <gtest/gtest.h>

#include "gx/draw_route.hpp"
#include "gx/gx.hpp"
#include "gx/hints.hpp"
#include "gx/pipeline.hpp"
#include "render/frame_state.hpp"
#include "render/hints.hpp"
#include "render/scene/classify.hpp"
#include "render/scene/identity.hpp"
#include "render/settings.hpp"
#include "render/stage_data.hpp"
#include "render_test_caps.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace aurora::render::scene::retained {
namespace {
constexpr uint32_t Stride = 12;

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

Mat3x4<float> rotate_y(float radians, float x, float y, float z) {
  const float c = std::cos(radians);
  const float s = std::sin(radians);
  return {{c, 0.f, s, x}, {0.f, 1.f, 0.f, y}, {-s, 0.f, c, z}};
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

double translation_distance(const Affine& a, double x, double y, double z) {
  const double dx = a.m[0][3] - x;
  const double dy = a.m[1][3] - y;
  const double dz = a.m[2][3] - z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

class RenderRetainedTest : public ::testing::Test {
protected:
  void SetUp() override {
    gx::hints::testing::reset_counters();
    detail::testing::reset();
    detail::testing::set_caps(test::full_caps());
    detail::testing::set_hints_override(1);
    detail::install_hints();
    gx::g_gxState = {};
    gx::g_gxState.pnMtx[0].pos = affine();
    testing::reset();
    set_demand(Demand::Tool, true);
    m_fifo.assign(1 << 16, 0);
    m_config.shaderConfig.vtxStride = Stride;
    m_config.shaderConfig.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  }

  void TearDown() override {
    testing::reset();
    detail::remove_hints();
    detail::testing::reset_hints();
    scene::testing::reset();
    detail::testing::reset();
    gx::g_gxState = {};
  }

  static const gx::hints::Sink& sink() { return *gx::hints::sink(); }

  void begin_frame(const Mat3x4<float>& view = affine()) {
    detail::begin_hint_frame();
    latch_gate(Settings{}, test::full_caps());
    scene::apply_frame_state(detail::FrameState{});
    const gx::hints::Camera main{
        .role = AURORA_CAMERA_MAIN,
        .view = view,
        .proj = perspective(),
        .type = GX_PERSPECTIVE,
        .gxProj = gx::hints::gx_projection(perspective(), GX_PERSPECTIVE),
    };
    sink().camera(sink().userData, main);
    sink().stage(sink().userData, Stage::SceneBegin);
    gx::g_gxState.proj = main.gxProj;
    m_offset = 0;
  }

  static void begin_object(uint64_t id, uint32_t categories = AURORA_CAT_DYNAMIC, uint32_t flags = 0,
                           uint32_t subId = 0, const Mat3x4<float>* world = nullptr) {
    sink().objectBegin(sink().userData, {.id = id,
                                         .subId = subId,
                                         .generation = 1,
                                         .categories = categories,
                                         .flags = flags,
                                         .hasWorld = world != nullptr,
                                         .world = world != nullptr ? *world : Mat3x4<float>{}});
  }
  static void end_object() { sink().objectEnd(sink().userData); }

  uint32_t draw(uint32_t vtxCount = 3, uint8_t fill = 0, bool merge = false, bool captureOnly = false) {
    const uint32_t size = vtxCount * Stride;
    std::fill_n(m_fifo.begin() + m_offset, size, fill);
    const gx::draw_route::Query query{
        .prim = GX_TRIANGLES,
        .fmt = GX_VTXFMT0,
        .vtxCount = vtxCount,
        .vertices = std::span<const uint8_t>{m_fifo.data() + m_offset, size},
    };
    const gx::draw_route::Router* router = gx::draw_route::router();
    router->route(router->userData, query);
    if (captureOnly) {
      testing::route_capture_only(scene::testing::classifier().last(), query);
    }
    const uint32_t offset = m_offset;
    m_offset += size;
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

  void object_draw(uint64_t id, uint32_t categories = AURORA_CAT_DYNAMIC, uint32_t flags = 0, uint8_t fill = 0) {
    begin_object(id, categories, flags);
    draw(3, fill);
    end_object();
  }

  static std::shared_ptr<const Frame> seal_stage(Stage stage = Stage::OpaqueDone) {
    detail::StageData data{.stage = stage};
    prepare(data);
    seal(data);
    return data.retained;
  }

  static FrameCounts end() {
    scene::seal_frame();
    return frame_counts();
  }

  std::vector<uint8_t> m_fifo;
  uint32_t m_offset = 0;
  uint32_t m_record = 0;
  gx::DrawData m_command{};
  gx::PipelineConfig m_config{};
  gx::ShaderInfo m_info{};
};

TEST_F(RenderRetainedTest, Keys) {
  std::array<uint32_t, 3> slots{};
  for (uint32_t frame = 0; frame < 3; ++frame) {
    begin_frame();
    begin_object(0x10);
    draw(3, 1);
    const uint32_t follower = draw(3, 2, true);
    draw(6, 3);
    end_object();
    const auto sealed = seal_stage();
    ASSERT_NE(sealed, nullptr);
    ASSERT_EQ(sealed->entries.size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
      const Entry& entry = sealed->entries[i];
      ASSERT_NE(entry.slot, TransientSlot);
      EXPECT_EQ(testing::instance(entry.slot).info.key.ordinal, i);
      if (frame == 0) {
        slots[i] = entry.slot;
      } else {
        EXPECT_EQ(entry.slot, slots[i]);
      }
    }
    const Entry& merged = sealed->entries[1];
    EXPECT_EQ(merged.vertRange.offset, follower);
    EXPECT_EQ(merged.vertRange.size, 3 * Stride);
    EXPECT_EQ(merged.immediates.vtxStart, follower);
    EXPECT_EQ(merged.record, sealed->entries[0].record);
    EXPECT_EQ(sealed->entries[2].vtxCount, 6u);
    const FrameCounts counts = end();
    EXPECT_EQ(counts.created, frame == 0 ? 3u : 0u);
    EXPECT_EQ(counts.entries, 3u);
    EXPECT_EQ(counts.rekeyed, 0u);
    EXPECT_EQ(counts.instances, 3u);
  }
  EXPECT_EQ(stats().instances, 3u);
  EXPECT_EQ(stats().created, 3u);
}

TEST_F(RenderRetainedTest, Occurrence) {
  std::array<uint32_t, 2> slots{};
  for (uint32_t frame = 0; frame < 2; ++frame) {
    begin_frame();
    object_draw(0x20);
    object_draw(0x20);
    const auto sealed = seal_stage();
    ASSERT_NE(sealed, nullptr);
    ASSERT_EQ(sealed->entries.size(), 2u);
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t slot = sealed->entries[i].slot;
      EXPECT_EQ(testing::instance(slot).info.key.occurrence, i);
      EXPECT_EQ(testing::instance(slot).info.key.ordinal, 0u);
      if (frame == 0) {
        slots[i] = slot;
      } else {
        EXPECT_EQ(slot, slots[i]);
      }
    }
    const FrameCounts counts = end();
    EXPECT_EQ(counts.created, frame == 0 ? 2u : 0u);
    EXPECT_EQ(counts.newOccurrence, frame == 0 ? 1u : 0u);
    EXPECT_EQ(counts.rekeyed, 0u);
  }
}

TEST_F(RenderRetainedTest, Rekeyed) {
  begin_frame();
  begin_object(0x30);
  draw();
  draw();
  end_object();
  EXPECT_EQ(end().created, 2u);

  begin_frame();
  begin_object(0x30);
  draw();
  draw();
  draw();
  end_object();
  FrameCounts counts = end();
  EXPECT_EQ(counts.created, 1u);
  EXPECT_EQ(counts.rekeyed, 1u);
  EXPECT_EQ(counts.newSubId, 0u);

  begin_frame();
  begin_object(0x30, AURORA_CAT_DYNAMIC, 0, 1);
  draw();
  end_object();
  begin_object(0x30);
  draw();
  draw();
  draw();
  end_object();
  counts = end();
  EXPECT_EQ(counts.created, 1u);
  EXPECT_EQ(counts.rekeyed, 0u);
  EXPECT_EQ(counts.newSubId, 1u);
}

TEST_F(RenderRetainedTest, Latch) {
  const auto immobile = [&](uint64_t id, const Mat3x4<float>& pn, uint32_t flags = AURORA_OBJ_IMMOBILE) {
    gx::g_gxState.pnMtx[0].pos = pn;
    begin_object(id, AURORA_CAT_STATIC, flags);
    draw();
    end_object();
  };
  begin_frame();
  immobile(0x40, affine(1e5f, 0.f, 0.f));
  EXPECT_EQ(end().created, 1u);
  const uint32_t slot = testing::find({.id = 0x40, .generation = 1});
  ASSERT_NE(slot, TransientSlot);
  EXPECT_TRUE(testing::instance(slot).latched);

  begin_frame();
  immobile(0x40, affine(1e5f + 0.05f, 0.f, 0.f));
  EXPECT_EQ(end().relatched, 0u);
  EXPECT_LT(translation_distance(testing::instance(slot).world, 1e5, 0.0, 0.0), 1e-9);

  begin_frame();
  immobile(0x40, affine(1e5f + 0.2f, 0.f, 0.f));
  EXPECT_EQ(end().relatched, 1u);
  EXPECT_GT(translation_distance(testing::instance(slot).world, 1e5, 0.0, 0.0), 0.15);

  Mat3x4<float> drift = affine(10.f, 0.f, 0.f);
  begin_frame();
  immobile(0x41, drift);
  end();
  drift.m0[0] = 1.f + 2e-4f;
  begin_frame();
  immobile(0x41, drift);
  EXPECT_EQ(end().relatched, 1u);

  begin_frame();
  immobile(0x42, affine(3.f, 0.f, 0.f), AURORA_OBJ_IMMOBILE | AURORA_OBJ_BILLBOARD);
  end();
  const uint32_t billboard = testing::find({.id = 0x42, .generation = 1});
  ASSERT_NE(billboard, TransientSlot);
  EXPECT_FALSE(testing::instance(billboard).latched);
  begin_frame();
  immobile(0x42, affine(8.f, 0.f, 0.f), AURORA_OBJ_IMMOBILE | AURORA_OBJ_BILLBOARD);
  EXPECT_EQ(end().relatched, 0u);
}

TEST_F(RenderRetainedTest, MapTransLatch) {
  const Mat3x4<float> model = rotate_y(std::numbers::pi_v<float> / 6.f, 1e3f, 0.f, 1e3f);
  const float yaw = std::numbers::pi_v<float> / 4.f;
  uint32_t relatched = 0;
  for (uint32_t frame = 0; frame < 120; ++frame) {
    const float jitter = static_cast<float>(frame % 3) - 1.f;
    const std::array<float, 3> eye{70710.68f + jitter, 0.f, 70710.68f};
    const Mat3x4<float> rotation = rotate_y(yaw, 0.f, 0.f, 0.f);
    Mat3x4<float> view = rotation;
    view.m0[3] = -(rotation.m0[0] * eye[0] + rotation.m0[1] * eye[1] + rotation.m0[2] * eye[2]);
    view.m1[3] = -(rotation.m1[0] * eye[0] + rotation.m1[1] * eye[1] + rotation.m1[2] * eye[2]);
    view.m2[3] = -(rotation.m2[0] * eye[0] + rotation.m2[1] * eye[1] + rotation.m2[2] * eye[2]);
    begin_frame(view);
    gx::g_gxState.pnMtx[0].pos = concat(view, model);
    begin_object(0x50, AURORA_CAT_STATIC | AURORA_CAT_TERRAIN, AURORA_OBJ_IMMOBILE);
    draw();
    end_object();
    relatched += end().relatched;
  }
  EXPECT_EQ(relatched, 0u);
  const uint32_t slot = testing::find({.id = 0x50, .generation = 1});
  ASSERT_NE(slot, TransientSlot);
  EXPECT_LT(translation_distance(testing::instance(slot).world, 1e3, 0.0, 1e3), 0.1);
}

TEST_F(RenderRetainedTest, StaticSignature) {
  std::array<uint8_t, 96> positions{};
  gx::g_gxState.vtxDesc[GX_VA_POS] = GX_INDEX16;
  gx::g_gxState.arrays[GX_VA_POS] = {
      .data = positions.data(), .size = static_cast<uint32_t>(positions.size()), .stride = 12};
  m_config.shaderConfig.attrs[GX_VA_POS] = {.attrType = GX_INDEX16, .cnt = 3, .compType = GX_F32, .stride = 12};
  m_config.shaderConfig.vtxStride = 2;
  const auto frame = [&](uint8_t fill) {
    begin_frame();
    object_draw(0x60, AURORA_CAT_STATIC, 0, fill);
    return end();
  };
  EXPECT_EQ(frame(1).created, 1u);
  FrameCounts counts = frame(1);
  EXPECT_EQ(counts.staticRestream, 0u);
  EXPECT_EQ(counts.signatureMisses, 0u);
  EXPECT_EQ(frame(2).staticRestream, 1u);
  positions[5] = 7;
  EXPECT_EQ(frame(2).staticRestream, 0u);
  testing::set_check(true);
  positions[6] = 9;
  counts = frame(2);
  EXPECT_EQ(counts.staticRestream, 1u);
  EXPECT_EQ(counts.signatureMisses, 1u);
  counts = frame(2);
  EXPECT_EQ(counts.staticRestream, 0u);
  EXPECT_EQ(counts.signatureMisses, 0u);
}

TEST_F(RenderRetainedTest, Origin) {
  const Mat3x4<float> v2 = affine(100.f, 200.f, 300.f);
  begin_frame();
  gx::g_gxState.pnMtx[0].pos = affine(110.f, 200.f, 300.f);
  begin_object(0x70, AURORA_CAT_DYNAMIC, 0, 0, &v2);
  draw();
  end_object();
  gx::g_gxState.pnMtx[0].pos = affine(100.f, 0.f, 0.f);
  draw();
  auto sealed = seal_stage();
  ASSERT_NE(sealed, nullptr);
  ASSERT_EQ(sealed->entries.size(), 2u);
  const Entry& object = sealed->entries[0];
  ASSERT_NE(object.slot, TransientSlot);
  EXPECT_EQ(testing::instance(object.slot).info.origin, (std::array<double, 3>{100.0, 200.0, 300.0}));
  EXPECT_FLOAT_EQ(object.worldRel.m0[3], 10.f);
  EXPECT_FLOAT_EQ(object.worldRel.m1[3], 0.f);
  const Entry& loose = sealed->entries[1];
  EXPECT_EQ(loose.slot, TransientSlot);
  EXPECT_NE(loose.flags & entry_flag::Transient, 0u);
  EXPECT_FLOAT_EQ(loose.worldRel.m0[3], 100.f);
  EXPECT_EQ(end().transient, 1u);

  begin_frame();
  gx::g_gxState.pnMtx[0].pos = affine(0.f, 0.f, 0.f);
  object_draw(0x71);
  ASSERT_NE(seal_stage(), nullptr);
  EXPECT_EQ(end().reorigins, 0u);
  begin_frame();
  gx::g_gxState.pnMtx[0].pos = affine(2100.f, 0.f, 0.f);
  begin_object(0x71);
  draw();
  end_object();
  sealed = seal_stage();
  EXPECT_EQ(end().reorigins, 1u);
  ASSERT_NE(sealed, nullptr);
  const uint32_t slot = sealed->entries[0].slot;
  EXPECT_EQ(testing::instance(slot).info.origin, (std::array<double, 3>{2100.0, 0.0, 0.0}));
  EXPECT_FLOAT_EQ(sealed->entries[0].worldRel.m0[3], 0.f);
  bool reorigin = false;
  for (const Event& event : sealed->events) {
    reorigin = reorigin || (event.type == EventType::Reorigin && event.slot == slot);
  }
  EXPECT_TRUE(reorigin);

  const auto classified = [&] {
    begin_frame();
    object_draw(0x72);
    object_draw(0x72);
    draw();
    const auto frame = seal_stage();
    const FrameStats& stats = scene::testing::classifier().stats();
    const std::array<uint32_t, 3> counts{stats.draws, stats.mergeBreaks, stats.captureCandidates};
    end();
    return std::pair{frame, counts};
  };
  const auto on = classified();
  set_demand(Demand::Tool, false);
  const auto off = classified();
  EXPECT_NE(on.first, nullptr);
  EXPECT_EQ(off.first, nullptr);
  EXPECT_EQ(on.second, off.second);
  EXPECT_FALSE(active());
  EXPECT_EQ(frame_counts().entries, 0u);
  EXPECT_FALSE(stats().active);
}

TEST_F(RenderRetainedTest, Delivery) {
  begin_frame();
  object_draw(0x80, AURORA_CAT_STATIC);
  ASSERT_NE(seal_stage(), nullptr);
  FrameCounts counts = end();
  EXPECT_EQ(counts.undeliveredStatics, 0u);
  EXPECT_EQ(counts.delivered, 1u);
  EXPECT_TRUE(testing::instance(testing::find({.id = 0x80, .generation = 1})).delivered);

  testing::set_overrun(1);
  begin_frame();
  begin_object(0x81, AURORA_CAT_STATIC);
  draw(3, 0, false, true);
  end_object();
  EXPECT_EQ(seal_stage(), nullptr);
  counts = end();
  EXPECT_EQ(counts.captureOnly, 1u);
  EXPECT_EQ(counts.undeliveredStatics, 1u);
  EXPECT_EQ(counts.lostCaptures, 1u);
  EXPECT_EQ(stats().lostCaptures, 1u);
  EXPECT_EQ(stats().undeliveredStatics, 1u);
  testing::set_overrun(0);

  begin_frame();
  object_draw(0x82);
  detail::StageData opaque{.stage = Stage::OpaqueDone};
  prepare(opaque);
  skip(opaque);
  EXPECT_EQ(opaque.retained, nullptr);
  sink().stage(sink().userData, Stage::OpaqueDone);
  object_draw(0x83, AURORA_CAT_DYNAMIC, 0, 4);
  const auto late = seal_stage(Stage::PrePost);
  ASSERT_NE(late, nullptr);
  ASSERT_EQ(late->entries.size(), 2u);
  EXPECT_EQ(late->stage, Stage::PrePost);
  EXPECT_EQ(late->entries[0].flags & entry_flag::Late, 0u);
  EXPECT_NE(late->entries[1].flags & entry_flag::Late, 0u);
  counts = end();
  EXPECT_EQ(counts.delivered, 2u);
  EXPECT_EQ(counts.late, 1u);
}

TEST_F(RenderRetainedTest, EventBound) {
  std::vector<std::pair<uint32_t, uint32_t>> created;
  uint64_t id = 0x1000;
  for (uint32_t frame = 0; frame < 10; ++frame) {
    begin_frame();
    std::vector<uint32_t> slots;
    for (uint32_t i = 0; i < 10; ++i) {
      object_draw(++id, AURORA_CAT_STATIC);
      slots.push_back(testing::find({.id = id, .generation = 1}));
    }
    end();
    for (uint32_t i = 0; i < slots.size(); ++i) {
      if (i < 3) {
        testing::evict(slots[i], EvictReason::Epoch);
      } else {
        created.emplace_back(slots[i], testing::instance(slots[i]).slotGeneration);
      }
    }
  }
  begin_frame();
  auto sealed = seal_stage();
  end();
  ASSERT_NE(sealed, nullptr);
  ASSERT_EQ(sealed->events.size(), created.size() + 1);
  EXPECT_EQ(sealed->events[0].type, EventType::Reset);
  for (size_t i = 0; i < created.size(); ++i) {
    const Event& event = sealed->events[i + 1];
    EXPECT_EQ(event.type, EventType::Created);
    EXPECT_EQ(std::pair(event.slot, event.slotGeneration), created[i]);
    EXPECT_EQ(event.info.slot, event.slot);
  }

  testing::evict(created.front().first, EvictReason::Epoch);
  begin_frame();
  sealed = seal_stage();
  end();
  ASSERT_NE(sealed, nullptr);
  ASSERT_EQ(sealed->events.size(), 1u);
  EXPECT_EQ(sealed->events[0].type, EventType::Evicted);
  EXPECT_EQ(sealed->events[0].reason, EvictReason::Epoch);
  EXPECT_EQ(sealed->events[0].slot, created.front().first);

  const auto immobile = [&](uint32_t frame) {
    for (uint32_t i = 0; i < 100; ++i) {
      gx::g_gxState.pnMtx[0].pos = affine(static_cast<float>(i), (frame & 1) != 0 ? 0.5f : 0.f, 0.f);
      object_draw(0x9000 + i, AURORA_CAT_STATIC, AURORA_OBJ_IMMOBILE);
    }
  };
  begin_frame();
  immobile(0);
  ASSERT_NE(seal_stage(), nullptr);
  end();
  uint32_t resyncs = 0;
  for (uint32_t frame = 1; frame <= 300; ++frame) {
    begin_frame();
    immobile(frame);
    if (frame == 150) {
      object_draw(0xA000);
    }
    resyncs += end().resyncs;
    if (frame == 150) {
      testing::evict(testing::find({.id = 0xA000, .generation = 1}), EvictReason::Dynamic);
    }
  }
  EXPECT_EQ(resyncs, 1u);
  begin_frame();
  sealed = seal_stage();
  const FrameCounts counts = end();
  ASSERT_NE(sealed, nullptr);
  ASSERT_EQ(sealed->events.size(), size_t{counts.instances} + 1);
  EXPECT_EQ(sealed->events[0].type, EventType::Reset);
  for (size_t i = 1; i < sealed->events.size(); ++i) {
    EXPECT_EQ(sealed->events[i].type, EventType::Created);
    EXPECT_NE(sealed->events[i].info.key.id, 0xA000u);
  }
  EXPECT_EQ(stats().resyncs, 1u);
}
} // namespace
} // namespace aurora::render::scene::retained

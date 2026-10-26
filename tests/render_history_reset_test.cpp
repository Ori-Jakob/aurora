#include <gtest/gtest.h>

#include "gx/hints.hpp"
#include "render/graph.hpp"
#include "render/hints.hpp"
#include "render/history_reset.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace aurora::render::detail {
namespace {
using history_reset::Camera;
using history_reset::classify;
using history_reset::Reason;

Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

gx::hints::Camera camera_at(float x, float y, float z, float yawDegrees = 0.f, uint32_t cutEpoch = 0) {
  const float yaw = yawDegrees * std::numbers::pi_v<float> / 180.f;
  const float fx = std::sin(yaw);
  const float fz = -std::cos(yaw);
  const Mat4x4<float> proj = perspective();
  return {
      .role = AURORA_CAMERA_MAIN,
      .view = {{-fz, 0.f, fx, -(-fz * x + fx * z)}, {0.f, 1.f, 0.f, -y}, {-fx, 0.f, -fz, -(-fx * x - fz * z)}},
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .cutEpoch = cutEpoch,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
}

SceneHints hints_for(const gx::hints::Camera& camera) {
  SceneHints hints;
  hints.present = SceneHasMainCamera;
  hints.mainCamera = camera;
  return hints;
}

Camera tracked(const gx::hints::Camera& camera) {
  Camera next;
  EXPECT_EQ(classify({}, hints_for(camera), next), Reason::CameraReacquired);
  EXPECT_TRUE(next.valid);
  return next;
}

Reason step(const gx::hints::Camera& from, const SceneHints& to) {
  Camera next;
  return classify(tracked(from), to, next);
}

TEST(HistoryReset, ReacquiredResets) {
  Camera next;
  EXPECT_EQ(classify({}, hints_for(camera_at(10.f, 20.f, 30.f, 15.f)), next), Reason::CameraReacquired);
  ASSERT_TRUE(next.valid);
  EXPECT_NEAR(next.eye.x, 10.f, 1e-3f);
  EXPECT_NEAR(next.eye.y, 20.f, 1e-3f);
  EXPECT_NEAR(next.eye.z, 30.f, 1e-3f);
  EXPECT_NEAR(next.forward.x, std::sin(15.f * std::numbers::pi_v<float> / 180.f), 1e-5f);
  EXPECT_NEAR(next.forward.z, -std::cos(15.f * std::numbers::pi_v<float> / 180.f), 1e-5f);
}

TEST(HistoryReset, CutEpochResets) {
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f, 0.f, 3), hints_for(camera_at(0.f, 0.f, 0.f, 0.f, 4))), Reason::CutEpoch);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f, 0.f, 3), hints_for(camera_at(0.f, 0.f, 0.f, 0.f, 3))), Reason::None);
}

TEST(HistoryReset, SceneEpochResets) {
  SceneHints hints = hints_for(camera_at(0.f, 0.f, 0.f));
  hints.present |= SceneHasEpoch;
  hints.sceneEpochChanged = true;
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::SceneEpoch);
  hints.sceneEpochChanged = false;
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::None);
}

TEST(HistoryReset, JumpOverTenMetresResets) {
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(1001.f, 0.f, 0.f))), Reason::CameraJump);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(600.f, 0.f, 800.5f))), Reason::CameraJump);
}

TEST(HistoryReset, NineMetresKeeps) {
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(900.f, 0.f, 0.f))), Reason::None);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(0.f, 999.f, 0.f))), Reason::None);
}

TEST(HistoryReset, FortyFiveDegreeTurnKeeps) {
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f, 10.f), hints_for(camera_at(0.f, 0.f, 0.f, 55.f))), Reason::None);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(0.f, 0.f, 0.f, 59.f))), Reason::None);
}

TEST(HistoryReset, TurnOverSixtyDegreesResets) {
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints_for(camera_at(0.f, 0.f, 0.f, 61.f))), Reason::CameraJump);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f, 90.f), hints_for(camera_at(0.f, 0.f, 0.f, -90.f))), Reason::CameraJump);
}

TEST(HistoryReset, EnvironmentUnitsScaleJump) {
  SceneHints hints = hints_for(camera_at(150.f, 0.f, 0.f));
  hints.present |= SceneHasEnvironment;
  hints.environment.value.unitsPerMeter = 10.f;
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::CameraJump);
  hints.mainCamera = camera_at(90.f, 0.f, 0.f);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::None);
}

TEST(HistoryReset, MissingEnvironmentUses100) {
  SceneHints hints = hints_for(camera_at(150.f, 0.f, 0.f));
  hints.environment.value.unitsPerMeter = 10.f;
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::None);
  hints.present |= SceneHasEnvironment;
  hints.environment.value.unitsPerMeter = 0.f;
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::None);
  hints.mainCamera = camera_at(1001.f, 0.f, 0.f);
  EXPECT_EQ(step(camera_at(0.f, 0.f, 0.f), hints), Reason::CameraJump);
}

TEST(HistoryReset, NoCameraNeverResets) {
  SceneHints hints = hints_for(camera_at(5000.f, 0.f, 0.f, 0.f, 9));
  hints.present = SceneHasEnvironment | SceneHasEpoch;
  hints.sceneEpochChanged = true;
  Camera next;
  EXPECT_EQ(classify({}, hints, next), Reason::None);
  EXPECT_FALSE(next.valid);
  EXPECT_EQ(classify(tracked(camera_at(0.f, 0.f, 0.f)), hints, next), Reason::None);
  EXPECT_FALSE(next.valid);
  Camera reacquired;
  EXPECT_EQ(classify(next, hints_for(camera_at(0.f, 0.f, 0.f)), reacquired), Reason::CameraReacquired);
  EXPECT_TRUE(reacquired.valid);
}

class HistoryResetFrames : public ::testing::Test {
protected:
  void SetUp() override {
    testing::set_hints_override(1);
    install_hints();
    history_reset::reset();
  }

  void TearDown() override {
    testing::reset_hints();
    history_reset::reset();
  }

  static Reason frame(const CompiledGraph& graph, uint64_t frameId, const gx::hints::Camera& camera) {
    begin_hint_frame();
    const gx::hints::Sink& sink = *gx::hints::sink();
    sink.camera(sink.userData, camera);
    return history_reset::begin_frame(graph, frameId);
  }
};

TEST_F(HistoryResetFrames, EmptyHistoriesClearState) {
  CompiledGraph withHistory;
  withHistory.histories.push_back(0);
  const CompiledGraph empty;
  EXPECT_EQ(frame(withHistory, 1, camera_at(0.f, 0.f, 0.f)), Reason::CameraReacquired);
  EXPECT_EQ(frame(withHistory, 2, camera_at(1.f, 0.f, 0.f)), Reason::None);
  EXPECT_EQ(frame(empty, 3, camera_at(2.f, 0.f, 0.f)), Reason::None);
  EXPECT_FALSE(scene_hints_frozen());
  EXPECT_EQ(history_reset::stats().frames, 2u);
  EXPECT_EQ(frame(withHistory, 4, camera_at(3.f, 0.f, 0.f)), Reason::CameraReacquired);
  EXPECT_EQ(frame(withHistory, 5, camera_at(3.f, 0.f, 0.f, 0.f, 1)), Reason::CutEpoch);
  const history_reset::Stats stats = history_reset::stats();
  EXPECT_EQ(stats.frames, 4u);
  EXPECT_EQ(stats.reasons[static_cast<size_t>(Reason::CameraReacquired)], 2u);
  EXPECT_EQ(stats.reasons[static_cast<size_t>(Reason::CutEpoch)], 1u);
  EXPECT_EQ(stats.reasons[static_cast<size_t>(Reason::None)], 1u);
}
} // namespace
} // namespace aurora::render::detail

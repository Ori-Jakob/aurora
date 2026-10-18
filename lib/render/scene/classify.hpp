#pragma once

#include "classifier.hpp"

#include "../../gx/hints.hpp"

#include <dolphin/gx/GXEnum.h>

#include <memory>
#include <vector>

namespace aurora::render::detail {
struct FrameState;
} // namespace aurora::render::detail

namespace aurora::render::scene {
struct FrameReport {
  FrameStats stats;
  bool mainCamera = false;
  uint32_t cutEpoch = 0;
  bool sceneEpochKnown = false;
  uint32_t sceneEpoch = 0;
  std::vector<ClassifiedDraw> draws;
  std::vector<ObjectStats> objects;
};

struct ReportRequest {
  bool records = false;
  bool objects = false;
};

// Game thread while the FIFO is idle
void install() noexcept;
void remove() noexcept;
void reset() noexcept;
void begin_frame() noexcept;
void apply_frame_state(const detail::FrameState& state) noexcept;
void seal_frame() noexcept;
void request_report(const ReportRequest& request) noexcept;

// FIFO thread
[[nodiscard]] bool identity_collecting() noexcept;
void on_camera(const gx::hints::Camera& camera) noexcept;
void on_object_begin(const gx::hints::ObjectBegin& object) noexcept;
void on_object_end() noexcept;
void on_scope_push(const gx::hints::ScopePush& scope) noexcept;
void on_scope_pop() noexcept;
void on_stage(Stage stage) noexcept;
void on_material(uint32_t materialClass) noexcept;
void on_tev_policy(const AuroraTevPolicy& policy) noexcept;

// Any thread
[[nodiscard]] std::shared_ptr<const FrameReport> latest_report() noexcept;

[[nodiscard]] bool blended(GXBlendMode mode, GXBlendFactor src, GXBlendFactor dst, GXLogicOp op) noexcept;

namespace testing {
Classifier& classifier() noexcept;
void set_identity_override(int value) noexcept;
void set_identity_env(int value) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::scene

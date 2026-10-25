#pragma once

#include "classifier.hpp"

#include "../../gx/hints.hpp"

#include <dolphin/gx/GXEnum.h>

#include <memory>
#include <vector>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

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

struct ViewHooks {
  void (*beginFrame)(const detail::FrameState& state) noexcept = nullptr;
  bool (*push)(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
               const DrawClass& cls) noexcept = nullptr;
};

struct ObserverHooks {
  void (*observe)(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                  const DrawClass& cls) noexcept = nullptr;
};

// Game thread while the FIFO is idle
void install() noexcept;
void remove() noexcept;
void reset() noexcept;
void begin_frame() noexcept;
void apply_frame_state(const detail::FrameState& state) noexcept;
void seal_frame() noexcept;
void request_report(const ReportRequest& request) noexcept;
void set_view_hooks(const ViewHooks* hooks) noexcept;
void set_observer_hooks(const ObserverHooks* hooks) noexcept;
void set_identity_demand(bool demand) noexcept;

// FIFO thread
[[nodiscard]] bool identity_collecting() noexcept;
[[nodiscard]] ViewCounts& view_counts() noexcept;
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

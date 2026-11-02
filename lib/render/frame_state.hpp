#pragma once

#include <aurora/render.hpp>

#include <cstdint>
#include <memory>

namespace aurora::render::detail {
struct CompiledGraph;

struct FrameState {
  uint64_t settingsGeneration = 0;
  Settings settings;
  std::shared_ptr<const CompiledGraph> graph;
  bool inspectorOpen = false;
  // The view the graph was compiled for, so per-frame view terms never disagree with its declares
  uint16_t debugView = 0;
};

std::shared_ptr<const FrameState> latch_frame();
const std::shared_ptr<const FrameState>& current_frame_state() noexcept;

namespace testing {
uint64_t settings_generation() noexcept;
} // namespace testing
} // namespace aurora::render::detail

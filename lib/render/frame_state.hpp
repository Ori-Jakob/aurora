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
};

std::shared_ptr<const FrameState> latch_frame();
const std::shared_ptr<const FrameState>& current_frame_state() noexcept;

namespace testing {
uint64_t settings_generation() noexcept;
} // namespace testing
} // namespace aurora::render::detail

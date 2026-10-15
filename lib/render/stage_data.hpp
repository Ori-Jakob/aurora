#pragma once

#include <aurora/render.hpp>

#include <cstdint>
#include <vector>

namespace aurora::render::detail {
struct FrameState;

struct StageData {
  Stage stage = Stage::SceneBegin;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  bool markerSeen = false;
  bool deferred = false;
  bool flushedAtFinish = false;
  const FrameState* frame = nullptr;
  uint32_t passCount = 0;
  std::vector<uint64_t> enabled;

  [[nodiscard]] bool is_enabled(uint32_t index) const noexcept {
    return index < passCount && ((enabled[index / 64] >> (index % 64)) & 1u) != 0;
  }
  [[nodiscard]] bool any_enabled() const noexcept {
    for (const auto word : enabled) {
      if (word != 0) {
        return true;
      }
    }
    return false;
  }
};
} // namespace aurora::render::detail

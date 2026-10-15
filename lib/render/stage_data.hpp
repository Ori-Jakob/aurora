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
  std::vector<uint64_t> dummies;
  std::vector<uint64_t> prevValid;

  [[nodiscard]] bool is_enabled(uint32_t index) const noexcept {
    return index < passCount && ((enabled[index / 64] >> (index % 64)) & 1u) != 0;
  }
  [[nodiscard]] bool dummy_bound(uint32_t stageUse) const noexcept {
    return stageUse / 64 < dummies.size() && ((dummies[stageUse / 64] >> (stageUse % 64)) & 1u) != 0;
  }
  [[nodiscard]] bool prev_valid(uint32_t historySlot) const noexcept {
    return historySlot / 64 < prevValid.size() && ((prevValid[historySlot / 64] >> (historySlot % 64)) & 1u) != 0;
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

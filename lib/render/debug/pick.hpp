#pragma once

#include "../graph.hpp"
#include "../graph_bind.hpp"

#include <aurora/gfx.hpp>
#include <aurora/math.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::debug::pick {
inline constexpr std::string_view PassName = "debug.texture_pick";
inline constexpr uint32_t WarmReadyPercent = 98;
inline constexpr double WarmTimeoutSeconds = 5.0;
inline constexpr uint32_t MaxRecordId = 0xFFFFFE;

enum class ArmState : uint8_t { Off, Warming, Armed };

struct CapabilityInputs {
  bool device = false;
  bool core = false;
  uint32_t msaaSamples = 1;
  bool hints = false;
  uint32_t colorBytes = 0;
  uint32_t colorBudget = 0;
};

struct Capability {
  bool available = false;
  std::string_view reason;
};

struct PixelPos {
  uint32_t x = 0;
  uint32_t y = 0;
  bool operator==(const PixelPos&) const = default;
};

struct Params {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t valid = 0;
};

struct WarmInputs {
  uint32_t ready = 0;
  uint32_t total = 0;
  double seconds = 0.0;
  uint32_t observedFrames = 0;
};

struct ArmInputs {
  bool wanted = false;
  bool warmComplete = false;
  bool baseLayoutChanged = false;
};

struct DecodedId {
  uint8_t sequence = 0;
  uint32_t record = 0;
};

constexpr uint32_t encode_id(uint64_t sequence, uint32_t record) noexcept {
  return static_cast<uint32_t>(sequence & 0xFF) << 24 | ((record + 1) & 0xFFFFFF);
}

constexpr std::optional<DecodedId> decode_id(uint32_t value) noexcept {
  if ((value & 0xFFFFFF) == 0) {
    return std::nullopt;
  }
  return DecodedId{static_cast<uint8_t>(value >> 24), (value & 0xFFFFFF) - 1};
}

Capability pick_capability(const CapabilityInputs& inputs) noexcept;
CapabilityInputs live_capability_inputs() noexcept;
std::optional<PixelPos> efb_pixel_at(Vec2<float> mouse, Vec2<float> fbScale, Vec2<uint32_t> surfaceSize,
                                     Vec2<uint32_t> efbSize) noexcept;
gfx::RenderTargetLayout armed_layout(const gfx::RenderTargetLayout& live) noexcept;
uint32_t layout_color_bytes(const gfx::RenderTargetLayout& layout) noexcept;
bool warm_complete(const WarmInputs& inputs) noexcept;
ArmState next_arm_state(ArmState current, const ArmInputs& inputs) noexcept;

bool resolve_gx_debug_id(const detail::StageData& stage, Access access, detail::GxBinding& out);

// Game thread
void initialize();
void shutdown() noexcept;
void ensure_pass();
bool pass_registered() noexcept;
void set_params(const Params& params);
// Any thread
std::optional<uint32_t> take_readback() noexcept;

namespace testing {
void set_readback(uint32_t value) noexcept;
} // namespace testing
} // namespace aurora::render::debug::pick

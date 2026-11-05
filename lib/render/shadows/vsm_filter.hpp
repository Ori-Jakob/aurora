#pragma once

#include "vsm_pages.hpp"

#include <aurora/render.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::shadows::vsm {
inline constexpr std::string_view FilterShader = "shadows/vsm_filter";
inline constexpr std::string_view BiasEnv = "AURORA_RENDER_VSM_BIAS";
inline constexpr uint32_t FallbackLevels = 2;
inline constexpr uint32_t MaxFilterTaps = 9;
inline constexpr float NormalOffsetBase = 0.5f;
inline constexpr float NormalOffsetSlope = 1.5f;
inline constexpr float DefaultDepthBias = 1.f;
inline constexpr float TapInset = 0.5f;

struct FilterParams {
  uint32_t taps = MaxFilterTaps;
  float normalScale = 1.f;
  float depthBias = DefaultDepthBias;
};

struct Bias {
  float normal = 1.f;
  float depth = DefaultDepthBias;
};

[[nodiscard]] std::optional<Bias> parse_bias(std::string_view text) noexcept;
[[nodiscard]] FilterParams filter_params(Quality quality, const std::optional<Bias>& bias = std::nullopt) noexcept;
[[nodiscard]] const std::optional<Bias>& env_bias() noexcept;
[[nodiscard]] uint32_t filter_taps(uint32_t taps) noexcept;
[[nodiscard]] std::span<const std::array<float, 2>> tap_offsets(uint32_t taps) noexcept;
[[nodiscard]] float normal_offset_texels(const FilterParams& params, float nDotL) noexcept;
[[nodiscard]] float tap_radius(uint32_t taps) noexcept;

struct ReceiverPage {
  std::array<float, 2> texel{};
  std::array<int32_t, 2> page{};
  std::array<float, 2> local{};
  float ref = 0.f;
  bool outOfRange = false;
};

struct TapPage {
  std::array<int32_t, 2> page{};
  std::array<float, 2> local{};
};

[[nodiscard]] float receiver_ref(float depth, bool& outOfRange) noexcept;
[[nodiscard]] ReceiverPage receiver_page(const ViewRecord& view, const std::array<float, 3>& pRel,
                                         float biasTexels) noexcept;
[[nodiscard]] TapPage tap_page(const ViewRecord& view, const std::array<float, 2>& texel,
                               const std::array<float, 2>& offset) noexcept;
} // namespace aurora::render::shadows::vsm

#pragma once

#include "../graph.hpp"
#include "../settings.hpp"
#include "../stage_camera.hpp"

#include <aurora/render.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace aurora::render::detail {
struct SceneHints;
} // namespace aurora::render::detail

namespace aurora::render::detail::ao {
inline constexpr float RadiusMetres = 0.73f;
inline constexpr float ThicknessMetres = 0.25f;
inline constexpr float FalloffRange = 0.615f;
inline constexpr float DistributionPower = 2.f;
inline constexpr float MipOffset = 3.3f;
inline constexpr uint32_t MipLevels = 5;
inline constexpr float FinalPower = 1.f;
// XeGTAO's 0.03 linear-light floor in stored space (0.03^(1/2.2))
inline constexpr float MinVisibility = 0.2f;
inline constexpr float SkyDepth = 1e30f;
inline constexpr float DenoiseBeta = 1.2f;
inline constexpr float DepthReject = 0.1f;
inline constexpr float MaxSamples = 16.f;
inline constexpr float LagSamples = 2.f;
inline constexpr float MovingSamples = 4.f;
inline constexpr float ClipGamma = 2.f;
inline constexpr float ClipFloor = 2.f / 255.f;
inline constexpr float MotionContrastPx = 0.5f;
inline constexpr uint32_t NoisePeriod = 64;
inline constexpr size_t MaxAoPasses = 4;
inline constexpr uint32_t SectorCount = 32;
inline constexpr float RawScale = 1.5f;
inline constexpr float PixelTooClose = 1.3f;
inline constexpr uint32_t HilbertWidth = 64;
inline constexpr uint32_t NoiseStride = 288;
inline constexpr std::array<uint32_t, 2> R2Steps{3242174889u, 2447445414u};

// Built by default with the composite, the only way AO reaches the screen
inline constexpr uint32_t BuiltOptionMask = (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4);

enum ParamFlags : uint32_t {
  FlagTemporalOff = 1u << 0,
  FlagIdentityForced = 1u << 1,
  FlagNormalFromDepth = 1u << 2,
  FlagSectorCeil = 1u << 3,
};

struct QualityRow {
  uint32_t slices = 0;
  uint32_t steps = 0;
  uint32_t divisor = 2;
};

inline constexpr std::array<QualityRow, 5> QualityTable{{
    {0, 0, 2},
    {1, 2, 2},
    {2, 2, 2},
    {3, 3, 2},
    {3, 3, 1},
}};

struct Params {
  StageCamera camera;
  float effectRadius = 0.f;
  float thickness = 0.f;
  float falloffRange = FalloffRange;
  float strength = 1.f;
  float mipOffset = MipOffset;
  float distributionPower = DistributionPower;
  uint32_t sliceCount = 0;
  uint32_t stepsPerSlice = 0;
  uint32_t noiseIndex = 0;
  uint32_t flags = 0;
  float denoiseBeta = DenoiseBeta;
  float depthReject = DepthReject;
  float maxSamples = MaxSamples;
  float radiusMetres = RadiusMetres;
  float thicknessMetres = ThicknessMetres;
  float minVisibility = MinVisibility;
};
static_assert(sizeof(Params) == 112);
static_assert(sizeof(Params) <= MaxPassParamsSize);

constexpr QualityRow quality_row(Quality quality) noexcept {
  const auto index = static_cast<size_t>(quality);
  return index < QualityTable.size() ? QualityTable[index] : QualityTable[0];
}

constexpr uint32_t working_divisor(Quality quality, uint32_t forced) noexcept {
  return forced == 1 || forced == 2 ? forced : quality_row(quality).divisor;
}

constexpr uint32_t grid_size(uint32_t full, uint32_t divisor) noexcept {
  return divisor == 0 ? full : (full + divisor - 1) / divisor;
}

constexpr uint32_t divisor_from_dims(uint32_t full, uint32_t grid) noexcept {
  return std::max((full + std::max(grid, 1u) - 1) / std::max(grid, 1u), 1u);
}

constexpr uint32_t rep_texel(uint32_t p, uint32_t divisor, uint32_t full) noexcept {
  return std::min(divisor * p + divisor / 2, std::max(full, 1u) - 1);
}

inline uint32_t sector_bits(float t0, float t1, bool ceilMode) noexcept {
  const float lo = std::clamp(t0, 0.f, 1.f);
  const float hi = std::clamp(t1, 0.f, 1.f);
  uint32_t start = 0;
  uint32_t end = 0;
  if (ceilMode) {
    start = std::min(static_cast<uint32_t>(lo * 32.f), 31u);
    end = std::min(start + static_cast<uint32_t>(std::max(std::ceil((hi - lo) * 32.f), 0.f)), 32u);
  } else {
    start = static_cast<uint32_t>(std::nearbyint(lo * 32.f));
    end = static_cast<uint32_t>(std::nearbyint(hi * 32.f));
  }
  if (end <= start) {
    return 0;
  }
  return (0xFFFFFFFFu >> (32u - (end - start))) << start;
}

constexpr uint32_t hilbert_index(uint32_t x, uint32_t y) noexcept {
  uint32_t index = 0;
  for (uint32_t level = HilbertWidth / 2; level > 0; level /= 2) {
    const uint32_t rx = (x & level) != 0 ? 1u : 0u;
    const uint32_t ry = (y & level) != 0 ? 1u : 0u;
    index += level * level * ((3u * rx) ^ ry);
    if (ry == 0) {
      if (rx == 1) {
        x = HilbertWidth - 1 - x;
        y = HilbertWidth - 1 - y;
      }
      std::swap(x, y);
    }
  }
  return index;
}

inline std::array<float, 2> noise(uint32_t x, uint32_t y, uint32_t noiseIndex) noexcept {
  const uint32_t index = hilbert_index(x, y) + NoiseStride * (noiseIndex % NoisePeriod);
  std::array<float, 2> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    const float value = 0.5f + static_cast<float>(index * R2Steps[i]) * (1.f / 4294967296.f);
    out[i] = value - std::floor(value);
  }
  return out;
}

constexpr std::array<float, 2> falloff_bounds(float lo, float hi, float weight, bool positiveSide) noexcept {
  if (positiveSide) {
    return {1.f - weight * (1.f - lo), 1.f - weight * (1.f - hi)};
  }
  return {weight * lo, weight * hi};
}

constexpr float apply_strength(float visibility, float strength, float floor) noexcept {
  return std::max(1.f - strength * (1.f - visibility), 1.f - std::min(strength, 1.f) * (1.f - floor));
}

struct EnvOverrides {
  std::optional<float> radius;
  std::optional<float> thickness;
  std::optional<float> strength;
  std::optional<float> floor;
  uint32_t divisor = 0;
  uint32_t flags = 0;
};

[[nodiscard]] uint32_t working_divisor(Quality quality) noexcept;
[[nodiscard]] std::optional<Quality> parse_quality(std::string_view text) noexcept;
[[nodiscard]] std::optional<uint32_t> parse_divisor(std::string_view text) noexcept;
[[nodiscard]] std::optional<Quality> env_ao_request() noexcept;
[[nodiscard]] BuiltOptions with_dev_built(BuiltOptions built, std::optional<Quality> request) noexcept;

[[nodiscard]] Params latched_params(const Settings& settings, const EnvOverrides& env) noexcept;
[[nodiscard]] bool identity_forced() noexcept;
[[nodiscard]] bool sink_forced() noexcept;

[[nodiscard]] bool wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool scene_ready(const StageData& stage, const void* state) noexcept;
void fill_camera(const StageData& stage, const void* state, std::span<std::byte> params);
void on_latch(const Settings& settings, const CompiledGraph* graph);
[[nodiscard]] bool pipelines_live() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] bool motion_valid(const StageData& stage) noexcept;
[[nodiscard]] float effective_strength() noexcept;

void initialize();
void shutdown() noexcept;
void worker_tick(uint64_t frameId) noexcept;

namespace testing {
struct Counters {
  uint32_t skippedNoScene = 0;
  uint32_t cameraFills = 0;
};
Counters counters() noexcept;
void reset() noexcept;
void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_sink_override(std::optional<bool> sink) noexcept;
void unregister(std::string_view pass) noexcept;
} // namespace testing
} // namespace aurora::render::detail::ao

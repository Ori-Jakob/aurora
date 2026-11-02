#pragma once

#include "../graph.hpp"
#include "../settings.hpp"
#include "../stage_camera.hpp"

#include <aurora/render.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

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

// W4 sets these with the composite; until then AO is reachable only through AURORA_RENDER_AO
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

[[nodiscard]] uint32_t working_divisor(Quality quality) noexcept;
[[nodiscard]] std::optional<Quality> parse_quality(std::string_view text) noexcept;
[[nodiscard]] std::optional<uint32_t> parse_divisor(std::string_view text) noexcept;
[[nodiscard]] std::optional<Quality> env_ao_request() noexcept;
[[nodiscard]] BuiltOptions with_dev_built(BuiltOptions built, std::optional<Quality> request) noexcept;

[[nodiscard]] bool wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool scene_ready(const StageData& stage, const void* state) noexcept;
void fill_camera(const StageData& stage, const void* state, std::span<std::byte> params);
void on_latch(const Settings& settings, const CompiledGraph* graph);
[[nodiscard]] bool pipelines_live() noexcept;

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
} // namespace testing
} // namespace aurora::render::detail::ao

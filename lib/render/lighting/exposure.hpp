#pragma once

#include "../graph.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::detail {
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::detail::lighting::exposure {
inline constexpr std::string_view Pass = "exposure.meter";
inline constexpr std::string_view State = "exposure.state";
inline constexpr std::string_view Partials = "exposure.partials";
inline constexpr std::string_view Resource = "exposure";
inline constexpr std::string_view Shader = "lighting/exposure";
inline constexpr std::string_view ModeEnv = "AURORA_RENDER_EXPOSURE";
inline constexpr std::string_view TonemapEnv = "AURORA_RENDER_TONEMAP";
inline constexpr std::string_view TargetEnv = "AURORA_RENDER_EXPOSURE_TARGET";
inline constexpr wgpu::TextureFormat StateFormat = wgpu::TextureFormat::RGBA32Uint;

inline constexpr uint32_t GridWidth = 192;
inline constexpr uint32_t GridHeight = 108;
inline constexpr uint32_t GridCells = GridWidth * GridHeight;
inline constexpr uint32_t Groups = 81;
inline constexpr uint32_t GroupSize = 64;
inline constexpr uint32_t SamplesPerThread = 4;
inline constexpr uint32_t Candidates = 32;
inline constexpr uint32_t PartialWords = 36;
inline constexpr uint64_t PartialBytes = uint64_t{Groups} * PartialWords * sizeof(float);
inline constexpr uint64_t BufferBytes = 16;
inline constexpr float StepEv = 0.25f;
inline constexpr float SnapStepEv = 0.5f;
inline constexpr float MaxEv = 8.f;
inline constexpr float Key = 0.18f;
inline constexpr float MatchTau = 0.1f;
inline constexpr float AutoUpTau = 1.f;
inline constexpr float AutoDownTau = 0.4f;
inline constexpr float HoldFraction = 0.01f;
inline constexpr uint32_t FogCutQ7 = 114;
inline constexpr float YFloor = 0x1p-10f;
inline constexpr float MaxDeltaMs = 250.f;
// Q9: MatchLegacy's median on T1h, T2s and T13m is +4.45 EV, so Manual 0 EV sits there instead of at H's white
inline constexpr float ManualAnchorEv = 4.5f;
static_assert(Groups * GroupSize * SamplesPerThread == GridCells && PartialBytes == 11664 && Groups % 9 == 0);

enum PartialWord : uint32_t {
  WordWeight = Candidates,
  WordTarget = Candidates + 1,
  WordCount = Candidates + 2,
  WordHdr = Candidates + 3,
};
static_assert(WordHdr + 1 == PartialWords);

enum class Target : uint8_t { Sb, S };

enum ParamFlags : uint32_t {
  FlagAgx = 1u << 0,
  FlagTargetS = 1u << 1,
  FlagSnap = 1u << 2,
  FlagReset = 1u << 3,
  FlagSsil = 1u << 4,
  FlagSsilExposed = 1u << 5,
  FlagLsm = 1u << 6,
};

enum StateFlags : uint32_t {
  StateValid = 1u << 0,
  StateModeShift = 1,
  StateModeMask = 3u << StateModeShift,
  StateAgx = 1u << 3,
  StateSnapped = 1u << 4,
  StateHeld = 1u << 5,
  StatePinned = 1u << 6,
  StateTargetS = 1u << 7,
};

struct Params {
  uint32_t mode = 0;
  uint32_t flags = 0;
  float bias = 0.f;
  float envBias = 0.f;
  float dt = 0.f;
  float anchor = 0.f;
  float ssilAlbedo = 0.f;
  uint32_t pad = 0;
};
static_assert(sizeof(Params) == 32 && offsetof(Params, dt) == 16 && offsetof(Params, anchor) == 20 &&
              offsetof(Params, ssilAlbedo) == 24);

// The GPU reads the previous s from the prev texel, so only the mirror carries it for the ratio word
struct Solved {
  float s = 0.f;
  uint32_t flags = 0;
  float target = 0.f;
  float weight = 0.f;
  float t = 0.f;
  float fLo = 0.f;
  float fHi = 0.f;
  float count = 0.f;
  float previous = 0.f;
};

constexpr TexDesc state_desc() noexcept {
  // The prev group samples it, so the usage cannot depend on a capture
  return {
      .format = StateFormat,
      .sizeClass = SizeClass::Absolute,
      .width = 2,
      .height = 1,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

[[nodiscard]] bool wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool armed() noexcept;
[[nodiscard]] bool pipelines_live() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] std::optional<Target> parse_target(std::string_view text) noexcept;
[[nodiscard]] Target target() noexcept;
[[nodiscard]] float env_bias(const SceneHints* hints) noexcept;
void fill_params(const Settings& settings, Target target, float deltaMs, bool snap, Params& out) noexcept;
PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);
void worker_tick(uint64_t frameId) noexcept;

namespace testing {
// Holds and pins live only in the GPU state flags, which tests read back
struct Counters {
  uint32_t runs = 0;
  uint32_t snaps = 0;
  uint32_t manual = 0;
};
[[nodiscard]] Counters counters() noexcept;
void reset() noexcept;
void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_delta_ms(std::optional<float> deltaMs) noexcept;
void set_target(std::optional<Target> target) noexcept;

[[nodiscard]] float exp2_exact(float s) noexcept;
[[nodiscard]] float candidate(float center, float step, uint32_t k) noexcept;
[[nodiscard]] bool snaps(const Params& params, const std::array<uint32_t, 8>& prev) noexcept;
[[nodiscard]] float center(const Params& params, const std::array<uint32_t, 8>& prev) noexcept;
[[nodiscard]] Solved solve(const Params& params, std::span<const float> partials,
                           const std::array<uint32_t, 8>& prev) noexcept;
[[nodiscard]] std::array<uint32_t, 8> texels(const Solved& state) noexcept;
[[nodiscard]] std::array<float, 4> words(const Solved& state) noexcept;
[[nodiscard]] float smoothing_alpha(float dt, float tau) noexcept;
// The composite multiplies stored sRGB by m, so the meter reads that product back in linear light
[[nodiscard]] std::array<float, 3> masked(std::array<float, 3> mapped, std::array<float, 3> m) noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting::exposure

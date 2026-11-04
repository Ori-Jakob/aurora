#pragma once

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::detail {
struct FrameState;
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::detail::composite {
inline constexpr std::string_view CopyPass = "legacy.copy";
inline constexpr std::string_view Pass = "composite.main";
inline constexpr std::string_view LegacyTex = "legacyTex";
inline constexpr std::string_view Shader = "post/composite";
inline constexpr std::string_view NoFogEnv = "AURORA_RENDER_COMPOSITE_NOFOG";
inline constexpr wgpu::TextureFormat AoFormat = wgpu::TextureFormat::RGBA8Unorm;
inline constexpr std::string_view RadianceEnv = "AURORA_RENDER_COMPOSITE_RADIANCE";
inline constexpr std::string_view RadianceDefine = "COMPOSITE_RADIANCE";
inline constexpr wgpu::TextureFormat RadianceFormat = wgpu::TextureFormat::RGBA16Float;
// The meter's fog cut, so radiance and exposure agree on which pixels are fog-drowned
inline constexpr uint32_t RadianceFogCutQ7 = 114;

// Bindings follow declaration order, so later terms append and never renumber
enum Binding : uint32_t {
  BindLegacy = 0,
  BindFog = 1,
  BindNormal = 2,
  BindAo = 3,
  BindSurface = 4,
  BindHdr = 5,
  BindExposure = 6,
  BindSsil = 7,
};

enum ParamFlags : uint32_t {
  FlagIdentity = 1u << 0,
  FlagNoFog = 1u << 1,
};
// Set only while the PBR term can change pixels, so legacyBlend = 1 runs the AO-only path bit for bit
inline constexpr uint32_t FlagPbr = 1u << 2;
inline constexpr uint32_t FlagTonemapAgX = 1u << 3;
inline constexpr uint32_t FlagSsil = 1u << 4;

struct Params {
  uint32_t flags = 0;
  float legacyBlend = 0.f;
  float ssilAlbedo = 0.f;
  uint32_t pad = 0;
};
static_assert(sizeof(Params) == 16 && offsetof(Params, legacyBlend) == 4 && offsetof(Params, ssilAlbedo) == 8);

[[nodiscard]] bool wanted(const Settings& settings) noexcept;
[[nodiscard]] bool ao_wanted(const Settings& settings) noexcept;
[[nodiscard]] bool pbr_term_wanted(const Settings& settings) noexcept;
[[nodiscard]] bool surface_bound(const CompiledGraph& graph) noexcept;
[[nodiscard]] bool hdr_produced(const CompiledGraph& graph) noexcept;
[[nodiscard]] bool ao_produced(const CompiledGraph& graph) noexcept;
[[nodiscard]] bool declare_wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool radiance_wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool radiance_bound(const CompiledGraph& graph) noexcept;
[[nodiscard]] bool radiance_env_on(const char* value) noexcept;
[[nodiscard]] bool ssil_bound(const CompiledGraph& graph) noexcept;
[[nodiscard]] uint32_t units_epoch() noexcept;
[[nodiscard]] uint32_t params_flags(bool identity, const char* noFog) noexcept;
[[nodiscard]] std::optional<std::string> check_contract(const CompiledGraph& graph);
[[nodiscard]] shaders::RenderPipelineDesc pipeline_desc(wgpu::TextureFormat efb, const wgpu::PipelineLayout& layout,
                                                        uint64_t layoutKey, bool radiance = false);
PassDesc copy_pass_desc();
PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] bool ao_ready(const Settings& settings) noexcept;
[[nodiscard]] bool pbr_active() noexcept;
[[nodiscard]] bool pbr_applied(const StageData& opaqueDone) noexcept;
[[nodiscard]] bool live() noexcept;
[[nodiscard]] bool enabled_this_frame(uint64_t frameId) noexcept;
[[nodiscard]] bool identity_forced() noexcept;
[[nodiscard]] bool ao_gate(const FrameState& state) noexcept;

namespace testing {
struct Counters {
  uint32_t copies = 0;
  uint32_t draws = 0;
  uint32_t skippedMismatch = 0;
  uint32_t skippedNoScene = 0;
  uint32_t skippedView = 0;
  uint32_t contractFailures = 0;
  uint32_t pbrDraws = 0;
  uint32_t idleFrames = 0;
  uint32_t hdrDraws = 0;
  uint32_t radianceDraws = 0;
  uint32_t radianceClears = 0;
  uint32_t ssilDraws = 0;
};
Counters counters() noexcept;
void reset() noexcept;
void set_flags(std::optional<uint32_t> flags) noexcept;
void force_params(std::optional<Params> params) noexcept;
void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_pbr_producers_override(std::optional<bool> live) noexcept;
void set_ready_override(std::optional<bool> ready) noexcept;
void set_radiance_forced(std::optional<bool> forced) noexcept;
void bump_units_epoch() noexcept;
[[nodiscard]] bool pbr_active_for(bool live, bool hdrLive, bool classView, const Settings& settings) noexcept;
[[nodiscard]] std::array<float, 3> legacy_ao(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7,
                                             float ao, float normalAlpha) noexcept;
[[nodiscard]] std::array<float, 3> legacy_pbr_hdr(std::array<float, 3> legacy, std::array<float, 3> fogColor,
                                                  uint32_t q7, float ao, float normalAlpha, std::array<float, 4> hdr,
                                                  float legacyBlend, float exposure, bool agx) noexcept;
[[nodiscard]] std::array<float, 2> units(uint32_t flags, float e, float invE) noexcept;
[[nodiscard]] std::array<float, 3> legacy_ssil(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7,
                                               float ao, float normalAlpha, std::array<float, 3> bounce) noexcept;
[[nodiscard]] std::array<float, 4> radiance(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7,
                                            float ao, float normalAlpha, std::array<float, 4> hdr, bool pbrFlag,
                                            float legacyBlend, float invE) noexcept;
} // namespace testing
} // namespace aurora::render::detail::composite

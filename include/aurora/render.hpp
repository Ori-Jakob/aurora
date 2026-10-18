#pragma once

#include <aurora/render_caps.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace aurora::render {
inline constexpr uint32_t ApiVersion = 1;

enum class Preset : uint8_t { Custom, Original, Low, Medium, High, Ultra };
enum class Quality : uint8_t { Off, Low, Medium, High, Ultra };
enum class Exposure : uint8_t { MatchLegacy, Auto, Manual };
enum class Tonemap : uint8_t { PbrNeutral, AgX };
enum class GiMode : uint8_t { Off, Voxel, Ddgi };
enum class RtMode : uint8_t { Off, Shadows, ShadowsAo, ShadowsAoReflections, PathTraceReference };
enum class RtBackend : uint8_t { Auto, Compute, Dxr };
enum class Reflections : uint8_t { Off, Ssr, SsrRt };
enum class GameShadows : uint8_t { Auto, Keep, Remove };
enum class GodRays : uint8_t { Off, Auto, ScreenSpace, Volumetric };
enum class TerrainDetail : uint8_t { Off, Parallax, Tessellated };
enum class GrassShadows : uint8_t { Off, Contact, Shell, Full };
enum class StaticWorld : uint8_t { Off, Validate, On };

struct ShadowSettings {
  Quality quality = Quality::Off;
  uint32_t pages = 1024, pageBudget = 64;
  GameShadows game = GameShadows::Auto;
  float legacyGain = 1.f;
  bool keepCues = false;

  bool operator==(const ShadowSettings&) const = default;
};

struct FogSettings {
  Quality volumetric = Quality::Off;
  GodRays godRays = GodRays::Off;
  float scattering = 1.f, shaftStrength = 1.f;
  bool legacyMatch = false, translucents = false;

  bool operator==(const FogSettings&) const = default;
};

struct CloudTuning {
  float coverage = 1.f, density = 1.f, altitude = 1.f, windSpeed = 1.f, silverLining = 1.f, powder = 1.f;

  bool operator==(const CloudTuning&) const = default;
};

struct SkySettings {
  Quality clouds = Quality::Off;
  float cloudStyle = -1.f;
  bool cloudShadows = false;
  float cloudShadowStrength = 1.f;
  bool keepCloudDome = false;
  CloudTuning tuning;

  bool operator==(const SkySettings&) const = default;
};

struct GrassSettings {
  Quality quality = Quality::Off;
  bool replaceGamePatches = true, terrainGrass = true;
  float distance = 0.f, density = 1.f, distantDensity = 0.f;
  GrassShadows shadows = GrassShadows::Contact;
  bool interaction = true;
  float windScale = 1.f;
  bool legacyShadowTransfer = true;

  bool operator==(const GrassSettings&) const = default;
};

struct TerrainSettings {
  TerrainDetail mode = TerrainDetail::Off;
  Quality quality = Quality::High;
  float strength = 1.f, distance = 0.f;
  bool flattenVolumes = false;

  bool operator==(const TerrainSettings&) const = default;
};

struct Settings {
  Preset preset = Preset::Original;
  bool textureMaterials = true, stochasticTiling = false, legacyMatMaps = false;
  bool pbr = false;
  float legacyBlend = 0.f;
  Exposure exposure = Exposure::MatchLegacy;
  float exposureBias = 0.f;
  Tonemap tonemap = Tonemap::PbrNeutral;
  bool taa = false;
  float taaSharpness = 0.25f;
  Quality ao = Quality::Off;
  bool ssil = false;
  ShadowSettings shadows;
  GiMode gi = GiMode::Off;
  RtMode rt = RtMode::Off;
  RtBackend rtBackend = RtBackend::Auto;
  Reflections reflections = Reflections::Off;
  FogSettings fog;
  SkySettings sky;
  GrassSettings grass;
  TerrainSettings terrain;
  bool gpuFoliage = false, gpuWeather = false, gpuParticles = false, gpuDeform = false, meshReplacement = false;
  StaticWorld staticWorld = StaticWorld::Off;
  uint16_t debugView = 0;

  bool operator==(const Settings&) const = default;
};

enum class Feature : uint8_t {
  GBuffer,
  Pbr,
  Taa,
  Ao,
  Ssil,
  Shadows,
  GameShadowSuppression,
  Gi,
  Rt,
  Reflections,
  DeferredFog,
  VolumetricFog,
  GodRays,
  Clouds,
  CloudShadows,
  Grass,
  GrassTerrain,
  TerrainParallax,
  TerrainTessellation,
  NativeTrace,
  GpuFoliage,
  GpuWeather,
  GpuParticles,
  GpuDeform,
  MeshReplacement,
  StaticWorld,
  TextureMaterials,
  StochasticTiling,
  LegacyMatMaps,
};
inline constexpr size_t FeatureCount = static_cast<size_t>(Feature::LegacyMatMaps) + 1;

enum class Reason : uint8_t {
  None,
  NeedsCore,
  ColorBudget,
  NoFragmentStorage,
  NoVertexStorage,
  NoAtomic64,
  NoMdi,
  NeedsD3D12,
  NativeUnavailable,
  NativeDisabledAfterCrash,
  MsaaActive,
  NeedsRetainedScene,
  NeedsCompanions,
  NeedsSunShadows,
  BindingLimit,
  LowVram,
  BuildDisabled,
  NotImplemented,
  NeedsPbr,
  RequiredByShadows,
  NeedsAo,
  NeedsRt,
};

struct FeatureSupport {
  bool supported = false;
  Reason reason = Reason::None;

  bool operator==(const FeatureSupport&) const = default;
};

Settings apply_settings(const Settings& requested);
const Settings& effective_settings() noexcept;
bool feature_enabled(const Settings& settings, Feature feature) noexcept;
bool feature_active(Feature feature) noexcept;
FeatureSupport feature_support(Feature feature, uint8_t option = 0) noexcept;
FeatureSupport feature_support(Feature feature, uint8_t option, const Settings& context) noexcept;
bool feature_ready(Feature feature) noexcept;
std::string_view feature_name(Feature feature) noexcept;
std::string_view reason_name(Reason reason) noexcept;

enum class Stage : uint8_t {
  SceneBegin,
  OpaqueDone,
  PrePost,
  SceneEnd,
  BeforeHud,
  FrameEnd,
  AfterFrame,
};
inline constexpr size_t StageCount = 7;

constexpr std::string_view stage_name(Stage stage) noexcept {
  switch (stage) {
  case Stage::SceneBegin:
    return "SceneBegin";
  case Stage::OpaqueDone:
    return "OpaqueDone";
  case Stage::PrePost:
    return "PrePost";
  case Stage::SceneEnd:
    return "SceneEnd";
  case Stage::BeforeHud:
    return "BeforeHud";
  case Stage::FrameEnd:
    return "FrameEnd";
  case Stage::AfterFrame:
    return "AfterFrame";
  }
  return "Unknown";
}

struct PassTiming {
  std::string_view name;
  float gpuMs = 0.f;
};

void draw_debug_window(bool* open);
std::span<const PassTiming> pass_timings() noexcept;
std::string graph_dump_json(bool includeFrameState = false);

struct HintCounters {
  uint64_t decoded = 0;
  uint64_t skipped = 0;
  uint64_t malformed = 0;
  uint64_t late = 0;
  uint64_t bytes = 0;
  bool operator==(const HintCounters&) const = default;
};

bool hints_enabled() noexcept;
HintCounters hint_counters(uint16_t id) noexcept;
} // namespace aurora::render

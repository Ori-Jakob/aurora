#pragma once

#include <aurora/render_caps.hpp>

#include <array>
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
enum class Upscaler : uint8_t { Off, Fsr, Dlss, Xess, Auto };
enum class UpscalerQuality : uint8_t { NativeAa, Quality, Balanced, Performance, UltraPerformance };

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
  bool textureMaterials = true, stochasticTiling = false, legacyMatMaps = false, generateMissingMips = false;
  bool pbr = false;
  float legacyBlend = 0.f;
  Exposure exposure = Exposure::MatchLegacy;
  float exposureBias = 0.f;
  Tonemap tonemap = Tonemap::PbrNeutral;
  bool taa = false;
  float taaSharpness = 0.35f;
  Upscaler upscaler = Upscaler::Off;
  UpscalerQuality upscalerQuality = UpscalerQuality::Quality;
  Quality ao = Quality::Off;
  // Radius in metres, so the reach does not depend on the game's units
  float aoRadius = 0.73f, aoStrength = 1.f;
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
  Upscaler,
};
inline constexpr size_t FeatureCount = static_cast<size_t>(Feature::Upscaler) + 1;

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
  NeedsTextureFormats,
  RuntimeMissing,
  DriverTooOld,
  UnsupportedGpu,
  OutputTooLarge,
  Probing,
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

struct UpscaleStatus {
  Upscaler active = Upscaler::Off;
  UpscalerQuality quality = UpscalerQuality::Quality;
  uint32_t sceneSize[2]{};
  uint32_t outputSize[2]{};
  std::string_view displayName;
  std::string_view path;
  bool ready = false;
};

UpscaleStatus upscale_status() noexcept;
void reenable_native_upscalers() noexcept;

struct AppIdentity {
  std::string_view projectId;
  std::string_view engineVersion;
};

void set_app_identity(const AppIdentity& identity) noexcept;

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

struct DrawRouteCounters {
  uint64_t routed = 0;
  uint64_t suppressed = 0;
  // Merges actually prevented; classify.json's mergeBreaks counts every requested break
  uint64_t mergeBreaks = 0;
  uint64_t substituted = 0;
  uint64_t captured = 0;
  bool operator==(const DrawRouteCounters&) const = default;
};

DrawRouteCounters draw_route_counters() noexcept;

struct RetainedStats {
  bool active = false;
  bool coverageComplete = false;
  uint32_t instances = 0;
  uint32_t staticInstances = 0;
  uint32_t staticObjects = 0;
  uint32_t undeliveredStatics = 0;
  uint64_t created = 0;
  uint64_t evicted = 0;
  uint64_t captured = 0;
  uint64_t churn = 0;
  uint64_t lostCaptures = 0;
  uint64_t resyncs = 0;
  uint64_t capturedOnly = 0;
  bool operator==(const RetainedStats&) const = default;
};

RetainedStats retained_stats() noexcept;
// Latched per frame; a producer stamps CAPTURE_ONLY only while it is true
bool retained_capture_active() noexcept;
// Latched at the end of each frame; never true without a settled producer
bool retained_coverage_complete() noexcept;
// Game thread; true while the caller is still capturing, and the first call makes it the producer
void retained_hold_coverage(bool hold) noexcept;

// M10
struct OpenPbrParams {
  float baseWeight = 1.f;
  float baseMetalness = 0.f;
  float baseDiffuseRoughness = 0.f;
  float specularWeight = 1.f;
  std::array<float, 3> specularColor{1.f, 1.f, 1.f};
  // Used where no orm map or manifest value exists; 0.85 is aurora's default, not OpenPBR's 0.3
  float specularRoughness = 0.85f;
  float specularIor = 1.5f;
  float coatWeight = 0.f;
  std::array<float, 3> coatColor{1.f, 1.f, 1.f};
  float coatRoughness = 0.f;
  float coatIor = 1.6f;
  float coatDarkening = 1.f;
  float fuzzWeight = 0.f;
  std::array<float, 3> fuzzColor{1.f, 1.f, 1.f};
  float fuzzRoughness = 0.5f;
  float subsurfaceWeight = 0.f;
  std::array<float, 3> subsurfaceColor{0.8f, 0.8f, 0.8f};
  float subsurfaceScatterAnisotropy = 0.f;
  std::array<float, 3> emissionColor{1.f, 1.f, 1.f};
  // Multiplies the emit map in legacy-white units; OpenPBR's emission_luminance is in nits
  float emissionScale = 1.f;
  bool geometryThinWalled = false;
  // Diffuse wrap for thin-walled surfaces, 0 to 1; OpenPBR has no such parameter
  float thinWalledWrap = 0.f;
  bool twoSided = false, noSsr = false;
  // subsurfaceColor multiplies the base colour, so a thin leaf passes its own colour; OpenPBR has no such parameter
  bool subsurfaceTintedByBase = false;

  bool operator==(const OpenPbrParams&) const = default;
};

// Game thread; takes effect at the next frame latch. Ids 128-254; 2-127 only to retune the built-in of the same name
void register_material_class(uint8_t id, std::string_view name, const OpenPbrParams& params);
} // namespace aurora::render

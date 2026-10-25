#include "settings.hpp"

#include "frame_state.hpp"
#include "graph.hpp"
#include "graph_api.hpp"
#include "graph_bind.hpp"
#include "module.hpp"
#include "stage.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../logging.hpp"
#include "../webgpu/caps.hpp"

#if AURORA_ENABLE_RENDER
#include "debug/debug.hpp"
#include "gx_ext/ext_provider.hpp"
#include "hints.hpp"
#include "scene/classify.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render"};

constexpr std::array<FeatureInfo, FeatureCount> FeatureTable{{
    {"GBuffer", Tier::Base, 1},
    {"Pbr", Tier::Pbr, 1},
    {"Taa", Tier::Base, 1},
    {"Ao", Tier::Base, 4},
    {"Ssil", Tier::Base, 1},
    {"Shadows", Tier::Base, 4},
    {"GameShadowSuppression", Tier::None, 2},
    {"Gi", Tier::Base, 2},
    {"Rt", Tier::Base, 4},
    {"Reflections", Tier::Pbr, 2},
    {"DeferredFog", Tier::Base, 1},
    {"VolumetricFog", Tier::Base, 4},
    {"GodRays", Tier::Base, 3},
    {"Clouds", Tier::Base, 4},
    {"CloudShadows", Tier::Base, 1},
    {"Grass", Tier::Base, 4},
    {"GrassTerrain", Tier::Base, 1},
    {"TerrainParallax", Tier::None, 1},
    {"TerrainTessellation", Tier::Base, 4},
    {"NativeTrace", Tier::None, 1},
    {"GpuFoliage", Tier::None, 1},
    {"GpuWeather", Tier::None, 1},
    {"GpuParticles", Tier::None, 1},
    {"GpuDeform", Tier::None, 1},
    {"MeshReplacement", Tier::Pbr, 1},
    {"StaticWorld", Tier::None, 2},
    {"TextureMaterials", Tier::None, 1},
    {"StochasticTiling", Tier::None, 1},
    {"LegacyMatMaps", Tier::None, 1},
}};
static_assert(FeatureTable.size() == FeatureCount);

constexpr FeatureInfo UnknownFeature{"Unknown", Tier::None, 0};

// Dependencies come before their dependents, so each gate sees its parents already clamped
constexpr std::array ClampOrder{
    Feature::TextureMaterials,
    Feature::StochasticTiling,
    Feature::LegacyMatMaps,
    Feature::Pbr,
    Feature::Taa,
    Feature::Ao,
    Feature::Ssil,
    Feature::Shadows,
    Feature::Rt,
    Feature::NativeTrace,
    Feature::Gi,
    Feature::Reflections,
    Feature::GameShadowSuppression,
    Feature::CloudShadows,
    Feature::Clouds,
    Feature::VolumetricFog,
    Feature::GodRays,
    Feature::Grass,
    Feature::GrassTerrain,
    Feature::TerrainTessellation,
    Feature::TerrainParallax,
    Feature::GpuFoliage,
    Feature::GpuWeather,
    Feature::GpuParticles,
    Feature::GpuDeform,
    Feature::MeshReplacement,
    Feature::StaticWorld,
};
static_assert(ClampOrder.size() == FeatureCount - 2);

// Hex tiling ships in M6; which textures use it is the material registry's call
constexpr BuiltOptions DefaultBuiltOptions = [] {
  BuiltOptions built{};
  built[static_cast<size_t>(Feature::TextureMaterials)] = 1u << 1;
  built[static_cast<size_t>(Feature::StochasticTiling)] = 1u << 1;
  return built;
}();

constexpr uint32_t CompanionSampledTextures = 16;
constexpr uint32_t TessellationVertexStorageBuffers = 4;
constexpr float MaxExposureBias = 8.f;
constexpr uint64_t ModuleStorageBufferSize = 32 * 1024 * 1024;

constexpr Settings Defaults{};

std::mutex g_requestMutex;
Settings g_lastRequest;
std::optional<Settings> g_pending;
Settings g_supportContext;
BuiltOptions g_built = DefaultBuiltOptions;
uint32_t g_msaaSamples = 1;
bool g_reclamp = false;
std::optional<Settings> g_latched;

uint64_t g_generation = 0;
std::shared_ptr<const FrameState> g_frameState;
#if AURORA_ENABLE_RENDER
GraphEnvironment g_graphEnvironment;
std::shared_ptr<const ExternalPassSet> g_externals;
uint64_t g_externalEpoch = 0;
DepthSource g_sceneDepthSource = DepthSource::Unavailable;
#endif

using FeatureMask = uint32_t;
static_assert(FeatureCount <= 32);

constexpr size_t feature_index(Feature feature) noexcept { return static_cast<size_t>(feature); }

uint8_t flag(bool value) noexcept { return value ? 1 : 0; }

template <typename E>
uint8_t enum_level(E value) noexcept {
  return static_cast<uint8_t>(value);
}

bool uses_gbuffer(const Settings& settings) noexcept {
  for (size_t i = 0; i < FeatureCount; ++i) {
    const auto feature = static_cast<Feature>(i);
    if (feature == Feature::GBuffer || feature == Feature::DeferredFog || FeatureTable[i].tier == Tier::None) {
      continue;
    }
    if (requested_option(settings, feature) != 0) {
      return true;
    }
  }
  return false;
}

template <typename E>
void sanitize_enum(E& value, E last, E fallback) noexcept {
  if (static_cast<uint8_t>(value) > static_cast<uint8_t>(last)) {
    value = fallback;
  }
}

void sanitize_range(float& value, float fallback, float low, float high) noexcept {
  value = std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}

void sanitize_non_negative(float& value, float fallback) noexcept {
  value = std::isfinite(value) ? std::max(value, 0.f) : fallback;
}

void write_level(Settings& settings, Feature feature, uint8_t value) noexcept {
  switch (feature) {
  case Feature::Pbr:
    settings.pbr = value != 0;
    break;
  case Feature::Taa:
    settings.taa = value != 0;
    break;
  case Feature::Ao:
    settings.ao = static_cast<Quality>(value);
    break;
  case Feature::Ssil:
    settings.ssil = value != 0;
    break;
  case Feature::Shadows:
    settings.shadows.quality = static_cast<Quality>(value);
    break;
  case Feature::GameShadowSuppression:
    settings.shadows.game = value == 0 ? GameShadows::Keep : GameShadows::Auto;
    break;
  case Feature::Gi:
    settings.gi = static_cast<GiMode>(value);
    break;
  case Feature::Rt:
    settings.rt = static_cast<RtMode>(value);
    break;
  case Feature::Reflections:
    settings.reflections = static_cast<Reflections>(value);
    break;
  case Feature::VolumetricFog:
    settings.fog.volumetric = static_cast<Quality>(value);
    break;
  case Feature::GodRays:
    settings.fog.godRays = static_cast<GodRays>(value);
    break;
  case Feature::Clouds:
    settings.sky.clouds = static_cast<Quality>(value);
    break;
  case Feature::CloudShadows:
    settings.sky.cloudShadows = value != 0;
    break;
  case Feature::Grass:
    settings.grass.quality = static_cast<Quality>(value);
    break;
  case Feature::GrassTerrain:
    settings.grass.terrainGrass = false;
    break;
  case Feature::TerrainParallax:
    settings.terrain.mode = TerrainDetail::Off;
    break;
  case Feature::TerrainTessellation:
    if (value == 0) {
      settings.terrain.mode = TerrainDetail::Parallax;
    } else {
      settings.terrain.quality = static_cast<Quality>(value);
    }
    break;
  case Feature::NativeTrace:
    settings.rtBackend = RtBackend::Compute;
    break;
  case Feature::GpuFoliage:
    settings.gpuFoliage = value != 0;
    break;
  case Feature::GpuWeather:
    settings.gpuWeather = value != 0;
    break;
  case Feature::GpuParticles:
    settings.gpuParticles = value != 0;
    break;
  case Feature::GpuDeform:
    settings.gpuDeform = value != 0;
    break;
  case Feature::MeshReplacement:
    settings.meshReplacement = value != 0;
    break;
  case Feature::StaticWorld:
    settings.staticWorld = static_cast<StaticWorld>(value);
    break;
  case Feature::TextureMaterials:
    settings.textureMaterials = value != 0;
    break;
  case Feature::StochasticTiling:
    settings.stochasticTiling = value != 0;
    break;
  case Feature::LegacyMatMaps:
    settings.legacyMatMaps = false;
    break;
  case Feature::GBuffer:
  case Feature::DeferredFog:
    break;
  }
}

const Settings& initial_settings() noexcept {
  static const Settings initial = [] {
    const Caps noCaps{};
    const BuiltOptions noneBuilt{};
    return clamp_settings(Settings{}, {noCaps, 1, noneBuilt, module_enabled()});
  }();
  return initial;
}

FeatureMask enabled_features(const Settings& settings) noexcept {
  FeatureMask mask = 0;
  for (size_t i = 0; i < FeatureCount; ++i) {
    if (requested_option(settings, static_cast<Feature>(i)) != 0) {
      mask |= FeatureMask{1} << i;
    }
  }
  return mask;
}

// The material features are GX shader paths that never write the streaming storage buffer
bool module_features_effective(const Settings& settings) noexcept {
  constexpr auto bit = [](Feature feature) { return FeatureMask{1} << feature_index(feature); };
  constexpr FeatureMask MaterialFeatures =
      bit(Feature::TextureMaterials) | bit(Feature::StochasticTiling) | bit(Feature::LegacyMatMaps);
  return (enabled_features(settings) & ~MaterialFeatures) != 0;
}

void log_latched_settings(uint64_t generation, FeatureMask previous, const Settings& latched) {
  const FeatureMask enabled = enabled_features(latched);
  std::string names;
  for (size_t i = 0; i < FeatureCount; ++i) {
    if (((enabled >> i) & 1u) == 0) {
      continue;
    }
    if (!names.empty()) {
      names += ", ";
    }
    names += FeatureTable[i].name;
  }
  if (names.empty()) {
    names = "legacy";
  }
  // Tuning changes arrive on every slider step, so only feature changes reach the info log
  const AuroraLogLevel level = enabled != previous ? LOG_INFO : LOG_DEBUG;
  Log.report(level, "Render settings generation {}: {}", generation, names);
}
} // namespace

const FeatureInfo& feature_info(Feature feature) noexcept {
  const size_t index = feature_index(feature);
  return index < FeatureTable.size() ? FeatureTable[index] : UnknownFeature;
}

const BuiltOptions& default_built_options() noexcept { return DefaultBuiltOptions; }

uint8_t requested_option(const Settings& settings, Feature feature) noexcept {
  switch (feature) {
  case Feature::GBuffer:
  case Feature::DeferredFog:
    return flag(uses_gbuffer(settings));
  case Feature::Pbr:
    return flag(settings.pbr);
  case Feature::Taa:
    return flag(settings.taa);
  case Feature::Ao:
    return enum_level(settings.ao);
  case Feature::Ssil:
    return flag(settings.ssil);
  case Feature::Shadows:
    return enum_level(settings.shadows.quality);
  case Feature::GameShadowSuppression:
    if (settings.shadows.game == GameShadows::Remove) {
      return 2;
    }
    return flag(settings.shadows.game == GameShadows::Auto && sun_shadows(settings));
  case Feature::Gi:
    return enum_level(settings.gi);
  case Feature::Rt:
    return enum_level(settings.rt);
  case Feature::Reflections:
    return enum_level(settings.reflections);
  case Feature::VolumetricFog:
    return enum_level(settings.fog.volumetric);
  case Feature::GodRays:
    return enum_level(settings.fog.godRays);
  case Feature::Clouds:
    return enum_level(settings.sky.clouds);
  case Feature::CloudShadows:
    return flag(settings.sky.cloudShadows);
  case Feature::Grass:
    return enum_level(settings.grass.quality);
  case Feature::GrassTerrain:
    return flag(settings.grass.quality != Quality::Off && settings.grass.terrainGrass);
  case Feature::TerrainParallax:
    return flag(settings.terrain.mode == TerrainDetail::Parallax);
  case Feature::TerrainTessellation:
    if (settings.terrain.mode != TerrainDetail::Tessellated) {
      return 0;
    }
    return std::max<uint8_t>(1, enum_level(settings.terrain.quality));
  case Feature::NativeTrace:
    return flag(settings.rt != RtMode::Off && settings.rtBackend == RtBackend::Dxr);
  case Feature::GpuFoliage:
    return flag(settings.gpuFoliage);
  case Feature::GpuWeather:
    return flag(settings.gpuWeather);
  case Feature::GpuParticles:
    return flag(settings.gpuParticles);
  case Feature::GpuDeform:
    return flag(settings.gpuDeform);
  case Feature::MeshReplacement:
    return flag(settings.meshReplacement);
  case Feature::StaticWorld:
    return enum_level(settings.staticWorld);
  case Feature::TextureMaterials:
    return flag(settings.textureMaterials);
  case Feature::StochasticTiling:
    return flag(settings.stochasticTiling);
  case Feature::LegacyMatMaps:
    return flag(settings.textureMaterials && settings.legacyMatMaps);
  }
  return 0;
}

bool sun_shadows(const Settings& settings) noexcept {
  return settings.shadows.quality != Quality::Off || settings.rt != RtMode::Off;
}

Tier device_tier(const Caps& caps, uint32_t msaaSamples) noexcept {
  const uint32_t budget = caps.limits.maxColorAttachmentBytesPerSample;
  if (!caps.core || msaaSamples != 1 || budget < BaseTierColorBytes) {
    return Tier::None;
  }
  return budget >= PbrTierColorBytes ? Tier::Pbr : Tier::Base;
}

FeatureSupport evaluate_support(Feature feature, uint8_t option, const Settings& context,
                                const SupportEnv& env) noexcept {
  if (!env.moduleEnabled) {
    return {false, Reason::BuildDisabled};
  }
  const size_t index = feature_index(feature);
  if (index >= FeatureCount) {
    return {false, Reason::NotImplemented};
  }
  const FeatureInfo& info = FeatureTable[index];
  const uint8_t level = option == 0 ? 1 : option;
  if (level > info.optionCount || (env.built[index] & (1u << level)) == 0) {
    return {false, Reason::NotImplemented};
  }

  const Caps& deviceCaps = env.caps;
  const uint32_t colorBudget = deviceCaps.limits.maxColorAttachmentBytesPerSample;
  const Tier tier = feature == Feature::Rt && level == 4 ? Tier::Pbr : info.tier;
  if (tier != Tier::None) {
    if (feature != Feature::GBuffer) {
      const FeatureSupport gbuffer = evaluate_support(Feature::GBuffer, 1, context, env);
      if (!gbuffer.supported) {
        return gbuffer;
      }
    } else if (!deviceCaps.core) {
      return {false, Reason::NeedsCore};
    } else if (env.msaaSamples > 1) {
      return {false, Reason::MsaaActive};
    } else if (colorBudget < BaseTierColorBytes) {
      return {false, Reason::ColorBudget};
    }
    if (tier == Tier::Pbr && colorBudget < PbrTierColorBytes) {
      return {false, Reason::ColorBudget};
    }
  }

  const uint32_t fragmentStorage = stage_binding_limits(deviceCaps, ShaderStage::Fragment).storageBuffers;
  switch (feature) {
  case Feature::Shadows:
  case Feature::Clouds:
    if (fragmentStorage == 0) {
      return {false, Reason::NoFragmentStorage};
    }
    break;
  case Feature::VolumetricFog:
    if (fragmentStorage == 0) {
      return {false, Reason::NoFragmentStorage};
    }
    if (level == 4 && (deviceCaps.vramSource == VramSource::Unknown || deviceCaps.vramBytes < HighVramBytes)) {
      return {false, Reason::LowVram};
    }
    break;
  case Feature::TerrainParallax:
    if (deviceCaps.limits.maxSampledTexturesPerShaderStage < CompanionSampledTextures) {
      return {false, Reason::NeedsCompanions};
    }
    break;
  case Feature::TerrainTessellation:
    if (stage_binding_limits(deviceCaps, ShaderStage::Vertex).storageBuffers < TessellationVertexStorageBuffers) {
      return {false, Reason::NoVertexStorage};
    }
    break;
  case Feature::NativeTrace:
    if (deviceCaps.backend != BACKEND_D3D12) {
      return {false, Reason::NeedsD3D12};
    }
    if (!deviceCaps.nativeD3D12) {
      return {false, Reason::NativeUnavailable};
    }
    break;
  default:
    break;
  }

  switch (feature) {
  case Feature::Ssil:
    if (context.ao == Quality::Off) {
      return {false, Reason::NeedsAo};
    }
    break;
  case Feature::Rt:
    if (level == 4 && !context.pbr) {
      return {false, Reason::NeedsPbr};
    }
    break;
  case Feature::Reflections:
    if (!context.pbr) {
      return {false, Reason::NeedsPbr};
    }
    if (level == 2 && context.rt == RtMode::Off) {
      return {false, Reason::NeedsRt};
    }
    break;
  case Feature::GodRays:
    if (level == 3 && context.shadows.quality == Quality::Off) {
      return {false, Reason::NeedsSunShadows};
    }
    break;
  case Feature::NativeTrace:
    if (context.rt == RtMode::Off) {
      return {false, Reason::NeedsRt};
    }
    break;
  default:
    break;
  }

  if (feature == Feature::CloudShadows && sun_shadows(context)) {
    return {true, Reason::RequiredByShadows};
  }
  return {true, Reason::None};
}

Settings sanitize_settings(const Settings& requested) noexcept {
  Settings s = requested;
  sanitize_enum(s.preset, Preset::Ultra, Defaults.preset);
  sanitize_enum(s.exposure, Exposure::Manual, Defaults.exposure);
  sanitize_enum(s.tonemap, Tonemap::AgX, Defaults.tonemap);
  sanitize_enum(s.ao, Quality::Ultra, Defaults.ao);
  sanitize_enum(s.shadows.quality, Quality::Ultra, Defaults.shadows.quality);
  sanitize_enum(s.shadows.game, GameShadows::Remove, Defaults.shadows.game);
  sanitize_enum(s.gi, GiMode::Ddgi, Defaults.gi);
  sanitize_enum(s.rt, RtMode::PathTraceReference, Defaults.rt);
  sanitize_enum(s.rtBackend, RtBackend::Dxr, Defaults.rtBackend);
  sanitize_enum(s.reflections, Reflections::SsrRt, Defaults.reflections);
  sanitize_enum(s.fog.volumetric, Quality::Ultra, Defaults.fog.volumetric);
  sanitize_enum(s.fog.godRays, GodRays::Volumetric, Defaults.fog.godRays);
  sanitize_enum(s.sky.clouds, Quality::Ultra, Defaults.sky.clouds);
  sanitize_enum(s.grass.quality, Quality::Ultra, Defaults.grass.quality);
  sanitize_enum(s.grass.shadows, GrassShadows::Full, Defaults.grass.shadows);
  sanitize_enum(s.terrain.mode, TerrainDetail::Tessellated, Defaults.terrain.mode);
  sanitize_enum(s.terrain.quality, Quality::Ultra, Defaults.terrain.quality);
  sanitize_enum(s.staticWorld, StaticWorld::On, Defaults.staticWorld);

  sanitize_range(s.legacyBlend, Defaults.legacyBlend, 0.f, 1.f);
  sanitize_range(s.taaSharpness, Defaults.taaSharpness, 0.f, 1.f);
  sanitize_range(s.exposureBias, Defaults.exposureBias, -MaxExposureBias, MaxExposureBias);
  if (!std::isfinite(s.sky.cloudStyle) || s.sky.cloudStyle < 0.f) {
    // Negative means the stage profile picks the style
    s.sky.cloudStyle = Defaults.sky.cloudStyle;
  } else {
    s.sky.cloudStyle = std::min(s.sky.cloudStyle, 1.f);
  }
  sanitize_non_negative(s.shadows.legacyGain, Defaults.shadows.legacyGain);
  sanitize_non_negative(s.fog.scattering, Defaults.fog.scattering);
  sanitize_non_negative(s.fog.shaftStrength, Defaults.fog.shaftStrength);
  sanitize_non_negative(s.sky.cloudShadowStrength, Defaults.sky.cloudShadowStrength);
  sanitize_non_negative(s.sky.tuning.coverage, Defaults.sky.tuning.coverage);
  sanitize_non_negative(s.sky.tuning.density, Defaults.sky.tuning.density);
  sanitize_non_negative(s.sky.tuning.altitude, Defaults.sky.tuning.altitude);
  sanitize_non_negative(s.sky.tuning.windSpeed, Defaults.sky.tuning.windSpeed);
  sanitize_non_negative(s.sky.tuning.silverLining, Defaults.sky.tuning.silverLining);
  sanitize_non_negative(s.sky.tuning.powder, Defaults.sky.tuning.powder);
  sanitize_non_negative(s.grass.distance, Defaults.grass.distance);
  sanitize_non_negative(s.grass.density, Defaults.grass.density);
  sanitize_non_negative(s.grass.distantDensity, Defaults.grass.distantDensity);
  sanitize_non_negative(s.grass.windScale, Defaults.grass.windScale);
  sanitize_non_negative(s.terrain.strength, Defaults.terrain.strength);
  sanitize_non_negative(s.terrain.distance, Defaults.terrain.distance);

  if (s.shadows.pages == 0) {
    s.shadows.pages = Defaults.shadows.pages;
  }
  if (s.shadows.pageBudget == 0) {
    s.shadows.pageBudget = Defaults.shadows.pageBudget;
  }
  return s;
}

Settings clamp_settings(const Settings& requested, const SupportEnv& env) noexcept {
  Settings s = sanitize_settings(requested);
  for (const Feature feature : ClampOrder) {
    const uint8_t wanted = requested_option(s, feature);
    if (wanted != 0) {
      uint8_t granted = wanted;
      while (granted != 0 && !evaluate_support(feature, granted, s, env).supported) {
        --granted;
      }
      if (granted != wanted) {
        write_level(s, feature, granted);
      }
    }
    if (feature == Feature::CloudShadows && sun_shadows(s) && !s.sky.cloudShadows &&
        evaluate_support(Feature::CloudShadows, 1, s, env).supported) {
      s.sky.cloudShadows = true;
    }
  }
  return s;
}

Settings submit_request(const Settings& requested) {
  std::lock_guard lock{g_requestMutex};
  Settings preview = clamp_settings(requested, {caps(), g_msaaSamples, g_built, module_enabled()});
  g_lastRequest = requested;
  g_pending = requested;
  g_supportContext = preview;
  return preview;
}

FeatureSupport current_support(Feature feature, uint8_t option, const Settings* context) noexcept {
  BuiltOptions built;
  uint32_t msaaSamples = 1;
  Settings supportContext;
  {
    std::lock_guard lock{g_requestMutex};
    built = g_built;
    msaaSamples = g_msaaSamples;
    if (context == nullptr) {
      supportContext = g_supportContext;
    }
  }
  if (context != nullptr) {
    supportContext = sanitize_settings(*context);
  }
  return evaluate_support(feature, option, supportContext, {caps(), msaaSamples, built, module_enabled()});
}

const Settings& latched_settings() noexcept { return g_latched ? *g_latched : initial_settings(); }

void reset_settings_state(uint32_t msaaSamples) noexcept {
  Settings result;
  {
    std::lock_guard lock{g_requestMutex};
    g_msaaSamples = msaaSamples;
    result = clamp_settings(g_lastRequest, {caps(), msaaSamples, g_built, module_enabled()});
    g_supportContext = result;
    g_pending.reset();
    g_reclamp = false;
  }
  g_latched = result;
}

void shutdown_settings_state() noexcept {
  g_frameState.reset();
  g_latched.reset();
#if AURORA_ENABLE_RENDER
  g_graphEnvironment = {};
  g_externals.reset();
  g_externalEpoch = 0;
  g_sceneDepthSource = DepthSource::Unavailable;
#endif
  std::lock_guard lock{g_requestMutex};
  g_msaaSamples = 1;
}

std::shared_ptr<const FrameState> latch_frame() {
  std::optional<Settings> request;
  BuiltOptions built;
  uint32_t msaaSamples = 1;
  {
    std::lock_guard lock{g_requestMutex};
    request.swap(g_pending);
    if (std::exchange(g_reclamp, false) && !request) {
      request = g_lastRequest;
    }
    built = g_built;
    msaaSamples = g_msaaSamples;
  }
  bool settingsChanged = false;
  if (request) {
    const SupportEnv env{.caps = caps(), .msaaSamples = msaaSamples, .built = built, .moduleEnabled = module_enabled()};
    Settings next = clamp_settings(*request, env);
    if (next != latched_settings()) {
      const FeatureMask previous = enabled_features(latched_settings());
      g_latched = std::move(next);
      ++g_generation;
      settingsChanged = true;
      log_latched_settings(g_generation, previous, *g_latched);
    }
  }
  if (settingsChanged && module_features_effective(latched_settings())) {
    gfx::detail::request_streaming_sizes({.storage = ModuleStorageBufferSize});
  }
  std::shared_ptr<const CompiledGraph> graph = g_frameState ? g_frameState->graph : nullptr;
  bool inspectorOpen = false;
#if AURORA_ENABLE_RENDER
  begin_hint_frame();
  latch_external_passes();
  debug::on_latch();
  const bool passesChanged = consume_graph_dirty();
  // The hash ignores ids and callbacks, so a re-registered external pass is only noticed by the epoch
  const bool externalsChanged = external_pass_epoch() != g_externalEpoch;
  if (externalsChanged) {
    std::shared_ptr<const ExternalPassSet> externals = snapshot_external_passes();
    g_externalEpoch = externals->epoch;
    g_externals = externals->nodes.empty() ? nullptr : std::move(externals);
  }
  GraphEnvironment environment = gather_graph_environment();
  const bool environmentChanged = environment != g_graphEnvironment;
  const DepthSource sceneDepthSource = gfx::current_scene_depth_source();
  const bool sceneDepthChanged = sceneDepthSource != g_sceneDepthSource;
  if (graph == nullptr || settingsChanged || passesChanged || externalsChanged || environmentChanged ||
      sceneDepthChanged) {
    g_graphEnvironment = std::move(environment);
    g_sceneDepthSource = sceneDepthSource;
    // An equal graphHash still carries new state and callbacks, so the new object is always published
    std::shared_ptr<CompiledGraph> compiled = compile_graph(
        latched_settings(), caps(), g_graphEnvironment,
        {.externals = g_externals, .externalEncode = &external_encode, .sceneDepthSource = sceneDepthSource});
    if (compiled) {
      prepare_graph_bindings(*compiled, {&latched_settings(), &caps(), &g_graphEnvironment, sceneDepthSource});
    }
    graph = compiled ? std::move(compiled) : empty_graph();
  }
  latch_pass_params();
  inspectorOpen = g_graphEnvironment.inspectorOpen;
#endif
  // An empty graph compiles to the same shared object, so the flag needs its own check
  if (!g_frameState || g_frameState->settingsGeneration != g_generation || g_frameState->graph != graph ||
      g_frameState->inspectorOpen != inspectorOpen) {
    g_frameState = std::make_shared<const FrameState>(
        FrameState{g_generation, latched_settings(), std::move(graph), inspectorOpen});
  }
#if AURORA_ENABLE_RENDER
  // The FIFO worker is idle until fifo::begin_frame
  scene::apply_frame_state(*g_frameState);
  gx_ext::apply_frame_state(*g_frameState);
#endif
  return g_frameState;
}

const std::shared_ptr<const FrameState>& current_frame_state() noexcept { return g_frameState; }

namespace testing {
void reset() noexcept {
  webgpu::reset_caps();
  {
    std::lock_guard lock{g_requestMutex};
    g_built = DefaultBuiltOptions;
    g_msaaSamples = 1;
    g_lastRequest = {};
    g_supportContext = {};
    g_pending.reset();
    g_reclamp = false;
  }
  g_latched.reset();
  g_generation = 0;
  g_frameState.reset();
  reset_frame_cull();
#if AURORA_ENABLE_RENDER
  g_graphEnvironment = {};
  g_externals.reset();
  g_externalEpoch = 0;
  g_sceneDepthSource = DepthSource::Unavailable;
  clear_passes();
#endif
}

uint64_t settings_generation() noexcept { return g_generation; }

void set_caps(const Caps& caps, uint32_t msaaSamples) noexcept {
  webgpu::set_caps(caps);
  std::lock_guard lock{g_requestMutex};
  g_msaaSamples = msaaSamples;
  g_reclamp = true;
}

void set_built_options(Feature feature, uint32_t optionMask) noexcept {
  const size_t index = feature_index(feature);
  if (index >= FeatureCount) {
    return;
  }
  std::lock_guard lock{g_requestMutex};
  g_built[index] = optionMask;
  g_reclamp = true;
}
} // namespace testing
} // namespace aurora::render::detail

#include <gtest/gtest.h>

#include "render/debug/debug.hpp"
#include "render/module.hpp"
#if AURORA_ENABLE_RENDER
#include "render/geometry/compact.hpp"
#endif
#include "render/settings.hpp"
#include "render_test_caps.hpp"
#include "webgpu/gpu.hpp"

#if !AURORA_ENABLE_RENDER
#include <imgui.h>
#endif

#include <limits>
#include <random>
#include <set>
#include <string_view>

namespace aurora::render::test {
namespace {
using detail::BuiltOptions;
using detail::clamp_settings;
using detail::evaluate_support;
using detail::feature_info;
using detail::requested_option;
using detail::sanitize_settings;

constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
constexpr float Infinity = std::numeric_limits<float>::infinity();

constexpr FeatureSupport Supported{true, Reason::None};

constexpr FeatureSupport unsupported(Reason reason) { return {false, reason}; }

Feature feature_at(size_t index) { return static_cast<Feature>(index); }

template <typename E>
E random_enum(std::mt19937& rng, E last) {
  // Values past the last enumerator exercise sanitize_settings
  return static_cast<E>(std::uniform_int_distribution<int>{0, static_cast<int>(last) + 3}(rng));
}

bool random_bool(std::mt19937& rng) { return std::uniform_int_distribution<int>{0, 1}(rng) != 0; }

float random_float(std::mt19937& rng) {
  switch (std::uniform_int_distribution<int>{0, 9}(rng)) {
  case 0:
    return NaN;
  case 1:
    return Infinity;
  case 2:
    return -Infinity;
  case 3:
    return 0.f;
  default:
    return std::uniform_real_distribution<float>{-20.f, 200.f}(rng);
  }
}

uint32_t random_count(std::mt19937& rng) {
  return random_bool(rng) ? 0 : std::uniform_int_distribution<uint32_t>{1, 4096}(rng);
}

Settings random_settings(std::mt19937& rng) {
  Settings s;
  s.preset = random_enum(rng, Preset::Ultra);
  s.textureMaterials = random_bool(rng);
  s.stochasticTiling = random_bool(rng);
  s.legacyMatMaps = random_bool(rng);
  s.pbr = random_bool(rng);
  s.legacyBlend = random_float(rng);
  s.exposure = random_enum(rng, Exposure::Manual);
  s.exposureBias = random_float(rng);
  s.tonemap = random_enum(rng, Tonemap::AgX);
  s.taa = random_bool(rng);
  s.taaSharpness = random_float(rng);
  s.ao = random_enum(rng, Quality::Ultra);
  s.ssil = random_bool(rng);
  s.shadows.quality = random_enum(rng, Quality::Ultra);
  s.shadows.pages = random_count(rng);
  s.shadows.pageBudget = random_count(rng);
  s.shadows.game = random_enum(rng, GameShadows::Remove);
  s.shadows.legacyGain = random_float(rng);
  s.shadows.keepCues = random_bool(rng);
  s.gi = random_enum(rng, GiMode::Ddgi);
  s.rt = random_enum(rng, RtMode::PathTraceReference);
  s.rtBackend = random_enum(rng, RtBackend::Dxr);
  s.reflections = random_enum(rng, Reflections::SsrRt);
  s.fog.volumetric = random_enum(rng, Quality::Ultra);
  s.fog.godRays = random_enum(rng, GodRays::Volumetric);
  s.fog.scattering = random_float(rng);
  s.fog.shaftStrength = random_float(rng);
  s.fog.legacyMatch = random_bool(rng);
  s.fog.translucents = random_bool(rng);
  s.sky.clouds = random_enum(rng, Quality::Ultra);
  s.sky.cloudStyle = random_float(rng);
  s.sky.cloudShadows = random_bool(rng);
  s.sky.cloudShadowStrength = random_float(rng);
  s.sky.keepCloudDome = random_bool(rng);
  s.sky.tuning.coverage = random_float(rng);
  s.sky.tuning.density = random_float(rng);
  s.sky.tuning.altitude = random_float(rng);
  s.sky.tuning.windSpeed = random_float(rng);
  s.sky.tuning.silverLining = random_float(rng);
  s.sky.tuning.powder = random_float(rng);
  s.grass.quality = random_enum(rng, Quality::Ultra);
  s.grass.replaceGamePatches = random_bool(rng);
  s.grass.terrainGrass = random_bool(rng);
  s.grass.distance = random_float(rng);
  s.grass.density = random_float(rng);
  s.grass.distantDensity = random_float(rng);
  s.grass.shadows = random_enum(rng, GrassShadows::Full);
  s.grass.interaction = random_bool(rng);
  s.grass.windScale = random_float(rng);
  s.grass.legacyShadowTransfer = random_bool(rng);
  s.terrain.mode = random_enum(rng, TerrainDetail::Tessellated);
  s.terrain.quality = random_enum(rng, Quality::Ultra);
  s.terrain.strength = random_float(rng);
  s.terrain.distance = random_float(rng);
  s.terrain.flattenVolumes = random_bool(rng);
  s.gpuFoliage = random_bool(rng);
  s.gpuWeather = random_bool(rng);
  s.gpuParticles = random_bool(rng);
  s.gpuDeform = random_bool(rng);
  s.meshReplacement = random_bool(rng);
  s.staticWorld = random_enum(rng, StaticWorld::On);
  s.debugView = static_cast<uint16_t>(std::uniform_int_distribution<int>{0, 0xFFFF}(rng));
  return s;
}

class RenderApiTest : public ::testing::Test {
protected:
  void SetUp() override { detail::testing::reset(); }
  void TearDown() override { detail::testing::reset(); }
};

TEST_F(RenderApiTest, DefaultsAreOriginal) {
  const Settings defaults{};
  EXPECT_EQ(defaults.preset, Preset::Original);
  for (size_t i = 0; i < FeatureCount; ++i) {
    const Feature feature = feature_at(i);
    EXPECT_EQ(feature_enabled(defaults, feature), feature == Feature::TextureMaterials) << feature_name(feature);
  }
}

TEST_F(RenderApiTest, GameShadowLevels) {
  EXPECT_EQ(requested_option(Settings{}, Feature::GameShadowSuppression), 0);

  Settings vsm;
  vsm.shadows.quality = Quality::Medium;
  EXPECT_EQ(requested_option(vsm, Feature::GameShadowSuppression), 1);

  Settings rtOnly;
  rtOnly.rt = RtMode::Shadows;
  EXPECT_EQ(requested_option(rtOnly, Feature::GameShadowSuppression), 1);

  Settings keep = vsm;
  keep.shadows.game = GameShadows::Keep;
  EXPECT_EQ(requested_option(keep, Feature::GameShadowSuppression), 0);

  Settings remove;
  remove.shadows.game = GameShadows::Remove;
  EXPECT_EQ(requested_option(remove, Feature::GameShadowSuppression), 2);

  const Caps caps = full_caps();
  const BuiltOptions all = built_all();
  EXPECT_EQ(clamp_settings(remove, env(caps, all)).shadows.game, GameShadows::Remove);

  BuiltOptions noSuppression = built_all();
  noSuppression[static_cast<size_t>(Feature::GameShadowSuppression)] = 0;
  EXPECT_EQ(clamp_settings(vsm, env(caps, noSuppression)).shadows.game, GameShadows::Keep);

  const Settings unbuilt = clamp_settings(vsm, env(caps, detail::default_built_options()));
  EXPECT_EQ(unbuilt.shadows.quality, Quality::Off);
  EXPECT_EQ(unbuilt.shadows.game, GameShadows::Auto);
}

TEST_F(RenderApiTest, NamesAreUniqueAndSpelled) {
  std::set<std::string_view> features;
  for (size_t i = 0; i < FeatureCount; ++i) {
    const std::string_view name = feature_name(feature_at(i));
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, "Unknown");
    EXPECT_TRUE(features.insert(name).second) << name;
  }
  std::set<std::string_view> reasons;
  for (size_t i = 0; i <= static_cast<size_t>(Reason::NeedsRt); ++i) {
    const std::string_view name = reason_name(static_cast<Reason>(i));
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, "Unknown");
    EXPECT_TRUE(reasons.insert(name).second) << name;
  }
  EXPECT_EQ(feature_name(Feature::GBuffer), "GBuffer");
  EXPECT_EQ(feature_name(Feature::LegacyMatMaps), "LegacyMatMaps");
  EXPECT_EQ(reason_name(Reason::NotImplemented), "NotImplemented");
  EXPECT_EQ(reason_name(Reason::NeedsRt), "NeedsRt");
  EXPECT_EQ(feature_name(static_cast<Feature>(FeatureCount)), "Unknown");
  EXPECT_EQ(reason_name(static_cast<Reason>(static_cast<size_t>(Reason::NeedsRt) + 1)), "Unknown");
}

TEST_F(RenderApiTest, UnbuiltFeaturesReportNotImplemented) {
  const Caps caps = full_caps();
  const BuiltOptions none = built_none();
  for (size_t i = 0; i < FeatureCount; ++i) {
    const Feature feature = feature_at(i);
    for (uint8_t option = 0; option <= feature_info(feature).optionCount; ++option) {
      EXPECT_EQ(evaluate_support(feature, option, Settings{}, env(caps, none)), unsupported(Reason::NotImplemented))
          << feature_name(feature) << " option " << int{option};
    }
  }
}

TEST_F(RenderApiTest, NotImplementedComesFirst) {
  const Caps compat = compat_caps();
  const BuiltOptions none = built_none();
  const BuiltOptions all = built_all();
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(compat, none)), unsupported(Reason::NotImplemented));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(compat, all)), unsupported(Reason::NeedsCore));

  Caps vulkan = full_caps();
  vulkan.backend = BACKEND_VULKAN;
  Settings rt;
  rt.rt = RtMode::Shadows;
  rt.rtBackend = RtBackend::Dxr;
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, rt, env(vulkan, none)), unsupported(Reason::NotImplemented));
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, rt, env(vulkan, all)), unsupported(Reason::NeedsD3D12));
}

TEST_F(RenderApiTest, ClampTurnsUnbuiltFeaturesOff) {
  const Caps caps = full_caps();
  const BuiltOptions none = built_none();
  const Settings clamped = clamp_settings(everything_on(), env(caps, none));
  for (size_t i = 0; i < FeatureCount; ++i) {
    EXPECT_FALSE(feature_enabled(clamped, feature_at(i))) << feature_name(feature_at(i));
  }
  EXPECT_EQ(clamped.shadows.game, GameShadows::Keep);
  EXPECT_EQ(clamped.terrain.mode, TerrainDetail::Off);
  EXPECT_TRUE(clamped.legacyMatMaps);
  EXPECT_TRUE(clamped.grass.terrainGrass);
  EXPECT_EQ(clamped.rtBackend, RtBackend::Dxr);
}

TEST_F(RenderApiTest, ClampKeepsTuningValues) {
  Settings request;
  request.legacyBlend = 0.5f;
  request.exposureBias = 1.5f;
  request.grass.distance = 80.f;
  request.debugView = 3;
  request.fog.scattering = 2.f;
  const Caps caps = full_caps();
  for (const BuiltOptions& built : {built_none(), built_all()}) {
    const Settings clamped = clamp_settings(request, env(caps, built));
    EXPECT_EQ(clamped.legacyBlend, 0.5f);
    EXPECT_EQ(clamped.exposureBias, 1.5f);
    EXPECT_EQ(clamped.grass.distance, 80.f);
    EXPECT_EQ(clamped.debugView, 3);
    EXPECT_EQ(clamped.fog.scattering, 2.f);
  }
}

TEST_F(RenderApiTest, SanitizeReplacesInvalidValues) {
  Settings request;
  request.legacyBlend = NaN;
  request.exposureBias = Infinity;
  EXPECT_EQ(sanitize_settings(request).legacyBlend, 0.f);
  EXPECT_EQ(sanitize_settings(request).exposureBias, 0.f);

  request.legacyBlend = 2.f;
  request.exposureBias = -20.f;
  request.ao = static_cast<Quality>(9);
  request.preset = static_cast<Preset>(7);
  request.shadows.pages = 0;
  request.sky.cloudStyle = -0.3f;
  const Settings sanitized = sanitize_settings(request);
  EXPECT_EQ(sanitized.legacyBlend, 1.f);
  EXPECT_EQ(sanitized.exposureBias, -8.f);
  EXPECT_EQ(sanitized.ao, Quality::Off);
  EXPECT_EQ(sanitized.preset, Preset::Original);
  EXPECT_EQ(sanitized.shadows.pages, 1024u);
  EXPECT_EQ(sanitized.sky.cloudStyle, -1.f);
}

TEST_F(RenderApiTest, ClampIsIdempotent) {
  struct Input {
    Caps caps;
    uint32_t msaaSamples;
  };
  const Input inputs[] = {
      {full_caps(), 1},
      {compat_caps(), 1},
      {metal32_caps(), 1},
      {full_caps(), 4},
  };
  const BuiltOptions builtSets[] = {built_none(), built_all()};
  std::mt19937 rng{1};
  for (int i = 0; i < 2000; ++i) {
    const Settings request = random_settings(rng);
    for (const Input& input : inputs) {
      for (const BuiltOptions& built : builtSets) {
        const auto supportEnv = env(input.caps, built, input.msaaSamples);
        const Settings once = clamp_settings(request, supportEnv);
        ASSERT_EQ(clamp_settings(once, supportEnv), once) << "iteration " << i;
      }
    }
  }
}

TEST_F(RenderApiTest, BaseTierNeedsCoreMsaaAndBudget) {
  const BuiltOptions all = built_all();
  const Caps compat = compat_caps();
  const Caps full = full_caps();
  Caps smallBudget = full_caps();
  smallBudget.limits.maxColorAttachmentBytesPerSample = 16;
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(compat, all)), unsupported(Reason::NeedsCore));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(full, all, 4)), unsupported(Reason::MsaaActive));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(smallBudget, all)), unsupported(Reason::ColorBudget));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(compat, all, 4)), unsupported(Reason::NeedsCore));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(full, all)), Supported);
}

TEST_F(RenderApiTest, PbrTierNeedsFortyFourBytes) {
  const BuiltOptions all = built_all();
  Caps budget32 = full_caps();
  budget32.limits.maxColorAttachmentBytesPerSample = 32;
  Caps budget64 = full_caps();
  budget64.limits.maxColorAttachmentBytesPerSample = 64;
  Settings pbr;
  pbr.pbr = true;
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(budget32, all)), unsupported(Reason::ColorBudget));
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(budget64, all)), Supported);
  EXPECT_EQ(evaluate_support(Feature::Reflections, 0, pbr, env(budget32, all)), unsupported(Reason::ColorBudget));
}

TEST_F(RenderApiTest, UnbuiltGBufferBlocksBaseFeatures) {
  const Caps caps = full_caps();
  BuiltOptions aoOnly = built_none();
  aoOnly[static_cast<size_t>(Feature::Ao)] = ~0u;
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(caps, aoOnly)), unsupported(Reason::NotImplemented));
}

TEST_F(RenderApiTest, DependencyReasons) {
  const Caps caps = full_caps();
  const BuiltOptions all = built_all();
  const auto supportEnv = env(caps, all);

  Settings ssil;
  ssil.ssil = true;
  EXPECT_EQ(evaluate_support(Feature::Ssil, 0, ssil, supportEnv), unsupported(Reason::NeedsAo));

  Settings reflections;
  reflections.reflections = Reflections::Ssr;
  EXPECT_EQ(evaluate_support(Feature::Reflections, 0, reflections, supportEnv), unsupported(Reason::NeedsPbr));

  Settings rtReflections;
  rtReflections.pbr = true;
  rtReflections.reflections = Reflections::SsrRt;
  EXPECT_EQ(evaluate_support(Feature::Reflections, 2, rtReflections, supportEnv), unsupported(Reason::NeedsRt));

  Settings godRays;
  godRays.fog.godRays = GodRays::Volumetric;
  EXPECT_EQ(evaluate_support(Feature::GodRays, 3, godRays, supportEnv), unsupported(Reason::NeedsSunShadows));

  Settings pathTrace;
  pathTrace.rt = RtMode::PathTraceReference;
  EXPECT_EQ(evaluate_support(Feature::Rt, 4, pathTrace, supportEnv), unsupported(Reason::NeedsPbr));
}

TEST_F(RenderApiTest, DependencyClamps) {
  const Caps caps = full_caps();
  const BuiltOptions all = built_all();
  const auto supportEnv = env(caps, all);

  Settings ssil;
  ssil.ssil = true;
  EXPECT_FALSE(clamp_settings(ssil, supportEnv).ssil);

  Settings reflections;
  reflections.reflections = Reflections::Ssr;
  EXPECT_EQ(clamp_settings(reflections, supportEnv).reflections, Reflections::Off);

  Settings rtReflections;
  rtReflections.pbr = true;
  rtReflections.reflections = Reflections::SsrRt;
  EXPECT_EQ(clamp_settings(rtReflections, supportEnv).reflections, Reflections::Ssr);

  Settings godRays;
  godRays.fog.godRays = GodRays::Volumetric;
  EXPECT_EQ(clamp_settings(godRays, supportEnv).fog.godRays, GodRays::ScreenSpace);

  Settings pathTrace;
  pathTrace.rt = RtMode::PathTraceReference;
  EXPECT_EQ(clamp_settings(pathTrace, supportEnv).rt, RtMode::ShadowsAoReflections);
}

TEST_F(RenderApiTest, CloudShadowsForcedBySunShadows) {
  const Caps caps = full_caps();
  const BuiltOptions all = built_all();
  const auto supportEnv = env(caps, all);

  Settings vsm;
  vsm.shadows.quality = Quality::Medium;
  vsm.sky.cloudShadows = false;
  const Settings clamped = clamp_settings(vsm, supportEnv);
  EXPECT_TRUE(clamped.sky.cloudShadows);
  EXPECT_EQ(evaluate_support(Feature::CloudShadows, 0, clamped, supportEnv),
            (FeatureSupport{true, Reason::RequiredByShadows}));

  Settings rtShadows;
  rtShadows.rt = RtMode::Shadows;
  EXPECT_TRUE(clamp_settings(rtShadows, supportEnv).sky.cloudShadows);
}

TEST_F(RenderApiTest, UltraFogNeedsVram) {
  const BuiltOptions all = built_all();
  Settings fog;
  fog.fog.volumetric = Quality::Ultra;

  Caps unknown = full_caps();
  unknown.vramBytes = 0;
  unknown.vramSource = VramSource::Unknown;
  EXPECT_EQ(evaluate_support(Feature::VolumetricFog, 4, fog, env(unknown, all)), unsupported(Reason::LowVram));
  EXPECT_EQ(clamp_settings(fog, env(unknown, all)).fog.volumetric, Quality::High);

  Caps small = full_caps();
  small.vramBytes = 4 * GiB;
  EXPECT_EQ(evaluate_support(Feature::VolumetricFog, 4, fog, env(small, all)), unsupported(Reason::LowVram));

  const Caps large = full_caps();
  EXPECT_EQ(evaluate_support(Feature::VolumetricFog, 4, fog, env(large, all)), Supported);
}

TEST_F(RenderApiTest, FragmentStorageGates) {
  const BuiltOptions all = built_all();
  Caps caps = full_caps();
  caps.limits.maxStorageBuffersInFragmentStage = 0;
  for (const Feature feature : {Feature::Shadows, Feature::VolumetricFog, Feature::Clouds}) {
    EXPECT_EQ(evaluate_support(feature, 0, Settings{}, env(caps, all)), unsupported(Reason::NoFragmentStorage))
        << feature_name(feature);
  }
}

TEST_F(RenderApiTest, TerrainFallsBackToParallax) {
  const BuiltOptions all = built_all();
  Settings tessellated;
  tessellated.terrain.mode = TerrainDetail::Tessellated;

  Caps caps = full_caps();
  caps.limits.maxStorageBuffersInVertexStage = 2;
  EXPECT_EQ(evaluate_support(Feature::TerrainTessellation, 0, tessellated, env(caps, all)),
            unsupported(Reason::NoVertexStorage));
  EXPECT_EQ(clamp_settings(tessellated, env(caps, all)).terrain.mode, TerrainDetail::Parallax);

  caps.limits.maxSampledTexturesPerShaderStage = 8;
  EXPECT_EQ(clamp_settings(tessellated, env(caps, all)).terrain.mode, TerrainDetail::Off);
  Settings parallax;
  parallax.terrain.mode = TerrainDetail::Parallax;
  EXPECT_EQ(evaluate_support(Feature::TerrainParallax, 0, parallax, env(caps, all)),
            unsupported(Reason::NeedsCompanions));
}

TEST_F(RenderApiTest, NativeTraceReasons) {
  const BuiltOptions all = built_all();
  Settings request;
  request.rt = RtMode::Shadows;
  request.rtBackend = RtBackend::Dxr;

  Caps vulkan = full_caps();
  vulkan.backend = BACKEND_VULKAN;
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, request, env(vulkan, all)), unsupported(Reason::NeedsD3D12));

  const Caps d3d12 = full_caps();
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, request, env(d3d12, all)),
            unsupported(Reason::NativeUnavailable));
  const Settings clamped = clamp_settings(request, env(d3d12, all));
  EXPECT_EQ(clamped.rt, RtMode::Shadows);
  EXPECT_EQ(clamped.rtBackend, RtBackend::Compute);

  Caps native = full_caps();
  native.nativeD3D12 = true;
  Settings rtOff = request;
  rtOff.rt = RtMode::Off;
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, rtOff, env(native, all)), unsupported(Reason::NeedsRt));
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, request, env(native, all)), Supported);
}

TEST_F(RenderApiTest, PresetDoesNotGateExperimental) {
  const Caps caps = full_caps();
  const BuiltOptions all = built_all();
  for (const Preset preset : {Preset::High, Preset::Custom}) {
    Settings request;
    request.preset = preset;
    request.staticWorld = StaticWorld::On;
    request.meshReplacement = true;
    const Settings clamped = clamp_settings(request, env(caps, all));
    EXPECT_EQ(clamped.staticWorld, StaticWorld::On);
    EXPECT_TRUE(clamped.meshReplacement);
    EXPECT_EQ(clamped.preset, preset);
  }
}

TEST_F(RenderApiTest, SubSettingsSurviveParentOff) {
  const Caps caps = full_caps();
  const BuiltOptions none = built_none();

  Settings matMaps;
  matMaps.textureMaterials = true;
  matMaps.legacyMatMaps = true;
  EXPECT_TRUE(clamp_settings(matMaps, env(caps, none)).legacyMatMaps);

  Settings grass;
  grass.grass.quality = Quality::Off;
  grass.grass.terrainGrass = true;
  EXPECT_TRUE(clamp_settings(grass, env(caps, none)).grass.terrainGrass);

  Settings rt;
  rt.rt = RtMode::Off;
  rt.rtBackend = RtBackend::Dxr;
  EXPECT_EQ(clamp_settings(rt, env(caps, none)).rtBackend, RtBackend::Dxr);
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderApiTest, ApplyReturnsPreviewAndSetsContext) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  const Settings preview = apply_settings({.ao = Quality::High, .ssil = true});
  EXPECT_TRUE(preview.ssil);
  EXPECT_EQ(feature_support(Feature::Ssil), Supported);

  apply_settings({.ssil = true});
  EXPECT_EQ(feature_support(Feature::Ssil), unsupported(Reason::NeedsAo));
  Settings context;
  context.ao = Quality::Low;
  EXPECT_EQ(feature_support(Feature::Ssil, 0, context), Supported);
}

TEST_F(RenderApiTest, InitializeHonoursEarlierRequest) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  webgpu::g_graphicsConfig.msaaSamples = 1;
  apply_settings({.ao = Quality::High});
  EXPECT_EQ(effective_settings().ao, Quality::Off);
  detail::initialize();
  EXPECT_EQ(effective_settings().ao, Quality::High);
  detail::shutdown();
  EXPECT_EQ(effective_settings().ao, Quality::Off);
}

TEST_F(RenderApiTest, CompactInitializeWithoutDeviceIsInert) {
  Caps deviceCaps = full_caps();
  deviceCaps.limits.maxComputeWorkgroupSizeX = 256;
  deviceCaps.limits.maxComputeWorkgroupsPerDimension = 65535;
  ASSERT_TRUE(compact::supported(deviceCaps));
  ASSERT_FALSE(webgpu::g_device);
  detail::testing::set_caps(deviceCaps);
  webgpu::g_graphicsConfig.msaaSamples = 1;
  detail::initialize();
  EXPECT_EQ(compact::pipeline(), shaders::NeverReady);
  EXPECT_FALSE(compact::ready());
  const compact::Stats before = compact::stats();
  compact::worker_frame_begin(1);
  EXPECT_EQ(compact::stats().bindGroupsLastFrame, before.bindGroupsLastFrame);
  detail::shutdown();
  EXPECT_EQ(compact::pipeline(), shaders::NeverReady);
}
#endif

TEST_F(RenderApiTest, CapsEmptyAfterReset) {
  detail::testing::set_caps(full_caps());
  ASSERT_TRUE(caps().valid);
  detail::testing::reset();
  EXPECT_EQ(caps(), Caps{});
  EXPECT_FALSE(caps().valid);
}

#if !AURORA_ENABLE_RENDER
TEST(RenderApi, DebugApiLinksWhenModuleOff) {
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = {1280.f, 720.f};
  io.Fonts->Build();
  ImGui::NewFrame();
  bool open = true;
  draw_debug_window(&open);
  draw_debug_window(nullptr);
  ImGui::Render();
  ImGui::DestroyContext();
  EXPECT_TRUE(open);
  EXPECT_TRUE(pass_timings().empty());
  EXPECT_EQ(graph_dump_json(true), debug::NotCompiledGraphJson);
  EXPECT_EQ(graph_dump_json(), debug::NotCompiledGraphJson);
}

TEST_F(RenderApiTest, ModuleOffReportsBuildDisabled) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  for (size_t i = 0; i < FeatureCount; ++i) {
    EXPECT_EQ(feature_support(feature_at(i)), unsupported(Reason::BuildDisabled)) << feature_name(feature_at(i));
  }
  const Settings applied = apply_settings(everything_on());
  for (size_t i = 0; i < FeatureCount; ++i) {
    EXPECT_FALSE(feature_enabled(applied, feature_at(i))) << feature_name(feature_at(i));
  }
}
#endif
} // namespace
} // namespace aurora::render::test

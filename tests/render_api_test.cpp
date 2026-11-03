#include <gtest/gtest.h>

#include "gfx/pipeline_cache.hpp"
#include "gfx/scene_targets.hpp"
#include "render/debug/debug.hpp"
#include "render/frame_state.hpp"
#include "render/module.hpp"
#include "render/ao/ao.hpp"
#if AURORA_ENABLE_RENDER
#include "render/geometry/compact.hpp"
#include "render/post/composite.hpp"
#endif
#include "render/settings.hpp"
#include "render_test_caps.hpp"
#include "webgpu/gpu.hpp"

#if !AURORA_ENABLE_RENDER
#include <imgui.h>
#endif

#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <random>
#include <set>
#include <string_view>
#include <utility>

namespace aurora::render::test {
namespace {
using detail::BuiltOptions;
using detail::clamp_settings;
using detail::evaluate_support;
using detail::feature_info;
using detail::ProbeEntry;
using detail::requested_option;
using detail::sanitize_settings;
using detail::UpscalerProbe;

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
  s.generateMissingMips = random_bool(rng);
  s.pbr = random_bool(rng);
  s.legacyBlend = random_float(rng);
  s.exposure = random_enum(rng, Exposure::Manual);
  s.exposureBias = random_float(rng);
  s.tonemap = random_enum(rng, Tonemap::AgX);
  s.taa = random_bool(rng);
  s.taaSharpness = random_float(rng);
  s.upscaler = random_enum(rng, Upscaler::Auto);
  s.upscalerQuality = random_enum(rng, UpscalerQuality::UltraPerformance);
  s.ao = random_enum(rng, Quality::Ultra);
  s.aoRadius = random_float(rng);
  s.aoStrength = random_float(rng);
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

constexpr std::array AllUpscalers{Upscaler::Off, Upscaler::Fsr, Upscaler::Dlss, Upscaler::Xess, Upscaler::Auto};
constexpr ProbeEntry CompiledOut{};
constexpr ProbeEntry Unknown{.known = false};
constexpr ProbeEntry Available{.support = Supported};
constexpr ProbeEntry NoGpu{.support = unsupported(Reason::UnsupportedGpu)};
constexpr ProbeEntry NoRuntime{.support = unsupported(Reason::RuntimeMissing)};

uint8_t level(Upscaler upscaler) { return static_cast<uint8_t>(upscaler); }

UpscalerProbe stub_probe(ProbeEntry dlss, ProbeEntry xess, bool xessXeFx = false) {
  UpscalerProbe probe;
  probe.native = {dlss, xess};
  probe.xessXeFx = xessXeFx;
  return probe;
}

detail::SupportEnv upscale_env(const Caps& caps, const BuiltOptions& built, const UpscalerProbe& probe,
                               uint32_t msaaSamples = 1) {
  return {caps, msaaSamples, built, true, &probe};
}

Settings upscale_request(Upscaler upscaler) {
  Settings s;
  s.upscaler = upscaler;
  return s;
}

FeatureSupport upscaler_support(Upscaler upscaler, const detail::SupportEnv& supportEnv) {
  return evaluate_support(Feature::Upscaler, level(upscaler), Settings{}, supportEnv);
}

Upscaler clamped_upscaler(Upscaler upscaler, const detail::SupportEnv& supportEnv) {
  return clamp_settings(upscale_request(upscaler), supportEnv).upscaler;
}

Caps vulkan_caps() {
  Caps caps = full_caps();
  caps.backend = BACKEND_VULKAN;
  return caps;
}

Caps no_tier1_caps() {
  Caps caps = full_caps();
  caps.textureFormatsTier1 = false;
  return caps;
}

int g_probeRequests = 0;
detail::testing::Clock::time_point g_fakeNow;

void count_probe_request() noexcept { ++g_probeRequests; }

detail::testing::Clock::time_point fake_now() noexcept { return g_fakeNow; }

class RenderApiTest : public ::testing::Test {
protected:
  void SetUp() override {
    detail::testing::reset();
    g_probeRequests = 0;
  }
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
  for (size_t i = 0; i <= static_cast<size_t>(Reason::Probing); ++i) {
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
  EXPECT_EQ(reason_name(static_cast<Reason>(static_cast<size_t>(Reason::Probing) + 1)), "Unknown");
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

TEST_F(RenderApiTest, MaterialFeaturesBuiltByDefault) {
  const BuiltOptions& built = detail::default_built_options();
  const std::array<std::pair<Caps, uint32_t>, 3> devices{{{full_caps(), 1}, {compat_caps(), 1}, {full_caps(), 4}}};
  for (const auto& [caps, msaa] : devices) {
    EXPECT_EQ(evaluate_support(Feature::TextureMaterials, 1, Settings{}, env(caps, built, msaa)), Supported);
    EXPECT_EQ(evaluate_support(Feature::StochasticTiling, 1, Settings{}, env(caps, built, msaa)), Supported);
    EXPECT_EQ(evaluate_support(Feature::LegacyMatMaps, 1, Settings{}, env(caps, built, msaa)),
              unsupported(Reason::NotImplemented));
    const Settings clamped =
        clamp_settings({.textureMaterials = true, .stochasticTiling = true}, env(caps, built, msaa));
    EXPECT_TRUE(clamped.textureMaterials) << msaa;
    EXPECT_TRUE(clamped.stochasticTiling) << msaa;
  }
}

TEST_F(RenderApiTest, TaaAndGBufferBuiltByDefault) {
  const BuiltOptions& built = detail::default_built_options();
  Caps budget24 = full_caps();
  budget24.limits.maxColorAttachmentBytesPerSample = 24;
  for (const Feature feature : {Feature::GBuffer, Feature::Taa}) {
    EXPECT_EQ(evaluate_support(feature, 1, Settings{}, env(full_caps(), built)), Supported) << feature_name(feature);
    EXPECT_EQ(evaluate_support(feature, 1, Settings{}, env(metal32_caps(), built)), Supported) << feature_name(feature);
    EXPECT_EQ(evaluate_support(feature, 1, Settings{}, env(compat_caps(), built)), unsupported(Reason::NeedsCore))
        << feature_name(feature);
    EXPECT_EQ(evaluate_support(feature, 1, Settings{}, env(full_caps(), built, 4)), unsupported(Reason::MsaaActive))
        << feature_name(feature);
    EXPECT_EQ(evaluate_support(feature, 1, Settings{}, env(budget24, built)), unsupported(Reason::ColorBudget))
        << feature_name(feature);
  }
  EXPECT_EQ(evaluate_support(Feature::DeferredFog, 1, Settings{}, env(full_caps(), built)), Supported);
  EXPECT_EQ(evaluate_support(Feature::DeferredFog, 1, Settings{}, env(metal32_caps(), built)), Supported);
  EXPECT_EQ(evaluate_support(Feature::DeferredFog, 1, Settings{}, env(compat_caps(), built)),
            unsupported(Reason::NeedsCore));
  EXPECT_EQ(evaluate_support(Feature::DeferredFog, 1, Settings{}, env(full_caps(), built, 4)),
            unsupported(Reason::MsaaActive));

  const Settings full = clamp_settings({.taa = true}, env(full_caps(), built));
  EXPECT_TRUE(full.taa);
  EXPECT_TRUE(feature_enabled(full, Feature::GBuffer));
  const Settings compat = clamp_settings({.taa = true}, env(compat_caps(), built));
  EXPECT_FALSE(compat.taa);
  EXPECT_FALSE(feature_enabled(compat, Feature::GBuffer));

#if AURORA_ENABLE_RENDER
  detail::testing::set_caps(full_caps());
  apply_settings({.taa = true});
  detail::latch_frame();
  EXPECT_TRUE(feature_enabled(effective_settings(), Feature::Taa));
  EXPECT_TRUE(feature_enabled(effective_settings(), Feature::DeferredFog));
  EXPECT_TRUE(feature_ready(Feature::DeferredFog));
  EXPECT_TRUE(feature_active(Feature::DeferredFog));
  gfx::shutdown_pipeline_cache();
#endif
}

TEST_F(RenderApiTest, GenerateMissingMipsIsNeverClamped) {
  const BuiltOptions all = built_all();
  const Settings request{.generateMissingMips = true};
  const std::array envs{env(full_caps(), all), env(Caps{}, all), detail::SupportEnv{full_caps(), 1, all, false}};
  for (size_t i = 0; i < envs.size(); ++i) {
    EXPECT_TRUE(clamp_settings(request, envs[i]).generateMissingMips) << "env " << i;
  }
  for (size_t i = 0; i < FeatureCount; ++i) {
    EXPECT_EQ(feature_enabled(request, feature_at(i)), feature_enabled(Settings{}, feature_at(i)))
        << feature_name(feature_at(i));
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

TEST_F(RenderApiTest, PbrDevBuildOnlyThroughEnv) {
  const BuiltOptions& defaults = detail::default_built_options();
  Caps budget32 = full_caps();
  budget32.limits.maxColorAttachmentBytesPerSample = 32;
  Caps budget64 = full_caps();
  budget64.limits.maxColorAttachmentBytesPerSample = 64;
  Settings pbr;
  pbr.pbr = true;
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(budget64, defaults)), unsupported(Reason::NotImplemented));
  EXPECT_FALSE(detail::clamp_settings(pbr, env(budget64, defaults)).pbr);
  EXPECT_EQ(detail::with_pbr_dev_built(defaults, false), defaults);
  const BuiltOptions dev = detail::with_pbr_dev_built(defaults, true);
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(budget64, dev)), Supported);
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(budget32, dev)), unsupported(Reason::ColorBudget));
  EXPECT_TRUE(detail::clamp_settings(pbr, env(budget64, dev)).pbr);
  EXPECT_FALSE(detail::clamp_settings(pbr, env(budget32, dev)).pbr);
}

TEST_F(RenderApiTest, LegacyBlendEnvParses) {
  EXPECT_EQ(detail::parse_legacy_blend("0"), 0.f);
  EXPECT_EQ(detail::parse_legacy_blend("0.25"), 0.25f);
  EXPECT_EQ(detail::parse_legacy_blend("1"), 1.f);
  EXPECT_EQ(detail::parse_legacy_blend("1.5"), 1.5f);
  EXPECT_EQ(detail::parse_legacy_blend("-2"), -2.f);
  for (const std::string_view text : {"", "x", "0.5x", " 0.5", "nan", "NaN", "inf", "-inf", "infinity", "1e40"}) {
    EXPECT_FALSE(detail::parse_legacy_blend(text).has_value()) << text;
  }
  Settings high;
  high.legacyBlend = *detail::parse_legacy_blend("1.5");
  EXPECT_EQ(detail::sanitize_settings(high).legacyBlend, 1.f);
  Settings low;
  low.legacyBlend = *detail::parse_legacy_blend("-2");
  EXPECT_EQ(detail::sanitize_settings(low).legacyBlend, 0.f);
}

TEST_F(RenderApiTest, UnbuiltGBufferBlocksBaseFeatures) {
  const Caps caps = full_caps();
  BuiltOptions aoOnly = built_none();
  aoOnly[static_cast<size_t>(Feature::Ao)] = ~0u;
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(caps, aoOnly)), unsupported(Reason::NotImplemented));
}

TEST_F(RenderApiTest, AoBuiltByDefault) {
  const BuiltOptions& built = detail::default_built_options();
  EXPECT_EQ(built[static_cast<size_t>(Feature::Ao)], detail::ao::BuiltOptionMask);
  for (const Quality quality : {Quality::Low, Quality::Medium, Quality::High, Quality::Ultra}) {
    EXPECT_EQ(evaluate_support(Feature::Ao, static_cast<uint8_t>(quality), Settings{}, env(full_caps(), built)),
              Supported)
        << static_cast<int>(quality);
    EXPECT_EQ(clamp_settings({.ao = quality}, env(full_caps(), built)).ao, quality);
  }
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(compat_caps(), built)), unsupported(Reason::NeedsCore));
  EXPECT_EQ(evaluate_support(Feature::Ao, 0, Settings{}, env(full_caps(), built, 4)), unsupported(Reason::MsaaActive));
  EXPECT_EQ(clamp_settings({.ao = Quality::High}, env(compat_caps(), built)).ao, Quality::Off);
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderApiTest, DeferredFogFollowsComposite) {
  detail::testing::set_caps(full_caps());
  apply_settings({.taa = true});
  detail::latch_frame();
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::DeferredFog));
  EXPECT_TRUE(feature_ready(Feature::DeferredFog));
  EXPECT_TRUE(detail::composite::ready(effective_settings()));

  apply_settings({.ao = Quality::High});
  detail::latch_frame();
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::Ao));
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::DeferredFog));
  EXPECT_FALSE(detail::composite::live());
  EXPECT_FALSE(detail::composite::ready(effective_settings()));
  EXPECT_FALSE(feature_ready(Feature::Ao));
  EXPECT_FALSE(feature_ready(Feature::DeferredFog));
  EXPECT_FALSE(feature_active(Feature::DeferredFog));
  EXPECT_TRUE(feature_ready(Feature::GBuffer));
  EXPECT_TRUE(feature_ready(Feature::Taa));

  apply_settings({});
  detail::latch_frame();
  EXPECT_TRUE(feature_ready(Feature::Ao));
  EXPECT_TRUE(feature_ready(Feature::DeferredFog));
  gfx::shutdown_pipeline_cache();
}

TEST_F(RenderApiTest, PbrReadyFollowsComposite) {
  detail::testing::set_caps(full_caps());
  detail::testing::set_built_options(Feature::Pbr, 1u << 1);
  apply_settings({.pbr = true});
  detail::latch_frame();
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::Pbr));
  EXPECT_NE(webgpu::scene_tier(), webgpu::SceneTier::Pbr);
  EXPECT_FALSE(feature_ready(Feature::Pbr));
  EXPECT_FALSE(feature_active(Feature::Pbr));
  EXPECT_FALSE(detail::composite::ready(effective_settings()));
  webgpu::g_graphicsConfig.gbuffer = true;
  EXPECT_EQ(webgpu::scene_tier(), webgpu::SceneTier::Base);
  EXPECT_FALSE(feature_ready(Feature::Pbr));
  webgpu::g_graphicsConfig.surfaceBuffer = true;
  EXPECT_EQ(webgpu::scene_tier(), webgpu::SceneTier::Pbr);
  EXPECT_FALSE(feature_ready(Feature::Pbr));
  webgpu::g_graphicsConfig.surfaceBuffer = false;
  webgpu::g_graphicsConfig.gbuffer = false;

  apply_settings({});
  detail::latch_frame();
  EXPECT_TRUE(feature_ready(Feature::Pbr));
  gfx::shutdown_pipeline_cache();
}
#endif

TEST_F(RenderApiTest, PbrStaysUnbuiltByDefault) {
  const BuiltOptions& built = detail::default_built_options();
  EXPECT_EQ(built[static_cast<size_t>(Feature::Pbr)], 0u);
  Settings pbr;
  pbr.pbr = true;
  EXPECT_EQ(evaluate_support(Feature::Pbr, 0, pbr, env(full_caps(), built)), unsupported(Reason::NotImplemented));
  EXPECT_FALSE(detail::clamp_settings(pbr, env(full_caps(), built)).pbr);
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderApiTest, AoReadyTracksPipelines) {
  detail::testing::set_caps(full_caps());
  detail::testing::set_built_options(Feature::Ao, detail::ao::BuiltOptionMask);
  detail::latch_frame();
  EXPECT_TRUE(feature_ready(Feature::Ao));
  EXPECT_TRUE(detail::ao::ready(effective_settings()));

  apply_settings({.ao = Quality::High});
  detail::latch_frame();
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::Ao));
  EXPECT_FALSE(detail::ao::pipelines_live());
  EXPECT_FALSE(detail::ao::ready(effective_settings()));
  EXPECT_FALSE(feature_ready(Feature::Ao));
  EXPECT_FALSE(feature_active(Feature::Ao));
  EXPECT_TRUE(feature_ready(Feature::GBuffer));

  detail::testing::set_built_options(Feature::Ao, 0);
  apply_settings({.ao = Quality::High});
  detail::latch_frame();
  EXPECT_FALSE(feature_enabled(effective_settings(), Feature::Ao));
  EXPECT_TRUE(feature_ready(Feature::Ao));
  gfx::shutdown_pipeline_cache();
}
#endif

TEST_F(RenderApiTest, AoTuningSanitized) {
  for (const float bad : {NaN, Infinity, -Infinity}) {
    Settings request;
    request.aoRadius = bad;
    request.aoStrength = bad;
    const Settings sanitized = sanitize_settings(request);
    EXPECT_EQ(sanitized.aoRadius, 0.73f);
    EXPECT_EQ(sanitized.aoStrength, 1.f);
  }
  Settings request;
  request.aoRadius = 0.1f;
  request.aoStrength = -1.f;
  EXPECT_EQ(sanitize_settings(request).aoRadius, 0.25f);
  EXPECT_EQ(sanitize_settings(request).aoStrength, 0.f);
  request.aoRadius = 9.f;
  request.aoStrength = 9.f;
  EXPECT_EQ(sanitize_settings(request).aoRadius, 1.5f);
  EXPECT_EQ(sanitize_settings(request).aoStrength, 1.5f);
  for (const float value : {0.25f, 0.5f, 0.73f, 1.f, 1.234567f, 1.5f}) {
    request.aoRadius = value;
    request.aoStrength = value;
    EXPECT_EQ(std::bit_cast<uint32_t>(sanitize_settings(request).aoRadius), std::bit_cast<uint32_t>(value));
    EXPECT_EQ(std::bit_cast<uint32_t>(sanitize_settings(request).aoStrength), std::bit_cast<uint32_t>(value));
  }
  request.aoStrength = 0.f;
  EXPECT_EQ(sanitize_settings(request).aoStrength, 0.f);
  const Caps caps = full_caps();
  for (const BuiltOptions& built : {built_none(), built_all()}) {
    Settings base;
    base.ao = Quality::High;
    base.taa = true;
    Settings tuned = base;
    tuned.aoRadius = 1.4f;
    tuned.aoStrength = 0.f;
    const Settings a = clamp_settings(base, env(caps, built));
    const Settings b = clamp_settings(tuned, env(caps, built));
    for (size_t feature = 0; feature < FeatureCount; ++feature) {
      const auto f = static_cast<Feature>(feature);
      EXPECT_EQ(feature_enabled(a, f), feature_enabled(b, f)) << feature_name(f);
      EXPECT_EQ(evaluate_support(f, 1, base, env(caps, built)), evaluate_support(f, 1, tuned, env(caps, built)))
          << feature_name(f);
    }
    EXPECT_EQ(b.aoRadius, 1.4f);
    EXPECT_EQ(b.aoStrength, 0.f);
  }
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

TEST_F(RenderApiTest, LegacyMatMapsSetterStores) {
  Settings request;
  request.textureMaterials = true;
  request.legacyMatMaps = true;
  const Settings built = clamp_settings(request, env(full_caps(), built_all()));
  EXPECT_TRUE(built.legacyMatMaps);
  EXPECT_TRUE(feature_enabled(built, Feature::LegacyMatMaps));
  const Settings unbuilt = clamp_settings(request, env(full_caps(), detail::default_built_options()));
  EXPECT_FALSE(unbuilt.legacyMatMaps);
  EXPECT_TRUE(unbuilt.textureMaterials);
  EXPECT_FALSE(feature_enabled(unbuilt, Feature::LegacyMatMaps));
}

TEST_F(RenderApiTest, NeedsCompanionsBelowSixteen) {
  const BuiltOptions all = built_all();
  Settings request;
  request.textureMaterials = true;
  request.legacyMatMaps = true;
  request.pbr = true;
  Caps caps = full_caps();
  EXPECT_EQ(evaluate_support(Feature::LegacyMatMaps, 1, request, env(caps, all)), Supported);
  caps.limits.maxSampledTexturesPerShaderStage = 15;
  EXPECT_EQ(evaluate_support(Feature::LegacyMatMaps, 1, request, env(caps, all)), unsupported(Reason::NeedsCompanions));
  const Settings clamped = clamp_settings(request, env(caps, all));
  EXPECT_FALSE(clamped.legacyMatMaps);
  EXPECT_TRUE(clamped.textureMaterials);
  EXPECT_NE(evaluate_support(Feature::Pbr, 1, request, env(caps, all)), unsupported(Reason::NeedsCompanions));
  EXPECT_TRUE(clamped.pbr);
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

TEST_F(RenderApiTest, UpscalerOptionsNotImplementedUntilBuilt) {
  EXPECT_EQ(feature_info(Feature::Upscaler).tier, detail::Tier::Base);
  EXPECT_EQ(feature_info(Feature::Upscaler).optionCount, 4);
  const BuiltOptions& built = detail::default_built_options();
  EXPECT_EQ(built[static_cast<size_t>(Feature::Upscaler)],
            (1u << level(Upscaler::Fsr)) | (1u << level(Upscaler::Dlss)) | (1u << level(Upscaler::Xess)) |
                (1u << level(Upscaler::Auto)));
  const Caps caps = full_caps();
  const UpscalerProbe probe = stub_probe(Available, Available, true);
  for (const Upscaler upscaler : {Upscaler::Dlss, Upscaler::Xess}) {
    EXPECT_EQ(evaluate_support(Feature::Upscaler, level(upscaler), Settings{}, upscale_env(caps, built, probe)),
              Supported)
        << int{level(upscaler)};
    EXPECT_EQ(clamped_upscaler(upscaler, upscale_env(caps, built, probe)), upscaler) << int{level(upscaler)};
  }
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(caps, built, probe)), Upscaler::Dlss);
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(caps, built, stub_probe(NoGpu, Available, true))),
            Upscaler::Xess);
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(caps, built, stub_probe(NoGpu, Available))), Upscaler::Fsr);
  const BuiltOptions none = built_none();
  for (uint8_t option = 0; option <= feature_info(Feature::Upscaler).optionCount; ++option) {
    EXPECT_EQ(evaluate_support(Feature::Upscaler, option, Settings{}, upscale_env(caps, none, probe)),
              unsupported(Reason::NotImplemented))
        << int{option};
  }
  for (const Upscaler upscaler : AllUpscalers) {
    EXPECT_EQ(clamped_upscaler(upscaler, upscale_env(caps, none, probe)), Upscaler::Off) << int{level(upscaler)};
  }
#if AURORA_ENABLE_RENDER
  detail::testing::set_caps(full_caps());
  detail::set_upscaler_probe(probe);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Dlss)), Supported);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Xess)), Supported);
#endif
}

TEST_F(RenderApiTest, UpscalerFsrSupport) {
  const BuiltOptions& built = detail::default_built_options();
  const UpscalerProbe compiledOut{};
  const Caps full = full_caps();
  for (const Upscaler upscaler : {Upscaler::Off, Upscaler::Fsr, Upscaler::Auto}) {
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(full, built, compiledOut)), Supported) << int{level(upscaler)};
  }
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(vulkan_caps(), built, compiledOut)), Supported);
  EXPECT_EQ(clamped_upscaler(Upscaler::Fsr, upscale_env(full, built, compiledOut)), Upscaler::Fsr);
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(full, built, compiledOut)), Upscaler::Fsr);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(vulkan_caps(), built, compiledOut)), Upscaler::Fsr);
  const Settings clamped = clamp_settings(upscale_request(Upscaler::Fsr), upscale_env(full, built, compiledOut));
  EXPECT_EQ(clamped.upscaler, Upscaler::Fsr);
  EXPECT_FALSE(clamped.taa);
  EXPECT_TRUE(feature_enabled(clamped, Feature::Upscaler));
  EXPECT_TRUE(feature_enabled(clamped, Feature::GBuffer));

  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(compat_caps(), built, compiledOut)),
            unsupported(Reason::NeedsCore));
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(full, built, compiledOut, 4)), unsupported(Reason::MsaaActive));
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(no_tier1_caps(), built, compiledOut)),
            unsupported(Reason::NeedsTextureFormats));
  EXPECT_EQ(upscaler_support(Upscaler::Auto, upscale_env(no_tier1_caps(), built, compiledOut)),
            unsupported(Reason::NeedsTextureFormats));
  EXPECT_EQ(clamped_upscaler(Upscaler::Fsr, upscale_env(compat_caps(), built, compiledOut)), Upscaler::Off);
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(no_tier1_caps(), built, compiledOut)), Upscaler::Off);
#if AURORA_ENABLE_RENDER
  detail::testing::set_caps(full_caps());
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Fsr)), Supported);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Auto)), Supported);
  EXPECT_EQ(g_probeRequests, 0);
#endif
}

TEST_F(RenderApiTest, UpscalerIndependentOfTaa) {
  const Caps caps = full_caps();
  BuiltOptions built = built_all();
  const UpscalerProbe probe = stub_probe(Available, Available, true);
  for (const bool taa : {false, true}) {
    for (const Upscaler upscaler : {Upscaler::Fsr, Upscaler::Dlss, Upscaler::Xess}) {
      Settings request = upscale_request(upscaler);
      request.taa = taa;
      const Settings clamped = clamp_settings(request, upscale_env(caps, built, probe));
      EXPECT_EQ(clamped.upscaler, upscaler) << taa;
      EXPECT_EQ(clamped.taa, taa);
      EXPECT_EQ(evaluate_support(Feature::Upscaler, level(upscaler), request, upscale_env(caps, built, probe)),
                Supported);
    }
  }
  built[static_cast<size_t>(Feature::Taa)] = 0;
  Settings request = upscale_request(Upscaler::Dlss);
  request.taa = true;
  const Settings clamped = clamp_settings(request, upscale_env(caps, built, probe));
  EXPECT_EQ(clamped.upscaler, Upscaler::Dlss);
  EXPECT_FALSE(clamped.taa);
  EXPECT_TRUE(feature_enabled(clamped, Feature::GBuffer));
}

TEST_F(RenderApiTest, UpscalerGatesThroughGBuffer) {
  const BuiltOptions all = built_all();
  const UpscalerProbe probe = stub_probe(Available, Available, true);
  const Caps compat = compat_caps();
  const Caps full = full_caps();
  Caps budget24 = full_caps();
  budget24.limits.maxColorAttachmentBytesPerSample = 24;
  BuiltOptions noGBuffer = built_all();
  noGBuffer[static_cast<size_t>(Feature::GBuffer)] = 0;
  for (const Upscaler upscaler : {Upscaler::Fsr, Upscaler::Dlss, Upscaler::Xess, Upscaler::Auto}) {
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(compat, all, probe)), unsupported(Reason::NeedsCore));
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(full, all, probe, 4)), unsupported(Reason::MsaaActive));
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(budget24, all, probe)), unsupported(Reason::ColorBudget));
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(full, noGBuffer, probe)), unsupported(Reason::NotImplemented));
    EXPECT_EQ(clamped_upscaler(upscaler, upscale_env(compat, all, probe)), Upscaler::Off);
    EXPECT_EQ(clamped_upscaler(upscaler, upscale_env(full, all, probe, 4)), Upscaler::Off);
    EXPECT_TRUE(feature_enabled(upscale_request(upscaler), Feature::GBuffer));
  }
  EXPECT_FALSE(feature_enabled(clamp_settings(upscale_request(Upscaler::Fsr), upscale_env(compat, all, probe)),
                               Feature::GBuffer));
}

TEST_F(RenderApiTest, UpscalerFsrNeedsTier1) {
  const BuiltOptions all = built_all();
  const Caps noTier1 = no_tier1_caps();
  const UpscalerProbe native = stub_probe(Available, Available, true);
  const UpscalerProbe none = stub_probe(NoGpu, NoGpu);
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(noTier1, all, none)), unsupported(Reason::NeedsTextureFormats));
  EXPECT_EQ(clamped_upscaler(Upscaler::Fsr, upscale_env(noTier1, all, none)), Upscaler::Off);
  EXPECT_EQ(upscaler_support(Upscaler::Auto, upscale_env(noTier1, all, none)),
            unsupported(Reason::NeedsTextureFormats));
  EXPECT_EQ(upscaler_support(Upscaler::Dlss, upscale_env(noTier1, all, native)), Supported);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(noTier1, all, native)), Upscaler::Dlss);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(noTier1, all, none)), Upscaler::Off);
  const Caps full = full_caps();
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(full, all, none)), Supported);
}

TEST_F(RenderApiTest, BuildDisabledBeforeNeedsD3D12) {
  const BuiltOptions all = built_all();
  const Caps vulkan = vulkan_caps();
  const Caps d3d12 = full_caps();
  const UpscalerProbe compiledOut{};
  for (const Upscaler upscaler : {Upscaler::Dlss, Upscaler::Xess}) {
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(vulkan, all, compiledOut)), unsupported(Reason::BuildDisabled));
    EXPECT_EQ(upscaler_support(upscaler, upscale_env(d3d12, all, compiledOut)), unsupported(Reason::BuildDisabled));
    EXPECT_EQ(upscaler_support(upscaler, env(vulkan, all)), unsupported(Reason::BuildDisabled));
    for (const ProbeEntry entry : {Available, Unknown, NoGpu}) {
      const UpscalerProbe compiledIn = stub_probe(entry, entry, true);
      EXPECT_EQ(upscaler_support(upscaler, upscale_env(vulkan, all, compiledIn)), unsupported(Reason::NeedsD3D12));
    }
  }
  EXPECT_EQ(upscaler_support(Upscaler::Fsr, upscale_env(vulkan, all, compiledOut)), Supported);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(vulkan, all, compiledOut)), Upscaler::Fsr);
}

TEST_F(RenderApiTest, ClampNeverCrossesVendor) {
  const BuiltOptions all = built_all();
  const Caps full = full_caps();
  const Caps noTier1 = no_tier1_caps();
  const UpscalerProbe onlyXess = stub_probe(NoGpu, Available, true);
  const UpscalerProbe onlyDlss = stub_probe(Available, NoRuntime);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(full, all, onlyXess)), Upscaler::Fsr);
  EXPECT_EQ(clamped_upscaler(Upscaler::Xess, upscale_env(full, all, onlyDlss)), Upscaler::Fsr);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(noTier1, all, onlyXess)), Upscaler::Off);
  EXPECT_EQ(clamped_upscaler(Upscaler::Xess, upscale_env(noTier1, all, onlyDlss)), Upscaler::Off);

  const std::array entries{CompiledOut, Unknown, Available, NoGpu};
  const std::array caps{full_caps(), vulkan_caps(), no_tier1_caps()};
  for (const Caps& deviceCaps : caps) {
    for (const ProbeEntry dlss : entries) {
      for (const ProbeEntry xess : entries) {
        for (const bool xessXeFx : {false, true}) {
          const UpscalerProbe probe = stub_probe(dlss, xess, xessXeFx);
          for (const Upscaler upscaler : {Upscaler::Fsr, Upscaler::Dlss, Upscaler::Xess}) {
            const Upscaler clamped = clamped_upscaler(upscaler, upscale_env(deviceCaps, all, probe));
            EXPECT_TRUE(clamped == upscaler || clamped == Upscaler::Fsr || clamped == Upscaler::Off)
                << int{level(upscaler)} << " -> " << int{level(clamped)};
          }
        }
      }
    }
  }
}

TEST_F(RenderApiTest, AutoResolvesByProbe) {
  const BuiltOptions all = built_all();
  const Caps full = full_caps();
  const auto resolve = [&](const Caps& deviceCaps, const UpscalerProbe& probe) {
    return clamped_upscaler(Upscaler::Auto, upscale_env(deviceCaps, all, probe));
  };
  EXPECT_EQ(resolve(full, stub_probe(Available, Available, true)), Upscaler::Dlss);
  EXPECT_EQ(resolve(full, stub_probe(NoGpu, Available, true)), Upscaler::Xess);
  EXPECT_EQ(resolve(full, stub_probe(CompiledOut, Available, true)), Upscaler::Xess);
  EXPECT_EQ(resolve(full, stub_probe(NoGpu, Available, false)), Upscaler::Fsr);
  EXPECT_EQ(resolve(full, stub_probe(NoGpu, NoRuntime, true)), Upscaler::Fsr);
  EXPECT_EQ(resolve(full, UpscalerProbe{}), Upscaler::Fsr);
  EXPECT_EQ(resolve(vulkan_caps(), stub_probe(Available, Available, true)), Upscaler::Fsr);
  EXPECT_EQ(resolve(no_tier1_caps(), stub_probe(NoGpu, Available, false)), Upscaler::Off);
  EXPECT_EQ(resolve(no_tier1_caps(), stub_probe(NoGpu, Available, true)), Upscaler::Xess);

  BuiltOptions noAuto = built_all();
  noAuto[static_cast<size_t>(Feature::Upscaler)] = (1u << 1) | (1u << 2) | (1u << 3);
  const UpscalerProbe probe = stub_probe(Available, Available, true);
  EXPECT_EQ(upscaler_support(Upscaler::Auto, upscale_env(full, noAuto, probe)), unsupported(Reason::NotImplemented));
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(full, noAuto, probe)), Upscaler::Off);
}

TEST_F(RenderApiTest, SupportAgreesWithClamp) {
  const std::array entries{CompiledOut, Unknown, Available, NoGpu, NoRuntime};
  struct Device {
    Caps caps;
    uint32_t msaaSamples;
  };
  const std::array devices{Device{full_caps(), 1}, Device{vulkan_caps(), 1}, Device{no_tier1_caps(), 1},
                           Device{compat_caps(), 1}, Device{full_caps(), 4}};
  constexpr size_t UpscalerIndex = static_cast<size_t>(Feature::Upscaler);
  std::array<BuiltOptions, 5> builtSets{built_all(), built_none(), built_all(), built_all(), built_all()};
  builtSets[2][UpscalerIndex] = (1u << 1) | (1u << 4);
  builtSets[3][UpscalerIndex] = 1u << 1;
  builtSets[4][UpscalerIndex] = (1u << 2) | (1u << 3) | (1u << 4);
  for (const Device& device : devices) {
    for (const BuiltOptions& built : builtSets) {
      for (const ProbeEntry dlss : entries) {
        for (const ProbeEntry xess : entries) {
          for (const bool xessXeFx : {false, true}) {
            const UpscalerProbe probe = stub_probe(dlss, xess, xessXeFx);
            const auto supportEnv = upscale_env(device.caps, built, probe, device.msaaSamples);
            for (const Upscaler upscaler : {Upscaler::Fsr, Upscaler::Dlss, Upscaler::Xess, Upscaler::Auto}) {
              const FeatureSupport support = upscaler_support(upscaler, supportEnv);
              const Settings clamped = clamp_settings(upscale_request(upscaler), supportEnv);
              const bool kept =
                  upscaler == Upscaler::Auto ? clamped.upscaler != Upscaler::Off : clamped.upscaler == upscaler;
              ASSERT_EQ(support.supported, kept) << int{level(upscaler)} << " reason " << reason_name(support.reason);
              ASSERT_NE(clamped.upscaler, Upscaler::Auto);
              ASSERT_EQ(clamp_settings(clamped, supportEnv), clamped);
            }
          }
        }
      }
    }
  }
}

TEST_F(RenderApiTest, ProbingWhileUnknown) {
  const BuiltOptions all = built_all();
  const Caps full = full_caps();
  const UpscalerProbe unknown = stub_probe(Unknown, Unknown);
  EXPECT_EQ(upscaler_support(Upscaler::Dlss, upscale_env(full, all, unknown)), unsupported(Reason::Probing));
  EXPECT_EQ(upscaler_support(Upscaler::Xess, upscale_env(full, all, unknown)), unsupported(Reason::Probing));
  EXPECT_EQ(upscaler_support(Upscaler::Auto, upscale_env(full, all, unknown)), Supported);
  EXPECT_EQ(clamped_upscaler(Upscaler::Dlss, upscale_env(full, all, unknown)), Upscaler::Fsr);
  EXPECT_EQ(clamped_upscaler(Upscaler::Auto, upscale_env(full, all, unknown)), Upscaler::Fsr);
  const Caps vulkan = vulkan_caps();
  EXPECT_EQ(upscaler_support(Upscaler::Dlss, upscale_env(vulkan, all, unknown)), unsupported(Reason::NeedsD3D12));

#if AURORA_ENABLE_RENDER
  detail::testing::set_caps(full_caps());
  enable_all_built();
  detail::set_probe_requester(&count_probe_request);
  detail::set_upscaler_probe(unknown);
  EXPECT_EQ(feature_support(Feature::Upscaler, 0), Supported);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Fsr)), Supported);
  EXPECT_EQ(g_probeRequests, 0);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Dlss)), unsupported(Reason::Probing));
  EXPECT_EQ(g_probeRequests, 1);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Auto)), Supported);
  EXPECT_EQ(g_probeRequests, 2);
  detail::set_upscaler_probe(stub_probe(Available, NoGpu));
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Dlss)), Supported);
  EXPECT_EQ(feature_support(Feature::Upscaler, level(Upscaler::Auto)), Supported);
  EXPECT_EQ(g_probeRequests, 2);
#endif
}

TEST_F(RenderApiTest, ReasonValuesAppendOnly) {
  constexpr std::array<std::pair<Reason, int>, 11> Values{{
      {Reason::NeedsD3D12, 7},
      {Reason::NativeUnavailable, 8},
      {Reason::NativeDisabledAfterCrash, 9},
      {Reason::BuildDisabled, 16},
      {Reason::NotImplemented, 17},
      {Reason::NeedsRt, 21},
      {Reason::NeedsTextureFormats, 22},
      {Reason::RuntimeMissing, 23},
      {Reason::DriverTooOld, 24},
      {Reason::UnsupportedGpu, 25},
      {Reason::OutputTooLarge, 26},
  }};
  for (const auto& [reason, value] : Values) {
    EXPECT_EQ(static_cast<int>(reason), value) << reason_name(reason);
  }
  EXPECT_EQ(static_cast<int>(Reason::Probing), 27);
  EXPECT_EQ(static_cast<int>(Feature::LegacyMatMaps), 28);
  EXPECT_EQ(static_cast<int>(Feature::Upscaler), 29);
  EXPECT_EQ(FeatureCount, 30u);
  EXPECT_EQ(feature_name(Feature::Upscaler), "Upscaler");
}

TEST_F(RenderApiTest, SettingsDefaultsUpscalerOff) {
  const Settings defaults{};
  EXPECT_EQ(defaults.upscaler, Upscaler::Off);
  EXPECT_EQ(defaults.upscalerQuality, UpscalerQuality::Quality);
  EXPECT_FALSE(feature_enabled(defaults, Feature::Upscaler));

  Settings invalid;
  invalid.upscaler = static_cast<Upscaler>(9);
  invalid.upscalerQuality = static_cast<UpscalerQuality>(7);
  EXPECT_EQ(sanitize_settings(invalid).upscaler, Upscaler::Off);
  EXPECT_EQ(sanitize_settings(invalid).upscalerQuality, UpscalerQuality::Quality);

  const BuiltOptions all = built_all();
  const Caps caps = full_caps();
  const UpscalerProbe probe = stub_probe(Available, Available, true);
  Settings quality;
  quality.upscalerQuality = UpscalerQuality::Performance;
  EXPECT_EQ(clamp_settings(quality, upscale_env(caps, all, probe)).upscalerQuality, UpscalerQuality::Performance);
  EXPECT_EQ(clamp_settings(defaults, upscale_env(caps, all, probe)).upscaler, Upscaler::Off);

  const UpscaleStatus status = upscale_status();
  EXPECT_EQ(status.active, Upscaler::Off);
  EXPECT_FALSE(status.ready);
  EXPECT_TRUE(status.displayName.empty());
}

#if AURORA_ENABLE_RENDER
TEST_F(RenderApiTest, UnknownProbeHoldsRequest) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  detail::set_probe_requester(&count_probe_request);
  detail::set_upscaler_probe(stub_probe(Unknown, Unknown));
  apply_settings({.ao = Quality::Low});
  detail::latch_frame();
  ASSERT_EQ(effective_settings().ao, Quality::Low);

  const Settings request{.upscaler = Upscaler::Dlss, .ao = Quality::High};
  for (int frame = 0; frame < 5; ++frame) {
    apply_settings(request);
    detail::latch_frame();
    EXPECT_EQ(effective_settings().upscaler, Upscaler::Off) << frame;
    EXPECT_EQ(effective_settings().ao, Quality::Low) << frame;
  }
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Off);
  EXPECT_EQ(g_probeRequests, 1);

  detail::set_upscaler_probe(stub_probe(Available, Unknown));
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Dlss);
  EXPECT_EQ(effective_settings().ao, Quality::High);
  EXPECT_EQ(g_probeRequests, 1);
  gfx::shutdown_pipeline_cache();
}

TEST_F(RenderApiTest, UnknownProbeLatchesAfterTimeout) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  detail::set_probe_requester(&count_probe_request);
  detail::testing::set_clock(&fake_now);
  g_fakeNow = {};
  detail::set_upscaler_probe(stub_probe(Unknown, Unknown));

  const Settings request{.upscaler = Upscaler::Dlss};
  apply_settings(request);
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Off);
  g_fakeNow += std::chrono::milliseconds{1999};
  apply_settings(request);
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Off);
  g_fakeNow += std::chrono::milliseconds{1};
  apply_settings(request);
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Fsr);
  g_fakeNow += std::chrono::seconds{5};
  apply_settings(request);
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Fsr);
  EXPECT_EQ(g_probeRequests, 1);

  detail::set_upscaler_probe(stub_probe(Available, Unknown));
  detail::latch_frame();
  EXPECT_EQ(effective_settings().upscaler, Upscaler::Dlss);
  gfx::shutdown_pipeline_cache();
}
#endif

TEST(ReasonNames, CoverEveryReason) {
  std::set<std::string_view> names;
  for (size_t i = 0; i <= static_cast<size_t>(Reason::Probing); ++i) {
    const std::string_view name = reason_name(static_cast<Reason>(i));
    EXPECT_FALSE(name.empty()) << i;
    EXPECT_NE(name, "Unknown") << i;
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  EXPECT_EQ(reason_name(Reason::NeedsTextureFormats), "NeedsTextureFormats");
  EXPECT_EQ(reason_name(Reason::RuntimeMissing), "RuntimeMissing");
  EXPECT_EQ(reason_name(Reason::DriverTooOld), "DriverTooOld");
  EXPECT_EQ(reason_name(Reason::UnsupportedGpu), "UnsupportedGpu");
  EXPECT_EQ(reason_name(Reason::OutputTooLarge), "OutputTooLarge");
  EXPECT_EQ(reason_name(Reason::Probing), "Probing");
  EXPECT_EQ(reason_name(static_cast<Reason>(static_cast<size_t>(Reason::Probing) + 1)), "Unknown");
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

#if AURORA_ENABLE_RENDER
TEST_F(RenderApiTest, FeatureReadyFollowsSceneLayout) {
  detail::testing::set_caps(full_caps());
  enable_all_built();
  apply_settings({.ao = Quality::High});
  detail::latch_frame();
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::GBuffer));
  ASSERT_NE(feature_info(Feature::GBuffer).tier, detail::Tier::None);
  EXPECT_TRUE(feature_ready(Feature::GBuffer));

  // Leaving the tier warms the legacy layout, which needs no device
  webgpu::g_graphicsConfig.gbuffer = true;
  gfx::detail::latch_scene_layout(gfx::SceneTier::Legacy);
  ASSERT_TRUE(gfx::scene_layout_transitioning());
  EXPECT_FALSE(feature_ready(Feature::GBuffer));
  EXPECT_FALSE(feature_active(Feature::GBuffer));
  EXPECT_FALSE(feature_ready(Feature::Ao));
  EXPECT_TRUE(feature_ready(Feature::TextureMaterials));
  EXPECT_TRUE(feature_ready(Feature::Ssil));

  gfx::detail::latch_scene_layout(gfx::SceneTier::Base);
  EXPECT_FALSE(gfx::scene_layout_transitioning());
  EXPECT_TRUE(feature_ready(Feature::GBuffer));
  webgpu::g_graphicsConfig.gbuffer = false;
  gfx::shutdown_pipeline_cache();
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

#include <gtest/gtest.h>

#include "graph_checks.hpp"
#include "hint_checks.hpp"
#include "options.hpp"
#include "procedural.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace render_demo;

namespace {
std::optional<Options> parse(std::vector<const char*> args, const char* envBackend = nullptr) {
  args.insert(args.begin(), "render_demo");
  std::string error;
  return parse_options(static_cast<int>(args.size()), args.data(), envBackend, error);
}
} // namespace

TEST(RenderDemoProcedural, TiledSizeRoundsUpToBlocks) {
  EXPECT_EQ(rgba8_tiled_size(64, 64), 16384u);
  EXPECT_EQ(rgba8_tiled_size(4, 4), 64u);
  EXPECT_EQ(rgba8_tiled_size(1, 1), 64u);
  EXPECT_EQ(rgba8_tiled_size(8, 4), 128u);
  EXPECT_EQ(rgba8_tiled_size(3, 5), 128u);
}

TEST(RenderDemoProcedural, EncodeRgba8TileLayout) {
  constexpr u32 Width = 8;
  constexpr u32 Height = 4;
  std::vector<u8> rgba(Width * Height * 4);
  for (u32 y = 0; y < Height; ++y) {
    for (u32 x = 0; x < Width; ++x) {
      u8* texel = rgba.data() + (y * Width + x) * 4;
      texel[0] = static_cast<u8>(x);
      texel[1] = static_cast<u8>(y);
      texel[2] = static_cast<u8>(x + y);
      texel[3] = static_cast<u8>(0x80 | x);
    }
  }
  std::vector<u8> out(rgba8_tiled_size(Width, Height), 0xCD);
  encode_rgba8_tiled(rgba, Width, Height, out);
  for (u32 block = 0; block < 2; ++block) {
    for (u32 y = 0; y < 4; ++y) {
      for (u32 x = 0; x < 4; ++x) {
        const u32 srcX = block * 4 + x;
        const u32 i = block * 64 + 2 * (4 * y + x);
        EXPECT_EQ(out[i], static_cast<u8>(0x80 | srcX));
        EXPECT_EQ(out[i + 1], static_cast<u8>(srcX));
        EXPECT_EQ(out[i + 32], static_cast<u8>(y));
        EXPECT_EQ(out[i + 33], static_cast<u8>(srcX + y));
      }
    }
  }
}

TEST(RenderDemoProcedural, EncodeRgba8PadsPartialBlocks) {
  std::vector<u8> rgba(3 * 3 * 4, 0xFF);
  std::vector<u8> out(rgba8_tiled_size(3, 3), 0xCD);
  encode_rgba8_tiled(rgba, 3, 3, out);
  for (u32 y = 0; y < 4; ++y) {
    for (u32 x = 0; x < 4; ++x) {
      const bool inside = x < 3 && y < 3;
      const u32 i = 2 * (4 * y + x);
      for (const u32 offset : {i, i + 1, i + 32, i + 33}) {
        EXPECT_EQ(out[offset], inside ? 0xFF : 0x00) << "x=" << x << " y=" << y;
      }
    }
  }
}

TEST(RenderDemoProcedural, GroundTextureMipChainSize) { EXPECT_EQ(make_ground_texture(64, 7).size(), 21952u); }

TEST(RenderDemoProcedural, GroundTextureFirstTexelIsGrout) {
  const auto texture = make_ground_texture(64, 7);
  EXPECT_EQ(texture[0], 255);
  EXPECT_EQ(texture[1], 44);
  EXPECT_EQ(texture[32], 64);
  EXPECT_EQ(texture[33], 40);
}

TEST(RenderDemoProcedural, TentacleTopology) {
  const auto strips = make_tentacle(3);
  ASSERT_EQ(strips.size(), 8u);
  for (size_t band = 0; band < strips.size(); ++band) {
    ASSERT_EQ(strips[band].size(), 22u);
    for (size_t i = 0; i < strips[band].size(); ++i) {
      const auto& v = strips[band][i];
      const u32 ring = static_cast<u32>(i % 2 == 0 ? band + 1 : band);
      const u32 bone = std::min(ring / 2, 3u);
      EXPECT_EQ(v.matrixIndex, static_cast<u8>(3 + 3 * bone));
      EXPECT_NEAR(v.position[1], 0.75f * static_cast<f32>(ring) - 1.5f * static_cast<f32>(bone), 1e-5f);
      const f32 length = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2]);
      EXPECT_NEAR(length, 1.f, 1e-5f);
    }
  }
}

TEST(RenderDemoOptions, Defaults) {
  const auto options = parse({});
  ASSERT_TRUE(options);
  EXPECT_EQ(options->backend, BACKEND_AUTO);
  EXPECT_EQ(options->profile, FEATURE_PROFILE_AUTO);
  EXPECT_EQ(options->frames, 0u);
  EXPECT_EQ(options->timeoutSeconds, 0u);
  EXPECT_EQ(options->width, 1280u);
  EXPECT_EQ(options->height, 960u);
  EXPECT_TRUE(options->dataDir.empty());
  EXPECT_TRUE(options->tracePath.empty());
  EXPECT_EQ(options->traceStart, 2u);
  EXPECT_EQ(options->traceFrames, 4u);
  EXPECT_FALSE(options->failOnError);
  EXPECT_FALSE(options->help);
}

TEST(RenderDemoOptions, ParsesEveryFlag) {
  const auto options = parse(
      {"--frames",      "120",    "--timeout",      "60",         "--backend",       "null",    "--profile",
       "metal32",       "--size", "800x600",        "--data-dir", "smoke/out",       "--trace", "out/demo.gxtrace",
       "--trace-start", "10",     "--trace-frames", "20",         "--fail-on-error", "--help"});
  ASSERT_TRUE(options);
  EXPECT_EQ(options->frames, 120u);
  EXPECT_EQ(options->timeoutSeconds, 60u);
  EXPECT_EQ(options->backend, BACKEND_NULL);
  EXPECT_EQ(options->profile, FEATURE_PROFILE_METAL32);
  EXPECT_EQ(options->width, 800u);
  EXPECT_EQ(options->height, 600u);
  EXPECT_EQ(options->dataDir, "smoke/out");
  EXPECT_EQ(options->tracePath, "out/demo.gxtrace");
  EXPECT_EQ(options->traceStart, 10u);
  EXPECT_EQ(options->traceFrames, 20u);
  EXPECT_TRUE(options->failOnError);
  EXPECT_TRUE(options->help);

  const auto shortHelp = parse({"-h"});
  ASSERT_TRUE(shortHelp);
  EXPECT_TRUE(shortHelp->help);
}

TEST(RenderDemoOptions, EnvironmentBackendIsFallback) {
  const auto fromEnv = parse({}, "null");
  ASSERT_TRUE(fromEnv);
  EXPECT_EQ(fromEnv->backend, BACKEND_NULL);

  const auto flagWins = parse({"--backend", "vulkan"}, "null");
  ASSERT_TRUE(flagWins);
  EXPECT_EQ(flagWins->backend, BACKEND_VULKAN);
}

TEST(RenderDemoOptions, InvalidEnvironmentBackendOnlyFailsWhenUsed) {
  EXPECT_FALSE(parse({}, "dx9"));
  const auto flagWins = parse({"--backend", "null"}, "dx9");
  ASSERT_TRUE(flagWins);
  EXPECT_EQ(flagWins->backend, BACKEND_NULL);
  const auto help = parse({"--help"}, "dx9");
  ASSERT_TRUE(help);
  EXPECT_TRUE(help->help);
}

TEST(RenderDemoOptions, RejectsInvalidInput) {
  EXPECT_FALSE(parse({"--frames", "-1"}));
  EXPECT_FALSE(parse({"--frames", "abc"}));
  EXPECT_FALSE(parse({"--frames"}));
  EXPECT_FALSE(parse({"--backend", "dx9"}));
  EXPECT_FALSE(parse({"--profile", "fast"}));
  EXPECT_FALSE(parse({"--size", "10"}));
  EXPECT_FALSE(parse({"--size", "0x10"}));
  EXPECT_FALSE(parse({"--bogus"}));
  EXPECT_FALSE(parse({"--trace-frames", "0"}));
  EXPECT_FALSE(parse({"--trace", ""}));
  EXPECT_FALSE(parse({"--trace-start", "x"}));
}

TEST(RenderDemoOptions, TraceWindowMustFitFrames) {
  EXPECT_FALSE(parse({"--frames", "5", "--trace", "t.gxtrace"}));
  EXPECT_FALSE(parse({"--frames", "29", "--trace", "t.gxtrace", "--trace-start", "10", "--trace-frames", "20"}));
  EXPECT_TRUE(parse({"--frames", "30", "--trace", "t.gxtrace", "--trace-start", "10", "--trace-frames", "20"}));
  EXPECT_TRUE(parse({"--frames", "0", "--trace", "t.gxtrace", "--trace-start", "10", "--trace-frames", "20"}));
  EXPECT_TRUE(parse({"--frames", "5", "--trace-start", "10"}));
  EXPECT_TRUE(parse({"--frames", "6", "--trace", "t.gxtrace"}));
}

TEST(RenderDemoOptions, ProfileNamesMatchAurora) {
  EXPECT_EQ(parse_profile("full"), FEATURE_PROFILE_FULL);
  EXPECT_EQ(parse_profile("compat"), FEATURE_PROFILE_COMPAT);
  EXPECT_EQ(parse_profile("metal"), FEATURE_PROFILE_METAL);
  EXPECT_EQ(parse_profile("metal32"), FEATURE_PROFILE_METAL32);
  EXPECT_FALSE(parse_profile("auto"));
}

namespace {
std::string parse_error(std::vector<const char*> args) {
  args.insert(args.begin(), "render_demo");
  std::string error;
  EXPECT_FALSE(parse_options(static_cast<int>(args.size()), args.data(), nullptr, error));
  return error;
}
} // namespace

TEST(RenderDemoOptions, ParsesVignetteFlags) {
  const auto defaults = parse({});
  ASSERT_TRUE(defaults);
  EXPECT_FALSE(defaults->vignette);
  EXPECT_TRUE(defaults->vignetteAnchor.empty());
  EXPECT_EQ(defaults->expectAnchorWarnings, 0u);
  EXPECT_EQ(defaults->msaa, 1u);

  const auto vignette = parse({"--vignette"});
  ASSERT_TRUE(vignette);
  EXPECT_TRUE(vignette->vignette);
  EXPECT_TRUE(vignette->vignetteAnchor.empty());

  const auto anchored = parse({"--vignette-anchor", "phase.ao", "--expect-anchor-warnings", "2"});
  ASSERT_TRUE(anchored);
  EXPECT_TRUE(anchored->vignette);
  EXPECT_EQ(anchored->vignetteAnchor, "phase.ao");
  EXPECT_EQ(anchored->expectAnchorWarnings, 2u);
  EXPECT_FALSE(parse({"--vignette-anchor", ""}));
  EXPECT_FALSE(parse({"--vignette", "--expect-anchor-warnings", "x"}));
}

TEST(RenderDemoOptions, RejectsExpectWarningsWithoutVignette) {
  EXPECT_EQ(parse_error({"--expect-anchor-warnings", "1"}), "--expect-anchor-warnings needs --vignette");
  EXPECT_TRUE(parse({"--expect-anchor-warnings", "1", "--vignette"}));
}

TEST(RenderDemoOptions, RejectsGraphPassesWithTrace) {
  EXPECT_EQ(parse_error({"--vignette", "--trace", "x"}), "render graph passes cannot be traced");
  EXPECT_EQ(parse_error({"--trace", "x", "--vignette-anchor", "phase.ao"}), "render graph passes cannot be traced");
}

TEST(RenderDemoOptions, ParsesMsaa) {
  for (const char* samples : {"1", "4"}) {
    const auto options = parse({"--msaa", samples});
    ASSERT_TRUE(options) << samples;
    EXPECT_EQ(options->msaa, static_cast<uint32_t>(std::stoul(samples)));
  }
  for (const char* samples : {"0", "2", "8", "x"}) {
    EXPECT_FALSE(parse({"--msaa", samples})) << samples;
  }
}

TEST(RenderDemoGraph, VignetteParamsDeterministic) {
  const VignetteParams first = vignette_params(0);
  EXPECT_NEAR(first.strength, 0.45f, 1e-6f);
  EXPECT_NEAR(vignette_params(240).strength, first.strength, 1e-6f);
  EXPECT_NEAR(vignette_params(300).strength, vignette_params(60).strength, 1e-6f);
  for (u32 frame = 0; frame < 480; ++frame) {
    const VignetteParams params = vignette_params(frame);
    EXPECT_GE(params.strength, 0.30f - 1e-6f) << frame;
    EXPECT_LE(params.strength, 0.60f + 1e-6f) << frame;
    EXPECT_LT(params.inner, params.outer) << frame;
    EXPECT_EQ(params.pad, 0.f) << frame;
  }
}

TEST(RenderDemoGraph, UnknownAnchorMatcher) {
  constexpr std::string_view Line =
      "render graph: 'demo.vignette' has unknown anchor 'phase.nope' in BeforeHud; ignored";
  EXPECT_TRUE(is_unknown_anchor_warning(LOG_WARNING, "aurora::render::graph", Line));
  EXPECT_FALSE(is_unknown_anchor_warning(LOG_WARNING, "aurora::render", Line));
  EXPECT_FALSE(is_unknown_anchor_warning(LOG_INFO, "aurora::render::graph", Line));
  EXPECT_FALSE(is_unknown_anchor_warning(LOG_WARNING, "aurora::render::graph", "render graph: 'demo.vignette' culled"));
}

TEST(RenderDemoOptions, ParsesDepthCheck) {
  const auto defaults = parse({});
  ASSERT_TRUE(defaults);
  EXPECT_FALSE(defaults->depthCheck);
  const auto check = parse({"--depth-check"});
  ASSERT_TRUE(check);
  EXPECT_TRUE(check->depthCheck);
  EXPECT_FALSE(check->vignette);
  EXPECT_EQ(parse_error({"--depth-check", "--frames", "39"}), "--depth-check needs at least 40 frames");
  EXPECT_EQ(parse_error({"--frames", "1", "--depth-check"}), "--depth-check needs at least 40 frames");
  EXPECT_TRUE(parse({"--depth-check", "--frames", "40"}));
  EXPECT_TRUE(parse({"--depth-check", "--frames", "0"}));
  EXPECT_TRUE(parse({"--frames", "39"}));
  EXPECT_EQ(parse_error({"--depth-check", "--trace", "x"}), "render graph passes cannot be traced");
}

namespace {
std::array<f32, 4> probe(f32 sky, f32 mid, f32 close, f32 width = 1280.f) { return {sky, mid, close, width}; }
} // namespace

TEST(RenderDemoGraph, DepthProbeAcceptsExpected) {
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.027f, 0.034f), true), std::nullopt);
}

TEST(RenderDemoGraph, DepthProbeRejectsSky) {
  EXPECT_EQ(check_depth_probe(probe(0.5f, 0.027f, 0.034f), true), "sky probe 0.5 is not the clear depth 0");
}

TEST(RenderDemoGraph, DepthProbeRejectsGroundOutOfRange) {
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.f, 0.034f), true), "ground probe 1 = 0 is outside (0, 1)");
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.027f, 1.f), true), "ground probe 2 = 1 is outside (0, 1)");
}

TEST(RenderDemoGraph, DepthProbeRejectsOrder) {
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.027f, 0.02f), true), "near probe 0.02 is not closer than mid probe 0.027");
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.027f, 0.027f), true), "near probe 0.027 is not closer than mid probe 0.027");
}

TEST(RenderDemoGraph, DepthProbeRejectsShaderNotRun) {
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.027f, 0.034f, 0.f), true), "probe shader did not run");
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.f, 0.f, 0.f), false), "probe shader did not run");
}

TEST(RenderDemoGraph, DepthProbeForwardZ) {
  EXPECT_EQ(check_depth_probe(probe(1.f, 0.98f, 0.97f), false), std::nullopt);
  EXPECT_EQ(check_depth_probe(probe(0.f, 0.98f, 0.97f), false), "sky probe 0 is not the clear depth 1");
  EXPECT_EQ(check_depth_probe(probe(1.f, 1.f, 0.97f), false), "ground probe 1 = 1 is outside (0, 1)");
  EXPECT_EQ(check_depth_probe(probe(1.f, 0.97f, 0.98f), false), "near probe 0.98 is not closer than mid probe 0.97");
  // The reversed-Z expectations fail a forward-Z depth buffer
  EXPECT_NE(check_depth_probe(probe(0.f, 0.027f, 0.034f), false), std::nullopt);
}

TEST(RenderDemoGraph, DepthProbeRowsSampleSkyAndGround) {
  EXPECT_EQ(DepthProbeRows.size(), 3u);
  EXPECT_LT(DepthProbeRows[0], DepthProbeRows[1]);
  EXPECT_LT(DepthProbeRows[1], DepthProbeRows[2]);
  EXPECT_LT(DepthProbeRows[2], 1.f);
  EXPECT_LT(DepthCheckFrame, 40u);
}

TEST(RenderDemoOptions, ParsesHintFlags) {
  const auto defaults = parse({});
  ASSERT_TRUE(defaults);
  EXPECT_TRUE(defaults->hints);
  EXPECT_FALSE(defaults->checkHints);
  const auto off = parse({"--no-hints"});
  ASSERT_TRUE(off);
  EXPECT_FALSE(off->hints);
  EXPECT_FALSE(off->checkHints);
  const auto checked = parse({"--check-hints"});
  ASSERT_TRUE(checked);
  EXPECT_TRUE(checked->hints);
  EXPECT_TRUE(checked->checkHints);
  const auto traced = parse({"--check-hints", "--trace", "x"});
  ASSERT_TRUE(traced);
  EXPECT_TRUE(traced->checkHints);
  EXPECT_EQ(traced->tracePath, "x");
  EXPECT_TRUE(parse({"--no-hints", "--check-hints", "--vignette", "--depth-check"}));
}

namespace {
constexpr size_t HintIdCount = LastHintId + 1 - FirstHintId;

struct FakeCounters {
  std::array<ObservedCounters, HintIdCount> ids{};
  ObservedCounters& operator[](u16 id) { return ids[id - FirstHintId]; }
};

ObservedCounters fake_lookup(u16 id, void* userData) {
  return static_cast<const FakeCounters*>(userData)->ids[id - FirstHintId];
}

// One demo frame as draw_scene emits it
HintTally demo_tally(u32 frames, bool captureOnlyBox) {
  HintTally tally;
  const u32 objects = captureOnlyBox ? 13 : 12;
  tally.add(GX_AURORA_RENDER_CAMERA, 2 * frames);
  tally.add(GX_AURORA_RENDER_SCENE_EPOCH, frames);
  tally.add(GX_AURORA_SET_ARRAY_IMMUTABLE, 3 * frames);
  tally.add(GX_AURORA_RENDER_LIGHTS, frames);
  tally.add(GX_AURORA_RENDER_ENVIRONMENT, frames);
  tally.add(GX_AURORA_RENDER_INFLUENCERS, frames);
  tally.add(GX_AURORA_RENDER_WIND_SOURCES, frames);
  tally.add(GX_AURORA_RENDER_FOG_VOLUMES, frames);
  tally.add(GX_AURORA_RENDER_STAGE, 6 * frames);
  tally.add(GX_AURORA_RENDER_OBJECT_BEGIN, objects * frames);
  tally.add(GX_AURORA_RENDER_OBJECT_END, objects * frames);
  tally.add(GX_AURORA_RENDER_SCOPE_PUSH, 2 * frames);
  tally.add(GX_AURORA_RENDER_SCOPE_POP, 2 * frames);
  tally.add(GX_AURORA_RENDER_MATERIAL, frames);
  tally.add(GX_AURORA_RENDER_TEV_POLICY, frames);
  tally.add(GX_AURORA_RENDER_SKELETON, frames);
  tally.captureOnlyDraws = captureOnlyBox ? frames : 0;
  return tally;
}

// What M3 aurora reports: the module decodes CAMERA through WIND_SOURCES and skips the rest
FakeCounters observe(const HintTally& tally, bool decoding) {
  FakeCounters counters;
  for (u32 id = FirstHintId; id <= LastHintId; ++id) {
    const auto hint = static_cast<u16>(id);
    const bool decoded = decoding && hint >= GX_AURORA_RENDER_CAMERA && hint <= GX_AURORA_RENDER_WIND_SOURCES;
    (decoded ? counters[hint].decoded : counters[hint].skipped) = tally.at(hint);
  }
  return counters;
}

std::optional<std::string> check(const HintTally& tally, bool decoding, FakeCounters& counters, u64 suppressed) {
  return check_hints(tally, decoding, fake_lookup, &counters, suppressed);
}
} // namespace

TEST(RenderDemoHints, TallyCountsPerId) {
  HintTally tally;
  tally.add(GX_AURORA_RENDER_CAMERA);
  tally.add(GX_AURORA_RENDER_CAMERA);
  tally.add(GX_AURORA_RENDER_STAGE, 6);
  tally.add(GX_AURORA_SET_ARRAY_IMMUTABLE, 3);
  tally.add(LastHintId);
  // Outside the hint range, so not a hint
  tally.add(0x0041);
  tally.add(0x0200);
  EXPECT_EQ(tally.at(GX_AURORA_RENDER_CAMERA), 2u);
  EXPECT_EQ(tally.at(GX_AURORA_RENDER_STAGE), 6u);
  EXPECT_EQ(tally.at(GX_AURORA_SET_ARRAY_IMMUTABLE), 3u);
  EXPECT_EQ(tally.at(LastHintId), 1u);
  EXPECT_EQ(tally.at(GX_AURORA_RENDER_LIGHTS), 0u);
  EXPECT_EQ(tally.at(0x0041), 0u);
  EXPECT_EQ(tally.total(), 12u);
  EXPECT_EQ(demo_tally(1, true).total(), 50u);
  EXPECT_EQ(demo_tally(1, false).total(), 48u);
}

TEST(RenderDemoHints, CheckAcceptsDecoding) {
  const HintTally tally = demo_tally(120, true);
  FakeCounters counters = observe(tally, true);
  EXPECT_EQ(check(tally, true, counters, 120), std::nullopt);
}

TEST(RenderDemoHints, CheckAcceptsSkipping) {
  const HintTally tally = demo_tally(120, false);
  FakeCounters counters = observe(tally, false);
  EXPECT_EQ(check(tally, false, counters, 0), std::nullopt);
  // --no-hints emits nothing, so every counter stays zero
  FakeCounters none;
  EXPECT_EQ(check(HintTally{}, true, none, 0), std::nullopt);
  EXPECT_EQ(check(HintTally{}, false, none, 0), std::nullopt);
}

TEST(RenderDemoHints, CheckRejectsMissingDecode) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_RENDER_CAMERA] = {.skipped = 8};
  EXPECT_EQ(check(tally, true, counters, 4), "CAMERA (0x0100): decoded 0, emitted 8");
  // A build that decodes but was told it does not is caught the other way round
  FakeCounters decoded = observe(tally, true);
  EXPECT_EQ(check(demo_tally(4, false), false, decoded, 0), "CAMERA (0x0100): decoded with decoding off 8, emitted 8");
}

TEST(RenderDemoHints, CheckRejectsMalformed) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_RENDER_ENVIRONMENT] = {.decoded = 3, .malformed = 1};
  EXPECT_EQ(check(tally, true, counters, 4), "ENVIRONMENT (0x0107): malformed 1, emitted 4");
}

TEST(RenderDemoHints, CheckRejectsLate) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_RENDER_LIGHTS].late = 1;
  EXPECT_EQ(check(tally, true, counters, 4), "LIGHTS (0x0106): late 1, emitted 4");
}

TEST(RenderDemoHints, CheckRejectsStrayHint) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_RENDER_SKY].skipped = 1;
  EXPECT_EQ(check(tally, true, counters, 4), "SKY (0x010F): decoded + skipped 1, emitted 0");
  FakeCounters reserved = observe(tally, true);
  reserved[0x0150].skipped = 2;
  EXPECT_EQ(check(tally, true, reserved, 4), "reserved (0x0150): decoded + skipped 2, emitted 0");
  FakeCounters lost = observe(tally, true);
  lost[GX_AURORA_RENDER_FOG_VOLUMES].skipped = 3;
  EXPECT_EQ(check(tally, true, lost, 4), "FOG_VOLUMES (0x0110): decoded + skipped 3, emitted 4");
}

TEST(RenderDemoHints, CheckRejectsWrongSuppression) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  EXPECT_EQ(check(tally, true, counters, 3), "3 draws suppressed, expected 4");
  EXPECT_EQ(check(tally, true, counters, 5), "5 draws suppressed, expected 4");
  // With decoding off nothing may be suppressed, whatever the scene drew
  const HintTally skipped = demo_tally(4, false);
  FakeCounters skippedCounters = observe(skipped, false);
  EXPECT_EQ(check(skipped, false, skippedCounters, 1), "1 draws suppressed, expected 0");
}

TEST(RenderDemoHints, CheckIgnoresLaterMilestoneDecodes) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_SET_ARRAY_IMMUTABLE] = {.decoded = 12};
  counters[GX_AURORA_RENDER_FOG_VOLUMES] = {.decoded = 4};
  EXPECT_EQ(check(tally, true, counters, 4), std::nullopt);
  // Decoding off still means nothing decodes, later milestones included
  const HintTally skipped = demo_tally(4, false);
  FakeCounters skippedCounters = observe(skipped, false);
  skippedCounters[GX_AURORA_RENDER_FOG_VOLUMES] = {.decoded = 4};
  EXPECT_EQ(check(skipped, false, skippedCounters, 0), "FOG_VOLUMES (0x0110): decoded with decoding off 4, emitted 4");
}

TEST(RenderDemoHints, NamesFollowTheCatalogue) {
  EXPECT_EQ(hint_name(GX_AURORA_SET_ARRAY_IMMUTABLE), "SET_ARRAY_IMMUTABLE");
  EXPECT_EQ(hint_name(GX_AURORA_RENDER_CAMERA), "CAMERA");
  EXPECT_EQ(hint_name(GX_AURORA_RENDER_WIND_SOURCES), "WIND_SOURCES");
  EXPECT_EQ(hint_name(GX_AURORA_RENDER_FOG_VOLUMES), "FOG_VOLUMES");
  EXPECT_EQ(hint_name(GX_AURORA_GPU_DESTROY_SYSTEM), "GPU_DESTROY_SYSTEM");
  EXPECT_EQ(hint_name(0x0052), "reserved");
  EXPECT_EQ(hint_name(0x01FF), "reserved");
}

TEST(RenderDemoHints, DistanceRadius) {
  // 1 + 0.05 d + 0.005 d^2 = 255
  const DistanceFalloff medium = distance_falloff(10.f, 0.5f, GX_DA_MEDIUM);
  EXPECT_NEAR(medium.radius, 220.444f, 0.01f);
  EXPECT_EQ(medium.exponent, 1.5f);
  // 1 + d / 15 = 255
  const DistanceFalloff gentle = distance_falloff(15.f, 0.5f, GX_DA_GENTLE);
  EXPECT_NEAR(gentle.radius, 3810.f, 0.01f);
  EXPECT_EQ(gentle.exponent, 1.f);
  // 1 + 0.01 d^2 = 255
  const DistanceFalloff steep = distance_falloff(10.f, 0.5f, GX_DA_STEEP);
  EXPECT_NEAR(steep.radius, 159.374f, 0.01f);
  EXPECT_EQ(steep.exponent, 2.f);
  EXPECT_NEAR(distance_falloff(1.f, 0.25f, GX_DA_STEEP).radius, std::sqrt(254.f / 3.f), 1e-3f);

  struct Case {
    f32 distance;
    f32 brightness;
    GXDistAttnFn fn;
  };
  // OFF, and every argument GXInitLightDistAttn turns into OFF, sends radius 0 and exponent 0, never an infinity
  constexpr Case NoFalloff[] = {
      {10.f, 0.5f, GX_DA_OFF}, {10.f, 1.f, GX_DA_MEDIUM}, {-1.f, 0.5f, GX_DA_GENTLE}, {10.f, -0.5f, GX_DA_STEEP}};
  for (const Case& c : NoFalloff) {
    const DistanceFalloff falloff = distance_falloff(c.distance, c.brightness, c.fn);
    EXPECT_EQ(falloff.radius, 0.f) << c.distance << " " << c.brightness << " " << c.fn;
    EXPECT_EQ(falloff.exponent, 0.f) << c.distance << " " << c.brightness << " " << c.fn;
  }
  // A zero reference distance or brightness makes the GX coefficients infinite; the radius stays a finite 0
  constexpr Case Degenerate[] = {{0.f, 0.5f, GX_DA_GENTLE}, {0.f, 0.5f, GX_DA_MEDIUM}, {0.f, 0.5f, GX_DA_STEEP},
                                 {10.f, 0.f, GX_DA_GENTLE}, {10.f, 0.f, GX_DA_MEDIUM}, {10.f, 0.f, GX_DA_STEEP}};
  for (const Case& c : Degenerate) {
    const DistanceFalloff falloff = distance_falloff(c.distance, c.brightness, c.fn);
    EXPECT_EQ(falloff.radius, 0.f) << c.distance << " " << c.brightness << " " << c.fn;
    EXPECT_GT(falloff.exponent, 0.f) << c.distance << " " << c.brightness << " " << c.fn;
  }
}

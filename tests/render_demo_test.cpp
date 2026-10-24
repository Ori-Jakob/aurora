#include <gtest/gtest.h>

#include "bench_report.hpp"
#include "cull_reference.hpp"
#include "graph_checks.hpp"
#include "hint_checks.hpp"
#include "options.hpp"
#include "procedural.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <numeric>
#include <optional>
#include <set>
#include <span>
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

TEST(RenderDemoOptions, ParsesCustomDrawProbe) {
  const auto defaults = parse({});
  ASSERT_TRUE(defaults);
  EXPECT_FALSE(defaults->customDrawProbe);
  const auto probe = parse({"--custom-draw-probe"});
  ASSERT_TRUE(probe);
  EXPECT_TRUE(probe->customDrawProbe);
  EXPECT_EQ(parse_error({"--custom-draw-probe", "--trace", "x"}), "custom draws cannot be traced");
  EXPECT_NE(usage().find("--custom-draw-probe "), std::string_view::npos);
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

// What M4 aurora reports: core hints always decode, the module decodes CAMERA through WIND_SOURCES, the rest skip
FakeCounters observe(const HintTally& tally, bool decoding) {
  FakeCounters counters;
  for (u32 id = FirstHintId; id <= LastHintId; ++id) {
    const auto hint = static_cast<u16>(id);
    const bool decoded = hint <= GX_AURORA_HINT_CORE_LAST ||
                         (decoding && hint >= GX_AURORA_RENDER_CAMERA && hint <= GX_AURORA_RENDER_WIND_SOURCES);
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

TEST(RenderDemoHints, GpuCommandsAreExempt) {
  const HintTally tally = demo_tally(4, true);
  FakeCounters counters = observe(tally, true);
  counters[GX_AURORA_GPU_UPLOAD] = {.decoded = 4, .bytes = 4 * 209};
  counters[GX_AURORA_LOAD_ARRAY_GPU] = {.decoded = 12, .bytes = 12 * 22};
  EXPECT_EQ(check(tally, true, counters, 4), std::nullopt);
  FakeCounters skipping = observe(demo_tally(4, false), false);
  skipping[GX_AURORA_GPU_UPLOAD] = {.decoded = 4};
  EXPECT_EQ(check(demo_tally(4, false), false, skipping, 0), std::nullopt);
  counters[GX_AURORA_GPU_UPLOAD].malformed = 1;
  EXPECT_EQ(check(tally, true, counters, 4), "GPU_UPLOAD (0x0120): malformed 1, emitted 0");
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
  // Decoding off still means no module hint decodes, later milestones included
  const HintTally skipped = demo_tally(4, false);
  FakeCounters skippedCounters = observe(skipped, false);
  skippedCounters[GX_AURORA_RENDER_FOG_VOLUMES] = {.decoded = 4};
  EXPECT_EQ(check(skipped, false, skippedCounters, 0), "FOG_VOLUMES (0x0110): decoded with decoding off 4, emitted 4");
}

TEST(RenderDemoHints, CoreHintsDecodeWithDecodingOff) {
  const HintTally tally = demo_tally(4, false);
  FakeCounters counters = observe(tally, false);
  EXPECT_EQ(counters[GX_AURORA_SET_ARRAY_IMMUTABLE].decoded, 12u);
  EXPECT_EQ(check(tally, false, counters, 0), std::nullopt);
  counters[GX_AURORA_SET_ARRAY_IMMUTABLE] = {.skipped = 12};
  EXPECT_EQ(check(tally, false, counters, 0), "SET_ARRAY_IMMUTABLE (0x0050): decoded 0, emitted 12");
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

namespace {
struct TestCamera {
  f32 proj[4][4]{};
  f32 view[3][4]{};
};

// C_MTXPerspective(55, 4 / 3, 0.5, 400) with the eye at (ex, ey, ez) looking down -z
TestCamera test_camera(f32 ex, f32 ey, f32 ez) {
  constexpr f32 Near = 0.5f;
  constexpr f32 Far = 400.f;
  const f32 cot = 1.f / std::tan(55.f * 0.5f * 3.14159265358979f / 180.f);
  TestCamera camera;
  camera.proj[0][0] = cot / (4.f / 3.f);
  camera.proj[1][1] = cot;
  camera.proj[2][2] = -Near / (Far - Near);
  camera.proj[2][3] = -(Far * Near) / (Far - Near);
  camera.proj[3][2] = -1.f;
  camera.view[0][0] = camera.view[1][1] = camera.view[2][2] = 1.f;
  camera.view[0][3] = -ex;
  camera.view[1][3] = -ey;
  camera.view[2][3] = -ez;
  return camera;
}

std::array<Plane, 6> test_planes(bool reversedZ = true) {
  const TestCamera camera = test_camera(0.f, 5.f, -480.f);
  return frustum_planes(view_projection(camera.proj, camera.view, reversedZ));
}

f32 plane_distance(const Plane& p, f32 x, f32 y, f32 z) { return p[0] * x + p[1] * y + p[2] * z + p[3]; }

bool kept_by_kernel(const CullInstance& s, std::span<const Plane, 6> planes) {
  for (const Plane& p : planes) {
    const f32 distance = p[0] * s.x + p[1] * s.y + p[2] * s.z;
    if (!(distance + p[3] >= -s.radius)) {
      return false;
    }
  }
  return true;
}

std::vector<CullInstance> instances(u32 count) {
  std::vector<CullInstance> out(count);
  fill_cull_instances(out);
  return out;
}

std::vector<u32> ids_of(std::span<const u32> words, const CullLayout& layout) {
  const auto first = words.begin() + layout.idsWord;
  return {first, first + words[HeaderTotal]};
}

std::vector<u32> with_ids(std::vector<u32> words, const CullLayout& layout, const std::vector<u32>& ids) {
  words[1] = static_cast<u32>(ids.size());
  words[HeaderTotal] = static_cast<u32>(ids.size());
  std::fill(words.begin() + layout.idsWord, words.end(), 0u);
  std::copy(ids.begin(), ids.end(), words.begin() + layout.idsWord);
  return words;
}

struct CheckFixture {
  std::vector<CullInstance> instances;
  std::array<Plane, 6> planes;
  CullLayout layout;
  std::vector<u32> words;
};

CheckFixture check_fixture(u32 count = 4096) {
  CheckFixture f{.instances = instances(count), .planes = test_planes(), .layout = cull_layout(count)};
  f.words = simulate_cull(f.instances, f.planes, CullCheckFrame);
  return f;
}

std::string check_failure(const CheckFixture& f, const std::vector<u32>& words) {
  return check_cull_readback(words, f.layout, f.instances, f.planes, CullCheckFrame).failure.value_or("");
}
} // namespace

TEST(RenderDemoCull, LayoutAlignsRegions) {
  for (const u32 n : {1u, 127u, 128u, 129u, 1u << 20, 1u << 22}) {
    SCOPED_TRACE(n);
    const CullLayout layout = cull_layout(n);
    EXPECT_EQ(layout.instances, n);
    EXPECT_EQ(layout.groups, (n + 127) / 128);
    EXPECT_EQ(layout.countsWord, HeaderWords);
    EXPECT_EQ(layout.maskWord % 64, 0u);
    EXPECT_EQ(layout.idsWord % 64, 0u);
    EXPECT_GE(layout.maskWord, layout.countsWord + layout.groups);
    EXPECT_GE(layout.idsWord, layout.maskWord + 4 * layout.groups);
    EXPECT_EQ(layout.totalWords, layout.idsWord + n);
    EXPECT_EQ(layout.bytes(), static_cast<u64>(layout.totalWords) * 4);
    EXPECT_EQ(layout.ids_offset() % 256, 0u);
  }
  EXPECT_LT(cull_layout(1u << 20).bytes(), 4500000u);
}

TEST(RenderDemoCull, InstancesAreExactAndUnique) {
  const auto all = instances(1u << 20);
  u64 digest = 0xcbf29ce484222325ull;
  for (u32 id = 0; id < all.size(); ++id) {
    const CullInstance& s = all[id];
    ASSERT_EQ(s.x, std::floor(s.x));
    ASSERT_EQ(s.z, std::floor(s.z));
    ASSERT_EQ(static_cast<s32>(s.x) + 512 + (static_cast<s32>(s.z) + 512) * 1024, static_cast<s32>(id));
    ASSERT_EQ(s.y * 4.f, std::floor(s.y * 4.f));
    ASSERT_LE(s.y, 1.75f);
    ASSERT_EQ(s.radius * 16.f, std::floor(s.radius * 16.f));
    ASSERT_GE(s.radius, 0.125f);
    ASSERT_LE(s.radius, 0.3125f);
    std::array<unsigned char, sizeof(CullInstance)> bytes;
    std::memcpy(bytes.data(), &s, sizeof(s));
    for (const unsigned char byte : bytes) {
      digest = (digest ^ byte) * 0x100000001b3ull;
    }
  }
  EXPECT_EQ(digest, 0x972b59bcb1940c5cull);
}

TEST(RenderDemoCull, ViewProjectionMatchesAuroraConversion) {
  const TestCamera camera = test_camera(0.f, 0.f, 0.f);
  const auto reversed = view_projection(camera.proj, camera.view, true);
  const auto forward = view_projection(camera.proj, camera.view, false);
  for (u32 c = 0; c < 4; ++c) {
    EXPECT_FLOAT_EQ(reversed[c * 4 + 2], -camera.proj[2][c]);
    EXPECT_FLOAT_EQ(forward[c * 4 + 2], camera.proj[2][c] + camera.proj[3][c]);
    for (const u32 r : {0u, 1u, 3u}) {
      EXPECT_FLOAT_EQ(reversed[c * 4 + r], camera.proj[r][c]);
      EXPECT_FLOAT_EQ(forward[c * 4 + r], camera.proj[r][c]);
    }
  }
  const auto depth = [](const std::array<f32, 16>& m, f32 z) { return (m[8 + 2] * z + m[12 + 2]) / (m[8 + 3] * z); };
  EXPECT_NEAR(depth(reversed, -0.5f), 1.f, 1e-5f);
  EXPECT_NEAR(depth(reversed, -400.f), 0.f, 1e-5f);
  EXPECT_NEAR(depth(forward, -0.5f), 0.f, 1e-5f);
  EXPECT_NEAR(depth(forward, -400.f), 1.f, 1e-5f);
  const TestCamera moved = test_camera(3.f, -2.f, 7.f);
  const auto translated = view_projection(moved.proj, moved.view, true);
  EXPECT_FLOAT_EQ(translated[12 + 3], 7.f);
}

TEST(RenderDemoCull, PlanesContainTargetAndRejectOutside) {
  for (const bool reversedZ : {true, false}) {
    SCOPED_TRACE(reversedZ);
    const TestCamera camera = test_camera(0.f, 0.f, 0.f);
    const auto planes = frustum_planes(view_projection(camera.proj, camera.view, reversedZ));
    for (const Plane& p : planes) {
      EXPECT_NEAR(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), 1.f, 1e-5f);
      EXPECT_GT(plane_distance(p, 0.f, 0.f, -10.f), 0.f);
    }
    const auto rejected = [&](f32 x, f32 y, f32 z) {
      return std::ranges::any_of(planes, [&](const Plane& p) { return plane_distance(p, x, y, z) < 0.f; });
    };
    EXPECT_TRUE(rejected(0.f, 0.f, 10.f));
    EXPECT_TRUE(rejected(0.f, 0.f, -500.f));
    EXPECT_TRUE(rejected(0.f, 0.f, -0.25f));
    EXPECT_TRUE(rejected(-1000.f, 0.f, -10.f));
    EXPECT_TRUE(rejected(0.f, 1000.f, -10.f));
  }
}

TEST(RenderDemoCull, SimulationMatchesCopyIf) {
  constexpr u32 Count = 300000;
  const auto all = instances(Count);
  const auto planes = test_planes();
  const CullLayout layout = cull_layout(Count);
  const std::vector<u32> words = simulate_cull(all, planes, 77);
  std::vector<u32> ids(Count);
  std::iota(ids.begin(), ids.end(), 0u);
  std::vector<u32> expected;
  std::ranges::copy_if(ids, std::back_inserter(expected), [&](u32 id) { return kept_by_kernel(all[id], planes); });
  ASSERT_EQ(words.size(), layout.totalWords);
  EXPECT_GT(expected.size(), 500u);
  EXPECT_LT(expected.size(), Count / 2);
  EXPECT_EQ(words[0], CullIndexCount);
  EXPECT_EQ(words[1], expected.size());
  EXPECT_EQ(words[2] | words[3] | words[4], 0u);
  EXPECT_EQ(words[HeaderFrameTag], 77u);
  EXPECT_EQ(words[HeaderTotal], expected.size());
  EXPECT_EQ(words[HeaderGroups], layout.groups);
  EXPECT_EQ(ids_of(words, layout), expected);
  for (u32 word = HeaderGroups + 1; word < HeaderWords; ++word) {
    EXPECT_EQ(words[word], 0u);
  }
}

TEST(RenderDemoCull, CheckAcceptsSimulation) {
  const CheckFixture f = check_fixture(1u << 16);
  const CullCheckResult result = check_cull_readback(f.words, f.layout, f.instances, f.planes, CullCheckFrame);
  EXPECT_EQ(result.failure, std::nullopt);
  EXPECT_GT(result.visible, 100u);
  EXPECT_LE(result.ambiguous, (1u << 16) / 100);
}

TEST(RenderDemoCull, CheckToleratesAmbiguous) {
  constexpr u32 Count = 300000;
  const auto all = instances(Count);
  // x = -100 with radius 0.125 touches the first plane exactly
  const std::array<Plane, 6> planes{{{1.f, 0.f, 0.f, 99.875f},
                                     {-1.f, 0.f, 0.f, 1000.f},
                                     {0.f, 1.f, 0.f, 1000.f},
                                     {0.f, -1.f, 0.f, 1000.f},
                                     {0.f, 0.f, 1.f, 1000.f},
                                     {0.f, 0.f, -1.f, 1000.f}}};
  const CullLayout layout = cull_layout(Count);
  const std::vector<u32> words = simulate_cull(all, planes, CullCheckFrame);
  std::vector<u32> unambiguous;
  for (const u32 id : ids_of(words, layout)) {
    if (classify(all[id], planes) != CullClass::Ambiguous) {
      unambiguous.push_back(id);
    }
  }
  ASSERT_LT(unambiguous.size(), words[HeaderTotal]);
  for (const auto& candidate : {words, with_ids(words, layout, unambiguous)}) {
    const CullCheckResult result = check_cull_readback(candidate, layout, all, planes, CullCheckFrame);
    EXPECT_EQ(result.failure, std::nullopt);
    EXPECT_GT(result.ambiguous, 0u);
  }
}

TEST(RenderDemoCull, CheckRejectsUnsorted) {
  const CheckFixture f = check_fixture();
  std::vector<u32> ids = ids_of(f.words, f.layout);
  ASSERT_GE(ids.size(), 3u);
  std::swap(ids[1], ids[2]);
  EXPECT_TRUE(check_failure(f, with_ids(f.words, f.layout, ids)).starts_with("ids are not strictly increasing"));
}

TEST(RenderDemoCull, CheckRejectsMissingVisible) {
  const CheckFixture f = check_fixture();
  std::vector<u32> ids = ids_of(f.words, f.layout);
  const auto visible =
      std::ranges::find_if(ids, [&](u32 id) { return classify(f.instances[id], f.planes) == CullClass::Visible; });
  ASSERT_NE(visible, ids.end());
  const u32 missing = *visible;
  ids.erase(visible);
  EXPECT_EQ(check_failure(f, with_ids(f.words, f.layout, ids)),
            "visible instance " + std::to_string(missing) + " is missing");
}

TEST(RenderDemoCull, CheckRejectsExtraCulled) {
  const CheckFixture f = check_fixture();
  std::vector<u32> ids = ids_of(f.words, f.layout);
  u32 culled = 0;
  while (classify(f.instances[culled], f.planes) != CullClass::Culled) {
    ++culled;
  }
  ids.insert(std::ranges::lower_bound(ids, culled), culled);
  EXPECT_EQ(check_failure(f, with_ids(f.words, f.layout, ids)),
            "culled instance " + std::to_string(culled) + " was kept");
}

TEST(RenderDemoCull, CheckRejectsCountMismatch) {
  const CheckFixture f = check_fixture();
  std::vector<u32> words = f.words;
  words[1] -= 1;
  EXPECT_TRUE(check_failure(f, words).starts_with("instance count "));
}

TEST(RenderDemoCull, CheckRejectsTotalMismatch) {
  const CheckFixture f = check_fixture();
  std::vector<u32> words = f.words;
  words[1] = words[HeaderTotal] = f.layout.instances + 1;
  EXPECT_EQ(check_failure(f, words), "total 4097 exceeds 4096 instances");
}

TEST(RenderDemoCull, CheckRejectsWrongFrameTag) {
  const CheckFixture f = check_fixture();
  std::vector<u32> words = f.words;
  words[HeaderFrameTag] = 7;
  EXPECT_EQ(check_failure(f, words), "frame tag 7, expected 30");
}

TEST(RenderDemoCull, CheckRejectsFirstInstanceNonZero) {
  const CheckFixture f = check_fixture();
  std::vector<u32> words = f.words;
  words[4] = 1;
  EXPECT_EQ(check_failure(f, words), "first instance 1, expected 0");
}

TEST(RenderDemoCull, CheckRejectsIndexCountWrong) {
  const CheckFixture f = check_fixture();
  std::vector<u32> words = f.words;
  words[0] = 36;
  EXPECT_EQ(check_failure(f, words), "index count 36, expected 24");
}

TEST(RenderDemoCull, CheckRejectsIdOutOfRange) {
  const CheckFixture f = check_fixture();
  std::vector<u32> ids = ids_of(f.words, f.layout);
  ASSERT_FALSE(ids.empty());
  ids.back() = f.layout.instances + 5;
  EXPECT_TRUE(check_failure(f, with_ids(f.words, f.layout, ids)).starts_with("id 4101 at slot "));
}

TEST(RenderDemoCull, CheckRejectsIllConditioned) {
  constexpr u32 Count = 4096;
  const auto all = instances(Count);
  // y = 0.25 with radius 0.125 touches the plane exactly, about one instance in 32
  const std::array<Plane, 6> planes{{{0.f, 1.f, 0.f, -0.375f},
                                     {1.f, 0.f, 0.f, 1000.f},
                                     {-1.f, 0.f, 0.f, 1000.f},
                                     {0.f, -1.f, 0.f, 1000.f},
                                     {0.f, 0.f, 1.f, 1000.f},
                                     {0.f, 0.f, -1.f, 1000.f}}};
  const std::vector<u32> words = simulate_cull(all, planes, CullCheckFrame);
  const auto failure = check_cull_readback(words, cull_layout(Count), all, planes, CullCheckFrame).failure;
  ASSERT_TRUE(failure.has_value());
  EXPECT_TRUE(failure->starts_with("ill-conditioned reference"));
}

TEST(RenderDemoCull, CheckRejectsStaleFrame) {
  CheckFixture f = check_fixture();
  f.words = simulate_cull(f.instances, f.planes, CullCheckFrame - 1);
  EXPECT_EQ(check_failure(f, f.words), "frame tag 29, expected 30");
}

TEST(RenderDemoCull, ShadersBakeTheLayout) {
  const CullLayout layout = cull_layout(1000);
  const std::string compute = cull_compute_wgsl(layout);
  EXPECT_NE(compute.find("const N: u32 = 1000u;"), std::string::npos);
  EXPECT_NE(compute.find("const IDS: u32 = " + std::to_string(layout.idsWord) + "u;"), std::string::npos);
  for (const char* entry : {"fn cs_count", "fn cs_scan", "fn cs_scatter"}) {
    EXPECT_NE(compute.find(entry), std::string::npos) << entry;
  }
  EXPECT_EQ(compute.find("atomic"), std::string::npos);
  const std::string draw = cull_draw_wgsl(1000);
  EXPECT_NE(draw.find("const N: u32 = 1000u;"), std::string::npos);
  EXPECT_EQ(draw.find("instance_index"), std::string::npos);
  EXPECT_EQ(draw.find("vertex_index"), std::string::npos);
}

namespace {
SampleStats stats_of(f64 p50, f64 p90, f64 mean, f64 min, f64 max, u32 count) {
  return {.p50 = p50, .p90 = p90, .mean = mean, .min = min, .max = max, .count = count};
}

BenchRun linear_run(f64 alpha, f64 gpuAlpha) {
  BenchRun run;
  for (const BenchPair& pair : bench_pairs()) {
    PairResult result{.pair = pair};
    const f64 n = pair.draws;
    const f64 k = pair.passes;
    result.deltaCpuUs.p50 = alpha + 0.02 * n + 1.5 * k;
    result.deltaGpuUs = SampleStats{.p50 = gpuAlpha + 0.004 * n + 0.75 * k};
    run.pairs.push_back(result);
  }
  return run;
}

ArmSamples ramp(f64 offset, f64 slope, u32 blocks, bool treatment) {
  ArmSamples samples;
  for (u32 k = 0; k < blocks; ++k) {
    // Block k runs A, B, B, A at times 4k to 4k + 3
    const f64 first = treatment ? 4.0 * k + 1 : 4.0 * k;
    const f64 second = treatment ? 4.0 * k + 2 : 4.0 * k + 3;
    for (const f64 t : {first, second}) {
      samples.totalUs.push_back(offset + slope * t);
      samples.encodeUs.push_back(1.0);
      samples.finishUs.push_back(1.0);
      samples.submitUs.push_back(1.0);
    }
  }
  return samples;
}
} // namespace

TEST(RenderDemoBench, PairMatrixIsStable) {
  const auto pairs = bench_pairs();
  ASSERT_EQ(pairs.size(), 37u);
  std::set<std::string> names;
  for (const BenchPair& pair : pairs) {
    EXPECT_TRUE(names.insert(pair_name(pair)).second) << pair_name(pair);
    EXPECT_LE(pair.draws, BenchMaxDraws);
    EXPECT_LE(pair.passes, BenchMaxPasses);
  }
  EXPECT_EQ(pair_name(pairs[0]), "direct.vi.gx.gpu.u16~indirect.vi.gx.gpu.u16.n50.k1");
  EXPECT_EQ(pair_name(pairs[1]), "direct.vi.gx.gpu.u16~indirect.vi.gx.gpu.u16.n500.k1");
  EXPECT_EQ(pair_name(pairs[30]), "indirect.vi.gx.gpu.u16~indirect.vi.gx.cpu.u16.n500.k1");
  EXPECT_EQ(pair_name(pairs[36]), "direct.vi.gx.gpu.u16~indirect.vi.subst.gpu.u32.n2000.k1");
  // The 500 and 2000 draw partners of every slope sit next to each other
  for (size_t i = 18; i + 1 < pairs.size(); ++i) {
    if (pairs[i].draws == 500 && pairs[i].passes == 1 && i != 30) {
      EXPECT_EQ(pairs[i + 1].draws, 2000u) << i;
      EXPECT_EQ(pairs[i + 1].passes, 1u) << i;
      EXPECT_TRUE(pairs[i + 1].a == pairs[i].a && pairs[i + 1].b == pairs[i].b) << i;
    }
  }
}

TEST(RenderDemoBench, AbbaPairingCancelsDrift) {
  const ArmSamples a = ramp(100.0, 0.5, 40, false);
  const ArmSamples b = ramp(107.0, 0.5, 40, true);
  const PairResult result = reduce_pair(bench_pairs()[0], a, b);
  EXPECT_NEAR(result.deltaCpuUs.p50, 7.0, 1e-9);
  EXPECT_NEAR(result.deltaCpuUs.min, 7.0, 1e-9);
  EXPECT_NEAR(result.deltaCpuUs.max, 7.0, 1e-9);
  EXPECT_EQ(result.deltaCpuUs.count, 40u);
  EXPECT_FALSE(result.deltaGpuUs.has_value());
}

TEST(RenderDemoBench, DriftAndInstability) {
  const PairResult flat = reduce_pair(bench_pairs()[0], ramp(100.0, 0.0, 40, false), ramp(105.0, 0.0, 40, true));
  EXPECT_FALSE(flat.unstable);
  EXPECT_EQ(flat.drift, 0.0);
  const PairResult drifting = reduce_pair(bench_pairs()[0], ramp(100.0, 0.5, 40, false), ramp(100.0, 0.5, 40, true));
  EXPECT_GT(drifting.drift, 0.1);
  EXPECT_TRUE(drifting.unstable);
  ArmSamples noisy = ramp(100.0, 0.0, 40, false);
  for (size_t i = 0; i < noisy.totalUs.size(); i += 4) {
    noisy.totalUs[i] = 200.0;
  }
  EXPECT_TRUE(reduce_pair(bench_pairs()[0], noisy, ramp(100.0, 0.0, 40, true)).unstable);
}

TEST(RenderDemoBench, StatsMatchSpikeKit) {
  const SampleStats s = sample_stats({10.0, 2.0, 4.0, 1.0, 3.0});
  EXPECT_DOUBLE_EQ(s.p50, 3.0);
  EXPECT_DOUBLE_EQ(s.p90, 7.6);
  EXPECT_DOUBLE_EQ(s.mean, 4.0);
  EXPECT_DOUBLE_EQ(s.min, 1.0);
  EXPECT_DOUBLE_EQ(s.max, 10.0);
  EXPECT_EQ(s.count, 5u);
  EXPECT_EQ(sample_stats({}).count, 0u);
  EXPECT_TRUE(unstable(stats_of(10.0, 12.6, 0, 0, 0, 3)));
  EXPECT_FALSE(unstable(stats_of(10.0, 12.5, 0, 0, 0, 3)));
}

TEST(RenderDemoBench, RebindSpread) {
  const auto count = [](u32 draws, u32 passes, u32 permille) {
    u32 binds = 0;
    for (u32 pass = 0; pass < passes; ++pass) {
      const u32 first = pass * draws / passes;
      const u32 last = (pass + 1) * draws / passes;
      for (u32 draw = first; draw < last; ++draw) {
        binds += rebinds(draw - first, draw, permille) ? 1 : 0;
      }
    }
    return binds;
  };
  EXPECT_EQ(count(2000, 1, 500), 1001u);
  EXPECT_EQ(count(2000, 1, 1000), 2000u);
  EXPECT_EQ(count(500, 8, 1000), 500u);
  EXPECT_EQ(count(2000, 8, 0), 8u);
  EXPECT_EQ(count(1000, 1, 361), 362u);
}

TEST(RenderDemoBench, DerivedFit) {
  const BenchRun run = linear_run(3.0, 1.0);
  for (const BenchPipeline pipeline : {BenchPipeline::VertexIndex, BenchPipeline::Plain}) {
    const BenchDerived derived = bench_derived(run, pipeline);
    ASSERT_TRUE(derived.cpuFit && derived.gpuFit);
    EXPECT_NEAR(derived.cpuFit->alpha, 3.0, 1e-9);
    EXPECT_NEAR(derived.cpuFit->beta, 0.02, 1e-12);
    EXPECT_NEAR(derived.cpuFit->gamma, 1.5, 1e-9);
    EXPECT_NEAR(derived.cpuFit->rmsResidual, 0.0, 1e-9);
    EXPECT_NEAR(derived.gpuFit->beta, 0.004, 1e-12);
    EXPECT_NEAR(derived.gpuFit->gamma, 0.75, 1e-9);
    EXPECT_NEAR(*derived.gate500CpuMs, 0.0145, 1e-12);
  }
  const BenchDerived vi = bench_derived(run, BenchPipeline::VertexIndex);
  for (const auto& beta : {vi.substBetaCpuUs, vi.u32BetaCpuUs, vi.bundleBetaCpuUs, vi.zeroBetaCpuUs}) {
    ASSERT_TRUE(beta.has_value());
    EXPECT_NEAR(*beta, 0.02, 1e-12);
  }
  EXPECT_NEAR(*vi.zeroBetaGpuUs, 0.004, 1e-12);
  EXPECT_NEAR(*vi.argsBarrierCpuUs, -14.5, 1e-9);
  EXPECT_FALSE(bench_derived(run, BenchPipeline::Plain).substBetaCpuUs.has_value());
  BenchRun clamped = run;
  clamped.pairs[4].clamp = "error: x";
  EXPECT_FALSE(bench_derived(clamped, BenchPipeline::VertexIndex).cpuFit.has_value());
  EXPECT_TRUE(bench_derived(clamped, BenchPipeline::Plain).cpuFit.has_value());
}

TEST(RenderDemoBench, JsonGolden) {
  const auto pairs = bench_pairs();
  BenchRun run{.device = "aurora",
               .backend = "null",
               .profile = "full",
               .adapter = "Null \"A\"",
               .adapterKind = "cpu",
               .build = "release",
               .validation = "skipped",
               .vendorId = 1,
               .deviceId = 2,
               .msaa = 4,
               .width = 64,
               .height = 32,
               .warmupBlocks = 1,
               .blocks = 2,
               .uniformPermille = 959,
               .texturePermille = 361,
               .erroredPairs = 1,
               .maxImmediateSize = 64,
               .vertexStorageBuffers = 8,
               .mdi = true,
               .timestamps = true};
  PairResult measured{.pair = pairs[0], .drift = 0.0125, .unstable = true};
  measured.a.totalUs = stats_of(1.5, 2.25, 1.75, 1.0, 3.0, 4);
  measured.a.gpuTotalUs = SampleStats{};
  measured.a.gpuInvalid = 2;
  measured.deltaCpuUs = stats_of(0.5, 0.75, 0.5, 0.25, 0.75, 2);
  run.pairs.push_back(measured);
  run.pairs.push_back({.pair = pairs[1], .clamp = "error: x"});
  const std::string zero = R"({"p50":0.000,"p90":0.000,"mean":0.000,"min":0.000,"max":0.000,"count":0})";
  const std::string arm = R"({"mode":"direct","pipeline":"vi","state":"gx","args":"gpu","index":"u16"})";
  const std::string treated = R"({"mode":"indirect","pipeline":"vi","state":"gx","args":"gpu","index":"u16"})";
  const std::string nullDerived = R"({"cpuFit":null,"gpuFit":null,"gate500":{"cpuMs":null,"gpuMs":null}})";
  const std::string nullBetas = R"({"cpuBetaUs":null,"gpuBetaUs":null})";
  const std::string expected =
      std::string{R"({"schema":"aurora.render_demo.indirect_bench/1","device":"aurora","backend":"null",)"} +
      R"("profile":"full","adapter":"Null \"A\"","adapterKind":"cpu","vendorId":1,"deviceId":2,"build":"release",)" +
      R"("validation":"skipped","msaa":4,"target":{"width":64,"height":32},"caps":{"multiDrawIndirect":true,)" +
      R"("indirectFirstInstance":false,"timestampQuery":true,"maxImmediateSize":64,"vertexStorageBuffers":8},)" +
      R"("rebind":{"uniformPermille":959,"texturePermille":361},"wait":"spin","warmupBlocks":1,"blocks":2,)" +
      R"("erroredPairs":1,"pairs":[{"name":"direct.vi.gx.gpu.u16~indirect.vi.gx.gpu.u16.n50.k1","draws":50,)" +
      R"("passes":1,"a":)" + arm + R"(,"b":)" + treated + R"(,"clamp":"","unstable":true,"drift":0.0125,)" +
      R"("arms":{"a":{"cpuUs":{"encode":)" + zero + R"(,"finish":)" + zero + R"(,"submit":)" + zero +
      R"(,"total":{"p50":1.500,"p90":2.250,"mean":1.750,"min":1.000,"max":3.000,"count":4}},)" +
      R"("gpuUs":{"total":)" + zero + R"(,"inPass":null,"args":null},"gpuInvalid":2},"b":{"cpuUs":{"encode":)" + zero +
      R"(,"finish":)" + zero + R"(,"submit":)" + zero + R"(,"total":)" + zero +
      R"(},"gpuUs":null,"gpuInvalid":0}},"deltaUs":{"cpu":)" +
      R"({"p50":0.500,"p90":0.750,"mean":0.500,"min":0.250,"max":0.750,"count":2},"gpu":null}},)" +
      R"({"name":"direct.vi.gx.gpu.u16~indirect.vi.gx.gpu.u16.n500.k1","draws":500,"passes":1,"a":)" + arm +
      R"(,"b":)" + treated + R"(,"clamp":"error: x","unstable":false,"drift":0.0000,"arms":null,"deltaUs":null}],)" +
      R"("derived":{"vi":)" + nullDerived + R"(,"plain":)" + nullDerived + R"(,"subst":)" + nullBetas + R"(,"u32":)" +
      nullBetas + R"(,"bundle":)" + nullBetas + R"(,"zero":)" + nullBetas +
      R"(,"argsBarrier":{"cpuUs":null,"gpuUs":null}}})" + "\n";
  EXPECT_EQ(bench_json(run), expected);
}

TEST(RenderDemoBench, SummaryFormat) {
  BenchRun run{.device = "aurora",
               .backend = "null",
               .profile = "full",
               .build = "release",
               .validation = "skipped",
               .width = 64,
               .height = 32,
               .erroredPairs = 1};
  EXPECT_EQ(bench_summary(run, "out.json"),
            "render_demo: bench: 0 pairs on null/full (device aurora, validation skipped, build release, msaa 1, "
            "64x32), mdi no, timestamps no, errored 1; 500 vi gx draws: indirect-direct cpu n/a, gpu n/a (paired "
            "p50); wrote out.json");
  BenchRun measured = linear_run(3.5, 1.25);
  measured.device = "aurora";
  measured.backend = "d3d12";
  measured.profile = "compat";
  measured.build = "release";
  measured.validation = "on";
  measured.msaa = 4;
  measured.width = 2560;
  measured.height = 1440;
  measured.mdi = true;
  measured.timestamps = true;
  EXPECT_EQ(bench_summary(measured, "b.json"),
            "render_demo: bench: 37 pairs on d3d12/compat (device aurora, validation on, build release, msaa 4, "
            "2560x1440), mdi yes, timestamps yes, errored 0; 500 vi gx draws: indirect-direct cpu +0.015 ms, gpu "
            "+0.004 ms (paired p50); wrote b.json");
}

TEST(RenderDemoOptions, ParsesCullFlags) {
  const auto options = parse({"--gpu-cull", "1048576", "--cull-check", "--cull-timings", "--frames", "600"});
  ASSERT_TRUE(options);
  EXPECT_EQ(options->gpuCull, 1048576u);
  EXPECT_TRUE(options->cullCheck);
  EXPECT_TRUE(options->cullTimings);
  EXPECT_TRUE(options->benchPath.empty());
  EXPECT_TRUE(parse({"--gpu-cull", "4194304", "--cull-check"}));
  EXPECT_TRUE(parse({"--gpu-cull", "1", "--cull-check", "--frames", "40"}));
}

TEST(RenderDemoOptions, ParsesBenchFlags) {
  const auto options = parse({"--bench-indirect", "out/bench.json", "--bench-blocks", "3", "--bench-rebind", "959,361",
                              "--backend", "null", "--profile", "compat", "--size", "2560x1440", "--data-dir", "d",
                              "--msaa", "4", "--fail-on-error"});
  ASSERT_TRUE(options);
  EXPECT_EQ(options->benchPath, "out/bench.json");
  EXPECT_EQ(options->benchBlocks, 3u);
  EXPECT_EQ(options->benchUniformPermille, 959u);
  EXPECT_EQ(options->benchTexturePermille, 361u);
  EXPECT_EQ(options->width, 2560u);
  EXPECT_EQ(options->msaa, 4u);
  const auto defaults = parse({"--bench-indirect", "b.json"});
  ASSERT_TRUE(defaults);
  EXPECT_EQ(defaults->benchBlocks, 150u);
  EXPECT_EQ(defaults->benchUniformPermille, 1000u);
  EXPECT_EQ(defaults->benchTexturePermille, 500u);
}

TEST(RenderDemoOptions, RejectsCullRangesAndDependencies) {
  EXPECT_FALSE(parse({"--gpu-cull", "0"}));
  EXPECT_FALSE(parse({"--gpu-cull", "4194305"}));
  EXPECT_FALSE(parse({"--gpu-cull", "many"}));
  EXPECT_EQ(parse_error({"--cull-check"}), "--cull-check needs --gpu-cull");
  EXPECT_EQ(parse_error({"--cull-timings"}), "--cull-timings needs --gpu-cull");
  EXPECT_EQ(parse_error({"--gpu-cull", "128", "--cull-check", "--frames", "39"}),
            "--cull-check needs at least 40 frames");
  EXPECT_EQ(parse_error({"--gpu-cull", "128", "--trace", "t.gxtrace"}), "render graph passes cannot be traced");
}

TEST(RenderDemoOptions, RejectsBenchRangesAndDependencies) {
  EXPECT_FALSE(parse({"--bench-indirect", "b.json", "--bench-blocks", "0"}));
  EXPECT_FALSE(parse({"--bench-indirect", "b.json", "--bench-rebind", "1001,0"}));
  EXPECT_FALSE(parse({"--bench-indirect", "b.json", "--bench-rebind", "500"}));
  EXPECT_FALSE(parse({"--bench-indirect", ""}));
  EXPECT_EQ(parse_error({"--bench-blocks", "3"}), "--bench-blocks needs --bench-indirect");
  EXPECT_EQ(parse_error({"--bench-rebind", "1000,500"}), "--bench-rebind needs --bench-indirect");
}

TEST(RenderDemoOptions, ParsesGpuChannelFlags) {
  const auto defaults = parse({});
  ASSERT_TRUE(defaults);
  EXPECT_FALSE(defaults->gpuArrays);
  EXPECT_EQ(defaults->gpuPadMiB, 0u);
  const auto options = parse({"--gpu-arrays", "--gpu-pad-mib", "40", "--trace", "t.gxtrace", "--no-hints"});
  ASSERT_TRUE(options);
  EXPECT_TRUE(options->gpuArrays);
  EXPECT_EQ(options->gpuPadMiB, 40u);
  EXPECT_FALSE(options->hints);
  EXPECT_FALSE(parse({"--gpu-pad-mib", "0"}));
  EXPECT_FALSE(parse({"--gpu-pad-mib", "1025"}));
  EXPECT_FALSE(parse({"--gpu-pad-mib"}));
}

TEST(RenderDemoOptions, BenchExcludesFrameLoopFlags) {
  const std::vector<std::vector<const char*>> frameLoop{{"--frames", "10"},
                                                        {"--timeout", "5"},
                                                        {"--trace", "t.gxtrace"},
                                                        {"--trace-start", "3"},
                                                        {"--trace-frames", "2"},
                                                        {"--vignette"},
                                                        {"--vignette-anchor", "phase.sky"},
                                                        {"--expect-anchor-warnings", "0"},
                                                        {"--depth-check"},
                                                        {"--no-hints"},
                                                        {"--check-hints"},
                                                        {"--custom-draw-probe"},
                                                        {"--gpu-cull", "4096"},
                                                        {"--cull-check"},
                                                        {"--cull-timings"},
                                                        {"--gpu-arrays"},
                                                        {"--gpu-pad-mib", "8"}};
  for (const auto& flag : frameLoop) {
    std::vector<const char*> args{"--bench-indirect", "b.json"};
    args.insert(args.end(), flag.begin(), flag.end());
    EXPECT_EQ(parse_error(args), "--bench-indirect cannot be combined with " + std::string{flag[0]});
  }
}

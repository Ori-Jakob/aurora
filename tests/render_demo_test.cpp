#include <gtest/gtest.h>

#include "graph_checks.hpp"
#include "options.hpp"
#include "procedural.hpp"

#include <algorithm>
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

#include <gtest/gtest.h>

#include "render_ao_reference.hpp"

#include "render/ao/ao.hpp"
#include "render/post/composite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace aurora::render::detail {
namespace {
namespace ref = aurora::test::ao;

float lin(float stored) { return stored <= 0.04045f ? stored / 12.92f : std::pow((stored + 0.055f) / 1.055f, 2.4f); }

TEST(SsilRules, BounceBoundedByShare) {
  std::mt19937 rng{0x5511u};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  for (int i = 0; i < 4096; ++i) {
    const std::array<float, 4> t{unit(rng) * ao::SsilRadianceClamp, unit(rng) * ao::SsilRadianceClamp,
                                 unit(rng) * ao::SsilRadianceClamp, unit(rng)};
    const float v = unit(rng);
    const std::array<float, 3> e = ao::ssil_bounce_irradiance(t, v);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_GE(e[c], 0.f);
      EXPECT_LE(e[c], 3.14159265f * (1.f - v) * ao::SsilRadianceClamp * (1.f + 1e-6f));
    }
  }
  EXPECT_EQ(ao::ssil_bounce_irradiance({16.f, 16.f, 16.f, 1.f}, 1.f), (std::array<float, 3>{0.f, 0.f, 0.f}));
}

TEST(SsilRules, RadianceClampKeepsChroma) {
  const ref::Rgba hot = ref::ssil_clamp_radiance({64.0, 32.0, 8.0, 1.0});
  EXPECT_DOUBLE_EQ(hot[0], 16.0);
  EXPECT_DOUBLE_EQ(hot[1], 8.0);
  EXPECT_DOUBLE_EQ(hot[2], 2.0);
  const ref::Rgba cool = ref::ssil_clamp_radiance({0.5, 0.25, 0.125, 1.0});
  EXPECT_EQ(cool, (ref::Rgba{0.5, 0.25, 0.125, 1.0}));
  for (const double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    const ref::Rgba out = ref::ssil_clamp_radiance({bad, 1.0, 1.0, 1.0});
    EXPECT_EQ(out[0], 0.0);
    EXPECT_EQ(out[1], 0.0);
    EXPECT_EQ(out[2], 0.0);
  }
}

TEST(SsilRules, PrevDepthWeightsIgnoreCounts) {
  const std::array<double, 4> logDepths{std::log2(100.0), std::log2(105.0), std::log2(150.0), std::log2(95.0)};
  const std::array<double, 4> w = ref::ssil_prev_depth_weights(0.25, 0.5, logDepths, 100.0, ao::DepthReject);
  EXPECT_DOUBLE_EQ(w[0], 0.75 * 0.5);
  EXPECT_DOUBLE_EQ(w[1], 0.25 * 0.5);
  EXPECT_DOUBLE_EQ(w[2], 0.0);
  EXPECT_DOUBLE_EQ(w[3], 0.25 * 0.5);
}

TEST(SsilRules, FacingRejectsBackFaces) {
  const ref::Normal3 toEmitter{0.0, 0.0, 1.0};
  EXPECT_EQ(ref::ssil_facing({0.0, 0.0, -1.0}, toEmitter), 1.0);
  EXPECT_EQ(ref::ssil_facing({0.0, 0.0, 1.0}, toEmitter), 0.0);
  EXPECT_EQ(ref::ssil_facing({1.0, 0.0, 0.0}, toEmitter), 1.0);
  EXPECT_EQ(ref::ssil_facing({0.0, 0.0, 0.25}, toEmitter), 0.0);
  EXPECT_DOUBLE_EQ(ref::ssil_facing({0.0, 0.0, 0.125}, toEmitter), 0.5);
}

TEST(SsilRules, MixedFootprintLeansFacing) {
  const ref::Normal3 facing{0.0, 0.0, -1.0};
  const ref::Normal3 away{0.0, 0.0, 1.0};
  const ref::Normal3 toEmitter{0.0, 0.0, 1.0};
  const ref::Normal3 mixed{0.5 * (facing[0] + away[0]), 0.5 * (facing[1] + away[1]), 0.5 * (facing[2] + away[2])};
  EXPECT_EQ(ref::ssil_facing(mixed, toEmitter), 1.0);
  const ref::Normal3 mostlyAway{0.0, 0.0, 0.25 * facing[2] + 0.75 * away[2]};
  EXPECT_EQ(ref::ssil_facing(mostlyAway, toEmitter), 0.0);
}

TEST(SsilRules, CodeZeroSourceHasZeroNormal) { EXPECT_EQ(ref::ssil_facing({0.0, 0.0, 0.0}, {0.3, 0.4, 0.866}), 1.0); }

TEST(SsilRules, MeanNormalPyramidMatchesHizWeights) {
  ref::CpuPyramid depth;
  depth.widths = {4, 2};
  depth.heights = {2, 1};
  depth.levels = {{100.0, 101.0, 140.0, 300.0, 100.5, 102.0, 141.0, 1e30}, {0.0, 0.0}};
  const double radius = 73.0;
  std::vector<ref::Rgba> radiance;
  std::vector<ref::Normal3> normals;
  for (int i = 0; i < 8; ++i) {
    radiance.push_back({0.1 * i, 0.2, 0.3, 1.0});
    normals.push_back({0.0, i % 2 == 0 ? 1.0 : -1.0, 0.0});
  }
  const ref::SsilPyramids p = ref::build_ssil_pyramids(radiance, normals, depth, radius);
  ASSERT_EQ(p.radiance.size(), 2u);
  for (uint32_t x = 0; x < 2; ++x) {
    const std::array<size_t, 4> taps{2 * x, 2 * x + 1, 4 + 2 * x, 5 + 2 * x};
    const std::array<double, 4> d{depth.levels[0][taps[0]], depth.levels[0][taps[1]], depth.levels[0][taps[2]],
                                  depth.levels[0][taps[3]]};
    const std::array<double, 4> w = ref::ssil_depth_mip_weights(d, radius);
    double total = 0.0;
    double weightedDepth = 0.0;
    double red = 0.0;
    for (size_t i = 0; i < 4; ++i) {
      total += w[i];
      weightedDepth += w[i] * d[i];
      red += w[i] * radiance[taps[i]][0];
    }
    EXPECT_NEAR(weightedDepth / total, ref::depth_mip_filter(d, radius), 1e-9 * d[0]);
    EXPECT_NEAR(p.radiance[1][x][0], red / total, 1e-12);
  }
}

TEST(SsilRules, SkyAndDistantAddNothing) {
  EXPECT_EQ(ao::ssil_bounce_irradiance({0.f, 0.f, 0.f, 0.f}, 0.2f), (std::array<float, 3>{0.f, 0.f, 0.f}));
  EXPECT_EQ(ao::ssil_legacy_gain(1.f, {1.f, 1.f, 1.f}, {0.5f, 0.5f, 0.5f}), (std::array<float, 3>{1.f, 1.f, 1.f}));
}

TEST(SsilRules, CameraFlipIsSymmetric) {
  const ref::Normal3 n{0.3, -0.2, 0.9};
  const ref::Normal3 flipped{-n[0], -n[1], -n[2]};
  const ref::Normal3 toEmitter{0.1, 0.2, 0.97};
  EXPECT_DOUBLE_EQ(ref::ssil_facing(flipped, toEmitter),
                   std::clamp(1.0 + 4.0 * (n[0] * toEmitter[0] + n[1] * toEmitter[1] + n[2] * toEmitter[2]), 0.0, 1.0));
}

TEST(SsilRules, FeedbackConverges) {
  const double albedo = 0.9;
  const double as = 0.4;
  const double e = 1.0;
  double h = 0.0;
  double last = 0.0;
  for (int frame = 0; frame < 64; ++frame) {
    last = h;
    h = albedo * (as * e + (1.0 - as) * h);
    EXPECT_LE(h, e);
    EXPECT_GE(h, last);
  }
  const double fixed = albedo * as * e / (1.0 - albedo * (1.0 - as));
  EXPECT_NEAR(h, fixed, 1e-3);
  EXPECT_LT(fixed, e);
}

TEST(CompositeRules, LegacyScaleIsAoWithoutBounce) {
  std::mt19937 rng{0x5512u};
  std::uniform_int_distribution<int> byte(0, 255);
  for (int i = 0; i < 4096; ++i) {
    const std::array<float, 3> l{byte(rng) / 255.f, byte(rng) / 255.f, byte(rng) / 255.f};
    const std::array<float, 3> fc{byte(rng) / 1020.f, byte(rng) / 1020.f, byte(rng) / 1020.f};
    const auto q7 = static_cast<uint32_t>(byte(rng) / 2);
    const float a = byte(rng) / 255.f;
    const float code = static_cast<float>(i % 4) / 3.f;
    const std::array<float, 3> want = composite::testing::legacy_ao(l, fc, q7, a, code);
    const std::array<float, 3> got = composite::testing::legacy_ssil(l, fc, q7, a, code, {0.f, 0.f, 0.f});
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_EQ(std::bit_cast<uint32_t>(got[c]), std::bit_cast<uint32_t>(want[c])) << i;
    }
  }
}

TEST(CompositeRules, LegacyGainBetweenAoAndOriginal) {
  std::mt19937 rng{0x5513u};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  for (int i = 0; i < 8192; ++i) {
    const float a = unit(rng);
    const std::array<float, 3> bounce{unit(rng) * 8.f, unit(rng) * 8.f, unit(rng) * 8.f};
    const std::array<float, 3> receiver{unit(rng), unit(rng) * unit(rng), unit(rng) * 0.01f};
    const std::array<float, 3> g = ao::ssil_legacy_gain(a, bounce, receiver);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_GE(g[c], a) << i;
      EXPECT_LE(g[c], 1.f) << i;
    }
  }
  const std::array<float, 3> black = ao::ssil_legacy_gain(0.5f, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f});
  EXPECT_EQ(black, (std::array<float, 3>{1.f, 1.f, 1.f}));
}

TEST(CompositeRules, LegacyGainKeepsReceiverHue) {
  const std::array<float, 3> grass{0.010f, 0.214f, 0.004f};
  const float rock = 0.214f;
  const std::array<float, 3> bounce{ao::SsilLegacyAlbedo * rock, ao::SsilLegacyAlbedo * rock,
                                    ao::SsilLegacyAlbedo * rock};
  const std::array<float, 3> g = ao::ssil_legacy_gain(0.6f, bounce, grass);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(g[c], 0.90f, 0.005f) << c;
    EXPECT_EQ(g[c], g[0]);
  }
}

TEST(CompositeRules, LegacyGainColouredBounceTints) {
  const std::array<float, 3> grey{0.2f, 0.2f, 0.2f};
  const std::array<float, 3> g = ao::ssil_legacy_gain(0.5f, {0.3f, 0.03f, 0.03f}, grey);
  EXPECT_GT(g[0], g[1]);
  EXPECT_EQ(g[1], g[2]);
  EXPECT_GE(g[1], 0.5f);
}

TEST(CompositeRules, LegacyGainGreyReceiverMatchesPbrFill) {
  for (const float stored : {0.2f, 0.45f, 0.7f}) {
    for (const float a : {0.3f, 0.6f, 0.9f}) {
      for (const float lb : {0.01f, 0.05f, 0.1f}) {
        const float r = lin(stored);
        const float bounce = ao::SsilLegacyAlbedo * lb;
        const std::array<float, 3> g = ao::ssil_legacy_gain(a, {bounce, bounce, bounce}, {r, r, r});
        const float as = std::pow(a, 2.2f);
        const float scale = std::pow(stored, 2.2f) / r;
        const float pbr = std::pow(stored, 2.2f) * as + (1.f - as) * bounce * scale;
        const float want = std::clamp(pbr, std::pow(stored * a, 2.2f), std::pow(stored, 2.2f));
        EXPECT_NEAR(std::pow(stored * g[0], 2.2f), want, 1e-4f * want + 1e-7f) << stored << " " << a << " " << lb;
      }
    }
  }
}

TEST(CompositeRules, IdentityUnchangedAtFullAo) {
  const std::array<float, 3> l{0.4f, 0.5f, 0.6f};
  const std::array<float, 3> out = composite::testing::legacy_ssil(l, {}, 0, 1.f, 1.f, {2.f, 2.f, 2.f});
  EXPECT_EQ(out, l);
  const std::array<float, 3> codeZero = composite::testing::legacy_ssil(l, {}, 0, 0.5f, 0.f, {2.f, 2.f, 2.f});
  EXPECT_EQ(codeZero, l);
}

TEST(ExposureRules, SbTargetCarriesBounce) {
  const std::array<float, 3> l{0.4f, 0.3f, 0.2f};
  const std::array<float, 3> ao = composite::testing::legacy_ao(l, {}, 0, 0.5f, 1.f);
  const std::array<float, 3> lifted = composite::testing::legacy_ssil(l, {}, 0, 0.5f, 1.f, {0.05f, 0.05f, 0.05f});
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_GT(lifted[c], ao[c]);
    EXPECT_LE(lifted[c], l[c]);
  }
}
} // namespace
} // namespace aurora::render::detail

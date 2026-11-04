#include <gtest/gtest.h>

#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "gx/surface_pack.hpp"
#include "render/ao/ao.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/lighting/clusters.hpp"
#include "render/lighting/deferred.hpp"
#include "render/lighting/exposure.hpp"
#include "render/lighting/openpbr.hpp"
#include "render/lighting/sky_probe.hpp"
#include "render/lighting/tonemap.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/post/composite.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace exposure = lighting::exposure;
namespace tonemap = lighting::tonemap;
using exposure::Params;
using exposure::Solved;
using Rgb = tonemap::Rgb;
using Texels = std::array<uint32_t, 8>;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float Inf = std::numeric_limits<float>::infinity();
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

double log2_mean(std::span<const Rgb> colours, float e, bool agx) {
  double sum = 0.0;
  for (const Rgb& c : colours) {
    const Rgb mapped = tonemap::apply({c[0] * e, c[1] * e, c[2] * e}, agx);
    sum += std::log2(std::max(tonemap::luminance(mapped), exposure::YFloor));
  }
  return sum / static_cast<double>(colours.size());
}

std::vector<Rgb> scene_colours(uint32_t seed, size_t count, float scale) {
  std::mt19937 rng{seed};
  std::uniform_real_distribution<float> unit(0.02f, 1.f);
  std::vector<Rgb> out(count);
  for (Rgb& c : out) {
    c = {unit(rng) * scale, unit(rng) * scale, unit(rng) * scale};
  }
  return out;
}

// Same sums, order and candidates as the meter, so the solve mirror can run on CPU-built partials
std::vector<float> partials_for(std::span<const Rgb> hdr, std::span<const float> yLegacy, const Params& params,
                                const Texels& prev) {
  const bool snap = exposure::testing::snaps(params, prev);
  const float step = snap ? exposure::SnapStepEv : exposure::StepEv;
  const float center = exposure::testing::center(params, prev);
  const bool agx = (params.flags & exposure::FlagAgx) != 0;
  std::vector<float> out(size_t{exposure::Groups} * exposure::PartialWords, 0.f);
  for (size_t i = 0; i < hdr.size(); ++i) {
    float* word = out.data() + (i % exposure::Groups) * exposure::PartialWords;
    word[exposure::WordWeight] += 1.f;
    word[exposure::WordTarget] += std::log2(std::max(yLegacy[i], exposure::YFloor));
    word[exposure::WordCount] += 1.f;
    for (uint32_t k = 0; k < exposure::Candidates; ++k) {
      const float e = std::exp2(exposure::testing::candidate(center, step, k));
      const Rgb mapped = tonemap::apply({hdr[i][0] * e, hdr[i][1] * e, hdr[i][2] * e}, agx);
      word[k] += std::log2(std::max(tonemap::luminance(mapped), exposure::YFloor));
    }
  }
  return out;
}

std::vector<float> linear_partials(float weight, float count, float offset, float target, const Params& params,
                                   const Texels& prev) {
  const bool snap = exposure::testing::snaps(params, prev);
  const float step = snap ? exposure::SnapStepEv : exposure::StepEv;
  const float center = exposure::testing::center(params, prev);
  std::vector<float> out(size_t{exposure::Groups} * exposure::PartialWords, 0.f);
  out[exposure::WordWeight] = weight;
  out[exposure::WordCount] = count;
  out[exposure::WordTarget] = weight * target;
  for (uint32_t k = 0; k < exposure::Candidates; ++k) {
    out[k] = weight * (offset + exposure::testing::candidate(center, step, k));
  }
  return out;
}

Texels valid_prev(float s, Exposure mode, bool agx = false) {
  Solved state;
  state.s = s;
  state.flags = exposure::StateValid | (static_cast<uint32_t>(mode) << exposure::StateModeShift) |
                (agx ? uint32_t{exposure::StateAgx} : 0u);
  return exposure::testing::texels(state);
}

Params params_of(Exposure mode, float dt = 1.f / 60.f, uint32_t flags = 0) {
  return {.mode = static_cast<uint32_t>(mode), .flags = flags, .dt = dt};
}

TEST(TonemapRules, NeutralReferenceValues) {
  const std::array<std::pair<float, float>, 4> cases{
      {{0.18f, 0.14f}, {0.5f, 0.46f}, {1.f, 0.869091f}, {0.05f, 0.015625f}}};
  for (const auto& [in, want] : cases) {
    const Rgb out = tonemap::pbr_neutral({in, in, in});
    for (const float c : out) {
      EXPECT_NEAR(c, want, 2e-6f) << in;
    }
  }
}

TEST(TonemapRules, AgxReferenceValues) {
  // agx_ref.py over Filament's AgX at 54c63c4 with BT.2087 around it, in f32
  const std::array<std::pair<Rgb, Rgb>, 8> cases{{
      {{0.18f, 0.18f, 0.18f}, {0.2148484f, 0.2148484f, 0.2148484f}},
      {{0.f, 0.f, 0.f}, {2.529312e-06f, 2.529312e-06f, 2.529312e-06f}},
      {{0.01f, 0.01f, 0.01f}, {0.005812999f, 0.005812965f, 0.00581299f}},
      {{1.f, 1.f, 1.f}, {0.5829212f, 0.5829213f, 0.5829213f}},
      {{16.f, 16.f, 16.f}, {0.9603773f, 0.9603773f, 0.9603776f}},
      {{1.f, 0.5f, 0.25f}, {0.6000227f, 0.4221885f, 0.3053987f}},
      {{0.1f, 0.4f, 0.9f}, {0.2013493f, 0.4003913f, 0.5889366f}},
      {{2.f, 0.05f, 0.05f}, {0.8925243f, 0.2526055f, 0.1926869f}},
  }};
  for (const auto& [in, want] : cases) {
    const Rgb out = tonemap::agx(in);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(out[c], want[c], 2e-5f) << in[0] << " " << in[1] << " " << in[2] << " channel " << c;
    }
  }
}

TEST(TonemapRules, AgxNeutralStaysNeutral) {
  for (float v = 1e-4f; v < 1e4f; v *= 1.37f) {
    const Rgb out = tonemap::agx({v, v, v});
    EXPECT_NEAR(out[0], out[1], 2e-6f) << v;
    EXPECT_NEAR(out[1], out[2], 2e-6f) << v;
  }
}

TEST(TonemapRules, AgxProperties) {
  float previous = -1.f;
  for (float v = 1e-6f; v < 6e4f; v *= 1.21f) {
    const Rgb out = tonemap::agx({v, v, v});
    for (const float c : out) {
      EXPECT_GE(c, 0.f);
      EXPECT_LE(c, 1.f);
    }
    // The contrast fit dips by about 2e-6 just above black
    EXPECT_GE(out[1], previous - 5e-6f) << v;
    previous = out[1];
  }
  EXPECT_LT(tonemap::agx({0.f, 0.f, 0.f})[1], 1e-5f);
  const float top = tonemap::agx({1e4f, 1e4f, 1e4f})[1];
  EXPECT_GT(top, 0.95f);
  EXPECT_EQ(tonemap::agx({6e4f, 6e4f, 6e4f})[1], top);
  const Rgb grey = tonemap::agx({0.18f, 0.18f, 0.18f});
  EXPECT_NEAR(grey[1], 0.2148f, 1e-3f);
}

TEST(TonemapRules, AgxInsetIsScaleSeparable) {
  std::mt19937 rng{11};
  std::uniform_real_distribution<float> unit(1e-3f, 4.f);
  for (int i = 0; i < 200; ++i) {
    const Rgb c{unit(rng), unit(rng), unit(rng)};
    const float s = std::uniform_real_distribution<float>(-8.f, 8.f)(rng);
    const float e = std::exp2(s);
    const Rgb direct = tonemap::agx({c[0] * e, c[1] * e, c[2] * e});
    Rgb l = tonemap::agx_inset_log2(c);
    for (float& v : l) {
      v += s;
    }
    const Rgb split = tonemap::agx_from_inset(l);
    for (size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(split[k], direct[k], 5e-5f) << i;
    }
  }
}

TEST(TonemapRules, NonFiniteAndNegativeInputsAreFinite) {
  const std::array<Rgb, 6> inputs{{
      {NaN, 0.5f, 0.5f},
      {Inf, Inf, Inf},
      {-Inf, 1.f, 2.f},
      {-1.f, -2.f, -3.f},
      {65504.f, 0.f, 1.f},
      {std::numeric_limits<float>::max(), 1.f, 1.f},
  }};
  for (const Rgb& in : inputs) {
    for (const bool agx : {false, true}) {
      const Rgb out = tonemap::apply(in, agx);
      for (const float c : out) {
        EXPECT_TRUE(std::isfinite(c)) << agx;
        EXPECT_GE(c, 0.f) << agx;
        EXPECT_LE(c, 1.f) << agx;
      }
    }
  }
  EXPECT_EQ(tonemap::sanitize({NaN, Inf, -Inf}), (Rgb{0.f, 65504.f, 0.f}));
  EXPECT_EQ(tonemap::sanitize({-0.5f, 2.f, 0.f}), (Rgb{0.f, 2.f, 0.f}));
}

TEST(TonemapRules, LuminanceMonotoneUnderScale) {
  const std::vector<Rgb> colours = scene_colours(5, 64, 1.f);
  for (const bool agx : {false, true}) {
    for (const Rgb& c : colours) {
      float previous = -1.f;
      for (float s = -10.f; s <= 10.f; s += 0.25f) {
        const float e = std::exp2(s);
        const float y = tonemap::luminance(tonemap::apply({c[0] * e, c[1] * e, c[2] * e}, agx));
        ASSERT_GE(y, previous - 1e-6f) << agx << " " << s;
        previous = y;
      }
    }
  }
}

TEST(ExposureRules, ParamsAndStateLayout) {
  EXPECT_EQ(sizeof(Params), 32u);
  EXPECT_EQ(offsetof(Params, mode), 0u);
  EXPECT_EQ(offsetof(Params, flags), 4u);
  EXPECT_EQ(offsetof(Params, bias), 8u);
  EXPECT_EQ(offsetof(Params, envBias), 12u);
  EXPECT_EQ(offsetof(Params, dt), 16u);
  EXPECT_EQ(offsetof(Params, anchor), 20u);
  EXPECT_EQ(exposure::PartialBytes, 81u * 36u * 4u);
  EXPECT_EQ(exposure::BufferBytes, 16u);
  EXPECT_EQ(exposure::Groups * exposure::GroupSize * exposure::SamplesPerThread, 192u * 108u);
  const TexDesc state = exposure::state_desc();
  EXPECT_EQ(state.format, wgpu::TextureFormat::RGBA32Uint);
  EXPECT_EQ(state.sizeClass, SizeClass::Absolute);
  EXPECT_EQ(state.width, 2u);
  EXPECT_EQ(state.height, 1u);
  EXPECT_TRUE(state.extraUsage & wgpu::TextureUsage::TextureBinding);
  Solved solved{
      .s = 1.5f, .flags = 3, .target = 2.f, .weight = 4.f, .t = -1.f, .fLo = -0.5f, .fHi = 0.5f, .count = 9.f};
  const Texels texels = exposure::testing::texels(solved);
  EXPECT_EQ(texels[0], std::bit_cast<uint32_t>(1.5f));
  EXPECT_EQ(texels[1], 3u);
  EXPECT_EQ(texels[2], std::bit_cast<uint32_t>(2.f));
  EXPECT_EQ(texels[3], std::bit_cast<uint32_t>(4.f));
  EXPECT_EQ(texels[4], std::bit_cast<uint32_t>(-1.f));
  EXPECT_EQ(texels[7], std::bit_cast<uint32_t>(9.f));
  EXPECT_EQ(static_cast<uint32_t>(Exposure::MatchLegacy), 0u);
  EXPECT_EQ(static_cast<uint32_t>(Exposure::Manual), 2u);
}

TEST(ExposureRules, MatchSolvesLogAverage) {
  const std::vector<Rgb> hdr = scene_colours(3, 2000, 0.05f);
  const std::vector<float> legacy(hdr.size(), 0.12f);
  for (const bool agx : {false, true}) {
    const Params params = params_of(Exposure::MatchLegacy, 1.f / 60.f, agx ? exposure::FlagAgx : 0u);
    const Solved solved = exposure::testing::solve(params, partials_for(hdr, legacy, params, {}), {});
    EXPECT_FALSE(solved.flags & exposure::StateHeld);
    EXPECT_FALSE(solved.flags & exposure::StatePinned);
    EXPECT_TRUE(solved.flags & exposure::StateSnapped);
    EXPECT_EQ(solved.s, solved.target);
    EXPECT_NEAR(solved.t, std::log2(0.12f), 1e-4f);
    EXPECT_NEAR(log2_mean(hdr, std::exp2(solved.s), agx), std::log2(0.12), 0.03) << agx;
  }
}

TEST(ExposureRules, AutoHitsKeyThroughTonemapper) {
  const std::vector<Rgb> hdr = scene_colours(4, 2000, 3.f);
  const std::vector<float> legacy(hdr.size(), 0.5f);
  for (const bool agx : {false, true}) {
    const Params params = params_of(Exposure::Auto, 1.f / 60.f, agx ? exposure::FlagAgx : 0u);
    const Solved solved = exposure::testing::solve(params, partials_for(hdr, legacy, params, {}), {});
    EXPECT_NEAR(solved.t, std::log2(exposure::Key), 1e-6f);
    EXPECT_NEAR(log2_mean(hdr, std::exp2(solved.s), agx), std::log2(0.18), 0.03) << agx;
  }
}

TEST(ExposureRules, ManualIsTwoToBias) {
  Params params = params_of(Exposure::Manual);
  params.bias = 1.5f;
  params.envBias = 0.5f;
  std::vector<float> partials(size_t{exposure::Groups} * exposure::PartialWords, NaN);
  const Solved solved = exposure::testing::solve(params, partials, valid_prev(-3.f, Exposure::Manual));
  EXPECT_EQ(solved.s, 2.f);
  EXPECT_EQ(solved.weight, 0.f);
  EXPECT_EQ(solved.t, 0.f);
  EXPECT_FALSE(solved.flags & (exposure::StateHeld | exposure::StatePinned));
  const std::array<float, 4> words = exposure::testing::words(solved);
  EXPECT_EQ(words[0], 4.f);
  EXPECT_EQ(words[1], 0.25f);
  EXPECT_EQ(words[2], 32.f);
  EXPECT_EQ(words[3], 1.f);
  params.bias = 0.f;
  params.envBias = 0.f;
  EXPECT_EQ(exposure::testing::words(exposure::testing::solve(params, partials, {}))[0], 1.f);
  params.bias = 20.f;
  EXPECT_EQ(exposure::testing::solve(params, partials, {}).s, exposure::MaxEv);
  for (const float s : {-8.f, -1.f, 0.f, 1.f, 3.f, 8.f}) {
    EXPECT_EQ(exposure::testing::exp2_exact(s), std::ldexp(1.f, static_cast<int>(s))) << s;
  }
  Params anchored;
  exposure::fill_params({.exposure = Exposure::Manual}, exposure::Target::Sb, 16.f, false, anchored);
  const float e0 = exposure::testing::words(exposure::testing::solve(anchored, partials, {}))[0];
  EXPECT_EQ(e0, exposure::testing::exp2_exact(exposure::ManualAnchorEv));
  anchored.bias = 1.f;
  EXPECT_EQ(exposure::testing::words(exposure::testing::solve(anchored, partials, {}))[0], 2.f * e0);
  anchored.bias = -1.f;
  EXPECT_EQ(exposure::testing::words(exposure::testing::solve(anchored, partials, {}))[0], 0.5f * e0);
}

TEST(ExposureRules, BiasOnlyOutsideMatch) {
  for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto, Exposure::Manual}) {
    Params params;
    exposure::fill_params({.exposure = mode, .exposureBias = 2.5f}, exposure::Target::Sb, 16.f, false, params);
    EXPECT_EQ(params.bias, mode == Exposure::MatchLegacy ? 0.f : 2.5f) << static_cast<int>(mode);
    EXPECT_EQ(params.mode, static_cast<uint32_t>(mode));
    EXPECT_NEAR(params.dt, 0.016f, 1e-7f);
    EXPECT_EQ(params.anchor, exposure::ManualAnchorEv);
  }
  Params params;
  exposure::fill_params({.exposure = Exposure::Auto, .exposureBias = 40.f}, exposure::Target::S, 1000.f, true, params);
  EXPECT_EQ(params.bias, exposure::MaxEv);
  EXPECT_EQ(params.dt, 0.25f);
  EXPECT_EQ(params.flags, uint32_t{exposure::FlagTargetS | exposure::FlagSnap});
  exposure::fill_params({.tonemap = Tonemap::AgX}, exposure::Target::Sb, NaN, false, params);
  EXPECT_EQ(params.flags, uint32_t{exposure::FlagAgx});
  EXPECT_EQ(params.dt, 0.f);
  const std::vector<Rgb> hdr = scene_colours(6, 2000, 1.f);
  const std::vector<float> legacy(hdr.size(), 0.2f);
  Params match = params_of(Exposure::MatchLegacy);
  const Solved plain = exposure::testing::solve(match, partials_for(hdr, legacy, match, {}), {});
  match.bias = 3.f;
  match.envBias = 2.f;
  EXPECT_EQ(exposure::testing::solve(match, partials_for(hdr, legacy, match, {}), {}).s, plain.s);
}

TEST(ExposureRules, EnvBiasOnlyOutsideMatch) {
  auto hints = std::make_shared<SceneHints>();
  hints->present = SceneHasEnvironment;
  hints->motionValid = true;
  hints->environment.value.exposureBias = 1.25f;
  EXPECT_EQ(exposure::env_bias(hints.get()), 1.25f);
  EXPECT_EQ(exposure::env_bias(nullptr), 0.f);
  hints->environment.value.exposureBias = NaN;
  EXPECT_EQ(exposure::env_bias(hints.get()), 0.f);
  hints->environment.value.exposureBias = -30.f;
  EXPECT_EQ(exposure::env_bias(hints.get()), -exposure::MaxEv);
  hints->environment.value.exposureBias = 1.25f;
  const PassDesc desc = exposure::pass_desc();
  ASSERT_NE(desc.stageParams, nullptr);
  const StageData stage{.stage = Stage::OpaqueDone, .sceneHints = hints};
  for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto, Exposure::Manual}) {
    Params params = params_of(mode);
    desc.stageParams(stage, nullptr, std::as_writable_bytes(std::span{&params, 1}));
    EXPECT_EQ(params.envBias, mode == Exposure::MatchLegacy ? 0.f : 1.25f) << static_cast<int>(mode);
    EXPECT_EQ(params.flags & exposure::FlagReset, 0u);
  }
  hints->motionValid = false;
  Params params = params_of(Exposure::Auto);
  desc.stageParams(stage, nullptr, std::as_writable_bytes(std::span{&params, 1}));
  EXPECT_NE(params.flags & exposure::FlagReset, 0u);
}

TEST(ExposureRules, SnapsOnResetModeAndTonemapChange) {
  const Texels prev = valid_prev(1.f, Exposure::MatchLegacy);
  const Params match = params_of(Exposure::MatchLegacy);
  EXPECT_FALSE(exposure::testing::snaps(match, prev));
  EXPECT_EQ(exposure::testing::center(match, prev), 1.f);
  EXPECT_TRUE(exposure::testing::snaps(match, {}));
  EXPECT_TRUE(exposure::testing::snaps(params_of(Exposure::MatchLegacy, 0.f, exposure::FlagReset), prev));
  EXPECT_TRUE(exposure::testing::snaps(params_of(Exposure::MatchLegacy, 0.f, exposure::FlagSnap), prev));
  EXPECT_TRUE(exposure::testing::snaps(params_of(Exposure::Auto), prev));
  EXPECT_TRUE(exposure::testing::snaps(params_of(Exposure::MatchLegacy, 0.f, exposure::FlagAgx), prev));
  EXPECT_FALSE(exposure::testing::snaps(params_of(Exposure::MatchLegacy, 0.f, exposure::FlagAgx),
                                        valid_prev(1.f, Exposure::MatchLegacy, true)));
  EXPECT_EQ(exposure::testing::center(match, valid_prev(7.f, Exposure::MatchLegacy)), 8.f - 15.5f * 0.25f);
  const std::vector<float> partials = linear_partials(1000.f, 1000.f, 0.f, 3.f, match, prev);
  const Solved smooth = exposure::testing::solve(match, partials, prev);
  EXPECT_FALSE(smooth.flags & exposure::StateSnapped);
  EXPECT_LT(smooth.s, 3.f);
  EXPECT_GT(smooth.s, 1.f);
  const Params reset = params_of(Exposure::MatchLegacy, 1.f / 60.f, exposure::FlagReset);
  const Solved snapped = exposure::testing::solve(reset, linear_partials(1000.f, 1000.f, 0.f, 3.f, reset, prev), prev);
  EXPECT_TRUE(snapped.flags & exposure::StateSnapped);
  EXPECT_NEAR(snapped.s, 3.f, 1e-5f);
  EXPECT_EQ(exposure::testing::words(snapped)[2], 1.f);
}

TEST(ExposureRules, HoldsWithFewSamples) {
  const Texels prev = valid_prev(1.5f, Exposure::Auto);
  const Params params = params_of(Exposure::Auto);
  const Solved held = exposure::testing::solve(params, linear_partials(200.f, 200.f, 0.f, 3.f, params, prev), prev);
  EXPECT_TRUE(held.flags & exposure::StateHeld);
  EXPECT_EQ(held.s, 1.5f);
  const Solved enough = exposure::testing::solve(params, linear_partials(208.f, 208.f, 0.f, 3.f, params, prev), prev);
  EXPECT_FALSE(enough.flags & exposure::StateHeld);
  EXPECT_NE(enough.s, 1.5f);
  const Solved empty = exposure::testing::solve(params, linear_partials(0.f, 0.f, 0.f, 3.f, params, {}), {});
  EXPECT_TRUE(empty.flags & exposure::StateHeld);
  EXPECT_EQ(empty.s, 0.f);
}

TEST(ExposureRules, HeldResetSnapsTheNextMeasuredFrame) {
  const Params reset = params_of(Exposure::MatchLegacy, 1.f / 60.f, exposure::FlagReset);
  const Params match = params_of(Exposure::MatchLegacy);
  const Solved few = exposure::testing::solve(reset, linear_partials(100.f, 100.f, 0.f, 3.f, reset, {}), {});
  const Solved empty = exposure::testing::solve(reset, linear_partials(0.f, 0.f, 0.f, 3.f, reset, {}), {});
  EXPECT_NEAR(few.s, 3.f, 1e-5f);
  EXPECT_EQ(empty.s, 0.f);
  const Texels kept = valid_prev(1.5f, Exposure::MatchLegacy);
  const Params snap = params_of(Exposure::MatchLegacy, 1.f / 60.f, exposure::FlagSnap);
  const Solved held = exposure::testing::solve(snap, linear_partials(100.f, 100.f, 0.f, 3.f, snap, kept), kept);
  EXPECT_EQ(held.s, 1.5f);
  for (const Solved& before : {few, empty, held}) {
    EXPECT_EQ(before.flags & (exposure::StateHeld | exposure::StateSnapped),
              uint32_t{exposure::StateHeld | exposure::StateSnapped});
    const Texels prev = exposure::testing::texels(before);
    EXPECT_TRUE(exposure::testing::snaps(match, prev));
    const Solved next = exposure::testing::solve(match, linear_partials(1000.f, 1000.f, 0.f, 5.f, match, prev), prev);
    EXPECT_TRUE(next.flags & exposure::StateSnapped);
    EXPECT_NEAR(next.s, 5.f, 1e-5f);
    EXPECT_EQ(exposure::testing::words(next)[2], 1.f);
    const Solved after = exposure::testing::solve(
        match, linear_partials(1000.f, 1000.f, 0.f, 6.f, match, exposure::testing::texels(next)),
        exposure::testing::texels(next));
    EXPECT_FALSE(after.flags & exposure::StateSnapped);
  }
}

TEST(ExposureRules, PinsUnreachableTarget) {
  const Params params = params_of(Exposure::MatchLegacy);
  const Solved high = exposure::testing::solve(params, linear_partials(1000.f, 1000.f, -40.f, 0.f, params, {}), {});
  EXPECT_TRUE(high.flags & exposure::StatePinned);
  EXPECT_EQ(high.s, 7.75f);
  const Solved low = exposure::testing::solve(params, linear_partials(1000.f, 1000.f, 40.f, 0.f, params, {}), {});
  EXPECT_TRUE(low.flags & exposure::StatePinned);
  EXPECT_EQ(low.s, -7.75f);
  const Solved inside = exposure::testing::solve(params, linear_partials(1000.f, 1000.f, 1.f, 0.f, params, {}), {});
  EXPECT_FALSE(inside.flags & exposure::StatePinned);
  EXPECT_NEAR(inside.s, -1.f, 1e-5f);
}

TEST(ExposureRules, SmoothingIsFrameRateIndependent) {
  for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto}) {
    const float target = mode == Exposure::Auto ? -2.f : 2.f;
    const Texels prev = valid_prev(0.f, mode);
    const auto step = [&](const Texels& from, float dt) {
      const Params params = params_of(mode, dt);
      const float offset = mode == Exposure::Auto ? std::log2(exposure::Key) - target : -target;
      return exposure::testing::solve(params, linear_partials(1000.f, 1000.f, offset, 0.f, params, from), from);
    };
    const Solved once = step(prev, 1.f / 30.f);
    const Solved half = step(prev, 1.f / 60.f);
    const Solved twice = step(exposure::testing::texels(half), 1.f / 60.f);
    EXPECT_NEAR(once.s, twice.s, 2e-5f) << static_cast<int>(mode);
    EXPECT_NE(once.s, target);
  }
}

TEST(ExposureRules, AsymmetricAutoTau) {
  const Texels prev = valid_prev(0.f, Exposure::Auto);
  const Params params = params_of(Exposure::Auto, 0.1f);
  const float offset = std::log2(exposure::Key);
  const Solved up =
      exposure::testing::solve(params, linear_partials(1000.f, 1000.f, offset - 1.f, 0.f, params, prev), prev);
  const Solved down =
      exposure::testing::solve(params, linear_partials(1000.f, 1000.f, offset + 1.f, 0.f, params, prev), prev);
  EXPECT_NEAR(up.target, 1.f, 1e-5f);
  EXPECT_NEAR(down.target, -1.f, 1e-5f);
  EXPECT_NEAR(up.s, exposure::testing::smoothing_alpha(0.1f, exposure::AutoUpTau), 1e-5f);
  EXPECT_NEAR(down.s, -exposure::testing::smoothing_alpha(0.1f, exposure::AutoDownTau), 1e-5f);
  const Params match = params_of(Exposure::MatchLegacy, 0.1f);
  const Texels matchPrev = valid_prev(0.f, Exposure::MatchLegacy);
  const Solved m =
      exposure::testing::solve(match, linear_partials(1000.f, 1000.f, -1.f, 0.f, match, matchPrev), matchPrev);
  EXPECT_NEAR(m.s, exposure::testing::smoothing_alpha(0.1f, exposure::MatchTau), 1e-5f);
}

TEST(ExposureRules, WordsAreIdentityAtWhite) {
  const std::array<float, 4> words = exposure::testing::words(Solved{});
  for (const float w : words) {
    EXPECT_EQ(std::bit_cast<uint32_t>(w), 0x3F800000u);
  }
  Solved state{.s = 0.f, .flags = exposure::StateValid, .previous = 2.f};
  EXPECT_EQ(exposure::testing::words(state)[2], 0.25f);
}

struct Readback {
  std::string_view resource;
  bool isBuffer = false;
  uint64_t size = 0;
  wgpu::Buffer buffer;
  int calls = 0;
};

void declare_readback(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& reader = *static_cast<const Readback*>(state);
  builder.after(phase::Exposure);
  if (reader.isBuffer) {
    builder.use_buffer(reader.resource, Access::CopySrc);
  } else {
    builder.use(reader.resource, Access::CopySrc);
  }
  builder.side_effect();
}

void encode_readback(PassContext& ctx, const void* state) {
  auto& reader = *static_cast<Readback*>(const_cast<void*>(state));
  ++reader.calls;
  if (!reader.buffer) {
    return;
  }
  if (reader.isBuffer) {
    const BufferView view = ctx.buffer(BufRef{0});
    if (view.buffer) {
      ctx.encoder().CopyBufferToBuffer(view.buffer, view.offset, reader.buffer, 0, std::min(view.size, reader.size));
    }
    return;
  }
  const wgpu::TexelCopyTextureInfo source{.texture = ctx.texture(TexRef{0})};
  const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = 256, .rowsPerImage = 1},
                                              .buffer = reader.buffer};
  const wgpu::Extent3D size{2, 1, 1};
  ctx.encoder().CopyTextureToBuffer(&source, &destination, &size);
}

PassDesc readback_desc(Readback& reader, std::string_view name) {
  return {
      .name = std::string{name},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 9,
      .declare = declare_readback,
      .encode = encode_readback,
      .state = &reader,
  };
}

struct StubTexture {
  std::string_view resource;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  std::string_view after;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  bool enabled = true;
};

void declare_stub(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& stub = *static_cast<const StubTexture*>(state);
  builder.after(stub.after);
  builder.create(stub.resource,
                 {.format = stub.format, .sizeClass = SizeClass::Efb, .extraUsage = wgpu::TextureUsage::TextureBinding},
                 Access::CopyDst);
}

bool stub_enabled(const StageData&, const void* state) { return static_cast<const StubTexture*>(state)->enabled; }

void encode_stub(PassContext& ctx, const void* state) {
  const auto& stub = *static_cast<const StubTexture*>(state);
  if (!stub.staging) {
    return;
  }
  const wgpu::Extent3D size = ctx.extent(TexRef{0});
  const wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = stub.rowBytes, .rowsPerImage = size.height},
                                       .buffer = stub.staging};
  const wgpu::TexelCopyTextureInfo to{.texture = ctx.texture(TexRef{0})};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  ctx.encoder().CopyBufferToTexture(&from, &to, &extent);
}

PassDesc stub_desc(StubTexture& stub, std::string_view name) {
  return {
      .name = std::string{name},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 0,
      .runsWithoutMarker = true,
      .declare = declare_stub,
      .enabled = stub_enabled,
      .encode = encode_stub,
      .state = &stub,
  };
}

std::shared_ptr<SceneHints> camera_hints(bool motionValid = true) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -10.f / (100000.f - 10.f);
  proj.m2[3] = -(100000.f * 10.f) / (100000.f - 10.f);
  proj.m3[2] = -1.f;
  auto hints = std::make_shared<SceneHints>();
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
  hints->present = SceneHasMainCamera | SceneHasLights;
  hints->motionValid = motionValid;
  return hints;
}

Settings pbr_settings(Exposure mode = Exposure::MatchLegacy, Tonemap tm = Tonemap::PbrNeutral,
                      float legacyBlend = 0.f) {
  Settings settings;
  settings.pbr = true;
  settings.legacyBlend = legacyBlend;
  settings.exposure = mode;
  settings.tonemap = tm;
  return settings;
}

struct MeterPixel {
  Rgb hdr{};
  bool lit = true;
  std::array<uint8_t, 3> legacy{128, 128, 128};
  std::array<uint8_t, 3> fc{};
  uint8_t q7 = 0;
  uint8_t ao = 255;
  uint8_t code = 1;
};

class ExposureDeviceTest : public ::testing::Test {
protected:
  bool open(const test::TestDeviceOptions& options) {
    m_options = options;
    m_options.createEfb = true;
    m_device = test::create_test_device(m_options, m_error);
    if (!m_device) {
      return false;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    gpu_heap::shutdown();
    reset_module_state();
    jobs::initialize(1);
    shaders::initialize();
    register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
    testing::set_hints_override(1);
    install_hints();
    m_started = true;
    start_modules();
    if (!webgpu::set_gbuffer(true)) {
      m_error = "the base G-buffer tier needs a core device";
      return false;
    }
    render::Caps c = render::caps();
    c.limits.maxColorAttachmentBytesPerSample = 128;
    webgpu::set_caps(c);
    if (!webgpu::set_scene_tier(webgpu::SceneTier::Pbr)) {
      m_error = "the PBR G-buffer tier is unavailable";
      return false;
    }
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    m_depth = test::make_depth(*m_device, size.width, size.height);
    return true;
  }

  void TearDown() override {
    if (m_started) {
      debug::set_view_override(0);
      webgpu::set_gbuffer(false);
      testing::reset_hints();
      shaders::shutdown();
      jobs::shutdown();
      reset_module_state();
      gpu_heap::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    exposure::shutdown();
    exposure::testing::reset();
    composite::shutdown();
    composite::testing::reset();
    lighting::deferred::shutdown();
    lighting::deferred::testing::reset();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  void start_modules() {
    composite::initialize();
    exposure::initialize();
    set_hints(camera_hints());
    exposure::testing::set_delta_ms(1000.f / 60.f);
    lighting::deferred::testing::force_enabled(true);
    register_pass(stub_desc(m_hdr, "test.hdr"));
  }

  void set_hints(std::shared_ptr<const SceneHints> hints) {
    m_hints = std::move(hints);
    composite::testing::set_camera_override(m_hints);
    exposure::testing::set_hints_override(m_hints);
  }

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(const Settings& settings) {
    m_settings = settings;
    m_env = gather_graph_environment();
    std::shared_ptr<CompiledGraph> graph =
        compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = DepthSource::Direct});
    prepare_graph_bindings(
        *graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = DepthSource::Direct});
    return graph;
  }

  void latch(const CompiledGraph* graph) {
    const PassSetup latched{.settings = &m_settings, .env = &m_env};
    exposure::on_latch(latched, graph);
    composite::on_latch(latched, graph);
    latch_pass_params();
  }

  static void wait_ready(const CompiledGraph& graph) {
    for (const CompiledNode& node : graph.nodes) {
      if (node.kind != NodeKind::Pass || node.status != CullReason::None) {
        continue;
      }
      for (const uint32_t id : node.pipelineIds) {
        if (id != 0) {
          ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << node.name;
        }
      }
    }
  }

  std::shared_ptr<CompiledGraph> live_graph(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    wait_ready(*graph);
    composite::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
    latch(graph.get());
    return graph;
  }

  static CullReason status(const CompiledGraph& graph, std::string_view name) {
    const NodeId node = graph.find_node(name);
    return node != InvalidNode ? graph.nodes[node].status : CullReason::Skipped;
  }

  static bool pass_enabled(const CompiledGraph& graph, const StageData& stage, std::string_view name) {
    const NodeId node = graph.find_node(name);
    return node != InvalidNode && graph.nodes[node].programIndex != InvalidNode &&
           stage.is_enabled(graph.nodes[node].programIndex);
  }

  StageData build(const FrameState& state, uint64_t frameId) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    StageData opaque{
        .stage = Stage::OpaqueDone,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = true,
        .frame = &state,
        .sceneDepth = test::scene_depth(m_depth),
    };
    build_stage_data(opaque);
    return opaque;
  }

  static gfx::Range uniform_sink(const uint8_t* data, size_t length) {
    const size_t alignment = std::max<size_t>(gfx::detail::resources().limits.minUniformBufferOffsetAlignment, 1);
    const size_t offset = (s_uniforms.size() + alignment - 1) / alignment * alignment;
    s_uniforms.resize(offset + length);
    std::memcpy(s_uniforms.data() + offset, data, length);
    return {static_cast<uint32_t>(offset), static_cast<uint32_t>(length)};
  }

  std::optional<std::string> encode(StageData& opaque) const {
    if (opaque.any_enabled()) {
      s_uniforms.assign(FrameUniformsSize, 0);
      build_stage_bindings(opaque, &uniform_sink);
      s_uniforms.resize((s_uniforms.size() + 3) & ~size_t{3});
      m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, s_uniforms.data(), s_uniforms.size());
    }
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    if (opaque.any_enabled()) {
      run_stage(encoder, opaque);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    return test::pop_validation_error(*m_device);
  }

  static inline std::vector<uint8_t> s_uniforms;
  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
  wgpu::Texture m_depth;
  std::shared_ptr<const SceneHints> m_hints;
  StubTexture m_hdr{.resource = resource::SceneHdr, .format = lighting::deferred::HdrFormat, .after = phase::Lighting};
};

class ExposureGraph : public ExposureDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 48, 1}})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
  }
};

TEST_F(ExposureGraph, DeclaresOnlyWithPbrAndSurface) {
  const auto graph = compile(pbr_settings());
  EXPECT_EQ(status(*graph, exposure::Pass), CullReason::None);
  EXPECT_EQ(status(*compile(Settings{}), exposure::Pass), CullReason::Skipped);
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  EXPECT_EQ(status(*compile(pbr_settings()), exposure::Pass), CullReason::Skipped);
}

TEST_F(ExposureGraph, ModeTonemapBiasKeepGraphHash) {
  const auto reference = live_graph(pbr_settings());
  const auto ids = reference->nodes[reference->find_node(exposure::Pass)].pipelineIds;
  for (const Exposure mode : {Exposure::Auto, Exposure::Manual}) {
    for (const Tonemap tm : {Tonemap::PbrNeutral, Tonemap::AgX}) {
      Settings settings = pbr_settings(mode, tm);
      settings.exposureBias = 2.f;
      const auto graph = live_graph(settings);
      EXPECT_EQ(graph->graphHash, reference->graphHash);
      EXPECT_EQ(graph->nodes[graph->find_node(exposure::Pass)].pipelineIds, ids);
    }
  }
  exposure::testing::set_target(exposure::Target::S);
  EXPECT_EQ(live_graph(pbr_settings())->graphHash, reference->graphHash);
}

TEST_F(ExposureGraph, MissingSceneHdrDisablesMeter) {
  const auto graph = live_graph(pbr_settings());
  ASSERT_TRUE(exposure::armed());
  const FrameState state{.graph = graph};
  EXPECT_TRUE(pass_enabled(*graph, build(state, 1), exposure::Pass));
  m_hdr.enabled = false;
  lighting::deferred::testing::force_enabled(false);
  EXPECT_FALSE(pass_enabled(*graph, build(state, 2), exposure::Pass));
  reset_module_state();
  exposure::initialize();
  composite::initialize();
  set_hints(camera_hints());
  const auto bare = live_graph(pbr_settings());
  EXPECT_NE(status(*bare, exposure::Pass), CullReason::None);
  EXPECT_FALSE(exposure::armed());
  EXPECT_FALSE(exposure::pipelines_live());
  EXPECT_TRUE(exposure::ready(m_settings));
}

TEST_F(ExposureGraph, DisabledAtFullLegacyBlend) {
  const auto lb0 = live_graph(pbr_settings());
  EXPECT_TRUE(exposure::armed());
  const auto lb1 = live_graph(pbr_settings(Exposure::MatchLegacy, Tonemap::PbrNeutral, 1.f));
  EXPECT_EQ(lb0->graphHash, lb1->graphHash);
  EXPECT_FALSE(exposure::armed());
  const FrameState state{.graph = lb1};
  EXPECT_FALSE(pass_enabled(*lb1, build(state, 1), exposure::Pass));
  debug::set_view_override(1);
  live_graph(pbr_settings());
  EXPECT_FALSE(exposure::armed());
}

TEST_F(ExposureGraph, RunsUnderIdentity) {
  composite::testing::set_flags(composite::FlagIdentity);
  const auto graph = live_graph(pbr_settings());
  EXPECT_TRUE(exposure::armed());
  const FrameState state{.graph = graph};
  EXPECT_TRUE(pass_enabled(*graph, build(state, 1), exposure::Pass));
}

TEST_F(ExposureGraph, StateCarriesTextureBinding) {
  const auto graph = compile(pbr_settings());
  const auto it = std::ranges::find(graph->resources, exposure::State, &CompiledResource::name);
  ASSERT_NE(it, graph->resources.end());
  EXPECT_TRUE(it->textureUsage & wgpu::TextureUsage::TextureBinding);
  EXPECT_TRUE(it->textureUsage & wgpu::TextureUsage::StorageBinding);
  EXPECT_NE(it->historySlot, InvalidResource);
}

TEST_F(ExposureGraph, AbsoluteHistoryAllocates) {
  const auto graph = live_graph(pbr_settings());
  const auto it = std::ranges::find(graph->resources, exposure::State, &CompiledResource::name);
  ASSERT_NE(it, graph->resources.end());
  EXPECT_EQ(it->extent.width, 2u);
  EXPECT_EQ(it->extent.height, 1u);
  const FrameState state{.graph = graph};
  StageData first = build(state, 1);
  EXPECT_EQ(encode(first), std::nullopt);
  StageData second = build(state, 2);
  EXPECT_TRUE(second.prev_valid(it->historySlot));
  EXPECT_EQ(encode(second), std::nullopt);
  EXPECT_TRUE(exposure::ready(m_settings));
  EXPECT_TRUE(exposure::pipelines_live());
}

TEST_F(ExposureGraph, PbrWaitsForMeterPipelines) {
  const auto graph = live_graph(pbr_settings());
  EXPECT_TRUE(exposure::pipelines_live());
  EXPECT_EQ(composite::pbr_active(), !composite::identity_forced());
  exposure::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
  composite::on_latch({.settings = &m_settings, .env = &m_env}, graph.get());
  EXPECT_FALSE(composite::pbr_active());
}

class ExposureGpu : public ExposureDeviceTest {
protected:
  void SetUp() override {
    const auto backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND"));
    if (!backend) {
      GTEST_SKIP() << "set AURORA_TEST_BACKEND to run the meter on a real backend";
    }
    if (!open({.backend = *backend, .efbSize = {exposure::GridWidth, exposure::GridHeight, 1}})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    register_pass(stub_desc(m_ao, "test.ao"));
    register_pass(readback_desc(m_state, "t.state_read"));
    register_pass(readback_desc(m_partials, "t.partials_read"));
    register_pass(readback_desc(m_words, "t.words_read"));
    for (Readback* reader : {&m_state, &m_partials, &m_words}) {
      const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                              .size = reader->size};
      reader->buffer = m_device->device.CreateBuffer(&descriptor);
    }
  }

  wgpu::Texture upload(wgpu::TextureFormat format, const std::vector<uint32_t>& words, uint32_t texelBytes = 4) const {
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    const wgpu::TextureDescriptor descriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = size,
        .format = format,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = size.width * texelBytes, .rowsPerImage = size.height};
    m_device->queue.WriteTexture(&destination, words.data(), words.size() * 4, &layout, &size);
    return texture;
  }

  static void publish(webgpu::TextureWithSampler& target, const wgpu::Texture& texture, wgpu::TextureFormat format) {
    target.texture = texture;
    target.view = texture.CreateView();
    target.size = {texture.GetWidth(), texture.GetHeight(), 1};
    target.format = format;
  }

  wgpu::Buffer staging(const void* data, size_t bytes) const {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = bytes};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(buffer, 0, data, bytes);
    return buffer;
  }

  void load(const std::vector<MeterPixel>& pixels) {
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    ASSERT_EQ(pixels.size(), size_t{size.width} * size.height);
    std::vector<uint32_t> color(pixels.size());
    std::vector<uint32_t> normal(pixels.size());
    std::vector<uint32_t> fog(pixels.size());
    std::vector<uint32_t> surface(pixels.size() * 2, 0);
    for (size_t i = 0; i < pixels.size(); ++i) {
      const MeterPixel& p = pixels[i];
      color[i] = uint32_t{p.legacy[2]} | uint32_t{p.legacy[1]} << 8 | uint32_t{p.legacy[0]} << 16 | 255u << 24;
      normal[i] = 512u | 512u << 10 | 1023u << 20 | uint32_t{p.code} << 30;
      fog[i] = uint32_t{p.fc[0]} | uint32_t{p.fc[1]} << 8 | uint32_t{p.fc[2]} << 16 | uint32_t{p.q7} << 25;
    }
    m_color = std::move(color);
    write_color();
    publish(webgpu::g_normalBuffer, upload(wgpu::TextureFormat::RGB10A2Unorm, normal),
            wgpu::TextureFormat::RGB10A2Unorm);
    publish(webgpu::g_fogRecordBuffer, upload(wgpu::TextureFormat::RGBA8Uint, fog), wgpu::TextureFormat::RGBA8Uint);
    publish(webgpu::g_surfaceBuffer, upload(webgpu::SurfaceBufferFormat, surface, 8), webgpu::SurfaceBufferFormat);
    m_hdr.rowBytes = (size.width * 8 + 255) & ~255u;
    std::vector<uint16_t> texels(size_t{m_hdr.rowBytes / 2} * size.height, 0);
    m_ao.rowBytes = (size.width * 4 + 255) & ~255u;
    std::vector<uint8_t> ao(size_t{m_ao.rowBytes} * size.height, 0);
    for (uint32_t y = 0; y < size.height; ++y) {
      for (uint32_t x = 0; x < size.width; ++x) {
        const MeterPixel& p = pixels[size_t{y} * size.width + x];
        uint16_t* t = texels.data() + size_t{y} * (m_hdr.rowBytes / 2) + size_t{x} * 4;
        for (size_t c = 0; c < 3; ++c) {
          t[c] = lighting::openpbr::to_half(p.hdr[c]);
        }
        t[3] = lighting::openpbr::to_half(p.lit ? 1.f : 0.f);
        ao[size_t{y} * m_ao.rowBytes + size_t{x} * 4] = p.ao;
      }
    }
    m_hdr.staging = staging(texels.data(), texels.size() * 2);
    m_ao.staging = staging(ao.data(), ao.size());
  }

  struct Frame {
    Texels state{};
    std::array<float, 4> words{};
    std::vector<float> partials;
  };

  // The composite writes the EFB in place, so every frame starts again from the game's colour
  void write_color() const {
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    const wgpu::TexelCopyTextureInfo destination{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = size.width * 4, .rowsPerImage = size.height};
    m_device->queue.WriteTexture(&destination, m_color.data(), m_color.size() * 4, &layout, &size);
  }

  // Production latches every frame, so one-latch flags such as the re-arm snap last exactly one frame
  Frame run(const FrameState& state, uint64_t frameId) {
    write_color();
    latch(state.graph.get());
    StageData opaque = build(state, frameId);
    EXPECT_TRUE(pass_enabled(*state.graph, opaque, exposure::Pass));
    EXPECT_EQ(encode(opaque), std::nullopt);
    Frame out;
    if (const auto words = test::read_buffer(*m_device, m_state.buffer, 0, 32)) {
      std::copy_n(words->begin(), 8, out.state.begin());
    } else {
      ADD_FAILURE() << "state readback failed";
    }
    if (const auto words = test::read_buffer(*m_device, m_words.buffer, 0, 16)) {
      for (size_t i = 0; i < 4; ++i) {
        out.words[i] = std::bit_cast<float>((*words)[i]);
      }
    }
    if (const auto words = test::read_buffer(*m_device, m_partials.buffer, 0, exposure::PartialBytes)) {
      out.partials.resize(words->size());
      std::ranges::transform(*words, out.partials.begin(), [](uint32_t w) { return std::bit_cast<float>(w); });
    }
    return out;
  }

  static float srgb_decode(float s) {
    const float x = std::clamp(s, 0.f, 1.f);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
  }

  // Built on the composite's AO-only mirror, so the meter is checked against what the composite would show
  static float legacy_y(const MeterPixel& p, bool targetS) {
    const std::array<float, 3> l{p.legacy[0] / 255.f, p.legacy[1] / 255.f, p.legacy[2] / 255.f};
    const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
    const float q = p.q7 / 127.f;
    const std::array<float, 3> sb =
        composite::testing::legacy_ao(l, fc, p.q7, targetS ? 1.f : p.ao / 255.f, p.code / 3.f);
    Rgb lin{};
    for (size_t c = 0; c < 3; ++c) {
      lin[c] = srgb_decode((sb[c] - q * fc[c]) / (1.f - q));
    }
    return std::max(tonemap::luminance(lin), exposure::YFloor);
  }

  static bool sampled(const MeterPixel& p) {
    if (!p.lit || p.q7 > exposure::FogCutQ7) {
      return false;
    }
    for (size_t c = 0; c < 3; ++c) {
      if (!std::isfinite(p.hdr[c]) || p.legacy[c] / 255.f - p.q7 / 127.f * (p.fc[c] / 255.f) < -2.f / 255.f) {
        return false;
      }
    }
    return true;
  }

  std::vector<MeterPixel> scene(uint32_t seed, float hdrScale, bool fogged) const {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> byte(20, 235);
    std::uniform_real_distribution<float> unit(0.05f, 1.f);
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    std::vector<MeterPixel> out(size_t{size.width} * size.height);
    for (MeterPixel& p : out) {
      for (size_t c = 0; c < 3; ++c) {
        p.hdr[c] = lighting::openpbr::from_half(lighting::openpbr::to_half(unit(rng) * hdrScale));
        p.legacy[c] = static_cast<uint8_t>(byte(rng));
      }
      if (fogged) {
        p.q7 = static_cast<uint8_t>(byte(rng) % 64);
        p.fc = {static_cast<uint8_t>(byte(rng) / 8), static_cast<uint8_t>(byte(rng) / 8),
                static_cast<uint8_t>(byte(rng) / 8)};
      }
      p.ao = static_cast<uint8_t>(byte(rng));
      p.code = static_cast<uint8_t>(byte(rng) % 4);
    }
    return out;
  }

  StubTexture m_ao{.resource = resource::AoFinal, .format = wgpu::TextureFormat::RGBA8Unorm, .after = phase::Ao};
  Readback m_state{.resource = exposure::State, .size = 256};
  Readback m_partials{.resource = exposure::Partials, .isBuffer = true, .size = exposure::PartialBytes};
  Readback m_words{.resource = exposure::Resource, .isBuffer = true, .size = exposure::BufferBytes};
  std::vector<uint32_t> m_color;
};

TEST_F(ExposureGpu, MeterMatchesMirror) {
  const std::vector<MeterPixel> pixels = scene(3, 0.3f, true);
  load(pixels);
  for (const Tonemap tm : {Tonemap::PbrNeutral, Tonemap::AgX}) {
    for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto}) {
      SCOPED_TRACE(std::to_string(static_cast<int>(tm)) + " " + std::to_string(static_cast<int>(mode)));
      const auto graph = live_graph(pbr_settings(mode, tm));
      const FrameState state{.graph = graph};
      const Frame first = run(state, 1);
      Params params;
      exposure::fill_params(m_settings, exposure::Target::Sb, 1000.f / 60.f, true, params);
      const Solved mirror = exposure::testing::solve(params, first.partials, {});
      const Texels texels = exposure::testing::texels(mirror);
      EXPECT_EQ(first.state[1], texels[1]);
      const float s = std::bit_cast<float>(first.state[0]);
      EXPECT_NEAR(s, mirror.s, 1e-5f);
      EXPECT_NEAR(first.words[0], exposure::testing::exp2_exact(s), 1e-5f * first.words[0]);
      EXPECT_EQ(first.words[2], 1.f);
      std::vector<Rgb> hdr;
      std::vector<float> legacy;
      for (const MeterPixel& p : pixels) {
        if (sampled(p)) {
          hdr.push_back(p.hdr);
          legacy.push_back(legacy_y(p, false));
        }
      }
      EXPECT_EQ(std::bit_cast<float>(first.state[7]), static_cast<float>(hdr.size()));
      double target = 0.0;
      double weight = 0.0;
      for (size_t i = 0, k = 0; i < pixels.size(); ++i) {
        if (sampled(pixels[i])) {
          const double w = 1.0 - pixels[i].q7 / 127.0;
          target += w * std::log2(legacy[k++]);
          weight += w;
        }
      }
      if (mode == Exposure::MatchLegacy) {
        EXPECT_NEAR(std::bit_cast<float>(first.state[4]), target / weight, 2e-4);
      }
      EXPECT_NEAR(std::bit_cast<float>(first.state[3]), weight, weight * 1e-5);
      const Frame second = run(state, 2);
      EXPECT_EQ(second.state[1] & exposure::StateSnapped, 0u);
      Texels prev = first.state;
      exposure::fill_params(m_settings, exposure::Target::Sb, 1000.f / 60.f, false, params);
      const Solved next = exposure::testing::solve(params, second.partials, prev);
      EXPECT_NEAR(std::bit_cast<float>(second.state[0]), next.s, 1e-5f);
      EXPECT_NEAR(second.words[2], std::exp2(next.s - s), 1e-5f);
    }
  }
}

TEST_F(ExposureGpu, SkipsClassZeroFoggedAndBad) {
  std::vector<MeterPixel> pixels = scene(5, 0.5f, false);
  size_t want = 0;
  for (size_t i = 0; i < pixels.size(); ++i) {
    MeterPixel& p = pixels[i];
    switch (i % 5) {
    case 0:
      p.lit = false;
      break;
    case 1:
      p.q7 = 115;
      break;
    case 2:
      p.q7 = 127;
      p.fc = {250, 0, 0};
      p.legacy = {10, 100, 100};
      break;
    case 3:
      p.q7 = 114;
      ++want;
      break;
    default:
      ++want;
      break;
    }
  }
  load(pixels);
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame frame = run(state, 1);
  EXPECT_EQ(std::bit_cast<float>(frame.state[7]), static_cast<float>(want));
  EXPECT_EQ(frame.state[1] & exposure::StateHeld, 0u);
}

TEST_F(ExposureGpu, ResetSnapsToSolve) {
  load(scene(7, 0.2f, false));
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame first = run(state, 1);
  EXPECT_NE(first.state[1] & exposure::StateSnapped, 0u);
  EXPECT_EQ(first.state[0], first.state[2]);
  load(scene(8, 2.f, false));
  const Frame smooth = run(state, 2);
  EXPECT_EQ(smooth.state[1] & exposure::StateSnapped, 0u);
  EXPECT_NE(smooth.state[0], smooth.state[2]);
  set_hints(camera_hints(false));
  latch(graph.get());
  const Frame reset = run(state, 3);
  EXPECT_NE(reset.state[1] & exposure::StateSnapped, 0u);
  EXPECT_EQ(reset.state[0], reset.state[2]);
  EXPECT_EQ(reset.words[2], 1.f);
  set_hints(camera_hints());
  m_settings.tonemap = Tonemap::AgX;
  latch(graph.get());
  const Frame changed = run(state, 4);
  EXPECT_NE(changed.state[1] & exposure::StateSnapped, 0u);
  EXPECT_NE(changed.state[1] & exposure::StateAgx, 0u);
}

TEST_F(ExposureGpu, HeldResetSnapsTheNextMeasuredFrame) {
  // Only the top row is PBR, which the grid samples 192 times, under the hold count
  std::vector<MeterPixel> pixels = scene(13, 0.3f, false);
  for (size_t i = exposure::GridWidth; i < pixels.size(); ++i) {
    pixels[i].lit = false;
  }
  load(pixels);
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame few = run(state, 1);
  ASSERT_EQ(std::bit_cast<float>(few.state[7]), static_cast<float>(exposure::GridWidth));
  EXPECT_EQ(few.state[1] & (exposure::StateHeld | exposure::StateSnapped),
            uint32_t{exposure::StateHeld | exposure::StateSnapped});
  EXPECT_EQ(few.state[0], few.state[2]);
  EXPECT_NE(std::bit_cast<float>(few.state[0]), 0.f);
  load(scene(14, 2.f, false));
  const Frame measured = run(state, 2);
  EXPECT_EQ(measured.state[1] & exposure::StateHeld, 0u);
  EXPECT_NE(measured.state[1] & exposure::StateSnapped, 0u);
  EXPECT_EQ(measured.state[0], measured.state[2]);
  EXPECT_EQ(measured.words[2], 1.f);
}

TEST_F(ExposureGpu, RepeatIsBitIdentical) {
  load(scene(9, 0.4f, true));
  std::vector<Texels> runs;
  for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto, Exposure::Manual}) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto graph = live_graph(pbr_settings(mode));
      const FrameState state{.graph = graph};
      Texels last{};
      for (uint64_t frame = 1; frame <= 3; ++frame) {
        // A replay restarts its hints, so each run's first frame resets the meter like gxreplay's --repeat
        set_hints(camera_hints(frame != 1));
        last = run(state, static_cast<uint64_t>(repeat) * 100 + frame + static_cast<uint64_t>(mode) * 1000).state;
      }
      runs.push_back(last);
    }
    EXPECT_EQ(runs[runs.size() - 2], runs.back()) << static_cast<int>(mode);
  }
}

TEST_F(ExposureGpu, EncodeValidationClean) {
  load(scene(10, 0.4f, true));
  for (const Exposure mode : {Exposure::MatchLegacy, Exposure::Auto, Exposure::Manual}) {
    const auto graph = live_graph(pbr_settings(mode, Tonemap::AgX));
    const FrameState state{.graph = graph};
    run(state, 1);
    run(state, 2);
  }
  EXPECT_GE(exposure::testing::counters().runs, 6u);
  EXPECT_EQ(exposure::testing::counters().manual, 2u);
}

TEST_F(ExposureGpu, PrevBindsWithoutCapture) {
  load(scene(11, 0.3f, false));
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame first = run(state, 1);
  const Frame second = run(state, 2);
  EXPECT_EQ(second.state[1] & exposure::StateSnapped, 0u);
  EXPECT_EQ(exposure::testing::counters().snaps, 1u);
  EXPECT_NE(first.state[1] & exposure::StateValid, 0u);
}

TEST_F(ExposureGpu, AoDummyIsIdentity) {
  std::vector<MeterPixel> pixels = scene(12, 0.3f, false);
  for (MeterPixel& p : pixels) {
    p.ao = 64;
    p.code = 3;
  }
  load(pixels);
  m_ao.enabled = false;
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame sb = run(state, 1);
  exposure::testing::set_target(exposure::Target::S);
  latch(graph.get());
  const Frame s = run(state, 5);
  EXPECT_EQ(sb.state[4], s.state[4]);
}

TEST_F(ExposureGpu, SbTargetAppliesAoOnReceiversOnly) {
  std::vector<MeterPixel> pixels = scene(13, 0.3f, false);
  for (MeterPixel& p : pixels) {
    p.ao = 64;
    p.code = 0;
  }
  load(pixels);
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame none = run(state, 1);
  for (MeterPixel& p : pixels) {
    p.code = 2;
  }
  load(pixels);
  const Frame receivers = run(state, 3);
  exposure::testing::set_target(exposure::Target::S);
  latch(graph.get());
  const Frame target = run(state, 5);
  EXPECT_EQ(none.state[4], target.state[4]);
  EXPECT_LT(std::bit_cast<float>(receivers.state[4]), std::bit_cast<float>(none.state[4]) - 0.5f);
}

TEST_F(ExposureGpu, NormalDummyMeansNoReceiver) {
  std::vector<MeterPixel> pixels = scene(14, 0.3f, false);
  for (MeterPixel& p : pixels) {
    p.ao = 32;
    p.code = 0;
  }
  load(pixels);
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame frame = run(state, 1);
  double target = 0.0;
  double weight = 0.0;
  for (const MeterPixel& p : pixels) {
    target += std::log2(legacy_y(p, true));
    weight += 1.0;
  }
  EXPECT_NEAR(std::bit_cast<float>(frame.state[4]), target / weight, 2e-4);
}

TEST_F(ExposureGpu, ManualIgnoresPartials) {
  load(scene(15, 0.3f, true));
  const auto match = live_graph(pbr_settings());
  const FrameState matchState{.graph = match};
  run(matchState, 1);
  Settings manual = pbr_settings(Exposure::Manual);
  manual.exposureBias = -1.f;
  const auto graph = live_graph(manual);
  const FrameState state{.graph = graph};
  const Frame frame = run(state, 3);
  Params params;
  exposure::fill_params(manual, exposure::Target::Sb, 16.f, false, params);
  const Texels want = exposure::testing::texels(exposure::testing::solve(params, {}, {}));
  EXPECT_EQ(frame.state[0], want[0]);
  EXPECT_EQ(frame.state[3], 0u);
  EXPECT_EQ(frame.state[4], 0u);
  EXPECT_EQ(frame.state[7], 0u);
  const float s = exposure::ManualAnchorEv - 1.f;
  EXPECT_EQ(std::bit_cast<float>(frame.state[0]), s);
  EXPECT_NEAR(frame.words[0], exposure::testing::exp2_exact(s), 1e-6f * frame.words[0]);
  EXPECT_NEAR(frame.words[1], exposure::testing::exp2_exact(-s), 1e-6f);
}

TEST_F(ExposureGpu, MeterTargetMatchesCompositeSb) {
  const std::vector<MeterPixel> pixels = scene(16, 0.3f, true);
  load(pixels);
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  const Frame frame = run(state, 1);
  double target = 0.0;
  double weight = 0.0;
  for (const MeterPixel& p : pixels) {
    if (sampled(p)) {
      const double w = 1.0 - p.q7 / 127.0;
      target += w * std::log2(legacy_y(p, false));
      weight += w;
    }
  }
  EXPECT_NEAR(std::bit_cast<float>(frame.state[4]), target / weight, 2e-4);
}

TEST_F(ExposureGpu, MeterTargetMatchesCompositeSbWithSsil) {
  const std::vector<MeterPixel> pixels = scene(17, 0.3f, true);
  load(pixels);
  const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
  std::mt19937 rng{0x55E0u};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  std::vector<std::array<float, 4>> irr(pixels.size());
  StubTexture ssil{
      .resource = resource::SsilIrradiance, .format = wgpu::TextureFormat::RGBA16Float, .after = phase::Ao};
  ssil.rowBytes = (size.width * 8 + 255) & ~255u;
  std::vector<uint16_t> texels(size_t{ssil.rowBytes / 2} * size.height, 0);
  for (size_t i = 0; i < irr.size(); ++i) {
    irr[i] = {0.6f * unit(rng), 0.4f * unit(rng), 0.2f * unit(rng), unit(rng)};
    for (size_t c = 0; c < 4; ++c) {
      texels[(i / size.width) * (ssil.rowBytes / 2) + (i % size.width) * 4 + c] = lighting::openpbr::to_half(irr[i][c]);
    }
  }
  ssil.staging = staging(texels.data(), texels.size() * 2);
  register_pass(stub_desc(ssil, "test.ssil"));
  composite::testing::set_pbr_producers_override(true);
  Settings settings = pbr_settings();
  settings.ao = Quality::High;
  const auto graph = live_graph(settings);
  ASSERT_TRUE(composite::pbr_active());
  const FrameState state{.graph = graph};
  const Frame first = run(state, 1);
  const Frame second = run(state, 2);
  // The meter reads last frame's exposure, so the second frame's bounce carries the first frame's e
  for (const auto& [frame, e] :
       {std::pair{&first, 1.f},
        std::pair{&second, exposure::testing::exp2_exact(std::bit_cast<float>(first.state[0]))}}) {
    double target = 0.0;
    double weight = 0.0;
    size_t bounced = 0;
    for (size_t i = 0; i < pixels.size(); ++i) {
      const MeterPixel& p = pixels[i];
      if (!sampled(p)) {
        continue;
      }
      const std::array<float, 3> l{p.legacy[0] / 255.f, p.legacy[1] / 255.f, p.legacy[2] / 255.f};
      const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
      const float q = p.q7 / 127.f;
      std::array<float, 3> bounce{};
      for (size_t c = 0; c < 3; ++c) {
        bounce[c] = ao::legacy_albedo() * e * lighting::openpbr::from_half(lighting::openpbr::to_half(irr[i][c]));
      }
      const std::array<float, 3> sb = composite::testing::legacy_ssil(l, fc, p.q7, p.ao / 255.f, p.code / 3.f, bounce);
      bounced += sb != composite::testing::legacy_ao(l, fc, p.q7, p.ao / 255.f, p.code / 3.f) ? 1 : 0;
      Rgb lin{};
      for (size_t c = 0; c < 3; ++c) {
        lin[c] = srgb_decode((sb[c] - q * fc[c]) / (1.f - q));
      }
      const double w = 1.0 - q;
      target += w * std::log2(std::max(tonemap::luminance(lin), exposure::YFloor));
      weight += w;
    }
    EXPECT_GT(bounced, pixels.size() / 4);
    EXPECT_NEAR(std::bit_cast<float>(frame->state[4]), target / weight, 2e-4) << e;
  }
  EXPECT_NE(std::bit_cast<float>(first.state[0]), 0.f);
}
} // namespace
} // namespace aurora::render::detail

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
#include "render/graph_layout.hpp"
#include "render/graph_pool.hpp"
#include "render/gx_ext/surface_table.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/lighting/clusters.hpp"
#include "render/lighting/deferred.hpp"
#include "render/lighting/openpbr.hpp"
#include "render/lighting/sky_probe.hpp"
#include "render/material/classes.hpp"
#include "render/material/registry.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/post/composite.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/shadows/sun_visibility.hpp"
#include "render/stage.hpp"
#include "render/stage_camera.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace clusters = lighting::clusters;
namespace deferred = lighting::deferred;
namespace openpbr = lighting::openpbr;
namespace classes = material::classes;
using aurora::test::scene_depth;
using Vec3 = openpbr::Vec3;

constexpr double Pi = std::numbers::pi;
constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float Near = 10.f;
constexpr float Far = 100000.f;

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 normalized(const Vec3& a) { return scale(a, 1.0 / std::sqrt(dot(a, a))); }

struct Frame {
  Vec3 t, b, n;
};

Frame frame_of(const Vec3& n) {
  const Vec3 helper = std::abs(n[0]) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
  const Vec3 t = normalized(cross(helper, n));
  return {t, cross(n, t), n};
}

Vec3 view_at(const Vec3& n, double muV) {
  const Frame f = frame_of(n);
  return normalized(add(scale(n, muV), scale(f.t, std::sqrt(std::max(0.0, 1.0 - muV * muV)))));
}

// Midpoint rule over the hemisphere, or the sphere for wrapped and transmitting rows
Vec3 integrate(const openpbr::Prepared& p, int steps, bool sphere) {
  const openpbr::Tables& tables = openpbr::testing::tables();
  const Frame f = frame_of(p.n);
  const double upper = sphere ? Pi : Pi / 2.0;
  const double dt = upper / steps;
  const double dp = 2.0 * Pi / (2 * steps);
  Vec3 sum{};
  for (int i = 0; i < steps; ++i) {
    const double theta = (i + 0.5) * dt;
    for (int j = 0; j < 2 * steps; ++j) {
      const double phi = (j + 0.5) * dp;
      const Vec3 l = add(add(scale(f.t, std::sin(theta) * std::cos(phi)), scale(f.b, std::sin(theta) * std::sin(phi))),
                         scale(f.n, std::cos(theta)));
      sum = add(sum, scale(openpbr::eval_light(tables, p, l), std::sin(theta) * dt * dp));
    }
  }
  return scale(sum, 1.0 / Pi);
}

classes::GpuRow default_row() { return classes::gpu_row(OpenPbrParams{}); }

classes::GpuRow built_in_row(std::string_view name) {
  for (const classes::BuiltIn& entry : classes::built_ins()) {
    if (entry.name == name) {
      return classes::gpu_row(entry.params);
    }
  }
  ADD_FAILURE() << "no built-in class " << name;
  return default_row();
}

OpenPbrParams coat_params() {
  OpenPbrParams params;
  params.coatWeight = 1.f;
  params.coatRoughness = 0.2f;
  params.coatColor = {0.9f, 0.8f, 0.7f};
  return params;
}

OpenPbrParams fuzz_params() {
  OpenPbrParams params;
  params.fuzzWeight = 0.8f;
  params.fuzzRoughness = 0.6f;
  params.fuzzColor = {0.9f, 0.9f, 1.f};
  return params;
}

OpenPbrParams thin_params() {
  OpenPbrParams params;
  params.geometryThinWalled = true;
  params.thinWalledWrap = 0.3f;
  params.subsurfaceWeight = 0.6f;
  params.subsurfaceColor = {0.6f, 0.9f, 0.4f};
  params.subsurfaceScatterAnisotropy = 0.2f;
  return params;
}

openpbr::Prepared prepare(const classes::GpuRow& row, const openpbr::Surface& surface, const Vec3& n, const Vec3& v) {
  return openpbr::prepare(openpbr::testing::tables(), row, surface, n, v);
}

uint16_t half(double value) { return openpbr::to_half(static_cast<float>(value)); }

double f16_ulp(double value) {
  const uint16_t bits = half(std::abs(value));
  return openpbr::from_half(static_cast<uint16_t>(bits + 1)) - openpbr::from_half(bits);
}

// Pinned openpbr_ltc_data.h rows (fuzz alpha) and columns (cos theta), written as the source prints them
struct LtcGolden {
  uint32_t row;
  uint32_t column;
  std::array<double, 3> value;
};
constexpr std::array<LtcGolden, 6> LtcGoldens{{
    {31, 31, {0.87958, 0.00003, 0.34187}},
    {31, 0, {0.72363, 0.00000, 0.74973}},
    {10, 20, {0.33479, -0.03417, 0.01041}},
    {20, 10, {0.56541, -0.10297, 0.39979}},
    {5, 27, {0.29160, -0.03604, 0.00003}},
    {27, 5, {0.71943, -0.06986, 0.62480}},
}};

TEST(OpenPbrTables, Shapes) {
  const openpbr::Tables& t = openpbr::testing::tables();
  EXPECT_EQ(t.lut3d.size(), openpbr::Lut3dTexels * 2);
  EXPECT_EQ(t.lut2d.size(), openpbr::Lut2dTexels * 4);
  const TexDesc volume = openpbr::lut3d_desc();
  EXPECT_EQ(volume.format, wgpu::TextureFormat::RG16Float);
  EXPECT_EQ(volume.dimension, wgpu::TextureDimension::e3D);
  EXPECT_EQ(volume.sizeClass, SizeClass::Absolute);
  EXPECT_EQ(volume.width, 32u);
  EXPECT_EQ(volume.height, 32u);
  EXPECT_EQ(volume.depthOrLayers, 32u);
  const TexDesc layers = openpbr::lut2d_desc();
  EXPECT_EQ(layers.format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_EQ(layers.dimension, wgpu::TextureDimension::e2D);
  EXPECT_EQ(layers.depthOrLayers, 3u);
  EXPECT_EQ(openpbr::Classes, "light.classes");
  EXPECT_EQ(openpbr::Lut3d, "light.lut3d");
  EXPECT_EQ(openpbr::Lut2d, "light.lut2d");
  EXPECT_EQ(openpbr::to_half(1.f), 0x3C00u);
  EXPECT_EQ(openpbr::to_half(0.5f), 0x3800u);
  EXPECT_EQ(openpbr::to_half(65504.f), 0x7BFFu);
  EXPECT_EQ(openpbr::to_half(65520.f), 0x7C00u);
  EXPECT_EQ(openpbr::to_half(1.f + 1.f / 2048.f), 0x3C00u);
  EXPECT_EQ(openpbr::to_half(1.f + 3.f / 2048.f), 0x3C02u);
  EXPECT_EQ(openpbr::to_half(std::ldexp(1.f, -24)), 0x0001u);
  EXPECT_EQ(openpbr::from_half(0x3555u), std::bit_cast<float>(0x3EAAA000u));
  for (size_t i = 0; i < openpbr::Lut3dTexels; i += 997) {
    EXPECT_EQ(t.lut3d[i * 2], openpbr::to_half(openpbr::data::IdealDielectricEc[i] / 65535.f)) << i;
    EXPECT_EQ(t.lut3d[i * 2 + 1], openpbr::to_half(openpbr::data::OpaqueDielectricEc[i] / 65535.f)) << i;
  }
  for (uint32_t x = 0; x < 32; ++x) {
    const uint16_t metal = openpbr::to_half(openpbr::data::IdealMetalAverageEc[x] / 65535.f);
    EXPECT_EQ(t.lut2d[(size_t{7} * 32 + x) * 4 + 3], metal) << x;
  }
}

TEST(OpenPbrTables, LtcAxisMatchesSource) {
  const openpbr::Tables& t = openpbr::testing::tables();
  for (const LtcGolden& golden : LtcGoldens) {
    const std::array<double, 4> got = openpbr::sample_lut2d(t, golden.column, golden.row, openpbr::LayerLtc);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_LE(std::abs(got[c] - golden.value[c]), f16_ulp(golden.value[c]) + 1e-5)
          << golden.row << "," << golden.column << " channel " << c;
    }
  }
  const double grazing = openpbr::sample_lut2d(t, 0.0, 31.0, openpbr::LayerLtc)[2];
  const double normal = openpbr::sample_lut2d(t, 31.0, 31.0, openpbr::LayerLtc)[2];
  EXPECT_GT(std::abs(grazing - normal), 0.3);
}

double separable_albedo(double mu, double alpha, double eta, bool metal, int steps) {
  const Vec3 n{0.0, 0.0, 1.0};
  const Vec3 v = view_at(n, mu);
  const double a2 = alpha * alpha;
  const auto g1 = [&](double c) { return 2.0 * c / (c + std::sqrt(a2 + (1.0 - a2) * c * c)); };
  double sum = 0.0;
  const double dt = Pi / 2.0 / steps;
  const double dp = 2.0 * Pi / (2 * steps);
  for (int i = 0; i < steps; ++i) {
    const double theta = (i + 0.5) * dt;
    for (int j = 0; j < 2 * steps; ++j) {
      const double phi = (j + 0.5) * dp;
      const Vec3 l{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
      const Vec3 h = normalized(add(v, l));
      const double nh = h[2];
      const double d = a2 / (Pi * std::pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
      double f = 1.0;
      if (!metal) {
        const double c = std::clamp(dot(v, h), 0.0, 1.0);
        const double ct = std::sqrt(1.0 - (1.0 - c * c) / (eta * eta));
        f = 0.5 * (std::pow((c - eta * ct) / (c + eta * ct), 2.0) + std::pow((ct - eta * c) / (ct + eta * c), 2.0));
      }
      sum += d * g1(mu) * g1(l[2]) * f / (4.0 * mu) * std::sin(theta) * dt * dp;
    }
  }
  return sum;
}

TEST(OpenPbrTables, IdealMetalMatchesIntegration) {
  const openpbr::Tables& t = openpbr::testing::tables();
  for (const uint32_t j : {16u, 24u, 31u}) {
    for (const uint32_t i : {6u, 16u, 28u}) {
      const double roughness = j / 31.0;
      const double mu = i / 31.0;
      const double ec = openpbr::sample_lut2d(t, i, j, openpbr::LayerMetal)[0];
      EXPECT_NEAR(1.0 - ec, separable_albedo(mu, roughness * roughness, 1.5, true, 160), 2e-3) << j << "," << i;
    }
  }
}

std::array<double, 3> dfg_integral(double mu, double alpha, int steps) {
  const Vec3 n{0.0, 0.0, 1.0};
  const Vec3 v = view_at(n, mu);
  const double a2 = alpha * alpha;
  const auto g1 = [&](double c) { return 2.0 * c / (c + std::sqrt(a2 + (1.0 - a2) * c * c)); };
  std::array<double, 3> sum{};
  const double dt = Pi / 2.0 / steps;
  const double dp = 2.0 * Pi / (2 * steps);
  for (int i = 0; i < steps; ++i) {
    const double theta = (i + 0.5) * dt;
    for (int j = 0; j < 2 * steps; ++j) {
      const double phi = (j + 0.5) * dp;
      const Vec3 l{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
      const Vec3 h = normalized(add(v, l));
      const double nh = h[2];
      const double d = a2 / (Pi * std::pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
      const double base = d * g1(mu) * g1(l[2]) / (4.0 * mu) * std::sin(theta) * dt * dp;
      const double vh = std::clamp(dot(v, h), 0.0, 1.0);
      const double m = 1.0 - vh;
      const double m5 = std::pow(m, 5.0);
      sum[0] += (1.0 - m5) * base;
      sum[1] += m5 * base;
      sum[2] += vh * m5 * m * base;
    }
  }
  return sum;
}

TEST(OpenPbrTables, DfgMatchesIntegration) {
  const openpbr::Tables& t = openpbr::testing::tables();
  for (const uint32_t j : {8u, 12u, 16u, 24u, 31u}) {
    for (const uint32_t i : {4u, 12u, 20u, 31u}) {
      const double root = j / 31.0;
      const double mu = i / 31.0;
      const std::array<double, 4> texel = openpbr::sample_lut2d(t, i, j, openpbr::LayerMetal);
      // The grid integral cannot resolve the sharpest lobe at grazing view; the EC table checks that texel instead
      if (j > 8) {
        const std::array<double, 3> want = dfg_integral(mu, root * root, 160);
        for (size_t c = 0; c < 3; ++c) {
          EXPECT_NEAR(texel[c + 1], want[c], 4e-3 + 1e-2 * want[c]) << j << "," << i << " term " << c;
        }
      }
      EXPECT_NEAR(texel[1] + texel[2], 1.0 - texel[0], 1e-2) << j << "," << i;
    }
  }
  const std::array<double, 3> mirror = openpbr::dfg(16.0 / 31.0, std::pow(16.0 / 31.0, 2.0));
  const std::array<double, 4> texel = openpbr::sample_lut2d(t, 16.0, 16.0, openpbr::LayerMetal);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(texel[c + 1], mirror[c], 1e-3) << c;
  }
}

TEST(OpenPbrTables, OpaqueDielectricIorAxis) {
  const openpbr::Tables& t = openpbr::testing::tables();
  EXPECT_DOUBLE_EQ(openpbr::ior_index(1.0), 16.0);
  EXPECT_DOUBLE_EQ(openpbr::ior_index(2.5), 31.0);
  EXPECT_NEAR(openpbr::ior_index(1.0 / 2.5), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(openpbr::ior_index(1.5), 21.0);
  for (const uint32_t j : {8u, 20u, 31u}) {
    for (const uint32_t i : {4u, 16u, 30u}) {
      EXPECT_NEAR(openpbr::sample_lut3d(t, i, j, 16.0)[1], 1.0, 1e-3) << j << "," << i;
      double previous = 2.0;
      for (const double ior : {1.2, 1.5, 2.0, 2.5}) {
        const double ec = openpbr::sample_lut3d(t, i, j, openpbr::ior_index(ior))[1];
        EXPECT_LT(ec, previous) << ior;
        previous = ec;
      }
    }
  }
  for (const uint32_t j : {16u, 24u, 31u}) {
    for (const uint32_t i : {6u, 16u, 28u}) {
      const double roughness = j / 31.0;
      const double ec = openpbr::sample_lut3d(t, i, j, 21.0)[1];
      EXPECT_NEAR(1.0 - ec, separable_albedo(i / 31.0, roughness * roughness, 1.5, false, 160), 1e-3) << j << "," << i;
    }
  }
}

struct Config {
  std::string name;
  classes::GpuRow row;
  double metal = 0.0;
  bool sphere = false;
};

std::vector<Config> white_configs() {
  return {
      {"default", default_row()},
      {"ground/grass", built_in_row("ground/grass")},
      {"ground/snow", built_in_row("ground/snow")},
      {"water/edge", built_in_row("water/edge")},
      {"foliage/grass", built_in_row("foliage/grass"), 0.0, true},
      {"default/cutout", built_in_row("default/cutout"), 0.0, true},
      {"metal", default_row(), 1.0},
  };
}

TEST(OpenPbrFurnace, WhiteConfigsConserve) {
  double worst = 0.0;
  for (const Config& config : white_configs()) {
    for (const double roughness : {0.5, 0.7, 0.85, 1.0}) {
      for (const double muV : {0.2, 0.5, 0.9}) {
        SCOPED_TRACE(config.name + " r " + std::to_string(roughness) + " mu " + std::to_string(muV));
        const Vec3 n = normalized({0.2, -0.3, 0.9});
        const openpbr::Prepared p =
            prepare(config.row, {.roughness = roughness, .metalness = config.metal}, n, view_at(n, muV));
        const Vec3 albedo = openpbr::directional_albedo(p);
        const Vec3 direct = integrate(p, 128, config.sphere);
        for (size_t c = 0; c < 3; ++c) {
          EXPECT_NEAR(albedo[c], 1.0, 1e-9);
          EXPECT_NEAR(direct[c], 1.0, 0.015);
          worst = std::max(worst, std::abs(direct[c] - 1.0));
        }
      }
    }
  }
  RecordProperty("worst_direct", std::to_string(worst));
  std::printf("white furnace integrated direct: worst %.6f\n", worst);
}

TEST(OpenPbrFurnace, NeverGains) {
  std::mt19937 rng{0xF0A9u};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const std::array<classes::GpuRow, 7> rows{default_row(),
                                            built_in_row("ground/rock"),
                                            built_in_row("foliage/grass"),
                                            classes::gpu_row(coat_params()),
                                            classes::gpu_row(fuzz_params()),
                                            classes::gpu_row(thin_params()),
                                            built_in_row("default/cutout")};
  for (int i = 0; i < 48; ++i) {
    const size_t which = static_cast<size_t>(i) % rows.size();
    const openpbr::Surface surface{.baseColor = {unit(rng), unit(rng), unit(rng)},
                                   .roughness = 0.45 + 0.55 * unit(rng),
                                   .metalness = i % 3 == 0 ? unit(rng) : 0.0};
    const Vec3 n = normalized({unit(rng) - 0.5, unit(rng) - 0.5, 1.0});
    const openpbr::Prepared p = prepare(rows[which], surface, n, view_at(n, 0.1 + 0.9 * unit(rng)));
    const Vec3 albedo = openpbr::directional_albedo(p);
    const Vec3 direct = integrate(p, 96, which == 2 || which >= 5);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_LE(albedo[c], 1.0 + 1e-9) << i;
      EXPECT_LE(direct[c], 1.015) << i;
    }
  }
}

TEST(OpenPbrMirror, ThinCutoutTransmitsItsBaseColour) {
  const classes::GpuRow row = built_in_row("default/cutout");
  const double w = row.v[15];
  ASSERT_GT(w, 0.0);
  EXPECT_EQ(std::bit_cast<uint32_t>(row.v[30]), classes::RowThin | classes::RowTwoSided | classes::RowSubsurfaceTint);
  const Vec3 base{0.3, 0.7, 0.2};
  const Vec3 n = normalized({0.1, 0.2, 1.0});
  const openpbr::Prepared p = prepare(row, {.baseColor = base, .roughness = 0.85}, n, view_at(n, 0.8));
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(p.transmit[c], base[c] * w * 0.5, 1e-12) << c;
    EXPECT_NEAR(p.diffuse[c], base[c] * (1.0 - w * 0.5), 1e-12) << c;
  }
  // A light behind the card comes through in the card's own colour; the default row stays dark there
  const openpbr::Tables& tables = openpbr::testing::tables();
  const Vec3 behind = normalized(add(scale(n, -1.0), {0.2, -0.1, 0.0}));
  const Vec3 through = openpbr::eval_light(tables, p, behind);
  const Vec3 opaque = openpbr::eval_light(
      tables, prepare(default_row(), {.baseColor = base, .roughness = 0.85}, n, view_at(n, 0.8)), behind);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_GT(through[c], 0.0) << c;
    EXPECT_NEAR(through[c] / base[c], through[0] / base[0], 1e-12) << c;
    EXPECT_EQ(opaque[c], 0.0) << c;
  }
}

TEST(OpenPbrMirror, ThinTransmissionFollowsDiffuseRoughness) {
  OpenPbrParams params = thin_params();
  params.specularWeight = 0.f;
  params.thinWalledWrap = 0.f;
  params.subsurfaceScatterAnisotropy = 0.f;
  params.baseDiffuseRoughness = 0.8f;
  const openpbr::Tables& tables = openpbr::testing::tables();
  const Vec3 n{0.0, 0.0, 1.0};
  // A black base gives both sides the same albedo, so the back lobe must be the front lobe mirrored
  const openpbr::Prepared p =
      prepare(classes::gpu_row(params), {.baseColor = {0.0, 0.0, 0.0}, .roughness = 0.85}, n, view_at(n, 0.6));
  ASSERT_EQ(p.diffuse, p.transmit);
  std::mt19937 rng{0x7B1Eu};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  double offLambert = 0.0;
  for (int i = 0; i < 64; ++i) {
    const Vec3 back = normalized({unit(rng) - 0.5, unit(rng) - 0.5, -0.05 - unit(rng)});
    const Vec3 through = openpbr::eval_light(tables, p, back, false);
    const Vec3 mirrored = openpbr::eval_light(tables, p, {back[0], back[1], -back[2]}, false);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(through[c], mirrored[c], 1e-12) << i << " " << c;
      offLambert = std::max(offLambert, std::abs(through[c] - p.transmit[c] * p.ecV * -back[2]));
    }
  }
  EXPECT_GT(offLambert, 1e-3);
  const Vec3 ambient = openpbr::eval_ambient(p, {Pi, Pi, Pi}, {0.0, 0.0, 0.0});
  const Vec3 direct = integrate(p, 160, true);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(ambient[c], direct[c], 3e-3) << c;
  }
}

TEST(OpenPbrMirror, CoatTakesTheBaseIorRelativeToIt) {
  const Vec3 n{0.0, 0.0, 1.0};
  const auto eta = [&](float coatWeight, float coatIor, float specularWeight = 1.f) {
    OpenPbrParams params = coat_params();
    params.coatWeight = coatWeight;
    params.coatIor = coatIor;
    params.specularWeight = specularWeight;
    return prepare(classes::gpu_row(params), {}, n, view_at(n, 0.7)).eta;
  };
  EXPECT_NEAR(eta(0.f, 1.6f), 1.5, 1e-6);
  EXPECT_NEAR(eta(1.f, 1.5f), 1.0, 1e-6);
  EXPECT_NEAR(eta(1.f, 1.3f), 1.5 / 1.3, 1e-6);
  EXPECT_NEAR(eta(1.f, 1.6f), 1.6 / 1.5, 1e-6);
  EXPECT_NEAR(eta(0.5f, 2.f), 1.0, 1e-6);
  EXPECT_NEAR(eta(1.f, 1.6f, 0.f), 1.0, 1e-6);
}

TEST(OpenPbrMirror, LambertAtZeroDiffuseRoughness) {
  OpenPbrParams params;
  params.specularWeight = 0.f;
  const classes::GpuRow row = classes::gpu_row(params);
  const openpbr::Tables& tables = openpbr::testing::tables();
  std::mt19937 rng{0x1A3Bu};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int i = 0; i < 200; ++i) {
    const Vec3 n = normalized({unit(rng) - 0.5, unit(rng) - 0.5, 1.0});
    const Vec3 v = view_at(n, 0.05 + 0.95 * unit(rng));
    const Vec3 l = view_at(normalized({unit(rng) - 0.5, unit(rng) - 0.5, 1.0}), 0.3 + 0.7 * unit(rng));
    const Vec3 rho{unit(rng), unit(rng), unit(rng)};
    const openpbr::Prepared p = prepare(row, {.baseColor = rho, .roughness = unit(rng)}, n, v);
    const Vec3 f = openpbr::eval_light(tables, p, l);
    const double nl = std::max(dot(n, l), 0.0);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(f[c], rho[c] * nl, 1e-4) << i;
    }
  }
}

TEST(OpenPbrMirror, AmbientMatchesIntegratedDirect) {
  OpenPbrParams tinted;
  tinted.specularColor = {0.9f, 0.6f, 0.3f};
  std::vector<Config> configs = white_configs();
  configs.push_back({"metal/f82", classes::gpu_row(tinted), 1.0});
  double worst = 0.0;
  for (const Config& config : configs) {
    if (config.sphere) {
      continue;
    }
    for (const double roughness : {0.6, 0.85, 1.0}) {
      for (const double muV : {0.3, 0.7}) {
        SCOPED_TRACE(config.name + " r " + std::to_string(roughness) + " mu " + std::to_string(muV));
        const Vec3 n{0.0, 0.0, 1.0};
        const openpbr::Prepared p =
            prepare(config.row, {.baseColor = {0.8, 0.5, 0.2}, .roughness = roughness, .metalness = config.metal}, n,
                    view_at(n, muV));
        const Vec3 ambient = openpbr::eval_ambient(p, {Pi, Pi, Pi}, {1.0, 1.0, 1.0});
        const Vec3 direct = integrate(p, 160, false);
        for (size_t c = 0; c < 3; ++c) {
          EXPECT_NEAR(ambient[c], direct[c], 3e-3) << c;
          worst = std::max(worst, std::abs(ambient[c] - direct[c]));
        }
      }
    }
  }
  RecordProperty("worst", std::to_string(worst));
  std::printf("ambient against integrated direct: worst %.6f\n", worst);
}

TEST(OpenPbrMirror, ReciprocalWhereExpected) {
  const openpbr::Tables& tables = openpbr::testing::tables();
  std::mt19937 rng{0x7EC1u};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (const Config& config : white_configs()) {
    if (config.sphere) {
      continue;
    }
    for (int i = 0; i < 40; ++i) {
      const Vec3 n{0.0, 0.0, 1.0};
      const Vec3 v = normalized({unit(rng) - 0.5, unit(rng) - 0.5, 0.2 + unit(rng)});
      const Vec3 l = normalized({unit(rng) - 0.5, unit(rng) - 0.5, 0.2 + unit(rng)});
      const openpbr::Surface surface{.baseColor = {unit(rng), unit(rng), unit(rng)},
                                     .roughness = 0.4 + 0.6 * unit(rng),
                                     .metalness = config.metal};
      const Vec3 a = openpbr::eval_light(tables, prepare(config.row, surface, n, v), l);
      const Vec3 b = openpbr::eval_light(tables, prepare(config.row, surface, n, l), v);
      for (size_t c = 0; c < 3; ++c) {
        EXPECT_NEAR(a[c] / l[2], b[c] / v[2], 1e-9 * (1.0 + std::abs(a[c] / l[2]))) << config.name << " " << i;
      }
    }
  }
}

TEST(OpenPbrClasses, DefaultsMatchClassRows) {
  classes::testing::reset();
  const std::shared_ptr<const classes::GpuRows> rows = classes::gpu_rows();
  ASSERT_NE(rows, nullptr);
  const classes::GpuRow row = default_row();
  EXPECT_EQ((*rows)[classes::Default], row);
  EXPECT_EQ((*rows)[classes::None], row);
  EXPECT_EQ((*rows)[classes::Reserved], row);
  EXPECT_EQ(row.v[3], 1.f);
  EXPECT_EQ(row.v[20], 1.5f);
  EXPECT_EQ(row.v[24], 0.f);
  EXPECT_EQ(row.v[27], 1.f);
  EXPECT_EQ(row.v[28], 0.85f);
  EXPECT_EQ(std::bit_cast<uint32_t>(row.v[30]), 0u);
  const classes::GpuRow foliage = built_in_row("foliage/grass");
  EXPECT_EQ(std::bit_cast<uint32_t>(foliage.v[30]), classes::RowThin | classes::RowTwoSided | classes::RowNoSsr);
  EXPECT_EQ(foliage.v[31], 0.5f);
  EXPECT_EQ((*rows)[7], foliage);
  EXPECT_EQ(built_in_row("ground/grass").v[24], 0.5f);
  EXPECT_EQ(sizeof(classes::GpuRow), openpbr::ClassBytes / openpbr::ClassRows);
}

TEST(OpenPbrClasses, LayeredRowsSelectTheLayeredVariant) {
  classes::testing::reset();
  // Thin walls run in the base text, so foliage/grass and default/cutout never pull a frame onto the layered one
  const openpbr::ClassSet builtIn = openpbr::layered_classes(*classes::gpu_rows());
  EXPECT_EQ(builtIn.count(), 0u);
  EXPECT_FALSE(openpbr::layered(default_row()));
  EXPECT_TRUE(openpbr::layered(classes::gpu_row(coat_params())));
  EXPECT_TRUE(openpbr::layered(classes::gpu_row(fuzz_params())));
  EXPECT_FALSE(openpbr::layered(classes::gpu_row(thin_params())));
  EXPECT_FALSE(openpbr::layered(built_in_row("default/cutout")));
  OpenPbrParams subsurface;
  subsurface.subsurfaceWeight = 0.5f;
  EXPECT_FALSE(openpbr::layered(classes::gpu_row(subsurface)));

  openpbr::ClassSet set = builtIn;
  set[0] = true;
  set[200] = true;
  const auto entry = [](uint32_t cls, uint32_t flags = 0) {
    return gx::surface::Entry{cls | flags << 8 | 0x09u << 16, 0, 0, 0};
  };
  EXPECT_TRUE(deferred::needs_layered(nullptr, set));
  const std::vector<gx::surface::Entry> plain{
      gx::surface::Entry{}, entry(0), entry(1), entry(2), entry(7), entry(8), entry(classes::ThinCutout)};
  EXPECT_FALSE(deferred::needs_layered(&plain, set));
  std::vector<gx::surface::Entry> coated = plain;
  coated.push_back(entry(200, gx::surface::Unlit));
  EXPECT_TRUE(deferred::needs_layered(&coated, set));
  EXPECT_FALSE(deferred::needs_layered(&coated, openpbr::ClassSet{}));
}

clusters::GpuLight prepared_light(const AuroraLight& light, const Mat3x4<float>& view) {
  SceneHints hints;
  hints.present = SceneHasMainCamera | SceneHasLights;
  hints.mainCamera.view = view;
  hints.lights = {light};
  std::vector<clusters::GpuLight> lights(clusters::MaxLights);
  clusters::LightListHeader header;
  clusters::UploadStats stats;
  clusters::prepare_lights(hints, 0, clusters::LightSpan{lights.data(), clusters::MaxLights}, header, stats);
  EXPECT_EQ(header.dirCount + header.localCount, 1u);
  return lights[0];
}

Mat3x4<float> identity_view() {
  return {Vec4<float>{1.f, 0.f, 0.f, 0.f}, Vec4<float>{0.f, 1.f, 0.f, 0.f}, Vec4<float>{0.f, 0.f, 1.f, 0.f}};
}

AuroraLight make_light(AuroraLightType type, std::array<float, 3> position, std::array<float, 3> direction,
                       float radius, uint32_t mask = 0, uint16_t flags = 0) {
  AuroraLight light{};
  light.id = 1;
  light.type = static_cast<uint8_t>(type);
  light.flags = flags;
  light.categoryMask = mask;
  std::ranges::copy(position, light.position);
  std::ranges::copy(direction, light.direction);
  light.color[0] = light.color[1] = light.color[2] = 1.f;
  light.radius = radius;
  light.spotInnerCos = 1.f;
  light.spotOuterCos = -1.f;
  light.falloffExponent = 2.f;
  light.volumetricScale = 1.f;
  return light;
}

TEST(DeferredParams, LayoutReservesViewToWorld) {
  EXPECT_EQ(sizeof(deferred::Params), 128u);
  EXPECT_EQ(offsetof(deferred::Params, camera), 0u);
  EXPECT_EQ(offsetof(deferred::Params, ambient), 48u);
  EXPECT_EQ(offsetof(deferred::Params, flags), 60u);
  EXPECT_EQ(offsetof(deferred::Params, viewToWorld), 64u);
  EXPECT_EQ(offsetof(deferred::Params, reserved), 112u);
  EXPECT_EQ(deferred::FlagSurfaceAmbient, 1u);
  EXPECT_EQ(deferred::ModeShift, 1u);
  EXPECT_EQ(deferred::BindSurfaceTable, 11u);
  EXPECT_EQ(deferred::BindSkyProbe, 12u);
  EXPECT_EQ(deferred::BindSkySh9, 13u);
  EXPECT_EQ(deferred::parse_mode("furnace-metal"), deferred::Mode::FurnaceMetal);
  EXPECT_EQ(deferred::parse_mode("grey"), deferred::Mode::Grey);
  EXPECT_EQ(deferred::parse_mode("Grey"), std::nullopt);
}

TEST(DeferredPipelines, SubgroupWalkNeedsSubgroups) {
  constexpr std::array<std::string_view, 0> none{};
  constexpr std::array<std::string_view, 2> core{"AURORA_CORE", "AURORA_F16"};
  constexpr std::array<std::string_view, 3> subgroups{"AURORA_CORE", "AURORA_F16", "AURORA_SUBGROUPS"};
  EXPECT_FALSE(deferred::subgroup_walk(none));
  EXPECT_FALSE(deferred::subgroup_walk(core));
  EXPECT_TRUE(deferred::subgroup_walk(subgroups));
}

TEST(DeferredMirror, FalloffFollowsDeclaredCurve) {
  for (const float p : {0.f, 1.f, 1.5f, 2.f}) {
    for (const float radius : {2015.f, 7555.f}) {
      for (const float gain : {1.f, 2.f}) {
        SCOPED_TRACE(std::to_string(p) + " r " + std::to_string(radius) + " gain " + std::to_string(gain));
        AuroraLight light = make_light(AURORA_LIGHT_POINT, {0.f, 0.f, -500.f}, {0.f, 0.f, -1.f}, radius);
        light.falloffExponent = p;
        // dusk's light tuning scales only the colour, so the gained light must keep the game's radius and curve
        for (float& c : light.color) {
          c *= gain;
        }
        const clusters::GpuLight gpu = prepared_light(light, identity_view());
        const auto lit = [&](double d) { return gpu.v2[0] * deferred::testing::attenuation(gpu, d); };
        double previous = lit(0.0);
        EXPECT_NEAR(previous, gain, 1e-6);
        EXPECT_NEAR(lit(radius), 0.0, 1e-6);
        EXPECT_LT(lit(radius * 0.999), 0.02 * gain);
        for (int i = 1; i <= 4096; ++i) {
          const double d = radius * i / 4096.0;
          const double a = lit(d);
          EXPECT_LE(a, previous + 1e-12) << "d " << d;
          if (p > 0.f) {
            const double k = 254.0 / std::pow(double{radius}, double{p});
            const double gx = 1.0 / (1.0 + k * std::pow(d, double{p}));
            EXPECT_LE(std::abs(a - gain * gx), gain / 255.0 + 1e-6) << "d " << d;
          }
          previous = a;
        }
      }
    }
  }
}

deferred::Params light_params(uint32_t flags = 0) {
  deferred::Params params;
  params.camera.flags = CameraValid;
  params.flags = flags;
  return params;
}

std::span<const classes::GpuRow, 256> class_rows() {
  static const std::shared_ptr<const classes::GpuRows> rows = classes::gpu_rows();
  return std::span<const classes::GpuRow, 256>{*rows};
}

deferred::testing::Pixel facing_pixel(uint32_t category) {
  deferred::testing::Pixel px;
  px.albedo = {0.8, 0.8, 0.8};
  px.category = category;
  px.position = {0.0, 0.0, 500.0};
  px.normal = {0.0, 0.0, -1.0};
  return px;
}

TEST(DeferredMirror, CategoryMaskSelectsReceivers) {
  const clusters::GpuLight sun = prepared_light(
      make_light(AURORA_LIGHT_DIRECTIONAL, {}, {0.f, 0.f, -1.f}, 0.f, AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER),
      identity_view());
  const clusters::GpuLight bulb = prepared_light(
      make_light(AURORA_LIGHT_POINT, {0.f, 0.f, -300.f}, {}, 5000.f, AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER),
      identity_view());
  EXPECT_EQ(clusters::light_byte(sun), 0x82u);
  EXPECT_EQ(clusters::light_byte(bulb), 0x82u);
  for (const clusters::GpuLight& light : {sun, bulb}) {
    const clusters::LightListHeader header{
        .dirCount = clusters::light_type(light) == AURORA_LIGHT_DIRECTIONAL ? 1u : 0u,
        .localCount = clusters::light_type(light) == AURORA_LIGHT_DIRECTIONAL ? 0u : 1u};
    for (const uint32_t category : {0x82u, 0x86u, 0x09u}) {
      const deferred::testing::Result lit =
          deferred::testing::shade(facing_pixel(category), header, std::span{&light, 1}, light_params(), class_rows());
      ASSERT_EQ(lit.alpha, 1.0);
      if (category == 0x09u) {
        EXPECT_EQ(lit.rgb, (Vec3{0.0, 0.0, 0.0})) << category;
      } else {
        EXPECT_GT(lit.rgb[0], 0.1) << category;
      }
    }
    clusters::GpuLight all = light;
    all.v4[0] &= ~0xFF00u;
    const deferred::testing::Result terrain =
        deferred::testing::shade(facing_pixel(0x09), header, std::span{&all, 1}, light_params(), class_rows());
    EXPECT_GT(terrain.rgb[0], 0.1);
  }
}

TEST(DeferredMirror, NoSpecularSkipsLobes) {
  const clusters::LightListHeader header{.dirCount = 1};
  deferred::testing::Pixel px = facing_pixel(0x09);
  px.roughness = 0.3;
  px.metalness = 1.0;
  px.normal = normalized({0.3, 0.0, -1.0});
  const clusters::GpuLight plain =
      prepared_light(make_light(AURORA_LIGHT_DIRECTIONAL, {}, {-0.3f, 0.f, -1.f}, 0.f), identity_view());
  const clusters::GpuLight fill = prepared_light(
      make_light(AURORA_LIGHT_DIRECTIONAL, {}, {-0.3f, 0.f, -1.f}, 0.f, 0, AURORA_LIGHT_NO_SPECULAR), identity_view());
  const deferred::testing::Result a =
      deferred::testing::shade(px, header, std::span{&plain, 1}, light_params(), class_rows());
  const deferred::testing::Result b =
      deferred::testing::shade(px, header, std::span{&fill, 1}, light_params(), class_rows());
  EXPECT_GT(a.rgb[0], 0.05);
  EXPECT_EQ(b.rgb, (Vec3{0.0, 0.0, 0.0}));
  px.metalness = 0.0;
  const deferred::testing::Result c =
      deferred::testing::shade(px, header, std::span{&fill, 1}, light_params(), class_rows());
  const deferred::testing::Result d =
      deferred::testing::shade(px, header, std::span{&plain, 1}, light_params(), class_rows());
  EXPECT_GT(c.rgb[0], 0.1);
  EXPECT_GT(d.rgb[0], c.rgb[0]);
}

TEST(DeferredMirror, AmbientIsLinearised) {
  auto hints = std::make_shared<SceneHints>();
  hints->present = SceneHasEnvironment;
  hints->environment.value.ambientColor[0] = 0.5f;
  hints->environment.value.ambientColor[1] = 0.04f;
  hints->environment.value.ambientColor[2] = 1.f;
  deferred::Params params;
  deferred::fill_params(hints.get(), false, deferred::Mode::Full, params);
  EXPECT_NEAR(params.ambient[0], 0.21404114f, 1e-6f);
  EXPECT_NEAR(params.ambient[1], 0.04f / 12.92f, 1e-7f);
  EXPECT_NEAR(params.ambient[2], 1.f, 1e-6f);
  EXPECT_EQ(params.flags, 0u);
  deferred::fill_params(nullptr, true, deferred::Mode::Furnace, params);
  EXPECT_EQ(params.ambient[0], 0.f);
  EXPECT_EQ(params.flags, deferred::FlagSurfaceAmbient | 4u << deferred::ModeShift);
  deferred::fill_params(hints.get(), false, deferred::Mode::Full, params);
  params.camera.flags = CameraValid;
  deferred::testing::Pixel px = facing_pixel(0x09);
  px.albedo = {1.0, 1.0, 1.0};
  const deferred::testing::Result out = deferred::testing::shade(px, {}, {}, params, class_rows());
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(out.rgb[c], params.ambient[c], 1e-6) << c;
  }
}

TEST(DeferredMirror, SurfaceAmbientUsesZ) {
  deferred::testing::Pixel px = facing_pixel(0x09);
  px.albedo = {1.0, 1.0, 1.0};
  px.ambient = {0.5, 0.5, 0.5};
  deferred::Params params = light_params(deferred::FlagSurfaceAmbient);
  params.ambient[0] = params.ambient[1] = params.ambient[2] = 0.9f;
  const deferred::testing::Result z = deferred::testing::shade(px, {}, {}, params, class_rows());
  EXPECT_NEAR(z.rgb[0], 0.21404114, 1e-6);
  params.flags = 0;
  const deferred::testing::Result env = deferred::testing::shade(px, {}, {}, params, class_rows());
  EXPECT_NEAR(env.rgb[0], 0.9, 1e-6);
  px.ambient = {};
  params.flags = deferred::FlagSurfaceAmbient;
  EXPECT_EQ(deferred::testing::shade(px, {}, {}, params, class_rows()).rgb, (Vec3{0.0, 0.0, 0.0}));
}

TEST(DeferredMirror, EmissionSitsUnderCoatAndFuzz) {
  std::array<classes::GpuRow, 256> rows{};
  std::ranges::copy(class_rows(), rows.begin());
  for (const auto& [id, params] :
       {std::pair{200u, OpenPbrParams{}}, std::pair{201u, coat_params()}, std::pair{202u, fuzz_params()}}) {
    OpenPbrParams emitter = params;
    emitter.emissionScale = 2.f;
    rows[id] = classes::gpu_row(emitter);
  }
  deferred::testing::Pixel px = facing_pixel(0x09);
  px.albedo = {1.0, 1.0, 1.0};
  px.emissive = {0.5, 0.25, 1.0};
  const auto shaded = [&](uint32_t cls) {
    px.classId = cls;
    return deferred::testing::shade(px, {}, {}, light_params(), rows).rgb;
  };
  const Vec3 plain = shaded(200);
  const Vec3 v = scale(px.position, -1.0 / std::sqrt(dot(px.position, px.position)));
  for (const uint32_t cls : {201u, 202u}) {
    const Vec3 under = shaded(cls);
    const Vec3 passed = openpbr::emission_scale(prepare(rows[cls], {.baseColor = {1.0, 1.0, 1.0}}, px.normal, v));
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_EQ(plain[c], px.emissive[c] * 2.0) << c;
      EXPECT_LT(passed[c], 1.0) << cls << " " << c;
      EXPECT_NEAR(under[c], plain[c] * passed[c], 1e-12) << cls << " " << c;
    }
  }
  EXPECT_EQ(openpbr::emission_scale(prepare(rows[200], {}, px.normal, v)), (Vec3{1.0, 1.0, 1.0}));
}

TEST(DeferredMirror, SsilFillsScreenAoShare) {
  const std::span<const classes::GpuRow, 256> rows = class_rows();
  deferred::testing::Pixel px;
  px.albedo = {0.8, 0.6, 0.4};
  px.ambient = {0.5, 0.5, 0.5};
  px.aoFinal = 0.5;
  px.ao = 0.3;
  px.normal = {0.2, 0.3, -0.9};
  px.position = {20.0, -10.0, 400.0};
  deferred::Params params;
  deferred::fill_params(nullptr, true, deferred::Mode::Full, params);
  const deferred::testing::Result off = deferred::testing::shade(px, {}, {}, params, rows);
  deferred::Params on = params;
  on.flags |= deferred::FlagSsil;
  EXPECT_EQ(deferred::testing::shade(px, {}, {}, on, rows).rgb, off.rgb);
  px.ssil = {0.2, 0.1, 0.05, 0.7};
  const deferred::testing::Result one = deferred::testing::shade(px, {}, {}, on, rows);
  px.ssil = {0.4, 0.2, 0.1, 0.7};
  const deferred::testing::Result two = deferred::testing::shade(px, {}, {}, on, rows);
  px.ao = 0.9;
  const deferred::testing::Result texture = deferred::testing::shade(px, {}, {}, on, rows);
  const deferred::testing::Result textureOff = deferred::testing::shade(px, {}, {}, params, rows);
  for (size_t c = 0; c < 3; ++c) {
    const double first = one.rgb[c] - off.rgb[c];
    EXPECT_GT(first, 0.0) << c;
    EXPECT_NEAR(two.rgb[c] - off.rgb[c], 2.0 * first, 1e-9) << c;
    EXPECT_NEAR(texture.rgb[c] - textureOff.rgb[c], 2.0 * first, 1e-9) << "AO8 must not scale the fill";
  }
  px.ao = 0.3;
  px.aoFinal = 1.0;
  EXPECT_EQ(deferred::testing::shade(px, {}, {}, on, rows).rgb, deferred::testing::shade(px, {}, {}, params, rows).rgb);
  px.aoFinal = 0.5;
  px.receiver = 0;
  EXPECT_EQ(deferred::testing::shade(px, {}, {}, on, rows).rgb, deferred::testing::shade(px, {}, {}, params, rows).rgb);
}

TEST(DeferredMirror, FurnaceAoIsScreenAoPlusBounce) {
  const std::span<const classes::GpuRow, 256> rows = class_rows();
  deferred::Params params;
  deferred::fill_params(nullptr, true, deferred::Mode::FurnaceAo, params);
  params.flags |= deferred::FlagSsil;
  for (const double strength : {0.0, 0.5, 1.0, 1.5}) {
    for (const double raw : {0.2, 0.45, 0.8, 1.0}) {
      const double v = ao::apply_strength(static_cast<float>(raw), static_cast<float>(strength), 0.2f);
      for (const double lb : {0.0, 0.3, 1.0}) {
        for (const uint32_t cls : {1u, 4u, 9u}) {
          deferred::testing::Pixel px;
          px.classId = cls;
          px.albedo = {0.3, 0.7, 0.1};
          px.ao = 0.4;
          px.aoFinal = v;
          px.ssil = {lb, lb, lb, 1.0};
          const deferred::testing::Result r = deferred::testing::shade(px, {}, {}, params, rows);
          const double as = std::pow(v, 2.2);
          for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(r.rgb[c], as + (1.0 - as) * lb, 2e-3) << strength << " " << raw << " " << lb << " " << cls;
            EXPECT_LE(r.rgb[c], 1.0 + 2e-3);
          }
        }
      }
    }
  }
}

TEST(DeferredMirror, BackFacingNormalsAreBent) {
  const Vec3 v{0.0, 0.0, -1.0};
  for (const Vec3 raw :
       {Vec3{0.0, 0.0, 1.0}, Vec3{0.6, 0.0, 0.8}, Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 0.0, 0.0}, Vec3{0.0, 0.0, 1e-4}}) {
    const Vec3 n = deferred::testing::bend_normal(raw, v);
    EXPECT_NEAR(dot(n, n), 1.0, 1e-9);
    EXPECT_GT(dot(n, v), 0.0);
  }
  EXPECT_EQ(deferred::testing::bend_normal({0.0, 0.0, 0.0}, v), v);
  deferred::testing::Pixel px = facing_pixel(0x09);
  px.normal = {0.0, 0.0, 1.0};
  px.ambient = {0.5, 0.5, 0.5};
  const deferred::testing::Result out =
      deferred::testing::shade(px, {}, {}, light_params(deferred::FlagSurfaceAmbient), class_rows());
  EXPECT_EQ(out.alpha, 1.0);
  EXPECT_TRUE(std::isfinite(out.rgb[0]));
}

Mat4x4<float> perspective(float n, float f) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -n / (f - n);
  proj.m2[3] = -(f * n) / (f - n);
  proj.m3[2] = -1.f;
  return proj;
}

std::shared_ptr<SceneHints> scene_hints(std::vector<AuroraLight> lights, bool withLights = true) {
  auto hints = std::make_shared<SceneHints>();
  const Mat4x4<float> proj = perspective(Near, Far);
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .view = identity_view(),
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
  hints->present = SceneHasMainCamera | (withLights ? SceneHasLights : 0u);
  hints->lights = std::move(lights);
  hints->motionValid = true;
  return hints;
}

Settings pbr_settings(float legacyBlend = 0.f) {
  Settings settings;
  settings.pbr = true;
  settings.legacyBlend = legacyBlend;
  return settings;
}

struct TableStub {
  std::vector<gx::surface::Entry> entries;
  bool enabled = true;
  int calls = 0;
};

void declare_table_stub(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Prepare);
  builder.create_buffer(resource::SurfaceTable, {.size = gx_ext::surface_table::TableBytes}, Access::CopyDst);
}

bool table_stub_enabled(const StageData&, const void* state) { return static_cast<const TableStub*>(state)->enabled; }

void encode_table_stub(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<TableStub*>(const_cast<void*>(state));
  ++stub.calls;
  const BufferView view = ctx.buffer(BufRef{0});
  if (view.buffer && !stub.entries.empty()) {
    ctx.encoder().WriteBuffer(view.buffer, view.offset, reinterpret_cast<const uint8_t*>(stub.entries.data()),
                              stub.entries.size() * sizeof(gx::surface::Entry));
  }
}

PassDesc table_stub_desc(TableStub& stub) {
  return {
      .name = "test.surface_table",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .order = 0,
      .declare = declare_table_stub,
      .enabled = table_stub_enabled,
      .encode = encode_table_stub,
      .state = &stub,
  };
}

struct Readback {
  std::string_view resource;
  bool isBuffer = false;
  wgpu::Buffer buffer;
  wgpu::Texture texture;
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
  if (!reader.isBuffer) {
    reader.texture = ctx.texture(TexRef{0});
    return;
  }
  const BufferView view = ctx.buffer(BufRef{0});
  if (view.buffer && reader.buffer) {
    ctx.encoder().CopyBufferToBuffer(view.buffer, view.offset, reader.buffer, 0,
                                     std::min(view.size, reader.buffer.GetSize()));
  }
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

class DeferredDeviceTest : public ::testing::Test {
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
    return use_pbr_tier();
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
    classes::testing::reset();
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    deferred::shutdown();
    deferred::testing::reset();
    openpbr::shutdown();
    clusters::shutdown();
    clusters::testing::reset();
    composite::shutdown();
    composite::testing::reset();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  void start_modules() {
    clusters::initialize();
    openpbr::initialize();
    deferred::initialize();
    register_pass(table_stub_desc(m_table));
  }

  bool use_pbr_tier() {
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
    return true;
  }

  void use_globals(const wgpu::Extent3D& size) {
    webgpu::set_gbuffer(false);
    m_globals.reset();
    m_options.efbSize = size;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
    start_modules();
    ASSERT_TRUE(use_pbr_tier());
  }

  void set_hints(const std::shared_ptr<const SceneHints>& hints) {
    m_hints = hints;
    clusters::testing::set_hints_override(hints);
    deferred::testing::set_hints_override(hints);
    composite::testing::set_camera_override(hints);
  }

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(const Settings& settings,
                                                       DepthSource source = DepthSource::Direct) {
    m_settings = settings;
    m_env = gather_graph_environment();
    std::shared_ptr<CompiledGraph> graph = compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = source});
    prepare_graph_bindings(*graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = source});
    return graph;
  }

  void latch(const CompiledGraph* graph) {
    const PassSetup latched{.settings = &m_settings, .env = &m_env};
    clusters::on_latch(latched, graph);
    deferred::on_latch(latched, graph);
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
        .sceneDepth = scene_depth(m_depth),
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
      FrameUniforms frame;
      const GraphExtent efb = opaque.frame->graph->efb_extent(Stage::OpaqueDone);
      frame.constants.efbSize[0] = efb.width;
      frame.constants.efbSize[1] = efb.height;
      s_uniforms.assign(FrameUniformsSize, 0);
      std::memcpy(s_uniforms.data(), &frame, sizeof(frame));
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

  void make_depth(uint32_t width, uint32_t height) { m_depth = test::make_depth(*m_device, width, height); }

  static inline std::vector<uint8_t> s_uniforms;
  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
  TableStub m_table;
  wgpu::Texture m_depth;
  std::shared_ptr<const SceneHints> m_hints;
};

class LightDeferredGraph : public DeferredDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 48, 1}})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    make_depth(64, 48);
    set_hints(scene_hints({make_light(AURORA_LIGHT_POINT, {0.f, 0.f, -300.f}, {}, 5000.f)}));
  }
};

TEST_F(LightDeferredGraph, DeclaresOnPbrSurface) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  ASSERT_TRUE(openpbr::resident());
  const auto graph = compile(pbr_settings());
  ASSERT_EQ(status(*graph, deferred::Pass), CullReason::None);
  ASSERT_EQ(status(*graph, clusters::Pass), CullReason::None);
  const NodeId node = graph->find_node(deferred::Pass);
  const auto uses = graph->node_uses(node);
  const std::array<std::string_view, 16> names{
      resource::Surface,          resource::GxNormal,       resource::GxDepth,
      resource::AoFinal,          clusters::List,           clusters::Grid,
      openpbr::Classes,           openpbr::Lut3d,           openpbr::Lut2d,
      resource::SceneHdr,         resource::SurfaceExt,     resource::SurfaceTable,
      lighting::sky_probe::Probe, lighting::sky_probe::Sh9, resource::SsilIrradiance,
      resource::SunVisibility};
  ASSERT_EQ(uses.size(), names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, names[i]) << i;
  }
  EXPECT_EQ(uses[deferred::BindAo].missing, Missing::BindDummyWhite);
  EXPECT_EQ(uses[deferred::BindSurfaceExt].missing, Missing::BindDummyBlack);
  EXPECT_EQ(uses[deferred::BindHdr].access, Access::StorageWrite);
  EXPECT_EQ(uses[deferred::BindList].access, Access::Uniform);
  EXPECT_EQ(uses[deferred::BindClasses].access, Access::Uniform);
  const CompiledResource& hdr = graph->resources[graph->find_resource(resource::SceneHdr)];
  EXPECT_EQ(hdr.tex.format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_EQ(hdr.extent.width, 64u);
  EXPECT_EQ(hdr.extent.height, 48u);
  EXPECT_TRUE(hdr.textureUsage & wgpu::TextureUsage::TextureBinding);
  EXPECT_TRUE(hdr.textureUsage & wgpu::TextureUsage::StorageBinding);
  EXPECT_EQ(graph->resources[graph->find_resource(openpbr::Lut3d)].kind, ResourceKind::Persistent);
  EXPECT_EQ(graph->nodes[node].paramsSize, sizeof(deferred::Params));
  EXPECT_TRUE(composite::hdr_produced(*graph));
  EXPECT_EQ(status(*compile(pbr_settings(1.f)), deferred::Pass), CullReason::None);
  EXPECT_EQ(status(*compile(Settings{}), deferred::Pass), CullReason::Skipped);
  EXPECT_EQ(status(*compile(pbr_settings(), DepthSource::Copy), deferred::Pass), CullReason::Skipped);
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  EXPECT_EQ(status(*compile(pbr_settings()), deferred::Pass), CullReason::Skipped);
}

TEST_F(LightDeferredGraph, BindingOrderAppendsSsil) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  const auto graph = compile(pbr_settings());
  const NodeId node = graph->find_node(deferred::Pass);
  ASSERT_NE(node, InvalidNode);
  const auto uses = graph->node_uses(node);
  ASSERT_GT(uses.size(), deferred::BindSsil);
  EXPECT_EQ(deferred::BindSsil, 14u);
  EXPECT_EQ(graph->resources[uses[deferred::BindSsil].resource].name, resource::SsilIrradiance);
  EXPECT_EQ(uses[deferred::BindSsil].missing, Missing::BindDummyBlack);
  EXPECT_TRUE(uses[deferred::BindSsil].staticDummy);
  EXPECT_FALSE(deferred::ssil_bound(*graph));
  deferred::Params params;
  deferred::fill_params(nullptr, true, deferred::Mode::Full, params, deferred::ssil_bound(*graph));
  EXPECT_EQ(params.flags & deferred::FlagSsil, 0u);
  EXPECT_EQ(deferred::parse_mode("furnace-ao"), deferred::Mode::FurnaceAo);
  EXPECT_EQ(static_cast<uint32_t>(deferred::Mode::FurnaceAo) << deferred::ModeShift & ~deferred::ModeMask, 0u);
  EXPECT_EQ(deferred::FlagSsil & deferred::ModeMask, 0u);
}

TEST_F(LightDeferredGraph, DisabledAtFullBlend) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  const auto graph = live_graph(pbr_settings(1.f));
  EXPECT_FALSE(clusters::armed());
  const FrameState state{.graph = graph};
  const StageData opaque = build(state, 1);
  EXPECT_FALSE(deferred::frame_enabled(opaque));
  EXPECT_FALSE(pass_enabled(*graph, opaque, deferred::Pass));
  EXPECT_FALSE(pass_enabled(*graph, opaque, clusters::Pass));
  EXPECT_FALSE(pass_enabled(*graph, opaque, composite::Pass));
  const auto lit = live_graph(pbr_settings(0.f));
  EXPECT_TRUE(clusters::armed());
  const FrameState litState{.graph = lit};
  const StageData litOpaque = build(litState, 2);
  EXPECT_TRUE(pass_enabled(*lit, litOpaque, deferred::Pass));
  EXPECT_TRUE(pass_enabled(*lit, litOpaque, clusters::Pass));
  EXPECT_TRUE(deferred::ran(litOpaque));
}

TEST_F(LightDeferredGraph, NoLightHintsDisable) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  set_hints(scene_hints({}, false));
  const auto graph = live_graph(pbr_settings(0.f));
  ASSERT_TRUE(clusters::armed());
  const FrameState state{.graph = graph};
  const StageData opaque = build(state, 1);
  EXPECT_FALSE(pass_enabled(*graph, opaque, deferred::Pass));
  EXPECT_FALSE(pass_enabled(*graph, opaque, composite::Pass));
  EXPECT_GE(deferred::testing::counters().noLights, 1u);
}

TEST_F(LightDeferredGraph, NonResidentLutsDropToDummy) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  const auto graph = live_graph(pbr_settings(0.f));
  ASSERT_TRUE(clusters::armed());
  ASSERT_TRUE(composite::pbr_active() || composite::identity_forced());
  const FrameState state{.graph = graph};
  const StageData lit = build(state, 1);
  ASSERT_TRUE(pass_enabled(*graph, lit, deferred::Pass));
  EXPECT_TRUE(deferred::ready(m_settings));
  openpbr::testing::drop_resident();
  const StageData dropped = build(state, 2);
  EXPECT_FALSE(pass_enabled(*graph, dropped, deferred::Pass));
  EXPECT_FALSE(pass_enabled(*graph, dropped, clusters::Pass));
  EXPECT_FALSE(pass_enabled(*graph, dropped, composite::Pass));
  EXPECT_FALSE(composite::pbr_applied(dropped));
  EXPECT_FALSE(deferred::ready(m_settings));
  EXPECT_GE(deferred::testing::counters().notResident, 1u);
}

TEST_F(LightDeferredGraph, MissingClustersDropsToDummy) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  clusters::shutdown();
  const auto graph = live_graph(pbr_settings(0.f));
  EXPECT_NE(status(*graph, deferred::Pass), CullReason::None);
  EXPECT_FALSE(composite::hdr_produced(*graph));
  EXPECT_FALSE(composite::pbr_active());
}

TEST_F(LightDeferredGraph, ReadyNeedsLighting) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  EXPECT_TRUE(lighting::ready(Settings{}));
  m_settings = pbr_settings();
  EXPECT_FALSE(lighting::ready(m_settings));
  const auto graph = live_graph(pbr_settings());
  EXPECT_TRUE(clusters::ready(m_settings));
  EXPECT_TRUE(deferred::ready(m_settings));
  EXPECT_TRUE(lighting::ready(m_settings));
  openpbr::testing::drop_resident();
  EXPECT_FALSE(lighting::ready(m_settings));
  EXPECT_TRUE(lighting::ready(Settings{}));
}

TEST_F(LightDeferredGraph, NoActiveConsumerAtFullBlend) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  const auto graph = live_graph(pbr_settings(1.f));
  ASSERT_EQ(status(*graph, clusters::Pass), CullReason::None);
  const FrameState state{.graph = graph};
  EXPECT_FALSE(pass_enabled(*graph, build(state, 1), clusters::Pass));
  EXPECT_EQ(clusters::testing::counters().dispatches, 0u);
}

// One pixel of the GPU fixture: Surface record and entry, GX normal and depth
struct GpuPixel {
  std::array<uint8_t, 3> albedo{200, 200, 200};
  uint8_t ao = 255;
  uint8_t roughness = 217;
  uint8_t metal = 0;
  uint8_t cls = 1;
  uint8_t flags = 0;
  uint8_t category = 0x09;
  std::array<uint8_t, 3> ambient{};
  Vec3 normal{0.0, 0.0, -1.0};
  double z = 500.0;
  uint8_t receiver = 1;
  bool slot = true;
  uint32_t entryBits = 0;
};

uint32_t unorm10(double value) {
  return static_cast<uint32_t>(std::lround(std::clamp((value + 1.0) * 0.5, 0.0, 1.0) * 1023.0));
}

Vec3 decoded_normal(const Vec3& n) {
  const auto decode = [](uint32_t code) { return code / 1023.0 * 2.0 - 1.0; };
  return {decode(unorm10(n[0])), decode(unorm10(n[1])), -decode(unorm10(-n[2]))};
}

struct StubTexture {
  std::string name;
  std::string_view resource;
  wgpu::TextureFormat format = wgpu::TextureFormat::RGBA8Unorm;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  int calls = 0;
};

void declare_stub_texture(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& stub = *static_cast<const StubTexture*>(state);
  builder.after(phase::Ao);
  builder.create(stub.resource, {.format = stub.format, .sizeClass = SizeClass::Efb}, Access::CopyDst);
}

void encode_stub_texture(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubTexture*>(const_cast<void*>(state));
  ++stub.calls;
  const wgpu::Extent3D size = ctx.extent(TexRef{0});
  const wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = stub.rowBytes, .rowsPerImage = size.height},
                                       .buffer = stub.staging};
  const wgpu::TexelCopyTextureInfo to{.texture = ctx.texture(TexRef{0})};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  ctx.encoder().CopyBufferToTexture(&from, &to, &extent);
}

PassDesc stub_texture_desc(StubTexture& stub) {
  return {
      .name = stub.name,
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 5,
      .runsWithoutMarker = true,
      .declare = declare_stub_texture,
      .encode = encode_stub_texture,
      .state = &stub,
  };
}

class RenderLightingGraph : public LightDeferredGraph {};

TEST_F(RenderLightingGraph, DeferredBindsVisSunAt15) {
  composite::initialize();
  composite::testing::set_pbr_producers_override(true);
  const auto graph = compile(pbr_settings());
  const NodeId node = graph->find_node(deferred::Pass);
  ASSERT_NE(node, InvalidNode);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), deferred::BindSunVisibility + 1u);
  EXPECT_EQ(deferred::BindSunVisibility, 15u);
  const CompiledUse& use = uses[deferred::BindSunVisibility];
  EXPECT_EQ(graph->resources[use.resource].name, resource::SunVisibility);
  EXPECT_EQ(use.access, Access::Sampled);
  EXPECT_EQ(use.missing, Missing::BindDummyWhite);
  EXPECT_TRUE(use.staticDummy);
  EXPECT_FALSE(deferred::sun_bound(*graph));
  StubTexture stub{.name = "t.vis_sun", .resource = resource::SunVisibility, .format = wgpu::TextureFormat::RG8Unorm};
  const PassId stubId = register_pass(stub_texture_desc(stub));
  const auto produced = compile(pbr_settings());
  unregister_pass(stubId);
  EXPECT_TRUE(deferred::sun_bound(*produced));
  EXPECT_FALSE(deferred::ssil_bound(*produced));
  deferred::Params params;
  deferred::fill_params(nullptr, true, deferred::Mode::Full, params, false, deferred::NoSunLight);
  EXPECT_EQ(params.flags & deferred::FlagSunShadow, 0u);
  EXPECT_EQ(params.reserved[0], 0u);
}

class LightDeferredGpu : public DeferredDeviceTest {
protected:
  void SetUp() override {
    const auto backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND"));
    if (!backend) {
      GTEST_SKIP() << "set AURORA_TEST_BACKEND to run light.deferred on a real backend";
    }
    if (!open({.backend = *backend, .efbSize = {64, 48, 1}})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    if (select_scene_depth_source(m_device->core, 1, true) != DepthSource::Direct) {
      GTEST_SKIP() << "needs direct scene depth";
    }
  }

  wgpu::Texture upload(wgpu::TextureFormat format, uint32_t width, uint32_t height, const void* data,
                       uint32_t texelBytes, const char* label) const {
    const wgpu::TextureDescriptor descriptor{
        .label = label,
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = {width, height, 1},
        .format = format,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * texelBytes, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device->queue.WriteTexture(&destination, data, size_t{width} * height * texelBytes, &layout, &size);
    return texture;
  }

  static void publish(webgpu::TextureWithSampler& target, const wgpu::Texture& texture) {
    target.texture = texture;
    target.view = texture.CreateView();
    target.size = {texture.GetWidth(), texture.GetHeight(), 1};
    target.format = texture.GetFormat();
  }

  std::vector<std::array<uint16_t, 4>> read_hdr(const wgpu::Texture& texture) const {
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    const uint32_t row = (width * 8 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<std::array<uint16_t, 4>> out(size_t{width} * height);
    if (!words) {
      ADD_FAILURE() << "scene.hdr readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      std::memcpy(out.data() + size_t{y} * width, bytes + size_t{y} * row, size_t{width} * 8);
    }
    return out;
  }

  StageCamera camera() const {
    StageCamera cam;
    EXPECT_TRUE(fill_stage_camera(m_hints.get(), cam));
    return cam;
  }

  // Uploads the pixels and returns their view-forward positions as the shader reconstructs them
  std::vector<deferred::testing::Pixel> stage_pixels(uint32_t width, uint32_t height,
                                                     const std::vector<GpuPixel>& pixels) {
    if (webgpu::g_frameBuffer.size.width != width || webgpu::g_frameBuffer.size.height != height) {
      use_globals({width, height, 1});
      set_hints(m_hints);
    }
    make_depth(width, height);
    const StageCamera cam = camera();
    std::vector<uint16_t> surface(pixels.size() * 4);
    std::vector<uint32_t> normal(pixels.size());
    std::vector<float> depth(pixels.size());
    m_table.entries.assign(1, gx::surface::Entry{});
    std::map<gx::surface::Entry, uint32_t> slots;
    std::vector<deferred::testing::Pixel> out(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
      const GpuPixel& p = pixels[i];
      uint32_t slot = 0;
      if (p.slot) {
        const gx::surface::Entry entry{
            uint32_t{p.cls} | uint32_t{p.flags} << 8 | uint32_t{p.category} << 16 | p.entryBits,
            uint32_t{p.ambient[0]} | uint32_t{p.ambient[1]} << 8 | uint32_t{p.ambient[2]} << 16, 0, 0};
        const auto [it, added] = slots.try_emplace(entry, static_cast<uint32_t>(m_table.entries.size()));
        if (added) {
          m_table.entries.push_back(entry);
        }
        slot = it->second;
      }
      const uint32_t x =
          uint32_t{p.albedo[0]} | uint32_t{p.albedo[1]} << 8 | uint32_t{p.albedo[2]} << 16 | uint32_t{p.ao} << 24;
      const uint32_t y = uint32_t{p.roughness} | uint32_t{p.metal} << 8 | slot << 16;
      surface[i * 4] = static_cast<uint16_t>(x & 0xFFFF);
      surface[i * 4 + 1] = static_cast<uint16_t>(x >> 16);
      surface[i * 4 + 2] = static_cast<uint16_t>(y & 0xFFFF);
      surface[i * 4 + 3] = static_cast<uint16_t>(y >> 16);
      normal[i] =
          unorm10(p.normal[0]) | unorm10(p.normal[1]) << 10 | unorm10(-p.normal[2]) << 20 | uint32_t{p.receiver} << 30;
      depth[i] = p.z > 0.0 ? static_cast<float>(cam.depthAdd - cam.depthMul / p.z) : 0.f;
      deferred::testing::Pixel& m = out[i];
      for (size_t c = 0; c < 3; ++c) {
        m.albedo[c] = p.albedo[c] * gx::surface::InvUnorm8;
        m.ambient[c] = p.ambient[c] * gx::surface::InvUnorm8;
      }
      m.ao = p.ao * gx::surface::InvUnorm8;
      m.roughness = p.roughness * gx::surface::InvUnorm8;
      m.metalness = p.metal * gx::surface::InvUnorm8;
      m.classId = p.slot ? p.cls : 0;
      m.flags = p.flags;
      m.category = p.category;
      m.normal = decoded_normal(p.normal);
      m.receiver = p.receiver;
      const uint32_t px = static_cast<uint32_t>(i % width);
      const uint32_t py = static_cast<uint32_t>(i / width);
      const float d = depth[i];
      m.depth = d;
      const double z = d > 0.f ? double{cam.depthMul} / (double{cam.depthAdd} - d) : 1e30;
      const double ndcX = 2.0 * (px + 0.5) / width - 1.0;
      const double ndcY = 1.0 - 2.0 * (py + 0.5) / height;
      const double cx = cam.p02 - 2.0 * cam.jitterPx[0] / width;
      const double cy = cam.p12 + 2.0 * cam.jitterPx[1] / height;
      m.position = {(ndcX + cx) * z / cam.p00, (ndcY + cy) * z / cam.p11, z};
    }
    publish(webgpu::g_surfaceBuffer,
            upload(webgpu::SurfaceBufferFormat, width, height, surface.data(), 8, "test surface"));
    publish(webgpu::g_normalBuffer,
            upload(wgpu::TextureFormat::RGB10A2Unorm, width, height, normal.data(), 4, "test normals"));
    test::fill_depth(*m_device, m_depth, depth);
    return out;
  }

  std::vector<std::array<uint16_t, 4>> run_hdr(const Settings& settings, uint64_t frame = 1) {
    Readback reader{.resource = resource::SceneHdr};
    const PassId readId = register_pass(readback_desc(reader, "t.hdr_read"));
    const auto graph = live_graph(settings);
    const FrameState state{.graph = graph};
    StageData stage = build(state, frame);
    EXPECT_TRUE(pass_enabled(*graph, stage, deferred::Pass));
    EXPECT_EQ(encode(stage), std::nullopt);
    unregister_pass(readId);
    EXPECT_EQ(reader.calls, 1);
    if (!reader.texture) {
      ADD_FAILURE() << "no scene.hdr";
      return {};
    }
    return read_hdr(reader.texture);
  }

  struct Mirror {
    clusters::LightListHeader header;
    std::vector<clusters::GpuLight> lights = std::vector<clusters::GpuLight>(clusters::MaxLights);
  };

  Mirror mirror_lights() const {
    Mirror out;
    clusters::UploadStats stats;
    clusters::prepare_lights(*m_hints, 0, clusters::LightSpan{out.lights.data(), clusters::MaxLights}, out.header,
                             stats);
    return out;
  }

  deferred::Params mirror_params(deferred::Mode mode = deferred::Mode::Full) const {
    deferred::Params params;
    deferred::fill_params(m_hints.get(), true, mode, params);
    return params;
  }

  void stage_ao(uint32_t width, uint32_t height, const std::vector<uint8_t>& ao) {
    m_ao = StubTexture{.name = "t.ao_final", .resource = resource::AoFinal};
    m_ao.rowBytes = (width * 4 + 255) & ~255u;
    std::vector<uint8_t> staged(size_t{m_ao.rowBytes} * height, 0);
    for (size_t i = 0; i < ao.size(); ++i) {
      staged[(i / width) * m_ao.rowBytes + (i % width) * 4] = ao[i];
    }
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = staged.size()};
    m_ao.staging = m_device->device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(m_ao.staging, 0, staged.data(), staged.size());
    m_aoId = register_pass(stub_texture_desc(m_ao));
  }

  PassId stage_ssil(uint32_t width, uint32_t height, const std::vector<std::array<float, 4>>& irr) {
    m_ssil = StubTexture{
        .name = "t.ssil_irr", .resource = resource::SsilIrradiance, .format = wgpu::TextureFormat::RGBA16Float};
    m_ssil.rowBytes = (width * 8 + 255) & ~255u;
    std::vector<uint16_t> staged(size_t{m_ssil.rowBytes / 2} * height, 0);
    for (size_t i = 0; i < irr.size(); ++i) {
      for (size_t c = 0; c < 4; ++c) {
        staged[(i / width) * (m_ssil.rowBytes / 2) + (i % width) * 4 + c] = openpbr::to_half(irr[i][c]);
      }
    }
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = staged.size() * 2};
    m_ssil.staging = m_device->device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(m_ssil.staging, 0, staged.data(), staged.size() * 2);
    // Under a latched FlagPbr the stub's values are pre-exposure, which is what light.deferred adds
    composite::testing::set_exposure_units(true);
    return register_pass(stub_texture_desc(m_ssil));
  }

  StubTexture m_ao;
  StubTexture m_ssil;
  PassId m_aoId = InvalidPassId;
};

GpuPixel random_gpu_pixel(std::mt19937& rng) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
  GpuPixel p;
  p.albedo = {b(), b(), b()};
  p.ao = static_cast<uint8_t>(128 + byte(rng) / 2);
  p.roughness = static_cast<uint8_t>(80 + byte(rng) % 176);
  p.metal = byte(rng) % 4 == 0 ? b() : 0;
  p.cls = static_cast<uint8_t>(1 + byte(rng) % 9);
  p.flags = p.cls == 7 || p.cls == classes::ThinCutout ? uint8_t{gx::surface::ThinTranslucent} : uint8_t{0};
  p.category = byte(rng) % 3 == 0 ? 0x82 : 0x09;
  const auto shade = static_cast<uint8_t>(32 * (byte(rng) % 8));
  p.ambient = {shade, static_cast<uint8_t>(255 - shade), 128};
  p.normal = normalized({unit(rng) - 0.5, unit(rng) - 0.5, -0.4 - unit(rng)});
  p.z = 300.0 + 3000.0 * unit(rng);
  return p;
}

std::vector<AuroraLight> mixed_lights() {
  std::vector<AuroraLight> lights;
  lights.push_back(make_light(AURORA_LIGHT_DIRECTIONAL, {}, {0.3f, -0.5f, -0.8f}, 0.f));
  lights.push_back(make_light(AURORA_LIGHT_DIRECTIONAL, {}, {-0.2f, 0.1f, -1.f}, 0.f,
                              AURORA_CAT_DYNAMIC | AURORA_CAT_PLAYER,
                              AURORA_LIGHT_CAMERA_FILL | AURORA_LIGHT_NO_SPECULAR));
  std::mt19937 rng{0x9A1Fu};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  for (uint32_t i = 0; i < 40; ++i) {
    AuroraLight light = make_light(
        AURORA_LIGHT_POINT, {(unit(rng) - 0.5f) * 3000.f, (unit(rng) - 0.5f) * 2000.f, -300.f - 3000.f * unit(rng)}, {},
        400.f + 2500.f * unit(rng));
    light.id = 10 + i;
    light.color[0] = unit(rng) * 2.f;
    light.color[1] = unit(rng) * 2.f;
    light.color[2] = unit(rng) * 2.f;
    light.falloffExponent = i % 3 == 0 ? 1.5f : 2.f;
    lights.push_back(light);
  }
  return lights;
}

TEST_F(LightDeferredGpu, MatchesMirror) {
  std::mt19937 rng{0xDEF1u};
  auto hints = scene_hints(mixed_lights());
  hints->present |= SceneHasEnvironment;
  hints->environment.value.ambientColor[0] = 0.3f;
  hints->environment.value.ambientColor[1] = 0.35f;
  hints->environment.value.ambientColor[2] = 0.4f;
  set_hints(hints);
  std::vector<GpuPixel> pixels(size_t{96} * 64);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
  }
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(96, 64, pixels);
  deferred::testing::force_enabled(true);
  const auto hdr = run_hdr(pbr_settings());
  ASSERT_EQ(hdr.size(), inputs.size());
  const Mirror m = mirror_lights();
  const deferred::Params params = mirror_params();
  size_t bad = 0;
  double worst = 0.0;
  std::string first;
  for (size_t i = 0; i < hdr.size(); ++i) {
    const deferred::testing::Result want =
        deferred::testing::shade(inputs[i], m.header, m.lights, params, class_rows());
    ASSERT_EQ(openpbr::from_half(hdr[i][3]), want.alpha) << i;
    for (size_t c = 0; c < 3; ++c) {
      const double got = openpbr::from_half(hdr[i][c]);
      const double error = std::abs(got - want.rgb[c]);
      const double allowed = 2e-3 + 5e-3 * want.rgb[c];
      worst = std::max(worst, error / (1e-3 + want.rgb[c]));
      if (error > allowed && bad++ == 0) {
        first = std::to_string(i) + " channel " + std::to_string(c) + " got " + std::to_string(got) + " want " +
                std::to_string(want.rgb[c]);
      }
    }
  }
  EXPECT_EQ(bad, 0u) << first;
  RecordProperty("worst_relative", std::to_string(worst));
}

TEST_F(LightDeferredGpu, BaseVariantMatchesLayered) {
  std::mt19937 rng{0xBA5Eu};
  auto hints = scene_hints(mixed_lights());
  hints->present |= SceneHasEnvironment;
  hints->environment.value.ambientColor[0] = 0.3f;
  hints->environment.value.ambientColor[1] = 0.35f;
  hints->environment.value.ambientColor[2] = 0.4f;
  set_hints(hints);
  std::vector<GpuPixel> pixels(size_t{96} * 64);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
    if (openpbr::layered(class_rows()[p.cls])) {
      p.cls = classes::Default;
    }
  }
  static_cast<void>(stage_pixels(96, 64, pixels));
  deferred::testing::force_enabled(true);
  deferred::testing::force_layered(true);
  const auto layered = run_hdr(pbr_settings());
  EXPECT_EQ(deferred::testing::counters().base, 0u);
  deferred::testing::force_layered(false);
  const auto base = run_hdr(pbr_settings(), 2);
  EXPECT_EQ(deferred::testing::counters().base, 1u);
  ASSERT_EQ(base.size(), layered.size());
  size_t different = 0;
  for (size_t i = 0; i < base.size(); ++i) {
    different += base[i] != layered[i] ? 1 : 0;
  }
  EXPECT_EQ(different, 0u);
}

TEST_F(LightDeferredGpu, SubgroupWalkMatchesFallback) {
  if (!deferred::subgroup_walk(shaders::implicit_defines())) {
    GTEST_SKIP() << "the device has no Subgroups";
  }
  std::mt19937 rng{0x5B6Fu};
  auto hints = scene_hints(mixed_lights());
  hints->present |= SceneHasEnvironment;
  hints->environment.value.ambientColor[0] = 0.3f;
  hints->environment.value.ambientColor[1] = 0.35f;
  hints->environment.value.ambientColor[2] = 0.4f;
  set_hints(hints);
  std::vector<GpuPixel> pixels(size_t{96} * 64);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
  }
  static_cast<void>(stage_pixels(96, 64, pixels));
  deferred::testing::force_enabled(true);
  deferred::testing::force_subgroup_walk(false);
  const auto fallback = run_hdr(pbr_settings());
  deferred::testing::force_subgroup_walk(true);
  const auto walked = run_hdr(pbr_settings(), 2);
  ASSERT_EQ(walked.size(), fallback.size());
  size_t different = 0;
  for (size_t i = 0; i < walked.size(); ++i) {
    different += walked[i] != fallback[i] ? 1 : 0;
  }
  EXPECT_EQ(different, 0u);
}

TEST_F(LightDeferredGpu, FurnaceSceneIsOne) {
  std::mt19937 rng{0xF00Du};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    if (i % 5 == 0) {
      pixels[i].normal = normalized({unit(rng) - 0.5, unit(rng) - 0.5, 0.2 + unit(rng)});
    }
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  for (const deferred::Mode mode : {deferred::Mode::Furnace, deferred::Mode::FurnaceMetal}) {
    deferred::testing::set_mode(mode);
    const auto hdr = run_hdr(pbr_settings(), mode == deferred::Mode::Furnace ? 1 : 2);
    ASSERT_EQ(hdr.size(), pixels.size());
    double worst = 0.0;
    for (size_t i = 0; i < hdr.size(); ++i) {
      ASSERT_EQ(openpbr::from_half(hdr[i][3]), 1.f) << i;
      for (size_t c = 0; c < 3; ++c) {
        worst = std::max(worst, std::abs(openpbr::from_half(hdr[i][c]) - 1.0));
      }
    }
    EXPECT_LE(worst, 0.015) << static_cast<int>(mode);
  }
}

TEST_F(LightDeferredGpu, DirectFurnace256) {
  std::vector<AuroraLight> lights;
  constexpr uint32_t Count = 256;
  const double golden = Pi * (3.0 - std::sqrt(5.0));
  for (uint32_t i = 0; i < Count; ++i) {
    const double y = 1.0 - 2.0 * (i + 0.5) / Count;
    const double r = std::sqrt(1.0 - y * y);
    const double phi = golden * i;
    const Vec3 dir{r * std::cos(phi), y, r * std::sin(phi)};
    AuroraLight light = make_light(
        AURORA_LIGHT_POINT,
        {static_cast<float>(dir[0] * 1e4), static_cast<float>(dir[1] * 1e4), static_cast<float>(-500.0 - dir[2] * 1e4)},
        {}, 1e7f);
    light.id = i + 1;
    light.color[0] = light.color[1] = light.color[2] = 4.f / Count;
    lights.push_back(light);
  }
  set_hints(scene_hints(std::move(lights)));
  std::vector<GpuPixel> pixels(size_t{16} * 16);
  for (GpuPixel& p : pixels) {
    p.albedo = {255, 255, 255};
    p.normal = {0.0, 0.0, -1.0};
    p.z = 500.0;
  }
  stage_pixels(16, 16, pixels);
  deferred::testing::force_enabled(true);
  deferred::testing::set_mode(deferred::Mode::Direct);
  const auto hdr = run_hdr(pbr_settings());
  ASSERT_EQ(hdr.size(), pixels.size());
  for (size_t i = 0; i < hdr.size(); ++i) {
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(openpbr::from_half(hdr[i][c]), 1.0, 0.03) << i;
    }
  }
}

TEST_F(LightDeferredGpu, ThinCutoutPassesLightFromBehind) {
  // Leaf and grass cards seen from below face away from the sun; their thin class lets it through
  std::vector<GpuPixel> pixels(size_t{8} * 8);
  for (size_t i = 0; i < pixels.size(); ++i) {
    GpuPixel& p = pixels[i];
    p.albedo = {90, 200, 60};
    if (i % 2 == 0) {
      p.cls = classes::ThinCutout;
      p.flags = static_cast<uint8_t>(gx::surface::ThinTranslucent | gx::surface::TwoSided);
    }
  }
  const auto light = [](bool behind) {
    return scene_hints({make_light(AURORA_LIGHT_POINT, {0.f, 0.f, behind ? -3000.f : 1000.f}, {}, 1e7f)});
  };
  set_hints(light(false));
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(8, 8, pixels);
  deferred::testing::force_enabled(true);
  deferred::testing::set_mode(deferred::Mode::Direct);
  uint64_t frame = 1;
  for (const bool behind : {false, true}) {
    set_hints(light(behind));
    const auto hdr = run_hdr(pbr_settings(), frame++);
    ASSERT_EQ(hdr.size(), inputs.size());
    const Mirror m = mirror_lights();
    const deferred::Params params = mirror_params(deferred::Mode::Direct);
    for (size_t i = 0; i < hdr.size(); ++i) {
      const deferred::testing::Result want =
          deferred::testing::shade(inputs[i], m.header, m.lights, params, class_rows());
      const bool thin = i % 2 == 0;
      for (size_t c = 0; c < 3; ++c) {
        const double got = openpbr::from_half(hdr[i][c]);
        EXPECT_NEAR(got, want.rgb[c], 2e-3 + 5e-3 * want.rgb[c]) << i << " behind " << behind;
        if (behind) {
          EXPECT_EQ(got > 1e-3, thin) << i << " channel " << c;
        } else {
          EXPECT_GT(got, 1e-3) << i << " channel " << c;
        }
      }
    }
  }
}

TEST_F(LightDeferredGpu, SsilFurnaceIncrementIsExact) {
  std::mt19937 rng{0x551Fu};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  set_hints(scene_hints({}));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  std::vector<uint8_t> ao(pixels.size());
  std::vector<std::array<float, 4>> irr(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    pixels[i].receiver = static_cast<uint8_t>(i % 9 == 0 ? 0 : 1 + i % 3);
    ao[i] = static_cast<uint8_t>(i % 7 == 0 ? 255 : 40 + i % 215);
    const float lb = unit(rng);
    irr[i] = {lb, lb * unit(rng), lb * unit(rng), unit(rng)};
  }
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  deferred::testing::set_mode(deferred::Mode::FurnaceAo);
  stage_ao(64, 48, ao);
  const auto off = run_hdr(pbr_settings(), 1);
  const PassId ssilId = stage_ssil(64, 48, irr);
  const auto on = run_hdr(pbr_settings(), 2);
  unregister_pass(ssilId);
  ASSERT_EQ(off.size(), pixels.size());
  ASSERT_EQ(on.size(), pixels.size());
  double worstFar = 0.0;
  double worstIncrement = 0.0;
  size_t filled = 0;
  for (size_t i = 0; i < pixels.size(); ++i) {
    if (inputs[i].classId == 0) {
      continue;
    }
    const double as = inputs[i].receiver >= 1 ? std::pow(ao[i] / 255.0, 2.2) : 1.0;
    for (size_t c = 0; c < 3; ++c) {
      const double hOff = openpbr::from_half(off[i][c]);
      const double hOn = openpbr::from_half(on[i][c]);
      const double lb = openpbr::from_half(openpbr::to_half(irr[i][c]));
      const double want = inputs[i].receiver >= 1 ? (1.0 - as) * lb : 0.0;
      worstFar = std::max(worstFar, std::abs(hOff - as) - (0.02 * as + 0x1p-10));
      worstIncrement = std::max(worstIncrement, std::abs(hOn - hOff - want) - (0x1p-9 + 0x1p-10 * hOn));
      EXPECT_GE(hOn - hOff, -0x1p-10) << i;
      filled += want > 0.01 ? 1 : 0;
    }
  }
  std::printf("furnace-ao: worst excess over the far band %.6f, over the increment band %.6f, %zu filled channels\n",
              worstFar, worstIncrement, filled);
  EXPECT_LE(worstFar, 0.0);
  EXPECT_LE(worstIncrement, 0.0);
  EXPECT_GT(filled, 1000u);
}

TEST_F(LightDeferredGpu, SsilOffIsBitIdentical) {
  std::mt19937 rng{0x5520u};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  std::vector<uint8_t> ao(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    ao[i] = static_cast<uint8_t>(40 + i % 216);
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  stage_ao(64, 48, ao);
  const auto off = run_hdr(pbr_settings(), 1);
  const PassId ssilId = stage_ssil(64, 48, std::vector<std::array<float, 4>>(pixels.size()));
  const auto zero = run_hdr(pbr_settings(), 2);
  unregister_pass(ssilId);
  EXPECT_EQ(off, zero);
}

TEST_F(LightDeferredGpu, SsilNotMultipliedByAo) {
  std::mt19937 rng{0x5521u};
  set_hints(scene_hints({}));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    pixels[i].receiver = 1;
    pixels[i].ao = i % 2 == 0 ? 255 : 64;
  }
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  stage_ao(64, 48, std::vector<uint8_t>(pixels.size(), 128));
  const std::vector<std::array<float, 4>> irr(pixels.size(), {0.25f, 0.25f, 0.25f, 1.f});
  const auto off = run_hdr(pbr_settings(), 1);
  const PassId ssilId = stage_ssil(64, 48, irr);
  const auto on = run_hdr(pbr_settings(), 2);
  unregister_pass(ssilId);
  deferred::Params params = mirror_params();
  params.flags |= deferred::FlagSsil;
  size_t checked = 0;
  for (size_t i = 0; i < pixels.size(); ++i) {
    if (inputs[i].classId == 0) {
      continue;
    }
    deferred::testing::Pixel px = inputs[i];
    px.aoFinal = 128.0 / 255.0;
    px.ssil = {0.25, 0.25, 0.25, 1.0};
    const deferred::testing::Result want = deferred::testing::shade(px, {}, {}, params, class_rows());
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_NEAR(openpbr::from_half(on[i][c]), want.rgb[c], 2e-3 + 5e-3 * want.rgb[c]) << i;
    }
    ++checked;
  }
  EXPECT_GT(checked, 1000u);
}

TEST_F(LightDeferredGpu, SsilWaitsForCompositeExposure) {
  std::mt19937 rng{0x5522u};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    pixels[i].receiver = 1;
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  stage_ao(64, 48, std::vector<uint8_t>(pixels.size(), 128));
  const auto off = run_hdr(pbr_settings(), 1);
  const PassId ssilId = stage_ssil(64, 48, std::vector<std::array<float, 4>>(pixels.size(), {0.25f, 0.2f, 0.15f, 1.f}));
  composite::testing::set_exposure_units(false);
  const auto unexposed = run_hdr(pbr_settings(), 2);
  composite::testing::set_exposure_units(true);
  const auto exposed = run_hdr(pbr_settings(), 3);
  unregister_pass(ssilId);
  EXPECT_EQ(off, unexposed);
  EXPECT_NE(off, exposed);
}

class RenderLighting : public LightDeferredGpu {
protected:
  PassId stage_vis_sun(uint32_t width, uint32_t height, uint8_t value) {
    m_vis =
        StubTexture{.name = "t.vis_sun", .resource = resource::SunVisibility, .format = wgpu::TextureFormat::RG8Unorm};
    m_vis.rowBytes = (width * 2 + 255) & ~255u;
    std::vector<uint8_t> staged(size_t{m_vis.rowBytes} * height, 0);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        staged[size_t{y} * m_vis.rowBytes + x * 2] = value;
        staged[size_t{y} * m_vis.rowBytes + x * 2 + 1] = 255;
      }
    }
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = staged.size()};
    m_vis.staging = m_device->device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(m_vis.staging, 0, staged.data(), staged.size());
    return register_pass(stub_texture_desc(m_vis));
  }

  void TearDown() override {
    shadows::sun_visibility::testing::set_light_index(std::nullopt);
    LightDeferredGpu::TearDown();
  }

  StubTexture m_vis;
};

TEST_F(RenderLighting, SunVisibilityScalesOnlyTheShadowedLight) {
  std::mt19937 rng{0x5C15u};
  std::vector<AuroraLight> lights{make_light(AURORA_LIGHT_DIRECTIONAL, {}, {0.3f, -0.5f, -0.8f}, 0.f),
                                  make_light(AURORA_LIGHT_DIRECTIONAL, {}, {-0.4f, -0.6f, -0.7f}, 0.f)};
  lights[1].id = 2;
  lights[1].color[0] = 0.8f;
  lights[1].color[2] = 1.4f;
  set_hints(scene_hints(lights));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    pixels[i].receiver = static_cast<uint8_t>(i % 3 == 0 ? 1 : 2 + i % 2);
  }
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  const auto off = run_hdr(pbr_settings(), 1);
  shadows::sun_visibility::testing::set_light_index(1);
  const PassId white = stage_vis_sun(64, 48, 255);
  const auto lit = run_hdr(pbr_settings(), 2);
  unregister_pass(white);
  EXPECT_EQ(lit, off) << "an all-white vis.sun under FlagSunShadow keeps the unshadowed bits";
  const PassId half = stage_vis_sun(64, 48, 128);
  std::array<std::vector<std::array<uint16_t, 4>>, 2> shadowed;
  for (uint32_t index = 0; index < 2; ++index) {
    shadows::sun_visibility::testing::set_light_index(index);
    shadowed[index] = run_hdr(pbr_settings(), 3 + index);
  }
  unregister_pass(half);
  const double v = 128.0 / 255.0;
  const deferred::Params params = mirror_params();
  for (uint32_t index = 0; index < 2; ++index) {
    Mirror m = mirror_lights();
    for (size_t c = 0; c < 3; ++c) {
      m.lights[index].v2[c] = static_cast<float>(m.lights[index].v2[c] * v);
    }
    size_t checked = 0;
    size_t darker = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (pixels[i].receiver < 2) {
        EXPECT_EQ(shadowed[index][i], off[i]) << "code 1 pixel " << i << " light " << index;
        continue;
      }
      const deferred::testing::Result want =
          deferred::testing::shade(inputs[i], m.header, m.lights, params, class_rows());
      for (size_t c = 0; c < 3; ++c) {
        ASSERT_NEAR(openpbr::from_half(shadowed[index][i][c]), want.rgb[c], 2e-3 + 5e-3 * want.rgb[c])
            << i << " light " << index;
      }
      ++checked;
      darker += shadowed[index][i] != off[i] ? 1 : 0;
    }
    EXPECT_GT(checked, 1000u);
    EXPECT_GT(darker, checked / 2);
  }
  EXPECT_NE(shadowed[0], shadowed[1]);
}

TEST_F(LightDeferredGpu, AoDummyClamps) {
  std::mt19937 rng{0xA0D0u};
  auto hints = scene_hints({});
  hints->present |= SceneHasEnvironment;
  hints->environment.value.ambientColor[0] = hints->environment.value.ambientColor[1] =
      hints->environment.value.ambientColor[2] = 0.5f;
  set_hints(hints);
  std::vector<GpuPixel> pixels(size_t{61} * 37);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
    p.ao = 255;
    p.metal = 0;
    p.cls = 1;
    p.ambient = {128, 128, 128};
  }
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(61, 37, pixels);
  deferred::testing::force_enabled(true);
  const auto hdr = run_hdr(pbr_settings());
  ASSERT_EQ(hdr.size(), inputs.size());
  const deferred::Params params = mirror_params();
  for (size_t i = 0; i < hdr.size(); ++i) {
    const deferred::testing::Result want = deferred::testing::shade(inputs[i], {}, {}, params, class_rows());
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_NEAR(openpbr::from_half(hdr[i][c]), want.rgb[c], 2e-3 + 5e-3 * want.rgb[c]) << i;
    }
  }
}

struct SkyStub {
  lighting::sky_probe::Sh9Block block;
  std::array<float, 4> probe{};
};

void declare_sky_stub(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Prepare);
  builder.create(
      lighting::sky_probe::Probe,
      {.format = wgpu::TextureFormat::RGBA16Float, .sizeClass = SizeClass::Absolute, .width = 8, .height = 8},
      Access::ColorWrite);
  builder.create_buffer(lighting::sky_probe::Sh9, {.size = lighting::sky_probe::Sh9Bytes}, Access::CopyDst);
}

void encode_sky_stub(PassContext& ctx, const void* state) {
  const auto& stub = *static_cast<const SkyStub*>(state);
  const BufferView view = ctx.buffer(BufRef{1});
  if (view.buffer) {
    ctx.encoder().WriteBuffer(view.buffer, view.offset, reinterpret_cast<const uint8_t*>(&stub.block),
                              sizeof(stub.block));
  }
  const std::array<ColorTarget, 1> colors{
      {{.ref = TexRef{0}, .clear = wgpu::Color{stub.probe[0], stub.probe[1], stub.probe[2], 1.0}}}};
  if (const wgpu::RenderPassEncoder pass = ctx.begin_render(colors, nullptr, "test.sky_stub")) {
    pass.End();
  }
}

PassDesc sky_stub_desc(const SkyStub& stub) {
  return {
      .name = "test.sky_stub",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .order = 4,
      .declare = declare_sky_stub,
      .encode = encode_sky_stub,
      .state = &stub,
  };
}

TEST_F(LightDeferredGpu, PerPixelAmbientScalesSkyLight) {
  namespace sky = lighting::sky_probe;
  SkyStub stub;
  const std::array<std::array<float, 3>, 9> c{{{2.f, 2.2f, 2.4f},
                                               {0.8f, 0.6f, 0.4f},
                                               {0.1f, 0.f, -0.1f},
                                               {0.2f, 0.1f, 0.f},
                                               {0.f, 0.05f, 0.f},
                                               {0.05f, 0.f, 0.f},
                                               {-0.1f, -0.1f, -0.05f},
                                               {0.f, 0.f, 0.03f},
                                               {0.02f, 0.f, 0.f}}};
  sky::testing::Sh sh{};
  for (size_t i = 0; i < 9; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      stub.block.c[i][k] = c[i][k];
      sh[i][k] = c[i][k];
    }
  }
  const std::array<float, 3> amb{0.2f, 0.25f, 0.3f};
  std::copy_n(amb.data(), 3, stub.block.amb);
  stub.block.amb[3] = 1.f;
  stub.block.info[0] = stub.block.info[1] = stub.block.info[3] = 1.f;
  stub.probe = {0.5f, 0.4f, 0.3f, 1.f};
  set_hints(scene_hints({}));
  std::vector<GpuPixel> pixels(3);
  const Vec3 normal = normalized({0.3, 0.5, -0.8});
  for (GpuPixel& p : pixels) {
    p.normal = normal;
    p.z = 800.0;
  }
  pixels[0].ambient = {100, 120, 140};
  pixels[1].ambient = {50, 200, 30};
  pixels[2].ambient = {90, 90, 90};
  pixels[2].flags = gx::surface::BakedLight;
  const std::vector<deferred::testing::Pixel> inputs = stage_pixels(3, 1, pixels);
  const PassId stubId = register_pass(sky_stub_desc(stub));
  deferred::testing::force_enabled(true);
  deferred::testing::set_mode(deferred::Mode::Indirect);
  const auto hdr = run_hdr(pbr_settings());
  unregister_pass(stubId);
  ASSERT_EQ(hdr.size(), 3u);
  const Vec3 probeAmb{amb[0], amb[1], amb[2]};
  std::array<Vec3, 3> got{};
  std::array<Vec3, 3> unit{};
  for (size_t i = 0; i < 3; ++i) {
    const deferred::testing::Pixel& px = inputs[i];
    const double distance = std::sqrt(dot(px.position, px.position));
    const Vec3 v = scale(px.position, -1.0 / distance);
    const Vec3 n = deferred::testing::bend_normal(px.normal, v);
    openpbr::Surface surface{.roughness = px.roughness, .metalness = 0.0};
    Vec3 a{};
    for (size_t k = 0; k < 3; ++k) {
      surface.baseColor[k] = sky::testing::srgb_to_linear(px.albedo[k]);
      a[k] = sky::testing::srgb_to_linear(px.ambient[k]);
      got[i][k] = openpbr::from_half(hdr[i][k]);
    }
    const openpbr::Prepared p = openpbr::prepare(openpbr::testing::tables(), class_rows()[1], surface, n, v);
    const Vec3 world{n[0], n[1], -n[2]};
    const bool baked = i == 2;
    const Vec3 e = sky::testing::env_irradiance(sky::Mode::Pixel, sh, probeAmb, world, a, baked);
    const Vec3 s = sky::testing::pixel_scale(sky::Mode::Pixel, probeAmb, a);
    const Vec3 radiance{s[0] * stub.probe[0], s[1] * stub.probe[1], s[2] * stub.probe[2]};
    const Vec3 want = openpbr::eval_ambient(p, e, radiance);
    unit[i] =
        openpbr::eval_ambient(p, sky::testing::sh_irradiance(sh, world), {stub.probe[0], stub.probe[1], stub.probe[2]});
    for (size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(got[i][k], want[k], 2e-3 + 5e-3 * want[k]) << i << " channel " << k;
    }
    if (baked) {
      const Vec3 flat = openpbr::eval_ambient(p, scale(a, Pi), radiance);
      for (size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(got[i][k], flat[k], 2e-3 + 5e-3 * flat[k]) << "baked channel " << k;
      }
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    const double ratio = sky::testing::srgb_to_linear(pixels[0].ambient[k] / 255.0) /
                         sky::testing::srgb_to_linear(pixels[1].ambient[k] / 255.0);
    EXPECT_NEAR((got[0][k] / unit[0][k]) / (got[1][k] / unit[1][k]), ratio, 1.5e-3 * ratio) << k;
  }
}

TEST_F(LightDeferredGpu, ClustersOffIsBitIdentical) {
  std::mt19937 rng{0xC10Fu};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{128} * 72);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
  }
  stage_pixels(128, 72, pixels);
  deferred::testing::force_enabled(true);
  const auto clustered = run_hdr(pbr_settings(), 1);
  clusters::testing::set_brute_force(true);
  const auto brute = run_hdr(pbr_settings(), 2);
  clusters::testing::set_brute_force(std::nullopt);
  ASSERT_EQ(clustered.size(), brute.size());
  size_t differ = 0;
  size_t lit = 0;
  for (size_t i = 0; i < clustered.size(); ++i) {
    differ += clustered[i] != brute[i] ? 1 : 0;
    lit += clustered[i][3] != 0 ? 1 : 0;
  }
  EXPECT_EQ(differ, 0u);
  EXPECT_GT(lit, clustered.size() / 2);
}

TEST_F(LightDeferredGpu, ClassZeroUnlitAndSkyWriteZero) {
  std::mt19937 rng{0x5C1Eu};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = random_gpu_pixel(rng);
    switch (i % 5) {
    case 0:
      pixels[i].slot = false;
      break;
    case 1:
      pixels[i].flags = gx::surface::Unlit;
      break;
    case 2:
      pixels[i].z = 0.0;
      break;
    case 3:
      pixels[i].cls = 0;
      break;
    default:
      break;
    }
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  const auto hdr = run_hdr(pbr_settings());
  ASSERT_EQ(hdr.size(), pixels.size());
  for (size_t i = 0; i < hdr.size(); ++i) {
    if (i % 5 == 4) {
      EXPECT_EQ(openpbr::from_half(hdr[i][3]), 1.f) << i;
    } else {
      EXPECT_EQ(hdr[i], (std::array<uint16_t, 4>{})) << i;
    }
  }
}

class DeferredGraph : public LightDeferredGpu {};

TEST_F(DeferredGraph, PostShadowAlpha) {
  std::mt19937 rng{0x75u};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (GpuPixel& pixel : pixels) {
    pixel = random_gpu_pixel(rng);
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  const auto plain = run_hdr(pbr_settings());
  for (size_t i = 0; i < pixels.size(); i += 2) {
    pixels[i].entryBits = gx::surface::EntryPostShadow;
  }
  stage_pixels(64, 48, pixels);
  const auto flagged = run_hdr(pbr_settings(), 2);
  ASSERT_EQ(plain.size(), pixels.size());
  ASSERT_EQ(flagged.size(), pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    // Only alpha tells the composite which pixels the game's fakes never darkened
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_EQ(flagged[i][c], plain[i][c]) << i;
    }
    ASSERT_EQ(openpbr::from_half(plain[i][3]), 1.f) << i;
    ASSERT_EQ(openpbr::from_half(flagged[i][3]), i % 2 == 0 ? 0.75f : 1.f) << i;
  }
}

void declare_garbage(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Ao);
  builder.use_buffer(clusters::List, Access::CopyDst);
}

void encode_garbage(PassContext& ctx, const void*) {
  const BufferView view = ctx.buffer(BufRef{0});
  const clusters::LightListHeader header{.dirCount = 1000, .localCount = 100000, .flags = 0xFFFFFFFFu};
  ctx.encoder().WriteBuffer(view.buffer, view.offset, reinterpret_cast<const uint8_t*>(&header), 8);
}

TEST_F(LightDeferredGpu, GarbageHeaderIsClamped) {
  std::mt19937 rng{0x6A7Bu};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
  }
  stage_pixels(64, 48, pixels);
  register_pass({.name = "t.garbage",
                 .stage = Stage::OpaqueDone,
                 .kind = PassKind::Encoder,
                 .order = 5,
                 .declare = declare_garbage,
                 .encode = encode_garbage});
  deferred::testing::force_enabled(true);
  const auto hdr = run_hdr(pbr_settings());
  ASSERT_EQ(hdr.size(), pixels.size());
  for (size_t i = 0; i < hdr.size(); ++i) {
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_TRUE(std::isfinite(openpbr::from_half(hdr[i][c]))) << i;
    }
  }
}

TEST_F(LightDeferredGpu, StaleTransientOverwritten) {
  std::mt19937 rng{0x57A1u};
  set_hints(scene_hints(mixed_lights()));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (GpuPixel& p : pixels) {
    p = random_gpu_pixel(rng);
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  Readback reader{.resource = resource::SceneHdr};
  register_pass(readback_desc(reader, "t.hdr_read"));
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  StageData first = build(state, 1);
  ASSERT_EQ(encode(first), std::nullopt);
  ASSERT_TRUE(reader.texture);
  const auto lit = read_hdr(reader.texture);
  ASSERT_TRUE(std::ranges::any_of(lit, [](const auto& t) { return t[3] != 0; }));
  for (GpuPixel& p : pixels) {
    p.slot = false;
  }
  stage_pixels(64, 48, pixels);
  StageData second = build(state, 2);
  ASSERT_EQ(encode(second), std::nullopt);
  const auto cleared = read_hdr(reader.texture);
  for (size_t i = 0; i < cleared.size(); ++i) {
    ASSERT_EQ(cleared[i], (std::array<uint16_t, 4>{})) << i;
  }
}

constexpr std::string_view ProbeShader = R"(//#import aurora/openpbr

@group(0) @binding(2) var linear_clamp: sampler;
@group(1) @binding(0) var<storage, read> probe_classes: array<ClassParams>;
@group(1) @binding(1) var probe_lut3d: texture_3d<f32>;
@group(1) @binding(2) var probe_lut2d: texture_2d_array<f32>;
@group(1) @binding(3) var<storage, read_write> probe_out: array<vec4f>;

fn probe_hash(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

fn probe_unit(seed: ptr<function, u32>) -> f32 {
  *seed = probe_hash(*seed);
  return f32(*seed >> 8u) * (1.0 / 16777216.0);
}

fn probe_dir(seed: ptr<function, u32>) -> vec3f {
  return normalize(vec3f(probe_unit(seed) - 0.5, probe_unit(seed) - 0.5, probe_unit(seed) - 0.5) + vec3f(0.0, 0.0, 1e-3));
}

fn probe_row(index: u32) -> ClassParams {
  var row = probe_classes[min(index, 7u) + 1u];
  if (index == 8u) {
    row.coat = vec4f(0.9, 0.8, 0.7, 1.0);
    row.ior.z = 0.2;
  } else if (index == 9u) {
    row.fuzz = vec4f(0.9, 0.9, 1.0, 0.8);
    row.base.y = 0.6;
  } else if (index == 10u) {
    row.subsurface = vec4f(0.6, 0.9, 0.4, 0.6);
    row.base.x = 0.5;
    row.base.z = 0.2;
    row.misc.z = bitcast<f32>(1u);
    row.misc.w = 0.3;
  } else if (index == 11u) {
    row.specular = vec4f(0.8, 0.6, 0.9, 0.5);
  }
  return row;
}

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&probe_out) / 2u) {
    return;
  }
  var seed = i * 9781u + 1u;
  let n = probe_dir(&seed);
  var v = probe_dir(&seed);
  if (dot(n, v) < -0.3) {
    v = -v;
  }
  let l = probe_dir(&seed);
  let s = OpenPbrSurface(vec3f(probe_unit(&seed), probe_unit(&seed), probe_unit(&seed)), probe_unit(&seed),
                         select(0.0, probe_unit(&seed), probe_unit(&seed) < 0.3));
  let row = probe_row(probe_hash(seed) % 12u);
  let p = openpbr_prepare(s, row, n, v, probe_lut3d, probe_lut2d, linear_clamp);
  probe_out[i * 2u] = vec4f(openpbr_eval_light(p, l, true, probe_lut3d, probe_lut2d, linear_clamp), 0.0);
  probe_out[i * 2u + 1u] = vec4f(openpbr_eval_ambient(p, vec3f(3.0, 2.0, 1.0), vec3f(0.5, 1.0, 1.5)), 0.0);
}
)";

constexpr uint32_t ProbeCases = 4096;
constexpr std::string_view ProbeOut = "light.test_probe";

void declare_probe(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Lighting);
  builder.import_persistent(openpbr::Classes, openpbr::classes_handle(), Access::StorageRead);
  builder.import_persistent_texture(openpbr::Lut3d, openpbr::lut3d_handle(), openpbr::lut3d_desc(), Access::Sampled);
  builder.import_persistent_texture(openpbr::Lut2d, openpbr::lut2d_handle(), openpbr::lut2d_desc(), Access::Sampled);
  builder.create_buffer(ProbeOut, {.size = uint64_t{ProbeCases} * 32}, Access::StorageWrite);
}

constexpr std::array<std::string_view, 1> ProbeDefines{openpbr::LayeredDefine};

void request_probe(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({.label = "t.probe",
                                     .shader = "test/openpbr_probe",
                                     .entryPoint = "probe",
                                     .defines = ProbeDefines,
                                     .layout = setup.layout,
                                     .layoutKey = setup.layoutHash})
               .value;
}

void encode_probe(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pipeline || !pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(ProbeCases / 64);
  pass.End();
}

uint32_t probe_hash(uint32_t x) {
  uint32_t h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

float probe_unit(uint32_t& seed) {
  seed = probe_hash(seed);
  return static_cast<float>(seed >> 8u) * (1.f / 16777216.f);
}

Vec3 probe_dir(uint32_t& seed) {
  const float x = probe_unit(seed) - 0.5f;
  const float y = probe_unit(seed) - 0.5f;
  const float z = probe_unit(seed) - 0.5f + 1e-3f;
  return normalized({x, y, z});
}

classes::GpuRow probe_row(const classes::GpuRows& rows, uint32_t index) {
  classes::GpuRow row = rows[std::min(index, 7u) + 1];
  if (index == 8) {
    row.v[4] = 0.9f, row.v[5] = 0.8f, row.v[6] = 0.7f, row.v[7] = 1.f;
    row.v[22] = 0.2f;
  } else if (index == 9) {
    row.v[8] = 0.9f, row.v[9] = 0.9f, row.v[10] = 1.f, row.v[11] = 0.8f;
    row.v[25] = 0.6f;
  } else if (index == 10) {
    row.v[12] = 0.6f, row.v[13] = 0.9f, row.v[14] = 0.4f, row.v[15] = 0.6f;
    row.v[24] = 0.5f;
    row.v[26] = 0.2f;
    row.v[30] = std::bit_cast<float>(1u);
    row.v[31] = 0.3f;
  } else if (index == 11) {
    row.v[0] = 0.8f, row.v[1] = 0.6f, row.v[2] = 0.9f, row.v[3] = 0.5f;
  }
  return row;
}

class OpenPbrGpu : public LightDeferredGpu {};

TEST_F(OpenPbrGpu, LibraryMatchesMirror) {
  ASSERT_TRUE(shaders::register_source("test/openpbr_probe", ProbeShader));
  register_pass({.name = "t.probe",
                 .stage = Stage::OpaqueDone,
                 .kind = PassKind::Compute,
                 .order = 3,
                 .declare = declare_probe,
                 .encode = encode_probe,
                 .pipelines = request_probe});
  Readback reader{.resource = ProbeOut, .isBuffer = true};
  const wgpu::BufferDescriptor staging{.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
                                       .size = uint64_t{ProbeCases} * 32};
  reader.buffer = m_device->device.CreateBuffer(&staging);
  register_pass(readback_desc(reader, "t.probe_read"));
  set_hints(scene_hints({}));
  make_depth(64, 48);
  const auto graph = live_graph(pbr_settings());
  ASSERT_EQ(status(*graph, "t.probe"), CullReason::None);
  const FrameState state{.graph = graph};
  StageData stage = build(state, 1);
  ASSERT_EQ(encode(stage), std::nullopt);
  ASSERT_EQ(reader.calls, 1);
  const auto words = test::read_buffer(*m_device, reader.buffer, 0, uint64_t{ProbeCases} * 32);
  ASSERT_TRUE(words.has_value());
  const std::shared_ptr<const classes::GpuRows> rows = classes::gpu_rows();
  const openpbr::Tables& tables = openpbr::testing::tables();
  size_t bad = 0;
  size_t facing = 0;
  std::string first;
  for (uint32_t i = 0; i < ProbeCases; ++i) {
    uint32_t seed = i * 9781u + 1u;
    const Vec3 n = probe_dir(seed);
    Vec3 v = probe_dir(seed);
    if (dot(n, v) < -0.3) {
      v = scale(v, -1.0);
    }
    const Vec3 l = probe_dir(seed);
    openpbr::Surface s;
    s.baseColor = {probe_unit(seed), probe_unit(seed), probe_unit(seed)};
    s.roughness = probe_unit(seed);
    const float metal = probe_unit(seed);
    s.metalness = probe_unit(seed) < 0.3f ? metal : 0.0;
    const classes::GpuRow row = probe_row(*rows, probe_hash(seed) % 12u);
    const openpbr::Prepared p = openpbr::prepare(tables, row, s, n, v);
    const Vec3 light = openpbr::eval_light(tables, p, l);
    const Vec3 ambient = openpbr::eval_ambient(p, {3.0, 2.0, 1.0}, {0.5, 1.0, 1.5});
    facing += dot(n, v) < 0.0 && dot(n, v) > -0.3 ? 1 : 0;
    for (size_t c = 0; c < 3; ++c) {
      const float gotLight = std::bit_cast<float>((*words)[size_t{i} * 8 + c]);
      const float gotAmbient = std::bit_cast<float>((*words)[size_t{i} * 8 + 4 + c]);
      const bool ok = std::abs(gotLight - light[c]) <= 2e-3 + 5e-3 * std::abs(light[c]) &&
                      std::abs(gotAmbient - ambient[c]) <= 2e-3 + 2e-3 * std::abs(ambient[c]);
      if (!ok && bad++ == 0) {
        first = std::to_string(i) + " c " + std::to_string(c) + " light " + std::to_string(gotLight) + "/" +
                std::to_string(light[c]) + " ambient " + std::to_string(gotAmbient) + "/" + std::to_string(ambient[c]);
      }
    }
  }
  EXPECT_LE(bad, ProbeCases / 400) << first;
  EXPECT_GT(facing, 100u);
}

constexpr std::string_view LutProbeShader = R"(//#import aurora/openpbr

@group(0) @binding(2) var linear_clamp: sampler;
@group(1) @binding(0) var probe_lut3d: texture_3d<f32>;
@group(1) @binding(1) var probe_lut2d: texture_2d_array<f32>;
@group(1) @binding(2) var<storage, read_write> probe_out: array<vec4f>;

fn probe_hash(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

fn probe_unit(seed: ptr<function, u32>) -> f32 {
  *seed = probe_hash(*seed);
  return f32(*seed >> 8u) * (1.0 / 16777216.0);
}

// Indices past both edges pin the clamp, and texel nodes pin the layout free of filtering error
fn probe_index(seed: ptr<function, u32>, node: bool) -> f32 {
  let x = probe_unit(seed) * 35.0 - 2.0;
  return select(x, floor(x), node);
}

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&probe_out) / 4u) {
    return;
  }
  var seed = i * 7919u + 3u;
  let node = i % 4u == 0u;
  let x = probe_index(&seed, node);
  let y = probe_index(&seed, node);
  let z = probe_index(&seed, node);
  probe_out[i * 4u] = vec4f(openpbr_lut3d(probe_lut3d, linear_clamp, x / OPENPBR_TABLE_MAX, y, z), 0.0, 0.0);
  for (var layer = 0u; layer < 3u; layer++) {
    probe_out[i * 4u + 1u + layer] = openpbr_lut2d(probe_lut2d, linear_clamp, x, y, layer);
  }
}
)";

constexpr uint32_t LutCases = 1024;
constexpr uint64_t LutProbeBytes = uint64_t{LutCases} * 64;
constexpr std::string_view LutProbeOut = "light.test_lut_probe";

void declare_lut_probe(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Lighting);
  builder.import_persistent_texture(openpbr::Lut3d, openpbr::lut3d_handle(), openpbr::lut3d_desc(), Access::Sampled);
  builder.import_persistent_texture(openpbr::Lut2d, openpbr::lut2d_handle(), openpbr::lut2d_desc(), Access::Sampled);
  builder.create_buffer(LutProbeOut, {.size = LutProbeBytes}, Access::StorageWrite);
}

void request_lut_probe(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({.label = "t.lut_probe",
                                     .shader = "test/openpbr_lut_probe",
                                     .entryPoint = "probe",
                                     .layout = setup.layout,
                                     .layoutKey = setup.layoutHash})
               .value;
}

void encode_lut_probe(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pipeline || !pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(LutCases / 64);
  pass.End();
}

template <size_t Axes, typename Sample>
double texel_spread(const std::array<double, Axes>& index, const Sample& sample) {
  double lo = std::numeric_limits<double>::infinity();
  double hi = -lo;
  for (uint32_t corner = 0; corner < (1u << Axes); ++corner) {
    std::array<double, Axes> at{};
    for (size_t a = 0; a < Axes; ++a) {
      const double base = std::floor(std::clamp(index[a], 0.0, 31.0));
      at[a] = std::min(base + ((corner >> a) & 1u), 31.0);
    }
    const double value = sample(at);
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  }
  return hi - lo;
}

TEST_F(OpenPbrGpu, LutsSampleAsMirror) {
  ASSERT_TRUE(shaders::register_source("test/openpbr_lut_probe", LutProbeShader));
  register_pass({.name = "t.lut_probe",
                 .stage = Stage::OpaqueDone,
                 .kind = PassKind::Compute,
                 .order = 3,
                 .declare = declare_lut_probe,
                 .encode = encode_lut_probe,
                 .pipelines = request_lut_probe});
  Readback reader{.resource = LutProbeOut, .isBuffer = true};
  const wgpu::BufferDescriptor staging{.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
                                       .size = LutProbeBytes};
  reader.buffer = m_device->device.CreateBuffer(&staging);
  register_pass(readback_desc(reader, "t.lut_probe_read"));
  set_hints(scene_hints({}));
  make_depth(64, 48);
  const auto graph = live_graph(pbr_settings());
  ASSERT_EQ(status(*graph, "t.lut_probe"), CullReason::None);
  const FrameState state{.graph = graph};
  StageData stage = build(state, 1);
  ASSERT_EQ(encode(stage), std::nullopt);
  ASSERT_EQ(reader.calls, 1);
  const auto words = test::read_buffer(*m_device, reader.buffer, 0, LutProbeBytes);
  ASSERT_TRUE(words.has_value());
  const openpbr::Tables& tables = openpbr::testing::tables();
  size_t bad = 0;
  size_t checked = 0;
  double worst = 0.0;
  std::string first;
  const auto check = [&](uint32_t i, size_t word, double want, double spread, bool exact) {
    const double got = std::bit_cast<float>((*words)[size_t{i} * 16 + word]);
    const double error = std::abs(got - want);
    // Filtering returns f16 precision from weights with few fractional bits, so the bound follows the blended texels
    const double bound = exact ? 0.0 : 1e-6 + f16_ulp(want) + spread / 64.0;
    if (!exact) {
      worst = std::max(worst, error / bound);
    }
    ++checked;
    if (!(error <= bound) && bad++ == 0) {
      first = std::to_string(i) + " word " + std::to_string(word) + " got " + std::to_string(got) + " want " +
              std::to_string(want) + " spread " + std::to_string(spread);
    }
  };
  for (uint32_t i = 0; i < LutCases; ++i) {
    uint32_t seed = i * 7919u + 3u;
    const bool node = i % 4 == 0;
    const auto index = [&] {
      const float x = probe_unit(seed) * 35.f - 2.f;
      return node ? std::floor(x) : x;
    };
    const float x = index();
    const float y = index();
    const float z = index();
    const std::array<double, 3> at3{openpbr::cos_index(x / 31.f), y, z};
    const std::array<double, 2> want3 = openpbr::sample_lut3d(tables, at3[0], at3[1], at3[2]);
    for (size_t c = 0; c < 2; ++c) {
      const double spread = texel_spread(
          at3, [&](const std::array<double, 3>& a) { return openpbr::sample_lut3d(tables, a[0], a[1], a[2])[c]; });
      check(i, c, want3[c], spread, false);
    }
    const std::array<double, 2> at2{x, y};
    for (uint32_t layer = 0; layer < 3; ++layer) {
      const std::array<double, 4> want2 = openpbr::sample_lut2d(tables, x, y, layer);
      for (size_t c = 0; c < 4; ++c) {
        const double spread = texel_spread(
            at2, [&](const std::array<double, 2>& a) { return openpbr::sample_lut2d(tables, a[0], a[1], layer)[c]; });
        check(i, 4 + layer * 4 + c, want2[c], spread, node);
      }
    }
  }
  RecordProperty("worst", std::to_string(worst));
  std::printf("luts against the mirror: worst %.3f of the bound over %zu values\n", worst, checked);
  EXPECT_EQ(bad, 0u) << first;
}

TEST_F(OpenPbrGpu, ClassBufferFollowsEpoch) {
  set_hints(scene_hints({make_light(AURORA_LIGHT_DIRECTIONAL, {}, {0.f, 0.f, -1.f}, 0.f)}));
  std::vector<GpuPixel> pixels(size_t{64} * 48);
  for (GpuPixel& p : pixels) {
    p.cls = 200;
    p.albedo = {255, 255, 255};
  }
  stage_pixels(64, 48, pixels);
  deferred::testing::force_enabled(true);
  const uint64_t before = openpbr::testing::uploaded_epoch();
  const auto plain = run_hdr(pbr_settings(), 1);
  EXPECT_EQ(openpbr::testing::uploaded_epoch(), before);
  OpenPbrParams params;
  params.baseWeight = 0.25f;
  ASSERT_TRUE(classes::register_class(200, "test/dim", params));
  classes::apply_frame_state(*material::latched_snapshot());
  ASSERT_NE(classes::epoch(), before);
  const auto dim = run_hdr(pbr_settings(), 2);
  EXPECT_EQ(openpbr::testing::uploaded_epoch(), classes::epoch());
  ASSERT_EQ(plain.size(), dim.size());
  for (size_t i = 0; i < plain.size(); ++i) {
    ASSERT_LT(openpbr::from_half(dim[i][0]), openpbr::from_half(plain[i][0]) * 0.5f) << i;
  }
}
} // namespace
} // namespace aurora::render::detail

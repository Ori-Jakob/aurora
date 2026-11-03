#include <gtest/gtest.h>

#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/gx_ext/surface_table.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/lighting/clusters.hpp"
#include "render/lighting/deferred.hpp"
#include "render/lighting/openpbr.hpp"
#include "render/lighting/sky_probe.hpp"
#include "render/material/classes.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/post/composite.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_camera.hpp"
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
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace clusters = lighting::clusters;
namespace deferred = lighting::deferred;
namespace openpbr = lighting::openpbr;
namespace sky = lighting::sky_probe;
using Vec3 = sky::testing::Vec3;
using sky::Hook;
using sky::HookMode;

constexpr double Pi = std::numbers::pi;
constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float Near = 10.f;
constexpr float Far = 100000.f;
constexpr uint32_t Size = sky::ProbeSize;

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 normalized(const Vec3& a) {
  const double l = std::sqrt(dot(a, a));
  return {a[0] / l, a[1] / l, a[2] / l};
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

// A camera pitched up so the frame shows a wide band of the upper hemisphere
Mat3x4<float> pitched_view(double degrees, float scale = 1.f) {
  const auto c = static_cast<float>(std::cos(degrees * Pi / 180.0));
  const auto s = static_cast<float>(std::sin(degrees * Pi / 180.0));
  return {Vec4<float>{scale, 0.f, 0.f, 0.f}, Vec4<float>{0.f, c * scale, s * scale, 0.f},
          Vec4<float>{0.f, -s * scale, c * scale, 0.f}};
}

struct Env {
  float skyVisibility = 1.f;
  std::array<float, 3> ambient{0.4f, 0.45f, 0.5f};
  std::array<float, 3> fog{0.6f, 0.65f, 0.7f};
};

std::shared_ptr<SceneHints> scene_hints(const Mat3x4<float>& view, std::optional<Env> env = Env{},
                                        bool motionValid = true) {
  auto hints = std::make_shared<SceneHints>();
  const Mat4x4<float> proj = perspective(Near, Far);
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .view = view,
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
  hints->present = SceneHasMainCamera | SceneHasLights;
  AuroraLight light{};
  light.type = AURORA_LIGHT_DIRECTIONAL;
  light.direction[1] = -1.f;
  light.color[0] = light.color[1] = light.color[2] = 1.f;
  light.spotInnerCos = 1.f;
  light.spotOuterCos = -1.f;
  hints->lights = {light};
  if (env) {
    hints->present |= SceneHasEnvironment;
    AuroraEnvironment& e = hints->environment.value;
    e.skyVisibility = env->skyVisibility;
    std::ranges::copy(env->ambient, e.ambientColor);
    std::ranges::copy(env->fog, e.fogColor);
    e.unitsPerMeter = 100.f;
  }
  hints->motionValid = motionValid;
  return hints;
}

Settings pbr_settings(float legacyBlend = 0.f) {
  Settings settings;
  settings.pbr = true;
  settings.legacyBlend = legacyBlend;
  return settings;
}

Vec3 lin3(const std::array<float, 3>& stored) {
  return {sky::testing::srgb_to_linear(stored[0]), sky::testing::srgb_to_linear(stored[1]),
          sky::testing::srgb_to_linear(stored[2])};
}

TEST(SkyProbeRules, OctahedralRoundTrip) {
  std::mt19937 rng{0x0C7u};
  std::normal_distribution<double> gauss;
  for (int i = 0; i < 4096; ++i) {
    const Vec3 d = normalized({gauss(rng), gauss(rng), gauss(rng)});
    const auto uv = sky::testing::oct_encode(d);
    const Vec3 back = sky::testing::oct_decode(uv[0], uv[1]);
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_NEAR(back[c], d[c], 1e-9) << i;
    }
  }
  const auto up = sky::testing::oct_encode({0.0, 1.0, 0.0});
  EXPECT_DOUBLE_EQ(up[0], 0.5);
  EXPECT_DOUBLE_EQ(up[1], 0.5);
  EXPECT_NEAR(sky::testing::oct_decode(0.0, 0.0)[1], -1.0, 1e-12);
}

TEST(SkyProbeRules, TexelWeightsSumTo4Pi) {
  for (const uint32_t size : {32u, 128u}) {
    double total = 0.0;
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        total += sky::testing::texel_weight((x + 0.5) / size, (y + 0.5) / size, size);
      }
    }
    EXPECT_NEAR(total, 4.0 * Pi, 4.0 * Pi * 0.01) << size;
  }
}

TEST(SkyProbeRules, ShOfConstantIsPiTimesConstant) {
  const Vec3 v{0.25, 0.5, 2.0};
  const std::vector<Vec3> texels(size_t{sky::ShSize} * sky::ShSize, v);
  const sky::testing::Sh sh = sky::testing::sh_project(texels, sky::ShSize, v);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(sh[0][c], Pi * std::sqrt(4.0 * Pi) * v[c], 1e-12);
    for (size_t i = 1; i < 9; ++i) {
      EXPECT_EQ(sh[i][c], 0.0) << i;
    }
  }
  const sky::testing::Sh quadrature = sky::testing::sh_project(texels, sky::ShSize);
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(quadrature[0][c], sh[0][c], 1e-6 * sh[0][c]);
    for (size_t i = 1; i < 9; ++i) {
      EXPECT_NEAR(quadrature[i][c], 0.0, 1e-3 * sh[0][c]) << i;
    }
  }
  std::mt19937 rng{0x5A5u};
  std::normal_distribution<double> gauss;
  for (int i = 0; i < 64; ++i) {
    const Vec3 e = sky::testing::sh_irradiance(sh, normalized({gauss(rng), gauss(rng), gauss(rng)}));
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(e[c], Pi * v[c], 1e-9);
    }
  }
}

TEST(SkyProbeRules, ProjectionInvertsViewPosition) {
  StageCamera cam;
  auto hints = scene_hints(pitched_view(20.0));
  hints->jitterPx[0] = 0.3f;
  hints->jitterPx[1] = -0.2f;
  ASSERT_TRUE(fill_stage_camera(hints.get(), cam));
  constexpr double W = 640.0;
  constexpr double H = 360.0;
  std::mt19937 rng{0x1E5u};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int i = 0; i < 256; ++i) {
    const double u = unit(rng);
    const double v = unit(rng);
    const double z = 10.0 + 5000.0 * unit(rng);
    const double cx = cam.p02 - 2.0 * cam.jitterPx[0] / W;
    const double cy = cam.p12 + 2.0 * cam.jitterPx[1] / H;
    const Vec3 p{(2.0 * u - 1.0 + cx) * z / cam.p00, (1.0 - 2.0 * v + cy) * z / cam.p11, z};
    const auto back = sky::testing::project_fwd(cam, p, W, H);
    EXPECT_NEAR(back[0], u, 1e-9);
    EXPECT_NEAR(back[1], v, 1e-9);
    EXPECT_EQ(back[2], 1.0);
  }
  EXPECT_EQ(sky::testing::project_fwd(cam, {0.0, 0.0, -1.0}, W, H)[2], 0.0);
}

TEST(SkyProbeRules, ViewRotationIsOrthonormal) {
  Mat3x4<float> view = pitched_view(35.0, 3.f);
  view.m0[1] = 0.4f;
  view.m0[3] = 120.f;
  const auto hints = scene_hints(view);
  ViewRotation r;
  ASSERT_TRUE(fill_view_rotation(hints.get(), r));
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double d = r.rows[a][0] * r.rows[b][0] + r.rows[a][1] * r.rows[b][1] + r.rows[a][2] * r.rows[b][2];
      EXPECT_NEAR(d, a == b ? 1.0 : 0.0, 1e-6) << a << "," << b;
    }
    EXPECT_EQ(r.rows[a][3], 0.f);
  }
  const Vec3 w{0.2, 0.7, -0.4};
  const Vec3 back = sky::testing::world_dir(r, sky::testing::view_dir_fwd(r, w));
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(back[c], w[c], 1e-6);
  }
  const auto flat =
      scene_hints({Vec4<float>{1.f, 0.f, 0.f, 0.f}, Vec4<float>{2.f, 0.f, 0.f, 0.f}, Vec4<float>{0.f, 0.f, 1.f, 0.f}});
  EXPECT_FALSE(fill_view_rotation(flat.get(), r));
  EXPECT_EQ(r.rows[0][0], 0.f);
  EXPECT_FALSE(fill_view_rotation(nullptr, r));
}

TEST(SkyProbeRules, ObserveNeedsSkyDrawsAndVisibility) {
  const auto observe = [](const SceneHints* hints, uint32_t draws, bool constant) {
    sky::ProjectParams p;
    sky::fill_project_params(hints, draws, constant, 0.5f, p);
    EXPECT_EQ(p.alpha, 0.5f);
    return (p.flags & sky::ProjectObserve) != 0;
  };
  const auto open = scene_hints(pitched_view(10.0));
  EXPECT_TRUE(observe(open.get(), 3, false));
  EXPECT_FALSE(observe(open.get(), 0, false));
  EXPECT_FALSE(observe(open.get(), 3, true));
  const auto indoors = scene_hints(pitched_view(10.0), Env{.skyVisibility = 0.f});
  EXPECT_FALSE(observe(indoors.get(), 3, false));
  const auto noEnv = scene_hints(pitched_view(10.0), std::nullopt);
  EXPECT_TRUE(observe(noEnv.get(), 3, false));
  const auto degenerate = scene_hints({});
  EXPECT_FALSE(observe(degenerate.get(), 3, false));
  sky::ProjectParams p;
  sky::fill_project_params(scene_hints(pitched_view(10.0), Env{}, false).get(), 3, false, 0.5f, p);
  EXPECT_NE(p.flags & sky::ProjectReset, 0u);
  sky::fill_project_params(open.get(), 3, false, 0.5f, p);
  EXPECT_EQ(p.flags & sky::ProjectReset, 0u);
  EXPECT_NE(p.camera.flags & CameraValid, 0u);
  EXPECT_NE(p.rotation.rows[1][1], 0.f);
}

TEST(SkyProbeRules, LowerHemisphereNeverObserved) {
  for (uint32_t y = 0; y < Size; ++y) {
    for (uint32_t x = 0; x < Size; ++x) {
      const double u = (x + 0.5) / Size;
      const double v = (y + 0.5) / Size;
      const bool inner = std::abs(u * 2.0 - 1.0) + std::abs(v * 2.0 - 1.0) <= 1.0;
      ASSERT_EQ(sky::testing::oct_decode(u, v)[1] >= 0.0, inner) << x << "," << y;
    }
  }
}

TEST(SkyProbeRules, GainMatchesAmbient) {
  const Vec3 a{0.2, 0.3, 0.4};
  const double y = sky::testing::luminance(a);
  EXPECT_NEAR(sky::testing::gain(sky::Policy::MatchAmbient, true, a, 0.5), y / 0.5, 1e-12);
  EXPECT_EQ(sky::testing::gain(sky::Policy::MatchAmbient, true, a, 1e9), double{sky::GainMin});
  EXPECT_EQ(sky::testing::gain(sky::Policy::MatchAmbient, true, a, 0.0), double{sky::GainMax});
  EXPECT_EQ(sky::testing::gain(sky::Policy::Physical, true, a, 0.5), 1.0);
  EXPECT_EQ(sky::testing::gain(sky::Policy::MatchAmbient, false, a, 0.5), 1.0);
  EXPECT_EQ(sky::policy_of({.mode = HookMode::Physical}), sky::Policy::Physical);
  EXPECT_EQ(sky::policy_of({.mode = HookMode::Global}), sky::Policy::MatchAmbient);
}

TEST(SkyProbeRules, AlphaFollowsDt) {
  EXPECT_NEAR(sky::blend_alpha(1000.f / 60.f), 1.0 - std::exp(-1.0 / 60.0 / 0.1), 1e-6);
  EXPECT_EQ(sky::blend_alpha(1000.f), sky::blend_alpha(sky::MaxDeltaMs));
  EXPECT_EQ(sky::blend_alpha(0.f), 0.f);
  EXPECT_EQ(sky::blend_alpha(std::numeric_limits<float>::quiet_NaN()), 0.f);
  EXPECT_LT(sky::blend_alpha(8.f), sky::blend_alpha(33.f));
}

TEST(SkyProbeRules, EnvValuesSanitised) {
  sky::FilterParams p;
  auto hints = scene_hints(pitched_view(0.0));
  AuroraEnvironment& e = hints->environment.value;
  e.skyVisibility = std::numeric_limits<float>::infinity();
  e.ambientColor[0] = std::numeric_limits<float>::quiet_NaN();
  e.ambientColor[1] = 2.f;
  e.ambientColor[2] = -1.f;
  e.fogColor[0] = 0.25f;
  EXPECT_TRUE(sky::fill_filter_params(hints.get(), {}, true, p));
  EXPECT_EQ(p.fog[3], 0.f);
  EXPECT_EQ(p.ambient[0], 0.f);
  EXPECT_EQ(p.ambient[1], 1.f);
  EXPECT_EQ(p.ambient[2], 0.f);
  EXPECT_EQ(p.fog[0], 0.25f);
  EXPECT_EQ(p.flags, uint32_t{sky::FilterHasEnv | sky::FilterObserved});
  sky::fill_filter_params(scene_hints(pitched_view(0.0), std::nullopt).get(),
                          {.mode = HookMode::Constant, .constant = 2.f}, false, p);
  EXPECT_EQ(p.fog[3], 1.f);
  EXPECT_EQ(p.ambient[0], 0.f);
  EXPECT_EQ(p.flags, uint32_t{sky::FilterConstant});
  EXPECT_EQ(p.constant, 2.f);
  sky::fill_filter_params(hints.get(), {.mode = HookMode::Physical}, false, p);
  EXPECT_EQ(p.flags, uint32_t{sky::FilterHasEnv | sky::FilterPhysical});
  sky::fill_filter_params(hints.get(), {.mode = HookMode::Global}, false, p);
  EXPECT_EQ(p.flags, uint32_t{sky::FilterHasEnv | sky::FilterGlobal});
}

TEST(SkyProbeRules, EnvHookParses) {
  EXPECT_EQ(sky::parse_hook("off"), (Hook{.mode = HookMode::Off}));
  EXPECT_EQ(sky::parse_hook("physical"), (Hook{.mode = HookMode::Physical}));
  EXPECT_EQ(sky::parse_hook("global"), (Hook{.mode = HookMode::Global}));
  EXPECT_EQ(sky::parse_hook("constant:1"), (Hook{.mode = HookMode::Constant, .constant = 1.f}));
  EXPECT_EQ(sky::parse_hook("constant:0.25"), (Hook{.mode = HookMode::Constant, .constant = 0.25f}));
  for (const std::string_view bad : {"", "on", "Off", "constant", "constant:", "constant:-1", "constant:nan",
                                     "constant:1x", "constant:inf", "constant:70000"}) {
    EXPECT_EQ(sky::parse_hook(bad), std::nullopt) << bad;
  }
}

TEST(SkyProbeRules, ModeSelection) {
  const Vec3 a{0.1, 0.1, 0.1};
  const Vec3 black{1e-4, 1e-4, 1e-4};
  EXPECT_EQ(sky::testing::mode_for({}, true, a), sky::Mode::Pixel);
  EXPECT_EQ(sky::testing::mode_for({}, true, black), sky::Mode::PixelFlat);
  EXPECT_EQ(sky::testing::mode_for({}, false, a), sky::Mode::PixelFlat);
  for (const HookMode mode : {HookMode::Constant, HookMode::Physical, HookMode::Global}) {
    EXPECT_EQ(sky::testing::mode_for({.mode = mode}, true, a), sky::Mode::Raw);
  }
}

TEST(SkyProbeRules, PixelScaleIsPerChannelRatio) {
  const Vec3 amb{0.2, 0.1, 0.4};
  const Vec3 s = sky::testing::pixel_scale(sky::Mode::Pixel, amb, {0.1, 0.3, 0.4});
  EXPECT_NEAR(s[0], 0.5, 1e-12);
  EXPECT_NEAR(s[1], 3.0, 1e-12);
  EXPECT_NEAR(s[2], 1.0, 1e-12);
  EXPECT_EQ(sky::testing::pixel_scale(sky::Mode::Pixel, amb, {100.0, 0.0, -1.0}), (Vec3{16.0, 0.0, 0.0}));
  const Vec3 floor = sky::testing::pixel_scale(sky::Mode::Pixel, {0.0, 0.0, 0.0}, {0x1p-12, 0.0, 0.0});
  EXPECT_NEAR(floor[0], 0.25, 1e-12);
  EXPECT_EQ(sky::testing::pixel_scale(sky::Mode::Raw, amb, {0.1, 0.3, 0.4}), (Vec3{1.0, 1.0, 1.0}));
  EXPECT_EQ(sky::testing::pixel_scale(sky::Mode::PixelFlat, amb, {0.1, 0.3, 0.4}), (Vec3{1.0, 1.0, 1.0}));
}

TEST(SkyProbeRules, InteriorIrradianceIsPixelAmbient) {
  const Vec3 linA = lin3({0.35f, 0.3f, 0.25f});
  const std::vector<Vec3> texels(size_t{sky::ShSize} * sky::ShSize, linA);
  const sky::testing::Sh sh = sky::testing::sh_project(texels, sky::ShSize, linA);
  std::mt19937 rng{0x1A7u};
  std::normal_distribution<double> gauss;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int i = 0; i < 64; ++i) {
    const Vec3 n = normalized({gauss(rng), gauss(rng), gauss(rng)});
    const Vec3 a{0.7 * unit(rng), 0.7 * unit(rng), 0.7 * unit(rng)};
    const Vec3 e = sky::testing::env_irradiance(sky::Mode::Pixel, sh, linA, n, a, false);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(e[c], Pi * a[c], 1e-9);
    }
  }
}

TEST(SkyProbeRules, BakedLightIsFlat) {
  std::vector<Vec3> texels(size_t{sky::ShSize} * sky::ShSize);
  for (uint32_t y = 0; y < sky::ShSize; ++y) {
    for (uint32_t x = 0; x < sky::ShSize; ++x) {
      const Vec3 d = sky::testing::oct_decode((x + 0.5) / sky::ShSize, (y + 0.5) / sky::ShSize);
      texels[size_t{y} * sky::ShSize + x] = {0.2 + std::max(d[1], 0.0), 0.3, 0.1 + 0.05 * d[0]};
    }
  }
  const sky::testing::Sh sh = sky::testing::sh_project(texels, sky::ShSize);
  const Vec3 amb{0.2, 0.3, 0.1};
  const Vec3 a{0.05, 0.6, 0.3};
  const Vec3 n = normalized({0.3, 0.8, -0.2});
  for (const sky::Mode mode : {sky::Mode::Pixel, sky::Mode::PixelFlat}) {
    const Vec3 e = sky::testing::env_irradiance(mode, sh, amb, n, a, true);
    for (size_t c = 0; c < 3; ++c) {
      EXPECT_NEAR(e[c], Pi * a[c], 1e-12);
    }
  }
  const Vec3 raw = sky::testing::env_irradiance(sky::Mode::Raw, sh, amb, n, a, true);
  const Vec3 shE = sky::testing::sh_irradiance(sh, n);
  EXPECT_EQ(raw, shE);
  const Vec3 lit = sky::testing::env_irradiance(sky::Mode::Pixel, sh, amb, n, a, false);
  EXPECT_NEAR(lit[1], 2.0 * shE[1], 1e-12);
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

void declare_hdr_sink(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Exposure);
  builder.use(resource::SceneHdr, Access::CopySrc);
  builder.side_effect();
}

void encode_nothing(PassContext&, const void*) {}

PassDesc hdr_sink_desc() {
  return {
      .name = "t.hdr_sink",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 8,
      .declare = declare_hdr_sink,
      .encode = encode_nothing,
  };
}

void declare_table_stub(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Prepare);
  builder.create_buffer(resource::SurfaceTable, {.size = gx_ext::surface_table::TableBytes}, Access::CopyDst);
}

PassDesc table_stub_desc() {
  return {
      .name = "test.surface_table",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .order = 0,
      .declare = declare_table_stub,
      .encode = encode_nothing,
  };
}

using Half4 = std::array<float, 4>;

class SkyDeviceTest : public ::testing::Test {
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
    material::classes::testing::reset();
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    sky::shutdown();
    sky::testing::reset();
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
    sky::initialize();
    register_pass(table_stub_desc());
    register_pass(hdr_sink_desc());
    sky::testing::set_sky_draws(1);
    sky::testing::set_alpha(0.25f);
    deferred::testing::force_enabled(false);
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

  void set_hints(const std::shared_ptr<const SceneHints>& hints) {
    m_hints = hints;
    clusters::testing::set_hints_override(hints);
    deferred::testing::set_hints_override(hints);
    sky::testing::set_hints_override(hints);
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
    clusters::on_latch(latched, graph);
    deferred::on_latch(latched, graph);
    sky::on_latch(latched, graph);
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
        .sceneDepth = m_depth ? test::scene_depth(m_depth) : nullptr,
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

  void bind(StageData& opaque) const {
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

  std::optional<std::string> encode(StageData& opaque) const {
    if (opaque.any_enabled()) {
      bind(opaque);
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
};

class SkyProbeGraph : public SkyDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 48, 1}})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    m_depth = test::make_depth(*m_device, 64, 48);
    set_hints(scene_hints(pitched_view(30.0)));
  }

  static std::array<uint32_t, MaxPassPipelines> node_pipelines(const CompiledGraph& graph, std::string_view name) {
    const NodeId node = graph.find_node(name);
    return node != InvalidNode ? graph.nodes[node].pipelineIds : std::array<uint32_t, MaxPassPipelines>{};
  }
};

TEST_F(SkyProbeGraph, DeclaresOnlyWithPbrAndSurface) {
  const auto graph = compile(pbr_settings());
  EXPECT_EQ(status(*graph, sky::ProjectPass), CullReason::None);
  EXPECT_EQ(status(*graph, sky::FilterPass), CullReason::None);
  EXPECT_EQ(status(*graph, sky::ViewPass), CullReason::Skipped);
  EXPECT_EQ(status(*compile(Settings{}), sky::ProjectPass), CullReason::Skipped);
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  const auto base = compile(pbr_settings());
  EXPECT_EQ(status(*base, sky::ProjectPass), CullReason::Skipped);
  EXPECT_EQ(status(*base, sky::FilterPass), CullReason::Skipped);
}

TEST_F(SkyProbeGraph, ProjectAndFilterOrderBeforeLighting) {
  const auto graph = compile(pbr_settings());
  const auto index = [&](std::string_view name) {
    const NodeId node = graph->find_node(name);
    return node != InvalidNode ? graph->nodes[node].programIndex : InvalidNode;
  };
  ASSERT_NE(index(sky::ProjectPass), InvalidNode);
  EXPECT_LT(index(clusters::Pass), index(sky::ProjectPass));
  EXPECT_LT(index(sky::ProjectPass), index(sky::FilterPass));
  EXPECT_LT(index(sky::FilterPass), index(deferred::Pass));
  EXPECT_EQ(sky::project_pass_desc().order, sky::ProjectOrder);
  EXPECT_EQ(sky::filter_pass_desc().order, sky::FilterOrder);
  EXPECT_EQ(sky::ProjectOrder, 3u);
  EXPECT_EQ(sky::FilterOrder, 5u);
}

TEST_F(SkyProbeGraph, DisabledAtFullLegacyBlendKeepsGraphHash) {
  const auto lb0 = live_graph(pbr_settings());
  EXPECT_TRUE(sky::armed());
  const auto lb1 = live_graph(pbr_settings(1.f));
  EXPECT_EQ(lb0->graphHash, lb1->graphHash);
  EXPECT_FALSE(sky::armed());
  const FrameState state{.graph = lb1};
  const StageData stage = build(state, 1);
  EXPECT_FALSE(pass_enabled(*lb1, stage, sky::ProjectPass));
  EXPECT_FALSE(pass_enabled(*lb1, stage, sky::FilterPass));
}

TEST_F(SkyProbeGraph, OffDeclaresNothing) {
  sky::testing::set_hook(Hook{.mode = HookMode::Off});
  const auto graph = live_graph(pbr_settings());
  EXPECT_EQ(status(*graph, sky::ProjectPass), CullReason::Skipped);
  EXPECT_EQ(status(*graph, sky::FilterPass), CullReason::Skipped);
  EXPECT_EQ(status(*graph, deferred::Pass), CullReason::None);
  EXPECT_FALSE(sky::armed());
  EXPECT_TRUE(sky::producing());
  EXPECT_TRUE(sky::ready(m_settings));
}

TEST_F(SkyProbeGraph, ViewPassOnlyUnderItsRows) {
  Readback reader{.resource = sky::ViewResource};
  register_pass(readback_desc(reader, "t.view_read"));
  for (const uint16_t id : {uint16_t{37}, uint16_t{38}}) {
    debug::set_view_override(id);
    const auto graph = compile(pbr_settings());
    EXPECT_EQ(status(*graph, sky::ViewPass), CullReason::None) << id;
  }
  for (const uint16_t id : {uint16_t{0}, uint16_t{36}, uint16_t{21}}) {
    debug::set_view_override(id);
    EXPECT_EQ(status(*compile(pbr_settings()), sky::ViewPass), CullReason::Skipped) << id;
  }
  debug::set_view_override(37);
  sky::testing::set_hook(Hook{.mode = HookMode::Off});
  EXPECT_EQ(status(*compile(pbr_settings()), sky::ViewPass), CullReason::Skipped);
}

TEST_F(SkyProbeGraph, EveryModeDeclaresTheSameLayout) {
  const auto reference = compile(pbr_settings());
  for (const Hook hook : {Hook{.mode = HookMode::Constant, .constant = 1.f}, Hook{.mode = HookMode::Physical},
                          Hook{.mode = HookMode::Global}}) {
    sky::testing::set_hook(hook);
    const auto graph = compile(pbr_settings());
    EXPECT_EQ(graph->graphHash, reference->graphHash) << static_cast<int>(hook.mode);
    for (const std::string_view pass : {sky::ProjectPass, sky::FilterPass, deferred::Pass}) {
      EXPECT_EQ(node_pipelines(*graph, pass), node_pipelines(*reference, pass)) << pass;
    }
  }
}

TEST_F(SkyProbeGraph, HistoryAndProbeCarryTextureBinding) {
  const TexDesc radiance = sky::radiance_desc();
  EXPECT_EQ(radiance.format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_EQ(radiance.sizeClass, SizeClass::Absolute);
  EXPECT_EQ(radiance.width, 128u);
  EXPECT_EQ(radiance.height, 128u);
  EXPECT_EQ(radiance.mips, 1u);
  EXPECT_TRUE(radiance.extraUsage & wgpu::TextureUsage::TextureBinding);
  const TexDesc probe = sky::probe_desc();
  EXPECT_EQ(probe.mips, 6u);
  EXPECT_TRUE(probe.extraUsage & wgpu::TextureUsage::TextureBinding);
  const auto graph = compile(pbr_settings());
  for (const std::string_view name : {sky::Radiance, sky::Probe}) {
    const auto it = std::ranges::find(graph->resources, name, &CompiledResource::name);
    ASSERT_NE(it, graph->resources.end()) << name;
    EXPECT_TRUE(it->textureUsage & wgpu::TextureUsage::TextureBinding) << name;
  }
}

TEST_F(SkyProbeGraph, ReadyFollowsDeclaredNodes) {
  EXPECT_TRUE(sky::ready(Settings{}));
  const auto graph = live_graph(pbr_settings());
  EXPECT_TRUE(sky::producing());
  EXPECT_TRUE(sky::ready(m_settings));
  EXPECT_TRUE(lighting::ready(m_settings));
  sky::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
  EXPECT_TRUE(sky::ready(m_settings));
}

TEST_F(SkyProbeGraph, PbrReadyWithSkyProbeOff) {
  sky::testing::set_hook(Hook{.mode = HookMode::Off});
  live_graph(pbr_settings());
  EXPECT_TRUE(sky::ready(m_settings));
  EXPECT_TRUE(lighting::ready(m_settings));
}

TEST_F(SkyProbeGraph, PbrReadyWithSkyProbeConstant) {
  sky::testing::set_hook(Hook{.mode = HookMode::Constant, .constant = 1.f});
  const auto graph = live_graph(pbr_settings());
  EXPECT_EQ(status(*graph, sky::ProjectPass), CullReason::None);
  EXPECT_TRUE(sky::ready(m_settings));
  EXPECT_TRUE(lighting::ready(m_settings));
}

TEST_F(SkyProbeGraph, CameraFilledAtStageBuild) {
  Readback reader{.resource = sky::Probe};
  register_pass(readback_desc(reader, "t.probe_read"));
  const auto graph = live_graph(pbr_settings());
  const FrameState state{.graph = graph};
  StageData stage = build(state, 1);
  ASSERT_TRUE(pass_enabled(*graph, stage, sky::ProjectPass));
  bind(stage);
  const NodeId project = graph->find_node(sky::ProjectPass);
  const NodeId filter = graph->find_node(sky::FilterPass);
  bool sawProject = false;
  bool sawFilter = false;
  StageCamera want;
  ASSERT_TRUE(fill_stage_camera(m_hints.get(), want));
  for (const StageBindings::Params& params : stage.bindings.params) {
    if (params.node == project) {
      sky::ProjectParams p;
      std::memcpy(&p, params.bytes.data(), sizeof(p));
      EXPECT_EQ(p.camera.p00, want.p00);
      EXPECT_EQ(p.camera.depthMul, want.depthMul);
      EXPECT_NE(p.camera.flags & CameraValid, 0u);
      EXPECT_NE(p.flags & sky::ProjectObserve, 0u);
      EXPECT_EQ(p.alpha, 0.25f);
      EXPECT_NEAR(p.rotation.rows[1][1], std::cos(30.0 * Pi / 180.0), 1e-6);
      sawProject = true;
    }
    if (params.node == filter) {
      sky::FilterParams p;
      std::memcpy(&p, params.bytes.data(), sizeof(p));
      EXPECT_EQ(p.ambient[0], 0.4f);
      EXPECT_EQ(p.fog[3], 1.f);
      EXPECT_EQ(p.flags, uint32_t{sky::FilterHasEnv | sky::FilterObserved});
      sawFilter = true;
    }
  }
  EXPECT_TRUE(sawProject);
  EXPECT_TRUE(sawFilter);
}

TEST_F(SkyProbeGraph, AbsoluteHistoryAllocates) {
  Readback reader{.resource = sky::Radiance};
  const PassId id = register_pass(readback_desc(reader, "t.radiance_read"));
  const auto readGraph = live_graph(pbr_settings());
  const FrameState state{.graph = readGraph};
  StageData stage = build(state, 1);
  EXPECT_EQ(encode(stage), std::nullopt);
  unregister_pass(id);
  ASSERT_TRUE(reader.texture);
  EXPECT_EQ(reader.texture.GetWidth(), 128u);
  EXPECT_EQ(reader.texture.GetHeight(), 128u);
  EXPECT_EQ(reader.texture.GetFormat(), wgpu::TextureFormat::RGBA16Float);
}

struct Inputs {
  uint32_t width = 96;
  uint32_t height = 64;
  std::vector<std::array<uint8_t, 4>> color;
  std::vector<float> depth;
  std::vector<std::array<uint8_t, 4>> fog;
};

std::array<uint8_t, 4> stored(double r, double g, double b) {
  const auto q = [](double v) { return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); };
  return {q(r), q(g), q(b), 255};
}

std::array<uint8_t, 4> gradient(const Vec3& d) {
  return stored(0.25 + 0.6 * std::max(d[1], 0.0), 0.45 + 0.3 * d[0], 0.75 + 0.2 * d[2]);
}

class SkyProbeGpu : public SkyDeviceTest {
protected:
  void SetUp() override {
    const auto backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND"));
    if (!backend) {
      GTEST_SKIP() << "set AURORA_TEST_BACKEND to run the sky probe on a real backend";
    }
    if (!open({.backend = *backend, .efbSize = {96, 64, 1}})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    if (select_scene_depth_source(m_device->core, 1, true) != DepthSource::Direct) {
      GTEST_SKIP() << "needs direct scene depth";
    }
    m_depth = test::make_depth(*m_device, 96, 64);
    set_hints(scene_hints(pitched_view(40.0)));
  }

  StageCamera camera() const {
    StageCamera cam;
    EXPECT_TRUE(fill_stage_camera(m_hints.get(), cam));
    return cam;
  }

  ViewRotation rotation() const {
    ViewRotation r;
    EXPECT_TRUE(fill_view_rotation(m_hints.get(), r));
    return r;
  }

  Vec3 pixel_dir(uint32_t x, uint32_t y, const Inputs& in) const {
    const StageCamera cam = camera();
    const double ndcX = 2.0 * (x + 0.5) / in.width - 1.0;
    const double ndcY = 1.0 - 2.0 * (y + 0.5) / in.height;
    const double cx = cam.p02 - 2.0 * cam.jitterPx[0] / in.width;
    const double cy = cam.p12 + 2.0 * cam.jitterPx[1] / in.height;
    return normalized(sky::testing::world_dir(rotation(), {(ndcX + cx) / cam.p00, (ndcY + cy) / cam.p11, 1.0}));
  }

  Inputs sky_everywhere() const {
    Inputs in;
    const size_t n = size_t{in.width} * in.height;
    in.color.resize(n);
    in.depth.assign(n, 0.f);
    in.fog.assign(n, {0, 0, 0, 0});
    for (uint32_t y = 0; y < in.height; ++y) {
      for (uint32_t x = 0; x < in.width; ++x) {
        in.color[size_t{y} * in.width + x] = gradient(pixel_dir(x, y, in));
      }
    }
    return in;
  }

  static Inputs terrain_everywhere() {
    Inputs in;
    const size_t n = size_t{in.width} * in.height;
    in.color.assign(n, {255, 0, 0, 255});
    in.depth.assign(n, 0.5f);
    in.fog.assign(n, {0, 0, 0, 0});
    return in;
  }

  wgpu::Texture upload(wgpu::TextureFormat format, uint32_t width, uint32_t height, const void* data,
                       const char* label) const {
    const wgpu::TextureDescriptor descriptor{
        .label = label,
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = {width, height, 1},
        .format = format,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device->queue.WriteTexture(&destination, data, size_t{width} * height * 4, &layout, &size);
    return texture;
  }

  void stage_inputs(const Inputs& in) {
    const bool bgra = webgpu::g_frameBuffer.texture.GetFormat() == wgpu::TextureFormat::BGRA8Unorm;
    std::vector<std::array<uint8_t, 4>> color = in.color;
    if (bgra) {
      for (auto& c : color) {
        std::swap(c[0], c[2]);
      }
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = in.width * 4, .rowsPerImage = in.height};
    const wgpu::Extent3D size{in.width, in.height, 1};
    m_device->queue.WriteTexture(&destination, color.data(), color.size() * 4, &layout, &size);
    const wgpu::Texture fog = upload(wgpu::TextureFormat::RGBA8Uint, in.width, in.height, in.fog.data(), "test fog");
    webgpu::g_fogRecordBuffer.texture = fog;
    webgpu::g_fogRecordBuffer.view = fog.CreateView();
    webgpu::g_fogRecordBuffer.size = {in.width, in.height, 1};
    webgpu::g_fogRecordBuffer.format = wgpu::TextureFormat::RGBA8Uint;
    test::fill_depth(*m_device, m_depth, in.depth);
  }

  std::vector<Half4> read_half(const wgpu::Texture& texture, uint32_t mip) const {
    const uint32_t width = std::max(1u, texture.GetWidth() >> mip);
    const uint32_t height = std::max(1u, texture.GetHeight() >> mip);
    const uint32_t row = (width * 8 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = texture, .mipLevel = mip};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<Half4> out(size_t{width} * height);
    if (!words) {
      ADD_FAILURE() << "readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        uint16_t h[4];
        std::memcpy(h, bytes + size_t{y} * row + size_t{x} * 8, 8);
        out[size_t{y} * width + x] = {openpbr::from_half(h[0]), openpbr::from_half(h[1]), openpbr::from_half(h[2]),
                                      openpbr::from_half(h[3])};
      }
    }
    return out;
  }

  struct Frame {
    wgpu::Texture radiance;
    wgpu::Texture probe;
    wgpu::Texture view;
    std::vector<uint32_t> sh9;
  };

  // Compiles once, so the History carries across the frames a test encodes
  void prepare(const Settings& settings, bool readProbe = true, bool readView = false) {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = sky::Sh9Bytes};
    m_sh.buffer = m_device->device.CreateBuffer(&descriptor);
    m_readProbe = readProbe;
    register_pass(readback_desc(m_radiance, "t.radiance_read"));
    if (readProbe) {
      register_pass(readback_desc(m_probe, "t.probe_read"));
      register_pass(readback_desc(m_sh, "t.sh9_read"));
    }
    if (readView) {
      register_pass(readback_desc(m_view, "t.view_read"));
    }
    m_graph = live_graph(settings);
    m_state = std::make_unique<FrameState>(FrameState{.graph = m_graph, .debugView = m_env.viewOverride});
  }

  Frame frame(uint64_t frameId) {
    const int shCalls = m_sh.calls;
    StageData stage = build(*m_state, frameId);
    EXPECT_EQ(encode(stage), std::nullopt);
    Frame out{.radiance = m_radiance.texture, .probe = m_probe.texture, .view = m_view.texture};
    if (m_readProbe && m_sh.calls == shCalls + 1) {
      out.sh9 = test::read_buffer(*m_device, m_sh.buffer, 0, sky::Sh9Bytes).value_or(std::vector<uint32_t>{});
    }
    return out;
  }

  Frame run(const Settings& settings, uint64_t frameId, bool readProbe = true, bool readView = false) {
    if (!m_graph) {
      prepare(settings, readProbe, readView);
    }
    return frame(frameId);
  }

  Readback m_radiance{.resource = sky::Radiance};
  Readback m_probe{.resource = sky::Probe};
  Readback m_view{.resource = sky::ViewResource};
  Readback m_sh{.resource = sky::Sh9, .isBuffer = true};
  bool m_readProbe = false;
  std::shared_ptr<CompiledGraph> m_graph;
  std::unique_ptr<FrameState> m_state;

  struct Expected {
    Vec3 rgb{};
    uint32_t taps = 0;
  };

  Expected expected_texel(uint32_t tx, uint32_t ty, const Inputs& in) const {
    const StageCamera cam = camera();
    const ViewRotation rot = rotation();
    Expected out;
    for (uint32_t b = 0; b < sky::TapsPerAxis; ++b) {
      for (uint32_t a = 0; a < sky::TapsPerAxis; ++a) {
        const double u = (tx + (a + 0.5) / sky::TapsPerAxis) / Size;
        const double v = (ty + (b + 0.5) / sky::TapsPerAxis) / Size;
        const Vec3 d = sky::testing::oct_decode(u, v);
        if (d[1] < 0.0) {
          continue;
        }
        const auto s = sky::testing::project_fwd(cam, sky::testing::view_dir_fwd(rot, d), in.width, in.height);
        if (s[2] < 0.5 || s[0] < 0.0 || s[1] < 0.0 || s[0] >= 1.0 || s[1] >= 1.0) {
          continue;
        }
        const auto px = std::min(static_cast<uint32_t>(s[0] * in.width), in.width - 1);
        const auto py = std::min(static_cast<uint32_t>(s[1] * in.height), in.height - 1);
        const size_t i = size_t{py} * in.width + px;
        if (in.depth[i] > 0.f || in.fog[i] != std::array<uint8_t, 4>{0, 0, 0, 0}) {
          continue;
        }
        for (size_t c = 0; c < 3; ++c) {
          out.rgb[c] += sky::testing::srgb_to_linear(in.color[i][c] / 255.0);
        }
        ++out.taps;
      }
    }
    if (out.taps != 0) {
      for (double& c : out.rgb) {
        c /= out.taps;
      }
    }
    return out;
  }
};

TEST_F(SkyProbeGpu, GradientSkyRoundTrips) {
  const Inputs in = sky_everywhere();
  stage_inputs(in);
  const Frame frame = run(pbr_settings(), 1, false);
  ASSERT_TRUE(frame.radiance);
  const std::vector<Half4> radiance = read_half(frame.radiance, 0);
  uint32_t full = 0;
  uint32_t bad = 0;
  std::string first;
  for (uint32_t y = 0; y < Size; ++y) {
    for (uint32_t x = 0; x < Size; ++x) {
      const Expected want = expected_texel(x, y, in);
      const Half4& got = radiance[size_t{y} * Size + x];
      if (want.taps == 0) {
        bad += got[3] != 0.f ? 1 : 0;
        continue;
      }
      if (want.taps < sky::TapsPerAxis * sky::TapsPerAxis) {
        continue;
      }
      ++full;
      for (size_t c = 0; c < 3; ++c) {
        if (std::abs(got[c] - want.rgb[c]) > 2e-3 + 1e-2 * want.rgb[c] || got[3] != 1.f) {
          if (bad++ == 0) {
            first = std::to_string(x) + "," + std::to_string(y) + " c" + std::to_string(c) + " got " +
                    std::to_string(got[c]) + " want " + std::to_string(want.rgb[c]);
          }
        }
      }
    }
  }
  EXPECT_GT(full, 200u);
  EXPECT_EQ(bad, 0u) << first;
}

TEST_F(SkyProbeGpu, TerrainNeverEntersProbe) {
  Inputs in = sky_everywhere();
  for (uint32_t y = 0; y < in.height; ++y) {
    for (uint32_t x = 0; x < in.width; ++x) {
      const size_t i = size_t{y} * in.width + x;
      if (x >= in.width / 2) {
        in.color[i] = {255, 0, 0, 255};
        in.depth[i] = 0.5f;
      } else if (y % 4 == 0) {
        in.color[i] = {0, 255, 0, 255};
        in.fog[i] = {10, 20, 30, 41};
      } else {
        in.color[i] = {0, 0, 200, 255};
      }
    }
  }
  stage_inputs(in);
  const Frame frame = run(pbr_settings(), 1, false);
  ASSERT_TRUE(frame.radiance);
  const std::vector<Half4> radiance = read_half(frame.radiance, 0);
  uint32_t observed = 0;
  for (const Half4& t : radiance) {
    if (t[3] > 0.f) {
      ++observed;
      ASSERT_EQ(t[0], 0.f);
      ASSERT_EQ(t[1], 0.f);
      ASSERT_GT(t[2], 0.5f);
    }
  }
  EXPECT_GT(observed, 50u);
}

TEST_F(SkyProbeGpu, UnseenTakesFill) {
  const Env env{.skyVisibility = 0.5f, .ambient = {0.3f, 0.35f, 0.4f}, .fog = {0.7f, 0.6f, 0.5f}};
  set_hints(scene_hints(pitched_view(40.0), env));
  stage_inputs(terrain_everywhere());
  const Frame frame = run(pbr_settings(), 1);
  ASSERT_TRUE(frame.probe);
  const std::vector<Half4> probe = read_half(frame.probe, 0);
  const Vec3 a = lin3(env.ambient);
  const Vec3 f = lin3(env.fog);
  const double g =
      std::clamp(sky::testing::luminance(a) / sky::testing::luminance(f), double{sky::GainMin}, double{sky::GainMax});
  for (uint32_t y = 0; y < Size; y += 7) {
    for (uint32_t x = 0; x < Size; x += 5) {
      const Vec3 d = sky::testing::oct_decode((x + 0.5) / Size, (y + 0.5) / Size);
      const Half4& got = probe[size_t{y} * Size + x];
      for (size_t c = 0; c < 3; ++c) {
        const double up = d[1] >= 0.0 ? g * f[c] : a[c];
        const double want = a[c] + (up - a[c]) * env.skyVisibility;
        ASSERT_NEAR(got[c], want, 2e-3 + 2e-3 * want) << x << "," << y << " c" << c;
      }
    }
  }
}

TEST_F(SkyProbeGpu, ResetClearsHistory) {
  stage_inputs(sky_everywhere());
  const Frame first = run(pbr_settings(), 1, false);
  ASSERT_TRUE(first.radiance);
  const auto seen = read_half(first.radiance, 0);
  EXPECT_GT(std::ranges::count_if(seen, [](const Half4& t) { return t[3] > 0.f; }), 100);
  set_hints(scene_hints(pitched_view(40.0), Env{}, false));
  stage_inputs(terrain_everywhere());
  const Frame second = run(pbr_settings(), 2, false);
  ASSERT_TRUE(second.radiance);
  for (const Half4& t : read_half(second.radiance, 0)) {
    ASSERT_EQ(t, (Half4{0.f, 0.f, 0.f, 0.f}));
  }
  EXPECT_GE(sky::testing::counters().resets, 2u);
}

TEST_F(SkyProbeGpu, GainNormalisesUpperMean) {
  const Env env{.ambient = {0.5f, 0.45f, 0.4f}, .fog = {0.3f, 0.35f, 0.5f}};
  set_hints(scene_hints(pitched_view(40.0), env));
  stage_inputs(sky_everywhere());
  const Frame frame = run(pbr_settings(), 1);
  ASSERT_TRUE(frame.probe);
  const std::vector<Half4> probe = read_half(frame.probe, 0);
  double sum = 0.0;
  double weight = 0.0;
  for (uint32_t y = 0; y < Size; ++y) {
    for (uint32_t x = 0; x < Size; ++x) {
      const double u = (x + 0.5) / Size;
      const double v = (y + 0.5) / Size;
      const Vec3 d = sky::testing::oct_decode(u, v);
      if (d[1] <= 0.0) {
        continue;
      }
      const Half4& t = probe[size_t{y} * Size + x];
      const double w = sky::testing::texel_weight(u, v, Size) * d[1];
      sum += w * sky::testing::luminance({t[0], t[1], t[2]});
      weight += w;
    }
  }
  EXPECT_NEAR(sum / weight, sky::testing::luminance(lin3(env.ambient)),
              2e-3 * sky::testing::luminance(lin3(env.ambient)));
  ASSERT_EQ(frame.sh9.size(), sky::Sh9Bytes / 4);
  sky::Sh9Block block;
  std::memcpy(&block, frame.sh9.data(), sizeof(block));
  EXPECT_EQ(block.info[0], 1.f);
  EXPECT_EQ(block.info[1], 1.f);
  EXPECT_EQ(block.info[2], 1.f);
  EXPECT_GT(block.info[3], 0.f);
  EXPECT_EQ(block.amb[3], 1.f);
  EXPECT_NEAR(block.rot[1][1], std::cos(40.0 * Pi / 180.0), 1e-6);
}

TEST_F(SkyProbeGpu, ConstantModeIsFurnaceExact) {
  sky::testing::set_hook(Hook{.mode = HookMode::Constant, .constant = 1.f});
  stage_inputs(sky_everywhere());
  const Frame frame = run(pbr_settings(), 1);
  ASSERT_TRUE(frame.probe);
  for (uint32_t mip = 0; mip < sky::ProbeMips; ++mip) {
    for (const Half4& t : read_half(frame.probe, mip)) {
      ASSERT_EQ(t, (Half4{1.f, 1.f, 1.f, 1.f})) << mip;
    }
  }
  ASSERT_EQ(frame.sh9.size(), sky::Sh9Bytes / 4);
  sky::Sh9Block block;
  std::memcpy(&block, frame.sh9.data(), sizeof(block));
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(block.c[0][c], Pi * std::sqrt(4.0 * Pi), 1e-4);
    for (size_t i = 1; i < 9; ++i) {
      EXPECT_EQ(block.c[i][c], 0.f) << i;
    }
  }
  EXPECT_EQ(block.info[0], 1.f);
  EXPECT_EQ(block.info[2], 0.f);
  EXPECT_EQ(block.amb[3], 0.f);
  EXPECT_GT(sky::testing::counters().constant, 0u);
}

TEST_F(SkyProbeGpu, PrevBindsWithoutConsumers) {
  stage_inputs(sky_everywhere());
  const Frame first = run(pbr_settings(), 1, false);
  ASSERT_TRUE(first.radiance);
  EXPECT_TRUE(first.radiance.GetUsage() & wgpu::TextureUsage::TextureBinding);
  const std::vector<Half4> before = read_half(first.radiance, 0);
  Inputs dark = sky_everywhere();
  for (auto& c : dark.color) {
    c = {0, 0, 0, 255};
  }
  stage_inputs(dark);
  const Frame second = run(pbr_settings(), 2, false);
  ASSERT_TRUE(second.radiance);
  const std::vector<Half4> after = read_half(second.radiance, 0);
  uint32_t blended = 0;
  for (size_t i = 0; i < after.size(); ++i) {
    if (before[i][3] > 0.f && after[i][3] > 0.f) {
      ASSERT_NEAR(after[i][2], before[i][2] * 0.75f, 2e-3) << i;
      ++blended;
    }
  }
  EXPECT_GT(blended, 100u);
}

TEST_F(SkyProbeGpu, MaskViewMatchesSkyTest) {
  Inputs in = sky_everywhere();
  for (uint32_t i = 0; i < in.depth.size(); ++i) {
    if (i % 3 == 0) {
      in.depth[i] = 0.25f;
    } else if (i % 7 == 0) {
      in.fog[i] = {0, 0, 0, 1};
    }
  }
  stage_inputs(in);
  debug::set_view_override(38);
  const Frame frame = run(pbr_settings(), 1, false, true);
  ASSERT_TRUE(frame.view);
  const uint32_t row = (in.width * 4 + 255) & ~255u;
  const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                          .size = uint64_t{row} * in.height};
  const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::TexelCopyTextureInfo source{.texture = frame.view};
  const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = in.height},
                                              .buffer = buffer};
  const wgpu::Extent3D size{in.width, in.height, 1};
  encoder.CopyTextureToBuffer(&source, &destination, &size);
  const wgpu::CommandBuffer commands = encoder.Finish();
  ASSERT_TRUE(test::submit_and_wait(*m_device, commands));
  const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * in.height);
  ASSERT_TRUE(words);
  const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
  for (uint32_t y = 0; y < in.height; ++y) {
    for (uint32_t x = 0; x < in.width; ++x) {
      const size_t i = size_t{y} * in.width + x;
      const bool skyPixel = in.depth[i] <= 0.f && in.fog[i] == std::array<uint8_t, 4>{0, 0, 0, 0};
      ASSERT_EQ(bytes[size_t{y} * row + size_t{x} * 4], skyPixel ? 255 : 0) << x << "," << y;
    }
  }
  EXPECT_EQ(sky::testing::counters().viewDraws, 1u);
}

TEST_F(SkyProbeGpu, RepeatIsBitIdentical) {
  prepare(pbr_settings());
  const auto sequence = [&](uint64_t start) {
    std::vector<std::vector<Half4>> out;
    stage_inputs(sky_everywhere());
    for (uint64_t id = start; id < start + 3; ++id) {
      const Frame f = frame(id);
      out.push_back(read_half(f.radiance, 0));
      out.push_back(read_half(f.probe, 0));
      out.push_back(read_half(f.probe, 4));
      out.push_back({});
      for (size_t w = 0; w + 4 <= f.sh9.size(); w += 4) {
        out.back().push_back({std::bit_cast<float>(f.sh9[w]), std::bit_cast<float>(f.sh9[w + 1]),
                              std::bit_cast<float>(f.sh9[w + 2]), std::bit_cast<float>(f.sh9[w + 3])});
      }
    }
    return out;
  };
  const auto a = sequence(1);
  set_hints(scene_hints(pitched_view(40.0), Env{}, false));
  stage_inputs(terrain_everywhere());
  frame(4);
  set_hints(scene_hints(pitched_view(40.0)));
  const auto b = sequence(5);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_TRUE(a[i] == b[i]) << i;
  }
}
} // namespace
} // namespace aurora::render::detail

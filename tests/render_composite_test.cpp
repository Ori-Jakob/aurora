#include <gtest/gtest.h>

#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "gx/surface_pack.hpp"
#include "render/ao/ao.hpp"
#include "render/ao/upsample.hpp"
#include "render/debug/views.hpp"
#include "render/fakeshadow/lsm.hpp"
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
#include "render/lighting/exposure.hpp"
#include "render/lighting/openpbr.hpp"
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
using aurora::test::scene_depth;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

std::shared_ptr<const SceneHints> camera_hints(bool camera = true) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -10.f / (100000.f - 10.f);
  proj.m2[3] = -(100000.f * 10.f) / (100000.f - 10.f);
  proj.m3[2] = -1.f;
  auto hints = std::make_shared<SceneHints>();
  if (camera) {
    hints->mainCamera = {
        .role = AURORA_CAMERA_MAIN,
        .proj = proj,
        .type = GX_PERSPECTIVE,
        .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
    };
    hints->present = SceneHasMainCamera;
    hints->motionValid = true;
  }
  return hints;
}

Settings ao_settings(Quality quality = Quality::High) {
  Settings settings;
  settings.ao = quality;
  return settings;
}

Settings pbr_settings(float legacyBlend, Quality ao = Quality::Off) {
  Settings settings;
  settings.ao = ao;
  settings.pbr = true;
  settings.legacyBlend = legacyBlend;
  return settings;
}

struct StubAo {
  wgpu::TextureFormat format = wgpu::TextureFormat::RGBA8Unorm;
  uint16_t divisor = 1;
  bool enabled = true;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  int calls = 0;
};

void declare_stub_ao(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& stub = *static_cast<const StubAo*>(state);
  builder.after(phase::Ao);
  builder.create(resource::AoFinal, {.format = stub.format, .sizeClass = SizeClass::Efb, .divisor = stub.divisor},
                 Access::CopyDst);
}

bool stub_ao_enabled(const StageData&, const void* state) { return static_cast<const StubAo*>(state)->enabled; }

void encode_stub_ao(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubAo*>(const_cast<void*>(state));
  ++stub.calls;
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

PassDesc stub_ao_desc(StubAo& stub) {
  return {
      .name = "test.ao",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 0,
      .runsWithoutMarker = true,
      .declare = declare_stub_ao,
      .enabled = stub_ao_enabled,
      .encode = encode_stub_ao,
      .state = &stub,
  };
}

struct StubSsil {
  wgpu::TextureFormat format = wgpu::TextureFormat::RGBA16Float;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  int calls = 0;
};

void declare_stub_ssil(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& stub = *static_cast<const StubSsil*>(state);
  builder.after(phase::Ao);
  builder.create(resource::SsilIrradiance, {.format = stub.format, .sizeClass = SizeClass::Efb}, Access::CopyDst);
}

void encode_stub_ssil(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubSsil*>(const_cast<void*>(state));
  ++stub.calls;
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

PassDesc stub_ssil_desc(StubSsil& stub) {
  return {
      .name = "test.ssil",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 1,
      .runsWithoutMarker = true,
      .declare = declare_stub_ssil,
      .encode = encode_stub_ssil,
      .state = &stub,
  };
}

struct StubHdr {
  wgpu::TextureFormat format = lighting::deferred::HdrFormat;
  bool enabled = true;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  int calls = 0;
};

void declare_stub_hdr(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& stub = *static_cast<const StubHdr*>(state);
  builder.after(phase::Lighting);
  builder.create(resource::SceneHdr,
                 {.format = stub.format, .sizeClass = SizeClass::Efb, .extraUsage = wgpu::TextureUsage::TextureBinding},
                 Access::CopyDst);
}

bool stub_hdr_enabled(const StageData&, const void* state) { return static_cast<const StubHdr*>(state)->enabled; }

void encode_stub_hdr(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubHdr*>(const_cast<void*>(state));
  ++stub.calls;
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

struct StubExposure {
  std::array<float, 4> words{1.f, 1.f, 1.f, 1.f};
  bool enabled = true;
  int calls = 0;
};

// Named as the meter, because the composite's contract accepts a real exposure buffer only from it
void declare_stub_exposure(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Exposure);
  builder.create_buffer(lighting::exposure::Resource, {.size = lighting::exposure::BufferBytes}, Access::CopyDst);
}

bool stub_exposure_enabled(const StageData&, const void* state) {
  return static_cast<const StubExposure*>(state)->enabled;
}

void encode_stub_exposure(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubExposure*>(const_cast<void*>(state));
  ++stub.calls;
  const BufferView view = ctx.buffer(BufRef{0});
  if (view.buffer) {
    ctx.encoder().WriteBuffer(view.buffer, view.offset, reinterpret_cast<const uint8_t*>(stub.words.data()),
                              sizeof(stub.words));
  }
}

PassDesc stub_exposure_desc(StubExposure& stub) {
  return {
      .name = std::string{lighting::exposure::Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .order = 0,
      .runsWithoutMarker = true,
      .declare = declare_stub_exposure,
      .enabled = stub_exposure_enabled,
      .encode = encode_stub_exposure,
      .state = &stub,
  };
}

PassDesc stub_hdr_desc(StubHdr& stub) {
  return {
      .name = "test.hdr",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 0,
      .runsWithoutMarker = true,
      .declare = declare_stub_hdr,
      .enabled = stub_hdr_enabled,
      .encode = encode_stub_hdr,
      .state = &stub,
  };
}

struct StubLsm {
  bool enabled = true;
  wgpu::Buffer staging;
  uint32_t rowBytes = 0;
  int calls = 0;
};

void declare_stub_lsm(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Capture);
  builder.before(phase::Geometry);
  builder.create(fakeshadow::lsm::Resource,
                 {.format = fakeshadow::lsm::Format,
                  .sizeClass = SizeClass::Efb,
                  .extraUsage = wgpu::TextureUsage::TextureBinding},
                 Access::CopyDst);
}

bool stub_lsm_enabled(const StageData&, const void* state) { return static_cast<const StubLsm*>(state)->enabled; }

void encode_stub_lsm(PassContext& ctx, const void* state) {
  auto& stub = *static_cast<StubLsm*>(const_cast<void*>(state));
  ++stub.calls;
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

// Stands in for lsm.mask's replay; the composite keys FlagLsm on whichever pass creates the mask
PassDesc stub_lsm_desc(StubLsm& stub) {
  return {
      .name = "test.lsm",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = fakeshadow::lsm::Order,
      .runsWithoutMarker = true,
      .declare = declare_stub_lsm,
      .enabled = stub_lsm_enabled,
      .encode = encode_stub_lsm,
      .state = &stub,
  };
}

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Render;
  uint16_t order = 0;
  std::function<void(InternalPassBuilder&)> declare;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  static_cast<const TestPass*>(state)->declare(builder);
}

void encode_nothing(PassContext&, const void*) {}

PassDesc desc_of(const TestPass& pass) {
  return {
      .name = pass.name,
      .stage = pass.stage,
      .kind = pass.kind,
      .order = pass.order,
      .runsWithoutMarker = true,
      .declare = declare_test_pass,
      .encode = encode_nothing,
      .state = &pass,
  };
}

size_t program_index(const CompiledGraph& graph, std::string_view name) {
  const auto& program = graph.stages[static_cast<size_t>(Stage::OpaqueDone)].nodes;
  const auto it = std::ranges::find(program, graph.find_node(name));
  return it == program.end() ? SIZE_MAX : static_cast<size_t>(it - program.begin());
}

bool pass_enabled(const CompiledGraph& graph, const StageData& stage, std::string_view name) {
  const NodeId node = graph.find_node(name);
  return node != InvalidNode && graph.nodes[node].programIndex != InvalidNode &&
         stage.is_enabled(graph.nodes[node].programIndex);
}

TEST(CompositeRules, ParamsAreSixteenBytes) {
  EXPECT_EQ(sizeof(composite::Params), 16u);
  EXPECT_EQ(offsetof(composite::Params, legacyBlend), 4u);
  EXPECT_LE(sizeof(composite::Params), MaxPassParamsSize);
  EXPECT_EQ(composite::FlagIdentity, 1u);
  EXPECT_EQ(composite::FlagNoFog, 2u);
  EXPECT_EQ(composite::FlagPbr, 4u);
  EXPECT_EQ(composite::FlagTonemapAgX, 8u);
  const composite::Params params;
  EXPECT_EQ(params.flags, 0u);
  EXPECT_EQ(params.legacyBlend, 0.f);
}

TEST(CompositeRules, TargetIsRgbOnlyInEfbFormat) {
  for (const wgpu::TextureFormat format : {wgpu::TextureFormat::BGRA8Unorm, wgpu::TextureFormat::RGBA8Unorm}) {
    const shaders::RenderPipelineDesc desc = composite::pipeline_desc(format, {}, 7);
    EXPECT_EQ(desc.shader, composite::Shader);
    EXPECT_EQ(desc.colorCount, 1u);
    EXPECT_EQ(desc.colors[0].format, format);
    EXPECT_EQ(desc.colors[0].writeMask, RgbMask);
    EXPECT_FALSE(desc.colors[0].blend);
    EXPECT_EQ(desc.depthFormat, wgpu::TextureFormat::Undefined);
    EXPECT_EQ(desc.sampleCount, 1u);
    EXPECT_EQ(desc.layoutKey, 7u);
    EXPECT_TRUE(desc.defines.empty());
    const shaders::RenderPipelineDesc radiance = composite::pipeline_desc(format, {}, 7, true);
    EXPECT_EQ(radiance.shader, composite::Shader);
    ASSERT_EQ(radiance.colorCount, 2u);
    EXPECT_EQ(radiance.colors[0].format, format);
    EXPECT_EQ(radiance.colors[0].writeMask, RgbMask);
    EXPECT_EQ(radiance.colors[1].format, wgpu::TextureFormat::RGBA16Float);
    EXPECT_EQ(radiance.colors[1].writeMask, wgpu::ColorWriteMask::All);
    EXPECT_FALSE(radiance.colors[1].blend);
    ASSERT_EQ(radiance.defines.size(), 1u);
    EXPECT_EQ(radiance.defines[0], "COMPOSITE_RADIANCE");
    EXPECT_EQ(radiance.layoutKey, 7u);
  }
}

TEST(CompositeRules, WantedFollowsAoOrPbr) {
  EXPECT_FALSE(composite::wanted(Settings{}));
  EXPECT_FALSE(composite::wanted({.taa = true}));
  EXPECT_FALSE(composite::wanted({.upscaler = Upscaler::Fsr}));
  EXPECT_FALSE(composite::ao_wanted(Settings{}));
  for (const Quality quality : {Quality::Low, Quality::Medium, Quality::High, Quality::Ultra}) {
    EXPECT_TRUE(composite::wanted(ao_settings(quality)));
    EXPECT_TRUE(composite::ao_wanted(ao_settings(quality)));
    EXPECT_TRUE(composite::ao_wanted(pbr_settings(0.f, quality)));
  }
  for (const float lb : {0.f, 0.5f, 1.f}) {
    EXPECT_TRUE(composite::wanted(pbr_settings(lb)));
    EXPECT_FALSE(composite::ao_wanted(pbr_settings(lb)));
  }
}

TEST(CompositeRules, DeclareSkipsWithoutTierOrMsaa) {
  GraphEnvironment env;
  env.gxFormats[static_cast<size_t>(GxResource::Color)] = wgpu::TextureFormat::BGRA8Unorm;
  env.gxFormats[static_cast<size_t>(GxResource::Fog)] = wgpu::TextureFormat::RGBA8Uint;
  env.gxFormats[static_cast<size_t>(GxResource::Normal)] = wgpu::TextureFormat::RGB10A2Unorm;
  Settings settings = ao_settings();
  const PassSetup setup{.settings = &settings, .env = &env};
  EXPECT_TRUE(composite::declare_wanted(setup));
  settings.debugView = 1;
  EXPECT_TRUE(composite::declare_wanted(setup));
  settings.debugView = 0;
  const Settings off;
  EXPECT_FALSE(composite::declare_wanted({.settings = &off, .env = &env}));
  EXPECT_FALSE(composite::declare_wanted({.settings = nullptr, .env = &env}));
  GraphEnvironment noFog = env;
  noFog.gxFormats[static_cast<size_t>(GxResource::Fog)] = wgpu::TextureFormat::Undefined;
  EXPECT_FALSE(composite::declare_wanted({.settings = &settings, .env = &noFog}));
  GraphEnvironment noNormal = env;
  noNormal.gxFormats[static_cast<size_t>(GxResource::Normal)] = wgpu::TextureFormat::Undefined;
  EXPECT_FALSE(composite::declare_wanted({.settings = &settings, .env = &noNormal}));
  GraphEnvironment msaa = env;
  msaa.gxSampleCount = 4;
  EXPECT_FALSE(composite::declare_wanted({.settings = &settings, .env = &msaa}));
}

TEST(CompositeRules, InputTableIsAppendOnly) {
  EXPECT_EQ(composite::BindLegacy, 0u);
  EXPECT_EQ(composite::BindFog, 1u);
  EXPECT_EQ(composite::BindNormal, 2u);
  EXPECT_EQ(composite::BindAo, 3u);
  EXPECT_EQ(composite::BindSurface, 4u);
  EXPECT_EQ(composite::BindHdr, 5u);
  EXPECT_EQ(composite::BindExposure, 6u);
  EXPECT_EQ(composite::BindSsil, 7u);
  EXPECT_EQ(composite::BindLsm, 8u);
  EXPECT_EQ(composite::FlagSsil, 16u);
  EXPECT_EQ(composite::FlagLsm, 32u);
  EXPECT_EQ(fakeshadow::lsm::Resource, "lsm.mask");
  EXPECT_EQ(offsetof(composite::Params, ssilAlbedo), 8u);
  EXPECT_EQ(lighting::exposure::Resource, "exposure");
  EXPECT_EQ(composite::LegacyTex, "legacyTex");
  EXPECT_EQ(composite::CopyPass, "legacy.copy");
  EXPECT_EQ(composite::Pass, "composite.main");
  EXPECT_EQ(composite::AoFormat, ao::upsample::Format);
}

TEST(CompositeRules, LsmFogDecodeMirror) {
  // A kept fake of strength a over white under fog q leaves q*fc*a + 1 - a in the mask; P then takes exactly 1 - a
  const std::array<float, 3> fc{0.8f, 0.5f, 0.1f};
  for (const uint32_t q7 : {0u, 40u, 100u}) {
    const float q = static_cast<float>(q7) / 127.f;
    for (const float a : {0.f, 0.3f, 0.7f}) {
      std::array<float, 3> mask{};
      for (size_t c = 0; c < 3; ++c) {
        mask[c] = q * fc[c] * a + 1.f - a;
      }
      const std::array<float, 3> m = fakeshadow::lsm::term(mask, q, fc);
      const std::array<float, 4> hdr{0.4f, 0.2f, 0.1f, 1.f};
      const std::array<float, 3> legacy{0.9f, 0.8f, 0.7f};
      const std::array<float, 3> shadowed =
          composite::testing::legacy_pbr_hdr(legacy, fc, q7, 1.f, 1.f, hdr, 0.f, 1.f, false, m);
      const std::array<float, 3> lit =
          composite::testing::legacy_pbr_hdr(legacy, fc, q7, 1.f, 1.f, hdr, 0.f, 1.f, false);
      for (size_t c = 0; c < 3; ++c) {
        EXPECT_NEAR(m[c], 1.f - a, 2e-6f) << q7 << " " << a << " " << c;
        // At lb 0 the output is L - S + (1 - q)*m*P, so the shadow scales only the PBR share
        const float s = std::max(legacy[c] - q * fc[c], 0.f);
        EXPECT_NEAR(shadowed[c] - (legacy[c] - s), (lit[c] - (legacy[c] - s)) * m[c], 1e-5f) << c;
      }
    }
  }
}

TEST(CompositeRules, LsmUnitMaskExact) {
  std::mt19937 rng{53};
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  for (int i = 0; i < 256; ++i) {
    const std::array<float, 3> fc{unit(rng), unit(rng), unit(rng)};
    const float q = unit(rng);
    // A white texel decodes to exactly 1 under any fog, so the mask's clear never shades P
    EXPECT_EQ(fakeshadow::lsm::term({1.f, 1.f, 1.f}, q, fc), (std::array<float, 3>{1.f, 1.f, 1.f}));
    const std::array<float, 3> legacy{unit(rng), unit(rng), unit(rng)};
    const std::array<float, 4> hdr{unit(rng) * 4.f, unit(rng), unit(rng), 1.f};
    const auto q7 = static_cast<uint32_t>(rng() % 128);
    const float ao = unit(rng);
    EXPECT_EQ(composite::testing::legacy_pbr_hdr(legacy, fc, q7, ao, 1.f, hdr, 0.3f, 1.5f, true, {1.f, 1.f, 1.f}),
              composite::testing::legacy_pbr_hdr(legacy, fc, q7, ao, 1.f, hdr, 0.3f, 1.5f, true));
  }
}

TEST(CompositeRules, NoFogEnvParses) {
  EXPECT_EQ(composite::NoFogEnv, "AURORA_RENDER_COMPOSITE_NOFOG");
  EXPECT_EQ(composite::params_flags(false, nullptr), 0u);
  EXPECT_EQ(composite::params_flags(false, "on"), uint32_t{composite::FlagNoFog});
  EXPECT_EQ(composite::params_flags(false, "off"), 0u);
  EXPECT_EQ(composite::params_flags(false, "ON"), 0u);
  EXPECT_EQ(composite::params_flags(false, ""), 0u);
  EXPECT_EQ(composite::params_flags(true, nullptr), uint32_t{composite::FlagIdentity});
  EXPECT_EQ(composite::params_flags(true, "on"), uint32_t{composite::FlagIdentity | composite::FlagNoFog});
}

int to_byte(float value) { return static_cast<int>(std::lround(std::clamp(value, 0.f, 1.f) * 255.f)); }

TEST(CompositeRules, MirrorBoundOnFoggedInputs) {
  std::mt19937 rng{0xC0FFEEu};
  std::uniform_int_distribution<int> byte(0, 255);
  constexpr std::array<float, 5> Aos{0.03f, 0.25f, 0.5f, 0.9f, 254.f / 255.f};
  constexpr std::array<float, 4> Alphas{0.f, 1.f / 3.f, 2.f / 3.f, 1.f};
  size_t checked = 0;
  size_t changed = 0;
  for (int i = 0; i < 4000; ++i) {
    const std::array<int, 3> p{byte(rng), byte(rng), byte(rng)};
    const std::array<int, 3> fcBytes{byte(rng), byte(rng), byte(rng)};
    const float f = static_cast<float>(i % 128) / 127.f;
    const auto q7 = static_cast<uint32_t>(std::min(std::lround(f * 127.f), 127l));
    std::array<int, 3> lBytes{};
    std::array<float, 3> l{};
    std::array<float, 3> fc{};
    for (size_t c = 0; c < 3; ++c) {
      lBytes[c] = static_cast<int>(std::lround(255.0 * ((1.0 - f) * p[c] / 255.0 + f * fcBytes[c] / 255.0)));
      l[c] = static_cast<float>(lBytes[c]) / 255.f;
      fc[c] = static_cast<float>(fcBytes[c]) / 255.f;
    }
    for (const float a : Aos) {
      for (const float alpha : Alphas) {
        const std::array<float, 3> out = composite::testing::legacy_ao(l, fc, q7, a, alpha);
        const bool receiver = alpha > 0.f;
        bool any = false;
        for (size_t c = 0; c < 3; ++c) {
          const int d = lBytes[c] - to_byte(out[c]);
          ASSERT_GE(d, 0) << "brightened: L " << lBytes[c] << " fc " << fcBytes[c] << " q7 " << q7 << " a " << a;
          const float s = l[c] - static_cast<float>(q7) / 127.f * fc[c];
          const int bound = static_cast<int>(std::ceil(255.f * std::max(s, 0.f) * (1.f - a))) + 1;
          ASSERT_LE(d, bound) << "L " << lBytes[c] << " fc " << fcBytes[c] << " q7 " << q7 << " a " << a;
          if (q7 == 127) {
            ASSERT_LE(d, 1);
            if (lBytes[c] == fcBytes[c]) {
              ASSERT_EQ(d, 0);
            }
          }
          if (!receiver) {
            ASSERT_EQ(out[c], l[c]);
          }
          any |= d != 0;
        }
        ++checked;
        changed += any ? 1 : 0;
      }
    }
    EXPECT_EQ(composite::testing::legacy_ao(l, fc, q7, 1.f, 1.f), l);
  }
  EXPECT_GT(changed, checked / 4);
  const std::array<float, 3> legacy{40.f / 255.f, 40.f / 255.f, 40.f / 255.f};
  const std::array<float, 3> stale{200.f / 255.f, 30.f / 255.f, 30.f / 255.f};
  EXPECT_EQ(composite::testing::legacy_ao(legacy, stale, 127, 0.25f, 1.f), legacy);
  EXPECT_NE(composite::testing::legacy_ao(legacy, std::array<float, 3>{}, 0, 0.25f, 1.f), legacy);
}

TEST(CompositeRules, PbrActiveTruthTable) {
  const bool identity = composite::identity_forced();
  for (const bool pbr : {false, true}) {
    for (const float lb : {0.f, 0.5f, 0.999f, 1.f}) {
      Settings settings;
      settings.pbr = pbr;
      settings.legacyBlend = lb;
      EXPECT_EQ(composite::pbr_term_wanted(settings), pbr && lb < 1.f && !identity) << lb;
      for (const bool live : {false, true}) {
        for (const bool hdrLive : {false, true}) {
          for (const bool classView : {false, true}) {
            const bool want = live && hdrLive && pbr && lb < 1.f && !identity && !classView;
            EXPECT_EQ(composite::testing::pbr_active_for(live, hdrLive, classView, settings), want)
                << pbr << " " << lb << " " << live << hdrLive << classView;
          }
        }
      }
    }
  }
}

struct MirrorInput {
  std::array<float, 3> l{};
  std::array<float, 3> fc{};
  uint32_t q7 = 0;
  float ao = 1.f;
  float alpha = 1.f;
  std::array<float, 4> hdr{};
};

MirrorInput random_input(std::mt19937& rng, int i) {
  std::uniform_int_distribution<int> byte(0, 255);
  constexpr std::array<float, 4> Alphas{0.f, 1.f / 3.f, 2.f / 3.f, 1.f};
  const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
  MirrorInput in;
  for (size_t c = 0; c < 3; ++c) {
    in.l[c] = b() / 255.f;
    in.fc[c] = b() / 255.f;
    in.hdr[c] = b() / 200.f;
  }
  in.hdr[3] = 1.f;
  in.q7 = static_cast<uint32_t>(byte(rng) % 128);
  in.ao = b() / 255.f;
  in.alpha = Alphas[static_cast<size_t>(i) % Alphas.size()];
  return in;
}

float encode_srgb(float linear) {
  const float x = std::clamp(linear, 0.f, 1.f);
  return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
}

TEST(CompositeRules, PbrMirrorBlendOneIsLegacyBits) {
  std::mt19937 rng{0xB1E7u};
  for (int i = 0; i < 4000; ++i) {
    MirrorInput in = random_input(rng, i);
    const std::array<float, 3> legacy = composite::testing::legacy_ao(in.l, in.fc, in.q7, in.ao, in.alpha);
    const float e = static_cast<float>(i % 4) + 0.5f;
    const bool agx = i % 2 == 0;
    ASSERT_EQ(composite::testing::legacy_pbr_hdr(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, 1.f, e, agx), legacy)
        << i;
    const float lb = static_cast<float>(i % 3) / 2.f;
    in.hdr[3] = static_cast<float>(i % 2) * 0.49f;
    ASSERT_EQ(composite::testing::legacy_pbr_hdr(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, lb, e, agx), legacy) << i;
  }
}

TEST(CompositeRules, PbrMirrorUsesExposureAndTonemap) {
  std::mt19937 rng{0xA1BEu};
  size_t fogged = 0;
  size_t bad = 0;
  constexpr std::array<float, 3> Exposures{0.5f, 1.f, 2.7f};
  for (int i = 0; i < 4000; ++i) {
    const MirrorInput in = random_input(rng, i);
    const float e = Exposures[static_cast<size_t>(i) % Exposures.size()];
    const bool agx = (i / 3) % 2 == 0;
    const std::array<float, 3> out =
        composite::testing::legacy_pbr_hdr(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, 0.f, e, agx);
    const lighting::tonemap::Rgb mapped = lighting::tonemap::apply({in.hdr[0] * e, in.hdr[1] * e, in.hdr[2] * e}, agx);
    const float q = static_cast<float>(in.q7) / 127.f;
    bool guard = false;
    for (size_t c = 0; c < 3; ++c) {
      guard |= in.l[c] - q * in.fc[c] < -2.f / 255.f;
    }
    if (guard) {
      ASSERT_EQ(out, in.l) << i;
      ++bad;
      continue;
    }
    bool all = true;
    for (size_t c = 0; c < 3; ++c) {
      const float s = in.l[c] - q * in.fc[c];
      const float base = (1.f - q) * encode_srgb(mapped[c]);
      ASSERT_NEAR(out[c], in.l[c] - std::max(s, 0.f) + base, 1e-5f) << i;
      if (s >= 0.f) {
        ASSERT_NEAR(out[c], q * in.fc[c] + base, 1e-5f) << i;
      } else {
        all = false;
      }
    }
    fogged += all ? 1 : 0;
  }
  EXPECT_GT(fogged, 1000u);
  EXPECT_GT(bad, 0u);
}

TEST(CompositeRules, LegacyScaleEqualsAo) {
  std::mt19937 rng{0x5CA1Eu};
  for (int i = 0; i < 2000; ++i) {
    MirrorInput in = random_input(rng, i);
    in.alpha = 1.f;
    const float q = static_cast<float>(in.q7) / 127.f;
    bool guard = false;
    for (size_t c = 0; c < 3; ++c) {
      guard |= in.l[c] - q * in.fc[c] < -2.f / 255.f;
    }
    if (guard) {
      continue;
    }
    const std::array<float, 3> half =
        composite::testing::legacy_pbr_hdr(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, 0.5f, 1.f, false);
    const std::array<float, 3> pbr =
        composite::testing::legacy_pbr_hdr(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, 0.f, 1.f, false);
    for (size_t c = 0; c < 3; ++c) {
      const float sp = std::max(in.l[c] - q * in.fc[c], 0.f);
      const float pv = pbr[c] - (in.l[c] - sp);
      ASSERT_NEAR(half[c], in.l[c] - sp + 0.5f * sp * in.ao + 0.5f * pv, 2e-6f) << i;
    }
  }
}

TEST(CompositeRules, NonFiniteHdrMirrorStaysFinite) {
  constexpr float Inf = std::numeric_limits<float>::infinity();
  for (const std::array<float, 4> hdr :
       {std::array<float, 4>{Inf, 1.f, 1.f, 1.f}, std::array<float, 4>{std::nanf(""), 0.f, 2.f, 1.f},
        std::array<float, 4>{-1.f, 65504.f, 0.f, 1.f}}) {
    for (const bool agx : {false, true}) {
      const std::array<float, 3> out =
          composite::testing::legacy_pbr_hdr({0.5f, 0.5f, 0.5f}, {0.f, 0.f, 0.f}, 0, 1.f, 1.f, hdr, 0.f, 2.f, agx);
      for (const float c : out) {
        EXPECT_TRUE(std::isfinite(c));
        EXPECT_GE(c, 0.f);
        EXPECT_LE(c, 1.f);
      }
    }
  }
}

GraphEnvironment radiance_env() {
  GraphEnvironment env;
  env.gxFormats[static_cast<size_t>(GxResource::Color)] = wgpu::TextureFormat::BGRA8Unorm;
  env.gxFormats[static_cast<size_t>(GxResource::Fog)] = wgpu::TextureFormat::RGBA8Uint;
  env.gxFormats[static_cast<size_t>(GxResource::Normal)] = wgpu::TextureFormat::RGB10A2Unorm;
  env.gxFormats[static_cast<size_t>(GxResource::Motion)] = wgpu::TextureFormat::RG16Float;
  return env;
}

struct RadianceForcedReset {
  ~RadianceForcedReset() {
    composite::testing::set_radiance_forced(std::nullopt);
    composite::testing::set_flags(std::nullopt);
  }
};

TEST(CompositeRules, RadianceWantedFollowsSsil) {
  const RadianceForcedReset reset;
  composite::testing::set_radiance_forced(false);
  const GraphEnvironment env = radiance_env();
  const auto wanted = [](const Settings& settings, const GraphEnvironment& e, DepthSource depth = DepthSource::Direct) {
    return composite::radiance_wanted({.settings = &settings, .env = &e, .sceneDepth = depth});
  };
  Settings ssil = ao_settings();
  ssil.ssil = true;
  EXPECT_TRUE(wanted(ssil, env));
  EXPECT_FALSE(wanted(ao_settings(), env));
  Settings noAo = ssil;
  noAo.ao = Quality::Off;
  EXPECT_FALSE(wanted(noAo, env));
  EXPECT_FALSE(wanted(ssil, env, DepthSource::Unavailable));
  GraphEnvironment noMotion = env;
  noMotion.gxFormats[static_cast<size_t>(GxResource::Motion)] = wgpu::TextureFormat::Undefined;
  EXPECT_FALSE(wanted(ssil, noMotion));
  GraphEnvironment noFog = env;
  noFog.gxFormats[static_cast<size_t>(GxResource::Fog)] = wgpu::TextureFormat::Undefined;
  EXPECT_FALSE(wanted(ssil, noFog));
  GraphEnvironment msaa = env;
  msaa.gxSampleCount = 4;
  EXPECT_FALSE(wanted(ssil, msaa));
  EXPECT_FALSE(composite::radiance_wanted({.settings = nullptr, .env = &env, .sceneDepth = DepthSource::Direct}));
  for (const uint32_t flags : {0u, uint32_t{composite::FlagIdentity}}) {
    composite::testing::set_flags(flags);
    for (const float lb : {0.f, 0.5f, 1.f}) {
      Settings pbr = ssil;
      pbr.pbr = true;
      pbr.legacyBlend = lb;
      EXPECT_TRUE(wanted(pbr, env)) << lb;
      pbr.ssil = false;
      EXPECT_FALSE(wanted(pbr, env)) << lb;
    }
  }
  composite::testing::set_radiance_forced(true);
  EXPECT_TRUE(wanted(ao_settings(), env));
  EXPECT_FALSE(wanted(noAo, env));
  EXPECT_FALSE(wanted(ao_settings(), msaa));
  EXPECT_FALSE(wanted(ao_settings(), noMotion));
}

TEST(CompositeRules, RadianceEnvParses) {
  EXPECT_EQ(composite::RadianceEnv, "AURORA_RENDER_COMPOSITE_RADIANCE");
  EXPECT_EQ(composite::RadianceDefine, "COMPOSITE_RADIANCE");
  EXPECT_EQ(composite::RadianceFormat, wgpu::TextureFormat::RGBA16Float);
  EXPECT_TRUE(composite::radiance_env_on("on"));
  for (const char* value : {"off", "ON", "", "1", "on ", "true"}) {
    EXPECT_FALSE(composite::radiance_env_on(value)) << value;
  }
  EXPECT_FALSE(composite::radiance_env_on(nullptr));
}

float radiance_stored(float linear) { return encode_srgb(linear); }

TEST(CompositeRules, RadianceMirrorRemovesFog) {
  constexpr std::array<std::array<uint8_t, 3>, 5> Fogs{
      {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {128, 153, 179}}};
  std::mt19937 rng{0xF06u};
  std::uniform_int_distribution<int> byte(0, 255);
  for (const auto& fcBytes : Fogs) {
    const std::array<float, 3> fc{fcBytes[0] / 255.f, fcBytes[1] / 255.f, fcBytes[2] / 255.f};
    const float norm = std::sqrt(fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2]);
    double along = 0.0;
    size_t count = 0;
    for (uint32_t q7 = 0; q7 <= 127; ++q7) {
      const float q = static_cast<float>(q7) / 127.f;
      const float divisor = std::max(1.f - q, 13.f / 127.f);
      for (int i = 0; i < 64; ++i) {
        std::array<float, 3> s{};
        std::array<float, 3> l{};
        for (size_t c = 0; c < 3; ++c) {
          s[c] = static_cast<float>(byte(rng)) / 255.f;
          l[c] = std::round(255.f * ((1.f - q) * s[c] + q * fc[c])) / 255.f;
        }
        const std::array<float, 4> r = composite::testing::radiance(l, fc, q7, 1.f, 1.f, {}, false, 0.f, 1.f);
        ASSERT_EQ(r[3], 1.f) << q7;
        double dot = 0.0;
        for (size_t c = 0; c < 3; ++c) {
          const float expected = s[c] * (1.f - q) / divisor;
          const float stored = radiance_stored(r[c]);
          ASSERT_NEAR(stored, expected, 0.5f / 255.f / divisor + 2e-5f) << q7 << " " << c;
          dot += (stored - expected) * fc[c] / norm;
        }
        along += dot;
        ++count;
      }
    }
    EXPECT_LE(std::abs(along / static_cast<double>(count)) * 255.0, 0.1) << int{fcBytes[0]} << int{fcBytes[1]};
  }
}

TEST(CompositeRules, RadianceMirrorBlendOneIsLegacy) {
  std::mt19937 rng{0x1E6u};
  size_t valid = 0;
  for (int i = 0; i < 4000; ++i) {
    MirrorInput in = random_input(rng, i);
    const std::array<float, 4> legacy =
        composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, false, 0.f, 1.f);
    ASSERT_EQ(composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, true, 1.f, 1.f), legacy) << i;
    const std::array<float, 4> pbr =
        composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, true, 0.f, 1.f);
    if (legacy[3] == 0.f) {
      ASSERT_EQ(pbr, legacy) << i;
      continue;
    }
    ++valid;
    ASSERT_EQ(pbr, (std::array<float, 4>{in.hdr[0], in.hdr[1], in.hdr[2], 1.f})) << i;
    in.hdr[3] = 0.49f;
    for (const float lb : {0.f, 0.5f}) {
      ASSERT_EQ(composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, true, lb, 1.f), legacy) << i;
    }
  }
  EXPECT_GT(valid, 1000u);
}

TEST(CompositeRules, RadianceUnitsFollowFlagPbrOnly) {
  std::mt19937 rng{0x0417u};
  for (const float e : {0.5f, 1.f, 2.7f}) {
    for (const uint32_t flags : {0u, uint32_t{composite::FlagIdentity}, uint32_t{composite::FlagNoFog},
                                 composite::FlagTonemapAgX, composite::FlagIdentity | composite::FlagTonemapAgX}) {
      EXPECT_EQ(composite::testing::units(flags, e, 1.f / e), (std::array<float, 2>{1.f, 1.f})) << flags;
      EXPECT_EQ(composite::testing::units(flags | composite::FlagPbr, e, 1.f / e), (std::array<float, 2>{e, 1.f / e}))
          << flags;
    }
    // A legacy receiver's bounce multiplies by .x, so legacy-to-legacy light does not depend on e
    const std::array<float, 2> units = composite::testing::units(composite::FlagPbr, e, 1.f / e);
    for (int i = 0; i < 500; ++i) {
      MirrorInput in = random_input(rng, i);
      in.hdr[3] = 0.f;
      const std::array<float, 4> plain =
          composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, true, 0.f, 1.f);
      const std::array<float, 4> scaled =
          composite::testing::radiance(in.l, in.fc, in.q7, in.ao, in.alpha, in.hdr, true, 0.f, units[1]);
      for (size_t c = 0; c < 3; ++c) {
        ASSERT_NEAR(scaled[c] * units[0], plain[c], 1e-6f * std::max(plain[c], 1.f)) << i;
      }
      ASSERT_EQ(scaled[3], plain[3]);
    }
  }
}

TEST(CompositeRules, RadianceLegacyBiasIsPinned) {
  constexpr float Grey = 0.18f;
  for (const bool agx : {false, true}) {
    const lighting::tonemap::Rgb mapped = lighting::tonemap::apply({Grey, Grey, Grey}, agx);
    EXPECT_NEAR(mapped[0], agx ? 0.2148f : 0.14f, 1e-3f) << agx;
    const float y = std::round(encode_srgb(mapped[0]) * 255.f) / 255.f;
    for (const float e : {0.5f, 1.f, 2.7f}) {
      // The legacy branch stores lin(y)/e with no tonemap inverse, so it reads T(0.18), not 0.18
      const std::array<float, 4> r = composite::testing::radiance({y, y, y}, {}, 0, 1.f, 1.f, {}, true, 0.f, 1.f / e);
      EXPECT_NEAR(r[0] * e / Grey, mapped[0] / Grey, 0.015f) << agx << " " << e;
    }
  }
}

TEST(CompositeRules, RadianceFogCutMatchesMeter) {
  EXPECT_EQ(composite::RadianceFogCutQ7, lighting::exposure::FogCutQ7);
  EXPECT_EQ(static_cast<float>(127u - composite::RadianceFogCutQ7) / 127.f, 13.f / 127.f);
  const std::array<float, 3> fc{0.8f, 0.6f, 0.4f};
  for (uint32_t q7 = 100; q7 <= 127; ++q7) {
    // The pixel is fog colour all through, so above the cut its radiance only darkens toward black
    const std::array<float, 4> r = composite::testing::radiance(fc, fc, q7, 1.f, 1.f, {}, false, 0.f, 1.f);
    const float q = static_cast<float>(q7) / 127.f;
    const float scale = q7 <= composite::RadianceFogCutQ7 ? 1.f : (1.f - q) / (13.f / 127.f);
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_NEAR(radiance_stored(r[c]), fc[c] * scale, 2e-5f) << q7;
      ASSERT_LE(radiance_stored(r[c]), fc[c] + 2e-5f) << q7;
    }
  }
  const std::array<float, 4> drowned = composite::testing::radiance(fc, fc, 127, 1.f, 1.f, {}, false, 0.f, 1.f);
  EXPECT_EQ(drowned, (std::array<float, 4>{0.f, 0.f, 0.f, 1.f}));
}

struct RadianceProbe {
  bool prev = false;
  int calls = 0;
  std::vector<bool> prevValid;
  wgpu::Buffer readback;
  uint32_t rowBytes = 0;
  wgpu::Extent3D extent{};
};

// A prev probe reads last frame's radiance where SSIL will; a copy probe reads this frame's after the composite
void declare_radiance_probe(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& probe = *static_cast<const RadianceProbe*>(state);
  if (probe.prev) {
    builder.after(phase::Ao);
    builder.use_prev(resource::RadianceDefogged);
  } else {
    builder.after(composite::Pass);
    builder.use(resource::RadianceDefogged, Access::CopySrc);
  }
  builder.side_effect();
}

void encode_radiance_probe(PassContext& ctx, const void* state) {
  auto& probe = *static_cast<RadianceProbe*>(const_cast<void*>(state));
  ++probe.calls;
  if (probe.prev) {
    probe.prevValid.push_back(ctx.prev_valid(TexRef{0}));
    return;
  }
  probe.extent = ctx.extent(TexRef{0});
  probe.rowBytes = (probe.extent.width * 8 + 255) & ~255u;
  const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                          .size = uint64_t{probe.rowBytes} * probe.extent.height};
  probe.readback = ctx.device().CreateBuffer(&descriptor);
  const wgpu::TexelCopyTextureInfo source{.texture = ctx.texture(TexRef{0})};
  const wgpu::TexelCopyBufferInfo destination{
      .layout = {.bytesPerRow = probe.rowBytes, .rowsPerImage = probe.extent.height}, .buffer = probe.readback};
  const wgpu::Extent3D size{probe.extent.width, probe.extent.height, 1};
  ctx.encoder().CopyTextureToBuffer(&source, &destination, &size);
}

PassDesc radiance_probe_desc(RadianceProbe& probe) {
  return {
      .name = probe.prev ? "t.radiance_prev" : "t.radiance_copy",
      .stage = Stage::OpaqueDone,
      .kind = probe.prev ? PassKind::Compute : PassKind::Copy,
      .order = 0,
      .runsWithoutMarker = true,
      .declare = declare_radiance_probe,
      .encode = encode_radiance_probe,
      .state = &probe,
  };
}

// History resets read the scene hints, so tests that cut the camera emit it through the hint sink
void emit_camera(uint32_t cutEpoch) {
  begin_hint_frame();
  gx::hints::Camera camera = camera_hints()->mainCamera;
  camera.view = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
  camera.cutEpoch = cutEpoch;
  const gx::hints::Sink& sink = *gx::hints::sink();
  sink.camera(sink.userData, camera);
}

class CompositeDeviceTest : public ::testing::Test {
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
    composite::shutdown();
    composite::testing::reset();
    lighting::deferred::shutdown();
    lighting::deferred::testing::reset();
    lighting::openpbr::shutdown();
    lighting::clusters::shutdown();
    lighting::clusters::testing::reset();
    gx_ext::surface_table::unregister_passes();
    ao::shutdown();
    ao::testing::reset();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  void start_modules() {
    composite::initialize();
    // No meter or probe module runs here, so tests that want the PBR term stand in for both
    composite::testing::set_pbr_producers_override(true);
    m_hints = camera_hints();
    composite::testing::set_camera_override(m_hints);
    ao::testing::set_camera_override(m_hints);
  }

  void use_globals(const wgpu::Extent3D& size) {
    webgpu::set_gbuffer(false);
    m_globals.reset();
    m_options.efbSize = size;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
    start_modules();
    ASSERT_TRUE(webgpu::set_gbuffer(true));
  }

  static bool use_pbr_tier() {
    render::Caps c = render::caps();
    c.limits.maxColorAttachmentBytesPerSample = 128;
    webgpu::set_caps(c);
    return webgpu::set_scene_tier(webgpu::SceneTier::Pbr);
  }

  // These tests build no light.deferred, so a stub creates scene.hdr and the per-frame lighting term is forced
  static void produce_hdr(StubHdr& stub) {
    register_pass(stub_hdr_desc(stub));
    lighting::deferred::testing::force_enabled(true);
  }

  static composite::Params latched_params(const CompiledGraph& graph) {
    composite::Params params;
    const std::span<const std::byte> bytes = latched_pass_params(graph.nodes[graph.find_node(composite::Pass)].id);
    EXPECT_EQ(bytes.size(), sizeof(params));
    std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(params)));
    return params;
  }

  static bool same_params(const composite::Params& a, const composite::Params& b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
  }

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(const Settings& settings) {
    m_settings = settings;
    m_env = gather_graph_environment();
    if (m_sceneSize) {
      m_env.sceneSize = *m_sceneSize;
    }
    if (m_envEdit) {
      m_envEdit(m_env);
    }
    std::shared_ptr<CompiledGraph> graph =
        compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = DepthSource::Direct});
    prepare_graph_bindings(
        *graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = DepthSource::Direct});
    return graph;
  }

  void latch(const CompiledGraph* graph) {
    const PassSetup latched{.settings = &m_settings, .env = &m_env};
    ao::on_latch(latched, graph);
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
    // A freed graph's address can come back, and the latch only re-checks a graph it has not seen
    composite::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
    latch(graph.get());
    return graph;
  }

  StageData build(const FrameState& state, uint64_t frameId, bool marker = true) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    StageData opaque{
        .stage = Stage::OpaqueDone,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = marker,
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

  std::optional<std::string> run(const FrameState& state, uint64_t frameId) {
    StageData opaque = build(state, frameId);
    return encode(opaque);
  }

  void make_depth() {
    const wgpu::Extent3D size = webgpu::g_frameBuffer.size;
    m_depth = test::make_depth(*m_device, size.width, size.height);
  }

  static inline std::vector<uint8_t> s_uniforms;
  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
  std::optional<GraphExtent> m_sceneSize;
  std::function<void(GraphEnvironment&)> m_envEdit;
  std::shared_ptr<const SceneHints> m_hints;
  wgpu::Texture m_depth;
};

class CompositeGraph : public CompositeDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 64, 1}})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    make_depth();
  }

  void expect_ran(const CompiledGraph& graph, const StageData& stage, bool ran) const {
    EXPECT_EQ(pass_enabled(graph, stage, composite::Pass), ran);
    EXPECT_EQ(pass_enabled(graph, stage, composite::CopyPass), ran);
  }

  void expect_refused(StubAo& stub, std::string_view what) {
    SCOPED_TRACE(what);
    const uint32_t failures = composite::testing::counters().contractFailures;
    register_pass(stub_ao_desc(stub));
    const auto graph = live_graph(ao_settings());
    ASSERT_EQ(graph->nodes[graph->find_node(composite::Pass)].status, CullReason::None);
    const std::optional<std::string> reason = composite::check_contract(*graph);
    ASSERT_TRUE(reason.has_value());
    EXPECT_NE(reason->find(resource::AoFinal), std::string::npos) << *reason;
    EXPECT_FALSE(composite::live());
    EXPECT_FALSE(composite::ready(m_settings));
    EXPECT_EQ(composite::testing::counters().contractFailures, failures + 1);
    latch(graph.get());
    EXPECT_EQ(composite::testing::counters().contractFailures, failures + 1);
    const FrameState state{.graph = graph};
    const StageData opaque = build(state, 1);
    expect_ran(*graph, opaque, false);
    EXPECT_FALSE(composite::enabled_this_frame(1));
    EXPECT_EQ(composite::testing::counters().draws, 0u);
    EXPECT_EQ(composite::testing::counters().copies, 0u);
  }

  void encode_sizes(bool pbr = false) {
    for (const auto [width, height] :
         {std::pair{1u, 1u}, std::pair{61u, 37u}, std::pair{1280u, 720u}, std::pair{4779u, 9u}}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      ASSERT_NO_FATAL_FAILURE(use_globals({width, height, 1}));
      make_depth();
      if (pbr) {
        ASSERT_TRUE(use_pbr_tier());
      }
      for (const uint32_t flags : {0u, uint32_t{composite::FlagIdentity}, uint32_t{composite::FlagNoFog}}) {
        composite::testing::reset();
        composite::testing::set_camera_override(m_hints);
        composite::testing::set_pbr_producers_override(true);
        composite::testing::set_flags(flags);
        StubAo stub;
        const PassId stubId = register_pass(stub_ao_desc(stub));
        StubHdr hdr;
        const PassId hdrId = pbr ? register_pass(stub_hdr_desc(hdr)) : InvalidPassId;
        lighting::deferred::testing::force_enabled(pbr ? std::optional<bool>{true} : std::nullopt);
        const auto graph = live_graph(pbr ? pbr_settings(0.f, Quality::High) : ao_settings());
        ASSERT_TRUE(composite::live());
        EXPECT_EQ(composite::surface_bound(*graph), pbr);
        const FrameState state{.graph = graph};
        EXPECT_EQ(run(state, 1), std::nullopt);
        EXPECT_EQ(run(state, 2), std::nullopt);
        EXPECT_EQ(composite::testing::counters().copies, 2u);
        EXPECT_EQ(composite::testing::counters().draws, 2u);
        EXPECT_EQ(composite::testing::counters().pbrDraws, pbr && !composite::identity_forced() ? 2u : 0u);
        EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
        EXPECT_EQ(stub.calls, 2);
        unregister_pass(stubId);
        if (hdrId != InvalidPassId) {
          unregister_pass(hdrId);
        }
      }
    }
  }
};

TEST_F(CompositeGraph, SkippedWhenAoOff) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  for (const Settings& settings : {Settings{}, Settings{.taa = true}, Settings{.upscaler = Upscaler::Fsr}}) {
    const auto graph = compile(settings);
    for (const std::string_view name : {composite::CopyPass, composite::Pass}) {
      const NodeId node = graph->find_node(name);
      ASSERT_NE(node, InvalidNode) << name;
      EXPECT_EQ(graph->nodes[node].status, CullReason::Skipped) << name;
      EXPECT_EQ(graph->nodes[node].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{})) << name;
    }
    EXPECT_EQ(graph->find_resource(composite::LegacyTex), InvalidResource);
    latch(graph.get());
    EXPECT_FALSE(composite::live());
    EXPECT_TRUE(composite::ready(settings));
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, 1), std::nullopt);
  }
  EXPECT_EQ(composite::testing::counters().copies, 0u);
  EXPECT_EQ(composite::testing::counters().draws, 0u);
  EXPECT_EQ(composite::testing::counters().contractFailures, 0u);
  EXPECT_EQ(shaders::stats().ready + shaders::stats().pending + shaders::stats().failed, 0u);
}

TEST_F(CompositeGraph, TaaOnlyGraphHashUnchanged) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  TestPass reader{
      .name = "t.after_taa", .stage = Stage::PrePost, .kind = PassKind::Copy, .declare = [](InternalPassBuilder& b) {
        b.use(resource::GxColor, Access::CopySrc);
        b.side_effect();
      }};
  register_pass(desc_of(reader));
  for (const Settings& settings : {Settings{.taa = true}, Settings{.upscaler = Upscaler::Fsr}, Settings{}}) {
    const auto with = compile(settings);
    composite::shutdown();
    const auto without = compile(settings);
    composite::initialize();
    EXPECT_FALSE(with->stage_empty(Stage::PrePost));
    EXPECT_EQ(with->graphHash, without->graphHash);
    EXPECT_EQ(with->find_resource(composite::LegacyTex), InvalidResource);
  }
}

TEST_F(CompositeGraph, SkippedWithoutFogRecord) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  webgpu::set_gbuffer(false);
  const auto graph = compile(ao_settings());
  EXPECT_EQ(graph->nodes[graph->find_node(composite::Pass)].status, CullReason::Skipped);
  EXPECT_EQ(graph->nodes[graph->find_node(composite::CopyPass)].status, CullReason::Skipped);
  ASSERT_TRUE(webgpu::set_gbuffer(true));
}

TEST_F(CompositeGraph, OrderCopyAoComposite) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = compile(ao_settings());
  ASSERT_EQ(composite::check_contract(*graph), std::nullopt);
  const size_t prepare = program_index(*graph, phase::Prepare);
  const size_t copy = program_index(*graph, composite::CopyPass);
  const size_t aoPhase = program_index(*graph, phase::Ao);
  const size_t ao = program_index(*graph, "test.ao");
  const size_t compositePhase = program_index(*graph, phase::Composite);
  const size_t main = program_index(*graph, composite::Pass);
  const size_t overlay = program_index(*graph, phase::Overlay);
  ASSERT_NE(main, SIZE_MAX);
  EXPECT_LT(prepare, copy);
  EXPECT_LT(copy, aoPhase);
  EXPECT_LT(aoPhase, ao);
  EXPECT_LT(ao, compositePhase);
  EXPECT_LT(compositePhase, main);
  EXPECT_LT(main, overlay);
  EXPECT_EQ(graph->nodes[graph->find_node(composite::Pass)].order, 0u);
  EXPECT_EQ(graph->nodes[graph->find_node(composite::CopyPass)].order, 0u);
}

TEST_F(CompositeGraph, BindingOrderLegacyFogNormalAoSurfaceHdr) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = compile(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 10u);
  const std::array<std::string_view, 10> names{
      composite::LegacyTex,      resource::Fog,      resource::GxNormal,           resource::AoFinal,
      resource::Surface,         resource::SceneHdr, lighting::exposure::Resource, resource::SsilIrradiance,
      fakeshadow::lsm::Resource, resource::GxColor};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, names[i]) << i;
    EXPECT_EQ(uses[i].access, i == 6 ? Access::Uniform : i == 9 ? Access::ColorRMW : Access::Sampled) << i;
    if (i < 4) {
      EXPECT_EQ(uses[i].missing, Missing::DisablePass) << i;
    }
  }
  EXPECT_EQ(uses[composite::BindFog].options.sampleType, wgpu::TextureSampleType::Uint);
  EXPECT_EQ(uses[composite::BindSurface].missing, Missing::BindDummyBlack);
  EXPECT_EQ(uses[composite::BindSurface].options.sampleType, wgpu::TextureSampleType::Uint);
  EXPECT_EQ(uses[composite::BindHdr].missing, Missing::BindDummyBlack);
  EXPECT_TRUE(uses[composite::BindHdr].staticDummy);
  EXPECT_EQ(uses[composite::BindExposure].missing, Missing::BindDummyWhite);
  EXPECT_TRUE(uses[composite::BindExposure].buffer);
  EXPECT_EQ(uses[composite::BindExposure].options.dummySize, lighting::exposure::BufferBytes);
  EXPECT_EQ(uses[composite::BindExposure].visibility, wgpu::ShaderStage::Fragment);
  EXPECT_TRUE(uses[composite::BindExposure].staticDummy);
  EXPECT_EQ(graph->nodes[node].paramsSize, sizeof(composite::Params));
}

TEST_F(CompositeGraph, BindingOrderAppendsExposure) {
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  StubExposure meter;
  register_pass(stub_exposure_desc(meter));
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 10u);
  EXPECT_EQ(graph->resources[uses[composite::BindExposure].resource].name, lighting::exposure::Resource);
  EXPECT_FALSE(uses[composite::BindExposure].staticDummy);
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
  EXPECT_TRUE(composite::live());
}

TEST_F(CompositeGraph, ExposureBindsWhiteDummyUnderTierB) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubExposure meter;
  const PassId meterId = register_pass(stub_exposure_desc(meter));
  const auto graph = live_graph(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_FALSE(graph->node_uses(node)[composite::BindExposure].staticDummy);
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
  unregister_pass(meterId);
  const auto bare = live_graph(ao_settings());
  EXPECT_TRUE(bare->node_uses(bare->find_node(composite::Pass))[composite::BindExposure].staticDummy);
  EXPECT_EQ(composite::check_contract(*bare), std::nullopt);
}

TEST_F(CompositeGraph, ExposureFromAnotherPassRefuses) {
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  const TestPass impostor{.name = "t.exposure", .kind = PassKind::Encoder, .declare = [](InternalPassBuilder& b) {
                            b.after(phase::Exposure);
                            b.create_buffer(lighting::exposure::Resource, {.size = 16}, Access::CopyDst);
                          }};
  register_pass(desc_of(impostor));
  const auto graph = live_graph(pbr_settings(0.f));
  const std::optional<std::string> reason = composite::check_contract(*graph);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find(lighting::exposure::Pass), std::string::npos) << *reason;
  EXPECT_FALSE(composite::live());
}

TEST_F(CompositeGraph, AgxFlagOnlyWhilePbrActive) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  Settings agx = pbr_settings(0.f, Quality::High);
  agx.tonemap = Tonemap::AgX;
  const auto graph = live_graph(agx);
  ASSERT_TRUE(composite::pbr_active());
  EXPECT_EQ(latched_params(*graph).flags & (composite::FlagPbr | composite::FlagTonemapAgX),
            composite::FlagPbr | composite::FlagTonemapAgX);
  m_settings.tonemap = Tonemap::PbrNeutral;
  latch(graph.get());
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagTonemapAgX, 0u);
  m_settings.tonemap = Tonemap::AgX;
  m_settings.legacyBlend = 1.f;
  latch(graph.get());
  EXPECT_EQ(latched_params(*graph).flags & (composite::FlagPbr | composite::FlagTonemapAgX), 0u);
  m_settings.legacyBlend = 0.f;
  composite::testing::set_pbr_producers_override(false);
  latch(graph.get());
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_EQ(latched_params(*graph).flags & (composite::FlagPbr | composite::FlagTonemapAgX), 0u);
}

TEST_F(CompositeGraph, PbrWaitsForMeterPipelines) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  composite::testing::set_pbr_producers_override(std::nullopt);
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_TRUE(composite::hdr_produced(*graph));
  EXPECT_FALSE(lighting::exposure::pipelines_live());
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagPbr, 0u);
  composite::testing::set_pbr_producers_override(true);
  latch(graph.get());
  EXPECT_TRUE(composite::pbr_active());
}

TEST_F(CompositeGraph, HdrBindsBlackDummyWithoutLighting) {
  ASSERT_TRUE(use_pbr_tier());
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_TRUE(graph->node_uses(node)[composite::BindHdr].staticDummy);
  EXPECT_FALSE(composite::hdr_produced(*graph));
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
  EXPECT_TRUE(composite::live());
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagPbr, 0u);
  StubHdr hdr;
  produce_hdr(hdr);
  const auto lit = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_FALSE(lit->node_uses(lit->find_node(composite::Pass))[composite::BindHdr].staticDummy);
  EXPECT_TRUE(composite::hdr_produced(*lit));
  EXPECT_EQ(composite::pbr_active(), !composite::identity_forced());
  EXPECT_EQ(lit->bindingPlans[lit->find_node(composite::Pass)].layoutHash, graph->bindingPlans[node].layoutHash);
}

TEST_F(CompositeGraph, HdrContractMismatchRefuses) {
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr{.format = wgpu::TextureFormat::RGBA8Unorm};
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f));
  ASSERT_EQ(graph->nodes[graph->find_node(composite::Pass)].status, CullReason::None);
  const std::optional<std::string> reason = composite::check_contract(*graph);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find(resource::SceneHdr), std::string::npos) << *reason;
  EXPECT_FALSE(composite::live());
  EXPECT_FALSE(composite::pbr_active());
}

TEST_F(CompositeGraph, IdleWithoutLighting) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  lighting::deferred::testing::force_enabled(false);
  const auto graph = live_graph(pbr_settings(0.f));
  ASSERT_TRUE(composite::pbr_active());
  const FrameState state{.graph = graph};
  const StageData opaque = build(state, 1);
  expect_ran(*graph, opaque, false);
  EXPECT_FALSE(composite::pbr_applied(opaque));
  EXPECT_EQ(composite::testing::counters().idleFrames, 1u);
  lighting::deferred::testing::force_enabled(true);
  expect_ran(*graph, build(state, 2), true);
  EXPECT_EQ(composite::testing::counters().idleFrames, 1u);
}

TEST_F(CompositeGraph, SurfaceBindsBlackDummyUnderTierB) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto base = compile(ao_settings());
  const NodeId node = base->find_node(composite::Pass);
  ASSERT_EQ(base->nodes[node].status, CullReason::None);
  EXPECT_TRUE(base->node_uses(node)[composite::BindSurface].staticDummy);
  EXPECT_FALSE(composite::surface_bound(*base));
  EXPECT_TRUE(composite::ao_produced(*base));
  EXPECT_EQ(composite::check_contract(*base), std::nullopt);
  ASSERT_TRUE(use_pbr_tier());
  const auto pbr = compile(ao_settings());
  const NodeId pbrNode = pbr->find_node(composite::Pass);
  ASSERT_EQ(pbr->nodes[pbrNode].status, CullReason::None);
  EXPECT_FALSE(pbr->node_uses(pbrNode)[composite::BindSurface].staticDummy);
  EXPECT_TRUE(composite::surface_bound(*pbr));
  EXPECT_EQ(composite::check_contract(*pbr), std::nullopt);
  EXPECT_EQ(base->bindingPlans[node].layoutHash, pbr->bindingPlans[pbrNode].layoutHash);
}

TEST_F(CompositeGraph, PbrOffKeepsAoDisablePass) {
  ASSERT_TRUE(use_pbr_tier());
  const auto missing = compile(ao_settings());
  EXPECT_NE(missing->nodes[missing->find_node(composite::Pass)].status, CullReason::None);
  EXPECT_NE(missing->nodes[missing->find_node(composite::CopyPass)].status, CullReason::None);
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  EXPECT_EQ(graph->node_uses(node)[composite::BindAo].missing, Missing::DisablePass);
  EXPECT_TRUE(composite::ao_produced(*graph));
  EXPECT_TRUE(composite::surface_bound(*graph));
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagPbr, 0u);
  const FrameState state{.graph = graph};
  expect_ran(*graph, build(state, 1), true);
  stub.enabled = false;
  expect_ran(*graph, build(state, 2), false);
}

TEST_F(CompositeGraph, PbrDeclaresWhiteAoDummy) {
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f));
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  ASSERT_EQ(graph->nodes[graph->find_node(composite::CopyPass)].status, CullReason::None);
  const CompiledUse& ao = graph->node_uses(node)[composite::BindAo];
  EXPECT_EQ(ao.missing, Missing::BindDummyWhite);
  EXPECT_TRUE(ao.staticDummy);
  EXPECT_FALSE(composite::ao_produced(*graph));
  EXPECT_TRUE(composite::surface_bound(*graph));
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
  EXPECT_TRUE(composite::live());
  EXPECT_TRUE(composite::ready(m_settings));
  if (!composite::identity_forced()) {
    EXPECT_TRUE(composite::pbr_active());
    const composite::Params params = latched_params(*graph);
    EXPECT_EQ(params.flags & composite::FlagPbr, composite::FlagPbr);
    EXPECT_EQ(params.legacyBlend, 0.f);
    const FrameState state{.graph = graph};
    expect_ran(*graph, build(state, 1), true);
  }
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  const auto base = compile(pbr_settings(0.f));
  EXPECT_NE(base->nodes[base->find_node(composite::Pass)].status, CullReason::None);
}

TEST_F(CompositeGraph, PbrWithoutAoProducerKeepsGameShadows) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_TRUE(composite::live());
  EXPECT_TRUE(composite::ready(m_settings));
  EXPECT_TRUE(composite::pbr_active());
  EXPECT_FALSE(composite::ao_produced(*graph));
  EXPECT_FALSE(composite::ao_gate(FrameState{.settings = m_settings, .graph = graph}));
  EXPECT_FALSE(composite::ao_ready(m_settings));
  EXPECT_TRUE(composite::ao_ready(pbr_settings(0.f)));
  EXPECT_TRUE(composite::ready(ao_settings()));
  EXPECT_FALSE(composite::ao_ready(ao_settings()));
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto withAo = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_TRUE(composite::ao_produced(*withAo));
  EXPECT_TRUE(composite::ao_gate(FrameState{.settings = m_settings, .graph = withAo}));
  EXPECT_TRUE(composite::ao_ready(m_settings));
}

TEST_F(CompositeGraph, BlendOneWithoutAoRunsNothing) {
  ASSERT_TRUE(use_pbr_tier());
  const auto graph = live_graph(pbr_settings(1.f));
  ASSERT_TRUE(composite::live());
  EXPECT_FALSE(composite::pbr_active());
  const composite::Params params = latched_params(*graph);
  EXPECT_EQ(params.flags & composite::FlagPbr, 0u);
  EXPECT_EQ(params.legacyBlend, 0.f);
  const FrameState state{.graph = graph};
  const StageData opaque = build(state, 1);
  expect_ran(*graph, opaque, false);
  EXPECT_FALSE(composite::enabled_this_frame(1));
  EXPECT_FALSE(composite::pbr_applied(opaque));
  EXPECT_EQ(composite::testing::counters().idleFrames, 1u);
  EXPECT_EQ(run(state, 2), std::nullopt);
  EXPECT_EQ(composite::testing::counters().copies, 0u);
  EXPECT_EQ(composite::testing::counters().draws, 0u);
  EXPECT_EQ(composite::testing::counters().idleFrames, 2u);
}

TEST_F(CompositeGraph, PbrParamsMatchPbrOffWhenInactive) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubHdr hdr;
  produce_hdr(hdr);
  const auto off = live_graph(ao_settings());
  const composite::Params offParams = latched_params(*off);
  ASSERT_TRUE(use_pbr_tier());
  const auto on = live_graph(pbr_settings(1.f, Quality::High));
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_TRUE(same_params(latched_params(*on), offParams));
  m_settings.legacyBlend = 0.f;
  m_settings.debugView = 1;
  latch(on.get());
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_TRUE(same_params(latched_params(*on), offParams));
  m_settings.debugView = 0;
  latch(on.get());
  if (!composite::identity_forced()) {
    EXPECT_TRUE(composite::pbr_active());
    EXPECT_FALSE(same_params(latched_params(*on), offParams));
  }
  m_settings.legacyBlend = 1.f;
  latch(on.get());
  EXPECT_TRUE(same_params(latched_params(*on), offParams));
}

TEST_F(CompositeGraph, BlendChangeKeepsGraphHash) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubHdr hdr;
  produce_hdr(hdr);
  constexpr std::string_view HistoryName = "t.history";
  const TestPass history{
      .name = "t.history_writer", .kind = PassKind::Compute, .declare = [](InternalPassBuilder& b) {
        b.after(phase::Ao);
        b.history("t.history",
                  {.format = wgpu::TextureFormat::RGBA8Unorm, .extraUsage = wgpu::TextureUsage::TextureBinding},
                  Access::StorageWrite);
        b.side_effect();
      }};
  register_pass(desc_of(history));
  ASSERT_TRUE(use_pbr_tier());
  const auto first = live_graph(pbr_settings(1.f, Quality::High));
  const NodeId node = first->find_node(composite::Pass);
  const shaders::PipelineStats before = shaders::stats();
  uint64_t frame = 0;
  const auto history_kept = [&](const std::shared_ptr<CompiledGraph>& graph, std::string_view name) {
    const FrameState state{.graph = graph};
    StageData stage = build(state, ++frame);
    const auto resource = graph->find_resource(name);
    if (resource == InvalidResource) {
      ADD_FAILURE() << "no " << name << " in the graph";
      return false;
    }
    const bool kept = stage.prev_valid(graph->resources[resource].historySlot);
    EXPECT_EQ(encode(stage), std::nullopt);
    return kept;
  };
  EXPECT_FALSE(history_kept(first, HistoryName));
  EXPECT_TRUE(history_kept(first, HistoryName));
  for (const float lb : {0.5f, 0.f, 1.f}) {
    const auto graph = live_graph(pbr_settings(lb, Quality::High));
    EXPECT_EQ(graph->graphHash, first->graphHash) << lb;
    EXPECT_EQ(graph->nodes[graph->find_node(composite::Pass)].pipelineIds, first->nodes[node].pipelineIds) << lb;
    const shaders::PipelineStats now = shaders::stats();
    EXPECT_EQ(now.ready + now.pending + now.failed, before.ready + before.pending + before.failed) << lb;
    EXPECT_EQ(composite::pbr_active(), lb < 1.f && !composite::identity_forced()) << lb;
    EXPECT_TRUE(history_kept(graph, HistoryName)) << lb;
  }
  composite::testing::set_radiance_forced(true);
  const auto radiance = live_graph(pbr_settings(1.f, Quality::High));
  ASSERT_TRUE(composite::radiance_bound(*radiance));
  EXPECT_NE(radiance->graphHash, first->graphHash);
  EXPECT_FALSE(history_kept(radiance, resource::RadianceDefogged));
  EXPECT_TRUE(history_kept(radiance, resource::RadianceDefogged));
  for (const float lb : {0.5f, 0.f, 1.f}) {
    const auto graph = live_graph(pbr_settings(lb, Quality::High));
    EXPECT_EQ(graph->graphHash, radiance->graphHash) << lb;
    EXPECT_EQ(graph->nodes[graph->find_node(composite::Pass)].pipelineIds,
              radiance->nodes[radiance->find_node(composite::Pass)].pipelineIds)
        << lb;
    EXPECT_TRUE(history_kept(graph, resource::RadianceDefogged)) << lb;
  }
}

TEST_F(CompositeGraph, BindingOrderAppendsLsm) {
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  const auto bare = live_graph(pbr_settings(0.f, Quality::High));
  const auto bareUses = bare->node_uses(bare->find_node(composite::Pass));
  ASSERT_EQ(bareUses.size(), 10u);
  EXPECT_EQ(bare->resources[bareUses[composite::BindLsm].resource].name, fakeshadow::lsm::Resource);
  EXPECT_TRUE(bareUses[composite::BindLsm].staticDummy);
  StubLsm mask;
  register_pass(stub_lsm_desc(mask));
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 10u);
  EXPECT_EQ(graph->resources[uses[composite::BindLsm].resource].name, fakeshadow::lsm::Resource);
  EXPECT_EQ(uses[composite::BindLsm].access, Access::Sampled);
  EXPECT_EQ(uses[composite::BindLsm].missing, Missing::BindDummyWhite);
  EXPECT_FALSE(uses[composite::BindLsm].staticDummy);
  EXPECT_EQ(graph->resources[uses[composite::BindLsm + 1].resource].name, resource::GxColor);
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
}

TEST_F(CompositeGraph, LsmBindsWhiteDummyWhenOff) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const auto uses = graph->node_uses(graph->find_node(composite::Pass));
  ASSERT_GT(uses.size(), composite::BindLsm);
  EXPECT_EQ(graph->resources[uses[composite::BindLsm].resource].name, fakeshadow::lsm::Resource);
  EXPECT_EQ(uses[composite::BindLsm].missing, Missing::BindDummyWhite);
  EXPECT_TRUE(uses[composite::BindLsm].staticDummy);
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagLsm, 0u);
}

TEST_F(CompositeGraph, FlagLsmOnlyWhenPassRan) {
  ASSERT_TRUE(use_pbr_tier());
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubHdr hdr;
  produce_hdr(hdr);
  StubLsm mask;
  register_pass(stub_lsm_desc(mask));
  uint64_t frame = 0;
  const auto flags_of = [&](const std::shared_ptr<CompiledGraph>& graph) -> std::optional<uint32_t> {
    const FrameState state{.graph = graph};
    StageData stage = build(state, ++frame);
    EXPECT_EQ(encode(stage), std::nullopt);
    const NodeId node = graph->find_node(composite::Pass);
    for (const StageBindings::Params& params : stage.bindings.params) {
      if (params.node == node) {
        composite::Params out;
        std::memcpy(&out, params.bytes.data(), sizeof(out));
        return out.flags;
      }
    }
    return std::nullopt;
  };
  // The latched params never carry it; only a frame whose mask pass ran sets it at stage build
  const auto pbr = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_EQ(latched_params(*pbr).flags & composite::FlagLsm, 0u);
  const std::optional<uint32_t> ran = flags_of(pbr);
  ASSERT_TRUE(ran.has_value());
  EXPECT_NE(*ran & composite::FlagPbr, 0u);
  EXPECT_NE(*ran & composite::FlagLsm, 0u);
  mask.enabled = false;
  const std::optional<uint32_t> skipped = flags_of(pbr);
  ASSERT_TRUE(skipped.has_value());
  EXPECT_EQ(*skipped & composite::FlagLsm, 0u);
  mask.enabled = true;
  const auto blendOne = live_graph(pbr_settings(1.f, Quality::High));
  EXPECT_EQ(blendOne->graphHash, pbr->graphHash);
  const std::optional<uint32_t> legacy = flags_of(blendOne);
  ASSERT_TRUE(legacy.has_value());
  EXPECT_EQ(*legacy & (composite::FlagPbr | composite::FlagLsm), 0u);
  composite::testing::force_params(composite::Params{.flags = composite::FlagPbr | composite::FlagIdentity});
  const auto identity = live_graph(pbr_settings(0.f, Quality::High));
  const std::optional<uint32_t> forced = flags_of(identity);
  ASSERT_TRUE(forced.has_value());
  EXPECT_EQ(*forced & composite::FlagLsm, 0u);
  composite::testing::force_params(std::nullopt);
}

TEST_F(CompositeGraph, BindingOrderAppendsSsil) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubSsil ssil;
  register_pass(stub_ssil_desc(ssil));
  composite::testing::set_radiance_forced(true);
  const auto graph = live_graph(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 11u);
  EXPECT_EQ(graph->resources[uses[composite::BindSsil].resource].name, resource::SsilIrradiance);
  EXPECT_EQ(uses[composite::BindSsil].missing, Missing::BindDummyBlack);
  EXPECT_FALSE(uses[composite::BindSsil].staticDummy);
  EXPECT_EQ(graph->resources[uses[9].resource].name, resource::GxColor);
  EXPECT_EQ(graph->resources[uses[10].resource].name, resource::RadianceDefogged);
  EXPECT_TRUE(composite::ssil_bound(*graph));
  const composite::Params params = latched_params(*graph);
  EXPECT_NE(params.flags & composite::FlagSsil, 0u);
  EXPECT_EQ(params.ssilAlbedo, ao::SsilLegacyAlbedo);
}

TEST_F(CompositeGraph, SsilBindsBlackDummyWithoutSsil) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  const auto uses = graph->node_uses(node);
  EXPECT_TRUE(uses[composite::BindSsil].staticDummy);
  EXPECT_FALSE(composite::ssil_bound(*graph));
  const composite::Params params = latched_params(*graph);
  EXPECT_EQ(params.flags & composite::FlagSsil, 0u);
  EXPECT_EQ(params.ssilAlbedo, 0.f);
  EXPECT_EQ(params.pad, 0u);
}

TEST_F(CompositeGraph, SsilContractMismatchClearsOnlyTheFlag) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubSsil ssil{.format = wgpu::TextureFormat::RGBA8Unorm};
  register_pass(stub_ssil_desc(ssil));
  const auto graph = live_graph(ao_settings());
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
  EXPECT_TRUE(composite::live());
  EXPECT_TRUE(composite::ssil_bound(*graph));
  EXPECT_EQ(latched_params(*graph).flags & composite::FlagSsil, 0u);
}

TEST_F(CompositeGraph, UnitsEpochFollowsFlagPbr) {
  ASSERT_TRUE(use_pbr_tier());
  composite::testing::set_pbr_producers_override(true);
  StubHdr hdr;
  produce_hdr(hdr);
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const uint32_t start = composite::units_epoch();
  const auto pbr = live_graph(pbr_settings(0.f, Quality::High));
  EXPECT_NE(latched_params(*pbr).flags & composite::FlagPbr, 0u);
  EXPECT_EQ(composite::units_epoch(), start + 1);
  latch(pbr.get());
  EXPECT_EQ(composite::units_epoch(), start + 1);
  m_settings.legacyBlend = 0.5f;
  latch(pbr.get());
  EXPECT_EQ(composite::units_epoch(), start + 1);
  m_settings.legacyBlend = 1.f;
  latch(pbr.get());
  EXPECT_EQ(latched_params(*pbr).flags & composite::FlagPbr, 0u);
  EXPECT_EQ(composite::units_epoch(), start + 2);
  m_settings.legacyBlend = 0.f;
  latch(pbr.get());
  EXPECT_EQ(composite::units_epoch(), start + 3);
}

TEST_F(CompositeGraph, PbrReadyNeedsTheTierWhileTheCompositeIsReady) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  struct SettingsReset {
    ~SettingsReset() {
      testing::reset();
      gfx::detail::testing::override_scene_depth_source(std::nullopt);
    }
  } reset;
  ASSERT_TRUE(use_pbr_tier());
  gfx::detail::testing::override_scene_depth_source(DepthSource::Direct);
  lighting::clusters::initialize();
  lighting::openpbr::initialize();
  lighting::deferred::initialize();
  gx_ext::surface_table::initialize();
  ASSERT_TRUE(lighting::openpbr::resident());
  testing::set_caps(caps());
  testing::set_built_options(Feature::Pbr, 1u << 1);
  apply_settings(pbr_settings(0.f, Quality::High));
  const std::shared_ptr<const FrameState> state = latch_frame();
  ASSERT_TRUE(state != nullptr && state->graph != nullptr);
  wait_ready(*state->graph);
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::Pbr));
  ASSERT_TRUE(feature_enabled(effective_settings(), Feature::DeferredFog));
  ASSERT_TRUE(composite::ready(effective_settings()));
  EXPECT_TRUE(lighting::ready(effective_settings()));
  EXPECT_TRUE(feature_ready(Feature::Pbr));
  EXPECT_TRUE(feature_ready(Feature::DeferredFog));
  EXPECT_FALSE(composite::ao_ready(effective_settings()));
  lighting::openpbr::testing::drop_resident();
  EXPECT_FALSE(feature_ready(Feature::Pbr));
  lighting::openpbr::shutdown();
  EXPECT_FALSE(feature_ready(Feature::Ao));
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  EXPECT_TRUE(composite::ready(effective_settings()));
  EXPECT_TRUE(feature_ready(Feature::DeferredFog));
  EXPECT_FALSE(feature_ready(Feature::Pbr));
}

TEST_F(CompositeGraph, SurfaceContractMismatchRefuses) {
  ASSERT_TRUE(use_pbr_tier());
  m_envEdit = [](GraphEnvironment& env) {
    env.gxFormats[static_cast<size_t>(GxResource::Surface)] = wgpu::TextureFormat::RGBA32Uint;
  };
  const auto graph = live_graph(pbr_settings(0.f));
  ASSERT_EQ(graph->nodes[graph->find_node(composite::Pass)].status, CullReason::None);
  const std::optional<std::string> reason = composite::check_contract(*graph);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find(resource::Surface), std::string::npos) << *reason;
  EXPECT_FALSE(composite::live());
  EXPECT_FALSE(composite::ready(m_settings));
  EXPECT_FALSE(composite::pbr_active());
  EXPECT_EQ(composite::testing::counters().contractFailures, 1u);
  const FrameState state{.graph = graph};
  expect_ran(*graph, build(state, 1), false);
}

TEST_F(CompositeGraph, SurfaceFollowsSceneExtent) {
  ASSERT_TRUE(use_pbr_tier());
  m_sceneSize = GraphExtent{48, 40};
  const auto graph = compile(pbr_settings(0.f));
  const CompiledResource& surface = graph->resources[graph->find_resource(resource::Surface)];
  EXPECT_EQ(surface.tex.format, wgpu::TextureFormat::RGBA16Uint);
  EXPECT_EQ(surface.extent.width, 48u);
  EXPECT_EQ(surface.extent.height, 40u);
  EXPECT_TRUE(composite::surface_bound(*graph));
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
}

TEST_F(CompositeGraph, EncodeAcceptsOneTexelDummies) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr{.enabled = false};
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f));
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1), std::nullopt);
  EXPECT_EQ(hdr.calls, 0);
  EXPECT_EQ(composite::testing::counters().draws, 1u);
  EXPECT_EQ(composite::testing::counters().pbrDraws, 1u);
  EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
  ASSERT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto base = live_graph(ao_settings());
  EXPECT_FALSE(composite::surface_bound(*base));
  const FrameState baseState{.graph = base};
  EXPECT_EQ(run(baseState, 2), std::nullopt);
  EXPECT_EQ(composite::testing::counters().draws, 2u);
  EXPECT_EQ(composite::testing::counters().pbrDraws, 1u);
  EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
}

TEST_F(CompositeGraph, PbrAppliedFollowsFinalMask) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubHdr hdr;
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f));
  const FrameState state{.graph = graph};
  uint64_t frame = 1;
  EXPECT_TRUE(composite::pbr_applied(build(state, frame++)));
  StageData other = build(state, frame++);
  other.stage = Stage::PrePost;
  EXPECT_FALSE(composite::pbr_applied(other));
  composite::testing::set_camera_override(camera_hints(false));
  EXPECT_FALSE(composite::pbr_applied(build(state, frame++)));
  composite::testing::set_camera_override(m_hints);
  m_settings.debugView = 1;
  latch(graph.get());
  EXPECT_FALSE(composite::pbr_applied(build(state, frame++)));
  m_settings.debugView = 0;
  m_settings.legacyBlend = 1.f;
  latch(graph.get());
  EXPECT_FALSE(composite::pbr_applied(build(state, frame++)));
  m_settings.legacyBlend = 0.25f;
  latch(graph.get());
  EXPECT_TRUE(composite::pbr_applied(build(state, frame++)));
  composite::testing::force_params(composite::Params{.flags = composite::FlagPbr, .legacyBlend = 1.f});
  latch(graph.get());
  EXPECT_TRUE(composite::pbr_applied(build(state, frame++)));
  composite::testing::force_params(composite::Params{});
  latch(graph.get());
  EXPECT_FALSE(composite::pbr_applied(build(state, frame++)));
}

TEST_F(CompositeGraph, PbrAppliedNeedsDeferred) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  ASSERT_TRUE(use_pbr_tier());
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  StubHdr hdr;
  produce_hdr(hdr);
  const auto graph = live_graph(pbr_settings(0.f, Quality::High));
  const FrameState state{.graph = graph};
  EXPECT_TRUE(composite::pbr_applied(build(state, 1)));
  lighting::deferred::testing::force_enabled(false);
  const StageData opaque = build(state, 2);
  expect_ran(*graph, opaque, true);
  EXPECT_FALSE(composite::pbr_applied(opaque));
}

TEST_F(CompositeGraph, LegacyTexTakesEfbFormat) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = compile(ao_settings());
  const ResourceId id = graph->find_resource(composite::LegacyTex);
  ASSERT_NE(id, InvalidResource);
  const CompiledResource& legacy = graph->resources[id];
  EXPECT_EQ(legacy.kind, ResourceKind::Transient);
  EXPECT_EQ(legacy.tex.format, m_env.gxFormats[static_cast<size_t>(GxResource::Color)]);
  EXPECT_EQ(legacy.tex.format, wgpu::TextureFormat::BGRA8Unorm);
  EXPECT_EQ(legacy.textureUsage, wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding);
  EXPECT_EQ(legacy.extent.width, 64u);
  EXPECT_EQ(legacy.extent.height, 64u);
}

TEST_F(CompositeGraph, LegacyFollowsSceneExtent) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  m_sceneSize = GraphExtent{48, 40};
  const auto graph = compile(ao_settings());
  const CompiledResource& legacy = graph->resources[graph->find_resource(composite::LegacyTex)];
  EXPECT_EQ(legacy.extent.width, 48u);
  EXPECT_EQ(legacy.extent.height, 40u);
  EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
}

TEST_F(CompositeGraph, ClassViewDisablesPerFrame) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const uint64_t hash = graph->graphHash;
  const FrameState state{.graph = graph};
  uint64_t frame = 1;
  expect_ran(*graph, build(state, frame++), true);
  ASSERT_TRUE(composite::ready(m_settings));
  const auto expect_view_off = [&] {
    latch(graph.get());
    const uint32_t skipped = composite::testing::counters().skippedView;
    const StageData opaque = build(state, frame);
    expect_ran(*graph, opaque, false);
    EXPECT_FALSE(composite::enabled_this_frame(frame));
    EXPECT_EQ(composite::testing::counters().skippedView, skipped + 1);
    EXPECT_TRUE(composite::ready(m_settings));
    EXPECT_TRUE(composite::live());
    ++frame;
  };
  m_env.viewOverride = 1;
  expect_view_off();
  m_env.viewOverride = 0;
  for (const uint16_t view : {uint16_t{1}, uint16_t{8}}) {
    m_settings.debugView = view;
    expect_view_off();
  }
  m_settings.debugView = 0;
  latch(graph.get());
  expect_ran(*graph, build(state, frame), true);
  EXPECT_TRUE(composite::enabled_this_frame(frame));
  EXPECT_EQ(compile(m_settings)->graphHash, hash);
}

TEST_F(CompositeGraph, GbufAndCompositeViewsKeepComposite) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const FrameState state{.graph = graph};
  uint64_t frame = 1;
  for (const uint16_t view : {uint16_t{16}, uint16_t{22}, uint16_t{25}}) {
    m_settings.debugView = view;
    latch(graph.get());
    expect_ran(*graph, build(state, frame), true);
    EXPECT_TRUE(composite::enabled_this_frame(frame)) << view;
    ++frame;
  }
  EXPECT_EQ(composite::testing::counters().skippedView, 0u);
}

TEST_F(CompositeGraph, RealAoProducerMatchesContract) {
  ao::initialize();
  for (const std::optional<GraphExtent> scene : {std::optional<GraphExtent>{}, std::optional{GraphExtent{48, 40}}}) {
    m_sceneSize = scene;
    const auto graph = compile(ao_settings());
    const NodeId node = graph->find_node(composite::Pass);
    ASSERT_NE(node, InvalidNode);
    EXPECT_EQ(graph->nodes[node].status, CullReason::None);
    EXPECT_EQ(graph->nodes[graph->find_node(composite::CopyPass)].status, CullReason::None);
    EXPECT_EQ(composite::check_contract(*graph), std::nullopt);
    const CompiledResource& final = graph->resources[graph->find_resource(resource::AoFinal)];
    const CompiledResource& legacy = graph->resources[graph->find_resource(composite::LegacyTex)];
    EXPECT_EQ(final.tex.format, wgpu::TextureFormat::RGBA8Unorm);
    EXPECT_EQ(final.kind, ResourceKind::Transient);
    EXPECT_EQ(final.extent.width, legacy.extent.width);
    EXPECT_EQ(final.extent.height, legacy.extent.height);
    EXPECT_EQ(legacy.extent.width, scene ? 48u : 64u);
    EXPECT_EQ(legacy.extent.height, scene ? 40u : 64u);
  }
}

TEST_F(CompositeGraph, ContractMismatchTurnsNotReady) {
  StubAo r8{.format = wgpu::TextureFormat::R8Unorm};
  expect_refused(r8, "R8Unorm");
  testing::clear_passes();
  composite::shutdown();
  composite::initialize();
  StubAo r16{.format = wgpu::TextureFormat::R16Float};
  expect_refused(r16, "R16Float");
  testing::clear_passes();
  composite::shutdown();
  composite::initialize();
  StubAo half{.divisor = 2};
  expect_refused(half, "half extent");
}

TEST_F(CompositeGraph, InterveningGxColorWriterDisablesComposite) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  TestPass writer{.name = "t.writer", .declare = [](InternalPassBuilder& b) {
                    b.after(phase::Ao);
                    b.use(resource::GxColor, Access::ColorRMW);
                  }};
  const PassId inside = register_pass(desc_of(writer));
  const auto refused = live_graph(ao_settings());
  const std::optional<std::string> reason = composite::check_contract(*refused);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find("t.writer"), std::string::npos) << *reason;
  EXPECT_FALSE(composite::live());
  EXPECT_EQ(composite::testing::counters().contractFailures, 1u);
  unregister_pass(inside);

  TestPass early{.name = "t.early", .declare = [](InternalPassBuilder& b) {
                   b.before(phase::Prepare);
                   b.use(resource::GxColor, Access::ColorRMW);
                 }};
  const PassId before = register_pass(desc_of(early));
  const auto accepted = live_graph(ao_settings());
  EXPECT_EQ(composite::check_contract(*accepted), std::nullopt);
  EXPECT_TRUE(composite::live());
  unregister_pass(before);

  TestPass late{.name = "t.late", .order = 0xFFFF, .declare = [](InternalPassBuilder& b) {
                  b.after(phase::Composite);
                  b.use(resource::GxColor, Access::ColorRMW);
                }};
  register_pass(desc_of(late));
  const auto after = live_graph(ao_settings());
  EXPECT_EQ(composite::check_contract(*after), std::nullopt);
  EXPECT_LT(program_index(*after, composite::Pass), program_index(*after, "t.late"));
  EXPECT_TRUE(composite::live());
}

TEST_F(CompositeGraph, CompileTimeMissingAoCullsBoth) {
  const auto graph = live_graph(ao_settings());
  EXPECT_NE(graph->nodes[graph->find_node(composite::Pass)].status, CullReason::None);
  EXPECT_NE(graph->nodes[graph->find_node(composite::CopyPass)].status, CullReason::None);
  EXPECT_TRUE(composite::check_contract(*graph).has_value());
  EXPECT_FALSE(composite::live());
  EXPECT_FALSE(composite::ready(m_settings));
  EXPECT_EQ(composite::testing::counters().contractFailures, 0u);
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1), std::nullopt);
  EXPECT_EQ(composite::testing::counters().copies, 0u);
}

TEST_F(CompositeGraph, FrameDropOfAoCullsBoth) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const FrameState state{.graph = graph};
  expect_ran(*graph, build(state, 1), true);
  stub.enabled = false;
  expect_ran(*graph, build(state, 2), false);
  EXPECT_TRUE(composite::live());
  stub.enabled = true;
  expect_ran(*graph, build(state, 3), true);
}

TEST_F(CompositeGraph, PendingPipelineCullsBoth) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  std::atomic_bool release = false;
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!release && !context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const auto graph = compile(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  ASSERT_EQ(shaders::status({graph->nodes[node].pipelineIds[0]}), shaders::PipelineStatus::Pending);
  latch(graph.get());
  EXPECT_TRUE(composite::live());
  EXPECT_FALSE(composite::ready(m_settings));
  const FrameState state{.graph = graph};
  expect_ran(*graph, build(state, 1), false);
  EXPECT_FALSE(composite::enabled_this_frame(1));
  release = true;
  gate.wait();
  wait_ready(*graph);
  expect_ran(*graph, build(state, 2), true);
  EXPECT_TRUE(composite::enabled_this_frame(2));
  EXPECT_TRUE(composite::ready(m_settings));
}

TEST_F(CompositeGraph, DisabledWithoutMainCamera) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const FrameState state{.graph = graph};
  composite::testing::set_camera_override(camera_hints(false));
  const uint32_t skipped = composite::testing::counters().skippedNoScene;
  expect_ran(*graph, build(state, 1), false);
  EXPECT_EQ(composite::testing::counters().skippedNoScene, skipped + 1);
  EXPECT_FALSE(composite::enabled_this_frame(1));
  composite::testing::set_camera_override(m_hints);
  expect_ran(*graph, build(state, 2), true);
}

TEST_F(CompositeGraph, EnabledThisFrameFollowsCull) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  TestPass reader{
      .name = "t.late_ao", .stage = Stage::PrePost, .kind = PassKind::Compute, .declare = [](InternalPassBuilder& b) {
        b.use(resource::AoFinal, Access::Sampled, Missing::BindDummyWhite);
        b.side_effect();
      }};
  register_pass(desc_of(reader));
  const auto graph = live_graph(ao_settings());
  const FrameState state{.graph = graph};
  const NodeId readerNode = graph->find_node("t.late_ao");
  ASSERT_NE(readerNode, InvalidNode);
  ASSERT_EQ(graph->nodes[readerNode].status, CullReason::None);
  const CompiledUse& readerUse = graph->node_uses(readerNode)[0];
  const auto white_in_prepost = [&](uint64_t frameId) {
    StageData post{.stage = Stage::PrePost, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(post);
    EXPECT_TRUE(pass_enabled(*graph, post, "t.late_ao"));
    return post.dummy_bound(readerUse.stageUse);
  };
  uint64_t frame = 1;
  build(state, frame);
  EXPECT_TRUE(composite::enabled_this_frame(frame));
  EXPECT_FALSE(white_in_prepost(frame));
  ++frame;

  m_env.viewOverride = 1;
  latch(graph.get());
  build(state, frame);
  EXPECT_FALSE(composite::enabled_this_frame(frame));
  EXPECT_FALSE(white_in_prepost(frame));
  m_env.viewOverride = 0;
  latch(graph.get());
  ++frame;

  composite::testing::set_camera_override(camera_hints(false));
  build(state, frame);
  EXPECT_FALSE(composite::enabled_this_frame(frame));
  EXPECT_FALSE(white_in_prepost(frame));
  composite::testing::set_camera_override(m_hints);
  ++frame;

  build(state, frame, false);
  EXPECT_FALSE(composite::enabled_this_frame(frame));
  EXPECT_FALSE(white_in_prepost(frame));
  ++frame;

  stub.enabled = false;
  build(state, frame);
  EXPECT_TRUE(white_in_prepost(frame));
  stub.enabled = true;
  ++frame;

  build(state, frame);
  EXPECT_TRUE(composite::enabled_this_frame(frame));
  EXPECT_FALSE(composite::enabled_this_frame(frame + 1));
}

TEST_F(CompositeGraph, EnabledThisFrameFalseUnderContractMismatch) {
  StubAo stub{.format = wgpu::TextureFormat::R8Unorm};
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const FrameState state{.graph = graph};
  build(state, 1);
  EXPECT_FALSE(composite::enabled_this_frame(1));
  EXPECT_EQ(stub.calls, 0);
}

TEST_F(CompositeGraph, EncodesValidationClean) { encode_sizes(); }

TEST_F(CompositeGraph, PbrEncodesValidationClean) { encode_sizes(true); }

TEST_F(CompositeGraph, EncodesValidationCleanWithProductionToggles) {
  TearDown();
  m_started = false;
  if (!open({.backend = wgpu::BackendType::Null, .efbSize = {64, 64, 1}, .productionToggles = true})) {
    GTEST_SKIP() << "no null device: " << m_error;
  }
  encode_sizes();
}

TEST_F(CompositeGraph, ParamsLatchedEveryFrameAndAfterReinit) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  const PassId first = graph->nodes[graph->find_node(composite::Pass)].id;
  const auto flags_of = [](PassId pass) {
    composite::Params params;
    const std::span<const std::byte> bytes = latched_pass_params(pass);
    EXPECT_EQ(bytes.size(), sizeof(params));
    std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(params)));
    return params.flags;
  };
  EXPECT_EQ(flags_of(first),
            composite::params_flags(composite::identity_forced(), std::getenv("AURORA_RENDER_COMPOSITE_NOFOG")));
  composite::testing::set_flags(composite::FlagIdentity);
  latch(graph.get());
  EXPECT_EQ(flags_of(first), uint32_t{composite::FlagIdentity});
  composite::testing::set_flags(composite::FlagNoFog);
  latch(graph.get());
  EXPECT_EQ(flags_of(first), uint32_t{composite::FlagNoFog});
  clear_pass_params(first);
  latch_pass_params();
  EXPECT_TRUE(latched_pass_params(first).empty());
  latch(graph.get());
  EXPECT_EQ(flags_of(first), uint32_t{composite::FlagNoFog});

  composite::shutdown();
  EXPECT_FALSE(composite::live());
  composite::initialize();
  const auto again = live_graph(ao_settings());
  const PassId second = again->nodes[again->find_node(composite::Pass)].id;
  EXPECT_NE(second, first);
  EXPECT_EQ(flags_of(second), uint32_t{composite::FlagNoFog});
  EXPECT_TRUE(composite::live());
}

TEST_F(CompositeGraph, ReadyFollowsPipelines) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const Settings high = ao_settings();
  EXPECT_TRUE(composite::ready(Settings{}));
  composite::on_latch({.settings = &high}, nullptr);
  EXPECT_FALSE(composite::ready(high));
  std::atomic_bool release = false;
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!release && !context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const auto graph = compile(high);
  latch(graph.get());
  EXPECT_FALSE(composite::ready(high));
  EXPECT_TRUE(composite::ready(Settings{}));
  release = true;
  gate.wait();
  wait_ready(*graph);
  EXPECT_TRUE(composite::ready(high));
  remove_hints();
  EXPECT_FALSE(composite::ready(high));
  EXPECT_TRUE(composite::ready(Settings{}));
  install_hints();
  EXPECT_TRUE(composite::ready(high));
}

TEST_F(CompositeGraph, AoGateFollowsComposite) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = live_graph(ao_settings());
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  FrameState state{.settings = m_settings, .graph = graph};
  EXPECT_TRUE(composite::ao_gate(state));
  state.settings.ao = Quality::Off;
  EXPECT_FALSE(composite::ao_gate(state));
  state.settings.ao = Quality::High;
  m_settings.debugView = 1;
  latch(graph.get());
  EXPECT_FALSE(composite::ao_gate(state));
  m_settings.debugView = 0;
  m_settings.aoStrength = 0.f;
  latch(graph.get());
  if (std::getenv("AURORA_RENDER_AO_STRENGTH") == nullptr) {
    EXPECT_FALSE(composite::ao_gate(state));
  }
  m_settings.aoStrength = 0.5f;
  latch(graph.get());
  EXPECT_TRUE(composite::ao_gate(state));
  composite::on_latch({.settings = &m_settings}, nullptr);
  EXPECT_FALSE(composite::ao_gate(state));
}

TEST_F(CompositeGraph, RadianceIsSecondAttachmentAfterColor) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  composite::testing::set_radiance_forced(true);
  const auto graph = live_graph(ao_settings());
  ASSERT_EQ(composite::check_contract(*graph), std::nullopt);
  ASSERT_TRUE(composite::live());
  ASSERT_TRUE(composite::radiance_bound(*graph));
  const NodeId node = graph->find_node(composite::Pass);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 11u);
  EXPECT_EQ(graph->resources[uses[9].resource].name, resource::GxColor);
  EXPECT_EQ(uses[9].access, Access::ColorRMW);
  EXPECT_EQ(graph->resources[uses[10].resource].name, resource::RadianceDefogged);
  EXPECT_EQ(uses[10].access, Access::ColorWrite);
  const CompiledResource& radiance = graph->resources[uses[10].resource];
  const CompiledResource& legacy = graph->resources[graph->find_resource(composite::LegacyTex)];
  EXPECT_EQ(radiance.kind, ResourceKind::History);
  EXPECT_EQ(radiance.creator, node);
  EXPECT_EQ(radiance.tex.format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_EQ(radiance.tex.sizeClass, SizeClass::Efb);
  EXPECT_EQ(radiance.tex.divisor, 1u);
  EXPECT_EQ(radiance.extent.width, legacy.extent.width);
  EXPECT_EQ(radiance.extent.height, legacy.extent.height);
  EXPECT_NE(radiance.textureUsage & wgpu::TextureUsage::TextureBinding, wgpu::TextureUsage::None);
  EXPECT_NE(radiance.textureUsage & wgpu::TextureUsage::RenderAttachment, wgpu::TextureUsage::None);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_NE(uses[i].access, Access::ColorWrite) << i;
  }
  const std::array<std::string_view, 9> inputs{
      composite::LegacyTex,     resource::Fog,      resource::GxNormal,           resource::AoFinal,
      resource::Surface,        resource::SceneHdr, lighting::exposure::Resource, resource::SsilIrradiance,
      fakeshadow::lsm::Resource};
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, inputs[i]) << i;
  }
  const FrameState state{.graph = graph};
  testing::record_bind_details(true);
  EXPECT_EQ(run(state, 1), std::nullopt);
  const std::vector<testing::AttachmentOps> ops = testing::last_attachment_ops();
  testing::record_bind_details(false);
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].load, wgpu::LoadOp::Load);
  EXPECT_EQ(ops[1].load, wgpu::LoadOp::Clear);
  EXPECT_EQ(ops[1].width, legacy.extent.width);
  EXPECT_EQ(composite::testing::counters().radianceDraws, 1u);
  EXPECT_EQ(composite::testing::counters().radianceClears, 0u);
}

TEST_F(CompositeGraph, SsilOffKeepsGraphAndPipeline) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  composite::testing::set_radiance_forced(false);
  const auto off = live_graph(ao_settings());
  const NodeId node = off->find_node(composite::Pass);
  ASSERT_EQ(off->nodes[node].status, CullReason::None);
  EXPECT_FALSE(composite::radiance_bound(*off));
  EXPECT_EQ(off->find_resource(resource::RadianceDefogged), InvalidResource);
  EXPECT_EQ(off->node_uses(node).size(), 10u);
  const composite::Params offParams = latched_params(*off);
  composite::testing::set_radiance_forced(true);
  const auto on = live_graph(ao_settings());
  ASSERT_TRUE(composite::radiance_bound(*on));
  EXPECT_NE(on->graphHash, off->graphHash);
  EXPECT_NE(on->nodes[on->find_node(composite::Pass)].pipelineIds[0], off->nodes[node].pipelineIds[0]);
  EXPECT_TRUE(same_params(latched_params(*on), offParams));
  composite::testing::set_radiance_forced(false);
  const auto back = live_graph(ao_settings());
  EXPECT_EQ(back->graphHash, off->graphHash);
  EXPECT_EQ(back->nodes[back->find_node(composite::Pass)].pipelineIds, off->nodes[node].pipelineIds);
  EXPECT_TRUE(same_params(latched_params(*back), offParams));
  Settings ssil = ao_settings();
  ssil.ssil = true;
  EXPECT_TRUE(composite::radiance_bound(*compile(ssil)));
  ssil.ao = Quality::Off;
  EXPECT_EQ(compile(ssil)->find_resource(resource::RadianceDefogged), InvalidResource);
}

TEST_F(CompositeGraph, RadianceHistoryIsSinkWithoutReader) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  composite::testing::set_radiance_forced(true);
  const auto graph = live_graph(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  const ResourceId id = graph->find_resource(resource::RadianceDefogged);
  ASSERT_NE(id, InvalidResource);
  EXPECT_EQ(graph->nodes[node].status, CullReason::None);
  for (const CompiledUse& use : graph->uses) {
    if (use.resource == id) {
      EXPECT_EQ(use.node, node) << graph->nodes[use.node].name;
    }
  }
  EXPECT_NE(std::ranges::find(graph->histories, id), graph->histories.end());
  const FrameState state{.graph = graph};
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    StageData opaque = build(state, frame);
    EXPECT_TRUE(pass_enabled(*graph, opaque, composite::Pass));
    EXPECT_EQ(opaque.prev_valid(graph->resources[id].historySlot), frame > 1) << frame;
    EXPECT_EQ(encode(opaque), std::nullopt);
  }
  EXPECT_EQ(composite::testing::counters().draws, 3u);
  EXPECT_EQ(composite::testing::counters().radianceDraws, 3u);
  EXPECT_EQ(composite::testing::counters().radianceClears, 0u);
}

TEST_F(CompositeGraph, RadianceContractMismatchRefuses) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  composite::testing::set_radiance_forced(true);
  for (int edit = 0; edit < 3; ++edit) {
    SCOPED_TRACE(edit);
    const auto graph = compile(ao_settings());
    ASSERT_EQ(composite::check_contract(*graph), std::nullopt);
    CompiledResource& radiance = graph->resources[graph->find_resource(resource::RadianceDefogged)];
    if (edit == 0) {
      radiance.tex.format = wgpu::TextureFormat::RGBA8Unorm;
    } else if (edit == 1) {
      radiance.extent.width /= 2;
    } else {
      radiance.kind = ResourceKind::Transient;
    }
    const std::optional<std::string> reason = composite::check_contract(*graph);
    ASSERT_TRUE(reason.has_value());
    EXPECT_NE(reason->find(resource::RadianceDefogged), std::string::npos) << *reason;
    composite::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
    latch(graph.get());
    EXPECT_FALSE(composite::live());
  }
  EXPECT_EQ(composite::testing::counters().contractFailures, 3u);
}

TEST_F(CompositeGraph, RadianceClearedOnEncodeBail) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  composite::testing::set_radiance_forced(true);
  // A scene extent the EFB does not match passes the contract but fails the encode-time legacy check
  m_sceneSize = GraphExtent{48, 40};
  const auto graph = live_graph(ao_settings());
  ASSERT_TRUE(composite::live());
  ASSERT_TRUE(composite::radiance_bound(*graph));
  const FrameState state{.graph = graph};
  testing::record_bind_details(true);
  EXPECT_EQ(run(state, 1), std::nullopt);
  const std::vector<testing::AttachmentOps> ops = testing::last_attachment_ops();
  testing::record_bind_details(false);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_FALSE(ops[0].depth);
  EXPECT_EQ(ops[0].load, wgpu::LoadOp::Clear);
  EXPECT_EQ(ops[0].store, wgpu::StoreOp::Store);
  EXPECT_EQ(ops[0].clearColor.r, 0.0);
  EXPECT_EQ(ops[0].clearColor.a, 0.0);
  EXPECT_EQ(ops[0].width, 48u);
  EXPECT_EQ(ops[0].height, 40u);
  const composite::testing::Counters c = composite::testing::counters();
  EXPECT_EQ(c.draws, 0u);
  EXPECT_EQ(c.radianceDraws, 0u);
  EXPECT_EQ(c.radianceClears, 1u);
  EXPECT_GE(c.skippedMismatch, 1u);
}

TEST_F(CompositeGraph, RadianceEncodeReadsTheEncodedGraph) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  uint64_t frame = 0;
  const auto graphs = [&] {
    composite::testing::set_radiance_forced(true);
    const auto with = compile(ao_settings());
    composite::testing::set_radiance_forced(false);
    const auto without = compile(ao_settings());
    wait_ready(*with);
    wait_ready(*without);
    EXPECT_TRUE(composite::radiance_bound(*with));
    EXPECT_FALSE(composite::radiance_bound(*without));
    return std::pair{with, without};
  };
  // A is enabled against its own latch, then B is latched before A encodes, as a worker can see it
  const auto encode_after_latch = [&](const std::shared_ptr<CompiledGraph>& a,
                                      const std::shared_ptr<CompiledGraph>& b) {
    composite::on_latch({.settings = &m_settings, .env = &m_env}, nullptr);
    latch(a.get());
    const FrameState state{.graph = a};
    StageData opaque = build(state, ++frame);
    EXPECT_TRUE(pass_enabled(*a, opaque, composite::Pass));
    latch(b.get());
    EXPECT_EQ(encode(opaque), std::nullopt);
  };
  const auto [with, without] = graphs();
  encode_after_latch(with, without);
  EXPECT_EQ(composite::testing::counters().draws, 1u);
  EXPECT_EQ(composite::testing::counters().radianceDraws, 1u);
  encode_after_latch(without, with);
  EXPECT_EQ(composite::testing::counters().draws, 2u);
  EXPECT_EQ(composite::testing::counters().radianceDraws, 1u);
  m_sceneSize = GraphExtent{48, 40};
  const auto [withBad, withoutBad] = graphs();
  encode_after_latch(withBad, withoutBad);
  EXPECT_EQ(composite::testing::counters().radianceClears, 1u);
  encode_after_latch(withoutBad, withBad);
  EXPECT_EQ(composite::testing::counters().radianceClears, 1u);
  EXPECT_EQ(composite::testing::counters().draws, 2u);
}

TEST_F(CompositeGraph, RadiancePrevFollowsCompositeRuns) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  RadianceProbe probe{.prev = true};
  register_pass(radiance_probe_desc(probe));
  composite::testing::set_radiance_forced(true);
  auto graph = live_graph(ao_settings());
  uint64_t frame = 0;
  uint32_t epoch = 0;
  const auto prev_valid = [&](const std::shared_ptr<CompiledGraph>& g) {
    emit_camera(epoch);
    const FrameState state{.graph = g};
    const size_t before = probe.prevValid.size();
    EXPECT_EQ(run(state, ++frame), std::nullopt);
    if (probe.prevValid.size() != before + 1) {
      ADD_FAILURE() << "the probe did not run on frame " << frame;
      return false;
    }
    return static_cast<bool>(probe.prevValid.back());
  };
  EXPECT_FALSE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
  m_settings.debugView = 1;
  latch(graph.get());
  EXPECT_TRUE(prev_valid(graph));
  m_settings.debugView = 0;
  latch(graph.get());
  EXPECT_FALSE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
  ++epoch;
  EXPECT_FALSE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
  m_sceneSize = GraphExtent{48, 40};
  const auto resized = live_graph(ao_settings());
  EXPECT_NE(resized->graphHash, graph->graphHash);
  EXPECT_FALSE(prev_valid(resized));
  m_sceneSize.reset();
  graph = live_graph(ao_settings());
  EXPECT_FALSE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
  composite::testing::set_radiance_forced(false);
  const auto off = live_graph(ao_settings());
  EXPECT_FALSE(composite::radiance_bound(*off));
  EXPECT_FALSE(prev_valid(off));
  EXPECT_FALSE(prev_valid(off));
  composite::testing::set_radiance_forced(true);
  graph = live_graph(ao_settings());
  EXPECT_FALSE(prev_valid(graph));
  EXPECT_TRUE(prev_valid(graph));
}

struct Rgba {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 0;
};

struct Pixel {
  Rgba color{};
  uint8_t code = 1;
  std::array<uint8_t, 3> fc{};
  uint8_t q7 = 0;
  uint8_t ao = 255;
  std::array<uint8_t, 3> albedo{};
  uint8_t cls = 0;
};

// The stub writes linear albedo as H, so P re-encodes it; class 0 writes alpha 0 like light.deferred
std::array<uint16_t, 4> hdr_words(const Pixel& p) {
  std::array<uint16_t, 4> out{};
  for (size_t c = 0; c < 3; ++c) {
    const float s = p.albedo[c] / 255.f;
    const float linear = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    out[c] = lighting::openpbr::to_half(p.cls != 0 ? linear : 0.f);
  }
  out[3] = lighting::openpbr::to_half(p.cls != 0 ? 1.f : 0.f);
  return out;
}

std::array<float, 4> hdr_of(const Pixel& p) {
  const std::array<uint16_t, 4> words = hdr_words(p);
  return {lighting::openpbr::from_half(words[0]), lighting::openpbr::from_half(words[1]),
          lighting::openpbr::from_half(words[2]), lighting::openpbr::from_half(words[3])};
}

struct PixelRun {
  std::optional<uint32_t> flags;
  Settings settings = ao_settings();
  bool aoProducer = true;
  bool hdrEnabled = true;
  std::optional<composite::Params> params;
  uint32_t draws = 1;
  std::optional<float> exposure;
  bool exposureEnabled = true;
  std::function<void(std::array<uint16_t, 4>&, size_t)> hdrEdit;
  bool radiance = false;
  uint32_t radianceClears = 0;
  std::function<std::array<float, 4>(size_t)> ssil;
  // RGBA bytes of lsm.mask per pixel; set, a stub replays it as the mask pass would
  std::function<std::array<uint8_t, 4>(size_t)> mask;
};

Pixel control() { return {.color = {200, 180, 160, 255}, .code = 1, .fc = {}, .q7 = 0, .ao = 64}; }

class CompositeGpu : public CompositeDeviceTest {
protected:
  void SetUp() override {
    const auto backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND"));
    if (!backend) {
      GTEST_SKIP() << "set AURORA_TEST_BACKEND to run the composite on a real backend";
    }
    if (!open({.backend = *backend, .efbSize = {64, 48, 1}})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    ASSERT_EQ(webgpu::g_frameBuffer.texture.GetFormat(), wgpu::TextureFormat::BGRA8Unorm);
    make_depth();
  }

  wgpu::Texture upload(wgpu::TextureFormat format, uint32_t width, uint32_t height, const std::vector<uint32_t>& words,
                       const char* label, uint32_t texelBytes = 4) const {
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
    m_device->queue.WriteTexture(&destination, words.data(), words.size() * 4, &layout, &size);
    return texture;
  }

  static void publish(webgpu::TextureWithSampler& target, const wgpu::Texture& texture, wgpu::TextureFormat format) {
    target.texture = texture;
    target.view = texture.CreateView();
    target.size = {texture.GetWidth(), texture.GetHeight(), 1};
    target.format = format;
  }

  std::vector<Rgba> read_color(uint32_t width, uint32_t height) const {
    const uint32_t row = (width * 4 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{width, height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * height);
    std::vector<Rgba> out(size_t{width} * height);
    if (!words) {
      ADD_FAILURE() << "colour readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint8_t* p = bytes + size_t{y} * row + size_t{x} * 4;
        out[size_t{y} * width + x] = {.r = p[2], .g = p[1], .b = p[0], .a = p[3]};
      }
    }
    return out;
  }

  std::vector<Rgba> composite_pixels(uint32_t width, uint32_t height, const std::vector<Pixel>& pixels,
                                     std::optional<uint32_t> flags = std::nullopt) {
    return run_pixels(width, height, pixels, {.flags = flags});
  }

  std::vector<Rgba> run_pixels(uint32_t width, uint32_t height, const std::vector<Pixel>& pixels,
                               const PixelRun& options) {
    if (webgpu::g_frameBuffer.size.width != width || webgpu::g_frameBuffer.size.height != height) {
      use_globals({width, height, 1});
      make_depth();
    }
    if (options.settings.pbr) {
      EXPECT_TRUE(use_pbr_tier());
    } else {
      EXPECT_TRUE(webgpu::set_scene_tier(webgpu::SceneTier::Base));
    }
    composite::testing::set_flags(options.flags);
    composite::testing::force_params(options.params);
    composite::testing::set_radiance_forced(options.radiance);
    std::vector<uint32_t> color(pixels.size());
    std::vector<uint32_t> normal(pixels.size());
    std::vector<uint32_t> fog(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
      const Pixel& p = pixels[i];
      color[i] = uint32_t{p.color.b} | uint32_t{p.color.g} << 8 | uint32_t{p.color.r} << 16 | uint32_t{p.color.a} << 24;
      normal[i] = 512u | 512u << 10 | 1023u << 20 | uint32_t{p.code} << 30;
      fog[i] = uint32_t{p.fc[0]} | uint32_t{p.fc[1]} << 8 | uint32_t{p.fc[2]} << 16 | uint32_t{p.q7} << 25;
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device->queue.WriteTexture(&destination, color.data(), color.size() * 4, &layout, &size);
    publish(webgpu::g_normalBuffer, upload(wgpu::TextureFormat::RGB10A2Unorm, width, height, normal, "test normals"),
            wgpu::TextureFormat::RGB10A2Unorm);
    publish(webgpu::g_fogRecordBuffer, upload(wgpu::TextureFormat::RGBA8Uint, width, height, fog, "test fog"),
            wgpu::TextureFormat::RGBA8Uint);
    if (options.settings.pbr) {
      std::vector<uint32_t> surface(pixels.size() * 2, 0);
      for (size_t i = 0; i < pixels.size(); ++i) {
        const Pixel& p = pixels[i];
        surface[i * 2] = uint32_t{p.albedo[0]} | uint32_t{p.albedo[1]} << 8 | uint32_t{p.albedo[2]} << 16 | 255u << 24;
        surface[i * 2 + 1] = 217u | uint32_t{p.cls} << gx::surface::SlotShift;
      }
      publish(webgpu::g_surfaceBuffer, upload(webgpu::SurfaceBufferFormat, width, height, surface, "test surface", 8),
              webgpu::SurfaceBufferFormat);
    }
    StubAo stub;
    stub.rowBytes = (width * 4 + 255) & ~255u;
    std::vector<uint8_t> staged(size_t{stub.rowBytes} * height, 0);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        staged[size_t{y} * stub.rowBytes + size_t{x} * 4] = pixels[size_t{y} * width + x].ao;
      }
    }
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = staged.size()};
    stub.staging = m_device->device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(stub.staging, 0, staged.data(), staged.size());
    const PassId stubId = options.aoProducer ? register_pass(stub_ao_desc(stub)) : InvalidPassId;
    StubHdr hdr;
    PassId hdrId = InvalidPassId;
    if (options.settings.pbr) {
      hdr.rowBytes = (width * 8 + 255) & ~255u;
      std::vector<uint16_t> texels(size_t{hdr.rowBytes / 2} * height, 0);
      for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          std::array<uint16_t, 4> words = hdr_words(pixels[size_t{y} * width + x]);
          if (options.hdrEdit) {
            options.hdrEdit(words, size_t{y} * width + x);
          }
          std::copy(words.begin(), words.end(), texels.begin() + size_t{y} * (hdr.rowBytes / 2) + size_t{x} * 4);
        }
      }
      const wgpu::BufferDescriptor hdrDescriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                                 .size = texels.size() * 2};
      hdr.staging = m_device->device.CreateBuffer(&hdrDescriptor);
      m_device->queue.WriteBuffer(hdr.staging, 0, texels.data(), texels.size() * 2);
      hdr.enabled = options.hdrEnabled;
      hdrId = register_pass(stub_hdr_desc(hdr));
      lighting::deferred::testing::force_enabled(true);
    }
    StubExposure meter;
    PassId meterId = InvalidPassId;
    if (options.exposure) {
      meter.words = {*options.exposure, 1.f / *options.exposure, 1.f, 1.f};
      meter.enabled = options.exposureEnabled;
      meterId = register_pass(stub_exposure_desc(meter));
    }
    StubSsil ssil;
    PassId ssilId = InvalidPassId;
    if (options.ssil) {
      ssil.rowBytes = (width * 8 + 255) & ~255u;
      std::vector<uint16_t> texels(size_t{ssil.rowBytes / 2} * height, 0);
      for (size_t i = 0; i < pixels.size(); ++i) {
        const std::array<float, 4> value = options.ssil(i);
        for (size_t c = 0; c < 4; ++c) {
          texels[(i / width) * (ssil.rowBytes / 2) + (i % width) * 4 + c] = lighting::openpbr::to_half(value[c]);
        }
      }
      const wgpu::BufferDescriptor ssilDescriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                                  .size = texels.size() * 2};
      ssil.staging = m_device->device.CreateBuffer(&ssilDescriptor);
      m_device->queue.WriteBuffer(ssil.staging, 0, texels.data(), texels.size() * 2);
      ssilId = register_pass(stub_ssil_desc(ssil));
    }
    StubLsm lsm;
    PassId lsmId = InvalidPassId;
    if (options.mask) {
      lsm.rowBytes = (width * 4 + 255) & ~255u;
      std::vector<uint8_t> texels(size_t{lsm.rowBytes} * height, 0);
      for (size_t i = 0; i < pixels.size(); ++i) {
        const std::array<uint8_t, 4> rgba = options.mask(i);
        uint8_t* texel = texels.data() + (i / width) * lsm.rowBytes + (i % width) * 4;
        texel[0] = rgba[2];
        texel[1] = rgba[1];
        texel[2] = rgba[0];
        texel[3] = rgba[3];
      }
      const wgpu::BufferDescriptor lsmDescriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                                 .size = texels.size()};
      lsm.staging = m_device->device.CreateBuffer(&lsmDescriptor);
      m_device->queue.WriteBuffer(lsm.staging, 0, texels.data(), texels.size());
      lsmId = register_pass(stub_lsm_desc(lsm));
    }
    RadianceProbe probe;
    const PassId probeId = options.radiance ? register_pass(radiance_probe_desc(probe)) : InvalidPassId;
    const composite::testing::Counters before = composite::testing::counters();
    const uint32_t draws = before.draws;
    const auto graph = live_graph(options.settings);
    EXPECT_TRUE(composite::live());
    EXPECT_EQ(composite::radiance_bound(*graph), options.radiance);
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, ++m_frame), std::nullopt);
    if (probeId != InvalidPassId) {
      unregister_pass(probeId);
      EXPECT_EQ(probe.calls, 1);
      m_radiance = read_radiance(probe);
      m_radianceExtent = probe.extent;
    }
    EXPECT_EQ(composite::testing::counters().radianceDraws,
              before.radianceDraws + (options.radiance ? options.draws : 0));
    EXPECT_EQ(composite::testing::counters().radianceClears, before.radianceClears + options.radianceClears);
    if (stubId != InvalidPassId) {
      unregister_pass(stubId);
    }
    if (hdrId != InvalidPassId) {
      unregister_pass(hdrId);
    }
    if (lsmId != InvalidPassId) {
      EXPECT_EQ(lsm.calls, 1);
      unregister_pass(lsmId);
    }
    if (ssilId != InvalidPassId) {
      EXPECT_TRUE(composite::ssil_bound(*graph));
      EXPECT_EQ(ssil.calls, 1);
      unregister_pass(ssilId);
    }
    if (meterId != InvalidPassId) {
      EXPECT_EQ(meter.calls, options.exposureEnabled ? 1 : 0);
      unregister_pass(meterId);
    }
    lighting::deferred::testing::force_enabled(std::nullopt);
    EXPECT_EQ(composite::testing::counters().draws, draws + options.draws);
    EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
    return read_color(width, height);
  }

  std::vector<std::array<float, 4>> read_radiance(const RadianceProbe& probe) const {
    std::vector<std::array<float, 4>> out(size_t{probe.extent.width} * probe.extent.height);
    if (!probe.readback) {
      ADD_FAILURE() << "the radiance probe copied nothing";
      return out;
    }
    const auto words = test::read_buffer(*m_device, probe.readback, 0, uint64_t{probe.rowBytes} * probe.extent.height);
    if (!words) {
      ADD_FAILURE() << "radiance readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < probe.extent.height; ++y) {
      for (uint32_t x = 0; x < probe.extent.width; ++x) {
        std::array<uint16_t, 4> texel{};
        std::memcpy(texel.data(), bytes + size_t{y} * probe.rowBytes + size_t{x} * 8, sizeof(texel));
        out[size_t{y} * probe.extent.width + x] = {
            lighting::openpbr::from_half(texel[0]), lighting::openpbr::from_half(texel[1]),
            lighting::openpbr::from_half(texel[2]), lighting::openpbr::from_half(texel[3])};
      }
    }
    return out;
  }

  static std::array<int, 3> mirror(const Pixel& p) {
    const std::array<float, 3> l{p.color.r / 255.f, p.color.g / 255.f, p.color.b / 255.f};
    const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
    const std::array<float, 3> out = composite::testing::legacy_ao(l, fc, p.q7, p.ao / 255.f, p.code / 3.f);
    return {to_byte(out[0]), to_byte(out[1]), to_byte(out[2])};
  }

  static std::array<int, 3> pbr_mirror(const Pixel& p, float legacyBlend, float e = 1.f, bool agx = false,
                                       std::optional<std::array<float, 4>> hdr = std::nullopt,
                                       std::array<float, 3> m = {1.f, 1.f, 1.f}) {
    const std::array<float, 3> l{p.color.r / 255.f, p.color.g / 255.f, p.color.b / 255.f};
    const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
    const std::array<float, 3> out = composite::testing::legacy_pbr_hdr(
        l, fc, p.q7, p.ao / 255.f, p.code / 3.f, hdr.value_or(hdr_of(p)), legacyBlend, e, agx, m);
    return {to_byte(out[0]), to_byte(out[1]), to_byte(out[2])};
  }

  static bool guard_tie(const Pixel& p) {
    for (size_t c = 0; c < 3; ++c) {
      const uint8_t l = c == 0 ? p.color.r : c == 1 ? p.color.g : p.color.b;
      if (std::abs(static_cast<float>(l) - p.q7 / 127.f * p.fc[c] + 2.f) < 1e-3f) {
        return true;
      }
    }
    return false;
  }

  static bool same(const Rgba& a, const Rgba& b) { return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a; }

  static bool same_rgb(const Rgba& a, const std::array<int, 3>& b, int tolerance) {
    return std::abs(a.r - b[0]) <= tolerance && std::abs(a.g - b[1]) <= tolerance && std::abs(a.b - b[2]) <= tolerance;
  }

  uint64_t m_frame = 0;
  std::vector<std::array<float, 4>> m_radiance;
  wgpu::Extent3D m_radianceExtent{};
};

std::vector<Pixel> grid(uint32_t width, uint32_t height, const std::function<Pixel(uint32_t, uint32_t)>& fn) {
  std::vector<Pixel> out(size_t{width} * height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      out[size_t{y} * width + x] = fn(x, y);
    }
  }
  return out;
}

TEST_F(CompositeGpu, IdentityKeepsInputExactly) {
  std::mt19937 rng{7};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) {
    return Pixel{.color = {static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)),
                           static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng))},
                 .code = static_cast<uint8_t>(byte(rng) % 4),
                 .fc = {static_cast<uint8_t>(byte(rng)), 0, 0},
                 .q7 = static_cast<uint8_t>(byte(rng) % 128),
                 .ao = static_cast<uint8_t>(byte(rng))};
  });
  const auto out = composite_pixels(64, 48, pixels, composite::FlagIdentity);
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
  }
}

TEST_F(CompositeGpu, CodeZeroUntouched) {
  const auto pixels = grid(64, 48, [](uint32_t x, uint32_t) {
    if (x < 8) {
      return control();
    }
    return Pixel{.color = {200, 150, 100, 255}, .code = 0, .fc = {}, .q7 = 0, .ao = 64};
  });
  const auto out = composite_pixels(64, 48, pixels);
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].code == 0) {
      ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
    } else {
      ASSERT_FALSE(same(out[i], pixels[i].color)) << i;
    }
  }
}

TEST_F(CompositeGpu, LeavesEfbAlphaUntouched) {
  const auto pixels = grid(64, 48, [](uint32_t x, uint32_t y) {
    Pixel p = control();
    p.color.a = static_cast<uint8_t>((y * 64 + x) & 0xFF);
    return p;
  });
  const auto out = composite_pixels(64, 48, pixels);
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i].a, pixels[i].color.a) << i;
    changed += out[i].r != pixels[i].color.r ? 1 : 0;
  }
  EXPECT_EQ(changed, out.size());
}

TEST_F(CompositeGpu, NeverBrightens) {
  std::mt19937 rng{11};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) {
    const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
    return Pixel{.color = {b(), b(), b(), 255},
                 .code = static_cast<uint8_t>(1 + byte(rng) % 3),
                 .fc = {b(), b(), b()},
                 .q7 = static_cast<uint8_t>(byte(rng) % 128),
                 .ao = b()};
  });
  const auto out = composite_pixels(64, 48, pixels);
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_LE(out[i].r, pixels[i].color.r) << i;
    ASSERT_LE(out[i].g, pixels[i].color.g) << i;
    ASSERT_LE(out[i].b, pixels[i].color.b) << i;
    changed += same(out[i], pixels[i].color) ? 0 : 1;
  }
  EXPECT_GT(changed, out.size() / 4);
}

TEST_F(CompositeGpu, BadGuardKeepsL) {
  const auto pixels = grid(64, 48, [](uint32_t x, uint32_t) {
    if (x < 8) {
      return control();
    }
    return Pixel{.color = {60, 200, 200, 255}, .code = 1, .fc = {120, 0, 0}, .q7 = 127, .ao = 32};
  });
  const auto out = composite_pixels(64, 48, pixels);
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].q7 == 127) {
      ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
    } else {
      ASSERT_FALSE(same(out[i], pixels[i].color)) << i;
    }
  }
}

TEST_F(CompositeGpu, AoBoundHolds) {
  constexpr std::array<uint8_t, 6> Q7s{0, 1, 64, 115, 126, 127};
  constexpr std::array<uint8_t, 6> Aos{0, 1, 64, 128, 254, 255};
  std::mt19937 rng{13};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto pixels = grid(128, 128, [&](uint32_t x, uint32_t y) {
    const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
    const uint32_t i = y * 128 + x;
    return Pixel{.color = {b(), b(), b(), 255},
                 .code = static_cast<uint8_t>(i % 4),
                 .fc = {b(), b(), b()},
                 .q7 = Q7s[(i / 4) % Q7s.size()],
                 .ao = Aos[(i / 24) % Aos.size()]};
  });
  const auto out = composite_pixels(128, 128, pixels);
  size_t changed = 0;
  size_t bad = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = pixels[i];
    const std::array<int, 3> l{p.color.r, p.color.g, p.color.b};
    const std::array<int, 3> got{out[i].r, out[i].g, out[i].b};
    const std::array<int, 3> want = mirror(p);
    bool guard = false;
    bool tie = false;
    for (size_t c = 0; c < 3; ++c) {
      const float scaled = static_cast<float>(l[c]) - p.q7 / 127.f * p.fc[c];
      guard |= scaled < -2.f;
      tie |= std::abs(scaled + 2.f) < 1e-3f;
    }
    for (size_t c = 0; c < 3; ++c) {
      const int d = l[c] - got[c];
      const float s = l[c] / 255.f - p.q7 / 127.f * p.fc[c] / 255.f;
      const int bound = static_cast<int>(std::ceil(255.f * std::max(s, 0.f) * (1.f - p.ao / 255.f))) + 1;
      ASSERT_GE(d, 0) << i;
      ASSERT_LE(d, bound) << i;
      if (p.code == 0 || p.ao == 255) {
        ASSERT_EQ(d, 0) << i;
      }
      if (tie) {
        continue;
      }
      ASSERT_LE(std::abs(got[c] - want[c]), 1) << i << " channel " << c;
      if (guard) {
        ASSERT_EQ(d, 0) << i;
      }
    }
    changed += got != l ? 1 : 0;
    bad += guard ? 1 : 0;
  }
  EXPECT_GT(changed, out.size() / 8);
  EXPECT_GT(bad, 0u);
}

TEST_F(CompositeGpu, FullFogWithinOneLsb) {
  std::mt19937 rng{17};
  std::uniform_int_distribution<int> byte(1, 254);
  std::uniform_int_distribution<int> step(-1, 1);
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) {
    if (x < 8) {
      return control();
    }
    const auto fc = static_cast<uint8_t>(byte(rng));
    const auto l = static_cast<uint8_t>(std::clamp(fc + step(rng), 0, 255));
    return Pixel{.color = {l, fc, l, 255}, .code = 1, .fc = {fc, fc, fc}, .q7 = 127, .ao = 13};
  });
  const auto out = composite_pixels(64, 48, pixels);
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = pixels[i];
    if (p.q7 != 127) {
      ASSERT_FALSE(same(out[i], p.color)) << i;
      continue;
    }
    ASSERT_LE(std::abs(out[i].r - p.color.r), 1) << i;
    ASSERT_EQ(out[i].g, p.color.g) << i;
    ASSERT_LE(std::abs(out[i].b - p.color.b), 1) << i;
  }
}

TEST_F(CompositeGpu, NoFogModeIsPlainMultiply) {
  std::mt19937 rng{19};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) {
    const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
    return Pixel{.color = {b(), b(), b(), 255},
                 .code = 1,
                 .fc = {b(), b(), b()},
                 .q7 = 127,
                 .ao = static_cast<uint8_t>(byte(rng) % 255)};
  });
  const auto out = composite_pixels(64, 48, pixels, composite::FlagNoFog);
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = pixels[i];
    const float a = p.ao / 255.f;
    ASSERT_LE(std::abs(out[i].r - to_byte(p.color.r / 255.f * a)), 1) << i;
    ASSERT_LE(std::abs(out[i].g - to_byte(p.color.g / 255.f * a)), 1) << i;
    ASSERT_LE(std::abs(out[i].b - to_byte(p.color.b / 255.f * a)), 1) << i;
  }
}

std::array<int, 3> ssil_mirror(const Pixel& p, const std::array<float, 4>& irr, float albedo) {
  const std::array<float, 3> l{p.color.r / 255.f, p.color.g / 255.f, p.color.b / 255.f};
  const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
  const std::array<float, 3> bounce{albedo * lighting::openpbr::from_half(lighting::openpbr::to_half(irr[0])),
                                    albedo * lighting::openpbr::from_half(lighting::openpbr::to_half(irr[1])),
                                    albedo * lighting::openpbr::from_half(lighting::openpbr::to_half(irr[2]))};
  const std::array<float, 3> out = composite::testing::legacy_ssil(l, fc, p.q7, p.ao / 255.f, p.code / 3.f, bounce);
  return {to_byte(out[0]), to_byte(out[1]), to_byte(out[2])};
}

struct SsilCase {
  std::vector<Pixel> pixels;
  std::vector<std::array<float, 4>> irr;
};

SsilCase ssil_cases(uint32_t count, uint32_t seed) {
  std::mt19937 rng{seed};
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  SsilCase out;
  for (uint32_t i = 0; i < count; ++i) {
    Pixel p{.color = {static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)),
                      255},
            .code = static_cast<uint8_t>(i % 7 == 0 ? 0 : 1 + i % 3),
            .fc = {static_cast<uint8_t>(byte(rng) / 4), static_cast<uint8_t>(byte(rng) / 4),
                   static_cast<uint8_t>(byte(rng) / 4)},
            .q7 = static_cast<uint8_t>(i % 5 == 0 ? byte(rng) / 2 : 0),
            .ao = static_cast<uint8_t>(i % 11 == 0 ? 255 : byte(rng))};
    out.pixels.push_back(p);
    const float scale = i % 13 == 0 ? 8.f : 0.6f;
    out.irr.push_back({unit(rng) * scale, unit(rng) * scale, unit(rng) * scale, unit(rng)});
  }
  return out;
}

TEST_F(CompositeGpu, SsilOffIsBitIdentical) {
  const SsilCase c = ssil_cases(64 * 48, 0x55A1u);
  const std::vector<Rgba> off = run_pixels(64, 48, c.pixels, {});
  const uint32_t before = composite::testing::counters().ssilDraws;
  const std::vector<Rgba> zero = run_pixels(64, 48, c.pixels, {.ssil = [](size_t) { return std::array<float, 4>{}; }});
  EXPECT_EQ(composite::testing::counters().ssilDraws, before + 1);
  size_t differ = 0;
  for (size_t i = 0; i < off.size(); ++i) {
    differ += off[i].r != zero[i].r || off[i].g != zero[i].g || off[i].b != zero[i].b || off[i].a != zero[i].a;
  }
  EXPECT_EQ(differ, 0u);
  for (size_t i = 0; i < off.size(); ++i) {
    const std::array<int, 3> want = mirror(c.pixels[i]);
    ASSERT_LE(std::abs(off[i].r - want[0]), 1) << i;
  }
}

TEST_F(CompositeGpu, SsilMatchesMirror) {
  const SsilCase c = ssil_cases(64 * 48, 0x55A2u);
  const std::vector<Rgba> out = run_pixels(64, 48, c.pixels, {.ssil = [&](size_t i) { return c.irr[i]; }});
  size_t worst = 0;
  size_t exact = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const std::array<int, 3> want = ssil_mirror(c.pixels[i], c.irr[i], ao::SsilLegacyAlbedo);
    const std::array<int, 3> got{out[i].r, out[i].g, out[i].b};
    size_t d = 0;
    for (size_t k = 0; k < 3; ++k) {
      d = std::max<size_t>(d, static_cast<size_t>(std::abs(got[k] - want[k])));
    }
    worst = std::max(worst, d);
    exact += d == 0 ? 1 : 0;
    EXPECT_EQ(out[i].a, c.pixels[i].color.a) << i;
  }
  std::printf("SSIL composite against the mirror: worst %zu code, %zu / %zu exact\n", worst, exact, out.size());
  EXPECT_LE(worst, 1u);
}

TEST_F(CompositeGpu, SsilNeverBrightens) {
  SsilCase c = ssil_cases(64 * 48, 0x55A3u);
  for (auto& irr : c.irr) {
    irr = {16.f, 16.f, 16.f, 1.f};
  }
  const std::vector<Rgba> ao = run_pixels(64, 48, c.pixels, {});
  const std::vector<Rgba> out = run_pixels(64, 48, c.pixels, {.ssil = [&](size_t i) { return c.irr[i]; }});
  size_t lifted = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = c.pixels[i];
    EXPECT_LE(out[i].r, p.color.r) << i;
    EXPECT_LE(out[i].g, p.color.g) << i;
    EXPECT_LE(out[i].b, p.color.b) << i;
    EXPECT_GE(out[i].r + 1, ao[i].r) << i;
    EXPECT_GE(out[i].g + 1, ao[i].g) << i;
    EXPECT_GE(out[i].b + 1, ao[i].b) << i;
    lifted += out[i].r > ao[i].r || out[i].g > ao[i].g || out[i].b > ao[i].b ? 1 : 0;
  }
  EXPECT_GT(lifted, out.size() / 4);
}

TEST_F(CompositeGpu, SsilKeepsReceiverHue) {
  std::vector<Pixel> pixels;
  std::vector<std::array<float, 4>> irr;
  for (uint32_t i = 0; i < 64 * 48; ++i) {
    const auto shade = static_cast<uint8_t>(60 + i % 160);
    pixels.push_back({.color = {static_cast<uint8_t>(shade / 3), shade, static_cast<uint8_t>(shade / 5), 255},
                      .code = 1,
                      .ao = static_cast<uint8_t>(90 + i % 120)});
    irr.push_back({0.2f, 0.2f, 0.2f, 1.f});
  }
  const std::vector<Rgba> out = run_pixels(64, 48, pixels, {.ssil = [&](size_t i) { return irr[i]; }});
  size_t checked = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = pixels[i];
    const double scaleG = static_cast<double>(out[i].g) / p.color.g;
    const double scaleR = static_cast<double>(out[i].r) / std::max<int>(p.color.r, 1);
    if (p.color.r >= 40 && out[i].g < p.color.g) {
      EXPECT_NEAR(scaleR, scaleG, 1.5 / p.color.r + 1.5 / p.color.g) << i;
      ++checked;
    }
  }
  EXPECT_GT(checked, 500u);
}

TEST_F(CompositeGpu, SsilIdentityStillDiscards) {
  std::vector<Pixel> pixels(64 * 48, control());
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i].ao = i % 2 == 0 ? 255 : 128;
    pixels[i].color.a = static_cast<uint8_t>(i % 251);
  }
  const std::vector<Rgba> out =
      run_pixels(64, 48, pixels, {.ssil = [](size_t) { return std::array<float, 4>{4.f, 4.f, 4.f, 1.f}; }});
  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_EQ(out[i].a, pixels[i].color.a) << i;
    if (pixels[i].ao == 255) {
      EXPECT_EQ(out[i].r, pixels[i].color.r) << i;
      EXPECT_EQ(out[i].g, pixels[i].color.g) << i;
      EXPECT_EQ(out[i].b, pixels[i].color.b) << i;
    }
  }
}

TEST_F(CompositeGpu, SsilUnderPbrScalesByExposure) {
  constexpr float E = 2.7f;
  const SsilCase c = ssil_cases(64 * 48, 0x55A4u);
  const std::vector<Rgba> out = run_pixels(
      64, 48, c.pixels,
      {.settings = pbr_settings(0.f, Quality::High), .exposure = E, .ssil = [&](size_t i) { return c.irr[i]; }});
  EXPECT_TRUE(composite::pbr_active());
  size_t worst = 0;
  size_t lifted = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const std::array<int, 3> want = ssil_mirror(c.pixels[i], c.irr[i], ao::SsilLegacyAlbedo * E);
    const std::array<int, 3> unexposed = ssil_mirror(c.pixels[i], c.irr[i], ao::SsilLegacyAlbedo);
    const std::array<int, 3> got{out[i].r, out[i].g, out[i].b};
    for (size_t k = 0; k < 3; ++k) {
      worst = std::max<size_t>(worst, static_cast<size_t>(std::abs(got[k] - want[k])));
    }
    lifted += want[0] > unexposed[0] + 1 || want[1] > unexposed[1] + 1 || want[2] > unexposed[2] + 1 ? 1 : 0;
  }
  std::printf("SSIL composite under FlagPbr at e %.1f: worst %zu code, %zu pixels above the unexposed bounce\n", E,
              worst, lifted);
  EXPECT_LE(worst, 1u);
  EXPECT_GT(lifted, out.size() / 10);
}

TEST_F(CompositeGpu, SsilSkippedMeterClearsRadiance) {
  const SsilCase c = ssil_cases(64 * 48, 0x55A5u);
  const Settings settings = pbr_settings(0.f, Quality::High);
  const auto irr = [&](size_t i) { return c.irr[i]; };
  run_pixels(64, 48, c.pixels, {.settings = settings, .exposure = 2.7f, .radiance = true, .ssil = irr});
  EXPECT_TRUE(composite::pbr_active());
  const size_t written =
      static_cast<size_t>(std::ranges::count_if(m_radiance, [](const std::array<float, 4>& r) { return r[3] > 0.f; }));
  EXPECT_GT(written, m_radiance.size() / 2);
  const std::vector<Rgba> aoOnly =
      run_pixels(64, 48, c.pixels, {.settings = settings, .exposure = 2.7f, .exposureEnabled = false});
  const std::vector<Rgba> skipped = run_pixels(64, 48, c.pixels,
                                               {.settings = settings,
                                                .exposure = 2.7f,
                                                .exposureEnabled = false,
                                                .radiance = true,
                                                .radianceClears = 1,
                                                .ssil = irr});
  ASSERT_EQ(skipped.size(), aoOnly.size());
  for (size_t i = 0; i < skipped.size(); ++i) {
    ASSERT_EQ(m_radiance[i], (std::array<float, 4>{})) << i;
    ASSERT_TRUE(same(skipped[i], aoOnly[i])) << i;
  }
}

TEST_F(CompositeGpu, OddExtentCoversEdges) {
  for (const auto [width, height] : {std::pair{61u, 37u}, std::pair{3u, 5u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const auto pixels = grid(width, height, [](uint32_t, uint32_t) { return control(); });
    const auto out = composite_pixels(width, height, pixels);
    const std::array<int, 3> want = mirror(control());
    for (size_t i = 0; i < out.size(); ++i) {
      ASSERT_LE(std::abs(out[i].r - want[0]), 1) << i;
      ASSERT_LE(std::abs(out[i].g - want[1]), 1) << i;
      ASSERT_LE(std::abs(out[i].b - want[2]), 1) << i;
      ASSERT_EQ(out[i].a, 255) << i;
    }
  }
}

Pixel random_pixel(std::mt19937& rng, uint8_t cls) {
  std::uniform_int_distribution<int> byte(0, 255);
  const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
  return Pixel{.color = {b(), b(), b(), 255},
               .code = static_cast<uint8_t>(byte(rng) % 4),
               .fc = {b(), b(), b()},
               .q7 = static_cast<uint8_t>(byte(rng) % 128),
               .ao = b(),
               .albedo = {b(), b(), b()},
               .cls = cls};
}

TEST_F(CompositeGpu, PbrBlendOneMatchesPbrOff) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{23};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto b = [&] { return static_cast<uint8_t>(byte(rng)); };
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t y) {
    if (x < 24) {
      const auto l = static_cast<uint8_t>(x * 10 + 15);
      return Pixel{.color = {l, static_cast<uint8_t>(y * 5 + 10), b(), 255},
                   .code = 1,
                   .fc = {b(), b(), b()},
                   .q7 = static_cast<uint8_t>(y % 2 == 0 ? 0 : 64),
                   .ao = 64,
                   .albedo = {b(), b(), b()},
                   .cls = 1};
    }
    if (x < 40) {
      return Pixel{.color = {0, 0, 0, 255}, .code = 1, .q7 = 0, .ao = 255, .albedo = {255, 255, 255}, .cls = 1};
    }
    return control();
  });
  const auto region = [](size_t i) { return i % 64 < 24 ? 1 : i % 64 < 40 ? 2 : 0; };
  const auto off = run_pixels(64, 48, pixels, {.settings = ao_settings()});
  const uint32_t pbrDraws = composite::testing::counters().pbrDraws;
  const auto on = run_pixels(64, 48, pixels,
                             {.settings = pbr_settings(1.f, Quality::High),
                              .params = composite::Params{.flags = composite::FlagPbr, .legacyBlend = 1.f}});
  EXPECT_EQ(composite::testing::counters().pbrDraws, pbrDraws + 1);
  size_t changed = 0;
  for (size_t i = 0; i < on.size(); ++i) {
    ASSERT_TRUE(same(on[i], off[i])) << i;
    if (region(i) != 2) {
      changed += same(on[i], pixels[i].color) ? 0 : 1;
    }
  }
  EXPECT_GT(changed, on.size() / 3);
  const auto near = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.99f, Quality::High)});
  const auto half = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.5f, Quality::High)});
  size_t halfChanged = 0;
  for (size_t i = 0; i < near.size(); ++i) {
    if (region(i) == 2) {
      ASSERT_LE(std::abs(near[i].r - 3), 1) << i;
      ASSERT_LE(std::abs(near[i].g - 3), 1) << i;
    } else if (region(i) == 1) {
      halfChanged += same(half[i], off[i]) ? 0 : 1;
    } else {
      ASSERT_TRUE(same(near[i], off[i])) << i;
      ASSERT_TRUE(same(half[i], off[i])) << i;
    }
  }
  EXPECT_GT(halfChanged, 0u);
  const auto culled = run_pixels(64, 48, pixels, {.settings = pbr_settings(1.f), .aoProducer = false, .draws = 0});
  for (size_t i = 0; i < culled.size(); ++i) {
    ASSERT_TRUE(same(culled[i], pixels[i].color)) << i;
  }
}

TEST_F(CompositeGpu, PbrBlendZeroUsesSceneHdr) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{29};
  const auto pixels = grid(128, 128, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 4 == 0 ? 0 : 1); });
  const auto out = run_pixels(128, 128, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (guard_tie(pixels[i])) {
      continue;
    }
    ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.f), 1)) << i;
    ASSERT_EQ(out[i].a, 255) << i;
    changed += pixels[i].cls != 0 && !same(out[i], pixels[i].color) ? 1 : 0;
  }
  EXPECT_GT(changed, out.size() / 4);
}

TEST_F(CompositeGpu, PbrBlendHalfMatchesMirror) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{31};
  const auto pixels = grid(128, 128, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 3 == 0 ? 0 : 1); });
  const auto off = run_pixels(128, 128, pixels, {.settings = ao_settings()});
  const auto out = run_pixels(128, 128, pixels, {.settings = pbr_settings(0.5f, Quality::High)});
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].cls == 0) {
      ASSERT_TRUE(same(out[i], off[i])) << i;
      continue;
    }
    if (guard_tie(pixels[i])) {
      continue;
    }
    ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.5f), 1)) << i;
    changed += same(out[i], off[i]) ? 0 : 1;
  }
  EXPECT_GT(changed, out.size() / 4);
}

TEST_F(CompositeGpu, PbrBlendHalfFollowsExposureAndTonemap) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{53};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 3 == 0 ? 0 : 1); });
  for (const float e : {0.5f, 1.f, 2.7f}) {
    for (const Tonemap tm : {Tonemap::PbrNeutral, Tonemap::AgX}) {
      SCOPED_TRACE(std::to_string(e) + (tm == Tonemap::AgX ? " agx" : " neutral"));
      Settings settings = pbr_settings(0.5f, Quality::High);
      settings.tonemap = tm;
      const auto out = run_pixels(64, 48, pixels, {.settings = settings, .exposure = e});
      for (size_t i = 0; i < out.size(); ++i) {
        if (guard_tie(pixels[i])) {
          continue;
        }
        ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.5f, e, tm == Tonemap::AgX), 1)) << i;
      }
    }
  }
}

TEST_F(CompositeGpu, ExposureDummyIsIdentity) {
  std::mt19937 rng{59};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 4 == 0 ? 0 : 1); });
  const auto dummy = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  const auto one = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .exposure = 1.f});
  for (size_t i = 0; i < one.size(); ++i) {
    ASSERT_TRUE(same(dummy[i], one[i])) << i;
  }
}

TEST_F(CompositeGpu, AoTermRefactorIsBitIdentical) {
  std::mt19937 rng{61};
  const auto pixels = grid(61, 37, [&](uint32_t, uint32_t) { return random_pixel(rng, 0); });
  const auto out = run_pixels(61, 37, pixels, {.settings = ao_settings()});
  for (size_t i = 0; i < out.size(); ++i) {
    if (guard_tie(pixels[i])) {
      continue;
    }
    ASSERT_TRUE(same_rgb(out[i], mirror(pixels[i]), 1)) << i;
  }
}

TEST_F(CompositeGpu, NonFiniteHdrStaysFinite) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  const auto pixels = grid(64, 48, [](uint32_t, uint32_t) {
    Pixel p = control();
    p.albedo = {200, 200, 200};
    p.cls = 1;
    return p;
  });
  constexpr std::array<uint16_t, 3> Specials{0x7C00u, 0x7E00u, 0xFC00u};
  const auto edit = [&](std::array<uint16_t, 4>& words, size_t i) { words[i % 3] = Specials[(i / 3) % 3]; };
  for (const Tonemap tm : {Tonemap::PbrNeutral, Tonemap::AgX}) {
    Settings settings = pbr_settings(0.f, Quality::High);
    settings.tonemap = tm;
    const auto out = run_pixels(64, 48, pixels, {.settings = settings, .exposure = 2.f, .hdrEdit = edit});
    for (size_t i = 0; i < out.size(); ++i) {
      std::array<uint16_t, 4> words = hdr_words(pixels[i]);
      edit(words, i);
      const std::array<float, 4> hdr{lighting::openpbr::from_half(words[0]), lighting::openpbr::from_half(words[1]),
                                     lighting::openpbr::from_half(words[2]), 1.f};
      ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.f, 2.f, tm == Tonemap::AgX, hdr), 1)) << i;
    }
  }
}

TEST_F(CompositeGpu, ClassZeroKeepsLegacyBits) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{37};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) {
    if (x < 8) {
      Pixel p = control();
      p.albedo = {10, 240, 10};
      p.cls = 1;
      return p;
    }
    return random_pixel(rng, 0);
  });
  const auto off = run_pixels(64, 48, pixels, {.settings = ao_settings()});
  const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].cls == 0) {
      ASSERT_TRUE(same(out[i], off[i])) << i;
    } else {
      ASSERT_FALSE(same(out[i], off[i])) << i;
    }
  }
}

TEST_F(CompositeGpu, PbrBadGuardKeepsL) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  const auto pixels = grid(64, 48, [](uint32_t x, uint32_t) {
    Pixel p =
        x < 8 ? control() : Pixel{.color = {60, 200, 200, 255}, .code = 1, .fc = {120, 0, 0}, .q7 = 127, .ao = 32};
    p.albedo = {10, 240, 10};
    p.cls = 1;
    return p;
  });
  const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].q7 == 127) {
      ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
    } else {
      ASSERT_FALSE(same(out[i], pixels[i].color)) << i;
    }
  }
}

TEST_F(CompositeGpu, DummyLoadsClamp) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{41};
  const auto pixels = grid(61, 37, [&](uint32_t x, uint32_t) {
    Pixel p = random_pixel(rng, x % 5 == 0 ? 0 : 1);
    p.ao = 255;
    return p;
  });
  const auto out = run_pixels(61, 37, pixels, {.settings = pbr_settings(0.f), .aoProducer = false});
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (pixels[i].cls == 0) {
      ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
      continue;
    }
    if (guard_tie(pixels[i])) {
      continue;
    }
    ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.f), 1)) << i;
    changed += same(out[i], pixels[i].color) ? 0 : 1;
  }
  EXPECT_GT(changed, out.size() / 4);
}

TEST_F(CompositeGpu, PbrLeavesEfbAlphaUntouched) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  const auto pixels = grid(64, 48, [](uint32_t x, uint32_t y) {
    Pixel p = control();
    p.color.a = static_cast<uint8_t>((y * 64 + x) & 0xFF);
    p.albedo = {10, 240, 10};
    p.cls = 1;
    return p;
  });
  const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  size_t changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i].a, pixels[i].color.a) << i;
    changed += out[i].r != pixels[i].color.r ? 1 : 0;
  }
  EXPECT_EQ(changed, out.size());
}

TEST_F(CompositeGpu, HdrDummyKeepsSb) {
  std::mt19937 rng{47};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 4 == 0 ? 0 : 1); });
  const auto off = run_pixels(64, 48, pixels, {.settings = ao_settings()});
  const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .hdrEnabled = false});
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(same(out[i], off[i])) << i;
  }
}

TEST_F(CompositeGpu, HdrAlphaZeroKeepsSb) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{67};
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) { return random_pixel(rng, 1); });
  // A bright H under every alpha, so a pixel that reads it cannot pass for Sb
  constexpr std::array<float, 4> Alphas{0.f, 0.49f, 0.5f, 1.f};
  const auto edit = [&](std::array<uint16_t, 4>& words, size_t i) {
    words = {lighting::openpbr::to_half(4.f), lighting::openpbr::to_half(2.f), lighting::openpbr::to_half(1.f),
             lighting::openpbr::to_half(Alphas[i % Alphas.size()])};
  };
  const auto off = run_pixels(64, 48, pixels, {.settings = ao_settings()});
  const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .hdrEdit = edit});
  size_t lit = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    std::array<uint16_t, 4> words{};
    edit(words, i);
    const std::array<float, 4> hdr{lighting::openpbr::from_half(words[0]), lighting::openpbr::from_half(words[1]),
                                   lighting::openpbr::from_half(words[2]), lighting::openpbr::from_half(words[3])};
    if (hdr[3] < 0.5f) {
      ASSERT_TRUE(same(out[i], off[i])) << i;
      continue;
    }
    if (guard_tie(pixels[i])) {
      continue;
    }
    ASSERT_TRUE(same_rgb(out[i], pbr_mirror(pixels[i], 0.f, 1.f, false, hdr), 1)) << i;
    lit += same(out[i], off[i]) ? 0 : 1;
  }
  EXPECT_GT(lit, out.size() / 8);
}

TEST_F(CompositeGpu, IdentityForcedIgnoresPbr) {
  std::mt19937 rng{43};
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) { return random_pixel(rng, 1); });
  const auto out = run_pixels(
      64, 48, pixels,
      {.settings = pbr_settings(0.f, Quality::High),
       .params = composite::Params{.flags = composite::FlagPbr | composite::FlagIdentity, .legacyBlend = 0.f}});
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
  }
}
TEST_F(CompositeGpu, LsmPreShadowReceiverMasked) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{71};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 4 == 0 ? 0 : 1); });
  std::vector<std::array<uint8_t, 4>> masks(pixels.size());
  for (size_t i = 0; i < masks.size(); ++i) {
    // Every third texel stays white, the rest carry a kept fake of any strength and hue
    masks[i] = i % 3 == 0 ? std::array<uint8_t, 4>{255, 255, 255, 0}
                          : std::array<uint8_t, 4>{static_cast<uint8_t>(rng()), static_cast<uint8_t>(rng()),
                                                   static_cast<uint8_t>(rng()), static_cast<uint8_t>(rng())};
  }
  const auto off = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High)});
  const auto out = run_pixels(
      64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .mask = [&](size_t i) { return masks[i]; }});
  size_t masked = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const Pixel& p = pixels[i];
    if (p.cls == 0) {
      ASSERT_TRUE(same(out[i], off[i])) << i;
      continue;
    }
    if (guard_tie(p)) {
      continue;
    }
    const float q = static_cast<float>(p.q7) / 127.f;
    const std::array<float, 3> m =
        fakeshadow::lsm::term({masks[i][0] / 255.f, masks[i][1] / 255.f, masks[i][2] / 255.f}, q,
                              {p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f});
    ASSERT_TRUE(same_rgb(out[i], pbr_mirror(p, 0.f, 1.f, false, std::nullopt, m), 1)) << i;
    if (i % 3 == 0) {
      ASSERT_TRUE(same(out[i], off[i])) << i;
    }
    masked += same(out[i], off[i]) ? 0 : 1;
  }
  EXPECT_GT(masked, out.size() / 8);
}

TEST_F(CompositeGpu, LsmPostShadowReceiverUnmasked) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{73};
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) { return random_pixel(rng, 1); });
  // Pixels drawn after the game's fake-shadow pass write 0.75, which the fakes never darkened
  const auto post = [](std::array<uint16_t, 4>& words, size_t) {
    words[3] = lighting::openpbr::to_half(fakeshadow::lsm::PostShadowAlpha);
  };
  const auto dark = [](size_t) { return std::array<uint8_t, 4>{30, 40, 50, 0}; };
  const auto off = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .hdrEdit = post});
  const auto out =
      run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .hdrEdit = post, .mask = dark});
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(same(out[i], off[i])) << i;
  }
  const auto pre = run_pixels(64, 48, pixels, {.settings = pbr_settings(0.f, Quality::High), .mask = dark});
  size_t masked = 0;
  for (size_t i = 0; i < pre.size(); ++i) {
    masked += same(pre[i], off[i]) ? 0 : 1;
  }
  EXPECT_GT(masked, pre.size() / 4);
}

TEST_F(CompositeGpu, LsmUnitMaskDiscards) {
  std::mt19937 rng{79};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 3 == 0 ? 0 : 1); });
  const auto white = [](size_t) { return std::array<uint8_t, 4>{255, 255, 255, 64}; };
  for (const float lb : {0.f, 0.5f, 1.f}) {
    const auto off = run_pixels(64, 48, pixels, {.settings = pbr_settings(lb, Quality::High)});
    const auto out = run_pixels(64, 48, pixels, {.settings = pbr_settings(lb, Quality::High), .mask = white});
    for (size_t i = 0; i < out.size(); ++i) {
      ASSERT_TRUE(same(out[i], off[i])) << lb << " " << i;
    }
  }
}

float srgb_linear(float s) { return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f); }

float half_ulp(float x) {
  const float a = std::abs(x);
  if (a < 6.103515625e-05f) {
    return 5.9604645e-08f;
  }
  return std::ldexp(1.f, std::ilogb(a) - 10);
}

bool radiance_near(const std::array<float, 4>& gpu, const std::array<float, 4>& cpu) {
  for (size_t c = 0; c < 4; ++c) {
    if (std::abs(gpu[c] - cpu[c]) > half_ulp(cpu[c]) + 1e-7f) {
      return false;
    }
  }
  return true;
}

std::array<float, 4> radiance_mirror(const Pixel& p, bool pbrFlag, float legacyBlend, float invE,
                                     std::optional<std::array<float, 4>> hdr = std::nullopt) {
  const std::array<float, 3> l{p.color.r / 255.f, p.color.g / 255.f, p.color.b / 255.f};
  const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
  return composite::testing::radiance(l, fc, p.q7, p.ao / 255.f, p.code / 3.f, hdr.value_or(hdr_of(p)), pbrFlag,
                                      legacyBlend, invE);
}

TEST_F(CompositeGpu, RadianceRewriteIsExact) {
  std::mt19937 rng{47};
  std::uniform_int_distribution<int> byte(0, 255);
  // Rows 0-2 sweep one channel through every code on open receivers, rows 3-5 on code-0 pixels under AO
  const auto pixels = grid(256, 6, [&](uint32_t x, uint32_t y) {
    std::array<uint8_t, 3> rgb{static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)),
                               static_cast<uint8_t>(byte(rng))};
    rgb[y % 3] = static_cast<uint8_t>(x);
    return Pixel{.color = {rgb[0], rgb[1], rgb[2], static_cast<uint8_t>(byte(rng))},
                 .code = static_cast<uint8_t>(y < 3 ? 1 : 0),
                 .fc = {static_cast<uint8_t>(byte(rng)), 0, 0},
                 .q7 = 0,
                 .ao = static_cast<uint8_t>(y < 3 ? 255 : 64)};
  });
  const auto out = run_pixels(256, 6, pixels, {.radiance = true});
  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
    ASSERT_EQ(m_radiance[i][3], 1.f) << i;
  }
}

TEST_F(CompositeGpu, RadianceVariantMatchesDefaultColour) {
  std::mt19937 rng{53};
  const auto pixels = grid(61, 37, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 3 == 0 ? 0 : 1); });
  for (const Settings& settings : {ao_settings(), pbr_settings(0.f, Quality::High), pbr_settings(0.5f, Quality::High),
                                   pbr_settings(1.f, Quality::High)}) {
    SCOPED_TRACE(settings.pbr ? settings.legacyBlend : -1.f);
    const auto plain = run_pixels(61, 37, pixels, {.settings = settings});
    const auto radiance = run_pixels(61, 37, pixels, {.settings = settings, .radiance = true});
    size_t changed = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
      ASSERT_TRUE(same(radiance[i], plain[i])) << i;
      changed += same(plain[i], pixels[i].color) ? 0 : 1;
    }
    EXPECT_GT(changed, plain.size() / 8);
  }
}

TEST_F(CompositeGpu, RadianceRemovesFogColour) {
  constexpr std::array<std::array<uint8_t, 3>, 5> Fogs{
      {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}, {128, 153, 179}}};
  constexpr uint32_t Rows = 8;
  std::mt19937 rng{59};
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<std::array<uint8_t, 3>> surface(128 * Rows * Fogs.size());
  for (auto& s : surface) {
    s = {static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng))};
  }
  const auto pixels = grid(128, Rows * Fogs.size(), [&](uint32_t x, uint32_t y) {
    const auto& fc = Fogs[y / Rows];
    const auto& s = surface[size_t{y} * 128 + x];
    const float q = static_cast<float>(x) / 127.f;
    std::array<uint8_t, 3> l{};
    for (size_t c = 0; c < 3; ++c) {
      l[c] = static_cast<uint8_t>(std::lround(255.f * ((1.f - q) * s[c] / 255.f + q * fc[c] / 255.f)));
    }
    return Pixel{.color = {l[0], l[1], l[2], 255}, .code = 1, .fc = fc, .q7 = static_cast<uint8_t>(x), .ao = 255};
  });
  run_pixels(128, Rows * Fogs.size(), pixels, {.radiance = true});
  for (size_t f = 0; f < Fogs.size(); ++f) {
    const std::array<float, 3> fc{Fogs[f][0] / 255.f, Fogs[f][1] / 255.f, Fogs[f][2] / 255.f};
    const float norm = std::sqrt(fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2]);
    double along = 0.0;
    size_t count = 0;
    for (uint32_t y = f * Rows; y < (f + 1) * Rows; ++y) {
      for (uint32_t x = 0; x < 128; ++x) {
        const size_t i = size_t{y} * 128 + x;
        const std::array<float, 4>& r = m_radiance[i];
        ASSERT_EQ(r[3], 1.f) << i;
        const float q = static_cast<float>(x) / 127.f;
        const float divisor = std::max(1.f - q, 13.f / 127.f);
        const float bound = 0.5f / 255.f / divisor;
        double dot = 0.0;
        for (size_t c = 0; c < 3; ++c) {
          const float expected = surface[i][c] / 255.f * (1.f - q) / divisor;
          const float lo = srgb_linear(std::max(expected - bound, 0.f));
          const float hi = srgb_linear(std::min(expected + bound, 1.f));
          ASSERT_GE(r[c], lo - half_ulp(lo) - 1e-7f) << i << " " << c;
          ASSERT_LE(r[c], hi + half_ulp(hi) + 1e-7f) << i << " " << c;
          dot += (encode_srgb(r[c]) - expected) * fc[c] / norm;
        }
        along += dot;
        ++count;
      }
    }
    EXPECT_LE(std::abs(along / static_cast<double>(count)) * 255.0, 0.1) << f;
  }
}

TEST_F(CompositeGpu, RadianceUnitsIgnoreExposureWithoutPbr) {
  std::mt19937 rng{61};
  const auto pixels = grid(64, 48, [&](uint32_t, uint32_t) { return random_pixel(rng, 0); });
  run_pixels(64, 48, pixels, {.radiance = true});
  const std::vector<std::array<float, 4>> white = m_radiance;
  run_pixels(64, 48, pixels, {.exposure = 2.7f, .radiance = true});
  size_t lit = 0;
  for (size_t i = 0; i < white.size(); ++i) {
    ASSERT_EQ(m_radiance[i], white[i]) << i;
    lit += white[i][0] > 0.f ? 1 : 0;
  }
  EXPECT_GT(lit, white.size() / 4);
  const auto colour = run_pixels(64, 48, pixels,
                                 {.params = composite::Params{.flags = composite::FlagPbr, .legacyBlend = 0.f},
                                  .exposure = 2.7f,
                                  .radiance = true});
  const auto plain = run_pixels(64, 48, pixels, {});
  for (size_t i = 0; i < white.size(); ++i) {
    ASSERT_TRUE(same(colour[i], plain[i])) << i;
    ASSERT_EQ(m_radiance[i][3], white[i][3]) << i;
    for (size_t c = 0; c < 3; ++c) {
      ASSERT_NEAR(m_radiance[i][c] * 2.7f, white[i][c], half_ulp(white[i][c]) * 2.7f + 1e-7f) << i << " " << c;
    }
  }
}

TEST_F(CompositeGpu, RadianceMatchesMirror) {
  if (composite::identity_forced()) {
    GTEST_SKIP() << "AURORA_RENDER_COMPOSITE_IDENTITY is set";
  }
  std::mt19937 rng{67};
  const auto pixels = grid(128, 64, [&](uint32_t x, uint32_t y) {
    Pixel p = random_pixel(rng, x % 4 == 0 ? 0 : 1);
    p.q7 = static_cast<uint8_t>(x);
    if (y % 8 == 0) {
      p.ao = 255;
    }
    return p;
  });
  size_t checked = 0;
  size_t pbr = 0;
  for (const float e : {0.5f, 1.f, 2.7f}) {
    for (const bool agx : {false, true}) {
      for (const float lb : {0.f, 0.5f, 1.f}) {
        SCOPED_TRACE(std::to_string(e) + (agx ? " agx " : " neutral ") + std::to_string(lb));
        Settings settings = pbr_settings(lb, Quality::High);
        settings.tonemap = agx ? Tonemap::AgX : Tonemap::PbrNeutral;
        run_pixels(128, 64, pixels, {.settings = settings, .exposure = e, .radiance = true});
        const bool armed = lb < 1.f;
        for (size_t i = 0; i < pixels.size(); ++i) {
          if (guard_tie(pixels[i])) {
            continue;
          }
          const std::array<float, 4> cpu = radiance_mirror(pixels[i], armed, lb, armed ? 1.f / e : 1.f);
          ASSERT_TRUE(radiance_near(m_radiance[i], cpu))
              << i << ": " << m_radiance[i][0] << " " << m_radiance[i][1] << " " << m_radiance[i][2] << " "
              << m_radiance[i][3] << " vs " << cpu[0] << " " << cpu[1] << " " << cpu[2] << " " << cpu[3];
          ++checked;
          pbr += armed && pixels[i].cls != 0 && cpu[3] > 0.f ? 1 : 0;
        }
      }
    }
  }
  EXPECT_GT(pbr, checked / 6);
}

TEST_F(CompositeGpu, RadianceBadPixelsInvalid) {
  std::mt19937 rng{71};
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) {
    Pixel p = random_pixel(rng, 0);
    if (x % 2 == 0) {
      p.color = {10, 12, 14, 255};
      p.fc = {255, 255, 255};
      p.q7 = 64;
    }
    return p;
  });
  const auto out = run_pixels(64, 48, pixels, {.radiance = true});
  size_t bad = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const std::array<float, 4> cpu = radiance_mirror(pixels[i], false, 0.f, 1.f);
    if (i % 2 == 0) {
      ASSERT_EQ(cpu[3], 0.f) << i;
      ASSERT_EQ(m_radiance[i], (std::array<float, 4>{})) << i;
      ASSERT_TRUE(same(out[i], pixels[i].color)) << i;
      ++bad;
    } else if (!guard_tie(pixels[i])) {
      ASSERT_EQ(m_radiance[i][3], cpu[3]) << i;
    }
  }
  EXPECT_EQ(bad, out.size() / 2);
}

TEST_F(CompositeGpu, RadianceLeavesEfbAlphaUntouched) {
  std::mt19937 rng{73};
  std::uniform_int_distribution<int> byte(0, 255);
  const auto pixels = grid(64, 48, [&](uint32_t x, uint32_t) {
    Pixel p = random_pixel(rng, x % 2 == 0 ? 0 : 1);
    p.color.a = static_cast<uint8_t>(byte(rng));
    return p;
  });
  for (const Settings& settings : {ao_settings(), pbr_settings(0.f, Quality::High)}) {
    const auto out = run_pixels(64, 48, pixels, {.settings = settings, .radiance = true});
    for (size_t i = 0; i < out.size(); ++i) {
      ASSERT_EQ(out[i].a, pixels[i].color.a) << i;
    }
  }
}

TEST_F(CompositeGpu, RadianceEncodesValidationClean) {
  for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{61u, 37u}, std::pair{4779u, 9u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    std::mt19937 rng{width};
    const auto pixels = grid(width, height, [&](uint32_t x, uint32_t) { return random_pixel(rng, x % 2); });
    for (const Settings& settings : {ao_settings(), pbr_settings(0.f, Quality::High)}) {
      run_pixels(width, height, pixels, {.settings = settings, .radiance = true});
      EXPECT_EQ(m_radianceExtent.width, width);
      EXPECT_EQ(m_radianceExtent.height, height);
      EXPECT_EQ(m_radiance.size(), size_t{width} * height);
    }
  }
}
} // namespace
} // namespace aurora::render::detail

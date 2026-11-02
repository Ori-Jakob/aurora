#include <gtest/gtest.h>

#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/ao/ao.hpp"
#include "render/ao/upsample.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_layout.hpp"
#include "render/graph_pool.hpp"
#include "render/hints.hpp"
#include "render/jobs.hpp"
#include "render/module.hpp"
#include "render/pass_context.hpp"
#include "render/post/composite.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
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
  EXPECT_LE(sizeof(composite::Params), MaxPassParamsSize);
  EXPECT_EQ(composite::FlagIdentity, 1u);
  EXPECT_EQ(composite::FlagNoFog, 2u);
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
  }
}

TEST(CompositeRules, WantedFollowsAoOnly) {
  EXPECT_FALSE(composite::wanted(Settings{}));
  EXPECT_FALSE(composite::wanted({.taa = true}));
  EXPECT_FALSE(composite::wanted({.upscaler = Upscaler::Fsr}));
  for (const Quality quality : {Quality::Low, Quality::Medium, Quality::High, Quality::Ultra}) {
    EXPECT_TRUE(composite::wanted(ao_settings(quality)));
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
  EXPECT_EQ(composite::LegacyTex, "legacyTex");
  EXPECT_EQ(composite::CopyPass, "legacy.copy");
  EXPECT_EQ(composite::Pass, "composite.main");
  EXPECT_EQ(composite::AoFormat, ao::upsample::Format);
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

  [[nodiscard]] std::shared_ptr<CompiledGraph> compile(const Settings& settings) {
    m_settings = settings;
    m_env = gather_graph_environment();
    if (m_sceneSize) {
      m_env.sceneSize = *m_sceneSize;
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

  void encode_sizes() {
    for (const auto [width, height] :
         {std::pair{1u, 1u}, std::pair{61u, 37u}, std::pair{1280u, 720u}, std::pair{4779u, 9u}}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      ASSERT_NO_FATAL_FAILURE(use_globals({width, height, 1}));
      make_depth();
      for (const uint32_t flags : {0u, uint32_t{composite::FlagIdentity}, uint32_t{composite::FlagNoFog}}) {
        composite::testing::reset();
        composite::testing::set_camera_override(m_hints);
        composite::testing::set_flags(flags);
        StubAo stub;
        const PassId stubId = register_pass(stub_ao_desc(stub));
        const auto graph = live_graph(ao_settings());
        ASSERT_TRUE(composite::live());
        const FrameState state{.graph = graph};
        EXPECT_EQ(run(state, 1), std::nullopt);
        EXPECT_EQ(run(state, 2), std::nullopt);
        EXPECT_EQ(composite::testing::counters().copies, 2u);
        EXPECT_EQ(composite::testing::counters().draws, 2u);
        EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
        EXPECT_EQ(stub.calls, 2);
        unregister_pass(stubId);
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

TEST_F(CompositeGraph, BindingOrderLegacyFogNormalAo) {
  StubAo stub;
  register_pass(stub_ao_desc(stub));
  const auto graph = compile(ao_settings());
  const NodeId node = graph->find_node(composite::Pass);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  const auto uses = graph->node_uses(node);
  ASSERT_EQ(uses.size(), 5u);
  const std::array<std::string_view, 5> names{composite::LegacyTex, resource::Fog, resource::GxNormal,
                                              resource::AoFinal, resource::GxColor};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, names[i]) << i;
    EXPECT_EQ(uses[i].access, i < 4 ? Access::Sampled : Access::ColorRMW) << i;
    if (i < 4) {
      EXPECT_EQ(uses[i].missing, Missing::DisablePass) << i;
    }
  }
  EXPECT_EQ(uses[composite::BindFog].options.sampleType, wgpu::TextureSampleType::Uint);
  EXPECT_EQ(graph->nodes[node].paramsSize, sizeof(composite::Params));
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
    if (webgpu::g_frameBuffer.size.width != width || webgpu::g_frameBuffer.size.height != height) {
      use_globals({width, height, 1});
      make_depth();
    }
    composite::testing::set_flags(flags);
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
    const PassId stubId = register_pass(stub_ao_desc(stub));
    const uint32_t draws = composite::testing::counters().draws;
    const auto graph = live_graph(ao_settings());
    EXPECT_TRUE(composite::live());
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, ++m_frame), std::nullopt);
    unregister_pass(stubId);
    EXPECT_EQ(composite::testing::counters().draws, draws + 1);
    EXPECT_EQ(composite::testing::counters().skippedMismatch, 0u);
    return read_color(width, height);
  }

  static std::array<int, 3> mirror(const Pixel& p) {
    const std::array<float, 3> l{p.color.r / 255.f, p.color.g / 255.f, p.color.b / 255.f};
    const std::array<float, 3> fc{p.fc[0] / 255.f, p.fc[1] / 255.f, p.fc[2] / 255.f};
    const std::array<float, 3> out = composite::testing::legacy_ao(l, fc, p.q7, p.ao / 255.f, p.code / 3.f);
    return {to_byte(out[0]), to_byte(out[1]), to_byte(out[2])};
  }

  static bool same(const Rgba& a, const Rgba& b) { return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a; }

  uint64_t m_frame = 0;
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
} // namespace
} // namespace aurora::render::detail

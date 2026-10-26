#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/frame.hpp"
#include "gx/hints.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/history_reset.hpp"
#include "render/jobs.hpp"
#include "render/module.hpp"
#include "render/post/taa.hpp"
#include "render/post/temporal_resolve.hpp"
#include "render/scene/motion.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

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
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace motion = scene::motion;
namespace records = gx_ext::records;
using wgpu::TextureFormat;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

gx::hints::Camera main_camera(uint32_t cutEpoch) {
  const Mat4x4<float> proj = perspective();
  return {
      .role = AURORA_CAMERA_MAIN,
      .view = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}},
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .cutEpoch = cutEpoch,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
}

Settings taa_settings(float sharpness = 0.f) {
  Settings settings;
  settings.taa = true;
  settings.taaSharpness = sharpness;
  return settings;
}

struct FakeStorage {
  static inline std::vector<uint8_t> bytes;
  static inline bool refuse = false;

  static void reset() {
    bytes.clear();
    refuse = false;
  }

  static gfx::Range push(std::span<const uint8_t> data) noexcept {
    if (refuse) {
      return {};
    }
    const size_t begin = std::max<size_t>((bytes.size() + 15) & ~size_t{15}, 16);
    bytes.resize(begin);
    bytes.insert(bytes.end(), data.begin(), data.end());
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(data.size())};
  }
};

std::atomic_bool g_stubReady{false};

bool stub_wanted(const Settings&) noexcept { return true; }

bool stub_unwanted(const Settings&) noexcept { return false; }

bool stub_ready() noexcept { return g_stubReady.load(); }

constexpr temporal_resolve::Resolver StubResolver{
    .pass = "test.upscaler",
    .jitterPhases = 16,
    .wanted = &stub_wanted,
    .jitter_ready = &stub_ready,
};

constexpr temporal_resolve::Resolver UnwantedResolver{
    .pass = "test.upscaler",
    .jitterPhases = 8,
    .wanted = &stub_unwanted,
    .jitter_ready = &stub_ready,
};

TEST(TaaRules, SharpnessZeroIsPassthrough) {
  EXPECT_EQ(taa::rcas_strength(0.f), 0.f);
  EXPECT_EQ(taa::rcas_strength(-0.5f), 0.f);
  EXPECT_EQ(taa::rcas_strength(std::numeric_limits<float>::quiet_NaN()), 0.f);
  EXPECT_EQ(taa::rcas_strength(0.25f), 0.25f);
  EXPECT_EQ(taa::rcas_strength(3.f), 1.f);
  EXPECT_EQ(taa::rcas_strength(std::numeric_limits<float>::infinity()), 1.f);
  EXPECT_EQ(taa::Params{}.sharpness, 0.f);
}

TEST(TaaRules, ParamsAreSixteenBytes) {
  static_assert(sizeof(taa::Params) == 16);
  static_assert(offsetof(taa::Params, sharpness) == 0);
  static_assert(offsetof(taa::Params, flags) == 4);
  EXPECT_EQ(taa::pass_desc().paramsSize, 16u);
  EXPECT_LE(taa::pass_desc().paramsSize, MaxPassParamsSize);
}

TEST(TaaRules, ApplyTargetIsRgbOnly) {
  for (const TextureFormat format : {TextureFormat::BGRA8Unorm, TextureFormat::RGBA8Unorm}) {
    const shaders::RenderPipelineDesc desc = taa::testing::apply_desc(format);
    EXPECT_EQ(desc.fragmentEntry, "fs_apply");
    ASSERT_EQ(desc.colorCount, 1u);
    EXPECT_EQ(desc.colors[0].format, format);
    EXPECT_EQ(desc.colors[0].writeMask, RgbMask);
    EXPECT_FALSE(desc.colors[0].blend);
    EXPECT_EQ(desc.sampleCount, 1u);
  }
}

TEST(TaaRules, ResolveTargetIsHistoryFormat) {
  for (const bool reset : {false, true}) {
    const shaders::RenderPipelineDesc desc = taa::testing::resolve_desc(reset);
    EXPECT_EQ(desc.fragmentEntry, "fs_resolve");
    ASSERT_EQ(desc.colorCount, 1u);
    EXPECT_EQ(desc.colors[0].format, TextureFormat::RGBA16Float);
    EXPECT_EQ(desc.colors[0].writeMask, wgpu::ColorWriteMask::All);
    EXPECT_EQ(desc.defines.size(), reset ? 1u : 0u);
    if (reset) {
      EXPECT_EQ(desc.defines[0], "TAA_RESET");
    }
  }
  const TexDesc history = taa::history_desc();
  EXPECT_EQ(history.format, taa::HistoryFormat);
  EXPECT_EQ(history.sizeClass, SizeClass::Efb);
  EXPECT_EQ(history.extraUsage, wgpu::TextureUsage::TextureBinding);
}

struct FrameOptions {
  uint32_t cutEpoch = 0;
  bool camera = true;
  bool marker = true;
  bool atFinish = false;
  bool invalidate = false;
  bool refuseStorage = false;
};

class TaaDeviceTest : public ::testing::Test {
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
    FakeStorage::reset();
    records::testing::set_push(&FakeStorage::push);
    motion::testing::set_push(&FakeStorage::push);
    motion::testing::set_jitter_env(motion::testing::JitterEnv::Off);
    motion::testing::set_check_range(0, 0);
    m_started = true;
    taa::initialize();
    if (!webgpu::set_gbuffer(true)) {
      m_error = "the base G-buffer tier needs a core device";
      return false;
    }
    return true;
  }

  void TearDown() override {
    if (m_started) {
      temporal_resolve::install(nullptr);
      g_stubReady = false;
      motion::testing::reset();
      records::testing::reset();
      testing::reset_hints();
      webgpu::set_gbuffer(false);
      shaders::shutdown();
      jobs::shutdown();
      reset_module_state();
      gpu_heap::shutdown();
      FakeStorage::reset();
    }
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    taa::shutdown();
    taa::testing::reset();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  void use_globals(const wgpu::Extent3D& size) {
    webgpu::set_gbuffer(false);
    m_globals.reset();
    m_options.efbSize = size;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
    taa::initialize();
    ASSERT_TRUE(webgpu::set_gbuffer(true));
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
    taa::on_latch(m_settings, graph);
    latch_pass_params();
  }

  static NodeId taa_node(const CompiledGraph& graph) { return graph.find_node(taa::Pass); }

  void wait_ready(const CompiledGraph& graph) const {
    const NodeId node = taa_node(graph);
    ASSERT_NE(node, InvalidNode);
    for (uint32_t i = 0; i < 3; ++i) {
      const uint32_t id = graph.nodes[node].pipelineIds[i];
      ASSERT_NE(id, 0u) << i;
      ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << i;
    }
  }

  std::shared_ptr<CompiledGraph> live_graph(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    const NodeId node = taa_node(*graph);
    EXPECT_NE(node, InvalidNode);
    if (node != InvalidNode) {
      EXPECT_EQ(graph->nodes[node].status, CullReason::None);
      wait_ready(*graph);
    }
    latch(graph.get());
    return graph;
  }

  StageData build(const FrameState& state, uint64_t frameId, const FrameOptions& options = {}) {
    FakeStorage::refuse = options.refuseStorage;
    begin_hint_frame();
    motion::latch(m_settings, true);
    latch(state.graph.get());
    records::testing::set_active(true, true, false);
    if (options.camera) {
      const gx::hints::Sink& sink = *gx::hints::sink();
      sink.camera(sink.userData, main_camera(options.cutEpoch));
    }
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    if (options.invalidate) {
      invalidate_frame_histories();
    }
    StageData post{
        .stage = Stage::PrePost,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = options.marker,
        .flushedAtFinish = options.atFinish,
        .frame = &state,
    };
    build_stage_data(post);
    FakeStorage::refuse = false;
    return post;
  }

  std::optional<std::string> run(const FrameState& state, uint64_t frameId, const FrameOptions& options = {}) {
    const StageData post = build(state, frameId, options);
    m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    if (post.any_enabled()) {
      run_stage(encoder, post);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    return test::pop_validation_error(*m_device);
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
};

class TemporalResolve : public TaaDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
  }
};

TEST_F(TemporalResolve, TaaIsTheDefaultResolver) {
  const temporal_resolve::Resolver& resolver = temporal_resolve::installed();
  EXPECT_EQ(resolver.pass, taa::Pass);
  EXPECT_EQ(resolver.jitterPhases, 8u);
  EXPECT_EQ(resolver.wanted, &taa::wanted);
  EXPECT_EQ(resolver.jitter_ready, &taa::jitter_enabled);
  EXPECT_EQ(temporal_resolve::jitter_phases(), 8u);
  EXPECT_TRUE(resolver.wanted(taa_settings()));
  EXPECT_FALSE(resolver.wanted(Settings{}));
  temporal_resolve::install(&StubResolver);
  EXPECT_EQ(temporal_resolve::installed().pass, "test.upscaler");
  temporal_resolve::install(nullptr);
  EXPECT_EQ(&temporal_resolve::installed(), &resolver);
}

TEST_F(TemporalResolve, InstalledResolverOwnsJitterAndSkipsTaa) {
  temporal_resolve::install(&StubResolver);
  const auto skipped = compile(taa_settings());
  const NodeId node = taa_node(*skipped);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(skipped->nodes[node].status, CullReason::Skipped);
  EXPECT_TRUE(skipped->histories.empty());
  latch(skipped.get());
  g_stubReady = false;
  EXPECT_FALSE(temporal_resolve::jitter_enabled());
  g_stubReady = true;
  EXPECT_TRUE(temporal_resolve::jitter_enabled());
  EXPECT_FALSE(taa::jitter_enabled());

  temporal_resolve::install(nullptr);
  const auto graph = live_graph(taa_settings());
  EXPECT_EQ(graph->nodes[taa_node(*graph)].status, CullReason::None);
  EXPECT_TRUE(temporal_resolve::jitter_enabled());
  g_stubReady = false;
  EXPECT_TRUE(temporal_resolve::jitter_enabled());
  EXPECT_EQ(temporal_resolve::jitter_enabled(), taa::jitter_enabled());
}

TEST_F(TemporalResolve, JitterPhasesFollowResolver) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::On);
  const uint32_t frame = gfx::current_frame();
  begin_hint_frame();
  motion::latch(Settings{}, true);
  EXPECT_EQ(motion::frame_jitter(), motion::detail::halton_jitter(frame % 8 + 1));
  for (uint32_t f = 0; f < 64; ++f) {
    const uint32_t phase = f % temporal_resolve::jitter_phases() + 1;
    EXPECT_GE(phase, 1u);
    EXPECT_LE(phase, 8u);
  }

  temporal_resolve::install(&StubResolver);
  EXPECT_EQ(temporal_resolve::jitter_phases(), 16u);
  begin_hint_frame();
  motion::latch(Settings{}, true);
  EXPECT_EQ(motion::frame_jitter(), motion::detail::halton_jitter(frame % 16 + 1));
  bool sawPastEight = false;
  for (uint32_t f = 0; f < 64; ++f) {
    const uint32_t phase = f % temporal_resolve::jitter_phases() + 1;
    EXPECT_GE(phase, 1u);
    EXPECT_LE(phase, 16u);
    sawPastEight = sawPastEight || phase > 8;
  }
  EXPECT_TRUE(sawPastEight);
  if (frame % 16 >= 8) {
    EXPECT_NE(motion::detail::halton_jitter(frame % 16 + 1), motion::detail::halton_jitter(frame % 8 + 1));
  }
}

TEST_F(TemporalResolve, WantedDrivesJitterLatch) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::Unset);
  const auto latch_jitter = [](const Settings& settings) {
    begin_hint_frame();
    motion::latch(settings, true);
    return motion::testing::jitter_active();
  };
  EXPECT_FALSE(latch_jitter(Settings{}));
  EXPECT_TRUE(latch_jitter(taa_settings()));

  temporal_resolve::install(&StubResolver);
  EXPECT_TRUE(latch_jitter(Settings{}));

  temporal_resolve::install(&UnwantedResolver);
  EXPECT_FALSE(latch_jitter(taa_settings()));

  temporal_resolve::install(nullptr);
  EXPECT_TRUE(latch_jitter(taa_settings()));
  motion::testing::set_jitter_env(motion::testing::JitterEnv::Off);
  EXPECT_FALSE(latch_jitter(taa_settings()));
}

class TaaGraph : public TaaDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
  }

  void expect_skipped(const Settings& settings) {
    const auto graph = compile(settings);
    const NodeId node = taa_node(*graph);
    ASSERT_NE(node, InvalidNode);
    EXPECT_EQ(graph->nodes[node].status, CullReason::Skipped);
    EXPECT_EQ(graph->nodes[node].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
    EXPECT_TRUE(graph->histories.empty());
    EXPECT_EQ(graph->find_resource(taa::History), InvalidResource);
    EXPECT_TRUE(graph->stage_empty(Stage::PrePost));
    latch(graph.get());
    EXPECT_FALSE(taa::jitter_enabled());
    EXPECT_EQ(taa::ready(settings), !settings.taa);
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, 1), std::nullopt);
    EXPECT_EQ(taa::testing::counters().resolves, 0u);
    EXPECT_EQ(taa::testing::counters().clearOnly, 0u);
    EXPECT_EQ(history_reset::stats().frames, 0u);
  }

  void encode_sizes() {
    for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{61u, 37u}, std::pair{1280u, 720u}}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      ASSERT_NO_FATAL_FAILURE(use_globals({width, height, 1}));
      taa::testing::reset();
      const auto graph = live_graph(taa_settings(0.5f));
      const FrameState state{.graph = graph};
      EXPECT_EQ(run(state, 1), std::nullopt);
      EXPECT_EQ(taa::testing::counters().resolves, 1u);
      EXPECT_EQ(taa::testing::counters().resets, 1u);
      EXPECT_EQ(taa::testing::counters().groupsCreated, 2u);
      for (uint64_t frame = 2; frame <= 4; ++frame) {
        EXPECT_EQ(run(state, frame), std::nullopt);
      }
      EXPECT_EQ(taa::testing::counters().resolves, 4u);
      EXPECT_EQ(taa::testing::counters().resets, 1u);
      EXPECT_EQ(taa::testing::counters().groupsCreated, 5u);
      EXPECT_EQ(taa::testing::counters().clearOnly, 0u);
      EXPECT_EQ(taa::testing::counters().skippedNoScene, 0u);
    }
  }
};

TEST_F(TaaGraph, SkippedWhenOff) { expect_skipped(Settings{}); }

TEST_F(TaaGraph, SkippedWithoutMotion) {
  ASSERT_TRUE(webgpu::set_gbuffer(false));
  expect_skipped(taa_settings());
}

TEST_F(TaaGraph, SkippedUnderDebugView) {
  Settings settings = taa_settings();
  settings.debugView = 16;
  expect_skipped(settings);
}

TEST_F(TaaGraph, DisabledWithoutMarkerCameraOrAtFinish) {
  const auto graph = live_graph(taa_settings());
  const FrameState state{.graph = graph};
  const uint16_t index = graph->nodes[taa_node(*graph)].programIndex;
  EXPECT_TRUE(build(state, 1).is_enabled(index));
  EXPECT_FALSE(build(state, 2, {.marker = false}).is_enabled(index));
  EXPECT_FALSE(build(state, 3, {.atFinish = true}).is_enabled(index));
  EXPECT_FALSE(build(state, 4, {.camera = false}).is_enabled(index));
  EXPECT_TRUE(build(state, 5).is_enabled(index));
  EXPECT_EQ(taa::testing::counters().skippedNoScene, 3u);
}

TEST_F(TaaGraph, BindingOrderDepthMotionReactive) {
  const auto graph = compile(taa_settings());
  const NodeId node = taa_node(*graph);
  ASSERT_NE(node, InvalidNode);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_EQ(graph->nodes[node].stage, Stage::PrePost);
  EXPECT_EQ(graph->nodes[node].passKind, PassKind::Encoder);
  const BindingPlan& plan = graph->bindingPlans[node];
  ASSERT_EQ(plan.slots.size(), 3u);
  const std::array<std::string_view, 3> names{resource::GxDepth, resource::Motion, resource::Reactive};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(plan.slots[i].binding, i);
    EXPECT_EQ(plan.slots[i].type, BindingType::SampledTexture);
    EXPECT_EQ(graph->resources[plan.slots[i].resource].name, names[i]) << i;
  }
  EXPECT_EQ(plan.slots[0].sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(plan.paramsSize, sizeof(taa::Params));
  const ResourceId history = graph->find_resource(taa::History);
  ASSERT_NE(history, InvalidResource);
  const CompiledResource& resource = graph->resources[history];
  EXPECT_EQ(resource.kind, ResourceKind::History);
  EXPECT_EQ(resource.tex.format, TextureFormat::RGBA16Float);
  EXPECT_TRUE(resource.textureUsage & wgpu::TextureUsage::TextureBinding);
  EXPECT_TRUE(resource.textureUsage & wgpu::TextureUsage::RenderAttachment);
  EXPECT_EQ(resource.extent.width, m_options.efbSize.width);
  EXPECT_EQ(resource.extent.height, m_options.efbSize.height);
}

TEST_F(TaaGraph, EncodesValidationClean) { encode_sizes(); }

TEST_F(TaaGraph, EncodesValidationCleanWithProductionToggles) {
  TearDown();
  m_started = false;
  if (!open({.backend = wgpu::BackendType::Null, .productionToggles = true})) {
    GTEST_SKIP() << "no null device: " << m_error;
  }
  encode_sizes();
}

TEST_F(TaaGraph, PendingPipelinesKeepJitterOff) {
  motion::testing::set_jitter_env(motion::testing::JitterEnv::Unset);
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
  const auto graph = compile(taa_settings());
  const NodeId node = taa_node(*graph);
  ASSERT_NE(node, InvalidNode);
  ASSERT_NE(graph->nodes[node].pipelineIds[0], 0u);
  EXPECT_EQ(shaders::status({graph->nodes[node].pipelineIds[0]}), shaders::PipelineStatus::Pending);
  latch(graph.get());
  EXPECT_FALSE(taa::jitter_enabled());
  EXPECT_FALSE(temporal_resolve::jitter_enabled());
  EXPECT_FALSE(taa::ready(m_settings));
  EXPECT_TRUE(taa::ready(Settings{}));
  const FrameState state{.graph = graph};
  for (uint64_t frame = 1; frame <= 2; ++frame) {
    const StageData stage = build(state, frame);
    EXPECT_FALSE(stage.is_enabled(graph->nodes[node].programIndex));
    ASSERT_NE(stage.sceneHints, nullptr);
    EXPECT_EQ(stage.sceneHints->jitterPx[0], 0.f);
    EXPECT_EQ(stage.sceneHints->jitterPx[1], 0.f);
  }
  release = true;
  gate.wait();
  wait_ready(*graph);
  latch(graph.get());
  EXPECT_TRUE(taa::jitter_enabled());
  EXPECT_TRUE(taa::ready(m_settings));
  const StageData stage = build(state, 3);
  EXPECT_TRUE(stage.is_enabled(graph->nodes[node].programIndex));
  ASSERT_TRUE(stage.sceneHints->motionValid);
  EXPECT_TRUE(stage.sceneHints->jitterPx[0] != 0.f || stage.sceneHints->jitterPx[1] != 0.f);
}

TEST_F(TaaGraph, InvalidateForcesReset) {
  const auto graph = live_graph(taa_settings());
  const FrameState state{.graph = graph};
  const std::array<std::pair<FrameOptions, uint32_t>, 6> frames{{
      {{}, 1},
      {{}, 1},
      {{.invalidate = true}, 2},
      {{}, 2},
      {{.cutEpoch = 1}, 3},
      {{.cutEpoch = 1}, 3},
  }};
  uint64_t frame = 0;
  for (const auto& [options, resets] : frames) {
    ++frame;
    EXPECT_EQ(run(state, frame, options), std::nullopt);
    EXPECT_EQ(taa::testing::counters().resolves, frame);
    EXPECT_EQ(taa::testing::counters().resets, resets) << "frame " << frame;
  }
  EXPECT_EQ(history_reset::stats().reasons[static_cast<size_t>(history_reset::Reason::CutEpoch)], 1u);
}

TEST_F(TaaGraph, InvalidMotionForcesReset) {
  const auto graph = live_graph(taa_settings());
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1), std::nullopt);
  EXPECT_EQ(run(state, 2), std::nullopt);
  ASSERT_EQ(taa::testing::counters().resets, 1u);
  const uint16_t slot = graph->resources[graph->find_resource(taa::History)].historySlot;
  const StageData stage = build(state, 3, {.refuseStorage = true});
  ASSERT_NE(stage.sceneHints, nullptr);
  EXPECT_FALSE(stage.sceneHints->motionValid);
  EXPECT_EQ(stage.sceneHints->motionReason, MotionReason::StorageFull);
  EXPECT_TRUE(stage.prev_valid(slot));
  const uint32_t groups = taa::testing::counters().groupsCreated;
  m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation);
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  run_stage(encoder, stage);
  const wgpu::CommandBuffer commands = encoder.Finish();
  EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  EXPECT_EQ(test::pop_validation_error(*m_device), std::nullopt);
  EXPECT_EQ(taa::testing::counters().resolves, 3u);
  EXPECT_EQ(taa::testing::counters().resets, 2u);
  EXPECT_EQ(taa::testing::counters().groupsCreated, groups);
  EXPECT_EQ(run(state, 4), std::nullopt);
  EXPECT_EQ(taa::testing::counters().resets, 2u);
}

TEST_F(TaaGraph, HintsOffKeepsJitterOff) {
  const auto graph = live_graph(taa_settings());
  EXPECT_TRUE(taa::jitter_enabled());
  EXPECT_TRUE(taa::ready(m_settings));
  remove_hints();
  latch(graph.get());
  EXPECT_FALSE(taa::jitter_enabled());
  EXPECT_FALSE(temporal_resolve::jitter_enabled());
  EXPECT_FALSE(taa::ready(m_settings));
  EXPECT_TRUE(taa::ready(Settings{}));
}

TEST_F(TaaGraph, ZeroPipelineIdsKeepJitterOff) {
  shaders::quiesce();
  const auto graph = compile(taa_settings());
  const NodeId node = taa_node(*graph);
  ASSERT_NE(node, InvalidNode);
  ASSERT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_EQ(graph->nodes[node].pipelineIds[0], 0u);
  latch(graph.get());
  EXPECT_FALSE(taa::jitter_enabled());
  EXPECT_FALSE(taa::ready(m_settings));
  const FrameState state{.graph = graph};
  EXPECT_FALSE(build(state, 1).is_enabled(graph->nodes[node].programIndex));
}

TEST_F(TaaGraph, ParamsLatchedEveryFrameAndAfterReinit) {
  const auto read_params = [](PassId id) -> std::optional<taa::Params> {
    const std::span<const std::byte> bytes = latched_pass_params(id);
    if (bytes.size() != sizeof(taa::Params)) {
      return std::nullopt;
    }
    taa::Params params;
    std::memcpy(&params, bytes.data(), sizeof(params));
    return params;
  };
  const auto graph = live_graph(taa_settings(0.75f));
  const PassId first = graph->nodes[taa_node(*graph)].id;
  ASSERT_NE(first, InvalidPassId);
  ASSERT_TRUE(read_params(first));
  EXPECT_EQ(read_params(first)->sharpness, 0.75f);

  clear_pass_params(first);
  latch_pass_params();
  EXPECT_FALSE(read_params(first));
  latch(graph.get());
  ASSERT_TRUE(read_params(first));
  EXPECT_EQ(read_params(first)->sharpness, 0.75f);
  m_settings.taaSharpness = 3.f;
  latch(graph.get());
  EXPECT_EQ(read_params(first)->sharpness, 1.f);
  m_settings.taa = false;
  latch(graph.get());
  EXPECT_EQ(read_params(first)->sharpness, 0.f);

  taa::shutdown();
  taa::initialize();
  const auto again = compile(taa_settings(0.5f));
  const PassId second = again->nodes[taa_node(*again)].id;
  ASSERT_NE(second, InvalidPassId);
  EXPECT_NE(second, first);
  latch_pass_params();
  EXPECT_FALSE(read_params(second));
  latch(again.get());
  ASSERT_TRUE(read_params(second));
  EXPECT_EQ(read_params(second)->sharpness, 0.5f);
}

constexpr std::string_view HistoryReadbackWgsl = R"(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(source);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  texels[id.y * size.x + id.x] = textureLoad(source, vec2i(id.xy), 0);
}
)";

struct Rgba {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 0;
};

class TaaGpu : public TaaDeviceTest {
protected:
  static constexpr uint32_t Width = 64;
  static constexpr uint32_t Height = 48;

  void SetUp() override {
    const wgpu::BackendType backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!open({.backend = backend, .efbSize = {Width, Height, 1}})) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    ASSERT_EQ(webgpu::g_frameBuffer.texture.GetFormat(), TextureFormat::BGRA8Unorm);
  }

  std::shared_ptr<CompiledGraph> start(float sharpness = 0.f) {
    m_graph = live_graph(taa_settings(sharpness));
    m_state = std::make_unique<FrameState>(FrameState{.graph = m_graph});
    return m_graph;
  }

  template <typename Fn>
  static std::vector<Rgba> image(Fn&& fn) {
    std::vector<Rgba> out(size_t{Width} * Height);
    for (uint32_t y = 0; y < Height; ++y) {
      for (uint32_t x = 0; x < Width; ++x) {
        out[size_t{y} * Width + x] = fn(x, y);
      }
    }
    return out;
  }

  static std::vector<Rgba> flat(Rgba color) {
    return image([color](uint32_t, uint32_t) { return color; });
  }

  void write_color(const std::vector<Rgba>& pixels) const {
    std::vector<uint8_t> bytes(pixels.size() * 4);
    for (size_t i = 0; i < pixels.size(); ++i) {
      bytes[i * 4 + 0] = pixels[i].b;
      bytes[i * 4 + 1] = pixels[i].g;
      bytes[i * 4 + 2] = pixels[i].r;
      bytes[i * 4 + 3] = pixels[i].a;
    }
    const wgpu::TexelCopyTextureInfo destination{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = Width * 4, .rowsPerImage = Height};
    const wgpu::Extent3D size{Width, Height, 1};
    m_device->queue.WriteTexture(&destination, bytes.data(), bytes.size(), &layout, &size);
  }

  void clear_targets(float motionX, float motionY, float reactive) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const auto clear = [&](const wgpu::TextureView& view, const wgpu::Color& value) {
      const wgpu::RenderPassColorAttachment attachment{
          .view = view,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = value,
      };
      const wgpu::RenderPassDescriptor descriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
      encoder.BeginRenderPass(&descriptor).End();
    };
    clear(webgpu::g_motionBuffer.view, {motionX, motionY, 0.0, 0.0});
    clear(webgpu::g_reactiveBuffer.view, {reactive, 0.0, 0.0, 0.0});
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  }

  std::vector<Rgba> read_color() const {
    const uint32_t row = (Width * 4 + 255) & ~255u;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{row} * Height};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo source{.texture = webgpu::g_frameBuffer.texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = row, .rowsPerImage = Height},
                                                .buffer = buffer};
    const wgpu::Extent3D size{Width, Height, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &size);
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, uint64_t{row} * Height);
    std::vector<Rgba> out(size_t{Width} * Height);
    if (!words) {
      ADD_FAILURE() << "colour readback failed";
      return out;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(words->data());
    for (uint32_t y = 0; y < Height; ++y) {
      for (uint32_t x = 0; x < Width; ++x) {
        const uint8_t* p = bytes + size_t{y} * row + size_t{x} * 4;
        out[size_t{y} * Width + x] = {.r = p[2], .g = p[1], .b = p[0], .a = p[3]};
      }
    }
    return out;
  }

  std::vector<std::array<float, 4>> read_history() {
    const ResourceId resource = m_graph->find_resource(taa::History);
    std::vector<std::array<float, 4>> out(size_t{Width} * Height);
    if (resource == InvalidResource) {
      ADD_FAILURE() << "no history resource";
      return out;
    }
    const wgpu::Texture texture = transient_pool().history_write(m_graph->resources[resource].historySlot).texture;
    const wgpu::Device& device = m_device->device;
    if (!m_readback) {
      const std::string code{HistoryReadbackWgsl};
      const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
      const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
      const wgpu::ComputePipelineDescriptor descriptor{
          .compute = {.module = device.CreateShaderModule(&moduleDescriptor), .entryPoint = "main"},
      };
      m_readback = device.CreateComputePipeline(&descriptor);
    }
    const uint64_t bytes = out.size() * sizeof(float) * 4;
    const wgpu::BufferDescriptor bufferDescriptor{.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                                  .size = bytes};
    const wgpu::Buffer buffer = device.CreateBuffer(&bufferDescriptor);
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = texture.CreateView()},
        wgpu::BindGroupEntry{.binding = 1, .buffer = buffer},
    };
    const wgpu::BindGroupDescriptor groupDescriptor{
        .layout = m_readback.GetBindGroupLayout(0), .entryCount = entries.size(), .entries = entries.data()};
    const wgpu::BindGroup group = device.CreateBindGroup(&groupDescriptor);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(m_readback);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups((Width + 7) / 8, (Height + 7) / 8);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
    const auto words = test::read_buffer(*m_device, buffer, 0, bytes);
    if (!words) {
      ADD_FAILURE() << "history readback failed";
      return out;
    }
    std::memcpy(out.data(), words->data(), bytes);
    return out;
  }

  std::vector<Rgba> frame(const std::vector<Rgba>& input, float motionX = 0.f, float motionY = 0.f,
                          float reactive = 0.f) {
    write_color(input);
    clear_targets(motionX, motionY, reactive);
    ++m_frame;
    EXPECT_EQ(run(*m_state, m_frame), std::nullopt);
    return read_color();
  }

  static int max_rgb_error(const std::vector<Rgba>& got, const std::vector<Rgba>& want,
                           const std::function<bool(uint32_t, uint32_t)>& include = {}) {
    int worst = 0;
    for (uint32_t y = 0; y < Height; ++y) {
      for (uint32_t x = 0; x < Width; ++x) {
        if (include && !include(x, y)) {
          continue;
        }
        const Rgba& a = got[size_t{y} * Width + x];
        const Rgba& b = want[size_t{y} * Width + x];
        worst = std::max({worst, std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
      }
    }
    return worst;
  }

  std::shared_ptr<CompiledGraph> m_graph;
  std::unique_ptr<FrameState> m_state;
  uint64_t m_frame = 0;
  wgpu::ComputePipeline m_readback;
};

std::vector<Rgba> random_image(uint32_t seed) {
  std::mt19937 rng{seed};
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<Rgba> out(size_t{64} * 48);
  for (Rgba& pixel : out) {
    pixel = {static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)), static_cast<uint8_t>(byte(rng)),
             static_cast<uint8_t>(byte(rng))};
  }
  return out;
}

TEST_F(TaaGpu, ResetCopiesInputExactly) {
  start();
  const std::vector<Rgba> input = random_image(1);
  const std::vector<Rgba> output = frame(input);
  EXPECT_EQ(taa::testing::counters().resets, 1u);
  EXPECT_EQ(max_rgb_error(output, input), 0);
  const auto history = read_history();
  for (size_t i = 0; i < history.size(); ++i) {
    EXPECT_EQ(history[i][3], 1.f) << i;
    EXPECT_NEAR(history[i][0], input[i].r / 255.f, 1e-3f) << i;
  }
}

TEST_F(TaaGpu, StaticInputConvergesWithinOneLsb) {
  start();
  const std::vector<Rgba> input = image([](uint32_t x, uint32_t y) {
    const double wave = 40.0 * std::sin(x * 0.11) * std::cos(y * 0.07);
    return Rgba{static_cast<uint8_t>(30 + x * 3), static_cast<uint8_t>(20 + y * 4),
                static_cast<uint8_t>(std::lround(128.0 + wave)), 200};
  });
  std::vector<Rgba> output;
  for (int i = 0; i < 30; ++i) {
    output = frame(input);
  }
  EXPECT_EQ(taa::testing::counters().resets, 1u);
  EXPECT_LE(max_rgb_error(output, input), 1);
  const auto history = read_history();
  EXPECT_GT(history[size_t{24} * Width + 32][3], taa::MovingSamples);
}

TEST_F(TaaGpu, LeavesEfbAlphaUntouched) {
  for (const float sharpness : {0.f, 1.f}) {
    SCOPED_TRACE(sharpness);
    if (m_graph) {
      reset_module_state();
      taa::initialize();
      m_frame = 0;
    }
    start(sharpness);
    for (uint32_t seed = 10; seed < 14; ++seed) {
      const std::vector<Rgba> input = random_image(seed);
      const std::vector<Rgba> output = frame(input, 0.001f * static_cast<float>(seed - 10), 0.f, 0.f);
      size_t changed = 0;
      for (size_t i = 0; i < input.size(); ++i) {
        changed += output[i].a != input[i].a ? 1 : 0;
      }
      EXPECT_EQ(changed, 0u) << "seed " << seed;
    }
  }
}

TEST_F(TaaGpu, ReactiveOneTakesCurrent) {
  start(0.f);
  const std::vector<Rgba> first = flat({200, 40, 90, 255});
  for (int i = 0; i < 8; ++i) {
    frame(first);
  }
  const std::vector<Rgba> second = random_image(7);
  EXPECT_EQ(max_rgb_error(frame(second, 0.f, 0.f, 1.f), second), 0);
  const auto history = read_history();
  EXPECT_EQ(history[0][3], 1.f);
}

TEST_F(TaaGpu, OffscreenMotionRejectsHistory) {
  start(0.f);
  const std::vector<Rgba> first = random_image(3);
  for (int i = 0; i < 6; ++i) {
    frame(first);
  }
  const std::vector<Rgba> second = random_image(4);
  EXPECT_EQ(max_rgb_error(frame(second, 4.f, 4.f), second), 0);
  for (int i = 0; i < 4; ++i) {
    frame(first);
  }
  const std::vector<Rgba> output = frame(second, 1.5f, 0.f);
  EXPECT_EQ(max_rgb_error(output, second, [](uint32_t x, uint32_t) { return x + 1 < Width * 3 / 4; }), 0);
}

TEST_F(TaaGpu, BorderPixelsStayExact) {
  const std::vector<Rgba> input = image([](uint32_t x, uint32_t y) {
    const bool border = x == 0 || y == 0 || x == Width - 1 || y == Height - 1;
    return border ? Rgba{255, 255, 255, 17} : Rgba{0, 0, 0, 33};
  });
  for (const float sharpness : {0.f, 1.f}) {
    SCOPED_TRACE(sharpness);
    if (m_graph) {
      reset_module_state();
      taa::initialize();
      m_frame = 0;
    }
    start(sharpness);
    std::vector<Rgba> output;
    for (int i = 0; i < 10; ++i) {
      output = frame(input);
    }
    EXPECT_EQ(
        max_rgb_error(output, input,
                      [](uint32_t x, uint32_t y) { return x == 0 || y == 0 || x == Width - 1 || y == Height - 1; }),
        0);
  }
}

TEST_F(TaaGpu, StepChangeUnderZeroMotionIsClipped) {
  start(0.f);
  const std::vector<Rgba> first = flat({30, 60, 90, 255});
  for (int i = 0; i < 12; ++i) {
    frame(first);
  }
  const std::vector<Rgba> second = flat({220, 180, 20, 255});
  EXPECT_LE(max_rgb_error(frame(second), second), 1);
  EXPECT_EQ(taa::testing::counters().resets, 1u);
}

TEST_F(TaaGpu, MotionCapsSampleCount) {
  start(0.f);
  const std::vector<Rgba> input = flat({120, 120, 120, 255});
  const float moving = 0.1f / static_cast<float>(Width);
  for (int i = 0; i < 20; ++i) {
    frame(input, moving, 0.f);
  }
  float most = 0.f;
  for (const auto& texel : read_history()) {
    most = std::max(most, texel[3]);
  }
  EXPECT_EQ(most, taa::MovingSamples);
  for (int i = 0; i < 20; ++i) {
    frame(input);
  }
  EXPECT_GT(read_history()[size_t{24} * Width + 32][3], taa::MovingSamples);
}

TEST_F(TaaGpu, ResetConvergesToEightPhaseMean) {
  start(0.f);
  std::array<uint8_t, 8> values{};
  double sum = 0.0;
  for (size_t k = 0; k < values.size(); ++k) {
    values[k] = static_cast<uint8_t>(102 + 7 * ((k * 5) % 8));
    sum += values[k];
  }
  const double mean = sum / static_cast<double>(values.size());
  std::vector<Rgba> output;
  for (const uint8_t value : values) {
    output = frame(image([value](uint32_t x, uint32_t y) {
      if ((x + y) % 2 == 0) {
        return Rgba{value, value, value, 255};
      }
      const uint8_t edge = y % 2 == 0 ? 255 : 0;
      return Rgba{edge, edge, edge, 255};
    }));
  }
  EXPECT_EQ(taa::testing::counters().resets, 1u);
  const auto history = read_history();
  double worst = 0.0;
  for (uint32_t y = 2; y + 2 < Height; ++y) {
    for (uint32_t x = 2; x + 2 < Width; ++x) {
      if ((x + y) % 2 != 0) {
        continue;
      }
      worst = std::max(worst, std::abs(output[size_t{y} * Width + x].r - mean));
      EXPECT_EQ(history[size_t{y} * Width + x][3], 8.f) << x << "," << y;
    }
  }
  EXPECT_LE(worst, 1.0);
}
} // namespace
} // namespace aurora::render::detail

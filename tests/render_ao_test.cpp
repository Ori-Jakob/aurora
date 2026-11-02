#include <gtest/gtest.h>

#include "render_ao_reference.hpp"
#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/ao/ao.hpp"
#include "render/ao/hiz.hpp"
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
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "render/stage_camera.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace aurora::render::detail {
namespace {
namespace ref = aurora::test::ao;
using aurora::test::scene_depth;

constexpr uint64_t PipelineWaitNs = 30'000'000'000;
constexpr float Near = 10.f;
constexpr float Far = 100000.f;

Mat4x4<float> perspective(float n, float f, float p02 = 0.f, float p12 = 0.f) {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m0[2] = p02;
  proj.m1[1] = 1.7320508f;
  proj.m1[2] = p12;
  proj.m2[2] = std::isinf(f) ? 0.f : -n / (f - n);
  proj.m2[3] = std::isinf(f) ? -n : -(f * n) / (f - n);
  proj.m3[2] = -1.f;
  return proj;
}

std::shared_ptr<const SceneHints> camera_hints(const Mat4x4<float>& proj, std::array<float, 2> jitter = {},
                                               std::optional<float> unitsPerMetre = std::nullopt,
                                               GXProjectionType type = GX_PERSPECTIVE) {
  auto hints = std::make_shared<SceneHints>();
  hints->mainCamera = {
      .role = AURORA_CAMERA_MAIN,
      .proj = proj,
      .type = type,
      .gxProj = gx::hints::gx_projection(proj, type),
  };
  hints->present = SceneHasMainCamera;
  hints->jitterPx[0] = jitter[0];
  hints->jitterPx[1] = jitter[1];
  if (unitsPerMetre) {
    hints->environment.value.unitsPerMeter = *unitsPerMetre;
    hints->present |= SceneHasEnvironment;
  }
  return hints;
}

Settings ao_settings(Quality quality) {
  Settings settings;
  settings.ao = quality;
  return settings;
}

ao::Params params_from(std::span<const std::byte> bytes) {
  ao::Params params;
  EXPECT_GE(bytes.size(), sizeof(params));
  std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(params)));
  return params;
}

uint32_t chain_levels(uint32_t width, uint32_t height) {
  return static_cast<uint32_t>(std::bit_width(std::max(width, height)));
}

TEST(AoRules, ParamsFitPassParams) {
  static_assert(sizeof(ao::Params) == 112);
  EXPECT_LE(sizeof(ao::Params), MaxPassParamsSize);
  EXPECT_EQ(offsetof(ao::Params, camera), 0u);
  EXPECT_EQ(offsetof(ao::Params, effectRadius), 48u);
  EXPECT_EQ(offsetof(ao::Params, sliceCount), 72u);
  EXPECT_EQ(offsetof(ao::Params, flags), 84u);
  EXPECT_EQ(offsetof(ao::Params, minVisibility), 108u);
  const PassDesc desc = ao::hiz::pass_desc();
  EXPECT_EQ(desc.paramsSize, sizeof(ao::Params));
  EXPECT_EQ(desc.stage, Stage::OpaqueDone);
  EXPECT_EQ(desc.kind, PassKind::Compute);
  EXPECT_EQ(desc.order, 1u);
  EXPECT_FALSE(desc.runsWithoutMarker);
  EXPECT_EQ(desc.enabled, static_cast<PassEnabledFn>(&ao::scene_ready));
  EXPECT_EQ(desc.stageParams, static_cast<PassStageParamsFn>(&ao::fill_camera));
}

TEST(AoRules, StageCameraLayoutIs48) {
  static_assert(sizeof(StageCamera) == 48);
  EXPECT_EQ(offsetof(StageCamera, depthMul), 0u);
  EXPECT_EQ(offsetof(StageCamera, depthAdd), 4u);
  EXPECT_EQ(offsetof(StageCamera, p00), 8u);
  EXPECT_EQ(offsetof(StageCamera, p11), 12u);
  EXPECT_EQ(offsetof(StageCamera, p02), 16u);
  EXPECT_EQ(offsetof(StageCamera, p12), 20u);
  EXPECT_EQ(offsetof(StageCamera, jitterPx), 24u);
  EXPECT_EQ(offsetof(StageCamera, unitsPerMeter), 32u);
  EXPECT_EQ(offsetof(StageCamera, flags), 36u);
}

TEST(AoRules, LinearizeMatchesNearFar) {
  const Mat4x4<float> proj = perspective(Near, Far);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(camera_hints(proj).get(), camera));
  EXPECT_EQ(camera.flags, CameraValid);
  EXPECT_EQ(camera.depthMul, proj.m2[3]);
  EXPECT_EQ(camera.depthAdd, proj.m2[2]);
  for (const double distance : {10.0, 37.5, 100.0, 1000.0, 25000.0}) {
    const auto stored = static_cast<float>(double{proj.m2[2]} - double{proj.m2[3]} / distance);
    const double z = ref::view_depth(camera, stored);
    EXPECT_NEAR(z / distance, 1.0, 1e-4) << distance;
  }
  EXPECT_NEAR(ref::view_depth(camera, 1.f), Near, 1e-4);
  EXPECT_EQ(ref::view_depth(camera, 0.f), double{ao::SkyDepth});
  EXPECT_EQ(ref::view_depth(camera, -0.f), double{ao::SkyDepth});
}

TEST(AoRules, JitterFoldsIntoOffCentreTerms) {
  constexpr double Width = 1280.0;
  constexpr double Height = 720.0;
  const Mat4x4<float> proj = perspective(Near, Far, 0.1f, -0.05f);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(camera_hints(proj, {0.3f, -0.4f}).get(), camera));
  EXPECT_EQ(camera.p02, 0.1f);
  EXPECT_EQ(camera.p12, -0.05f);
  EXPECT_EQ(camera.jitterPx[0], 0.3f);
  EXPECT_EQ(camera.jitterPx[1], -0.4f);
  for (const std::array<double, 3> point :
       {std::array{12.0, -7.0, 50.0}, std::array{-300.0, 120.0, 900.0}, std::array{0.0, 0.0, 10.0}}) {
    const double w = point[2];
    const double clipX = double{proj.m0[0]} * point[0] - double{proj.m0[2]} * w;
    const double clipY = double{proj.m1[1]} * point[1] - double{proj.m1[2]} * w;
    const double ndcX = clipX / w + 2.0 * camera.jitterPx[0] / Width;
    const double ndcY = clipY / w - 2.0 * camera.jitterPx[1] / Height;
    const double u = (ndcX + 1.0) * 0.5;
    const double v = (1.0 - ndcY) * 0.5;
    const std::array<double, 3> got = ref::view_position_fwd(camera, u, v, w, Width, Height);
    EXPECT_NEAR(got[0], point[0], 1e-6 * w);
    EXPECT_NEAR(got[1], point[1], 1e-6 * w);
    EXPECT_EQ(got[2], point[2]);
  }
}

TEST(AoRules, UnitsPerMeterDefaultsTo100) {
  SceneHints hints;
  EXPECT_EQ(units_per_metre(hints), 100.f);
  hints.environment.value.unitsPerMeter = 50.f;
  EXPECT_EQ(units_per_metre(hints), 100.f);
  hints.present |= SceneHasEnvironment;
  EXPECT_EQ(units_per_metre(hints), 50.f);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(camera_hints(perspective(Near, Far)).get(), camera));
  EXPECT_EQ(camera.unitsPerMeter, 100.f);
}

TEST(AoRules, NonFiniteUnitsFallBack) {
  SceneHints hints;
  hints.present = SceneHasEnvironment;
  for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(), 0.f, -5.f, -0.f}) {
    hints.environment.value.unitsPerMeter = value;
    EXPECT_EQ(units_per_metre(hints), 100.f) << value;
  }
}

TEST(AoRules, EnvironmentUnitsScaleRadius) {
  ao::testing::reset();
  const ao::Params latched;
  for (const auto [upm, expected] : {std::pair{std::optional<float>{}, 100.f}, std::pair{std::optional{50.f}, 50.f}}) {
    std::array<std::byte, sizeof(ao::Params)> bytes{};
    std::memcpy(bytes.data(), &latched, sizeof(latched));
    const StageData stage{.frameIndex = 70, .sceneHints = camera_hints(perspective(Near, Far), {}, upm)};
    ao::fill_camera(stage, nullptr, bytes);
    const ao::Params filled = params_from(bytes);
    EXPECT_EQ(filled.camera.unitsPerMeter, expected);
    EXPECT_EQ(filled.effectRadius, ao::RadiusMetres * expected);
    EXPECT_EQ(filled.thickness, ao::ThicknessMetres * expected);
    EXPECT_EQ(filled.noiseIndex, 70u % ao::NoisePeriod);
    EXPECT_EQ(filled.sliceCount, latched.sliceCount);
    EXPECT_EQ(filled.radiusMetres, ao::RadiusMetres);
  }
  ao::testing::reset();
}

TEST(AoRules, QualityTableMatchesXeGtao) {
  const auto row = [](Quality quality) { return ao::quality_row(quality); };
  EXPECT_EQ(row(Quality::Low).slices, 1u);
  EXPECT_EQ(row(Quality::Low).steps, 2u);
  EXPECT_EQ(row(Quality::Medium).slices, 2u);
  EXPECT_EQ(row(Quality::Medium).steps, 2u);
  EXPECT_EQ(row(Quality::High).slices, 3u);
  EXPECT_EQ(row(Quality::High).steps, 3u);
  EXPECT_EQ(row(Quality::Ultra).slices, row(Quality::High).slices);
  EXPECT_EQ(row(Quality::Ultra).steps, row(Quality::High).steps);
  EXPECT_EQ(row(Quality::Ultra).divisor, 1u);
  EXPECT_EQ(ao::BuiltOptionMask, 0b11110u);
  EXPECT_EQ(ao::MipLevels, 5u);
  EXPECT_EQ(ao::FalloffRange, 0.615f);
  EXPECT_EQ(ao::MipOffset, 3.3f);
  EXPECT_EQ(ao::DistributionPower, 2.f);
  EXPECT_NEAR(ao::RadiusMetres, 0.5f * 1.457f, 0.005f);
}

TEST(AoRules, WorkingDivisorPolicy) {
  for (const Quality quality : {Quality::Low, Quality::Medium, Quality::High}) {
    EXPECT_EQ(ao::working_divisor(quality, 0), 2u);
    EXPECT_EQ(ao::working_divisor(quality, 4), 2u);
  }
  EXPECT_EQ(ao::working_divisor(Quality::Ultra, 0), 1u);
  EXPECT_EQ(ao::working_divisor(Quality::Ultra, 4), 1u);
  for (const Quality quality : {Quality::Low, Quality::Medium, Quality::High, Quality::Ultra}) {
    EXPECT_EQ(ao::working_divisor(quality, 1), 1u);
    EXPECT_EQ(ao::working_divisor(quality, 2), 2u);
    if (std::getenv("AURORA_RENDER_AO_DIVISOR") == nullptr) {
      EXPECT_EQ(ao::working_divisor(quality), ao::quality_row(quality).divisor);
    }
  }
  EXPECT_EQ(ao::parse_divisor("1"), 1u);
  EXPECT_EQ(ao::parse_divisor("2"), 2u);
  EXPECT_EQ(ao::parse_divisor("4"), std::nullopt);
  EXPECT_EQ(ao::parse_divisor("half"), std::nullopt);
}

TEST(AoRules, DivisorFromDims) {
  for (const uint32_t divisor : {1u, 2u}) {
    for (uint32_t full = 2; full <= 300; ++full) {
      EXPECT_EQ(ao::divisor_from_dims(full, ao::grid_size(full, divisor)), divisor) << full;
    }
    for (const uint32_t full : {1440u, 1593u, 1792u, 2560u, 2688u, 3186u, 4779u}) {
      EXPECT_EQ(ao::divisor_from_dims(full, ao::grid_size(full, divisor)), divisor) << full;
    }
  }
  EXPECT_EQ(ao::grid_size(4779, 2), 2390u);
  EXPECT_EQ(ao::grid_size(2688, 2), 1344u);
  EXPECT_EQ(ao::divisor_from_dims(1, 1), 1u);
  EXPECT_EQ(ao::divisor_from_dims(5, 0), 5u);
}

TEST(AoRules, RepresentativeTexelClamped) {
  for (const uint32_t divisor : {1u, 2u}) {
    for (uint32_t full = 1; full <= 80; ++full) {
      for (uint32_t p = 0; p < ao::grid_size(full, divisor); ++p) {
        const uint32_t q = ao::rep_texel(p, divisor, full);
        EXPECT_LT(q, full);
        EXPECT_EQ(q, std::min(divisor * p + divisor / 2, full - 1));
        EXPECT_GE(q + 1, std::min(divisor * p + 1, full));
      }
    }
  }
  EXPECT_EQ(ao::rep_texel(2389, 2, 4779), 4778u);
  EXPECT_EQ(ao::rep_texel(0, 2, 1), 0u);
}

TEST(AoRules, DevBuiltOnlyWithEnv) {
  const BuiltOptions& base = default_built_options();
  EXPECT_EQ(ao::with_dev_built(base, std::nullopt), base);
  EXPECT_EQ(ao::with_dev_built(base, Quality::Off), base);
  for (const Quality quality : {Quality::Low, Quality::High, Quality::Ultra}) {
    const BuiltOptions built = ao::with_dev_built(base, quality);
    for (size_t i = 0; i < built.size(); ++i) {
      const uint32_t want = static_cast<Feature>(i) == Feature::Ao ? base[i] | ao::BuiltOptionMask : base[i];
      EXPECT_EQ(built[i], want) << feature_name(static_cast<Feature>(i));
    }
  }
  EXPECT_EQ(ao::parse_quality("off"), Quality::Off);
  EXPECT_EQ(ao::parse_quality("low"), Quality::Low);
  EXPECT_EQ(ao::parse_quality("medium"), Quality::Medium);
  EXPECT_EQ(ao::parse_quality("high"), Quality::High);
  EXPECT_EQ(ao::parse_quality("ultra"), Quality::Ultra);
  EXPECT_EQ(ao::parse_quality("High"), std::nullopt);
  EXPECT_EQ(ao::parse_quality("3"), std::nullopt);
}

TEST(AoRules, MinVisibilityIsLinearFloor) {
  EXPECT_NEAR(std::pow(double{ao::MinVisibility}, 2.2), 0.03, 0.002);
  EXPECT_EQ(ao::FinalPower, 1.f);
}

struct Consumer {
  wgpu::Texture texture;
  int calls = 0;
};

void declare_consumer(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.after(phase::Ao);
  builder.use(ao::hiz::Resource, Access::CopySrc);
  builder.side_effect();
}

void encode_consumer(PassContext& ctx, const void* state) {
  auto& consumer = *static_cast<Consumer*>(const_cast<void*>(state));
  ++consumer.calls;
  consumer.texture = ctx.texture(TexRef{0});
}

PassDesc consumer_desc(Consumer& consumer) {
  return {
      .name = "t.ao_read",
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 2,
      .declare = declare_consumer,
      .encode = encode_consumer,
      .state = &consumer,
  };
}

class AoDeviceTest : public ::testing::Test {
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
    m_started = true;
    ao::initialize();
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
    ao::shutdown();
    ao::testing::reset();
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
    ao::initialize();
    ASSERT_TRUE(webgpu::set_gbuffer(true));
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
    ao::on_latch(m_settings, graph);
    latch_pass_params();
  }

  static NodeId pyramid_node(const CompiledGraph& graph) { return graph.find_node(ao::hiz::Pass); }

  static bool pyramid_described(const CompiledGraph& graph) {
    const ResourceId id = graph.find_resource(ao::hiz::Resource);
    return id != InvalidResource && graph.resources[id].described;
  }

  void wait_ready(const CompiledGraph& graph) const {
    const NodeId node = pyramid_node(graph);
    ASSERT_NE(node, InvalidNode);
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t id = graph.nodes[node].pipelineIds[i];
      ASSERT_NE(id, 0u) << i;
      ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << i;
    }
  }

  std::shared_ptr<CompiledGraph> live_graph(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    const NodeId node = pyramid_node(*graph);
    EXPECT_NE(node, InvalidNode);
    if (node != InvalidNode) {
      EXPECT_EQ(graph->nodes[node].status, CullReason::None);
      wait_ready(*graph);
    }
    latch(graph.get());
    return graph;
  }

  StageData build(const FrameState& state, uint64_t frameId, std::shared_ptr<const SceneDepth> depth,
                  bool atFinish = false) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    StageData opaque{
        .stage = Stage::OpaqueDone,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = true,
        .flushedAtFinish = atFinish,
        .frame = &state,
        .sceneDepth = std::move(depth),
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

  std::optional<std::string> run(const FrameState& state, uint64_t frameId, std::shared_ptr<const SceneDepth> depth,
                                 bool atFinish = false) const {
    StageData opaque = build(state, frameId, std::move(depth), atFinish);
    return encode(opaque);
  }

  [[nodiscard]] bool pyramid_enabled(const CompiledGraph& graph, const StageData& stage) const {
    const NodeId node = pyramid_node(graph);
    return node != InvalidNode && graph.nodes[node].programIndex != InvalidNode &&
           stage.is_enabled(graph.nodes[node].programIndex);
  }

  static inline std::vector<uint8_t> s_uniforms;
  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  Settings m_settings;
  GraphEnvironment m_env;
};

class AoGraph : public AoDeviceTest {
protected:
  void SetUp() override {
    if (!open({.backend = wgpu::BackendType::Null})) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
  }

  [[nodiscard]] std::shared_ptr<const SceneDepth> depth64() const {
    return scene_depth(test::make_depth(*m_device, 64, 64));
  }
};

TEST_F(AoGraph, SkippedWhenAoOff) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = compile(Settings{});
  const NodeId node = pyramid_node(*graph);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, CullReason::Skipped);
  EXPECT_FALSE(pyramid_described(*graph));
  EXPECT_EQ(graph->nodes[node].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
}

TEST_F(AoGraph, SkippedWithoutTierB) {
  webgpu::set_gbuffer(false);
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = compile(ao_settings(Quality::High));
  EXPECT_EQ(graph->nodes[pyramid_node(*graph)].status, CullReason::Skipped);
  EXPECT_FALSE(pyramid_described(*graph));
}

TEST_F(AoGraph, SkippedWithoutDirectDepth) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  for (const DepthSource source : {DepthSource::Copy, DepthSource::Unavailable}) {
    const auto graph = compile(ao_settings(Quality::High), source);
    EXPECT_EQ(graph->nodes[pyramid_node(*graph)].status, CullReason::Skipped) << scene_depth_source_name(source);
  }
  const auto direct = compile(ao_settings(Quality::High));
  EXPECT_EQ(direct->nodes[pyramid_node(*direct)].status, CullReason::None);
}

TEST_F(AoGraph, CulledWithoutConsumer) {
  const auto graph = compile(ao_settings(Quality::High));
  const NodeId node = pyramid_node(*graph);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, CullReason::NoConsumer);
  EXPECT_EQ(graph->nodes[node].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
  latch(graph.get());
  EXPECT_FALSE(ao::pipelines_live());
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1, depth64()), std::nullopt);
  EXPECT_EQ(ao::hiz::testing::counters().dispatches, 0u);
}

TEST_F(AoGraph, DisabledUnderClassView) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  uint64_t frame = 1;
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, frame++, depth64())));
  for (const debug::DebugView& view : debug::ClassViews) {
    debug::set_view_override(view.id);
    latch(graph.get());
    EXPECT_FALSE(pyramid_enabled(*graph, build(state, frame++, depth64()))) << view.name;
  }
  debug::set_view_override(0);
  m_settings.debugView = 3;
  latch(graph.get());
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, frame++, depth64())));
  m_settings.debugView = 0;
  latch(graph.get());
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, frame++, depth64())));
}

TEST_F(AoGraph, RunsUnderGbufView) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  for (const debug::GbufView& view : debug::GbufViews) {
    debug::set_view_override(view.id);
    latch(graph.get());
    EXPECT_TRUE(pyramid_enabled(*graph, build(state, view.id, depth64()))) << view.name;
  }
  debug::set_view_override(29);
  latch(graph.get());
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, 40, depth64())));
}

TEST_F(AoGraph, DisabledWithoutUsableCamera) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  const uint32_t skippedBefore = ao::testing::counters().skippedNoScene;
  ao::testing::set_camera_override(std::make_shared<const SceneHints>());
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, 1, depth64())));
  Mat4x4<float> ortho = perspective(Near, Far);
  ortho.m0[3] = 0.f;
  ao::testing::set_camera_override(camera_hints(ortho, {}, std::nullopt, GX_ORTHOGRAPHIC));
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, 2, depth64())));
  Mat4x4<float> broken = perspective(Near, Far);
  broken.m2[3] = std::numeric_limits<float>::quiet_NaN();
  ao::testing::set_camera_override(camera_hints(broken));
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, 3, depth64())));
  ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, 4, depth64(), true)));
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, 5, depth64())));
  EXPECT_GE(ao::testing::counters().skippedNoScene, skippedBefore + 4);
  ao::testing::set_camera_override(nullptr);
  EXPECT_FALSE(pyramid_enabled(*graph, build(state, 6, depth64())));
}

TEST_F(AoGraph, PyramidSizeFollowsDivisor) {
  for (const auto [width, height] :
       {std::pair{2560u, 1440u}, std::pair{3840u, 2160u}, std::pair{3186u, 1792u}, std::pair{4779u, 2688u}}) {
    use_globals({width, height, 1});
    Consumer consumer;
    register_pass(consumer_desc(consumer));
    for (const Quality quality : {Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                   std::to_string(static_cast<int>(quality)));
      const uint32_t divisor = quality == Quality::Ultra ? 1u : 2u;
      const auto graph = compile(ao_settings(quality));
      ASSERT_EQ(graph->nodes[pyramid_node(*graph)].status, CullReason::None);
      const CompiledResource& pyramid = graph->resources[graph->find_resource(ao::hiz::Resource)];
      EXPECT_EQ(pyramid.tex.format, wgpu::TextureFormat::R32Float);
      EXPECT_EQ(pyramid.tex.divisor, divisor);
      EXPECT_EQ(pyramid.extent.width, (width + divisor - 1) / divisor);
      EXPECT_EQ(pyramid.extent.height, (height + divisor - 1) / divisor);
      EXPECT_EQ(pyramid.mipLevels, chain_levels(pyramid.extent.width, pyramid.extent.height));
      EXPECT_TRUE(pyramid.textureUsage & wgpu::TextureUsage::TextureBinding);
      EXPECT_TRUE(pyramid.textureUsage & wgpu::TextureUsage::StorageBinding);
      if (width == 4779) {
        EXPECT_EQ(pyramid.extent.width, divisor == 2 ? 2390u : 4779u);
        EXPECT_EQ(pyramid.extent.height, divisor == 2 ? 1344u : 2688u);
      }
    }
  }
}

TEST_F(AoGraph, ParamsLatchedEveryFrameAndAfterReinit) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto medium = live_graph(ao_settings(Quality::Medium));
  const PassId first = medium->nodes[pyramid_node(*medium)].id;
  ao::Params params = params_from(latched_pass_params(first));
  EXPECT_EQ(params.sliceCount, 2u);
  EXPECT_EQ(params.stepsPerSlice, 2u);
  EXPECT_EQ(params.strength, 1.f);
  EXPECT_EQ(params.radiusMetres, ao::RadiusMetres);
  EXPECT_EQ(params.thicknessMetres, ao::ThicknessMetres);
  EXPECT_EQ(params.falloffRange, ao::FalloffRange);
  EXPECT_EQ(params.minVisibility, ao::MinVisibility);
  EXPECT_EQ(params.flags, 0u);
  EXPECT_EQ(params.effectRadius, 0.f);
  EXPECT_TRUE(ao::pipelines_live());

  m_settings = ao_settings(Quality::High);
  latch(medium.get());
  params = params_from(latched_pass_params(first));
  EXPECT_EQ(params.sliceCount, 3u);
  EXPECT_EQ(params.stepsPerSlice, 3u);

  clear_pass_params(first);
  latch_pass_params();
  EXPECT_TRUE(latched_pass_params(first).empty());
  latch(medium.get());
  EXPECT_EQ(params_from(latched_pass_params(first)).sliceCount, 3u);

  ao::shutdown();
  EXPECT_FALSE(ao::pipelines_live());
  ao::initialize();
  const auto again = live_graph(ao_settings(Quality::Low));
  const PassId second = again->nodes[pyramid_node(*again)].id;
  EXPECT_NE(second, first);
  params = params_from(latched_pass_params(second));
  EXPECT_EQ(params.sliceCount, 1u);
  EXPECT_EQ(params.stepsPerSlice, 2u);
}

TEST_F(AoGraph, CameraFilledAtStageBuild) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::High));
  const Mat4x4<float> proj = perspective(Near, Far, 0.125f, -0.0625f);
  ao::testing::set_camera_override(camera_hints(proj, {0.25f, -0.375f}, 50.f));
  const FrameState state{.graph = graph};
  const uint32_t fillsBefore = ao::testing::counters().cameraFills;
  const StageData opaque = build(state, 70, depth64());
  ASSERT_TRUE(pyramid_enabled(*graph, opaque));
  EXPECT_EQ(ao::testing::counters().cameraFills, fillsBefore + 1);
  const NodeId node = pyramid_node(*graph);
  const auto it = std::ranges::find(opaque.bindings.params, node, &StageBindings::Params::node);
  ASSERT_NE(it, opaque.bindings.params.end());
  ASSERT_EQ(it->size, sizeof(ao::Params));
  const ao::Params params = params_from(std::span{it->bytes}.first(it->size));
  EXPECT_EQ(params.camera.depthMul, proj.m2[3]);
  EXPECT_EQ(params.camera.depthAdd, proj.m2[2]);
  EXPECT_EQ(params.camera.p00, proj.m0[0]);
  EXPECT_EQ(params.camera.p11, proj.m1[1]);
  EXPECT_EQ(params.camera.p02, 0.125f);
  EXPECT_EQ(params.camera.p12, -0.0625f);
  EXPECT_EQ(params.camera.jitterPx[0], 0.25f);
  EXPECT_EQ(params.camera.jitterPx[1], -0.375f);
  EXPECT_EQ(params.camera.unitsPerMeter, 50.f);
  EXPECT_EQ(params.camera.flags, CameraValid);
  EXPECT_EQ(params.effectRadius, ao::RadiusMetres * 50.f);
  EXPECT_EQ(params.thickness, ao::ThicknessMetres * 50.f);
  EXPECT_EQ(params.noiseIndex, 70u % ao::NoisePeriod);
  EXPECT_EQ(params.sliceCount, 3u);
  const ao::Params latched = params_from(latched_pass_params(graph->nodes[node].id));
  EXPECT_EQ(latched.camera.flags, 0u);
  EXPECT_EQ(latched.effectRadius, 0.f);
}

TEST_F(AoGraph, EncodesValidationClean) {
  for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{2u, 1u}, std::pair{1u, 3u}, std::pair{61u, 37u},
                                     std::pair{64u, 64u}, std::pair{1280u, 720u}}) {
    for (const Quality quality : {Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                   std::to_string(static_cast<int>(quality)));
      use_globals({width, height, 1});
      ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
      Consumer consumer;
      register_pass(consumer_desc(consumer));
      const auto graph = live_graph(ao_settings(quality));
      const CompiledResource& pyramid = graph->resources[graph->find_resource(ao::hiz::Resource)];
      const uint32_t levels = ao::hiz::written_levels(pyramid.mipLevels);
      const auto depth = scene_depth(test::make_depth(*m_device, width, height));
      const FrameState state{.graph = graph};
      EXPECT_EQ(run(state, 1, depth), std::nullopt);
      EXPECT_EQ(ao::hiz::testing::counters().dispatches, levels);
      EXPECT_EQ(ao::hiz::testing::counters().groupsCreated, levels - 1);
      EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 1u);
      EXPECT_EQ(run(state, 2, depth), std::nullopt);
      EXPECT_EQ(ao::hiz::testing::counters().dispatches, 2 * levels);
      EXPECT_EQ(ao::hiz::testing::counters().groupsCreated, levels - 1);
      on_worker_frame_begin(2 + ao::hiz::RetainFrames, &state);
      EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 1u);
      on_worker_frame_begin(3 + ao::hiz::RetainFrames, &state);
      EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 0u);
      EXPECT_EQ(consumer.calls, 2);
    }
  }
}

TEST_F(AoGraph, EnvUnsetBuildsNothing) {
  if (std::getenv("AURORA_RENDER_AO") != nullptr) {
    GTEST_SKIP() << "AURORA_RENDER_AO is set";
  }
  EXPECT_EQ(ao::env_ao_request(), std::nullopt);
  EXPECT_EQ(ao::with_dev_built(default_built_options(), ao::env_ao_request()), default_built_options());
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = compile(Settings{});
  EXPECT_EQ(graph->nodes[pyramid_node(*graph)].status, CullReason::Skipped);
  EXPECT_FALSE(pyramid_described(*graph));
  latch(graph.get());
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1, depth64()), std::nullopt);
  EXPECT_EQ(ao::hiz::testing::counters().dispatches, 0u);
  EXPECT_EQ(ao::testing::counters().cameraFills, 0u);
  EXPECT_EQ(shaders::stats().ready + shaders::stats().pending + shaders::stats().failed, 0u);
}

class AoHizGpu : public AoDeviceTest {
protected:
  void SetUp() override {
    m_backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!reopen(false)) {
      GTEST_SKIP() << m_skip;
    }
  }

  bool reopen(bool preciseTimestamps) {
    if (m_started) {
      TearDown();
      m_started = false;
    }
    if (!open({.backend = m_backend, .preciseTimestamps = preciseTimestamps})) {
      m_skip = "no adapter: " + m_error;
      return false;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      m_skip = "the null backend runs no shaders";
      return false;
    }
    if (select_scene_depth_source(m_device->core, 1, true) != DepthSource::Direct) {
      m_skip = "needs direct scene depth";
      return false;
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(m_device->adapterInfo.backendType));
    return true;
  }

  static std::vector<float> scene_values(std::mt19937& rng, const Mat4x4<float>& proj, uint32_t width,
                                         uint32_t height) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const auto stored = [&](double distance) {
      return static_cast<float>(double{proj.m2[2]} - double{proj.m2[3]} / distance);
    };
    std::vector<float> out(size_t{width} * height);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        float& value = out[size_t{y} * width + x];
        if (unit(rng) < 0.08) {
          value = 0.f;
        } else if (y % 3 == 0) {
          value = stored(Near * std::pow(2000.0, unit(rng)));
        } else {
          value = stored(80.0 + 40.0 * std::sin(x * 0.05) + 25.0 * std::cos(y * 0.07) + 0.001 * x * y);
        }
      }
    }
    return out;
  }

  void expect_mirror(const wgpu::Texture& texture, const ref::CpuPyramid& want, std::string_view what) const {
    ASSERT_TRUE(texture) << what;
    for (uint32_t level = 0; level < want.level_count(); ++level) {
      const std::vector<float> got = test::read_texture_mip(*m_device, texture, level);
      ASSERT_EQ(got.size(), want.levels[level].size()) << what << " level " << level;
      const double tolerance = level == 0 ? 1e-6 : 1e-5;
      size_t bad = 0;
      size_t first = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        const double expected = want.levels[level][i];
        if (!std::isfinite(got[i]) || got[i] <= 0.f ||
            std::abs(double{got[i]} - expected) > tolerance * std::max(std::abs(expected), 1.0)) {
          first = bad == 0 ? i : first;
          ++bad;
        }
      }
      EXPECT_EQ(bad, 0u) << what << " level " << level << ": first mismatch at texel " << first << ", got "
                         << got[first] << ", want " << want.levels[level][first];
    }
  }

  wgpu::BackendType m_backend = wgpu::BackendType::Undefined;
  std::string m_skip;
};

TEST_F(AoHizGpu, PyramidMatchesCpuMirror) {
  std::mt19937 rng{0xA0712u};
  const Mat4x4<float> proj = perspective(Near, Far, 0.03f, -0.02f);
  struct Case {
    uint32_t width;
    uint32_t height;
    Quality quality;
  };
  for (const Case c : {Case{3, 5, Quality::High}, Case{61, 37, Quality::High}, Case{1279, 719, Quality::High},
                       Case{4779, 2688, Quality::Ultra}, Case{4779, 2688, Quality::High}}) {
    SCOPED_TRACE(std::to_string(c.width) + "x" + std::to_string(c.height) + " q" +
                 std::to_string(static_cast<int>(c.quality)));
    use_globals({c.width, c.height, 1});
    const auto hints = camera_hints(proj, {0.25f, -0.125f});
    ao::testing::set_camera_override(hints);
    Consumer consumer;
    register_pass(consumer_desc(consumer));
    const auto graph = live_graph(ao_settings(c.quality));
    const uint32_t divisor = ao::working_divisor(c.quality);
    const std::vector<float> values = scene_values(rng, proj, c.width, c.height);
    const wgpu::Texture depth = test::make_depth(*m_device, c.width, c.height);
    test::fill_depth(*m_device, depth, values);
    const FrameState state{.graph = graph};
    ASSERT_EQ(run(state, 1, scene_depth(depth)), std::nullopt);
    ASSERT_EQ(consumer.calls, 1);
    StageCamera camera;
    ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
    const uint32_t gridWidth = ao::grid_size(c.width, divisor);
    const uint32_t gridHeight = ao::grid_size(c.height, divisor);
    const uint32_t levels = ao::hiz::written_levels(chain_levels(gridWidth, gridHeight));
    const ref::CpuPyramid want = ref::build_pyramid(values, c.width, c.height, divisor, camera,
                                                    double{ao::RadiusMetres * DefaultUnitsPerMetre}, levels);
    ASSERT_EQ(consumer.texture.GetWidth(), gridWidth);
    ASSERT_EQ(consumer.texture.GetHeight(), gridHeight);
    expect_mirror(consumer.texture, want, "pyramid");
  }
}

TEST_F(AoHizGpu, InfiniteFarStaysFinite) {
  constexpr uint32_t Width = 64;
  constexpr uint32_t Height = 48;
  use_globals({Width, Height, 1});
  const Mat4x4<float> proj = perspective(Near, std::numeric_limits<float>::infinity());
  ASSERT_EQ(proj.m2[2], 0.f);
  const auto hints = camera_hints(proj);
  ao::testing::set_camera_override(hints);
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::Ultra));
  constexpr std::array<float, 8> Depths{0.f, 1e-30f, 1e-20f, 1e-10f, 1e-6f, 0.001f, 0.5f, 1.f};
  std::vector<float> values(size_t{Width} * Height);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = Depths[(i * 7 + i / Width) % Depths.size()];
  }
  const wgpu::Texture depth = test::make_depth(*m_device, Width, Height);
  test::fill_depth(*m_device, depth, values);
  const FrameState state{.graph = graph};
  ASSERT_EQ(run(state, 1, scene_depth(depth)), std::nullopt);
  StageCamera camera;
  ASSERT_TRUE(fill_stage_camera(hints.get(), camera));
  const ref::CpuPyramid want =
      ref::build_pyramid(values, Width, Height, 1, camera, double{ao::RadiusMetres * DefaultUnitsPerMetre},
                         ao::hiz::written_levels(chain_levels(Width, Height)));
  expect_mirror(consumer.texture, want, "infinite far");
  const std::vector<float> mip0 = test::read_texture_mip(*m_device, consumer.texture, 0);
  EXPECT_TRUE(std::ranges::any_of(mip0, [](float z) { return z == ao::SkyDepth; }));
  EXPECT_TRUE(std::ranges::any_of(mip0, [](float z) { return z == Near; }));
}

TEST_F(AoHizGpu, CacheAndCountersFollowLevels) {
  for (const auto [width, height, levels] : {std::tuple{3u, 5u, 2u}, std::tuple{1279u, 719u, 5u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    use_globals({width, height, 1});
    ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
    Consumer consumer;
    register_pass(consumer_desc(consumer));
    const auto graph = live_graph(ao_settings(Quality::High));
    const auto depth = scene_depth(test::make_depth(*m_device, width, height));
    const FrameState state{.graph = graph};
    ASSERT_EQ(run(state, 1, depth), std::nullopt);
    EXPECT_EQ(ao::hiz::testing::counters().dispatches, levels);
    EXPECT_EQ(ao::hiz::testing::counters().groupsCreated, levels - 1);
    EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 1u);
    ASSERT_EQ(run(state, 2, depth), std::nullopt);
    EXPECT_EQ(ao::hiz::testing::counters().dispatches, 2 * levels);
    EXPECT_EQ(ao::hiz::testing::counters().groupsCreated, levels - 1);
    on_worker_frame_begin(3 + ao::hiz::RetainFrames, &state);
    EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 0u);
    ASSERT_EQ(run(state, 4 + ao::hiz::RetainFrames, depth), std::nullopt);
    EXPECT_EQ(ao::hiz::testing::counters().groupsCreated, 2 * (levels - 1));
    EXPECT_EQ(ao::hiz::testing::counters().cacheEntries, 1u);
  }
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  return values[values.size() / 2];
}

using AoGpu = AoHizGpu;

TEST_F(AoGpu, Bench) {
  const char* enabled = std::getenv("AURORA_TEST_BENCH");
  if (enabled == nullptr || std::string_view{enabled} != "1") {
    GTEST_SKIP() << "set AURORA_TEST_BENCH=1 to run";
  }
  ASSERT_TRUE(reopen(true)) << m_skip;
  if (!m_device->device.HasFeature(wgpu::FeatureName::TimestampQuery)) {
    GTEST_SKIP() << "needs timestamp queries";
  }
  constexpr uint32_t Warmup = 50;
  constexpr uint32_t Timed = 200;
  std::mt19937 rng{0xBE7C4u};
  const Mat4x4<float> proj = perspective(Near, Far);
  for (const auto [width, height] : {std::pair{1280u, 720u}, std::pair{1920u, 1080u}, std::pair{2560u, 1440u},
                                     std::pair{3840u, 2160u}, std::pair{4779u, 2688u}}) {
    use_globals({width, height, 1});
    ao::testing::set_camera_override(camera_hints(proj));
    Consumer consumer;
    register_pass(consumer_desc(consumer));
    const auto graph = live_graph(ao_settings(Quality::High));
    const wgpu::Texture depth = test::make_depth(*m_device, width, height);
    test::fill_depth(*m_device, depth, scene_values(rng, proj, width, height));
    const auto sceneDepth = scene_depth(depth);
    const FrameState state{.graph = graph};
    const wgpu::QuerySetDescriptor queryDescriptor{.type = wgpu::QueryType::Timestamp, .count = 2};
    const wgpu::QuerySet queries = m_device->device.CreateQuerySet(&queryDescriptor);
    const wgpu::BufferDescriptor resolvedDescriptor{
        .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, .size = 16};
    const wgpu::Buffer resolved = m_device->device.CreateBuffer(&resolvedDescriptor);
    std::vector<double> gpuMs;
    for (uint32_t frame = 1; frame <= Warmup + Timed; ++frame) {
      StageData opaque = build(state, frame, sceneDepth);
      s_uniforms.assign(FrameUniformsSize, 0);
      build_stage_bindings(opaque, &uniform_sink);
      s_uniforms.resize((s_uniforms.size() + 3) & ~size_t{3});
      m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, s_uniforms.data(), s_uniforms.size());
      const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
      encoder.WriteTimestamp(queries, 0);
      run_stage(encoder, opaque);
      encoder.WriteTimestamp(queries, 1);
      encoder.ResolveQuerySet(queries, 0, 2, resolved, 0);
      const wgpu::CommandBuffer commands = encoder.Finish();
      ASSERT_TRUE(test::submit_and_wait(*m_device, commands));
      if (frame <= Warmup) {
        continue;
      }
      const std::vector<uint32_t> stamps =
          test::read_buffer(*m_device, resolved, 0, 16).value_or(std::vector<uint32_t>{});
      ASSERT_EQ(stamps.size(), 4u);
      const uint64_t begin = stamps[0] | (uint64_t{stamps[1]} << 32);
      const uint64_t end = stamps[2] | (uint64_t{stamps[3]} << 32);
      gpuMs.push_back(static_cast<double>(end - begin) / 1e6);
    }
    std::printf("ao hiz bench %ux%u backend %d adapter %s: gpu median %.4f ms\n", width, height,
                static_cast<int>(m_device->adapterInfo.backendType),
                std::string{std::string_view{m_device->adapterInfo.device}}.c_str(), median(gpuMs));
  }
}
} // namespace
} // namespace aurora::render::detail

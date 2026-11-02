#include <gtest/gtest.h>

#include "render_ao_reference.hpp"
#include "render_depth_device.hpp"
#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/ao/ao.hpp"
#include "render/ao/denoise.hpp"
#include "render/ao/gtao.hpp"
#include "render/ao/hiz.hpp"
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
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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
  hints->motionValid = true;
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
    EXPECT_EQ(filled.noiseIndex, stage.sceneHints->hintFrame % ao::NoisePeriod);
    EXPECT_EQ(filled.sliceCount, latched.sliceCount);
    EXPECT_EQ(filled.radiusMetres, ao::RadiusMetres);
  }
  ao::testing::reset();
}

TEST(AoRules, NoiseFollowsHintFrame) {
  ao::testing::reset();
  const ao::Params latched;
  const auto noise_index = [&](uint32_t frameIndex, uint64_t hintFrame) {
    auto hints = std::make_shared<SceneHints>(*camera_hints(perspective(Near, Far)));
    hints->hintFrame = hintFrame;
    std::array<std::byte, sizeof(ao::Params)> bytes{};
    std::memcpy(bytes.data(), &latched, sizeof(latched));
    ao::fill_camera({.frameIndex = frameIndex, .sceneHints = std::move(hints)}, nullptr, bytes);
    return params_from(bytes).noiseIndex;
  };
  EXPECT_EQ(noise_index(70, 135), 135u % ao::NoisePeriod);
  // A replay repeat restarts the hint frame but not the gfx frame index
  EXPECT_EQ(noise_index(70 + 300, 135), noise_index(70, 135));
  EXPECT_NE(noise_index(70, 136), noise_index(70, 135));
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
    if (m_sceneSize) {
      m_env.sceneSize = *m_sceneSize;
    }
    std::shared_ptr<CompiledGraph> graph = compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = source});
    prepare_graph_bindings(*graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = source});
    return graph;
  }

  void latch(const CompiledGraph* graph) {
    ao::on_latch({.settings = &m_settings, .env = &m_env}, graph);
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
    for (const std::string_view name : {ao::gtao::Pass, ao::denoise::Pass, ao::upsample::Pass}) {
      const NodeId chained = graph.find_node(name);
      if (chained == InvalidNode || graph.nodes[chained].status != CullReason::None) {
        continue;
      }
      for (const uint32_t id : graph.nodes[chained].pipelineIds) {
        if (id != 0) {
          ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << name;
        }
      }
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
                  bool atFinish = false, bool invalidate = false) const {
    on_worker_frame_begin(frameId, &state);
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &state};
    build_stage_data(begin);
    if (invalidate) {
      invalidate_frame_histories();
    }
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
                                 bool atFinish = false, bool invalidate = false) const {
    StageData opaque = build(state, frameId, std::move(depth), atFinish, invalidate);
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
  std::optional<GraphExtent> m_sceneSize;
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
  for (const std::string_view name : {ao::hiz::Pass, ao::gtao::Pass, ao::denoise::Pass}) {
    const NodeId node = graph->find_node(name);
    ASSERT_NE(node, InvalidNode) << name;
    EXPECT_EQ(graph->nodes[node].status, CullReason::None) << name;
  }
  const NodeId upsample = graph->find_node(ao::upsample::Pass);
  ASSERT_NE(upsample, InvalidNode);
  EXPECT_EQ(graph->nodes[upsample].status, CullReason::NoConsumer);
  EXPECT_EQ(graph->nodes[upsample].pipelineIds, (std::array<uint32_t, MaxPassPipelines>{}));
}

TEST_F(AoGraph, DisabledUnderClassView) {
  Consumer consumer;
  register_pass(consumer_desc(consumer));
  const auto graph = live_graph(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  uint64_t frame = 1;
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, frame++, depth64())));
  for (const debug::DebugView& view : debug::ClassViews) {
    m_env.viewOverride = view.id;
    latch(graph.get());
    EXPECT_FALSE(pyramid_enabled(*graph, build(state, frame++, depth64()))) << view.name;
  }
  m_env.viewOverride = 0;
  // Only the override the graph environment latched counts, the way TAA's declare sees it
  debug::set_view_override(debug::ClassViews.front().id);
  latch(graph.get());
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, frame++, depth64())));
  debug::set_view_override(0);
  m_env.viewOverride = 16;
  m_settings.debugView = 3;
  latch(graph.get());
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, frame++, depth64())));
  m_env.viewOverride = 0;
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
    m_env.viewOverride = view.id;
    latch(graph.get());
    EXPECT_TRUE(pyramid_enabled(*graph, build(state, view.id, depth64()))) << view.name;
  }
  m_env.viewOverride = 29;
  latch(graph.get());
  EXPECT_TRUE(pyramid_enabled(*graph, build(state, 40, depth64())));
  m_env.viewOverride = 0;
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
  uint32_t enabled = 0;
  for (const std::string_view name : {ao::hiz::Pass, ao::gtao::Pass, ao::denoise::Pass, ao::upsample::Pass}) {
    const NodeId chained = graph->find_node(name);
    enabled += chained != InvalidNode && graph->nodes[chained].programIndex != InvalidNode &&
                       opaque.is_enabled(graph->nodes[chained].programIndex)
                   ? 1
                   : 0;
  }
  EXPECT_EQ(enabled, 3u);
  EXPECT_EQ(ao::testing::counters().cameraFills, fillsBefore + enabled);
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
  ASSERT_NE(opaque.sceneHints, nullptr);
  EXPECT_EQ(params.noiseIndex, opaque.sceneHints->hintFrame % ao::NoisePeriod);
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

TEST(AoRules, SectorBitsRoundEdgeCases) {
  EXPECT_EQ(ao::sector_bits(0.f, 0.49f / 32.f, false), 0u);
  EXPECT_EQ(ao::sector_bits(0.f, 0.51f / 32.f, false), 1u);
  EXPECT_EQ(ao::sector_bits(0.f, 1.f, false), 0xFFFFFFFFu);
  EXPECT_EQ(ao::sector_bits(-3.f, 7.f, false), 0xFFFFFFFFu);
  EXPECT_EQ(ao::sector_bits(31.f / 32.f, 1.f, false), 0x80000000u);
  EXPECT_EQ(ao::sector_bits(0.7f, 0.3f, false), 0u);
  EXPECT_EQ(ao::sector_bits(1.f, 1.f, false), 0u);
  EXPECT_EQ(ao::sector_bits(2.f / 32.f, 5.f / 32.f, false), 0b11100u);
  EXPECT_EQ(ao::sector_bits(1.49f / 32.f, 2.51f / 32.f, false), 0b110u);
}

TEST(AoRules, SectorBitsCeilVariant) {
  EXPECT_EQ(ao::sector_bits(0.f, 0.49f / 32.f, true), 1u);
  EXPECT_EQ(ao::sector_bits(0.f, 1.f, true), 0xFFFFFFFFu);
  EXPECT_EQ(ao::sector_bits(31.5f / 32.f, 1.f, true), 0x80000000u);
  EXPECT_EQ(ao::sector_bits(0.7f, 0.3f, true), 0u);
  EXPECT_EQ(ao::sector_bits(1.f, 1.f, true), 0u);
  EXPECT_EQ(ao::sector_bits(1.2f / 32.f, 2.1f / 32.f, true), 0b10u);
  EXPECT_EQ(ao::sector_bits(0.1f / 32.f, 0.2f / 32.f, true), 1u);
  EXPECT_EQ(ao::sector_bits(0.1f / 32.f, 0.2f / 32.f, false), 0u);
  EXPECT_EQ(ao::sector_bits(5.9f / 32.f, 6.1f / 32.f, true), 0b100000u);
}

TEST(AoRules, HilbertMatchesXeGtao) {
  constexpr uint32_t Width = ao::HilbertWidth;
  std::vector<std::array<uint32_t, 2>> cells(Width * Width, {UINT32_MAX, UINT32_MAX});
  for (uint32_t y = 0; y < Width; ++y) {
    for (uint32_t x = 0; x < Width; ++x) {
      const uint32_t index = ao::hilbert_index(x, y);
      ASSERT_LT(index, Width * Width);
      EXPECT_EQ(cells[index][0], UINT32_MAX) << "index " << index << " repeats";
      cells[index] = {x, y};
      EXPECT_EQ(ao::hilbert_index(x + Width, y + 3 * Width), index);
    }
  }
  EXPECT_EQ(ao::hilbert_index(0, 0), 0u);
  for (size_t i = 1; i < cells.size(); ++i) {
    const uint32_t dx = cells[i][0] > cells[i - 1][0] ? cells[i][0] - cells[i - 1][0] : cells[i - 1][0] - cells[i][0];
    const uint32_t dy = cells[i][1] > cells[i - 1][1] ? cells[i][1] - cells[i - 1][1] : cells[i - 1][1] - cells[i][1];
    EXPECT_EQ(dx + dy, 1u) << "curve breaks at " << i;
  }
}

TEST(AoRules, FalloffMovesBoundsToLowEdge) {
  for (const bool positive : {true, false}) {
    const auto same = ao::falloff_bounds(0.3f, 0.6f, 1.f, positive);
    EXPECT_EQ(same[0], 0.3f);
    EXPECT_EQ(same[1], 0.6f);
    const auto none = ao::falloff_bounds(0.3f, 0.6f, 0.f, positive);
    EXPECT_EQ(none[0], positive ? 1.f : 0.f);
    EXPECT_EQ(none[1], positive ? 1.f : 0.f);
    EXPECT_EQ(ao::sector_bits(none[0], none[1], false), 0u);
    const auto half = ao::falloff_bounds(0.3f, 0.6f, 0.5f, positive);
    EXPECT_FLOAT_EQ(half[0], positive ? 0.65f : 0.15f);
    EXPECT_FLOAT_EQ(half[1], positive ? 0.8f : 0.3f);
    EXPECT_FLOAT_EQ(half[1] - half[0], 0.15f);
  }
}

TEST(AoRules, NoiseRotatesEveryFrame) {
  for (const auto [x, y] : {std::pair{0u, 0u}, std::pair{17u, 40u}, std::pair{1279u, 719u}}) {
    std::array<float, 2> previous = ao::noise(x, y, 0);
    for (uint32_t frame = 1; frame < ao::NoisePeriod; ++frame) {
      const std::array<float, 2> noise = ao::noise(x, y, frame);
      EXPECT_NE(noise, previous) << frame;
      EXPECT_GE(noise[0], 0.f);
      EXPECT_LT(noise[0], 1.f);
      EXPECT_GE(noise[1], 0.f);
      EXPECT_LT(noise[1], 1.f);
      previous = noise;
    }
    EXPECT_EQ(ao::noise(x, y, ao::NoisePeriod), ao::noise(x, y, 0));
  }
}

TEST(AoRules, StrengthCurve) {
  const float floor = ao::MinVisibility;
  for (float v = floor; v <= 1.f; v += 0.01f) {
    EXPECT_EQ(ao::apply_strength(v, 0.f, floor), 1.f) << v;
    EXPECT_FLOAT_EQ(ao::apply_strength(v, 1.f, floor), v) << v;
    EXPECT_NEAR(1.f - ao::apply_strength(v, 0.5f, floor), 0.5f * (1.f - v), 1e-6f) << v;
    EXPECT_GE(ao::apply_strength(v, 1.5f, floor), floor - 1e-6f) << v;
    EXPECT_LE(ao::apply_strength(v, 1.5f, floor), v + 1e-6f) << v;
  }
  for (float k = 0.f; k <= 1.5f; k += 0.05f) {
    EXPECT_EQ(ao::apply_strength(1.f, k, floor), 1.f) << k;
  }
}

TEST(AoRules, TuningDefaultsMatchConstants) {
  EXPECT_EQ(Settings{}.aoRadius, ao::RadiusMetres);
  EXPECT_EQ(Settings{}.aoStrength, 1.f);
  const ao::Params params = ao::latched_params(Settings{}, {});
  EXPECT_EQ(params.radiusMetres, ao::RadiusMetres);
  EXPECT_EQ(params.strength, 1.f);
  EXPECT_EQ(params.thicknessMetres, ao::ThicknessMetres);
  EXPECT_EQ(params.minVisibility, ao::MinVisibility);
}

struct Reader {
  std::string_view resource;
  wgpu::Texture texture;
  int calls = 0;
};

void declare_reader(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  builder.after(phase::Ao);
  builder.use(static_cast<const Reader*>(state)->resource, Access::CopySrc);
  builder.side_effect();
}

void encode_reader(PassContext& ctx, const void* state) {
  auto& reader = *static_cast<Reader*>(const_cast<void*>(state));
  ++reader.calls;
  reader.texture = ctx.texture(TexRef{0});
}

PassDesc reader_desc(Reader& reader) {
  return {
      .name = "t.read." + std::string{reader.resource},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .order = 3,
      .declare = declare_reader,
      .encode = encode_reader,
      .state = &reader,
  };
}

constexpr std::array<std::string_view, 4> ChainPasses{ao::hiz::Pass, ao::gtao::Pass, ao::denoise::Pass,
                                                      ao::upsample::Pass};

std::span<const CompiledUse> node_uses(const CompiledGraph& graph, NodeId node) {
  return std::span{graph.uses}.subspan(graph.nodes[node].firstUse, graph.nodes[node].useCount);
}

class AoChainGraph : public AoGraph {
protected:
  void latch_chain(const CompiledGraph& graph) {
    latch(&graph);
    for (const std::string_view name : ChainPasses) {
      const NodeId node = graph.find_node(name);
      if (node == InvalidNode || graph.nodes[node].status != CullReason::None) {
        continue;
      }
      for (const uint32_t id : graph.nodes[node].pipelineIds) {
        if (id != 0) {
          ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << name;
        }
      }
    }
  }
};

TEST_F(AoChainGraph, GtaoBindsHizThenNormal) {
  Reader reader{.resource = ao::gtao::Resource};
  register_pass(reader_desc(reader));
  const auto graph = compile(ao_settings(Quality::High));
  const NodeId node = graph->find_node(ao::gtao::Pass);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_EQ(graph->nodes[node].order, 0u);
  const auto uses = node_uses(*graph, node);
  ASSERT_GE(uses.size(), 3u);
  EXPECT_EQ(graph->resources[uses[0].resource].name, ao::hiz::Resource);
  EXPECT_EQ(uses[0].access, Access::Sampled);
  EXPECT_EQ(uses[0].options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(graph->resources[uses[1].resource].name, resource::GxNormal);
  EXPECT_EQ(uses[1].access, Access::Sampled);
  EXPECT_EQ(graph->resources[uses[2].resource].name, ao::gtao::Resource);
  EXPECT_EQ(uses[2].access, Access::StorageWrite);
  const CompiledResource& raw = graph->resources[graph->find_resource(ao::gtao::Resource)];
  EXPECT_EQ(raw.tex.format, wgpu::TextureFormat::RGBA8Unorm);
  EXPECT_EQ(raw.tex.divisor, 2u);
}

TEST_F(AoChainGraph, UpsampleCreatesRenderSizeRgba8) {
  for (const auto [width, height] : {std::pair{2560u, 1440u}, std::pair{4779u, 2688u}, std::pair{61u, 37u}}) {
    use_globals({width, height, 1});
    ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
    Reader reader{.resource = resource::AoFinal};
    register_pass(reader_desc(reader));
    for (const Quality quality : {Quality::Low, Quality::High, Quality::Ultra}) {
      const auto graph = compile(ao_settings(quality));
      const NodeId node = graph->find_node(ao::upsample::Pass);
      ASSERT_NE(node, InvalidNode);
      EXPECT_EQ(graph->nodes[node].status, CullReason::None);
      EXPECT_EQ(graph->nodes[node].passKind, PassKind::Render);
      EXPECT_EQ(graph->nodes[node].order, 2u);
      const CompiledResource& final = graph->resources[graph->find_resource(resource::AoFinal)];
      EXPECT_EQ(final.tex.format, wgpu::TextureFormat::RGBA8Unorm);
      EXPECT_EQ(final.tex.divisor, 1u);
      EXPECT_EQ(final.extent.width, width);
      EXPECT_EQ(final.extent.height, height);
      EXPECT_TRUE(final.textureUsage & wgpu::TextureUsage::RenderAttachment);
      EXPECT_TRUE(final.textureUsage & wgpu::TextureUsage::TextureBinding);
    }
  }
}

TEST_F(AoChainGraph, UpsampleReadsNormal) {
  Reader reader{.resource = resource::AoFinal};
  register_pass(reader_desc(reader));
  const auto graph = compile(ao_settings(Quality::High));
  const NodeId node = graph->find_node(ao::upsample::Pass);
  ASSERT_NE(node, InvalidNode);
  const auto uses = node_uses(*graph, node);
  ASSERT_EQ(uses.size(), 5u);
  const std::array<std::string_view, 5> names{ao::denoise::History, ao::hiz::Resource, resource::GxDepth,
                                              resource::GxNormal, resource::AoFinal};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, names[i]) << i;
    EXPECT_EQ(uses[i].missing, Missing::DisablePass) << i;
  }
  EXPECT_EQ(uses[1].options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(uses[2].options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(uses[4].access, Access::ColorWrite);
}

TEST_F(AoChainGraph, FinalIsTransientNotHistory) {
  Reader reader{.resource = resource::AoFinal};
  register_pass(reader_desc(reader));
  const auto graph = compile(ao_settings(Quality::High));
  for (const std::string_view name : {resource::AoFinal, ao::gtao::Resource, ao::hiz::Resource}) {
    const ResourceId id = graph->find_resource(name);
    ASSERT_NE(id, InvalidResource) << name;
    EXPECT_EQ(graph->resources[id].kind, ResourceKind::Transient) << name;
  }
  ASSERT_EQ(graph->histories.size(), 1u);
  EXPECT_EQ(graph->resources[graph->histories[0]].name, ao::denoise::History);
}

TEST_F(AoChainGraph, SinkEnvKeepsChainAlive) {
  const auto culled = compile(ao_settings(Quality::High));
  for (const std::string_view name : ChainPasses) {
    EXPECT_EQ(culled->nodes[culled->find_node(name)].status,
              name == ao::upsample::Pass ? CullReason::NoConsumer : CullReason::None)
        << name;
  }
  ao::testing::set_sink_override(true);
  const auto sunk = compile(ao_settings(Quality::High));
  for (const std::string_view name : ChainPasses) {
    EXPECT_EQ(sunk->nodes[sunk->find_node(name)].status, CullReason::None) << name;
  }
  EXPECT_TRUE(sunk->nodes[sunk->find_node(ao::upsample::Pass)].sideEffect);
  EXPECT_NE(sunk->graphHash, culled->graphHash);
  latch_chain(*sunk);
  EXPECT_TRUE(ao::pipelines_live());
  const FrameState state{.graph = sunk};
  EXPECT_EQ(run(state, 1, scene_depth(test::make_depth(*m_device, 64, 64))), std::nullopt);
  EXPECT_EQ(ao::gtao::testing::counters().dispatches, 1u);
  EXPECT_EQ(ao::upsample::testing::counters().draws, 1u);
}

TEST_F(AoChainGraph, ChainDisabledWhenPyramidDisabled) {
  Reader reader{.resource = resource::AoFinal};
  register_pass(reader_desc(reader));
  const auto graph = compile(ao_settings(Quality::High));
  latch_chain(*graph);
  const FrameState state{.graph = graph};
  const auto encoded = [] {
    return std::array{ao::hiz::testing::counters().dispatches, ao::gtao::testing::counters().dispatches,
                      ao::upsample::testing::counters().draws};
  };
  EXPECT_EQ(run(state, 1, depth64()), std::nullopt);
  const auto live = encoded();
  EXPECT_GT(live[0], 0u);
  EXPECT_EQ(live[1], 1u);
  EXPECT_EQ(live[2], 1u);
  EXPECT_EQ(reader.calls, 1);
  ao::testing::set_camera_override(std::make_shared<const SceneHints>());
  EXPECT_EQ(run(state, 2, depth64()), std::nullopt);
  EXPECT_EQ(encoded(), live);
  ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
  m_env.viewOverride = debug::ClassViews.front().id;
  latch(graph.get());
  EXPECT_EQ(run(state, 3, depth64()), std::nullopt);
  EXPECT_EQ(encoded(), live);
  EXPECT_EQ(reader.calls, 1);
  m_env.viewOverride = 0;
  latch(graph.get());
  EXPECT_EQ(run(state, 4, depth64()), std::nullopt);
  EXPECT_EQ(encoded()[2], 2u);
}

TEST_F(AoChainGraph, TuningReachesParams) {
  Reader reader{.resource = resource::AoFinal};
  register_pass(reader_desc(reader));
  Settings settings = ao_settings(Quality::High);
  settings.aoRadius = 1.f;
  settings.aoStrength = 0.5f;
  const auto graph = compile(settings);
  latch_chain(*graph);
  ao::testing::set_camera_override(camera_hints(perspective(Near, Far), {}, 50.f));
  const FrameState state{.graph = graph};
  const StageData opaque = build(state, 9, depth64());
  for (const std::string_view name : ChainPasses) {
    const NodeId node = graph->find_node(name);
    const ao::Params latched = params_from(latched_pass_params(graph->nodes[node].id));
    EXPECT_EQ(latched.radiusMetres, 1.f) << name;
    EXPECT_EQ(latched.strength, 0.5f) << name;
    const auto it = std::ranges::find(opaque.bindings.params, node, &StageBindings::Params::node);
    ASSERT_NE(it, opaque.bindings.params.end()) << name;
    const ao::Params filled = params_from(std::span{it->bytes}.first(it->size));
    EXPECT_EQ(filled.effectRadius, 50.f) << name;
    EXPECT_EQ(filled.strength, 0.5f) << name;
  }
  EXPECT_EQ(ao::effective_strength(), 0.5f);
  const ao::Params env =
      ao::latched_params(settings, {.radius = 0.6f, .thickness = 2.f, .strength = 1.2f, .floor = 0.03f});
  EXPECT_EQ(env.radiusMetres, 0.6f);
  EXPECT_EQ(env.strength, 1.2f);
  EXPECT_EQ(env.thicknessMetres, 2.f);
  EXPECT_EQ(env.minVisibility, 0.03f);
  const ao::Params clamped = ao::latched_params(settings, {.radius = 9.f, .strength = 9.f});
  EXPECT_EQ(clamped.radiusMetres, 1.5f);
  EXPECT_EQ(clamped.strength, 1.5f);
}

TEST_F(AoChainGraph, TuningKeepsGraphHash) {
  Reader reader{.resource = resource::AoFinal};
  register_pass(reader_desc(reader));
  Settings settings = ao_settings(Quality::High);
  const auto base = compile(settings);
  latch_chain(*base);
  const shaders::PipelineStats before = shaders::stats();
  for (const auto [radius, strength] : {std::pair{0.25f, 0.f}, std::pair{1.5f, 1.5f}, std::pair{1.f, 0.5f}}) {
    settings.aoRadius = radius;
    settings.aoStrength = strength;
    const auto tuned = compile(settings);
    EXPECT_EQ(tuned->graphHash, base->graphHash);
    for (const std::string_view name : ChainPasses) {
      EXPECT_EQ(tuned->nodes[tuned->find_node(name)].pipelineIds, base->nodes[base->find_node(name)].pipelineIds)
          << name;
    }
    for (size_t feature = 0; feature < FeatureCount; ++feature) {
      EXPECT_EQ(feature_enabled(settings, static_cast<Feature>(feature)),
                feature_enabled(ao_settings(Quality::High), static_cast<Feature>(feature)));
    }
  }
  const shaders::PipelineStats after = shaders::stats();
  EXPECT_EQ(after.ready + after.pending + after.failed, before.ready + before.pending + before.failed);
}

TEST_F(AoChainGraph, EncodesValidationClean) {
  for (const auto [width, height] :
       {std::pair{1u, 1u}, std::pair{3u, 5u}, std::pair{61u, 37u}, std::pair{1280u, 720u}}) {
    for (const Quality quality : {Quality::Low, Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                   std::to_string(static_cast<int>(quality)));
      use_globals({width, height, 1});
      ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
      Reader reader{.resource = resource::AoFinal};
      register_pass(reader_desc(reader));
      const auto graph = compile(ao_settings(quality));
      latch_chain(*graph);
      const FrameState state{.graph = graph};
      EXPECT_EQ(run(state, 1, scene_depth(test::make_depth(*m_device, width, height))), std::nullopt);
      EXPECT_EQ(ao::gtao::testing::counters().dispatches, 1u);
      EXPECT_EQ(ao::upsample::testing::counters().draws, 1u);
      EXPECT_EQ(reader.calls, 1);
    }
  }
}

class AoDenoiseGraph : public AoChainGraph {
protected:
  std::shared_ptr<CompiledGraph> live_chain(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    latch_chain(*graph);
    return graph;
  }

  static ao::denoise::testing::Counters denoise_counters() { return ao::denoise::testing::counters(); }

  static void set_chain_flags(const CompiledGraph& graph, uint32_t flags) {
    for (const std::string_view name : ChainPasses) {
      const NodeId node = graph.find_node(name);
      if (node == InvalidNode || graph.nodes[node].status != CullReason::None) {
        continue;
      }
      ao::Params params = params_from(latched_pass_params(graph.nodes[node].id));
      params.flags = flags;
      set_pass_params(graph.nodes[node].id, std::as_bytes(std::span{&params, 1}));
    }
    latch_pass_params();
  }

  void encode_sizes() {
    for (const auto [width, height] :
         {std::pair{1u, 1u}, std::pair{3u, 5u}, std::pair{61u, 37u}, std::pair{1280u, 720u}}) {
      for (const Quality quality : {Quality::Low, Quality::High, Quality::Ultra}) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                     std::to_string(static_cast<int>(quality)));
        use_globals({width, height, 1});
        ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
        Reader reader{.resource = resource::AoFinal};
        register_pass(reader_desc(reader));
        const auto graph = live_chain(ao_settings(quality));
        const FrameState state{.graph = graph};
        const auto depth = scene_depth(test::make_depth(*m_device, width, height));
        EXPECT_EQ(run(state, 1, depth), std::nullopt);
        EXPECT_EQ(run(state, 2, depth), std::nullopt);
        EXPECT_EQ(denoise_counters().dispatches, 2u);
        EXPECT_EQ(denoise_counters().temporalResets, 1u);
        EXPECT_EQ(denoise_counters().temporalValid, 1u);
        EXPECT_EQ(ao::upsample::testing::counters().draws, 2u);
        EXPECT_EQ(reader.calls, 2);
      }
    }
  }
};

TEST_F(AoDenoiseGraph, DenoiseOwnsHistory) {
  const auto graph = compile(ao_settings(Quality::High));
  const NodeId node = graph->find_node(ao::denoise::Pass);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, CullReason::None);
  EXPECT_EQ(graph->nodes[node].passKind, PassKind::Compute);
  EXPECT_EQ(graph->nodes[node].order, 1u);
  const auto uses = node_uses(*graph, node);
  ASSERT_EQ(uses.size(), 4u);
  const std::array<std::string_view, 4> names{ao::gtao::Resource, ao::hiz::Resource, resource::Motion,
                                              ao::denoise::History};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(graph->resources[uses[i].resource].name, names[i]) << i;
  }
  EXPECT_EQ(uses[1].options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(uses[3].access, Access::StorageWrite);
  const ResourceId id = graph->find_resource(ao::denoise::History);
  ASSERT_NE(id, InvalidResource);
  const CompiledResource& history = graph->resources[id];
  EXPECT_EQ(history.kind, ResourceKind::History);
  EXPECT_EQ(history.creator, node);
  EXPECT_EQ(history.tex.format, wgpu::TextureFormat::RGBA16Float);
  EXPECT_TRUE(history.textureUsage & wgpu::TextureUsage::StorageBinding);
  EXPECT_TRUE(history.textureUsage & wgpu::TextureUsage::TextureBinding);
  ASSERT_EQ(graph->histories.size(), 1u);
  EXPECT_EQ(graph->histories[0], id);
  EXPECT_NE(graph->nodes[node].pipelineIds[0], 0u);
  EXPECT_NE(graph->nodes[node].pipelineIds[1], 0u);
  EXPECT_NE(graph->nodes[node].pipelineIds[0], graph->nodes[node].pipelineIds[1]);
}

TEST_F(AoDenoiseGraph, ResetAfterInvalidate) {
  const auto graph = live_chain(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  const std::array<std::pair<bool, uint32_t>, 4> frames{{{false, 1}, {false, 1}, {true, 2}, {false, 2}}};
  uint64_t frame = 0;
  for (const auto& [invalidate, resets] : frames) {
    ++frame;
    EXPECT_EQ(run(state, frame, depth64(), false, invalidate), std::nullopt);
    EXPECT_EQ(denoise_counters().temporalResets, resets) << "frame " << frame;
  }
  EXPECT_EQ(denoise_counters().temporalValid, 2u);
  EXPECT_EQ(denoise_counters().dispatches, 4u);
}

TEST_F(AoDenoiseGraph, InvalidMotionResets) {
  const auto graph = live_chain(ao_settings(Quality::High));
  const FrameState state{.graph = graph};
  EXPECT_EQ(run(state, 1, depth64()), std::nullopt);
  EXPECT_EQ(run(state, 2, depth64()), std::nullopt);
  ASSERT_EQ(denoise_counters().temporalResets, 1u);
  auto invalid = std::make_shared<SceneHints>(*camera_hints(perspective(Near, Far)));
  invalid->motionValid = false;
  ao::testing::set_camera_override(invalid);
  StageData stage = build(state, 3, depth64());
  const uint16_t slot = graph->resources[graph->find_resource(ao::denoise::History)].historySlot;
  EXPECT_TRUE(stage.prev_valid(slot));
  EXPECT_EQ(encode(stage), std::nullopt);
  EXPECT_EQ(denoise_counters().temporalResets, 2u);
  ao::testing::set_camera_override(camera_hints(perspective(Near, Far)));
  EXPECT_EQ(run(state, 4, depth64()), std::nullopt);
  EXPECT_EQ(denoise_counters().temporalResets, 2u);
  EXPECT_EQ(denoise_counters().temporalValid, 2u);
}

TEST_F(AoDenoiseGraph, TemporalOffAlwaysResets) {
  EXPECT_NE(ao::latched_params(ao_settings(Quality::High), {.flags = ao::FlagTemporalOff}).flags & ao::FlagTemporalOff,
            0u);
  const auto graph = live_chain(ao_settings(Quality::High));
  set_chain_flags(*graph, ao::FlagTemporalOff);
  const FrameState state{.graph = graph};
  const NodeId gtao = graph->find_node(ao::gtao::Pass);
  for (uint64_t frame = 1; frame <= 4; ++frame) {
    StageData opaque = build(state, frame, depth64());
    const auto it = std::ranges::find(opaque.bindings.params, gtao, &StageBindings::Params::node);
    ASSERT_NE(it, opaque.bindings.params.end());
    ASSERT_NE(opaque.sceneHints, nullptr);
    EXPECT_EQ(params_from(std::span{it->bytes}.first(it->size)).noiseIndex,
              opaque.sceneHints->hintFrame % ao::NoisePeriod);
    EXPECT_EQ(encode(opaque), std::nullopt);
  }
  EXPECT_EQ(denoise_counters().temporalResets, 4u);
  EXPECT_EQ(denoise_counters().temporalValid, 0u);
  set_chain_flags(*graph, 0);
  EXPECT_EQ(run(state, 5, depth64()), std::nullopt);
  EXPECT_EQ(denoise_counters().temporalValid, 1u);
}

TEST_F(AoDenoiseGraph, ResetsOnGraphHashChange) {
  const auto first = live_chain(ao_settings(Quality::High));
  ao::testing::set_sink_override(true);
  const auto second = live_chain(ao_settings(Quality::High));
  ASSERT_NE(second->graphHash, first->graphHash);
  const FrameState a{.graph = first};
  const FrameState b{.graph = second};
  const std::array<std::pair<const FrameState*, uint32_t>, 5> frames{{{&a, 1}, {&a, 1}, {&b, 2}, {&b, 2}, {&a, 3}}};
  uint64_t frame = 0;
  for (const auto& [state, resets] : frames) {
    ++frame;
    latch(state->graph.get());
    EXPECT_EQ(run(*state, frame, depth64()), std::nullopt);
    EXPECT_EQ(denoise_counters().temporalResets, resets) << "frame " << frame;
  }
  EXPECT_EQ(denoise_counters().temporalValid, 2u);
}

TEST_F(AoDenoiseGraph, ResetsOnSceneSizeChange) {
  const auto full = live_chain(ao_settings(Quality::High));
  m_sceneSize = GraphExtent{640, 360};
  const auto split = live_chain(ao_settings(Quality::High));
  ASSERT_NE(split->graphHash, full->graphHash);
  const auto extent = [](const CompiledGraph& graph) {
    return graph.resources[graph.find_resource(ao::denoise::History)].extent;
  };
  EXPECT_EQ(extent(*full).width, 640u);
  EXPECT_EQ(extent(*split).width, 320u);
  EXPECT_EQ(extent(*split).height, 180u);
  const FrameState a{.graph = full};
  const FrameState b{.graph = split};
  latch(full.get());
  EXPECT_EQ(run(a, 1, depth64()), std::nullopt);
  EXPECT_EQ(run(a, 2, depth64()), std::nullopt);
  latch(split.get());
  EXPECT_EQ(run(b, 3, depth64()), std::nullopt);
  EXPECT_EQ(denoise_counters().temporalResets, 2u);
  EXPECT_EQ(run(b, 4, depth64()), std::nullopt);
  EXPECT_EQ(denoise_counters().temporalResets, 2u);
  EXPECT_EQ(denoise_counters().temporalValid, 2u);
}

TEST_F(AoDenoiseGraph, TuningKeepsHistory) {
  Settings settings = ao_settings(Quality::High);
  const auto base = live_chain(settings);
  const FrameState first{.graph = base};
  EXPECT_EQ(run(first, 1, depth64()), std::nullopt);
  EXPECT_EQ(run(first, 2, depth64()), std::nullopt);
  ASSERT_EQ(denoise_counters().temporalResets, 1u);
  uint64_t frame = 2;
  for (const auto [radius, strength] : {std::pair{1.2f, 0.5f}, std::pair{0.25f, 0.f}, std::pair{1.5f, 1.5f}}) {
    settings.aoRadius = radius;
    settings.aoStrength = strength;
    const auto tuned = live_chain(settings);
    ASSERT_EQ(tuned->graphHash, base->graphHash);
    const FrameState state{.graph = tuned};
    StageData stage = build(state, ++frame, depth64());
    const uint16_t slot = tuned->resources[tuned->find_resource(ao::denoise::History)].historySlot;
    EXPECT_TRUE(stage.prev_valid(slot));
    EXPECT_EQ(encode(stage), std::nullopt);
  }
  EXPECT_EQ(denoise_counters().temporalResets, 1u);
  EXPECT_EQ(denoise_counters().temporalValid, 4u);
}

TEST_F(AoDenoiseGraph, HistoryIsWorkingSize) {
  for (const auto [width, height] : {std::pair{2560u, 1440u}, std::pair{4779u, 2688u}, std::pair{61u, 37u}}) {
    use_globals({width, height, 1});
    for (const Quality quality : {Quality::Low, Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                   std::to_string(static_cast<int>(quality)));
      const uint32_t divisor = quality == Quality::Ultra ? 1u : 2u;
      const auto graph = compile(ao_settings(quality));
      const CompiledResource& history = graph->resources[graph->find_resource(ao::denoise::History)];
      const CompiledResource& raw = graph->resources[graph->find_resource(ao::gtao::Resource)];
      EXPECT_EQ(history.tex.divisor, divisor);
      EXPECT_EQ(history.extent.width, (width + divisor - 1) / divisor);
      EXPECT_EQ(history.extent.height, (height + divisor - 1) / divisor);
      EXPECT_EQ(history.extent.width, raw.extent.width);
      EXPECT_EQ(history.extent.height, raw.extent.height);
      EXPECT_EQ(history.mipLevels, 1u);
    }
  }
}

TEST_F(AoDenoiseGraph, ReadyFollowsPipelines) {
  install_hints();
  struct HintsGuard {
    bool installed = true;
    HintsGuard() = default;
    HintsGuard(const HintsGuard&) = delete;
    HintsGuard& operator=(const HintsGuard&) = delete;
    ~HintsGuard() {
      if (installed) {
        remove_hints();
      }
    }
  } hints;
  const Settings high = ao_settings(Quality::High);
  EXPECT_TRUE(ao::ready(Settings{}));
  ao::on_latch({.settings = &high}, nullptr);
  EXPECT_FALSE(ao::ready(high));
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
  const NodeId node = graph->find_node(ao::denoise::Pass);
  ASSERT_NE(node, InvalidNode);
  EXPECT_EQ(shaders::status({graph->nodes[node].pipelineIds[0]}), shaders::PipelineStatus::Pending);
  latch(graph.get());
  EXPECT_FALSE(ao::ready(high));
  EXPECT_TRUE(ao::ready(Settings{}));
  release = true;
  gate.wait();
  latch_chain(*graph);
  EXPECT_TRUE(ao::ready(high));
  remove_hints();
  hints.installed = false;
  EXPECT_FALSE(ao::ready(high));
  EXPECT_TRUE(ao::ready(Settings{}));
}

TEST_F(AoDenoiseGraph, EncodesValidationClean) { encode_sizes(); }

TEST_F(AoDenoiseGraph, EncodesValidationCleanWithProductionToggles) {
  TearDown();
  m_started = false;
  if (!open({.backend = wgpu::BackendType::Null, .productionToggles = true})) {
    GTEST_SKIP() << "no null device: " << m_error;
  }
  encode_sizes();
  for (const uint32_t flags : {uint32_t{ao::FlagTemporalOff}, uint32_t{ao::FlagIdentityForced},
                               uint32_t{ao::FlagNormalFromDepth | ao::FlagSectorCeil}}) {
    SCOPED_TRACE(flags);
    use_globals({61, 37, 1});
    auto hints = std::make_shared<SceneHints>(*camera_hints(perspective(Near, Far)));
    ao::testing::set_camera_override(hints);
    const auto graph = live_chain(ao_settings(Quality::High));
    set_chain_flags(*graph, flags);
    const FrameState state{.graph = graph};
    const auto depth = scene_depth(test::make_depth(*m_device, 61, 37));
    EXPECT_EQ(run(state, 1, depth), std::nullopt);
    hints->motionValid = false;
    EXPECT_EQ(run(state, 2, depth), std::nullopt);
    hints->motionValid = true;
    EXPECT_EQ(run(state, 3, depth), std::nullopt);
    EXPECT_EQ(denoise_counters().dispatches, 3u);
  }
}

using Vec3d = std::array<double, 3>;

struct Box {
  Vec3d min;
  Vec3d max;
  uint32_t tag;
};

struct CustomHit {
  double t = 0.0;
  Vec3d normal{};
  uint32_t tag = 0;
};

struct Rig {
  uint32_t width = 64;
  uint32_t height = 48;
  double eyeHeight = 300.0;
  double pitch = 0.0;
  bool ground = true;
  std::vector<Box> boxes;
  std::function<std::optional<CustomHit>(const Vec3d& eye, const Vec3d& dir)> custom;
  std::function<uint32_t(const Vec3d& point, uint32_t tag)> classify;
  std::function<uint32_t(uint32_t tag)> code;
  std::function<Vec3d(const Vec3d& normal, uint32_t tag)> shade;
};

struct Frame {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> depth;
  std::vector<uint32_t> normals;
  std::vector<uint32_t> tags;
};

constexpr uint32_t TagGround = 1;
constexpr uint32_t TagWall = 2;
constexpr uint32_t TagPole = 3;
constexpr uint32_t TagBand = 10;
constexpr uint32_t TagOpen = 11;

double dot3(const Vec3d& a, const Vec3d& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

std::optional<CustomHit> hit_box(const Box& box, const Vec3d& eye, const Vec3d& dir) {
  double t0 = -1e300;
  double t1 = 1e300;
  int axis = -1;
  double axisSign = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(dir[i]) < 1e-12) {
      if (eye[i] < box.min[i] || eye[i] > box.max[i]) {
        return std::nullopt;
      }
      continue;
    }
    double a = (box.min[i] - eye[i]) / dir[i];
    double b = (box.max[i] - eye[i]) / dir[i];
    double s = -1.0;
    if (a > b) {
      std::swap(a, b);
      s = 1.0;
    }
    if (a > t0) {
      t0 = a;
      axis = i;
      axisSign = s;
    }
    t1 = std::min(t1, b);
  }
  if (axis < 0 || t0 > t1 || t0 <= 0.0) {
    return std::nullopt;
  }
  CustomHit hit{.t = t0, .tag = box.tag};
  hit.normal[axis] = axisSign;
  return hit;
}

Frame render_rig(const Rig& rig, const Mat4x4<float>& proj) {
  Frame frame{.width = rig.width, .height = rig.height};
  const size_t count = size_t{rig.width} * rig.height;
  frame.depth.assign(count, 0.f);
  frame.normals.assign(count, ref::pack_normal({0.0, 0.0, 1.0}, 0));
  frame.tags.assign(count, 0);
  const Vec3d fwd{0.0, -std::sin(rig.pitch), std::cos(rig.pitch)};
  const Vec3d up{0.0, std::cos(rig.pitch), std::sin(rig.pitch)};
  const Vec3d right{1.0, 0.0, 0.0};
  const Vec3d eye{0.0, rig.eyeHeight, 0.0};
  for (uint32_t y = 0; y < rig.height; ++y) {
    for (uint32_t x = 0; x < rig.width; ++x) {
      const double vx = ((x + 0.5) / rig.width * 2.0 - 1.0) / proj.m0[0];
      const double vy = (1.0 - (y + 0.5) / rig.height * 2.0) / proj.m1[1];
      const Vec3d dir{right[0] * vx + up[0] * vy + fwd[0], right[1] * vx + up[1] * vy + fwd[1],
                      right[2] * vx + up[2] * vy + fwd[2]};
      std::optional<CustomHit> best;
      const auto consider = [&](const std::optional<CustomHit>& hit) {
        if (hit && hit->t > 0.0 && (!best || hit->t < best->t)) {
          best = hit;
        }
      };
      if (rig.ground && dir[1] < 0.0) {
        consider(CustomHit{.t = -eye[1] / dir[1], .normal = {0.0, 1.0, 0.0}, .tag = TagGround});
      }
      for (const Box& box : rig.boxes) {
        consider(hit_box(box, eye, dir));
      }
      if (rig.custom) {
        consider(rig.custom(eye, dir));
      }
      if (!best || best->t > 0.99 * Far) {
        continue;
      }
      const Vec3d point{eye[0] + best->t * dir[0], eye[1] + best->t * dir[1], eye[2] + best->t * dir[2]};
      const uint32_t tag = rig.classify ? rig.classify(point, best->tag) : best->tag;
      const Vec3d shaded = rig.shade ? rig.shade(best->normal, tag) : best->normal;
      const Vec3d n{dot3(shaded, right), dot3(shaded, up), dot3(shaded, fwd)};
      const size_t i = size_t{y} * rig.width + x;
      frame.depth[i] = static_cast<float>(double{proj.m2[2]} - double{proj.m2[3]} / best->t);
      frame.normals[i] = ref::pack_normal({n[0], n[1], -n[2]}, rig.code ? rig.code(tag) : 3u);
      frame.tags[i] = tag;
    }
  }
  return frame;
}

float half_to_float(uint32_t bits) {
  const uint32_t exponent = (bits >> 10) & 0x1Fu;
  const uint32_t mantissa = bits & 0x3FFu;
  const float sign = (bits & 0x8000u) != 0 ? -1.f : 1.f;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -24);
  }
  return sign * std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
}

uint16_t float_to_half(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 112;
  uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent >= 31) {
    return sign | 0x7C00u;
  }
  uint32_t shift = 13;
  uint32_t half = 0;
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    shift = static_cast<uint32_t>(14 - exponent);
  } else {
    half = static_cast<uint32_t>(exponent) << 10;
  }
  half |= mantissa >> shift;
  const uint32_t rest = mantissa & ((1u << shift) - 1);
  const uint32_t halfway = 1u << (shift - 1);
  if (rest > halfway || (rest == halfway && (half & 1u) != 0)) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

uint8_t grid_edges(const std::vector<double>& hiz, uint32_t width, uint32_t height, uint32_t x, uint32_t y) {
  const auto at = [&](int64_t px, int64_t py) {
    px = std::clamp<int64_t>(px, 0, width - 1);
    py = std::clamp<int64_t>(py, 0, height - 1);
    return hiz[static_cast<size_t>(py) * width + static_cast<size_t>(px)];
  };
  return ref::pack_edges(ref::calculate_edges(at(x, y), at(int64_t{x} - 1, y), at(int64_t{x} + 1, y),
                                              at(x, int64_t{y} - 1), at(x, int64_t{y} + 1)));
}

std::array<uint32_t, 2> history_from_raw(uint32_t raw, double z) {
  const bool receiver = ((raw >> 16) & 0xFFu) > 127;
  const auto visibility = static_cast<float>(std::min((raw & 0xFFu) / 255.0 * ao::RawScale, 1.0));
  const uint16_t r = float_to_half(visibility);
  const uint16_t g = float_to_half(receiver ? 1.f : 0.f);
  const uint16_t b = float_to_half(static_cast<float>(std::log2(z)));
  const uint16_t a = float_to_half(static_cast<float>(((raw >> 8) & 0xFFu) / 255.0));
  return {r | (uint32_t{g} << 16), b | (uint32_t{a} << 16)};
}

struct TestPass {
  std::string name;
  PassKind kind = PassKind::Copy;
  uint16_t order = 0;
  std::function<void(InternalPassBuilder&)> declare;
  std::function<void(PassContext&)> encode;
  mutable int calls = 0;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  static_cast<const TestPass*>(state)->declare(builder);
}

void encode_test_pass(PassContext& ctx, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  ++pass.calls;
  pass.encode(ctx);
}

PassDesc desc_of(const TestPass& pass) {
  return {
      .name = pass.name,
      .stage = Stage::OpaqueDone,
      .kind = pass.kind,
      .order = pass.order,
      .declare = declare_test_pass,
      .encode = encode_test_pass,
      .state = &pass,
  };
}

struct ChainResult {
  Frame frame;
  std::vector<uint32_t> raw;
  std::vector<uint32_t> final;
  uint32_t gridWidth = 0;
  uint32_t gridHeight = 0;
  uint32_t divisor = 2;
  StageCamera camera;
  ao::Params params;
  ref::CpuPyramid pyramid;
};

class AoChainGpu : public AoHizGpu {
protected:
  using Result = ChainResult;

  using RawFn = std::function<uint32_t(uint32_t x, uint32_t y, double z, uint8_t edges)>;

  void upload_normals(const Frame& frame) {
    const wgpu::TextureDescriptor descriptor{
        .label = "test normals",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = {frame.width, frame.height, 1},
        .format = wgpu::TextureFormat::RGB10A2Unorm,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = frame.width * 4, .rowsPerImage = frame.height};
    const wgpu::Extent3D size{frame.width, frame.height, 1};
    m_device->queue.WriteTexture(&destination, frame.normals.data(), frame.normals.size() * 4, &layout, &size);
    webgpu::g_normalBuffer.texture = texture;
    webgpu::g_normalBuffer.view = texture.CreateView();
    webgpu::g_normalBuffer.size = size;
    webgpu::g_normalBuffer.format = wgpu::TextureFormat::RGB10A2Unorm;
  }

  std::shared_ptr<CompiledGraph> chain(const Settings& settings) {
    std::shared_ptr<CompiledGraph> graph = compile(settings);
    for (const std::string_view name : ChainPasses) {
      const NodeId node = graph->find_node(name);
      if (node == InvalidNode || graph->nodes[node].status != CullReason::None) {
        continue;
      }
      for (const uint32_t id : graph->nodes[node].pipelineIds) {
        if (id != 0) {
          EXPECT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready) << name;
        }
      }
    }
    latch(graph.get());
    return graph;
  }

  static void edit_params(const CompiledGraph& graph, const std::function<void(ao::Params&)>& edit) {
    for (const std::string_view name : ChainPasses) {
      const NodeId node = graph.find_node(name);
      if (node == InvalidNode) {
        continue;
      }
      ao::Params params = params_from(latched_pass_params(graph.nodes[node].id));
      edit(params);
      set_pass_params(graph.nodes[node].id, std::as_bytes(std::span{&params, 1}));
    }
    latch_pass_params();
  }

  std::vector<uint32_t> texels(const wgpu::Texture& texture) const {
    std::vector<uint32_t> out;
    for (const float value : test::read_texture_mip(*m_device, texture, 0)) {
      out.push_back(std::bit_cast<uint32_t>(value));
    }
    return out;
  }

  static ao::Params cpu_params(const CompiledGraph& graph, uint64_t frameId) {
    const NodeId node = graph.find_node(ao::hiz::Pass);
    ao::Params params = params_from(latched_pass_params(graph.nodes[node].id));
    params.effectRadius = params.radiusMetres * DefaultUnitsPerMetre;
    params.thickness = params.thicknessMetres * DefaultUnitsPerMetre;
    params.noiseIndex = static_cast<uint32_t>(frameId % ao::NoisePeriod);
    return params;
  }

  Result run_rig(const Rig& rig, Quality quality, const std::function<void(ao::Params&)>& edit = {},
                 const RawFn& stubRaw = {}, uint64_t frameId = 5) {
    use_globals({rig.width, rig.height, 1});
    const Mat4x4<float> proj = perspective(Near, Far);
    const auto hints = camera_hints(proj);
    ao::testing::set_camera_override(hints);
    Result result;
    result.divisor = ao::working_divisor(quality);
    result.gridWidth = ao::grid_size(rig.width, result.divisor);
    result.gridHeight = ao::grid_size(rig.height, result.divisor);
    result.frame = render_rig(rig, proj);
    EXPECT_TRUE(fill_stage_camera(hints.get(), result.camera));
    const double effectRadius = double{ao::RadiusMetres} * DefaultUnitsPerMetre;
    result.pyramid =
        ref::build_pyramid(result.frame.depth, rig.width, rig.height, result.divisor, result.camera, effectRadius,
                           ao::hiz::written_levels(chain_levels(result.gridWidth, result.gridHeight)));
    Reader rawReader{.resource = ao::gtao::Resource};
    Reader finalReader{.resource = resource::AoFinal};
    std::vector<uint32_t> staged;
    const uint32_t rowWords = ((result.gridWidth * 8 + 255) & ~255u) / 4;
    wgpu::Buffer staging;
    TestPass stub{.name = "t.history"};
    if (stubRaw) {
      ao::testing::unregister(ao::gtao::Pass);
      ao::testing::unregister(ao::denoise::Pass);
      staged.assign(size_t{rowWords} * result.gridHeight, 0);
      result.raw.assign(size_t{result.gridWidth} * result.gridHeight, 0);
      const std::vector<double>& hiz = result.pyramid.levels[0];
      for (uint32_t y = 0; y < result.gridHeight; ++y) {
        for (uint32_t x = 0; x < result.gridWidth; ++x) {
          const double z = hiz[size_t{y} * result.gridWidth + x];
          const uint32_t value = stubRaw(x, y, z, grid_edges(hiz, result.gridWidth, result.gridHeight, x, y));
          result.raw[size_t{y} * result.gridWidth + x] = value;
          const std::array<uint32_t, 2> texel = history_from_raw(value, z);
          staged[size_t{y} * rowWords + x * 2] = texel[0];
          staged[size_t{y} * rowWords + x * 2 + 1] = texel[1];
        }
      }
      const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                              .size = staged.size() * 4};
      staging = m_device->device.CreateBuffer(&descriptor);
      m_device->queue.WriteBuffer(staging, 0, staged.data(), staged.size() * 4);
      const TexDesc desc = ao::denoise::history_desc(result.divisor);
      stub.declare = [desc](InternalPassBuilder& b) {
        b.after(phase::Ao);
        b.create(ao::denoise::History, desc, Access::CopyDst);
      };
      stub.encode = [&](PassContext& ctx) {
        const wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = rowWords * 4, .rowsPerImage = result.gridHeight},
                                             .buffer = staging};
        const wgpu::TexelCopyTextureInfo to{.texture = ctx.texture(TexRef{0})};
        const wgpu::Extent3D extent{result.gridWidth, result.gridHeight, 1};
        ctx.encoder().CopyBufferToTexture(&from, &to, &extent);
      };
      register_pass(desc_of(stub));
    } else {
      register_pass(reader_desc(rawReader));
    }
    register_pass(reader_desc(finalReader));
    upload_normals(result.frame);
    const auto graph = chain(ao_settings(quality));
    if (edit) {
      edit_params(*graph, edit);
    }
    result.params = cpu_params(*graph, frameId);
    const wgpu::Texture depth = test::make_depth(*m_device, rig.width, rig.height);
    test::fill_depth(*m_device, depth, result.frame.depth);
    const FrameState state{.graph = graph};
    EXPECT_EQ(run(state, frameId, scene_depth(depth)), std::nullopt);
    EXPECT_EQ(finalReader.calls, 1);
    if (!finalReader.texture || (!stubRaw && !rawReader.texture)) {
      ADD_FAILURE() << "the AO chain did not run";
      return result;
    }
    if (!stubRaw) {
      result.raw = texels(rawReader.texture);
    }
    result.final = texels(finalReader.texture);
    return result;
  }

  static double raw_visibility(uint32_t texel) { return std::min((texel & 0xFFu) / 255.0 * ao::RawScale, 1.0); }
  static uint32_t red(uint32_t texel) { return texel & 0xFFu; }

  static std::vector<size_t> grid_tagged(const Result& r, uint32_t tag) {
    std::vector<size_t> out;
    for (uint32_t y = 0; y < r.gridHeight; ++y) {
      for (uint32_t x = 0; x < r.gridWidth; ++x) {
        const uint32_t qx = ao::rep_texel(x, r.divisor, r.frame.width);
        const uint32_t qy = ao::rep_texel(y, r.divisor, r.frame.height);
        if (r.frame.tags[size_t{qy} * r.frame.width + qx] == tag) {
          out.push_back(size_t{y} * r.gridWidth + x);
        }
      }
    }
    return out;
  }

  static double mean_raw(const Result& r, std::span<const size_t> texels) {
    double sum = 0.0;
    for (const size_t i : texels) {
      sum += raw_visibility(r.raw[i]);
    }
    return texels.empty() ? 0.0 : sum / static_cast<double>(texels.size());
  }

  std::vector<ref::GtaoTexel> cpu_gtao(const Result& r) const {
    return ref::estimate_gtao({.depth = r.frame.depth,
                               .normals = r.frame.normals,
                               .width = r.frame.width,
                               .height = r.frame.height,
                               .divisor = r.divisor,
                               .camera = r.camera,
                               .params = r.params},
                              r.pyramid);
  }

  std::vector<double> cpu_upsample(const Result& r) const {
    return ref::upsample({.depth = r.frame.depth,
                          .normals = r.frame.normals,
                          .raw = r.raw,
                          .hiz = r.pyramid.levels[0],
                          .width = r.frame.width,
                          .height = r.frame.height,
                          .gridWidth = r.gridWidth,
                          .gridHeight = r.gridHeight,
                          .camera = r.camera,
                          .params = r.params});
  }
};

using AoGtaoGpu = AoChainGpu;
using AoUpsampleGpu = AoChainGpu;

constexpr double Deg = std::numbers::pi / 180.0;

Rig ground_rig(uint32_t width, uint32_t height, double pitch) {
  return {.width = width, .height = height, .eyeHeight = 300.0, .pitch = pitch * Deg};
}

Rig frontal_rig(uint32_t width, uint32_t height, double distance) {
  return {.width = width,
          .height = height,
          .eyeHeight = 0.0,
          .ground = false,
          .boxes = {{{-1e5, -1e5, distance}, {1e5, 1e5, distance + 100.0}, TagWall}}};
}

TEST_F(AoGtaoGpu, FlatPlaneIsOpen) {
  for (const Quality quality : {Quality::High, Quality::Ultra}) {
    const Result r = run_rig(ground_rig(320, 240, 30.0), quality);
    const std::vector<size_t> ground = grid_tagged(r, TagGround);
    ASSERT_GT(ground.size(), r.raw.size() / 3);
    size_t open = 0;
    size_t exact = 0;
    for (const size_t i : ground) {
      open += raw_visibility(r.raw[i]) >= 0.97 ? 1 : 0;
      exact += red(r.raw[i]) >= 170 ? 1 : 0;
    }
    EXPECT_GE(open, ground.size() * 99 / 100) << static_cast<int>(quality);
    EXPECT_GE(exact, ground.size() * 95 / 100) << static_cast<int>(quality);
  }
}

TEST_F(AoGtaoGpu, BoxCreaseIsDark) {
  Rig rig{.width = 320, .height = 240, .eyeHeight = 250.0, .pitch = 15.0 * Deg};
  rig.boxes = {{{-300.0, 0.0, 700.0}, {300.0, 400.0, 760.0}, TagWall}};
  rig.classify = [](const Vec3d& p, uint32_t tag) {
    if (std::abs(p[0]) < 200.0 && ((tag == TagGround && p[2] > 688.0) || (tag == TagWall && p[1] < 12.0))) {
      return TagBand;
    }
    return tag == TagGround && p[2] < 500.0 && std::abs(p[0]) < 200.0 ? TagOpen : tag;
  };
  const Result r = run_rig(rig, Quality::High);
  const std::vector<size_t> crease = grid_tagged(r, TagBand);
  const std::vector<size_t> open = grid_tagged(r, TagOpen);
  ASSERT_GT(crease.size(), 20u);
  ASSERT_GT(open.size(), 100u);
  const double creaseMean = mean_raw(r, crease);
  std::printf("box crease mean %.4f over %zu texels, open %.4f\n", creaseMean, crease.size(), mean_raw(r, open));
  EXPECT_LE(creaseMean, 0.75);
  EXPECT_GE(mean_raw(r, open), 0.97);
}

TEST_F(AoGtaoGpu, ThinPoleHaloBounded) {
  Rig rig{.width = 320, .height = 240, .eyeHeight = 200.0};
  rig.boxes = {{{-300.0, 0.0, 300.0}, {300.0, 400.0, 340.0}, TagWall},
               {{-10.0, 0.0, 260.0}, {10.0, 300.0, 280.0}, TagPole}};
  rig.classify = [](const Vec3d& p, uint32_t tag) {
    return tag == TagWall && p[1] > 120.0 && p[1] < 260.0 ? (std::abs(p[0]) > 150.0 ? TagOpen : TagBand) : tag;
  };
  std::array<double, 2> strips{};
  std::array<double, 2> interiors{};
  for (size_t run = 0; run < 2; ++run) {
    const float thickness = run == 0 ? ao::ThicknessMetres : 1e6f;
    const Result r = run_rig(rig, Quality::High, [thickness](ao::Params& p) { p.thicknessMetres = thickness; });
    double strip = 0.0;
    double interior = 0.0;
    size_t stripCount = 0;
    size_t interiorCount = 0;
    for (uint32_t y = 0; y < r.frame.height; ++y) {
      int64_t left = -1;
      int64_t right = -1;
      for (uint32_t x = 0; x < r.frame.width; ++x) {
        if (r.frame.tags[size_t{y} * r.frame.width + x] == TagPole) {
          left = left < 0 ? x : left;
          right = x;
        }
      }
      for (uint32_t x = 0; x < r.frame.width; ++x) {
        const size_t i = size_t{y} * r.frame.width + x;
        const double value = red(r.final[i]) / 255.0;
        const int64_t gap = std::min(std::abs(int64_t{x} - left), std::abs(int64_t{x} - right));
        if (r.frame.tags[i] == TagBand && left >= 0 && gap >= 2 && gap <= 6) {
          strip += value;
          ++stripCount;
        } else if (r.frame.tags[i] == TagOpen) {
          interior += value;
          ++interiorCount;
        }
      }
    }
    ASSERT_GT(stripCount, 50u);
    ASSERT_GT(interiorCount, 100u);
    strips[run] = strip / stripCount;
    interiors[run] = interior / interiorCount;
  }
  std::printf("pole strip %.4f (interior %.4f) at 0.25 m, %.4f at 1e6 m\n", strips[0], interiors[0], strips[1]);
  EXPECT_GE(strips[0], 0.9);
  EXPECT_LE(strips[1], 0.8);
}

TEST_F(AoGtaoGpu, TiltedNormalsMatchContinuousCoverage) {
  Rig rig = frontal_rig(256, 192, 500.0);
  const double tilt = 15.0 * Deg;
  rig.shade = [tilt](const Vec3d& n, uint32_t) { return Vec3d{n[0], std::sin(tilt), n[2] * std::cos(tilt)}; };
  std::array<double, 2> means{};
  double continuous = 0.0;
  for (size_t run = 0; run < 2; ++run) {
    const Result r =
        run_rig(rig, Quality::High, [run](ao::Params& p) { p.flags |= run == 1 ? uint32_t{ao::FlagSectorCeil} : 0u; });
    const std::vector<ref::GtaoTexel> cpu = cpu_gtao(r);
    double gpu = 0.0;
    double cont = 0.0;
    size_t count = 0;
    for (uint32_t y = 4; y + 4 < r.gridHeight; ++y) {
      for (uint32_t x = 4; x + 4 < r.gridWidth; ++x) {
        const size_t i = size_t{y} * r.gridWidth + x;
        gpu += raw_visibility(r.raw[i]);
        cont += cpu[i].continuous;
        ++count;
      }
    }
    means[run] = gpu / count;
    continuous = cont / count;
  }
  std::printf("tilted interior: round %.4f, ceil %.4f, continuous %.4f\n", means[0], means[1], continuous);
  EXPECT_NEAR(means[0], continuous, 0.006);
}

TEST_F(AoGtaoGpu, FacetedSmoothNormalsStayOpen) {
  Rig rig{.width = 256, .height = 192, .eyeHeight = 300.0, .ground = false};
  constexpr double Radius = 400.0;
  constexpr double Facet = 10.0 * Deg;
  const Vec3d centre{0.0, 300.0, 900.0};
  rig.custom = [centre](const Vec3d& eye, const Vec3d& dir) -> std::optional<CustomHit> {
    const double apothem = Radius * std::cos(Facet / 2.0);
    double enter = -1e300;
    double exit = 1e300;
    for (int k = 0; k < 36; ++k) {
      const double a = (k + 0.5) * Facet;
      const Vec3d n{0.0, std::sin(a), -std::cos(a)};
      const double denom = dot3(n, dir);
      const double dist = apothem - dot3(n, {eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]});
      if (std::abs(denom) < 1e-12) {
        if (dist < 0.0) {
          return std::nullopt;
        }
        continue;
      }
      const double t = dist / denom;
      if (denom < 0.0 && t > enter) {
        enter = t;
      } else if (denom > 0.0) {
        exit = std::min(exit, t);
      }
    }
    if (enter > exit || enter <= 0.0) {
      return std::nullopt;
    }
    const Vec3d p{eye[0] + enter * dir[0], eye[1] + enter * dir[1], eye[2] + enter * dir[2]};
    const double smooth = std::atan2(p[1] - centre[1], centre[2] - p[2]);
    return CustomHit{.t = enter, .normal = {0.0, std::sin(smooth), -std::cos(smooth)}, .tag = TagWall};
  };
  rig.classify = [centre](const Vec3d& p, uint32_t tag) {
    return std::abs(p[1] - centre[1]) < 0.6 * Radius && std::abs(p[0]) < 300.0 ? TagBand : tag;
  };
  const Result r = run_rig(rig, Quality::High);
  const std::vector<size_t> surface = grid_tagged(r, TagBand);
  ASSERT_GT(surface.size(), 500u);
  size_t exact = 0;
  for (const size_t i : surface) {
    exact += red(r.raw[i]) >= 170 ? 1 : 0;
  }
  const double mean = mean_raw(r, surface);
  std::printf("faceted surface mean %.4f, exactly open %.4f\n", mean, static_cast<double>(exact) / surface.size());
  EXPECT_GE(mean, 0.97);
  EXPECT_GE(exact, surface.size() * 3 / 4);
}

TEST_F(AoGtaoGpu, FloorHoldsInDeepSlot) {
  Rig slot{.width = 320, .height = 240, .eyeHeight = 700.0, .pitch = 60.0 * Deg};
  slot.boxes = {{{-600.0, 0.0, 200.0}, {-15.0, 400.0, 900.0}, TagWall},
                {{15.0, 0.0, 200.0}, {600.0, 400.0, 900.0}, TagWall}};
  const Result deep = run_rig(slot, Quality::High, [](ao::Params& p) { p.thicknessMetres = 1e6f; });
  for (size_t i = 0; i < deep.raw.size(); ++i) {
    if (((deep.raw[i] >> 16) & 0xFFu) != 0) {
      EXPECT_GE(red(deep.raw[i]), 34u) << i;
    }
  }
  for (const uint32_t texel : deep.final) {
    EXPECT_GE(red(texel), 51u);
  }
  Rig crease{.width = 320, .height = 240, .eyeHeight = 250.0, .pitch = 15.0 * Deg};
  crease.boxes = {{{-300.0, 0.0, 700.0}, {300.0, 400.0, 760.0}, TagWall}};
  constexpr float Floor = 0.75f;
  const Result floored = run_rig(crease, Quality::High, [](ao::Params& p) { p.minVisibility = Floor; });
  size_t atFloor = 0;
  for (const uint32_t texel : floored.raw) {
    if (((texel >> 16) & 0xFFu) != 0) {
      EXPECT_GE(red(texel), 127u);
      atFloor += red(texel) <= 128u ? 1 : 0;
    }
  }
  EXPECT_GT(atFloor, 10u);
  for (size_t i = 0; i < floored.final.size(); ++i) {
    if (floored.frame.depth[i] != 0.f) {
      EXPECT_GE(red(floored.final[i]), 190u) << i;
    }
  }
}

TEST_F(AoGtaoGpu, CodeZeroAndSkyWriteOne) {
  Rig rig = ground_rig(128, 96, 10.0);
  rig.boxes = {{{-100.0, 0.0, 600.0}, {100.0, 150.0, 650.0}, TagWall}};
  rig.code = [](uint32_t tag) { return tag == TagWall ? 0u : 3u; };
  const Result r = run_rig(rig, Quality::High);
  size_t sky = 0;
  size_t zero = 0;
  for (uint32_t y = 0; y < r.gridHeight; ++y) {
    for (uint32_t x = 0; x < r.gridWidth; ++x) {
      const size_t i = size_t{y} * r.gridWidth + x;
      const uint32_t tag =
          r.frame
              .tags[size_t{ao::rep_texel(y, 2, r.frame.height)} * r.frame.width + ao::rep_texel(x, 2, r.frame.width)];
      if (tag == 0 || tag == TagWall) {
        EXPECT_EQ(red(r.raw[i]), 170u) << x << "," << y;
        EXPECT_EQ((r.raw[i] >> 16) & 0xFFu, 0u) << x << "," << y;
        (tag == 0 ? sky : zero) += 1;
      }
    }
  }
  EXPECT_GT(sky, 10u);
  EXPECT_GT(zero, 10u);
  for (size_t i = 0; i < r.final.size(); ++i) {
    if (r.frame.tags[i] == 0 || r.frame.tags[i] == TagWall) {
      EXPECT_EQ(red(r.final[i]), 255u) << i;
    }
    if (r.frame.tags[i] == 0) {
      EXPECT_EQ(r.final[i] & 0xFFFF00u, 0u) << i;
    }
  }
}

TEST_F(AoGtaoGpu, DistantSurfaceWritesOne) {
  const Result r = run_rig(frontal_rig(128, 96, 40000.0), Quality::High);
  for (const uint32_t texel : r.raw) {
    EXPECT_EQ(red(texel), 170u);
    EXPECT_EQ((texel >> 16) & 0xFFu, 255u);
  }
  for (const uint32_t texel : r.final) {
    EXPECT_EQ(red(texel), 255u);
  }
}

TEST_F(AoGtaoGpu, DepthNormalsMatchOnPlane) {
  Rig rig = ground_rig(256, 192, 35.0);
  rig.boxes = {{{-150.0, 0.0, 600.0}, {150.0, 200.0, 700.0}, TagWall}};
  const Result gbuffer = run_rig(rig, Quality::High);
  const Result depth = run_rig(rig, Quality::High, [](ao::Params& p) { p.flags |= ao::FlagNormalFromDepth; });
  const std::vector<size_t> ground = grid_tagged(gbuffer, TagGround);
  ASSERT_GT(ground.size(), 1000u);
  double delta = 0.0;
  for (const size_t i : ground) {
    delta += std::abs(raw_visibility(gbuffer.raw[i]) - raw_visibility(depth.raw[i]));
  }
  EXPECT_LE(delta / ground.size(), 0.02);
}

TEST_F(AoGtaoGpu, MatchesCpuReference) {
  Rig rig{.width = 256, .height = 192, .eyeHeight = 260.0, .pitch = 20.0 * Deg};
  rig.boxes = {{{-250.0, 0.0, 650.0}, {50.0, 300.0, 720.0}, TagWall},
               {{90.0, 0.0, 560.0}, {130.0, 90.0, 600.0}, TagWall},
               {{-5.0, 0.0, 600.0}, {5.0, 220.0, 610.0}, TagPole}};
  for (const Quality quality : {Quality::Medium, Quality::High, Quality::Ultra}) {
    const Result r = run_rig(rig, quality);
    const std::vector<ref::GtaoTexel> cpu = cpu_gtao(r);
    double delta = 0.0;
    size_t count = 0;
    size_t edgeMismatches = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
      EXPECT_EQ(((r.raw[i] >> 16) & 0xFFu) != 0, cpu[i].receiver) << i;
      edgeMismatches += ((r.raw[i] >> 8) & 0xFFu) != cpu[i].edges ? 1 : 0;
      if (cpu[i].receiver) {
        delta += std::abs(raw_visibility(r.raw[i]) - cpu[i].visibility);
        ++count;
      }
    }
    ASSERT_GT(count, 1000u);
    EXPECT_LE(edgeMismatches, cpu.size() / 100);
    std::printf("quality %d: mean |gpu - cpu| %.5f over %zu texels\n", static_cast<int>(quality), delta / count, count);
    EXPECT_LE(delta / count, 0.02);
  }
}

TEST_F(AoGtaoGpu, OddSizesValidationClean) {
  for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{3u, 5u}, std::pair{61u, 37u}}) {
    for (const Quality quality : {Quality::Low, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
      const Result r = run_rig(ground_rig(width, height, 25.0), quality);
      EXPECT_EQ(r.raw.size(), size_t{r.gridWidth} * r.gridHeight);
      EXPECT_EQ(r.final.size(), size_t{width} * height);
    }
  }
}

uint32_t raw_texel(uint32_t r, uint8_t edges, bool receiver) {
  return r | (uint32_t{edges} << 8) | (receiver ? 0xFF0000u : 0u);
}

void expect_matches_mirror(const ChainResult& r, const std::vector<double>& want, std::string_view what) {
  size_t bad = 0;
  size_t first = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const auto expected = static_cast<int>(std::lround(want[i] * 255.0));
    if (std::abs(static_cast<int>(r.final[i] & 0xFFu) - expected) > 1) {
      first = bad == 0 ? i : first;
      ++bad;
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": first mismatch at " << first << " got " << (r.final[first] & 0xFFu) << " want "
                     << want[first] * 255.0;
}

TEST_F(AoUpsampleGpu, AllOnesStaysExact) {
  for (const Quality quality : {Quality::High, Quality::Ultra}) {
    const Result r = run_rig(ground_rig(64, 48, 25.0), quality, {},
                             [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(170, e, true); });
    for (const uint32_t texel : r.final) {
      EXPECT_EQ(texel & 0xFFu, 255u);
    }
  }
}

TEST_F(AoUpsampleGpu, IdentityWritesWhite) {
  const Result r = run_rig(
      frontal_rig(64, 48, 500.0), Quality::High, [](ao::Params& p) { p.flags |= ao::FlagIdentityForced; },
      [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(40, e, true); });
  for (const uint32_t texel : r.final) {
    EXPECT_EQ(texel & 0xFFu, 255u);
  }
}

TEST_F(AoUpsampleGpu, SkyWritesOne) {
  const Result r = run_rig(ground_rig(64, 48, 10.0), Quality::High, {},
                           [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(40, e, true); });
  size_t sky = 0;
  size_t dark = 0;
  for (size_t i = 0; i < r.final.size(); ++i) {
    if (r.frame.depth[i] == 0.f) {
      EXPECT_EQ(r.final[i] & 0xFFFFFFu, 0xFFu) << i;
      ++sky;
    } else {
      dark += (r.final[i] & 0xFFu) < 255u ? 1 : 0;
    }
  }
  EXPECT_GT(sky, 100u);
  EXPECT_GE(dark, (r.final.size() - sky) * 3 / 4);
}

TEST_F(AoUpsampleGpu, CodeZeroPixelWritesOne) {
  Rig rig = frontal_rig(64, 48, 500.0);
  rig.classify = [](const Vec3d& p, uint32_t tag) {
    return std::abs(p[0]) < 20.0 && std::abs(p[1]) < 20.0 ? TagBand : tag;
  };
  rig.code = [](uint32_t tag) { return tag == TagBand ? 0u : 3u; };
  const Result r =
      run_rig(rig, Quality::High, {}, [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(60, e, true); });
  size_t zero = 0;
  for (size_t i = 0; i < r.final.size(); ++i) {
    if (r.frame.tags[i] == TagBand) {
      EXPECT_EQ(r.final[i] & 0xFFu, 255u) << i;
      ++zero;
    } else {
      EXPECT_LT(r.final[i] & 0xFFu, 255u) << i;
    }
  }
  EXPECT_GT(zero, 4u);
  expect_matches_mirror(r, cpu_upsample(r), "code zero");
}

TEST_F(AoUpsampleGpu, DepthStepLeakBounded) {
  for (const double step : {0.05, 0.15}) {
    for (const Quality quality : {Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(step) + " q" + std::to_string(static_cast<int>(quality)));
      Rig rig{.width = 64, .height = 48, .eyeHeight = 0.0, .ground = false};
      rig.boxes = {{{-1e5, -1e5, 500.0}, {0.0, 1e5, 600.0}, TagWall},
                   {{0.0, -1e5, 500.0 * (1.0 + step)}, {1e5, 1e5, 700.0}, TagOpen}};
      const Result r = run_rig(rig, quality, {}, [&](uint32_t x, uint32_t, double, uint8_t e) {
        return raw_texel(ao::rep_texel(x, ao::working_divisor(quality), 64) < 32 ? 60u : 170u, e, true);
      });
      const uint32_t d = ao::working_divisor(quality);
      for (uint32_t y = 0; y < r.frame.height; ++y) {
        for (uint32_t x = 32 + d; x < r.frame.width; ++x) {
          EXPECT_GE(r.final[size_t{y} * r.frame.width + x] & 0xFFu, 254u) << x << "," << y;
        }
      }
      expect_matches_mirror(r, cpu_upsample(r), "depth step");
    }
  }
}

TEST_F(AoUpsampleGpu, UnrepresentedSurfaceWritesOne) {
  Rig rig = frontal_rig(64, 48, 500.0);
  const double column = 20.0;
  rig.custom = [column](const Vec3d&, const Vec3d& dir) -> std::optional<CustomHit> {
    if (std::floor((dir[0] * 1.2990381 + 1.0) * 32.0) != column) {
      return std::nullopt;
    }
    return CustomHit{.t = 250.0, .normal = {0.0, 0.0, -1.0}, .tag = TagPole};
  };

  const Result r =
      run_rig(rig, Quality::High, {}, [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(60, e, true); });
  size_t strip = 0;
  for (size_t i = 0; i < r.final.size(); ++i) {
    if (r.frame.tags[i] == TagPole) {
      EXPECT_EQ(i % r.frame.width % 2, 0u);
      EXPECT_EQ(r.final[i] & 0xFFu, 255u) << i;
      ++strip;
    } else {
      EXPECT_LT(r.final[i] & 0xFFu, 255u) << i;
    }
  }
  EXPECT_EQ(strip, size_t{r.frame.height});
}

TEST_F(AoUpsampleGpu, NonReceiverTapsExcluded) {
  const Result r =
      run_rig(frontal_rig(64, 48, 500.0), Quality::High, {}, [](uint32_t x, uint32_t y, double, uint8_t e) {
        const bool receiver = (x + y) % 2 == 0;
        return raw_texel(receiver ? 170u : 10u, e, receiver);
      });
  for (const uint32_t texel : r.final) {
    EXPECT_EQ(texel & 0xFFu, 255u);
  }
}

TEST_F(AoUpsampleGpu, DepthKeyRoundTrips) {
  const Result r = run_rig(ground_rig(64, 48, 20.0), Quality::High, {},
                           [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(100, e, true); });
  size_t checked = 0;
  for (size_t i = 0; i < r.final.size(); ++i) {
    const uint32_t key = (r.final[i] >> 8) & 0xFFFFu;
    const float depth = r.frame.depth[i];
    if (depth == 0.f) {
      EXPECT_EQ(key, 0u) << i;
      continue;
    }
    EXPECT_NEAR(half_to_float(key), depth, depth * 0x1p-10f + 6e-8f) << i;
    EXPECT_EQ(r.final[i] >> 24, 0u) << i;
    ++checked;
  }
  EXPECT_GT(checked, 1000u);
}

TEST_F(AoUpsampleGpu, StrengthZeroWritesOne) {
  const Result r = run_rig(
      frontal_rig(64, 48, 500.0), Quality::High, [](ao::Params& p) { p.strength = 0.f; },
      [](uint32_t, uint32_t, double, uint8_t e) { return raw_texel(34, e, true); });
  for (const uint32_t texel : r.final) {
    EXPECT_EQ(texel & 0xFFu, 255u);
  }
}

TEST_F(AoUpsampleGpu, StrengthMatchesMirror) {
  for (const float strength : {0.5f, 1.5f}) {
    std::mt19937 rng{0x5A17u};
    std::uniform_int_distribution<uint32_t> dist(34, 170);
    std::vector<uint32_t> values(64 * 48);
    for (uint32_t& value : values) {
      value = dist(rng);
    }
    Rig rig = ground_rig(64, 48, 30.0);
    rig.boxes = {{{-60.0, 0.0, 700.0}, {60.0, 120.0, 760.0}, TagWall}};
    const Result r = run_rig(
        rig, Quality::High, [strength](ao::Params& p) { p.strength = strength; },
        [&](uint32_t x, uint32_t y, double, uint8_t e) { return raw_texel(values[size_t{y} * 64 + x], e, true); });
    expect_matches_mirror(r, cpu_upsample(r), strength < 1.f ? "strength 0.5" : "strength 1.5");
    for (const uint32_t texel : r.final) {
      EXPECT_GE(texel & 0xFFu, 51u);
    }
  }
}

TEST_F(AoUpsampleGpu, OddSizesCovered) {
  for (const auto [width, height] : {std::pair{3u, 5u}, std::pair{61u, 37u}}) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const Result r =
        run_rig(ground_rig(width, height, 15.0), Quality::High, {}, [](uint32_t x, uint32_t y, double, uint8_t e) {
          return raw_texel(40 + (x * 7 + y * 13) % 130, e, true);
        });
    ASSERT_EQ(r.final.size(), size_t{width} * height);
    for (size_t i = 0; i < r.final.size(); ++i) {
      const uint32_t key = (r.final[i] >> 8) & 0xFFFFu;
      EXPECT_EQ(key != 0, r.frame.depth[i] != 0.f) << i;
    }
    expect_matches_mirror(r, cpu_upsample(r), "odd size");
  }
}

struct HistoryTexel {
  float ao = 0.f;
  float count = 0.f;
  float logDepth = 0.f;
  float edges = 0.f;
};

uint32_t raw_of(double visibility, uint8_t edges, bool receiver = true) {
  const long r = std::clamp(std::lround(visibility / ao::RawScale * 255.0), 0L, 255L);
  return raw_texel(static_cast<uint32_t>(r), edges, receiver);
}

uint32_t packed_edges(const HistoryTexel& texel) { return static_cast<uint32_t>(texel.edges * 255.5f); }

class AoDenoiseGpu : public AoChainGpu {
protected:
  using RawFn = std::function<uint32_t(uint32_t x, uint32_t y, uint8_t edges)>;
  using MotionFn = std::function<std::array<float, 2>(uint32_t x, uint32_t y)>;

  void begin(const Rig& rig, Quality quality = Quality::High) {
    use_globals({rig.width, rig.height, 1});
    m_proj = perspective(Near, Far);
    m_hints = camera_hints(m_proj);
    ao::testing::set_camera_override(m_hints);
    m_divisor = ao::working_divisor(quality);
    m_gridWidth = ao::grid_size(rig.width, m_divisor);
    m_gridHeight = ao::grid_size(rig.height, m_divisor);
    m_scene = render_rig(rig, m_proj);
    upload_normals(m_scene);
    set_motion({});
    m_depth = test::make_depth(*m_device, rig.width, rig.height);
    set_depth(m_scene.depth);
    m_rowWords = ((m_gridWidth * 4 + 255) & ~255u) / 4;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = uint64_t{m_rowWords} * m_gridHeight * 4};
    m_staging = m_device->device.CreateBuffer(&descriptor);
    ao::testing::unregister(ao::gtao::Pass);
    const TexDesc desc = ao::gtao::raw_desc(m_divisor);
    m_stub = TestPass{.name = "t.raw"};
    m_stub.declare = [desc](InternalPassBuilder& b) {
      b.after(phase::Ao);
      b.create(ao::gtao::Resource, desc, Access::CopyDst);
    };
    m_stub.encode = [this](PassContext& ctx) {
      const wgpu::TexelCopyBufferInfo from{.layout = {.bytesPerRow = m_rowWords * 4, .rowsPerImage = m_gridHeight},
                                           .buffer = m_staging};
      const wgpu::TexelCopyTextureInfo to{.texture = ctx.texture(TexRef{0})};
      const wgpu::Extent3D extent{m_gridWidth, m_gridHeight, 1};
      ctx.encoder().CopyBufferToTexture(&from, &to, &extent);
    };
    register_pass(desc_of(m_stub));
    m_reader = Reader{.resource = ao::denoise::History};
    register_pass(reader_desc(m_reader));
    m_graph = chain(ao_settings(quality));
  }

  void set_depth(const std::vector<float>& depth) {
    test::fill_depth(*m_device, m_depth, depth);
    StageCamera camera;
    ASSERT_TRUE(fill_stage_camera(m_hints.get(), camera));
    m_hiz = ref::build_pyramid(depth, m_scene.width, m_scene.height, m_divisor, camera,
                               double{ao::RadiusMetres} * DefaultUnitsPerMetre, 1)
                .levels[0];
  }

  void set_motion(const MotionFn& motion) {
    const uint32_t width = m_scene.width;
    const uint32_t height = m_scene.height;
    std::vector<uint32_t> texels(size_t{width} * height, 0);
    if (motion) {
      for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
          const std::array<float, 2> m = motion(x, y);
          texels[size_t{y} * width + x] = float_to_half(m[0]) | (uint32_t{float_to_half(m[1])} << 16);
        }
      }
    }
    const wgpu::TextureDescriptor descriptor{
        .label = "test motion",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
        .size = {width, height, 1},
        .format = webgpu::MotionBufferFormat,
    };
    const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = texture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * 4, .rowsPerImage = height};
    const wgpu::Extent3D size{width, height, 1};
    m_device->queue.WriteTexture(&destination, texels.data(), texels.size() * 4, &layout, &size);
    webgpu::g_motionBuffer.texture = texture;
    webgpu::g_motionBuffer.view = texture.CreateView();
    webgpu::g_motionBuffer.size = size;
    webgpu::g_motionBuffer.format = webgpu::MotionBufferFormat;
  }

  std::vector<HistoryTexel> step(uint64_t frameId, const RawFn& raw) {
    std::vector<uint32_t> staged(size_t{m_rowWords} * m_gridHeight, 0);
    m_raw.assign(size_t{m_gridWidth} * m_gridHeight, 0);
    for (uint32_t y = 0; y < m_gridHeight; ++y) {
      for (uint32_t x = 0; x < m_gridWidth; ++x) {
        const uint32_t value = raw(x, y, grid_edges(m_hiz, m_gridWidth, m_gridHeight, x, y));
        m_raw[size_t{y} * m_gridWidth + x] = value;
        staged[size_t{y} * m_rowWords + x] = value;
      }
    }
    m_device->queue.WriteBuffer(m_staging, 0, staged.data(), staged.size() * 4);
    const FrameState state{.graph = m_graph};
    EXPECT_EQ(run(state, frameId, scene_depth(m_depth)), std::nullopt);
    if (!m_reader.texture) {
      ADD_FAILURE() << "the denoiser did not run";
      return {};
    }
    return read_history(m_reader.texture);
  }

  std::vector<HistoryTexel> read_history(const wgpu::Texture& texture) const {
    const uint32_t width = texture.GetWidth();
    const uint32_t height = texture.GetHeight();
    const uint32_t rowBytes = (width * 8 + 255) & ~255u;
    const uint64_t size = uint64_t{rowBytes} * height;
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                            .size = size};
    const wgpu::Buffer buffer = m_device->device.CreateBuffer(&descriptor);
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = rowBytes, .rowsPerImage = height},
                                                .buffer = buffer};
    const wgpu::Extent3D extent{width, height, 1};
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&source, &destination, &extent);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    const std::optional<std::vector<uint32_t>> words = test::read_buffer(*m_device, buffer, 0, size);
    if (!words) {
      ADD_FAILURE() << "history readback failed";
      return {};
    }
    std::vector<HistoryTexel> out(size_t{width} * height);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const size_t at = size_t{y} * (rowBytes / 4) + size_t{x} * 2;
        const uint32_t rg = (*words)[at];
        const uint32_t ba = (*words)[at + 1];
        out[size_t{y} * width + x] = {half_to_float(rg & 0xFFFFu), half_to_float(rg >> 16), half_to_float(ba & 0xFFFFu),
                                      half_to_float(ba >> 16)};
      }
    }
    return out;
  }

  std::vector<double> spatial(const std::vector<uint32_t>& raw) const {
    const auto at = [&](int64_t x, int64_t y) {
      x = std::clamp<int64_t>(x, 0, m_gridWidth - 1);
      y = std::clamp<int64_t>(y, 0, m_gridHeight - 1);
      return raw[static_cast<size_t>(y) * m_gridWidth + static_cast<size_t>(x)];
    };
    const auto visibility = [](uint32_t t) { return std::min((t & 0xFFu) / 255.0 * ao::RawScale, 1.0); };
    const auto receiver = [](uint32_t t) { return ((t >> 16) & 0xFFu) > 127; };
    const auto edges = [](uint32_t t) { return ref::unpack_edges(static_cast<uint8_t>(t >> 8)); };
    std::vector<double> out(raw.size(), 1.0);
    for (uint32_t y = 0; y < m_gridHeight; ++y) {
      for (uint32_t x = 0; x < m_gridWidth; ++x) {
        const int64_t ix = x;
        const int64_t iy = y;
        if (!receiver(at(ix, iy))) {
          continue;
        }
        const auto l = edges(at(ix - 1, iy));
        const auto r = edges(at(ix + 1, iy));
        const auto t = edges(at(ix, iy - 1));
        const auto b = edges(at(ix, iy + 1));
        auto c = edges(at(ix, iy));
        c = {c[0] * l[1], c[1] * r[0], c[2] * t[3], c[3] * b[2]};
        const double leak = std::clamp(1.5 - (c[0] + c[1] + c[2] + c[3]), 0.0, 1.0) / 1.5 * 0.5;
        for (double& side : c) {
          side = std::clamp(side + leak, 0.0, 1.0);
        }
        const std::array<double, 9> weights{0.425 * (c[0] * l[2] + c[2] * t[0]),
                                            c[2],
                                            0.425 * (c[2] * t[1] + c[1] * r[2]),
                                            c[0],
                                            double{ao::DenoiseBeta},
                                            c[1],
                                            0.425 * (c[3] * b[0] + c[0] * l[3]),
                                            c[3],
                                            0.425 * (c[1] * r[3] + c[3] * b[1])};
        double sum = 0.0;
        double weightSum = 0.0;
        bool open = true;
        for (int i = 0; i < 9; ++i) {
          const uint32_t tap = at(ix + i % 3 - 1, iy + i / 3 - 1);
          if (!receiver(tap) || weights[i] <= 0.0) {
            continue;
          }
          sum += weights[i] * visibility(tap);
          weightSum += weights[i];
          open = open && visibility(tap) >= 1.0;
        }
        out[size_t{y} * m_gridWidth + x] = open ? 1.0 : sum / weightSum;
      }
    }
    return out;
  }

  [[nodiscard]] std::vector<float> depth_with_block(double distance, uint32_t x0, uint32_t y0, uint32_t x1,
                                                    uint32_t y1) const {
    std::vector<float> depth = m_scene.depth;
    const auto stored = static_cast<float>(double{m_proj.m2[2]} - double{m_proj.m2[3]} / distance);
    for (uint32_t y = y0; y < y1; ++y) {
      for (uint32_t x = x0; x < x1; ++x) {
        depth[size_t{y} * m_scene.width + x] = stored;
      }
    }
    return depth;
  }

  [[nodiscard]] uint32_t rep(uint32_t p, uint32_t full) const { return ao::rep_texel(p, m_divisor, full); }

  Mat4x4<float> m_proj{};
  std::shared_ptr<const SceneHints> m_hints;
  uint32_t m_divisor = 2;
  uint32_t m_gridWidth = 0;
  uint32_t m_gridHeight = 0;
  uint32_t m_rowWords = 0;
  Frame m_scene;
  wgpu::Texture m_depth;
  wgpu::Buffer m_staging;
  std::vector<double> m_hiz;
  std::vector<uint32_t> m_raw;
  TestPass m_stub;
  Reader m_reader;
  std::shared_ptr<CompiledGraph> m_graph;
};

TEST_F(AoDenoiseGpu, ResetWritesSpatialResult) {
  begin(frontal_rig(64, 48, 500.0));
  std::mt19937 rng{0xD3A0u};
  std::uniform_real_distribution<double> unit(0.2, 1.0);
  std::vector<double> values(size_t{m_gridWidth} * m_gridHeight);
  for (double& value : values) {
    value = unit(rng);
  }
  const auto history = step(1, [&](uint32_t x, uint32_t y, uint8_t e) {
    const size_t i = size_t{y} * m_gridWidth + x;
    return raw_of(values[i], e, i % 7 != 0);
  });
  ASSERT_EQ(history.size(), m_raw.size());
  const std::vector<double> want = spatial(m_raw);
  for (size_t i = 0; i < history.size(); ++i) {
    const bool receiver = i % 7 != 0;
    EXPECT_NEAR(history[i].ao, receiver ? want[i] : 1.0, 2e-3) << i;
    EXPECT_EQ(history[i].count, receiver ? 1.f : 0.f) << i;
    EXPECT_NEAR(history[i].logDepth, std::log2(m_hiz[i]), 5e-3) << i;
    EXPECT_EQ(packed_edges(history[i]), (m_raw[i] >> 8) & 0xFFu) << i;
  }
  EXPECT_EQ(ao::denoise::testing::counters().temporalResets, 1u);
}

TEST_F(AoDenoiseGpu, StaticInputConverges) {
  begin(frontal_rig(64, 48, 500.0));
  std::mt19937 rng{0x57A7u};
  std::uniform_real_distribution<double> unit(0.3, 1.0);
  std::vector<double> values(size_t{m_gridWidth} * m_gridHeight);
  for (double& value : values) {
    value = unit(rng);
  }
  std::vector<HistoryTexel> history;
  for (uint64_t frame = 1; frame <= 20; ++frame) {
    history = step(frame, [&](uint32_t x, uint32_t y, uint8_t e) { return raw_of(values[y * m_gridWidth + x], e); });
  }
  ASSERT_EQ(history.size(), m_raw.size());
  const std::vector<double> want = spatial(m_raw);
  for (size_t i = 0; i < history.size(); ++i) {
    EXPECT_EQ(history[i].count, ao::MaxSamples) << i;
    EXPECT_NEAR(history[i].ao, want[i], 2e-3) << i;
  }
  EXPECT_EQ(ao::denoise::testing::counters().temporalResets, 1u);
  EXPECT_EQ(ao::denoise::testing::counters().temporalValid, 19u);
}

TEST_F(AoDenoiseGpu, NoisyStaticInputHoldsSamples) {
  begin(frontal_rig(96, 64, 500.0));
  std::mt19937 rng{0x5161Au};
  std::normal_distribution<double> noise(0.6, 0.05);
  std::vector<HistoryTexel> previous;
  for (uint64_t frame = 1; frame <= 32; ++frame) {
    const auto history = step(frame, [&](uint32_t, uint32_t, uint8_t e) { return raw_of(noise(rng), e); });
    ASSERT_EQ(history.size(), m_raw.size());
    if (!previous.empty()) {
      size_t capped = 0;
      for (size_t i = 0; i < history.size(); ++i) {
        capped += history[i].count < std::min(previous[i].count + 1.f, ao::MaxSamples) - 0.5f ? 1 : 0;
      }
      EXPECT_LE(capped * 10, history.size()) << "frame " << frame << ": " << capped << " lag-capped texels";
    }
    previous = history;
  }
  double counts = 0.0;
  for (const HistoryTexel& texel : previous) {
    counts += texel.count;
  }
  EXPECT_GE(counts / static_cast<double>(previous.size()), 12.0);
}

TEST_F(AoDenoiseGpu, MovingOccluderTrailBounded) {
  begin(frontal_rig(64, 48, 500.0));
  std::mt19937 rng{0x7A11u};
  std::uniform_real_distribution<double> jitter(-0.05, 0.05);
  const auto patch = [](uint32_t x, uint32_t y) { return x >= 10 && x < 22 && y >= 8 && y < 16; };
  std::vector<HistoryTexel> history;
  for (uint64_t frame = 1; frame <= 36; ++frame) {
    history = step(frame, [&](uint32_t x, uint32_t y, uint8_t e) {
      return raw_of(frame <= 32 && patch(x, y) ? 0.4 : 0.9 + jitter(rng), e);
    });
  }
  ASSERT_EQ(history.size(), m_raw.size());
  size_t checked = 0;
  for (uint32_t y = 10; y < 14; ++y) {
    for (uint32_t x = 12; x < 20; ++x) {
      EXPECT_NEAR(history[size_t{y} * m_gridWidth + x].ao, 0.9, 0.05) << x << "," << y;
      ++checked;
    }
  }
  EXPECT_EQ(checked, 32u);
}

TEST_F(AoDenoiseGpu, MotionContrastCapsSamples) {
  constexpr float Pixel = 2.f / 64.f;
  const auto constant = [](uint32_t, uint32_t, uint8_t e) { return raw_of(0.7, e); };
  begin(frontal_rig(64, 48, 500.0));
  set_motion([](uint32_t x, uint32_t) {
    return x >= 24 && x < 40 ? std::array<float, 2>{3.f * Pixel, 0.f} : std::array<float, 2>{};
  });
  std::vector<HistoryTexel> history;
  for (uint64_t frame = 1; frame <= 20; ++frame) {
    history = step(frame, constant);
  }
  ASSERT_EQ(history.size(), m_raw.size());
  ASSERT_LT(rep(11, 64), 24u);
  ASSERT_GE(rep(12, 64), 24u);
  ASSERT_LT(rep(19, 64), 40u);
  ASSERT_GE(rep(20, 64), 40u);
  for (uint32_t y = 2; y < m_gridHeight - 2; ++y) {
    EXPECT_EQ(history[size_t{y} * m_gridWidth + 11].count, ao::MovingSamples) << y;
    EXPECT_EQ(history[size_t{y} * m_gridWidth + 20].count, ao::MovingSamples) << y;
    EXPECT_EQ(history[size_t{y} * m_gridWidth + 3].count, ao::MaxSamples) << y;
  }

  begin(frontal_rig(64, 48, 500.0));
  set_motion([](uint32_t, uint32_t) { return std::array<float, 2>{3.f * Pixel, 0.f}; });
  for (uint64_t frame = 1; frame <= 24; ++frame) {
    history = step(frame, constant);
  }
  ASSERT_EQ(history.size(), m_raw.size());
  for (uint32_t y = 2; y < m_gridHeight - 2; ++y) {
    for (uint32_t x = 8; x < m_gridWidth; ++x) {
      EXPECT_GT(history[size_t{y} * m_gridWidth + x].count, ao::MovingSamples) << x << "," << y;
    }
    for (uint32_t x = 26; x < m_gridWidth; ++x) {
      EXPECT_EQ(history[size_t{y} * m_gridWidth + x].count, ao::MaxSamples) << x << "," << y;
    }
  }
}

TEST_F(AoDenoiseGpu, DisocclusionRejectsHistory) {
  begin(frontal_rig(64, 48, 500.0));
  const auto constant = [](uint32_t, uint32_t, uint8_t e) { return raw_of(0.8, e); };
  for (uint64_t frame = 1; frame <= 10; ++frame) {
    step(frame, constant);
  }
  set_depth(depth_with_block(250.0, 16, 12, 48, 36));
  const auto history = step(11, constant);
  ASSERT_EQ(history.size(), m_raw.size());
  size_t inside = 0;
  for (uint32_t y = 0; y < m_gridHeight; ++y) {
    for (uint32_t x = 0; x < m_gridWidth; ++x) {
      const uint32_t qx = rep(x, 64);
      const uint32_t qy = rep(y, 48);
      const float count = history[size_t{y} * m_gridWidth + x].count;
      if (qx >= 18 && qx < 46 && qy >= 14 && qy < 34) {
        EXPECT_EQ(count, 1.f) << x << "," << y;
        ++inside;
      } else if (qx < 12 || qx >= 52 || qy < 8 || qy >= 40) {
        EXPECT_EQ(count, 11.f) << x << "," << y;
      }
    }
  }
  EXPECT_GT(inside, 100u);
}

TEST_F(AoDenoiseGpu, OffscreenMotionResets) {
  constexpr float Pixel = 2.f / 64.f;
  const auto constant = [](uint32_t, uint32_t, uint8_t e) { return raw_of(0.8, e); };
  begin(frontal_rig(64, 48, 500.0));
  set_motion([](uint32_t x, uint32_t y) {
    return x >= 16 && x < 32 && y >= 16 && y < 32 ? std::array<float, 2>{4.f, 4.f} : std::array<float, 2>{};
  });
  std::vector<HistoryTexel> history;
  for (uint64_t frame = 1; frame <= 6; ++frame) {
    history = step(frame, constant);
  }
  ASSERT_EQ(history.size(), m_raw.size());
  for (uint32_t y = 8; y < 16; ++y) {
    for (uint32_t x = 8; x < 16; ++x) {
      EXPECT_EQ(history[size_t{y} * m_gridWidth + x].count, 1.f) << x << "," << y;
    }
    EXPECT_EQ(history[size_t{y} * m_gridWidth + 24].count, 6.f) << y;
  }

  begin(frontal_rig(64, 48, 500.0));
  set_motion([](uint32_t, uint32_t) { return std::array<float, 2>{10.f * Pixel, 0.f}; });
  for (uint64_t frame = 1; frame <= 6; ++frame) {
    history = step(frame, constant);
  }
  ASSERT_EQ(history.size(), m_raw.size());
  for (uint32_t y = 0; y < m_gridHeight; ++y) {
    for (uint32_t x = 0; x < m_gridWidth; ++x) {
      const bool offscreen = rep(x, 64) + 0.5f < 10.f;
      EXPECT_EQ(history[size_t{y} * m_gridWidth + x].count == 1.f, offscreen) << x << "," << y;
    }
  }
}

TEST_F(AoDenoiseGpu, NonReceiverWritesZeroCount) {
  begin(frontal_rig(64, 48, 500.0));
  const auto receiver = [](uint32_t x, uint32_t y) { return (x / 3 + y) % 4 != 0; };
  std::vector<HistoryTexel> history;
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    history = step(frame, [&](uint32_t x, uint32_t y, uint8_t e) { return raw_of(0.5, e, receiver(x, y)); });
  }
  ASSERT_EQ(history.size(), m_raw.size());
  for (uint32_t y = 0; y < m_gridHeight; ++y) {
    for (uint32_t x = 0; x < m_gridWidth; ++x) {
      const HistoryTexel& texel = history[size_t{y} * m_gridWidth + x];
      if (receiver(x, y)) {
        EXPECT_EQ(texel.count, 3.f) << x << "," << y;
        EXPECT_LT(texel.ao, 1.f) << x << "," << y;
      } else {
        EXPECT_EQ(texel.count, 0.f) << x << "," << y;
        EXPECT_EQ(texel.ao, 1.f) << x << "," << y;
      }
    }
  }
}

TEST_F(AoDenoiseGpu, HistoryCarriesEdges) {
  begin(ground_rig(64, 48, 20.0));
  for (uint64_t frame = 1; frame <= 2; ++frame) {
    const auto history = step(frame, [&](uint32_t x, uint32_t y, uint8_t e) {
      return raw_of(0.6, e, m_hiz[size_t{y} * m_gridWidth + x] < 0.5 * ao::SkyDepth);
    });
    ASSERT_EQ(history.size(), m_raw.size());
    size_t edged = 0;
    for (size_t i = 0; i < history.size(); ++i) {
      EXPECT_EQ(packed_edges(history[i]), (m_raw[i] >> 8) & 0xFFu) << i;
      edged += ((m_raw[i] >> 8) & 0xFFu) != 0xFFu ? 1 : 0;
    }
    EXPECT_GT(edged, 10u);
  }
}

TEST_F(AoDenoiseGpu, EdgesStopBlur) {
  constexpr uint8_t RightClosed = (3 << 6) | (0 << 4) | (3 << 2) | 3;
  constexpr uint8_t LeftClosed = (0 << 6) | (3 << 4) | (3 << 2) | 3;
  for (const bool edged : {true, false}) {
    SCOPED_TRACE(edged ? "edges" : "open");
    begin(frontal_rig(64, 48, 500.0));
    const auto history = step(1, [&](uint32_t x, uint32_t, uint8_t e) {
      uint8_t edges = e;
      if (edged && x == 15) {
        edges = RightClosed;
      } else if (edged && x == 16) {
        edges = LeftClosed;
      }
      return raw_of(x < 16 ? 0.3 : 1.0, edges);
    });
    ASSERT_EQ(history.size(), m_raw.size());
    for (uint32_t y = 2; y < m_gridHeight - 2; ++y) {
      const float left = history[size_t{y} * m_gridWidth + 15].ao;
      const float right = history[size_t{y} * m_gridWidth + 16].ao;
      if (edged) {
        EXPECT_NEAR(left, 0.3f, 1e-3f) << y;
        EXPECT_EQ(right, 1.f) << y;
      } else {
        EXPECT_GT(left, 0.32f) << y;
        EXPECT_LT(right, 0.98f) << y;
      }
    }
  }
}

TEST_F(AoDenoiseGpu, BorderTexelsValidationClean) {
  std::mt19937 rng{0xB0DEu};
  std::uniform_real_distribution<double> unit(0.2, 1.0);
  for (const auto [width, height] : {std::pair{1u, 1u}, std::pair{3u, 5u}, std::pair{61u, 37u}}) {
    for (const Quality quality : {Quality::High, Quality::Ultra}) {
      SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " q" +
                   std::to_string(static_cast<int>(quality)));
      begin(frontal_rig(width, height, 500.0), quality);
      std::vector<double> values(size_t{m_gridWidth} * m_gridHeight);
      for (double& value : values) {
        value = unit(rng);
      }
      for (uint64_t frame = 1; frame <= 3; ++frame) {
        const auto history =
            step(frame, [&](uint32_t x, uint32_t y, uint8_t e) { return raw_of(values[y * m_gridWidth + x], e); });
        ASSERT_EQ(history.size(), size_t{m_gridWidth} * m_gridHeight);
        for (const HistoryTexel& texel : history) {
          EXPECT_TRUE(std::isfinite(texel.ao) && std::isfinite(texel.logDepth));
          EXPECT_EQ(texel.count, static_cast<float>(frame));
          EXPECT_GE(texel.ao, 0.f);
          EXPECT_LE(texel.ao, 1.f);
        }
      }
      EXPECT_EQ(ao::denoise::testing::counters().dispatches, 3u);
    }
  }
}
} // namespace
} // namespace aurora::render::detail

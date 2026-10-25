#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "logging.hpp"
#include "render/debug/debug.hpp"
#include "render/debug/graph_inspection.hpp"
#include "render/frame_state.hpp"
#include "render/geometry/compact.hpp"
#include "render/geometry/indirect.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_layout.hpp"
#include "render/graph_pool.hpp"
#include "render/jobs.hpp"
#include "render/module.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
using wgpu::TextureFormat;
using wgpu::TextureSampleType;
using wgpu::TextureViewDimension;

constexpr uint64_t KiB = 1024;

constexpr std::string_view FrameWgsl = R"(struct AuroraFrame {
  efb_size: vec2<u32>,
  output_size: vec2<u32>,
  inv_efb_size: vec2<f32>,
  inv_output_size: vec2<f32>,
  frame_index: u32,
  flags: u32,
  msaa_samples: u32,
  pad0: u32,
  reserved: array<vec4<u32>, 13>,
}
@group(0) @binding(0) var<uniform> aurora_frame: AuroraFrame;
)";

constexpr std::string_view FullscreenVertexWgsl = R"(
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  uint16_t order = 0;
  uint32_t paramsSize = 0;
  std::function<void(InternalPassBuilder&)> declare;
  std::function<void(PassContext&)> encode;
  std::function<void(const PipelineSetup&, std::span<uint32_t, MaxPassPipelines>)> pipelines;
  bool encodes = true;
  mutable int calls = 0;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  if (pass.declare) {
    pass.declare(builder);
  }
}

void encode_test_pass(PassContext& ctx, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  ++pass.calls;
  if (pass.encode) {
    pass.encode(ctx);
  }
}

void pipelines_test_pass(const PipelineSetup& setup, const void* state, std::span<uint32_t, MaxPassPipelines> ids) {
  static_cast<const TestPass*>(state)->pipelines(setup, ids);
}

PassDesc desc_of(const TestPass& pass) {
  return {
      .name = pass.name,
      .stage = pass.stage,
      .kind = pass.kind,
      .order = pass.order,
      .paramsSize = pass.paramsSize,
      .declare = declare_test_pass,
      .encode = pass.encodes ? encode_test_pass : nullptr,
      .pipelines = pass.pipelines ? pipelines_test_pass : nullptr,
      .state = &pass,
  };
}

TexDesc absolute(TextureFormat format, uint32_t width = 64, uint32_t height = 64, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

std::vector<uint8_t> g_uniforms;
std::vector<size_t> g_pushes;

gfx::Range fake_sink(const uint8_t* data, size_t length) {
  const size_t alignment = std::max<size_t>(gfx::detail::resources().limits.minUniformBufferOffsetAlignment, 1);
  const size_t offset = (g_uniforms.size() + alignment - 1) / alignment * alignment;
  g_uniforms.resize(offset + length);
  std::memcpy(g_uniforms.data() + offset, data, length);
  g_pushes.push_back(length);
  return {static_cast<uint32_t>(offset), static_cast<uint32_t>(length)};
}

struct LogLine {
  AuroraLogLevel level;
  std::string message;
};

std::mutex g_logMutex;
std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  std::lock_guard lock{g_logMutex};
  g_logLines.push_back({level, std::string{message, len}});
}

size_t count_logs(AuroraLogLevel level, std::string_view text) {
  std::lock_guard lock{g_logMutex};
  return static_cast<size_t>(std::ranges::count_if(g_logLines, [&](const LogLine& line) {
    return line.level == level && line.message.find(text) != std::string::npos;
  }));
}

void set_bit(std::vector<uint64_t>& bits, size_t index, bool value) {
  if (bits.size() <= index / 64) {
    bits.resize(index / 64 + 1, 0);
  }
  const uint64_t mask = uint64_t{1} << (index % 64);
  bits[index / 64] = value ? bits[index / 64] | mask : bits[index / 64] & ~mask;
}

struct TestFrame {
  explicit TestFrame(std::shared_ptr<const CompiledGraph> compiled)
  : graph(std::move(compiled)), state{.graph = graph} {}

  std::shared_ptr<const CompiledGraph> graph;
  FrameState state;

  [[nodiscard]] NodeId node(std::string_view name) const { return graph->find_node(name); }
  [[nodiscard]] uint32_t program_index(std::string_view name) const { return graph->nodes[node(name)].programIndex; }
  [[nodiscard]] const CompiledUse& use(std::string_view name, size_t index) const {
    return graph->node_uses(node(name))[index];
  }
};

StageData stage_for(const TestFrame& frame, Stage stage, uint64_t frameId) {
  const CompiledGraph& graph = *frame.graph;
  const StageProgram& program = graph.stages[static_cast<size_t>(stage)];
  const auto passCount = static_cast<uint32_t>(program.passes.size());
  StageData data{
      .stage = stage,
      .frameId = frameId,
      .frameIndex = static_cast<uint32_t>(frameId),
      .frame = &frame.state,
      .passCount = passCount,
      .enabled = std::vector<uint64_t>((passCount + 63) / 64, ~uint64_t{0}),
      .dummies = std::vector<uint64_t>((program.useBits + 63) / 64, 0),
      .prevValid = std::vector<uint64_t>((graph.histories.size() + 63) / 64, 0),
  };
  for (const NodeId node : program.nodes) {
    for (const CompiledUse& use : graph.node_uses(node)) {
      if (use.staticDummy) {
        set_bit(data.dummies, use.stageUse, true);
      }
    }
  }
  return data;
}

wgpu::ShaderModule shader_module(std::string_view source) {
  const std::string code{source};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
  return webgpu::g_device.CreateShaderModule(&descriptor);
}

wgpu::ComputePipeline compute_pipeline(const wgpu::PipelineLayout& layout, std::string_view source) {
  const wgpu::ComputePipelineDescriptor descriptor{
      .layout = layout,
      .compute = {.module = shader_module(source), .entryPoint = "main"},
  };
  return webgpu::g_device.CreateComputePipeline(&descriptor);
}

wgpu::RenderPipeline render_pipeline(const wgpu::PipelineLayout& layout, std::string_view source, TextureFormat format,
                                     uint32_t samples = 1) {
  const wgpu::ShaderModule module = shader_module(source);
  const wgpu::ColorTargetState target{.format = format};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor descriptor{
      .layout = layout,
      .vertex = {.module = module, .entryPoint = "vs"},
      .multisample = {.count = samples},
      .fragment = &fragment,
  };
  return webgpu::g_device.CreateRenderPipeline(&descriptor);
}

std::string frame_shader(std::string_view body) { return std::string{FrameWgsl} + std::string{body}; }

class GraphBindTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    gpu_heap::shutdown();
    reset_module_state();
    {
      std::lock_guard lock{g_logMutex};
      g_logLines.clear();
    }
    g_config.logCallback = capture_log;
    testing::record_bind_details(true);
  }

  void TearDown() override {
    testing::record_bind_details(false);
    g_config.logCallback = nullptr;
    if (!m_device) {
      return;
    }
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    gpu_heap::shutdown();
    testing::clear_passes();
    reset_frame_cull();
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
    g_uniforms.clear();
    g_pushes.clear();
  }

  void use_globals(const test::TestDeviceOptions& options) {
    m_globals.reset();
    m_options = options;
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
  }

  static GraphInputs test_inputs() {
    GraphInputs inputs;
    inputs.env = gather_graph_environment();
    return inputs;
  }

  static std::shared_ptr<const CompiledGraph>
  compile(const std::vector<TestPass>& passes, GraphInputs inputs = test_inputs(), std::span<const PassId> ids = {}) {
    std::vector<PassDesc> descs;
    for (const TestPass& pass : passes) {
      descs.push_back(desc_of(pass));
    }
    inputs.passes = descs;
    inputs.ids = ids;
    const std::shared_ptr<CompiledGraph> graph = compile_graph(inputs);
    prepare_graph_bindings(*graph);
    return graph;
  }

  void push_validation_scope() const { m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation); }

  [[nodiscard]] std::optional<std::string> pop_validation_scope() const {
    return test::pop_validation_error(*m_device);
  }

  void upload_uniforms() const {
    if (g_uniforms.empty()) {
      return;
    }
    g_uniforms.resize((g_uniforms.size() + 3) & ~size_t{3});
    m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, g_uniforms.data(), g_uniforms.size());
  }

  std::optional<std::string> execute(StageData& stage, bool bind = true) const {
    if (bind) {
      build_stage_bindings(stage, &fake_sink);
    }
    upload_uniforms();
    push_validation_scope();
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    run_stage(encoder, stage);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    return pop_validation_scope();
  }

  std::optional<std::string> run_frame(const TestFrame& frame, uint64_t frameId,
                                       const std::function<void(StageData&)>& adjust = {},
                                       Stage stage = Stage::OpaqueDone) const {
    g_uniforms.clear();
    on_worker_frame_begin(frameId, &frame.state);
    StageData data = stage_for(frame, stage, frameId);
    if (adjust) {
      adjust(data);
    }
    return execute(data);
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

std::vector<TestPass> declaration_order_passes() {
  return {
      {.name = "t.make",
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) {
             b.create_buffer("t.a", {.size = 256}, Access::StorageWrite);
             b.create("t.tex", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
             b.create_buffer("t.args", {.size = 64}, Access::StorageWrite);
           }},
      {.name = "t.pass",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use_buffer("t.a", Access::StorageRead);
             b.use("t.tex", Access::Sampled);
             b.create_buffer("t.out", {.size = 256}, Access::StorageWrite);
             b.use_buffer("t.args", Access::Indirect);
             b.create("t.img", absolute(TextureFormat::RGBA16Float), Access::StorageWrite);
             b.side_effect();
           }},
  };
}

constexpr std::string_view OrderedBindings = R"(
@group(1) @binding(0) var<storage, read> t_a: array<u32>;
@group(1) @binding(1) var t_tex: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> t_out: array<u32>;
@group(1) @binding(3) var t_img: texture_storage_2d<rgba16float, write>;
)";

constexpr std::string_view SwappedBindings = R"(
@group(1) @binding(0) var<storage, read> t_a: array<u32>;
@group(1) @binding(2) var t_tex: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> t_out: array<u32>;
@group(1) @binding(3) var t_img: texture_storage_2d<rgba16float, write>;
)";

constexpr std::string_view OrderedMain = R"(
@compute @workgroup_size(1)
fn main() {
  let size = textureDimensions(t_tex);
  t_out[0] = t_a[0] + size.x + aurora_frame.frame_index;
  textureStore(t_img, vec2<i32>(0, 0), vec4<f32>(1.0));
}
)";

TEST_F(GraphBindTest, WgslGroupOrderValidates) {
  auto passes = declaration_order_passes();
  const std::string ordered = frame_shader(std::string{OrderedBindings} + std::string{OrderedMain});
  const std::string swapped = frame_shader(std::string{SwappedBindings} + std::string{OrderedMain});
  passes[1].encode = [&](PassContext& ctx) {
    const wgpu::ComputePipeline pipeline = compute_pipeline(ctx.pipeline_layout(), ordered);
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.SetPipeline(pipeline);
    pass.DispatchWorkgroups(1);
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("t.pass")];
  ASSERT_TRUE(plan.layout);
  ASSERT_EQ(plan.slots.size(), 4u);

  push_validation_scope();
  compute_pipeline(plan.layout->pipelineLayout, ordered);
  EXPECT_EQ(pop_validation_scope(), std::nullopt);
  push_validation_scope();
  compute_pipeline(plan.layout->pipelineLayout, swapped);
  EXPECT_NE(pop_validation_scope(), std::nullopt);

  EXPECT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[1].calls, 1);
}

TEST_F(GraphBindTest, ParamsLayoutValidates) {
  const std::string source = frame_shader(std::string{R"(
struct Params {
  tint: vec4<f32>,
}
@group(0) @binding(1) var<uniform> params: Params;
@fragment
fn fs() -> @location(0) vec4<f32> {
  return params.tint * f32(aurora_frame.msaa_samples);
}
)"} + std::string{FullscreenVertexWgsl});
  std::vector<TestPass> passes{
      {.name = "p.draw",
       .kind = PassKind::Render,
       .paramsSize = 16,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("p.color", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           }},
  };
  passes[0].encode = [&](PassContext& ctx) {
    const wgpu::RenderPipeline pipeline = render_pipeline(ctx.pipeline_layout(), source, TextureFormat::RGBA8Unorm);
    const wgpu::RenderPassEncoder pass = ctx.begin_render();
    pass.SetPipeline(pipeline);
    pass.Draw(3);
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  EXPECT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
}

TEST_F(GraphBindTest, FrameGroupLayoutIsStable) {
  const auto passes = declaration_order_passes();
  push_validation_scope();
  const TestFrame first{compile(passes)};
  const WGPUBindGroupLayout layout = frame_layout().Get();
  const TestFrame second{compile(passes)};
  EXPECT_EQ(pop_validation_scope(), std::nullopt);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(frame_layout().Get(), layout);
  EXPECT_TRUE(frame_group());

  const auto entries = frame_layout_entries();
  EXPECT_TRUE(entries[0].buffer.hasDynamicOffset);
  EXPECT_TRUE(entries[1].buffer.hasDynamicOffset);
  EXPECT_EQ(entries[2].sampler.type, wgpu::SamplerBindingType::Filtering);
  EXPECT_EQ(entries[3].sampler.type, wgpu::SamplerBindingType::NonFiltering);
  EXPECT_EQ(entries[4].sampler.type, wgpu::SamplerBindingType::Filtering);
  EXPECT_EQ(entries[5].sampler.type, wgpu::SamplerBindingType::Comparison);

  // A filtering sampler where the layout wants comparison must fail, which shows b5's type is live
  const wgpu::SamplerDescriptor linearDescriptor{.magFilter = wgpu::FilterMode::Linear,
                                                 .minFilter = wgpu::FilterMode::Linear};
  const wgpu::Sampler linear = m_device->device.CreateSampler(&linearDescriptor);
  const wgpu::SamplerDescriptor compareDescriptor{.compare = wgpu::CompareFunction::GreaterEqual};
  const wgpu::Sampler compare = m_device->device.CreateSampler(&compareDescriptor);
  const wgpu::SamplerDescriptor nearestDescriptor{};
  const wgpu::Sampler nearest = m_device->device.CreateSampler(&nearestDescriptor);
  const wgpu::Buffer& uniforms = gfx::detail::resources().uniformBuffer;
  const auto group_with = [&](const wgpu::Sampler& at5) {
    const std::array bindings{
        wgpu::BindGroupEntry{.binding = 0, .buffer = uniforms, .size = FrameUniformsSize},
        wgpu::BindGroupEntry{.binding = 1, .buffer = uniforms, .size = MaxPassParamsSize},
        wgpu::BindGroupEntry{.binding = 2, .sampler = linear},
        wgpu::BindGroupEntry{.binding = 3, .sampler = nearest},
        wgpu::BindGroupEntry{.binding = 4, .sampler = linear},
        wgpu::BindGroupEntry{.binding = 5, .sampler = at5},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .layout = frame_layout(), .entryCount = bindings.size(), .entries = bindings.data()};
    push_validation_scope();
    m_device->device.CreateBindGroup(&descriptor);
    return pop_validation_scope();
  };
  EXPECT_EQ(group_with(compare), std::nullopt);
  EXPECT_NE(group_with(linear), std::nullopt);
}

TEST_F(GraphBindTest, LayoutsAreDeduplicated) {
  const auto passes = declaration_order_passes();
  const TestFrame first{compile(passes)};
  const TestFrame second{compile(passes)};
  ASSERT_NE(first.graph, second.graph);
  const BindingPlan& a = first.graph->bindingPlans[first.node("t.pass")];
  const BindingPlan& b = second.graph->bindingPlans[second.node("t.pass")];
  ASSERT_TRUE(a.layout);
  ASSERT_TRUE(b.layout);
  EXPECT_NE(a.layoutSerial, 0u);
  EXPECT_EQ(a.layoutSerial, b.layoutSerial);
  EXPECT_EQ(a.layout->pipelineLayout.Get(), b.layout->pipelineLayout.Get());
  EXPECT_EQ(a.layout->passLayout.Get(), b.layout->passLayout.Get());
  const BindingPlan& maker = first.graph->bindingPlans[first.node("t.make")];
  EXPECT_NE(maker.layoutSerial, a.layoutSerial);
}

std::vector<TestPass> steady_passes() {
  return {
      {.name = "s.a",
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) { b.create("s.ta", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite); }},
      {.name = "s.b",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("s.ta", Access::Sampled);
             b.create("s.tb", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "s.c",
       .order = 2,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("s.tb", Access::Sampled);
             b.side_effect();
           }},
  };
}

TEST_F(GraphBindTest, SteadyFrameCreatesNoBindGroups) {
  const auto passes = steady_passes();
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  const BindCacheStats first = bind_cache_stats();
  EXPECT_EQ(first.created, passes.size());
  EXPECT_EQ(first.entries, passes.size());
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  const BindCacheStats steady = bind_cache_stats();
  EXPECT_EQ(steady.created, first.created);
  EXPECT_EQ(steady.hits, first.hits + 2 * passes.size());
  EXPECT_EQ(steady.evicted, 0u);
  EXPECT_EQ(steady.staleEntries, 0u);
  for (const TestPass& pass : passes) {
    EXPECT_EQ(pass.calls, 3) << pass.name;
  }

  // An equal recompile keeps layout and physical serials, so it binds from the same entries
  const TestFrame again{compile(passes)};
  ASSERT_EQ(run_frame(again, 4), std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, first.created);
}

TEST_F(GraphBindTest, HistoryCachesTwoGroups) {
  std::vector<TestPass> passes{
      {.name = "h.own",
       .kind = PassKind::Render,
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) { b.history("h.tex", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite); }},
      {.name = "h.read",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use_prev("h.tex");
             b.side_effect();
           }},
  };
  std::vector<bool> valid;
  passes[1].encode = [&](PassContext& ctx) { valid.push_back(ctx.prev_valid(TexRef{0})); };
  const TestFrame frame{compile(passes)};
  const CompiledUse& prev = frame.use("h.read", 0);
  const uint16_t history = frame.graph->resources[prev.resource].historySlot;
  ASSERT_NE(history, InvalidResource);
  ASSERT_TRUE(prev.hasFallback);

  ASSERT_EQ(run_frame(frame, 1, [&](StageData& stage) { set_bit(stage.dummies, prev.stageUse, true); }), std::nullopt);
  const auto firstIdentities = testing::last_identities(frame.node("h.read"));
  ASSERT_EQ(firstIdentities.size(), 1u);
  EXPECT_EQ(firstIdentities[0].kind, IdentityKind::Dummy);
  const uint64_t created = bind_cache_stats().created;
  EXPECT_EQ(created, 2u);

  std::vector<uint64_t> serials;
  for (uint64_t frameId = 2; frameId <= 5; ++frameId) {
    ASSERT_EQ(run_frame(frame, frameId, [&](StageData& stage) { set_bit(stage.prevValid, history, true); }),
              std::nullopt);
    const auto identities = testing::last_identities(frame.node("h.read"));
    ASSERT_EQ(identities.size(), 1u);
    EXPECT_EQ(identities[0].kind, IdentityKind::History);
    serials.push_back(identities[0].value);
  }
  EXPECT_NE(serials[0], serials[1]);
  EXPECT_EQ(serials[0], serials[2]);
  EXPECT_EQ(serials[1], serials[3]);
  EXPECT_EQ(bind_cache_stats().created, created + 2);
  EXPECT_EQ(valid, (std::vector<bool>{false, true, true, true, true}));
}

TEST_F(GraphBindTest, PersistentGenerationBumpRebinds) {
  const PersistentHandle handle = create_persistent("p.buf", PersistentKind::Buffer);
  ASSERT_NE(handle.id, 0u);
  const auto make_buffer = [&] {
    const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Storage, .size = 256};
    return m_device->device.CreateBuffer(&descriptor);
  };
  set_persistent_buffer(handle, make_buffer(), 0, 256);
  std::vector<TestPass> passes{
      {.name = "p.read",
       .declare =
           [handle](InternalPassBuilder& b) {
             b.import_persistent("p.buf", handle, Access::StorageRead);
             b.side_effect();
           }},
  };
  wgpu::Buffer bound;
  passes[0].encode = [&](PassContext& ctx) { bound = ctx.buffer(BufRef{0}).buffer; };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, 1u);

  set_persistent_buffer(handle, make_buffer(), 0, 256);
  const wgpu::Buffer latest = make_buffer();
  set_persistent_buffer(handle, latest, 0, 128);
  PersistentBuffer resolved;
  ASSERT_TRUE(resolve_persistent(handle, resolved));
  EXPECT_EQ(resolved.generation, 3u);

  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  const BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.evicted, 1u);
  EXPECT_EQ(stats.staleEntries, 0u);
  EXPECT_EQ(stats.entries, 1u);
  EXPECT_EQ(bound.Get(), latest.Get());
  const auto identities = testing::last_identities(frame.node("p.read"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Persistent);
  EXPECT_EQ(identities[0].value, uint64_t{handle.id} << 32 | 3);
}

TEST_F(GraphBindTest, HeapGrowthLeavesNoStaleGroups) {
  const gpu_heap::PoolId pool = gpu_heap::create_pool({
      .name = "heap.test",
      .usage = wgpu::BufferUsage::Storage,
      .initialSize = 4 * KiB,
      .maxSize = 1024 * KiB,
  });
  ASSERT_NE(pool, gpu_heap::PoolId::Invalid);
  const PersistentHandle handle = heap_pool_handle(pool);
  EXPECT_EQ(heap_pool_handle(pool), handle);
  std::vector<TestPass> passes{
      {.name = "heap.read",
       .declare =
           [handle](InternalPassBuilder& b) {
             b.import_persistent("heap.test", handle, Access::StorageRead);
             b.side_effect();
           }},
  };
  BufferView bound;
  passes[0].encode = [&](PassContext& ctx) { bound = ctx.buffer(BufRef{0}); };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, 1u);
  EXPECT_EQ(bound.buffer.Get(), gpu_heap::binding(pool).buffer.Get());

  ASSERT_TRUE(gpu_heap::allocate(pool, 4 * KiB));
  ASSERT_FALSE(gpu_heap::allocate(pool, 4 * KiB));
  on_worker_frame_begin(2, &frame.state);
  ASSERT_EQ(gpu_heap::binding(pool).generation, 2u);
  EXPECT_EQ(bind_cache_stats().staleEntries, 0u);
  EXPECT_EQ(bind_cache_stats().evicted, 1u);

  StageData stage = stage_for(frame, Stage::OpaqueDone, 2);
  ASSERT_EQ(execute(stage), std::nullopt);
  BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.staleEntries, 0u);
  EXPECT_EQ(stats.evicted, 1u);
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(bound.buffer.Get(), gpu_heap::binding(pool).buffer.Get());
  EXPECT_EQ(bound.size, gpu_heap::binding(pool).size);

  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.staleEntries, 0u);
}

TEST_F(GraphBindTest, TransientReleaseEvictsGroups) {
  const auto passes = steady_passes();
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  const auto identities = testing::last_identities(frame.node("s.c"));
  ASSERT_EQ(identities.size(), 1u);
  ASSERT_EQ(identities[0].kind, IdentityKind::Transient);
  EXPECT_EQ(bind_cache_stats().entries, 3u);

  const TestFrame empty{empty_graph()};
  force_memory_pressure();
  on_worker_frame_begin(2, &empty.state);
  EXPECT_TRUE(transient_pool().empty());
  const BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.entries, 0u);
  EXPECT_EQ(stats.evicted, 3u);
  EXPECT_EQ(stats.staleEntries, 0u);
}

TEST_F(GraphBindTest, MissingProducerBindsDummy) {
  std::vector<TestPass> passes{
      {.name = "m.make",
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) { b.create("m.tex", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite); },
       .encodes = false},
      {.name = "m.read",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("m.tex", Access::Sampled, Missing::BindDummyWhite);
             b.side_effect();
           }},
  };
  const TestFrame frame{compile(passes)};
  const CompiledUse& use = frame.use("m.read", 0);
  ASSERT_TRUE(use.hasFallback);
  EXPECT_EQ(use.fallback.color, DummyColor::White);
  ASSERT_EQ(run_frame(frame, 1,
                      [&](StageData& stage) {
                        set_bit(stage.enabled, frame.program_index("m.make"), false);
                        set_bit(stage.dummies, use.stageUse, true);
                      }),
            std::nullopt);
  auto identities = testing::last_identities(frame.node("m.read"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Dummy);

  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  identities = testing::last_identities(frame.node("m.read"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Transient);
  EXPECT_EQ(bind_cache_stats().created, 2u);
  EXPECT_EQ(passes[1].calls, 2);
}

TEST_F(GraphBindTest, DummiesMatchDummyKey) {
  std::vector<TestPass> passes{
      {.name = "d.pass",
       .declare =
           [](InternalPassBuilder& b) {
             const auto texture = [&](std::string_view name, TextureSampleType type, TextureViewDimension dimension) {
               b.use(name, Access::Sampled,
                     UseOptions{.missing = Missing::BindDummyBlack, .dimension = dimension, .sampleType = type});
             };
             texture("d.float", TextureSampleType::Float, TextureViewDimension::e2D);
             texture("d.uint", TextureSampleType::Uint, TextureViewDimension::e2DArray);
             texture("d.depth", TextureSampleType::Depth, TextureViewDimension::e2D);
             texture("d.cube", TextureSampleType::Float, TextureViewDimension::Cube);
             texture("d.volume", TextureSampleType::UnfilterableFloat, TextureViewDimension::e3D);
             b.use_buffer("d.buf", Access::StorageRead,
                          UseOptions{.missing = Missing::BindDummyBlack, .dummySize = 112});
             b.use_buffer("d.uni", Access::Uniform, UseOptions{.missing = Missing::BindDummyWhite, .dummySize = 64});
             b.use("d.store", Access::StorageRead,
                   UseOptions{.missing = Missing::BindDummyWhite, .format = TextureFormat::R32Float});
             b.side_effect();
           }},
  };
  std::vector<BufferView> buffers;
  passes[0].encode = [&](PassContext& ctx) {
    buffers.push_back(ctx.buffer(BufRef{5}));
    buffers.push_back(ctx.buffer(BufRef{6}));
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(frame.graph->nodes[frame.node("d.pass")].status, CullReason::None);
  const std::array expected{
      DummyKey{.dimension = TextureViewDimension::e2D, .sampleType = TextureSampleType::Float},
      DummyKey{.dimension = TextureViewDimension::e2DArray, .sampleType = TextureSampleType::Uint},
      DummyKey{.dimension = TextureViewDimension::e2D, .sampleType = TextureSampleType::Depth},
      DummyKey{.dimension = TextureViewDimension::Cube, .sampleType = TextureSampleType::Float},
      DummyKey{.dimension = TextureViewDimension::e3D, .sampleType = TextureSampleType::UnfilterableFloat},
      DummyKey{.buffer = true, .bufferSize = 112},
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    const CompiledUse& use = frame.use("d.pass", i);
    EXPECT_TRUE(use.staticDummy) << i;
    ASSERT_TRUE(use.hasFallback) << i;
    EXPECT_EQ(use.fallback, expected[i]) << i;
  }
  EXPECT_EQ(frame.use("d.pass", 7).fallback.storageFormat, TextureFormat::R32Float);

  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(buffers.size(), 2u);
  EXPECT_EQ(buffers[0].size, 112u);
  EXPECT_EQ(buffers[1].size, 64u);
  const auto identities = testing::last_identities(frame.node("d.pass"));
  ASSERT_EQ(identities.size(), 8u);
  std::set<uint64_t> serials;
  for (const BindIdentity& identity : identities) {
    EXPECT_EQ(identity.kind, IdentityKind::Dummy);
    serials.insert(identity.value);
  }
  EXPECT_EQ(serials.size(), identities.size());

  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(testing::last_identities(frame.node("d.pass")), identities);
  EXPECT_EQ(bind_cache_stats().created, 1u);
}

TEST_F(GraphBindTest, GxDummiesUseFixedFormats) {
  std::vector<TestPass> passes{
      {.name = "gd.pass",
       .declare =
           [](InternalPassBuilder& b) {
             b.use("gbuf.debug_id", Access::Sampled, Missing::BindDummyBlack);
             b.use("gbuf.fog", Access::Sampled, Missing::BindDummyWhite);
             b.use("gbuf.surface", Access::Sampled, Missing::BindDummyWhite);
             b.side_effect();
           }},
  };
  push_validation_scope();
  const TestFrame frame{compile(passes)};
  EXPECT_EQ(pop_validation_scope(), std::nullopt);
  ASSERT_EQ(frame.graph->nodes[frame.node("gd.pass")].status, CullReason::None);
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("gd.pass")];
  ASSERT_TRUE(plan.layout);
  ASSERT_EQ(plan.slots.size(), 3u);
  for (size_t i = 0; i < plan.slots.size(); ++i) {
    const CompiledUse& use = frame.use("gd.pass", i);
    EXPECT_TRUE(use.staticDummy) << i;
    EXPECT_EQ(plan.slots[i].sampleType, TextureSampleType::Uint) << i;
    EXPECT_EQ(use.fallback.sampleType, plan.slots[i].sampleType) << i;
  }
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  const auto identities = testing::last_identities(frame.node("gd.pass"));
  ASSERT_EQ(identities.size(), 3u);
  for (const BindIdentity& identity : identities) {
    EXPECT_EQ(identity.kind, IdentityKind::Dummy);
  }
}

struct Gate {
  bool open = true;
  wgpu::Buffer buffer;
};

bool gate_source(void* context, PersistentBuffer& out) {
  const auto& gate = *static_cast<const Gate*>(context);
  if (!gate.open) {
    return false;
  }
  out = {.buffer = gate.buffer, .size = 256, .generation = 1};
  return true;
}

TEST_F(GraphBindTest, SkippedOwnerLeavesHistoryStale) {
  const wgpu::BufferDescriptor descriptor{.usage = wgpu::BufferUsage::Storage, .size = 256};
  Gate gate{.buffer = m_device->device.CreateBuffer(&descriptor)};
  const PersistentHandle handle = create_persistent_buffer_source("hs.gate", &gate_source, &gate);
  std::vector<TestPass> passes{
      {.name = "hs.own",
       .order = 0,
       .declare =
           [handle](InternalPassBuilder& b) {
             b.history("hs.tex", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
             b.import_persistent("hs.gate", handle, Access::StorageRead);
           }},
      {.name = "hs.read",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use_prev("hs.tex");
             b.side_effect();
           }},
  };
  std::vector<bool> valid;
  passes[1].encode = [&](PassContext& ctx) { valid.push_back(ctx.prev_valid(TexRef{0})); };
  const TestFrame frame{compile(passes)};
  const CompiledUse& prev = frame.use("hs.read", 0);
  const uint16_t history = frame.graph->resources[prev.resource].historySlot;
  ASSERT_NE(history, InvalidResource);
  // The cull calls prev valid whenever the owner was scheduled; only the execute-time skip differs
  const auto run = [&](uint64_t frameId, bool scheduled) {
    return run_frame(frame, frameId, [&](StageData& stage) {
      if (scheduled) {
        set_bit(stage.prevValid, history, true);
      } else {
        set_bit(stage.dummies, prev.stageUse, true);
      }
    });
  };
  const auto reader_kind = [&] { return testing::last_identities(frame.node("hs.read")).at(0).kind; };

  ASSERT_EQ(run(1, false), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  const uint64_t written = transient_pool().history_write(history).serial;

  gate.open = false;
  ASSERT_EQ(run(2, true), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(reader_kind(), IdentityKind::History);
  EXPECT_EQ(testing::last_identities(frame.node("hs.read")).at(0).value, written);

  gate.open = true;
  ASSERT_EQ(run(3, true), std::nullopt);
  EXPECT_EQ(transient_pool().history_prev(history).serial, written);
  EXPECT_FALSE(transient_pool().history_fresh(history));
  EXPECT_EQ(reader_kind(), IdentityKind::Dummy);
  EXPECT_EQ(passes[0].calls, 2);

  ASSERT_EQ(run(4, true), std::nullopt);
  EXPECT_TRUE(transient_pool().history_fresh(history));
  EXPECT_EQ(reader_kind(), IdentityKind::History);
  EXPECT_EQ(valid, (std::vector<bool>{false, true, false, true}));
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'hs.gate'"), 1u);
}

TEST_F(GraphBindTest, MissedReleaseShowsStale) {
  const auto passes = steady_passes();
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(bind_cache_stats().staleEntries, 0u);

  transient_pool().set_release_listener(nullptr);
  const TestFrame empty{empty_graph()};
  force_memory_pressure();
  on_worker_frame_begin(2, &empty.state);
  ASSERT_TRUE(transient_pool().empty());
  const BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.entries, 3u);
  EXPECT_EQ(stats.staleEntries, 3u);
  EXPECT_EQ(stats.evicted, 0u);
}

TEST_F(GraphBindTest, GxViewChangeMissesOnceThenAgesOut) {
  use_globals({.efbSize = {64, 64, 1}, .createEfb = true});
  std::vector<TestPass> passes{
      {.name = "ga.read",
       .declare =
           [](InternalPassBuilder& b) {
             b.use("gx.color", Access::Sampled);
             b.side_effect();
           }},
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, 1u);

  webgpu::g_frameBuffer = webgpu::create_render_texture(64, 64, true);
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.entries, 2u);
  EXPECT_EQ(stats.evicted, 0u);

  for (uint64_t frameId = 3; frameId <= 50; ++frameId) {
    ASSERT_EQ(run_frame(frame, frameId), std::nullopt) << frameId;
    if (frameId == 47) {
      EXPECT_EQ(bind_cache_stats().entries, 2u);
    }
  }
  stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.entries, 1u);
  EXPECT_EQ(stats.evicted, 1u);
  EXPECT_EQ(stats.hits, 48u);
  EXPECT_EQ(stats.staleEntries, 0u);
}

TEST_F(GraphBindTest, CulledConsumerIsNotEncoded) {
  const auto passes = steady_passes();
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1, [&](StageData& stage) { set_bit(stage.enabled, frame.program_index("s.c"), false); }),
            std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(passes[1].calls, 1);
  EXPECT_EQ(passes[2].calls, 0);
  EXPECT_TRUE(testing::last_identities(frame.node("s.c")).empty());
}

TEST_F(GraphBindTest, UnresolvedPersistentImportSkips) {
  const PersistentHandle handle = create_persistent("u.buf", PersistentKind::Buffer);
  std::vector<TestPass> passes{
      {.name = "u.import",
       .order = 0,
       .declare =
           [handle](InternalPassBuilder& b) {
             b.import_persistent("u.buf", handle, Access::StorageRead);
             b.side_effect();
           }},
      {.name = "u.use",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use_buffer("u.buf", Access::StorageRead, Missing::BindDummyBlack);
             b.side_effect();
           }},
  };
  BufferView dummy;
  passes[1].encode = [&](PassContext& ctx) { dummy = ctx.buffer(BufRef{0}); };
  const TestFrame frame{compile(passes)};
  const CompiledUse& use = frame.use("u.use", 0);
  ASSERT_TRUE(use.hasFallback);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(passes[0].calls, 0);
  EXPECT_EQ(passes[1].calls, 2);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'u.buf'"), 1u);
  EXPECT_TRUE(dummy.buffer);
  EXPECT_EQ(dummy.size, use.fallback.bufferSize);
  const auto identities = testing::last_identities(frame.node("u.use"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Dummy);
}

TEST_F(GraphBindTest, GxNormalWithoutTextureIsMissing) {
  std::vector<TestPass> passes{
      {.name = "n.dummy",
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("gx.normal", Access::Sampled, Missing::BindDummyBlack);
             b.side_effect();
           }},
      {.name = "n.skip",
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("gx.normal", Access::Sampled);
             b.side_effect();
           }},
  };
  GraphInputs inputs = test_inputs();
  const auto normal = static_cast<size_t>(GxResource::Normal);
  inputs.env.gxFormats[normal] = webgpu::NormalBufferFormat;
  inputs.env.gxUsage[normal] = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
  const TestFrame frame{compile(passes, inputs)};
  ASSERT_EQ(frame.graph->nodes[frame.node("n.skip")].status, CullReason::None);
  ASSERT_FALSE(webgpu::g_normalBuffer.texture);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(passes[1].calls, 0);
  const auto identities = testing::last_identities(frame.node("n.dummy"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Dummy);
  EXPECT_EQ(count_logs(LOG_WARNING, "needs 'gx.normal'"), 1u);
}

// Null by default; hardware runs pick their backend from the environment
class GraphBindDeviceTest : public GraphBindTest {
protected:
  GraphBindDeviceTest() {
    m_options.backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Null);
    m_options.efbSize = {64, 64, 1};
  }
};

TEST_F(GraphBindDeviceTest, GxNormalBufferBindsAsSampled) {
  if (!webgpu::g_hasCoreFeatures) {
    GTEST_SKIP() << "the normal buffer needs a core device";
  }
  push_validation_scope();
  ASSERT_TRUE(webgpu::enable_normal_buffer());
  ASSERT_EQ(pop_validation_scope(), std::nullopt);
  const webgpu::TextureWithSampler& normal = webgpu::g_normalBuffer;
  ASSERT_TRUE(normal.texture);
  const wgpu::TextureUsage usage = normal.texture.GetUsage();
  EXPECT_TRUE(usage & wgpu::TextureUsage::TextureBinding);
  EXPECT_TRUE(usage & wgpu::TextureUsage::RenderAttachment);
  EXPECT_TRUE(usage & wgpu::TextureUsage::CopySrc);

  GraphInputs inputs = test_inputs();
  const auto index = static_cast<size_t>(GxResource::Normal);
  // The graph plans against the environment, so it has to describe the texture gpu.cpp made
  EXPECT_EQ(inputs.env.gxFormats[index], normal.texture.GetFormat());
  EXPECT_EQ(inputs.env.gxUsage[index], usage);

  const std::string writeSource = std::string{R"(
@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.5, 0.5, 1.0, 1.0);
}
)"} + std::string{FullscreenVertexWgsl};
  const std::string readSource = std::string{R"(
@group(1) @binding(0) var nb_normal: texture_2d<f32>;

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(nb_normal, vec2<u32>(position.xy), 0);
}
)"} + std::string{FullscreenVertexWgsl};
  std::vector<TestPass> passes{
      {.name = "nb.write",
       .kind = PassKind::Render,
       .order = 0,
       .declare = [](InternalPassBuilder& b) { b.use("gx.normal", Access::ColorRMW); }},
      {.name = "nb.read",
       .kind = PassKind::Render,
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             // A refused resolve then shows up as a dummy binding rather than a skipped pass
             b.use("gx.normal", Access::Sampled, Missing::BindDummyBlack);
             b.create("nb.out", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           }},
  };
  const auto draw = [](PassContext& ctx, const std::string& source, TexRef target) {
    const wgpu::RenderPipeline pipeline = render_pipeline(ctx.pipeline_layout(), source, ctx.format(target));
    const wgpu::RenderPassEncoder pass = ctx.begin_render();
    pass.SetPipeline(pipeline);
    pass.Draw(3);
    pass.End();
  };
  passes[0].encode = [&](PassContext& ctx) { draw(ctx, writeSource, TexRef{0}); };
  passes[1].encode = [&](PassContext& ctx) { draw(ctx, readSource, TexRef{1}); };
  const TestFrame frame{compile(passes, inputs)};
  ASSERT_EQ(frame.graph->nodes[frame.node("nb.write")].status, CullReason::None);
  ASSERT_EQ(frame.graph->nodes[frame.node("nb.read")].status, CullReason::None);
  EXPECT_EQ(frame.graph->bindingPlans[frame.node("nb.write")].colors.size(), 1u);
  EXPECT_FALSE(frame.use("nb.read", 0).staticDummy);
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("nb.read")];
  ASSERT_EQ(plan.slots.size(), 1u);
  EXPECT_EQ(plan.slots[0].type, BindingType::SampledTexture);
  EXPECT_EQ(plan.slots[0].viewDimension, TextureViewDimension::e2D);
  EXPECT_EQ(plan.slots[0].sampleType, TextureSampleType::Float);

  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(passes[1].calls, 1);
  const auto identities = testing::last_identities(frame.node("nb.read"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Gx);
  EXPECT_EQ(identities[0].value, reinterpret_cast<uintptr_t>(normal.view.Get()));
  EXPECT_EQ(count_logs(LOG_WARNING, "needs 'gx.normal'"), 0u);
}

TEST_F(GraphBindDeviceTest, GxGBufferTargetsBindAsSampled) {
  if (!webgpu::g_hasCoreFeatures) {
    GTEST_SKIP() << "the G-buffer needs a core device";
  }
  push_validation_scope();
  ASSERT_TRUE(webgpu::set_gbuffer(true));
  ASSERT_EQ(pop_validation_scope(), std::nullopt);
  struct Target {
    GxResource resource;
    const webgpu::TextureWithSampler* texture;
  };
  const std::array targets{
      Target{GxResource::Motion, &webgpu::g_motionBuffer},
      Target{GxResource::Reactive, &webgpu::g_reactiveBuffer},
      Target{GxResource::Fog, &webgpu::g_fogRecordBuffer},
  };
  GraphInputs inputs = test_inputs();
  for (const Target& target : targets) {
    const auto index = static_cast<size_t>(target.resource);
    ASSERT_TRUE(target.texture->texture) << gx_resource_name(target.resource);
    const wgpu::TextureUsage usage = target.texture->texture.GetUsage();
    EXPECT_EQ(target.texture->texture.GetFormat(), gx_fixed_format(target.resource));
    EXPECT_EQ(target.texture->texture.GetSampleCount(), 1u);
    EXPECT_EQ(inputs.env.gxFormats[index], target.texture->texture.GetFormat());
    EXPECT_EQ(inputs.env.gxUsage[index], usage);
    EXPECT_TRUE(usage & wgpu::TextureUsage::CopySrc);
  }
  const auto normal = static_cast<size_t>(GxResource::Normal);
  ASSERT_TRUE(webgpu::g_normalBuffer.texture);
  EXPECT_EQ(inputs.env.gxFormats[normal], webgpu::NormalBufferFormat);

  std::vector<TestPass> passes{
      {.name = "gb.read",
       .order = 0,
       .declare =
           [&targets](InternalPassBuilder& b) {
             for (const Target& target : targets) {
               b.use(gx_resource_name(target.resource), Access::Sampled, Missing::BindDummyBlack);
             }
             b.use("gx.normal", Access::Sampled, Missing::BindDummyBlack);
             b.side_effect();
           }},
  };
  const TestFrame frame{compile(passes, inputs)};
  ASSERT_EQ(frame.graph->nodes[frame.node("gb.read")].status, CullReason::None);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  const auto identities = testing::last_identities(frame.node("gb.read"));
  ASSERT_EQ(identities.size(), targets.size() + 1);
  for (size_t i = 0; i < targets.size(); ++i) {
    EXPECT_EQ(identities[i].kind, IdentityKind::Gx) << i;
    EXPECT_EQ(identities[i].value, reinterpret_cast<uintptr_t>(targets[i].texture->view.Get())) << i;
  }
  EXPECT_EQ(identities[targets.size()].kind, IdentityKind::Gx);
  EXPECT_EQ(identities[targets.size()].value, reinterpret_cast<uintptr_t>(webgpu::g_normalBuffer.view.Get()));

  ASSERT_TRUE(webgpu::set_gbuffer(false));
  for (const Target& target : targets) {
    EXPECT_FALSE(target.texture->texture) << gx_resource_name(target.resource);
  }
  EXPECT_FALSE(webgpu::g_normalBuffer.texture);
  inputs = test_inputs();
  EXPECT_EQ(inputs.env.gxFormats[static_cast<size_t>(GxResource::Motion)], TextureFormat::Undefined);
  EXPECT_EQ(inputs.env.gxFormats[normal], TextureFormat::Undefined);

  ASSERT_TRUE(webgpu::set_gbuffer(true));
  const WGPUTexture shared = webgpu::g_normalBuffer.texture.Get();
  ASSERT_TRUE(webgpu::enable_normal_buffer());
  EXPECT_EQ(webgpu::g_normalBuffer.texture.Get(), shared);
  ASSERT_TRUE(webgpu::set_gbuffer(false));
  EXPECT_EQ(webgpu::g_normalBuffer.texture.Get(), shared);
  EXPECT_FALSE(webgpu::g_motionBuffer.texture);
  EXPECT_EQ(test_inputs().env.gxFormats[normal], webgpu::NormalBufferFormat);
}

TEST_F(GraphBindTest, LoadOpsFollowCreator) {
  using testing::AttachmentOps;
  std::vector<std::vector<AttachmentOps>> recorded;
  const auto record = [&](const wgpu::RenderPassEncoder& pass) {
    pass.End();
    recorded.push_back(testing::last_attachment_ops());
  };
  std::vector<TestPass> passes{
      {.name = "l.create",
       .kind = PassKind::Render,
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("l.color", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.create("l.depth", absolute(TextureFormat::Depth32Float), Access::DepthWrite);
           }},
      {.name = "l.load",
       .kind = PassKind::Render,
       .order = 1,
       .declare = [](InternalPassBuilder& b) { b.use("l.color", Access::ColorWrite); }},
      {.name = "l.rmw",
       .kind = PassKind::Render,
       .order = 2,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("l.color", Access::ColorRMW);
             b.side_effect();
           }},
  };
  passes[0].encode = [&](PassContext& ctx) { record(ctx.begin_render()); };
  passes[1].encode = [&](PassContext& ctx) {
    record(ctx.begin_render());
    const std::array targets{ColorTarget{.ref = TexRef{0}, .clear = wgpu::Color{1, 0, 0, 1}}};
    record(ctx.begin_render(targets, nullptr));
  };
  passes[2].encode = [&](PassContext& ctx) { record(ctx.begin_render()); };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(recorded.size(), 4u);

  ASSERT_EQ(recorded[0].size(), 2u);
  EXPECT_FALSE(recorded[0][0].depth);
  EXPECT_EQ(recorded[0][0].load, wgpu::LoadOp::Clear);
  EXPECT_EQ(recorded[0][0].clearColor.a, 0.0);
  EXPECT_EQ(recorded[0][0].store, wgpu::StoreOp::Store);
  EXPECT_TRUE(recorded[0][1].depth);
  EXPECT_EQ(recorded[0][1].load, wgpu::LoadOp::Clear);
  EXPECT_EQ(recorded[0][1].clearDepth, 0.0f);
  EXPECT_EQ(recorded[0][1].width, 64u);

  ASSERT_EQ(recorded[1].size(), 1u);
  EXPECT_EQ(recorded[1][0].load, wgpu::LoadOp::Load);
  ASSERT_EQ(recorded[2].size(), 1u);
  EXPECT_EQ(recorded[2][0].load, wgpu::LoadOp::Clear);
  EXPECT_EQ(recorded[2][0].clearColor.r, 1.0);
  ASSERT_EQ(recorded[3].size(), 1u);
  EXPECT_EQ(recorded[3][0].load, wgpu::LoadOp::Load);
  EXPECT_FALSE(recorded[3][0].resolves);
}

TEST_F(GraphBindTest, DepthStencilAttachmentValidates) {
  std::vector<TestPass> passes{
      {.name = "ds.pass",
       .kind = PassKind::Render,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("ds.depth", absolute(TextureFormat::Depth24PlusStencil8), Access::DepthWrite);
             b.side_effect();
           }},
  };
  std::vector<testing::AttachmentOps> ops;
  passes[0].encode = [&](PassContext& ctx) {
    ctx.begin_render().End();
    ops = testing::last_attachment_ops();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_TRUE(ops[0].depth);
  EXPECT_EQ(ops[0].load, wgpu::LoadOp::Clear);
}

TEST_F(GraphBindTest, MipAttachmentValidates) {
  std::vector<TestPass> passes{
      {.name = "mip.pass",
       .kind = PassKind::Render,
       .declare =
           [](InternalPassBuilder& b) {
             TexDesc desc = absolute(TextureFormat::RGBA16Float);
             desc.mips = 3;
             b.create("mip.tex", desc, Access::ColorWrite);
             b.side_effect();
           }},
  };
  std::vector<testing::AttachmentOps> ops;
  passes[0].encode = [&](PassContext& ctx) {
    const std::array targets{ColorTarget{.ref = TexRef{0}, .mip = 1}};
    ctx.begin_render(targets, nullptr).End();
    ops = testing::last_attachment_ops();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].width, 32u);
  EXPECT_EQ(ops[0].height, 32u);
}

TEST_F(GraphBindTest, LayerAttachmentValidates) {
  std::vector<TestPass> passes{
      {.name = "layer.array",
       .kind = PassKind::Render,
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("layer.a", absolute(TextureFormat::RGBA8Unorm, 64, 64, 4), Access::ColorWrite);
             b.side_effect();
           }},
      {.name = "layer.volume",
       .kind = PassKind::Render,
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             TexDesc desc = absolute(TextureFormat::RGBA16Float, 16, 16, 8);
             desc.dimension = wgpu::TextureDimension::e3D;
             b.create("layer.v", desc, Access::ColorWrite);
             b.side_effect();
           }},
  };
  passes[0].encode = [](PassContext& ctx) {
    const std::array targets{ColorTarget{.ref = TexRef{0}, .layer = 2}};
    ctx.begin_render(targets, nullptr).End();
  };
  passes[1].encode = [](PassContext& ctx) {
    const std::array targets{ColorTarget{.ref = TexRef{0}, .layer = 5}};
    ctx.begin_render(targets, nullptr).End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(passes[1].calls, 1);
}

TEST_F(GraphBindTest, MsaaColorRmwResolves) {
  use_globals({.efbSize = {64, 64, 1}, .msaaSamples = 4, .createEfb = true});
  ASSERT_TRUE(webgpu::g_frameBufferResolved.texture);
  const std::string source = std::string{R"(
@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
)"} + std::string{FullscreenVertexWgsl};
  std::vector<TestPass> passes{
      {.name = "msaa.rmw",
       .kind = PassKind::Render,
       .declare = [](InternalPassBuilder& b) { b.use("gx.color", Access::ColorRMW); }},
  };
  uint32_t samples = 0;
  std::vector<testing::AttachmentOps> ops;
  passes[0].encode = [&](PassContext& ctx) {
    samples = ctx.sample_count(TexRef{0});
    const wgpu::RenderPipeline pipeline =
        render_pipeline(ctx.pipeline_layout(), source, ctx.format(TexRef{0}), samples);
    const wgpu::RenderPassEncoder pass = ctx.begin_render();
    ops = testing::last_attachment_ops();
    pass.SetPipeline(pipeline);
    pass.Draw(3);
    pass.End();
  };
  GraphInputs inputs = test_inputs();
  ASSERT_EQ(inputs.env.gxSampleCount, 4u);
  const TestFrame frame{compile(passes, inputs)};
  ASSERT_EQ(frame.graph->nodes[frame.node("msaa.rmw")].status, CullReason::None);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(samples, 4u);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_TRUE(ops[0].resolves);
  EXPECT_EQ(ops[0].load, wgpu::LoadOp::Load);
  EXPECT_EQ(ops[0].width, 64u);
}

TEST_F(GraphBindTest, ManualPassBindsOnlyGroup0) {
  const std::string frameOnly = frame_shader(R"(
@compute @workgroup_size(1)
fn main() {
  _ = aurora_frame.frame_index;
}
)");
  const std::string ownGroup = frame_shader(R"(
@group(1) @binding(0) var<storage, read_write> own: array<u32>;
@compute @workgroup_size(1)
fn main() {
  own[0] = aurora_frame.frame_index;
}
)");
  std::vector<TestPass> passes{
      {.name = "manual.pass",
       .declare =
           [](InternalPassBuilder& b) {
             b.create_buffer("manual.buf", {.size = 256}, Access::StorageWrite);
             b.manual_bindings();
             b.side_effect();
           }},
  };
  std::optional<std::string> frameOnlyError;
  std::optional<std::string> ownWithPassLayout;
  passes[0].encode = [&](PassContext& ctx) {
    EXPECT_TRUE(ctx.pass_layout());
    EXPECT_FALSE(ctx.pass_group());
    push_validation_scope();
    compute_pipeline(ctx.pipeline_layout(), frameOnly);
    frameOnlyError = pop_validation_scope();
    push_validation_scope();
    compute_pipeline(ctx.pipeline_layout(), ownGroup);
    ownWithPassLayout = pop_validation_scope();

    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Storage},
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.entryCount = 1, .entries = &entry};
    const wgpu::BindGroupLayout own = ctx.device().CreateBindGroupLayout(&layoutDescriptor);
    const std::array layouts{ctx.frame_layout(), own};
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = layouts.size(),
                                                                  .bindGroupLayouts = layouts.data()};
    const wgpu::ComputePipeline pipeline =
        compute_pipeline(ctx.device().CreatePipelineLayout(&pipelineLayoutDescriptor), ownGroup);
    const BufferView buffer = ctx.buffer(BufRef{0});
    const wgpu::BindGroupEntry binding{.binding = 0, .buffer = buffer.buffer, .size = buffer.size};
    const wgpu::BindGroupDescriptor groupDescriptor{.layout = own, .entryCount = 1, .entries = &binding};
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(1, ctx.device().CreateBindGroup(&groupDescriptor));
    pass.DispatchWorkgroups(1);
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(frameOnlyError, std::nullopt);
  EXPECT_NE(ownWithPassLayout, std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, 0u);
}

TestPass params_pass(uint32_t paramsSize) {
  return {
      .name = "par.pass",
      .paramsSize = paramsSize,
      .declare =
          [](InternalPassBuilder& b) {
            b.create_buffer("par.out", {.size = 256}, Access::StorageWrite);
            b.side_effect();
          },
  };
}

std::vector<std::byte> pattern(size_t size, uint8_t first) {
  std::vector<std::byte> bytes(size);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>(first + i);
  }
  return bytes;
}

std::vector<std::byte> params_bytes(const StageData& stage) {
  if (stage.bindings.params.empty()) {
    return {};
  }
  const auto& params = stage.bindings.params.front();
  return {params.bytes.begin(), params.bytes.begin() + params.size};
}

class GraphBindParamsTest : public GraphBindTest {
protected:
  void register_params_pass(uint32_t paramsSize) {
    m_passes = {params_pass(paramsSize)};
    m_id = register_pass(desc_of(m_passes[0]));
    ASSERT_NE(m_id, InvalidPassId);
  }

  [[nodiscard]] TestFrame compile_params() const { return TestFrame{compile(m_passes, test_inputs(), {&m_id, 1})}; }

  std::vector<TestPass> m_passes;
  PassId m_id = InvalidPassId;
};

TEST_F(GraphBindParamsTest, ParamsReachEncode) {
  register_params_pass(48);
  std::vector<std::byte> seen;
  uint32_t offset = UINT32_MAX;
  m_passes[0].encode = [&](PassContext& ctx) {
    seen.assign(ctx.params().begin(), ctx.params().end());
    offset = ctx.params_offset();
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.End();
  };
  const auto payload = pattern(40, 1);
  ASSERT_TRUE(set_pass_params(m_id, payload));
  latch_pass_params();
  const TestFrame frame = compile_params();
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);

  auto expected = payload;
  expected.resize(48);
  EXPECT_EQ(seen, expected);
  const uint32_t alignment = gfx::detail::resources().limits.minUniformBufferOffsetAlignment;
  ASSERT_NE(offset, UINT32_MAX);
  EXPECT_EQ(offset % alignment, 0u);
  EXPECT_EQ(std::ranges::count(g_pushes, MaxPassParamsSize), 1);
  EXPECT_EQ(std::ranges::count(g_pushes, FrameUniformsSize), 1);
  ASSERT_GE(g_uniforms.size(), offset + MaxPassParamsSize);
  EXPECT_EQ(std::memcmp(g_uniforms.data() + offset, payload.data(), payload.size()), 0);
  EXPECT_TRUE(std::all_of(g_uniforms.begin() + offset + 40, g_uniforms.begin() + offset + MaxPassParamsSize,
                          [](uint8_t value) { return value == 0; }));
}

TEST_F(GraphBindParamsTest, ParamsOver128Rejected) {
  register_params_pass(16);
  const auto first = pattern(16, 7);
  ASSERT_TRUE(set_pass_params(m_id, first));
  latch_pass_params();
  EXPECT_FALSE(set_pass_params(m_id, pattern(129, 0)));
  EXPECT_EQ(count_logs(LOG_WARNING, "payload size 129 exceeds 128"), 1u);
  EXPECT_FALSE(set_pass_params(InvalidPassId, first));
  latch_pass_params();
  const auto latched = latched_pass_params(m_id);
  EXPECT_TRUE(std::ranges::equal(latched, first));
}

TEST_F(GraphBindParamsTest, ParamsAreSticky) {
  register_params_pass(16);
  const auto payload = pattern(16, 3);
  ASSERT_TRUE(set_pass_params(m_id, payload));
  const TestFrame frame = compile_params();
  for (uint64_t frameId = 1; frameId <= 3; ++frameId) {
    latch_pass_params();
    StageData stage = stage_for(frame, Stage::OpaqueDone, frameId);
    build_stage_bindings(stage, &fake_sink);
    EXPECT_EQ(params_bytes(stage), payload) << frameId;
  }
  clear_pass_params(m_id);
  latch_pass_params();
  StageData stage = stage_for(frame, Stage::OpaqueDone, 4);
  build_stage_bindings(stage, &fake_sink);
  EXPECT_EQ(params_bytes(stage), std::vector<std::byte>(16));
}

TEST_F(GraphBindParamsTest, UnsetParamsReadZero) {
  register_params_pass(16);
  latch_pass_params();
  const TestFrame frame = compile_params();
  StageData stage = stage_for(frame, Stage::OpaqueDone, 1);
  build_stage_bindings(stage, &fake_sink);
  ASSERT_EQ(stage.bindings.params.size(), 1u);
  const uint32_t offset = stage.bindings.params[0].uniformOffset;
  EXPECT_EQ(std::ranges::count(g_pushes, MaxPassParamsSize), 1);
  ASSERT_GE(g_uniforms.size(), offset + MaxPassParamsSize);
  EXPECT_TRUE(std::all_of(g_uniforms.begin() + offset, g_uniforms.begin() + offset + MaxPassParamsSize,
                          [](uint8_t value) { return value == 0; }));
  EXPECT_EQ(params_bytes(stage), std::vector<std::byte>(16));
}

TEST_F(GraphBindParamsTest, ParamsApplyFromNextLatch) {
  register_params_pass(8);
  const auto a = pattern(8, 10);
  const auto b = pattern(8, 50);
  const TestFrame frame = compile_params();
  ASSERT_TRUE(set_pass_params(m_id, a));
  latch_pass_params();
  ASSERT_TRUE(set_pass_params(m_id, b));
  StageData before = stage_for(frame, Stage::OpaqueDone, 1);
  build_stage_bindings(before, &fake_sink);
  EXPECT_EQ(params_bytes(before), a);
  latch_pass_params();
  StageData after = stage_for(frame, Stage::OpaqueDone, 2);
  build_stage_bindings(after, &fake_sink);
  EXPECT_EQ(params_bytes(after), b);
}

TEST_F(GraphBindParamsTest, UnregisterDropsParams) {
  register_params_pass(8);
  ASSERT_TRUE(set_pass_params(m_id, pattern(8, 1)));
  latch_pass_params();
  EXPECT_EQ(latched_pass_params(m_id).size(), 8u);
  unregister_pass(m_id);
  EXPECT_FALSE(set_pass_params(m_id, pattern(8, 1)));
  latch_pass_params();
  EXPECT_TRUE(latched_pass_params(m_id).empty());
  const PassId again = register_pass(desc_of(m_passes[0]));
  ASSERT_NE(again, InvalidPassId);
  EXPECT_NE(again, m_id);
  latch_pass_params();
  EXPECT_TRUE(latched_pass_params(again).empty());
}

TEST_F(GraphBindTest, FrameConstantsPushedOncePerFrame) {
  std::vector<TestPass> passes{
      {.name = "fc.opaque", .stage = Stage::OpaqueDone, .declare = [](InternalPassBuilder& b) { b.side_effect(); }},
      {.name = "fc.post", .stage = Stage::PrePost, .declare = [](InternalPassBuilder& b) { b.side_effect(); }},
  };
  GraphInputs inputs = test_inputs();
  inputs.env.efbSize = {640, 480};
  inputs.env.outputSize = {1280, 960};
  const TestFrame frame{compile(passes, inputs)};
  StageData opaque = stage_for(frame, Stage::OpaqueDone, 7);
  StageData post = stage_for(frame, Stage::PrePost, 7);
  build_stage_bindings(opaque, &fake_sink);
  build_stage_bindings(post, &fake_sink);
  ASSERT_EQ(g_pushes, (std::vector<size_t>{FrameUniformsSize}));
  EXPECT_EQ(opaque.bindings.constantsOffset, post.bindings.constantsOffset);

  const auto word = [&](size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, g_uniforms.data() + opaque.bindings.constantsOffset + offset, sizeof(value));
    return value;
  };
  const auto real = [&](size_t offset) {
    float value = 0;
    std::memcpy(&value, g_uniforms.data() + opaque.bindings.constantsOffset + offset, sizeof(value));
    return value;
  };
  EXPECT_EQ(word(0), 640u);
  EXPECT_EQ(word(4), 480u);
  EXPECT_EQ(word(8), 1280u);
  EXPECT_EQ(word(12), 960u);
  EXPECT_FLOAT_EQ(real(16), 1.0f / 640.0f);
  EXPECT_FLOAT_EQ(real(20), 1.0f / 480.0f);
  EXPECT_FLOAT_EQ(real(24), 1.0f / 1280.0f);
  EXPECT_FLOAT_EQ(real(28), 1.0f / 960.0f);
  EXPECT_EQ(word(32), 7u);
  EXPECT_EQ(word(36) & FrameFlagReversedZ, FrameFlagReversedZ);
  EXPECT_EQ(word(36) & FrameFlagMultisampled, 0u);
  EXPECT_EQ(word(40), 1u);

  StageData next = stage_for(frame, Stage::OpaqueDone, 8);
  build_stage_bindings(next, &fake_sink);
  EXPECT_EQ(g_pushes.size(), 2u);
  EXPECT_NE(next.bindings.constantsOffset, opaque.bindings.constantsOffset);
}

TEST_F(GraphBindTest, EmptyProgramPushesNothing) {
  std::vector<TestPass> passes{
      {.name = "e.opaque", .declare = [](InternalPassBuilder& b) { b.side_effect(); }},
  };
  const TestFrame frame{compile(passes)};
  ASSERT_TRUE(frame.graph->stage_empty(Stage::SceneBegin));
  StageData empty{.stage = Stage::SceneBegin, .frameId = 1, .frame = &frame.state};
  EXPECT_FALSE(build_stage_data(empty));
  EXPECT_TRUE(empty.bindings.params.empty());
  // No frame is recording, so a push from the default sink would warn
  EXPECT_EQ(count_logs(LOG_WARNING, "push_uniform"), 0u);

  StageData opaque{.stage = Stage::OpaqueDone, .frameId = 1, .markerSeen = true, .frame = &frame.state};
  EXPECT_TRUE(build_stage_data(opaque));
  EXPECT_EQ(count_logs(LOG_WARNING, "push_uniform"), 1u);
}

TEST_F(GraphBindTest, NoPassesCreateNothing) {
  testing::reset();
  const std::shared_ptr<const FrameState> state = latch_frame();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->graph, empty_graph());
  EXPECT_FALSE(testing::frame_objects_created());
  for (size_t s = 0; s < StageCount; ++s) {
    StageData stage{.stage = static_cast<Stage>(s), .frameId = 1, .markerSeen = true, .frame = state.get()};
    EXPECT_FALSE(build_stage_data(stage));
  }
  on_worker_frame_begin(1, state.get());
  const BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 0u);
  EXPECT_EQ(stats.entries, 0u);
  EXPECT_EQ(count_logs(LOG_WARNING, "push_uniform"), 0u);
  EXPECT_FALSE(testing::frame_objects_created());

  const TestPass pass{.name = "np.pass", .declare = [](InternalPassBuilder& b) { b.side_effect(); }};
  ASSERT_NE(register_pass(desc_of(pass)), InvalidPassId);
  const std::shared_ptr<const FrameState> registered = latch_frame();
  ASSERT_NE(registered->graph, empty_graph());
  EXPECT_TRUE(testing::frame_objects_created());
  const NodeId node = registered->graph->find_node("np.pass");
  ASSERT_NE(node, InvalidNode);
  EXPECT_TRUE(registered->graph->bindingPlans[node].layout);
  testing::reset();
}

TEST_F(GraphBindTest, UndeclaredAccessChecks) {
  std::vector<TestPass> passes{
      {.name = "ua.make",
       .order = 0,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("ua.src", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "ua.pass",
       .kind = PassKind::Render,
       .order = 1,
       .declare =
           [](InternalPassBuilder& b) {
             b.use("ua.src", Access::Sampled);
             b.create("ua.color", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           }},
  };
  passes[1].encode = [](PassContext& ctx) {
    const std::array sampled{ColorTarget{.ref = TexRef{0}}};
#ifndef NDEBUG
    EXPECT_DEATH(static_cast<void>(ctx.view(TexRef{99})), "");
    EXPECT_DEATH(static_cast<void>(ctx.begin_render(sampled, nullptr)), "");
#else
    EXPECT_FALSE(ctx.view(TexRef{99}));
    EXPECT_FALSE(ctx.texture(TexRef{99}));
    EXPECT_FALSE(ctx.begin_render(sampled, nullptr));
    EXPECT_FALSE(ctx.buffer(BufRef{0}).buffer);
    EXPECT_FALSE(ctx.prev(TexRef{0}));
    EXPECT_FALSE(ctx.begin_compute());
    EXPECT_EQ(ctx.extent(TexRef{99}).width, 0u);
#endif
    EXPECT_TRUE(ctx.view(TexRef{0}));
    EXPECT_EQ(ctx.extent(TexRef{0}).width, 64u);
    EXPECT_EQ(ctx.format(TexRef{1}), TextureFormat::RGBA8Unorm);
    EXPECT_EQ(ctx.sample_count(TexRef{1}), 1u);
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[1].calls, 1);
}

TEST_F(GraphBindTest, BeginComputeValidates) {
  std::vector<TestPass> passes{
      {.name = "bc.pass",
       .declare =
           [](InternalPassBuilder& b) {
             b.create_buffer("bc.out", {.size = 256}, Access::StorageWrite);
             b.side_effect();
           }},
  };
  passes[0].encode = [](PassContext& ctx) {
    EXPECT_EQ(ctx.stage(), Stage::OpaqueDone);
    EXPECT_EQ(ctx.stage_data().stage, Stage::OpaqueDone);
    EXPECT_EQ(ctx.name(), "bc.pass");
    EXPECT_EQ(ctx.frame_id(), 1u);
    EXPECT_TRUE(ctx.frame_group());
    EXPECT_TRUE(ctx.pass_group());
    EXPECT_TRUE(ctx.pipeline_layout());
    EXPECT_NE(ctx.layout_hash(), 0u);
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    ASSERT_TRUE(pass);
    pass.End();
    ctx.begin_compute("second").End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
}

TEST_F(GraphBindTest, BindIdentityKindsNeverCollide) {
  const BindIdentity dummy{.kind = IdentityKind::Dummy, .value = 42};
  const BindIdentity transient{.kind = IdentityKind::Transient, .value = 42};
  EXPECT_NE(dummy, transient);
  EXPECT_NE(bind_group_key(1, std::array{dummy}), bind_group_key(1, std::array{transient}));
  EXPECT_NE(bind_group_key(1, std::array{transient}), bind_group_key(2, std::array{transient}));
  const BindIdentity storage{.kind = IdentityKind::Transient, .selector = 1, .value = 42};
  EXPECT_NE(bind_group_key(1, std::array{transient}), bind_group_key(1, std::array{storage}));
  EXPECT_EQ(bind_group_key(1, std::array{transient}), bind_group_key(1, std::array{transient}));
}

constexpr uint64_t PipelineWaitNs = 5'000'000'000;

constexpr std::string_view GraphFillWgsl = R"(//#import aurora/frame
@group(1) @binding(0) var<storage, read_write> fill_out: array<u32>;
@compute @workgroup_size(1)
fn main() {
  fill_out[0] = aurora_frame.frame_index;
}
)";

constexpr std::string_view GraphBrokenWgsl = R"(@compute @workgroup_size(1)
fn main() {
  let bad: u32 = 1.5;
}
)";

class GraphBindPipelineTest : public GraphBindTest {
protected:
  void SetUp() override {
    GraphBindTest::SetUp();
    if (!m_device) {
      return;
    }
    jobs::initialize(1);
    shaders::initialize();
    ASSERT_TRUE(shaders::register_source("test/graph_fill", GraphFillWgsl));
    ASSERT_TRUE(shaders::register_source("test/graph_broken", GraphBrokenWgsl));
  }

  void TearDown() override {
    if (m_device) {
      compact::shutdown();
      shaders::shutdown();
      jobs::shutdown();
    }
    GraphBindTest::TearDown();
  }

  static TestPass fill_pass(std::string_view shader) {
    TestPass pass{
        .name = "gp.fill",
        .declare =
            [](InternalPassBuilder& b) {
              b.create_buffer("gp.buf", {.size = 256}, Access::StorageWrite);
              b.side_effect();
            },
    };
    pass.pipelines = [shader](const PipelineSetup& setup, std::span<uint32_t, MaxPassPipelines> ids) {
      ids[0] = shaders::request_compute(
                   {.label = "gp.fill", .shader = shader, .layout = setup.layout, .layoutKey = setup.layoutHash})
                   .value;
    };
    pass.encode = [](PassContext& ctx) {
      const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline());
      ASSERT_TRUE(pipeline);
      const wgpu::ComputePassEncoder pass = ctx.begin_compute();
      pass.SetPipeline(pipeline);
      pass.DispatchWorkgroups(1);
      pass.End();
    };
    return pass;
  }

  static TestPass reader_pass() {
    return {
        .name = "gp.read",
        .order = 1,
        .declare =
            [](InternalPassBuilder& b) {
              b.use_buffer("gp.buf", Access::StorageRead);
              b.side_effect();
            },
        .encodes = false,
    };
  }

  // Every frame starts at SceneBegin, which resets the cull state the later stages read
  static StageData build_opaque(const TestFrame& frame, uint64_t frameId) {
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = &frame.state};
    build_stage_data(begin);
    StageData opaque{.stage = Stage::OpaqueDone, .frameId = frameId, .markerSeen = true, .frame = &frame.state};
    build_stage_data(opaque);
    return opaque;
  }
};

TEST_F(GraphBindPipelineTest, HookRequestsWithPassLayout) {
  std::vector<TestPass> passes{fill_pass("test/graph_fill")};
  std::optional<PipelineSetup> seen;
  const auto request = passes[0].pipelines;
  passes[0].pipelines = [&](const PipelineSetup& setup, std::span<uint32_t, MaxPassPipelines> ids) {
    seen = setup;
    request(setup, ids);
  };
  const TestFrame frame{compile(passes)};
  const NodeId node = frame.node("gp.fill");
  const CompiledNode& compiled = frame.graph->nodes[node];
  const BindingPlan& plan = frame.graph->bindingPlans[node];
  ASSERT_TRUE(seen);
  ASSERT_TRUE(plan.layout);
  EXPECT_NE(seen->pass, nullptr);
  EXPECT_EQ(seen->frameLayout.Get(), frame_layout().Get());
  EXPECT_EQ(seen->passLayout.Get(), plan.layout->passLayout.Get());
  EXPECT_EQ(seen->layout.Get(), plan.layout->pipelineLayout.Get());
  EXPECT_EQ(seen->layoutHash, plan.layoutHash);
  EXPECT_EQ(seen->local.data(), compiled.local.data());
  ASSERT_NE(compiled.pipelineIds[0], 0u);
  EXPECT_EQ(compiled.pipelineIds[1], 0u);
  ASSERT_EQ(shaders::wait({compiled.pipelineIds[0]}, PipelineWaitNs), shaders::PipelineStatus::Ready);

  const auto encode = passes[0].encode;
  passes[0].encode = [&](PassContext& ctx) {
    EXPECT_EQ(ctx.pipeline().value, compiled.pipelineIds[0]);
    EXPECT_FALSE(ctx.pipeline(MaxPassPipelines));
    encode(ctx);
  };
  EXPECT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);

  // Equal declarations give an equal layout hash, so a recompile reuses the pipeline
  const TestFrame again{compile(passes)};
  EXPECT_EQ(again.graph->nodes[again.node("gp.fill")].pipelineIds, compiled.pipelineIds);
  EXPECT_EQ(shaders::stats().ready, 1u);
}

TEST_F(GraphBindPipelineTest, PendingPipelineGatesPassAndConsumers) {
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
  const std::vector<TestPass> passes{fill_pass("test/graph_fill"), reader_pass()};
  const TestFrame frame{compile(passes)};
  const uint32_t id = frame.graph->nodes[frame.node("gp.fill")].pipelineIds[0];
  ASSERT_NE(id, 0u);
  EXPECT_EQ(shaders::status({id}), shaders::PipelineStatus::Pending);

  const StageData pending = build_opaque(frame, 1);
  EXPECT_FALSE(pending.is_enabled(frame.program_index("gp.fill")));
  EXPECT_FALSE(pending.is_enabled(frame.program_index("gp.read")));

  release = true;
  gate.wait();
  ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Ready);
  const StageData ready = build_opaque(frame, 2);
  EXPECT_TRUE(ready.is_enabled(frame.program_index("gp.fill")));
  EXPECT_TRUE(ready.is_enabled(frame.program_index("gp.read")));
}

TEST_F(GraphBindPipelineTest, FailedPipelineKeepsPassDisabled) {
  const std::vector<TestPass> passes{fill_pass("test/graph_broken"), reader_pass()};
  const TestFrame frame{compile(passes)};
  const uint32_t id = frame.graph->nodes[frame.node("gp.fill")].pipelineIds[0];
  ASSERT_EQ(shaders::wait({id}, PipelineWaitNs), shaders::PipelineStatus::Failed);
  for (uint64_t frameId = 1; frameId <= 3; ++frameId) {
    const StageData stage = build_opaque(frame, frameId);
    EXPECT_FALSE(stage.is_enabled(frame.program_index("gp.fill")));
    EXPECT_FALSE(stage.is_enabled(frame.program_index("gp.read")));
  }
  EXPECT_EQ(count_logs(LOG_WARNING, "test/graph_broken.wgsl:3:"), 1u);
  EXPECT_EQ(passes[0].calls, 0);
}

TEST_F(GraphBindPipelineTest, HookWithoutPipelineStaysDisabled) {
  debug::testing::reset();
  std::vector<TestPass> passes{fill_pass("test/graph_fill"), reader_pass()};
  passes[0].pipelines = [](const PipelineSetup&, std::span<uint32_t, MaxPassPipelines> ids) { ids[0] = 0; };
  TestFrame frame{compile(passes)};
  frame.state.inspectorOpen = true;
  const CompiledNode& node = frame.graph->nodes[frame.node("gp.fill")];
  ASSERT_EQ(node.status, CullReason::None);
  EXPECT_EQ(node.pipelineIds[0], 0u);
  EXPECT_TRUE(shaders::pipelines_ready(node.pipelineIds));
  EXPECT_FALSE(pass_pipelines_ready(node));
  for (uint64_t frameId = 1; frameId <= 3; ++frameId) {
    const StageData stage = build_opaque(frame, frameId);
    EXPECT_FALSE(stage.is_enabled(frame.program_index("gp.fill")));
    EXPECT_FALSE(stage.is_enabled(frame.program_index("gp.read")));
  }
  debug::publish_frame_cull(3, frame.graph->graphHash);
  debug::FrameCullSnapshot cull;
  ASSERT_TRUE(debug::latest_frame_cull(cull));
  ASSERT_LT(node.frameSlot, cull.reasons.size());
  EXPECT_EQ(cull.reasons[node.frameSlot], std::optional<std::string>{"PipelineNotReady"});
  EXPECT_EQ(passes[0].calls, 0);
  debug::testing::reset();
}

TEST_F(GraphBindPipelineTest, RejectedRequestsKeepPassesDisabled) {
  std::vector<TestPass> passes{fill_pass("Not/Valid")};
  TestPass wide{
      .name = "gp.wide",
      .order = 1,
      .declare =
          [](InternalPassBuilder& b) {
            b.create_buffer("gp.wide_buf", {.size = 256}, Access::StorageWrite);
            b.side_effect();
          },
  };
  wide.pipelines = [](const PipelineSetup& setup, std::span<uint32_t, MaxPassPipelines> ids) {
    ids[0] = shaders::request_render({.label = "gp.wide",
                                      .shader = "test/graph_fill",
                                      .layout = setup.layout,
                                      .layoutKey = setup.layoutHash,
                                      .colorCount = shaders::MaxColorTargets + 1})
                 .value;
  };
  passes.push_back(std::move(wide));
  for (uint64_t round = 1; round <= 2; ++round) {
    const TestFrame frame{compile(passes)};
    const StageData stage = build_opaque(frame, round);
    for (const std::string_view name : {"gp.fill", "gp.wide"}) {
      const uint32_t id = frame.graph->nodes[frame.node(name)].pipelineIds[0];
      ASSERT_NE(id, 0u) << name;
      EXPECT_EQ(shaders::status({id}), shaders::PipelineStatus::Failed) << name;
      EXPECT_FALSE(stage.is_enabled(frame.program_index(name))) << name;
    }
  }
  EXPECT_EQ(count_logs(LOG_WARNING, "'Not/Valid'"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "more than 8"), 1u);
  EXPECT_EQ(passes[0].calls + passes[1].calls, 0);
}

TEST_F(GraphBindPipelineTest, LatchPassesCompileSetupToHook) {
  testing::reset();
  const PassSetup* seen = nullptr;
  const Settings* settings = nullptr;
  const Caps* deviceCaps = nullptr;
  TestPass pass = fill_pass("test/graph_fill");
  const auto request = pass.pipelines;
  pass.pipelines = [&](const PipelineSetup& setup, std::span<uint32_t, MaxPassPipelines> ids) {
    seen = setup.pass;
    settings = setup.pass->settings;
    deviceCaps = setup.pass->caps;
    EXPECT_NE(setup.pass->env, nullptr);
    request(setup, ids);
  };
  ASSERT_NE(register_pass(desc_of(pass)), InvalidPassId);
  const std::shared_ptr<const FrameState> state = latch_frame();
  ASSERT_NE(seen, nullptr);
  EXPECT_EQ(settings, &latched_settings());
  EXPECT_EQ(deviceCaps, &caps());
  const NodeId node = state->graph->find_node("gp.fill");
  ASSERT_NE(node, InvalidNode);
  EXPECT_NE(state->graph->nodes[node].pipelineIds[0], 0u);
  testing::reset();
}

constexpr uint32_t CompactCapacity = 4096;

const compact::GraphJob CompactJob{
    .op = compact::Op::Compact,
    .capacity = CompactCapacity,
    .input = "t.flags",
    .output = "t.list",
    .scratch = "t.scratch",
    .keepMask = 1,
    .args = "t.args",
    .argsSpec = {.kind = compact::ArgsKind::DrawIndexed, .field = 1, .record = {3, 0, 0, 0, 0}},
};

void declare_compact(InternalPassBuilder& builder, const PassSetup& setup, const void* state) {
  compact::declare(builder, setup, *static_cast<const compact::GraphJob*>(state));
}

std::vector<TestPass> compact_passes() {
  std::vector<TestPass> passes{
      {.name = "t.flags",
       .declare =
           [](InternalPassBuilder& b) {
             b.create_buffer("t.flags", {.size = CompactCapacity * sizeof(uint32_t)}, Access::StorageWrite);
           },
       .encodes = false},
      {.name = "t.use",
       .order = 2,
       .declare =
           [](InternalPassBuilder& b) {
             b.use_buffer("t.args", Access::Indirect);
             b.use_buffer("t.list", Access::StorageRead);
             b.side_effect();
           }},
  };
  passes[1].encode = [](PassContext& ctx) {
    const wgpu::ComputePipeline pipeline = compute_pipeline(ctx.pipeline_layout(), frame_shader(R"(
@compute @workgroup_size(1)
fn main() {
  _ = aurora_frame.frame_index;
}
)"));
    const BufferView args = ctx.buffer(BufRef{0});
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.SetPipeline(pipeline);
    pass.DispatchWorkgroupsIndirect(args.buffer, args.offset);
    pass.End();
  };
  return passes;
}

std::shared_ptr<const CompiledGraph> compile_compact(const std::vector<TestPass>& passes, const compact::GraphJob& job,
                                                     const Caps& deviceCaps) {
  std::vector<PassDesc> descs{desc_of(passes[0]),
                              compact::graph_pass("t.compact", Stage::OpaqueDone, declare_compact, &job),
                              desc_of(passes[1])};
  descs[1].order = 1;
  GraphInputs inputs;
  inputs.env = gather_graph_environment();
  inputs.caps = &deviceCaps;
  inputs.passes = descs;
  const std::shared_ptr<CompiledGraph> graph = compile_graph(inputs);
  prepare_graph_bindings(*graph);
  return graph;
}

TEST_F(GraphBindPipelineTest, CompactPassEncodes) {
  compact::initialize();
  ASSERT_TRUE(compact::supported(caps()));
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
  std::vector<TestPass> passes = compact_passes();
  const TestFrame frame{compile_compact(passes, CompactJob, caps())};
  const CompiledNode& node = frame.graph->nodes[frame.node("t.compact")];
  ASSERT_EQ(node.status, CullReason::None);
  EXPECT_TRUE(node.manualBindings);
  ASSERT_NE(node.pipelineIds[0], 0u);
  EXPECT_EQ(node.pipelineIds[0], compact::pipeline().value);
  EXPECT_EQ(shaders::status({node.pipelineIds[0]}), shaders::PipelineStatus::Pending);
  const StageData pending = build_opaque(frame, 1);
  EXPECT_FALSE(pending.is_enabled(frame.program_index("t.compact")));
  EXPECT_FALSE(pending.is_enabled(frame.program_index("t.use")));
  release = true;
  gate.wait();
  ASSERT_EQ(shaders::wait({node.pipelineIds[0]}, PipelineWaitNs), shaders::PipelineStatus::Ready);
  const StageData readyStage = build_opaque(frame, 2);
  EXPECT_TRUE(readyStage.is_enabled(frame.program_index("t.compact")));
  EXPECT_TRUE(readyStage.is_enabled(frame.program_index("t.use")));

  const auto usage = [&](std::string_view name) {
    return frame.graph->resources[frame.graph->find_resource(name)].bufferUsage;
  };
  constexpr wgpu::BufferUsage StorageCopy = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  EXPECT_EQ(static_cast<wgpu::BufferUsage>(usage("t.scratch") & StorageCopy), StorageCopy);
  EXPECT_EQ(static_cast<wgpu::BufferUsage>(usage("t.args") & (StorageCopy | wgpu::BufferUsage::Indirect)),
            StorageCopy | wgpu::BufferUsage::Indirect);
  EXPECT_TRUE(usage("t.list") & wgpu::BufferUsage::Storage);

  const compact::Stats before = compact::stats();
  EXPECT_EQ(run_frame(frame, 3), std::nullopt);
  EXPECT_EQ(compact::stats().jobs - before.jobs, 1u);
  EXPECT_EQ(compact::stats().dispatches - before.dispatches, 4u);
  EXPECT_EQ(passes[1].calls, 1);
  const uint64_t created = compact::stats().bindGroupsCreated;
  EXPECT_EQ(run_frame(frame, 4), std::nullopt);
  EXPECT_EQ(compact::stats().jobs - before.jobs, 2u);
  EXPECT_EQ(compact::stats().bindGroupsCreated, created);
  EXPECT_EQ(compact::stats().rejected, before.rejected);
}

TEST_F(GraphBindPipelineTest, CompactPassSkipsInvalidPlan) {
  compact::initialize();
  Caps compatCaps = caps();
  compatCaps.core = false;
  compatCaps.profile = FEATURE_PROFILE_COMPAT;
  compatCaps.limits.maxStorageBufferBindingSize = 128ull << 20;
  ASSERT_EQ(compact::max_capacity(compatCaps), 33'554'432u);
  compact::GraphJob job = CompactJob;
  job.capacity = 33'554'433;
  std::vector<TestPass> passes = compact_passes();
  const TestFrame frame{compile_compact(passes, job, compatCaps)};
  EXPECT_EQ(frame.graph->nodes[frame.node("t.compact")].status, CullReason::Skipped);
  EXPECT_EQ(frame.graph->nodes[frame.node("t.use")].status, CullReason::MissingInput);
  const compact::Stats before = compact::stats();
  EXPECT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(compact::stats().jobs, before.jobs);
  EXPECT_EQ(compact::stats().passes, before.passes);
  EXPECT_EQ(compact::stats().dispatches, before.dispatches);
  EXPECT_EQ(passes[1].calls, 0);
}

TEST_F(GraphBindTest, BindCacheStatsFromAnotherThread) {
  const auto steady = steady_passes();
  const auto other = declaration_order_passes();
  const TestFrame a{compile(steady)};
  const TestFrame b{compile(other)};
  std::atomic_bool done = false;
  std::atomic_uint64_t reads = 0;
  std::thread reader([&] {
    do {
      const BindCacheStats stats = bind_cache_stats();
      reads.fetch_add(1 + (stats.created > 1000000 ? stats.entries : 0), std::memory_order_relaxed);
    } while (!done.load(std::memory_order_relaxed));
  });
  while (reads.load(std::memory_order_relaxed) == 0) {
    std::this_thread::yield();
  }
  for (uint64_t frameId = 1; frameId <= 40; ++frameId) {
    if (frameId % 5 == 0) {
      force_memory_pressure();
    }
    EXPECT_EQ(run_frame(frameId % 2 == 0 ? a : b, frameId), std::nullopt);
  }
  done.store(true, std::memory_order_relaxed);
  reader.join();
  EXPECT_GT(reads.load(), 0u);
  EXPECT_EQ(bind_cache_stats().staleEntries, 0u);
}

std::vector<std::string_view> recorded_names(webgpu::gpu_prof::detail::EventKind kind) {
  std::vector<std::string_view> names;
  for (const auto& event : webgpu::gpu_prof::testing::recording_events()) {
    if (event.kind == kind) {
      names.emplace_back(event.name);
    }
  }
  return names;
}

TEST_F(GraphBindTest, TimingZonesWrapCopyAndEncoderPasses) {
  namespace gpu_prof = webgpu::gpu_prof;
  using gpu_prof::detail::EventKind;
  gpu_prof::shutdown();
  gpu_prof::initialize();
  if (!gpu_prof::supported()) {
    GTEST_SKIP() << "the device has no timestamp queries";
  }
  gpu_prof::request(2);
  const auto sideEffect = [](InternalPassBuilder& b) { b.side_effect(); };
  const auto openCompute = [](PassContext& ctx) {
    if (const wgpu::ComputePassEncoder pass = ctx.begin_compute()) {
      pass.End();
    }
  };
  const std::vector<TestPass> passes{
      {.name = "t.compute", .order = 0, .declare = sideEffect, .encode = openCompute},
      {.name = "t.copy", .kind = PassKind::Copy, .order = 1, .declare = sideEffect},
      {.name = "t.encoder", .kind = PassKind::Encoder, .order = 2, .declare = sideEffect, .encode = openCompute},
      {.name = "t.render",
       .kind = PassKind::Render,
       .order = 3,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("t.target", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           },
       .encode =
           [](PassContext& ctx) {
             if (const wgpu::RenderPassEncoder pass = ctx.begin_render()) {
               pass.End();
             }
           }},
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(frame.graph->framePasses.size(), passes.size());
  on_worker_frame_begin(1, &frame.state);
  StageData stage = stage_for(frame, Stage::OpaqueDone, 1);
  build_stage_bindings(stage, &fake_sink);
  upload_uniforms();
  push_validation_scope();
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  gpu_prof::frame_begin(encoder);
  run_stage(encoder, stage);
  const std::vector<std::string_view> zones{"t.copy", "t.encoder"};
  const std::vector<std::string_view> passTimestamps{"t.compute", "t.encoder", "t.render"};
  EXPECT_EQ(recorded_names(EventKind::ZoneBegin), zones);
  EXPECT_EQ(recorded_names(EventKind::PassBegin), passTimestamps);
  run_stage({}, stage);
  EXPECT_EQ(recorded_names(EventKind::ZoneBegin), zones);
  EXPECT_EQ(recorded_names(EventKind::PassBegin), passTimestamps);
  gpu_prof::frame_end(encoder);
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);
  gpu_prof::after_submit();
  EXPECT_EQ(pop_validation_scope(), std::nullopt);
  EXPECT_TRUE(gpu_prof::wait_idle(5'000'000'000));
  gpu_prof::shutdown();
  for (const TestPass& pass : passes) {
    EXPECT_EQ(pass.calls, 2) << pass.name;
  }
}

constexpr std::string_view WriteWordsWgsl = R"(
@group(1) @binding(0) var<storage, read_write> first_words: array<u32>;
@compute @workgroup_size(1)
fn main() {
  first_words[0] = aurora_frame.frame_index;
}
)";

constexpr std::string_view WriteTwoWgsl = R"(
@group(1) @binding(0) var<storage, read_write> first_words: array<u32>;
@group(1) @binding(1) var<storage, read_write> second_words: array<u32>;
@compute @workgroup_size(1)
fn main() {
  first_words[0] = first_words[0] + 1u;
  second_words[0] = aurora_frame.frame_index;
}
)";

constexpr std::string_view WhiteFragmentWgsl = R"(
@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0);
}
)";

TestPass import_pass(std::string name, PassKind kind, std::vector<std::pair<std::string, Access>> imports,
                     std::vector<PersistentHandle> handles, uint16_t order = 0) {
  return {
      .name = std::move(name),
      .kind = kind,
      .order = order,
      .declare =
          [imports = std::move(imports), handles = std::move(handles)](InternalPassBuilder& b) {
            for (size_t i = 0; i < imports.size(); ++i) {
              b.import_persistent(imports[i].first, handles[i], imports[i].second);
            }
            b.side_effect();
          },
  };
}

TEST_F(GraphBindTest, HeapBlockBindsBlockRange) {
  const HeapBlock written = create_heap_block(gpu_heap::Region::DrawArgs, 256, "hb.written");
  const HeapBlock args = create_heap_block(gpu_heap::Region::DrawArgs, 512, "hb.args");
  const HeapBlock indices = create_heap_block(gpu_heap::Region::Index32, 1 * KiB, "hb.indices");
  ASSERT_TRUE(written);
  ASSERT_TRUE(args);
  ASSERT_TRUE(indices);
  std::vector<TestPass> passes{
      import_pass("hb.write", PassKind::Compute, {{"hb.written", Access::StorageWrite}}, {written.handle}),
      {.name = "hb.draw",
       .kind = PassKind::Render,
       .order = 1,
       .declare =
           [&](InternalPassBuilder& b) {
             b.create("hb.color", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.import_persistent("hb.args", args.handle, Access::Indirect);
             b.import_persistent("hb.indices", indices.handle, Access::Index);
             b.side_effect();
           }},
  };
  std::vector<BufferView> bound;
  passes[0].encode = [&](PassContext& ctx) { bound.push_back(ctx.buffer(BufRef{0})); };
  passes[1].encode = [&](PassContext& ctx) {
    bound.push_back(ctx.buffer(BufRef{1}));
    bound.push_back(ctx.buffer(BufRef{2}));
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(bound.size(), 3u);
  const std::array blocks{written.block, args.block, indices.block};
  for (size_t i = 0; i < blocks.size(); ++i) {
    const gpu_heap::BlockView view = gpu_heap::block_view(blocks[i]);
    ASSERT_TRUE(view) << i;
    EXPECT_EQ(bound[i].buffer.Get(), view.buffer.Get()) << i;
    EXPECT_EQ(bound[i].offset, view.offset) << i;
    EXPECT_EQ(bound[i].size, view.size) << i;
  }
  EXPECT_EQ(bound[0].buffer.Get(), bound[1].buffer.Get());
  EXPECT_NE(bound[0].offset, bound[1].offset);
  const auto identities = testing::last_identities(frame.node("hb.write"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].kind, IdentityKind::Persistent);
  EXPECT_EQ(identities[0].value, uint64_t{written.handle.id} << 32 | 1);
  EXPECT_TRUE(testing::last_identities(frame.node("hb.draw")).empty());
}

TEST_F(GraphBindTest, HeapBlockSkipsPassUntilMaterialized) {
  gpu_heap::set_recording_frame(1);
  const HeapBlock block = create_heap_block(gpu_heap::Region::DrawArgs, 256, "hs.block");
  ASSERT_TRUE(block);
  std::vector<TestPass> passes{
      import_pass("hs.write", PassKind::Compute, {{"hs.block", Access::StorageWrite}}, {block.handle}),
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 0);
  EXPECT_EQ(gpu_heap::block_state(block.block), gpu_heap::BlockState::Pending);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'hs.block'"), 1u);

  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  const auto identities = testing::last_identities(frame.node("hs.write"));
  ASSERT_EQ(identities.size(), 1u);
  EXPECT_EQ(identities[0].value, uint64_t{block.handle.id} << 32 | 1);
  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  EXPECT_EQ(passes[0].calls, 2);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'hs.block'"), 1u);
}

TEST_F(GraphBindTest, HeapBlockGrowthRebinds) {
  const HeapBlock block = create_heap_block(gpu_heap::Region::DrawArgs, 256, "hg.block");
  std::vector<TestPass> passes{
      import_pass("hg.read", PassKind::Compute, {{"hg.block", Access::StorageRead}}, {block.handle}),
  };
  BufferView bound;
  passes[0].encode = [&](PassContext& ctx) { bound = ctx.buffer(BufRef{0}); };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(bind_cache_stats().created, 1u);
  const gpu_heap::BlockView before = gpu_heap::block_view(block.block);
  EXPECT_EQ(bound.buffer.Get(), before.buffer.Get());

  ASSERT_NE(gpu_heap::create_block(gpu_heap::Region::DrawArgs, 64 * KiB), gpu_heap::BlockHandle::Invalid);
  on_worker_frame_begin(2, &frame.state);
  const gpu_heap::BlockView after = gpu_heap::block_view(block.block);
  ASSERT_EQ(after.generation, 2u);
  ASSERT_NE(after.buffer.Get(), before.buffer.Get());
  BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.evicted, 1u);
  EXPECT_EQ(stats.staleEntries, 0u);

  StageData stage = stage_for(frame, Stage::OpaqueDone, 2);
  ASSERT_EQ(execute(stage), std::nullopt);
  stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.evicted, 1u);
  EXPECT_EQ(bound.buffer.Get(), after.buffer.Get());
  EXPECT_EQ(bound.offset, before.offset);
  EXPECT_EQ(testing::last_identities(frame.node("hg.read")).at(0).value, uint64_t{block.handle.id} << 32 | 2);

  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  stats = bind_cache_stats();
  EXPECT_EQ(stats.created, 2u);
  EXPECT_EQ(stats.staleEntries, 0u);
}

TEST_F(GraphBindTest, DestroyedHeapBlockBindsUntilBoundary) {
  HeapBlock block = create_heap_block(gpu_heap::Region::DrawArgs, 256, "hd.block");
  const PersistentHandle handle = block.handle;
  std::vector<TestPass> passes{
      import_pass("hd.read", PassKind::Compute, {{"hd.block", Access::StorageRead}}, {handle}),
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);

  on_worker_frame_begin(2, &frame.state);
  destroy_heap_block(block);
  EXPECT_FALSE(block);
  StageData stage = stage_for(frame, Stage::OpaqueDone, 2);
  ASSERT_EQ(execute(stage), std::nullopt);
  EXPECT_EQ(passes[0].calls, 2);
  EXPECT_EQ(bind_cache_stats().entries, 1u);

  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  EXPECT_EQ(passes[0].calls, 2);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'hd.block'"), 1u);
  const BindCacheStats stats = bind_cache_stats();
  EXPECT_EQ(stats.entries, 0u);
  EXPECT_EQ(stats.evicted, 1u);
  PersistentBuffer resolved;
  EXPECT_FALSE(resolve_persistent(handle, resolved));
}

TEST_F(GraphBindTest, ImportLackingUsageSkips) {
  const HeapBlock args = create_heap_block(gpu_heap::Region::DrawArgs, 256, "lu.args");
  const HeapBlock indices = create_heap_block(gpu_heap::Region::Index32, 256, "lu.indices");
  std::vector<TestPass> passes{
      import_pass("lu.vertex", PassKind::Encoder, {{"lu.args", Access::Vertex}}, {args.handle}, 0),
      import_pass("lu.uniform", PassKind::Encoder, {{"lu.args", Access::Uniform}}, {args.handle}, 1),
      import_pass("lu.indirect", PassKind::Compute, {{"lu.indices", Access::Indirect}}, {indices.handle}, 2),
  };
  const TestFrame frame{compile(passes)};
  for (const TestPass& pass : passes) {
    ASSERT_EQ(frame.graph->nodes[frame.node(pass.name)].status, CullReason::None) << pass.name;
  }
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  for (const TestPass& pass : passes) {
    EXPECT_EQ(pass.calls, 0) << pass.name;
  }
  EXPECT_EQ(count_logs(LOG_WARNING, "lacks that usage"), 3u);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'lu.args' as Vertex"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'lu.args' as Uniform"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "imports 'lu.indices' as Indirect"), 1u);
}

TEST_F(GraphBindTest, ReadAndWriteOfOneRegionSkips) {
  std::vector<HeapBlock> blocks;
  for (const char* name : {"rw.out", "rw.args", "rw.out2", "rw.in"}) {
    blocks.push_back(create_heap_block(gpu_heap::Region::DrawArgs, 256, name));
  }
  std::vector<TestPass> passes{
      import_pass("rw.indirect", PassKind::Compute, {{"rw.out", Access::StorageWrite}, {"rw.args", Access::Indirect}},
                  {blocks[0].handle, blocks[1].handle}, 0),
      import_pass("rw.read", PassKind::Compute, {{"rw.out2", Access::StorageWrite}, {"rw.in", Access::StorageRead}},
                  {blocks[2].handle, blocks[3].handle}, 1),
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(passes[0].calls, 0);
  EXPECT_EQ(passes[1].calls, 0);
  EXPECT_EQ(count_logs(LOG_WARNING, "binds one buffer writable"), 2u);
  EXPECT_EQ(count_logs(LOG_WARNING, "writable through 'rw.out' and read-only through 'rw.args'"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "writable through 'rw.out2' and read-only through 'rw.in'"), 1u);
}

TEST_F(GraphBindTest, WritableBlocksOfOneRegionBind) {
  const HeapBlock first = create_heap_block(gpu_heap::Region::DrawArgs, 256, "wb.first");
  const HeapBlock second = create_heap_block(gpu_heap::Region::DrawArgs, 256, "wb.second");
  std::vector<TestPass> passes{
      import_pass("wb.pass", PassKind::Compute, {{"wb.first", Access::StorageRW}, {"wb.second", Access::StorageWrite}},
                  {first.handle, second.handle}),
  };
  const std::string source = frame_shader(WriteTwoWgsl);
  passes[0].encode = [&](PassContext& ctx) {
    const wgpu::ComputePipeline pipeline = compute_pipeline(ctx.pipeline_layout(), source);
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.SetPipeline(pipeline);
    pass.DispatchWorkgroups(1);
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(count_logs(LOG_WARNING, "binds one buffer writable"), 0u);

  Caps d3d11 = caps();
  d3d11.backend = BACKEND_D3D11;
  webgpu::set_caps(std::move(d3d11));
  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_EQ(count_logs(LOG_WARNING, "single-UAV limit"), 1u);
}

TEST_F(GraphBindTest, DifferentRegionsBind) {
  const HeapBlock out = create_heap_block(gpu_heap::Region::DrawArgs, 256, "dr.out");
  const HeapBlock args = create_heap_block(gpu_heap::Region::DispatchArgs, 256, "dr.args");
  std::vector<TestPass> passes{
      import_pass("dr.pass", PassKind::Compute, {{"dr.out", Access::StorageWrite}, {"dr.args", Access::Indirect}},
                  {out.handle, args.handle}),
  };
  const std::string source = frame_shader(WriteWordsWgsl);
  bool dispatched = false;
  passes[0].encode = [&](PassContext& ctx) {
    const wgpu::ComputePipeline pipeline = compute_pipeline(ctx.pipeline_layout(), source);
    const wgpu::ComputePassEncoder pass = ctx.begin_compute();
    pass.SetPipeline(pipeline);
    dispatched = indirect::dispatch(pass, ctx.buffer(BufRef{1}), 0);
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_TRUE(dispatched);
  EXPECT_EQ(count_logs(LOG_WARNING, "binds one buffer writable"), 0u);
}

TEST_F(GraphBindTest, OverlappingWritableImportsSkip) {
  const HeapBlock block = create_heap_block(gpu_heap::Region::DrawArgs, 256, "ov.a");
  std::vector<TestPass> passes{
      import_pass("ov.pass", PassKind::Compute, {{"ov.a", Access::StorageWrite}, {"ov.b", Access::StorageWrite}},
                  {block.handle, block.handle}),
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(frame.graph->nodes[frame.node("ov.pass")].status, CullReason::None);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 0);
  EXPECT_EQ(count_logs(LOG_WARNING, "binds one buffer writable"), 1u);
}

TEST_F(GraphBindTest, ManualComputePassIsExempt) {
  const HeapBlock out = create_heap_block(gpu_heap::Region::DrawArgs, 256, "mx.out");
  const HeapBlock args = create_heap_block(gpu_heap::Region::DrawArgs, 256, "mx.args");
  std::vector<TestPass> passes{
      {.name = "mx.pass",
       .declare =
           [&](InternalPassBuilder& b) {
             b.import_persistent("mx.out", out.handle, Access::StorageWrite);
             b.import_persistent("mx.args", args.handle, Access::Indirect);
             b.manual_bindings();
             b.side_effect();
           }},
  };
  const std::string writeSource = frame_shader(WriteWordsWgsl);
  const std::string frameSource = frame_shader(R"(
@compute @workgroup_size(1)
fn main() {
  _ = aurora_frame.frame_index;
}
)");
  bool dispatched = false;
  passes[0].encode = [&](PassContext& ctx) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Storage},
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.entryCount = 1, .entries = &entry};
    const wgpu::BindGroupLayout own = ctx.device().CreateBindGroupLayout(&layoutDescriptor);
    const std::array layouts{ctx.frame_layout(), own};
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = layouts.size(),
                                                                  .bindGroupLayouts = layouts.data()};
    const wgpu::ComputePipeline write =
        compute_pipeline(ctx.device().CreatePipelineLayout(&pipelineLayoutDescriptor), writeSource);
    const BufferView target = ctx.buffer(BufRef{0});
    const wgpu::BindGroupEntry binding{
        .binding = 0, .buffer = target.buffer, .offset = target.offset, .size = target.size};
    const wgpu::BindGroupDescriptor groupDescriptor{.layout = own, .entryCount = 1, .entries = &binding};
    const wgpu::ComputePassEncoder writePass = ctx.begin_compute();
    writePass.SetPipeline(write);
    writePass.SetBindGroup(1, ctx.device().CreateBindGroup(&groupDescriptor));
    writePass.DispatchWorkgroups(1);
    writePass.End();
    const wgpu::ComputePassEncoder indirectPass = ctx.begin_compute();
    indirectPass.SetPipeline(compute_pipeline(ctx.pipeline_layout(), frameSource));
    dispatched = indirect::dispatch(indirectPass, ctx.buffer(BufRef{1}), 0);
    indirectPass.End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 1);
  EXPECT_TRUE(dispatched);
  EXPECT_EQ(count_logs(LOG_WARNING, "binds one buffer writable"), 0u);
}

TEST_F(GraphBindTest, EncoderPassChecksIndirectAgainstGroup) {
  const HeapBlock out = create_heap_block(gpu_heap::Region::DrawArgs, 256, "ex.out");
  const HeapBlock args = create_heap_block(gpu_heap::Region::DrawArgs, 256, "ex.args");
  std::vector<TestPass> passes{
      import_pass("ex.pass", PassKind::Encoder, {{"ex.out", Access::StorageWrite}, {"ex.args", Access::Indirect}},
                  {out.handle, args.handle}),
  };
  const TestFrame frame{compile(passes)};
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(passes[0].calls, 0);
  EXPECT_EQ(count_logs(LOG_WARNING, "writable through 'ex.out' and read-only through 'ex.args'"), 1u);
}

TEST_F(GraphBindTest, IndirectDummyIsZeroAndValid) {
  std::vector<TestPass> passes{
      {.name = "id.draw",
       .kind = PassKind::Render,
       .declare =
           [](InternalPassBuilder& b) {
             b.create("id.color", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.use_buffer("id.args", Access::Indirect, UseOptions{.missing = Missing::BindDummyBlack, .dummySize = 16});
             b.side_effect();
           }},
  };
  const std::string source = frame_shader(std::string{WhiteFragmentWgsl} + std::string{FullscreenVertexWgsl});
  BufferView args;
  bool drew = false;
  uint32_t indexedSlots = UINT32_MAX;
  uint64_t rejected = 0;
  passes[0].encode = [&](PassContext& ctx) {
    args = ctx.buffer(BufRef{1});
    const wgpu::RenderPipeline pipeline = render_pipeline(ctx.pipeline_layout(), source, TextureFormat::RGBA8Unorm);
    const wgpu::RenderPassEncoder pass = ctx.begin_render();
    pass.SetPipeline(pipeline);
    drew = indirect::draw(pass, args, 0);
    const uint64_t before = indirect::encode_stats().rejected;
    indexedSlots = indirect::draw_indexed_slots(pass, args, 0, 1);
    rejected = indirect::encode_stats().rejected - before;
    pass.End();
  };
  const TestFrame frame{compile(passes)};
  ASSERT_TRUE(frame.use("id.draw", 1).staticDummy);
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_TRUE(args.buffer);
  const auto draws = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Index | wgpu::BufferUsage::Indirect;
  EXPECT_EQ(static_cast<wgpu::BufferUsage>(args.buffer.GetUsage() & draws), static_cast<wgpu::BufferUsage>(draws));
  EXPECT_EQ(args.size, 16u);
  EXPECT_TRUE(drew);
  EXPECT_EQ(indexedSlots, 0u);
  EXPECT_EQ(rejected, 1u);
}

wgpu::Buffer test_buffer(const test::TestDevice& device, wgpu::BufferUsage usage, uint64_t size) {
  const wgpu::BufferDescriptor descriptor{.usage = usage, .size = size};
  return device.device.CreateBuffer(&descriptor);
}

std::optional<std::string> encode_draws(test::TestDevice& device,
                                        const std::function<void(const wgpu::RenderPassEncoder&)>& encode) {
  const wgpu::TextureDescriptor targetDescriptor{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .size = {4, 4, 1},
      .format = TextureFormat::RGBA8Unorm,
  };
  const wgpu::Texture target = device.device.CreateTexture(&targetDescriptor);
  const std::string source = std::string{WhiteFragmentWgsl} + std::string{FullscreenVertexWgsl};
  const wgpu::RenderPipeline pipeline = render_pipeline({}, source, TextureFormat::RGBA8Unorm);
  device.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  const wgpu::CommandEncoder encoder = device.device.CreateCommandEncoder();
  const wgpu::RenderPassColorAttachment color{
      .view = target.CreateView(),
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
  };
  const wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1, .colorAttachments = &color};
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(pipeline);
  encode(pass);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  device.queue.Submit(1, &commands);
  return test::pop_validation_error(device);
}

TEST_F(GraphBindTest, IndirectHelpersRejectBadRanges) {
  const wgpu::Buffer args = test_buffer(*m_device, wgpu::BufferUsage::Indirect | wgpu::BufferUsage::CopyDst, 80);
  const wgpu::Buffer plain = test_buffer(*m_device, wgpu::BufferUsage::Storage, 80);
  const wgpu::Buffer indices = test_buffer(*m_device, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, 64);
  const BufferView view{.buffer = args, .size = 80};
  EXPECT_EQ(indirect::capacity(view, indirect::Record::Draw), 5u);
  EXPECT_EQ(indirect::capacity(view, indirect::Record::DrawIndexed), 4u);
  EXPECT_EQ(indirect::capacity(view, indirect::Record::Dispatch), 6u);
  const indirect::EncodeStats before = indirect::encode_stats();
  const auto error = encode_draws(*m_device, [&](const wgpu::RenderPassEncoder& pass) {
    EXPECT_TRUE(indirect::draw(pass, view, 4));
    EXPECT_FALSE(indirect::draw(pass, view, 5));
    EXPECT_FALSE(indirect::draw(pass, BufferView{.buffer = args, .offset = 2, .size = 64}, 0));
    EXPECT_FALSE(indirect::draw(pass, BufferView{.buffer = plain, .size = 80}, 0));
    EXPECT_FALSE(indirect::draw(pass, BufferView{}, 0));
    EXPECT_EQ(indirect::draw_slots(pass, view, 0, 0), 0u);
    EXPECT_EQ(indirect::draw_slots(pass, view, 3, 3), 0u);
    EXPECT_TRUE(indirect::bind_indices(pass, BufferView{.buffer = indices, .size = 64}));
    EXPECT_FALSE(indirect::bind_indices(pass, BufferView{.buffer = indices, .size = 6}));
    EXPECT_FALSE(indirect::bind_indices(pass, BufferView{.buffer = args, .size = 64}));
    EXPECT_TRUE(indirect::draw_indexed(pass, view, 3));
    EXPECT_FALSE(indirect::draw_indexed(pass, view, 4));
  });
  EXPECT_EQ(error, std::nullopt);

  push_validation_scope();
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(compute_pipeline({}, "@compute @workgroup_size(1)\nfn main() {}\n"));
  EXPECT_TRUE(indirect::dispatch(pass, view, 5));
  EXPECT_FALSE(indirect::dispatch(pass, view, 6));
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);
  EXPECT_EQ(pop_validation_scope(), std::nullopt);

  const indirect::EncodeStats after = indirect::encode_stats();
  EXPECT_EQ(after.rejected - before.rejected, 10u);
  EXPECT_EQ(after.encoded - before.encoded, 3u);
}

TEST_F(GraphBindTest, DrawSlotsUseMultiDrawWhenAvailable) {
  if (!caps().multiDrawIndirect) {
    GTEST_SKIP() << "the device has no multi-draw indirect";
  }
  const wgpu::Buffer args = test_buffer(*m_device, wgpu::BufferUsage::Indirect, 8 * sizeof(indirect::DrawIndexedArgs));
  const wgpu::Buffer indices = test_buffer(*m_device, wgpu::BufferUsage::Index, 64);
  const BufferView view{.buffer = args, .size = args.GetSize()};
  const indirect::EncodeStats before = indirect::encode_stats();
  uint32_t draws = 0;
  uint32_t indexed = 0;
  const auto error = encode_draws(*m_device, [&](const wgpu::RenderPassEncoder& pass) {
    draws = indirect::draw_slots(pass, view, 0, 8);
    ASSERT_TRUE(indirect::bind_indices(pass, BufferView{.buffer = indices, .size = 64}));
    indexed = indirect::draw_indexed_slots(pass, view, 0, 8);
  });
  EXPECT_EQ(error, std::nullopt);
  EXPECT_EQ(draws, 1u);
  EXPECT_EQ(indexed, 1u);
  const indirect::EncodeStats after = indirect::encode_stats();
  EXPECT_EQ(after.multiDraws - before.multiDraws, 2u);
  EXPECT_EQ(after.loopDraws - before.loopDraws, 0u);
  EXPECT_EQ(after.rejected - before.rejected, 0u);
}

TEST_F(GraphBindTest, DrawSlotsFallBackWithoutMdi) {
  Caps loop = caps();
  loop.multiDrawIndirect = false;
  webgpu::set_caps(std::move(loop));
  const wgpu::Buffer args = test_buffer(*m_device, wgpu::BufferUsage::Indirect, 8 * sizeof(indirect::DrawIndexedArgs));
  const wgpu::Buffer indices = test_buffer(*m_device, wgpu::BufferUsage::Index, 64);
  const BufferView view{.buffer = args, .size = args.GetSize()};
  const indirect::EncodeStats before = indirect::encode_stats();
  uint32_t draws = 0;
  uint32_t indexed = 0;
  const auto error = encode_draws(*m_device, [&](const wgpu::RenderPassEncoder& pass) {
    draws = indirect::draw_slots(pass, view, 0, 8);
    ASSERT_TRUE(indirect::bind_indices(pass, BufferView{.buffer = indices, .size = 64}));
    indexed = indirect::draw_indexed_slots(pass, view, 0, 8);
  });
  EXPECT_EQ(error, std::nullopt);
  EXPECT_EQ(draws, 8u);
  EXPECT_EQ(indexed, 8u);
  const indirect::EncodeStats after = indirect::encode_stats();
  EXPECT_EQ(after.loopDraws - before.loopDraws, 16u);
  EXPECT_EQ(after.multiDraws - before.multiDraws, 0u);
  EXPECT_EQ(after.encoded - before.encoded, 16u);
}

template <typename Record>
std::vector<uint32_t> record_words(const Record& record) {
  std::vector<uint32_t> words(sizeof(Record) / sizeof(uint32_t));
  std::memcpy(words.data(), &record, sizeof(Record));
  return words;
}

TEST(IndirectRecords, CpuMakersFollowRules) {
  static_assert(offsetof(indirect::DrawIndexedArgs, baseVertex) == 12 &&
                offsetof(indirect::DrawIndexedArgs, firstInstance) == 16);
  static_assert(offsetof(indirect::DrawArgs, firstInstance) == 12 && offsetof(indirect::DispatchArgs, z) == 8);
  EXPECT_EQ(indirect::record_size(indirect::Record::Draw), 16u);
  EXPECT_EQ(indirect::record_size(indirect::Record::DrawIndexed), 20u);
  EXPECT_EQ(indirect::record_size(indirect::Record::Dispatch), 12u);
  using Words = std::vector<uint32_t>;
  const Words zeroDraw(4, 0);
  const Words zeroIndexed(5, 0);
  const Words zeroDispatch(3, 0);

  EXPECT_EQ(record_words(indirect::make_draw(3, 2, 5, 8, 2)), (Words{3, 2, 5, 0}));
  EXPECT_EQ(record_words(indirect::make_draw(8, 1, 0, 8, 1)), (Words{8, 1, 0, 0}));
  EXPECT_EQ(record_words(indirect::make_draw(4, 1, 5, 8, 1)), zeroDraw);
  EXPECT_EQ(record_words(indirect::make_draw(0, 1, 9, 8, 1)), zeroDraw);
  EXPECT_EQ(record_words(indirect::make_draw(1, 3, 0, 8, 2)), zeroDraw);
  EXPECT_EQ(record_words(indirect::make_draw(1, 1, UINT32_MAX, 512, 1)), zeroDraw);
  EXPECT_EQ(record_words(indirect::make_draw(2, 1, UINT32_MAX - 1, indirect::Unbounded, 1)), zeroDraw);
  EXPECT_EQ(record_words(indirect::make_draw(1, 1, UINT32_MAX - 1, indirect::Unbounded, 1)),
            (Words{1, 1, UINT32_MAX - 1, 0}));
  EXPECT_EQ(record_words(indirect::make_draw(1, UINT32_MAX, 0, 8, indirect::Unbounded)), (Words{1, UINT32_MAX, 0, 0}));

  EXPECT_EQ(record_words(indirect::make_draw_indexed(6, 1, 2, -4, 8, 1)),
            (Words{6, 1, 2, static_cast<uint32_t>(-4), 0}));
  EXPECT_EQ(record_words(indirect::make_draw_indexed(0, 1, 9, 0, 8, 1)), zeroIndexed);
  EXPECT_EQ(record_words(indirect::make_draw_indexed(7, 1, 2, 0, 8, 1)), zeroIndexed);
  EXPECT_EQ(record_words(indirect::make_draw_indexed(6, 2, 2, 0, 8, 1)), zeroIndexed);
  EXPECT_EQ(record_words(indirect::make_draw_indexed(UINT32_MAX, 1, 0, 0, indirect::Unbounded, 1)),
            (Words{UINT32_MAX, 1, 0, 0, 0}));
  EXPECT_EQ(record_words(indirect::make_draw_indexed(1, 1, UINT32_MAX, 0, indirect::Unbounded, 1)), zeroIndexed);

  EXPECT_EQ(record_words(indirect::make_dispatch(65535, 1, 7)), (Words{65535, 1, 7}));
  EXPECT_EQ(record_words(indirect::make_dispatch(65536, 1, 1)), zeroDispatch);
  EXPECT_EQ(record_words(indirect::make_dispatch(1, 65536, 1)), zeroDispatch);
  EXPECT_EQ(record_words(indirect::make_dispatch(1, 1, UINT32_MAX)), zeroDispatch);

  const BufferView view{.size = 100};
  EXPECT_EQ(indirect::capacity(view, indirect::Record::Draw), 6u);
  EXPECT_EQ(indirect::capacity(view, indirect::Record::DrawIndexed), 5u);
  EXPECT_EQ(indirect::capacity(view, indirect::Record::Dispatch), 8u);
  const gpu_heap::BlockView block{.offset = 512, .size = 256, .generation = 3};
  const BufferView fromBlock = indirect::buffer_view(block);
  EXPECT_EQ(fromBlock.offset, 512u);
  EXPECT_EQ(fromBlock.size, 256u);
}

class IndirectGpu : public ::testing::Test {
protected:
  void SetUp() override {
    m_options.backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no adapter: " << error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(m_device->adapterInfo.backendType));
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
  }

  void TearDown() override {
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

// Triangle k covers only texel k of a 16-texel row, so each record's vertex base picks the texel it lights
constexpr std::string_view SlotVertexWgsl = R"(
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let left = -1.0 + f32(index / 3u) * 0.125;
  let corner = index % 3u;
  if (corner == 2u) {
    return vec4<f32>(left + 0.125, 0.0, 0.0, 1.0);
  }
  return vec4<f32>(left, select(-1.0, 1.0, corner == 1u), 0.0, 1.0);
}
)";

TEST_F(IndirectGpu, MultiDrawMatchesLoop) {
  constexpr uint32_t Texels = 16;
  const wgpu::Device& device = m_device->device;
  const auto make_buffer = [&](wgpu::BufferUsage usage, const void* data, uint64_t size) {
    const wgpu::BufferDescriptor descriptor{.usage = usage | wgpu::BufferUsage::CopyDst, .size = size};
    const wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
    m_device->queue.WriteBuffer(buffer, 0, data, size);
    return buffer;
  };
  std::array<indirect::DrawArgs, 8> draws{};
  std::array<indirect::DrawIndexedArgs, 8> indexedDraws{};
  constexpr std::array<uint32_t, 3> DrawSlots{1, 4, 6};
  constexpr std::array<uint32_t, 3> IndexedSlots{0, 3, 7};
  for (const uint32_t slot : DrawSlots) {
    draws[slot] = indirect::make_draw(3, 1, 3 * slot, 3 * 8, 1);
  }
  constexpr uint32_t IndexCount = 3 * 8;
  std::array<uint32_t, IndexCount> indexList{};
  for (uint32_t i = 0; i < IndexCount; ++i) {
    indexList[i] = i;
  }
  for (const uint32_t slot : IndexedSlots) {
    indexedDraws[slot] =
        indirect::make_draw_indexed(3, 1, 3 * slot, static_cast<int32_t>(3 * (slot + 1)), IndexCount, 1);
  }
  const wgpu::Buffer drawArgs = make_buffer(wgpu::BufferUsage::Indirect, draws.data(), sizeof(draws));
  const wgpu::Buffer indexedArgs = make_buffer(wgpu::BufferUsage::Indirect, indexedDraws.data(), sizeof(indexedDraws));
  const wgpu::Buffer indices = make_buffer(wgpu::BufferUsage::Index, indexList.data(), sizeof(indexList));

  const std::string code = std::string{WhiteFragmentWgsl} + std::string{SlotVertexWgsl};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
  const wgpu::ShaderModule module = device.CreateShaderModule(&moduleDescriptor);
  const wgpu::BlendComponent add{
      .operation = wgpu::BlendOperation::Add,
      .srcFactor = wgpu::BlendFactor::One,
      .dstFactor = wgpu::BlendFactor::One,
  };
  const wgpu::BlendState blend{.color = add, .alpha = add};
  const wgpu::ColorTargetState target{.format = TextureFormat::RGBA16Float, .blend = &blend};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .vertex = {.module = module, .entryPoint = "vs"},
      .fragment = &fragment,
  };
  const wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDescriptor);

  const auto render = [&](bool indexed, uint32_t& encoded) -> std::optional<std::vector<uint32_t>> {
    const wgpu::TextureDescriptor textureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {Texels, 1, 1},
        .format = TextureFormat::RGBA16Float,
    };
    const wgpu::Texture texture = device.CreateTexture(&textureDescriptor);
    const wgpu::BufferDescriptor readDescriptor{.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                                                .size = 256};
    const wgpu::Buffer texels = device.CreateBuffer(&readDescriptor);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    const wgpu::RenderPassColorAttachment color{
        .view = texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0, 0, 0, 0},
    };
    const wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1, .colorAttachments = &color};
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    if (indexed) {
      indirect::bind_indices(pass, BufferView{.buffer = indices, .size = sizeof(indexList)});
      encoded =
          indirect::draw_indexed_slots(pass, BufferView{.buffer = indexedArgs, .size = sizeof(indexedDraws)}, 0, 8);
    } else {
      encoded = indirect::draw_slots(pass, BufferView{.buffer = drawArgs, .size = sizeof(draws)}, 0, 8);
    }
    pass.End();
    const wgpu::TexelCopyTextureInfo source{.texture = texture};
    const wgpu::TexelCopyBufferInfo destination{.layout = {.bytesPerRow = 256, .rowsPerImage = 1}, .buffer = texels};
    const wgpu::Extent3D extent{Texels, 1, 1};
    encoder.CopyTextureToBuffer(&source, &destination, &extent);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    return test::read_buffer(*m_device, texels, 0, Texels * 8);
  };
  const auto image_of = [&](std::span<const uint32_t> lit) {
    std::vector<uint32_t> words(Texels * 2, 0);
    for (const uint32_t texel : lit) {
      words[texel * 2] = 0x3C003C00u;
      words[texel * 2 + 1] = 0x3C003C00u;
    }
    return words;
  };
  const std::array<std::vector<uint32_t>, 2> expected{image_of(std::array<uint32_t, 3>{1, 4, 6}),
                                                      image_of(std::array<uint32_t, 3>{1, 7, 15})};

  std::vector<bool> modes{false};
  if (caps().multiDrawIndirect) {
    modes.push_back(true);
  }
  ::testing::Test::RecordProperty("multiDrawIndirect", caps().multiDrawIndirect ? 1 : 0);
  std::array<std::vector<uint32_t>, 2> loopImages;
  for (const bool multiDraw : modes) {
    Caps mode = caps();
    mode.multiDrawIndirect = multiDraw;
    webgpu::set_caps(std::move(mode));
    for (const bool indexed : {false, true}) {
      SCOPED_TRACE(std::string{multiDraw ? "multi-draw" : "loop"} + (indexed ? ", indexed" : ""));
      uint32_t encoded = 0;
      const auto image = render(indexed, encoded);
      ASSERT_TRUE(image);
      EXPECT_EQ(encoded, multiDraw ? 1u : 8u);
      EXPECT_EQ(*image, expected[indexed]);
      if (multiDraw) {
        EXPECT_EQ(*image, loopImages[indexed]);
      } else {
        loopImages[indexed] = *image;
      }
    }
  }
}
} // namespace
} // namespace aurora::render::detail

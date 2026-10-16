#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "logging.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_api.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_layout.hpp"
#include "render/graph_pool.hpp"
#include "render/module.hpp"
#include "render/scene_depth.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
using wgpu::TextureFormat;

static_assert(sizeof(render::FrameConstants) == 48);
static_assert(std::is_same_v<detail::FrameConstants, render::FrameConstants>);
static_assert(offsetof(FrameUniforms, constants) == 0 && sizeof(FrameUniforms) == 256);
static_assert(detail::MaxPassParamsSize == render::MaxPassParamsSize);

struct External {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  uint32_t paramsSize = 0;
  std::function<void(PassBuilder&)> declare;
  std::function<void(render::PassContext&)> encode;
  int encodes = 0;
  std::vector<uint32_t> submitted;
};

void declare_external(PassBuilder& builder, void* userdata) {
  const auto& pass = *static_cast<const External*>(userdata);
  if (pass.declare) {
    pass.declare(builder);
  }
}

void encode_external(render::PassContext& ctx, void* userdata) {
  auto& pass = *static_cast<External*>(userdata);
  ++pass.encodes;
  if (pass.encode) {
    pass.encode(ctx);
  }
}

void external_submitted(uint32_t frame, void* userdata) {
  static_cast<External*>(userdata)->submitted.push_back(frame);
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

struct TestFrame {
  explicit TestFrame(std::shared_ptr<const CompiledGraph> compiled)
  : graph(std::move(compiled)), state{.graph = graph} {}

  std::shared_ptr<const CompiledGraph> graph;
  FrameState state;

  [[nodiscard]] NodeId node(std::string_view name) const { return graph->find_node(name); }
};

StageData stage_for(const TestFrame& frame, Stage stage, uint64_t frameId) {
  const CompiledGraph& graph = *frame.graph;
  const StageProgram& program = graph.stages[static_cast<size_t>(stage)];
  const auto passCount = static_cast<uint32_t>(program.passes.size());
  return {
      .stage = stage,
      .frameId = frameId,
      .frameIndex = static_cast<uint32_t>(frameId * 10),
      .frame = &frame.state,
      .passCount = passCount,
      .enabled = std::vector<uint64_t>((passCount + 63) / 64, ~uint64_t{0}),
      .dummies = std::vector<uint64_t>((program.useBits + 63) / 64, 0),
      .prevValid = std::vector<uint64_t>((graph.histories.size() + 63) / 64, 0),
  };
}

wgpu::ShaderModule shader_module(std::string_view source) {
  const std::string code{source};
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = code.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl};
  return webgpu::g_device.CreateShaderModule(&descriptor);
}

TexDesc efb_tex(TextureFormat format) { return {.format = format, .sizeClass = SizeClass::Efb}; }

class RenderGraphApiGpu : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    reset_module_state();
  }

  void TearDown() override {
    if (!m_device) {
      return;
    }
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    gpu_heap::shutdown();
    testing::clear_passes();
    testing::reset_external_passes();
    reset_frame_cull();
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  static void reset_module_state() {
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    // Module initialisation registers it in the app, and these tests never run that
    register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
    testing::clear_passes();
    testing::reset_external_passes();
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

  PassId add(std::string name, PassKind kind, std::function<void(PassBuilder&)> declare,
             std::function<void(render::PassContext&)> encode = {}, uint32_t paramsSize = 0, bool afterSubmit = false) {
    External& pass = m_externals.emplace_back(External{
        .name = std::move(name),
        .kind = kind,
        .paramsSize = paramsSize,
        .declare = std::move(declare),
        .encode = std::move(encode),
    });
    const PassId id = render::register_pass({
        .name = pass.name.c_str(),
        .stage = pass.stage,
        .kind = pass.kind,
        .paramsSize = pass.paramsSize,
        .declare = declare_external,
        .encode = encode_external,
        .afterSubmit = afterSubmit ? external_submitted : nullptr,
        .userdata = &pass,
    });
    EXPECT_NE(id, InvalidPass);
    return id;
  }

  // Mirrors the latch: params and enabled bits first, then a compile that appends the external set
  static TestFrame compile_externals(DepthSource sceneDepthSource = DepthSource::Unavailable) {
    latch_external_passes();
    const std::shared_ptr<const ExternalPassSet> set = snapshot_external_passes();
    std::vector<PassDesc> passes;
    append_external_passes(*set, &external_encode, passes);
    GraphInputs inputs;
    inputs.env = gather_graph_environment();
    inputs.passes = passes;
    inputs.externals = set;
    inputs.sceneDepthSource = sceneDepthSource;
    const std::shared_ptr<CompiledGraph> graph = compile_graph(inputs);
    prepare_graph_bindings(*graph);
    return TestFrame{graph};
  }

  void push_validation_scope() const { m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation); }

  [[nodiscard]] std::optional<std::string> pop_validation_scope() const {
    return test::pop_validation_error(*m_device);
  }

  std::optional<std::string> run_frame(const TestFrame& frame, uint64_t frameId, StageData* out = nullptr,
                                       std::shared_ptr<const SceneDepth> sceneDepth = nullptr) const {
    g_uniforms.clear();
    g_pushes.clear();
    on_worker_frame_begin(frameId, &frame.state);
    StageData stage = stage_for(frame, Stage::OpaqueDone, frameId);
    stage.sceneDepth = std::move(sceneDepth);
    build_stage_bindings(stage, &fake_sink);
    if (!g_uniforms.empty()) {
      g_uniforms.resize((g_uniforms.size() + 3) & ~size_t{3});
      m_device->queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, g_uniforms.data(), g_uniforms.size());
    }
    push_validation_scope();
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    run_stage(encoder, stage);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    if (out != nullptr) {
      *out = std::move(stage);
    }
    return pop_validation_scope();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::deque<External> m_externals;
};

TEST_F(RenderGraphApiGpu, FrameConstantsMatchInternal) {
  EXPECT_EQ(sizeof(render::FrameConstants), 48u);
  EXPECT_EQ(offsetof(FrameUniforms, constants), 0u);
  EXPECT_EQ(sizeof(FrameUniforms), FrameUniformsSize);
  EXPECT_EQ(render::FrameFlagReversedZ, 1u);
  EXPECT_EQ(render::FrameFlagMultisampled, 2u);
  EXPECT_EQ(static_cast<uint32_t>(render::FrameBinding::Params), 1u);
  EXPECT_EQ(static_cast<uint32_t>(render::FrameBinding::DepthCompare), 5u);
}

TEST_F(RenderGraphApiGpu, FrameWgslValidates) {
  const std::string source = std::string{render::FrameWgsl} + R"(
struct Params {
  scale: vec4f,
};
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  _ = aurora_frame.inv_efb_size * params.scale.xy;
}
)";
  std::optional<std::string> error = "not encoded";
  add(
      "mod.frame", PassKind::Compute, [](PassBuilder& b) { b.side_effect(); },
      [&](render::PassContext& ctx) {
        EXPECT_TRUE(ctx.pass_layout());
        push_validation_scope();
        const wgpu::ComputePipelineDescriptor descriptor{
            .layout = ctx.pipeline_layout(),
            .compute = {.module = shader_module(source), .entryPoint = "main"},
        };
        const wgpu::ComputePipeline pipeline = ctx.device().CreateComputePipeline(&descriptor);
        error = pop_validation_scope();
        if (const wgpu::ComputePassEncoder pass = ctx.begin_compute()) {
          pass.SetPipeline(pipeline);
          pass.DispatchWorkgroups(1);
          pass.End();
        }
      },
      16);
  const TestFrame frame = compile_externals();
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.frame")].status, CullReason::None);
  EXPECT_EQ(run_frame(frame, 1), std::nullopt);
  EXPECT_EQ(error, std::nullopt);
}

TEST_F(RenderGraphApiGpu, ExternalParamsReachContext) {
  std::vector<std::byte> seen;
  const PassId id = add(
      "mod.params", PassKind::Compute, [](PassBuilder& b) { b.side_effect(); },
      [&](render::PassContext& ctx) { seen.assign(ctx.params().begin(), ctx.params().end()); }, 16);
  const std::array<std::byte, 4> payload{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  ASSERT_TRUE(render::set_pass_params(id, payload.data(), payload.size()));
  const TestFrame frame = compile_externals();
  StageData stage;
  ASSERT_EQ(run_frame(frame, 1, &stage), std::nullopt);

  EXPECT_EQ(std::ranges::count(g_pushes, MaxPassParamsSize), 1);
  ASSERT_EQ(stage.bindings.params.size(), 1u);
  const uint32_t offset = stage.bindings.params[0].uniformOffset;
  ASSERT_GE(g_uniforms.size(), offset + MaxPassParamsSize);
  EXPECT_EQ(std::memcmp(g_uniforms.data() + offset, payload.data(), payload.size()), 0);
  EXPECT_TRUE(std::all_of(g_uniforms.begin() + offset + payload.size(), g_uniforms.begin() + offset + MaxPassParamsSize,
                          [](uint8_t value) { return value == 0; }));
  std::vector<std::byte> expected{payload.begin(), payload.end()};
  expected.resize(16);
  EXPECT_EQ(seen, expected);
}

TEST_F(RenderGraphApiGpu, ExternalBindingPlanOrder) {
  add("mod.a", PassKind::Compute,
      [](PassBuilder& b) { b.create("mod.tex", efb_tex(TextureFormat::RGBA8Unorm), Access::StorageWrite); });
  add(
      "mod.b", PassKind::Compute,
      [](PassBuilder& b) {
        b.use("mod.tex", Access::Sampled);
        b.create("mod.out", efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite);
        b.side_effect();
      },
      {}, 16);
  const TestFrame frame = compile_externals();
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.a")].status, CullReason::None);
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.b")].status, CullReason::None);
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("mod.b")];
  ASSERT_EQ(plan.slots.size(), 2u);
  EXPECT_EQ(plan.slots[0].binding, 0u);
  EXPECT_EQ(plan.slots[0].type, BindingType::SampledTexture);
  EXPECT_EQ(plan.slots[0].sampleType, wgpu::TextureSampleType::Float);
  EXPECT_EQ(plan.slots[0].viewDimension, wgpu::TextureViewDimension::e2D);
  EXPECT_EQ(plan.slots[1].binding, 1u);
  EXPECT_EQ(plan.slots[1].type, BindingType::WriteOnlyStorageTexture);
  EXPECT_EQ(plan.slots[1].storageFormat, TextureFormat::RGBA16Float);
  for (const BindingSlot& slot : plan.slots) {
    EXPECT_EQ(slot.visibility, wgpu::ShaderStage::Compute);
  }
  EXPECT_EQ(plan.paramsSize, 16u);
}

TEST_F(RenderGraphApiGpu, RenderBindingsAreFragmentOnly) {
  add("mod.src", PassKind::Compute,
      [](PassBuilder& b) { b.create("mod.src", efb_tex(TextureFormat::RGBA8Unorm), Access::StorageWrite); });
  add("mod.draw", PassKind::Render, [](PassBuilder& b) {
    b.use("mod.src", Access::Sampled);
    b.use(resource::GxColor, Access::ColorRMW);
  });
  const TestFrame frame = compile_externals();
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.draw")].status, CullReason::None);
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("mod.draw")];
  ASSERT_EQ(plan.slots.size(), 1u);
  EXPECT_EQ(plan.slots[0].visibility, wgpu::ShaderStage::Fragment);
  ASSERT_EQ(plan.colors.size(), 1u);
}

TEST_F(RenderGraphApiGpu, ColorTargetMasksAlphaOnGxColor) {
  use_globals({.efbSize = {64, 64, 1}, .createEfb = true});
  constexpr auto Rgb = wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
  std::optional<wgpu::ColorTargetState> gxTarget;
  std::optional<wgpu::ColorTargetState> ownTarget;
  const auto endPass = [](render::PassContext& ctx) {
    if (const wgpu::RenderPassEncoder pass = ctx.begin_render()) {
      pass.End();
    }
  };
  add(
      "mod.gx", PassKind::Render, [](PassBuilder& b) { b.use(resource::GxColor, Access::ColorRMW); },
      [&](render::PassContext& ctx) {
        gxTarget = ctx.color_target(TexRef{0});
        EXPECT_EQ(ctx.sample_count(TexRef{0}), 1u);
        endPass(ctx);
      });
  add(
      "mod.own", PassKind::Render,
      [](PassBuilder& b) {
        b.create("mod.target", efb_tex(TextureFormat::RGBA16Float), Access::ColorWrite);
        b.side_effect();
      },
      [&](render::PassContext& ctx) {
        ownTarget = ctx.color_target(TexRef{0});
        endPass(ctx);
      });
  const TestFrame frame = compile_externals();
  ASSERT_EQ(run_frame(frame, 1), std::nullopt);
  ASSERT_TRUE(gxTarget.has_value());
  ASSERT_TRUE(ownTarget.has_value());
  EXPECT_EQ(gxTarget->writeMask, Rgb);
  EXPECT_EQ(gxTarget->format, webgpu::g_frameBuffer.format);
  EXPECT_EQ(gxTarget->blend, nullptr);
  EXPECT_EQ(ownTarget->writeMask, wgpu::ColorWriteMask::All);
  EXPECT_EQ(ownTarget->format, TextureFormat::RGBA16Float);
}

TEST_F(RenderGraphApiGpu, ExternalEncodeRunsCallbackAndQueuesAfterSubmit) {
  std::vector<uint32_t> frames;
  std::string name;
  const PassId id = add(
      "mod.submit", PassKind::Compute, [](PassBuilder& b) { b.side_effect(); },
      [&](render::PassContext& ctx) {
        frames.push_back(ctx.frame());
        name = ctx.name();
        EXPECT_EQ(ctx.stage(), Stage::OpaqueDone);
      },
      0, true);
  const External& pass = m_externals.back();
  const TestFrame frame = compile_externals();
  StageData stage;
  ASSERT_EQ(run_frame(frame, 1, &stage), std::nullopt);
  ASSERT_EQ(pass.encodes, 1);
  EXPECT_EQ(frames, (std::vector<uint32_t>{stage.frameIndex}));
  EXPECT_EQ(name, "mod.submit");
  EXPECT_TRUE(pass.submitted.empty());
  render::detail::after_frame(nullptr);
  EXPECT_EQ(pass.submitted, (std::vector<uint32_t>{stage.frameIndex}));
  render::detail::after_frame(nullptr);
  EXPECT_EQ(pass.submitted.size(), 1u);

  ASSERT_EQ(run_frame(frame, 2), std::nullopt);
  EXPECT_EQ(pass.encodes, 2);
  render::unregister_pass(id);
  render::detail::after_frame(nullptr);
  EXPECT_EQ(pass.submitted.size(), 1u);
  ASSERT_EQ(run_frame(frame, 3), std::nullopt);
  EXPECT_EQ(pass.encodes, 2);
}

struct DepthTestTexture {
  wgpu::Texture texture;
  wgpu::TextureView view;
};

DepthTestTexture depth_test_texture(TextureFormat format, uint32_t samples, wgpu::TextureUsage usage) {
  const wgpu::TextureDescriptor descriptor{
      .label = "scene depth test",
      .usage = usage,
      .size = {64, 64, 1},
      .format = format,
      .sampleCount = samples,
  };
  wgpu::Texture texture = webgpu::g_device.CreateTexture(&descriptor);
  wgpu::TextureView view = texture.CreateView();
  return {.texture = std::move(texture), .view = std::move(view)};
}

constexpr wgpu::TextureUsage SampledDepthUsage =
    wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;

std::shared_ptr<const SceneDepth> sampled_scene_depth(DepthSource source) {
  const bool copy = source == DepthSource::Copy;
  const TextureFormat format = copy ? TextureFormat::R32Float : TextureFormat::Depth32Float;
  DepthTestTexture sampled = depth_test_texture(format, 1, SampledDepthUsage);
  return std::make_shared<const SceneDepth>(SceneDepth{
      .source = source,
      .sampledTexture = std::move(sampled.texture),
      .sampled = std::move(sampled.view),
      .sampledFormat = format,
      .size = {64, 64, 1},
  });
}

TEST_F(RenderGraphApiGpu, SceneDepthSampledPlanIsUnfilterableFloat2d) {
  add("mod.cs", PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxDepth, Access::Sampled);
    b.side_effect();
  });
  add("mod.fs", PassKind::Render, [](PassBuilder& b) {
    b.use(resource::GxDepth, Access::Sampled);
    b.use(resource::GxColor, Access::ColorRMW);
  });
  const TestFrame frame = compile_externals(DepthSource::Direct);
  for (const auto& [name, visibility] :
       {std::pair{"mod.cs", wgpu::ShaderStage::Compute}, std::pair{"mod.fs", wgpu::ShaderStage::Fragment}}) {
    ASSERT_EQ(frame.graph->nodes[frame.node(name)].status, CullReason::None) << name;
    const BindingPlan& plan = frame.graph->bindingPlans[frame.node(name)];
    ASSERT_EQ(plan.slots.size(), 1u) << name;
    const BindingSlot& slot = plan.slots[0];
    EXPECT_EQ(slot.type, BindingType::SampledTexture) << name;
    EXPECT_EQ(slot.sampleType, wgpu::TextureSampleType::UnfilterableFloat) << name;
    EXPECT_EQ(slot.viewDimension, wgpu::TextureViewDimension::e2D) << name;
    EXPECT_FALSE(slot.multisampled) << name;
    EXPECT_EQ(slot.visibility, visibility) << name;
  }
}

TEST_F(RenderGraphApiGpu, SceneDepthBindsDepthTextureAndSnapshot) {
  constexpr std::string_view Source = R"(
@group(1) @binding(0) var scene_depth: texture_2d<f32>;

@compute @workgroup_size(1)
fn main() {
  _ = textureLoad(scene_depth, vec2u(0u, 0u), 0).r;
}
)";
  std::optional<std::string> pipelineError = "not encoded";
  std::vector<TextureFormat> formats;
  std::vector<TextureFormat> textureFormats;
  add(
      "mod.depth", PassKind::Compute,
      [](PassBuilder& b) {
        b.use(resource::GxDepth, Access::Sampled);
        b.side_effect();
      },
      [&](render::PassContext& ctx) {
        formats.push_back(ctx.format(TexRef{0}));
        const wgpu::Texture texture = ctx.texture(TexRef{0});
        textureFormats.push_back(texture ? texture.GetFormat() : TextureFormat::Undefined);
        EXPECT_EQ(ctx.extent(TexRef{0}).width, 64u);
        EXPECT_EQ(ctx.sample_count(TexRef{0}), 1u);
        push_validation_scope();
        const wgpu::ComputePipelineDescriptor descriptor{
            .layout = ctx.pipeline_layout(),
            .compute = {.module = shader_module(Source), .entryPoint = "main"},
        };
        const wgpu::ComputePipeline pipeline = ctx.device().CreateComputePipeline(&descriptor);
        pipelineError = pop_validation_scope();
        if (const wgpu::ComputePassEncoder pass = ctx.begin_compute()) {
          pass.SetPipeline(pipeline);
          pass.DispatchWorkgroups(1);
          pass.End();
        }
      });
  const TestFrame frame = compile_externals(DepthSource::Direct);
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.depth")].status, CullReason::None);
  EXPECT_EQ(run_frame(frame, 1, nullptr, sampled_scene_depth(DepthSource::Direct)), std::nullopt);
  EXPECT_EQ(pipelineError, std::nullopt);
  EXPECT_EQ(run_frame(frame, 2, nullptr, sampled_scene_depth(DepthSource::Copy)), std::nullopt);
  EXPECT_EQ(pipelineError, std::nullopt);
  EXPECT_EQ(formats, (std::vector{TextureFormat::Depth32Float, TextureFormat::R32Float}));
  EXPECT_EQ(textureFormats, formats);
}

TEST_F(RenderGraphApiGpu, DepthReadIsAttachmentOnly) {
  std::optional<TextureFormat> format;
  std::optional<uint32_t> samples;
  WGPUTexture bound = nullptr;
  add(
      "mod.depth_read", PassKind::Render,
      [](PassBuilder& b) {
        b.use(resource::GxDepth, Access::DepthRead);
        b.side_effect();
      },
      [&](render::PassContext& ctx) {
        format = ctx.format(TexRef{0});
        samples = ctx.sample_count(TexRef{0});
        bound = ctx.texture(TexRef{0}).Get();
        if (const wgpu::RenderPassEncoder pass = ctx.begin_render()) {
          pass.End();
        }
      });
  const TestFrame frame = compile_externals(DepthSource::Direct);
  ASSERT_EQ(frame.graph->nodes[frame.node("mod.depth_read")].status, CullReason::None);
  const BindingPlan& plan = frame.graph->bindingPlans[frame.node("mod.depth_read")];
  EXPECT_TRUE(plan.slots.empty());
  EXPECT_TRUE(plan.colors.empty());
  ASSERT_TRUE(plan.depth.has_value());
  EXPECT_EQ(plan.depth->access, Access::DepthRead);
  EXPECT_FALSE(plan.depth->creatorWrite);

  const DepthTestTexture target = depth_test_texture(TextureFormat::Depth32Float, 1, SampledDepthUsage);
  const auto depth = std::make_shared<const SceneDepth>(SceneDepth{
      .source = DepthSource::Direct,
      .attachmentTexture = target.texture,
      .attachment = target.view,
      .attachmentFormat = TextureFormat::Depth32Float,
      .size = {64, 64, 1},
  });
  testing::record_bind_details(true);
  const std::optional<std::string> error = run_frame(frame, 1, nullptr, depth);
  testing::record_bind_details(false);
  EXPECT_EQ(error, std::nullopt);
  EXPECT_EQ(format, TextureFormat::Depth32Float);
  EXPECT_EQ(samples, 1u);
  EXPECT_EQ(bound, target.texture.Get());
  const std::vector<testing::AttachmentOps> ops = testing::last_attachment_ops();
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_TRUE(ops[0].depth);
  EXPECT_TRUE(ops[0].readOnly);
  EXPECT_EQ(ops[0].load, wgpu::LoadOp::Undefined);
  EXPECT_EQ(ops[0].store, wgpu::StoreOp::Undefined);
  EXPECT_EQ(ops[0].width, 64u);
}

TEST_F(RenderGraphApiGpu, GxDepthResolverPerAccess) {
  const DepthTestTexture sampled = depth_test_texture(TextureFormat::R32Float, 1, SampledDepthUsage);
  const DepthTestTexture attachment =
      depth_test_texture(TextureFormat::Depth32Float, 4, wgpu::TextureUsage::RenderAttachment);
  StageData stage{.stage = Stage::FrameEnd};
  GxBinding out;
  EXPECT_FALSE(resolve_gx_depth(stage, Access::Sampled, out));

  stage.sceneDepth = std::make_shared<const SceneDepth>(SceneDepth{
      .source = DepthSource::Copy,
      .sampledTexture = sampled.texture,
      .sampled = sampled.view,
      .sampledFormat = TextureFormat::R32Float,
      .attachmentTexture = attachment.texture,
      .attachment = attachment.view,
      .attachmentFormat = TextureFormat::Depth32Float,
      .attachmentSamples = 4,
      .size = {64, 64, 1},
  });
  ASSERT_TRUE(resolve_gx_depth(stage, Access::Sampled, out));
  EXPECT_EQ(out.texture.Get(), sampled.texture.Get());
  EXPECT_EQ(out.sampleView.Get(), sampled.view.Get());
  EXPECT_EQ(out.format, TextureFormat::R32Float);
  EXPECT_EQ(out.samples, 1u);
  EXPECT_EQ(out.size.width, 64u);

  out = {};
  ASSERT_TRUE(resolve_gx_depth(stage, Access::DepthRead, out));
  EXPECT_EQ(out.texture.Get(), attachment.texture.Get());
  EXPECT_EQ(out.attachmentView.Get(), attachment.view.Get());
  EXPECT_EQ(out.format, TextureFormat::Depth32Float);
  EXPECT_EQ(out.samples, 4u);
  EXPECT_EQ(out.size.height, 64u);

  stage.sceneDepth = std::make_shared<const SceneDepth>(SceneDepth{
      .source = DepthSource::Copy,
      .attachmentTexture = attachment.texture,
      .attachment = attachment.view,
      .attachmentFormat = TextureFormat::Depth32Float,
      .attachmentSamples = 4,
  });
  EXPECT_FALSE(resolve_gx_depth(stage, Access::Sampled, out));
  EXPECT_TRUE(resolve_gx_depth(stage, Access::DepthRead, out));
}
} // namespace
} // namespace aurora::render::detail

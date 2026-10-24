#include <gtest/gtest.h>

#include "render_graph_planned.hpp"

#include "render/geometry/hzb.hpp"
#include "render/graph.hpp"
#include "render/graph_layout.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
namespace {
using wgpu::ShaderStage;
using wgpu::TextureFormat;
using wgpu::TextureSampleType;
using wgpu::TextureViewDimension;

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  PassClass cls = PassClass::Required;
  uint16_t order = 0;
  uint32_t paramsSize = 0;
  std::function<void(InternalPassBuilder&)> body;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  static_cast<const TestPass*>(state)->body(builder);
}

GraphInputs test_inputs(const GraphLimits& limits = planned::full_limits()) {
  GraphInputs inputs;
  inputs.env.limits = limits;
  inputs.env.efbSize = {1280, 720};
  inputs.env.outputSize = {1280, 720};
  inputs.env.gxFormats = planned::gx_formats_tier_p();
  inputs.env.gxUsage = planned::gx_usage_future();
  return inputs;
}

std::shared_ptr<CompiledGraph> compile(const std::vector<TestPass>& passes, GraphInputs inputs = test_inputs()) {
  std::vector<PassDesc> descs;
  for (const TestPass& pass : passes) {
    descs.push_back({
        .name = pass.name,
        .stage = pass.stage,
        .kind = pass.kind,
        .cls = pass.cls,
        .order = pass.order,
        .paramsSize = pass.paramsSize,
        .declare = declare_test_pass,
        .state = &pass,
    });
  }
  inputs.passes = descs;
  return compile_graph(inputs);
}

const CompiledNode& node_of(const CompiledGraph& graph, std::string_view name) {
  const NodeId node = graph.find_node(name);
  EXPECT_NE(node, InvalidNode) << name;
  return graph.nodes.at(node);
}

const BindingPlan& plan_of(const CompiledGraph& graph, std::string_view name) {
  return graph.bindingPlans.at(graph.find_node(name));
}

TexDesc absolute(TextureFormat format, uint32_t width = 64, uint32_t height = 64, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

std::vector<TestPass> declaration_order_passes(uint32_t paramsSize = 0, Access first = Access::StorageRead) {
  return {
      {.name = "t.make",
       .order = 0,
       .body =
           [](InternalPassBuilder& b) {
             b.create_buffer("t.a", {.size = 256}, Access::StorageWrite);
             b.create("t.tex", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
             b.create_buffer("t.args", {.size = 64}, Access::StorageWrite);
           }},
      {.name = "t.pass",
       .order = 1,
       .paramsSize = paramsSize,
       .body =
           [first](InternalPassBuilder& b) {
             b.use_buffer("t.a", first);
             b.use("t.tex", Access::Sampled);
             b.create_buffer("t.out", {.size = 256}, Access::StorageWrite);
             b.use_buffer("t.args", Access::Indirect);
             b.create("t.img", absolute(TextureFormat::RGBA16Float), Access::StorageWrite);
             b.side_effect();
           }},
  };
}

BindingSlot sampled_slot(TextureFormat format, bool float32Filterable, uint32_t layers = 1) {
  const std::array resources{CompiledResource{.name = "r", .described = true, .tex = absolute(format, 4, 4, layers)}};
  const std::array uses{CompiledUse{.resource = 0, .access = Access::Sampled, .visibility = ShaderStage::Compute}};
  GraphEnvironment env;
  env.limits.features.float32Filterable = float32Filterable;
  BindingPlan plan;
  EXPECT_EQ(derive_binding_plan({.uses = uses, .resources = resources, .env = &env}, plan), CullReason::None);
  EXPECT_EQ(plan.slots.size(), 1u);
  return plan.slots.empty() ? BindingSlot{} : plan.slots.front();
}

TEST(GraphLayoutTest, Group1FollowsDeclarationOrder) {
  const auto passes = declaration_order_passes();
  const auto graph = compile(passes);
  ASSERT_EQ(node_of(*graph, "t.pass").status, CullReason::None);
  const BindingPlan& plan = plan_of(*graph, "t.pass");
  ASSERT_EQ(plan.slots.size(), 4u);
  const std::array types{BindingType::ReadOnlyStorageBuffer, BindingType::SampledTexture, BindingType::StorageBuffer,
                         BindingType::WriteOnlyStorageTexture};
  const std::array uses{0u, 1u, 2u, 4u};
  for (uint32_t i = 0; i < plan.slots.size(); ++i) {
    const BindingSlot& slot = plan.slots[i];
    EXPECT_EQ(slot.binding, i);
    EXPECT_EQ(slot.type, types[i]) << i;
    EXPECT_EQ(slot.use, uses[i]) << i;
    EXPECT_EQ(slot.visibility, ShaderStage::Compute) << i;
    EXPECT_FALSE(slot.multisampled);
  }
  EXPECT_EQ(plan.slots[1].sampleType, TextureSampleType::Float);
  EXPECT_EQ(plan.slots[1].viewDimension, TextureViewDimension::e2D);
  EXPECT_EQ(plan.slots[3].storageFormat, TextureFormat::RGBA16Float);
  EXPECT_EQ(plan.slots[3].viewDimension, TextureViewDimension::e2D);
  EXPECT_TRUE(plan.colors.empty());
  EXPECT_FALSE(plan.depth);
  EXPECT_FALSE(plan.manual);
  EXPECT_EQ(plan.layoutHash, layout_hash(plan));
  EXPECT_FALSE(plan.layout);
}

TEST(GraphLayoutTest, ParamsStayInGroupZero) {
  const auto without = declaration_order_passes(0);
  const auto with = declaration_order_passes(20);
  const auto a = compile(without);
  const auto b = compile(with);
  const BindingPlan& plain = plan_of(*a, "t.pass");
  const BindingPlan& params = plan_of(*b, "t.pass");
  ASSERT_EQ(params.slots.size(), 4u);
  for (size_t i = 0; i < params.slots.size(); ++i) {
    EXPECT_EQ(params.slots[i].binding, i);
    EXPECT_EQ(params.slots[i].type, plain.slots[i].type);
  }
  EXPECT_EQ(params.layoutHash, plain.layoutHash);
  EXPECT_EQ(params.paramsSize, 20u);
  EXPECT_EQ(plain.paramsSize, 0u);
  EXPECT_EQ(node_of(*b, "t.pass").paramsSize, 20u);
}

TEST(GraphLayoutTest, BindingTypesPerAccess) {
  struct Row {
    Access access;
    BindingType texture;
    BindingType buffer;
  };
  constexpr std::array Rows{
      Row{Access::Sampled, BindingType::SampledTexture, BindingType::None},
      Row{Access::Uniform, BindingType::None, BindingType::UniformBuffer},
      Row{Access::StorageRead, BindingType::ReadOnlyStorageTexture, BindingType::ReadOnlyStorageBuffer},
      Row{Access::StorageWrite, BindingType::WriteOnlyStorageTexture, BindingType::StorageBuffer},
      Row{Access::StorageRW, BindingType::ReadWriteStorageTexture, BindingType::StorageBuffer},
      Row{Access::ColorWrite, BindingType::None, BindingType::None},
      Row{Access::ColorRMW, BindingType::None, BindingType::None},
      Row{Access::DepthRead, BindingType::None, BindingType::None},
      Row{Access::DepthWrite, BindingType::None, BindingType::None},
      Row{Access::CopySrc, BindingType::None, BindingType::None},
      Row{Access::CopyDst, BindingType::None, BindingType::None},
      Row{Access::Vertex, BindingType::None, BindingType::None},
      Row{Access::Index, BindingType::None, BindingType::None},
      Row{Access::Indirect, BindingType::None, BindingType::None},
  };
  static_assert(Rows.size() == static_cast<size_t>(Access::Indirect) + 1);
  for (const Row& row : Rows) {
    EXPECT_EQ(binding_type(row.access, false), row.texture) << access_name(row.access);
    EXPECT_EQ(binding_type(row.access, true), row.buffer) << access_name(row.access);
  }

  EXPECT_EQ(sampled_slot(TextureFormat::Depth32Float, false).sampleType, TextureSampleType::Depth);
  EXPECT_EQ(sampled_slot(TextureFormat::R32Float, false).sampleType, TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(sampled_slot(TextureFormat::R32Float, true).sampleType, TextureSampleType::Float);
  EXPECT_EQ(sampled_slot(TextureFormat::R32Uint, false).sampleType, TextureSampleType::Uint);
  EXPECT_EQ(sampled_slot(TextureFormat::R32Sint, false).sampleType, TextureSampleType::Sint);
  EXPECT_EQ(sampled_slot(TextureFormat::RGBA8Unorm, false).viewDimension, TextureViewDimension::e2D);
  EXPECT_EQ(sampled_slot(TextureFormat::RGBA8Unorm, false, 4).viewDimension, TextureViewDimension::e2DArray);
}

TEST(GraphLayoutTest, VisibilityFollowsUse) {
  const std::vector<TestPass> passes{
      {.name = "v.make",
       .order = 0,
       .body =
           [](InternalPassBuilder& b) {
             b.create("v.src", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "v.default",
       .kind = PassKind::Render,
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.use("v.src", Access::Sampled);
             b.create("v.a", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           }},
      {.name = "v.both",
       .kind = PassKind::Render,
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.use("v.src", Access::Sampled, UseOptions{.visibility = ShaderStage::Vertex | ShaderStage::Fragment});
             b.create("v.b", absolute(TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.side_effect();
           }},
  };
  const auto graph = compile(passes);
  const BindingPlan& fragment = plan_of(*graph, "v.default");
  const BindingPlan& both = plan_of(*graph, "v.both");
  ASSERT_EQ(fragment.slots.size(), 1u);
  ASSERT_EQ(both.slots.size(), 1u);
  EXPECT_EQ(fragment.slots[0].visibility, ShaderStage::Fragment);
  EXPECT_EQ(both.slots[0].visibility, ShaderStage::Vertex | ShaderStage::Fragment);
  ASSERT_EQ(fragment.colors.size(), 1u);
  EXPECT_EQ(fragment.colors[0].use, 1u);
  EXPECT_TRUE(fragment.colors[0].creatorWrite);
  EXPECT_EQ(fragment.colors[0].samples, 1u);
  EXPECT_NE(fragment.layoutHash, both.layoutHash);
}

TEST(GraphLayoutTest, ManualPlanHasNoGroup1Entries) {
  const std::vector<TestPass> passes{
      {.name = "m.pass",
       .order = 0,
       .paramsSize = 16,
       .body =
           [](InternalPassBuilder& b) {
             b.create("m.out", absolute(TextureFormat::RGBA16Float), Access::StorageWrite);
             b.create_buffer("m.buf", {.size = 256}, Access::StorageWrite);
             b.manual_bindings();
             b.side_effect();
           }},
  };
  const auto graph = compile(passes);
  ASSERT_EQ(node_of(*graph, "m.pass").status, CullReason::None);
  const BindingPlan& plan = plan_of(*graph, "m.pass");
  EXPECT_TRUE(plan.manual);
  EXPECT_TRUE(plan.slots.empty());
  EXPECT_EQ(plan.paramsSize, 16u);

  BindingPlan automatic = plan;
  automatic.manual = false;
  EXPECT_NE(layout_hash(automatic), plan.layoutHash);
}

TEST(GraphLayoutTest, SampleCountMismatchDisables) {
  const std::vector<TestPass> passes{
      {.name = "s.mixed",
       .kind = PassKind::Render,
       .body =
           [](InternalPassBuilder& b) {
             b.use("gx.color", Access::ColorRMW);
             b.create("s.extra", {.format = TextureFormat::RGBA8Unorm}, Access::ColorWrite);
           }},
  };
  GraphInputs msaa = test_inputs();
  msaa.env.gxSampleCount = 4;
  const auto disabled = compile(passes, msaa);
  EXPECT_EQ(node_of(*disabled, "s.mixed").status, CullReason::SampleCountMismatch);
  EXPECT_TRUE(std::ranges::any_of(disabled->diagnostics, [](const Diagnostic& diagnostic) {
    return diagnostic.reason == CullReason::SampleCountMismatch &&
           diagnostic.message.find("4 samples") != std::string::npos;
  }));
  EXPECT_TRUE(plan_of(*disabled, "s.mixed").colors.empty());

  const auto single = compile(passes);
  EXPECT_EQ(node_of(*single, "s.mixed").status, CullReason::None);
  const BindingPlan& plan = plan_of(*single, "s.mixed");
  ASSERT_EQ(plan.colors.size(), 2u);
  EXPECT_FALSE(plan.colors[0].creatorWrite);
  EXPECT_TRUE(plan.colors[1].creatorWrite);

  const std::vector<TestPass> alone{
      {.name = "s.gx",
       .kind = PassKind::Render,
       .body = [](InternalPassBuilder& b) { b.use("gx.color", Access::ColorRMW); }},
  };
  const auto gx = compile(alone, msaa);
  EXPECT_EQ(node_of(*gx, "s.gx").status, CullReason::None);
  ASSERT_EQ(plan_of(*gx, "s.gx").colors.size(), 1u);
  EXPECT_EQ(plan_of(*gx, "s.gx").colors[0].samples, 4u);
}

TEST(GraphLayoutTest, BindingViewMismatchOnCompat) {
  const std::vector<TestPass> passes{
      {.name = "c.make",
       .order = 0,
       .body =
           [](InternalPassBuilder& b) {
             b.create("c.faces", absolute(TextureFormat::RGBA8Unorm, 64, 64, 6), Access::StorageWrite);
           }},
      {.name = "c.cube",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.use("c.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::Cube});
             b.side_effect();
           }},
      {.name = "c.array",
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.use("c.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::e2DArray});
             b.side_effect();
           }},
  };
  const auto compat = compile(passes, test_inputs(planned::compat_limits()));
  EXPECT_EQ(node_of(*compat, "c.make").status, CullReason::None);
  EXPECT_EQ(node_of(*compat, "c.cube").status, CullReason::None);
  EXPECT_EQ(node_of(*compat, "c.array").status, CullReason::BindingViewMismatch);
  EXPECT_EQ(plan_of(*compat, "c.cube").slots.at(0).viewDimension, TextureViewDimension::Cube);
  const ResourceId faces = compat->find_resource("c.faces");
  EXPECT_TRUE(std::ranges::any_of(compat->diagnostics, [&](const Diagnostic& diagnostic) {
    return diagnostic.reason == CullReason::BindingViewMismatch && diagnostic.resource == faces;
  }));
  EXPECT_EQ(compat->slots.slots.at(compat->slots.resourceSlot.at(faces)).tex.bindingView, TextureViewDimension::Cube);

  const auto core = compile(passes);
  EXPECT_EQ(node_of(*core, "c.cube").status, CullReason::None);
  EXPECT_EQ(node_of(*core, "c.array").status, CullReason::None);
}

TEST(GraphLayoutTest, BindingViewMismatchCascades) {
  const TexDesc faces = absolute(TextureFormat::RGBA8Unorm, 64, 64, 6);
  const std::vector<TestPass> passes{
      {.name = "k.make",
       .order = 0,
       .body = [faces](InternalPassBuilder& b) { b.create("k.faces", faces, Access::StorageWrite); }},
      {.name = "k.feed",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.create("k.side", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "k.cube",
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.use("k.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::Cube});
             b.side_effect();
           }},
      {.name = "k.array",
       .order = 3,
       .body =
           [](InternalPassBuilder& b) {
             b.use("k.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::e2DArray});
             b.use("k.side", Access::Sampled);
             b.create("k.out", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "k.needs",
       .order = 4,
       .body =
           [](InternalPassBuilder& b) {
             b.use("k.out", Access::Sampled);
             b.side_effect();
           }},
      {.name = "k.dummy",
       .order = 5,
       .body =
           [](InternalPassBuilder& b) {
             b.use("k.out", Access::Sampled, Missing::BindDummyBlack);
             b.side_effect();
           }},
  };
  const auto graph = compile(passes, test_inputs(planned::compat_limits()));
  EXPECT_EQ(node_of(*graph, "k.make").status, CullReason::None);
  EXPECT_EQ(node_of(*graph, "k.cube").status, CullReason::None);
  EXPECT_EQ(node_of(*graph, "k.array").status, CullReason::BindingViewMismatch);
  EXPECT_EQ(node_of(*graph, "k.needs").status, CullReason::MissingInput);
  EXPECT_EQ(node_of(*graph, "k.feed").status, CullReason::NoConsumer);
  EXPECT_EQ(node_of(*graph, "k.dummy").status, CullReason::None);
  EXPECT_TRUE(graph->node_uses(graph->find_node("k.dummy"))[0].staticDummy);

  const auto reason_of = [&](std::string_view name) {
    const NodeId node = graph->find_node(name);
    const auto it = std::ranges::find(graph->diagnostics, node, &Diagnostic::node);
    return it != graph->diagnostics.end() ? it->reason : CullReason::None;
  };
  EXPECT_EQ(reason_of("k.array"), CullReason::BindingViewMismatch);
  EXPECT_EQ(reason_of("k.needs"), CullReason::MissingInput);
  EXPECT_EQ(reason_of("k.feed"), CullReason::NoConsumer);

  const SlotPlan& slots = graph->slots;
  ASSERT_NE(slots.resourceSlot.at(graph->find_resource("k.faces")), InvalidSlot);
  EXPECT_EQ(slots.slots.at(slots.resourceSlot.at(graph->find_resource("k.faces"))).tex.bindingView,
            TextureViewDimension::Cube);
  EXPECT_EQ(slots.resourceSlot.at(graph->find_resource("k.side")), InvalidSlot);
  EXPECT_EQ(slots.resourceSlot.at(graph->find_resource("k.out")), InvalidSlot);
  EXPECT_EQ(slots.slots.size(), 1u);
  EXPECT_TRUE(plan_of(*graph, "k.array").slots.empty());
  EXPECT_TRUE(plan_of(*graph, "k.feed").slots.empty());
}

TEST(GraphLayoutTest, DummyAndCulledReadersKeepCompatViews) {
  const UseOptions cube{.missing = Missing::BindDummyBlack, .dimension = TextureViewDimension::Cube};
  const UseOptions array{.missing = Missing::BindDummyBlack, .dimension = TextureViewDimension::e2DArray};
  const std::vector<TestPass> passes{
      {.name = "g.make",
       .cls = PassClass::Diagnostic,
       .order = 0,
       .body =
           [](InternalPassBuilder& b) {
             b.create("g.faces", absolute(TextureFormat::RGBA8Unorm, 64, 64, 6), Access::StorageWrite);
           }},
      {.name = "g.cube",
       .order = 1,
       .body =
           [cube](InternalPassBuilder& b) {
             b.use("g.faces", Access::Sampled, cube);
             b.side_effect();
           }},
      {.name = "g.array",
       .order = 2,
       .body =
           [array](InternalPassBuilder& b) {
             b.use("g.faces", Access::Sampled, array);
             b.side_effect();
           }},
      {.name = "n.make",
       .order = 3,
       .body =
           [](InternalPassBuilder& b) {
             b.create("n.faces", absolute(TextureFormat::RGBA8Unorm, 64, 64, 6), Access::StorageWrite);
           }},
      {.name = "n.culled",
       .order = 4,
       .body =
           [](InternalPassBuilder& b) {
             b.use("n.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::Cube});
             b.create("n.junk", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
      {.name = "n.live",
       .order = 5,
       .body =
           [](InternalPassBuilder& b) {
             b.use("n.faces", Access::Sampled, UseOptions{.dimension = TextureViewDimension::e2DArray});
             b.side_effect();
           }},
  };
  const auto graph = compile(passes, test_inputs(planned::compat_limits()));
  EXPECT_EQ(node_of(*graph, "g.make").status, CullReason::InspectorClosed);
  EXPECT_EQ(node_of(*graph, "g.cube").status, CullReason::None);
  EXPECT_EQ(node_of(*graph, "g.array").status, CullReason::None);
  EXPECT_TRUE(graph->node_uses(graph->find_node("g.cube"))[0].staticDummy);
  EXPECT_TRUE(graph->node_uses(graph->find_node("g.array"))[0].staticDummy);
  EXPECT_EQ(node_of(*graph, "n.culled").status, CullReason::NoConsumer);
  EXPECT_EQ(node_of(*graph, "n.live").status, CullReason::None);
  EXPECT_EQ(node_of(*graph, "n.make").status, CullReason::None);
  const ResourceId faces = graph->find_resource("n.faces");
  EXPECT_EQ(graph->slots.slots.at(graph->slots.resourceSlot.at(faces)).tex.bindingView, TextureViewDimension::e2DArray);
  EXPECT_FALSE(std::ranges::any_of(graph->diagnostics, [](const Diagnostic& diagnostic) {
    return diagnostic.reason == CullReason::BindingViewMismatch;
  }));
}

TEST(GraphLayoutTest, GxFormatsFollowFixedFormats) {
  std::array<CompiledResource, 2> resources{};
  resources[0] = {.name = "gbuf.debug_id", .kind = ResourceKind::Gx, .gx = GxResource::DebugId};
  resources[1] = {.name = "gbuf.fog", .kind = ResourceKind::Gx, .gx = GxResource::Fog};
  const std::array uses{
      CompiledUse{.resource = 0, .access = Access::Sampled, .visibility = ShaderStage::Compute},
      CompiledUse{.resource = 1, .access = Access::StorageRead, .visibility = ShaderStage::Compute},
  };
  const GraphEnvironment env;
  BindingPlan plan;
  ASSERT_EQ(derive_binding_plan({.uses = uses, .resources = resources, .env = &env}, plan), CullReason::None);
  ASSERT_EQ(plan.slots.size(), 2u);
  EXPECT_EQ(plan.slots[0].sampleType, TextureSampleType::Uint);
  EXPECT_EQ(plan.slots[0].viewDimension, TextureViewDimension::e2D);
  EXPECT_EQ(plan.slots[1].type, BindingType::ReadOnlyStorageTexture);
  EXPECT_EQ(plan.slots[1].storageFormat, TextureFormat::RGBA8Uint);

  CompiledGraph graph;
  graph.resources.assign(resources.begin(), resources.end());
  EXPECT_EQ(dummy_key(graph, uses[0]).sampleType, plan.slots[0].sampleType);
  EXPECT_EQ(dummy_key(graph, uses[1]).storageFormat, plan.slots[1].storageFormat);
}

TEST(GraphLayoutTest, LayoutHashStable) {
  const auto passes = declaration_order_passes();
  auto reversed = passes;
  std::ranges::reverse(reversed);
  const auto a = compile(passes);
  const auto b = compile(reversed);
  ASSERT_NE(a->find_node("t.pass"), b->find_node("t.pass"));
  EXPECT_EQ(plan_of(*a, "t.pass").layoutHash, plan_of(*b, "t.pass").layoutHash);
  EXPECT_NE(plan_of(*a, "t.pass").layoutHash, 0u);

  const auto changed = declaration_order_passes(0, Access::Uniform);
  const auto c = compile(changed);
  ASSERT_EQ(node_of(*c, "t.pass").status, CullReason::None);
  EXPECT_EQ(plan_of(*c, "t.pass").slots.at(0).type, BindingType::UniformBuffer);
  EXPECT_NE(plan_of(*c, "t.pass").layoutHash, plan_of(*a, "t.pass").layoutHash);
}

TEST(GraphLayoutTest, FrameGroupCostMatchesLayout) {
  BindingCounts counts;
  uint32_t dynamic = 0;
  const auto entries = frame_layout_entries();
  ASSERT_EQ(entries.size(), 6u);
  for (uint32_t i = 0; i < entries.size(); ++i) {
    const wgpu::BindGroupLayoutEntry& entry = entries[i];
    EXPECT_EQ(entry.binding, i);
    EXPECT_EQ(entry.visibility, ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute);
    if (entry.sampler.type != wgpu::SamplerBindingType::BindingNotUsed) {
      ++counts.samplers;
    }
    if (entry.buffer.type == wgpu::BufferBindingType::Uniform) {
      ++counts.uniformBuffers;
      dynamic += entry.buffer.hasDynamicOffset ? 1 : 0;
    }
  }
  EXPECT_EQ(counts, FrameGroupCost);
  EXPECT_EQ(dynamic, FrameGroupDynamicUniformBuffers);
  EXPECT_EQ(entries[static_cast<size_t>(FrameBinding::NearestClamp)].sampler.type,
            wgpu::SamplerBindingType::NonFiltering);
  EXPECT_EQ(entries[static_cast<size_t>(FrameBinding::DepthCompare)].sampler.type,
            wgpu::SamplerBindingType::Comparison);
  EXPECT_EQ(MaxPassParamsSize, 128u);
  EXPECT_EQ(sizeof(FrameUniforms), FrameUniformsSize);
}

TEST(GraphLayoutTest, CulledPassesKeepNoPlan) {
  std::vector<TestPass> passes = declaration_order_passes();
  passes.push_back({.name = "t.unused", .order = 2, .body = [](InternalPassBuilder& b) {
                      b.create("t.lonely", absolute(TextureFormat::RGBA8Unorm), Access::StorageWrite);
                    }});
  const auto graph = compile(passes);
  EXPECT_EQ(graph->bindingPlans.size(), graph->nodes.size());
  EXPECT_EQ(node_of(*graph, "t.unused").status, CullReason::NoConsumer);
  EXPECT_TRUE(plan_of(*graph, "t.unused").slots.empty());
  EXPECT_EQ(plan_of(*graph, "t.unused").layoutHash, 0u);
}

TEST(GraphLayoutTest, OversizedParamsRejectedAtRegistration) {
  testing::clear_passes();
  const PassEncodeFn encode = [](PassContext&, const void*) {};
  EXPECT_EQ(register_pass({.name = "p.big", .paramsSize = MaxPassParamsSize + 1, .encode = encode}), InvalidPassId);
  EXPECT_NE(register_pass({.name = "p.fits", .paramsSize = MaxPassParamsSize, .encode = encode}), InvalidPassId);
  testing::clear_passes();
}

TEST(GraphLayoutTest, FarPyramidBindsUnfilterable) {
  const std::vector<TestPass> passes{
      {.name = "t.build",
       .body =
           [](InternalPassBuilder& b) {
             b.after(phase::Geometry);
             b.create(hzb::Resource, hzb::pyramid_desc(), Access::StorageWrite);
             b.create("t.r32", {.format = TextureFormat::R32Float, .sizeClass = SizeClass::Efb}, Access::StorageWrite);
           }},
      {.name = "t.read",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after(phase::Geometry);
             b.use(hzb::Resource, Access::Sampled, UseOptions{.missing = Missing::BindDummyBlack});
             b.use("t.r32", Access::Sampled, UseOptions{.missing = Missing::BindDummyBlack});
             b.side_effect();
           }},
  };
  GraphInputs inputs = test_inputs();
  inputs.env.limits.features.float32Filterable = true;
  const auto graph = compile(passes, inputs);
  ASSERT_EQ(node_of(*graph, "t.read").status, CullReason::None);
  const BindingPlan& plan = plan_of(*graph, "t.read");
  ASSERT_EQ(plan.slots.size(), 2u);
  EXPECT_EQ(plan.slots[0].sampleType, TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(plan.slots[0].viewDimension, TextureViewDimension::e2D);
  EXPECT_EQ(plan.slots[1].sampleType, TextureSampleType::Float);
  const auto uses = graph->node_uses(graph->find_node("t.read"));
  EXPECT_EQ(dummy_key(*graph, uses[0], inputs.env.limits.features).sampleType, TextureSampleType::UnfilterableFloat);
  EXPECT_EQ(dummy_key(*graph, uses[1], inputs.env.limits.features).sampleType, TextureSampleType::Float);
}
} // namespace
} // namespace aurora::render::detail

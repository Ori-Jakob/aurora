#include <gtest/gtest.h>

#include "logging.hpp"
#include "render/graph.hpp"
#include "render_graph_planned.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
namespace {
using planned::compat_limits;
using planned::full_limits;

struct LogLine {
  AuroraLogLevel level;
  std::string module;
  std::string message;
};

std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  g_logLines.push_back({level, module != nullptr ? module : "", std::string{message, len}});
}

size_t count_logs(AuroraLogLevel level, std::string_view needle = {}) {
  return static_cast<size_t>(std::ranges::count_if(g_logLines, [&](const LogLine& line) {
    return line.level == level && line.message.find(needle) != std::string::npos;
  }));
}

struct TestPass {
  std::string name;
  Stage stage = Stage::OpaqueDone;
  PassKind kind = PassKind::Compute;
  PassClass cls = PassClass::Required;
  uint16_t order = 0;
  std::function<void(InternalPassBuilder&)> body;
  bool sink = true;
};

void declare_test_pass(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  const auto& pass = *static_cast<const TestPass*>(state);
  if (pass.body) {
    pass.body(builder);
  }
  if (pass.sink) {
    builder.side_effect();
  }
}

GraphInputs test_inputs() {
  GraphInputs inputs;
  inputs.env.limits = full_limits();
  inputs.env.efbSize = {1280, 720};
  inputs.env.outputSize = {1280, 720};
  inputs.env.gxFormats = planned::gx_formats_tier_p();
  inputs.env.gxUsage = planned::gx_usage_future();
  inputs.env.inspectorOpen = true;
  return inputs;
}

std::vector<PassDesc> pass_descs(const std::vector<TestPass>& passes) {
  std::vector<PassDesc> descs;
  for (const TestPass& pass : passes) {
    descs.push_back({
        .name = pass.name,
        .stage = pass.stage,
        .kind = pass.kind,
        .cls = pass.cls,
        .order = pass.order,
        .declare = declare_test_pass,
        .state = &pass,
    });
  }
  return descs;
}

std::shared_ptr<CompiledGraph> compile(const std::vector<TestPass>& passes, GraphInputs base = test_inputs()) {
  const std::vector<PassDesc> descs = pass_descs(passes);
  base.passes = descs;
  return compile_graph(base);
}

const CompiledNode& node(const CompiledGraph& graph, std::string_view name) {
  static const CompiledNode absent{};
  const NodeId id = graph.find_node(name);
  EXPECT_NE(id, InvalidNode) << name;
  return id != InvalidNode ? graph.nodes[id] : absent;
}

const CompiledResource& resource(const CompiledGraph& graph, std::string_view name) {
  static const CompiledResource absent{};
  const ResourceId id = graph.find_resource(name);
  EXPECT_NE(id, InvalidResource) << name;
  return id != InvalidResource ? graph.resources[id] : absent;
}

const CompiledUse* find_use(const CompiledGraph& graph, std::string_view nodeName, std::string_view resourceName) {
  const NodeId id = graph.find_node(nodeName);
  if (id == InvalidNode) {
    return nullptr;
  }
  for (const CompiledUse& use : graph.node_uses(id)) {
    if (graph.resources[use.resource].name == resourceName) {
      return &use;
    }
  }
  return nullptr;
}

NodeId producer(const CompiledGraph& graph, std::string_view nodeName, std::string_view resourceName) {
  const CompiledUse* use = find_use(graph, nodeName, resourceName);
  EXPECT_NE(use, nullptr) << nodeName << " " << resourceName;
  return use != nullptr ? use->producer : InvalidNode;
}

std::string node_name(const CompiledGraph& graph, NodeId id) {
  return id != InvalidNode ? graph.nodes[id].name : std::string{"<none>"};
}

std::vector<std::string> names(const CompiledGraph& graph, std::span<const NodeId> ids) {
  std::vector<std::string> out;
  for (const NodeId id : ids) {
    out.push_back(graph.nodes[id].name);
  }
  return out;
}

std::vector<std::string> stage_nodes(const CompiledGraph& graph, Stage stage) {
  return names(graph, graph.stages[static_cast<size_t>(stage)].nodes);
}

std::vector<std::string> stage_passes(const CompiledGraph& graph, Stage stage) {
  return names(graph, graph.stages[static_cast<size_t>(stage)].passes);
}

std::vector<std::string> chain_names(Stage stage) {
  std::vector<std::string> out;
  for (const ChainNode& chain : stage_chain(stage)) {
    out.emplace_back(chain.name);
  }
  return out;
}

std::vector<std::string> insert_after(std::vector<std::string> list, std::string_view anchor, std::string name) {
  const auto it = std::ranges::find(list, anchor);
  EXPECT_NE(it, list.end()) << anchor;
  list.insert(it + 1, std::move(name));
  return list;
}

size_t position(const std::vector<std::string>& list, std::string_view name) {
  const auto it = std::ranges::find(list, name);
  EXPECT_NE(it, list.end()) << name;
  return static_cast<size_t>(it - list.begin());
}

bool has_edge(const CompiledGraph& graph, std::string_view from, std::string_view to, EdgeKind kind) {
  const NodeId a = graph.find_node(from);
  const NodeId b = graph.find_node(to);
  return std::ranges::any_of(graph.edges,
                             [&](const Edge& edge) { return edge.from == a && edge.to == b && edge.kind == kind; });
}

bool has_any_edge(const CompiledGraph& graph, std::string_view name) {
  const NodeId id = graph.find_node(name);
  return std::ranges::any_of(graph.edges, [&](const Edge& edge) { return edge.from == id || edge.to == id; });
}

std::vector<Diagnostic> declaration_diagnostics(const CompiledGraph& graph) {
  std::vector<Diagnostic> out;
  for (const Diagnostic& diagnostic : graph.diagnostics) {
    switch (diagnostic.reason) {
    case CullReason::MissingInput:
    case CullReason::MissingTarget:
    case CullReason::NoConsumer:
    case CullReason::InspectorClosed:
    case CullReason::CoercedMissing:
      break;
    default:
      out.push_back(diagnostic);
    }
  }
  return out;
}

std::string first_message(const CompiledGraph& graph) {
  const auto diagnostics = declaration_diagnostics(graph);
  return diagnostics.empty() ? std::string{} : diagnostics.front().message;
}

bool active(const CompiledGraph& graph, std::string_view name) { return node(graph, name).status == CullReason::None; }

TexDesc efb_tex(wgpu::TextureFormat format, uint16_t divisor = 1) {
  return {.format = format, .sizeClass = SizeClass::Efb, .divisor = divisor};
}

TexDesc absolute_tex(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

std::function<void(InternalPassBuilder&)> storage_buffers(std::string_view prefix, uint32_t count,
                                                          Access access = Access::StorageRead) {
  return [prefix = std::string{prefix}, count, access](InternalPassBuilder& b) {
    for (uint32_t i = 0; i < count; ++i) {
      b.import_persistent(prefix + std::to_string(i), {100 + i}, access);
    }
  };
}

class RenderGraph : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset_logged_diagnostics();
    g_logLines.clear();
    g_config.logCallback = capture_log;
  }

  void TearDown() override { g_config.logCallback = nullptr; }
};

TEST_F(RenderGraph, PhasesAreFrozen) {
  struct Row {
    Stage stage;
    ChainKind kind;
    std::string_view name;
  };
  using enum Stage;
  constexpr auto Gx = ChainKind::Gx;
  constexpr auto Phase = ChainKind::Phase;
  const std::vector<Row> expected{
      {SceneBegin, Gx, "gx@SceneBegin"},
      {SceneBegin, Phase, "phase.scene_prep"},
      {OpaqueDone, Gx, "gx@OpaqueDone"},
      {OpaqueDone, Phase, "phase.capture"},
      {OpaqueDone, Phase, "phase.geometry"},
      {OpaqueDone, Phase, "phase.prepare"},
      {OpaqueDone, Phase, "phase.ao"},
      {OpaqueDone, Phase, "phase.cloud_shadows"},
      {OpaqueDone, Phase, "phase.shadows"},
      {OpaqueDone, Phase, "phase.gi"},
      {OpaqueDone, Phase, "phase.lighting"},
      {OpaqueDone, Phase, "phase.reflections"},
      {OpaqueDone, Phase, "phase.exposure"},
      {OpaqueDone, Phase, "phase.sky"},
      {OpaqueDone, Phase, "phase.volumetrics"},
      {OpaqueDone, Phase, "phase.composite"},
      {OpaqueDone, ChainKind::Overlay, "gx.overlay"},
      {PrePost, Gx, "gx@PrePost"},
      {PrePost, Phase, "phase.pre_taa"},
      {PrePost, Phase, "phase.taa"},
      {PrePost, Phase, "phase.post_taa"},
      {SceneEnd, Gx, "gx@SceneEnd"},
      {SceneEnd, Phase, "phase.scene_end"},
      {BeforeHud, Gx, "gx@BeforeHud"},
      {BeforeHud, Phase, "phase.before_hud"},
      {FrameEnd, Gx, "gx@FrameEnd"},
      {FrameEnd, Phase, "phase.frame_end"},
      {AfterFrame, Phase, "phase.after_frame"},
  };
  const auto nodes = chain_nodes();
  ASSERT_EQ(nodes.size(), expected.size());
  EXPECT_EQ(ChainNodeCount, 28u);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(nodes[i].stage, expected[i].stage) << i;
    EXPECT_EQ(nodes[i].kind, expected[i].kind) << i;
    EXPECT_EQ(nodes[i].name, expected[i].name) << i;
  }
  EXPECT_EQ(stage_chain(OpaqueDone).size(), 15u);
  EXPECT_EQ(chain_node_stage("phase.taa"), PrePost);
  EXPECT_FALSE(chain_node_stage("phase.fog").has_value());
}

TEST_F(RenderGraph, GxResourcesAreFrozen) {
  constexpr std::array<std::string_view, GxResourceCount> Expected{
      "gx.color", "gx.depth", "gx.normal", "gbuf.motion", "gbuf.reactive", "gbuf.fog", "gbuf.surface", "gbuf.debug_id",
  };
  for (size_t i = 0; i < Expected.size(); ++i) {
    EXPECT_EQ(gx_resource_name(static_cast<GxResource>(i)), Expected[i]);
    EXPECT_EQ(find_gx_resource(Expected[i]), static_cast<GxResource>(i));
  }
  EXPECT_EQ(find_gx_resource("gbuf.fog"), GxResource::Fog);
  EXPECT_EQ(gx_fixed_format(GxResource::Surface), wgpu::TextureFormat::RGBA32Uint);
  EXPECT_EQ(gx_fixed_format(GxResource::Color), wgpu::TextureFormat::Undefined);
}

template <typename E, typename F>
void expect_unique_names(size_t count, F name) {
  std::set<std::string_view> seen;
  for (size_t i = 0; i < count; ++i) {
    const std::string_view text = name(static_cast<E>(i));
    EXPECT_FALSE(text.empty()) << i;
    EXPECT_NE(text, "Unknown") << i;
    EXPECT_TRUE(seen.insert(text).second) << text;
  }
  EXPECT_EQ(name(static_cast<E>(count)), "Unknown");
}

TEST_F(RenderGraph, NameHelpers) {
  expect_unique_names<GxResource>(GxResourceCount, gx_resource_name);
  expect_unique_names<Access>(static_cast<size_t>(Access::Indirect) + 1, access_name);
  expect_unique_names<Missing>(static_cast<size_t>(Missing::BindDummyWhite) + 1, missing_name);
  expect_unique_names<PassKind>(static_cast<size_t>(PassKind::Native) + 1, pass_kind_name);
  expect_unique_names<PassClass>(static_cast<size_t>(PassClass::Diagnostic) + 1, pass_class_name);
  expect_unique_names<ResourceKind>(static_cast<size_t>(ResourceKind::External) + 1, resource_kind_name);
  expect_unique_names<CullReason>(static_cast<size_t>(CullReason::BindingViewMismatch) + 1, cull_reason_name);
  expect_unique_names<EdgeKind>(static_cast<size_t>(EdgeKind::Waw) + 1, edge_kind_name);
}

TEST_F(RenderGraph, EmptyGraph) {
  const auto& empty = *empty_graph();
  EXPECT_TRUE(empty.nodes.empty());
  EXPECT_EQ(empty.graphHash, 0u);
  for (size_t s = 0; s < StageCount; ++s) {
    EXPECT_TRUE(empty.stage_empty(static_cast<Stage>(s)));
  }

  const auto graph = compile_graph(GraphInputs{});
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->nodes.size(), ChainNodeCount);
  for (size_t s = 0; s < StageCount; ++s) {
    EXPECT_TRUE(graph->stage_empty(static_cast<Stage>(s)));
    EXPECT_EQ(stage_nodes(*graph, static_cast<Stage>(s)), chain_names(static_cast<Stage>(s)));
  }
  EXPECT_TRUE(graph->diagnostics.empty());
}

TEST_F(RenderGraph, PhaseSegmentsHoldPasses) {
  const auto expected =
      insert_after(insert_after(chain_names(Stage::OpaqueDone), "phase.geometry", "p"), "phase.prepare", "q");
  const auto p = [](uint16_t order) {
    return TestPass{.name = "p", .order = order, .body = [](InternalPassBuilder& b) { b.after("phase.geometry"); }};
  };
  const auto q = [](uint16_t order) {
    return TestPass{.name = "q", .order = order, .body = [](InternalPassBuilder& b) { b.after("phase.prepare"); }};
  };
  EXPECT_EQ(stage_nodes(*compile({p(10), q(0)}), Stage::OpaqueDone), expected);
  EXPECT_EQ(stage_nodes(*compile({p(0), q(10)}), Stage::OpaqueDone), expected);
  EXPECT_EQ(stage_nodes(*compile({q(0), p(10)}), Stage::OpaqueDone), expected);

  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  const auto graph = compile_graph(planned::planned_inputs(full_limits(), passes, reads));
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_GT(position(nodes, "vg.cull"), position(nodes, "phase.geometry"));
  EXPECT_LT(position(nodes, "vg.cull"), position(nodes, "phase.prepare"));
}

std::vector<TestPass> creator_and_reader(uint16_t readerOrder, uint16_t creatorOrder) {
  return {
      {.name = "b",
       .order = readerOrder,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("x", Access::Sampled);
           }},
      {.name = "a",
       .order = creatorOrder,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.create("x", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::StorageWrite);
           }},
  };
}

TEST_F(RenderGraph, RawOrdersConsumerAfterCreator) {
  const auto graph = compile(creator_and_reader(0, 0));
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(has_edge(*graph, "a", "b", EdgeKind::Raw));
  EXPECT_EQ(producer(*graph, "b", "x"), graph->find_node("a"));
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
}

TEST_F(RenderGraph, CreatorFirstIgnoresOrderField) {
  const auto graph = compile(creator_and_reader(0, 5));
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(has_edge(*graph, "a", "b", EdgeKind::Raw));
}

std::vector<TestPass> persistent_reader_writer(uint16_t readerOrder, uint16_t writerOrder) {
  return {
      {.name = "r",
       .order = readerOrder,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.import_persistent("p", {1}, Access::StorageRead);
           }},
      {.name = "w",
       .order = writerOrder,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.import_persistent("p", {1}, Access::StorageWrite);
           }},
  };
}

TEST_F(RenderGraph, WarOrdersReaderBeforeWriter) {
  const auto graph = compile(persistent_reader_writer(0, 1));
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"r", "w"}));
  EXPECT_TRUE(has_edge(*graph, "r", "w", EdgeKind::War));
  EXPECT_EQ(producer(*graph, "r", "p"), InvalidNode);
  EXPECT_FALSE(find_use(*graph, "r", "p")->unresolved);
}

TEST_F(RenderGraph, RawAfterNonCreatorWrite) {
  const auto graph = compile(persistent_reader_writer(1, 0));
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"w", "r"}));
  EXPECT_TRUE(has_edge(*graph, "w", "r", EdgeKind::Raw));
  EXPECT_EQ(producer(*graph, "r", "p"), graph->find_node("w"));
}

// A write into an existing version also reads it, so two in-place writers are ordered by a Raw edge; only a pure
// write after another writer, such as gx.overlay's, gets a Waw edge
TEST_F(RenderGraph, WawOrdersWriters) {
  const auto writer = [](std::string name, uint16_t order) {
    return TestPass{.name = std::move(name), .order = order, .body = [](InternalPassBuilder& b) {
                      b.after("phase.lighting");
                      b.import_persistent("p", {1}, Access::StorageWrite);
                    }};
  };
  auto graph = compile({writer("w1", 0), writer("w2", 1)});
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"w1", "w2"}));
  EXPECT_TRUE(has_edge(*graph, "w1", "w2", EdgeKind::Raw));
  EXPECT_EQ(producer(*graph, "w2", "p"), graph->find_node("w1"));

  graph = compile({writer("w2", 0), writer("w1", 0)});
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"w2", "w1"}));
  EXPECT_TRUE(has_edge(*graph, "w2", "w1", EdgeKind::Raw));

  graph = compile({{.name = "c", .kind = PassKind::Render, .body = [](InternalPassBuilder& b) {
                      b.after("phase.composite");
                      b.use("gx.color", Access::ColorRMW);
                    }}});
  EXPECT_TRUE(has_edge(*graph, "c", "gx.overlay", EdgeKind::Waw));
  EXPECT_EQ(producer(*graph, "c", "gx.color"), graph->find_node("gx@OpaqueDone"));
}

TEST_F(RenderGraph, GxWritesStartEveryStage) {
  const auto graph = compile({
      {.name = "opaque",
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("gx.depth", Access::Sampled);
           }},
      {.name = "post",
       .stage = Stage::PrePost,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.taa");
             b.use("gx.depth", Access::Sampled);
           }},
      {.name = "late", .body = [](InternalPassBuilder& b) { b.use("gx.depth", Access::Sampled); }},
  });
  EXPECT_EQ(producer(*graph, "opaque", "gx.depth"), graph->find_node("gx@OpaqueDone"));
  EXPECT_EQ(producer(*graph, "post", "gx.depth"), graph->find_node("gx@PrePost"));
  EXPECT_EQ(producer(*graph, "late", "gx.depth"), graph->find_node("gx.overlay"));
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
}

TEST_F(RenderGraph, GxWriteIsPureWrite) {
  const auto graph =
      compile({{.name = "reader", .body = [](InternalPassBuilder& b) { b.use("gx.depth", Access::Sampled); }}});
  const auto uses = graph->node_uses(graph->find_node("gx@OpaqueDone"));
  EXPECT_EQ(uses.size(), GxResourceCount);
  for (const CompiledUse& use : uses) {
    EXPECT_EQ(use.role, UseRole::GxWrite);
    EXPECT_EQ(use.producer, InvalidNode);
    EXPECT_FALSE(use.unresolved);
  }
}

TEST_F(RenderGraph, GxPseudoIsAlwaysFirst) {
  const auto graph = compile({{.name = "early", .body = [](InternalPassBuilder& b) { b.before("phase.capture"); }}});
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  ASSERT_GE(nodes.size(), 2u);
  EXPECT_EQ(nodes[0], "gx@OpaqueDone");
  EXPECT_EQ(nodes[1], "early");
}

std::vector<TestPass> cycle_passes(PassClass readerClass) {
  return {
      {.name = "cycle.a",
       .cls = readerClass,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("y", Access::Sampled, Missing::BindDummyBlack);
           }},
      {.name = "cycle.b",
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.composite");
             b.create("y", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::StorageWrite);
           }},
  };
}

TEST_F(RenderGraph, CycleNamesBothPasses) {
  const auto graph = compile(cycle_passes(PassClass::Required));
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  const Diagnostic& cycle = diagnostics[0];
  EXPECT_EQ(cycle.reason, CullReason::Cycle);
  EXPECT_EQ(node_name(*graph, cycle.node), "cycle.b");
  EXPECT_EQ(node_name(*graph, cycle.other), "cycle.a");
  EXPECT_EQ(cycle.resource, graph->find_resource("y"));
  EXPECT_NE(cycle.message.find("'cycle.a'"), std::string::npos) << cycle.message;
  EXPECT_NE(cycle.message.find("'cycle.b'"), std::string::npos) << cycle.message;
  EXPECT_NE(cycle.message.find("Raw"), std::string::npos) << cycle.message;
  EXPECT_EQ(node(*graph, "cycle.b").status, CullReason::Cycle);
  EXPECT_TRUE(active(*graph, "cycle.a"));
  EXPECT_EQ(count_logs(LOG_ERROR), 1u);
  EXPECT_EQ(count_logs(LOG_ERROR, "form a cycle"), 1u);
}

TEST_F(RenderGraph, AnchorCycleNamesBothPasses) {
  const auto graph = compile({
      {.name = "loop.a", .body = [](InternalPassBuilder& b) { b.after("loop.b"); }},
      {.name = "loop.b", .body = [](InternalPassBuilder& b) { b.after("loop.a"); }},
  });
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].reason, CullReason::Cycle);
  const std::string& message = diagnostics[0].message;
  EXPECT_NE(message.find("'loop.a'"), std::string::npos) << message;
  EXPECT_NE(message.find("'loop.b'"), std::string::npos) << message;
  EXPECT_NE(message.find("anchor"), std::string::npos) << message;
  EXPECT_EQ(node(*graph, "loop.b").status, CullReason::Cycle);
  EXPECT_TRUE(active(*graph, "loop.a"));
}

TEST_F(RenderGraph, CycleVictimPrefersOptional) {
  const auto graph = compile(cycle_passes(PassClass::Optional));
  EXPECT_EQ(node(*graph, "cycle.a").status, CullReason::Cycle);
  EXPECT_TRUE(active(*graph, "cycle.b"));
}

TEST_F(RenderGraph, CycleRecovers) {
  const auto graph = compile(cycle_passes(PassClass::Required));
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_EQ(nodes, insert_after(chain_names(Stage::OpaqueDone), "phase.lighting", "cycle.a"));
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].reason, CullReason::Cycle);
}

TEST_F(RenderGraph, BackwardsEdgeDisablesReader) {
  const auto graph = compile({
      {.name = "early", .body = [](InternalPassBuilder& b) { b.use("t", Access::Sampled); }},
      {.name = "late",
       .stage = Stage::PrePost,
       .body =
           [](InternalPassBuilder& b) {
             b.create("t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
           }},
  });
  EXPECT_EQ(node(*graph, "early").status, CullReason::BackwardsEdge);
  EXPECT_TRUE(active(*graph, "late"));
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(node_name(*graph, diagnostics[0].other), "late");
  EXPECT_NE(diagnostics[0].message.find("created later by 'late'"), std::string::npos) << diagnostics[0].message;
}

TEST_F(RenderGraph, PrevReadIsNotBackwards) {
  const auto graph = compile({
      {.name = "reader", .body = [](InternalPassBuilder& b) { b.use_prev("taa.color"); }},
      {.name = "owner",
       .stage = Stage::PrePost,
       .body =
           [](InternalPassBuilder& b) {
             b.history("taa.color", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::StorageWrite);
           }},
  });
  EXPECT_TRUE(active(*graph, "reader"));
  EXPECT_TRUE(active(*graph, "owner"));
  EXPECT_FALSE(has_edge(*graph, "owner", "reader", EdgeKind::Raw));
  EXPECT_EQ(producer(*graph, "reader", "taa.color"), InvalidNode);
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
}

TEST_F(RenderGraph, OwnerSwitchIsNotABackwardsEdge) {
  const auto graph = compile({
      {.name = "native.write",
       .stage = Stage::AfterFrame,
       .kind = PassKind::Native,
       .body =
           [](InternalPassBuilder& b) {
             b.import_persistent("rt.native.hits", {26}, Access::StorageWrite, Owner::NativeD3D12);
           }},
      {.name = "reader", .body = [](InternalPassBuilder& b) { b.use_buffer("rt.native.hits", Access::StorageRead); }},
  });
  EXPECT_TRUE(active(*graph, "native.write"));
  EXPECT_TRUE(active(*graph, "reader"));
  EXPECT_EQ(producer(*graph, "reader", "rt.native.hits"), InvalidNode);
  EXPECT_EQ(stage_passes(*graph, Stage::AfterFrame), (std::vector<std::string>{"native.write"}));
  EXPECT_TRUE(graph->hasNativePasses);
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
}

TEST_F(RenderGraph, NativeOnlyInAfterFrame) {
  const auto graph = compile({
      {.name = "native.early", .kind = PassKind::Native},
      {.name = "compute.late", .stage = Stage::AfterFrame},
      {.name = "maker",
       .body = [](InternalPassBuilder& b) { b.create_buffer("t", {.size = 256}, Access::StorageWrite); }},
      {.name = "native.transient",
       .stage = Stage::AfterFrame,
       .kind = PassKind::Native,
       .body = [](InternalPassBuilder& b) { b.use_buffer("t", Access::StorageRead); }},
  });
  EXPECT_EQ(node(*graph, "native.early").status, CullReason::WrongStage);
  EXPECT_EQ(node(*graph, "compute.late").status, CullReason::WrongStage);
  EXPECT_EQ(node(*graph, "native.transient").status, CullReason::InvalidDeclaration);
  EXPECT_TRUE(active(*graph, "maker"));
  EXPECT_FALSE(graph->hasNativePasses);
}

struct DeclarationCase {
  std::string label;
  std::vector<TestPass> passes;
  std::string pass;
  CullReason reason;
};

TEST_F(RenderGraph, DeclarationErrors) {
  using wgpu::TextureFormat;
  const auto color = efb_tex(TextureFormat::RGBA8Unorm);
  std::vector<DeclarationCase> cases;
  cases.push_back({"duplicate name", {{.name = "dup"}, {.name = "dup", .order = 1}}, "dup", CullReason::DuplicateName});
  cases.push_back(
      {"duplicate creator",
       {{.name = "first", .body = [=](InternalPassBuilder& b) { b.create("x", color, Access::StorageWrite); }},
        {.name = "second", .body = [=](InternalPassBuilder& b) { b.create("x", color, Access::StorageWrite); }}},
       "second",
       CullReason::DuplicateCreator});
  cases.push_back(
      {"create gx",
       {{.name = "p", .body = [=](InternalPassBuilder& b) { b.create("gx.color", color, Access::StorageWrite); }}},
       "p",
       CullReason::InvalidDeclaration});
  cases.push_back({"colour write in compute",
                   {{.name = "p", .body = [=](InternalPassBuilder& b) { b.create("x", color, Access::ColorWrite); }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"colour rmw create",
                   {{.name = "p",
                     .kind = PassKind::Render,
                     .body = [=](InternalPassBuilder& b) { b.create("x", color, Access::ColorRMW); }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"same resource twice",
                   {{.name = "p",
                     .body =
                         [=](InternalPassBuilder& b) {
                           b.create("x", color, Access::StorageWrite);
                           b.use("x", Access::Sampled);
                         }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"unknown gbuf",
                   {{.name = "p", .body = [](InternalPassBuilder& b) { b.use("gbuf.unknown", Access::Sampled); }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"render without attachments",
                   {{.name = "p",
                     .kind = PassKind::Render,
                     .body = [](InternalPassBuilder& b) { b.use("gx.depth", Access::Sampled); }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"attachment extents",
                   {{.name = "p",
                     .kind = PassKind::Render,
                     .body =
                         [](InternalPassBuilder& b) {
                           b.use("gx.color", Access::ColorRMW);
                           b.create("half", efb_tex(TextureFormat::RGBA8Unorm, 2), Access::ColorWrite);
                         }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"create and import",
                   {{.name = "maker",
                     .body = [](InternalPassBuilder& b) { b.create_buffer("p", {.size = 64}, Access::StorageWrite); }},
                    {.name = "importer",
                     .body = [](InternalPassBuilder& b) { b.import_persistent("p", {7}, Access::StorageRead); }}},
                   "importer",
                   CullReason::ConflictingKind});
  cases.push_back(
      {"import descriptions",
       {{.name = "one",
         .body =
             [](InternalPassBuilder& b) {
               b.import_persistent_texture("p", {7}, absolute_tex(TextureFormat::RGBA8Unorm, 64, 64), Access::Sampled);
             }},
        {.name = "two",
         .body =
             [](InternalPassBuilder& b) {
               b.import_persistent_texture("p", {7}, absolute_tex(TextureFormat::RGBA16Float, 64, 64), Access::Sampled);
             }}},
       "two",
       CullReason::ConflictingKind});
  cases.push_back({"absolute width 0",
                   {{.name = "p",
                     .body =
                         [](InternalPassBuilder& b) {
                           b.create("x", absolute_tex(TextureFormat::RGBA8Unorm, 0, 64), Access::StorageWrite);
                         }}},
                   "p",
                   CullReason::InvalidDeclaration});
  cases.push_back({"divisor 0",
                   {{.name = "p",
                     .body =
                         [](InternalPassBuilder& b) {
                           b.create("x", efb_tex(TextureFormat::RGBA8Unorm, 0), Access::StorageWrite);
                         }}},
                   "p",
                   CullReason::InvalidDeclaration});

  for (const DeclarationCase& test : cases) {
    SCOPED_TRACE(test.label);
    const auto graph = compile(test.passes);
    const auto diagnostics = declaration_diagnostics(*graph);
    ASSERT_EQ(diagnostics.size(), 1u);
    EXPECT_EQ(diagnostics[0].reason, test.reason) << diagnostics[0].message;
    const NodeId expected = test.reason == CullReason::DuplicateName ? NodeId{1} : graph->find_node(test.pass);
    EXPECT_EQ(diagnostics[0].node, expected) << diagnostics[0].message;
    EXPECT_EQ(graph->nodes[expected].status, test.reason);
    if (test.label == "import descriptions") {
      EXPECT_EQ(resource(*graph, "p").tex.format, TextureFormat::RGBA8Unorm);
      EXPECT_TRUE(active(*graph, "one"));
    }
  }
}

TEST_F(RenderGraph, SkipDiscardsDeclarations) {
  const auto graph =
      compile({{.name = "skipped", .body = [](InternalPassBuilder& b) {
                  b.create("skipped.tex", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
                  b.after("phase.nope");
                  b.skip();
                }}});
  EXPECT_EQ(graph->find_resource("skipped.tex"), InvalidResource);
  EXPECT_EQ(node(*graph, "skipped").status, CullReason::Skipped);
  EXPECT_EQ(node(*graph, "skipped").useCount, 0u);
  EXPECT_TRUE(graph->diagnostics.empty());
  EXPECT_TRUE(g_logLines.empty());
  EXPECT_TRUE(graph->stage_empty(Stage::OpaqueDone));
}

TEST_F(RenderGraph, OpaquePass) {
  GraphInputs inputs = test_inputs();
  const std::vector<PassDesc> descs{
      {.name = "late", .stage = Stage::OpaqueDone, .order = 1},
      {.name = "early", .stage = Stage::OpaqueDone, .order = 0},
      {.name = "tie", .stage = Stage::OpaqueDone, .order = 1},
  };
  inputs.passes = descs;
  const auto graph = compile_graph(inputs);
  const CompiledNode& early = node(*graph, "early");
  EXPECT_TRUE(early.opaque);
  EXPECT_TRUE(early.sideEffect);
  EXPECT_EQ(early.useCount, 0u);
  auto expected = chain_names(Stage::OpaqueDone);
  expected.insert(expected.end(), {"early", "late", "tie"});
  EXPECT_EQ(stage_nodes(*graph, Stage::OpaqueDone), expected);
  EXPECT_TRUE(graph->diagnostics.empty());
}

TEST_F(RenderGraph, UnknownAnchorFallsBackAndLogsOnce) {
  const std::vector<TestPass> passes{{.name = "lost", .body = [](InternalPassBuilder& b) { b.after("phase.nope"); }}};
  const auto graph = compile(passes);
  auto expected = chain_names(Stage::OpaqueDone);
  expected.emplace_back("lost");
  EXPECT_EQ(stage_nodes(*graph, Stage::OpaqueDone), expected);
  ASSERT_EQ(graph->diagnostics.size(), 1u);
  const Diagnostic& diagnostic = graph->diagnostics[0];
  EXPECT_EQ(diagnostic.reason, CullReason::UnknownAnchor);
  EXPECT_EQ(diagnostic.detail, "phase.nope");
  EXPECT_EQ(node(*graph, "lost").status, CullReason::None);
  ASSERT_EQ(g_logLines.size(), 1u);
  EXPECT_EQ(g_logLines[0].level, LOG_WARNING);
  EXPECT_EQ(g_logLines[0].module, "aurora::render::graph");
  EXPECT_NE(g_logLines[0].message.find("has unknown anchor"), std::string::npos);
  EXPECT_EQ(g_logLines[0].message, "render graph: 'lost' has unknown anchor 'phase.nope' in OpaqueDone; ignored");

  const auto again = compile(passes);
  EXPECT_EQ(again->diagnostics.size(), 1u);
  EXPECT_EQ(g_logLines.size(), 1u);
}

TEST_F(RenderGraph, AnchorToOtherStageFallsBack) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) { b.after("phase.taa"); }}});
  EXPECT_EQ(stage_nodes(*graph, Stage::OpaqueDone).back(), "p");
  ASSERT_EQ(graph->diagnostics.size(), 1u);
  EXPECT_EQ(graph->diagnostics[0].reason, CullReason::UnknownAnchor);
  EXPECT_EQ(graph->diagnostics[0].detail, "phase.taa");
}

TEST_F(RenderGraph, AnchorToDisabledPassFloats) {
  const auto graph = compile({
      {.name = "p", .body = storage_buffers("sb", 11)},
      {.name = "q", .body = [](InternalPassBuilder& b) { b.after("p"); }},
  });
  EXPECT_EQ(node(*graph, "p").status, CullReason::BindingLimit);
  EXPECT_TRUE(active(*graph, "q"));
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  ASSERT_GE(nodes.size(), 2u);
  EXPECT_EQ(nodes[0], "gx@OpaqueDone");
  EXPECT_EQ(nodes[1], "q");
  EXPECT_EQ(count_logs(LOG_WARNING, "unknown anchor"), 0u);
}

TEST_F(RenderGraph, AnchorConflict) {
  const auto graph = compile({
      {.name = "inverted",
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.before("phase.ao");
           }},
      {.name = "before.gx", .body = [](InternalPassBuilder& b) { b.before("gx@OpaqueDone"); }},
      {.name = "self", .body = [](InternalPassBuilder& b) { b.after("self"); }},
  });
  EXPECT_EQ(node(*graph, "inverted").status, CullReason::AnchorConflict);
  EXPECT_EQ(node(*graph, "before.gx").status, CullReason::AnchorConflict);
  EXPECT_EQ(node(*graph, "self").status, CullReason::AnchorConflict);
  EXPECT_TRUE(graph->stage_empty(Stage::OpaqueDone));
}

TEST_F(RenderGraph, PassAnchorBeatsOrder) {
  const auto graph = compile({
      {.name = "p", .order = 9, .body = [](InternalPassBuilder& b) { b.after("phase.ao"); }},
      {.name = "q", .order = 0, .body = [](InternalPassBuilder& b) { b.after("p"); }},
  });
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  const size_t ao = position(nodes, "phase.ao");
  ASSERT_LT(ao + 3, nodes.size());
  EXPECT_EQ(nodes[ao + 1], "p");
  EXPECT_EQ(nodes[ao + 2], "q");
  EXPECT_EQ(nodes[ao + 3], "phase.cloud_shadows");
}

TEST_F(RenderGraph, BeforePhaseBound) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) { b.before("phase.lighting"); }}});
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_GT(position(nodes, "p"), position(nodes, "gx@OpaqueDone"));
  EXPECT_LT(position(nodes, "p"), position(nodes, "phase.lighting"));
  EXPECT_TRUE(has_edge(*graph, "p", "phase.lighting", EdgeKind::Bound));
}

TEST_F(RenderGraph, LimitStorageBuffers) {
  GraphInputs inputs = test_inputs();
  inputs.env.limits.compute.storageBuffers = 8;
  auto graph = compile({{.name = "big", .body = storage_buffers("sb", 9)}}, inputs);
  EXPECT_EQ(node(*graph, "big").status, CullReason::BindingLimit);
  auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].message,
            "render graph: 'big' needs 9 storage buffers in the compute stage, limit 8; disabled");
  EXPECT_EQ(count_logs(LOG_WARNING, "needs 9 storage buffers"), 1u);

  graph = compile({{.name = "big", .body = storage_buffers("sb", 8)}}, inputs);
  EXPECT_TRUE(active(*graph, "big"));
  EXPECT_EQ(node(*graph, "big").bindings[2].storageBuffers, 8u);
}

TEST_F(RenderGraph, LimitCountsFrameGroup) {
  GraphInputs inputs = test_inputs();
  inputs.env.limits.compute.uniformBuffers = 12;
  const uint32_t fits = 12 - FrameGroupCost.uniformBuffers;
  auto graph = compile({{.name = "ub", .body = storage_buffers("ub", fits + 1, Access::Uniform)}}, inputs);
  EXPECT_EQ(node(*graph, "ub").status, CullReason::BindingLimit);

  graph = compile({{.name = "ub", .body = storage_buffers("ub", fits, Access::Uniform)}}, inputs);
  const CompiledNode& ub = node(*graph, "ub");
  EXPECT_EQ(ub.status, CullReason::None);
  EXPECT_EQ(ub.bindings[2].uniformBuffers, 12u);
  EXPECT_EQ(ub.bindings[2].samplers, FrameGroupCost.samplers);
  EXPECT_EQ(ub.bindings[0], BindingCounts{});
  EXPECT_EQ(ub.bindings[1], BindingCounts{});
}

TEST_F(RenderGraph, LimitFragmentStorageCompat) {
  GraphInputs inputs = test_inputs();
  inputs.env.limits = compat_limits();
  const auto importer = [](uint32_t buffers, bool texture) {
    return TestPass{.name = "importer", .body = [=](InternalPassBuilder& b) {
                      storage_buffers("vb", buffers)(b);
                      if (texture) {
                        b.import_persistent_texture("vt", {50}, absolute_tex(wgpu::TextureFormat::R32Float, 64, 64),
                                                    Access::StorageRead);
                      }
                    }};
  };
  const auto vertex_reader = [](uint32_t buffers, bool texture) {
    return TestPass{.name = "draw", .kind = PassKind::Render, .order = 1, .body = [=](InternalPassBuilder& b) {
                      b.use("gx.color", Access::ColorRMW);
                      const UseOptions options{.visibility = wgpu::ShaderStage::Vertex};
                      for (uint32_t i = 0; i < buffers; ++i) {
                        b.use_buffer("vb" + std::to_string(i), Access::StorageRead, options);
                      }
                      if (texture) {
                        b.use("vt", Access::StorageRead, options);
                      }
                    }};
  };
  const TestPass fragment{.name = "draw", .kind = PassKind::Render, .body = [](InternalPassBuilder& b) {
                            b.use("gx.color", Access::ColorRMW);
                            b.import_persistent("fb", {60}, Access::StorageRead);
                          }};

  auto graph = compile({fragment}, inputs);
  EXPECT_TRUE(active(*graph, "draw"));

  GraphInputs noFragment = inputs;
  noFragment.env.limits.fragment.storageBuffers = 0;
  graph = compile({fragment}, noFragment);
  EXPECT_EQ(node(*graph, "draw").status, CullReason::BindingLimit);
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  const std::string& message = diagnostics[0].message;
  EXPECT_NE(message.find("needs 1"), std::string::npos) << message;
  EXPECT_NE(message.find("limit 0"), std::string::npos) << message;

  graph = compile({importer(3, false), vertex_reader(3, false)}, inputs);
  EXPECT_EQ(node(*graph, "draw").status, CullReason::BindingLimit);
  EXPECT_NE(first_message(*graph).find("vertex stage, limit 2"), std::string::npos);
  graph = compile({importer(2, false), vertex_reader(2, false)}, inputs);
  EXPECT_TRUE(active(*graph, "draw"));
  graph = compile({importer(0, true), vertex_reader(0, true)}, inputs);
  EXPECT_EQ(node(*graph, "draw").status, CullReason::BindingLimit);
  EXPECT_TRUE(active(*graph, "importer"));
}

TEST_F(RenderGraph, LimitSampledTextures) {
  GraphInputs inputs = test_inputs();
  inputs.env.limits.fragment.sampledTextures = 16;
  const auto sampler = [](uint32_t count) {
    return TestPass{.name = "sampler", .kind = PassKind::Render, .body = [count](InternalPassBuilder& b) {
                      b.use("gx.color", Access::ColorRMW);
                      for (uint32_t i = 0; i < count; ++i) {
                        b.import_persistent_texture("t" + std::to_string(i), {200 + i},
                                                    absolute_tex(wgpu::TextureFormat::RGBA8Unorm, 16, 16),
                                                    Access::Sampled);
                      }
                    }};
  };
  EXPECT_EQ(node(*compile({sampler(17)}, inputs), "sampler").status, CullReason::BindingLimit);
  const auto graph = compile({sampler(16)}, inputs);
  EXPECT_TRUE(active(*graph, "sampler"));
  EXPECT_EQ(node(*graph, "sampler").bindings[1].sampledTextures, 16u);
}

TEST_F(RenderGraph, LimitColorBytes) {
  const auto targets = [](bool wide) {
    return TestPass{.name = "mrt", .kind = PassKind::Render, .body = [wide](InternalPassBuilder& b) {
                      b.use("gx.color", Access::ColorRMW);
                      b.create("c0", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::ColorWrite);
                      b.create("c1", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::ColorWrite);
                      if (wide) {
                        b.create("c2", efb_tex(wgpu::TextureFormat::RGBA32Uint), Access::ColorWrite);
                      }
                    }};
  };
  GraphInputs inputs = test_inputs();
  inputs.env.limits.colorAttachmentBytesPerSample = 32;
  auto graph = compile({targets(false)}, inputs);
  EXPECT_TRUE(active(*graph, "mrt"));
  EXPECT_EQ(node(*graph, "mrt").colorBytes, 24u);
  graph = compile({targets(true)}, inputs);
  EXPECT_EQ(node(*graph, "mrt").status, CullReason::BindingLimit);
  EXPECT_EQ(node(*graph, "mrt").colorBytes, 40u);
  inputs.env.limits.colorAttachmentBytesPerSample = 64;
  EXPECT_TRUE(active(*compile({targets(true)}, inputs), "mrt"));

  const std::array formats{wgpu::TextureFormat::BGRA8Unorm, wgpu::TextureFormat::RGBA16Float,
                           wgpu::TextureFormat::RGBA16Float, wgpu::TextureFormat::RGBA32Uint};
  EXPECT_EQ(color_bytes_per_sample(formats), 40u);
  EXPECT_FALSE(color_target_cost(wgpu::TextureFormat::Depth32Float).has_value());
}

TestPass many_targets(uint32_t count, bool manual) {
  return TestPass{.name = "mrt", .kind = PassKind::Render, .body = [=](InternalPassBuilder& b) {
                    if (manual) {
                      b.manual_bindings();
                    }
                    b.use("gx.color", Access::ColorRMW);
                    for (uint32_t i = 1; i < count; ++i) {
                      b.create("c" + std::to_string(i), efb_tex(wgpu::TextureFormat::R8Unorm), Access::ColorWrite);
                    }
                  }};
}

TEST_F(RenderGraph, LimitColorAttachmentCount) {
  EXPECT_EQ(node(*compile({many_targets(9, false)}), "mrt").status, CullReason::BindingLimit);
  EXPECT_TRUE(active(*compile({many_targets(8, false)}), "mrt"));
}

TEST_F(RenderGraph, StorageFormatSupport) {
  using wgpu::TextureFormat;
  const auto create_storage = [](TextureFormat format) {
    return TestPass{.name = "p",
                    .body = [format](InternalPassBuilder& b) { b.create("t", efb_tex(format), Access::StorageWrite); }};
  };
  const auto read_write = [](TextureFormat format) {
    return TestPass{.name = "p", .body = [format](InternalPassBuilder& b) {
                      b.import_persistent_texture("t", {9}, absolute_tex(format, 64, 64), Access::StorageRW);
                    }};
  };
  const auto status = [](const TestPass& pass, auto&& configure) {
    GraphInputs inputs = test_inputs();
    configure(inputs.env.limits.features);
    return node(*compile({pass}, inputs), "p").status;
  };
  const auto none = [](GraphFeatures&) {};
  const auto noTier1 = [](GraphFeatures& f) { f.tier1Formats = false; };
  const auto tier2 = [](GraphFeatures& f) { f.tier2Formats = true; };
  const auto bgra8 = [](GraphFeatures& f) { f.bgra8Storage = true; };
  const auto compat = [](GraphFeatures& f) { f.core = false; };

  EXPECT_EQ(status(create_storage(TextureFormat::RG16Float), noTier1), CullReason::FormatUnsupported);
  EXPECT_EQ(status(create_storage(TextureFormat::RG16Float), none), CullReason::None);
  EXPECT_EQ(status(read_write(TextureFormat::RGBA16Float), none), CullReason::FormatUnsupported);
  EXPECT_EQ(status(read_write(TextureFormat::RGBA8Unorm), none), CullReason::FormatUnsupported);
  EXPECT_EQ(status(read_write(TextureFormat::RGBA16Float), tier2), CullReason::None);
  EXPECT_EQ(status(read_write(TextureFormat::RGBA8Unorm), tier2), CullReason::None);
  EXPECT_EQ(status(read_write(TextureFormat::R32Float), none), CullReason::None);
  EXPECT_EQ(status(create_storage(TextureFormat::BGRA8Unorm), none), CullReason::FormatUnsupported);
  EXPECT_EQ(status(create_storage(TextureFormat::BGRA8Unorm), bgra8), CullReason::None);
  EXPECT_EQ(status(create_storage(TextureFormat::RG32Float), none), CullReason::None);
  EXPECT_EQ(status(create_storage(TextureFormat::RG32Float), compat), CullReason::FormatUnsupported);
}

TEST_F(RenderGraph, RenderFormatSupport) {
  const TestPass pass{.name = "p", .kind = PassKind::Render, .body = [](InternalPassBuilder& b) {
                        b.create("t", efb_tex(wgpu::TextureFormat::RG11B10Ufloat), Access::ColorWrite);
                      }};
  GraphInputs inputs = test_inputs();
  inputs.env.limits.features.tier1Formats = false;
  inputs.env.limits.features.rg11b10Renderable = false;
  const auto graph = compile({pass}, inputs);
  EXPECT_EQ(node(*graph, "p").status, CullReason::FormatUnsupported);
  EXPECT_NE(first_message(*graph).find("rg11b10ufloat"), std::string::npos);
  inputs.env.limits.features.rg11b10Renderable = true;
  EXPECT_TRUE(active(*compile({pass}, inputs), "p"));
}

TEST_F(RenderGraph, ExtraUsageFormatCheck) {
  const TestPass pass{.name = "p", .kind = PassKind::Render, .body = [](InternalPassBuilder& b) {
                        TexDesc desc = efb_tex(wgpu::TextureFormat::RG8Unorm);
                        desc.extraUsage = wgpu::TextureUsage::StorageBinding;
                        b.create("t", desc, Access::ColorWrite);
                      }};
  GraphInputs inputs = test_inputs();
  inputs.env.limits.features.tier1Formats = false;
  EXPECT_EQ(node(*compile({pass}, inputs), "p").status, CullReason::FormatUnsupported);
  inputs.env.limits.features.tier1Formats = true;
  const auto graph = compile({pass}, inputs);
  EXPECT_TRUE(active(*graph, "p"));
  EXPECT_TRUE(static_cast<bool>(resource(*graph, "t").textureUsage & wgpu::TextureUsage::StorageBinding));
  EXPECT_TRUE(static_cast<bool>(resource(*graph, "t").textureUsage & wgpu::TextureUsage::RenderAttachment));
}

TEST_F(RenderGraph, ResourceLimit) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) {
                                 b.create("volume",
                                          {.format = wgpu::TextureFormat::RGBA8Unorm,
                                           .dimension = wgpu::TextureDimension::e3D,
                                           .sizeClass = SizeClass::Absolute,
                                           .width = 16,
                                           .height = 16,
                                           .depthOrLayers = 4096},
                                          Access::StorageWrite);
                               }}});
  EXPECT_EQ(node(*graph, "p").status, CullReason::ResourceLimit);
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_NE(diagnostics[0].message.find("depth 4096, limit 2048"), std::string::npos) << diagnostics[0].message;
}

TEST_F(RenderGraph, ManualBindingsSkipCounts) {
  const auto graph = compile({{.name = "manual", .body = [](InternalPassBuilder& b) {
                                 b.manual_bindings();
                                 storage_buffers("sb", 20)(b);
                               }}});
  EXPECT_TRUE(active(*graph, "manual"));
  EXPECT_TRUE(node(*graph, "manual").manualBindings);
  EXPECT_EQ(node(*compile({many_targets(9, true)}), "mrt").status, CullReason::BindingLimit);
}

TEST_F(RenderGraph, DisabledPassLeavesProgram) {
  const auto graph = compile({
      {.name = "owner",
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.import_persistent_texture("shared", {5}, absolute_tex(wgpu::TextureFormat::R32Float, 64, 64),
                                         Access::Sampled);
           }},
      {.name = "big",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("shared", Access::StorageRead);
             storage_buffers("sb", 11)(b);
           }},
  });
  EXPECT_EQ(node(*graph, "big").status, CullReason::BindingLimit);
  EXPECT_EQ(position(stage_passes(*graph, Stage::OpaqueDone), "owner"), 0u);
  const auto nodes = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_EQ(std::ranges::find(nodes, "big"), nodes.end());
  EXPECT_EQ(node(*graph, "big").programIndex, InvalidNode);
  EXPECT_FALSE(has_any_edge(*graph, "big"));
  EXPECT_EQ(resource(*graph, "shared").textureUsage, wgpu::TextureUsage::TextureBinding);
  EXPECT_EQ(resource(*graph, "sb0").bufferUsage, wgpu::BufferUsage::None);
}

TEST_F(RenderGraph, ExtentsResolve) {
  const auto pass =
      TestPass{.name = "p", .body = [](InternalPassBuilder& b) {
                 b.create("half", efb_tex(wgpu::TextureFormat::RGBA8Unorm, 2), Access::StorageWrite);
                 b.create("chain", {.format = wgpu::TextureFormat::R32Float, .mips = 0}, Access::StorageWrite);
                 b.create_buffer("tiles", {.sizeClass = SizeClass::Efb, .bytesPerTexel = 3, .divisor = 16},
                                 Access::StorageWrite);
               }};
  GraphInputs inputs = test_inputs();
  inputs.env.efbSize = {1281, 721};
  auto graph = compile({pass}, inputs);
  EXPECT_TRUE(active(*graph, "p"));
  EXPECT_EQ(resource(*graph, "half").extent.width, 641u);
  EXPECT_EQ(resource(*graph, "half").extent.height, 361u);
  EXPECT_EQ(resource(*graph, "tiles").bufferSize, 11184u);

  graph = compile({pass});
  EXPECT_EQ(resource(*graph, "chain").mipLevels, 11u);

  inputs.env.efbSize = {0, 0};
  graph = compile({pass}, inputs);
  EXPECT_TRUE(active(*graph, "p"));
  EXPECT_EQ(resource(*graph, "half").extent.width, 1u);
  EXPECT_EQ(resource(*graph, "half").extent.height, 1u);
  EXPECT_EQ(resource(*graph, "gx.color").extent.width, 1u);
}

TEST_F(RenderGraph, UsageUnion) {
  const auto graph = compile({
      {.name = "draw",
       .kind = PassKind::Render,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.create("image", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::ColorWrite);
             b.create_buffer("args", {.size = 64}, Access::StorageWrite);
           }},
      {.name = "sample",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("image", Access::Sampled);
             b.use_buffer("args", Access::Indirect);
           }},
      {.name = "copy",
       .kind = PassKind::Copy,
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("image", Access::CopySrc);
           }},
  });
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
  EXPECT_EQ(resource(*graph, "image").textureUsage,
            wgpu::TextureUsage(wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                               wgpu::TextureUsage::CopySrc));
  EXPECT_EQ(resource(*graph, "args").bufferUsage,
            wgpu::BufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect));
}

TEST_F(RenderGraph, LocalBlobIsPerCompile) {
  uint8_t marker = 7;
  const std::vector<TestPass> passes{{.name = "p", .body = [&marker](InternalPassBuilder& b) {
                                        const TexRef ref = b.create("t", efb_tex(wgpu::TextureFormat::RGBA8Unorm),
                                                                    Access::StorageWrite);
                                        auto local = b.local();
                                        std::memcpy(local.data(), &ref, sizeof(ref));
                                        local[sizeof(ref)] = std::byte{marker};
                                      }}};
  const auto first = compile(passes);
  marker = 9;
  const auto second = compile(passes);
  TexRef ref;
  std::memcpy(&ref, node(*first, "p").local.data(), sizeof(ref));
  EXPECT_EQ(ref.index, 0u);
  EXPECT_EQ(node(*first, "p").local[sizeof(ref)], std::byte{7});
  EXPECT_EQ(node(*second, "p").local[sizeof(ref)], std::byte{9});
  EXPECT_EQ(node(*first, "p").local[sizeof(ref) + 1], std::byte{0});
}

TEST_F(RenderGraph, RefsArePassLocalOrdinals) {
  std::vector<uint16_t> refs;
  const auto graph =
      compile({{.name = "p", .body = [&refs](InternalPassBuilder& b) {
                  refs.push_back(b.create("a", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite).index);
                  refs.push_back(b.use("gx.color", Access::ColorWrite).index);
                  refs.push_back(b.use("gx.depth", Access::Sampled).index);
                  refs.push_back(b.import_persistent("c", {3}, Access::StorageRead).index);
                }}});
  EXPECT_EQ(refs, (std::vector<uint16_t>{0, 0xFFFF, 1, 2}));
  EXPECT_FALSE(TexRef{}.valid());
  EXPECT_EQ(node(*graph, "p").useCount, 3u);
  const auto uses = graph->node_uses(graph->find_node("p"));
  EXPECT_EQ(graph->resources[uses[1].resource].name, "gx.depth");
  EXPECT_EQ(node(*graph, "p").status, CullReason::InvalidDeclaration);
}

std::vector<TestPass> hash_passes() {
  return {
      {.name = "h.create",
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.create("h.tex", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::StorageWrite);
             b.import_persistent("h.buf", {4}, Access::StorageWrite);
           }},
      {.name = "h.read",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("h.tex", Access::Sampled);
             b.use_buffer("h.buf", Access::StorageRead);
           }},
  };
}

TEST_F(RenderGraph, GraphHashDeterministic) {
  const auto passes = hash_passes();
  const auto first = compile(passes);
  const auto second = compile(passes);
  EXPECT_NE(first->graphHash, 0u);
  EXPECT_EQ(first->graphHash, second->graphHash);
}

TEST_F(RenderGraph, GraphHashIgnoresInterningOrder) {
  const auto passes = hash_passes();
  auto shuffled = passes;
  shuffled.insert(shuffled.begin(), TestPass{.name = "h.broken", .body = [](InternalPassBuilder& b) {
                                               b.use_buffer("h.buf", Access::StorageRead);
                                               b.use("h.tex", Access::Sampled);
                                               b.use("h.unrelated", Access::Sampled);
                                               b.create("gx.color", efb_tex(wgpu::TextureFormat::RGBA8Unorm),
                                                        Access::StorageWrite);
                                             }});
  const auto base = compile(passes);
  const auto graph = compile(shuffled);
  EXPECT_EQ(node(*graph, "h.broken").status, CullReason::InvalidDeclaration);
  EXPECT_NE(graph->find_resource("h.buf"), base->find_resource("h.buf"));
  EXPECT_EQ(graph->graphHash, base->graphHash);
}

bool always_enabled(const StageData&, const void*) { return true; }

TEST_F(RenderGraph, GraphHashIgnoresStatePointers) {
  const auto passes = hash_passes();
  const auto copies = passes;
  const auto base = compile(passes);
  GraphInputs inputs = test_inputs();
  auto descs = pass_descs(copies);
  for (PassDesc& desc : descs) {
    desc.enabled = always_enabled;
  }
  const std::vector<PassId> ids{0x100000001, 0x200000002};
  inputs.passes = descs;
  inputs.ids = ids;
  const auto graph = compile_graph(inputs);
  EXPECT_EQ(graph->nodes[0].id, ids[0]);
  EXPECT_EQ(graph->nodes[0].state, &copies[0]);
  EXPECT_EQ(graph->graphHash, base->graphHash);
}

TEST_F(RenderGraph, GraphHashIgnoresDisabledPasses) {
  auto passes = hash_passes();
  const auto base = compile(passes);
  passes.push_back({.name = "h.big", .body = [](InternalPassBuilder& b) {
                      b.use("h.tex", Access::Sampled);
                      storage_buffers("h.sb", 11)(b);
                    }});
  const auto graph = compile(passes);
  EXPECT_EQ(node(*graph, "h.big").status, CullReason::BindingLimit);
  EXPECT_EQ(graph->graphHash, base->graphHash);
}

TEST_F(RenderGraph, GraphHashTracksContent) {
  const auto base = compile(hash_passes())->graphHash;
  const auto variant = [](auto&& edit) {
    auto passes = hash_passes();
    GraphInputs inputs = test_inputs();
    edit(passes, inputs);
    const auto descs = pass_descs(passes);
    inputs.passes = descs;
    return compile_graph(inputs)->graphHash;
  };
  const auto readerBody = [](Access access, Missing missing) {
    return [=](InternalPassBuilder& b) {
      b.after("phase.lighting");
      b.use("h.tex", access, missing);
      b.use_buffer("h.buf", Access::StorageRead);
    };
  };
  EXPECT_NE(base, variant([&](std::vector<TestPass>& p, GraphInputs&) {
              p[1].body = readerBody(Access::StorageRead, Missing::DisablePass);
            }));
  EXPECT_NE(base, variant([&](std::vector<TestPass>& p, GraphInputs&) {
              p[1].body = readerBody(Access::Sampled, Missing::BindDummyBlack);
            }));
  EXPECT_NE(base, variant([](std::vector<TestPass>& p, GraphInputs&) {
              p[0].body = [](InternalPassBuilder& b) {
                b.after("phase.lighting");
                b.create("h.tex", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
                b.import_persistent("h.buf", {4}, Access::StorageWrite);
              };
            }));
  EXPECT_NE(base, variant([](std::vector<TestPass>&, GraphInputs& inputs) { inputs.env.efbSize = {1920, 1080}; }));
  EXPECT_NE(base, variant([](std::vector<TestPass>& p, GraphInputs&) {
              p.push_back({.name = "h.side", .order = 5, .body = [](InternalPassBuilder& b) { b.after("phase.ao"); }});
            }));

  auto passes = hash_passes();
  GraphInputs inputs = test_inputs();
  auto descs = pass_descs(passes);
  inputs.passes = descs;
  const auto plain = compile_graph(inputs)->graphHash;
  EXPECT_EQ(plain, base);
  descs[1].runsWithoutMarker = true;
  EXPECT_NE(compile_graph(inputs)->graphHash, base);

  const auto ordered = [](uint16_t first, uint16_t second) {
    return compile({{.name = "o.first", .order = first, .body = [](InternalPassBuilder& b) { b.after("phase.ao"); }},
                    {.name = "o.second", .order = second, .body = [](InternalPassBuilder& b) { b.after("phase.ao"); }}})
        ->graphHash;
  };
  EXPECT_EQ(ordered(0, 1), ordered(3, 4));
  EXPECT_NE(ordered(0, 1), ordered(1, 0));
}

std::vector<TestPass> golden_passes() {
  return {
      {.name = "g.create",
       .order = 0,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.create("g.tex", efb_tex(wgpu::TextureFormat::RGBA16Float), Access::StorageWrite);
           },
       .sink = false},
      {.name = "g.sample",
       .kind = PassKind::Render,
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("g.tex", Access::Sampled);
             b.use("gx.color", Access::ColorRMW);
           },
       .sink = false},
      {.name = "g.gx",
       .kind = PassKind::Render,
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("gx.depth", Access::DepthRead);
             b.use("gx.color", Access::ColorRMW);
           },
       .sink = false},
      {.name = "g.persist",
       .order = 3,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.import_persistent("g.buf", {1}, Access::StorageWrite);
           },
       .sink = false},
  };
}

// Recorded from the first passing run; a change here means the hash serialization changed and its tag must be bumped
constexpr uint64_t kGoldenGraphHash = 13004159072905762061ull;

TEST_F(RenderGraph, GraphHashGolden) {
  const auto graph = compile(golden_passes());
  EXPECT_TRUE(graph->diagnostics.empty());
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone),
            (std::vector<std::string>{"g.create", "g.sample", "g.gx", "g.persist"}));
  EXPECT_EQ(graph->graphHash, kGoldenGraphHash);
}

TEST_F(RenderGraph, GraphLimitsFromDevice) {
  wgpu::Limits limits{};
  limits.maxSampledTexturesPerShaderStage = 40;
  limits.maxStorageBuffersPerShaderStage = 10;
  limits.maxStorageTexturesPerShaderStage = 8;
  limits.maxColorAttachmentBytesPerSample = 128;
  limits.maxBufferSize = 1ull << 30;
  wgpu::CompatibilityModeLimits compat{};
  compat.maxStorageBuffersInVertexStage = 2;
  compat.maxStorageTexturesInVertexStage = 0;
  compat.maxStorageBuffersInFragmentStage = 4;
  compat.maxStorageTexturesInFragmentStage = 4;
  const GraphLimits mapped = graph_limits(limits, compat, {.core = false, .tier1Formats = true});
  for (const BindingCounts& stage : {mapped.vertex, mapped.fragment, mapped.compute}) {
    EXPECT_EQ(stage.sampledTextures, 40u);
    EXPECT_EQ(stage.samplers, DefaultStageLimits.samplers);
  }
  EXPECT_EQ(mapped.compute.storageBuffers, 10u);
  EXPECT_EQ(mapped.compute.storageTextures, 8u);
  EXPECT_EQ(mapped.vertex.storageBuffers, 2u);
  EXPECT_EQ(mapped.vertex.storageTextures, 0u);
  EXPECT_EQ(mapped.fragment.storageBuffers, 4u);
  EXPECT_EQ(mapped.fragment.storageTextures, 4u);
  EXPECT_EQ(mapped.colorAttachmentBytesPerSample, 128u);
  EXPECT_EQ(mapped.bufferSize, 1ull << 30);
  EXPECT_EQ(mapped.textureDimension2D, GraphLimits{}.textureDimension2D);
  EXPECT_FALSE(mapped.features.core);
  EXPECT_TRUE(mapped.features.tier1Formats);
}

void declare_bad_gx_read(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use("gx.color", Access::Sampled);
  builder.use_buffer("ok.args", Access::Indirect);
  builder.create("own", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
  builder.after("phase.capture");
}

TEST_F(RenderGraph, GxReadDeclarationsAreValidated) {
  GraphInputs inputs = test_inputs();
  const std::vector<GxReadDesc> reads{
      {.node = "phase.lighting", .declare = declare_bad_gx_read},
      {.node = "gx@OpaqueDone", .declare = declare_bad_gx_read},
  };
  inputs.gxReads = reads;
  const auto graph = compile_graph(inputs);
  const auto diagnostics = declaration_diagnostics(*graph);
  ASSERT_EQ(diagnostics.size(), 4u);
  EXPECT_EQ(diagnostics[0].node, InvalidNode);
  EXPECT_EQ(diagnostics[0].detail, "phase.lighting");
  for (const Diagnostic& diagnostic : diagnostics) {
    EXPECT_EQ(diagnostic.reason, CullReason::InvalidDeclaration) << diagnostic.message;
  }
  const NodeId chain = graph->find_node("gx@OpaqueDone");
  EXPECT_EQ(node(*graph, "gx@OpaqueDone").status, CullReason::None);
  const auto uses = graph->node_uses(chain);
  ASSERT_EQ(uses.size(), GxResourceCount + 1);
  EXPECT_EQ(graph->resources[uses.back().resource].name, "ok.args");
  EXPECT_EQ(uses.back().access, Access::Indirect);
  EXPECT_EQ(graph->find_resource("own"), InvalidResource);
}

struct PlannedGraph {
  std::vector<PassDesc> passes;
  std::vector<GxReadDesc> reads;
  std::shared_ptr<CompiledGraph> graph;
};

PlannedGraph compile_planned(bool reversed = false, bool inspectorOpen = true, GraphLimits limits = full_limits()) {
  PlannedGraph out{planned::planned_passes(reversed), planned::planned_gx_reads(), nullptr};
  GraphInputs inputs = planned::planned_inputs(limits, out.passes, out.reads);
  inputs.env.inspectorOpen = inspectorOpen;
  out.graph = compile_graph(inputs);
  return out;
}

TEST_F(RenderGraph, PlannedSetCompilesAcyclic) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  for (const Diagnostic& diagnostic : graph.diagnostics) {
    ADD_FAILURE() << diagnostic.message;
  }
  size_t passCount = 0;
  for (const PassDesc& desc : planned.passes) {
    EXPECT_TRUE(active(graph, desc.name)) << desc.name;
    ++passCount;
  }
  EXPECT_EQ(passCount, 68u);
  for (size_t s = 0; s < StageCount; ++s) {
    const auto stage = static_cast<Stage>(s);
    std::vector<std::string> rows;
    for (const PassDesc& desc : planned.passes) {
      if (desc.stage == stage) {
        rows.push_back(desc.name);
      }
    }
    EXPECT_EQ(stage_passes(graph, stage), rows) << stage_name(stage);
  }
  EXPECT_EQ(stage_passes(graph, Stage::AfterFrame), (std::vector<std::string>{"native.dxr"}));

  // Each pass is bounded by exactly one phase and the chain node after it, and phases never share an order value
  std::set<std::pair<std::string, uint16_t>> phaseOrders;
  for (const PassDesc& desc : planned.passes) {
    const NodeId id = graph.find_node(desc.name);
    std::vector<NodeId> lower;
    std::vector<NodeId> upper;
    for (const Edge& edge : graph.edges) {
      if (edge.kind == EdgeKind::Bound && edge.to == id) {
        lower.push_back(edge.from);
      } else if (edge.kind == EdgeKind::Bound && edge.from == id) {
        upper.push_back(edge.to);
      }
    }
    ASSERT_EQ(lower.size(), 1u) << desc.name;
    const CompiledNode& phase = graph.nodes[lower[0]];
    EXPECT_EQ(phase.kind, NodeKind::Phase) << desc.name;
    EXPECT_TRUE(phaseOrders.emplace(phase.name, desc.order).second) << desc.name;
    const auto chain = stage_chain(desc.stage);
    if (phase.chainIndex + 1u < chain.size()) {
      ASSERT_EQ(upper.size(), 1u) << desc.name;
      EXPECT_EQ(graph.nodes[upper[0]].name, chain[phase.chainIndex + 1].name) << desc.name;
    } else {
      EXPECT_TRUE(upper.empty()) << desc.name;
    }
    const auto nodes = stage_nodes(graph, desc.stage);
    EXPECT_GT(position(nodes, desc.name), position(nodes, phase.name));
  }

  for (const Edge& edge : graph.edges) {
    const auto& from = graph.nodes[edge.from];
    const auto& to = graph.nodes[edge.to];
    ASSERT_EQ(from.stage, to.stage);
    const auto nodes = stage_nodes(graph, from.stage);
    EXPECT_LT(position(nodes, from.name), position(nodes, to.name)) << from.name << " -> " << to.name;
  }
  for (const CompiledUse& use : graph.uses) {
    EXPECT_FALSE(use.unresolved) << graph.nodes[use.node].name << " " << graph.resources[use.resource].name;
  }

  const auto& lighting = node(graph, "light.deferred").bindings[2];
  EXPECT_EQ(lighting, (BindingCounts{.sampledTextures = 10,
                                     .samplers = FrameGroupCost.samplers,
                                     .storageBuffers = 4,
                                     .storageTextures = 1,
                                     .uniformBuffers = FrameGroupCost.uniformBuffers}));
  const auto& composite = node(graph, "composite.main");
  EXPECT_EQ(composite.bindings[1], (BindingCounts{.sampledTextures = 14,
                                                  .samplers = FrameGroupCost.samplers,
                                                  .storageBuffers = 2,
                                                  .uniformBuffers = FrameGroupCost.uniformBuffers}));
  EXPECT_EQ(composite.colorBytes, 16u);
  EXPECT_EQ(node(graph, "vg.resolve").colorBytes, 40u);
  EXPECT_EQ(node(graph, "grass.draw").colorBytes, 40u);

  GraphLimits defaults = full_limits();
  defaults.vertex = defaults.fragment = defaults.compute = DefaultStageLimits;
  EXPECT_TRUE(declaration_diagnostics(*compile_planned(false, true, defaults).graph).empty());

  GraphLimits noTier1 = full_limits();
  noTier1.features.tier1Formats = false;
  const auto withoutTier1 = compile_planned(false, true, noTier1);
  const auto diagnostics = declaration_diagnostics(*withoutTier1.graph);
  ASSERT_EQ(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].reason, CullReason::FormatUnsupported);
  EXPECT_EQ(node_name(*withoutTier1.graph, diagnostics[0].node), "gi.voxelize");
}

TEST_F(RenderGraph, PlannedSetIndependentOfRegistrationOrder) {
  const auto forward = compile_planned(false);
  const auto reversed = compile_planned(true);
  for (size_t s = 0; s < StageCount; ++s) {
    EXPECT_EQ(stage_nodes(*forward.graph, static_cast<Stage>(s)), stage_nodes(*reversed.graph, static_cast<Stage>(s)));
  }
  EXPECT_EQ(forward.graph->graphHash, reversed.graph->graphHash);
  EXPECT_TRUE(reversed.graph->diagnostics.empty());
}

TEST_F(RenderGraph, PlannedSetProducers) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  const auto expect_producer = [&](std::string_view pass, std::string_view resourceName, std::string_view expected) {
    EXPECT_EQ(node_name(graph, producer(graph, pass, resourceName)), expected) << pass << " " << resourceName;
  };
  expect_producer("composite.main", "ao.final", "ao.combine");
  expect_producer("light.deferred", "vis.sun", "grass.contact");
  expect_producer("exposure.meter", "scene.hdr", "refl.apply");
  expect_producer("legacy.copy", "gx.color", "grass.far_field");
  expect_producer("grass.far_field", "gx.depth", "grass.draw");
  expect_producer("gx@PrePost", "vis.sun", "grass.contact");
  expect_producer("terrain.classify", "grass.influence", "grass.influence");
  EXPECT_EQ(producer(graph, "ao.gtao", "radiance.defogged"), InvalidNode);
  EXPECT_FALSE(has_edge(graph, "grass.contact", "gx@PrePost", EdgeKind::Raw));
}

TEST_F(RenderGraph, PlannedSetUsage) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  using Tex = wgpu::TextureUsage;
  using Buf = wgpu::BufferUsage;
  EXPECT_EQ(resource(graph, "legacyTex").textureUsage, Tex(Tex::CopyDst | Tex::TextureBinding));
  for (const std::string_view name : {"vis.sun", "scene.hdr", "ao.final"}) {
    EXPECT_EQ(resource(graph, name).textureUsage,
              Tex(Tex::StorageBinding | Tex::TextureBinding | Tex::RenderAttachment))
        << name;
  }
  EXPECT_EQ(resource(graph, "fog.integrated").textureUsage, Tex(Tex::StorageBinding | Tex::TextureBinding));
  EXPECT_EQ(resource(graph, "env.ub").bufferUsage, Buf(Buf::Storage | Buf::Uniform));
  EXPECT_EQ(resource(graph, "grass.args").bufferUsage, Buf(Buf::Storage | Buf::Indirect));
}

TEST_F(RenderGraph, CompileTime) {
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  const GraphInputs inputs = planned::planned_inputs(full_limits(), passes, reads);
  constexpr int Runs = 100;
  uint64_t hash = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < Runs; ++i) {
    hash ^= compile_graph(inputs)->graphHash;
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const double mean = elapsed / Runs;
  RecordProperty("mean_compile_ms", std::to_string(mean));
  EXPECT_EQ(hash, 0u);
#ifdef NDEBUG
  EXPECT_LT(mean, 1.0);
#endif
}

class RenderGraphCull : public RenderGraph {};

using wgpu::TextureFormat;

TestPass creator_of(std::string name, std::string created, uint16_t order = 0) {
  return {.name = std::move(name),
          .order = order,
          .body =
              [created](InternalPassBuilder& b) {
                b.after("phase.lighting");
                b.create(created, efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite);
              },
          .sink = false};
}

const CompiledUse& use_of(const CompiledGraph& graph, std::string_view pass, std::string_view resourceName) {
  static const CompiledUse absent{};
  const CompiledUse* use = find_use(graph, pass, resourceName);
  EXPECT_NE(use, nullptr) << pass << " " << resourceName;
  return use != nullptr ? *use : absent;
}

std::vector<std::string> consumer_names(const CompiledGraph& graph, std::string_view pass) {
  const CompiledNode& compiled = node(graph, pass);
  return names(graph, std::span{graph.consumers}.subspan(compiled.firstConsumer, compiled.consumerCount));
}

const Diagnostic* diagnostic_for(const CompiledGraph& graph, std::string_view pass, CullReason reason) {
  const NodeId id = graph.find_node(pass);
  const auto it = std::ranges::find_if(graph.diagnostics, [&](const Diagnostic& diagnostic) {
    return diagnostic.node == id && diagnostic.reason == reason;
  });
  return it != graph.diagnostics.end() ? &*it : nullptr;
}

TEST_F(RenderGraphCull, ChainWithoutSinkIsCulled) {
  const auto graph = compile({
      creator_of("a", "x"),
      {.name = "b",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("x", Access::Sampled);
             b.create("y", efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite);
           },
       .sink = false},
  });
  EXPECT_EQ(node(*graph, "a").status, CullReason::NoConsumer);
  EXPECT_EQ(node(*graph, "b").status, CullReason::NoConsumer);
  EXPECT_TRUE(graph->stage_empty(Stage::OpaqueDone));
  EXPECT_EQ(stage_nodes(*graph, Stage::OpaqueDone), chain_names(Stage::OpaqueDone));
  EXPECT_FALSE(has_any_edge(*graph, "a"));
  EXPECT_FALSE(has_any_edge(*graph, "b"));
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
  EXPECT_EQ(count_logs(LOG_DEBUG, "culled"), 2u);
}

TEST_F(RenderGraphCull, SinksKeepChains) {
  using Ending = std::function<void(InternalPassBuilder&)>;
  const std::vector<std::pair<std::string, Ending>> endings{
      {"gx", [](InternalPassBuilder& b) { b.use("gx.color", Access::ColorRMW); }},
      {"persistent", [](InternalPassBuilder& b) { b.import_persistent("p", {1}, Access::StorageWrite); }},
      {"history",
       [](InternalPassBuilder& b) { b.history("h", efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite); }},
      {"external", [](InternalPassBuilder& b) { b.use_buffer("rt.native.hits", Access::StorageWrite); }},
      {"side effect", [](InternalPassBuilder& b) { b.side_effect(); }},
  };
  for (const auto& [label, ending] : endings) {
    std::vector<TestPass> passes{
        creator_of("a", "x"),
        {.name = "b",
         .kind = label == "gx" ? PassKind::Render : PassKind::Compute,
         .order = 1,
         .body =
             [ending](InternalPassBuilder& b) {
               b.after("phase.lighting");
               b.use("x", Access::Sampled);
               ending(b);
             },
         .sink = false},
        {.name = "native",
         .stage = Stage::AfterFrame,
         .kind = PassKind::Native,
         .body =
             [](InternalPassBuilder& b) {
               b.import_persistent("rt.native.hits", {26}, Access::StorageWrite, Owner::NativeD3D12);
             },
         .sink = false},
    };
    const auto graph = compile(passes);
    EXPECT_TRUE(active(*graph, "a")) << label;
    EXPECT_TRUE(active(*graph, "b")) << label;
    EXPECT_TRUE(node(*graph, "b").sink) << label;
    EXPECT_FALSE(node(*graph, "a").sink) << label;
    EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"a", "b"})) << label;
    EXPECT_TRUE(declaration_diagnostics(*graph).empty()) << label << " " << first_message(*graph);
  }
}

void declare_gx_read_of_x(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use("x", Access::Sampled, Missing::BindDummyBlack);
}

TEST_F(RenderGraphCull, GxReadKeepsProducerLive) {
  GraphInputs inputs = test_inputs();
  const std::vector<GxReadDesc> reads{{.node = "gx@PrePost", .declare = declare_gx_read_of_x}};
  inputs.gxReads = reads;
  auto graph = compile({creator_of("maker", "x")}, inputs);
  EXPECT_TRUE(active(*graph, "maker"));
  EXPECT_EQ(producer(*graph, "gx@PrePost", "x"), graph->find_node("maker"));
  EXPECT_EQ(consumer_names(*graph, "maker"), (std::vector<std::string>{"gx@PrePost"}));

  graph = compile({creator_of("maker", "x")});
  EXPECT_EQ(node(*graph, "maker").status, CullReason::NoConsumer);
}

TEST_F(RenderGraphCull, DiagnosticNeedsInspector) {
  const auto capture = [](bool sideEffect) {
    return TestPass{.name = "capture",
                    .stage = Stage::FrameEnd,
                    .kind = PassKind::Copy,
                    .cls = PassClass::Diagnostic,
                    .body = [](InternalPassBuilder& b) { b.use("gx.color", Access::CopySrc); },
                    .sink = sideEffect};
  };
  GraphInputs closed = test_inputs();
  closed.env.inspectorOpen = false;
  auto graph = compile({capture(true)}, closed);
  EXPECT_EQ(node(*graph, "capture").status, CullReason::InspectorClosed);
  ASSERT_EQ(graph->diagnostics.size(), 1u);
  EXPECT_EQ(graph->diagnostics[0].reason, CullReason::InspectorClosed);
  EXPECT_EQ(count_logs(LOG_DEBUG, "inspector is closed"), 1u);
  EXPECT_TRUE(graph->stage_empty(Stage::FrameEnd));

  for (const bool sideEffect : {true, false}) {
    graph = compile({capture(sideEffect)});
    EXPECT_TRUE(active(*graph, "capture")) << sideEffect;
    EXPECT_TRUE(node(*graph, "capture").sink) << sideEffect;
    EXPECT_TRUE(graph->diagnostics.empty()) << sideEffect;
  }
}

TEST_F(RenderGraphCull, InspectorClosedFeedsMissing) {
  const auto passes = [](Missing missing) {
    TestPass diagnostic = creator_of("d", "x");
    diagnostic.cls = PassClass::Diagnostic;
    return std::vector<TestPass>{
        diagnostic,
        {.name = "r",
         .order = 1,
         .body =
             [missing](InternalPassBuilder& b) {
               b.after("phase.lighting");
               b.use("x", Access::Sampled, missing);
             }},
    };
  };
  GraphInputs inputs = test_inputs();
  inputs.env.inspectorOpen = false;
  auto graph = compile(passes(Missing::DisablePass), inputs);
  EXPECT_EQ(node(*graph, "d").status, CullReason::InspectorClosed);
  EXPECT_EQ(node(*graph, "r").status, CullReason::MissingInput);
  const Diagnostic* missing = diagnostic_for(*graph, "r", CullReason::MissingInput);
  ASSERT_NE(missing, nullptr);
  EXPECT_EQ(missing->resource, graph->find_resource("x"));

  graph = compile(passes(Missing::BindDummyBlack), inputs);
  EXPECT_EQ(node(*graph, "d").status, CullReason::InspectorClosed);
  EXPECT_TRUE(active(*graph, "r"));
  EXPECT_TRUE(use_of(*graph, "r", "x").staticDummy);
}

TEST_F(RenderGraphCull, OpaquePassIsKept) {
  GraphInputs inputs = test_inputs();
  const std::vector<PassDesc> descs{{.name = "opaque", .stage = Stage::OpaqueDone}};
  inputs.passes = descs;
  const auto graph = compile_graph(inputs);
  EXPECT_TRUE(active(*graph, "opaque"));
  EXPECT_TRUE(node(*graph, "opaque").sink);
  EXPECT_EQ(stage_passes(*graph, Stage::OpaqueDone), (std::vector<std::string>{"opaque"}));
  EXPECT_TRUE(graph->diagnostics.empty());
}

TEST_F(RenderGraphCull, MissingDisablePassPropagates) {
  TestPass creator = creator_of("c", "x");
  creator.body = [](InternalPassBuilder& b) {
    b.after("phase.lighting");
    b.create("x", efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite);
    storage_buffers("sb", 11)(b);
  };
  const auto graph = compile({
      creator,
      {.name = "m",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("x", Access::Sampled);
             b.create("y", efb_tex(TextureFormat::RGBA16Float), Access::StorageWrite);
           },
       .sink = false},
      {.name = "r",
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("y", Access::Sampled);
           }},
  });
  EXPECT_EQ(node(*graph, "c").status, CullReason::BindingLimit);
  EXPECT_EQ(node(*graph, "m").status, CullReason::MissingInput);
  EXPECT_EQ(node(*graph, "r").status, CullReason::MissingInput);
  for (const auto& [pass, name] : {std::pair{"m", "x"}, std::pair{"r", "y"}}) {
    const Diagnostic* missing = diagnostic_for(*graph, pass, CullReason::MissingInput);
    ASSERT_NE(missing, nullptr) << pass;
    EXPECT_EQ(missing->resource, graph->find_resource(name)) << pass;
    EXPECT_NE(missing->message.find(std::string{"'"} + name + "'"), std::string::npos) << missing->message;
  }
  EXPECT_TRUE(graph->stage_empty(Stage::OpaqueDone));
}

TEST_F(RenderGraphCull, MissingBindsDummyWhite) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) {
                                 b.after("phase.lighting");
                                 b.use("never.created", Access::Sampled, Missing::BindDummyWhite);
                               }}});
  EXPECT_TRUE(active(*graph, "p"));
  const CompiledUse& use = use_of(*graph, "p", "never.created");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_TRUE(use.hasFallback);
  EXPECT_EQ(use.fallback, (DummyKey{.buffer = false,
                                    .color = DummyColor::White,
                                    .dimension = wgpu::TextureViewDimension::e2D,
                                    .sampleType = wgpu::TextureSampleType::Float}));
  EXPECT_EQ(use.producer, InvalidNode);
  const ResourceId id = graph->find_resource("never.created");
  EXPECT_FALSE(std::ranges::any_of(graph->edges, [&](const Edge& edge) { return edge.resource == id; }));
  EXPECT_EQ(resource(*graph, "never.created").textureUsage, wgpu::TextureUsage::None);
  EXPECT_TRUE(graph->diagnostics.empty());
}

DummyKey fallback_from_disabled_creator(const TexDesc& desc, Access access, bool float32Filterable = true,
                                        UseOptions options = {.missing = Missing::BindDummyBlack}) {
  const std::vector<TestPass> passes{
      {.name = "creator",
       .kind = access == Access::DepthWrite ? PassKind::Render : PassKind::Compute,
       .body =
           [desc, access](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.create("t", desc, access);
             storage_buffers("sb", 11)(b);
           },
       .sink = false},
      {.name = "reader",
       .order = 1,
       .body =
           [options](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("t", Access::Sampled, options);
           }},
  };
  GraphInputs inputs = test_inputs();
  inputs.env.limits.features.float32Filterable = float32Filterable;
  const auto graph = compile(passes, inputs);
  EXPECT_EQ(node(*graph, "creator").status, CullReason::BindingLimit);
  EXPECT_TRUE(active(*graph, "reader"));
  const CompiledUse& use = use_of(*graph, "reader", "t");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_TRUE(use.hasFallback);
  EXPECT_EQ(use.fallback, dummy_key(*graph, use, inputs.env.limits.features));
  return use.fallback;
}

TEST_F(RenderGraphCull, DummyShapeFromDisabledCreator) {
  using Dimension = wgpu::TextureViewDimension;
  using Sample = wgpu::TextureSampleType;
  const TexDesc volume{.format = TextureFormat::R16Float,
                       .dimension = wgpu::TextureDimension::e3D,
                       .sizeClass = SizeClass::Absolute,
                       .width = 32,
                       .height = 32,
                       .depthOrLayers = 32};
  auto key = fallback_from_disabled_creator(volume, Access::StorageWrite);
  EXPECT_EQ(key, (DummyKey{.color = DummyColor::Black, .dimension = Dimension::e3D, .sampleType = Sample::Float}));

  key = fallback_from_disabled_creator(efb_tex(TextureFormat::R32Uint), Access::StorageWrite);
  EXPECT_EQ(key.sampleType, Sample::Uint);
  EXPECT_EQ(key.dimension, Dimension::e2D);
  key = fallback_from_disabled_creator(efb_tex(TextureFormat::Depth32Float), Access::DepthWrite);
  EXPECT_EQ(key.sampleType, Sample::Depth);
  key = fallback_from_disabled_creator(efb_tex(TextureFormat::R32Float), Access::StorageWrite, false);
  EXPECT_EQ(key.sampleType, Sample::UnfilterableFloat);
  key = fallback_from_disabled_creator(efb_tex(TextureFormat::R32Float), Access::StorageWrite, true);
  EXPECT_EQ(key.sampleType, Sample::Float);
  key = fallback_from_disabled_creator(absolute_tex(TextureFormat::RGBA8Unorm, 64, 64, 4), Access::StorageWrite);
  EXPECT_EQ(key.dimension, Dimension::e2DArray);
  EXPECT_EQ(key.sampleType, Sample::Float);
  EXPECT_EQ(key.storageFormat, TextureFormat::Undefined);
}

TEST_F(RenderGraphCull, DummyShapeFromOptions) {
  const auto key = fallback_from_disabled_creator(
      absolute_tex(TextureFormat::RGBA8Unorm, 64, 64), Access::StorageWrite, true,
      {.missing = Missing::BindDummyBlack, .dimension = wgpu::TextureViewDimension::Cube});
  EXPECT_EQ(key.dimension, wgpu::TextureViewDimension::Cube);
  EXPECT_EQ(key.sampleType, wgpu::TextureSampleType::Float);
}

TEST_F(RenderGraphCull, BufferDummy) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) {
                                 b.after("phase.lighting");
                                 b.use_buffer("missing.buf", Access::StorageRead,
                                              {.missing = Missing::BindDummyBlack, .dummySize = 100});
                               }}});
  EXPECT_TRUE(active(*graph, "p"));
  const CompiledUse& use = use_of(*graph, "p", "missing.buf");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_EQ(use.fallback, (DummyKey{.buffer = true, .color = DummyColor::Black, .bufferSize = 112}));
}

TEST_F(RenderGraphCull, StorageDummyNeedsFormat) {
  const auto passes = [](wgpu::TextureFormat format) {
    return std::vector<TestPass>{
        {.name = "p",
         .body =
             [format](InternalPassBuilder& b) {
               b.after("phase.lighting");
               b.use("unknown", Access::StorageRead, {.missing = Missing::BindDummyBlack, .format = format});
               b.create("out", efb_tex(TextureFormat::RGBA8Unorm), Access::StorageWrite);
             },
         .sink = false},
        {.name = "c",
         .order = 1,
         .body =
             [](InternalPassBuilder& b) {
               b.after("phase.lighting");
               b.use("out", Access::Sampled);
             }},
    };
  };
  auto graph = compile(passes(TextureFormat::Undefined));
  EXPECT_EQ(node(*graph, "p").status, CullReason::InvalidDeclaration);
  EXPECT_EQ(node(*graph, "c").status, CullReason::MissingInput);
  EXPECT_NE(first_message(*graph).find("no known format"), std::string::npos) << first_message(*graph);

  graph = compile(passes(TextureFormat::R32Float));
  EXPECT_TRUE(active(*graph, "p"));
  EXPECT_TRUE(active(*graph, "c"));
  const CompiledUse& use = use_of(*graph, "p", "unknown");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_EQ(use.fallback.storageFormat, TextureFormat::R32Float);
  EXPECT_TRUE(declaration_diagnostics(*graph).empty());
}

TEST_F(RenderGraphCull, MissingWriteDisables) {
  const auto graph = compile({{.name = "p", .body = [](InternalPassBuilder& b) {
                                 b.after("phase.lighting");
                                 b.use("missing.t", Access::StorageRW, Missing::BindDummyBlack);
                               }}});
  EXPECT_EQ(node(*graph, "p").status, CullReason::MissingTarget);
  const Diagnostic* missing = diagnostic_for(*graph, "p", CullReason::MissingTarget);
  ASSERT_NE(missing, nullptr);
  EXPECT_EQ(missing->resource, graph->find_resource("missing.t"));
}

TEST_F(RenderGraphCull, UnavailableGxTarget) {
  GraphInputs inputs = test_inputs();
  inputs.env.gxFormats = planned::gx_formats_tier_b();
  const auto graph = compile(
      {
          {.name = "writer",
           .kind = PassKind::Render,
           .body =
               [](InternalPassBuilder& b) {
                 b.after("phase.lighting");
                 b.use("gbuf.surface", Access::ColorRMW);
               }},
          {.name = "reader",
           .order = 1,
           .body =
               [](InternalPassBuilder& b) {
                 b.after("phase.lighting");
                 b.use("gbuf.surface", Access::Sampled, Missing::BindDummyBlack);
               }},
      },
      inputs);
  EXPECT_EQ(node(*graph, "writer").status, CullReason::MissingTarget);
  EXPECT_TRUE(active(*graph, "reader"));
  const CompiledUse& use = use_of(*graph, "reader", "gbuf.surface");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_EQ(use.fallback.sampleType, wgpu::TextureSampleType::Uint);
  EXPECT_EQ(use.fallback.dimension, wgpu::TextureViewDimension::e2D);
}

TEST_F(RenderGraphCull, GxUsageGatesAccess) {
  const std::vector<TestPass> passes{
      {.name = "depth.copy",
       .kind = PassKind::Copy,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("gx.depth", Access::CopySrc);
           }},
      {.name = "normal.read",
       .order = 1,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("gx.normal", Access::Sampled, Missing::BindDummyBlack);
           }},
      {.name = "color.copy",
       .kind = PassKind::Copy,
       .order = 2,
       .body =
           [](InternalPassBuilder& b) {
             b.after("phase.lighting");
             b.use("gx.color", Access::CopySrc);
           }},
  };
  GraphInputs inputs = test_inputs();
  inputs.env.gxUsage = planned::gx_usage_today();
  auto graph = compile(passes, inputs);
  EXPECT_EQ(node(*graph, "depth.copy").status, CullReason::MissingInput);
  EXPECT_TRUE(active(*graph, "normal.read"));
  EXPECT_FALSE(use_of(*graph, "normal.read", "gx.normal").staticDummy);
  EXPECT_TRUE(active(*graph, "color.copy"));

  auto& normal = inputs.env.gxUsage[static_cast<size_t>(GxResource::Normal)];
  normal = normal & ~wgpu::TextureUsage::TextureBinding;
  graph = compile(passes, inputs);
  EXPECT_TRUE(active(*graph, "normal.read"));
  EXPECT_TRUE(use_of(*graph, "normal.read", "gx.normal").staticDummy);
}

TEST_F(RenderGraphCull, MsaaHidesGx) {
  const auto render = [](std::string name, std::string resourceName, Access access, uint16_t order) {
    return TestPass{.name = std::move(name),
                    .kind = PassKind::Render,
                    .order = order,
                    .body = [resourceName, access](InternalPassBuilder& b) {
                      b.after("phase.lighting");
                      b.use(resourceName, access);
                    }};
  };
  GraphInputs inputs = test_inputs();
  inputs.env.gxSampleCount = 4;
  const auto graph = compile(
      {
          render("normal.rmw", "gx.normal", Access::ColorRMW, 0),
          render("depth.write", "gx.depth", Access::DepthWrite, 1),
          render("color.rmw", "gx.color", Access::ColorRMW, 2),
          {.name = "color.sample",
           .order = 3,
           .body =
               [](InternalPassBuilder& b) {
                 b.after("phase.lighting");
                 b.use("gx.color", Access::Sampled);
               }},
      },
      inputs);
  EXPECT_EQ(node(*graph, "normal.rmw").status, CullReason::MissingTarget);
  EXPECT_EQ(node(*graph, "depth.write").status, CullReason::MissingTarget);
  EXPECT_TRUE(active(*graph, "color.rmw"));
  EXPECT_TRUE(active(*graph, "color.sample"));
}

void declare_lost_gx_read(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use("lost.t", Access::Sampled);
}

TEST_F(RenderGraphCull, GxReadDisablePassCoerced) {
  GraphInputs inputs = test_inputs();
  const std::vector<GxReadDesc> reads{{.node = "gx@OpaqueDone", .declare = declare_lost_gx_read}};
  inputs.gxReads = reads;
  const auto graph = compile({}, inputs);
  const CompiledUse& use = use_of(*graph, "gx@OpaqueDone", "lost.t");
  EXPECT_TRUE(use.staticDummy);
  EXPECT_TRUE(use.hasFallback);
  EXPECT_EQ(use.fallback.color, DummyColor::Black);
  EXPECT_EQ(node(*graph, "gx@OpaqueDone").status, CullReason::None);
  ASSERT_EQ(graph->diagnostics.size(), 1u);
  EXPECT_EQ(graph->diagnostics[0].reason, CullReason::CoercedMissing);
  EXPECT_EQ(graph->diagnostics[0].node, graph->find_node("gx@OpaqueDone"));
  EXPECT_EQ(graph->diagnostics[0].resource, graph->find_resource("lost.t"));
  EXPECT_EQ(count_logs(LOG_DEBUG, "lost.t"), 1u);
}

TEST_F(RenderGraphCull, ConsumersFollowModifiers) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  EXPECT_EQ(consumer_names(graph, "light.deferred"),
            (std::vector<std::string>{"refl.apply", "exposure.meter", "composite.main"}));
  const auto sun = consumer_names(graph, "sun.visibility");
  for (const std::string_view expected : {"grass.contact", "light.deferred", "gx@PrePost"}) {
    EXPECT_NE(std::ranges::find(sun, expected), sun.end()) << expected;
  }
  EXPECT_EQ(consumer_names(graph, "refl.ssr"), (std::vector<std::string>{"refl.rt", "refl.denoise"}));
}

TEST_F(RenderGraphCull, PlannedSetAllLive) {
  auto planned = compile_planned(false, false);
  const CompiledGraph& closed = *planned.graph;
  for (const Diagnostic& diagnostic : closed.diagnostics) {
    EXPECT_EQ(diagnostic.reason, CullReason::InspectorClosed) << diagnostic.message;
  }
  ASSERT_EQ(closed.diagnostics.size(), 1u);
  EXPECT_EQ(closed.diagnostics[0].reason, CullReason::InspectorClosed);
  EXPECT_EQ(node_name(closed, closed.diagnostics[0].node), "debug.capture");
  for (const PassDesc& desc : planned.passes) {
    if (desc.name != "debug.capture") {
      EXPECT_TRUE(active(closed, desc.name)) << desc.name;
    }
  }

  planned = compile_planned(false, true);
  EXPECT_TRUE(planned.graph->diagnostics.empty());
  size_t live = 0;
  for (const PassDesc& desc : planned.passes) {
    live += active(*planned.graph, desc.name) ? 1 : 0;
  }
  EXPECT_EQ(live, 68u);
  EXPECT_EQ(planned.graph->framePasses.size(), 68u);
}

void expect_degraded_reasons(const CompiledGraph& graph, std::span<const PassDesc> passes) {
  const std::set<CullReason> never{CullReason::Cycle,
                                   CullReason::BackwardsEdge,
                                   CullReason::InvalidDeclaration,
                                   CullReason::DuplicateCreator,
                                   CullReason::ConflictingKind,
                                   CullReason::WrongStage};
  const std::set<CullReason> allowed{CullReason::MissingInput, CullReason::MissingTarget,
                                     CullReason::BindingLimit, CullReason::FormatUnsupported,
                                     CullReason::NoConsumer,   CullReason::InspectorClosed};
  for (const Diagnostic& diagnostic : graph.diagnostics) {
    EXPECT_FALSE(never.contains(diagnostic.reason)) << diagnostic.message;
  }
  for (const PassDesc& desc : passes) {
    const CullReason status = node(graph, desc.name).status;
    if (status != CullReason::None) {
      EXPECT_TRUE(allowed.contains(status)) << desc.name << " " << cull_reason_name(status);
    }
  }
}

TEST_F(RenderGraphCull, PlannedSetUnderMetal32) {
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  GraphInputs inputs = planned::planned_inputs(planned::metal32_limits(), passes, reads);
  inputs.env.gxFormats = planned::gx_formats_tier_b();
  const auto graph = compile_graph(inputs);
  expect_degraded_reasons(*graph, passes);
  EXPECT_EQ(node(*graph, "light.deferred").status, CullReason::MissingInput);
  const Diagnostic* missing = diagnostic_for(*graph, "light.deferred", CullReason::MissingInput);
  ASSERT_NE(missing, nullptr);
  EXPECT_EQ(missing->resource, graph->find_resource("gbuf.surface"));
  EXPECT_TRUE(active(*graph, "composite.main"));
  EXPECT_TRUE(active(*graph, "aurora.taa"));
}

TEST_F(RenderGraphCull, PlannedSetUnderCompat) {
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  GraphInputs inputs = planned::planned_inputs(compat_limits(), passes, reads);
  inputs.env.gxFormats = planned::gx_formats_legacy();
  inputs.env.gxUsage = planned::gx_usage_today();
  const auto graph = compile_graph(inputs);
  expect_degraded_reasons(*graph, passes);
  EXPECT_TRUE(active(*graph, "demo.vignette"));
}

using EnabledFn = std::function<bool(std::string_view)>;
using StageHook = std::function<void(Stage, FrameCullState&)>;

struct CullFrame {
  std::array<StageMask, StageCount> masks;
  std::array<bool, StageCount> culled{};
};

CullFrame run_frame(const CompiledGraph& graph, FrameCullState& state, const EnabledFn& enabled = {},
                    const StageHook& before = {}) {
  CullFrame frame;
  begin_frame_cull(graph, state);
  for (size_t s = 0; s < StageCount; ++s) {
    const auto stage = static_cast<Stage>(s);
    if (before) {
      before(stage, state);
    }
    const StageProgram& program = graph.stages[s];
    if (program.passes.empty()) {
      continue;
    }
    std::vector<uint64_t> bits((program.passes.size() + 63) / 64, 0);
    for (size_t i = 0; i < program.passes.size(); ++i) {
      if (!enabled || enabled(graph.nodes[program.passes[i]].name)) {
        bits[i / 64] |= uint64_t{1} << (i % 64);
      }
    }
    cull_stage(graph, stage, bits, state, frame.masks[s]);
    frame.culled[s] = true;
  }
  return frame;
}

EnabledFn all_except(std::set<std::string, std::less<>> off) {
  return [off = std::move(off)](std::string_view name) { return !off.contains(name); };
}

bool runs(const CompiledGraph& graph, const CullFrame& frame, std::string_view pass) {
  const CompiledNode& compiled = node(graph, pass);
  return test_bit(frame.masks[static_cast<size_t>(compiled.stage)].passes, compiled.programIndex);
}

bool binds_dummy(const CompiledGraph& graph, const CullFrame& frame, std::string_view pass,
                 std::string_view resourceName) {
  const CompiledUse& use = use_of(graph, pass, resourceName);
  if (use.node == InvalidNode) {
    return false;
  }
  return test_bit(frame.masks[static_cast<size_t>(graph.nodes[use.node].stage)].dummies, use.stageUse);
}

std::set<std::string> dummy_bits(const CompiledGraph& graph, const CullFrame& frame) {
  std::set<std::string> out;
  for (size_t s = 0; s < StageCount; ++s) {
    for (const NodeId id : graph.stages[s].nodes) {
      for (const CompiledUse& use : graph.node_uses(id)) {
        if (test_bit(frame.masks[s].dummies, use.stageUse)) {
          out.insert(graph.nodes[id].name + " " + graph.resources[use.resource].name);
        }
      }
    }
  }
  return out;
}

std::set<std::string> static_dummies(const CompiledGraph& graph) {
  std::set<std::string> out;
  for (size_t s = 0; s < StageCount; ++s) {
    for (const NodeId id : graph.stages[s].nodes) {
      for (const CompiledUse& use : graph.node_uses(id)) {
        if (use.staticDummy) {
          out.insert(graph.nodes[id].name + " " + graph.resources[use.resource].name);
        }
      }
    }
  }
  return out;
}

bool prev_valid(const CompiledGraph& graph, const CullFrame& frame, Stage stage, std::string_view history) {
  return test_bit(frame.masks[static_cast<size_t>(stage)].prevValid, resource(graph, history).historySlot);
}

TEST_F(RenderGraphCull, FrameAllEnabled) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto first = run_frame(graph, state);
  auto expected = static_dummies(graph);
  const auto staticOnly = expected;
  expected.insert({"ao.gtao radiance.defogged", "refl.ssr radiance.defogged"});
  EXPECT_EQ(dummy_bits(graph, first), expected);
  for (const NodeId id : graph.framePasses) {
    EXPECT_TRUE(runs(graph, first, graph.nodes[id].name)) << graph.nodes[id].name;
  }

  const auto second = run_frame(graph, state);
  EXPECT_EQ(dummy_bits(graph, second), staticOnly);
  for (const NodeId id : graph.framePasses) {
    EXPECT_TRUE(runs(graph, second, graph.nodes[id].name)) << graph.nodes[id].name;
  }
}

TEST_F(RenderGraphCull, FrameDisabledProducerBindsDummy) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"capture.replay"}));
  EXPECT_FALSE(runs(graph, frame, "capture.replay"));
  EXPECT_TRUE(runs(graph, frame, "grass.draw"));
  EXPECT_TRUE(runs(graph, frame, "composite.main"));
  EXPECT_TRUE(binds_dummy(graph, frame, "grass.draw", "lsm.mask"));
  EXPECT_TRUE(binds_dummy(graph, frame, "composite.main", "lsm.mask"));
  EXPECT_TRUE(runs(graph, frame, "grass.clipmap"));
  EXPECT_TRUE(runs(graph, frame, "grass.cull"));
}

TEST_F(RenderGraphCull, FrameDisablePassPropagates) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"hzb.opaque"}));
  for (const std::string_view pass : {"hzb.opaque", "vg.cull", "vg.raster", "vg.resolve"}) {
    EXPECT_FALSE(runs(graph, frame, pass)) << pass;
  }
  EXPECT_TRUE(runs(graph, frame, "grass.hzb"));
  EXPECT_TRUE(runs(graph, frame, "grass.draw"));
}

TEST_F(RenderGraphCull, FrameCreatorRuleIgnoresModifiers) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"refl.apply"}));
  EXPECT_TRUE(runs(graph, frame, "exposure.meter"));
  EXPECT_TRUE(runs(graph, frame, "composite.main"));
  EXPECT_TRUE(runs(graph, frame, "light.deferred"));
  EXPECT_FALSE(binds_dummy(graph, frame, "composite.main", "scene.hdr"));
  EXPECT_FALSE(binds_dummy(graph, frame, "exposure.meter", "scene.hdr"));
}

TEST_F(RenderGraphCull, FrameBackwardSkipInStage) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"refl.apply"}));
  for (const std::string_view pass : {"refl.denoise", "refl.rt", "refl.ssr"}) {
    EXPECT_FALSE(runs(graph, frame, pass)) << pass;
  }
  EXPECT_TRUE(runs(graph, frame, "ao.gtao"));
}

TEST_F(RenderGraphCull, FrameBackwardSkipIsConservativeAcrossStages) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"godrays.apply"}));
  EXPECT_FALSE(runs(graph, frame, "godrays.blur"));
  EXPECT_TRUE(runs(graph, frame, "godrays.mask"));
  EXPECT_TRUE(runs(graph, frame, "aurora.taa"));
}

TEST_F(RenderGraphCull, FrameCrossStageForward) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"clouds.upsample"}));
  for (const std::string_view pass : {"godrays.mask", "godrays.blur", "godrays.apply"}) {
    EXPECT_FALSE(runs(graph, frame, pass)) << pass;
  }
  EXPECT_TRUE(runs(graph, frame, "aurora.taa"));
  EXPECT_TRUE(binds_dummy(graph, frame, "aurora.taa", "clouds.full"));
  EXPECT_TRUE(runs(graph, frame, "composite.main"));
  EXPECT_TRUE(binds_dummy(graph, frame, "composite.main", "clouds.full"));
  EXPECT_TRUE(runs(graph, frame, "clouds.reconstruct"));
}

TEST_F(RenderGraphCull, PersistentProducerOffNoEffect) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({"csm.update"}));
  EXPECT_TRUE(runs(graph, frame, "sun.visibility"));
  for (const CompiledUse& use : graph.node_uses(graph.find_node("sun.visibility"))) {
    EXPECT_FALSE(test_bit(frame.masks[static_cast<size_t>(Stage::OpaqueDone)].dummies, use.stageUse))
        << graph.resources[use.resource].name;
  }
}

TEST_F(RenderGraphCull, SinksNeverBackwardSkipped) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  const auto consumers = consumer_names(graph, "clouds.reconstruct");
  ASSERT_FALSE(consumers.empty());
  FrameCullState state;
  const auto frame = run_frame(graph, state, all_except({consumers.begin(), consumers.end()}));
  for (const std::string& consumer : consumers) {
    EXPECT_FALSE(runs(graph, frame, consumer)) << consumer;
  }
  EXPECT_TRUE(node(graph, "clouds.reconstruct").sink);
  EXPECT_TRUE(runs(graph, frame, "clouds.reconstruct"));
}

TEST_F(RenderGraphCull, HistoryValidity) {
  constexpr auto SceneBegin = Stage::SceneBegin;
  constexpr auto OpaqueDone = Stage::OpaqueDone;
  constexpr auto PrePost = Stage::PrePost;
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  const auto cut_before = [](Stage target) {
    return [target](Stage stage, FrameCullState& state) {
      if (stage == target) {
        invalidate_histories(state);
      }
    };
  };
  FrameCullState state;

  auto frame = run_frame(graph, state);
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(std::ranges::all_of(state.prevValid, [](uint64_t word) { return word == 0; }));

  frame = run_frame(graph, state);
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(prev_valid(graph, frame, SceneBegin, "grass.influence"));
  EXPECT_FALSE(binds_dummy(graph, frame, "ao.gtao", "radiance.defogged"));

  // A cut before OpaqueDone clears the rest of this frame, but OpaqueDone's owners still run after it
  frame = run_frame(graph, state, {}, cut_before(OpaqueDone));
  EXPECT_TRUE(prev_valid(graph, frame, SceneBegin, "radiance.defogged"));
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(binds_dummy(graph, frame, "ao.gtao", "radiance.defogged"));
  frame = run_frame(graph, state);
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "grass.influence"));

  frame = run_frame(graph, state, {}, cut_before(PrePost));
  frame = run_frame(graph, state);
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(prev_valid(graph, frame, PrePost, "taa.color"));

  frame = run_frame(graph, state);
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  invalidate_histories(state);
  frame = run_frame(graph, state);
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_FALSE(prev_valid(graph, frame, PrePost, "taa.color"));

  frame = run_frame(graph, state, all_except({"composite.main"}));
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  frame = run_frame(graph, state);
  EXPECT_FALSE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "ao.final"));

  frame = run_frame(graph, state);
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  GraphInputs resized = planned::planned_inputs(full_limits(), passes, reads);
  resized.env.inspectorOpen = true;
  resized.env.efbSize = {1920, 1080};
  const auto other = compile_graph(resized);
  ASSERT_NE(other->graphHash, graph.graphHash);
  frame = run_frame(*other, state);
  EXPECT_FALSE(prev_valid(*other, frame, OpaqueDone, "radiance.defogged"));
  frame = run_frame(*other, state);
  EXPECT_TRUE(prev_valid(*other, frame, OpaqueDone, "radiance.defogged"));

  frame = run_frame(graph, state);
  frame = run_frame(graph, state);
  EXPECT_TRUE(prev_valid(graph, frame, OpaqueDone, "radiance.defogged"));
  auto shifted = planned::planned_passes();
  const TestPass broken{.name = "broken", .body = storage_buffers("broken.sb", 11)};
  auto brokenDesc = pass_descs({broken});
  brokenDesc[0].state = &broken;
  shifted.insert(shifted.begin(), brokenDesc[0]);
  GraphInputs shiftedInputs = planned::planned_inputs(full_limits(), shifted, reads);
  shiftedInputs.env.inspectorOpen = true;
  const auto same = compile_graph(shiftedInputs);
  EXPECT_EQ(node(*same, "broken").status, CullReason::BindingLimit);
  ASSERT_EQ(same->graphHash, graph.graphHash);
  EXPECT_NE(same->find_node("composite.main"), graph.find_node("composite.main"));
  frame = run_frame(*same, state);
  EXPECT_TRUE(prev_valid(*same, frame, OpaqueDone, "radiance.defogged"));
  EXPECT_TRUE(prev_valid(*same, frame, PrePost, "taa.color"));
}

TEST_F(RenderGraphCull, EmptyMask) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  FrameCullState state;
  run_frame(graph, state);
  const auto frame = run_frame(graph, state, all_except({"demo.vignette"}));
  const StageMask& hud = frame.masks[static_cast<size_t>(Stage::BeforeHud)];
  EXPECT_TRUE(frame.culled[static_cast<size_t>(Stage::BeforeHud)]);
  EXPECT_FALSE(hud.any());
  EXPECT_TRUE(std::ranges::any_of(hud.prevValid, [](uint64_t word) { return word != 0; }));
  EXPECT_TRUE(graph.stage_empty(Stage::SceneEnd));
  EXPECT_FALSE(frame.culled[static_cast<size_t>(Stage::SceneEnd)]);
}

TEST_F(RenderGraphCull, MaskSizes) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  EXPECT_EQ(graph.stages[static_cast<size_t>(Stage::OpaqueDone)].passes.size(), 53u);
  FrameCullState state;
  const auto frame = run_frame(graph, state);
  for (size_t s = 0; s < StageCount; ++s) {
    if (!frame.culled[s]) {
      continue;
    }
    EXPECT_EQ(frame.masks[s].passes.size(), (graph.stages[s].passes.size() + 63) / 64) << s;
    EXPECT_EQ(frame.masks[s].dummies.size(), (graph.stages[s].useBits + 63) / 64) << s;
    EXPECT_EQ(frame.masks[s].prevValid.size(), (graph.histories.size() + 63) / 64) << s;
  }

  constexpr size_t Count = 130;
  std::vector<TestPass> passes;
  for (size_t i = 0; i < Count; ++i) {
    passes.push_back({.name = "s" + std::to_string(1000 + i), .stage = Stage::PrePost});
  }
  const auto wide = compile(passes);
  ASSERT_EQ(wide->stages[static_cast<size_t>(Stage::PrePost)].passes.size(), Count);
  FrameCullState wideState;
  const auto wideFrame = run_frame(
      *wide, wideState, [](std::string_view name) { return std::stoi(std::string{name.substr(1)}) % 3 == 0; });
  const StageMask& mask = wideFrame.masks[static_cast<size_t>(Stage::PrePost)];
  EXPECT_EQ(mask.passes.size(), 3u);
  for (size_t i = 0; i < Count; ++i) {
    EXPECT_EQ(runs(*wide, wideFrame, "s" + std::to_string(1000 + i)), (1000 + i) % 3 == 0) << i;
  }
  EXPECT_FALSE(test_bit(mask.passes, Count));
}

struct ReasonFrame {
  CullFrame frame;
  std::array<std::vector<FrameCullReason>, StageCount> reasons;
};

ReasonFrame run_frame_with_reasons(const CompiledGraph& graph, FrameCullState& state, const EnabledFn& enabled) {
  ReasonFrame out;
  begin_frame_cull(graph, state);
  for (size_t s = 0; s < StageCount; ++s) {
    const StageProgram& program = graph.stages[s];
    if (program.passes.empty()) {
      continue;
    }
    std::vector<uint64_t> bits((program.passes.size() + 63) / 64, 0);
    for (size_t i = 0; i < program.passes.size(); ++i) {
      if (enabled(graph.nodes[program.passes[i]].name)) {
        bits[i / 64] |= uint64_t{1} << (i % 64);
      }
    }
    // Seeded with a value cull_stage must overwrite
    out.reasons[s].assign(program.passes.size(), FrameCullReason::NoActiveConsumer);
    cull_stage(graph, static_cast<Stage>(s), bits, state, out.frame.masks[s], out.reasons[s]);
    out.frame.culled[s] = true;
  }
  return out;
}

FrameCullReason reason_of(const CompiledGraph& graph, const ReasonFrame& frame, std::string_view pass) {
  const CompiledNode& compiled = node(graph, pass);
  return frame.reasons[static_cast<size_t>(compiled.stage)][compiled.programIndex];
}

TEST_F(RenderGraphCull, CullReasonsMatchMask) {
  const auto planned = compile_planned();
  const CompiledGraph& graph = *planned.graph;
  const std::vector<std::set<std::string, std::less<>>> cases{
      {}, {"capture.replay"}, {"hzb.opaque"}, {"refl.apply"}, {"godrays.apply"}, {"clouds.upsample"},
  };
  for (const auto& off : cases) {
    FrameCullState plainState;
    FrameCullState reasonState;
    const CullFrame plain = run_frame(graph, plainState, all_except(off));
    const ReasonFrame frame = run_frame_with_reasons(graph, reasonState, all_except(off));
    for (size_t s = 0; s < StageCount; ++s) {
      if (!plain.culled[s]) {
        continue;
      }
      EXPECT_EQ(frame.frame.masks[s].passes, plain.masks[s].passes) << s;
      EXPECT_EQ(frame.frame.masks[s].dummies, plain.masks[s].dummies) << s;
      for (size_t i = 0; i < graph.stages[s].passes.size(); ++i) {
        const std::string& name = graph.nodes[graph.stages[s].passes[i]].name;
        const FrameCullReason reason = frame.reasons[s][i];
        EXPECT_EQ(reason == FrameCullReason::Ran, test_bit(plain.masks[s].passes, i)) << name;
        EXPECT_EQ(reason == FrameCullReason::NotEnabled, off.contains(name)) << name;
      }
    }
  }

  FrameCullState forwardState;
  const ReasonFrame forward = run_frame_with_reasons(graph, forwardState, all_except({"hzb.opaque"}));
  EXPECT_EQ(reason_of(graph, forward, "hzb.opaque"), FrameCullReason::NotEnabled);
  EXPECT_EQ(reason_of(graph, forward, "vg.cull"), FrameCullReason::InputDisabled);
  FrameCullState backwardState;
  const ReasonFrame backward = run_frame_with_reasons(graph, backwardState, all_except({"refl.apply"}));
  EXPECT_EQ(reason_of(graph, backward, "refl.denoise"), FrameCullReason::NoActiveConsumer);
  EXPECT_EQ(reason_of(graph, backward, "ao.gtao"), FrameCullReason::Ran);
  EXPECT_EQ(frame_cull_reason_name(FrameCullReason::InputDisabled), "InputDisabled");
  EXPECT_EQ(frame_cull_reason_name(static_cast<FrameCullReason>(9)), "Unknown");
}
} // namespace
} // namespace aurora::render::detail

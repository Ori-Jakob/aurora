#include <gtest/gtest.h>

#include "render/debug/graph_inspection.hpp"
#include "render/graph.hpp"
#include "render/graph_api.hpp"
#include "render/json.hpp"
#include "render_graph_planned.hpp"
#include "webgpu/gpu_prof.hpp"

#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
using debug::format_graph_json;
using debug::GraphInspection;

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in{path, std::ios::binary};
  return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

void write_actual(std::string_view name, const std::string& text) {
  const std::filesystem::path dir{AURORA_RENDER_FIXTURE_OUT};
  std::filesystem::create_directories(dir);
  std::ofstream out{dir / name, std::ios::binary};
  out << text;
}

// Checkouts may turn the fixtures into CRLF, so the comparison is on parsed documents
void expect_matches_fixture(std::string_view name, const std::string& actual) {
  write_actual(name, actual);
  const std::string expected = read_text(std::filesystem::path{AURORA_RENDER_FIXTURES_DIR} / name);
  ASSERT_FALSE(expected.empty()) << "missing fixture " << name;
  const json::Doc actualDoc = json::read(actual);
  const json::Doc expectedDoc = json::read(expected);
  ASSERT_NE(actualDoc, nullptr) << actual;
  ASSERT_NE(expectedDoc, nullptr) << name;
  EXPECT_TRUE(yyjson_equals(yyjson_doc_get_root(actualDoc.get()), yyjson_doc_get_root(expectedDoc.get())))
      << name << " differs; the output is in " << AURORA_RENDER_FIXTURE_OUT;
}

// Checkouts may turn the fixtures into CRLF, so carriage returns are dropped before comparing
void expect_text_matches_fixture(std::string_view name, const std::string& actual) {
  write_actual(name, actual);
  std::string expected = read_text(std::filesystem::path{AURORA_RENDER_FIXTURES_DIR} / name);
  ASSERT_FALSE(expected.empty()) << "missing fixture " << name;
  std::erase(expected, '\r');
  EXPECT_EQ(actual, expected) << name << " differs; the output is in " << AURORA_RENDER_FIXTURE_OUT;
}

GraphInspection format_inspection() {
  GraphInspection graph;
  graph.graphHash = 0xab;
  graph.stages = {
      {"SceneBegin", 0, 0}, {"OpaqueDone", 0, 3}, {"PrePost", 3, 0},    {"SceneEnd", 3, 0},
      {"BeforeHud", 3, 0},  {"FrameEnd", 3, 0},   {"AfterFrame", 3, 0},
  };
  graph.passes = {
      {.name = "gx@OpaqueDone", .stage = "OpaqueDone", .nodeKind = "Gx", .writes = {{"gx.color", "ColorWrite"}}},
      {.name = "phase.prepare", .stage = "OpaqueDone", .nodeKind = "Phase"},
      {
          .name = "test.blur",
          .stage = "OpaqueDone",
          .nodeKind = "Pass",
          .kind = "Compute",
          .cls = "Optional",
          .order = 5,
          .registration = 7,
          .active = false,
          .cull = "NoConsumer",
          .detail = "no sink reachable",
          .external = true,
          .reads = {{"gx.color", "Sampled", "DisablePass", "Resource"}},
          .writes = {{"test.tmp", "StorageWrite"}},
      },
  };
  graph.resources = {
      {
          .name = "gx.color",
          .kind = "Gx",
          .type = "texture",
          .format = "BGRA8Unorm",
          .size = "Efb",
          .width = 640,
          .height = 480,
          .usage = {"CopySrc", "TextureBinding", "RenderAttachment"},
          .bytes = 1228800,
          .firstPass = 0,
          .lastPass = 2,
      },
      {
          .name = "test.tmp",
          .kind = "Transient",
          .type = "texture",
          .format = "RGBA16Float",
          .size = "Efb/2",
          .width = 320,
          .height = 240,
          .usage = {"TextureBinding", "StorageBinding"},
          .bytes = 614400,
          .slot = 0,
          .firstPass = 2,
          .lastPass = 2,
      },
  };
  graph.edges = {
      {.from = 0, .to = 1, .kind = "Chain"},
      {.from = 0, .to = 2, .kind = "Raw", .resource = "gx.color"},
      {.from = 1, .to = 2, .kind = "Anchor"},
  };
  graph.poolSlots = 1;
  graph.logicalBytes = 614400;
  graph.slotBytes = 614400;
  return graph;
}

yyjson_val* key(yyjson_val* object, const char* name) { return yyjson_obj_get(object, name); }

std::string_view str(yyjson_val* value) {
  const char* text = yyjson_get_str(value);
  return text != nullptr ? std::string_view{text, yyjson_get_len(value)} : std::string_view{};
}

TEST(RenderGraphDump, FormatMatchesFixture) {
  const std::string json = format_graph_json(format_inspection());
  expect_matches_fixture("graph_dump_format.json", json);
  // The parsed comparison ignores key order, so the bytes are compared too once line endings are normalised
  std::string fixture = read_text(std::filesystem::path{AURORA_RENDER_FIXTURES_DIR} / "graph_dump_format.json");
  std::erase(fixture, '\r');
  EXPECT_EQ(json, fixture);
  ASSERT_GE(json.size(), 2u);
  EXPECT_TRUE(json.ends_with("}\n"));
  EXPECT_EQ(json.substr(0, json.find('\n')), "{");
}

TEST(RenderGraphDump, EscapesAndNulls) {
  GraphInspection graph = format_inspection();
  graph.passes[2].name = "te\"st\\x";
  graph.passes[2].cull.reset();
  const json::Doc doc = json::read(format_graph_json(graph));
  ASSERT_NE(doc, nullptr);
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  yyjson_val* pass = yyjson_arr_get(key(root, "passes"), 2);
  EXPECT_EQ(str(key(pass, "name")), "te\"st\\x");
  EXPECT_TRUE(yyjson_is_null(key(pass, "cull")));
  EXPECT_EQ(str(key(pass, "detail")), "no sink reachable");
  EXPECT_TRUE(yyjson_is_null(key(key(root, "pool"), "claimedBytes")));
  EXPECT_TRUE(yyjson_is_null(key(key(root, "pool"), "idleBytes")));
  EXPECT_TRUE(yyjson_is_null(key(yyjson_arr_get(key(root, "passes"), 0), "kind")));
  EXPECT_EQ(str(key(root, "graphHash")), "0x00000000000000ab");

  graph.claimedBytes = 4096;
  graph.idleBytes = 0;
  const json::Doc live = json::read(format_graph_json(graph));
  ASSERT_NE(live, nullptr);
  yyjson_val* pool = key(yyjson_doc_get_root(live.get()), "pool");
  EXPECT_EQ(yyjson_get_uint(key(pool, "claimedBytes")), 4096u);
  EXPECT_TRUE(yyjson_is_uint(key(pool, "idleBytes")));
}

TEST(RenderGraphDump, FrameStateSectionsOnlyWhenRequested) {
  GraphInspection graph = format_inspection();
  graph.passes[0].frameSlot = 1;
  graph.passes[2].frameSlot = 0;
  const json::Doc bare = json::read(format_graph_json(graph));
  ASSERT_NE(bare, nullptr);
  EXPECT_EQ(key(yyjson_doc_get_root(bare.get()), "frame"), nullptr);
  EXPECT_EQ(key(yyjson_doc_get_root(bare.get()), "timings"), nullptr);

  const debug::FrameCullSnapshot frame{.frameId = 42, .graphHash = 0xab, .reasons = {"NoActiveConsumer", std::nullopt}};
  const char* const passName = "EFB 0";
  const webgpu::gpu_prof::FrameTimes timings{
      .sequence = 9, .frameMs = 3.2154f, .dropped = 2, .passes = {{passName, 1.1116f}, {"test.blur", 0.25f}}};
  const json::Doc full = json::read(format_graph_json(graph, &frame, &timings));
  ASSERT_NE(full, nullptr);
  yyjson_val* root = yyjson_doc_get_root(full.get());
  yyjson_val* frameKey = key(root, "frame");
  ASSERT_NE(frameKey, nullptr);
  EXPECT_EQ(yyjson_get_uint(key(frameKey, "frameId")), 42u);
  yyjson_val* framePasses = key(frameKey, "passes");
  ASSERT_EQ(yyjson_arr_size(framePasses), 2u);
  EXPECT_EQ(yyjson_get_uint(key(yyjson_arr_get(framePasses, 0), "index")), 0u);
  EXPECT_TRUE(yyjson_is_null(key(yyjson_arr_get(framePasses, 0), "reason")));
  EXPECT_EQ(yyjson_get_uint(key(yyjson_arr_get(framePasses, 1), "index")), 2u);
  EXPECT_EQ(str(key(yyjson_arr_get(framePasses, 1), "reason")), "NoActiveConsumer");

  yyjson_val* timingKey = key(root, "timings");
  ASSERT_NE(timingKey, nullptr);
  EXPECT_EQ(yyjson_get_uint(key(timingKey, "sequence")), 9u);
  EXPECT_DOUBLE_EQ(yyjson_get_real(key(timingKey, "frameMs")), 3.215);
  EXPECT_EQ(yyjson_get_uint(key(timingKey, "dropped")), 2u);
  yyjson_val* timed = key(timingKey, "passes");
  ASSERT_EQ(yyjson_arr_size(timed), timings.passes.size());
  EXPECT_EQ(str(key(yyjson_arr_get(timed, 0), "name")), "EFB 0");
  EXPECT_DOUBLE_EQ(yyjson_get_real(key(yyjson_arr_get(timed, 0), "ms")), 1.112);
  EXPECT_DOUBLE_EQ(yyjson_get_real(key(yyjson_arr_get(timed, 1), "ms")), 0.25);
}

using DeclareBody = std::function<void(InternalPassBuilder&)>;

void declare_body(InternalPassBuilder& builder, const PassSetup&, const void* state) {
  (*static_cast<const DeclareBody*>(state))(builder);
}

TexDesc absolute_rgba8() {
  return {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Absolute, .width = 64, .height = 64};
}

const debug::InspectPass& find_pass(const GraphInspection& graph, std::string_view name, uint32_t* index = nullptr) {
  static const debug::InspectPass absent{};
  const auto it = std::ranges::find(graph.passes, name, &debug::InspectPass::name);
  EXPECT_NE(it, graph.passes.end()) << name;
  if (it == graph.passes.end()) {
    return absent;
  }
  if (index != nullptr) {
    *index = static_cast<uint32_t>(it - graph.passes.begin());
  }
  return *it;
}

bool has_edge(const GraphInspection& graph, uint32_t from, uint32_t to, std::string_view kind,
              std::optional<std::string_view> resource = std::nullopt) {
  return std::ranges::any_of(graph.edges, [&](const debug::InspectEdge& edge) {
    return edge.from == from && edge.to == to && edge.kind == kind &&
           (!resource || (edge.resource && *edge.resource == *resource));
  });
}

std::shared_ptr<CompiledGraph> compile_basic_graph() {
  static const std::vector<DeclareBody> bodies{
      [](InternalPassBuilder& b) {
        b.after("phase.prepare");
        b.create("test.tex", absolute_rgba8(), Access::StorageWrite);
      },
      [](InternalPassBuilder& b) {
        b.after("phase.ao");
        b.use("test.tex", Access::Sampled);
        b.create_buffer("test.buf", {.size = 256}, Access::StorageWrite);
      },
      [](InternalPassBuilder& b) {
        b.after("phase.composite");
        b.use_buffer("test.buf", Access::StorageRead);
        b.use("gx.color", Access::ColorRMW);
      },
      [](InternalPassBuilder& b) { b.create("test.unused", absolute_rgba8(), Access::StorageWrite); },
      [](InternalPassBuilder& b) {
        b.side_effect();
        b.use("test.absent", Access::Sampled, Missing::DisablePass);
      },
      [](InternalPassBuilder& b) {
        b.side_effect();
        b.use("test.absent2", Access::Sampled, Missing::BindDummyWhite);
      },
  };
  const std::array<std::pair<std::string_view, PassKind>, 6> passes{{
      {"test.write", PassKind::Compute},
      {"test.read", PassKind::Compute},
      {"test.sink", PassKind::Render},
      {"test.orphan", PassKind::Compute},
      {"test.missing", PassKind::Compute},
      {"test.dummy", PassKind::Compute},
  }};
  std::vector<PassDesc> descs;
  for (size_t i = 0; i < passes.size(); ++i) {
    descs.push_back({
        .name = std::string{passes[i].first},
        .stage = Stage::OpaqueDone,
        .kind = passes[i].second,
        .cls = PassClass::Optional,
        .declare = declare_body,
        .state = &bodies[i],
    });
  }
  return compile_graph(planned::planned_inputs(planned::full_limits(), descs, {}));
}

TEST(RenderGraphDump, CompiledGraphMatchesFixture) {
  const auto compiled = compile_basic_graph();
  const GraphInspection graph = inspect(*compiled);
  expect_matches_fixture("graph_dump_basic.json", format_graph_json(graph));

  uint32_t write = 0;
  uint32_t read = 0;
  uint32_t sink = 0;
  find_pass(graph, "test.write", &write);
  find_pass(graph, "test.read", &read);
  find_pass(graph, "test.sink", &sink);
  EXPECT_LT(write, read);
  EXPECT_LT(read, sink);
  EXPECT_TRUE(has_edge(graph, write, read, "Raw", "test.tex"));
  EXPECT_TRUE(has_edge(graph, read, sink, "Raw", "test.buf"));
  uint32_t prepare = 0;
  uint32_t ao = 0;
  uint32_t composite = 0;
  find_pass(graph, "phase.prepare", &prepare);
  find_pass(graph, "phase.ao", &ao);
  find_pass(graph, "phase.composite", &composite);
  // Anchors on phases become the lower bound of the pass's segment; Anchor edges join two passes
  EXPECT_TRUE(has_edge(graph, prepare, write, "Bound"));
  EXPECT_TRUE(has_edge(graph, ao, read, "Bound"));
  EXPECT_TRUE(has_edge(graph, composite, sink, "Bound"));

  EXPECT_EQ(find_pass(graph, "test.orphan").cull, "NoConsumer");
  const debug::InspectPass& missing = find_pass(graph, "test.missing");
  EXPECT_EQ(missing.cull, "MissingInput");
  ASSERT_TRUE(missing.detail.has_value());
  EXPECT_NE(missing.detail->find("test.absent"), std::string::npos);
  const debug::InspectPass& dummy = find_pass(graph, "test.dummy");
  EXPECT_TRUE(dummy.active);
  ASSERT_EQ(dummy.reads.size(), 1u);
  EXPECT_EQ(dummy.reads[0].binding, "DummyWhite");

  const auto tex = std::ranges::find(graph.resources, "test.tex", &debug::InspectResource::name);
  ASSERT_NE(tex, graph.resources.end());
  EXPECT_TRUE(tex->slot.has_value());
  EXPECT_EQ(tex->firstPass, write);
  EXPECT_EQ(tex->lastPass, read);
  EXPECT_EQ(tex->bytes, 64u * 64u * 4u);
  EXPECT_FALSE(graph.claimedBytes.has_value());
  EXPECT_FALSE(graph.idleBytes.has_value());
  EXPECT_EQ(graph.graphHash, compiled->graphHash);
  EXPECT_EQ(graph.stages.size(), StageCount);
  EXPECT_TRUE(std::ranges::is_sorted(graph.resources, {}, &debug::InspectResource::name));
}

TEST(RenderGraphDump, DotMatchesFixture) {
  const auto compiled = compile_basic_graph();
  const std::string dot = debug::format_graph_dot(inspect(*compiled));
  expect_text_matches_fixture("graph_dump_basic.dot", dot);
  EXPECT_EQ(dot.find('\r'), std::string::npos);
  EXPECT_TRUE(dot.starts_with("digraph render_graph {\n"));
  EXPECT_TRUE(dot.ends_with("}\n"));
}

TEST(RenderGraphDump, DotEscapesNames) {
  GraphInspection graph = format_inspection();
  graph.passes[2].name = "te\"st\\x";
  const debug::FrameCullSnapshot frame{.frameId = 1, .graphHash = 0xab, .reasons = {"NotEnabled"}};
  graph.passes[0].frameSlot = 0;
  const std::string dot = debug::format_graph_dot(graph, &frame);
  EXPECT_NE(dot.find("label=\"te\\\"st\\\\x\\nNoConsumer\""), std::string::npos) << dot;
  EXPECT_NE(dot.find("style=dashed"), std::string::npos);
  EXPECT_NE(dot.find("last frame: NotEnabled"), std::string::npos);
  EXPECT_NE(dot.find("n0 -> n2 [label=\"gx.color\"]"), std::string::npos);
  EXPECT_NE(dot.find("n1 -> n2 [style=dotted]"), std::string::npos);
  EXPECT_NE(dot.find("n0 -> n1;"), std::string::npos);
  EXPECT_NE(dot.find("subgraph cluster_1 {\n    label=\"OpaqueDone\";"), std::string::npos);
}

void declare_depth_reader(PassBuilder& builder, void*) {
  builder.use(render::resource::GxDepth, Access::Sampled);
  builder.side_effect();
}

void encode_nothing(render::PassContext&, void*) {}

TEST(RenderGraphDump, DumpReportsSceneDepthAndExternals) {
  testing::reset_external_passes();
  const PassId id = render::register_pass({
      .name = "mod.depth",
      .kind = PassKind::Compute,
      .declare = declare_depth_reader,
      .encode = encode_nothing,
  });
  ASSERT_NE(id, render::InvalidPass);
  const std::shared_ptr<const ExternalPassSet> set = snapshot_external_passes();
  std::vector<PassDesc> passes;
  append_external_passes(*set, nullptr, passes);
  GraphInputs inputs = planned::planned_inputs(planned::full_limits(), passes, {});
  inputs.externals = set;
  inputs.env.gxSampleCount = 4;
  inputs.sceneDepthSource = DepthSource::Copy;
  const auto compiled = compile_graph(inputs);
  testing::reset_external_passes();

  const json::Doc doc = json::read(format_graph_json(inspect(*compiled)));
  ASSERT_NE(doc, nullptr);
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_EQ(str(key(root, "sceneDepth")), "copy");
  yyjson_val* passList = key(root, "passes");
  bool found = false;
  for (size_t i = 0; i < yyjson_arr_size(passList); ++i) {
    yyjson_val* pass = yyjson_arr_get(passList, i);
    if (str(key(pass, "name")) == "mod.depth") {
      found = true;
      EXPECT_TRUE(yyjson_get_bool(key(pass, "external")));
      EXPECT_TRUE(yyjson_get_bool(key(pass, "active")));
    }
  }
  EXPECT_TRUE(found);
}
} // namespace
} // namespace aurora::render::detail

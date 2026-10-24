#include <gtest/gtest.h>

#include "logging.hpp"
#include "render/debug/graph_inspection.hpp"
#include "render/frame_state.hpp"
#include "render/graph.hpp"
#include "render/graph_api.hpp"
#include "render/scene_depth.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
#include "render_graph_planned.hpp"

#include <aurora/render.hpp>
#include <aurora/render_graph.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
static_assert(std::is_base_of_v<PassBuilder, InternalPassBuilder>);
static_assert(static_cast<uint8_t>(Stage::AfterFrame) == 6);
static_assert(std::is_same_v<detail::Access, render::Access> && std::is_same_v<detail::TexDesc, render::TexDesc> &&
              std::is_same_v<detail::BufferView, render::BufferView> &&
              std::is_same_v<detail::FrameConstants, render::FrameConstants>);

struct LogLine {
  AuroraLogLevel level;
  std::string module;
  std::string message;
};

std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  g_logLines.push_back({level, module != nullptr ? module : "", std::string{message, len}});
}

size_t count_logs(AuroraLogLevel level, std::string_view needle = {}, std::string_view module = {}) {
  return static_cast<size_t>(std::ranges::count_if(g_logLines, [&](const LogLine& line) {
    return line.level == level && line.message.find(needle) != std::string::npos &&
           (module.empty() || line.module == module);
  }));
}

struct External {
  std::string name;
  Stage stage = Stage::BeforeHud;
  PassKind kind = PassKind::Render;
  uint16_t order = 0;
  uint32_t paramsSize = 0;
  std::function<void(PassBuilder&)> declare;
  bool enabled = true;
  int enabledCalls = 0;
  std::vector<std::byte> enabledParams;
  std::function<void()> onEnabled;
};

void declare_external(PassBuilder& builder, void* userdata) {
  const auto& pass = *static_cast<const External*>(userdata);
  if (pass.declare) {
    pass.declare(builder);
  }
}

bool enabled_external(std::span<const std::byte> params, void* userdata) {
  auto& pass = *static_cast<External*>(userdata);
  ++pass.enabledCalls;
  pass.enabledParams.assign(params.begin(), params.end());
  if (pass.onEnabled) {
    pass.onEnabled();
  }
  return pass.enabled;
}

void encode_external(render::PassContext&, void*) {}

ExternalPassDesc desc_of(External& pass) {
  return {
      .name = pass.name.c_str(),
      .stage = pass.stage,
      .kind = pass.kind,
      .order = pass.order,
      .paramsSize = pass.paramsSize,
      .declare = declare_external,
      .enabled = enabled_external,
      .encode = encode_external,
      .userdata = &pass,
  };
}

ExternalPassDesc minimal_desc(const char* name) {
  return {.name = name, .declare = declare_external, .encode = encode_external};
}

struct Internal {
  std::string name;
  Stage stage = Stage::BeforeHud;
  PassKind kind = PassKind::Compute;
  uint16_t order = 0;
};

void declare_internal(InternalPassBuilder& builder, const PassSetup&, const void*) { builder.side_effect(); }

PassDesc internal_desc(const Internal& pass) {
  return {
      .name = pass.name,
      .stage = pass.stage,
      .kind = pass.kind,
      .order = pass.order,
      .declare = declare_internal,
      .state = &pass,
  };
}

GraphInputs test_inputs() {
  GraphInputs inputs;
  inputs.env.limits = planned::full_limits();
  inputs.env.efbSize = {1280, 720};
  inputs.env.outputSize = {1280, 720};
  inputs.env.gxFormats = planned::gx_formats_tier_p();
  inputs.env.gxUsage = planned::gx_usage_future();
  inputs.env.inspectorOpen = true;
  return inputs;
}

TexDesc efb_tex(wgpu::TextureFormat format) { return {.format = format, .sizeClass = SizeClass::Efb}; }

const CompiledNode& node(const CompiledGraph& graph, std::string_view name) {
  static const CompiledNode absent{};
  const NodeId id = graph.find_node(name);
  EXPECT_NE(id, InvalidNode) << name;
  return id != InvalidNode ? graph.nodes[id] : absent;
}

CullReason status(const CompiledGraph& graph, std::string_view name) { return node(graph, name).status; }

std::string diagnostic(const CompiledGraph& graph, std::string_view name) {
  const NodeId id = graph.find_node(name);
  for (const Diagnostic& entry : graph.diagnostics) {
    if (entry.node == id && entry.reason == graph.nodes[id].status) {
      return entry.message;
    }
  }
  return {};
}

const CompiledUse* use_of(const CompiledGraph& graph, std::string_view nodeName, std::string_view resourceName) {
  for (const CompiledUse& use : graph.node_uses(graph.find_node(nodeName))) {
    if (graph.resources[use.resource].name == resourceName) {
      return &use;
    }
  }
  return nullptr;
}

std::vector<std::string> stage_nodes(const CompiledGraph& graph, Stage stage) {
  std::vector<std::string> names;
  for (const NodeId id : graph.stages[static_cast<size_t>(stage)].nodes) {
    names.push_back(graph.nodes[id].name);
  }
  return names;
}

std::vector<std::string> stage_passes(const CompiledGraph& graph, Stage stage) {
  std::vector<std::string> names;
  for (const NodeId id : graph.stages[static_cast<size_t>(stage)].passes) {
    names.push_back(graph.nodes[id].name);
  }
  return names;
}

ptrdiff_t position(const std::vector<std::string>& names, std::string_view name) {
  const auto it = std::ranges::find(names, name);
  EXPECT_NE(it, names.end()) << name;
  return it - names.begin();
}

std::vector<std::byte> bytes(std::initializer_list<uint8_t> values) {
  std::vector<std::byte> out;
  for (const uint8_t value : values) {
    out.push_back(static_cast<std::byte>(value));
  }
  return out;
}

std::vector<std::byte> latched_bytes(PassId id, size_t count) {
  const LatchedExternalPass* latched = latched_external_pass(id);
  EXPECT_NE(latched, nullptr);
  if (latched == nullptr) {
    return {};
  }
  return {latched->params.begin(), latched->params.begin() + count};
}

void writes_gx_color(PassBuilder& b) { b.use(resource::GxColor, Access::ColorRMW); }

class RenderGraphApi : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset_external_passes();
    testing::reset_logged_diagnostics();
    g_logLines.clear();
    g_config.logCallback = capture_log;
  }

  void TearDown() override {
    g_config.logCallback = nullptr;
    testing::reset_external_passes();
  }

  External& make(std::string name, Stage stage, PassKind kind, std::function<void(PassBuilder&)> declare,
                 uint16_t order = 0) {
    return m_externals.emplace_back(External{
        .name = std::move(name),
        .stage = stage,
        .kind = kind,
        .order = order,
        .declare = std::move(declare),
    });
  }

  PassId add(std::string name, Stage stage, PassKind kind, std::function<void(PassBuilder&)> declare,
             uint16_t order = 0) {
    const PassId id = render::register_pass(desc_of(make(std::move(name), stage, kind, std::move(declare), order)));
    EXPECT_NE(id, InvalidPass);
    return id;
  }

  PassId add_params(std::string name, uint32_t paramsSize) {
    External& pass = make(std::move(name), Stage::BeforeHud, PassKind::Render, writes_gx_color);
    pass.paramsSize = paramsSize;
    const PassId id = render::register_pass(desc_of(pass));
    EXPECT_NE(id, InvalidPass);
    return id;
  }

  // Internal passes come first, as they do in the latch
  static std::shared_ptr<CompiledGraph> compile_externals(const std::vector<Internal>& internals = {},
                                                          GraphInputs base = test_inputs()) {
    const std::shared_ptr<const ExternalPassSet> set = snapshot_external_passes();
    std::vector<PassDesc> passes;
    for (const Internal& pass : internals) {
      passes.push_back(internal_desc(pass));
    }
    append_external_passes(*set, nullptr, passes);
    base.passes = passes;
    base.externals = set;
    return compile_graph(base);
  }

  std::deque<External> m_externals;
};

TEST_F(RenderGraphApi, RegisterRejectsInvalidNames) {
  const std::string tooLong = "mod." + std::string(60, 'x');
  ASSERT_EQ(tooLong.size(), 64u);
  const std::array<const char*, 8> names{nullptr, "",      "noprefix",  "Mod.pass",
                                         "mod.",  ".pass", "mod..pass", tooLong.c_str()};
  for (const char* name : names) {
    const size_t before = count_logs(LOG_WARNING, "register_pass: ");
    EXPECT_EQ(render::register_pass(minimal_desc(name)), InvalidPass) << (name != nullptr ? name : "null");
    EXPECT_EQ(count_logs(LOG_WARNING, "register_pass: "), before + 1) << (name != nullptr ? name : "null");
  }
  const std::string longest = tooLong.substr(1);
  EXPECT_NE(render::register_pass(minimal_desc(longest.c_str())), InvalidPass);
  EXPECT_NE(render::register_pass(minimal_desc("my_mod.pass_2.sub")), InvalidPass);
}

constexpr std::array<std::string_view, 38> FrozenPrefixes{
    "aurora",    "gx",    "phase",   "gbuf",     "env",   "grass", "sky",    "clouds",   "terrain", "gi",
    "ddgi",      "scene", "capture", "rt",       "hzb",   "vg",    "legacy", "depth",    "light",   "ao",
    "ssil",      "csm",   "vsm",     "sun",      "vis",   "lsm",   "refl",   "exposure", "godrays", "fog",
    "composite", "taa",   "native",  "radiance", "debug", "bvh",   "voxel",  "heap",
};

TEST_F(RenderGraphApi, RegisterRejectsReservedPrefixes) {
  for (const std::string_view prefix : FrozenPrefixes) {
    EXPECT_TRUE(reserved_prefix(prefix)) << prefix;
    const std::string name = std::string{prefix} + ".pass";
    EXPECT_EQ(render::register_pass(minimal_desc(name.c_str())), InvalidPass) << name;
  }
  EXPECT_EQ(count_logs(LOG_WARNING, "uses reserved prefix"), FrozenPrefixes.size());
}

TEST_F(RenderGraphApi, ReservedPrefixesCoverPlannedPassSet) {
  const std::vector<PassDesc> passes = planned::planned_passes();
  const std::vector<GxReadDesc> reads = planned::planned_gx_reads();
  const auto graph = compile_graph(planned::planned_inputs(planned::full_limits(), passes, reads));
  std::vector<std::string> names;
  for (const CompiledNode& compiled : graph->nodes) {
    names.push_back(compiled.name);
  }
  for (const CompiledResource& resource : graph->resources) {
    names.push_back(resource.name);
  }
  for (const ChainNode& chain : chain_nodes()) {
    names.emplace_back(chain.name);
  }
  size_t checked = 0;
  for (const std::string& name : names) {
    // An external name always has a dot, so undotted internal names cannot collide with one
    if (name.find('.') == std::string::npos) {
      continue;
    }
    const std::string_view prefix = external_prefix(name);
    if (prefix == "demo") {
      continue;
    }
    EXPECT_TRUE(reserved_prefix(prefix)) << name;
    ++checked;
  }
  EXPECT_GT(checked, 100u);
  EXPECT_FALSE(reserved_prefix("demo"));
}

TEST_F(RenderGraphApi, RegisterRejectsDuplicateLiveName) {
  ASSERT_NE(render::register_pass(minimal_desc("mod.a")), InvalidPass);
  EXPECT_EQ(render::register_pass(minimal_desc("mod.a")), InvalidPass);
  EXPECT_EQ(count_logs(LOG_WARNING, "a pass named 'mod.a' is already registered"), 1u);
  EXPECT_NE(render::register_pass(minimal_desc("mod.b")), InvalidPass);
}

TEST_F(RenderGraphApi, ReRegisterGetsNewGeneration) {
  const PassId first = render::register_pass(minimal_desc("mod.a"));
  ASSERT_NE(first, InvalidPass);
  render::unregister_pass(first);
  const PassId second = render::register_pass(minimal_desc("mod.a"));
  ASSERT_NE(second, InvalidPass);
  EXPECT_NE(second, first);
  EXPECT_EQ(second & 0xFFFFFFFFu, first & 0xFFFFFFFFu);
  EXPECT_FALSE(external_pass_alive(first));
  EXPECT_TRUE(external_pass_alive(second));
  render::unregister_pass(first);
  EXPECT_TRUE(external_pass_alive(second));
}

TEST_F(RenderGraphApi, RegisterRejectsAfterFrameAndNative) {
  ExternalPassDesc desc = minimal_desc("mod.pass");
  desc.stage = Stage::AfterFrame;
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  desc.stage = static_cast<Stage>(9);
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  desc.stage = Stage::FrameEnd;
  desc.kind = PassKind::Native;
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  desc.kind = static_cast<PassKind>(9);
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  EXPECT_EQ(count_logs(LOG_WARNING, "register_pass: 'mod.pass'"), 4u);
  desc.kind = PassKind::Encoder;
  EXPECT_NE(render::register_pass(desc), InvalidPass);
}

TEST_F(RenderGraphApi, RegisterRejectsMissingCallbacks) {
  ExternalPassDesc desc = minimal_desc("mod.pass");
  desc.declare = nullptr;
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  desc = minimal_desc("mod.pass");
  desc.encode = nullptr;
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  EXPECT_EQ(count_logs(LOG_WARNING, "has no declare callback"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "has no encode callback"), 1u);
}

TEST_F(RenderGraphApi, RegisterRejectsOversizedParams) {
  ExternalPassDesc desc = minimal_desc("mod.pass");
  desc.paramsSize = MaxPassParamsSize + 1;
  EXPECT_EQ(render::register_pass(desc), InvalidPass);
  EXPECT_EQ(count_logs(LOG_WARNING, "has 129 bytes of params, more than 128"), 1u);
  desc.paramsSize = MaxPassParamsSize;
  EXPECT_NE(render::register_pass(desc), InvalidPass);
}

TEST_F(RenderGraphApi, RegisterRejectsBeyondCapacity) {
  std::vector<std::string> names;
  for (size_t i = 0; i <= MaxExternalPasses; ++i) {
    names.push_back("mod.p" + std::to_string(i));
  }
  std::vector<PassId> ids;
  for (size_t i = 0; i < MaxExternalPasses; ++i) {
    ids.push_back(render::register_pass(minimal_desc(names[i].c_str())));
    ASSERT_NE(ids.back(), InvalidPass) << i;
  }
  EXPECT_EQ(render::register_pass(minimal_desc(names.back().c_str())), InvalidPass);
  EXPECT_EQ(count_logs(LOG_WARNING, "would exceed 64 registered passes"), 1u);
  render::unregister_pass(ids[10]);
  EXPECT_NE(render::register_pass(minimal_desc(names.back().c_str())), InvalidPass);
}

TEST_F(RenderGraphApi, UnregisterStopsEncodeImmediately) {
  const PassId id = render::register_pass(minimal_desc("mod.pass"));
  ASSERT_NE(id, InvalidPass);
  EXPECT_TRUE(external_pass_alive(id));
  const uint64_t epoch = external_pass_epoch();
  render::unregister_pass(id);
  EXPECT_FALSE(external_pass_alive(id));
  EXPECT_GT(external_pass_epoch(), epoch);
}

TEST_F(RenderGraphApi, InvalidIdsAreBoundsChecked) {
  ASSERT_NE(render::register_pass(minimal_desc("mod.pass")), InvalidPass);
  latch_external_passes();
  const uint64_t epoch = external_pass_epoch();
  const std::array<PassId, 4> ids{InvalidPass, (1ull << 32) | 64, (1ull << 32) | 1000, UINT64_MAX};
  const std::array<std::byte, 4> payload{};
  for (const PassId id : ids) {
    EXPECT_FALSE(external_pass_alive(id)) << id;
    EXPECT_FALSE(render::set_pass_params(id, payload.data(), payload.size())) << id;
    render::unregister_pass(id);
    EXPECT_EQ(latched_external_pass(id), nullptr) << id;
  }
  EXPECT_EQ(external_pass_epoch(), epoch);
}

TEST_F(RenderGraphApi, ParamsApplyFromNextLatch) {
  const PassId id = add_params("mod.pass", 4);
  const auto a = bytes({1, 2, 3, 4});
  const auto b = bytes({5, 6, 7, 8});
  ASSERT_TRUE(render::set_pass_params(id, a.data(), a.size()));
  latch_external_passes();
  EXPECT_EQ(latched_bytes(id, 4), a);
  ASSERT_TRUE(render::set_pass_params(id, b.data(), b.size()));
  EXPECT_EQ(latched_bytes(id, 4), a);
  latch_external_passes();
  EXPECT_EQ(latched_bytes(id, 4), b);
}

TEST_F(RenderGraphApi, ParamsZeroFilledAndNeverSetIsZero) {
  const PassId id = add_params("mod.pass", 16);
  const PassId unset = add_params("mod.unset", 16);
  const std::vector<std::byte> full(16, std::byte{0xAB});
  ASSERT_TRUE(render::set_pass_params(id, full.data(), full.size()));
  latch_external_passes();
  EXPECT_EQ(latched_bytes(id, 16), full);
  const auto shorter = bytes({1, 2, 3, 4});
  ASSERT_TRUE(render::set_pass_params(id, shorter.data(), shorter.size()));
  latch_external_passes();
  auto expected = shorter;
  expected.resize(16);
  EXPECT_EQ(latched_bytes(id, 16), expected);
  EXPECT_EQ(latched_bytes(unset, 16), std::vector<std::byte>(16));
}

TEST_F(RenderGraphApi, ParamsRejectsOversizeAndDeadId) {
  const PassId id = add_params("mod.pass", 16);
  const std::vector<std::byte> payload(17);
  EXPECT_FALSE(render::set_pass_params(id, payload.data(), payload.size()));
  EXPECT_FALSE(render::set_pass_params(id, payload.data(), payload.size()));
  EXPECT_EQ(count_logs(LOG_WARNING, "'mod.pass' takes at most 16 bytes of params, not 17"), 1u);
  EXPECT_FALSE(render::set_pass_params(id, nullptr, 4));
  EXPECT_EQ(count_logs(LOG_WARNING, "'mod.pass' got no data for 4 bytes"), 1u);
  EXPECT_TRUE(render::set_pass_params(id, nullptr, 0));

  render::unregister_pass(id);
  EXPECT_FALSE(render::set_pass_params(id, payload.data(), 4));
  EXPECT_FALSE(render::set_pass_params(id, payload.data(), 4));
  EXPECT_EQ(count_logs(LOG_WARNING, "is not registered"), 1u);
}

TEST_F(RenderGraphApi, EnabledRunsOncePerLatchWithLatchedParams) {
  const PassId id = add_params("mod.pass", 4);
  External& pass = m_externals.back();
  const auto a = bytes({9, 8, 7, 6});
  ASSERT_TRUE(render::set_pass_params(id, a.data(), a.size()));
  for (int i = 0; i < 3; ++i) {
    latch_external_passes();
  }
  EXPECT_EQ(pass.enabledCalls, 3);
  EXPECT_EQ(pass.enabledParams, a);
  const auto set = snapshot_external_passes();
  ASSERT_EQ(set->nodes.size(), 1u);
  EXPECT_TRUE(external_enabled(StageData{}, &set->nodes[0]));

  pass.enabled = false;
  latch_external_passes();
  EXPECT_EQ(pass.enabledCalls, 4);
  EXPECT_FALSE(external_enabled(StageData{}, &set->nodes[0]));
  EXPECT_EQ(count_logs(LOG_WARNING), 0u);
}

TEST_F(RenderGraphApi, EnabledCallbackMaySetParams) {
  const PassId id = add_params("mod.pass", 4);
  External& pass = m_externals.back();
  const auto a = bytes({1, 1, 1, 1});
  const auto b = bytes({2, 2, 2, 2});
  pass.onEnabled = [&] { EXPECT_TRUE(render::set_pass_params(id, b.data(), b.size())); };
  ASSERT_TRUE(render::set_pass_params(id, a.data(), a.size()));
  latch_external_passes();
  EXPECT_EQ(latched_bytes(id, 4), a);
  latch_external_passes();
  EXPECT_EQ(latched_bytes(id, 4), b);
}

TEST_F(RenderGraphApi, EnabledCallbackUnregisteringLaterPassSkipsIt) {
  add_params("mod.first", 0);
  External& first = m_externals.back();
  const PassId secondId = add_params("mod.second", 0);
  External& second = m_externals.back();
  first.enabled = false;
  first.onEnabled = [&] { render::unregister_pass(secondId); };
  latch_external_passes();
  EXPECT_EQ(first.enabledCalls, 1);
  EXPECT_EQ(second.enabledCalls, 0);
  const LatchedExternalPass* latched = latched_external_pass(secondId);
  ASSERT_NE(latched, nullptr);
  EXPECT_FALSE(latched->enabled);
  EXPECT_TRUE(untraceable_external_pass().empty());

  first.onEnabled = {};
  latch_external_passes();
  EXPECT_EQ(second.enabledCalls, 0);
  EXPECT_EQ(latched_external_pass(secondId), nullptr);
}

TEST_F(RenderGraphApi, DisabledExternalsAreNotUntraceable) {
  add_params("mod.first", 0);
  External& first = m_externals.back();
  add_params("mod.second", 0);
  External& second = m_externals.back();
  first.enabled = false;
  second.enabled = false;
  latch_external_passes();
  EXPECT_TRUE(untraceable_external_pass().empty());
  second.enabled = true;
  latch_external_passes();
  EXPECT_EQ(untraceable_external_pass(), "mod.second");
  first.enabled = true;
  latch_external_passes();
  EXPECT_EQ(untraceable_external_pass(), "mod.first");
}

TEST_F(RenderGraphApi, ExternalDeclarationsCannotImportOrReadHistory) {
  // The callback receives an internal builder under its public base, so a cast reaches the internal members
  add("mod.prev", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    static_cast<InternalPassBuilder&>(b).use_prev("mod.hist");
    b.side_effect();
  });
  add("mod.import", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    static_cast<InternalPassBuilder&>(b).import_persistent("mod.buf", {1}, Access::StorageRead);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.prev"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.prev"),
            "render graph: 'mod.prev' may only create, keep history of or use 'mod.hist'; disabled");
  EXPECT_EQ(status(*graph, "mod.import"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.import"),
            "render graph: 'mod.import' may only create, keep history of or use 'mod.buf'; disabled");
}

TEST_F(RenderGraphApi, ExternalNodesAreOptionalAndRunWithoutMarker) {
  const std::vector<Internal> internals{{.name = "internal.a"}, {.name = "internal.b"}};
  const PassId id = add("mod.pass", Stage::OpaqueDone, PassKind::Render, writes_gx_color);
  const auto graph = compile_externals(internals);
  const CompiledNode& external = node(*graph, "mod.pass");
  EXPECT_TRUE(external.external);
  EXPECT_TRUE(external.runsWithoutMarker);
  EXPECT_EQ(external.cls, PassClass::Optional);
  EXPECT_EQ(external.id, id);
  EXPECT_EQ(external.status, CullReason::None);
  EXPECT_GT(external.registration, node(*graph, "internal.a").registration);
  EXPECT_GT(external.registration, node(*graph, "internal.b").registration);
  EXPECT_FALSE(node(*graph, "internal.a").external);
  ASSERT_NE(graph->externals, nullptr);
  EXPECT_EQ(graph->externals->nodes.size(), 1u);
}

TEST_F(RenderGraphApi, DefaultAnchorIsStageEnd) {
  add("mod.hud", Stage::BeforeHud, PassKind::Render, writes_gx_color);
  add("mod.opaque", Stage::OpaqueDone, PassKind::Render, writes_gx_color);
  const auto graph = compile_externals();
  ASSERT_EQ(status(*graph, "mod.hud"), CullReason::None);
  ASSERT_EQ(status(*graph, "mod.opaque"), CullReason::None);
  const auto hud = stage_nodes(*graph, Stage::BeforeHud);
  EXPECT_GT(position(hud, "mod.hud"), position(hud, phase::BeforeHud));
  const auto opaque = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_GT(position(opaque, "mod.opaque"), position(opaque, phase::Overlay));
}

TEST_F(RenderGraphApi, KnownAnchorOrdersPass) {
  add("mod.composite", Stage::OpaqueDone, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after(phase::Composite);
  });
  add("mod.pre_taa", Stage::PrePost, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.before(phase::Taa);
  });
  const auto graph = compile_externals();
  const auto opaque = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_GT(position(opaque, "mod.composite"), position(opaque, phase::Composite));
  EXPECT_LT(position(opaque, "mod.composite"), position(opaque, phase::Overlay));
  const auto post = stage_nodes(*graph, Stage::PrePost);
  EXPECT_LT(position(post, "mod.pre_taa"), position(post, phase::Taa));
  EXPECT_EQ(count_logs(LOG_WARNING), 0u);
}

TEST_F(RenderGraphApi, UnknownAnchorFallsBackToStageEnd) {
  add("mod.pass", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after("phase.nope");
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.pass"), CullReason::None);
  const auto hud = stage_nodes(*graph, Stage::BeforeHud);
  EXPECT_GT(position(hud, "mod.pass"), position(hud, phase::BeforeHud));
}

TEST_F(RenderGraphApi, UnknownAnchorLogsOncePerPassAcrossCompiles) {
  const auto declare = [](PassBuilder& b) {
    writes_gx_color(b);
    b.after("phase.nope");
  };
  const PassId id = add("mod.pass", Stage::BeforeHud, PassKind::Render, declare);
  for (int i = 0; i < 3; ++i) {
    compile_externals();
  }
  render::unregister_pass(id);
  add("mod.pass", Stage::BeforeHud, PassKind::Render, declare);
  compile_externals();
  constexpr std::string_view Text = "render graph: 'mod.pass' has unknown anchor 'phase.nope' in BeforeHud; ignored";
  EXPECT_EQ(count_logs(LOG_WARNING, Text, "aurora::render::graph"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "has unknown anchor"), 1u);

  add("mod.other", Stage::BeforeHud, PassKind::Render, declare);
  compile_externals();
  compile_externals();
  EXPECT_EQ(count_logs(LOG_WARNING, "has unknown anchor", "aurora::render::graph"), 2u);
}

TEST_F(RenderGraphApi, CrossStageAnchorFallsBackLikeInternal) {
  add("mod.early", Stage::OpaqueDone, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after(phase::Taa);
  });
  add("mod.late", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after(phase::Composite);
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.early"), CullReason::None);
  EXPECT_EQ(status(*graph, "mod.late"), CullReason::None);
  const auto opaque = stage_nodes(*graph, Stage::OpaqueDone);
  EXPECT_GT(position(opaque, "mod.early"), position(opaque, phase::Overlay));
  const auto hud = stage_nodes(*graph, Stage::BeforeHud);
  EXPECT_GT(position(hud, "mod.late"), position(hud, phase::BeforeHud));
  EXPECT_EQ(count_logs(LOG_WARNING, "'mod.early' has unknown anchor 'phase.taa' in OpaqueDone"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "'mod.late' has unknown anchor 'phase.composite' in BeforeHud"), 1u);
}

TEST_F(RenderGraphApi, CycleBetweenExternalsDisablesTheLaterOne) {
  const std::vector<Internal> internals{{.name = "internal.pass"}};
  add("mod.a", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after("mod.b");
  });
  add("mod.b", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    writes_gx_color(b);
    b.after("mod.a");
  });
  const auto graph = compile_externals(internals);
  EXPECT_EQ(status(*graph, "mod.a"), CullReason::None);
  EXPECT_EQ(status(*graph, "mod.b"), CullReason::Cycle);
  EXPECT_EQ(status(*graph, "internal.pass"), CullReason::None);
  const auto cycleLines = std::ranges::count_if(g_logLines, [](const LogLine& line) {
    return line.level == LOG_ERROR && line.message.find("cycle") != std::string::npos &&
           line.message.find("'mod.a'") != std::string::npos && line.message.find("'mod.b'") != std::string::npos;
  });
  EXPECT_EQ(cycleLines, 1);
}

TEST_F(RenderGraphApi, ContractRejectsGxDepthWriteAndGbufWrites) {
  add("mod.depth", Stage::BeforeHud, PassKind::Render,
      [](PassBuilder& b) { b.use(resource::GxDepth, Access::DepthWrite); });
  add("mod.fog", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::Fog, Access::StorageWrite);
    b.side_effect();
  });
  add("mod.surface", Stage::BeforeHud, PassKind::Render,
      [](PassBuilder& b) { b.use(resource::Surface, Access::ColorRMW); });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.depth"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.depth"),
            "render graph: 'mod.depth' DepthWrite is not allowed on 'gx.depth'; disabled");
  EXPECT_EQ(status(*graph, "mod.fog"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.fog"),
            "render graph: 'mod.fog' StorageWrite is not allowed on 'gbuf.fog'; disabled");
  EXPECT_EQ(status(*graph, "mod.surface"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.surface"),
            "render graph: 'mod.surface' ColorRMW is not allowed on 'gbuf.surface'; disabled");
  EXPECT_EQ(count_logs(LOG_ERROR, "is not allowed on"), 3u);
}

TEST_F(RenderGraphApi, SecondUseOfOneResourceRejected) {
  add("mod.both", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::Sampled);
    b.use(resource::GxColor, Access::ColorRMW);
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.both"), CullReason::InvalidDeclaration);
  EXPECT_NE(diagnostic(*graph, "mod.both").find("uses 'gx.color' twice"), std::string::npos);
}

TEST_F(RenderGraphApi, UnknownAndCrossPrefixResourcesRejected) {
  add("mod.a", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use("other.tex", Access::Sampled);
    b.side_effect();
  });
  add("mod.phase", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(phase::Ao, Access::Sampled);
    b.side_effect();
  });
  add("mod.bad", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use("mod.Bad", Access::Sampled);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.a"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.a"), "render graph: 'mod.a' cannot use 'other.tex'; disabled");
  EXPECT_EQ(status(*graph, "mod.phase"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.phase"), "render graph: 'mod.phase' cannot use 'phase.ao'; disabled");
  EXPECT_EQ(status(*graph, "mod.bad"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.bad"),
            "render graph: 'mod.bad' resource 'mod.Bad' does not follow the external name rules; disabled");
}

TEST_F(RenderGraphApi, SamePrefixChainAllowed) {
  add("mod.a", Stage::BeforeHud, PassKind::Compute,
      [](PassBuilder& b) { b.create("mod.tex", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite); });
  add("mod.b", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use("mod.tex", Access::Sampled);
    b.side_effect();
  });
  add("mod.c", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use("mod.none", Access::Sampled);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.a"), CullReason::None);
  EXPECT_EQ(status(*graph, "mod.b"), CullReason::None);
  EXPECT_EQ(status(*graph, "mod.c"), CullReason::MissingInput);
  const auto passes = stage_passes(*graph, Stage::BeforeHud);
  EXPECT_LT(position(passes, "mod.a"), position(passes, "mod.b"));
  const CompiledUse* read = use_of(*graph, "mod.b", "mod.tex");
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(read->producer, graph->find_node("mod.a"));
}

TEST_F(RenderGraphApi, CreatedNamesMustUseThePrefix) {
  add("mod.a", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.create("other.tex", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
    b.side_effect();
  });
  add("mod.b", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.create("mod.tex", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::Sampled);
    b.side_effect();
  });
  add("mod.c", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.create("mod.own", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
    b.use("mod.own", Access::Sampled);
    b.side_effect();
  });
  add("mod.d", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.history("gx.history", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.a"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.a"), "render graph: 'mod.a' resource 'other.tex' must use prefix 'mod'; disabled");
  EXPECT_EQ(status(*graph, "mod.b"), CullReason::InvalidDeclaration);
  EXPECT_NE(diagnostic(*graph, "mod.b").find("not a plain write"), std::string::npos);
  EXPECT_EQ(status(*graph, "mod.c"), CullReason::InvalidDeclaration);
  EXPECT_NE(diagnostic(*graph, "mod.c").find("uses 'mod.own' twice"), std::string::npos);
  EXPECT_EQ(status(*graph, "mod.d"), CullReason::InvalidDeclaration);
  EXPECT_NE(diagnostic(*graph, "mod.d").find("must use prefix 'mod'"), std::string::npos);
}

TEST_F(RenderGraphApi, KindAccessTable) {
  add("mod.compute_rmw", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::ColorRMW);
    b.side_effect();
  });
  add("mod.copy_storage", Stage::BeforeHud, PassKind::Copy, [](PassBuilder& b) {
    b.create("mod.t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
    b.side_effect();
  });
  add("mod.compute_copy", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::CopySrc);
    b.side_effect();
  });
  add("mod.no_attachment", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::Sampled);
    b.side_effect();
  });
  const auto graph = compile_externals();
  for (const std::string_view name : {"mod.compute_rmw", "mod.copy_storage", "mod.compute_copy", "mod.no_attachment"}) {
    EXPECT_EQ(status(*graph, name), CullReason::InvalidDeclaration) << name;
  }
  EXPECT_NE(diagnostic(*graph, "mod.no_attachment").find("render pass without attachments"), std::string::npos);
}

TEST_F(RenderGraphApi, AbsentContractResourceAppliesMissing) {
  add("mod.absent", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::AoFinal, Access::Sampled);
    b.side_effect();
  });
  add("mod.white", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::AoFinal, Access::Sampled, Missing::BindDummyWhite);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.absent"), CullReason::MissingInput);
  EXPECT_NE(diagnostic(*graph, "mod.absent").find("ao.final"), std::string::npos);
  EXPECT_EQ(status(*graph, "mod.white"), CullReason::None);
  const CompiledUse* read = use_of(*graph, "mod.white", "ao.final");
  ASSERT_NE(read, nullptr);
  EXPECT_TRUE(read->staticDummy);
  ASSERT_TRUE(read->hasFallback);
  EXPECT_EQ(read->fallback.color, DummyColor::White);
}

TEST_F(RenderGraphApi, NoConsumerIsCulled) {
  add("mod.lonely", Stage::BeforeHud, PassKind::Compute,
      [](PassBuilder& b) { b.create("mod.t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite); });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.lonely"), CullReason::NoConsumer);
  EXPECT_FALSE(diagnostic(*graph, "mod.lonely").empty());
  EXPECT_EQ(count_logs(LOG_DEBUG, "nothing reads what 'mod.lonely' writes"), 1u);
}

TEST_F(RenderGraphApi, SideEffectKeepsPass) {
  add("mod.kept", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.create("mod.t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::StorageWrite);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.kept"), CullReason::None);
}

TEST_F(RenderGraphApi, OrderThenRegistrationBreaksTies) {
  const std::vector<Internal> internals{{.name = "internal.first"}, {.name = "internal.late", .order = 1}};
  const auto sink = [](PassBuilder& b) { b.side_effect(); };
  add("mod.second", Stage::BeforeHud, PassKind::Compute, sink);
  add("mod.third", Stage::BeforeHud, PassKind::Compute, sink);
  add("mod.late", Stage::BeforeHud, PassKind::Compute, sink, 1);
  const auto graph = compile_externals(internals);
  EXPECT_EQ(stage_passes(*graph, Stage::BeforeHud),
            (std::vector<std::string>{"internal.first", "mod.second", "mod.third", "internal.late", "mod.late"}));
}

TEST_F(RenderGraphApi, MsaaKeepsGxColorForExternals) {
  GraphInputs inputs = test_inputs();
  inputs.env.gxSampleCount = 4;
  add("mod.rmw", Stage::BeforeHud, PassKind::Render, writes_gx_color);
  add("mod.sample", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::Sampled);
    b.side_effect();
  });
  add("mod.normal_rmw", Stage::BeforeHud, PassKind::Render,
      [](PassBuilder& b) { b.use(resource::GxNormal, Access::ColorRMW); });
  add("mod.normal_read", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxNormal, Access::Sampled, Missing::BindDummyBlack);
    b.side_effect();
  });
  const auto graph = compile_externals({}, inputs);
  EXPECT_EQ(status(*graph, "mod.rmw"), CullReason::None);
  EXPECT_EQ(status(*graph, "mod.sample"), CullReason::None);
  // Only Sampled is in the gx.normal contract, so the write fails before availability is judged
  EXPECT_EQ(status(*graph, "mod.normal_rmw"), CullReason::InvalidDeclaration);
  EXPECT_EQ(status(*graph, "mod.normal_read"), CullReason::None);
  const CompiledUse* normal = use_of(*graph, "mod.normal_read", "gx.normal");
  ASSERT_NE(normal, nullptr);
  EXPECT_TRUE(normal->staticDummy);
  const BindingPlan& plan = graph->bindingPlans[graph->find_node("mod.rmw")];
  ASSERT_EQ(plan.colors.size(), 1u);
  EXPECT_EQ(plan.colors[0].samples, 4u);
}

TEST_F(RenderGraphApi, AttachmentSampleCountMismatchRejected) {
  const auto declare = [](PassBuilder& b) {
    writes_gx_color(b);
    b.create("mod.t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::ColorWrite);
  };
  add("mod.mixed", Stage::BeforeHud, PassKind::Render, declare);
  GraphInputs msaa = test_inputs();
  msaa.env.gxSampleCount = 4;
  EXPECT_EQ(status(*compile_externals({}, msaa), "mod.mixed"), CullReason::SampleCountMismatch);
  EXPECT_EQ(status(*compile_externals(), "mod.mixed"), CullReason::None);
}

TEST_F(RenderGraphApi, PublicPhaseNamesMatchPhaseTable) {
  struct Row {
    std::string_view name;
    Stage stage;
  };
  const std::array<Row, 21> rows{{
      {phase::ScenePrep, Stage::SceneBegin},
      {phase::Capture, Stage::OpaqueDone},
      {phase::Geometry, Stage::OpaqueDone},
      {phase::Prepare, Stage::OpaqueDone},
      {phase::Ao, Stage::OpaqueDone},
      {phase::CloudShadows, Stage::OpaqueDone},
      {phase::Shadows, Stage::OpaqueDone},
      {phase::Gi, Stage::OpaqueDone},
      {phase::Lighting, Stage::OpaqueDone},
      {phase::Reflections, Stage::OpaqueDone},
      {phase::Exposure, Stage::OpaqueDone},
      {phase::Sky, Stage::OpaqueDone},
      {phase::Volumetrics, Stage::OpaqueDone},
      {phase::Composite, Stage::OpaqueDone},
      {phase::Overlay, Stage::OpaqueDone},
      {phase::PreTaa, Stage::PrePost},
      {phase::Taa, Stage::PrePost},
      {phase::PostTaa, Stage::PrePost},
      {phase::SceneEnd, Stage::SceneEnd},
      {phase::BeforeHud, Stage::BeforeHud},
      {phase::FrameEnd, Stage::FrameEnd},
  }};
  for (const Row& row : rows) {
    EXPECT_EQ(chain_node_stage(row.name), row.stage) << row.name;
  }
  size_t markers = 0;
  for (const ChainNode& chain : chain_nodes()) {
    if (chain.kind == ChainKind::Gx || chain.stage == Stage::AfterFrame) {
      continue;
    }
    ++markers;
    EXPECT_TRUE(std::ranges::any_of(rows, [&](const Row& row) { return row.name == chain.name; })) << chain.name;
  }
  EXPECT_EQ(markers, rows.size());
}

TEST_F(RenderGraphApi, SharedVocabularyCompiles) {
  EXPECT_EQ(detail::AllMips, render::AllMips);
  EXPECT_EQ(detail::FrameFlagReversedZ, render::FrameFlagReversedZ);
  EXPECT_EQ(detail::FrameFlagMultisampled, render::FrameFlagMultisampled);
  EXPECT_EQ(static_cast<uint32_t>(detail::FrameBinding::Params), 1u);
  const TexRef ref{};
  EXPECT_FALSE(ref.valid());
  EXPECT_TRUE((BufRef{0}.valid()));
  EXPECT_EQ(stage_name(Stage::BeforeHud), "BeforeHud");
}

TEST_F(RenderGraphApi, ExternalFlagReachesInspection) {
  const std::vector<Internal> internals{{.name = "internal.pass"}};
  add("mod.pass", Stage::BeforeHud, PassKind::Render, writes_gx_color);
  const auto graph = compile_externals(internals);
  const debug::GraphInspection inspection = inspect(*graph);
  const auto find = [&](std::string_view name) {
    return std::ranges::find(inspection.passes, name, &debug::InspectPass::name);
  };
  ASSERT_NE(find("mod.pass"), inspection.passes.end());
  ASSERT_NE(find("internal.pass"), inspection.passes.end());
  EXPECT_TRUE(find("mod.pass")->external);
  EXPECT_FALSE(find("internal.pass")->external);
  EXPECT_NE(debug::format_graph_json(inspection).find("\"external\": true"), std::string::npos);
}

GraphInputs depth_inputs(DepthSource source, uint32_t samples = 1) {
  GraphInputs inputs = test_inputs();
  inputs.env.gxSampleCount = samples;
  inputs.sceneDepthSource = source;
  return inputs;
}

void samples_gx_depth(PassBuilder& b, Missing missing = Missing::DisablePass) {
  b.use(resource::GxDepth, Access::Sampled, missing);
  b.side_effect();
}

void reads_gx_depth_attachment(PassBuilder& b) {
  b.use(resource::GxDepth, Access::DepthRead);
  b.side_effect();
}

void declare_internal_depth_write(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use(resource::GxDepth, Access::DepthWrite);
}

TEST_F(RenderGraphApi, UnavailableSceneDepthAppliesMissing) {
  add("mod.disabled", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { samples_gx_depth(b); });
  add("mod.black", Stage::BeforeHud, PassKind::Compute,
      [](PassBuilder& b) { samples_gx_depth(b, Missing::BindDummyBlack); });
  add("mod.read", Stage::BeforeHud, PassKind::Render, reads_gx_depth_attachment);
  const auto graph = compile_externals({}, depth_inputs(DepthSource::Unavailable));
  EXPECT_EQ(graph->sceneDepthSource, DepthSource::Unavailable);
  EXPECT_EQ(status(*graph, "mod.disabled"), CullReason::MissingInput);
  EXPECT_NE(diagnostic(*graph, "mod.disabled").find("'gx.depth'"), std::string::npos);
  EXPECT_EQ(status(*graph, "mod.black"), CullReason::None);
  const CompiledUse* black = use_of(*graph, "mod.black", "gx.depth");
  ASSERT_NE(black, nullptr);
  EXPECT_TRUE(black->staticDummy);
  ASSERT_TRUE(black->hasFallback);
  EXPECT_EQ(black->fallback, (DummyKey{.color = DummyColor::Black,
                                       .dimension = wgpu::TextureViewDimension::e2D,
                                       .sampleType = wgpu::TextureSampleType::UnfilterableFloat}));
  EXPECT_EQ(status(*graph, "mod.read"), CullReason::None);
}

TEST_F(RenderGraphApi, MsaaGxDepthFollowsSource) {
  add("mod.sample", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { samples_gx_depth(b); });
  add("mod.read", Stage::BeforeHud, PassKind::Render, reads_gx_depth_attachment);
  const auto compile_at = [](DepthSource source) {
    const std::shared_ptr<const ExternalPassSet> set = snapshot_external_passes();
    std::vector<PassDesc> passes{{
        .name = "internal.depth_write",
        .stage = Stage::BeforeHud,
        .kind = PassKind::Render,
        .declare = declare_internal_depth_write,
    }};
    append_external_passes(*set, nullptr, passes);
    GraphInputs inputs = depth_inputs(source, 4);
    inputs.passes = passes;
    inputs.externals = set;
    return compile_graph(inputs);
  };
  const auto copy = compile_at(DepthSource::Copy);
  EXPECT_EQ(status(*copy, "mod.sample"), CullReason::None);
  EXPECT_EQ(status(*copy, "mod.read"), CullReason::None);
  EXPECT_EQ(status(*copy, "internal.depth_write"), CullReason::MissingTarget);
  const BindingPlan& plan = copy->bindingPlans[copy->find_node("mod.read")];
  ASSERT_TRUE(plan.depth.has_value());
  EXPECT_EQ(plan.depth->access, Access::DepthRead);
  EXPECT_EQ(plan.depth->samples, 4u);

  const auto unavailable = compile_at(DepthSource::Unavailable);
  EXPECT_EQ(status(*unavailable, "mod.sample"), CullReason::MissingInput);
  EXPECT_EQ(status(*unavailable, "mod.read"), CullReason::None);
  EXPECT_EQ(status(*unavailable, "internal.depth_write"), CullReason::MissingTarget);
}

TEST_F(RenderGraphApi, SampledAndDepthReadOnOnePassRejected) {
  add("mod.both", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    b.use(resource::GxDepth, Access::Sampled);
    b.use(resource::GxDepth, Access::DepthRead);
  });
  const auto graph = compile_externals({}, depth_inputs(DepthSource::Direct));
  EXPECT_EQ(status(*graph, "mod.both"), CullReason::InvalidDeclaration);
  EXPECT_NE(diagnostic(*graph, "mod.both").find("uses 'gx.depth' twice"), std::string::npos);
}

TEST_F(RenderGraphApi, DepthReadAttachmentSampleCountMismatchRejected) {
  add("mod.mixed", Stage::BeforeHud, PassKind::Render, [](PassBuilder& b) {
    b.use(resource::GxDepth, Access::DepthRead);
    b.create("mod.t", efb_tex(wgpu::TextureFormat::RGBA8Unorm), Access::ColorWrite);
    b.side_effect();
  });
  EXPECT_EQ(status(*compile_externals({}, depth_inputs(DepthSource::Copy, 4)), "mod.mixed"),
            CullReason::SampleCountMismatch);
  EXPECT_EQ(status(*compile_externals({}, depth_inputs(DepthSource::Direct)), "mod.mixed"), CullReason::None);
}

TEST_F(RenderGraphApi, SceneDepthChoiceLoggedOnce) {
  const auto choices = [] {
    return std::ranges::count_if(g_logLines, [](const LogLine& line) {
      return line.level == LOG_INFO && line.module == "aurora::render::graph" &&
             line.message.starts_with("Scene depth for render passes: ");
    });
  };
  const auto lines = [](const std::string& text) {
    return std::ranges::count_if(g_logLines, [&](const LogLine& line) {
      return line.level == LOG_INFO && line.module == "aurora::render::graph" && line.message == text;
    });
  };
  constexpr auto Depth = wgpu::TextureFormat::Depth32Float;
  const PassId reader =
      add("mod.depth", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { samples_gx_depth(b); });
  compile_externals({}, depth_inputs(DepthSource::Direct));
  EXPECT_EQ(lines(scene_depth_choice_text(DepthSource::Direct, 1, true, Depth)), 1);
  EXPECT_EQ(choices(), 1);
  compile_externals({}, depth_inputs(DepthSource::Direct));
  EXPECT_EQ(choices(), 1);
  compile_externals({}, depth_inputs(DepthSource::Copy, 4));
  EXPECT_EQ(lines(scene_depth_choice_text(DepthSource::Copy, 4, true, Depth)), 1);
  EXPECT_EQ(choices(), 2);

  render::unregister_pass(reader);
  add("mod.color", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) {
    b.use(resource::GxColor, Access::Sampled);
    b.side_effect();
  });
  compile_externals({}, depth_inputs(DepthSource::Unavailable));
  EXPECT_EQ(choices(), 2);

  // The fixture's reset clears the remembered choice, so the next depth reader logs again
  testing::reset_logged_diagnostics();
  add("mod.depth2", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { samples_gx_depth(b); });
  GraphInputs compat = depth_inputs(DepthSource::Unavailable);
  compat.env.limits = planned::compat_limits();
  compile_externals({}, compat);
  EXPECT_EQ(lines("Scene depth for render passes: unavailable (compatibility mode)"), 1);
  EXPECT_EQ(choices(), 3);
}

TEST_F(RenderGraphApi, StageSamplerMaskTracksActivePasses) {
  add("mod.plain", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { b.side_effect(); });
  add("mod.depth", Stage::BeforeHud, PassKind::Compute, [](PassBuilder& b) { samples_gx_depth(b); });
  add("mod.dummy", Stage::BeforeHud, PassKind::Compute,
      [](PassBuilder& b) { samples_gx_depth(b, Missing::BindDummyBlack); });
  const auto graph = compile_externals({}, depth_inputs(DepthSource::Direct));
  ASSERT_EQ(stage_passes(*graph, Stage::BeforeHud).size(), 3u);
  const FrameState state{.graph = graph};
  const auto samples = [&](Stage stage, std::initializer_list<std::string_view> names) {
    StageData data{.stage = stage, .frame = &state, .passCount = 3, .enabled = {0}};
    for (const std::string_view name : names) {
      data.enabled[0] |= uint64_t{1} << node(*graph, name).programIndex;
    }
    return stage_samples_scene_depth(data);
  };
  EXPECT_TRUE(samples(Stage::BeforeHud, {"mod.plain", "mod.depth", "mod.dummy"}));
  EXPECT_TRUE(samples(Stage::BeforeHud, {"mod.depth"}));
  EXPECT_TRUE(samples(Stage::BeforeHud, {"mod.dummy"}));
  EXPECT_FALSE(samples(Stage::BeforeHud, {"mod.plain"}));
  EXPECT_FALSE(samples(Stage::BeforeHud, {}));
  EXPECT_FALSE(samples(Stage::FrameEnd, {"mod.plain", "mod.depth", "mod.dummy"}));
  EXPECT_FALSE(stage_samples_scene_depth(StageData{.stage = Stage::BeforeHud}));
  EXPECT_EQ(stage_scene_depth_source(StageData{.stage = Stage::BeforeHud, .frame = &state}), DepthSource::Direct);
  EXPECT_EQ(stage_scene_depth_source(StageData{.stage = Stage::BeforeHud}), DepthSource::Unavailable);

  // Without a source the reader is disabled and the dummy reader binds a static dummy, so neither asks for depth
  const auto unavailable = compile_externals({}, depth_inputs(DepthSource::Unavailable));
  const auto& bits = unavailable->sceneDepthSamplers[static_cast<size_t>(Stage::BeforeHud)];
  EXPECT_TRUE(std::ranges::all_of(bits, [](uint64_t word) { return word == 0; }));
}

TEST_F(RenderGraphApi, FarPyramidIsNotInTheContract) {
  add("mod.cull", Stage::OpaqueDone, PassKind::Compute, [](PassBuilder& b) {
    b.after(phase::Geometry);
    b.use("hzb.depth", Access::Sampled, Missing::BindDummyBlack);
    b.side_effect();
  });
  const auto graph = compile_externals();
  EXPECT_EQ(status(*graph, "mod.cull"), CullReason::InvalidDeclaration);
  EXPECT_EQ(diagnostic(*graph, "mod.cull"), "render graph: 'mod.cull' cannot use 'hzb.depth'; disabled");
}
} // namespace
} // namespace aurora::render::detail

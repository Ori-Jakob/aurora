#include <gtest/gtest.h>

#include "logging.hpp"
#include "render/graph.hpp"
#include "render/graph_slots.hpp"
#include "render_graph_planned.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
namespace {
using wgpu::BufferUsage;
using wgpu::TextureFormat;
using wgpu::TextureUsage;

struct LogLine {
  AuroraLogLevel level;
  std::string message;
};

std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  g_logLines.push_back({level, std::string{message, len}});
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
  uint16_t order = 0;
  std::function<void(InternalPassBuilder&)> body;
  bool sink = false;
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

GraphInputs test_inputs(GraphExtent efb = {1280, 720}) {
  GraphInputs inputs;
  inputs.env.limits = planned::full_limits();
  inputs.env.efbSize = efb;
  inputs.env.outputSize = efb;
  inputs.env.gxFormats = planned::gx_formats_tier_p();
  inputs.env.gxUsage = planned::gx_usage_future();
  return inputs;
}

std::shared_ptr<CompiledGraph> compile(const std::vector<TestPass>& passes, GraphInputs inputs = test_inputs(),
                                       std::span<const GxReadDesc> gxReads = {}) {
  std::vector<PassDesc> descs;
  for (const TestPass& pass : passes) {
    descs.push_back({
        .name = pass.name,
        .stage = pass.stage,
        .kind = pass.kind,
        .order = pass.order,
        .declare = declare_test_pass,
        .state = &pass,
    });
  }
  inputs.passes = descs;
  inputs.gxReads = gxReads;
  return compile_graph(inputs);
}

TexDesc efb_tex(TextureFormat format, uint16_t divisor = 1) {
  return {.format = format, .sizeClass = SizeClass::Efb, .divisor = divisor};
}

TexDesc absolute_tex(TextureFormat format, uint32_t width, uint32_t height, uint32_t layers = 1) {
  return {
      .format = format, .sizeClass = SizeClass::Absolute, .width = width, .height = height, .depthOrLayers = layers};
}

void expect_all_active(const CompiledGraph& graph) {
  for (const CompiledNode& node : graph.nodes) {
    if (node.kind == NodeKind::Pass) {
      EXPECT_EQ(node.status, CullReason::None) << node.name << " " << cull_reason_name(node.status);
    }
  }
}

ResourceId resource_id(const CompiledGraph& graph, std::string_view name) {
  const ResourceId id = graph.find_resource(name);
  EXPECT_NE(id, InvalidResource) << name;
  return id;
}

SlotId slot_of(const CompiledGraph& graph, std::string_view name) {
  const ResourceId id = resource_id(graph, name);
  return id != InvalidResource ? graph.slots.resourceSlot[id] : InvalidSlot;
}

Lifetime lifetime_of(const CompiledGraph& graph, std::string_view name) {
  const ResourceId id = resource_id(graph, name);
  return id != InvalidResource ? graph.slots.lifetimes[id] : Lifetime{};
}

uint32_t position_of(const CompiledGraph& graph, std::string_view node) {
  const NodeId id = graph.find_node(node);
  EXPECT_NE(id, InvalidNode) << node;
  return id != InvalidNode ? graph.slots.positions[id] : UINT32_MAX;
}

size_t slot_occupants(const CompiledGraph& graph, SlotId slot) {
  return static_cast<size_t>(std::ranges::count(graph.slots.resourceSlot, slot));
}

std::map<std::string, SlotId> slots_by_name(const CompiledGraph& graph) {
  std::map<std::string, SlotId> out;
  for (size_t r = 0; r < graph.resources.size(); ++r) {
    out.emplace(graph.resources[r].name, graph.slots.resourceSlot[r]);
  }
  return out;
}

std::function<void(InternalPassBuilder&)> creates(std::string name, TexDesc desc) {
  return [name = std::move(name), desc](InternalPassBuilder& b) { b.create(name, desc, Access::StorageWrite); };
}

std::function<void(InternalPassBuilder&)> reads_then_creates(std::string read, std::string name, TexDesc desc) {
  return [read = std::move(read), name = std::move(name), desc](InternalPassBuilder& b) {
    b.use(read, Access::Sampled);
    b.create(name, desc, Access::StorageWrite);
  };
}

std::function<void(InternalPassBuilder&)> reads(std::vector<std::string> names) {
  return [names = std::move(names)](InternalPassBuilder& b) {
    for (const std::string& name : names) {
      b.use(name, Access::Sampled);
    }
  };
}

class GraphSlotsTest : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset_logged_diagnostics();
    g_logLines.clear();
    g_config.logCallback = capture_log;
  }

  void TearDown() override { g_config.logCallback = nullptr; }
};

TEST_F(GraphSlotsTest, KeyBytesCoversFormats) {
  const auto bytes = [](TextureFormat format, uint32_t width, uint32_t height, uint32_t depth = 1, uint32_t mips = 1,
                        wgpu::TextureDimension dimension = wgpu::TextureDimension::e2D) {
    return key_bytes(TexKey{
        .format = format,
        .dimension = dimension,
        .width = width,
        .height = height,
        .depthOrLayers = depth,
        .mips = mips,
    });
  };
  EXPECT_EQ(bytes(TextureFormat::RGBA16Float, 1280, 720), 7'372'800u);
  EXPECT_EQ(bytes(TextureFormat::R32Uint, 1280, 720), 3'686'400u);
  EXPECT_EQ(bytes(TextureFormat::Depth24PlusStencil8, 1280, 720), 3'686'400u);
  EXPECT_EQ(bytes(TextureFormat::RGBA16Float, 160, 90, 64, 2, wgpu::TextureDimension::e3D), 8'294'400u);
  EXPECT_EQ(bytes(TextureFormat::BC7RGBAUnorm, 256, 256, 1, 2), 81'920u);
  // Layers are not halved per mip, unlike 3D depth
  EXPECT_EQ(bytes(TextureFormat::RGBA8Unorm, 64, 64, 6, 2), (64 * 64 + 32 * 32) * 4 * 6u);
  EXPECT_EQ(bytes(TextureFormat::Stencil8, 64, 64), 4096u);
  EXPECT_EQ(bytes(TextureFormat::Depth32FloatStencil8, 64, 64), 32'768u);
  EXPECT_EQ(bytes(TextureFormat::ASTC12x12Unorm, 24, 25), 2 * 3 * 16u);
  EXPECT_EQ(bytes(TextureFormat::ETC2RGB8Unorm, 8, 8), 4 * 8u);
  EXPECT_EQ(bytes(TextureFormat::RG11B10Ufloat, 2, 2, 1, 2), 5 * 4u);
  EXPECT_EQ(count_logs(LOG_WARNING), 0u);
}

TEST_F(GraphSlotsTest, KeyBytesUnknownFormatIsZero) {
  constexpr auto Unknown = TextureFormat::R8BG8Biplanar420Unorm;
  EXPECT_FALSE(texture_format_info(Unknown).has_value());
  const TexKey key{.format = Unknown, .width = 64, .height = 64};
  EXPECT_EQ(key_bytes(key), 0u);
  EXPECT_EQ(key_bytes(key), 0u);
  EXPECT_EQ(count_logs(LOG_WARNING, "has no size"), 1u);
  EXPECT_EQ(poison_method(TexKey{.format = Unknown, .usage = TextureUsage::CopyDst}), PoisonMethod::None);
}

TEST_F(GraphSlotsTest, KeyUsesResolvedMips) {
  const auto graph = compile({{
      .name = "make",
      .body = creates("t", {.format = TextureFormat::RGBA16Float, .sizeClass = SizeClass::Efb, .mips = 0}),
      .sink = true,
  }});
  expect_all_active(*graph);
  const TexKey key = make_tex_key(*graph, resource_id(*graph, "t"));
  EXPECT_EQ(key.mips, 11u);
  EXPECT_EQ(key.width, 1280u);
  EXPECT_EQ(key.height, 720u);
  EXPECT_EQ(key.sizeClass, SizeClass::Efb);
  EXPECT_EQ(key.usage, TextureUsage::StorageBinding);
}

TEST_F(GraphSlotsTest, ChainSharesDisjointSlots) {
  const TexDesc hdr = efb_tex(TextureFormat::RGBA16Float);
  const auto graph = compile({
      {.name = "p0", .order = 0, .body = creates("A", hdr)},
      {.name = "p1", .order = 1, .body = reads_then_creates("A", "B", hdr)},
      {.name = "p2", .order = 2, .body = reads_then_creates("B", "C", hdr)},
      {.name = "p3", .order = 3, .body = reads_then_creates("C", "D", efb_tex(TextureFormat::R32Float))},
      {.name = "p4",
       .kind = PassKind::Render,
       .order = 4,
       .body =
           [](InternalPassBuilder& b) {
             b.use("D", Access::Sampled);
             b.use("gx.color", Access::ColorRMW);
           }},
  });
  expect_all_active(*graph);
  EXPECT_EQ(slot_of(*graph, "A"), slot_of(*graph, "C"));
  EXPECT_NE(slot_of(*graph, "A"), slot_of(*graph, "B"));
  EXPECT_NE(slot_of(*graph, "D"), slot_of(*graph, "A"));
  EXPECT_NE(slot_of(*graph, "D"), slot_of(*graph, "B"));
  EXPECT_EQ(graph->slots.slots.size(), 3u);
  EXPECT_EQ(std::ranges::count_if(graph->slots.resourceSlot, [](SlotId slot) { return slot != InvalidSlot; }), 4);
  EXPECT_EQ(lifetime_of(*graph, "A").first, position_of(*graph, "p0"));
  EXPECT_EQ(lifetime_of(*graph, "A").last, position_of(*graph, "p1"));
  EXPECT_EQ(graph->slots.slots[slot_of(*graph, "A")].anchor, "A");
  EXPECT_EQ(graph->slots.logicalBytes, 3 * 7'372'800u + 3'686'400u);
  EXPECT_EQ(graph->slots.slotBytes, 2 * 7'372'800u + 3'686'400u);
}

TEST_F(GraphSlotsTest, OverlapNeedsSeparateSlots) {
  const TexDesc hdr = efb_tex(TextureFormat::RGBA16Float);
  const auto graph = compile({
      {.name = "p0", .order = 0, .body = creates("X", hdr)},
      {.name = "p1", .order = 1, .body = creates("Y", hdr)},
      {.name = "p2", .order = 2, .body = reads({"X", "Y"}), .sink = true},
  });
  expect_all_active(*graph);
  EXPECT_NE(slot_of(*graph, "X"), slot_of(*graph, "Y"));
  EXPECT_EQ(graph->slots.slots.size(), 2u);
}

TEST_F(GraphSlotsTest, UsageUnionSplitsKeys) {
  const auto build = [](wgpu::TextureUsage extra) {
    TexDesc attachable = efb_tex(TextureFormat::RGBA16Float);
    attachable.extraUsage = extra;
    return compile({
        {.name = "p0", .order = 0, .body = creates("A", efb_tex(TextureFormat::RGBA16Float))},
        {.name = "p1", .order = 1, .body = reads({"A"}), .sink = true},
        {.name = "p2", .order = 2, .body = creates("C", attachable)},
        {.name = "p3", .order = 3, .body = reads({"C"}), .sink = true},
    });
  };
  const auto plain = build(TextureUsage::None);
  expect_all_active(*plain);
  EXPECT_EQ(slot_of(*plain, "A"), slot_of(*plain, "C"));

  const auto split = build(TextureUsage::RenderAttachment);
  expect_all_active(*split);
  EXPECT_NE(slot_of(*split, "A"), slot_of(*split, "C"));
  EXPECT_TRUE(split->slots.slots[slot_of(*split, "C")].tex.usage & TextureUsage::RenderAttachment);
}

TEST_F(GraphSlotsTest, InclusiveEndsConflict) {
  const TexDesc hdr = efb_tex(TextureFormat::RGBA16Float);
  const auto graph = compile({
      {.name = "p0", .order = 0, .body = creates("A", hdr)},
      {.name = "p1", .order = 1, .body = reads_then_creates("A", "C", hdr)},
      {.name = "p2", .order = 2, .body = reads({"C"}), .sink = true},
  });
  expect_all_active(*graph);
  EXPECT_EQ(lifetime_of(*graph, "A").last, position_of(*graph, "p1"));
  EXPECT_EQ(lifetime_of(*graph, "C").first, position_of(*graph, "p1"));
  EXPECT_NE(slot_of(*graph, "A"), slot_of(*graph, "C"));
}

TEST_F(GraphSlotsTest, CrossStageLifetime) {
  const TexDesc hdr = efb_tex(TextureFormat::RGBA16Float);
  const auto graph = compile({
      {.name = "e.make", .order = 0, .body = creates("E", hdr)},
      {.name = "f.make", .order = 1, .body = creates("F", hdr)},
      {.name = "f.read", .order = 2, .body = reads({"F"}), .sink = true},
      {.name = "e.read", .stage = Stage::PrePost, .order = 0, .body = reads({"E"}), .sink = true},
      {.name = "g.make", .stage = Stage::PrePost, .order = 1, .body = creates("G", hdr)},
      {.name = "g.read", .stage = Stage::PrePost, .order = 2, .body = reads({"G"}), .sink = true},
  });
  expect_all_active(*graph);
  EXPECT_EQ(lifetime_of(*graph, "E").first, position_of(*graph, "e.make"));
  EXPECT_EQ(lifetime_of(*graph, "E").last, position_of(*graph, "e.read"));
  EXPECT_NE(slot_of(*graph, "F"), slot_of(*graph, "E"));
  EXPECT_EQ(slot_of(*graph, "G"), slot_of(*graph, "E"));
}

void read_late_target(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use("A", Access::Sampled);
}

TEST_F(GraphSlotsTest, GxConsumerExtendsLifetime) {
  const TexDesc hdr = efb_tex(TextureFormat::RGBA16Float);
  const std::array reads_a{GxReadDesc{.node = "gx@PrePost", .declare = read_late_target}};
  const auto graph = compile(
      {
          {.name = "a.make", .order = 0, .body = creates("A", hdr)},
          {.name = "a.read", .order = 1, .body = reads({"A"}), .sink = true},
          {.name = "b.make", .order = 2, .body = creates("B", hdr)},
          {.name = "b.read", .order = 3, .body = reads({"B"}), .sink = true},
      },
      test_inputs(), reads_a);
  expect_all_active(*graph);
  EXPECT_EQ(lifetime_of(*graph, "A").last, position_of(*graph, "gx@PrePost"));
  EXPECT_NE(slot_of(*graph, "B"), slot_of(*graph, "A"));
}

void read_early_target(InternalPassBuilder& builder, const PassSetup&, const void*) {
  builder.use("X", Access::Sampled, Missing::BindDummyBlack);
}

// A creator active for the whole frame only makes a use static when the use runs in an earlier stage
TEST_F(GraphSlotsTest, StaticDummyDoesNotExtendLifetime) {
  const std::array reads_x{GxReadDesc{.node = "gx.overlay", .declare = read_early_target}};
  const auto graph = compile(
      {
          {.name = "x.make",
           .stage = Stage::PrePost,
           .order = 0,
           .body = creates("X", efb_tex(TextureFormat::R32Float))},
          {.name = "x.read", .stage = Stage::PrePost, .order = 1, .body = reads({"X"}), .sink = true},
      },
      test_inputs(), reads_x);
  expect_all_active(*graph);
  const NodeId overlay = graph->find_node("gx.overlay");
  const auto uses = graph->node_uses(overlay);
  const auto staticRead =
      std::ranges::find_if(uses, [&](const CompiledUse& use) { return graph->resources[use.resource].name == "X"; });
  ASSERT_NE(staticRead, uses.end());
  EXPECT_TRUE(staticRead->staticDummy);
  EXPECT_EQ(lifetime_of(*graph, "X").first, position_of(*graph, "x.make"));
  EXPECT_EQ(lifetime_of(*graph, "X").last, position_of(*graph, "x.read"));
}

TEST_F(GraphSlotsTest, SlotCountEqualsMaxOverlap) {
  constexpr std::array Formats{TextureFormat::RGBA16Float, TextureFormat::R32Float, TextureFormat::RGBA8Unorm};
  constexpr int KeyCount = 6;
  struct Made {
    std::string name;
    int creator = 0;
    int last = 0;
    int key = 0;
    std::vector<int> readers;
  };
  std::mt19937 rng{20261015};
  const auto pick = [&](int count) { return static_cast<int>(rng() % static_cast<uint32_t>(count)); };
  for (int round = 0; round < 200; ++round) {
    const int passCount = 4 + pick(12);
    std::vector<Made> made;
    for (int p = 0; p + 1 < passCount; ++p) {
      const int count = pick(3);
      for (int c = 0; c < count; ++c) {
        Made resource{.name = fmt::format("r{}.{}", p, c), .creator = p, .key = pick(KeyCount)};
        const int readers = 1 + pick(3);
        for (int k = 0; k < readers; ++k) {
          resource.readers.push_back(p + 1 + pick(passCount - p - 1));
        }
        std::ranges::sort(resource.readers);
        resource.readers.erase(std::ranges::unique(resource.readers).begin(), resource.readers.end());
        resource.last = resource.readers.back();
        made.push_back(std::move(resource));
      }
    }
    std::vector<TestPass> passes;
    for (int p = 0; p < passCount; ++p) {
      passes.push_back({
          .name = fmt::format("p{}", p),
          .order = static_cast<uint16_t>(p),
          .body =
              [&made, &Formats, p](InternalPassBuilder& b) {
                for (const Made& resource : made) {
                  if (std::ranges::find(resource.readers, p) != resource.readers.end()) {
                    b.use(resource.name, Access::Sampled);
                  }
                }
                for (const Made& resource : made) {
                  if (resource.creator == p) {
                    const auto divisor = static_cast<uint16_t>(1 + resource.key / Formats.size());
                    b.create(resource.name, efb_tex(Formats[resource.key % Formats.size()], divisor),
                             Access::StorageWrite);
                  }
                }
              },
          .sink = true,
      });
    }
    const auto graph = compile(passes);
    expect_all_active(*graph);
    for (int key = 0; key < KeyCount; ++key) {
      int peak = 0;
      for (int p = 0; p < passCount; ++p) {
        const auto live = std::ranges::count_if(made, [&](const Made& resource) {
          return resource.key == key && resource.creator <= p && p <= resource.last;
        });
        peak = std::max(peak, static_cast<int>(live));
      }
      std::set<SlotId> used;
      for (const Made& resource : made) {
        if (resource.key == key) {
          used.insert(slot_of(*graph, resource.name));
        }
      }
      EXPECT_EQ(static_cast<int>(used.size()), peak) << "round " << round << " key " << key;
    }
    for (const Made& a : made) {
      for (const Made& b : made) {
        if (&a != &b && slot_of(*graph, a.name) == slot_of(*graph, b.name)) {
          EXPECT_TRUE(a.last < b.creator || b.last < a.creator) << a.name << " " << b.name << " round " << round;
        }
      }
    }
  }
}

TEST_F(GraphSlotsTest, PlanIsDeterministic) {
  const auto forward = planned::planned_passes();
  const auto reversed = planned::planned_passes(true);
  const auto reads = planned::planned_gx_reads();
  const auto first = compile_graph(planned::planned_inputs(planned::full_limits(), forward, reads));
  const auto second = compile_graph(planned::planned_inputs(planned::full_limits(), forward, reads));
  const auto backwards = compile_graph(planned::planned_inputs(planned::full_limits(), reversed, reads));
  EXPECT_EQ(slots_by_name(*first), slots_by_name(*second));
  EXPECT_EQ(slots_by_name(*first), slots_by_name(*backwards));
  EXPECT_EQ(first->slots.slots.size(), backwards->slots.slots.size());
  for (size_t s = 0; s < first->slots.slots.size(); ++s) {
    EXPECT_EQ(first->slots.slots[s].anchor, backwards->slots.slots[s].anchor);
  }
}

TEST_F(GraphSlotsTest, PlannedSetSnapshot) {
  const auto passes = planned::planned_passes();
  const auto reads = planned::planned_gx_reads();
  const auto graph = compile_graph(planned::planned_inputs(planned::full_limits(), passes, reads));
  const SlotPlan& plan = graph->slots;
  ASSERT_EQ(plan.slots.size(), 35u);
  EXPECT_EQ(std::ranges::count(plan.slots, false, &SlotPlan::Slot::isBuffer), 21);
  EXPECT_EQ(std::ranges::count(plan.slots, true, &SlotPlan::Slot::isBuffer), 14);
  EXPECT_EQ(plan.logicalBytes, 455'984'600u);
  EXPECT_EQ(plan.slotBytes, 366'396'888u);
  EXPECT_EQ(plan.efbSize, (GraphExtent{2560, 1440}));

  const auto shared = [&](std::initializer_list<std::string_view> names) {
    const SlotId slot = slot_of(*graph, *names.begin());
    for (const std::string_view name : names) {
      EXPECT_EQ(slot_of(*graph, name), slot) << name;
    }
    EXPECT_EQ(slot_occupants(*graph, slot), names.size()) << *names.begin();
  };
  const auto apart = [&](std::string_view a, std::string_view b) {
    EXPECT_NE(slot_of(*graph, a), slot_of(*graph, b)) << a << " " << b;
  };
  shared({"ao.raw", "vsm.sun"});
  shared({"rt.ao"});
  shared({"rt.shadow"});
  shared({"vis.sun"});
  shared({"scene.hdr"});
  shared({"ssil.raw", "gi.irr"});
  shared({"refl.spec", "clouds.full"});
  shared({"gi.raw", "refl.raw", "clouds.trace"});
  shared({"terrain.patches", "fog.tiles"});
  shared({"vg.visible", "vsm.draws"});
  apart("godrays.mask", "godrays.blur");
  apart("fog.integrated", "fog.relight");
  apart("rt.rays", "rt.hits");

  ASSERT_EQ(plan.histories.size(), graph->histories.size());
  for (size_t h = 0; h < plan.histories.size(); ++h) {
    EXPECT_EQ(plan.histories[h].name, graph->resources[graph->histories[h]].name);
    EXPECT_EQ(graph->slots.resourceSlot[graph->histories[h]], InvalidSlot);
  }
  EXPECT_TRUE(std::ranges::is_sorted(plan.poisonPoints));
  EXPECT_EQ(plan.poisonPoints.size(), 42u);
}

TEST_F(GraphSlotsTest, BuffersBucketByPow2AndUsage) {
  const auto graph = compile({
      {.name = "x.make",
       .order = 0,
       .body = [](InternalPassBuilder& b) { b.create_buffer("X", {.size = 3 * 1024}, Access::StorageWrite); }},
      {.name = "x.read",
       .order = 1,
       .body = [](InternalPassBuilder& b) { b.use_buffer("X", Access::StorageRead); },
       .sink = true},
      {.name = "y.make",
       .order = 2,
       .body = [](InternalPassBuilder& b) { b.create_buffer("Y", {.size = 4 * 1024}, Access::StorageWrite); }},
      {.name = "y.read",
       .order = 3,
       .body = [](InternalPassBuilder& b) { b.use_buffer("Y", Access::StorageRead); },
       .sink = true},
      {.name = "z.make",
       .kind = PassKind::Copy,
       .order = 4,
       .body = [](InternalPassBuilder& b) { b.create_buffer("Z", {.size = 4 * 1024}, Access::CopyDst); }},
      {.name = "z.read",
       .order = 5,
       .body = [](InternalPassBuilder& b) { b.use_buffer("Z", Access::Uniform); },
       .sink = true},
  });
  expect_all_active(*graph);
  EXPECT_EQ(slot_of(*graph, "X"), slot_of(*graph, "Y"));
  EXPECT_NE(slot_of(*graph, "Z"), slot_of(*graph, "X"));
  const SlotPlan::Slot& shared = graph->slots.slots[slot_of(*graph, "X")];
  EXPECT_TRUE(shared.isBuffer);
  EXPECT_EQ(shared.buf.bucketSize, 4096u);
  EXPECT_EQ(shared.buf.usage, BufferUsage::Storage);
  EXPECT_EQ(graph->slots.slots[slot_of(*graph, "Z")].buf.usage, BufferUsage::CopyDst | BufferUsage::Uniform);
  EXPECT_EQ(buffer_bucket(1), 256u);
  EXPECT_EQ(buffer_bucket(256), 256u);
  EXPECT_EQ(buffer_bucket(257), 512u);
}

TEST_F(GraphSlotsTest, EfbClassBuffersKeepSizeClass) {
  const auto graph = compile({{
      .name = "make",
      .body =
          [](InternalPassBuilder& b) {
            b.create_buffer("e", {.sizeClass = SizeClass::Efb, .bytesPerTexel = 4, .divisor = 4}, Access::StorageWrite);
          },
      .sink = true,
  }});
  expect_all_active(*graph);
  const CompiledResource& resource = graph->resources[resource_id(*graph, "e")];
  EXPECT_EQ(resource.bufferSize, 230'400u);
  const BufKey key = make_buf_key(resource, planned::full_limits());
  EXPECT_EQ(key.bucketSize, 262'144u);
  EXPECT_EQ(key.sizeClass, SizeClass::Efb);
  EXPECT_EQ(graph->slots.slots[slot_of(*graph, "e")].buf, key);
}

TEST_F(GraphSlotsTest, BucketsStayWithinOddLimits) {
  constexpr uint64_t KiB = 1024;
  GraphInputs inputs = test_inputs();
  inputs.env.limits.bufferSize = 3 * 1024 * KiB;
  inputs.env.limits.storageBufferBindingSize = 1536 * KiB;
  inputs.env.limits.uniformBufferBindingSize = 48 * KiB;
  const auto copied = [](std::string name, uint64_t size, Access access) {
    return std::vector<TestPass>{
        {.name = name + ".make",
         .kind = PassKind::Copy,
         .body = [name, size](InternalPassBuilder& b) { b.create_buffer(name, {.size = size}, Access::CopyDst); }},
        {.name = name + ".read",
         .order = 1,
         .body = [name, access](InternalPassBuilder& b) { b.use_buffer(name, access); },
         .sink = true},
    };
  };
  std::vector<TestPass> passes;
  for (auto&& chain : {copied("args", 2560 * KiB, Access::Indirect), copied("ub", 40 * KiB, Access::Uniform)}) {
    passes.insert(passes.end(), chain.begin(), chain.end());
  }
  passes.push_back({.name = "big.make", .order = 2, .body = [](InternalPassBuilder& b) {
                      b.create_buffer("big", {.size = 1200 * KiB}, Access::StorageWrite);
                    }});
  passes.push_back({.name = "small.make", .order = 2, .body = [](InternalPassBuilder& b) {
                      b.create_buffer("small", {.size = 100 * KiB}, Access::StorageWrite);
                    }});
  passes.push_back({.name = "storage.read",
                    .order = 3,
                    .body =
                        [](InternalPassBuilder& b) {
                          b.use_buffer("big", Access::StorageRead);
                          b.use_buffer("small", Access::StorageRead);
                        },
                    .sink = true});
  const auto graph = compile(passes, inputs);
  expect_all_active(*graph);
  const auto bucket = [&](std::string_view name) { return graph->slots.slots[slot_of(*graph, name)].buf.bucketSize; };
  // Rounding up would pass the limit, so these keep their exact sizes
  EXPECT_EQ(bucket("args"), 2560 * KiB);
  EXPECT_EQ(bucket("big"), 1200 * KiB);
  EXPECT_EQ(bucket("ub"), 40 * KiB);
  EXPECT_EQ(bucket("small"), 128 * KiB);
  EXPECT_EQ(buffer_bucket(2560 * KiB, 3 * 1024 * KiB), 2560 * KiB);
  EXPECT_EQ(buffer_bucket(2048 * KiB, 3 * 1024 * KiB), 2048 * KiB);
  EXPECT_EQ(buffer_bucket(2049 * KiB, 3 * 1024 * KiB), 2049 * KiB);
}

TEST_F(GraphSlotsTest, BindingViewFromUses) {
  const auto build = [](wgpu::TextureViewDimension dimension) {
    return compile({
        {.name = "make", .order = 0, .body = creates("cube", absolute_tex(TextureFormat::RGBA8Unorm, 64, 64, 6))},
        {.name = "read",
         .order = 1,
         .body = [dimension](InternalPassBuilder& b) { b.use("cube", Access::Sampled, {.dimension = dimension}); },
         .sink = true},
    });
  };
  const auto cube = build(wgpu::TextureViewDimension::Cube);
  expect_all_active(*cube);
  EXPECT_EQ(binding_view_dimension(*cube, resource_id(*cube, "cube")), wgpu::TextureViewDimension::Cube);
  EXPECT_EQ(make_tex_key(*cube, resource_id(*cube, "cube")).bindingView, wgpu::TextureViewDimension::Cube);

  const auto array = build(wgpu::TextureViewDimension::Undefined);
  expect_all_active(*array);
  EXPECT_EQ(make_tex_key(*array, resource_id(*array, "cube")).bindingView, wgpu::TextureViewDimension::e2DArray);
}

TEST_F(GraphSlotsTest, PoisonMethodByUsage) {
  const auto texture = [](TextureFormat format, wgpu::TextureUsage usage) {
    return poison_method(TexKey{.format = format, .width = 16, .height = 16, .usage = usage});
  };
  EXPECT_EQ(texture(TextureFormat::RGBA8Unorm, TextureUsage::RenderAttachment | TextureUsage::TextureBinding),
            PoisonMethod::RenderClear);
  EXPECT_EQ(texture(TextureFormat::RGBA16Float, TextureUsage::StorageBinding | TextureUsage::TextureBinding),
            PoisonMethod::ComputeClear);
  EXPECT_EQ(texture(TextureFormat::RGBA8Unorm, TextureUsage::CopyDst | TextureUsage::TextureBinding),
            PoisonMethod::BufferCopy);
  EXPECT_EQ(texture(TextureFormat::Depth32Float, TextureUsage::CopyDst), PoisonMethod::None);
  EXPECT_EQ(texture(TextureFormat::RGBA8Unorm, TextureUsage::TextureBinding), PoisonMethod::None);
  EXPECT_EQ(poison_method(BufKey{.bucketSize = 256, .usage = BufferUsage::CopyDst | BufferUsage::Storage}),
            PoisonMethod::BufferCopy);
  EXPECT_EQ(poison_method(BufKey{.bucketSize = 256, .usage = BufferUsage::Storage}), PoisonMethod::BufferFill);
  EXPECT_EQ(poison_method(BufKey{.bucketSize = 256, .usage = BufferUsage::Uniform}), PoisonMethod::None);
}
} // namespace
} // namespace aurora::render::detail

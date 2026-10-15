#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "render/debug/debug.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/json.hpp"
#include "render/settings.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu_prof.hpp"

#include <aurora/render.hpp>
#include <imgui.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render {
namespace {
namespace gpu_prof = webgpu::gpu_prof;
using gpu_prof::detail::Event;
using gpu_prof::detail::EventKind;

#ifdef TRACY_ENABLE
constexpr bool TracyBuild = true;
#else
constexpr bool TracyBuild = false;
#endif
constexpr uint64_t WaitNs = 5'000'000'000;

const char* const NameA = "A";
const char* const NameB = "B";

TEST(GpuProfAggregate, SumsByNameInFirstSeenOrder) {
  const std::array events{
      Event{NameA, 0, EventKind::PassBegin}, Event{nullptr, 1, EventKind::End},
      Event{NameB, 2, EventKind::ZoneBegin}, Event{nullptr, 3, EventKind::End},
      Event{NameA, 4, EventKind::PassBegin}, Event{nullptr, 5, EventKind::End},
  };
  const std::array<uint64_t, 6> timestamps{1'000'000, 2'000'000, 2'000'000, 5'000'000, 6'000'000, 6'500'000};
  gpu_prof::FrameTimes out;
  gpu_prof::detail::aggregate(events, timestamps, 1'000'000, 6'500'000, out);
  ASSERT_EQ(out.passes.size(), 2u);
  EXPECT_EQ(out.passes[0].name, NameA);
  EXPECT_FLOAT_EQ(out.passes[0].ms, 1.5f);
  EXPECT_EQ(out.passes[1].name, NameB);
  EXPECT_FLOAT_EQ(out.passes[1].ms, 3.0f);
}

TEST(GpuProfAggregate, DropsUnwrittenAndReversedPairs) {
  const std::array events{
      Event{NameA, 0, EventKind::PassBegin}, Event{nullptr, 1, EventKind::End},
      Event{NameA, 2, EventKind::PassBegin}, Event{nullptr, 3, EventKind::End},
      Event{NameA, 4, EventKind::ZoneBegin}, Event{nullptr, 5, EventKind::End},
      Event{NameB, 6, EventKind::PassBegin}, Event{nullptr, 7, EventKind::End},
  };
  const std::array<uint64_t, 8> timestamps{0, 0, 5'000'000, 0, 5'000'000, 4'000'000, 1'000'000, 3'000'000};
  gpu_prof::FrameTimes out;
  gpu_prof::detail::aggregate(events, timestamps, 1'000'000, 3'000'000, out);
  ASSERT_EQ(out.passes.size(), 1u);
  EXPECT_EQ(out.passes[0].name, NameB);
  EXPECT_FLOAT_EQ(out.passes[0].ms, 2.0f);
}

TEST(GpuProfAggregate, NestedSameNameCountsOnce) {
  const std::array events{
      Event{NameA, 0, EventKind::ZoneBegin}, Event{NameA, 2, EventKind::PassBegin}, Event{nullptr, 3, EventKind::End},
      Event{NameB, 4, EventKind::PassBegin}, Event{nullptr, 5, EventKind::End},     Event{nullptr, 1, EventKind::End},
      Event{NameA, 6, EventKind::PassBegin}, Event{nullptr, 7, EventKind::End},
  };
  const std::array<uint64_t, 8> timestamps{1'000'000, 5'000'000, 2'000'000, 3'000'000,
                                           3'000'000, 4'000'000, 6'000'000, 6'500'000};
  gpu_prof::FrameTimes out;
  gpu_prof::detail::aggregate(events, timestamps, 1'000'000, 6'500'000, out);
  ASSERT_EQ(out.passes.size(), 2u);
  EXPECT_EQ(out.passes[0].name, NameA);
  EXPECT_FLOAT_EQ(out.passes[0].ms, 4.5f);
  EXPECT_EQ(out.passes[1].name, NameB);
  EXPECT_FLOAT_EQ(out.passes[1].ms, 1.0f);

  // An encoder zone that read back as zero leaves the passes inside it to carry the time
  const std::array<uint64_t, 8> unwritten{0, 0, 2'000'000, 3'000'000, 3'000'000, 4'000'000, 6'000'000, 6'500'000};
  gpu_prof::detail::aggregate(events, unwritten, 2'000'000, 6'500'000, out);
  ASSERT_EQ(out.passes.size(), 2u);
  EXPECT_EQ(out.passes[0].name, NameA);
  EXPECT_FLOAT_EQ(out.passes[0].ms, 1.5f);
  EXPECT_FLOAT_EQ(out.passes[1].ms, 1.0f);
}

TEST(GpuProfAggregate, FrameMsFromBounds) {
  gpu_prof::FrameTimes out;
  out.passes.push_back({NameA, 9.f});
  gpu_prof::detail::aggregate({}, {}, 1'000'000, 7'000'000, out);
  EXPECT_FLOAT_EQ(out.frameMs, 6.0f);
  EXPECT_TRUE(out.passes.empty());
  gpu_prof::detail::aggregate({}, {}, 7'000'000, 1'000'000, out);
  EXPECT_EQ(out.frameMs, 0.f);
}

class GpuProfCollection : public ::testing::Test {
protected:
  void SetUp() override {
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no test device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
    gpu_prof::shutdown();
    // Tracy builds collect every frame, which is what the request tests must not see
    gpu_prof::set_always(false);
    gpu_prof::initialize();
    if (!gpu_prof::supported()) {
      GTEST_SKIP() << "the device has no timestamp queries";
    }
  }

  void TearDown() override {
    if (!m_device) {
      return;
    }
    EXPECT_TRUE(gpu_prof::wait_idle(WaitNs));
    gpu_prof::shutdown();
    gpu_prof::set_always(TracyBuild);
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  void run_frame(uint32_t zones = 1) const {
    submit_frame(zones);
    // The render worker pumps events once per frame
    m_device->instance.ProcessEvents();
  }

  void submit_frame(uint32_t zones = 1) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    gpu_prof::frame_begin(encoder);
    for (uint32_t i = 0; i < zones; ++i) {
      const gpu_prof::Zone zone{encoder, "test.zone"};
    }
    gpu_prof::frame_end(encoder);
    const wgpu::CommandBuffer commands = encoder.Finish();
    m_device->queue.Submit(1, &commands);
    gpu_prof::after_submit();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(GpuProfCollection, NoResolveUntilRequested) {
  const uint64_t resolves = gpu_prof::testing::resolve_count();
  for (int i = 0; i < 3; ++i) {
    run_frame();
  }
  EXPECT_EQ(gpu_prof::testing::resolve_count(), resolves);
  EXPECT_FALSE(gpu_prof::testing::resources_created());
  EXPECT_EQ(gpu_prof::last_frame_sequence(), 3u);

  gpu_prof::request(2);
  for (int i = 0; i < 3; ++i) {
    run_frame();
  }
  EXPECT_EQ(gpu_prof::testing::resolve_count(), resolves + 2);
  ASSERT_TRUE(gpu_prof::wait_idle(WaitNs));
  gpu_prof::FrameTimes times;
  ASSERT_TRUE(gpu_prof::latest(times));
  EXPECT_EQ(times.sequence, 5u);
  EXPECT_EQ(times.dropped, 0u);
  EXPECT_FALSE(gpu_prof::latest(times));

  run_frame();
  EXPECT_FALSE(gpu_prof::testing::resources_created());
  EXPECT_EQ(gpu_prof::testing::resolve_count(), resolves + 2);
}

// The null queue finishes work as it is submitted, so only a real device can leave a readback in flight
class GpuProfDeviceCollection : public GpuProfCollection {
protected:
  GpuProfDeviceCollection() {
    m_options.backend =
        test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
  }
};

TEST_F(GpuProfDeviceCollection, ReadbackKeepsResourcesAfterExpiry) {
  if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
    GTEST_SKIP() << "the null backend completes readbacks immediately";
  }
  constexpr uint64_t CopyBytes = 64ull << 20;
  const wgpu::BufferDescriptor sourceDescriptor{.usage = wgpu::BufferUsage::CopySrc, .size = CopyBytes};
  const wgpu::BufferDescriptor targetDescriptor{.usage = wgpu::BufferUsage::CopyDst, .size = CopyBytes};
  const wgpu::Buffer source = m_device->device.CreateBuffer(&sourceDescriptor);
  const wgpu::Buffer target = m_device->device.CreateBuffer(&targetDescriptor);
  gpu_prof::request(1);
  const wgpu::CommandEncoder first = m_device->device.CreateCommandEncoder();
  gpu_prof::frame_begin(first);
  {
    const gpu_prof::Zone zone{first, "test.copy"};
    first.CopyBufferToBuffer(source, 0, target, 0, CopyBytes);
  }
  gpu_prof::frame_end(first);
  const wgpu::CommandBuffer firstCommands = first.Finish();
  m_device->queue.Submit(1, &firstCommands);
  gpu_prof::after_submit();
  ASSERT_TRUE(gpu_prof::testing::resources_created());

  // The keep-alive has run out while frame 1's copy and readback are still on the GPU
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  gpu_prof::frame_begin(encoder);
  EXPECT_TRUE(gpu_prof::testing::resources_created());
  gpu_prof::frame_end(encoder);
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);
  gpu_prof::after_submit();
  ASSERT_TRUE(gpu_prof::wait_idle(WaitNs));
  gpu_prof::FrameTimes times;
  ASSERT_TRUE(gpu_prof::latest(times));
  EXPECT_EQ(times.sequence, 1u);
  EXPECT_TRUE(gpu_prof::testing::resources_created());

  run_frame();
  EXPECT_FALSE(gpu_prof::testing::resources_created());
}

TEST_F(GpuProfCollection, AlwaysCollectsEveryFrame) {
  const uint64_t resolves = gpu_prof::testing::resolve_count();
  gpu_prof::set_always(true);
  for (int i = 0; i < 3; ++i) {
    run_frame();
  }
  gpu_prof::set_always(false);
  EXPECT_EQ(gpu_prof::testing::resolve_count(), resolves + 3);
}

TEST_F(GpuProfCollection, HoldsFiveHundredElevenZones) {
  gpu_prof::request(2);
  run_frame(511);
  ASSERT_TRUE(gpu_prof::wait_idle(WaitNs));
  gpu_prof::FrameTimes times;
  ASSERT_TRUE(gpu_prof::latest(times));
  EXPECT_EQ(times.dropped, 0u);

  run_frame(600);
  ASSERT_TRUE(gpu_prof::wait_idle(WaitNs));
  ASSERT_TRUE(gpu_prof::latest(times));
  EXPECT_EQ(times.sequence, 2u);
  EXPECT_EQ(times.dropped, 89u);
}

class ImGuiFrame {
public:
  ImGuiFrame() {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.DisplaySize = {1280.f, 720.f};
    io.Fonts->Build();
  }
  ~ImGuiFrame() { ImGui::DestroyContext(); }
  ImGuiFrame(const ImGuiFrame&) = delete;
  ImGuiFrame& operator=(const ImGuiFrame&) = delete;

  template <typename F>
  void frame(F&& body) const {
    ImGui::NewFrame();
    body();
    ImGui::Render();
  }
};

void declare_side_effect(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.side_effect();
}

void declare_orphan(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.create(
      "test.orphan_out",
      {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = detail::SizeClass::Absolute, .width = 4, .height = 4},
      detail::Access::StorageWrite);
}

bool never_enabled(const detail::StageData&, const void*) { return false; }

void encode_nothing(detail::PassContext&, const void*) {}

void draw_nothing() {}

detail::PassId register_test_pass(std::string name, Stage stage, uint16_t order,
                                  detail::PassDeclareFn declare = declare_side_effect,
                                  detail::PassClass cls = detail::PassClass::Optional,
                                  detail::PassEnabledFn enabled = nullptr) {
  return detail::register_pass({
      .name = std::move(name),
      .stage = stage,
      .kind = detail::PassKind::Compute,
      .cls = cls,
      .order = order,
      .declare = declare,
      .enabled = enabled,
      .encode = encode_nothing,
  });
}

class RenderDebugWindow : public ::testing::Test {
protected:
  void SetUp() override {
    detail::testing::reset();
    debug::testing::reset();
  }

  void TearDown() override {
    debug::shutdown();
    detail::testing::reset();
    debug::testing::reset();
  }
};

TEST_F(RenderDebugWindow, DrawsEveryTabHeadless) {
  register_test_pass("test.kept", Stage::OpaqueDone, 0);
  register_test_pass("test.culled", Stage::OpaqueDone, 1, declare_orphan);
  const auto frame = detail::latch_frame();
  ASSERT_EQ(frame->graph->framePasses.size(), 1u);
  const ImGuiFrame imgui;
  debug::initialize();
  ASSERT_GE(debug::tabs().size(), 2u);
  for (const debug::Tab& tab : debug::tabs()) {
    imgui.frame([&] {
      ImGui::Begin("t");
      tab.draw();
      ImGui::End();
    });
  }
  bool open = true;
  imgui.frame([&] { draw_debug_window(&open); });
  EXPECT_TRUE(open);
}

TEST_F(RenderDebugWindow, DrawsUninitializedWindow) {
  const ImGuiFrame imgui;
  bool open = true;
  imgui.frame([&] { draw_debug_window(&open); });
  EXPECT_TRUE(debug::tabs().empty());
}

TEST_F(RenderDebugWindow, InspectorOpenExpires) {
  const ImGuiFrame imgui;
  EXPECT_FALSE(debug::inspector_open());
  for (int i = 0; i < 5; ++i) {
    debug::on_latch();
  }
  EXPECT_FALSE(debug::inspector_open());

  bool closed = false;
  imgui.frame([&] { draw_debug_window(&closed); });
  EXPECT_FALSE(debug::inspector_open());
  imgui.frame([&] { draw_debug_window(nullptr); });
  EXPECT_FALSE(debug::inspector_open());

  bool open = true;
  imgui.frame([&] { draw_debug_window(&open); });
  EXPECT_TRUE(debug::inspector_open());
  debug::on_latch();
  debug::on_latch();
  EXPECT_TRUE(debug::inspector_open());
  debug::on_latch();
  EXPECT_FALSE(debug::inspector_open());
}

TEST_F(RenderDebugWindow, FrameCullSnapshotPairing) {
  register_test_pass("test.a", Stage::OpaqueDone, 0);
  register_test_pass("test.b", Stage::OpaqueDone, 1);
  register_test_pass("test.c", Stage::PrePost, 0);
  register_test_pass("test.d", Stage::BeforeHud, 0);
  const auto frame = detail::latch_frame();
  ASSERT_NE(frame, nullptr);
  ASSERT_NE(frame->graph, nullptr);
  const detail::CompiledGraph& graph = *frame->graph;
  ASSERT_EQ(graph.framePasses.size(), 4u);
  for (const std::string_view name : {"test.a", "test.b", "test.c", "test.d"}) {
    ASSERT_NE(graph.find_node(name), detail::InvalidNode) << name;
  }
  const auto node = [&](std::string_view name) -> const detail::CompiledNode& {
    return graph.nodes[graph.find_node(name)];
  };
  const auto slot = [&](std::string_view name) { return node(name).frameSlot; };

  debug::FrameCullSnapshot snapshot;
  EXPECT_FALSE(debug::latest_frame_cull(snapshot));
  std::array<detail::FrameCullReason, 2> opaque{};
  opaque[node("test.a").programIndex] = detail::FrameCullReason::Ran;
  opaque[node("test.b").programIndex] = detail::FrameCullReason::NotEnabled;
  const std::array prePost{detail::FrameCullReason::InputDisabled};
  debug::record_stage_cull(graph, Stage::OpaqueDone, 7, opaque);
  debug::record_stage_cull(graph, Stage::PrePost, 7, prePost);
  debug::publish_frame_cull(7, graph.graphHash);
  ASSERT_TRUE(debug::latest_frame_cull(snapshot));
  EXPECT_EQ(snapshot.frameId, 7u);
  EXPECT_EQ(snapshot.graphHash, graph.graphHash);
  ASSERT_EQ(snapshot.reasons.size(), 4u);
  EXPECT_EQ(snapshot.reasons[slot("test.a")], std::nullopt);
  EXPECT_EQ(snapshot.reasons[slot("test.b")], "NotEnabled");
  EXPECT_EQ(snapshot.reasons[slot("test.c")], "InputDisabled");
  EXPECT_EQ(snapshot.reasons[slot("test.d")], "StageNotRun");

  const std::string paired = graph_dump_json(true);
  const json::Doc pairedDoc = json::read(paired);
  ASSERT_NE(pairedDoc, nullptr);
  yyjson_val* frameKey = yyjson_obj_get(yyjson_doc_get_root(pairedDoc.get()), "frame");
  ASSERT_NE(frameKey, nullptr);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(frameKey, "frameId")), 7u);
  EXPECT_EQ(yyjson_arr_size(yyjson_obj_get(frameKey, "passes")), 4u);
  // Nothing was ever timed without a device, so there is no timings section to pair
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(pairedDoc.get()), "timings"), nullptr);

  debug::record_stage_cull(graph, Stage::OpaqueDone, 8, opaque);
  debug::publish_frame_cull(8, graph.graphHash ^ 1);
  const std::string stale = graph_dump_json(true);
  const json::Doc staleDoc = json::read(stale);
  ASSERT_NE(staleDoc, nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(staleDoc.get()), "frame"), nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(staleDoc.get()), "timings"), nullptr);

  const json::Doc plain = json::read(graph_dump_json());
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(plain.get()), "frame"), nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(plain.get()), "timings"), nullptr);
  EXPECT_NE(yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(plain.get()), "pool"), "claimedBytes"), nullptr);
}

TEST_F(RenderDebugWindow, StageCullRecordsOnlyWhileInspecting) {
  register_test_pass("test.off", Stage::OpaqueDone, 0, declare_side_effect, detail::PassClass::Optional, never_enabled);
  register_test_pass("test.late", Stage::PrePost, 0, declare_side_effect, detail::PassClass::Optional, never_enabled);
  const auto latched = detail::latch_frame();
  ASSERT_NE(latched, nullptr);
  const auto run_frame = [&](bool inspecting, uint64_t frameId) {
    const detail::FrameState state{.graph = latched->graph, .inspectorOpen = inspecting};
    for (size_t s = 0; s < StageCount; ++s) {
      detail::StageData stage{
          .stage = static_cast<Stage>(s),
          .frameId = frameId,
          .markerSeen = true,
          .frame = &state,
      };
      EXPECT_FALSE(detail::build_stage_data(stage));
    }
  };

  debug::FrameCullSnapshot snapshot;
  run_frame(false, 3);
  EXPECT_FALSE(debug::latest_frame_cull(snapshot));
  run_frame(true, 4);
  ASSERT_TRUE(debug::latest_frame_cull(snapshot));
  EXPECT_EQ(snapshot.frameId, 4u);
  EXPECT_EQ(snapshot.graphHash, latched->graph->graphHash);
  ASSERT_EQ(snapshot.reasons.size(), 2u);
  for (const auto& reason : snapshot.reasons) {
    EXPECT_EQ(reason, "NotEnabled");
  }
  run_frame(false, 5);
  ASSERT_TRUE(debug::latest_frame_cull(snapshot));
  EXPECT_EQ(snapshot.frameId, 4u);
  detail::reset_frame_cull();
}

TEST_F(RenderDebugWindow, TabRegistryOrdersAndReplaces) {
  debug::initialize();
  debug::register_tab("Zeta", 5, draw_nothing);
  debug::register_tab("Alpha", 5, draw_nothing);
  debug::register_tab("Zeta", 50, draw_nothing);
  std::vector<std::string> names;
  for (const debug::Tab& tab : debug::tabs()) {
    names.push_back(tab.name);
  }
  EXPECT_EQ(names, (std::vector<std::string>{"Overview", "Alpha", "Graph", "Zeta"}));
  EXPECT_EQ(debug::tabs().back().order, 50);
}

TEST_F(RenderDebugWindow, PassTimingsEmptyWithoutTimestamps) {
  gpu_prof::shutdown();
  const uint64_t sequence = gpu_prof::last_frame_sequence();
  EXPECT_TRUE(pass_timings().empty());
  EXPECT_TRUE(pass_timings().empty());
  EXPECT_EQ(gpu_prof::last_frame_sequence(), sequence);
  EXPECT_FALSE(gpu_prof::supported());
}

TEST(RenderGraphDump, NotCompiledBeforeFirstLatch) {
  detail::testing::reset();
  EXPECT_EQ(graph_dump_json(), debug::NotCompiledGraphJson);
  EXPECT_EQ(graph_dump_json(true), debug::NotCompiledGraphJson);
  const json::Doc doc = json::read(debug::NotCompiledGraphJson);
  ASSERT_NE(doc, nullptr);
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_EQ(yyjson_obj_size(root), 3u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "schema")), "aurora.render.graph");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "version")), 1u);
  EXPECT_TRUE(yyjson_is_false(yyjson_obj_get(root, "compiled")));
}

class RenderDebugWindowLatch : public RenderDebugWindow {
protected:
  void SetUp() override {
    RenderDebugWindow::SetUp();
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
  }

  void TearDown() override {
    if (m_device) {
      detail::testing::reset();
      detail::shutdown_bindings();
      detail::testing::reset_transient_pool(nullptr);
      gpu_heap::shutdown();
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
      m_globals.reset();
      m_device.reset();
    }
    RenderDebugWindow::TearDown();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(RenderDebugWindowLatch, LatchCarriesInspectorFlag) {
  register_test_pass("test.diagnostic", Stage::OpaqueDone, 0, declare_orphan, detail::PassClass::Diagnostic);
  const ImGuiFrame imgui;
  const auto diagnostic_status = [](const detail::FrameState& frame) {
    const detail::CompiledGraph& graph = *frame.graph;
    const detail::NodeId node = graph.find_node("test.diagnostic");
    return node != detail::InvalidNode ? graph.nodes[node].status : detail::CullReason::Skipped;
  };

  const auto closed = detail::latch_frame();
  EXPECT_FALSE(closed->inspectorOpen);
  EXPECT_EQ(diagnostic_status(*closed), detail::CullReason::InspectorClosed);

  bool open = true;
  imgui.frame([&] { draw_debug_window(&open); });
  const auto opened = detail::latch_frame();
  EXPECT_TRUE(opened->inspectorOpen);
  EXPECT_NE(opened->graph, closed->graph);
  EXPECT_EQ(diagnostic_status(*opened), detail::CullReason::None);
  EXPECT_EQ(detail::latch_frame()->graph, opened->graph);

  std::shared_ptr<const detail::FrameState> expired;
  for (int i = 0; i < 2; ++i) {
    expired = detail::latch_frame();
  }
  EXPECT_FALSE(expired->inspectorOpen);
  EXPECT_EQ(diagnostic_status(*expired), detail::CullReason::InspectorClosed);
}
} // namespace
} // namespace aurora::render

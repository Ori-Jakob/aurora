#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "render/ao/ao.hpp"
#include "render/ao/gtao.hpp"
#include "render/debug/capture.hpp"
#include "render/debug/debug.hpp"
#include "render/debug/decomp_view.hpp"
#include "render/debug/dual_source.hpp"
#include "render/debug/gbuf_view.hpp"
#include "render/debug/views.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/jobs.hpp"
#include "render/json.hpp"
#include "render/post/composite.hpp"
#include "render/scene/classify.hpp"
#include "render/settings.hpp"
#include "render/shaders.hpp"
#include "render/stage.hpp"
#include "webgpu/gpu_prof.hpp"

#include <aurora/render.hpp>
#include <aurora/render_caps.hpp>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
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
    gpu_prof::testing::hold_readbacks(false);
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
  // A fast GPU can finish and drain frame 1 before frame 2 begins, which would leave nothing in flight to keep
  gpu_prof::testing::hold_readbacks(true);
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

  // The keep-alive has run out while frame 1's readback is still in the ring
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  gpu_prof::frame_begin(encoder);
  EXPECT_TRUE(gpu_prof::testing::resources_created());
  gpu_prof::frame_end(encoder);
  const wgpu::CommandBuffer commands = encoder.Finish();
  m_device->queue.Submit(1, &commands);
  gpu_prof::after_submit();
  ASSERT_TRUE(gpu_prof::wait_idle(WaitNs));
  EXPECT_TRUE(gpu_prof::testing::resources_created());
  gpu_prof::testing::hold_readbacks(false);
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
    debug::shutdown_classification();
    debug::shutdown();
    scene::testing::reset();
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
  const auto tabs = debug::tabs();
  for (const std::string_view name : {"Textures", "Fake shadows"}) {
    EXPECT_NE(std::ranges::find(tabs, name, &debug::Tab::name), tabs.end()) << name;
  }
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
  EXPECT_EQ(names, (std::vector<std::string>{"Overview", "Alpha", "Graph", "Textures", "Zeta", "Fake shadows"}));
  const auto tabs = debug::tabs();
  const auto zeta = std::ranges::find(tabs, std::string{"Zeta"}, &debug::Tab::name);
  ASSERT_NE(zeta, tabs.end());
  EXPECT_EQ(zeta->order, 50);
  const auto textures = std::ranges::find(tabs, std::string{"Textures"}, &debug::Tab::name);
  ASSERT_NE(textures, tabs.end());
  EXPECT_EQ(textures->order, debug::TexturesTabOrder);
  const auto fakeShadows = std::ranges::find(tabs, std::string{"Fake shadows"}, &debug::Tab::name);
  ASSERT_NE(fakeShadows, tabs.end());
  EXPECT_EQ(fakeShadows->order, debug::FakeShadowsTabOrder);
}

TEST_F(RenderDebugWindow, TabVisibleFollowsSelection) {
  debug::register_tab("Alpha", 5, draw_nothing);
  debug::register_tab("Textures", debug::TexturesTabOrder, draw_nothing);
  EXPECT_FALSE(debug::tab_visible("Textures"));
  bool open = true;
  {
    const ImGuiFrame imgui;
    for (int i = 0; i < 2; ++i) {
      imgui.frame([&] { draw_debug_window(&open); });
      debug::on_latch();
    }
    EXPECT_TRUE(debug::tab_visible("Alpha"));
    EXPECT_FALSE(debug::tab_visible("Textures"));
  }
  debug::register_tab("Alpha", 60, draw_nothing);
  {
    const ImGuiFrame imgui;
    for (int i = 0; i < 2; ++i) {
      imgui.frame([&] { draw_debug_window(&open); });
      debug::on_latch();
    }
    EXPECT_TRUE(debug::tab_visible("Textures"));
    EXPECT_FALSE(debug::tab_visible("Alpha"));
  }
  debug::on_latch();
  debug::on_latch();
  EXPECT_FALSE(debug::tab_visible("Textures"));
  EXPECT_FALSE(debug::tab_visible("Missing"));
}

TEST_F(RenderDebugWindow, SceneTabRegisteredAtSceneOrder) {
  debug::initialize();
  debug::initialize_classification();
  const auto tabs = debug::tabs();
  const auto scene = std::ranges::find(tabs, std::string{"Scene"}, &debug::Tab::name);
  ASSERT_NE(scene, tabs.end());
  EXPECT_EQ(scene->order, debug::SceneTabOrder);
  EXPECT_EQ(scene->draw, &debug::draw_scene_tab);
  EXPECT_TRUE(std::ranges::is_sorted(tabs, {}, &debug::Tab::order));
  const auto graph = std::ranges::find(tabs, std::string{"Graph"}, &debug::Tab::name);
  const auto textures = std::ranges::find(tabs, std::string{"Textures"}, &debug::Tab::name);
  ASSERT_NE(graph, tabs.end());
  ASSERT_NE(textures, tabs.end());
  EXPECT_LT(graph, scene);
  EXPECT_LT(scene, textures);
}

TEST_F(RenderDebugWindow, SceneTabDrawsHeadless) {
  const ImGuiFrame imgui;
  const auto draw_tab = [&] {
    imgui.frame([] {
      ImGui::Begin("t");
      debug::draw_scene_tab();
      ImGui::End();
    });
  };
  draw_tab();

  // A frame with a MAIN camera, one object draw and one scope-only draw, published with records and objects
  scene::Classifier& classifier = scene::testing::classifier();
  classifier.reserve();
  classifier.begin_frame();
  classifier.set_record(true);
  classifier.set_objects(true);
  classifier.camera(scene::Role::Main);
  EXPECT_TRUE(classifier.stage(Stage::SceneBegin));
  classifier.object_begin({.id = 0x10, .generation = 1, .categories = scene::category::Terrain});
  classifier.route({.mainCamera = true, .perspective = true, .vertices = 4});
  classifier.object_end();
  classifier.scope_push({.categories = scene::category::Emissive});
  classifier.route({.mainCamera = true, .indexed = true, .vertices = 4});
  classifier.scope_pop();
  scene::seal_frame();
  const auto report = scene::latest_report();
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->draws.size(), 2u);
  ASSERT_EQ(report->objects.size(), 1u);
  draw_tab();

  debug::set_view_override(static_cast<uint16_t>(debug::ClassView::Object));
  debug::set_view_bit(scene::category::Decal);
  draw_tab();
  EXPECT_EQ(debug::view_override(), static_cast<uint16_t>(debug::ClassView::Object));
  EXPECT_EQ(debug::view_bit(), scene::category::Decal);
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

TEST_F(RenderDebugWindow, ViewsTabDrawsHeadless) {
  ASSERT_NE(detail::latch_frame(), nullptr);
  const ImGuiFrame imgui;
  debug::initialize();
  debug::initialize_classification();
  const auto tabs = debug::tabs();
  const auto views = std::ranges::find(tabs, std::string{"Views"}, &debug::Tab::name);
  const auto graph = std::ranges::find(tabs, std::string{"Graph"}, &debug::Tab::name);
  const auto scene = std::ranges::find(tabs, std::string{"Scene"}, &debug::Tab::name);
  ASSERT_NE(views, tabs.end());
  ASSERT_NE(graph, tabs.end());
  ASSERT_NE(scene, tabs.end());
  EXPECT_EQ(views->order, debug::ViewsTabOrder);
  EXPECT_EQ(views->draw, &debug::draw_views_tab);
  EXPECT_LT(graph, views);
  EXPECT_LT(views, scene);
  ASSERT_TRUE(debug::capture::request({.resource = "gx.color", .anchor = "phase.pre_taa"}).has_value());
  const auto draw_tab = [&] {
    imgui.frame([] {
      ImGui::Begin("t");
      debug::draw_views_tab();
      ImGui::End();
    });
  };
  draw_tab();
  debug::set_view_override(debug::FirstGbufView);
  debug::views_on_latch();
  draw_tab();
  EXPECT_EQ(debug::gbuf_view::active_view(), debug::find_gbuf_view(debug::FirstGbufView));
}

constexpr wgpu::TextureUsage SceneTargetUsage =
    wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc;

detail::GraphEnvironment scene_env(bool gbuffer, uint32_t samples = 1) {
  detail::GraphEnvironment env;
  env.efbSize = {64, 64};
  env.outputSize = {64, 64};
  const auto set = [&](detail::GxResource resource, wgpu::TextureFormat format, wgpu::TextureUsage usage) {
    env.gxFormats[static_cast<size_t>(resource)] = format;
    env.gxUsage[static_cast<size_t>(resource)] = usage;
  };
  set(detail::GxResource::Color, wgpu::TextureFormat::BGRA8Unorm, SceneTargetUsage | wgpu::TextureUsage::CopyDst);
  set(detail::GxResource::Depth, wgpu::TextureFormat::Depth32Float,
      wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding);
  if (gbuffer) {
    set(detail::GxResource::Normal, wgpu::TextureFormat::RGB10A2Unorm, SceneTargetUsage);
    set(detail::GxResource::Motion, wgpu::TextureFormat::RG16Float, SceneTargetUsage);
    set(detail::GxResource::Reactive, wgpu::TextureFormat::R8Unorm, SceneTargetUsage);
    set(detail::GxResource::Fog, wgpu::TextureFormat::RGBA8Uint, SceneTargetUsage);
  }
  env.gxSampleCount = samples;
  return env;
}

std::shared_ptr<detail::CompiledGraph> compile_env(const detail::GraphEnvironment& env) {
  return detail::compile_graph(detail::latched_settings(), caps(), env);
}

size_t position(const detail::CompiledGraph& graph, Stage stage, std::string_view name) {
  const std::vector<detail::NodeId>& nodes = graph.stages[static_cast<size_t>(stage)].nodes;
  const auto it = std::ranges::find(nodes, graph.find_node(name));
  EXPECT_NE(it, nodes.end()) << name;
  return static_cast<size_t>(it - nodes.begin());
}

const detail::CompiledUse* find_use(const detail::CompiledGraph& graph, detail::NodeId node, std::string_view name) {
  for (const detail::CompiledUse& use : graph.node_uses(node)) {
    if (graph.resources[use.resource].name == name) {
      return &use;
    }
  }
  return nullptr;
}

using UseSummary = std::tuple<std::string, detail::Access, Missing, wgpu::TextureSampleType>;

struct PassSummary {
  std::string name;
  std::vector<UseSummary> uses;
  uint64_t layoutHash = 0;
  uint32_t paramsSize = 0;
  detail::CullReason status = detail::CullReason::None;
  bool sink = false;
  uint16_t order = 0;
  bool operator==(const PassSummary&) const = default;
};

std::vector<PassSummary> summarize_passes(const detail::CompiledGraph& graph) {
  std::vector<PassSummary> out;
  for (const std::string_view name :
       {detail::composite::CopyPass, detail::composite::Pass, std::string_view{"depth.pyramid"}, detail::ao::gtao::Pass,
        std::string_view{"ao.denoise"}, std::string_view{"ao.upsample"}}) {
    const detail::NodeId node = graph.find_node(name);
    EXPECT_NE(node, detail::InvalidNode) << name;
    if (node == detail::InvalidNode) {
      continue;
    }
    PassSummary& summary = out.emplace_back();
    summary.name = name;
    for (const detail::CompiledUse& use : graph.node_uses(node)) {
      summary.uses.emplace_back(graph.resources[use.resource].name, use.access, use.missing, use.options.sampleType);
    }
    summary.layoutHash = graph.bindingPlans[node].layoutHash;
    summary.paramsSize = graph.nodes[node].paramsSize;
    summary.status = graph.nodes[node].status;
    summary.sink = graph.nodes[node].sink;
    summary.order = graph.nodes[node].order;
  }
  return out;
}

class RenderDebugViews : public RenderDebugWindow {
protected:
  void SetUp() override {
    RenderDebugWindow::SetUp();
    debug::gbuf_view::shutdown();
    debug::capture::testing::reset();
  }

  void TearDown() override {
    debug::set_view_override(0);
    debug::capture::testing::reset();
    if (m_composite) {
      stop_composite();
    }
    RenderDebugWindow::TearDown();
  }

  std::shared_ptr<detail::CompiledGraph> select(uint16_t id, bool gbuffer) {
    debug::set_view_override(id);
    debug::views_on_latch();
    return compile_env(scene_env(gbuffer));
  }

  void start_composite() {
    stop_composite();
    detail::ao::initialize();
    detail::composite::initialize();
    m_composite = true;
  }

  static void stop_composite() {
    detail::composite::shutdown();
    detail::composite::testing::reset();
    detail::ao::shutdown();
    detail::ao::testing::reset();
  }

  static std::shared_ptr<detail::CompiledGraph> compile_ao(const detail::GraphEnvironment& env) {
    Settings settings = detail::latched_settings();
    settings.ao = Quality::High;
    return detail::compile_graph(settings, caps(), env, {.sceneDepthSource = DepthSource::Direct});
  }

  std::shared_ptr<detail::CompiledGraph> select_ao(uint16_t id) {
    debug::set_view_override(id);
    debug::views_on_latch();
    return compile_ao(scene_env(true));
  }

  bool m_composite = false;
};

TEST_F(RenderDebugViews, SelectionRegistersSnapshotAndBlit) {
  const auto graph = select(18, true);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(debug::gbuf_view::active_view(), debug::find_gbuf_view(18));
  const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
  const detail::NodeId blit = graph->find_node(debug::gbuf_view::BlitPass);
  ASSERT_NE(snapshot, detail::InvalidNode);
  ASSERT_NE(blit, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::None);
  EXPECT_EQ(graph->nodes[blit].status, detail::CullReason::None);
  EXPECT_EQ(graph->nodes[snapshot].stage, Stage::OpaqueDone);
  EXPECT_EQ(graph->nodes[snapshot].passKind, detail::PassKind::Render);
  EXPECT_EQ(graph->nodes[snapshot].order, debug::gbuf_view::LastPassOrder);
  EXPECT_EQ(graph->nodes[blit].stage, Stage::BeforeHud);
  EXPECT_GT(position(*graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
            position(*graph, Stage::OpaqueDone, "phase.composite"));
  EXPECT_LT(position(*graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
            position(*graph, Stage::OpaqueDone, "gx.overlay"));
  EXPECT_GT(position(*graph, Stage::BeforeHud, debug::gbuf_view::BlitPass),
            position(*graph, Stage::BeforeHud, "phase.before_hud"));
  const detail::CompiledUse* source = find_use(*graph, snapshot, "gbuf.fog");
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->access, detail::Access::Sampled);
  EXPECT_EQ(source->options.sampleType, wgpu::TextureSampleType::Uint);
  const detail::CompiledUse* created = find_use(*graph, snapshot, debug::gbuf_view::SnapshotResource);
  ASSERT_NE(created, nullptr);
  EXPECT_EQ(created->access, detail::Access::ColorWrite);
  const detail::CompiledUse* target = find_use(*graph, blit, "gx.color");
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->access, detail::Access::ColorRMW);
  EXPECT_NE(find_use(*graph, blit, debug::gbuf_view::SnapshotResource), nullptr);
}

TEST_F(RenderDebugViews, MotionViewSnapshotsAtPrePost) {
  const auto graph = select(16, true);
  ASSERT_NE(graph, nullptr);
  const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
  ASSERT_NE(snapshot, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::None);
  EXPECT_EQ(graph->nodes[snapshot].stage, Stage::PrePost);
  EXPECT_GT(position(*graph, Stage::PrePost, debug::gbuf_view::SnapshotPass),
            position(*graph, Stage::PrePost, "phase.post_taa"));
  const detail::CompiledUse* source = find_use(*graph, snapshot, "gbuf.motion");
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
}

TEST_F(RenderDebugViews, UnavailableTargetSkipsSnapshot) {
  const auto graph = select(16, false);
  ASSERT_NE(graph, nullptr);
  const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
  const detail::NodeId blit = graph->find_node(debug::gbuf_view::BlitPass);
  ASSERT_NE(snapshot, detail::InvalidNode);
  ASSERT_NE(blit, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::Skipped);
  EXPECT_EQ(graph->nodes[blit].status, detail::CullReason::Skipped);
  EXPECT_TRUE(graph->stage_empty(Stage::BeforeHud));
}

TEST_F(RenderDebugViews, DeselectRestoresGraphHash) {
  register_test_pass("test.kept", Stage::OpaqueDone, 0);
  const auto plain = select(0, true);
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(debug::gbuf_view::active_view(), nullptr);
  const uint64_t generation = debug::gbuf_view::generation();
  const auto viewed = select(21, true);
  ASSERT_NE(viewed, nullptr);
  EXPECT_NE(viewed->graphHash, plain->graphHash);
  EXPECT_GT(debug::gbuf_view::generation(), generation);
  const auto restored = select(0, true);
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->graphHash, plain->graphHash);
  EXPECT_EQ(restored->find_node(debug::gbuf_view::SnapshotPass), detail::InvalidNode);
  EXPECT_EQ(debug::gbuf_view::active_view(), nullptr);

  start_composite();
  const auto composited = select_ao(0);
  ASSERT_NE(composited, nullptr);
  for (const uint16_t id : {uint16_t{23}, uint16_t{25}, uint16_t{26}}) {
    SCOPED_TRACE(id);
    const uint64_t before = debug::gbuf_view::generation();
    const auto shown = select_ao(id);
    ASSERT_NE(shown, nullptr);
    const detail::NodeId snapshot = shown->find_node(debug::gbuf_view::SnapshotPass);
    ASSERT_NE(snapshot, detail::InvalidNode);
    EXPECT_EQ(shown->nodes[snapshot].status, detail::CullReason::None);
    EXPECT_NE(shown->graphHash, composited->graphHash);
    EXPECT_GT(debug::gbuf_view::generation(), before);
    const auto back = select_ao(0);
    ASSERT_NE(back, nullptr);
    EXPECT_EQ(back->graphHash, composited->graphHash);
    EXPECT_EQ(back->find_node(debug::gbuf_view::SnapshotPass), detail::InvalidNode);
  }
}

TEST_F(RenderDebugViews, DecompositionSelectionRegistersPerModeUses) {
  start_composite();
  struct Row {
    uint16_t id;
    std::string_view input;
    std::string_view after;
    std::string_view before;
  };
  constexpr std::array<Row, 4> Rows{{
      {24, "", "phase.volumetrics", "phase.composite"},
      {25, "legacyTex", "phase.composite", "gx.overlay"},
      {26, "gbuf.reactive", "phase.volumetrics", "phase.composite"},
      {27, "ao.final", "phase.volumetrics", "phase.composite"},
  }};
  for (const Row& row : Rows) {
    SCOPED_TRACE(row.id);
    const auto graph = select_ao(row.id);
    ASSERT_NE(graph, nullptr);
    const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
    const detail::NodeId blit = graph->find_node(debug::gbuf_view::BlitPass);
    ASSERT_NE(snapshot, detail::InvalidNode);
    ASSERT_NE(blit, detail::InvalidNode);
    EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::None);
    EXPECT_EQ(graph->nodes[blit].status, detail::CullReason::None);
    EXPECT_EQ(graph->nodes[snapshot].stage, Stage::OpaqueDone);
    EXPECT_EQ(graph->nodes[snapshot].passKind, detail::PassKind::Render);
    EXPECT_EQ(graph->nodes[snapshot].order, debug::gbuf_view::LastPassOrder);
    EXPECT_GT(position(*graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
              position(*graph, Stage::OpaqueDone, row.after));
    EXPECT_LT(position(*graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
              position(*graph, Stage::OpaqueDone, row.before));
    std::vector<std::string_view> expected{"gx.color", "gbuf.fog", "gx.normal"};
    if (!row.input.empty()) {
      expected.push_back(row.input);
    }
    std::vector<std::string_view> sampled;
    for (const detail::CompiledUse& use : graph->node_uses(snapshot)) {
      const std::string_view name = graph->resources[use.resource].name;
      if (use.access == detail::Access::ColorWrite) {
        EXPECT_EQ(name, debug::gbuf_view::SnapshotResource);
        continue;
      }
      EXPECT_EQ(use.access, detail::Access::Sampled) << name;
      EXPECT_EQ(use.missing, Missing::DisablePass) << name;
      EXPECT_EQ(use.options.sampleType,
                name == "gbuf.fog" ? wgpu::TextureSampleType::Uint : wgpu::TextureSampleType::UnfilterableFloat)
          << name;
      sampled.push_back(name);
    }
    EXPECT_EQ(sampled, expected);
    const detail::BindingPlan& plan = graph->bindingPlans[snapshot];
    ASSERT_EQ(plan.slots.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(graph->resources[plan.slots[i].resource].name, expected[i]);
      EXPECT_EQ(plan.slots[i].binding, i);
    }
    const detail::CompiledResource& target = graph->resources[graph->find_resource(debug::gbuf_view::SnapshotResource)];
    EXPECT_EQ(target.tex.format, wgpu::TextureFormat::RGBA8Unorm);
    EXPECT_NE(find_use(*graph, blit, "gx.color"), nullptr);
  }
}

TEST_F(RenderDebugViews, IdentityViewSortsAfterCompositeWriter) {
  start_composite();
  const auto expect_after = [](const detail::CompiledGraph& graph) {
    const detail::NodeId composite = graph.find_node(detail::composite::Pass);
    const detail::NodeId view = graph.find_node(debug::gbuf_view::SnapshotPass);
    ASSERT_NE(composite, detail::InvalidNode);
    ASSERT_NE(view, detail::InvalidNode);
    EXPECT_EQ(graph.nodes[composite].status, detail::CullReason::None);
    EXPECT_EQ(graph.nodes[view].status, detail::CullReason::None);
    const detail::ResourceId color = graph.find_resource(resource::GxColor);
    EXPECT_TRUE(std::ranges::any_of(graph.edges, [&](const detail::Edge& edge) {
      return edge.from == composite && edge.to == view && edge.kind == detail::EdgeKind::Raw && edge.resource == color;
    }));
    EXPECT_GT(position(graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
              position(graph, Stage::OpaqueDone, detail::composite::Pass));
    EXPECT_LT(position(graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
              position(graph, Stage::OpaqueDone, "gx.overlay"));
  };
  const auto first = select_ao(25);
  ASSERT_NE(first, nullptr);
  expect_after(*first);
  detail::composite::shutdown();
  detail::composite::initialize();
  const auto reregistered = compile_ao(scene_env(true));
  ASSERT_NE(reregistered, nullptr);
  expect_after(*reregistered);
}

TEST_F(RenderDebugViews, PreCompositeRowsSeeLegacyColour) {
  start_composite();
  for (const uint16_t id : {uint16_t{24}, uint16_t{26}, uint16_t{27}}) {
    SCOPED_TRACE(id);
    const auto graph = select_ao(id);
    ASSERT_NE(graph, nullptr);
    const std::vector<detail::NodeId>& program = graph->stages[static_cast<size_t>(Stage::OpaqueDone)].nodes;
    const auto copy = std::ranges::find(program, graph->find_node(detail::composite::CopyPass));
    const auto view = std::ranges::find(program, graph->find_node(debug::gbuf_view::SnapshotPass));
    const auto composite = std::ranges::find(program, graph->find_node(detail::composite::Pass));
    ASSERT_NE(copy, program.end());
    ASSERT_NE(view, program.end());
    ASSERT_NE(composite, program.end());
    EXPECT_LT(copy, view);
    EXPECT_LT(view, composite);
    for (auto it = std::next(copy); it != view; ++it) {
      for (const detail::CompiledUse& use : graph->node_uses(*it)) {
        EXPECT_FALSE(use.access == detail::Access::ColorRMW && graph->resources[use.resource].name == "gx.color")
            << graph->nodes[*it].name;
      }
    }
    const detail::CompiledUse* color = find_use(*graph, *view, "gx.color");
    ASSERT_NE(color, nullptr);
    EXPECT_EQ(color->access, detail::Access::Sampled);
    EXPECT_EQ(detail::composite::check_contract(*graph), std::nullopt);
  }
}

TEST_F(RenderDebugViews, ViewLeavesOtherPassesUnchanged) {
  start_composite();
  const auto plain = select_ao(0);
  ASSERT_NE(plain, nullptr);
  const std::vector<PassSummary> expected = summarize_passes(*plain);
  ASSERT_EQ(expected.size(), 6u);
  for (const PassSummary& pass : expected) {
    EXPECT_EQ(pass.status, detail::CullReason::None) << pass.name;
  }
  for (const uint16_t id : {uint16_t{25}, uint16_t{27}}) {
    SCOPED_TRACE(id);
    const auto viewed = select_ao(id);
    ASSERT_NE(viewed, nullptr);
    EXPECT_EQ(viewed->nodes[viewed->find_node(debug::gbuf_view::SnapshotPass)].status, detail::CullReason::None);
    EXPECT_EQ(summarize_passes(*viewed), expected);
    EXPECT_EQ(detail::composite::check_contract(*viewed), std::nullopt);
  }
}

TEST_F(RenderDebugViews, DecompositionSkipsWithoutGbuffer) {
  for (uint16_t id = 24; id <= 27; ++id) {
    SCOPED_TRACE(id);
    const auto graph = select(id, false);
    ASSERT_NE(graph, nullptr);
    const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
    const detail::NodeId blit = graph->find_node(debug::gbuf_view::BlitPass);
    ASSERT_NE(snapshot, detail::InvalidNode);
    ASSERT_NE(blit, detail::InvalidNode);
    EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::Skipped);
    EXPECT_EQ(graph->nodes[blit].status, detail::CullReason::Skipped);
    EXPECT_TRUE(graph->stage_empty(Stage::BeforeHud));
  }
}

TEST_F(RenderDebugViews, AoRowWithoutProducerSkipsBlit) {
  for (const uint16_t id : {uint16_t{23}, uint16_t{28}, uint16_t{29}}) {
    SCOPED_TRACE(id);
    const auto graph = select(id, true);
    ASSERT_NE(graph, nullptr);
    const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
    const detail::NodeId blit = graph->find_node(debug::gbuf_view::BlitPass);
    ASSERT_NE(snapshot, detail::InvalidNode);
    ASSERT_NE(blit, detail::InvalidNode);
    EXPECT_NE(graph->nodes[snapshot].status, detail::CullReason::None);
    EXPECT_NE(graph->nodes[snapshot].status, detail::CullReason::Skipped);
    EXPECT_NE(graph->nodes[blit].status, detail::CullReason::None);
    EXPECT_TRUE(graph->stage_empty(Stage::BeforeHud));
  }
}

TEST_F(RenderDebugViews, AoRawHasSingleWriter) {
  start_composite();
  const auto graph = select_ao(28);
  ASSERT_NE(graph, nullptr);
  const detail::ResourceId raw = graph->find_resource(detail::ao::gtao::Resource);
  ASSERT_NE(raw, detail::InvalidResource);
  std::vector<std::string_view> writers;
  for (const detail::CompiledUse& use : graph->uses) {
    if (use.resource == raw && use.access != detail::Access::Sampled) {
      writers.push_back(graph->nodes[use.node].name);
    }
  }
  EXPECT_EQ(writers, (std::vector<std::string_view>{detail::ao::gtao::Pass}));
  EXPECT_EQ(graph->resources[raw].creator, graph->find_node(detail::ao::gtao::Pass));
  const detail::NodeId snapshot = graph->find_node(debug::gbuf_view::SnapshotPass);
  ASSERT_NE(snapshot, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[snapshot].status, detail::CullReason::None);
  const detail::CompiledUse* read = find_use(*graph, snapshot, detail::ao::gtao::Resource);
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(read->access, detail::Access::Sampled);
  EXPECT_EQ(read->options.sampleType, wgpu::TextureSampleType::UnfilterableFloat);
  EXPECT_GT(position(*graph, Stage::OpaqueDone, debug::gbuf_view::SnapshotPass),
            position(*graph, Stage::OpaqueDone, detail::ao::gtao::Pass));
}

TEST_F(RenderDebugViews, BlitTargetKeepsAlpha) {
  const shaders::RenderPipelineDesc blit = debug::gbuf_view::blit_pipeline_desc(scene_env(true));
  ASSERT_EQ(blit.colorCount, 1u);
  EXPECT_EQ(blit.colors[0].format, wgpu::TextureFormat::BGRA8Unorm);
  EXPECT_EQ(blit.colors[0].writeMask,
            wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue);
  EXPECT_FALSE(blit.colors[0].blend);
  EXPECT_EQ(blit.fragmentEntry, "fs_blit");
  EXPECT_EQ(blit.sampleCount, 1u);
  EXPECT_TRUE(blit.defines.empty());
  EXPECT_EQ(debug::gbuf_view::blit_pipeline_desc(scene_env(true, 4)).sampleCount, 4u);

  const shaders::RenderPipelineDesc fog = debug::gbuf_view::snapshot_pipeline_desc(*debug::find_gbuf_view(18));
  ASSERT_EQ(fog.defines.size(), 1u);
  EXPECT_EQ(fog.defines[0], "SOURCE_UINT");
  ASSERT_EQ(fog.constants.size(), 1u);
  EXPECT_EQ(fog.constants[0].name, "VIEW_MODE");
  EXPECT_EQ(fog.constants[0].value, 2.0);
  EXPECT_EQ(fog.colors[0].format, wgpu::TextureFormat::RGBA8Unorm);
  EXPECT_EQ(fog.colors[0].writeMask, wgpu::ColorWriteMask::All);
  const shaders::RenderPipelineDesc motion = debug::gbuf_view::snapshot_pipeline_desc(*debug::find_gbuf_view(16));
  EXPECT_TRUE(motion.defines.empty());
  EXPECT_EQ(motion.constants[0].value, 0.0);
  for (const debug::GbufView& view : debug::GbufViews) {
    if (view.kind == debug::GbufViewKind::Target) {
      ASSERT_EQ(debug::gbuf_view::snapshot_pipeline_desc(view).constants.size(), 1u) << view.name;
      EXPECT_EQ(debug::gbuf_view::snapshot_pipeline_desc(view).constants[0].value, view.mode) << view.name;
    }
  }
}

void declare_texture_source(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.after("phase.geometry");
  builder.create(
      "test.tex",
      {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = detail::SizeClass::Absolute, .width = 4, .height = 4},
      detail::Access::StorageWrite);
  builder.side_effect();
}

void declare_late(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.after("phase.composite");
  builder.side_effect();
}

class RenderDebugCapture : public RenderDebugViews {};

TEST_F(RenderDebugCapture, ParsesEnvList) {
  const auto requests = debug::capture::parse_env("gbuf.fog@gx@OpaqueDone;gx.color@phase.pre_taa;;bad;@x;y@", "out");
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_EQ(requests[0].resource, "gbuf.fog");
  EXPECT_EQ(requests[0].anchor, "gx@OpaqueDone");
  EXPECT_TRUE(requests[0].latest);
  EXPECT_EQ(requests[0].dir, "out");
  EXPECT_EQ(requests[1].resource, "gx.color");
  EXPECT_EQ(requests[1].anchor, "phase.pre_taa");
  EXPECT_TRUE(debug::capture::parse_env("", "out").empty());
}

TEST_F(RenderDebugCapture, CopyLayoutPadsRows) {
  const auto fog = debug::capture::copy_layout(wgpu::TextureFormat::RGBA8Uint, 33, 3);
  ASSERT_TRUE(fog.has_value());
  EXPECT_EQ(fog->bytesPerPixel, 4u);
  EXPECT_EQ(fog->rowBytes, 132u);
  EXPECT_EQ(fog->paddedRowBytes, 256u);
  EXPECT_EQ(fog->bufferSize, 768u);
  const auto reactive = debug::capture::copy_layout(wgpu::TextureFormat::R8Unorm, 300, 2);
  ASSERT_TRUE(reactive.has_value());
  EXPECT_EQ(reactive->rowBytes, 300u);
  EXPECT_EQ(reactive->paddedRowBytes, 512u);
  const auto wide = debug::capture::copy_layout(wgpu::TextureFormat::RGBA32Uint, 16, 1);
  ASSERT_TRUE(wide.has_value());
  EXPECT_EQ(wide->paddedRowBytes, 256u);
  EXPECT_EQ(debug::capture::copy_layout(wgpu::TextureFormat::RGBA16Float, 1, 1)->bytesPerPixel, 8u);
  EXPECT_FALSE(debug::capture::copy_layout(wgpu::TextureFormat::Depth32Float, 4, 4).has_value());
  EXPECT_FALSE(debug::capture::copy_layout(wgpu::TextureFormat::RGBA8Unorm, 0, 4).has_value());
}

TEST_F(RenderDebugCapture, FogSummaryCountsRecords) {
  const std::vector<uint8_t> records{
      0, 0, 0, 0, 10, 20, 30, (5 << 1) | 1, 10, 20, 30, 4 << 1, 40, 50, 60, 0, 0, 0, 0, 1 << 1, 0, 0, 0, 0,
  };
  const debug::capture::FogSummary fog = debug::capture::summarize_fog(records);
  EXPECT_EQ(fog.records, 6u);
  EXPECT_EQ(fog.nonZero, 4u);
  EXPECT_EQ(fog.withFog, 3u);
  EXPECT_EQ(fog.locked, 1u);
  EXPECT_EQ(fog.distinct, (std::vector<uint32_t>{0x0A141E, 0x28323C, 0x000000}));
  EXPECT_FALSE(fog.distinctOverflow);
  EXPECT_EQ(fog.qHistogram[0], 1u);
  EXPECT_EQ(fog.qHistogram[1], 1u);
  EXPECT_EQ(fog.qHistogram[4], 1u);
  EXPECT_EQ(fog.qHistogram[5], 1u);

  std::vector<uint8_t> many;
  for (uint8_t i = 1; i <= 70; ++i) {
    many.insert(many.end(), {i, 0, 0, 2});
  }
  const debug::capture::FogSummary overflow = debug::capture::summarize_fog(many);
  EXPECT_EQ(overflow.distinct.size(), debug::capture::MaxDistinctColors);
  EXPECT_TRUE(overflow.distinctOverflow);
  EXPECT_EQ(overflow.withFog, 70u);
}

TEST_F(RenderDebugCapture, MotionSummaryInPixels) {
  const auto texel = [](uint16_t x, uint16_t y) {
    return std::array<uint8_t, 4>{static_cast<uint8_t>(x & 0xFF), static_cast<uint8_t>(x >> 8),
                                  static_cast<uint8_t>(y & 0xFF), static_cast<uint8_t>(y >> 8)};
  };
  std::vector<uint8_t> bytes;
  for (const auto& value :
       {texel(0x3000, 0xB400), texel(0x2C00, 0), texel(0x4400, 0x4400), texel(0, 0), texel(0x7C00, 0)}) {
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  const debug::capture::MotionSummary motion = debug::capture::summarize_motion(bytes, 64, 32);
  EXPECT_EQ(motion.nonZero, 4u);
  EXPECT_DOUBLE_EQ(motion.maxPx, 5.657);
  EXPECT_DOUBLE_EQ(motion.meanPx, 3.828);
  const debug::capture::NormalSummary normal =
      debug::capture::summarize_normal(std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0xC0, 0, 0, 0, 0x40});
  EXPECT_EQ(normal.codeCounts, (std::array<uint64_t, 4>{1, 1, 0, 1}));
}

TEST_F(RenderDebugCapture, HeaderJson) {
  const debug::capture::Capture fog{.resource = "gbuf.fog",
                                    .anchor = "gx@OpaqueDone",
                                    .frame = 7,
                                    .format = wgpu::TextureFormat::RGBA8Uint,
                                    .width = 2,
                                    .height = 1,
                                    .bytes = {0, 0, 0, 0, 1, 2, 3, 3}};
  const std::string text = debug::capture::header_json(fog, "gbuf.fog@gx@OpaqueDone.bin");
  const json::Doc doc = json::read(text);
  ASSERT_NE(doc, nullptr) << text;
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "schema")), "aurora.render.capture");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "version")), 1u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "resource")), "gbuf.fog");
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "anchor")), "gx@OpaqueDone");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "frame")), 7u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "format")), "RGBA8Uint");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "width")), 2u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "height")), 1u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "bytesPerPixel")), 4u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "rowBytes")), 8u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "bin")), "gbuf.fog@gx@OpaqueDone.bin");
  yyjson_val* summary = yyjson_obj_get(root, "fog");
  ASSERT_NE(summary, nullptr);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(summary, "records")), 2u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(summary, "nonZero")), 1u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(summary, "locked")), 1u);
  EXPECT_EQ(yyjson_arr_size(yyjson_obj_get(summary, "qHistogram")), 128u);
  yyjson_val* distinct = yyjson_obj_get(summary, "distinct");
  ASSERT_EQ(yyjson_arr_size(distinct), 1u);
  EXPECT_STREQ(yyjson_get_str(yyjson_arr_get(distinct, 0)), "#010203");
  EXPECT_EQ(yyjson_obj_get(root, "motion"), nullptr);
  EXPECT_TRUE(text.ends_with("}\n"));

  debug::capture::Capture motion = fog;
  motion.format = wgpu::TextureFormat::RG16Float;
  const json::Doc motionDoc = json::read(debug::capture::header_json(motion, "m.bin"));
  ASSERT_NE(motionDoc, nullptr);
  EXPECT_NE(yyjson_obj_get(yyjson_doc_get_root(motionDoc.get()), "motion"), nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(motionDoc.get()), "fog"), nullptr);
  debug::capture::Capture normal = fog;
  normal.format = wgpu::TextureFormat::RGB10A2Unorm;
  const json::Doc normalDoc = json::read(debug::capture::header_json(normal, "n.bin"));
  ASSERT_NE(normalDoc, nullptr);
  yyjson_val* codes = yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(normalDoc.get()), "normal"), "codeCounts");
  EXPECT_EQ(yyjson_arr_size(codes), 4u);
}

TEST_F(RenderDebugCapture, ViewSummaryCountsColours) {
  std::vector<uint8_t> bytes;
  const auto add = [&bytes](uint32_t rgb, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      bytes.insert(bytes.end(),
                   {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8), static_cast<uint8_t>(rgb), 255});
    }
  };
  add(0x000000, 5);
  add(0x474747, 3);
  add(0xFF8000, 2);
  add(0x474747, 3);
  add(0xFF0000, 1);
  const debug::capture::ColorSummary summary = debug::capture::summarize_colors(bytes);
  EXPECT_EQ(summary.nonBlack, 9u);
  EXPECT_EQ(summary.distinct,
            (std::vector<debug::capture::ColorCount>{{0x474747, 6}, {0x000000, 5}, {0xFF8000, 2}, {0xFF0000, 1}}));
  EXPECT_FALSE(summary.distinctOverflow);

  std::vector<uint8_t> many;
  for (uint32_t i = 1; i <= 65; ++i) {
    many.insert(many.end(), {static_cast<uint8_t>(i), 0, 0, 255});
  }
  const debug::capture::ColorSummary full = debug::capture::summarize_colors(std::span{many}.first(64 * 4));
  EXPECT_EQ(full.distinct.size(), debug::capture::MaxDistinctColors);
  EXPECT_FALSE(full.distinctOverflow);
  const debug::capture::ColorSummary overflow = debug::capture::summarize_colors(many);
  EXPECT_EQ(overflow.distinct.size(), debug::capture::MaxDistinctColors);
  EXPECT_TRUE(overflow.distinctOverflow);
  EXPECT_EQ(overflow.nonBlack, 65u);

  const debug::capture::Capture view{.resource = std::string{debug::gbuf_view::SnapshotResource},
                                     .anchor = std::string{debug::gbuf_view::SnapshotPass},
                                     .frame = 7,
                                     .format = wgpu::TextureFormat::RGBA8Unorm,
                                     .width = 14,
                                     .height = 1,
                                     .bytes = bytes};
  const std::string text = debug::capture::header_json(view, "debug.view@debug.view.bin");
  const json::Doc doc = json::read(text);
  ASSERT_NE(doc, nullptr) << text;
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "frame")), 7u);
  yyjson_val* colour = yyjson_obj_get(root, "colour");
  ASSERT_NE(colour, nullptr);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(colour, "nonBlack")), 9u);
  EXPECT_FALSE(yyjson_get_bool(yyjson_obj_get(colour, "distinctOverflow")));
  yyjson_val* distinct = yyjson_obj_get(colour, "distinct");
  ASSERT_EQ(yyjson_arr_size(distinct), 4u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(yyjson_arr_get(distinct, 0), "rgb")), "#474747");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(yyjson_arr_get(distinct, 0), "count")), 6u);
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(yyjson_arr_get(distinct, 3), "rgb")), "#ff0000");
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "schema")), "aurora.render.capture");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "version")), 1u);

  debug::capture::Capture later = view;
  later.frame = 8;
  std::string expected = text;
  const size_t frame = expected.find("\"frame\": 7");
  ASSERT_NE(frame, std::string::npos) << text;
  expected.replace(frame, std::string_view{"\"frame\": 7"}.size(), "\"frame\": 8");
  EXPECT_EQ(debug::capture::header_json(later, "debug.view@debug.view.bin"), expected);

  debug::capture::Capture color = view;
  color.resource = "gx.color";
  const json::Doc plain = json::read(debug::capture::header_json(color, "gx.color.bin"));
  ASSERT_NE(plain, nullptr);
  EXPECT_EQ(yyjson_obj_get(yyjson_doc_get_root(plain.get()), "colour"), nullptr);
}

TEST_F(RenderDebugCapture, ChainAnchorPlacesAtSlotEnd) {
  register_test_pass("test.src", Stage::OpaqueDone, 0, declare_texture_source);
  register_test_pass("test.late", Stage::OpaqueDone, 7, declare_late);
  ASSERT_EQ(debug::capture::request({.resource = "test.tex", .anchor = "phase.composite"}), 0u);
  debug::capture::resolve(nullptr);
  ASSERT_NE(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  const auto graph = compile_env({});
  ASSERT_NE(graph, nullptr);
  const detail::NodeId node = graph->find_node("debug.capture.0");
  ASSERT_NE(node, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, detail::CullReason::None);
  EXPECT_EQ(graph->nodes[node].stage, Stage::OpaqueDone);
  EXPECT_EQ(graph->nodes[node].order, debug::capture::ChainOrder);
  EXPECT_TRUE(graph->nodes[node].sideEffect);
  const size_t capture = position(*graph, Stage::OpaqueDone, "debug.capture.0");
  EXPECT_GT(capture, position(*graph, Stage::OpaqueDone, "phase.composite"));
  EXPECT_GT(capture, position(*graph, Stage::OpaqueDone, "test.late"));
  EXPECT_LT(capture, position(*graph, Stage::OpaqueDone, "gx.overlay"));
  const detail::CompiledUse* source = find_use(*graph, node, "test.tex");
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->access, detail::Access::CopySrc);
}

TEST_F(RenderDebugCapture, PassAnchorPlacesAfterPass) {
  register_test_pass("test.src", Stage::OpaqueDone, 5, declare_texture_source);
  ASSERT_EQ(debug::capture::request({.resource = "test.tex", .anchor = "test.src"}), 0u);
  debug::capture::resolve(nullptr);
  EXPECT_EQ(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  const auto first = compile_env({});
  debug::capture::resolve(first.get());
  ASSERT_NE(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  const auto graph = compile_env({});
  const detail::NodeId node = graph->find_node("debug.capture.0");
  ASSERT_NE(node, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, detail::CullReason::None);
  EXPECT_EQ(graph->nodes[node].order, 5u);
  const size_t capture = position(*graph, Stage::OpaqueDone, "debug.capture.0");
  EXPECT_GT(capture, position(*graph, Stage::OpaqueDone, "test.src"));
  EXPECT_LT(capture, position(*graph, Stage::OpaqueDone, "phase.prepare"));
  const detail::NodeId source = graph->find_node("test.src");
  EXPECT_TRUE(std::ranges::any_of(graph->edges, [&](const detail::Edge& edge) {
    return edge.from == source && edge.to == node && edge.kind == detail::EdgeKind::Anchor;
  }));
}

TEST_F(RenderDebugCapture, EnvPassAnchorResolvesAfterFirstCompile) {
  register_test_pass("test.a", Stage::OpaqueDone, 3, declare_texture_source);
  for (debug::capture::Request& request : debug::capture::parse_env("test.tex@test.a;test.tex@test.missing", "out")) {
    ASSERT_TRUE(debug::capture::request(std::move(request)).has_value());
  }
  debug::capture::resolve(nullptr);
  auto entries = debug::capture::entries();
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].status, debug::capture::Status::Waiting);
  EXPECT_EQ(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  const auto first = compile_env({});
  debug::capture::resolve(first.get());
  ASSERT_NE(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  EXPECT_EQ(debug::capture::entries()[0].status, debug::capture::Status::Armed);
  const auto graph = compile_env({});
  EXPECT_GT(position(*graph, Stage::OpaqueDone, "debug.capture.0"), position(*graph, Stage::OpaqueDone, "test.a"));
  for (uint32_t resolves = 2; resolves < debug::capture::ResolveLimit - 1; ++resolves) {
    debug::capture::resolve(graph.get());
  }
  EXPECT_EQ(debug::capture::entries()[1].status, debug::capture::Status::Waiting);
  debug::capture::resolve(graph.get());
  entries = debug::capture::entries();
  EXPECT_EQ(entries[1].status, debug::capture::Status::Unavailable);
  EXPECT_NE(entries[1].message.find("test.missing"), std::string::npos);
  EXPECT_EQ(entries[0].status, debug::capture::Status::Armed);
}

TEST_F(RenderDebugCapture, MissingCopySrcReportsMissingInput) {
  ASSERT_EQ(debug::capture::request({.resource = "gx.depth", .anchor = "phase.composite"}), 0u);
  debug::capture::resolve(nullptr);
  ASSERT_NE(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  const auto graph = compile_env(scene_env(true));
  const detail::NodeId node = graph->find_node("debug.capture.0");
  ASSERT_NE(node, detail::InvalidNode);
  EXPECT_EQ(graph->nodes[node].status, detail::CullReason::MissingInput);
  debug::capture::resolve(graph.get());
  const auto entries = debug::capture::entries();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].status, debug::capture::Status::MissingInput);
  EXPECT_NE(entries[0].message.find("gx.depth"), std::string::npos);
  EXPECT_EQ(debug::capture::testing::pass_id(0), detail::InvalidPassId);
}

TEST_F(RenderDebugCapture, AfterFrameAnchorUnavailable) {
  ASSERT_EQ(debug::capture::request({.resource = "gx.color", .anchor = "phase.after_frame"}), 0u);
  debug::capture::resolve(nullptr);
  const auto entries = debug::capture::entries();
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].status, debug::capture::Status::Unavailable);
  EXPECT_EQ(debug::capture::testing::pass_id(0), detail::InvalidPassId);
}

std::filesystem::path scratch_dir(std::string_view name) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "aurora_render_debug_test" / name;
  std::filesystem::remove_all(dir);
  return dir;
}

TEST_F(RenderDebugCapture, OnceRequestUnregistersAfterWrite) {
  const std::filesystem::path dir = scratch_dir("once");
  jobs::set_synchronous(true);
  ASSERT_EQ(debug::capture::request({.resource = "gx.color", .anchor = "phase.composite", .dir = dir.string()}), 0u);
  debug::capture::resolve(nullptr);
  ASSERT_NE(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  for (uint32_t i = 1; i < debug::capture::ArmDelay; ++i) {
    debug::capture::resolve(nullptr);
  }
  EXPECT_EQ(debug::capture::entries()[0].status, debug::capture::Status::Waiting);
  debug::capture::resolve(nullptr);
  EXPECT_EQ(debug::capture::entries()[0].status, debug::capture::Status::Armed);
  debug::capture::testing::deliver(0, {.resource = "gx.color",
                                       .anchor = "phase.composite",
                                       .frame = 42,
                                       .format = wgpu::TextureFormat::RGBA8Unorm,
                                       .width = 2,
                                       .height = 2,
                                       .bytes = std::vector<uint8_t>(16, 0x80)});
  jobs::set_synchronous(false);
  EXPECT_EQ(debug::capture::entries()[0].status, debug::capture::Status::Written);
  debug::capture::resolve(nullptr);
  EXPECT_EQ(debug::capture::testing::pass_id(0), detail::InvalidPassId);
  EXPECT_EQ(debug::capture::entries()[0].status, debug::capture::Status::Written);
  EXPECT_EQ(std::filesystem::file_size(dir / "gx.color@phase.composite_42.bin"), 16u);
  EXPECT_TRUE(std::filesystem::exists(dir / "gx.color@phase.composite_42.json"));
  EXPECT_TRUE(std::filesystem::exists(dir / "gx.color@phase.composite_42.png"));
  std::filesystem::remove_all(dir);
}

class RenderDebugCaptureDevice : public RenderDebugCapture {
protected:
  void SetUp() override {
    RenderDebugCapture::SetUp();
    const char* backend = std::getenv("AURORA_TEST_BACKEND");
    if (backend == nullptr) {
      GTEST_SKIP() << "AURORA_TEST_BACKEND is not set";
    }
    m_options.backend = test::parse_test_backend(backend).value_or(wgpu::BackendType::Null);
    std::string error;
    m_device = test::create_test_device(m_options, error);
    if (!m_device) {
      GTEST_SKIP() << "no test device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, m_options);
  }

  void TearDown() override {
    if (m_device) {
      EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
      m_globals.reset();
      m_device.reset();
    }
    RenderDebugCapture::TearDown();
  }

  test::TestDeviceOptions m_options;
  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(RenderDebugCaptureDevice, RoundTripOnBackend) {
  constexpr uint32_t Width = 67;
  constexpr uint32_t Height = 5;
  const wgpu::TextureDescriptor descriptor{
      .usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst,
      .size = {Width, Height, 1},
      .format = wgpu::TextureFormat::RGBA8Uint,
  };
  const wgpu::Texture texture = m_device->device.CreateTexture(&descriptor);
  std::vector<uint8_t> pixels(Width * Height * 4);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>((i * 7) % 251);
  }
  const wgpu::TexelCopyTextureInfo destination{.texture = texture};
  const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = Width * 4, .rowsPerImage = Height};
  const wgpu::Extent3D extent{Width, Height, 1};
  m_device->queue.WriteTexture(&destination, pixels.data(), pixels.size(), &layout, &extent);
  const std::optional<debug::capture::Capture> capture = debug::capture::testing::read_texture(texture, 3);
  ASSERT_TRUE(capture.has_value());
  EXPECT_EQ(capture->width, Width);
  EXPECT_EQ(capture->height, Height);
  EXPECT_EQ(capture->bytes, pixels);

  const std::filesystem::path dir = scratch_dir("roundtrip");
  debug::capture::Capture named = *capture;
  named.resource = "gbuf.fog";
  named.anchor = "phase.composite";
  ASSERT_TRUE(debug::capture::write_capture(dir, "roundtrip", named));
  std::ifstream bin{dir / "roundtrip.bin", std::ios::binary};
  const std::vector<uint8_t> written{std::istreambuf_iterator<char>{bin}, std::istreambuf_iterator<char>{}};
  EXPECT_EQ(written, pixels);
  bin.close();
  std::filesystem::remove_all(dir);
}

constexpr uint64_t ViewPipelineWaitNs = 60'000'000'000;

class ViewShaderGpu : public RenderDebugCaptureDevice {
protected:
  void SetUp() override {
    RenderDebugCaptureDevice::SetUp();
    if (IsSkipped() || !m_device) {
      return;
    }
    jobs::initialize(1);
    shaders::initialize();
    m_shaders = true;
  }

  void TearDown() override {
    if (m_shaders) {
      shaders::shutdown();
      jobs::shutdown();
    }
    RenderDebugCaptureDevice::TearDown();
  }

  wgpu::Texture texture(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t bpp,
                        std::span<const uint8_t> bytes) const {
    const wgpu::TextureDescriptor descriptor{
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .size = {width, height, 1},
        .format = format,
    };
    const wgpu::Texture out = m_device->device.CreateTexture(&descriptor);
    const wgpu::TexelCopyTextureInfo destination{.texture = out};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = width * bpp, .rowsPerImage = height};
    const wgpu::Extent3D extent{width, height, 1};
    m_device->queue.WriteTexture(&destination, bytes.data(), bytes.size(), &layout, &extent);
    return out;
  }

  std::optional<debug::capture::Capture> draw(shaders::RenderPipelineDesc desc, std::span<const wgpu::Texture> inputs,
                                              uint32_t width, uint32_t height, uint64_t key) const {
    const wgpu::BindGroupLayoutEntry frameEntry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
        .buffer = {.type = wgpu::BufferBindingType::Uniform},
    };
    const wgpu::BindGroupLayoutDescriptor frameDesc{.entryCount = 1, .entries = &frameEntry};
    const wgpu::BindGroupLayout frameLayout = m_device->device.CreateBindGroupLayout(&frameDesc);
    std::vector<wgpu::BindGroupLayoutEntry> passEntries;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
      const bool uint = inputs[i].GetFormat() == wgpu::TextureFormat::RGBA8Uint;
      passEntries.push_back({
          .binding = i,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = uint ? wgpu::TextureSampleType::Uint : wgpu::TextureSampleType::UnfilterableFloat},
      });
    }
    const wgpu::BindGroupLayoutDescriptor passDesc{.entryCount = passEntries.size(), .entries = passEntries.data()};
    const wgpu::BindGroupLayout passLayout = m_device->device.CreateBindGroupLayout(&passDesc);
    const std::array layouts{frameLayout, passLayout};
    const wgpu::PipelineLayoutDescriptor layoutDesc{.bindGroupLayoutCount = layouts.size(),
                                                    .bindGroupLayouts = layouts.data()};
    desc.layout = m_device->device.CreatePipelineLayout(&layoutDesc);
    desc.layoutKey = key;
    const shaders::PipelineId id = shaders::request_render(desc);
    EXPECT_EQ(shaders::wait(id, ViewPipelineWaitNs), shaders::PipelineStatus::Ready);
    const wgpu::RenderPipeline pipeline = shaders::render_pipeline(id);
    if (!pipeline) {
      return std::nullopt;
    }
    const wgpu::BufferDescriptor bufferDesc{.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                            .size = 256};
    const wgpu::Buffer frame = m_device->device.CreateBuffer(&bufferDesc);
    const wgpu::BindGroupEntry frameBinding{.binding = 0, .buffer = frame, .size = 256};
    const wgpu::BindGroupDescriptor frameGroupDesc{.layout = frameLayout, .entryCount = 1, .entries = &frameBinding};
    const wgpu::BindGroup frameGroup = m_device->device.CreateBindGroup(&frameGroupDesc);
    std::vector<wgpu::BindGroupEntry> passBindings;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
      passBindings.push_back({.binding = i, .textureView = inputs[i].CreateView()});
    }
    const wgpu::BindGroupDescriptor passGroupDesc{
        .layout = passLayout, .entryCount = passBindings.size(), .entries = passBindings.data()};
    const wgpu::BindGroup passGroup = m_device->device.CreateBindGroup(&passGroupDesc);
    const wgpu::TextureDescriptor targetDesc{
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {width, height, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    const wgpu::Texture target = m_device->device.CreateTexture(&targetDesc);
    const wgpu::RenderPassColorAttachment attachment{
        .view = target.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.0, 0.0, 0.0, 0.0},
    };
    const wgpu::RenderPassDescriptor passDescriptor{.colorAttachmentCount = 1, .colorAttachments = &attachment};
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, frameGroup);
    pass.SetBindGroup(1, passGroup);
    pass.Draw(3);
    pass.End();
    EXPECT_TRUE(test::submit_and_wait(*m_device, encoder.Finish()));
    return debug::capture::testing::read_texture(target, 1);
  }

  bool m_shaders = false;
};

using DecompViewGpu = ViewShaderGpu;
using GbufViewGpu = ViewShaderGpu;

struct DecompCase {
  debug::decomp_view::Texel legacy;
  debug::decomp_view::Texel out;
  debug::decomp_view::Texel rec;
  uint32_t code = 0;
  uint8_t reactive = 0;
};

std::vector<uint8_t> texel_bytes(std::span<const DecompCase> cases, debug::decomp_view::Texel DecompCase::* field) {
  std::vector<uint8_t> bytes;
  for (const DecompCase& c : cases) {
    const debug::decomp_view::Texel& t = c.*field;
    bytes.insert(bytes.end(), t.begin(), t.end());
  }
  return bytes;
}

std::vector<uint8_t> normal_bytes(std::span<const DecompCase> cases) {
  std::vector<uint8_t> bytes;
  for (const DecompCase& c : cases) {
    const uint32_t packed = 512u | 512u << 10 | 1023u << 20 | c.code << 30;
    bytes.insert(bytes.end(), {static_cast<uint8_t>(packed), static_cast<uint8_t>(packed >> 8),
                               static_cast<uint8_t>(packed >> 16), static_cast<uint8_t>(packed >> 24)});
  }
  return bytes;
}

debug::Rgb8 pixel(const debug::capture::Capture& capture, size_t index) {
  return {capture.bytes[index * 4], capture.bytes[index * 4 + 1], capture.bytes[index * 4 + 2]};
}

TEST_F(DecompViewGpu, ClassifiesSyntheticTexels) {
  constexpr debug::decomp_view::Texel L{100, 100, 100, 7};
  constexpr debug::decomp_view::Texel NoFog{};
  constexpr debug::decomp_view::Texel Lifting{101, 50, 50, 127 << 1};
  constexpr debug::decomp_view::Texel Bad{110, 0, 0, 127 << 1};
  constexpr debug::decomp_view::Texel Fogged{30, 30, 30, 10 << 1};
  constexpr debug::decomp_view::Texel Locked{30, 30, 30, (10 << 1) | 1};
  const std::array<DecompCase, 8> cases{{
      {L, L, NoFog, 3, 0},
      {L, {99, 100, 100, 7}, NoFog, 3, 0},
      {L, {101, 100, 100, 7}, Lifting, 2, 0},
      {L, {101, 100, 100, 7}, NoFog, 2, 0},
      {L, {99, 100, 100, 7}, NoFog, 0, 0},
      {L, {100, 100, 100, 8}, Fogged, 1, 200},
      {L, {70, 100, 100, 7}, Bad, 2, 1},
      {L, {100, 100, 100, 7}, Locked, 3, 0},
  }};
  constexpr uint32_t Width = 8;
  const wgpu::Texture legacy =
      texture(wgpu::TextureFormat::RGBA8Unorm, Width, 1, 4, texel_bytes(cases, &DecompCase::legacy));
  const wgpu::Texture out = texture(wgpu::TextureFormat::RGBA8Unorm, Width, 1, 4, texel_bytes(cases, &DecompCase::out));
  const wgpu::Texture fog = texture(wgpu::TextureFormat::RGBA8Uint, Width, 1, 4, texel_bytes(cases, &DecompCase::rec));
  const wgpu::Texture normal = texture(wgpu::TextureFormat::RGB10A2Unorm, Width, 1, 4, normal_bytes(cases));
  std::vector<uint8_t> reactiveBytes;
  for (const DecompCase& c : cases) {
    reactiveBytes.push_back(c.reactive);
  }
  const wgpu::Texture reactive = texture(wgpu::TextureFormat::R8Unorm, Width, 1, 1, reactiveBytes);

  const std::array identityInputs{out, fog, normal, legacy};
  const auto identity =
      draw(debug::decomp_view::pipeline_desc(debug::DecompMode::Identity), identityInputs, Width, 1, 0xD1);
  ASSERT_TRUE(identity.has_value());
  const std::array<debug::Rgb8, 8> identityExpected{{
      {0, 0, 0},
      {71, 71, 71},
      {255, 128, 0},
      {0, 255, 255},
      {255, 0, 0},
      {255, 255, 0},
      {0, 0, 0},
      {0, 0, 0},
  }};
  for (size_t i = 0; i < cases.size(); ++i) {
    const DecompCase& c = cases[i];
    const debug::Rgb8 mirror = debug::decomp_view::identity_color(c.legacy, c.out, c.rec, c.code);
    EXPECT_EQ(pixel(*identity, i), mirror) << i;
    if (i < 6) {
      EXPECT_EQ(pixel(*identity, i), identityExpected[i]) << i;
    }
    EXPECT_EQ(identity->bytes[i * 4 + 3], 255) << i;
  }
  const debug::Rgb8 darkened = pixel(*identity, 6);
  EXPECT_EQ(darkened, (debug::Rgb8{debug::identity_grey(30), debug::identity_grey(30), debug::identity_grey(30)}));

  const std::array baseInputs{legacy, fog, normal};
  const auto defog = draw(debug::decomp_view::pipeline_desc(debug::DecompMode::Defog), baseInputs, Width, 1, 0xD0);
  ASSERT_TRUE(defog.has_value());
  EXPECT_EQ(pixel(*defog, 0), (debug::Rgb8{100, 100, 100}));
  EXPECT_EQ(pixel(*defog, 6), debug::DefogBadColor);
  for (size_t i = 0; i < cases.size(); ++i) {
    const debug::Rgb8 mirror = debug::decomp_view::defog_color(cases[i].legacy, cases[i].rec);
    const debug::Rgb8 got = pixel(*defog, i);
    EXPECT_LE(std::abs(int{got.r} - int{mirror.r}), 1) << i;
    EXPECT_LE(std::abs(int{got.g} - int{mirror.g}), 1) << i;
    EXPECT_LE(std::abs(int{got.b} - int{mirror.b}), 1) << i;
  }

  const std::array coverageInputs{legacy, fog, normal, reactive};
  const auto coverage =
      draw(debug::decomp_view::pipeline_desc(debug::DecompMode::Coverage), coverageInputs, Width, 1, 0xD2);
  ASSERT_TRUE(coverage.has_value());
  EXPECT_EQ(pixel(*coverage, 5), (debug::Rgb8{158, 228, 158}));
  EXPECT_EQ(pixel(*coverage, 6), (debug::Rgb8{255, 0, 255}));
  EXPECT_EQ(pixel(*coverage, 7), (debug::Rgb8{255, 160, 0}));
  EXPECT_EQ(pixel(*coverage, 4), (debug::Rgb8{64, 64, 64}));
  EXPECT_EQ(pixel(*coverage, 0), (debug::Rgb8{80, 160, 255}));
  for (size_t i = 0; i < cases.size(); ++i) {
    const DecompCase& c = cases[i];
    EXPECT_EQ(pixel(*coverage, i), debug::decomp_view::coverage_color(c.rec, c.code, c.legacy, c.reactive)) << i;
  }

  constexpr uint32_t MulWidth = 4;
  constexpr uint32_t MulHeight = 2;
  const std::array<DecompCase, 8> mulCases{{
      {{200, 100, 40, 0}, {}, NoFog, 1, 0},
      {{200, 100, 40, 0}, {}, NoFog, 2, 0},
      {{200, 100, 40, 0}, {}, NoFog, 3, 0},
      {{200, 100, 40, 0}, {}, NoFog, 1, 0},
      {{200, 100, 40, 0}, {}, NoFog, 0, 0},
      {{180, 90, 30, 0}, {}, Fogged, 2, 0},
      {{180, 90, 30, 0}, {}, Fogged, 0, 0},
      {{180, 90, 30, 0}, {}, Fogged, 1, 0},
  }};
  const wgpu::Texture mulLegacy =
      texture(wgpu::TextureFormat::RGBA8Unorm, MulWidth, MulHeight, 4, texel_bytes(mulCases, &DecompCase::legacy));
  const wgpu::Texture mulFog =
      texture(wgpu::TextureFormat::RGBA8Uint, MulWidth, MulHeight, 4, texel_bytes(mulCases, &DecompCase::rec));
  const wgpu::Texture mulNormal =
      texture(wgpu::TextureFormat::RGB10A2Unorm, MulWidth, MulHeight, 4, normal_bytes(mulCases));
  const std::array<uint8_t, 8> aoBytes{64, 0, 0, 0, 255, 0, 0, 0};
  const wgpu::Texture ao = texture(wgpu::TextureFormat::RGBA8Unorm, 2, 1, 4, aoBytes);
  const std::array multiplyInputs{mulLegacy, mulFog, mulNormal, ao};
  const auto multiply =
      draw(debug::decomp_view::pipeline_desc(debug::DecompMode::Multiply), multiplyInputs, MulWidth, MulHeight, 0xD3);
  ASSERT_TRUE(multiply.has_value());
  EXPECT_EQ(pixel(*multiply, 0), (debug::Rgb8{50, 25, 10}));
  EXPECT_EQ(pixel(*multiply, 2), (debug::Rgb8{200, 100, 40}));
  EXPECT_EQ(pixel(*multiply, 4), (debug::Rgb8{200, 100, 40}));
  for (size_t i = 0; i < mulCases.size(); ++i) {
    const uint8_t a = aoBytes[(i % MulWidth) / 2 * 4];
    const debug::Rgb8 mirror = debug::decomp_view::multiply_color(mulCases[i].legacy, mulCases[i].code, a);
    const debug::Rgb8 got = pixel(*multiply, i);
    EXPECT_LE(std::abs(int{got.r} - int{mirror.r}), 1) << i;
    EXPECT_LE(std::abs(int{got.g} - int{mirror.g}), 1) << i;
    EXPECT_LE(std::abs(int{got.b} - int{mirror.b}), 1) << i;
  }
}

TEST_F(GbufViewGpu, AoRowsDecode) {
  const auto grey = [](uint8_t v) { return debug::Rgb8{v, v, v}; };
  const std::array<uint8_t, 8> finalBytes{40, 0, 0, 0, 220, 0, 0, 0};
  const std::array finalInputs{texture(wgpu::TextureFormat::RGBA8Unorm, 2, 1, 4, finalBytes)};
  const auto final =
      draw(debug::gbuf_view::snapshot_pipeline_desc(*debug::find_gbuf_view(23)), finalInputs, 4, 2, 0xE7);
  ASSERT_TRUE(final.has_value());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(pixel(*final, i), grey(i % 4 < 2 ? 40 : 220)) << i;
  }

  std::vector<uint8_t> rawBytes;
  for (const uint8_t r : {uint8_t{0}, uint8_t{100}, uint8_t{170}, uint8_t{255}}) {
    rawBytes.insert(rawBytes.end(), {r, 77, 1, 0});
  }
  const std::array rawInputs{texture(wgpu::TextureFormat::RGBA8Unorm, 4, 1, 4, rawBytes)};
  const auto raw = draw(debug::gbuf_view::snapshot_pipeline_desc(*debug::find_gbuf_view(28)), rawInputs, 4, 1, 0xE8);
  ASSERT_TRUE(raw.has_value());
  EXPECT_EQ(pixel(*raw, 0), grey(0));
  EXPECT_EQ(pixel(*raw, 1), grey(150));
  EXPECT_EQ(pixel(*raw, 2), grey(255));
  EXPECT_EQ(pixel(*raw, 3), grey(255));

  const std::array<float, 8> depths{0.f,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    1000.f,
                                    1e30f,
                                    -std::numeric_limits<float>::infinity(),
                                    1.f,
                                    65535.f};
  std::vector<uint8_t> depthBytes(depths.size() * 4);
  std::memcpy(depthBytes.data(), depths.data(), depthBytes.size());
  const std::array hizInputs{texture(wgpu::TextureFormat::R32Float, 4, 2, 4, depthBytes)};
  const auto hiz = draw(debug::gbuf_view::snapshot_pipeline_desc(*debug::find_gbuf_view(29)), hizInputs, 4, 2, 0xE9);
  ASSERT_TRUE(hiz.has_value());
  constexpr debug::Rgb8 Magenta{255, 0, 255};
  EXPECT_EQ(pixel(*hiz, 0), grey(0));
  EXPECT_EQ(pixel(*hiz, 1), Magenta);
  EXPECT_EQ(pixel(*hiz, 2), Magenta);
  EXPECT_EQ(pixel(*hiz, 4), grey(255));
  EXPECT_EQ(pixel(*hiz, 5), Magenta);
  for (const size_t i : {size_t{3}, size_t{6}, size_t{7}}) {
    const auto expected =
        static_cast<int>(std::lround(std::clamp(std::log2(1.0 + depths[i]) / 18.0, 0.0, 1.0) * 255.0));
    const debug::Rgb8 got = pixel(*hiz, i);
    EXPECT_LE(std::abs(int{got.r} - expected), 1) << i;
    EXPECT_EQ(got.r, got.g) << i;
    EXPECT_EQ(got.r, got.b) << i;
  }
}

TEST(RenderDebugDualSource, JsonAggregatesByLayout) {
  debug::dual_source::Report report{.dualSourceBlending = true, .backend = "D3D12"};
  debug::dual_source::accumulate(report, {.dstAlphaDraws = 10, .dualSourcePipelines = 2}, 1);
  debug::dual_source::accumulate(report, {.dstAlphaDraws = 6, .prepassDraws = 3, .prepassPipelines = 1}, 5);
  debug::dual_source::accumulate(report, {.dstAlphaDraws = 4, .prepassDraws = 1}, 5);
  EXPECT_EQ(debug::dual_source::lost_dual_source_draws(report), 4u);
  const std::string text = debug::dual_source::format_json(report);
  const json::Doc doc = json::read(text);
  ASSERT_NE(doc, nullptr) << text;
  yyjson_val* root = yyjson_doc_get_root(doc.get());
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(root, "schema")), "aurora.render.dual_source");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "version")), 1u);
  yyjson_val* device = yyjson_obj_get(root, "device");
  EXPECT_TRUE(yyjson_get_bool(yyjson_obj_get(device, "dualSourceBlending")));
  EXPECT_STREQ(yyjson_get_str(yyjson_obj_get(device, "backend")), "D3D12");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "frames")), 3u);
  yyjson_val* pipelines = yyjson_obj_get(root, "pipelines");
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(pipelines, "dualSource")), 2u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(pipelines, "alphaPrepass")), 1u);
  yyjson_val* layouts = yyjson_obj_get(root, "layouts");
  ASSERT_EQ(yyjson_arr_size(layouts), 2u);
  yyjson_val* single = yyjson_arr_get(layouts, 0);
  yyjson_val* mrt = yyjson_arr_get(layouts, 1);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(single, "colorAttachments")), 1u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(single, "frames")), 1u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(single, "prepassDraws")), 0u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(mrt, "colorAttachments")), 5u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(mrt, "frames")), 2u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(mrt, "dstAlphaDraws")), 10u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(mrt, "prepassDraws")), 4u);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(root, "lostDualSourceDraws")), 4u);
  EXPECT_NE(yyjson_get_len(yyjson_obj_get(root, "note")), 0u);

  report.dualSourceBlending = false;
  EXPECT_EQ(debug::dual_source::lost_dual_source_draws(report), 0u);
  const json::Doc unsupported = json::read(debug::dual_source::format_json(report));
  ASSERT_NE(unsupported, nullptr);
  EXPECT_EQ(yyjson_get_uint(yyjson_obj_get(yyjson_doc_get_root(unsupported.get()), "lostDualSourceDraws")), 0u);
}
} // namespace
} // namespace aurora::render

#include "debug.hpp"

#include "../frame_state.hpp"
#include "../gpu_heap.hpp"
#include "../graph_pool.hpp"
#include "../shaders.hpp"
#include "../../webgpu/gpu_prof.hpp"

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace aurora::render::debug {
namespace {
namespace gpu_prof = webgpu::gpu_prof;

constexpr float AverageWeight = 0.1f;
constexpr std::string_view StageNotRun = "StageNotRun";
constexpr std::string_view PipelineNotReady = "PipelineNotReady";

// Game thread only, like the init and the ImGui frame that use them
std::vector<Tab> g_tabs;
std::vector<VramCounter> g_vramSources;
TimingAverages g_averages;
gpu_prof::FrameTimes g_windowTimes;

std::atomic_uint64_t g_latch{0};
std::atomic_uint64_t g_lastDrawnLatch{0};
std::atomic_bool g_everDrawn{false};

// Recording thread only
FrameCullSnapshot g_pendingCull;
bool g_pendingActive = false;

std::mutex g_cullMutex;
FrameCullSnapshot g_publishedCull;
bool g_cullPublished = false;

uint64_t gpu_heap_bytes() { return gpu_heap::total_bytes(); }

uint64_t transient_bytes() { return detail::transient_pool_bytes(); }

void update_averages() {
  if (!gpu_prof::latest(g_windowTimes)) {
    return;
  }
  const bool first = g_averages.sequence == 0;
  g_averages.sequence = g_windowTimes.sequence;
  g_averages.frameMs =
      first ? g_windowTimes.frameMs : g_averages.frameMs + AverageWeight * (g_windowTimes.frameMs - g_averages.frameMs);
  g_averages.dropped = g_windowTimes.dropped;
  for (AveragedPass& pass : g_averages.passes) {
    pass.inLatest = false;
  }
  for (const gpu_prof::PassTime& time : g_windowTimes.passes) {
    const auto it = std::ranges::find(g_averages.passes, time.name, &AveragedPass::name);
    if (it == g_averages.passes.end()) {
      g_averages.passes.push_back({.name = time.name, .ms = time.ms, .inLatest = true});
    } else {
      it->ms += AverageWeight * (time.ms - it->ms);
      it->inLatest = true;
    }
  }
}
} // namespace

void initialize() {
  register_tab("Overview", OverviewTabOrder, &draw_overview_tab);
  register_tab("Graph", GraphTabOrder, &draw_graph_tab);
  register_vram_source("GPU heap", &gpu_heap_bytes);
  register_vram_source("Transient pool", &transient_bytes);
}

void shutdown() {
  g_tabs.clear();
  g_vramSources.clear();
  g_averages = {};
  g_windowTimes = {};
  reset_graph_tab();
  testing::reset();
}

void register_tab(std::string_view name, int order, TabDrawFn draw) {
  std::erase_if(g_tabs, [&](const Tab& tab) { return tab.name == name; });
  g_tabs.push_back({.name = std::string{name}, .order = order, .draw = draw});
  std::ranges::sort(g_tabs, {}, [](const Tab& tab) { return std::tie(tab.order, tab.name); });
}

std::span<const Tab> tabs() noexcept { return g_tabs; }

void register_vram_source(std::string_view name, VramBytesFn bytes) {
  const auto it = std::ranges::find(g_vramSources, name, &VramCounter::name);
  if (it != g_vramSources.end()) {
    it->bytes = bytes;
    return;
  }
  g_vramSources.push_back({.name = std::string{name}, .bytes = bytes});
}

std::span<const VramCounter> vram_sources() noexcept { return g_vramSources; }

void draw_window(bool* open) {
  if (open == nullptr || !*open) {
    return;
  }
  g_lastDrawnLatch.store(g_latch.load(std::memory_order_relaxed), std::memory_order_relaxed);
  g_everDrawn.store(true, std::memory_order_release);
  gpu_prof::request(gpu_prof::PassTimingKeepAliveFrames);
  update_averages();

  ImGui::SetNextWindowSize({720.f, 560.f}, ImGuiCond_FirstUseEver);
  if (ImGui::Begin(RendererWindowId, open)) {
    if (g_tabs.empty()) {
      ImGui::TextUnformatted("Renderer module not initialized");
    } else if (ImGui::BeginTabBar("##renderer_tabs")) {
      for (const Tab& tab : g_tabs) {
        ImGui::PushID(tab.name.c_str());
        if (ImGui::BeginTabItem(tab.name.c_str())) {
          if (tab.draw != nullptr) {
            tab.draw();
          }
          ImGui::EndTabItem();
        }
        ImGui::PopID();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
}

std::span<const PassTiming> pass_timings() noexcept {
  // One caller thread, so the span stays valid until the next call
  static std::vector<PassTiming> timings;
  static gpu_prof::FrameTimes latest;
  gpu_prof::request(gpu_prof::PassTimingKeepAliveFrames);
  if (gpu_prof::latest(latest)) {
    timings.clear();
    for (const gpu_prof::PassTime& pass : latest.passes) {
      timings.push_back({.name = pass.name, .gpuMs = pass.ms});
    }
  }
  return timings;
}

std::string graph_dump_json(bool includeFrameState) {
  const std::shared_ptr<const detail::FrameState> frame = detail::current_frame_state();
  if (frame == nullptr || frame->graph == nullptr) {
    return std::string{NotCompiledGraphJson};
  }
  GraphInspection inspection = detail::inspect(*frame->graph);
  const detail::PoolStats pool = detail::transient_pool().stats();
  inspection.claimedBytes = pool.claimedBytes;
  inspection.idleBytes = pool.idleBytes;
  if (!includeFrameState) {
    return format_graph_json(inspection);
  }
  // Keeps timings collecting, so the next dump has a fresh frame
  gpu_prof::request(gpu_prof::PassTimingKeepAliveFrames);
  FrameCullSnapshot cull;
  const bool paired = latest_frame_cull(cull) && cull.graphHash == inspection.graphHash;
  gpu_prof::FrameTimes timings;
  const bool timed = gpu_prof::latest(timings);
  return format_graph_json(inspection, paired ? &cull : nullptr, timed ? &timings : nullptr);
}

const TimingAverages& timing_averages() noexcept { return g_averages; }

void on_latch() noexcept { g_latch.fetch_add(1, std::memory_order_relaxed); }

bool inspector_open() noexcept {
  return g_everDrawn.load(std::memory_order_acquire) &&
         g_latch.load(std::memory_order_relaxed) - g_lastDrawnLatch.load(std::memory_order_relaxed) <= 2;
}

void record_stage_cull(const detail::CompiledGraph& graph, Stage stage, uint64_t frameId,
                       std::span<const detail::FrameCullReason> reasons) {
  if (!g_pendingActive || g_pendingCull.frameId != frameId) {
    g_pendingCull.frameId = frameId;
    g_pendingCull.graphHash = graph.graphHash;
    g_pendingCull.reasons.assign(graph.framePasses.size(), std::string{StageNotRun});
    g_pendingActive = true;
  }
  const detail::StageProgram& program = graph.stages[static_cast<size_t>(stage)];
  const size_t count = std::min(reasons.size(), program.passes.size());
  for (size_t i = 0; i < count; ++i) {
    const detail::CompiledNode& node = graph.nodes[program.passes[i]];
    if (node.frameSlot >= g_pendingCull.reasons.size()) {
      continue;
    }
    std::optional<std::string>& entry = g_pendingCull.reasons[node.frameSlot];
    if (reasons[i] == detail::FrameCullReason::Ran) {
      entry.reset();
    } else if (reasons[i] == detail::FrameCullReason::NotEnabled && !shaders::pipelines_ready(node.pipelineIds)) {
      entry = std::string{PipelineNotReady};
    } else {
      entry = std::string{detail::frame_cull_reason_name(reasons[i])};
    }
  }
}

void publish_frame_cull(uint64_t frameId, uint64_t graphHash) {
  if (!g_pendingActive || g_pendingCull.frameId != frameId) {
    return;
  }
  g_pendingCull.graphHash = graphHash;
  g_pendingActive = false;
  std::lock_guard lock{g_cullMutex};
  std::swap(g_publishedCull, g_pendingCull);
  g_cullPublished = true;
}

bool latest_frame_cull(FrameCullSnapshot& out) {
  std::lock_guard lock{g_cullMutex};
  if (!g_cullPublished) {
    return false;
  }
  out = g_publishedCull;
  return true;
}

namespace testing {
void reset() noexcept {
  g_latch.store(0, std::memory_order_relaxed);
  g_lastDrawnLatch.store(0, std::memory_order_relaxed);
  g_everDrawn.store(false, std::memory_order_relaxed);
  g_pendingCull = {};
  g_pendingActive = false;
  std::lock_guard lock{g_cullMutex};
  g_publishedCull = {};
  g_cullPublished = false;
}
} // namespace testing
} // namespace aurora::render::debug

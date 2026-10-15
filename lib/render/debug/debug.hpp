#pragma once

#include "graph_inspection.hpp"

#include "../graph.hpp"

#include <aurora/render.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug {
using TabDrawFn = void (*)();
using VramBytesFn = uint64_t (*)();

inline constexpr int OverviewTabOrder = 0;
inline constexpr int GraphTabOrder = 10;
inline constexpr int ViewsTabOrder = 20;
inline constexpr int SceneTabOrder = 30;
inline constexpr int LightsTabOrder = 40;
inline constexpr int TexturesTabOrder = 50;
inline constexpr int ShadowsTabOrder = 60;
inline constexpr int AtmosphereTabOrder = 70;
inline constexpr int GrassTerrainTabOrder = 80;
inline constexpr int RtGiTabOrder = 90;
inline constexpr int SystemsTabOrder = 100;
inline constexpr std::string_view NotCompiledGraphJson =
    "{\n  \"schema\": \"aurora.render.graph\",\n  \"version\": 1,\n  \"compiled\": false\n}\n";
inline constexpr const char* RendererWindowId = "Renderer###aurora_renderer";

struct Tab {
  std::string name;
  int order = 0;
  TabDrawFn draw = nullptr;
};

struct VramCounter {
  std::string name;
  VramBytesFn bytes = nullptr;
};

struct AveragedPass {
  const char* name = nullptr;
  float ms = 0.f;
  bool inLatest = false;
};

struct TimingAverages {
  uint64_t sequence = 0;
  float frameMs = 0.f;
  uint32_t dropped = 0;
  std::vector<AveragedPass> passes;
};

void initialize();
void shutdown();
void register_tab(std::string_view name, int order, TabDrawFn draw);
std::span<const Tab> tabs() noexcept;
void register_vram_source(std::string_view name, VramBytesFn bytes);
std::span<const VramCounter> vram_sources() noexcept;

void draw_window(bool* open);
std::span<const PassTiming> pass_timings() noexcept;
std::string graph_dump_json(bool includeFrameState);
const TimingAverages& timing_averages() noexcept;

void on_latch() noexcept;
bool inspector_open() noexcept;

void record_stage_cull(const detail::CompiledGraph& graph, Stage stage, uint64_t frameId,
                       std::span<const detail::FrameCullReason> reasons);
void publish_frame_cull(uint64_t frameId, uint64_t graphHash);
bool latest_frame_cull(FrameCullSnapshot& out);

void draw_overview_tab();
void draw_graph_tab();
void reset_graph_tab();

namespace testing {
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::debug

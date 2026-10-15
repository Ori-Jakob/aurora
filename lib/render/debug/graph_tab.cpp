#include "debug.hpp"

#include "../frame_state.hpp"
#include "../graph_pool.hpp"
#include "../../internal.hpp"
#include "../../io.hpp"

#include <SDL3/SDL_error.h>
#include <imgui.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace aurora::render::debug {
namespace {
constexpr double MiB = 1024.0 * 1024.0;
constexpr float CulledAlpha = 0.5f;
constexpr float TimelineRows = 14.f;
constexpr float DetailRows = 12.f;
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
constexpr std::string_view NoReason = "-";

struct TabState {
  std::weak_ptr<const detail::CompiledGraph> graph;
  bool inspected = false;
  GraphInspection inspection;
  std::unordered_set<std::string> passNames;
  uint64_t averagesSequence = 0;
  std::unordered_map<std::string_view, float> averageMs;
  int stageFilter = 0;
  bool showCulled = true;
  int selected = -1;
  std::string dumpStatus;
  bool dumpFailed = false;
};

// Game thread only, like the ImGui frame
TabState g_state;

void text(std::string_view value) {
  value = value.empty() ? std::string_view{""} : value;
  ImGui::TextUnformatted(value.data(), value.data() + value.size());
}

bool same_owner(const std::weak_ptr<const detail::CompiledGraph>& cached,
                const std::shared_ptr<const detail::CompiledGraph>& graph) noexcept {
  return !cached.owner_before(graph) && !graph.owner_before(cached);
}

void refresh(const std::shared_ptr<const detail::CompiledGraph>& graph) {
  // Ownership, not the hash: the hash ignores culled passes, and a freed graph's address can come back
  if (!g_state.inspected || !same_owner(g_state.graph, graph)) {
    g_state.graph = graph;
    g_state.inspected = true;
    g_state.inspection = detail::inspect(*graph);
    g_state.passNames.clear();
    for (const InspectPass& pass : g_state.inspection.passes) {
      if (pass.kind) {
        g_state.passNames.insert(pass.name);
      }
    }
    if (g_state.selected >= static_cast<int>(g_state.inspection.passes.size())) {
      g_state.selected = -1;
    }
  }
  const TimingAverages& averages = timing_averages();
  if (g_state.averagesSequence != averages.sequence) {
    g_state.averagesSequence = averages.sequence;
    g_state.averageMs.clear();
    for (const AveragedPass& pass : averages.passes) {
      if (pass.inLatest) {
        g_state.averageMs.emplace(pass.name, pass.ms);
      }
    }
  }
}

std::optional<float> pass_ms(const InspectPass& pass) {
  if (!pass.kind) {
    return std::nullopt;
  }
  const auto it = g_state.averageMs.find(pass.name);
  return it != g_state.averageMs.end() ? std::optional{it->second} : std::nullopt;
}

float gx_ms() {
  float total = 0.f;
  for (const auto& [name, ms] : g_state.averageMs) {
    if (!g_state.passNames.contains(std::string{name})) {
      total += ms;
    }
  }
  return total;
}

const std::optional<std::string>* last_frame_reason(const InspectPass& pass, const FrameCullSnapshot* cull) {
  if (cull == nullptr || !pass.frameSlot || *pass.frameSlot >= cull->reasons.size()) {
    return nullptr;
  }
  return &cull->reasons[*pass.frameSlot];
}

std::string_view reason_text(const InspectPass& pass, const FrameCullSnapshot* cull) {
  if (pass.cull) {
    return *pass.cull;
  }
  const auto* reason = last_frame_reason(pass, cull);
  return reason != nullptr && reason->has_value() ? std::string_view{**reason} : NoReason;
}

bool culled(const InspectPass& pass, const FrameCullSnapshot* cull) {
  const auto* reason = last_frame_reason(pass, cull);
  return !pass.active || (reason != nullptr && reason->has_value());
}

bool stage_visible(const InspectPass& pass) {
  if (g_state.stageFilter == 0) {
    return true;
  }
  return pass.stage == stage_name(static_cast<Stage>(g_state.stageFilter - 1));
}

void draw_bar(float first, float last) {
  const ImVec2 start = ImGui::GetCursorScreenPos();
  const float width = std::max(ImGui::GetContentRegionAvail().x, 1.f);
  const float height = ImGui::GetTextLineHeight();
  const float x0 = start.x + width * std::clamp(first, 0.f, 1.f);
  const float x1 = start.x + width * std::clamp(last, 0.f, 1.f);
  ImGui::GetWindowDrawList()->AddRectFilled({x0, start.y}, {std::max(x1, x0 + 1.f), start.y + height},
                                            ImGui::GetColorU32(ImGuiCol_PlotHistogram));
  ImGui::Dummy({width, height});
}

void dump_json() {
  const std::string json = graph_dump_json(true);
  const auto path =
      io::fs_path_from_string(g_config.cachePath != nullptr ? g_config.cachePath : ".") / "render_graph.json";
  if (json.empty()) {
    g_state.dumpFailed = true;
    g_state.dumpStatus = "Failed to format the graph dump";
    return;
  }
  const bool written = io::write_file_atomic(path, {reinterpret_cast<const uint8_t*>(json.data()), json.size()});
  g_state.dumpFailed = !written;
  g_state.dumpStatus = written ? "Wrote " + io::fs_path_to_string(path)
                               : "Failed to write " + io::fs_path_to_string(path) + ": " + SDL_GetError();
}

void draw_header(const GraphInspection& inspection, const FrameCullSnapshot* cull) {
  uint32_t total = 0;
  uint32_t active = 0;
  uint32_t lastFrame = 0;
  for (const InspectPass& pass : inspection.passes) {
    if (!pass.kind) {
      continue;
    }
    ++total;
    active += pass.active ? 1 : 0;
    const auto* reason = last_frame_reason(pass, cull);
    lastFrame += reason != nullptr && reason->has_value() ? 1 : 0;
  }
  ImGui::Text("Graph 0x%016llx", static_cast<unsigned long long>(inspection.graphHash));
  ImGui::Text("Passes: %u total, %u active, %u culled at compile", total, active, total - active);
  if (cull != nullptr) {
    ImGui::Text("Last frame: %u culled (frame %llu)", lastFrame, static_cast<unsigned long long>(cull->frameId));
  } else {
    ImGui::TextUnformatted("Last frame: n/a");
  }

  const std::string preview = g_state.stageFilter == 0
                                  ? std::string{"All"}
                                  : std::string{stage_name(static_cast<Stage>(g_state.stageFilter - 1))};
  ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.f);
  if (ImGui::BeginCombo("Stage", preview.c_str())) {
    for (int i = 0; i <= static_cast<int>(StageCount); ++i) {
      const std::string label = i == 0 ? std::string{"All"} : std::string{stage_name(static_cast<Stage>(i - 1))};
      if (ImGui::Selectable(label.c_str(), g_state.stageFilter == i)) {
        g_state.stageFilter = i;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Show culled", &g_state.showCulled);
  ImGui::SameLine();
  if (ImGui::Button("Dump JSON")) {
    dump_json();
  }
  if (!g_state.dumpStatus.empty()) {
    if (g_state.dumpFailed) {
      ImGui::TextColored(WarningColor, "%s", g_state.dumpStatus.c_str());
    } else {
      ImGui::TextUnformatted(g_state.dumpStatus.c_str());
    }
  }
}

void draw_timeline(const GraphInspection& inspection, const FrameCullSnapshot* cull) {
  const float gxTotal = gx_ms();
  float largest = gxTotal;
  for (const InspectPass& pass : inspection.passes) {
    largest = std::max(largest, pass_ms(pass).value_or(0.f));
  }
  constexpr ImGuiTableFlags Flags = TableFlags | ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##timeline", 5, Flags, {0.f, ImGui::GetTextLineHeightWithSpacing() * TimelineRows})) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 2.f);
  ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("GPU ms", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("##bar", ImGuiTableColumnFlags_WidthStretch, 1.f);
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < inspection.passes.size(); ++i) {
    const InspectPass& pass = inspection.passes[i];
    const bool dim = culled(pass, cull);
    if (!stage_visible(pass) || (dim && !g_state.showCulled)) {
      continue;
    }
    const std::optional<float> ms = pass_ms(pass);
    ImGui::TableNextRow();
    if (dim) {
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * CulledAlpha);
    }
    ImGui::TableNextColumn();
    text(pass.stage);
    ImGui::TableNextColumn();
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::Selectable(pass.name.c_str(), g_state.selected == static_cast<int>(i),
                          ImGuiSelectableFlags_SpanAllColumns)) {
      g_state.selected = static_cast<int>(i);
    }
    ImGui::PopID();
    if (dim && ImGui::IsItemHovered()) {
      const std::string_view reason = reason_text(pass, cull);
      ImGui::SetTooltip("%.*s", static_cast<int>(reason.size()), reason.data());
    }
    ImGui::TableNextColumn();
    text(pass.kind ? std::string_view{*pass.kind} : std::string_view{pass.nodeKind});
    ImGui::TableNextColumn();
    if (ms) {
      ImGui::Text("%.3f", *ms);
    }
    ImGui::TableNextColumn();
    if (ms && largest > 0.f) {
      draw_bar(0.f, *ms / largest);
    }
    if (dim) {
      ImGui::PopStyleVar();
    }
  }
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::TableNextColumn();
  ImGui::TextUnformatted("GX passes");
  ImGui::TableNextColumn();
  ImGui::TableNextColumn();
  ImGui::Text("%.3f", gxTotal);
  ImGui::TableNextColumn();
  if (largest > 0.f) {
    draw_bar(0.f, gxTotal / largest);
  }
  ImGui::EndTable();
}

void draw_cull_reasons(const GraphInspection& inspection, const FrameCullSnapshot* cull) {
  if (!ImGui::CollapsingHeader("Cull reasons", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  if (!ImGui::BeginTable("##cull_reasons", 5, TableFlags)) {
    return;
  }
  ImGui::TableSetupColumn("Pass");
  ImGui::TableSetupColumn("Stage");
  ImGui::TableSetupColumn("Structural");
  ImGui::TableSetupColumn("Last frame");
  ImGui::TableSetupColumn("Detail", ImGuiTableColumnFlags_WidthStretch, 3.f);
  ImGui::TableHeadersRow();
  for (const InspectPass& pass : inspection.passes) {
    const auto* lastFrame = last_frame_reason(pass, cull);
    const bool frameReason = lastFrame != nullptr && lastFrame->has_value();
    if (!pass.cull && !frameReason) {
      continue;
    }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    text(pass.name);
    ImGui::TableNextColumn();
    text(pass.stage);
    ImGui::TableNextColumn();
    text(pass.cull ? std::string_view{*pass.cull} : NoReason);
    ImGui::TableNextColumn();
    text(frameReason ? std::string_view{**lastFrame} : NoReason);
    ImGui::TableNextColumn();
    if (pass.detail) {
      ImGui::TextWrapped("%s", pass.detail->c_str());
    }
  }
  ImGui::EndTable();
}

void draw_resources(const GraphInspection& inspection) {
  if (!ImGui::CollapsingHeader("Resources", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  const auto passCount = static_cast<float>(std::max<size_t>(inspection.passes.size(), 1));
  if (ImGui::BeginTable("##resources", 7, TableFlags)) {
    ImGui::TableSetupColumn("Name");
    ImGui::TableSetupColumn("Kind");
    ImGui::TableSetupColumn("Format");
    ImGui::TableSetupColumn("Size");
    ImGui::TableSetupColumn("MiB");
    ImGui::TableSetupColumn("Slot");
    ImGui::TableSetupColumn("Lifetime", ImGuiTableColumnFlags_WidthStretch, 2.f);
    ImGui::TableHeadersRow();
    for (const InspectResource& resource : inspection.resources) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(resource.name);
      ImGui::TableNextColumn();
      text(resource.kind);
      ImGui::TableNextColumn();
      text(resource.type == "buffer" ? std::string_view{"buffer"} : std::string_view{resource.format});
      ImGui::TableNextColumn();
      text(resource.size);
      ImGui::TableNextColumn();
      ImGui::Text("%.2f", static_cast<double>(resource.bytes) / MiB);
      ImGui::TableNextColumn();
      if (resource.slot) {
        ImGui::Text("%u", *resource.slot);
      } else {
        text(NoReason);
      }
      ImGui::TableNextColumn();
      if (resource.firstPass && resource.lastPass) {
        draw_bar(static_cast<float>(*resource.firstPass) / passCount,
                 static_cast<float>(*resource.lastPass + 1) / passCount);
      }
    }
    ImGui::EndTable();
  }
  const detail::PoolStats pool = detail::transient_pool().stats();
  ImGui::Text("Logical %.2f MiB, pooled %.2f MiB in %u slots, claimed %.2f MiB, idle %.2f MiB",
              static_cast<double>(inspection.logicalBytes) / MiB, static_cast<double>(inspection.slotBytes) / MiB,
              inspection.poolSlots, static_cast<double>(pool.claimedBytes) / MiB,
              static_cast<double>(pool.idleBytes) / MiB);
}

void draw_detail(const GraphInspection& inspection, const FrameCullSnapshot* cull) {
  if (g_state.selected < 0 || g_state.selected >= static_cast<int>(inspection.passes.size())) {
    return;
  }
  const auto index = static_cast<uint32_t>(g_state.selected);
  const InspectPass& pass = inspection.passes[index];
  if (ImGui::BeginChild("##pass_detail", {0.f, ImGui::GetTextLineHeightWithSpacing() * DetailRows},
                        ImGuiChildFlags_Borders)) {
    ImGui::Text("%s (%s, %s)", pass.name.c_str(), pass.nodeKind.c_str(), pass.stage.c_str());
    if (pass.kind) {
      ImGui::Text("Kind %s, class %s, order %u, registration %u", pass.kind->c_str(),
                  pass.cls ? pass.cls->c_str() : "-", pass.order.value_or(0), pass.registration.value_or(0));
    }
    ImGui::Text("Active %s, side effect %s", pass.active ? "yes" : "no", pass.sideEffect ? "yes" : "no");
    if (pass.frameSlot) {
      const auto* reason = last_frame_reason(pass, cull);
      const std::string_view lastFrame =
          reason == nullptr ? std::string_view{"n/a"}
                            : (reason->has_value() ? std::string_view{**reason} : std::string_view{"ran"});
      ImGui::Text("Frame slot %u, last frame %.*s", *pass.frameSlot, static_cast<int>(lastFrame.size()),
                  lastFrame.data());
    }
    if (pass.cull) {
      ImGui::Text("Culled: %s", pass.cull->c_str());
    }
    if (pass.detail) {
      ImGui::TextWrapped("%s", pass.detail->c_str());
    }
    for (const InspectRead& read : pass.reads) {
      ImGui::BulletText("reads %s as %s (%s, %s)", read.resource.c_str(), read.access.c_str(), read.missing.c_str(),
                        read.binding.c_str());
    }
    for (const InspectWrite& write : pass.writes) {
      ImGui::BulletText("writes %s as %s", write.resource.c_str(), write.access.c_str());
    }
    for (const InspectEdge& edge : inspection.edges) {
      if (edge.to != index && edge.from != index) {
        continue;
      }
      const bool incoming = edge.to == index;
      const InspectPass& other = inspection.passes[incoming ? edge.from : edge.to];
      ImGui::BulletText("%s %s %s%s%s", incoming ? "after" : "before", other.name.c_str(), edge.kind.c_str(),
                        edge.resource ? " on " : "", edge.resource ? edge.resource->c_str() : "");
    }
  }
  ImGui::EndChild();
}
} // namespace

void draw_graph_tab() {
  const std::shared_ptr<const detail::FrameState> frame = detail::current_frame_state();
  if (frame == nullptr || frame->graph == nullptr) {
    ImGui::TextUnformatted("No compiled graph");
    return;
  }
  refresh(frame->graph);
  const GraphInspection& inspection = g_state.inspection;
  FrameCullSnapshot snapshot;
  const bool paired = latest_frame_cull(snapshot) && snapshot.graphHash == inspection.graphHash;
  const FrameCullSnapshot* cull = paired ? &snapshot : nullptr;
  draw_header(inspection, cull);
  draw_timeline(inspection, cull);
  draw_cull_reasons(inspection, cull);
  draw_resources(inspection);
  draw_detail(inspection, cull);
}

void reset_graph_tab() { g_state = {}; }
} // namespace aurora::render::debug

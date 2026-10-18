#include "views.hpp"

#include "debug.hpp"

#include "../scene/classify.hpp"
#include "../../gx/hints.hpp"

#include <aurora/gfx.h>
#include <aurora/render.hpp>
#include <imgui.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug {
namespace {
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
constexpr ImGuiTableFlags ListFlags = TableFlags | ImGuiTableFlags_ScrollY;
constexpr float ListRows = 12.f;
constexpr uint16_t FirstModuleHint = GX_AURORA_RENDER_CAMERA;
constexpr uint16_t LastModuleHint = GX_AURORA_RENDER_WIND_SOURCES;

struct TabState {
  int kindFilter = 0;
  int roleFilter = 0;
  bool mainOnly = false;
  std::vector<uint32_t> rows;
};

// Game thread only, like the ImGui frame
TabState g_state;

void text(std::string_view value) {
  value = value.empty() ? std::string_view{""} : value;
  ImGui::TextUnformatted(value.data(), value.data() + value.size());
}

void count_line(const char* label, uint64_t value, bool warn = false) {
  if (warn && value != 0) {
    ImGui::TextColored(WarningColor, "%s: %llu", label, static_cast<unsigned long long>(value));
  } else {
    ImGui::Text("%s: %llu", label, static_cast<unsigned long long>(value));
  }
}

template <typename Enum, size_t N, typename NameFn>
void count_table(const char* id, const char* header, const std::array<uint32_t, N>& counts, NameFn name) {
  if (!ImGui::BeginTable(id, 2, TableFlags)) {
    return;
  }
  ImGui::TableSetupColumn(header);
  ImGui::TableSetupColumn("Draws");
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < N; ++i) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    text(name(static_cast<Enum>(i)));
    ImGui::TableNextColumn();
    ImGui::Text("%u", counts[i]);
  }
  ImGui::EndTable();
}

std::string category_names(uint32_t bits) {
  std::string names;
  for (uint32_t rest = bits; rest != 0; rest &= rest - 1) {
    const std::string_view name = scene::category_name(static_cast<uint32_t>(std::countr_zero(rest)));
    if (!name.empty()) {
      names += names.empty() ? "" : " ";
      names += name;
    }
  }
  return names;
}

std::string role_names(uint8_t roles) {
  std::string names;
  for (size_t role = 0; role < scene::RoleCount; ++role) {
    if ((roles & (1u << role)) != 0) {
      names += names.empty() ? "" : " ";
      names += scene::role_name(static_cast<scene::Role>(role));
    }
  }
  return names;
}

void draw_header(const scene::FrameReport& report) {
  ImGui::Text("Frame %llu", static_cast<unsigned long long>(report.stats.frame));
  ImGui::Text("MAIN camera: %s, cut epoch %u", report.mainCamera ? "yes" : "no", report.cutEpoch);
  if (report.sceneEpochKnown) {
    ImGui::Text("Scene epoch: %u", report.sceneEpoch);
  } else {
    ImGui::TextUnformatted("Scene epoch: unknown");
  }
  std::string markers;
  for (size_t i = 0; i <= static_cast<size_t>(Stage::FrameEnd); ++i) {
    const int32_t draw = report.stats.markerDraw[i];
    if (draw >= 0) {
      markers += markers.empty() ? "" : ", ";
      markers += stage_name(static_cast<Stage>(i));
      markers += " @ " + std::to_string(draw);
    }
  }
  ImGui::Text("Markers: %s", markers.empty() ? "none" : markers.c_str());
}

void draw_view_controls() {
  ImGui::SeparatorText("View");
  const uint16_t current = view_override();
  const DebugView* selected = find_view(current);
  const std::string preview = selected != nullptr ? std::string{selected->label} : std::string{"Off"};
  if (ImGui::BeginCombo("Debug view", preview.c_str())) {
    if (ImGui::Selectable("Off", current == 0)) {
      set_view_override(0);
    }
    for (const DebugView& view : ClassViews) {
      const std::string label{view.label};
      if (ImGui::Selectable(label.c_str(), view.id == current)) {
        set_view_override(view.id);
      }
    }
    ImGui::EndCombo();
  }
  const uint32_t bit = view_bit();
  const std::string bitName{scene::category_name(static_cast<uint32_t>(std::countr_zero(bit)))};
  if (ImGui::BeginCombo("Bit", bitName.c_str())) {
    for (uint32_t index = 0; index < 32; ++index) {
      const std::string name{scene::category_name(index)};
      if (!name.empty() && ImGui::Selectable(name.c_str(), bit == 1u << index)) {
        set_view_bit(1u << index);
      }
    }
    ImGui::EndCombo();
  }
  ImGui::Text("graphics.debugView = %u%s", effective_settings().debugView, current != 0 ? " (override active)" : "");
}

void draw_counters(const scene::FrameStats& stats) {
  ImGui::SeparatorText("Classification");
  count_line("Draws", stats.draws);
  count_line("Suppressed", stats.suppressed);
  count_line("Merge breaks", stats.mergeBreaks);
  count_line("Capture candidates", stats.captureCandidates);
  count_line("FAKE_SHADOW", stats.fakeShadow);
  count_line("FAKE_SHADOW and OVERLAY", stats.fakeShadowOverlay, true);
  count_line("Unclassified", stats.unclassified);
  count_line("TEV-policy draws", stats.tevPolicyDraws);
  count_line("Sky misses", stats.skyMisses, true);
  count_table<scene::Kind>("##kinds", "Kind", stats.kinds, scene::kind_name);
  count_table<scene::Role>("##roles", "Role", stats.roles, scene::role_name);
  count_table<scene::MainCheck>("##checks", "Main-class test", stats.checks, scene::check_name);
  count_table<scene::Window>("##windows", "Window", stats.windows, scene::window_name);
  if (ImGui::BeginTable("##under_scope", 7, TableFlags)) {
    for (const char* column : {"Role scope", "Draws", "As role", "Main", "Role only", "Captures", "Sky kind"}) {
      ImGui::TableSetupColumn(column);
    }
    ImGui::TableHeadersRow();
    for (size_t role = 0; role < scene::RoleCount; ++role) {
      const scene::RoleScopeCounts& counts = stats.underScope[role];
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(scene::role_name(static_cast<scene::Role>(role)));
      for (const uint32_t value :
           {counts.draws, counts.asRole, counts.main, counts.roleOnly, counts.captures, counts.skyKind}) {
        ImGui::TableNextColumn();
        ImGui::Text("%u", value);
      }
    }
    ImGui::EndTable();
  }
  if (ImGui::BeginTable("##categories", 2, TableFlags)) {
    ImGui::TableSetupColumn("Category");
    ImGui::TableSetupColumn("Draws");
    ImGui::TableHeadersRow();
    for (uint32_t bit = 0; bit < stats.categories.size(); ++bit) {
      const std::string_view name = scene::category_name(bit);
      if (name.empty()) {
        continue;
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(name);
      ImGui::TableNextColumn();
      ImGui::Text("%u", stats.categories[bit]);
    }
    ImGui::EndTable();
  }
}

void draw_hint_health(const scene::HintCounts& hints) {
  ImGui::SeparatorText("Hint health");
  ImGui::Text("Cameras %u (MAIN %u), scopes %u/%u, objects %u/%u, markers %u", hints.cameras, hints.mainCameras,
              hints.scopePushes, hints.scopePops, hints.objectBegins, hints.objectEnds, hints.markers);
  count_line("Errors", hints.errors(), true);
  if (hints.errors() != 0) {
    ImGui::TextColored(WarningColor,
                       "stale markers %u, scope overflow %u, scope underflow %u, nested objects %u, object "
                       "underflow %u, open scopes %u, open objects %u",
                       hints.staleMarkers, hints.scopeOverflow, hints.scopeUnderflow, hints.nestedObjects,
                       hints.objectUnderflow, hints.openScopes, hints.openObjects);
  }
  if (ImGui::BeginTable("##hint_ids", 6, TableFlags)) {
    for (const char* column : {"Hint", "Decoded", "Skipped", "Malformed", "Late", "Bytes"}) {
      ImGui::TableSetupColumn(column);
    }
    ImGui::TableHeadersRow();
    for (uint16_t id = FirstModuleHint; id <= LastModuleHint; ++id) {
      const HintCounters counters = hint_counters(id);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      text(gx::hints::name(id));
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(counters.decoded));
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(counters.skipped));
      ImGui::TableNextColumn();
      if (counters.malformed != 0) {
        ImGui::TextColored(WarningColor, "%llu", static_cast<unsigned long long>(counters.malformed));
      } else {
        ImGui::TextUnformatted("0");
      }
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(counters.late));
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(counters.bytes));
    }
    ImGui::EndTable();
  }
  const DrawRouteCounters route = draw_route_counters();
  ImGui::Text("Routed %llu, suppressed %llu, merge breaks %llu, substituted %llu",
              static_cast<unsigned long long>(route.routed), static_cast<unsigned long long>(route.suppressed),
              static_cast<unsigned long long>(route.mergeBreaks), static_cast<unsigned long long>(route.substituted));
}

void draw_view_counts(const scene::ViewCounts& view) {
  ImGui::SeparatorText("View draws");
  ImGui::Text("Substituted %u, untouched %u, pending %u", view.substituted, view.untouched, view.pending);
  count_line("Over budget", view.overBudget, true);
  ImGui::Text("View uniform bytes %u, frame uniform bytes %u", view.uniformBytes, aurora_get_stats()->lastUniformSize);
}

void draw_objects(const scene::FrameReport& report) {
  ImGui::SeparatorText("Objects");
  if (report.objects.empty()) {
    ImGui::TextUnformatted("No objects this frame");
    return;
  }
  const float height = ImGui::GetTextLineHeightWithSpacing() * ListRows;
  if (!ImGui::BeginTable("##objects", 8, ListFlags, {0.f, height})) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  for (const char* column : {"Id", "Sub", "Gen", "Categories", "Flags", "Roles", "Draws", "Main"}) {
    ImGui::TableSetupColumn(column);
  }
  ImGui::TableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(report.objects.size()));
  while (clipper.Step()) {
    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
      const scene::ObjectStats& object = report.objects[static_cast<size_t>(row)];
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("0x%016llx", static_cast<unsigned long long>(object.ref.id));
      ImGui::TableNextColumn();
      ImGui::Text("%u", object.ref.subId);
      ImGui::TableNextColumn();
      ImGui::Text("%u", object.ref.generation);
      ImGui::TableNextColumn();
      text(category_names(object.categories));
      ImGui::TableNextColumn();
      ImGui::Text("0x%x", object.flags);
      ImGui::TableNextColumn();
      text(role_names(object.roles));
      ImGui::TableNextColumn();
      ImGui::Text("%u", object.draws);
      ImGui::TableNextColumn();
      ImGui::Text("%u", object.mainDraws);
    }
  }
  ImGui::EndTable();
}

bool filter_combo(const char* label, int& filter, size_t count, auto name) {
  bool changed = false;
  const std::string preview = filter == 0 ? std::string{"All"} : std::string{name(static_cast<size_t>(filter - 1))};
  if (ImGui::BeginCombo(label, preview.c_str())) {
    if (ImGui::Selectable("All", filter == 0)) {
      filter = 0;
      changed = true;
    }
    for (size_t i = 0; i < count; ++i) {
      const std::string item{name(i)};
      if (ImGui::Selectable(item.c_str(), filter == static_cast<int>(i) + 1)) {
        filter = static_cast<int>(i) + 1;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

void draw_records(const scene::FrameReport& report) {
  ImGui::SeparatorText("Draws");
  if (report.stats.droppedRecords != 0) {
    ImGui::TextColored(WarningColor, "Records dropped: %u", report.stats.droppedRecords);
  }
  if (report.draws.empty()) {
    ImGui::TextUnformatted("No draw records this frame");
    return;
  }
  filter_combo("Kind", g_state.kindFilter, scene::KindCount,
               [](size_t i) { return scene::kind_name(static_cast<scene::Kind>(i)); });
  filter_combo("Role", g_state.roleFilter, scene::RoleCount,
               [](size_t i) { return scene::role_name(static_cast<scene::Role>(i)); });
  ImGui::Checkbox("Main-class only", &g_state.mainOnly);
  g_state.rows.clear();
  for (size_t i = 0; i < report.draws.size(); ++i) {
    const scene::DrawClass& cls = report.draws[i].cls;
    if ((g_state.kindFilter != 0 && static_cast<int>(cls.kind) != g_state.kindFilter - 1) ||
        (g_state.roleFilter != 0 && static_cast<int>(cls.key.role) != g_state.roleFilter - 1) ||
        (g_state.mainOnly && !cls.mainClass)) {
      continue;
    }
    g_state.rows.push_back(static_cast<uint32_t>(i));
  }
  const float height = ImGui::GetTextLineHeightWithSpacing() * ListRows;
  if (!ImGui::BeginTable("##draws", 10, ListFlags, {0.f, height})) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  for (const char* column :
       {"Index", "Kind", "Role", "Check", "Window", "Object", "Categories", "Flags", "Scope", "Vertices"}) {
    ImGui::TableSetupColumn(column);
  }
  ImGui::TableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(g_state.rows.size()));
  while (clipper.Step()) {
    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
      const scene::ClassifiedDraw& draw = report.draws[g_state.rows[static_cast<size_t>(row)]];
      const scene::DrawClass& cls = draw.cls;
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("%u", cls.index);
      ImGui::TableNextColumn();
      text(scene::kind_name(cls.kind));
      ImGui::TableNextColumn();
      text(scene::role_name(cls.key.role));
      ImGui::TableNextColumn();
      text(scene::check_name(cls.check));
      ImGui::TableNextColumn();
      text(scene::window_name(cls.key.window));
      ImGui::TableNextColumn();
      if (cls.key.object.valid) {
        ImGui::Text("0x%llx/%u", static_cast<unsigned long long>(cls.key.object.id), cls.key.object.subId);
      } else {
        ImGui::TextUnformatted("-");
      }
      ImGui::TableNextColumn();
      text(category_names(cls.key.categories));
      ImGui::TableNextColumn();
      ImGui::Text("0x%x", cls.key.flags);
      ImGui::TableNextColumn();
      ImGui::Text("0x%08x", cls.key.scope);
      ImGui::TableNextColumn();
      ImGui::Text("%u%s", draw.vertices, draw.indexed ? " indexed" : "");
    }
  }
  ImGui::EndTable();
}
} // namespace

void draw_scene_tab() {
  // One copy of the pointer per draw; the report itself is immutable and never copied
  const std::shared_ptr<const scene::FrameReport> report = scene::latest_report();
  draw_view_controls();
  if (report == nullptr) {
    ImGui::TextUnformatted("No classified frame yet");
    return;
  }
  ImGui::SeparatorText("Frame");
  draw_header(*report);
  draw_counters(report->stats);
  draw_hint_health(report->stats.hints);
  draw_view_counts(report->stats.view);
  draw_objects(*report);
  draw_records(*report);
}
} // namespace aurora::render::debug

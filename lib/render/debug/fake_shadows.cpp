#include "fake_shadows.hpp"

#include "debug.hpp"

#include "../json.hpp"
#include "../material/registry.hpp"
#include "../scene/classify.hpp"

#include <aurora/render.hpp>
#include <aurora/texture.hpp>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace aurora::render::debug::fake_shadows {
namespace {
using fakeshadow::Force;
using fakeshadow::Gate;
using fakeshadow::GateCount;
using fakeshadow::Outcome;
using fakeshadow::OutcomeCount;

constexpr uint16_t BaseCategories = 0x1FFF;
constexpr uint32_t BaseCategoryCount = 13;
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                       ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX |
                                       ImGuiTableFlags_ScrollY;
constexpr float TableRows = 14.f;
constexpr std::array<const char*, OutcomeCount> OutcomeNames{
    "bound",           "inScope",      "white",        "clear",         "skipped",
    "gateOff",         "outOfScope",   "outOfRole",    "entryOwned",    "refusedIndirect",
    "refusedCoverage", "refusedWhite", "refusedClear", "refusedWrites", "refusedShared"};
constexpr std::array<const char*, 4> NeutralNames{"none", "white", "clear", "skip"};
constexpr std::array<const char*, GateCount> GateNames{"light", "cloud", "ao"};
constexpr std::array<const char*, 3> ForceNames{"Auto", "Off", "On"};
constexpr std::array<const char*, 3> ForceKeys{"auto", "off", "on"};

static_assert(static_cast<size_t>(Outcome::RefusedShared) + 1 == OutcomeCount);

std::array<std::optional<Force>, GateCount> g_queued{};
uint8_t g_changed = 0;
int g_forceVisible = -1;
uint16_t g_selected = 0;

bool visible() noexcept { return g_forceVisible >= 0 ? g_forceVisible != 0 : tab_visible(TabName); }

std::string ascii_upper(std::string_view value) {
  std::string out{value};
  for (char& c : out) {
    if (c >= 'a' && c <= 'z') {
      c = static_cast<char>(c - 'a' + 'A');
    }
  }
  return out;
}

const char* neutral_name(fakeshadow::Neutral neutral) noexcept {
  const auto index = static_cast<size_t>(neutral);
  return index < NeutralNames.size() ? NeutralNames[index] : "?";
}

const char* gate_name(uint8_t gateBit) noexcept {
  const auto index = static_cast<size_t>(std::countr_zero(gateBit));
  return std::has_single_bit(gateBit) && index < GateNames.size() ? GateNames[index] : "none";
}

std::string flags_text(const Row& row) {
  std::string out;
  for (const auto& [on, name] : {std::pair{!row.scoped, "no scope"}, std::pair{row.unseen, "unseen"},
                                 std::pair{!row.registered, "unregistered"}}) {
    if (on) {
      out += out.empty() ? "" : ", ";
      out += name;
    }
  }
  return out;
}

yyjson_mut_val* counts_json(yyjson_mut_doc* doc, const fakeshadow::OutcomeCounts& counts) {
  yyjson_mut_val* out = yyjson_mut_obj(doc);
  for (size_t i = 0; i < OutcomeCount; ++i) {
    yyjson_mut_obj_add_uint(doc, out, OutcomeNames[i], counts[i]);
  }
  return out;
}

void draw_status(const fakeshadow::TextureReport* report, const Rows& rows) {
  ImGui::TextUnformatted("No gate producer before M13: gates follow Force");
  const bool hintsOff = report != nullptr ? report->hintsOff : !hints_enabled();
  if (hintsOff) {
    ImGui::TextColored(WarningColor, "Render hints are off, so no draw reaches the gate");
  }
  if (!effective_settings().textureMaterials) {
    ImGui::TextColored(WarningColor, "Texture Materials is off, so the gate is idle");
  }
  if (report == nullptr) {
    ImGui::TextUnformatted("No fake shadow report yet");
    return;
  }
  ImGui::Text("State: %s", report->active ? "active" : report->observing ? "observing" : "idle");
  ImGui::Text("Rules: %zu, unscoped %u, unseen %u", rows.rows.size(), rows.unscoped, rows.unseen);
}

void draw_gates(const fakeshadow::TextureReport* report) {
  ImGui::SeparatorText("Gates");
  for (size_t i = 0; i < GateCount; ++i) {
    const auto gate = static_cast<Gate>(i);
    int current = static_cast<int>(shown_force(gate));
    ImGui::SetNextItemWidth(100.f);
    if (ImGui::Combo(GateNames[i], &current, ForceNames.data(), static_cast<int>(ForceNames.size()))) {
      queue_force(gate, static_cast<Force>(current));
    }
    const fakeshadow::GateView view = report != nullptr ? report->gates[i] : fakeshadow::GateView{};
    ImGui::SameLine();
    ImGui::Text("effective %s, source %s%s", view.effective ? "on" : "off", view.live ? "on" : "off",
                g_queued[i] ? " (applies next frame)" : "");
  }
}

void draw_cross_check() {
  ImGui::SeparatorText("Scene cross-check");
  const std::shared_ptr<const scene::FrameReport> scene = scene::latest_report();
  if (scene == nullptr) {
    ImGui::TextUnformatted("No classified frame yet");
    return;
  }
  ImGui::Text("FAKE_SHADOW: %u", scene->stats.fakeShadow);
  ImGui::Text("FAKE_SHADOW and OVERLAY: %u", scene->stats.fakeShadowOverlay);
}

void draw_rows(const fakeshadow::TextureReport* report, const Rows& rows) {
  ImGui::SeparatorText("Per key");
  if (ImGui::Button("Copy report JSON")) {
    ImGui::SetClipboardText(report_json(report, rows).c_str());
  }
  const auto selected = std::ranges::find(rows.rows, g_selected, &Row::materialId);
  ImGui::SameLine();
  ImGui::BeginDisabled(selected == rows.rows.end());
  if (ImGui::Button("Copy key") && selected != rows.rows.end()) {
    ImGui::SetClipboardText(selected->key.c_str());
  }
  ImGui::EndDisabled();
  if (rows.rows.empty()) {
    ImGui::TextUnformatted("No fake shadow materials registered");
    return;
  }
  const ImVec2 size{0.f, ImGui::GetTextLineHeightWithSpacing() * TableRows};
  if (!ImGui::BeginTable("##fake_shadow_keys", static_cast<int>(5 + OutcomeCount), TableFlags, size)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(1, 1);
  for (const char* header : {"Key", "Neutral", "Gate", "Scope", "Flags"}) {
    ImGui::TableSetupColumn(header);
  }
  for (const char* header : OutcomeNames) {
    ImGui::TableSetupColumn(header);
  }
  ImGui::TableHeadersRow();
  for (const Row& row : rows.rows) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushID(row.materialId);
    if (ImGui::Selectable(row.key.c_str(), row.materialId == g_selected, ImGuiSelectableFlags_SpanAllColumns)) {
      g_selected = row.materialId;
    }
    ImGui::PopID();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(neutral_name(row.neutral));
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(gate_name(row.gateBit));
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(scope_text(row.scopeMask, row.scoped).c_str());
    ImGui::TableNextColumn();
    const std::string flags = flags_text(row);
    ImGui::TextColored(WarningColor, "%s", flags.c_str());
    for (const uint64_t count : row.counts) {
      ImGui::TableNextColumn();
      ImGui::Text("%llu", static_cast<unsigned long long>(count));
    }
  }
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::TextUnformatted("Total");
  for (int i = 0; i < 4; ++i) {
    ImGui::TableNextColumn();
  }
  for (const uint64_t count : rows.totals) {
    ImGui::TableNextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(count));
  }
  ImGui::EndTable();
}
} // namespace

void initialize() { register_tab(TabName, FakeShadowsTabOrder, &draw_tab); }

void shutdown() noexcept {
  for (size_t i = 0; i < GateCount; ++i) {
    if ((g_changed >> i & 1u) != 0) {
      fakeshadow::set_force(static_cast<Gate>(i), Force::Auto);
    }
  }
  fakeshadow::set_stats_enabled(false);
  g_queued = {};
  g_changed = 0;
  g_selected = 0;
}

void on_latch() noexcept {
  for (size_t i = 0; i < GateCount; ++i) {
    if (const std::optional<Force> force = std::exchange(g_queued[i], std::nullopt)) {
      fakeshadow::set_force(static_cast<Gate>(i), *force);
      g_changed |= static_cast<uint8_t>(1u << i);
    }
  }
  fakeshadow::set_stats_enabled(visible());
}

void draw_tab() {
  const std::shared_ptr<const fakeshadow::TextureReport> report = fakeshadow::latest_report();
  const std::shared_ptr<const material::Snapshot> snapshot = material::latched_snapshot();
  const Rows rows = build_rows(report.get(), snapshot.get());
  draw_status(report.get(), rows);
  draw_gates(report.get());
  draw_cross_check();
  draw_rows(report.get(), rows);
}

void queue_force(Gate gate, Force force) noexcept { g_queued[static_cast<size_t>(gate)] = force; }

Force shown_force(Gate gate) noexcept {
  const std::optional<Force>& queued = g_queued[static_cast<size_t>(gate)];
  return queued ? *queued : fakeshadow::force(gate);
}

Rows build_rows(const fakeshadow::TextureReport* report, const material::Snapshot* snapshot) {
  Rows out;
  if (report == nullptr || report->table == nullptr) {
    return out;
  }
  const std::vector<fakeshadow::RuleInfo>& rules = report->table->rules;
  out.rows.reserve(rules.size());
  for (size_t i = 0; i < rules.size(); ++i) {
    const fakeshadow::Rule& rule = rules[i].rule;
    Row& row = out.rows.emplace_back();
    row.materialId = rule.materialId;
    row.key = rules[i].key;
    row.neutral = rule.neutral;
    row.gateBit = rule.gateBit;
    row.scopeMask = rule.scopeMask;
    row.scoped = rule.scoped;
    if (snapshot != nullptr && snapshot->live(rule.materialId)) {
      const material::MaterialRecord& record = snapshot->record(rule.materialId);
      const uint32_t gate = material::shadow_gate(record.summary);
      row.registered = true;
      row.neutral = static_cast<fakeshadow::Neutral>(material::shadow_neutral(record.summary));
      row.gateBit = static_cast<uint8_t>(gate != 0 ? 1u << (gate - 1) : 0u);
      row.scoped = (record.summary & material::ShadowScope) != 0;
      row.scopeMask = row.scoped ? static_cast<uint16_t>(record.shadowScope & BaseCategories) : BaseCategories;
      if (const texture::TextureSourceKey* key = snapshot->key(rule.materialId); key != nullptr) {
        row.key = texture::format_material_key(*key);
      }
    }
    if (i < report->counts.size()) {
      row.counts = report->counts[i];
    }
    row.unseen = row.counts[static_cast<size_t>(Outcome::Bound)] == 0;
    out.unscoped += row.scoped ? 0 : 1;
    out.unseen += row.unseen ? 1 : 0;
    for (size_t outcome = 0; outcome < OutcomeCount; ++outcome) {
      out.totals[outcome] += row.counts[outcome];
    }
  }
  return out;
}

std::string scope_text(uint16_t scopeMask, bool scoped) {
  if (!scoped) {
    return "all";
  }
  std::string out;
  for (uint32_t bit = 0; bit < BaseCategoryCount; ++bit) {
    if ((scopeMask >> bit & 1u) != 0) {
      out += out.empty() ? "" : " ";
      out += ascii_upper(scene::category_name(bit));
    }
  }
  return out.empty() ? "none" : out;
}

std::string report_json(const fakeshadow::TextureReport* report, const Rows& rows) {
  json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  yyjson_mut_doc* out = doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(out);
  yyjson_mut_doc_set_root(out, root);
  yyjson_mut_obj_add_bool(out, root, "active", report != nullptr && report->active);
  yyjson_mut_obj_add_bool(out, root, "observing", report != nullptr && report->observing);
  yyjson_mut_obj_add_bool(out, root, "hintsOff", report != nullptr && report->hintsOff);
  yyjson_mut_obj_add_uint(out, root, "generation",
                          report != nullptr && report->table != nullptr ? report->table->generation : 0);
  yyjson_mut_val* gates = yyjson_mut_obj_add_obj(out, root, "gates");
  for (size_t i = 0; i < GateCount; ++i) {
    const fakeshadow::GateView view = report != nullptr ? report->gates[i] : fakeshadow::GateView{};
    yyjson_mut_val* gate = yyjson_mut_obj_add_obj(out, gates, GateNames[i]);
    yyjson_mut_obj_add_str(out, gate, "force", ForceKeys[static_cast<size_t>(view.force)]);
    yyjson_mut_obj_add_bool(out, gate, "live", view.live);
    yyjson_mut_obj_add_bool(out, gate, "effective", view.effective);
  }
  yyjson_mut_val* list = yyjson_mut_obj_add_arr(out, root, "rules");
  for (const Row& row : rows.rows) {
    yyjson_mut_val* entry = yyjson_mut_arr_add_obj(out, list);
    yyjson_mut_obj_add_strncpy(out, entry, "key", row.key.data(), row.key.size());
    yyjson_mut_obj_add_uint(out, entry, "id", row.materialId);
    yyjson_mut_obj_add_str(out, entry, "neutral", neutral_name(row.neutral));
    yyjson_mut_obj_add_str(out, entry, "gate", gate_name(row.gateBit));
    if (row.scoped) {
      yyjson_mut_val* scope = yyjson_mut_obj_add_arr(out, entry, "scope");
      for (uint32_t bit = 0; bit < BaseCategoryCount; ++bit) {
        if ((row.scopeMask >> bit & 1u) != 0) {
          const std::string name = ascii_upper(scene::category_name(bit));
          yyjson_mut_arr_add_strncpy(out, scope, name.data(), name.size());
        }
      }
    } else {
      yyjson_mut_obj_add_str(out, entry, "scope", "all");
    }
    yyjson_mut_obj_add_bool(out, entry, "registered", row.registered);
    yyjson_mut_obj_add_bool(out, entry, "unseen", row.unseen);
    yyjson_mut_obj_add_val(out, entry, "counts", counts_json(out, row.counts));
  }
  yyjson_mut_obj_add_val(out, root, "totals", counts_json(out, rows.totals));
  return json::write(out, true);
}

namespace testing {
void reset() noexcept {
  shutdown();
  g_forceVisible = -1;
}

void force_visible(int value) noexcept { g_forceVisible = value; }
} // namespace testing
} // namespace aurora::render::debug::fake_shadows

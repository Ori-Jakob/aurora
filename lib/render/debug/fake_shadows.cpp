#include "fake_shadows.hpp"

#include "debug.hpp"

#include "../json.hpp"
#include "../fakeshadow/gate.hpp"
#include "../material/registry.hpp"
#include "../scene/classify.hpp"
#include "../shadows/vsm_pages.hpp"

#include <aurora/render.hpp>
#include <aurora/texture.hpp>
#include <dolphin/gx/GXAurora.h>
#include <fmt/format.h>
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
constexpr std::array<const char*, fakeshadow::gate::ConsumerForceCount> ConsumerForceNames{"Auto", "Off", "On",
                                                                                           "Flip30"};
constexpr std::array<const char*, 3> GameNames{"Auto", "Keep", "Remove"};
constexpr std::array<const char*, 4> CoverNames{"missing", "full", "partial", "excluded only"};
constexpr uint8_t SourceCount = 8;

static_assert(static_cast<size_t>(Outcome::RefusedShared) + 1 == OutcomeCount);

std::array<std::optional<Force>, GateCount> g_queued{};
uint8_t g_changed = 0;
int g_forceVisible = -1;
uint16_t g_selected = 0;
std::optional<fakeshadow::gate::ConsumerForce> g_queuedGate;
bool g_gateChanged = false;

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

const char* yes_no(bool value) noexcept { return value ? "yes" : "no"; }

void draw_gate_state() {
  ImGui::SeparatorText("Game shadow gate");
  const GateState s = fakeshadow::gate::state();
  const shadows::vsm::SunPlan& plan = shadows::vsm::plan();
  const std::string_view idle = shadows::vsm::idle_name(plan.idle);
  ImGui::Text("Frame %u, setting %s, keep cues %s", s.frame, GameNames[static_cast<size_t>(s.game)],
              yes_no(s.keepCues));
  ImGui::Text("Consumer %s, previous resolve %s, coverage complete %s", yes_no(s.consumer), yes_no(s.prevResolveRan),
              yes_no(s.coverageComplete));
  ImGui::Text("Plan: %s, light %u at %.1f deg (%.*s)", plan.producing ? "producing" : "off", plan.lightId,
              plan.elevationDeg, static_cast<int>(idle.size()), idle.data());
  ImGui::Text("Declared: sun %s, KEY_ROOM %s, lantern %s; interior %s; sky %.2f", yes_no(s.sunDeclared),
              yes_no(s.keyRoomDeclared), yes_no(s.lanternDeclared), yes_no(s.interior), s.skyVisibility);
  const bool keyRoom = (plan.lightFlags & AURORA_LIGHT_KEY_ROOM) != 0;
  std::string covered = "Covered:";
  for (uint8_t i = 0; i < SourceCount; ++i) {
    const auto source = static_cast<ShadowSource>(i);
    covered += fmt::format(" {} {}", fakeshadow::gate::source_name(source),
                           yes_no(fakeshadow::gate::covered(s, keyRoom, source, ShadowClass::Real)));
  }
  ImGui::TextWrapped("%s", covered.c_str());
  int force = static_cast<int>(shown_gate_force());
  ImGui::SetNextItemWidth(100.f);
  if (ImGui::Combo("Consumer force", &force, ConsumerForceNames.data(), static_cast<int>(ConsumerForceNames.size()))) {
    queue_gate_force(static_cast<fakeshadow::gate::ConsumerForce>(force));
  }
  if (g_queuedGate) {
    ImGui::SameLine();
    ImGui::TextUnformatted("(applies next frame)");
  }
}

void counter_row(const std::string& name, uint64_t current, uint64_t total) {
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::TextUnformatted(name.c_str());
  ImGui::TableNextColumn();
  ImGui::Text("%llu", static_cast<unsigned long long>(current));
  ImGui::TableNextColumn();
  ImGui::Text("%llu", static_cast<unsigned long long>(total));
}

void draw_gate_counters() {
  const GameShadowStats stats = fakeshadow::gate::stats(false);
  if (!ImGui::BeginTable("##gate_counters", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
    return;
  }
  ImGui::TableSetupColumn("Counter");
  ImGui::TableSetupColumn("Frame");
  ImGui::TableSetupColumn("Total");
  ImGui::TableHeadersRow();
  const GameShadowCounts& c = stats.current;
  const GameShadowCounts& t = stats.total;
  counter_row("entries", c.entries, t.entries);
  for (size_t i = 1; i < ShadowDecisionCount; ++i) {
    counter_row(std::string{fakeshadow::gate::decision_name(static_cast<ShadowDecision>(i))}, c.decisions[i],
                t.decisions[i]);
  }
  counter_row("missing caster (latch)", c.missingCasterLatch, t.missingCasterLatch);
  counter_row("missing caster (cast bits)", c.missingCasterCast, t.missingCasterCast);
  counter_row("cast bits pending", c.castBitsPending, t.castBitsPending);
  counter_row("double-darkening entries", c.doubleDarkEntries, t.doubleDarkEntries);
  counter_row("kept by the setting", c.userKept, t.userKept);
  counter_row("transitions", c.transitions, t.transitions);
  counter_row("flip violations", c.flipViolations, t.flipViolations);
  for (size_t i = 1; i < ShadowDecisionCount; ++i) {
    if (t.safetyReturns[i] != 0) {
      counter_row("safety return: " + std::string{fakeshadow::gate::decision_name(static_cast<ShadowDecision>(i))},
                  c.safetyReturns[i], t.safetyReturns[i]);
    }
  }
  counter_row("safety returns per key, max", c.safetyReturnsPerKeyMax, t.safetyReturnsPerKeyMax);
  counter_row("provisional casters", c.provisional, t.provisional);
  counter_row("provisional failed", c.provisionalFailed, t.provisionalFailed);
  counter_row("partial casters", c.partialCaster, t.partialCaster);
  counter_row("table overflow", c.overflow, t.overflow);
  counter_row("sun loss frames", c.sunLossFrames, t.sunLossFrames);
  ImGui::EndTable();
}

std::string caster_text(const fakeshadow::gate::EntryRow& entry) {
  std::string out;
  for (const fakeshadow::gate::CasterRow& caster : entry.casters) {
    out += fmt::format("{}{:x} {}{}", out.empty() ? "" : ", ", caster.id, CoverNames[static_cast<size_t>(caster.cover)],
                       caster.enteredNow ? " (new)" : "");
  }
  return out;
}

void draw_gate_entries() {
  const std::vector<fakeshadow::gate::EntryRow> entries = fakeshadow::gate::entry_rows();
  if (entries.empty()) {
    ImGui::TextUnformatted("No game shadow entries decided this frame");
    return;
  }
  const ImVec2 size{0.f, ImGui::GetTextLineHeightWithSpacing() * TableRows};
  if (!ImGui::BeginTable("##gate_entries", 6, TableFlags, size)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(1, 1);
  for (const char* header : {"Key", "Source", "Class", "Decision", "Hold", "Casters"}) {
    ImGui::TableSetupColumn(header);
  }
  ImGui::TableHeadersRow();
  for (const fakeshadow::gate::EntryRow& entry : entries) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("%016llx", static_cast<unsigned long long>(entry.key));
    for (const std::string_view value :
         {fakeshadow::gate::source_name(entry.source), fakeshadow::gate::class_name(entry.cls),
          fakeshadow::gate::decision_name(entry.decision)}) {
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(value.data(), value.data() + value.size());
    }
    ImGui::TableNextColumn();
    ImGui::Text("%u", entry.hold);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(caster_text(entry).c_str());
  }
  ImGui::EndTable();
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
  if (std::exchange(g_gateChanged, false)) {
    fakeshadow::gate::set_force(fakeshadow::gate::ConsumerForce::Auto);
  }
  g_queuedGate.reset();
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
  if (const std::optional<fakeshadow::gate::ConsumerForce> force = std::exchange(g_queuedGate, std::nullopt)) {
    fakeshadow::gate::set_force(*force);
    g_gateChanged = true;
  }
  fakeshadow::set_stats_enabled(visible());
}

void draw_tab() {
  const std::shared_ptr<const fakeshadow::TextureReport> report = fakeshadow::latest_report();
  const std::shared_ptr<const material::Snapshot> snapshot = material::latched_snapshot();
  const Rows rows = build_rows(report.get(), snapshot.get());
  draw_gate_state();
  draw_gate_counters();
  draw_gate_entries();
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

void queue_gate_force(fakeshadow::gate::ConsumerForce force) noexcept { g_queuedGate = force; }

fakeshadow::gate::ConsumerForce shown_gate_force() noexcept { return g_queuedGate.value_or(fakeshadow::gate::force()); }

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

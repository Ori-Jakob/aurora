#include "textures.hpp"

#include "debug.hpp"
#include "views.hpp"

#include "../fakeshadow/texture_gate.hpp"
#include "../material/registry.hpp"
#include "../material/user_layer.hpp"
#include "../scene/classify.hpp"
#include "../../gfx/pipeline_cache.hpp"
#include "../../gfx/texture_replacement.hpp"
#include "../../gx/guest_memory.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../gx/texture.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render.hpp>
#include <imgui.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace aurora::render::debug::textures {
namespace {
using Clock = std::chrono::steady_clock;

constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr ImGuiTableFlags TableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;
constexpr float TableRows = 14.f;
constexpr std::array<std::string_view, 4> FlagNames{"stochastic", "noPbr", "cutout", "unlit"};
constexpr std::array<const char*, 4> NeutralNames{"", "white", "clear", "skip"};
constexpr std::array<const char*, 4> GateNames{"", "light", "cloud", "ao"};
constexpr std::array<std::string_view, 13> ScopeNames{"STATIC", "DYNAMIC",  "SKINNED",  "TERRAIN", "FOLIAGE",
                                                      "WATER",  "SKY",      "PARTICLE", "DECAL",   "UI",
                                                      "PLAYER", "EMISSIVE", "OVERLAY"};
constexpr std::array<float, 4> StochasticMin{0.0625f, 0.f, 0.f, 1.f};
constexpr std::array<float, 4> StochasticMax{16.f, 1.f, 1.f, 32.f};
constexpr std::array<float, 4> StochasticDefault{1.f, 0.f, 0.6f, 7.f};

struct MemoKey {
  const void* data = nullptr;
  uint32_t texObjId = 0;
  uint32_t texDataVersion = 0;
  uint32_t tlutObjId = 0;
  uint32_t tlutDataVersion = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t format = 0;
  bool operator==(const MemoKey&) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const MemoKey& key) {
    return H::combine(std::move(h), key.data, key.texObjId, key.texDataVersion, key.tlutObjId, key.tlutDataVersion,
                      key.width, key.height, key.format);
  }
};

struct MemoValue {
  texture::TextureSourceKey key;
  bool hasKey = false;
  bool pointerReplacement = false;
};

struct RowId {
  uint64_t a = 0;
  uint64_t b = 0;
  uint32_t size = 0;
  uint32_t format = 0;
  SlotKind kind = SlotKind::Empty;
  bool keyed = false;
  bool operator==(const RowId&) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const RowId& id) {
    return H::combine(std::move(h), id.a, id.b, id.size, id.format, id.kind, id.keyed);
  }
};

struct Live {
  std::vector<DrawRecord> draws;
  std::vector<SlotRecord> slots;
  std::vector<KeyRow> keys;
  absl::flat_hash_map<RowId, uint32_t> rows;
  std::vector<gx::PipelineConfig> warm;
  absl::flat_hash_set<gfx::PipelineRef> warmSeen;
  Overflow overflow;
  uint32_t pendingKeys = 0;
  uint32_t pausedKeys = 0;
  uint32_t memoInserts = 0;
  uint64_t hashedBytes = 0;
  bool recorded = false;
};

struct Warm {
  std::vector<gx::PipelineConfig> pending;
  std::vector<gfx::PipelineRef> refs;
  Clock::time_point start;
  uint32_t observedFrames = 0;
  uint32_t ready = 0;
  uint32_t total = 0;
};

struct Hover {
  uint64_t sequence = 0;
  uint32_t record = 0;
  bool valid = false;
  bool pinned = false;
};

struct Selection {
  texture::TextureSourceKey key;
  bool valid = false;
  bool pointerReplacement = false;
};

struct Ui {
  std::array<char, 64> filter{};
  bool mainOnly = false;
  bool registeredOnly = false;
  bool neutralOnly = false;
  std::array<char, 72> className{};
  std::string message;
};

bool default_pointer_probe(const GXTexObj_& obj) noexcept {
  const auto result = gfx::texture_replacement::find_pointer_replacement(obj);
  return result.has_value() && static_cast<bool>(result->handle);
}

Live g_live;
absl::flat_hash_map<MemoKey, MemoValue> g_memo;
uint64_t g_sequence = 1;
pick::ArmState g_arm = pick::ArmState::Off;
uint64_t g_hashBudget = HashBudgetBytes;
bool (*g_pointerProbe)(const GXTexObj_&) noexcept = &default_pointer_probe;

bool g_installed = false;
bool g_reserved = false;
int g_forceVisible = -1;
bool g_pickEnabled = false;
std::array<std::shared_ptr<const TexturesReport>, ReportRing> g_ring;
size_t g_ringNext = 0;
SeenSet g_seen;
Warm g_warm;
uint64_t g_armedBaseKey = 0;
std::string_view g_reason;
Hover g_hover;
Selection g_selection;
Ui g_ui;
std::unique_ptr<material::UserLayer> g_userLayer;

constexpr scene::ObserverHooks g_hooks{.observe = &observe};

void text(std::string_view value) { ImGui::TextUnformatted(value.data(), value.data() + value.size()); }

std::string_view kind_name(SlotKind kind) noexcept {
  switch (kind) {
  case SlotKind::Empty:
    return "empty";
  case SlotKind::Static:
    return "static";
  case SlotKind::Palette:
    return "palette";
  case SlotKind::Copy:
    return "copy";
  case SlotKind::DynamicPalette:
    return "copy palette";
  }
  return "?";
}

std::string_view arm_name(pick::ArmState state) noexcept {
  switch (state) {
  case pick::ArmState::Off:
    return "off";
  case pick::ArmState::Warming:
    return "warming";
  case pick::ArmState::Armed:
    return "armed";
  }
  return "?";
}

SlotKind slot_kind(const GXTexObj_& obj) noexcept {
  const bool palette = gx::is_palette_format(obj.format());
  if (gx::g_gxState.copyTextures.contains(obj.data)) {
    return palette ? SlotKind::DynamicPalette : SlotKind::Copy;
  }
  if (!obj.has_data()) {
    return SlotKind::Empty;
  }
  return palette ? SlotKind::Palette : SlotKind::Static;
}

KeyState resolve_key(Live& live, const GXTexObj_& obj, const GXTlutObj_* tlut, MemoValue& out) noexcept {
  const MemoKey memoKey{
      .data = obj.data,
      .texObjId = obj.texObjId,
      .texDataVersion = obj.texDataVersion,
      .tlutObjId = tlut != nullptr ? tlut->tlutObjId : 0,
      .tlutDataVersion = tlut != nullptr ? tlut->tlutDataVersion : 0,
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
  };
  if (const auto it = g_memo.find(memoKey); it != g_memo.end()) {
    out = it->second;
    return out.hasKey ? KeyState::Ready : KeyState::Failed;
  }
  if (gx::guest_memory::g_hooks.load(std::memory_order_acquire) != nullptr) {
    return KeyState::Paused;
  }
  const size_t textureBytes = gx::texture::texture_source_size(memoKey.format, memoKey.width, memoKey.height, 1);
  const size_t tlutBytes = tlut != nullptr ? gx::texture::tlut_source_size(tlut->numEntries) : 0;
  if (live.hashedBytes + textureBytes + tlutBytes > g_hashBudget || live.memoInserts >= KeyMemoInsertsPerFrame ||
      g_memo.size() >= KeyMemoCapacity) {
    return KeyState::Pending;
  }
  std::optional<texture::TextureSourceKey> key;
  if (textureBytes != 0) {
    const auto textureData = gx::guest_memory::read(gx::guest_memory::Kind::Texture, obj.data, textureBytes);
    std::span<const uint8_t> tlutData;
    if (tlut != nullptr && tlutBytes != 0) {
      tlutData = gx::guest_memory::read(gx::guest_memory::Kind::Tlut, tlut->data, tlutBytes);
    }
    key = texture::compute_source_key(memoKey.format, memoKey.width, memoKey.height, textureData, tlutData);
  }
  out = {.key = key.value_or(texture::TextureSourceKey{}),
         .hasKey = key.has_value(),
         .pointerReplacement = g_pointerProbe(obj)};
  g_memo.emplace(memoKey, out);
  ++live.memoInserts;
  live.hashedBytes += textureBytes + tlutBytes;
  return out.hasKey ? KeyState::Ready : KeyState::Failed;
}

uint32_t upsert_row(Live& live, SlotRecord& slot, const GXTexObj_& obj, const GXTlutObj_* tlut, bool main) noexcept {
  MemoValue value;
  KeyState state = KeyState::None;
  if ((slot.kind == SlotKind::Static || slot.kind == SlotKind::Palette) && obj.texObjId != 0) {
    state = resolve_key(live, obj, tlut, value);
  }
  slot.pointerReplacement = value.pointerReplacement;
  const uint32_t size = uint32_t{slot.width} << 16 | slot.height;
  const RowId id = state == KeyState::Ready
                       ? RowId{value.key.textureHash, value.key.tlutHash, size, slot.format, slot.kind, true}
                       : RowId{reinterpret_cast<uintptr_t>(obj.data),
                               uint64_t{obj.texObjId} << 32 | obj.texDataVersion,
                               size,
                               slot.format,
                               slot.kind,
                               false};
  uint32_t row = NoRow;
  if (const auto it = live.rows.find(id); it != live.rows.end()) {
    row = it->second;
  } else {
    if (live.keys.size() >= MaxKeyRows) {
      ++live.overflow.keys;
      return NoRow;
    }
    row = static_cast<uint32_t>(live.keys.size());
    live.keys.push_back({
        .key = value.key,
        .material = slot.material,
        .width = slot.width,
        .height = slot.height,
        .format = slot.format,
        .kind = slot.kind,
        .state = state,
        .pointerReplacement = value.pointerReplacement,
    });
    live.rows.emplace(id, row);
    live.pendingKeys += state == KeyState::Pending ? 1 : 0;
    live.pausedKeys += state == KeyState::Paused ? 1 : 0;
  }
  KeyRow& entry = live.keys[row];
  ++entry.draws;
  entry.mainDraws += main ? 1 : 0;
  entry.neutralDraws += slot.neutral ? 1 : 0;
  entry.replaced |= slot.replaced;
  entry.stochastic |= slot.stochastic;
  return row;
}

void record_slot(Live& live, uint32_t index, const gx::PipelineConfig& config, const gx::ShaderInfo& info, bool main,
                 bool neutral) noexcept {
  const GXTexObj_& obj = gx::g_gxState.loadedTextures[index];
  const gfx::TextureBind& bind = gx::g_gxState.textures[index];
  SlotRecord& slot = live.slots.emplace_back();
  slot.texmap = static_cast<uint8_t>(index);
  slot.width = static_cast<uint16_t>(std::min<uint32_t>(obj.width(), UINT16_MAX));
  slot.height = static_cast<uint16_t>(std::min<uint32_t>(obj.height(), UINT16_MAX));
  slot.format = static_cast<uint8_t>(obj.format());
  slot.wrapS = static_cast<uint8_t>(obj.wrap_s());
  slot.wrapT = static_cast<uint8_t>(obj.wrap_t());
  slot.kind = slot_kind(obj);
  slot.material = bind.material;
  slot.replaced = bind.ref && bind.ref->isReplacement;
  slot.stochastic = (config.shaderConfig.ext.stochasticMask >> index & 1u) != 0;
  slot.indirect = info.sampledIndTextures.test(index);
  slot.neutral = neutral;
  const GXTlutObj_* tlut = nullptr;
  if (slot.kind == SlotKind::Palette) {
    const auto tlutIndex = static_cast<size_t>(obj.tlut);
    if (tlutIndex < gx::g_gxState.loadedTluts.size() && gx::g_gxState.loadedTluts[tlutIndex].data != nullptr) {
      tlut = &gx::g_gxState.loadedTluts[tlutIndex];
    }
  }
  slot.row = upsert_row(live, slot, obj, tlut, main);
}

void note_config(Live& live, const gx::DrawData& draw, const gx::PipelineConfig& config) noexcept {
  if (live.warmSeen.contains(draw.pipeline)) {
    return;
  }
  if (live.warm.size() >= MaxWarmConfigs) {
    ++live.overflow.configs;
    return;
  }
  live.warmSeen.insert(draw.pipeline);
  live.warm.push_back(config);
}

void reserve_live() {
  Live& live = g_live;
  live.draws.reserve(MaxDraws);
  live.slots.reserve(MaxSlots);
  live.keys.reserve(MaxKeyRows);
  live.rows.clear();
  live.rows.reserve(MaxKeyRows);
  g_memo.reserve(KeyMemoCapacity);
  g_reserved = true;
}

void release_live() {
  g_live = {};
  g_memo = {};
  g_reserved = false;
}

void seal() {
  Live& live = g_live;
  if (live.recorded) {
    auto report = std::make_shared<TexturesReport>();
    report->sequence = g_sequence;
    report->draws.assign(live.draws.begin(), live.draws.end());
    report->slots.assign(live.slots.begin(), live.slots.end());
    report->keys.assign(live.keys.begin(), live.keys.end());
    report->overflow = live.overflow;
    report->pendingKeys = live.pendingKeys;
    report->pausedKeys = live.pausedKeys;
    report->hashedBytes = live.hashedBytes;
    for (const SlotRecord& slot : report->slots) {
      if (slot.material.id != 0 && slot.material.id < g_seen.size()) {
        g_seen.set(slot.material.id);
      }
    }
    g_ring[g_ringNext] = std::move(report);
    g_ringNext = (g_ringNext + 1) % ReportRing;
    if (g_arm == pick::ArmState::Warming) {
      g_warm.pending.insert(g_warm.pending.end(), live.warm.begin(), live.warm.end());
      ++g_warm.observedFrames;
    }
  }
  live.draws.clear();
  live.slots.clear();
  live.keys.clear();
  live.warm.clear();
  live.overflow = {};
  live.pendingKeys = 0;
  live.pausedKeys = 0;
  live.memoInserts = 0;
  live.hashedBytes = 0;
  live.recorded = false;
  ++g_sequence;
}

bool visible() noexcept { return g_forceVisible >= 0 ? g_forceVisible != 0 : tab_visible(TabName); }

void begin_warm() {
  g_warm.pending.clear();
  g_warm.refs.clear();
  g_warm.start = Clock::now();
  g_warm.observedFrames = 0;
  g_warm.ready = 0;
  g_warm.total = 0;
  g_live.warm.reserve(MaxWarmConfigs);
  g_live.warmSeen.clear();
  g_live.warmSeen.reserve(MaxWarmConfigs);
}

void end_warm() {
  g_warm = {};
  g_live.warm = {};
  g_live.warmSeen = {};
}

bool warm_step(const gfx::RenderTargetLayout& live) {
  const gfx::RenderTargetLayout armed = pick::armed_layout(live);
  for (uint32_t lookups = 0; g_warm.refs.size() < g_warm.pending.size() && lookups < WarmLookupsPerLatch; ++lookups) {
    g_warm.refs.push_back(gfx::find_debug_pipeline(g_warm.pending[g_warm.refs.size()], armed));
  }
  uint32_t ready = 0;
  for (const gfx::PipelineRef ref : g_warm.refs) {
    ready += gfx::pipeline_state(ref) != gfx::PipelineState::Pending ? 1 : 0;
  }
  const gfx::WarmProgress clear = gfx::warm_known_pipelines(armed, gfx::ShaderType::Clear);
  g_warm.ready = ready + clear.ready;
  g_warm.total = static_cast<uint32_t>(g_warm.pending.size()) + clear.total;
  const double seconds = std::chrono::duration<double>(Clock::now() - g_warm.start).count();
  return pick::warm_complete(
      {.ready = g_warm.ready, .total = g_warm.total, .seconds = seconds, .observedFrames = g_warm.observedFrames});
}

void disarm() {
  if (g_arm == pick::ArmState::Armed) {
    webgpu::set_debug_id_buffer(false);
  }
  g_arm = pick::ArmState::Off;
}

void step_arm(bool shown) {
  const pick::Capability capability = pick::pick_capability(pick::live_capability_inputs());
  g_reason = capability.reason;
  const gfx::RenderTargetLayout live = gfx::scene_render_target_layout();
  const uint64_t baseKey = gfx::pipeline_base_layout(live).key;
  const bool complete = g_arm == pick::ArmState::Warming && warm_step(live);
  const pick::ArmState next =
      pick::next_arm_state(g_arm, {.wanted = shown && g_pickEnabled && capability.available,
                                   .warmComplete = complete,
                                   .baseLayoutChanged = g_arm == pick::ArmState::Armed && baseKey != g_armedBaseKey});
  if (next == g_arm) {
    return;
  }
  disarm();
  if (next == pick::ArmState::Warming) {
    begin_warm();
    g_arm = next;
    return;
  }
  end_warm();
  if (next == pick::ArmState::Armed) {
    pick::ensure_pass();
    if (!webgpu::set_debug_id_buffer(true)) {
      g_reason = "the pick target could not be created";
      g_pickEnabled = false;
      return;
    }
    g_armedBaseKey = baseKey;
    g_arm = next;
  }
}

void update_pick_params() {
  if (!pick::pass_registered()) {
    return;
  }
  pick::Params params;
  if (g_arm == pick::ArmState::Armed && ImGui::GetCurrentContext() != nullptr) {
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse && ImGui::IsMousePosValid(&io.MousePos)) {
      const auto& surface = webgpu::g_graphicsConfig.surfaceConfiguration;
      const auto pixel = pick::efb_pixel_at(
          {io.MousePos.x, io.MousePos.y}, {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y},
          {surface.width, surface.height}, {webgpu::g_frameBuffer.size.width, webgpu::g_frameBuffer.size.height});
      if (pixel) {
        params = {pixel->x, pixel->y, 1};
      }
    }
  }
  pick::set_params(params);
}

void resolve_readback() {
  const auto value = pick::take_readback();
  if (!value || g_hover.pinned) {
    return;
  }
  const auto decoded = pick::decode_id(*value);
  g_hover.valid = decoded.has_value();
  if (decoded) {
    g_hover.sequence = decoded->sequence;
    g_hover.record = decoded->record;
  }
}

material::UserLayer& user_layer() {
  if (!g_userLayer) {
    g_userLayer = std::make_unique<material::UserLayer>();
  }
  if (!g_userLayer->opened()) {
    g_userLayer->open();
  }
  return *g_userLayer;
}

std::string key_text(const KeyRow& row) {
  if (row.state == KeyState::Ready) {
    return texture::format_material_key(row.key);
  }
  switch (row.state) {
  case KeyState::Pending:
    return "(key pending)";
  case KeyState::Paused:
    return "(keys paused)";
  case KeyState::Failed:
    return "(no key)";
  default:
    return std::string{"("} + std::string{kind_name(row.kind)} + ")";
  }
}

void select_row(const KeyRow& row) {
  g_selection = {.key = row.key, .valid = row.state == KeyState::Ready, .pointerReplacement = row.pointerReplacement};
  g_ui.message.clear();
  const material::UserEntry entry =
      g_selection.valid ? user_layer().read_entry(g_selection.key) : material::UserEntry{};
  std::snprintf(g_ui.className.data(), g_ui.className.size(), "%s", entry.className.c_str());
}

void draw_status(const TexturesReport* report) {
  const gfx::RenderTargetLayout live = gfx::scene_render_target_layout();
  const gfx::RenderTargetLayout base = gfx::pipeline_base_layout(live);
  ImGui::Text("Pick: %s", std::string{arm_name(g_arm)}.c_str());
  if (g_arm == pick::ArmState::Warming) {
    ImGui::SameLine();
    ImGui::Text("(warm %u/%u)", g_warm.ready, g_warm.total);
  }
  if (!g_reason.empty()) {
    ImGui::SameLine();
    ImGui::TextColored(WarningColor, "unavailable: %s", std::string{g_reason}.c_str());
  }
  ImGui::Text("Colour cost: %u B now, %u B without the pick, %u B with it", pick::layout_color_bytes(live),
              pick::layout_color_bytes(base), pick::layout_color_bytes(pick::armed_layout(live)));
  if (report != nullptr) {
    ImGui::Text("Frame %llu: %zu draws, %zu slots, %zu keys, %u pending, %u paused, %llu bytes hashed",
                static_cast<unsigned long long>(report->sequence), report->draws.size(), report->slots.size(),
                report->keys.size(), report->pendingKeys, report->pausedKeys,
                static_cast<unsigned long long>(report->hashedBytes));
    const Overflow& o = report->overflow;
    if (o.draws + o.slots + o.keys + o.configs != 0) {
      ImGui::TextColored(WarningColor, "Overflow: %u draws, %u slots, %u keys, %u configs", o.draws, o.slots, o.keys,
                         o.configs);
    }
  }
  const material::Snapshot* snapshot = material::latched_raw();
  ImGui::Text("Flagged keys seen %u of %u registered", flagged_seen(g_seen, snapshot),
              snapshot != nullptr ? snapshot->liveCount : 0u);
}

bool row_passes(const KeyRow& row, const material::Snapshot* snapshot, std::string_view stem) {
  if ((g_ui.mainOnly && row.mainDraws == 0) || (g_ui.neutralOnly && row.neutralDraws == 0)) {
    return false;
  }
  if (g_ui.registeredOnly && (row.state != KeyState::Ready || snapshot == nullptr || snapshot->find(row.key) == 0)) {
    return false;
  }
  const std::string_view filter{g_ui.filter.data()};
  return filter.empty() || stem.find(filter) != std::string_view::npos;
}

void draw_key_table(const TexturesReport& report) {
  ImGui::SetNextItemWidth(220.f);
  ImGui::InputText("Filter", g_ui.filter.data(), g_ui.filter.size());
  ImGui::SameLine();
  ImGui::Checkbox("Main draws", &g_ui.mainOnly);
  ImGui::SameLine();
  ImGui::Checkbox("Registered", &g_ui.registeredOnly);
  ImGui::SameLine();
  ImGui::Checkbox("Neutralized", &g_ui.neutralOnly);
  const material::Snapshot* snapshot = material::latched_raw();
  const ImVec2 size{0.f, ImGui::GetTextLineHeightWithSpacing() * TableRows};
  if (!ImGui::BeginTable("##texture_keys", 10, TableFlags, size)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  for (const char* header :
       {"Key", "Size", "Fmt", "Kind", "Draws", "Main", "Neutralized", "Repl", "Bind", "Registry"}) {
    ImGui::TableSetupColumn(header);
  }
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < report.keys.size(); ++i) {
    const KeyRow& row = report.keys[i];
    const std::string stem = key_text(row);
    if (!row_passes(row, snapshot, stem)) {
      continue;
    }
    const uint16_t registered = row.state == KeyState::Ready && snapshot != nullptr ? snapshot->find(row.key) : 0;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushID(static_cast<int>(i));
    const bool selected = g_selection.valid && row.state == KeyState::Ready && g_selection.key == row.key;
    if (ImGui::Selectable(stem.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
      select_row(row);
    }
    ImGui::PopID();
    ImGui::TableNextColumn();
    ImGui::Text("%ux%u", static_cast<unsigned>(row.width), static_cast<unsigned>(row.height));
    ImGui::TableNextColumn();
    ImGui::Text("%u", static_cast<unsigned>(row.format));
    ImGui::TableNextColumn();
    text(kind_name(row.kind));
    ImGui::TableNextColumn();
    ImGui::Text("%u", row.draws);
    ImGui::TableNextColumn();
    ImGui::Text("%u", row.mainDraws);
    ImGui::TableNextColumn();
    ImGui::Text("%u", row.neutralDraws);
    ImGui::TableNextColumn();
    text(row.pointerReplacement ? "pointer" : row.replaced ? "yes" : "");
    ImGui::TableNextColumn();
    ImGui::Text("%u/%08x%s", row.material.id, row.material.flags, row.stochastic ? " hex" : "");
    ImGui::TableNextColumn();
    if (registered != 0) {
      ImGui::Text("%u/%08x", registered, snapshot->record(registered).summary);
    }
  }
  ImGui::EndTable();
}

void draw_hover() {
  ImGui::SeparatorText("Picked draw");
  ImGui::Checkbox("Pin", &g_hover.pinned);
  const auto report = g_hover.valid ? report_for(static_cast<uint8_t>(g_hover.sequence)) : nullptr;
  if (report == nullptr || g_hover.record >= report->draws.size()) {
    ImGui::TextUnformatted(g_arm == pick::ArmState::Armed ? "Nothing under the cursor" : "Tick Pick to pick draws");
    return;
  }
  const DrawRecord& draw = report->draws[g_hover.record];
  ImGui::Text("Draw %u: %s, %s, object %llx%s%s", draw.index, std::string{scene::role_name(draw.role)}.c_str(),
              std::string{scene::window_name(draw.window)}.c_str(), static_cast<unsigned long long>(draw.objectId),
              draw.mainClass ? ", main" : "", draw.skipped ? ", skipped" : "");
  for (uint32_t i = 0; i < draw.slotCount && draw.slotBegin + i < report->slots.size(); ++i) {
    const SlotRecord& slot = report->slots[draw.slotBegin + i];
    const KeyRow* row = slot.row < report->keys.size() ? &report->keys[slot.row] : nullptr;
    const std::string label =
        fmt::format("tex{} {} {}x{}{}{}{}##slot{}", slot.texmap,
                    row != nullptr ? key_text(*row) : std::string{kind_name(slot.kind)}, slot.width, slot.height,
                    slot.indirect ? " ind" : "", slot.stochastic ? " hex" : "", slot.neutral ? " neutral" : "", i);
    if (ImGui::Selectable(label.c_str()) && row != nullptr) {
      select_row(*row);
    }
  }
}

std::vector<std::string_view> views_of(const std::vector<std::string>& values) {
  return {values.begin(), values.end()};
}

void draw_editor() {
  ImGui::SeparatorText("Selected key");
  if (!g_selection.valid) {
    ImGui::TextUnformatted("Select a keyed row or a picked slot");
    return;
  }
  const texture::TextureSourceKey& key = g_selection.key;
  const std::string stem = texture::format_material_key(key);
  text(stem);
  if (ImGui::Button("Copy key")) {
    ImGui::SetClipboardText(stem.c_str());
  }
  ImGui::SameLine();
  const std::string merged = texture::find_material_json(key, true);
  if (ImGui::Button("Copy entry JSON")) {
    ImGui::SetClipboardText(merged.c_str());
  }
  if (merged.empty()) {
    ImGui::TextUnformatted("No merged entry");
  } else {
    text(merged);
  }

  material::UserLayer& layer = user_layer();
  ImGui::SeparatorText("User layer");
  const std::string where =
      layer.path().empty() ? std::string{material::UserLayer::SessionLabel} : layer.path().string();
  ImGui::Text("%s (%s)%s", where.c_str(),
              layer.state() == material::UserLayerState::Loaded    ? "loaded"
              : layer.state() == material::UserLayerState::Missing ? "new file"
              : layer.state() == material::UserLayerState::Invalid ? "invalid; fix it or revert"
                                                                   : "session only",
              layer.dirty() ? ", edited" : "");
  ImGui::BeginDisabled(!layer.editable() || g_selection.pointerReplacement);
  const material::UserEntry entry = layer.read_entry(key);
  if (!entry.exists || g_selection.pointerReplacement) {
    if (ImGui::Button("Add by key")) {
      layer.add_entry(key);
    }
    if (g_selection.pointerReplacement) {
      ImGui::SameLine();
      ImGui::TextColored(WarningColor, "pointer replacement: no material lookup");
    }
  } else {
    std::vector<std::string_view> flags = views_of(entry.flags);
    bool flagsChanged = false;
    for (const std::string_view name : FlagNames) {
      bool on = std::ranges::find(flags, name) != flags.end();
      ImGui::SameLine();
      if (ImGui::Checkbox(std::string{name}.c_str(), &on)) {
        flagsChanged = true;
        if (on) {
          flags.push_back(name);
        } else {
          std::erase(flags, name);
        }
      }
    }
    if (flagsChanged) {
      layer.set_flags(key, flags);
    }
    ImGui::SetNextItemWidth(220.f);
    if (ImGui::InputText("Class", g_ui.className.data(), g_ui.className.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
      layer.set_class(key, g_ui.className.data());
    }
    for (size_t i = 0; i < material::StochasticFieldNames.size(); ++i) {
      const std::string name{material::StochasticFieldNames[i]};
      float value = static_cast<float>(entry.stochastic[i].value_or(StochasticDefault[i]));
      ImGui::SetNextItemWidth(220.f);
      if (ImGui::SliderFloat(name.c_str(), &value, StochasticMin[i], StochasticMax[i])) {
        layer.set_stochastic(key, name, value);
      }
      if (entry.stochastic[i]) {
        ImGui::SameLine();
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::SmallButton("Reset")) {
          layer.reset_field(key, "stochastic/" + name);
        }
        ImGui::PopID();
      }
    }
    int neutral = 0;
    int gate = 0;
    for (int i = 1; i < static_cast<int>(NeutralNames.size()); ++i) {
      neutral = entry.neutral == NeutralNames[i] ? i : neutral;
      gate = entry.gate == GateNames[i] ? i : gate;
    }
    std::vector<std::string_view> scope = views_of(entry.scope);
    bool shadowChanged = false;
    ImGui::SetNextItemWidth(120.f);
    shadowChanged |= ImGui::Combo("Neutral", &neutral, NeutralNames.data(), static_cast<int>(NeutralNames.size()));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.f);
    shadowChanged |= ImGui::Combo("Gate", &gate, GateNames.data(), static_cast<int>(GateNames.size()));
    for (size_t i = 0; i < ScopeNames.size(); ++i) {
      bool on = std::ranges::find(scope, ScopeNames[i]) != scope.end();
      if (i % 7 != 0) {
        ImGui::SameLine();
      }
      if (ImGui::Checkbox(std::string{ScopeNames[i]}.c_str(), &on)) {
        shadowChanged = true;
        if (on) {
          scope.push_back(ScopeNames[i]);
        } else {
          std::erase(scope, ScopeNames[i]);
        }
      }
    }
    if (shadowChanged) {
      layer.set_fake_shadow(key, NeutralNames[neutral], GateNames[gate], scope);
    }
    if (ImGui::Button("Remove entry")) {
      layer.remove_entry(key);
    }
  }
  ImGui::EndDisabled();
  ImGui::BeginDisabled(!layer.dirty() || !layer.editable());
  if (ImGui::Button("Apply")) {
    const material::UserApplyResult result = layer.apply();
    g_ui.message = result.applied ? std::string{"Applied"} : result.message;
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Revert")) {
    layer.revert();
    g_ui.message.clear();
  }
  if (!g_ui.message.empty()) {
    text(g_ui.message);
  }
}
} // namespace

void initialize() {
  register_tab(TabName, TexturesTabOrder, &draw_tab);
  pick::initialize();
}

void shutdown() noexcept {
  scene::set_observer_hooks(nullptr);
  g_installed = false;
  if (g_arm == pick::ArmState::Armed) {
    webgpu::g_debugIdBuffer = {};
    webgpu::g_graphicsConfig.debugIdBuffer = false;
  }
  g_arm = pick::ArmState::Off;
  pick::shutdown();
  g_ring = {};
  g_ringNext = 0;
  g_seen.reset();
  g_warm = {};
  g_hover = {};
  g_selection = {};
  g_ui = {};
  g_userLayer.reset();
  release_live();
}

void on_latch() noexcept {
  seal();
  const bool shown = visible();
  if (!shown && g_reserved) {
    release_live();
  } else if (shown) {
    if (g_memo.size() > KeyMemoCapacity * 3 / 4) {
      g_memo.clear();
    }
    reserve_live();
  }
  if (shown != g_installed) {
    scene::set_observer_hooks(shown ? &g_hooks : nullptr);
    g_installed = shown;
  }
  if (!shown && g_arm == pick::ArmState::Off) {
    return;
  }
  step_arm(shown);
  update_pick_params();
  resolve_readback();
}

void draw_tab() {
  if (!webgpu::g_device) {
    ImGui::TextUnformatted("No GPU device");
    return;
  }
  if (!hints_enabled()) {
    ImGui::TextUnformatted("Render hints are off, so there is no draw list");
    return;
  }
  const std::shared_ptr<const TexturesReport> report = latest_report();
  draw_status(report.get());
  const pick::Capability capability = pick::pick_capability(pick::live_capability_inputs());
  ImGui::BeginDisabled(!capability.available);
  ImGui::Checkbox("Pick", &g_pickEnabled);
  ImGui::EndDisabled();
  if (report != nullptr) {
    draw_key_table(*report);
  }
  draw_hover();
  draw_editor();
}

void set_pick_enabled(bool enabled) noexcept { g_pickEnabled = enabled; }

pick::ArmState arm_state() noexcept { return g_arm; }

std::shared_ptr<const TexturesReport> latest_report() noexcept {
  return g_ring[(g_ringNext + ReportRing - 1) % ReportRing];
}

std::shared_ptr<const TexturesReport> report_for(uint8_t sequence) noexcept {
  for (const auto& report : g_ring) {
    if (report != nullptr && static_cast<uint8_t>(report->sequence) == sequence) {
      return report;
    }
  }
  return nullptr;
}

uint32_t flagged_seen(const SeenSet& seen, const material::Snapshot* snapshot) noexcept {
  if (snapshot == nullptr) {
    return 0;
  }
  uint32_t count = 0;
  const size_t limit = std::min(seen.size(), snapshot->records.size());
  for (size_t id = 1; id < limit; ++id) {
    count += seen.test(id) && snapshot->live(static_cast<uint32_t>(id)) ? 1 : 0;
  }
  return count;
}

void observe(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
             const scene::DrawClass& cls) noexcept {
  Live& live = g_live;
  live.recorded = true;
  if (g_arm == pick::ArmState::Warming) {
    note_config(live, draw, config);
  }
  if (live.draws.size() >= MaxDraws) {
    ++live.overflow.draws;
    return;
  }
  const auto record = static_cast<uint32_t>(live.draws.size());
  DrawRecord& out = live.draws.emplace_back();
  out.index = cls.index;
  out.role = cls.key.role;
  out.window = cls.key.window;
  out.categories = cls.key.categories;
  out.objectId = cls.key.object.valid ? cls.key.object.id : 0;
  out.mainClass = cls.mainClass;
  out.inSpan = in_view_span(cls);
  out.slotBegin = static_cast<uint32_t>(live.slots.size());
  const fakeshadow::DrawEdit& edit = fakeshadow::last_edit();
  const uint8_t neutral = edit.applied ? static_cast<uint8_t>(edit.white | edit.clear | edit.skip) : 0;
  out.skipped = edit.applied && edit.edit == fakeshadow::Edit::Skipped;
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    if (live.slots.size() >= MaxSlots) {
      ++live.overflow.slots;
      continue;
    }
    record_slot(live, i, config, info, cls.mainClass, (neutral >> i & 1u) != 0);
    ++out.slotCount;
  }
  if (g_arm == pick::ArmState::Armed && out.inSpan && draw.instanceCount != 0) {
    draw.immediateData._pad = pick::encode_id(g_sequence, record);
  }
}

namespace testing {
void reset() noexcept {
  shutdown();
  g_forceVisible = -1;
  g_pickEnabled = false;
  g_hashBudget = HashBudgetBytes;
  g_pointerProbe = &default_pointer_probe;
  g_sequence = 1;
}

void force_visible(int value) noexcept { g_forceVisible = value; }

void set_hash_budget(uint64_t bytes) noexcept { g_hashBudget = bytes; }

void set_arm_state(pick::ArmState state) noexcept { g_arm = state; }

void set_pointer_probe(bool (*probe)(const GXTexObj_& obj) noexcept) noexcept {
  g_pointerProbe = probe != nullptr ? probe : &default_pointer_probe;
}

void set_hover_pinned(bool pinned) noexcept { g_hover.pinned = pinned; }

bool observer_installed() noexcept { return g_installed; }

size_t memo_size() noexcept { return g_memo.size(); }

uint64_t sequence() noexcept { return g_sequence; }

Capacities live_capacities() noexcept {
  return {.draws = g_live.draws.capacity(), .slots = g_live.slots.capacity(), .keys = g_live.keys.capacity()};
}

HoverState hover() noexcept { return {.sequence = g_hover.sequence, .record = g_hover.record, .valid = g_hover.valid}; }
} // namespace testing
} // namespace aurora::render::debug::textures

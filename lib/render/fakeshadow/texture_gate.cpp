#include "texture_gate.hpp"

#include "../frame_state.hpp"
#include "../hints.hpp"
#include "../material/registry.hpp"
#include "../scene/classifier.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>
#include <aurora/texture.hpp>

#include <atomic>
#include <mutex>
#include <utility>

namespace aurora::render::fakeshadow {
namespace {
constexpr Module Log{"aurora::render::fakeshadow"};

constexpr uint16_t BaseCategories = 0x1FFF;
constexpr size_t MemoSize = 16;
constexpr std::array<std::string_view, GateCount> GateNames{"light", "cloud", "ao"};

static_assert(scene::category::Static == 1u << 0 && scene::category::Overlay == 1u << 12);
static_assert((scene::category::FakeShadow & BaseCategories) == 0);
static_assert(static_cast<uint32_t>(Neutral::White) == 1 && static_cast<uint32_t>(Neutral::Skip) == 3);

struct MemoEntry {
  gfx::BindGroupRef base = 0;
  uint8_t white = 0;
  uint8_t clear = 0;
  gfx::BindGroupRef group = 0;
};

struct Total {
  std::string key;
  OutcomeCounts counts{};
};

struct PublishState {
  std::array<GateView, GateCount> gates{};
  bool active = false;
  bool observing = false;
  bool hintsOff = false;
  const RuleTable* table = nullptr;

  bool operator==(const PublishState&) const = default;
};

const std::shared_ptr<const RuleTable>& empty_table() noexcept {
  static const std::shared_ptr<const RuleTable> table = std::make_shared<const RuleTable>();
  return table;
}

std::array<std::atomic<Force>, GateCount> g_forces{};
std::atomic<bool> g_statsEnabled{false};

std::array<GateSource, GateCount> g_sources{};
uint8_t g_envGates = 0;
bool g_testRules = false;
testing::NeutralBuilder g_builder = &gx::build_neutral_bind_groups;
std::shared_ptr<const RuleTable> g_table = empty_table();
std::vector<Rule> g_rules;
std::vector<uint16_t> g_ruleById;
std::vector<bool> g_warnedUnscoped;
std::vector<Total> g_totals;
PublishState g_published;

std::array<GateView, GateCount> g_gates{};
uint8_t g_gateBits = 0;
bool g_counting = false;
bool g_active = false;
bool g_hintsOff = false;

std::vector<OutcomeCounts> g_counts;
bool g_leadFlagged = false;
bool g_folded = false;
DrawEdit g_lastEdit;
std::array<MemoEntry, MemoSize> g_memo{};
size_t g_memoSize = 0;

std::mutex g_reportMutex;
std::shared_ptr<const TextureReport> g_report;

constexpr bool in_scope(Outcome outcome) noexcept {
  switch (outcome) {
  case Outcome::White:
  case Outcome::Clear:
  case Outcome::Skipped:
  case Outcome::GateOff:
  case Outcome::RefusedCoverage:
  case Outcome::RefusedWhite:
  case Outcome::RefusedClear:
  case Outcome::RefusedWrites:
  case Outcome::RefusedShared:
    return true;
  default:
    return false;
  }
}

std::string gate_names(uint8_t mask) {
  std::string names;
  for (size_t i = 0; i < GateCount; ++i) {
    if ((mask >> i & 1u) != 0) {
      if (!names.empty()) {
        names += ',';
      }
      names += GateNames[i];
    }
  }
  return names;
}

void install_rules(std::shared_ptr<const RuleTable> table, std::vector<Rule> rules, std::vector<uint16_t> byId) {
  g_table = std::move(table);
  g_rules = std::move(rules);
  g_ruleById = g_rules.empty() ? std::vector<uint16_t>{} : std::move(byId);
  g_counts.assign(g_rules.size(), OutcomeCounts{});
}

void rebuild_rules(const material::Snapshot& snapshot) {
  auto table = std::make_shared<RuleTable>();
  table->generation = snapshot.generation;
  std::vector<Rule> rules;
  std::vector<uint16_t> byId(snapshot.records.size(), 0);
  for (uint32_t id = 1; id < snapshot.records.size(); ++id) {
    if (!snapshot.live(id)) {
      continue;
    }
    const material::MaterialRecord& record = snapshot.record(id);
    const uint32_t neutral = material::shadow_neutral(record.summary);
    const uint32_t gate = material::shadow_gate(record.summary);
    const texture::TextureSourceKey* key = snapshot.key(id);
    if (key == nullptr || (record.summary & material::FakeShadow) == 0 || neutral == 0 || gate == 0) {
      continue;
    }
    const bool scoped = (record.summary & material::ShadowScope) != 0;
    const Rule rule{
        .materialId = static_cast<uint16_t>(id),
        .neutral = static_cast<Neutral>(neutral),
        .gateBit = static_cast<uint8_t>(1u << (gate - 1)),
        .scopeMask = scoped ? static_cast<uint16_t>(record.shadowScope & BaseCategories) : BaseCategories,
        .scoped = scoped,
    };
    std::string name = texture::format_material_key(*key);
    if (!scoped) {
      ++table->unscoped;
      if (g_warnedUnscoped.size() <= id) {
        g_warnedUnscoped.resize(id + 1, false);
      }
      if (!g_warnedUnscoped[id]) {
        g_warnedUnscoped[id] = true;
        Log.warn("Fake shadow material {} has no scope, so it applies to every base category", name);
      }
    }
    byId[id] = static_cast<uint16_t>(rules.size() + 1);
    rules.push_back(rule);
    table->rules.push_back({rule, std::move(name)});
  }
  install_rules(std::move(table), std::move(rules), std::move(byId));
}

void fold() noexcept {
  if (std::exchange(g_folded, true)) {
    return;
  }
  for (size_t index = 0; index < g_rules.size() && index < g_counts.size(); ++index) {
    const uint16_t id = g_rules[index].materialId;
    if (g_totals.size() <= id) {
      g_totals.resize(id + 1u);
    }
    Total& total = g_totals[id];
    if (total.key.empty() && index < g_table->rules.size()) {
      total.key = g_table->rules[index].key;
    }
    for (size_t outcome = 0; outcome < OutcomeCount; ++outcome) {
      total.counts[outcome] += g_counts[index][outcome];
    }
  }
}

void log_totals() noexcept {
  for (const Total& total : g_totals) {
    if (total.key.empty()) {
      continue;
    }
    const OutcomeCounts& c = total.counts;
    Log.info(
        "Fake shadow {}: bound {} inScope {} white {} clear {} skipped {} gateOff {} outOfScope {} outOfRole {} "
        "entryOwned {} refusedIndirect {} refusedCoverage {} refusedWhite {} refusedClear {} refusedWrites {} "
        "refusedShared {}",
        total.key, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14]);
  }
}

void clear_frame() noexcept {
  for (OutcomeCounts& counts : g_counts) {
    counts.fill(0);
  }
  g_leadFlagged = false;
  g_folded = false;
  g_lastEdit = {};
  g_memoSize = 0;
}

gx::GXBindGroups neutral_groups(const gx::GXBindGroups& base, const gx::ShaderInfo& info,
                                const Decision& decision) noexcept {
  for (size_t i = 0; i < g_memoSize; ++i) {
    const MemoEntry& entry = g_memo[i];
    if (entry.base == base.textureBindGroup && entry.white == decision.white && entry.clear == decision.clear) {
      return {.textureBindGroup = entry.group};
    }
  }
  const bool baseLevel = gx::g_gxState.proj[3][3] != 0.f || static_cast<bool>(gx::g_gxState.alphaCompare);
  const gx::GXBindGroups groups = g_builder(info, decision.white, decision.clear, baseLevel);
  if (g_memoSize < MemoSize) {
    g_memo[g_memoSize++] = {
        .base = base.textureBindGroup,
        .white = decision.white,
        .clear = decision.clear,
        .group = groups.textureBindGroup,
    };
  }
  return groups;
}
} // namespace

void install() noexcept {
  const char* value = SDL_getenv("AURORA_RENDER_FAKE_SHADOW_GATES");
  if (value == nullptr) {
    return;
  }
  const testing::GateList list = testing::parse_gate_list(value);
  if (list.unknown) {
    Log.warn("AURORA_RENDER_FAKE_SHADOW_GATES={}: unknown gate names ignored", value);
  }
  if (list.mask == 0) {
    return;
  }
  g_envGates = list.mask;
  for (size_t i = 0; i < GateCount; ++i) {
    if ((list.mask >> i & 1u) != 0) {
      g_forces[i].store(Force::On, std::memory_order_relaxed);
    }
  }
  Log.info("Fake shadow gates {} forced on by AURORA_RENDER_FAKE_SHADOW_GATES", gate_names(list.mask));
}

void remove() noexcept {
  if (g_envGates != 0) {
    fold();
    log_totals();
  }
  for (auto& force : g_forces) {
    force.store(Force::Auto, std::memory_order_relaxed);
  }
  g_statsEnabled.store(false, std::memory_order_relaxed);
  g_envGates = 0;
  g_totals.clear();
  g_counting = false;
  g_active = false;
  clear_frame();
}

void begin_frame(const detail::FrameState& state) noexcept {
  if (!g_testRules) {
    if (const material::Snapshot* latched = material::latched_raw();
        latched != nullptr && latched->generation != g_table->generation) {
      rebuild_rules(*material::latched_snapshot());
    }
  }
  g_gateBits = 0;
  for (size_t i = 0; i < GateCount; ++i) {
    const GateSource source = g_sources[i];
    const bool live = source != nullptr && source(state) && state.settings.shadows.game != GameShadows::Keep;
    const Force force = g_forces[i].load(std::memory_order_relaxed);
    const bool effective = force == Force::On || (force == Force::Auto && live);
    g_gates[i] = {.live = live, .effective = effective, .force = force};
    g_gateBits |= static_cast<uint8_t>(effective ? 1u << i : 0u);
  }
  g_hintsOff = !detail::hints_enabled();
  g_counting = !g_rules.empty() && !g_hintsOff && state.settings.textureMaterials &&
               (g_gateBits != 0 || g_statsEnabled.load(std::memory_order_relaxed));
  g_active = g_counting && g_gateBits != 0;
  clear_frame();
}

void seal() noexcept {
  if (g_envGates != 0) {
    fold();
  }
  const PublishState state{
      .gates = g_gates,
      .active = g_active,
      .observing = g_counting && !g_active,
      .hintsOff = g_hintsOff,
      .table = g_table.get(),
  };
  if (!g_counting && state == g_published) {
    return;
  }
  g_published = state;
  auto report = std::make_shared<TextureReport>();
  report->gates = state.gates;
  report->active = state.active;
  report->observing = state.observing;
  report->hintsOff = state.hintsOff;
  report->table = g_table;
  report->counts = g_counts;
  std::shared_ptr<const TextureReport> published = std::move(report);
  std::lock_guard lock{g_reportMutex};
  g_report.swap(published);
}

void set_gate_source(Gate gate, GateSource source) noexcept { g_sources[static_cast<size_t>(gate)] = source; }

void reset_frame() noexcept {
  if (g_envGates != 0) {
    fold();
  }
  clear_frame();
}

void set_force(Gate gate, Force force) noexcept {
  g_forces[static_cast<size_t>(gate)].store(force, std::memory_order_relaxed);
}

Force force(Gate gate) noexcept { return g_forces[static_cast<size_t>(gate)].load(std::memory_order_relaxed); }

void set_stats_enabled(bool enabled) noexcept { g_statsEnabled.store(enabled, std::memory_order_relaxed); }

std::shared_ptr<const TextureReport> latest_report() noexcept {
  std::lock_guard lock{g_reportMutex};
  return g_report;
}

Edit push(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
          const scene::DrawClass& cls) noexcept {
  if (!g_counting) {
    return Edit::Untouched;
  }
  g_lastEdit = {};
  g_leadFlagged = false;
  std::array<Slot, gx::MaxTextures> slots{};
  size_t count = 0;
  uint8_t bound = 0;
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    const gfx::TextureBind& bind = gx::g_gxState.textures[i];
    if (!bind || (!info.sampledTextures.test(i) && !info.sampledIndTextures.test(i))) {
      continue;
    }
    bound |= static_cast<uint8_t>(1u << i);
    const uint32_t id = bind.material.id;
    if (id != 0 && id < g_ruleById.size() && g_ruleById[id] != 0) {
      slots[count++] = {
          .index = static_cast<uint8_t>(i),
          .rule = static_cast<uint16_t>(g_ruleById[id] - 1),
          .indirect = info.sampledIndTextures.test(i),
      };
    }
  }
  if (count == 0) {
    return Edit::Untouched;
  }
  DrawFacts drawFacts = facts(config);
  drawFacts.boundMask = bound;
  std::array<Outcome, gx::MaxTextures> results{};
  const Decision decision = decide({slots.data(), count}, cls, drawFacts, g_gateBits, g_rules, {results.data(), count});
  for (size_t i = 0; i < count; ++i) {
    OutcomeCounts& counts = g_counts[slots[i].rule];
    ++counts[static_cast<size_t>(Outcome::Bound)];
    if (in_scope(results[i])) {
      ++counts[static_cast<size_t>(Outcome::InScope)];
    }
    ++counts[static_cast<size_t>(results[i])];
    g_lastEdit.flagged |= static_cast<uint8_t>(1u << slots[i].index);
  }
  g_lastEdit.white = decision.white;
  g_lastEdit.clear = decision.clear;
  g_lastEdit.skip = decision.skip;
  if (!g_active) {
    return Edit::Untouched;
  }
  if (decision.skip != 0) {
    // An empty command draws nothing and is never a merge target
    draw.instanceCount = 0;
    g_lastEdit.edit = Edit::Skipped;
    g_lastEdit.applied = true;
    return Edit::Skipped;
  }
  if ((decision.white | decision.clear) != 0) {
    draw.bindGroups = neutral_groups(draw.bindGroups, info, decision);
    g_lastEdit.edit = Edit::Substituted;
    g_lastEdit.applied = true;
    return Edit::Substituted;
  }
  g_leadFlagged = true;
  return Edit::Untouched;
}

bool lead_flagged() noexcept { return g_leadFlagged; }

const DrawEdit& last_edit() noexcept { return g_lastEdit; }

DrawFacts facts(const gx::PipelineConfig& config) noexcept {
  const auto alphaFactor = [](GXBlendFactor factor) { return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA; };
  const bool blend = config.blendMode == GX_BM_BLEND;
  DrawFacts out{
      .writes = (config.depthCompare && config.depthUpdate) || config.alphaUpdate,
      .alphaConsumed = (blend && (alphaFactor(config.blendFacSrc) || alphaFactor(config.blendFacDst))) ||
                       (config.alphaUpdate && config.dstAlpha == UINT32_MAX),
      .alphaTest = static_cast<bool>(config.shaderConfig.alphaCompare),
      .srcAlphaBlend = blend && config.blendFacSrc == GX_BL_SRCALPHA &&
                       (config.blendFacDst == GX_BL_INVSRCALPHA || config.blendFacDst == GX_BL_ONE),
  };
  const gx::ShaderConfig& shader = config.shaderConfig;
  for (uint32_t i = 0; i < shader.tevStageCount && i < gx::MaxTevStages; ++i) {
    const gx::TevStage& stage = shader.tevStages[i];
    const auto& pass = stage.alphaPass;
    if (stage.texMapId < gx::MaxTextures &&
        (pass.a == GX_CA_TEXA || pass.b == GX_CA_TEXA || pass.c == GX_CA_TEXA || pass.d == GX_CA_TEXA)) {
      // White forces this alpha to 1, which turns texture-shaped coverage into a solid overlay
      out.texAlphaMask |= static_cast<uint8_t>(1u << stage.texMapId);
    }
  }
  return out;
}

Decision decide(std::span<const Slot> slots, const scene::DrawClass& cls, const DrawFacts& drawFacts, uint8_t gates,
                std::span<const Rule> rules, std::span<Outcome> results) noexcept {
  uint8_t white = 0;
  uint8_t clear = 0;
  uint8_t skip = 0;
  for (size_t i = 0; i < slots.size(); ++i) {
    const Slot& slot = slots[i];
    const Rule& rule = rules[slot.rule];
    const auto bit = static_cast<uint8_t>(1u << slot.index);
    Outcome& result = results[i];
    if (slot.indirect) {
      result = Outcome::RefusedIndirect;
    } else if (cls.key.role != scene::Role::Main || cls.offscreen) {
      result = Outcome::OutOfRole;
    } else if ((cls.key.categories & scene::category::FakeShadow) != 0) {
      result = Outcome::EntryOwned;
    } else if ((cls.key.categories & rule.scopeMask) == 0) {
      result = Outcome::OutOfScope;
    } else if ((gates & rule.gateBit) == 0) {
      result = Outcome::GateOff;
    } else if (rule.neutral == Neutral::White) {
      if (drawFacts.alphaTest) {
        result = Outcome::RefusedCoverage;
      } else if ((drawFacts.texAlphaMask & bit) != 0 && drawFacts.alphaConsumed) {
        result = Outcome::RefusedWhite;
      } else {
        result = Outcome::White;
        white |= bit;
      }
    } else if (rule.neutral == Neutral::Clear) {
      if (drawFacts.alphaTest) {
        result = Outcome::RefusedCoverage;
      } else if (!drawFacts.srcAlphaBlend) {
        result = Outcome::RefusedClear;
      } else {
        result = Outcome::Clear;
        clear |= bit;
      }
    } else if (rule.neutral == Neutral::Skip) {
      if (drawFacts.writes) {
        result = Outcome::RefusedWrites;
      } else {
        result = Outcome::Skipped;
        skip |= bit;
      }
    } else {
      result = Outcome::GateOff;
    }
  }
  if (skip == 0) {
    return {.white = white, .clear = clear};
  }
  // Skip removes the whole draw, so every other bound texture must be neutralised too
  const auto kept = static_cast<uint8_t>(drawFacts.boundMask & ~(white | clear | skip));
  for (size_t i = 0; i < slots.size(); ++i) {
    Outcome& result = results[i];
    if (kept == 0 && (result == Outcome::White || result == Outcome::Clear)) {
      result = Outcome::Skipped;
    } else if (kept != 0 && result == Outcome::Skipped) {
      result = Outcome::RefusedShared;
    }
  }
  if (kept == 0) {
    return {.skip = static_cast<uint8_t>(white | clear | skip)};
  }
  return {.white = white, .clear = clear};
}

namespace testing {
void set_rules(std::span<const Rule> rules) noexcept {
  auto table = std::make_shared<RuleTable>();
  std::vector<uint16_t> byId;
  for (const Rule& rule : rules) {
    if (byId.size() <= rule.materialId) {
      byId.resize(rule.materialId + 1u, 0);
    }
    byId[rule.materialId] = static_cast<uint16_t>(table->rules.size() + 1);
    table->unscoped += rule.scoped ? 0 : 1;
    table->rules.push_back({rule, {}});
  }
  g_testRules = true;
  install_rules(std::move(table), {rules.begin(), rules.end()}, std::move(byId));
}

void set_neutral_builder(NeutralBuilder builder) noexcept {
  g_builder = builder != nullptr ? builder : &gx::build_neutral_bind_groups;
}

GateList parse_gate_list(std::string_view text) noexcept {
  GateList list;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view name = text.substr(0, comma);
    text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    if (name.empty() || name == "0") {
      continue;
    }
    if (name == "all") {
      list.mask |= (1u << GateCount) - 1;
      continue;
    }
    bool known = false;
    for (size_t i = 0; i < GateCount; ++i) {
      if (name == GateNames[i]) {
        list.mask |= static_cast<uint8_t>(1u << i);
        known = true;
      }
    }
    list.unknown |= !known;
  }
  return list;
}

void reset() noexcept {
  for (auto& force : g_forces) {
    force.store(Force::Auto, std::memory_order_relaxed);
  }
  g_statsEnabled.store(false, std::memory_order_relaxed);
  g_sources = {};
  g_envGates = 0;
  g_testRules = false;
  g_builder = &gx::build_neutral_bind_groups;
  install_rules(empty_table(), {}, {});
  g_warnedUnscoped.clear();
  g_totals.clear();
  g_published = {};
  g_gates = {};
  g_gateBits = 0;
  g_counting = false;
  g_active = false;
  g_hintsOff = false;
  clear_frame();
  std::shared_ptr<const TextureReport> previous;
  std::lock_guard lock{g_reportMutex};
  g_report.swap(previous);
}
} // namespace testing
} // namespace aurora::render::fakeshadow

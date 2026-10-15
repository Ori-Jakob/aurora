#include "graph.hpp"

#include "../gfx/hash.hpp"
#include "../logging.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render"};

struct Slot {
  std::optional<CompiledPass> pass;
  uint32_t generation = 1;
};

std::mutex g_passMutex;
std::vector<Slot> g_passSlots;
std::vector<uint32_t> g_freePassSlots;
uint64_t g_nextSequence = 1;
std::atomic_bool g_graphDirty = false;

constexpr uint32_t pass_index(PassId id) noexcept { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }
constexpr uint32_t pass_generation(PassId id) noexcept { return static_cast<uint32_t>(id >> 32); }
constexpr PassId make_pass_id(uint32_t index, uint32_t generation) noexcept {
  return static_cast<PassId>(generation) << 32 | index;
}

// Requires g_passMutex held.
Slot* find_pass_slot_locked(PassId id) noexcept {
  const auto index = pass_index(id);
  if (id == InvalidPassId || index >= g_passSlots.size()) {
    return nullptr;
  }
  auto& slot = g_passSlots[index];
  if (slot.generation != pass_generation(id) || !slot.pass) {
    return nullptr;
  }
  return &slot;
}

// Requires g_passMutex held.
bool has_live_pass_named(const std::string& name) noexcept {
  return std::ranges::any_of(g_passSlots, [&](const Slot& slot) { return slot.pass && slot.pass->desc.name == name; });
}
} // namespace

PassId register_pass(const PassDesc& desc) {
  if (desc.name.empty()) {
    Log.warn("register_pass: pass name is empty");
    return InvalidPassId;
  }
  if (desc.encode == nullptr) {
    Log.warn("register_pass: '{}' has no encode callback", desc.name);
    return InvalidPassId;
  }
  if (desc.kind == PassKind::Native) {
    Log.warn("register_pass: '{}' rejected; native passes need the D3D12 backend", desc.name);
    return InvalidPassId;
  }
  if (desc.stage == Stage::AfterFrame) {
    Log.warn("register_pass: '{}' rejected; AfterFrame runs native passes only", desc.name);
    return InvalidPassId;
  }
  if (static_cast<size_t>(desc.stage) >= StageCount) {
    Log.warn("register_pass: '{}' has unknown stage {}", desc.name, static_cast<uint32_t>(desc.stage));
    return InvalidPassId;
  }

  std::lock_guard lock{g_passMutex};
  if (has_live_pass_named(desc.name)) {
    Log.warn("register_pass: a pass named '{}' is already registered", desc.name);
    return InvalidPassId;
  }
  uint32_t index;
  if (!g_freePassSlots.empty()) {
    index = g_freePassSlots.back();
    g_freePassSlots.pop_back();
  } else {
    index = static_cast<uint32_t>(g_passSlots.size());
    g_passSlots.emplace_back();
  }
  auto& slot = g_passSlots[index];
  const PassId id = make_pass_id(index, slot.generation);
  slot.pass = CompiledPass{.desc = desc, .id = id, .sequence = g_nextSequence++};
  g_graphDirty = true;
  return id;
}

void unregister_pass(PassId id) noexcept {
  std::lock_guard lock{g_passMutex};
  auto* slot = find_pass_slot_locked(id);
  if (slot == nullptr) {
    return;
  }
  slot->pass.reset();
  ++slot->generation;
  g_freePassSlots.push_back(pass_index(id));
  g_graphDirty = true;
}

bool consume_graph_dirty() noexcept { return g_graphDirty.exchange(false); }

std::shared_ptr<const CompiledGraph> compile_graph(const Settings&, const Caps&) {
  std::vector<CompiledPass> passes;
  {
    std::lock_guard lock{g_passMutex};
    for (const auto& slot : g_passSlots) {
      if (slot.pass) {
        passes.push_back(*slot.pass);
      }
    }
  }
  if (passes.empty()) {
    return empty_graph();
  }
  std::ranges::stable_sort(passes, [](const CompiledPass& a, const CompiledPass& b) {
    return std::tie(a.desc.stage, a.desc.order, a.sequence) < std::tie(b.desc.stage, b.desc.order, b.sequence);
  });

  auto graph = std::make_shared<CompiledGraph>();
  Hasher hasher{};
  for (uint32_t i = 0; i < passes.size(); ++i) {
    const auto& desc = passes[i].desc;
    auto& range = graph->stages[static_cast<size_t>(desc.stage)];
    if (range.count == 0) {
      range.first = i;
    }
    ++range.count;
    hasher.update(static_cast<uint8_t>(desc.stage));
    hasher.update(static_cast<uint8_t>(desc.kind));
    hasher.update(static_cast<uint8_t>(desc.cls));
    hasher.update(desc.order);
    hasher.update(static_cast<uint8_t>(desc.runsWithoutMarker));
    hasher.update(static_cast<uint32_t>(desc.name.size()));
    hasher.update(desc.name.data(), desc.name.size());
  }
  graph->graphHash = hasher.digest();
  graph->passes = std::move(passes);
  return graph;
}

namespace testing {
void clear_passes() noexcept {
  std::lock_guard lock{g_passMutex};
  g_passSlots.clear();
  g_freePassSlots.clear();
  g_nextSequence = 1;
  g_graphDirty = false;
}
} // namespace testing
} // namespace aurora::render::detail

#include "graph_api.hpp"

#include "graph.hpp"

#include "../logging.hpp"

#include <aurora/render_graph.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

namespace aurora::render {
namespace {
constexpr Module Log{"aurora::render"};
} // namespace

namespace detail {
namespace {
constexpr size_t MaxExternalNameLength = 63;

// Frozen in v1 and append-only; every internal dotted name must use one of these
constexpr std::array<std::string_view, 38> ReservedPrefixes{
    "aurora",    "gx",    "phase",   "gbuf",     "env",   "grass", "sky",    "clouds",   "terrain", "gi",
    "ddgi",      "scene", "capture", "rt",       "hzb",   "vg",    "legacy", "depth",    "light",   "ao",
    "ssil",      "csm",   "vsm",     "sun",      "vis",   "lsm",   "refl",   "exposure", "godrays", "fog",
    "composite", "taa",   "native",  "radiance", "debug", "bvh",   "voxel",  "heap",
};

constexpr bool name_char(char c) noexcept { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; }

constexpr uint32_t slot_index(PassId id) noexcept { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }

constexpr bool indexable(PassId id) noexcept { return id != InvalidPass && slot_index(id) < MaxExternalPasses; }

constexpr PassId make_id(uint32_t index, uint32_t generation) noexcept {
  return static_cast<PassId>(generation) << 32 | index;
}

constexpr uint32_t next_generation(uint32_t generation) noexcept {
  return generation == UINT32_MAX ? 1 : generation + 1;
}

constexpr std::array<uint32_t, MaxExternalPasses> FirstGenerations = [] {
  std::array<uint32_t, MaxExternalPasses> generations{};
  generations.fill(1);
  return generations;
}();

struct Registry {
  std::mutex mutex;
  std::array<std::optional<ExternalPassRecord>, MaxExternalPasses> records;
  std::array<uint32_t, MaxExternalPasses> generations = FirstGenerations;
  std::vector<uint32_t> freeSlots;
  uint32_t usedSlots = 0;
  uint64_t nextSequence = 1;
  std::string untraceable;
};

Registry g_registry;
std::array<std::atomic<PassId>, MaxExternalPasses> g_live{};
std::atomic<uint64_t> g_epoch{0};
// Only the latch writes these, while the recording thread is idle, so recording reads them without the lock
std::array<LatchedExternalPass, MaxExternalPasses> g_latched{};
std::vector<LatchedExternalPass> g_latchedOrder;

std::mutex g_logMutex;
std::unordered_set<std::string> g_logged;

// Requires g_registry.mutex held.
ExternalPassRecord* find_record_locked(PassId id) noexcept {
  if (!indexable(id)) {
    return nullptr;
  }
  auto& record = g_registry.records[slot_index(id)];
  return record && record->id == id ? &*record : nullptr;
}

// Requires g_registry.mutex held.
void clear_registry_locked() noexcept {
  Registry& registry = g_registry;
  for (uint32_t i = 0; i < MaxExternalPasses; ++i) {
    g_live[i].store(InvalidPass, std::memory_order_release);
    if (registry.records[i]) {
      registry.records[i].reset();
      // A stale id from before the clear must never match the slot again
      registry.generations[i] = next_generation(registry.generations[i]);
    }
  }
  registry.freeSlots.clear();
  registry.usedSlots = 0;
  registry.untraceable.clear();
  g_latched.fill({});
  g_latchedOrder.clear();
  g_epoch.fetch_add(1, std::memory_order_acq_rel);
}
} // namespace

bool valid_external_name(std::string_view name) noexcept {
  if (name.empty() || name.size() > MaxExternalNameLength || name.front() < 'a' || name.front() > 'z') {
    return false;
  }
  bool dotted = false;
  size_t segment = 0;
  for (const char c : name) {
    if (c == '.') {
      if (segment == 0) {
        return false;
      }
      dotted = true;
      segment = 0;
      continue;
    }
    if (!name_char(c)) {
      return false;
    }
    ++segment;
  }
  return dotted && segment != 0;
}

std::string_view external_prefix(std::string_view name) noexcept { return name.substr(0, name.find('.')); }

bool reserved_prefix(std::string_view prefix) noexcept {
  return std::ranges::find(ReservedPrefixes, prefix) != ReservedPrefixes.end();
}

uint64_t external_pass_epoch() noexcept { return g_epoch.load(std::memory_order_acquire); }

std::shared_ptr<const ExternalPassSet> snapshot_external_passes() {
  auto set = std::make_shared<ExternalPassSet>();
  std::lock_guard lock{g_registry.mutex};
  set->epoch = g_epoch.load(std::memory_order_acquire);
  for (const auto& record : g_registry.records) {
    if (record) {
      set->nodes.push_back({
          .name = record->name,
          .prefix = record->prefix,
          .desc = record->desc,
          .id = record->id,
          .sequence = record->sequence,
      });
    }
  }
  std::ranges::sort(set->nodes, {}, &ExternalNodeState::sequence);
  return set;
}

void append_external_passes(const ExternalPassSet& set, PassEncodeFn encode, std::vector<PassDesc>& out) {
  for (const ExternalNodeState& node : set.nodes) {
    out.push_back({
        .name = node.name,
        .stage = node.desc.stage,
        .kind = node.desc.kind,
        .cls = PassClass::Optional,
        .order = node.desc.order,
        .runsWithoutMarker = true,
        .paramsSize = node.desc.paramsSize,
        .external = true,
        .declare = &external_declare,
        .enabled = &external_enabled,
        .encode = encode,
        .state = &node,
    });
  }
}

std::span<const LatchedExternalPass> latch_external_passes() {
  std::vector<std::pair<uint64_t, uint32_t>> order;
  std::vector<std::string> names;
  {
    std::lock_guard lock{g_registry.mutex};
    for (uint32_t i = 0; i < MaxExternalPasses; ++i) {
      const auto& record = g_registry.records[i];
      if (!record) {
        g_latched[i] = {};
        continue;
      }
      g_latched[i] = {
          .id = record->id,
          .enabledFn = record->desc.enabled,
          .userdata = record->desc.userdata,
          .paramsSize = record->desc.paramsSize,
          .params = record->pending,
      };
      order.emplace_back(record->sequence, i);
    }
    std::ranges::sort(order);
    names.reserve(order.size());
    for (const auto& [sequence, slot] : order) {
      names.push_back(g_registry.records[slot]->name);
    }
  }
  // Runs without the lock, so a callback may set params for the next frame
  g_latchedOrder.clear();
  std::string untraceable;
  for (size_t i = 0; i < order.size(); ++i) {
    LatchedExternalPass& latched = g_latched[order[i].second];
    // An earlier callback may have unregistered this pass, and it must not be called after that returned
    if (!external_pass_alive(latched.id)) {
      latched.enabled = false;
      g_latchedOrder.push_back(latched);
      continue;
    }
    latched.enabled = latched.enabledFn == nullptr ||
                      latched.enabledFn(std::span{latched.params}.first(latched.paramsSize), latched.userdata);
    if (latched.enabled && untraceable.empty()) {
      untraceable = names[i];
    }
    g_latchedOrder.push_back(latched);
  }
  std::lock_guard lock{g_registry.mutex};
  g_registry.untraceable = std::move(untraceable);
  return g_latchedOrder;
}

const LatchedExternalPass* latched_external_pass(PassId id) noexcept {
  if (!indexable(id)) {
    return nullptr;
  }
  const LatchedExternalPass& latched = g_latched[slot_index(id)];
  return latched.id == id ? &latched : nullptr;
}

bool external_pass_alive(PassId id) noexcept {
  return indexable(id) && g_live[slot_index(id)].load(std::memory_order_acquire) == id;
}

bool external_enabled(const StageData&, const void* state) {
  const auto& node = *static_cast<const ExternalNodeState*>(state);
  const LatchedExternalPass* latched = latched_external_pass(node.id);
  return latched != nullptr && latched->enabled && external_pass_alive(node.id);
}

std::string untraceable_external_pass() {
  std::lock_guard lock{g_registry.mutex};
  return g_registry.untraceable;
}

bool log_once(std::string_view pass, std::string_view reason, std::string_view detail) {
  std::string key;
  key.reserve(pass.size() + reason.size() + detail.size() + 2);
  key.append(pass).append(1, '\x1f').append(reason).append(1, '\x1f').append(detail);
  std::lock_guard lock{g_logMutex};
  return g_logged.insert(std::move(key)).second;
}

void shutdown_external_passes() noexcept {
  std::lock_guard lock{g_registry.mutex};
  clear_registry_locked();
}

namespace testing {
void reset_external_passes() noexcept {
  {
    std::lock_guard lock{g_registry.mutex};
    clear_registry_locked();
    g_registry.generations = FirstGenerations;
    g_registry.nextSequence = 1;
  }
  std::lock_guard lock{g_logMutex};
  g_logged.clear();
}
} // namespace testing
} // namespace detail

PassId register_pass(const ExternalPassDesc& desc) {
  const auto refuse = [](const std::string& reason) {
    Log.warn("register_pass: {}", reason);
    return InvalidPass;
  };
  if (desc.name == nullptr) {
    return refuse("pass name is null");
  }
  const std::string_view name{desc.name};
  if (!detail::valid_external_name(name)) {
    return refuse(fmt::format("'{}' does not follow the external name rules", name));
  }
  const std::string_view prefix = detail::external_prefix(name);
  if (detail::reserved_prefix(prefix)) {
    return refuse(fmt::format("'{}' uses reserved prefix '{}'", name, prefix));
  }
  if (desc.stage == Stage::AfterFrame) {
    return refuse(fmt::format("'{}' cannot run in AfterFrame, which runs native passes only", name));
  }
  if (static_cast<size_t>(desc.stage) >= StageCount) {
    return refuse(fmt::format("'{}' has unknown stage {}", name, static_cast<uint32_t>(desc.stage)));
  }
  if (desc.kind == PassKind::Native) {
    return refuse(fmt::format("'{}' is a native pass, which external passes cannot be", name));
  }
  if (desc.kind > PassKind::Native) {
    return refuse(fmt::format("'{}' has unknown kind {}", name, static_cast<uint32_t>(desc.kind)));
  }
  if (desc.declare == nullptr) {
    return refuse(fmt::format("'{}' has no declare callback", name));
  }
  if (desc.encode == nullptr) {
    return refuse(fmt::format("'{}' has no encode callback", name));
  }
  if (desc.paramsSize > MaxPassParamsSize) {
    return refuse(fmt::format("'{}' has {} bytes of params, more than {}", name, desc.paramsSize, MaxPassParamsSize));
  }

  detail::Registry& registry = detail::g_registry;
  std::string problem;
  PassId id = InvalidPass;
  {
    std::lock_guard lock{registry.mutex};
    const bool duplicate = std::ranges::any_of(
        registry.records, [&](const auto& record) { return record.has_value() && record->name == name; });
    if (duplicate) {
      problem = fmt::format("a pass named '{}' is already registered", name);
    } else if (registry.freeSlots.empty() && registry.usedSlots == detail::MaxExternalPasses) {
      problem = fmt::format("'{}' would exceed {} registered passes", name, detail::MaxExternalPasses);
    } else {
      uint32_t index = 0;
      if (!registry.freeSlots.empty()) {
        index = registry.freeSlots.back();
        registry.freeSlots.pop_back();
      } else {
        index = registry.usedSlots++;
      }
      id = detail::make_id(index, registry.generations[index]);
      auto& record = registry.records[index].emplace(detail::ExternalPassRecord{
          .name = std::string{name},
          .prefix = std::string{prefix},
          .desc = desc,
          .id = id,
          .sequence = registry.nextSequence++,
      });
      // The caller's name may not outlive this call; the record keeps its own copy
      record.desc.name = nullptr;
      detail::g_live[index].store(id, std::memory_order_release);
      detail::g_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  if (!problem.empty()) {
    return refuse(problem);
  }
  return id;
}

void unregister_pass(PassId id) noexcept {
  detail::Registry& registry = detail::g_registry;
  std::lock_guard lock{registry.mutex};
  if (detail::find_record_locked(id) == nullptr) {
    return;
  }
  const uint32_t index = detail::slot_index(id);
  // Liveness drops first, so the worker never starts this pass's callbacks again
  detail::g_live[index].store(InvalidPass, std::memory_order_release);
  registry.records[index].reset();
  registry.generations[index] = detail::next_generation(registry.generations[index]);
  registry.freeSlots.push_back(index);
  detail::g_epoch.fetch_add(1, std::memory_order_acq_rel);
}

bool set_pass_params(PassId id, const void* data, size_t size) noexcept {
  enum class Problem : uint8_t { None, Dead, TooLarge, NoData };
  Problem problem = Problem::None;
  std::string name;
  uint32_t paramsSize = 0;
  {
    std::lock_guard lock{detail::g_registry.mutex};
    detail::ExternalPassRecord* record = detail::find_record_locked(id);
    if (record == nullptr) {
      problem = Problem::Dead;
    } else if (size > record->desc.paramsSize) {
      problem = Problem::TooLarge;
    } else if (data == nullptr && size > 0) {
      problem = Problem::NoData;
    } else {
      record->pending.fill(std::byte{0});
      if (size > 0) {
        std::copy_n(static_cast<const std::byte*>(data), size, record->pending.begin());
      }
      return true;
    }
    if (record != nullptr) {
      name = record->name;
      paramsSize = record->desc.paramsSize;
    }
  }
  switch (problem) {
  case Problem::Dead: {
    const std::string pass = fmt::format("{:#x}", id);
    if (detail::log_once(pass, "dead", {})) {
      Log.warn("set_pass_params: pass {} is not registered", pass);
    }
    break;
  }
  case Problem::TooLarge:
    if (detail::log_once(name, "size", {})) {
      Log.warn("set_pass_params: '{}' takes at most {} bytes of params, not {}", name, paramsSize, size);
    }
    break;
  case Problem::NoData:
    if (detail::log_once(name, "data", {})) {
      Log.warn("set_pass_params: '{}' got no data for {} bytes", name, size);
    }
    break;
  case Problem::None:
    break;
  }
  return false;
}
} // namespace aurora::render

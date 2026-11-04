#include "gc.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace aurora::render::scene::retained::gc {
namespace {
constexpr uint32_t RecentSlots = RecentKeys * 2;
constexpr uint32_t RecentMask = RecentSlots - 1;
static_assert((RecentSlots & RecentMask) == 0 && RecentKeys < UINT16_MAX);

struct Sweep {
  uint64_t epochFrame = 0;
  uint64_t sweepAt = 0;
};

struct Release {
  uintptr_t begin = 0;
  uint32_t size = 0;
  uint64_t seq = 0;
};

struct Recent {
  Key key;
  uint64_t sceneFrame = 0;
  EvictReason reason = EvictReason::None;
  bool used = false;
};

struct Quarantined {
  uint32_t slot = 0;
  uint64_t freeAt = 0;
};

std::array<Sweep, MaxSweeps> g_sweeps{};
uint32_t g_sweepHead = 0;
uint32_t g_sweepCount = 0;
bool g_epochKnown = false;
uint32_t g_epoch = 0;

std::array<Release, MaxReleases> g_releases{};
uint32_t g_releaseCount = 0;
uint32_t g_releaseOverflow = 0;
uint64_t g_releaseSeq = 0;

std::vector<Recent> g_recent;
std::vector<uint16_t> g_recentIndex;
uint32_t g_recentHead = 0;

std::vector<Quarantined> g_quarantine;
uint32_t g_quarantineHead = 0;
uint32_t g_quarantineCount = 0;
uint64_t g_seal = 0;

std::vector<std::pair<uintptr_t, uint32_t>> g_byPos;

uint32_t recent_home(const Key& key) noexcept {
  uint64_t x = key.id ^ (uint64_t{key.subId} << 32) ^ (uint64_t{key.generation} << 16) ^ (uint64_t{key.ordinal} << 48) ^
               (uint64_t{key.occurrence} << 40);
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return static_cast<uint32_t>(x ^ (x >> 31)) & RecentMask;
}

void recent_erase(uint32_t pos) noexcept {
  g_recentIndex[pos] = 0;
  uint32_t hole = pos;
  for (uint32_t next = (pos + 1) & RecentMask; g_recentIndex[next] != 0; next = (next + 1) & RecentMask) {
    const uint32_t home = recent_home(g_recent[g_recentIndex[next] - 1].key);
    const bool movable = hole <= next ? (home <= hole || home > next) : (home <= hole && home > next);
    if (movable) {
      g_recentIndex[hole] = g_recentIndex[next];
      g_recentIndex[next] = 0;
      hole = next;
    }
  }
}

void forget_oldest() noexcept {
  const Recent& oldest = g_recent[g_recentHead];
  if (!oldest.used) {
    return;
  }
  const auto value = static_cast<uint16_t>(g_recentHead + 1);
  for (uint32_t pos = recent_home(oldest.key); g_recentIndex[pos] != 0; pos = (pos + 1) & RecentMask) {
    if (g_recentIndex[pos] == value) {
      recent_erase(pos);
      return;
    }
  }
}

void push_sweep(uint64_t epochFrame) noexcept {
  const uint64_t sweepAt = epochFrame + SweepDelay;
  if (g_sweepCount == MaxSweeps) {
    // Keeping the newest E only delays its sweep, so a full ring never evicts anything early
    g_sweeps[(g_sweepHead + g_sweepCount - 1) % MaxSweeps].sweepAt = sweepAt;
    return;
  }
  g_sweeps[(g_sweepHead + g_sweepCount) % MaxSweeps] = {.epochFrame = epochFrame, .sweepAt = sweepAt};
  ++g_sweepCount;
}

void mark_released(std::span<const uint32_t> live, std::span<Life> lives) noexcept {
  g_byPos.clear();
  for (const uint32_t slot : live) {
    if (lives[slot].posBegin != 0 && !lives[slot].released) {
      g_byPos.emplace_back(lives[slot].posBegin, slot);
    }
  }
  std::sort(g_byPos.begin(), g_byPos.end());
  for (uint32_t i = 0; i < g_releaseCount; ++i) {
    const Release& range = g_releases[i];
    auto it = std::lower_bound(g_byPos.begin(), g_byPos.end(), std::pair<uintptr_t, uint32_t>{range.begin, 0});
    for (; it != g_byPos.end() && it->first - range.begin < range.size; ++it) {
      Life& life = lives[it->second];
      // An array bound after the release belongs to a newer declaration at the same address
      if (life.posSeq <= range.seq) {
        life.released = true;
      }
    }
  }
}
} // namespace

void allocate() {
  g_recent.assign(RecentKeys, Recent{});
  g_recentIndex.assign(RecentSlots, 0);
  g_quarantine.assign(MaxSlots, Quarantined{});
  g_byPos.reserve(MaxSlots);
  reset();
}

void release() noexcept {
  g_recent = {};
  g_recentIndex = {};
  g_quarantine = {};
  g_byPos = {};
  reset();
}

void reset() noexcept {
  g_sweepHead = 0;
  g_sweepCount = 0;
  g_epochKnown = false;
  g_epoch = 0;
  g_releaseCount = 0;
  g_releaseOverflow = 0;
  g_releaseSeq = 0;
  std::fill(g_recent.begin(), g_recent.end(), Recent{});
  std::fill(g_recentIndex.begin(), g_recentIndex.end(), uint16_t{0});
  g_recentHead = 0;
  g_quarantineHead = 0;
  g_quarantineCount = 0;
  g_seal = 0;
}

SweepResult sweep(const Seal& seal, std::span<const uint32_t> live, std::span<Life> lives,
                  std::vector<Eviction>& out) noexcept {
  ++g_seal;
  SweepResult result{.releases = g_releaseCount, .releaseOverflow = std::exchange(g_releaseOverflow, 0)};
  if (g_releaseCount != 0) {
    mark_released(live, lives);
    g_releaseCount = 0;
  }
  if (seal.epochKnown) {
    // Without a MAIN camera the clock holds, so E is the next scene frame either way
    if (g_epochKnown && seal.epoch != g_epoch) {
      push_sweep(seal.sceneFrame);
      result.epochChanged = true;
    }
    g_epochKnown = true;
    g_epoch = seal.epoch;
  }
  if (!seal.scene) {
    return result;
  }
  bool due = false;
  uint64_t dueEpoch = 0;
  while (g_sweepCount != 0 && seal.sceneFrame + 1 >= g_sweeps[g_sweepHead].sweepAt) {
    due = true;
    dueEpoch = std::max(dueEpoch, g_sweeps[g_sweepHead].epochFrame);
    g_sweepHead = (g_sweepHead + 1) % MaxSweeps;
    --g_sweepCount;
  }
  for (const uint32_t slot : live) {
    const Life& life = lives[slot];
    if (life.lastSeen >= seal.sceneFrame) {
      continue;
    }
    if (life.released) {
      out.push_back({.slot = slot, .reason = EvictReason::Released});
    } else if (!life.statics) {
      if (life.lastSeen + DynamicFrames <= seal.sceneFrame) {
        out.push_back({.slot = slot, .reason = EvictReason::Dynamic});
      }
    } else if (due && life.lastSeen < dueEpoch) {
      out.push_back({.slot = slot, .reason = EvictReason::Epoch});
    }
  }
  return result;
}

void quarantine(uint32_t slot) noexcept {
  if (g_quarantine.empty()) {
    return;
  }
  g_quarantine[(g_quarantineHead + g_quarantineCount) % MaxSlots] = {.slot = slot, .freeAt = g_seal + QuarantineSeals};
  ++g_quarantineCount;
}

void free_quarantined(std::vector<uint32_t>& freeList) noexcept {
  while (g_quarantineCount != 0 && g_quarantine[g_quarantineHead].freeAt <= g_seal) {
    freeList.push_back(g_quarantine[g_quarantineHead].slot);
    g_quarantineHead = (g_quarantineHead + 1) % MaxSlots;
    --g_quarantineCount;
  }
}

void remember(const Key& key, EvictReason reason, uint64_t sceneFrame) noexcept {
  if (g_recent.empty() || reason == EvictReason::Reset || reason == EvictReason::Deactivate) {
    return;
  }
  forget_oldest();
  g_recent[g_recentHead] = {.key = key, .sceneFrame = sceneFrame, .reason = reason, .used = true};
  uint32_t pos = recent_home(key);
  while (g_recentIndex[pos] != 0) {
    pos = (pos + 1) & RecentMask;
  }
  g_recentIndex[pos] = static_cast<uint16_t>(g_recentHead + 1);
  g_recentHead = (g_recentHead + 1) % RecentKeys;
}

Recreation recreated(const Key& key, uint64_t sceneFrame) noexcept {
  if (g_recent.empty()) {
    return {};
  }
  for (uint32_t pos = recent_home(key); g_recentIndex[pos] != 0; pos = (pos + 1) & RecentMask) {
    Recent& entry = g_recent[g_recentIndex[pos] - 1];
    if (entry.key == key) {
      const Recreation found{.reason = entry.reason, .age = sceneFrame - entry.sceneFrame};
      entry.used = false;
      recent_erase(pos);
      return found;
    }
  }
  return {};
}

uint32_t pending_sweeps() noexcept { return g_sweepCount; }

void on_release(uintptr_t begin, uint32_t size) noexcept {
  const uint64_t seq = g_releaseSeq++;
  if (g_releaseCount < MaxReleases) {
    g_releases[g_releaseCount++] = {.begin = begin, .size = size, .seq = seq};
  } else {
    ++g_releaseOverflow;
  }
}

uint64_t release_seq() noexcept { return g_releaseSeq; }
} // namespace aurora::render::scene::retained::gc

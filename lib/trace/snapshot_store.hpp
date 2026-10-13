#pragma once

#include "address_map.hpp"
#include "trace_format.hpp"

#include "../gx/guest_memory.hpp"

#include <absl/container/flat_hash_map.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace aurora::trace {
class SnapshotStore {
public:
  SnapshotStore();
  SnapshotStore(const SnapshotStore&) = delete;
  SnapshotStore& operator=(const SnapshotStore&) = delete;

  void reset();
  bool add_blob(const Hash128& hash, std::span<const uint8_t> bytes, std::string& error);
  bool apply(const MemoryMapRecord& record, std::string& error);
  void begin_record() noexcept { m_scratchUsed = 0; }
  std::span<const uint8_t> resolve(uint64_t address, size_t size) noexcept;
  [[nodiscard]] const gx::guest_memory::Hooks* hooks() const noexcept { return &m_hooks; }
  [[nodiscard]] uint64_t misses() const noexcept { return m_misses; }
  [[nodiscard]] uint64_t resolves() const noexcept { return m_resolves; }
  [[nodiscard]] size_t blob_count() const noexcept { return m_blobs.size(); }

private:
  static std::span<const uint8_t> resolve_thunk(void* userData, gx::guest_memory::Kind kind, uint64_t address,
                                                size_t size) noexcept;
  std::span<const uint8_t> resolve(uint64_t address, size_t size, std::optional<gx::guest_memory::Kind> kind) noexcept;

  gx::guest_memory::Hooks m_hooks;
  std::vector<std::vector<uint8_t>> m_blobs;
  absl::flat_hash_map<Hash128, uint32_t> m_blobIds;
  AddressMap m_map;
  std::vector<AddressMap::Piece> m_pieces;
  std::vector<std::vector<uint8_t>> m_scratch;
  size_t m_scratchUsed = 0;
  uint64_t m_misses = 0;
  uint64_t m_resolves = 0;
};
} // namespace aurora::trace

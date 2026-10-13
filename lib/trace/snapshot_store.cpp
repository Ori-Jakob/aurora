#include "snapshot_store.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace aurora::trace {
namespace {
constexpr Module Log{"aurora::trace::store"};
constexpr uint64_t LoggedMisses = 16;

std::string kind_name(std::optional<gx::guest_memory::Kind> kind) {
  if (!kind) {
    return "unknown";
  }
  switch (*kind) {
  case gx::guest_memory::Kind::VertexArray:
    return "VertexArray";
  case gx::guest_memory::Kind::IndexedLoad:
    return "IndexedLoad";
  case gx::guest_memory::Kind::Texture:
    return "Texture";
  case gx::guest_memory::Kind::Tlut:
    return "Tlut";
  }
  return fmt::format("kind {}", static_cast<unsigned>(*kind));
}
} // namespace

SnapshotStore::SnapshotStore() : m_hooks{.resolve = resolve_thunk, .userData = this} {}

void SnapshotStore::reset() {
  m_blobs.clear();
  m_blobIds.clear();
  m_map.clear();
  m_scratchUsed = 0;
  m_misses = 0;
  m_resolves = 0;
}

bool SnapshotStore::add_blob(const Hash128& hash, std::span<const uint8_t> bytes, std::string& error) {
  if (hash_bytes(bytes) != hash) {
    error = fmt::format("corrupt trace: {}-byte blob {:016x}{:016x} does not match its hash", bytes.size(), hash.high,
                        hash.low);
    return false;
  }
  if (m_blobIds.contains(hash)) {
    return true;
  }
  m_blobIds.emplace(hash, static_cast<uint32_t>(m_blobs.size()));
  m_blobs.emplace_back(bytes.begin(), bytes.end());
  return true;
}

bool SnapshotStore::apply(const MemoryMapRecord& record, std::string& error) {
  const auto it = m_blobIds.find(record.blob);
  if (it == m_blobIds.end()) {
    error = fmt::format("corrupt trace: MEMORY_MAP at 0x{:X} names an unknown blob", record.address);
    return false;
  }
  if (m_blobs[it->second].size() != record.size) {
    error = fmt::format("corrupt trace: MEMORY_MAP at 0x{:X} is {} bytes but its blob is {}", record.address,
                        record.size, m_blobs[it->second].size());
    return false;
  }
  if (record.size > std::numeric_limits<uint64_t>::max() - record.address) {
    error = fmt::format("corrupt trace: MEMORY_MAP at 0x{:X} wraps the address space", record.address);
    return false;
  }
  m_map.assign(record.address, record.size, it->second);
  return true;
}

std::span<const uint8_t> SnapshotStore::resolve(uint64_t address, size_t size) noexcept {
  return resolve(address, size, std::nullopt);
}

std::span<const uint8_t> SnapshotStore::resolve_thunk(void* userData, gx::guest_memory::Kind kind, uint64_t address,
                                                      size_t size) noexcept {
  return static_cast<SnapshotStore*>(userData)->resolve(address, size, kind);
}

std::span<const uint8_t> SnapshotStore::resolve(uint64_t address, size_t size,
                                                std::optional<gx::guest_memory::Kind> kind) noexcept {
  ++m_resolves;
  if (size == 0) {
    return {};
  }
  const uint64_t clampedSize = std::min<uint64_t>(size, std::numeric_limits<uint64_t>::max() - address);
  m_map.collect(address, clampedSize, m_pieces);
  if (m_pieces.size() == 1 && m_pieces[0].address == address && m_pieces[0].size == size) {
    const auto& piece = m_pieces[0];
    return {m_blobs[piece.blob].data() + piece.blobOffset, size};
  }

  if (m_scratchUsed == m_scratch.size()) {
    m_scratch.emplace_back();
  }
  auto& scratch = m_scratch[m_scratchUsed++];
  scratch.assign(size, 0);
  uint64_t covered = 0;
  for (const auto& piece : m_pieces) {
    std::memcpy(scratch.data() + (piece.address - address), m_blobs[piece.blob].data() + piece.blobOffset, piece.size);
    covered += piece.size;
  }
  if (covered != size) {
    if (m_misses < LoggedMisses) {
      Log.warn("Unresolved {}-byte {} read at 0x{:X} ({} bytes missing); zero-filled", size, kind_name(kind), address,
               size - covered);
    }
    ++m_misses;
  }
  return scratch;
}
} // namespace aurora::trace

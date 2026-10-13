#include "address_map.hpp"

#include <algorithm>

namespace aurora::trace {
void AddressMap::assign(uint64_t address, uint64_t size, uint32_t blob) {
  if (size == 0) {
    return;
  }
  const uint64_t end = address + size;
  auto it = m_segments.upper_bound(address);
  if (it != m_segments.begin()) {
    auto prev = std::prev(it);
    if (prev->second.end > address) {
      it = prev;
    }
  }
  while (it != m_segments.end() && it->first < end) {
    const uint64_t start = it->first;
    const Segment segment = it->second;
    if (start < address) {
      it->second.end = address;
      ++it;
    } else {
      it = m_segments.erase(it);
    }
    if (segment.end > end) {
      m_segments.emplace(end, Segment{segment.end, segment.blob, segment.blobOffset + (end - start)});
      break;
    }
  }
  m_segments.insert_or_assign(address, Segment{end, blob, 0});
}

bool AddressMap::covers_exactly(uint64_t address, uint64_t size, uint32_t blob) const noexcept {
  const auto it = m_segments.find(address);
  return it != m_segments.end() && it->second.end == address + size && it->second.blob == blob &&
         it->second.blobOffset == 0;
}

void AddressMap::collect(uint64_t address, uint64_t size, std::vector<Piece>& out) const {
  out.clear();
  if (size == 0) {
    return;
  }
  const uint64_t end = address + size;
  auto it = m_segments.upper_bound(address);
  if (it != m_segments.begin()) {
    auto prev = std::prev(it);
    if (prev->second.end > address) {
      it = prev;
    }
  }
  for (; it != m_segments.end() && it->first < end; ++it) {
    const uint64_t start = std::max(it->first, address);
    const uint64_t stop = std::min(it->second.end, end);
    out.push_back({
        .address = start,
        .size = stop - start,
        .blob = it->second.blob,
        .blobOffset = it->second.blobOffset + (start - it->first),
    });
  }
}
} // namespace aurora::trace

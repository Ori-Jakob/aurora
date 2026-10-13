#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace aurora::trace {
class AddressMap {
public:
  struct Piece {
    uint64_t address;
    uint64_t size;
    uint32_t blob;
    uint64_t blobOffset;
  };

  void assign(uint64_t address, uint64_t size, uint32_t blob);
  [[nodiscard]] bool covers_exactly(uint64_t address, uint64_t size, uint32_t blob) const noexcept;
  void collect(uint64_t address, uint64_t size, std::vector<Piece>& out) const;
  void clear() noexcept { m_segments.clear(); }
  [[nodiscard]] size_t segment_count() const noexcept { return m_segments.size(); }

private:
  struct Segment {
    uint64_t end;
    uint32_t blob;
    uint64_t blobOffset;
  };

  std::map<uint64_t, Segment> m_segments;
};
} // namespace aurora::trace

#include "tlsf.hpp"

#include <algorithm>
#include <bit>
#include <cassert>

namespace aurora::render {
namespace {
constexpr uint32_t MinGranularity = 4;
constexpr uint64_t SmallUnits = uint64_t{1} << Tlsf::SecondLevelLog2;

struct ListIndex {
  uint32_t fl = 0;
  uint32_t sl = 0;
};

ListIndex insert_index(uint64_t units) noexcept {
  if (units < SmallUnits) {
    return {0, static_cast<uint32_t>(units)};
  }
  const auto f = static_cast<uint32_t>(std::bit_width(units) - 1);
  return {
      f - (Tlsf::SecondLevelLog2 - 1),
      static_cast<uint32_t>(units >> (f - Tlsf::SecondLevelLog2)) - static_cast<uint32_t>(SmallUnits),
  };
}

// Rounding up to the next class boundary makes every block in the class found large enough
ListIndex search_index(uint64_t units) noexcept {
  if (units >= SmallUnits) {
    units += (uint64_t{1} << (std::bit_width(units) - 1 - Tlsf::SecondLevelLog2)) - 1;
  }
  return insert_index(units);
}
} // namespace

Tlsf::Tlsf(uint64_t capacity, uint32_t granularity)
: m_granularity(std::bit_ceil(std::max(granularity, MinGranularity))) {
  assert(granularity == m_granularity);
  m_granularityLog2 = static_cast<uint32_t>(std::countr_zero(m_granularity));
  m_capacityUnits = capacity >> m_granularityLog2;
  assert(m_capacityUnits > 0);
  for (auto& heads : m_heads) {
    heads.fill(InvalidBlock);
  }
  if (m_capacityUnits == 0) {
    return;
  }
  const BlockId block = new_record();
  m_blocks[block].units = m_capacityUnits;
  m_lastPhys = block;
  insert_free(block);
}

std::optional<Tlsf::Allocation> Tlsf::allocate(uint64_t size, uint64_t alignment) {
  assert(alignment == 0 || std::has_single_bit(alignment));
  if (size == 0 || (alignment != 0 && !std::has_single_bit(alignment)) || size > capacity()) {
    return std::nullopt;
  }
  const uint64_t units = (size + m_granularity - 1) >> m_granularityLog2;
  const uint64_t alignUnits = alignment > m_granularity ? alignment >> m_granularityLog2 : 1;
  if (alignUnits > m_capacityUnits || units + alignUnits - 1 > m_capacityUnits) {
    return std::nullopt;
  }
  BlockId block = find_free(units + alignUnits - 1);
  if (block == InvalidBlock) {
    return std::nullopt;
  }
  remove_free(block);
  if (alignUnits > 1) {
    const uint64_t offset = m_blocks[block].offset;
    const uint64_t padding = ((offset + alignUnits - 1) & ~(alignUnits - 1)) - offset;
    if (padding > 0) {
      const BlockId aligned = split(block, padding);
      insert_free(block);
      block = aligned;
    }
  }
  if (m_blocks[block].units > units) {
    insert_free(split(block, units));
  }
  m_usedUnits += units;
  ++m_allocations;
  return Allocation{
      .block = block,
      .offset = m_blocks[block].offset << m_granularityLog2,
      .size = units << m_granularityLog2,
  };
}

void Tlsf::free(BlockId block) {
  const bool valid = block < m_blocks.size() && m_blocks[block].live && !m_blocks[block].free;
  assert(valid);
  if (!valid) {
    return;
  }
  m_usedUnits -= m_blocks[block].units;
  --m_allocations;
  const BlockId prev = m_blocks[block].prevPhys;
  if (prev != InvalidBlock && m_blocks[prev].free) {
    remove_free(prev);
    merge_next(prev);
    block = prev;
  }
  const BlockId next = m_blocks[block].nextPhys;
  if (next != InvalidBlock && m_blocks[next].free) {
    remove_free(next);
    merge_next(block);
  }
  insert_free(block);
}

void Tlsf::grow(uint64_t newCapacity) {
  const uint64_t newUnits = newCapacity >> m_granularityLog2;
  assert(newUnits >= m_capacityUnits);
  if (newUnits <= m_capacityUnits) {
    return;
  }
  const uint64_t added = newUnits - m_capacityUnits;
  if (m_lastPhys != InvalidBlock && m_blocks[m_lastPhys].free) {
    remove_free(m_lastPhys);
    m_blocks[m_lastPhys].units += added;
    insert_free(m_lastPhys);
  } else {
    const BlockId block = new_record();
    m_blocks[block].offset = m_capacityUnits;
    m_blocks[block].units = added;
    m_blocks[block].prevPhys = m_lastPhys;
    if (m_lastPhys != InvalidBlock) {
      m_blocks[m_lastPhys].nextPhys = block;
    }
    m_lastPhys = block;
    insert_free(block);
  }
  m_capacityUnits = newUnits;
}

Tlsf::Stats Tlsf::stats() const noexcept {
  Stats result{
      .capacity = capacity(),
      .used = used(),
      .allocations = m_allocations,
      .freeBlocks = m_freeBlocks,
  };
  if (m_firstLevelBits != 0) {
    // Classes are disjoint and ordered, so the largest free block sits in the highest non-empty list
    const auto fl = static_cast<uint32_t>(std::bit_width(m_firstLevelBits) - 1);
    const auto sl = static_cast<uint32_t>(std::bit_width(m_secondLevelBits[fl]) - 1);
    uint64_t largest = 0;
    for (BlockId block = m_heads[fl][sl]; block != InvalidBlock; block = m_blocks[block].nextFree) {
      largest = std::max(largest, m_blocks[block].units);
    }
    result.largestFree = largest << m_granularityLog2;
  }
  return result;
}

uint64_t Tlsf::capacity() const noexcept { return m_capacityUnits << m_granularityLog2; }

uint64_t Tlsf::used() const noexcept { return m_usedUnits << m_granularityLog2; }

uint32_t Tlsf::allocations() const noexcept { return m_allocations; }

uint32_t Tlsf::granularity() const noexcept { return m_granularity; }

bool Tlsf::validate() const {
  const size_t recordCount = m_blocks.size();
  BlockId first = m_lastPhys;
  for (size_t steps = 0; first != InvalidBlock && m_blocks[first].prevPhys != InvalidBlock; ++steps) {
    if (steps > recordCount) {
      return false;
    }
    first = m_blocks[first].prevPhys;
  }
  uint64_t end = 0;
  uint64_t used = 0;
  uint32_t allocations = 0;
  uint32_t freeBlocks = 0;
  bool prevFree = false;
  BlockId prev = InvalidBlock;
  size_t steps = 0;
  std::vector<bool> unlisted(recordCount, false);
  for (BlockId block = first; block != InvalidBlock; block = m_blocks[block].nextPhys) {
    if (block >= recordCount || ++steps > recordCount) {
      return false;
    }
    const Block& b = m_blocks[block];
    if (!b.live || b.units == 0 || b.offset != end || b.prevPhys != prev || (b.free && prevFree)) {
      return false;
    }
    if (b.free) {
      unlisted[block] = true;
      ++freeBlocks;
    } else {
      used += b.units;
      ++allocations;
    }
    prevFree = b.free;
    end += b.units;
    prev = block;
  }
  if (prev != m_lastPhys || end != m_capacityUnits || used != m_usedUnits || allocations != m_allocations ||
      freeBlocks != m_freeBlocks) {
    return false;
  }
  uint32_t listed = 0;
  for (uint32_t fl = 0; fl < FirstLevelCount; ++fl) {
    for (uint32_t sl = 0; sl < SecondLevelCount; ++sl) {
      const bool bit = ((m_secondLevelBits[fl] >> sl) & 1u) != 0;
      if (bit != (m_heads[fl][sl] != InvalidBlock)) {
        return false;
      }
      BlockId prevListed = InvalidBlock;
      for (BlockId block = m_heads[fl][sl]; block != InvalidBlock; block = m_blocks[block].nextFree) {
        if (block >= recordCount || ++listed > freeBlocks) {
          return false;
        }
        const Block& b = m_blocks[block];
        const ListIndex index = insert_index(b.units);
        if (!unlisted[block] || b.prevFree != prevListed || index.fl != fl || index.sl != sl) {
          return false;
        }
        unlisted[block] = false;
        prevListed = block;
      }
    }
    if ((((m_firstLevelBits >> fl) & 1u) != 0) != (m_secondLevelBits[fl] != 0)) {
      return false;
    }
  }
  return (m_firstLevelBits >> FirstLevelCount) == 0 && listed == freeBlocks;
}

Tlsf::BlockId Tlsf::new_record() {
  if (!m_unusedRecords.empty()) {
    const BlockId block = m_unusedRecords.back();
    m_unusedRecords.pop_back();
    m_blocks[block] = Block{.live = true};
    return block;
  }
  m_blocks.push_back(Block{.live = true});
  return static_cast<BlockId>(m_blocks.size() - 1);
}

void Tlsf::release_record(BlockId block) {
  m_blocks[block] = Block{};
  m_unusedRecords.push_back(block);
}

void Tlsf::insert_free(BlockId block) {
  Block& b = m_blocks[block];
  const ListIndex index = insert_index(b.units);
  BlockId& head = m_heads[index.fl][index.sl];
  b.free = true;
  b.prevFree = InvalidBlock;
  b.nextFree = head;
  if (head != InvalidBlock) {
    m_blocks[head].prevFree = block;
  }
  head = block;
  m_secondLevelBits[index.fl] |= 1u << index.sl;
  m_firstLevelBits |= uint64_t{1} << index.fl;
  ++m_freeBlocks;
}

void Tlsf::remove_free(BlockId block) {
  Block& b = m_blocks[block];
  const ListIndex index = insert_index(b.units);
  BlockId& head = m_heads[index.fl][index.sl];
  if (b.prevFree != InvalidBlock) {
    m_blocks[b.prevFree].nextFree = b.nextFree;
  } else {
    head = b.nextFree;
  }
  if (b.nextFree != InvalidBlock) {
    m_blocks[b.nextFree].prevFree = b.prevFree;
  }
  if (head == InvalidBlock) {
    m_secondLevelBits[index.fl] &= ~(1u << index.sl);
    if (m_secondLevelBits[index.fl] == 0) {
      m_firstLevelBits &= ~(uint64_t{1} << index.fl);
    }
  }
  b.free = false;
  b.prevFree = InvalidBlock;
  b.nextFree = InvalidBlock;
  --m_freeBlocks;
}

Tlsf::BlockId Tlsf::find_free(uint64_t units) const noexcept {
  ListIndex index = search_index(units);
  if (index.fl >= FirstLevelCount) {
    return InvalidBlock;
  }
  uint32_t mask = m_secondLevelBits[index.fl] & (~0u << index.sl);
  if (mask == 0) {
    const uint64_t higher = m_firstLevelBits & (~uint64_t{0} << (index.fl + 1));
    if (higher == 0) {
      return InvalidBlock;
    }
    index.fl = static_cast<uint32_t>(std::countr_zero(higher));
    mask = m_secondLevelBits[index.fl];
  }
  return m_heads[index.fl][std::countr_zero(mask)];
}

Tlsf::BlockId Tlsf::split(BlockId block, uint64_t units) {
  const BlockId rest = new_record();
  Block& b = m_blocks[block];
  Block& tail = m_blocks[rest];
  tail.offset = b.offset + units;
  tail.units = b.units - units;
  tail.prevPhys = block;
  tail.nextPhys = b.nextPhys;
  if (b.nextPhys != InvalidBlock) {
    m_blocks[b.nextPhys].prevPhys = rest;
  } else {
    m_lastPhys = rest;
  }
  b.nextPhys = rest;
  b.units = units;
  return rest;
}

void Tlsf::merge_next(BlockId block) {
  Block& b = m_blocks[block];
  const BlockId next = b.nextPhys;
  b.units += m_blocks[next].units;
  b.nextPhys = m_blocks[next].nextPhys;
  if (b.nextPhys != InvalidBlock) {
    m_blocks[b.nextPhys].prevPhys = block;
  } else {
    m_lastPhys = block;
  }
  release_record(next);
}

} // namespace aurora::render

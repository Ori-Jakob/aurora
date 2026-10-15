#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace aurora::render {

class Tlsf {
public:
  using BlockId = uint32_t;
  static constexpr BlockId InvalidBlock = UINT32_MAX;
  static constexpr uint32_t SecondLevelLog2 = 5;

  struct Allocation {
    BlockId block = InvalidBlock;
    uint64_t offset = 0;
    uint64_t size = 0;
  };

  struct Stats {
    uint64_t capacity = 0;
    uint64_t used = 0;
    uint64_t largestFree = 0;
    uint32_t allocations = 0;
    uint32_t freeBlocks = 0;
  };

  Tlsf(uint64_t capacity, uint32_t granularity);

  std::optional<Allocation> allocate(uint64_t size, uint64_t alignment = 0);
  void free(BlockId block);
  void grow(uint64_t newCapacity);

  [[nodiscard]] Stats stats() const noexcept;
  [[nodiscard]] uint64_t capacity() const noexcept;
  [[nodiscard]] uint64_t used() const noexcept;
  [[nodiscard]] uint32_t allocations() const noexcept;
  [[nodiscard]] uint32_t granularity() const noexcept;
  [[nodiscard]] bool validate() const;

private:
  struct Block {
    uint64_t offset = 0;
    uint64_t units = 0;
    BlockId prevPhys = InvalidBlock;
    BlockId nextPhys = InvalidBlock;
    BlockId prevFree = InvalidBlock;
    BlockId nextFree = InvalidBlock;
    bool free = false;
    bool live = false;
  };
  static constexpr uint32_t FirstLevelCount = 60;
  static constexpr uint32_t SecondLevelCount = 1u << SecondLevelLog2;

  BlockId new_record();
  void release_record(BlockId block);
  void insert_free(BlockId block);
  void remove_free(BlockId block);
  BlockId find_free(uint64_t units) const noexcept;
  BlockId split(BlockId block, uint64_t units);
  void merge_next(BlockId block);

  std::vector<Block> m_blocks;
  std::vector<BlockId> m_unusedRecords;
  std::array<std::array<BlockId, SecondLevelCount>, FirstLevelCount> m_heads{};
  std::array<uint32_t, FirstLevelCount> m_secondLevelBits{};
  uint64_t m_firstLevelBits = 0;
  uint64_t m_capacityUnits = 0;
  uint64_t m_usedUnits = 0;
  uint32_t m_granularity = 0;
  uint32_t m_granularityLog2 = 0;
  uint32_t m_allocations = 0;
  uint32_t m_freeBlocks = 0;
  BlockId m_lastPhys = InvalidBlock;
};

} // namespace aurora::render

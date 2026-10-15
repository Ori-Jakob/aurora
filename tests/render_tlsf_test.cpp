#include <gtest/gtest.h>

#include "render/tlsf.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <random>
#include <vector>

namespace aurora::render {
namespace {
constexpr uint64_t KiB = 1024;

void expect_same_stats(const Tlsf::Stats& lhs, const Tlsf::Stats& rhs) {
  EXPECT_EQ(lhs.capacity, rhs.capacity);
  EXPECT_EQ(lhs.used, rhs.used);
  EXPECT_EQ(lhs.largestFree, rhs.largestFree);
  EXPECT_EQ(lhs.allocations, rhs.allocations);
  EXPECT_EQ(lhs.freeBlocks, rhs.freeBlocks);
}

TEST(TlsfTest, AllocatesWithinCapacity) {
  Tlsf heap{256 * KiB, 256};
  std::vector<uint64_t> offsets;
  for (int i = 0; i < 4; ++i) {
    const auto allocation = heap.allocate(64 * KiB);
    ASSERT_TRUE(allocation.has_value());
    EXPECT_LE(allocation->offset + allocation->size, 256 * KiB);
    offsets.push_back(allocation->offset);
  }
  std::ranges::sort(offsets);
  EXPECT_EQ(std::ranges::adjacent_find(offsets), offsets.end());
  EXPECT_FALSE(heap.allocate(64 * KiB).has_value());
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, RoundsToGranularity) {
  Tlsf heap{64 * KiB, 256};
  const auto small = heap.allocate(1);
  const auto large = heap.allocate(257);
  ASSERT_TRUE(small.has_value());
  ASSERT_TRUE(large.has_value());
  EXPECT_EQ(small->size, 256u);
  EXPECT_EQ(large->size, 512u);
  EXPECT_EQ(heap.stats().used, 768u);
  EXPECT_EQ(heap.used(), 768u);
  EXPECT_EQ(heap.allocations(), 2u);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, RespectsAlignment) {
  Tlsf heap{64 * KiB, 16};
  const auto first = heap.allocate(48);
  ASSERT_TRUE(first.has_value());
  const uint32_t freeBefore = heap.stats().freeBlocks;
  const auto aligned = heap.allocate(64, 4096);
  ASSERT_TRUE(aligned.has_value());
  EXPECT_EQ(aligned->offset % 4096, 0u);
  EXPECT_EQ(aligned->size, 64u);
  EXPECT_GT(heap.stats().freeBlocks, freeBefore);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, FreeCoalescesNeighbours) {
  Tlsf heap{256 * KiB, 256};
  const auto a = heap.allocate(64 * KiB);
  const auto b = heap.allocate(64 * KiB);
  const auto c = heap.allocate(64 * KiB);
  const auto d = heap.allocate(64 * KiB);
  ASSERT_TRUE(a && b && c && d);
  heap.free(b->block);
  heap.free(a->block);
  heap.free(c->block);
  EXPECT_EQ(heap.stats().freeBlocks, 1u);
  EXPECT_EQ(heap.stats().largestFree, 192 * KiB);
  EXPECT_TRUE(heap.validate());
  heap.free(d->block);
  const auto stats = heap.stats();
  EXPECT_EQ(stats.freeBlocks, 1u);
  EXPECT_EQ(stats.largestFree, 256 * KiB);
  EXPECT_EQ(stats.used, 0u);
  EXPECT_EQ(stats.allocations, 0u);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, ExhaustionReturnsNullopt) {
  Tlsf heap{16 * KiB, 256};
  for (int i = 0; i < 64; ++i) {
    ASSERT_TRUE(heap.allocate(256).has_value());
  }
  EXPECT_FALSE(heap.allocate(1).has_value());
  EXPECT_FALSE(heap.allocate(16 * KiB).has_value());
  EXPECT_EQ(heap.stats().used, heap.capacity());
  EXPECT_EQ(heap.stats().freeBlocks, 0u);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, GoodFitClassRule) {
  // 101 units sit in the [100, 101] class; a 101-unit search rounds up into [102, 103]
  Tlsf heap{404, 4};
  EXPECT_EQ(heap.capacity(), 404u);
  EXPECT_FALSE(heap.allocate(404).has_value());
  EXPECT_TRUE(heap.allocate(400).has_value());
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, GrowExtendsTailFreeBlock) {
  constexpr uint32_t Unit = 256;
  Tlsf heap{1024 * Unit, Unit};
  ASSERT_TRUE(heap.allocate(1023 * Unit).has_value());
  heap.grow(2048 * Unit);
  const auto stats = heap.stats();
  EXPECT_EQ(stats.capacity, 2048 * Unit);
  EXPECT_EQ(stats.freeBlocks, 1u);
  EXPECT_EQ(stats.largestFree, 1025 * Unit);
  const auto tail = heap.allocate(1024 * Unit);
  ASSERT_TRUE(tail.has_value());
  EXPECT_EQ(tail->offset, 1023 * Unit);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, GrowAfterFullHeap) {
  Tlsf heap{64 * KiB, 256};
  ASSERT_TRUE(heap.allocate(64 * KiB).has_value());
  EXPECT_EQ(heap.stats().freeBlocks, 0u);
  heap.grow(128 * KiB);
  EXPECT_EQ(heap.stats().freeBlocks, 1u);
  const auto added = heap.allocate(64 * KiB);
  ASSERT_TRUE(added.has_value());
  EXPECT_EQ(added->offset, 64 * KiB);
  EXPECT_TRUE(heap.validate());
}

TEST(TlsfTest, RandomizedAgainstReference) {
  constexpr uint64_t Capacity = 4096 * KiB;
  constexpr uint32_t Granularity = 16;
  constexpr std::array<uint64_t, 3> Alignments{0, 256, 4096};
  struct Live {
    Tlsf::BlockId block;
    uint64_t offset;
    uint64_t size;
  };
  Tlsf heap{Capacity, Granularity};
  std::mt19937 rng{1234};
  std::uniform_int_distribution<uint64_t> sizes{1, 64 * KiB};
  std::uniform_int_distribution<size_t> alignments{0, Alignments.size() - 1};
  std::bernoulli_distribution allocates{0.6};
  std::vector<Live> live;
  std::map<uint64_t, uint64_t> intervals;
  for (int op = 0; op < 20000; ++op) {
    if (live.empty() || allocates(rng)) {
      const uint64_t size = sizes(rng);
      const uint64_t alignment = Alignments[alignments(rng)];
      if (const auto allocation = heap.allocate(size, alignment)) {
        const uint64_t end = allocation->offset + allocation->size;
        ASSERT_EQ(allocation->size, (size + Granularity - 1) / Granularity * Granularity);
        ASSERT_EQ(allocation->offset % std::max<uint64_t>(alignment, Granularity), 0u);
        ASSERT_LE(end, Capacity);
        const auto next = intervals.lower_bound(allocation->offset);
        if (next != intervals.end()) {
          ASSERT_GE(next->first, end) << "op " << op;
        }
        if (next != intervals.begin()) {
          ASSERT_LE(std::prev(next)->second, allocation->offset) << "op " << op;
        }
        intervals.emplace(allocation->offset, end);
        live.push_back({allocation->block, allocation->offset, allocation->size});
      }
    } else {
      std::uniform_int_distribution<size_t> pick{0, live.size() - 1};
      const size_t index = pick(rng);
      heap.free(live[index].block);
      intervals.erase(live[index].offset);
      live[index] = live.back();
      live.pop_back();
    }
    ASSERT_TRUE(heap.validate()) << "op " << op;
  }
  uint64_t used = 0;
  for (const auto& allocation : live) {
    used += allocation.size;
  }
  EXPECT_EQ(heap.stats().used, used);
  EXPECT_EQ(heap.used(), used);
  EXPECT_EQ(heap.stats().allocations, live.size());
  EXPECT_EQ(heap.allocations(), live.size());
}

TEST(TlsfTest, ZeroSizeRejected) {
  Tlsf heap{64 * KiB, 256};
  EXPECT_FALSE(heap.allocate(0).has_value());
  EXPECT_EQ(heap.stats().allocations, 0u);
}

TEST(TlsfTest, DoubleFree) {
  Tlsf heap{64 * KiB, 256};
  const auto allocation = heap.allocate(256);
  ASSERT_TRUE(allocation.has_value());
  heap.free(allocation->block);
#ifndef NDEBUG
  EXPECT_DEATH(heap.free(allocation->block), "");
#else
  const auto before = heap.stats();
  heap.free(allocation->block);
  expect_same_stats(before, heap.stats());
  EXPECT_TRUE(heap.validate());
#endif
}

TEST(TlsfTest, LargeCapacity) {
  Tlsf heap{uint64_t{1} << 40, 256};
  const auto allocation = heap.allocate(uint64_t{1} << 39);
  ASSERT_TRUE(allocation.has_value());
  EXPECT_EQ(allocation->size, uint64_t{1} << 39);
  EXPECT_TRUE(heap.validate());
}
} // namespace
} // namespace aurora::render

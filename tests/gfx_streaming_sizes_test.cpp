#include <gtest/gtest.h>

#include "gfx/resources.hpp"

#include <cstdint>

namespace aurora::gfx::detail {
namespace {
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t GiB = 1024 * MiB;
constexpr StreamingLimits Limits{.vertex = 128 * MiB, .index = 128 * MiB, .storage = 128 * MiB, .staging = 256 * MiB};

AuroraStats high_water(uint64_t vertex, uint64_t index, uint64_t storage) {
  AuroraStats stats{};
  stats.lastVertSize = static_cast<uint32_t>(vertex);
  stats.lastIndexSize = static_cast<uint32_t>(index);
  stats.lastStorageSize = static_cast<uint32_t>(storage);
  return stats;
}

TEST(StreamingSizesTest, DefaultLayoutMatchesLegacy) {
  constexpr StagingLayout layout = staging_layout({});
  EXPECT_EQ(layout.vertex, 0u);
  EXPECT_EQ(layout.uniform, 5 * MiB);
  EXPECT_EQ(layout.index, 29 * MiB);
  EXPECT_EQ(layout.storage, 31 * MiB);
  EXPECT_EQ(layout.textureUpload, 39 * MiB);
  EXPECT_EQ(layout.total, 63 * MiB);
}

TEST(StreamingSizesTest, NoGrowthAtOrBelowThreeQuarters) {
  const auto growth = next_streaming_sizes({}, high_water(0, 0, 6 * MiB), {}, Limits);
  EXPECT_EQ(growth.sizes, StreamingSizes{});
  EXPECT_FALSE(growth.stagingCapped);
}

TEST(StreamingSizesTest, GrowsAboveThreeQuarters) {
  const auto growth = next_streaming_sizes({}, high_water(0, 0, 6 * MiB + 1), {}, Limits);
  EXPECT_EQ(growth.sizes.storage, 16 * MiB);
  EXPECT_EQ(growth.sizes.vertex, VertexBufferSize);
  EXPECT_EQ(growth.sizes.index, IndexBufferSize);
  EXPECT_FALSE(growth.stagingCapped);
}

TEST(StreamingSizesTest, RequestedMinimumApplied) {
  const auto growth = next_streaming_sizes({}, high_water(0, 0, 0), {.storage = 32 * MiB}, Limits);
  EXPECT_EQ(growth.sizes.storage, 32 * MiB);
  EXPECT_EQ(growth.sizes.vertex, VertexBufferSize);
  EXPECT_EQ(growth.sizes.index, IndexBufferSize);
}

TEST(StreamingSizesTest, NeverShrinks) {
  const auto growth = next_streaming_sizes({.storage = 16 * MiB}, high_water(0, 0, 0), {.storage = 8 * MiB}, Limits);
  EXPECT_EQ(growth.sizes.storage, 16 * MiB);
}

TEST(StreamingSizesTest, CappedByLimits) {
  const auto growth = next_streaming_sizes({.storage = 128 * MiB}, high_water(0, 0, 128 * MiB), {}, Limits);
  EXPECT_EQ(growth.sizes.storage, 128 * MiB);
  EXPECT_FALSE(growth.stagingCapped);
}

TEST(StreamingSizesTest, CappedByStagingTotal) {
  const StreamingSizes current{.vertex = 80 * MiB, .index = 2 * MiB, .storage = 64 * MiB};
  // 80 + 24 + 2 + 128 + 24 = 258 MiB would not fit one 256 MiB staging slot
  const auto growth = next_streaming_sizes(current, high_water(0, 0, 64 * MiB), {}, Limits);
  EXPECT_EQ(growth.sizes, current);
  EXPECT_TRUE(growth.stagingCapped);
}

TEST(StreamingSizesTest, StagingCapKeepsFittingGrowth) {
  const StreamingSizes current{.vertex = 80 * MiB, .index = 2 * MiB, .storage = 64 * MiB};
  constexpr StreamingLimits limits{.vertex = 96 * MiB, .index = 128 * MiB, .storage = 128 * MiB, .staging = 256 * MiB};
  const auto growth = next_streaming_sizes(current, high_water(80 * MiB, 0, 64 * MiB), {}, limits);
  EXPECT_EQ(growth.sizes.vertex, 96 * MiB);
  EXPECT_EQ(staging_layout(growth.sizes).total, 210 * MiB);
  EXPECT_EQ(growth.sizes.index, 2 * MiB);
  EXPECT_EQ(growth.sizes.storage, 64 * MiB);
  EXPECT_TRUE(growth.stagingCapped);
}

TEST(StreamingSizesTest, HighWaterComparedInU64) {
  constexpr StreamingLimits limits{.vertex = 2 * GiB, .index = 2 * GiB, .storage = 2 * GiB, .staging = 8 * GiB};
  const auto growth = next_streaming_sizes({.storage = 1 * GiB}, high_water(0, 0, 1 * GiB), {}, limits);
  EXPECT_EQ(growth.sizes.storage, 2 * GiB);
}

TEST(StreamingSizesTest, VertexAndIndexIndependent) {
  const auto growth = next_streaming_sizes({}, high_water(9 * MiB / 2, 0, 0), {}, Limits);
  EXPECT_EQ(growth.sizes.vertex, 10 * MiB);
  EXPECT_EQ(growth.sizes.index, IndexBufferSize);
  EXPECT_EQ(growth.sizes.storage, StorageBufferSize);
}

TEST(ResidentCapacityTest, NoRequestKeepsCapacity) {
  EXPECT_EQ(next_resident_capacity(0, 0, 8 * MiB, 128 * MiB), 0u);
  EXPECT_EQ(next_resident_capacity(2 * MiB, 0, 8 * MiB, 128 * MiB), 2 * MiB);
}

TEST(ResidentCapacityTest, RequestWithinCapacityStillGrows) {
  EXPECT_EQ(next_resident_capacity(2 * MiB, 2 * MiB, 8 * MiB, 128 * MiB), 4 * MiB);
  EXPECT_EQ(next_resident_capacity(1 * MiB, 1000 * 1024, 8 * MiB, 128 * MiB), 2 * MiB);
}

TEST(ResidentCapacityTest, GrowsToAtLeastOneMiB) {
  EXPECT_EQ(next_resident_capacity(0, 256, 8 * MiB, 128 * MiB), 1 * MiB);
}

TEST(ResidentCapacityTest, GrowsToTwiceTheRequestRoundedUp) {
  EXPECT_EQ(next_resident_capacity(0, 3 * MiB, 8 * MiB, 128 * MiB), 8 * MiB);
  EXPECT_EQ(next_resident_capacity(1 * MiB, 1 * MiB + 1, 8 * MiB, 128 * MiB), 4 * MiB);
}

TEST(ResidentCapacityTest, CappedAtAQuarterOfTheBound) {
  EXPECT_EQ(next_resident_capacity(0, 3 * MiB, 1 * MiB, 16 * MiB), 4 * MiB);
}

TEST(ResidentCapacityTest, CappedByTheStreamingRange) {
  EXPECT_EQ(next_resident_capacity(0, 3 * MiB, 126 * MiB, 128 * MiB), 2 * MiB);
}

TEST(ResidentCapacityTest, CappedByThirtyTwoBitOffsets) {
  constexpr uint64_t streaming = 4 * GiB - 2 * MiB;
  EXPECT_EQ(next_resident_capacity(0, 3 * MiB, streaming, 64 * GiB), UINT32_MAX - streaming);
}

TEST(ResidentCapacityTest, NeverShrinks) {
  EXPECT_EQ(next_resident_capacity(8 * MiB, 9 * MiB, 124 * MiB, 128 * MiB), 8 * MiB);
  EXPECT_EQ(next_resident_capacity(16 * MiB, 5 * MiB, 8 * MiB, 128 * MiB), 16 * MiB);
}

TEST(ResidentLayoutTest, StreamingGrowthBumpsTheEpoch) {
  const ResidentLayout last{.streamingStorage = 8 * MiB, .capacity = 1 * MiB, .epoch = 3};
  const auto next = next_resident_layout(last, 16 * MiB, 1 * MiB);
  EXPECT_EQ(next.epoch, 4u);
  EXPECT_EQ(next.streamingStorage, 16 * MiB);
  EXPECT_EQ(next.capacity, 1 * MiB);
}

TEST(ResidentLayoutTest, ArenaGrowthBumpsTheEpoch) {
  const ResidentLayout last{.streamingStorage = 8 * MiB, .capacity = 1 * MiB, .epoch = 3};
  const auto next = next_resident_layout(last, 8 * MiB, 2 * MiB);
  EXPECT_EQ(next.epoch, 4u);
  EXPECT_EQ(next.capacity, 2 * MiB);
  EXPECT_EQ(next_resident_layout(next, 8 * MiB, 2 * MiB).epoch, 4u);
}

TEST(ResidentLayoutTest, UnchangedLayoutKeepsTheEpoch) {
  const ResidentLayout last{.streamingStorage = 8 * MiB, .capacity = 1 * MiB, .epoch = 3};
  EXPECT_EQ(next_resident_layout(last, 8 * MiB, 1 * MiB).epoch, 3u);
}

TEST(ResidentLayoutTest, StreamingStorageLeavesRoomForTheArena) {
  EXPECT_EQ(streaming_storage_limit(128 * MiB, 8 * MiB), 120 * MiB);
  EXPECT_EQ(streaming_storage_limit(128 * MiB, 0), 128 * MiB);
}

TEST(ChannelStorageTest, RequestsRoundToPowersOfTwoMiB) {
  EXPECT_EQ(round_streaming_request(0), 0u);
  EXPECT_EQ(round_streaming_request(1), MiB);
  EXPECT_EQ(round_streaming_request(5000003), 8 * MiB);
  EXPECT_EQ(round_streaming_request(32 * MiB), 32 * MiB);
  EXPECT_EQ(round_streaming_limit(128 * MiB + 12), 128 * MiB);
}

void expect_segments_aligned(const StreamingSizes& sizes) {
  const StagingLayout layout = staging_layout(sizes);
  for (const uint64_t offset : {layout.uniform, layout.index, layout.storage, layout.textureUpload}) {
    EXPECT_EQ(offset % 8, 0u) << offset;
  }
  EXPECT_EQ(layout.total % 4, 0u);
  EXPECT_EQ(sizes.storage % MiB, 0u);
}

TEST(ChannelStorageTest, OddRequestsKeepSegmentAlignment) {
  const uint64_t bound = round_streaming_limit(128 * MiB + 12);
  const StreamingLimits limits{.vertex = bound, .index = bound, .storage = bound, .staging = 4 * GiB};
  const auto odd = next_streaming_sizes({}, high_water(0, 0, 0),
                                        {.vertex = 0, .index = 0, .storage = round_streaming_request(5000003)}, limits);
  EXPECT_EQ(odd.sizes.storage, 8 * MiB);
  expect_segments_aligned(odd.sizes);
  const auto capped = next_streaming_sizes(
      {}, high_water(0, 0, 0), {.vertex = 0, .index = 0, .storage = round_streaming_request(200 * MiB + 3)}, limits);
  EXPECT_EQ(capped.sizes.storage, 128 * MiB);
  expect_segments_aligned(capped.sizes);
  const auto grown = next_streaming_sizes(capped.sizes, high_water(0, 0, 127 * MiB), {}, limits);
  expect_segments_aligned(grown.sizes);
}

TEST(ChannelStorageTest, BudgetExamples) {
  EXPECT_EQ(channel_storage_budget(8 * MiB, 3 * MiB, true, UINT64_MAX), 2 * MiB);
  EXPECT_EQ(channel_storage_budget(8 * MiB, 1 * MiB, true, UINT64_MAX), 6 * MiB);
  EXPECT_EQ(channel_storage_budget(8 * MiB, 0, false, UINT64_MAX), 4 * MiB);
  EXPECT_EQ(channel_storage_budget(8 * MiB, 8 * MiB * 6 / 10, true, UINT64_MAX), 0u);
  EXPECT_EQ(channel_storage_budget(1024 * MiB, 0, true, 256 * MiB), 256 * MiB);
  EXPECT_EQ(legacy_storage_reserve(32 * MiB, 0, true), 8 * MiB);
}

TEST(ChannelStorageTest, RequestCoversDemandAboveLegacy) {
  for (const uint64_t legacy : {uint64_t{0}, uint64_t{1}, 3 * MiB, 5 * MiB + 7, 40 * MiB}) {
    for (const uint64_t demand : {uint64_t{1}, uint64_t{4096}, MiB + 1, 9 * MiB, 100 * MiB + 3}) {
      const uint64_t storage = round_streaming_request(channel_storage_request(demand, legacy));
      EXPECT_GE(channel_storage_budget(storage, legacy, true, UINT64_MAX), demand) << legacy << " " << demand;
      EXPECT_LE(legacy + demand, storage / 2) << legacy << " " << demand;
    }
  }
}
} // namespace
} // namespace aurora::gfx::detail

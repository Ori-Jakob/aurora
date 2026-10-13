#include "trace/address_map.hpp"
#include "trace/capture_session.hpp"
#include "trace/snapshot_store.hpp"
#include "trace/trace_reader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace aurora::trace {
namespace {
constexpr uint32_t BlobA = 1;
constexpr uint32_t BlobB = 2;
constexpr uint32_t BlobC = 3;

std::filesystem::path temp_trace(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "aurora_trace_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / (name + ".gxtrace");
  std::filesystem::remove(path);
  return path;
}

std::vector<uint8_t> filled(size_t size, uint8_t value) { return std::vector<uint8_t>(size, value); }

std::vector<uint8_t> pattern(size_t size, uint8_t seed) {
  std::vector<uint8_t> bytes(size);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<uint8_t>(seed + i * 13);
  }
  return bytes;
}

MemoryMapRecord map_of(uint64_t address, const std::vector<uint8_t>& blob) {
  return {.address = address, .size = static_cast<uint32_t>(blob.size()), .blob = hash_bytes(blob)};
}

void add_and_map(SnapshotStore& store, uint64_t address, const std::vector<uint8_t>& blob) {
  std::string error;
  ASSERT_TRUE(store.add_blob(hash_bytes(blob), blob, error)) << error;
  ASSERT_TRUE(store.apply(map_of(address, blob), error)) << error;
}

bool bytes_equal(std::span<const uint8_t> actual, std::span<const uint8_t> expected) {
  return std::ranges::equal(actual, expected);
}

class SessionFile {
public:
  explicit SessionFile(const std::string& name) : m_path(temp_trace(name)) {
    std::string error;
    EXPECT_TRUE(session.open(m_path, 0, 0, error)) << error;
    EXPECT_TRUE(session.write_header_chunks({}, {}, error)) << error;
  }

  void finish() {
    std::string error;
    if (session.in_frame()) {
      EXPECT_TRUE(session.end_frame(error)) << error;
    }
    EXPECT_TRUE(session.finalize(0, {}, error)) << error;
  }

  struct Contents {
    std::vector<FrameView> frames;
    size_t blobs = 0;
    size_t maps = 0;
    std::optional<Trailer> trailer;
  };

  Contents read() const {
    Contents contents;
    TraceReader reader;
    std::string error;
    EXPECT_TRUE(reader.open(m_path, error)) << error;
    const BlobSink sink = [&](const Hash128&, std::span<const uint8_t>, std::string&) {
      ++contents.blobs;
      return true;
    };
    while (true) {
      FrameView frame;
      if (reader.next(frame, sink, error) != TraceReader::Status::Frame) {
        break;
      }
      contents.maps +=
          static_cast<size_t>(std::ranges::count(frame.records, RecordType::MemoryMap, &FrameRecordView::type));
      contents.frames.push_back(std::move(frame));
    }
    EXPECT_TRUE(error.empty()) << error;
    contents.trailer = reader.trailer();
    return contents;
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return m_path; }

  CaptureSession session;

private:
  std::filesystem::path m_path;
};

const std::vector<uint8_t> kFifo{0x61, 0x41, 0x00, 0x00, 0x01};

TEST(AddressMap, AssignOverwritesTrimsAndSplits) {
  AddressMap map;
  map.assign(0, 100, BlobA);
  map.assign(40, 20, BlobB);
  std::vector<AddressMap::Piece> pieces;
  map.collect(0, 100, pieces);
  ASSERT_EQ(pieces.size(), 3u);
  EXPECT_EQ(pieces[0].address, 0u);
  EXPECT_EQ(pieces[0].size, 40u);
  EXPECT_EQ(pieces[0].blob, BlobA);
  EXPECT_EQ(pieces[0].blobOffset, 0u);
  EXPECT_EQ(pieces[1].address, 40u);
  EXPECT_EQ(pieces[1].size, 20u);
  EXPECT_EQ(pieces[1].blob, BlobB);
  EXPECT_EQ(pieces[1].blobOffset, 0u);
  EXPECT_EQ(pieces[2].address, 60u);
  EXPECT_EQ(pieces[2].size, 40u);
  EXPECT_EQ(pieces[2].blob, BlobA);
  EXPECT_EQ(pieces[2].blobOffset, 60u);
  EXPECT_EQ(map.segment_count(), 3u);

  map.assign(20, 70, BlobC);
  map.collect(0, 100, pieces);
  ASSERT_EQ(pieces.size(), 3u);
  EXPECT_EQ(pieces[0].size, 20u);
  EXPECT_EQ(pieces[1].address, 20u);
  EXPECT_EQ(pieces[1].blob, BlobC);
  EXPECT_EQ(pieces[2].address, 90u);
  EXPECT_EQ(pieces[2].blob, BlobA);
  EXPECT_EQ(pieces[2].blobOffset, 90u);
}

TEST(AddressMap, CoversExactlyRequiresUntrimmedSegment) {
  AddressMap map;
  map.assign(0x1000, 100, BlobA);
  EXPECT_TRUE(map.covers_exactly(0x1000, 100, BlobA));
  EXPECT_FALSE(map.covers_exactly(0x1000, 50, BlobA));
  EXPECT_FALSE(map.covers_exactly(0x1000, 100, BlobB));
  EXPECT_FALSE(map.covers_exactly(0x1010, 84, BlobA));

  map.assign(0x1000, 16, BlobB);
  EXPECT_FALSE(map.covers_exactly(0x1000, 100, BlobA));
  EXPECT_FALSE(map.covers_exactly(0x1010, 84, BlobA));
  EXPECT_TRUE(map.covers_exactly(0x1000, 16, BlobB));

  map.assign(0x1000, 100, BlobA);
  EXPECT_TRUE(map.covers_exactly(0x1000, 100, BlobA));
  EXPECT_EQ(map.segment_count(), 1u);
}

TEST(SnapshotStore, FastPathReturnsSpanIntoBlob) {
  SnapshotStore store;
  const auto blob = pattern(64, 1);
  add_and_map(store, 0x80001000, blob);
  const auto whole = store.resolve(0x80001000, 64);
  ASSERT_TRUE(bytes_equal(whole, blob));
  store.begin_record();
  const auto part = store.resolve(0x80001010, 16);
  EXPECT_EQ(part.data(), whole.data() + 16);
  EXPECT_TRUE(bytes_equal(part, std::span{blob}.subspan(16, 16)));
  EXPECT_EQ(store.misses(), 0u);
  EXPECT_EQ(store.resolves(), 2u);
  EXPECT_TRUE(store.resolve(0x80001000, 0).empty());
}

TEST(SnapshotStore, AssemblesAcrossSegmentsAndZeroFillsHoles) {
  SnapshotStore store;
  add_and_map(store, 0x1000, filled(16, 0xAA));
  add_and_map(store, 0x1010, filled(16, 0xBB));

  const auto joined = store.resolve(0x1008, 16);
  std::vector<uint8_t> expected(8, 0xAA);
  expected.resize(16, 0xBB);
  EXPECT_TRUE(bytes_equal(joined, expected));
  EXPECT_EQ(store.misses(), 0u);

  const auto holey = store.resolve(0x1018, 16);
  expected.assign(8, 0xBB);
  expected.resize(16, 0x00);
  EXPECT_TRUE(bytes_equal(holey, expected));
  EXPECT_EQ(store.misses(), 1u);

  EXPECT_TRUE(bytes_equal(store.resolve(0x9000, 4), filled(4, 0)));
  EXPECT_EQ(store.misses(), 2u);
}

TEST(SnapshotStore, SpansLiveUntilNextRecord) {
  SnapshotStore store;
  add_and_map(store, 0x1000, filled(16, 0x11));
  add_and_map(store, 0x1010, filled(16, 0x22));
  add_and_map(store, 0x1020, filled(16, 0x33));

  store.begin_record();
  const auto first = store.resolve(0x1008, 16);
  const auto second = store.resolve(0x1018, 16);
  std::vector<uint8_t> expectedFirst(8, 0x11);
  expectedFirst.resize(16, 0x22);
  std::vector<uint8_t> expectedSecond(8, 0x22);
  expectedSecond.resize(16, 0x33);
  EXPECT_NE(first.data(), second.data());
  EXPECT_TRUE(bytes_equal(first, expectedFirst));
  EXPECT_TRUE(bytes_equal(second, expectedSecond));

  store.begin_record();
  const auto third = store.resolve(0x1018, 16);
  EXPECT_TRUE(bytes_equal(third, expectedSecond));
}

TEST(SnapshotStore, RejectsHashMismatch) {
  SnapshotStore store;
  const auto blob = pattern(32, 5);
  std::string error;
  EXPECT_FALSE(store.add_blob(hash_bytes(pattern(32, 6)), blob, error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(store.blob_count(), 0u);
  EXPECT_FALSE(store.apply(map_of(0x1000, blob), error));
}

TEST(CaptureSession, RepeatedExactRangeEmitsNoMap) {
  SessionFile file{"repeated_range"};
  const auto bytes = pattern(144, 2);
  file.session.begin_frame(0, {.efbWidth = 640, .efbHeight = 480});
  for (int i = 0; i < 3; ++i) {
    file.session.observe(0, 0x80200000, bytes);
    file.session.processed(kFifo);
  }
  file.finish();

  const auto contents = file.read();
  EXPECT_EQ(contents.maps, 1u);
  EXPECT_EQ(contents.blobs, 1u);
  ASSERT_EQ(contents.frames.size(), 1u);
  ASSERT_TRUE(contents.trailer);
  EXPECT_EQ(contents.trailer->mapRecords, 1u);
  EXPECT_EQ(contents.trailer->blobCount, 1u);
  EXPECT_EQ(contents.trailer->fifoBytes, kFifo.size() * 3);
}

TEST(CaptureSession, ChangedContentEmitsMapAndReusesKnownBlob) {
  SessionFile file{"changed_content"};
  const auto a = pattern(64, 1);
  const auto b = pattern(64, 2);
  file.session.begin_frame(0, {});
  file.session.observe(0, 0x1000, a);
  file.session.processed(kFifo);
  file.session.observe(0, 0x1000, b);
  file.session.processed(kFifo);
  file.session.observe(0, 0x1000, a);
  file.session.processed(kFifo);
  file.finish();

  const auto contents = file.read();
  EXPECT_EQ(contents.maps, 3u);
  EXPECT_EQ(contents.blobs, 2u);
}

TEST(CaptureSession, SubRangeAndOverwriteThenRestoreEmitMaps) {
  SessionFile file{"sub_range"};
  const auto x = pattern(64, 1);
  const auto y = pattern(64, 9);
  const std::vector<uint8_t> xSub(x.begin() + 16, x.begin() + 32);
  file.session.begin_frame(0, {});
  file.session.observe(2, 0x1000, x);
  file.session.observe(1, 0x1010, xSub);
  file.session.observe(2, 0x1000, x);
  file.session.observe(2, 0x1000, x);
  file.session.observe(2, 0x1000, y);
  file.session.observe(2, 0x1000, x);
  file.session.processed(kFifo);
  file.finish();

  const auto contents = file.read();
  EXPECT_EQ(contents.maps, 5u);
  EXPECT_EQ(contents.blobs, 3u);

  SnapshotStore store;
  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(file.path(), error)) << error;
  FrameView frame;
  const BlobSink sink = [&](const Hash128& hash, std::span<const uint8_t> bytes, std::string& sinkError) {
    return store.add_blob(hash, bytes, sinkError);
  };
  ASSERT_EQ(reader.next(frame, sink, error), TraceReader::Status::Frame) << error;
  for (const auto& record : frame.records) {
    if (record.type == RecordType::MemoryMap) {
      MemoryMapRecord map;
      ASSERT_TRUE(decode_memory_map(record.payload, map));
      ASSERT_TRUE(store.apply(map, error)) << error;
    }
  }
  EXPECT_TRUE(bytes_equal(store.resolve(0x1000, 64), x));
  EXPECT_EQ(store.misses(), 0u);
}

TEST(CaptureSession, RecordsOutsideFrameCarryIntoNextFrame) {
  SessionFile file{"carry"};
  const std::vector<uint8_t> first{0x61, 0x41, 0, 0, 1};
  const std::vector<uint8_t> carried{0x61, 0x42, 0, 0, 2};
  const std::vector<uint8_t> second{0x61, 0x43, 0, 0, 3};
  file.session.begin_frame(10, {});
  file.session.processed(first);
  std::string error;
  ASSERT_TRUE(file.session.end_frame(error)) << error;
  file.session.processed(carried);
  file.session.begin_frame(11, {.efbWidth = 320, .efbHeight = 240});
  file.session.processed(second);
  file.finish();

  const auto contents = file.read();
  ASSERT_EQ(contents.frames.size(), 2u);
  const auto& records = contents.frames[1].records;
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].type, RecordType::Fifo);
  EXPECT_TRUE(bytes_equal(records[0].payload, carried));
  EXPECT_TRUE(bytes_equal(records[1].payload, second));
  EXPECT_EQ(contents.frames[1].config.efbWidth, 320u);
  EXPECT_EQ(contents.frames[1].sourceFrame, 11u);
  ASSERT_TRUE(contents.trailer);
  EXPECT_EQ(contents.trailer->carriedRecords, 1u);
}

TEST(CaptureSession, RandomizedObservationsReplayExactly) {
  constexpr uint64_t ArenaBase = 0x80400000;
  constexpr uint32_t ArenaSize = 4096;
  constexpr uint32_t ChunkCount = 64;
  constexpr uint32_t ChunksPerFrame = 8;
  struct Read {
    uint64_t address;
    std::vector<uint8_t> bytes;
  };

  std::mt19937 rng{1234};
  const auto random = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>{lo, hi}(rng); };
  std::vector<uint8_t> arena(ArenaSize);
  std::ranges::generate(arena, [&] { return static_cast<uint8_t>(random(0, 3)); });
  std::vector<std::vector<Read>> chunks(ChunkCount);

  SessionFile file{"randomized"};
  for (uint32_t chunk = 0; chunk < ChunkCount; ++chunk) {
    if (chunk % ChunksPerFrame == 0) {
      file.session.begin_frame(chunk / ChunksPerFrame, {});
    }
    const uint32_t mutations = random(0, 4);
    for (uint32_t i = 0; i < mutations; ++i) {
      const uint32_t offset = random(0, ArenaSize - 1);
      const uint32_t length = random(1, std::min<uint32_t>(64, ArenaSize - offset));
      const auto value = static_cast<uint8_t>(random(0, 3));
      std::fill_n(arena.begin() + offset, length, value);
    }
    const uint32_t reads = random(1, 8);
    for (uint32_t i = 0; i < reads; ++i) {
      const uint32_t size = random(1, 512);
      const uint32_t offset = random(0, ArenaSize - size);
      Read read{ArenaBase + offset, {arena.begin() + offset, arena.begin() + offset + size}};
      file.session.observe(static_cast<uint8_t>(random(0, 3)), read.address, read.bytes);
      chunks[chunk].push_back(std::move(read));
    }
    const std::vector<uint8_t> marker{static_cast<uint8_t>(chunk)};
    file.session.processed(marker);
    if (chunk % ChunksPerFrame == ChunksPerFrame - 1) {
      std::string error;
      ASSERT_TRUE(file.session.end_frame(error)) << error;
    }
  }
  file.finish();

  SnapshotStore store;
  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(file.path(), error)) << error;
  const BlobSink sink = [&](const Hash128& hash, std::span<const uint8_t> bytes, std::string& sinkError) {
    return store.add_blob(hash, bytes, sinkError);
  };
  FrameView frame;
  uint32_t replayedChunks = 0;
  uint32_t mismatches = 0;
  while (reader.next(frame, sink, error) == TraceReader::Status::Frame) {
    for (const auto& record : frame.records) {
      if (record.type == RecordType::MemoryMap) {
        MemoryMapRecord map;
        ASSERT_TRUE(decode_memory_map(record.payload, map));
        ASSERT_TRUE(store.apply(map, error)) << error;
        continue;
      }
      store.begin_record();
      for (const uint8_t marker : record.payload) {
        ASSERT_EQ(marker, static_cast<uint8_t>(replayedChunks));
        for (const auto& read : chunks[replayedChunks]) {
          if (!bytes_equal(store.resolve(read.address, read.bytes.size()), read.bytes)) {
            ++mismatches;
          }
        }
        ++replayedChunks;
      }
    }
  }
  EXPECT_TRUE(error.empty()) << error;
  EXPECT_EQ(replayedChunks, ChunkCount);
  EXPECT_EQ(mismatches, 0u);
  EXPECT_EQ(store.misses(), 0u);
}
} // namespace
} // namespace aurora::trace

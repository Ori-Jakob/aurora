#include "trace/snapshot_store.hpp"
#include "trace/trace_format.hpp"
#include "trace/trace_reader.hpp"
#include "trace/trace_writer.hpp"

#include <SDL3/SDL_stdinc.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace aurora::trace {
namespace {
std::filesystem::path temp_trace(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "aurora_trace_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / (name + ".gxtrace");
  std::filesystem::remove(path);
  return path;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream in{path, std::ios::binary};
  return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

struct ChunkLocation {
  ChunkType type;
  size_t offset;
  ChunkHeader header;
};

std::vector<ChunkLocation> list_chunks(const std::vector<uint8_t>& file) {
  std::vector<ChunkLocation> chunks;
  size_t offset = FileHeaderSize;
  while (offset + ChunkHeaderSize <= file.size()) {
    ChunkHeader header;
    EXPECT_TRUE(decode_chunk_header({file.data() + offset, ChunkHeaderSize}, header));
    chunks.push_back({static_cast<ChunkType>(header.type), offset, header});
    offset += ChunkHeaderSize + header.storedSize;
  }
  return chunks;
}

std::vector<uint8_t> fifo_bytes(std::initializer_list<uint8_t> bytes) { return bytes; }

std::vector<uint8_t> pattern(size_t size, uint8_t seed) {
  std::vector<uint8_t> bytes(size);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<uint8_t>(seed + i * 7);
  }
  return bytes;
}

RenderConfig config_640() {
  return {.efbWidth = 640, .efbHeight = 480, .viFbWidth = 640, .viEfbHeight = 480, .viewportPolicy = 2};
}

struct ReadResult {
  std::vector<FrameView> frames;
  std::vector<std::pair<Hash128, std::vector<uint8_t>>> blobs;
  TraceReader::Status status = TraceReader::Status::Error;
  std::string error;
};

ReadResult read_all(TraceReader& reader) {
  ReadResult result;
  const BlobSink sink = [&](const Hash128& hash, std::span<const uint8_t> bytes, std::string&) {
    result.blobs.emplace_back(hash, std::vector<uint8_t>{bytes.begin(), bytes.end()});
    return true;
  };
  while (true) {
    FrameView frame;
    result.status = reader.next(frame, sink, result.error);
    if (result.status != TraceReader::Status::Frame) {
      break;
    }
    result.frames.push_back(std::move(frame));
  }
  return result;
}

struct TestFrame {
  RenderConfig config = config_640();
  std::vector<std::vector<uint8_t>> blobs;
  std::vector<std::pair<RecordType, std::vector<uint8_t>>> records;
};

MemoryMapRecord map_of(uint64_t address, const std::vector<uint8_t>& blob, uint8_t kind = 0) {
  return {.address = address, .size = static_cast<uint32_t>(blob.size()), .kind = kind, .blob = hash_bytes(blob)};
}

std::vector<uint8_t> encode_map(const MemoryMapRecord& record) {
  std::vector<uint8_t> out;
  LeWriter writer{out};
  encode_memory_map(writer, record);
  return out;
}

void write_trace(const std::filesystem::path& path, const std::vector<TestFrame>& frames, int32_t level = 0,
                 bool finalize = true) {
  TraceWriter writer;
  std::string error;
  ASSERT_TRUE(writer.open(path, level, error)) << error;
  TraceInfo info{.label = "test"};
  ASSERT_TRUE(writer.write_chunk(ChunkType::Info, encode_info(info), error)) << error;
  ASSERT_TRUE(writer.write_chunk(ChunkType::StatePreamble, {}, error)) << error;
  uint32_t blobCount = 0;
  for (uint32_t i = 0; i < frames.size(); ++i) {
    const auto& frame = frames[i];
    if (!frame.blobs.empty()) {
      BlobListBuilder blobs;
      for (const auto& blob : frame.blobs) {
        blobs.add(hash_bytes(blob), blob);
      }
      blobCount += static_cast<uint32_t>(frame.blobs.size());
      ASSERT_TRUE(writer.write_chunk(ChunkType::Blobs, blobs.finish(), error)) << error;
    }
    FrameBuilder builder;
    builder.begin(i, 100 + i, frame.config);
    for (const auto& [type, payload] : frame.records) {
      if (type == RecordType::Fifo) {
        builder.add_fifo(payload);
      } else {
        MemoryMapRecord record;
        ASSERT_TRUE(decode_memory_map(payload, record));
        builder.add_memory_map(record);
      }
    }
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, builder.finish(), error)) << error;
  }
  if (finalize) {
    Trailer trailer{.frameCount = static_cast<uint32_t>(frames.size()), .blobCount = blobCount};
    ASSERT_TRUE(writer.finalize(0, trailer.frameCount, blobCount, encode_trailer(trailer), error)) << error;
  }
}

class TraceFormatLevel : public ::testing::TestWithParam<int32_t> {};

TEST_P(TraceFormatLevel, RoundTripsAllChunkAndRecordTypes) {
  const auto path = temp_trace("round_trip_" + std::to_string(GetParam()));
  std::mt19937 rng{42};
  const std::vector<uint8_t> tiny{0x5A};
  std::vector<uint8_t> random(4096);
  std::generate(random.begin(), random.end(), [&] { return static_cast<uint8_t>(rng()); });
  const std::vector<uint8_t> zeros(4096, 0);
  std::vector<uint8_t> statePreamble(37);
  std::iota(statePreamble.begin(), statePreamble.end(), uint8_t{1});

  const TraceInfo info{
      .captureFlags = CaptureReplacementsBypassed | CaptureSynchronousPublish,
      .backend = BACKEND_VULKAN,
      .msaaSamples = 4,
      .maxAnisotropy = 8,
      .efbColorFormat = 0x16,
      .depthFormat = 0x2E,
      .deviceFeatures = DeviceCoreFeatures | DeviceBcTextures,
      .featureProfile = FEATURE_PROFILE_COMPAT,
      .resampler = SAMPLER_AREA,
      .requestedFrames = 2,
      .label = "\xC3\x98rdon \xE2\x9C\x93",
      .adapter = "Test Adapter",
  };
  const RenderConfig config0 = config_640();
  const RenderConfig config1{.efbWidth = 1920,
                             .efbHeight = 1440,
                             .viFbWidth = 608,
                             .viEfbHeight = 448,
                             .viewportPolicy = 0,
                             .normalBuffer = true};
  const auto map0 = map_of(0x80200000, tiny, 0);
  const auto map1 = map_of(0x80300000, random, 2);
  const auto map2 = map_of(0x80400000, zeros, 3);
  const auto fifo0 = fifo_bytes({0x61, 0x41, 0x12, 0x34, 0x56});
  const auto fifo1 = fifo_bytes({0x61, 0x42, 0x00, 0x00, 0x01});
  const auto fifo2 = fifo_bytes({0x08, 0x50, 0x00, 0x00, 0x02, 0x00});

  std::string error;
  {
    TraceWriter writer;
    ASSERT_TRUE(writer.open(path, GetParam(), error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::Info, encode_info(info), error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::StatePreamble, statePreamble, error)) << error;
    BlobListBuilder blobs;
    blobs.add(hash_bytes(tiny), tiny);
    blobs.add(hash_bytes(random), random);
    ASSERT_TRUE(writer.write_chunk(ChunkType::Blobs, blobs.finish(), error)) << error;
    FrameBuilder frame;
    frame.begin(0, 7, config0);
    frame.add_memory_map(map0);
    frame.add_memory_map(map1);
    frame.add_fifo(fifo0);
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, frame.finish(), error)) << error;
    blobs.clear();
    blobs.add(hash_bytes(zeros), zeros);
    ASSERT_TRUE(writer.write_chunk(ChunkType::Blobs, blobs.finish(), error)) << error;
    frame.begin(1, 8, config1);
    frame.add_fifo(fifo1);
    frame.add_memory_map(map2);
    frame.add_fifo(fifo2);
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, frame.finish(), error)) << error;
    const Trailer trailer{.frameCount = 2,
                          .blobCount = 3,
                          .blobBytes = 1 + 4096 + 4096,
                          .fifoBytes = 16,
                          .mapRecords = 3,
                          .carriedRecords = 0,
                          .endFlags = 0,
                          .reason = "done"};
    ASSERT_TRUE(writer.finalize(0, 2, 3, encode_trailer(trailer), error)) << error;
  }

  TraceReader reader;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_EQ(reader.header().flags & FileFinalized, FileFinalized);
  EXPECT_EQ(reader.header().frameCount, 2u);
  EXPECT_EQ(reader.header().blobCount, 3u);
  EXPECT_TRUE(reader.complete());
  const auto& readInfo = reader.info();
  EXPECT_EQ(readInfo.captureFlags, info.captureFlags);
  EXPECT_EQ(readInfo.backend, info.backend);
  EXPECT_EQ(readInfo.msaaSamples, info.msaaSamples);
  EXPECT_EQ(readInfo.maxAnisotropy, info.maxAnisotropy);
  EXPECT_EQ(readInfo.efbColorFormat, info.efbColorFormat);
  EXPECT_EQ(readInfo.depthFormat, info.depthFormat);
  EXPECT_EQ(readInfo.deviceFeatures, info.deviceFeatures);
  EXPECT_EQ(readInfo.featureProfile, info.featureProfile);
  EXPECT_EQ(readInfo.resampler, info.resampler);
  EXPECT_EQ(readInfo.requestedFrames, info.requestedFrames);
  EXPECT_EQ(readInfo.label, info.label);
  EXPECT_EQ(readInfo.adapter, info.adapter);
  EXPECT_TRUE(std::ranges::equal(reader.state_preamble(), statePreamble));

  const auto result = read_all(reader);
  ASSERT_EQ(result.status, TraceReader::Status::End) << result.error;
  ASSERT_EQ(result.frames.size(), 2u);
  ASSERT_EQ(result.blobs.size(), 3u);
  EXPECT_EQ(result.blobs[0].second, tiny);
  EXPECT_EQ(result.blobs[1].second, random);
  EXPECT_EQ(result.blobs[2].second, zeros);
  EXPECT_EQ(result.blobs[2].first, hash_bytes(zeros));

  const auto& frame0 = result.frames[0];
  EXPECT_EQ(frame0.frameIndex, 0u);
  EXPECT_EQ(frame0.sourceFrame, 7u);
  EXPECT_EQ(frame0.config, config0);
  ASSERT_EQ(frame0.records.size(), 3u);
  EXPECT_EQ(frame0.records[0].type, RecordType::MemoryMap);
  EXPECT_TRUE(std::ranges::equal(frame0.records[0].payload, encode_map(map0)));
  EXPECT_EQ(frame0.records[1].type, RecordType::MemoryMap);
  EXPECT_TRUE(std::ranges::equal(frame0.records[1].payload, encode_map(map1)));
  EXPECT_EQ(frame0.records[2].type, RecordType::Fifo);
  EXPECT_TRUE(std::ranges::equal(frame0.records[2].payload, fifo0));

  const auto& frame1 = result.frames[1];
  EXPECT_EQ(frame1.frameIndex, 1u);
  EXPECT_EQ(frame1.sourceFrame, 8u);
  EXPECT_EQ(frame1.config, config1);
  ASSERT_EQ(frame1.records.size(), 3u);
  EXPECT_EQ(frame1.records[0].type, RecordType::Fifo);
  EXPECT_TRUE(std::ranges::equal(frame1.records[0].payload, fifo1));
  EXPECT_EQ(frame1.records[1].type, RecordType::MemoryMap);
  MemoryMapRecord decoded;
  ASSERT_TRUE(decode_memory_map(frame1.records[1].payload, decoded));
  EXPECT_EQ(decoded.address, map2.address);
  EXPECT_EQ(decoded.size, map2.size);
  EXPECT_EQ(decoded.kind, map2.kind);
  EXPECT_EQ(decoded.blob, map2.blob);
  EXPECT_EQ(frame1.records[2].type, RecordType::Fifo);
  EXPECT_TRUE(std::ranges::equal(frame1.records[2].payload, fifo2));

  ASSERT_TRUE(reader.trailer().has_value());
  EXPECT_EQ(reader.trailer()->frameCount, 2u);
  EXPECT_EQ(reader.trailer()->blobCount, 3u);
  EXPECT_EQ(reader.trailer()->blobBytes, 1u + 4096 + 4096);
  EXPECT_EQ(reader.trailer()->fifoBytes, 16u);
  EXPECT_EQ(reader.trailer()->mapRecords, 3u);
  EXPECT_EQ(reader.trailer()->reason, "done");

  ASSERT_TRUE(reader.rewind(error)) << error;
  const auto again = read_all(reader);
  EXPECT_EQ(again.status, TraceReader::Status::End) << again.error;
  EXPECT_EQ(again.frames.size(), 2u);
}

INSTANTIATE_TEST_SUITE_P(TraceFormat, TraceFormatLevel, ::testing::Values(0, 3));

TEST(TraceFormat, WriterMergesConsecutiveFifoRecords) {
  const auto path = temp_trace("merge_fifo");
  const auto blob = pattern(32, 3);
  TestFrame frame;
  frame.blobs = {blob};
  frame.records = {
      {RecordType::Fifo, fifo_bytes({1, 2})},    {RecordType::Fifo, fifo_bytes({3})},
      {RecordType::Fifo, fifo_bytes({4, 5, 6})}, {RecordType::MemoryMap, encode_map(map_of(0x1000, blob))},
      {RecordType::Fifo, fifo_bytes({7})},
  };
  write_trace(path, {frame});

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  const auto result = read_all(reader);
  ASSERT_EQ(result.status, TraceReader::Status::End) << result.error;
  ASSERT_EQ(result.frames.size(), 1u);
  const auto& records = result.frames[0].records;
  ASSERT_EQ(records.size(), 3u);
  EXPECT_EQ(records[0].type, RecordType::Fifo);
  EXPECT_TRUE(std::ranges::equal(records[0].payload, fifo_bytes({1, 2, 3, 4, 5, 6})));
  EXPECT_EQ(records[1].type, RecordType::MemoryMap);
  EXPECT_EQ(records[2].type, RecordType::Fifo);
  EXPECT_TRUE(std::ranges::equal(records[2].payload, fifo_bytes({7})));
}

TEST(TraceFormat, ChecksumMismatchIsCorrupt) {
  const auto path = temp_trace("checksum");
  TestFrame frame;
  frame.records = {{RecordType::Fifo, fifo_bytes({0x61, 0x41, 0, 0, 1})}};
  write_trace(path, {frame});

  auto file = read_bytes(path);
  const auto chunks = list_chunks(file);
  const auto fram = std::ranges::find(chunks, ChunkType::Frame, &ChunkLocation::type);
  ASSERT_NE(fram, chunks.end());
  file[fram->offset + ChunkHeaderSize + fram->header.storedSize - 1] ^= 0xFF;
  write_bytes(path, file);

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  FrameView view;
  EXPECT_EQ(reader.next(view, {}, error), TraceReader::Status::Error);
  EXPECT_NE(error.find("checksum"), std::string::npos) << error;
}

TEST(TraceFormat, RejectsUnknownMajorVersion) {
  const auto path = temp_trace("major_version");
  write_trace(path, {TestFrame{}});
  auto file = read_bytes(path);
  file[8] = 2;
  file[9] = 0;
  write_bytes(path, file);

  TraceReader reader;
  std::string error;
  EXPECT_FALSE(reader.open(path, error));
  EXPECT_NE(error.find("unsupported trace version"), std::string::npos) << error;
}

TEST(TraceFormat, SkipsUnknownChunksAndRecords) {
  const auto path = temp_trace("unknown_chunks");
  std::string error;
  {
    TraceWriter writer;
    ASSERT_TRUE(writer.open(path, 0, error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::Info, encode_info({}), error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::StatePreamble, {}, error)) << error;
    const auto unknown = pattern(11, 9);
    ASSERT_TRUE(writer.write_chunk(static_cast<ChunkType>(fourcc('Z', 'Z', 'Z', 'Z')), unknown, error)) << error;

    std::vector<uint8_t> records;
    LeWriter out{records};
    out.u8(9);
    out.u8(0);
    out.u16(0);
    out.u32(3);
    out.bytes(fifo_bytes({0xDE, 0xAD, 0x01}));
    FrameBuilder frame;
    frame.begin(0, 0, config_640());
    frame.add_encoded_records(records, 1);
    frame.add_fifo(fifo_bytes({0x61, 0x41, 0, 0, 1}));
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, frame.finish(), error)) << error;
    ASSERT_TRUE(writer.write_chunk(static_cast<ChunkType>(fourcc('Z', 'Z', 'Z', 'Z')), unknown, error)) << error;
    frame.begin(1, 1, config_640());
    frame.add_fifo(fifo_bytes({0x61, 0x42, 0, 0, 1}));
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, frame.finish(), error)) << error;
    const Trailer trailer{.frameCount = 2};
    ASSERT_TRUE(writer.finalize(0, 2, 0, encode_trailer(trailer), error)) << error;
  }

  TraceReader reader;
  ASSERT_TRUE(reader.open(path, error)) << error;
  const auto result = read_all(reader);
  ASSERT_EQ(result.status, TraceReader::Status::End) << result.error;
  ASSERT_EQ(result.frames.size(), 2u);
  ASSERT_EQ(result.frames[0].records.size(), 1u);
  EXPECT_EQ(result.frames[0].records[0].type, RecordType::Fifo);
  ASSERT_EQ(result.frames[1].records.size(), 1u);
  EXPECT_TRUE(std::ranges::equal(result.frames[1].records[0].payload, fifo_bytes({0x61, 0x42, 0, 0, 1})));
}

TEST(TraceFormat, ReadsUnfinalizedTraceUpToLastCompleteFrame) {
  const auto path = temp_trace("unfinalized");
  std::vector<TestFrame> frames(3);
  for (uint32_t i = 0; i < frames.size(); ++i) {
    frames[i].records = {{RecordType::Fifo, pattern(64, static_cast<uint8_t>(i))}};
  }
  write_trace(path, frames);

  auto file = read_bytes(path);
  const auto chunks = list_chunks(file);
  std::vector<ChunkLocation> frameChunks;
  std::ranges::copy_if(chunks, std::back_inserter(frameChunks),
                       [](const ChunkLocation& chunk) { return chunk.type == ChunkType::Frame; });
  ASSERT_EQ(frameChunks.size(), 3u);
  file.resize(frameChunks[2].offset + ChunkHeaderSize + frameChunks[2].header.storedSize / 2);
  file[16] &= static_cast<uint8_t>(~FileFinalized);
  write_bytes(path, file);

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_FALSE(reader.complete());
  const auto result = read_all(reader);
  EXPECT_EQ(result.status, TraceReader::Status::End) << result.error;
  EXPECT_EQ(result.frames.size(), 2u);
  EXPECT_FALSE(reader.trailer().has_value());
}

TEST(TraceFormat, MemoryMapMustReferenceKnownBlob) {
  const auto path = temp_trace("unknown_blob");
  TestFrame frame;
  frame.records = {{RecordType::MemoryMap, encode_map(map_of(0x1000, pattern(16, 1)))}};
  write_trace(path, {frame});

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  const auto result = read_all(reader);
  EXPECT_EQ(result.status, TraceReader::Status::Error);
  EXPECT_NE(result.error.find("unknown blob"), std::string::npos) << result.error;
}

TEST(TraceFormat, RejectsFramesWithoutRecordsOrWithImpossibleCounts) {
  for (const uint32_t recordCount : {0u, 0xFFFFFFFFu}) {
    const auto path = temp_trace("record_count_" + std::to_string(recordCount));
    TraceWriter writer;
    std::string error;
    ASSERT_TRUE(writer.open(path, 0, error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::Info, encode_info(TraceInfo{.label = "test"}), error)) << error;
    ASSERT_TRUE(writer.write_chunk(ChunkType::StatePreamble, {}, error)) << error;
    std::vector<uint8_t> payload;
    LeWriter frame{payload};
    frame.u32(0);
    frame.u32(0);
    frame.u32(recordCount);
    ASSERT_TRUE(writer.write_chunk(ChunkType::Frame, payload, error)) << error;
    const Trailer trailer{.frameCount = 1};
    ASSERT_TRUE(writer.finalize(0, 1, 0, encode_trailer(trailer), error)) << error;

    TraceReader reader;
    ASSERT_TRUE(reader.open(path, error)) << error;
    const auto result = read_all(reader);
    EXPECT_EQ(result.status, TraceReader::Status::Error) << recordCount;
    EXPECT_NE(result.error.find("frame 0"), std::string::npos) << result.error;
  }
}

TEST(TraceFormat, ReadsExternalCaptureFromEnvironment) {
  const char* file = SDL_getenv("AURORA_TRACE_FILE");
  if (file == nullptr || *file == '\0') {
    GTEST_SKIP() << "AURORA_TRACE_FILE is not set";
  }
  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(file, error)) << error;
  SnapshotStore store;
  const BlobSink sink = [&](const Hash128& hash, std::span<const uint8_t> bytes, std::string& sinkError) {
    return store.add_blob(hash, bytes, sinkError);
  };
  uint32_t frames = 0;
  FrameView frame;
  TraceReader::Status status;
  while ((status = reader.next(frame, sink, error)) == TraceReader::Status::Frame) {
    EXPECT_EQ(frame.frameIndex, frames);
    for (const auto& record : frame.records) {
      if (record.type == RecordType::MemoryMap) {
        MemoryMapRecord map;
        ASSERT_TRUE(decode_memory_map(record.payload, map));
        ASSERT_TRUE(store.apply(map, error)) << error;
      }
    }
    ++frames;
  }
  ASSERT_EQ(status, TraceReader::Status::End) << error;
  EXPECT_GT(frames, 0u);
  if (reader.trailer()) {
    EXPECT_EQ(reader.trailer()->frameCount, frames);
  }
}
} // namespace
} // namespace aurora::trace

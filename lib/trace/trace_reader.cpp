#include "trace_reader.hpp"

#include <SDL3/SDL_error.h>
#include <fmt/format.h>

#include <array>

#ifdef AURORA_TRACE_ZSTD
#include <zstd.h>
#endif

namespace aurora::trace {
namespace {
std::string chunk_name(uint32_t type) {
  std::string name(4, '?');
  for (size_t i = 0; i < name.size(); ++i) {
    const auto c = static_cast<char>(type >> (i * 8));
    if (c >= 0x20 && c < 0x7F) {
      name[i] = c;
    }
  }
  return name;
}
} // namespace

bool TraceReader::open(const std::filesystem::path& path, std::string& error) {
  *this = {};
  m_file = io::open_file(path, "rb");
  if (!m_file) {
    error = fmt::format("failed to open {}: {}", io::fs_path_to_string(path), SDL_GetError());
    return false;
  }
  const Sint64 size = SDL_GetIOSize(m_file.get());
  if (size < 0) {
    error = fmt::format("failed to read {}: {}", io::fs_path_to_string(path), SDL_GetError());
    return false;
  }
  m_fileSize = static_cast<uint64_t>(size);

  std::array<uint8_t, FileHeaderSize> headerBytes{};
  if (m_fileSize < FileHeaderSize || !io::read_at(m_file.get(), 0, headerBytes.data(), headerBytes.size())) {
    error = "not a GX trace";
    return false;
  }
  if (!decode_file_header(headerBytes, m_header, error)) {
    return false;
  }
  m_offset = FileHeaderSize;

  ChunkHeader chunk;
  std::vector<uint8_t> payload;
  auto status = read_chunk(chunk, payload, error);
  if (status == ChunkStatus::Truncated) {
    error = "truncated trace: missing INFO chunk";
  }
  if (status != ChunkStatus::Ok) {
    return false;
  }
  if (chunk.type != static_cast<uint32_t>(ChunkType::Info) || !decode_info(payload, m_info)) {
    error = "corrupt trace: the first chunk must be INFO";
    return false;
  }

  status = read_chunk(chunk, payload, error);
  if (status == ChunkStatus::Truncated) {
    error = "truncated trace: missing STAT chunk";
  }
  if (status != ChunkStatus::Ok) {
    return false;
  }
  if (chunk.type != static_cast<uint32_t>(ChunkType::StatePreamble)) {
    error = "corrupt trace: the second chunk must be STAT";
    return false;
  }
  m_statePreamble = std::move(payload);
  m_firstFrameOffset = m_offset;
  return true;
}

bool TraceReader::complete() const noexcept {
  return (m_header.flags & FileFinalized) != 0 && (m_header.flags & FileAborted) == 0;
}

TraceReader::ChunkStatus TraceReader::read_chunk(ChunkHeader& header, std::vector<uint8_t>& payload,
                                                 std::string& error) {
  const uint64_t chunkOffset = m_offset;
  if (m_fileSize - m_offset < ChunkHeaderSize) {
    return ChunkStatus::Truncated;
  }
  std::array<uint8_t, ChunkHeaderSize> headerBytes{};
  if (!io::read_at(m_file.get(), m_offset, headerBytes.data(), headerBytes.size()) ||
      !decode_chunk_header(headerBytes, header)) {
    error = fmt::format("failed to read trace chunk at offset {}: {}", chunkOffset, SDL_GetError());
    return ChunkStatus::Error;
  }
  if (header.storedSize > m_fileSize - m_offset - ChunkHeaderSize) {
    return ChunkStatus::Truncated;
  }
  const auto name = chunk_name(header.type);
  if (header.rawSize > MaxChunkRawSize || header.storedSize > MaxChunkRawSize) {
    error = fmt::format("corrupt trace: {} chunk at offset {} is {} bytes", name, chunkOffset, header.rawSize);
    return ChunkStatus::Error;
  }
  m_stored.resize(header.storedSize);
  if (!io::read_exact(m_file.get(), m_stored.data(), m_stored.size())) {
    error = fmt::format("failed to read trace chunk at offset {}: {}", chunkOffset, SDL_GetError());
    return ChunkStatus::Error;
  }

  switch (header.compression) {
  case Compression::None:
    if (header.storedSize != header.rawSize) {
      error = fmt::format("corrupt trace: {} chunk at offset {} has mismatched sizes", name, chunkOffset);
      return ChunkStatus::Error;
    }
    payload = m_stored;
    break;
  case Compression::Zstd: {
#ifdef AURORA_TRACE_ZSTD
    payload.resize(header.rawSize);
    const size_t result = ZSTD_decompress(payload.data(), payload.size(), m_stored.data(), m_stored.size());
    if (ZSTD_isError(result) || result != header.rawSize) {
      error = fmt::format("corrupt trace: {} chunk at offset {} does not decompress to {} bytes", name, chunkOffset,
                          header.rawSize);
      return ChunkStatus::Error;
    }
    break;
#else
    error =
        fmt::format("{} chunk at offset {} is zstd-compressed, but this build has no zstd support", name, chunkOffset);
    return ChunkStatus::Error;
#endif
  }
  default:
    error = fmt::format("corrupt trace: {} chunk at offset {} uses unknown compression {}", name, chunkOffset,
                        static_cast<uint32_t>(header.compression));
    return ChunkStatus::Error;
  }

  if (checksum_bytes(payload) != header.checksum) {
    error = fmt::format("corrupt trace: {} chunk at offset {} fails its checksum", name, chunkOffset);
    return ChunkStatus::Error;
  }
  m_offset += ChunkHeaderSize + header.storedSize;
  return ChunkStatus::Ok;
}

TraceReader::Status TraceReader::next(FrameView& frame, const BlobSink& sink, std::string& error) {
  while (!m_ended) {
    ChunkHeader chunk;
    std::vector<uint8_t> payload;
    const auto status = read_chunk(chunk, payload, error);
    if (status == ChunkStatus::Error) {
      return Status::Error;
    }
    if (status == ChunkStatus::Truncated) {
      if ((m_header.flags & FileFinalized) != 0) {
        error = fmt::format("truncated trace: no ENDT chunk after {} frames", m_framesRead);
        return Status::Error;
      }
      m_ended = true;
      break;
    }

    switch (static_cast<ChunkType>(chunk.type)) {
    case ChunkType::Blobs:
      if (!parse_blobs(payload, sink, error)) {
        return Status::Error;
      }
      break;
    case ChunkType::Frame:
      if (!parse_frame(std::move(payload), frame, error)) {
        return Status::Error;
      }
      ++m_framesRead;
      return Status::Frame;
    case ChunkType::End: {
      Trailer trailer;
      if (!decode_trailer(payload, trailer)) {
        error = "corrupt trace: unreadable ENDT chunk";
        return Status::Error;
      }
      if ((m_header.flags & FileFinalized) != 0 && trailer.frameCount != m_framesRead) {
        error = fmt::format("corrupt trace: ENDT lists {} frames but {} were read", trailer.frameCount, m_framesRead);
        return Status::Error;
      }
      m_trailer = std::move(trailer);
      m_ended = true;
      break;
    }
    case ChunkType::Info:
    case ChunkType::StatePreamble:
      error = fmt::format("corrupt trace: unexpected {} chunk", chunk_name(chunk.type));
      return Status::Error;
    default:
      break;
    }
  }
  return Status::End;
}

bool TraceReader::parse_blobs(std::span<const uint8_t> payload, const BlobSink& sink, std::string& error) {
  LeReader reader{payload};
  uint32_t count = 0;
  if (!reader.u32(count)) {
    error = "corrupt trace: unreadable BLOB chunk";
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    Hash128 hash;
    uint32_t size = 0;
    std::span<const uint8_t> bytes;
    if (!reader.hash(hash) || !reader.u32(size) || !reader.bytes(size, bytes)) {
      error = fmt::format("corrupt trace: BLOB entry {} of {} overruns its chunk", i, count);
      return false;
    }
    m_blobSizes[hash] = size;
    if (sink && !sink(hash, bytes, error)) {
      return false;
    }
  }
  return true;
}

bool TraceReader::parse_frame(std::vector<uint8_t>&& payload, FrameView& frame, std::string& error) {
  frame.storage = std::move(payload);
  frame.records.clear();
  LeReader reader{frame.storage};
  uint32_t recordCount = 0;
  if (!reader.u32(frame.frameIndex) || !reader.u32(frame.sourceFrame) || !reader.u32(recordCount)) {
    error = "corrupt trace: unreadable FRAM header";
    return false;
  }
  if (frame.frameIndex != m_framesRead) {
    error = fmt::format("corrupt trace: expected frame {} but found frame {}", m_framesRead, frame.frameIndex);
    return false;
  }

  frame.records.reserve(recordCount);
  for (uint32_t i = 0; i < recordCount; ++i) {
    uint8_t type = 0;
    uint8_t reserved8 = 0;
    uint16_t reserved16 = 0;
    uint32_t size = 0;
    std::span<const uint8_t> record;
    if (!reader.u8(type) || !reader.u8(reserved8) || !reader.u16(reserved16) || !reader.u32(size) ||
        !reader.bytes(size, record)) {
      error = fmt::format("corrupt trace: record {} overruns frame {}", i, frame.frameIndex);
      return false;
    }
    if (i == 0) {
      if (type != static_cast<uint8_t>(RecordType::RenderConfig) || !decode_render_config(record, frame.config)) {
        error = fmt::format("corrupt trace: frame {} does not start with RENDER_CONFIG", frame.frameIndex);
        return false;
      }
      continue;
    }

    switch (static_cast<RecordType>(type)) {
    case RecordType::RenderConfig: {
      RenderConfig config;
      if (!decode_render_config(record, config)) {
        error = fmt::format("corrupt trace: short RENDER_CONFIG in frame {}", frame.frameIndex);
        return false;
      }
      break;
    }
    case RecordType::MemoryMap: {
      MemoryMapRecord map;
      if (!decode_memory_map(record, map)) {
        error = fmt::format("corrupt trace: short MEMORY_MAP in frame {}", frame.frameIndex);
        return false;
      }
      const auto blob = m_blobSizes.find(map.blob);
      if (blob == m_blobSizes.end()) {
        error = fmt::format("corrupt trace: MEMORY_MAP at 0x{:X} in frame {} names an unknown blob", map.address,
                            frame.frameIndex);
        return false;
      }
      if (blob->second != map.size) {
        error = fmt::format("corrupt trace: MEMORY_MAP at 0x{:X} in frame {} is {} bytes but its blob is {}",
                            map.address, frame.frameIndex, map.size, blob->second);
        return false;
      }
      break;
    }
    case RecordType::Fifo:
      break;
    default:
      continue;
    }
    frame.records.push_back({static_cast<RecordType>(type), record});
  }
  return true;
}

bool TraceReader::rewind(std::string& error) {
  if (!m_file) {
    error = "trace is not open";
    return false;
  }
  m_offset = m_firstFrameOffset;
  m_framesRead = 0;
  m_ended = false;
  m_trailer.reset();
  m_blobSizes.clear();
  return true;
}
} // namespace aurora::trace

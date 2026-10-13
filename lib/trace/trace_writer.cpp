#include "trace_writer.hpp"

#include <SDL3/SDL_error.h>
#include <fmt/format.h>

#include <array>

#ifdef AURORA_TRACE_ZSTD
#include <zstd.h>
#endif

namespace aurora::trace {
TraceWriter::~TraceWriter() { discard(); }

bool TraceWriter::open(const std::filesystem::path& path, int32_t compressionLevel, std::string& error) {
  discard();
  m_file = io::open_atomic_file(path, io::AtomicFileMode::Truncate);
  if (!m_file) {
    error = fmt::format("failed to open {}: {}", io::fs_path_to_string(path), SDL_GetError());
    return false;
  }
  m_path = path;
  m_compressionLevel = compressionLevel;
  m_offset = 0;
  m_chunkCount = 0;
  constexpr std::array<uint8_t, FileHeaderSize> placeholder{};
  if (!write(placeholder, error)) {
    discard();
    return false;
  }
  return true;
}

bool TraceWriter::write(std::span<const uint8_t> bytes, std::string& error) {
  if (!io::write_exact(m_file.get(), bytes.data(), bytes.size())) {
    error = fmt::format("failed to write {}: {}", io::fs_path_to_string(m_path), SDL_GetError());
    return false;
  }
  m_offset += bytes.size();
  return true;
}

bool TraceWriter::write_chunk(ChunkType type, std::span<const uint8_t> payload, std::string& error) {
  if (!m_file) {
    error = "trace file is not open";
    return false;
  }
  if (payload.size() > MaxChunkRawSize) {
    error = fmt::format("{}-byte chunk exceeds the {}-byte trace chunk limit", payload.size(), MaxChunkRawSize);
    return false;
  }

  ChunkHeader header{
      .type = static_cast<uint32_t>(type),
      .rawSize = static_cast<uint32_t>(payload.size()),
      .checksum = checksum_bytes(payload),
  };
  std::span<const uint8_t> stored = payload;
#ifdef AURORA_TRACE_ZSTD
  if (m_compressionLevel > 0 && !payload.empty()) {
    m_scratch.resize(ZSTD_compressBound(payload.size()));
    const size_t compressed =
        ZSTD_compress(m_scratch.data(), m_scratch.size(), payload.data(), payload.size(), m_compressionLevel);
    if (!ZSTD_isError(compressed) && compressed < payload.size()) {
      header.compression = Compression::Zstd;
      stored = {m_scratch.data(), compressed};
    }
  }
#endif
  header.storedSize = static_cast<uint32_t>(stored.size());

  std::vector<uint8_t> headerBytes;
  headerBytes.reserve(ChunkHeaderSize);
  LeWriter out{headerBytes};
  encode_chunk_header(out, header);
  if (!write(headerBytes, error) || !write(stored, error)) {
    return false;
  }
  ++m_chunkCount;
  return true;
}

bool TraceWriter::finalize(uint32_t fileFlags, uint32_t frameCount, uint32_t blobCount,
                           std::span<const uint8_t> trailer, std::string& error) {
  const uint64_t trailerOffset = m_offset;
  if (!write_chunk(ChunkType::End, trailer, error)) {
    return false;
  }
  FileHeader header;
  header.flags = fileFlags | FileFinalized;
  header.frameCount = frameCount;
  header.blobCount = blobCount;
  header.chunkCount = m_chunkCount;
  header.trailerOffset = trailerOffset;
  const auto headerBytes = encode_file_header(header);
  if (!io::write_at(m_file.get(), 0, headerBytes.data(), headerBytes.size())) {
    error = fmt::format("failed to write {}: {}", io::fs_path_to_string(m_path), SDL_GetError());
    return false;
  }
  if (!m_file.commit()) {
    error = fmt::format("failed to commit {}: {}", io::fs_path_to_string(m_path), SDL_GetError());
    return false;
  }
  m_file = {};
  return true;
}

void TraceWriter::discard() noexcept {
  m_file = {};
  m_offset = 0;
  m_chunkCount = 0;
}

void FrameBuilder::begin(uint32_t frameIndex, uint32_t sourceFrame, const RenderConfig& config) {
  clear();
  LeWriter out{m_data};
  out.u32(frameIndex);
  out.u32(sourceFrame);
  out.u32(0);
  out.u8(static_cast<uint8_t>(RecordType::RenderConfig));
  out.u8(0);
  out.u16(0);
  out.u32(RenderConfigSize);
  encode_render_config(out, config);
  m_recordCount = 1;
}

void FrameBuilder::add_memory_map(const MemoryMapRecord& record) {
  LeWriter out{m_data};
  out.u8(static_cast<uint8_t>(RecordType::MemoryMap));
  out.u8(0);
  out.u16(0);
  out.u32(MemoryMapSize);
  encode_memory_map(out, record);
  ++m_recordCount;
  m_lastFifoRecord = NoFifoRecord;
}

void FrameBuilder::add_fifo(std::span<const uint8_t> fifo) {
  if (fifo.empty()) {
    return;
  }
  LeWriter out{m_data};
  if (m_lastFifoRecord != NoFifoRecord) {
    m_lastFifoSize += static_cast<uint32_t>(fifo.size());
    out.patch_u32(m_lastFifoRecord + 4, m_lastFifoSize);
    out.bytes(fifo);
    return;
  }
  m_lastFifoRecord = m_data.size();
  m_lastFifoSize = static_cast<uint32_t>(fifo.size());
  out.u8(static_cast<uint8_t>(RecordType::Fifo));
  out.u8(0);
  out.u16(0);
  out.u32(m_lastFifoSize);
  out.bytes(fifo);
  ++m_recordCount;
}

void FrameBuilder::add_encoded_records(std::span<const uint8_t> records, uint32_t count) {
  LeWriter out{m_data};
  out.bytes(records);
  m_recordCount += count;
  m_lastFifoRecord = NoFifoRecord;
}

std::span<const uint8_t> FrameBuilder::finish() {
  LeWriter out{m_data};
  out.patch_u32(8, m_recordCount);
  return m_data;
}

void FrameBuilder::clear() noexcept {
  m_data.clear();
  m_recordCount = 0;
  m_lastFifoRecord = NoFifoRecord;
  m_lastFifoSize = 0;
}

void BlobListBuilder::add(const Hash128& hash, std::span<const uint8_t> bytes) {
  LeWriter out{m_data};
  if (m_data.empty()) {
    out.u32(0);
  }
  out.hash(hash);
  out.u32(static_cast<uint32_t>(bytes.size()));
  out.bytes(bytes);
  ++m_count;
  m_rawBytes += bytes.size();
}

std::span<const uint8_t> BlobListBuilder::finish() {
  if (m_data.empty()) {
    LeWriter{m_data}.u32(0);
  }
  LeWriter{m_data}.patch_u32(0, m_count);
  return m_data;
}

void BlobListBuilder::clear() noexcept {
  m_data.clear();
  m_count = 0;
  m_rawBytes = 0;
}
} // namespace aurora::trace

#include "capture_session.hpp"

#include "../internal.hpp"

#include <limits>

namespace aurora::trace {
namespace {
constexpr Module Log{"aurora::trace::session"};
} // namespace

bool CaptureSession::open(const std::filesystem::path& path, int32_t compressionLevel, uint32_t fileFlags,
                          std::string& error) {
  reset_state();
  m_fileFlags = fileFlags;
  return m_writer.open(path, compressionLevel, error);
}

bool CaptureSession::write_header_chunks(const TraceInfo& info, std::span<const uint8_t> statePreamble,
                                         std::string& error) {
  const auto infoBytes = encode_info(info);
  return m_writer.write_chunk(ChunkType::Info, infoBytes, error) &&
         m_writer.write_chunk(ChunkType::StatePreamble, statePreamble, error);
}

void CaptureSession::begin_frame(uint32_t sourceFrame, const RenderConfig& config) {
  m_frame.begin(m_trailer.frameCount, sourceFrame, config);
  if (m_carryCount != 0) {
    m_frame.add_encoded_records(m_carry, m_carryCount);
    m_trailer.carriedRecords += m_carryCount;
    m_carry.clear();
    m_carryCount = 0;
  }
  m_inFrame = true;
}

void CaptureSession::observe(uint8_t kind, uint64_t address, std::span<const uint8_t> bytes) {
  if (bytes.empty() || !m_writer.is_open()) {
    return;
  }
  if (bytes.size() > std::numeric_limits<uint32_t>::max() ||
      bytes.size() > std::numeric_limits<uint64_t>::max() - address) {
    Log.warn("Ignoring {}-byte guest read at 0x{:X}: the range cannot be recorded", bytes.size(), address);
    return;
  }

  const Hash128 hash = hash_bytes(bytes);
  const auto [it, inserted] = m_blobIds.try_emplace(hash, static_cast<uint32_t>(m_blobIds.size()));
  if (inserted) {
    m_blobs.add(hash, bytes);
    // A frame that streams a lot of new content would otherwise build a chunk past the format limit
    if (m_blobs.raw_bytes() >= MaxBlobChunkBytes && m_error.empty()) {
      flush_blobs(m_error);
    }
  }
  const uint32_t blob = it->second;
  if (m_mirror.covers_exactly(address, bytes.size(), blob)) {
    return;
  }

  const MemoryMapRecord record{
      .address = address,
      .size = static_cast<uint32_t>(bytes.size()),
      .kind = kind,
      .blob = hash,
  };
  if (m_inFrame) {
    m_frame.add_memory_map(record);
  } else {
    std::vector<uint8_t> payload;
    LeWriter out{payload};
    encode_memory_map(out, record);
    add_record(RecordType::MemoryMap, payload);
  }
  m_mirror.assign(address, bytes.size(), blob);
  ++m_trailer.mapRecords;
}

void CaptureSession::processed(std::span<const uint8_t> fifo) {
  if (fifo.empty() || !m_writer.is_open()) {
    return;
  }
  m_trailer.fifoBytes += fifo.size();
  if (m_inFrame) {
    m_frame.add_fifo(fifo);
  } else {
    add_record(RecordType::Fifo, fifo);
  }
}

void CaptureSession::add_record(RecordType type, std::span<const uint8_t> payload) {
  LeWriter out{m_carry};
  out.u8(static_cast<uint8_t>(type));
  out.u8(0);
  out.u16(0);
  out.u32(static_cast<uint32_t>(payload.size()));
  out.bytes(payload);
  ++m_carryCount;
  if (!m_carryLogged) {
    m_carryLogged = true;
    Log.warn("GX commands were decoded between frames; carrying them into the next captured frame");
  }
}

bool CaptureSession::flush_blobs(std::string& error) {
  if (m_blobs.empty()) {
    return true;
  }
  const uint32_t count = m_blobs.count();
  const uint64_t rawBytes = m_blobs.raw_bytes();
  if (!m_writer.write_chunk(ChunkType::Blobs, m_blobs.finish(), error)) {
    return false;
  }
  m_trailer.blobCount += count;
  m_trailer.blobBytes += rawBytes;
  m_blobs.clear();
  return true;
}

bool CaptureSession::end_frame(std::string& error) {
  if (!m_inFrame) {
    error = "no captured frame is open";
    return false;
  }
  m_inFrame = false;
  if (!m_error.empty()) {
    error = m_error;
    return false;
  }
  if (!flush_blobs(error) || !m_writer.write_chunk(ChunkType::Frame, m_frame.finish(), error)) {
    return false;
  }
  m_frame.clear();
  ++m_trailer.frameCount;
  return true;
}

bool CaptureSession::finalize(uint32_t endFlags, std::string_view reason, std::string& error) {
  if (m_inFrame) {
    error = "a captured frame is still open";
    return false;
  }
  if (!m_error.empty()) {
    error = m_error;
    return false;
  }
  m_trailer.endFlags = endFlags;
  m_trailer.reason = reason;
  const uint32_t fileFlags = (endFlags & EndAborted) != 0 ? m_fileFlags | FileAborted : m_fileFlags;
  const auto trailer = encode_trailer(m_trailer);
  if (!m_writer.finalize(fileFlags, m_trailer.frameCount, m_trailer.blobCount, trailer, error)) {
    return false;
  }
  reset_state();
  return true;
}

void CaptureSession::discard() noexcept {
  m_writer.discard();
  reset_state();
}

void CaptureSession::reset_state() noexcept {
  m_frame.clear();
  m_blobs.clear();
  m_blobIds.clear();
  m_mirror.clear();
  m_carry.clear();
  m_carryCount = 0;
  m_carryLogged = false;
  m_inFrame = false;
  m_trailer = {};
  m_error.clear();
}
} // namespace aurora::trace

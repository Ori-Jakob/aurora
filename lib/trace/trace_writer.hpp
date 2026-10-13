#pragma once

#include "trace_format.hpp"

#include "../io.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace aurora::trace {
class TraceWriter {
public:
  TraceWriter() = default;
  ~TraceWriter();
  TraceWriter(const TraceWriter&) = delete;
  TraceWriter& operator=(const TraceWriter&) = delete;

  bool open(const std::filesystem::path& path, int32_t compressionLevel, std::string& error);
  bool write_chunk(ChunkType type, std::span<const uint8_t> payload, std::string& error);
  bool finalize(uint32_t fileFlags, uint32_t frameCount, uint32_t blobCount, std::span<const uint8_t> trailer,
                std::string& error);
  void discard() noexcept;
  [[nodiscard]] bool is_open() const noexcept { return static_cast<bool>(m_file); }
  [[nodiscard]] uint64_t bytes_written() const noexcept { return m_offset; }

private:
  bool write(std::span<const uint8_t> bytes, std::string& error);

  io::AtomicFileWriter m_file;
  std::filesystem::path m_path;
  int32_t m_compressionLevel = 0;
  uint64_t m_offset = 0;
  uint64_t m_chunkCount = 0;
  std::vector<uint8_t> m_scratch;
};

class FrameBuilder {
public:
  void begin(uint32_t frameIndex, uint32_t sourceFrame, const RenderConfig& config);
  void add_memory_map(const MemoryMapRecord& record);
  void add_fifo(std::span<const uint8_t> fifo);
  void add_encoded_records(std::span<const uint8_t> records, uint32_t count);
  [[nodiscard]] std::span<const uint8_t> finish();
  void clear() noexcept;

private:
  static constexpr size_t NoFifoRecord = SIZE_MAX;

  std::vector<uint8_t> m_data;
  uint32_t m_recordCount = 0;
  size_t m_lastFifoRecord = NoFifoRecord;
  uint32_t m_lastFifoSize = 0;
};

class BlobListBuilder {
public:
  void add(const Hash128& hash, std::span<const uint8_t> bytes);
  [[nodiscard]] bool empty() const noexcept { return m_count == 0; }
  [[nodiscard]] uint32_t count() const noexcept { return m_count; }
  [[nodiscard]] uint64_t raw_bytes() const noexcept { return m_rawBytes; }
  [[nodiscard]] std::span<const uint8_t> finish();
  void clear() noexcept;

private:
  std::vector<uint8_t> m_data;
  uint32_t m_count = 0;
  uint64_t m_rawBytes = 0;
};
} // namespace aurora::trace

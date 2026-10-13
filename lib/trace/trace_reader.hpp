#pragma once

#include "trace_format.hpp"

#include "../io.hpp"

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace aurora::trace {
struct FrameRecordView {
  RecordType type;
  std::span<const uint8_t> payload;
};
struct FrameView {
  uint32_t frameIndex = 0;
  uint32_t sourceFrame = 0;
  RenderConfig config;
  std::vector<uint8_t> storage;
  std::vector<FrameRecordView> records;
};
using BlobSink = std::function<bool(const Hash128& hash, std::span<const uint8_t> bytes, std::string& error)>;

class TraceReader {
public:
  enum class Status : uint8_t { Frame, End, Error };

  bool open(const std::filesystem::path& path, std::string& error);
  [[nodiscard]] const FileHeader& header() const noexcept { return m_header; }
  [[nodiscard]] const TraceInfo& info() const noexcept { return m_info; }
  [[nodiscard]] std::span<const uint8_t> state_preamble() const noexcept { return m_statePreamble; }
  [[nodiscard]] bool complete() const noexcept;
  Status next(FrameView& frame, const BlobSink& sink, std::string& error);
  [[nodiscard]] const std::optional<Trailer>& trailer() const noexcept { return m_trailer; }
  bool rewind(std::string& error);

private:
  enum class ChunkStatus : uint8_t { Ok, Truncated, Error };

  ChunkStatus read_chunk(ChunkHeader& header, std::vector<uint8_t>& payload, std::string& error);
  bool parse_blobs(std::span<const uint8_t> payload, const BlobSink& sink, std::string& error);
  bool parse_frame(std::vector<uint8_t>&& payload, FrameView& frame, std::string& error);

  io::Stream m_file;
  uint64_t m_fileSize = 0;
  uint64_t m_offset = 0;
  uint64_t m_firstFrameOffset = 0;
  FileHeader m_header;
  TraceInfo m_info;
  std::vector<uint8_t> m_statePreamble;
  std::optional<Trailer> m_trailer;
  absl::flat_hash_map<Hash128, uint32_t> m_blobSizes;
  uint32_t m_framesRead = 0;
  bool m_ended = false;
  std::vector<uint8_t> m_stored;
};
} // namespace aurora::trace

#pragma once

#include "address_map.hpp"
#include "trace_format.hpp"
#include "trace_writer.hpp"

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::trace {
class CaptureSession {
public:
  bool open(const std::filesystem::path& path, int32_t compressionLevel, uint32_t fileFlags, std::string& error);
  bool write_header_chunks(const TraceInfo& info, std::span<const uint8_t> statePreamble, std::string& error);
  void begin_frame(uint32_t sourceFrame, const RenderConfig& config);
  void observe(uint8_t kind, uint64_t address, std::span<const uint8_t> bytes);
  void processed(std::span<const uint8_t> fifo);
  bool end_frame(std::string& error);
  bool finalize(uint32_t endFlags, std::string_view reason, std::string& error);
  void discard() noexcept;
  [[nodiscard]] bool is_open() const noexcept { return m_writer.is_open(); }
  [[nodiscard]] bool in_frame() const noexcept { return m_inFrame; }
  [[nodiscard]] uint32_t frames_written() const noexcept { return m_trailer.frameCount; }
  [[nodiscard]] uint64_t bytes_written() const noexcept { return m_writer.bytes_written(); }

private:
  void reset_state() noexcept;
  void add_record(RecordType type, std::span<const uint8_t> payload);
  bool flush_blobs(std::string& error);

  TraceWriter m_writer;
  FrameBuilder m_frame;
  BlobListBuilder m_blobs;
  absl::flat_hash_map<Hash128, uint32_t> m_blobIds;
  AddressMap m_mirror;
  std::vector<uint8_t> m_carry;
  uint32_t m_carryCount = 0;
  bool m_carryLogged = false;
  bool m_inFrame = false;
  uint32_t m_fileFlags = 0;
  Trailer m_trailer;
  std::string m_error;
};
} // namespace aurora::trace

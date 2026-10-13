#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace aurora::trace {
enum class CaptureState : uint8_t {
  Idle,
  Pending,
  Capturing,
  Finished,
  Failed,
};

struct CaptureOptions {
  std::filesystem::path path;
  uint32_t frameCount = 1;
  int32_t compressionLevel = 3;
  std::string label;
};

struct CaptureStatus {
  CaptureState state = CaptureState::Idle;
  std::filesystem::path path;
  uint32_t framesRequested = 0;
  uint32_t framesCaptured = 0;
  uint64_t bytesWritten = 0;
  std::string error;
};

bool start_capture(const CaptureOptions& options);
void stop_capture();
CaptureStatus capture_status();
bool is_capturing() noexcept;
} // namespace aurora::trace

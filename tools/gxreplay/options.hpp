#pragma once

#include <aurora/aurora.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::gxreplay {
struct FrameRange {
  uint32_t first = 0;
  uint32_t last = UINT32_MAX;

  [[nodiscard]] bool contains(uint32_t frame) const noexcept { return frame >= first && frame < last; }
  bool operator==(const FrameRange&) const = default;
};

struct Options {
  std::filesystem::path trace;
  std::filesystem::path outDir;
  std::filesystem::path dataDir;
  std::filesystem::path replacementsDir;
  std::filesystem::path writeSelfTest;
  AuroraBackend backend = BACKEND_AUTO;
  AuroraFeatureProfile profile = FEATURE_PROFILE_AUTO;
  FrameRange frames;
  uint32_t repeat = 1;
  bool writePng = true;
  bool selfTest = false;
  bool info = false;
  bool allowIncomplete = false;
  bool allowDeviceMismatch = false;
  bool allowMisses = false;
  bool allowCpu = false;
  bool showWindow = false;
  bool verbose = false;
  bool help = false;
};

std::optional<AuroraBackend> parse_backend(std::string_view name) noexcept;
std::string_view backend_name(AuroraBackend backend) noexcept;
std::optional<AuroraFeatureProfile> parse_profile(std::string_view name) noexcept;
std::optional<FrameRange> parse_frame_range(std::string_view text) noexcept;
std::optional<Options> parse_options(int argc, const char* const* argv, const char* envBackend, std::string& error);
std::string_view usage() noexcept;
} // namespace aurora::gxreplay

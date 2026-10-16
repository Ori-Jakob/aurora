#pragma once

#include <aurora/aurora.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace render_demo {
struct Options {
  AuroraBackend backend = BACKEND_AUTO;
  AuroraFeatureProfile profile = FEATURE_PROFILE_AUTO;
  uint32_t frames = 0;
  uint32_t timeoutSeconds = 0;
  uint32_t width = 1280;
  uint32_t height = 960;
  std::string dataDir;
  std::string tracePath;
  uint32_t traceStart = 2;
  uint32_t traceFrames = 4;
  bool failOnError = false;
  bool help = false;
  bool vignette = false;
  std::string vignetteAnchor;
  uint32_t expectAnchorWarnings = 0;
  bool expectAnchorWarningsSet = false;
  uint32_t msaa = 1;
};

std::optional<AuroraBackend> parse_backend(std::string_view name) noexcept;
std::optional<AuroraFeatureProfile> parse_profile(std::string_view name) noexcept;
std::string_view backend_name(AuroraBackend backend) noexcept;
std::optional<Options> parse_options(int argc, const char* const* argv, const char* envBackend, std::string& error);
std::string_view usage() noexcept;
} // namespace render_demo

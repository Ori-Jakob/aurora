#pragma once

#include "../graph.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::debug::capture {
inline constexpr size_t MaxRequests = 4;
inline constexpr size_t ReadbackSlots = 3;
inline constexpr uint32_t ResolveLimit = 300;
inline constexpr uint32_t ArmDelay = 16;
inline constexpr uint32_t ArmedLimit = 300;
inline constexpr uint16_t ChainOrder = 0xFFFF;
inline constexpr size_t MaxDistinctColors = 64;
inline constexpr std::string_view Schema = "aurora.render.capture";
inline constexpr std::array<std::string_view, MaxRequests> PassNames{
    "debug.capture.0",
    "debug.capture.1",
    "debug.capture.2",
    "debug.capture.3",
};

enum class Status : uint8_t { Idle, Waiting, Armed, Written, Unavailable, MissingInput, TimedOut, Failed };
enum class PlaceResult : uint8_t { Placed, Wait, Unavailable };

struct Request {
  std::string resource;
  std::string anchor;
  bool latest = false;
  std::string dir;
};

struct EntryInfo {
  Request request;
  Status status = Status::Idle;
  std::string message;
  uint64_t dropped = 0;
};

struct Placement {
  Stage stage = Stage::OpaqueDone;
  std::string after;
  std::string before;
  uint16_t order = 0;
};

struct CopyLayout {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bytesPerPixel = 0;
  uint32_t rowBytes = 0;
  uint32_t paddedRowBytes = 0;
  uint64_t bufferSize = 0;
};

struct FogSummary {
  uint64_t records = 0;
  uint64_t nonZero = 0;
  uint64_t withFog = 0;
  uint64_t locked = 0;
  std::vector<uint32_t> distinct;
  bool distinctOverflow = false;
  std::array<uint64_t, 128> qHistogram{};
};

struct MotionSummary {
  uint64_t nonZero = 0;
  double maxPx = 0.0;
  double meanPx = 0.0;
};

struct NormalSummary {
  std::array<uint64_t, 4> codeCounts{};
};

struct Capture {
  std::string resource;
  std::string anchor;
  uint64_t frame = 0;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> bytes;
};

[[nodiscard]] std::string_view status_name(Status status) noexcept;
[[nodiscard]] std::vector<Request> parse_env(std::string_view text, std::string_view dir);
[[nodiscard]] std::optional<CopyLayout> copy_layout(wgpu::TextureFormat format, uint32_t width,
                                                    uint32_t height) noexcept;
[[nodiscard]] FogSummary summarize_fog(std::span<const uint8_t> bytes);
[[nodiscard]] MotionSummary summarize_motion(std::span<const uint8_t> bytes, uint32_t width, uint32_t height);
[[nodiscard]] NormalSummary summarize_normal(std::span<const uint8_t> bytes);
[[nodiscard]] std::string header_json(const Capture& capture, std::string_view binName);
[[nodiscard]] std::vector<uint8_t> preview_rgba(const Capture& capture);
[[nodiscard]] std::string file_stem(std::string_view resource, std::string_view anchor, std::optional<uint64_t> frame);
bool write_capture(const std::filesystem::path& dir, std::string_view stem, const Capture& capture);
[[nodiscard]] PlaceResult place(std::string_view anchor, const detail::CompiledGraph* graph, Placement& out);
[[nodiscard]] std::string default_dir();

std::optional<size_t> request(Request request);
[[nodiscard]] std::vector<EntryInfo> entries();
void initialize();
void shutdown() noexcept;
void on_latch();
void resolve(const detail::CompiledGraph* graph);

namespace testing {
[[nodiscard]] detail::PassId pass_id(size_t index) noexcept;
void deliver(size_t index, Capture capture);
std::optional<Capture> read_texture(const wgpu::Texture& texture, uint64_t frame);
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::debug::capture

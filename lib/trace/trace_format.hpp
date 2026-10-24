#pragma once

#include <aurora/aurora.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::trace {
constexpr uint32_t fourcc(char a, char b, char c, char d) noexcept {
  return static_cast<uint32_t>(static_cast<uint8_t>(a)) | static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8 |
         static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16 | static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24;
}

inline constexpr std::array<uint8_t, 8> FileMagic{'A', 'G', 'X', 'T', 'R', 'A', 'C', 'E'};
inline constexpr uint16_t FormatMajor = 1;
inline constexpr uint16_t FormatMinor = 0;
inline constexpr uint16_t ChunkVersion = 1;
inline constexpr uint32_t FileHeaderSize = 64;
inline constexpr uint32_t ChunkHeaderSize = 20;
inline constexpr uint32_t FrameHeaderSize = 12;
inline constexpr uint32_t RecordHeaderSize = 8;
inline constexpr uint32_t RenderConfigSize = 24;
inline constexpr uint32_t MemoryMapSize = 32;
inline constexpr uint32_t MaxChunkRawSize = 1u << 30;
inline constexpr uint64_t MaxBlobChunkBytes = 256ull << 20;

enum class ChunkType : uint32_t {
  Info = fourcc('I', 'N', 'F', 'O'),
  StatePreamble = fourcc('S', 'T', 'A', 'T'),
  Blobs = fourcc('B', 'L', 'O', 'B'),
  Frame = fourcc('F', 'R', 'A', 'M'),
  End = fourcc('E', 'N', 'D', 'T'),
};
enum class Compression : uint8_t { None = 0, Zstd = 1 };
enum class RecordType : uint8_t { RenderConfig = 1, MemoryMap = 2, Fifo = 3 };
enum FileFlag : uint32_t { FileFinalized = 1u << 0, FileAborted = 1u << 1, FileSynthetic = 1u << 2 };
enum CaptureFlag : uint32_t {
  CaptureReplacementsBypassed = 1u << 0,
  CaptureBlockingPipelines = 1u << 1,
  CaptureSynchronousPublish = 1u << 2,
  CaptureDeviceFeaturesV2 = 1u << 3,
};
enum DeviceFeature : uint32_t {
  DeviceCoreFeatures = 1u << 0,
  DeviceDualSourceBlending = 1u << 1,
  DeviceBcTextures = 1u << 2,
  DeviceAstcTextures = 1u << 3,
  DeviceTextureSwizzle = 1u << 4,
  DeviceSubgroups = 1u << 5,
  DeviceShaderF16 = 1u << 6,
};
enum EndFlag : uint32_t {
  EndAborted = 1u << 0,
  EndViChangedMidFrame = 1u << 1,
  EndGpuStateUncaptured = 1u << 2,
  EndGpuStorageMissed = 1u << 3,
};
uint32_t compared_device_features(uint32_t captureFlags) noexcept;

struct Hash128 {
  uint64_t low = 0;
  uint64_t high = 0;
  bool operator==(const Hash128&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const Hash128& v) {
    return H::combine(std::move(h), v.low, v.high);
  }
};
Hash128 hash_bytes(std::span<const uint8_t> bytes) noexcept;
uint32_t checksum_bytes(std::span<const uint8_t> bytes) noexcept;

struct FileHeader {
  uint16_t versionMajor = FormatMajor;
  uint16_t versionMinor = FormatMinor;
  uint32_t flags = 0;
  uint32_t frameCount = 0;
  uint32_t blobCount = 0;
  uint64_t chunkCount = 0;
  uint64_t trailerOffset = 0;
};
struct ChunkHeader {
  uint32_t type = 0;
  Compression compression = Compression::None;
  uint16_t chunkVersion = ChunkVersion;
  uint32_t storedSize = 0;
  uint32_t rawSize = 0;
  uint32_t checksum = 0;
};
struct TraceInfo {
  uint32_t captureFlags = 0;
  AuroraBackend backend = BACKEND_AUTO;
  uint32_t msaaSamples = 1;
  uint32_t maxAnisotropy = 16;
  uint32_t efbColorFormat = 0;
  uint32_t depthFormat = 0;
  uint32_t deviceFeatures = 0;
  AuroraFeatureProfile featureProfile = FEATURE_PROFILE_AUTO;
  AuroraSampler resampler = SAMPLER_BILINEAR;
  uint32_t requestedFrames = 0;
  std::string label;
  std::string adapter;
};
struct RenderConfig {
  uint32_t efbWidth = 0;
  uint32_t efbHeight = 0;
  uint32_t viFbWidth = 640;
  uint32_t viEfbHeight = 480;
  uint8_t viewportPolicy = 0;
  bool normalBuffer = false;
  uint32_t streamingStorageMiB = 0;
  bool operator==(const RenderConfig&) const = default;
};
struct MemoryMapRecord {
  uint64_t address = 0;
  uint32_t size = 0;
  uint8_t kind = 0;
  Hash128 blob;
};
struct Trailer {
  uint32_t frameCount = 0;
  uint32_t blobCount = 0;
  uint64_t blobBytes = 0;
  uint64_t fifoBytes = 0;
  uint64_t mapRecords = 0;
  uint64_t carriedRecords = 0;
  uint32_t endFlags = 0;
  std::string reason;
};

class LeWriter {
public:
  explicit LeWriter(std::vector<uint8_t>& out) noexcept : m_out(out) {}
  void u8(uint8_t v);
  void u16(uint16_t v);
  void u32(uint32_t v);
  void u64(uint64_t v);
  void hash(const Hash128& v);
  void bytes(std::span<const uint8_t> v);
  void string16(std::string_view v);
  void patch_u32(size_t offset, uint32_t v) noexcept;
  [[nodiscard]] size_t size() const noexcept { return m_out.size(); }

private:
  std::vector<uint8_t>& m_out;
};

class LeReader {
public:
  explicit LeReader(std::span<const uint8_t> data) noexcept : m_data(data) {}
  bool u8(uint8_t& v) noexcept;
  bool u16(uint16_t& v) noexcept;
  bool u32(uint32_t& v) noexcept;
  bool u64(uint64_t& v) noexcept;
  bool hash(Hash128& v) noexcept;
  bool bytes(size_t count, std::span<const uint8_t>& v) noexcept;
  bool string16(std::string& v);
  [[nodiscard]] size_t remaining() const noexcept { return m_data.size() - m_offset; }
  [[nodiscard]] size_t offset() const noexcept { return m_offset; }

private:
  std::span<const uint8_t> m_data;
  size_t m_offset = 0;
};

std::vector<uint8_t> encode_file_header(const FileHeader& header);
bool decode_file_header(std::span<const uint8_t> bytes, FileHeader& header, std::string& error);
void encode_chunk_header(LeWriter& out, const ChunkHeader& header);
bool decode_chunk_header(std::span<const uint8_t> bytes, ChunkHeader& header);
std::vector<uint8_t> encode_info(const TraceInfo& info);
bool decode_info(std::span<const uint8_t> bytes, TraceInfo& info);
void encode_render_config(LeWriter& out, const RenderConfig& config);
bool decode_render_config(std::span<const uint8_t> bytes, RenderConfig& config);
void encode_memory_map(LeWriter& out, const MemoryMapRecord& record);
bool decode_memory_map(std::span<const uint8_t> bytes, MemoryMapRecord& record);
std::vector<uint8_t> encode_trailer(const Trailer& trailer);
bool decode_trailer(std::span<const uint8_t> bytes, Trailer& trailer);
} // namespace aurora::trace

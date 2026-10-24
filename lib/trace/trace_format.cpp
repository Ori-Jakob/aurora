#include "trace_format.hpp"

#include <xxhash.h>

#include <algorithm>
#include <cstring>

namespace aurora::trace {
Hash128 hash_bytes(std::span<const uint8_t> bytes) noexcept {
  const XXH128_hash_t hash = XXH3_128bits(bytes.data(), bytes.size());
  return {.low = hash.low64, .high = hash.high64};
}

uint32_t checksum_bytes(std::span<const uint8_t> bytes) noexcept { return XXH32(bytes.data(), bytes.size(), 0); }

uint32_t compared_device_features(uint32_t captureFlags) noexcept {
  constexpr uint32_t V1 =
      DeviceCoreFeatures | DeviceDualSourceBlending | DeviceBcTextures | DeviceAstcTextures | DeviceTextureSwizzle;
  return (captureFlags & CaptureDeviceFeaturesV2) != 0 ? UINT32_MAX : V1;
}

void LeWriter::u8(uint8_t v) { m_out.push_back(v); }

void LeWriter::u16(uint16_t v) {
  m_out.push_back(static_cast<uint8_t>(v));
  m_out.push_back(static_cast<uint8_t>(v >> 8));
}

void LeWriter::u32(uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    m_out.push_back(static_cast<uint8_t>(v >> (i * 8)));
  }
}

void LeWriter::u64(uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    m_out.push_back(static_cast<uint8_t>(v >> (i * 8)));
  }
}

void LeWriter::hash(const Hash128& v) {
  u64(v.low);
  u64(v.high);
}

void LeWriter::bytes(std::span<const uint8_t> v) { m_out.insert(m_out.end(), v.begin(), v.end()); }

void LeWriter::string16(std::string_view v) {
  const size_t length = std::min<size_t>(v.size(), UINT16_MAX);
  u16(static_cast<uint16_t>(length));
  bytes({reinterpret_cast<const uint8_t*>(v.data()), length});
}

void LeWriter::patch_u32(size_t offset, uint32_t v) noexcept {
  for (int i = 0; i < 4; ++i) {
    m_out[offset + i] = static_cast<uint8_t>(v >> (i * 8));
  }
}

bool LeReader::u8(uint8_t& v) noexcept {
  if (remaining() < 1) {
    return false;
  }
  v = m_data[m_offset++];
  return true;
}

bool LeReader::u16(uint16_t& v) noexcept {
  if (remaining() < 2) {
    return false;
  }
  v = static_cast<uint16_t>(m_data[m_offset] | m_data[m_offset + 1] << 8);
  m_offset += 2;
  return true;
}

bool LeReader::u32(uint32_t& v) noexcept {
  if (remaining() < 4) {
    return false;
  }
  v = 0;
  for (int i = 0; i < 4; ++i) {
    v |= static_cast<uint32_t>(m_data[m_offset + i]) << (i * 8);
  }
  m_offset += 4;
  return true;
}

bool LeReader::u64(uint64_t& v) noexcept {
  if (remaining() < 8) {
    return false;
  }
  v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(m_data[m_offset + i]) << (i * 8);
  }
  m_offset += 8;
  return true;
}

bool LeReader::hash(Hash128& v) noexcept { return u64(v.low) && u64(v.high); }

bool LeReader::bytes(size_t count, std::span<const uint8_t>& v) noexcept {
  if (remaining() < count) {
    return false;
  }
  v = m_data.subspan(m_offset, count);
  m_offset += count;
  return true;
}

bool LeReader::string16(std::string& v) {
  uint16_t length = 0;
  std::span<const uint8_t> data;
  if (!u16(length) || !bytes(length, data)) {
    return false;
  }
  v.assign(reinterpret_cast<const char*>(data.data()), data.size());
  return true;
}

std::vector<uint8_t> encode_file_header(const FileHeader& header) {
  std::vector<uint8_t> out;
  out.reserve(FileHeaderSize);
  LeWriter writer{out};
  writer.bytes(FileMagic);
  writer.u16(header.versionMajor);
  writer.u16(header.versionMinor);
  writer.u32(FileHeaderSize);
  writer.u32(header.flags);
  writer.u32(header.frameCount);
  writer.u32(header.blobCount);
  writer.u32(0);
  writer.u64(header.chunkCount);
  writer.u64(header.trailerOffset);
  out.resize(FileHeaderSize, 0);
  return out;
}

bool decode_file_header(std::span<const uint8_t> bytes, FileHeader& header, std::string& error) {
  if (bytes.size() < FileHeaderSize || !std::equal(FileMagic.begin(), FileMagic.end(), bytes.begin())) {
    error = "not a GX trace";
    return false;
  }
  LeReader reader{bytes.subspan(FileMagic.size())};
  uint32_t headerSize = 0;
  uint32_t reserved = 0;
  reader.u16(header.versionMajor);
  reader.u16(header.versionMinor);
  reader.u32(headerSize);
  reader.u32(header.flags);
  reader.u32(header.frameCount);
  reader.u32(header.blobCount);
  reader.u32(reserved);
  reader.u64(header.chunkCount);
  reader.u64(header.trailerOffset);
  if (header.versionMajor != FormatMajor) {
    error = "unsupported trace version";
    return false;
  }
  if (headerSize != FileHeaderSize) {
    error = "unsupported trace header size";
    return false;
  }
  return true;
}

void encode_chunk_header(LeWriter& out, const ChunkHeader& header) {
  out.u32(header.type);
  out.u8(static_cast<uint8_t>(header.compression));
  out.u8(0);
  out.u16(header.chunkVersion);
  out.u32(header.storedSize);
  out.u32(header.rawSize);
  out.u32(header.checksum);
}

bool decode_chunk_header(std::span<const uint8_t> bytes, ChunkHeader& header) {
  LeReader reader{bytes};
  uint8_t compression = 0;
  uint8_t reserved = 0;
  if (!reader.u32(header.type) || !reader.u8(compression) || !reader.u8(reserved) || !reader.u16(header.chunkVersion) ||
      !reader.u32(header.storedSize) || !reader.u32(header.rawSize) || !reader.u32(header.checksum)) {
    return false;
  }
  header.compression = static_cast<Compression>(compression);
  return true;
}

std::vector<uint8_t> encode_info(const TraceInfo& info) {
  std::vector<uint8_t> out;
  LeWriter writer{out};
  writer.u32(info.captureFlags);
  writer.u32(static_cast<uint32_t>(info.backend));
  writer.u32(info.msaaSamples);
  writer.u32(info.maxAnisotropy);
  writer.u32(info.efbColorFormat);
  writer.u32(info.depthFormat);
  writer.u32(info.deviceFeatures);
  writer.u32(static_cast<uint32_t>(info.featureProfile));
  writer.u32(static_cast<uint32_t>(info.resampler));
  writer.u32(info.requestedFrames);
  writer.string16(info.label);
  writer.string16(info.adapter);
  return out;
}

bool decode_info(std::span<const uint8_t> bytes, TraceInfo& info) {
  LeReader reader{bytes};
  uint32_t backend = 0;
  uint32_t profile = 0;
  uint32_t resampler = 0;
  if (!reader.u32(info.captureFlags) || !reader.u32(backend) || !reader.u32(info.msaaSamples) ||
      !reader.u32(info.maxAnisotropy) || !reader.u32(info.efbColorFormat) || !reader.u32(info.depthFormat) ||
      !reader.u32(info.deviceFeatures) || !reader.u32(profile) || !reader.u32(resampler) ||
      !reader.u32(info.requestedFrames) || !reader.string16(info.label) || !reader.string16(info.adapter)) {
    return false;
  }
  info.backend = static_cast<AuroraBackend>(backend);
  info.featureProfile = static_cast<AuroraFeatureProfile>(profile);
  info.resampler = static_cast<AuroraSampler>(resampler);
  return true;
}

void encode_render_config(LeWriter& out, const RenderConfig& config) {
  out.u32(config.efbWidth);
  out.u32(config.efbHeight);
  out.u32(config.viFbWidth);
  out.u32(config.viEfbHeight);
  out.u8(config.viewportPolicy);
  out.u8(config.normalBuffer ? 1 : 0);
  out.u16(0);
  out.u32(config.streamingStorageMiB);
}

bool decode_render_config(std::span<const uint8_t> bytes, RenderConfig& config) {
  LeReader reader{bytes};
  uint8_t normalBuffer = 0;
  uint16_t reserved = 0;
  if (bytes.size() < RenderConfigSize || !reader.u32(config.efbWidth) || !reader.u32(config.efbHeight) ||
      !reader.u32(config.viFbWidth) || !reader.u32(config.viEfbHeight) || !reader.u8(config.viewportPolicy) ||
      !reader.u8(normalBuffer) || !reader.u16(reserved) || !reader.u32(config.streamingStorageMiB)) {
    return false;
  }
  config.normalBuffer = normalBuffer != 0;
  return true;
}

void encode_memory_map(LeWriter& out, const MemoryMapRecord& record) {
  out.u64(record.address);
  out.u32(record.size);
  out.u8(record.kind);
  out.u8(0);
  out.u16(0);
  out.hash(record.blob);
}

bool decode_memory_map(std::span<const uint8_t> bytes, MemoryMapRecord& record) {
  LeReader reader{bytes};
  uint8_t reserved8 = 0;
  uint16_t reserved16 = 0;
  return bytes.size() >= MemoryMapSize && reader.u64(record.address) && reader.u32(record.size) &&
         reader.u8(record.kind) && reader.u8(reserved8) && reader.u16(reserved16) && reader.hash(record.blob);
}

std::vector<uint8_t> encode_trailer(const Trailer& trailer) {
  std::vector<uint8_t> out;
  LeWriter writer{out};
  writer.u32(trailer.frameCount);
  writer.u32(trailer.blobCount);
  writer.u64(trailer.blobBytes);
  writer.u64(trailer.fifoBytes);
  writer.u64(trailer.mapRecords);
  writer.u64(trailer.carriedRecords);
  writer.u32(trailer.endFlags);
  writer.string16(trailer.reason);
  return out;
}

bool decode_trailer(std::span<const uint8_t> bytes, Trailer& trailer) {
  LeReader reader{bytes};
  return reader.u32(trailer.frameCount) && reader.u32(trailer.blobCount) && reader.u64(trailer.blobBytes) &&
         reader.u64(trailer.fifoBytes) && reader.u64(trailer.mapRecords) && reader.u64(trailer.carriedRecords) &&
         reader.u32(trailer.endFlags) && reader.string16(trailer.reason);
}
} // namespace aurora::trace

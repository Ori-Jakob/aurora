#pragma once

#include "gx/gx.hpp"
#include "gx/pipeline.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::gx::corpus {
constexpr uint16_t CorpusFormatVersion = 1;
constexpr uint16_t BlobFormatVersion = 1;
constexpr size_t RecordSize = 868;
constexpr size_t RecordAttrsOffset = 8;
constexpr size_t RecordAttrStride = 8;
constexpr size_t RecordSwapOffset = 216;
constexpr size_t RecordStagesOffset = 232;
constexpr size_t RecordStageStride = 34;
constexpr size_t RecordChannelsOffset = 776;
constexpr size_t RecordChannelStride = 5;
constexpr size_t RecordTcgsOffset = 796;
constexpr size_t RecordTcgStride = 6;
constexpr size_t RecordAlphaCompareOffset = 844;
constexpr size_t RecordIndStagesOffset = 849;
constexpr size_t RecordIndStageStride = 4;
constexpr size_t RecordReservedOffset = 865;
constexpr uint8_t NoAttachment = 0xFF;
constexpr std::string_view GoldenHeader = "# aurora gx shader golden v1";

struct Variant {
  DstAlphaMode dstAlphaMode = DstAlphaMode::None;
  uint32_t normalAttachment = UINT32_MAX;
  bool operator==(const Variant&) const = default;
};
struct Record {
  std::array<uint8_t, RecordSize> bytes{};
  bool operator==(const Record&) const = default;
};
struct Entry {
  Record record;
  std::string label;
};
struct GoldenLine {
  uint64_t key = 0;
  uint64_t wgslHash = 0;
  uint64_t wgslSize = 0;
  std::string label;
};
struct Blob {
  uint32_t configVersion = 0;
  uint64_t dbHash = 0;
  uint32_t firstFrameUsed = 0;
  uint32_t sourceIndex = 0;
  std::vector<uint8_t> bytes;
};
struct ImportStats {
  uint32_t accepted = 0;
  uint32_t duplicate = 0;
  uint32_t wrongVersion = 0;
  uint32_t wrongSize = 0;
  uint32_t notLegacy = 0;
  uint32_t invalid = 0;
};

std::optional<Record> encode(const ShaderConfig& config, Variant variant) noexcept;
ShaderConfig decode(const Record& record) noexcept;
Variant decode_variant(const Record& record) noexcept;
std::optional<std::string> validate(const Record& record);
uint64_t record_key(const Record& record) noexcept;
uint64_t text_hash(std::string_view text) noexcept;
std::string generate_wgsl(const Record& record);

void canonicalize(ShaderConfig& config) noexcept;
std::vector<Entry> generate_synthetic();
std::vector<Variant> pipeline_variants(const PipelineConfig& config);
ImportStats import_blob(const Blob& blob, std::vector<Entry>& out);

void sort_and_dedupe(std::vector<Entry>& entries);
std::vector<GoldenLine> golden_lines(std::span<const Entry> entries);
std::string format_golden(std::span<const GoldenLine> lines);
std::optional<std::vector<GoldenLine>> parse_golden(std::string_view text, std::string* error);
bool write_corpus(const std::filesystem::path& path, std::span<const Entry> entries);
std::optional<std::vector<Entry>> read_corpus(const std::filesystem::path& path, std::string* error);
std::optional<std::vector<Blob>> read_blobs(const std::filesystem::path& path, std::string* error);
} // namespace aurora::gx::corpus

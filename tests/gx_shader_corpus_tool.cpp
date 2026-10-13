#include "gx_shader_corpus.hpp"

#include "gfx/resources.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
using namespace aurora::gx;
using namespace aurora::gx::corpus;

using Options = std::map<std::string, std::string, std::less<>>;

int usage() {
  fmt::print(stderr,
             "usage: gx_shader_corpus <command> [options]\n"
             "  info\n"
             "  generate --out F.gxsc\n"
             "  golden --out F.txt\n"
             "  import --blobs B.gxpb [--merge F.gxsc] --out F.gxsc\n"
             "  hash --corpus F.gxsc --out H.txt\n"
             "  emit --corpus F.gxsc --keys k1,k2|all --out-dir D\n");
  return 64;
}

std::optional<std::string> option(const Options& options, std::string_view name) {
  const auto it = options.find(name);
  if (it == options.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool write_text(const std::filesystem::path& path, std::string_view text) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  return static_cast<bool>(file);
}

std::optional<std::vector<Entry>> load_corpus(const std::filesystem::path& path) {
  std::string error;
  auto entries = read_corpus(path, &error);
  if (!entries) {
    fmt::print(stderr, "{}: {}\n", path.string(), error);
  }
  return entries;
}

std::optional<std::vector<Entry>> valid_synthetic() {
  auto entries = generate_synthetic();
  bool ok = true;
  for (const auto& entry : entries) {
    if (const auto reason = validate(entry.record)) {
      fmt::print(stderr, "invalid synthetic record {}: {}\n", entry.label, *reason);
      ok = false;
    }
  }
  if (!ok) {
    return std::nullopt;
  }
  return entries;
}

int command_info() {
  fmt::print(
      "{{\"recordFormat\":{},\"recordSize\":{},\"pipelineConfigVersion\":{},\"pipelineConfigSize\":{},"
      "\"shaderConfigSize\":{}}}\n",
      CorpusFormatVersion, RecordSize, GXPipelineConfigVersion, sizeof(PipelineConfig), sizeof(ShaderConfig));
  return 0;
}

int command_generate(const Options& options) {
  const auto out = option(options, "out");
  if (!out) {
    return usage();
  }
  const auto entries = valid_synthetic();
  if (!entries) {
    return 1;
  }
  if (!write_corpus(*out, *entries)) {
    fmt::print(stderr, "cannot write {}\n", *out);
    return 1;
  }
  fmt::print("{{\"records\":{}}}\n", entries->size());
  return 0;
}

int command_golden(const Options& options) {
  const auto out = option(options, "out");
  if (!out) {
    return usage();
  }
  auto entries = valid_synthetic();
  if (!entries) {
    return 1;
  }
  sort_and_dedupe(*entries);
  const auto lines = golden_lines(*entries);
  if (!write_text(*out, format_golden(lines))) {
    fmt::print(stderr, "cannot write {}\n", *out);
    return 1;
  }
  fmt::print("{{\"records\":{}}}\n", lines.size());
  return 0;
}

std::string stats_json(const ImportStats& stats, uint32_t blobs) {
  return fmt::format(
      "{{\"blobs\":{},\"accepted\":{},\"duplicate\":{},\"wrongVersion\":{},\"wrongSize\":{},"
      "\"notLegacy\":{},\"invalid\":{}}}",
      blobs, stats.accepted, stats.duplicate, stats.wrongVersion, stats.wrongSize, stats.notLegacy, stats.invalid);
}

int command_import(const Options& options) {
  const auto blobsPath = option(options, "blobs");
  const auto out = option(options, "out");
  if (!blobsPath || !out) {
    return usage();
  }
  std::string error;
  const auto blobs = read_blobs(*blobsPath, &error);
  if (!blobs) {
    fmt::print(stderr, "{}: {}\n", *blobsPath, error);
    return 1;
  }
  std::vector<Entry> entries;
  // A missing merge target is the first import; anything unreadable must not be overwritten.
  if (const auto merge = option(options, "merge"); merge && std::filesystem::exists(*merge)) {
    auto existing = load_corpus(*merge);
    if (!existing) {
      return 1;
    }
    entries = std::move(*existing);
  }
  const size_t before = entries.size();
  std::map<uint32_t, std::pair<ImportStats, uint32_t>> perVersion;
  for (const auto& blob : *blobs) {
    const auto stats = import_blob(blob, entries);
    auto& [total, count] = perVersion[blob.configVersion];
    total.accepted += stats.accepted;
    total.duplicate += stats.duplicate;
    total.wrongVersion += stats.wrongVersion;
    total.wrongSize += stats.wrongSize;
    total.notLegacy += stats.notLegacy;
    total.invalid += stats.invalid;
    ++count;
  }
  sort_and_dedupe(entries);
  if (!write_corpus(*out, entries)) {
    fmt::print(stderr, "cannot write {}\n", *out);
    return 1;
  }
  std::string versions;
  for (const auto& [version, value] : perVersion) {
    if (!versions.empty()) {
      versions += ',';
    }
    versions += fmt::format("\"{}\":{}", version, stats_json(value.first, value.second));
  }
  fmt::print("{{\"recordsBefore\":{},\"records\":{},\"versions\":{{{}}}}}\n", before, entries.size(), versions);
  return 0;
}

int command_hash(const Options& options) {
  const auto corpusPath = option(options, "corpus");
  const auto out = option(options, "out");
  if (!corpusPath || !out) {
    return usage();
  }
  const auto entries = load_corpus(*corpusPath);
  if (!entries) {
    return 1;
  }
  std::string text;
  size_t invalid = 0;
  for (size_t i = 0; i < entries->size(); ++i) {
    const auto& entry = (*entries)[i];
    // A crash inside the generator cannot be caught, so the last marker names the culprit.
    fmt::print(stderr, "@{} {}\n", i, entry.label);
    std::fflush(stderr);
    const uint64_t key = record_key(entry.record);
    if (const auto reason = validate(entry.record)) {
      text += fmt::format("{:016x} invalid 0 {} :: {}\n", key, entry.label, *reason);
      ++invalid;
      continue;
    }
    const auto wgsl = generate_wgsl(entry.record);
    text += fmt::format("{:016x} {:016x} {} {}\n", key, text_hash(wgsl), wgsl.size(), entry.label);
  }
  if (!write_text(*out, text)) {
    fmt::print(stderr, "cannot write {}\n", *out);
    return 1;
  }
  fmt::print("{{\"records\":{},\"invalid\":{}}}\n", entries->size(), invalid);
  return invalid == 0 ? 0 : 2;
}

int command_emit(const Options& options) {
  const auto corpusPath = option(options, "corpus");
  const auto keysArg = option(options, "keys");
  const auto outDir = option(options, "out-dir");
  if (!corpusPath || !keysArg || !outDir) {
    return usage();
  }
  const auto entries = load_corpus(*corpusPath);
  if (!entries) {
    return 1;
  }
  const bool all = *keysArg == "all";
  std::set<std::string, std::less<>> wanted;
  for (size_t start = 0; !all && start <= keysArg->size();) {
    const size_t comma = std::min(keysArg->find(',', start), keysArg->size());
    if (comma > start) {
      wanted.emplace(keysArg->substr(start, comma - start));
    }
    start = comma + 1;
  }
  std::error_code ec;
  std::filesystem::create_directories(*outDir, ec);
  int result = 0;
  size_t written = 0;
  for (const auto& entry : *entries) {
    const auto key = fmt::format("{:016x}", record_key(entry.record));
    if (!all && wanted.erase(key) == 0) {
      continue;
    }
    if (const auto reason = validate(entry.record)) {
      fmt::print(stderr, "skipping invalid {} {}: {}\n", key, entry.label, *reason);
      result = 1;
      continue;
    }
    if (!write_text(std::filesystem::path(*outDir) / (key + ".wgsl"), generate_wgsl(entry.record))) {
      fmt::print(stderr, "cannot write {}.wgsl\n", key);
      return 1;
    }
    ++written;
  }
  for (const auto& key : wanted) {
    fmt::print(stderr, "key {} not in corpus\n", key);
    result = 1;
  }
  fmt::print("{{\"written\":{}}}\n", written);
  return result;
}
} // namespace

int main(int argc, char** argv) {
  // uniformSize depends on this device limit; the WGSL text does not.
  aurora::gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
  if (argc < 2) {
    return usage();
  }
  const std::string_view command = argv[1];
  Options options;
  for (int i = 2; i < argc; i += 2) {
    const std::string_view name = argv[i];
    if (!name.starts_with("--") || i + 1 >= argc) {
      return usage();
    }
    options.emplace(std::string{name.substr(2)}, argv[i + 1]);
  }
  if (command == "info") {
    return command_info();
  }
  if (command == "generate") {
    return command_generate(options);
  }
  if (command == "golden") {
    return command_golden(options);
  }
  if (command == "import") {
    return command_import(options);
  }
  if (command == "hash") {
    return command_hash(options);
  }
  if (command == "emit") {
    return command_emit(options);
  }
  return usage();
}

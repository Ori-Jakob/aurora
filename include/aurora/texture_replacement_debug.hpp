#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <imgui.h>

namespace aurora::gfx::texture_replacement {
enum class DebugMaterialKind : uint8_t {
  Replacement,
  NamedPbr,
};

enum class DebugMapSlot : uint8_t {
  Base,
  Rmaos,
  Roughness,
  Metallic,
  Ao,
  Specular,
  Normal,
  Emissive,
  Count,
};

struct DebugMaterialInfo {
  DebugMaterialKind kind = DebugMaterialKind::Replacement;
  std::string name;
  std::filesystem::path directory;
  std::array<std::filesystem::path, static_cast<size_t>(DebugMapSlot::Count)> maps{};
  uint64_t textureHash = 0;
  uint64_t tlutHash = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t format = 0;
  bool hasTlut = false;
  bool hasMips = false;
  bool cached = false;
  bool failed = false;
  bool reportedMissing = false;
  bool textureHashWildcard = false;
  bool tlutHashWildcard = false;
};

struct DebugInventoryStats {
  std::filesystem::path replacementRoot;
  std::filesystem::path dumpRoot;
  uint32_t indexedReplacementCount = 0;
  uint32_t pbrReplacementCount = 0;
  uint32_t namedPbrMaterialCount = 0;
  uint32_t namedPbrTextureCount = 0;
  uint32_t cachedReplacementCount = 0;
  uint32_t failedReplacementCount = 0;
  uint32_t reportedMissingCount = 0;
  uint32_t watchedFileCount = 0;
  uint64_t cachedReplacementBytes = 0;
  bool autoRefreshEnabled = false;
  bool directorySignatureValid = false;
};

struct DebugTexturePreview {
  ImTextureID textureId = 0;
  std::string formatName;
  std::string error;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mipCount = 0;
  uint64_t byteSize = 0;
};

std::vector<DebugMaterialInfo> debug_collect_materials();
DebugInventoryStats debug_inventory_stats();
DebugTexturePreview debug_load_texture_preview(const std::filesystem::path& path);
void debug_clear_preview_cache();
} // namespace aurora::gfx::texture_replacement

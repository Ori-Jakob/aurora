#pragma once

#include "map_pick.hpp"

#include "../graph.hpp"
#include "../../gx/gx.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace aurora::gfx {
struct TextureRef;
} // namespace aurora::gfx

namespace aurora::render::scene {
struct DrawClass;
} // namespace aurora::render::scene

namespace aurora::render::scene::atlas {
inline constexpr uint32_t PageSize = 4096;
inline constexpr uint32_t TileSize = 128;
inline constexpr uint32_t TilesPerRow = PageSize / TileSize;
inline constexpr uint32_t MaxTiles = TilesPerRow * TilesPerRow;
inline constexpr uint32_t MaxMaterials = 4096;
inline constexpr uint32_t NoMaterial = 0xFFFF;
inline constexpr uint32_t NoTile = 0xFFFF;
inline constexpr uint32_t BlitsPerFrame = 16;
inline constexpr uint32_t BlitRetryLatches = 3;
inline constexpr uint32_t MaxTaps = 4;
inline constexpr uint32_t MaterialWords = 8;
inline constexpr uint64_t MaterialsBytes = uint64_t{MaxMaterials} * MaterialWords * 4;
inline constexpr uint64_t MeansBytes = uint64_t{MaxTiles} * 4;
inline constexpr wgpu::TextureFormat PageFormat = wgpu::TextureFormat::RGBA8Unorm;

namespace material_flag {
inline constexpr uint32_t AlphaTest = 1u << 0;
inline constexpr uint32_t AlbedoNoUv = 1u << 1;
inline constexpr uint32_t AlphaNoUv = 1u << 2;
inline constexpr uint32_t AlbedoPending = 1u << 3;
inline constexpr uint32_t AlphaPending = 1u << 4;
inline constexpr uint32_t AlbedoMissing = 1u << 5;
inline constexpr uint32_t AlphaMissing = 1u << 6;
} // namespace material_flag

namespace attribute_flag {
inline constexpr uint32_t WordKnown = 1u << 0;
inline constexpr uint32_t CullNone = 1u << 1;
} // namespace attribute_flag

struct GpuMaterial {
  std::array<uint32_t, MaterialWords> words{};
  bool operator==(const GpuMaterial&) const = default;
};

struct DrawAttributes {
  uint32_t word = 0;
  uint32_t constColor = 0xFFFFFFFF;
  uint32_t receiver = 0;
  bool operator==(const DrawAttributes&) const = default;
};

struct Blit {
  uint32_t tile = NoTile;
  std::shared_ptr<gfx::TextureRef> texture;
  uint32_t level = 0;
  uint32_t taps = 1;
  bool srgb = false;
};

struct Snapshot {
  std::vector<GpuMaterial> materials;
  std::vector<Blit> blits;
};

struct MaterialInfo {
  bool valid = false;
  GXTexObj_ albedo{};
  GXTexObj_ alpha{};
  std::weak_ptr<gfx::TextureRef> albedoTexture;
  std::weak_ptr<gfx::TextureRef> alphaTexture;
};

struct Stats {
  bool active = false;
  uint32_t materials = 0;
  uint32_t materialsUsed = 0;
  uint32_t tiles = 0;
  uint32_t tilesReady = 0;
  uint32_t tilesPending = 0;
  uint64_t resolved = 0;
  uint64_t blits = 0;
  uint64_t acks = 0;
  uint64_t retries = 0;
  uint64_t evictions = 0;
  uint64_t missing = 0;
  uint64_t unsupported = 0;
  uint64_t copies = 0;
  uint64_t noCache = 0;
  uint64_t noUv = 0;
  uint64_t pageFull = 0;
  uint64_t noMaterial = 0;
  uint64_t attributeCaptures = 0;
  uint64_t attributeRecomputes = 0;
};

[[nodiscard]] GpuMaterial pack_material(uint32_t albedoTile, uint32_t alphaTile, uint32_t wraps, uint32_t alphaChannel,
                                        bool alphaUsesUv0, uint32_t flags, const gx::AlphaCompare& compare,
                                        uint32_t registryId) noexcept;
[[nodiscard]] std::array<uint32_t, 3> pack_attributes(const DrawAttributes& attributes) noexcept;
[[nodiscard]] Blit blit_for(uint32_t tile, std::shared_ptr<gfx::TextureRef> texture) noexcept;
[[nodiscard]] bool supported_format(wgpu::TextureFormat format) noexcept;

// FIFO thread
[[nodiscard]] bool active() noexcept;
[[nodiscard]] uint64_t material_tuple(const gx::GXState& state, const map_pick::MapSelection& maps) noexcept;
// The returned id carries one reference for the caller
[[nodiscard]] uint32_t material_for(const gx::GXState& state, const map_pick::MapSelection& maps,
                                    bool visible) noexcept;
void release(uint32_t id) noexcept;
[[nodiscard]] DrawAttributes draw_attributes(const DrawAttributes& last, uint64_t& tuple,
                                             const gx::PipelineConfig* config, const gx::ShaderInfo* info,
                                             const DrawClass* cls, const gx::GXState& state) noexcept;
[[nodiscard]] std::shared_ptr<const Snapshot> snapshot() noexcept;
[[nodiscard]] MaterialInfo info(uint32_t id) noexcept;

// Game thread while the FIFO is idle
void initialize();
void shutdown();
void set_active(bool active) noexcept;
void latch_resources(bool active) noexcept;
void latch() noexcept;
void reset() noexcept;
[[nodiscard]] Stats published_stats() noexcept;
[[nodiscard]] detail::PersistentHandle page_handle() noexcept;
[[nodiscard]] detail::PersistentHandle materials_handle() noexcept;
[[nodiscard]] detail::PersistentHandle means_handle() noexcept;
[[nodiscard]] TexDesc page_desc() noexcept;

// Render worker
void ack(std::span<const uint32_t> tiles) noexcept;
[[nodiscard]] detail::PassDesc pass_desc();

namespace testing {
using ResolveFn = void (*)(const gx::ShaderInfo& info) noexcept;

struct Tile {
  bool live = false;
  bool ready = false;
  uint32_t refs = 0;
};

void reset() noexcept;
void set_active(bool active) noexcept;
void set_resolver(ResolveFn resolve) noexcept;
[[nodiscard]] uint32_t refs(uint32_t material) noexcept;
[[nodiscard]] Tile tile(uint32_t tile) noexcept;
[[nodiscard]] uint32_t material_tile(uint32_t material, bool alpha) noexcept;
bool encode_blits(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& page, const BufferView& means,
                  const Snapshot& snapshot);
} // namespace testing
} // namespace aurora::render::scene::atlas

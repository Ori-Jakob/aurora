#pragma once

#include "../graph.hpp"
#include "../stage_camera.hpp"

#include <aurora/render.hpp>
#include <dolphin/gx/GXAurora.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::detail {
struct SceneHints;
} // namespace aurora::render::detail

namespace aurora::render::detail::lighting {
[[nodiscard]] bool lighting_wanted(const PassSetup& setup) noexcept;
} // namespace aurora::render::detail::lighting

namespace aurora::render::detail::lighting::clusters {
inline constexpr std::string_view Pass = "light.clusters";
inline constexpr std::string_view HeatmapPass = "light.heatmap";
inline constexpr std::string_view HeatmapResource = "light.heatmap";
inline constexpr std::string_view List = "light.list";
inline constexpr std::string_view Grid = "light.grid";
inline constexpr std::string_view Shader = "lighting/clusters";
inline constexpr std::string_view HeatmapShader = "lighting/heatmap";

inline constexpr uint32_t TilesX = 16;
inline constexpr uint32_t TilesY = 9;
inline constexpr uint32_t Slices = 24;
inline constexpr uint32_t ClusterCount = TilesX * TilesY * Slices;
inline constexpr uint32_t MaskWords = 8;
inline constexpr uint32_t MaxLights = 256;
inline constexpr uint32_t MaxDirectional = 16;
inline constexpr float SliceNearMetres = 0.5f;
inline constexpr float SliceFarMetres = 1000.f;
inline constexpr float RadiusInflate = 1e-3f;
inline constexpr float DepthInflate = 1e-5f;
inline constexpr uint32_t ListVersion = 1;
inline constexpr uint32_t WorkgroupSize = 64;
inline constexpr uint32_t HeatmapWorkgroupSize = 8;
inline constexpr float MaxColour = 65504.f / 3.14159265358979323846f;
inline constexpr float MaxFalloff = 4.f;
inline constexpr float DefaultFalloff = 2.f;
static_assert(MaxLights == AURORA_MAX_LIGHTS);
static_assert(ClusterCount == 3456 && ClusterCount % WorkgroupSize == 0);
static_assert(MaskWords * 32 == MaxLights);

inline constexpr std::array<float, Slices + 1> SliceScale{
    0x1p+0f,       0x1.5f627ep+0f, 0x1.e24f3ap+0f, 0x1.4b0218p+1, 0x1.c65738p+1, 0x1.37d03p+2,  0x1.abfe6ap+2,
    0x1.25bb3ep+3, 0x1.932cbcp+3,  0x1.14b2b8p+4,  0x1.7bcb8p+4,  0x1.04a712p+5, 0x1.65c558p+5, 0x1.eb1338p+5,
    0x1.510624p+6, 0x1.ce9916p+6,  0x1.3d7afp+7,   0x1.b3c5b2p+7, 0x1.2b11dcp+8, 0x1.9a808cp+8, 0x1.19ba16p+9,
    0x1.82b288p+9, 0x1.0963c8p+10, 0x1.6c45eap+10, 0x1.f4p+10,
};

enum HeaderFlags : uint32_t {
  HeaderTruncated = 1u << 0,
  HeaderMaskWidened = 1u << 1,
  HeaderStress = 1u << 2,
  HeaderSkipMask = 1u << 3,
};

enum ParamFlags : uint32_t {
  ParamBruteForce = 1u << 0,
};

struct LightListHeader {
  uint32_t dirCount = 0;
  uint32_t localCount = 0;
  uint32_t version = ListVersion;
  uint32_t flags = 0;
  float sliceNear = 0.f;
  float sliceLogScale = 0.f;
  float unitsPerMetre = 0.f;
  uint32_t pad = 0;
};
static_assert(sizeof(LightListHeader) == 32);

struct GpuLight {
  std::array<float, 4> v0{};
  std::array<float, 4> v1{};
  std::array<float, 4> v2{};
  std::array<float, 4> v3{};
  std::array<uint32_t, 4> v4{};
};
static_assert(sizeof(GpuLight) == 80 && offsetof(GpuLight, v4) == 64);

inline constexpr uint64_t ListBytes = sizeof(LightListHeader) + uint64_t{MaxLights} * sizeof(GpuLight);
inline constexpr uint64_t GridBytes = uint64_t{ClusterCount} * MaskWords * sizeof(uint32_t);
static_assert(ListBytes == 20512 && GridBytes == 110592);

struct Params {
  StageCamera camera;
  uint32_t flags = 0;
  uint32_t pad[3]{};
};
static_assert(sizeof(Params) == 64 && sizeof(Params) <= MaxPassParamsSize);

[[nodiscard]] constexpr uint32_t light_type(const GpuLight& light) noexcept { return light.v4[0] & 0xFFu; }
[[nodiscard]] constexpr uint32_t light_byte(const GpuLight& light) noexcept { return (light.v4[0] >> 8) & 0xFFu; }
[[nodiscard]] constexpr uint32_t light_flags(const GpuLight& light) noexcept { return light.v4[0] >> 16; }
[[nodiscard]] constexpr uint32_t hint_index(const GpuLight& light) noexcept { return light.v4[2] >> 8; }

struct UploadStats {
  uint32_t invalid = 0;
  uint32_t skipped = 0;
  uint32_t truncated = 0;
  uint32_t widened = 0;
};

using LightSpan = std::span<GpuLight, MaxLights>;

void prepare_lights(const SceneHints& hints, uint32_t skipMask, LightSpan lights, LightListHeader& header,
                    UploadStats& stats) noexcept;
uint32_t append_stress_lights(uint32_t count, const StageCamera& camera, LightSpan lights,
                              LightListHeader& header) noexcept;
[[nodiscard]] uint32_t slice_of(float z, const LightListHeader& header) noexcept;
[[nodiscard]] uint32_t pixel_cluster(uint32_t px, uint32_t py, uint32_t width, uint32_t height, float z,
                                     const LightListHeader& header) noexcept;
[[nodiscard]] constexpr uint32_t cluster_index(uint32_t tx, uint32_t ty, uint32_t slice) noexcept {
  return (slice * TilesY + ty) * TilesX + tx;
}
[[nodiscard]] constexpr uint32_t tile_of(uint32_t p, uint32_t size, uint32_t tiles) noexcept {
  const uint32_t extent = size == 0 ? 1 : size;
  const uint32_t tile = ((2 * p + 1) * tiles) / (2 * extent);
  return tile < tiles - 1 ? tile : tiles - 1;
}

PassDesc pass_desc();
PassDesc heatmap_pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);
[[nodiscard]] bool armed() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;

namespace testing {
struct Counters {
  uint32_t frames = 0;
  uint32_t dispatches = 0;
  uint32_t heatmapDispatches = 0;
  uint32_t maxLights = 0;
  uint32_t invalid = 0;
  uint32_t skipped = 0;
  uint32_t truncated = 0;
  uint32_t widened = 0;
};
Counters counters() noexcept;
void reset() noexcept;
void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_brute_force(std::optional<bool> bruteForce) noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting::clusters

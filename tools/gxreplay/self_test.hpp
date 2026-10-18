#pragma once

#include "trace/trace_format.hpp"

#include <dolphin/gx/GXAurora.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace aurora::gxreplay {
inline constexpr std::array<uint16_t, 5> SelfTestHintIds{0x0050, 0x0051, 0x00FF, 0x0100, 0x01FF};
inline constexpr std::array<uint32_t, 4> SelfTestPayloadSizes{0, 1, 7, 4096};
inline constexpr uint32_t SelfTestFrameCount = 3;
inline constexpr uint64_t SelfTestTextureAddress = 0x80100000;
inline constexpr uint64_t SelfTestPositionAddress = 0x80200000;
inline constexpr uint64_t SelfTestColorAddress = 0x80200100;
inline constexpr uint64_t SelfTestCopyAddress = 0x80300000;
inline constexpr uint32_t SelfTestWidth = 640;
inline constexpr uint32_t SelfTestHeight = 480;

struct SelfTestObject {
  uint64_t id;
  uint32_t categories;
};
// One per quad, in draw order; quad 3 also sits in an EMISSIVE scope
inline constexpr std::array<SelfTestObject, 3> SelfTestObjects{{
    {0x10, AURORA_CAT_TERRAIN},
    {0x11, AURORA_CAT_DECAL},
    {0x12, AURORA_CAT_FOLIAGE},
}};

struct SelfTestColor {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  bool operator==(const SelfTestColor&) const = default;
};
using SelfTestQuadColors = std::array<SelfTestColor, SelfTestObjects.size()>;

struct SelfTestCamera {
  f32 view[3][4];
  f32 proj[4][4];
};

// The valid hints of the hint variant carry these values, so tests and probes can rely on them
SelfTestCamera self_test_camera() noexcept;
AuroraLight self_test_light() noexcept;
AuroraEnvironment self_test_environment() noexcept;
AuroraTevPolicy self_test_tev_policy() noexcept;
AuroraInfluencer self_test_influencer() noexcept;
AuroraWindSource self_test_wind_source() noexcept;
AuroraGrassLightSet self_test_grass_light_set() noexcept;
AuroraSky self_test_sky() noexcept;
AuroraFogVolume self_test_fog_volume() noexcept;

trace::TraceInfo self_test_info();
uint8_t self_test_hint_byte(size_t index) noexcept;
bool write_self_test_trace(const std::filesystem::path& path, bool withHints, std::string& error);
bool check_self_test_frames(std::span<const std::vector<uint8_t>> withHints,
                            std::span<const std::vector<uint8_t>> plain, bool pixelProbes, std::string& error);
// A debug view gives each quad a colour only its classification knows, so the caller supplies one per frame and quad
bool check_self_test_view_frames(std::span<const std::vector<uint8_t>> images,
                                 std::span<const SelfTestQuadColors> quadColors, std::string& error);
} // namespace aurora::gxreplay

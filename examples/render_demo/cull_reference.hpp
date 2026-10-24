#pragma once

#include <dolphin/types.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace render_demo {
inline constexpr u32 CullWorkgroupSize = 128;
inline constexpr u32 MaxCullInstances = 1u << 22;
inline constexpr u32 CullCheckFrame = 30;
inline constexpr u32 CullCheckMinFrames = 40;
inline constexpr u32 CullIndexCount = 24;

struct CullInstance {
  f32 x, y, z, radius;
};

struct CullLayout {
  u32 instances = 0, groups = 0;
  u32 countsWord = 0, maskWord = 0, idsWord = 0, totalWords = 0;
  u64 bytes() const noexcept;
  u64 ids_offset() const noexcept;
};

inline constexpr u32 HeaderWords = 64, HeaderFrameTag = 5, HeaderTotal = 6, HeaderGroups = 7;

using Plane = std::array<f32, 4>;

struct CullParams {
  std::array<Plane, 6> planes;
  u32 frameTag;
  u32 pad[3];
};
static_assert(sizeof(CullParams) == 112);

struct CullDrawParams {
  std::array<f32, 16> viewProj;
};
static_assert(sizeof(CullDrawParams) == 64);

CullLayout cull_layout(u32 instances) noexcept;
CullInstance cull_instance(u32 id) noexcept;
void fill_cull_instances(std::span<CullInstance> out) noexcept;
std::array<f32, 16> view_projection(const f32 (&proj)[4][4], const f32 (&view)[3][4], bool reversedZ) noexcept;
std::array<Plane, 6> frustum_planes(const std::array<f32, 16>& viewProj) noexcept;

enum class CullClass : u8 { Culled, Ambiguous, Visible };

CullClass classify(const CullInstance& instance, std::span<const Plane, 6> planes) noexcept;
std::vector<u32> simulate_cull(std::span<const CullInstance> instances, std::span<const Plane, 6> planes, u32 frameTag);
std::string cull_compute_wgsl(const CullLayout& layout);
std::string cull_draw_wgsl(u32 instances);

struct CullCheckResult {
  std::optional<std::string> failure;
  u32 visible = 0, ambiguous = 0;
};

CullCheckResult check_cull_readback(std::span<const u32> words, const CullLayout& layout,
                                    std::span<const CullInstance> instances, std::span<const Plane, 6> planes,
                                    u32 expectedFrameTag);
} // namespace render_demo

#pragma once

#include "graph_checks.hpp"
#include "hints.hpp"

#include <aurora/gfx.hpp>
#include <dolphin/mtx.h>
#include <dolphin/types.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace render_demo {
inline constexpr u32 OverlayCheckFrame = 30;
inline constexpr u32 OverlayLateFrame = 32;

struct SceneOptions {
  bool hints = true;
  bool captureOnlyBox = false;
  aurora::gfx::DrawTypeId customDrawProbe = aurora::gfx::InvalidDrawType;
  bool gpuArrays = false;
  uint32_t gpuPadMiB = 0;
  uint64_t gpuCopySystem = 0;
  bool fogBands = false;
  bool aoCheck = false;
  bool fwdEnvCheck = false;
  bool unlitCheck = false;
  bool companionCheck = false;
  bool overlayCheck = false;
  bool classCheck = false;
  bool ssilCheck = false;
};

struct OverlayColors {
  std::array<u8, 4> quad{};
  std::array<u8, 4> box{};
};

struct SceneCamera {
  Mtx44 projection;
  Mtx view;
  Point3d eye;
  Point3d target;
};

enum class MotionProbe : u32 { Ground, Box, Tentacle, Water };

struct AoProbe {
  f32 ndc[2];
  f32 offsetPx;
  AoProbeClass cls;
};
static_assert(sizeof(AoProbe) == 16);

struct FwdEnvProbe {
  f32 ndc[2];
  FwdEnvProbeClass cls;
  u32 pad;
};
static_assert(sizeof(FwdEnvProbe) == 16);

struct UnlitProbe {
  f32 ndc[2];
  UnlitProbeClass cls;
  u32 pad;
};
static_assert(sizeof(UnlitProbe) == 16);

struct CompanionProbe {
  f32 ndc[2];
  u32 quad;
  u32 cell;
};
static_assert(sizeof(CompanionProbe) == 16);

struct ClassProbe {
  f32 ndc[2];
  ClassQuad quad;
  u32 pad;
};
static_assert(sizeof(ClassProbe) == 16);

struct SsilProbe {
  f32 ndc[2];
  SsilProbeClass cls;
  u32 pad;
};
static_assert(sizeof(SsilProbe) == 16);

struct FogResolveCounts {
  u32 checked = 0;
  u32 live = 0;
  u32 mismatches = 0;
};

SceneCamera scene_camera(u32 frame) noexcept;
std::array<std::array<f32, 2>, 4> motion_probe_points(u32 frame) noexcept;
std::optional<std::array<f32, 2>> motion_probe_expected(MotionProbe probe, u32 frame, std::array<f32, 2> ndc) noexcept;
const std::array<FogParams, 2>& fog_bands() noexcept;
std::vector<AoProbe> ao_check_probes();
std::vector<std::array<f32, 2>> ao_trail_probes(u32 frame);
std::vector<FwdEnvProbe> fwd_env_probes();
FwdEnvExpect fwd_env_expect() noexcept;
std::vector<UnlitProbe> unlit_probes();
std::vector<CompanionProbe> companion_probes();
CompanionAxes companion_axes() noexcept;
std::vector<ClassProbe> class_probes();
std::vector<SsilProbe> ssil_check_probes();
std::string class_manifest(bool edited);
std::span<const u8> ground_texture_base() noexcept;
std::array<f32, 2> overlay_probe() noexcept;
OverlayColors overlay_colors() noexcept;
FogResolveCounts fog_resolve_counts() noexcept;
uint32_t gpu_uploads_per_frame(const SceneOptions& options) noexcept;
bool init_scene();
void draw_scene(u32 frame, const SceneOptions& options);
void shutdown_scene();
} // namespace render_demo

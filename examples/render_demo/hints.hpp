#pragma once

#include "hint_checks.hpp"

#include <dolphin/gx.h>
#include <dolphin/mtx.h>

#include <span>

namespace render_demo {
struct FogParams {
  GXFogType type;
  f32 start;
  f32 end;
  f32 nearZ;
  f32 farZ;
  GXColor color;
};

void set_hint_emission(bool enabled) noexcept;
[[nodiscard]] bool hint_emission() noexcept;
[[nodiscard]] const HintTally& hint_tally() noexcept;
void count_capture_only_draw() noexcept;

void hint_camera(AuroraCameraRole role, const Mtx view, const Mtx44 projection, GXProjectionType type);
void hint_scene_epoch(u32 epoch);
void hint_array_immutable(const void* data, u32 size);
void hint_lights(std::span<const AuroraLight> lights);
void hint_environment(const AuroraEnvironment& environment);
void hint_influencers(std::span<const AuroraInfluencer> influencers);
void hint_wind_sources(std::span<const AuroraWindSource> sources);
void hint_fog_volumes(std::span<const AuroraFogVolume> volumes);
void hint_stage(AuroraRenderStage stage);
void begin_object(u64 id, u32 categories, u32 flags);
void end_object();
void push_scope(u32 categories, u32 flags, AuroraCameraRole role);
void pop_scope();
void hint_material(u32 materialClass);
void hint_tev_policy(const AuroraTevPolicy& policy);
void hint_skeleton(u64 objectId, std::span<const Mtx> joints);

AuroraLight directional_light(u32 id, const Vec& direction, GXColor color) noexcept;
AuroraLight point_light(u32 id, const Vec& position, GXColor color, f32 refDistance, f32 refBrightness,
                        GXDistAttnFn fn) noexcept;
AuroraLight spot_light(u32 id, const Vec& position, const Vec& direction, GXColor color, f32 cutoffDegrees,
                       f32 refDistance, f32 refBrightness, GXDistAttnFn fn) noexcept;
AuroraEnvironment environment(GXColor ambient, const FogParams& fog, const Vec& windDir, f32 windStrength) noexcept;
AuroraFogVolume fog_volume(const Vec& center, const Vec& radius, f32 density) noexcept;
} // namespace render_demo

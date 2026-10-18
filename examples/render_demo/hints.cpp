#include "hints.hpp"

#include <cmath>

namespace render_demo {
namespace {
constexpr f32 Pi = 3.14159265358979f;

bool g_emit = true;
HintTally g_tally;

template <typename Emit>
void emit(u16 id, Emit&& write) {
  if (g_emit) {
    write();
    g_tally.add(id);
  }
}

void set3(f32 (&out)[3], f32 x, f32 y, f32 z) noexcept {
  out[0] = x;
  out[1] = y;
  out[2] = z;
}

void set3(f32 (&out)[3], const Vec& value) noexcept { set3(out, value.x, value.y, value.z); }

f32 srgb_to_linear(u8 value) noexcept {
  const f32 c = static_cast<f32>(value) / 255.f;
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Lights carry linear colour, while the environment carries GX register values
void set_linear(f32 (&out)[3], GXColor color) noexcept {
  set3(out, srgb_to_linear(color.r), srgb_to_linear(color.g), srgb_to_linear(color.b));
}

void set_register(f32 (&out)[3], GXColor color) noexcept {
  set3(out, static_cast<f32>(color.r) / 255.f, static_cast<f32>(color.g) / 255.f, static_cast<f32>(color.b) / 255.f);
}

AuroraLight base_light(u32 id, AuroraLightType type, GXColor color) noexcept {
  AuroraLight light{};
  light.id = id;
  light.type = static_cast<u8>(type);
  set_linear(light.color, color);
  light.spotInnerCos = 1.f;
  light.spotOuterCos = -1.f;
  light.volumetricScale = 1.f;
  return light;
}
} // namespace

void set_hint_emission(bool enabled) noexcept { g_emit = enabled; }

bool hint_emission() noexcept { return g_emit; }

const HintTally& hint_tally() noexcept { return g_tally; }

void count_capture_only_draw() noexcept { ++g_tally.captureOnlyDraws; }

void hint_camera(AuroraCameraRole role, const Mtx view, const Mtx44 projection, GXProjectionType type) {
  emit(GX_AURORA_RENDER_CAMERA, [&] { GXAuroraSetCamera(role, view, projection, type, 0); });
}

void hint_scene_epoch(u32 epoch) {
  emit(GX_AURORA_RENDER_SCENE_EPOCH, [&] { GXAuroraSetSceneEpoch(epoch); });
}

void hint_array_immutable(const void* data, u32 size) {
  emit(GX_AURORA_SET_ARRAY_IMMUTABLE, [&] { GXAuroraSetArrayImmutable(data, size, GX_TRUE, 1); });
}

void hint_lights(std::span<const AuroraLight> lights) {
  emit(GX_AURORA_RENDER_LIGHTS, [&] { GXAuroraSetLights(lights.data(), static_cast<u16>(lights.size())); });
}

void hint_environment(const AuroraEnvironment& environment) {
  emit(GX_AURORA_RENDER_ENVIRONMENT, [&] { GXAuroraSetEnvironment(&environment); });
}

void hint_influencers(std::span<const AuroraInfluencer> influencers) {
  emit(GX_AURORA_RENDER_INFLUENCERS,
       [&] { GXAuroraSetInfluencers(influencers.data(), static_cast<u16>(influencers.size())); });
}

void hint_wind_sources(std::span<const AuroraWindSource> sources) {
  emit(GX_AURORA_RENDER_WIND_SOURCES,
       [&] { GXAuroraSetWindSources(sources.data(), static_cast<u16>(sources.size())); });
}

void hint_fog_volumes(std::span<const AuroraFogVolume> volumes) {
  emit(GX_AURORA_RENDER_FOG_VOLUMES, [&] { GXAuroraSetFogVolumes(volumes.data(), static_cast<u16>(volumes.size())); });
}

void hint_stage(AuroraRenderStage stage) {
  emit(GX_AURORA_RENDER_STAGE, [&] { GXAuroraStage(stage); });
}

void begin_object(u64 id, u32 categories, u32 flags) {
  emit(GX_AURORA_RENDER_OBJECT_BEGIN, [&] { GXAuroraBeginObject(id, 0, 1, categories, flags); });
}

void end_object() {
  emit(GX_AURORA_RENDER_OBJECT_END, [] { GXAuroraEndObject(); });
}

void push_scope(u32 categories, u32 flags, AuroraCameraRole role) {
  emit(GX_AURORA_RENDER_SCOPE_PUSH, [&] { GXAuroraPushScope(categories, flags, role); });
}

void pop_scope() {
  emit(GX_AURORA_RENDER_SCOPE_POP, [] { GXAuroraPopScope(); });
}

void hint_material(u32 materialClass) {
  emit(GX_AURORA_RENDER_MATERIAL, [&] { GXAuroraSetMaterialClass(materialClass); });
}

void hint_tev_policy(const AuroraTevPolicy& policy) {
  emit(GX_AURORA_RENDER_TEV_POLICY, [&] { GXAuroraSetTevPolicy(&policy); });
}

void hint_skeleton(u64 objectId, std::span<const Mtx> joints) {
  emit(GX_AURORA_RENDER_SKELETON,
       [&] { GXAuroraSetSkeleton(objectId, joints.data(), static_cast<u16>(joints.size())); });
}

AuroraLight directional_light(u32 id, const Vec& direction, GXColor color) noexcept {
  AuroraLight light = base_light(id, AURORA_LIGHT_DIRECTIONAL, color);
  set3(light.direction, direction);
  return light;
}

AuroraLight point_light(u32 id, const Vec& position, GXColor color, f32 refDistance, f32 refBrightness,
                        GXDistAttnFn fn) noexcept {
  AuroraLight light = base_light(id, AURORA_LIGHT_POINT, color);
  set3(light.position, position);
  const DistanceFalloff falloff = distance_falloff(refDistance, refBrightness, fn);
  light.radius = falloff.radius;
  light.falloffExponent = falloff.exponent;
  return light;
}

AuroraLight spot_light(u32 id, const Vec& position, const Vec& direction, GXColor color, f32 cutoffDegrees,
                       f32 refDistance, f32 refBrightness, GXDistAttnFn fn) noexcept {
  AuroraLight light = base_light(id, AURORA_LIGHT_SPOT, color);
  set3(light.position, position);
  set3(light.direction, direction);
  const DistanceFalloff falloff = distance_falloff(refDistance, refBrightness, fn);
  light.radius = falloff.radius;
  light.falloffExponent = falloff.exponent;
  // GX_SP_COS2 starts falling off right at the axis, so there is no full-intensity inner cone
  light.spotInnerCos = 1.f;
  light.spotOuterCos = std::cos(cutoffDegrees * Pi / 180.f);
  return light;
}

AuroraEnvironment environment(GXColor ambient, const FogParams& fog, const Vec& windDir, f32 windStrength) noexcept {
  AuroraEnvironment env{};
  env.skyVisibility = 1.f;
  set_register(env.ambientColor, ambient);
  set_register(env.fogColor, fog.color);
  set3(env.windDir, windDir);
  env.windStrength = windStrength;
  env.timeOfDay = 12.f;
  env.unitsPerMeter = 1.f;
  env.fogType = static_cast<u32>(fog.type);
  env.fogStart = fog.start;
  env.fogEnd = fog.end;
  env.viewNear = fog.nearZ;
  env.viewFar = fog.farZ;
  env.shadowSun = 1.f;
  env.shadowLocal = 1.f;
  env.grassDensityScale = 1.f;
  // Negative values ask for the engine default; phaseG spans [-1, 1], so its sentinel is -2
  env.hazeAtFogEnd = -1.f;
  env.hazeDensity = -1.f;
  set3(env.mediaAlbedo, -1.f, -1.f, -1.f);
  env.phaseG = -2.f;
  env.phaseG2 = -2.f;
  env.phaseBlend = -1.f;
  env.noiseScale = -1.f;
  env.noiseAmount = -1.f;
  env.relightStrength = -1.f;
  env.relightAmbient = -1.f;
  env.sunGlow = -1.f;
  env.volumeFar = -1.f;
  return env;
}

AuroraFogVolume fog_volume(const Vec& center, const Vec& radius, f32 density) noexcept {
  AuroraFogVolume volume{};
  set3(volume.center, center);
  set3(volume.radius, radius);
  volume.shape = AURORA_FOG_SHAPE_ELLIPSOID;
  volume.mode = AURORA_FOG_MODE_ADD;
  volume.density = density;
  return volume;
}
} // namespace render_demo

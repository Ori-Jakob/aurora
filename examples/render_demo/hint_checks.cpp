#include "hint_checks.hpp"

#include <cmath>
#include <cstdio>

namespace render_demo {
namespace {
std::string failure(u16 id, const char* what, u64 observed, u64 emitted) {
  const std::string_view name = hint_name(id);
  char text[160];
  std::snprintf(text, sizeof(text), "%.*s (0x%04X): %s %llu, emitted %llu", static_cast<int>(name.size()), name.data(),
                id, what, static_cast<unsigned long long>(observed), static_cast<unsigned long long>(emitted));
  return std::string{text};
}
} // namespace

void HintTally::add(u16 id, u32 count) noexcept {
  if (id >= FirstHintId && id <= LastHintId) {
    emitted[id - FirstHintId] += count;
  }
}

u32 HintTally::at(u16 id) const noexcept {
  return id >= FirstHintId && id <= LastHintId ? emitted[id - FirstHintId] : 0;
}

u64 HintTally::total() const noexcept {
  u64 sum = 0;
  for (const u32 count : emitted) {
    sum += count;
  }
  return sum;
}

// gx::hints::name is internal to aurora, so the demo keeps its own copy of the catalogue names
std::string_view hint_name(u16 id) noexcept {
  switch (id) {
  case GX_AURORA_SET_ARRAY_IMMUTABLE:
    return "SET_ARRAY_IMMUTABLE";
  case GX_AURORA_RELEASE_ARRAY_IMMUTABLE:
    return "RELEASE_ARRAY_IMMUTABLE";
  case GX_AURORA_RENDER_CAMERA:
    return "CAMERA";
  case GX_AURORA_RENDER_OBJECT_BEGIN:
    return "OBJECT_BEGIN";
  case GX_AURORA_RENDER_OBJECT_END:
    return "OBJECT_END";
  case GX_AURORA_RENDER_SCOPE_PUSH:
    return "SCOPE_PUSH";
  case GX_AURORA_RENDER_SCOPE_POP:
    return "SCOPE_POP";
  case GX_AURORA_RENDER_STAGE:
    return "STAGE";
  case GX_AURORA_RENDER_LIGHTS:
    return "LIGHTS";
  case GX_AURORA_RENDER_ENVIRONMENT:
    return "ENVIRONMENT";
  case GX_AURORA_RENDER_MATERIAL:
    return "MATERIAL";
  case GX_AURORA_RENDER_SKELETON:
    return "SKELETON";
  case GX_AURORA_RENDER_SCENE_EPOCH:
    return "SCENE_EPOCH";
  case GX_AURORA_RENDER_TEV_POLICY:
    return "TEV_POLICY";
  case GX_AURORA_RENDER_INFLUENCERS:
    return "INFLUENCERS";
  case GX_AURORA_RENDER_WIND_SOURCES:
    return "WIND_SOURCES";
  case GX_AURORA_RENDER_GRASS_PATCHES:
    return "GRASS_PATCHES";
  case GX_AURORA_RENDER_SKY:
    return "SKY";
  case GX_AURORA_RENDER_FOG_VOLUMES:
    return "FOG_VOLUMES";
  case GX_AURORA_GPU_UPLOAD:
    return "GPU_UPLOAD";
  case GX_AURORA_GPU_RESERVE:
    return "GPU_RESERVE";
  case GX_AURORA_GPU_DISPATCH:
    return "GPU_DISPATCH";
  case GX_AURORA_LOAD_ARRAY_GPU:
    return "LOAD_ARRAY_GPU";
  case GX_AURORA_GPU_UPDATE_PERSISTENT:
    return "GPU_UPDATE_PERSISTENT";
  case GX_AURORA_GPU_CREATE_SYSTEM:
    return "GPU_CREATE_SYSTEM";
  case GX_AURORA_GPU_DESTROY_SYSTEM:
    return "GPU_DESTROY_SYSTEM";
  default:
    return "reserved";
  }
}

std::optional<std::string> check_hints(const HintTally& tally, bool decoding, CounterLookup lookup, void* userData,
                                       u64 suppressedDraws) {
  for (u32 value = FirstHintId; value <= LastHintId; ++value) {
    const auto id = static_cast<u16>(value);
    const ObservedCounters seen = lookup(id, userData);
    const u64 emitted = tally.at(id);
    if (seen.malformed != 0) {
      return failure(id, "malformed", seen.malformed, emitted);
    }
    if (seen.late != 0) {
      return failure(id, "late", seen.late, emitted);
    }
    if (seen.decoded + seen.skipped != emitted) {
      return failure(id, "decoded + skipped", seen.decoded + seen.skipped, emitted);
    }
    // Core hints decode without the render module
    const bool core = id <= GX_AURORA_HINT_CORE_LAST;
    // Only the M3 decoders are pinned, so a later milestone that starts decoding another ID keeps this passing
    const bool decodedInM3 = id >= GX_AURORA_RENDER_CAMERA && id <= GX_AURORA_RENDER_WIND_SOURCES;
    if ((core || (decoding && decodedInM3)) && seen.decoded != emitted) {
      return failure(id, "decoded", seen.decoded, emitted);
    }
    if (!decoding && !core && seen.decoded != 0) {
      return failure(id, "decoded with decoding off", seen.decoded, emitted);
    }
  }
  const u64 expected = decoding ? tally.captureOnlyDraws : 0;
  if (suppressedDraws != expected) {
    char text[96];
    std::snprintf(text, sizeof(text), "%llu draws suppressed, expected %llu",
                  static_cast<unsigned long long>(suppressedDraws), static_cast<unsigned long long>(expected));
    return std::string{text};
  }
  return std::nullopt;
}

DistanceFalloff distance_falloff(f32 refDistance, f32 refBrightness, GXDistAttnFn fn) noexcept {
  // The same argument guard and f32 coefficients as GXInitLightDistAttn, so both values describe the XF light
  if (refDistance < 0.f || refBrightness < 0.f || refBrightness >= 1.f) {
    fn = GX_DA_OFF;
  }
  DistanceFalloff out;
  f32 k1 = 0.f;
  f32 k2 = 0.f;
  switch (fn) {
  case GX_DA_GENTLE:
    out.exponent = 1.f;
    k1 = (1.f - refBrightness) / (refBrightness * refDistance);
    break;
  case GX_DA_MEDIUM:
    out.exponent = 1.5f;
    k1 = 0.5f * (1.f - refBrightness) / (refBrightness * refDistance);
    k2 = 0.5f * (1.f - refBrightness) / (refBrightness * refDistance * refDistance);
    break;
  case GX_DA_STEEP:
    out.exponent = 2.f;
    k2 = (1.f - refBrightness) / (refBrightness * refDistance * refDistance);
    break;
  default:
    // A light without distance falloff has no reach to send, and an infinite radius must never reach the wire
    return out;
  }
  // k0 is 1, so the attenuation 1 / (k0 + k1 d + k2 d^2) reaches 1/255 where k1 d + k2 d^2 = 254
  const double a = k2;
  const double b = k1;
  const auto radius = static_cast<f32>(a > 0.0 ? (-b + std::sqrt(b * b + 4.0 * a * 254.0)) / (2.0 * a) : 254.0 / b);
  // A zero reference distance or brightness gives infinite coefficients, so such a light reaches nothing
  out.radius = std::isfinite(radius) ? radius : 0.f;
  return out;
}
} // namespace render_demo

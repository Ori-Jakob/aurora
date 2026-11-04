#pragma once

#include "../gfx/scene_targets.hpp"
#include "gx.hpp"
#include "pipeline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace aurora::gx::render_ext {
struct StochasticParams {
  float scale = 1.f;
  float rotation = 0.f;
  float contrast = 0.6f;
  float exponent = 7.f;
};

inline Vec4<float> pack(const StochasticParams& params) noexcept {
  constexpr StochasticParams Defaults{};
  const auto finite_or = [](float value, float fallback) { return std::isfinite(value) ? value : fallback; };
  float invScale = 1.f;
  if (std::isfinite(params.scale) && params.scale > 0.f) {
    invScale = params.scale > 16.f ? 1.f / 16.f : 1.f / params.scale;
  }
  return {invScale, std::clamp(finite_or(params.rotation, Defaults.rotation), 0.f, 1.f),
          std::clamp(finite_or(params.contrast, Defaults.contrast), 0.f, 1.f),
          std::clamp(finite_or(params.exponent, Defaults.exponent), 1.f, 32.f)};
}

using StochasticLanes = std::array<Vec4<float>, MaxTextures>;
// FIFO thread only; written at the ext step, read by fill_uniform
inline StochasticLanes g_stochasticUniforms{};

inline bool store_lanes(const StochasticLanes& lanes, u8 mask) noexcept {
  bool changed = false;
  for (u32 i = 0; i < MaxTextures; ++i) {
    if ((mask >> i & 1u) != 0 && g_stochasticUniforms[i] != lanes[i]) {
      g_stochasticUniforms[i] = lanes[i];
      changed = true;
    }
  }
  return changed;
}

struct CompanionLane {
  Vec4<float> v{0.f, 0.f, 1.f, 0.f};

  bool operator==(const CompanionLane&) const = default;
};
inline constexpr u32 LaneAlbedo = 1;
inline constexpr u32 LaneNrm = 2;
inline constexpr u32 LaneMat = 4;
inline constexpr u32 LaneOrm = 8;
inline constexpr u32 LaneEmit = 16;
inline constexpr u32 LaneDelight = 32;
inline constexpr float MatNormalYSign = -1.f;
inline constexpr float MatNormalStrength = 0.5f;
inline constexpr float NrmNormalStrength = 1.f;
inline CompanionLane g_companionLane{};

inline bool store_companion_lane(const CompanionLane& lane) noexcept {
  if (g_companionLane == lane) {
    return false;
  }
  g_companionLane = lane;
  return true;
}

struct CompanionGates {
  bool materials = false;
  bool matMaps = false;
  bool normals = false;
  bool delight = false;

  bool operator==(const CompanionGates&) const = default;
};
inline CompanionGates g_companionGates{};

inline bool set_companion_gates(const CompanionGates& gates) noexcept {
  if (g_companionGates == gates) {
    return false;
  }
  g_companionGates = gates;
  return true;
}

// Manifest scalars are render-module data, so the core knows only each role's default
inline CompanionLane companion_lane(u32 bits, const CompanionGates& gates) noexcept {
  float strength = 0.f;
  if (gates.normals) {
    strength = (bits & LaneNrm) != 0 ? NrmNormalStrength : (bits & LaneMat) != 0 ? MatNormalStrength : 0.f;
  }
  const float ySign = (bits & LaneMat) != 0 ? MatNormalYSign : 1.f;
  return {.v = {static_cast<float>(bits), strength, ySign, 0.f}};
}

// The FIFO also derives for a pending layout, so derive only writes lanes and keeps output bits to G-buffer layouts
struct Hooks {
  RenderExtConfig (*derive)(void* userData, const PipelineConfig& config, const ShaderInfo& info,
                            const gfx::RenderTargetLayout& layout, StochasticLanes& lanes) noexcept = nullptr;
  // Scalars are uniform data, so a manifest edit changes the lane and never a pipeline
  CompanionLane (*companion_params)(void* userData, const gfx::TextureMaterialRef& material,
                                    const CompanionLane& lane) noexcept = nullptr;
  void* userData = nullptr;
};

inline std::atomic<const Hooks*> g_hooks{nullptr};

// Swapped only while the FIFO is drained, like draw_route::set_router
inline void set_hooks(const Hooks* hooks) noexcept { g_hooks.store(hooks, std::memory_order_release); }
inline const Hooks* hooks() noexcept { return g_hooks.load(std::memory_order_acquire); }

// FIFO thread only; the router stores the routed draw's class inputs here and derive reads them
inline u32 g_drawInputs = 0;
// abuf layout of the record that DrawImmediateData::_pad locates; 0 means no record
inline constexpr u32 DrawRecordWords = 8;
inline constexpr u32 DrawRecordFlagsWord = 1;
inline constexpr u32 DrawRecordReceiverShift = 6;
inline constexpr u32 DrawRecordPrevMtxWord = 4;
inline constexpr u32 DrawRecordMaterialWord = 5;
inline constexpr u32 MaterialWordRoughnessFactor = 1u << 24;
inline constexpr u32 MaterialWordMetalnessFactor = 1u << 25;
inline constexpr u32 MaterialWordFlagMask = 0x1Cu << 24;
inline constexpr u32 MaterialWordTwoSided = 0x8u << 24;
inline constexpr u32 MaterialWordAmbient = 1u << 31;
inline constexpr u32 MotionSourceMask = 3u;
inline constexpr u32 MotionSourceNone = 0;
inline constexpr u32 MotionSourceCamera = 1;
inline constexpr u32 MotionSourceObject = 2;
inline constexpr u32 MotionSourcePalette = 3;
inline constexpr u32 RecordFlagJitter = 1u << 2;
inline constexpr u32 RecordVertexColorShift = 3;
inline constexpr u32 RecordFlagUnlit = 1u << 5;
inline constexpr u32 DrawRecordTevPolicyWord = 7;
// Core-visible copy of records::flag::ScreenAo for the generator
inline constexpr u32 RecordFlagScreenAo = 1u << 11;
inline constexpr u32 RecordFlagPrevProjFrame = 1u << 12;
inline constexpr u32 RecordFlagReactive = 1u << 13;
inline constexpr u32 RecordFlagBackFaceFlip = 1u << 14;
// Twelve bits cover the Surface table and keep record bits 28-31 free
inline constexpr u32 RecordSurfaceSlotShift = 16;
inline constexpr u32 RecordSurfaceSlotBits = 12;
inline constexpr u32 RecordSurfaceSlotMask = ((1u << RecordSurfaceSlotBits) - 1u) << RecordSurfaceSlotShift;
inline constexpr u32 FrameMotionWord = 4;
inline constexpr u32 FrameMotionJitterWord = 0;
inline constexpr u32 FrameMotionValidWord = 2;
inline constexpr u32 FrameMotionPrevProjWord = 4;
inline constexpr u32 FrameMotionReprojectWord = 16;
inline constexpr u32 FrameMotionWords = 28;
inline constexpr float MotionInvalidNdc = 4.f;
inline constexpr u32 MotionPaletteSlots = 10;

inline bool stochastic_bind_ok(const gfx::TextureBind& bind) noexcept {
  return bind.ref != nullptr && !bind.ref->attachmentTextureView && bind.texObj.wrap_s() == GX_REPEAT &&
         bind.texObj.wrap_t() == GX_REPEAT;
}

struct Stats {
  std::atomic_uint64_t fallbackDraws{0};
  std::atomic_uint64_t pendingDraws{0};
  std::atomic_uint64_t failedVariants{0};
  std::atomic_uint64_t budgetDrops{0};
  std::atomic_uint64_t pbrFallbackOutput1{0};
  std::atomic_uint64_t pbrFallbackOutput0{0};
  std::atomic_uint64_t pbrTwinAdopted{0};
  std::atomic_uint64_t fwdEnvDemotions{0};
  std::atomic_uint64_t pbrAmbientUnknown{0};
};
inline Stats g_stats;

inline RenderExtConfig normalize(RenderExtConfig ext, const ShaderInfo& info) noexcept {
  ext.stochasticMask &= static_cast<u8>(info.stochasticEligible.to_ulong());
  if (ext.output == 2 && ext.fwdEnv != 0) {
    ext.output = 1;
    g_stats.fwdEnvDemotions.fetch_add(1, std::memory_order_relaxed);
  }
  if (ext.output != 2) {
    ext.layer0 = 0;
    ext.layer1 = 0;
    ext.pom = 0;
    ext.surfaceExt = 0;
  }
  // Only fallback_config sets it, so a derived config never carries it
  ext.surfaceReset = 0;
  if (ext.layer0 != 0) {
    const u32 map = ext.layer0 - 1u;
    if (map >= MaxTextures || !info.sampledTextures.test(map) || info.sampledIndTextures.test(map) ||
        (ext.stochasticMask >> map & 1u) != 0) {
      ext.layer0 = 0;
    } else if (!info.companionFits) {
      ext.layer0 = 0;
      g_stats.budgetDrops.fetch_add(1, std::memory_order_relaxed);
    }
  }
  ext.layer1 = 0;
  ext.pom = 0;
  return ext;
}

inline RenderExtConfig without_texture_bits(RenderExtConfig ext) noexcept {
  ext.stochasticMask = 0;
  ext.layer0 = 0;
  ext.layer1 = 0;
  ext.pom = 0;
  return ext;
}

inline PipelineConfig legacy_config(const PipelineConfig& config) noexcept {
  PipelineConfig legacy = config;
  legacy.shaderConfig.ext = without_texture_bits(legacy.shaderConfig.ext);
  return legacy;
}

// Draws as plain GX under the same layout; later milestones clear their own output bits here too
inline PipelineConfig fallback_config(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) noexcept {
  PipelineConfig fallback = legacy_config(config);
  // An in-window stand-in must still zero Surface, or its pixels would keep the record of the draw below
  fallback.shaderConfig.ext.surfaceReset =
      config.shaderConfig.ext.output != 0 && gfx::layout_has_surface(layout) ? 1 : 0;
  fallback.shaderConfig.ext.output = 0;
  fallback.shaderConfig.ext.velocity = 0;
  fallback.shaderConfig.ext.fwdEnv = 0;
  fallback.shaderConfig.ext.surfaceExt = 0;
  return fallback;
}

// normalize drops surfaceReset, so a stored stand-in is checked against the one shape fallback_config writes
inline bool stand_in_ext(const RenderExtConfig& ext) noexcept {
  return ext == RenderExtConfig{.tessFetch = ext.tessFetch, .surfaceReset = 1};
}

// A PBR draw still clears the surface class while its output-2 variant compiles
inline PipelineConfig output1_config(const PipelineConfig& config) noexcept {
  PipelineConfig twin = config;
  if (twin.shaderConfig.ext.output == 2) {
    twin.shaderConfig.ext.output = 1;
  }
  twin.shaderConfig.ext.layer0 = 0;
  twin.shaderConfig.ext.layer1 = 0;
  twin.shaderConfig.ext.pom = 0;
  twin.shaderConfig.ext.surfaceExt = 0;
  return twin;
}

inline bool needs_gbuffer(const RenderExtConfig& ext) noexcept {
  return ext.output != 0 || ext.velocity != 0 || ext.fwdEnv != 0;
}

// Output bits mean something only under the M7 targets; elsewhere their text and targets are the ext-0 pipeline's
inline bool output_admitted(const RenderExtConfig& ext, const gfx::RenderTargetLayout& layout) noexcept {
  const bool surface = ext.output == 2 || ext.surfaceReset != 0 || ext.surfaceExt != 0;
  if (surface && !gfx::layout_has_surface(layout)) {
    return false;
  }
  return !needs_gbuffer(ext) || gfx::layout_has_gbuffer(layout);
}
} // namespace aurora::gx::render_ext

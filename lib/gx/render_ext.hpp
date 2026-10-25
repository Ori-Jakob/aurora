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

// The FIFO also derives for a pending layout, so derive only writes lanes and keeps output bits to G-buffer layouts
struct Hooks {
  RenderExtConfig (*derive)(void* userData, const PipelineConfig& config, const ShaderInfo& info,
                            const gfx::RenderTargetLayout& layout, StochasticLanes& lanes) noexcept = nullptr;
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
inline constexpr u32 MotionSourceMask = 3u;
inline constexpr u32 MotionSourceNone = 0;
inline constexpr u32 MotionSourceCamera = 1;
inline constexpr u32 MotionSourceObject = 2;
inline constexpr u32 MotionSourcePalette = 3;
inline constexpr u32 RecordFlagJitter = 1u << 2;
inline constexpr u32 RecordFlagPrevProjFrame = 1u << 12;
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

inline RenderExtConfig normalize(RenderExtConfig ext, const ShaderInfo& info) noexcept {
  ext.stochasticMask &= static_cast<u8>(info.stochasticEligible.to_ulong());
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
inline PipelineConfig fallback_config(const PipelineConfig& config) noexcept {
  PipelineConfig fallback = legacy_config(config);
  fallback.shaderConfig.ext.output = 0;
  fallback.shaderConfig.ext.velocity = 0;
  return fallback;
}

inline bool needs_gbuffer(const RenderExtConfig& ext) noexcept {
  return ext.output != 0 || ext.velocity != 0 || ext.fwdEnv != 0;
}

// Output bits mean something only under the M7 targets; elsewhere their text and targets are the ext-0 pipeline's
inline bool output_admitted(const RenderExtConfig& ext, const gfx::RenderTargetLayout& layout) noexcept {
  return !needs_gbuffer(ext) || gfx::layout_has_gbuffer(layout);
}

struct Stats {
  std::atomic_uint64_t fallbackDraws{0};
  std::atomic_uint64_t pendingDraws{0};
  std::atomic_uint64_t failedVariants{0};
  std::atomic_uint64_t budgetDrops{0};
};
inline Stats g_stats;
} // namespace aurora::gx::render_ext

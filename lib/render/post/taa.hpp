#pragma once

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace aurora::render::detail::taa {
inline constexpr std::string_view Pass = "aurora.taa";
inline constexpr std::string_view History = "taa.color";
inline constexpr std::string_view Shader = "post/taa";
inline constexpr wgpu::TextureFormat HistoryFormat = wgpu::TextureFormat::RGBA16Float;
inline constexpr uint64_t RetainFrames = 8;
inline constexpr float MinBlend = 0.015f;
inline constexpr float MovingSamples = 10.f;

struct Params {
  float sharpness = 0.f;
  uint32_t flags = 0;
  uint32_t pad[2]{};
};
static_assert(sizeof(Params) == 16);

constexpr float rcas_strength(float sharpness) noexcept { return sharpness > 0.f ? std::min(sharpness, 1.f) : 0.f; }

constexpr TexDesc history_desc() noexcept {
  return {
      .format = HistoryFormat,
      .sizeClass = SizeClass::Efb,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
void worker_tick(uint64_t frameId) noexcept;
void on_latch(const Settings& settings, const CompiledGraph* graph);
[[nodiscard]] bool wanted(const Settings& settings) noexcept;
[[nodiscard]] bool jitter_enabled() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;

namespace testing {
struct Counters {
  uint32_t resolves = 0;
  uint32_t resets = 0;
  uint32_t skippedNoScene = 0;
  uint32_t groupsCreated = 0;
  uint32_t clearOnly = 0;
};
Counters counters() noexcept;
void reset() noexcept;
shaders::RenderPipelineDesc apply_desc(wgpu::TextureFormat format);
shaders::RenderPipelineDesc resolve_desc(bool reset);
} // namespace testing
} // namespace aurora::render::detail::taa

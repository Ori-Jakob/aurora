#pragma once

#include "fsr.hpp"
#include "../upscale.hpp"
#include "../../../graph.hpp"
#include "../../../shaders.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string_view>

namespace aurora::render::detail::upscale::fsr_pass {
inline constexpr std::string_view Pass = "aurora.fsr";
inline constexpr std::string_view ApplyShader = "post/upscale/fsr/fsr_apply";
inline constexpr float ReactiveMax = 0.9f;

struct Params {
  float frameTimeMs = fsr::NominalFrameMs;
  float sharpness = 0.f;
  float jitterSign[2] = {1.f, 1.f};
  float motionSign[2] = {1.f, 1.f};
  uint32_t pad[2]{};
};
static_assert(sizeof(Params) == 32);

PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
bool prepare(bool inPlace, wgpu::TextureFormat colorFormat);
void on_latch(const Settings& settings, const CompiledGraph* graph, const Signs& signs);
[[nodiscard]] bool jitter_ready() noexcept;
[[nodiscard]] bool ready() noexcept;

namespace testing {
struct Counters {
  uint32_t encodes = 0;
  uint32_t resets = 0;
  uint32_t published = 0;
  uint32_t applied = 0;
  uint32_t skipped = 0;
  uint32_t skippedNoScene = 0;
};

struct Frame {
  bool reset = false;
  bool historyBound = false;
  fsr::Params params;
  wgpu::Texture published;
};

Counters counters() noexcept;
Frame last_frame() noexcept;
void reset() noexcept;
shaders::RenderPipelineDesc apply_desc(wgpu::TextureFormat format, bool linear = false);
} // namespace testing
} // namespace aurora::render::detail::upscale::fsr_pass

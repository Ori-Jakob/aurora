#pragma once

#include "../../graph.hpp"
#include "../../shaders.hpp"
#include "../../stage.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::detail::upscale::compose {
inline constexpr std::string_view Shader = "post/upscale/upscale_compose";
inline constexpr std::string_view Label = "upscale.compose";

struct Params {
  float scale[2] = {1.f, 1.f};
  float jitter[2] = {0.f, 0.f};
  float sceneSize[2] = {0.f, 0.f};
  float outputSize[2] = {0.f, 0.f};
  float rcasStrength = 0.f;
  uint32_t flags = 0;
  uint32_t pad[2]{};
};
static_assert(sizeof(Params) == 48);

struct Source {
  wgpu::TextureView view;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  ColorEncoding encoding = ColorEncoding::Stored;
};

struct Targets {
  wgpu::TextureView sceneColor;
  wgpu::TextureView sceneDepth;
  wgpu::Texture colorTexture;
  wgpu::TextureView color;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureView depth;
  wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined;
};

[[nodiscard]] Params make_params(const SceneComposeData& data, GraphExtent scene, GraphExtent output) noexcept;
bool prepare_pipelines(ComposeMode mode, wgpu::TextureFormat colorFormat, wgpu::TextureFormat depthFormat,
                       ColorEncoding encoding = ColorEncoding::Stored);
void prepare(SceneComposeData& data);
void encode(wgpu::CommandEncoder& cmd, const SceneComposeData& data);
void publish_source(const Source& source) noexcept;
void request_native_step(const gfx::detail::NativeStep& step, const Source& out) noexcept;
bool take_native_step(gfx::detail::NativeStep& step) noexcept;
void write_native_status(bool ok) noexcept;
void drop_native_step() noexcept;
// Render worker only; a cached bind group would otherwise keep a freed native output's memory alive
void forget_source(const wgpu::TextureView& view) noexcept;
void shutdown() noexcept;

namespace testing {
struct Counters {
  uint32_t encoded = 0;
  uint32_t sourced = 0;
  uint32_t skipped = 0;
  uint32_t native = 0;
  uint32_t statusWrites = 0;
  uint32_t lastStatus = 0;
};
Counters counters() noexcept;
void reset() noexcept;
void hold_pipelines(bool hold) noexcept;
shaders::RenderPipelineDesc pipeline_desc(ComposeMode mode, wgpu::TextureFormat colorFormat,
                                          wgpu::TextureFormat depthFormat,
                                          ColorEncoding encoding = ColorEncoding::Stored);
bool encode(wgpu::CommandEncoder& cmd, const SceneComposeData& data, const Targets& targets,
            const std::optional<Source>& source);
[[nodiscard]] bool caches_source(const wgpu::TextureView& view) noexcept;
} // namespace testing
} // namespace aurora::render::detail::upscale::compose

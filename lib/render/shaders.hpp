#pragma once

#include "shaders_embed.hpp"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::shaders {
inline constexpr uint32_t MaxColorTargets = 8;

enum class PipelineStatus : uint8_t { Pending, Ready, Failed };

struct PipelineId {
  uint32_t value = 0;
  [[nodiscard]] explicit operator bool() const noexcept { return value != 0; }
  bool operator==(const PipelineId&) const = default;
};
inline constexpr PipelineId NeverReady{UINT32_MAX};

struct Constant {
  std::string_view name;
  double value = 0.0;
};

struct ComputePipelineDesc {
  std::string_view label;
  std::string_view shader;
  std::string_view entryPoint = "main";
  std::span<const std::string_view> defines;
  std::span<const Constant> constants;
  wgpu::PipelineLayout layout;
  uint64_t layoutKey = 0;
};

struct ColorTarget {
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  bool blend = false;
  wgpu::BlendState blendState{};
  wgpu::ColorWriteMask writeMask = wgpu::ColorWriteMask::All;
};

struct RenderPipelineDesc {
  std::string_view label;
  std::string_view shader;
  std::string_view vertexEntry = "vs_main";
  std::string_view fragmentEntry = "fs_main";
  std::span<const std::string_view> defines;
  std::span<const Constant> constants;
  wgpu::PipelineLayout layout;
  uint64_t layoutKey = 0;
  wgpu::PrimitiveTopology topology = wgpu::PrimitiveTopology::TriangleList;
  wgpu::FrontFace frontFace = wgpu::FrontFace::CCW;
  wgpu::CullMode cullMode = wgpu::CullMode::None;
  uint32_t colorCount = 0;
  std::array<ColorTarget, MaxColorTargets> colors{};
  wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined;
  bool depthWrite = false;
  wgpu::CompareFunction depthCompare = wgpu::CompareFunction::Always;
  uint32_t sampleCount = 1;
};

struct PipelineStats {
  uint32_t pending = 0;
  uint32_t ready = 0;
  uint32_t failed = 0;
  uint32_t modules = 0;
};

struct PipelineFailure {
  std::string label;
  std::string message;
};

void initialize();
void quiesce();
void shutdown();

bool register_source(std::string_view id, std::string_view source);
[[nodiscard]] bool has_source(std::string_view id);
[[nodiscard]] std::span<const std::string_view> implicit_defines() noexcept;

PipelineId request_compute(const ComputePipelineDesc& desc);
PipelineId request_render(const RenderPipelineDesc& desc);
[[nodiscard]] PipelineStatus status(PipelineId id) noexcept;
[[nodiscard]] bool ready(PipelineId id) noexcept;
[[nodiscard]] bool pipelines_ready(std::span<const uint32_t> ids) noexcept;
[[nodiscard]] wgpu::ComputePipeline compute_pipeline(PipelineId id) noexcept;
[[nodiscard]] wgpu::RenderPipeline render_pipeline(PipelineId id) noexcept;
PipelineStatus wait(PipelineId id, uint64_t timeoutNs = UINT64_MAX);
bool wait_idle(uint64_t timeoutNs);
[[nodiscard]] PipelineStats stats() noexcept;
[[nodiscard]] uint64_t session() noexcept;
std::vector<PipelineFailure> failures();
bool validate_module(std::string_view id, std::span<const std::string_view> defines, std::string& error);

namespace testing {
uint32_t waiters() noexcept;
} // namespace testing
} // namespace aurora::render::shaders

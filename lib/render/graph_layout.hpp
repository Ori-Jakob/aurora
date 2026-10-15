#pragma once

#include "graph.hpp"

#include <aurora/gfx.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aurora::render::detail {
inline constexpr uint32_t FrameGroupIndex = 0;
inline constexpr uint32_t PassGroupIndex = 1;
inline constexpr uint32_t FrameUniformsSize = 256;
inline constexpr uint32_t MaxPassParamsSize = static_cast<uint32_t>(gfx::InlineDrawPayloadSize);

enum class FrameBinding : uint32_t {
  Constants = 0,
  Params = 1,
  LinearClamp = 2,
  NearestClamp = 3,
  LinearRepeat = 4,
  DepthCompare = 5,
};

struct FrameConstants {
  uint32_t efbSize[2];
  uint32_t outputSize[2];
  float invEfbSize[2];
  float invOutputSize[2];
  uint32_t frameIndex;
  uint32_t flags;
  uint32_t msaaSamples;
  uint32_t pad0;
};
static_assert(sizeof(FrameConstants) == 48);
static_assert(offsetof(FrameConstants, invEfbSize) == 16);
static_assert(offsetof(FrameConstants, frameIndex) == 32);

struct FrameUniforms {
  FrameConstants constants{};
  std::array<uint32_t, 52> reserved{};
};
static_assert(sizeof(FrameUniforms) == FrameUniformsSize);

inline constexpr uint32_t FrameFlagReversedZ = 1u << 0;
inline constexpr uint32_t FrameFlagMultisampled = 1u << 1;

std::span<const wgpu::BindGroupLayoutEntry> frame_layout_entries() noexcept;

struct BindingInputs {
  PassKind kind = PassKind::Compute;
  uint32_t paramsSize = 0;
  bool manual = false;
  std::span<const CompiledUse> uses;
  std::span<const CompiledResource> resources;
  const GraphEnvironment* env = nullptr;
  std::span<const wgpu::TextureViewDimension> bindingViews;
};

BindingType binding_type(Access access, bool buffer) noexcept;
wgpu::TextureViewDimension use_view_dimension(const CompiledUse& use, const CompiledResource& resource) noexcept;
CullReason derive_binding_plan(const BindingInputs& inputs, BindingPlan& out, ResourceId* mismatch = nullptr);
uint64_t layout_hash(const BindingPlan& plan) noexcept;
} // namespace aurora::render::detail

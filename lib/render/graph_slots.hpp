#pragma once

#include "graph.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>

namespace aurora::render::detail {
struct FormatInfo {
  uint8_t blockWidth = 1;
  uint8_t blockHeight = 1;
  uint8_t bytesPerBlock = 0;
  bool depth = false;
  bool stencil = false;
  bool compressed = false;
  wgpu::TextureSampleType sampleType = wgpu::TextureSampleType::Float;
};

constexpr uint64_t ceil_div(uint64_t value, uint64_t divisor) noexcept { return (value + divisor - 1) / divisor; }

std::optional<FormatInfo> texture_format_info(wgpu::TextureFormat format) noexcept;
wgpu::TextureViewDimension binding_view_dimension(const CompiledGraph& graph, ResourceId resource) noexcept;
TexKey make_tex_key(const CompiledGraph& graph, ResourceId resource) noexcept;
BufKey make_buf_key(const CompiledResource& resource, const GraphLimits& limits) noexcept;
uint64_t buffer_bucket(uint64_t size, uint64_t limit = UINT64_MAX) noexcept;
uint64_t key_bytes(const TexKey& key) noexcept;
uint64_t key_bytes(const BufKey& key) noexcept;

SlotPlan plan_slots(const CompiledGraph& graph, const GraphLimits& limits);

enum class PoisonMethod : uint8_t { None, RenderClear, ComputeClear, BufferCopy, BufferFill };

PoisonMethod poison_method(const TexKey& key) noexcept;
PoisonMethod poison_method(const BufKey& key) noexcept;
} // namespace aurora::render::detail

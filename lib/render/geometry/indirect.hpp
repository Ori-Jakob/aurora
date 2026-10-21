#pragma once

#include "../gpu_heap.hpp"

#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>

namespace aurora::render::indirect {
struct DrawArgs {
  uint32_t vertexCount = 0;
  uint32_t instanceCount = 0;
  uint32_t firstVertex = 0;
  uint32_t firstInstance = 0;
};

struct DrawIndexedArgs {
  uint32_t indexCount = 0;
  uint32_t instanceCount = 0;
  uint32_t firstIndex = 0;
  int32_t baseVertex = 0;
  uint32_t firstInstance = 0;
};

struct DispatchArgs {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
};

static_assert(sizeof(DrawArgs) == 16 && sizeof(DrawIndexedArgs) == 20 && sizeof(DispatchArgs) == 12);

inline constexpr uint32_t MaxWorkgroupsPerDimension = 65535;
inline constexpr uint32_t Unbounded = UINT32_MAX;

enum class Record : uint8_t { Draw, DrawIndexed, Dispatch };

constexpr uint64_t record_size(Record record) noexcept {
  switch (record) {
  case Record::Draw:
    return sizeof(DrawArgs);
  case Record::DrawIndexed:
    return sizeof(DrawIndexedArgs);
  case Record::Dispatch:
    return sizeof(DispatchArgs);
  }
  return 0;
}

constexpr DrawArgs make_draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
                             uint32_t vertexCapacity, uint32_t instanceCapacity) noexcept {
  if (firstVertex > vertexCapacity || vertexCount > vertexCapacity - firstVertex || instanceCount > instanceCapacity) {
    return {};
  }
  return {.vertexCount = vertexCount, .instanceCount = instanceCount, .firstVertex = firstVertex};
}

constexpr DrawIndexedArgs make_draw_indexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                                            int32_t baseVertex, uint32_t indexCapacity,
                                            uint32_t instanceCapacity) noexcept {
  if (firstIndex > indexCapacity || indexCount > indexCapacity - firstIndex || instanceCount > instanceCapacity) {
    return {};
  }
  return {.indexCount = indexCount, .instanceCount = instanceCount, .firstIndex = firstIndex, .baseVertex = baseVertex};
}

constexpr DispatchArgs make_dispatch(uint32_t x, uint32_t y, uint32_t z) noexcept {
  if (x > MaxWorkgroupsPerDimension || y > MaxWorkgroupsPerDimension || z > MaxWorkgroupsPerDimension) {
    return {};
  }
  return {.x = x, .y = y, .z = z};
}

BufferView buffer_view(const gpu_heap::BlockView& block) noexcept;
uint32_t capacity(const BufferView& view, Record record) noexcept;
bool draw(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t slot);
bool draw_indexed(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t slot);
uint32_t draw_slots(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t first, uint32_t count);
uint32_t draw_indexed_slots(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t first,
                            uint32_t count);
bool dispatch(const wgpu::ComputePassEncoder& pass, const BufferView& args, uint32_t slot);
bool bind_indices(const wgpu::RenderPassEncoder& pass, const BufferView& indices);

struct EncodeStats {
  uint64_t encoded = 0;
  uint64_t rejected = 0;
  uint64_t multiDraws = 0;
  uint64_t loopDraws = 0;
};

EncodeStats encode_stats() noexcept;
} // namespace aurora::render::indirect

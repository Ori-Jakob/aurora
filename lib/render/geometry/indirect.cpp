#include "indirect.hpp"

#include "../../logging.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <string_view>

namespace aurora::render::indirect {
namespace {
constexpr Module Log{"aurora::render::indirect"};

enum class Helper : uint8_t { Draw, DrawIndexed, DrawSlots, DrawIndexedSlots, Dispatch, BindIndices };
constexpr std::array<std::string_view, 6> HelperNames{
    "draw", "draw_indexed", "draw_slots", "draw_indexed_slots", "dispatch", "bind_indices",
};

std::array<std::atomic_bool, HelperNames.size()> g_warned{};
std::atomic_uint64_t g_encoded{0};
std::atomic_uint64_t g_rejected{0};
std::atomic_uint64_t g_multiDraws{0};
std::atomic_uint64_t g_loopDraws{0};

void add(std::atomic_uint64_t& counter, uint64_t value) noexcept {
  counter.fetch_add(value, std::memory_order_relaxed);
}

bool reject(Helper helper, std::string_view reason) {
  add(g_rejected, 1);
  const auto index = static_cast<size_t>(helper);
  if (!g_warned[index].exchange(true, std::memory_order_relaxed)) {
    Log.warn("indirect {}: {}, so nothing was encoded", HelperNames[index], reason);
  }
  return false;
}

bool check_args(Helper helper, const BufferView& args, Record record, uint32_t first, uint32_t count) {
  if (!args.buffer) {
    return reject(helper, "no buffer");
  }
  if (!(args.buffer.GetUsage() & wgpu::BufferUsage::Indirect)) {
    return reject(helper, "the buffer lacks Indirect usage");
  }
  if (args.offset % 4 != 0) {
    return reject(helper, "the offset is not a multiple of 4");
  }
  const uint32_t slots = capacity(args, record);
  if (count == 0 || first > slots || count > slots - first) {
    return reject(helper, "the slot range does not fit the view");
  }
  return true;
}

uint64_t slot_offset(const BufferView& args, Record record, uint32_t slot) noexcept {
  return args.offset + uint64_t{slot} * record_size(record);
}
} // namespace

BufferView buffer_view(const gpu_heap::BlockView& block) noexcept {
  return {.buffer = block.buffer, .offset = block.offset, .size = block.size};
}

uint32_t capacity(const BufferView& view, Record record) noexcept {
  return static_cast<uint32_t>(std::min<uint64_t>(view.size / record_size(record), UINT32_MAX));
}

bool draw(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t slot) {
  if (!check_args(Helper::Draw, args, Record::Draw, slot, 1)) {
    return false;
  }
  pass.DrawIndirect(args.buffer, slot_offset(args, Record::Draw, slot));
  add(g_encoded, 1);
  return true;
}

bool draw_indexed(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t slot) {
  if (!check_args(Helper::DrawIndexed, args, Record::DrawIndexed, slot, 1)) {
    return false;
  }
  pass.DrawIndexedIndirect(args.buffer, slot_offset(args, Record::DrawIndexed, slot));
  add(g_encoded, 1);
  return true;
}

uint32_t draw_slots(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t first, uint32_t count) {
  if (!check_args(Helper::DrawSlots, args, Record::Draw, first, count)) {
    return 0;
  }
  // No count buffer, so the multi-draw and the per-slot loop issue the same draws
  if (caps().multiDrawIndirect) {
    pass.MultiDrawIndirect(args.buffer, slot_offset(args, Record::Draw, first), count);
    add(g_multiDraws, 1);
    add(g_encoded, 1);
    return 1;
  }
  for (uint32_t slot = first; slot < first + count; ++slot) {
    pass.DrawIndirect(args.buffer, slot_offset(args, Record::Draw, slot));
  }
  add(g_loopDraws, count);
  add(g_encoded, count);
  return count;
}

uint32_t draw_indexed_slots(const wgpu::RenderPassEncoder& pass, const BufferView& args, uint32_t first,
                            uint32_t count) {
  if (!check_args(Helper::DrawIndexedSlots, args, Record::DrawIndexed, first, count)) {
    return 0;
  }
  if (caps().multiDrawIndirect) {
    pass.MultiDrawIndexedIndirect(args.buffer, slot_offset(args, Record::DrawIndexed, first), count);
    add(g_multiDraws, 1);
    add(g_encoded, 1);
    return 1;
  }
  for (uint32_t slot = first; slot < first + count; ++slot) {
    pass.DrawIndexedIndirect(args.buffer, slot_offset(args, Record::DrawIndexed, slot));
  }
  add(g_loopDraws, count);
  add(g_encoded, count);
  return count;
}

bool dispatch(const wgpu::ComputePassEncoder& pass, const BufferView& args, uint32_t slot) {
  if (!check_args(Helper::Dispatch, args, Record::Dispatch, slot, 1)) {
    return false;
  }
  pass.DispatchWorkgroupsIndirect(args.buffer, slot_offset(args, Record::Dispatch, slot));
  add(g_encoded, 1);
  return true;
}

bool bind_indices(const wgpu::RenderPassEncoder& pass, const BufferView& indices) {
  if (!indices.buffer) {
    return reject(Helper::BindIndices, "no buffer");
  }
  if (!(indices.buffer.GetUsage() & wgpu::BufferUsage::Index)) {
    return reject(Helper::BindIndices, "the buffer lacks Index usage");
  }
  if (indices.offset % 4 != 0 || indices.size % 4 != 0 || indices.size == 0) {
    return reject(Helper::BindIndices, "the range is empty or not whole u32 indices");
  }
  pass.SetIndexBuffer(indices.buffer, wgpu::IndexFormat::Uint32, indices.offset, indices.size);
  return true;
}

EncodeStats encode_stats() noexcept {
  return {
      .encoded = g_encoded.load(std::memory_order_relaxed),
      .rejected = g_rejected.load(std::memory_order_relaxed),
      .multiDraws = g_multiDraws.load(std::memory_order_relaxed),
      .loopDraws = g_loopDraws.load(std::memory_order_relaxed),
  };
}
} // namespace aurora::render::indirect

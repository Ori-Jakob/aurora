#pragma once

#include "types.hpp"

#include <aurora/render.hpp>
#include <aurora/render_graph.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <type_traits>

namespace aurora::gfx::detail {

struct FramePacket;

using ComputeBinder = bool (*)(const void* payload, const wgpu::ComputePassEncoder& pass);
struct CopyDestination {
  wgpu::Buffer buffer;
  uint64_t base = 0;
  uint64_t size = 0;
};
using CopyTarget = CopyDestination (*)(const void* payload);
inline constexpr size_t ComputePayloadSize = 64;

struct RecordedFrame {
  FramePacket* packet;
  size_t frameSlot;
};

void begin_recording(FramePacket& packet, size_t frameSlot);
RecordedFrame end_recording();
void shutdown_recording();
void increment_merged_draw_count() noexcept;
uint64_t take_resident_request() noexcept;
bool should_arm_mod_normal(bool requested, bool offscreen, bool normalPresent, bool core, uint32_t msaa) noexcept;

namespace testing {
struct OffscreenCacheStats {
  uint64_t bytes;
  size_t entries;
};
void suppress_render_worker(bool suppress) noexcept;
void seed_offscreen_cache(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                          wgpu::TextureFormat depthFormat);
OffscreenCacheStats offscreen_cache_stats() noexcept;
bool is_offscreen_cached(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                         wgpu::TextureFormat depthFormat);
void override_scene_depth_source(std::optional<render::DepthSource> source) noexcept;
void fake_depth_snapshots(bool fake) noexcept;
uint32_t stage_depth_snapshot_generation(render::Stage slot) noexcept;
uint32_t stage_depth_snapshot_requests(render::Stage slot) noexcept;
}

} // namespace aurora::gfx::detail

namespace aurora::gfx {
struct ColorPassDescriptor {
  const char* label = nullptr;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureView depthStencilView;
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  wgpu::Extent3D targetSize;
  uint32_t sampleCount = 1;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Clear;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;
  float depthClearValue = 0.f;
  bool hasStencil = false;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

void finish();
void begin_color_pass(const ColorPassDescriptor& desc);
void end_color_pass();
void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size);
void begin_offscreen(uint32_t width, uint32_t height);
void end_offscreen();
bool has_normal_attachment() noexcept;
RenderTargetLayout get_render_target_layout() noexcept;

namespace tex_palette_conv {
struct ConvRequest;
}
void queue_palette_conv(tex_palette_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_uniform(const uint8_t* data, size_t length);
template <typename T>
Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_storage(const uint8_t* data, size_t length);
template <typename T>
Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_channel_storage(std::span<const uint8_t> bytes) noexcept;
// 16-byte aligned and never at offset 0, so 0 can mean none; refused instead of grown when mapped staging is full
Range push_storage_packed(std::span<const uint8_t> bytes) noexcept;
Range reserve_channel_storage(size_t length, bool zero) noexcept;
uint64_t last_channel_storage_bytes() noexcept;
inline constexpr uint32_t DispatchMaxInputs = 8;
inline constexpr uint32_t DispatchUserSize = 128;
inline constexpr uint32_t DispatchParamsSize = 256;
// Mirrors AuroraDispatch in aurora/stream.wgsl byte for byte; the two change together
struct DispatchParams {
  std::array<uint32_t, 4> output;
  std::array<uint32_t, 16> inputs;
  std::array<uint32_t, 4> groups;
  std::array<uint32_t, 4> elements;
  std::array<uint32_t, 4> reserved;
  std::array<uint8_t, DispatchUserSize> user;
};
static_assert(sizeof(DispatchParams) == DispatchParamsSize);
static_assert(sizeof(DispatchParams::inputs) == DispatchMaxInputs * 2 * sizeof(uint32_t));
inline constexpr uint32_t DispatchParamsBudget = 1024 * 1024;
struct DispatchArgs {
  Range output;
  std::span<const Range> inputs;
  std::array<uint32_t, 3> groups{};
  std::array<uint32_t, 3> elements{};
  std::span<const uint8_t> user;
};
enum class RecordResult : uint8_t {
  Recorded,
  NoRecording,
  NoOpenPass,
  BadRange,
  TooManyInputs,
  UserTooLarge,
  GroupLimit,
  UniformSpace,
  ParamsBudget,
  SegmentReadsState,
};
namespace detail {
RecordResult record_dispatch(ComputeBinder bind, std::span<const std::byte> payload, const DispatchArgs& args) noexcept;
RecordResult record_copy(CopyTarget target, std::span<const std::byte> payload, Range staged,
                         uint32_t dstOffset) noexcept;
void note_segment_reads_gpu_state() noexcept;
namespace testing {
size_t pending_compute_entries() noexcept;
}

template <typename T>
const T& compute_payload(const void* data) noexcept {
  void* bytes = const_cast<void*>(data);
  return *std::launder(static_cast<const T*>(std::memmove(bytes, bytes, sizeof(T))));
}

template <typename T>
inline constexpr bool ComputePayload =
    std::is_trivially_copyable_v<T> && sizeof(T) <= ComputePayloadSize && alignof(T) <= alignof(std::max_align_t);
} // namespace detail

template <auto Bind, typename T>
RecordResult record_dispatch(const T& payload, const DispatchArgs& args) noexcept {
  static_assert(detail::ComputePayload<T>);
  constexpr detail::ComputeBinder binder = [](const void* data, const wgpu::ComputePassEncoder& pass) {
    return Bind(detail::compute_payload<T>(data), pass);
  };
  return detail::record_dispatch(binder, std::as_bytes(std::span{&payload, 1}), args);
}

template <auto Target, typename T>
RecordResult record_copy(const T& payload, Range staged, uint32_t dstOffset) noexcept {
  static_assert(detail::ComputePayload<T>);
  constexpr detail::CopyTarget target = [](const void* data) { return Target(detail::compute_payload<T>(data)); };
  return detail::record_copy(target, std::as_bytes(std::span{&payload, 1}), staged, dstOffset);
}
struct ResidentStorage {
  uint64_t frameId = 0;
  uint32_t base = 0;
  uint32_t capacity = 0;
  uint32_t limit = 0;
  uint32_t blockAlign = 0;
  uint32_t epoch = 0;
};
ResidentStorage resident_storage() noexcept;
void copy_to_resident(Range staged, uint32_t offset, uint32_t blockSize) noexcept;
void request_resident_capacity(uint64_t bytes) noexcept;
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage);

template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();
template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat);
uint32_t align_uniform(uint32_t value);
// Optional uniform pushes check this first, because a push past the mapped frame buffer aborts
uint64_t uniform_space_left() noexcept;
// Set by end_recording on the thread that ends frames, so the next latch reads it without the render worker
uint32_t last_recorded_uniform_size() noexcept;
Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;
void push_debug_group(std::string label);
void insert_debug_marker(std::string label);
void record_stage(render::Stage stage);
render::DepthSource current_scene_depth_source() noexcept;
} // namespace aurora::gfx

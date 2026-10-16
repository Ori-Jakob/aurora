#pragma once

#include "gpu_heap.hpp"
#include "graph.hpp"
#include "graph_layout.hpp"
#include "pass_context.hpp"

#include "../gfx/types.hpp"

#include <aurora/gfx.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
struct StageData;

struct BindingLayout {
  wgpu::BindGroupLayout passLayout;
  wgpu::PipelineLayout pipelineLayout;
  uint64_t serial = 0;
};

const wgpu::BindGroupLayout& frame_layout();
const wgpu::BindGroup& frame_group();
void prepare_graph_bindings(CompiledGraph& graph, const PassSetup& setup = {});
void shutdown_bindings();

enum class PersistentKind : uint8_t { Buffer, Texture };

struct PersistentBuffer {
  wgpu::Buffer buffer;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint32_t generation = 0;
};

struct PersistentTexture {
  wgpu::Texture texture;
  wgpu::TextureView view;
  uint32_t generation = 0;
};

using PersistentBufferSource = bool (*)(void* context, PersistentBuffer& out);

PersistentHandle create_persistent(std::string_view name, PersistentKind kind);
PersistentHandle create_persistent_buffer_source(std::string_view name, PersistentBufferSource source, void* context);
void set_persistent_buffer(PersistentHandle handle, wgpu::Buffer buffer, uint64_t offset, uint64_t size);
void set_persistent_texture(PersistentHandle handle, wgpu::Texture texture, wgpu::TextureView view);
void destroy_persistent(PersistentHandle handle);
bool resolve_persistent(PersistentHandle handle, PersistentBuffer& out);
bool resolve_persistent(PersistentHandle handle, PersistentTexture& out);
PersistentHandle heap_pool_handle(gpu_heap::PoolId pool);

struct GxBinding {
  wgpu::Texture texture;
  wgpu::TextureView sampleView;
  wgpu::TextureView attachmentView;
  wgpu::TextureView resolveView;
  wgpu::Texture copyTexture;
  wgpu::Extent3D size;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint32_t samples = 1;
};

using GxResolveFn = bool (*)(const StageData& stage, Access access, GxBinding& out);
void register_gx_resolver(GxResource resource, GxResolveFn resolve);

enum class IdentityKind : uint8_t { Transient, History, Persistent, Gx, Dummy };

struct BindIdentity {
  IdentityKind kind = IdentityKind::Transient;
  uint32_t selector = 0;
  uint64_t value = 0;

  bool operator==(const BindIdentity&) const = default;
};

struct BindCacheStats {
  uint32_t entries = 0;
  uint32_t staleEntries = 0;
  uint64_t created = 0;
  uint64_t hits = 0;
  uint64_t evicted = 0;
};

BindCacheStats bind_cache_stats() noexcept;
void bind_cache_tick(uint64_t frameId);
void bind_cache_on_release(uint64_t serial);
uint64_t bind_group_key(uint64_t layoutSerial, std::span<const BindIdentity> identities) noexcept;

using UniformSink = gfx::Range (*)(const uint8_t* data, size_t length);
void build_stage_bindings(StageData& stage, UniformSink sink = nullptr);

bool encode_pass(const wgpu::CommandEncoder& cmd, const StageData& stage, NodeId node);

void external_encode(PassContext& ctx, const void* state);
void queue_after_frame(gfx::AfterSubmitCallback callback);
void run_after_frame_callbacks() noexcept;
void clear_after_frame_callbacks() noexcept;

namespace testing {
struct AttachmentOps {
  bool depth = false;
  wgpu::LoadOp load = wgpu::LoadOp::Undefined;
  wgpu::StoreOp store = wgpu::StoreOp::Undefined;
  wgpu::Color clearColor{};
  float clearDepth = 0.0f;
  bool readOnly = false;
  bool resolves = false;
  uint32_t width = 0;
  uint32_t height = 0;
};

void record_bind_details(bool enabled) noexcept;
std::vector<BindIdentity> last_identities(NodeId node);
std::vector<AttachmentOps> last_attachment_ops();
void reset_bind_cache();
bool frame_objects_created() noexcept;
} // namespace testing
} // namespace aurora::render::detail

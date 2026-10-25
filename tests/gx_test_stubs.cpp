// Stub implementations for renderer symbols that the GX/FIFO/command_processor code
// references but that live in the full renderer (gfx, gx.cpp, model/shader.cpp, etc.).
// These allow the test binary to link without pulling in WebGPU runtime.

#include "gx/gx.hpp"
#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/resources.hpp"
#include "gfx/scene_targets.hpp"
#include "gfx/depth_peek.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/tex_copy_conv.hpp"
#include "gfx/tex_palette_conv.hpp"
#include "gfx/texture.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <vector>

// --- aurora::g_config ---
namespace aurora {
AuroraConfig g_config{};
} // namespace aurora

// --- aurora::log_internal ---
namespace aurora {
std::atomic_uint32_t g_testLogWarnings{0};
std::atomic_uint32_t g_testLogErrors{0};

void log_internal(AuroraLogLevel level, const char* module, const char* message, unsigned int len) noexcept {
  if (level == LOG_WARNING) {
    g_testLogWarnings.fetch_add(1, std::memory_order_relaxed);
  } else if (level == LOG_ERROR) {
    g_testLogErrors.fetch_add(1, std::memory_order_relaxed);
  }
  fprintf(stderr, "[%d] %s: %.*s\n", static_cast<int>(level), module, len, message);
}
} // namespace aurora

// --- fmt::formatter<AuroraLogLevel> ---
auto fmt::formatter<AuroraLogLevel>::format(AuroraLogLevel level, format_context& ctx) const
    -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}", static_cast<int>(level));
}

namespace aurora::gfx {
uint32_t g_testMergedDrawCount = 0;
} // namespace aurora::gfx

// --- GPU resources (default-constructed, not used in tests) ---
namespace aurora::gfx::detail {
Resources& resources() noexcept {
  static Resources resources;
  return resources;
}

void increment_merged_draw_count() noexcept { ++g_testMergedDrawCount; }
} // namespace aurora::gfx::detail

namespace aurora::webgpu {
GraphicsConfig g_graphicsConfig{};
bool g_dualSourceBlendingSupported = false;
} // namespace aurora::webgpu

// --- GXState ---
namespace aurora::gx {
GXState g_gxState{};
void set_viewport_policy(AuroraViewportPolicy policy) noexcept {}
} // namespace aurora::gx

namespace aurora::vi {
Vec2<uint32_t> configured_fb_size() noexcept { return {640, 480}; }
void configure(const GXRenderModeObj*) noexcept {}
} // namespace aurora::vi

// --- get_texture ---
namespace aurora::gx {
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
namespace texture {
void invalidate_bindings() noexcept {}
uint64_t current_bind_generation() noexcept { return 1; }
GXState::CopyTextureRef resolve_copy_texture(const GXState::CopyTextureKey& key) noexcept {
  auto& ref = g_gxState.copyTextures[key.dest];
  ++ref.revision;
  return ref;
}
} // namespace texture
void evict_texture_object(u32 texObjId) noexcept {
  for (auto& obj : g_gxState.loadedTextures) {
    if (obj.texObjId == texObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_tlut_object(u32 tlutObjId) noexcept {
  for (auto& obj : g_gxState.loadedTluts) {
    if (obj.tlutObjId == tlutObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_copy_texture(const void* dest) noexcept {
  g_gxState.copyTextures.erase(dest);
}
void shutdown() noexcept {}
Vec2<uint32_t> logical_fb_size() noexcept { return {640, 480}; }
gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept { return logicalViewport; }
gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept { return logicalScissor; }
void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}
void set_render_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.renderViewport = viewport;
  gfx::set_viewport(viewport);
}
void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(scissor));
}
void set_render_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.renderScissor = scissor;
  gfx::set_scissor(scissor);
}
} // namespace aurora::gx

// --- Shader/pipeline stubs ---
namespace aurora::gx {
uint32_t g_testPipelineBuildCount = 0;
uint32_t g_testShaderInfoCount = 0;
uint32_t g_testUniformBuildCount = 0;
// Each shader info build returns its own uniform size, so this shows which build a uniform was made from
u32 g_testLastUniformInfoSize = 0;
uint32_t g_testResolveCount = 0;
// The shader info the last resolve saw, so a test can tie it to the same draw's build
u32 g_testLastResolveInfoSize = 0;
// Written into every populated config, so a test can make a rebuild produce a different config
GXCompare g_testPopulateDepthFunc = GX_NEVER;
u32 g_testPopulateTevStageCount = 0;
std::bitset<MaxTextures> g_testStochasticEligible;
std::bitset<MaxTextures> g_testSampledTextures;
bool g_testLastBindBaseLevel = false;
void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt) noexcept {
  ++g_testPipelineBuildCount;
  config.depthFunc = g_testPopulateDepthFunc;
  config.shaderConfig.tevStageCount = g_testPopulateTevStageCount;
}
GXBindGroups build_bind_groups(const ShaderInfo& info, bool baseLevel) noexcept {
  g_testLastBindBaseLevel = baseLevel;
  return {};
}
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept {
  ShaderInfo info{.uniformSize = ++g_testShaderInfoCount};
  info.sampledTextures |= g_testSampledTextures;
  info.stochasticEligible = g_testStochasticEligible;
  info.stochasticTextures = std::bitset<MaxTextures>{config.ext.stochasticMask} & g_testStochasticEligible;
  return info;
}
gfx::Range build_draw_uniform(const ShaderInfo& info) noexcept {
  ++g_testUniformBuildCount;
  g_testLastUniformInfoSize = info.uniformSize;
  return {.size = 1};
}
void clear_draw_uniforms() noexcept {}
void resolve_sampled_textures(const ShaderInfo& info) noexcept {
  ++g_testResolveCount;
  g_testLastResolveInfoSize = info.uniformSize;
}
} // namespace aurora::gx

// --- Buffer push stubs ---
namespace aurora::gfx {
namespace {
// Offsets advance like the real frame buffers, so the merge contiguity CHECKs compare real ranges
size_t s_vertCursor = 0;
size_t s_idxCursor = 0;
// An empty Range has offset 0 too, so real uploads start past it
constexpr size_t StorageBase = 256;
size_t s_storageCursor = StorageBase;

Range push_range(size_t& cursor, size_t length, size_t alignment) {
  const size_t offset = alignment != 0 ? AURORA_ALIGN(cursor, alignment) : cursor;
  cursor = offset + length;
  return {static_cast<uint32_t>(offset), static_cast<uint32_t>(length)};
}

// Like the recorder, only a GX draw that is still the newest command of its pass can take a merge
bool s_lastCommandIsGxDraw = false;
Viewport s_viewport{};
ClipRect s_scissor{};
} // namespace

uint64_t g_testPushedVertexBytes = 0;
uint64_t g_testPushedIndexBytes = 0;
uint64_t g_testPushedStorageBytes = 0;
uint32_t g_testStoragePushes = 0;
// Off by default, so older tests still read and upload every indexed array on every draw
bool g_testStorageRanges = false;
// Mirrors what storage pushes copied, so a test can check the bytes behind a draw's arrayStart
std::vector<uint8_t> g_testStorage;
// Runs once a push has copied, so a test can rewrite guest memory mid-upload as the game thread can
void (*g_testAfterStoragePush)() = nullptr;
// Runs before a push copies, so a test can rewrite guest memory between the shadow fill and the upload
void (*g_testBeforeStoragePush)() = nullptr;
// Without guest hooks a read aliases guest memory, so only the source pointer shows what a push copied from
const uint8_t* g_testLastStorageSource = nullptr;
ResidentStorage g_testResidentStorage{};
std::vector<detail::ResidentCopy> g_testResidentCopies;
uint64_t g_testResidentRequest = 0;
bool g_testChannelStorageFull = false;
uint32_t g_testChannelPushes = 0;

namespace testing {
void reset_resident_stubs() noexcept {
  g_testResidentStorage = {};
  g_testResidentCopies.clear();
  g_testResidentRequest = 0;
}

void reset_stub_ranges() noexcept {
  s_vertCursor = 0;
  s_idxCursor = 0;
  s_storageCursor = StorageBase;
  g_testStorage.clear();
  reset_resident_stubs();
}
} // namespace testing

Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  g_testPushedVertexBytes += length;
  return push_range(s_vertCursor, length, alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment) {
  g_testPushedIndexBytes += length;
  return push_range(s_idxCursor, length, alignment);
}
Range push_uniform(const uint8_t* data, size_t length) { return {}; }
Range push_storage(const uint8_t* data, size_t length) {
  g_testPushedStorageBytes += length;
  ++g_testStoragePushes;
  g_testLastStorageSource = data;
  if (!g_testStorageRanges) {
    return {};
  }
  if (g_testBeforeStoragePush != nullptr) {
    g_testBeforeStoragePush();
  }
  const auto range = push_range(s_storageCursor, length, 256);
  g_testStorage.resize(s_storageCursor);
  if (length != 0) {
    std::memcpy(g_testStorage.data() + range.offset, data, length);
  }
  if (g_testAfterStoragePush != nullptr) {
    g_testAfterStoragePush();
  }
  return range;
}
Range push_channel_storage(std::span<const uint8_t> bytes) noexcept {
  ++g_testChannelPushes;
  if (g_testChannelStorageFull) {
    return {};
  }
  const auto range = push_range(s_storageCursor, bytes.size(), 256);
  s_storageCursor = range.offset + AURORA_ALIGN(bytes.size(), size_t{4});
  g_testStorage.resize(s_storageCursor);
  std::memcpy(g_testStorage.data() + range.offset, bytes.data(), bytes.size());
  return range;
}
Range reserve_channel_storage(size_t length, bool zero) noexcept {
  ++g_testChannelPushes;
  if (g_testChannelStorageFull) {
    return {};
  }
  const auto range = push_range(s_storageCursor, AURORA_ALIGN(length, size_t{4}), 256);
  g_testStorage.resize(s_storageCursor);
  std::memset(g_testStorage.data() + range.offset, zero ? 0 : 0xCD, range.size);
  return range;
}
ResidentStorage resident_storage() noexcept { return g_testResidentStorage; }
void copy_to_resident(Range staged, uint32_t offset, uint32_t blockSize) noexcept {
  g_testResidentCopies.push_back({staged.offset, offset, AURORA_ALIGN(staged.size, 4u), blockSize});
}
void request_resident_capacity(uint64_t bytes) noexcept {
  g_testResidentRequest = std::max(g_testResidentRequest, bytes);
}

Vec2<uint32_t> get_render_target_size() noexcept { return {640, 480}; }
// The recorder appends a command only when the value changes
void set_viewport(const Viewport& viewport) noexcept {
  if (viewport != s_viewport) {
    s_viewport = viewport;
    s_lastCommandIsGxDraw = false;
  }
}
void set_scissor(const ClipRect& scissor) noexcept {
  if (scissor != s_scissor) {
    s_scissor = scissor;
    s_lastCommandIsGxDraw = false;
  }
}
void set_scissor(uint32_t x, uint32_t y, uint32_t w, uint32_t h) noexcept {}
bool g_testNormalAttachment = false;
bool has_normal_attachment() noexcept { return g_testNormalAttachment; }
RenderTargetLayout get_render_target_layout() noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{ColorAttachmentSemantic::SceneColor, wgpu::TextureFormat::RGBA8Unorm}}},
      .depthStencilFormat = wgpu::TextureFormat::Depth24Plus,
      .sampleCount = 1,
  };
  if (g_testNormalAttachment) {
    layout.colorAttachmentCount = 2;
    layout.colorAttachments[1] = {ColorAttachmentSemantic::Normal, wgpu::TextureFormat::RGB10A2Unorm};
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}
} // namespace aurora::gfx

namespace aurora::gfx::detail {
ChannelBudget g_testChannelBudget{.bytes = UINT64_MAX / 4, .alignment = 256};
uint64_t g_testChannelDemand = 0;
ChannelBudget channel_budget() noexcept { return g_testChannelBudget; }
void request_channel_storage(uint64_t bytesPerFrame) noexcept {
  g_testChannelDemand = std::max(g_testChannelDemand, bytesPerFrame);
}
ComputeFailures g_testComputeFailures;
ComputeFailures compute_failures() noexcept { return g_testComputeFailures; }
} // namespace aurora::gfx::detail

// --- Pipeline/draw command stubs ---
namespace aurora::gfx {
template <>
PipelineRef pipeline_ref<clear::PipelineConfig>(const clear::PipelineConfig& config) {
  return 0;
}
template <>
void push_draw_command<clear::DrawData>(clear::DrawData data) {
  s_lastCommandIsGxDraw = false;
}
uint32_t g_testPipelineRefCount = 0;
// Every lookup returns a new nonzero ref, so a draw's ref shows which lookup it kept
PipelineRef g_testLastPipelineRef = 0;
// Resolves seen by the last lookup, so a test can tell whether it came after the draw's resolve
uint32_t g_testResolvesAtLookup = 0;
gx::RenderExtConfig g_testLastLookupExt{};
template <>
PipelineRef pipeline_ref<gx::PipelineConfig>(const gx::PipelineConfig& config) {
  g_testResolvesAtLookup = gx::g_testResolveCount;
  g_testLastLookupExt = config.shaderConfig.ext;
  g_testLastPipelineRef = ++g_testPipelineRefCount;
  return g_testLastPipelineRef;
}
PipelineState g_testPipelineState = PipelineState::Ready;
uint64_t g_testPipelineEpoch = 0;
uint32_t g_testPipelineStateQueries = 0;
PipelineRef g_testReadyLegacyRef = 0;
uint32_t g_testReadyProbes = 0;
gx::RenderExtConfig g_testLastReadyExt{};
PipelineState pipeline_state(PipelineRef ref) noexcept {
  ++g_testPipelineStateQueries;
  return g_testPipelineState;
}
uint64_t pipeline_epoch() noexcept { return g_testPipelineEpoch; }
PipelineRef ready_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept {
  ++g_testReadyProbes;
  g_testLastReadyExt = config.shaderConfig.ext;
  return g_testReadyLegacyRef;
}
bool g_testBlockingPipelines = false;
void set_blocking_pipelines(bool blocking) noexcept { g_testBlockingPipelines = blocking; }
bool blocking_pipelines() noexcept { return g_testBlockingPipelines; }
const LayoutWarm* g_testLayoutWarm = nullptr;
std::vector<gx::PipelineConfig> g_testWarmedConfigs;
const LayoutWarm* layout_warm() noexcept { return g_testLayoutWarm; }
void warm_scene_variant(const gx::PipelineConfig& config) { g_testWarmedConfigs.push_back(config); }
gx::DrawData g_testLastDraw{};
uint32_t g_testDrawCount = 0;
std::atomic<uint32_t> g_testProcessedDrawCount{0};
// Off by default so tests written before merging was emulated still see one command per draw
bool g_testMergeDraws = false;

template <>
void push_draw_command<gx::DrawData>(gx::DrawData data) {
  g_testLastDraw = data;
  ++g_testDrawCount;
  s_lastCommandIsGxDraw = true;
  g_testProcessedDrawCount.fetch_add(1, std::memory_order_release);
}
template <>
gx::DrawData* get_last_draw_command() {
  return g_testMergeDraws && g_testDrawCount != 0 && s_lastCommandIsGxDraw ? &g_testLastDraw : nullptr;
}
} // namespace aurora::gfx

// --- TextureBind::get_descriptor ---
namespace aurora::gfx {
wgpu::SamplerDescriptor TextureBind::get_descriptor(bool) const noexcept { return wgpu::SamplerDescriptor{}; }
} // namespace aurora::gfx

// --- Texture creation/write/replacement stubs ---
namespace aurora::gfx {
TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                    ArrayRef<uint8_t> data, bool tlut, const char* label) noexcept {
  return {};
}
TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  return {};
}
TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  return {};
}
TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept { return {}; }
void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {}
void queue_texture_upload(TextureUpload upload) {}
void queue_texture_upload_data(const uint8_t* data, size_t length, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) {}
void queue_palette_conv(tex_palette_conv::ConvRequest req) {}
namespace testing {
std::atomic<uint32_t> beginOffscreenCount{0};
std::atomic<uint32_t> endOffscreenCount{0};
std::atomic<uint32_t> resolvePassCount{0};
std::atomic<uint32_t> offscreenWidth{0};
std::atomic<uint32_t> offscreenHeight{0};
} // namespace testing

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat) {
  // Each of these starts a new pass, which never holds a merge target
  s_lastCommandIsGxDraw = false;
  testing::resolvePassCount.fetch_add(1, std::memory_order_release);
}
void begin_offscreen(uint32_t width, uint32_t height) {
  s_lastCommandIsGxDraw = false;
  testing::offscreenWidth.store(width, std::memory_order_relaxed);
  testing::offscreenHeight.store(height, std::memory_order_relaxed);
  testing::beginOffscreenCount.fetch_add(1, std::memory_order_release);
}
void end_offscreen() {
  s_lastCommandIsGxDraw = false;
  testing::endOffscreenCount.fetch_add(1, std::memory_order_release);
}
bool g_testOffscreen = false;
bool is_offscreen() noexcept { return g_testOffscreen; }
} // namespace aurora::gfx

namespace aurora::gfx::depth_peek {
namespace {
bool s_snapshotRequested = false;
uint32_t s_width = 0;
uint32_t s_height = 0;
std::vector<uint32_t> s_data;
} // namespace

void initialize() {}
void shutdown() {}
void request_snapshot() noexcept { s_snapshotRequested = true; }
void poll() noexcept {}
void encode_frame_snapshot(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                           wgpu::Extent3D sourceSize, uint32_t msaaSamples) noexcept {}
void after_submit() noexcept {}

bool read_latest(uint16_t x, uint16_t y, uint32_t& z) noexcept {
  if (x >= s_width || y >= s_height || s_data.empty()) {
    return false;
  }
  z = s_data[static_cast<size_t>(y) * s_width + x] & 0x00ffffffu;
  return true;
}

namespace testing {
void reset() noexcept {
  s_snapshotRequested = false;
  s_width = 0;
  s_height = 0;
  s_data.clear();
}

bool snapshot_requested() noexcept { return s_snapshotRequested; }

void set_latest(uint32_t width, uint32_t height, const std::vector<uint32_t>& data) {
  s_width = width;
  s_height = height;
  s_data = data;
}
} // namespace testing
} // namespace aurora::gfx::depth_peek

namespace aurora::gfx::tex_copy_conv {
bool needs_conversion(GXTexFmt fmt) { return false; }
} // namespace aurora::gfx::tex_copy_conv

namespace aurora::gfx::tex_palette_conv {
void queue(ConvRequest req) {}
} // namespace aurora::gfx::tex_palette_conv

namespace aurora::gfx::texture_replacement {
u32 compute_texture_upload_size(const GXTexObj_& obj) noexcept { return 0; }
bool has_replacement(const GXTexObj_&) noexcept { return false; }
bool has_replacement(const GXTexObj_&, const GXTlutObj_&) noexcept { return false; }
} // namespace aurora::gfx::texture_replacement

// --- Window stub ---
#include "../lib/window.hpp"
namespace aurora::window {
AuroraWindowSize get_window_size() { return {640, 480, 640, 480, 640, 480, 1.0f}; }
void set_frame_buffer_aspect_fit(bool) {}
} // namespace aurora::window

// --- WebGPU C API stubs (prevent linker errors from wgpu:: destructors) ---
extern "C" {
void wgpuDeviceRelease(WGPUDevice) {}
void wgpuQueueRelease(WGPUQueue) {}
void wgpuSurfaceRelease(WGPUSurface) {}
void wgpuBufferRelease(WGPUBuffer) {}
void wgpuTextureRelease(WGPUTexture) {}
void wgpuTextureViewRelease(WGPUTextureView) {}
void wgpuSamplerRelease(WGPUSampler) {}
void wgpuShaderModuleRelease(WGPUShaderModule) {}
void wgpuRenderPipelineRelease(WGPURenderPipeline) {}
void wgpuBindGroupRelease(WGPUBindGroup) {}
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout) {}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout) {}
void wgpuInstanceRelease(WGPUInstance) {}
void wgpuDeviceAddRef(WGPUDevice) {}
void wgpuQueueAddRef(WGPUQueue) {}
void wgpuSurfaceAddRef(WGPUSurface) {}
void wgpuBufferAddRef(WGPUBuffer) {}
void wgpuTextureAddRef(WGPUTexture) {}
void wgpuTextureViewAddRef(WGPUTextureView) {}
void wgpuInstanceAddRef(WGPUInstance) {}
}

void aurora::gfx::push_debug_group(std::string) {}
void push_debug_group(const char*) {}
void pop_debug_group() {}
void aurora::gfx::insert_debug_marker(std::string) {}

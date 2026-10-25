#include "recording.hpp"

#include "encoding.hpp"
#include "frame.hpp"
#include "lru_cache.hpp"
#include "resource_cache.hpp"
#include "scene_targets.hpp"

#include "clear.hpp"
#include "pipeline_cache.hpp"
#include "render_worker.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture.hpp"
#include "../gx/fifo.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#include "../render/scene_depth.hpp"
#include "../render/stage.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../trace/capture.hpp"
#include "../window.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;

namespace {
constexpr Module Log{"aurora::gfx"};

struct FrameRecorder {
  FramePacket* packet = nullptr;
  size_t frameSlot = 0;
  uint32_t currentRenderPass = UINT32_MAX;
  uint32_t drawCallCount = 0;
  uint32_t mergedDrawCallCount = 0;
  bool inOffscreen = false;
  std::optional<RenderPass> suspendedEfbPass;
  Viewport suspendedEfbViewport;
  ClipRect suspendedEfbScissor;
  webgpu::TextureWithSampler offscreenColor;
  webgpu::TextureWithSampler offscreenDepth;
  Viewport cachedViewport;
  ClipRect cachedScissor;
  bool suppressRenderWorker = false;
  bool normalRequested = false;
  uint8_t nextStage = 0;
  uint8_t pendingStageEnd = 0;
  std::array<bool, render::StageCount> stageMarkerSeen{};
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif

  [[nodiscard]] bool active() const noexcept { return packet != nullptr; }

  FramePacket& frame() const {
    CHECK(packet != nullptr, "No active recording session");
    return *packet;
  }

  RenderPassList& passes() const { return frame().renderPasses; }
};

FrameRecorder g_recorder;
uint32_t g_lastRecordedUniformSize = 0;
std::atomic_uint64_t g_residentRequest{0};
uint64_t g_channelStorageBytes = 0;
uint64_t g_lastChannelStorageBytes = 0;

std::string pass_label(std::string_view kind) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (!g_recorder.debugGroupStack.empty()) {
    return fmt::format("{} ({})", kind, g_recorder.debugGroupStack.back());
  }
#endif
  return std::string{kind};
}

void set_efb_targets(RenderPass& pass) {
  const auto layout = scene_render_target_layout();
  pass.colorAttachmentCount = layout.colorAttachmentCount;
  auto& sceneColor = pass.colorAttachments[SceneColorAttachmentIndex];
  sceneColor.semantic = ColorAttachmentSemantic::SceneColor;
  sceneColor.format = layout.colorAttachments[SceneColorAttachmentIndex].format;
  sceneColor.size = webgpu::g_frameBuffer.size;
  sceneColor.view = webgpu::g_frameBuffer.view;
  sceneColor.resolveView = layout.sampleCount > 1 ? webgpu::g_frameBufferResolved.view : nullptr;
  for (uint32_t i = SceneColorAttachmentIndex + 1; i < layout.colorAttachmentCount; ++i) {
    auto& color = pass.colorAttachments[i];
    color.semantic = layout.colorAttachments[i].semantic;
    color.format = layout.colorAttachments[i].format;
    const webgpu::TextureWithSampler* target = nullptr;
    switch (color.semantic) {
      DEFAULT_FATAL("Scene render-target attachment {} has no backing texture", i);
    case ColorAttachmentSemantic::Normal:
      target = &webgpu::g_normalBuffer;
      break;
    case ColorAttachmentSemantic::Motion:
      target = &webgpu::g_motionBuffer;
      break;
    case ColorAttachmentSemantic::Reactive:
      target = &webgpu::g_reactiveBuffer;
      break;
    case ColorAttachmentSemantic::Fog:
      target = &webgpu::g_fogRecordBuffer;
      break;
    case ColorAttachmentSemantic::DebugId:
      target = &webgpu::g_debugIdBuffer;
      break;
    }
    color.size = target->size;
    color.view = target->view;
  }
  pass.depthStencilView = webgpu::g_depthBuffer.view;
  pass.depthStencilFormat = layout.depthStencilFormat;
  pass.copySourceTexture =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.texture : webgpu::g_frameBuffer.texture;
  pass.copySourceView =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : webgpu::g_frameBuffer.view;
  pass.copySourceDepthView = webgpu::g_depthBuffer.view;
  if (webgpu::normal_present()) {
    pass.copySourceNormalTexture = webgpu::g_normalBuffer.texture;
  }
  pass.msaaSamples = layout.sampleCount;
  pass.hasDepth = true;
  pass.hasStencil = false;
}

void set_single_color_target(RenderPass& pass, wgpu::TextureFormat format, const wgpu::Extent3D size,
                             wgpu::TextureView view, wgpu::TextureView resolveView = {}) {
  auto& color = pass.colorAttachments[SceneColorAttachmentIndex];
  color.semantic = ColorAttachmentSemantic::SceneColor;
  color.format = format;
  color.size = size;
  color.view = std::move(view);
  color.resolveView = std::move(resolveView);
  pass.colorAttachmentCount = 1;
}

struct OffscreenCacheKey {
  uint32_t width;
  uint32_t height;
  wgpu::TextureFormat colorFormat;
  wgpu::TextureFormat depthFormat;

  bool operator==(const OffscreenCacheKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const OffscreenCacheKey& key) {
    return H::combine(std::move(h), key.width, key.height, key.colorFormat, key.depthFormat);
  }
};
struct OffscreenCacheEntry {
  webgpu::TextureWithSampler color;
  webgpu::TextureWithSampler depth;

  uint64_t size_bytes() const {
    return calc_texture_size(color.format, color.size.width, color.size.height, 1) +
           calc_texture_size(depth.format, depth.size.width, depth.size.height, 1);
  }
};
LruCache<OffscreenCacheKey, OffscreenCacheEntry> g_offscreenCache{RenderTextureCacheLimits};

// Pooled destinations for the public resolve_pass API. Entries are recycled
// per frame slot: a slot is only re-acquired after the render worker has
// submitted its previous frame, and queue serialization orders the new frame's
// copies after the old frame's reads.
struct PassSnapshotEntry {
  webgpu::TextureWithSampler color;
  webgpu::TextureWithSampler depth; // R32Float raw depth
  webgpu::TextureWithSampler normal;
};
struct PassSnapshotPool {
  std::vector<PassSnapshotEntry> entries;
  size_t used = 0;
};
std::array<PassSnapshotPool, FrameSlotCount> g_passSnapshotPools;

PassSnapshotEntry& acquire_pass_snapshot(uint32_t width, uint32_t height, bool wantColor, bool wantDepth,
                                         bool wantNormal) {
  auto& pool = g_passSnapshotPools[g_recorder.frameSlot];
  if (pool.used == pool.entries.size()) {
    pool.entries.emplace_back();
  }
  auto& entry = pool.entries[pool.used++];
  const wgpu::Extent3D size{width, height, 1};
  if (wantColor && (!entry.color.texture || entry.color.size.width != width || entry.color.size.height != height ||
                    entry.color.format != webgpu::g_graphicsConfig.surfaceConfiguration.format)) {
    const auto format = webgpu::g_graphicsConfig.surfaceConfiguration.format;
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Color",
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.color = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = format,
    };
  }
  if (wantDepth && (!entry.depth.texture || entry.depth.size.width != width || entry.depth.size.height != height)) {
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Depth",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = wgpu::TextureFormat::R32Float,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.depth = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = wgpu::TextureFormat::R32Float,
    };
  }
  if (wantNormal && (!entry.normal.texture || entry.normal.size.width != width || entry.normal.size.height != height)) {
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Normal",
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = webgpu::NormalBufferFormat,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.normal = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = webgpu::NormalBufferFormat,
    };
  }
  return entry;
}

FramePacket& current_frame_packet() { return g_recorder.frame(); }

RenderPassList& current_render_passes() { return g_recorder.passes(); }

StagingHighWater current_high_water(const FramePacket& frame) noexcept {
  return {
      .verts = static_cast<uint32_t>(frame.verts.size()),
      .uniforms = static_cast<uint32_t>(frame.uniforms.size()),
      .indices = static_cast<uint32_t>(frame.indices.size()),
      .storage = static_cast<uint32_t>(frame.storage.size()),
      .textureUpload = static_cast<uint32_t>(frame.textureUpload.size()),
      .textureUploadCount = frame.textureUploads.size(),
  };
}

FrameOp capture_frame_op(FramePacket& frame, FrameOpType type, uint32_t index) {
  AURORA_ASSERT(frame.pendingCompute == UINT32_MAX, "Frame op {} captured with a compute batch still pending",
                magic_enum::enum_name(type));
  FrameOp op{
      .type = type,
      .index = index,
      .renderPass =
          type == FrameOpType::RenderPass && index < frame.renderPasses.size() ? &frame.renderPasses[index] : nullptr,
      .textureCopy = type == FrameOpType::TextureCopy && index < frame.textureCopies.size()
                         ? &frame.textureCopies[index]
                         : nullptr,
      .encoderTask =
          type == FrameOpType::EncoderTask && index < frame.encoderTasks.size() ? &frame.encoderTasks[index] : nullptr,
      .stageData = type == FrameOpType::RenderStage && index < frame.stages.size() ? &frame.stages[index] : nullptr,
      .computeBatch =
          type == FrameOpType::Compute && index < frame.computeBatches.size() ? &frame.computeBatches[index] : nullptr,
      .highWater = current_high_water(frame),
  };
  op.textureUploads.reserve(op.highWater.textureUploadCount);
  for (size_t i = 0; i < op.highWater.textureUploadCount; ++i) {
    op.textureUploads.push_back(&frame.textureUploads[i]);
  }
  return op;
}

void seal_pass(FramePacket& frame, uint32_t passIndex) {
  if (passIndex >= frame.renderPasses.size()) {
    return;
  }
  auto& pass = frame.renderPasses[passIndex];
  if (pass.sealed) {
    return;
  }
  pass.sealed = true;
}

Range push(ByteBuffer& target, const uint8_t* data, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  const auto begin = target.size();
  if (length > 0) {
    target.append(data, length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

Range map(ByteBuffer& target, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  auto begin = target.size();
  if (length > 0) {
    target.append_zeroes(length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

// For our public API, warn instead of fatal-ing when called outside an active recording frame.
bool check_recording(const char* name) {
  if (!g_recorder.active())
    UNLIKELY {
      Log.warn("{}: called outside an active frame", name);
      return false;
    }
  return true;
}

void push_command(CommandType type, const Command::Data& data) {
  if (g_recorder.currentRenderPass == UINT32_MAX)
    UNLIKELY {
      Log.warn("Dropping command {}", magic_enum::enum_name(type));
      return;
    }
  auto& renderPass = current_render_passes()[g_recorder.currentRenderPass];
  AURORA_ASSERT(!renderPass.sealed, "Attempted to append command {} to sealed render pass {}",
                magic_enum::enum_name(type), g_recorder.currentRenderPass);
  if (type == CommandType::Draw || type == CommandType::CustomDraw) {
    renderPass.hasDraws = true;
  }
  renderPass.commands.push_back({
      .type = type,
#ifdef AURORA_GFX_DEBUG_GROUPS
      .debugGroupStack = g_recorder.debugGroupStack,
#endif
      .data = data,
  });
}

template <class T>
concept InlinePayload =
    std::is_trivially_copyable_v<T> && std::is_aggregate_v<T> && std::is_trivially_destructible_v<T>;

template <InlinePayload T>
T& inline_payload(void* ptr) noexcept {
  // C++20 equivalent of C++23's std::start_lifetime_as
  return *std::launder(static_cast<T*>(std::memmove(ptr, ptr, sizeof(T))));
}

template <auto Renderer, InlinePayload T>
void encode_draw(void* payload, const wgpu::RenderPassEncoder& pass, const RenderPass& passInfo) {
  const T& data = inline_payload<T>(payload);
  if constexpr (std::is_invocable_v<decltype(Renderer), const T&, const wgpu::RenderPassEncoder&,
                                    const wgpu::Extent3D&>) {
    Renderer(data, pass, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
  } else {
    static_assert(std::is_invocable_v<decltype(Renderer), const T&, const wgpu::RenderPassEncoder&>);
    Renderer(data, pass);
  }
}

template <auto Renderer, InlinePayload T>
DrawCommand make_draw_command(const T& data) {
  static_assert(sizeof(T) <= InlineDrawPayloadSize);
  static_assert(alignof(T) <= alignof(std::max_align_t));
  DrawCommand command{.encoder = encode_draw<Renderer, T>};
  std::memcpy(command.payload.data(), &data, sizeof(data));
  return command;
}

void push_draw_command(DrawCommand data) {
  push_command(CommandType::Draw, Command::Data{.draw = data});
  ++g_recorder.drawCallCount;
}

OffscreenCacheEntry get_offscreen_textures(uint32_t width, uint32_t height) {
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  const OffscreenCacheKey key{width, height, colorFormat, depthFormat};
  if (const auto* entry = g_offscreenCache.find(key)) {
    return *entry;
  }
  const wgpu::Extent3D size{width, height, 1};
  const wgpu::TextureDescriptor colorDesc{
      .label = "Offscreen Color",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
               wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto colorTexture = webgpu::g_device.CreateTexture(&colorDesc);
  auto colorView = colorTexture.CreateView();
  webgpu::TextureWithSampler color{
      .texture = std::move(colorTexture),
      .view = std::move(colorView),
      .size = size,
      .format = colorFormat,
  };
  const wgpu::TextureDescriptor depthDesc{
      .label = "Offscreen Depth",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto depthTexture = webgpu::g_device.CreateTexture(&depthDesc);
  auto depthView = depthTexture.CreateView();
  webgpu::TextureWithSampler depth{
      .texture = std::move(depthTexture),
      .view = std::move(depthView),
      .size = size,
      .format = depthFormat,
  };
  OffscreenCacheEntry entry{
      .color = std::move(color),
      .depth = std::move(depth),
  };
  g_offscreenCache.insert(key, entry, entry.size_bytes());
  return entry;
}

void enqueue_pass(FramePacket& frame, uint32_t passIndex);

constexpr uint8_t LastMarkerStage = static_cast<uint8_t>(render::Stage::FrameEnd);
constexpr uint8_t NoLiveMarker = UINT8_MAX;

void run_stages_until(uint8_t end, uint8_t liveMarker);

void resume_efb_pass_loading(const RenderPass& prevPass) {
  RenderPass newPass{
      .label = pass_label("EFB"),
      .colorAttachments = prevPass.colorAttachments,
      .colorAttachmentCount = prevPass.colorAttachmentCount,
      .depthStencilView = prevPass.depthStencilView,
      .depthStencilFormat = prevPass.depthStencilFormat,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .copySourceNormalTexture = prevPass.copySourceNormalTexture,
      .msaaSamples = prevPass.msaaSamples,
      .clearDepth = false,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  for (uint32_t i = 0; i < newPass.colorAttachmentCount; ++i) {
    newPass.colorAttachments[i].loadOp = wgpu::LoadOp::Undefined;
    newPass.colorAttachments[i].clear = false;
  }
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void suspend_efb() {
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "suspend_efb called outside of an active recording frame");
  AURORA_ASSERT(!g_recorder.inOffscreen, "suspend_efb called while offscreen rendering is active");
  AURORA_ASSERT(!g_recorder.suspendedEfbPass, "suspend_efb called with an EFB pass already suspended");

  auto& currentPass = current_render_passes()[g_recorder.currentRenderPass];
  if (!currentPass.has_consumer()) {
    // Offscreen passes encode before this one, so they would sample its palette targets before conversion
    if (!currentPass.paletteConvs.empty()) {
      static bool warned = false;
      if (!std::exchange(warned, true)) {
        Log.warn("Suspended an EFB pass with pending palette conversions");
      }
    }
    g_recorder.suspendedEfbPass = std::move(currentPass);
    current_render_passes().pop_back();
    --g_recorder.currentRenderPass;
  } else {
    enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  }
  g_recorder.suspendedEfbViewport = g_recorder.cachedViewport;
  g_recorder.suspendedEfbScissor = g_recorder.cachedScissor;
}

void finish_current_offscreen() {
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "finish_current_offscreen called outside of an active recording frame");
  AURORA_ASSERT(g_recorder.inOffscreen, "finish_current_offscreen called without an active offscreen pass");

  auto& offscreenPass = current_render_passes()[g_recorder.currentRenderPass];
  offscreenPass.discardable = !offscreenPass.has_consumer();
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  g_recorder.offscreenColor = {};
  g_recorder.offscreenDepth = {};
}

void start_offscreen(uint32_t width, uint32_t height) {
  AURORA_ASSERT(width != 0 && height != 0, "start_offscreen requires nonzero dimensions ({}x{})", width, height);
  AURORA_ASSERT(g_recorder.active(), "start_offscreen called outside of an active recording frame");

  auto offscreenEntry = get_offscreen_textures(width, height);
  g_recorder.offscreenColor = std::move(offscreenEntry.color);
  g_recorder.offscreenDepth = std::move(offscreenEntry.depth);

  RenderPass newPass{
      .label = pass_label("Offscreen"),
      .depthStencilView = g_recorder.offscreenDepth.view,
      .depthStencilFormat = g_recorder.offscreenDepth.format,
      .copySourceTexture = g_recorder.offscreenColor.texture,
      .copySourceView = g_recorder.offscreenColor.view,
      .copySourceDepthView = g_recorder.offscreenDepth.view,
      .msaaSamples = 1,
      .clearDepthValue = gx::UseReversedZ ? 0.f : 1.f,
      .clearDepth = true,
      .hasDepth = true,
      .hasStencil = false,
  };
  set_single_color_target(newPass, g_recorder.offscreenColor.format, {width, height}, g_recorder.offscreenColor.view);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;
  g_recorder.inOffscreen = true;

  g_recorder.cachedViewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
  g_recorder.cachedScissor = {0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void restore_efb() {
  AURORA_ASSERT(g_recorder.active(), "restore_efb called outside of an active recording frame");
  AURORA_ASSERT(g_recorder.inOffscreen, "restore_efb called without a suspended EFB pass");

  g_recorder.inOffscreen = false;
  if (g_recorder.suspendedEfbPass) {
    current_render_passes().emplace_back(std::move(*g_recorder.suspendedEfbPass));
    g_recorder.suspendedEfbPass.reset();
  } else {
    auto& pass = current_render_passes().emplace_back();
    pass.label = pass_label("EFB");
    set_efb_targets(pass);
    for (uint32_t i = 0; i < pass.colorAttachmentCount; ++i) {
      pass.colorAttachments[i].clear = false;
      pass.colorAttachments[i].loadOp = wgpu::LoadOp::Undefined;
    }
    pass.clearDepth = false;
  }
  ++g_recorder.currentRenderPass;
  set_efb_targets(current_render_passes()[g_recorder.currentRenderPass]);

  g_recorder.cachedViewport = g_recorder.suspendedEfbViewport;
  g_recorder.cachedScissor = g_recorder.suspendedEfbScissor;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});

  if (g_recorder.pendingStageEnd > g_recorder.nextStage) {
    run_stages_until(std::exchange(g_recorder.pendingStageEnd, 0), NoLiveMarker);
  }
  g_recorder.pendingStageEnd = 0;
}

void enqueue_op(FramePacket& frame, uint32_t opIndex) {
  if (opIndex >= frame.ops.size() || g_recorder.suppressRenderWorker) {
    return;
  }
  auto op = frame.ops[opIndex];
  render_worker::enqueue_encode_pass(frame.frameId, opIndex, [packet = &frame, op = std::move(op)] {
    if (!op_has_payload(op)) {
      return;
    }
    encode_op(packet->encoder, *packet, op);
  });
}

void flush_compute(FramePacket& frame) {
  if (frame.pendingCompute == UINT32_MAX) {
    return;
  }
  const uint32_t batch = std::exchange(frame.pendingCompute, UINT32_MAX);
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::Compute, batch));
  enqueue_op(frame, opIndex);
}

// Ops enqueue eagerly, so a batch emitted before the next op lands ahead of every pass not yet enqueued
void emit_op(FramePacket& frame, FrameOpType type, uint32_t index) {
  flush_compute(frame);
  frame.segmentReadsGpuState = false;
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, type, index));
  enqueue_op(frame, opIndex);
}

void enqueue_pass(FramePacket& frame, uint32_t passIndex) {
  seal_pass(frame, passIndex);
  emit_op(frame, FrameOpType::RenderPass, passIndex);
}

// Keyed by the batch's first sampling stage, so batches never share one; queue order keeps reads before the rewrite
struct StageDepthSnapshot {
  webgpu::TextureWithSampler texture;
  uint32_t generation = 0;
  uint32_t requests = 0;
};
std::array<StageDepthSnapshot, render::StageCount> g_stageDepthSnapshots;
std::optional<render::DepthSource> g_sceneDepthOverride;
bool g_fakeDepthSnapshots = false;

struct StageBatch {
  std::array<render::detail::StageData, render::StageCount> stages;
  uint8_t count = 0;
  uint8_t snapshotSlot = UINT8_MAX;
};

StageBatch build_stage_batch(uint8_t end, uint8_t liveMarker, bool atFinish) {
  auto& frame = current_frame_packet();
  StageBatch batch;
  for (uint8_t index = g_recorder.nextStage; index < end; ++index) {
    const bool seen = g_recorder.stageMarkerSeen[index];
    render::detail::StageData& data = batch.stages[batch.count];
    data = render::detail::StageData{
        .stage = static_cast<render::Stage>(index),
        .frameId = frame.frameId,
        .frameIndex = frame.frameIndex,
        .markerSeen = seen,
        .deferred = seen && index != liveMarker,
        .flushedAtFinish = atFinish,
        .frame = frame.renderState.get(),
    };
    if (!render::detail::build_stage_data(data)) {
      continue;
    }
    if (batch.snapshotSlot == UINT8_MAX && render::detail::stage_samples_scene_depth(data)) {
      batch.snapshotSlot = index;
    }
    ++batch.count;
  }
  g_recorder.nextStage = std::max(g_recorder.nextStage, end);
  return batch;
}

webgpu::TextureWithSampler create_stage_depth_snapshot(const wgpu::Extent3D& size) {
  if (g_fakeDepthSnapshots) {
    return {.size = size, .format = wgpu::TextureFormat::R32Float};
  }
  // The snapshot shader renders it and passes sample it; unlike resolve_pass snapshots nothing copies from it
  const wgpu::TextureDescriptor desc{
      .label = "Stage Depth Snapshot",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {size.width, size.height, 1},
      .format = wgpu::TextureFormat::R32Float,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = webgpu::g_device.CreateTexture(&desc);
  auto view = texture.CreateView();
  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = desc.size,
      .format = desc.format,
  };
}

// EFB passes attach the EFB depth and read it back through the same view; begin_color_pass targets do neither
bool is_efb_depth_pass(const RenderPass& pass) noexcept {
  return pass.depthStencilView.Get() == webgpu::g_depthBuffer.view.Get() &&
         pass.copySourceDepthView.Get() == webgpu::g_depthBuffer.view.Get();
}

render::detail::SceneDepth capture_scene_depth(RenderPass* sealed, const StageBatch& batch) {
  AURORA_ASSERT(!g_recorder.inOffscreen, "Scene depth captured while offscreen rendering is active");
  // Taken from the batch's own graph, so the views always match the plan it was compiled with
  render::detail::SceneDepth depth{.source = render::detail::stage_scene_depth_source(batch.stages[0])};
  if (sealed != nullptr && !is_efb_depth_pass(*sealed)) {
    static bool warned = false;
    if (!std::exchange(warned, true)) {
      Log.warn("Render stages sealed a pass without the EFB depth; their scene depth is missing");
    }
    return depth;
  }
  depth.attachmentTexture = webgpu::g_depthBuffer.texture;
  if (sealed != nullptr) {
    depth.attachment = sealed->depthStencilView;
    depth.attachmentFormat = sealed->depthStencilFormat;
    depth.attachmentSamples = sealed->msaaSamples;
    depth.size = sealed->colorAttachments[SceneColorAttachmentIndex].size;
  } else {
    depth.attachment = webgpu::g_depthBuffer.view;
    depth.attachmentFormat = webgpu::g_graphicsConfig.depthFormat;
    depth.attachmentSamples = webgpu::g_graphicsConfig.msaaSamples;
    depth.size = webgpu::g_frameBuffer.size;
  }
  if (batch.snapshotSlot == UINT8_MAX) {
    return depth;
  }
  if (depth.source == render::DepthSource::Direct) {
    depth.sampledTexture = webgpu::g_depthBuffer.texture;
    depth.sampled = sealed != nullptr ? sealed->copySourceDepthView : webgpu::g_depthBuffer.view;
    depth.sampledFormat = depth.attachmentFormat;
    // Direct sampling binds the whole view; a stencil aspect would need a depth-only view
    AURORA_ASSERT(depth.sampledFormat == wgpu::TextureFormat::Depth32Float ||
                      depth.sampledFormat == wgpu::TextureFormat::Depth24Plus ||
                      depth.sampledFormat == wgpu::TextureFormat::Depth16Unorm,
                  "Scene depth format {} cannot be sampled directly", static_cast<uint32_t>(depth.sampledFormat));
  } else if (depth.source == render::DepthSource::Copy && sealed != nullptr) {
    auto& snapshot = g_stageDepthSnapshots[batch.snapshotSlot];
    const bool missing = g_fakeDepthSnapshots ? snapshot.generation == 0 : !snapshot.texture.texture;
    if (missing || snapshot.texture.size.width != depth.size.width ||
        snapshot.texture.size.height != depth.size.height) {
      snapshot.texture = create_stage_depth_snapshot(depth.size);
      ++snapshot.generation;
    }
    AURORA_ASSERT(!sealed->snapshotDepthDst, "Scene depth captured twice on one render pass");
    sealed->snapshotDepthDst = snapshot.texture.view;
    depth.sampledTexture = snapshot.texture.texture;
    depth.sampled = snapshot.texture.view;
    depth.sampledFormat = wgpu::TextureFormat::R32Float;
    ++snapshot.requests;
  }
  return depth;
}

void emit_stage_batch(StageBatch& batch, const render::detail::SceneDepth& depth) {
  if (batch.count == 0) {
    return;
  }
  auto& frame = current_frame_packet();
  const auto shared = std::make_shared<const render::detail::SceneDepth>(depth);
  for (uint8_t i = 0; i < batch.count; ++i) {
    render::detail::StageData& data = batch.stages[i];
    data.sceneDepth = shared;
    const auto stageIndex = static_cast<uint32_t>(frame.stages.size());
    frame.stages.emplace_back(std::move(data));
    emit_op(frame, FrameOpType::RenderStage, stageIndex);
  }
}

// The open pass is sealed only once a stage has work, so an empty graph leaves the EFB pass untouched
bool emit_stage_ops(uint8_t end, uint8_t liveMarker, bool atFinish) {
  StageBatch batch = build_stage_batch(end, liveMarker, atFinish);
  if (batch.count == 0) {
    return false;
  }
  auto& frame = current_frame_packet();
  RenderPass* sealed =
      g_recorder.currentRenderPass != UINT32_MAX ? &current_render_passes()[g_recorder.currentRenderPass] : nullptr;
  // Captured before the discardable check, so a snapshot keeps an otherwise empty sealed pass alive
  const render::detail::SceneDepth depth = capture_scene_depth(sealed, batch);
  if (sealed != nullptr) {
    sealed->discardable = !sealed->has_consumer() && !sealed->has_content();
    enqueue_pass(frame, g_recorder.currentRenderPass);
  }
  emit_stage_batch(batch, depth);
  return true;
}

void run_stages_until(uint8_t end, uint8_t liveMarker) {
  const bool hadPass = g_recorder.currentRenderPass != UINT32_MAX;
  if (emit_stage_ops(end, liveMarker, false) && hadPass) {
    resume_efb_pass_loading(current_render_passes()[g_recorder.currentRenderPass]);
  }
}

enum class StageWarning : uint8_t { NotAMarker, LateOrRepeated };

void warn_stage_once(uint8_t index, StageWarning kind) {
  // Every invalid value shares the last slot, so a corrupt stream logs once instead of once per byte value
  static std::array<std::array<bool, 2>, 8> warned{};
  bool& flag = warned[std::min<uint8_t>(index, 7)][static_cast<size_t>(kind)];
  if (std::exchange(flag, true)) {
    return;
  }
  const std::string_view text =
      kind == StageWarning::NotAMarker ? "is not a GX stage marker" : "arrived after a later stage or twice; ignored";
  Log.warn("Render stage {} {}", render::stage_name(static_cast<render::Stage>(index)), text);
}
} // namespace

namespace detail {

bool should_arm_mod_normal(bool requested, bool offscreen, bool normalPresent, bool core, uint32_t msaa) noexcept {
  return requested && !offscreen && !normalPresent && core && msaa == 1;
}

void begin_recording(FramePacket& packet, size_t frameSlot) {
  CHECK(!g_recorder.active(), "A recording session is already active");
  if (g_recorder.normalRequested && webgpu::enable_normal_buffer()) {
    g_recorder.normalRequested = false;
  }
  g_recorder.packet = &packet;
  g_recorder.frameSlot = frameSlot;
  g_passSnapshotPools[frameSlot].used = 0;
  g_recorder.drawCallCount = 0;
  g_recorder.mergedDrawCallCount = 0;
  g_recorder.suspendedEfbPass.reset();
  g_recorder.nextStage = 0;
  g_recorder.pendingStageEnd = 0;
  g_recorder.stageMarkerSeen = {};
  g_channelStorageBytes = 0;

  current_render_passes().emplace_back();
  auto& pass = current_render_passes()[0];
  pass.label = pass_label("EFB");
  set_efb_targets(pass);
  pass.colorAttachments[SceneColorAttachmentIndex].clearValue = gx::g_gxState.clearColor;
  pass.clearDepthValue = gx::clear_depth_value();
  g_recorder.currentRenderPass = 0;
  g_recorder.cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_recorder.cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

RecordedFrame end_recording() {
  CHECK(g_recorder.active(), "No active recording session");
  AURORA_ASSERT(!g_recorder.inOffscreen, "end_frame called while offscreen rendering is active");
  AURORA_ASSERT(g_recorder.currentRenderPass == UINT32_MAX,
                "end_frame called before finish finalized the current render pass");
  CHECK(g_recorder.pendingStageEnd == 0, "end_frame called with render stages still pending");
  CHECK(g_recorder.frame().pendingCompute == UINT32_MAX, "end_frame called with a compute batch still pending");
  auto& frame = g_recorder.frame();
  frame.stats.drawCallCount = g_recorder.drawCallCount;
  frame.stats.mergedDrawCallCount = g_recorder.mergedDrawCallCount;
  frame.stats.lastVertSize = frame.verts.size();
  frame.stats.lastUniformSize = frame.uniforms.size();
  g_lastRecordedUniformSize = frame.stats.lastUniformSize;
  frame.stats.lastIndexSize = frame.indices.size();
  frame.stats.lastStorageSize = frame.storage.size();
  frame.stats.lastTextureUploadSize = frame.textureUpload.size();
  g_lastChannelStorageBytes = g_channelStorageBytes;

  for (auto& array : gx::g_gxState.arrays) {
    array.cachedRange = {};
  }
#if defined(AURORA_GFX_DEBUG_GROUPS)
  if (!g_recorder.debugGroupStack.empty()) {
    for (auto& item : std::ranges::reverse_view(g_recorder.debugGroupStack)) {
      Log.warn("Debug group was not popped at end of frame: {}", item);
    }
    g_recorder.debugGroupStack.clear();
  }
#endif
  const RecordedFrame recorded{.packet = g_recorder.packet, .frameSlot = g_recorder.frameSlot};
  g_recorder.packet = nullptr;
  g_recorder.frameSlot = 0;
  g_offscreenCache.end_frame();
  TracyPlot("aurora: offscreenCacheBytes", static_cast<int64_t>(g_offscreenCache.cached_bytes()));
  TracyPlot("aurora: offscreenCacheEntries", static_cast<int64_t>(g_offscreenCache.cached_count()));
  return recorded;
}

void shutdown_recording() {
  release_compute_encoding();
  for (auto& pool : g_passSnapshotPools) {
    pool = {};
  }
  g_stageDepthSnapshots = {};
  g_sceneDepthOverride.reset();
  g_fakeDepthSnapshots = false;
  g_recorder.currentRenderPass = UINT32_MAX;
  g_offscreenCache.clear();
  g_recorder.offscreenColor = {};
  g_recorder.offscreenDepth = {};
  g_recorder.suspendedEfbPass.reset();
  g_recorder.inOffscreen = false;
  g_recorder.packet = nullptr;
  g_recorder.frameSlot = 0;
  g_lastRecordedUniformSize = 0;
  g_channelStorageBytes = 0;
  g_lastChannelStorageBytes = 0;
  g_recorder.suppressRenderWorker = false;
  g_recorder.normalRequested = false;
  g_recorder.nextStage = 0;
  g_recorder.pendingStageEnd = 0;
  g_recorder.stageMarkerSeen = {};
}

namespace testing {

void suppress_render_worker(bool suppress) noexcept { g_recorder.suppressRenderWorker = suppress; }

void seed_offscreen_cache(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                          wgpu::TextureFormat depthFormat) {
  const wgpu::Extent3D size{width, height, 1};
  const OffscreenCacheEntry entry{
      .color = {.size = size, .format = colorFormat},
      .depth = {.size = size, .format = depthFormat},
  };
  g_offscreenCache.insert({width, height, colorFormat, depthFormat}, entry, entry.size_bytes());
}

OffscreenCacheStats offscreen_cache_stats() noexcept {
  return {.bytes = g_offscreenCache.cached_bytes(), .entries = g_offscreenCache.cached_count()};
}

bool is_offscreen_cached(uint32_t width, uint32_t height, wgpu::TextureFormat colorFormat,
                         wgpu::TextureFormat depthFormat) {
  return g_offscreenCache.contains({width, height, colorFormat, depthFormat});
}

void override_scene_depth_source(std::optional<render::DepthSource> source) noexcept { g_sceneDepthOverride = source; }

void fake_depth_snapshots(bool fake) noexcept { g_fakeDepthSnapshots = fake; }

uint32_t stage_depth_snapshot_generation(render::Stage slot) noexcept {
  const auto index = static_cast<size_t>(slot);
  return index < g_stageDepthSnapshots.size() ? g_stageDepthSnapshots[index].generation : 0;
}

uint32_t stage_depth_snapshot_requests(render::Stage slot) noexcept {
  const auto index = static_cast<size_t>(slot);
  return index < g_stageDepthSnapshots.size() ? g_stageDepthSnapshots[index].requests : 0;
}

} // namespace testing

void increment_merged_draw_count() noexcept {
  if (g_recorder.active()) {
    ++g_recorder.mergedDrawCallCount;
  }
}

uint64_t take_resident_request() noexcept { return g_residentRequest.exchange(0, std::memory_order_relaxed); }

} // namespace detail

void queue_texture_upload(TextureUpload upload) {
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    AURORA_ASSERT(!current_render_passes()[g_recorder.currentRenderPass].sealed,
                  "Attempted to append texture upload to sealed render pass {}", g_recorder.currentRenderPass);
  }
  current_frame_packet().textureUploads.emplace_back(std::move(upload));
}

void queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) {
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  auto& frame = current_frame_packet();
  if (frame.textureUpload.size() + copyBytesPerRow * rowsPerImage <= TextureUploadSize) {
    const auto range = push_texture_data(data, bytesPerRow, rowsPerImage);
    const wgpu::TexelCopyBufferLayout layout{
        .offset = range.offset,
        .bytesPerRow = bytesPerRow,
        .rowsPerImage = rowsPerImage,
    };
    queue_texture_upload(TextureUpload{layout, std::move(tex), size});
    return;
  }

  const uint64_t uploadSize = copyBytesPerRow * rowsPerImage;
  const wgpu::BufferDescriptor descriptor{
      .label = "Overflow Texture Upload Buffer",
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = uploadSize,
      .mappedAtCreation = true,
  };
  auto buffer = webgpu::g_device.CreateBuffer(&descriptor);
  auto* dst = static_cast<uint8_t*>(buffer.GetMappedRange(0, uploadSize));
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  buffer.Unmap();

  const wgpu::TexelCopyBufferLayout layout{
      .offset = 0,
      .bytesPerRow = bytesPerRow,
      .rowsPerImage = rowsPerImage,
  };
  queue_texture_upload(TextureUpload{layout, std::move(tex), size, std::move(buffer)});
}

void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recorder.currentRenderPass);
    g_recorder.currentRenderPass = UINT32_MAX;
  }

  const auto copyIndex = static_cast<uint32_t>(frame.textureCopies.size());
  frame.textureCopies.emplace_back(TextureCopy{
      .src = std::move(src),
      .dst = std::move(dst),
      .size = size,
  });
  emit_op(frame, FrameOpType::TextureCopy, copyIndex);
}

void begin_color_pass(const ColorPassDescriptor& desc) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recorder.currentRenderPass);
  }

  RenderPass pass{
      .label = desc.label != nullptr ? desc.label : "",
      .depthStencilView = desc.depthStencilView,
      .depthStencilFormat = desc.depthStencilFormat,
      .msaaSamples = desc.sampleCount,
      .clearDepthValue = desc.depthClearValue,
      .depthLoadOp = desc.depthLoadOp,
      .depthStoreOp = desc.depthStoreOp,
      .stencilLoadOp = desc.stencilLoadOp,
      .stencilStoreOp = desc.stencilStoreOp,
      .stencilClearValue = desc.stencilClearValue,
      .clearDepth = desc.depthLoadOp == wgpu::LoadOp::Clear,
      .hasDepth = desc.hasDepth,
      .hasStencil = desc.hasStencil,
  };
  set_single_color_target(pass, desc.colorFormat, desc.targetSize, desc.colorView, desc.resolveView);
  auto& color = pass.colorAttachments[SceneColorAttachmentIndex];
  color.clearValue = {
      static_cast<float>(desc.clearColor.r),
      static_cast<float>(desc.clearColor.g),
      static_cast<float>(desc.clearColor.b),
      static_cast<float>(desc.clearColor.a),
  };
  color.loadOp = desc.colorLoadOp;
  color.storeOp = desc.colorStoreOp;
  color.clear = desc.colorLoadOp == wgpu::LoadOp::Clear;
  pass.commands.reserve(128);
  frame.renderPasses.emplace_back(std::move(pass));
  g_recorder.currentRenderPass = static_cast<uint32_t>(frame.renderPasses.size() - 1);

  g_recorder.cachedViewport = {
      0.f, 0.f, static_cast<float>(desc.targetSize.width), static_cast<float>(desc.targetSize.height), 0.f, 1.f};
  g_recorder.cachedScissor = {0, 0, static_cast<int32_t>(desc.targetSize.width),
                              static_cast<int32_t>(desc.targetSize.height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void end_color_pass() {
  ZoneScoped;
  if (g_recorder.currentRenderPass == UINT32_MAX) {
    return;
  }
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  g_recorder.currentRenderPass = UINT32_MAX;
}

template <>
gx::DrawData* get_last_draw_command() {
  if (g_recorder.currentRenderPass >= current_render_passes().size()) {
    return nullptr;
  }
  auto& last = current_render_passes()[g_recorder.currentRenderPass].commands.back();
  if (last.type != CommandType::Draw || last.data.draw.encoder != encode_draw<gx::render, gx::DrawData>) {
    return nullptr;
  }
  return &inline_payload<gx::DrawData>(last.data.draw.payload.data());
}

Vec2<uint32_t> get_render_target_size() noexcept {
  if (g_recorder.currentRenderPass < current_render_passes().size()) {
    const auto& size =
        current_render_passes()[g_recorder.currentRenderPass].colorAttachments[SceneColorAttachmentIndex].size;
    return {size.width, size.height};
  }
  const auto windowSize = window::get_window_size();
  return {windowSize.fb_width, windowSize.fb_height};
}

void set_viewport(const Viewport& cmd) noexcept {
  if (cmd != g_recorder.cachedViewport) {
    push_command(CommandType::SetViewport, Command::Data{.setViewport = cmd});
    g_recorder.cachedViewport = cmd;
  }
}

void set_scissor(const ClipRect& cmd) noexcept {
  if (cmd != g_recorder.cachedScissor) {
    push_command(CommandType::SetScissor, Command::Data{.setScissor = cmd});
    g_recorder.cachedScissor = cmd;
  }
}

template <>
void push_draw_command(clear::DrawData data) {
  push_draw_command(make_draw_command<clear::render>(data));
}

template <>
PipelineRef pipeline_ref(const clear::PipelineConfig& config) {
  return find_pipeline(config, get_render_target_layout());
}

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat) {
  // Resolve current render pass
  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  prevPass.resolveSourceFormat = sourceFormat;
  const auto srcW = static_cast<float>(prevPass.colorAttachments[SceneColorAttachmentIndex].size.width);
  const auto srcH = static_cast<float>(prevPass.colorAttachments[SceneColorAttachmentIndex].size.height);
  const tex_copy_conv::Uniforms uniforms{
      .offset = {static_cast<float>(rect.x) / srcW, static_cast<float>(rect.y) / srcH},
      .scale = {static_cast<float>(rect.width) / srcW, static_cast<float>(rect.height) / srcH},
      .opaqueAlpha = !gx::efb_has_alpha(sourceFormat),
  };
  prevPass.resolveUniformRange = push_uniform(uniforms);
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);

  // Populate new render pass from previous
  const auto msaaSamples = prevPass.msaaSamples;
  const auto& targetSize = prevPass.colorAttachments[SceneColorAttachmentIndex].size;
  const auto left = std::clamp<int32_t>(rect.x, 0, static_cast<int32_t>(targetSize.width));
  const auto top = std::clamp<int32_t>(rect.y, 0, static_cast<int32_t>(targetSize.height));
  const auto right = std::clamp<int32_t>(rect.x + rect.width, left, static_cast<int32_t>(targetSize.width));
  const auto bottom = std::clamp<int32_t>(rect.y + rect.height, top, static_cast<int32_t>(targetSize.height));
  const ClipRect clearRect{.x = left, .y = top, .width = right - left, .height = bottom - top};
  const bool fullTarget = left == 0 && top == 0 && right == targetSize.width && bottom == targetSize.height;
  RenderPass newPass{
      .label = pass_label(g_recorder.inOffscreen ? "Offscreen" : "EFB"),
      .colorAttachments = prevPass.colorAttachments,
      .colorAttachmentCount = prevPass.colorAttachmentCount,
      .depthStencilView = prevPass.depthStencilView,
      .depthStencilFormat = prevPass.depthStencilFormat,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .copySourceNormalTexture = prevPass.copySourceNormalTexture,
      .msaaSamples = msaaSamples,
      .clearDepthValue = clearDepthValue,
      .clearDepth = clearDepth && fullTarget,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  const bool fullColorClear = clearColor && clearAlpha && fullTarget;
  for (uint32_t i = 0; i < newPass.colorAttachmentCount; ++i) {
    auto& color = newPass.colorAttachments[i];
    color.loadOp = wgpu::LoadOp::Undefined;
    if (clears_with_depth(color.semantic) || clears_with_color(color.semantic)) {
      // Geometry targets reset with depth, colour records with colour
      color.clear = (clears_with_depth(color.semantic) && clearDepth && fullTarget) ||
                    (clears_with_color(color.semantic) && clearColor && fullTarget);
    } else if (color.semantic == ColorAttachmentSemantic::DebugId) {
      color.clear = clearColor && fullTarget;
    } else {
      color.clear = false;
    }
  }
  if (fullColorClear) {
    auto& sceneColor = newPass.colorAttachments[SceneColorAttachmentIndex];
    sceneColor.clear = true;
    sceneColor.clearValue = clearColorValue;
  }
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;

  const bool drawClearColor = clearColor && !fullColorClear;
  const bool drawClearAlpha = clearAlpha && !fullColorClear;
  const bool drawClearDepth = clearDepth && !fullTarget;
  if (clearRect.width > 0 && clearRect.height > 0 && (drawClearColor || drawClearAlpha || drawClearDepth)) {
    // If we're only clearing a portion of the render target (or only one of color/alpha), perform a clear draw
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .clearColor = drawClearColor,
            .clearAlpha = drawClearAlpha,
            .clearDepth = drawClearDepth,
        }),
        .color =
            wgpu::Color{
                .r = clearColorValue.x(),
                .g = clearColorValue.y(),
                .b = clearColorValue.z(),
                .a = clearColorValue.w(),
            },
        .depth = clearDepthValue,
        .rect = clearRect,
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void queue_palette_conv(tex_palette_conv::ConvRequest req) {
  auto& renderPass = current_render_passes()[g_recorder.currentRenderPass];
  AURORA_ASSERT(!renderPass.sealed, "Attempted to append palette conversion to sealed render pass {}",
                g_recorder.currentRenderPass);
  renderPass.paletteConvs.push_back(std::move(req));
}

bool is_offscreen() noexcept { return g_recorder.inOffscreen; }

bool has_normal_attachment() noexcept {
  CHECK(g_recorder.currentRenderPass != UINT32_MAX, "has_normal_attachment called outside of a frame");
  const auto& pass = current_render_passes()[g_recorder.currentRenderPass];
  for (uint32_t i = 0; i < pass.colorAttachmentCount; ++i) {
    if (pass.colorAttachments[i].semantic == ColorAttachmentSemantic::Normal) {
      return true;
    }
  }
  return false;
}

RenderTargetLayout get_render_target_layout() noexcept {
  CHECK(g_recorder.currentRenderPass != UINT32_MAX, "get_render_target_layout called outside of a frame");
  return current_render_passes()[g_recorder.currentRenderPass].target_layout();
}

bool push_custom_draw(DrawTypeId type, const void* payload, size_t payloadSize) {
  if (type == InvalidDrawType) {
    Log.warn("push_custom_draw: invalid draw type");
    return false;
  }
  if (payloadSize > InlineDrawPayloadSize) {
    Log.warn("push_custom_draw: payload size {} exceeds inline payload size {}", payloadSize, InlineDrawPayloadSize);
    return false;
  }
  if (payloadSize > 0 && payload == nullptr) {
    Log.warn("push_custom_draw: non-zero payload size with null payload");
    return false;
  }
  if (!find_runtime_draw_type(type)) {
    Log.warn("push_custom_draw: unregistered draw type {:#x}", type);
    return false;
  }

  trace::detail::note_untraceable("push_custom_draw");
  gx::fifo::drain();

  if (!g_recorder.active() || g_recorder.currentRenderPass == UINT32_MAX) {
    Log.warn("push_custom_draw: called outside an active render pass");
    return false;
  }

  CustomDrawCommand draw{};
  draw.type = type;
  draw.payloadSize = static_cast<uint32_t>(payloadSize);
  if (payloadSize > 0) {
    std::memcpy(draw.payload.data(), payload, payloadSize);
  }
  push_command(CommandType::CustomDraw, Command::Data{.customDraw = draw});
  ++g_recorder.drawCallCount;
  return true;
}

void begin_offscreen(uint32_t width, uint32_t height) {
  ZoneScoped;
  AURORA_ASSERT(width != 0 && height != 0, "begin_offscreen requires nonzero dimensions ({}x{})", width, height);
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "begin_offscreen called outside of an active recording frame");

  if (g_recorder.inOffscreen) {
    finish_current_offscreen();
  } else {
    suspend_efb();
  }
  start_offscreen(width, height);
}

void end_offscreen() {
  ZoneScoped;
  finish_current_offscreen();
  restore_efb();
}

bool create_pass(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) {
    Log.warn("create_pass: invalid size {}x{}", width, height);
    return false;
  }

  trace::detail::note_untraceable("create_pass");
  gx::fifo::drain();

  if (!g_recorder.active() || g_recorder.currentRenderPass == UINT32_MAX) {
    Log.warn("create_pass: called outside an active render pass");
    return false;
  }
  if (g_recorder.inOffscreen) {
    Log.warn("create_pass: an offscreen pass is already active (nesting is unsupported)");
    return false;
  }

  begin_offscreen(width, height);
  return true;
}

bool resolve_pass(const ResolveDesc& desc, ResolvedTargets& out) {
  out = {};
  trace::detail::note_untraceable("resolve_pass");
  gx::fifo::drain();

  if (!g_recorder.active() || g_recorder.currentRenderPass == UINT32_MAX) {
    Log.warn("resolve_pass: called outside an active render pass");
    return false;
  }

  bool wantDepth = desc.depth;
  if (wantDepth && !tex_copy_conv::snapshot_depth_supported()) {
    Log.warn("resolve_pass: depth snapshots are unsupported on this device");
    wantDepth = false;
  }

  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  const uint32_t width = prevPass.colorAttachments[SceneColorAttachmentIndex].size.width;
  const uint32_t height = prevPass.colorAttachments[SceneColorAttachmentIndex].size.height;
  const bool wantNormal = desc.normal && prevPass.copySourceNormalTexture;
  // A request under the G-buffer reads its normal, so it must not leave a sticky legacy buffer behind
  if (detail::should_arm_mod_normal(desc.normal, g_recorder.inOffscreen, webgpu::normal_present(),
                                    webgpu::g_hasCoreFeatures, webgpu::g_graphicsConfig.msaaSamples)) {
    g_recorder.normalRequested = true;
  }
  // Requesting no snapshots is a plain pass break (or offscreen close, discarding its output).
  if (desc.color || wantDepth || wantNormal) {
    auto& entry = acquire_pass_snapshot(width, height, desc.color, wantDepth, wantNormal);
    if (desc.color) {
      prevPass.snapshotColorDst = entry.color.texture;
      out.colorTexture = entry.color.texture;
      out.color = entry.color.view;
      out.colorFormat = entry.color.format;
    }
    if (wantDepth) {
      prevPass.snapshotDepthDst = entry.depth.view;
      out.depthTexture = entry.depth.texture;
      out.depth = entry.depth.view;
    }
    if (wantNormal) {
      prevPass.snapshotNormalDst = entry.normal.texture;
      out.normalTexture = entry.normal.texture;
      out.normal = entry.normal.view;
    }
  }
  out.width = width;
  out.height = height;

  if (g_recorder.inOffscreen) {
    // Seal the offscreen pass and resume the EFB.
    end_offscreen();
    return true;
  }

  // Seal the current EFB pass and continue on a new one that loads the existing contents.
  // EFB writes persist into the loading continuation, so content keeps the pass alive even
  // without a consumer.
  prevPass.discardable = !prevPass.has_consumer() && !prevPass.has_content();
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  resume_efb_pass_loading(prevPass);
  return true;
}

bool push_encoder_task(EncoderTaskId type, const void* payload, size_t payloadSize) {
  if (type == InvalidEncoderTask) {
    Log.warn("push_encoder_task: invalid encoder task type");
    return false;
  }
  if (payloadSize > InlineDrawPayloadSize) {
    Log.warn("push_encoder_task: payload size {} exceeds inline payload size {}", payloadSize, InlineDrawPayloadSize);
    return false;
  }
  if (payloadSize > 0 && payload == nullptr) {
    Log.warn("push_encoder_task: non-zero payload size with null payload");
    return false;
  }
  if (!find_runtime_encoder_task_type(type)) {
    Log.warn("push_encoder_task: unregistered encoder task type {:#x}", type);
    return false;
  }

  trace::detail::note_untraceable("push_encoder_task");
  gx::fifo::drain();

  if (!g_recorder.active() || g_recorder.currentRenderPass == UINT32_MAX) {
    Log.warn("push_encoder_task: called outside an active render pass");
    return false;
  }
  if (g_recorder.inOffscreen) {
    Log.warn("push_encoder_task: unsupported while an offscreen pass is active");
    return false;
  }

  // Seal the current EFB pass, record the task between it and a continuation
  // pass that loads the existing contents. EFB writes persist into the continuation, so
  // content keeps the sealed pass alive even without a consumer.
  auto& frame = current_frame_packet();
  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  prevPass.discardable = !prevPass.has_consumer() && !prevPass.has_content();
  enqueue_pass(frame, g_recorder.currentRenderPass);

  const auto taskIndex = static_cast<uint32_t>(frame.encoderTasks.size());
  auto& task = frame.encoderTasks.emplace_back(EncoderTask{.type = type});
  task.payloadSize = static_cast<uint32_t>(payloadSize);
  if (payloadSize > 0) {
    std::memcpy(task.payload.data(), payload, payloadSize);
  }
  emit_op(frame, FrameOpType::EncoderTask, taskIndex);

  resume_efb_pass_loading(prevPass);
  return true;
}

void record_stage(render::Stage stage) {
  const auto index = static_cast<uint8_t>(stage);
  if (index > LastMarkerStage) {
    warn_stage_once(index, StageWarning::NotAMarker);
    return;
  }
  if (!g_recorder.active() || g_recorder.packet->renderState == nullptr) {
    return;
  }
  if (index < std::max(g_recorder.nextStage, g_recorder.pendingStageEnd) || g_recorder.stageMarkerSeen[index]) {
    warn_stage_once(index, StageWarning::LateOrRepeated);
    return;
  }
  g_recorder.stageMarkerSeen[index] = true;
  if (g_recorder.inOffscreen || g_recorder.currentRenderPass == UINT32_MAX) {
    g_recorder.pendingStageEnd = std::max<uint8_t>(g_recorder.pendingStageEnd, index + 1);
    return;
  }
  const auto end = std::max<uint8_t>(index + 1, std::exchange(g_recorder.pendingStageEnd, 0));
  run_stages_until(end, index);
}

template <>
void push_draw_command(gx::DrawData data) {
  push_draw_command(make_draw_command<gx::render>(data));
}

#ifdef AURORA_ENABLE_RMLUI
template <>
void push_draw_command(rmlui::DrawData data) {
  push_draw_command(make_draw_command<rmlui::render>(data));
}
#endif

template <>
PipelineRef pipeline_ref(const gx::PipelineConfig& config) {
  return find_pipeline(config, get_render_target_layout());
}

#ifdef AURORA_ENABLE_RMLUI
template <>
PipelineRef pipeline_ref(const rmlui::PipelineConfig& config) {
  return find_pipeline(config);
}
#endif

void finish() {
  ZoneScoped;
  if (!g_recorder.active()) {
    return;
  }
  AURORA_ASSERT(!g_recorder.inOffscreen, "finish called while offscreen rendering is active");
  // Built before the final pass is sealed, so the batch's scene depth can still be captured on it
  StageBatch batch = g_recorder.packet->renderState != nullptr
                         ? build_stage_batch(LastMarkerStage + 1, NoLiveMarker, true)
                         : StageBatch{};
  RenderPass* finalPass = nullptr;
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    auto& frame = current_frame_packet();
    frame.uniforms.append_zeroes(gx::MaxUniformSize);
    finalPass = &frame.renderPasses[g_recorder.currentRenderPass];
    finalPass->captureDepthSnapshot = true;
  }
  const render::detail::SceneDepth depth =
      batch.count != 0 ? capture_scene_depth(finalPass, batch) : render::detail::SceneDepth{};
  if (finalPass != nullptr) {
    enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
    g_recorder.currentRenderPass = UINT32_MAX;
  }
  emit_stage_batch(batch, depth);
  g_recorder.pendingStageEnd = 0;
}

render::DepthSource current_scene_depth_source() noexcept {
  if (g_sceneDepthOverride) {
    return *g_sceneDepthOverride;
  }
  return render::detail::select_scene_depth_source(webgpu::g_hasCoreFeatures, webgpu::g_graphicsConfig.msaaSamples,
                                                   tex_copy_conv::snapshot_depth_supported());
}

Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  if (!check_recording("push_verts")) {
    return {};
  }
  return push(current_frame_packet().verts, data, length, alignment);
}

Range push_indices(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  if (!check_recording("push_indices")) {
    return {};
  }
  return push(current_frame_packet().indices, data, length, alignment);
}

Range push_uniform(const uint8_t* data, size_t length) {
  ZoneScoped;
  if (!check_recording("push_uniform")) {
    return {};
  }
  return push(current_frame_packet().uniforms, data, length, resources().limits.minUniformBufferOffsetAlignment);
}

Range push_storage(const uint8_t* data, size_t length) {
  ZoneScoped;
  if (!check_recording("push_storage")) {
    return {};
  }
  return push(current_frame_packet().storage, data, length, resources().limits.minStorageBufferOffsetAlignment);
}

Range push_channel_storage(std::span<const uint8_t> bytes) noexcept {
  if (!g_recorder.active()) {
    return {};
  }
  ByteBuffer& storage = current_frame_packet().storage;
  const size_t alignment = resources().limits.minStorageBufferOffsetAlignment;
  const size_t before = storage.size();
  const size_t begin = alignment != 0 ? AURORA_ALIGN(before, alignment) : before;
  const size_t padded = AURORA_ALIGN(bytes.size(), size_t{4});
  if (!storage.owned() && begin + padded > storage.capacity()) {
    return {};
  }
  const Range range = push(storage, bytes.data(), bytes.size(), alignment);
  // The next op's copy restarts at a 4-byte boundary, so a range must end on one
  if (padded > bytes.size()) {
    constexpr std::array<uint8_t, 4> Zeroes{};
    storage.append(Zeroes.data(), padded - bytes.size());
  }
  g_channelStorageBytes += storage.size() - before;
  return range;
}

Range reserve_channel_storage(size_t length, bool zero) noexcept {
  if (!g_recorder.active()) {
    return {};
  }
  ByteBuffer& storage = current_frame_packet().storage;
  const size_t alignment = resources().limits.minStorageBufferOffsetAlignment;
  const size_t before = storage.size();
  const size_t begin = alignment != 0 ? AURORA_ALIGN(before, alignment) : before;
  const size_t padded = AURORA_ALIGN(length, size_t{4});
  if (!storage.owned() && begin + padded > storage.capacity()) {
    return {};
  }
  const Range range = map(storage, padded, alignment);
  if (zero) {
    std::memset(storage.data() + range.offset, 0, padded);
  }
  g_channelStorageBytes += storage.size() - before;
  return range;
}

uint64_t last_channel_storage_bytes() noexcept { return g_lastChannelStorageBytes; }

ResidentStorage resident_storage() noexcept {
  if (!g_recorder.active()) {
    return {};
  }
  const FramePacket& frame = current_frame_packet();
  return {
      .frameId = frame.frameId,
      .base = static_cast<uint32_t>(frame.sizes.storage),
      .capacity = frame.residentCapacity,
      .limit = frame.residentLimit,
      .blockAlign = resources().limits.minStorageBufferOffsetAlignment,
      .epoch = frame.residentEpoch,
  };
}

void copy_to_resident(Range staged, uint32_t offset, uint32_t blockSize) noexcept {
  if (!g_recorder.active()) {
    return;
  }
  current_frame_packet().residentCopies.push_back({
      .src = staged.offset,
      .dst = offset,
      .copySize = AURORA_ALIGN(staged.size, 4u),
      .blockSize = blockSize,
  });
}

void request_resident_capacity(uint64_t bytes) noexcept {
  uint64_t current = g_residentRequest.load(std::memory_order_relaxed);
  while (current < bytes && !g_residentRequest.compare_exchange_weak(current, bytes, std::memory_order_relaxed)) {}
}

Range push_texture_data(const uint8_t* data, u32 bytesPerRow, u32 rowsPerImage) {
  // For CopyBufferToTexture, we need an alignment of 256 per row (see Dawn kTextureBytesPerRowAlignment)
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  const auto range = map(current_frame_packet().textureUpload, copyBytesPerRow * rowsPerImage, 0);
  u8* dst = current_frame_packet().textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}

uint32_t align_uniform(uint32_t value) {
  return AURORA_ALIGN(value, detail::resources().limits.minUniformBufferOffsetAlignment);
}

uint64_t uniform_space_left() noexcept {
  if (!g_recorder.active()) {
    return 0;
  }
  const ByteBuffer& uniforms = current_frame_packet().uniforms;
  if (uniforms.owned()) {
    return UINT64_MAX;
  }
  const size_t alignment = resources().limits.minUniformBufferOffsetAlignment;
  const size_t begin = alignment != 0 ? AURORA_ALIGN(uniforms.size(), alignment) : uniforms.size();
  return begin < uniforms.capacity() ? uniforms.capacity() - begin : 0;
}

namespace {
RecordResult refuse_record(std::string_view what, RecordResult result) noexcept {
  static std::array<bool, magic_enum::enum_count<RecordResult>()> warned{};
  if (!std::exchange(warned[static_cast<size_t>(result)], true)) {
    Log.warn("{}: {}; nothing queued", what, magic_enum::enum_name(result));
  }
  return result;
}

RecordResult check_segment(std::string_view what) noexcept {
  if (!g_recorder.active()) {
    return refuse_record(what, RecordResult::NoRecording);
  }
  if (g_recorder.currentRenderPass == UINT32_MAX) {
    return refuse_record(what, RecordResult::NoOpenPass);
  }
  if (current_frame_packet().segmentReadsGpuState) {
    return refuse_record(what, RecordResult::SegmentReadsState);
  }
  return RecordResult::Recorded;
}

bool bad_storage_range(const FramePacket& frame, Range range) noexcept {
  return range.offset % 4 != 0 || range.size % 4 != 0 || uint64_t{range.offset} + range.size > frame.storage.size();
}

ComputeBatch& pending_batch(FramePacket& frame) {
  if (frame.pendingCompute == UINT32_MAX) {
    frame.pendingCompute = static_cast<uint32_t>(frame.computeBatches.size());
    frame.computeBatches.emplace_back();
  }
  return frame.computeBatches[frame.pendingCompute];
}
} // namespace

namespace detail {
RecordResult record_dispatch(ComputeBinder bind, std::span<const std::byte> payload,
                             const DispatchArgs& args) noexcept {
  AURORA_ASSERT(!render_worker::is_worker_thread(), "record_dispatch called on the render worker");
  constexpr std::string_view What = "record_dispatch";
  if (const RecordResult result = check_segment(What); result != RecordResult::Recorded) {
    return result;
  }
  FramePacket& frame = current_frame_packet();
  if (bad_storage_range(frame, args.output) ||
      std::ranges::any_of(args.inputs, [&](const Range& input) { return bad_storage_range(frame, input); })) {
    return refuse_record(What, RecordResult::BadRange);
  }
  if (args.inputs.size() > DispatchMaxInputs) {
    return refuse_record(What, RecordResult::TooManyInputs);
  }
  if (args.user.size() > DispatchUserSize) {
    return refuse_record(What, RecordResult::UserTooLarge);
  }
  const auto& limits = resources().limits;
  if (std::ranges::any_of(args.groups,
                          [&](uint32_t groups) { return groups > limits.maxComputeWorkgroupsPerDimension; })) {
    return refuse_record(What, RecordResult::GroupLimit);
  }
  AURORA_ASSERT(payload.size() <= ComputePayloadSize, "record_dispatch payload of {} bytes", payload.size());
  ComputeEntry entry{.kind = ComputeEntry::Kind::Dispatch, .range = args.output, .groups = args.groups, .bind = bind};
  std::memcpy(entry.payload.data(), payload.data(), payload.size());
  if (std::ranges::find(args.groups, 0u) != args.groups.end()) {
    if (args.output.size != 0) {
      pending_batch(frame).entries.push_back(entry);
    }
    return RecordResult::Recorded;
  }
  if (frame.computeParamsBytes + DispatchParamsSize > DispatchParamsBudget) {
    return refuse_record(What, RecordResult::ParamsBudget);
  }
  if (uniform_space_left() < DispatchParamsSize + gx::MaxUniformSize) {
    return refuse_record(What, RecordResult::UniformSpace);
  }
  DispatchParams params{};
  params.output = {args.output.offset / 4, args.output.size / 4, 0, 0};
  for (size_t i = 0; i < args.inputs.size(); ++i) {
    params.inputs[2 * i] = args.inputs[i].offset / 4;
    params.inputs[2 * i + 1] = args.inputs[i].size / 4;
  }
  params.groups = {args.groups[0], args.groups[1], args.groups[2], static_cast<uint32_t>(args.inputs.size())};
  params.elements = {args.elements[0], args.elements[1], args.elements[2], 0};
  if (!args.user.empty()) {
    std::memcpy(params.user.data(), args.user.data(), args.user.size());
  }
  const uint32_t alignment = limits.minUniformBufferOffsetAlignment;
  const Range block = push(frame.uniforms, reinterpret_cast<const uint8_t*>(&params), sizeof(params), alignment);
  frame.computeParamsBytes += alignment != 0 ? AURORA_ALIGN(DispatchParamsSize, alignment) : DispatchParamsSize;
  entry.params = block.offset;
  pending_batch(frame).entries.push_back(entry);
  return RecordResult::Recorded;
}

RecordResult record_copy(CopyTarget target, std::span<const std::byte> payload, Range staged,
                         uint32_t dstOffset) noexcept {
  AURORA_ASSERT(!render_worker::is_worker_thread(), "record_copy called on the render worker");
  constexpr std::string_view What = "record_copy";
  if (const RecordResult result = check_segment(What); result != RecordResult::Recorded) {
    return result;
  }
  FramePacket& frame = current_frame_packet();
  if (staged.size == 0 || bad_storage_range(frame, staged) || dstOffset % 4 != 0) {
    return refuse_record(What, RecordResult::BadRange);
  }
  AURORA_ASSERT(payload.size() <= ComputePayloadSize, "record_copy payload of {} bytes", payload.size());
  ComputeEntry entry{
      .kind = ComputeEntry::Kind::Copy,
      .range = staged,
      .dstOffset = dstOffset,
      .target = target,
  };
  std::memcpy(entry.payload.data(), payload.data(), payload.size());
  pending_batch(frame).entries.push_back(entry);
  return RecordResult::Recorded;
}

void note_segment_reads_gpu_state() noexcept {
  if (g_recorder.active()) {
    current_frame_packet().segmentReadsGpuState = true;
  }
}

namespace testing {
size_t pending_compute_entries() noexcept {
  if (!g_recorder.active()) {
    return 0;
  }
  const FramePacket& frame = current_frame_packet();
  return frame.pendingCompute < frame.computeBatches.size() ? frame.computeBatches[frame.pendingCompute].entries.size()
                                                            : 0;
}
} // namespace testing
} // namespace detail

uint32_t last_recorded_uniform_size() noexcept { return g_lastRecordedUniformSize; }

void insert_debug_marker(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  auto& markers = current_frame_packet().debugMarkers;
  const auto idx = markers.size();
  markers.emplace_back(std::move(label));
  push_command(CommandType::DebugMarker, {.debugMarkerIndex = idx});
#endif
}

void push_debug_group(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  g_recorder.debugGroupStack.push_back(std::move(label));
#endif
}
} // namespace aurora::gfx

namespace aurora::render {
DepthSource scene_depth_source() noexcept {
#if AURORA_ENABLE_RENDER
  if (webgpu::g_device) {
    return gfx::current_scene_depth_source();
  }
#endif
  return DepthSource::Unavailable;
}
} // namespace aurora::render

void push_debug_group(const char* label) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  aurora::gfx::g_recorder.debugGroupStack.emplace_back(label);
#endif
}
void pop_debug_group() {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (aurora::gfx::g_recorder.debugGroupStack.empty()) {
    aurora::gfx::Log.error("Debug group stack underflowed!");
    return;
  }

  aurora::gfx::g_recorder.debugGroupStack.pop_back();
#endif
}

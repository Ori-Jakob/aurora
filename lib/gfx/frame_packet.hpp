#pragma once

#include "pipeline_cache.hpp"
#include "recording.hpp"
#include "resources.hpp"
#include "types.hpp"
#include "tex_palette_conv.hpp"
#include "texture.hpp"
#include "../render/stage_data.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace aurora::gfx::detail {

struct StagingHighWater {
  uint32_t verts = 0;
  uint32_t uniforms = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
  uint32_t textureUpload = 0;
  size_t textureUploadCount = 0;
};

struct CustomDrawCommand {
  DrawTypeId type = 0;
  uint32_t payloadSize = 0;
  alignas(std::max_align_t) std::array<std::byte, InlineDrawPayloadSize> payload{};
};

struct RenderPass;
using DrawEncoder = void (*)(void* payload, const wgpu::RenderPassEncoder& pass, const RenderPass& passInfo);

struct DrawCommand {
  DrawEncoder encoder = nullptr;
  alignas(std::max_align_t) std::array<std::byte, InlineDrawPayloadSize> payload{};
};

enum class CommandType {
  SetViewport,
  SetScissor,
  Draw,
  CustomDraw,
  DebugMarker,
};

struct Command {
  CommandType type;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif
  union Data {
    Viewport setViewport;
    ClipRect setScissor;
    DrawCommand draw;
    CustomDrawCommand customDraw;
    size_t debugMarkerIndex;
  } data;
};

using CommandList = std::vector<Command>;

struct RenderPass {
  struct ColorAttachment {
    ColorAttachmentSemantic semantic = ColorAttachmentSemantic::Auxiliary;
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    wgpu::Extent3D size;
    wgpu::TextureView view;
    wgpu::TextureView resolveView;
    Vec4<float> clearValue{0.f, 0.f, 0.f, 0.f};
    wgpu::LoadOp loadOp = wgpu::LoadOp::Undefined;
    wgpu::StoreOp storeOp = wgpu::StoreOp::Store;
    bool clear = true;
  };

  std::string label;
  std::array<ColorAttachment, MaxColorAttachments> colorAttachments;
  uint32_t colorAttachmentCount = 0;
  wgpu::TextureView depthStencilView;
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  wgpu::Texture copySourceTexture;
  wgpu::TextureView copySourceView;
  wgpu::TextureView copySourceDepthView;
  wgpu::Texture copySourceNormalTexture;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  GXPixelFmt resolveSourceFormat = GX_PF_RGBA6_Z24;
  ClipRect resolveRect;
  Range resolveUniformRange;
  wgpu::Texture snapshotColorDst;
  wgpu::TextureView snapshotDepthDst;
  wgpu::Texture snapshotNormalDst;
  float clearDepthValue = 1.f;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
  CommandList commands;
  bool clearDepth = true;
  bool hasDepth = true;
  bool hasStencil = false;
  bool hasDraws = false;
  bool discardable = false;
  bool captureDepthSnapshot = false;
  bool sealed = false;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;

  RenderTargetLayout target_layout() const noexcept;
  bool has_consumer() const { return resolveTarget || snapshotColorDst || snapshotDepthDst || snapshotNormalDst; }
  bool has_content() const {
    if (hasDraws || clearDepth) {
      return true;
    }
    for (uint32_t i = 0; i < colorAttachmentCount; ++i) {
      if (colorAttachments[i].clear) {
        return true;
      }
    }
    return false;
  }
};

struct RenderTargetLayoutMemo {
  bool valid = false;
  uint32_t colorAttachmentCount = 0;
  std::array<ColorAttachmentSemantic, MaxColorAttachments> semantics{};
  std::array<wgpu::TextureFormat, MaxColorAttachments> formats{};
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  uint32_t sampleCount = 0;
  uint64_t key = 0;
};

inline void finalize_render_target_layout(RenderTargetLayout& layout) noexcept {
  // Every draw asks for its pass layout, so equal hashed fields reuse the last key
  thread_local RenderTargetLayoutMemo memo;
  bool same = memo.valid && memo.colorAttachmentCount == layout.colorAttachmentCount &&
              memo.depthStencilFormat == layout.depthStencilFormat && memo.sampleCount == layout.sampleCount;
  for (uint32_t i = 0; same && i < layout.colorAttachmentCount; ++i) {
    same = memo.semantics[i] == layout.colorAttachments[i].semantic &&
           memo.formats[i] == layout.colorAttachments[i].format;
  }
  if (same) {
    layout.key = memo.key;
    return;
  }

  Hasher hasher{};
  hasher.update(layout.colorAttachmentCount);
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    hasher.update(layout.colorAttachments[i].semantic);
    hasher.update(layout.colorAttachments[i].format);
  }
  hasher.update(static_cast<uint32_t>(layout.depthStencilFormat));
  hasher.update(layout.sampleCount);
  layout.key = hasher.digest();

  memo.valid = true;
  memo.colorAttachmentCount = layout.colorAttachmentCount;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    memo.semantics[i] = layout.colorAttachments[i].semantic;
    memo.formats[i] = layout.colorAttachments[i].format;
  }
  memo.depthStencilFormat = layout.depthStencilFormat;
  memo.sampleCount = layout.sampleCount;
  memo.key = layout.key;
}

inline RenderTargetLayout RenderPass::target_layout() const noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = colorAttachmentCount,
      .depthStencilFormat = hasDepth || hasStencil ? depthStencilFormat : wgpu::TextureFormat::Undefined,
      .sampleCount = msaaSamples,
  };
  for (uint32_t i = 0; i < colorAttachmentCount; ++i) {
    layout.colorAttachments[i] = {
        .semantic = colorAttachments[i].semantic,
        .format = colorAttachments[i].format,
        .width = colorAttachments[i].size.width,
        .height = colorAttachments[i].size.height,
    };
  }
  finalize_render_target_layout(layout);
  return layout;
}

struct TextureCopy {
  wgpu::TexelCopyTextureInfo src;
  wgpu::TexelCopyTextureInfo dst;
  wgpu::Extent3D size;
};

struct EncoderTask {
  EncoderTaskId type = InvalidEncoderTask;
  std::array<uint8_t, InlineDrawPayloadSize> payload{};
  uint32_t payloadSize = 0;
};

struct ComputeEntry {
  enum class Kind : uint8_t { Dispatch, Copy };
  Kind kind = Kind::Dispatch;
  Range range;
  uint32_t params = 0;
  uint32_t dstOffset = 0;
  std::array<uint32_t, 3> groups{};
  ComputeBinder bind = nullptr;
  CopyTarget target = nullptr;
  alignas(std::max_align_t) std::array<std::byte, ComputePayloadSize> payload{};
};

struct ComputeBatch {
  std::vector<ComputeEntry> entries;
};

enum class FrameOpType : uint8_t {
  RenderPass,
  TextureCopy,
  EncoderTask,
  RenderStage,
  Compute,
};

struct FrameOp {
  FrameOpType type = FrameOpType::RenderPass;
  uint32_t index = 0;
  RenderPass* renderPass = nullptr;
  TextureCopy* textureCopy = nullptr;
  EncoderTask* encoderTask = nullptr;
  const render::detail::StageData* stageData = nullptr;
  const ComputeBatch* computeBatch = nullptr;
  StagingHighWater highWater;
  std::vector<const TextureUpload*> textureUploads;
};

inline bool op_has_payload(const FrameOp& op) noexcept {
  switch (op.type) {
  case FrameOpType::RenderPass:
    return op.renderPass != nullptr;
  case FrameOpType::TextureCopy:
    return op.textureCopy != nullptr;
  case FrameOpType::EncoderTask:
    return op.encoderTask != nullptr;
  case FrameOpType::RenderStage:
    return op.stageData != nullptr;
  case FrameOpType::Compute:
    return op.computeBatch != nullptr;
  }
  return false;
}

using RenderPassList = std::deque<RenderPass>;

struct ResidentCopy {
  uint32_t src;
  uint32_t dst;
  uint32_t copySize;
  uint32_t blockSize;
};

struct FramePacket {
  RenderPassList renderPasses;
  std::deque<TextureCopy> textureCopies;
  std::deque<EncoderTask> encoderTasks;
  std::deque<render::detail::StageData> stages;
  std::deque<ComputeBatch> computeBatches;
  std::deque<FrameOp> ops;
  std::deque<TextureUpload> textureUploads;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugMarkers;
#endif
  ByteBuffer verts;
  ByteBuffer uniforms;
  ByteBuffer indices;
  ByteBuffer storage;
  ByteBuffer textureUpload;
  wgpu::CommandEncoder encoder;
  std::vector<AfterSubmitCallback> afterSubmitCallbacks;
  std::vector<ResidentCopy> residentCopies;
  std::shared_ptr<const render::detail::FrameState> renderState;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  size_t stagingBuffer = 0;
  StreamingSizes sizes;
  uint32_t residentCapacity = 0;
  uint32_t residentLimit = 0;
  uint32_t residentEpoch = 0;
  uint32_t pendingCompute = UINT32_MAX;
  uint32_t computeParamsBytes = 0;
  bool segmentReadsGpuState = false;
  StagingHighWater copied;
  AuroraStats stats{};
};

} // namespace aurora::gfx::detail

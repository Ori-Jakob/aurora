#include "encoding.hpp"

#include "frame.hpp"

#include "clear.hpp"
#include "depth_peek.hpp"
#include "pipeline_cache.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#include "../render/stage.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;
using webgpu::g_device;
using webgpu::g_queue;

namespace {
constexpr Module Log{"aurora::gfx"};
WGPURenderPipeline g_currentPipeline = nullptr;

void apply_viewport(const wgpu::RenderPassEncoder& pass, const Viewport& vp) {
  const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
  const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
  pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
}

void apply_scissor(const wgpu::RenderPassEncoder& pass, const ClipRect& sc, const wgpu::Extent3D& size) {
  const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
  const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
  const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
  const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
  pass.SetScissorRect(x, y, w, h);
}

DrawContext make_draw_context(const RenderPass& passInfo) {
  auto& res = resources();
  return {
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
      .layout = passInfo.target_layout(),
  };
}

void render_custom_draw(const CustomDrawCommand& draw, const wgpu::RenderPassEncoder& pass,
                        const RenderPass& passInfo) {
  const auto drawType = find_runtime_draw_type(draw.type);
  if (!drawType) {
    // Unregistered between record and replay; the command is a no-op.
    return;
  }

  const auto context = make_draw_context(passInfo);
  drawType->draw(context, pass, draw.payload.data(), draw.payloadSize, drawType->userdata);
}

void execute_encoder_task(wgpu::CommandEncoder& cmd, FramePacket& frame, const EncoderTask& task) {
  const auto taskType = find_runtime_encoder_task_type(task.type);
  if (!taskType) {
    // Unregistered between record and encode; the task is a no-op.
    return;
  }

  auto& res = resources();
  const EncoderTaskContext context{
      .device = g_device,
      .queue = g_queue,
      .vertexBuffer = res.vertexBuffer,
      .indexBuffer = res.indexBuffer,
      .uniformBuffer = res.uniformBuffer,
      .storageBuffer = res.storageBuffer,
  };
  taskType->callback(context, cmd, task.payload.data(), task.payloadSize, taskType->userdata);
  if (taskType->afterSubmit != nullptr) {
    const auto payload = task.payload;
    const auto payloadSize = task.payloadSize;
    const auto callback = taskType->afterSubmit;
    const auto userdata = taskType->userdata;
    frame.afterSubmitCallbacks.emplace_back([payload, payloadSize, callback, userdata] {
      const EncoderTaskCompletionContext completionContext{
          .device = g_device,
          .queue = g_queue,
      };
      callback(completionContext, payload.data(), payloadSize, userdata);
    });
  }
}

void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, RenderPass& passInfo) {
  ZoneScoped;
  const auto encodeStart = std::chrono::steady_clock::now();
  DrawEncoder lastEncoder = nullptr;
  // Everything the encoders skip as already bound must be bound again once other code may have changed it
  const auto reset_encoder_state = [&] {
    g_currentPipeline = nullptr;
    lastEncoder = nullptr;
    gx::reset_pass_bindings();
  };
  reset_encoder_state();
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> lastDebugGroupStack;
#endif
  Viewport currentViewport{};
  ClipRect currentScissor{};
  bool hasViewport = false;
  bool hasScissor = false;

  // Bind bind group for the whole pass
  pass.SetBindGroup(0, resources().staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);

  for (auto& cmd : passInfo.commands) {
#ifdef AURORA_GFX_DEBUG_GROUPS
    {
      size_t firstDiff = lastDebugGroupStack.size();
      for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
        if (i >= cmd.debugGroupStack.size() || cmd.debugGroupStack[i] != lastDebugGroupStack[i]) {
          firstDiff = i;
          break;
        }
      }
      for (size_t i = firstDiff; i < lastDebugGroupStack.size(); ++i) {
        pass.PopDebugGroup();
      }
      for (size_t i = firstDiff; i < cmd.debugGroupStack.size(); ++i) {
        pass.PushDebugGroup(cmd.debugGroupStack[i].c_str());
      }
      lastDebugGroupStack = cmd.debugGroupStack;
    }
#endif
    switch (cmd.type) {
    case CommandType::SetViewport: {
      const auto& vp = cmd.data.setViewport;
      apply_viewport(pass, vp);
      currentViewport = vp;
      hasViewport = true;
    } break;
    case CommandType::SetScissor: {
      const auto& sc = cmd.data.setScissor;
      apply_scissor(pass, sc, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      currentScissor = sc;
      hasScissor = true;
    } break;
    case CommandType::Draw: {
      auto& draw = cmd.data.draw;
      if (draw.encoder != nullptr) {
        if (draw.encoder != lastEncoder) {
          gx::reset_pass_bindings();
          lastEncoder = draw.encoder;
        }
        draw.encoder(draw.payload.data(), pass, passInfo);
      }
    } break;
    case CommandType::CustomDraw: {
      render_custom_draw(cmd.data.customDraw, pass, passInfo);
      reset_encoder_state();
      pass.SetBindGroup(0, resources().staticBindGroup);
      pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);
      if (hasViewport) {
        apply_viewport(pass, currentViewport);
      }
      if (hasScissor) {
        apply_scissor(pass, currentScissor, passInfo.colorAttachments[SceneColorAttachmentIndex].size);
      }
    } break;
    case CommandType::DebugMarker: {
#if defined(AURORA_GFX_DEBUG_GROUPS)
      pass.InsertDebugMarker(wgpu::StringView(frame.debugMarkers[cmd.data.debugMarkerIndex]));
#endif
    } break;
    }
  }

#ifdef AURORA_GFX_DEBUG_GROUPS
  for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
    pass.PopDebugGroup();
  }
#endif
  gx::add_encode_ns(static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - encodeStart).count()));
}

void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex) {
  ZoneScoped;
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(cmd, conv);
  }
  if (passInfo.discardable) {
    // This pass has no effect and can be safely discarded (e.g. an empty EFB segment between two back-to-back pass
    // breaks, or an unresolved offscreen pass).
    return;
  }

  std::array<wgpu::RenderPassColorAttachment, MaxColorAttachments> attachments{};
  for (uint32_t i = 0; i < passInfo.colorAttachmentCount; ++i) {
    const auto& source = passInfo.colorAttachments[i];
    attachments[i] = {
        .view = source.view,
        .resolveTarget = source.resolveView,
        .loadOp = source.loadOp != wgpu::LoadOp::Undefined ? source.loadOp
                                                           : (source.clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load),
        .storeOp = source.storeOp,
        .clearValue =
            {
                .r = source.clearValue.x(),
                .g = source.clearValue.y(),
                .b = source.clearValue.z(),
                .a = source.clearValue.w(),
            },
    };
  }
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{};
  const wgpu::RenderPassDepthStencilAttachment* depthStencilAttachmentPtr = nullptr;
  if (passInfo.depthStencilView) {
    depthStencilAttachment = {
        .view = passInfo.depthStencilView,
        .depthLoadOp = passInfo.hasDepth ? (passInfo.depthLoadOp != wgpu::LoadOp::Undefined
                                                ? passInfo.depthLoadOp
                                                : (passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load))
                                         : wgpu::LoadOp::Undefined,
        .depthStoreOp = passInfo.hasDepth ? passInfo.depthStoreOp : wgpu::StoreOp::Undefined,
        .depthClearValue = passInfo.clearDepthValue,
        .stencilLoadOp = passInfo.hasStencil ? passInfo.stencilLoadOp : wgpu::LoadOp::Undefined,
        .stencilStoreOp = passInfo.hasStencil ? passInfo.stencilStoreOp : wgpu::StoreOp::Undefined,
        .stencilClearValue = passInfo.stencilClearValue,
    };
    depthStencilAttachmentPtr = &depthStencilAttachment;
  }
  const auto label = passInfo.label.empty() ? fmt::format("Render pass {}", passIndex)
                                            : fmt::format("{} {}", passInfo.label, passIndex);
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = label.c_str(),
      .colorAttachmentCount = passInfo.colorAttachmentCount,
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };

  auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  render_pass(pass, frame, passInfo);
  pass.End();

  if (passInfo.captureDepthSnapshot) {
    depth_peek::encode_frame_snapshot(cmd, passInfo.copySourceDepthView,
                                      passInfo.colorAttachments[SceneColorAttachmentIndex].size, passInfo.msaaSamples);
  }

  if (passInfo.resolveTarget) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    if (isDepth && passInfo.msaaSamples > 1) {
      Log.fatal("Depth tex copies from multisampled EFB targets are not supported");
    }
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcFmt = passInfo.resolveSourceFormat,
        .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
    };
    const bool needsOpaqueAlpha = !isDepth && !gx::efb_has_alpha(convReq.srcFmt);
    const bool sameFormat =
        passInfo.resolveTarget->format == passInfo.colorAttachments[SceneColorAttachmentIndex].format;
    if (needsConversion) {
      tex_copy_conv::run(cmd, convReq);
    } else if (needsScaling || needsOpaqueAlpha || !sameFormat) {
      tex_copy_conv::blit(cmd, convReq);
    } else {
      const webgpu::gpu_prof::Zone zone{cmd, "EFB copy"};
      const wgpu::TexelCopyTextureInfo src{
          .texture = passInfo.copySourceTexture,
          .origin =
              wgpu::Origin3D{
                  .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                  .y = static_cast<uint32_t>(passInfo.resolveRect.y),
              },
      };
      const wgpu::TexelCopyTextureInfo dst{
          .texture = passInfo.resolveTarget->texture,
      };
      const wgpu::Extent3D size{
          .width = static_cast<uint32_t>(passInfo.resolveRect.width),
          .height = static_cast<uint32_t>(passInfo.resolveRect.height),
          .depthOrArrayLayers = 1,
      };
      cmd.CopyTextureToTexture(&src, &dst, &size);
    }
  }

  if (passInfo.snapshotColorDst) {
    const webgpu::gpu_prof::Zone zone{cmd, "Pass snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = passInfo.copySourceTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotColorDst,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
  if (passInfo.snapshotDepthDst) {
    tex_copy_conv::snapshot_depth(cmd, passInfo.copySourceDepthView, passInfo.msaaSamples, passInfo.snapshotDepthDst);
  }
  if (passInfo.snapshotNormalDst) {
    const webgpu::gpu_prof::Zone zone{cmd, "Normal snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = passInfo.copySourceNormalTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotNormalDst,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
  if (passInfo.snapshotFogDst) {
    const webgpu::gpu_prof::Zone zone{cmd, "Fog snapshot"};
    const wgpu::TexelCopyTextureInfo src{
        .texture = passInfo.copySourceFogTexture,
    };
    const wgpu::TexelCopyTextureInfo dst{
        .texture = passInfo.snapshotFogDst,
    };
    const wgpu::Extent3D size{
        .width = passInfo.colorAttachments[SceneColorAttachmentIndex].size.width,
        .height = passInfo.colorAttachments[SceneColorAttachmentIndex].size.height,
        .depthOrArrayLayers = 1,
    };
    cmd.CopyTextureToTexture(&src, &dst, &size);
  }
}

constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~3u; }

void copy_staging_buffer_range(wgpu::CommandEncoder& cmd, const FramePacket& frame, uint32_t& copied,
                               uint32_t highWater, uint64_t stagingOffset, const wgpu::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd = AURORA_ALIGN(highWater, 4);
  cmd.CopyBufferToBuffer(staging_buffer(frame.stagingBuffer), stagingOffset + copyStart, dst, copyStart,
                         copyEnd - copyStart);
  copied = highWater;
}

bool needs_staging_copy(const FramePacket& frame, const FrameOp& op) {
  const auto& highWater = op.highWater;
  if (highWater.verts > frame.copied.verts || highWater.uniforms > frame.copied.uniforms ||
      highWater.indices > frame.copied.indices || highWater.storage > frame.copied.storage) {
    return true;
  }
  if constexpr (UseTextureBuffer) {
    return op.textureUploads.size() > frame.copied.textureUploadCount;
  }
  return false;
}

void copy_staging_to_high_water(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "Staging copies"};
  const auto& highWater = op.highWater;
  auto& res = resources();
  const auto layout = staging_layout(frame.sizes);
  copy_staging_buffer_range(cmd, frame, frame.copied.verts, highWater.verts, layout.vertex, res.vertexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.uniforms, highWater.uniforms, layout.uniform, res.uniformBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.indices, highWater.indices, layout.index, res.indexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.storage, highWater.storage, layout.storage, res.storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      const wgpu::TexelCopyBufferInfo buf{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .offset = item.buffer ? item.layout.offset : item.layout.offset + layout.textureUpload,
                  .bytesPerRow = AURORA_ALIGN(item.layout.bytesPerRow, 256),
                  .rowsPerImage = item.layout.rowsPerImage,
              },
          .buffer = item.buffer ? item.buffer : staging_buffer(frame.stagingBuffer),
      };
      cmd.CopyBufferToTexture(&buf, &item.tex, &item.size);
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}

struct StreamGroup {
  wgpu::Buffer storage;
  uint64_t storageSize = 0;
  wgpu::Buffer uniform;
  wgpu::BindGroup group;
};
StreamGroup g_streamGroup;
uint32_t g_computeSegments = 0;
wgpu::Buffer g_computeScratch;
std::atomic_uint64_t g_binderFailures{0};
std::atomic_uint64_t g_droppedCopies{0};
bool g_warnedBinder = false;
bool g_warnedCopy = false;

const wgpu::BindGroup& stream_group(const FramePacket& frame) {
  const auto& res = resources();
  auto& cached = g_streamGroup;
  if (cached.group && cached.storage.Get() == res.storageBuffer.Get() && cached.storageSize == frame.sizes.storage &&
      cached.uniform.Get() == res.uniformBuffer.Get()) {
    return cached.group;
  }
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = res.storageBuffer, .offset = 0, .size = frame.sizes.storage},
      wgpu::BindGroupEntry{.binding = 1, .buffer = res.uniformBuffer, .offset = 0, .size = DispatchParamsSize},
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "GPU systems stream bind group",
      .layout = compute_stream_layout(),
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  cached = {
      .storage = res.storageBuffer,
      .storageSize = frame.sizes.storage,
      .uniform = res.uniformBuffer,
      .group = g_device.CreateBindGroup(&descriptor),
  };
  return cached.group;
}

bool zero_groups(const ComputeEntry& entry) noexcept {
  return std::ranges::find(entry.groups, 0u) != entry.groups.end();
}

void count_failure(std::atomic_uint64_t& counter, bool& warned, std::string_view message) noexcept {
  counter.fetch_add(1, std::memory_order_relaxed);
  if (!std::exchange(warned, true)) {
    Log.warn("{}", message);
  }
}

void encode_compute(wgpu::CommandEncoder& cmd, FramePacket& frame, const ComputeBatch& batch) {
  ZoneScoped;
  const auto& res = resources();
  bool dispatches = false;
  for (const auto& entry : batch.entries) {
    if (entry.kind == ComputeEntry::Kind::Dispatch && entry.range.size > 0) {
      cmd.ClearBuffer(res.storageBuffer, entry.range.offset, entry.range.size);
    }
    dispatches = dispatches || entry.kind == ComputeEntry::Kind::Dispatch;
  }
  // Stateless kernels share the scratch as their state binding, so nothing they write outlives the op
  if (dispatches && g_computeScratch) {
    cmd.ClearBuffer(g_computeScratch, 0, g_computeScratch.GetSize());
  }
  wgpu::ComputePassEncoder pass;
  const auto end_pass = [&] {
    if (pass) {
      pass.End();
      pass = nullptr;
    }
  };
  for (const auto& entry : batch.entries) {
    if (entry.kind == ComputeEntry::Kind::Copy) {
      end_pass();
      const CopyDestination dst = entry.target(entry.payload.data());
      if (dst.buffer && uint64_t{entry.dstOffset} + entry.range.size <= dst.size &&
          dst.base + dst.size <= dst.buffer.GetSize()) {
        cmd.CopyBufferToBuffer(staging_buffer(frame.stagingBuffer),
                               staging_layout(frame.sizes).storage + entry.range.offset, dst.buffer,
                               dst.base + entry.dstOffset, entry.range.size);
      } else {
        count_failure(g_droppedCopies, g_warnedCopy, "GPU system state copy dropped: its destination is not placed");
      }
      continue;
    }
    if (zero_groups(entry)) {
      continue;
    }
    if (!pass) {
      constexpr std::string_view Label = "GPU systems";
      const wgpu::ComputePassDescriptor descriptor{
          .label = Label,
          .timestampWrites = webgpu::gpu_prof::pass_writes(Label),
      };
      pass = cmd.BeginComputePass(&descriptor);
      pass.SetBindGroup(0, compute_empty_group());
      ++g_computeSegments;
    }
    const uint32_t offset = entry.params;
    pass.SetBindGroup(1, stream_group(frame), 1, &offset);
    if (entry.bind(entry.payload.data(), pass)) {
      pass.DispatchWorkgroups(entry.groups[0], entry.groups[1], entry.groups[2]);
    } else {
      count_failure(g_binderFailures, g_warnedBinder,
                    "GPU dispatch skipped: its kernel or state could not be bound, so its output stays zero");
    }
  }
  end_pass();
}
} // namespace

namespace detail {
void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  copy_staging_to_high_water(cmd, frame, op);
  switch (op.type) {
  case FrameOpType::RenderPass:
    if (op.renderPass != nullptr) {
      render(cmd, frame, *op.renderPass, op.index);
    }
    break;
  case FrameOpType::TextureCopy:
    if (op.textureCopy != nullptr) {
      const webgpu::gpu_prof::Zone zone{cmd, "Texture copy"};
      cmd.CopyTextureToTexture(&op.textureCopy->src, &op.textureCopy->dst, &op.textureCopy->size);
    }
    break;
  case FrameOpType::EncoderTask:
    if (op.encoderTask != nullptr) {
      execute_encoder_task(cmd, frame, *op.encoderTask);
    }
    break;
  case FrameOpType::RenderStage:
    if (op.stageData != nullptr) {
      ::aurora::render::detail::run_stage(cmd, *op.stageData);
    }
    break;
  case FrameOpType::Compute:
    if (op.computeBatch != nullptr) {
      encode_compute(cmd, frame, *op.computeBatch);
    }
    break;
  }
}

void release_compute_encoding() noexcept {
  g_streamGroup = {};
  g_warnedBinder = false;
  g_warnedCopy = false;
}

void set_compute_scratch(wgpu::Buffer buffer) noexcept { g_computeScratch = std::move(buffer); }

ComputeFailures compute_failures() noexcept {
  return {
      .binds = g_binderFailures.load(std::memory_order_relaxed),
      .copies = g_droppedCopies.load(std::memory_order_relaxed),
  };
}

namespace testing {
uint32_t compute_segments_encoded() noexcept { return g_computeSegments; }
} // namespace testing
} // namespace detail

void bind_pipeline(const wgpu::RenderPipeline& pipeline, const wgpu::RenderPassEncoder& pass) {
  if (pipeline.Get() == g_currentPipeline) {
    return;
  }
  pass.SetPipeline(pipeline);
  g_currentPipeline = pipeline.Get();
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) {
  CompiledPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    return false;
  }
  AURORA_ASSERT(!pipeline.prepass, "Prepass pipelines require individual binding");
  bind_pipeline(pipeline.main, pass);
  return true;
}
} // namespace aurora::gfx

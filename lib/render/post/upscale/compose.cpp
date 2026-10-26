#include "compose.hpp"

#include "upscale.hpp"
#include "../../hints.hpp"
#include "../../debug/capture.hpp"

#include "../../../gfx/frame_packet.hpp"
#include "../../../gfx/recording.hpp"
#include "../../../gfx/resources.hpp"
#include "../../../internal.hpp"
#include "../../../logging.hpp"
#include "../../../webgpu/gpu.hpp"
#include "../../../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace aurora::render::detail::upscale::compose {
namespace {
constexpr Module Log{"aurora::render::upscale"};
constexpr std::array<std::string_view, 1> InplaceDefines{"COMPOSE_INPLACE"};
constexpr uint64_t SplitLayoutKey = 0x55505343'53504C54ull;
constexpr uint64_t InplaceLayoutKey = 0x55505343'494E504Cull;
constexpr size_t GroupCacheCapacity = 4;
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

struct PipelineEntry {
  ComposeMode mode = ComposeMode::Split;
  ColorEncoding encoding = ColorEncoding::Stored;
  wgpu::TextureFormat color = wgpu::TextureFormat::Undefined;
  wgpu::TextureFormat depth = wgpu::TextureFormat::Undefined;
  shaders::PipelineId id;
};

struct Shared {
  wgpu::BindGroupLayout splitLayout;
  wgpu::BindGroupLayout inplaceLayout;
  wgpu::PipelineLayout splitPipelineLayout;
  wgpu::PipelineLayout inplacePipelineLayout;
  wgpu::Sampler sampler;
  wgpu::Buffer statusOff;
  wgpu::Buffer statusOn;
  wgpu::Buffer statusNative;
  wgpu::Texture dummy;
  wgpu::TextureView dummyView;
};

struct GroupEntry {
  wgpu::BindGroupLayout layout;
  wgpu::Buffer uniforms;
  wgpu::TextureView sceneColor;
  wgpu::TextureView sceneDepth;
  wgpu::TextureView source;
  wgpu::Buffer status;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

std::mutex g_mutex;
Shared g_shared;
std::vector<PipelineEntry> g_pipelines;
std::atomic_bool g_hold{false};

struct RequestedStep {
  gfx::detail::NativeStep step;
  Source out;
};

// Render worker only
std::optional<Source> g_source;
std::optional<RequestedStep> g_requested;
std::optional<Source> g_nativeOut;
std::vector<GroupEntry> g_groups;
uint64_t g_encodes = 0;
bool g_warnedSkip = false;
bool g_warnedStaleStep = false;

std::atomic_uint32_t g_encoded{0};
std::atomic_uint32_t g_sourced{0};
std::atomic_uint32_t g_skipped{0};
std::atomic_uint32_t g_nativeBound{0};
std::atomic_uint32_t g_statusWrites{0};
std::atomic_uint32_t g_lastStatus{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

wgpu::BindGroupLayoutEntry uniform_entry(uint32_t binding, uint64_t size, bool dynamic) {
  return {
      .binding = binding,
      .visibility = wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = dynamic, .minBindingSize = size},
  };
}

wgpu::BindGroupLayoutEntry texture_entry(uint32_t binding, wgpu::TextureSampleType type) {
  return {
      .binding = binding,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = type, .viewDimension = wgpu::TextureViewDimension::e2D},
  };
}

wgpu::Buffer status_buffer(const wgpu::Device& device, uint32_t value, const char* label) {
  const wgpu::BufferDescriptor descriptor{
      .label = label,
      .usage = wgpu::BufferUsage::Uniform,
      .size = 16,
      .mappedAtCreation = true,
  };
  wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
  const std::array<uint32_t, 4> words{value, 0, 0, 0};
  std::memcpy(buffer.GetMappedRange(0, sizeof(words)), words.data(), sizeof(words));
  buffer.Unmap();
  return buffer;
}

bool ensure_shared_locked() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return false;
  }
  if (g_shared.splitLayout) {
    return true;
  }
  const std::array splitEntries{
      uniform_entry(0, sizeof(Params), true),
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      },
      texture_entry(2, wgpu::TextureSampleType::Float),
      texture_entry(3, wgpu::TextureSampleType::Depth),
      texture_entry(4, wgpu::TextureSampleType::Float),
      uniform_entry(5, 16, false),
  };
  const std::array inplaceEntries{
      uniform_entry(0, sizeof(Params), true),
      texture_entry(4, wgpu::TextureSampleType::Float),
      uniform_entry(5, 16, false),
  };
  const wgpu::BindGroupLayoutDescriptor splitDescriptor{
      .label = "upscale.compose split",
      .entryCount = splitEntries.size(),
      .entries = splitEntries.data(),
  };
  const wgpu::BindGroupLayoutDescriptor inplaceDescriptor{
      .label = "upscale.compose in place",
      .entryCount = inplaceEntries.size(),
      .entries = inplaceEntries.data(),
  };
  Shared shared;
  shared.splitLayout = device.CreateBindGroupLayout(&splitDescriptor);
  shared.inplaceLayout = device.CreateBindGroupLayout(&inplaceDescriptor);
  const wgpu::PipelineLayoutDescriptor splitPipeline{
      .label = "upscale.compose split",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &shared.splitLayout,
  };
  const wgpu::PipelineLayoutDescriptor inplacePipeline{
      .label = "upscale.compose in place",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &shared.inplaceLayout,
  };
  shared.splitPipelineLayout = device.CreatePipelineLayout(&splitPipeline);
  shared.inplacePipelineLayout = device.CreatePipelineLayout(&inplacePipeline);
  const wgpu::SamplerDescriptor sampler{
      .label = "upscale.compose linear",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .addressModeW = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  shared.sampler = device.CreateSampler(&sampler);
  shared.statusOff = status_buffer(device, 0, "upscale.compose status 0");
  shared.statusOn = status_buffer(device, 1, "upscale.compose status 1");
  const wgpu::BufferDescriptor nativeStatus{
      .label = "upscale.compose native status",
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = 16,
  };
  shared.statusNative = device.CreateBuffer(&nativeStatus);
  const wgpu::TextureDescriptor dummy{
      .label = "upscale.compose empty source",
      .usage = wgpu::TextureUsage::TextureBinding,
      .size = {1, 1, 1},
      .format = wgpu::TextureFormat::RGBA8Unorm,
  };
  shared.dummy = device.CreateTexture(&dummy);
  shared.dummyView = shared.dummy.CreateView();
  g_shared = std::move(shared);
  return true;
}

wgpu::RenderPipeline find_pipeline(ComposeMode mode, ColorEncoding encoding, wgpu::TextureFormat color,
                                   wgpu::TextureFormat depth, Shared& shared) {
  shaders::PipelineId id;
  {
    std::lock_guard lock{g_mutex};
    if (!ensure_shared_locked()) {
      return {};
    }
    shared = g_shared;
    for (const PipelineEntry& entry : g_pipelines) {
      if (entry.mode == mode && entry.encoding == encoding && entry.color == color && entry.depth == depth) {
        id = entry.id;
        break;
      }
    }
  }
  return id ? shaders::render_pipeline(id) : wgpu::RenderPipeline{};
}

wgpu::BindGroup cached_group(const wgpu::Device& device, const wgpu::BindGroupLayout& layout, bool split,
                             const wgpu::Buffer& uniforms, const Targets& targets, const wgpu::TextureView& source,
                             const wgpu::Buffer& status, const wgpu::Sampler& sampler) {
  for (GroupEntry& entry : g_groups) {
    if (entry.layout.Get() == layout.Get() && entry.uniforms.Get() == uniforms.Get() &&
        entry.sceneColor.Get() == targets.sceneColor.Get() && entry.sceneDepth.Get() == targets.sceneDepth.Get() &&
        entry.source.Get() == source.Get() && entry.status.Get() == status.Get()) {
      entry.lastUsed = g_encodes;
      return entry.group;
    }
  }
  if (g_groups.size() >= GroupCacheCapacity) {
    g_groups.erase(std::ranges::min_element(g_groups, {}, &GroupEntry::lastUsed));
  }
  std::vector<wgpu::BindGroupEntry> entries{
      {.binding = 0, .buffer = uniforms, .size = sizeof(Params)},
      {.binding = 4, .textureView = source},
      {.binding = 5, .buffer = status, .size = 16},
  };
  if (split) {
    entries.push_back({.binding = 1, .sampler = sampler});
    entries.push_back({.binding = 2, .textureView = targets.sceneColor});
    entries.push_back({.binding = 3, .textureView = targets.sceneDepth});
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = "upscale.compose",
      .layout = layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  GroupEntry& entry = g_groups.emplace_back(GroupEntry{
      .layout = layout,
      .uniforms = uniforms,
      .sceneColor = targets.sceneColor,
      .sceneDepth = targets.sceneDepth,
      .source = source,
      .status = status,
      .group = device.CreateBindGroup(&descriptor),
      .lastUsed = g_encodes,
  });
  return entry.group;
}

bool encode_targets(wgpu::CommandEncoder& cmd, const SceneComposeData& data, const Targets& targets,
                    const std::optional<Source>& source, bool native) {
  ++g_encodes;
  const bool split = data.mode == ComposeMode::Split;
  const bool sourced = source.has_value() && static_cast<bool>(source->view);
  const ColorEncoding encoding = sourced ? source->encoding : ColorEncoding::Stored;
  Shared shared;
  const wgpu::RenderPipeline pipeline = find_pipeline(
      data.mode, encoding, targets.colorFormat, split ? targets.depthFormat : wgpu::TextureFormat::Undefined, shared);
  const wgpu::Buffer& uniforms = gfx::detail::resources().uniformBuffer;
  const bool targetsReady = targets.color && (!split || (targets.sceneColor && targets.sceneDepth && targets.depth));
  if (!pipeline || !uniforms || !targetsReady) {
    bump(g_skipped);
    if (!std::exchange(g_warnedSkip, true)) {
      Log.warn("Scene compose skipped: its pipeline or targets are missing, so the output keeps stale colour");
    }
    return false;
  }
  const wgpu::TextureView sourceView = sourced ? source->view : (split ? targets.sceneColor : shared.dummyView);
  const bool nativeStatus = native && sourced;
  const wgpu::Buffer& status = nativeStatus ? shared.statusNative : (sourced ? shared.statusOn : shared.statusOff);
  const wgpu::BindGroup group = cached_group(webgpu::g_device, split ? shared.splitLayout : shared.inplaceLayout, split,
                                             uniforms, targets, sourceView, status, shared.sampler);
  const wgpu::RenderPassColorAttachment color{
      .view = targets.color,
      .loadOp = split ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDepthStencilAttachment depth{
      .view = targets.depth,
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 0.f,
  };
  const wgpu::RenderPassDescriptor descriptor{
      .label = "upscale.compose",
      .colorAttachmentCount = 1,
      .colorAttachments = &color,
      .depthStencilAttachment = split ? &depth : nullptr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(Label),
  };
  const wgpu::RenderPassEncoder pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group, 1, &data.uniformOffset);
  pass.Draw(3);
  pass.End();
  if (targets.colorTexture) {
    debug::capture::after_scene_compose(cmd, targets.colorTexture, data.frameId);
  }
  bump(g_encoded);
  if (sourced) {
    bump(g_sourced);
  }
  if (nativeStatus) {
    bump(g_nativeBound);
  }
  return true;
}
} // namespace

Params make_params(const SceneComposeData& data, GraphExtent scene, GraphExtent output) noexcept {
  const auto scale = [](uint32_t from, uint32_t to) {
    return to != 0 ? static_cast<float>(from) / static_cast<float>(to) : 1.f;
  };
  return {
      .scale = {scale(scene.width, output.width), scale(scene.height, output.height)},
      .jitter = {data.jitterPx[0], data.jitterPx[1]},
      .sceneSize = {static_cast<float>(scene.width), static_cast<float>(scene.height)},
      .outputSize = {static_cast<float>(output.width), static_cast<float>(output.height)},
      .rcasStrength = data.rcasStrength,
  };
}

bool prepare_pipelines(ComposeMode mode, wgpu::TextureFormat colorFormat, wgpu::TextureFormat depthFormat) {
  const wgpu::TextureFormat depth = mode == ComposeMode::Split ? depthFormat : wgpu::TextureFormat::Undefined;
  shaders::PipelineId id;
  {
    std::lock_guard lock{g_mutex};
    if (!ensure_shared_locked()) {
      return false;
    }
    for (const PipelineEntry& entry : g_pipelines) {
      if (entry.mode == mode && entry.encoding == ColorEncoding::Stored && entry.color == colorFormat &&
          entry.depth == depth) {
        id = entry.id;
        break;
      }
    }
    if (!id) {
      shaders::RenderPipelineDesc desc = testing::pipeline_desc(mode, colorFormat, depth);
      const bool split = mode == ComposeMode::Split;
      desc.layout = split ? g_shared.splitPipelineLayout : g_shared.inplacePipelineLayout;
      desc.layoutKey = split ? SplitLayoutKey : InplaceLayoutKey;
      id = shaders::request_render(desc);
      g_pipelines.push_back({.mode = mode, .color = colorFormat, .depth = depth, .id = id});
    }
  }
  return !g_hold.load(std::memory_order_relaxed) && shaders::ready(id);
}

void prepare(SceneComposeData& data) {
  const GraphExtent scene = scene_size();
  const GraphExtent output = output_size();
  data.mode = split_live() ? ComposeMode::Split : ComposeMode::InPlace;
  data.ratio[0] = scene.width != 0 ? static_cast<float>(output.width) / static_cast<float>(scene.width) : 1.f;
  data.ratio[1] = scene.height != 0 ? static_cast<float>(output.height) / static_cast<float>(scene.height) : 1.f;
  data.rcasStrength = rcas_strength();
  if (data.mode == ComposeMode::Split) {
    const std::shared_ptr<const SceneHints> frozen = scene_hints_frozen() ? scene_hints() : nullptr;
    const SceneHints& hints = frozen != nullptr ? *frozen : live_scene_hints();
    data.jitterPx[0] = hints.jitterPx[0];
    data.jitterPx[1] = hints.jitterPx[1];
  }
  const Params params = make_params(data, scene, output);
  data.uniformOffset = gfx::push_uniform(params).offset;
}

void encode(wgpu::CommandEncoder& cmd, const SceneComposeData& data) {
  const std::optional<Source> published = std::exchange(g_source, std::nullopt);
  const std::optional<Source> native = std::exchange(g_nativeOut, std::nullopt);
  Targets targets;
  if (data.mode == ComposeMode::Split) {
    targets = {
        .sceneColor = webgpu::scene_color().view,
        .sceneDepth = webgpu::scene_depth().view,
        .colorTexture = webgpu::g_frameBuffer.texture,
        .color = webgpu::g_frameBuffer.view,
        .colorFormat = webgpu::g_frameBuffer.format,
        .depth = webgpu::g_depthBuffer.view,
        .depthFormat = webgpu::g_graphicsConfig.depthFormat,
    };
  } else {
    const webgpu::TextureWithSampler& color = webgpu::scene_color();
    targets = {.colorTexture = color.texture, .color = color.view, .colorFormat = color.format};
  }
  encode_targets(cmd, data, targets, native ? native : published, native.has_value());
}

void publish_source(const Source& source) noexcept { g_source = source; }

void request_native_step(const gfx::detail::NativeStep& step, const Source& out) noexcept {
  const bool complete = step.begin_inputs != nullptr && step.run != nullptr && step.begin_output != nullptr &&
                        step.end_output != nullptr && out.view;
  CHECK(complete, "Native step {} is missing a callback or its output", step.name);
  if (complete) {
    g_requested = RequestedStep{.step = step, .out = out};
  }
}

bool take_native_step(gfx::detail::NativeStep& step) noexcept {
  if (!g_requested) {
    return false;
  }
  step = g_requested->step;
  g_nativeOut = g_requested->out;
  g_requested.reset();
  return true;
}

void write_native_status(bool ok) noexcept {
  wgpu::Buffer buffer;
  {
    std::lock_guard lock{g_mutex};
    if (!ensure_shared_locked()) {
      return;
    }
    buffer = g_shared.statusNative;
  }
  const std::array<uint32_t, 4> words{ok ? 1u : 0u, 0, 0, 0};
  webgpu::g_queue.WriteBuffer(buffer, 0, words.data(), sizeof(words));
  bump(g_statusWrites);
  g_lastStatus.store(words[0], std::memory_order_relaxed);
}

void drop_native_step() noexcept {
  g_nativeOut.reset();
  if (g_requested && !std::exchange(g_warnedStaleStep, true)) {
    Log.warn("Native step {} was requested on a frame without a scene compose, so it never ran",
             g_requested->step.name);
  }
  g_requested.reset();
}

void shutdown() noexcept {
  g_source.reset();
  g_requested.reset();
  g_nativeOut.reset();
  g_groups.clear();
  g_encodes = 0;
  g_warnedSkip = false;
  g_warnedStaleStep = false;
  std::lock_guard lock{g_mutex};
  g_pipelines.clear();
  g_shared = {};
}

namespace testing {
Counters counters() noexcept {
  return {
      .encoded = g_encoded.load(std::memory_order_relaxed),
      .sourced = g_sourced.load(std::memory_order_relaxed),
      .skipped = g_skipped.load(std::memory_order_relaxed),
      .native = g_nativeBound.load(std::memory_order_relaxed),
      .statusWrites = g_statusWrites.load(std::memory_order_relaxed),
      .lastStatus = g_lastStatus.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_encoded.store(0, std::memory_order_relaxed);
  g_sourced.store(0, std::memory_order_relaxed);
  g_skipped.store(0, std::memory_order_relaxed);
  g_nativeBound.store(0, std::memory_order_relaxed);
  g_statusWrites.store(0, std::memory_order_relaxed);
  g_lastStatus.store(0, std::memory_order_relaxed);
  g_hold.store(false, std::memory_order_relaxed);
}

void hold_pipelines(bool hold) noexcept { g_hold.store(hold, std::memory_order_relaxed); }

shaders::RenderPipelineDesc pipeline_desc(ComposeMode mode, wgpu::TextureFormat colorFormat,
                                          wgpu::TextureFormat depthFormat) {
  const bool split = mode == ComposeMode::Split;
  shaders::RenderPipelineDesc desc{
      .label = split ? "upscale.compose split" : "upscale.compose in place",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = split ? "fs_split" : "fs_inplace",
      .defines = split ? std::span<const std::string_view>{} : std::span<const std::string_view>{InplaceDefines},
      .colorCount = 1,
      .depthFormat = split ? depthFormat : wgpu::TextureFormat::Undefined,
      .depthWrite = split,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  desc.colors[0] = {.format = colorFormat, .writeMask = split ? wgpu::ColorWriteMask::All : RgbMask};
  return desc;
}

bool encode(wgpu::CommandEncoder& cmd, const SceneComposeData& data, const Targets& targets,
            const std::optional<Source>& source) {
  return encode_targets(cmd, data, targets, source, false);
}
} // namespace testing
} // namespace aurora::render::detail::upscale::compose

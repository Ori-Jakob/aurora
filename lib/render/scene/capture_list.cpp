#include "capture_list.hpp"

#include "classifier.hpp"
#include "../graph.hpp"
#include "../graph_bind.hpp"
#include "../pass_context.hpp"
#include "../scene_depth.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"
#include "../../gfx/encoding.hpp"
#include "../../gfx/frame_packet.hpp"
#include "../../gfx/pipeline_cache.hpp"
#include "../../gx/draw_route.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render_graph.hpp>

#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <utility>

namespace aurora::render::scene::capture_list {
namespace {
constexpr Module Log{"aurora::render::scene"};

constexpr std::string_view CheckShader = "scene/capture_check";
constexpr uint64_t CheckLayoutKey = 0x63617074636865ull;
constexpr uint32_t CheckWorkgroup = 8;
constexpr uint64_t CheckBytes = 2 * sizeof(uint32_t);

struct Slot {
  Consumer consumer;
  bool registered = false;
  bool enabled = false;
};

// Game thread
std::array<Slot, MaxConsumers> g_slots;
// FIFO thread, latched while it is idle
std::array<Consumer, MaxConsumers> g_live;
uint8_t g_liveMask = 0;
// The worker checks a list against the layout its consumer had when the list was recorded
std::array<std::atomic_uint64_t, MaxConsumers> g_layoutKeys{};
const testing::PipelineHooks* g_hooks = nullptr;
bool g_warnedLayout = false;

struct Counters {
  std::atomic_uint64_t selected, cloned, pending, lateRefused, gpuRefused, replayed;
} g_counts;

// Only one thread writes each, so a plain add keeps lock prefixes off the draw path
inline void bump(std::atomic_uint64_t& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

bool uses_gpu_arrays() noexcept {
  const gx::GXState& state = gx::g_gxState;
  for (uint32_t mask = state.gpuArrayMask; mask != 0; mask &= mask - 1) {
    const auto attr = static_cast<uint32_t>(GX_VA_POS + std::countr_zero(mask));
    if (state.vtxDesc[attr] == GX_INDEX8 || state.vtxDesc[attr] == GX_INDEX16) {
      return true;
    }
  }
  return false;
}

gfx::PipelineRef ready_ref(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) noexcept {
  return g_hooks != nullptr && g_hooks->ready != nullptr ? g_hooks->ready(config, layout)
                                                         : gfx::ready_pipeline(config, layout);
}

void warm_ref(const gx::PipelineConfig& config, const gfx::RenderTargetLayout& layout) noexcept {
  if (g_hooks != nullptr && g_hooks->warm != nullptr) {
    g_hooks->warm(config, layout);
    return;
  }
  gfx::warm_pipeline(config, layout);
}

struct CheckLocal {
  uint16_t color = UINT16_MAX;
  uint16_t depth = UINT16_MAX;
  uint16_t scene = UINT16_MAX;
};
static_assert(sizeof(CheckLocal) <= detail::PassLocalSize);

struct Check {
  ConsumerId consumer = InvalidConsumer;
  detail::PassId pass = detail::InvalidPassId;
  wgpu::Color clearColor{};
  float clearDepth = 0.f;
  wgpu::BindGroupLayout groupLayout;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::Buffer counts;
  wgpu::Buffer readback;
  std::atomic_uint32_t armed{UINT32_MAX};
  std::mutex mutex;
  testing::CheckResult result;
} g_check;

// The replay target the check renders into: the scene colour format and depth, one sample
gfx::RenderTargetLayout check_layout() noexcept {
  const gfx::RenderTargetLayout scene = gfx::scene_render_target_layout();
  gfx::RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .depthStencilFormat = scene.depthStencilFormat,
      .sampleCount = 1,
  };
  layout.colorAttachments[0] = {
      .semantic = gfx::ColorAttachmentSemantic::SceneColor,
      .format = scene.colorAttachments[gfx::SceneColorAttachmentIndex].format,
  };
  gfx::detail::finalize_render_target_layout(layout);
  return layout;
}

bool check_select(const DrawClass& cls, void*) noexcept { return !cls.offscreen && cls.key.window <= Window::Opaque; }

void set_result(const testing::CheckResult& result) {
  std::lock_guard lock{g_check.mutex};
  g_check.result = result;
}

void declare_check(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (g_check.consumer == InvalidConsumer || !g_check.pipelineLayout || setup.env == nullptr ||
      !setup.gx_available(detail::GxResource::Color) || !setup.gx_available(detail::GxResource::Depth) ||
      setup.env->gxSampleCount != 1) {
    builder.skip();
    return;
  }
  builder.after(phase::Capture);
  builder.before(phase::Geometry);
  builder.manual_bindings();
  builder.side_effect();
  const auto& formats = setup.env->gxFormats;
  CheckLocal local;
  local.color = builder
                    .create(CheckColor,
                            {.format = formats[static_cast<size_t>(detail::GxResource::Color)],
                             .sizeClass = SizeClass::Efb,
                             .extraUsage = wgpu::TextureUsage::TextureBinding},
                            Access::ColorWrite)
                    .index;
  local.depth =
      builder
          .create(CheckDepth,
                  {.format = formats[static_cast<size_t>(detail::GxResource::Depth)], .sizeClass = SizeClass::Efb},
                  Access::DepthWrite)
          .index;
  local.scene = builder
                    .use(resource::GxColor, Access::Sampled,
                         detail::UseOptions{.missing = Missing::DisablePass, .visibility = wgpu::ShaderStage::Compute})
                    .index;
  const std::span<std::byte, detail::PassLocalSize> out = builder.local();
  std::memcpy(out.data(), &local, sizeof(local));
}

void check_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  if (!g_check.pipelineLayout) {
    ids[0] = shaders::NeverReady.value;
    return;
  }
  ids[0] = shaders::request_compute({
                                        .label = CheckPass,
                                        .shader = CheckShader,
                                        .entryPoint = "compare",
                                        .layout = g_check.pipelineLayout,
                                        .layoutKey = CheckLayoutKey,
                                    })
               .value;
}

void queue_readback(const wgpu::CommandEncoder& encoder, uint32_t frame, wgpu::Extent3D extent, uint32_t draws) {
  encoder.CopyBufferToBuffer(g_check.counts, 0, g_check.readback, 0, CheckBytes);
  const wgpu::Buffer readback = g_check.readback;
  detail::queue_after_frame([readback, frame, extent, draws] {
    readback.MapAsync(wgpu::MapMode::Read, 0, CheckBytes, wgpu::CallbackMode::AllowSpontaneous,
                      [readback, frame, extent, draws](wgpu::MapAsyncStatus status, wgpu::StringView) {
                        testing::CheckResult result{.state = testing::CheckState::Failed,
                                                    .frame = frame,
                                                    .width = extent.width,
                                                    .height = extent.height,
                                                    .draws = draws};
                        const auto* words =
                            status == wgpu::MapAsyncStatus::Success
                                ? static_cast<const uint32_t*>(readback.GetConstMappedRange(0, CheckBytes))
                                : nullptr;
                        if (words != nullptr) {
                          result.state = testing::CheckState::Ready;
                          result.mismatches = words[0];
                          result.lastMismatch = words[1];
                        }
                        if (status == wgpu::MapAsyncStatus::Success) {
                          readback.Unmap();
                        }
                        set_result(result);
                      });
  });
}

void encode_check(detail::PassContext& ctx, const void*) {
  CheckLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const auto& lists = ctx.stage_data().captureList;
  const wgpu::ComputePipeline compare = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::Extent3D extent = ctx.extent(TexRef{local.color});
  const uint32_t frame = ctx.frame_index();
  const bool armed = g_check.armed.load(std::memory_order_acquire) == frame;
  if (lists == nullptr || !compare || extent.width == 0 || extent.height == 0) {
    if (armed) {
      set_result({.state = testing::CheckState::Failed, .frame = frame});
    }
    return;
  }
  const detail::ColorTarget color{.ref = TexRef{local.color}, .clear = g_check.clearColor};
  const detail::DepthTarget depth{.ref = TexRef{local.depth}, .clear = g_check.clearDepth};
  {
    const wgpu::RenderPassEncoder pass = ctx.begin_render(std::span{&color, 1}, &depth, "capture.check replay");
    if (!pass) {
      return;
    }
    encode(ctx, pass, g_check.consumer);
    pass.End();
  }
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .textureView = ctx.view(TexRef{local.scene})},
      wgpu::BindGroupEntry{.binding = 1, .textureView = ctx.view(TexRef{local.color})},
      wgpu::BindGroupEntry{.binding = 2, .buffer = g_check.counts, .size = CheckBytes},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "capture.check",
      .layout = g_check.groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const wgpu::BindGroup group = ctx.device().CreateBindGroup(&groupDescriptor);
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  encoder.ClearBuffer(g_check.counts, 0, CheckBytes);
  const wgpu::ComputePassDescriptor passDescriptor{.label = "capture.check compare"};
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&passDescriptor);
  pass.SetPipeline(compare);
  pass.SetBindGroup(0, group);
  pass.DispatchWorkgroups((extent.width + CheckWorkgroup - 1) / CheckWorkgroup,
                          (extent.height + CheckWorkgroup - 1) / CheckWorkgroup);
  pass.End();
  if (armed) {
    queue_readback(encoder, frame, extent, lists->lists[g_check.consumer].draws);
  }
}

detail::PassDesc check_pass_desc() {
  return {
      .name = std::string{CheckPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .cls = detail::PassClass::Required,
      .order = 2,
      .declare = declare_check,
      .encode = encode_check,
      .pipelines = check_pipelines,
  };
}

void create_check_resources() {
  const wgpu::Device& device = webgpu::g_device;
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage, .minBindingSize = CheckBytes},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "capture.check",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  g_check.groupLayout = device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "capture.check",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_check.groupLayout,
  };
  g_check.pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  const wgpu::BufferDescriptor countsDescriptor{
      .label = "capture.check counts",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
      .size = CheckBytes,
  };
  g_check.counts = device.CreateBuffer(&countsDescriptor);
  const wgpu::BufferDescriptor readbackDescriptor{
      .label = "capture.check readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = CheckBytes,
  };
  g_check.readback = device.CreateBuffer(&readbackDescriptor);
}
} // namespace

ConsumerId register_consumer(const Consumer& consumer) noexcept {
  if (consumer.select == nullptr) {
    return InvalidConsumer;
  }
  for (ConsumerId id = 0; id < MaxConsumers; ++id) {
    Slot& slot = g_slots[id];
    if (!slot.registered) {
      slot = {.consumer = consumer, .registered = true};
      return id;
    }
  }
  Log.warn("Capture list: no room for consumer {}; {} are registered", consumer.name != nullptr ? consumer.name : "?",
           MaxConsumers);
  return InvalidConsumer;
}

void unregister_consumer(ConsumerId id) noexcept {
  if (id < MaxConsumers) {
    g_slots[id] = {};
  }
}

void set_enabled(ConsumerId id, bool enabled) noexcept {
  if (id < MaxConsumers && g_slots[id].registered) {
    g_slots[id].enabled = enabled;
  }
}

void apply_frame_state() noexcept {
  if (g_check.consumer != InvalidConsumer) {
    g_slots[g_check.consumer].consumer.layout = check_layout();
  }
  uint8_t mask = 0;
  for (ConsumerId id = 0; id < MaxConsumers; ++id) {
    const Slot& slot = g_slots[id];
    if (slot.registered && slot.enabled) {
      g_live[id] = slot.consumer;
      mask |= static_cast<uint8_t>(1u << id);
    }
    g_layoutKeys[id].store(slot.consumer.layout.key, std::memory_order_relaxed);
  }
  g_liveMask = mask;
}

uint8_t select(const DrawClass& cls, const gx::draw_route::Query&) noexcept {
  if (g_liveMask == 0 || cls.key.window > Window::Opaque) {
    return 0;
  }
  uint8_t mask = 0;
  for (uint32_t live = g_liveMask; live != 0; live &= live - 1) {
    const auto id = static_cast<uint32_t>(std::countr_zero(live));
    const Consumer& consumer = g_live[id];
    if (consumer.select(cls, consumer.user)) {
      mask |= static_cast<uint8_t>(1u << id);
    }
  }
  if (mask == 0) {
    return 0;
  }
  // GPU-written arrays are only valid for the draw that bound them
  if (uses_gpu_arrays()) {
    bump(g_counts.gpuRefused, static_cast<uint64_t>(std::popcount(mask)));
    return 0;
  }
  bump(g_counts.selected, static_cast<uint64_t>(std::popcount(mask)));
  return mask;
}

void on_push(const gx::DrawData& draw, const gx::PipelineConfig& config, const DrawClass& cls, uint8_t mask) noexcept {
  const uint64_t targetKey = gfx::get_render_target_layout().key;
  for (uint32_t bits = mask & g_liveMask; bits != 0; bits &= bits - 1) {
    const auto id = static_cast<uint32_t>(std::countr_zero(bits));
    const Consumer& consumer = g_live[id];
    if (consumer.accept != nullptr && !consumer.accept(draw, cls, consumer.user)) {
      continue;
    }
    gx::DrawData clone = draw;
    if (consumer.layout.key != targetKey) {
      gx::PipelineConfig legacy = config;
      legacy.shaderConfig.ext = {};
      const gfx::PipelineRef ref = ready_ref(legacy, consumer.layout);
      if (ref == 0) {
        warm_ref(legacy, consumer.layout);
        bump(g_counts.pending);
        continue;
      }
      clone.pipeline = ref;
      // The legacy pipeline has no companion lane to read it
      clone.bindGroups.companionBindGroup = 0;
    }
    if (consumer.edit != nullptr && !consumer.edit(clone, cls, consumer.user)) {
      continue;
    }
    if (!gfx::append_capture_draw(id, consumer.layout.key, clone)) {
      bump(g_counts.lateRefused);
      continue;
    }
    bump(g_counts.cloned);
  }
}

void encode(const detail::PassContext& ctx, const wgpu::RenderPassEncoder& pass, ConsumerId id) {
  const auto& frame = ctx.stage_data().captureList;
  const auto& depth = ctx.stage_data().sceneDepth;
  if (frame == nullptr || depth == nullptr || id >= MaxConsumers) {
    return;
  }
  const gfx::CaptureListFrame::List& list = frame->lists[id];
  if (list.draws == 0) {
    return;
  }
  // Its pipelines were built for the layout the consumer had at push, which a latch since may have changed
  if (list.layoutKey != g_layoutKeys[id].load(std::memory_order_relaxed)) {
    if (!std::exchange(g_warnedLayout, true)) {
      Log.warn("Capture list {} was recorded for another target layout; its replay is skipped", id);
    }
    return;
  }
  gfx::detail::encode_command_list(pass, list.commands, gfx::detail::replay_pass_info(depth->size));
  g_counts.replayed.fetch_add(list.draws, std::memory_order_relaxed);
}

Stats stats() noexcept {
  return {
      .selected = g_counts.selected.load(std::memory_order_relaxed),
      .cloned = g_counts.cloned.load(std::memory_order_relaxed),
      .pending = g_counts.pending.load(std::memory_order_relaxed),
      .lateRefused = g_counts.lateRefused.load(std::memory_order_relaxed),
      .gpuRefused = g_counts.gpuRefused.load(std::memory_order_relaxed),
      .discarded = gfx::discarded_capture_draws(),
      .replayed = g_counts.replayed.load(std::memory_order_relaxed),
  };
}

void shutdown() noexcept {
  testing::remove_check();
  g_slots = {};
  g_live = {};
  g_liveMask = 0;
}

namespace testing {
void set_pipeline_hooks(const PipelineHooks* hooks) noexcept { g_hooks = hooks; }

ConsumerId install_check(const wgpu::Color& clearColor, float clearDepth) {
  if (g_check.consumer != InvalidConsumer) {
    return g_check.consumer;
  }
  const ConsumerId id = register_consumer({.name = "capture.check", .select = check_select, .layout = check_layout()});
  if (id == InvalidConsumer) {
    return id;
  }
  set_enabled(id, true);
  g_check.consumer = id;
  g_check.clearColor = clearColor;
  g_check.clearDepth = clearDepth;
  g_check.armed.store(UINT32_MAX, std::memory_order_relaxed);
  set_result({});
  if (webgpu::g_device) {
    create_check_resources();
  }
  g_check.pass = detail::register_pass(check_pass_desc());
  return id;
}

void remove_check() noexcept {
  if (g_check.pass != detail::InvalidPassId) {
    detail::unregister_pass(g_check.pass);
    g_check.pass = detail::InvalidPassId;
  }
  if (g_check.consumer != InvalidConsumer) {
    unregister_consumer(g_check.consumer);
    g_check.consumer = InvalidConsumer;
  }
  g_check.groupLayout = {};
  g_check.pipelineLayout = {};
  g_check.counts = {};
  g_check.readback = {};
}

void arm_check(uint32_t frameIndex) noexcept {
  set_result({.state = CheckState::Armed, .frame = frameIndex});
  g_check.armed.store(frameIndex, std::memory_order_release);
}

CheckResult check_result() noexcept {
  std::lock_guard lock{g_check.mutex};
  return g_check.result;
}

void reset() noexcept {
  shutdown();
  g_hooks = nullptr;
  for (auto& key : g_layoutKeys) {
    key.store(0, std::memory_order_relaxed);
  }
  g_warnedLayout = false;
  for (std::atomic_uint64_t* counter : {&g_counts.selected, &g_counts.cloned, &g_counts.pending, &g_counts.lateRefused,
                                        &g_counts.gpuRefused, &g_counts.replayed}) {
    counter->store(0, std::memory_order_relaxed);
  }
  set_result({});
}
} // namespace testing
} // namespace aurora::render::scene::capture_list

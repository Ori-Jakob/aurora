#include "denoise.hpp"

#include "gtao.hpp"
#include "hiz.hpp"

#include "../pass_context.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::detail::ao::denoise {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::ao"};
constexpr TexRef HistoryTarget{3};
constexpr uint32_t PrevGroupIndex = 2;
constexpr size_t CacheCapacity = 4;
constexpr uint64_t LayoutSalt = 0x414F2D44454E4F53ull;
constexpr std::array<std::string_view, 1> ResetDefines{"AO_RESET"};

struct PipelineLayoutEntry {
  wgpu::BindGroupLayout passLayout;
  wgpu::PipelineLayout layout;
};

struct CacheEntry {
  wgpu::BindGroupLayout layout;
  wgpu::TextureView prev;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

std::mutex g_layoutMutex;
wgpu::BindGroupLayout g_prevLayout;
std::vector<PipelineLayoutEntry> g_pipelineLayouts;

std::vector<CacheEntry> g_cache;
wgpu::Texture g_dummyTexture;
wgpu::TextureView g_dummyView;

std::atomic_uint32_t g_dispatches{0};
std::atomic_uint32_t g_valid{0};
std::atomic_uint32_t g_resets{0};
std::atomic_uint32_t g_groupsCreated{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Ao);
  builder.use(gtao::Resource, Access::Sampled, Missing::DisablePass);
  builder.use(hiz::Resource, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::Motion, Access::Sampled, Missing::DisablePass);
  builder.history(History, history_desc(working_divisor(setup.settings->ao)), Access::StorageWrite);
}

wgpu::PipelineLayout pipeline_layout(const PipelineSetup& setup) {
  if (!webgpu::g_device || !setup.frameLayout || !setup.passLayout) {
    return {};
  }
  std::lock_guard lock{g_layoutMutex};
  if (!g_prevLayout) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
    };
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "ao.denoise history",
        .entryCount = 1,
        .entries = &entry,
    };
    g_prevLayout = webgpu::g_device.CreateBindGroupLayout(&descriptor);
  }
  for (const PipelineLayoutEntry& entry : g_pipelineLayouts) {
    if (entry.passLayout.Get() == setup.passLayout.Get()) {
      return entry.layout;
    }
  }
  const std::array groups{setup.frameLayout, setup.passLayout, g_prevLayout};
  const wgpu::PipelineLayoutDescriptor descriptor{
      .label = "ao.denoise",
      .bindGroupLayoutCount = groups.size(),
      .bindGroupLayouts = groups.data(),
  };
  wgpu::PipelineLayout layout = webgpu::g_device.CreatePipelineLayout(&descriptor);
  g_pipelineLayouts.push_back({.passLayout = setup.passLayout, .layout = layout});
  return layout;
}

wgpu::BindGroupLayout prev_layout() {
  std::lock_guard lock{g_layoutMutex};
  return g_prevLayout;
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  const wgpu::PipelineLayout layout = pipeline_layout(setup);
  if (!layout) {
    return;
  }
  const uint64_t key = setup.layoutHash ^ LayoutSalt;
  ids[0] = shaders::request_compute({
                                        .label = "ao.denoise",
                                        .shader = Shader,
                                        .entryPoint = "main",
                                        .layout = layout,
                                        .layoutKey = key,
                                    })
               .value;
  ids[1] = shaders::request_compute({
                                        .label = "ao.denoise reset",
                                        .shader = Shader,
                                        .entryPoint = "main",
                                        .defines = ResetDefines,
                                        .layout = layout,
                                        .layoutKey = key,
                                    })
               .value;
}

const wgpu::TextureView& dummy_prev(const wgpu::Device& device) {
  if (!g_dummyView && device) {
    const wgpu::TextureDescriptor descriptor{
        .label = "ao.denoise empty history",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1, 1},
        .format = Format,
    };
    g_dummyTexture = device.CreateTexture(&descriptor);
    g_dummyView = g_dummyTexture.CreateView();
  }
  return g_dummyView;
}

wgpu::BindGroup cached_group(const PassContext& ctx, const wgpu::BindGroupLayout& layout,
                             const wgpu::TextureView& prev) {
  const uint64_t frame = ctx.frame_id();
  for (CacheEntry& entry : g_cache) {
    if (entry.layout.Get() == layout.Get() && entry.prev.Get() == prev.Get()) {
      entry.lastUsed = frame;
      return entry.group;
    }
  }
  if (g_cache.size() >= CacheCapacity) {
    g_cache.erase(std::ranges::min_element(g_cache, {}, &CacheEntry::lastUsed));
  }
  const wgpu::BindGroupEntry entry{.binding = 0, .textureView = prev};
  const wgpu::BindGroupDescriptor descriptor{
      .label = "ao.denoise history",
      .layout = layout,
      .entryCount = 1,
      .entries = &entry,
  };
  CacheEntry& added = g_cache.emplace_back(CacheEntry{
      .layout = layout,
      .prev = prev,
      .group = ctx.device().CreateBindGroup(&descriptor),
      .lastUsed = frame,
  });
  bump(g_groupsCreated);
  return added.group;
}

uint32_t latched_flags(const PassContext& ctx) noexcept {
  const std::span<const std::byte> bytes = ctx.params();
  Params params;
  if (bytes.size() < sizeof(params)) {
    return 0;
  }
  std::memcpy(&params, bytes.data(), sizeof(params));
  return params.flags;
}

void encode(PassContext& ctx, const void*) {
  const bool temporalOff = (latched_flags(ctx) & FlagTemporalOff) != 0;
  const bool prevValid = ctx.prev_valid(HistoryTarget);
  const bool motionValid = motion_valid(ctx.stage_data());
  wgpu::TextureView prev = prevValid && motionValid && !temporalOff ? ctx.prev(HistoryTarget) : wgpu::TextureView{};
  const bool reset = !prev;
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(reset ? 1 : 0));
  const wgpu::BindGroupLayout layout = prev_layout();
  if (reset) {
    prev = dummy_prev(ctx.device());
  }
  if (!pipeline || !layout || !prev) {
    return;
  }
  const wgpu::Extent3D size = ctx.extent(HistoryTarget);
  const wgpu::BindGroup group = cached_group(ctx, layout, prev);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(PrevGroupIndex, group);
  pass.DispatchWorkgroups(groups(size.width), groups(size.height));
  pass.End();
  bump(g_dispatches);
  bump(reset ? g_resets : g_valid);
  if (reset && !temporalOff) {
    Log.debug("AO history reset at frame {}: {}", ctx.frame_id(),
              !prevValid    ? "no valid history"
              : motionValid ? "no previous texture"
                            : "motion invalid");
  }
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 1,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = scene_ready,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_camera,
  };
}

void worker_tick(uint64_t frameId) noexcept {
  if (g_cache.empty()) {
    return;
  }
  std::erase_if(g_cache, [frameId](const CacheEntry& entry) { return frameId > entry.lastUsed + RetainFrames; });
}

void shutdown() noexcept {
  g_cache.clear();
  g_dummyView = {};
  g_dummyTexture = {};
  std::lock_guard lock{g_layoutMutex};
  g_pipelineLayouts.clear();
  g_prevLayout = {};
}

namespace testing {
Counters counters() noexcept {
  return {
      .dispatches = g_dispatches.load(std::memory_order_relaxed),
      .temporalValid = g_valid.load(std::memory_order_relaxed),
      .temporalResets = g_resets.load(std::memory_order_relaxed),
      .groupsCreated = g_groupsCreated.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cache.clear();
  g_dispatches.store(0, std::memory_order_relaxed);
  g_valid.store(0, std::memory_order_relaxed);
  g_resets.store(0, std::memory_order_relaxed);
  g_groupsCreated.store(0, std::memory_order_relaxed);
}
} // namespace testing
} // namespace aurora::render::detail::ao::denoise

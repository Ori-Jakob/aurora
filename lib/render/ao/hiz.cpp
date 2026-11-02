#include "hiz.hpp"

#include "../graph_layout.hpp"
#include "../pass_context.hpp"
#include "../shaders.hpp"

#include "../../gx/gx.hpp"

#include <array>
#include <atomic>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::detail::ao::hiz {
namespace {
static_assert(gx::UseReversedZ);

constexpr size_t CacheCapacity = 4;
constexpr TexRef Target{1};

struct CacheEntry {
  wgpu::Texture texture;
  wgpu::BindGroupLayout layout;
  std::array<wgpu::BindGroup, MaxCachedLevels> groups;
  uint64_t lastUsed = 0;
};

std::vector<CacheEntry> g_cache;

std::atomic_uint32_t g_dispatches{0};
std::atomic_uint32_t g_groupsCreated{0};
std::atomic_uint32_t g_cacheEntries{0};

inline void bump(std::atomic_uint32_t& counter, uint32_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void publish_cache_size() noexcept {
  g_cacheEntries.store(static_cast<uint32_t>(g_cache.size()), std::memory_order_relaxed);
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.create(Resource, pyramid_desc(working_divisor(setup.settings->ao)), Access::StorageWrite);
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "depth.pyramid linearize",
                                        .shader = Shader,
                                        .entryPoint = "linearize",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
  ids[1] = shaders::request_compute({
                                        .label = "depth.pyramid filter",
                                        .shader = Shader,
                                        .entryPoint = "filter_mip",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

const CacheEntry& cached_groups(const PassContext& ctx, const wgpu::Texture& texture, uint32_t levels) {
  const wgpu::BindGroupLayout& layout = ctx.pass_layout();
  const uint64_t frame = ctx.frame_id();
  for (CacheEntry& entry : g_cache) {
    if (entry.texture.Get() == texture.Get() && entry.layout.Get() == layout.Get()) {
      entry.lastUsed = frame;
      return entry;
    }
  }
  if (g_cache.size() >= CacheCapacity) {
    g_cache.erase(std::ranges::min_element(g_cache, {}, &CacheEntry::lastUsed));
  }
  CacheEntry& entry = g_cache.emplace_back(CacheEntry{.texture = texture, .layout = layout, .lastUsed = frame});
  for (uint32_t level = 1; level < levels; ++level) {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = ctx.view(Target, level - 1)},
        wgpu::BindGroupEntry{.binding = 1, .textureView = ctx.view(Target, level)},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .label = "depth.pyramid level",
        .layout = layout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    entry.groups[level] = ctx.device().CreateBindGroup(&descriptor);
  }
  bump(g_groupsCreated, levels - 1);
  publish_cache_size();
  return entry;
}

void encode(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline linearize = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::ComputePipeline filter = shaders::compute_pipeline(ctx.pipeline(1));
  if (!linearize || !filter) {
    return;
  }
  const wgpu::Texture texture = ctx.texture(Target);
  if (!texture) {
    return;
  }
  const uint32_t levels = written_levels(texture.GetMipLevelCount());
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const CacheEntry& entry = cached_groups(ctx, texture, levels);
  const uint32_t width = texture.GetWidth();
  const uint32_t height = texture.GetHeight();
  pass.SetPipeline(linearize);
  pass.DispatchWorkgroups(groups(width), groups(height));
  if (levels > 1) {
    pass.SetPipeline(filter);
  }
  for (uint32_t level = 1; level < levels; ++level) {
    pass.SetBindGroup(PassGroupIndex, entry.groups[level]);
    pass.DispatchWorkgroups(groups(level_size(width, level)), groups(level_size(height, level)));
  }
  pass.End();
  bump(g_dispatches, levels);
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
  publish_cache_size();
}

void clear_cache() noexcept {
  g_cache.clear();
  publish_cache_size();
}

namespace testing {
Counters counters() noexcept {
  return {
      .dispatches = g_dispatches.load(std::memory_order_relaxed),
      .groupsCreated = g_groupsCreated.load(std::memory_order_relaxed),
      .cacheEntries = g_cacheEntries.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cache.clear();
  g_dispatches.store(0, std::memory_order_relaxed);
  g_groupsCreated.store(0, std::memory_order_relaxed);
  g_cacheEntries.store(0, std::memory_order_relaxed);
}
} // namespace testing
} // namespace aurora::render::detail::ao::hiz

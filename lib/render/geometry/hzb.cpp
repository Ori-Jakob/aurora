#include "hzb.hpp"

#include "../graph_layout.hpp"
#include "../pass_context.hpp"
#include "../shaders.hpp"

#include "../../gx/gx.hpp"
#include "../../internal.hpp"

#include <array>
#include <atomic>
#include <string>
#include <vector>

namespace aurora::render::detail::hzb {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::hzb"};
static_assert(gx::UseReversedZ);

constexpr size_t CacheCapacity = 4;
constexpr TexRef Target{1};

struct CacheEntry {
  wgpu::Texture texture;
  wgpu::BindGroupLayout layout;
  std::array<wgpu::BindGroup, MaxLevels> groups;
  uint64_t lastUsed = 0;
};

std::vector<CacheEntry> g_cache;
PassId g_pass = InvalidPassId;

std::atomic_uint32_t g_dispatches{0};
std::atomic_uint32_t g_groupsCreated{0};
std::atomic_uint32_t g_cacheEntries{0};

inline void bump(std::atomic_uint32_t& counter, uint32_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void publish_cache_size() noexcept {
  g_cacheEntries.store(static_cast<uint32_t>(g_cache.size()), std::memory_order_relaxed);
}

void declare_opaque(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (setup.sceneDepth != DepthSource::Direct) {
    builder.skip();
    return;
  }
  builder.after(phase::Geometry);
  declare_build(builder, resource::GxDepth, Resource);
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
        .label = "hzb level",
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
} // namespace

TexRef use(InternalPassBuilder& builder, Missing missing) {
  CHECK(missing != Missing::BindDummyWhite, "hzb: a white dummy would report every object as occluded");
  return builder.use(Resource, Access::Sampled,
                     UseOptions{.missing = missing, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
}

void declare_build(InternalPassBuilder& builder, std::string_view source, std::string_view target) {
  CHECK(find_far_depth_pyramid(target) != nullptr, "hzb: '{}' has no far depth pyramid row", target);
  builder.use(source, Access::Sampled,
              UseOptions{.missing = Missing::BindDummyBlack, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.create(target, pyramid_desc(), Access::StorageWrite);
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = OpaquePass,
                                        .shader = Shader,
                                        .entryPoint = "main",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode_build(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  if (!pipeline) {
    return;
  }
  const wgpu::Texture texture = ctx.texture(Target);
  if (!texture) {
    return;
  }
  const uint32_t levels = texture.GetMipLevelCount();
  CHECK(levels <= MaxLevels, "hzb: {} levels is more than {}", levels, MaxLevels);
  if (levels > MaxLevels) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const CacheEntry& entry = cached_groups(ctx, texture, levels);
  const uint32_t width = texture.GetWidth();
  const uint32_t height = texture.GetHeight();
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(groups(width), groups(height));
  for (uint32_t level = 1; level < levels; ++level) {
    pass.SetBindGroup(PassGroupIndex, entry.groups[level]);
    pass.DispatchWorkgroups(groups(level_size(width, level)), groups(level_size(height, level)));
  }
  pass.End();
  bump(g_dispatches, levels);
}

PassDesc opaque_pass_desc() {
  return {
      .name = std::string{OpaquePass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .declare = declare_opaque,
      .encode = encode_build,
      .pipelines = request_pipelines,
  };
}

void initialize() {
  if (g_pass == InvalidPassId) {
    g_pass = register_pass(opaque_pass_desc());
  }
}

void worker_tick(uint64_t frameId) noexcept {
  if (g_cache.empty()) {
    return;
  }
  std::erase_if(g_cache, [frameId](const CacheEntry& entry) { return frameId > entry.lastUsed + RetainFrames; });
  publish_cache_size();
}

void shutdown() noexcept {
  if (g_pass != InvalidPassId) {
    unregister_pass(g_pass);
    g_pass = InvalidPassId;
  }
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
} // namespace aurora::render::detail::hzb

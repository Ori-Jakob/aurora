#include "ssil_radiance.hpp"

#include "denoise.hpp"

#include "../graph_layout.hpp"
#include "../pass_context.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../logging.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::detail::ao::ssil_radiance {
namespace {
constexpr Module Log{"aurora::render::ao"};
constexpr TexRef PrevRadiance{0};
constexpr TexRef PrevHistory{1};
constexpr TexRef Motion{2};
constexpr TexRef Hiz{3};
constexpr TexRef SceneNormal{4};
constexpr TexRef RadianceOut{5};
constexpr TexRef NormalOut{6};
constexpr uint32_t ReprojectPipeline = 0;
constexpr uint32_t ResetPipeline = 1;
constexpr uint32_t FilterPipeline = 2;
constexpr uint32_t FurnacePipeline = 3;
constexpr std::array<std::string_view, 1> ResetDefines{"SSIL_RESET"};
constexpr std::array<std::string_view, 1> FurnaceDefines{"SSIL_FURNACE"};

struct CacheEntry {
  wgpu::BindGroupLayout layout;
  wgpu::Texture radiance;
  wgpu::Texture normal;
  wgpu::Texture hiz;
  wgpu::TextureView motion;
  wgpu::TextureView sceneNormal;
  std::array<wgpu::BindGroup, MipLevels> groups;
  uint64_t lastUsed = 0;
};

std::vector<CacheEntry> g_cache;

std::atomic_uint32_t g_dispatches{0};
std::atomic_uint32_t g_groupsCreated{0};
std::atomic_uint32_t g_cacheEntries{0};
std::atomic_uint32_t g_reprojected{0};
std::atomic_uint32_t g_cleared{0};
std::atomic_uint32_t g_clearedNoRadiance{0};
std::atomic_uint32_t g_clearedNoHistory{0};
std::atomic_uint32_t g_clearedMotion{0};
std::atomic_uint32_t g_clearedUnits{0};

inline void bump(std::atomic_uint32_t& counter, uint32_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void publish_cache_size() noexcept {
  g_cacheEntries.store(static_cast<uint32_t>(g_cache.size()), std::memory_order_relaxed);
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!ssil_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.use_prev(resource::RadianceDefogged, Missing::BindDummyBlack);
  builder.use_prev(denoise::History, Missing::BindDummyBlack);
  builder.use(resource::Motion, Access::Sampled, Missing::DisablePass);
  builder.use(hiz::Resource, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  const uint32_t divisor = working_divisor(setup.settings->ao);
  builder.create(Resource, radiance_desc(divisor), Access::StorageWrite);
  builder.create(Normal, normal_desc(divisor), Access::StorageWrite);
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  const auto request = [&](const char* label, std::string_view entry, std::span<const std::string_view> defines) {
    return shaders::request_compute({
                                        .label = label,
                                        .shader = Shader,
                                        .entryPoint = entry,
                                        .defines = defines,
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
        .value;
  };
  ids[ReprojectPipeline] = request("ssil.radiance reproject", "reproject", {});
  ids[ResetPipeline] = request("ssil.radiance reset", "reproject", ResetDefines);
  ids[FilterPipeline] = request("ssil.radiance filter", "filter_mip", {});
  if (ssil_furnace()) {
    ids[FurnacePipeline] = request("ssil.radiance furnace", "reproject", FurnaceDefines);
  }
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

// Each level reads the one below in place of the history views, so cached groups hold no history
const CacheEntry& cached_groups(const PassContext& ctx, uint32_t levels) {
  const wgpu::BindGroupLayout& layout = ctx.pass_layout();
  const wgpu::Texture radiance = ctx.texture(RadianceOut);
  const wgpu::Texture normal = ctx.texture(NormalOut);
  const wgpu::Texture hizTexture = ctx.texture(Hiz);
  const wgpu::TextureView motion = ctx.view(Motion);
  const wgpu::TextureView sceneNormal = ctx.view(SceneNormal);
  const uint64_t frame = ctx.frame_id();
  for (CacheEntry& entry : g_cache) {
    if (entry.layout.Get() == layout.Get() && entry.radiance.Get() == radiance.Get() &&
        entry.normal.Get() == normal.Get() && entry.hiz.Get() == hizTexture.Get() &&
        entry.motion.Get() == motion.Get() && entry.sceneNormal.Get() == sceneNormal.Get()) {
      entry.lastUsed = frame;
      return entry;
    }
  }
  if (g_cache.size() >= CacheCapacity) {
    g_cache.erase(std::ranges::min_element(g_cache, {}, &CacheEntry::lastUsed));
  }
  CacheEntry& entry = g_cache.emplace_back(CacheEntry{
      .layout = layout,
      .radiance = radiance,
      .normal = normal,
      .hiz = hizTexture,
      .motion = motion,
      .sceneNormal = sceneNormal,
      .lastUsed = frame,
  });
  for (uint32_t level = 1; level < levels; ++level) {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = PrevRadiance.index, .textureView = ctx.view(RadianceOut, level - 1)},
        wgpu::BindGroupEntry{.binding = PrevHistory.index, .textureView = ctx.view(NormalOut, level - 1)},
        wgpu::BindGroupEntry{.binding = Motion.index, .textureView = motion},
        wgpu::BindGroupEntry{.binding = Hiz.index, .textureView = ctx.view(Hiz, level - 1)},
        wgpu::BindGroupEntry{.binding = SceneNormal.index, .textureView = sceneNormal},
        wgpu::BindGroupEntry{.binding = RadianceOut.index, .textureView = ctx.view(RadianceOut, level)},
        wgpu::BindGroupEntry{.binding = NormalOut.index, .textureView = ctx.view(NormalOut, level)},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .label = "ssil.radiance level",
        .layout = layout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    entry.groups[level] = ctx.device().CreateBindGroup(&descriptor);
  }
  bump(g_groupsCreated, levels > 0 ? levels - 1 : 0);
  publish_cache_size();
  return entry;
}

void encode(PassContext& ctx, const void*) {
  const bool furnace = ssil_furnace();
  const bool radianceValid = ctx.prev_valid(PrevRadiance);
  const bool historyValid = ctx.prev_valid(PrevHistory);
  const bool motionValid = motion_valid(ctx.stage_data());
  const bool unitsChanged = (latched_flags(ctx) & FlagSsilReset) != 0;
  const bool reset = !(radianceValid && historyValid && motionValid && !unitsChanged);
  const uint32_t first = furnace ? FurnacePipeline : reset ? ResetPipeline : ReprojectPipeline;
  const wgpu::ComputePipeline reproject = shaders::compute_pipeline(ctx.pipeline(first));
  const wgpu::ComputePipeline filter = shaders::compute_pipeline(ctx.pipeline(FilterPipeline));
  const wgpu::Texture radiance = ctx.texture(RadianceOut);
  if (!reproject || !filter || !radiance || !ctx.texture(NormalOut)) {
    return;
  }
  const uint32_t levels = written_levels(radiance.GetMipLevelCount());
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const CacheEntry& entry = cached_groups(ctx, levels);
  const uint32_t width = radiance.GetWidth();
  const uint32_t height = radiance.GetHeight();
  pass.SetPipeline(reproject);
  pass.DispatchWorkgroups(groups(width), groups(height));
  if (levels > 1) {
    pass.SetPipeline(filter);
  }
  for (uint32_t level = 1; level < levels; ++level) {
    pass.SetBindGroup(PassGroupIndex, entry.groups[level]);
    pass.DispatchWorkgroups(groups(hiz::level_size(width, level)), groups(hiz::level_size(height, level)));
  }
  pass.End();
  bump(g_dispatches, levels);
  if (furnace || !reset) {
    bump(g_reprojected);
    return;
  }
  bump(g_cleared);
  bump(!radianceValid  ? g_clearedNoRadiance
       : !historyValid ? g_clearedNoHistory
       : !motionValid  ? g_clearedMotion
                       : g_clearedUnits);
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 6,
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
  const testing::Counters c = testing::counters();
  if (c.reprojected != 0 || c.cleared != 0) {
    Log.info(
        "SSIL radiance: reprojected {} frames, cleared {} ({} no radiance, {} no AO history, {} motion invalid, "
        "{} units changed)",
        c.reprojected, c.cleared, c.clearedNoRadiance, c.clearedNoHistory, c.clearedMotion, c.clearedUnits);
  }
}

namespace testing {
Counters counters() noexcept {
  return {
      .dispatches = g_dispatches.load(std::memory_order_relaxed),
      .groupsCreated = g_groupsCreated.load(std::memory_order_relaxed),
      .cacheEntries = g_cacheEntries.load(std::memory_order_relaxed),
      .reprojected = g_reprojected.load(std::memory_order_relaxed),
      .cleared = g_cleared.load(std::memory_order_relaxed),
      .clearedNoRadiance = g_clearedNoRadiance.load(std::memory_order_relaxed),
      .clearedNoHistory = g_clearedNoHistory.load(std::memory_order_relaxed),
      .clearedMotion = g_clearedMotion.load(std::memory_order_relaxed),
      .clearedUnits = g_clearedUnits.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cache.clear();
  for (std::atomic_uint32_t* counter : {&g_dispatches, &g_groupsCreated, &g_cacheEntries, &g_reprojected, &g_cleared,
                                        &g_clearedNoRadiance, &g_clearedNoHistory, &g_clearedMotion, &g_clearedUnits}) {
    counter->store(0, std::memory_order_relaxed);
  }
}
} // namespace testing
} // namespace aurora::render::detail::ao::ssil_radiance

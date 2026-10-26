#include "taa.hpp"

#include "temporal_resolve.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"

#include "../../internal.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <array>
#include <atomic>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The resolve ports Bevy's taa.wgsl, its history filter MJP's Catmull-Rom gist, and the sharpening AMD FSR1's RCAS
namespace aurora::render::detail::taa {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::taa"};
constexpr TexRef Color{0};
constexpr TexRef HistoryTarget{4};
constexpr uint32_t SourceGroupIndex = 2;
constexpr size_t CacheCapacity = 6;
constexpr uint64_t LayoutSalt = 0x5441412D53524331ull;
constexpr std::array<std::string_view, 1> ResetDefines{"TAA_RESET"};
constexpr std::array<shaders::Constant, 3> ResolveConstants{{
    {"MIN_BLEND", MinBlend},
    {"MOVING_SAMPLES", MovingSamples},
    {"CLIP_SCALE", ClipScale},
}};
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

struct LatchSlot {
  bool live = false;
  std::array<uint32_t, MaxPassPipelines> ids{};
};

struct PipelineLayoutEntry {
  wgpu::BindGroupLayout passLayout;
  wgpu::PipelineLayout layout;
};

struct CacheEntry {
  wgpu::BindGroupLayout layout;
  wgpu::TextureView first;
  wgpu::TextureView second;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

PassId g_pass = InvalidPassId;

LatchSlot g_slot;
const CompiledGraph* g_latchGraph = nullptr;

std::mutex g_layoutMutex;
wgpu::BindGroupLayout g_sourceLayout;
std::vector<PipelineLayoutEntry> g_pipelineLayouts;

std::vector<CacheEntry> g_cache;
wgpu::Texture g_dummyTexture;
wgpu::TextureView g_dummyView;

std::atomic_bool g_clearOnlySeen{false};
std::atomic_uint32_t g_resolves{0};
std::atomic_uint32_t g_resets{0};
std::atomic_uint32_t g_skippedNoScene{0};
std::atomic_uint32_t g_groupsCreated{0};
std::atomic_uint32_t g_clearOnly{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  const Settings* settings = setup.settings;
  if (settings == nullptr || !settings->taa || settings->debugView != 0 || !setup.gx_available(GxResource::Motion) ||
      temporal_resolve::installed().pass != Pass) {
    builder.skip();
    return;
  }
  builder.after(phase::Taa);
  builder.use(resource::GxColor, Access::ColorRMW);
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::BindDummyBlack, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::Motion, Access::Sampled, UseOptions{.missing = Missing::DisablePass});
  builder.use(resource::Reactive, Access::Sampled, UseOptions{.missing = Missing::BindDummyBlack});
  builder.history(History, history_desc(), Access::ColorWrite);
}

bool enabled(const StageData& stage, const void*) {
  const bool scene = stage.markerSeen && !stage.flushedAtFinish && stage.sceneHints != nullptr &&
                     stage.sceneHints->has(SceneHasMainCamera);
  if (!scene) {
    bump(g_skippedNoScene);
  }
  return scene;
}

wgpu::PipelineLayout pipeline_layout(const PipelineSetup& setup) {
  if (!webgpu::g_device || !setup.frameLayout || !setup.passLayout) {
    return {};
  }
  std::lock_guard lock{g_layoutMutex};
  if (!g_sourceLayout) {
    std::array<wgpu::BindGroupLayoutEntry, 2> entries{};
    for (uint32_t i = 0; i < entries.size(); ++i) {
      entries[i] = {
          .binding = i,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "aurora.taa source",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_sourceLayout = webgpu::g_device.CreateBindGroupLayout(&descriptor);
  }
  for (const PipelineLayoutEntry& entry : g_pipelineLayouts) {
    if (entry.passLayout.Get() == setup.passLayout.Get()) {
      return entry.layout;
    }
  }
  const std::array groups{setup.frameLayout, setup.passLayout, g_sourceLayout};
  const wgpu::PipelineLayoutDescriptor descriptor{
      .label = "aurora.taa",
      .bindGroupLayoutCount = groups.size(),
      .bindGroupLayouts = groups.data(),
  };
  wgpu::PipelineLayout layout = webgpu::g_device.CreatePipelineLayout(&descriptor);
  g_pipelineLayouts.push_back({.passLayout = setup.passLayout, .layout = layout});
  return layout;
}

wgpu::BindGroupLayout source_layout() {
  std::lock_guard lock{g_layoutMutex};
  return g_sourceLayout;
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  if (setup.pass == nullptr || setup.pass->env == nullptr) {
    return;
  }
  const wgpu::PipelineLayout layout = pipeline_layout(setup);
  if (!layout) {
    return;
  }
  const uint64_t key = setup.layoutHash ^ LayoutSalt;
  shaders::RenderPipelineDesc resolve = testing::resolve_desc(false);
  resolve.layout = layout;
  resolve.layoutKey = key;
  shaders::RenderPipelineDesc reset = testing::resolve_desc(true);
  reset.layout = layout;
  reset.layoutKey = key;
  shaders::RenderPipelineDesc apply =
      testing::apply_desc(setup.pass->env->gxFormats[static_cast<size_t>(GxResource::Color)]);
  apply.layout = layout;
  apply.layoutKey = key;
  ids[0] = shaders::request_render(resolve).value;
  ids[1] = shaders::request_render(reset).value;
  ids[2] = shaders::request_render(apply).value;
}

const wgpu::TextureView& dummy_prev(const wgpu::Device& device) {
  if (!g_dummyView && device) {
    const wgpu::TextureDescriptor descriptor{
        .label = "aurora.taa empty history",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1, 1},
        .format = HistoryFormat,
    };
    g_dummyTexture = device.CreateTexture(&descriptor);
    g_dummyView = g_dummyTexture.CreateView();
  }
  return g_dummyView;
}

wgpu::BindGroup cached_group(const PassContext& ctx, const wgpu::BindGroupLayout& layout,
                             const wgpu::TextureView& first, const wgpu::TextureView& second) {
  const uint64_t frame = ctx.frame_id();
  for (CacheEntry& entry : g_cache) {
    if (entry.layout.Get() == layout.Get() && entry.first.Get() == first.Get() && entry.second.Get() == second.Get()) {
      entry.lastUsed = frame;
      return entry.group;
    }
  }
  if (g_cache.size() >= CacheCapacity) {
    g_cache.erase(std::ranges::min_element(g_cache, {}, &CacheEntry::lastUsed));
  }
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .textureView = first},
      wgpu::BindGroupEntry{.binding = 1, .textureView = second},
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "aurora.taa source",
      .layout = layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  CacheEntry& entry = g_cache.emplace_back(CacheEntry{
      .layout = layout,
      .first = first,
      .second = second,
      .group = ctx.device().CreateBindGroup(&descriptor),
      .lastUsed = frame,
  });
  bump(g_groupsCreated);
  return entry.group;
}

void draw(const PassContext& ctx, TexRef target, const wgpu::RenderPipeline& pipeline, const wgpu::BindGroup& group,
          const char* label) {
  const std::array<ColorTarget, 1> colors{{{.ref = target}}};
  const wgpu::RenderPassEncoder pass = ctx.begin_render(colors, nullptr, label);
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(SourceGroupIndex, group);
  pass.Draw(3);
  pass.End();
}

void encode_clear_only(const PassContext& ctx) {
  const std::array<ColorTarget, 1> colors{{{.ref = HistoryTarget}}};
  if (const wgpu::RenderPassEncoder pass = ctx.begin_render(colors, nullptr, "aurora.taa clear")) {
    pass.End();
  }
  bump(g_clearOnly);
  g_clearOnlySeen.store(true, std::memory_order_relaxed);
}

void encode(PassContext& ctx, const void*) {
  const StageData& stage = ctx.stage_data();
  const bool motionValid = stage.sceneHints != nullptr && stage.sceneHints->motionValid;
  const bool valid = ctx.prev_valid(HistoryTarget) && motionValid;
  const wgpu::RenderPipeline resolve = shaders::render_pipeline(ctx.pipeline(valid ? 0 : 1));
  const wgpu::RenderPipeline apply = shaders::render_pipeline(ctx.pipeline(2));
  const uint32_t samples = ctx.sample_count(Color);
  CHECK(samples == 1, "taa: gx.color has {} samples", samples);
  const wgpu::BindGroupLayout layout = source_layout();
  if (!resolve || !apply || samples != 1 || !layout) {
    encode_clear_only(ctx);
    return;
  }
  const wgpu::TextureView color = ctx.view(Color);
  const wgpu::TextureView history = ctx.view(HistoryTarget);
  wgpu::TextureView prev = valid ? ctx.prev(HistoryTarget) : wgpu::TextureView{};
  if (!prev) {
    prev = dummy_prev(ctx.device());
  }
  if (!color || !history || !prev) {
    encode_clear_only(ctx);
    return;
  }
  const wgpu::BindGroup resolveGroup = cached_group(ctx, layout, color, prev);
  const wgpu::BindGroup applyGroup = cached_group(ctx, layout, history, history);
  draw(ctx, HistoryTarget, resolve, resolveGroup, "aurora.taa resolve");
  draw(ctx, Color, apply, applyGroup, "aurora.taa apply");
  bump(g_resolves);
  if (!valid) {
    bump(g_resets);
  }
}

bool pipelines_live() noexcept {
  return g_slot.live && hints_enabled() && g_slot.ids[0] != 0 && shaders::pipelines_ready(g_slot.ids);
}
} // namespace

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::PrePost,
      .kind = PassKind::Encoder,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = request_pipelines,
  };
}

void initialize() {
  if (g_pass == InvalidPassId) {
    g_pass = register_pass(pass_desc());
  }
  temporal_resolve::install(nullptr);
}

void shutdown() noexcept {
  if (g_pass != InvalidPassId) {
    unregister_pass(g_pass);
    g_pass = InvalidPassId;
  }
  g_slot = {};
  g_latchGraph = nullptr;
  g_clearOnlySeen.store(false, std::memory_order_relaxed);
  g_cache.clear();
  g_dummyView = {};
  g_dummyTexture = {};
  std::lock_guard lock{g_layoutMutex};
  g_pipelineLayouts.clear();
  g_sourceLayout = {};
}

void worker_tick(uint64_t frameId) noexcept {
  if (g_cache.empty()) {
    return;
  }
  std::erase_if(g_cache, [frameId](const CacheEntry& entry) { return frameId > entry.lastUsed + RetainFrames; });
}

void on_latch(const Settings& settings, const CompiledGraph* graph) {
  if (g_pass != InvalidPassId) {
    const Params params{.sharpness = settings.taa ? rcas_strength(settings.taaSharpness) : 0.f};
    set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
  }
  if (graph != g_latchGraph) {
    g_latchGraph = graph;
    g_clearOnlySeen.store(false, std::memory_order_relaxed);
  }
  LatchSlot slot;
  if (graph != nullptr) {
    const NodeId node = graph->find_node(Pass);
    if (node != InvalidNode && graph->nodes[node].kind == NodeKind::Pass &&
        graph->nodes[node].status == CullReason::None) {
      slot.live = true;
      slot.ids = graph->nodes[node].pipelineIds;
    }
  }
  g_slot = slot;
}

bool wanted(const Settings& settings) noexcept { return settings.taa; }

bool jitter_enabled() noexcept {
  return g_slot.live && hints_enabled() && g_slot.ids[0] != 0 && !g_clearOnlySeen.load(std::memory_order_relaxed) &&
         shaders::pipelines_ready(g_slot.ids);
}

bool ready(const Settings& settings) noexcept { return !settings.taa || pipelines_live(); }

namespace testing {
Counters counters() noexcept {
  return {
      .resolves = g_resolves.load(std::memory_order_relaxed),
      .resets = g_resets.load(std::memory_order_relaxed),
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
      .groupsCreated = g_groupsCreated.load(std::memory_order_relaxed),
      .clearOnly = g_clearOnly.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cache.clear();
  g_clearOnlySeen.store(false, std::memory_order_relaxed);
  g_resolves.store(0, std::memory_order_relaxed);
  g_resets.store(0, std::memory_order_relaxed);
  g_skippedNoScene.store(0, std::memory_order_relaxed);
  g_groupsCreated.store(0, std::memory_order_relaxed);
  g_clearOnly.store(0, std::memory_order_relaxed);
}

shaders::RenderPipelineDesc apply_desc(wgpu::TextureFormat format) {
  shaders::RenderPipelineDesc desc{
      .label = "aurora.taa apply",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_apply",
      .colorCount = 1,
  };
  desc.colors[0] = {.format = format, .writeMask = RgbMask};
  return desc;
}

shaders::RenderPipelineDesc resolve_desc(bool reset) {
  shaders::RenderPipelineDesc desc{
      .label = reset ? "aurora.taa reset" : "aurora.taa resolve",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_resolve",
      .defines = reset ? std::span<const std::string_view>{ResetDefines} : std::span<const std::string_view>{},
      .constants = reset ? std::span<const shaders::Constant>{} : std::span<const shaders::Constant>{ResolveConstants},
      .colorCount = 1,
  };
  desc.colors[0] = {.format = HistoryFormat};
  return desc;
}
} // namespace testing
} // namespace aurora::render::detail::taa

#include "sun_visibility.hpp"

#include "../frame_state.hpp"
#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../lighting/deferred.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aurora::render::shadows::sun_visibility {
namespace {
using detail::StageData;

constexpr Module Log{"aurora::render::shadows"};
constexpr uint64_t CounterOffset = uint64_t{CounterFirst} * 4;
constexpr uint64_t CounterBytes = uint64_t{CounterWords} * 4;
constexpr int ShutdownWaitMs = 500;
constexpr size_t ListedFrames = 16;

struct Slot {
  wgpu::Buffer buffer;
  bool busy = false;
};

struct Latest {
  bool valid = false;
  uint64_t frame = 0;
  std::array<uint32_t, CounterWords> words{};
};

detail::PassId g_pass = detail::InvalidPassId;
std::optional<uint32_t> g_lightOverride;
std::optional<bool> g_frameOverride;
std::atomic_uint64_t g_producing{0};
std::atomic_uint64_t g_clearOnly{0};
std::mutex g_mutex;
std::array<Slot, ReadbackSlots> g_ring;
Counters g_counters;
Latest g_latest;
SunShadowState g_state;
std::vector<uint64_t> g_unshadowedFrames;

uint32_t word(const std::array<uint32_t, CounterWords>& words, uint32_t stat) noexcept {
  return words[stat - CounterFirst];
}

void record(const Latest& latest) {
  std::lock_guard lock{g_mutex};
  const auto& w = latest.words;
  ++g_counters.readbacks;
  if (word(w, vsm::stat::SunUnshadowed) != 0) {
    if (g_counters.unshadowedFrames++ == 0) {
      g_counters.firstUnshadowed = latest.frame;
    }
    if (g_unshadowedFrames.size() < ListedFrames) {
      g_unshadowedFrames.push_back(latest.frame);
    }
  }
  g_counters.samples += word(w, vsm::stat::SunSamples);
  g_counters.sampledUnshadowed += word(w, vsm::stat::SunSampledUnshadowed);
  g_counters.sampledFallback += word(w, vsm::stat::SunSampledFallback);
  g_counters.sampledStale += word(w, vsm::stat::SunSampledStale);
  g_counters.sampledOutOfRange += word(w, vsm::stat::SunSampledOutOfRange);
  if (!g_latest.valid || latest.frame >= g_latest.frame) {
    g_latest = latest;
  }
}

void copy_counters(const wgpu::CommandEncoder& encoder, const BufferView& stats, uint64_t frame) {
  if (!stats.buffer || stats.size < CounterOffset + CounterBytes) {
    return;
  }
  wgpu::Buffer buffer;
  size_t index = 0;
  {
    std::lock_guard lock{g_mutex};
    const auto it = std::ranges::find_if(g_ring, [](const Slot& slot) { return !slot.busy; });
    if (it == g_ring.end()) {
      return;
    }
    if (!it->buffer) {
      const wgpu::BufferDescriptor descriptor{
          .label = "sun.visibility counters",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = CounterBytes,
      };
      it->buffer = webgpu::g_device.CreateBuffer(&descriptor);
    }
    it->busy = true;
    buffer = it->buffer;
    index = static_cast<size_t>(it - g_ring.begin());
  }
  encoder.CopyBufferToBuffer(stats.buffer, stats.offset + CounterOffset, buffer, 0, CounterBytes);
  detail::queue_after_frame([buffer, index, frame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, CounterBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, index, frame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      Latest latest{.valid = true, .frame = frame};
                      const bool mapped = status == wgpu::MapAsyncStatus::Success;
                      if (mapped) {
                        if (const void* bytes = buffer.GetConstMappedRange(0, CounterBytes); bytes != nullptr) {
                          std::memcpy(latest.words.data(), bytes, CounterBytes);
                        }
                        buffer.Unmap();
                        record(latest);
                      }
                      std::lock_guard lock{g_mutex};
                      g_ring[index].busy = false;
                    });
  });
}

bool declarable(const detail::PassSetup& setup) noexcept {
  return vsm::active() && setup.env != nullptr && setup.env->retained && vsm::handle(vsm::Table).id != 0 &&
         vsm::handle(vsm::Pool).id != 0 && vsm::handle(vsm::ViewsName).id != 0 && vsm::handle(vsm::StatsName).id != 0;
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.use(
      resource::GxDepth, Access::Sampled,
      detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  builder.import_persistent_texture(vsm::Pool, vsm::handle(vsm::Pool), vsm::pool_desc(), Access::Sampled);
  builder.import_persistent(vsm::Table, vsm::handle(vsm::Table), Access::StorageRead);
  builder.import_persistent(vsm::ViewsName, vsm::handle(vsm::ViewsName), Access::Uniform);
  builder.import_persistent(vsm::StatsName, vsm::handle(vsm::StatsName), Access::StorageRW);
  builder.create(resource::SunVisibility, {.format = Format, .sizeClass = SizeClass::Efb}, Access::ColorWrite);
}

bool enabled(const StageData& stage, const void*) {
  return vsm::chain_enabled(stage) || (vsm::shadow_view_selected() && vsm::active());
}

void request_pipelines(const detail::PipelineSetup& setup, const void*,
                       std::span<uint32_t, detail::MaxPassPipelines> ids) {
  shaders::RenderPipelineDesc desc = pipeline_desc();
  desc.layout = setup.layout;
  desc.layoutKey = setup.layoutHash;
  const shaders::PipelineId id = shaders::request_render(desc);
  ids[0] = id.value;
  vsm::register_chain_pipeline(id);
}

void stage_params(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params;
  std::memcpy(&params, bytes.data(), sizeof(params));
  const detail::SceneHints* hints = stage.sceneHints.get();
  (void)detail::fill_stage_camera(hints, params.camera);
  (void)detail::fill_view_rotation(hints, params.rotation);
  params.flags = vsm::chain_enabled(stage) ? flag::Producing | flag::Counters : 0u;
  params.frame = hints != nullptr ? static_cast<uint32_t>(hints->hintFrame) : 0u;
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode(detail::PassContext& ctx, const void*) {
  Params params;
  std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
  const bool producing = (params.flags & flag::Producing) != 0 && (vsm::encoded_mask() & vsm::chain_bit::Update) != 0;
  const wgpu::RenderPipeline pipeline = producing ? shaders::render_pipeline(ctx.pipeline(0)) : wgpu::RenderPipeline{};
  const detail::ColorTarget target{.ref = TexRef{binding::Target}, .clear = wgpu::Color{1.0, 1.0, 1.0, 1.0}};
  const wgpu::RenderPassEncoder pass = ctx.begin_render(std::span{&target, 1}, nullptr);
  if (!pass) {
    return;
  }
  if (pipeline) {
    pass.SetPipeline(pipeline);
    pass.Draw(3);
  }
  pass.End();
  if (!pipeline) {
    g_clearOnly.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  g_producing.fetch_add(1, std::memory_order_relaxed);
  const StageData& stage = ctx.stage_data();
  copy_counters(ctx.encoder(), ctx.buffer(BufRef{binding::Stats}),
                stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0);
}

float share(uint32_t part, uint32_t whole) noexcept {
  return whole != 0 ? static_cast<float>(part) / static_cast<float>(whole) : 0.f;
}
} // namespace

shaders::RenderPipelineDesc pipeline_desc() {
  shaders::RenderPipelineDesc desc{
      .label = Pass,
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .colorCount = 1,
  };
  desc.colors[0] = {.format = Format};
  return desc;
}

Params latched_params(const Settings* settings) noexcept {
  const vsm::FilterParams filter =
      vsm::filter_params(settings != nullptr ? settings->shadows.quality : Quality::High, vsm::env_bias());
  Params params;
  params.taps = filter.taps;
  params.normalScale = filter.normalScale;
  params.depthBias = filter.depthBias;
  return params;
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Render,
      .cls = detail::PassClass::Required,
      .order = Order,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = stage_params,
  };
}

void initialize() {
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
}

void shutdown() noexcept {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  std::unique_lock lock{g_mutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{ShutdownWaitMs};
  const auto pending = [] { return std::ranges::any_of(g_ring, [](const Slot& slot) { return slot.busy; }); };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  const Counters c = g_counters;
  std::vector<uint64_t> listed = std::move(g_unshadowedFrames);
  g_unshadowedFrames.clear();
  g_ring = {};
  lock.unlock();
  const uint64_t producing = g_producing.load(std::memory_order_relaxed);
  const uint64_t clearOnly = g_clearOnly.load(std::memory_order_relaxed);
  if (producing != 0 || clearOnly != 0) {
    std::string frames;
    for (const uint64_t frame : listed) {
      frames += frames.empty() ? fmt::format("{}", frame) : fmt::format(" {}", frame);
    }
    const auto mean = [&](uint64_t part) {
      return c.samples != 0 ? static_cast<double>(part) / static_cast<double>(c.samples) : 0.0;
    };
    Log.info(
        "VSM: sun visibility drew {} frames and cleared {}; unshadowed on {} of {} read frames (hint frames "
        "[{}]); sampled shares: unshadowed {:.6f}, fallback {:.6f}, stale {:.6f}, out of range {:.6f} over {} "
        "samples",
        producing, clearOnly, c.unshadowedFrames, c.readbacks, frames, mean(c.sampledUnshadowed),
        mean(c.sampledFallback), mean(c.sampledStale), mean(c.sampledOutOfRange), c.samples);
  }
  testing::reset();
}

void on_latch(const Settings* settings) noexcept {
  if (g_pass != detail::InvalidPassId) {
    if (vsm::active()) {
      const Params params = latched_params(settings);
      (void)detail::set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
    } else {
      detail::clear_pass_params(g_pass);
    }
  }
  const vsm::SunPlan& plan = vsm::plan();
  SunShadowState next{
      .producing = plan.producing,
      .consumers = vsm::sun_term_consumes() || vsm::shadow_view_selected(),
      .lightId = plan.lightId,
      .lightFlags = plan.lightFlags,
      .elevationDeg = plan.elevationDeg,
      .levels = plan.producing ? vsm::SunLevels : 0u,
      .idle = plan.idle,
  };
  Latest latest;
  {
    std::lock_guard lock{g_mutex};
    latest = g_latest;
  }
  if (latest.valid) {
    const auto& w = latest.words;
    const uint32_t samples = word(w, vsm::stat::SunSamples);
    next.unshadowed = word(w, vsm::stat::SunUnshadowed) != 0;
    next.unshadowedShare = share(word(w, vsm::stat::SunSampledUnshadowed), samples);
    next.fallbackShare = share(word(w, vsm::stat::SunSampledFallback), samples);
    next.staleShare = share(word(w, vsm::stat::SunSampledStale), samples);
    next.outOfRangeShare = share(word(w, vsm::stat::SunSampledOutOfRange), samples);
    next.statsFrame = latest.frame;
  }
  g_state = next;
}

bool frame_enabled(const StageData& stage) noexcept { return g_frameOverride.value_or(vsm::chain_enabled(stage)); }

bool frame_applied(const StageData& stage) noexcept {
  const detail::CompiledGraph* graph = stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
  if (graph == nullptr || stage.stage != Stage::OpaqueDone || !frame_enabled(stage)) {
    return false;
  }
  const detail::ResourceId id = graph->find_resource(resource::SunVisibility);
  if (id == detail::InvalidResource) {
    return false;
  }
  const detail::NodeId creator = graph->resources[id].creator;
  return creator != detail::InvalidNode && graph->nodes[creator].programIndex != detail::InvalidNode &&
         stage.is_enabled(graph->nodes[creator].programIndex);
}

uint32_t light_index(const StageData& stage) noexcept {
  if (g_lightOverride) {
    return *g_lightOverride;
  }
  if (!frame_enabled(stage) || !vsm::sun_term_consumes()) {
    return detail::lighting::deferred::NoSunLight;
  }
  const detail::SceneHints* hints = stage.sceneHints.get();
  const uint32_t id = vsm::plan().lightId;
  if (hints != nullptr) {
    for (size_t i = 0; i < hints->lights.size(); ++i) {
      if (hints->lights[i].id == id && hints->lights[i].type == AURORA_LIGHT_DIRECTIONAL) {
        return static_cast<uint32_t>(i);
      }
    }
  }
  return detail::lighting::deferred::NoSunLight;
}

SunShadowState state() noexcept { return g_state; }

namespace testing {
void reset() noexcept {
  g_lightOverride.reset();
  g_frameOverride.reset();
  g_producing.store(0, std::memory_order_relaxed);
  g_clearOnly.store(0, std::memory_order_relaxed);
  std::lock_guard lock{g_mutex};
  g_counters = {};
  g_latest = {};
  g_state = {};
  g_unshadowedFrames.clear();
}

void set_light_index(std::optional<uint32_t> index) noexcept { g_lightOverride = index; }

void set_frame_enabled(std::optional<bool> enabled) noexcept { g_frameOverride = enabled; }

Counters counters() noexcept {
  std::lock_guard lock{g_mutex};
  Counters c = g_counters;
  c.producing = g_producing.load(std::memory_order_relaxed);
  c.clearOnly = g_clearOnly.load(std::memory_order_relaxed);
  return c;
}
} // namespace testing
} // namespace aurora::render::shadows::sun_visibility

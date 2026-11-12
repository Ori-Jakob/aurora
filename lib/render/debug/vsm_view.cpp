#include "vsm_view.hpp"

#include "debug.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../shadows/sun_visibility.hpp"
#include "../shadows/vsm_alloc.hpp"
#include "../shadows/vsm_check.hpp"
#include "../shadows/vsm_cull.hpp"
#include "../../webgpu/gpu.hpp"

#include <fmt/format.h>
#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <span>
#include <thread>

namespace aurora::render::debug::vsm_view {
namespace {
namespace vsm = shadows::vsm;
using detail::StageData;

constexpr int ShutdownWaitMs = 500;
constexpr ImVec4 WarningColor{1.f, 0.4f, 0.4f, 1.f};
constexpr std::array<shaders::Constant, VsmViewModeCount> ModeConstants{{
    {"VSM_VIEW_MODE", 0},
    {"VSM_VIEW_MODE", 1},
    {"VSM_VIEW_MODE", 2},
    {"VSM_VIEW_MODE", 3},
    {"VSM_VIEW_MODE", 4},
}};

struct Slot {
  wgpu::Buffer buffer;
  bool busy = false;
};

detail::PassId g_pass = detail::InvalidPassId;
std::atomic_uint64_t g_producing{0};
std::atomic_uint64_t g_noSun{0};
std::mutex g_mutex;
std::array<Slot, ReadbackSlots> g_ring;
Shown g_shown;

void publish(const Shown& next) {
  std::lock_guard lock{g_mutex};
  if (!g_shown.valid || next.frame >= g_shown.frame) {
    g_shown = next;
  }
}

void copy_stats(const wgpu::CommandEncoder& encoder, const BufferView& stats, uint64_t frame) {
  if (!stats.buffer || stats.size < vsm::StatsBytes) {
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
          .label = "vsm.view stats",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = vsm::StatsBytes,
      };
      it->buffer = webgpu::g_device.CreateBuffer(&descriptor);
    }
    it->busy = true;
    buffer = it->buffer;
    index = static_cast<size_t>(it - g_ring.begin());
  }
  encoder.CopyBufferToBuffer(stats.buffer, stats.offset, buffer, 0, vsm::StatsBytes);
  detail::queue_after_frame([buffer, index, frame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, vsm::StatsBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, index, frame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      if (status == wgpu::MapAsyncStatus::Success) {
                        Shown next{.valid = true, .frame = frame};
                        if (const void* bytes = buffer.GetConstMappedRange(0, vsm::StatsBytes); bytes != nullptr) {
                          std::memcpy(next.stats.data(), bytes, vsm::StatsBytes);
                        }
                        buffer.Unmap();
                        publish(next);
                      }
                      std::lock_guard lock{g_mutex};
                      g_ring[index].busy = false;
                    });
  });
}

bool declarable(const detail::PassSetup& setup) noexcept {
  return vsm::active() && setup.env != nullptr && setup.env->retained && vsm::handle(vsm::Table).id != 0 &&
         vsm::handle(vsm::Phys).id != 0 && vsm::handle(vsm::Pool).id != 0 && vsm::handle(vsm::ViewsName).id != 0 &&
         vsm::handle(vsm::StatsName).id != 0;
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  const std::optional<VsmViewMode> mode = selected_mode(setup.debug_view());
  if (!mode || !declarable(setup)) {
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
  builder.import_persistent(vsm::Phys, vsm::handle(vsm::Phys), Access::StorageRead);
  // Without a sun the chain is off and these transients are never written, but the slate fill still runs
  builder.use_buffer(vsm::Requests, Access::StorageRead, Missing::BindDummyBlack);
  builder.use_buffer(vsm::RenderList, Access::StorageRead, Missing::BindDummyBlack);
  builder.import_persistent(vsm::ViewsName, vsm::handle(vsm::ViewsName), Access::Uniform);
  builder.import_persistent(vsm::StatsName, vsm::handle(vsm::StatsName), Access::StorageRead);
  builder.create(VsmViewResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                 Access::StorageWrite);
  const auto value = static_cast<uint32_t>(*mode);
  std::memcpy(builder.local().data(), &value, sizeof(value));
}

bool enabled(const StageData&, const void*) { return vsm::active(); }

void request_pipelines(const detail::PipelineSetup& setup, const void*,
                       std::span<uint32_t, detail::MaxPassPipelines> ids) {
  uint32_t value = 0;
  std::memcpy(&value, setup.local.data(), sizeof(value));
  const auto mode = static_cast<VsmViewMode>(std::min(value, static_cast<uint32_t>(VsmViewMode::NoSun)));
  for (const auto [slot, want] : {std::pair{0u, mode}, std::pair{1u, VsmViewMode::NoSun}}) {
    shaders::ComputePipelineDesc desc = pipeline_desc(want);
    desc.layout = setup.layout;
    desc.layoutKey = setup.layoutHash;
    ids[slot] = shaders::request_compute(desc).value;
  }
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
  params.flags = vsm::chain_enabled(stage) ? flag::Producing : 0u;
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode(detail::PassContext& ctx, const void*) {
  Params params;
  std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
  const bool producing = (params.flags & flag::Producing) != 0 && (vsm::encoded_mask() & vsm::chain_bit::Update) != 0;
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(producing ? 0 : 1));
  const wgpu::Extent3D extent = ctx.extent(TexRef{binding::Target});
  if (!pipeline || extent.width == 0 || extent.height == 0) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups((extent.width + Workgroup - 1) / Workgroup, (extent.height + Workgroup - 1) / Workgroup);
  pass.End();
  if (!producing) {
    g_noSun.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  g_producing.fetch_add(1, std::memory_order_relaxed);
  const StageData& stage = ctx.stage_data();
  copy_stats(ctx.encoder(), ctx.buffer(BufRef{binding::Stats}),
             stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0);
}

float share(float value) noexcept { return value * 100.f; }

void text(std::string_view value) { ImGui::TextUnformatted(value.data(), value.data() + value.size()); }

void draw_pages(const Shown& shown) {
  if (!shown.valid) {
    ImGui::TextUnformatted("No page counts read back yet");
    return;
  }
  const auto& s = shown.stats;
  ImGui::Text("Frame %llu: %u requested, %u cached hits, %u rendered (%u priority, %u new, %u reused)",
              static_cast<unsigned long long>(shown.frame), s[vsm::stat::Requested], s[vsm::stat::Hits],
              s[vsm::stat::Rendered], s[vsm::stat::NeedPriority], s[vsm::stat::NeedNew], s[vsm::stat::NeedReuse]);
  ImGui::Text("Deferred: %u new (coarser), %u stale; pool overflow %u, evicted %u", s[vsm::stat::DeferredNew],
              s[vsm::stat::DeferredReuse], s[vsm::stat::OverflowPool], s[vsm::stat::Evicted]);
  if (s[vsm::stat::PlayerDeferred] != 0) {
    ImGui::TextColored(WarningColor, "Player pages deferred: %u", s[vsm::stat::PlayerDeferred]);
  } else {
    ImGui::TextUnformatted("Player pages deferred: 0");
  }
  ImGui::Text("Pairs: %u opaque, %u cutout, %u pages skipped at the cap", s[vsm::stat::OpaquePairs],
              s[vsm::stat::CutoutPairs], s[vsm::stat::SkippedPages]);
}

void draw_check() {
  if (!vsm::check::config().on) {
    return;
  }
  const vsm::check::Result r = vsm::check::result();
  const ImVec4 color = r.failures != 0 || r.rasterMismatched != 0 ? WarningColor : ImVec4{1.f, 1.f, 1.f, 1.f};
  ImGui::TextColored(color, "CHECK: %llu readbacks, %llu failures; raster %llu pages, %llu mismatched",
                     static_cast<unsigned long long>(r.readbacks), static_cast<unsigned long long>(r.failures),
                     static_cast<unsigned long long>(r.rasterPages),
                     static_cast<unsigned long long>(r.rasterMismatched));
}

void draw_timings() {
  float total = 0.f;
  bool any = false;
  if (!ImGui::BeginTable("##vsm_ms", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
    return;
  }
  ImGui::TableSetupColumn("Pass");
  ImGui::TableSetupColumn("GPU ms");
  ImGui::TableHeadersRow();
  for (const PassTiming& timing : pass_timings()) {
    const std::string_view name = timing.name;
    if (!name.starts_with("vsm.") && name != shadows::sun_visibility::Pass) {
      continue;
    }
    any = true;
    total += timing.gpuMs;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    text(name);
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", timing.gpuMs);
  }
  if (any) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted("Total");
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", total);
  }
  ImGui::EndTable();
  if (!any) {
    ImGui::TextUnformatted("No GPU timings for the shadow passes");
  }
}
} // namespace

std::optional<VsmViewMode> row_mode(const GbufView& row) noexcept {
  if (row.resource != VsmViewResource || row.id < FirstVsmView ||
      row.id >= FirstVsmView + static_cast<uint16_t>(VsmViewMode::NoSun)) {
    return std::nullopt;
  }
  return static_cast<VsmViewMode>(row.id - FirstVsmView);
}

std::optional<VsmViewMode> selected_mode(uint16_t view) noexcept {
  const GbufView* row = find_gbuf_view(view);
  return row != nullptr ? row_mode(*row) : std::nullopt;
}

std::string reason_text(const vsm::SunPlan& plan, bool consumed, bool keyRoomOnly) {
  if (plan.producing) {
    return consumed ? fmt::format("light {} at {:.1f} deg", plan.lightId, plan.elevationDeg)
                    : std::string{"no consumer"};
  }
  switch (plan.idle) {
  case vsm::Idle::Inactive:
    return "real-time shadows inactive";
  case vsm::Idle::Pipelines:
    return "pipelines pending";
  case vsm::Idle::NoCamera:
    return "no main camera";
  case vsm::Idle::NoSun:
    return keyRoomOnly ? "KEY_ROOM only" : "no directional";
  case vsm::Idle::SunDisabled:
    return "the environment has no sun";
  case vsm::Idle::LowSun:
    return fmt::format("below {:.0f} deg (at {:.1f} deg)", vsm::LowSunOnDegrees, plan.elevationDeg);
  case vsm::Idle::None:
    break;
  }
  return "unknown";
}

shaders::ComputePipelineDesc pipeline_desc(VsmViewMode mode) {
  const auto index = std::min<size_t>(static_cast<size_t>(mode), ModeConstants.size() - 1);
  return {
      .label = Pass,
      .shader = Shader,
      .entryPoint = "view_main",
      .constants = std::span{&ModeConstants[index], 1},
  };
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Optional,
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
  g_ring = {};
  g_shown = {};
  lock.unlock();
  g_producing.store(0, std::memory_order_relaxed);
  g_noSun.store(0, std::memory_order_relaxed);
}

void on_latch(const Settings* settings) noexcept {
  if (g_pass == detail::InvalidPassId) {
    return;
  }
  if (!vsm::active()) {
    detail::clear_pass_params(g_pass);
    return;
  }
  const vsm::FilterParams filter =
      vsm::filter_params(settings != nullptr ? settings->shadows.quality : Quality::High, vsm::env_bias());
  Params params;
  params.taps = filter.taps;
  params.normalScale = filter.normalScale;
  params.depthBias = filter.depthBias;
  (void)detail::set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
}

Shown shown() noexcept {
  std::lock_guard lock{g_mutex};
  return g_shown;
}

void draw_stats() {
  const shadows::sun_visibility::SunShadowState state = shadows::sun_visibility::state();
  const vsm::SunPlan& plan = vsm::plan();
  const std::string reason = reason_text(plan, state.consumers, vsm::key_room_only());
  ImGui::Indent();
  if (plan.producing && state.consumers) {
    ImGui::Text("Sun: %s, flags 0x%x", reason.c_str(), plan.lightFlags);
  } else {
    ImGui::TextColored(WarningColor, "Sun shadows off: %s", reason.c_str());
  }
  draw_pages(shown());
  if (state.statsFrame != 0) {
    ImGui::Text(
        "Receivers (1 in 64, frame %llu): unshadowed %.3f%%, fallback %.3f%%, stale %.3f%%, out of range "
        "%.3f%%",
        static_cast<unsigned long long>(state.statsFrame), share(state.unshadowedShare), share(state.fallbackShare),
        share(state.staleShare), share(state.outOfRangeShare));
    if (state.unshadowed) {
      ImGui::TextColored(WarningColor, "Some receivers found no usable page");
    }
  }
  ImGui::Text("Pool: %u pages, generation %u", vsm::pool_pages(), vsm::pool_generation());
  draw_check();
  draw_timings();
  ImGui::Unindent();
}

namespace testing {
void reset() noexcept {
  g_producing.store(0, std::memory_order_relaxed);
  g_noSun.store(0, std::memory_order_relaxed);
  std::lock_guard lock{g_mutex};
  g_shown = {};
}

Counters counters() noexcept {
  return {.producing = g_producing.load(std::memory_order_relaxed), .noSun = g_noSun.load(std::memory_order_relaxed)};
}
} // namespace testing
} // namespace aurora::render::debug::vsm_view

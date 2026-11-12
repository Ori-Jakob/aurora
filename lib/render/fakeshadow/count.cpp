#include "count.hpp"

#include "gate.hpp"
#include "lsm.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../settings.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"
#include "../shadows/vsm_pages.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <thread>

namespace aurora::render::fakeshadow::count {
namespace {
constexpr Module Log{"aurora::render::fakeshadow"};
constexpr detail::TexRef Normal{3};
constexpr detail::BufRef Counters{5};
constexpr int FlushWaitMs = 500;

struct Slot {
  wgpu::Buffer buffer;
  bool busy = false;
};

detail::PassId g_pass = detail::InvalidPassId;
std::optional<Mode> g_modeOverride;
std::mutex g_mutex;
std::array<Slot, ReadbackSlots> g_ring;
Counts g_totals;

bool consumes(const detail::StageData&) noexcept { return active(); }

void record(uint64_t frame, const std::array<uint32_t, WordCount>& words) {
  {
    std::lock_guard lock{g_mutex};
    ++g_totals.frames;
    g_totals.ddPbr += words[WordDdPbr];
    g_totals.ddLegacy += words[WordDdLegacy];
    g_totals.touched += words[WordTouched];
    g_totals.shadowed += words[WordShadowed];
  }
  gate::note_dd_pixels(frame, words[WordDdPbr], words[WordDdLegacy]);
}

void copy_counts(const wgpu::CommandEncoder& encoder, const BufferView& counts, uint64_t frame) {
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
          .label = "fakeshadow.count readback",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = WordBytes,
      };
      it->buffer = webgpu::g_device.CreateBuffer(&descriptor);
    }
    it->busy = true;
    buffer = it->buffer;
    index = static_cast<size_t>(it - g_ring.begin());
  }
  encoder.CopyBufferToBuffer(counts.buffer, counts.offset, buffer, 0, WordBytes);
  detail::queue_after_frame([buffer, index, frame] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, WordBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, index, frame](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      if (status == wgpu::MapAsyncStatus::Success) {
                        std::array<uint32_t, WordCount> words{};
                        if (const void* bytes = buffer.GetConstMappedRange(0, WordBytes); bytes != nullptr) {
                          std::memcpy(words.data(), bytes, WordBytes);
                          buffer.Unmap();
                          record(frame, words);
                        } else {
                          buffer.Unmap();
                        }
                      }
                      std::lock_guard lock{g_mutex};
                      g_ring[index].busy = false;
                    });
  });
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!active() || setup.settings == nullptr || setup.env == nullptr || !lsm::declared(setup) ||
      !setup.gx_available(detail::GxResource::Normal) || !setup.gx_available(detail::GxResource::Fog)) {
    builder.skip();
    return;
  }
  builder.after(phase::Lighting);
  // Without a mask nothing kept can darken twice
  builder.use(lsm::Resource, Access::Sampled, Missing::DisablePass);
  builder.use(resource::SunVisibility, Access::Sampled, Missing::BindDummyWhite);
  builder.use(resource::SceneHdr, Access::Sampled, Missing::BindDummyBlack);
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  builder.use(resource::Fog, Access::Sampled,
              detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.create_buffer(Buffer,
                        {.size = WordBytes, .extraUsage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst},
                        Access::StorageWrite);
  // Its only reader is the readback after submit
  builder.side_effect();
}

bool enabled(const detail::StageData& stage, const void*) {
  return active() && !stage.flushedAtFinish && lsm::frame_enabled(stage);
}

void request_pipelines(const detail::PipelineSetup& setup, const void*,
                       std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = Pass,
                                        .shader = Shader,
                                        .entryPoint = "count",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void fill_stage(const detail::StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params;
  // V counts only where the composite really applies it, so the counter sees the frame the screen shows
  if (detail::composite::stage_sun(stage, params.sun)) {
    params.flags |= FlagSun;
  }
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode(detail::PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView counts = ctx.buffer(Counters);
  if (!pipeline || !counts.buffer || counts.size < WordBytes) {
    return;
  }
  ctx.encoder().ClearBuffer(counts.buffer, counts.offset, WordBytes);
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const wgpu::Extent3D size = ctx.extent(Normal);
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups((size.width + GroupSize - 1) / GroupSize, (size.height + GroupSize - 1) / GroupSize);
  pass.End();
  const detail::StageData& stage = ctx.stage_data();
  copy_counts(ctx.encoder(), counts, stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0);
}
} // namespace

Mode mode() noexcept {
  if (g_modeOverride) {
    return *g_modeOverride;
  }
  switch (detail::game_shadows_env()) {
  case detail::GameShadowsEnv::Count:
    return Mode::Rev3;
  case detail::GameShadowsEnv::CountAll:
    return Mode::All;
  default:
    return Mode::Off;
  }
}

bool active() noexcept { return mode() != Mode::Off; }

bool selects(Mode mode, ShadowDecision decision) noexcept {
  switch (mode) {
  case Mode::All:
    return true;
  case Mode::Rev3:
    return decision == ShadowDecision::KeepCaster || decision == ShadowDecision::KeepCasterUnknown ||
           decision == ShadowDecision::KeepOverflow;
  case Mode::Off:
    break;
  }
  return false;
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = detail::PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = Order,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_stage,
  };
}

void initialize() {
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
  shadows::vsm::register_consumer_term(&consumes);
}

void shutdown() noexcept {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  flush();
  const Counts c = totals();
  if (c.frames != 0) {
    Log.info(
        "Fake-shadow count: {} frames read, double-darkened pixels {} PBR and {} legacy, {} under kept fakes, "
        "{} under real shadow",
        c.frames, c.ddPbr, c.ddLegacy, c.touched, c.shadowed);
  }
  std::lock_guard lock{g_mutex};
  g_ring = {};
  g_totals = {};
}

void flush() noexcept {
  std::unique_lock lock{g_mutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{FlushWaitMs};
  const auto pending = [] { return std::ranges::any_of(g_ring, [](const Slot& slot) { return slot.busy; }); };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
}

Counts totals() noexcept {
  std::lock_guard lock{g_mutex};
  return g_totals;
}

namespace testing {
void set_mode(std::optional<Mode> mode) noexcept { g_modeOverride = mode; }

void reset() noexcept {
  g_modeOverride.reset();
  std::lock_guard lock{g_mutex};
  g_totals = {};
}
} // namespace testing
} // namespace aurora::render::fakeshadow::count

#include "lsm.hpp"

#include "../debug/views.hpp"
#include "../frame_state.hpp"
#include "../pass_context.hpp"
#include "../post/composite.hpp"
#include "../scene/capture_list.hpp"
#include "../scene/classifier.hpp"
#include "../stage_data.hpp"
#include "../../gfx/frame_packet.hpp"
#include "../../gx/gx.hpp"
#include "../../logging.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <span>
#include <string>

namespace aurora::render::fakeshadow::lsm {
namespace {
constexpr Module Log{"aurora::render::fakeshadow"};
constexpr detail::TexRef Mask{0};
constexpr detail::TexRef Depth{1};

struct Counters {
  std::atomic_uint64_t latches, declaredLatches, consumedLatches, selected, depthRefused, frames, replayed,
      postShadowDraws;
} g_counts;

constexpr std::array CounterList{&Counters::latches,  &Counters::declaredLatches, &Counters::consumedLatches,
                                 &Counters::selected, &Counters::depthRefused,    &Counters::frames,
                                 &Counters::replayed, &Counters::postShadowDraws};

// Its address tells the capture list's slot that this module still owns it
char g_owner = 0;
detail::PassId g_pass = detail::InvalidPassId;
scene::capture_list::ConsumerId g_consumer = scene::capture_list::InvalidConsumer;
bool g_initialized = false;
wgpu::TextureFormat g_depthFormat = wgpu::TextureFormat::Undefined;
std::atomic_bool g_consumed{false};
std::atomic_bool g_consumedFifo{false};
std::atomic_uint64_t g_graphHash{0};
// Clones pend while their mask pipelines compile, so only pends after the first latches point at a cold cache
std::atomic_uint64_t g_pendingAtWarmUp{0};
std::optional<bool> g_consumedOverride;

// Only one thread writes each, so a plain add keeps lock prefixes off the draw path
inline void bump(std::atomic_uint64_t& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

bool consumer_select(const scene::DrawClass& cls, void*) noexcept {
  if (!select_draw(cls)) {
    return false;
  }
  // The mask attaches the scene depth read-only, which a depth-writing pipeline may not draw into
  const gx::GXState& state = gx::g_gxState;
  if (state.depthCompare && state.depthUpdate) {
    bump(g_counts.depthRefused);
    return false;
  }
  bump(g_counts.selected);
  return true;
}

scene::capture_list::Consumer consumer_desc(wgpu::TextureFormat depth) noexcept {
  return {
      .name = ConsumerName.data(),
      .select = consumer_select,
      .layout = layout(depth),
      .user = &g_owner,
  };
}

void ensure_registered(wgpu::TextureFormat depth) noexcept {
  const scene::capture_list::Consumer desc = consumer_desc(depth);
  if (g_consumer != scene::capture_list::InvalidConsumer && scene::capture_list::owns(g_consumer, desc)) {
    if (depth != g_depthFormat) {
      scene::capture_list::set_layout(g_consumer, desc.layout);
    }
  } else {
    g_consumer = scene::capture_list::register_consumer(desc);
    if (g_consumer == scene::capture_list::InvalidConsumer) {
      Log.warn("LSM: the capture list refused its consumer, so kept game shadows stay off PBR pixels");
    }
  }
  g_depthFormat = depth;
}

bool node_live(const detail::CompiledGraph& graph, std::string_view name) noexcept {
  const detail::NodeId node = graph.find_node(name);
  return node != detail::InvalidNode && graph.nodes[node].kind == detail::NodeKind::Pass &&
         graph.nodes[node].status == detail::CullReason::None;
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declared(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Capture);
  builder.before(phase::Geometry);
  builder.create(Resource,
                 {.format = Format, .sizeClass = SizeClass::Efb, .extraUsage = wgpu::TextureUsage::TextureBinding},
                 Access::ColorWrite);
  builder.use(resource::GxDepth, Access::DepthRead);
}

bool enabled(const detail::StageData& stage, const void*) {
  return consumed_fifo() && !stage.flushedAtFinish && scene::capture_list::queued(g_consumer) != 0;
}

void fill_stage(const detail::StageData&, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  // The fakes' alpha passes count from the EFB's clear alpha, so the mask starts from this frame's
  const Params params{.clearAlpha = std::clamp(gx::g_gxState.clearColor[3], 0.f, 1.f)};
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void encode(detail::PassContext& ctx, const void*) {
  Params params;
  const std::span<const std::byte> bytes = ctx.params();
  std::memcpy(&params, bytes.data(), std::min(bytes.size(), sizeof(params)));
  const detail::ColorTarget color{.ref = Mask, .clear = wgpu::Color{1.0, 1.0, 1.0, params.clearAlpha}};
  const detail::DepthTarget depth{.ref = Depth};
  const wgpu::RenderPassEncoder pass = ctx.begin_render(std::span{&color, 1}, &depth, "lsm.mask replay");
  if (!pass) {
    return;
  }
  scene::capture_list::encode(ctx, pass, g_consumer);
  pass.End();
  bump(g_counts.frames);
  const auto& lists = ctx.stage_data().captureList;
  if (lists != nullptr && g_consumer < scene::capture_list::MaxConsumers) {
    bump(g_counts.replayed, lists->lists[g_consumer].draws);
  }
}
} // namespace

void initialize() {
  if (g_initialized) {
    return;
  }
  g_initialized = true;
  ensure_registered(wgpu::TextureFormat::Undefined);
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
}

void shutdown() noexcept {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  if (g_consumer != scene::capture_list::InvalidConsumer &&
      scene::capture_list::owns(g_consumer, consumer_desc(g_depthFormat))) {
    scene::capture_list::unregister_consumer(g_consumer);
  }
  const bool ran = g_initialized && g_counts.declaredLatches.load(std::memory_order_relaxed) != 0;
  g_initialized = false;
  g_consumer = scene::capture_list::InvalidConsumer;
  g_depthFormat = wgpu::TextureFormat::Undefined;
  g_consumed.store(false, std::memory_order_relaxed);
  g_consumedFifo.store(false, std::memory_order_relaxed);
  if (ran) {
    const Stats s = stats();
    Log.info(
        "LSM: mask drawn on {} frames, consumed on {} of {} declared latches; {} clones replayed, {} selected, "
        "{} refused for depth writes, {} capture-list clones pending ({} after warm-up), {} post-shadow draws "
        "(graph 0x{:016x})",
        s.frames, s.consumedLatches, s.declaredLatches, s.replayed, s.selected, s.depthRefused, s.pending,
        s.pendingAfterWarmUp, s.postShadowDraws, s.graphHash);
  }
  for (const auto member : CounterList) {
    (g_counts.*member).store(0, std::memory_order_relaxed);
  }
  g_graphHash.store(0, std::memory_order_relaxed);
  g_pendingAtWarmUp.store(0, std::memory_order_relaxed);
}

void on_latch(const detail::PassSetup& latched, const detail::CompiledGraph* graph) {
  if (!g_initialized) {
    return;
  }
  const wgpu::TextureFormat depth = latched.env != nullptr
                                        ? latched.env->gxFormats[static_cast<size_t>(detail::GxResource::Depth)]
                                        : wgpu::TextureFormat::Undefined;
  ensure_registered(depth);
  const bool classView = debug::find_view(latched.debug_view()) != nullptr;
  // Latched FlagPbr already excludes lb = 1, identity and class views, so every lb >= 1 path stays clone-free
  const bool declaredNow = graph != nullptr && node_live(*graph, Pass);
  bool consumed = declaredNow && g_consumer != scene::capture_list::InvalidConsumer &&
                  detail::composite::exposure_units() && !detail::composite::identity_forced() && !classView;
  consumed = g_consumedOverride.value_or(consumed);
  scene::capture_list::set_enabled(g_consumer, consumed);
  g_consumed.store(consumed, std::memory_order_relaxed);
  g_graphHash.store(graph != nullptr ? graph->graphHash : 0, std::memory_order_relaxed);
  bump(g_counts.latches);
  if (g_counts.latches.load(std::memory_order_relaxed) == WarmUpLatches) {
    g_pendingAtWarmUp.store(scene::capture_list::stats().pending, std::memory_order_relaxed);
  }
  if (declaredNow) {
    bump(g_counts.declaredLatches);
  }
  if (consumed) {
    bump(g_counts.consumedLatches);
  }
}

void apply_frame_state() noexcept {
  g_consumedFifo.store(g_consumedOverride.value_or(g_consumed.load(std::memory_order_relaxed)),
                       std::memory_order_relaxed);
}

bool declared(const detail::PassSetup& setup) noexcept {
  if (setup.settings == nullptr || setup.env == nullptr || !setup.settings->pbr || setup.env->gxSampleCount != 1) {
    return false;
  }
  constexpr auto DepthIndex = static_cast<size_t>(detail::GxResource::Depth);
  return setup.gx_available(detail::GxResource::Surface) && setup.gx_available(detail::GxResource::Depth) &&
         (setup.env->gxUsage[DepthIndex] & wgpu::TextureUsage::RenderAttachment);
}

bool consumed() noexcept { return g_consumed.load(std::memory_order_relaxed); }

gfx::RenderTargetLayout layout(wgpu::TextureFormat depth) noexcept {
  gfx::RenderTargetLayout out{
      .colorAttachmentCount = 1,
      .depthStencilFormat = depth,
      .sampleCount = 1,
  };
  out.colorAttachments[0] = {.semantic = gfx::ColorAttachmentSemantic::SceneColor, .format = Format};
  gfx::detail::finalize_render_target_layout(out);
  return out;
}

bool select_draw(const scene::DrawClass& cls) noexcept {
  return (cls.key.categories & scene::category::FakeShadow) != 0 && cls.key.mainCamera &&
         cls.key.role == scene::Role::Main && !cls.offscreen && cls.key.window <= scene::Window::Opaque;
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = detail::PassKind::Render,
      .cls = detail::PassClass::Required,
      .order = Order,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .stageParams = fill_stage,
  };
}

bool consumed_fifo() noexcept { return g_consumedFifo.load(std::memory_order_relaxed); }

void note_post_shadow() noexcept { bump(g_counts.postShadowDraws); }

bool frame_enabled(const detail::StageData& stage) noexcept {
  const detail::CompiledGraph* graph = stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
  if (graph == nullptr || stage.stage != Stage::OpaqueDone) {
    return false;
  }
  const detail::ResourceId id = graph->find_resource(Resource);
  if (id == detail::InvalidResource) {
    return false;
  }
  const detail::NodeId creator = graph->resources[id].creator;
  return creator != detail::InvalidNode && graph->nodes[creator].programIndex != detail::InvalidNode &&
         stage.is_enabled(graph->nodes[creator].programIndex);
}

Stats stats() noexcept {
  const uint64_t pending = scene::capture_list::stats().pending;
  const bool warm = g_counts.latches.load(std::memory_order_relaxed) >= WarmUpLatches;
  return {
      .latches = g_counts.latches.load(std::memory_order_relaxed),
      .declaredLatches = g_counts.declaredLatches.load(std::memory_order_relaxed),
      .consumedLatches = g_counts.consumedLatches.load(std::memory_order_relaxed),
      .selected = g_counts.selected.load(std::memory_order_relaxed),
      .depthRefused = g_counts.depthRefused.load(std::memory_order_relaxed),
      .frames = g_counts.frames.load(std::memory_order_relaxed),
      .replayed = g_counts.replayed.load(std::memory_order_relaxed),
      .pending = pending,
      .pendingAfterWarmUp = warm ? pending - g_pendingAtWarmUp.load(std::memory_order_relaxed) : 0,
      .postShadowDraws = g_counts.postShadowDraws.load(std::memory_order_relaxed),
      .graphHash = g_graphHash.load(std::memory_order_relaxed),
  };
}

std::array<float, 3> term(std::array<float, 3> mask, float q, std::array<float, 3> fc) noexcept {
  std::array<float, 3> out{};
  for (size_t c = 0; c < out.size(); ++c) {
    const float d = 1.f - q * fc[c];
    out[c] = mask[c] >= 1.f || d < 1.f / 255.f ? 1.f : std::clamp((mask[c] - q * fc[c]) / d, 0.f, 1.f);
  }
  return out;
}

namespace testing {
void reset() noexcept {
  g_consumedOverride.reset();
  g_consumedFifo.store(false, std::memory_order_relaxed);
  g_consumed.store(false, std::memory_order_relaxed);
  for (const auto member : CounterList) {
    (g_counts.*member).store(0, std::memory_order_relaxed);
  }
}

void set_consumed(std::optional<bool> consumed) noexcept { g_consumedOverride = consumed; }

uint32_t consumer() noexcept { return g_consumer; }
} // namespace testing
} // namespace aurora::render::fakeshadow::lsm

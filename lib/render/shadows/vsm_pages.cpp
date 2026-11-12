#include "vsm_pages.hpp"

#include "sun_visibility.hpp"
#include "vsm_alloc.hpp"
#include "vsm_check.hpp"
#include "vsm_cull.hpp"
#include "vsm_invalidate.hpp"
#include "vsm_raster.hpp"

#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../history_reset.hpp"
#include "../pass_context.hpp"
#include "../settings.hpp"
#include "../stage_data.hpp"
#include "../debug/views.hpp"
#include "../debug/vsm_view.hpp"
#include "../fakeshadow/gate.hpp"
#include "../geometry/compact.hpp"
#include "../post/composite.hpp"
#include "../scene/identity.hpp"
#include "../../gfx/scene_targets.hpp"
#include "../../gx/hints.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>
#include <aurora/render_caps.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numbers>
#include <string>
#include <vector>

namespace aurora::render::shadows::vsm {
namespace {
using detail::PassId;
using detail::PersistentHandle;
using detail::SceneHints;
using detail::StageData;

constexpr Module Log{"aurora::render::shadows"};
constexpr double DegreesPerRadian = 180.0 / std::numbers::pi;
constexpr double ScreenHeightReference = 1440.0;
constexpr uint32_t NoRef = UINT16_MAX;

enum Persistent : uint32_t {
  TableHandle,
  PhysHandle,
  ViewsHandle,
  StatsHandle,
  HistoryHandle,
  CastBitsHandle,
  ChunkBoxesHandle,
  PoolHandle,
  HandleCount
};
constexpr std::array<std::string_view, HandleCount> HandleNames{Table,   Phys,     ViewsName,  StatsName,
                                                                History, CastBits, ChunkBoxes, Pool};

struct UpdateParams {
  uint32_t producing = 0;
  uint32_t lightId = 0;
  uint32_t lightFlags = 0;
  uint32_t quality = 0;
  double dir[3]{};
  double texel0 = 0.0;
  float elevationDeg = 0.f;
  float levelBias = 0.f;
  uint32_t pages = 0;
  uint32_t pagesX = 0;
  uint32_t budget = 0;
  uint32_t generation = 0;
  uint32_t chainMask = 0;
  uint32_t fifoSet = 0;
  uint64_t planFrame = 0;
  uint32_t coverageComplete = 0;
  uint32_t reserved = 0;
  uint64_t hintFrame = 0;
};
static_assert(sizeof(UpdateParams) <= MaxPassParamsSize);

struct UpdateLocal {
  uint16_t views = NoRef;
  uint16_t table = NoRef;
  uint16_t phys = NoRef;
  uint16_t stats = NoRef;
};
static_assert(sizeof(UpdateLocal) <= detail::PassLocalSize);

struct LatchedHints {
  uint64_t frame = 0;
  bool mainCamera = false;
  bool noSun = false;
  bool keyRoom = false;
  double unitsPerMetre = DefaultUnitsPerMetre;
  SunSelection sun;
};

struct SunRate {
  bool hasPrev = false;
  uint32_t prevLight = 0;
  std::array<double, 3> prevDir{};
  uint64_t samples = 0;
  uint64_t moving = 0;
  double sum = 0.0;
  double max = 0.0;
};

struct Worker {
  bool started = false;
  bool held = false;
  uint32_t lastFrameIndex = 0;
  uint32_t generation = 0;
  uint32_t clock = 0;
  uint32_t mask = 0;
  uint32_t expected = 0;
  uint32_t pendingReasons = 0;
  SunClipmap clipmap;
  detail::history_reset::Camera camera;
  testing::WorkerCounts counts;
};

std::array<PersistentHandle, HandleCount> g_handles{};
std::array<PassId, PassCount> g_passes{};
PassId g_castBitsPass = detail::InvalidPassId;
bool g_wanted = false;
bool g_active = false;
std::optional<bool> g_activeOverride;
PoolExtent g_extent;
bool g_warnedReduced = false;
bool g_resourcesLive = false;
uint32_t g_generation = 0;
PoolExtent g_liveExtent;
wgpu::Buffer g_table;
wgpu::Buffer g_phys;
wgpu::Buffer g_views;
wgpu::Buffer g_stats;
wgpu::Buffer g_history;
wgpu::Buffer g_castBits;
wgpu::Buffer g_chunkBoxes;
wgpu::Texture g_pool;
LatchedHints g_hints;
bool g_lowSunOn = false;
SunPlan g_plan;
std::optional<SunPlan> g_planOverride;
bool g_chainReadyLatched = false;
uint32_t g_expectedMask = 0;
uint64_t g_latches = 0;
std::array<uint64_t, IdleCount> g_idle{};
SunRate g_rate;
std::array<ConsumerTerm, MaxConsumerTerms> g_terms{};
uint32_t g_termCount = 0;

std::mutex g_chainMutex;
std::vector<uint32_t> g_chainIds;
std::vector<uint32_t> g_pendingChain;
bool (*g_pipelineQuery)(uint32_t id) noexcept = nullptr;
std::optional<bool> g_compactOverride;
std::optional<bool> g_forceOverride;
std::atomic_bool g_shadowView{false};
std::optional<bool> g_sunTermOverride;
std::optional<bool> g_shadowViewOverride;

uint64_t g_snapFrame = UINT64_MAX;
bool g_snapReady = false;
std::atomic_bool g_fifoReady{false};

Worker g_worker;

double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> cross3(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

std::array<double, 3> sub3(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

std::array<double, 3> normalize3(const std::array<double, 3>& a) noexcept {
  const double length = std::sqrt(dot3(a, a));
  return length > 0.0 ? std::array{a[0] / length, a[1] / length, a[2] / length} : std::array{0.0, 0.0, 0.0};
}

double angle_between(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept {
  const std::array<double, 3> c = cross3(a, b);
  return std::atan2(std::sqrt(dot3(c, c)), dot3(a, b));
}

void basis(const std::array<double, 3>& d, std::array<double, 3>& bx, std::array<double, 3>& by) noexcept {
  const std::array<double, 3> helper = std::abs(d[1]) < 0.99 ? std::array{0.0, 1.0, 0.0} : std::array{1.0, 0.0, 0.0};
  bx = normalize3(cross3(helper, d));
  by = cross3(d, bx);
}

std::optional<std::array<double, 3>> light_direction(const AuroraLight& light) noexcept {
  const std::array<double, 3> d{light.direction[0], light.direction[1], light.direction[2]};
  if (!std::isfinite(d[0]) || !std::isfinite(d[1]) || !std::isfinite(d[2]) || dot3(d, d) < 1e-12) {
    return std::nullopt;
  }
  return normalize3(d);
}

bool camera_usable(const SceneHints* hints) noexcept {
  return hints != nullptr && hints->has(detail::SceneHasMainCamera) &&
         gx::hints::main_camera_usable(hints->mainCamera.gxProj);
}

bool env_force() noexcept {
  static const bool forced = [] {
    const char* value = SDL_getenv("AURORA_RENDER_VSM_FORCE");
    if (value == nullptr) {
      return false;
    }
    if (std::string_view{value} != "1") {
      Log.warn("AURORA_RENDER_VSM_FORCE={} is not 1; ignored", value);
      return false;
    }
    Log.warn("AURORA_RENDER_VSM_FORCE=1: running the virtual shadow map chain without a consumer");
    return true;
  }();
  return forced;
}

float quality_bias(Quality quality) noexcept {
  switch (quality) {
  case Quality::Low:
  case Quality::Medium:
    return 0.5f;
  case Quality::Ultra:
    return -0.5f;
  default:
    return 0.f;
  }
}

bool query_ready(uint32_t id) noexcept {
  return g_pipelineQuery != nullptr ? g_pipelineQuery(id) : shaders::ready(shaders::PipelineId{id});
}

bool chain_snapshot(const StageData& stage) noexcept {
  // One readiness answer per frame, because a pass still compiling would otherwise drop out alone
  if (g_snapFrame == stage.frameId) {
    return g_snapReady;
  }
  std::vector<uint32_t> ids;
  {
    std::lock_guard lock{g_chainMutex};
    ids = g_chainIds;
  }
  bool ready = std::ranges::all_of(ids, query_ready);
  ready = ready && g_compactOverride.value_or(compact::ready());
  g_snapFrame = stage.frameId;
  g_snapReady = ready;
  g_fifoReady.store(ready, std::memory_order_relaxed);
  return ready;
}

bool declarable(const detail::PassSetup& setup) noexcept {
  return active() && setup.env != nullptr && setup.env->retained && g_handles[TableHandle].id != 0;
}

void declare_update(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declarable(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  UpdateLocal local;
  local.views = builder.import_persistent(ViewsName, g_handles[ViewsHandle], Access::StorageRW).index;
  local.table = builder.import_persistent(Table, g_handles[TableHandle], Access::StorageRW).index;
  local.phys = builder.import_persistent(Phys, g_handles[PhysHandle], Access::StorageRW).index;
  local.stats = builder.import_persistent(StatsName, g_handles[StatsHandle], Access::StorageRW).index;
  const std::span<std::byte, detail::PassLocalSize> out = builder.local();
  std::memcpy(out.data(), &local, sizeof(local));
}

bool update_enabled(const StageData& stage, const void*) { return chain_enabled(stage); }

void update_pipelines(const detail::PipelineSetup& setup, const void*,
                      std::span<uint32_t, detail::MaxPassPipelines> ids) {
  const shaders::PipelineId id = shaders::request_compute({
      .label = UpdatePass,
      .shader = Shader,
      .entryPoint = "vsm_update",
      .layout = setup.layout,
      .layoutKey = setup.layoutHash,
  });
  ids[0] = id.value;
  register_chain_pipeline(id);
}

void update_stage_params(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(UpdateParams)) {
    return;
  }
  UpdateParams params;
  std::memcpy(&params, bytes.data(), sizeof(params));
  params.fifoSet = 1;
  params.coverageComplete = scene::retained::coverage_complete() ? 1u : 0u;
  params.hintFrame = stage.sceneHints != nullptr ? stage.sceneHints->hintFrame : 0;
  std::memcpy(bytes.data(), &params, sizeof(params));
}

void split(double value, float& hi, float& lo) noexcept {
  hi = static_cast<float>(value);
  lo = static_cast<float>(value - static_cast<double>(hi));
}

void encode_update(detail::PassContext& ctx, const void*) {
  UpdateParams params;
  std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(params)));
  UpdateLocal local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const BufferView views = ctx.buffer(BufRef{local.views});
  const BufferView table = ctx.buffer(BufRef{local.table});
  const BufferView phys = ctx.buffer(BufRef{local.phys});
  const BufferView stats = ctx.buffer(BufRef{local.stats});
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const StageData& stage = ctx.stage_data();
  if (!pipeline || !views.buffer || !table.buffer || !phys.buffer || !stats.buffer || views.size < ViewsBytes ||
      stage.retained == nullptr || params.fifoSet == 0) {
    // Later passes test the Update bit, so a bail must not leave last frame's mask behind
    g_worker.mask = 0;
    return;
  }
  // Views come from the frozen hints on the worker, so every VSM pass in the frame sees one camera and one latch
  const auto& inv = stage.retained->invView.m;
  const FrameDecision decision = decide_frame({
      .frameIndex = stage.frameIndex,
      .hintFrame = params.hintFrame,
      .hints = stage.sceneHints.get(),
      .eye = {inv[0][3], inv[1][3], inv[2][3]},
      .planDir = {params.dir[0], params.dir[1], params.dir[2]},
      .lightId = params.lightId,
      .texel0 = params.texel0,
      .producing = params.producing != 0,
      .coverageComplete = params.coverageComplete != 0,
      .pages = params.pages,
      .pagesX = params.pagesX,
      .budget = params.budget,
      .levelBias = params.levelBias,
      .generation = params.generation,
      .chainMask = params.chainMask,
      .signals = frame_signals(stage),
  });
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  encoder.WriteBuffer(views.buffer, views.offset, reinterpret_cast<const uint8_t*>(&decision.views), ViewsBytes);
  encoder.ClearBuffer(stats.buffer, stats.offset, std::min(stats.size, StatsBytes));
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(MaxVirtualPages / UpdateWorkgroup);
  pass.End();
  mark_encoded(chain_bit::Update);
}

void declare_cast_bits(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!declarable(setup) || g_handles[CastBitsHandle].id == 0) {
    builder.skip();
    return;
  }
  builder.after(phase::Shadows);
  builder.import_persistent(CastBits, g_handles[CastBitsHandle], Access::CopySrc);
  builder.side_effect();
}

bool cast_bits_enabled(const StageData& stage, const void*) {
  const bool chain = chain_enabled(stage);
  if (stage.sceneHints != nullptr) {
    fakeshadow::gate::note_fifo_frame(stage.sceneHints->hintFrame, sun_visibility::frame_enabled(stage));
  }
  return chain && fakeshadow::gate::cast_bits_wanted();
}

void encode_cast_bits(detail::PassContext& ctx, const void*) {
  const StageData& stage = ctx.stage_data();
  if ((encoded_mask() & chain_bit::Scan) == 0 || stage.sceneHints == nullptr) {
    return;
  }
  fakeshadow::gate::copy_cast_bits(ctx.encoder(), ctx.buffer(BufRef{0}), stage.sceneHints->hintFrame);
}

void create_resources() {
  const wgpu::Device& device = webgpu::g_device;
  const auto buffer = [&](const char* label, uint64_t size, wgpu::BufferUsage usage) {
    const wgpu::BufferDescriptor descriptor{.label = label, .usage = usage, .size = size};
    return device.CreateBuffer(&descriptor);
  };
  constexpr wgpu::BufferUsage Common =
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  if (!g_table) {
    g_table = buffer("vsm.pt", TableBytes, Common);
    g_phys = buffer("vsm.phys", PhysBytes, Common);
    g_views = buffer("vsm.views", ViewsBytes, Common | wgpu::BufferUsage::Uniform);
    g_stats = buffer("vsm.stats", StatsBytes, Common);
    g_history = buffer("vsm.history", HistoryBytes, Common);
    g_castBits = buffer("vsm.cast_bits", CastBitsBytes, Common);
    g_chunkBoxes = buffer("vsm.chunk_boxes", ChunkBoxesBytes, Common);
    detail::set_persistent_buffer(g_handles[TableHandle], g_table, 0, TableBytes);
    detail::set_persistent_buffer(g_handles[PhysHandle], g_phys, 0, PhysBytes);
    detail::set_persistent_buffer(g_handles[ViewsHandle], g_views, 0, ViewsBytes);
    detail::set_persistent_buffer(g_handles[StatsHandle], g_stats, 0, StatsBytes);
    detail::set_persistent_buffer(g_handles[HistoryHandle], g_history, 0, HistoryBytes);
    detail::set_persistent_buffer(g_handles[CastBitsHandle], g_castBits, 0, CastBitsBytes);
    detail::set_persistent_buffer(g_handles[ChunkBoxesHandle], g_chunkBoxes, 0, ChunkBoxesBytes);
  }
  if (g_pool && g_liveExtent.width == g_extent.width && g_liveExtent.height == g_extent.height) {
    return;
  }
  const wgpu::TextureDescriptor descriptor{
      .label = "vsm.pool",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .size = {g_extent.width, g_extent.height, 1},
      .format = wgpu::TextureFormat::Depth32Float,
  };
  g_pool = device.CreateTexture(&descriptor);
  detail::set_persistent_texture(g_handles[PoolHandle], g_pool, g_pool.CreateView());
  g_liveExtent = g_extent;
  ++g_generation;
  Log.info("VSM: page pool of {} pages ({}x{} Depth32Float, {} MiB), generation {}", g_extent.pages, g_extent.width,
           g_extent.height, (uint64_t{g_extent.width} * g_extent.height * 4) >> 20, g_generation);
}

void destroy_resources() noexcept {
  for (uint32_t i = 0; i < PoolHandle; ++i) {
    if (g_handles[i].id != 0) {
      detail::set_persistent_buffer(g_handles[i], {}, 0, 0);
    }
  }
  if (g_handles[PoolHandle].id != 0) {
    detail::set_persistent_texture(g_handles[PoolHandle], {}, {});
  }
  g_table = {};
  g_phys = {};
  g_views = {};
  g_stats = {};
  g_history = {};
  g_castBits = {};
  g_chunkBoxes = {};
  g_pool = {};
  g_liveExtent = {};
  g_resourcesLive = false;
}

uint32_t expected_mask(const detail::CompiledGraph* graph) noexcept {
  if (graph == nullptr) {
    return 0;
  }
  constexpr std::array<std::pair<std::string_view, uint32_t>, 18> Chain{{
      {UpdatePass, chain_bit::Update},
      {ScanPass, chain_bit::Scan},
      {ScanListPass, chain_bit::ScanList},
      {DirtyPass, chain_bit::Dirty},
      {MarkPass, chain_bit::Mark},
      {ClassifyPass, chain_bit::Classify},
      {AllocListsPass, chain_bit::AllocLists},
      {AssignPass, chain_bit::Assign},
      {ClearPass, chain_bit::Clear},
      {CommitPass, chain_bit::Commit},
      {ChunkBoundsPass, chain_bit::ChunkBounds},
      {CullCandidatesPass, chain_bit::CullCandidates},
      {CullListPass, chain_bit::CullList},
      {CullCountPass, chain_bit::CullCount},
      {CullScanPass, chain_bit::CullScan},
      {CullSkipPass, chain_bit::CullSkip},
      {CullEmitPass, chain_bit::CullEmit},
      {RasterPass, chain_bit::Raster},
  }};
  uint32_t mask = 0;
  for (const auto& [name, bit] : Chain) {
    const detail::NodeId node = graph->find_node(name);
    mask |= node != detail::InvalidNode && graph->nodes[node].status == detail::CullReason::None ? bit : 0u;
  }
  return mask;
}

void record_rate(const SunSelection& sun) noexcept {
  SunRate& r = g_rate;
  if (!sun.found) {
    r.hasPrev = false;
    return;
  }
  if (r.hasPrev && r.prevLight == sun.lightId) {
    const double degrees = angle_between(r.prevDir, sun.dirWorld) * DegreesPerRadian;
    ++r.samples;
    if (degrees > 0.0) {
      ++r.moving;
      r.sum += degrees;
      r.max = std::max(r.max, degrees);
    }
  }
  r.hasPrev = true;
  r.prevLight = sun.lightId;
  r.prevDir = sun.dirWorld;
}

std::string level_list(const std::array<uint64_t, SunLevels>& counts) {
  std::string out;
  for (size_t i = 0; i < counts.size(); ++i) {
    out += fmt::format("{}{}", i == 0 ? "" : " ", counts[i]);
  }
  return out;
}

void log_summary() {
  const testing::WorkerCounts& c = g_worker.counts;
  if (g_latches == 0 && c.frames == 0) {
    return;
  }
  std::string idle;
  for (size_t i = 1; i < IdleCount; ++i) {
    idle += fmt::format("{}{} {}", i == 1 ? "" : ", ", idle_name(static_cast<Idle>(i)), g_idle[i]);
  }
  Log.info("VSM: {} latches, {} producing; not producing: {}", g_latches, g_idle[0], idle);
  const SunRate& r = g_rate;
  Log.info(
      "VSM: sun moved on {} of {} latches with the same key light, mean {:.5f} deg per moving frame, max "
      "{:.5f} deg, total {:.4f} deg",
      r.moving, r.samples, r.moving != 0 ? r.sum / static_cast<double>(r.moving) : 0.0, r.max, r.sum);
  constexpr std::array<std::string_view, full_reason::Count> Reasons{
      "first",           "gap",       "chain incomplete", "pool", "key light", "anchor", "check", "retained reset",
      "clear instances", "unapplied", "scene epoch"};
  std::string reasons;
  for (size_t i = 0; i < Reasons.size(); ++i) {
    reasons += fmt::format("{}{} {}", i == 0 ? "" : ", ", Reasons[i], c.reasons[i]);
  }
  Log.info(
      "VSM: {} chain frames, {} fulls ({}), {} held, {} exempt (cut {}, coverage incomplete {}, scene epoch {}); "
      "relatches per level [{}], re-centres per level [{}]",
      c.frames, c.fulls, reasons, c.held, c.exempt, c.exemptCut, c.exemptCoverage, c.exemptEpoch,
      level_list(c.relatches), level_list(c.recentres));
}
} // namespace

std::array<float, 3> ViewRecord::direction() const noexcept {
  const double x = anchorRel[8];
  const double y = anchorRel[9];
  const double z = anchorRel[10];
  const double length = std::sqrt(x * x + y * y + z * z);
  if (!(length > 0.0)) {
    return {0.f, 0.f, 0.f};
  }
  return {static_cast<float>(-x / length), static_cast<float>(-y / length), static_cast<float>(-z / length)};
}

PoolFormat pool_format() noexcept { return PoolFormat::DepthAtlas; }

TexDesc pool_desc() noexcept {
  return {
      .format = wgpu::TextureFormat::Depth32Float,
      .sizeClass = SizeClass::Absolute,
      .width = std::max(g_extent.width, PageTexels),
      .height = std::max(g_extent.height, PageTexels),
  };
}

PoolExtent pool_extent(uint32_t pages, uint32_t maxDimension) noexcept {
  PoolExtent out;
  uint32_t p = std::clamp(pages, MinPoolPages, MaxPoolPages);
  if (WindowPages * PageTexels > maxDimension) {
    out.pages = p;
    out.pagesX = WindowPages;
    return out;
  }
  const uint32_t rowsMax = maxDimension / PageTexels;
  for (;;) {
    const uint32_t pagesX = p <= 2048 ? 32u : 64u;
    const uint32_t rows = (p + pagesX - 1) / pagesX;
    const bool wide = pagesX * PageTexels <= maxDimension;
    if (wide && rows <= rowsMax) {
      out.pages = p;
      out.pagesX = pagesX;
      out.width = pagesX * PageTexels;
      out.height = rows * PageTexels;
      out.fits = true;
      return out;
    }
    if (p == MinPoolPages) {
      out.pages = p;
      out.pagesX = pagesX;
      return out;
    }
    out.reduced = true;
    p = std::max(MinPoolPages, std::min(p - 1, wide ? pagesX * rowsMax : 2048u));
  }
}

SunSelection select_sun(const SceneHints& hints) noexcept {
  SunSelection best;
  if (!hints.has(detail::SceneHasLights)) {
    return best;
  }
  constexpr uint32_t Excluded = AURORA_LIGHT_NO_SHADOW | AURORA_LIGHT_CAMERA_FILL | AURORA_LIGHT_KEY_ROOM;
  uint32_t bestPriority = 0;
  for (size_t i = 0; i < hints.lights.size(); ++i) {
    const AuroraLight& light = hints.lights[i];
    if (light.type != AURORA_LIGHT_DIRECTIONAL || light.categoryMask != 0 || (light.flags & Excluded) != 0) {
      continue;
    }
    const std::optional<std::array<double, 3>> d = light_direction(light);
    if (!d || (best.found && light.shadowPriority <= bestPriority)) {
      continue;
    }
    bestPriority = light.shadowPriority;
    best = {
        .found = true,
        .lightId = light.id,
        .hintIndex = static_cast<uint32_t>(i),
        .flags = light.flags,
        .dirWorld = *d,
        .elevationDeg = std::asin(std::clamp(-(*d)[1], -1.0, 1.0)) * DegreesPerRadian,
    };
  }
  return best;
}

std::string_view idle_name(Idle reason) noexcept {
  constexpr std::array<std::string_view, IdleCount> Names{"producing", "inactive", "pipelines", "no camera",
                                                          "no sun",    "sun off",  "low sun"};
  const auto index = static_cast<size_t>(reason);
  return index < Names.size() ? Names[index] : "unknown";
}

SunPlan decide_plan(const PlanInputs& in, bool& lowSunOn) noexcept {
  SunPlan out;
  out.frame = in.frame;
  if (in.sun.found) {
    out.lightId = in.sun.lightId;
    out.lightFlags = in.sun.flags;
    out.dirWorld = in.sun.dirWorld;
    out.elevationDeg = in.sun.elevationDeg;
    lowSunOn = in.sun.elevationDeg >= (lowSunOn ? LowSunOffDegrees : LowSunOnDegrees);
  } else {
    lowSunOn = false;
  }
  if (!in.active) {
    out.idle = Idle::Inactive;
  } else if (!in.chainReady) {
    out.idle = Idle::Pipelines;
  } else if (!in.mainCamera) {
    out.idle = Idle::NoCamera;
  } else if (!in.sun.found) {
    out.idle = Idle::NoSun;
  } else if (in.noSun) {
    out.idle = Idle::SunDisabled;
  } else if (!lowSunOn) {
    out.idle = Idle::LowSun;
  } else {
    out.idle = Idle::None;
  }
  out.producing = out.idle == Idle::None;
  return out;
}

const SunPlan& plan() noexcept { return g_planOverride ? *g_planOverride : g_plan; }

int32_t window_origin(double texel) noexcept {
  const double page = std::floor(texel / PageTexels) - static_cast<double>(WindowPages / 2);
  return static_cast<int32_t>(std::clamp(page, -2147483648.0 + WindowPages, 2147483647.0 - WindowPages));
}

bool kept_after_shift(uint32_t sx, uint32_t sy, const std::array<int32_t, 2>& prev,
                      const std::array<int32_t, 2>& next) noexcept {
  const int64_t dx = int64_t{abs_page(prev[0], sx)} - next[0];
  const int64_t dy = int64_t{abs_page(prev[1], sy)} - next[1];
  return dx >= 0 && dx < WindowPages && dy >= 0 && dy < WindowPages;
}

double level_texel(double texel0, uint32_t level) noexcept { return std::ldexp(texel0, static_cast<int>(level)); }

double depth_range(double texel0, uint32_t level) noexcept {
  const double radius = level_texel(texel0, level) * (WindowTexels / 2);
  return std::clamp(DepthRangePerRadius * radius, MinDepthRange, MaxDepthRange);
}

double latch_threshold(uint32_t level) noexcept {
  const double shift = std::ldexp(LatchTexels * Texel0Metres, static_cast<int>(level)) / LatchHeightMetres;
  return std::min(shift, MaxLatchDegrees / DegreesPerRadian);
}

std::array<float, 3> view_coords(const ViewRecord& view, const std::array<float, 3>& pRel) noexcept {
  std::array<float, 3> out{};
  for (size_t r = 0; r < 3; ++r) {
    const float* row = view.camRel + r * 4;
    out[r] = row[0] * pRel[0] + row[1] * pRel[1] + row[2] * pRel[2] + row[3];
  }
  return out;
}

std::array<float, 3> view_coords_anchor(const ViewRecord& view, const std::array<float, 3>& pAnchor) noexcept {
  std::array<float, 3> out{};
  for (size_t r = 0; r < 3; ++r) {
    const float* row = view.anchorRel + r * 4;
    out[r] = row[0] * pAnchor[0] + row[1] * pAnchor[1] + row[2] * pAnchor[2] + row[3];
  }
  return out;
}

uint32_t select_level(const Views& views, float footprint, const std::array<float, 3>& pRel, float bias) noexcept {
  const float texel0 = views.header.texel0;
  if (!(texel0 > 0.f)) {
    return SunLevels;
  }
  const float f = footprint > 0.f ? footprint : 1e-30f;
  const float wanted = std::ceil(std::log2(f / texel0) + bias);
  const int32_t first = std::isfinite(wanted) ? static_cast<int32_t>(std::clamp(wanted, 0.f, SunLevels - 1.f)) : 0;
  constexpr float Margin = FilterMarginTexels;
  for (auto level = static_cast<uint32_t>(first); level < SunLevels; ++level) {
    const std::array<float, 3> c = view_coords(views.views[level_view(level)], pRel);
    if (c[0] >= Margin && c[0] < WindowTexels - Margin && c[1] >= Margin && c[1] < WindowTexels - Margin) {
      return level;
    }
  }
  return SunLevels;
}

LatchDecision latch_schedule(std::span<const LevelLatch, SunLevels> levels, const std::array<double, 3>& dir,
                             const std::array<double, 3>& eyeRel, double texel0, bool forceAll) noexcept {
  LatchDecision out;
  if (forceAll || std::ranges::any_of(levels, [](const LevelLatch& l) { return !l.valid; })) {
    out.all = true;
    return out;
  }
  std::array<double, SunLevels> angles{};
  for (uint32_t k = 0; k < SunLevels; ++k) {
    angles[k] = angle_between(levels[k].dir, dir);
    if (angles[k] > 4.0 * latch_threshold(k)) {
      out.all = true;
      out.lagged = true;
      return out;
    }
  }
  for (uint32_t k = 0; k < SunLevels; ++k) {
    if (angles[k] > latch_threshold(k)) {
      out.level = static_cast<int32_t>(k);
      return out;
    }
    if (std::abs(dot3(levels[k].dir, eyeRel) - levels[k].zc) > depth_range(texel0, k) / 4.0) {
      out.level = static_cast<int32_t>(k);
      out.recentre = true;
      return out;
    }
  }
  return out;
}

void SunClipmap::relatch(uint32_t level, const std::array<double, 3>& dir) noexcept {
  const std::array<double, 3> eyeRel = sub3(m_eye, m_anchor);
  m_latch[level] = {.valid = true, .dir = dir, .zc = dot3(dir, eyeRel)};
  basis(dir, m_level[level].bx, m_level[level].by);
  m_level[level].flags |= view_flag::Freed;
  m_level[level].latchSerial = ++m_serial;
}

SunClipmap::Result SunClipmap::update(const Input& in) noexcept {
  Result out;
  const std::array<double, 3> offset = sub3(in.eye, m_anchor);
  const bool far = std::sqrt(dot3(offset, offset)) > AnchorReach;
  if (!m_valid || far || in.sceneEpochChanged || in.texel0 != m_texel0) {
    for (size_t i = 0; i < 3; ++i) {
      m_anchor[i] = std::round(in.eye[i] / AnchorSnap) * AnchorSnap;
    }
    out.anchorMoved = m_valid;
  }
  out.keyLight = m_valid && in.lightId != m_lightId;
  m_lightId = in.lightId;
  m_texel0 = in.texel0;
  m_eye = in.eye;
  const std::array<double, 3> eyeRel = sub3(m_eye, m_anchor);
  const bool forceAll = !m_valid || in.full || out.anchorMoved || out.keyLight;
  const LatchDecision decision = latch_schedule(m_latch, in.dir, eyeRel, in.texel0, forceAll);
  for (Level& level : m_level) {
    level.flags = 0;
    level.prevOrigin = level.origin;
  }
  if (decision.all) {
    out.relatchAll = true;
    out.lagged = decision.lagged;
    for (uint32_t k = 0; k < SunLevels; ++k) {
      relatch(k, in.dir);
    }
  } else if (decision.level >= 0) {
    const auto k = static_cast<uint32_t>(decision.level);
    out.level = decision.level;
    out.recentre = decision.recentre;
    if (decision.recentre) {
      m_latch[k].zc = dot3(m_latch[k].dir, eyeRel);
      m_level[k].flags |= view_flag::Recentred;
    } else {
      relatch(k, in.dir);
    }
  }
  for (uint32_t k = 0; k < SunLevels; ++k) {
    Level& level = m_level[k];
    const double texel = level_texel(m_texel0, k);
    level.origin = {window_origin(dot3(level.bx, eyeRel) / texel), window_origin(dot3(level.by, eyeRel) / texel)};
    if ((level.flags & (view_flag::Freed | view_flag::Recentred)) != 0) {
      level.prevOrigin = level.origin;
    } else if (level.origin != level.prevOrigin) {
      level.flags |= view_flag::Wrap;
    }
  }
  m_valid = true;
  return out;
}

void SunClipmap::fill(Views& views) const noexcept {
  ViewsHeader& h = views.header;
  h.viewCount = SunLevels;
  for (size_t i = 0; i < 3; ++i) {
    split(m_anchor[i], h.anchorHi[i], h.anchorLo[i]);
  }
  h.texel0 = static_cast<float>(m_texel0);
  const std::array<double, 3> eyeRel = sub3(m_eye, m_anchor);
  for (uint32_t k = 0; k < SunLevels; ++k) {
    const uint32_t index = level_view(k);
    ViewRecord& v = views.views[index];
    const Level& level = m_level[k];
    const LevelLatch& latch = m_latch[k];
    const double texel = level_texel(m_texel0, k);
    const double range = depth_range(m_texel0, k);
    const std::array<std::array<double, 3>, 3> rows{
        std::array{level.bx[0] / texel, level.bx[1] / texel, level.bx[2] / texel},
        std::array{level.by[0] / texel, level.by[1] / texel, level.by[2] / texel},
        std::array{-latch.dir[0] / range, -latch.dir[1] / range, -latch.dir[2] / range},
    };
    const std::array<double, 3> anchorT{
        -static_cast<double>(PageTexels) * level.origin[0],
        -static_cast<double>(PageTexels) * level.origin[1],
        0.5 + latch.zc / range,
    };
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 3; ++c) {
        v.anchorRel[r * 4 + c] = static_cast<float>(rows[r][c]);
        v.camRel[r * 4 + c] = static_cast<float>(rows[r][c]);
      }
      v.anchorRel[r * 4 + 3] = static_cast<float>(anchorT[r]);
      v.camRel[r * 4 + 3] = static_cast<float>(anchorT[r] + dot3(rows[r], eyeRel));
    }
    v.origin[0] = level.origin[0];
    v.origin[1] = level.origin[1];
    v.prevOrigin[0] = level.prevOrigin[0];
    v.prevOrigin[1] = level.prevOrigin[1];
    v.texel = static_cast<float>(texel);
    v.depthScale = static_cast<float>(1.0 / range);
    v.flags = view_flag::Active | level.flags | static_cast<uint32_t>(ViewKind::SunLevel) << view_flag::KindShift |
              (index * PagesPerView) << view_flag::TableShift;
    v.pages = WindowPages | WindowPages << 16;
  }
}

bool exempt_frame(const SunClipmap::Result& result, const FrameFlags& flags) noexcept {
  return flags.full || result.relatchAll || result.level >= 0 || flags.sceneEpochChanged || flags.cut ||
         !flags.coverageComplete;
}

std::array<double, 3> frame_direction(const SceneHints* hints, uint32_t lightId,
                                      const std::array<double, 3>& planDir) noexcept {
  if (hints != nullptr) {
    for (const AuroraLight& light : hints->lights) {
      if (light.id != lightId || light.type != AURORA_LIGHT_DIRECTIONAL) {
        continue;
      }
      if (const std::optional<std::array<double, 3>> d = light_direction(light)) {
        return *d;
      }
    }
  }
  return normalize3(planDir);
}

FrameDecision decide_frame(const FrameInput& in) noexcept {
  Worker& w = g_worker;
  FrameDecision d;
  uint32_t reasons = 0;
  if (!w.started) {
    reasons |= full_reason::First;
  } else {
    if (in.frameIndex != w.lastFrameIndex + 1) {
      reasons |= full_reason::Gap;
    }
    if ((w.mask & w.expected) != w.expected) {
      reasons |= full_reason::ChainIncomplete;
    }
    if (in.generation != w.generation) {
      reasons |= full_reason::PoolGeneration;
    }
  }
  if (check::force_full(in.hintFrame)) {
    reasons |= full_reason::Check;
  }
  const FrameSignals& sig = in.signals;
  reasons |= (sig.retainedReset ? full_reason::RetainedReset : 0u) |
             (sig.clearInstances ? full_reason::ClearInstances : 0u) | (sig.sceneEpoch ? full_reason::SceneEpoch : 0u);
  d.held = sig.unapplied;
  if (d.held) {
    w.pendingReasons |= full_reason::Unapplied;
  } else {
    reasons |= std::exchange(w.pendingReasons, 0u);
  }
  const SunClipmap::Input input{
      .eye = in.eye,
      .dir = frame_direction(in.hints, in.lightId, in.planDir),
      .lightId = in.lightId,
      .texel0 = in.texel0,
      .full = reasons != 0,
      .sceneEpochChanged = sig.sceneEpoch,
  };
  d.clipmap = w.clipmap.update(input);
  if (d.clipmap.keyLight) {
    reasons |= full_reason::KeyLight;
  }
  if (d.clipmap.anchorMoved) {
    reasons |= full_reason::Anchor;
  }
  d.reasons = reasons;
  d.full = reasons != 0;
  if (in.hints != nullptr) {
    detail::history_reset::Camera next;
    d.cut = detail::history_reset::classify(w.camera, *in.hints, next) != detail::history_reset::Reason::None;
    w.camera = next;
  }
  const bool relatched = d.clipmap.relatchAll || d.clipmap.level >= 0;
  d.exempt = d.held || exempt_frame(d.clipmap, {.full = d.full,
                                                .sceneEpochChanged = sig.sceneEpoch,
                                                .cut = d.cut,
                                                .coverageComplete = in.coverageComplete});
  w.clock = d.full ? 1 : w.clock + 1;

  w.clipmap.fill(d.views);
  ViewsHeader& h = d.views.header;
  h.frameIndex = in.frameIndex;
  h.clock = w.clock;
  h.flags = (in.producing ? header_flag::Producing : 0u) | (d.full ? header_flag::Full : 0u) |
            (d.exempt ? header_flag::Exempt : 0u) | (relatched ? header_flag::Relatch : 0u) |
            (d.held ? header_flag::Held : 0u) | (check::config().on ? header_flag::Check : 0u);
  h.fullReason = reasons;
  h.budget = d.exempt ? in.pages : in.budget;
  h.poolPages = in.pages;
  h.poolPagesX = in.pagesX;
  h.levelBias = in.levelBias;
  for (size_t i = 0; i < 3; ++i) {
    h.sunDir[i] = static_cast<float>(input.dir[i]);
  }
  h.lightId = in.lightId;
  h.maxPairs = MaxPairs;
  h.generation = in.generation;

  w.started = true;
  w.held = d.held;
  w.lastFrameIndex = in.frameIndex;
  w.generation = in.generation;
  w.mask = 0;
  w.expected = in.chainMask;
  testing::WorkerCounts& c = w.counts;
  ++c.frames;
  c.fulls += d.full ? 1 : 0;
  c.held += d.held ? 1 : 0;
  c.exempt += d.exempt ? 1 : 0;
  c.exemptCut += d.cut ? 1 : 0;
  c.exemptCoverage += in.coverageComplete ? 0 : 1;
  c.exemptEpoch += sig.sceneEpoch ? 1 : 0;
  for (uint32_t i = 0; i < full_reason::Count; ++i) {
    c.reasons[i] += (reasons >> i) & 1u;
  }
  if (d.clipmap.relatchAll) {
    for (uint64_t& n : c.relatches) {
      ++n;
    }
  } else if (d.clipmap.level >= 0) {
    ++(d.clipmap.recentre ? c.recentres : c.relatches)[static_cast<size_t>(d.clipmap.level)];
  }
  check::record_views(d.views, in.hintFrame);
  return d;
}

bool frame_held() noexcept { return g_worker.held; }

UpdateStats apply_update(const Views& views, std::span<uint32_t> table, std::span<uint32_t> phys) noexcept {
  UpdateStats stats;
  const auto free_entry = [&](uint32_t i, uint32_t e, uint32_t& counter) {
    table[i] = 0;
    if ((e & entry::Allocated) == 0) {
      return;
    }
    const size_t p = size_t{e & entry::PhysMask} * PhysWords;
    if (p + 1 < phys.size()) {
      phys[p] = 0;
      phys[p + 1] = 0;
    }
    ++counter;
  };
  const size_t count = std::min<size_t>(table.size(), MaxVirtualPages);
  if ((views.header.flags & header_flag::Full) != 0) {
    for (size_t i = 0; i < count; ++i) {
      stats.freedFull += (table[i] & entry::Allocated) != 0 ? 1 : 0;
      table[i] = 0;
    }
    std::ranges::fill(phys, 0u);
    return stats;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t e = table[i];
    if (e == 0) {
      continue;
    }
    const ViewRecord& v = views.views[i / PagesPerView];
    const uint32_t flags = v.view_flags();
    if ((flags & view_flag::Active) == 0 || (flags & view_flag::Freed) != 0) {
      free_entry(i, e, stats.freedRelatch);
      continue;
    }
    if ((flags & view_flag::Recentred) != 0) {
      free_entry(i, e, stats.freedRecentre);
      continue;
    }
    if ((flags & view_flag::Wrap) == 0) {
      continue;
    }
    const uint32_t local = i - std::min(i, v.table_offset());
    const uint32_t sx = local % WindowPages;
    const uint32_t sy = (local / WindowPages) % WindowPages;
    if (!kept_after_shift(sx, sy, {v.prevOrigin[0], v.prevOrigin[1]}, {v.origin[0], v.origin[1]})) {
      free_entry(i, e, stats.freedWrap);
    }
  }
  return stats;
}

void initialize() {
  for (uint32_t i = 0; i < HandleCount; ++i) {
    if (g_handles[i].id == 0) {
      g_handles[i] = detail::create_persistent(HandleNames[i], i == PoolHandle ? detail::PersistentKind::Texture
                                                                               : detail::PersistentKind::Buffer);
    }
  }
  const std::array<detail::PassDesc, PassCount> descs = pass_descs();
  for (size_t i = 0; i < PassCount; ++i) {
    if (g_passes[i] == detail::InvalidPassId) {
      g_passes[i] = detail::register_pass(descs[i]);
    }
  }
  if (g_castBitsPass == detail::InvalidPassId) {
    g_castBitsPass = detail::register_pass(cast_bits_pass_desc());
  }
}

void shutdown() {
  check::shutdown();
  log_summary();
  alloc_shutdown();
  raster_shutdown();
  for (PassId& pass : g_passes) {
    if (pass != detail::InvalidPassId) {
      detail::unregister_pass(pass);
      pass = detail::InvalidPassId;
    }
  }
  if (g_castBitsPass != detail::InvalidPassId) {
    detail::unregister_pass(g_castBitsPass);
    g_castBitsPass = detail::InvalidPassId;
  }
  destroy_resources();
  for (PersistentHandle& h : g_handles) {
    if (h.id != 0) {
      detail::destroy_persistent(h);
      h = {};
    }
  }
  testing::reset();
}

void latch_hints() noexcept {
  const SceneHints& hints = detail::live_scene_hints();
  g_hints.frame = hints.hintFrame;
  g_hints.mainCamera = camera_usable(&hints);
  g_hints.sun = select_sun(hints);
  g_hints.keyRoom = false;
  if (hints.has(detail::SceneHasLights)) {
    for (const AuroraLight& light : hints.lights) {
      g_hints.keyRoom =
          g_hints.keyRoom || (light.type == AURORA_LIGHT_DIRECTIONAL && (light.flags & AURORA_LIGHT_KEY_ROOM) != 0);
    }
  }
  const bool environment = hints.has(detail::SceneHasEnvironment);
  g_hints.noSun = environment && (hints.environment.value.flags & AURORA_ENV_NO_SUN) != 0;
  const float upm = environment ? hints.environment.value.unitsPerMeter : 0.f;
  g_hints.unitsPerMetre = std::isfinite(upm) && upm > 0.f ? upm : DefaultUnitsPerMetre;
}

void latch_demand(const Settings& settings) noexcept {
  g_wanted = settings.shadows.quality != Quality::Off;
  scene::retained::set_demand(scene::retained::Demand::Consumer, g_wanted);
  scene::retained::set_cover_filter(g_wanted ? &is_caster : nullptr);
}

void latch_gate(const Settings& settings, const Caps& caps) noexcept {
  g_wanted = settings.shadows.quality != Quality::Off;
  g_extent = pool_extent(settings.shadows.pages, caps.limits.maxTextureDimension2D);
  if (g_wanted && g_extent.fits && g_extent.reduced && !std::exchange(g_warnedReduced, true)) {
    Log.warn("VSM: the page pool shrinks from {} to {} pages to fit a {} texel texture", settings.shadows.pages,
             g_extent.pages, caps.limits.maxTextureDimension2D);
  }
  const bool next = g_wanted && scene::retained::active() && caps.core &&
                    stage_binding_limits(caps, ShaderStage::Compute).storageBuffers >= MaxPassStorageBuffers &&
                    stage_binding_limits(caps, ShaderStage::Vertex).storageBuffers >= RasterVertexStorage &&
                    gfx::layout_has_gbuffer(gfx::scene_render_target_layout()) && g_extent.fits;
  if (next != g_active) {
    Log.info("VSM: sun virtual shadow maps {}", next ? "active" : "inactive");
  }
  g_active = next;
}

void latch_resources(bool on) noexcept {
  if (!on) {
    if (g_resourcesLive) {
      destroy_resources();
    }
    return;
  }
  if (!webgpu::g_device || g_handles[TableHandle].id == 0) {
    return;
  }
  create_resources();
  g_resourcesLive = true;
}

void on_latch(const detail::PassSetup& setup, const detail::CompiledGraph* graph) {
  g_chainReadyLatched = g_fifoReady.load(std::memory_order_relaxed);
  const debug::GbufView* row = debug::find_gbuf_view(debug::selected_view_id(detail::latched_settings()));
  g_shadowView.store(row != nullptr && debug::shadow_view(*row), std::memory_order_relaxed);
  {
    std::lock_guard lock{g_chainMutex};
    if (!g_pendingChain.empty()) {
      std::ranges::sort(g_pendingChain);
      g_pendingChain.erase(std::ranges::unique(g_pendingChain).begin(), g_pendingChain.end());
      g_chainIds = std::exchange(g_pendingChain, {});
    }
  }
  g_expectedMask = expected_mask(graph);
  record_rate(g_hints.sun);
  if (!g_planOverride) {
    g_plan = decide_plan(
        {
            .active = active(),
            .chainReady = g_chainReadyLatched,
            .mainCamera = g_hints.mainCamera,
            .noSun = g_hints.noSun,
            .sun = g_hints.sun,
            .frame = g_hints.frame,
        },
        g_lowSunOn);
  }
  sun_visibility::on_latch(setup.settings);
  debug::vsm_view::on_latch(setup.settings);
  const PassId updatePass = g_passes[0];
  if (!active() || updatePass == detail::InvalidPassId) {
    if (updatePass != detail::InvalidPassId) {
      detail::clear_pass_params(updatePass);
    }
    return;
  }
  ++g_latches;
  ++g_idle[static_cast<size_t>(plan().idle)];
  const SunPlan& p = plan();
  const Settings* settings = setup.settings;
  UpdateParams params;
  params.producing = p.producing ? 1u : 0u;
  params.lightId = p.lightId;
  params.lightFlags = p.lightFlags;
  params.quality = settings != nullptr ? static_cast<uint32_t>(settings->shadows.quality) : 0u;
  std::copy(p.dirWorld.begin(), p.dirWorld.end(), params.dir);
  params.texel0 = Texel0Metres * g_hints.unitsPerMetre;
  params.elevationDeg = static_cast<float>(p.elevationDeg);
  uint32_t height = 0;
  if (graph != nullptr) {
    height = graph->sceneSize.height != 0 ? graph->sceneSize.height : graph->efbSize.height;
  }
  const double scale = std::max(1.0, static_cast<double>(height) / ScreenHeightReference);
  params.levelBias =
      static_cast<float>(std::log2(scale)) + (settings != nullptr ? quality_bias(settings->shadows.quality) : 0.f);
  params.pages = g_extent.pages;
  params.pagesX = g_extent.pagesX;
  params.budget =
      settings != nullptr ? std::clamp(settings->shadows.pageBudget, MinPageBudget, MaxPageBudget) : MinPageBudget;
  params.generation = g_generation;
  params.chainMask = g_expectedMask;
  params.planFrame = p.frame;
  detail::set_pass_params(updatePass, std::as_bytes(std::span{&params, 1}));
}

bool active() noexcept { return g_activeOverride.value_or(g_active); }

bool chain_ready_latched() noexcept { return g_chainReadyLatched; }

bool ready() noexcept { return !active() || (g_resourcesLive && g_chainReadyLatched); }

void register_chain_pipeline(shaders::PipelineId id) noexcept {
  if (!id) {
    return;
  }
  std::lock_guard lock{g_chainMutex};
  g_pendingChain.push_back(id.value);
}

void register_consumer_term(ConsumerTerm term) noexcept {
  if (term == nullptr ||
      std::find(g_terms.begin(), g_terms.begin() + g_termCount, term) != g_terms.begin() + g_termCount ||
      g_termCount == g_terms.size()) {
    return;
  }
  g_terms[g_termCount++] = term;
}

bool frame_producing(const StageData& stage) noexcept {
  return plan().producing && camera_usable(stage.sceneHints.get()) && !stage.flushedAtFinish &&
         stage.retained != nullptr && stage.retained->plan != nullptr;
}

bool sun_term_consumes() noexcept { return g_sunTermOverride.value_or(detail::composite::sun_term_applied()); }

bool shadow_view_selected() noexcept {
  return g_shadowViewOverride.value_or(g_shadowView.load(std::memory_order_relaxed));
}

bool key_room_only() noexcept { return g_hints.keyRoom && !g_hints.sun.found; }

uint32_t pool_pages() noexcept { return g_liveExtent.pages; }

uint32_t pool_generation() noexcept { return g_generation; }

bool consumed(const StageData& stage) noexcept {
  // Persistent writers are graph sinks, so only this term stops the chain when nothing reads it
  if (g_forceOverride.value_or(env_force()) || sun_term_consumes() || shadow_view_selected()) {
    return true;
  }
  for (uint32_t i = 0; i < g_termCount; ++i) {
    if (g_terms[i](stage)) {
      return true;
    }
  }
  return false;
}

bool chain_enabled(const StageData& stage) noexcept {
  const bool ready = chain_snapshot(stage);
  return ready && frame_producing(stage) && consumed(stage);
}

void mark_encoded(uint32_t bit) noexcept { g_worker.mask |= bit; }

uint32_t encoded_mask() noexcept { return g_worker.mask; }

uint32_t expected_chain() noexcept { return g_worker.expected; }

PersistentHandle handle(std::string_view name) noexcept {
  for (uint32_t i = 0; i < HandleCount; ++i) {
    if (HandleNames[i] == name) {
      return g_handles[i];
    }
  }
  return {};
}

std::array<detail::PassDesc, PassCount> pass_descs() {
  return {
      detail::PassDesc{
          .name = std::string{UpdatePass},
          .stage = Stage::OpaqueDone,
          .kind = PassKind::Compute,
          .cls = detail::PassClass::Required,
          .order = 0,
          .paramsSize = sizeof(UpdateParams),
          .declare = declare_update,
          .enabled = update_enabled,
          .encode = encode_update,
          .pipelines = update_pipelines,
          .stageParams = update_stage_params,
      },
      scan_pass_desc(),
      scan_list_pass_desc(),
      chunk_bounds_pass_desc(),
      dirty_pass_desc(),
      mark_pass_desc(),
      classify_pass_desc(),
      alloc_lists_pass_desc(),
      assign_pass_desc(),
      cull_candidates_pass_desc(),
      cull_list_pass_desc(),
      cull_count_pass_desc(),
      cull_scan_pass_desc(),
      cull_skip_pass_desc(),
      cull_emit_pass_desc(),
      clear_pass_desc(),
      raster_pass_desc(),
      commit_pass_desc(),
      check::copy_pass_desc(),
      check::compare_pass_desc(),
      check::pass_desc(),
  };
}

detail::PassDesc cast_bits_pass_desc() {
  return {
      .name = std::string{fakeshadow::gate::CastBitsPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .cls = detail::PassClass::Required,
      .order = fakeshadow::gate::CastBitsOrder,
      .declare = declare_cast_bits,
      .enabled = cast_bits_enabled,
      .encode = encode_cast_bits,
  };
}

namespace testing {
void reset() noexcept {
  g_activeOverride.reset();
  g_planOverride.reset();
  g_wanted = false;
  g_active = false;
  g_extent = {};
  g_warnedReduced = false;
  g_hints = {};
  g_lowSunOn = false;
  g_plan = {};
  g_chainReadyLatched = false;
  g_expectedMask = 0;
  g_latches = 0;
  g_idle = {};
  g_rate = {};
  g_terms = {};
  g_termCount = 0;
  {
    std::lock_guard lock{g_chainMutex};
    g_chainIds.clear();
    g_pendingChain.clear();
    g_pipelineQuery = nullptr;
    g_compactOverride.reset();
    g_forceOverride.reset();
  }
  g_shadowView.store(false, std::memory_order_relaxed);
  g_sunTermOverride.reset();
  g_shadowViewOverride.reset();
  g_snapFrame = UINT64_MAX;
  g_snapReady = false;
  g_fifoReady.store(false, std::memory_order_relaxed);
  g_worker = {};
  check::testing::reset();
  set_atlas_handle(std::nullopt);
}

void force_consumed(std::optional<bool> force) noexcept { g_forceOverride = force; }

void set_active(std::optional<bool> on) noexcept { g_activeOverride = on; }

void set_plan(const SunPlan& p) noexcept { g_planOverride = p; }

void set_pipeline_query(bool (*query)(uint32_t id) noexcept) noexcept {
  std::lock_guard lock{g_chainMutex};
  g_pipelineQuery = query;
}

void set_compact_ready(std::optional<bool> ready) noexcept { g_compactOverride = ready; }

void set_sun_term(std::optional<bool> applied) noexcept { g_sunTermOverride = applied; }

void set_shadow_view(std::optional<bool> selected) noexcept { g_shadowViewOverride = selected; }

std::span<const uint32_t> chain_pipelines() noexcept { return g_chainIds; }

WorkerCounts worker_counts() noexcept { return g_worker.counts; }
} // namespace testing
} // namespace aurora::render::shadows::vsm

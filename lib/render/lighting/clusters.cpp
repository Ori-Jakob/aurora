#include "clusters.hpp"

#include "../debug/views.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../scene/receiver_category.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gx/hints.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::detail::lighting {
bool lighting_wanted(const PassSetup& setup) noexcept {
  return setup.settings != nullptr && setup.env != nullptr && setup.settings->pbr &&
         setup.gx_available(GxResource::Surface) && setup.gx_available(GxResource::Normal) &&
         setup.sceneDepth == DepthSource::Direct && setup.env->gxSampleCount == 1;
}
} // namespace aurora::render::detail::lighting

namespace aurora::render::detail::lighting::clusters {
namespace {
constexpr Module Log{"aurora::render::lighting"};
constexpr std::string_view DeferredPass = "light.deferred";
constexpr std::string_view CompositePass = "composite.main";
constexpr BufRef ListRef{0};
constexpr BufRef GridRef{1};
constexpr TexRef HeatmapTarget{3};
constexpr double MinDirection = 1e-6;

struct DevHooks {
  uint32_t stress = 0;
  bool bruteForce = false;
  uint32_t skipMask = 0;
  std::optional<uint64_t> dumpFrame;
};

struct LatchSlot {
  bool live = false;
  bool armed = false;
  std::array<uint32_t, MaxPassPipelines> ids{};
};

struct Totals {
  uint32_t frames = 0;
  uint32_t maxLights = 0;
  uint32_t maxDirectional = 0;
  uint32_t maxLocal = 0;
  uint32_t invalid = 0;
  uint32_t skipped = 0;
  uint32_t truncated = 0;
  uint32_t widened = 0;
};

PassId g_pass = InvalidPassId;
PassId g_heatmap = InvalidPassId;
std::mutex g_slotMutex;
LatchSlot g_slot;
std::atomic_bool g_armed{false};
std::shared_ptr<const SceneHints> g_hintsOverride;
std::optional<bool> g_bruteOverride;
std::mutex g_totalsMutex;
Totals g_totals;
std::atomic_bool g_widenWarned{false};
std::atomic_uint32_t g_dispatches{0};
std::atomic_uint32_t g_heatmapDispatches{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

template <typename T>
std::optional<T> parse_number(std::string_view text, int base) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

const DevHooks& dev_hooks() noexcept {
  static const DevHooks hooks = [] {
    DevHooks out;
    if (const char* value = SDL_getenv("AURORA_RENDER_LIGHT_STRESS")) {
      const std::optional<uint32_t> count = parse_number<uint32_t>(value, 10);
      if (count && *count <= MaxLights) {
        Log.warn("AURORA_RENDER_LIGHT_STRESS={}: adding seeded test lights, for verification only", value);
        out.stress = *count;
      } else {
        Log.warn("AURORA_RENDER_LIGHT_STRESS={} is not a light count from 0 to {}; ignored", value, MaxLights);
      }
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_LIGHT_CLUSTERS")) {
      if (std::string_view{value} == "off") {
        Log.warn("AURORA_RENDER_LIGHT_CLUSTERS=off: every cluster lists every light, for verification only");
        out.bruteForce = true;
      } else {
        Log.warn("AURORA_RENDER_LIGHT_CLUSTERS={} is not off; ignored", value);
      }
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_LIGHT_SKIP")) {
      std::string_view text{value};
      if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
      }
      const std::optional<uint32_t> mask = parse_number<uint32_t>(text, 16);
      if (mask && *mask <= 0xFFFFu) {
        Log.warn("AURORA_RENDER_LIGHT_SKIP={}: dropping lights with these flags, for verification only", value);
        out.skipMask = *mask;
      } else {
        Log.warn("AURORA_RENDER_LIGHT_SKIP={} is not a hex light flag mask; ignored", value);
      }
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_LIGHTS_DUMP")) {
      if (const std::optional<uint64_t> frame = parse_number<uint64_t>(value, 10)) {
        Log.warn("AURORA_RENDER_LIGHTS_DUMP={}: logging the prepared light list of that hint frame", value);
        out.dumpFrame = frame;
      } else {
        Log.warn("AURORA_RENDER_LIGHTS_DUMP={} is not a hint frame number; ignored", value);
      }
    }
    return out;
  }();
  return hooks;
}

const SceneHints* stage_hints(const StageData& stage) noexcept {
  return g_hintsOverride != nullptr ? g_hintsOverride.get() : stage.sceneHints.get();
}

bool camera_usable(const SceneHints* hints) noexcept {
  return hints != nullptr && hints->has(SceneHasMainCamera) && gx::hints::main_camera_usable(hints->mainCamera.gxProj);
}

bool node_live(const CompiledGraph& graph, std::string_view name) noexcept {
  const NodeId node = graph.find_node(name);
  return node != InvalidNode && graph.nodes[node].kind == NodeKind::Pass &&
         graph.nodes[node].status == CullReason::None;
}

bool heatmap_selected(uint16_t view) noexcept {
  const debug::GbufView* row = debug::find_gbuf_view(view);
  return row != nullptr && row->resource == HeatmapResource;
}

using Rows = std::array<std::array<double, 4>, 3>;

bool finite3(const f32 (&v)[3]) noexcept { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

std::array<double, 3> to_view(const Rows& rows, const f32 (&v)[3], double w) noexcept {
  std::array<double, 3> out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = rows[i][0] * v[0] + rows[i][1] * v[1] + rows[i][2] * v[2] + rows[i][3] * w;
  }
  out[2] = -out[2];
  return out;
}

std::optional<GpuLight> prepare_one(const AuroraLight& in, uint32_t index, const Rows& rows, double scale,
                                    uint32_t skipMask, LightListHeader& header, UploadStats& stats) noexcept {
  const bool local = in.type != AURORA_LIGHT_DIRECTIONAL;
  const bool spot = in.type == AURORA_LIGHT_SPOT;
  bool valid = in.type <= AURORA_LIGHT_SPOT && finite3(in.color);
  std::array<float, 3> colour{};
  for (size_t i = 0; valid && i < 3; ++i) {
    colour[i] = std::clamp(in.color[i], 0.f, MaxColour);
  }
  valid = valid && (colour[0] > 0.f || colour[1] > 0.f || colour[2] > 0.f);
  valid = valid && (!local || (finite3(in.position) && std::isfinite(in.radius) && in.radius > 0.f));
  valid = valid && ((local && !spot) || finite3(in.direction));
  valid = valid && (!spot || (std::isfinite(in.spotInnerCos) && std::isfinite(in.spotOuterCos)));
  std::array<double, 3> direction{};
  if (valid && (!local || spot)) {
    direction = to_view(rows, in.direction, 0.0);
    const double length = std::hypot(direction[0], direction[1], direction[2]);
    valid = std::hypot(double{in.direction[0]}, double{in.direction[1]}, double{in.direction[2]}) >= MinDirection &&
            length >= MinDirection && std::isfinite(length);
    for (double& d : direction) {
      d /= valid ? length : 1.0;
    }
  }
  const double radius = local ? double{in.radius} * scale : 0.0;
  valid = valid && (!local || (std::isfinite(radius) && radius > 0.0));
  if (!valid) {
    ++stats.invalid;
    return std::nullopt;
  }
  if ((in.flags & skipMask) != 0) {
    ++stats.skipped;
    return std::nullopt;
  }
  const scene::receiver_category::LightByte byte = scene::receiver_category::light_byte(in.categoryMask);
  if (byte.widened) {
    ++stats.widened;
    header.flags |= HeaderMaskWidened;
  }
  const float p = std::isfinite(in.falloffExponent) ? std::clamp(in.falloffExponent, 0.f, MaxFalloff) : DefaultFalloff;
  const double k = local && p > 0.f ? 254.0 / std::pow(radius, double{p}) : 254.0;
  float spotScale = 0.f;
  float spotOffset = 1.f;
  if (spot) {
    float inner = std::clamp(in.spotInnerCos, -1.f, 1.f);
    float outer = std::clamp(in.spotOuterCos, -1.f, 1.f);
    if (inner < outer) {
      std::swap(inner, outer);
    }
    spotScale = 1.f / std::max(inner - outer, 1e-4f);
    spotOffset = -outer * spotScale;
  }
  const float sourceRadius = std::isfinite(in.sourceRadius) ? std::max(in.sourceRadius, 0.f) : 0.f;
  GpuLight out;
  if (local) {
    const std::array<double, 3> position = to_view(rows, in.position, 1.0);
    out.v0 = {static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2]),
              static_cast<float>(radius)};
  }
  out.v1 = {static_cast<float>(direction[0]), static_cast<float>(direction[1]), static_cast<float>(direction[2]),
            spotScale};
  out.v2 = {colour[0], colour[1], colour[2],
            static_cast<float>(std::min(k, double{std::numeric_limits<float>::max()}))};
  out.v3 = {p, spotOffset, static_cast<float>(double{sourceRadius} * scale),
            std::isfinite(in.volumetricScale) ? in.volumetricScale : 1.f};
  out.v4 = {uint32_t{in.type} | uint32_t{byte.byte} << 8 | uint32_t{in.flags} << 16, in.id,
            uint32_t{in.shadowPriority} | index << 8, 0};
  if (!std::isfinite(out.v0[0]) || !std::isfinite(out.v0[1]) || !std::isfinite(out.v0[2]) ||
      !std::isfinite(out.v0[3])) {
    ++stats.invalid;
    return std::nullopt;
  }
  return out;
}

uint64_t splitmix(uint64_t& state) noexcept {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

float unit(uint64_t& state) noexcept { return static_cast<float>(splitmix(state) >> 40) * (1.f / 16777216.f); }

void dump(const SceneHints& hints, const LightListHeader& header, std::span<const GpuLight> lights) {
  Log.info("Render lights dump, hint frame {}: {} directional, {} local, flags 0x{:x}, units per metre {:g}",
           hints.hintFrame, header.dirCount, header.localCount, header.flags, header.unitsPerMetre);
  for (size_t i = 0; i < lights.size(); ++i) {
    const GpuLight& l = lights[i];
    Log.info(
        "  [{}] id 0x{:08x} type {} flags 0x{:x} byte 0x{:02x} hint {} pos ({:g}, {:g}, {:g}) r {:g} dir ({:g}, "
        "{:g}, {:g}) colour ({:g}, {:g}, {:g}) p {:g} k {:g} spot ({:g}, {:g})",
        i, l.v4[1], light_type(l), light_flags(l), light_byte(l), hint_index(l), l.v0[0], l.v0[1], l.v0[2], l.v0[3],
        l.v1[0], l.v1[1], l.v1[2], l.v2[0], l.v2[1], l.v2[2], l.v3[0], l.v2[3], l.v1[3], l.v3[1]);
  }
}

void record_frame(const LightListHeader& header, const UploadStats& stats) {
  std::lock_guard lock{g_totalsMutex};
  ++g_totals.frames;
  const uint32_t count = header.dirCount + header.localCount;
  if (count > g_totals.maxLights || g_totals.frames == 1) {
    g_totals.maxLights = count;
    g_totals.maxDirectional = header.dirCount;
    g_totals.maxLocal = header.localCount;
  }
  g_totals.invalid += stats.invalid;
  g_totals.skipped += stats.skipped;
  g_totals.truncated += stats.truncated;
  g_totals.widened += stats.widened;
}

void fill_params(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params;
  fill_stage_camera(stage_hints(stage), params.camera);
  params.flags = g_bruteOverride.value_or(dev_hooks().bruteForce) ? uint32_t{ParamBruteForce} : 0u;
  std::memcpy(bytes.data(), &params, sizeof(Params));
}

bool scene_enabled(const StageData& stage, const void*) {
  return !stage.flushedAtFinish && camera_usable(stage_hints(stage));
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!lighting_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.create_buffer(List, {.size = ListBytes, .extraUsage = wgpu::BufferUsage::CopyDst}, Access::StorageWrite);
  builder.create_buffer(Grid, {.size = GridBytes}, Access::StorageWrite);
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "light.clusters",
                                        .shader = Shader,
                                        .entryPoint = "assign",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const BufferView list = ctx.buffer(ListRef);
  const BufferView grid = ctx.buffer(GridRef);
  if (!pipeline || !list.buffer || !grid.buffer || list.size < ListBytes || grid.size < GridBytes) {
    return;
  }
  const DevHooks& hooks = dev_hooks();
  std::vector<GpuLight> lights(MaxLights);
  const LightSpan span{lights.data(), MaxLights};
  LightListHeader header;
  UploadStats stats;
  const SceneHints* hints = stage_hints(ctx.stage_data());
  if (hints != nullptr) {
    prepare_lights(*hints, hooks.skipMask, span, header, stats);
  }
  if (hooks.stress != 0) {
    Params params;
    std::memcpy(&params, ctx.params().data(), std::min(ctx.params().size(), sizeof(Params)));
    append_stress_lights(hooks.stress, params.camera, span, header);
  }
  const uint32_t count = header.dirCount + header.localCount;
  if (hooks.dumpFrame && hints != nullptr && hints->hintFrame == *hooks.dumpFrame) {
    dump(*hints, header, span.first(count));
  }
  if (stats.widened != 0 && !g_widenWarned.exchange(true, std::memory_order_relaxed)) {
    Log.warn("Render lights: {} light masks name categories outside the receiver byte; those lights light everything",
             stats.widened);
  }
  const wgpu::CommandEncoder& encoder = ctx.encoder();
  encoder.WriteBuffer(list.buffer, list.offset, reinterpret_cast<const uint8_t*>(&header), sizeof(header));
  if (count != 0) {
    encoder.WriteBuffer(list.buffer, list.offset + sizeof(header), reinterpret_cast<const uint8_t*>(lights.data()),
                        uint64_t{count} * sizeof(GpuLight));
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(ClusterCount / WorkgroupSize);
  pass.End();
  bump(g_dispatches);
  record_frame(header, stats);
}

void declare_heatmap(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!lighting_wanted(setup) || !heatmap_selected(setup.debug_view())) {
    builder.skip();
    return;
  }
  builder.after(phase::Lighting);
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use_buffer(List, Access::StorageRead, Missing::DisablePass);
  builder.use_buffer(Grid, Access::StorageRead, Missing::DisablePass);
  builder.create(HeatmapResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                 Access::StorageWrite);
}

void request_heatmap(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "light.heatmap",
                                        .shader = HeatmapShader,
                                        .entryPoint = "heatmap",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode_heatmap(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::Texture target = ctx.texture(HeatmapTarget);
  if (!pipeline || !target) {
    return;
  }
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const auto groups = [](uint32_t size) { return (size + HeatmapWorkgroupSize - 1) / HeatmapWorkgroupSize; };
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(groups(target.GetWidth()), groups(target.GetHeight()));
  pass.End();
  bump(g_heatmapDispatches);
}
} // namespace

void prepare_lights(const SceneHints& hints, uint32_t skipMask, LightSpan lights, LightListHeader& header,
                    UploadStats& stats) noexcept {
  const float upm = units_per_metre(hints);
  header = {
      .flags = skipMask != 0 ? uint32_t{HeaderSkipMask} : 0u,
      .sliceNear = SliceNearMetres * upm,
      .sliceLogScale = static_cast<float>(Slices / std::log(double{SliceFarMetres} / double{SliceNearMetres})),
      .unitsPerMetre = upm,
  };
  if (!hints.has(SceneHasLights)) {
    return;
  }
  const Mat3x4<float>& view = hints.mainCamera.view;
  Rows rows{};
  for (size_t i = 0; i < 3; ++i) {
    const Vec4<float>& row = i == 0 ? view.m0 : i == 1 ? view.m1 : view.m2;
    rows[i] = {row[0], row[1], row[2], row[3]};
  }
  const double scale = std::hypot(rows[0][0], rows[0][1], rows[0][2]);
  const auto count = static_cast<uint32_t>(std::min<size_t>(hints.lights.size(), std::numeric_limits<uint32_t>::max()));
  for (const bool directional : {true, false}) {
    for (uint32_t i = 0; i < count; ++i) {
      const AuroraLight& in = hints.lights[i];
      if ((in.type == AURORA_LIGHT_DIRECTIONAL) != directional) {
        continue;
      }
      const std::optional<GpuLight> prepared = prepare_one(in, i, rows, scale, skipMask, header, stats);
      if (!prepared) {
        continue;
      }
      const uint32_t used = header.dirCount + header.localCount;
      if ((directional && header.dirCount >= MaxDirectional) || used >= MaxLights) {
        ++stats.truncated;
        header.flags |= HeaderTruncated;
        continue;
      }
      lights[used] = *prepared;
      ++(directional ? header.dirCount : header.localCount);
    }
  }
}

uint32_t append_stress_lights(uint32_t count, const StageCamera& camera, LightSpan lights,
                              LightListHeader& header) noexcept {
  if (count == 0) {
    return 0;
  }
  header.flags |= HeaderStress;
  const uint32_t used = header.dirCount + header.localCount;
  const uint32_t added = std::min(count, MaxLights - std::min(used, MaxLights));
  const float upm = header.unitsPerMetre > 0.f ? header.unitsPerMetre : DefaultUnitsPerMetre;
  const float p00 = std::isfinite(camera.p00) && camera.p00 > 0.f ? camera.p00 : 1.f;
  const float p11 = std::isfinite(camera.p11) && camera.p11 > 0.f ? camera.p11 : 1.f;
  for (uint32_t i = 0; i < added; ++i) {
    uint64_t state = 0x5EED11u + i;
    const float z = upm * (1.f + 59.f * unit(state));
    const float x = (2.f * unit(state) - 1.f) * z / p00;
    const float y = (2.f * unit(state) - 1.f) * z / p11;
    const float r = upm * (1.f + 7.f * unit(state));
    GpuLight& light = lights[used + i];
    light = {};
    light.v0 = {x, y, z, r};
    light.v2 = {0.2f + 0.8f * unit(state), 0.2f + 0.8f * unit(state), 0.2f + 0.8f * unit(state), 254.f / (r * r)};
    light.v3 = {DefaultFalloff, 1.f, 0.f, 1.f};
    light.v4 = {AURORA_LIGHT_POINT, 0xF0000000u | i, 0xFFFFFFu << 8, 0};
  }
  header.localCount += added;
  return added;
}

uint32_t slice_of(float z, const LightListHeader& header) noexcept {
  if (!(z > header.sliceNear)) {
    return 0;
  }
  const float s = std::floor(std::log(z / header.sliceNear) * header.sliceLogScale);
  return static_cast<uint32_t>(std::clamp(s, 0.f, static_cast<float>(Slices - 1)));
}

uint32_t pixel_cluster(uint32_t px, uint32_t py, uint32_t width, uint32_t height, float z,
                       const LightListHeader& header) noexcept {
  return cluster_index(tile_of(px, width, TilesX), tile_of(py, height, TilesY), slice_of(z, header));
}

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 2,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = scene_enabled,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_params,
  };
}

PassDesc heatmap_pass_desc() {
  return {
      .name = std::string{HeatmapPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Optional,
      .order = 1,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare_heatmap,
      .enabled = scene_enabled,
      .encode = encode_heatmap,
      .pipelines = request_heatmap,
      .stageParams = fill_params,
  };
}

void initialize() {
  if (g_pass == InvalidPassId) {
    g_pass = register_pass(pass_desc());
    g_heatmap = register_pass(heatmap_pass_desc());
  }
}

void shutdown() noexcept {
  for (PassId* pass : {&g_pass, &g_heatmap}) {
    if (*pass != InvalidPassId) {
      unregister_pass(*pass);
      *pass = InvalidPassId;
    }
  }
  {
    std::lock_guard lock{g_slotMutex};
    g_slot = {};
  }
  g_armed.store(false, std::memory_order_relaxed);
  std::lock_guard lock{g_totalsMutex};
  const Totals& t = g_totals;
  if (t.frames != 0) {
    Log.info(
        "Render lights: {} frames, up to {} lights ({} directional, {} local), {} invalid, {} truncated, {} "
        "skipped, {} masks widened",
        t.frames, t.maxLights, t.maxDirectional, t.maxLocal, t.invalid, t.truncated, t.skipped, t.widened);
  }
  g_totals = {};
}

void on_latch(const PassSetup& latched, const CompiledGraph* graph) {
  LatchSlot slot;
  if (graph != nullptr && node_live(*graph, Pass)) {
    slot.live = true;
    slot.ids = graph->nodes[graph->find_node(Pass)].pipelineIds;
  }
  const Settings* settings = latched.settings;
  const bool classView = debug::find_view(latched.debug_view()) != nullptr;
  slot.armed = settings != nullptr && graph != nullptr && settings->pbr && settings->legacyBlend < 1.f &&
               node_live(*graph, DeferredPass) && node_live(*graph, CompositePass) && !classView;
  std::lock_guard lock{g_slotMutex};
  g_slot = slot;
  g_armed.store(slot.armed, std::memory_order_relaxed);
}

bool armed() noexcept { return g_armed.load(std::memory_order_relaxed); }

bool ready(const Settings& settings) noexcept {
  if (!settings.pbr) {
    return true;
  }
  std::lock_guard lock{g_slotMutex};
  return g_slot.live && g_slot.ids[0] != 0 && shaders::pipelines_ready(g_slot.ids) && hints_enabled();
}

namespace testing {
Counters counters() noexcept {
  std::lock_guard lock{g_totalsMutex};
  return {
      .frames = g_totals.frames,
      .dispatches = g_dispatches.load(std::memory_order_relaxed),
      .heatmapDispatches = g_heatmapDispatches.load(std::memory_order_relaxed),
      .maxLights = g_totals.maxLights,
      .invalid = g_totals.invalid,
      .skipped = g_totals.skipped,
      .truncated = g_totals.truncated,
      .widened = g_totals.widened,
  };
}

void reset() noexcept {
  g_hintsOverride.reset();
  g_bruteOverride.reset();
  g_widenWarned.store(false, std::memory_order_relaxed);
  g_dispatches.store(0, std::memory_order_relaxed);
  g_heatmapDispatches.store(0, std::memory_order_relaxed);
  std::lock_guard lock{g_totalsMutex};
  g_totals = {};
}

void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept { g_hintsOverride = std::move(hints); }

void set_brute_force(std::optional<bool> bruteForce) noexcept { g_bruteOverride = bruteForce; }
} // namespace testing
} // namespace aurora::render::detail::lighting::clusters

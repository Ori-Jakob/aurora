#include "deferred.hpp"

#include "openpbr.hpp"

#include "../ao/ao.hpp"
#include "../frame_state.hpp"
#include "../gx_ext/unlit.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"

#include "../../gx/hints.hpp"
#include "../../gx/surface_pack.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numbers>
#include <utility>

namespace aurora::render::detail::lighting {
bool ready(const Settings& settings) noexcept { return clusters::ready(settings) && deferred::ready(settings); }
} // namespace aurora::render::detail::lighting

namespace aurora::render::detail::lighting::deferred {
namespace {
constexpr Module Log{"aurora::render::lighting"};
constexpr TexRef HdrRef{BindHdr};
constexpr double Pi = std::numbers::pi;

struct LatchSlot {
  bool live = false;
  std::array<uint32_t, MaxPassPipelines> ids{};
};

PassId g_pass = InvalidPassId;
std::mutex g_slotMutex;
LatchSlot g_slot;
std::atomic_bool g_surfaceAmbient{true};
std::shared_ptr<const SceneHints> g_hintsOverride;
std::optional<Mode> g_modeOverride;
std::optional<bool> g_forceEnabled;
std::atomic_uint32_t g_ran{0};
std::atomic_uint32_t g_armed{0};
std::atomic_uint32_t g_identity{0};
std::atomic_uint32_t g_noLights{0};
std::atomic_uint32_t g_notResident{0};
std::atomic_uint32_t g_pending{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

Mode env_mode() noexcept {
  static const Mode mode = [] {
    const char* value = SDL_getenv(std::string{ModeEnv}.c_str());
    if (value == nullptr) {
      return Mode::Full;
    }
    if (const std::optional<Mode> parsed = parse_mode(value)) {
      Log.warn("{}={}: light.deferred shades a test configuration, for verification only", ModeEnv, value);
      return *parsed;
    }
    Log.warn("{}={} is not grey, direct, indirect, furnace or furnace-metal; ignored", ModeEnv, value);
    return Mode::Full;
  }();
  return mode;
}

Mode current_mode() noexcept { return g_modeOverride.value_or(env_mode()); }

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

LatchSlot latched_slot() {
  std::lock_guard lock{g_slotMutex};
  return g_slot;
}

double srgb_decode(double stored) noexcept {
  const double s = std::clamp(stored, 0.0, 1.0);
  return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}

void fill_stage_params(const StageData& stage, const void*, std::span<std::byte> bytes) {
  if (bytes.size() < sizeof(Params)) {
    return;
  }
  Params params;
  fill_params(stage_hints(stage), g_surfaceAmbient.load(std::memory_order_relaxed), current_mode(), params);
  std::memcpy(bytes.data(), &params, sizeof(Params));
}

bool enabled(const StageData& stage, const void*) {
  if (g_forceEnabled) {
    return *g_forceEnabled;
  }
  if (!clusters::armed()) {
    return false;
  }
  bump(g_armed);
  const SceneHints* hints = stage_hints(stage);
  if (hints == nullptr || !hints->has(SceneHasLights)) {
    bump(g_noLights);
  }
  if (!openpbr::resident()) {
    bump(g_notResident);
  }
  const bool on = frame_enabled(stage);
  if (on && !shaders::pipelines_ready(latched_slot().ids)) {
    bump(g_pending);
  }
  return on;
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!lighting_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Lighting);
  builder.use(resource::Surface, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  builder.use(resource::GxDepth, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  builder.use(resource::AoFinal, Access::Sampled, Missing::BindDummyWhite);
  builder.use_buffer(clusters::List, Access::StorageRead, Missing::DisablePass);
  builder.use_buffer(clusters::Grid, Access::StorageRead, Missing::DisablePass);
  builder.import_persistent(openpbr::Classes, openpbr::classes_handle(), Access::StorageRead);
  builder.import_persistent_texture(openpbr::Lut3d, openpbr::lut3d_handle(), openpbr::lut3d_desc(), Access::Sampled);
  builder.import_persistent_texture(openpbr::Lut2d, openpbr::lut2d_handle(), openpbr::lut2d_desc(), Access::Sampled);
  builder.create(resource::SceneHdr,
                 {.format = HdrFormat,
                  .sizeClass = SizeClass::Efb,
                  .divisor = 1,
                  .extraUsage = wgpu::TextureUsage::TextureBinding},
                 Access::StorageWrite);
  builder.use(resource::SurfaceExt, Access::Sampled,
              UseOptions{.missing = Missing::BindDummyBlack, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use_buffer(resource::SurfaceTable, Access::StorageRead, UseOptions{.missing = Missing::DisablePass});
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  ids[0] = shaders::request_compute({
                                        .label = "light.deferred",
                                        .shader = Shader,
                                        .entryPoint = "shade",
                                        .layout = setup.layout,
                                        .layoutKey = setup.layoutHash,
                                    })
               .value;
}

void encode(PassContext& ctx, const void*) {
  const wgpu::ComputePipeline pipeline = shaders::compute_pipeline(ctx.pipeline(0));
  const wgpu::Texture target = ctx.texture(HdrRef);
  if (!pipeline || !target) {
    return;
  }
  openpbr::refresh_classes(ctx.encoder());
  const wgpu::ComputePassEncoder pass = ctx.begin_compute();
  if (!pass) {
    return;
  }
  const auto groups = [](uint32_t size) { return (size + WorkgroupSize - 1) / WorkgroupSize; };
  pass.SetPipeline(pipeline);
  pass.DispatchWorkgroups(groups(target.GetWidth()), groups(target.GetHeight()));
  pass.End();
  bump(g_ran);
  if (ao::identity_forced()) {
    bump(g_identity);
  }
}

using Vec3 = testing::Vec3;

Vec3 add(const Vec3& a, const Vec3& b) noexcept { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 scale(const Vec3& a, double s) noexcept { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 mul(const Vec3& a, const Vec3& b) noexcept { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }
double dot3(const Vec3& a, const Vec3& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
} // namespace

std::optional<Mode> parse_mode(std::string_view text) noexcept {
  constexpr std::array<std::pair<std::string_view, Mode>, 5> Modes{{
      {"grey", Mode::Grey},
      {"direct", Mode::Direct},
      {"indirect", Mode::Indirect},
      {"furnace", Mode::Furnace},
      {"furnace-metal", Mode::FurnaceMetal},
  }};
  for (const auto& [name, mode] : Modes) {
    if (text == name) {
      return mode;
    }
  }
  return std::nullopt;
}

std::array<float, 3> srgb_to_linear(const float (&stored)[3]) noexcept {
  std::array<float, 3> out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = std::isfinite(stored[i]) ? static_cast<float>(srgb_decode(stored[i])) : 0.f;
  }
  return out;
}

void fill_params(const SceneHints* hints, bool surfaceAmbient, Mode mode, Params& out) noexcept {
  out = {};
  fill_stage_camera(hints, out.camera);
  if (hints != nullptr && hints->has(SceneHasEnvironment)) {
    const std::array<float, 3> ambient = srgb_to_linear(hints->environment.value.ambientColor);
    std::ranges::copy(ambient, out.ambient);
  }
  out.flags = (surfaceAmbient ? uint32_t{FlagSurfaceAmbient} : 0u) | static_cast<uint32_t>(mode) << ModeShift;
}

bool frame_enabled(const StageData& stage) noexcept {
  if (g_forceEnabled) {
    return *g_forceEnabled;
  }
  const SceneHints* hints = stage_hints(stage);
  return clusters::armed() && !stage.flushedAtFinish && camera_usable(hints) && hints->has(SceneHasLights) &&
         openpbr::resident();
}

bool ready(const Settings& settings) noexcept {
  if (!settings.pbr) {
    return true;
  }
  const LatchSlot slot = latched_slot();
  return slot.live && slot.ids[0] != 0 && shaders::pipelines_ready(slot.ids) && hints_enabled() && openpbr::resident();
}

bool ran(const StageData& opaqueDone) noexcept {
  if (g_forceEnabled) {
    return *g_forceEnabled;
  }
  const CompiledGraph* graph = opaqueDone.frame != nullptr ? opaqueDone.frame->graph.get() : nullptr;
  if (graph == nullptr || opaqueDone.stage != Stage::OpaqueDone) {
    return false;
  }
  const NodeId node = graph->find_node(Pass);
  return node != InvalidNode && graph->nodes[node].programIndex != InvalidNode &&
         opaqueDone.is_enabled(graph->nodes[node].programIndex);
}

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = request_pipelines,
      .stageParams = fill_stage_params,
  };
}

void initialize() {
  if (g_pass == InvalidPassId) {
    g_pass = register_pass(pass_desc());
  }
}

void shutdown() noexcept {
  if (g_pass != InvalidPassId) {
    unregister_pass(g_pass);
    g_pass = InvalidPassId;
  }
  {
    std::lock_guard lock{g_slotMutex};
    g_slot = {};
  }
  const testing::Counters c = testing::counters();
  constexpr std::string_view Line =
      "Render lighting: deferred ran on {} frames ({} armed, {} under identity, {} without "
      "light hints, {} not resident, {} pipeline-pending)";
  if (c.ran != 0 || c.armed != 0) {
    Log.info(Line, c.ran, c.armed, c.identity, c.noLights, c.notResident, c.pending);
  } else {
    Log.debug(Line, c.ran, c.armed, c.identity, c.noLights, c.notResident, c.pending);
  }
  for (std::atomic_uint32_t* counter : {&g_ran, &g_armed, &g_identity, &g_noLights, &g_notResident, &g_pending}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void on_latch(const PassSetup&, const CompiledGraph* graph) {
  LatchSlot slot;
  if (graph != nullptr && node_live(*graph, Pass)) {
    slot.live = true;
    slot.ids = graph->nodes[graph->find_node(Pass)].pipelineIds;
  }
  g_surfaceAmbient.store(gx_ext::unlit::classes_enabled(), std::memory_order_relaxed);
  std::lock_guard lock{g_slotMutex};
  g_slot = slot;
}

namespace testing {
Counters counters() noexcept {
  return {
      .ran = g_ran.load(std::memory_order_relaxed),
      .armed = g_armed.load(std::memory_order_relaxed),
      .identity = g_identity.load(std::memory_order_relaxed),
      .noLights = g_noLights.load(std::memory_order_relaxed),
      .notResident = g_notResident.load(std::memory_order_relaxed),
      .pending = g_pending.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_hintsOverride.reset();
  g_modeOverride.reset();
  g_forceEnabled.reset();
  for (std::atomic_uint32_t* counter : {&g_ran, &g_armed, &g_identity, &g_noLights, &g_notResident, &g_pending}) {
    counter->store(0, std::memory_order_relaxed);
  }
}

void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept { g_hintsOverride = std::move(hints); }

void set_mode(std::optional<Mode> mode) noexcept { g_modeOverride = mode; }

void force_enabled(std::optional<bool> enabled) noexcept { g_forceEnabled = enabled; }

double attenuation(const clusters::GpuLight& light, double distance) noexcept {
  const double r = light.v0[3];
  const double p = light.v3[0];
  if (p <= 0.0) {
    const double x = distance / r;
    const double w = std::clamp(1.0 - x * x * x * x, 0.0, 1.0);
    return w * w;
  }
  const double f = 1.0 / (1.0 + light.v2[3] * std::pow(std::max(distance, 0.0), p));
  return std::max(0.0, (f - 1.0 / 255.0) / (1.0 - 1.0 / 255.0));
}

bool category_match(const clusters::GpuLight& light, uint32_t category) noexcept {
  const uint32_t mask = clusters::light_byte(light);
  return mask == 0 || (mask & category) != 0;
}

Vec3 bend_normal(const Vec3& normal, const Vec3& view) noexcept {
  const double length = std::sqrt(dot3(normal, normal));
  Vec3 n = length >= 1e-3 ? scale(normal, 1.0 / length) : view;
  const double nv = dot3(n, view);
  if (nv < 1e-3) {
    const Vec3 bent = add(n, scale(view, 1e-3 - nv));
    const double bentLength = std::sqrt(dot3(bent, bent));
    n = bentLength >= 1e-6 ? scale(bent, 1.0 / bentLength) : view;
  }
  return n;
}

Result shade(const Pixel& px, const clusters::LightListHeader& header, std::span<const clusters::GpuLight> lights,
             const Params& params, std::span<const material::classes::GpuRow, 256> rows) {
  const bool lit = px.classId != 0 && (px.flags & gx::surface::Unlit) == 0 && px.depth > 0.0 && px.cameraValid;
  if (!lit) {
    return {};
  }
  const auto mode = static_cast<Mode>((params.flags & ModeMask) >> ModeShift);
  const bool furnace = mode == Mode::Furnace || mode == Mode::FurnaceMetal;
  const double distance = std::sqrt(dot3(px.position, px.position));
  const Vec3 v = scale(px.position, -1.0 / distance);
  const Vec3 n = bend_normal(px.normal, v);
  openpbr::Surface surface{.roughness = px.roughness, .metalness = px.metalness};
  for (size_t i = 0; i < 3; ++i) {
    surface.baseColor[i] = srgb_decode(px.albedo[i]);
  }
  if (mode == Mode::Grey) {
    surface.baseColor = {0.5, 0.5, 0.5};
  }
  if (furnace) {
    surface.baseColor = {1.0, 1.0, 1.0};
    surface.metalness = mode == Mode::FurnaceMetal ? 1.0 : 0.0;
  }
  const material::classes::GpuRow& row = rows[std::min(px.classId, 255u)];
  const openpbr::Tables& tables = openpbr::testing::tables();
  const openpbr::Prepared p = openpbr::prepare(tables, row, surface, n, v);
  double ao = std::min(px.receiver >= 1 ? std::pow(px.aoFinal, 2.2) : 1.0, px.ao);
  Vec3 a{};
  for (size_t i = 0; i < 3; ++i) {
    a[i] = (params.flags & FlagSurfaceAmbient) != 0 ? srgb_decode(px.ambient[i]) : params.ambient[i];
  }
  if (furnace) {
    a = {1.0, 1.0, 1.0};
    ao = 1.0;
  }
  Vec3 h{};
  if (mode != Mode::Direct) {
    h = scale(openpbr::eval_ambient(p, scale(a, Pi), a), ao);
  }
  if (mode != Mode::Indirect && !furnace) {
    const bool thin = (px.flags & gx::surface::ThinTranslucent) != 0;
    const uint32_t dirCount = std::min(header.dirCount, clusters::MaxDirectional);
    const uint32_t localCount = std::min(header.localCount, clusters::MaxLights - dirCount);
    for (uint32_t i = 0; i < dirCount + localCount && i < lights.size(); ++i) {
      const clusters::GpuLight& l = lights[i];
      Vec3 toLight{};
      double atten = 1.0;
      if (i < dirCount) {
        if (!category_match(l, px.category)) {
          continue;
        }
        toLight = {-l.v1[0], -l.v1[1], -l.v1[2]};
      } else {
        const Vec3 to{l.v0[0] - px.position[0], l.v0[1] - px.position[1], l.v0[2] - px.position[2]};
        const double d2 = dot3(to, to);
        if (d2 >= double{l.v0[3]} * l.v0[3]) {
          continue;
        }
        const double d = std::sqrt(d2);
        toLight = d > 0.0 ? scale(to, 1.0 / d) : n;
        double spot = 1.0;
        if (clusters::light_type(l) == AURORA_LIGHT_SPOT) {
          const double s = std::clamp(-dot3({l.v1[0], l.v1[1], l.v1[2]}, toLight) * l.v1[3] + l.v3[1], 0.0, 1.0);
          spot = s * s;
        }
        if (spot <= 0.0 || !category_match(l, px.category)) {
          continue;
        }
        atten = attenuation(l, d) * spot;
        if (atten <= 0.0) {
          continue;
        }
      }
      if (dot3(n, toLight) <= 0.0 && !thin) {
        continue;
      }
      const bool specular = (clusters::light_flags(l) & AURORA_LIGHT_NO_SPECULAR) == 0;
      const Vec3 colour{l.v2[0], l.v2[1], l.v2[2]};
      h = add(h, mul(colour, scale(openpbr::eval_light(tables, p, toLight, specular), atten)));
    }
  }
  if (mode == Mode::Full) {
    const Vec3 emission{row.v[16] * row.v[19], row.v[17] * row.v[19], row.v[18] * row.v[19]};
    h = add(h, mul(px.emissive, emission));
  }
  Result out;
  for (size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(h[i])) {
      return {};
    }
    out.rgb[i] = std::clamp(h[i], 0.0, double{HdrMax});
  }
  out.alpha = 1.0;
  return out;
}
} // namespace testing
} // namespace aurora::render::detail::lighting::deferred

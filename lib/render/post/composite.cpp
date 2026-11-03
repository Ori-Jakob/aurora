#include "composite.hpp"

#include "../ao/ao.hpp"
#include "../debug/views.hpp"
#include "../fakeshadow/texture_gate.hpp"
#include "../frame_state.hpp"
#include "../hints.hpp"
#include "../lighting/deferred.hpp"
#include "../pass_context.hpp"
#include "../scene/overlay.hpp"
#include "../stage_data.hpp"

#include "../../internal.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iterator>
#include <mutex>
#include <span>
#include <utility>

namespace aurora::render::detail::composite {
namespace {
[[maybe_unused]] constexpr Module Log{"aurora::render::composite"};
constexpr TexRef CopyColor{0};
constexpr TexRef CopyLegacy{1};
constexpr TexRef Legacy{BindLegacy};
constexpr TexRef Fog{BindFog};
constexpr TexRef Normal{BindNormal};
constexpr TexRef AoFinal{BindAo};
constexpr TexRef SurfaceRec{BindSurface};
constexpr TexRef Hdr{BindHdr};
constexpr TexRef Color{BindHdr + 1};
constexpr uint64_t NoFrame = UINT64_MAX;
constexpr wgpu::ColorWriteMask RgbMask =
    wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;

struct LatchSlot {
  const CompiledGraph* graph = nullptr;
  bool live = false;
  bool classView = false;
  bool hdrLive = false;
  bool aoProduced = false;
  bool aoLive = false;
  bool pbrActive = false;
  std::array<uint32_t, 1> ids{};
};

PassId g_copyPass = InvalidPassId;
PassId g_pass = InvalidPassId;
std::mutex g_slotMutex;
LatchSlot g_slot;
std::shared_ptr<const SceneHints> g_cameraOverride;
std::optional<uint32_t> g_flagsOverride;
std::optional<Params> g_paramsOverride;
bool g_contractWarned = false;

std::atomic_uint64_t g_enabledFrame{NoFrame};
std::atomic_bool g_encodeMismatch{false};
std::atomic_bool g_mismatchWarned{false};
std::atomic_uint32_t g_copies{0};
std::atomic_uint32_t g_draws{0};
std::atomic_uint32_t g_skippedMismatch{0};
std::atomic_uint32_t g_skippedNoScene{0};
std::atomic_uint32_t g_skippedView{0};
std::atomic_uint32_t g_contractFailures{0};
std::atomic_uint32_t g_pbrDraws{0};
std::atomic_uint32_t g_idleFrames{0};
std::atomic_uint32_t g_hdrDraws{0};

inline void bump(std::atomic_uint32_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

LatchSlot latched_slot() {
  std::lock_guard lock{g_slotMutex};
  return g_slot;
}

bool no_fog_forced() noexcept {
  static const bool forced = [] {
    const char* value = SDL_getenv(std::string{NoFogEnv}.c_str());
    if (value == nullptr) {
      return false;
    }
    const bool on = params_flags(false, value) == FlagNoFog;
    if (on) {
      Log.warn("{}=on: AO multiplies the stored colour without the fog split, for contrast only", NoFogEnv);
    } else {
      Log.warn("{}={} is not on; ignored", NoFogEnv, value);
    }
    return on;
  }();
  return forced;
}

bool node_live(const CompiledGraph& graph, std::string_view name) noexcept {
  const NodeId node = graph.find_node(name);
  return node != InvalidNode && graph.nodes[node].kind == NodeKind::Pass &&
         graph.nodes[node].status == CullReason::None;
}

bool same_extent(const wgpu::Extent3D& a, const wgpu::Extent3D& b) noexcept {
  return a.width == b.width && a.height == b.height;
}

const CompiledUse* composite_use(const CompiledGraph& graph, std::string_view resource) noexcept {
  const NodeId node = graph.find_node(Pass);
  const ResourceId id = graph.find_resource(resource);
  if (node == InvalidNode || id == InvalidResource || graph.nodes[node].kind != NodeKind::Pass ||
      graph.nodes[node].status != CullReason::None) {
    return nullptr;
  }
  for (const CompiledUse& use : graph.node_uses(node)) {
    if (use.resource == id) {
      return &use;
    }
  }
  return nullptr;
}

bool params_carry_pbr(std::span<const std::byte> bytes) noexcept {
  Params params;
  if (bytes.size() < sizeof(params)) {
    return false;
  }
  std::memcpy(&params, bytes.data(), sizeof(params));
  return (params.flags & FlagPbr) != 0;
}

bool ready_for(const LatchSlot& slot, const Settings& settings) noexcept {
  if (!wanted(settings)) {
    return true;
  }
  return slot.live && hints_enabled() && slot.ids[0] != 0 && shaders::pipelines_ready(slot.ids);
}

bool legacy_matches(const PassContext& ctx, TexRef color, TexRef legacy) {
  return ctx.format(color) == ctx.format(legacy) && same_extent(ctx.extent(color), ctx.extent(legacy)) &&
         ctx.sample_count(color) == 1;
}

void encode_mismatch(const PassContext& ctx, TexRef color, TexRef other, std::string_view what) {
  if (!g_mismatchWarned.exchange(true, std::memory_order_relaxed)) {
    const wgpu::Extent3D a = ctx.extent(color);
    const wgpu::Extent3D b = ctx.extent(other);
    Log.warn("{}: gx.color is {} {}x{} ({} samples) but {} is {} {}x{}; this frame keeps the legacy colour", ctx.name(),
             magic_enum::enum_name(ctx.format(color)), a.width, a.height, ctx.sample_count(color), what,
             magic_enum::enum_name(ctx.format(other)), b.width, b.height);
  }
  bump(g_skippedMismatch);
  g_encodeMismatch.store(true, std::memory_order_relaxed);
}

void declare_copy(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declare_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Prepare);
  builder.use(resource::GxColor, Access::CopySrc);
  // The copy keeps the EFB format, which differs between backends' swapchains
  builder.create(LegacyTex,
                 {.format = setup.env->gxFormats[static_cast<size_t>(GxResource::Color)], .sizeClass = SizeClass::Efb},
                 Access::CopyDst);
}

void encode_copy(PassContext& ctx, const void*) {
  const uint32_t samples = ctx.sample_count(CopyColor);
  CHECK(samples == 1, "composite: gx.color has {} samples", samples);
  if (!legacy_matches(ctx, CopyColor, CopyLegacy)) {
    encode_mismatch(ctx, CopyColor, CopyLegacy, LegacyTex);
    return;
  }
  const wgpu::Texture color = ctx.texture(CopyColor);
  const wgpu::Texture legacy = ctx.texture(CopyLegacy);
  if (!color || !legacy) {
    return;
  }
  const wgpu::Extent3D size = ctx.extent(CopyLegacy);
  const wgpu::TexelCopyTextureInfo source{.texture = color};
  const wgpu::TexelCopyTextureInfo destination{.texture = legacy};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  ctx.encoder().CopyTextureToTexture(&source, &destination, &extent);
  bump(g_copies);
}

void declare(InternalPassBuilder& builder, const PassSetup& setup, const void*) {
  if (!declare_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(phase::Composite);
  const bool pbrDeclared = setup.settings->pbr && setup.gx_available(GxResource::Surface);
  // Without AO there is nothing to composite yet, so a frame without it culls this pass and the copy
  builder.use(LegacyTex, Access::Sampled, Missing::DisablePass);
  builder.use(resource::Fog, Access::Sampled,
              UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use(resource::GxNormal, Access::Sampled, Missing::DisablePass);
  // With PBR as a second producer a missing ao.final means no AO, not no composite
  builder.use(resource::AoFinal, Access::Sampled, pbrDeclared ? Missing::BindDummyWhite : Missing::DisablePass);
  // Always declared so tiers B and P share one layout; a black dummy reads slot 0, a white Uint one would not
  builder.use(resource::Surface, Access::Sampled,
              UseOptions{.missing = Missing::BindDummyBlack, .sampleType = wgpu::TextureSampleType::Uint});
  // Always declared, so every tier and PBR off share one layout; the black dummy has alpha 0 and keeps Sb
  builder.use(resource::SceneHdr, Access::Sampled, Missing::BindDummyBlack);
  builder.use(resource::GxColor, Access::ColorRMW);
}

bool enabled(const StageData& stage, const void*) {
  const LatchSlot slot = latched_slot();
  if (stage.frame == nullptr || slot.graph != stage.frame->graph.get() || !slot.live) {
    return false;
  }
  if (slot.classView) {
    bump(g_skippedView);
    return false;
  }
  const SceneHints* hints = g_cameraOverride != nullptr ? g_cameraOverride.get() : stage.sceneHints.get();
  if (hints == nullptr || !hints->has(SceneHasMainCamera)) {
    bump(g_skippedNoScene);
    return false;
  }
  // Without H every PBR pixel keeps Sb, so with AO off the copy and the composite would only discard
  if (!slot.aoLive && !(slot.pbrActive && lighting::deferred::frame_enabled(stage))) {
    bump(g_idleFrames);
    return false;
  }
  // With PBR declared a missing ao.final binds white, so later ao.final readers must accept the dummy
  if (shaders::pipelines_ready(slot.ids)) {
    g_enabledFrame.store(stage.frameId, std::memory_order_relaxed);
  }
  return true;
}

void request_pipelines(const PipelineSetup& setup, const void*, std::span<uint32_t, MaxPassPipelines> ids) {
  if (setup.pass == nullptr || setup.pass->env == nullptr) {
    return;
  }
  const wgpu::TextureFormat efb = setup.pass->env->gxFormats[static_cast<size_t>(GxResource::Color)];
  ids[0] = shaders::request_render(pipeline_desc(efb, setup.layout, setup.layoutHash)).value;
}

void encode(PassContext& ctx, const void*) {
  const uint32_t samples = ctx.sample_count(Color);
  CHECK(samples == 1, "composite: gx.color has {} samples", samples);
  if (!legacy_matches(ctx, Color, Legacy)) {
    encode_mismatch(ctx, Color, Legacy, LegacyTex);
    return;
  }
  for (const auto& [ref, name] : {std::pair{Fog, resource::Fog}, std::pair{Normal, resource::GxNormal}}) {
    if (!same_extent(ctx.extent(ref), ctx.extent(Color))) {
      encode_mismatch(ctx, Color, ref, name);
      return;
    }
  }
  // White and black dummies are 1x1 and every load clamps, so they never fail the extent check
  for (const auto& [ref, name] : {std::pair{AoFinal, resource::AoFinal}, std::pair{SurfaceRec, resource::Surface},
                                  std::pair{Hdr, resource::SceneHdr}}) {
    const wgpu::Extent3D extent = ctx.extent(ref);
    if (!same_extent(extent, ctx.extent(Color)) && (extent.width != 1 || extent.height != 1)) {
      encode_mismatch(ctx, Color, ref, name);
      return;
    }
  }
  const wgpu::RenderPipeline pipeline = shaders::render_pipeline(ctx.pipeline(0));
  if (!pipeline) {
    return;
  }
  const wgpu::RenderPassEncoder pass = ctx.begin_render();
  if (!pass) {
    return;
  }
  pass.SetPipeline(pipeline);
  pass.Draw(3);
  pass.End();
  bump(g_draws);
  if (params_carry_pbr(ctx.params())) {
    bump(g_pbrDraws);
    if (lighting::deferred::ran(ctx.stage_data())) {
      bump(g_hdrDraws);
    }
  }
}

bool m8_env_set() noexcept { return ao::env_ao_request().has_value() || identity_forced() || no_fog_forced(); }
} // namespace

bool wanted(const Settings& settings) noexcept {
  // AO and PBR are the composite producers; later producers widen this one predicate
  return settings.ao != Quality::Off || settings.pbr;
}

bool ao_wanted(const Settings& settings) noexcept { return settings.ao != Quality::Off; }

bool pbr_term_wanted(const Settings& settings) noexcept {
  return settings.pbr && settings.legacyBlend < 1.f && !identity_forced();
}

bool surface_bound(const CompiledGraph& graph) noexcept {
  const CompiledUse* use = composite_use(graph, resource::Surface);
  return use != nullptr && !use->staticDummy;
}

bool hdr_produced(const CompiledGraph& graph) noexcept {
  const CompiledUse* use = composite_use(graph, resource::SceneHdr);
  return use != nullptr && !use->staticDummy;
}

bool ao_produced(const CompiledGraph& graph) noexcept {
  const CompiledUse* use = composite_use(graph, resource::AoFinal);
  return use != nullptr && !use->staticDummy;
}

bool declare_wanted(const PassSetup& setup) noexcept {
  return setup.settings != nullptr && setup.env != nullptr && wanted(*setup.settings) &&
         setup.gx_available(GxResource::Fog) && setup.gx_available(GxResource::Normal) && setup.env->gxSampleCount == 1;
}

uint32_t params_flags(bool identity, const char* noFog) noexcept {
  uint32_t flags = identity ? FlagIdentity : 0u;
  if (noFog != nullptr && std::string_view{noFog} == "on") {
    flags |= FlagNoFog;
  }
  return flags;
}

std::optional<std::string> check_contract(const CompiledGraph& graph) {
  if (!node_live(graph, Pass)) {
    return fmt::format("{} is not compiled", Pass);
  }
  if (!node_live(graph, CopyPass)) {
    return fmt::format("{} is not compiled", CopyPass);
  }
  const bool aoProduced = ao_produced(graph);
  const ResourceId legacyId = graph.find_resource(LegacyTex);
  const ResourceId aoId = graph.find_resource(resource::AoFinal);
  if (legacyId == InvalidResource || (aoProduced && aoId == InvalidResource)) {
    return fmt::format("{} or {} is missing", LegacyTex, resource::AoFinal);
  }
  const CompiledResource& legacy = graph.resources[legacyId];
  if (aoProduced && graph.resources[aoId].tex.format != AoFormat) {
    return fmt::format("{} is {}, not {}", resource::AoFinal, magic_enum::enum_name(graph.resources[aoId].tex.format),
                       magic_enum::enum_name(AoFormat));
  }
  if (surface_bound(graph)) {
    const CompiledResource& surface = graph.resources[graph.find_resource(resource::Surface)];
    if (surface.tex.format != webgpu::SurfaceBufferFormat) {
      return fmt::format("{} is {}, not {}", resource::Surface, magic_enum::enum_name(surface.tex.format),
                         magic_enum::enum_name(webgpu::SurfaceBufferFormat));
    }
    if (!same_extent(surface.extent, legacy.extent)) {
      return fmt::format("{} is {}x{} but {} is {}x{}", resource::Surface, surface.extent.width, surface.extent.height,
                         LegacyTex, legacy.extent.width, legacy.extent.height);
    }
  }
  if (hdr_produced(graph)) {
    const CompiledResource& hdr = graph.resources[graph.find_resource(resource::SceneHdr)];
    if (hdr.tex.format != lighting::deferred::HdrFormat) {
      return fmt::format("{} is {}, not {}", resource::SceneHdr, magic_enum::enum_name(hdr.tex.format),
                         magic_enum::enum_name(lighting::deferred::HdrFormat));
    }
    if (!same_extent(hdr.extent, legacy.extent)) {
      return fmt::format("{} is {}x{} but {} is {}x{}", resource::SceneHdr, hdr.extent.width, hdr.extent.height,
                         LegacyTex, legacy.extent.width, legacy.extent.height);
    }
  }
  for (const std::string_view name : {resource::AoFinal, resource::Fog, resource::GxNormal}) {
    if (name == resource::AoFinal && !aoProduced) {
      continue;
    }
    const ResourceId id = graph.find_resource(name);
    const wgpu::Extent3D extent = id != InvalidResource ? graph.resources[id].extent : wgpu::Extent3D{};
    if (!same_extent(extent, legacy.extent)) {
      return fmt::format("{} is {}x{} but {} is {}x{}", name, extent.width, extent.height, LegacyTex,
                         legacy.extent.width, legacy.extent.height);
    }
  }
  const std::vector<NodeId>& program = graph.stages[static_cast<size_t>(Stage::OpaqueDone)].nodes;
  const auto first = std::ranges::find(program, graph.find_node(CopyPass));
  const auto last = std::ranges::find(program, graph.find_node(Pass));
  if (first == program.end() || last == program.end() || first > last) {
    return fmt::format("{} does not run before {}", CopyPass, Pass);
  }
  // legacyTex is the colour at phase.prepare, so a writer between copy and composite would be erased where AO lands
  for (auto it = std::next(first); it != last; ++it) {
    if (graph.nodes[*it].kind != NodeKind::Pass) {
      continue;
    }
    for (const CompiledUse& use : graph.node_uses(*it)) {
      const CompiledResource& target = graph.resources[use.resource];
      if (use.access == Access::ColorRMW && target.kind == ResourceKind::Gx && target.gx == GxResource::Color) {
        return fmt::format("{} writes gx.color between {} and {}", graph.nodes[*it].name, CopyPass, Pass);
      }
    }
  }
  return std::nullopt;
}

shaders::RenderPipelineDesc pipeline_desc(wgpu::TextureFormat efb, const wgpu::PipelineLayout& layout,
                                          uint64_t layoutKey) {
  shaders::RenderPipelineDesc desc{
      .label = "composite.main",
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .layout = layout,
      .layoutKey = layoutKey,
      .colorCount = 1,
  };
  // EFB alpha is the game's shadow stencil, so the composite writes RGB only
  desc.colors[0] = {.format = efb, .writeMask = RgbMask};
  return desc;
}

PassDesc copy_pass_desc() {
  return {
      .name = std::string{CopyPass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Copy,
      .cls = PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .declare = declare_copy,
      .encode = encode_copy,
  };
}

PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Render,
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
    g_copyPass = register_pass(copy_pass_desc());
    g_pass = register_pass(pass_desc());
  }
  fakeshadow::set_gate_source(fakeshadow::Gate::Ao, &ao_gate);
  scene::overlay::set_pbr_applied_source(&pbr_applied);
}

void shutdown() noexcept {
  fakeshadow::set_gate_source(fakeshadow::Gate::Ao, nullptr);
  scene::overlay::set_pbr_applied_source(nullptr);
  for (PassId* pass : {&g_copyPass, &g_pass}) {
    if (*pass != InvalidPassId) {
      unregister_pass(*pass);
      *pass = InvalidPassId;
    }
  }
  {
    std::lock_guard lock{g_slotMutex};
    g_slot = {};
  }
  g_contractWarned = false;
  g_enabledFrame.store(NoFrame, std::memory_order_relaxed);
  g_encodeMismatch.store(false, std::memory_order_relaxed);
  const testing::Counters c = testing::counters();
  constexpr std::string_view Line =
      "Composite drew on {} frames ({} skipped for mismatched inputs, {} without a scene "
      "camera, {} under a class view, {} graphs refused)";
  constexpr std::string_view PbrLine =
      "Composite PBR term ran on {} frames ({} with scene.hdr, {} idle without AO or H)";
  if (m8_env_set()) {
    Log.info(Line, c.draws, c.skippedMismatch, c.skippedNoScene, c.skippedView, c.contractFailures);
  } else {
    Log.debug(Line, c.draws, c.skippedMismatch, c.skippedNoScene, c.skippedView, c.contractFailures);
  }
  if (m8_env_set() || c.pbrDraws != 0 || c.idleFrames != 0) {
    Log.info(PbrLine, c.pbrDraws, c.hdrDraws, c.idleFrames);
  } else {
    Log.debug(PbrLine, c.pbrDraws, c.hdrDraws, c.idleFrames);
  }
}

void on_latch(const PassSetup& latched, const CompiledGraph* graph) {
  const Settings* settings = latched.settings;
  std::unique_lock lock{g_slotMutex};
  if (graph != g_slot.graph) {
    g_slot.graph = graph;
    g_slot.live = false;
    g_slot.hdrLive = graph != nullptr && hdr_produced(*graph);
    g_slot.aoProduced = graph != nullptr && ao_produced(*graph);
    g_slot.ids = {};
    g_encodeMismatch.store(false, std::memory_order_relaxed);
    if (graph != nullptr) {
      const std::optional<std::string> reason = check_contract(*graph);
      if (node_live(*graph, Pass)) {
        g_slot.ids[0] = graph->nodes[graph->find_node(Pass)].pipelineIds[0];
      }
      if (!reason) {
        g_slot.live = true;
      } else if (node_live(*graph, Pass)) {
        bump(g_contractFailures);
        if (!std::exchange(g_contractWarned, true)) {
          Log.warn("Composite off: {}", *reason);
        } else {
          Log.debug("Composite off: {}", *reason);
        }
      }
    }
  } else if (g_encodeMismatch.load(std::memory_order_relaxed)) {
    g_slot.live = false;
  }
  // The view the graph was compiled for, so the composite and TAA's declare never disagree
  g_slot.classView = debug::find_view(latched.debug_view()) != nullptr;
  g_slot.aoLive = settings != nullptr && ao_wanted(*settings) && g_slot.aoProduced;
  g_slot.pbrActive =
      settings != nullptr && testing::pbr_active_for(g_slot.live, g_slot.hdrLive, g_slot.classView, *settings);
  const bool pbrActive = g_slot.pbrActive;
  lock.unlock();
  if (g_pass != InvalidPassId) {
    Params params{.flags = g_flagsOverride.value_or(params_flags(identity_forced(), nullptr) |
                                                    (no_fog_forced() ? uint32_t{FlagNoFog} : 0u))};
    // legacyBlend reaches the shader only with the flag, so full legacy blend keeps PBR-off uniform bytes
    if (pbrActive) {
      params.flags |= FlagPbr;
      params.legacyBlend = settings->legacyBlend;
    }
    params = g_paramsOverride.value_or(params);
    set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
  }
}

bool ready(const Settings& settings) noexcept { return ready_for(latched_slot(), settings); }

bool ao_ready(const Settings& settings) noexcept {
  const LatchSlot slot = latched_slot();
  // The composite can be live for PBR alone, so AO also needs its own producer
  return ready_for(slot, settings) && (!ao_wanted(settings) || slot.aoProduced);
}

bool pbr_active() noexcept { return latched_slot().pbrActive; }

bool pbr_applied(const StageData& opaqueDone) noexcept {
  const CompiledGraph* graph = opaqueDone.frame != nullptr ? opaqueDone.frame->graph.get() : nullptr;
  if (graph == nullptr || opaqueDone.stage != Stage::OpaqueDone) {
    return false;
  }
  const NodeId node = graph->find_node(Pass);
  if (node == InvalidNode || graph->nodes[node].programIndex == InvalidNode ||
      !opaqueDone.is_enabled(graph->nodes[node].programIndex)) {
    return false;
  }
  if (!lighting::deferred::ran(opaqueDone)) {
    return false;
  }
  for (const StageBindings::Params& params : opaqueDone.bindings.params) {
    if (params.node == node) {
      return params_carry_pbr(std::span{params.bytes}.first(params.size));
    }
  }
  return false;
}

bool live() noexcept { return latched_slot().live; }

bool enabled_this_frame(uint64_t frameId) noexcept { return g_enabledFrame.load(std::memory_order_relaxed) == frameId; }

bool identity_forced() noexcept { return ao::identity_forced(); }

bool ao_gate(const FrameState& state) noexcept {
  const LatchSlot slot = latched_slot();
  // Baked AO is dropped only while a produced ao.final reaches the screen; PBR alone binds a white dummy and keeps it
  return state.settings.ao != Quality::Off && slot.live && slot.aoProduced && !slot.classView && !identity_forced() &&
         ao::effective_strength() > 0.f;
}

namespace testing {
Counters counters() noexcept {
  return {
      .copies = g_copies.load(std::memory_order_relaxed),
      .draws = g_draws.load(std::memory_order_relaxed),
      .skippedMismatch = g_skippedMismatch.load(std::memory_order_relaxed),
      .skippedNoScene = g_skippedNoScene.load(std::memory_order_relaxed),
      .skippedView = g_skippedView.load(std::memory_order_relaxed),
      .contractFailures = g_contractFailures.load(std::memory_order_relaxed),
      .pbrDraws = g_pbrDraws.load(std::memory_order_relaxed),
      .idleFrames = g_idleFrames.load(std::memory_order_relaxed),
      .hdrDraws = g_hdrDraws.load(std::memory_order_relaxed),
  };
}

void reset() noexcept {
  g_cameraOverride.reset();
  g_flagsOverride.reset();
  g_paramsOverride.reset();
  g_contractWarned = false;
  g_mismatchWarned.store(false, std::memory_order_relaxed);
  g_copies.store(0, std::memory_order_relaxed);
  g_draws.store(0, std::memory_order_relaxed);
  g_skippedMismatch.store(0, std::memory_order_relaxed);
  g_skippedNoScene.store(0, std::memory_order_relaxed);
  g_skippedView.store(0, std::memory_order_relaxed);
  g_contractFailures.store(0, std::memory_order_relaxed);
  g_pbrDraws.store(0, std::memory_order_relaxed);
  g_idleFrames.store(0, std::memory_order_relaxed);
  g_hdrDraws.store(0, std::memory_order_relaxed);
}

void set_flags(std::optional<uint32_t> flags) noexcept { g_flagsOverride = flags; }

void force_params(std::optional<Params> params) noexcept { g_paramsOverride = params; }

bool pbr_active_for(bool live, bool hdrLive, bool classView, const Settings& settings) noexcept {
  return live && hdrLive && pbr_term_wanted(settings) && !classView;
}

void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept { g_cameraOverride = std::move(hints); }

std::array<float, 3> legacy_ao(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7, float ao,
                               float normalAlpha) noexcept {
  if (std::round(std::clamp(normalAlpha, 0.f, 1.f) * 3.f) < 1.f || ao >= 1.f) {
    return legacy;
  }
  const float q = static_cast<float>(std::min(q7, 127u)) / 127.f;
  std::array<float, 3> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    const float s = legacy[i] - q * fogColor[i];
    if (s < -2.f / 255.f) {
      return legacy;
    }
    out[i] = legacy[i] - std::max(s, 0.f) * (1.f - ao);
  }
  return out;
}

std::array<float, 3> legacy_pbr_hdr(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7, float ao,
                                    float normalAlpha, std::array<float, 4> hdr, float legacyBlend) noexcept {
  if (hdr[3] < 0.5f || legacyBlend >= 1.f) {
    return legacy_ao(legacy, fogColor, q7, ao, normalAlpha);
  }
  const bool receiver = std::round(std::clamp(normalAlpha, 0.f, 1.f) * 3.f) >= 1.f;
  const float a = receiver && ao < 1.f ? ao : 1.f;
  const float q = static_cast<float>(std::min(q7, 127u)) / 127.f;
  std::array<float, 3> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    const float s = legacy[i] - q * fogColor[i];
    if (s < -2.f / 255.f) {
      return legacy;
    }
    const float sp = std::max(s, 0.f);
    const float linear = std::clamp(hdr[i], 0.f, 1.f);
    const float encoded = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.f / 2.4f) - 0.055f;
    const float t = 1.f - legacyBlend;
    out[i] = legacy[i] - sp + (sp * a * (1.f - t) + (1.f - q) * encoded * t);
  }
  return out;
}
} // namespace testing
} // namespace aurora::render::detail::composite

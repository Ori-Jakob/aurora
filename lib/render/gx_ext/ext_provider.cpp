#include "ext_provider.hpp"
#include "fwd_env.hpp"
#include "records.hpp"
#include "unlit.hpp"

#include "../frame_state.hpp"
#include "../material/classes.hpp"
#include "../material/registry.hpp"
#include "../post/composite.hpp"
#include "../settings.hpp"
#include "../../gfx/companion_cache.hpp"
#include "../../gfx/scene_targets.hpp"
#include "../../gx/pipeline.hpp"
#include "../../gx/render_ext.hpp"
#include "../../gx/texture.hpp"
#include "../../logging.hpp"

#include <aurora/render_caps.hpp>

#include <SDL3/SDL_stdinc.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace aurora::render::gx_ext {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};

enum class EnvMode : uint8_t { Unset, On, All };

std::shared_ptr<const material::Snapshot> g_snapshot;
bool g_force = false;
bool g_active = false;
bool g_testForce = false;
bool g_envRead = false;
EnvMode g_env = EnvMode::Unset;
bool g_companionsOff = false;
bool g_delight = false;
std::optional<bool> g_companionSupport;
std::atomic_uint64_t g_layerDraws{0};

using gfx::companion_cache::role_bit;
using gfx::texture_replacement::CompanionRole;

constexpr uint32_t CompanionSampledTextures = 16;
constexpr uint8_t LayerRoles = role_bit(CompanionRole::Albedo) | role_bit(CompanionRole::Nrm) |
                               role_bit(CompanionRole::Orm) | role_bit(CompanionRole::Mat) |
                               role_bit(CompanionRole::Emit);

EnvMode read_env() noexcept {
  const char* value = SDL_getenv("AURORA_RENDER_STOCHASTIC");
  if (value == nullptr) {
    return EnvMode::Unset;
  }
  const std::string_view mode{value};
  if (mode == "all") {
    Log.warn("AURORA_RENDER_STOCHASTIC=all: hex tiling forced on every eligible repeating texture");
    return EnvMode::All;
  }
  if (mode == "on") {
    Log.warn("AURORA_RENDER_STOCHASTIC=on: the stochastic tiling settings gate is forced open");
    return EnvMode::On;
  }
  return EnvMode::Unset;
}

gx::render_ext::StochasticParams record_params(const material::MaterialRecord& record) noexcept {
  gx::render_ext::StochasticParams params;
  if ((record.stochasticSet & material::StochasticScale) != 0) {
    params.scale = record.scale;
  }
  if ((record.stochasticSet & material::StochasticRotation) != 0) {
    params.rotation = record.rotation;
  }
  if ((record.stochasticSet & material::StochasticContrast) != 0) {
    params.contrast = record.contrast;
  }
  if ((record.stochasticSet & material::StochasticExponent) != 0) {
    params.exponent = record.exponent;
  }
  return params;
}

u8 stochastic_mask(const gx::ShaderInfo& info, gx::render_ext::StochasticLanes& lanes) noexcept {
  const material::Snapshot* snapshot = g_snapshot.get();
  u8 mask = 0;
  for (u32 i = 0; i < gx::MaxTextures; ++i) {
    if (!info.stochasticEligible.test(i)) {
      continue;
    }
    gx::render_ext::StochasticParams params;
    if (!g_force) {
      const gfx::TextureMaterialRef& material = gx::g_gxState.textures[i].material;
      if (snapshot == nullptr || material.id == 0 || (material.flags & material::Stochastic) == 0 ||
          material.id >= snapshot->records.size()) {
        continue;
      }
      // W3's lookup reads the live registry and this the latched one; requiring both skips the publish-to-latch frame
      const material::MaterialRecord& record = snapshot->record(material.id);
      if ((record.summary & material::Stochastic) == 0 || (record.summary & material::Cutout) != 0) {
        continue;
      }
      params = record_params(record);
    }
    lanes[i] = gx::render_ext::pack(params);
    mask |= static_cast<u8>(1u << i);
  }
  return mask;
}

bool companions_supported() noexcept {
  if (g_companionSupport) {
    return *g_companionSupport;
  }
  const Caps& caps = render::caps();
  return caps.core && caps.bcTextures && caps.limits.maxSampledTexturesPerShaderStage >= CompanionSampledTextures;
}

uint8_t layer0_for(const gx::PipelineConfig& config, const gx::ShaderInfo& info) noexcept {
  if (g_companionsOff || !companions_supported()) {
    return 0;
  }
  for (u32 i = 0; i < gx::MaxTextures; ++i) {
    if ((info.sampledTextures.test(i) || info.sampledIndTextures.test(i)) &&
        (gx::g_gxState.textures[i].material.flags & material::FakeShadow) != 0) {
      return 0;
    }
  }
  const gx::ShaderConfig& shader = config.shaderConfig;
  const uint16_t stages = gx::unlit_facts(shader, info).layerStages;
  const uint8_t roles = LayerRoles | (g_delight ? role_bit(CompanionRole::Delight) : uint8_t{0});
  for (u32 s = 0; s < shader.tevStageCount && s < gx::MaxTevStages; ++s) {
    const u32 map = shader.tevStages[s].texMapId;
    if ((stages >> s & 1u) == 0 || map > GX_TEXMAP7) {
      continue;
    }
    const gfx::TextureBind& bind = gx::g_gxState.textures[map];
    if ((bind.material.companionRoles & roles) == 0 || (bind.ref != nullptr && bind.ref->attachmentTextureView)) {
      continue;
    }
    return static_cast<uint8_t>(map + 1);
  }
  return 0;
}

// Indexed, never resident: the key must not depend on when the map finishes loading
bool emit_indexed(uint8_t layer0) noexcept {
  return layer0 != 0 &&
         (gx::g_gxState.textures[layer0 - 1u].material.companionRoles & role_bit(CompanionRole::Emit)) != 0;
}

gx::RenderExtConfig derive(void*, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                           const gfx::RenderTargetLayout& layout, gx::render_ext::StochasticLanes& lanes) noexcept {
  gx::RenderExtConfig ext{};
  if (g_active) {
    ext.stochasticMask = stochastic_mask(info, lanes);
  }
  if (gfx::layout_has_gbuffer(layout)) {
    ext.output = output_for(config, (gx::render_ext::g_drawInputs & records::InWindow) != 0);
    // Follows the layout tier, never a setting, so AO toggles never change a pipeline
    ext.fwdEnv = fwd_env_for(config, (gx::render_ext::g_drawInputs & records::ForwardWindow) != 0);
    if (ext.output == 1 && gfx::layout_has_surface(layout) &&
        unlit::output2(config, info, gx::render_ext::g_drawInputs)) {
      ext.output = 2;
      ext.layer0 = layer0_for(config, info);
      ext.surfaceExt =
          unlit::surface_ext(config, info, gx::render_ext::g_drawInputs) || emit_indexed(ext.layer0) ? 1 : 0;
      if (ext.layer0 != 0) {
        g_layerDraws.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  return ext;
}

gx::render_ext::CompanionLane companion_params(void*, const gfx::TextureMaterialRef& material,
                                               const gx::render_ext::CompanionLane& lane) noexcept {
  gx::render_ext::CompanionLane out = lane;
  const u32 bits = static_cast<u32>(lane.v[0]);
  if (!gx::render_ext::g_companionGates.normals || (bits & (gx::render_ext::LaneNrm | gx::render_ext::LaneMat)) == 0) {
    out.v[1] = 0.f;
    return out;
  }
  const material::Snapshot* snapshot = g_snapshot.get();
  if (snapshot == nullptr || material.id == 0) {
    return out;
  }
  const material::MaterialScalars& scalars = snapshot->scalars_of(material.id);
  if ((scalars.set & material::ScalarNormalStrength) != 0) {
    out.v[1] = scalars.normalStrength;
  }
  if ((scalars.set & material::ScalarNormalFlipY) != 0 && scalars.normalFlipY) {
    out.v[2] = -out.v[2];
  }
  return out;
}

const gx::render_ext::Hooks kHooks{.derive = &derive, .companion_params = &companion_params};

void set_active(bool active, bool force) noexcept {
  g_active = active;
  g_force = force;
  gx::texture::invalidate_bindings();
}
} // namespace

uint8_t output_for(const gx::PipelineConfig& c, bool inWindow) noexcept {
  return inWindow && ((c.depthCompare && c.depthUpdate) || c.colorUpdate) ? 1 : 0;
}

uint8_t fwd_env_for(const gx::PipelineConfig& c, bool forwardWindow) noexcept {
  return forwardWindow && c.colorUpdate ? 1 : 0;
}

namespace {
void apply_companions(const detail::FrameState& state, bool generationChanged) noexcept {
  const gfx::LayoutWarm* warm = gfx::layout_warm();
  const bool surface = gfx::layout_has_surface(gfx::scene_render_target_layout()) ||
                       (warm != nullptr && gfx::layout_has_surface(warm->pending));
  const Settings& s = state.settings;
  const bool consumer = s.pbr && surface;
  gx::texture::set_companion_consumer(consumer);
  gx::render_ext::CompanionGates gates{};
  if (consumer) {
    gates = {
        .materials = s.textureMaterials,
        .matMaps = (s.textureMaterials && s.legacyMatMaps) || detail::mat_maps_env(),
        .normals = s.legacyBlend < 1.f && !detail::composite::identity_forced(),
        .delight = s.textureMaterials && g_delight,
    };
  }
  // A scalar edit keeps every material id, so only a rebind rebuilds the lanes that carry it
  if (gx::render_ext::set_companion_gates(gates) || (consumer && generationChanged)) {
    gx::texture::invalidate_bindings();
  }
  records::set_back_face_flip(gates.normals && unlit::classes_enabled());
}
} // namespace

void apply_frame_state(const detail::FrameState& state) noexcept {
  records::apply_frame_state(state);
  fwd_env::apply_frame_state(state);
  unlit::apply_frame_state(state);
  if (!g_envRead) {
    g_envRead = true;
    g_env = read_env();
    if (const char* value = SDL_getenv("AURORA_RENDER_COMPANIONS");
        value != nullptr && std::string_view{value} == "off") {
      Log.warn("AURORA_RENDER_COMPANIONS=off: no draw samples companion material maps");
      g_companionsOff = true;
    }
    if (const char* value = SDL_getenv("AURORA_RENDER_DELIGHT"); value != nullptr && std::string_view{value} == "on") {
      Log.warn(
          "AURORA_RENDER_DELIGHT=on: delight maps scale Oklab L; the colour space is provisional until the "
          "P-track's first S3 sample");
      g_delight = true;
    }
  }
  std::shared_ptr<const material::Snapshot> latched = material::latched_snapshot();
  const bool generationChanged = g_snapshot == nullptr || latched->generation != g_snapshot->generation;
  if (generationChanged) {
    g_snapshot = std::move(latched);
  }
  material::classes::apply_frame_state(*g_snapshot);
  const bool force = g_testForce || g_env == EnvMode::All;
  const bool gateOpen = g_env == EnvMode::On || (state.settings.stochasticTiling && state.settings.textureMaterials);
  const bool want = force || (gateOpen && g_snapshot->summaryCount[0] != 0);
  const bool activeChanged = want != g_active || force != g_force || (want && generationChanged);
  if (activeChanged) {
    set_active(want, force);
  }
  apply_companions(state, generationChanged && !activeChanged);
  gx::render_ext::set_hooks(g_active || records::gbuffer() || gfx::gbuffer_live_or_pending() ? &kHooks : nullptr);
}

void shutdown() noexcept {
  if (const uint64_t layered = g_layerDraws.exchange(0, std::memory_order_relaxed); layered != 0) {
    Log.info("Companion layer: {} output-2 derivations with layer 0, companion skips {}", layered,
             gx::bind_counts().companionSkips);
  }
  unlit::shutdown();
  records::shutdown();
  material::classes::shutdown();
  gx::render_ext::set_hooks(nullptr);
  gx::render_ext::set_companion_gates({});
  g_snapshot.reset();
  g_active = false;
  g_force = false;
}

namespace testing {
void set_force(bool force) noexcept { g_testForce = force; }

bool active() noexcept { return g_active; }

void reset() noexcept {
  shutdown();
  records::testing::reset();
  fwd_env::testing::reset();
  unlit::testing::reset();
  g_testForce = false;
  g_envRead = false;
  g_env = EnvMode::Unset;
  g_companionsOff = false;
  g_delight = false;
  g_companionSupport.reset();
  g_layerDraws.store(0, std::memory_order_relaxed);
  gx::texture::set_companion_consumer(false);
}

void set_companion_support(std::optional<bool> supported) noexcept { g_companionSupport = supported; }
} // namespace testing
} // namespace aurora::render::gx_ext

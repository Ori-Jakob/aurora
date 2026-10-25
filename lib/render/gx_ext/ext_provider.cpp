#include "ext_provider.hpp"
#include "records.hpp"

#include "../frame_state.hpp"
#include "../material/registry.hpp"
#include "../../gx/render_ext.hpp"
#include "../../gx/texture.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <memory>
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

gx::RenderExtConfig derive(void*, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                           const gfx::RenderTargetLayout& layout, gx::render_ext::StochasticLanes& lanes) noexcept {
  gx::RenderExtConfig ext{};
  if (g_active) {
    ext.stochasticMask = stochastic_mask(info, lanes);
  }
  if (gfx::layout_has_gbuffer(layout)) {
    ext.output = output_for(config, (gx::render_ext::g_drawInputs & records::InWindow) != 0);
  }
  return ext;
}

const gx::render_ext::Hooks kHooks{.derive = &derive};

void set_active(bool active, bool force) noexcept {
  g_active = active;
  g_force = force;
  gx::texture::invalidate_bindings();
}
} // namespace

uint8_t output_for(const gx::PipelineConfig& c, bool inWindow) noexcept {
  return inWindow && ((c.depthCompare && c.depthUpdate) || c.colorUpdate) ? 1 : 0;
}

void apply_frame_state(const detail::FrameState& state) noexcept {
  records::apply_frame_state(state);
  if (!g_envRead) {
    g_envRead = true;
    g_env = read_env();
  }
  std::shared_ptr<const material::Snapshot> latched = material::latched_snapshot();
  const bool generationChanged = g_snapshot == nullptr || latched->generation != g_snapshot->generation;
  if (generationChanged) {
    g_snapshot = std::move(latched);
  }
  const bool force = g_testForce || g_env == EnvMode::All;
  const bool gateOpen = g_env == EnvMode::On || (state.settings.stochasticTiling && state.settings.textureMaterials);
  const bool want = force || (gateOpen && g_snapshot->summaryCount[0] != 0);
  if (want != g_active || force != g_force || (want && generationChanged)) {
    set_active(want, force);
  }
  gx::render_ext::set_hooks(g_active || records::gbuffer() ? &kHooks : nullptr);
}

void shutdown() noexcept {
  records::shutdown();
  gx::render_ext::set_hooks(nullptr);
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
  g_testForce = false;
  g_envRead = false;
  g_env = EnvMode::Unset;
}
} // namespace testing
} // namespace aurora::render::gx_ext

#include "unlit.hpp"
#include "records.hpp"

#include "../material/registry.hpp"
#include "../scene/classifier.hpp"
#include "../../gfx/hash.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../gx/render_ext.hpp"
#include "../../gx/unlit_plan.hpp"
#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <absl/container/flat_hash_set.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <string_view>
#include <utility>

namespace aurora::render::gx_ext::unlit {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};
constexpr size_t VerdictCount = 7;
constexpr size_t MaxSeenConfigs = 65536;

enum Demotion : uint8_t {
  DemoteVerdict,
  DemoteVaryings,
  DemoteNoPbr,
  DemoteEfbCopy,
  DemoteNoNrm,
  DemoteLines,
  DemoteBlended,
  DemotionCount
};

// Game thread, while the FIFO is idle
bool g_envRead = false;
bool g_disabled = false;
bool g_policyOff = false;
bool g_classesOff = false;
int g_vertexOverride = -1;

// FIFO thread
absl::flat_hash_set<uint64_t> g_seen;
std::array<std::atomic_uint64_t, VerdictCount> g_verdicts{};
std::array<std::atomic_uint64_t, DemotionCount> g_demotions{};

void bump(std::atomic_uint64_t& counter) noexcept { counter.fetch_add(1, std::memory_order_relaxed); }

void read_env() noexcept {
  if (const char* value = SDL_getenv("AURORA_RENDER_UNLIT"); value != nullptr && std::string_view{value} == "off") {
    Log.warn("AURORA_RENDER_UNLIT=off: no draw takes the unlit surface pass");
    g_disabled = true;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_UNLIT_POLICY");
      value != nullptr && std::string_view{value} == "off") {
    Log.warn("AURORA_RENDER_UNLIT_POLICY=off: TEV register policies are ignored by the unlit pass");
    g_policyOff = true;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_MATERIAL_CLASSES"); value != nullptr) {
    const std::string_view mode{value};
    if (mode == "off") {
      Log.warn("AURORA_RENDER_MATERIAL_CLASSES=off: every PBR draw takes the default class with no ambient or flip");
      g_classesOff = true;
    } else if (mode.starts_with("force:")) {
      Log.warn("AURORA_RENDER_MATERIAL_CLASSES={}: every PBR draw takes class {}", mode, mode.substr(6));
      material::classes::set_force(mode.substr(6));
    }
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_VERTEX_COLOR"); value != nullptr) {
    const std::string_view mode{value};
    g_vertexOverride = mode == "tint" ? 0 : mode == "baked" ? 1 : mode == "ignore" ? 2 : -1;
    if (g_vertexOverride >= 0) {
      Log.warn("AURORA_RENDER_VERTEX_COLOR={}: every unlit draw takes this vertex-colour policy", mode);
    }
  }
}

bool binds_ok(const gx::ShaderInfo& info) noexcept {
  for (u32 i = 0; i < gx::MaxTextures; ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    const gfx::TextureBind& bind = gx::g_gxState.textures[i];
    if (bind.ref != nullptr && bind.ref->attachmentTextureView) {
      bump(g_demotions[DemoteEfbCopy]);
      return false;
    }
    if ((bind.material.flags & material::NoPbr) != 0) {
      bump(g_demotions[DemoteNoPbr]);
      return false;
    }
  }
  return true;
}

void note_config(const gx::ShaderConfig& config, const gx::UnlitFacts& facts) noexcept {
  if (g_seen.size() >= MaxSeenConfigs) {
    return;
  }
  const uint64_t key = XXH3_64bits(&config, offsetof(gx::ShaderConfig, ext));
  if (!g_seen.insert(key).second) {
    return;
  }
  bump(g_verdicts[static_cast<size_t>(facts.verdict) % VerdictCount]);
  const gx::unlit::Plan plan = gx::unlit::plan(config);
  if (!plan.pass()) {
    Log.debug("Unlit TEV: config {:016x} {} at rule R{} stage {}", key, gx::unlit::verdict_name(plan.verdict),
              plan.failRule, plan.failStage);
  }
}

struct LayerWalk {
  uint32_t vertexPolicy = 0;
  bool unlit = false;
  std::array<uint16_t, material::classes::FieldCount> ids{};
};

uint32_t layer_maps(const gx::ShaderConfig& config, const gx::ShaderInfo& info) noexcept;

LayerWalk walk_layers(const gx::PipelineConfig& config, const gx::ShaderInfo& info, bool bakedObject) noexcept {
  const gx::ShaderConfig& shader = config.shaderConfig;
  const uint32_t maps = layer_maps(shader, info);
  LayerWalk walk;
  bool policySet = false;
  for (u32 s = 0; s < shader.tevStageCount && s < gx::MaxTevStages; ++s) {
    const u32 map = shader.tevStages[s].texMapId;
    if (map > GX_TEXMAP7 || (maps >> map & 1u) == 0) {
      continue;
    }
    const gfx::TextureMaterialRef& ref = gx::g_gxState.textures[map].material;
    if (const uint32_t vc = material::vertex_color(ref.flags); vc != 0 && !policySet) {
      walk.vertexPolicy = vc - 1;
      policySet = true;
    }
    walk.unlit = walk.unlit || (ref.flags & material::Unlit) != 0;
    if (ref.id == 0) {
      continue;
    }
    const uint8_t fields = material::classes::supplied(ref.id);
    for (size_t f = 0; f < walk.ids.size(); ++f) {
      if (walk.ids[f] == 0 && (fields >> f & 1u) != 0) {
        walk.ids[f] = static_cast<uint16_t>(ref.id);
      }
    }
  }
  if (!policySet) {
    walk.vertexPolicy = bakedObject ? 1 : 0;
  }
  if (g_vertexOverride >= 0) {
    walk.vertexPolicy = static_cast<uint32_t>(g_vertexOverride);
  }
  return walk;
}

uint32_t layer_maps(const gx::ShaderConfig& config, const gx::ShaderInfo& info) noexcept {
  const uint16_t stages = gx::unlit_facts(config, info).layerStages;
  uint32_t maps = 0;
  for (u32 s = 0; s < config.tevStageCount && s < gx::MaxTevStages; ++s) {
    if ((stages >> s & 1u) != 0 && config.tevStages[s].texMapId <= GX_TEXMAP7) {
      maps |= 1u << config.tevStages[s].texMapId;
    }
  }
  return maps;
}
} // namespace

bool output2(const gx::PipelineConfig& config, const gx::ShaderInfo& info, uint32_t inputs) noexcept {
  if (g_disabled || (inputs & records::PbrEligible) == 0 || !(config.depthCompare && config.depthUpdate) ||
      !config.colorUpdate) {
    return false;
  }
  if (!gx::effectively_opaque(config)) {
    bump(g_demotions[DemoteBlended]);
    return false;
  }
  if (!binds_ok(info)) {
    return false;
  }
  const gx::ShaderConfig& shader = config.shaderConfig;
  const gx::UnlitFacts facts = gx::unlit_facts(shader, info);
  note_config(shader, facts);
  if (!facts.admitted) {
    const auto verdict = static_cast<gx::unlit::Verdict>(facts.verdict);
    const bool pass = verdict == gx::unlit::Verdict::Passthrough || verdict == gx::unlit::Verdict::Linear ||
                      verdict == gx::unlit::Verdict::Scale;
    bump(g_demotions[!pass                                         ? DemoteVerdict
                     : shader.attrs[GX_VA_NRM].attrType == GX_NONE ? DemoteNoNrm
                     : shader.lineMode != 0                        ? DemoteLines
                                                                   : DemoteVaryings]);
    return false;
  }
  return true;
}

DrawMaterial draw_material(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                           const scene::DrawClass& cls) noexcept {
  const LayerWalk walk = walk_layers(config, info, (cls.key.flags & scene::object_flag::BakedVertexLight) != 0);
  DrawMaterial out{.vertexPolicy = walk.vertexPolicy,
                   .unlit = walk.unlit || (cls.key.categories & scene::category::Emissive) != 0};
  if (g_classesOff) {
    out.word = material::classes::default_word(false);
    return out;
  }
  out.word = material::classes::material_word(walk.ids.data(), cls.key.materialClass, cls.key.categories,
                                              config.cullMode, (cls.key.flags & scene::object_flag::DoubleSided) != 0,
                                              static_cast<bool>(config.shaderConfig.alphaCompare), out.source);
  return out;
}

bool surface_ext(const gx::PipelineConfig& config, const gx::ShaderInfo& info, uint32_t inputs) noexcept {
  if (g_classesOff) {
    return false;
  }
  const gx::unlit::Ambient ambient = gx::unlit::ambient_of(config.shaderConfig, info);
  if (ambient.source == gx::unlit::AmbientSource::Vertex) {
    return true;
  }
  return ambient.source == gx::unlit::AmbientSource::Register && ambient.rawVertex &&
         walk_layers(config, info, (inputs & records::BakedVertex) != 0).vertexPolicy == 1;
}

uint32_t vertex_policy(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                       const scene::DrawClass& cls) noexcept {
  return walk_layers(config, info, (cls.key.flags & scene::object_flag::BakedVertexLight) != 0).vertexPolicy;
}

bool unlit_flag(const gx::PipelineConfig& config, const gx::ShaderInfo& info, const scene::DrawClass& cls) noexcept {
  return (cls.key.categories & scene::category::Emissive) != 0 || walk_layers(config, info, false).unlit;
}

bool policy_enabled() noexcept { return !g_policyOff; }

bool classes_enabled() noexcept { return !g_classesOff; }

void apply_frame_state(const detail::FrameState&) noexcept {
  if (!std::exchange(g_envRead, true)) {
    read_env();
  }
}

void shutdown() noexcept {
  uint64_t configs = 0;
  uint64_t passed = 0;
  for (size_t i = 0; i < VerdictCount; ++i) {
    configs += g_verdicts[i].load(std::memory_order_relaxed);
    passed += i <= static_cast<size_t>(gx::unlit::Verdict::Scale) ? g_verdicts[i].load(std::memory_order_relaxed) : 0;
  }
  if (configs != 0) {
    const auto v = [](gx::unlit::Verdict verdict) {
      return g_verdicts[static_cast<size_t>(verdict)].load(std::memory_order_relaxed);
    };
    const auto d = [](Demotion demotion) { return g_demotions[demotion].load(std::memory_order_relaxed); };
    const auto& stats = gx::render_ext::g_stats;
    const records::Stats recordStats = records::stats();
    using gx::unlit::Verdict;
    Log.info(
        "Unlit TEV: output-2 draws {}, configs {} pass / {} fail (passthrough {}, linear {}, scale {}, spec-only "
        "{}, other {}, nonlinear {}, light texgen {}), demoted (verdict {}, varyings {}, NoPbr {}, EFB copy {}, "
        "no NRM {}, lines {}, blended {}), fallback output 1 {} / output 0 {}, twin adopted {}, policy blocks {}, "
        "fwdEnv demotions {}, in-window blended main draws {}, classes manifest {} / hint {} / default {} / none {}, "
        "two-sided {}, cull-front {}, mirrored {}, palette two-sided {}, ambient unknown {} configs",
        recordStats.pbrRecords, passed, configs - passed, v(Verdict::Passthrough), v(Verdict::Linear),
        v(Verdict::Scale), v(Verdict::FailSpecOnly), v(Verdict::FailOther), v(Verdict::FailNonlinear),
        v(Verdict::FailLightTexgen), d(DemoteVerdict), d(DemoteVaryings), d(DemoteNoPbr), d(DemoteEfbCopy),
        d(DemoteNoNrm), d(DemoteLines), d(DemoteBlended), stats.pbrFallbackOutput1.load(std::memory_order_relaxed),
        stats.pbrFallbackOutput0.load(std::memory_order_relaxed), stats.pbrTwinAdopted.load(std::memory_order_relaxed),
        recordStats.policyBlocks, stats.fwdEnvDemotions.load(std::memory_order_relaxed), recordStats.blendedInWindow,
        recordStats.classManifest, recordStats.classHint, recordStats.classDefault, recordStats.classNone,
        recordStats.twoSided, recordStats.cullFront, recordStats.mirrored, recordStats.paletteTwoSided,
        stats.pbrAmbientUnknown.load(std::memory_order_relaxed));
  }
  testing::reset();
}

namespace testing {
void reset() noexcept {
  g_seen.clear();
  for (auto& counter : g_verdicts) {
    counter.store(0, std::memory_order_relaxed);
  }
  for (auto& counter : g_demotions) {
    counter.store(0, std::memory_order_relaxed);
  }
  g_envRead = false;
  g_disabled = false;
  g_policyOff = false;
  g_classesOff = false;
  g_vertexOverride = -1;
}
} // namespace testing
} // namespace aurora::render::gx_ext::unlit

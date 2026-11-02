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

uint32_t vertex_policy(const gx::PipelineConfig& config, const gx::ShaderInfo& info,
                       const scene::DrawClass& cls) noexcept {
  if (g_vertexOverride >= 0) {
    return static_cast<uint32_t>(g_vertexOverride);
  }
  const uint32_t maps = layer_maps(config.shaderConfig, info);
  for (u32 s = 0; s < config.shaderConfig.tevStageCount && s < gx::MaxTevStages; ++s) {
    const u32 map = config.shaderConfig.tevStages[s].texMapId;
    if (map > GX_TEXMAP7 || (maps >> map & 1u) == 0) {
      continue;
    }
    if (const uint32_t vc = material::vertex_color(gx::g_gxState.textures[map].material.flags); vc != 0) {
      return vc - 1;
    }
  }
  return (cls.key.flags & scene::object_flag::BakedVertexLight) != 0 ? 1 : 0;
}

bool unlit_flag(const gx::PipelineConfig& config, const gx::ShaderInfo& info, const scene::DrawClass& cls) noexcept {
  if ((cls.key.categories & scene::category::Emissive) != 0) {
    return true;
  }
  const uint32_t maps = layer_maps(config.shaderConfig, info);
  for (u32 map = 0; map < gx::MaxTextures; ++map) {
    if ((maps >> map & 1u) != 0 && (gx::g_gxState.textures[map].material.flags & material::Unlit) != 0) {
      return true;
    }
  }
  return false;
}

bool policy_enabled() noexcept { return !g_policyOff; }

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
        "fwdEnv demotions {}, in-window blended main draws {}",
        recordStats.pbrRecords, passed, configs - passed, v(Verdict::Passthrough), v(Verdict::Linear),
        v(Verdict::Scale), v(Verdict::FailSpecOnly), v(Verdict::FailOther), v(Verdict::FailNonlinear),
        v(Verdict::FailLightTexgen), d(DemoteVerdict), d(DemoteVaryings), d(DemoteNoPbr), d(DemoteEfbCopy),
        d(DemoteNoNrm), d(DemoteLines), d(DemoteBlended), stats.pbrFallbackOutput1.load(std::memory_order_relaxed),
        stats.pbrFallbackOutput0.load(std::memory_order_relaxed), stats.pbrTwinAdopted.load(std::memory_order_relaxed),
        recordStats.policyBlocks, stats.fwdEnvDemotions.load(std::memory_order_relaxed), recordStats.blendedInWindow);
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
  g_vertexOverride = -1;
}
} // namespace testing
} // namespace aurora::render::gx_ext::unlit

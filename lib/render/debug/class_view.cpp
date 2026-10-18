#include "views.hpp"

#include "../frame_state.hpp"
#include "../scene/classify.hpp"
#include "../../gfx/pipeline_cache.hpp"
#include "../../gfx/recording.hpp"
#include "../../gfx/resources.hpp"
#include "../../gx/gx.hpp"
#include "../../gx/pipeline.hpp"
#include "../../gx/shader_info.hpp"
#include "../../logging.hpp"

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <utility>

namespace aurora::render::debug {
namespace {
constexpr Module Log{"aurora::render::debug"};
constexpr uint32_t MaxViewUniformBytes = 8u << 20;

struct Variant {
  gfx::PipelineRef pipeline = 0;
  gx::ShaderInfo info;
  uint8_t konst = 3;
  bool refused = false;
  bool ready = false;
};

// A draw that repeats the previous one's uniform, variant and colour reuses its patched uniform
struct UniformReuse {
  gfx::Range source;
  gfx::PipelineRef pipeline = 0;
  Rgb8 color;
  gfx::Range range;
  bool valid = false;
};

using VariantLookup = gfx::PipelineRef (*)(const gx::PipelineConfig&, const gfx::RenderTargetLayout&);

// FIFO thread; begin_view_frame changes them while the FIFO is idle
ClassView g_view = ClassView::Off;
uint32_t g_activeBit = 0;
absl::flat_hash_map<gfx::PipelineRef, Variant> g_variants;
UniformReuse g_reuse;
uint32_t g_budget = MaxViewUniformBytes;
uint32_t g_lastViewBytes = 0;
bool g_warnedBudget = false;
bool g_warnedSpace = false;
VariantLookup g_lookup = &gfx::find_debug_pipeline;
bool g_assumeReady = false;
uint32_t g_budgetOverride = 0;

constexpr bool packed_alpha_compare(GXTevOp op) noexcept {
  return op >= GX_TEV_COMP_R8_GT && op <= GX_TEV_COMP_BGR24_EQ;
}

constexpr bool k3_color_channel(GXTevKAlphaSel sel) noexcept {
  return sel == GX_TEV_KASEL_K3_R || sel == GX_TEV_KASEL_K3_G || sel == GX_TEV_KASEL_K3_B;
}

// Under the K3 fallback a packed compare may read K3 through any register an earlier stage wrote
bool k3_fallback_changes_alpha(const gx::TevStage& stage) noexcept {
  return k3_color_channel(stage.kaSel) || packed_alpha_compare(stage.alphaOp.op);
}

uint32_t view_budget() noexcept {
  if (g_budgetOverride != 0) {
    return g_budgetOverride;
  }
  // The rest of the frame needs what it used last time, and the view takes at most half of what is left
  const uint32_t lastUniformSize = gfx::last_recorded_uniform_size();
  const uint64_t legacy = lastUniformSize > g_lastViewBytes ? lastUniformSize - g_lastViewBytes : 0;
  const uint64_t free = gfx::UniformBufferSize > legacy ? gfx::UniformBufferSize - legacy : 0;
  return static_cast<uint32_t>(std::min<uint64_t>(MaxViewUniformBytes, free / 2));
}
} // namespace

bool make_view_variant(const gx::PipelineConfig& config, const gx::ShaderInfo& info, gx::PipelineConfig& out) noexcept {
  const auto& shader = config.shaderConfig;
  if (shader.tevStageCount == 0) {
    return false;
  }
  const gx::TevStage& last = shader.tevStages[shader.tevStageCount - 1];
  // A packed alpha compare reads this stage's colour arguments, which the variant replaces
  if (packed_alpha_compare(last.alphaOp.op)) {
    return false;
  }
  uint8_t konst = 3;
  while (konst > 0 && info.sampledKColors.test(konst)) {
    --konst;
  }
  const bool k3Fallback = info.sampledKColors.all();
  if (k3Fallback) {
    konst = 3;
    for (uint32_t i = 0; i < shader.tevStageCount; ++i) {
      if (k3_fallback_changes_alpha(shader.tevStages[i])) {
        return false;
      }
    }
  }
  out = config;
  gx::TevStage& stage = out.shaderConfig.tevStages[shader.tevStageCount - 1];
  stage.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_KONST};
  stage.colorOp.op = GX_TEV_ADD;
  stage.colorOp.bias = GX_TB_ZERO;
  stage.colorOp.scale = GX_CS_SCALE_1;
  stage.colorOp.clamp = true;
  stage.kcSel = static_cast<GXTevKColorSel>(GX_TEV_KCSEL_K0 + konst);
  out.shaderConfig.fogType = GX_FOG_NONE;
  out.shaderConfig.fogRangeEnabled = 0;
  return true;
}

void begin_view_frame(const detail::FrameState& state) noexcept {
  // begin_hint_frame already zeroed this frame's counts, so last frame's view bytes come from its sealed report
  const std::shared_ptr<const scene::FrameReport> report = scene::latest_report();
  g_lastViewBytes = report != nullptr ? report->stats.view.uniformBytes : 0;
  const uint16_t id = view_override() != 0 ? view_override() : state.settings.debugView;
  const DebugView* view = find_view(id);
  const ClassView next = view != nullptr ? static_cast<ClassView>(view->id) : ClassView::Off;
  if (next == ClassView::Off) {
    g_variants.clear();
  }
  g_view = next;
  g_activeBit = view_bit();
  g_reuse = {};
  g_budget = view_budget();
  scene::set_identity_demand(g_view != ClassView::Off);
}

bool push_view(gx::DrawData& draw, const gx::PipelineConfig& config, const gx::ShaderInfo& info,
               const scene::DrawClass& cls) noexcept {
  if (g_view == ClassView::Off) {
    return false;
  }
  scene::ViewCounts& counts = scene::view_counts();
  if (!in_view_span(cls) || config.shaderConfig.tevStageCount == 0) {
    ++counts.untouched;
    return false;
  }
  // The draw's pipeline key already includes the target layout, so it also keys the variant
  auto [it, inserted] = g_variants.try_emplace(draw.pipeline);
  Variant& variant = it->second;
  if (inserted) {
    gx::PipelineConfig variantConfig{};
    if (!make_view_variant(config, info, variantConfig)) {
      variant.refused = true;
    } else {
      const gx::TevStage& last = variantConfig.shaderConfig.tevStages[variantConfig.shaderConfig.tevStageCount - 1];
      variant.konst = static_cast<uint8_t>(last.kcSel - GX_TEV_KCSEL_K0);
      variant.pipeline = g_lookup(variantConfig, gfx::get_render_target_layout());
      variant.info = gx::build_shader_info(variantConfig.shaderConfig);
    }
  }
  if (variant.refused) {
    ++counts.untouched;
    return false;
  }
  if (!variant.ready) {
    gfx::CompiledPipeline compiled;
    variant.ready = g_assumeReady || gfx::get_pipeline(variant.pipeline, compiled);
    // The draw keeps its own pipeline rather than vanishing while the variant compiles
    if (!variant.ready) {
      ++counts.pending;
      return false;
    }
  }

  const Rgb8 color = view_color(g_view, g_activeBit, cls);
  const bool reuse = g_reuse.valid && g_reuse.source == draw.uniformRange && g_reuse.pipeline == variant.pipeline &&
                     g_reuse.color == color;
  if (!reuse) {
    if (counts.uniformBytes + gfx::align_uniform(variant.info.uniformSize) > g_budget) {
      ++counts.overBudget;
      if (!std::exchange(g_warnedBudget, true)) {
        Log.warn("Debug view: skipped draws past the {} MiB view uniform budget this frame", g_budget >> 20);
      }
      return false;
    }
    // The budget follows last frame; this frame's mapped buffer must still have room, or the push would abort
    if (gfx::uniform_space_left() < variant.info.uniformSize) {
      ++counts.overBudget;
      if (!std::exchange(g_warnedSpace, true)) {
        Log.warn("Debug view: skipped draws the frame's uniform buffer had no room left for");
      }
      return false;
    }
    auto& kcolor = gx::g_gxState.kcolors[variant.konst];
    const Vec4<float> saved = kcolor;
    kcolor = {color.r / 255.f, color.g / 255.f, color.b / 255.f, saved.w()};
    const gfx::Range range = gx::build_uniform(variant.info);
    kcolor = saved;
    counts.uniformBytes += gfx::align_uniform(range.size);
    g_reuse = {
        .source = draw.uniformRange, .pipeline = variant.pipeline, .color = color, .range = range, .valid = true};
  }
  draw.pipeline = variant.pipeline;
  draw.uniformRange = g_reuse.range;
  ++counts.substituted;
  return true;
}

namespace testing {
void set_variant_lookup(VariantLookup lookup) noexcept {
  g_lookup = lookup != nullptr ? lookup : &gfx::find_debug_pipeline;
  g_variants.clear();
}

void assume_variants_ready(bool ready) noexcept { g_assumeReady = ready; }

void set_view_uniform_budget(uint32_t bytes) noexcept {
  g_budgetOverride = bytes;
  if (bytes != 0) {
    g_budget = bytes;
  }
}

void reset_views() noexcept {
  g_view = ClassView::Off;
  g_activeBit = 0;
  g_variants.clear();
  g_reuse = {};
  g_budget = MaxViewUniformBytes;
  g_lastViewBytes = 0;
  g_warnedBudget = false;
  g_warnedSpace = false;
  g_lookup = &gfx::find_debug_pipeline;
  g_assumeReady = false;
  g_budgetOverride = 0;
  scene::set_identity_demand(false);
}

ClassView active_view() noexcept { return g_view; }

uint32_t view_uniform_budget() noexcept { return g_budget; }
} // namespace testing
} // namespace aurora::render::debug

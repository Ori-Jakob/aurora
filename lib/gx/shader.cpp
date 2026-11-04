#include "../gfx/hash.hpp"
#include "../gfx/scene_targets.hpp"
#include "../gfx/types.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "hex_wgsl.hpp"
#include "render_ext.hpp"
#include "shader_info.hpp"
#include "surface_wgsl.hpp"
#include "unlit_plan.hpp"
#include "vertex_fetch.hpp"

#include <dolphin/gx/GXEnum.h>

#include <absl/container/flat_hash_set.h>
#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tracy/Tracy.hpp"

namespace aurora::gx {
using namespace fmt::literals;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {
constexpr Module Log{"aurora::gfx::gx"};
using vertex_fetch::attr_load;
using vertex_fetch::attr_load_nbt_slice;
using vertex_fetch::nbt_slice_local;
using vertex_fetch::NbtSlice;
using vertex_fetch::vtx_attr;

std::string_view chan_comp(GXTevColorChan chan) noexcept {
  switch (chan) {
  case GX_CH_RED:
    return "r";
  case GX_CH_GREEN:
    return "g";
  case GX_CH_BLUE:
    return "b";
  case GX_CH_ALPHA:
    return "a";
  default:
    return "?";
  }
}

bool is_alpha_bump_channel(GXChannelID id) noexcept { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

std::string tev_mask_expr(const std::string& value, u32 mask) {
  // t_IndTexCoord is already expanded into the 0..255 indirect sample domain.
  return fmt::format("(f32(u32({}) & 0x{:X}u) / 255.0)", value, mask);
}

std::string alpha_bump_sel(size_t stageIdx, const ShaderConfig& config, const TevStage& stage) {
  if (stage.indTexStage >= config.numIndStages || stage.indTexAlphaSel == GX_ITBA_OFF) {
    return "0.0";
  }

  std::string baseCoord;
  switch (stage.indTexAlphaSel) {
    DEFAULT_FATAL("invalid indTexAlphaSel {} for stage {}", underlying(stage.indTexAlphaSel), stageIdx);
  case GX_ITBA_S:
    baseCoord = fmt::format("t_IndTexCoord{}.x", underlying(stage.indTexStage));
    break;
  case GX_ITBA_T:
    baseCoord = fmt::format("t_IndTexCoord{}.y", underlying(stage.indTexStage));
    break;
  case GX_ITBA_U:
    baseCoord = fmt::format("t_IndTexCoord{}.z", underlying(stage.indTexStage));
    break;
  case GX_ITBA_OFF:
    return "0.0";
  }

  switch (stage.indTexFormat) {
    DEFAULT_FATAL("invalid indirect format {} for stage {}", underlying(stage.indTexFormat), stageIdx);
  case GX_ITF_8:
    return tev_mask_expr(baseCoord, 0xF8u);
  case GX_ITF_5:
    return tev_mask_expr(baseCoord, 0xE0u);
  case GX_ITF_4:
    return tev_mask_expr(baseCoord, 0xF0u);
  case GX_ITF_3:
    return tev_mask_expr(baseCoord, 0xF8u);
  }
}

std::string color_arg_reg(GXTevColorArg arg, size_t stageIdx, const ShaderConfig& config, const TevStage& stage) {
  switch (arg) {
    DEFAULT_FATAL("invalid color arg {}", underlying(arg));
  case GX_CC_CPREV:
    return "prev.rgb";
  case GX_CC_APREV:
    return "vec3f(prev.a)";
  case GX_CC_C0:
    return "tevreg0.rgb";
  case GX_CC_A0:
    return "vec3f(tevreg0.a)";
  case GX_CC_C1:
    return "tevreg1.rgb";
  case GX_CC_A1:
    return "vec3f(tevreg1.a)";
  case GX_CC_C2:
    return "tevreg2.rgb";
  case GX_CC_A2:
    return "vec3f(tevreg2.a)";
  case GX_CC_TEXC: {
    if (stage.texMapId == GX_TEXMAP_NULL) {
      return "vec3f(1.0)";
    }
    CHECK(stage.texMapId >= GX_TEXMAP0 && stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("sampled{}.{}{}{}", stageIdx, chan_comp(swap.red), chan_comp(swap.green), chan_comp(swap.blue));
  }
  case GX_CC_TEXA: {
    if (stage.texMapId == GX_TEXMAP_NULL) {
      return "vec3f(1.0)";
    }
    CHECK(stage.texMapId >= GX_TEXMAP0 && stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("vec3f(sampled{}.{})", stageIdx, chan_comp(swap.alpha));
  }
  case GX_CC_RASC: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "vec3f(0.0)";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return fmt::format("vec3f({})", alpha);
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("rast{}.{}{}{}", idx, chan_comp(swap.red), chan_comp(swap.green), chan_comp(swap.blue));
  }
  case GX_CC_RASA: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "vec3f(0.0)";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return fmt::format("vec3f({})", alpha);
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("vec3f(rast{}.{})", idx, chan_comp(swap.alpha));
  }
  case GX_CC_ONE:
    return "vec3f(1.0)";
  case GX_CC_HALF:
    return "vec3f(0.5)";
  case GX_CC_KONST: {
    switch (stage.kcSel) {
      DEFAULT_FATAL("invalid kcSel {}", underlying(stage.kcSel));
    case GX_TEV_KCSEL_8_8:
      return "vec3f(1.0)";
    case GX_TEV_KCSEL_7_8:
      return "vec3f(7.0/8.0)";
    case GX_TEV_KCSEL_6_8:
      return "vec3f(6.0/8.0)";
    case GX_TEV_KCSEL_5_8:
      return "vec3f(5.0/8.0)";
    case GX_TEV_KCSEL_4_8:
      return "vec3f(4.0/8.0)";
    case GX_TEV_KCSEL_3_8:
      return "vec3f(3.0/8.0)";
    case GX_TEV_KCSEL_2_8:
      return "vec3f(2.0/8.0)";
    case GX_TEV_KCSEL_1_8:
      return "vec3f(1.0/8.0)";
    case GX_TEV_KCSEL_K0:
      return "ubuf.kcolor0.rgb";
    case GX_TEV_KCSEL_K1:
      return "ubuf.kcolor1.rgb";
    case GX_TEV_KCSEL_K2:
      return "ubuf.kcolor2.rgb";
    case GX_TEV_KCSEL_K3:
      return "ubuf.kcolor3.rgb";
    case GX_TEV_KCSEL_K0_R:
      return "vec3f(ubuf.kcolor0.r)";
    case GX_TEV_KCSEL_K1_R:
      return "vec3f(ubuf.kcolor1.r)";
    case GX_TEV_KCSEL_K2_R:
      return "vec3f(ubuf.kcolor2.r)";
    case GX_TEV_KCSEL_K3_R:
      return "vec3f(ubuf.kcolor3.r)";
    case GX_TEV_KCSEL_K0_G:
      return "vec3f(ubuf.kcolor0.g)";
    case GX_TEV_KCSEL_K1_G:
      return "vec3f(ubuf.kcolor1.g)";
    case GX_TEV_KCSEL_K2_G:
      return "vec3f(ubuf.kcolor2.g)";
    case GX_TEV_KCSEL_K3_G:
      return "vec3f(ubuf.kcolor3.g)";
    case GX_TEV_KCSEL_K0_B:
      return "vec3f(ubuf.kcolor0.b)";
    case GX_TEV_KCSEL_K1_B:
      return "vec3f(ubuf.kcolor1.b)";
    case GX_TEV_KCSEL_K2_B:
      return "vec3f(ubuf.kcolor2.b)";
    case GX_TEV_KCSEL_K3_B:
      return "vec3f(ubuf.kcolor3.b)";
    case GX_TEV_KCSEL_K0_A:
      return "vec3f(ubuf.kcolor0.a)";
    case GX_TEV_KCSEL_K1_A:
      return "vec3f(ubuf.kcolor1.a)";
    case GX_TEV_KCSEL_K2_A:
      return "vec3f(ubuf.kcolor2.a)";
    case GX_TEV_KCSEL_K3_A:
      return "vec3f(ubuf.kcolor3.a)";
    }
  }
  case GX_CC_ZERO:
    return "vec3f(0.0)";
  }
}

std::string alpha_arg_reg(GXTevAlphaArg arg, size_t stageIdx, const ShaderConfig& config, const TevStage& stage) {
  switch (arg) {
    DEFAULT_FATAL("invalid alpha arg {}", underlying(arg));
  case GX_CA_APREV:
    return "prev.a";
  case GX_CA_A0:
    return "tevreg0.a";
  case GX_CA_A1:
    return "tevreg1.a";
  case GX_CA_A2:
    return "tevreg2.a";
  case GX_CA_TEXA: {
    if (stage.texMapId == GX_TEXMAP_NULL) {
      return "1.0";
    }
    CHECK(stage.texMapId >= GX_TEXMAP0 && stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("sampled{}.{}", stageIdx, chan_comp(swap.alpha));
  }
  case GX_CA_RASA: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "0.0";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return alpha;
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("rast{}.{}", idx, chan_comp(swap.alpha));
  }
  case GX_CA_KONST: {
    switch (stage.kaSel) {
      DEFAULT_FATAL("invalid kaSel {}", underlying(stage.kaSel));
    case GX_TEV_KASEL_8_8:
      return "1.0";
    case GX_TEV_KASEL_7_8:
      return "(7.0/8.0)";
    case GX_TEV_KASEL_6_8:
      return "(6.0/8.0)";
    case GX_TEV_KASEL_5_8:
      return "(5.0/8.0)";
    case GX_TEV_KASEL_4_8:
      return "(4.0/8.0)";
    case GX_TEV_KASEL_3_8:
      return "(3.0/8.0)";
    case GX_TEV_KASEL_2_8:
      return "(2.0/8.0)";
    case GX_TEV_KASEL_1_8:
      return "(1.0/8.0)";
    case GX_TEV_KASEL_K0_R:
      return "ubuf.kcolor0.r";
    case GX_TEV_KASEL_K1_R:
      return "ubuf.kcolor1.r";
    case GX_TEV_KASEL_K2_R:
      return "ubuf.kcolor2.r";
    case GX_TEV_KASEL_K3_R:
      return "ubuf.kcolor3.r";
    case GX_TEV_KASEL_K0_G:
      return "ubuf.kcolor0.g";
    case GX_TEV_KASEL_K1_G:
      return "ubuf.kcolor1.g";
    case GX_TEV_KASEL_K2_G:
      return "ubuf.kcolor2.g";
    case GX_TEV_KASEL_K3_G:
      return "ubuf.kcolor3.g";
    case GX_TEV_KASEL_K0_B:
      return "ubuf.kcolor0.b";
    case GX_TEV_KASEL_K1_B:
      return "ubuf.kcolor1.b";
    case GX_TEV_KASEL_K2_B:
      return "ubuf.kcolor2.b";
    case GX_TEV_KASEL_K3_B:
      return "ubuf.kcolor3.b";
    case GX_TEV_KASEL_K0_A:
      return "ubuf.kcolor0.a";
    case GX_TEV_KASEL_K1_A:
      return "ubuf.kcolor1.a";
    case GX_TEV_KASEL_K2_A:
      return "ubuf.kcolor2.a";
    case GX_TEV_KASEL_K3_A:
      return "ubuf.kcolor3.a";
    }
  }
  case GX_CA_ZERO:
    return "0.0";
  }
}

bool tev_color_arg_is_normalized(GXTevColorArg arg, const std::array<bool, MaxTevRegs>& colorNormalized,
                                 const std::array<bool, MaxTevRegs>& alphaNormalized) {
  switch (arg) {
  case GX_CC_CPREV:
    return colorNormalized[GX_TEVPREV];
  case GX_CC_APREV:
    return alphaNormalized[GX_TEVPREV];
  case GX_CC_C0:
    return colorNormalized[GX_TEVREG0];
  case GX_CC_A0:
    return alphaNormalized[GX_TEVREG0];
  case GX_CC_C1:
    return colorNormalized[GX_TEVREG1];
  case GX_CC_A1:
    return alphaNormalized[GX_TEVREG1];
  case GX_CC_C2:
    return colorNormalized[GX_TEVREG2];
  case GX_CC_A2:
    return alphaNormalized[GX_TEVREG2];
  default:
    return true;
  }
}

bool tev_alpha_arg_is_normalized(GXTevAlphaArg arg, const std::array<bool, MaxTevRegs>& alphaNormalized) {
  switch (arg) {
  case GX_CA_APREV:
    return alphaNormalized[GX_TEVPREV];
  case GX_CA_A0:
    return alphaNormalized[GX_TEVREG0];
  case GX_CA_A1:
    return alphaNormalized[GX_TEVREG1];
  case GX_CA_A2:
    return alphaNormalized[GX_TEVREG2];
  default:
    return true;
  }
}

std::string tev_op(GXTevOp op, std::string_view bias, std::string_view scale, std::string_view a, std::string_view b,
                   std::string_view c, std::string_view d, std::string_view zero) {
  switch (op) {
    DEFAULT_FATAL("unimplemented tev op {}", underlying(op));
  case GX_TEV_ADD:
  case GX_TEV_SUB: {
    std::string_view neg = op == GX_TEV_SUB ? "-"sv : ""sv;
    return fmt::format("(({0}mix({1}, {2}, {3}) + {4}){5}){6}", neg, a, b, c, d, bias, scale);
  }
  case GX_TEV_COMP_R8_GT:
    return fmt::format("select({3}, {2}, round({0}.r * 255.0) > round({1}.r * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_R8_EQ:
    return fmt::format("select({3}, {2}, round({0}.r * 255.0) == round({1}.r * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_GR16_GT:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rg * 255.0, vec2(1.0, 256.0))) > round(dot({1}.rg * 255.0, vec2(1.0, 256.0))))"
        " + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_GR16_EQ:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rg * 255.0, vec2(1.0, 256.0))) == round(dot({1}.rg * 255.0, vec2(1.0, 256.0))))"
        " + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_BGR24_GT:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rgb * 255.0, vec3(1.0, 256.0, 65536.0))) > round(dot({1}.rgb * 255.0, "
        "vec3(1.0, 256.0, 65536.0)))) + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_BGR24_EQ:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rgb * 255.0, vec3(1.0, 256.0, 65536.0))) == round(dot({1}.rgb * 255.0, "
        "vec3(1.0, 256.0, 65536.0)))) + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_RGB8_GT:
    return fmt::format("select({3}, {2}, round({0} * 255.0) > round({1} * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_RGB8_EQ:
    return fmt::format("select({3}, {2}, round({0} * 255.0) == round({1} * 255.0)) + {4}", a, b, c, zero, d);
  }
}

std::string tev_color_op(GXTevOp op, std::string_view bias, std::string_view scale, bool clamp, std::string_view a,
                         std::string_view b, std::string_view c, std::string_view d) {
  std::string expr = tev_op(op, bias, scale, a, b, c, d, "vec3(0)"sv);
  return clamp ? fmt::format("clamp({}, vec3f(0.0), vec3f(1.0))", expr)
               : fmt::format("clamp({}, vec3f(-4.0), vec3f(4.0))", expr);
}

std::string tev_alpha_op(GXTevOp op, std::string_view bias, std::string_view scale, bool clamp, std::string_view a,
                         std::string_view b, std::string_view c, std::string_view d) {
  std::string expr = tev_op(op, bias, scale, a, b, c, d, "0.0"sv);
  return clamp ? fmt::format("clamp({}, 0.0, 1.0)", expr) : fmt::format("clamp({}, -4.0, 4.0)", expr);
}

std::string_view tev_bias(GXTevBias bias) {
  switch (bias) {
    DEFAULT_FATAL("invalid tev bias {}", underlying(bias));
  case GX_TB_ZERO:
    return ""sv;
  case GX_TB_ADDHALF:
    return " + 0.5"sv;
  case GX_TB_SUBHALF:
    return " - 0.5"sv;
  }
}

struct AlphaCompareExpr {
  std::string expr;
  int constant = -1;
};

AlphaCompareExpr alpha_compare_const(bool value) { return {value ? "true"s : "false"s, value ? 1 : 0}; }

AlphaCompareExpr alpha_compare_not(const AlphaCompareExpr& expr) {
  if (expr.constant != -1) {
    return alpha_compare_const(expr.constant == 0);
  }
  return {fmt::format("!{}", expr.expr), -1};
}

AlphaCompareExpr alpha_compare_and(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant == 0 || rhs.constant == 0) {
    return alpha_compare_const(false);
  }
  if (lhs.constant == 1) {
    return rhs;
  }
  if (rhs.constant == 1) {
    return lhs;
  }
  return {fmt::format("({} && {})", lhs.expr, rhs.expr), -1};
}

AlphaCompareExpr alpha_compare_or(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant == 1 || rhs.constant == 1) {
    return alpha_compare_const(true);
  }
  if (lhs.constant == 0) {
    return rhs;
  }
  if (rhs.constant == 0) {
    return lhs;
  }
  return {fmt::format("({} || {})", lhs.expr, rhs.expr), -1};
}

AlphaCompareExpr alpha_compare_xor(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant != -1 && rhs.constant != -1) {
    return alpha_compare_const(lhs.constant != rhs.constant);
  }
  if (lhs.constant == 0) {
    return rhs;
  }
  if (rhs.constant == 0) {
    return lhs;
  }
  if (lhs.constant == 1) {
    return alpha_compare_not(rhs);
  }
  if (rhs.constant == 1) {
    return alpha_compare_not(lhs);
  }
  return {fmt::format("({} != {})", lhs.expr, rhs.expr), -1};
}

AlphaCompareExpr alpha_compare_xnor(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant != -1 && rhs.constant != -1) {
    return alpha_compare_const(lhs.constant == rhs.constant);
  }
  if (lhs.constant == 0) {
    return alpha_compare_not(rhs);
  }
  if (rhs.constant == 0) {
    return alpha_compare_not(lhs);
  }
  if (lhs.constant == 1) {
    return rhs;
  }
  if (rhs.constant == 1) {
    return lhs;
  }
  return {fmt::format("({} == {})", lhs.expr, rhs.expr), -1};
}

AlphaCompareExpr alpha_compare(GXCompare comp, u8 ref) {
  const auto iref = static_cast<u32>(ref);
  switch (comp) {
    DEFAULT_FATAL("invalid alpha comp {}", underlying(comp));
  case GX_NEVER:
    return alpha_compare_const(false);
  case GX_LESS:
    if (ref == 0) {
      return alpha_compare_const(false);
    }
    return {fmt::format("(alphaCompare < {}u)", iref), -1};
  case GX_LEQUAL:
    if (ref == 255) {
      return alpha_compare_const(true);
    }
    return {fmt::format("(alphaCompare <= {}u)", iref), -1};
  case GX_EQUAL:
    return {fmt::format("(alphaCompare == {}u)", iref), -1};
  case GX_NEQUAL:
    return {fmt::format("(alphaCompare != {}u)", iref), -1};
  case GX_GEQUAL:
    if (ref == 0) {
      return alpha_compare_const(true);
    }
    return {fmt::format("(alphaCompare >= {}u)", iref), -1};
  case GX_GREATER:
    if (ref == 255) {
      return alpha_compare_const(false);
    }
    return {fmt::format("(alphaCompare > {}u)", iref), -1};
  case GX_ALWAYS:
    return alpha_compare_const(true);
  }
}

std::string_view tev_scale(GXTevScale scale) {
  switch (scale) {
    DEFAULT_FATAL("invalid tev scale {}", underlying(scale));
  case GX_CS_SCALE_1:
    return ""sv;
  case GX_CS_SCALE_2:
    return " * 2.0"sv;
  case GX_CS_SCALE_4:
    return " * 4.0"sv;
  case GX_CS_DIVIDE_2:
    return " / 2.0"sv;
  }
}

constexpr std::array<std::string_view, GX_CC_ZERO + 1> TevColorArgNames{
    "CPREV"sv, "APREV"sv, "C0"sv,   "A0"sv,   "C1"sv,  "A1"sv,   "C2"sv,    "A2"sv,
    "TEXC"sv,  "TEXA"sv,  "RASC"sv, "RASA"sv, "ONE"sv, "HALF"sv, "KONST"sv, "ZERO"sv,
};
constexpr std::array<std::string_view, GX_CA_ZERO + 1> TevAlphaArgNames{
    "APREV"sv, "A0"sv, "A1"sv, "A2"sv, "TEXA"sv, "RASA"sv, "KONST"sv, "ZERO"sv,
};

constexpr bool is_emboss_texgen(GXTexGenType type) noexcept { return type >= GX_TG_BUMP0 && type <= GX_TG_BUMP7; }

auto lighting_func(const ShaderConfig& config, const ColorChannelConfig& cc, u8 i, bool alpha) -> std::string {
  std::string_view swizzle = alpha ? ".a"sv : ""sv;
  std::string outVar = fmt::format("out.cc{}", i);
  std::string_view posVar = "mv_pos"sv;
  std::string_view nrmVar = "mv_nrm"sv;

  std::string ambSrc, matSrc;
  if (cc.ambSrc == GX_SRC_VTX) {
    ambSrc = vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i));
  } else if (cc.ambSrc == GX_SRC_REG) {
    ambSrc = fmt::format("ubuf.cc{0}{1}_amb", i, alpha ? "a"sv : ""sv);
  }

  if (cc.matSrc == GX_SRC_VTX) {
    matSrc = vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i));
  } else if (cc.matSrc == GX_SRC_REG) {
    matSrc = fmt::format("ubuf.cc{0}{1}_mat", i, alpha ? "a"sv : ""sv);
  }

  if (!cc.lightingEnabled) {
    return fmt::format("\n    {0}{2} = {1}{2};", outVar, matSrc, swizzle);
  }
  GXDiffuseFn diffFn = cc.diffFn;
  std::string lightAttnFn;
  if (cc.attnFn == GX_AF_NONE) {
    lightAttnFn = "attn = 1.0;"s;
  } else if (cc.attnFn == GX_AF_SPOT) {
    lightAttnFn = fmt::format(R"""(
          var cosine = max(0.0, dot(ldir, light.dir));
          var cos_attn = dot(light.cos_att, vec3f(1.0, cosine, cosine * cosine));
          var dist_attn = dot(light.dist_att, vec3f(1.0, dist, dist2));
          attn = max(0.0, cos_attn / dist_attn);)""");
  } else if (cc.attnFn == GX_AF_SPEC) {
    std::string dist_attn = diffFn != GX_DF_NONE
                                ? "max(0.0, dot(normalize(light.dist_att), vec3f(1.0, attn, attn * attn)));"
                                : "max(0.0, dot(light.dist_att, vec3f(1.0, attn, attn * attn)));";
    lightAttnFn = fmt::format(R"""(
          attn = select(0.0, max(0.0, dot({0}, light.dir)), dot({0}, ldir) >= 0.0);
          var cos_attn = dot(light.cos_att, vec3f(1.0, attn, attn * attn));
          var dist_attn = {1};
          attn = max(0.0, cos_attn / dist_attn);)""",
                              nrmVar, dist_attn);
  }
  std::string_view lightDiffFn;
  if (diffFn == GX_DF_NONE) {
    lightDiffFn = "1.0"sv;
  } else if (diffFn == GX_DF_SIGN) {
    lightDiffFn = "dot(ldir, mv_nrm)"sv;
  } else if (diffFn == GX_DF_CLAMP) {
    lightDiffFn = "max(0.0, dot(ldir, mv_nrm))"sv;
  }
  return fmt::format(R"""(
    {{
      var lighting = {5};
      for (var i = 0u; i < {1}u; i++) {{
          if ((ubuf.lightState{0}{9} & (1u << i)) == 0u) {{ continue; }}
          var light = ubuf.lights[i];
          var ldir = light.pos - {6};
          var dist2 = dot(ldir, ldir);
          var dist = sqrt(dist2);
          ldir = ldir / dist;
          var attn: f32;{2}
          var diff = {3};
          lighting = lighting + (attn * diff * light.color);
      }}
      {7}{8} = ({4} * clamp(lighting, vec4f(0.0), vec4f(1.0))){8};
    }})""",
                     i, GX::MaxLights, lightAttnFn, lightDiffFn, matSrc, ambSrc, posVar, outVar, swizzle,
                     alpha ? "a"sv : ""sv);
}

absl::flat_hash_set<gfx::ShaderRef> s_seenShaders;

HashType shader_hash(const ShaderConfig& config, DstAlphaMode dstAlphaMode, const ShaderTargets& targets) noexcept {
  auto hash = xxh3_hash(dstAlphaMode, xxh3_hash(targets.normal, xxh3_hash(config)));
  for (const uint32_t target : {targets.debugId, targets.motion, targets.reactive, targets.gbufferNormal, targets.fog,
                                targets.surface, targets.surfaceExt}) {
    if (target != UINT32_MAX) {
      hash = xxh3_hash(target, hash);
    }
  }
  // Only text that carries the environment differs, so every M7 hash and label stays as it was
  if (uses_forward_env(config, targets)) {
    hash = xxh3_hash(targets.fwdEnv, hash);
  }
  return hash;
}

struct OutputMember {
  uint32_t location;
  std::string declaration;
  std::string assignment;
};

struct UnlitPass {
  u16 scaleDivide = 0;
};

struct TevNormalized {
  bool color = false;
  bool alpha = false;
};

std::string replace_all(std::string text, std::string_view from, std::string_view to) {
  for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size())) {
    text.replace(pos, from.size(), to);
  }
  return text;
}

std::string surface_ext_ambient(const ShaderConfig& config, const ShaderInfo& info) {
  const unlit::Ambient ambient = unlit::ambient_of(config, info);
  const u32 c = ambient.channel;
  const bool raw = (info.unlitRawVtx >> c & 1u) != 0;
  const ColorChannelConfig& channel = config.colorChannels[c];
  std::string value;
  if (ambient.source == unlit::AmbientSource::Register && channel.lightingEnabled && channel.ambSrc == GX_SRC_REG &&
      info.sampledColorChannels.test(c)) {
    value = fmt::format("ubuf.cc{}_amb.rgb", c);
  } else if (ambient.source == unlit::AmbientSource::Vertex && raw) {
    value = fmt::format("in.ccv{}", c);
  } else {
    render_ext::g_stats.pbrAmbientUnknown.fetch_add(1, std::memory_order_relaxed);
    return "vec3f(0.0)";
  }
  if (ambient.rawVertex && raw) {
    value += fmt::format(" * select(vec3f(1.0), in.ccv{}, ((gx_record_word({}u) >> {}u) & 3u) == 1u)", c,
                         render_ext::DrawRecordFlagsWord, render_ext::RecordVertexColorShift);
  }
  return fmt::format("select(vec3f(0.0), {}, (umat & {:#x}u) != 0u)", value, render_ext::MaterialWordAmbient);
}

bool direct_stage(const TevStage& stage) noexcept {
  return stage.indTexMtxId == GX_ITM_OFF && stage.indTexWrapS == GX_ITW_OFF && stage.indTexWrapT == GX_ITW_OFF &&
         !stage.indTexAddPrev;
}

struct CompanionSite {
  u32 map = 0;
  u32 coord = 0;
};

CompanionSite companion_site(const ShaderConfig& config, const ShaderInfo& info) noexcept {
  const u32 map = config.ext.layer0 - 1u;
  std::optional<u32> fallback;
  for (u32 s = 0; s < config.tevStageCount; ++s) {
    const auto& stage = config.tevStages[s];
    if (!uses_texture_sample(stage) || stage.texMapId != map || stage.texCoordId == GX_TEXCOORD_NULL) {
      continue;
    }
    if ((info.unlitLayerStages >> s & 1u) != 0) {
      return {map, static_cast<u32>(stage.texCoordId)};
    }
    if (!fallback && direct_stage(stage)) {
      fallback = static_cast<u32>(stage.texCoordId);
    }
  }
  return {map, fallback.value_or(0)};
}

constexpr std::string_view CompanionWgsl = R"(
fn gx_linear_to_srgb(c: vec3f) -> vec3f {
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, c * 12.92, c <= vec3f(0.0031308));
}
fn gx_srgb_to_linear(c: vec3f) -> vec3f {
    let hi = pow((max(c, vec3f(0.0)) + 0.055) / 1.055, vec3f(2.4));
    return select(hi, c / 12.92, c <= vec3f(0.04045));
}
fn gx_linear_to_oklab(c: vec3f) -> vec3f {
    let lms = c * mat3x3f(vec3f(0.4122214708, 0.5363325363, 0.0514459929),
                          vec3f(0.2119034982, 0.6806995451, 0.1073969566),
                          vec3f(0.0883024619, 0.2817188376, 0.6299787005));
    return pow(max(lms, vec3f(0.0)), vec3f(1.0 / 3.0)) * mat3x3f(vec3f(0.2104542553, 0.7936177850, -0.0040720468),
                                                                 vec3f(1.9779984951, -2.4285922050, 0.4505937099),
                                                                 vec3f(0.0259040371, 0.7827717662, -0.8086757660));
}
fn gx_oklab_to_linear(lab: vec3f) -> vec3f {
    let l = lab * mat3x3f(vec3f(1.0, 0.3963377774, 0.2158037573),
                          vec3f(1.0, -0.1055613458, -0.0638541728),
                          vec3f(1.0, -0.0894841775, -1.2914855480));
    return (l * l * l) * mat3x3f(vec3f(4.0767416621, -3.3077115913, 0.2309699292),
                                 vec3f(-1.2684380046, 2.6097574011, -0.3413193965),
                                 vec3f(-0.0041960863, -0.7034186147, 1.7076147010));
}
// Provisional: S3 works on Oklab low-frequency L (rev3.md:1371); pinned by the first S3 sample
fn gx_companion_delight(c: vec3f, d: f32) -> vec3f {
    let lab = gx_linear_to_oklab(gx_srgb_to_linear(c));
    let k = exp2(-(d - 0.5) * 8.0);
    return gx_linear_to_srgb(saturate(gx_oklab_to_linear(vec3f(lab.x * k, lab.yz))));
}
fn gx_view_pos(c: vec3f) -> vec3f {
    let z = -c.z;
    return vec3f((c.x - ubuf.proj[0].z * z - ubuf.proj[0].w) / ubuf.proj[0].x,
                 (c.y - ubuf.proj[1].z * z - ubuf.proj[1].w) / ubuf.proj[1].y, z);
}
// C1 +Y up: green points toward v = 0, the first stored row; the solve ignores screen y direction and handles mirrored UVs
fn gx_perturb(n: vec3f, dp1: vec3f, dp2: vec3f, duv1: vec2f, duv2: vec2f, t: vec3f) -> vec3f {
    let j = duv1.x * duv2.y - duv2.x * duv1.y;
    let s = select(-1.0, 1.0, j >= 0.0);
    let pu = s * (dp1 * duv2.y - dp2 * duv1.y);
    let pv = s * (dp2 * duv1.x - dp1 * duv2.x);
    let tu = pu - n * dot(n, pu);
    let tv = pv - n * dot(n, pv);
    let uv2 = dot(duv1, duv1) + dot(duv2, duv2);
    let p2 = (dot(dp1, dp1) + dot(dp2, dp2)) * uv2;
    if (abs(j) <= 1e-6 * uv2 || dot(tu, tu) <= 1e-6 * p2 || dot(tv, tv) <= 1e-6 * p2) {
        return n;
    }
    return normalize(t.x * normalize(tu) - t.y * normalize(tv) + t.z * n);
}
fn gx_companion_tangent(c: vec4f, lane: vec4f) -> vec3f {
    let mat = (u32(lane.x) & 4u) != 0u;
    let xy = select(c.rg * 2.0 - 1.0, vec2f(c.a, c.g) * (255.0 / 127.0) - 128.0 / 127.0, mat) * vec2f(1.0, lane.z);
    return normalize(vec3f(xy * lane.y, sqrt(saturate(1.0 - dot(xy, xy)))));
})";
} // namespace

std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode,
                                uint32_t normalAttachment) noexcept {
  return build_shader_source(config, dstAlphaMode, ShaderTargets{.normal = normalAttachment});
}

std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode,
                                const ShaderTargets& targets) noexcept {
  ZoneScoped;
  // The receiver normal reuses the legacy normal path; only its alpha differs
  const uint32_t nrmTarget = normal_target(config, targets);
  const bool receiverNormal = nrmTarget != UINT32_MAX && nrmTarget != targets.normal;
  const auto hash = shader_hash(config, dstAlphaMode, targets);
  const auto info = build_shader_info(config);
  const bool companionText = uses_companions(config, info, targets);
  const CompanionSite companion = companionText ? companion_site(config, info) : CompanionSite{};
  CHECK(!companionText || targets.motion != UINT32_MAX, "companion text needs the clip position varying");
  // Emission has only the SurfaceExt word to go to, so unkeyed companion text never samples it
  const bool emitText = companionText && uses_surface_ext(config, info, targets);
  if (EnableDebugPrints && !s_seenShaders.contains(hash)) {
    s_seenShaders.insert(hash);

    Log.info("Shader config (hash {:x}):", hash);
    {
      for (int i = 0; i < config.tevStageCount; ++i) {
        const auto& stage = config.tevStages[i];
        Log.info("  tevStages[{}]:", i);
        Log.info("    color_a: {}", TevColorArgNames[stage.colorPass.a]);
        Log.info("    color_b: {}", TevColorArgNames[stage.colorPass.b]);
        Log.info("    color_c: {}", TevColorArgNames[stage.colorPass.c]);
        Log.info("    color_d: {}", TevColorArgNames[stage.colorPass.d]);
        Log.info("    alpha_a: {}", TevAlphaArgNames[stage.alphaPass.a]);
        Log.info("    alpha_b: {}", TevAlphaArgNames[stage.alphaPass.b]);
        Log.info("    alpha_c: {}", TevAlphaArgNames[stage.alphaPass.c]);
        Log.info("    alpha_d: {}", TevAlphaArgNames[stage.alphaPass.d]);
        Log.info("    color_op_clamp: {}", stage.colorOp.clamp);
        Log.info("    color_op_op: {}", stage.colorOp.op);
        Log.info("    color_op_bias: {}", stage.colorOp.bias);
        Log.info("    color_op_scale: {}", stage.colorOp.scale);
        Log.info("    color_op_reg_id: {}", stage.colorOp.outReg);
        Log.info("    alpha_op_clamp: {}", stage.alphaOp.clamp);
        Log.info("    alpha_op_op: {}", stage.alphaOp.op);
        Log.info("    alpha_op_bias: {}", stage.alphaOp.bias);
        Log.info("    alpha_op_scale: {}", stage.alphaOp.scale);
        Log.info("    alpha_op_reg_id: {}", stage.alphaOp.outReg);
        Log.info("    kc_sel: {}", stage.kcSel);
        Log.info("    ka_sel: {}", stage.kaSel);
        Log.info("    texCoordId: {}", stage.texCoordId);
        Log.info("    texMapId: {}", stage.texMapId);
        Log.info("    channelId: {}", stage.channelId);
        Log.info("    tevSwapRas: {}", stage.tevSwapRas);
        Log.info("    tevSwapTex: {}", stage.tevSwapTex);
        Log.info("    indTexStage: {}", stage.indTexStage);
        Log.info("    indTexFormat: {}", stage.indTexFormat);
        Log.info("    indTexBiasSel: {}", stage.indTexBiasSel);
        Log.info("    indTexAlphaSel: {}", stage.indTexAlphaSel);
        Log.info("    indTexMtxId: {}", stage.indTexMtxId);
        Log.info("    indTexWrapS: {}", stage.indTexWrapS);
        Log.info("    indTexWrapT: {}", stage.indTexWrapT);
        Log.info("    indTexUseOrigLOD: {}", stage.indTexUseOrigLOD);
        Log.info("    indTexAddPrev: {}", stage.indTexAddPrev);
      }
      Log.info("  numIndStages: {}", config.numIndStages);
      for (u32 i = 0; i < config.numIndStages; ++i) {
        const auto& stage = config.indStages[i];
        Log.info("  indStages[{}]: texCoordId {} texMapId {} scaleS {} scaleT {}", i, stage.texCoordId, stage.texMapId,
                 stage.scaleS, stage.scaleT);
      }
      for (int i = 0; i < config.colorChannels.size(); ++i) {
        const auto& chan = config.colorChannels[i];
        Log.info("  colorChannels[{}]: enabled {} mat {} amb {}", static_cast<GXChannelID>(i), chan.lightingEnabled,
                 chan.matSrc, chan.ambSrc);
      }
      for (int i = 0; i < config.tcgs.size(); ++i) {
        const auto& tcg = config.tcgs[i];
        if (tcg.src != GX_MAX_TEXGENSRC) {
          Log.info("  tcg[{}]: src {} mtx {} post {} type {} norm {}", i, tcg.src, tcg.mtx, tcg.postMtx, tcg.type,
                   tcg.normalize);
        }
      }
      Log.info("  alphaCompare: comp0 {} ref0 {} op {} comp1 {} ref1 {}", config.alphaCompare.comp0,
               config.alphaCompare.ref0, config.alphaCompare.op, config.alphaCompare.comp1, config.alphaCompare.ref1);
      Log.info("  fogType: {}", config.fogType);
      Log.info("  fogRangeEnabled: {}", config.fogRangeEnabled);
    }
  }

  std::string uniformPre;
  std::string uniBufAttrs;
  std::string texBindings;
  std::string vtxOutAttrs;
  std::string vtxInAttrs;
  std::string vtxXfrAttrsPre;
  std::string vtxXfrAttrs;
  size_t vtxOutIdx = 0;

  // Load points for line/point expansion
  std::string_view vidxAttr = "vidx"sv;
  if (config.lineMode != 0) {
    vtxInAttrs += ",\n    @builtin(instance_index) iidx: u32";
    uniBufAttrs +=
        "\n    line_width: f32,"
        "\n    line_aspect_y: f32,"
        "\n    line_tex_offset: f32,"
        "\n    line_texcoord_mask: u32,";
    if (config.lineMode == 3) {
      // GX_POINTS: each instance = one vertex, expand to quad
      vtxXfrAttrsPre += fmt::format(
          "\n    let in_vidx = iidx;"
          "\n    let in_pos = {};"
          "\n    let in_pnmtxidx = {};"
          "\n    let mv_pos = vec4f(in_pos, 1.0) * ubuf.postex_mtx[in_pnmtxidx];",
          attr_load(config, GX_VA_POS, "in_vidx"sv), attr_load(config, GX_VA_PNMTXIDX, "in_vidx"sv));
    } else {
      // GX_LINES / GX_LINESTRIP: each instance = two vertices, expand to quad
      vtxXfrAttrsPre += fmt::format(
          "\n    let use_b = vidx >= 2u;"
          "\n    let vidx_a = iidx * {}u;"
          "\n    let vidx_b = vidx_a + 1u;"
          "\n    let in_vidx = select(vidx_a, vidx_b, use_b);"
          "\n    let pos_a = {};"
          "\n    let pos_b = {};"
          "\n    let in_pos = select(pos_a, pos_b, use_b);"
          "\n    let pnmtxidx_a = {};"
          "\n    let pnmtxidx_b = {};"
          "\n    let in_pnmtxidx = select(pnmtxidx_a, pnmtxidx_b, use_b);"
          "\n    let mv_pos_a = vec4f(pos_a, 1.0) * ubuf.postex_mtx[pnmtxidx_a];"
          "\n    let mv_pos_b = vec4f(pos_b, 1.0) * ubuf.postex_mtx[pnmtxidx_b];"
          "\n    let mv_pos = select(mv_pos_a, mv_pos_b, use_b);",
          config.lineMode == 1 ? 2 : 1, attr_load(config, GX_VA_POS, "vidx_a"sv),
          attr_load(config, GX_VA_POS, "vidx_b"sv), attr_load(config, GX_VA_PNMTXIDX, "vidx_a"sv),
          attr_load(config, GX_VA_PNMTXIDX, "vidx_b"sv));
    }
    vidxAttr = "in_vidx"sv;
  } else if (config.attrs[GX_VA_PNMTXIDX].attrType == GX_NONE) {
    vtxXfrAttrsPre += "\n    let in_pnmtxidx = imm.current_pnmtx;";
  }

  // Load vertex attributes
  for (GXAttr attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7; attr = static_cast<GXAttr>(attr + 1)) {
    const auto attrType = config.attrs[attr].attrType;
    if (attrType == GX_NONE) {
      continue;
    }
    // in_pnmtxidx and in_pos written above for line mode
    if ((attr != GX_VA_PNMTXIDX && attr != GX_VA_POS) || config.lineMode == 0) {
      vtxXfrAttrsPre += fmt::format("\n    let {} = {};", vtx_attr(config, attr), attr_load(config, attr, vidxAttr));
    }
  }
  bool needsBinrm = false;
  bool needsTangent = false;
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const bool emboss = is_emboss_texgen(config.tcgs[i].type);
    needsBinrm = needsBinrm || config.tcgs[i].src == GX_TG_BINRM || emboss;
    needsTangent = needsTangent || config.tcgs[i].src == GX_TG_TANGENT || emboss;
  }
  if (needsBinrm) {
    vtxXfrAttrsPre += fmt::format("\n    let {} = {};", nbt_slice_local(NbtSlice::B),
                                  attr_load_nbt_slice(config, NbtSlice::B, vidxAttr));
  }
  if (needsTangent) {
    vtxXfrAttrsPre += fmt::format("\n    let {} = {};", nbt_slice_local(NbtSlice::T),
                                  attr_load_nbt_slice(config, NbtSlice::T, vidxAttr));
  }

  if (config.lineMode == 0) {
    vtxXfrAttrsPre += fmt::format(
        "\n    let mv_pos = vec4f({}, 1.0) * ubuf.postex_mtx[in_pnmtxidx];"
        "\n    out.pos = vec4f(mv_pos, 1.0) * ubuf.proj;",
        vtx_attr(config, GX_VA_POS));
  } else if (config.lineMode == 3) {
    // GX_POINTS: expand single vertex to axis-aligned screen-space square
    vtxXfrAttrsPre +=
        "\n    let clip = vec4f(mv_pos, 1.0) * ubuf.proj;"
        "\n    let viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));"
        "\n    let point_size = ubuf.line_width * min(viewport_scale.x, viewport_scale.y);"
        "\n    let x_sign = select(-1.0, 1.0, (vidx & 1u) != 0u);"
        "\n    let y_sign = select(-1.0, 1.0, vidx >= 2u);"
        "\n    let offset_px = vec2f(x_sign, y_sign) * (point_size / 2.0);"
        "\n    let offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;"
        "\n    out.pos = vec4f(clip.xy + offset_ndc * clip.w, clip.zw);";
  } else {
    // GX_LINES / GX_LINESTRIP: expand line segment perpendicular to direction
    vtxXfrAttrsPre +=
        "\n    let clip_a = vec4f(mv_pos_a, 1.0) * ubuf.proj;"
        "\n    let clip_b = vec4f(mv_pos_b, 1.0) * ubuf.proj;"
        "\n    let ndc_a = clip_a.xy / clip_a.w;"
        "\n    let ndc_b = clip_b.xy / clip_b.w;"
        "\n    let viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));"
        "\n    let delta_px = (ndc_b - ndc_a) / 2.0 * ubuf.render_viewport_size;"
        "\n    let dir_px = select(vec2f(1.0, 0.0), normalize(delta_px), dot(delta_px, delta_px) > 1e-10);"
        "\n    let perp_px = vec2f(-dir_px.y, dir_px.x);"
        "\n    let line_width = ubuf.line_width * min(viewport_scale.x, viewport_scale.y);"
        "\n    let offset_px = perp_px * (line_width / 2.0) * select(-1.0, 1.0, (vidx & 1u) != 0u);"
        "\n    let offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;"
        "\n    let clip_base = select(clip_a, clip_b, use_b);"
        "\n    out.pos = vec4f(clip_base.xy + offset_ndc * clip_base.w, clip_base.zw);";
  }
  vtxXfrAttrsPre += fmt::format(
      "\n    let nrm_tmp = vec4f({}, 0.0) * ubuf.nrm_mtx[in_pnmtxidx];"
      "\n    let mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);",
      vtx_attr(config, GX_VA_NRM));
  const bool useNormalTarget = nrmTarget != UINT32_MAX && config.attrs[GX_VA_NRM].attrType != GX_NONE;
  const bool backFaceText = useNormalTarget && uses_unlit(config, info, targets);
  if (backFaceText) {
    // w is n.v at the vertex, so the back-face flip leaves alone game normals that face the viewer from both sides
    vtxOutAttrs += fmt::format("\n    @location({}) mv_nrm: vec4f,", vtxOutIdx++);
    vtxXfrAttrsPre +=
        "\n    out.mv_nrm = vec4f(mv_nrm, dot(mv_nrm, select(vec3f(0.0, 0.0, 1.0), -mv_pos, ubuf.proj[3].w == "
        "0.0)));";
  } else if (useNormalTarget || EnableNormalVisualization) {
    vtxOutAttrs += fmt::format("\n    @location({}) mv_nrm: vec3f,", vtxOutIdx++);
    vtxXfrAttrsPre += "\n    out.mv_nrm = mv_nrm;";
  }

  uniBufAttrs += "\n    proj: mat4x4f,";
  uniBufAttrs += fmt::format("\n    postex_mtx: array<mat3x4f, {}>,", MaxPnMtx + MaxTexMtx);
  uniBufAttrs += fmt::format("\n    nrm_mtx: array<mat3x4f, {}>,", MaxPnMtx);
  std::string fragmentFnPre;
  std::string fragmentFn;

  static std::array regName{"prev"sv, "tevreg0"sv, "tevreg1"sv, "tevreg2"sv};
  const auto emit_tev_stages = [&](std::string& out, const UnlitPass* unlitPass) -> TevNormalized {
    std::array<bool, MaxTevRegs> colorNormalized{};
    std::array<bool, MaxTevRegs> alphaNormalized{};
    const auto reg = [unlitPass](std::string value) {
      return unlitPass != nullptr ? replace_all(std::move(value), "ubuf.kcolor"sv, "upk"sv) : value;
    };
    const std::string_view label = unlitPass != nullptr ? "Unlit"sv : "TEV"sv;
    for (u32 idx = 0; idx < config.tevStageCount; ++idx) {
      const auto& stage = config.tevStages[idx];
      const auto color_arg = [&](GXTevColorArg arg) {
        auto value = reg(color_arg_reg(arg, idx, config, stage));
        if (tev_color_arg_is_normalized(arg, colorNormalized, alphaNormalized)) {
          return fmt::format("vec3f({})", value);
        }
        return fmt::format("tev_overflow_vec3f({})", value);
      };
      const bool alphaCompareColor = stage.alphaOp.op >= GX_TEV_COMP_R8_GT && stage.alphaOp.op <= GX_TEV_COMP_BGR24_EQ;
      // Packed alpha comparisons use color A/B from before this stage writes its color result.
      if (alphaCompareColor) {
        out += fmt::format("\n    let tev_cmp_a{0} = {1};\n    let tev_cmp_b{0} = {2};", idx,
                           color_arg(stage.colorPass.a), color_arg(stage.colorPass.b));
      }
      {
        std::string_view outReg = regName[stage.colorOp.outReg];
        const bool divided = unlitPass != nullptr && (unlitPass->scaleDivide >> idx & 1u) != 0;
        std::string op = tev_color_op(
            stage.colorOp.op, tev_bias(stage.colorOp.bias), tev_scale(divided ? GX_CS_SCALE_1 : stage.colorOp.scale),
            stage.colorOp.clamp, color_arg(stage.colorPass.a), color_arg(stage.colorPass.b),
            color_arg(stage.colorPass.c), reg(color_arg_reg(stage.colorPass.d, idx, config, stage)));
        out += fmt::format("\n    // {3} stage {2}\n    {0} = vec4f({1}, {0}.a);", outReg, op, idx, label);
        colorNormalized[stage.colorOp.outReg] = stage.colorOp.clamp;
      }
      {
        const auto alpha_arg = [&](GXTevAlphaArg arg) {
          auto value = reg(alpha_arg_reg(arg, idx, config, stage));
          if (tev_alpha_arg_is_normalized(arg, alphaNormalized)) {
            return value;
          }
          return fmt::format("tev_overflow_f32({})", value);
        };
        std::string_view outReg = regName[stage.alphaOp.outReg];
        std::string op = tev_alpha_op(
            stage.alphaOp.op, tev_bias(stage.alphaOp.bias), tev_scale(stage.alphaOp.scale), stage.alphaOp.clamp,
            alphaCompareColor ? fmt::format("tev_cmp_a{}", idx) : alpha_arg(stage.alphaPass.a),
            alphaCompareColor ? fmt::format("tev_cmp_b{}", idx) : alpha_arg(stage.alphaPass.b),
            alpha_arg(stage.alphaPass.c), reg(alpha_arg_reg(stage.alphaPass.d, idx, config, stage)));
        out += fmt::format("\n    {0}.a = {1};", outReg, op);
        alphaNormalized[stage.alphaOp.outReg] = stage.alphaOp.clamp;
      }
    }

    const auto& lastStage = config.tevStages[config.tevStageCount - 1];
    if (lastStage.colorOp.outReg != 0) {
      out += fmt::format("\n    prev = vec4f({0}.rgb, prev.a);", regName[lastStage.colorOp.outReg]);
    }
    if (lastStage.alphaOp.outReg != 0) {
      out += fmt::format("\n    prev.a = {0}.a;", regName[lastStage.alphaOp.outReg]);
    }
    return {colorNormalized[lastStage.colorOp.outReg], alphaNormalized[lastStage.alphaOp.outReg]};
  };
  const TevNormalized litNormalized = emit_tev_stages(fragmentFn, nullptr);
  const bool prevColorNormalized = litNormalized.color;
  const bool prevAlphaNormalized = litNormalized.alpha;
  const bool fwdEnv = uses_forward_env(config, targets);
  // Unclamped TEV colour wraps after fog, so AO would land before the wrap
  const bool fwdAo = fwdEnv && tev_colour_clamped(config);
  CHECK(tev_colour_clamped(config) == prevColorNormalized, "GX colour clamp disagrees with the last TEV stage");
  CHECK(!fwdEnv || targets.motion != UINT32_MAX, "forward environment needs the record reader");

  if (info.loadsTevReg.test(0)) {
    uniBufAttrs += "\n    tevprev: vec4f,";
    fragmentFnPre += "\n    var prev = ubuf.tevprev;";
  } else {
    fragmentFnPre += "\n    var prev: vec4f;";
  }
  for (int i = 1 /* Skip TEVPREV */; i < info.loadsTevReg.size(); ++i) {
    if (info.loadsTevReg.test(i)) {
      uniBufAttrs += fmt::format("\n    tevreg{}: vec4f,", i - 1);
      fragmentFnPre += fmt::format("\n    var tevreg{0} = ubuf.tevreg{0};", i - 1);
    } else if (info.writesTevReg.test(i)) {
      fragmentFnPre += fmt::format("\n    var tevreg{0}: vec4f;", i - 1);
    }
  }

  if (info.lightingEnabled) {
    uniBufAttrs += fmt::format(FMT_STRING(R"""(
    lights: array<Light, {}>,
    lightState0: u32,
    lightState1: u32,
    lightState0a: u32,
    lightState1a: u32,)"""),
                               GX::MaxLights);
    uniformPre +=
        "\n"
        "struct Light {\n"
        "    pos: vec3f,\n"
        "    dir: vec3f,\n"
        "    color: vec4f,\n"
        "    cos_att: vec3f,\n"
        "    dist_att: vec3f,\n"
        "};";
  }

  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }

    const auto& cc = config.colorChannels[i];
    const auto& cca = config.colorChannels[i + GX_ALPHA0];
    if (cc.lightingEnabled && cc.ambSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    cc{0}_amb: vec4f,", i);
    }
    if (cc.matSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    cc{0}_mat: vec4f,", i);
    }
    if (cca.lightingEnabled && cca.ambSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    cc{0}a_amb: vec4f,", i);
    }
    if (cca.matSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    cc{0}a_mat: vec4f,", i);
    }

    vtxOutAttrs += fmt::format("\n    @location({}) cc{}: vec4f,", vtxOutIdx++, i);
    vtxXfrAttrs += lighting_func(config, cc, i, false);
    vtxXfrAttrs += lighting_func(config, cca, i, true);
    fragmentFnPre += fmt::format("\n    var rast{0} = in.cc{0};", i);
  }
  for (int i = 0; i < info.sampledKColors.size(); ++i) {
    if (info.sampledKColors.test(i)) {
      uniBufAttrs += fmt::format("\n    kcolor{}: vec4f,", i);
    }
  }
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type == GX_TG_MTX3x4) {
      vtxOutAttrs += fmt::format("\n    @location({}) tex{}_uvw: vec3f,", vtxOutIdx++, i);
    } else {
      vtxOutAttrs += fmt::format("\n    @location({}) tex{}_uv: vec2f,", vtxOutIdx++, i);
    }
    if (is_emboss_texgen(tcg.type)) {
      // Emboss bump: offset the source texcoord by the light projected onto tangent/binormal
      const u32 lightIdx = tcg.type - GX_TG_BUMP0;
      vtxXfrAttrs += fmt::format(
          "\n    let bump_ldir{0} = normalize(ubuf.lights[{1}].pos - mv_pos);"
          "\n    let bump_tan{0} = vec4f(in_tangent, 0.0) * ubuf.nrm_mtx[in_pnmtxidx];"
          "\n    let bump_bin{0} = vec4f(in_binrm, 0.0) * ubuf.nrm_mtx[in_pnmtxidx];"
          "\n    out.tex{0}_uv = tc{2}_proj.xy + vec2f(dot(bump_ldir{0}, bump_tan{0}), dot(bump_ldir{0}, "
          "bump_bin{0}));",
          i, lightIdx, tcg.embossSrc);
      fragmentFnPre += fmt::format("\n    var tex{0}_uv = in.tex{0}_uv.xy;", i);
      continue;
    }
    if (tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = vec4f({}, 1.0, 1.0);", i,
                                 vtx_attr(config, GXAttr(GX_VA_TEX0 + (tcg.src - GX_TG_TEX0))));
    } else if (tcg.src >= GX_TG_TEXCOORD0 && tcg.src <= GX_TG_TEXCOORD6) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = tc{};", i, (tcg.src - GX_TG_TEXCOORD0));
    } else if (tcg.src == GX_TG_POS) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = vec4f({}, 1.0);", i, vtx_attr(config, GX_VA_POS));
    } else if (tcg.src == GX_TG_NRM) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = vec4f({}, 1.0);", i, vtx_attr(config, GX_VA_NRM));
    } else if (tcg.src == GX_TG_COLOR0) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = out.cc0;", i);
    } else if (tcg.src == GX_TG_COLOR1) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = out.cc1;", i);
    } else if (tcg.src == GX_TG_BINRM) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = vec4f({}, 1.0);", i, nbt_slice_local(NbtSlice::B));
    } else if (tcg.src == GX_TG_TANGENT) {
      vtxXfrAttrs += fmt::format("\n    var tc{} = vec4f({}, 1.0);", i, nbt_slice_local(NbtSlice::T));
    } else
      UNLIKELY FATAL("unhandled tcg src {}", underlying(tcg.src));
    if (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4) {
      if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) {
        vtxXfrAttrs += fmt::format("\n    var tc{0}_tmp = tc{0} * ubuf.postex_mtx[in_texmtxidx{0} / 3u];", i);
      } else if (tcg.mtx == GX_IDENTITY) {
        vtxXfrAttrs += fmt::format("\n    var tc{0}_tmp = tc{0}.xyz;", i);
      } else {
        u32 texMtxIdx = (tcg.mtx) / 3;
        vtxXfrAttrs += fmt::format("\n    var tc{0}_tmp = tc{0} * ubuf.postex_mtx[{1}];", i, texMtxIdx);
      }
      if (tcg.type == GX_TG_MTX2x4) {
        vtxXfrAttrs += fmt::format("\n    tc{0}_tmp.z = 1.0f;", i);
      }
    } else if (tcg.type == GX_TG_SRTG) {
      vtxXfrAttrs += fmt::format("\n    var tc{0}_tmp = vec3f(tc{0}.xy, 1.0f);", i);
    }
    if (tcg.normalize) {
      vtxXfrAttrs += fmt::format("\n    tc{0}_tmp = normalize(tc{0}_tmp);", i);
    }
    if (tcg.postMtx == GX_PTIDENTITY) {
      vtxXfrAttrs += fmt::format("\n    var tc{0}_proj = tc{0}_tmp;", i);
    } else {
      u32 postMtxIdx = (tcg.postMtx - GX_PTTEXMTX0) / 3;
      vtxXfrAttrs +=
          fmt::format("\n    var tc{0}_proj = vec4f(tc{0}_tmp.xyz, 1.0) * ubuf.postmtx[{1}];", i, postMtxIdx);
    }
    // Apply line/point tex offset
    if (config.lineMode == 3) {
      // GX_POINTS: offset S for right columns, T for bottom rows
      vtxXfrAttrs += fmt::format(
          "\n    if ((ubuf.line_texcoord_mask & (1u << {0})) != 0u) {{"
          "\n        if ((vidx & 1u) != 0u) {{ tc{0}_proj.x += ubuf.line_tex_offset; }}"
          "\n        if (vidx >= 2u) {{ tc{0}_proj.y += ubuf.line_tex_offset; }}"
          "\n    }}",
          i);
    } else if (config.lineMode != 0) {
      // GX_LINES / GX_LINESTRIP: offset one axis for perpendicular side
      vtxXfrAttrs += fmt::format(
          "\n    if ((ubuf.line_texcoord_mask & (1u << {0})) != 0u && (vidx & 1u) != 0u) {{"
          "\n        tc{0}_proj.y += ubuf.line_tex_offset;"
          "\n    }}",
          i);
    }
    if (tcg.type == GX_TG_MTX3x4) {
      vtxXfrAttrs += fmt::format("\n    out.tex{0}_uvw = tc{0}_proj.xyz;", i);
      fragmentFnPre += fmt::format("\n    var tex{0}_uv = in.tex{0}_uvw.xy / in.tex{0}_uvw.z;", i);
    } else {
      vtxXfrAttrs += fmt::format("\n    out.tex{0}_uv = tc{0}_proj.xy;", i);
      fragmentFnPre += fmt::format("\n    var tex{0}_uv = in.tex{0}_uv.xy;", i);
    }
  }
  // Multiple TEV stages may reference the same indirect stage,
  // so we sample each indirect texture only once.
  const auto ind_scale = [](const GXIndTexScale s) -> std::string_view {
    switch (s) {
    case GX_ITS_1:
      return "1.0"sv;
    case GX_ITS_2:
      return "(1.0 / 2.0)"sv;
    case GX_ITS_4:
      return "(1.0 / 4.0)"sv;
    case GX_ITS_8:
      return "(1.0 / 8.0)"sv;
    case GX_ITS_16:
      return "(1.0 / 16.0)"sv;
    case GX_ITS_32:
      return "(1.0 / 32.0)"sv;
    case GX_ITS_64:
      return "(1.0 / 64.0)"sv;
    case GX_ITS_128:
      return "(1.0 / 128.0)"sv;
    case GX_ITS_256:
      return "(1.0 / 256.0)"sv;
    default:
      FATAL("unhandled indirect scale {}", underlying(s));
    }
  };
  for (int i = 0; i < info.usedIndStages.size(); ++i) {
    if (!info.usedIndStages.test(i)) {
      continue;
    }
    const auto& indStage = config.indStages[i];
    const u32 texCoordId = underlying(indStage.texCoordId);
    const u32 texMapId = underlying(indStage.texMapId);
    // GX applies the SU texture-coordinate scale before the indirect stage scale.
    // The shader carries normalized UVs, so convert that texel-space result back
    // into normalized coordinates for the indirect texture sample.
    const auto scaleExpr =
        fmt::format("tex{0}_uv * ubuf.texcoord_scale[{0}].xy * vec2f({1}, {2}) / ubuf.tex{3}_size_bias.xy", texCoordId,
                    ind_scale(indStage.scaleS), ind_scale(indStage.scaleT), texMapId);
    fragmentFnPre += fmt::format(
        "\n    // Indirect stage {0}"
        "\n    var t_IndTexCoord{0} = 255.0 * textureSampleBias(tex{1}, tex{1}_samp, {2}, "
        "ubuf.tex{1}_size_bias.z).abg;",
        i, texMapId, scaleExpr);
  }
  if (info.usedIndStages.any()) {
    fragmentFnPre += "\n    var t_TexCoord = vec2f(0.0);";
  }
  std::bitset<MaxTextures * MaxTexCoord> hexSites;
  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    const bool needsIndirectCoord = stage.indTexMtxId != GX_ITM_OFF;
    const bool hasIndirectStage = stage.indTexStage < config.numIndStages;
    const bool needsTevTexCoord =
        needsIndirectCoord || stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
    const bool needsTextureSample = uses_texture_sample(stage);
    if (!needsTevTexCoord && !needsTextureSample) {
      continue;
    }
    const bool hasBaseTexCoord = stage.texCoordId != GX_TEXCOORD_NULL;
    const bool hasBaseTexture = stage.texMapId != GX_TEXMAP_NULL;
    const bool hasBaseCoord = hasBaseTexCoord && hasBaseTexture;
    std::string uvIn;
    if (needsTevTexCoord) {
      fragmentFnPre += fmt::format("\n    // TEV stage {} indirect", i);

      // Apply indirect texture matrix (produces a texel-space offset)
      std::string indirectOffsetTexel;
      if (needsIndirectCoord && hasIndirectStage) {
        std::string_view fmtShift;
        switch (stage.indTexFormat) {
        case GX_ITF_8:
          break;
        case GX_ITF_5:
          fmtShift = " / 8.0"sv;
          break;
        case GX_ITF_4:
          fmtShift = " / 16.0"sv;
          break;
        case GX_ITF_3:
          fmtShift = " / 32.0"sv;
          break;
        default:
          FATAL("unhandled indirect format {}", underlying(stage.indTexFormat));
        }
        if (fmtShift.empty()) {
          fragmentFnPre += fmt::format("\n    var ind{0}_coord = t_IndTexCoord{1};", i, underlying(stage.indTexStage));
        } else {
          fragmentFnPre += fmt::format("\n    var ind{0}_coord = floor(t_IndTexCoord{1}{2});", i,
                                       underlying(stage.indTexStage), fmtShift);
        }

        if (stage.indTexBiasSel != GX_ITB_NONE) {
          auto bias = stage.indTexFormat == GX_ITF_8 ? "-128.0"sv : "1.0"sv;
          auto biasS = "0.0"sv, biasT = "0.0"sv, biasU = "0.0"sv;
          if (stage.indTexBiasSel == GX_ITB_S || stage.indTexBiasSel == GX_ITB_ST || stage.indTexBiasSel == GX_ITB_SU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasS = "1.0"sv;
          }
          if (stage.indTexBiasSel == GX_ITB_T || stage.indTexBiasSel == GX_ITB_ST || stage.indTexBiasSel == GX_ITB_TU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasT = "1.0"sv;
          }
          if (stage.indTexBiasSel == GX_ITB_U || stage.indTexBiasSel == GX_ITB_SU || stage.indTexBiasSel == GX_ITB_TU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasU = "1.0"sv;
          }
          fragmentFnPre += fmt::format("\n    ind{0}_coord = ind{0}_coord + vec3f({1}, {2}, {3}) * {4};", i, biasS,
                                       biasT, biasU, bias);
        }

        if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
          // Static 2x3 matrix: dot(mat_row, vec3(S,T,U)) * scale
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_0;
          fragmentFnPre += fmt::format(
              "\n    let ind{0}_c0 = ubuf.ind_mtx[{1}][0];"
              "\n    let ind{0}_c1 = ubuf.ind_mtx[{1}][1];",
              i, mtxIdx);
          indirectOffsetTexel = fmt::format(
              "vec2f("
              "dot(vec3f(ind{0}_c0.xz, ind{0}_c1.x), ind{0}_coord), "
              "dot(vec3f(ind{0}_c0.yw, ind{0}_c1.y), ind{0}_coord)"
              ") * ind{0}_c1.z",
              i);
        } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2 && hasBaseCoord) {
          // Dynamic S: result = scaled texcoord * ind_coord.x * scale / 256
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_S0;
          u32 regTexCoord = underlying(stage.texCoordId);
          indirectOffsetTexel = fmt::format(
              "tex{1}_uv * ubuf.texcoord_scale[{1}].xy * ind{0}_coord.x"
              " * ubuf.ind_mtx[{2}][1][2] / 256.0",
              i, regTexCoord, mtxIdx);
        } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2 && hasBaseCoord) {
          // Dynamic T: result = scaled texcoord * ind_coord.y * scale / 256
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_T0;
          u32 regTexCoord = underlying(stage.texCoordId);
          indirectOffsetTexel = fmt::format(
              "tex{1}_uv * ubuf.texcoord_scale[{1}].xy * ind{0}_coord.y"
              " * ubuf.ind_mtx[{2}][1][2] / 256.0",
              i, regTexCoord, mtxIdx);
        }
      }

      // Don't convert to/from texel space if we can avoid it
      const bool useSimpleCoords = stage.indTexMtxId == GX_ITM_OFF && !stage.indTexAddPrev;

      // Wrap base coord and combine with the indirect translation.
      auto wrap_comp = [](GXIndTexWrap wrap, std::string&& coord) -> std::string {
        switch (wrap) {
        case GX_ITW_OFF:
          return std::move(coord);
        case GX_ITW_256:
          return fmt::format("({} % 256.0)", coord);
        case GX_ITW_128:
          return fmt::format("({} % 128.0)", coord);
        case GX_ITW_64:
          return fmt::format("({} % 64.0)", coord);
        case GX_ITW_32:
          return fmt::format("({} % 32.0)", coord);
        case GX_ITW_16:
          return fmt::format("({} % 16.0)", coord);
        case GX_ITW_0:
          return "0.0";
        default:
          FATAL("unhandled indirect wrap {}", underlying(wrap));
        }
      };
      std::string baseCoordExpr;
      if (hasBaseCoord) {
        u32 texCoordId = underlying(stage.texCoordId);
        if (useSimpleCoords) {
          baseCoordExpr = fmt::format("tex{}_uv", texCoordId);
        } else {
          fragmentFnPre +=
              fmt::format("\n    var ind{0}_texel = tex{1}_uv * ubuf.texcoord_scale[{1}].xy;", i, texCoordId);
          baseCoordExpr = fmt::format("ind{}_texel", i);
        }
      }
      std::string wrappedExpr = baseCoordExpr;
      if (!baseCoordExpr.empty() && (stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF)) {
        wrappedExpr = fmt::format("vec2f({}, {})", wrap_comp(stage.indTexWrapS, fmt::format("{}.x", baseCoordExpr)),
                                  wrap_comp(stage.indTexWrapT, fmt::format("{}.y", baseCoordExpr)));
      }

      std::string finalCoord;
      if (!wrappedExpr.empty() && !indirectOffsetTexel.empty()) {
        finalCoord = fmt::format("{} + ({})", wrappedExpr, indirectOffsetTexel);
      } else if (!wrappedExpr.empty()) {
        finalCoord = wrappedExpr;
      } else {
        finalCoord = indirectOffsetTexel;
      }

      if (info.usedIndStages.any() && !finalCoord.empty()) {
        if (stage.indTexAddPrev) {
          fragmentFnPre += fmt::format("\n    t_TexCoord += {};", finalCoord);
        } else {
          fragmentFnPre += fmt::format("\n    t_TexCoord = {};", finalCoord);
        }

        if (needsTextureSample && hasBaseTexture) {
          u32 texMapId = underlying(stage.texMapId);
          if (useSimpleCoords) {
            fragmentFnPre += fmt::format("\n    var ind{0}_uv = t_TexCoord;", i);
          } else {
            fragmentFnPre += fmt::format("\n    var ind{0}_uv = t_TexCoord / ubuf.tex{1}_size_bias.xy;", i, texMapId);
          }
          uvIn = fmt::format("ind{0}_uv", i);
        }
      }
    }
    if (!needsTextureSample) {
      continue;
    }

    CHECK(stage.texMapId != GX_TEXMAP_NULL, "unmapped texture for stage {}", i);
    CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "unmapped texcoord for stage {}", i);
    const bool direct = uvIn.empty();
    if (direct) {
      // No indirect texturing
      uvIn = fmt::format("tex{0}_uv", underlying(stage.texCoordId));
    }
    if (direct && info.stochasticTextures.test(stage.texMapId)) {
      const u32 map = underlying(stage.texMapId);
      const u32 coord = underlying(stage.texCoordId);
      if (!hexSites.test(map * MaxTexCoord + coord)) {
        hexSites.set(map * MaxTexCoord + coord);
        fragmentFnPre += hex_site(map, coord);
      }
      fragmentFnPre += fmt::format("\n    var sampled{} = hex{}_{};", i, map, coord);
      continue;
    }
    fragmentFnPre +=
        fmt::format("\n    var sampled{0} = textureSampleBias(tex{1}, tex{1}_samp, {2}, ubuf.tex{1}_size_bias.z);", i,
                    underlying(stage.texMapId), uvIn);
  }
  if (companionText) {
    fragmentFnPre += fmt::format(
        "\n    let cmp_alb = textureSampleBias(ctex_alb, tex{0}_samp, tex{1}_uv, ubuf.tex{0}_size_bias.z);"
        "\n    let cmp_nrm = textureSampleBias(ctex_nrm, tex{0}_samp, tex{1}_uv, ubuf.tex{0}_size_bias.z);"
        "\n    let cmp_orm = textureSampleBias(ctex_orm, tex{0}_samp, tex{1}_uv, ubuf.tex{0}_size_bias.z);{2}"
        "\n    let cmp_p = gx_view_pos(in.mo_cur);"
        "\n    let cmp_dp1 = dpdx(cmp_p);"
        "\n    let cmp_dp2 = dpdy(cmp_p);"
        "\n    let cmp_duv1 = dpdx(tex{1}_uv);"
        "\n    let cmp_duv2 = dpdy(tex{1}_uv);",
        companion.map, companion.coord,
        emitText ? fmt::format("\n    let cmp_emit = textureSampleBias(ctex_emit, tex{0}_samp, tex{1}_uv, "
                               "ubuf.tex{0}_size_bias.z);",
                               companion.map, companion.coord)
                 : std::string{});
    uniformPre += CompanionWgsl;
  }
  if (info.usesPTTexMtx.any()) {
    uniBufAttrs += fmt::format("\n    postmtx: array<mat3x4f, {}>,", MaxPTTexMtx);
  }
  if (fwdAo) {
    // About 0.4% and 2% of the view distance; f16 keys stay near 0.05%
    constexpr float ForwardAoGapStart = 0.004f, ForwardAoGapEnd = 0.02f;
    uniformPre += fmt::format(
        "\n// G and B hold the f16 bits of the receiver's depth; reversed Z, so 0 means nothing behind"
        "\nfn fwd_screen_ao(t: vec4f, z: f32) -> f32 {{"
        "\n    if (t.r >= 1.0) {{"
        "\n        return 1.0;"
        "\n    }}"
        "\n    let key = u32(round(t.g * 255.0)) | (u32(round(t.b * 255.0)) << 8u);"
        "\n    let zr = unpack2x16float(key).x;"
        "\n    // Relative depth gap, so only draws lying on the receiver take its AO"
        "\n    let gap = abs(z - zr) / max(zr, 1e-7);"
        "\n    let w = select(0.0, 1.0 - smoothstep({:.3f}, {:.3f}, gap), zr > 0.0);"
        "\n    return 1.0 - w * (1.0 - t.r);"
        "\n}}",
        ForwardAoGapStart, ForwardAoGapEnd);
    fragmentFn += fmt::format(
        "\n    // Screen factors darken the surface before fog, so fog keeps its colour"
        "\n    if ((gx_record_word({}u) & {}u) != 0u) {{"
        "\n        let fwd_px = min(vec2u(in.pos.xy), textureDimensions(fwd_ao) - vec2u(1u));"
        "\n        prev = vec4f(prev.rgb * fwd_screen_ao(textureLoad(fwd_ao, fwd_px, 0), in.pos.z), prev.a);"
        "\n    }}",
        render_ext::DrawRecordFlagsWord, render_ext::RecordFlagScreenAo);
  }
  if (info.usesFog) {
    uniformPre +=
        "\n"
        "struct Fog {\n"
        "    color: vec4f,\n"
        "    a: f32,\n"
        "    b: f32,\n"
        "    c: f32,\n"
        "    range_center: f32,\n"
        "    range_k: array<vec4f, 3>,\n"
        "}";
    uniBufAttrs += "\n    fog: Fog,";

    const std::string_view fogDepth = UseReversedZ ? "(1.0 - in.pos.z)" : "in.pos.z";
    if ((config.fogType & 0x08) != 0) {
      fragmentFn += fmt::format("\n    // Orthographic fog\n    var fogBase = ubuf.fog.a * {};", fogDepth);
    } else {
      fragmentFn +=
          fmt::format("\n    // Perspective fog\n    var fogBase = ubuf.fog.a / (ubuf.fog.b - {});", fogDepth);
    }
    if (config.fogRangeEnabled) {
      fragmentFn += "\n        fogBase *= bitcast<f32>(abuf[imm.fog_range_base + u32(in.pos.x)]);";
    }
    fragmentFn += "\n    var fogF = clamp(fogBase - ubuf.fog.c, 0.0, 1.0);";
    switch (config.fogType) {
      DEFAULT_FATAL("invalid fog type {}", config.fogType);
    case GX_FOG_PERSP_LIN:
    case GX_FOG_ORTHO_LIN:
      fragmentFn += "\n    var fogZ = fogF;";
      break;
    case GX_FOG_PERSP_EXP:
    case GX_FOG_ORTHO_EXP:
      fragmentFn += "\n    var fogZ = 1.0 - exp2(-8.0 * fogF);";
      break;
    case GX_FOG_PERSP_EXP2:
    case GX_FOG_ORTHO_EXP2:
      fragmentFn += "\n    var fogZ = 1.0 - exp2(-8.0 * fogF * fogF);";
      break;
    case GX_FOG_PERSP_REVEXP:
    case GX_FOG_ORTHO_REVEXP:
      fragmentFn += "\n    var fogZ = exp2(-8.0 * (1.0 - fogF));";
      break;
    case GX_FOG_PERSP_REVEXP2:
    case GX_FOG_ORTHO_REVEXP2:
      fragmentFn +=
          "\n    fogF = 1.0 - fogF;"
          "\n    var fogZ = exp2(-8.0 * fogF * fogF);";
      break;
    }
    fragmentFn += "\n    prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fogZ, 0.0, 1.0)), prev.a);";
  }
  uniBufAttrs += fmt::format("\n    texcoord_scale: array<vec4f, {}>,", MaxTexCoord);
  if (info.usedIndTexMtxs.any()) {
    uniBufAttrs += fmt::format("\n    ind_mtx: array<mat2x4f, {}>,", MaxIndTexMtxs);
  }
  for (int i = 0; i < info.sampledTextures.size(); ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    uniBufAttrs += fmt::format("\n    tex{}_size_bias: vec4f,", i);
    texBindings += fmt::format(
        "\n@group(2) @binding({1})\n"
        "var tex{0}: texture_2d<f32>;\n"
        "@group(2) @binding({2})\n"
        "var tex{0}_samp: sampler;",
        i, i * 2, i * 2 + 1);
  }
  if (fwdEnv) {
    texBindings += "\n@group(3) @binding(0)\nvar fwd_ao: texture_2d<f32>;";
  }
  if (info.stochasticTextures.any()) {
    for (u32 i = 0; i < MaxTextures; ++i) {
      if (info.stochasticTextures.test(i)) {
        uniBufAttrs += fmt::format("\n    tex{}_hex: vec4f,", i);
      }
    }
    append_hex_module(uniformPre, static_cast<uint32_t>(info.stochasticTextures.to_ulong()));
  }
  if (companionText) {
    uniBufAttrs += "\n    mat0: vec4f,";
    texBindings += fmt::format(
        "\n@group(2) @binding({0})\nvar ctex_alb: texture_2d<f32>;"
        "\n@group(2) @binding({1})\nvar ctex_nrm: texture_2d<f32>;"
        "\n@group(2) @binding({2})\nvar ctex_orm: texture_2d<f32>;",
        MaxTextures * 2, MaxTextures * 2 + 1, MaxTextures * 2 + 2);
    if (emitText) {
      texBindings += fmt::format("\n@group(2) @binding({})\nvar ctex_emit: texture_2d<f32>;", MaxTextures * 2 + 3);
    }
  }
  if (!prevColorNormalized && !prevAlphaNormalized) {
    fragmentFn += "\n    prev = tev_overflow_vec4f(prev);";
  } else if (!prevColorNormalized) {
    fragmentFn += "\n    prev = vec4f(tev_overflow_vec3f(prev.rgb), prev.a);";
  } else if (!prevAlphaNormalized) {
    fragmentFn += "\n    prev.a = tev_overflow_f32(prev.a);";
  }
  if (config.alphaCompare) {
    const auto comp0 = alpha_compare(config.alphaCompare.comp0, config.alphaCompare.ref0);
    const auto comp1 = alpha_compare(config.alphaCompare.comp1, config.alphaCompare.ref1);
    AlphaCompareExpr pass;
    switch (config.alphaCompare.op) {
      DEFAULT_FATAL("invalid alpha compare op {}", underlying(config.alphaCompare.op));
    case GX_AOP_AND:
      pass = alpha_compare_and(comp0, comp1);
      break;
    case GX_AOP_OR:
      pass = alpha_compare_or(comp0, comp1);
      break;
    case GX_AOP_XOR:
      pass = alpha_compare_xor(comp0, comp1);
      break;
    case GX_AOP_XNOR:
      pass = alpha_compare_xnor(comp0, comp1);
      break;
    }
    const auto discard = alpha_compare_not(pass);
    if (discard.constant == 1) {
      fragmentFn += "\n    // Alpha compare\n    discard;";
    } else if (discard.constant != 0) {
      fragmentFn +=
          "\n    // Alpha compare"
          "\n    let alphaCompare = u32(round(clamp(prev.a, 0.0, 1.0) * 255.0));";
      fragmentFn += fmt::format("\n    if ({}) {{ discard; }}", discard.expr);
    }
  }
  if constexpr (EnableNormalVisualization) {
    fragmentFn += "\n    prev = vec4f(in.mv_nrm.xyz, prev.a);";
  }
  if (targets.reactive != UINT32_MAX) {
    // Coverage is the TEV alpha that blends, not the destination alpha the Replace override writes
    fragmentFn += "\n    let reactive_alpha = prev.a;";
  }

  if (dstAlphaMode == DstAlphaMode::Replace) {
    // GX destination alpha is set via the blend constant
    fragmentFn += "\n    prev.a = 1.0;";
  }

  if (targets.motion != UINT32_MAX) {
    CHECK(dstAlphaMode != DstAlphaMode::DualSource, "GX motion output with dual-source blending");
    uniformPre +=
        "\nfn gx_record_word(k: u32) -> u32 { return select(0u, load_word(&abuf, imm._pad + k), imm._pad != 0u); }"
        "\nfn gx_abuf_vec4(word: u32) -> vec4f { return bitcast<vec4f>(vec4u(load_word(&abuf, word), "
        "load_word(&abuf, word + 1u), load_word(&abuf, word + 2u), load_word(&abuf, word + 3u))); }"
        "\nfn gx_abuf_mtx(word: u32) -> mat3x4f { return mat3x4f(gx_abuf_vec4(word), gx_abuf_vec4(word + 4u), "
        "gx_abuf_vec4(word + 8u)); }";
    vtxOutAttrs += fmt::format("\n    @location({}) mo_cur: vec3f,", vtxOutIdx++);
    vtxOutAttrs += fmt::format("\n    @location({}) mo_prev: vec3f,", vtxOutIdx++);
    const std::string paletteIndex =
        config.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE
            ? fmt::format("min(in_pnmtxidx, {}u) * 12u", render_ext::MotionPaletteSlots - 1)
            : "0u"s;
    const std::string_view lineOffset =
        config.lineMode != 0 ? "\n        mo_prev = vec4f(mo_prev.xy + offset_ndc * mo_prev.w, mo_prev.zw);"sv : ""sv;
    const u32 frameWord = render_ext::FrameMotionWord;
    const u32 prevProjWord = render_ext::FrameMotionPrevProjWord;
    vtxXfrAttrs += fmt::format(
        "\n    let mo_cur = out.pos;"
        "\n    let mo_flags = gx_record_word({0}u);"
        "\n    let mo_source = mo_flags & {1}u;"
        "\n    var mo_prev = mo_cur;"
        "\n    if (mo_source != 0u) {{"
        "\n        let mo_frame = gx_record_word(0u) + {2}u;"
        "\n        var mo_mv: vec3f;"
        "\n        if (mo_source == {3}u) {{"
        "\n            mo_mv = vec4f(mv_pos, 1.0) * gx_abuf_mtx(mo_frame + {4}u);"
        "\n        }} else {{"
        "\n            mo_mv = vec4f(in_pos, 1.0) * gx_abuf_mtx(gx_record_word({5}u) + {6});"
        "\n        }}"
        "\n        let mo_v = vec4f(mo_mv, 1.0);"
        "\n        if ((mo_flags & {7}u) != 0u) {{"
        "\n            mo_prev = vec4f(dot(mo_v, gx_abuf_vec4(mo_frame + {8}u)), "
        "dot(mo_v, gx_abuf_vec4(mo_frame + {9}u)), 0.0, dot(mo_v, gx_abuf_vec4(mo_frame + {10}u)));"
        "\n        }} else {{"
        "\n            mo_prev = mo_v * ubuf.proj;"
        "\n        }}{11}"
        "\n    }}"
        "\n    if ((mo_flags & {12}u) != 0u) {{"
        "\n        let mo_jit = gx_abuf_vec4(gx_record_word(0u) + {13}u).xy;"
        "\n        out.pos = vec4f(out.pos.xy + mo_jit * vec2f(2.0, -2.0) / ubuf.render_viewport_size * out.pos.w, "
        "out.pos.zw);"
        "\n    }}"
        "\n    out.mo_cur = mo_cur.xyw;"
        "\n    out.mo_prev = mo_prev.xyw;",
        render_ext::DrawRecordFlagsWord, render_ext::MotionSourceMask, frameWord, render_ext::MotionSourceCamera,
        render_ext::FrameMotionReprojectWord, render_ext::DrawRecordPrevMtxWord, paletteIndex,
        render_ext::RecordFlagPrevProjFrame, prevProjWord, prevProjWord + 4, prevProjWord + 8, lineOffset,
        render_ext::RecordFlagJitter, frameWord + render_ext::FrameMotionJitterWord);
  }

  const bool unlitText = uses_unlit(config, info, targets);
  const unlit::Plan unlitPlan = unlitText ? unlit::plan(config) : unlit::Plan{};
  if (unlitText) {
    for (u32 i = 0; i < MaxColorChannels / 2; ++i) {
      if ((info.unlitRawVtx >> i & 1u) != 0) {
        vtxOutAttrs += fmt::format("\n    @location({}) ccv{}: vec3f,", vtxOutIdx++, i);
        vtxXfrAttrs +=
            fmt::format("\n    out.ccv{} = {}.rgb;", i, vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i)));
      }
    }
    uniformPre +=
        "\nfn gx_tev_pol(v: vec4f, reg: u32, loc: u32) -> vec4f {"
        "\n    if (loc == 0u) {"
        "\n        return v;"
        "\n    }"
        "\n    let sel = (vec4u(load_word(&abuf, loc) >> (4u * reg)) & vec4u(1u, 2u, 4u, 8u)) != vec4u(0u);"
        "\n    return select(v, unpack4x8unorm(load_word(&abuf, loc + 1u + reg)), sel);"
        "\n}";
    append_surface_pack(uniformPre);
  }

  CHECK(vtxOutIdx <= info.varyings, "GX shader emits {} inter-stage locations, planned {}", vtxOutIdx, info.varyings);
  const std::string_view nrmIn = backFaceText ? "in.mv_nrm.xyz"sv : "in.mv_nrm"sv;
  const std::string flipLet =
      backFaceText ? fmt::format(
                         "\n    let uflip = !gx_ff && in.mv_nrm.w < 0.0 && (gx_record_word({}u) & {:#x}u) "
                         "!= 0u;",
                         render_ext::DrawRecordFlagsWord, render_ext::RecordFlagBackFaceFlip)
                   : std::string{};
  const auto flipped = [backFaceText](std::string_view n) {
    return backFaceText ? fmt::format("select({0}, -{0}, uflip)", n) : std::string{n};
  };
  std::string fragmentOutput;
  std::string_view fragmentOutputType = "@location(0) vec4f"sv;
  std::string fragmentReturn = "\n    return prev;"s;
  std::vector<OutputMember> members;
  if (nrmTarget != UINT32_MAX) {
    const std::string_view alpha =
        receiverNormal ? "gbuffer_rx_alpha(imm._pad, nrm_len_sq)"sv : "select(0.0, 1.0, nrm_len_sq > 1e-10)"sv;
    std::string assignment = useNormalTarget
                                 ? fmt::format(
                                       "\n    let nrm_len_sq = dot({0}, {0});"
                                       "\n    let unit_nrm = select(vec3f(0.0), normalize({0}), nrm_len_sq > 1e-10);{1}"
                                       "\n    out.normal = vec4f({2} * 0.5 + 0.5, {3});",
                                       nrmIn, flipLet, flipped("unit_nrm"), alpha)
                                 : "\n    out.normal = vec4f(0.5, 0.5, 0.5, 0.0);"s;
    if (useNormalTarget && companionText) {
      assignment = fmt::format(
          "\n    let nrm_len_sq = dot({4}, {4});"
          "\n    let unit_nrm = select(vec3f(0.0), normalize({4}), nrm_len_sq > 1e-10);{1}"
          "\n    let legacy_normal = vec4f({2} * 0.5 + 0.5, {0});"
          "\n    // Perspective only: mo_cur carries clip xyw, so an orthographic view depth is not recoverable"
          "\n    let use_mapped = ubuf.mat0.y > 0.0 && (u32(ubuf.mat0.x) & 6u) != 0u && ubuf.proj[3].w == 0.0 && "
          "nrm_len_sq > 1e-10;"
          "\n    let mapped_nrm = gx_perturb(unit_nrm, cmp_dp1, cmp_dp2, cmp_duv1, cmp_duv2, "
          "gx_companion_tangent(cmp_nrm, ubuf.mat0));"
          "\n    out.normal = select(legacy_normal, vec4f({3} * 0.5 + 0.5, {0}), use_mapped);",
          alpha, flipLet, flipped("unit_nrm"), flipped("mapped_nrm"), nrmIn);
    }
    if (useNormalTarget && receiverNormal) {
      uniformPre += "\n" + receiver_alpha_wgsl();
    }
    members.push_back({nrmTarget, fmt::format("    @location({}) normal: vec4f,\n", nrmTarget), std::move(assignment)});
  }
  if (targets.debugId != UINT32_MAX) {
    members.push_back({targets.debugId, fmt::format("    @location({}) debug_id: u32,\n", targets.debugId),
                       "\n    out.debug_id = imm._pad;"s});
  }
  if (targets.motion != UINT32_MAX) {
    members.push_back(
        {targets.motion, fmt::format("    @location({}) motion: vec2f,\n", targets.motion),
         fmt::format("\n    let mo_w = in.mo_prev.z;"
                     "\n    let mo_d = in.mo_cur.xy / in.mo_cur.z - in.mo_prev.xy / mo_w;"
                     "\n    out.motion = select(vec2f({0:.1f}), mo_d, mo_w > 1e-5 && all(abs(mo_d) < vec2f({0:.1f})));",
                     render_ext::MotionInvalidNdc)});
  }
  if (targets.reactive != UINT32_MAX) {
    members.push_back(
        {targets.reactive, fmt::format("    @location({}) reactive: vec4f,\n", targets.reactive),
         fmt::format("\n    let reactive_flags = select(0u, load_word(&abuf, imm._pad + {}u), imm._pad != 0u);"
                     "\n    out.reactive = vec4f(select(0.0, saturate(reactive_alpha), (reactive_flags & {}u) != 0u));",
                     render_ext::DrawRecordFlagsWord, render_ext::RecordFlagReactive)});
  }
  if (targets.fog != UINT32_MAX && config.ext.output != 0) {
    const u32 lock = config.fogType != GX_FOG_PERSP_LIN ? 1 : 0;
    members.push_back({targets.fog, fmt::format("    @location({}) fog: vec4u,\n", targets.fog),
                       info.usesFog
                           ? fmt::format("\n    out.fog = vec4u(vec3u(round(ubuf.fog.color.rgb * 255.0)), "
                                         "(min(u32(round(clamp(fogZ, 0.0, 1.0) * 127.0)), 127u) << 1u) | {}u);",
                                         lock)
                           : "\n    out.fog = vec4u(0u);"s});
  }
  if (uses_surface(targets)) {
    std::string assignment = "\n    out.surface = vec4u(0u);"s;
    if (unlitText) {
      std::string inner = fmt::format("\n    let upol = gx_record_word({}u);", render_ext::DrawRecordTevPolicyWord);
      if (unlitPlan.rewrite[0] == unlit::RewriteVertex || unlitPlan.rewrite[1] == unlit::RewriteVertex) {
        inner += fmt::format("\n    let uvc = (gx_record_word({}u) >> {}u) & 3u;", render_ext::DrawRecordFlagsWord,
                             render_ext::RecordVertexColorShift);
      }
      inner += info.loadsTevReg.test(0) ? "\n    var prev = ubuf.tevprev;"sv : "\n    var prev: vec4f;"sv;
      for (int i = 1; i < info.loadsTevReg.size(); ++i) {
        if (info.loadsTevReg.test(i)) {
          inner += fmt::format("\n    var tevreg{0} = gx_tev_pol(ubuf.tevreg{0}, {0}u, upol);", i - 1);
        } else if (info.writesTevReg.test(i)) {
          inner += fmt::format("\n    var tevreg{0}: vec4f;", i - 1);
        }
      }
      for (int i = 0; i < info.sampledKColors.size(); ++i) {
        if (info.sampledKColors.test(i)) {
          inner += fmt::format("\n    let upk{0} = gx_tev_pol(ubuf.kcolor{0}, {1}u, upol);", i, 3 + i);
        }
      }
      for (u32 i = 0; i < MaxColorChannels / 2; ++i) {
        if (!info.sampledColorChannels.test(i)) {
          continue;
        }
        switch (unlitPlan.rewrite[i]) {
        case unlit::RewriteMaterial:
          inner += fmt::format("\n    var rast{0} = vec4f(ubuf.cc{0}_mat.rgb, in.cc{0}.a);", i);
          break;
        case unlit::RewriteVertex:
          inner +=
              (unlitPlan.rawVtx >> i & 1u) != 0
                  ? fmt::format("\n    var rast{0} = vec4f(select(in.ccv{0}, vec3f(1.0), uvc != 0u), in.cc{0}.a);", i)
                  : fmt::format("\n    var rast{0} = vec4f(select(vec3f(0.0), vec3f(1.0), uvc != 0u), in.cc{0}.a);", i);
          break;
        case unlit::RewriteZero:
          inner += fmt::format("\n    var rast{0} = vec4f(vec3f(0.0), in.cc{0}.a);", i);
          break;
        default:
          break;
        }
      }
      for (u32 i = 0; i < config.tevStageCount; ++i) {
        const auto& stage = config.tevStages[i];
        if (!uses_texture_sample(stage) || stage.texMapId > GX_TEXMAP7) {
          continue;
        }
        const u32 map = 1u << stage.texMapId;
        if (((unlitPlan.neutralOne | unlitPlan.neutralZero) & map) != 0) {
          inner +=
              fmt::format("\n    let sampled{} = vec4f({});", i, (unlitPlan.neutralOne & map) != 0 ? "1.0" : "0.0");
        } else if (companionText && stage.texMapId == companion.map && stage.texCoordId == companion.coord &&
                   direct_stage(stage) && !info.stochasticTextures.test(stage.texMapId)) {
          // The lane is uniform, so the branch keeps the Oklab work off every draw without a delight map
          inner += fmt::format(
              "\n    var delit{0} = sampled{0};"
              "\n    if ((u32(ubuf.mat0.x) & {1}u) != 0u) {{"
              "\n        delit{0} = vec4f(gx_companion_delight(sampled{0}.rgb, cmp_alb.r), sampled{0}.a);"
              "\n    }}"
              "\n    let sampled{0} = select(delit{0}, vec4f(gx_linear_to_srgb(cmp_alb.rgb), sampled{0}.a), "
              "(u32(ubuf.mat0.x) & {2}u) != 0u);",
              i, render_ext::LaneDelight, render_ext::LaneAlbedo);
        }
      }
      const UnlitPass pass{.scaleDivide = unlitPlan.scaleDivide};
      const TevNormalized unlitNormalized = emit_tev_stages(inner, &pass);
      inner +=
          unlitNormalized.color ? "\n    unlit_rgb = prev.rgb;"sv : "\n    unlit_rgb = tev_overflow_vec3f(prev.rgb);"sv;
      if (unlit::ambient_of(config, info).source == unlit::AmbientSource::Unknown) {
        render_ext::g_stats.pbrAmbientUnknown.fetch_add(1, std::memory_order_relaxed);
      }
      // Class, flags, category, ambient and emission sit in the draw's table entry, so the record carries a slot
      assignment = fmt::format(
          "\n    // Unlit TEV (SP1 rules v1): no fog, lit alpha kept, only unlit_rgb leaves the block"
          "\n    var unlit_rgb: vec3f;"
          "\n    {{{0}"
          "\n    }}"
          "\n    let uslot = (gx_record_word({1}u) >> {2}u) & {3}u;"
          "\n    let umat = gx_record_word({4}u);"
          "\n    let urough = f32(umat & 0xffu) * bitcast<f32>(0x3b808081u);"
          "\n    let umetal = f32((umat >> 8u) & 0xffu) * bitcast<f32>(0x3b808081u);"
          "{5}"
          "\n    out.surface = gx_surface_pack(unlit_rgb, {6}, {7}, {8}, uslot);",
          replace_all(std::move(inner), "\n", "\n    "), render_ext::DrawRecordFlagsWord,
          render_ext::RecordSurfaceSlotShift, (1u << render_ext::RecordSurfaceSlotBits) - 1u,
          render_ext::DrawRecordMaterialWord,
          companionText ? "\n    let uorm = (u32(ubuf.mat0.x) & 8u) != 0u;"sv : ""sv,
          companionText ? "select(1.0, cmp_orm.r, uorm)"sv : "1.0"sv,
          companionText ? fmt::format("select(urough, cmp_orm.g * select(1.0, urough, (umat & {:#x}u) != 0u), uorm)",
                                      render_ext::MaterialWordRoughnessFactor)
                        : "urough"s,
          companionText ? fmt::format("select(umetal, cmp_orm.b * select(1.0, umetal, (umat & {:#x}u) != 0u), uorm)",
                                      render_ext::MaterialWordMetalnessFactor)
                        : "umetal"s);
    }
    members.push_back(
        {targets.surface, fmt::format("    @location({}) surface: vec4u,\n", targets.surface), std::move(assignment)});
    if (uses_surface_ext(config, info, targets)) {
      CHECK(targets.surface < targets.surfaceExt, "SurfaceExt reads the material word Surface's member loads");
      members.push_back(
          {targets.surfaceExt, fmt::format("    @location({}) surface_ext: vec4u,\n", targets.surfaceExt),
           fmt::format("\n    out.surface_ext = gx_surface_ext_pack({}, {});", surface_ext_ambient(config, info),
                       emitText ? fmt::format("select(vec3f(0.0), cmp_emit.rgb, (u32(ubuf.mat0.x) & {}u) "
                                              "!= 0u)",
                                              render_ext::LaneEmit)
                                : "vec3f(0.0)"s)});
    }
  }
  if (!members.empty()) {
    std::ranges::stable_sort(members, {}, &OutputMember::location);
    fragmentOutput = "\nstruct FragmentOutput {\n    @location(0) color: vec4f,\n";
    fragmentReturn = "\n    var out: FragmentOutput;\n    out.color = prev;";
    for (const OutputMember& member : members) {
      fragmentOutput += member.declaration;
      fragmentReturn += member.assignment;
    }
    fragmentOutput += "};\n";
    fragmentOutputType = "FragmentOutput"sv;
    fragmentReturn += "\n    return out;";
  }

  if (dstAlphaMode == DstAlphaMode::DualSource) {
    fragmentOutput =
        "\nstruct FragmentOutput {\n"
        "    @location(0) @blend_src(0) color: vec4f,\n"
        "    @location(0) @blend_src(1) blend: vec4f,\n"
        "};\n";
    fragmentOutputType = "FragmentOutput"sv;
    fragmentReturn = "\n    return FragmentOutput(vec4f(prev.rgb, 1.0), prev);";
  }

  auto shaderSource = fmt::format(R"""(
{13}

fn tev_overflow_f32(in: f32) -> f32 {{
  let byte_space = in * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}}

fn tev_overflow_vec3f(in: vec3f) -> vec3f {{
  let byte_space = in * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}}

fn tev_overflow_vec4f(in: vec4f) -> vec4f {{
  let byte_space = in * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}}

{8}

{14}
var<immediate> imm: Immediate;

struct Uniform {{
    render_viewport_size: vec2f,
    logical_viewport_size: vec2f,{0}
}};
@group(0) @binding(0)
var<storage, read> vbuf: array<u32>;
@group(0) @binding(1)
var<storage, read> abuf: array<u32>;
@group(1) @binding(0)
var<uniform> ubuf: Uniform;{1}

struct VertexOutput {{
    @builtin(position) pos: vec4f,{2}
}};

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32{3}
) -> VertexOutput {{
    var out: VertexOutput;{7}{4}
    return out;
}}

{9}
@fragment
fn fs_main(in: VertexOutput{12}) -> {10} {{{6}{5}{11}
}}
)""",
                                  uniBufAttrs, texBindings, vtxOutAttrs, vtxInAttrs, vtxXfrAttrs, fragmentFn,
                                  fragmentFnPre, vtxXfrAttrsPre, uniformPre, fragmentOutput, fragmentOutputType,
                                  fragmentReturn, backFaceText ? ", @builtin(front_facing) gx_ff: bool"sv : ""sv,
                                  vertex_fetch::wgsl_helpers(), vertex_fetch::wgsl_immediate_struct());
  if (dstAlphaMode == DstAlphaMode::DualSource) {
    shaderSource.insert(0, "enable dual_source_blending;\n");
  }
  if (EnableDebugPrints) {
    Log.info("Generated shader (hash {:x}): {}", hash, shaderSource);
  }

  return shaderSource;
}

ShaderTargets shader_targets(const gfx::RenderTargetLayout& layout) noexcept {
  ShaderTargets targets;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Normal) {
      targets.normal = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::DebugId) {
      targets.debugId = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Motion) {
      targets.motion = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Reactive) {
      targets.reactive = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Fog) {
      targets.fog = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Surface) {
      targets.surface = i;
    } else if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::SurfaceExt) {
      targets.surfaceExt = i;
    }
  }
  if (gfx::layout_has_gbuffer(layout)) {
    targets.gbufferNormal = std::exchange(targets.normal, UINT32_MAX);
  }
  targets.fwdEnv = gfx::layout_has_gbuffer(layout);
  return targets;
}

namespace testing {
uint64_t shader_hash(const ShaderConfig& config, DstAlphaMode dstAlphaMode, const ShaderTargets& targets) noexcept {
  return gx::shader_hash(config, dstAlphaMode, targets);
}

std::string_view companion_wgsl() noexcept { return CompanionWgsl; }
} // namespace testing

std::string receiver_alpha_wgsl() {
  return fmt::format(
      "fn gbuffer_rx_alpha(rec: u32, nrm_len_sq: f32) -> f32 {{ "
      "let flags = select(0u, abuf[rec + {}u], rec != 0u); "
      "return f32(select(0u, (flags >> {}u) & 3u, nrm_len_sq > 1e-10)) / 3.0; }}",
      render_ext::DrawRecordFlagsWord, render_ext::DrawRecordReceiverShift);
}

wgpu::ShaderModule build_shader(const ShaderConfig& config, const gfx::RenderTargetLayout& layout,
                                DstAlphaMode dstAlphaMode) noexcept {
  ZoneScoped;
  const ShaderTargets targets = shader_targets(layout);
  const auto shaderSource = build_shader_source(config, dstAlphaMode, targets);
  const auto hash = shader_hash(config, dstAlphaMode, targets);
  wgpu::ShaderSourceWGSL wgslDescriptor{};
  wgslDescriptor.code = shaderSource.c_str();
  const auto label = fmt::format("GX Shader {:x}", hash);
  const auto shaderDescriptor = wgpu::ShaderModuleDescriptor{
      .nextInChain = &wgslDescriptor,
      .label = label.c_str(),
  };
  return webgpu::g_device.CreateShaderModule(&shaderDescriptor);
}
} // namespace aurora::gx

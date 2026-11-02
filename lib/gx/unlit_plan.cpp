#include "unlit_plan.hpp"

#include "../gfx/hash.hpp"
#include "shader_info.hpp"

#include <dolphin/gx/GXEnum.h>

#include <absl/container/flat_hash_map.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <initializer_list>

namespace aurora::gx::unlit {
namespace {
constexpr size_t CacheLimit = 4096;

enum class Dep : uint8_t { K, Lin, Nl };

struct Taint {
  uint8_t rule = 0;
  uint8_t stage = NoStage;

  explicit operator bool() const noexcept { return rule != 0; }
};

struct Value {
  Dep dep = Dep::K;
  bool constant = false;
  uint8_t chans = 0;
  Taint taint{};
};

constexpr Value Constant{.dep = Dep::K, .constant = true};

Taint first(Taint a, Taint b) noexcept { return a ? a : b; }

Value vmax(std::initializer_list<Value> values) noexcept {
  Value out{};
  bool constant = true;
  for (const Value& v : values) {
    out.dep = std::max(out.dep, v.dep);
    constant = constant && v.constant;
    out.chans |= v.chans;
    out.taint = first(out.taint, v.taint);
  }
  out.constant = out.dep == Dep::K && constant;
  return out;
}

enum class Texgen : uint8_t { None, Normal, Colour };

struct LightTexgen {
  Texgen kind = Texgen::None;
  uint8_t channel = 0;
};

LightTexgen light_texgen(const ShaderConfig& config, u32 coord) noexcept {
  if (coord >= MaxTexCoord) {
    return {};
  }
  const TcgConfig& tg = config.tcgs[coord];
  if ((tg.type >= GX_TG_BUMP0 && tg.type <= GX_TG_BUMP7) || tg.src == GX_TG_NRM || tg.src == GX_TG_BINRM ||
      tg.src == GX_TG_TANGENT) {
    return {.kind = Texgen::Normal};
  }
  if (tg.src == GX_TG_COLOR0 || tg.src == GX_TG_COLOR1 || tg.type == GX_TG_SRTG) {
    return {.kind = Texgen::Colour, .channel = static_cast<uint8_t>(tg.src == GX_TG_COLOR1 ? 1 : 0)};
  }
  return {};
}

bool has_map(const TevStage& stage) noexcept { return stage.texMapId <= GX_TEXMAP7; }

uint8_t map_bit(const TevStage& stage) noexcept {
  return has_map(stage) ? static_cast<uint8_t>(1u << stage.texMapId) : 0;
}

bool is_tex(GXTevColorArg arg) noexcept { return arg == GX_CC_TEXC || arg == GX_CC_TEXA; }

bool is_alpha_bump(GXChannelID id) noexcept { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

struct Position {
  bool add = false;
  bool mul = false;
  bool other = false;
  uint8_t maps = 0;
};

Position tex_position(const ShaderConfig& config, u32 count, u32 coord) noexcept {
  Position pos;
  for (u32 s = 0; s < count; ++s) {
    const TevStage& st = config.tevStages[s];
    if (st.texCoordId != coord || !has_map(st)) {
      continue;
    }
    pos.maps |= map_bit(st);
    const auto& [a, b, c, d] = st.colorPass;
    if (is_tex(d) && !is_tex(a) && !is_tex(b) && !is_tex(c)) {
      pos.add = true;
    } else if (is_tex(c) && a == GX_CC_ZERO && d == GX_CC_ZERO) {
      pos.mul = true;
    } else if (is_tex(b) && a == GX_CC_ZERO && !is_tex(c) && !is_tex(d)) {
      pos.mul = true;
    } else if (is_tex(a) || is_tex(b) || is_tex(c) || is_tex(d)) {
      pos.other = true;
    }
  }
  return pos;
}

struct Neutral {
  uint8_t one = 0;
  uint8_t zero = 0;
  Taint fail{};
};

Neutral neutral_maps_v0(const ShaderConfig& config, u32 count) noexcept {
  Neutral neutral;
  for (u32 t = 0; t < MaxTexCoord; ++t) {
    uint8_t firstStage = NoStage;
    for (u32 s = 0; s < count && firstStage == NoStage; ++s) {
      if (config.tevStages[s].texCoordId == t) {
        firstStage = static_cast<uint8_t>(s);
      }
    }
    if (firstStage == NoStage) {
      continue;
    }
    const LightTexgen texgen = light_texgen(config, t);
    if (texgen.kind == Texgen::None) {
      continue;
    }
    const Position pos = tex_position(config, count, t);
    if (pos.add && !pos.mul && !pos.other) {
      neutral.zero |= pos.maps;
      neutral.one &= ~pos.maps;
    } else if (pos.mul && !pos.add && !pos.other) {
      neutral.one |= pos.maps;
      neutral.zero &= ~pos.maps;
    } else {
      neutral.fail = first(neutral.fail,
                           {.rule = static_cast<uint8_t>(texgen.kind == Texgen::Colour ? 8 : 7), .stage = firstStage});
    }
  }
  return neutral;
}

bool uses_indirect(const ShaderConfig& config, const TevStage& stage) noexcept {
  return stage.indTexStage < config.numIndStages && stage.indTexStage < MaxIndStages &&
         (stage.indTexMtxId != GX_ITM_OFF || is_alpha_bump(stage.channelId));
}

uint16_t layer_stages(const ShaderConfig& config, u32 count, uint8_t neutralMaps) noexcept {
  uint8_t indirectMaps = 0;
  for (u32 s = 0; s < count; ++s) {
    const TevStage& st = config.tevStages[s];
    if (uses_indirect(config, st) && config.indStages[st.indTexStage].texMapId <= GX_TEXMAP7) {
      indirectMaps |= static_cast<uint8_t>(1u << config.indStages[st.indTexStage].texMapId);
    }
  }
  uint16_t stages = 0;
  for (u32 s = 0; s < count; ++s) {
    const TevStage& st = config.tevStages[s];
    const auto& pass = st.colorPass;
    const bool readsTexc = pass.a == GX_CC_TEXC || pass.b == GX_CC_TEXC || pass.c == GX_CC_TEXC || pass.d == GX_CC_TEXC;
    const bool tevCoord = st.indTexMtxId != GX_ITM_OFF || st.indTexWrapS != GX_ITW_OFF ||
                          st.indTexWrapT != GX_ITW_OFF || st.indTexAddPrev;
    if (!readsTexc || !has_map(st) || ((neutralMaps | indirectMaps) & map_bit(st)) != 0 ||
        st.texCoordId >= MaxTexCoord || tevCoord) {
      continue;
    }
    const TcgConfig& tg = config.tcgs[st.texCoordId];
    if ((tg.type == GX_TG_MTX2x4 || tg.type == GX_TG_MTX3x4) && tg.src >= GX_TG_TEX0 && tg.src <= GX_TG_TEXCOORD6) {
      stages |= static_cast<uint16_t>(1u << s);
    }
  }
  return stages;
}

Value combine(const TevOp& op, const Value& a, const Value& b, const Value& c, const Value& d, uint8_t stage) noexcept {
  const Taint r9{.rule = 9, .stage = stage};
  if (op.op >= GX_TEV_COMP_R8_GT) {
    if (a.dep != Dep::K || b.dep != Dep::K) {
      return {.dep = Dep::Nl, .taint = r9};
    }
    return vmax({c, d});
  }
  Value lerp;
  if (c.dep == Dep::Lin) {
    if (a.dep != Dep::K || b.dep != Dep::K) {
      lerp = {.dep = Dep::Nl, .taint = r9};
    } else if (!(a.constant || b.constant)) {
      lerp = {.dep = Dep::Lin, .chans = c.chans, .taint = first(c.taint, r9)};
    } else {
      lerp = {.dep = Dep::Lin, .chans = c.chans, .taint = c.taint};
    }
  } else if (c.dep == Dep::Nl) {
    lerp = {.dep = Dep::Nl, .taint = first(c.taint, r9)};
  } else {
    lerp = vmax({a, b, c});
  }
  return vmax({lerp, d});
}
} // namespace

Plan analyse(const ShaderConfig& config) noexcept {
  const u32 count = std::min<u32>(config.tevStageCount, MaxTevStages);
  const Neutral neutral = neutral_maps_v0(config, count);
  const uint8_t neutralMaps = neutral.one | neutral.zero;
  std::array<bool, MaxColorChannels> lit{};
  for (u32 i = 0; i < MaxColorChannels; ++i) {
    lit[i] = config.colorChannels[i].lightingEnabled;
  }
  std::array<Value, MaxTevRegs> col{};
  std::array<Value, MaxTevRegs> alp{};
  uint16_t scaleDivide = 0;
  bool chainLit = false;
  for (u32 s = 0; s < count; ++s) {
    const TevStage& st = config.tevStages[s];
    const auto stage = static_cast<uint8_t>(s);
    const Taint r5{.rule = 5, .stage = stage};
    const Taint r8{.rule = 8, .stage = stage};
    const bool rasNull = st.channelId == GX_COLOR_ZERO || st.channelId == GX_COLOR_NULL;
    // Conservative v1 extension: SP1 measured no such program, so no SP1 verdict changes
    bool ownLit = false;
    if (uses_indirect(config, st)) {
      const LightTexgen texgen = light_texgen(config, config.indStages[st.indTexStage].texCoordId);
      ownLit = texgen.kind == Texgen::Colour && lit[texgen.channel];
    }
    const bool tevCoord = st.indTexMtxId != GX_ITM_OFF || st.indTexWrapS != GX_ITW_OFF ||
                          st.indTexWrapT != GX_ITW_OFF || st.indTexAddPrev;
    if (tevCoord) {
      chainLit = (st.indTexAddPrev && chainLit) || ownLit;
    }
    const bool indirectLit = ownLit || (tevCoord && chainLit);

    const auto tex_value = [&]() -> Value {
      if (!has_map(st) || (neutralMaps & map_bit(st)) != 0) {
        return Constant;
      }
      const LightTexgen texgen = light_texgen(config, st.texCoordId);
      if ((texgen.kind == Texgen::Colour && lit[texgen.channel]) || indirectLit) {
        return {.dep = Dep::Nl, .taint = r8};
      }
      return {};
    };
    // Conservative v1 extension: SP1 measured no such program, so no SP1 verdict changes
    const auto ras_component = [&](GXTevColorChan comp, bool colourPath) -> Value {
      const u32 cidx = color_channel(st.channelId);
      const bool alpha = comp == GX_CH_ALPHA;
      const u32 idx = alpha ? cidx + 2 : cidx;
      if (!lit[idx]) {
        return {};
      }
      return {.dep = Dep::Lin, .chans = static_cast<uint8_t>(1u << idx), .taint = alpha && colourPath ? r5 : Taint{}};
    };
    const auto rast = [&](bool rgb, bool colourPath) -> Value {
      if (rasNull) {
        return Constant;
      }
      if (is_alpha_bump(st.channelId)) {
        return indirectLit ? Value{.dep = Dep::Nl, .taint = r8} : Constant;
      }
      const TevSwap& swap = config.tevSwapTable[st.tevSwapRas % MaxTevSwap];
      if (rgb) {
        return vmax({ras_component(swap.red, colourPath), ras_component(swap.green, colourPath),
                     ras_component(swap.blue, colourPath)});
      }
      return ras_component(swap.alpha, colourPath);
    };
    const auto carg = [&](GXTevColorArg arg) -> Value {
      switch (arg) {
      case GX_CC_CPREV:
        return col[0];
      case GX_CC_C0:
        return col[1];
      case GX_CC_C1:
        return col[2];
      case GX_CC_C2:
        return col[3];
      case GX_CC_APREV:
      case GX_CC_A0:
      case GX_CC_A1:
      case GX_CC_A2: {
        Value v = alp[(arg - GX_CC_APREV) / 2];
        if (v.dep != Dep::K) {
          v.taint = first(v.taint, r5);
        }
        return v;
      }
      case GX_CC_TEXC:
      case GX_CC_TEXA:
        return tex_value();
      case GX_CC_RASC:
        return rast(true, true);
      case GX_CC_RASA:
        return rast(false, true);
      default:
        return Constant;
      }
    };
    const auto aarg = [&](GXTevAlphaArg arg) -> Value {
      switch (arg) {
      case GX_CA_APREV:
      case GX_CA_A0:
      case GX_CA_A1:
      case GX_CA_A2:
        return alp[arg - GX_CA_APREV];
      case GX_CA_TEXA:
        return tex_value();
      case GX_CA_RASA:
        return rast(false, false);
      default:
        return Constant;
      }
    };

    const auto& cp = st.colorPass;
    const Value cout = combine(st.colorOp, carg(cp.a), carg(cp.b), carg(cp.c), carg(cp.d), stage);
    if (cout.dep == Dep::Lin && (st.colorOp.scale == GX_CS_SCALE_2 || st.colorOp.scale == GX_CS_SCALE_4)) {
      scaleDivide |= static_cast<uint16_t>(1u << s);
    }
    const auto& ap = st.alphaPass;
    const Value aout = combine(st.alphaOp, aarg(ap.a), aarg(ap.b), aarg(ap.c), aarg(ap.d), stage);
    col[st.colorOp.outReg % MaxTevRegs] = cout;
    alp[st.alphaOp.outReg % MaxTevRegs] = aout;
  }

  const Value out = count == 0 ? col[0] : col[config.tevStages[count - 1].colorOp.outReg % MaxTevRegs];
  Plan p;
  const auto fail = [&p](Verdict verdict, Taint taint) {
    p.verdict = verdict;
    p.failRule = taint.rule;
    p.failStage = taint.stage;
  };
  if (out.dep == Dep::K) {
    p.verdict = Verdict::Passthrough;
  } else if (out.taint.rule == 5) {
    fail(Verdict::FailOther, out.taint);
  } else if (out.dep == Dep::Nl || out.taint) {
    const Taint taint = out.taint ? out.taint : Taint{.rule = 9};
    fail(taint.rule == 8 ? Verdict::FailLightTexgen : Verdict::FailNonlinear, taint);
  } else if (neutral.fail) {
    fail(Verdict::FailLightTexgen, neutral.fail);
  } else {
    const uint8_t colour = out.chans & 3u;
    const auto spec = [&config](u32 c) { return config.colorChannels[c].attnFn == GX_AF_SPEC; };
    if (colour != 0 && ((colour & 1u) == 0 || spec(0)) && ((colour & 2u) == 0 || spec(1))) {
      fail(Verdict::FailSpecOnly, {.rule = 4});
    } else {
      for (u32 c = 0; c < 2; ++c) {
        if ((colour & (1u << c)) == 0) {
          continue;
        }
        const ColorChannelConfig& channel = config.colorChannels[c];
        if (channel.attnFn == GX_AF_SPEC) {
          p.rewrite[c] = RewriteZero;
        } else if (channel.matSrc == GX_SRC_REG) {
          p.rewrite[c] = RewriteMaterial;
        } else {
          p.rewrite[c] = RewriteVertex;
          if (config.attrs[GX_VA_CLR0 + c].attrType != GX_NONE) {
            p.rawVtx |= static_cast<uint8_t>(1u << c);
          }
        }
      }
      p.verdict = scaleDivide != 0 ? Verdict::Scale : Verdict::Linear;
      p.scaleDivide = scaleDivide;
      p.neutralOne = neutral.one;
      p.neutralZero = neutral.zero;
    }
  }
  p.layerStages = layer_stages(config, count, p.neutralOne | p.neutralZero);
  return p;
}

Plan plan(const ShaderConfig& config) noexcept {
  thread_local absl::flat_hash_map<uint64_t, Plan> cache;
  const uint64_t key = XXH3_64bits(&config, offsetof(ShaderConfig, ext));
  if (const auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  if (cache.size() >= CacheLimit) {
    cache.clear();
  }
  const Plan p = analyse(config);
  cache.emplace(key, p);
  return p;
}

bool admitted(const Plan& p, const ShaderConfig& config, uint32_t baseVaryings) noexcept {
  return p.pass() && config.attrs[GX_VA_NRM].attrType != GX_NONE && config.lineMode == 0 &&
         baseVaryings + raw_vertex_varyings(p) <= MaxGxVaryings;
}

uint32_t raw_vertex_varyings(const Plan& p) noexcept { return static_cast<uint32_t>(std::popcount(p.rawVtx)); }

std::string_view verdict_name(Verdict verdict) noexcept {
  switch (verdict) {
  case Verdict::Passthrough:
    return "PASS_PASSTHROUGH";
  case Verdict::Linear:
    return "PASS_LINEAR";
  case Verdict::Scale:
    return "PASS_SCALE";
  case Verdict::FailSpecOnly:
    return "FAIL_SPEC_ONLY";
  case Verdict::FailOther:
    return "FAIL_OTHER";
  case Verdict::FailNonlinear:
    return "FAIL_NONLINEAR";
  case Verdict::FailLightTexgen:
    return "FAIL_LIGHT_TEXGEN";
  }
  return "UNKNOWN";
}
} // namespace aurora::gx::unlit

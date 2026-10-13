#include "gx_shader_corpus.hpp"

#include "gx/shader_info.hpp"

#include <fmt/format.h>
#include <xxhash.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <type_traits>
#include <utility>

namespace aurora::gx::corpus {
namespace {
static_assert(RecordAttrsOffset + MaxVtxAttr * RecordAttrStride == RecordSwapOffset);
static_assert(RecordSwapOffset + MaxTevSwap * 4 == RecordStagesOffset);
static_assert(RecordStagesOffset + MaxTevStages * RecordStageStride == RecordChannelsOffset);
static_assert(RecordChannelsOffset + MaxColorChannels * RecordChannelStride == RecordTcgsOffset);
static_assert(RecordTcgsOffset + MaxTexCoord * RecordTcgStride == RecordAlphaCompareOffset);
static_assert(RecordAlphaCompareOffset + 5 == RecordIndStagesOffset);
static_assert(RecordIndStagesOffset + MaxIndStages * RecordIndStageStride == RecordReservedOffset);
static_assert(RecordReservedOffset + 3 == RecordSize);
static_assert(sizeof(AttrConfig) == RecordAttrStride);

constexpr size_t StageColorClamp = 12;
constexpr size_t StageAlphaClamp = 17;
constexpr size_t StageUseOrigLod = 32;
constexpr size_t StageAddPrev = 33;
constexpr size_t AttrLe = 6;
constexpr size_t AttrNbt3 = 7;
constexpr size_t ChannelLightingEnabled = 4;
constexpr size_t TcgNormalize = 4;
constexpr size_t RecordFogRangeEnabled = 3;
constexpr size_t RecordLineMode = 2;
constexpr size_t RecordDstAlphaMode = 6;
constexpr size_t RecordNormalAttachment = 7;

constexpr uint64_t RandomSeed = 0x6A09E667F3BCC908ull;
constexpr uint32_t RandomCount = 256;
constexpr uint32_t RandomAttempts = 4096;

constexpr std::array<uint8_t, 11> FogTypes{
    GX_FOG_NONE,         GX_FOG_PERSP_LIN,     GX_FOG_PERSP_EXP,     GX_FOG_PERSP_EXP2,
    GX_FOG_PERSP_REVEXP, GX_FOG_PERSP_REVEXP2, GX_FOG_ORTHO_LIN,     GX_FOG_ORTHO_EXP,
    GX_FOG_ORTHO_EXP2,   GX_FOG_ORTHO_REVEXP,  GX_FOG_ORTHO_REVEXP2,
};
constexpr std::array<GXTevOp, 10> TevOps{
    GX_TEV_ADD,          GX_TEV_SUB,           GX_TEV_COMP_R8_GT,    GX_TEV_COMP_R8_EQ,   GX_TEV_COMP_GR16_GT,
    GX_TEV_COMP_GR16_EQ, GX_TEV_COMP_BGR24_GT, GX_TEV_COMP_BGR24_EQ, GX_TEV_COMP_RGB8_GT, GX_TEV_COMP_RGB8_EQ,
};
// Weighted toward ADD/SUB because comparison ops are rare in real materials.
constexpr std::array<GXTevOp, 14> RandomTevOps{
    GX_TEV_ADD,           GX_TEV_ADD,           GX_TEV_ADD,          GX_TEV_ADD,          GX_TEV_SUB,
    GX_TEV_SUB,           GX_TEV_COMP_R8_GT,    GX_TEV_COMP_R8_EQ,   GX_TEV_COMP_GR16_GT, GX_TEV_COMP_GR16_EQ,
    GX_TEV_COMP_BGR24_GT, GX_TEV_COMP_BGR24_EQ, GX_TEV_COMP_RGB8_GT, GX_TEV_COMP_RGB8_EQ,
};
constexpr std::array<GXIndTexMtxID, 10> IndMtxIds{
    GX_ITM_OFF, GX_ITM_0, GX_ITM_1, GX_ITM_2, GX_ITM_S0, GX_ITM_S1, GX_ITM_S2, GX_ITM_T0, GX_ITM_T1, GX_ITM_T2,
};
constexpr std::array<GXChannelID, 10> ChannelIds{
    GX_COLOR0,   GX_COLOR1,     GX_ALPHA0,     GX_ALPHA1,      GX_COLOR0A0,
    GX_COLOR1A1, GX_COLOR_ZERO, GX_ALPHA_BUMP, GX_ALPHA_BUMPN, GX_COLOR_NULL,
};

constexpr bool valid_kcsel(uint32_t value) {
  return value <= GX_TEV_KCSEL_1_8 || (value >= GX_TEV_KCSEL_K0 && value <= GX_TEV_KCSEL_K3_A);
}
constexpr bool valid_kasel(uint32_t value) {
  return value <= GX_TEV_KASEL_1_8 || (value >= GX_TEV_KASEL_K0_R && value <= GX_TEV_KASEL_K3_A);
}

template <size_t N, typename T, bool (*Valid)(uint32_t)>
constexpr std::array<T, N> valid_values() {
  std::array<T, N> values{};
  size_t count = 0;
  for (uint32_t value = 0; value <= 0xFF && count < N; ++value) {
    if (Valid(value)) {
      values[count++] = static_cast<T>(value);
    }
  }
  return values;
}
constexpr auto KColorSels = valid_values<28, GXTevKColorSel, valid_kcsel>();
constexpr auto KAlphaSels = valid_values<24, GXTevKAlphaSel, valid_kasel>();

struct NamedVariant {
  Variant variant;
  std::string_view name;
};
constexpr std::array<NamedVariant, 5> Variants{{
    {{DstAlphaMode::None, UINT32_MAX}, "none"},
    {{DstAlphaMode::None, 1}, "none.n1"},
    {{DstAlphaMode::Replace, UINT32_MAX}, "replace"},
    {{DstAlphaMode::Replace, 1}, "replace.n1"},
    {{DstAlphaMode::DualSource, UINT32_MAX}, "dual"},
}};

std::string_view variant_name(const Variant& variant) {
  for (const auto& named : Variants) {
    if (named.variant == variant) {
      return named.name;
    }
  }
  return "other";
}

template <typename T>
constexpr uint32_t num(T value) {
  if constexpr (std::is_enum_v<T>) {
    return static_cast<uint32_t>(static_cast<std::make_unsigned_t<std::underlying_type_t<T>>>(value));
  } else {
    return static_cast<uint32_t>(value);
  }
}

// Bools are read through their bytes because blob data may hold values other than 0 and 1.
template <typename T>
uint64_t raw_value(const T& value) noexcept {
  if constexpr (std::is_same_v<T, bool>) {
    return std::bit_cast<uint8_t>(value);
  } else if constexpr (std::is_enum_v<T>) {
    return static_cast<std::make_unsigned_t<std::underlying_type_t<T>>>(value);
  } else {
    return static_cast<std::make_unsigned_t<T>>(value);
  }
}

// One visitor drives both encode and decode so the two can never disagree on field order.
template <typename Config, typename Fn>
void visit_fields(Config& config, Fn&& fn) {
  for (auto& attr : config.attrs) {
    fn(attr.attrType);
    fn(attr.cnt);
    fn(attr.compType);
    fn(attr.offset);
    fn(attr.stride);
    fn(attr.frac);
    fn(attr.le);
    fn(attr.nbt3);
  }
  for (auto& swap : config.tevSwapTable) {
    fn(swap.red);
    fn(swap.green);
    fn(swap.blue);
    fn(swap.alpha);
  }
  for (auto& stage : config.tevStages) {
    fn(stage.colorPass.a);
    fn(stage.colorPass.b);
    fn(stage.colorPass.c);
    fn(stage.colorPass.d);
    fn(stage.alphaPass.a);
    fn(stage.alphaPass.b);
    fn(stage.alphaPass.c);
    fn(stage.alphaPass.d);
    fn(stage.colorOp.op);
    fn(stage.colorOp.bias);
    fn(stage.colorOp.scale);
    fn(stage.colorOp.outReg);
    fn(stage.colorOp.clamp);
    fn(stage.alphaOp.op);
    fn(stage.alphaOp.bias);
    fn(stage.alphaOp.scale);
    fn(stage.alphaOp.outReg);
    fn(stage.alphaOp.clamp);
    fn(stage.kcSel);
    fn(stage.kaSel);
    fn(stage.texCoordId);
    fn(stage.texMapId);
    fn(stage.channelId);
    fn(stage.tevSwapRas);
    fn(stage.tevSwapTex);
    fn(stage.indTexStage);
    fn(stage.indTexFormat);
    fn(stage.indTexBiasSel);
    fn(stage.indTexAlphaSel);
    fn(stage.indTexMtxId);
    fn(stage.indTexWrapS);
    fn(stage.indTexWrapT);
    fn(stage.indTexUseOrigLOD);
    fn(stage.indTexAddPrev);
  }
  for (auto& channel : config.colorChannels) {
    fn(channel.matSrc);
    fn(channel.ambSrc);
    fn(channel.diffFn);
    fn(channel.attnFn);
    fn(channel.lightingEnabled);
  }
  for (auto& tcg : config.tcgs) {
    fn(tcg.type);
    fn(tcg.src);
    fn(tcg.mtx);
    fn(tcg.postMtx);
    fn(tcg.normalize);
    fn(tcg.embossSrc);
  }
  fn(config.alphaCompare.comp0);
  fn(config.alphaCompare.ref0);
  fn(config.alphaCompare.op);
  fn(config.alphaCompare.comp1);
  fn(config.alphaCompare.ref1);
  for (auto& ind : config.indStages) {
    fn(ind.texCoordId);
    fn(ind.texMapId);
    fn(ind.scaleS);
    fn(ind.scaleT);
  }
}

bool is_bump(GXTexGenType type) { return type >= GX_TG_BUMP0 && type <= GX_TG_BUMP7; }
bool is_tex_src(GXTexGenSrc src) { return src >= GX_TG_TEX0 && src <= GX_TG_TEX7; }
bool is_texcoord_src(GXTexGenSrc src) { return src >= GX_TG_TEXCOORD0 && src <= GX_TG_TEXCOORD6; }

bool uses_texture_arg(const TevStage& stage) {
  const auto color = [](GXTevColorArg arg) { return arg == GX_CC_TEXC || arg == GX_CC_TEXA; };
  const auto& c = stage.colorPass;
  const auto& a = stage.alphaPass;
  return color(c.a) || color(c.b) || color(c.c) || color(c.d) || a.a == GX_CA_TEXA || a.b == GX_CA_TEXA ||
         a.c == GX_CA_TEXA || a.d == GX_CA_TEXA;
}

bool valid_op(GXTevOp op) {
  return op == GX_TEV_ADD || op == GX_TEV_SUB || (op >= GX_TEV_COMP_R8_GT && op <= GX_TEV_COMP_RGB8_EQ);
}
bool valid_fog(uint8_t type) { return std::find(FogTypes.begin(), FogTypes.end(), type) != FogTypes.end(); }
bool valid_ind_mtx(GXIndTexMtxID id) { return std::find(IndMtxIds.begin(), IndMtxIds.end(), id) != IndMtxIds.end(); }
bool valid_slot(uint32_t value, uint32_t max) { return value <= max || value == 0xFF; }

bool same_attr(const AttrConfig& lhs, const AttrConfig& rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(AttrConfig)) == 0;
}

bool valid_label(std::string_view label) {
  if (label.empty() || label.size() > 255) {
    return false;
  }
  return std::all_of(label.begin(), label.end(), [](char c) { return c > ' ' && c <= '~'; });
}

template <typename P>
auto& pass_slot(P& pass, uint32_t slot) {
  switch (slot) {
  case 0:
    return pass.a;
  case 1:
    return pass.b;
  case 2:
    return pass.c;
  default:
    return pass.d;
  }
}

class SplitMix64 {
public:
  explicit SplitMix64(uint64_t seed) : m_state(seed) {}

  uint64_t next() {
    m_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = m_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  uint32_t below(uint64_t n) { return static_cast<uint32_t>(next() % n); }

private:
  uint64_t m_state;
};

class Sink {
public:
  bool add(ShaderConfig config, std::string_view label, Variant variant = {}) {
    canonicalize(config);
    const auto record = encode(config, variant);
    if (!record) {
      fmt::print(stderr, "gx shader corpus: cannot encode {}\n", label);
      std::abort();
    }
    if (!m_seen.insert(record->bytes).second) {
      return false;
    }
    m_entries.push_back({*record, fmt::format("{}@{}", label, variant_name(variant))});
    return true;
  }

  std::vector<Entry> take() { return std::move(m_entries); }

private:
  std::set<std::array<uint8_t, RecordSize>> m_seen;
  std::vector<Entry> m_entries;
};

AttrConfig direct_attr(uint32_t cnt, uint32_t compType) {
  return {.attrType = GX_DIRECT, .cnt = static_cast<u8>(cnt), .compType = static_cast<u8>(compType)};
}

AttrConfig indexed_attr(GXAttrType type, uint32_t cnt, uint32_t compType, uint32_t stride) {
  return {
      .attrType = static_cast<u8>(type),
      .cnt = static_cast<u8>(cnt),
      .compType = static_cast<u8>(compType),
      .stride = static_cast<u8>(stride),
      .le = false,
  };
}

AttrConfig make_attr(GXAttr attr, uint32_t type, uint32_t cnt, uint32_t compType, bool nbt3) {
  AttrConfig config = direct_attr(cnt, compType);
  if (type != GX_DIRECT) {
    const uint32_t size = comp_type_size(attr, static_cast<GXCompType>(compType));
    config = indexed_attr(static_cast<GXAttrType>(type), cnt, compType, size * (nbt3 ? 3 : cnt));
  }
  config.nbt3 = nbt3;
  return config;
}

GXTexGenSrc tex_src(uint32_t unit) { return static_cast<GXTexGenSrc>(GX_TG_TEX0 + unit); }

TcgConfig texgen(GXTexGenType type, GXTexGenSrc src, GXTexMtx mtx = GX_IDENTITY, GXPTTexMtx postMtx = GX_PTIDENTITY) {
  return {.type = type, .src = src, .mtx = mtx, .postMtx = postMtx};
}

ColorChannelConfig lit_channel(GXColorSrc matSrc, GXColorSrc ambSrc, GXDiffuseFn diffFn, GXAttnFn attnFn) {
  return {.matSrc = matSrc, .ambSrc = ambSrc, .diffFn = diffFn, .attnFn = attnFn, .lightingEnabled = true};
}

TevStage textured_stage(uint32_t coord, uint32_t map) {
  TevStage stage;
  stage.colorPass = {GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO};
  stage.alphaPass = {GX_CA_ZERO, GX_CA_APREV, GX_CA_TEXA, GX_CA_ZERO};
  stage.texCoordId = static_cast<GXTexCoordID>(coord);
  stage.texMapId = static_cast<GXTexMapID>(map);
  return stage;
}

GXTevColorArg reg_color_arg(uint32_t reg) { return static_cast<GXTevColorArg>(GX_CC_CPREV + reg * 2); }
GXTevAlphaArg reg_alpha_arg(uint32_t reg) { return static_cast<GXTevAlphaArg>(GX_CA_APREV + reg); }

const ColorChannelConfig DefaultLit = lit_channel(GX_SRC_VTX, GX_SRC_REG, GX_DF_CLAMP, GX_AF_SPOT);

ShaderConfig base_config() {
  ShaderConfig config{};
  config.attrs[GX_VA_POS] = direct_attr(3, GX_F32);
  config.attrs[GX_VA_CLR0] = direct_attr(1, GX_RGBA8);
  config.attrs[GX_VA_TEX0] = direct_attr(2, GX_F32);
  config.tevSwapTable = {
      TevSwap{},
      TevSwap{GX_CH_RED, GX_CH_RED, GX_CH_RED, GX_CH_ALPHA},
      TevSwap{GX_CH_GREEN, GX_CH_GREEN, GX_CH_GREEN, GX_CH_ALPHA},
      TevSwap{GX_CH_BLUE, GX_CH_BLUE, GX_CH_BLUE, GX_CH_ALPHA},
  };
  config.tcgs[0] = texgen(GX_TG_MTX2x4, GX_TG_TEX0);
  config.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_VTX};
  config.colorChannels[GX_ALPHA0] = {.matSrc = GX_SRC_VTX};
  auto& stage = config.tevStages[0];
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  stage.alphaPass = {GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO};
  stage.texCoordId = GX_TEXCOORD0;
  stage.texMapId = GX_TEXMAP0;
  stage.channelId = GX_COLOR0A0;
  config.tevStageCount = 1;
  return config;
}

void variant_family(Sink& sink) {
  const auto b0 = base_config();
  auto nrm = b0;
  nrm.attrs[GX_VA_NRM] = direct_attr(3, GX_F32);
  auto nrmLit = nrm;
  nrmLit.colorChannels[GX_COLOR0] = DefaultLit;
  auto lit = b0;
  lit.colorChannels[GX_COLOR0] = DefaultLit;
  auto fog = b0;
  fog.fogType = GX_FOG_PERSP_LIN;
  auto alpha = b0;
  alpha.alphaCompare = {GX_GREATER, 128, GX_AOP_AND, GX_ALWAYS, 0};
  const std::array<std::pair<std::string_view, ShaderConfig>, 6> bases{{
      {"b0", b0},
      {"nrm", nrm},
      {"nrm_lit", nrmLit},
      {"lit", lit},
      {"fog", fog},
      {"alpha", alpha},
  }};
  for (const auto& [name, config] : bases) {
    for (const auto& named : Variants) {
      sink.add(config, fmt::format("variant/base={}", name), named.variant);
    }
  }
}

void tev_count_family(Sink& sink) {
  for (uint32_t n = 1; n <= MaxTevStages; ++n) {
    auto config = base_config();
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t unit = i % MaxTexCoord;
      config.tevStages[i] = textured_stage(unit, unit);
      config.tcgs[unit] = texgen(GX_TG_MTX2x4, tex_src(unit));
      config.attrs[GX_VA_TEX0 + unit] = direct_attr(2, GX_F32);
    }
    config.tevStageCount = n;
    sink.add(config, fmt::format("tev.count/n={}", n));
  }
}

void tev_arg_family(Sink& sink) {
  constexpr std::string_view SlotNames = "abcd";
  for (uint32_t slot = 0; slot < 4; ++slot) {
    for (uint32_t arg = 0; arg <= GX_CC_ZERO; ++arg) {
      auto config = base_config();
      auto& stage = config.tevStages[0];
      pass_slot(stage.colorPass, slot) = static_cast<GXTevColorArg>(arg);
      if (arg == GX_CC_KONST) {
        stage.kcSel = GX_TEV_KCSEL_K0;
      }
      sink.add(config, fmt::format("tev.carg/slot={}/arg={}", SlotNames[slot], arg));
    }
  }
  for (uint32_t slot = 0; slot < 4; ++slot) {
    for (uint32_t arg = 0; arg <= GX_CA_ZERO; ++arg) {
      auto config = base_config();
      auto& stage = config.tevStages[0];
      pass_slot(stage.alphaPass, slot) = static_cast<GXTevAlphaArg>(arg);
      if (arg == GX_CA_KONST) {
        stage.kaSel = GX_TEV_KASEL_K0_A;
      }
      sink.add(config, fmt::format("tev.aarg/slot={}/arg={}", SlotNames[slot], arg));
    }
  }
}

void tev_op_family(Sink& sink) {
  for (const bool alpha : {false, true}) {
    for (const auto op : TevOps) {
      for (const bool clamp : {false, true}) {
        auto config = base_config();
        auto& stage = config.tevStages[1];
        stage.colorPass = {GX_CC_C0, GX_CC_TEXC, GX_CC_ONE, GX_CC_ZERO};
        stage.alphaPass = {GX_CA_A0, GX_CA_TEXA, GX_CA_KONST, GX_CA_ZERO};
        stage.kaSel = GX_TEV_KASEL_8_8;
        stage.texCoordId = GX_TEXCOORD0;
        stage.texMapId = GX_TEXMAP0;
        auto& tevOp = alpha ? stage.alphaOp : stage.colorOp;
        tevOp.op = op;
        tevOp.clamp = clamp;
        config.tevStageCount = 2;
        sink.add(config, fmt::format("tev.cop/chan={}/op={}/clamp={}", alpha ? "alpha" : "color", num(op), num(clamp)));
      }
    }
  }
  for (const bool alpha : {false, true}) {
    for (uint32_t bias = 0; bias <= GX_TB_SUBHALF; ++bias) {
      for (uint32_t scale = 0; scale <= GX_CS_DIVIDE_2; ++scale) {
        auto config = base_config();
        auto& tevOp = alpha ? config.tevStages[0].alphaOp : config.tevStages[0].colorOp;
        tevOp.bias = static_cast<GXTevBias>(bias);
        tevOp.scale = static_cast<GXTevScale>(scale);
        sink.add(config,
                 fmt::format("tev.bias_scale/chan={}/bias={}/scale={}", alpha ? "alpha" : "color", bias, scale));
      }
    }
  }
}

void tev_outreg_family(Sink& sink) {
  for (uint32_t c = 0; c < MaxTevRegs; ++c) {
    for (uint32_t a = 0; a < MaxTevRegs; ++a) {
      auto config = base_config();
      config.tevStages[0].colorOp.outReg = static_cast<GXTevRegID>(c);
      config.tevStages[0].alphaOp.outReg = static_cast<GXTevRegID>(a);
      auto& next = config.tevStages[1];
      next.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, reg_color_arg(c)};
      next.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, reg_alpha_arg(a)};
      config.tevStageCount = 2;
      sink.add(config, fmt::format("tev.outreg/c={}/a={}", c, a));
    }
  }
  for (uint32_t reg = 1; reg < MaxTevRegs; ++reg) {
    auto config = base_config();
    config.tevStages[0].colorOp.outReg = static_cast<GXTevRegID>(reg);
    sink.add(config, fmt::format("tev.outreg/last/c={}", reg));
  }
  for (uint32_t reg = 1; reg < MaxTevRegs; ++reg) {
    auto config = base_config();
    config.tevStages[0].alphaOp.outReg = static_cast<GXTevRegID>(reg);
    sink.add(config, fmt::format("tev.outreg/last/a={}", reg));
  }
}

void tev_konst_family(Sink& sink) {
  for (const auto sel : KColorSels) {
    auto config = base_config();
    config.tevStages[0].colorPass.c = GX_CC_KONST;
    config.tevStages[0].kcSel = sel;
    sink.add(config, fmt::format("tev.konst/kc={}", num(sel)));
  }
  for (const auto sel : KAlphaSels) {
    auto config = base_config();
    config.tevStages[0].alphaPass.c = GX_CA_KONST;
    config.tevStages[0].kaSel = sel;
    sink.add(config, fmt::format("tev.konst/ka={}", num(sel)));
  }
}

void tev_swap_family(Sink& sink) {
  const std::array<TevSwap, MaxTevSwap> table{
      TevSwap{GX_CH_GREEN, GX_CH_BLUE, GX_CH_RED, GX_CH_ALPHA},
      TevSwap{GX_CH_ALPHA, GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE},
      TevSwap{GX_CH_BLUE, GX_CH_ALPHA, GX_CH_RED, GX_CH_GREEN},
      TevSwap{GX_CH_RED, GX_CH_RED, GX_CH_RED, GX_CH_BLUE},
  };
  for (uint32_t ras = 0; ras < MaxTevSwap; ++ras) {
    for (uint32_t tex = 0; tex < MaxTevSwap; ++tex) {
      auto config = base_config();
      config.tevSwapTable = table;
      config.tevStages[0].tevSwapRas = static_cast<GXTevSwapSel>(ras);
      config.tevStages[0].tevSwapTex = static_cast<GXTevSwapSel>(tex);
      sink.add(config, fmt::format("tev.swap/ras={}/tex={}", ras, tex));
    }
  }
}

void tev_channel_family(Sink& sink) {
  for (const auto id : ChannelIds) {
    auto config = base_config();
    config.tevStages[0].channelId = id;
    sink.add(config, fmt::format("tev.channel/id={}", num(id)));
  }
}

void tev_overflow_family(Sink& sink) {
  auto color = base_config();
  color.tevStages[0].colorOp.clamp = false;
  sink.add(color, "tev.overflow/case=color");
  auto alpha = base_config();
  alpha.tevStages[0].alphaOp.clamp = false;
  sink.add(alpha, "tev.overflow/case=alpha");
  auto both = base_config();
  both.tevStages[0].colorOp.clamp = false;
  both.tevStages[0].alphaOp.clamp = false;
  sink.add(both, "tev.overflow/case=both");

  auto regColor = base_config();
  regColor.tevStages[0].colorOp.outReg = GX_TEVREG0;
  regColor.tevStages[0].colorOp.clamp = false;
  regColor.tevStages[1] = textured_stage(0, 0);
  regColor.tevStages[1].colorPass.b = GX_CC_C0;
  regColor.tevStageCount = 2;
  sink.add(regColor, "tev.overflow/case=reg_color");

  auto regAlpha = base_config();
  regAlpha.tevStages[0].alphaOp.outReg = GX_TEVREG0;
  regAlpha.tevStages[0].alphaOp.clamp = false;
  regAlpha.tevStages[1] = textured_stage(0, 0);
  regAlpha.tevStages[1].alphaPass.b = GX_CA_A0;
  regAlpha.tevStageCount = 2;
  sink.add(regAlpha, "tev.overflow/case=reg_alpha");
}

void chan_family(Sink& sink) {
  for (uint32_t ch = 0; ch < 2; ++ch) {
    const auto clr = static_cast<GXAttr>(GX_VA_CLR0 + ch);
    const auto prepare = [&](bool nrm) {
      auto config = base_config();
      if (ch == 1) {
        config.attrs[GX_VA_CLR1] = direct_attr(1, GX_RGBA8);
        config.tevStages[0].channelId = GX_COLOR1A1;
      }
      if (nrm) {
        config.attrs[GX_VA_NRM] = direct_attr(3, GX_F32);
      }
      return config;
    };
    const auto set = [ch](ShaderConfig& config, const ColorChannelConfig& color, const ColorChannelConfig& alpha) {
      config.colorChannels[ch] = color;
      config.colorChannels[ch + GX_ALPHA0] = alpha;
    };
    for (const auto mat : {GX_SRC_REG, GX_SRC_VTX}) {
      auto config = prepare(false);
      set(config, {.matSrc = mat}, {.matSrc = mat});
      sink.add(config, fmt::format("chan/ch={}/unlit/mat={}", ch, num(mat)));
    }
    for (uint32_t diff = 0; diff <= GX_DF_CLAMP; ++diff) {
      for (uint32_t attn = 0; attn <= GX_AF_NONE; ++attn) {
        for (const auto mat : {GX_SRC_REG, GX_SRC_VTX}) {
          for (const auto amb : {GX_SRC_REG, GX_SRC_VTX}) {
            auto config = prepare(true);
            const auto lit = lit_channel(mat, amb, static_cast<GXDiffuseFn>(diff), static_cast<GXAttnFn>(attn));
            set(config, lit, lit);
            sink.add(config,
                     fmt::format("chan/ch={}/lit/diff={}/attn={}/mat={}/amb={}", ch, diff, attn, num(mat), num(amb)));
          }
        }
      }
    }
    const ColorChannelConfig unlit{.matSrc = GX_SRC_VTX};
    auto alphaLit = prepare(true);
    set(alphaLit, unlit, DefaultLit);
    sink.add(alphaLit, fmt::format("chan/ch={}/alpha_lit", ch));
    auto colorLit = prepare(true);
    set(colorLit, DefaultLit, unlit);
    sink.add(colorLit, fmt::format("chan/ch={}/color_lit", ch));
    auto litNoNrm = prepare(false);
    set(litNoNrm, DefaultLit, DefaultLit);
    sink.add(litNoNrm, fmt::format("chan/ch={}/lit_no_nrm", ch));
    auto vtxNoClr = prepare(true);
    vtxNoClr.attrs[clr] = {};
    const auto vtxLit = lit_channel(GX_SRC_VTX, GX_SRC_VTX, GX_DF_CLAMP, GX_AF_SPOT);
    set(vtxNoClr, vtxLit, vtxLit);
    sink.add(vtxNoClr, fmt::format("chan/ch={}/vtx_no_clr", ch));
  }
}

ShaderConfig texgen_src_config(GXTexGenType type, GXTexGenSrc src) {
  auto config = base_config();
  if (is_texcoord_src(src)) {
    // The chained coordinate must be generated first, so it is sampled from coord n by stage 0.
    const uint32_t n = src - GX_TG_TEXCOORD0;
    config.tevStages[0].texCoordId = static_cast<GXTexCoordID>(n);
    config.tcgs[n] = texgen(GX_TG_MTX2x4, GX_TG_TEX0);
    config.tevStages[1] = textured_stage(GX_TEXCOORD7, GX_TEXMAP1);
    config.tcgs[GX_TEXCOORD7] = texgen(type, src);
    config.tevStageCount = 2;
    return config;
  }
  config.tcgs[0] = texgen(type, src);
  if (src == GX_TG_NRM) {
    config.attrs[GX_VA_NRM] = direct_attr(3, GX_F32);
  } else if (src == GX_TG_BINRM || src == GX_TG_TANGENT) {
    config.attrs[GX_VA_NRM] = direct_attr(9, GX_F32);
  } else if (is_tex_src(src)) {
    config.attrs[GX_VA_TEX0 + (src - GX_TG_TEX0)] = direct_attr(2, GX_F32);
  } else if (src == GX_TG_COLOR1) {
    config.attrs[GX_VA_CLR1] = direct_attr(1, GX_RGBA8);
    config.colorChannels[GX_COLOR1] = {.matSrc = GX_SRC_VTX};
    config.colorChannels[GX_ALPHA1] = {.matSrc = GX_SRC_VTX};
    config.tevStages[0].channelId = GX_COLOR1A1;
  }
  return config;
}

void texgen_family(Sink& sink) {
  for (uint32_t src = 0; src < GX_MAX_TEXGENSRC; ++src) {
    for (const auto type : {GX_TG_MTX2x4, GX_TG_MTX3x4}) {
      sink.add(texgen_src_config(type, static_cast<GXTexGenSrc>(src)),
               fmt::format("texgen.src/src={}/type={}", src, num(type)));
    }
  }
  for (const auto src : {GX_TG_COLOR0, GX_TG_COLOR1}) {
    sink.add(texgen_src_config(GX_TG_SRTG, src), fmt::format("texgen.src/src={}/type={}", num(src), num(GX_TG_SRTG)));
  }

  std::vector<GXTexMtx> mtxs{GX_IDENTITY};
  for (uint32_t i = 0; i < MaxTexMtx; ++i) {
    mtxs.push_back(static_cast<GXTexMtx>(GX_TEXMTX0 + i * 3));
  }
  mtxs.push_back(static_cast<GXTexMtx>(GX_PNMTX0));
  mtxs.push_back(static_cast<GXTexMtx>(GX_PNMTX9));
  for (const auto mtx : mtxs) {
    for (const auto type : {GX_TG_MTX2x4, GX_TG_MTX3x4}) {
      auto config = base_config();
      config.tcgs[0] = texgen(type, GX_TG_TEX0, mtx);
      sink.add(config, fmt::format("texgen.mtx/mtx={}/type={}", num(mtx), num(type)));
    }
  }
  for (const auto type : {GX_TG_MTX2x4, GX_TG_MTX3x4}) {
    auto config = base_config();
    config.tcgs[0].type = type;
    config.attrs[GX_VA_TEX0MTXIDX] = direct_attr(1, GX_U8);
    sink.add(config, fmt::format("texgen.mtx/texmtxidx/type={}", num(type)));
  }

  for (const auto post : {GX_PTIDENTITY, GX_PTTEXMTX0, GX_PTTEXMTX1, GX_PTTEXMTX19}) {
    for (const bool normalize : {false, true}) {
      auto config = base_config();
      config.tcgs[0].postMtx = post;
      config.tcgs[0].normalize = normalize;
      sink.add(config, fmt::format("texgen.post/post={}/norm={}", num(post), num(normalize)));
    }
  }
  auto envMap = base_config();
  envMap.attrs[GX_VA_NRM] = direct_attr(3, GX_F32);
  envMap.tcgs[0] = {
      .type = GX_TG_MTX2x4, .src = GX_TG_NRM, .mtx = GX_TEXMTX0, .postMtx = GX_PTTEXMTX0, .normalize = true};
  sink.add(envMap, "texgen.post/envmap");
  for (const auto post : {GX_PTIDENTITY, GX_PTTEXMTX1}) {
    auto config = base_config();
    config.tcgs[0] = texgen(GX_TG_MTX3x4, GX_TG_POS, GX_TEXMTX1, post);
    sink.add(config, fmt::format("texgen.post/proj/post={}", num(post)));
  }

  const auto emboss = [](GXTexGenType bump, const AttrConfig& nrm) {
    auto config = base_config();
    config.attrs[GX_VA_NRM] = nrm;
    config.tcgs[1] = {
        .type = bump, .src = GX_TG_TEXCOORD0, .mtx = GX_IDENTITY, .postMtx = GX_PTIDENTITY, .embossSrc = 0};
    config.tevStages[1] = textured_stage(GX_TEXCOORD1, GX_TEXMAP0);
    config.tevStageCount = 2;
    return config;
  };
  for (uint32_t light = 0; light < GX::MaxLights; ++light) {
    const auto bump = static_cast<GXTexGenType>(GX_TG_BUMP0 + light);
    sink.add(emboss(bump, direct_attr(9, GX_F32)), fmt::format("texgen.emboss/bump={}", num(bump)));
  }
  for (const auto& nrm :
       {make_attr(GX_VA_NRM, GX_DIRECT, 9, GX_S16, false), make_attr(GX_VA_NRM, GX_INDEX8, 9, GX_S16, true),
        make_attr(GX_VA_NRM, GX_INDEX16, 9, GX_S16, true)}) {
    sink.add(emboss(GX_TG_BUMP0, nrm),
             fmt::format("texgen.emboss/nrm_type={}/nrm_comp={}/nbt3={}", nrm.attrType, nrm.compType, num(nrm.nbt3)));
  }
}

ShaderConfig indirect_base() {
  auto config = base_config();
  config.attrs[GX_VA_TEX1] = direct_attr(2, GX_F32);
  config.tcgs[1] = texgen(GX_TG_MTX2x4, GX_TG_TEX1);
  config.indStages[0] = {GX_TEXCOORD1, GX_TEXMAP1, GX_ITS_1, GX_ITS_1};
  config.numIndStages = 1;
  auto& stage = config.tevStages[0];
  stage.indTexStage = GX_INDTEXSTAGE0;
  stage.indTexFormat = GX_ITF_8;
  stage.indTexBiasSel = GX_ITB_STU;
  stage.indTexMtxId = GX_ITM_0;
  return config;
}

void indirect_family(Sink& sink) {
  const auto sweep = [&sink](std::string_view name, uint32_t count, auto&& apply) {
    for (uint32_t value = 0; value < count; ++value) {
      auto config = indirect_base();
      apply(config, value);
      sink.add(config, fmt::format("indirect/{}={}", name, value));
    }
  };
  sweep("fmt", GX_MAX_ITFORMAT,
        [](ShaderConfig& c, uint32_t v) { c.tevStages[0].indTexFormat = static_cast<GXIndTexFormat>(v); });
  sweep("bias", GX_MAX_ITBIAS,
        [](ShaderConfig& c, uint32_t v) { c.tevStages[0].indTexBiasSel = static_cast<GXIndTexBiasSel>(v); });
  for (const auto mtx : IndMtxIds) {
    auto config = indirect_base();
    config.tevStages[0].indTexMtxId = mtx;
    sink.add(config, fmt::format("indirect/mtx={}", num(mtx)));
  }
  sweep("wrap_s", GX_MAX_ITWRAP,
        [](ShaderConfig& c, uint32_t v) { c.tevStages[0].indTexWrapS = static_cast<GXIndTexWrap>(v); });
  sweep("wrap_t", GX_MAX_ITWRAP,
        [](ShaderConfig& c, uint32_t v) { c.tevStages[0].indTexWrapT = static_cast<GXIndTexWrap>(v); });
  auto addPrev = indirect_base();
  addPrev.tevStages[0].indTexAddPrev = true;
  sink.add(addPrev, "indirect/add_prev");
  auto origLod = indirect_base();
  origLod.tevStages[0].indTexUseOrigLOD = true;
  sink.add(origLod, "indirect/orig_lod");
  sweep("scale_s", GX_MAX_ITSCALE,
        [](ShaderConfig& c, uint32_t v) { c.indStages[0].scaleS = static_cast<GXIndTexScale>(v); });
  sweep("scale_t", GX_MAX_ITSCALE,
        [](ShaderConfig& c, uint32_t v) { c.indStages[0].scaleT = static_cast<GXIndTexScale>(v); });
  for (uint32_t n = 1; n <= MaxIndStages; ++n) {
    auto config = indirect_base();
    const auto first = config.tevStages[0];
    for (uint32_t k = 0; k < n; ++k) {
      config.indStages[k] = config.indStages[0];
      auto& stage = config.tevStages[k];
      if (k > 0) {
        stage = textured_stage(GX_TEXCOORD0, GX_TEXMAP0);
        stage.indTexFormat = first.indTexFormat;
        stage.indTexBiasSel = first.indTexBiasSel;
        stage.indTexMtxId = first.indTexMtxId;
      }
      stage.indTexStage = static_cast<GXIndTexStageID>(k);
    }
    config.numIndStages = n;
    config.tevStageCount = n;
    sink.add(config, fmt::format("indirect/stages={}", n));
  }
  auto disabled = indirect_base();
  disabled.tevStages[0].indTexStage = GX_INDTEXSTAGE1;
  sink.add(disabled, "indirect/disabled");
  auto noBaseTex = indirect_base();
  noBaseTex.tevStages[0].texMapId = GX_TEXMAP_NULL;
  sink.add(noBaseTex, "indirect/no_base_tex");
  for (const auto mtx : {GX_ITM_S0, GX_ITM_T0}) {
    auto config = indirect_base();
    config.tevStages[0].texCoordId = GX_TEXCOORD_NULL;
    config.tevStages[0].texMapId = GX_TEXMAP_NULL;
    config.tevStages[0].indTexMtxId = mtx;
    sink.add(config, fmt::format("indirect/no_base_coord/mtx={}", num(mtx)));
  }
  for (uint32_t sel = GX_ITBA_S; sel <= GX_ITBA_U; ++sel) {
    for (const auto channel : {GX_ALPHA_BUMP, GX_ALPHA_BUMPN}) {
      for (uint32_t format = 0; format < GX_MAX_ITFORMAT; ++format) {
        auto config = indirect_base();
        auto& stage = config.tevStages[0];
        stage.channelId = channel;
        stage.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(sel);
        stage.indTexFormat = static_cast<GXIndTexFormat>(format);
        sink.add(config, fmt::format("indirect/alpha={}/chan={}/fmt={}", sel, num(channel), format));
      }
    }
  }
}

void fog_family(Sink& sink) {
  for (const auto type : FogTypes) {
    for (const bool range : {false, true}) {
      auto config = base_config();
      config.fogType = type;
      config.fogRangeEnabled = range;
      sink.add(config, fmt::format("fog/type={}/range={}", type, num(range)));
    }
  }
}

void alpha_family(Sink& sink) {
  const auto add = [&sink](GXCompare comp0, uint32_t ref0, GXAlphaOp op, GXCompare comp1, uint32_t ref1,
                           std::string_view label) {
    auto config = base_config();
    config.alphaCompare = {comp0, ref0, op, comp1, ref1};
    sink.add(config, label);
  };
  for (const auto op : {GX_AOP_AND, GX_AOP_XOR}) {
    for (uint32_t comp0 = 0; comp0 <= GX_ALWAYS; ++comp0) {
      for (uint32_t comp1 = 0; comp1 <= GX_ALWAYS; ++comp1) {
        add(static_cast<GXCompare>(comp0), 64, op, static_cast<GXCompare>(comp1), 192,
            fmt::format("alpha/c0={}/c1={}/op={}", comp0, comp1, num(op)));
      }
    }
  }
  for (const auto op : {GX_AOP_OR, GX_AOP_XNOR}) {
    for (const auto comp0 : {GX_NEVER, GX_GREATER, GX_ALWAYS}) {
      for (const auto comp1 : {GX_NEVER, GX_GREATER, GX_ALWAYS}) {
        add(comp0, 64, op, comp1, 192, fmt::format("alpha/c0={}/c1={}/op={}", num(comp0), num(comp1), num(op)));
      }
    }
  }
  for (const uint32_t ref : {0u, 255u}) {
    for (const auto comp : {GX_LESS, GX_LEQUAL, GX_GEQUAL, GX_GREATER}) {
      for (const auto& [op, other] : {std::pair{GX_AOP_AND, GX_ALWAYS}, std::pair{GX_AOP_OR, GX_NEVER}}) {
        add(comp, ref, op, other, 0, fmt::format("alpha/edge/ref={}/c0={}/op={}", ref, num(comp), num(op)));
      }
    }
  }
}

void line_family(Sink& sink) {
  for (uint32_t mode = 0; mode <= 3; ++mode) {
    for (const bool pnmtx : {false, true}) {
      for (const bool tex : {false, true}) {
        auto config = base_config();
        config.lineMode = static_cast<u8>(mode);
        if (pnmtx) {
          config.attrs[GX_VA_PNMTXIDX] = direct_attr(1, GX_U8);
        }
        if (!tex) {
          config.tevStages[0].texCoordId = GX_TEXCOORD_NULL;
          config.tevStages[0].texMapId = GX_TEXMAP_NULL;
          config.tcgs[0] = {};
          config.attrs[GX_VA_TEX0] = {};
        }
        sink.add(config, fmt::format("line/mode={}/pnmtx={}/tex={}", mode, num(pnmtx), num(tex)));
      }
    }
  }
  for (uint32_t mode = 1; mode <= 3; ++mode) {
    auto config = base_config();
    config.lineMode = static_cast<u8>(mode);
    config.attrs[GX_VA_POS] = make_attr(GX_VA_POS, GX_INDEX16, 3, GX_F32, false);
    sink.add(config, fmt::format("line/mode={}/pos={}", mode, num(GX_INDEX16)));
  }
}

void vtx_family(Sink& sink) {
  constexpr std::array AttrTypes{GX_DIRECT, GX_INDEX8, GX_INDEX16};
  for (const uint32_t cnt : {2u, 3u}) {
    for (uint32_t comp = 0; comp <= GX_F32; ++comp) {
      for (const auto type : AttrTypes) {
        auto config = base_config();
        config.attrs[GX_VA_POS] = make_attr(GX_VA_POS, type, cnt, comp, false);
        sink.add(config, fmt::format("vtx.pos/cnt={}/comp={}/type={}", cnt, comp, num(type)));
      }
    }
  }
  for (const uint32_t frac : {0u, 7u, 14u}) {
    auto config = base_config();
    config.attrs[GX_VA_POS] = direct_attr(3, GX_S16);
    config.attrs[GX_VA_POS].frac = static_cast<u8>(frac);
    sink.add(config, fmt::format("vtx.pos/frac={}", frac));
  }
  for (const bool le : {false, true}) {
    auto config = base_config();
    config.attrs[GX_VA_POS] = make_attr(GX_VA_POS, GX_INDEX16, 3, GX_S16, false);
    config.attrs[GX_VA_POS].le = le;
    sink.add(config, fmt::format("vtx.pos/le={}", num(le)));
  }

  for (const auto form : {GX_NRM_XYZ, GX_NRM_NBT, GX_NRM_NBT3}) {
    for (const auto comp : {GX_S8, GX_S16, GX_F32}) {
      for (const auto type : AttrTypes) {
        auto config = base_config();
        config.colorChannels[GX_COLOR0] = DefaultLit;
        config.attrs[GX_VA_NRM] = make_attr(GX_VA_NRM, type, form == GX_NRM_XYZ ? 3 : 9, comp, form == GX_NRM_NBT3);
        if (form != GX_NRM_XYZ) {
          config.tcgs[1] = texgen(GX_TG_MTX2x4, GX_TG_BINRM);
          config.tcgs[2] = texgen(GX_TG_MTX2x4, GX_TG_TANGENT);
          config.tevStages[1] = textured_stage(GX_TEXCOORD1, GX_TEXMAP1);
          config.tevStages[2] = textured_stage(GX_TEXCOORD2, GX_TEXMAP2);
          config.tevStageCount = 3;
        }
        sink.add(config, fmt::format("vtx.nrm/form={}/comp={}/type={}", num(form), num(comp), num(type)));
      }
    }
  }

  for (const auto attr : {GX_VA_CLR0, GX_VA_CLR1}) {
    for (uint32_t comp = 0; comp <= GX_RGBA8; ++comp) {
      for (const auto type : AttrTypes) {
        auto config = base_config();
        if (attr == GX_VA_CLR1) {
          config.tevStages[0].channelId = GX_COLOR1A1;
          config.colorChannels[GX_COLOR1] = {.matSrc = GX_SRC_VTX};
          config.colorChannels[GX_ALPHA1] = {.matSrc = GX_SRC_VTX};
        }
        config.attrs[attr] = make_attr(attr, type, 1, comp, false);
        sink.add(config, fmt::format("vtx.clr/attr={}/comp={}/type={}", num(attr), comp, num(type)));
      }
    }
  }

  for (const uint32_t cnt : {1u, 2u}) {
    for (uint32_t comp = 0; comp <= GX_F32; ++comp) {
      auto config = base_config();
      config.attrs[GX_VA_TEX0] = direct_attr(cnt, comp);
      sink.add(config, fmt::format("vtx.tex/cnt={}/comp={}", cnt, comp));
    }
  }
  for (const auto type : {GX_INDEX8, GX_INDEX16}) {
    auto config = base_config();
    config.attrs[GX_VA_TEX0] = make_attr(GX_VA_TEX0, type, 2, GX_F32, false);
    sink.add(config, fmt::format("vtx.tex/type={}", num(type)));
  }
  for (uint32_t unit = 1; unit < MaxTexCoord; ++unit) {
    auto config = base_config();
    const auto attr = static_cast<GXAttr>(GX_VA_TEX0 + unit);
    config.attrs[GX_VA_TEX0] = {};
    config.attrs[attr] = direct_attr(2, GX_F32);
    config.tcgs[0].src = tex_src(unit);
    sink.add(config, fmt::format("vtx.tex/attr={}", num(attr)));
  }

  auto pnmtx = base_config();
  pnmtx.attrs[GX_VA_PNMTXIDX] = direct_attr(1, GX_U8);
  pnmtx.attrs[GX_VA_NRM] = direct_attr(3, GX_F32);
  sink.add(pnmtx, "vtx.mtxidx/pnmtx");
  for (uint32_t unit = 0; unit < MaxTexCoord; ++unit) {
    auto config = base_config();
    config.attrs[GX_VA_TEX0MTXIDX + unit] = direct_attr(1, GX_U8);
    config.tcgs[unit] = texgen(GX_TG_MTX2x4, GX_TG_TEX0);
    config.tevStages[0].texCoordId = static_cast<GXTexCoordID>(unit);
    sink.add(config, fmt::format("vtx.mtxidx/tex={}", unit));
  }
}

void random_op(SplitMix64& rng, TevOp& op, bool last) {
  op.op = RandomTevOps[rng.below(RandomTevOps.size())];
  op.bias = static_cast<GXTevBias>(rng.below(3));
  op.scale = static_cast<GXTevScale>(rng.below(4));
  op.clamp = rng.below(4) != 0;
  if (last) {
    const bool prev = rng.below(4) != 0;
    op.outReg = prev ? GX_TEVPREV : static_cast<GXTevRegID>(1 + rng.below(3));
  } else {
    op.outReg = static_cast<GXTevRegID>(rng.below(4));
  }
}

// Every draw is its own statement: argument evaluation order differs between compilers.
ShaderConfig random_config(SplitMix64& rng, Variant& variant) {
  ShaderConfig config{};
  const uint32_t stageCount = 1 + rng.below(MaxTevStages);
  for (auto& swap : config.tevSwapTable) {
    swap.red = static_cast<GXTevColorChan>(rng.below(4));
    swap.green = static_cast<GXTevColorChan>(rng.below(4));
    swap.blue = static_cast<GXTevColorChan>(rng.below(4));
    swap.alpha = static_cast<GXTevColorChan>(rng.below(4));
  }
  const uint32_t numInd = rng.below(3);
  for (uint32_t i = 0; i < numInd; ++i) {
    auto& ind = config.indStages[i];
    ind.texCoordId = static_cast<GXTexCoordID>(rng.below(MaxTexCoord));
    ind.texMapId = static_cast<GXTexMapID>(rng.below(MaxTextures));
    ind.scaleS = static_cast<GXIndTexScale>(rng.below(GX_MAX_ITSCALE));
    ind.scaleT = static_cast<GXIndTexScale>(rng.below(GX_MAX_ITSCALE));
  }
  config.numIndStages = numInd;
  for (uint32_t i = 0; i < stageCount; ++i) {
    auto& stage = config.tevStages[i];
    const bool last = i + 1 == stageCount;
    for (uint32_t slot = 0; slot < 4; ++slot) {
      pass_slot(stage.colorPass, slot) = static_cast<GXTevColorArg>(rng.below(GX_CC_ZERO + 1));
    }
    for (uint32_t slot = 0; slot < 4; ++slot) {
      pass_slot(stage.alphaPass, slot) = static_cast<GXTevAlphaArg>(rng.below(GX_CA_ZERO + 1));
    }
    random_op(rng, stage.colorOp, last);
    random_op(rng, stage.alphaOp, last);
    stage.kcSel = KColorSels[rng.below(KColorSels.size())];
    stage.kaSel = KAlphaSels[rng.below(KAlphaSels.size())];
    const uint32_t map = rng.below(MaxTextures + 1);
    stage.texMapId = map == MaxTextures ? GX_TEXMAP_NULL : static_cast<GXTexMapID>(map);
    stage.texCoordId = static_cast<GXTexCoordID>(rng.below(MaxTexCoord));
    stage.channelId = ChannelIds[rng.below(ChannelIds.size())];
    stage.tevSwapRas = static_cast<GXTevSwapSel>(rng.below(MaxTevSwap));
    stage.tevSwapTex = static_cast<GXTevSwapSel>(rng.below(MaxTevSwap));
    if (numInd > 0) {
      const bool indirect = rng.below(4) == 0;
      if (indirect) {
        stage.indTexStage = static_cast<GXIndTexStageID>(rng.below(numInd));
        stage.indTexFormat = static_cast<GXIndTexFormat>(rng.below(GX_MAX_ITFORMAT));
        stage.indTexBiasSel = static_cast<GXIndTexBiasSel>(rng.below(GX_MAX_ITBIAS));
        stage.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(rng.below(GX_MAX_ITBALPHA));
        stage.indTexMtxId = IndMtxIds[rng.below(IndMtxIds.size())];
        stage.indTexWrapS = static_cast<GXIndTexWrap>(rng.below(GX_MAX_ITWRAP));
        stage.indTexWrapT = static_cast<GXIndTexWrap>(rng.below(GX_MAX_ITWRAP));
        stage.indTexUseOrigLOD = rng.below(2) != 0;
        stage.indTexAddPrev = rng.below(2) != 0;
      }
    }
  }
  config.tevStageCount = stageCount;

  bool needsNbt = false;
  for (uint32_t i = 0; i < MaxTexCoord; ++i) {
    auto& tcg = config.tcgs[i];
    auto type = static_cast<GXTexGenType>(rng.below(GX_TG_SRTG + 1));
    if (i == 0 && is_bump(type)) {
      type = GX_TG_MTX2x4;
    }
    tcg.type = type;
    auto src = static_cast<GXTexGenSrc>(rng.below(GX_MAX_TEXGENSRC));
    // A texcoord can only chain from an earlier one, otherwise it reads the matching vertex attribute.
    if (is_texcoord_src(src) && static_cast<uint32_t>(src - GX_TG_TEXCOORD0) >= i) {
      src = tex_src(src - GX_TG_TEXCOORD0);
    }
    tcg.src = src;
    tcg.mtx = static_cast<GXTexMtx>(3 * rng.below(21));
    const uint32_t post = rng.below(MaxPTTexMtx + 1);
    tcg.postMtx = post == MaxPTTexMtx ? GX_PTIDENTITY : static_cast<GXPTTexMtx>(GX_PTTEXMTX0 + 3 * post);
    tcg.normalize = rng.below(2) != 0;
    tcg.embossSrc = i == 0 ? 0 : static_cast<u8>(rng.below(i));
    needsNbt = needsNbt || src == GX_TG_BINRM || src == GX_TG_TANGENT || is_bump(type);
  }

  for (const auto& tcg : config.tcgs) {
    if (!is_tex_src(tcg.src)) {
      continue;
    }
    const auto attr = static_cast<GXAttr>(GX_VA_TEX0 + (tcg.src - GX_TG_TEX0));
    if (config.attrs[attr].attrType != GX_NONE) {
      continue;
    }
    const uint32_t type = 1 + rng.below(3);
    const uint32_t cnt = 1 + rng.below(2);
    const uint32_t comp = rng.below(GX_F32 + 1);
    config.attrs[attr] = make_attr(attr, type, cnt, comp, false);
  }
  const uint32_t posType = 1 + rng.below(3);
  const uint32_t posCnt = 2 + rng.below(2);
  const uint32_t posComp = rng.below(GX_F32 + 1);
  config.attrs[GX_VA_POS] = make_attr(GX_VA_POS, posType, posCnt, posComp, false);
  // Form 3 leaves NRM absent; NBT forms are forced when a texgen needs binormals.
  const uint32_t nrmForm = needsNbt ? 1 + rng.below(2) : rng.below(4);
  if (nrmForm < 3) {
    const uint32_t nrmType = 1 + rng.below(3);
    const uint32_t nrmComp = rng.below(GX_F32 + 1);
    config.attrs[GX_VA_NRM] = make_attr(GX_VA_NRM, nrmType, nrmForm == 0 ? 3 : 9, nrmComp, nrmForm == 2);
  }
  for (const auto attr : {GX_VA_CLR0, GX_VA_CLR1}) {
    const uint32_t type = rng.below(4);
    if (type == GX_NONE) {
      continue;
    }
    const uint32_t comp = rng.below(GX_RGBA8 + 1);
    config.attrs[attr] = make_attr(attr, type, 1, comp, false);
  }
  const bool pnmtx = rng.below(2) != 0;
  if (pnmtx) {
    config.attrs[GX_VA_PNMTXIDX] = direct_attr(1, GX_U8);
  }
  for (uint32_t unit = 0; unit < MaxTexCoord; ++unit) {
    const bool present = rng.below(4) == 0;
    if (present) {
      config.attrs[GX_VA_TEX0MTXIDX + unit] = direct_attr(1, GX_U8);
    }
  }

  for (auto& channel : config.colorChannels) {
    channel.matSrc = static_cast<GXColorSrc>(rng.below(2));
    channel.ambSrc = static_cast<GXColorSrc>(rng.below(2));
    channel.diffFn = static_cast<GXDiffuseFn>(rng.below(3));
    channel.attnFn = static_cast<GXAttnFn>(rng.below(3));
    channel.lightingEnabled = rng.below(2) != 0;
  }
  config.fogType = FogTypes[rng.below(FogTypes.size())];
  const bool fogRange = rng.below(2) != 0;
  config.fogRangeEnabled = fogRange;
  config.alphaCompare.comp0 = static_cast<GXCompare>(rng.below(GX_ALWAYS + 1));
  config.alphaCompare.ref0 = rng.below(256);
  config.alphaCompare.op = static_cast<GXAlphaOp>(rng.below(GX_MAX_ALPHAOP));
  config.alphaCompare.comp1 = static_cast<GXCompare>(rng.below(GX_ALWAYS + 1));
  config.alphaCompare.ref1 = rng.below(256);
  const bool line = rng.below(5) == 0;
  if (line) {
    config.lineMode = static_cast<u8>(1 + rng.below(3));
  }
  variant = Variants[rng.below(Variants.size())].variant;
  return config;
}

void random_family(Sink& sink) {
  SplitMix64 rng{RandomSeed};
  uint32_t accepted = 0;
  for (uint32_t attempt = 0; attempt < RandomAttempts && accepted < RandomCount; ++attempt) {
    Variant variant;
    auto config = random_config(rng, variant);
    canonicalize(config);
    const auto record = encode(config, variant);
    if (!record || validate(*record)) {
      continue;
    }
    if (sink.add(config, fmt::format("random/n={}", accepted), variant)) {
      ++accepted;
    }
  }
}

class ByteWriter {
public:
  void u8(uint8_t value) { m_bytes.push_back(value); }
  void u16(uint16_t value) { put(value, 2); }
  void u32(uint32_t value) { put(value, 4); }
  void raw(std::span<const uint8_t> bytes) { m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end()); }
  void raw(std::string_view text) { m_bytes.insert(m_bytes.end(), text.begin(), text.end()); }
  const std::vector<uint8_t>& bytes() const { return m_bytes; }

private:
  void put(uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
      m_bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
  }

  std::vector<uint8_t> m_bytes;
};

class ByteReader {
public:
  explicit ByteReader(std::span<const uint8_t> data) : m_data(data) {}

  template <typename T>
  bool read(T& out) {
    if (remaining() < sizeof(T)) {
      return false;
    }
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
      value |= static_cast<uint64_t>(m_data[m_pos + i]) << (8 * i);
    }
    m_pos += sizeof(T);
    out = static_cast<T>(value);
    return true;
  }

  bool take(size_t size, std::span<const uint8_t>& out) {
    if (remaining() < size) {
      return false;
    }
    out = m_data.subspan(m_pos, size);
    m_pos += size;
    return true;
  }

  size_t remaining() const { return m_data.size() - m_pos; }

private:
  std::span<const uint8_t> m_data;
  size_t m_pos = 0;
};

std::optional<std::vector<uint8_t>> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::nullopt;
  }
  std::vector<uint8_t> data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (file.bad()) {
    return std::nullopt;
  }
  return data;
}

bool write_file(const std::filesystem::path& path, std::span<const uint8_t> data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    return false;
  }
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(file);
}

template <typename T>
std::optional<T> fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return std::nullopt;
}

std::optional<Record> read_record_v1(ByteReader& in) {
  std::span<const uint8_t> bytes;
  if (!in.take(RecordSize, bytes)) {
    return std::nullopt;
  }
  Record record;
  std::copy(bytes.begin(), bytes.end(), record.bytes.begin());
  return record;
}

std::optional<uint64_t> parse_hex(std::string_view text) {
  if (text.size() != 16) {
    return std::nullopt;
  }
  uint64_t value = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, 16);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> parse_decimal(std::string_view text) {
  uint64_t value = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || ec != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}
} // namespace

std::optional<Record> encode(const ShaderConfig& config, Variant variant) noexcept {
  Record record;
  size_t pos = 0;
  bool ok = true;
  const auto put = [&](const auto& value) {
    const uint64_t raw = raw_value(value);
    ok = ok && raw <= 0xFF;
    record.bytes[pos++] = static_cast<uint8_t>(raw);
  };
  put(config.fogType);
  put(config.vtxStride);
  put(static_cast<u8>(config.lineMode));
  put(static_cast<u8>(config.fogRangeEnabled));
  put(config.tevStageCount);
  put(config.numIndStages);
  put(variant.dstAlphaMode);
  // 0xFF is reserved for UINT32_MAX, so a real attachment index of 255 cannot be stored.
  ok = ok && variant.normalAttachment != NoAttachment;
  put(variant.normalAttachment == UINT32_MAX ? uint32_t{NoAttachment} : variant.normalAttachment);
  visit_fields(config, [&](const auto& field) { put(field); });
  if (!ok) {
    return std::nullopt;
  }
  return record;
}

ShaderConfig decode(const Record& record) noexcept {
  const auto& bytes = record.bytes;
  ShaderConfig config{};
  config.fogType = bytes[0];
  config.vtxStride = bytes[1];
  config.lineMode = static_cast<u8>(bytes[RecordLineMode] & 3);
  config.fogRangeEnabled = bytes[RecordFogRangeEnabled] != 0;
  config.tevStageCount = bytes[4];
  config.numIndStages = bytes[5];
  size_t pos = RecordAttrsOffset;
  visit_fields(config, [&](auto& field) {
    using T = std::remove_reference_t<decltype(field)>;
    const uint8_t byte = bytes[pos++];
    if constexpr (std::is_same_v<T, bool>) {
      field = byte != 0;
    } else {
      field = static_cast<T>(byte);
    }
  });
  return config;
}

Variant decode_variant(const Record& record) noexcept {
  const uint8_t attachment = record.bytes[RecordNormalAttachment];
  return {
      .dstAlphaMode = static_cast<DstAlphaMode>(record.bytes[RecordDstAlphaMode]),
      .normalAttachment = attachment == NoAttachment ? UINT32_MAX : attachment,
  };
}

std::optional<std::string> validate(const Record& record) {
  const auto& bytes = record.bytes;
  for (size_t i = RecordReservedOffset; i < RecordSize; ++i) {
    if (bytes[i] != 0) {
      return fmt::format("reserved byte {} is {}", i, bytes[i]);
    }
  }
  const auto bad_bool = [&](size_t offset) { return bytes[offset] > 1; };
  if (bad_bool(RecordFogRangeEnabled)) {
    return "fogRangeEnabled is not 0 or 1";
  }
  for (size_t i = 0; i < MaxVtxAttr; ++i) {
    const size_t base = RecordAttrsOffset + i * RecordAttrStride;
    if (bad_bool(base + AttrLe) || bad_bool(base + AttrNbt3)) {
      return fmt::format("attr {} le/nbt3 is not 0 or 1", i);
    }
  }
  for (size_t i = 0; i < MaxTevStages; ++i) {
    const size_t base = RecordStagesOffset + i * RecordStageStride;
    if (bad_bool(base + StageColorClamp) || bad_bool(base + StageAlphaClamp) || bad_bool(base + StageUseOrigLod) ||
        bad_bool(base + StageAddPrev)) {
      return fmt::format("stage {} has a bool that is not 0 or 1", i);
    }
  }
  for (size_t i = 0; i < MaxColorChannels; ++i) {
    if (bad_bool(RecordChannelsOffset + i * RecordChannelStride + ChannelLightingEnabled)) {
      return fmt::format("channel {} lightingEnabled is not 0 or 1", i);
    }
  }
  for (size_t i = 0; i < MaxTexCoord; ++i) {
    if (bad_bool(RecordTcgsOffset + i * RecordTcgStride + TcgNormalize)) {
      return fmt::format("tcg {} normalize is not 0 or 1", i);
    }
  }
  if (bytes[RecordLineMode] > 3) {
    return fmt::format("lineMode {} out of range", bytes[RecordLineMode]);
  }
  if (bytes[RecordDstAlphaMode] > num(DstAlphaMode::DualSource)) {
    return fmt::format("dstAlphaMode {} out of range", bytes[RecordDstAlphaMode]);
  }
  if (bytes[RecordNormalAttachment] != NoAttachment && bytes[RecordNormalAttachment] != 1) {
    return fmt::format("normalAttachment {} is not 0xFF or 1", bytes[RecordNormalAttachment]);
  }
  // pipeline.cpp only selects DualSource for single-attachment layouts.
  if (bytes[RecordDstAlphaMode] == num(DstAlphaMode::DualSource) && bytes[RecordNormalAttachment] != NoAttachment) {
    return "DualSource requires normalAttachment 0xFF";
  }

  const auto config = decode(record);
  if (config.tevStageCount < 1 || config.tevStageCount > MaxTevStages) {
    return fmt::format("tevStageCount {} out of range", config.tevStageCount);
  }
  if (config.numIndStages > MaxIndStages) {
    return fmt::format("numIndStages {} out of range", config.numIndStages);
  }
  if (!valid_fog(config.fogType)) {
    return fmt::format("fogType {} is invalid", config.fogType);
  }

  for (size_t i = GX_VA_TEX7 + 1; i < MaxVtxAttr; ++i) {
    if (!same_attr(config.attrs[i], AttrConfig{})) {
      return fmt::format("attr {} must be unset", i);
    }
  }
  if (config.attrs[GX_VA_POS].attrType == GX_NONE) {
    return "POS is absent";
  }
  for (uint32_t i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attr = config.attrs[i];
    if (attr.attrType > GX_INDEX16) {
      return fmt::format("attr {} type {} out of range", i, attr.attrType);
    }
    if (attr.attrType == GX_NONE) {
      continue;
    }
    if (i <= GX_VA_TEX7MTXIDX) {
      if (attr.attrType != GX_DIRECT || attr.cnt != 1) {
        return fmt::format("matrix index attr {} must be DIRECT with cnt 1", i);
      }
      continue;
    }
    bool cntOk = false;
    uint32_t maxComp = GX_F32;
    if (i == GX_VA_POS) {
      cntOk = attr.cnt == 2 || attr.cnt == 3;
    } else if (i == GX_VA_NRM) {
      cntOk = attr.cnt == 3 || attr.cnt == 9;
    } else if (i == GX_VA_CLR0 || i == GX_VA_CLR1) {
      cntOk = attr.cnt == 1;
      maxComp = GX_RGBA8;
    } else {
      cntOk = attr.cnt == 1 || attr.cnt == 2;
    }
    if (!cntOk) {
      return fmt::format("attr {} cnt {} is invalid", i, attr.cnt);
    }
    if (attr.compType > maxComp) {
      return fmt::format("attr {} compType {} is invalid", i, attr.compType);
    }
  }

  for (uint32_t i = 0; i < MaxTevStages; ++i) {
    const auto& stage = config.tevStages[i];
    for (uint32_t slot = 0; slot < 4; ++slot) {
      if (num(pass_slot(stage.colorPass, slot)) > GX_CC_ZERO) {
        return fmt::format("stage {} colour arg {} out of range", i, slot);
      }
      if (num(pass_slot(stage.alphaPass, slot)) > GX_CA_ZERO) {
        return fmt::format("stage {} alpha arg {} out of range", i, slot);
      }
    }
    for (const auto* op : {&stage.colorOp, &stage.alphaOp}) {
      if (!valid_op(op->op)) {
        return fmt::format("stage {} op {} is invalid", i, num(op->op));
      }
      if (num(op->bias) > GX_TB_SUBHALF) {
        return fmt::format("stage {} bias {} out of range", i, num(op->bias));
      }
      if (num(op->scale) > GX_CS_DIVIDE_2) {
        return fmt::format("stage {} scale {} out of range", i, num(op->scale));
      }
      if (num(op->outReg) > GX_TEVREG2) {
        return fmt::format("stage {} outReg {} out of range", i, num(op->outReg));
      }
    }
    if (!valid_kcsel(num(stage.kcSel))) {
      return fmt::format("stage {} kcSel {} is invalid", i, num(stage.kcSel));
    }
    if (!valid_kasel(num(stage.kaSel))) {
      return fmt::format("stage {} kaSel {} is invalid", i, num(stage.kaSel));
    }
    if (!valid_slot(num(stage.texMapId), GX_TEXMAP7) || !valid_slot(num(stage.texCoordId), GX_TEXCOORD7)) {
      return fmt::format("stage {} texture slots out of range", i);
    }
    if (!valid_slot(num(stage.channelId), GX_ALPHA_BUMPN)) {
      return fmt::format("stage {} channelId {} is invalid", i, num(stage.channelId));
    }
    if (num(stage.tevSwapRas) > GX_TEV_SWAP3 || num(stage.tevSwapTex) > GX_TEV_SWAP3) {
      return fmt::format("stage {} swap select out of range", i);
    }
    if (num(stage.indTexStage) > GX_INDTEXSTAGE3 || num(stage.indTexFormat) > GX_ITF_3 ||
        num(stage.indTexBiasSel) > GX_ITB_STU || num(stage.indTexAlphaSel) > GX_ITBA_U ||
        !valid_ind_mtx(stage.indTexMtxId) || num(stage.indTexWrapS) > GX_ITW_0 || num(stage.indTexWrapT) > GX_ITW_0) {
      return fmt::format("stage {} indirect fields out of range", i);
    }
    if (i < config.tevStageCount && uses_texture_arg(stage) && stage.texMapId != GX_TEXMAP_NULL &&
        stage.texCoordId == GX_TEXCOORD_NULL) {
      return fmt::format("stage {} samples a texture without a texcoord", i);
    }
  }

  for (uint32_t i = 0; i < MaxIndStages; ++i) {
    const auto& ind = config.indStages[i];
    if (num(ind.texCoordId) > GX_TEXCOORD7 || num(ind.texMapId) > GX_TEXMAP7 || num(ind.scaleS) > GX_ITS_256 ||
        num(ind.scaleT) > GX_ITS_256) {
      return fmt::format("indirect stage {} out of range", i);
    }
  }
  for (const auto& swap : config.tevSwapTable) {
    if (num(swap.red) > GX_CH_ALPHA || num(swap.green) > GX_CH_ALPHA || num(swap.blue) > GX_CH_ALPHA ||
        num(swap.alpha) > GX_CH_ALPHA) {
      return "swap table entry out of range";
    }
  }
  for (uint32_t i = 0; i < MaxColorChannels; ++i) {
    const auto& channel = config.colorChannels[i];
    if (num(channel.matSrc) > GX_SRC_VTX || num(channel.ambSrc) > GX_SRC_VTX || num(channel.diffFn) > GX_DF_CLAMP ||
        num(channel.attnFn) > GX_AF_NONE) {
      return fmt::format("channel {} out of range", i);
    }
  }
  for (uint32_t i = 0; i < MaxTexCoord; ++i) {
    const auto& tcg = config.tcgs[i];
    const uint32_t mtx = num(tcg.mtx);
    const uint32_t post = num(tcg.postMtx);
    const bool postOk =
        post == GX_PTIDENTITY || (post >= GX_PTTEXMTX0 && post <= GX_PTTEXMTX19 && (post - GX_PTTEXMTX0) % 3 == 0);
    if (num(tcg.type) > GX_TG_SRTG || num(tcg.src) > GX_MAX_TEXGENSRC || mtx > GX_IDENTITY || mtx % 3 != 0 || !postOk ||
        tcg.embossSrc > GX_TEXCOORD7) {
      return fmt::format("tcg {} out of range", i);
    }
  }
  const auto& ac = config.alphaCompare;
  if (num(ac.comp0) > GX_ALWAYS || num(ac.comp1) > GX_ALWAYS || num(ac.op) > GX_AOP_XNOR) {
    return "alpha compare out of range";
  }

  // Ranges are now known to be safe, so the real dependency analysis can run.
  const auto info = build_shader_info(config);
  const auto& nrm = config.attrs[GX_VA_NRM];
  for (uint32_t i = 0; i < MaxTexCoord; ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    const bool emboss = is_bump(tcg.type);
    if (!emboss && tcg.src > GX_TG_COLOR1) {
      return fmt::format("texcoord {} samples tcg src {}", i, num(tcg.src));
    }
    if (!emboss && is_tex_src(tcg.src) && config.attrs[GX_VA_TEX0 + (tcg.src - GX_TG_TEX0)].attrType == GX_NONE) {
      return fmt::format("texcoord {} reads TEX{} which is absent", i, tcg.src - GX_TG_TEX0);
    }
    if ((tcg.src == GX_TG_BINRM || tcg.src == GX_TG_TANGENT || emboss) && (nrm.attrType == GX_NONE || nrm.cnt != 9)) {
      return fmt::format("texcoord {} needs NBT normals", i);
    }
  }
  return std::nullopt;
}

uint64_t record_key(const Record& record) noexcept { return XXH3_64bits(record.bytes.data(), record.bytes.size()); }

uint64_t text_hash(std::string_view text) noexcept { return XXH3_64bits(text.data(), text.size()); }

std::string generate_wgsl(const Record& record) {
  const auto variant = decode_variant(record);
  return build_shader_source(decode(record), variant.dstAlphaMode, variant.normalAttachment);
}

void canonicalize(ShaderConfig& config) noexcept {
  for (auto& channel : config.colorChannels) {
    if (!channel.lightingEnabled) {
      channel = {.matSrc = channel.matSrc};
    }
  }
  if (config.alphaCompare.comp0 == GX_ALWAYS && config.alphaCompare.comp1 == GX_ALWAYS) {
    config.alphaCompare = AlphaCompare{};
  }
  for (uint32_t i = config.tevStageCount; i < MaxTevStages; ++i) {
    config.tevStages[i] = TevStage{};
  }
  for (uint32_t i = config.numIndStages; i < MaxIndStages; ++i) {
    config.indStages[i] = IndStage{};
  }
  u8 offset = 0;
  for (uint32_t i = 0; i < MaxVtxAttr; ++i) {
    auto& attr = config.attrs[i];
    if (attr.attrType == GX_NONE || i > GX_VA_TEX7) {
      attr = {};
      continue;
    }
    attr.offset = offset;
    switch (attr.attrType) {
    case GX_DIRECT:
      attr.stride = 0;
      attr.le = false;
      offset += comp_type_size(static_cast<GXAttr>(i), static_cast<GXCompType>(attr.compType)) * attr.cnt;
      break;
    case GX_INDEX8:
      offset += attr.nbt3 ? 3 : 1;
      break;
    case GX_INDEX16:
      offset += attr.nbt3 ? 6 : 2;
      break;
    default:
      break;
    }
  }
  config.vtxStride = offset;
}

std::vector<Entry> generate_synthetic() {
  Sink sink;
  variant_family(sink);
  tev_count_family(sink);
  tev_arg_family(sink);
  tev_op_family(sink);
  tev_outreg_family(sink);
  tev_konst_family(sink);
  tev_swap_family(sink);
  tev_channel_family(sink);
  tev_overflow_family(sink);
  chan_family(sink);
  texgen_family(sink);
  indirect_family(sink);
  fog_family(sink);
  alpha_family(sink);
  line_family(sink);
  vtx_family(sink);
  random_family(sink);
  return sink.take();
}

std::vector<Variant> pipeline_variants(const PipelineConfig& config) {
  constexpr Variant NoneFF{DstAlphaMode::None, UINT32_MAX};
  constexpr Variant None1{DstAlphaMode::None, 1};
  constexpr Variant ReplaceFF{DstAlphaMode::Replace, UINT32_MAX};
  constexpr Variant Replace1{DstAlphaMode::Replace, 1};
  constexpr Variant DualFF{DstAlphaMode::DualSource, UINT32_MAX};
  const bool dstAlpha = config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0;
  if (!dstAlpha) {
    return {NoneFF, None1};
  }
  const auto usesSourceAlpha = [](GXBlendFactor factor) {
    return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
  };
  const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                (usesSourceAlpha(config.blendFacSrc) || usesSourceAlpha(config.blendFacDst));
  if (!needsSourceAlpha) {
    return {ReplaceFF, Replace1};
  }
  // Device support and layout decide between dual source and the prepass, so keep every outcome.
  return {DualFF, ReplaceFF, NoneFF, Replace1, None1};
}

ImportStats import_blob(const Blob& blob, std::vector<Entry>& out) {
  ImportStats stats;
  if (blob.configVersion != GXPipelineConfigVersion) {
    ++stats.wrongVersion;
    return stats;
  }
  if (blob.bytes.size() != sizeof(PipelineConfig)) {
    ++stats.wrongSize;
    return stats;
  }
  uint32_t leading = 0;
  std::memcpy(&leading, blob.bytes.data(), sizeof(leading));
  if (leading != blob.configVersion) {
    ++stats.wrongVersion;
    return stats;
  }
  for (const size_t offset : {offsetof(PipelineConfig, depthCompare), offsetof(PipelineConfig, depthUpdate),
                              offsetof(PipelineConfig, alphaUpdate), offsetof(PipelineConfig, colorUpdate)}) {
    if (blob.bytes[offset] > 1) {
      ++stats.invalid;
      return stats;
    }
  }
  PipelineConfig config;
  std::memcpy(&config, blob.bytes.data(), sizeof(config));
  for (const auto& variant : pipeline_variants(config)) {
    const auto record = encode(config.shaderConfig, variant);
    if (!record || !(decode(*record) == config.shaderConfig)) {
      ++stats.notLegacy;
      continue;
    }
    if (validate(*record)) {
      ++stats.invalid;
      continue;
    }
    const bool known = std::any_of(out.begin(), out.end(), [&](const Entry& entry) { return entry.record == *record; });
    if (known) {
      ++stats.duplicate;
      continue;
    }
    out.push_back({*record, fmt::format("tp/v{}/{:016x}@{}", blob.configVersion, blob.dbHash, variant_name(variant))});
    ++stats.accepted;
  }
  return stats;
}

void sort_and_dedupe(std::vector<Entry>& entries) {
  std::vector<std::pair<uint64_t, size_t>> order;
  order.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    order.emplace_back(record_key(entries[i].record), i);
  }
  std::sort(order.begin(), order.end());
  std::vector<Entry> sorted;
  sorted.reserve(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    if (i > 0 && order[i].first == order[i - 1].first) {
      continue;
    }
    sorted.push_back(std::move(entries[order[i].second]));
  }
  entries = std::move(sorted);
}

std::vector<GoldenLine> golden_lines(std::span<const Entry> entries) {
  std::vector<GoldenLine> lines;
  lines.reserve(entries.size());
  for (const auto& entry : entries) {
    // generate_wgsl is noexcept and aborts on bad input, so stop with a readable reason instead.
    if (const auto reason = validate(entry.record)) {
      fmt::print(stderr, "gx shader corpus: {} is invalid: {}\n", entry.label, *reason);
      std::abort();
    }
    const auto text = generate_wgsl(entry.record);
    lines.push_back({record_key(entry.record), text_hash(text), text.size(), entry.label});
  }
  std::stable_sort(lines.begin(), lines.end(), [](const auto& lhs, const auto& rhs) { return lhs.key < rhs.key; });
  const auto last =
      std::unique(lines.begin(), lines.end(), [](const auto& lhs, const auto& rhs) { return lhs.key == rhs.key; });
  lines.erase(last, lines.end());
  return lines;
}

std::string format_golden(std::span<const GoldenLine> lines) {
  std::string text{GoldenHeader};
  text += '\n';
  for (const auto& line : lines) {
    text += fmt::format("{:016x} {:016x} {} {}\n", line.key, line.wgslHash, line.wgslSize, line.label);
  }
  return text;
}

std::optional<std::vector<GoldenLine>> parse_golden(std::string_view text, std::string* error) {
  using Result = std::vector<GoldenLine>;
  Result lines;
  size_t lineNumber = 0;
  bool sawHeader = false;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    ++lineNumber;
    // core.autocrlf checkouts carry CR before every LF.
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (!sawHeader) {
      if (line != GoldenHeader) {
        return fail<Result>(error, "missing or wrong header");
      }
      sawHeader = true;
      continue;
    }
    std::array<std::string_view, 4> fields;
    size_t count = 0;
    while (!line.empty() && count < fields.size()) {
      const size_t space = count + 1 < fields.size() ? line.find(' ') : std::string_view::npos;
      fields[count++] = line.substr(0, space);
      line = space == std::string_view::npos ? std::string_view{} : line.substr(space + 1);
    }
    if (count != fields.size() || !line.empty()) {
      return fail<Result>(error, fmt::format("line {}: expected 4 fields", lineNumber));
    }
    const auto key = parse_hex(fields[0]);
    const auto hash = parse_hex(fields[1]);
    const auto size = parse_decimal(fields[2]);
    if (!key || !hash || !size) {
      return fail<Result>(error, fmt::format("line {}: malformed numbers", lineNumber));
    }
    if (!valid_label(fields[3])) {
      return fail<Result>(error, fmt::format("line {}: invalid label", lineNumber));
    }
    if (!lines.empty() && *key <= lines.back().key) {
      return fail<Result>(error, fmt::format("line {}: keys are not strictly ascending", lineNumber));
    }
    lines.push_back({*key, *hash, *size, std::string{fields[3]}});
  }
  if (!sawHeader) {
    return fail<Result>(error, "missing or wrong header");
  }
  return lines;
}

bool write_corpus(const std::filesystem::path& path, std::span<const Entry> entries) {
  std::vector<Entry> sorted(entries.begin(), entries.end());
  sort_and_dedupe(sorted);
  ByteWriter out;
  out.raw("GXSC"sv);
  out.u16(CorpusFormatVersion);
  out.u16(static_cast<uint16_t>(RecordSize));
  out.u32(static_cast<uint32_t>(sorted.size()));
  out.u32(0);
  for (const auto& entry : sorted) {
    if (!valid_label(entry.label)) {
      return false;
    }
    out.u8(static_cast<uint8_t>(entry.label.size()));
    out.raw(entry.label);
    out.raw(entry.record.bytes);
  }
  return write_file(path, out.bytes());
}

std::optional<std::vector<Entry>> read_corpus(const std::filesystem::path& path, std::string* error) {
  using Result = std::vector<Entry>;
  const auto data = read_file(path);
  if (!data) {
    return fail<Result>(error, fmt::format("cannot read {}", path.string()));
  }
  ByteReader in{*data};
  std::span<const uint8_t> magic;
  uint16_t version = 0;
  uint16_t recordSize = 0;
  uint32_t count = 0;
  uint32_t reserved = 0;
  if (!in.take(4, magic) || !in.read(version) || !in.read(recordSize) || !in.read(count) || !in.read(reserved)) {
    return fail<Result>(error, "truncated header");
  }
  if (!std::equal(magic.begin(), magic.end(), "GXSC")) {
    return fail<Result>(error, "bad magic");
  }
  if (reserved != 0) {
    return fail<Result>(error, "reserved header field is not zero");
  }
  // A future format must keep reading v1 and converting it here, or the local TP corpus is lost.
  if (version != 1) {
    return fail<Result>(error, fmt::format("unsupported corpus format {}", version));
  }
  if (recordSize != RecordSize) {
    return fail<Result>(error, fmt::format("record size {} does not match format {}", recordSize, version));
  }
  Result entries;
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t labelSize = 0;
    std::span<const uint8_t> label;
    if (!in.read(labelSize) || !in.take(labelSize, label)) {
      return fail<Result>(error, fmt::format("truncated at entry {}", i));
    }
    const auto record = read_record_v1(in);
    if (!record) {
      return fail<Result>(error, fmt::format("truncated at entry {}", i));
    }
    std::string text(label.begin(), label.end());
    if (!valid_label(text)) {
      return fail<Result>(error, fmt::format("entry {} has an invalid label", i));
    }
    entries.push_back({*record, std::move(text)});
  }
  if (in.remaining() != 0) {
    return fail<Result>(error, fmt::format("{} trailing bytes", in.remaining()));
  }
  for (size_t i = 1; i < entries.size(); ++i) {
    const uint64_t previous = record_key(entries[i - 1].record);
    const uint64_t current = record_key(entries[i].record);
    if (current == previous) {
      return fail<Result>(error, fmt::format("duplicate key {:016x}", current));
    }
    if (current < previous) {
      return fail<Result>(error, fmt::format("entry {} is not sorted by key", i));
    }
  }
  return entries;
}

std::optional<std::vector<Blob>> read_blobs(const std::filesystem::path& path, std::string* error) {
  using Result = std::vector<Blob>;
  const auto data = read_file(path);
  if (!data) {
    return fail<Result>(error, fmt::format("cannot read {}", path.string()));
  }
  ByteReader in{*data};
  std::span<const uint8_t> magic;
  uint16_t version = 0;
  uint16_t reserved16 = 0;
  uint32_t count = 0;
  uint32_t reserved32 = 0;
  if (!in.take(4, magic) || !in.read(version) || !in.read(reserved16) || !in.read(count) || !in.read(reserved32)) {
    return fail<Result>(error, "truncated header");
  }
  if (!std::equal(magic.begin(), magic.end(), "GXPB")) {
    return fail<Result>(error, "bad magic");
  }
  if (version != BlobFormatVersion) {
    return fail<Result>(error, fmt::format("unsupported blob format {}", version));
  }
  if (reserved16 != 0 || reserved32 != 0) {
    return fail<Result>(error, "reserved header field is not zero");
  }
  Result blobs;
  for (uint32_t i = 0; i < count; ++i) {
    Blob blob;
    uint32_t size = 0;
    std::span<const uint8_t> bytes;
    if (!in.read(blob.configVersion) || !in.read(size) || !in.read(blob.dbHash) || !in.read(blob.firstFrameUsed) ||
        !in.read(blob.sourceIndex) || !in.take(size, bytes)) {
      return fail<Result>(error, fmt::format("truncated at blob {}", i));
    }
    blob.bytes.assign(bytes.begin(), bytes.end());
    blobs.push_back(std::move(blob));
  }
  if (in.remaining() != 0) {
    return fail<Result>(error, fmt::format("{} trailing bytes", in.remaining()));
  }
  return blobs;
}
} // namespace aurora::gx::corpus

#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"

#include "gx/gx.hpp"
#include "gx/shader_info.hpp"
#include "gx/unlit_plan.hpp"

#include <fmt/format.h>

#if AURORA_ENABLE_RENDER
#include "render/json.hpp"
#endif

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace aurora::gx::unlit {
inline void PrintTo(Verdict verdict, std::ostream* os) { *os << verdict_name(verdict); }
} // namespace aurora::gx::unlit

namespace aurora::gx {
namespace {
using unlit::Plan;
using unlit::Verdict;

ColorChannelConfig lit_channel(GXColorSrc matSrc, GXAttnFn attnFn = GX_AF_SPOT) {
  return {.matSrc = matSrc, .ambSrc = GX_SRC_REG, .diffFn = GX_DF_CLAMP, .attnFn = attnFn, .lightingEnabled = true};
}

ShaderConfig lit_config(GXColorSrc matSrc = GX_SRC_VTX) {
  ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0};
  config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1};
  config.colorChannels[GX_COLOR0] = lit_channel(matSrc);
  config.colorChannels[GX_ALPHA0] = {.matSrc = GX_SRC_VTX};
  return config;
}

ShaderConfig corpus_config(GXColorSrc matSrc = GX_SRC_VTX) {
  ShaderConfig config = lit_config(matSrc);
  config.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_REG, GX_AF_SPEC);
  return config;
}

TevStage tev(GXTevColorArg a, GXTevColorArg b, GXTevColorArg c, GXTevColorArg d, u32 map = 0, u32 coord = 0) {
  TevStage stage;
  stage.colorPass = {a, b, c, d};
  stage.texMapId = static_cast<GXTexMapID>(map);
  stage.texCoordId = map == GX_TEXMAP_NULL ? GX_TEXCOORD_NULL : static_cast<GXTexCoordID>(coord);
  stage.channelId = GX_COLOR0A0;
  return stage;
}

TevStage scaled(TevStage stage, GXTevScale scale) {
  stage.colorOp.scale = scale;
  return stage;
}

TevStage on_channel(TevStage stage, GXChannelID channel) {
  stage.channelId = channel;
  return stage;
}

TevStage with_alpha(TevStage stage, GXTevAlphaArg a, GXTevAlphaArg b, GXTevAlphaArg c, GXTevAlphaArg d) {
  stage.alphaPass = {a, b, c, d};
  return stage;
}

ShaderConfig with_stages(ShaderConfig config, std::initializer_list<TevStage> stages) {
  u32 count = 0;
  for (const TevStage& stage : stages) {
    config.tevStages[count++] = stage;
  }
  config.tevStageCount = count;
  return config;
}

ShaderConfig ma00_shape() {
  TevStage cloud = tev(GX_CC_ZERO, GX_CC_ONE, GX_CC_KONST, GX_CC_TEXC, 1, 1);
  cloud.kcSel = GX_TEV_KCSEL_K1_R;
  cloud.colorOp.outReg = GX_TEVREG2;
  cloud.channelId = GX_COLOR_ZERO;
  TevStage light = scaled(tev(GX_CC_KONST, GX_CC_C2, GX_CC_CPREV, GX_CC_C0, GX_TEXMAP_NULL), GX_CS_SCALE_4);
  light.kcSel = GX_TEV_KCSEL_K0;
  light.channelId = GX_COLOR_ZERO;
  ShaderConfig config = with_stages(corpus_config(), {with_alpha(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                                                 GX_CA_ZERO, GX_CA_RASA, GX_CA_TEXA, GX_CA_ZERO),
                                                      cloud, light});
  config.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_TEXMTX1};
  return config;
}

ShaderConfig neutral_map_shape() {
  ShaderConfig config = with_stages(
      corpus_config(), {with_alpha(scaled(tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_TEXC, GX_CC_ZERO), GX_CS_SCALE_2),
                                   GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST)});
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_NRM, .mtx = GX_TEXMTX0};
  return config;
}

ShaderConfig indirect_config(GXTexGenSrc indirectSrc) {
  TevStage stage = tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_TEXC, GX_CC_ZERO);
  stage.indTexStage = GX_INDTEXSTAGE0;
  stage.indTexMtxId = GX_ITM_0;
  ShaderConfig config = with_stages(lit_config(), {stage});
  config.numIndStages = 1;
  config.indStages[0] = {.texCoordId = GX_TEXCOORD1, .texMapId = GX_TEXMAP1, .scaleS = GX_ITS_1, .scaleT = GX_ITS_1};
  config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = indirectSrc};
  return config;
}

TEST(GxUnlitPlan, R1RegisterMaterial) {
  const Plan p =
      unlit::analyse(with_stages(lit_config(GX_SRC_REG), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)}));
  EXPECT_EQ(p.verdict, Verdict::Linear);
  EXPECT_TRUE(p.pass());
  EXPECT_EQ(p.rewrite[0], unlit::RewriteMaterial);
  EXPECT_EQ(p.rewrite[1], unlit::RewriteNone);
  EXPECT_EQ(p.rawVtx, 0u);
  EXPECT_EQ(p.scaleDivide, 0u);
  EXPECT_EQ(p.layerStages, 1u);
  EXPECT_EQ(p.failRule, 0u);
  EXPECT_EQ(p.failStage, unlit::NoStage);
}

TEST(GxUnlitPlan, R2VertexMaterial) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)});
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Linear);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteVertex);
  EXPECT_EQ(p.rawVtx, 1u);
  EXPECT_EQ(unlit::raw_vertex_varyings(p), 1u);
  config.attrs[GX_VA_CLR0] = {};
  const Plan noColour = unlit::analyse(config);
  EXPECT_EQ(noColour.rewrite[0], unlit::RewriteVertex);
  EXPECT_EQ(noColour.rawVtx, 0u);
  EXPECT_EQ(unlit::raw_vertex_varyings(noColour), 0u);
}

TEST(GxUnlitPlan, R3UnlitChannelUnchanged) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)});
  config.colorChannels[GX_COLOR0].lightingEnabled = false;
  config.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_VTX);
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Passthrough);
  EXPECT_TRUE(p.pass());
  EXPECT_EQ(p.rewrite[0], unlit::RewriteNone);
  EXPECT_EQ(p.rewrite[1], unlit::RewriteNone);
  EXPECT_EQ(p.rawVtx, 0u);
  EXPECT_EQ(p.layerStages, 1u);
}

TEST(GxUnlitPlan, R4SpecOnlyFails) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)});
  config.colorChannels[GX_COLOR0].attnFn = GX_AF_SPEC;
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::FailSpecOnly);
  EXPECT_FALSE(p.pass());
  EXPECT_EQ(p.failRule, 4u);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteNone);
}

TEST(GxUnlitPlan, R4MixedZeroesSpecChannel) {
  ShaderConfig config =
      with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                 on_channel(tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC), GX_COLOR1A1)});
  config.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_VTX, GX_AF_SPEC);
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Linear);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteVertex);
  EXPECT_EQ(p.rewrite[1], unlit::RewriteZero);
  EXPECT_EQ(p.rawVtx, 1u);
}

TEST(GxUnlitPlan, R5LitAlphaIntoColourFails) {
  ShaderConfig direct = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO)});
  direct.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  const Plan p = unlit::analyse(direct);
  EXPECT_EQ(p.verdict, Verdict::FailOther);
  EXPECT_EQ(p.failRule, 5u);
  EXPECT_EQ(p.failStage, 0u);

  ShaderConfig throughRegister = with_stages(
      lit_config(),
      {with_alpha(tev(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC), GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO),
       tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_APREV, GX_CC_ZERO, GX_TEXMAP_NULL)});
  throughRegister.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  const Plan viaRegister = unlit::analyse(throughRegister);
  EXPECT_EQ(viaRegister.verdict, Verdict::FailOther);
  EXPECT_EQ(viaRegister.failStage, 1u);

  ShaderConfig alphaOnly = with_stages(lit_config(), {with_alpha(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                                                 GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO)});
  alphaOnly.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  EXPECT_EQ(unlit::analyse(alphaOnly).verdict, Verdict::Linear);
}

TEST(GxUnlitPlan, R6DividesThroughRegisters) {
  ShaderConfig config = ma00_shape();
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Scale);
  EXPECT_EQ(p.scaleDivide, 1u << 2);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteVertex);
  EXPECT_EQ(p.rewrite[1], unlit::RewriteNone);
  EXPECT_EQ(p.rawVtx, 1u);
  EXPECT_EQ(p.layerStages, 1u);
  config.tevStages[2].colorOp.scale = GX_CS_SCALE_1;
  const Plan unscaled = unlit::analyse(config);
  EXPECT_EQ(unscaled.verdict, Verdict::Linear);
  EXPECT_EQ(unscaled.scaleDivide, 0u);
}

TEST(GxUnlitPlan, R7NormalTexgenNeutral) {
  const Plan multiplied = unlit::analyse(neutral_map_shape());
  EXPECT_EQ(multiplied.verdict, Verdict::Scale);
  EXPECT_EQ(multiplied.scaleDivide, 1u);
  EXPECT_EQ(multiplied.neutralOne, 1u);
  EXPECT_EQ(multiplied.neutralZero, 0u);
  EXPECT_EQ(multiplied.layerStages, 0u);

  ShaderConfig added = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                                  tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, 1, 1)});
  added.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_NRM};
  const Plan additive = unlit::analyse(added);
  EXPECT_EQ(additive.verdict, Verdict::Linear);
  EXPECT_EQ(additive.neutralOne, 0u);
  EXPECT_EQ(additive.neutralZero, 1u << 1);
  EXPECT_EQ(additive.layerStages, 1u);

  ShaderConfig other = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                                  tev(GX_CC_TEXC, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV, 1, 1)});
  for (const TcgConfig texgen : {TcgConfig{.type = GX_TG_MTX2x4, .src = GX_TG_BINRM},
                                 TcgConfig{.type = GX_TG_BUMP0, .src = GX_TG_TEX0, .embossSrc = 0}}) {
    other.tcgs[1] = texgen;
    const Plan p = unlit::analyse(other);
    EXPECT_EQ(p.verdict, Verdict::FailLightTexgen);
    EXPECT_EQ(p.failRule, 7u);
    EXPECT_EQ(p.failStage, 1u);
  }
}

TEST(GxUnlitPlan, R8ColourTexgenFollowsV0) {
  for (const GXTexGenType type : {GX_TG_MTX2x4, GX_TG_SRTG}) {
    ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_TEXC, GX_CC_ZERO)});
    config.tcgs[0] = {.type = type, .src = GX_TG_COLOR0};
    const Plan p = unlit::analyse(config);
    EXPECT_EQ(p.verdict, Verdict::Linear);
    EXPECT_EQ(p.neutralOne, 1u);
  }

  ShaderConfig unlitOther = with_stages(lit_config(), {tev(GX_CC_TEXC, GX_CC_ZERO, GX_CC_RASC, GX_CC_ZERO)});
  unlitOther.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_COLOR1};
  const Plan mapFail = unlit::analyse(unlitOther);
  EXPECT_EQ(mapFail.verdict, Verdict::FailLightTexgen);
  EXPECT_EQ(mapFail.failRule, 8u);
  EXPECT_EQ(mapFail.failStage, 0u);

  ShaderConfig litOther = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC),
                                                     tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_TEXC, 1, 1)});
  litOther.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_COLOR0};
  const Plan tainted = unlit::analyse(litOther);
  EXPECT_EQ(tainted.verdict, Verdict::FailLightTexgen);
  EXPECT_EQ(tainted.failRule, 8u);
  EXPECT_EQ(tainted.failStage, 1u);
}

TEST(GxUnlitPlan, R9CompareProductLerpFail) {
  TevStage compare = tev(GX_CC_RASC, GX_CC_TEXC, GX_CC_ONE, GX_CC_ZERO);
  compare.colorOp.op = GX_TEV_COMP_R8_GT;
  const TevStage product = tev(GX_CC_RASC, GX_CC_ZERO, GX_CC_RASC, GX_CC_ZERO);
  const TevStage lerp = tev(GX_CC_TEXC, GX_CC_C0, GX_CC_RASC, GX_CC_ZERO);
  for (const TevStage& stage : {compare, product, lerp}) {
    const Plan p = unlit::analyse(with_stages(lit_config(), {stage}));
    EXPECT_EQ(p.verdict, Verdict::FailNonlinear);
    EXPECT_EQ(p.failRule, 9u);
    EXPECT_EQ(p.failStage, 0u);
  }
  const Plan factor =
      unlit::analyse(with_stages(lit_config(), {product, tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_CPREV, GX_CC_ZERO)}));
  EXPECT_EQ(factor.verdict, Verdict::FailNonlinear);
  EXPECT_EQ(factor.failStage, 0u);
}

TEST(GxUnlitPlan, R9ConstLerpStaysLinear) {
  TevStage compare = tev(GX_CC_TEXC, GX_CC_KONST, GX_CC_RASC, GX_CC_ZERO);
  compare.colorOp.op = GX_TEV_COMP_R8_GT;
  for (const TevStage& stage : {tev(GX_CC_KONST, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                tev(GX_CC_TEXC, GX_CC_ONE, GX_CC_RASC, GX_CC_ZERO), compare}) {
    const Plan p = unlit::analyse(with_stages(lit_config(), {stage}));
    EXPECT_EQ(p.verdict, Verdict::Linear);
    EXPECT_EQ(p.rewrite[0], unlit::RewriteVertex);
  }
}

TEST(GxUnlitPlan, R12PassthroughTakesRasaFromAlphaChannel) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO)});
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Passthrough);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteNone);
  config.colorChannels[GX_COLOR0].lightingEnabled = false;
  config.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  EXPECT_EQ(unlit::analyse(config).verdict, Verdict::FailOther);
}

TEST(GxUnlitPlan, RasSwapAlphaIntoColourFails) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)});
  config.colorChannels[GX_COLOR0].lightingEnabled = false;
  config.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  config.tevSwapTable[1] = {.red = GX_CH_ALPHA};
  EXPECT_EQ(unlit::analyse(config).verdict, Verdict::Passthrough);
  config.tevStages[0].tevSwapRas = GX_TEV_SWAP1;
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::FailOther);
  EXPECT_EQ(p.failRule, 5u);
  EXPECT_EQ(p.failStage, 0u);
}

TEST(GxUnlitPlan, RasaSwapToColourIsRewritten) {
  ShaderConfig config = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO)});
  config.tevSwapTable[1] = {.alpha = GX_CH_RED};
  EXPECT_EQ(unlit::analyse(config).verdict, Verdict::Passthrough);
  config.tevStages[0].tevSwapRas = GX_TEV_SWAP1;
  const Plan p = unlit::analyse(config);
  EXPECT_EQ(p.verdict, Verdict::Linear);
  EXPECT_EQ(p.rewrite[0], unlit::RewriteVertex);
  EXPECT_EQ(p.rawVtx, 1u);

  TevStage swapped =
      with_alpha(tev(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC), GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO);
  swapped.tevSwapRas = GX_TEV_SWAP1;
  ShaderConfig alphaPath =
      with_stages(lit_config(), {swapped, tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_APREV, GX_CC_ZERO, GX_TEXMAP_NULL)});
  alphaPath.tevSwapTable[1] = {.alpha = GX_CH_RED};
  EXPECT_EQ(unlit::analyse(alphaPath).verdict, Verdict::FailOther);
}

TEST(GxUnlitPlan, IndirectLitColourTexgenFails) {
  const Plan p = unlit::analyse(indirect_config(GX_TG_COLOR0));
  EXPECT_EQ(p.verdict, Verdict::FailLightTexgen);
  EXPECT_EQ(p.failRule, 8u);
  EXPECT_EQ(p.failStage, 0u);

  EXPECT_EQ(unlit::analyse(indirect_config(GX_TG_NRM)).verdict, Verdict::Linear);
  EXPECT_EQ(unlit::analyse(indirect_config(GX_TG_COLOR1)).verdict, Verdict::Linear);
  ShaderConfig unused = indirect_config(GX_TG_COLOR0);
  unused.tevStages[0].indTexMtxId = GX_ITM_OFF;
  EXPECT_EQ(unlit::analyse(unused).verdict, Verdict::Linear);

  TevStage bump = on_channel(tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC, GX_TEXMAP_NULL), GX_ALPHA_BUMP);
  bump.indTexStage = GX_INDTEXSTAGE0;
  bump.indTexAlphaSel = GX_ITBA_S;
  ShaderConfig bumped = with_stages(unused, {unused.tevStages[0], bump});
  const Plan alphaBump = unlit::analyse(bumped);
  EXPECT_EQ(alphaBump.verdict, Verdict::FailLightTexgen);
  EXPECT_EQ(alphaBump.failStage, 1u);
  bumped.tcgs[1].src = GX_TG_NRM;
  EXPECT_EQ(unlit::analyse(bumped).verdict, Verdict::Linear);

  TevStage chained = tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, 2, 0);
  chained.indTexAddPrev = true;
  ShaderConfig chain =
      with_stages(indirect_config(GX_TG_COLOR0), {indirect_config(GX_TG_COLOR0).tevStages[0],
                                                  tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO), chained});
  chain.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC};
  const Plan carried = unlit::analyse(chain);
  EXPECT_EQ(carried.verdict, Verdict::FailLightTexgen);
  EXPECT_EQ(carried.failStage, 2u);
  chain.tevStages[2].indTexAddPrev = false;
  EXPECT_EQ(unlit::analyse(chain).verdict, Verdict::Linear);
}

struct Shape {
  std::string_view name;
  ShaderConfig config;
  Verdict verdict;
  uint16_t scaleDivide = 0;
  uint8_t neutralOne = 0;
  uint8_t neutralZero = 0;
};

std::vector<Shape> corpus_shapes() {
  std::vector<Shape> shapes;
  {
    TevStage base = scaled(tev(GX_CC_C1, GX_CC_KONST, GX_CC_TEXC, GX_CC_ZERO), GX_CS_SCALE_4);
    base.kcSel = GX_TEV_KCSEL_K2;
    base.channelId = GX_COLOR_ZERO;
    TevStage env = with_alpha(tev(GX_CC_CPREV, GX_CC_KONST, GX_CC_TEXC, GX_CC_ZERO, 1, 1), GX_CA_RASA, GX_CA_ZERO,
                              GX_CA_ZERO, GX_CA_ZERO);
    env.kcSel = GX_TEV_KCSEL_K1;
    ShaderConfig config = with_stages(corpus_config(), {base, env});
    config.colorChannels[GX_COLOR0].lightingEnabled = false;
    config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_NRM, .mtx = GX_TEXMTX0};
    shapes.push_back({"R12", config, Verdict::Passthrough});
  }
  {
    TevStage stage = with_alpha(scaled(tev(GX_CC_KONST, GX_CC_TEXC, GX_CC_RASC, GX_CC_C0), GX_CS_SCALE_4), GX_CA_ZERO,
                                GX_CA_RASA, GX_CA_TEXA, GX_CA_ZERO);
    stage.kcSel = GX_TEV_KCSEL_K0;
    shapes.push_back({"R2+R6", with_stages(corpus_config(), {stage}), Verdict::Scale, 1u});
  }
  {
    TevStage base = tev(GX_CC_C1, GX_CC_KONST, GX_CC_TEXC, GX_CC_ZERO);
    base.kcSel = GX_TEV_KCSEL_K1;
    const TevStage light = scaled(tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_CPREV, GX_CC_ZERO, 1, 1), GX_CS_SCALE_2);
    shapes.push_back({"R1+R6", with_stages(corpus_config(GX_SRC_REG), {base, light}), Verdict::Scale, 1u << 1});
    shapes.push_back(
        {"R1", with_stages(corpus_config(GX_SRC_REG), {base, scaled(light, GX_CS_DIVIDE_2)}), Verdict::Linear});
  }
  {
    TevStage base = with_alpha(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO), GX_CA_TEXA, GX_CA_ZERO, GX_CA_KONST,
                               GX_CA_RASA);
    base.kaSel = GX_TEV_KASEL_K3_A;
    TevStage cloud = with_alpha(tev(GX_CC_ZERO, GX_CC_ONE, GX_CC_KONST, GX_CC_TEXC, 1, 1), GX_CA_ZERO, GX_CA_ZERO,
                                GX_CA_ZERO, GX_CA_APREV);
    cloud.kcSel = GX_TEV_KCSEL_K1_R;
    cloud.colorOp.outReg = GX_TEVREG2;
    cloud.channelId = GX_COLOR_ZERO;
    TevStage light = scaled(tev(GX_CC_KONST, GX_CC_C2, GX_CC_CPREV, GX_CC_C0, GX_TEXMAP_NULL), GX_CS_SCALE_4);
    light.kcSel = GX_TEV_KCSEL_K0;
    light.channelId = GX_COLOR_ZERO;
    const TevStage mask = tev(GX_CC_C1, GX_CC_CPREV, GX_CC_APREV, GX_CC_ZERO, GX_TEXMAP_NULL);
    ShaderConfig config = with_stages(corpus_config(), {base, cloud, light, mask});
    config.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_TEXMTX1};
    shapes.push_back({"R2+R6 policy reads", config, Verdict::Scale, 1u << 2});
  }
  shapes.push_back({"R2",
                    with_stages(corpus_config(), {with_alpha(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                                             GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO)}),
                    Verdict::Linear});
  {
    ShaderConfig config =
        with_stages(corpus_config(), {scaled(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO), GX_CS_SCALE_2),
                                      tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC, 1, 1)});
    config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_COLOR0};
    shapes.push_back({"R2+R6+R7/R8", config, Verdict::Scale, 1u, 0, 1u << 1});
  }
  {
    const TevStage base = with_alpha(scaled(tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO), GX_CS_SCALE_4),
                                     GX_CA_ZERO, GX_CA_RASA, GX_CA_TEXA, GX_CA_ZERO);
    const TevStage detail = with_alpha(scaled(tev(GX_CC_ZERO, GX_CC_C0, GX_CC_TEXA, GX_CC_CPREV, 1, 0), GX_CS_SCALE_2),
                                       GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV);
    shapes.push_back({"b86d3a3eccd3501e", with_stages(corpus_config(), {base, detail}), Verdict::Scale, 0b11u});
  }
  shapes.push_back({"090ec64471285f52", neutral_map_shape(), Verdict::Scale, 1u, 1u});
  return shapes;
}

TEST(GxUnlitPlan, CorpusShapes) {
  for (const Shape& shape : corpus_shapes()) {
    const Plan p = unlit::analyse(shape.config);
    EXPECT_EQ(p.verdict, shape.verdict) << shape.name;
    EXPECT_EQ(p.scaleDivide, shape.scaleDivide) << shape.name;
    EXPECT_EQ(p.neutralOne, shape.neutralOne) << shape.name;
    EXPECT_EQ(p.neutralZero, shape.neutralZero) << shape.name;
    EXPECT_EQ(p.failRule, 0u) << shape.name;
  }
}

TEST(GxUnlitPlan, PlanIsCopiedOut) {
  static_assert(std::is_same_v<decltype(unlit::plan(std::declval<const ShaderConfig&>())), Plan>);
  static_assert(std::is_same_v<decltype(unlit::analyse(std::declval<const ShaderConfig&>())), Plan>);
  static_assert(std::is_trivially_copyable_v<Plan> && sizeof(Plan) <= 16);
  const ShaderConfig config = ma00_shape();
  Plan first = unlit::plan(config);
  const Plan expected = first;
  first.verdict = Verdict::FailOther;
  first.scaleDivide = 0xFFFF;
  first.rewrite[0] = unlit::RewriteZero;
  const Plan second = unlit::plan(config);
  EXPECT_EQ(second, expected);
  EXPECT_EQ(second, unlit::analyse(config));
  EXPECT_NE(first, second);
}

TEST(GxUnlitPlan, AdmittedNeedsNrmAndVaryingRoom) {
  ShaderConfig config =
      with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                 on_channel(tev(GX_CC_CPREV, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC), GX_COLOR1A1)});
  config.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_VTX);
  config.attrs[GX_VA_CLR1] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  const Plan p = unlit::analyse(config);
  ASSERT_EQ(p.verdict, Verdict::Linear);
  ASSERT_EQ(p.rawVtx, 0b11u);
  ASSERT_EQ(unlit::raw_vertex_varyings(p), 2u);
  EXPECT_FALSE(unlit::admitted(p, config, 13));
  EXPECT_TRUE(unlit::admitted(p, config, 12));

  ShaderConfig noNormals = config;
  noNormals.attrs[GX_VA_NRM] = {};
  EXPECT_FALSE(unlit::admitted(unlit::analyse(noNormals), noNormals, 4));
  ShaderConfig lines = config;
  lines.lineMode = 1;
  EXPECT_FALSE(unlit::admitted(unlit::analyse(lines), lines, 4));
  ShaderConfig failing = config;
  failing.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_VTX);
  failing.tevStages[0].colorPass.c = GX_CC_RASA;
  const Plan failed = unlit::analyse(failing);
  ASSERT_FALSE(failed.pass());
  EXPECT_FALSE(unlit::admitted(failed, failing, 4));
  const ShaderConfig textured = with_stages(lit_config(), {tev(GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC)});
  const Plan passthrough = unlit::analyse(textured);
  ASSERT_EQ(passthrough.verdict, Verdict::Passthrough);
  EXPECT_TRUE(unlit::admitted(passthrough, textured, MaxGxVaryings));
  EXPECT_FALSE(unlit::admitted(passthrough, textured, MaxGxVaryings + 1));
}

TEST(GxUnlitPlan, IgnoresExtBits) {
  ShaderConfig config = ma00_shape();
  config.alphaCompare.ref0 = 0x5A5A;
  const Plan base = unlit::analyse(config);
  const Plan cached = unlit::plan(config);
  config.ext.output = 2;
  config.ext.velocity = 1;
  config.ext.pom = 1;
  config.ext.fwdEnv = 1;
  config.ext.stochasticMask = 0xFF;
  config.ext.layer0 = 3;
  config.ext.layer1 = 5;
  EXPECT_EQ(unlit::analyse(config), base);
  EXPECT_EQ(unlit::plan(config), cached);
  EXPECT_EQ(cached, base);
}

TEST(GxUnlitPlan, CachedEqualsUncached) {
  std::vector<ShaderConfig> configs{ma00_shape(), neutral_map_shape(), indirect_config(GX_TG_COLOR0),
                                    indirect_config(GX_TG_NRM)};
  for (const Shape& shape : corpus_shapes()) {
    configs.push_back(shape.config);
  }
  for (const ShaderConfig& config : configs) {
    const Plan uncached = unlit::analyse(config);
    EXPECT_EQ(unlit::plan(config), uncached);
    EXPECT_EQ(unlit::plan(config), uncached);
  }
  ShaderConfig churn = ma00_shape();
  const Plan expected = unlit::analyse(churn);
  for (u32 i = 0; i < 4096 + 512; ++i) {
    churn.alphaCompare.ref1 = i;
    ASSERT_EQ(unlit::plan(churn), expected) << i;
  }
  for (const ShaderConfig& config : configs) {
    EXPECT_EQ(unlit::plan(config), unlit::analyse(config));
  }
}

TEST(GxUnlitPlan, RuleTableVersionPinned) {
  static_assert(unlit::RuleTableVersion == 1);
  EXPECT_EQ(unlit::RuleTableVersion, 1u);
  EXPECT_EQ(unlit::verdict_name(Verdict::Passthrough), "PASS_PASSTHROUGH");
  EXPECT_EQ(unlit::verdict_name(Verdict::Linear), "PASS_LINEAR");
  EXPECT_EQ(unlit::verdict_name(Verdict::Scale), "PASS_SCALE");
  EXPECT_EQ(unlit::verdict_name(Verdict::FailSpecOnly), "FAIL_SPEC_ONLY");
  EXPECT_EQ(unlit::verdict_name(Verdict::FailOther), "FAIL_OTHER");
  EXPECT_EQ(unlit::verdict_name(Verdict::FailNonlinear), "FAIL_NONLINEAR");
  EXPECT_EQ(unlit::verdict_name(Verdict::FailLightTexgen), "FAIL_LIGHT_TEXGEN");
}

#if AURORA_ENABLE_RENDER
std::string read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

bool parse_record(std::string_view hex, corpus::Record& record) {
  if (hex.size() != record.bytes.size() * 2) {
    return false;
  }
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  };
  for (size_t i = 0; i < record.bytes.size(); ++i) {
    const int hi = nibble(hex[2 * i]);
    const int lo = nibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      return false;
    }
    record.bytes[i] = static_cast<uint8_t>(hi << 4 | lo);
  }
  return true;
}

std::string_view str(yyjson_val* value) {
  const char* text = yyjson_get_str(value);
  return text != nullptr ? std::string_view{text} : std::string_view{};
}

struct Sp1Row {
  std::string verdict;
  uint16_t scaleDivide = 0;
  uint8_t neutralOne = 0;
  uint8_t neutralZero = 0;
};
#endif

TEST(GxUnlitPlan, Sp1CorpusMatchesRules) {
  const char* programsPath = std::getenv("AURORA_SP1_PROGRAMS");
  const char* rulesPath = std::getenv("AURORA_SP1_RULES");
  if (programsPath == nullptr || *programsPath == '\0' || rulesPath == nullptr || *rulesPath == '\0') {
    GTEST_SKIP() << "AURORA_SP1_PROGRAMS and AURORA_SP1_RULES are not set";
  }
#if !AURORA_ENABLE_RENDER
  GTEST_SKIP() << "the JSON reader is part of the render module";
#else
  std::string error;
  const std::string rulesText = read_file(rulesPath);
  const render::json::Doc rulesDoc = render::json::read(rulesText, &error);
  ASSERT_TRUE(rulesDoc) << rulesPath << ": " << error;
  yyjson_val* rulePrograms = yyjson_obj_get(yyjson_doc_get_root(rulesDoc.get()), "programs");
  ASSERT_TRUE(yyjson_is_arr(rulePrograms)) << rulesPath;
  std::map<std::string, Sp1Row, std::less<>> rows;
  size_t index = 0;
  size_t max = 0;
  yyjson_val* entry = nullptr;
  yyjson_arr_foreach(rulePrograms, index, max, entry) {
    Sp1Row row{.verdict = std::string{str(yyjson_obj_get(entry, "verdictNoPolicy"))}};
    size_t i = 0;
    size_t n = 0;
    yyjson_val* stage = nullptr;
    yyjson_arr_foreach(yyjson_obj_get(entry, "scaleDivide"), i, n, stage) {
      row.scaleDivide |= static_cast<uint16_t>(1u << yyjson_get_uint(stage));
    }
    yyjson_val* map = nullptr;
    yyjson_val* neutral = nullptr;
    yyjson_obj_foreach(yyjson_obj_get(entry, "neutralTexmaps"), i, n, map, neutral) {
      const auto bit = static_cast<uint8_t>(1u << std::atoi(yyjson_get_str(map)));
      (yyjson_get_uint(neutral) != 0 ? row.neutralOne : row.neutralZero) |= bit;
    }
    rows.emplace(std::string{str(yyjson_obj_get(entry, "key"))}, std::move(row));
  }

  const std::string programsText = read_file(programsPath);
  const render::json::Doc programsDoc = render::json::read(programsText, &error);
  ASSERT_TRUE(programsDoc) << programsPath << ": " << error;
  yyjson_val* programs = yyjson_obj_get(yyjson_doc_get_root(programsDoc.get()), "programs");
  ASSERT_TRUE(yyjson_is_arr(programs)) << programsPath;
  std::map<std::string, u32, std::less<>> histogram;
  u32 total = 0;
  u32 matched = 0;
  u32 admittedCount = 0;
  u32 noNormals = 0;
  u32 noRoom = 0;
  yyjson_arr_foreach(programs, index, max, entry) {
    ++total;
    const std::string_view key = str(yyjson_obj_get(entry, "key"));
    corpus::Record record;
    ASSERT_TRUE(parse_record(str(yyjson_obj_get(entry, "record")), record)) << key;
    EXPECT_EQ(fmt::format("{:016x}", corpus::record_key(record)), key);
    const auto row = rows.find(key);
    ASSERT_NE(row, rows.end()) << key;
    const ShaderConfig config = corpus::decode(record);
    const Plan p = unlit::analyse(config);
    EXPECT_EQ(unlit::plan(config), p) << key;
    const bool same = unlit::verdict_name(p.verdict) == row->second.verdict &&
                      p.scaleDivide == row->second.scaleDivide && p.neutralOne == row->second.neutralOne &&
                      p.neutralZero == row->second.neutralZero;
    EXPECT_TRUE(same) << key << ": " << unlit::verdict_name(p.verdict) << " divide " << p.scaleDivide << " one "
                      << +p.neutralOne << " zero " << +p.neutralZero << ", SP1 " << row->second.verdict << " divide "
                      << row->second.scaleDivide << " one " << +row->second.neutralOne << " zero "
                      << +row->second.neutralZero;
    matched += same ? 1u : 0u;
    ++histogram[std::string{unlit::verdict_name(p.verdict)}];
    const u32 base = build_shader_info(config).varyings;
    const bool normals = config.attrs[GX_VA_NRM].attrType != GX_NONE;
    const bool room = base + static_cast<u32>(std::popcount(p.rawVtx)) <= MaxGxVaryings;
    const bool admitted = unlit::admitted(p, config, base);
    EXPECT_EQ(admitted, p.pass() && normals && config.lineMode == 0 && room) << key;
    admittedCount += admitted ? 1u : 0u;
    noNormals += p.pass() && !normals ? 1u : 0u;
    noRoom += p.pass() && normals && !room ? 1u : 0u;
  }
  EXPECT_EQ(total, rows.size());
  EXPECT_EQ(matched, total);
  std::string line = fmt::format("SP1 corpus: {}/{} programs match rules v1 at policy mask 0;", matched, total);
  for (const auto& [verdict, count] : histogram) {
    line += fmt::format(" {} {}", verdict, count);
  }
  line +=
      fmt::format("; admitted {}, pass without NRM {}, pass without varying room {}", admittedCount, noNormals, noRoom);
  std::printf("%s\n", line.c_str());
  RecordProperty("matched", static_cast<int>(matched));
  RecordProperty("admitted", static_cast<int>(admittedCount));
#endif
}
} // namespace
} // namespace aurora::gx

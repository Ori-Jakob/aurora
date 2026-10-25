#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"

#include "gfx/resources.hpp"
#include "gfx/scene_targets.hpp"
#include "gx/shader_info.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace aurora::gx::corpus {
namespace {

constexpr size_t StageColorBias = 9;
constexpr size_t StageColorOutReg = 11;
constexpr size_t StageColorClamp = 12;
constexpr size_t StageKcSel = 18;
constexpr size_t StageKaSel = 19;
constexpr size_t StageTexCoordId = 20;
constexpr size_t StageChannelId = 22;
constexpr size_t StageWrapS = 30;
constexpr size_t TcgSrc = 1;
constexpr size_t TcgPostMtx = 3;
constexpr size_t IndStageScaleS = 2;

class CorpusEnvironment : public ::testing::Environment {
public:
  // build_shader_info aligns uniformSize to this device limit; without a device it is undefined.
  void SetUp() override { gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256; }
};
[[maybe_unused]] const auto* const g_environment = ::testing::AddGlobalTestEnvironment(new CorpusEnvironment);

const std::vector<Entry>& synthetic() {
  static const auto entries = generate_synthetic();
  return entries;
}

Record base_record() {
  const auto& entries = synthetic();
  const auto it =
      std::find_if(entries.begin(), entries.end(), [](const Entry& e) { return e.label == "variant/base=b0@none"; });
  EXPECT_NE(it, entries.end());
  return it == entries.end() ? Record{} : it->record;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
uint32_t num(T value) {
  if constexpr (std::is_enum_v<T>) {
    return static_cast<uint32_t>(static_cast<std::make_unsigned_t<std::underlying_type_t<T>>>(value));
  } else {
    return static_cast<uint32_t>(value);
  }
}

std::set<std::string> collect_coverage(const std::vector<Entry>& entries) {
  std::set<std::string> seen;
  const auto add = [&seen](std::string key) { seen.insert(std::move(key)); };
  for (const auto& entry : entries) {
    const auto c = decode(entry.record);
    const auto v = decode_variant(entry.record);
    const auto info = build_shader_info(c);
    const auto& nrm = c.attrs[GX_VA_NRM];
    add(fmt::format("count/{}", c.tevStageCount));
    for (uint32_t s = 0; s < c.tevStageCount; ++s) {
      const auto& st = c.tevStages[s];
      const std::array colorArgs{st.colorPass.a, st.colorPass.b, st.colorPass.c, st.colorPass.d};
      const std::array alphaArgs{st.alphaPass.a, st.alphaPass.b, st.alphaPass.c, st.alphaPass.d};
      for (uint32_t slot = 0; slot < 4; ++slot) {
        add(fmt::format("carg/{}/{}", slot, num(colorArgs[slot])));
        add(fmt::format("aarg/{}/{}", slot, num(alphaArgs[slot])));
      }
      add(fmt::format("cop/{}/{}", num(st.colorOp.op), num(st.colorOp.clamp)));
      add(fmt::format("aop/{}/{}", num(st.alphaOp.op), num(st.alphaOp.clamp)));
      add(fmt::format("cbias/{}/{}", num(st.colorOp.bias), num(st.colorOp.scale)));
      add(fmt::format("abias/{}/{}", num(st.alphaOp.bias), num(st.alphaOp.scale)));
      add(fmt::format("outreg/{}", num(st.colorOp.outReg)));
      add(fmt::format("outreg/{}", num(st.alphaOp.outReg)));
      if (std::find(colorArgs.begin(), colorArgs.end(), GX_CC_KONST) != colorArgs.end()) {
        add(fmt::format("kc/{}", num(st.kcSel)));
      }
      if (std::find(alphaArgs.begin(), alphaArgs.end(), GX_CA_KONST) != alphaArgs.end()) {
        add(fmt::format("ka/{}", num(st.kaSel)));
      }
      const bool ras = std::find(colorArgs.begin(), colorArgs.end(), GX_CC_RASC) != colorArgs.end() &&
                       std::find(alphaArgs.begin(), alphaArgs.end(), GX_CA_RASA) != alphaArgs.end();
      if (ras) {
        add(fmt::format("channel/{}", num(st.channelId)));
      }
      add(fmt::format("swap/{}/{}", num(st.tevSwapRas), num(st.tevSwapTex)));
      if (num(st.indTexStage) < c.numIndStages) {
        add(fmt::format("indfmt/{}", num(st.indTexFormat)));
        add(fmt::format("indbias/{}", num(st.indTexBiasSel)));
        add(fmt::format("indmtx/{}", num(st.indTexMtxId)));
        add(fmt::format("wraps/{}", num(st.indTexWrapS)));
        add(fmt::format("wrapt/{}", num(st.indTexWrapT)));
        add(fmt::format("addprev/{}", num(st.indTexAddPrev)));
        add(fmt::format("origlod/{}", num(st.indTexUseOrigLOD)));
        if (st.channelId == GX_ALPHA_BUMP || st.channelId == GX_ALPHA_BUMPN) {
          add(fmt::format("indalpha/{}/{}", num(st.indTexAlphaSel), num(st.channelId)));
        }
        if (st.indTexMtxId != GX_ITM_OFF && (st.texMapId == GX_TEXMAP_NULL || st.texCoordId == GX_TEXCOORD_NULL)) {
          add("ind_nobase");
        }
      } else if (st.indTexMtxId != GX_ITM_OFF) {
        add("ind_disabled");
      }
    }
    const auto& last = c.tevStages[c.tevStageCount - 1];
    if (last.colorOp.outReg != GX_TEVPREV) {
      add("last_color_nonprev");
    }
    if (last.alphaOp.outReg != GX_TEVPREV) {
      add("last_alpha_nonprev");
    }
    add(fmt::format("numind/{}", c.numIndStages));
    for (uint32_t k = 0; k < c.numIndStages; ++k) {
      add(fmt::format("indscale/{}/{}", num(c.indStages[k].scaleS), num(c.indStages[k].scaleT)));
    }
    for (uint32_t ch = 0; ch < 2; ++ch) {
      if (!info.sampledColorChannels.test(ch)) {
        continue;
      }
      for (const auto& cc : {c.colorChannels[ch], c.colorChannels[ch + GX_ALPHA0]}) {
        add(fmt::format("matsrc/{}", num(cc.matSrc)));
        if (!cc.lightingEnabled) {
          continue;
        }
        add(fmt::format("lit/{}/{}/{}", ch, num(cc.diffFn), num(cc.attnFn)));
        add(fmt::format("ambsrc/{}", num(cc.ambSrc)));
        if (nrm.attrType == GX_NONE) {
          add("lit_no_nrm");
        }
      }
    }
    for (uint32_t i = 0; i < MaxTexCoord; ++i) {
      if (!info.sampledTexCoords.test(i)) {
        continue;
      }
      const auto& tcg = c.tcgs[i];
      const bool emboss = tcg.type >= GX_TG_BUMP0 && tcg.type <= GX_TG_BUMP7;
      add(fmt::format("tgtype/{}", num(tcg.type)));
      if (emboss) {
        add(fmt::format("emboss_nrm/{}/{}/{}", nrm.attrType, nrm.compType, num(nrm.nbt3)));
        continue;
      }
      add(fmt::format("tgsrc/{}", num(tcg.src)));
      add(fmt::format("tgmtx/{}", num(tcg.mtx)));
      add(fmt::format("tgpost/{}", num(tcg.postMtx)));
      add(fmt::format("tgnorm/{}", num(tcg.normalize)));
      if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) {
        add("texmtxidx");
      }
    }
    add(fmt::format("fog/{}/{}", c.fogType, num(c.fogRangeEnabled)));
    const auto& ac = c.alphaCompare;
    add(fmt::format("ac0/{}", num(ac.comp0)));
    add(fmt::format("ac1/{}", num(ac.comp1)));
    add(fmt::format("acop/{}", num(ac.op)));
    if (ac.ref0 == 0 || ac.ref0 == 255) {
      add(fmt::format("acedge/{}/{}", ac.ref0, num(ac.comp0)));
    }
    add(fmt::format("variant/{}/{}", num(v.dstAlphaMode), v.normalAttachment));
    if (v.normalAttachment != UINT32_MAX) {
      add(fmt::format("normal/{}", nrm.attrType == GX_NONE ? "absent" : (info.lightingEnabled ? "lit" : "unlit")));
    }
    add(fmt::format("line/{}/{}/{}", num(c.lineMode), num(c.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE),
                    num(info.sampledTexCoords.any())));
    for (uint32_t attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
      const auto& a = c.attrs[attr];
      if (a.attrType != GX_NONE) {
        add(fmt::format("attr/{}/{}/{}/{}/{}", attr, a.attrType, a.cnt, a.compType, num(a.nbt3)));
      }
    }
  }
  return seen;
}

std::vector<std::string> expected_coverage() {
  std::vector<std::string> keys;
  const auto add = [&keys](std::string key) { keys.push_back(std::move(key)); };
  for (uint32_t n = 1; n <= MaxTevStages; ++n) {
    add(fmt::format("count/{}", n));
  }
  for (uint32_t slot = 0; slot < 4; ++slot) {
    for (uint32_t arg = 0; arg <= GX_CC_ZERO; ++arg) {
      add(fmt::format("carg/{}/{}", slot, arg));
    }
    for (uint32_t arg = 0; arg <= GX_CA_ZERO; ++arg) {
      add(fmt::format("aarg/{}/{}", slot, arg));
    }
  }
  for (const uint32_t op : {0u, 1u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}) {
    for (uint32_t clamp = 0; clamp < 2; ++clamp) {
      add(fmt::format("cop/{}/{}", op, clamp));
      add(fmt::format("aop/{}/{}", op, clamp));
    }
  }
  for (uint32_t bias = 0; bias <= GX_TB_SUBHALF; ++bias) {
    for (uint32_t scale = 0; scale <= GX_CS_DIVIDE_2; ++scale) {
      add(fmt::format("cbias/{}/{}", bias, scale));
      add(fmt::format("abias/{}/{}", bias, scale));
    }
  }
  for (uint32_t reg = 0; reg < MaxTevRegs; ++reg) {
    add(fmt::format("outreg/{}", reg));
  }
  add("last_color_nonprev");
  add("last_alpha_nonprev");
  for (uint32_t sel = 0; sel <= GX_TEV_KCSEL_K3_A; ++sel) {
    if (sel <= GX_TEV_KCSEL_1_8 || sel >= GX_TEV_KCSEL_K0) {
      add(fmt::format("kc/{}", sel));
    }
    if (sel <= GX_TEV_KASEL_1_8 || sel >= GX_TEV_KASEL_K0_R) {
      add(fmt::format("ka/{}", sel));
    }
  }
  for (const uint32_t id : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 0xFFu}) {
    add(fmt::format("channel/{}", id));
  }
  for (uint32_t ras = 0; ras < MaxTevSwap; ++ras) {
    for (uint32_t tex = 0; tex < MaxTevSwap; ++tex) {
      add(fmt::format("swap/{}/{}", ras, tex));
    }
  }
  for (uint32_t ch = 0; ch < 2; ++ch) {
    for (uint32_t diff = 0; diff <= GX_DF_CLAMP; ++diff) {
      for (uint32_t attn = 0; attn <= GX_AF_NONE; ++attn) {
        add(fmt::format("lit/{}/{}/{}", ch, diff, attn));
      }
    }
  }
  for (uint32_t src = 0; src < 2; ++src) {
    add(fmt::format("matsrc/{}", src));
    add(fmt::format("ambsrc/{}", src));
  }
  add("lit_no_nrm");
  for (uint32_t type = 0; type <= GX_TG_SRTG; ++type) {
    add(fmt::format("tgtype/{}", type));
  }
  for (uint32_t src = 0; src <= GX_TG_COLOR1; ++src) {
    add(fmt::format("tgsrc/{}", src));
  }
  for (uint32_t mtx = GX_TEXMTX0; mtx <= GX_IDENTITY; mtx += 3) {
    add(fmt::format("tgmtx/{}", mtx));
  }
  add(fmt::format("tgmtx/{}", num(GX_PNMTX0)));
  add(fmt::format("tgmtx/{}", num(GX_PNMTX9)));
  for (const uint32_t post : {125u, 64u, 67u, 121u}) {
    add(fmt::format("tgpost/{}", post));
  }
  add("tgnorm/0");
  add("tgnorm/1");
  add("texmtxidx");
  add(fmt::format("emboss_nrm/{}/{}/0", num(GX_DIRECT), num(GX_S16)));
  add(fmt::format("emboss_nrm/{}/{}/1", num(GX_INDEX8), num(GX_S16)));
  add(fmt::format("emboss_nrm/{}/{}/1", num(GX_INDEX16), num(GX_S16)));
  for (uint32_t n = 1; n <= MaxIndStages; ++n) {
    add(fmt::format("numind/{}", n));
  }
  for (uint32_t format = 0; format < GX_MAX_ITFORMAT; ++format) {
    add(fmt::format("indfmt/{}", format));
  }
  for (uint32_t bias = 0; bias < GX_MAX_ITBIAS; ++bias) {
    add(fmt::format("indbias/{}", bias));
  }
  for (uint32_t sel = GX_ITBA_S; sel <= GX_ITBA_U; ++sel) {
    add(fmt::format("indalpha/{}/{}", sel, num(GX_ALPHA_BUMP)));
    add(fmt::format("indalpha/{}/{}", sel, num(GX_ALPHA_BUMPN)));
  }
  for (const uint32_t mtx : {0u, 1u, 2u, 3u, 5u, 6u, 7u, 9u, 10u, 11u}) {
    add(fmt::format("indmtx/{}", mtx));
  }
  for (uint32_t wrap = 0; wrap < GX_MAX_ITWRAP; ++wrap) {
    add(fmt::format("wraps/{}", wrap));
    add(fmt::format("wrapt/{}", wrap));
  }
  add("addprev/1");
  add("origlod/1");
  for (uint32_t scale = 0; scale < GX_MAX_ITSCALE; ++scale) {
    add(fmt::format("indscale/{}/0", scale));
    add(fmt::format("indscale/0/{}", scale));
  }
  add("ind_disabled");
  add("ind_nobase");
  for (const uint32_t fog : {0u, 2u, 4u, 5u, 6u, 7u, 10u, 12u, 13u, 14u, 15u}) {
    add(fmt::format("fog/{}/0", fog));
    add(fmt::format("fog/{}/1", fog));
  }
  for (uint32_t comp = 0; comp <= GX_ALWAYS; ++comp) {
    add(fmt::format("ac0/{}", comp));
    add(fmt::format("ac1/{}", comp));
  }
  for (uint32_t op = 0; op <= GX_AOP_XNOR; ++op) {
    add(fmt::format("acop/{}", op));
  }
  for (const uint32_t ref : {0u, 255u}) {
    for (const auto comp : {GX_LESS, GX_LEQUAL, GX_GEQUAL, GX_GREATER}) {
      add(fmt::format("acedge/{}/{}", ref, num(comp)));
    }
  }
  add("variant/0/4294967295");
  add("variant/0/1");
  add("variant/1/4294967295");
  add("variant/1/1");
  add("variant/2/4294967295");
  add("normal/absent");
  add("normal/unlit");
  add("normal/lit");
  for (uint32_t mode = 0; mode <= 3; ++mode) {
    for (uint32_t pnmtx = 0; pnmtx < 2; ++pnmtx) {
      for (uint32_t tex = 0; tex < 2; ++tex) {
        add(fmt::format("line/{}/{}/{}", mode, pnmtx, tex));
      }
    }
  }
  for (const uint32_t type : {GX_DIRECT, GX_INDEX8, GX_INDEX16}) {
    for (const uint32_t cnt : {2u, 3u}) {
      for (uint32_t comp = 0; comp <= GX_F32; ++comp) {
        add(fmt::format("attr/{}/{}/{}/{}/0", num(GX_VA_POS), type, cnt, comp));
      }
    }
    for (const uint32_t comp : {GX_S8, GX_S16, GX_F32}) {
      add(fmt::format("attr/{}/{}/3/{}/0", num(GX_VA_NRM), type, comp));
      add(fmt::format("attr/{}/{}/9/{}/0", num(GX_VA_NRM), type, comp));
      add(fmt::format("attr/{}/{}/9/{}/1", num(GX_VA_NRM), type, comp));
    }
    for (const uint32_t attr : {GX_VA_CLR0, GX_VA_CLR1}) {
      for (uint32_t comp = 0; comp <= GX_RGBA8; ++comp) {
        add(fmt::format("attr/{}/{}/1/{}/0", attr, type, comp));
      }
    }
  }
  for (const uint32_t cnt : {1u, 2u}) {
    for (uint32_t comp = 0; comp <= GX_F32; ++comp) {
      add(fmt::format("attr/{}/{}/{}/{}/0", num(GX_VA_TEX0), num(GX_DIRECT), cnt, comp));
    }
  }
  add(fmt::format("attr/{}/{}/2/{}/0", num(GX_VA_TEX0), num(GX_INDEX8), num(GX_F32)));
  add(fmt::format("attr/{}/{}/2/{}/0", num(GX_VA_TEX0), num(GX_INDEX16), num(GX_F32)));
  for (uint32_t unit = 1; unit < MaxTexCoord; ++unit) {
    add(fmt::format("attr/{}/{}/2/{}/0", GX_VA_TEX0 + unit, num(GX_DIRECT), num(GX_F32)));
  }
  return keys;
}

ShaderConfig hand_built_config() {
  constexpr std::array OtherOps{GX_TEV_SUB,           GX_TEV_COMP_R8_GT,   GX_TEV_COMP_R8_EQ,
                                GX_TEV_COMP_GR16_GT,  GX_TEV_COMP_GR16_EQ, GX_TEV_COMP_BGR24_GT,
                                GX_TEV_COMP_BGR24_EQ, GX_TEV_COMP_RGB8_GT, GX_TEV_COMP_RGB8_EQ};
  constexpr std::array OtherMtx{GX_ITM_0,  GX_ITM_1,  GX_ITM_2,  GX_ITM_S0, GX_ITM_S1,
                                GX_ITM_S2, GX_ITM_T0, GX_ITM_T1, GX_ITM_T2};
  ShaderConfig c{};
  c.fogType = GX_FOG_ORTHO_EXP2;
  c.vtxStride = 77;
  c.lineMode = 2;
  c.fogRangeEnabled = true;
  for (uint32_t i = 0; i < MaxVtxAttr; ++i) {
    c.attrs[i] = {
        .attrType = static_cast<u8>(1 + i % 3),
        .cnt = static_cast<u8>(i + 1),
        .compType = static_cast<u8>(i % 5),
        .offset = static_cast<u8>(i * 3 + 1),
        .stride = static_cast<u8>(i + 2),
        .frac = static_cast<u8>(i % 15 + 1),
        .le = false,
        .nbt3 = true,
    };
  }
  for (uint32_t i = 0; i < MaxTevSwap; ++i) {
    c.tevSwapTable[i] = {
        static_cast<GXTevColorChan>(1 + i % 3),
        static_cast<GXTevColorChan>((2 + i % 3) % 4),
        static_cast<GXTevColorChan>((3 + i % 3) % 4),
        static_cast<GXTevColorChan>(i % 3),
    };
  }
  for (uint32_t s = 0; s < MaxTevStages; ++s) {
    auto& st = c.tevStages[s];
    st.colorPass = {static_cast<GXTevColorArg>(s % 15), static_cast<GXTevColorArg>((s + 1) % 15),
                    static_cast<GXTevColorArg>((s + 2) % 15), static_cast<GXTevColorArg>((s + 3) % 15)};
    st.alphaPass = {static_cast<GXTevAlphaArg>(s % 7), static_cast<GXTevAlphaArg>((s + 1) % 7),
                    static_cast<GXTevAlphaArg>((s + 2) % 7), static_cast<GXTevAlphaArg>((s + 3) % 7)};
    for (auto* op : {&st.colorOp, &st.alphaOp}) {
      op->op = OtherOps[s % OtherOps.size()];
      op->bias = static_cast<GXTevBias>(1 + s % 2);
      op->scale = static_cast<GXTevScale>(1 + s % 3);
      op->outReg = static_cast<GXTevRegID>(1 + s % 3);
      op->clamp = false;
    }
    st.kcSel = static_cast<GXTevKColorSel>(GX_TEV_KCSEL_K0 + s);
    st.kaSel = static_cast<GXTevKAlphaSel>(GX_TEV_KASEL_K0_R + s);
    st.texCoordId = static_cast<GXTexCoordID>(s % 8);
    st.texMapId = static_cast<GXTexMapID>((s + 1) % 8);
    st.channelId = static_cast<GXChannelID>(s % 9);
    st.tevSwapRas = static_cast<GXTevSwapSel>(1 + s % 3);
    st.tevSwapTex = static_cast<GXTevSwapSel>(1 + (s + 1) % 3);
    st.indTexStage = static_cast<GXIndTexStageID>(1 + s % 3);
    st.indTexFormat = static_cast<GXIndTexFormat>(1 + s % 3);
    st.indTexBiasSel = static_cast<GXIndTexBiasSel>(1 + s % 7);
    st.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(1 + s % 3);
    st.indTexMtxId = OtherMtx[s % OtherMtx.size()];
    st.indTexWrapS = static_cast<GXIndTexWrap>(1 + s % 6);
    st.indTexWrapT = static_cast<GXIndTexWrap>(1 + (s + 1) % 6);
    st.indTexUseOrigLOD = true;
    st.indTexAddPrev = true;
  }
  c.tevStageCount = 9;
  for (uint32_t i = 0; i < MaxColorChannels; ++i) {
    c.colorChannels[i] = {
        .matSrc = GX_SRC_VTX,
        .ambSrc = GX_SRC_VTX,
        .diffFn = static_cast<GXDiffuseFn>(1 + i % 2),
        .attnFn = static_cast<GXAttnFn>(i % 2),
        .lightingEnabled = true,
    };
  }
  for (uint32_t i = 0; i < MaxTexCoord; ++i) {
    c.tcgs[i] = {
        .type = i == 0 ? GX_TG_MTX3x4 : static_cast<GXTexGenType>(1 + i),
        .src = static_cast<GXTexGenSrc>(i),
        .mtx = static_cast<GXTexMtx>(3 * i),
        .postMtx = static_cast<GXPTTexMtx>(GX_PTTEXMTX0 + 3 * i),
        .normalize = true,
        .embossSrc = static_cast<u8>(1 + i % 7),
    };
  }
  c.alphaCompare = {GX_GREATER, 17, GX_AOP_XOR, GX_LESS, 200};
  for (uint32_t i = 0; i < MaxIndStages; ++i) {
    c.indStages[i] = {static_cast<GXTexCoordID>(1 + i % 7), static_cast<GXTexMapID>(2 + i % 6),
                      static_cast<GXIndTexScale>(1 + i), static_cast<GXIndTexScale>(2 + i)};
  }
  c.numIndStages = 3;
  return c;
}

PipelineConfig pipeline_with(const ShaderConfig& shader) {
  PipelineConfig pc{};
  pc.shaderConfig = shader;
  pc.depthFunc = GX_LEQUAL;
  pc.cullMode = GX_CULL_BACK;
  pc.blendMode = GX_BM_NONE;
  pc.blendFacSrc = GX_BL_ONE;
  pc.blendFacDst = GX_BL_ZERO;
  pc.blendOp = GX_LO_CLEAR;
  pc.dstAlpha = UINT32_MAX;
  pc.depthCompare = true;
  pc.depthUpdate = true;
  pc.alphaUpdate = true;
  pc.colorUpdate = true;
  return pc;
}

Blob blob_from(const PipelineConfig& pc, uint64_t dbHash) {
  Blob blob{.configVersion = pc.version, .dbHash = dbHash};
  blob.bytes.resize(sizeof(pc));
  std::memcpy(blob.bytes.data(), &pc, sizeof(pc));
  return blob;
}

std::filesystem::path temp_file(std::string_view name) {
  return std::filesystem::temp_directory_path() / fmt::format("aurora_gx_shader_corpus_{}", name);
}

constexpr std::string_view AllowHeader = "# aurora gx shader g2 allow v1";
constexpr std::string_view AllowEntry = "lit-normal-target";

struct AllowList {
  std::vector<std::string> entries;
  std::vector<GoldenLine> lines;
};

std::filesystem::path allow_path() {
  return std::filesystem::path{AURORA_GX_SHADER_GOLDEN}.parent_path() / "gx_shader_g2_allow.txt";
}

std::optional<uint64_t> parse_number(std::string_view text, int base) {
  uint64_t value = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
  if (text.empty() || ec != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<AllowList> parse_allow(std::string_view text, std::string& error) {
  AllowList list;
  size_t number = 0;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    ++number;
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (number == 1) {
      if (line != AllowHeader) {
        error = "missing or wrong header";
        return std::nullopt;
      }
      continue;
    }
    if (line.empty()) {
      continue;
    }
    std::vector<std::string_view> fields;
    for (size_t start = 0; start <= line.size();) {
      const size_t space = std::min(line.find(' ', start), line.size());
      fields.push_back(line.substr(start, space - start));
      start = space + 1;
    }
    const auto key = fields[0].size() == 16 ? parse_number(fields[0], 16) : std::nullopt;
    if (!key) {
      if (fields.size() < 3 || !list.lines.empty() ||
          std::find(list.entries.begin(), list.entries.end(), fields[0]) != list.entries.end()) {
        error = fmt::format("line {}: malformed entry", number);
        return std::nullopt;
      }
      list.entries.emplace_back(fields[0]);
      continue;
    }
    const auto hash = fields.size() == 4 && fields[1].size() == 16 ? parse_number(fields[1], 16) : std::nullopt;
    const auto size = fields.size() == 4 ? parse_number(fields[2], 10) : std::nullopt;
    if (!hash || !size || std::find(list.entries.begin(), list.entries.end(), fields[3]) == list.entries.end()) {
      error = fmt::format("line {}: malformed key line", number);
      return std::nullopt;
    }
    if (!list.lines.empty() && *key <= list.lines.back().key) {
      error = fmt::format("line {}: keys are not strictly ascending", number);
      return std::nullopt;
    }
    list.lines.push_back({*key, *hash, *size, std::string{fields[3]}});
  }
  if (number == 0) {
    error = "missing or wrong header";
    return std::nullopt;
  }
  return list;
}

bool in_allow_class(const Record& record) {
  const Variant variant = decode_variant(record);
  const ShaderConfig config = decode(record);
  return variant.normalAttachment != UINT32_MAX && config.attrs[GX_VA_NRM].attrType != GX_NONE &&
         build_shader_info(config).lightingEnabled;
}

std::map<uint64_t, std::string> allow_class(const std::vector<Entry>& entries) {
  std::map<uint64_t, std::string> keys;
  for (const auto& entry : entries) {
    if (in_allow_class(entry.record)) {
      keys.emplace(record_key(entry.record), entry.label);
    }
  }
  return keys;
}

TEST(GxShaderCorpusTest, ShaderConfigLayoutIsKnown) {
  // A new legacy field must bump the record format, so this has to fail before the fixture silently drifts.
  // ext is a fork field the records never carry, so RecordSize stays 868
  static_assert(sizeof(ShaderConfig) == 2724);
  EXPECT_EQ(RecordSize, 868u);
}

TEST(GxShaderCorpusTest, HandBuiltConfigRoundTrips) {
  const auto config = hand_built_config();
  const Variant variant{DstAlphaMode::Replace, 1};
  const auto record = encode(config, variant);
  ASSERT_TRUE(record.has_value());
  EXPECT_TRUE(decode(*record) == config);
  EXPECT_EQ(decode_variant(*record), variant);
}

TEST(GxShaderCorpusTest, GeneratorIsDeterministic) {
  const auto first = generate_synthetic();
  const auto second = generate_synthetic();
  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].record, second[i].record) << first[i].label;
    EXPECT_EQ(first[i].label, second[i].label);
  }
}

TEST(GxShaderCorpusTest, SyntheticRecordsAreValid) {
  ASSERT_FALSE(synthetic().empty());
  for (const auto& entry : synthetic()) {
    const auto reason = validate(entry.record);
    EXPECT_FALSE(reason.has_value()) << entry.label << ": " << reason.value_or("");
  }
}

TEST(GxShaderCorpusTest, RecordsRoundTrip) {
  for (const auto& entry : synthetic()) {
    const auto record = encode(decode(entry.record), decode_variant(entry.record));
    ASSERT_TRUE(record.has_value()) << entry.label;
    EXPECT_EQ(*record, entry.record) << entry.label;
  }
}

TEST(GxShaderCorpusTest, RecordKeysAreUnique) {
  std::map<uint64_t, const Entry*> keys;
  for (const auto& entry : synthetic()) {
    const auto [it, inserted] = keys.emplace(record_key(entry.record), &entry);
    if (!inserted) {
      EXPECT_EQ(it->second->record, entry.record) << it->second->label << " and " << entry.label << " share a key";
    }
  }
  EXPECT_EQ(keys.size(), synthetic().size());
}

TEST(GxShaderCorpusTest, SyntheticCoverageIsComplete) {
  const auto seen = collect_coverage(synthetic());
  for (const auto& key : expected_coverage()) {
    EXPECT_TRUE(seen.contains(key)) << "uncovered: " << key;
  }
}

TEST(GxShaderCorpusTest, ValidateRejectsCrashingConfigs) {
  const auto base = base_record();
  ASSERT_FALSE(validate(base).has_value());
  const size_t stage0 = RecordStagesOffset;
  const size_t tcg0 = RecordTcgsOffset;
  const size_t pos = RecordAttrsOffset + GX_VA_POS * RecordAttrStride;
  const size_t pnmtx = RecordAttrsOffset + GX_VA_PNMTXIDX * RecordAttrStride;
  struct Mutation {
    std::string_view name;
    std::vector<std::pair<size_t, uint8_t>> bytes;
  };
  const std::vector<Mutation> mutations{
      {"tevStageCount 0", {{4, 0}}},
      {"tevStageCount 17", {{4, 17}}},
      {"fogType 1", {{0, 1}}},
      {"fogType 3", {{0, 3}}},
      {"outReg 4", {{stage0 + StageColorOutReg, 4}}},
      {"bias 3", {{stage0 + StageColorBias, 3}}},
      {"kcSel 8", {{stage0 + StageKcSel, 8}}},
      {"kaSel 8", {{stage0 + StageKaSel, 8}}},
      {"channelId 9", {{stage0 + StageChannelId, 9}}},
      {"texture arg without texcoord", {{stage0 + StageTexCoordId, 0xFF}}},
      {"sampled tcg src 21", {{tcg0 + TcgSrc, 21}}},
      {"TG_TEX3 without TEX3", {{tcg0 + TcgSrc, GX_TG_TEX3}}},
      {"BINRM without NBT", {{tcg0 + TcgSrc, GX_TG_BINRM}}},
      {"postMtx 63", {{tcg0 + TcgPostMtx, 63}}},
      {"indirect scale 9", {{RecordIndStagesOffset + IndStageScaleS, 9}}},
      {"wrap 7", {{stage0 + StageWrapS, 7}}},
      {"clamp byte 2", {{stage0 + StageColorClamp, 2}}},
      {"POS absent", {{pos, GX_NONE}}},
      {"PNMTXIDX INDEX8", {{pnmtx, GX_INDEX8}, {pnmtx + 1, 1}}},
      {"DualSource with normalAttachment 1", {{6, 2}, {7, 1}}},
      {"reserved byte", {{RecordReservedOffset, 1}}},
  };
  for (const auto& mutation : mutations) {
    auto record = base;
    for (const auto& [offset, value] : mutation.bytes) {
      record.bytes[offset] = value;
    }
    EXPECT_TRUE(validate(record).has_value()) << mutation.name;
  }
}

TEST(GxShaderCorpusTest, CorpusFileRoundTrip) {
  std::vector<Entry> entries(synthetic().begin(), synthetic().begin() + 8);
  const auto path = temp_file("roundtrip.gxsc");
  ASSERT_TRUE(write_corpus(path, entries));
  std::string error;
  const auto loaded = read_corpus(path, &error);
  ASSERT_TRUE(loaded.has_value()) << error;
  sort_and_dedupe(entries);
  ASSERT_EQ(loaded->size(), entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    EXPECT_EQ((*loaded)[i].record, entries[i].record);
    EXPECT_EQ((*loaded)[i].label, entries[i].label);
  }

  const auto good = read_bytes(path);
  const auto expect_rejected = [&](std::string_view name, const std::vector<uint8_t>& bytes) {
    const auto bad = temp_file("bad.gxsc");
    write_bytes(bad, bytes);
    std::string reason;
    EXPECT_FALSE(read_corpus(bad, &reason).has_value()) << name;
    EXPECT_FALSE(reason.empty()) << name;
    std::filesystem::remove(bad);
  };
  auto badMagic = good;
  badMagic[0] = 'X';
  expect_rejected("magic", badMagic);
  auto badVersion = good;
  badVersion[4] = 2;
  expect_rejected("version", badVersion);
  auto badRecordSize = good;
  badRecordSize[6] = 0;
  expect_rejected("recordSize", badRecordSize);
  auto badReserved = good;
  badReserved[12] = 1;
  expect_rejected("reserved", badReserved);
  auto truncated = good;
  truncated.pop_back();
  expect_rejected("truncation", truncated);
  auto trailing = good;
  trailing.push_back(0);
  expect_rejected("trailing bytes", trailing);

  ASSERT_TRUE(write_corpus(path, std::span(entries).first(1)));
  auto duplicate = read_bytes(path);
  const std::vector<uint8_t> firstEntry(duplicate.begin() + 16, duplicate.end());
  duplicate.insert(duplicate.end(), firstEntry.begin(), firstEntry.end());
  duplicate[8] = 2;
  expect_rejected("duplicate key", duplicate);
  std::filesystem::remove(path);
}

TEST(GxShaderCorpusTest, ImportDerivesPipelineVariants) {
  const auto shader = decode(base_record());
  auto pc = pipeline_with(shader);
  EXPECT_EQ(pipeline_variants(pc).size(), 2u);
  std::vector<Entry> out;
  auto stats = import_blob(blob_from(pc, 1), out);
  EXPECT_EQ(stats.accepted, 2u);
  EXPECT_EQ(out.size(), 2u);
  stats = import_blob(blob_from(pc, 1), out);
  EXPECT_EQ(stats.duplicate, 2u);

  auto replace = pc;
  replace.dstAlpha = 0x80;
  const auto replaceVariants = pipeline_variants(replace);
  ASSERT_EQ(replaceVariants.size(), 2u);
  EXPECT_EQ(replaceVariants[0].dstAlphaMode, DstAlphaMode::Replace);
  EXPECT_EQ(replaceVariants[1].dstAlphaMode, DstAlphaMode::Replace);

  auto sourceAlpha = replace;
  sourceAlpha.blendMode = GX_BM_BLEND;
  sourceAlpha.blendFacSrc = GX_BL_SRCALPHA;
  sourceAlpha.blendFacDst = GX_BL_INVSRCALPHA;
  const auto sourceVariants = pipeline_variants(sourceAlpha);
  ASSERT_EQ(sourceVariants.size(), 5u);
  EXPECT_EQ(sourceVariants[0].dstAlphaMode, DstAlphaMode::DualSource);
  std::vector<Entry> sourceOut;
  EXPECT_EQ(import_blob(blob_from(sourceAlpha, 2), sourceOut).accepted, 5u);

  auto wrongVersion = blob_from(pc, 3);
  wrongVersion.configVersion = GXPipelineConfigVersion - 1;
  EXPECT_EQ(import_blob(wrongVersion, out).wrongVersion, 1u);

  auto wrongSize = blob_from(pc, 4);
  wrongSize.bytes.pop_back();
  EXPECT_EQ(import_blob(wrongSize, out).wrongSize, 1u);

  auto padded = pc;
  padded.shaderConfig.pad1 = 1;
  const auto notLegacy = import_blob(blob_from(padded, 5), out);
  EXPECT_GT(notLegacy.notLegacy, 0u);
  EXPECT_EQ(notLegacy.accepted, 0u);

  auto stochastic = pc;
  stochastic.shaderConfig.ext.stochasticMask = 1;
  const auto extRow = import_blob(blob_from(stochastic, 7), out);
  EXPECT_GT(extRow.notLegacy, 0u);
  EXPECT_EQ(extRow.accepted, 0u);

  auto badBool = blob_from(pc, 6);
  badBool.bytes[offsetof(PipelineConfig, alphaUpdate)] = 2;
  EXPECT_EQ(import_blob(badBool, out).invalid, 1u);
}

TEST(GxShaderCorpusTest, VariantsReachGenerator) {
  const auto config = decode(base_record());
  const std::array variants{
      Variant{DstAlphaMode::None, UINT32_MAX},       Variant{DstAlphaMode::None, 1},
      Variant{DstAlphaMode::Replace, UINT32_MAX},    Variant{DstAlphaMode::Replace, 1},
      Variant{DstAlphaMode::DualSource, UINT32_MAX},
  };
  std::vector<std::string> texts;
  for (const auto& variant : variants) {
    const auto record = encode(config, variant);
    ASSERT_TRUE(record.has_value());
    texts.push_back(generate_wgsl(*record));
  }
  for (size_t i = 0; i < texts.size(); ++i) {
    for (size_t j = i + 1; j < texts.size(); ++j) {
      EXPECT_NE(texts[i], texts[j]) << i << " vs " << j;
    }
  }
  EXPECT_TRUE(texts[4].starts_with("enable dual_source_blending;\n"));
}

TEST(GxShaderGoldenTest, DebugIdOutputNeedsTarget) {
  const ShaderConfig config = decode(base_record());
  const std::string legacy = build_shader_source(config, DstAlphaMode::None);
  EXPECT_EQ(legacy.find("debug_id"), std::string::npos);
  EXPECT_EQ(build_shader_source(config, DstAlphaMode::None, ShaderTargets{}), legacy);
  EXPECT_EQ(build_shader_source(config, DstAlphaMode::Replace, ShaderTargets{.normal = 1}),
            build_shader_source(config, DstAlphaMode::Replace, 1u));

  const std::string ids = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.debugId = 1});
  EXPECT_NE(ids.find("@location(1) debug_id: u32,"), std::string::npos);
  EXPECT_NE(ids.find("out.debug_id = imm._pad;"), std::string::npos);
  EXPECT_EQ(ids.find("out.normal"), std::string::npos);

  const std::string both = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.normal = 1, .debugId = 2});
  EXPECT_NE(both.find("@location(1) normal: vec4f,"), std::string::npos);
  EXPECT_NE(both.find("@location(2) debug_id: u32,"), std::string::npos);
  EXPECT_LT(both.find("out.normal"), both.find("out.debug_id"));

  const std::string dual = build_shader_source(config, DstAlphaMode::DualSource, ShaderTargets{.debugId = 1});
  EXPECT_EQ(dual.find("debug_id"), std::string::npos);
  EXPECT_EQ(dual, build_shader_source(config, DstAlphaMode::DualSource));
}

TEST(GxShaderGoldenTest, GBufferLayoutDropsLegacyNormal) {
  const gfx::SceneLayoutBase base{
      .color = wgpu::TextureFormat::BGRA8Unorm,
      .depth = wgpu::TextureFormat::Depth32Float,
      .size = {64, 64, 1},
      .samples = 1,
  };
  const gfx::RenderTargetLayout gbuffer = gfx::compose_scene_layout({.gbuffer = true}, base);
  EXPECT_EQ(shader_targets(gbuffer).normal, UINT32_MAX);
  EXPECT_EQ(shader_targets(gbuffer).debugId, UINT32_MAX);
  EXPECT_EQ(shader_targets(gfx::compose_scene_layout({.normal = true, .gbuffer = true}, base)).normal, UINT32_MAX);
  EXPECT_EQ(shader_targets(gfx::compose_scene_layout({.normal = true}, base)).normal, 1u);
  const ShaderTargets ids = shader_targets(gfx::compose_scene_layout({.gbuffer = true, .debugId = true}, base));
  EXPECT_EQ(ids.normal, UINT32_MAX);
  EXPECT_EQ(ids.debugId, 5u);
  const ShaderTargets legacyIds = shader_targets(gfx::compose_scene_layout({.normal = true, .debugId = true}, base));
  EXPECT_EQ(legacyIds.normal, 1u);
  EXPECT_EQ(legacyIds.debugId, 2u);

  const ShaderTargets targets = shader_targets(gbuffer);
  EXPECT_EQ(targets.motion, 2u);
  EXPECT_EQ(targets.reactive, 3u);
  size_t compared = 0;
  for (const Entry& entry : synthetic()) {
    const ShaderConfig config = decode(entry.record);
    const DstAlphaMode mode = decode_variant(entry.record).dstAlphaMode;
    if (config.ext != RenderExtConfig{} || mode == DstAlphaMode::DualSource) {
      continue;
    }
    const std::string text = build_shader_source(config, mode, targets);
    EXPECT_EQ(text, build_shader_source(config, mode, ShaderTargets{.motion = 2, .reactive = 3})) << entry.label;
    EXPECT_EQ(text.find("in.mv_nrm"), std::string::npos) << entry.label;
    EXPECT_EQ(text.find("normal: vec4f"), std::string::npos) << entry.label;
    ++compared;
  }
  EXPECT_GT(compared, 0u);
  RecordProperty("compared", static_cast<int>(compared));
}

TEST(GxShaderGoldenTest, FixtureIsWellFormed) {
  const auto text = read_text(AURORA_GX_SHADER_GOLDEN);
  ASSERT_FALSE(text.empty()) << "missing fixture " << AURORA_GX_SHADER_GOLDEN;
  std::string error;
  const auto lines = parse_golden(text, &error);
  ASSERT_TRUE(lines.has_value()) << error;
  EXPECT_FALSE(lines->empty());
  for (size_t i = 0; i < lines->size(); ++i) {
    EXPECT_FALSE((*lines)[i].label.empty());
    if (i > 0) {
      EXPECT_LT((*lines)[i - 1].key, (*lines)[i].key);
    }
  }
}

TEST(GxShaderGoldenTest, AllowListIsWellFormed) {
  std::string error;
  const auto allow = parse_allow(read_text(allow_path()), error);
  ASSERT_TRUE(allow.has_value()) << allow_path().string() << ": " << error;
  ASSERT_EQ(allow->entries.size(), 1u);
  EXPECT_EQ(allow->entries.front(), AllowEntry);
  auto classKeys = allow_class(synthetic());
  EXPECT_FALSE(classKeys.empty());
  for (const auto& line : allow->lines) {
    EXPECT_EQ(line.label, AllowEntry);
    if (classKeys.erase(line.key) == 0) {
      ADD_FAILURE() << fmt::format("allow-listed key {:016x} is outside the synthetic allow class", line.key);
    }
  }
  for (const auto& [key, label] : classKeys) {
    ADD_FAILURE() << fmt::format("allow class key {:016x} ({}) is not listed", key, label);
  }
}

TEST(GxShaderGoldenTest, MatchesFixture) {
  std::string error;
  const auto fixture = parse_golden(read_text(AURORA_GX_SHADER_GOLDEN), &error);
  ASSERT_TRUE(fixture.has_value()) << error;
  const auto allow = parse_allow(read_text(allow_path()), error);
  ASSERT_TRUE(allow.has_value()) << allow_path().string() << ": " << error;
  auto entries = synthetic();
  sort_and_dedupe(entries);
  const auto actual = golden_lines(entries);
  const auto classKeys = allow_class(entries);

  std::map<uint64_t, const GoldenLine*> expected;
  for (const auto& line : *fixture) {
    expected.emplace(line.key, &line);
  }
  // Upstream's text for these records is invalid WGSL (6c8e4c5); the fixture keeps it so merges still compare
  std::map<uint64_t, const GoldenLine*> allowed;
  for (const auto& line : allow->lines) {
    allowed.emplace(line.key, &line);
  }
  std::vector<uint64_t> offending;
  for (const auto& line : actual) {
    const auto it = expected.find(line.key);
    if (it == expected.end()) {
      ADD_FAILURE() << fmt::format("new key {:016x} ({}): run regen-golden --mode add", line.key, line.label);
      offending.push_back(line.key);
      continue;
    }
    const GoldenLine* want = it->second;
    const bool inClass = classKeys.contains(line.key);
    if (const auto listed = allowed.find(line.key); listed != allowed.end()) {
      if (!inClass) {
        ADD_FAILURE() << fmt::format("allow-listed key {:016x} ({}) is outside the allow class", line.key, line.label);
      }
      want = listed->second;
      allowed.erase(listed);
    } else if (inClass) {
      ADD_FAILURE() << fmt::format("allow class key {:016x} ({}) is not listed: run regen-golden", line.key,
                                   line.label);
    }
    if (want->wgslHash != line.wgslHash || want->wgslSize != line.wgslSize) {
      ADD_FAILURE() << fmt::format(
          "WGSL changed for {:016x} ({}): G2 violation, legacy shader text must match upstream", line.key, line.label);
      offending.push_back(line.key);
    }
    expected.erase(it);
  }
  for (const auto& [key, line] : expected) {
    ADD_FAILURE() << fmt::format("missing key {:016x} ({}): generator coverage was lost", key, line->label);
  }
  for (const auto& listed : allowed) {
    ADD_FAILURE() << fmt::format("allow-listed key {:016x} is not a synthetic record", listed.first);
  }
  if (!::testing::Test::HasFailure()) {
    return;
  }
  const std::filesystem::path outDir = AURORA_GX_SHADER_GOLDEN_OUT;
  std::filesystem::create_directories(outDir);
  const auto actualText = format_golden(actual);
  write_bytes(outDir / "actual.txt", std::vector<uint8_t>(actualText.begin(), actualText.end()));
  size_t written = 0;
  for (const auto& entry : entries) {
    const auto key = record_key(entry.record);
    if (written == 32 || std::find(offending.begin(), offending.end(), key) == offending.end()) {
      continue;
    }
    const auto text = generate_wgsl(entry.record);
    write_bytes(outDir / fmt::format("{:016x}.wgsl", key), std::vector<uint8_t>(text.begin(), text.end()));
    ++written;
  }
}

TEST(GxShaderGoldenTest, LocalCorpusGenerates) {
  const char* path = std::getenv("AURORA_GX_SHADER_CORPUS");
  if (path == nullptr || *path == '\0') {
    GTEST_SKIP() << "AURORA_GX_SHADER_CORPUS is not set";
  }
  std::string error;
  const auto entries = read_corpus(path, &error);
  ASSERT_TRUE(entries.has_value()) << error;
  for (const auto& entry : *entries) {
    const auto reason = validate(entry.record);
    ASSERT_FALSE(reason.has_value()) << entry.label << ": " << reason.value_or("");
    EXPECT_FALSE(generate_wgsl(entry.record).empty()) << entry.label;
  }
}

} // namespace
} // namespace aurora::gx::corpus

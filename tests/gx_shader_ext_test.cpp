#include <gtest/gtest.h>

#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "gx/shader_info.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gx {
namespace {
using Site = std::pair<u32, u32>;

constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
constexpr float Infinity = std::numeric_limits<float>::infinity();

TevStage textured_stage(u32 coord, u32 map) {
  TevStage stage;
  stage.colorPass = {GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO};
  stage.alphaPass = {GX_CA_ZERO, GX_CA_APREV, GX_CA_TEXA, GX_CA_ZERO};
  stage.texCoordId = static_cast<GXTexCoordID>(coord);
  stage.texMapId = static_cast<GXTexMapID>(map);
  return stage;
}

// Each site is (texcoord, texmap), sampled by its own stage through an MTX2x4 texgen from the matching TEXn
ShaderConfig config_with(std::span<const Site> sites) {
  ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_VTX};
  config.colorChannels[GX_ALPHA0] = {.matSrc = GX_SRC_VTX};
  for (size_t i = 0; i < sites.size(); ++i) {
    const auto [coord, map] = sites[i];
    config.attrs[GX_VA_TEX0 + coord] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
    config.tcgs[coord] = {.type = GX_TG_MTX2x4, .src = static_cast<GXTexGenSrc>(GX_TG_TEX0 + coord)};
    config.tevStages[i] = textured_stage(coord, map);
  }
  config.tevStageCount = static_cast<u32>(sites.size());
  return config;
}

ShaderConfig single_site() {
  constexpr std::array sites{Site{0, 0}};
  return config_with(sites);
}

ShaderConfig with_mask(ShaderConfig config, u8 mask) {
  config.ext.stochasticMask = mask;
  return config;
}

std::string wgsl(const ShaderConfig& config) { return build_shader_source(config, DstAlphaMode::None); }

size_t count(std::string_view text, std::string_view needle) {
  size_t found = 0;
  for (size_t pos = text.find(needle); pos != std::string_view::npos; pos = text.find(needle, pos + 1)) {
    ++found;
  }
  return found;
}

std::string uniform_struct(const std::string& text) {
  const size_t begin = text.find("struct Uniform {");
  const size_t end = text.find("};", begin);
  return begin == std::string::npos || end == std::string::npos ? std::string{} : text.substr(begin, end - begin);
}

std::string vertex_output(const std::string& text) {
  const size_t begin = text.find("struct VertexOutput {");
  const size_t end = text.find("};", begin);
  return begin == std::string::npos || end == std::string::npos ? std::string{} : text.substr(begin, end - begin);
}

Vec4<float> read_vec4(std::span<const uint8_t> bytes, size_t offset) {
  Vec4<float> value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

// Eight directly sampled maps with every optional uniform block on, so the hex tail no longer fits
ShaderConfig heavy_config(bool fog, bool postMtx) {
  std::array<Site, 8> sites{};
  for (u32 i = 0; i < sites.size(); ++i) {
    sites[i] = {i, i};
  }
  ShaderConfig config = config_with(sites);
  if (postMtx) {
    for (auto& tcg : config.tcgs) {
      tcg.postMtx = GX_PTTEXMTX0;
    }
  }
  for (const auto channel : {GX_COLOR0, GX_ALPHA0, GX_COLOR1, GX_ALPHA1}) {
    config.colorChannels[channel] = {.matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .lightingEnabled = true};
  }
  auto& regs = config.tevStages[8];
  regs.colorPass = {GX_CC_C0, GX_CC_C1, GX_CC_C2, GX_CC_RASC};
  regs.alphaPass = {GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
  regs.kaSel = GX_TEV_KASEL_K0_A;
  regs.channelId = GX_COLOR0A0;
  auto& konst = config.tevStages[9];
  konst.colorPass = {GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC};
  konst.alphaPass = {GX_CA_KONST, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
  konst.kcSel = GX_TEV_KCSEL_K1;
  konst.kaSel = GX_TEV_KASEL_K2_A;
  konst.channelId = GX_COLOR1A1;
  auto& last = config.tevStages[10];
  last.colorPass = {GX_CC_KONST, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV};
  last.kcSel = GX_TEV_KCSEL_K3;
  auto& indirect = config.tevStages[11];
  indirect.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV};
  indirect.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_APREV};
  indirect.texCoordId = GX_TEXCOORD0;
  indirect.indTexMtxId = GX_ITM_0;
  config.indStages[0] = {GX_TEXCOORD0, GX_TEXMAP7, GX_ITS_1, GX_ITS_1};
  config.numIndStages = 1;
  config.tevStageCount = 12;
  config.fogType = fog ? GX_FOG_PERSP_LIN : GX_FOG_NONE;
  return config;
}

class GxShaderExtTest : public ::testing::Test {
protected:
  void SetUp() override {
    m_alignment = gfx::detail::resources().limits.minUniformBufferOffsetAlignment;
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
    g_gxState = {};
    render_ext::g_stochasticUniforms = {};
  }

  void TearDown() override {
    g_gxState = {};
    render_ext::g_stochasticUniforms = {};
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = m_alignment;
  }

private:
  uint32_t m_alignment = 0;
};

TEST_F(GxShaderExtTest, LayoutIsFrozen) {
  static_assert(sizeof(RenderExtConfig) == 4);
  static_assert(offsetof(ShaderConfig, ext) == 2720);
  static_assert(sizeof(ShaderConfig) == 2724);
  static_assert(sizeof(PipelineConfig) == 2776);
  static_assert(offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, ext) == 2728);
  EXPECT_EQ(GXPipelineConfigVersion, 14u);
  EXPECT_EQ(RenderShaderRev, 1u);
  EXPECT_EQ(GXExtRowVersion, 65550u);

  const auto bytes = [](const RenderExtConfig& ext) { return std::bit_cast<std::array<uint8_t, 4>>(ext); };
  EXPECT_EQ(bytes(RenderExtConfig{.output = 3}), (std::array<uint8_t, 4>{0x03, 0, 0, 0}));
  EXPECT_EQ(bytes(RenderExtConfig{.velocity = 1}), (std::array<uint8_t, 4>{0x04, 0, 0, 0}));
  EXPECT_EQ(bytes(RenderExtConfig{.tessFetch = 1}), (std::array<uint8_t, 4>{0x08, 0, 0, 0}));
  EXPECT_EQ(bytes(RenderExtConfig{.pom = 1}), (std::array<uint8_t, 4>{0x10, 0, 0, 0}));
  EXPECT_EQ(bytes(RenderExtConfig{.fwdEnv = 1}), (std::array<uint8_t, 4>{0x20, 0, 0, 0}));
  EXPECT_EQ(bytes(RenderExtConfig{.stochasticMask = 0xA5, .layer0 = 3, .layer1 = 8}),
            (std::array<uint8_t, 4>{0, 0xA5, 3, 8}));
}

TEST_F(GxShaderExtTest, PersistedVersionTagsExtRows) {
  PipelineConfig config{};
  EXPECT_EQ(persisted_version(config), 14u);
  config.shaderConfig.ext.stochasticMask = 1;
  EXPECT_EQ(persisted_version(config), GXExtRowVersion);
  EXPECT_EQ(persisted_version(render_ext::legacy_config(config)), 14u);
  config.shaderConfig.ext = {.output = 1};
  EXPECT_EQ(persisted_version(config), GXExtRowVersion);
  EXPECT_EQ(persisted_version(render_ext::legacy_config(config)), GXExtRowVersion);
}

TEST_F(GxShaderExtTest, FallbackConfigClearsOutputAndVelocity) {
  PipelineConfig config{};
  config.shaderConfig.ext = {.output = 2, .velocity = 1, .pom = 1, .stochasticMask = 0b11, .layer0 = 1, .layer1 = 2};
  EXPECT_TRUE(render_ext::fallback_config(config).shaderConfig.ext == RenderExtConfig{});
  EXPECT_EQ(persisted_version(render_ext::fallback_config(config)), 14u);
  const RenderExtConfig legacy = render_ext::legacy_config(config).shaderConfig.ext;
  EXPECT_EQ(legacy.output, 2u);
  EXPECT_EQ(legacy.velocity, 1u);

  for (const RenderExtConfig ext : {RenderExtConfig{}, RenderExtConfig{.stochasticMask = 0b101},
                                    RenderExtConfig{.pom = 1, .layer0 = 3, .layer1 = 4}}) {
    config.shaderConfig.ext = ext;
    const PipelineConfig fallback = render_ext::fallback_config(config);
    const PipelineConfig expected = render_ext::legacy_config(config);
    EXPECT_EQ(std::memcmp(&fallback, &expected, sizeof(PipelineConfig)), 0);
  }
}

TEST_F(GxShaderExtTest, NormalizeKeepsOnlyEligibleStochastic) {
  const ShaderInfo info = build_shader_info(single_site());
  ASSERT_EQ(info.stochasticEligible.to_ulong(), 1u);
  const RenderExtConfig ext{.output = 2, .stochasticMask = 0b111, .layer0 = 1};
  const RenderExtConfig normalized = render_ext::normalize(ext, info);
  EXPECT_EQ(normalized.stochasticMask, 1u);
  EXPECT_EQ(normalized.output, 2u);
  EXPECT_EQ(normalized.layer0, 1u);
  EXPECT_TRUE(render_ext::normalize(normalized, info) == normalized);
  const RenderExtConfig legacy = render_ext::without_texture_bits(ext);
  EXPECT_EQ(legacy.stochasticMask, 0u);
  EXPECT_EQ(legacy.layer0, 0u);
  EXPECT_EQ(legacy.output, 2u);
}

TEST_F(GxShaderExtTest, NonStochasticExtLeavesTextUnchanged) {
  const ShaderConfig legacy = single_site();
  const std::string expected = wgsl(legacy);
  ShaderConfig other = legacy;
  other.ext = {.output = 1, .velocity = 1, .tessFetch = 1, .fwdEnv = 1, .layer0 = 2, .layer1 = 3};
  EXPECT_EQ(wgsl(other), expected);
  EXPECT_EQ(wgsl(with_mask(legacy, 0b1000)), expected);
  EXPECT_EQ(build_shader_info(other).uniformSize, build_shader_info(legacy).uniformSize);
}

TEST_F(GxShaderExtTest, EligibleMapUsesHexSampling) {
  const std::string text = wgsl(with_mask(single_site(), 1));
  EXPECT_NE(text.find("// Hex tiling after Mikkelsen 2022 (mmikk/hextile-demo, MIT)"), std::string::npos);
  EXPECT_NE(text.find("fn hex_taps(uv: vec2f, invScale: f32, rotation: f32, texSize: vec2f) -> HexTaps"),
            std::string::npos);
  EXPECT_EQ(count(text, "fn hex_tex0("), 1u);
  EXPECT_EQ(count(text, "textureSampleGrad(tex0, tex0_samp,"), 3u);
  EXPECT_NE(text.find("let hex0_0 = hex_tex0(tex0_uv, ubuf.tex0_size_bias.z);"), std::string::npos);
  EXPECT_NE(text.find("var sampled0 = hex0_0;"), std::string::npos);
  EXPECT_NE(text.find("\n    tex0_hex: vec4f,"), std::string::npos);
  EXPECT_EQ(text.find("textureSampleBias(tex0"), std::string::npos);
  EXPECT_EQ(text.find("for ("), std::string::npos);
}

TEST_F(GxShaderExtTest, IneligibleSitesStayLegacy) {
  constexpr std::array sites{Site{0, 0}, Site{1, 1}, Site{2, 2}, Site{3, 3}};
  ShaderConfig config = config_with(sites);
  config.tcgs[1].type = GX_TG_MTX3x4;
  config.tcgs[2].src = GX_TG_POS;
  config.tevStages[3].indTexWrapS = GX_ITW_16;
  const ShaderInfo info = build_shader_info(config);
  EXPECT_EQ(info.stochasticEligible.to_ulong(), 1u);
  const std::string text = wgsl(with_mask(config, 0b1111));
  EXPECT_EQ(count(text, "fn hex_tex"), 1u);
  EXPECT_NE(text.find("var sampled0 = hex0_0;"), std::string::npos);
  for (const std::string_view legacy : {"textureSampleBias(tex1", "textureSampleBias(tex2", "textureSampleBias(tex3"}) {
    EXPECT_NE(text.find(legacy), std::string::npos) << legacy;
  }
  EXPECT_EQ(text.find("tex1_hex"), std::string::npos);

  ShaderConfig lines = single_site();
  lines.lineMode = 1;
  EXPECT_TRUE(build_shader_info(lines).stochasticEligible.none());
}

TEST_F(GxShaderExtTest, RepeatedSitesShareTaps) {
  constexpr std::array sites{Site{0, 0}, Site{0, 0}, Site{1, 0}};
  const std::string text = wgsl(with_mask(config_with(sites), 1));
  EXPECT_EQ(count(text, "fn hex_tex0("), 1u);
  EXPECT_EQ(count(text, "let hex0_0 = "), 1u);
  EXPECT_EQ(count(text, "let hex0_1 = "), 1u);
  EXPECT_NE(text.find("var sampled0 = hex0_0;"), std::string::npos);
  EXPECT_NE(text.find("var sampled1 = hex0_0;"), std::string::npos);
  EXPECT_NE(text.find("var sampled2 = hex0_1;"), std::string::npos);
}

TEST_F(GxShaderExtTest, UniformIsLegacyPrefix) {
  constexpr std::array sites{Site{0, 0}, Site{1, 2}};
  const ShaderConfig legacy = config_with(sites);
  const ShaderConfig ext = with_mask(legacy, 0b101);
  const std::string legacyStruct = uniform_struct(wgsl(legacy));
  const std::string extStruct = uniform_struct(wgsl(ext));
  ASSERT_FALSE(legacyStruct.empty());
  ASSERT_EQ(legacyStruct.back(), '\n');
  EXPECT_EQ(extStruct,
            legacyStruct.substr(0, legacyStruct.size() - 1) + "\n    tex0_hex: vec4f,\n    tex2_hex: vec4f,\n");
  const std::vector<uint8_t> legacyBytes = testing::uniform_bytes(build_shader_info(legacy));
  const std::vector<uint8_t> extBytes = testing::uniform_bytes(build_shader_info(ext));
  ASSERT_EQ(extBytes.size(), legacyBytes.size() + 32);
  EXPECT_TRUE(std::equal(legacyBytes.begin(), legacyBytes.end(), extBytes.begin()));
  EXPECT_EQ(build_shader_info(ext).uniformSize, gfx::align_uniform(static_cast<uint32_t>(extBytes.size())));
}

TEST_F(GxShaderExtTest, BudgetDropsEligibility) {
  EXPECT_TRUE(stochastic_fits(3824, 1));
  EXPECT_FALSE(stochastic_fits(3824, 2));
  EXPECT_TRUE(stochastic_fits(1000, 8));

  const ShaderInfo roomy = build_shader_info(heavy_config(false, false));
  EXPECT_EQ(roomy.stochasticEligible.to_ulong(), 0x7Fu);

  const uint64_t drops = render_ext::g_stats.budgetDrops.load();
  const ShaderInfo full = build_shader_info(with_mask(heavy_config(true, true), 0x7F));
  EXPECT_EQ(full.uniformSize, MaxUniformSize);
  EXPECT_TRUE(full.stochasticEligible.none());
  EXPECT_TRUE(full.stochasticTextures.none());
  EXPECT_EQ(render_ext::g_stats.budgetDrops.load(), drops + 1);
}

TEST_F(GxShaderExtTest, MigratesV13Rows) {
  PipelineConfig config{};
  config.shaderConfig = single_site();
  config.depthFunc = GX_LEQUAL;
  config.cullMode = GX_CULL_BACK;
  config.blendMode = GX_BM_BLEND;
  config.blendFacSrc = GX_BL_SRCALPHA;
  config.blendFacDst = GX_BL_INVSRCALPHA;
  config.blendOp = GX_LO_COPY;
  config.dstAlpha = 0x40;
  config.polygonOffsetBits = 0x12345678;
  config.depthCompare = true;
  config.colorUpdate = true;
  std::array<uint8_t, sizeof(PipelineConfig)> current{};
  std::memcpy(current.data(), &config, sizeof(config));
  std::vector<uint8_t> v13(current.begin(), current.begin() + 2728);
  v13.insert(v13.end(), current.begin() + 2732, current.end());
  ASSERT_EQ(v13.size(), 2772u);
  const uint32_t version13 = 13;
  std::memcpy(v13.data(), &version13, sizeof(version13));

  const auto migrated = migrate_v13_config(v13);
  ASSERT_TRUE(migrated.has_value());
  EXPECT_EQ(std::memcmp(&*migrated, &config, sizeof(config)), 0);
  EXPECT_EQ(migrated->version, GXPipelineConfigVersion);
  EXPECT_TRUE(migrated->shaderConfig.ext == RenderExtConfig{});

  auto shortRow = v13;
  shortRow.pop_back();
  EXPECT_FALSE(migrate_v13_config(shortRow).has_value());
  auto otherVersion = v13;
  otherVersion[0] = 12;
  EXPECT_FALSE(migrate_v13_config(otherVersion).has_value());
  EXPECT_FALSE(migrate_v13_config(std::span<const uint8_t>{current}).has_value());
}

TEST_F(GxShaderExtTest, PackFollowsC1Domain) {
  const auto pack = [](render_ext::StochasticParams params) { return render_ext::pack(params); };
  EXPECT_TRUE(pack({}) == (Vec4<float>{1.f, 0.f, 0.6f, 7.f}));
  EXPECT_FLOAT_EQ(pack({.scale = 2.f}).x(), 0.5f);
  EXPECT_FLOAT_EQ(pack({.scale = 16.f}).x(), 1.f / 16.f);
  EXPECT_FLOAT_EQ(pack({.scale = 32.f}).x(), 1.f / 16.f);
  EXPECT_FLOAT_EQ(pack({.scale = 0.25f}).x(), 4.f);
  for (const float bad : {0.f, -1.f, NaN, Infinity}) {
    EXPECT_FLOAT_EQ(pack({.scale = bad}).x(), 1.f) << bad;
  }
  EXPECT_FLOAT_EQ(pack({.rotation = 2.f}).y(), 1.f);
  EXPECT_FLOAT_EQ(pack({.rotation = -1.f}).y(), 0.f);
  EXPECT_FLOAT_EQ(pack({.rotation = NaN}).y(), 0.f);
  EXPECT_FLOAT_EQ(pack({.contrast = 1.5f}).z(), 1.f);
  EXPECT_FLOAT_EQ(pack({.contrast = NaN}).z(), 0.6f);
  EXPECT_FLOAT_EQ(pack({.exponent = 0.f}).w(), 1.f);
  EXPECT_FLOAT_EQ(pack({.exponent = 100.f}).w(), 32.f);
  EXPECT_FLOAT_EQ(pack({.exponent = NaN}).w(), 7.f);
}

TEST_F(GxShaderExtTest, RotationUsesTexelAspect) {
  const std::string text = wgsl(with_mask(single_site(), 1));
  EXPECT_NE(text.find("// Rotate in texel space so non-square textures do not shear"), std::string::npos);
  EXPECT_NE(text.find("c + hex_rotate((uv - c) * texSize, cs) / texSize + h.xy"), std::string::npos);
  EXPECT_NE(text.find("let sz = max(ubuf.tex0_size_bias.xy, vec2f(1.0));"), std::string::npos);
  EXPECT_NE(text.find("let t = hex_taps(uv, ubuf.tex0_hex.x, ubuf.tex0_hex.y, sz);"), std::string::npos);
  EXPECT_NE(text.find("let dx = dpdx(uv) * g * sz;"), std::string::npos);
  for (const std::string_view tap : {"t.uv0, hex_rotate(dx, t.cs0) / sz, hex_rotate(dy, t.cs0) / sz",
                                     "t.uv1, hex_rotate(dx, t.cs1) / sz, hex_rotate(dy, t.cs1) / sz",
                                     "t.uv2, hex_rotate(dx, t.cs2) / sz, hex_rotate(dy, t.cs2) / sz"}) {
    EXPECT_NE(text.find(tap), std::string::npos) << tap;
  }
}

TEST_F(GxShaderExtTest, UniformTailMatchesLanes) {
  constexpr std::array sites{Site{0, 1}, Site{1, 4}};
  const ShaderConfig legacy = config_with(sites);
  const ShaderConfig ext = with_mask(legacy, 1u << 1 | 1u << 4);
  g_gxState.textures[1].texObj.mWidth = 32;
  g_gxState.textures[1].texObj.mHeight = 16;
  g_gxState.textures[4].texObj.mWidth = 64;
  g_gxState.textures[4].texObj.mHeight = 128;
  const Vec4<float> lane1{0.5f, 0.25f, 0.6f, 7.f};
  const Vec4<float> lane4{0.25f, 1.f, 0.3f, 9.f};
  render_ext::g_stochasticUniforms[1] = lane1;
  render_ext::g_stochasticUniforms[4] = lane4;

  const ShaderInfo extInfo = build_shader_info(ext);
  const std::vector<uint8_t> legacyBytes = testing::uniform_bytes(build_shader_info(legacy));
  const std::vector<uint8_t> extBytes = testing::uniform_bytes(extInfo);
  ASSERT_EQ(extBytes.size(), legacyBytes.size() + 32);
  EXPECT_EQ(extInfo.uniformSize, gfx::align_uniform(static_cast<uint32_t>(extBytes.size())));
  const size_t tail = extBytes.size() - 32;
  EXPECT_TRUE(read_vec4(extBytes, tail) == lane1);
  EXPECT_TRUE(read_vec4(extBytes, tail + 16) == lane4);
  EXPECT_TRUE(read_vec4(extBytes, tail - 16) == (Vec4<float>{64.f, 128.f, 0.f, 0.f}));
  EXPECT_TRUE(read_vec4(extBytes, tail - 32) == (Vec4<float>{32.f, 16.f, 0.f, 0.f}));
  EXPECT_TRUE(std::equal(legacyBytes.begin(), legacyBytes.end(), extBytes.begin()));
}

TEST_F(GxShaderExtTest, LitNormalTargetDeclaresNormalVarying) {
  ShaderConfig config = single_site();
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .lightingEnabled = true};
  config.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  config.tevStages[0].channelId = GX_COLOR0A0;
  ASSERT_TRUE(build_shader_info(config).lightingEnabled);
  for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
    const std::string text = build_shader_source(config, mode, 1u);
    EXPECT_NE(vertex_output(text).find("mv_nrm: vec3f,"), std::string::npos) << static_cast<int>(mode);
    EXPECT_NE(text.find("out.mv_nrm = mv_nrm;"), std::string::npos) << static_cast<int>(mode);
    EXPECT_NE(text.find("dot(in.mv_nrm, in.mv_nrm)"), std::string::npos) << static_cast<int>(mode);
  }
  const std::string untargeted = build_shader_source(config, DstAlphaMode::None, UINT32_MAX);
  EXPECT_FALSE(vertex_output(untargeted).empty());
  EXPECT_EQ(vertex_output(untargeted).find("mv_nrm"), std::string::npos);
  EXPECT_EQ(untargeted.find("in.mv_nrm"), std::string::npos);
}
} // namespace
} // namespace aurora::gx

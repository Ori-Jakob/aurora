#include <gtest/gtest.h>

#include "gx_shader_corpus.hpp"

#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include "gfx/scene_targets.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "gx/render_ext.hpp"
#include "gx/shader_info.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

size_t location_count(const std::string& text) { return count(vertex_output(text), "@location("); }

// NRM, two unlit vertex-colour channels and eight MTX3x4 texgens: every legacy varying at once
ShaderConfig worst_case_varyings() {
  std::array<Site, MaxTexCoord> sites{};
  for (u32 i = 0; i < sites.size(); ++i) {
    sites[i] = {i, i};
  }
  ShaderConfig config = config_with(sites);
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.colorChannels[GX_COLOR1] = {.matSrc = GX_SRC_VTX};
  config.colorChannels[GX_ALPHA1] = {.matSrc = GX_SRC_VTX};
  for (auto& tcg : config.tcgs) {
    tcg.type = GX_TG_MTX3x4;
  }
  config.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_RASC, GX_CC_TEXC, GX_CC_ZERO};
  config.tevStages[0].channelId = GX_COLOR0A0;
  config.tevStages[1].colorPass = {GX_CC_ZERO, GX_CC_RASC, GX_CC_TEXC, GX_CC_CPREV};
  config.tevStages[1].channelId = GX_COLOR1A1;
  return config;
}

size_t check_varyings(std::span<const corpus::Entry> entries) {
  size_t maxLocations = 0;
  for (const corpus::Entry& entry : entries) {
    const ShaderConfig config = corpus::decode(entry.record);
    const DstAlphaMode mode = corpus::decode_variant(entry.record).dstAlphaMode;
    const u32 planned = build_shader_info(config).varyings;
    EXPECT_LE(planned, MaxGxVaryings) << entry.label;
    for (const ShaderTargets targets :
         {ShaderTargets{}, ShaderTargets{.normal = 1}, ShaderTargets{.normal = 1, .motion = 2}}) {
      // MRT layouts never select dual-source blending, so motion text never meets it
      if (targets.motion != UINT32_MAX && mode == DstAlphaMode::DualSource) {
        continue;
      }
      const size_t locations = location_count(build_shader_source(config, mode, targets));
      EXPECT_EQ(locations, varying_count(config, targets)) << entry.label << " normal " << targets.normal;
      EXPECT_LE(locations, planned) << entry.label << " normal " << targets.normal;
      maxLocations = std::max(maxLocations, locations);
    }
    if (mode != DstAlphaMode::DualSource) {
      ShaderConfig marker = config;
      marker.ext.output = 1;
      const ShaderTargets gbuffer{.motion = 2, .reactive = 3, .gbufferNormal = 1};
      const size_t locations = location_count(build_shader_source(marker, mode, gbuffer));
      EXPECT_EQ(locations, varying_count(marker, gbuffer)) << entry.label << " receiver normal";
      EXPECT_LE(locations, planned) << entry.label << " receiver normal";
      maxLocations = std::max(maxLocations, locations);
    }
  }
  return maxLocations;
}

TEST_F(GxShaderExtTest, VaryingCountMatchesGenerator) {
  const size_t maxLocations = check_varyings(corpus::generate_synthetic());
  EXPECT_GT(maxLocations, 0u);
  RecordProperty("maxVaryings", static_cast<int>(maxLocations));
}

TEST_F(GxShaderExtTest, VaryingWorstCaseFitsBudget) {
  const ShaderConfig config = worst_case_varyings();
  const ShaderInfo info = build_shader_info(config);
  ASSERT_FALSE(info.lightingEnabled);
  EXPECT_EQ(info.sampledColorChannels.count(), 2u);
  EXPECT_EQ(info.sampledTexCoords.count(), MaxTexCoord);
  EXPECT_EQ(static_cast<u32>(info.varyings), 13u);
  EXPECT_LE(static_cast<u32>(info.varyings), MaxGxVaryings);
  const std::string text = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.normal = 1});
  EXPECT_EQ(location_count(text), 11u);
  EXPECT_EQ(count(vertex_output(text), "_uvw: vec3f,"), MaxTexCoord);
  EXPECT_EQ(varying_count(config, ShaderTargets{.normal = 1}), 11u);
  const ShaderTargets motion{.normal = 1, .motion = 2};
  EXPECT_EQ(location_count(build_shader_source(config, DstAlphaMode::None, motion)), 13u);
  EXPECT_EQ(varying_count(config, motion), 13u);
}

TEST_F(GxShaderExtTest, LitDrawWithNormalTargetCountsNormal) {
  ShaderConfig config = worst_case_varyings();
  config.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .lightingEnabled = true};
  const ShaderInfo info = build_shader_info(config);
  ASSERT_TRUE(info.lightingEnabled);
  EXPECT_EQ(static_cast<u32>(info.varyings), 13u);
  EXPECT_EQ(location_count(build_shader_source(config, DstAlphaMode::None, ShaderTargets{.normal = 1})), 11u);
  EXPECT_EQ(varying_count(config, ShaderTargets{.normal = 1}), 11u);
  EXPECT_EQ(location_count(build_shader_source(config, DstAlphaMode::None, ShaderTargets{})), 10u);
  EXPECT_EQ(varying_count(config, ShaderTargets{}), 10u);
}

TEST_F(GxShaderExtTest, LocalCorpusVaryingsFit) {
  const char* path = std::getenv("AURORA_GX_SHADER_CORPUS");
  if (path == nullptr || *path == '\0') {
    GTEST_SKIP() << "AURORA_GX_SHADER_CORPUS is not set";
  }
  std::string error;
  const auto entries = corpus::read_corpus(path, &error);
  ASSERT_TRUE(entries.has_value()) << error;
  ASSERT_FALSE(entries->empty());
  const size_t maxLocations = check_varyings(*entries);
  RecordProperty("records", static_cast<int>(entries->size()));
  RecordProperty("maxVaryings", static_cast<int>(maxLocations));
}

std::string fragment_output(const std::string& text) {
  const size_t begin = text.find("struct FragmentOutput {");
  const size_t end = text.find("};", begin);
  return begin == std::string::npos || end == std::string::npos ? std::string{} : text.substr(begin, end - begin);
}

ShaderConfig with_palette(ShaderConfig config) {
  config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
  return config;
}

TEST_F(GxShaderExtTest, MotionNeedsMotionTarget) {
  const ShaderConfig config = single_site();
  for (const std::string& text :
       {wgsl(config), build_shader_source(config, DstAlphaMode::None, 1u),
        build_shader_source(config, DstAlphaMode::None, ShaderTargets{.normal = 1, .debugId = 2})}) {
    EXPECT_EQ(text.find("mo_cur"), std::string::npos);
    EXPECT_EQ(text.find("gx_record_word"), std::string::npos);
    EXPECT_EQ(text.find("motion: vec2f"), std::string::npos);
  }
  const ShaderTargets targets{.motion = 2};
  const std::string text = build_shader_source(config, DstAlphaMode::None, targets);
  EXPECT_EQ(count(vertex_output(text), "mo_cur: vec3f,"), 1u);
  EXPECT_EQ(count(vertex_output(text), "mo_prev: vec3f,"), 1u);
  EXPECT_EQ(location_count(text), location_count(wgsl(config)) + MotionVaryings);
  EXPECT_EQ(varying_count(config, targets), varying_count(config, ShaderTargets{}) + MotionVaryings);
  EXPECT_NE(fragment_output(text).find("@location(2) motion: vec2f,"), std::string::npos);
  EXPECT_NE(text.find("fn gx_record_word(k: u32) -> u32"), std::string::npos);
  EXPECT_NE(text.find("let mo_flags = gx_record_word(1u);"), std::string::npos);
  EXPECT_NE(text.find("gx_abuf_mtx(mo_frame + 16u)"), std::string::npos);
  EXPECT_NE(text.find("gx_abuf_mtx(gx_record_word(4u) + 0u)"), std::string::npos);
  EXPECT_NE(text.find("(mo_flags & 4096u)"), std::string::npos);
  EXPECT_NE(text.find("(mo_flags & 4u)"), std::string::npos);
  EXPECT_NE(text.find("out.motion = select(vec2f(4.0), mo_d, mo_w > 1e-5 && all(abs(mo_d) < vec2f(4.0)));"),
            std::string::npos);
  EXPECT_LT(text.find("let mo_cur = out.pos;"), text.find("out.pos = vec4f(out.pos.xy + mo_jit"));
  EXPECT_LT(text.find("out.pos = vec4f(mv_pos, 1.0) * ubuf.proj;"), text.find("let mo_cur = out.pos;"));
  EXPECT_EQ(text.find("offset_ndc * mo_prev.w"), std::string::npos);
  EXPECT_EQ(text.find("reactive"), std::string::npos);
  const std::string ids = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.debugId = 5, .motion = 2});
  EXPECT_LT(fragment_output(ids).find("motion: vec2f"), fragment_output(ids).find("debug_id: u32"));
}

TEST_F(GxShaderExtTest, PaletteMotionIndexesPrevBlock) {
  const ShaderConfig config = with_palette(single_site());
  const std::string text = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.motion = 2});
  EXPECT_NE(text.find("gx_abuf_mtx(gx_record_word(4u) + min(in_pnmtxidx, 9u) * 12u)"), std::string::npos);
  EXPECT_EQ(text.find("gx_record_word(4u) + 0u"), std::string::npos);
  EXPECT_NE(text.find("mo_mv = vec4f(in_pos, 1.0) *"), std::string::npos);
  EXPECT_NE(text.find("mo_mv = vec4f(mv_pos, 1.0) *"), std::string::npos);
}

TEST_F(GxShaderExtTest, PointAndLineMotionAddsOffset) {
  constexpr std::string_view Offset = "mo_prev = vec4f(mo_prev.xy + offset_ndc * mo_prev.w, mo_prev.zw);";
  for (const u8 lineMode : {u8{1}, u8{2}, u8{3}}) {
    ShaderConfig config = single_site();
    config.lineMode = lineMode;
    const std::string text = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.motion = 2});
    EXPECT_EQ(count(text, Offset), 1u) << static_cast<int>(lineMode);
    EXPECT_LT(text.find("if (mo_source != 0u)"), text.find(Offset)) << static_cast<int>(lineMode);
    EXPECT_EQ(location_count(text), varying_count(config, ShaderTargets{.motion = 2})) << static_cast<int>(lineMode);
  }
  const std::string triangles = build_shader_source(single_site(), DstAlphaMode::None, ShaderTargets{.motion = 2});
  EXPECT_EQ(count(triangles, Offset), 0u);
}

TEST_F(GxShaderExtTest, ReactiveNeedsReactiveTarget) {
  const ShaderConfig config = single_site();
  EXPECT_EQ(build_shader_source(config, DstAlphaMode::None, ShaderTargets{.motion = 2}).find("reactive"),
            std::string::npos);
  const std::string text = build_shader_source(config, DstAlphaMode::Replace, ShaderTargets{.reactive = 3});
  EXPECT_NE(fragment_output(text).find("@location(3) reactive: vec4f,"), std::string::npos);
  EXPECT_NE(text.find("let reactive_flags = select(0u, load_word(&abuf, imm._pad + 1u), imm._pad != 0u);"),
            std::string::npos);
  EXPECT_NE(text.find("out.reactive = vec4f(select(0.0, saturate(prev.a), (reactive_flags & " +
                      std::to_string(render_ext::RecordFlagReactive) + "u) != 0u));"),
            std::string::npos);
  EXPECT_EQ(text.find("out.reactive = vec4f(saturate(prev.a));"), std::string::npos);
  EXPECT_EQ(text.find("mo_cur"), std::string::npos);
  EXPECT_EQ(location_count(text), location_count(wgsl(config)));
  const std::string both = build_shader_source(config, DstAlphaMode::None, ShaderTargets{.motion = 2, .reactive = 3});
  EXPECT_LT(fragment_output(both).find("motion: vec2f"), fragment_output(both).find("reactive: vec4f"));
  EXPECT_LT(both.find("out.motion ="), both.find("out.reactive ="));
}

TEST_F(GxShaderExtTest, ReactiveWriteClearsUnderOpaqueAndDepthWriters) {
  const auto rule = [](GXBlendMode mode, GXBlendFactor src, GXBlendFactor dst, bool depthCompare, bool depthUpdate,
                       bool colorUpdate) {
    PipelineConfig config{};
    config.blendMode = mode;
    config.blendFacSrc = src;
    config.blendFacDst = dst;
    config.blendOp = GX_LO_CLEAR;
    config.depthCompare = depthCompare;
    config.depthUpdate = depthUpdate;
    config.colorUpdate = colorUpdate;
    const ReactiveWrite fromConfig = reactive_write(config);
    EXPECT_EQ(fromConfig, reactive_write(config, {.colorUpdate = colorUpdate, .depthUpdate = depthUpdate}));
    return fromConfig;
  };
  EXPECT_EQ(rule(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, true, false, true), ReactiveWrite::Clear);
  EXPECT_EQ(rule(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, false, false, true), ReactiveWrite::Clear);
  EXPECT_EQ(rule(GX_BM_BLEND, GX_BL_ONE, GX_BL_ZERO, true, false, true), ReactiveWrite::Clear);
  EXPECT_EQ(rule(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, true, true, true), ReactiveWrite::Clear);
  EXPECT_EQ(rule(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, true, false, true), ReactiveWrite::Raise);
  EXPECT_EQ(rule(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, false, true, true), ReactiveWrite::Raise);
  EXPECT_EQ(rule(GX_BM_SUBTRACT, GX_BL_ONE, GX_BL_ONE, true, false, true), ReactiveWrite::Raise);
  EXPECT_EQ(rule(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, true, true, false), ReactiveWrite::None);
  EXPECT_EQ(rule(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, true, false, false), ReactiveWrite::None);
}
ShaderConfig lit_normal_config() {
  ShaderConfig config = single_site();
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.colorChannels[GX_COLOR0] = {.matSrc = GX_SRC_REG, .ambSrc = GX_SRC_REG, .lightingEnabled = true};
  config.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  config.tevStages[0].channelId = GX_COLOR0A0;
  return config;
}

ShaderConfig with_output(ShaderConfig config, u8 output) {
  config.ext.output = output;
  return config;
}

constexpr ShaderTargets GbufferTargets{.motion = 2, .reactive = 3, .gbufferNormal = 1};

TEST_F(GxShaderExtTest, OutputRowsAdmittedOnlyUnderGbufferLayouts) {
  const gfx::SceneLayoutBase base{
      .color = wgpu::TextureFormat::BGRA8Unorm,
      .depth = wgpu::TextureFormat::Depth32Float,
      .size = {64, 64, 1},
      .samples = 1,
  };
  const gfx::RenderTargetLayout gbuffer = gfx::compose_scene_layout({.gbuffer = true}, base);
  gfx::RenderTargetLayout offscreen = gbuffer;
  offscreen.colorAttachmentCount = 1;
  gfx::detail::finalize_render_target_layout(offscreen);
  const std::array without{gfx::compose_scene_layout({}, base), gfx::compose_scene_layout({.normal = true}, base),
                           gfx::compose_scene_layout({.normal = true, .debugId = true}, base), offscreen};
  const std::array with{gbuffer, gfx::compose_scene_layout({.gbuffer = true, .debugId = true}, base)};
  const std::array plain{RenderExtConfig{}, RenderExtConfig{.stochasticMask = 0b101}, RenderExtConfig{.pom = 1}};
  const std::array marked{RenderExtConfig{.output = 1}, RenderExtConfig{.output = 2}, RenderExtConfig{.velocity = 1},
                          RenderExtConfig{.fwdEnv = 1}, RenderExtConfig{.output = 1, .stochasticMask = 1}};
  for (const RenderExtConfig& ext : plain) {
    EXPECT_FALSE(render_ext::needs_gbuffer(ext));
    for (const auto& layout : without) {
      EXPECT_TRUE(render_ext::output_admitted(ext, layout));
    }
    for (const auto& layout : with) {
      EXPECT_TRUE(render_ext::output_admitted(ext, layout));
    }
  }
  for (const RenderExtConfig& ext : marked) {
    EXPECT_TRUE(render_ext::needs_gbuffer(ext));
    for (const auto& layout : without) {
      EXPECT_FALSE(render_ext::output_admitted(ext, layout)) << layout.colorAttachmentCount;
    }
    for (const auto& layout : with) {
      EXPECT_TRUE(render_ext::output_admitted(ext, layout)) << layout.colorAttachmentCount;
    }
  }
  const ShaderTargets targets = shader_targets(gbuffer);
  EXPECT_EQ(targets.normal, UINT32_MAX);
  EXPECT_EQ(targets.gbufferNormal, 1u);
  EXPECT_EQ(shader_targets(without[1]).gbufferNormal, UINT32_MAX);
  EXPECT_EQ(shader_targets(without[1]).normal, 1u);
}

TEST_F(GxShaderExtTest, GbufferNormalAlphaCarriesReceiverCode) {
  const std::string helper = receiver_alpha_wgsl();
  EXPECT_NE(helper.find("fn gbuffer_rx_alpha(rec: u32, nrm_len_sq: f32) -> f32"), std::string::npos);
  EXPECT_NE(helper.find("abuf[rec + 1u]"), std::string::npos);
  EXPECT_NE(helper.find(">> 6u) & 3u"), std::string::npos);
  EXPECT_NE(helper.find("rec != 0u"), std::string::npos);
  EXPECT_NE(helper.find("nrm_len_sq > 1e-10"), std::string::npos);

  const ShaderConfig lit = lit_normal_config();
  ShaderConfig unlit = single_site();
  unlit.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  for (const ShaderConfig& config : {lit, unlit}) {
    for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
      const std::string marker = build_shader_source(with_output(config, 1), mode, GbufferTargets);
      EXPECT_NE(fragment_output(marker).find("@location(1) normal: vec4f,"), std::string::npos);
      EXPECT_EQ(count(marker, "out.normal = vec4f(unit_nrm * 0.5 + 0.5, gbuffer_rx_alpha(imm._pad, nrm_len_sq));"), 1u);
      EXPECT_EQ(count(marker, helper), 1u);
      EXPECT_EQ(marker.find("select(0.0, 1.0, nrm_len_sq > 1e-10)"), std::string::npos);

      const std::string plain = build_shader_source(config, mode, GbufferTargets);
      EXPECT_EQ(fragment_output(plain).find("normal: vec4f"), std::string::npos);
      EXPECT_EQ(plain.find("gbuffer_rx_alpha"), std::string::npos);
      EXPECT_EQ(plain, build_shader_source(config, mode, ShaderTargets{.motion = 2, .reactive = 3}));

      for (const u8 output : {u8{1}, u8{2}}) {
        EXPECT_EQ(build_shader_source(with_output(config, output), mode), build_shader_source(config, mode));
        EXPECT_EQ(build_shader_source(with_output(config, output), mode, 1u), build_shader_source(config, mode, 1u));
        EXPECT_EQ(build_shader_source(with_output(config, output), mode, ShaderTargets{.normal = 1, .debugId = 2}),
                  build_shader_source(config, mode, ShaderTargets{.normal = 1, .debugId = 2}));
      }
    }
  }

  const std::string noNormal = build_shader_source(with_output(single_site(), 1), DstAlphaMode::None, GbufferTargets);
  EXPECT_NE(fragment_output(noNormal).find("@location(1) normal: vec4f,"), std::string::npos);
  EXPECT_NE(noNormal.find("out.normal = vec4f(0.5, 0.5, 0.5, 0.0);"), std::string::npos);
  EXPECT_EQ(noNormal.find("gbuffer_rx_alpha"), std::string::npos);
  EXPECT_EQ(noNormal.find("mv_nrm:"), std::string::npos);
}

TEST_F(GxShaderExtTest, GbufferLitDrawCarriesNormalVarying) {
  const ShaderConfig lit = lit_normal_config();
  ASSERT_TRUE(build_shader_info(lit).lightingEnabled);
  const ShaderConfig marker = with_output(lit, 1);
  const std::string text = build_shader_source(marker, DstAlphaMode::None, GbufferTargets);
  EXPECT_EQ(count(vertex_output(text), "mv_nrm: vec3f,"), 1u);
  EXPECT_EQ(count(text, "out.mv_nrm = mv_nrm;"), 1u);
  EXPECT_EQ(count(text, "let nrm_len_sq = dot(in.mv_nrm, in.mv_nrm);"), 1u);
  EXPECT_EQ(location_count(text), varying_count(marker, GbufferTargets));
  EXPECT_EQ(varying_count(marker, GbufferTargets), varying_count(lit, GbufferTargets) + 1);
  EXPECT_LE(location_count(text), static_cast<size_t>(build_shader_info(marker).varyings));

  const std::string plain = build_shader_source(lit, DstAlphaMode::None, GbufferTargets);
  EXPECT_EQ(vertex_output(plain).find("mv_nrm"), std::string::npos);
  EXPECT_EQ(location_count(plain), varying_count(lit, GbufferTargets));

  const ShaderTargets legacy{.normal = 1};
  EXPECT_EQ(varying_count(marker, legacy), varying_count(lit, legacy));
  EXPECT_EQ(location_count(build_shader_source(marker, DstAlphaMode::None, legacy)), varying_count(marker, legacy));
}

constexpr std::array<u8, 11> FogTypes{
    GX_FOG_NONE,         GX_FOG_PERSP_LIN,     GX_FOG_PERSP_EXP,     GX_FOG_PERSP_EXP2,
    GX_FOG_PERSP_REVEXP, GX_FOG_PERSP_REVEXP2, GX_FOG_ORTHO_LIN,     GX_FOG_ORTHO_EXP,
    GX_FOG_ORTHO_EXP2,   GX_FOG_ORTHO_REVEXP,  GX_FOG_ORTHO_REVEXP2,
};

ShaderConfig fogged(u8 type) {
  ShaderConfig config = single_site();
  config.fogType = type;
  return config;
}

TEST_F(GxShaderExtTest, FogRecordNeedsTargetAndOutput) {
  for (const u8 output : {u8{0}, u8{1}}) {
    const ShaderConfig config = with_output(fogged(GX_FOG_PERSP_LIN), output);
    for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
      EXPECT_EQ(build_shader_source(config, mode).find("fog: vec4u"), std::string::npos);
      EXPECT_EQ(build_shader_source(config, mode, 1u).find("fog: vec4u"), std::string::npos);
      EXPECT_EQ(build_shader_source(config, mode).find("out.fog"), std::string::npos);
    }
  }
  const ShaderConfig plain = fogged(GX_FOG_PERSP_LIN);
  EXPECT_EQ(build_shader_source(plain, DstAlphaMode::None, ShaderTargets{.normal = 1, .fog = 4}),
            build_shader_source(plain, DstAlphaMode::None, ShaderTargets{.normal = 1}));
  EXPECT_EQ(build_shader_source(plain, DstAlphaMode::None, GbufferTargets),
            build_shader_source(plain, DstAlphaMode::None,
                                ShaderTargets{.motion = 2, .reactive = 3, .gbufferNormal = 1, .fog = 4}));

  const ShaderConfig marker = with_output(plain, 1);
  const std::string text = build_shader_source(marker, DstAlphaMode::None, ShaderTargets{.normal = 1, .fog = 4});
  EXPECT_EQ(count(fragment_output(text), "@location(4) fog: vec4u,"), 1u);
  EXPECT_EQ(count(text, "out.fog ="), 1u);
  EXPECT_EQ(location_count(text), location_count(build_shader_source(marker, DstAlphaMode::None, 1u)));

  const std::string only = build_shader_source(marker, DstAlphaMode::None, ShaderTargets{.fog = 1});
  EXPECT_EQ(fragment_output(only),
            "struct FragmentOutput {\n    @location(0) color: vec4f,\n    @location(1) fog: vec4u,\n");

  const std::string ids =
      build_shader_source(marker, DstAlphaMode::None, ShaderTargets{.normal = 1, .debugId = 5, .fog = 4});
  EXPECT_LT(fragment_output(ids).find("normal: vec4f"), fragment_output(ids).find("fog: vec4u"));
  EXPECT_LT(fragment_output(ids).find("fog: vec4u"), fragment_output(ids).find("debug_id: u32"));
  EXPECT_LT(ids.find("out.fog ="), ids.rfind("return out;"));
}

TEST_F(GxShaderExtTest, FogRecordLockFollowsType) {
  constexpr std::string_view Mix = "prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fogZ, 0.0, 1.0)), prev.a);";
  constexpr std::string_view Record =
      "out.fog = vec4u(vec3u(round(ubuf.fog.color.rgb * 255.0)), "
      "(min(u32(round(clamp(fogZ, 0.0, 1.0) * 127.0)), 127u) << 1u) | ";
  for (const u8 type : FogTypes) {
    const std::string text =
        build_shader_source(with_output(fogged(type), 1), DstAlphaMode::None, ShaderTargets{.fog = 1});
    EXPECT_EQ(count(text, "out.fog ="), 1u) << int{type};
    if (type == GX_FOG_NONE) {
      EXPECT_NE(text.find("out.fog = vec4u(0u);"), std::string::npos);
      EXPECT_EQ(text.find(Record), std::string::npos);
      continue;
    }
    const std::string lock = type == GX_FOG_PERSP_LIN ? "0u);" : "1u);";
    EXPECT_NE(text.find(std::string{Record} + lock), std::string::npos) << int{type};
    ASSERT_NE(text.find(Mix), std::string::npos) << int{type};
    EXPECT_LT(text.find(Mix), text.find("out.fog =")) << int{type};
  }
}

TEST_F(GxShaderExtTest, FogRecordPredicate) {
  struct Blend {
    GXBlendMode mode;
    GXBlendFactor src;
    GXBlendFactor dst;
    GXLogicOp op;
    bool opaque;
  };
  constexpr std::array<Blend, 7> Blends{{
      {GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR, true},
      {GX_BM_BLEND, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR, true},
      {GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_CLEAR, false},
      {GX_BM_LOGIC, GX_BL_ONE, GX_BL_ZERO, GX_LO_COPY, true},
      {GX_BM_LOGIC, GX_BL_ONE, GX_BL_ZERO, GX_LO_CLEAR, false},
      {GX_BM_LOGIC, GX_BL_ONE, GX_BL_ZERO, GX_LO_NOOP, false},
      {GX_BM_SUBTRACT, GX_BL_ONE, GX_BL_ONE, GX_LO_CLEAR, false},
  }};
  for (const u8 output : {u8{0}, u8{1}, u8{2}}) {
    for (const bool colorUpdate : {false, true}) {
      for (const Blend& blend : Blends) {
        PipelineConfig config{};
        config.shaderConfig.ext.output = output;
        config.blendMode = blend.mode;
        config.blendFacSrc = blend.src;
        config.blendFacDst = blend.dst;
        config.blendOp = blend.op;
        config.colorUpdate = colorUpdate;
        const PipelineOptions options{.colorUpdate = colorUpdate};
        EXPECT_EQ(writes_fog_record(config, options), output != 0 && colorUpdate && blend.opaque)
            << int{output} << " " << colorUpdate << " " << int{blend.mode} << " " << int{blend.op};
      }
    }
  }

  const auto populated = [](GXPixelFmt format, GXBlendFactor src, GXBlendFactor dst) {
    g_gxState = {};
    g_gxState.pixelFmt = format;
    g_gxState.blendMode = GX_BM_BLEND;
    g_gxState.blendFacSrc = src;
    g_gxState.blendFacDst = dst;
    g_gxState.colorUpdate = true;
    PipelineConfig config{};
    populate_pipeline_config(config, GX_TRIANGLES, GX_VTXFMT0);
    config.shaderConfig.ext.output = 1;
    return writes_fog_record(config, PipelineOptions{.colorUpdate = config.colorUpdate});
  };
  EXPECT_TRUE(populated(GX_PF_RGB8_Z24, GX_BL_DSTALPHA, GX_BL_INVDSTALPHA));
  EXPECT_FALSE(populated(GX_PF_RGBA6_Z24, GX_BL_DSTALPHA, GX_BL_INVDSTALPHA));
  EXPECT_FALSE(populated(GX_PF_RGB8_Z24, GX_BL_INVDSTALPHA, GX_BL_DSTALPHA));
}
} // namespace
} // namespace aurora::gx

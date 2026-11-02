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
#include "gx/unlit_plan.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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

TEST_F(GxShaderExtTest, FallbackConfigClearsFwdEnv) {
  PipelineConfig config{};
  config.shaderConfig.ext = {.output = 1, .fwdEnv = 1};
  EXPECT_TRUE(render_ext::fallback_config(config).shaderConfig.ext == RenderExtConfig{});
  EXPECT_EQ(persisted_version(render_ext::fallback_config(config)), 14u);
  EXPECT_EQ(render_ext::legacy_config(config).shaderConfig.ext.fwdEnv, 1u);
  config.shaderConfig.ext = {.fwdEnv = 1};
  EXPECT_EQ(persisted_version(config), GXExtRowVersion);
  EXPECT_EQ(persisted_version(render_ext::fallback_config(config)), 14u);
}

TEST_F(GxShaderExtTest, NormalizeKeepsOnlyEligibleStochastic) {
  const ShaderInfo info = build_shader_info(single_site());
  ASSERT_EQ(info.stochasticEligible.to_ulong(), 1u);
  const RenderExtConfig ext{.output = 2, .stochasticMask = 0b111, .layer0 = 1};
  const RenderExtConfig normalized = render_ext::normalize(ext, info);
  EXPECT_EQ(normalized.stochasticMask, 1u);
  EXPECT_EQ(normalized.output, 2u);
  EXPECT_EQ(normalized.layer0, 0u);
  EXPECT_TRUE(render_ext::normalize(normalized, info) == normalized);
  ASSERT_TRUE(info.companionFits);
  EXPECT_EQ(render_ext::normalize({.output = 2, .pom = 1, .layer0 = 1, .layer1 = 2}, info),
            (RenderExtConfig{.output = 2, .layer0 = 1}));
  for (const RenderExtConfig other :
       {RenderExtConfig{.output = 1, .layer0 = 1}, RenderExtConfig{.layer0 = 1},
        RenderExtConfig{.output = 2, .layer0 = 2}, RenderExtConfig{.output = 2, .layer0 = 9}}) {
    EXPECT_EQ(render_ext::normalize(other, info).layer0, 0u) << int{other.output} << " " << int{other.layer0};
  }
  ShaderInfo indirect = info;
  indirect.sampledIndTextures.set(0);
  EXPECT_EQ(render_ext::normalize({.output = 2, .layer0 = 1}, indirect).layer0, 0u);
  ShaderInfo full = info;
  full.companionFits = false;
  const uint64_t drops = render_ext::g_stats.budgetDrops.load();
  EXPECT_EQ(render_ext::normalize({.output = 2, .layer0 = 1}, full).layer0, 0u);
  EXPECT_EQ(render_ext::g_stats.budgetDrops.load(), drops + 1);
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
  ShaderConfig output2 = legacy;
  output2.ext = {.output = 2, .velocity = 1, .layer0 = 2};
  EXPECT_EQ(wgsl(output2), expected);
  EXPECT_EQ(build_shader_info(output2).uniformSize, build_shader_info(legacy).uniformSize);
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
  ShaderConfig output2 = legacy;
  output2.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  ShaderConfig marked = output2;
  marked.ext.output = 2;
  ASSERT_TRUE(build_shader_info(marked).unlitPlanned);
  EXPECT_EQ(build_shader_info(marked).uniformSize, build_shader_info(output2).uniformSize);
  EXPECT_EQ(testing::uniform_bytes(build_shader_info(marked)), testing::uniform_bytes(build_shader_info(output2)));
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
      ShaderConfig unlit = config;
      unlit.ext.output = 2;
      const ShaderTargets pbr{.motion = 2, .reactive = 3, .gbufferNormal = 1, .fog = 4, .surface = 5};
      const u32 unlitPlanned = build_shader_info(unlit).varyings;
      EXPECT_LE(unlitPlanned, MaxGxVaryings) << entry.label << " output 2";
      const size_t unlitLocations = location_count(build_shader_source(unlit, mode, pbr));
      EXPECT_EQ(unlitLocations, varying_count(unlit, pbr)) << entry.label << " output 2";
      EXPECT_LE(unlitLocations, unlitPlanned) << entry.label << " output 2";
      maxLocations = std::max(maxLocations, unlitLocations);
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
  EXPECT_NE(text.find("out.reactive = vec4f(select(0.0, saturate(reactive_alpha), (reactive_flags & " +
                      std::to_string(render_ext::RecordFlagReactive) + "u) != 0u));"),
            std::string::npos);
  EXPECT_EQ(text.find("saturate(prev.a)"), std::string::npos);
  // Replace forces prev.a to 1, so coverage has to be read before it
  EXPECT_EQ(count(text, "let reactive_alpha = prev.a;"), 1u);
  EXPECT_LT(text.find("let reactive_alpha = prev.a;"), text.find("prev.a = 1.0;"));
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

TEST_F(GxShaderExtTest, ReactiveTargetStateFollowsReactiveWrite) {
  const gfx::SceneLayoutBase base{
      .color = wgpu::TextureFormat::BGRA8Unorm,
      .depth = wgpu::TextureFormat::Depth32Float,
      .size = {64, 64, 1},
      .samples = 1,
  };
  const gfx::RenderTargetLayout layout = gfx::compose_scene_layout({.gbuffer = true}, base);
  const uint32_t target = shader_targets(layout).reactive;
  ASSERT_NE(target, UINT32_MAX);
  PipelineConfig opaque{};
  opaque.blendMode = GX_BM_NONE;
  opaque.blendFacSrc = GX_BL_ONE;
  opaque.blendFacDst = GX_BL_ZERO;
  opaque.depthCompare = true;
  PipelineConfig translucent = opaque;
  translucent.blendMode = GX_BM_BLEND;
  translucent.blendFacSrc = GX_BL_SRCALPHA;
  translucent.blendFacDst = GX_BL_INVSRCALPHA;
  const auto reactive = [&](const PipelineConfig& config, const PipelineOptions& options) {
    return color_target_states(config, layout, options)[target];
  };

  const wgpu::ColorTargetState clear = reactive(opaque, {.colorUpdate = true, .depthUpdate = true});
  EXPECT_EQ(clear.format, layout.colorAttachments[target].format);
  EXPECT_EQ(clear.writeMask, wgpu::ColorWriteMask::All);
  EXPECT_EQ(clear.blend, nullptr);
  const wgpu::ColorTargetState raise = reactive(translucent, {.colorUpdate = true});
  EXPECT_EQ(raise.writeMask, wgpu::ColorWriteMask::All);
  ASSERT_NE(raise.blend, nullptr);
  EXPECT_EQ(raise.blend->color.operation, wgpu::BlendOperation::Max);
  EXPECT_EQ(raise.blend->alpha.operation, wgpu::BlendOperation::Max);
  const wgpu::ColorTargetState none = reactive(translucent, {.depthUpdate = true});
  EXPECT_EQ(none.writeMask, wgpu::ColorWriteMask::None);
  EXPECT_EQ(none.blend, nullptr);
  // The alpha prepass runs before the colour pass, so it must leave coverage to it
  const PipelineOptions prepass{.dstAlphaMode = DstAlphaMode::Replace, .alphaUpdate = true};
  for (const PipelineConfig& config : {opaque, translucent}) {
    const wgpu::ColorTargetState state = reactive(config, prepass);
    EXPECT_EQ(state.writeMask, wgpu::ColorWriteMask::None);
    EXPECT_EQ(state.blend, nullptr);
  }
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
  const std::array marked{RenderExtConfig{.output = 1}, RenderExtConfig{.velocity = 1}, RenderExtConfig{.fwdEnv = 1},
                          RenderExtConfig{.output = 1, .stochasticMask = 1}};
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
  for (const auto& layout : with) {
    EXPECT_TRUE(shader_targets(layout).fwdEnv);
  }
  for (const auto& layout : without) {
    EXPECT_FALSE(shader_targets(layout).fwdEnv) << layout.colorAttachmentCount;
  }
  const std::array surface{gfx::compose_scene_layout({.gbuffer = true, .surface = true}, base),
                           gfx::compose_scene_layout({.gbuffer = true, .surface = true, .debugId = true}, base)};
  for (const RenderExtConfig& ext : {RenderExtConfig{.output = 2}, RenderExtConfig{.output = 2, .layer0 = 1}}) {
    EXPECT_TRUE(render_ext::needs_gbuffer(ext));
    for (const auto& layout : without) {
      EXPECT_FALSE(render_ext::output_admitted(ext, layout)) << layout.colorAttachmentCount;
    }
    for (const auto& layout : with) {
      EXPECT_FALSE(render_ext::output_admitted(ext, layout)) << layout.colorAttachmentCount;
    }
    for (const auto& layout : surface) {
      EXPECT_TRUE(render_ext::output_admitted(ext, layout)) << layout.colorAttachmentCount;
    }
  }
  for (const auto& layout : surface) {
    EXPECT_TRUE(render_ext::output_admitted(RenderExtConfig{.output = 1}, layout));
    EXPECT_NE(shader_targets(layout).surface, UINT32_MAX);
  }
  for (const auto& layout : with) {
    EXPECT_EQ(shader_targets(layout).surface, UINT32_MAX);
  }
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

constexpr ShaderTargets FwdEnvTargets{.motion = 2, .reactive = 3, .gbufferNormal = 1, .fog = 4, .fwdEnv = true};
constexpr std::string_view FwdBinding = "@group(3) @binding(0)";
constexpr std::string_view FwdHelper = "fn fwd_screen_ao(";
constexpr std::string_view FwdLoad = "textureLoad(fwd_ao,";

ShaderConfig with_fwd_env(ShaderConfig config) {
  config.ext.fwdEnv = 1;
  return config;
}

// Clamped colour, wrapping alpha and a live alpha test, so the overflow and compare lines follow the block
ShaderConfig clamped_with_tail(u8 fogType) {
  ShaderConfig config = fogged(fogType);
  config.tevStages[0].alphaOp.clamp = false;
  config.alphaCompare = {.comp0 = GX_GREATER, .ref0 = 8, .op = GX_AOP_AND, .comp1 = GX_ALWAYS, .ref1 = 0};
  return config;
}

TEST_F(GxShaderExtTest, FwdEnvNeedsLayoutAndBit) {
  const ShaderConfig plain = single_site();
  for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
    EXPECT_EQ(build_shader_source(with_fwd_env(plain), mode), build_shader_source(plain, mode));
    EXPECT_EQ(build_shader_source(with_fwd_env(plain), mode, 1u), build_shader_source(plain, mode, 1u));
    ShaderTargets noEnv = FwdEnvTargets;
    noEnv.fwdEnv = false;
    EXPECT_EQ(build_shader_source(plain, mode, FwdEnvTargets), build_shader_source(plain, mode, noEnv));
    EXPECT_EQ(build_shader_source(with_fwd_env(plain), mode, noEnv), build_shader_source(plain, mode, noEnv));
  }

  for (const u8 fogType : {u8{GX_FOG_PERSP_LIN}, u8{GX_FOG_ORTHO_EXP}, u8{GX_FOG_NONE}}) {
    const ShaderConfig config = with_fwd_env(clamped_with_tail(fogType));
    ASSERT_TRUE(tev_colour_clamped(config));
    const std::string text = build_shader_source(config, DstAlphaMode::None, FwdEnvTargets);
    EXPECT_EQ(count(text, FwdBinding), 1u) << int{fogType};
    EXPECT_EQ(count(text, FwdHelper), 1u) << int{fogType};
    EXPECT_EQ(count(text, FwdLoad), 1u) << int{fogType};
    const size_t block = text.find(FwdLoad);
    const size_t body = text.find("fn fs_main(");
    ASSERT_NE(body, std::string::npos);
    EXPECT_LT(text.rfind("// TEV stage"), block) << int{fogType};
    if (fogType != GX_FOG_NONE) {
      const size_t mix = text.find("prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb");
      ASSERT_NE(mix, std::string::npos);
      EXPECT_LT(block, mix) << int{fogType};
    }
    const size_t overflow = text.find("prev.a = tev_overflow_f32(prev.a);", body);
    const size_t compare = text.find("// Alpha compare", body);
    ASSERT_NE(overflow, std::string::npos);
    ASSERT_NE(compare, std::string::npos);
    EXPECT_LT(block, overflow) << int{fogType};
    EXPECT_LT(block, compare) << int{fogType};
    EXPECT_NE(text.find("& 2048u) != 0u)"), std::string::npos);
    EXPECT_NE(text.find("smoothstep(0.004, 0.020, gap)"), std::string::npos);
    EXPECT_EQ(location_count(text),
              location_count(build_shader_source(clamped_with_tail(fogType), DstAlphaMode::None, FwdEnvTargets)));
    EXPECT_EQ(uniform_struct(text),
              uniform_struct(build_shader_source(clamped_with_tail(fogType), DstAlphaMode::None, FwdEnvTargets)));
  }

  ShaderConfig unclamped = with_fwd_env(fogged(GX_FOG_PERSP_LIN));
  unclamped.tevStages[0].colorOp.clamp = false;
  ASSERT_FALSE(tev_colour_clamped(unclamped));
  const std::string text = build_shader_source(unclamped, DstAlphaMode::None, FwdEnvTargets);
  EXPECT_EQ(count(text, FwdBinding), 1u);
  EXPECT_EQ(text.find(FwdHelper), std::string::npos);
  EXPECT_EQ(text.find(FwdLoad), std::string::npos);
}

TEST_F(GxShaderExtTest, FwdEnvPredicateMatchesText) {
  ShaderConfig unclamped = single_site();
  unclamped.tevStages[0].colorOp.clamp = false;
  const std::array configs{single_site(), fogged(GX_FOG_PERSP_EXP), unclamped, lit_normal_config()};
  ShaderTargets noEnv = FwdEnvTargets;
  noEnv.fwdEnv = false;
  const std::array targets{FwdEnvTargets, noEnv, ShaderTargets{.normal = 1}, ShaderTargets{}};
  for (const ShaderConfig& base : configs) {
    for (const bool bit : {false, true}) {
      const ShaderConfig config = bit ? with_fwd_env(base) : base;
      for (const ShaderTargets& target : targets) {
        const std::string text = build_shader_source(config, DstAlphaMode::None, target);
        EXPECT_EQ(text.find(FwdBinding) != std::string::npos, uses_forward_env(config, target));
        EXPECT_EQ(text.find(FwdLoad) != std::string::npos,
                  uses_forward_env(config, target) && tev_colour_clamped(config));
      }
    }
  }
}
constexpr ShaderTargets PbrTargets{.motion = 2, .reactive = 3, .gbufferNormal = 1, .fog = 4, .surface = 5};
constexpr ShaderTargets PbrIdTargets{
    .debugId = 6, .motion = 2, .reactive = 3, .gbufferNormal = 1, .fog = 4, .surface = 5};
constexpr std::string_view UnlitBlock =
    "// Unlit TEV (SP1 rules v1): no fog, lit alpha kept, only unlit_rgb leaves the block";
constexpr std::string_view SurfaceZero = "out.surface = vec4u(0u);";

ColorChannelConfig lit_channel(GXColorSrc matSrc, GXAttnFn attnFn = GX_AF_SPOT) {
  return {.matSrc = matSrc, .ambSrc = GX_SRC_REG, .diffFn = GX_DF_CLAMP, .attnFn = attnFn, .lightingEnabled = true};
}

ShaderConfig unlit_base(GXColorSrc matSrc = GX_SRC_REG) {
  ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
  config.attrs[GX_VA_TEX1] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0};
  config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1};
  config.colorChannels[GX_COLOR0] = lit_channel(matSrc);
  config.colorChannels[GX_ALPHA0] = {.matSrc = GX_SRC_VTX};
  return config;
}

TevStage unlit_tev(GXTevColorArg a, GXTevColorArg b, GXTevColorArg c, GXTevColorArg d, u32 map = 0, u32 coord = 0) {
  TevStage stage;
  stage.colorPass = {a, b, c, d};
  stage.texMapId = static_cast<GXTexMapID>(map);
  stage.texCoordId = map == GX_TEXMAP_NULL ? GX_TEXCOORD_NULL : static_cast<GXTexCoordID>(coord);
  stage.channelId = GX_COLOR0A0;
  return stage;
}

ShaderConfig staged(ShaderConfig config, std::initializer_list<TevStage> stages, u8 output = 2) {
  u32 n = 0;
  for (const TevStage& stage : stages) {
    config.tevStages[n++] = stage;
  }
  config.tevStageCount = n;
  config.ext.output = output;
  return config;
}

ShaderConfig modulate(u8 output = 2) {
  return staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO)}, output);
}

ShaderConfig ma00_like(u8 output = 2) {
  TevStage cloud = unlit_tev(GX_CC_ZERO, GX_CC_ONE, GX_CC_KONST, GX_CC_TEXC, 1, 1);
  cloud.kcSel = GX_TEV_KCSEL_K1_R;
  cloud.colorOp.outReg = GX_TEVREG2;
  cloud.channelId = GX_COLOR_ZERO;
  TevStage light = unlit_tev(GX_CC_KONST, GX_CC_C2, GX_CC_CPREV, GX_CC_C0, GX_TEXMAP_NULL);
  light.colorOp.scale = GX_CS_SCALE_4;
  light.kcSel = GX_TEV_KCSEL_K0;
  light.channelId = GX_COLOR_ZERO;
  TevStage base = unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO);
  base.alphaPass = {GX_CA_ZERO, GX_CA_RASA, GX_CA_TEXA, GX_CA_ZERO};
  ShaderConfig config = staged(unlit_base(GX_SRC_VTX), {base, cloud, light}, output);
  config.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_TEXMTX1};
  return config;
}

std::string fs_body(const std::string& text) {
  const size_t begin = text.find("fn fs_main(");
  return begin == std::string::npos ? std::string{} : text.substr(begin);
}

std::string unlit_block(const std::string& text) {
  const size_t begin = text.find(UnlitBlock);
  const size_t end = text.find("out.surface = gx_surface_pack(", begin);
  return begin == std::string::npos || end == std::string::npos ? std::string{} : text.substr(begin, end - begin);
}

TEST_F(GxShaderExtTest, UnlitNeedsSurfaceAndMotionTargets) {
  const ShaderConfig config = modulate();
  const ShaderInfo info = build_shader_info(config);
  ASSERT_TRUE(info.unlitPlanned);
  ASSERT_TRUE(info.unlitAdmitted);
  EXPECT_TRUE(uses_unlit(config, info, PbrTargets));
  EXPECT_TRUE(uses_unlit(config, info, PbrIdTargets));
  const std::string text = build_shader_source(config, DstAlphaMode::None, PbrTargets);
  EXPECT_EQ(count(text, UnlitBlock), 1u);
  EXPECT_NE(fragment_output(text).find("@location(5) surface: vec4u,"), std::string::npos);
  EXPECT_NE(text.find("fn gx_surface_pack("), std::string::npos);
  EXPECT_NE(text.find("fn gx_tev_pol("), std::string::npos);
  EXPECT_EQ(text.find(SurfaceZero), std::string::npos);

  ShaderTargets noMotion = PbrTargets;
  noMotion.motion = UINT32_MAX;
  EXPECT_FALSE(uses_unlit(config, info, noMotion));
  const std::string zero = build_shader_source(config, DstAlphaMode::None, noMotion);
  EXPECT_EQ(zero.find(UnlitBlock), std::string::npos);
  EXPECT_EQ(count(zero, SurfaceZero), 1u);
  EXPECT_EQ(zero.find("fn gx_tev_pol("), std::string::npos);

  for (const ShaderTargets& targets : {GbufferTargets, ShaderTargets{}, ShaderTargets{.normal = 1}}) {
    EXPECT_FALSE(uses_unlit(config, info, targets));
    const std::string plain = build_shader_source(config, DstAlphaMode::None, targets);
    EXPECT_EQ(plain.find("surface"), std::string::npos);
    EXPECT_EQ(plain, build_shader_source(modulate(1), DstAlphaMode::None, targets));
  }
  EXPECT_FALSE(uses_unlit(modulate(1), build_shader_info(modulate(1)), PbrTargets));
}

TEST_F(GxShaderExtTest, UnlitLitPathIsPrefix) {
  for (const ShaderConfig& config : {modulate(), ma00_like()}) {
    ShaderConfig twin = config;
    twin.ext.output = 1;
    for (const ShaderTargets& targets : {PbrTargets, PbrIdTargets}) {
      for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace}) {
        const std::string unlit = build_shader_source(config, mode, targets);
        const std::string lit = build_shader_source(twin, mode, targets);
        const std::string unlitBody = fs_body(unlit);
        const std::string litBody = fs_body(lit);
        const size_t split = litBody.find("\n    out.surface = vec4u(0u);");
        ASSERT_NE(split, std::string::npos);
        EXPECT_EQ(unlitBody.substr(0, split), litBody.substr(0, split));
        EXPECT_EQ(unlitBody.compare(split, UnlitBlock.size() + 5, "\n    " + std::string{UnlitBlock}), 0);
        EXPECT_EQ(fragment_output(unlit), fragment_output(lit));
        const std::string litOut = vertex_output(lit);
        const std::string unlitOut = vertex_output(unlit);
        EXPECT_EQ(unlitOut.substr(0, litOut.size()), litOut);
      }
    }
  }
  const std::string vertexColour = build_shader_source(ma00_like(), DstAlphaMode::None, PbrTargets);
  const std::string out = vertex_output(vertexColour);
  EXPECT_EQ(out.substr(out.rfind("@location(")), "@location(6) ccv0: vec3f,\n");
  EXPECT_EQ(count(vertexColour, "out.ccv0 = in_clr0.rgb;"), 1u);
}

TEST_F(GxShaderExtTest, UnlitDividesScaleStages) {
  const ShaderConfig config = ma00_like();
  ASSERT_EQ(unlit::plan(config).scaleDivide, 0b100u);
  const std::string text = build_shader_source(config, DstAlphaMode::None, PbrTargets);
  const std::string block = unlit_block(text);
  ASSERT_FALSE(block.empty());
  const size_t litStage = text.find("// TEV stage 2");
  ASSERT_NE(litStage, std::string::npos);
  EXPECT_NE(text.substr(litStage, text.find('\n', text.find('\n', litStage) + 1) - litStage).find(" * 4.0"),
            std::string::npos);
  const size_t unlitStage = block.find("// Unlit stage 2");
  ASSERT_NE(unlitStage, std::string::npos);
  EXPECT_EQ(block.find(" * 4.0"), std::string::npos);
  EXPECT_EQ(count(block, "// Unlit stage "), 3u);
  EXPECT_NE(block.find("unlit_rgb = prev.rgb;"), std::string::npos);
}

TEST_F(GxShaderExtTest, UnlitKonstUsesPolicyLocals) {
  const std::string text = build_shader_source(ma00_like(), DstAlphaMode::None, PbrTargets);
  const std::string block = unlit_block(text);
  ASSERT_FALSE(block.empty());
  EXPECT_NE(block.find("let upol = gx_record_word(7u);"), std::string::npos);
  EXPECT_NE(block.find("let upk0 = gx_tev_pol(ubuf.kcolor0, 3u, upol);"), std::string::npos);
  EXPECT_NE(block.find("let upk1 = gx_tev_pol(ubuf.kcolor1, 4u, upol);"), std::string::npos);
  EXPECT_NE(block.find("var tevreg0 = gx_tev_pol(ubuf.tevreg0, 0u, upol);"), std::string::npos);
  EXPECT_NE(block.find("var tevreg2: vec4f;"), std::string::npos);
  EXPECT_NE(block.find("vec3f(upk1.r)"), std::string::npos);
  EXPECT_NE(block.find("vec3f(upk0.rgb)"), std::string::npos);
  EXPECT_EQ(count(block, "ubuf.kcolor"), 2u);
  const std::string lit = text.substr(0, text.find(UnlitBlock));
  EXPECT_NE(lit.find("vec3f(ubuf.kcolor1.r)"), std::string::npos);
  EXPECT_EQ(lit.find("upk"), std::string::npos);
}

TEST_F(GxShaderExtTest, UnlitRastFollowsRules) {
  const std::string r1 = unlit_block(build_shader_source(modulate(), DstAlphaMode::None, PbrTargets));
  EXPECT_NE(r1.find("var rast0 = vec4f(ubuf.cc0_mat.rgb, in.cc0.a);"), std::string::npos);
  EXPECT_EQ(r1.find("uvc"), std::string::npos);

  const std::string r2 = unlit_block(build_shader_source(ma00_like(), DstAlphaMode::None, PbrTargets));
  EXPECT_NE(r2.find("let uvc = (gx_record_word(1u) >> 3u) & 3u;"), std::string::npos);
  EXPECT_NE(r2.find("var rast0 = vec4f(select(in.ccv0, vec3f(1.0), uvc != 0u), in.cc0.a);"), std::string::npos);

  ShaderConfig mixed =
      staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                            unlit_tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_ONE, GX_CC_CPREV, GX_TEXMAP_NULL)});
  mixed.tevStages[1].channelId = GX_COLOR1A1;
  mixed.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_REG, GX_AF_SPEC);
  ASSERT_EQ(unlit::plan(mixed).rewrite[1], unlit::RewriteZero);
  const std::string r4 = unlit_block(build_shader_source(mixed, DstAlphaMode::None, PbrTargets));
  EXPECT_NE(r4.find("var rast1 = vec4f(vec3f(0.0), in.cc1.a);"), std::string::npos);
  EXPECT_NE(r4.find("var rast0 = vec4f(ubuf.cc0_mat.rgb, in.cc0.a);"), std::string::npos);

  ShaderConfig neutral = staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                               unlit_tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, 1, 1)});
  neutral.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_NRM, .mtx = GX_TEXMTX0};
  ASSERT_EQ(unlit::plan(neutral).neutralOne, 0b10u);
  const std::string r7 = unlit_block(build_shader_source(neutral, DstAlphaMode::None, PbrTargets));
  EXPECT_NE(r7.find("let sampled1 = vec4f(1.0);"), std::string::npos);
  EXPECT_EQ(r7.find("let sampled0 ="), std::string::npos);

  ShaderConfig swapped = staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO)});
  swapped.tevSwapTable[1] = {GX_CH_RED, GX_CH_GREEN, GX_CH_BLUE, GX_CH_RED};
  swapped.tevStages[0].tevSwapRas = GX_TEV_SWAP1;
  ASSERT_TRUE(unlit::plan(swapped).pass());
  ASSERT_EQ(unlit::plan(swapped).rewrite[0], unlit::RewriteMaterial);
  const std::string text = build_shader_source(swapped, DstAlphaMode::None, PbrTargets);
  const std::string swap = unlit_block(text);
  EXPECT_NE(swap.find("var rast0 = vec4f(ubuf.cc0_mat.rgb, in.cc0.a);"), std::string::npos);
  EXPECT_NE(swap.find("vec3f(rast0.r)"), std::string::npos);
}

TEST_F(GxShaderExtTest, SurfaceWrittenAsLayoutProperty) {
  ShaderConfig failing = staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASA, GX_CC_ZERO)});
  failing.colorChannels[GX_ALPHA0] = lit_channel(GX_SRC_REG);
  ASSERT_FALSE(unlit::plan(failing).pass());
  ShaderConfig noNormal = modulate();
  noNormal.attrs[GX_VA_NRM] = {};
  ASSERT_FALSE(build_shader_info(noNormal).unlitAdmitted);
  for (const ShaderConfig& config : {modulate(0), modulate(1), failing, noNormal}) {
    for (const ShaderTargets& targets : {PbrTargets, PbrIdTargets}) {
      const std::string text = build_shader_source(config, DstAlphaMode::None, targets);
      EXPECT_EQ(count(text, SurfaceZero), 1u) << int{config.ext.output};
      EXPECT_NE(fragment_output(text).find("@location(5) surface: vec4u,"), std::string::npos);
      EXPECT_EQ(text.find(UnlitBlock), std::string::npos);
      EXPECT_EQ(text.find("ccv"), std::string::npos);
    }
  }
  const std::string ids = build_shader_source(modulate(), DstAlphaMode::None, PbrIdTargets);
  EXPECT_LT(ids.find("out.surface = gx_surface_pack("), ids.find("out.debug_id = imm._pad;"));
  for (const ShaderConfig& config : {modulate(0), modulate(1), modulate(), ma00_like(), failing}) {
    for (const ShaderTargets& targets : {GbufferTargets, ShaderTargets{}, ShaderTargets{.normal = 1, .debugId = 2}}) {
      const std::string text = build_shader_source(config, DstAlphaMode::None, targets);
      EXPECT_EQ(text.find("surface"), std::string::npos);
      ShaderConfig marker = config;
      marker.ext.output = std::min<u8>(marker.ext.output, 1);
      EXPECT_EQ(text, build_shader_source(marker, DstAlphaMode::None, targets));
    }
  }
}

TEST_F(GxShaderExtTest, SurfaceHashOnlyWhenTargetPresent) {
  ShaderTargets base = PbrTargets;
  base.surface = UINT32_MAX;
  for (const ShaderConfig& config : {modulate(0), modulate(1), modulate()}) {
    EXPECT_NE(testing::shader_hash(config, DstAlphaMode::None, PbrTargets),
              testing::shader_hash(config, DstAlphaMode::None, base));
    EXPECT_EQ(testing::shader_hash(config, DstAlphaMode::None, GbufferTargets),
              testing::shader_hash(config, DstAlphaMode::None,
                                   ShaderTargets{.motion = 2, .reactive = 3, .gbufferNormal = 1}));
  }
  EXPECT_NE(testing::shader_hash(modulate(1), DstAlphaMode::None, PbrTargets),
            testing::shader_hash(modulate(), DstAlphaMode::None, PbrTargets));
}

ShaderConfig crowded(u32 texgens) {
  ShaderConfig config = unlit_base(GX_SRC_VTX);
  config.attrs[GX_VA_CLR1] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  config.colorChannels[GX_COLOR1] = lit_channel(GX_SRC_VTX);
  config.colorChannels[GX_ALPHA1] = {.matSrc = GX_SRC_VTX};
  for (u32 i = 0; i < texgens; ++i) {
    config.attrs[GX_VA_TEX0 + i] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
    config.tcgs[i] = {.type = GX_TG_MTX3x4, .src = static_cast<GXTexGenSrc>(GX_TG_TEX0 + i)};
    config.tevStages[i] =
        unlit_tev(GX_CC_ZERO, i == 0 ? GX_CC_TEXC : GX_CC_CPREV, i == 0 ? GX_CC_RASC : GX_CC_TEXC, GX_CC_ZERO, i, i);
  }
  config.tevStages[texgens] = unlit_tev(GX_CC_ZERO, GX_CC_RASC, GX_CC_ONE, GX_CC_CPREV, GX_TEXMAP_NULL);
  config.tevStages[texgens].channelId = GX_COLOR1A1;
  config.tevStageCount = texgens + 1;
  config.ext.output = 2;
  return config;
}

TEST_F(GxShaderExtTest, UnlitVaryingDropRule) {
  const ShaderConfig full = crowded(MaxTexCoord);
  ASSERT_EQ(unlit::plan(full).rawVtx, 0b11u);
  const ShaderInfo fullInfo = build_shader_info(full);
  EXPECT_EQ(fullInfo.varyings, 13u);
  EXPECT_FALSE(fullInfo.unlitAdmitted);
  EXPECT_FALSE(unlit_facts(full, fullInfo).admitted);
  const std::string fullText = build_shader_source(full, DstAlphaMode::None, PbrTargets);
  EXPECT_EQ(count(fullText, SurfaceZero), 1u);
  EXPECT_EQ(location_count(fullText), varying_count(full, PbrTargets));

  const ShaderConfig roomy = crowded(MaxTexCoord - 1);
  const ShaderInfo roomyInfo = build_shader_info(roomy);
  EXPECT_TRUE(roomyInfo.unlitAdmitted);
  EXPECT_EQ(roomyInfo.varyings, MaxGxVaryings);
  const std::string roomyText = build_shader_source(roomy, DstAlphaMode::None, PbrTargets);
  EXPECT_EQ(location_count(roomyText), MaxGxVaryings);
  EXPECT_EQ(varying_count(roomy, PbrTargets), MaxGxVaryings);
  EXPECT_EQ(count(vertex_output(roomyText), "ccv"), 2u);
}

TEST_F(GxShaderExtTest, UnlitFactsMatchAcrossOutputs) {
  for (const ShaderConfig& base : {modulate(), ma00_like(), crowded(MaxTexCoord), crowded(MaxTexCoord - 1)}) {
    const ShaderInfo planned = build_shader_info(base);
    ASSERT_TRUE(planned.unlitPlanned);
    const UnlitFacts expected = unlit_facts(base, planned);
    for (const u8 output : {u8{0}, u8{1}}) {
      ShaderConfig other = base;
      other.ext.output = output;
      const ShaderInfo info = build_shader_info(other);
      EXPECT_FALSE(info.unlitPlanned);
      const UnlitFacts facts = unlit_facts(other, info);
      EXPECT_EQ(facts.admitted, expected.admitted);
      EXPECT_EQ(facts.rawVtx, expected.rawVtx);
      EXPECT_EQ(facts.layerStages, expected.layerStages);
      EXPECT_EQ(facts.verdict, expected.verdict);
    }
  }
}

TEST_F(GxShaderExtTest, NormalizeDemotesOutput2WithFwdEnv) {
  const uint64_t before = render_ext::g_stats.fwdEnvDemotions.load();
  const ShaderInfo info{};
  EXPECT_EQ(render_ext::normalize({.output = 2, .fwdEnv = 1}, info).output, 1u);
  EXPECT_EQ(render_ext::g_stats.fwdEnvDemotions.load(), before + 1);
  EXPECT_EQ(render_ext::normalize({.output = 2}, info).output, 2u);
  EXPECT_EQ(render_ext::normalize({.output = 1, .fwdEnv = 1}, info).output, 1u);
  EXPECT_EQ(render_ext::g_stats.fwdEnvDemotions.load(), before + 1);
  PipelineConfig config{};
  config.shaderConfig.ext = {.output = 2, .stochasticMask = 3, .layer0 = 1};
  const PipelineConfig twin = render_ext::output1_config(config);
  EXPECT_TRUE(twin.shaderConfig.ext == (RenderExtConfig{.output = 1, .stochasticMask = 3}));
}

ShaderConfig layered(ShaderConfig config, u8 layer0 = 1) {
  config.ext.layer0 = layer0;
  return config;
}

std::string companion_text(const ShaderConfig& config, const ShaderTargets& targets = PbrTargets) {
  return build_shader_source(config, DstAlphaMode::None, targets);
}

TEST_F(GxShaderExtTest, CompanionTextNeedsSurfaceAndLayer) {
  const ShaderConfig config = layered(modulate());
  const ShaderInfo info = build_shader_info(config);
  ASSERT_TRUE(info.unlitAdmitted);
  EXPECT_TRUE(info.companionLane);
  for (const ShaderTargets& targets : {PbrTargets, PbrIdTargets}) {
    EXPECT_TRUE(uses_companions(config, info, targets));
    const std::string text = companion_text(config, targets);
    EXPECT_EQ(count(text, "@group(2) @binding(16)\nvar ctex_alb: texture_2d<f32>;"), 1u);
    EXPECT_EQ(count(text, "@group(2) @binding(17)\nvar ctex_nrm: texture_2d<f32>;"), 1u);
    EXPECT_EQ(count(text, "@group(2) @binding(18)\nvar ctex_orm: texture_2d<f32>;"), 1u);
    EXPECT_EQ(count(text, "fn gx_perturb("), 1u);
    EXPECT_EQ(count(text, "fn gx_linear_to_srgb("), 1u);
    EXPECT_EQ(count(text, "let cmp_p = gx_view_pos(in.mo_cur);"), 1u);
    EXPECT_NE(text.find("textureSampleBias(ctex_nrm, tex0_samp, tex0_uv, ubuf.tex0_size_bias.z)"), std::string::npos);
  }
  const std::array<ShaderTargets, 3> without{GbufferTargets, ShaderTargets{}, ShaderTargets{.normal = 1}};
  for (const ShaderTargets& targets : without) {
    EXPECT_FALSE(uses_companions(config, info, targets));
    EXPECT_EQ(companion_text(config, targets).find("ctex_"), std::string::npos);
  }
  for (const ShaderConfig& other : {modulate(), layered(modulate(1)), layered(modulate(0))}) {
    const ShaderInfo otherInfo = build_shader_info(other);
    EXPECT_FALSE(uses_companions(other, otherInfo, PbrTargets));
    EXPECT_FALSE(otherInfo.companionLane);
    EXPECT_EQ(companion_text(other).find("ctex_"), std::string::npos);
  }
  ShaderConfig noNormal = layered(modulate());
  noNormal.attrs[GX_VA_NRM] = {};
  EXPECT_FALSE(uses_companions(noNormal, build_shader_info(noNormal), PbrTargets));
  ShaderTargets noMotion = PbrTargets;
  noMotion.motion = UINT32_MAX;
  EXPECT_FALSE(uses_companions(config, info, noMotion));
}

TEST_F(GxShaderExtTest, CompanionsAddNoVaryings) {
  for (const ShaderConfig& base : {modulate(), ma00_like(), crowded(MaxTexCoord - 1)}) {
    const ShaderConfig config = layered(base);
    ASSERT_TRUE(build_shader_info(config).unlitAdmitted);
    EXPECT_EQ(build_shader_info(config).varyings, build_shader_info(base).varyings);
    for (const ShaderTargets& targets : {PbrTargets, PbrIdTargets}) {
      const std::string text = companion_text(config, targets);
      ASSERT_NE(text.find("ctex_alb"), std::string::npos);
      EXPECT_EQ(vertex_output(text), vertex_output(companion_text(base, targets)));
      EXPECT_EQ(location_count(text), varying_count(config, targets));
      EXPECT_EQ(varying_count(config, targets), varying_count(base, targets));
    }
  }
}

TEST_F(GxShaderExtTest, CompanionNormalSelectKeepsLegacyExpression) {
  const ShaderConfig base = modulate();
  const std::string plain = companion_text(base);
  const std::string text = companion_text(layered(base));
  constexpr std::string_view Alpha = "gbuffer_rx_alpha(imm._pad, nrm_len_sq)";
  const std::string legacy = fmt::format("out.normal = vec4f(unit_nrm * 0.5 + 0.5, {});", Alpha);
  EXPECT_EQ(count(plain, legacy), 1u);
  EXPECT_EQ(count(text, legacy), 0u);
  EXPECT_EQ(count(text, fmt::format("let legacy_normal = vec4f(unit_nrm * 0.5 + 0.5, {});", Alpha)), 1u);
  EXPECT_EQ(count(text, fmt::format("out.normal = select(legacy_normal, vec4f(mapped_nrm * 0.5 + 0.5, {}), "
                                    "use_mapped);",
                                    Alpha)),
            1u);
  EXPECT_NE(text.find("ubuf.mat0.y > 0.0 && (u32(ubuf.mat0.x) & 6u) != 0u && ubuf.proj[3].w == 0.0"),
            std::string::npos);
  const std::string lets =
      "\n    let nrm_len_sq = dot(in.mv_nrm, in.mv_nrm);"
      "\n    let unit_nrm = select(vec3f(0.0), normalize(in.mv_nrm), nrm_len_sq > 1e-10);";
  EXPECT_EQ(count(plain, lets), 1u);
  EXPECT_EQ(count(text, lets), 1u);
  const std::string body = fs_body(text);
  const std::string plainBody = fs_body(plain);
  const size_t taps = body.find("\n    let cmp_alb = ");
  ASSERT_NE(taps, std::string::npos);
  EXPECT_EQ(body.substr(0, taps), plainBody.substr(0, taps));
  EXPECT_LT(taps, body.find("discard"));
  EXPECT_EQ(fragment_output(text), fragment_output(plain));
}

TEST_F(GxShaderExtTest, CompanionLaneIsLastUniform) {
  ShaderConfig config = layered(modulate());
  const ShaderInfo info = build_shader_info(config);
  const ShaderInfo plain = build_shader_info(modulate());
  ASSERT_TRUE(info.companionLane);
  const std::string uniforms = uniform_struct(companion_text(config));
  ASSERT_FALSE(uniforms.empty());
  EXPECT_EQ(uniforms.substr(uniforms.rfind("\n    ")), "\n    mat0: vec4f,\n");
  render_ext::g_companionLane = {.v = {3.f, 0.5f, -1.f, 0.f}};
  const std::vector<uint8_t> bytes = testing::uniform_bytes(info);
  const std::vector<uint8_t> plainBytes = testing::uniform_bytes(plain);
  ASSERT_EQ(bytes.size(), plainBytes.size() + 16);
  EXPECT_TRUE(std::equal(plainBytes.begin(), plainBytes.end(), bytes.begin()));
  const Vec4<float> lane = read_vec4(bytes, plainBytes.size());
  EXPECT_EQ(lane, render_ext::g_companionLane.v);
  EXPECT_EQ(info.uniformSize, gfx::align_uniform(static_cast<uint32_t>(bytes.size())));
  render_ext::g_companionLane = {};
  ShaderConfig stochastic =
      layered(staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO),
                                    unlit_tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, 1, 1)}));
  stochastic.ext.stochasticMask = 0b10;
  const ShaderInfo both = build_shader_info(stochastic);
  ASSERT_TRUE(both.companionLane);
  ASSERT_TRUE(both.stochasticTextures.test(1));
  const std::string bothUniforms = uniform_struct(companion_text(stochastic));
  EXPECT_LT(bothUniforms.find("tex1_hex: vec4f,"), bothUniforms.find("mat0: vec4f,"));
  EXPECT_EQ(bothUniforms.substr(bothUniforms.rfind("\n    ")), "\n    mat0: vec4f,\n");
}

TEST_F(GxShaderExtTest, AlbedoSubstitutesOnlyLayerPair) {
  ShaderConfig config = staged(unlit_base(), {unlit_tev(GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO, 0, 0),
                                              unlit_tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, 0, 1),
                                              unlit_tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, 1, 0),
                                              unlit_tev(GX_CC_ZERO, GX_CC_CPREV, GX_CC_TEXC, GX_CC_ZERO, 0, 0)});
  config = layered(config);
  ASSERT_TRUE(unlit::plan(config).pass());
  ASSERT_TRUE(build_shader_info(config).unlitAdmitted);
  const std::string text = companion_text(config);
  const std::string block = unlit_block(text);
  ASSERT_FALSE(block.empty());
  const auto substituted = [&](u32 stage) {
    return block.find(fmt::format("let sampled{0} = select(sampled{0}, vec4f(gx_linear_to_srgb(cmp_alb.rgb), "
                                  "sampled{0}.a), (u32(ubuf.mat0.x) & 1u) != 0u);",
                                  stage)) != std::string::npos;
  };
  EXPECT_TRUE(substituted(0));
  EXPECT_FALSE(substituted(1));
  EXPECT_FALSE(substituted(2));
  EXPECT_TRUE(substituted(3));
  const std::string lit = text.substr(0, text.find(UnlitBlock));
  EXPECT_EQ(lit.find("cmp_alb.rgb"), std::string::npos);
  EXPECT_NE(text.find("out.surface = gx_surface_pack(unlit_rgb, select(1.0, cmp_orm.r, uorm), select(0.85, cmp_orm.g, "
                      "uorm), select(0.0, cmp_orm.b, uorm), "),
            std::string::npos);
  const std::string plain = companion_text(modulate());
  EXPECT_NE(plain.find("out.surface = gx_surface_pack(unlit_rgb, 1.0, 0.85, 0.0, "), std::string::npos);
}

TEST_F(GxShaderExtTest, CompanionPredicateDrivesLayoutAndKind) {
  const ShaderConfig config = layered(modulate());
  const ShaderInfo info = build_shader_info(config);
  EXPECT_TRUE(uses_companions(config, info, PbrTargets));
  EXPECT_FALSE(uses_forward_env(config, PbrTargets));
  RenderExtConfig fwd = config.ext;
  fwd.fwdEnv = 1;
  const RenderExtConfig normalized = render_ext::normalize(fwd, info);
  EXPECT_EQ(normalized.output, 1u);
  EXPECT_EQ(normalized.layer0, 0u);
  PipelineConfig pipeline{};
  pipeline.shaderConfig = config;
  const PipelineConfig twin = render_ext::output1_config(pipeline);
  EXPECT_EQ(twin.shaderConfig.ext.layer0, 0u);
  EXPECT_FALSE(uses_companions(twin.shaderConfig, build_shader_info(twin.shaderConfig), PbrTargets));
  const PipelineConfig fallback = render_ext::fallback_config(pipeline);
  EXPECT_EQ(fallback.shaderConfig.ext.layer0, 0u);
  EXPECT_FALSE(uses_companions(fallback.shaderConfig, build_shader_info(fallback.shaderConfig), PbrTargets));
}
} // namespace
} // namespace aurora::gx

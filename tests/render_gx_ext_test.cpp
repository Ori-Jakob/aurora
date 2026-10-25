#include <gtest/gtest.h>

#include "gx/gx.hpp"
#include "gx/render_ext.hpp"
#include "gx/texture.hpp"
#include "render/frame_state.hpp"
#include "render/gx_ext/ext_provider.hpp"
#include "render/material/registry.hpp"

#include <SDL3/SDL_stdinc.h>
#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <array>
#include <memory>
#include <string_view>

namespace aurora::render::gx_ext {
namespace {
constexpr std::string_view FlaggedKey = "tex1_16x16_000000000000a001_14";
constexpr std::string_view PlainKey = "tex1_16x16_000000000000a004_14";
constexpr std::string_view Flagged =
    R"({"version":1,"materials":{"tex1_16x16_000000000000a001_14":{"flags":["stochastic"]}}})";
constexpr std::string_view Plain =
    R"({"version":1,"materials":{"tex1_16x16_000000000000a004_14":{"class":"ground/grass"}}})";

detail::FrameState frame(bool stochasticTiling, bool textureMaterials) {
  detail::FrameState state;
  state.settings.stochasticTiling = stochasticTiling;
  state.settings.textureMaterials = textureMaterials;
  return state;
}

gx::ShaderInfo eligible(u8 mask) {
  gx::ShaderInfo info;
  info.stochasticEligible = mask;
  return info;
}

gfx::TextureBind repeat_bind() {
  gfx::TextureBind bind;
  bind.ref =
      std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                        wgpu::Extent3D{8, 8, 1}, wgpu::TextureFormat::RGBA8Unorm, 1, GX_TF_RGBA8);
  bind.texObj.mode0 = GX_REPEAT | GX_REPEAT << 2;
  return bind;
}

gfx::TextureMaterialRef material_ref(std::string_view key) {
  const auto snapshot = material::latched_snapshot();
  const uint16_t id = snapshot->find(*texture::parse_material_key(key));
  return {id, snapshot->record(id).summary};
}

class RenderGxExt : public ::testing::Test {
protected:
  void SetUp() override { reset_all(); }
  void TearDown() override { reset_all(); }

  static void reset_all() {
    testing::reset();
    texture::clear_material_manifests();
    material::latch();
    gx::g_gxState = {};
  }

  static void add(std::string_view text) {
    ASSERT_EQ(texture::register_material_manifest(text, {.layer = texture::MaterialLayer::User}).status,
              texture::MaterialManifestStatus::Accepted);
    material::latch();
  }

  static gx::RenderExtConfig derive(const gx::ShaderInfo& info, gx::render_ext::StochasticLanes& lanes) {
    const gx::render_ext::Hooks* hooks = gx::render_ext::hooks();
    EXPECT_NE(hooks, nullptr);
    return hooks != nullptr ? hooks->derive(hooks->userData, gx::PipelineConfig{}, info, 0, lanes)
                            : gx::RenderExtConfig{};
  }
};

TEST_F(RenderGxExt, GateNeedsSettingsAndFlaggedMaterial) {
  add(Plain);
  apply_frame_state(frame(true, true));
  EXPECT_FALSE(testing::active());
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);

  add(Flagged);
  apply_frame_state(frame(false, true));
  EXPECT_FALSE(testing::active());
  apply_frame_state(frame(true, false));
  EXPECT_FALSE(testing::active());
  apply_frame_state(frame(true, true));
  EXPECT_TRUE(testing::active());
  EXPECT_NE(gx::render_ext::hooks(), nullptr);
  apply_frame_state(frame(false, true));
  EXPECT_FALSE(testing::active());
  EXPECT_EQ(gx::render_ext::hooks(), nullptr);
}

TEST_F(RenderGxExt, RegistryChangeWhileActiveInvalidates) {
  add(Flagged);
  apply_frame_state(frame(true, true));
  ASSERT_TRUE(testing::active());
  const uint64_t generation = gx::texture::current_bind_generation();
  apply_frame_state(frame(true, true));
  EXPECT_EQ(gx::texture::current_bind_generation(), generation);

  add(Plain);
  apply_frame_state(frame(true, true));
  EXPECT_TRUE(testing::active());
  EXPECT_GT(gx::texture::current_bind_generation(), generation);
}

TEST_F(RenderGxExt, ForceFlagsEveryEligibleRepeatSlot) {
  testing::set_force(true);
  apply_frame_state(frame(false, false));
  ASSERT_TRUE(testing::active());
  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(0b101), lanes).stochasticMask, 0b101u);
  EXPECT_TRUE(lanes[0] == gx::render_ext::pack({}));
  EXPECT_TRUE(lanes[2] == gx::render_ext::pack({}));

  gfx::TextureBind bind = repeat_bind();
  EXPECT_TRUE(gx::render_ext::stochastic_bind_ok(bind));
  bind.texObj.mode0 = GX_CLAMP | GX_REPEAT << 2;
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(bind));
  bind.texObj.mode0 = GX_REPEAT | GX_MIRROR << 2;
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(bind));
  EXPECT_FALSE(gx::render_ext::stochastic_bind_ok(gfx::TextureBind{}));

  testing::set_force(false);
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(testing::active());
}

TEST_F(RenderGxExt, EnvOnOpensGateButNeedsFlags) {
  SDL_SetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_STOCHASTIC", "on", true);
  testing::reset();
  apply_frame_state(frame(false, false));
  EXPECT_FALSE(testing::active());
  add(Flagged);
  apply_frame_state(frame(false, false));
  EXPECT_TRUE(testing::active());
  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(1), lanes).stochasticMask, 0u);
  gx::g_gxState.textures[0].material = material_ref(FlaggedKey);
  EXPECT_EQ(derive(eligible(1), lanes).stochasticMask, 1u);
  SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), "AURORA_RENDER_STOCHASTIC");
}

TEST_F(RenderGxExt, DeriveUsesManifestParamsAndDefaults) {
  add(R"({"version":1,"materials":{)"
      R"("tex1_16x16_000000000000a001_14":{"flags":["stochastic"],"stochastic":{"scale":4,"rotation":0.5}},)"
      R"("tex1_16x16_000000000000a002_14":{"flags":["stochastic"]},)"
      R"("tex1_16x16_000000000000a003_14":{"flags":["stochastic","cutout"]},)"
      R"("tex1_16x16_000000000000a004_14":{"class":"ground/grass"}}})");
  apply_frame_state(frame(true, true));
  ASSERT_TRUE(testing::active());
  const gfx::TextureMaterialRef params = material_ref(FlaggedKey);
  const gfx::TextureMaterialRef defaults = material_ref("tex1_16x16_000000000000a002_14");
  ASSERT_NE(params.id, 0u);
  ASSERT_NE(defaults.id, 0u);
  auto& textures = gx::g_gxState.textures;
  textures[0].material = params;
  textures[1].material = defaults;
  textures[2].material = material_ref("tex1_16x16_000000000000a003_14");
  textures[3].material = material_ref(PlainKey);
  textures[4].material = {params.id, 0};
  textures[5].material = {0, material::Stochastic};
  textures[6].material = {60000, material::Stochastic};
  textures[7].material = defaults;

  gx::render_ext::StochasticLanes lanes{};
  EXPECT_EQ(derive(eligible(0xFF), lanes).stochasticMask, 0b10000011u);
  EXPECT_TRUE(lanes[0] == gx::render_ext::pack({.scale = 4.f, .rotation = 0.5f}));
  EXPECT_TRUE(lanes[1] == gx::render_ext::pack({}));
  EXPECT_TRUE(lanes[7] == gx::render_ext::pack({}));
  EXPECT_EQ(derive(eligible(0b10), lanes).stochasticMask, 0b10u);
}
} // namespace
} // namespace aurora::render::gx_ext

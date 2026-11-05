#include <gtest/gtest.h>

#include "gfx/texture.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "render/gx_ext/unlit.hpp"
#include "render/scene/atlas.hpp"
#include "render/scene/classifier.hpp"
#include "render/scene/receiver_category.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace aurora::render::scene::atlas {
namespace {
namespace flag = material_flag;

const uint8_t g_texels[4096]{};

gfx::TextureHandle texture(uint32_t size = 256, wgpu::TextureFormat format = wgpu::TextureFormat::RGBA8Unorm,
                           uint32_t mips = 1) {
  return std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                           wgpu::Extent3D{size, size, 1}, format, mips, GX_TF_RGBA8);
}

GXTexObj_ texture_object(uint32_t id, uint32_t format = GX_TF_RGBA8) {
  GXTexObj_ obj{};
  obj.data = &g_texels[id % sizeof(g_texels)];
  obj.mWidth = 256;
  obj.mHeight = 256;
  obj.mFormat = format;
  obj.texObjId = id;
  obj.mode0 = static_cast<uint32_t>(GX_REPEAT) | static_cast<uint32_t>(GX_MIRROR) << 2;
  return obj;
}

void bind_texture(GXTexMapID map, const GXTexObj_& obj, gfx::TextureHandle handle, uint32_t registry = 0) {
  gx::g_gxState.loadedTextures[map] = obj;
  gx::g_gxState.textures[map] = gfx::TextureBind{obj, std::move(handle)};
  gx::g_gxState.textures[map].material.id = registry;
}

map_pick::MapSelection albedo_only(GXTexMapID map = GX_TEXMAP0) {
  return {.albedo = {.map = map, .coord = GX_TEXCOORD0, .stage = 0, .texcoordSource = true, .srcTex = 0}};
}

GpuMaterial words_of(uint32_t id) {
  const std::shared_ptr<const Snapshot> snap = snapshot();
  return snap != nullptr && id < snap->materials.size() ? snap->materials[id] : GpuMaterial{};
}

uint32_t flags_of(uint32_t id) { return words_of(id).words[1] >> 16; }

class RenderAtlas : public ::testing::Test {
protected:
  void SetUp() override {
    testing::reset();
    testing::set_active(true);
    gx::g_gxState = {};
  }

  void TearDown() override {
    testing::reset();
    gx::g_gxState = {};
  }
};

TEST_F(RenderAtlas, SkipsCopiesAndNoCacheTextures) {
  const gfx::TextureHandle plain = texture();
  bind_texture(GX_TEXMAP0, texture_object(1), plain);
  const uint32_t ok = material_for(gx::g_gxState, albedo_only(), true);
  ASSERT_NE(ok, NoMaterial);
  EXPECT_NE(testing::material_tile(ok, false), NoTile);
  EXPECT_EQ(flags_of(ok) & flag::AlbedoMissing, 0u);

  const GXTexObj_ copy = texture_object(2);
  bind_texture(GX_TEXMAP1, copy, texture());
  gx::g_gxState.copyTextures[copy.data] = {};
  const uint32_t copied = material_for(gx::g_gxState, albedo_only(GX_TEXMAP1), true);
  ASSERT_NE(copied, NoMaterial);
  EXPECT_EQ(testing::material_tile(copied, false), NoTile);
  EXPECT_NE(flags_of(copied) & flag::AlbedoMissing, 0u);
  gx::g_gxState.copyTextures.clear();

  GXTexObj_ uncached = texture_object(3);
  uncached.set_no_cache(true);
  bind_texture(GX_TEXMAP2, uncached, texture());
  const uint32_t noCache = material_for(gx::g_gxState, albedo_only(GX_TEXMAP2), true);
  EXPECT_EQ(testing::material_tile(noCache, false), NoTile);

  GXTexObj_ indexed = texture_object(4, GX_TF_C8);
  indexed.tlut = GX_TLUT0;
  gx::g_gxState.loadedTluts[0].set_no_cache(true);
  bind_texture(GX_TEXMAP3, indexed, texture());
  const uint32_t tlut = material_for(gx::g_gxState, albedo_only(GX_TEXMAP3), true);
  EXPECT_EQ(testing::material_tile(tlut, false), NoTile);
  EXPECT_NE(flags_of(tlut) & flag::AlbedoMissing, 0u);

  bind_texture(GX_TEXMAP4, texture_object(5), texture(64, wgpu::TextureFormat::RG8Unorm));
  const uint32_t odd = material_for(gx::g_gxState, albedo_only(GX_TEXMAP4), true);
  EXPECT_EQ(testing::material_tile(odd, false), NoTile);

  latch();
  const Stats stats = published_stats();
  EXPECT_EQ(stats.copies, 1u);
  EXPECT_EQ(stats.noCache, 2u);
  EXPECT_EQ(stats.unsupported, 1u);
  EXPECT_EQ(stats.tiles, 1u);
}

TEST_F(RenderAtlas, MaterialKeyHoldsTextureFieldsOnly) {
  gx::g_gxState.colorChannelState[0].matColor = {1.f, 0.f, 0.f, 1.f};
  const uint32_t red = material_for(gx::g_gxState, {}, true);
  uint64_t redTuple = 0;
  const DrawAttributes redAttributes = draw_attributes({}, redTuple, nullptr, nullptr, nullptr, gx::g_gxState);
  gx::g_gxState.colorChannelState[0].matColor = {0.f, 0.f, 1.f, 0.5f};
  const uint32_t blue = material_for(gx::g_gxState, {}, true);
  uint64_t blueTuple = 0;
  const DrawAttributes blueAttributes = draw_attributes({}, blueTuple, nullptr, nullptr, nullptr, gx::g_gxState);
  EXPECT_EQ(red, blue);
  EXPECT_EQ(redAttributes.constColor, 0xFF0000FFu);
  EXPECT_EQ(blueAttributes.constColor, 0x80FF0000u);
  EXPECT_EQ(redAttributes.receiver & attribute_flag::WordKnown << 8, 0u);
  EXPECT_EQ(testing::refs(red), 2u);

  bind_texture(GX_TEXMAP0, texture_object(7), texture(), 42);
  gx::PipelineConfig config{};
  config.shaderConfig.tevStageCount = 1;
  config.cullMode = GX_CULL_NONE;
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  DrawClass cls;
  cls.key.categories = category::Static | category::Terrain;
  const uint32_t none = material_for(gx::g_gxState, albedo_only(), true);
  uint64_t noneTuple = 0;
  const DrawAttributes noneAttributes = draw_attributes({}, noneTuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_EQ(noneAttributes.word, gx_ext::unlit::draw_material(config, info, cls).word);
  EXPECT_NE(noneAttributes.receiver & attribute_flag::CullNone << 8, 0u);
  config.cullMode = GX_CULL_BACK;
  const uint32_t back = material_for(gx::g_gxState, albedo_only(), true);
  uint64_t backTuple = 0;
  const DrawAttributes backAttributes = draw_attributes({}, backTuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_EQ(backAttributes.word, gx_ext::unlit::draw_material(config, info, cls).word);
  EXPECT_EQ(backAttributes.receiver & attribute_flag::CullNone << 8, 0u);
  EXPECT_NE(backAttributes.receiver & attribute_flag::WordKnown << 8, 0u);
  EXPECT_EQ(backAttributes.receiver & 0xFFu, receiver_category::compress(cls.key.categories));
  EXPECT_EQ(none, back);
  EXPECT_NE(none, red);
  EXPECT_EQ(words_of(none).words[3], 42u);
}

TEST_F(RenderAtlas, DrawAttributesRefreshOnChange) {
  gx::PipelineConfig config{};
  config.shaderConfig.tevStageCount = 1;
  config.cullMode = GX_CULL_BACK;
  const gx::ShaderInfo info = gx::build_shader_info(config.shaderConfig);
  DrawClass cls;
  cls.key.categories = category::Dynamic;
  uint64_t tuple = 0;
  DrawAttributes attributes = draw_attributes({}, tuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_EQ(published_stats().attributeRecomputes, 0u);
  latch();
  EXPECT_EQ(published_stats().attributeRecomputes, 1u);
  gx::g_gxState.colorChannelState[0].matColor = {0.f, 1.f, 0.f, 1.f};
  attributes = draw_attributes(attributes, tuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_EQ(attributes.constColor, 0xFF00FF00u);
  gx::g_gxState.colorChannelConfig[0].matSrc = GX_SRC_VTX;
  attributes = draw_attributes(attributes, tuple, nullptr, nullptr, nullptr, gx::g_gxState);
  EXPECT_EQ(attributes.constColor, 0xFFFFFFFFu);
  latch();
  EXPECT_EQ(published_stats().attributeRecomputes, 1u);
  EXPECT_EQ(published_stats().attributeCaptures, 3u);
  cls.key.categories = category::Dynamic | category::Player;
  attributes = draw_attributes(attributes, tuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_EQ(attributes.receiver & 0xFFu, receiver_category::compress(cls.key.categories));
  EXPECT_EQ(attributes.receiver & attribute_flag::CullNone << 8, 0u);
  config.cullMode = GX_CULL_NONE;
  attributes = draw_attributes(attributes, tuple, &config, &info, &cls, gx::g_gxState);
  EXPECT_NE(attributes.receiver & attribute_flag::CullNone << 8, 0u);
  attributes = draw_attributes(attributes, tuple, &config, &info, &cls, gx::g_gxState);
  latch();
  EXPECT_EQ(published_stats().attributeRecomputes, 3u);
}

TEST_F(RenderAtlas, TileLifecycle) {
  gfx::TextureHandle handle = texture();
  bind_texture(GX_TEXMAP0, texture_object(1), handle);
  const uint32_t id = material_for(gx::g_gxState, albedo_only(), true);
  const uint32_t tile = testing::material_tile(id, false);
  ASSERT_NE(tile, NoTile);
  EXPECT_TRUE(testing::tile(tile).live);
  EXPECT_EQ(testing::tile(tile).refs, 1u);
  EXPECT_EQ(material_for(gx::g_gxState, albedo_only(), true), id);
  EXPECT_EQ(testing::refs(id), 2u);
  release(id);
  release(id);
  latch();
  EXPECT_EQ(testing::tile(tile).refs, 1u);
  latch();
  EXPECT_EQ(testing::refs(id), 0u);
  EXPECT_EQ(testing::tile(tile).refs, 0u);
  EXPECT_TRUE(testing::tile(tile).live);
  EXPECT_EQ(material_for(gx::g_gxState, albedo_only(), true), id);
  EXPECT_EQ(testing::material_tile(id, false), tile);
  release(id);
  latch();
  latch();
  gx::g_gxState.textures[GX_TEXMAP0] = {};
  handle.reset();
  latch();
  EXPECT_FALSE(testing::tile(tile).live);
}

TEST_F(RenderAtlas, PendingUntilAcked) {
  const gfx::TextureHandle handle = texture(512, wgpu::TextureFormat::BGRA8UnormSrgb, 3);
  bind_texture(GX_TEXMAP0, texture_object(1), handle);
  const uint32_t id = material_for(gx::g_gxState, albedo_only(), true);
  const uint32_t tile = testing::material_tile(id, false);
  const std::shared_ptr<const Snapshot> first = snapshot();
  ASSERT_EQ(first->blits.size(), 1u);
  EXPECT_EQ(first->blits[0].tile, tile);
  EXPECT_EQ(first->blits[0].level, 2u);
  EXPECT_EQ(first->blits[0].taps, 1u);
  EXPECT_TRUE(first->blits[0].srgb);
  EXPECT_NE(first->materials[id].words[1] >> 16 & flag::AlbedoPending, 0u);
  EXPECT_TRUE(snapshot()->blits.empty());
  const std::array<uint32_t, 1> acked{tile};
  ack(acked);
  EXPECT_NE(flags_of(id) & flag::AlbedoPending, 0u);
  latch();
  EXPECT_EQ(flags_of(id) & flag::AlbedoPending, 0u);
  EXPECT_TRUE(testing::tile(tile).ready);

  bind_texture(GX_TEXMAP1, texture_object(2), texture(64));
  const uint32_t other = material_for(gx::g_gxState, albedo_only(GX_TEXMAP1), true);
  const uint32_t otherTile = testing::material_tile(other, false);
  const std::shared_ptr<const Snapshot> sent = snapshot();
  ASSERT_EQ(sent->blits.size(), 1u);
  EXPECT_EQ(sent->blits[0].taps, 1u);
  EXPECT_EQ(sent->blits[0].level, 0u);
  for (uint32_t i = 0; i < BlitRetryLatches; ++i) {
    EXPECT_TRUE(snapshot()->blits.empty()) << i;
    latch();
  }
  const std::shared_ptr<const Snapshot> retried = snapshot();
  ASSERT_EQ(retried->blits.size(), 1u);
  EXPECT_EQ(retried->blits[0].tile, otherTile);
  EXPECT_EQ(published_stats().retries, 1u);
  EXPECT_FALSE(testing::tile(otherTile).ready);
}

TEST_F(RenderAtlas, PackedRecordRoundTrips) {
  const gx::AlphaCompare compare{GX_GREATER, 128, GX_AOP_XNOR, GX_LEQUAL, 200};
  const uint32_t wraps = GX_REPEAT | GX_MIRROR << 2 | GX_CLAMP << 4 | GX_REPEAT << 6;
  const GpuMaterial m =
      pack_material(5, 1023, wraps, GX_CH_BLUE, true, flag::AlphaTest | flag::AlbedoPending, compare, 0xDEADBEEF);
  EXPECT_EQ(m.words[0] & 0xFFFFu, 5u);
  EXPECT_EQ(m.words[0] >> 16, 1023u);
  EXPECT_EQ(m.words[1] & 3u, static_cast<uint32_t>(GX_REPEAT));
  EXPECT_EQ(m.words[1] >> 2 & 3u, static_cast<uint32_t>(GX_MIRROR));
  EXPECT_EQ(m.words[1] >> 4 & 3u, static_cast<uint32_t>(GX_CLAMP));
  EXPECT_EQ(m.words[1] >> 6 & 3u, static_cast<uint32_t>(GX_REPEAT));
  EXPECT_EQ(m.words[1] >> 8 & 3u, static_cast<uint32_t>(GX_CH_BLUE));
  EXPECT_EQ(m.words[1] >> 10 & 1u, 1u);
  EXPECT_EQ(m.words[1] >> 16, flag::AlphaTest | flag::AlbedoPending);
  EXPECT_EQ(m.words[2] & 7u, static_cast<uint32_t>(GX_GREATER));
  EXPECT_EQ(m.words[2] >> 3 & 7u, static_cast<uint32_t>(GX_LEQUAL));
  EXPECT_EQ(m.words[2] >> 6 & 3u, static_cast<uint32_t>(GX_AOP_XNOR));
  EXPECT_EQ(m.words[2] >> 8 & 0xFFu, 128u);
  EXPECT_EQ(m.words[2] >> 16 & 0xFFu, 200u);
  EXPECT_EQ(m.words[3], 0xDEADBEEFu);
  for (size_t i = 4; i < m.words.size(); ++i) {
    EXPECT_EQ(m.words[i], 0u) << i;
  }
  const GpuMaterial open = pack_material(NoTile, NoTile, 0, 0, false, 0, compare, 0);
  EXPECT_EQ(open.words[2] & 0x3Fu, static_cast<uint32_t>(GX_ALWAYS) | static_cast<uint32_t>(GX_ALWAYS) << 3);
  const std::array<uint32_t, 3> words =
      pack_attributes({.word = 0x00010203u, .constColor = 0x80402010u, .receiver = 0x82u | 3u << 8});
  EXPECT_EQ(words, (std::array<uint32_t, 3>{0x00010203u, 0x80402010u, 0x382u}));
  EXPECT_EQ(sizeof(GpuMaterial), MaterialWords * 4);
}

TEST_F(RenderAtlas, MaterialAndTileIdsWaitOneLatch) {
  bind_texture(GX_TEXMAP0, texture_object(1), texture());
  const uint32_t first = material_for(gx::g_gxState, albedo_only(), true);
  release(first);
  bind_texture(GX_TEXMAP0, texture_object(2), texture());
  const uint32_t second = material_for(gx::g_gxState, albedo_only(), true);
  EXPECT_NE(second, first);
  latch();
  bind_texture(GX_TEXMAP0, texture_object(3), texture());
  const uint32_t third = material_for(gx::g_gxState, albedo_only(), true);
  EXPECT_NE(third, first);
  latch();
  bind_texture(GX_TEXMAP0, texture_object(4), texture());
  EXPECT_EQ(material_for(gx::g_gxState, albedo_only(), true), first);

  testing::reset();
  testing::set_active(true);
  std::vector<gfx::TextureHandle> handles;
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < MaxTiles; ++i) {
    handles.push_back(texture());
    bind_texture(GX_TEXMAP0, texture_object(100 + i), handles.back());
    ids.push_back(material_for(gx::g_gxState, albedo_only(), true));
  }
  const uint32_t freedTile = testing::material_tile(ids[7], false);
  release(ids[7]);
  latch();
  latch();
  EXPECT_EQ(testing::tile(freedTile).refs, 0u);
  bind_texture(GX_TEXMAP0, texture_object(5000), texture());
  const uint32_t blocked = material_for(gx::g_gxState, albedo_only(), true);
  EXPECT_EQ(testing::material_tile(blocked, false), NoTile);
  latch();
  EXPECT_EQ(published_stats().pageFull, 1u);
  bind_texture(GX_TEXMAP0, texture_object(5001), texture());
  const uint32_t evicting = material_for(gx::g_gxState, albedo_only(), true);
  EXPECT_EQ(testing::material_tile(evicting, false), freedTile);
  latch();
  EXPECT_EQ(published_stats().evictions, 1u);
}
} // namespace
} // namespace aurora::render::scene::atlas

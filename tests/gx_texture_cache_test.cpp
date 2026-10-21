#include "gx/texture.hpp"
#include "gx/guest_memory.hpp"
#include "gfx/lru_cache.hpp"

#include <aurora/texture.hpp>
#include <gtest/gtest.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gx::testing {
void reset_texture_stubs();
uint64_t texture_allocations();
uint64_t palette_conversions();
uint64_t copy_allocations();
gfx::TextureHandle make_texture_handle(uint32_t width, uint32_t height, u32 format = GX_TF_RGBA8_PC);
void set_replacement(gfx::TextureHandle handle, uint64_t id = 1);
void set_source_replacement(aurora::texture::TextureSourceKey key, gfx::TextureHandle handle);
} // namespace aurora::gx::testing

namespace aurora::gx {
namespace {
GXTexObj_ make_texture(const void* data, u32 id, u32 format = GX_TF_RGBA8_PC, u32 width = 2, u32 height = 2) {
  GXTexObj_ obj{};
  obj.data = data;
  obj.mWidth = width;
  obj.mHeight = height;
  obj.mFormat = format;
  obj.texObjId = id;
  obj.texDataVersion = 1;
  return obj;
}

GXTlutObj_ make_tlut(const void* data, u32 id, u16 entries = 16) {
  GXTlutObj_ tlut{};
  tlut.data = data;
  tlut.format = GX_TL_RGB5A3;
  tlut.numEntries = entries;
  tlut.tlutObjId = id;
  tlut.tlutDataVersion = 1;
  return tlut;
}

GXState::CopyTextureRef copy_texture(const void* dest, u32 width, u32 height = 1, GXTexFmt format = GX_TF_RGBA8) {
  return texture::resolve_copy_texture({.dest = dest, .width = width, .height = height, .format = format});
}

class GxTextureCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    texture::shutdown();
    g_gxState = {};
    testing::reset_texture_stubs();
    texture::end_frame();
  }

  void TearDown() override { texture::shutdown(); }
};

TEST_F(GxTextureCacheTest, CalculatesTiledAndLinearMipSourceSizes) {
  EXPECT_EQ(texture::texture_source_size(GX_TF_I4, 8, 8, 1), 32);
  EXPECT_EQ(texture::texture_source_size(GX_TF_I4, 9, 9, 3), 192);
  EXPECT_EQ(texture::texture_source_size(GX_TF_RGBA8, 5, 5, 1), 256);
  EXPECT_EQ(texture::texture_source_size(GX_TF_RGBA8_PC, 5, 3, 3), 72);
  EXPECT_EQ(texture::texture_source_size(GX_TF_BC1_PC, 5, 5, 2), 40);
  EXPECT_EQ(texture::tlut_source_size(256), 512);
}

TEST_F(GxTextureCacheTest, ReusesIdenticalContentAtDifferentAddresses) {
  std::array<uint8_t, 16> first{};
  std::array<uint8_t, 16> second{};
  const auto firstHandle = texture::resolve_static_texture(make_texture(first.data(), 1));
  const auto secondHandle = texture::resolve_static_texture(make_texture(second.data(), 2));

  EXPECT_EQ(firstHandle, secondHandle);
  EXPECT_EQ(testing::texture_allocations(), 1);
  EXPECT_EQ(texture_stats().contentHits, 1);
}

TEST_F(GxTextureCacheTest, ObjectHitDoesNotHashAgain) {
  std::array<uint8_t, 16> pixels{};
  const auto obj = make_texture(pixels.data(), 1);
  texture::resolve_static_texture(obj);
  const uint64_t hashedBytes = texture_stats().hashedBytes;

  texture::resolve_static_texture(obj);

  EXPECT_EQ(texture_stats().hashedBytes, hashedBytes);
  EXPECT_EQ(texture_stats().objectHits, 1);
  EXPECT_EQ(testing::texture_allocations(), 1);
}

TEST_F(GxTextureCacheTest, ChangedContentAndMetadataMiss) {
  std::array<uint8_t, 16> first{};
  std::array<uint8_t, 16> second{};
  second[7] = 1;
  texture::resolve_static_texture(make_texture(first.data(), 1));
  texture::resolve_static_texture(make_texture(second.data(), 2));

  std::array<uint8_t, 32> wider{};
  texture::resolve_static_texture(make_texture(wider.data(), 3, GX_TF_RGBA8_PC, 4, 2));

  EXPECT_EQ(testing::texture_allocations(), 3);
  EXPECT_EQ(texture_stats().misses, 3);
}

TEST_F(GxTextureCacheTest, DestroyedObjectStillReusesContent) {
  std::array<uint8_t, 16> pixels{};
  auto obj = make_texture(pixels.data(), 1);
  obj.set_no_cache(true);

  const auto first = texture::resolve_static_texture(obj);
  const auto second = texture::resolve_static_texture(obj);

  EXPECT_EQ(first, second);
  EXPECT_EQ(testing::texture_allocations(), 1);
  EXPECT_EQ(texture_stats().contentHits, 1);
  EXPECT_EQ(texture_stats().objectHits, 0);
}

TEST_F(GxTextureCacheTest, PaletteContentIncludesTlutBytesAndMetadata) {
  std::array<uint8_t, 32> indicesA{};
  std::array<uint8_t, 32> indicesB{};
  std::array<uint8_t, 32> paletteA{};
  std::array<uint8_t, 32> paletteB{};
  auto objA = make_texture(indicesA.data(), 1, GX_TF_C4, 4, 4);
  auto objB = make_texture(indicesB.data(), 2, GX_TF_C4, 4, 4);
  auto tlutA = make_tlut(paletteA.data(), 1);
  auto tlutB = make_tlut(paletteB.data(), 2);

  const auto first = texture::resolve_static_palette_texture(objA, tlutA);
  const auto second = texture::resolve_static_palette_texture(objB, tlutB);
  EXPECT_EQ(first, second);
  EXPECT_EQ(testing::texture_allocations(), 1);

  paletteB[3] = 1;
  tlutB.tlutDataVersion = 2;
  const auto third = texture::resolve_static_palette_texture(objB, tlutB);
  EXPECT_NE(first, third);
  EXPECT_EQ(testing::texture_allocations(), 2);

  auto tlutC = make_tlut(paletteA.data(), 3);
  tlutC.format = GX_TL_RGB565;
  texture::resolve_static_palette_texture(make_texture(indicesA.data(), 3, GX_TF_C4, 4, 4), tlutC);
  EXPECT_EQ(testing::texture_allocations(), 3);
}

TEST_F(GxTextureCacheTest, StaticClearReresolvesBoundTextureButRetainsContent) {
  std::array<uint8_t, 16> pixels{};
  g_gxState.loadedTextures[0] = make_texture(pixels.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  const auto first = g_gxState.textures[0].ref;

  clear_static_texture_cache();
  resolve_sampled_textures(info);

  EXPECT_EQ(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::texture_allocations(), 1);
  EXPECT_EQ(texture_stats().contentHits, 1);
}

TEST_F(GxTextureCacheTest, ReplacementAppearsOnAlreadyBoundTexture) {
  std::array<uint8_t, 16> pixels{};
  g_gxState.loadedTextures[0] = make_texture(pixels.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  const auto raw = g_gxState.textures[0].ref;

  const auto replacement = testing::make_texture_handle(8, 8);
  testing::set_replacement(replacement);
  clear_static_texture_cache();
  resolve_sampled_textures(info);

  EXPECT_NE(raw, replacement);
  EXPECT_EQ(g_gxState.textures[0].ref, replacement);
  EXPECT_EQ(texture_stats().replacementHits, 1);
}

TEST_F(GxTextureCacheTest, PendingReplacementPublishEvictsItsVanillaObjectEntry) {
  std::array<uint8_t, 16> pixels{};
  const auto obj = make_texture(pixels.data(), 1);
  testing::set_replacement({}, 77);
  const auto vanilla = texture::resolve_static_texture(obj);
  ASSERT_NE(vanilla, nullptr);

  const auto replacement = testing::make_texture_handle(8, 8);
  testing::set_replacement(replacement, 77);
  texture::invalidate_replacement(77);
  const auto resolved = texture::resolve_static_texture(obj);

  EXPECT_NE(vanilla, replacement);
  EXPECT_EQ(resolved, replacement);
  EXPECT_EQ(texture_stats().objectHits, 0);
}

TEST_F(GxTextureCacheTest, PendingReplacementPublishRebindsLoadedSlot) {
  std::array<uint8_t, 16> pixels{};
  g_gxState.loadedTextures[0] = make_texture(pixels.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  testing::set_replacement({}, 91);
  resolve_sampled_textures(info);
  const auto vanilla = g_gxState.textures[0].ref;
  ASSERT_NE(vanilla, nullptr);

  const auto replacement = testing::make_texture_handle(8, 8);
  testing::set_replacement(replacement, 91);
  texture::invalidate_replacement(91);
  texture::invalidate_bindings();
  resolve_sampled_textures(info);

  EXPECT_NE(vanilla, replacement);
  EXPECT_EQ(g_gxState.textures[0].ref, replacement);
}

TEST_F(GxTextureCacheTest, SourceReplacementReusesContentHashPass) {
  std::array<uint8_t, 16> pixels{};
  pixels[4] = 7;
  const auto replacement = testing::make_texture_handle(8, 8);
  testing::set_source_replacement(
      {
          .textureHash = XXH64(pixels.data(), pixels.size(), 0),
          .width = 2,
          .height = 2,
          .format = GX_TF_RGBA8_PC,
          .hasTlut = false,
      },
      replacement);

  const auto resolved = texture::resolve_static_texture(make_texture(pixels.data(), 1));

  EXPECT_EQ(resolved, replacement);
  EXPECT_EQ(testing::texture_allocations(), 0);
  EXPECT_EQ(texture_stats().hashedBytes, pixels.size());
  EXPECT_EQ(texture_stats().replacementHits, 1);
}

TEST_F(GxTextureCacheTest, TlutRefreshInvalidatesBoundPaletteTexture) {
  std::array<uint8_t, 32> indices{};
  std::array<uint8_t, 32> palette{};
  g_gxState.loadedTextures[0] = make_texture(indices.data(), 1, GX_TF_C4, 4, 4);
  g_gxState.loadedTextures[0].tlut = GX_TLUT0;
  g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  const auto first = g_gxState.textures[0].ref;

  palette[1] = 1;
  ++g_gxState.loadedTluts[0].tlutDataVersion;
  texture::invalidate_bindings();
  resolve_sampled_textures(info);

  EXPECT_NE(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::texture_allocations(), 2);
}

TEST_F(GxTextureCacheTest, CopyRevisionRequeuesDynamicPaletteConversion) {
  std::array<uint8_t, 32> palette{};
  copy_texture(palette.data(), 4, 4, GX_TF_I8);
  g_gxState.loadedTextures[0] = make_texture(palette.data(), 1, GX_TF_C8, 4, 4);
  g_gxState.loadedTextures[0].tlut = GX_TLUT0;
  g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1, 256);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  EXPECT_EQ(testing::palette_conversions(), 1);

  copy_texture(palette.data(), 4, 4, GX_TF_I8);
  resolve_sampled_textures(info);

  EXPECT_EQ(testing::palette_conversions(), 2);
}

TEST_F(GxTextureCacheTest, CopyRecreationAtSameDestinationRebindsHandle) {
  std::array<uint8_t, 16> destination{};
  const auto first = copy_texture(destination.data(), 2, 2).handle;
  g_gxState.loadedTextures[0] = make_texture(destination.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  EXPECT_EQ(g_gxState.textures[0].ref, first);

  evict_copy_texture(destination.data());
  const auto second = copy_texture(destination.data(), 2, 2).handle;
  resolve_sampled_textures(info);

  EXPECT_NE(first, second);
  EXPECT_EQ(g_gxState.textures[0].ref, second);
}

TEST_F(GxTextureCacheTest, CopyCacheReusesAlternatingSizesAcrossFrames) {
  uint8_t destination{};
  const auto quarter = copy_texture(&destination, 160, 120);
  const auto eighth = copy_texture(&destination, 80, 60);
  for (uint64_t frame = 0; frame < 4 * gfx::RenderTextureCacheLimits.idleFrames.value(); ++frame) {
    const auto repeatedQuarter = copy_texture(&destination, 160, 120);
    EXPECT_EQ(repeatedQuarter.handle, quarter.handle);
    EXPECT_GT(repeatedQuarter.revision, quarter.revision);
    EXPECT_EQ(copy_texture(&destination, 80, 60).handle, eighth.handle);
    EXPECT_EQ(copy_texture(&destination, 160, 120).handle, quarter.handle);
    texture::end_frame();
  }
  EXPECT_EQ(testing::copy_allocations(), 2);
  EXPECT_EQ(texture_stats().copyCacheBytes, 80 * 60 * 4);
  EXPECT_EQ(texture_stats().copyCacheEntries, 1);
  EXPECT_EQ(texture_stats().currentCopyBytes, 160 * 120 * 4);
}

TEST_F(GxTextureCacheTest, CopyCacheKeysIncludeBothDimensionsFormatAndDestination) {
  std::array<uint8_t, 2> destinations{};
  const auto base = copy_texture(&destinations[0], 8, 8, GX_TF_I4);
  const auto wide = copy_texture(&destinations[0], 16, 8, GX_TF_I4);
  const auto tall = copy_texture(&destinations[0], 8, 16, GX_TF_I4);
  const auto format = copy_texture(&destinations[0], 8, 8, GX_TF_I8);
  const auto other = copy_texture(&destinations[1], 8, 8, GX_TF_I4);
  EXPECT_NE(base.handle, wide.handle);
  EXPECT_NE(base.handle, tall.handle);
  EXPECT_NE(base.handle, format.handle);
  EXPECT_NE(base.handle, other.handle);
  EXPECT_EQ(copy_texture(&destinations[0], 8, 8, GX_TF_I4).handle, base.handle);
  EXPECT_EQ(testing::copy_allocations(), 5);
  // I4 copy textures occupy RGBA8 GPU storage, not the tiled source's 4 bits/pixel.
  EXPECT_EQ(texture_stats().copyCacheBytes, (16 * 8 + 8 * 16 + 8 * 8) * 4);
  EXPECT_EQ(texture_stats().currentCopyBytes, 2 * 8 * 8 * 4);
}

TEST_F(GxTextureCacheTest, CopyCacheBudgetUsesLruOrderWithinOneFrame) {
  texture::set_copy_cache_budget_for_testing(80);
  uint8_t destination{};
  const std::weak_ptr<gfx::TextureRef> a = copy_texture(&destination, 4).handle; // 16 bytes
  const std::weak_ptr<gfx::TextureRef> b = copy_texture(&destination, 8).handle; // 32 bytes
  copy_texture(&destination, 4); // Refresh A after B, without advancing the frame.
  const std::weak_ptr<gfx::TextureRef> c = copy_texture(&destination, 12).handle; // 48 bytes
  copy_texture(&destination, 16);

  EXPECT_FALSE(a.expired());
  EXPECT_TRUE(b.expired());
  EXPECT_FALSE(c.expired());
  EXPECT_EQ(texture_stats().copyCacheBytes, 64);
  EXPECT_EQ(texture_stats().copyCacheEntries, 2);
  EXPECT_EQ(texture_stats().copyCacheEvictions, 1);
  EXPECT_EQ(testing::copy_allocations(), 4);
  EXPECT_EQ(copy_texture(&destination, 4).handle, a.lock());
}

TEST_F(GxTextureCacheTest, CopyCacheBudgetIsSharedAcrossDestinations) {
  texture::set_copy_cache_budget_for_testing(64);
  std::array<uint8_t, 2> destinations{};
  const std::weak_ptr<gfx::TextureRef> a = copy_texture(&destinations[0], 16).handle;
  copy_texture(&destinations[0], 8);
  const std::weak_ptr<gfx::TextureRef> b = copy_texture(&destinations[1], 16).handle;
  copy_texture(&destinations[1], 8);
  EXPECT_TRUE(a.expired());
  EXPECT_FALSE(b.expired());
  EXPECT_EQ(texture_stats().copyCacheBytes, 64);
  EXPECT_EQ(texture_stats().currentCopyBytes, 64);
  EXPECT_EQ(g_gxState.copyTextures.size(), 2);
}

TEST_F(GxTextureCacheTest, CopyCacheHitAlsoTrimsNewlySupersededLargeCopy) {
  texture::set_copy_cache_budget_for_testing(32);
  uint8_t destination{};
  const auto small = copy_texture(&destination, 4);
  const std::weak_ptr<gfx::TextureRef> large = copy_texture(&destination, 64).handle;
  EXPECT_EQ(texture_stats().copyCacheBytes, 16);
  EXPECT_EQ(copy_texture(&destination, 4).handle, small.handle);
  EXPECT_TRUE(large.expired());
  EXPECT_EQ(texture_stats().copyCacheBytes, 0);
  EXPECT_EQ(testing::copy_allocations(), 2);
}

TEST_F(GxTextureCacheTest, CopyCacheSweepExpiresVariantsButPreservesCurrentContents) {
  std::array<uint8_t, 2> destinations{};
  // Keep an older current copy ahead of an expired variant in LRU order.
  const std::weak_ptr<gfx::TextureRef> persistent = copy_texture(&destinations[0], 16).handle;
  const std::weak_ptr<gfx::TextureRef> obsolete = copy_texture(&destinations[1], 8).handle;
  const std::weak_ptr<gfx::TextureRef> current = copy_texture(&destinations[1], 4).handle;
  // SetUp starts at frame 1. Frame 32 is a sweep, but these entries are only 31 frames old.
  for (uint64_t i = 1; i < gfx::RenderTextureCacheLimits.idleFrames.value(); ++i) {
    texture::end_frame();
  }
  EXPECT_FALSE(obsolete.expired());
  for (uint64_t i = 1; i < gfx::RenderTextureCacheLimits.sweepFrames; ++i) {
    texture::end_frame();
  }
  EXPECT_FALSE(obsolete.expired());
  texture::end_frame(); // Frame 48: first eligible sweep.
  EXPECT_TRUE(obsolete.expired());
  EXPECT_FALSE(persistent.expired());
  EXPECT_FALSE(current.expired());
  EXPECT_EQ(texture_stats().copyCacheBytes, 0);
  EXPECT_EQ(texture_stats().copyCacheEntries, 0);
  EXPECT_EQ(texture_stats().currentCopyBytes, 80);
  EXPECT_EQ(copy_texture(&destinations[1], 4).handle, current.lock());
}

TEST_F(GxTextureCacheTest, CopyCacheCanDisableReuseWithoutDiscardingCurrentCopies) {
  texture::set_copy_cache_budget_for_testing(0);
  uint8_t destination{};
  const std::weak_ptr<gfx::TextureRef> large = copy_texture(&destination, 4096, 4096).handle;
  EXPECT_FALSE(large.expired());
  EXPECT_EQ(copy_texture(&destination, 4096, 4096).handle, large.lock());
  for (uint64_t i = 0; i < 2 * gfx::RenderTextureCacheLimits.idleFrames.value(); ++i) {
    texture::end_frame();
  }
  EXPECT_FALSE(large.expired());
  copy_texture(&destination, 4);
  EXPECT_TRUE(large.expired());
  EXPECT_EQ(texture_stats().copyCacheBytes, 0);
  EXPECT_EQ(testing::copy_allocations(), 2);
}

TEST_F(GxTextureCacheTest, CopyCacheBoundsContinuousParticleSizeChurn) {
  uint8_t destination{};
  std::vector<std::weak_ptr<gfx::TextureRef>> copies;
  for (uint32_t width = 256; width < 2048; ++width) {
    copies.push_back(copy_texture(&destination, width, 1024, GX_TF_RGB565).handle);
    EXPECT_LE(texture_stats().copyCacheBytes, gfx::RenderTextureCacheLimits.bytes);
  }
  EXPECT_TRUE(copies.front().expired());
  EXPECT_FALSE(copies.back().expired());
  EXPECT_LT(texture_stats().copyCacheEntries, 16);
  EXPECT_GT(texture_stats().copyCacheEvictions, 0);
}

TEST_F(GxTextureCacheTest, CopyCacheEvictionRetainsOutstandingOwners) {
  uint8_t destination{};
  auto recordedCopy = copy_texture(&destination, 8);
  const std::weak_ptr<gfx::TextureRef> old = recordedCopy.handle;
  copy_texture(&destination, 4);
  texture::set_copy_cache_budget_for_testing(0);
  EXPECT_EQ(texture_stats().copyCacheBytes, 0);
  EXPECT_FALSE(old.expired());
  // Render passes retain TextureHandles in the same way until their work is retired.
  recordedCopy.handle.reset();
  EXPECT_TRUE(old.expired());
}

TEST_F(GxTextureCacheTest, CopyCacheEvictionRetiresDependentPaletteConversions) {
  uint8_t destination{};
  std::array<uint8_t, 512> palette{};
  copy_texture(&destination, 8, 8, GX_TF_I8);
  g_gxState.loadedTextures[0] = make_texture(&destination, 1, GX_TF_C8, 8, 8);
  g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1, 256);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);
  const std::weak_ptr<gfx::TextureRef> converted = g_gxState.textures[0].ref;
  g_gxState.textures[0].reset();

  texture::set_copy_cache_budget_for_testing(0);
  EXPECT_FALSE(converted.expired()); // Its source is still the current copy.
  copy_texture(&destination, 16, 8, GX_TF_I8);
  EXPECT_TRUE(converted.expired());
  resolve_sampled_textures(info);
  EXPECT_EQ(testing::palette_conversions(), 2);
}

TEST_F(GxTextureCacheTest, DestroyCopyRetiresEveryVariantAndOnlyThatDestination) {
  std::array<uint8_t, 2> destinations{};
  const std::weak_ptr<gfx::TextureRef> a = copy_texture(&destinations[0], 4).handle;
  const std::weak_ptr<gfx::TextureRef> b = copy_texture(&destinations[0], 8).handle;
  const std::weak_ptr<gfx::TextureRef> c = copy_texture(&destinations[1], 4).handle;
  const std::weak_ptr<gfx::TextureRef> d = copy_texture(&destinations[1], 8).handle;
  evict_copy_texture(&destinations[0]);
  evict_copy_texture(&destinations[0]); // Repeated retirement must not corrupt accounting.
  EXPECT_TRUE(a.expired());
  EXPECT_TRUE(b.expired());
  EXPECT_FALSE(c.expired());
  EXPECT_FALSE(d.expired());
  EXPECT_FALSE(g_gxState.copyTextures.contains(&destinations[0]));
  EXPECT_EQ(texture_stats().copyCacheBytes, 16);
  EXPECT_EQ(texture_stats().copyCacheEntries, 1);
  EXPECT_EQ(texture_stats().currentCopyBytes, 32);
  EXPECT_EQ(copy_texture(&destinations[1], 4).handle, c.lock());
}

TEST_F(GxTextureCacheTest, CopyCacheClearAndShutdownResetOwnershipAndAccounting) {
  uint8_t destination{};
  for (bool shutdown : {false, true}) {
    const std::weak_ptr<gfx::TextureRef> a = copy_texture(&destination, 4).handle;
    const std::weak_ptr<gfx::TextureRef> b = copy_texture(&destination, 8).handle;
    if (shutdown) {
      texture::shutdown();
    } else {
      clear_copy_texture_cache();
    }
    EXPECT_TRUE(a.expired());
    EXPECT_TRUE(b.expired());
    EXPECT_TRUE(g_gxState.copyTextures.empty());
    EXPECT_EQ(texture_stats().copyCacheBytes, 0);
    EXPECT_EQ(texture_stats().copyCacheEntries, 0);
    EXPECT_EQ(texture_stats().currentCopyBytes, 0);
  }
  copy_texture(&destination, 4);
  copy_texture(&destination, 8);
  EXPECT_EQ(texture_stats().copyCacheBytes, 16);
  EXPECT_EQ(texture_stats().currentCopyBytes, 32);
}

TEST_F(GxTextureCacheTest, ObjectAgingKeepsContentEntry) {
  std::array<uint8_t, 16> pixels{};
  const auto obj = make_texture(pixels.data(), 1);
  const auto first = texture::resolve_static_texture(obj);
  for (uint64_t i = 0; i <= texture::ObjectCacheIdleFrames; ++i) {
    texture::end_frame();
  }

  const auto second = texture::resolve_static_texture(obj);

  EXPECT_EQ(first, second);
  EXPECT_EQ(testing::texture_allocations(), 1);
  EXPECT_EQ(texture_stats().contentHits, 1);
  EXPECT_EQ(texture_stats().objectHits, 0);
}

TEST_F(GxTextureCacheTest, BoundObjectIsNotAgedOut) {
  std::array<uint8_t, 16> pixels{};
  g_gxState.loadedTextures[0] = make_texture(pixels.data(), 1);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  resolve_sampled_textures(info);

  for (uint64_t i = 0; i <= texture::ObjectCacheIdleFrames; ++i) {
    texture::end_frame();
    resolve_sampled_textures(info);
  }
  texture::invalidate_bindings();
  resolve_sampled_textures(info);

  EXPECT_EQ(testing::texture_allocations(), 1);
  EXPECT_EQ(texture_stats().hashedBytes, 0);
  EXPECT_EQ(texture_stats().objectHits, 1);
}

TEST_F(GxTextureCacheTest, ContentCacheEvictsLeastRecentlyUsedEntry) {
  texture::set_content_cache_budget_for_testing(128);
  std::array<uint8_t, 64> a{};
  std::array<uint8_t, 64> b{};
  std::array<uint8_t, 64> c{};
  b[0] = 1;
  c[0] = 2;
  texture::resolve_static_texture(make_texture(a.data(), 1, GX_TF_RGBA8_PC, 4, 4));
  texture::resolve_static_texture(make_texture(b.data(), 2, GX_TF_RGBA8_PC, 4, 4));
  texture::resolve_static_texture(make_texture(a.data(), 3, GX_TF_RGBA8_PC, 4, 4));
  texture::resolve_static_texture(make_texture(c.data(), 4, GX_TF_RGBA8_PC, 4, 4));
  texture::resolve_static_texture(make_texture(b.data(), 5, GX_TF_RGBA8_PC, 4, 4));

  EXPECT_EQ(testing::texture_allocations(), 4);
  EXPECT_EQ(texture_stats().evictions, 2);
  EXPECT_EQ(texture_stats().contentCacheBytes, 128);
  EXPECT_EQ(texture_stats().contentCacheEntries, 2);
}

TEST_F(GxTextureCacheTest, OversizedEntryIsReturnedButNotRetained) {
  texture::set_content_cache_budget_for_testing(32);
  std::array<uint8_t, 64> pixels{};
  auto obj = make_texture(pixels.data(), 1, GX_TF_RGBA8_PC, 4, 4);
  obj.set_no_cache(true);

  EXPECT_NE(texture::resolve_static_texture(obj), nullptr);
  EXPECT_NE(texture::resolve_static_texture(obj), nullptr);

  EXPECT_EQ(testing::texture_allocations(), 2);
  EXPECT_EQ(texture_stats().contentCacheEntries, 0);
  EXPECT_EQ(texture_stats().evictions, 0);
}

TEST_F(GxTextureCacheTest, ContentEvictionPreservesObjectAndRecordedOwners) {
  std::array<uint8_t, 64> pixels{};
  const auto obj = make_texture(pixels.data(), 1, GX_TF_RGBA8_PC, 4, 4);
  auto recorded = texture::resolve_static_texture(obj);
  const std::weak_ptr<gfx::TextureRef> resource = recorded;

  texture::set_content_cache_budget_for_testing(0);
  EXPECT_EQ(texture_stats().contentCacheBytes, 0);
  EXPECT_EQ(texture_stats().contentCacheEntries, 0);
  EXPECT_EQ(texture::resolve_static_texture(obj), recorded);
  EXPECT_EQ(testing::texture_allocations(), 1);

  evict_texture_object(obj.texObjId);
  EXPECT_FALSE(resource.expired());
  recorded.reset();
  EXPECT_TRUE(resource.expired());
}

TEST_F(GxTextureCacheTest, ShutdownRestoresContentCacheBudgetAndAccounting) {
  std::array<uint8_t, 64> pixels{};
  const auto obj = make_texture(pixels.data(), 1, GX_TF_RGBA8_PC, 4, 4);
  texture::resolve_static_texture(obj);
  texture::set_content_cache_budget_for_testing(0);
  texture::shutdown();
  EXPECT_EQ(texture_stats().contentCacheBytes, 0);
  EXPECT_EQ(texture_stats().contentCacheEntries, 0);
  EXPECT_EQ(texture_stats().evictions, 0);

  texture::resolve_static_texture(obj);
  EXPECT_EQ(texture_stats().contentCacheBytes, 64);
  EXPECT_EQ(texture_stats().contentCacheEntries, 1);
}

struct GuestRead {
  guest_memory::Kind kind;
  uint64_t address;
  size_t size;
};
std::vector<GuestRead> sGuestReads;
std::vector<std::vector<uint8_t>> sResolvedBlocks;

void observe_guest_read(void*, guest_memory::Kind kind, uint64_t address, std::span<const uint8_t> bytes) noexcept {
  sGuestReads.push_back({kind, address, bytes.size()});
}

// Fills each read with a byte derived from its address so identical guest buffers resolve to different content.
std::span<const uint8_t> resolve_by_address(void*, guest_memory::Kind, uint64_t address, size_t size) noexcept {
  auto& block = sResolvedBlocks.emplace_back(size, static_cast<uint8_t>(address >> 4));
  return {block.data(), block.size()};
}

class ScopedGuestHooks {
public:
  explicit ScopedGuestHooks(bool resolve) {
    sGuestReads.clear();
    sResolvedBlocks.clear();
    mHooks.observe = observe_guest_read;
    mHooks.resolve = resolve ? resolve_by_address : nullptr;
    guest_memory::set_hooks(&mHooks);
  }
  ~ScopedGuestHooks() { guest_memory::set_hooks(nullptr); }

private:
  guest_memory::Hooks mHooks;
};

uint64_t address_of(const void* ptr) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)); }

size_t count_reads(guest_memory::Kind kind) {
  return static_cast<size_t>(std::count_if(sGuestReads.begin(), sGuestReads.end(),
                                           [kind](const GuestRead& read) { return read.kind == kind; }));
}

TEST_F(GxTextureCacheTest, GuestTextureReadsAreObservedOnlyOnMiss) {
  std::array<uint8_t, 16> pixels{};
  const auto obj = make_texture(pixels.data(), 1);
  ScopedGuestHooks hooks{false};

  texture::resolve_static_texture(obj);
  ASSERT_EQ(sGuestReads.size(), 2u);
  for (const auto& read : sGuestReads) {
    EXPECT_EQ(read.kind, guest_memory::Kind::Texture);
    EXPECT_EQ(read.address, address_of(pixels.data()));
    EXPECT_EQ(read.size, 16u);
  }

  sGuestReads.clear();
  texture::resolve_static_texture(obj);
  EXPECT_TRUE(sGuestReads.empty());
}

TEST_F(GxTextureCacheTest, ResolverBytesDriveTextureContentIdentity) {
  alignas(32) std::array<uint8_t, 32> first{};
  alignas(32) std::array<uint8_t, 32> second{};
  ScopedGuestHooks hooks{true};

  texture::resolve_static_texture(make_texture(first.data(), 1));
  texture::resolve_static_texture(make_texture(second.data(), 2));

  EXPECT_EQ(testing::texture_allocations(), 2);
}

TEST_F(GxTextureCacheTest, PaletteReadsObserveTextureAndTlut) {
  std::array<uint8_t, 32> indices{};
  std::array<uint8_t, 32> palette{};
  ScopedGuestHooks hooks{false};

  texture::resolve_static_palette_texture(make_texture(indices.data(), 1, GX_TF_C4, 4, 4),
                                          make_tlut(palette.data(), 1));

  EXPECT_EQ(count_reads(guest_memory::Kind::Texture), 2u);
  EXPECT_EQ(count_reads(guest_memory::Kind::Tlut), 2u);
  for (const auto& read : sGuestReads) {
    if (read.kind == guest_memory::Kind::Tlut) {
      EXPECT_EQ(read.address, address_of(palette.data()));
      EXPECT_EQ(read.size, 32u);
    }
  }
}

TEST_F(GxTextureCacheTest, DynamicPaletteReadsTlutButNotCopyDestination) {
  std::array<uint8_t, 512> palette{};
  copy_texture(palette.data(), 4, 4, GX_TF_I8);
  g_gxState.loadedTextures[0] = make_texture(palette.data(), 1, GX_TF_C8, 4, 4);
  g_gxState.loadedTextures[0].tlut = GX_TLUT0;
  g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1, 256);
  ShaderInfo info{};
  info.sampledTextures.set(0);

  ScopedGuestHooks hooks{false};
  resolve_sampled_textures(info);

  ASSERT_EQ(sGuestReads.size(), 1u);
  EXPECT_EQ(sGuestReads[0].kind, guest_memory::Kind::Tlut);
  EXPECT_EQ(sGuestReads[0].address, address_of(palette.data()));
  EXPECT_EQ(sGuestReads[0].size, 512u);
  EXPECT_EQ(count_reads(guest_memory::Kind::Texture), 0u);
}

TEST_F(GxTextureCacheTest, ResetTextureCachesRereadsObjectAndTlutSources) {
  std::array<uint8_t, 16> pixels{};
  std::array<uint8_t, 512> palette{};
  copy_texture(palette.data(), 4, 4, GX_TF_I8);
  g_gxState.loadedTextures[0] = make_texture(pixels.data(), 1);
  g_gxState.loadedTextures[1] = make_texture(palette.data(), 2, GX_TF_C8, 4, 4);
  g_gxState.loadedTextures[1].tlut = GX_TLUT0;
  g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1, 256);
  ShaderInfo info{};
  info.sampledTextures.set(0);
  info.sampledTextures.set(1);

  ScopedGuestHooks hooks{false};
  resolve_sampled_textures(info);
  ASSERT_FALSE(sGuestReads.empty());

  sGuestReads.clear();
  texture::invalidate_bindings();
  resolve_sampled_textures(info);
  EXPECT_TRUE(sGuestReads.empty());

  reset_texture_caches();
  resolve_sampled_textures(info);
  const auto has_read = [](guest_memory::Kind kind, const void* data, size_t size) {
    return std::any_of(sGuestReads.begin(), sGuestReads.end(), [&](const GuestRead& read) {
      return read.kind == kind && read.address == address_of(data) && read.size == size;
    });
  };
  EXPECT_TRUE(has_read(guest_memory::Kind::Texture, pixels.data(), pixels.size()));
  EXPECT_TRUE(has_read(guest_memory::Kind::Tlut, palette.data(), palette.size()));
}

using Palette = std::array<uint8_t, 512>;

void load_copy_palette_texture(u32 slot, const void* destination, u32 texObjId, GXTlut tlut) {
  g_gxState.loadedTextures[slot] = make_texture(destination, texObjId, GX_TF_C8, 4, 4);
  g_gxState.loadedTextures[slot].tlut = tlut;
}

ShaderInfo sampling(u32 slot) {
  ShaderInfo info{};
  info.sampledTextures.set(slot);
  return info;
}

void bump_tlut(u32 slot, Palette& palette) {
  ++palette[1];
  ++g_gxState.loadedTluts[slot].tlutDataVersion;
  texture::invalidate_bindings();
}

class GxDynamicPaletteTest : public GxTextureCacheTest {
protected:
  void SetUp() override {
    GxTextureCacheTest::SetUp();
    copy_texture(&destination, 4, 4, GX_TF_I8);
    load_copy_palette_texture(0, &destination, 1, GX_TLUT0);
    g_gxState.loadedTluts[0] = make_tlut(palette.data(), 1, 256);
  }

  uint8_t destination{};
  Palette palette{};
};

TEST_F(GxDynamicPaletteTest, TlutBumpInLaterFrameReusesDynamicPaletteTarget) {
  ScopedGuestHooks hooks{false};
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;
  const uint64_t copies = testing::copy_allocations();
  const uint64_t textures = testing::texture_allocations();
  ASSERT_EQ(count_reads(guest_memory::Kind::Tlut), 1u);

  texture::end_frame();
  bump_tlut(0, palette);
  copy_texture(&destination, 4, 4, GX_TF_I8);
  resolve_sampled_textures(sampling(0));

  EXPECT_EQ(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::palette_conversions(), 2);
  EXPECT_EQ(testing::copy_allocations(), copies);
  EXPECT_EQ(testing::texture_allocations(), textures + 1);
  EXPECT_EQ(count_reads(guest_memory::Kind::Tlut), 2u);
}

TEST_F(GxDynamicPaletteTest, TlutBumpWithoutCopyInLaterFrameReconvertsKeptTarget) {
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;
  const uint64_t copies = testing::copy_allocations();

  texture::end_frame();
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_EQ(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::palette_conversions(), 2);
  EXPECT_EQ(testing::copy_allocations(), copies);
}

TEST_F(GxDynamicPaletteTest, TlutBumpInSameFrameKeepsSeparateDynamicPaletteTargets) {
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;
  const uint64_t copies = testing::copy_allocations();

  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_NE(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::copy_allocations(), copies + 1);
  EXPECT_EQ(testing::palette_conversions(), 2);
}

TEST_F(GxDynamicPaletteTest, TlutBumpAfterTouchInSameFrameKeepsSeparateTargets) {
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;

  texture::end_frame();
  resolve_sampled_textures(sampling(0));
  ASSERT_EQ(g_gxState.textures[0].ref, first);
  ASSERT_EQ(testing::palette_conversions(), 1);
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_NE(g_gxState.textures[0].ref, first);
}

TEST_F(GxDynamicPaletteTest, StaleTlutSlotBindingBlocksReuse) {
  Palette other{};
  g_gxState.loadedTluts[1] = make_tlut(other.data(), 2, 256);
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;

  texture::end_frame();
  // GXInitTexObjTlut moves the object to another slot without changing its id or data version
  g_gxState.loadedTextures[0].tlut = GX_TLUT1;
  resolve_sampled_textures(sampling(0));
  ASSERT_EQ(g_gxState.textures[0].ref, first);
  g_gxState.loadedTextures[0].tlut = GX_TLUT0;
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_NE(g_gxState.textures[0].ref, first);

  // The block ends with its frame, so the next frame's bump keeps and reconverts the target
  const auto blocked = g_gxState.textures[0].ref;
  const uint64_t copies = testing::copy_allocations();
  const uint64_t conversions = testing::palette_conversions();
  texture::end_frame();
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_EQ(g_gxState.textures[0].ref, blocked);
  EXPECT_EQ(testing::copy_allocations(), copies);
  EXPECT_EQ(testing::palette_conversions(), conversions + 1);
}

TEST_F(GxDynamicPaletteTest, ShutdownEndsReuseBlock) {
  Palette other{};
  g_gxState.loadedTluts[1] = make_tlut(other.data(), 2, 256);
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;
  texture::end_frame();
  g_gxState.loadedTextures[0].tlut = GX_TLUT1;
  resolve_sampled_textures(sampling(0));
  ASSERT_EQ(g_gxState.textures[0].ref, first);
  g_gxState.loadedTextures[0].tlut = GX_TLUT0;

  // The block was set in frame 2, and the frame count reaches 2 again after the shutdown
  texture::shutdown();
  g_gxState.textures[0] = {};
  copy_texture(&destination, 4, 4, GX_TF_I8);
  resolve_sampled_textures(sampling(0));
  const auto target = g_gxState.textures[0].ref;
  const uint64_t copies = testing::copy_allocations();
  texture::end_frame();
  texture::end_frame();
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_EQ(g_gxState.textures[0].ref, target);
  EXPECT_EQ(testing::copy_allocations(), copies);
}

TEST_F(GxDynamicPaletteTest, MatchingTouchDoesNotBlockReuse) {
  uint8_t otherDestination{};
  Palette other{};
  copy_texture(&otherDestination, 4, 4, GX_TF_I8);
  load_copy_palette_texture(1, &otherDestination, 2, GX_TLUT1);
  g_gxState.loadedTluts[1] = make_tlut(other.data(), 2, 256);
  ShaderInfo both = sampling(0);
  both.sampledTextures.set(1);
  resolve_sampled_textures(both);
  const auto first = g_gxState.textures[0].ref;

  texture::end_frame();
  resolve_sampled_textures(sampling(1));
  bump_tlut(0, palette);
  resolve_sampled_textures(sampling(0));

  EXPECT_EQ(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::palette_conversions(), 3);
}

TEST_F(GxDynamicPaletteTest, LiveBindingFromEarlierFrameKeepsItsTarget) {
  // Two slots hold one TLUT object at two data versions, and both textures sample the same copy
  g_gxState.loadedTluts[1] = g_gxState.loadedTluts[0];
  ++g_gxState.loadedTluts[1].tlutDataVersion;
  g_gxState.loadedTextures[0].tlut = GX_TLUT1;
  load_copy_palette_texture(1, &destination, 2, GX_TLUT0);
  resolve_sampled_textures(sampling(0));
  const auto first = g_gxState.textures[0].ref;

  texture::end_frame();
  resolve_sampled_textures(sampling(1));

  EXPECT_NE(g_gxState.textures[1].ref, first);
  EXPECT_EQ(g_gxState.textures[0].ref, first);
  EXPECT_EQ(testing::palette_conversions(), 2);
}
} // namespace
} // namespace aurora::gx

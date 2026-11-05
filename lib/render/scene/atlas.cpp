#include "atlas.hpp"

#include "classifier.hpp"
#include "identity.hpp"
#include "receiver_category.hpp"

#include "../graph_bind.hpp"
#include "../pass_context.hpp"
#include "../shaders.hpp"
#include "../stage_data.hpp"
#include "../gx_ext/unlit.hpp"
#include "../material/classes.hpp"
#include "../../gfx/hash.hpp"
#include "../../gfx/texture.hpp"
#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <absl/container/flat_hash_map.h>
#include <aurora/render_caps.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>

namespace aurora::render::scene::atlas {
namespace {
constexpr Module Log{"aurora::render::retained"};
constexpr std::string_view PassName = "scene.atlas";
constexpr std::string_view LibraryShader = "scene/atlas";
constexpr std::string_view PageName = "scene.atlas_page";
constexpr std::string_view MaterialsName = "scene.materials";
constexpr std::string_view MeansName = "scene.atlas_means";
constexpr uint64_t BlitLayoutKey = 0x61746C6173626C01ull;
constexpr uint64_t MeanLayoutKey = 0x61746C61736D6E01ull;
constexpr uint32_t ImmediateSize = 16;
constexpr uint64_t Never = 0;
constexpr std::array<shaders::Constant, 1> SrgbSource{{{"SRGB_SOURCE", 1.0}}};

struct Resolved {
  uint32_t tile = NoTile;
  uint32_t wraps = 0;
  uint32_t registry = 0;
  bool missing = false;
  GXTexObj_ obj{};
  std::weak_ptr<gfx::TextureRef> texture;
};

struct MaterialKey {
  uint32_t albedoTile = NoTile;
  uint32_t alphaTile = NoTile;
  uint32_t wraps = 0;
  uint32_t channel = 0;
  uint32_t flags = 0;
  uint32_t compare = 0;
  uint32_t registry = 0;
  bool alphaUsesUv0 = false;

  bool operator==(const MaterialKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const MaterialKey& key) {
    return H::combine(std::move(h), key.albedoTile, key.alphaTile, key.wraps, key.channel, key.flags, key.compare,
                      key.registry, key.alphaUsesUv0);
  }
};

struct MaterialEntry {
  MaterialKey key;
  gx::AlphaCompare compare{};
  bool live = false;
  bool released = false;
  uint32_t refs = 0;
  uint64_t releasedLatch = Never;
  MaterialInfo info;
};

struct TileEntry {
  std::weak_ptr<gfx::TextureRef> texture;
  const gfx::TextureRef* key = nullptr;
  bool live = false;
  bool ready = false;
  bool queued = false;
  uint32_t refs = 0;
  uint64_t lastUsed = Never;
  uint64_t releasedLatch = Never;
  uint64_t sentLatch = Never;
};

void default_resolve(const gx::ShaderInfo& info) noexcept { gx::resolve_sampled_textures(info); }

// FIFO thread, and the game thread while the FIFO is idle
bool g_active = false;
bool g_allocated = false;
uint64_t g_latch = 1;
std::vector<MaterialEntry> g_materials;
absl::flat_hash_map<MaterialKey, uint32_t> g_materialIndex;
std::vector<uint32_t> g_freeMaterials;
std::vector<uint32_t> g_releasedMaterials;
uint32_t g_used = 0;
std::vector<TileEntry> g_tiles;
absl::flat_hash_map<const gfx::TextureRef*, uint32_t> g_tileIndex;
std::vector<uint32_t> g_freeTiles;
std::deque<uint32_t> g_queue;
Stats g_stats;
testing::ResolveFn g_resolve = &default_resolve;

std::mutex g_ackMutex;
std::vector<uint32_t> g_acks;
std::mutex g_publishMutex;
Stats g_published;

// Game thread
detail::PersistentHandle g_pageHandle;
detail::PersistentHandle g_materialsHandle;
detail::PersistentHandle g_meansHandle;
wgpu::Texture g_page;
wgpu::Buffer g_materialsBuffer;
wgpu::Buffer g_meansBuffer;
detail::PassId g_pass = detail::InvalidPassId;
wgpu::BindGroupLayout g_blitGroupLayout;
wgpu::BindGroupLayout g_meanGroupLayout;
wgpu::PipelineLayout g_blitLayout;
wgpu::PipelineLayout g_meanLayout;
wgpu::Sampler g_sampler;
std::atomic_bool g_pageFresh{false};

struct Local {
  uint16_t page = UINT16_MAX;
  uint16_t materials = UINT16_MAX;
  uint16_t means = UINT16_MAX;
};
static_assert(sizeof(Local) <= detail::PassLocalSize);

uint32_t pack_rgba8(const Vec4<float>& c) noexcept {
  const auto unorm = [](float v) {
    return static_cast<uint32_t>(std::lround(std::clamp(std::isfinite(v) ? v : 0.f, 0.f, 1.f) * 255.f));
  };
  return unorm(c[0]) | unorm(c[1]) << 8 | unorm(c[2]) << 16 | unorm(c[3]) << 24;
}

bool srgb_format(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::RGBA8UnormSrgb:
  case wgpu::TextureFormat::BGRA8UnormSrgb:
  case wgpu::TextureFormat::BC1RGBAUnormSrgb:
  case wgpu::TextureFormat::BC2RGBAUnormSrgb:
  case wgpu::TextureFormat::BC3RGBAUnormSrgb:
  case wgpu::TextureFormat::BC7RGBAUnormSrgb:
    return true;
  default:
    return false;
  }
}

void free_tile(uint32_t id) noexcept {
  TileEntry& tile = g_tiles[id];
  if (const auto it = g_tileIndex.find(tile.key); it != g_tileIndex.end() && it->second == id) {
    g_tileIndex.erase(it);
  }
  std::erase(g_queue, id);
  tile = {};
  g_freeTiles.push_back(id);
}

void ref_tile(uint32_t id) noexcept {
  if (id != NoTile) {
    ++g_tiles[id].refs;
  }
}

void unref_tile(uint32_t id) noexcept {
  if (id == NoTile || !g_tiles[id].live || g_tiles[id].refs == 0) {
    return;
  }
  if (--g_tiles[id].refs == 0) {
    g_tiles[id].releasedLatch = g_latch;
  }
}

uint32_t evict_tile() noexcept {
  uint32_t best = NoTile;
  for (uint32_t id = 0; id < g_tiles.size(); ++id) {
    const TileEntry& tile = g_tiles[id];
    // A blit in flight may still land in this tile, so its ack would mark the next texture ready
    if (!tile.live || tile.refs != 0 || tile.sentLatch != Never || tile.releasedLatch >= g_latch) {
      continue;
    }
    if (best == NoTile || tile.lastUsed < g_tiles[best].lastUsed) {
      best = id;
    }
  }
  if (best != NoTile) {
    free_tile(best);
    ++g_stats.evictions;
  }
  return best;
}

uint32_t tile_for(const gfx::TextureHandle& texture) noexcept {
  if (const auto it = g_tileIndex.find(texture.get()); it != g_tileIndex.end()) {
    TileEntry& tile = g_tiles[it->second];
    if (tile.texture.lock() == texture) {
      tile.lastUsed = g_latch;
      return it->second;
    }
    free_tile(it->second);
  }
  if (g_freeTiles.empty() && evict_tile() == NoTile) {
    ++g_stats.pageFull;
    return NoTile;
  }
  const uint32_t id = g_freeTiles.back();
  g_freeTiles.pop_back();
  g_tiles[id] = {.texture = texture,
                 .key = texture.get(),
                 .live = true,
                 .queued = true,
                 .lastUsed = g_latch,
                 .releasedLatch = g_latch};
  g_tileIndex[texture.get()] = id;
  g_queue.push_back(id);
  return id;
}

Resolved resolve(const gx::GXState& state, const map_pick::MapPick& pick, bool visible) noexcept {
  Resolved out{.missing = true};
  if (!pick.valid() || pick.map >= gx::MaxTextures) {
    return out;
  }
  const GXTexObj_& loaded = visible ? state.textures[pick.map].texObj : state.loadedTextures[pick.map];
  out.obj = loaded;
  // Copies, palette conversions and no-cache textures change under one object, so a tile would go stale (M4 C9)
  if (loaded.data != nullptr && state.copyTextures.contains(loaded.data)) {
    ++g_stats.copies;
    return out;
  }
  if (loaded.no_cache() ||
      (gx::is_palette_format(loaded.format()) && static_cast<size_t>(loaded.tlut) < state.loadedTluts.size() &&
       state.loadedTluts[loaded.tlut].no_cache())) {
    ++g_stats.noCache;
    return out;
  }
  if (!visible) {
    gx::ShaderInfo info{};
    info.sampledTextures.set(pick.map);
    g_resolve(info);
  }
  const gfx::TextureBind& bind = state.textures[pick.map];
  if (!bind.ref) {
    ++g_stats.missing;
    return out;
  }
  if (!supported_format(bind.ref->format)) {
    ++g_stats.unsupported;
    return out;
  }
  ++g_stats.resolved;
  out.tile = tile_for(bind.ref);
  out.missing = out.tile == NoTile;
  out.wraps = static_cast<uint32_t>(bind.texObj.wrap_s()) | static_cast<uint32_t>(bind.texObj.wrap_t()) << 2;
  out.registry = bind.material.id;
  out.texture = bind.ref;
  return out;
}

uint32_t compare_word(const gx::AlphaCompare& c) noexcept {
  return (static_cast<uint32_t>(c.comp0) & 7u) | (static_cast<uint32_t>(c.comp1) & 7u) << 3 |
         (static_cast<uint32_t>(c.op) & 3u) << 6 | (c.ref0 & 0xFFu) << 8 | (c.ref1 & 0xFFu) << 16;
}

uint32_t allocate_material() noexcept {
  if (!g_freeMaterials.empty()) {
    const uint32_t id = g_freeMaterials.back();
    g_freeMaterials.pop_back();
    return id;
  }
  return g_used < MaxMaterials ? g_used++ : NoMaterial;
}

GpuMaterial pack_entry(const MaterialEntry& m) noexcept {
  uint32_t flags = m.key.flags;
  const auto pending = [](uint32_t tile) { return tile != NoTile && !g_tiles[tile].ready; };
  flags |= pending(m.key.albedoTile) ? material_flag::AlbedoPending : 0u;
  flags |= pending(m.key.alphaTile) ? material_flag::AlphaPending : 0u;
  return pack_material(m.key.albedoTile, m.key.alphaTile, m.key.wraps, m.key.channel, m.key.alphaUsesUv0, flags,
                       m.compare, m.key.registry);
}

void free_material(uint32_t id) noexcept {
  MaterialEntry& m = g_materials[id];
  unref_tile(m.key.albedoTile);
  unref_tile(m.key.alphaTile);
  if (const auto it = g_materialIndex.find(m.key); it != g_materialIndex.end() && it->second == id) {
    g_materialIndex.erase(it);
  }
  m = {};
  g_freeMaterials.push_back(id);
}

void clear_state() noexcept {
  g_materials.assign(g_allocated ? MaxMaterials : 0, MaterialEntry{});
  g_materialIndex.clear();
  g_freeMaterials.clear();
  g_releasedMaterials.clear();
  g_used = 0;
  g_tiles.assign(g_allocated ? MaxTiles : 0, TileEntry{});
  g_tileIndex.clear();
  g_freeTiles.clear();
  for (uint32_t id = static_cast<uint32_t>(g_tiles.size()); id-- > 0;) {
    g_freeTiles.push_back(id);
  }
  g_queue.clear();
  std::lock_guard lock{g_ackMutex};
  g_acks.clear();
}

void publish() noexcept {
  Stats stats = g_stats;
  stats.active = g_active;
  stats.materialsUsed = g_used;
  for (const MaterialEntry& m : g_materials) {
    stats.materials += m.live ? 1 : 0;
  }
  for (const TileEntry& tile : g_tiles) {
    stats.tiles += tile.live ? 1 : 0;
    stats.tilesReady += tile.live && tile.ready ? 1 : 0;
    stats.tilesPending += tile.live && !tile.ready ? 1 : 0;
  }
  std::lock_guard lock{g_publishMutex};
  g_published = stats;
}

bool create_layouts() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device || !caps().core || caps().limits.maxImmediateSize < ImmediateSize) {
    return false;
  }
  const std::array<wgpu::BindGroupLayoutEntry, 2> blitEntries{{
      {.binding = 0,
       .visibility = wgpu::ShaderStage::Fragment,
       .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D}},
      {.binding = 1,
       .visibility = wgpu::ShaderStage::Fragment,
       .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
  }};
  const wgpu::BindGroupLayoutDescriptor blitGroup{
      .label = "scene.atlas.blit", .entryCount = blitEntries.size(), .entries = blitEntries.data()};
  g_blitGroupLayout = device.CreateBindGroupLayout(&blitGroup);
  const std::array<wgpu::BindGroupLayoutEntry, 2> meanEntries{{
      {.binding = 0,
       .visibility = wgpu::ShaderStage::Compute,
       .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                   .viewDimension = wgpu::TextureViewDimension::e2D}},
      {.binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage}},
  }};
  const wgpu::BindGroupLayoutDescriptor meanGroup{
      .label = "scene.atlas.mean", .entryCount = meanEntries.size(), .entries = meanEntries.data()};
  g_meanGroupLayout = device.CreateBindGroupLayout(&meanGroup);
  const wgpu::PipelineLayoutDescriptor blitLayout{.label = "scene.atlas.blit",
                                                  .bindGroupLayoutCount = 1,
                                                  .bindGroupLayouts = &g_blitGroupLayout,
                                                  .immediateSize = ImmediateSize};
  g_blitLayout = device.CreatePipelineLayout(&blitLayout);
  const wgpu::PipelineLayoutDescriptor meanLayout{.label = "scene.atlas.mean",
                                                  .bindGroupLayoutCount = 1,
                                                  .bindGroupLayouts = &g_meanGroupLayout,
                                                  .immediateSize = ImmediateSize};
  g_meanLayout = device.CreatePipelineLayout(&meanLayout);
  const wgpu::SamplerDescriptor sampler{
      .label = "scene.atlas",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Nearest,
  };
  g_sampler = device.CreateSampler(&sampler);
  return true;
}

shaders::RenderPipelineDesc blit_desc(bool srgb) {
  shaders::RenderPipelineDesc desc{
      .label = PassName,
      .shader = LibraryShader,
      .vertexEntry = "vs_blit",
      .fragmentEntry = "fs_blit",
      .constants = srgb ? std::span<const shaders::Constant>{SrgbSource} : std::span<const shaders::Constant>{},
      .layout = g_blitLayout,
      .layoutKey = BlitLayoutKey,
      .colorCount = 1,
  };
  desc.colors[0] = {.format = PageFormat};
  return desc;
}

shaders::ComputePipelineDesc mean_desc() {
  return {
      .label = PassName,
      .shader = LibraryShader,
      .entryPoint = "cs_mean",
      .layout = g_meanLayout,
      .layoutKey = MeanLayoutKey,
  };
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (setup.env == nullptr || !setup.env->retained || !g_blitLayout || g_pageHandle.id == 0) {
    builder.skip();
    return;
  }
  builder.after(phase::Capture);
  builder.before(phase::Geometry);
  builder.manual_bindings();
  Local local;
  local.page = builder.import_persistent_texture(PageName, g_pageHandle, page_desc(), Access::ColorRMW).index;
  local.materials = builder.import_persistent(MaterialsName, g_materialsHandle, Access::CopyDst).index;
  local.means = builder.import_persistent(MeansName, g_meansHandle, Access::StorageRW).index;
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

bool enabled(const detail::StageData& stage, const void*) {
  return stage.retained != nullptr && stage.retained->atlas != nullptr;
}

void pass_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  if (!g_blitLayout) {
    ids[0] = ids[1] = ids[2] = shaders::NeverReady.value;
    return;
  }
  ids[0] = shaders::request_render(blit_desc(false)).value;
  ids[1] = shaders::request_render(blit_desc(true)).value;
  ids[2] = shaders::request_compute(mean_desc()).value;
}

struct BlitPipelines {
  wgpu::RenderPipeline linear;
  wgpu::RenderPipeline srgb;
  wgpu::ComputePipeline mean;
};

std::vector<uint32_t> record_blits(const wgpu::CommandEncoder& encoder, const BlitPipelines& pipelines,
                                   const wgpu::TextureView& page, const BufferView& means, const Snapshot& snapshot,
                                   bool clear) {
  std::vector<uint32_t> blitted;
  if (!pipelines.linear || !pipelines.srgb || !pipelines.mean || !page || !means.buffer) {
    return blitted;
  }
  const wgpu::RenderPassColorAttachment color{
      .view = page,
      .loadOp = clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor descriptor{
      .label = "scene.atlas", .colorAttachmentCount = 1, .colorAttachments = &color};
  blitted.reserve(snapshot.blits.size());
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&descriptor);
  for (const Blit& blit : snapshot.blits) {
    if (!blit.texture || !blit.texture->sampleTextureView || blit.tile >= MaxTiles) {
      continue;
    }
    const std::array<wgpu::BindGroupEntry, 2> entries{{
        {.binding = 0, .textureView = blit.texture->sampleTextureView},
        {.binding = 1, .sampler = g_sampler},
    }};
    const wgpu::BindGroupDescriptor group{.label = "scene.atlas.blit",
                                          .layout = g_blitGroupLayout,
                                          .entryCount = entries.size(),
                                          .entries = entries.data()};
    const uint32_t x = blit.tile % TilesPerRow * TileSize;
    const uint32_t y = blit.tile / TilesPerRow * TileSize;
    const std::array<uint32_t, 4> push{x, y, blit.level, blit.taps};
    pass.SetPipeline(blit.srgb ? pipelines.srgb : pipelines.linear);
    pass.SetBindGroup(0, webgpu::g_device.CreateBindGroup(&group));
    pass.SetViewport(static_cast<float>(x), static_cast<float>(y), TileSize, TileSize, 0.f, 1.f);
    pass.SetScissorRect(x, y, TileSize, TileSize);
    pass.SetImmediates(0, push.data(), ImmediateSize);
    pass.Draw(3);
    blitted.push_back(blit.tile);
  }
  pass.End();
  if (blitted.empty()) {
    return blitted;
  }
  const std::array<wgpu::BindGroupEntry, 2> meanEntries{{
      {.binding = 0, .textureView = page},
      {.binding = 1, .buffer = means.buffer, .offset = means.offset, .size = means.size},
  }};
  const wgpu::BindGroupDescriptor meanGroup{.label = "scene.atlas.mean",
                                            .layout = g_meanGroupLayout,
                                            .entryCount = meanEntries.size(),
                                            .entries = meanEntries.data()};
  const wgpu::ComputePassEncoder compute = encoder.BeginComputePass();
  compute.SetPipeline(pipelines.mean);
  compute.SetBindGroup(0, webgpu::g_device.CreateBindGroup(&meanGroup));
  for (const uint32_t tile : blitted) {
    const std::array<uint32_t, 4> push{tile, 0, 0, 0};
    compute.SetImmediates(0, push.data(), ImmediateSize);
    compute.DispatchWorkgroups(1);
  }
  compute.End();
  return blitted;
}

void encode_blits(detail::PassContext& ctx, const Local& local, const Snapshot& snapshot) {
  const BlitPipelines pipelines{
      .linear = shaders::render_pipeline(ctx.pipeline(0)),
      .srgb = shaders::render_pipeline(ctx.pipeline(1)),
      .mean = shaders::compute_pipeline(ctx.pipeline(2)),
  };
  const wgpu::TextureView page = ctx.view(TexRef{local.page});
  const BufferView means = ctx.buffer(BufRef{local.means});
  if (!pipelines.linear || !pipelines.srgb || !pipelines.mean || !page || !means.buffer) {
    return;
  }
  const std::vector<uint32_t> blitted =
      record_blits(ctx.encoder(), pipelines, page, means, snapshot, g_pageFresh.exchange(false));
  if (!blitted.empty()) {
    ack(blitted);
  }
}

void encode(detail::PassContext& ctx, const void*) {
  const auto& frame = ctx.stage_data().retained;
  const Snapshot* snapshot = frame != nullptr ? frame->atlas.get() : nullptr;
  if (snapshot == nullptr) {
    return;
  }
  Local local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const BufferView materials = ctx.buffer(BufRef{local.materials});
  if (materials.buffer && !snapshot->materials.empty()) {
    const uint64_t bytes = std::min<uint64_t>(snapshot->materials.size() * sizeof(GpuMaterial), materials.size);
    ctx.encoder().WriteBuffer(materials.buffer, materials.offset,
                              reinterpret_cast<const uint8_t*>(snapshot->materials.data()), bytes);
  }
  if (!snapshot->blits.empty()) {
    encode_blits(ctx, local, *snapshot);
  }
}
} // namespace

GpuMaterial pack_material(uint32_t albedoTile, uint32_t alphaTile, uint32_t wraps, uint32_t alphaChannel,
                          bool alphaUsesUv0, uint32_t flags, const gx::AlphaCompare& compare,
                          uint32_t registryId) noexcept {
  GpuMaterial out;
  out.words[0] = (albedoTile & 0xFFFFu) | (alphaTile & 0xFFFFu) << 16;
  out.words[1] = (wraps & 0xFFu) | (alphaChannel & 3u) << 8 | (alphaUsesUv0 ? 1u << 10 : 0u) | (flags & 0xFFFFu) << 16;
  out.words[2] = (flags & material_flag::AlphaTest) != 0 ? compare_word(compare) : compare_word({});
  out.words[3] = registryId;
  return out;
}

std::array<uint32_t, 3> pack_attributes(const DrawAttributes& attributes) noexcept {
  return {attributes.word, attributes.constColor, attributes.receiver};
}

bool supported_format(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::RGBA8UnormSrgb:
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::BGRA8UnormSrgb:
  case wgpu::TextureFormat::BC1RGBAUnorm:
  case wgpu::TextureFormat::BC1RGBAUnormSrgb:
  case wgpu::TextureFormat::BC2RGBAUnorm:
  case wgpu::TextureFormat::BC2RGBAUnormSrgb:
  case wgpu::TextureFormat::BC3RGBAUnorm:
  case wgpu::TextureFormat::BC3RGBAUnormSrgb:
  case wgpu::TextureFormat::BC4RUnorm:
  case wgpu::TextureFormat::BC4RSnorm:
  case wgpu::TextureFormat::BC5RGUnorm:
  case wgpu::TextureFormat::BC5RGSnorm:
  case wgpu::TextureFormat::BC6HRGBUfloat:
  case wgpu::TextureFormat::BC6HRGBFloat:
  case wgpu::TextureFormat::BC7RGBAUnorm:
  case wgpu::TextureFormat::BC7RGBAUnormSrgb:
    return true;
  default:
    return false;
  }
}

Blit blit_for(uint32_t tile, std::shared_ptr<gfx::TextureRef> texture) noexcept {
  Blit out{.tile = tile};
  if (!texture) {
    return out;
  }
  const uint32_t size = std::max({texture->size.width, texture->size.height, 1u});
  const uint32_t levels = std::max(texture->mipCount, 1u);
  uint32_t level = 0;
  while (level + 1 < levels && (size >> (level + 1)) >= TileSize) {
    ++level;
  }
  out.level = level;
  out.taps = std::clamp((size >> level) / TileSize, 1u, MaxTaps);
  out.srgb = srgb_format(texture->format);
  out.texture = std::move(texture);
  return out;
}

bool active() noexcept { return g_active; }

uint64_t material_tuple(const gx::GXState& state, const map_pick::MapSelection& maps) noexcept {
  Hasher hasher;
  const auto pick = [&](const map_pick::MapPick& p) {
    hasher.update(p.map);
    hasher.update(p.coord);
    hasher.update(p.texcoordSource);
    hasher.update(p.channel);
    if (p.valid() && p.map < gx::MaxTextures) {
      const GXTexObj_& obj = state.loadedTextures[p.map];
      hasher.update(obj.texObjId);
      hasher.update(obj.texDataVersion);
      hasher.update(obj.mode0);
      hasher.update(obj.flags);
      hasher.update(reinterpret_cast<uintptr_t>(obj.data));
      if (gx::is_palette_format(obj.format()) && static_cast<size_t>(obj.tlut) < state.loadedTluts.size()) {
        hasher.update(state.loadedTluts[obj.tlut].tlutObjId);
        hasher.update(state.loadedTluts[obj.tlut].tlutDataVersion);
      }
    }
  };
  pick(maps.albedo);
  pick(maps.alpha);
  hasher.update(maps.alphaTest);
  hasher.update(maps.alphaTest ? compare_word(state.alphaCompare) : 0u);
  return hasher.digest() | 1u;
}

uint32_t material_for(const gx::GXState& state, const map_pick::MapSelection& maps, bool visible) noexcept {
  if (!g_active || !g_allocated) {
    return NoMaterial;
  }
  MaterialKey key{};
  MaterialInfo info{.valid = true};
  key.flags = maps.alphaTest ? material_flag::AlphaTest : 0u;
  if (maps.albedo.valid()) {
    const Resolved r = resolve(state, maps.albedo, visible);
    key.albedoTile = r.tile;
    key.wraps |= r.wraps;
    key.registry = r.registry;
    key.flags |= r.missing ? material_flag::AlbedoMissing : 0u;
    info.albedo = r.obj;
    info.albedoTexture = r.texture;
    if (!maps.albedo.texcoordSource) {
      key.flags |= material_flag::AlbedoNoUv;
      ++g_stats.noUv;
    }
  }
  if (maps.alpha.valid()) {
    const Resolved r = resolve(state, maps.alpha, visible);
    key.alphaTile = r.tile;
    key.wraps |= r.wraps << 4;
    key.channel = maps.alpha.channel & 3u;
    key.alphaUsesUv0 = maps.albedo.valid() && maps.alpha.coord == maps.albedo.coord;
    key.flags |= r.missing ? material_flag::AlphaMissing : 0u;
    info.alpha = r.obj;
    info.alphaTexture = r.texture;
    if (!maps.alpha.texcoordSource) {
      key.flags |= material_flag::AlphaNoUv;
      ++g_stats.noUv;
    }
  }
  key.compare = maps.alphaTest ? compare_word(state.alphaCompare) : 0u;
  uint32_t id = NoMaterial;
  if (const auto it = g_materialIndex.find(key); it != g_materialIndex.end()) {
    id = it->second;
  } else {
    id = allocate_material();
    if (id == NoMaterial) {
      ++g_stats.noMaterial;
      return NoMaterial;
    }
    MaterialEntry& m = g_materials[id];
    m = {.key = key, .compare = state.alphaCompare, .live = true, .info = std::move(info)};
    ref_tile(key.albedoTile);
    ref_tile(key.alphaTile);
    g_materialIndex[key] = id;
  }
  MaterialEntry& m = g_materials[id];
  m.released = false;
  ++m.refs;
  return id;
}

void release(uint32_t id) noexcept {
  if (id >= g_materials.size() || !g_materials[id].live || g_materials[id].refs == 0) {
    return;
  }
  MaterialEntry& m = g_materials[id];
  if (--m.refs == 0) {
    m.releasedLatch = g_latch;
    if (!std::exchange(m.released, true)) {
      g_releasedMaterials.push_back(id);
    }
  }
}

DrawAttributes draw_attributes(const DrawAttributes& last, uint64_t& tuple, const gx::PipelineConfig* config,
                               const gx::ShaderInfo* info, const DrawClass* cls, const gx::GXState& state) noexcept {
  DrawAttributes out = last;
  const gx::ColorChannelConfig& chan = state.colorChannelConfig[0];
  out.constColor = chan.matSrc == GX_SRC_REG ? pack_rgba8(state.colorChannelState[0].matColor) : 0xFFFFFFFFu;
  ++g_stats.attributeCaptures;
  if (config == nullptr || info == nullptr || cls == nullptr) {
    return out;
  }
  Hasher hasher;
  hasher.update(config->cullMode);
  hasher.update(static_cast<bool>(config->shaderConfig.alphaCompare));
  hasher.update(cls->key.materialClass);
  hasher.update(cls->key.categories);
  hasher.update(cls->key.flags & (object_flag::DoubleSided | object_flag::BakedVertexLight));
  hasher.update(material::classes::epoch());
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    if (info->sampledTextures.test(i)) {
      hasher.update(state.textures[i].material.id);
    }
  }
  const uint64_t next = hasher.digest() | 1u;
  // The word follows the instance's own draw, as M10 requires, so materials never carry another draw's class or cull
  if (next != tuple || (out.receiver & attribute_flag::WordKnown << 8) == 0) {
    out.word = gx_ext::unlit::draw_material(*config, *info, *cls).word;
    out.receiver = receiver_category::compress(cls->key.categories) |
                   (attribute_flag::WordKnown | (config->cullMode == GX_CULL_NONE ? attribute_flag::CullNone : 0u))
                       << 8;
    tuple = next;
    ++g_stats.attributeRecomputes;
  }
  return out;
}

std::shared_ptr<const Snapshot> snapshot() noexcept {
  if (!g_active || !g_allocated) {
    return nullptr;
  }
  auto out = std::make_shared<Snapshot>();
  out->materials.resize(g_used);
  for (uint32_t id = 0; id < g_used; ++id) {
    if (g_materials[id].live) {
      out->materials[id] = pack_entry(g_materials[id]);
    }
  }
  while (!g_queue.empty() && out->blits.size() < BlitsPerFrame) {
    const uint32_t id = g_queue.front();
    g_queue.pop_front();
    TileEntry& tile = g_tiles[id];
    tile.queued = false;
    if (!tile.live || tile.ready) {
      continue;
    }
    std::shared_ptr<gfx::TextureRef> texture = tile.texture.lock();
    if (!texture) {
      continue;
    }
    tile.sentLatch = g_latch;
    out->blits.push_back(blit_for(id, std::move(texture)));
    ++g_stats.blits;
  }
  return out;
}

MaterialInfo info(uint32_t id) noexcept {
  return id < g_materials.size() && g_materials[id].live ? g_materials[id].info : MaterialInfo{};
}

void initialize() {
  if (!create_layouts()) {
    return;
  }
  g_pageHandle = detail::create_persistent(PageName, detail::PersistentKind::Texture);
  g_materialsHandle = detail::create_persistent(MaterialsName, detail::PersistentKind::Buffer);
  g_meansHandle = detail::create_persistent(MeansName, detail::PersistentKind::Buffer);
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
}

void shutdown() {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  latch_resources(false);
  for (detail::PersistentHandle* handle : {&g_pageHandle, &g_materialsHandle, &g_meansHandle}) {
    if (handle->id != 0) {
      detail::destroy_persistent(*handle);
      *handle = {};
    }
  }
  g_sampler = {};
  g_meanLayout = {};
  g_blitLayout = {};
  g_meanGroupLayout = {};
  g_blitGroupLayout = {};
  set_active(false);
}

void set_active(bool active) noexcept {
  if (active == g_allocated) {
    g_active = active;
    return;
  }
  g_allocated = active;
  g_active = active;
  clear_state();
  if (!active) {
    g_materials = {};
    g_tiles = {};
    g_freeTiles = {};
  }
  publish();
}

void latch_resources(bool active) noexcept {
  const wgpu::Device& device = webgpu::g_device;
  if (!active || !device || g_pageHandle.id == 0) {
    if (g_page) {
      detail::set_persistent_texture(g_pageHandle, {}, {});
      detail::set_persistent_buffer(g_materialsHandle, {}, 0, 0);
      detail::set_persistent_buffer(g_meansHandle, {}, 0, 0);
    }
    g_page = {};
    g_materialsBuffer = {};
    g_meansBuffer = {};
    return;
  }
  if (g_page) {
    return;
  }
  const wgpu::TextureDescriptor page{
      .label = "scene.atlas_page",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .size = {PageSize, PageSize, 1},
      .format = PageFormat,
  };
  g_page = device.CreateTexture(&page);
  const wgpu::BufferDescriptor materials{.label = "scene.materials",
                                         .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                         .size = MaterialsBytes};
  g_materialsBuffer = device.CreateBuffer(&materials);
  const wgpu::BufferDescriptor means{.label = "scene.atlas_means",
                                     .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                     .size = MeansBytes};
  g_meansBuffer = device.CreateBuffer(&means);
  detail::set_persistent_texture(g_pageHandle, g_page, g_page.CreateView());
  detail::set_persistent_buffer(g_materialsHandle, g_materialsBuffer, 0, MaterialsBytes);
  detail::set_persistent_buffer(g_meansHandle, g_meansBuffer, 0, MeansBytes);
  g_pageFresh.store(true, std::memory_order_relaxed);
}

void latch() noexcept {
  if (!g_allocated) {
    return;
  }
  ++g_latch;
  std::vector<uint32_t> acks;
  {
    std::lock_guard lock{g_ackMutex};
    acks.swap(g_acks);
  }
  for (const uint32_t id : acks) {
    if (id < g_tiles.size() && g_tiles[id].live && g_tiles[id].sentLatch != Never) {
      g_tiles[id].ready = true;
      g_tiles[id].sentLatch = Never;
      ++g_stats.acks;
    }
  }
  for (uint32_t id = 0; id < g_tiles.size(); ++id) {
    TileEntry& tile = g_tiles[id];
    if (!tile.live) {
      continue;
    }
    if (!tile.ready && tile.sentLatch != Never && g_latch - tile.sentLatch >= BlitRetryLatches) {
      tile.sentLatch = Never;
      tile.queued = true;
      g_queue.push_back(id);
      ++g_stats.retries;
    }
    if (tile.refs == 0 && tile.sentLatch == Never && tile.releasedLatch < g_latch && tile.texture.expired()) {
      free_tile(id);
    }
  }
  std::erase_if(g_releasedMaterials, [](uint32_t id) {
    MaterialEntry& m = g_materials[id];
    if (!m.live || !m.released) {
      return true;
    }
    if (m.refs != 0) {
      m.released = false;
      return true;
    }
    // An instance freed this latch may still name the id in the frame in flight, so it recycles one latch later
    if (m.releasedLatch >= g_latch - 1) {
      return false;
    }
    free_material(id);
    return true;
  });
  publish();
}

void reset() noexcept {
  if (g_allocated) {
    clear_state();
  }
  g_stats = {};
  publish();
}

Stats published_stats() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_published;
}

detail::PersistentHandle page_handle() noexcept { return g_pageHandle; }

detail::PersistentHandle materials_handle() noexcept { return g_materialsHandle; }

detail::PersistentHandle means_handle() noexcept { return g_meansHandle; }

TexDesc page_desc() noexcept {
  return {.format = PageFormat, .sizeClass = SizeClass::Absolute, .width = PageSize, .height = PageSize};
}

void ack(std::span<const uint32_t> tiles) noexcept {
  std::lock_guard lock{g_ackMutex};
  g_acks.insert(g_acks.end(), tiles.begin(), tiles.end());
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{PassName},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .cls = detail::PassClass::Required,
      .order = 1,
      .runsWithoutMarker = true,
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = pass_pipelines,
  };
}

namespace testing {
void reset() noexcept {
  set_active(false);
  g_latch = 1;
  g_stats = {};
  g_resolve = &default_resolve;
  publish();
}

void set_active(bool active) noexcept { atlas::set_active(active); }

void set_resolver(ResolveFn resolve) noexcept { g_resolve = resolve != nullptr ? resolve : &default_resolve; }

uint32_t refs(uint32_t material) noexcept {
  return material < g_materials.size() && g_materials[material].live ? g_materials[material].refs : 0;
}

Tile tile(uint32_t tile) noexcept {
  if (tile >= g_tiles.size()) {
    return {};
  }
  return {.live = g_tiles[tile].live, .ready = g_tiles[tile].ready, .refs = g_tiles[tile].refs};
}

bool encode_blits(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& page, const BufferView& means,
                  const Snapshot& snapshot) {
  const shaders::PipelineId linear = shaders::request_render(blit_desc(false));
  const shaders::PipelineId srgb = shaders::request_render(blit_desc(true));
  const shaders::PipelineId mean = shaders::request_compute(mean_desc());
  for (const shaders::PipelineId id : {linear, srgb, mean}) {
    if (shaders::wait(id) != shaders::PipelineStatus::Ready) {
      return false;
    }
  }
  const BlitPipelines pipelines{
      .linear = shaders::render_pipeline(linear),
      .srgb = shaders::render_pipeline(srgb),
      .mean = shaders::compute_pipeline(mean),
  };
  return !record_blits(encoder, pipelines, page, means, snapshot, true).empty();
}

uint32_t material_tile(uint32_t material, bool alpha) noexcept {
  if (material >= g_materials.size() || !g_materials[material].live) {
    return NoTile;
  }
  return alpha ? g_materials[material].key.alphaTile : g_materials[material].key.albedoTile;
}
} // namespace testing
} // namespace aurora::render::scene::atlas

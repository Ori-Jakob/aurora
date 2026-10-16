#include "graph_bind.hpp"

#include "frame_state.hpp"
#include "graph_api.hpp"
#include "graph_pool.hpp"
#include "graph_slots.hpp"
#include "shaders.hpp"
#include "stage_data.hpp"

#include "../gfx/hash.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resources.hpp"
#include "../gx/gx.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aurora::render::detail {
static_assert(sizeof(StageBindings::Params::bytes) == MaxPassParamsSize);
static_assert(std::is_same_v<decltype(StageBindings::Params::node), NodeId> && InvalidNode == UINT16_MAX);

struct BoundUse {
  const PhysicalTexture* physical = nullptr;
  wgpu::Texture texture;
  wgpu::TextureView view;
  wgpu::TextureView resolveView;
  wgpu::Buffer buffer;
  uint64_t offset = 0;
  uint64_t size = 0;
  wgpu::Extent3D extent{};
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint32_t samples = 1;
  PersistentHandle persistent;
  uint32_t generation = 0;
  BindIdentity identity;
};

struct BoundPass {
  const CompiledGraph* graph = nullptr;
  NodeId node = InvalidNode;
  const BindingPlan* plan = nullptr;
  std::span<const CompiledUse> uses;
  std::vector<BoundUse> bound;
  std::shared_ptr<const BindingLayout> layout;
  wgpu::BindGroupLayout frameLayout;
  wgpu::BindGroup frameGroup;
  wgpu::BindGroup passGroup;
  uint32_t constantsOffset = 0;
  uint32_t paramsOffset = 0;
  std::span<const std::byte> params;
};

namespace {
constexpr Module Log{"aurora::render::graph"};
constexpr uint64_t RetainFrames = 32;
constexpr uint64_t SweepPeriod = 16;
constexpr uint32_t PersistentIdLimit = 1u << 31;

constexpr bool is_attachment(Access access) noexcept {
  return access == Access::ColorWrite || access == Access::ColorRMW || access == Access::DepthRead ||
         access == Access::DepthWrite;
}

constexpr bool is_storage_texture(BindingType type) noexcept {
  return type == BindingType::ReadOnlyStorageTexture || type == BindingType::WriteOnlyStorageTexture ||
         type == BindingType::ReadWriteStorageTexture;
}

struct LayoutRecord {
  bool manual = false;
  uint64_t hash = 0;
  std::vector<BindingSlot> slots;
  std::shared_ptr<const BindingLayout> layout;
};

// Latch thread and render worker; created under the mutex and replaced only at shutdown
struct SharedObjects {
  std::mutex mutex;
  wgpu::BindGroupLayout frameLayout;
  wgpu::BindGroupLayout emptyLayout;
  std::array<wgpu::Sampler, 4> samplers;
  wgpu::BindGroup frameGroup;
  std::vector<LayoutRecord> layouts;
  uint64_t nextSerial = 1;
};

SharedObjects g_shared;

bool same_shape(const BindingSlot& a, const BindingSlot& b) noexcept {
  return a.binding == b.binding && a.type == b.type && a.visibility == b.visibility &&
         a.viewDimension == b.viewDimension && a.sampleType == b.sampleType && a.storageFormat == b.storageFormat &&
         a.multisampled == b.multisampled;
}

// Requires g_shared.mutex held.
void ensure_frame_layout_locked() {
  if (g_shared.frameLayout || !webgpu::g_device) {
    return;
  }
  const auto entries = frame_layout_entries();
  const wgpu::BindGroupLayoutDescriptor frameDescriptor{
      .label = "graph frame",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_shared.frameLayout = webgpu::g_device.CreateBindGroupLayout(&frameDescriptor);
  const wgpu::BindGroupLayoutDescriptor emptyDescriptor{.label = "graph empty"};
  g_shared.emptyLayout = webgpu::g_device.CreateBindGroupLayout(&emptyDescriptor);
}

wgpu::Sampler create_sampler(const char* label, wgpu::FilterMode filter, wgpu::AddressMode address,
                             wgpu::CompareFunction compare = wgpu::CompareFunction::Undefined) {
  const wgpu::SamplerDescriptor descriptor{
      .label = label,
      .addressModeU = address,
      .addressModeV = address,
      .addressModeW = address,
      .magFilter = filter,
      .minFilter = filter,
      .mipmapFilter =
          filter == wgpu::FilterMode::Linear ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest,
      .compare = compare,
  };
  return webgpu::g_device.CreateSampler(&descriptor);
}

// Requires g_shared.mutex held.
void ensure_frame_group_locked() {
  ensure_frame_layout_locked();
  const wgpu::Buffer& uniforms = gfx::detail::resources().uniformBuffer;
  if (g_shared.frameGroup || !g_shared.frameLayout || !uniforms) {
    return;
  }
  using wgpu::AddressMode;
  using wgpu::FilterMode;
  auto& samplers = g_shared.samplers;
  samplers = {
      create_sampler("graph linear clamp", FilterMode::Linear, AddressMode::ClampToEdge),
      create_sampler("graph nearest clamp", FilterMode::Nearest, AddressMode::ClampToEdge),
      create_sampler("graph linear repeat", FilterMode::Linear, AddressMode::Repeat),
      create_sampler("graph depth compare", FilterMode::Linear, AddressMode::ClampToEdge,
                     gx::UseReversedZ ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual),
  };
  const auto binding = [](FrameBinding value) { return static_cast<uint32_t>(value); };
  const std::array entries{
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::Constants), .buffer = uniforms, .size = FrameUniformsSize},
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::Params), .buffer = uniforms, .size = MaxPassParamsSize},
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::LinearClamp), .sampler = samplers[0]},
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::NearestClamp), .sampler = samplers[1]},
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::LinearRepeat), .sampler = samplers[2]},
      wgpu::BindGroupEntry{.binding = binding(FrameBinding::DepthCompare), .sampler = samplers[3]},
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "graph frame",
      .layout = g_shared.frameLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_shared.frameGroup = webgpu::g_device.CreateBindGroup(&descriptor);
}

wgpu::BindGroupLayoutEntry layout_entry(const BindingSlot& slot) {
  wgpu::BindGroupLayoutEntry entry{.binding = slot.binding, .visibility = slot.visibility};
  switch (slot.type) {
  case BindingType::UniformBuffer:
    entry.buffer = {.type = wgpu::BufferBindingType::Uniform};
    break;
  case BindingType::ReadOnlyStorageBuffer:
    entry.buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage};
    break;
  case BindingType::StorageBuffer:
    entry.buffer = {.type = wgpu::BufferBindingType::Storage};
    break;
  case BindingType::SampledTexture:
    entry.texture = {
        .sampleType = slot.sampleType,
        .viewDimension = slot.viewDimension,
        .multisampled = slot.multisampled,
    };
    break;
  case BindingType::ReadOnlyStorageTexture:
  case BindingType::WriteOnlyStorageTexture:
  case BindingType::ReadWriteStorageTexture:
    entry.storageTexture = {
        .access = slot.type == BindingType::ReadOnlyStorageTexture    ? wgpu::StorageTextureAccess::ReadOnly
                  : slot.type == BindingType::WriteOnlyStorageTexture ? wgpu::StorageTextureAccess::WriteOnly
                                                                      : wgpu::StorageTextureAccess::ReadWrite,
        .format = slot.storageFormat,
        .viewDimension = slot.viewDimension,
    };
    break;
  default:
    break;
  }
  return entry;
}

std::shared_ptr<const BindingLayout> find_layout(const BindingPlan& plan) {
  std::lock_guard lock{g_shared.mutex};
  for (const LayoutRecord& record : g_shared.layouts) {
    if (record.hash == plan.layoutHash && record.manual == plan.manual &&
        std::ranges::equal(record.slots, plan.slots, same_shape)) {
      return record.layout;
    }
  }
  ensure_frame_layout_locked();
  if (!g_shared.frameLayout) {
    return nullptr;
  }
  auto layout = std::make_shared<BindingLayout>();
  layout->serial = g_shared.nextSerial++;
  if (plan.manual || plan.slots.empty()) {
    layout->passLayout = g_shared.emptyLayout;
  } else {
    std::vector<wgpu::BindGroupLayoutEntry> entries;
    entries.reserve(plan.slots.size());
    for (const BindingSlot& slot : plan.slots) {
      entries.push_back(layout_entry(slot));
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "graph pass",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    layout->passLayout = webgpu::g_device.CreateBindGroupLayout(&descriptor);
  }
  const std::array groups{g_shared.frameLayout, layout->passLayout};
  const wgpu::PipelineLayoutDescriptor descriptor{
      .label = "graph pass",
      .bindGroupLayoutCount = plan.manual ? size_t{1} : groups.size(),
      .bindGroupLayouts = groups.data(),
  };
  layout->pipelineLayout = webgpu::g_device.CreatePipelineLayout(&descriptor);
  g_shared.layouts.push_back({.manual = plan.manual, .hash = plan.layoutHash, .slots = plan.slots, .layout = layout});
  return layout;
}

struct PersistentView {
  uint32_t generation = 0;
  uint32_t mip = 0;
  uint32_t layer = 0;
  wgpu::TextureView view;
};

struct PersistentRecord {
  std::string name;
  PersistentKind kind = PersistentKind::Buffer;
  PersistentBufferSource source = nullptr;
  void* context = nullptr;
  PersistentBuffer buffer;
  PersistentTexture texture;
  std::vector<PersistentView> attachmentViews;
};

std::mutex g_persistentMutex;
std::unordered_map<uint32_t, PersistentRecord> g_persistent;
std::unordered_map<uint32_t, PersistentHandle> g_heapHandles;
uint32_t g_nextPersistentId = 1;

// Requires g_persistentMutex held.
PersistentHandle add_persistent_locked(PersistentRecord record) {
  CHECK(g_nextPersistentId < PersistentIdLimit, "render graph: persistent handles exhausted");
  if (g_nextPersistentId >= PersistentIdLimit) {
    return {};
  }
  const uint32_t id = g_nextPersistentId++;
  g_persistent.emplace(id, std::move(record));
  return {id};
}

// Requires g_persistentMutex held.
PersistentRecord* find_persistent_locked(PersistentHandle handle, PersistentKind kind) noexcept {
  const auto it = g_persistent.find(handle.id);
  return it != g_persistent.end() && it->second.kind == kind ? &it->second : nullptr;
}

constexpr uint32_t next_generation(uint32_t generation) noexcept {
  return generation == UINT32_MAX ? 1 : generation + 1;
}

bool heap_pool_source(void* context, PersistentBuffer& out) {
  const auto pool = static_cast<gpu_heap::PoolId>(reinterpret_cast<uintptr_t>(context));
  const gpu_heap::PoolBinding binding = gpu_heap::binding(pool);
  out = {.buffer = binding.buffer, .size = binding.size, .generation = binding.generation};
  return binding.buffer && binding.generation != 0;
}

uint32_t persistent_generation(uint32_t id) {
  PersistentBufferSource source = nullptr;
  void* context = nullptr;
  {
    std::lock_guard lock{g_persistentMutex};
    const auto it = g_persistent.find(id);
    if (it == g_persistent.end()) {
      return 0;
    }
    const PersistentRecord& record = it->second;
    if (record.kind == PersistentKind::Texture) {
      return record.texture.generation;
    }
    if (record.source == nullptr) {
      return record.buffer.generation;
    }
    source = record.source;
    context = record.context;
  }
  PersistentBuffer buffer;
  return source(context, buffer) ? buffer.generation : 0;
}

wgpu::TextureView persistent_attachment_view(PersistentHandle handle, uint32_t generation, uint32_t mip,
                                             uint32_t layer) {
  std::lock_guard lock{g_persistentMutex};
  PersistentRecord* record = find_persistent_locked(handle, PersistentKind::Texture);
  if (record == nullptr || record->texture.generation != generation || !record->texture.texture) {
    return {};
  }
  for (const PersistentView& cached : record->attachmentViews) {
    if (cached.generation == generation && cached.mip == mip && cached.layer == layer) {
      return cached.view;
    }
  }
  const wgpu::Texture& texture = record->texture.texture;
  const bool volume = texture.GetDimension() == wgpu::TextureDimension::e3D;
  const wgpu::TextureViewDescriptor descriptor{
      .dimension = volume ? wgpu::TextureViewDimension::e3D : wgpu::TextureViewDimension::e2D,
      .baseMipLevel = mip,
      .mipLevelCount = 1,
      .baseArrayLayer = volume ? 0 : layer,
      .arrayLayerCount = volume ? wgpu::kArrayLayerCountUndefined : 1,
      .aspect = wgpu::TextureAspect::All,
  };
  return record->attachmentViews
      .emplace_back(PersistentView{
          .generation = generation,
          .mip = mip,
          .layer = layer,
          .view = texture.CreateView(&descriptor),
      })
      .view;
}

std::array<std::atomic<GxResolveFn>, GxResourceCount> g_gxResolvers{};

bool resolve_gx_color(const StageData&, Access, GxBinding& out) {
  const webgpu::TextureWithSampler& efb = webgpu::g_frameBuffer;
  const uint32_t samples = std::max(webgpu::g_graphicsConfig.msaaSamples, 1u);
  const webgpu::TextureWithSampler& resolved = samples > 1 ? webgpu::g_frameBufferResolved : efb;
  if (!efb.texture || !resolved.texture) {
    return false;
  }
  out = {
      .texture = efb.texture,
      .sampleView = resolved.view,
      .attachmentView = efb.view,
      .resolveView = samples > 1 ? resolved.view : wgpu::TextureView{},
      .copyTexture = resolved.texture,
      .size = efb.size,
      .format = efb.format,
      .samples = samples,
  };
  return true;
}

bool resolve_gx_normal(const StageData&, Access access, GxBinding& out) {
  const webgpu::TextureWithSampler& normal = webgpu::g_normalBuffer;
  if (!webgpu::g_graphicsConfig.normalBuffer || !normal.texture) {
    return false;
  }
  if (access == Access::Sampled && !(normal.texture.GetUsage() & wgpu::TextureUsage::TextureBinding)) {
    return false;
  }
  out = {
      .texture = normal.texture,
      .sampleView = normal.view,
      .attachmentView = normal.view,
      .copyTexture = normal.texture,
      .size = normal.size,
      .format = normal.texture.GetFormat(),
      .samples = normal.texture.GetSampleCount(),
  };
  return true;
}

GxResolveFn gx_resolver(GxResource resource) noexcept {
  const auto index = static_cast<size_t>(resource);
  if (index >= GxResourceCount) {
    return nullptr;
  }
  if (const GxResolveFn registered = g_gxResolvers[index].load(std::memory_order_acquire)) {
    return registered;
  }
  switch (resource) {
  case GxResource::Color:
    return &resolve_gx_color;
  case GxResource::Normal:
    return &resolve_gx_normal;
  default:
    return nullptr;
  }
}

struct Dummy {
  DummyKey key;
  uint64_t serial = 0;
  wgpu::Texture texture;
  wgpu::TextureView view;
  wgpu::Buffer buffer;
  wgpu::Extent3D extent{};
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};

struct CacheEntry {
  uint64_t layoutSerial = 0;
  std::vector<BindIdentity> identities;
  wgpu::BindGroup group;
  uint64_t lastUsedFrame = 0;
};

// Render worker only
struct WorkerState {
  std::unordered_multimap<uint64_t, CacheEntry> cache;
  std::unordered_map<uint32_t, uint32_t> generations;
  std::vector<Dummy> dummies;
  std::unordered_map<NodeId, std::vector<BindIdentity>> lastIdentities;
  std::vector<testing::AttachmentOps> lastAttachmentOps;
  std::unordered_set<std::string> logged;
  bool recount = false;
  uint64_t releasedSeen = 0;
};

WorkerState g_worker;

struct PublishedStats {
  std::atomic_uint32_t entries{0};
  std::atomic_uint32_t staleEntries{0};
  std::atomic_uint64_t created{0};
  std::atomic_uint64_t hits{0};
  std::atomic_uint64_t evicted{0};
};

PublishedStats g_stats;
std::atomic_bool g_recordDetails{false};

// Recording thread only; every stage of a frame binds the same FrameUniforms push
struct FrameConstantsPush {
  uint64_t frameId = 0;
  uint32_t offset = 0;
  bool pushed = false;
};

FrameConstantsPush g_frameConstants;

// Render worker only; every encode of a frame comes before that frame's after-frame point
std::vector<gfx::AfterSubmitCallback> g_afterFrame;

void reset_worker_state() {
  g_worker = {};
  g_afterFrame.clear();
  g_stats.entries.store(0, std::memory_order_relaxed);
  g_stats.staleEntries.store(0, std::memory_order_relaxed);
  g_stats.created.store(0, std::memory_order_relaxed);
  g_stats.hits.store(0, std::memory_order_relaxed);
  g_stats.evicted.store(0, std::memory_order_relaxed);
  g_frameConstants = {};
}

void log_once(const std::string& key, const std::string& message) {
  if (g_worker.logged.insert(key).second) {
    Log.warn("{}", message);
  }
}

bool references_generation(const CacheEntry& entry, uint32_t id, uint32_t generation) noexcept {
  const uint64_t value = uint64_t{id} << 32 | generation;
  return std::ranges::any_of(entry.identities, [&](const BindIdentity& identity) {
    return identity.kind == IdentityKind::Persistent && identity.value == value;
  });
}

uint32_t count_stale() {
  if (g_worker.cache.empty()) {
    return 0;
  }
  const std::vector<uint64_t> live = transient_pool().serials();
  const auto stale = [&](const BindIdentity& identity) {
    if (identity.kind == IdentityKind::Transient || identity.kind == IdentityKind::History) {
      return !std::ranges::binary_search(live, identity.value);
    }
    if (identity.kind != IdentityKind::Persistent) {
      return false;
    }
    const auto it = g_worker.generations.find(static_cast<uint32_t>(identity.value >> 32));
    return it == g_worker.generations.end() || it->second != static_cast<uint32_t>(identity.value);
  };
  uint32_t count = 0;
  for (const auto& [key, entry] : g_worker.cache) {
    count += std::ranges::any_of(entry.identities, stale) ? 1 : 0;
  }
  return count;
}

void publish_cache() {
  WorkerState& worker = g_worker;
  g_stats.entries.store(static_cast<uint32_t>(worker.cache.size()), std::memory_order_relaxed);
  // A release the listener missed still moves the pool's count, so it forces a recount as well
  const uint64_t released = transient_pool().stats().released;
  if (!worker.recount && released == worker.releasedSeen) {
    return;
  }
  worker.recount = false;
  worker.releasedSeen = released;
  g_stats.staleEntries.store(count_stale(), std::memory_order_relaxed);
}

template <typename Predicate>
size_t evict_if(Predicate predicate) {
  const size_t evicted = std::erase_if(g_worker.cache, [&](const auto& item) { return predicate(item.second); });
  g_stats.evicted.fetch_add(evicted, std::memory_order_relaxed);
  g_worker.recount = g_worker.recount || evicted != 0;
  return evicted;
}

void note_generation(uint32_t id, uint32_t generation) {
  const auto [it, inserted] = g_worker.generations.try_emplace(id, generation);
  if (inserted || it->second == generation) {
    return;
  }
  const uint32_t old = std::exchange(it->second, generation);
  g_worker.recount = true;
  evict_if([&](const CacheEntry& entry) { return references_generation(entry, id, old); });
  publish_cache();
}

wgpu::TextureFormat dummy_format(const DummyKey& key) noexcept {
  if (key.storageFormat != wgpu::TextureFormat::Undefined) {
    return key.storageFormat;
  }
  switch (key.sampleType) {
  case wgpu::TextureSampleType::Uint:
    return wgpu::TextureFormat::RGBA32Uint;
  case wgpu::TextureSampleType::Sint:
    return wgpu::TextureFormat::RGBA32Sint;
  case wgpu::TextureSampleType::Depth:
    return wgpu::TextureFormat::Depth16Unorm;
  default:
    return wgpu::TextureFormat::RGBA8Unorm;
  }
}

Dummy create_dummy(const DummyKey& key) {
  Dummy dummy{.key = key, .serial = next_resource_serial()};
  const bool white = key.color == DummyColor::White;
  if (key.buffer) {
    const uint64_t size = std::max<uint64_t>(key.bufferSize, 16);
    const wgpu::BufferDescriptor descriptor{
        .label = "graph dummy",
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = size,
        .mappedAtCreation = true,
    };
    dummy.buffer = webgpu::g_device.CreateBuffer(&descriptor);
    if (void* data = dummy.buffer.GetMappedRange(0, size)) {
      const uint32_t word = white ? 0x3F800000u : 0;
      std::fill_n(static_cast<uint32_t*>(data), size / sizeof(uint32_t), word);
    }
    dummy.buffer.Unmap();
    return dummy;
  }
  const bool storage = key.storageFormat != wgpu::TextureFormat::Undefined;
  const bool cube =
      key.dimension == wgpu::TextureViewDimension::Cube || key.dimension == wgpu::TextureViewDimension::CubeArray;
  dummy.format = dummy_format(key);
  dummy.extent = {1, 1, cube ? 6u : 1u};
  // Compat textures bind with one view dimension, fixed at creation
  const wgpu::TextureBindingViewDimension bindingView{wgpu::TextureBindingViewDimension::Init{
      .textureBindingViewDimension = key.dimension,
  }};
  const wgpu::TextureDescriptor descriptor{
      .nextInChain = webgpu::g_hasCoreFeatures ? nullptr : &bindingView,
      .label = "graph dummy",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst |
               (storage ? wgpu::TextureUsage::StorageBinding : wgpu::TextureUsage::None),
      .dimension =
          key.dimension == wgpu::TextureViewDimension::e3D ? wgpu::TextureDimension::e3D : wgpu::TextureDimension::e2D,
      .size = dummy.extent,
      .format = dummy.format,
  };
  dummy.texture = webgpu::g_device.CreateTexture(&descriptor);
  const auto info = texture_format_info(dummy.format);
  const uint32_t texel = info ? info->bytesPerBlock : 4;
  const std::vector<uint8_t> texels(size_t{texel} * dummy.extent.depthOrArrayLayers, white ? 0xFF : 0);
  const wgpu::TexelCopyTextureInfo destination{.texture = dummy.texture};
  const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = texel, .rowsPerImage = 1};
  webgpu::g_queue.WriteTexture(&destination, texels.data(), texels.size(), &layout, &dummy.extent);
  const wgpu::TextureViewDescriptor viewDescriptor{.label = "graph dummy", .dimension = key.dimension};
  dummy.view = dummy.texture.CreateView(&viewDescriptor);
  return dummy;
}

void bind_dummy(const DummyKey& key, BoundUse& out) {
  auto it = std::ranges::find(g_worker.dummies, key, &Dummy::key);
  if (it == g_worker.dummies.end()) {
    it = g_worker.dummies.insert(g_worker.dummies.end(), create_dummy(key));
  }
  out = {
      .texture = it->texture,
      .view = it->view,
      .buffer = it->buffer,
      .size = key.buffer ? it->buffer.GetSize() : 0,
      .extent = it->extent,
      .format = it->format,
      .identity = {.kind = IdentityKind::Dummy, .value = it->serial},
  };
}

void bind_physical(const PhysicalTexture& physical, IdentityKind kind, BoundUse& out) {
  out.physical = &physical;
  out.texture = physical.texture;
  out.extent = {physical.key.width, physical.key.height, physical.key.depthOrLayers};
  out.format = physical.key.format;
  out.identity = {.kind = kind, .value = physical.serial};
}

enum class Resolution : uint8_t { Bound, Missing, Skip };

Resolution resolve_persistent_use(const CompiledNode& node, const CompiledResource& resource, BoundUse& out) {
  const uint32_t id = resource.handle.id;
  if (resource.buffer) {
    PersistentBuffer buffer;
    if (!resolve_persistent(resource.handle, buffer)) {
      return Resolution::Missing;
    }
    note_generation(id, buffer.generation);
    out.buffer = buffer.buffer;
    out.offset = buffer.offset;
    out.size = buffer.size;
    out.identity = {.kind = IdentityKind::Persistent, .value = uint64_t{id} << 32 | buffer.generation};
    return Resolution::Bound;
  }
  PersistentTexture texture;
  if (!resolve_persistent(resource.handle, texture)) {
    return Resolution::Missing;
  }
  const wgpu::TextureFormat format = texture.texture.GetFormat();
  if (format != resource.tex.format) {
    log_once(fmt::format("{}@{}@format", node.name, resource.name),
             fmt::format("render graph: '{}' imports '{}' as format {}, but the texture has format {}; skipped",
                         node.name, resource.name, static_cast<uint32_t>(resource.tex.format),
                         static_cast<uint32_t>(format)));
    return Resolution::Skip;
  }
  note_generation(id, texture.generation);
  out.texture = texture.texture;
  out.view = texture.view;
  out.extent = {texture.texture.GetWidth(), texture.texture.GetHeight(), texture.texture.GetDepthOrArrayLayers()};
  out.format = format;
  out.persistent = resource.handle;
  out.generation = texture.generation;
  out.identity = {.kind = IdentityKind::Persistent, .value = uint64_t{id} << 32 | texture.generation};
  return Resolution::Bound;
}

Resolution resolve_use(const StageData& stage, const CompiledGraph& graph, const CompiledNode& node,
                       const CompiledUse& use, BoundUse& out) {
  const CompiledResource& resource = graph.resources[use.resource];
  const TransientPool& pool = transient_pool();
  const bool pooled = pool.materialized_graph() == &graph;
  switch (resource.kind) {
  case ResourceKind::Transient: {
    const SlotId slot =
        use.resource < graph.slots.resourceSlot.size() ? graph.slots.resourceSlot[use.resource] : InvalidSlot;
    if (!pooled || slot == InvalidSlot) {
      return Resolution::Missing;
    }
    if (!resource.buffer) {
      bind_physical(pool.texture(slot), IdentityKind::Transient, out);
      return out.texture ? Resolution::Bound : Resolution::Missing;
    }
    const PhysicalBuffer& physical = pool.buffer(slot);
    out.buffer = physical.buffer;
    out.size = resource.bufferSize;
    // Disjoint transients can share one buffer and bind different sizes of it
    out.identity = {
        .kind = IdentityKind::Transient,
        .selector = static_cast<uint32_t>(resource.bufferSize),
        .value = physical.serial,
    };
    return out.buffer ? Resolution::Bound : Resolution::Missing;
  }
  case ResourceKind::History: {
    const uint16_t history = resource.historySlot;
    // An owner scheduled last frame may still have been skipped, and then prev holds nothing it wrote
    if (!pooled || history == InvalidResource || (use.role == UseRole::Prev && !pool.history_fresh(history))) {
      return Resolution::Missing;
    }
    const PhysicalTexture& physical =
        use.role == UseRole::Prev ? pool.history_prev(history) : pool.history_write(history);
    bind_physical(physical, IdentityKind::History, out);
    return out.texture ? Resolution::Bound : Resolution::Missing;
  }
  case ResourceKind::Persistent:
    return resolve_persistent_use(node, resource, out);
  case ResourceKind::Gx: {
    const GxResolveFn resolve = gx_resolver(resource.gx);
    GxBinding binding;
    if (resolve == nullptr || !resolve(stage, use.access, binding)) {
      return Resolution::Missing;
    }
    const bool attachment = is_attachment(use.access);
    out.texture = attachment || !binding.copyTexture ? binding.texture : binding.copyTexture;
    out.view = attachment ? binding.attachmentView : binding.sampleView;
    out.resolveView = attachment ? binding.resolveView : wgpu::TextureView{};
    out.extent = binding.size;
    out.format = binding.format;
    out.samples = attachment ? binding.samples : 1;
    out.identity = {.kind = IdentityKind::Gx, .value = reinterpret_cast<uintptr_t>(out.view.Get())};
    return Resolution::Bound;
  }
  default:
    return Resolution::Missing;
  }
}

wgpu::BindGroupEntry group_entry(const BindingSlot& slot, const BoundUse& bound) {
  wgpu::BindGroupEntry entry{.binding = slot.binding};
  switch (slot.type) {
  case BindingType::UniformBuffer:
  case BindingType::ReadOnlyStorageBuffer:
  case BindingType::StorageBuffer:
    entry.buffer = bound.buffer;
    entry.offset = bound.offset;
    entry.size = bound.size;
    break;
  case BindingType::SampledTexture:
    entry.textureView = bound.physical != nullptr ? bound.physical->binding_view(slot.viewDimension) : bound.view;
    break;
  default:
    // A storage binding views a single mip, and the plan has no mip to choose, so it is mip 0
    entry.textureView = bound.physical != nullptr ? bound.physical->mip_view(0, slot.viewDimension) : bound.view;
    break;
  }
  return entry;
}

wgpu::BindGroup fetch_group(const BoundPass& pass, const std::vector<BindIdentity>& identities, uint64_t frameId) {
  const BindingLayout& layout = *pass.layout;
  const uint64_t key = bind_group_key(layout.serial, identities);
  const auto [first, last] = g_worker.cache.equal_range(key);
  for (auto it = first; it != last; ++it) {
    CacheEntry& entry = it->second;
    if (entry.layoutSerial == layout.serial && entry.identities == identities) {
      entry.lastUsedFrame = frameId;
      g_stats.hits.fetch_add(1, std::memory_order_relaxed);
      return entry.group;
    }
  }
  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(pass.plan->slots.size());
  for (const BindingSlot& slot : pass.plan->slots) {
    entries.push_back(group_entry(slot, pass.bound[slot.use]));
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = pass.graph->nodes[pass.node].name.c_str(),
      .layout = layout.passLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  wgpu::BindGroup group = webgpu::g_device.CreateBindGroup(&descriptor);
  g_worker.cache.emplace(key, CacheEntry{
                                  .layoutSerial = layout.serial,
                                  .identities = identities,
                                  .group = group,
                                  .lastUsedFrame = frameId,
                              });
  g_stats.created.fetch_add(1, std::memory_order_relaxed);
  g_worker.recount = true;
  publish_cache();
  return group;
}

bool bind_pass(const StageData& stage, BoundPass& pass) {
  const CompiledGraph& graph = *pass.graph;
  const CompiledNode& node = graph.nodes[pass.node];
  pass.bound.resize(pass.uses.size());
  for (size_t i = 0; i < pass.uses.size(); ++i) {
    const CompiledUse& use = pass.uses[i];
    BoundUse& out = pass.bound[i];
    if (use.hasFallback && stage.dummy_bound(use.stageUse)) {
      bind_dummy(use.fallback, out);
      continue;
    }
    const Resolution resolution = resolve_use(stage, graph, node, use, out);
    if (resolution == Resolution::Bound) {
      continue;
    }
    if (resolution == Resolution::Missing && use.hasFallback) {
      bind_dummy(use.fallback, out);
      continue;
    }
    if (resolution == Resolution::Missing) {
      const std::string& name = graph.resources[use.resource].name;
      const bool imported = use.role == UseRole::Import;
      log_once(fmt::format("{}@{}", node.name, name),
               imported
                   ? fmt::format("render graph: '{}' imports '{}', which has no resource yet; skipped", node.name, name)
                   : fmt::format("render graph: '{}' needs '{}', which is not available; skipped", node.name, name));
    }
    return false;
  }
  pass.layout = pass.plan->layout ? pass.plan->layout : find_layout(*pass.plan);
  if (!pass.layout) {
    return false;
  }
  if (pass.plan->manual) {
    return true;
  }
  std::vector<BindIdentity> identities;
  identities.reserve(pass.plan->slots.size());
  for (const BindingSlot& slot : pass.plan->slots) {
    BindIdentity identity = pass.bound[slot.use].identity;
    if (is_storage_texture(slot.type)) {
      identity.selector = 1;
    }
    identities.push_back(identity);
  }
  pass.passGroup = fetch_group(pass, identities, stage.frameId);
  if (g_recordDetails.load(std::memory_order_relaxed)) {
    g_worker.lastIdentities[pass.node] = std::move(identities);
  }
  return true;
}

FrameUniforms frame_uniforms(const CompiledGraph& graph, const StageData& stage) noexcept {
  const uint32_t samples = std::max(webgpu::g_graphicsConfig.msaaSamples, 1u);
  const auto inverse = [](uint32_t size) { return size != 0 ? 1.0f / static_cast<float>(size) : 0.0f; };
  FrameUniforms uniforms;
  FrameConstants& constants = uniforms.constants;
  constants.efbSize[0] = graph.efbSize.width;
  constants.efbSize[1] = graph.efbSize.height;
  constants.outputSize[0] = graph.outputSize.width;
  constants.outputSize[1] = graph.outputSize.height;
  constants.invEfbSize[0] = inverse(graph.efbSize.width);
  constants.invEfbSize[1] = inverse(graph.efbSize.height);
  constants.invOutputSize[0] = inverse(graph.outputSize.width);
  constants.invOutputSize[1] = inverse(graph.outputSize.height);
  constants.frameIndex = stage.frameIndex;
  constants.flags = (gx::UseReversedZ ? FrameFlagReversedZ : 0) | (samples > 1 ? FrameFlagMultisampled : 0);
  constants.msaaSamples = samples;
  return uniforms;
}

const BoundUse* bound_use(const BoundPass& pass, uint16_t index) noexcept {
  return index < pass.bound.size() ? &pass.bound[index] : nullptr;
}

const CompiledUse* declared_use(const BoundPass& pass, uint16_t index, bool buffer) noexcept {
  return index < pass.uses.size() && pass.uses[index].buffer == buffer ? &pass.uses[index] : nullptr;
}

std::optional<wgpu::TextureViewDimension> slot_view(const BoundPass& pass, uint16_t index) noexcept {
  for (const BindingSlot& slot : pass.plan->slots) {
    if (slot.use == index) {
      return slot.viewDimension;
    }
  }
  return std::nullopt;
}

wgpu::TextureView attachment_view(const BoundUse& bound, uint32_t mip, uint32_t layer) {
  if (bound.physical != nullptr) {
    return bound.physical->attachment_view(mip, layer);
  }
  if (bound.persistent.id != 0) {
    return persistent_attachment_view(bound.persistent, bound.generation, mip, layer);
  }
  return bound.view;
}

constexpr bool creator_write(const CompiledUse& use) noexcept {
  return use.role == UseRole::Create || use.role == UseRole::History;
}

constexpr bool owned_elsewhere(ResourceKind kind) noexcept {
  return kind == ResourceKind::Gx || kind == ResourceKind::Persistent;
}

bool targets_gx_color(const PassContext& ctx, TexRef ref) noexcept {
  const CompiledGraph* graph = ctx.stage_data().frame != nullptr ? ctx.stage_data().frame->graph.get() : nullptr;
  const CompiledNode& node = ctx.pass();
  if (graph == nullptr || ref.index >= node.useCount) {
    return false;
  }
  const CompiledResource& resource = graph->resources[graph->uses[node.firstUse + ref.index].resource];
  return resource.kind == ResourceKind::Gx && resource.gx == GxResource::Color;
}

class ExternalPassContext final : public render::PassContext {
public:
  explicit ExternalPassContext(PassContextState& state) noexcept : render::PassContext(state) {}
};
} // namespace

struct PassContextState {
  PassContext& ctx;
  const ExternalNodeState& node;
};

const wgpu::BindGroupLayout& frame_layout() {
  std::lock_guard lock{g_shared.mutex};
  ensure_frame_layout_locked();
  return g_shared.frameLayout;
}

const wgpu::BindGroup& frame_group() {
  std::lock_guard lock{g_shared.mutex};
  ensure_frame_group_locked();
  return g_shared.frameGroup;
}

void prepare_graph_bindings(CompiledGraph& graph, const PassSetup& setup) {
  const auto live = [&](size_t node) {
    const CompiledNode& compiled = graph.nodes[node];
    return node < graph.bindingPlans.size() && compiled.kind == NodeKind::Pass && compiled.status == CullReason::None &&
           compiled.passKind != PassKind::Native;
  };
  bool any = false;
  for (size_t node = 0; node < graph.nodes.size() && !any; ++node) {
    any = live(node);
  }
  // Device-free tests latch real graphs, and they never encode
  if (!any || !webgpu::g_device) {
    return;
  }
  wgpu::BindGroupLayout frameLayout;
  {
    std::lock_guard lock{g_shared.mutex};
    ensure_frame_group_locked();
    frameLayout = g_shared.frameLayout;
  }
  for (size_t node = 0; node < graph.nodes.size(); ++node) {
    if (!live(node)) {
      continue;
    }
    BindingPlan& plan = graph.bindingPlans[node];
    plan.layout = find_layout(plan);
    plan.layoutSerial = plan.layout ? plan.layout->serial : 0;
    CompiledNode& compiled = graph.nodes[node];
    // The graph is still private to the latch here, so the ids can be written before it is published
    if (compiled.pipelines != nullptr && plan.layout) {
      const PipelineSetup pipelineSetup{
          .pass = &setup,
          .frameLayout = frameLayout,
          .passLayout = plan.layout->passLayout,
          .layout = plan.layout->pipelineLayout,
          .layoutHash = plan.layoutHash,
          .local = compiled.local,
      };
      compiled.pipelines(pipelineSetup, compiled.state, compiled.pipelineIds);
    }
  }
}

void shutdown_bindings() {
  transient_pool().set_release_listener(nullptr);
  reset_worker_state();
  {
    std::lock_guard lock{g_shared.mutex};
    g_shared.frameGroup = {};
    g_shared.samplers = {};
    g_shared.layouts.clear();
    g_shared.emptyLayout = {};
    g_shared.frameLayout = {};
  }
  std::lock_guard lock{g_persistentMutex};
  g_persistent.clear();
  g_heapHandles.clear();
}

PersistentHandle create_persistent(std::string_view name, PersistentKind kind) {
  std::lock_guard lock{g_persistentMutex};
  return add_persistent_locked({.name = std::string{name}, .kind = kind});
}

PersistentHandle create_persistent_buffer_source(std::string_view name, PersistentBufferSource source, void* context) {
  std::lock_guard lock{g_persistentMutex};
  return add_persistent_locked(
      {.name = std::string{name}, .kind = PersistentKind::Buffer, .source = source, .context = context});
}

void set_persistent_buffer(PersistentHandle handle, wgpu::Buffer buffer, uint64_t offset, uint64_t size) {
  std::lock_guard lock{g_persistentMutex};
  PersistentRecord* record = find_persistent_locked(handle, PersistentKind::Buffer);
  CHECK(record != nullptr && record->source == nullptr, "render graph: persistent buffer {} cannot be set", handle.id);
  if (record == nullptr || record->source != nullptr) {
    return;
  }
  record->buffer = {
      .buffer = std::move(buffer),
      .offset = offset,
      .size = size,
      .generation = next_generation(record->buffer.generation),
  };
}

void set_persistent_texture(PersistentHandle handle, wgpu::Texture texture, wgpu::TextureView view) {
  std::lock_guard lock{g_persistentMutex};
  PersistentRecord* record = find_persistent_locked(handle, PersistentKind::Texture);
  CHECK(record != nullptr, "render graph: persistent texture {} cannot be set", handle.id);
  if (record == nullptr) {
    return;
  }
  record->texture = {
      .texture = std::move(texture),
      .view = std::move(view),
      .generation = next_generation(record->texture.generation),
  };
  record->attachmentViews.clear();
}

void destroy_persistent(PersistentHandle handle) {
  std::lock_guard lock{g_persistentMutex};
  g_persistent.erase(handle.id);
}

bool resolve_persistent(PersistentHandle handle, PersistentBuffer& out) {
  PersistentBufferSource source = nullptr;
  void* context = nullptr;
  {
    std::lock_guard lock{g_persistentMutex};
    const PersistentRecord* record = find_persistent_locked(handle, PersistentKind::Buffer);
    if (record == nullptr) {
      out = {};
      return false;
    }
    source = record->source;
    context = record->context;
    out = record->buffer;
  }
  // A source may call back into the registry, so it runs without the lock
  if (source != nullptr && !source(context, out)) {
    out = {};
    return false;
  }
  return out.buffer && out.generation != 0;
}

bool resolve_persistent(PersistentHandle handle, PersistentTexture& out) {
  std::lock_guard lock{g_persistentMutex};
  const PersistentRecord* record = find_persistent_locked(handle, PersistentKind::Texture);
  out = record != nullptr ? record->texture : PersistentTexture{};
  return out.texture && out.generation != 0;
}

PersistentHandle heap_pool_handle(gpu_heap::PoolId pool) {
  const auto raw = static_cast<uint32_t>(pool);
  std::lock_guard lock{g_persistentMutex};
  if (const auto it = g_heapHandles.find(raw); it != g_heapHandles.end()) {
    return it->second;
  }
  const PersistentHandle handle = add_persistent_locked({
      .name = fmt::format("heap pool {}", raw),
      .kind = PersistentKind::Buffer,
      .source = &heap_pool_source,
      .context = reinterpret_cast<void*>(static_cast<uintptr_t>(raw)),
  });
  if (handle.id != 0) {
    g_heapHandles.emplace(raw, handle);
  }
  return handle;
}

void register_gx_resolver(GxResource resource, GxResolveFn resolve) {
  const auto index = static_cast<size_t>(resource);
  CHECK(index < GxResourceCount, "render graph: unknown GX resource {}", index);
  if (index < GxResourceCount) {
    g_gxResolvers[index].store(resolve, std::memory_order_release);
  }
}

BindCacheStats bind_cache_stats() noexcept {
  return {
      .entries = g_stats.entries.load(std::memory_order_relaxed),
      .staleEntries = g_stats.staleEntries.load(std::memory_order_relaxed),
      .created = g_stats.created.load(std::memory_order_relaxed),
      .hits = g_stats.hits.load(std::memory_order_relaxed),
      .evicted = g_stats.evicted.load(std::memory_order_relaxed),
  };
}

void bind_cache_tick(uint64_t frameId) {
  WorkerState& worker = g_worker;
  if (worker.cache.empty() && worker.generations.empty()) {
    return;
  }
  // Pull sources such as gpu_heap only bump generations here, so their stale groups go before this frame binds
  for (auto it = worker.generations.begin(); it != worker.generations.end();) {
    const auto [id, seen] = *it;
    const uint32_t current = persistent_generation(id);
    if (current != seen) {
      evict_if([&](const CacheEntry& entry) { return references_generation(entry, id, seen); });
    }
    worker.recount = worker.recount || current != seen;
    if (current == 0) {
      it = worker.generations.erase(it);
    } else {
      it->second = current;
      ++it;
    }
  }
  if (frameId % SweepPeriod == 0) {
    evict_if([&](const CacheEntry& entry) { return frameId > entry.lastUsedFrame + RetainFrames; });
  }
  publish_cache();
}

void bind_cache_on_release(uint64_t serial) {
  if (g_worker.cache.empty()) {
    return;
  }
  const size_t evicted = evict_if([&](const CacheEntry& entry) {
    return std::ranges::any_of(entry.identities, [&](const BindIdentity& identity) {
      return (identity.kind == IdentityKind::Transient || identity.kind == IdentityKind::History) &&
             identity.value == serial;
    });
  });
  if (evicted != 0) {
    publish_cache();
  }
}

uint64_t bind_group_key(uint64_t layoutSerial, std::span<const BindIdentity> identities) noexcept {
  Hasher hasher{0};
  hasher.update(layoutSerial);
  hasher.update(static_cast<uint32_t>(identities.size()));
  for (const BindIdentity& identity : identities) {
    hasher.update(static_cast<uint8_t>(identity.kind));
    hasher.update(identity.selector);
    hasher.update(identity.value);
  }
  return hasher.digest();
}

void build_stage_bindings(StageData& stage, UniformSink sink) {
  const CompiledGraph* graph = stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
  if (graph == nullptr) {
    return;
  }
  if (sink == nullptr) {
    sink = static_cast<gfx::Range (*)(const uint8_t*, size_t)>(&gfx::push_uniform);
  }
  if (!g_frameConstants.pushed || g_frameConstants.frameId != stage.frameId) {
    const FrameUniforms uniforms = frame_uniforms(*graph, stage);
    const gfx::Range range = sink(reinterpret_cast<const uint8_t*>(&uniforms), sizeof(uniforms));
    g_frameConstants = {.frameId = stage.frameId, .offset = range.offset, .pushed = true};
  }
  StageBindings& bindings = stage.bindings;
  bindings.constantsOffset = g_frameConstants.offset;
  bindings.params.clear();
  const StageProgram& program = graph->stages[static_cast<size_t>(stage.stage)];
  for (uint32_t i = 0; i < stage.passCount && i < program.passes.size(); ++i) {
    const CompiledNode& node = graph->nodes[program.passes[i]];
    if (!stage.is_enabled(i) || node.paramsSize == 0) {
      continue;
    }
    const uint32_t size = std::min(node.paramsSize, MaxPassParamsSize);
    StageBindings::Params& params = bindings.params.emplace_back(StageBindings::Params{
        .node = program.passes[i],
        .size = static_cast<uint16_t>(size),
    });
    // Internal and external ids come from separate registries and can be equal
    std::span<const std::byte> latched;
    if (node.external) {
      if (const LatchedExternalPass* external = latched_external_pass(node.id)) {
        latched = std::span{external->params}.first(std::min<size_t>(external->paramsSize, size));
      }
    } else {
      latched = latched_pass_params(node.id);
    }
    std::copy_n(latched.begin(), std::min<size_t>(latched.size(), size), params.bytes.begin());
    params.uniformOffset = sink(reinterpret_cast<const uint8_t*>(params.bytes.data()), params.bytes.size()).offset;
  }
}

bool encode_pass(const wgpu::CommandEncoder& cmd, const StageData& stage, NodeId node) {
  const CompiledGraph* graph = stage.frame != nullptr ? stage.frame->graph.get() : nullptr;
  if (graph == nullptr || node >= graph->nodes.size()) {
    return false;
  }
  const CompiledNode& compiled = graph->nodes[node];
  if (compiled.encode == nullptr) {
    return false;
  }
  static const BindingPlan EmptyPlan;
  BoundPass pass{
      .graph = graph,
      .node = node,
      .plan = node < graph->bindingPlans.size() ? &graph->bindingPlans[node] : &EmptyPlan,
      .uses = graph->node_uses(node),
      .constantsOffset = stage.bindings.constantsOffset,
      .paramsOffset = stage.bindings.constantsOffset,
  };
  for (const StageBindings::Params& params : stage.bindings.params) {
    if (params.node == node) {
      pass.paramsOffset = params.uniformOffset;
      pass.params = std::span{params.bytes}.first(params.size);
      break;
    }
  }
  // Without an encoder nothing can be recorded, so nothing is bound either
  if (cmd) {
    if (!bind_pass(stage, pass)) {
      return false;
    }
    pass.frameLayout = frame_layout();
    pass.frameGroup = frame_group();
  }
  PassContext ctx{cmd, stage, compiled, pass};
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (cmd) {
    cmd.PushDebugGroup(compiled.name.c_str());
  }
#endif
  compiled.encode(ctx, compiled.state);
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (cmd) {
    cmd.PopDebugGroup();
  }
#endif
  return true;
}

void external_encode(PassContext& ctx, const void* state) {
  const auto& node = *static_cast<const ExternalNodeState*>(state);
  if (!external_pass_alive(node.id)) {
    return;
  }
  PassContextState contextState{ctx, node};
  ExternalPassContext external{contextState};
  node.desc.encode(external, node.desc.userdata);
  if (node.desc.afterSubmit != nullptr) {
    queue_after_frame([id = node.id, callback = node.desc.afterSubmit, userdata = node.desc.userdata,
                       frame = ctx.stage_data().frameIndex] {
      if (external_pass_alive(id)) {
        callback(frame, userdata);
      }
    });
  }
}

void queue_after_frame(gfx::AfterSubmitCallback callback) { g_afterFrame.push_back(std::move(callback)); }

void run_after_frame_callbacks() noexcept {
  if (g_afterFrame.empty()) {
    return;
  }
  const std::vector<gfx::AfterSubmitCallback> callbacks = std::exchange(g_afterFrame, {});
  for (const auto& callback : callbacks) {
    if (callback) {
      callback();
    }
  }
}

void clear_after_frame_callbacks() noexcept { g_afterFrame.clear(); }

Stage PassContext::stage() const noexcept { return m_stage.stage; }

uint64_t PassContext::frame_id() const noexcept { return m_stage.frameId; }

uint32_t PassContext::frame_index() const noexcept { return m_stage.frameIndex; }

const wgpu::Device& PassContext::device() const noexcept { return webgpu::g_device; }

std::span<const std::byte> PassContext::params() const noexcept { return m_bound.params; }

template <typename Encoder>
void PassContext::bind_groups_to(const Encoder& pass) const {
  if (!pass) {
    return;
  }
  if (m_bound.frameGroup) {
    const std::array offsets{m_bound.constantsOffset, m_bound.paramsOffset};
    pass.SetBindGroup(FrameGroupIndex, m_bound.frameGroup, offsets.size(), offsets.data());
  }
  if (!m_bound.plan->manual && m_bound.passGroup) {
    pass.SetBindGroup(PassGroupIndex, m_bound.passGroup);
  }
}

void PassContext::bind_groups(const wgpu::ComputePassEncoder& pass) const { bind_groups_to(pass); }

void PassContext::bind_groups(const wgpu::RenderPassEncoder& pass) const { bind_groups_to(pass); }

wgpu::ComputePassEncoder PassContext::begin_compute(const char* label) const {
  const bool allowed = m_pass.passKind == PassKind::Compute || m_pass.passKind == PassKind::Encoder;
  CHECK(allowed, "render graph: '{}' is a {} pass and cannot begin a compute pass", m_pass.name,
        pass_kind_name(m_pass.passKind));
  if (!allowed || !m_encoder) {
    return {};
  }
  const wgpu::ComputePassDescriptor descriptor{
      .label = label != nullptr ? label : m_pass.name.c_str(),
      .timestampWrites = webgpu::gpu_prof::pass_writes(m_pass.name),
  };
  wgpu::ComputePassEncoder pass = m_encoder.BeginComputePass(&descriptor);
  bind_groups_to(pass);
  return pass;
}

wgpu::RenderPassEncoder PassContext::begin_render(const char* label) const {
  const BindingPlan& plan = *m_bound.plan;
  std::array<ColorTarget, gfx::MaxColorAttachments> colors;
  size_t count = 0;
  for (const AttachmentSlot& attachment : plan.colors) {
    if (count < colors.size()) {
      colors[count++] = {.ref = {static_cast<uint16_t>(attachment.use)}};
    }
  }
  std::optional<DepthTarget> depth;
  if (plan.depth) {
    depth = DepthTarget{.ref = {static_cast<uint16_t>(plan.depth->use)}};
  }
  return begin_render(std::span{colors}.first(count), depth ? &*depth : nullptr, label);
}

wgpu::RenderPassEncoder PassContext::begin_render(std::span<const ColorTarget> colors, const DepthTarget* depth,
                                                  const char* label) const {
  const bool allowed = m_pass.passKind == PassKind::Render || m_pass.passKind == PassKind::Encoder;
  CHECK(allowed, "render graph: '{}' is a {} pass and cannot begin a render pass", m_pass.name,
        pass_kind_name(m_pass.passKind));
  CHECK(colors.size() <= gfx::MaxColorAttachments, "render graph: '{}' attaches {} colour targets", m_pass.name,
        colors.size());
  if (!allowed) {
    return {};
  }
  const auto& resources = m_bound.graph->resources;
  const bool record = g_recordDetails.load(std::memory_order_relaxed);
  std::array<wgpu::RenderPassColorAttachment, gfx::MaxColorAttachments> attachments{};
  std::vector<testing::AttachmentOps> ops;
  wgpu::Extent3D target{};
  const size_t colorCount = std::min(colors.size(), attachments.size());
  for (size_t i = 0; i < colorCount; ++i) {
    const ColorTarget& color = colors[i];
    const CompiledUse* use = declared_use(m_bound, color.ref.index, false);
    const bool isColor = use != nullptr && (use->access == Access::ColorWrite || use->access == Access::ColorRMW);
    CHECK(isColor, "render graph: '{}' attaches texture {} without ColorWrite or ColorRMW", m_pass.name,
          color.ref.index);
    const BoundUse* bound = isColor ? bound_use(m_bound, color.ref.index) : nullptr;
    if (bound == nullptr) {
      return {};
    }
    const bool creator = creator_write(*use);
    const bool owned = owned_elsewhere(resources[use->resource].kind);
    CHECK(!color.clear || (use->access == Access::ColorWrite && (creator || !owned)),
          "render graph: '{}' clears '{}', which it may only load", m_pass.name, resources[use->resource].name);
    const bool clear = use->access == Access::ColorWrite && (creator || (color.clear && !owned));
    const wgpu::Color value = clear ? color.clear.value_or(wgpu::Color{0, 0, 0, 0}) : wgpu::Color{};
    const bool volume = bound->texture && bound->texture.GetDimension() == wgpu::TextureDimension::e3D;
    attachments[i] = {
        .view = attachment_view(*bound, color.mip, color.layer),
        .depthSlice = volume ? color.layer : wgpu::kDepthSliceUndefined,
        .resolveTarget = bound->resolveView,
        .loadOp = clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = value,
    };
    const uint32_t width = std::max(1u, bound->extent.width >> color.mip);
    const uint32_t height = std::max(1u, bound->extent.height >> color.mip);
    if (target.width == 0) {
      target = {width, height, 1};
    }
    if (!record) {
      continue;
    }
    ops.push_back({
        .load = attachments[i].loadOp,
        .store = wgpu::StoreOp::Store,
        .clearColor = value,
        .resolves = static_cast<bool>(bound->resolveView),
        .width = width,
        .height = height,
    });
  }

  wgpu::RenderPassDepthStencilAttachment depthAttachment{};
  if (depth != nullptr) {
    const CompiledUse* use = declared_use(m_bound, depth->ref.index, false);
    const bool isDepth = use != nullptr && (use->access == Access::DepthRead || use->access == Access::DepthWrite);
    CHECK(isDepth, "render graph: '{}' attaches texture {} without DepthRead or DepthWrite", m_pass.name,
          depth->ref.index);
    const BoundUse* bound = isDepth ? bound_use(m_bound, depth->ref.index) : nullptr;
    if (bound == nullptr) {
      return {};
    }
    const auto info = texture_format_info(bound->format);
    const bool stencil = info && info->stencil;
    depthAttachment.view = attachment_view(*bound, depth->mip, depth->layer);
    testing::AttachmentOps op{.depth = true};
    if (use->access == Access::DepthRead) {
      depthAttachment.depthReadOnly = true;
      depthAttachment.stencilReadOnly = stencil;
      op.readOnly = true;
    } else {
      const bool creator = creator_write(*use);
      const bool owned = owned_elsewhere(resources[use->resource].kind);
      CHECK(!depth->clear || creator || !owned, "render graph: '{}' clears '{}', which it may only load", m_pass.name,
            resources[use->resource].name);
      const bool clear = creator || (depth->clear && !owned);
      const wgpu::LoadOp load = clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
      depthAttachment.depthLoadOp = load;
      depthAttachment.depthStoreOp = wgpu::StoreOp::Store;
      // Reversed Z puts the far plane at 0
      depthAttachment.depthClearValue = clear ? depth->clear.value_or(gx::UseReversedZ ? 0.0f : 1.0f) : 0.0f;
      if (stencil) {
        depthAttachment.stencilLoadOp = load;
        depthAttachment.stencilStoreOp = wgpu::StoreOp::Store;
        depthAttachment.stencilClearValue = 0;
      }
      op.load = load;
      op.store = wgpu::StoreOp::Store;
      op.clearDepth = depthAttachment.depthClearValue;
    }
    op.width = std::max(1u, bound->extent.width >> depth->mip);
    op.height = std::max(1u, bound->extent.height >> depth->mip);
    if (target.width == 0) {
      target = {op.width, op.height, 1};
    }
    if (record) {
      ops.push_back(op);
    }
  }

  if (record) {
    g_worker.lastAttachmentOps = std::move(ops);
  }
  if (!m_encoder) {
    return {};
  }
  const wgpu::RenderPassDescriptor descriptor{
      .label = label != nullptr ? label : m_pass.name.c_str(),
      .colorAttachmentCount = colorCount,
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depth != nullptr ? &depthAttachment : nullptr,
      .timestampWrites = webgpu::gpu_prof::pass_writes(m_pass.name),
  };
  wgpu::RenderPassEncoder pass = m_encoder.BeginRenderPass(&descriptor);
  if (target.width != 0) {
    pass.SetViewport(0, 0, static_cast<float>(target.width), static_cast<float>(target.height), 0, 1);
  }
  bind_groups_to(pass);
  return pass;
}

wgpu::TextureView PassContext::view(TexRef ref, uint32_t mip) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  CHECK(use != nullptr, "render graph: '{}' did not declare texture {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  if (bound == nullptr) {
    return {};
  }
  if (bound->physical == nullptr) {
    return bound->view;
  }
  const auto dimension = slot_view(m_bound, ref.index).value_or(bound->physical->key.bindingView);
  return mip == AllMips ? bound->physical->binding_view(dimension) : bound->physical->mip_view(mip, dimension);
}

wgpu::Texture PassContext::texture(TexRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  CHECK(use != nullptr, "render graph: '{}' did not declare texture {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  return bound != nullptr ? bound->texture : wgpu::Texture{};
}

wgpu::TextureView PassContext::prev(TexRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  const CompiledResource* resource = use != nullptr ? &m_bound.graph->resources[use->resource] : nullptr;
  const bool history = resource != nullptr && resource->kind == ResourceKind::History;
  CHECK(history, "render graph: '{}' reads the previous frame of texture {}, which is not a history", m_pass.name,
        ref.index);
  const TransientPool& pool = transient_pool();
  if (!history || resource->historySlot == InvalidResource || pool.materialized_graph() != m_bound.graph) {
    return {};
  }
  const PhysicalTexture& physical = pool.history_prev(resource->historySlot);
  return physical.binding_view(physical.key.bindingView);
}

bool PassContext::prev_valid(TexRef ref) const noexcept {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  if (use == nullptr) {
    return false;
  }
  const CompiledResource& resource = m_bound.graph->resources[use->resource];
  const TransientPool& pool = transient_pool();
  return resource.kind == ResourceKind::History && m_stage.prev_valid(resource.historySlot) &&
         pool.materialized_graph() == m_bound.graph && pool.history_fresh(resource.historySlot);
}

BufferView PassContext::buffer(BufRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, true);
  CHECK(use != nullptr, "render graph: '{}' did not declare buffer {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  if (bound == nullptr) {
    return {};
  }
  return {.buffer = bound->buffer, .offset = bound->offset, .size = bound->size};
}

wgpu::Extent3D PassContext::extent(TexRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  CHECK(use != nullptr, "render graph: '{}' did not declare texture {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  return bound != nullptr ? bound->extent : wgpu::Extent3D{};
}

wgpu::TextureFormat PassContext::format(TexRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  CHECK(use != nullptr, "render graph: '{}' did not declare texture {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  return bound != nullptr ? bound->format : wgpu::TextureFormat::Undefined;
}

uint32_t PassContext::sample_count(TexRef ref) const {
  const CompiledUse* use = declared_use(m_bound, ref.index, false);
  CHECK(use != nullptr, "render graph: '{}' did not declare texture {}", m_pass.name, ref.index);
  const BoundUse* bound = use != nullptr ? bound_use(m_bound, ref.index) : nullptr;
  return bound != nullptr ? bound->samples : 1;
}

const wgpu::BindGroup& PassContext::frame_group() const noexcept { return m_bound.frameGroup; }

uint32_t PassContext::frame_offset() const noexcept { return m_bound.constantsOffset; }

const wgpu::BindGroup& PassContext::pass_group() const noexcept { return m_bound.passGroup; }

uint32_t PassContext::params_offset() const noexcept { return m_bound.paramsOffset; }

const wgpu::BindGroupLayout& PassContext::frame_layout() const noexcept { return m_bound.frameLayout; }

const wgpu::BindGroupLayout& PassContext::pass_layout() const noexcept {
  static const wgpu::BindGroupLayout Empty;
  return m_bound.layout ? m_bound.layout->passLayout : Empty;
}

const wgpu::PipelineLayout& PassContext::pipeline_layout() const noexcept {
  static const wgpu::PipelineLayout Empty;
  return m_bound.layout ? m_bound.layout->pipelineLayout : Empty;
}

uint64_t PassContext::layout_hash() const noexcept { return m_bound.plan->layoutHash; }

shaders::PipelineId PassContext::pipeline(uint32_t index) const noexcept {
  return {index < MaxPassPipelines ? m_pass.pipelineIds[index] : 0};
}

namespace testing {
void record_bind_details(bool enabled) noexcept { g_recordDetails.store(enabled, std::memory_order_relaxed); }

std::vector<BindIdentity> last_identities(NodeId node) {
  const auto it = g_worker.lastIdentities.find(node);
  return it != g_worker.lastIdentities.end() ? it->second : std::vector<BindIdentity>{};
}

std::vector<AttachmentOps> last_attachment_ops() { return g_worker.lastAttachmentOps; }

void reset_bind_cache() { reset_worker_state(); }

bool frame_objects_created() noexcept {
  std::lock_guard lock{g_shared.mutex};
  return g_shared.frameLayout || g_shared.frameGroup || !g_shared.layouts.empty();
}
} // namespace testing
} // namespace aurora::render::detail

namespace aurora::render {
Stage PassContext::stage() const noexcept { return m_state->ctx.stage(); }

std::string_view PassContext::name() const noexcept { return m_state->ctx.name(); }

uint32_t PassContext::frame() const noexcept { return m_state->ctx.stage_data().frameIndex; }

std::span<const std::byte> PassContext::params() const noexcept { return m_state->ctx.params(); }

const wgpu::Device& PassContext::device() const noexcept { return m_state->ctx.device(); }

const wgpu::CommandEncoder& PassContext::encoder() const noexcept { return m_state->ctx.encoder(); }

wgpu::ComputePassEncoder PassContext::begin_compute() { return m_state->ctx.begin_compute(); }

wgpu::RenderPassEncoder PassContext::begin_render() { return m_state->ctx.begin_render(); }

wgpu::Texture PassContext::texture(TexRef ref) const { return m_state->ctx.texture(ref); }

wgpu::TextureView PassContext::view(TexRef ref, uint32_t mip) const { return m_state->ctx.view(ref, mip); }

wgpu::TextureView PassContext::prev(TexRef ref) const { return m_state->ctx.prev(ref); }

bool PassContext::prev_valid(TexRef ref) const noexcept { return m_state->ctx.prev_valid(ref); }

BufferView PassContext::buffer(BufRef ref) const { return m_state->ctx.buffer(ref); }

wgpu::Extent3D PassContext::extent(TexRef ref) const { return m_state->ctx.extent(ref); }

wgpu::TextureFormat PassContext::format(TexRef ref) const { return m_state->ctx.format(ref); }

uint32_t PassContext::sample_count(TexRef ref) const { return m_state->ctx.sample_count(ref); }

wgpu::ColorTargetState PassContext::color_target(TexRef ref) const {
  // EFB alpha belongs to the game (destination-alpha blends and EFB copies read it), so gx.color writes keep it
  const wgpu::ColorWriteMask mask =
      detail::targets_gx_color(m_state->ctx, ref)
          ? wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue
          : wgpu::ColorWriteMask::All;
  return {.format = format(ref), .blend = nullptr, .writeMask = mask};
}

const wgpu::BindGroupLayout& PassContext::frame_layout() const noexcept { return m_state->ctx.frame_layout(); }

const wgpu::BindGroupLayout& PassContext::pass_layout() const noexcept { return m_state->ctx.pass_layout(); }

const wgpu::PipelineLayout& PassContext::pipeline_layout() const noexcept { return m_state->ctx.pipeline_layout(); }
} // namespace aurora::render

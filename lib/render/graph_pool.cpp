#include "graph_pool.hpp"

#include "frame_state.hpp"
#include "gpu_heap.hpp"
#include "graph_slots.hpp"
#include "stage_data.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render::graph"};
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint32_t MaxWorkgroups = 65535;

std::atomic_uint64_t g_nextSerial{1};
std::atomic_bool g_forcedPressure{false};
// -1 until set_debug_poison or the first materialize decides it
std::atomic_int g_poison{-1};
std::atomic_uint32_t g_unpoisonable{0};

constexpr bool default_poison() noexcept {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif
}

void decide_poison() {
  if (g_poison.load(std::memory_order_relaxed) >= 0) {
    return;
  }
  int decided = default_poison() ? 1 : 0;
  if (const char* value = SDL_getenv("AURORA_RENDER_POISON")) {
    const std::string_view text{value};
    if (text == "0" || text == "1") {
      decided = text == "1" ? 1 : 0;
    }
  }
  int expected = -1;
  g_poison.compare_exchange_strong(expected, decided, std::memory_order_relaxed);
}

enum ViewKind : uint8_t { BindingView, MipView, AttachmentView };

class DawnAllocator final : public PhysicalAllocator {
public:
  PhysicalTexture create_texture(const TexKey& key, std::string_view label) override {
    CHECK(texture_format_info(key.format).has_value(), "transient pool: '{}' has a format without a size", label);
    const std::string text{label};
    // Compat textures bind with one view dimension, fixed at creation
    const wgpu::TextureBindingViewDimension bindingView{wgpu::TextureBindingViewDimension::Init{
        .textureBindingViewDimension = key.bindingView,
    }};
    const wgpu::TextureDescriptor descriptor{
        .nextInChain = webgpu::g_hasCoreFeatures ? nullptr : &bindingView,
        .label = text.c_str(),
        .usage = key.usage,
        .dimension = key.dimension,
        .size = {key.width, key.height, key.depthOrLayers},
        .format = key.format,
        .mipLevelCount = key.mips,
    };
    PhysicalTexture physical;
    physical.serial = next_resource_serial();
    physical.texture = webgpu::g_device.CreateTexture(&descriptor);
    physical.key = key;
    physical.bytes = key_bytes(key);
    return physical;
  }

  PhysicalBuffer create_buffer(const BufKey& key, std::string_view label) override {
    const std::string text{label};
    const wgpu::BufferDescriptor descriptor{
        .label = text.c_str(),
        .usage = key.usage,
        .size = key.bucketSize,
    };
    return {
        .serial = next_resource_serial(),
        .buffer = webgpu::g_device.CreateBuffer(&descriptor),
        .key = key,
        .bytes = key.bucketSize,
    };
  }
};

wgpu::ComputePipeline create_pipeline(const wgpu::BindGroupLayout& layout, const std::string& source,
                                      const char* label) {
  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{
      .code = source.c_str(),
  }};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &wgslSource,
      .label = label,
  };
  const auto module = webgpu::g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &layout,
  };
  const auto pipelineLayout = webgpu::g_device.CreatePipelineLayout(&layoutDescriptor);
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = label,
      .layout = pipelineLayout,
      .compute =
          wgpu::ComputeState{
              .module = module,
              .entryPoint = "main",
          },
  };
  return webgpu::g_device.CreateComputePipeline(&descriptor);
}

wgpu::BindGroup create_group(const wgpu::BindGroupLayout& layout, const wgpu::BindGroupEntry& entry) {
  const wgpu::BindGroupDescriptor descriptor{
      .label = "graph poison",
      .layout = layout,
      .entryCount = 1,
      .entries = &entry,
  };
  return webgpu::g_device.CreateBindGroup(&descriptor);
}

bool eight_bit_integer(wgpu::TextureFormat format) noexcept {
  using wgpu::TextureFormat;
  switch (format) {
  case TextureFormat::R8Uint:
  case TextureFormat::RG8Uint:
  case TextureFormat::RGBA8Uint:
  case TextureFormat::R8Sint:
  case TextureFormat::RG8Sint:
  case TextureFormat::RGBA8Sint:
    return true;
  default:
    return false;
  }
}

wgpu::Color poison_color(wgpu::TextureFormat format, const FormatInfo& info) noexcept {
  switch (info.sampleType) {
  case wgpu::TextureSampleType::Uint: {
    // rgb10a2uint has a two-bit alpha, which 186 would overflow
    if (format == wgpu::TextureFormat::RGB10A2Uint) {
      return {186, 186, 186, 3};
    }
    const double value = eight_bit_integer(format) ? 186 : 2989;
    return {value, value, value, value};
  }
  case wgpu::TextureSampleType::Sint: {
    const double value = eight_bit_integer(format) ? -70 : -2989;
    return {value, value, value, value};
  }
  default:
    return {1, 0, 1, 1};
  }
}

struct TexelName {
  wgpu::TextureFormat format;
  std::string_view name;
};

constexpr std::array TexelNames{
    TexelName{wgpu::TextureFormat::R8Unorm, "r8unorm"},
    TexelName{wgpu::TextureFormat::R8Snorm, "r8snorm"},
    TexelName{wgpu::TextureFormat::R8Uint, "r8uint"},
    TexelName{wgpu::TextureFormat::R8Sint, "r8sint"},
    TexelName{wgpu::TextureFormat::R16Unorm, "r16unorm"},
    TexelName{wgpu::TextureFormat::R16Snorm, "r16snorm"},
    TexelName{wgpu::TextureFormat::R16Uint, "r16uint"},
    TexelName{wgpu::TextureFormat::R16Sint, "r16sint"},
    TexelName{wgpu::TextureFormat::R16Float, "r16float"},
    TexelName{wgpu::TextureFormat::RG8Unorm, "rg8unorm"},
    TexelName{wgpu::TextureFormat::RG8Snorm, "rg8snorm"},
    TexelName{wgpu::TextureFormat::RG8Uint, "rg8uint"},
    TexelName{wgpu::TextureFormat::RG8Sint, "rg8sint"},
    TexelName{wgpu::TextureFormat::R32Uint, "r32uint"},
    TexelName{wgpu::TextureFormat::R32Sint, "r32sint"},
    TexelName{wgpu::TextureFormat::R32Float, "r32float"},
    TexelName{wgpu::TextureFormat::RG16Unorm, "rg16unorm"},
    TexelName{wgpu::TextureFormat::RG16Snorm, "rg16snorm"},
    TexelName{wgpu::TextureFormat::RG16Uint, "rg16uint"},
    TexelName{wgpu::TextureFormat::RG16Sint, "rg16sint"},
    TexelName{wgpu::TextureFormat::RG16Float, "rg16float"},
    TexelName{wgpu::TextureFormat::RGBA8Unorm, "rgba8unorm"},
    TexelName{wgpu::TextureFormat::RGBA8Snorm, "rgba8snorm"},
    TexelName{wgpu::TextureFormat::RGBA8Uint, "rgba8uint"},
    TexelName{wgpu::TextureFormat::RGBA8Sint, "rgba8sint"},
    TexelName{wgpu::TextureFormat::BGRA8Unorm, "bgra8unorm"},
    TexelName{wgpu::TextureFormat::RGB10A2Uint, "rgb10a2uint"},
    TexelName{wgpu::TextureFormat::RGB10A2Unorm, "rgb10a2unorm"},
    TexelName{wgpu::TextureFormat::RG11B10Ufloat, "rg11b10ufloat"},
    TexelName{wgpu::TextureFormat::RG32Uint, "rg32uint"},
    TexelName{wgpu::TextureFormat::RG32Sint, "rg32sint"},
    TexelName{wgpu::TextureFormat::RG32Float, "rg32float"},
    TexelName{wgpu::TextureFormat::RGBA16Unorm, "rgba16unorm"},
    TexelName{wgpu::TextureFormat::RGBA16Snorm, "rgba16snorm"},
    TexelName{wgpu::TextureFormat::RGBA16Uint, "rgba16uint"},
    TexelName{wgpu::TextureFormat::RGBA16Sint, "rgba16sint"},
    TexelName{wgpu::TextureFormat::RGBA16Float, "rgba16float"},
    TexelName{wgpu::TextureFormat::RGBA32Uint, "rgba32uint"},
    TexelName{wgpu::TextureFormat::RGBA32Sint, "rgba32sint"},
    TexelName{wgpu::TextureFormat::RGBA32Float, "rgba32float"},
};

std::string_view wgsl_texel_format(wgpu::TextureFormat format) noexcept {
  const auto it = std::ranges::find(TexelNames, format, &TexelName::format);
  return it != TexelNames.end() ? it->name : std::string_view{};
}

std::string clear_kernel_source(std::string_view texel, wgpu::TextureViewDimension dimension, const wgpu::Color& color,
                                wgpu::TextureSampleType type) {
  std::string value;
  if (type == wgpu::TextureSampleType::Uint) {
    value = fmt::format("vec4<u32>({}u, {}u, {}u, {}u)", color.r, color.g, color.b, color.a);
  } else if (type == wgpu::TextureSampleType::Sint) {
    value = fmt::format("vec4<i32>({}, {}, {}, {})", color.r, color.g, color.b, color.a);
  } else {
    value = fmt::format("vec4<f32>({:.1f}, {:.1f}, {:.1f}, {:.1f})", color.r, color.g, color.b, color.a);
  }
  std::string_view storage = "texture_storage_2d";
  std::string_view bounds = "id.x < size.x && id.y < size.y";
  std::string_view coords = "id.xy";
  if (dimension == wgpu::TextureViewDimension::e2DArray) {
    storage = "texture_storage_2d_array";
    bounds = "id.x < size.x && id.y < size.y && id.z < textureNumLayers(dst)";
    coords = "id.xy, id.z";
  } else if (dimension == wgpu::TextureViewDimension::e3D) {
    storage = "texture_storage_3d";
    bounds = "id.x < size.x && id.y < size.y && id.z < size.z";
    coords = "id";
  }
  return fmt::format(R"(@group(0) @binding(0) var dst: {}<{}, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let size = textureDimensions(dst);
  if ({}) {{
    textureStore(dst, {}, {});
  }}
}}
)",
                     storage, texel, bounds, coords, value);
}

constexpr std::string_view FillKernelSource = R"(@group(0) @binding(0) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = id.y * groups.x * 64u + id.x;
  if (index < arrayLength(&dst)) {
    dst[index] = 0xFFFFFFFFu;
  }
}
)";

// Render worker only; the cursor and the history flags start over every frame
struct PoisonState {
  const CompiledGraph* graph = nullptr;
  uint64_t frameId = 0;
  size_t cursor = 0;
  std::vector<bool> historyDone;
  std::unordered_set<std::string> logged;
  Poisoner poisoner;
};

PoisonState g_poisonState;

void reset_poison_state() { g_poisonState = {}; }

// A compute clear also needs a WGSL texel format for the texture
bool poisonable(const TexKey& key) noexcept {
  const PoisonMethod method = poison_method(key);
  return method != PoisonMethod::None &&
         (method != PoisonMethod::ComputeClear || !wgsl_texel_format(key.format).empty());
}

void poison_texture(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture, const std::string& name) {
  if (poisonable(texture.key)) {
    g_poisonState.poisoner.encode(cmd, texture);
    return;
  }
  if (g_poisonState.logged.insert(name).second) {
    g_unpoisonable.fetch_add(1, std::memory_order_relaxed);
    Log.warn("render graph: '{}' has no usage that can poison it; left as is", name);
  }
}

uint64_t serial_of(const auto& physical) noexcept {
  return physical.isBuffer ? physical.buffer.serial : physical.texture.serial;
}

SizeClass size_class_of(const auto& physical) noexcept {
  return physical.isBuffer ? physical.buffer.key.sizeClass : physical.texture.key.sizeClass;
}

bool known(const GraphExtent& extent) noexcept { return extent.width != 0 && extent.height != 0; }
} // namespace

const wgpu::TextureView& PhysicalTexture::binding_view(wgpu::TextureViewDimension dimension) const {
  CHECK(webgpu::g_hasCoreFeatures || dimension == key.bindingView,
        "transient pool: compat texture bound as view dimension {}, created for {}", static_cast<uint32_t>(dimension),
        static_cast<uint32_t>(key.bindingView));
  return cached_view(BindingView, 0, 0, dimension);
}

const wgpu::TextureView& PhysicalTexture::mip_view(uint32_t mip, wgpu::TextureViewDimension dimension) const {
  return cached_view(MipView, mip, 0, dimension);
}

const wgpu::TextureView& PhysicalTexture::attachment_view(uint32_t mip, uint32_t layerOrSlice) const {
  // A 3D attachment selects its slice with depthSlice, so every slice of a mip shares one view
  if (key.dimension == wgpu::TextureDimension::e3D) {
    return cached_view(AttachmentView, mip, 0, wgpu::TextureViewDimension::e3D);
  }
  return cached_view(AttachmentView, mip, layerOrSlice, wgpu::TextureViewDimension::e2D);
}

const wgpu::TextureView& PhysicalTexture::cached_view(uint8_t kind, uint32_t mip, uint32_t layer,
                                                      wgpu::TextureViewDimension dimension) const {
  if (!texture) {
    static const wgpu::TextureView Empty;
    return Empty;
  }
  for (const CachedView& cached : m_views) {
    if (cached.kind == kind && cached.mip == mip && cached.layer == layer && cached.dimension == dimension) {
      return cached.view;
    }
  }
  const auto info = texture_format_info(key.format);
  const bool depthStencil = info && info->depth && info->stencil;
  const bool attachment = kind == AttachmentView;
  const wgpu::TextureViewDescriptor descriptor{
      .dimension = dimension,
      .baseMipLevel = kind == BindingView ? 0 : mip,
      .mipLevelCount = kind == BindingView ? key.mips : 1,
      .baseArrayLayer = attachment ? layer : 0,
      .arrayLayerCount =
          attachment && dimension != wgpu::TextureViewDimension::e3D ? 1 : wgpu::kArrayLayerCountUndefined,
      .aspect = depthStencil && !attachment ? wgpu::TextureAspect::DepthOnly : wgpu::TextureAspect::All,
  };
  return m_views
      .emplace_back(CachedView{
          .kind = kind,
          .mip = mip,
          .layer = layer,
          .dimension = dimension,
          .view = texture.CreateView(&descriptor),
      })
      .view;
}

std::unique_ptr<PhysicalAllocator> make_dawn_allocator() { return std::make_unique<DawnAllocator>(); }

uint64_t next_resource_serial() noexcept { return g_nextSerial.fetch_add(1, std::memory_order_relaxed); }

TransientPool::TransientPool(std::unique_ptr<PhysicalAllocator> allocator) : m_allocator(std::move(allocator)) {}

TransientPool::~TransientPool() = default;

void TransientPool::materialize(const std::shared_ptr<const CompiledGraph>& graph, Clock::time_point now) {
  if (graph.get() == m_graph.get()) {
    return;
  }
  static const SlotPlan EmptyPlan;
  const SlotPlan& plan = graph != nullptr ? graph->slots : EmptyPlan;
  if (empty() && plan.slots.empty() && plan.histories.empty()) {
    m_graph = graph;
    return;
  }
  decide_poison();
  std::vector<uint64_t> released;

  const size_t existing = m_physicals.size();
  std::vector<bool> wasClaimed(existing);
  for (size_t i = 0; i < existing; ++i) {
    wasClaimed[i] = std::exchange(m_physicals[i]->claimed, false);
  }
  const auto best = [&](const SlotPlan::Slot& slot, bool sameAnchor) -> Physical* {
    Physical* found = nullptr;
    for (const auto& physical : m_physicals) {
      const bool fits = !physical->claimed && physical->isBuffer == slot.isBuffer &&
                        (slot.isBuffer ? physical->buffer.key == slot.buf : physical->texture.key == slot.tex) &&
                        (!sameAnchor || physical->anchor == slot.anchor);
      if (fits && (found == nullptr || serial_of(*physical) < serial_of(*found))) {
        found = physical.get();
      }
    }
    return found;
  };
  // Anchors first, so a slot without its own physical cannot take one another slot served last time
  m_slots.assign(plan.slots.size(), nullptr);
  for (size_t s = 0; s < plan.slots.size(); ++s) {
    if (Physical* physical = best(plan.slots[s], true)) {
      physical->claimed = true;
      m_slots[s] = physical;
    }
  }
  for (size_t s = 0; s < plan.slots.size(); ++s) {
    if (m_slots[s] != nullptr) {
      continue;
    }
    Physical* physical = best(plan.slots[s], false);
    if (physical == nullptr) {
      physical = &create_physical(plan.slots[s]);
    }
    physical->claimed = true;
    m_slots[s] = physical;
  }
  for (size_t s = 0; s < plan.slots.size(); ++s) {
    m_slots[s]->anchor = plan.slots[s].anchor;
  }
  for (size_t i = 0; i < existing; ++i) {
    if (!m_physicals[i]->claimed && wasClaimed[i]) {
      m_physicals[i]->idleSince = now;
    }
  }

  // A dropped pair idles like a slot physical, so a temporal feature toggled back on allocates nothing
  std::vector<HistoryPair> histories;
  histories.reserve(plan.histories.size());
  for (const SlotPlan::History& entry : plan.histories) {
    const auto matches = [&](const HistoryPair& pair) {
      return pair.name == entry.name && pair.textures[0].key == entry.key;
    };
    if (const auto kept = std::ranges::find_if(m_histories, matches); kept != m_histories.end()) {
      histories.push_back(std::move(*kept));
      m_histories.erase(kept);
      continue;
    }
    if (const auto idle = std::ranges::find_if(m_idleHistories, matches); idle != m_idleHistories.end()) {
      histories.push_back(std::move(*idle));
      m_idleHistories.erase(idle);
      continue;
    }
    HistoryPair& pair = histories.emplace_back(HistoryPair{.name = entry.name});
    for (size_t i = 0; i < pair.textures.size(); ++i) {
      pair.textures[i] = m_allocator->create_texture(entry.key, fmt::format("graph:{}#{}", entry.name, i));
      ++m_created;
    }
  }
  for (HistoryPair& pair : m_histories) {
    pair.pendingSwap = false;
    pair.fresh = false;
    pair.idleSince = now;
    m_idleHistories.push_back(std::move(pair));
  }
  m_histories = std::move(histories);

  // An empty graph has no extents, so it never counts as a resize
  const bool efbResized = known(plan.efbSize) && known(m_efbSize) && plan.efbSize != m_efbSize;
  const bool outputResized = known(plan.outputSize) && known(m_outputSize) && plan.outputSize != m_outputSize;
  if (known(plan.efbSize)) {
    m_efbSize = plan.efbSize;
  }
  if (known(plan.outputSize)) {
    m_outputSize = plan.outputSize;
  }
  const auto stale = [&](SizeClass sizeClass) {
    return (efbResized && sizeClass == SizeClass::Efb) || (outputResized && sizeClass == SizeClass::Output);
  };
  if (efbResized || outputResized) {
    std::erase_if(m_physicals, [&](const std::unique_ptr<Physical>& physical) {
      if (physical->claimed || !stale(size_class_of(*physical))) {
        return false;
      }
      release(*physical, released);
      return true;
    });
    std::erase_if(m_idleHistories, [&](const HistoryPair& pair) {
      if (!stale(pair.textures[0].key.sizeClass)) {
        return false;
      }
      release(pair, released);
      return true;
    });
  }

  m_graph = graph;
  publish();
  notify(released);
}

const PhysicalTexture& TransientPool::texture(SlotId slot) const {
  static const PhysicalTexture Missing;
  const bool valid = slot < m_slots.size() && !m_slots[slot]->isBuffer;
  CHECK(valid, "transient pool: slot {} is not a materialized texture", slot);
  return valid ? m_slots[slot]->texture : Missing;
}

const PhysicalBuffer& TransientPool::buffer(SlotId slot) const {
  static const PhysicalBuffer Missing;
  const bool valid = slot < m_slots.size() && m_slots[slot]->isBuffer;
  CHECK(valid, "transient pool: slot {} is not a materialized buffer", slot);
  return valid ? m_slots[slot]->buffer : Missing;
}

const PhysicalTexture& TransientPool::history_write(uint16_t historySlot) const {
  static const PhysicalTexture Missing;
  const bool valid = historySlot < m_histories.size();
  CHECK(valid, "transient pool: history {} is not materialized", historySlot);
  return valid ? m_histories[historySlot].textures[m_histories[historySlot].read ^ 1] : Missing;
}

const PhysicalTexture& TransientPool::history_prev(uint16_t historySlot) const {
  static const PhysicalTexture Missing;
  const bool valid = historySlot < m_histories.size();
  CHECK(valid, "transient pool: history {} is not materialized", historySlot);
  return valid ? m_histories[historySlot].textures[m_histories[historySlot].read] : Missing;
}

void TransientPool::mark_history_written(uint16_t historySlot) noexcept {
  if (historySlot < m_histories.size()) {
    m_histories[historySlot].pendingSwap = true;
  }
}

void TransientPool::apply_history_swaps() noexcept {
  for (HistoryPair& pair : m_histories) {
    pair.fresh = std::exchange(pair.pendingSwap, false);
    if (pair.fresh) {
      pair.read ^= 1;
    }
  }
}

bool TransientPool::history_fresh(uint16_t historySlot) const noexcept {
  return historySlot < m_histories.size() && m_histories[historySlot].fresh;
}

std::vector<uint64_t> TransientPool::serials() const {
  std::vector<uint64_t> out;
  out.reserve(m_physicals.size() + 2 * (m_histories.size() + m_idleHistories.size()));
  for (const auto& physical : m_physicals) {
    out.push_back(serial_of(*physical));
  }
  for (const auto* pairs : {&m_histories, &m_idleHistories}) {
    for (const HistoryPair& pair : *pairs) {
      out.push_back(pair.textures[0].serial);
      out.push_back(pair.textures[1].serial);
    }
  }
  std::ranges::sort(out);
  return out;
}

void TransientPool::collect(Clock::time_point now, bool pressure) {
  std::vector<uint64_t> released;
  const auto expired = [&](Clock::time_point idleSince) { return pressure || now - idleSince >= IdleRelease; };
  std::erase_if(m_physicals, [&](const std::unique_ptr<Physical>& physical) {
    if (physical->claimed || !expired(physical->idleSince)) {
      return false;
    }
    release(*physical, released);
    return true;
  });
  std::erase_if(m_idleHistories, [&](const HistoryPair& pair) {
    if (!expired(pair.idleSince)) {
      return false;
    }
    release(pair, released);
    return true;
  });
  publish();
  notify(released);
}

void TransientPool::set_release_listener(ReleaseListener listener) noexcept {
  m_listener.store(listener, std::memory_order_release);
}

PoolStats TransientPool::stats() const noexcept {
  const auto& in = m_published;
  return {
      .claimedBytes = in.claimedBytes.load(std::memory_order_relaxed),
      .idleBytes = in.idleBytes.load(std::memory_order_relaxed),
      .historyBytes = in.historyBytes.load(std::memory_order_relaxed),
      .logicalBytes = in.logicalBytes.load(std::memory_order_relaxed),
      .slotBytes = in.slotBytes.load(std::memory_order_relaxed),
      .textures = in.textures.load(std::memory_order_relaxed),
      .buffers = in.buffers.load(std::memory_order_relaxed),
      .idle = in.idle.load(std::memory_order_relaxed),
      .created = in.created.load(std::memory_order_relaxed),
      .released = in.released.load(std::memory_order_relaxed),
      .unpoisonable = g_unpoisonable.load(std::memory_order_relaxed),
  };
}

uint64_t TransientPool::total_bytes() const noexcept {
  const auto& in = m_published;
  return in.claimedBytes.load(std::memory_order_relaxed) + in.idleBytes.load(std::memory_order_relaxed) +
         in.historyBytes.load(std::memory_order_relaxed);
}

const CompiledGraph* TransientPool::materialized_graph() const noexcept { return m_graph.get(); }

bool TransientPool::empty() const noexcept {
  return m_physicals.empty() && m_histories.empty() && m_idleHistories.empty();
}

void TransientPool::clear() {
  std::vector<uint64_t> released;
  for (const auto& physical : m_physicals) {
    release(*physical, released);
  }
  for (const HistoryPair& pair : m_histories) {
    release(pair, released);
  }
  for (const HistoryPair& pair : m_idleHistories) {
    release(pair, released);
  }
  m_physicals.clear();
  m_slots.clear();
  m_histories.clear();
  m_idleHistories.clear();
  m_graph.reset();
  m_efbSize = {};
  m_outputSize = {};
  reset_poison_state();
  publish();
  notify(released);
}

TransientPool::Physical& TransientPool::create_physical(const SlotPlan::Slot& slot) {
  auto physical = std::make_unique<Physical>();
  physical->isBuffer = slot.isBuffer;
  const std::string label = "graph:" + slot.anchor;
  if (slot.isBuffer) {
    physical->buffer = m_allocator->create_buffer(slot.buf, label);
  } else {
    physical->texture = m_allocator->create_texture(slot.tex, label);
  }
  ++m_created;
  return *m_physicals.emplace_back(std::move(physical));
}

void TransientPool::release(const Physical& physical, std::vector<uint64_t>& released) {
  released.push_back(serial_of(physical));
  ++m_released;
}

void TransientPool::release(const HistoryPair& pair, std::vector<uint64_t>& released) {
  for (const PhysicalTexture& texture : pair.textures) {
    released.push_back(texture.serial);
    ++m_released;
  }
}

void TransientPool::notify(const std::vector<uint64_t>& released) const {
  const ReleaseListener listener = m_listener.load(std::memory_order_acquire);
  if (listener == nullptr) {
    return;
  }
  for (const uint64_t serial : released) {
    listener(serial);
  }
}

void TransientPool::publish() noexcept {
  uint64_t claimedBytes = 0;
  uint64_t idleBytes = 0;
  uint64_t historyBytes = 0;
  uint32_t textures = 0;
  uint32_t buffers = 0;
  uint32_t idle = 0;
  for (const auto& physical : m_physicals) {
    const uint64_t bytes = physical->isBuffer ? physical->buffer.bytes : physical->texture.bytes;
    (physical->claimed ? claimedBytes : idleBytes) += bytes;
    ++(physical->isBuffer ? buffers : textures);
    idle += physical->claimed ? 0 : 1;
  }
  for (const HistoryPair& pair : m_histories) {
    historyBytes += pair.textures[0].bytes + pair.textures[1].bytes;
    textures += 2;
  }
  for (const HistoryPair& pair : m_idleHistories) {
    idleBytes += pair.textures[0].bytes + pair.textures[1].bytes;
    textures += 2;
    idle += 2;
  }
  auto& out = m_published;
  out.claimedBytes.store(claimedBytes, std::memory_order_relaxed);
  out.idleBytes.store(idleBytes, std::memory_order_relaxed);
  out.historyBytes.store(historyBytes, std::memory_order_relaxed);
  out.logicalBytes.store(m_graph != nullptr ? m_graph->slots.logicalBytes : 0, std::memory_order_relaxed);
  out.slotBytes.store(m_graph != nullptr ? m_graph->slots.slotBytes : 0, std::memory_order_relaxed);
  out.textures.store(textures, std::memory_order_relaxed);
  out.buffers.store(buffers, std::memory_order_relaxed);
  out.idle.store(idle, std::memory_order_relaxed);
  out.created.store(m_created, std::memory_order_relaxed);
  out.released.store(m_released, std::memory_order_relaxed);
}

TransientPool& transient_pool() {
  static TransientPool pool{make_dawn_allocator()};
  return pool;
}

void transient_pool_tick(const std::shared_ptr<const CompiledGraph>& graph, TransientPool::Clock::time_point now) {
  TransientPool& pool = transient_pool();
  const bool emptyPlan = graph == nullptr || (graph->slots.slots.empty() && graph->slots.histories.empty());
  if (pool.empty() && emptyPlan) {
    return;
  }
  pool.materialize(graph, now);
  pool.apply_history_swaps();
  pool.collect(now, memory_pressure(pool.total_bytes() + gpu_heap::total_bytes()));
}

uint64_t transient_pool_bytes() noexcept { return transient_pool().total_bytes(); }

uint64_t memory_budget() noexcept {
  const Caps& deviceCaps = caps();
  if (deviceCaps.vramSource == VramSource::Unknown) {
    return 1024 * MiB;
  }
  return std::clamp<uint64_t>(deviceCaps.vramBytes / 4, 512 * MiB, 2048 * MiB);
}

bool memory_pressure(uint64_t rendererBytes) noexcept {
  return g_forcedPressure.exchange(false, std::memory_order_relaxed) || rendererBytes > memory_budget();
}

void force_memory_pressure() noexcept { g_forcedPressure.store(true, std::memory_order_relaxed); }

bool debug_poison_enabled() noexcept {
  const int value = g_poison.load(std::memory_order_relaxed);
  return value < 0 ? default_poison() : value != 0;
}

void set_debug_poison(bool enabled) noexcept { g_poison.store(enabled ? 1 : 0, std::memory_order_relaxed); }

void Poisoner::encode(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture) {
  if (!texture.texture) {
    return;
  }
  switch (poison_method(texture.key)) {
  case PoisonMethod::RenderClear:
    render_clear(cmd, texture);
    break;
  case PoisonMethod::ComputeClear:
    compute_clear(cmd, texture);
    break;
  case PoisonMethod::BufferCopy:
    copy_clear(cmd, texture);
    break;
  default:
    break;
  }
}

void Poisoner::encode(const wgpu::CommandEncoder& cmd, const PhysicalBuffer& buffer, uint64_t size) {
  const uint64_t bytes = std::min(size, buffer.bytes) & ~uint64_t{3};
  if (!buffer.buffer || bytes == 0) {
    return;
  }
  switch (poison_method(buffer.key)) {
  case PoisonMethod::BufferCopy:
    cmd.CopyBufferToBuffer(source(bytes), 0, buffer.buffer, 0, bytes);
    break;
  case PoisonMethod::BufferFill:
    fill(cmd, buffer.buffer, bytes);
    break;
  default:
    break;
  }
}

const wgpu::Buffer& Poisoner::source(uint64_t size) {
  size = (size + 3) & ~uint64_t{3};
  if (m_source && m_sourceSize >= size) {
    return m_source;
  }
  const wgpu::BufferDescriptor descriptor{
      .label = "graph poison source",
      .usage = wgpu::BufferUsage::CopySrc,
      .size = size,
      .mappedAtCreation = true,
  };
  m_source = webgpu::g_device.CreateBuffer(&descriptor);
  if (void* data = m_source.GetMappedRange(0, size)) {
    std::memset(data, 0xFF, size);
  }
  m_source.Unmap();
  m_sourceSize = size;
  return m_source;
}

const Poisoner::ClearKernel* Poisoner::clear_kernel(wgpu::TextureFormat format, wgpu::TextureViewDimension dimension) {
  for (const ClearKernel& kernel : m_kernels) {
    if (kernel.format == format && kernel.dimension == dimension) {
      return &kernel;
    }
  }
  const std::string_view texel = wgsl_texel_format(format);
  const auto info = texture_format_info(format);
  if (texel.empty() || !info) {
    return nullptr;
  }
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute,
      .storageTexture =
          wgpu::StorageTextureBindingLayout{
              .access = wgpu::StorageTextureAccess::WriteOnly,
              .format = format,
              .viewDimension = dimension,
          },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "graph poison",
      .entryCount = 1,
      .entries = &entry,
  };
  ClearKernel& kernel = m_kernels.emplace_back(ClearKernel{
      .format = format,
      .dimension = dimension,
      .layout = webgpu::g_device.CreateBindGroupLayout(&layoutDescriptor),
  });
  kernel.pipeline = create_pipeline(
      kernel.layout, clear_kernel_source(texel, dimension, poison_color(format, *info), info->sampleType),
      "graph poison clear");
  return &kernel;
}

void Poisoner::render_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture) const {
  const TexKey& key = texture.key;
  const auto info = texture_format_info(key.format);
  if (!info) {
    return;
  }
  const bool volume = key.dimension == wgpu::TextureDimension::e3D;
  const wgpu::Color color = poison_color(key.format, *info);
  for (uint32_t mip = 0; mip < key.mips; ++mip) {
    const uint32_t layers = volume ? std::max(1u, key.depthOrLayers >> mip) : key.depthOrLayers;
    for (uint32_t layer = 0; layer < layers; ++layer) {
      const wgpu::TextureView& view = texture.attachment_view(mip, layer);
      wgpu::RenderPassColorAttachment colorAttachment{};
      wgpu::RenderPassDepthStencilAttachment depthStencil{};
      wgpu::RenderPassDescriptor descriptor{
          .label = "graph poison",
          .colorAttachmentCount = 0,
      };
      if (info->depth || info->stencil) {
        depthStencil.view = view;
        if (info->depth) {
          depthStencil.depthLoadOp = wgpu::LoadOp::Clear;
          depthStencil.depthStoreOp = wgpu::StoreOp::Store;
          depthStencil.depthClearValue = 0.5f;
        }
        if (info->stencil) {
          depthStencil.stencilLoadOp = wgpu::LoadOp::Clear;
          depthStencil.stencilStoreOp = wgpu::StoreOp::Store;
          depthStencil.stencilClearValue = 0xBA;
        }
        descriptor.depthStencilAttachment = &depthStencil;
      } else {
        colorAttachment = {
            .view = view,
            .depthSlice = volume ? layer : wgpu::kDepthSliceUndefined,
            .loadOp = wgpu::LoadOp::Clear,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue = color,
        };
        descriptor.colorAttachmentCount = 1;
        descriptor.colorAttachments = &colorAttachment;
      }
      cmd.BeginRenderPass(&descriptor).End();
    }
  }
}

void Poisoner::compute_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture) {
  const TexKey& key = texture.key;
  const bool volume = key.dimension == wgpu::TextureDimension::e3D;
  // Storage views cannot be cubes, so the clear walks the texture's own layout
  const auto dimension = volume                  ? wgpu::TextureViewDimension::e3D
                         : key.depthOrLayers > 1 ? wgpu::TextureViewDimension::e2DArray
                                                 : wgpu::TextureViewDimension::e2D;
  const ClearKernel* kernel = clear_kernel(key.format, dimension);
  if (kernel == nullptr) {
    return;
  }
  for (uint32_t mip = 0; mip < key.mips; ++mip) {
    const wgpu::BindGroup group =
        create_group(kernel->layout, {.binding = 0, .textureView = texture.mip_view(mip, dimension)});
    const uint32_t width = std::max(1u, key.width >> mip);
    const uint32_t height = std::max(1u, key.height >> mip);
    const uint32_t depth = volume ? std::max(1u, key.depthOrLayers >> mip) : key.depthOrLayers;
    const wgpu::ComputePassDescriptor passDescriptor{.label = "graph poison"};
    const wgpu::ComputePassEncoder pass = cmd.BeginComputePass(&passDescriptor);
    pass.SetPipeline(kernel->pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups((width + 7) / 8, (height + 7) / 8, depth);
    pass.End();
  }
}

void Poisoner::copy_clear(const wgpu::CommandEncoder& cmd, const PhysicalTexture& texture) {
  const TexKey& key = texture.key;
  const auto info = texture_format_info(key.format);
  if (!info) {
    return;
  }
  const bool volume = key.dimension == wgpu::TextureDimension::e3D;
  for (uint32_t mip = 0; mip < key.mips; ++mip) {
    const auto blocksWide = static_cast<uint32_t>(ceil_div(std::max(1u, key.width >> mip), info->blockWidth));
    const auto blocksHigh = static_cast<uint32_t>(ceil_div(std::max(1u, key.height >> mip), info->blockHeight));
    const uint32_t bytesPerRow = (blocksWide * info->bytesPerBlock + 255) & ~255u;
    const uint32_t depth = volume ? std::max(1u, key.depthOrLayers >> mip) : 1;
    const wgpu::TexelCopyBufferInfo from{
        .layout = {.bytesPerRow = bytesPerRow, .rowsPerImage = blocksHigh},
        .buffer = source(uint64_t{bytesPerRow} * blocksHigh * depth),
    };
    const wgpu::Extent3D size{blocksWide * info->blockWidth, blocksHigh * info->blockHeight, depth};
    const uint32_t layers = volume ? 1 : key.depthOrLayers;
    for (uint32_t layer = 0; layer < layers; ++layer) {
      const wgpu::TexelCopyTextureInfo to{
          .texture = texture.texture,
          .mipLevel = mip,
          .origin = {0, 0, layer},
      };
      cmd.CopyBufferToTexture(&from, &to, &size);
    }
  }
}

void Poisoner::fill(const wgpu::CommandEncoder& cmd, const wgpu::Buffer& buffer, uint64_t size) {
  if (!m_fill) {
    const wgpu::BindGroupLayoutEntry entry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Storage},
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "graph poison",
        .entryCount = 1,
        .entries = &entry,
    };
    m_fillLayout = webgpu::g_device.CreateBindGroupLayout(&layoutDescriptor);
    m_fill = create_pipeline(m_fillLayout, std::string{FillKernelSource}, "graph poison fill");
  }
  const wgpu::BindGroup group = create_group(m_fillLayout, {.binding = 0, .buffer = buffer, .size = size});
  const uint64_t groups = ceil_div(size / 4, 64);
  const auto x = static_cast<uint32_t>(std::min<uint64_t>(groups, MaxWorkgroups));
  const auto y = static_cast<uint32_t>(ceil_div(groups, x));
  const wgpu::ComputePassDescriptor passDescriptor{.label = "graph poison"};
  const wgpu::ComputePassEncoder pass = cmd.BeginComputePass(&passDescriptor);
  pass.SetPipeline(m_fill);
  pass.SetBindGroup(0, group);
  pass.DispatchWorkgroups(x, y, 1);
  pass.End();
}

void poison_before(const wgpu::CommandEncoder& cmd, const StageData& stage, NodeId node) {
  if (!cmd || !debug_poison_enabled() || stage.frame == nullptr || stage.frame->graph == nullptr) {
    return;
  }
  const CompiledGraph& graph = *stage.frame->graph;
  TransientPool& pool = transient_pool();
  if (pool.materialized_graph() != &graph) {
    return;
  }
  PoisonState& state = g_poisonState;
  if (state.graph != &graph || state.frameId != stage.frameId) {
    state.graph = &graph;
    state.frameId = stage.frameId;
    state.cursor = 0;
    state.historyDone.assign(graph.histories.size(), false);
  }
  const SlotPlan& plan = graph.slots;
  const uint32_t position = node < plan.positions.size() ? plan.positions[node] : UINT32_MAX;
  // A slot's earlier occupant is dead once a later lifetime starts, so nothing live is overwritten
  while (state.cursor < plan.poisonPoints.size() && plan.poisonPoints[state.cursor].first <= position) {
    const ResourceId resource = plan.poisonPoints[state.cursor++].second;
    const SlotId slot = plan.resourceSlot[resource];
    const std::string& name = graph.resources[resource].name;
    if (plan.slots[slot].isBuffer) {
      g_poisonState.poisoner.encode(cmd, pool.buffer(slot), pool.buffer(slot).bytes);
    } else {
      poison_texture(cmd, pool.texture(slot), name);
    }
  }
  for (const CompiledUse& use : graph.node_uses(node)) {
    const CompiledResource& resource = graph.resources[use.resource];
    const uint16_t history = resource.historySlot;
    const bool valid = stage.prev_valid(history) && pool.history_fresh(history);
    if (use.role != UseRole::History || history >= state.historyDone.size() || valid || state.historyDone[history]) {
      continue;
    }
    state.historyDone[history] = true;
    poison_texture(cmd, pool.history_prev(history), resource.name);
  }
}

void after_pass(const StageData& stage, NodeId node) {
  if (stage.frame == nullptr || stage.frame->graph == nullptr) {
    return;
  }
  const CompiledGraph& graph = *stage.frame->graph;
  TransientPool& pool = transient_pool();
  if (pool.materialized_graph() != &graph) {
    return;
  }
  for (const CompiledUse& use : graph.node_uses(node)) {
    if (use.role == UseRole::History) {
      pool.mark_history_written(graph.resources[use.resource].historySlot);
    }
  }
}

namespace testing {
void reset_transient_pool(std::unique_ptr<PhysicalAllocator> allocator) {
  TransientPool& pool = transient_pool();
  pool.m_listener.store(nullptr, std::memory_order_release);
  pool.clear();
  pool.m_allocator = allocator != nullptr ? std::move(allocator) : make_dawn_allocator();
  pool.m_created = 0;
  pool.m_released = 0;
  pool.publish();
  g_forcedPressure.store(false, std::memory_order_relaxed);
  g_poison.store(-1, std::memory_order_relaxed);
  g_unpoisonable.store(0, std::memory_order_relaxed);
}
} // namespace testing
} // namespace aurora::render::detail

#include "graph_slots.hpp"

#include "../internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render::graph"};

using wgpu::TextureFormat;
using wgpu::TextureSampleType;

struct FormatRow {
  TextureFormat format;
  FormatInfo info;
};

constexpr FormatRow colour(TextureFormat format, uint8_t bytes, TextureSampleType type = TextureSampleType::Float) {
  return {format, {.bytesPerBlock = bytes, .sampleType = type}};
}

constexpr FormatRow depth_stencil(TextureFormat format, uint8_t bytes, bool depth, bool stencil) {
  return {format,
          {
              .bytesPerBlock = bytes,
              .depth = depth,
              .stencil = stencil,
              .sampleType = depth ? TextureSampleType::Depth : TextureSampleType::Uint,
          }};
}

constexpr FormatRow block(TextureFormat format, uint8_t width, uint8_t height, uint8_t bytes) {
  return {format, {.blockWidth = width, .blockHeight = height, .bytesPerBlock = bytes, .compressed = true}};
}

constexpr auto Uint = TextureSampleType::Uint;
constexpr auto Sint = TextureSampleType::Sint;
constexpr auto Unfilterable = TextureSampleType::UnfilterableFloat;

constexpr std::array FormatRows{
    colour(TextureFormat::R8Unorm, 1),
    colour(TextureFormat::R8Snorm, 1),
    colour(TextureFormat::R8Uint, 1, Uint),
    colour(TextureFormat::R8Sint, 1, Sint),
    colour(TextureFormat::R16Unorm, 2),
    colour(TextureFormat::R16Snorm, 2),
    colour(TextureFormat::R16Uint, 2, Uint),
    colour(TextureFormat::R16Sint, 2, Sint),
    colour(TextureFormat::R16Float, 2),
    colour(TextureFormat::RG8Unorm, 2),
    colour(TextureFormat::RG8Snorm, 2),
    colour(TextureFormat::RG8Uint, 2, Uint),
    colour(TextureFormat::RG8Sint, 2, Sint),
    colour(TextureFormat::R32Float, 4, Unfilterable),
    colour(TextureFormat::R32Uint, 4, Uint),
    colour(TextureFormat::R32Sint, 4, Sint),
    colour(TextureFormat::RG16Unorm, 4),
    colour(TextureFormat::RG16Snorm, 4),
    colour(TextureFormat::RG16Uint, 4, Uint),
    colour(TextureFormat::RG16Sint, 4, Sint),
    colour(TextureFormat::RG16Float, 4),
    colour(TextureFormat::RGBA8Unorm, 4),
    colour(TextureFormat::RGBA8UnormSrgb, 4),
    colour(TextureFormat::RGBA8Snorm, 4),
    colour(TextureFormat::RGBA8Uint, 4, Uint),
    colour(TextureFormat::RGBA8Sint, 4, Sint),
    colour(TextureFormat::BGRA8Unorm, 4),
    colour(TextureFormat::BGRA8UnormSrgb, 4),
    colour(TextureFormat::RGB10A2Uint, 4, Uint),
    colour(TextureFormat::RGB10A2Unorm, 4),
    colour(TextureFormat::RG11B10Ufloat, 4),
    colour(TextureFormat::RGB9E5Ufloat, 4),
    colour(TextureFormat::RG32Float, 8, Unfilterable),
    colour(TextureFormat::RG32Uint, 8, Uint),
    colour(TextureFormat::RG32Sint, 8, Sint),
    colour(TextureFormat::RGBA16Unorm, 8),
    colour(TextureFormat::RGBA16Snorm, 8),
    colour(TextureFormat::RGBA16Uint, 8, Uint),
    colour(TextureFormat::RGBA16Sint, 8, Sint),
    colour(TextureFormat::RGBA16Float, 8),
    colour(TextureFormat::RGBA32Float, 16, Unfilterable),
    colour(TextureFormat::RGBA32Uint, 16, Uint),
    colour(TextureFormat::RGBA32Sint, 16, Sint),
    depth_stencil(TextureFormat::Stencil8, 1, false, true),
    depth_stencil(TextureFormat::Depth16Unorm, 2, true, false),
    depth_stencil(TextureFormat::Depth24Plus, 4, true, false),
    // Packed sizes of D24S8 and D32S8; drivers may pad them
    depth_stencil(TextureFormat::Depth24PlusStencil8, 4, true, true),
    depth_stencil(TextureFormat::Depth32Float, 4, true, false),
    depth_stencil(TextureFormat::Depth32FloatStencil8, 8, true, true),
    block(TextureFormat::BC1RGBAUnorm, 4, 4, 8),
    block(TextureFormat::BC1RGBAUnormSrgb, 4, 4, 8),
    block(TextureFormat::BC2RGBAUnorm, 4, 4, 16),
    block(TextureFormat::BC2RGBAUnormSrgb, 4, 4, 16),
    block(TextureFormat::BC3RGBAUnorm, 4, 4, 16),
    block(TextureFormat::BC3RGBAUnormSrgb, 4, 4, 16),
    block(TextureFormat::BC4RUnorm, 4, 4, 8),
    block(TextureFormat::BC4RSnorm, 4, 4, 8),
    block(TextureFormat::BC5RGUnorm, 4, 4, 16),
    block(TextureFormat::BC5RGSnorm, 4, 4, 16),
    block(TextureFormat::BC6HRGBUfloat, 4, 4, 16),
    block(TextureFormat::BC6HRGBFloat, 4, 4, 16),
    block(TextureFormat::BC7RGBAUnorm, 4, 4, 16),
    block(TextureFormat::BC7RGBAUnormSrgb, 4, 4, 16),
    block(TextureFormat::ETC2RGB8Unorm, 4, 4, 8),
    block(TextureFormat::ETC2RGB8UnormSrgb, 4, 4, 8),
    block(TextureFormat::ETC2RGB8A1Unorm, 4, 4, 8),
    block(TextureFormat::ETC2RGB8A1UnormSrgb, 4, 4, 8),
    block(TextureFormat::ETC2RGBA8Unorm, 4, 4, 16),
    block(TextureFormat::ETC2RGBA8UnormSrgb, 4, 4, 16),
    block(TextureFormat::EACR11Unorm, 4, 4, 8),
    block(TextureFormat::EACR11Snorm, 4, 4, 8),
    block(TextureFormat::EACRG11Unorm, 4, 4, 16),
    block(TextureFormat::EACRG11Snorm, 4, 4, 16),
    block(TextureFormat::ASTC4x4Unorm, 4, 4, 16),
    block(TextureFormat::ASTC4x4UnormSrgb, 4, 4, 16),
    block(TextureFormat::ASTC5x4Unorm, 5, 4, 16),
    block(TextureFormat::ASTC5x4UnormSrgb, 5, 4, 16),
    block(TextureFormat::ASTC5x5Unorm, 5, 5, 16),
    block(TextureFormat::ASTC5x5UnormSrgb, 5, 5, 16),
    block(TextureFormat::ASTC6x5Unorm, 6, 5, 16),
    block(TextureFormat::ASTC6x5UnormSrgb, 6, 5, 16),
    block(TextureFormat::ASTC6x6Unorm, 6, 6, 16),
    block(TextureFormat::ASTC6x6UnormSrgb, 6, 6, 16),
    block(TextureFormat::ASTC8x5Unorm, 8, 5, 16),
    block(TextureFormat::ASTC8x5UnormSrgb, 8, 5, 16),
    block(TextureFormat::ASTC8x6Unorm, 8, 6, 16),
    block(TextureFormat::ASTC8x6UnormSrgb, 8, 6, 16),
    block(TextureFormat::ASTC8x8Unorm, 8, 8, 16),
    block(TextureFormat::ASTC8x8UnormSrgb, 8, 8, 16),
    block(TextureFormat::ASTC10x5Unorm, 10, 5, 16),
    block(TextureFormat::ASTC10x5UnormSrgb, 10, 5, 16),
    block(TextureFormat::ASTC10x6Unorm, 10, 6, 16),
    block(TextureFormat::ASTC10x6UnormSrgb, 10, 6, 16),
    block(TextureFormat::ASTC10x8Unorm, 10, 8, 16),
    block(TextureFormat::ASTC10x8UnormSrgb, 10, 8, 16),
    block(TextureFormat::ASTC10x10Unorm, 10, 10, 16),
    block(TextureFormat::ASTC10x10UnormSrgb, 10, 10, 16),
    block(TextureFormat::ASTC12x10Unorm, 12, 10, 16),
    block(TextureFormat::ASTC12x10UnormSrgb, 12, 10, 16),
    block(TextureFormat::ASTC12x12Unorm, 12, 12, 16),
    block(TextureFormat::ASTC12x12UnormSrgb, 12, 12, 16),
};

std::mutex g_formatMutex;
std::unordered_set<uint32_t> g_warnedFormats;

void warn_unknown_format(TextureFormat format) noexcept {
  {
    std::lock_guard lock{g_formatMutex};
    if (!g_warnedFormats.insert(static_cast<uint32_t>(format)).second) {
      return;
    }
  }
  Log.warn("render graph: texture format {} has no size; counted as 0 bytes", static_cast<uint32_t>(format));
}

bool same_key(const SlotPlan::Slot& a, const SlotPlan::Slot& b) noexcept {
  return a.isBuffer == b.isBuffer && (a.isBuffer ? a.buf == b.buf : a.tex == b.tex);
}

uint64_t slot_bytes(const SlotPlan::Slot& slot) noexcept {
  return slot.isBuffer ? key_bytes(slot.buf) : key_bytes(slot.tex);
}
} // namespace

std::optional<FormatInfo> texture_format_info(wgpu::TextureFormat format) noexcept {
  const auto it = std::ranges::find(FormatRows, format, &FormatRow::format);
  return it != FormatRows.end() ? std::optional{it->info} : std::nullopt;
}

wgpu::TextureViewDimension binding_view_dimension(const CompiledGraph& graph, ResourceId resource) noexcept {
  for (const StageProgram& program : graph.stages) {
    for (const NodeId node : program.nodes) {
      for (const CompiledUse& use : graph.node_uses(node)) {
        if (use.resource == resource && binds(use.access) && !use.staticDummy &&
            use.options.dimension != wgpu::TextureViewDimension::Undefined) {
          return use.options.dimension;
        }
      }
    }
  }
  const CompiledResource& compiled = graph.resources[resource];
  if (compiled.tex.dimension == wgpu::TextureDimension::e3D) {
    return wgpu::TextureViewDimension::e3D;
  }
  return compiled.extent.depthOrArrayLayers > 1 ? wgpu::TextureViewDimension::e2DArray
                                                : wgpu::TextureViewDimension::e2D;
}

TexKey make_tex_key(const CompiledGraph& graph, ResourceId resource) noexcept {
  const CompiledResource& compiled = graph.resources[resource];
  CHECK(compiled.mipLevels >= 1, "render graph: '{}' has no mip levels", compiled.name);
  return {
      .format = compiled.tex.format,
      .dimension = compiled.tex.dimension,
      .bindingView = binding_view_dimension(graph, resource),
      .width = compiled.extent.width,
      .height = compiled.extent.height,
      .depthOrLayers = compiled.extent.depthOrArrayLayers,
      .mips = compiled.mipLevels,
      .usage = compiled.textureUsage,
      .sizeClass = compiled.tex.sizeClass,
  };
}

BufKey make_buf_key(const CompiledResource& resource, const GraphLimits& limits) noexcept {
  uint64_t limit = limits.bufferSize;
  if (resource.bufferUsage & wgpu::BufferUsage::Storage) {
    limit = std::min(limit, limits.storageBufferBindingSize);
  }
  if (resource.bufferUsage & wgpu::BufferUsage::Uniform) {
    limit = std::min(limit, limits.uniformBufferBindingSize);
  }
  return {
      .bucketSize = buffer_bucket(resource.bufferSize, limit),
      .usage = resource.bufferUsage,
      .sizeClass = resource.buf.sizeClass,
  };
}

uint64_t buffer_bucket(uint64_t size, uint64_t limit) noexcept {
  const uint64_t bucket = std::bit_ceil(std::max<uint64_t>(size, 256));
  // W6 checked the exact size, and a device limit need not be a power of two
  return bucket <= limit ? bucket : size;
}

uint64_t key_bytes(const TexKey& key) noexcept {
  const auto info = texture_format_info(key.format);
  if (!info) {
    warn_unknown_format(key.format);
    return 0;
  }
  uint64_t bytes = 0;
  for (uint32_t mip = 0; mip < key.mips && mip < 32; ++mip) {
    const uint32_t width = std::max(1u, key.width >> mip);
    const uint32_t height = std::max(1u, key.height >> mip);
    const uint32_t depth =
        key.dimension == wgpu::TextureDimension::e3D ? std::max(1u, key.depthOrLayers >> mip) : key.depthOrLayers;
    bytes += ceil_div(width, info->blockWidth) * ceil_div(height, info->blockHeight) * info->bytesPerBlock * depth;
  }
  return bytes;
}

uint64_t key_bytes(const BufKey& key) noexcept { return key.bucketSize; }

SlotPlan plan_slots(const CompiledGraph& graph, const GraphLimits& limits) {
  SlotPlan plan;
  plan.efbSize = graph.efbSize;
  plan.outputSize = graph.outputSize;
  plan.positions.assign(graph.nodes.size(), UINT32_MAX);
  uint32_t position = 0;
  for (const StageProgram& program : graph.stages) {
    for (const NodeId node : program.nodes) {
      plan.positions[node] = position++;
    }
  }

  plan.resourceSlot.assign(graph.resources.size(), InvalidSlot);
  plan.lifetimes.assign(graph.resources.size(), {});
  for (const CompiledUse& use : graph.uses) {
    const CompiledResource& resource = graph.resources[use.resource];
    const uint32_t at = plan.positions[use.node];
    if (resource.kind != ResourceKind::Transient || !resource.described || at == UINT32_MAX || use.staticDummy ||
        use.role == UseRole::Prev) {
      continue;
    }
    Lifetime& lifetime = plan.lifetimes[use.resource];
    lifetime.first = std::min(lifetime.first, at);
    lifetime.last = std::max(lifetime.last, at);
  }

  std::vector<ResourceId> transients;
  for (size_t r = 0; r < graph.resources.size(); ++r) {
    if (plan.lifetimes[r].first != UINT32_MAX) {
      transients.push_back(static_cast<ResourceId>(r));
    }
  }
  // Names, not ids, so the registration order cannot change the plan
  std::ranges::sort(transients, {}, [&](ResourceId r) {
    return std::pair{plan.lifetimes[r].first, std::string_view{graph.resources[r].name}};
  });

  // Greedy colouring in order of first use is optimal for intervals: per key, slots = the peak overlap
  std::vector<uint32_t> slotEnds;
  for (const ResourceId r : transients) {
    const CompiledResource& resource = graph.resources[r];
    const Lifetime lifetime = plan.lifetimes[r];
    SlotPlan::Slot candidate{.isBuffer = resource.buffer};
    if (resource.buffer) {
      candidate.buf = make_buf_key(resource, limits);
    } else {
      candidate.tex = make_tex_key(graph, r);
    }
    const uint64_t bytes = slot_bytes(candidate);
    auto chosen = static_cast<SlotId>(plan.slots.size());
    for (size_t s = 0; s < plan.slots.size(); ++s) {
      if (slotEnds[s] < lifetime.first && same_key(plan.slots[s], candidate)) {
        chosen = static_cast<SlotId>(s);
        break;
      }
    }
    if (chosen == plan.slots.size()) {
      candidate.anchor = resource.name;
      plan.slots.push_back(std::move(candidate));
      slotEnds.push_back(lifetime.last);
      plan.slotBytes += bytes;
    } else {
      slotEnds[chosen] = lifetime.last;
    }
    plan.resourceSlot[r] = chosen;
    plan.logicalBytes += bytes;
    plan.poisonPoints.emplace_back(lifetime.first, r);
  }
  std::ranges::sort(plan.poisonPoints);

  plan.histories.reserve(graph.histories.size());
  for (const ResourceId r : graph.histories) {
    plan.histories.push_back({.name = graph.resources[r].name, .key = make_tex_key(graph, r)});
  }
  return plan;
}

PoisonMethod poison_method(const TexKey& key) noexcept {
  if (key.usage & wgpu::TextureUsage::RenderAttachment) {
    return PoisonMethod::RenderClear;
  }
  if (key.usage & wgpu::TextureUsage::StorageBinding) {
    return PoisonMethod::ComputeClear;
  }
  const auto info = texture_format_info(key.format);
  if ((key.usage & wgpu::TextureUsage::CopyDst) && info && !info->depth && !info->stencil) {
    return PoisonMethod::BufferCopy;
  }
  return PoisonMethod::None;
}

PoisonMethod poison_method(const BufKey& key) noexcept {
  if (key.usage & wgpu::BufferUsage::CopyDst) {
    return PoisonMethod::BufferCopy;
  }
  if (key.usage & wgpu::BufferUsage::Storage) {
    return PoisonMethod::BufferFill;
  }
  return PoisonMethod::None;
}
} // namespace aurora::render::detail

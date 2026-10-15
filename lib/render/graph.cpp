#include "graph.hpp"

#include "../gfx/hash.hpp"
#include "../logging.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aurora::render::detail {
namespace {
constexpr Module Log{"aurora::render::graph"};

static_assert(StageCount == 7 && static_cast<size_t>(Stage::AfterFrame) == 6);

constexpr std::array ChainNodes{
#define AURORA_RENDER_NODE(stage, kind, name) ChainNode{Stage::stage, ChainKind::kind, name},
#include "phases.inl"
#undef AURORA_RENDER_NODE
};
static_assert(ChainNodes.size() == ChainNodeCount);

struct GxRow {
  std::string_view name;
  wgpu::TextureFormat format;
};

constexpr std::array GxRows{
#define AURORA_RENDER_GX(id, label, format) GxRow{label, wgpu::TextureFormat::format},
#include "phases.inl"
#undef AURORA_RENDER_GX
};
static_assert(GxRows.size() == GxResourceCount);
static_assert(static_cast<size_t>(GxResource::DebugId) + 1 == GxResourceCount);

struct ChainRange {
  uint16_t first = 0;
  uint16_t count = 0;
};

constexpr std::array<ChainRange, StageCount> ChainRanges = [] {
  std::array<ChainRange, StageCount> ranges{};
  for (size_t i = 0; i < ChainNodes.size(); ++i) {
    auto& range = ranges[static_cast<size_t>(ChainNodes[i].stage)];
    if (range.count == 0) {
      range.first = static_cast<uint16_t>(i);
    }
    ++range.count;
  }
  return ranges;
}();

constexpr bool chain_is_contiguous() {
  for (size_t i = 1; i < ChainNodes.size(); ++i) {
    if (ChainNodes[i].stage < ChainNodes[i - 1].stage) {
      return false;
    }
  }
  return std::ranges::all_of(ChainRanges, [](const ChainRange& range) { return range.count != 0; });
}
static_assert(chain_is_contiguous());

enum FormatCap : uint16_t {
  CapStorage = 1 << 0,
  CapStorageCore = 1 << 1,
  CapStorageTier1 = 1 << 2,
  CapStorageBgra8 = 1 << 3,
  CapReadWrite = 1 << 4,
  CapReadWriteTier2 = 1 << 5,
  CapRenderTier1 = 1 << 6,
  CapRenderRg11b10 = 1 << 7,
  CapDepth = 1 << 8,
  CapNoCopy = 1 << 9,
};

struct FormatRow {
  wgpu::TextureFormat format;
  std::string_view name;
  uint8_t colorBytes;
  uint8_t colorAlignment;
  uint16_t caps;
};

using wgpu::TextureFormat;
constexpr std::array FormatRows{
    FormatRow{TextureFormat::R8Unorm, "r8unorm", 1, 1, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::R8Snorm, "r8snorm", 1, 1, CapStorageTier1 | CapRenderTier1},
    FormatRow{TextureFormat::R8Uint, "r8uint", 1, 1, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::R8Sint, "r8sint", 1, 1, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::R16Uint, "r16uint", 2, 2, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::R16Sint, "r16sint", 2, 2, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::R16Float, "r16float", 2, 2, CapStorageTier1 | CapReadWriteTier2},
    FormatRow{TextureFormat::RG8Unorm, "rg8unorm", 2, 1, CapStorageTier1},
    FormatRow{TextureFormat::RG8Snorm, "rg8snorm", 2, 2, CapStorageTier1 | CapRenderTier1},
    FormatRow{TextureFormat::RG8Uint, "rg8uint", 2, 1, CapStorageTier1},
    FormatRow{TextureFormat::RG8Sint, "rg8sint", 2, 1, CapStorageTier1},
    FormatRow{TextureFormat::R32Uint, "r32uint", 4, 4, CapStorage | CapReadWrite},
    FormatRow{TextureFormat::R32Sint, "r32sint", 4, 4, CapStorage | CapReadWrite},
    FormatRow{TextureFormat::R32Float, "r32float", 4, 4, CapStorage | CapReadWrite},
    FormatRow{TextureFormat::RG16Uint, "rg16uint", 4, 2, CapStorageTier1},
    FormatRow{TextureFormat::RG16Sint, "rg16sint", 4, 2, CapStorageTier1},
    FormatRow{TextureFormat::RG16Float, "rg16float", 4, 2, CapStorageTier1},
    FormatRow{TextureFormat::RGBA8Unorm, "rgba8unorm", 8, 1, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA8UnormSrgb, "rgba8unorm-srgb", 8, 1, 0},
    FormatRow{TextureFormat::RGBA8Snorm, "rgba8snorm", 4, 4, CapStorage | CapRenderTier1},
    FormatRow{TextureFormat::RGBA8Uint, "rgba8uint", 4, 1, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA8Sint, "rgba8sint", 4, 1, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::BGRA8Unorm, "bgra8unorm", 8, 1, CapStorageBgra8},
    FormatRow{TextureFormat::BGRA8UnormSrgb, "bgra8unorm-srgb", 8, 1, 0},
    FormatRow{TextureFormat::RGB10A2Uint, "rgb10a2uint", 8, 4, CapStorageTier1},
    FormatRow{TextureFormat::RGB10A2Unorm, "rgb10a2unorm", 8, 4, CapStorageTier1},
    FormatRow{TextureFormat::RG11B10Ufloat, "rg11b10ufloat", 8, 4, CapStorageTier1 | CapRenderRg11b10},
    FormatRow{TextureFormat::RG32Uint, "rg32uint", 8, 4, CapStorageCore},
    FormatRow{TextureFormat::RG32Sint, "rg32sint", 8, 4, CapStorageCore},
    FormatRow{TextureFormat::RG32Float, "rg32float", 8, 4, CapStorageCore},
    FormatRow{TextureFormat::RGBA16Uint, "rgba16uint", 8, 2, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA16Sint, "rgba16sint", 8, 2, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA16Float, "rgba16float", 8, 2, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA32Uint, "rgba32uint", 16, 4, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA32Sint, "rgba32sint", 16, 4, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::RGBA32Float, "rgba32float", 16, 4, CapStorage | CapReadWriteTier2},
    FormatRow{TextureFormat::Depth16Unorm, "depth16unorm", 0, 1, CapDepth},
    FormatRow{TextureFormat::Depth24Plus, "depth24plus", 0, 1, CapDepth | CapNoCopy},
    FormatRow{TextureFormat::Depth24PlusStencil8, "depth24plus-stencil8", 0, 1, CapDepth | CapNoCopy},
    FormatRow{TextureFormat::Depth32Float, "depth32float", 0, 1, CapDepth},
};

const FormatRow* find_format(wgpu::TextureFormat format) noexcept {
  const auto it = std::ranges::find(FormatRows, format, &FormatRow::format);
  return it != FormatRows.end() ? &*it : nullptr;
}

std::string format_name(wgpu::TextureFormat format) {
  const FormatRow* row = find_format(format);
  return row != nullptr ? std::string{row->name} : fmt::format("format {}", static_cast<uint32_t>(format));
}

bool is_depth_format(wgpu::TextureFormat format) noexcept {
  const FormatRow* row = find_format(format);
  return row != nullptr && (row->caps & CapDepth) != 0;
}

wgpu::TextureSampleType format_sample_type(wgpu::TextureFormat format, bool float32Filterable) noexcept {
  if (is_depth_format(format)) {
    return wgpu::TextureSampleType::Depth;
  }
  switch (format) {
  case TextureFormat::R8Uint:
  case TextureFormat::R16Uint:
  case TextureFormat::R32Uint:
  case TextureFormat::RG8Uint:
  case TextureFormat::RG16Uint:
  case TextureFormat::RG32Uint:
  case TextureFormat::RGBA8Uint:
  case TextureFormat::RGBA16Uint:
  case TextureFormat::RGBA32Uint:
  case TextureFormat::RGB10A2Uint:
    return wgpu::TextureSampleType::Uint;
  case TextureFormat::R8Sint:
  case TextureFormat::R16Sint:
  case TextureFormat::R32Sint:
  case TextureFormat::RG8Sint:
  case TextureFormat::RG16Sint:
  case TextureFormat::RG32Sint:
  case TextureFormat::RGBA8Sint:
  case TextureFormat::RGBA16Sint:
  case TextureFormat::RGBA32Sint:
    return wgpu::TextureSampleType::Sint;
  case TextureFormat::R32Float:
  case TextureFormat::RG32Float:
  case TextureFormat::RGBA32Float:
    return float32Filterable ? wgpu::TextureSampleType::Float : wgpu::TextureSampleType::UnfilterableFloat;
  default:
    return wgpu::TextureSampleType::Float;
  }
}

constexpr size_t bit_words(size_t bits) noexcept { return (bits + 63) / 64; }

void assign_bit(std::vector<uint64_t>& bits, size_t index, bool value) noexcept {
  if (index / 64 >= bits.size()) {
    return;
  }
  const uint64_t mask = uint64_t{1} << (index % 64);
  bits[index / 64] = value ? bits[index / 64] | mask : bits[index / 64] & ~mask;
}

bool input_missing(const CompiledGraph& graph, const CompiledUse& use, Stage stage, const FrameCullState& state,
                   const StageMask& mask) noexcept {
  const CompiledResource& resource = graph.resources[use.resource];
  if (use.role == UseRole::Prev) {
    return !test_bit(state.prevValid, resource.historySlot);
  }
  // Only the creator decides: with a modifier off, readers still get the creator's valid version
  if (use.role != UseRole::Use ||
      (resource.kind != ResourceKind::Transient && resource.kind != ResourceKind::History)) {
    return false;
  }
  if (resource.creator == InvalidNode) {
    return true;
  }
  const CompiledNode& creator = graph.nodes[resource.creator];
  if (creator.stage == stage) {
    return !test_bit(mask.passes, creator.programIndex);
  }
  return creator.stage > stage || !test_bit(state.ran, creator.frameSlot);
}

constexpr std::array<std::string_view, 14> AccessNames{
    "Sampled",   "Uniform",    "StorageRead", "StorageWrite", "StorageRW", "ColorWrite", "ColorRMW",
    "DepthRead", "DepthWrite", "CopySrc",     "CopyDst",      "Vertex",    "Index",      "Indirect",
};
static_assert(AccessNames.size() == static_cast<size_t>(Access::Indirect) + 1);
constexpr std::array<std::string_view, 3> MissingNames{"DisablePass", "BindDummyBlack", "BindDummyWhite"};
static_assert(MissingNames.size() == static_cast<size_t>(Missing::BindDummyWhite) + 1);
constexpr std::array<std::string_view, 5> PassKindNames{"Compute", "Render", "Copy", "Encoder", "Native"};
static_assert(PassKindNames.size() == static_cast<size_t>(PassKind::Native) + 1);
constexpr std::array<std::string_view, 3> PassClassNames{"Required", "Optional", "Diagnostic"};
static_assert(PassClassNames.size() == static_cast<size_t>(PassClass::Diagnostic) + 1);
constexpr std::array<std::string_view, 5> ResourceKindNames{"Transient", "History", "Persistent", "Gx", "External"};
static_assert(ResourceKindNames.size() == static_cast<size_t>(ResourceKind::External) + 1);
constexpr std::array<std::string_view, 19> CullReasonNames{
    "None",
    "Skipped",
    "InvalidDeclaration",
    "DuplicateName",
    "DuplicateCreator",
    "ConflictingKind",
    "WrongStage",
    "AnchorConflict",
    "BindingLimit",
    "FormatUnsupported",
    "ResourceLimit",
    "BackwardsEdge",
    "Cycle",
    "MissingInput",
    "MissingTarget",
    "NoConsumer",
    "InspectorClosed",
    "UnknownAnchor",
    "CoercedMissing",
};
static_assert(CullReasonNames.size() == static_cast<size_t>(CullReason::CoercedMissing) + 1);
constexpr std::array<std::string_view, 6> EdgeKindNames{"Chain", "Bound", "Anchor", "Raw", "War", "Waw"};
static_assert(EdgeKindNames.size() == static_cast<size_t>(EdgeKind::Waw) + 1);

template <typename E, size_t N>
std::string_view enum_name(const std::array<std::string_view, N>& names, E value) noexcept {
  const auto index = static_cast<size_t>(value);
  return index < N ? names[index] : "Unknown";
}

constexpr bool reads(Access access) noexcept {
  switch (access) {
  case Access::Sampled:
  case Access::Uniform:
  case Access::StorageRead:
  case Access::DepthRead:
  case Access::CopySrc:
  case Access::Vertex:
  case Access::Index:
  case Access::Indirect:
    return true;
  default:
    return false;
  }
}

constexpr bool writes(Access access) noexcept { return !reads(access); }

constexpr bool binds(Access access) noexcept {
  return access == Access::Sampled || access == Access::Uniform || access == Access::StorageRead ||
         access == Access::StorageWrite || access == Access::StorageRW;
}

constexpr bool is_storage(Access access) noexcept {
  return access == Access::StorageRead || access == Access::StorageWrite || access == Access::StorageRW;
}

constexpr bool is_color(Access access) noexcept { return access == Access::ColorWrite || access == Access::ColorRMW; }

constexpr bool is_depth(Access access) noexcept { return access == Access::DepthRead || access == Access::DepthWrite; }

constexpr bool texture_only(Access access) noexcept {
  return access == Access::Sampled || is_color(access) || is_depth(access);
}

constexpr bool buffer_only(Access access) noexcept {
  return access == Access::Uniform || access == Access::Vertex || access == Access::Index || access == Access::Indirect;
}

constexpr bool creates(Access access) noexcept {
  return access == Access::StorageWrite || access == Access::ColorWrite || access == Access::DepthWrite ||
         access == Access::CopyDst;
}

constexpr bool kind_allows(PassKind kind, Access access) noexcept {
  switch (kind) {
  case PassKind::Compute:
    return access == Access::Indirect || binds(access);
  case PassKind::Render:
    return access != Access::CopySrc && access != Access::CopyDst;
  case PassKind::Copy:
    return access == Access::CopySrc || access == Access::CopyDst;
  case PassKind::Encoder:
    return true;
  case PassKind::Native:
    return is_storage(access) || access == Access::CopySrc || access == Access::CopyDst;
  }
  return false;
}

wgpu::ShaderStage default_visibility(PassKind kind) noexcept {
  switch (kind) {
  case PassKind::Compute:
    return wgpu::ShaderStage::Compute;
  case PassKind::Render:
    return wgpu::ShaderStage::Fragment;
  case PassKind::Encoder:
    return wgpu::ShaderStage::Compute | wgpu::ShaderStage::Fragment;
  default:
    return wgpu::ShaderStage::None;
  }
}

wgpu::TextureUsage texture_usage(Access access) noexcept {
  switch (access) {
  case Access::Sampled:
    return wgpu::TextureUsage::TextureBinding;
  case Access::StorageRead:
  case Access::StorageWrite:
  case Access::StorageRW:
    return wgpu::TextureUsage::StorageBinding;
  case Access::ColorWrite:
  case Access::ColorRMW:
  case Access::DepthRead:
  case Access::DepthWrite:
    return wgpu::TextureUsage::RenderAttachment;
  case Access::CopySrc:
    return wgpu::TextureUsage::CopySrc;
  case Access::CopyDst:
    return wgpu::TextureUsage::CopyDst;
  default:
    return wgpu::TextureUsage::None;
  }
}

wgpu::BufferUsage buffer_usage(Access access) noexcept {
  switch (access) {
  case Access::Uniform:
    return wgpu::BufferUsage::Uniform;
  case Access::StorageRead:
  case Access::StorageWrite:
  case Access::StorageRW:
    return wgpu::BufferUsage::Storage;
  case Access::Vertex:
    return wgpu::BufferUsage::Vertex;
  case Access::Index:
    return wgpu::BufferUsage::Index;
  case Access::Indirect:
    return wgpu::BufferUsage::Indirect;
  case Access::CopySrc:
    return wgpu::BufferUsage::CopySrc;
  case Access::CopyDst:
    return wgpu::BufferUsage::CopyDst;
  default:
    return wgpu::BufferUsage::None;
  }
}

bool gx_access_allowed(GxResource resource, Access access) noexcept {
  if (access == Access::Sampled || access == Access::CopySrc) {
    return true;
  }
  return resource == GxResource::Depth ? is_depth(access) : access == Access::ColorRMW;
}

// Under MSAA every EFB texture is multisampled; only the resolved colour and the colour attachment stay usable
bool msaa_allows(GxResource resource, Access access) noexcept {
  return resource == GxResource::Color &&
         (access == Access::Sampled || access == Access::CopySrc || access == Access::ColorRMW);
}

bool gx_use_available(const GraphEnvironment& env, GxResource resource, Access access) noexcept {
  const auto index = static_cast<size_t>(resource);
  if (env.gxFormats[index] == wgpu::TextureFormat::Undefined) {
    return false;
  }
  const auto needed = texture_usage(access);
  if (needed != wgpu::TextureUsage::None && !(env.gxUsage[index] & needed)) {
    return false;
  }
  return env.gxSampleCount <= 1 || msaa_allows(resource, access);
}

bool format_supports(wgpu::TextureFormat format, Access access, const GraphFeatures& features) noexcept {
  const FormatRow* row = find_format(format);
  switch (access) {
  case Access::StorageRead:
  case Access::StorageWrite:
    if (row == nullptr) {
      return false;
    }
    if ((row->caps & CapStorage) != 0) {
      return true;
    }
    if ((row->caps & CapStorageCore) != 0) {
      return features.core;
    }
    if ((row->caps & CapStorageTier1) != 0) {
      return features.tier1Formats;
    }
    return (row->caps & CapStorageBgra8) != 0 && access == Access::StorageWrite && features.bgra8Storage;
  case Access::StorageRW:
    if (row == nullptr) {
      return false;
    }
    if ((row->caps & CapReadWrite) != 0) {
      return true;
    }
    return (row->caps & CapReadWriteTier2) != 0 && features.tier2Formats;
  case Access::ColorWrite:
  case Access::ColorRMW:
    if (row == nullptr || row->colorBytes == 0) {
      return false;
    }
    if ((row->caps & CapRenderTier1) != 0) {
      return features.tier1Formats;
    }
    if ((row->caps & CapRenderRg11b10) != 0) {
      return features.rg11b10Renderable || features.tier1Formats;
    }
    return true;
  case Access::DepthRead:
  case Access::DepthWrite:
    return row != nullptr && (row->caps & CapDepth) != 0;
  case Access::CopySrc:
  case Access::CopyDst:
    return row == nullptr || (row->caps & CapNoCopy) == 0;
  default:
    return true;
  }
}

constexpr bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

bool valid_name(std::string_view name) noexcept {
  return !name.empty() && name.size() <= MaxNameLength &&
         std::ranges::none_of(name, [](char c) { return is_space(c) || c == '@'; });
}

bool is_gx_name(std::string_view name) noexcept { return name.starts_with("gx.") || name.starts_with("gbuf."); }

std::optional<NodeId> find_chain_node(std::string_view name) noexcept {
  for (size_t i = 0; i < ChainNodes.size(); ++i) {
    if (ChainNodes[i].name == name) {
      return static_cast<NodeId>(i);
    }
  }
  return std::nullopt;
}

bool valid_pass_name(std::string_view name) noexcept {
  return valid_name(name) && !name.starts_with("phase.") && !name.starts_with("gx@") && !find_chain_node(name);
}

uint32_t scaled(uint32_t size, uint16_t divisor) noexcept { return std::max(1u, (size + divisor - 1) / divisor); }

GraphExtent class_extent(SizeClass sizeClass, const GraphEnvironment& env) noexcept {
  return sizeClass == SizeClass::Output ? env.outputSize : env.efbSize;
}

wgpu::Extent3D texture_extent(const TexDesc& desc, const GraphEnvironment& env) noexcept {
  if (desc.sizeClass == SizeClass::Absolute) {
    return {desc.width, desc.height, desc.depthOrLayers};
  }
  const GraphExtent size = class_extent(desc.sizeClass, env);
  return {scaled(size.width, desc.divisor), scaled(size.height, desc.divisor), desc.depthOrLayers};
}

uint32_t full_mip_chain(const wgpu::Extent3D& extent, wgpu::TextureDimension dimension) noexcept {
  uint32_t largest = std::max(extent.width, extent.height);
  if (dimension == wgpu::TextureDimension::e3D) {
    largest = std::max(largest, extent.depthOrArrayLayers);
  }
  return static_cast<uint32_t>(std::bit_width(std::max(largest, 1u)));
}

uint64_t buffer_size(const BufDesc& desc, const GraphEnvironment& env) noexcept {
  uint64_t size = desc.size;
  if (desc.sizeClass != SizeClass::Absolute) {
    const GraphExtent extent = class_extent(desc.sizeClass, env);
    size = uint64_t{scaled(extent.width, desc.divisor)} * scaled(extent.height, desc.divisor) * desc.bytesPerTexel;
  }
  return (size + 15) & ~uint64_t{15};
}

std::optional<std::string> check_texture_desc(std::string_view name, const TexDesc& desc, const GraphEnvironment& env) {
  if (desc.format == wgpu::TextureFormat::Undefined) {
    return fmt::format("declares '{}' without a format", name);
  }
  if (desc.dimension != wgpu::TextureDimension::e2D && desc.dimension != wgpu::TextureDimension::e3D) {
    return fmt::format("declares '{}' with a dimension other than 2D or 3D", name);
  }
  if (desc.divisor == 0) {
    return fmt::format("declares '{}' with divisor 0", name);
  }
  if (desc.sizeClass == SizeClass::Absolute) {
    if (desc.width == 0 || desc.height == 0 || desc.depthOrLayers == 0) {
      return fmt::format("declares '{}' with a zero extent", name);
    }
  } else if (desc.width != 0 || desc.height != 0) {
    return fmt::format("declares '{}' with an explicit size and a relative size class", name);
  } else if (desc.depthOrLayers == 0) {
    return fmt::format("declares '{}' with zero depth or layers", name);
  }
  const uint32_t fullChain = full_mip_chain(texture_extent(desc, env), desc.dimension);
  if (desc.mips > fullChain) {
    return fmt::format("declares '{}' with {} mips, more than the full chain of {}", name, desc.mips, fullChain);
  }
  return std::nullopt;
}

std::optional<std::string> check_buffer_desc(std::string_view name, const BufDesc& desc) {
  if (desc.divisor == 0) {
    return fmt::format("declares '{}' with divisor 0", name);
  }
  if (desc.sizeClass == SizeClass::Absolute) {
    if (desc.size == 0) {
      return fmt::format("declares '{}' with size 0", name);
    }
  } else if (desc.size != 0 || desc.bytesPerTexel == 0) {
    return fmt::format("declares '{}' with a relative size class but no bytes per texel", name);
  }
  return std::nullopt;
}

wgpu::ShaderStage use_visibility(PassKind kind, Access access, wgpu::ShaderStage requested) noexcept {
  if (!binds(access)) {
    return wgpu::ShaderStage::None;
  }
  return requested != wgpu::ShaderStage::None ? requested : default_visibility(kind);
}

std::optional<AuroraLogLevel> diagnostic_level(CullReason reason) noexcept {
  switch (reason) {
  case CullReason::InvalidDeclaration:
  case CullReason::DuplicateName:
  case CullReason::DuplicateCreator:
  case CullReason::ConflictingKind:
  case CullReason::WrongStage:
  case CullReason::AnchorConflict:
  case CullReason::BackwardsEdge:
  case CullReason::Cycle:
    return LOG_ERROR;
  case CullReason::BindingLimit:
  case CullReason::FormatUnsupported:
  case CullReason::ResourceLimit:
  case CullReason::UnknownAnchor:
    return LOG_WARNING;
  case CullReason::MissingInput:
  case CullReason::MissingTarget:
  case CullReason::NoConsumer:
  case CullReason::InspectorClosed:
  case CullReason::CoercedMissing:
    return LOG_DEBUG;
  default:
    return std::nullopt;
  }
}

std::mutex g_logMutex;
std::unordered_set<std::string> g_loggedDiagnostics;

bool first_report(std::string key) {
  std::lock_guard lock{g_logMutex};
  return g_loggedDiagnostics.insert(std::move(key)).second;
}

void hash_string(Hasher& hasher, std::string_view value) {
  hasher.update(static_cast<uint32_t>(value.size()));
  hasher.update(value.data(), value.size());
}

constexpr uint16_t InvalidRef = 0xFFFF;

struct RegisteredPass {
  PassDesc desc;
  PassId id = InvalidPassId;
  uint64_t sequence = 0;
};

struct Slot {
  std::optional<RegisteredPass> pass;
  uint32_t generation = 1;
};

std::mutex g_passMutex;
std::vector<Slot> g_passSlots;
std::vector<uint32_t> g_freePassSlots;
uint64_t g_nextSequence = 1;
std::atomic_bool g_graphDirty = false;

constexpr uint32_t pass_index(PassId id) noexcept { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }
constexpr uint32_t pass_generation(PassId id) noexcept { return static_cast<uint32_t>(id >> 32); }
constexpr PassId make_pass_id(uint32_t index, uint32_t generation) noexcept {
  return static_cast<PassId>(generation) << 32 | index;
}

// Requires g_passMutex held.
Slot* find_pass_slot_locked(PassId id) noexcept {
  const auto index = pass_index(id);
  if (id == InvalidPassId || index >= g_passSlots.size()) {
    return nullptr;
  }
  auto& slot = g_passSlots[index];
  if (slot.generation != pass_generation(id) || !slot.pass) {
    return nullptr;
  }
  return &slot;
}

// Requires g_passMutex held.
bool has_live_pass_named(const std::string& name) noexcept {
  return std::ranges::any_of(g_passSlots, [&](const Slot& slot) { return slot.pass && slot.pass->desc.name == name; });
}
} // namespace

struct GraphBuild {
  struct DeclaredUse {
    std::string name;
    UseRole role = UseRole::Use;
    Access access = Access::Sampled;
    bool buffer = false;
    UseOptions options;
    TexDesc tex;
    BufDesc buf;
    PersistentHandle handle;
    Owner owner = Owner::Dawn;
  };

  struct Anchor {
    std::string name;
    bool after = true;
  };

  struct Placement {
    uint16_t lower = 0;
    uint16_t upper = InvalidNode;
    std::vector<std::pair<NodeId, bool>> passes;
  };

  struct Conflict {
    CullReason reason = CullReason::None;
    NodeId other = InvalidNode;
    ResourceId resource = InvalidResource;
    std::string message;
  };

  GraphBuild(const GraphInputs& inputs, CompiledGraph& graph) : m_inputs(inputs), m_graph(graph) {}

  void compile() {
    create_nodes();
    declare_passes();
    declare_gx_reads();
    flatten_uses();
    check_declarations();
    check_limits();
    check_backwards_edges();
    order_stages();
    cull_unconsumed();
    finish_programs();
    hash_graph();
  }

  uint16_t declare(const InternalPassBuilder& builder, DeclaredUse use) {
    if (builder.m_gxReadsOnly) {
      return declare_gx_read(builder.m_node, std::move(use));
    }
    if (m_skipped) {
      return InvalidRef;
    }
    if (auto problem = check_declaration(builder.m_node, use)) {
      reject(std::move(*problem));
      return InvalidRef;
    }
    if (!m_resourceIds.contains(use.name)) {
      ++m_pendingNames;
    }
    m_declared.push_back(std::move(use));
    return static_cast<uint16_t>(m_declared.size() - 1);
  }

  void declare_anchor(const InternalPassBuilder& builder, std::string_view name, bool after) {
    if (builder.m_gxReadsOnly) {
      gx_read_problem(builder.m_node, "cannot declare anchors", InvalidResource);
      return;
    }
    if (!m_skipped) {
      m_anchors.push_back({std::string{name}, after});
    }
  }

  void declare_side_effect(const InternalPassBuilder& builder) {
    if (builder.m_gxReadsOnly) {
      gx_read_problem(builder.m_node, "cannot declare side effects", InvalidResource);
      return;
    }
    m_sideEffect = true;
  }

  void declare_manual_bindings(const InternalPassBuilder& builder) {
    if (builder.m_gxReadsOnly) {
      gx_read_problem(builder.m_node, "cannot declare manual bindings", InvalidResource);
      return;
    }
    m_manualBindings = true;
  }

  void declare_skip(const InternalPassBuilder& builder) {
    if (builder.m_gxReadsOnly) {
      gx_read_problem(builder.m_node, "cannot be skipped", InvalidResource);
      return;
    }
    m_skipped = true;
  }

  std::span<std::byte, PassLocalSize> local(const InternalPassBuilder& builder) noexcept {
    if (builder.m_gxReadsOnly) {
      gx_read_problem(builder.m_node, "has no local storage", InvalidResource);
      return m_scratch;
    }
    return m_graph.nodes[builder.m_node].local;
  }

private:
  [[nodiscard]] bool active(NodeId node) const noexcept {
    const CompiledNode& compiled = m_graph.nodes[node];
    return compiled.kind != NodeKind::Pass || compiled.status == CullReason::None;
  }

  [[nodiscard]] std::string_view node_name(NodeId node) const noexcept {
    return node != InvalidNode ? std::string_view{m_graph.nodes[node].name} : std::string_view{};
  }

  [[nodiscard]] std::string_view resource_name(ResourceId resource) const noexcept {
    return resource != InvalidResource ? std::string_view{m_graph.resources[resource].name} : std::string_view{};
  }

  [[nodiscard]] NodeId chain_node_id(Stage stage, uint16_t chainIndex) const noexcept {
    return static_cast<NodeId>(m_chainBase + ChainRanges[static_cast<size_t>(stage)].first + chainIndex);
  }

  [[nodiscard]] std::span<CompiledUse> uses_of(NodeId node) noexcept {
    const CompiledNode& compiled = m_graph.nodes[node];
    return std::span{m_graph.uses}.subspan(compiled.firstUse, compiled.useCount);
  }

  // A storage-read dummy is created with the resource's format, so one without any known format cannot exist
  [[nodiscard]] static bool needs_dummy_format(const CompiledUse& use, const CompiledResource& resource) noexcept {
    return !use.buffer && use.role == UseRole::Use && use.access == Access::StorageRead &&
           use.missing != Missing::DisablePass && !resource.described && resource.kind != ResourceKind::Gx &&
           use.options.format == wgpu::TextureFormat::Undefined;
  }

  [[nodiscard]] bool is_pure_write(const CompiledUse& use) const noexcept {
    return use.role == UseRole::GxWrite || ((use.role == UseRole::Create || use.role == UseRole::History) &&
                                            m_graph.resources[use.resource].creator == use.node);
  }

  void add_diagnostic(Diagnostic diagnostic) {
    if (const auto level = diagnostic_level(diagnostic.reason)) {
      auto key = fmt::format("{}\x1f{}\x1f{}\x1f{}\x1f{}", static_cast<uint32_t>(diagnostic.reason),
                             node_name(diagnostic.node), node_name(diagnostic.other),
                             resource_name(diagnostic.resource), diagnostic.detail);
      if (first_report(std::move(key))) {
        Log.report(*level, "{}", diagnostic.message);
      }
    }
    m_graph.diagnostics.push_back(std::move(diagnostic));
  }

  void disable(NodeId node, CullReason reason, std::string message, NodeId other = InvalidNode,
               ResourceId resource = InvalidResource) {
    CompiledNode& compiled = m_graph.nodes[node];
    if (compiled.status != CullReason::None) {
      return;
    }
    compiled.status = reason;
    add_diagnostic({
        .reason = reason,
        .node = node,
        .other = other,
        .resource = resource,
        .message = std::move(message),
    });
  }

  void reject(std::string problem) {
    if (!m_rejection) {
      m_rejection = std::move(problem);
    }
  }

  void gx_read_problem(NodeId node, std::string_view problem, ResourceId resource) {
    add_diagnostic({
        .reason = CullReason::InvalidDeclaration,
        .node = node,
        .resource = resource,
        .message = fmt::format("render graph: GX read on '{}' {}; ignored", node_name(node), problem),
    });
  }

  ResourceId intern(std::string_view name) {
    auto [it, inserted] =
        m_resourceIds.try_emplace(std::string{name}, static_cast<ResourceId>(m_graph.resources.size()));
    if (inserted) {
      m_graph.resources.push_back({.name = std::string{name}});
      m_describers.push_back(InvalidNode);
      m_describerRoles.push_back(UseRole::Use);
      m_mentioned.push_back(false);
    }
    return it->second;
  }

  [[nodiscard]] std::optional<ResourceId> find_resource_id(std::string_view name) const {
    const auto it = m_resourceIds.find(std::string{name});
    return it != m_resourceIds.end() ? std::optional{it->second} : std::nullopt;
  }

  void create_nodes() {
    const auto& env = m_inputs.env;
    const size_t accepted = std::min(m_inputs.passes.size(), MaxPasses);
    m_passCount = static_cast<NodeId>(accepted);
    m_chainBase = m_passCount;
    m_graph.efbSize = env.efbSize;
    m_graph.outputSize = env.outputSize;
    m_graph.nodes.resize(accepted + ChainNodeCount);
    for (size_t i = 0; i < accepted; ++i) {
      const PassDesc& desc = m_inputs.passes[i];
      CompiledNode& node = m_graph.nodes[i];
      node.name = desc.name;
      node.kind = NodeKind::Pass;
      node.stage = desc.stage;
      node.passKind = desc.kind;
      node.cls = desc.cls;
      node.order = desc.order;
      node.registration = static_cast<uint16_t>(i);
      node.id = i < m_inputs.ids.size() ? m_inputs.ids[i] : InvalidPassId;
      node.opaque = desc.declare == nullptr;
      node.sideEffect = node.opaque;
      node.runsWithoutMarker = desc.runsWithoutMarker;
      node.enabled = desc.enabled;
      node.encode = desc.encode;
      node.encodeNative = desc.encodeNative;
      node.state = desc.state;
    }
    for (size_t i = accepted; i < m_inputs.passes.size(); ++i) {
      add_diagnostic({
          .reason = CullReason::InvalidDeclaration,
          .detail = m_inputs.passes[i].name,
          .message = fmt::format("render graph: too many passes; ignored '{}'", m_inputs.passes[i].name),
      });
    }
    for (size_t k = 0; k < ChainNodes.size(); ++k) {
      const ChainNode& chain = ChainNodes[k];
      CompiledNode& node = m_graph.nodes[m_chainBase + k];
      node.name = std::string{chain.name};
      node.kind = chain.kind == ChainKind::Gx      ? NodeKind::Gx
                  : chain.kind == ChainKind::Phase ? NodeKind::Phase
                                                   : NodeKind::Overlay;
      node.stage = chain.stage;
      node.chainIndex = static_cast<uint16_t>(k - ChainRanges[static_cast<size_t>(chain.stage)].first);
    }

    m_nodeUses.resize(m_graph.nodes.size());
    m_nodeAnchors.resize(m_graph.nodes.size());
    const GraphExtent efb{std::max(env.efbSize.width, 1u), std::max(env.efbSize.height, 1u)};
    for (size_t g = 0; g < GxResourceCount; ++g) {
      const ResourceId id = intern(GxRows[g].name);
      CompiledResource& resource = m_graph.resources[id];
      resource.kind = ResourceKind::Gx;
      resource.gx = static_cast<GxResource>(g);
      resource.tex.format = env.gxFormats[g];
      resource.described = env.gxFormats[g] != wgpu::TextureFormat::Undefined;
      resource.extent = {efb.width, efb.height, 1};
      m_mentioned[id] = true;
    }
    for (size_t k = 0; k < ChainNodes.size(); ++k) {
      if (ChainNodes[k].kind == ChainKind::Phase) {
        continue;
      }
      const auto node = static_cast<NodeId>(m_chainBase + k);
      for (size_t g = 0; g < GxResourceCount; ++g) {
        if (env.gxFormats[g] == wgpu::TextureFormat::Undefined) {
          continue;
        }
        const auto resource = static_cast<GxResource>(g);
        m_nodeUses[node].push_back({
            .node = node,
            .resource = static_cast<ResourceId>(g),
            .role = UseRole::GxWrite,
            .access = resource == GxResource::Depth ? Access::DepthWrite : Access::ColorRMW,
        });
      }
    }
  }

  void declare_passes() {
    for (NodeId i = 0; i < m_passCount; ++i) {
      CompiledNode& node = m_graph.nodes[i];
      if (!valid_pass_name(node.name)) {
        disable(i, CullReason::InvalidDeclaration,
                fmt::format("render graph: pass name '{}' is invalid; disabled", node.name));
        continue;
      }
      if (const auto [it, inserted] = m_passNames.try_emplace(node.name, i); !inserted) {
        disable(i, CullReason::DuplicateName,
                fmt::format("render graph: a pass named '{}' is already declared; disabled the later one", node.name),
                it->second);
        continue;
      }
      if (static_cast<size_t>(node.stage) >= StageCount) {
        disable(i, CullReason::InvalidDeclaration,
                fmt::format("render graph: '{}' has unknown stage {}; disabled", node.name,
                            static_cast<uint32_t>(node.stage)));
        continue;
      }
      if ((node.passKind == PassKind::Native) != (node.stage == Stage::AfterFrame)) {
        disable(i, CullReason::WrongStage,
                fmt::format("render graph: '{}' is a {} pass and cannot run in {}; disabled", node.name,
                            pass_kind_name(node.passKind), stage_name(node.stage)));
      }
      const PassDesc& desc = m_inputs.passes[i];
      if (desc.declare == nullptr) {
        continue;
      }
      m_declared.clear();
      m_anchors.clear();
      m_pendingNames = 0;
      m_skipped = false;
      m_sideEffect = false;
      m_manualBindings = false;
      m_rejection.reset();
      InternalPassBuilder builder{*this, i, false};
      const PassSetup setup{m_inputs.settings, m_inputs.caps, &m_inputs.env};
      desc.declare(builder, setup, desc.state);
      if (m_skipped) {
        if (node.status == CullReason::None) {
          node.status = CullReason::Skipped;
        }
        continue;
      }
      commit(i);
    }
  }

  void commit(NodeId nodeId) {
    CompiledNode& node = m_graph.nodes[nodeId];
    node.sideEffect = node.sideEffect || m_sideEffect;
    node.manualBindings = m_manualBindings;
    std::optional<Conflict> conflict;
    auto& uses = m_nodeUses[nodeId];
    for (const DeclaredUse& declared : m_declared) {
      const ResourceId resource = intern(declared.name);
      uses.push_back({
          .node = nodeId,
          .resource = resource,
          .role = declared.role,
          .access = declared.access,
          .missing = declared.options.missing,
          .buffer = declared.buffer,
          .visibility = use_visibility(node.passKind, declared.access, declared.options.visibility),
          .options = declared.options,
      });
      auto claimed = claim(nodeId, resource, declared);
      if (!conflict && claimed) {
        conflict = std::move(claimed);
      }
    }
    for (Anchor& anchor : m_anchors) {
      m_nodeAnchors[nodeId].push_back(std::move(anchor));
    }
    if (m_rejection) {
      disable(nodeId, CullReason::InvalidDeclaration,
              fmt::format("render graph: '{}' {}; disabled", node.name, *m_rejection));
    }
    if (conflict) {
      disable(nodeId, conflict->reason, std::move(conflict->message), conflict->other, conflict->resource);
    }
  }

  std::optional<Conflict> claim(NodeId nodeId, ResourceId resourceId, const DeclaredUse& declared) {
    CompiledResource& resource = m_graph.resources[resourceId];
    if (declared.role == UseRole::Use || declared.role == UseRole::Prev) {
      if (!resource.described && !m_mentioned[resourceId]) {
        resource.buffer = declared.buffer;
        resource.kind = declared.role == UseRole::Prev ? ResourceKind::History : ResourceKind::Transient;
      }
      m_mentioned[resourceId] = true;
      return std::nullopt;
    }
    m_mentioned[resourceId] = true;
    const NodeId describer = m_describers[resourceId];
    if (describer == InvalidNode) {
      describe(nodeId, resourceId, declared);
      return std::nullopt;
    }
    const std::string_view name = m_graph.nodes[nodeId].name;
    const std::string_view otherName = m_graph.nodes[describer].name;
    const UseRole prior = m_describerRoles[resourceId];
    if (declared.role == UseRole::Import && prior == UseRole::Import) {
      const bool same = resource.handle == declared.handle && resource.owner == declared.owner &&
                        resource.buffer == declared.buffer && (declared.buffer || resource.tex == declared.tex);
      if (same) {
        return std::nullopt;
      }
      return Conflict{
          .reason = CullReason::ConflictingKind,
          .other = describer,
          .resource = resourceId,
          .message = fmt::format("render graph: '{}' imports '{}' unlike '{}' does; disabled '{}'", name, resource.name,
                                 otherName, name),
      };
    }
    if (declared.role == prior) {
      return Conflict{
          .reason = CullReason::DuplicateCreator,
          .other = describer,
          .resource = resourceId,
          .message = fmt::format("render graph: '{}' creates '{}', which '{}' already creates; disabled '{}'", name,
                                 resource.name, otherName, name),
      };
    }
    return Conflict{
        .reason = CullReason::ConflictingKind,
        .other = describer,
        .resource = resourceId,
        .message =
            fmt::format("render graph: '{}' declares '{}' as {}, but '{}' declared it as {}; disabled '{}'", name,
                        resource.name, role_kind_name(declared), otherName, resource_kind_name(resource.kind), name),
    };
  }

  static std::string_view role_kind_name(const DeclaredUse& declared) noexcept {
    switch (declared.role) {
    case UseRole::Create:
      return resource_kind_name(ResourceKind::Transient);
    case UseRole::History:
      return resource_kind_name(ResourceKind::History);
    default:
      return resource_kind_name(declared.owner == Owner::NativeD3D12 ? ResourceKind::External
                                                                     : ResourceKind::Persistent);
    }
  }

  void describe(NodeId nodeId, ResourceId resourceId, const DeclaredUse& declared) {
    CompiledResource& resource = m_graph.resources[resourceId];
    m_describers[resourceId] = nodeId;
    m_describerRoles[resourceId] = declared.role;
    resource.described = true;
    resource.buffer = declared.buffer;
    switch (declared.role) {
    case UseRole::Create:
      resource.kind = ResourceKind::Transient;
      resource.creator = nodeId;
      break;
    case UseRole::History:
      resource.kind = ResourceKind::History;
      resource.creator = nodeId;
      break;
    default:
      resource.kind = declared.owner == Owner::NativeD3D12 ? ResourceKind::External : ResourceKind::Persistent;
      resource.handle = declared.handle;
      resource.owner = declared.owner;
      break;
    }
    if (declared.buffer) {
      if (declared.role == UseRole::Create) {
        resource.buf = declared.buf;
        resource.bufferSize = buffer_size(declared.buf, m_inputs.env);
      }
      return;
    }
    resource.tex = declared.tex;
    resource.extent = texture_extent(declared.tex, m_inputs.env);
    resource.mipLevels =
        declared.tex.mips == 0 ? full_mip_chain(resource.extent, declared.tex.dimension) : declared.tex.mips;
  }

  std::optional<std::string> check_declaration(NodeId nodeId, const DeclaredUse& use) const {
    const CompiledNode& node = m_graph.nodes[nodeId];
    const std::string_view access = access_name(use.access);
    if (!valid_name(use.name)) {
      return fmt::format("names invalid resource '{}'", use.name);
    }
    if (std::ranges::any_of(m_declared, [&](const DeclaredUse& other) { return other.name == use.name; })) {
      return fmt::format("uses '{}' twice", use.name);
    }
    if (is_gx_name(use.name)) {
      if (use.role != UseRole::Use) {
        return fmt::format("cannot create, import or keep history of GX resource '{}'", use.name);
      }
      const auto gx = find_gx_resource(use.name);
      if (!gx) {
        return fmt::format("uses unknown GX resource '{}'", use.name);
      }
      if (use.buffer) {
        return fmt::format("uses GX texture '{}' as a buffer", use.name);
      }
      if (!gx_access_allowed(*gx, use.access)) {
        return fmt::format("cannot use GX resource '{}' for {}", use.name, access);
      }
    }
    if (use.buffer ? texture_only(use.access) : buffer_only(use.access)) {
      return fmt::format("uses {} '{}' for {}", use.buffer ? "buffer" : "texture", use.name, access);
    }
    if (!kind_allows(node.passKind, use.access)) {
      return fmt::format("is a {} pass and cannot use '{}' for {}", pass_kind_name(node.passKind), use.name, access);
    }
    const bool native = node.passKind == PassKind::Native;
    switch (use.role) {
    case UseRole::Create:
      if (native) {
        return fmt::format("is a native pass and cannot create '{}'", use.name);
      }
      if (!creates(use.access)) {
        return fmt::format("creates '{}' with {}, which is not a plain write", use.name, access);
      }
      break;
    case UseRole::History:
      if (native) {
        return fmt::format("is a native pass and cannot own history '{}'", use.name);
      }
      if (use.access != Access::StorageWrite && use.access != Access::ColorWrite) {
        return fmt::format("writes history '{}' with {}", use.name, access);
      }
      break;
    case UseRole::Prev:
      if (native) {
        return fmt::format("is a native pass and cannot read history '{}'", use.name);
      }
      break;
    case UseRole::Import:
      if (use.handle.id == 0) {
        return fmt::format("imports '{}' with handle 0", use.name);
      }
      if (native != (use.owner == Owner::NativeD3D12)) {
        return native ? fmt::format("is a native pass and cannot import Dawn resource '{}'", use.name)
                      : fmt::format("imports external '{}' outside a native pass", use.name);
      }
      break;
    default:
      break;
    }
    if (node.passKind == PassKind::Render && (use.access == Access::StorageWrite || use.access == Access::StorageRW) &&
        use_visibility(node.passKind, use.access, use.options.visibility) != wgpu::ShaderStage::Fragment) {
      return fmt::format("writes storage '{}' outside the fragment stage", use.name);
    }
    if (use.role == UseRole::Create || use.role == UseRole::History || use.role == UseRole::Import) {
      auto problem = use.buffer ? (use.role == UseRole::Create ? check_buffer_desc(use.name, use.buf) : std::nullopt)
                                : check_texture_desc(use.name, use.tex, m_inputs.env);
      if (problem) {
        return problem;
      }
    }
    if (!m_resourceIds.contains(use.name) && m_resourceIds.size() + m_pendingNames >= MaxResources) {
      return std::string{"declares too many resources"};
    }
    return std::nullopt;
  }

  uint16_t declare_gx_read(NodeId nodeId, DeclaredUse use) {
    const auto existing = find_resource_id(use.name);
    const ResourceId known = existing.value_or(InvalidResource);
    std::optional<std::string> problem;
    const bool readAccess = use.access == Access::Sampled || use.access == Access::Uniform ||
                            use.access == Access::StorageRead || use.access == Access::Vertex ||
                            use.access == Access::Index || use.access == Access::Indirect;
    auto& uses = m_nodeUses[nodeId];
    if (use.role != UseRole::Use || !readAccess) {
      problem = fmt::format("may only read '{}'", use.name);
    } else if (!valid_name(use.name)) {
      problem = fmt::format("names invalid resource '{}'", use.name);
    } else if (is_gx_name(use.name)) {
      problem = fmt::format("cannot read GX resource '{}'", use.name);
    } else if (use.buffer ? texture_only(use.access) : buffer_only(use.access)) {
      problem =
          fmt::format("uses {} '{}' for {}", use.buffer ? "buffer" : "texture", use.name, access_name(use.access));
    } else if (existing &&
               std::ranges::any_of(uses, [&](const CompiledUse& other) { return other.resource == known; })) {
      problem = fmt::format("uses '{}' twice", use.name);
    } else if (existing && m_mentioned[known] && m_graph.resources[known].buffer != use.buffer) {
      problem = fmt::format("uses '{}' as a {}, but it is a {}", use.name, use.buffer ? "buffer" : "texture",
                            use.buffer ? "texture" : "buffer");
    } else if (!use.buffer && use.access == Access::StorageRead &&
               use.options.format == wgpu::TextureFormat::Undefined &&
               !(existing && m_graph.resources[known].described)) {
      // Every GX read can fall back to a dummy, and a storage-read dummy needs a format
      problem = fmt::format("reads storage texture '{}' with no known format", use.name);
    } else if (!existing && m_resourceIds.size() >= MaxResources) {
      problem = std::string{"declares too many resources"};
    }
    if (problem) {
      gx_read_problem(nodeId, *problem, known);
      return InvalidRef;
    }
    const ResourceId resource = intern(use.name);
    if (!m_mentioned[resource]) {
      m_graph.resources[resource].buffer = use.buffer;
      m_mentioned[resource] = true;
    }
    uses.push_back({
        .node = nodeId,
        .resource = resource,
        .role = UseRole::Use,
        .access = use.access,
        .missing = use.options.missing,
        .buffer = use.buffer,
        .visibility = binds(use.access) ? use.options.visibility : wgpu::ShaderStage::None,
        .options = use.options,
    });
    return static_cast<uint16_t>(uses.size() - 1);
  }

  void declare_gx_reads() {
    const PassSetup setup{m_inputs.settings, m_inputs.caps, &m_inputs.env};
    for (const GxReadDesc& desc : m_inputs.gxReads) {
      const auto chain = find_chain_node(desc.node);
      if (!chain || ChainNodes[*chain].kind == ChainKind::Phase) {
        add_diagnostic({
            .reason = CullReason::InvalidDeclaration,
            .detail = std::string{desc.node},
            .message =
                fmt::format("render graph: GX read names '{}', which is not a GX chain node; ignored", desc.node),
        });
        continue;
      }
      if (desc.declare == nullptr) {
        continue;
      }
      InternalPassBuilder builder{*this, static_cast<NodeId>(m_chainBase + *chain), true};
      desc.declare(builder, setup, desc.state);
    }
  }

  void flatten_uses() {
    for (size_t i = 0; i < m_graph.nodes.size(); ++i) {
      CompiledNode& node = m_graph.nodes[i];
      node.firstUse = static_cast<uint32_t>(m_graph.uses.size());
      node.useCount = static_cast<uint32_t>(m_nodeUses[i].size());
      m_graph.uses.insert(m_graph.uses.end(), m_nodeUses[i].begin(), m_nodeUses[i].end());
    }
    m_nodeUses.clear();
    m_resourceUses.assign(m_graph.resources.size(), {});
    for (size_t i = 0; i < m_graph.uses.size(); ++i) {
      m_resourceUses[m_graph.uses[i].resource].push_back(static_cast<uint32_t>(i));
    }
  }

  void check_declarations() {
    m_placements.resize(m_passCount);
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (m_graph.nodes[i].status != CullReason::None) {
        continue;
      }
      if (auto problem = check_pass_uses(i)) {
        disable(i, CullReason::InvalidDeclaration,
                fmt::format("render graph: '{}' {}; disabled", m_graph.nodes[i].name, *problem));
        continue;
      }
      place(i);
      if (m_graph.nodes[i].cls == PassClass::Diagnostic && !m_inputs.env.inspectorOpen) {
        disable(i, CullReason::InspectorClosed,
                fmt::format("render graph: '{}' is a diagnostic pass and the inspector is closed; disabled",
                            m_graph.nodes[i].name));
      }
    }
  }

  std::optional<std::string> check_pass_uses(NodeId nodeId) {
    const CompiledNode& node = m_graph.nodes[nodeId];
    uint32_t colors = 0;
    uint32_t depths = 0;
    std::optional<std::pair<uint32_t, uint32_t>> size;
    bool sizeMismatch = false;
    for (const CompiledUse& use : uses_of(nodeId)) {
      const CompiledResource& resource = m_graph.resources[use.resource];
      if (resource.buffer != use.buffer) {
        return fmt::format("uses '{}' as a {}, but it is a {}", resource.name, use.buffer ? "buffer" : "texture",
                           use.buffer ? "texture" : "buffer");
      }
      if (use.role == UseRole::Prev && resource.described && resource.kind != ResourceKind::History) {
        return fmt::format("reads the previous frame of '{}', which is not a history", resource.name);
      }
      if (needs_dummy_format(use, resource)) {
        return fmt::format("reads storage texture '{}' with a dummy fallback but no known format", resource.name);
      }
      if (node.passKind == PassKind::Native && resource.kind != ResourceKind::External) {
        return fmt::format("is a native pass and cannot use {} resource '{}'", resource_kind_name(resource.kind),
                           resource.name);
      }
      if (!is_color(use.access) && !is_depth(use.access)) {
        continue;
      }
      ++(is_color(use.access) ? colors : depths);
      if (!resource.described && resource.kind != ResourceKind::Gx) {
        continue;
      }
      const std::pair extent{resource.extent.width, resource.extent.height};
      if (!size) {
        size = extent;
      } else if (*size != extent) {
        sizeMismatch = true;
      }
    }
    if (node.passKind != PassKind::Render) {
      return std::nullopt;
    }
    if (colors + depths == 0) {
      return std::string{"is a render pass without attachments"};
    }
    if (depths > 1) {
      return std::string{"has more than one depth attachment"};
    }
    if (sizeMismatch) {
      return std::string{"has attachments of different sizes"};
    }
    return std::nullopt;
  }

  void place(NodeId nodeId) {
    const CompiledNode& node = m_graph.nodes[nodeId];
    const auto& chain = ChainRanges[static_cast<size_t>(node.stage)];
    const auto last = static_cast<uint16_t>(chain.count - 1);
    Placement& placement = m_placements[nodeId];
    bool conflict = false;
    bool anchoredToPass = false;
    std::optional<uint16_t> after;
    std::optional<uint16_t> before;
    for (const Anchor& anchor : m_nodeAnchors[nodeId]) {
      if (anchor.name == node.name) {
        conflict = true;
        continue;
      }
      if (const auto index = find_chain_node(anchor.name); index && ChainNodes[*index].stage == node.stage) {
        const auto chainIndex = static_cast<uint16_t>(*index - chain.first);
        auto& bound = anchor.after ? after : before;
        bound = bound ? (anchor.after ? std::max(*bound, chainIndex) : std::min(*bound, chainIndex)) : chainIndex;
        continue;
      }
      if (const auto it = m_passNames.find(anchor.name);
          it != m_passNames.end() && m_graph.nodes[it->second].stage == node.stage) {
        anchoredToPass = true;
        placement.passes.emplace_back(it->second, anchor.after);
        continue;
      }
      add_diagnostic({
          .reason = CullReason::UnknownAnchor,
          .node = nodeId,
          .detail = anchor.name,
          .message = fmt::format("render graph: '{}' has unknown anchor '{}' in {}; ignored", node.name, anchor.name,
                                 stage_name(node.stage)),
      });
    }
    if (after) {
      placement.lower = *after;
      placement.upper = before ? *before : (*after < last ? static_cast<uint16_t>(*after + 1) : InvalidNode);
      conflict = conflict || (before && *after >= *before);
    } else if (before) {
      placement.lower = 0;
      placement.upper = *before;
      conflict = conflict || *before == 0;
    } else {
      placement.lower = anchoredToPass ? 0 : last;
    }
    if (conflict) {
      disable(
          nodeId, CullReason::AnchorConflict,
          fmt::format("render graph: '{}' has conflicting anchors in {}; disabled", node.name, stage_name(node.stage)));
    }
  }

  void check_limits() {
    m_boundAsStorage.assign(m_graph.resources.size(), false);
    m_boundAsUniform.assign(m_graph.resources.size(), false);
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (!active(i)) {
        continue;
      }
      for (const CompiledUse& use : uses_of(i)) {
        if (use.buffer && is_storage(use.access)) {
          m_boundAsStorage[use.resource] = true;
        } else if (use.access == Access::Uniform) {
          m_boundAsUniform[use.resource] = true;
        }
      }
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (!active(i)) {
        continue;
      }
      if (auto failure = check_pass_limits(i)) {
        disable(i, failure->first, std::move(failure->second));
      }
    }
  }

  std::optional<std::pair<CullReason, std::string>> check_pass_limits(NodeId nodeId) {
    CompiledNode& node = m_graph.nodes[nodeId];
    const GraphLimits& limits = m_inputs.env.limits;
    const auto uses = uses_of(nodeId);
    const auto fail = [&](CullReason reason, std::string message) {
      return std::optional{std::pair{reason, std::move(message)}};
    };
    const bool counted =
        node.passKind == PassKind::Compute || node.passKind == PassKind::Render || node.passKind == PassKind::Encoder;
    if (counted && !node.manualBindings) {
      constexpr std::array StageBits{wgpu::ShaderStage::Vertex, wgpu::ShaderStage::Fragment,
                                     wgpu::ShaderStage::Compute};
      constexpr std::array<std::string_view, 3> StageNames{"vertex", "fragment", "compute"};
      wgpu::ShaderStage stages = default_visibility(node.passKind);
      for (const CompiledUse& use : uses) {
        stages = stages | use.visibility;
      }
      std::array<BindingCounts, 3> counts{};
      for (size_t s = 0; s < StageBits.size(); ++s) {
        if (stages & StageBits[s]) {
          counts[s] = FrameGroupCost;
        }
      }
      uint32_t bindings = 0;
      uint32_t vertexBuffers = 0;
      for (const CompiledUse& use : uses) {
        if (use.access == Access::Vertex) {
          ++vertexBuffers;
        }
        if (!binds(use.access)) {
          continue;
        }
        ++bindings;
        for (size_t s = 0; s < StageBits.size(); ++s) {
          if (!(use.visibility & StageBits[s])) {
            continue;
          }
          auto& count = counts[s];
          if (use.access == Access::Sampled) {
            ++count.sampledTextures;
          } else if (use.access == Access::Uniform) {
            ++count.uniformBuffers;
          } else if (use.buffer) {
            ++count.storageBuffers;
          } else {
            ++count.storageTextures;
          }
        }
      }
      node.bindings = counts;
      const std::array stageLimits{limits.vertex, limits.fragment, limits.compute};
      for (size_t s = 0; s < StageBits.size(); ++s) {
        const BindingCounts& count = counts[s];
        const BindingCounts& limit = stageLimits[s];
        const std::array<std::tuple<uint32_t, uint32_t, std::string_view>, 5> quantities{{
            {count.sampledTextures, limit.sampledTextures, "sampled textures"},
            {count.samplers, limit.samplers, "samplers"},
            {count.storageBuffers, limit.storageBuffers, "storage buffers"},
            {count.storageTextures, limit.storageTextures, "storage textures"},
            {count.uniformBuffers, limit.uniformBuffers, "uniform buffers"},
        }};
        for (const auto& [needed, available, quantity] : quantities) {
          if (needed > available) {
            return fail(CullReason::BindingLimit,
                        fmt::format("render graph: '{}' needs {} {} in the {} stage, limit {}; disabled", node.name,
                                    needed, quantity, StageNames[s], available));
          }
        }
      }
      if (FrameGroupDynamicUniformBuffers > limits.dynamicUniformBuffers) {
        return fail(CullReason::BindingLimit,
                    fmt::format("render graph: '{}' needs {} dynamic uniform buffers, limit {}; disabled", node.name,
                                FrameGroupDynamicUniformBuffers, limits.dynamicUniformBuffers));
      }
      if (bindings > limits.bindingsPerGroup) {
        return fail(CullReason::BindingLimit,
                    fmt::format("render graph: '{}' needs {} bindings in its pass group, limit {}; disabled", node.name,
                                bindings, limits.bindingsPerGroup));
      }
      if (vertexBuffers > limits.vertexBuffers) {
        return fail(CullReason::BindingLimit,
                    fmt::format("render graph: '{}' needs {} vertex buffers, limit {}; disabled", node.name,
                                vertexBuffers, limits.vertexBuffers));
      }
    }

    if (node.passKind == PassKind::Render || node.passKind == PassKind::Encoder) {
      uint32_t colorCount = 0;
      std::vector<wgpu::TextureFormat> formats;
      for (const CompiledUse& use : uses) {
        if (!is_color(use.access)) {
          continue;
        }
        const CompiledResource& resource = m_graph.resources[use.resource];
        if (resource.kind == ResourceKind::Gx && !gx_use_available(m_inputs.env, resource.gx, use.access)) {
          continue;
        }
        ++colorCount;
        if (resource.tex.format != wgpu::TextureFormat::Undefined) {
          formats.push_back(resource.tex.format);
        }
      }
      node.colorBytes = color_bytes_per_sample(formats);
      if (colorCount > limits.colorAttachments) {
        return fail(CullReason::BindingLimit,
                    fmt::format("render graph: '{}' needs {} colour attachments, limit {}; disabled", node.name,
                                colorCount, limits.colorAttachments));
      }
      if (node.colorBytes > limits.colorAttachmentBytesPerSample) {
        return fail(CullReason::BindingLimit,
                    fmt::format("render graph: '{}' needs {} colour attachment bytes per sample, limit {}; disabled",
                                node.name, node.colorBytes, limits.colorAttachmentBytesPerSample));
      }
    }

    if (node.passKind != PassKind::Native) {
      const GraphFeatures& features = limits.features;
      for (const CompiledUse& use : uses) {
        const CompiledResource& resource = m_graph.resources[use.resource];
        const wgpu::TextureFormat format = resource.tex.format;
        if (use.buffer || format == wgpu::TextureFormat::Undefined || format_supports(format, use.access, features)) {
          continue;
        }
        return fail(CullReason::FormatUnsupported,
                    fmt::format("render graph: '{}' uses '{}' ({}) for {}, which the device does not support; disabled",
                                node.name, resource.name, format_name(format), access_name(use.access)));
      }
      for (const CompiledUse& use : uses) {
        const CompiledResource& resource = m_graph.resources[use.resource];
        if (use.buffer || m_describers[use.resource] != nodeId) {
          continue;
        }
        const wgpu::TextureFormat format = resource.tex.format;
        const wgpu::TextureUsage extra = resource.tex.extraUsage;
        const std::array<std::pair<wgpu::TextureUsage, Access>, 4> checks{{
            {wgpu::TextureUsage::StorageBinding, Access::StorageWrite},
            {wgpu::TextureUsage::RenderAttachment, is_depth_format(format) ? Access::DepthWrite : Access::ColorWrite},
            {wgpu::TextureUsage::CopySrc, Access::CopySrc},
            {wgpu::TextureUsage::CopyDst, Access::CopyDst},
        }};
        for (const auto& [usage, access] : checks) {
          if ((extra & usage) && !format_supports(format, access, features)) {
            return fail(CullReason::FormatUnsupported,
                        fmt::format("render graph: '{}' gives '{}' ({}) a usage for {}, which the device does not "
                                    "support; disabled",
                                    node.name, resource.name, format_name(format), access_name(access)));
          }
        }
      }
    }

    for (const CompiledUse& use : uses) {
      if (m_describers[use.resource] != nodeId) {
        continue;
      }
      const CompiledResource& resource = m_graph.resources[use.resource];
      const auto exceeds = [&](std::string_view quantity, uint64_t value, uint64_t limit) {
        return fail(CullReason::ResourceLimit,
                    fmt::format("render graph: '{}' declares '{}' with {} {}, limit {}; disabled", node.name,
                                resource.name, quantity, value, limit));
      };
      if (resource.buffer) {
        if (resource.bufferSize > limits.bufferSize) {
          return exceeds("size", resource.bufferSize, limits.bufferSize);
        }
        if (m_boundAsStorage[use.resource] && resource.bufferSize > limits.storageBufferBindingSize) {
          return exceeds("storage binding size", resource.bufferSize, limits.storageBufferBindingSize);
        }
        if (m_boundAsUniform[use.resource] && resource.bufferSize > limits.uniformBufferBindingSize) {
          return exceeds("uniform binding size", resource.bufferSize, limits.uniformBufferBindingSize);
        }
        continue;
      }
      const wgpu::Extent3D& extent = resource.extent;
      if (resource.tex.dimension == wgpu::TextureDimension::e3D) {
        const std::array<std::pair<std::string_view, uint32_t>, 3> axes{
            {{"width", extent.width}, {"height", extent.height}, {"depth", extent.depthOrArrayLayers}}};
        for (const auto& [axis, value] : axes) {
          if (value > limits.textureDimension3D) {
            return exceeds(axis, value, limits.textureDimension3D);
          }
        }
        continue;
      }
      if (extent.width > limits.textureDimension2D) {
        return exceeds("width", extent.width, limits.textureDimension2D);
      }
      if (extent.height > limits.textureDimension2D) {
        return exceeds("height", extent.height, limits.textureDimension2D);
      }
      if (extent.depthOrArrayLayers > limits.textureArrayLayers) {
        return exceeds("layers", extent.depthOrArrayLayers, limits.textureArrayLayers);
      }
    }
    return std::nullopt;
  }

  void check_backwards_edges() {
    // Decided against the creators active before this step, so the result does not depend on registration order
    std::vector<bool> wasActive(m_passCount);
    for (NodeId i = 0; i < m_passCount; ++i) {
      wasActive[i] = active(i);
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (!wasActive[i]) {
        continue;
      }
      const CompiledNode& node = m_graph.nodes[i];
      for (const CompiledUse& use : uses_of(i)) {
        const CompiledResource& resource = m_graph.resources[use.resource];
        if (use.role != UseRole::Use ||
            (resource.kind != ResourceKind::Transient && resource.kind != ResourceKind::History) ||
            resource.creator == InvalidNode || !wasActive[resource.creator]) {
          continue;
        }
        const CompiledNode& creator = m_graph.nodes[resource.creator];
        if (creator.stage <= node.stage) {
          continue;
        }
        disable(i, CullReason::BackwardsEdge,
                fmt::format("render graph: '{}' in {} reads '{}' created later by '{}' in {}; disabled '{}'", node.name,
                            stage_name(node.stage), resource.name, creator.name, stage_name(creator.stage), node.name),
                resource.creator, use.resource);
        break;
      }
    }
  }

  [[nodiscard]] bool use_available(const CompiledUse& use) const noexcept {
    const CompiledResource& resource = m_graph.resources[use.resource];
    if (use.role == UseRole::Prev) {
      return resource.kind == ResourceKind::History && resource.creator != InvalidNode && active(resource.creator);
    }
    switch (resource.kind) {
    case ResourceKind::Gx:
      return gx_use_available(m_inputs.env, resource.gx, use.access);
    case ResourceKind::Persistent:
    case ResourceKind::External:
      return true;
    default:
      // Passes reading a later stage's resource are already disabled as backwards edges; GX reads are not
      return resource.creator != InvalidNode && active(resource.creator) &&
             m_graph.nodes[resource.creator].stage <= m_graph.nodes[use.node].stage;
    }
  }

  void apply_missing() {
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t i = 0; i < m_graph.nodes.size(); ++i) {
        const auto nodeId = static_cast<NodeId>(i);
        if (!active(nodeId)) {
          continue;
        }
        const bool chain = m_graph.nodes[i].kind != NodeKind::Pass;
        for (CompiledUse& use : uses_of(nodeId)) {
          if ((use.role != UseRole::Use && use.role != UseRole::Prev) || use.staticDummy || use_available(use)) {
            continue;
          }
          const std::string_view name = m_graph.nodes[i].name;
          const std::string_view resource = m_graph.resources[use.resource].name;
          if (chain) {
            use.staticDummy = true;
            if (use.missing == Missing::DisablePass) {
              add_diagnostic({
                  .reason = CullReason::CoercedMissing,
                  .node = nodeId,
                  .resource = use.resource,
                  .message = fmt::format("render graph: GX read on '{}' needs missing '{}'; bound a black dummy", name,
                                         resource),
              });
            }
            continue;
          }
          if (writes(use.access)) {
            disable(nodeId, CullReason::MissingTarget,
                    fmt::format("render graph: '{}' writes '{}', which is not available; disabled", name, resource),
                    InvalidNode, use.resource);
            changed = true;
            break;
          }
          if (use.missing == Missing::DisablePass) {
            disable(nodeId, CullReason::MissingInput,
                    fmt::format("render graph: '{}' reads '{}', which is not available; disabled", name, resource),
                    InvalidNode, use.resource);
            changed = true;
            break;
          }
          use.staticDummy = true;
        }
      }
    }
  }

  void add_edge(NodeId from, NodeId to, EdgeKind kind, ResourceId resource = InvalidResource) {
    const uint64_t key =
        uint64_t{from} | uint64_t{to} << 16 | uint64_t{resource} << 32 | uint64_t{static_cast<uint8_t>(kind)} << 48;
    if (m_edgeKeys.insert(key).second) {
      m_edges.push_back({from, to, kind, resource});
    }
  }

  void add_placement_edges() {
    for (size_t s = 0; s < StageCount; ++s) {
      const auto stage = static_cast<Stage>(s);
      for (uint16_t k = 1; k < ChainRanges[s].count; ++k) {
        add_edge(chain_node_id(stage, k - 1), chain_node_id(stage, k), EdgeKind::Chain);
      }
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (!active(i)) {
        continue;
      }
      const Stage stage = m_graph.nodes[i].stage;
      const Placement& placement = m_placements[i];
      add_edge(chain_node_id(stage, placement.lower), i, EdgeKind::Bound);
      if (placement.upper != InvalidNode) {
        add_edge(i, chain_node_id(stage, placement.upper), EdgeKind::Bound);
      }
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (!active(i)) {
        continue;
      }
      for (const auto& [other, after] : m_placements[i].passes) {
        if (other == i || !active(other)) {
          continue;
        }
        if (after) {
          add_edge(other, i, EdgeKind::Anchor);
        } else {
          add_edge(i, other, EdgeKind::Anchor);
        }
      }
    }
  }

  [[nodiscard]] uint64_t sort_key(NodeId nodeId) const noexcept {
    const CompiledNode& node = m_graph.nodes[nodeId];
    if (node.kind != NodeKind::Pass) {
      return uint64_t{1} << 63 | uint64_t{node.chainIndex} << 16 | nodeId;
    }
    return uint64_t{node.order} << 32 | uint64_t{node.registration} << 16 | nodeId;
  }

  bool sort_stage(Stage stage, size_t edgeCount, std::vector<NodeId>& order) {
    order.clear();
    std::vector<NodeId>& members = m_members;
    members.clear();
    const auto& chain = ChainRanges[static_cast<size_t>(stage)];
    for (uint16_t k = 0; k < chain.count; ++k) {
      members.push_back(chain_node_id(stage, k));
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (active(i) && m_graph.nodes[i].stage == stage) {
        members.push_back(i);
      }
    }
    for (size_t i = 0; i < members.size(); ++i) {
      m_localIndex[members[i]] = static_cast<uint16_t>(i);
    }
    std::vector<uint32_t> indegree(members.size(), 0);
    std::vector<uint32_t> offsets(members.size() + 1, 0);
    const std::span edges{m_edges.data(), edgeCount};
    for (const Edge& edge : edges) {
      if (m_localIndex[edge.from] != InvalidNode && m_localIndex[edge.to] != InvalidNode) {
        ++offsets[m_localIndex[edge.from] + 1];
        ++indegree[m_localIndex[edge.to]];
      }
    }
    for (size_t i = 0; i < members.size(); ++i) {
      offsets[i + 1] += offsets[i];
    }
    std::vector<NodeId> targets(offsets.back());
    std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
    for (const Edge& edge : edges) {
      if (m_localIndex[edge.from] != InvalidNode && m_localIndex[edge.to] != InvalidNode) {
        targets[fill[m_localIndex[edge.from]]++] = edge.to;
      }
    }
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> ready;
    for (size_t i = 0; i < members.size(); ++i) {
      if (indegree[i] == 0) {
        ready.push(sort_key(members[i]));
      }
    }
    while (!ready.empty()) {
      const auto node = static_cast<NodeId>(ready.top() & 0xFFFF);
      ready.pop();
      order.push_back(node);
      const uint16_t local = m_localIndex[node];
      for (uint32_t e = offsets[local]; e < offsets[local + 1]; ++e) {
        if (--indegree[m_localIndex[targets[e]]] == 0) {
          ready.push(sort_key(targets[e]));
        }
      }
    }
    const bool sorted = order.size() == members.size();
    if (!sorted) {
      break_cycle(edges, indegree);
    }
    for (const NodeId member : members) {
      m_localIndex[member] = InvalidNode;
    }
    return sorted;
  }

  // Requires m_localIndex filled for the stalled stage; nodes left with a nonzero in-degree were never emitted
  void break_cycle(std::span<const Edge> edges, const std::vector<uint32_t>& indegree) {
    const auto remains = [&](NodeId node) {
      const uint16_t local = m_localIndex[node];
      return local != InvalidNode && indegree[local] != 0;
    };
    NodeId start = InvalidNode;
    for (const NodeId member : m_members) {
      if (remains(member) && (start == InvalidNode || sort_key(member) < sort_key(start))) {
        start = member;
      }
    }
    // Walking incoming edges always stays on the cycle; outgoing ones can dead-end on a node that only hangs off it
    std::unordered_map<NodeId, size_t> step;
    std::vector<NodeId> path;
    std::vector<size_t> pathEdges;
    NodeId current = start;
    while (!step.contains(current)) {
      step.emplace(current, path.size());
      path.push_back(current);
      size_t incoming = edges.size();
      for (size_t e = 0; e < edges.size(); ++e) {
        if (edges[e].to == current && remains(edges[e].from)) {
          incoming = e;
          break;
        }
      }
      pathEdges.push_back(incoming);
      current = edges[incoming].from;
    }
    const size_t first = step[current];
    std::vector<size_t> cycle;
    for (size_t i = pathEdges.size(); i-- > first;) {
      cycle.push_back(pathEdges[i]);
    }
    const auto find_edge = [&](auto&& predicate) -> const Edge* {
      for (const size_t e : cycle) {
        if (predicate(edges[e].kind)) {
          return &edges[e];
        }
      }
      return nullptr;
    };
    const Edge* reported = find_edge([](EdgeKind kind) { return kind >= EdgeKind::Raw; });
    if (reported == nullptr) {
      reported = find_edge([](EdgeKind kind) { return kind == EdgeKind::Anchor; });
    }
    if (reported == nullptr) {
      reported = find_edge([](EdgeKind kind) { return kind == EdgeKind::Bound; });
    }
    if (reported == nullptr) {
      reported = &edges[cycle.front()];
    }
    NodeId a = reported->from;
    NodeId b = reported->to;
    if (reported->kind == EdgeKind::Bound && m_graph.nodes[a].kind != NodeKind::Pass) {
      std::swap(a, b);
    }
    NodeId victim = InvalidNode;
    for (size_t i = first; i < path.size(); ++i) {
      const NodeId node = path[i];
      if (m_graph.nodes[node].kind != NodeKind::Pass) {
        continue;
      }
      if (victim == InvalidNode || std::pair{m_graph.nodes[node].cls, m_graph.nodes[node].registration} >
                                       std::pair{m_graph.nodes[victim].cls, m_graph.nodes[victim].registration}) {
        victim = node;
      }
    }
    const NodeId other = a == victim ? b : a;
    std::string message;
    if (reported->kind >= EdgeKind::Raw) {
      message = fmt::format("render graph: '{}' and '{}' form a cycle through {} on '{}'; disabled '{}'", node_name(a),
                            node_name(b), edge_kind_name(reported->kind), resource_name(reported->resource),
                            node_name(victim));
    } else {
      message = fmt::format("render graph: '{}' and '{}' form a cycle through {}; disabled '{}'", node_name(a),
                            node_name(b), reported->kind == EdgeKind::Anchor ? "anchor" : "bound", node_name(victim));
    }
    disable(victim, CullReason::Cycle, std::move(message), other, reported->resource);
  }

  void add_data_edges() {
    for (CompiledUse& use : m_graph.uses) {
      use.producer = InvalidNode;
      use.unresolved = false;
    }
    std::vector<uint32_t> sequence;
    std::vector<NodeId> readers;
    for (size_t r = 0; r < m_graph.resources.size(); ++r) {
      const CompiledResource& resource = m_graph.resources[r];
      sequence.clear();
      for (const uint32_t index : m_resourceUses[r]) {
        const CompiledUse& use = m_graph.uses[index];
        if (active(use.node) && use.role != UseRole::Prev && !use.staticDummy) {
          sequence.push_back(index);
        }
      }
      if (sequence.empty()) {
        continue;
      }
      const auto position = [&](uint32_t index) {
        const NodeId node = m_graph.uses[index].node;
        return std::pair{m_graph.nodes[node].stage, m_position[node]};
      };
      std::ranges::sort(sequence, {}, position);
      const bool created = resource.kind == ResourceKind::Transient || resource.kind == ResourceKind::History;
      if (created && resource.creator != InvalidNode && active(resource.creator)) {
        const auto creatorUse = std::ranges::find_if(
            sequence, [&](uint32_t index) { return m_graph.uses[index].node == resource.creator; });
        if (creatorUse != sequence.end()) {
          const Stage stage = m_graph.nodes[resource.creator].stage;
          const auto stageStart = std::ranges::find_if(
              sequence, [&](uint32_t index) { return m_graph.nodes[m_graph.uses[index].node].stage == stage; });
          std::rotate(stageStart, creatorUse, creatorUse + 1);
        }
      }
      NodeId lastWriter = InvalidNode;
      readers.clear();
      for (const uint32_t index : sequence) {
        CompiledUse& use = m_graph.uses[index];
        const NodeId node = use.node;
        const Stage stage = m_graph.nodes[node].stage;
        const bool pure = is_pure_write(use);
        const bool writer = writes(use.access);
        const bool reader = reads(use.access) || (writer && !pure);
        const bool sameStage = lastWriter != InvalidNode && m_graph.nodes[lastWriter].stage == stage;
        if (reader) {
          use.producer = lastWriter;
          use.unresolved = lastWriter == InvalidNode && (created || resource.kind == ResourceKind::Gx);
          if (sameStage) {
            add_edge(lastWriter, node, EdgeKind::Raw, static_cast<ResourceId>(r));
          }
        }
        if (writer) {
          for (const NodeId other : readers) {
            if (other != node && m_graph.nodes[other].stage == stage) {
              add_edge(other, node, EdgeKind::War, static_cast<ResourceId>(r));
            }
          }
          if (sameStage && !reader) {
            add_edge(lastWriter, node, EdgeKind::Waw, static_cast<ResourceId>(r));
          }
          lastWriter = node;
          readers.clear();
        } else if (reader) {
          readers.push_back(node);
        }
      }
    }
  }

  void order_stages() {
    m_position.assign(m_graph.nodes.size(), 0);
    m_localIndex.assign(m_graph.nodes.size(), InvalidNode);
    std::vector<NodeId> order;
    std::array<std::vector<NodeId>, StageCount> programs;
    for (;;) {
      apply_missing();
      m_edges.clear();
      m_edgeKeys.clear();
      add_placement_edges();
      const size_t nominalEdges = m_edges.size();
      bool sorted = true;
      for (size_t s = 0; s < StageCount && sorted; ++s) {
        sorted = sort_stage(static_cast<Stage>(s), nominalEdges, order);
        for (size_t i = 0; sorted && i < order.size(); ++i) {
          m_position[order[i]] = static_cast<uint32_t>(i);
        }
      }
      if (!sorted) {
        continue;
      }
      add_data_edges();
      for (size_t s = 0; s < StageCount && sorted; ++s) {
        sorted = sort_stage(static_cast<Stage>(s), m_edges.size(), programs[s]);
      }
      if (sorted) {
        break;
      }
    }
    for (size_t s = 0; s < StageCount; ++s) {
      m_graph.stages[s].nodes = std::move(programs[s]);
    }
    m_graph.edges = std::move(m_edges);
  }

  void mark_sinks() {
    for (size_t i = 0; i < m_graph.nodes.size(); ++i) {
      CompiledNode& node = m_graph.nodes[i];
      if (node.kind != NodeKind::Pass) {
        node.sink = true;
        continue;
      }
      if (node.status != CullReason::None) {
        continue;
      }
      node.sink = node.sideEffect || (node.cls == PassClass::Diagnostic && m_inputs.env.inspectorOpen) ||
                  std::ranges::any_of(uses_of(static_cast<NodeId>(i)), [&](const CompiledUse& use) {
                    const ResourceKind kind = m_graph.resources[use.resource].kind;
                    return writes(use.access) && kind != ResourceKind::Transient;
                  });
    }
  }

  void cull_unconsumed() {
    mark_sinks();
    std::vector<bool> live(m_graph.nodes.size(), false);
    std::vector<NodeId> pending;
    for (size_t i = 0; i < m_graph.nodes.size(); ++i) {
      if (active(static_cast<NodeId>(i)) && m_graph.nodes[i].sink) {
        live[i] = true;
        pending.push_back(static_cast<NodeId>(i));
      }
    }
    // A modifier also reads the version it changes, so walking producers keeps the earlier writers live too
    while (!pending.empty()) {
      const NodeId node = pending.back();
      pending.pop_back();
      for (const CompiledUse& use : uses_of(node)) {
        if (!use.staticDummy && use.producer != InvalidNode && !live[use.producer]) {
          live[use.producer] = true;
          pending.push_back(use.producer);
        }
      }
    }
    for (NodeId i = 0; i < m_passCount; ++i) {
      if (active(i) && !live[i]) {
        disable(i, CullReason::NoConsumer,
                fmt::format("render graph: nothing reads what '{}' writes; culled", m_graph.nodes[i].name));
      }
    }
    for (StageProgram& program : m_graph.stages) {
      std::erase_if(program.nodes, [&](NodeId node) { return !active(node); });
    }
    std::erase_if(m_graph.edges, [&](const Edge& edge) { return !active(edge.from) || !active(edge.to); });
  }

  [[nodiscard]] const CompiledUse* find_use(NodeId node, ResourceId resource) noexcept {
    for (const CompiledUse& use : uses_of(node)) {
      if (use.resource == resource) {
        return &use;
      }
    }
    return nullptr;
  }

  void collect_consumers(const std::vector<uint32_t>& position) {
    std::vector<std::vector<NodeId>> lists(m_graph.nodes.size());
    for (const StageProgram& program : m_graph.stages) {
      for (const NodeId consumer : program.nodes) {
        for (const CompiledUse& use : uses_of(consumer)) {
          if (use.staticDummy || use.role == UseRole::Prev) {
            continue;
          }
          // Reading a modified version depends on every writer of that version, not only the last one
          for (NodeId producer = use.producer; producer != InvalidNode;) {
            lists[producer].push_back(consumer);
            const CompiledUse* written = find_use(producer, use.resource);
            if (written == nullptr || !writes(written->access) || is_pure_write(*written)) {
              break;
            }
            producer = written->producer;
          }
        }
      }
    }
    for (size_t i = 0; i < lists.size(); ++i) {
      auto& list = lists[i];
      std::ranges::sort(list, {}, [&](NodeId node) { return std::pair{m_graph.nodes[node].stage, position[node]}; });
      const auto duplicates = std::ranges::unique(list);
      list.erase(duplicates.begin(), duplicates.end());
      CompiledNode& node = m_graph.nodes[i];
      node.firstConsumer = static_cast<uint32_t>(m_graph.consumers.size());
      node.consumerCount = static_cast<uint32_t>(list.size());
      m_graph.consumers.insert(m_graph.consumers.end(), list.begin(), list.end());
    }
  }

  void finish_programs() {
    std::vector<uint32_t> position(m_graph.nodes.size(), 0);
    for (StageProgram& program : m_graph.stages) {
      for (size_t i = 0; i < program.nodes.size(); ++i) {
        const NodeId node = program.nodes[i];
        position[node] = static_cast<uint32_t>(i);
        if (m_graph.nodes[node].kind == NodeKind::Pass) {
          m_graph.nodes[node].programIndex = static_cast<uint16_t>(program.passes.size());
          program.passes.push_back(node);
        }
      }
    }
    for (CompiledResource& resource : m_graph.resources) {
      if (resource.described && resource.kind != ResourceKind::Gx) {
        resource.textureUsage = resource.buffer ? wgpu::TextureUsage::None : resource.tex.extraUsage;
        resource.bufferUsage = resource.buffer ? resource.buf.extraUsage : wgpu::BufferUsage::None;
      }
    }
    for (CompiledUse& use : m_graph.uses) {
      if (!active(use.node) || use.staticDummy) {
        continue;
      }
      CompiledResource& resource = m_graph.resources[use.resource];
      if (resource.buffer) {
        resource.bufferUsage = resource.bufferUsage | buffer_usage(use.access);
      } else {
        resource.textureUsage = resource.textureUsage | texture_usage(use.access);
      }
    }
    for (const StageProgram& program : m_graph.stages) {
      for (const NodeId node : program.passes) {
        m_graph.hasNativePasses = m_graph.hasNativePasses || m_graph.nodes[node].passKind == PassKind::Native;
        m_graph.nodes[node].frameSlot = static_cast<uint16_t>(m_graph.framePasses.size());
        m_graph.framePasses.push_back(node);
      }
    }
    // Only histories something writes get a slot; a prev read of an ownerless one is a static dummy
    for (size_t r = 0; r < m_graph.resources.size(); ++r) {
      const CompiledResource& resource = m_graph.resources[r];
      if (resource.kind == ResourceKind::History && resource.creator != InvalidNode && active(resource.creator)) {
        m_graph.histories.push_back(static_cast<ResourceId>(r));
      }
    }
    std::ranges::sort(m_graph.histories, {},
                      [&](ResourceId r) -> std::string_view { return m_graph.resources[r].name; });
    for (size_t h = 0; h < m_graph.histories.size(); ++h) {
      m_graph.resources[m_graph.histories[h]].historySlot = static_cast<uint16_t>(h);
    }
    const GraphFeatures& features = m_inputs.env.limits.features;
    for (StageProgram& program : m_graph.stages) {
      uint32_t stageUse = 0;
      for (const NodeId node : program.nodes) {
        const bool chain = m_graph.nodes[node].kind != NodeKind::Pass;
        for (CompiledUse& use : uses_of(node)) {
          use.stageUse = stageUse++;
          // A GX read cannot disable its chain node, so it always needs something to bind
          const bool dummyable = use.staticDummy || chain || use.missing != Missing::DisablePass;
          if ((use.role == UseRole::Use || use.role == UseRole::Prev) && reads(use.access) && dummyable) {
            use.hasFallback = true;
            use.fallback = dummy_key(m_graph, use, features);
          }
        }
      }
      program.useBits = stageUse;
    }
    collect_consumers(position);
  }

  void hash_graph() {
    Hasher hasher{0};
    constexpr std::string_view Tag = "aurora.render.graph/1";
    hasher.update(Tag.data(), Tag.size());
    hasher.update(m_graph.efbSize.width);
    hasher.update(m_graph.efbSize.height);
    hasher.update(m_graph.outputSize.width);
    hasher.update(m_graph.outputSize.height);
    std::vector<bool> referenced(m_graph.resources.size(), false);
    for (const StageProgram& program : m_graph.stages) {
      hasher.update(static_cast<uint32_t>(program.nodes.size()));
      for (const NodeId nodeId : program.nodes) {
        const CompiledNode& node = m_graph.nodes[nodeId];
        hash_string(hasher, node.name);
        hasher.update(static_cast<uint8_t>(node.kind));
        hasher.update(static_cast<uint8_t>(node.passKind));
        hasher.update(static_cast<uint8_t>(node.cls));
        const auto flags = static_cast<uint8_t>(uint8_t{node.sideEffect} | uint8_t{node.manualBindings} << 1 |
                                                uint8_t{node.sink} << 2 | uint8_t{node.runsWithoutMarker} << 3 |
                                                uint8_t{node.opaque} << 4);
        hasher.update(flags);
        hasher.update(node.useCount);
        for (const CompiledUse& use : uses_of(nodeId)) {
          referenced[use.resource] = true;
          hash_string(hasher, m_graph.resources[use.resource].name);
          hasher.update(static_cast<uint8_t>(use.role));
          hasher.update(static_cast<uint8_t>(use.access));
          hasher.update(static_cast<uint8_t>(use.missing));
          hasher.update(static_cast<uint8_t>(use.buffer));
          hasher.update(static_cast<uint64_t>(use.visibility));
          hasher.update(static_cast<uint32_t>(use.options.dimension));
          hasher.update(static_cast<uint32_t>(use.options.sampleType));
          hasher.update(static_cast<uint32_t>(use.options.format));
          hasher.update(use.options.dummySize);
          hasher.update(static_cast<uint8_t>(use.staticDummy));
          hasher.update(static_cast<uint8_t>(use.hasFallback));
          if (use.hasFallback) {
            hasher.update(static_cast<uint8_t>(use.fallback.buffer));
            hasher.update(static_cast<uint8_t>(use.fallback.color));
            hasher.update(static_cast<uint32_t>(use.fallback.dimension));
            hasher.update(static_cast<uint32_t>(use.fallback.sampleType));
            hasher.update(static_cast<uint32_t>(use.fallback.storageFormat));
            hasher.update(use.fallback.bufferSize);
          }
        }
      }
    }
    std::vector<ResourceId> resources;
    for (size_t r = 0; r < referenced.size(); ++r) {
      if (referenced[r]) {
        resources.push_back(static_cast<ResourceId>(r));
      }
    }
    std::ranges::sort(resources, {}, [&](ResourceId r) -> std::string_view { return m_graph.resources[r].name; });
    hasher.update(static_cast<uint32_t>(resources.size()));
    for (const ResourceId r : resources) {
      const CompiledResource& resource = m_graph.resources[r];
      hash_string(hasher, resource.name);
      hasher.update(static_cast<uint8_t>(resource.kind));
      hasher.update(static_cast<uint8_t>(resource.buffer));
      if (resource.buffer) {
        hasher.update(resource.bufferSize);
        hasher.update(static_cast<uint64_t>(resource.bufferUsage));
      } else {
        hasher.update(static_cast<uint32_t>(resource.tex.format));
        hasher.update(static_cast<uint32_t>(resource.tex.dimension));
        hasher.update(resource.extent.width);
        hasher.update(resource.extent.height);
        hasher.update(resource.extent.depthOrArrayLayers);
        hasher.update(resource.mipLevels);
        hasher.update(static_cast<uint64_t>(resource.textureUsage));
      }
      if (resource.kind == ResourceKind::Persistent || resource.kind == ResourceKind::External) {
        hasher.update(resource.handle.id);
        hasher.update(static_cast<uint8_t>(resource.owner));
      }
    }
    m_graph.graphHash = hasher.digest();
  }

  const GraphInputs& m_inputs;
  CompiledGraph& m_graph;
  NodeId m_passCount = 0;
  NodeId m_chainBase = 0;

  std::vector<DeclaredUse> m_declared;
  std::vector<Anchor> m_anchors;
  size_t m_pendingNames = 0;
  bool m_skipped = false;
  bool m_sideEffect = false;
  bool m_manualBindings = false;
  std::optional<std::string> m_rejection;
  std::array<std::byte, PassLocalSize> m_scratch{};

  std::vector<std::vector<CompiledUse>> m_nodeUses;
  std::vector<std::vector<Anchor>> m_nodeAnchors;
  std::unordered_map<std::string, ResourceId> m_resourceIds;
  std::unordered_map<std::string_view, NodeId> m_passNames;
  std::vector<NodeId> m_describers;
  std::vector<UseRole> m_describerRoles;
  std::vector<bool> m_mentioned;
  std::vector<Placement> m_placements;
  std::vector<bool> m_boundAsStorage;
  std::vector<bool> m_boundAsUniform;
  std::vector<std::vector<uint32_t>> m_resourceUses;
  std::vector<uint32_t> m_position;
  std::vector<uint16_t> m_localIndex;
  std::vector<NodeId> m_members;
  std::vector<Edge> m_edges;
  std::unordered_set<uint64_t> m_edgeKeys;
};

InternalPassBuilder::InternalPassBuilder(GraphBuild& build, NodeId node, bool gxReadsOnly) noexcept
: m_build(&build), m_node(node), m_gxReadsOnly(gxReadsOnly) {}

TexRef InternalPassBuilder::create(std::string_view name, const TexDesc& desc, Access access) {
  return {m_build->declare(*this, {.name = std::string{name}, .role = UseRole::Create, .access = access, .tex = desc})};
}

BufRef InternalPassBuilder::create_buffer(std::string_view name, const BufDesc& desc, Access access) {
  return {m_build->declare(
      *this, {.name = std::string{name}, .role = UseRole::Create, .access = access, .buffer = true, .buf = desc})};
}

TexRef InternalPassBuilder::history(std::string_view name, const TexDesc& desc, Access access) {
  return {
      m_build->declare(*this, {.name = std::string{name}, .role = UseRole::History, .access = access, .tex = desc})};
}

TexRef InternalPassBuilder::use(std::string_view name, Access access, Missing missing) {
  return use(name, access, UseOptions{.missing = missing});
}

TexRef InternalPassBuilder::use(std::string_view name, Access access, const UseOptions& options) {
  return {m_build->declare(*this, {.name = std::string{name}, .access = access, .options = options})};
}

TexRef InternalPassBuilder::use_prev(std::string_view name, Missing missing) {
  return {m_build->declare(
      *this,
      {.name = std::string{name}, .role = UseRole::Prev, .access = Access::Sampled, .options = {.missing = missing}})};
}

BufRef InternalPassBuilder::use_buffer(std::string_view name, Access access, Missing missing) {
  return use_buffer(name, access, UseOptions{.missing = missing});
}

BufRef InternalPassBuilder::use_buffer(std::string_view name, Access access, const UseOptions& options) {
  return {m_build->declare(*this, {.name = std::string{name}, .access = access, .buffer = true, .options = options})};
}

BufRef InternalPassBuilder::import_persistent(std::string_view name, PersistentHandle handle, Access access,
                                              Owner owner) {
  return {m_build->declare(*this, {.name = std::string{name},
                                   .role = UseRole::Import,
                                   .access = access,
                                   .buffer = true,
                                   .handle = handle,
                                   .owner = owner})};
}

TexRef InternalPassBuilder::import_persistent_texture(std::string_view name, PersistentHandle handle,
                                                      const TexDesc& desc, Access access, Owner owner) {
  return {m_build->declare(*this, {.name = std::string{name},
                                   .role = UseRole::Import,
                                   .access = access,
                                   .tex = desc,
                                   .handle = handle,
                                   .owner = owner})};
}

void InternalPassBuilder::after(std::string_view passOrPhase) { m_build->declare_anchor(*this, passOrPhase, true); }

void InternalPassBuilder::before(std::string_view passOrPhase) { m_build->declare_anchor(*this, passOrPhase, false); }

void InternalPassBuilder::side_effect() { m_build->declare_side_effect(*this); }

void InternalPassBuilder::manual_bindings() { m_build->declare_manual_bindings(*this); }

void InternalPassBuilder::skip() { m_build->declare_skip(*this); }

std::span<std::byte, PassLocalSize> InternalPassBuilder::local() noexcept { return m_build->local(*this); }

PassId register_pass(const PassDesc& desc) {
  if (desc.name.empty()) {
    Log.warn("register_pass: pass name is empty");
    return InvalidPassId;
  }
  if (desc.encode == nullptr) {
    Log.warn("register_pass: '{}' has no encode callback", desc.name);
    return InvalidPassId;
  }
  if (desc.kind == PassKind::Native) {
    Log.warn("register_pass: '{}' rejected; native passes need the D3D12 backend", desc.name);
    return InvalidPassId;
  }
  if (desc.stage == Stage::AfterFrame) {
    Log.warn("register_pass: '{}' rejected; AfterFrame runs native passes only", desc.name);
    return InvalidPassId;
  }
  if (static_cast<size_t>(desc.stage) >= StageCount) {
    Log.warn("register_pass: '{}' has unknown stage {}", desc.name, static_cast<uint32_t>(desc.stage));
    return InvalidPassId;
  }

  std::lock_guard lock{g_passMutex};
  if (has_live_pass_named(desc.name)) {
    Log.warn("register_pass: a pass named '{}' is already registered", desc.name);
    return InvalidPassId;
  }
  uint32_t index;
  if (!g_freePassSlots.empty()) {
    index = g_freePassSlots.back();
    g_freePassSlots.pop_back();
  } else {
    index = static_cast<uint32_t>(g_passSlots.size());
    g_passSlots.emplace_back();
  }
  auto& slot = g_passSlots[index];
  const PassId id = make_pass_id(index, slot.generation);
  slot.pass = RegisteredPass{.desc = desc, .id = id, .sequence = g_nextSequence++};
  g_graphDirty = true;
  return id;
}

void unregister_pass(PassId id) noexcept {
  std::lock_guard lock{g_passMutex};
  auto* slot = find_pass_slot_locked(id);
  if (slot == nullptr) {
    return;
  }
  slot->pass.reset();
  ++slot->generation;
  g_freePassSlots.push_back(pass_index(id));
  g_graphDirty = true;
}

bool consume_graph_dirty() noexcept { return g_graphDirty.exchange(false); }

std::shared_ptr<CompiledGraph> compile_graph(const GraphInputs& inputs) {
  auto graph = std::make_shared<CompiledGraph>();
  GraphBuild{inputs, *graph}.compile();
  return graph;
}

std::shared_ptr<CompiledGraph> compile_graph(const Settings& settings, const Caps& caps, const GraphEnvironment& env) {
  std::vector<RegisteredPass> registered;
  {
    std::lock_guard lock{g_passMutex};
    for (const auto& slot : g_passSlots) {
      if (slot.pass) {
        registered.push_back(*slot.pass);
      }
    }
  }
  if (registered.empty()) {
    return nullptr;
  }
  std::ranges::sort(registered, {}, &RegisteredPass::sequence);
  std::vector<PassDesc> passes;
  std::vector<PassId> ids;
  passes.reserve(registered.size());
  ids.reserve(registered.size());
  for (auto& pass : registered) {
    passes.push_back(std::move(pass.desc));
    ids.push_back(pass.id);
  }
  return compile_graph(GraphInputs{
      .passes = passes,
      .ids = ids,
      .settings = &settings,
      .caps = &caps,
      .env = env,
  });
}

GraphLimits graph_limits(const wgpu::Limits& limits, const wgpu::CompatibilityModeLimits& compat,
                         const GraphFeatures& features) noexcept {
  const auto u32 = [](uint32_t value, uint32_t fallback) {
    return value == wgpu::kLimitU32Undefined ? fallback : value;
  };
  const auto u64 = [](uint64_t value, uint64_t fallback) {
    return value == wgpu::kLimitU64Undefined ? fallback : value;
  };
  GraphLimits out;
  for (BindingCounts* stage : {&out.vertex, &out.fragment, &out.compute}) {
    stage->sampledTextures = u32(limits.maxSampledTexturesPerShaderStage, stage->sampledTextures);
    stage->samplers = u32(limits.maxSamplersPerShaderStage, stage->samplers);
    stage->uniformBuffers = u32(limits.maxUniformBuffersPerShaderStage, stage->uniformBuffers);
  }
  out.compute.storageBuffers = u32(limits.maxStorageBuffersPerShaderStage, out.compute.storageBuffers);
  out.compute.storageTextures = u32(limits.maxStorageTexturesPerShaderStage, out.compute.storageTextures);
  out.vertex.storageBuffers = u32(compat.maxStorageBuffersInVertexStage, out.vertex.storageBuffers);
  out.vertex.storageTextures = u32(compat.maxStorageTexturesInVertexStage, out.vertex.storageTextures);
  out.fragment.storageBuffers = u32(compat.maxStorageBuffersInFragmentStage, out.fragment.storageBuffers);
  out.fragment.storageTextures = u32(compat.maxStorageTexturesInFragmentStage, out.fragment.storageTextures);
  out.bindingsPerGroup = u32(limits.maxBindingsPerBindGroup, out.bindingsPerGroup);
  out.dynamicUniformBuffers = u32(limits.maxDynamicUniformBuffersPerPipelineLayout, out.dynamicUniformBuffers);
  out.colorAttachments = u32(limits.maxColorAttachments, out.colorAttachments);
  out.colorAttachmentBytesPerSample = u32(limits.maxColorAttachmentBytesPerSample, out.colorAttachmentBytesPerSample);
  out.vertexBuffers = u32(limits.maxVertexBuffers, out.vertexBuffers);
  out.textureDimension2D = u32(limits.maxTextureDimension2D, out.textureDimension2D);
  out.textureDimension3D = u32(limits.maxTextureDimension3D, out.textureDimension3D);
  out.textureArrayLayers = u32(limits.maxTextureArrayLayers, out.textureArrayLayers);
  out.bufferSize = u64(limits.maxBufferSize, out.bufferSize);
  out.storageBufferBindingSize = u64(limits.maxStorageBufferBindingSize, out.storageBufferBindingSize);
  out.uniformBufferBindingSize = u64(limits.maxUniformBufferBindingSize, out.uniformBufferBindingSize);
  out.features = features;
  return out;
}

std::span<const ChainNode> chain_nodes() noexcept { return ChainNodes; }

std::span<const ChainNode> stage_chain(Stage stage) noexcept {
  const auto index = static_cast<size_t>(stage);
  if (index >= StageCount) {
    return {};
  }
  return std::span{ChainNodes}.subspan(ChainRanges[index].first, ChainRanges[index].count);
}

std::optional<Stage> chain_node_stage(std::string_view name) noexcept {
  const auto index = find_chain_node(name);
  return index ? std::optional{ChainNodes[*index].stage} : std::nullopt;
}

std::optional<GxResource> find_gx_resource(std::string_view name) noexcept {
  for (size_t i = 0; i < GxRows.size(); ++i) {
    if (GxRows[i].name == name) {
      return static_cast<GxResource>(i);
    }
  }
  return std::nullopt;
}

wgpu::TextureFormat gx_fixed_format(GxResource resource) noexcept {
  const auto index = static_cast<size_t>(resource);
  return index < GxRows.size() ? GxRows[index].format : wgpu::TextureFormat::Undefined;
}

std::optional<ColorTargetCost> color_target_cost(wgpu::TextureFormat format) noexcept {
  const FormatRow* row = find_format(format);
  if (row == nullptr || row->colorBytes == 0) {
    return std::nullopt;
  }
  return ColorTargetCost{row->colorBytes, row->colorAlignment};
}

uint32_t color_bytes_per_sample(std::span<const wgpu::TextureFormat> formats) noexcept {
  uint32_t total = 0;
  for (const auto format : formats) {
    if (const auto cost = color_target_cost(format)) {
      total = (total + cost->alignment - 1) / cost->alignment * cost->alignment + cost->bytes;
    }
  }
  return total;
}

std::string_view gx_resource_name(GxResource resource) noexcept {
  const auto index = static_cast<size_t>(resource);
  return index < GxRows.size() ? GxRows[index].name : "Unknown";
}

std::string_view access_name(Access access) noexcept { return enum_name(AccessNames, access); }

std::string_view missing_name(Missing missing) noexcept { return enum_name(MissingNames, missing); }

std::string_view pass_kind_name(PassKind kind) noexcept { return enum_name(PassKindNames, kind); }

std::string_view pass_class_name(PassClass cls) noexcept { return enum_name(PassClassNames, cls); }

std::string_view resource_kind_name(ResourceKind kind) noexcept { return enum_name(ResourceKindNames, kind); }

std::string_view cull_reason_name(CullReason reason) noexcept { return enum_name(CullReasonNames, reason); }

std::string_view edge_kind_name(EdgeKind kind) noexcept { return enum_name(EdgeKindNames, kind); }

DummyKey dummy_key(const CompiledGraph& graph, const CompiledUse& use, const GraphFeatures& features) noexcept {
  DummyKey key{.color = use.missing == Missing::BindDummyWhite ? DummyColor::White : DummyColor::Black};
  if (use.buffer) {
    constexpr uint64_t Largest = 0xFFFFFFF0u;
    const uint64_t size = std::min<uint64_t>(std::max<uint64_t>(16, use.options.dummySize), Largest);
    key.buffer = true;
    key.bufferSize = static_cast<uint32_t>((size + 15) & ~uint64_t{15});
    return key;
  }
  const CompiledResource& resource = graph.resources[use.resource];
  auto format = wgpu::TextureFormat::Undefined;
  auto dimension = wgpu::TextureViewDimension::e2D;
  if (resource.kind == ResourceKind::Gx) {
    format = resource.tex.format != wgpu::TextureFormat::Undefined ? resource.tex.format : gx_fixed_format(resource.gx);
  } else if (resource.described) {
    format = resource.tex.format;
    if (resource.tex.dimension == wgpu::TextureDimension::e3D) {
      dimension = wgpu::TextureViewDimension::e3D;
    } else if (resource.tex.depthOrLayers > 1) {
      dimension = wgpu::TextureViewDimension::e2DArray;
    }
  }
  key.dimension = use.options.dimension != wgpu::TextureViewDimension::Undefined ? use.options.dimension : dimension;
  key.sampleType = use.options.sampleType != wgpu::TextureSampleType::Undefined
                       ? use.options.sampleType
                       : format_sample_type(format, features.float32Filterable);
  if (use.access == Access::StorageRead) {
    key.storageFormat = format != wgpu::TextureFormat::Undefined ? format : use.options.format;
  }
  return key;
}

void begin_frame_cull(const CompiledGraph& graph, FrameCullState& state) noexcept {
  const size_t ranWords = bit_words(graph.framePasses.size());
  const size_t historyWords = bit_words(graph.histories.size());
  // Equal hashes give equal frame and history slots, so last frame's bits still describe this graph
  if (!state.primed || state.graphHash != graph.graphHash || state.ran.size() != ranWords ||
      state.prevValid.size() != historyWords) {
    state.ran.assign(ranWords, 0);
    state.prevValid.assign(historyWords, 0);
    state.graphHash = graph.graphHash;
    state.primed = true;
    state.nextStage = 0;
    state.cutStage = 0;
    return;
  }
  for (size_t h = 0; h < graph.histories.size(); ++h) {
    const CompiledResource& resource = graph.resources[graph.histories[h]];
    bool valid = false;
    if (resource.creator != InvalidNode) {
      const CompiledNode& owner = graph.nodes[resource.creator];
      // An owner culled before the cut wrote content the cut discards
      valid = test_bit(state.ran, owner.frameSlot) && static_cast<uint8_t>(owner.stage) >= state.cutStage;
    }
    assign_bit(state.prevValid, h, valid);
  }
  std::ranges::fill(state.ran, 0);
  state.nextStage = 0;
  state.cutStage = 0;
}

void invalidate_histories(FrameCullState& state) noexcept {
  std::ranges::fill(state.prevValid, 0);
  state.cutStage = state.nextStage;
}

void cull_stage(const CompiledGraph& graph, Stage stage, std::span<const uint64_t> enabled, FrameCullState& state,
                StageMask& out) {
  assert(static_cast<uint8_t>(stage) >= state.nextStage);
  const StageProgram& program = graph.stages[static_cast<size_t>(stage)];
  out.passes.assign(bit_words(program.passes.size()), 0);
  for (size_t i = 0; i < program.passes.size(); ++i) {
    assign_bit(out.passes, i, test_bit(enabled, i));
  }
  out.dummies.assign(bit_words(program.useBits), 0);
  out.prevValid = state.prevValid;
  const auto turn_off = [&](NodeId nodeId) {
    assign_bit(out.passes, graph.nodes[nodeId].programIndex, false);
    for (const CompiledUse& use : graph.node_uses(nodeId)) {
      assign_bit(out.dummies, use.stageUse, false);
    }
  };

  for (const NodeId nodeId : program.nodes) {
    const CompiledNode& node = graph.nodes[nodeId];
    const bool chain = node.kind != NodeKind::Pass;
    if (!chain && !test_bit(out.passes, node.programIndex)) {
      continue;
    }
    bool on = true;
    for (const CompiledUse& use : graph.node_uses(nodeId)) {
      if (use.staticDummy) {
        assign_bit(out.dummies, use.stageUse, true);
        continue;
      }
      if (!input_missing(graph, use, stage, state, out)) {
        continue;
      }
      if (!chain && (writes(use.access) || use.missing == Missing::DisablePass)) {
        on = false;
        break;
      }
      assign_bit(out.dummies, use.stageUse, true);
    }
    if (!on) {
      turn_off(nodeId);
    }
  }

  // Consumers in later stages are not culled yet, so they count as running
  for (auto it = program.nodes.rbegin(); it != program.nodes.rend(); ++it) {
    const CompiledNode& node = graph.nodes[*it];
    if (node.kind != NodeKind::Pass || node.sink || !test_bit(out.passes, node.programIndex)) {
      continue;
    }
    const auto consumers = std::span{graph.consumers}.subspan(node.firstConsumer, node.consumerCount);
    const bool consumed = std::ranges::any_of(consumers, [&](NodeId consumerId) {
      const CompiledNode& consumer = graph.nodes[consumerId];
      return consumer.stage > stage || consumer.kind != NodeKind::Pass || test_bit(out.passes, consumer.programIndex);
    });
    if (!consumed) {
      turn_off(*it);
    }
  }

  for (const NodeId nodeId : program.passes) {
    const CompiledNode& node = graph.nodes[nodeId];
    assign_bit(state.ran, node.frameSlot, test_bit(out.passes, node.programIndex));
  }
  state.nextStage = static_cast<uint8_t>(static_cast<size_t>(stage) + 1);
}

namespace testing {
void clear_passes() noexcept {
  std::lock_guard lock{g_passMutex};
  g_passSlots.clear();
  g_freePassSlots.clear();
  g_nextSequence = 1;
  g_graphDirty = false;
}

void reset_logged_diagnostics() noexcept {
  std::lock_guard lock{g_logMutex};
  g_loggedDiagnostics.clear();
}
} // namespace testing
} // namespace aurora::render::detail

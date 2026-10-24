#pragma once

#include <aurora/aurora.h>
#include <aurora/render.hpp>
#include <aurora/render_graph.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::render::debug {
struct GraphInspection;
} // namespace aurora::render::debug

namespace aurora::render::detail {
struct StageData;
class PassContext;
struct NativeContext;
struct ExternalPassSet;

using render::Access;
using render::AllMips;
using render::BufDesc;
using render::BufferView;
using render::BufRef;
using render::FrameBinding;
using render::FrameConstants;
using render::FrameFlagMultisampled;
using render::FrameFlagReversedZ;
using render::Missing;
using render::PassKind;
using render::SizeClass;
using render::TexDesc;
using render::TexRef;

enum class GxResource : uint8_t {
#define AURORA_RENDER_GX(id, label, format) id,
#include "phases.inl"
#undef AURORA_RENDER_GX
};
inline constexpr size_t GxResourceCount = 8;
inline constexpr size_t ChainNodeCount = 28;

// Public phase names and stage scheduling both rely on these rows following Stage order
constexpr bool chain_rows_follow_stages() noexcept {
  constexpr std::array rows{
#define AURORA_RENDER_NODE(stage, kind, name) Stage::stage,
#include "phases.inl"
#undef AURORA_RENDER_NODE
  };
  size_t next = 0;
  for (const Stage stage : rows) {
    const auto index = static_cast<size_t>(stage);
    if (index != next && index + 1 != next) {
      return false;
    }
    next = index + 1;
  }
  return rows.size() == ChainNodeCount && next == StageCount;
}
static_assert(chain_rows_follow_stages());

enum class ChainKind : uint8_t { Gx, Phase, Overlay };

struct ChainNode {
  Stage stage;
  ChainKind kind;
  std::string_view name;
};

constexpr bool binds(Access access) noexcept {
  return access == Access::Sampled || access == Access::Uniform || access == Access::StorageRead ||
         access == Access::StorageWrite || access == Access::StorageRW;
}

wgpu::BufferUsage buffer_usage(Access access) noexcept;

enum class PassClass : uint8_t { Required, Optional, Diagnostic };
enum class Owner : uint8_t { Dawn, NativeD3D12 };
enum class ResourceKind : uint8_t { Transient, History, Persistent, Gx, External };
enum class UseRole : uint8_t { Create, History, Use, Prev, Import, GxWrite };
enum class NodeKind : uint8_t { Pass, Gx, Phase, Overlay };
enum class EdgeKind : uint8_t { Chain, Bound, Anchor, Raw, War, Waw };
enum class CullReason : uint8_t {
  None,
  Skipped,
  InvalidDeclaration,
  DuplicateName,
  DuplicateCreator,
  ConflictingKind,
  WrongStage,
  AnchorConflict,
  BindingLimit,
  FormatUnsupported,
  ResourceLimit,
  BackwardsEdge,
  Cycle,
  MissingInput,
  MissingTarget,
  NoConsumer,
  InspectorClosed,
  UnknownAnchor,
  CoercedMissing,
  SampleCountMismatch,
  BindingViewMismatch,
};
enum class DummyColor : uint8_t { Black, White };

using NodeId = uint16_t;
using ResourceId = uint16_t;
inline constexpr NodeId InvalidNode = 0xFFFF;
inline constexpr ResourceId InvalidResource = 0xFFFF;
inline constexpr uint32_t InvalidIndex = 0xFFFFFFFF;
inline constexpr size_t MaxPasses = 4096;
inline constexpr size_t MaxResources = 4096;
inline constexpr size_t MaxNameLength = 64;
inline constexpr size_t PassLocalSize = 64;

struct PersistentHandle {
  uint32_t id = 0;
  bool operator==(const PersistentHandle&) const = default;
};

struct UseOptions {
  Missing missing = Missing::DisablePass;
  wgpu::ShaderStage visibility = wgpu::ShaderStage::None;
  wgpu::TextureViewDimension dimension = wgpu::TextureViewDimension::Undefined;
  wgpu::TextureSampleType sampleType = wgpu::TextureSampleType::Undefined;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  uint32_t dummySize = 256;
};

struct GraphFeatures {
  bool core = true;
  bool tier1Formats = false;
  bool tier2Formats = false;
  bool bgra8Storage = false;
  bool rg11b10Renderable = false;
  bool float32Filterable = false;
  bool operator==(const GraphFeatures&) const = default;
};

struct BindingCounts {
  uint32_t sampledTextures = 0;
  uint32_t samplers = 0;
  uint32_t storageBuffers = 0;
  uint32_t storageTextures = 0;
  uint32_t uniformBuffers = 0;
  bool operator==(const BindingCounts&) const = default;
};

inline constexpr BindingCounts DefaultStageLimits{16, 16, 8, 4, 12};
inline constexpr BindingCounts FrameGroupCost{.samplers = 4, .uniformBuffers = 2};
inline constexpr uint32_t FrameGroupDynamicUniformBuffers = 2;

struct GraphLimits {
  BindingCounts vertex = DefaultStageLimits;
  BindingCounts fragment = DefaultStageLimits;
  BindingCounts compute = DefaultStageLimits;
  uint32_t bindingsPerGroup = 1000;
  uint32_t dynamicUniformBuffers = 8;
  uint32_t colorAttachments = 8;
  uint32_t colorAttachmentBytesPerSample = 32;
  uint32_t vertexBuffers = 8;
  uint32_t textureDimension2D = 8192;
  uint32_t textureDimension3D = 2048;
  uint32_t textureArrayLayers = 256;
  uint64_t bufferSize = 256ull << 20;
  uint64_t storageBufferBindingSize = 128ull << 20;
  uint64_t uniformBufferBindingSize = 64ull << 10;
  GraphFeatures features;
  bool operator==(const GraphLimits&) const = default;
};

struct GraphExtent {
  uint32_t width = 0;
  uint32_t height = 0;
  bool operator==(const GraphExtent&) const = default;
};

struct GraphEnvironment {
  GraphLimits limits;
  GraphExtent efbSize;
  GraphExtent outputSize;
  std::array<wgpu::TextureFormat, GxResourceCount> gxFormats{};
  std::array<wgpu::TextureUsage, GxResourceCount> gxUsage{};
  uint32_t gxSampleCount = 1;
  AuroraFeatureProfile profile = FEATURE_PROFILE_FULL;
  bool inspectorOpen = false;
  bool operator==(const GraphEnvironment&) const = default;
};

struct PassSetup {
  const Settings* settings = nullptr;
  const Caps* caps = nullptr;
  const GraphEnvironment* env = nullptr;
  DepthSource sceneDepth = DepthSource::Unavailable;
  bool gx_available(GxResource resource) const noexcept {
    return env != nullptr && env->gxFormats[static_cast<size_t>(resource)] != wgpu::TextureFormat::Undefined;
  }
};

// A min-depth pyramid that a white dummy, filtering, a stale stage or a GX read would make over-cull
struct FarDepthPyramid {
  std::string_view name;
  Stage stage;
  std::string_view fence;
};
const FarDepthPyramid* find_far_depth_pyramid(std::string_view name) noexcept;

class InternalPassBuilder;
using PassDeclareFn = void (*)(InternalPassBuilder& builder, const PassSetup& setup, const void* state);
using PassEnabledFn = bool (*)(const StageData& stage, const void* state);
using PassEncodeFn = void (*)(PassContext& ctx, const void* state);
using PassEncodeNativeFn = void (*)(NativeContext& ctx, const void* state);

inline constexpr size_t MaxPassPipelines = 4;

struct PipelineSetup {
  const PassSetup* pass = nullptr;
  wgpu::BindGroupLayout frameLayout;
  wgpu::BindGroupLayout passLayout;
  wgpu::PipelineLayout layout;
  uint64_t layoutHash = 0;
  std::span<const std::byte, PassLocalSize> local;
};

using PassPipelinesFn = void (*)(const PipelineSetup& setup, const void* state,
                                 std::span<uint32_t, MaxPassPipelines> pipelines);

struct PassDesc {
  std::string name;
  Stage stage = Stage::SceneBegin;
  PassKind kind = PassKind::Encoder;
  PassClass cls = PassClass::Required;
  uint16_t order = 0;
  bool runsWithoutMarker = false;
  uint32_t paramsSize = 0;
  bool external = false;
  PassDeclareFn declare = nullptr;
  PassEnabledFn enabled = nullptr;
  PassEncodeFn encode = nullptr;
  PassEncodeNativeFn encodeNative = nullptr;
  PassPipelinesFn pipelines = nullptr;
  const void* state = nullptr;
};

using PassId = uint64_t;
inline constexpr PassId InvalidPassId = 0;

PassId register_pass(const PassDesc& desc);
void unregister_pass(PassId id) noexcept;
bool consume_graph_dirty() noexcept;

bool set_pass_params(PassId pass, std::span<const std::byte> payload);
void clear_pass_params(PassId pass) noexcept;
void latch_pass_params();
std::span<const std::byte> latched_pass_params(PassId pass) noexcept;

struct GxReadDesc {
  std::string_view node;
  PassDeclareFn declare = nullptr;
  const void* state = nullptr;
};

struct GraphInputs {
  std::span<const PassDesc> passes;
  std::span<const PassId> ids;
  std::span<const GxReadDesc> gxReads;
  const Settings* settings = nullptr;
  const Caps* caps = nullptr;
  GraphEnvironment env;
  std::shared_ptr<const ExternalPassSet> externals;
  DepthSource sceneDepthSource = DepthSource::Unavailable;
};

struct DummyKey {
  bool buffer = false;
  DummyColor color = DummyColor::Black;
  wgpu::TextureViewDimension dimension = wgpu::TextureViewDimension::e2D;
  wgpu::TextureSampleType sampleType = wgpu::TextureSampleType::Float;
  wgpu::TextureFormat storageFormat = wgpu::TextureFormat::Undefined;
  uint32_t bufferSize = 0;
  bool operator==(const DummyKey&) const = default;
};

struct CompiledUse {
  NodeId node = InvalidNode;
  ResourceId resource = InvalidResource;
  UseRole role = UseRole::Use;
  Access access = Access::Sampled;
  Missing missing = Missing::DisablePass;
  bool buffer = false;
  wgpu::ShaderStage visibility = wgpu::ShaderStage::None;
  UseOptions options;
  NodeId producer = InvalidNode;
  bool unresolved = false;
  bool staticDummy = false;
  bool hasFallback = false;
  DummyKey fallback;
  uint32_t stageUse = InvalidIndex;
};

struct CompiledResource {
  std::string name;
  ResourceKind kind = ResourceKind::Transient;
  bool buffer = false;
  bool described = false;
  TexDesc tex;
  BufDesc buf;
  wgpu::Extent3D extent{};
  uint32_t mipLevels = 1;
  uint64_t bufferSize = 0;
  wgpu::TextureUsage textureUsage = wgpu::TextureUsage::None;
  wgpu::BufferUsage bufferUsage = wgpu::BufferUsage::None;
  NodeId creator = InvalidNode;
  PersistentHandle handle;
  Owner owner = Owner::Dawn;
  GxResource gx = GxResource::Color;
  uint16_t historySlot = InvalidResource;
};

struct CompiledNode {
  std::string name;
  NodeKind kind = NodeKind::Pass;
  Stage stage = Stage::OpaqueDone;
  PassKind passKind = PassKind::Compute;
  PassClass cls = PassClass::Required;
  uint16_t order = 0;
  uint16_t registration = InvalidNode;
  uint16_t chainIndex = InvalidNode;
  PassId id = InvalidPassId;
  CullReason status = CullReason::None;
  bool opaque = false;
  bool sideEffect = false;
  bool manualBindings = false;
  bool runsWithoutMarker = false;
  bool external = false;
  bool sink = false;
  uint32_t firstUse = 0;
  uint32_t useCount = 0;
  uint32_t firstConsumer = 0;
  uint32_t consumerCount = 0;
  uint16_t programIndex = InvalidNode;
  uint16_t frameSlot = InvalidNode;
  std::array<BindingCounts, 3> bindings{};
  uint32_t colorBytes = 0;
  uint32_t paramsSize = 0;
  PassEnabledFn enabled = nullptr;
  PassEncodeFn encode = nullptr;
  PassEncodeNativeFn encodeNative = nullptr;
  PassPipelinesFn pipelines = nullptr;
  std::array<uint32_t, MaxPassPipelines> pipelineIds{};
  const void* state = nullptr;
  std::array<std::byte, PassLocalSize> local{};
};

struct Edge {
  NodeId from = InvalidNode;
  NodeId to = InvalidNode;
  EdgeKind kind = EdgeKind::Chain;
  ResourceId resource = InvalidResource;
};

struct Diagnostic {
  CullReason reason = CullReason::None;
  NodeId node = InvalidNode;
  NodeId other = InvalidNode;
  ResourceId resource = InvalidResource;
  std::string detail;
  std::string message;
};

struct StageProgram {
  std::vector<NodeId> passes;
  std::vector<NodeId> nodes;
  uint32_t useBits = 0;
};

struct TexKey {
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  wgpu::TextureDimension dimension = wgpu::TextureDimension::e2D;
  wgpu::TextureViewDimension bindingView = wgpu::TextureViewDimension::e2D;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depthOrLayers = 1;
  uint32_t mips = 1;
  wgpu::TextureUsage usage = wgpu::TextureUsage::None;
  SizeClass sizeClass = SizeClass::Absolute;

  bool operator==(const TexKey&) const = default;
};

struct BufKey {
  uint64_t bucketSize = 0;
  wgpu::BufferUsage usage = wgpu::BufferUsage::None;
  SizeClass sizeClass = SizeClass::Absolute;

  bool operator==(const BufKey&) const = default;
};

using SlotId = uint16_t;
inline constexpr SlotId InvalidSlot = UINT16_MAX;

struct Lifetime {
  uint32_t first = UINT32_MAX;
  uint32_t last = 0;
};

struct SlotPlan {
  struct Slot {
    bool isBuffer = false;
    TexKey tex;
    BufKey buf;
    std::string anchor;
  };
  struct History {
    std::string name;
    TexKey key;
  };
  std::vector<Slot> slots;
  std::vector<SlotId> resourceSlot;
  std::vector<Lifetime> lifetimes;
  std::vector<uint32_t> positions;
  std::vector<History> histories;
  std::vector<std::pair<uint32_t, ResourceId>> poisonPoints;
  GraphExtent efbSize;
  GraphExtent outputSize;
  uint64_t logicalBytes = 0;
  uint64_t slotBytes = 0;
};

enum class BindingType : uint8_t {
  None,
  UniformBuffer,
  ReadOnlyStorageBuffer,
  StorageBuffer,
  SampledTexture,
  ReadOnlyStorageTexture,
  WriteOnlyStorageTexture,
  ReadWriteStorageTexture,
};

struct BindingSlot {
  uint32_t use = InvalidIndex;
  ResourceId resource = InvalidResource;
  uint32_t binding = 0;
  BindingType type = BindingType::None;
  wgpu::ShaderStage visibility = wgpu::ShaderStage::None;
  wgpu::TextureViewDimension viewDimension = wgpu::TextureViewDimension::Undefined;
  wgpu::TextureSampleType sampleType = wgpu::TextureSampleType::Undefined;
  wgpu::TextureFormat storageFormat = wgpu::TextureFormat::Undefined;
  bool multisampled = false;
  bool prev = false;
};

struct AttachmentSlot {
  uint32_t use = InvalidIndex;
  ResourceId resource = InvalidResource;
  Access access = Access::ColorWrite;
  bool creatorWrite = false;
  uint32_t samples = 1;
};

// Defined with the Dawn objects in graph_bind.hpp, so graph code links without Dawn
struct BindingLayout;

struct BindingPlan {
  std::vector<BindingSlot> slots;
  std::vector<AttachmentSlot> colors;
  std::optional<AttachmentSlot> depth;
  uint32_t paramsSize = 0;
  uint64_t layoutHash = 0;
  bool manual = false;
  std::shared_ptr<const BindingLayout> layout;
  uint64_t layoutSerial = 0;
};

struct CompiledGraph {
  uint64_t graphHash = 0;
  GraphExtent efbSize;
  GraphExtent outputSize;
  bool hasNativePasses = false;
  std::vector<CompiledNode> nodes;
  std::vector<CompiledUse> uses;
  std::vector<CompiledResource> resources;
  std::vector<Edge> edges;
  std::vector<NodeId> consumers;
  std::vector<NodeId> framePasses;
  std::vector<ResourceId> histories;
  std::array<StageProgram, StageCount> stages{};
  std::vector<Diagnostic> diagnostics;
  std::vector<BindingPlan> bindingPlans;
  SlotPlan slots;
  std::shared_ptr<const ExternalPassSet> externals;
  DepthSource sceneDepthSource = DepthSource::Unavailable;
  std::array<std::vector<uint64_t>, StageCount> sceneDepthSamplers{};

  [[nodiscard]] bool stage_empty(Stage stage) const noexcept {
    return stages[static_cast<size_t>(stage)].passes.empty();
  }
  [[nodiscard]] NodeId find_node(std::string_view name) const noexcept {
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].name == name) {
        return static_cast<NodeId>(i);
      }
    }
    return InvalidNode;
  }
  [[nodiscard]] ResourceId find_resource(std::string_view name) const noexcept {
    for (size_t i = 0; i < resources.size(); ++i) {
      if (resources[i].name == name) {
        return static_cast<ResourceId>(i);
      }
    }
    return InvalidResource;
  }
  [[nodiscard]] std::span<const CompiledUse> node_uses(NodeId node) const noexcept {
    const auto& compiled = nodes[node];
    return std::span{uses}.subspan(compiled.firstUse, compiled.useCount);
  }
};

inline const std::shared_ptr<const CompiledGraph>& empty_graph() noexcept {
  static const std::shared_ptr<const CompiledGraph> graph = std::make_shared<const CompiledGraph>();
  return graph;
}

struct GraphBuild;

struct PassBuilderState {
  GraphBuild* build = nullptr;
  NodeId node = InvalidNode;
  bool gxReadsOnly = false;
  bool external = false;
  std::string_view prefix;
  Stage stage = Stage::SceneBegin;
  PassKind kind = PassKind::Compute;
};

class InternalPassBuilder : public PassBuilder {
public:
  using PassBuilder::use;
  using PassBuilder::use_buffer;

  TexRef use(std::string_view name, Access access, const UseOptions& options);
  TexRef use_prev(std::string_view name, Missing missing = Missing::BindDummyBlack);
  BufRef use_buffer(std::string_view name, Access access, const UseOptions& options);
  BufRef import_persistent(std::string_view name, PersistentHandle handle, Access access, Owner owner = Owner::Dawn);
  TexRef import_persistent_texture(std::string_view name, PersistentHandle handle, const TexDesc& desc, Access access,
                                   Owner owner = Owner::Dawn);
  void skip();
  std::span<std::byte, PassLocalSize> local() noexcept;

private:
  friend struct GraphBuild;
  friend void external_declare(InternalPassBuilder& builder, const PassSetup& setup, const void* state);
  explicit InternalPassBuilder(PassBuilderState& state) noexcept : PassBuilder(state) {}
};

void external_declare(InternalPassBuilder& builder, const PassSetup& setup, const void* state);

struct ColorTargetCost {
  uint32_t bytes = 0;
  uint32_t alignment = 1;
};

struct LatchGraphInputs {
  std::shared_ptr<const ExternalPassSet> externals;
  PassEncodeFn externalEncode = nullptr;
  DepthSource sceneDepthSource = DepthSource::Unavailable;
};

std::shared_ptr<CompiledGraph> compile_graph(const GraphInputs& inputs);
std::shared_ptr<CompiledGraph> compile_graph(const Settings& settings, const Caps& caps, const GraphEnvironment& env,
                                             const LatchGraphInputs& latch = {});
GraphEnvironment gather_graph_environment();
GraphLimits graph_limits(const wgpu::Limits& limits, const wgpu::CompatibilityModeLimits& compat,
                         const GraphFeatures& features) noexcept;

std::span<const ChainNode> chain_nodes() noexcept;
std::span<const ChainNode> stage_chain(Stage stage) noexcept;
std::optional<Stage> chain_node_stage(std::string_view name) noexcept;
std::optional<GxResource> find_gx_resource(std::string_view name) noexcept;
wgpu::TextureFormat gx_fixed_format(GxResource resource) noexcept;
std::optional<ColorTargetCost> color_target_cost(wgpu::TextureFormat format) noexcept;
uint32_t color_bytes_per_sample(std::span<const wgpu::TextureFormat> formats) noexcept;

std::string_view gx_resource_name(GxResource resource) noexcept;
std::string_view access_name(Access access) noexcept;
std::string_view missing_name(Missing missing) noexcept;
std::string_view pass_kind_name(PassKind kind) noexcept;
std::string_view pass_class_name(PassClass cls) noexcept;
std::string_view resource_kind_name(ResourceKind kind) noexcept;
std::string_view cull_reason_name(CullReason reason) noexcept;
std::string_view edge_kind_name(EdgeKind kind) noexcept;

debug::GraphInspection inspect(const CompiledGraph& graph);

struct StageMask {
  std::vector<uint64_t> passes;
  std::vector<uint64_t> dummies;
  std::vector<uint64_t> prevValid;

  [[nodiscard]] bool any() const noexcept {
    for (const auto word : passes) {
      if (word != 0) {
        return true;
      }
    }
    return false;
  }
};

enum class FrameCullReason : uint8_t { Ran, NotEnabled, InputDisabled, NoActiveConsumer };
std::string_view frame_cull_reason_name(FrameCullReason reason) noexcept;

struct FrameCullState {
  uint64_t graphHash = 0;
  bool primed = false;
  uint8_t nextStage = 0;
  uint8_t cutStage = 0;
  std::vector<uint64_t> ran;
  std::vector<uint64_t> prevValid;
};

constexpr bool test_bit(std::span<const uint64_t> bits, size_t index) noexcept {
  return index / 64 < bits.size() && ((bits[index / 64] >> (index % 64)) & 1) != 0;
}

DummyKey dummy_key(const CompiledGraph& graph, const CompiledUse& use, const GraphFeatures& features = {}) noexcept;
void begin_frame_cull(const CompiledGraph& graph, FrameCullState& state) noexcept;
void invalidate_histories(FrameCullState& state) noexcept;
void cull_stage(const CompiledGraph& graph, Stage stage, std::span<const uint64_t> enabled, FrameCullState& state,
                StageMask& out, std::span<FrameCullReason> reasons = {});

namespace testing {
void clear_passes() noexcept;
void reset_logged_diagnostics() noexcept;
} // namespace testing
} // namespace aurora::render::detail

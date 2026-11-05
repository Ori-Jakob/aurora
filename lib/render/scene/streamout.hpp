#pragma once

#include "affine.hpp"
#include "streamout_layout.hpp"

#include "../graph.hpp"
#include "../shaders.hpp"
#include "../../gfx/recording.hpp"
#include "../../gx/gx.hpp"

#include <aurora/gfx.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace aurora::render {
struct Caps;
} // namespace aurora::render

namespace aurora::render::detail {
struct FrameState;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::scene::retained {
struct Frame;
} // namespace aurora::render::scene::retained

namespace aurora::render::scene::streamout {
inline constexpr uint32_t NoSlot = 0xFFFFFFFF;

struct Capture {
  uint32_t slot = NoSlot;
  uint32_t slotGeneration = 0;
  uint32_t identityHash = 0;
  bool newGeometry = false;
  bool latched = false;
  bool teleport = false;
  bool gpuArrays = false;
  bool delivered = false;
  uint16_t entryFlags = 0;
  uint32_t classBits = 0;
  uint32_t categories = 0;
  uint32_t objectFlags = 0;
  uint64_t objectId = 0;
  const gx::AttrConfig* attrs = nullptr;
  uint8_t stride = 0;
  std::span<const uint8_t> records;
  const gx::DrawImmediateData* immediates = nullptr;
  uint8_t prim = 0;
  uint32_t vtxCount = 0;
  gfx::Range idxRange{};
  uint32_t indexCount = 0;
  uint32_t pnSlot = 0;
  const Affine* world = nullptr;
  const Affine* invView = nullptr;
  double eye = 0.0;
  uint32_t clusterFirst = 0;
  uint32_t clusterWord = 0;
  uint32_t material = NoMaterial;
  std::array<uint32_t, 3> attributes{};
};

enum class EventType : uint8_t { Created, Restreamed, Moved, MaterialChanged, UvChanged, ClustersChanged, Freed };

struct PlanEvent {
  EventType type = EventType::Created;
  uint32_t slot = NoSlot;
};

struct VariantDispatch {
  shaders::PipelineId pipeline;
  uint64_t session = 0;
  uint32_t chunkFirst = 0;
  uint32_t chunkCount = 0;
};

struct CheckRange {
  uint32_t slot = NoSlot;
  uint32_t generation = 0;
  uint32_t vertexFirst = 0;
  uint32_t vertexCount = 0;
  uint32_t indexFirst = 0;
  uint32_t indexCount = 0;
  bool hasNormalAttr = false;
  bool rigidStatic = false;
};

enum class ParityFamily : uint8_t { Fresh, Skipped, Latched };

struct ParityEntry {
  uint32_t slot = NoSlot;
  uint32_t generation = 0;
  uint64_t objectId = 0;
  ParityFamily family = ParityFamily::Fresh;
  bool player = false;
  uint32_t seenMask = 0;
  gx::AttrConfig pos{};
  gx::AttrConfig pnmtx{};
  uint8_t stride = 0;
  uint32_t vtxCount = 0;
  std::vector<uint8_t> records;
  std::vector<uint8_t> slice;
  uint32_t sliceFirst = 0;
  std::array<Mat3x4<float>, palette::Matrices> pn{};
  std::array<double, palette::Matrices> slotDistance{};
  std::array<double, 3> origin{};
  double eye = 0.0;
};

struct CheckRequest {
  uint32_t frame = 0;
  uint32_t serial = 0;
  std::vector<CheckRange> ranges;
  std::vector<ParityEntry> parity;
  Affine view{};
  bool wire = false;
  Mat4x4<float> proj{};
};

struct LiveInstance {
  uint32_t slot = NoSlot;
  uint32_t indexCount = 0;
  uint32_t clusterFirst = 0;
  uint32_t clusterCount = 0;
  uint32_t flags = 0;
  uint32_t classBits = 0;
  std::array<double, 3> origin{};
};

struct Plan {
  uint64_t frame = 0;
  uint32_t frameIndex = 0;
  uint32_t serial = 0;
  uint32_t baseSerial = 0;
  uint32_t seenBreaks = 0;
  bool clearInstances = false;
  bool late = false;
  bool headerOnly = false;
  uint32_t blockWord = 0;
  uint32_t writeJobs = 0;
  uint32_t rollJobs = 0;
  uint32_t idxChunks = 0;
  std::vector<VariantDispatch> variants;
  std::vector<PlanEvent> events;
  std::shared_ptr<const CheckRequest> check;
  std::shared_ptr<const std::vector<LiveInstance>> live;

  [[nodiscard]] bool header_only() const noexcept { return headerOnly; }
};

struct Coverage {
  uint32_t frame = 0;
  uint32_t pending = 0;
  uint32_t streamed = 0;
};

struct Stats {
  uint64_t plans = 0;
  uint64_t committed = 0;
  uint64_t uncommitted = 0;
  uint64_t headerOnly = 0;
  uint64_t pieces = 0;
  uint64_t streamedVertices = 0;
  uint64_t moved = 0;
  uint64_t uvChanged = 0;
  uint64_t freed = 0;
  uint64_t clustersChanged = 0;
  uint64_t materialChanged = 0;
  uint64_t lineSkipped = 0;
  uint64_t largeSkipped = 0;
  uint64_t slotSkipped = 0;
  uint64_t paletteStreamed = 0;
  uint64_t paletteSkipped = 0;
  uint64_t paletteLatched = 0;
  uint64_t paletteReorigin = 0;
  uint64_t paletteShared = 0;
  uint64_t paletteOob = 0;
  uint64_t latePlans = 0;
  uint64_t latePieces = 0;
  uint64_t transientSkipped = 0;
  uint64_t variantPending = 0;
  uint64_t variantFailed = 0;
  uint64_t capDeferred = 0;
  uint64_t poolFull = 0;
  uint64_t overrunDeferred = 0;
  uint64_t headroomDeferred = 0;
  uint64_t refused = 0;
  uint64_t lost = 0;
  uint64_t resyncs = 0;
  uint32_t breakCount = 0;
  uint32_t variants = 0;
  uint32_t live = 0;
};

[[nodiscard]] bool supported(const Caps& caps) noexcept;
void initialize();
void shutdown();
void latch_resources(bool active) noexcept;
[[nodiscard]] bool begin_frame(bool active, uint64_t frame) noexcept;
void end_frame() noexcept;
void reset() noexcept;
[[nodiscard]] bool settled() noexcept;

void on_capture(const Capture& capture) noexcept;
void on_freed(uint32_t slot) noexcept;
void on_clusters(uint32_t slot, uint32_t slotGeneration, uint32_t first, uint32_t word) noexcept;
[[nodiscard]] bool streamed(uint32_t slot) noexcept;
void seal(retained::Frame& frame, const detail::StageData& stage) noexcept;
void skip(const detail::StageData& stage) noexcept;
void sealed(const detail::StageData& stage) noexcept;
void drop_pending() noexcept;
void commit_stage_params(const detail::StageData& stage, const void* state, std::span<std::byte> params);

[[nodiscard]] Coverage coverage() noexcept;
[[nodiscard]] std::shared_ptr<const std::vector<LiveInstance>> live() noexcept;
[[nodiscard]] Stats stats() noexcept;
[[nodiscard]] bool applied(uint32_t serial) noexcept;
[[nodiscard]] detail::PersistentHandle pool_handle(uint32_t binding) noexcept;

void worker_frame_begin(uint64_t frameId, const detail::FrameState* frame);

[[nodiscard]] detail::PassDesc pass_desc();
[[nodiscard]] detail::PassDesc late_pass_desc();
[[nodiscard]] std::string variant_source(std::span<const gx::AttrConfig, gx::MaxVtxAttr> attrs, uint8_t stride,
                                         uint8_t uvTex, uint8_t alphaTex);
[[nodiscard]] std::string_view variant_shader_prefix() noexcept;

namespace testing {
using PushFn = gfx::Range (*)(std::span<const uint8_t> bytes) noexcept;

struct SlotState {
  bool streamed = false;
  bool worldSent = false;
  uint32_t vertexFirst = 0;
  uint32_t vertexCount = 0;
  uint32_t indexFirst = 0;
  uint32_t indexCount = 0;
  uint32_t flags = 0;
  uint32_t streams = 0;
  uint32_t recordWrites = 0;
  Affine lastSent{};
  uint32_t clusterFirst = 0;
  uint32_t clusterWord = 0;
  uint32_t paletteWord = RigidMatrix;
  uint32_t usedMask = 0;
  uint32_t seenMask = 0;
  std::array<double, 3> origin{};
};

void reset() noexcept;
void set_push(PushFn push) noexcept;
void set_room(const gfx::detail::StreamRoom* room) noexcept;
void set_variants_ready(bool ready) noexcept;
void set_check(bool on) noexcept;
[[nodiscard]] SlotState slot(uint32_t slot) noexcept;
[[nodiscard]] std::span<const uint32_t> last_block() noexcept;
[[nodiscard]] std::shared_ptr<const Plan> last_plan() noexcept;
[[nodiscard]] uint32_t variant_count() noexcept;
void break_chain() noexcept;
bool encode_plan(const wgpu::CommandEncoder& encoder, std::span<const BufferView, 8> views, const Plan& plan);
bool wait_variants(const Plan& plan);

enum class Chain : uint8_t { Accept, Break, Skip };
struct ChainState {
  uint32_t lastEncoded = 0;
  bool broken = false;
};
[[nodiscard]] Chain chain_step(ChainState& state, uint32_t serial, uint32_t baseSerial, bool clearInstances,
                               bool acknowledged = false) noexcept;
} // namespace testing
} // namespace aurora::render::scene::streamout

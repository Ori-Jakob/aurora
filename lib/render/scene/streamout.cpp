#include "streamout.hpp"

#include "identity.hpp"
#include "map_pick.hpp"
#include "../graph_bind.hpp"
#include "../hints.hpp"
#include "../pass_context.hpp"
#include "../stage_data.hpp"
#include "../tlsf.hpp"
#include "../../gfx/frame.hpp"
#include "../../gfx/hash.hpp"
#include "../../gfx/png_io.hpp"
#include "../../gfx/resources.hpp"
#include "../../gx/command_processor.hpp"
#include "../../gx/vertex_fetch.hpp"
#include "../../internal.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"
#include "../../webgpu/gpu_prof.hpp"

#include <aurora/render_caps.hpp>
#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

namespace aurora::render::scene::streamout {
namespace {
constexpr Module Log{"aurora::render::retained"};
constexpr uint64_t LayoutKey = 0x73747265616D3031;
constexpr Tlsf::BlockId NoBlock = Tlsf::InvalidBlock;
constexpr uint64_t Never = UINT64_MAX;
constexpr uint16_t NoVariant = UINT16_MAX;
constexpr uint8_t NoTex = 0xFF;
constexpr uint32_t ReserveFrames = 8;
constexpr uint64_t CheckEvery = 60;
constexpr uint64_t CheckFirst = 30;
constexpr uint64_t CheckLast = 300;
constexpr uint64_t MaxCheckBytes = 16ull << 20;
constexpr uint32_t MaxCheckLog = 32;
constexpr uint32_t MaxCheckErrors = 8;
constexpr uint32_t WireWidth = 640;
constexpr uint32_t WireHeight = 480;
constexpr uint32_t HeaderWords = frame::LateFrame + 1;
constexpr uint64_t ShutdownWaitMs = 2000;
constexpr uint32_t PoolCount = 5;
constexpr uint32_t PersistentCount = 8;
constexpr std::array<std::string_view, 3> PipelineEntries{"write_jobs", "roll_jobs", "write_indices"};
constexpr std::string_view LibraryShader = "scene/streamout";
constexpr std::string_view PassName = "scene.streamout";
constexpr std::string_view LatePassName = "scene.streamout_late";

struct PassPhases {
  std::string_view after;
  std::string_view before;
};
constexpr PassPhases OpaquePhases{phase::Capture, phase::Geometry};
constexpr PassPhases LatePhases{phase::PreTaa, phase::Taa};
constexpr std::array<std::string_view, PersistentCount> PersistentNames{
    "scene.gx_vertex", "scene.gx_storage", "scene.gx_index", "scene.vertices",
    "scene.indices",   "scene.instances",  "scene.palettes", "scene.frame",
};
constexpr std::array<Access, PersistentCount> PersistentAccess{
    Access::StorageRead,  Access::StorageRead, Access::StorageRead, Access::StorageWrite,
    Access::StorageWrite, Access::StorageRW,   Access::StorageRW,   Access::StorageRW,
};
constexpr uint16_t NoRef = UINT16_MAX;
constexpr uint32_t StaticBits = retained::class_bit::Static | retained::class_bit::Immobile;
constexpr uint32_t NoBank = UINT32_MAX;
constexpr uint32_t NoPalette = UINT32_MAX;
constexpr uint32_t MaxParityVertices = 65536;
constexpr double FreshParityLimit = 1e-3;
constexpr double NearPalette = 2000.0;
constexpr uint32_t ReportedObjects = 8;
constexpr std::array<std::string_view, 3> FamilyNames{"fresh", "skipped", "latched"};
constexpr std::array<std::string_view, 3> BucketNames{"player", "near", "far"};
static_assert(palette::Matrices == gx::MaxPnMtx);

struct SlotState {
  uint32_t generation = 0;
  bool streamed = false;
  bool recordSent = false;
  bool staticPending = false;
  bool lastCaptureOnly = false;
  bool worldSent = false;
  uint32_t vertexFirst = 0;
  uint32_t vertexCount = 0;
  uint32_t indexFirst = 0;
  uint32_t indexCount = 0;
  Tlsf::BlockId vtxBlock = NoBlock;
  Tlsf::BlockId idxBlock = NoBlock;
  uint32_t flags = 0;
  uint32_t material = 0;
  Affine lastSent{};
  uint64_t lastSeen = Never;
  std::array<uint32_t, 12> uv{};
  uint32_t streams = 0;
  uint32_t recordWrites = 0;
  std::array<uint32_t, 2> clusters{};
  std::array<uint32_t, 2> clusterWant{};
  bool clusterQueued = false;
  uint32_t bank = NoBank;
  uint32_t usedMask = 0;
  uint32_t seenMask = 0;
  std::array<double, 3> origin{};
};

struct PaletteCapture {
  std::array<gx::PnMtx, palette::Matrices> pn{};
  Affine invView{};
  uint32_t usedMask = 0;
  uint32_t seenMask = 0;
};

enum class PaletteFamily : uint8_t { None, Fresh, Skipped, Latched };

struct PalettePick {
  uint32_t bank = NoBank;
  bool fresh = false;
  std::array<double, 3> origin{};
  std::array<float, 3> delta{};
  std::array<Affine, palette::Matrices> world{};
  std::array<uint32_t, palette::BankWords> words{};
};

struct PaletteCounts {
  uint64_t streamed = 0;
  uint64_t skipped = 0;
  uint64_t latched = 0;
  uint64_t reorigin = 0;
  uint64_t shared = 0;
  uint64_t oob = 0;
};

struct Variant {
  uint64_t key = 0;
  std::string id;
  std::array<gx::AttrConfig, gx::MaxVtxAttr> attrs{};
  uint8_t stride = 0;
  uint8_t uvTex = NoTex;
  uint8_t alphaTex = NoTex;
  shaders::PipelineId pipeline;
  uint64_t session = 0;
  uint64_t registeredSession = 0;
  bool failedLogged = false;
};

struct Pending {
  uint32_t slot = NoSlot;
  uint32_t generation = 0;
  uint32_t identityHash = 0;
  uint16_t entryFlags = 0;
  uint16_t variant = NoVariant;
  uint32_t classBits = 0;
  uint32_t categories = 0;
  uint32_t objectFlags = 0;
  uint32_t flags = 0;
  uint32_t pieceFlags = 0;
  uint32_t material = 0;
  bool newGeometry = false;
  bool latched = false;
  bool teleport = false;
  bool gpuArrays = false;
  bool hasNormalAttr = false;
  uint8_t prim = 0;
  uint32_t vtxCount = 0;
  uint32_t indexCount = 0;
  uint32_t idxByte = 0;
  gx::DrawImmediateData immediates{};
  Affine world{};
  std::array<uint32_t, record::NormalWords> normal{};
  std::array<uint32_t, 12> uv{};
  std::array<uint32_t, 2> clusters{};
  double eye = 0.0;
  uint64_t objectId = 0;
  uint32_t palette = NoPalette;
  uint32_t parity = NoPalette;
};

struct Commit {
  uint32_t slot = NoSlot;
  SlotState next;
  Tlsf::BlockId freeVtx = NoBlock;
  Tlsf::BlockId freeIdx = NoBlock;
  uint32_t freeBank = NoBank;
  uint32_t history = NoPalette;
  uint64_t objectId = 0;
};

struct Freed {
  uint32_t slot = NoSlot;
  Tlsf::BlockId vtx = NoBlock;
  Tlsf::BlockId idx = NoBlock;
  uint32_t bank = NoBank;
  bool recordSent = false;
};

struct CameraState {
  bool valid = false;
  std::array<float, 12> view{};
  std::array<float, 12> invView{};
  std::array<double, 3> eye{};
  uint32_t cutEpoch = 0;
};

struct FrameUse {
  uint32_t pieces = 0;
  uint32_t streamVertices = 0;
  uint32_t newVertices = 0;
  uint32_t palettes = 0;
};

struct Tentative {
  std::shared_ptr<Plan> plan;
  std::vector<Tlsf::BlockId> vtxAllocs;
  std::vector<Tlsf::BlockId> idxAllocs;
  std::vector<uint32_t> bankAllocs;
  std::vector<PalettePick> palettes;
  PaletteCounts paletteCounts;
  std::vector<Commit> commits;
  std::vector<Freed> frees;
  uint32_t captureOnly = 0;
  uint32_t lostAtCommit = 0;
  uint64_t sealStorage = 0;
  uint32_t blockBytes = 0;
  bool committed = false;
  uint32_t pieces = 0;
  CameraState camera;
  uint64_t movedLatched = 0;
  FrameUse use;
  std::vector<std::pair<uint32_t, std::array<uint32_t, 2>>> clusterCommits;
};

struct Pick {
  uint32_t pending = 0;
  bool stream = false;
  bool topology = false;
  bool full = false;
  bool moved = false;
  bool uv = false;
  bool clusters = false;
  bool prevValid = false;
  uint32_t vertexFirst = 0;
  uint32_t indexFirst = 0;
  Tlsf::BlockId vtxBlock = NoBlock;
  Tlsf::BlockId idxBlock = NoBlock;
  bool newAlloc = false;
  uint32_t palette = NoPalette;
  uint32_t paletteWord = RigidMatrix;
  PaletteFamily family = PaletteFamily::None;
};

struct Local {
  std::array<uint16_t, PersistentCount> refs{};
};
static_assert(sizeof(Local) <= detail::PassLocalSize);

bool g_envRead = false;
bool g_active = false;
bool g_allocated = false;
std::vector<SlotState> g_slots;
std::optional<Tlsf> g_vertexPool;
std::optional<Tlsf> g_indexPool;
uint64_t g_vertexPoolBytes = VertexPoolBytes;
std::vector<Pending> g_pending;
std::vector<PaletteCapture> g_palettes;
std::vector<ParityEntry> g_paritySnaps;
uint32_t g_parityVertices = 0;
std::vector<uint32_t> g_freeBanks;
std::vector<std::array<Affine, palette::Matrices>> g_bankWorld;
std::unordered_map<uint64_t, uint64_t> g_paletteObjects;
std::vector<Variant> g_variants;
uint16_t g_lastVariant = NoVariant;
bool g_warnedVariants = false;
std::vector<Freed> g_freed;
std::vector<std::pair<uint32_t, uint32_t>> g_clusterDirty;
Tentative g_tentative;
uint32_t g_serial = 0;
uint32_t g_lastCommitted = 0;
bool g_clearPending = true;
uint32_t g_seenBreaks = 0;
uint32_t g_pendingStatics = 0;
uint32_t g_streamedCount = 0;
uint32_t g_owed = 0;
uint32_t g_refusedLost = 0;
uint32_t g_undelivered = 0;
uint64_t g_hintFrame = 0;

struct SealRoom {
  std::array<uint64_t, ReserveFrames> after{};
  uint32_t ring = 0;
  uint64_t end = 0;
  bool sealed = false;
};
std::array<SealRoom, 2> g_seals{};
std::array<uint64_t, ReserveFrames> g_blocks{};
uint32_t g_blockRing = 0;
uint64_t g_frameBlocks = 0;
CameraState g_prevCamera;
CameraState g_framePrevCamera;
bool g_frameReset = false;
FrameUse g_frameUse;
bool g_liveChanged = true;
Stats g_stats;
std::vector<uint32_t> g_block;
std::shared_ptr<const Plan> g_lastPlan;
bool g_checkMode = false;
uint64_t g_checkFirst = CheckFirst;
uint64_t g_checkLast = CheckLast;
std::string g_wirePath;
testing::PushFn g_push = nullptr;
const gfx::detail::StreamRoom* g_room = nullptr;
bool g_variantsReady = false;
bool g_checkOverride = false;

std::atomic_uint64_t g_coverage{0};
std::atomic_uint32_t g_breakCount{0};
std::atomic_uint32_t g_appliedSerial{0};
std::mutex g_publishMutex;
Stats g_published;
std::shared_ptr<const std::vector<LiveInstance>> g_live;

wgpu::Device g_device;
wgpu::BindGroupLayout g_groupLayout;
wgpu::PipelineLayout g_pipelineLayout;
detail::PassId g_pass = detail::InvalidPassId;
detail::PassId g_latePass = detail::InvalidPassId;
std::array<detail::PersistentHandle, PersistentCount> g_handles{};
std::array<wgpu::Buffer, PoolCount> g_pools;
bool g_poolsLive = false;

struct WorkerState {
  testing::ChainState chain;
  std::array<wgpu::Buffer, PersistentCount> buffers;
  std::array<uint64_t, PersistentCount> sizes{};
  wgpu::BindGroup group;
  uint64_t frame = 0;
  uint64_t lastUsed = 0;
  bool warnedBreak = false;
};
WorkerState g_worker;

struct CheckLayout {
  uint64_t frameOffset = 0;
  std::vector<uint64_t> vertexOffsets;
  std::vector<uint64_t> indexOffsets;
  size_t ranges = 0;
  uint64_t size = 0;
};

struct CheckResults {
  uint64_t readbacks = 0;
  uint64_t instances = 0;
  uint64_t vertices = 0;
  uint64_t indices = 0;
  uint64_t failures = 0;
  uint64_t skipped = 0;
  bool busy = false;
  wgpu::Buffer buffer;
  uint64_t bufferSize = 0;
  std::unordered_map<uint32_t, std::array<uint32_t, 3>> lastStream;
};
std::mutex g_checkMutex;
CheckResults g_check;

struct ParityBucket {
  std::vector<float> errors;
  double max = 0.0;
};

struct ParityResults {
  std::array<std::array<ParityBucket, BucketNames.size()>, FamilyNames.size()> buckets;
  uint64_t entries = 0;
  uint64_t oob = 0;
  uint64_t missing = 0;
  uint64_t failures = 0;
  double skippedRatio = 0.0;
};
ParityResults g_parity;

constexpr uint32_t ceil_div(uint32_t value, uint32_t divisor) noexcept { return (value + divisor - 1) / divisor; }

uint32_t f32_bits(float value) noexcept { return std::bit_cast<uint32_t>(value); }

float bits_f32(uint32_t value) noexcept { return std::bit_cast<float>(value); }

float ordered_to_f32(uint32_t u) noexcept { return bits_f32((u & 0x80000000u) != 0 ? (u & 0x7FFFFFFFu) : ~u); }

gfx::Range default_push(std::span<const uint8_t> bytes) noexcept { return gfx::push_storage_packed(bytes); }

void read_env() noexcept {
  if (std::exchange(g_envRead, true)) {
    return;
  }
  if (const char* value = SDL_getenv("AURORA_RENDER_RETAINED_WIRE")) {
    g_wirePath = value;
    Log.warn("Retained scene: wireframe check image goes to {} by AURORA_RENDER_RETAINED_WIRE", g_wirePath);
  }
}

uint32_t derived_index_count(uint8_t prim, uint32_t vtxCount) noexcept {
  switch (prim) {
  case GX_QUADS:
    return vtxCount / 4 * 6;
  case GX_TRIANGLES:
    return vtxCount / 3 * 3;
  case GX_TRIANGLESTRIP:
  case GX_TRIANGLEFAN:
    return vtxCount >= 3 ? (vtxCount - 2) * 3 : 0;
  default:
    return 0;
  }
}

void reset_banks() {
  g_freeBanks.resize(PalettePairs);
  for (uint32_t i = 0; i < PalettePairs; ++i) {
    g_freeBanks[i] = PalettePairs - 1 - i;
  }
}

void clear_frame_captures() noexcept {
  g_pending.clear();
  g_palettes.clear();
  g_paritySnaps.clear();
  g_parityVertices = 0;
}

void allocate() {
  if (g_allocated) {
    return;
  }
  g_slots.assign(InstanceSlots, SlotState{});
  g_vertexPool.emplace(g_vertexPoolBytes, PoolGranule);
  g_indexPool.emplace(IndexPoolBytes, PoolGranule);
  g_pending.reserve(retained::MaxEntries);
  g_bankWorld.assign(PalettePairs, {});
  reset_banks();
  g_allocated = true;
}

void release_state() noexcept {
  g_slots = {};
  g_vertexPool.reset();
  g_indexPool.reset();
  g_pending = {};
  g_palettes = {};
  g_paritySnaps = {};
  g_parityVertices = 0;
  g_freeBanks = {};
  g_bankWorld = {};
  g_freed = {};
  g_clusterDirty = {};
  g_tentative = {};
  g_allocated = false;
  g_pendingStatics = 0;
  g_streamedCount = 0;
}

void publish_coverage() noexcept {
  const uint64_t packed = (uint64_t{static_cast<uint32_t>(g_hintFrame)} << 32) |
                          (uint64_t{std::min(g_pendingStatics, 0xFFFFu)} << 16) |
                          uint64_t{std::min(g_streamedCount, 0xFFFFu)};
  g_coverage.store(packed, std::memory_order_release);
}

void rebuild_live() {
  if (!std::exchange(g_liveChanged, false)) {
    return;
  }
  auto live = std::make_shared<std::vector<LiveInstance>>();
  for (uint32_t slot = 0; slot < g_slots.size(); ++slot) {
    const SlotState& st = g_slots[slot];
    if (st.streamed) {
      live->push_back({.slot = slot,
                       .indexCount = st.indexCount,
                       .clusterFirst = st.clusters[0],
                       .clusterCount = st.clusters[1] & 0xFFFFFFu,
                       .flags = st.flags});
    }
  }
  std::lock_guard lock{g_publishMutex};
  g_live = std::move(live);
}

void clear_stream_state() noexcept {
  for (SlotState& st : g_slots) {
    const uint32_t generation = st.generation;
    st = {};
    st.generation = generation;
  }
  if (g_allocated) {
    g_vertexPool.emplace(g_vertexPoolBytes, PoolGranule);
    g_indexPool.emplace(IndexPoolBytes, PoolGranule);
    reset_banks();
  }
  g_freed.clear();
  g_clusterDirty.clear();
  g_tentative = {};
  g_pendingStatics = 0;
  g_streamedCount = 0;
  g_clearPending = true;
  g_liveChanged = true;
}

uint64_t variant_key(const gx::AttrConfig* attrs, uint8_t stride, uint8_t uvTex, uint8_t alphaTex) noexcept {
  Hasher hasher;
  hasher.update(GeneratorRevision);
  hasher.update(attrs[GX_VA_PNMTXIDX]);
  hasher.update(attrs[GX_VA_POS]);
  hasher.update(attrs[GX_VA_NRM]);
  hasher.update(attrs[GX_VA_CLR0]);
  hasher.update(uvTex);
  hasher.update(alphaTex);
  if (uvTex != NoTex) {
    hasher.update(attrs[GX_VA_TEX0 + uvTex]);
  }
  if (alphaTex != NoTex) {
    hasher.update(attrs[GX_VA_TEX0 + alphaTex]);
  }
  hasher.update(stride);
  return hasher.digest();
}

uint16_t find_variant(const gx::AttrConfig* attrs, uint8_t stride, uint8_t uvTex, uint8_t alphaTex) {
  const uint64_t key = variant_key(attrs, stride, uvTex, alphaTex);
  if (g_lastVariant != NoVariant && g_variants[g_lastVariant].key == key) {
    return g_lastVariant;
  }
  for (size_t i = 0; i < g_variants.size(); ++i) {
    if (g_variants[i].key == key) {
      g_lastVariant = static_cast<uint16_t>(i);
      return g_lastVariant;
    }
  }
  if (g_variants.size() >= MaxVariants) {
    if (!std::exchange(g_warnedVariants, true)) {
      Log.warn("Retained scene: more than {} stream-out vertex formats; the rest are not streamed", MaxVariants);
    }
    return NoVariant;
  }
  Variant& variant = g_variants.emplace_back();
  variant.key = key;
  variant.id = fmt::format("{}{:016x}", variant_shader_prefix(), key);
  std::copy_n(attrs, gx::MaxVtxAttr, variant.attrs.begin());
  variant.stride = stride;
  variant.uvTex = uvTex;
  variant.alphaTex = alphaTex;
  g_lastVariant = static_cast<uint16_t>(g_variants.size() - 1);
  return g_lastVariant;
}

shaders::PipelineStatus variant_status(Variant& variant) {
  if (g_variantsReady) {
    return shaders::PipelineStatus::Ready;
  }
  if (!g_pipelineLayout) {
    return shaders::PipelineStatus::Pending;
  }
  const uint64_t session = shaders::session();
  if (variant.registeredSession != session) {
    shaders::register_source(variant.id, variant_source(variant.attrs, variant.stride, variant.uvTex, variant.alphaTex),
                             shaders::SourceScope::Isolated);
    variant.registeredSession = session;
  }
  if (!variant.pipeline || variant.session != session) {
    variant.pipeline = shaders::request_compute({
        .label = variant.id,
        .shader = variant.id,
        .entryPoint = "main",
        .layout = g_pipelineLayout,
        .layoutKey = LayoutKey,
    });
    variant.session = session;
  }
  const shaders::PipelineStatus status = shaders::status(variant.pipeline);
  if (status == shaders::PipelineStatus::Failed && !std::exchange(variant.failedLogged, true)) {
    Log.warn("Retained scene: stream-out variant {} failed to build; its draws are not streamed", variant.id);
  }
  return status;
}

Affine identity_affine() noexcept {
  Affine out{};
  for (int r = 0; r < 3; ++r) {
    out.m[r][r] = 1.0;
  }
  return out;
}

const Mat3x4<float>& tex_matrix(uint32_t mtx) noexcept {
  const uint32_t index = mtx / 3;
  return index < gx::MaxPnMtx ? gx::g_gxState.pnMtx[index].pos : gx::g_gxState.texMtxs[index - gx::MaxPnMtx];
}

bool compose_uv(GXTexCoordID coord, const gx::AttrConfig* attrs, std::array<double, 6>& out) noexcept {
  const gx::TcgConfig& tcg = gx::g_gxState.tcgs[coord];
  bool affine = !tcg.normalize && attrs[GX_VA_TEX0MTXIDX + coord].attrType == GX_NONE;
  Affine m = identity_affine();
  if (tcg.mtx != GX_IDENTITY) {
    m = to_affine(tex_matrix(tcg.mtx));
  }
  std::array<std::array<double, 3>, 3> tmp{};
  for (int k = 0; k < 3; ++k) {
    tmp[k] = {m.m[k][0], m.m[k][1], m.m[k][2] + m.m[k][3]};
  }
  if (tcg.type == GX_TG_MTX2x4) {
    tmp[2] = {0.0, 0.0, 1.0};
  }
  Affine post = identity_affine();
  if (tcg.postMtx != GX_PTIDENTITY) {
    const uint32_t index = (tcg.postMtx - GX_PTTEXMTX0) / 3;
    post = to_affine(gx::g_gxState.ptTexMtxs[std::min<uint32_t>(index, gx::MaxPTTexMtx - 1)]);
  }
  std::array<std::array<double, 3>, 3> proj{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k) {
        value += post.m[r][k] * tmp[k][c];
      }
      proj[r][c] = value + (c == 2 ? post.m[r][3] : 0.0);
    }
  }
  double scale = 1.0;
  if (tcg.type == GX_TG_MTX3x4) {
    if (proj[2][0] != 0.0 || proj[2][1] != 0.0 || proj[2][2] == 0.0) {
      affine = false;
    } else {
      scale = 1.0 / proj[2][2];
    }
  }
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r * 3 + c] = proj[r][c] * scale;
    }
  }
  return affine;
}

void store_uv(const std::array<double, 6>& xform, uint32_t* out) noexcept {
  for (size_t i = 0; i < xform.size(); ++i) {
    out[i] = f32_bits(static_cast<float>(xform[i]));
  }
}

double determinant(const Affine& a) noexcept {
  const auto& m = a.m;
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

void split(double value, uint32_t& hi, uint32_t& lo) noexcept {
  const float h = static_cast<float>(value);
  hi = f32_bits(h);
  lo = f32_bits(static_cast<float>(value - static_cast<double>(h)));
}

void put_world(const Affine& world, uint32_t* out) noexcept {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r * 4 + c] = f32_bits(static_cast<float>(world.m[r][c]));
    }
    split(world.m[r][3], out[r * 4 + 3], out[12 + r]);
  }
  out[15] = 0;
}

std::array<float, 12> affine_rows(const Affine& a) noexcept {
  std::array<float, 12> out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out[r * 4 + c] = static_cast<float>(a.m[r][c]);
    }
  }
  return out;
}

bool same_affine(const Affine& a, const Affine& b) noexcept { return std::memcmp(&a, &b, sizeof(Affine)) == 0; }

bool category_static(const Pending& p) noexcept { return (p.classBits & StaticBits) != 0; }

bool capture_only(const Pending& p) noexcept { return (p.entryFlags & retained::entry_flag::CaptureOnly) != 0; }

void undo_allocations(Tentative& t) noexcept {
  for (const Tlsf::BlockId block : t.vtxAllocs) {
    g_vertexPool->free(block);
  }
  for (const Tlsf::BlockId block : t.idxAllocs) {
    g_indexPool->free(block);
  }
  g_freeBanks.insert(g_freeBanks.end(), t.bankAllocs.rbegin(), t.bankAllocs.rend());
  t.vtxAllocs.clear();
  t.idxAllocs.clear();
  t.bankAllocs.clear();
}

void rollback() noexcept {
  if (!g_tentative.plan || g_tentative.committed) {
    g_tentative = {};
    return;
  }
  undo_allocations(g_tentative);
  g_freed.insert(g_freed.begin(), g_tentative.frees.begin(), g_tentative.frees.end());
  ++g_stats.uncommitted;
  g_tentative = {};
}

void report(uint32_t lost) noexcept {
  if (lost != 0) {
    g_stats.lost += lost;
    retained::report_lost(lost);
  }
}

bool check_frame() noexcept {
  if (g_checkOverride) {
    return true;
  }
  return g_checkMode && webgpu::g_backendType != wgpu::BackendType::Null && g_hintFrame >= g_checkFirst &&
         g_hintFrame <= g_checkLast && (g_hintFrame - g_checkFirst) % CheckEvery == 0;
}

uint32_t add_palette(const Capture& c, const SlotState& st) {
  PaletteCapture& pc = g_palettes.emplace_back();
  std::copy_n(gx::g_gxState.pnMtx.begin(), palette::Matrices, pc.pn.begin());
  pc.invView = *c.invView;
  if (st.streamed && st.bank != NoBank && !c.newGeometry && st.vertexCount == c.vtxCount) {
    pc.usedMask = st.usedMask;
    pc.seenMask = st.seenMask;
  } else {
    const uint32_t offset = c.attrs[GX_VA_PNMTXIDX].offset;
    for (uint32_t v = 0; v < c.vtxCount; ++v) {
      const uint32_t slot = c.records[size_t{v} * c.stride + offset] / 3u;
      pc.usedMask |= 1u << std::min(slot, palette::LastSlot);
      pc.seenMask |= slot < palette::MaskSlots ? 1u << slot : palette::MaskOverflow;
    }
  }
  return static_cast<uint32_t>(g_palettes.size() - 1);
}

uint32_t position_bytes(uint8_t compType) noexcept {
  return compType == GX_U8 || compType == GX_S8 ? 1 : compType == GX_U16 || compType == GX_S16 ? 2 : 4;
}

uint32_t add_parity(const Capture& c, const PaletteCapture& pc) {
  if (!check_frame() || g_seals[0].sealed || c.gpuArrays || g_parityVertices + c.vtxCount > MaxParityVertices) {
    return NoPalette;
  }
  ParityEntry snap{
      .slot = c.slot,
      .generation = c.slotGeneration,
      .objectId = c.objectId,
      .player = (c.classBits & retained::class_bit::Player) != 0,
      .seenMask = pc.seenMask,
      .pos = c.attrs[GX_VA_POS],
      .pnmtx = c.attrs[GX_VA_PNMTXIDX],
      .stride = c.stride,
      .vtxCount = c.vtxCount,
  };
  const size_t recordBytes = size_t{c.vtxCount} * c.stride;
  snap.records.assign(c.records.begin(), c.records.begin() + static_cast<std::ptrdiff_t>(recordBytes));
  for (uint32_t i = 0; i < palette::Matrices; ++i) {
    snap.pn[i] = pc.pn[i].pos;
  }
  if (snap.pos.attrType == GX_INDEX8 || snap.pos.attrType == GX_INDEX16) {
    const uint32_t idxBytes = snap.pos.attrType == GX_INDEX8 ? 1 : 2;
    if (snap.pos.offset + idxBytes > c.stride) {
      return NoPalette;
    }
    uint32_t lo = UINT32_MAX;
    uint32_t hi = 0;
    for (uint32_t v = 0; v < c.vtxCount; ++v) {
      const uint8_t* at = c.records.data() + size_t{v} * c.stride + snap.pos.offset;
      const uint32_t index = idxBytes == 1 ? at[0] : (uint32_t{at[0]} << 8) | at[1];
      lo = std::min(lo, index);
      hi = std::max(hi, index);
    }
    const gx::AttrArray& array = gx::g_gxState.arrays[GX_VA_POS];
    const uint64_t end = uint64_t{hi} * snap.pos.stride + uint64_t{position_bytes(snap.pos.compType)} * snap.pos.cnt;
    if (c.vtxCount == 0 || end > array.size) {
      return NoPalette;
    }
    snap.sliceFirst = lo * snap.pos.stride;
    const auto size = static_cast<uint32_t>(end - snap.sliceFirst);
    const std::span<const uint8_t> slice = gx::fifo::array_bytes(array, snap.sliceFirst, size);
    if (slice.size() != size) {
      return NoPalette;
    }
    snap.slice.assign(slice.begin(), slice.end());
  }
  g_parityVertices += c.vtxCount;
  g_paritySnaps.push_back(std::move(snap));
  return static_cast<uint32_t>(g_paritySnaps.size() - 1);
}

class BlockWriter {
public:
  std::vector<uint32_t> words;
  std::vector<uint32_t> data;

  uint32_t add_data(const uint32_t* source, uint32_t count) {
    const auto at = static_cast<uint32_t>(data.size());
    data.insert(data.end(), source, source + count);
    return at;
  }
};

struct BuildInput {
  std::vector<Pick> picks;
  const std::vector<PalettePick>* palettes = nullptr;
  std::vector<Freed> frees;
  std::vector<uint32_t> clusterSlots;
  CameraState camera;
  CameraState prevCamera;
  uint32_t frameIndex = 0;
  uint32_t serial = 0;
  bool clearInstances = false;
  bool reset = false;
  bool late = false;
};

struct BuildOutput {
  std::vector<uint32_t> block;
  std::vector<VariantDispatch> variants;
  std::vector<Commit> commits;
  std::vector<std::pair<uint32_t, std::array<uint32_t, 2>>> clusterCommits;
  std::vector<PlanEvent> events;
  uint32_t writeJobs = 0;
  uint32_t rollJobs = 0;
  uint32_t idxChunks = 0;
  uint32_t pieces = 0;
  uint32_t vertices = 0;
  uint32_t sharedPalettes = 0;
};

void record_words(const Pending& p, const Pick& pick, uint32_t frameIndex, const SlotState& prev, uint32_t flags,
                  std::array<uint32_t, record::CpuWords>& out) {
  out.fill(0);
  out[record::Flags] = flags;
  out[record::Categories] = p.categories;
  out[record::ObjectFlags] = p.objectFlags;
  out[record::IdentityHash] = p.identityHash;
  out[record::VertexFirst] = pick.vertexFirst;
  out[record::VertexCount] = p.vtxCount;
  out[record::IndexFirst] = pick.indexFirst;
  out[record::IndexCount] = p.indexCount;
  out[record::PaletteFirst] = pick.paletteWord;
  out[record::Material] = p.material;
  out[record::WorldFrame] = frameIndex;
  out[record::Group] = NoGroup;
  put_world(p.world, &out[record::World]);
  put_world(pick.prevValid ? prev.lastSent : p.world, &out[record::PrevWorld]);
  std::copy(p.normal.begin(), p.normal.end(), out.begin() + record::Normal);
  std::copy(p.uv.begin(), p.uv.end(), out.begin() + record::UvBase);
  out[record::UvFrame] = frameIndex;
  out[record::MaterialFrame] = frameIndex;
  out[record::ClusterFirst] = p.clusters[0];
  out[record::ClusterCount] = p.clusters[1];
  out[record::BoundsOffset] = NoBounds;
}

void header_words(const BuildInput& in, std::array<uint32_t, HeaderWords>& out) {
  out.fill(0);
  out[frame::FrameIndex] = in.frameIndex;
  out[frame::Flags] = (in.camera.valid ? frame::CameraValid : 0) | (in.reset ? frame::Reset : 0);
  out[frame::LateFrame] = in.late ? in.frameIndex : 0;
  out[frame::Serial] = in.serial;
  for (int i = 0; i < 3; ++i) {
    split(in.camera.eye[i], out[frame::CamHi + i], out[frame::CamLo + i]);
    split(in.prevCamera.eye[i], out[frame::PrevCamHi + i], out[frame::PrevCamLo + i]);
  }
  for (int i = 0; i < 12; ++i) {
    out[frame::View + i] = f32_bits(in.camera.view[i]);
    out[frame::InvView + i] = f32_bits(in.camera.invView[i]);
    out[frame::PrevView + i] = f32_bits(in.prevCamera.view[i]);
  }
}

BuildOutput build_block(const BuildInput& in) {
  BuildOutput out;
  BlockWriter writer;
  struct Job {
    uint32_t target, dst, src, count;
  };
  std::vector<Job> jobs;
  std::array<uint32_t, HeaderWords> header{};
  header_words(in, header);
  const uint32_t headerWords = in.late ? HeaderWords : frame::LateFrame;
  jobs.push_back({block::TargetFrame, 0, writer.add_data(header.data(), headerWords), headerWords});
  const uint32_t zero = 0;
  const uint32_t zeroWord = writer.add_data(&zero, 1);

  std::vector<const Pick*> streamed;
  for (const Pick& pick : in.picks) {
    if (pick.stream) {
      streamed.push_back(&pick);
    }
  }
  std::ranges::stable_sort(streamed, {}, [](const Pick* pick) { return g_pending[pick->pending].variant; });

  for (const Freed& freed : in.frees) {
    const bool rewritten = std::ranges::any_of(
        in.picks, [&](const Pick& pick) { return pick.full && g_pending[pick.pending].slot == freed.slot; });
    if (freed.recordSent && !rewritten) {
      jobs.push_back({block::TargetInstances, freed.slot * record::Words + record::Flags, zeroWord, 1});
    }
    out.events.push_back({EventType::Freed, freed.slot});
  }

  for (const Pick& pick : in.picks) {
    const Pending& p = g_pending[pick.pending];
    const SlotState& prev = g_slots[p.slot];
    Commit& commit = out.commits.emplace_back();
    commit.slot = p.slot;
    SlotState& next = commit.next;
    next = prev;
    next.generation = p.generation;
    next.lastSeen = g_hintFrame;
    next.lastCaptureOnly = capture_only(p);
    const uint32_t flags = p.flags | (pick.prevValid ? instance_flag::PrevWorldValid : 0);
    const uint32_t base = p.slot * record::Words;
    if (pick.full) {
      std::array<uint32_t, record::CpuWords> words{};
      record_words(p, pick, in.frameIndex, prev, flags, words);
      jobs.push_back({block::TargetInstances, base, writer.add_data(words.data(), record::CpuWords), record::CpuWords});
      next.recordSent = true;
      next.worldSent = true;
      next.lastSent = p.world;
      next.flags = p.flags;
      next.material = p.material;
      next.uv = p.uv;
      next.clusters = p.clusters;
      ++next.recordWrites;
      if (!prev.recordSent) {
        out.events.push_back({EventType::Created, p.slot});
      } else if (prev.clusters != p.clusters) {
        out.events.push_back({EventType::ClustersChanged, p.slot});
      }
    } else {
      if (pick.moved) {
        std::array<uint32_t, record::MovedLast - record::MovedFirst + 1> words{};
        words[0] = in.frameIndex;
        words[record::Group - record::MovedFirst] = NoGroup;
        put_world(p.world, &words[record::World - record::MovedFirst]);
        put_world(pick.prevValid ? prev.lastSent : p.world, &words[record::PrevWorld - record::MovedFirst]);
        std::copy(p.normal.begin(), p.normal.end(), words.begin() + (record::Normal - record::MovedFirst));
        jobs.push_back({block::TargetInstances, base + record::Flags, writer.add_data(&flags, 1), 1});
        jobs.push_back({block::TargetInstances, base + record::MovedFirst,
                        writer.add_data(words.data(), static_cast<uint32_t>(words.size())),
                        static_cast<uint32_t>(words.size())});
        next.lastSent = p.world;
        next.worldSent = true;
        ++next.recordWrites;
        out.events.push_back({EventType::Moved, p.slot});
      }
      if (pick.uv) {
        std::array<uint32_t, record::UvLast - record::UvFirst + 1> words{};
        std::copy(p.uv.begin(), p.uv.end(), words.begin());
        words[record::UvFrame - record::UvFirst] = in.frameIndex;
        jobs.push_back({block::TargetInstances, base + record::UvFirst,
                        writer.add_data(words.data(), static_cast<uint32_t>(words.size())),
                        static_cast<uint32_t>(words.size())});
        next.uv = p.uv;
        ++next.recordWrites;
        out.events.push_back({EventType::UvChanged, p.slot});
      }
      if (pick.clusters) {
        const std::array<uint32_t, 3> words{p.clusters[0], p.clusters[1], NoBounds};
        jobs.push_back({block::TargetInstances, base + record::ClusterFirst, writer.add_data(words.data(), 3), 3});
        next.clusters = p.clusters;
        out.events.push_back({EventType::ClustersChanged, p.slot});
      }
    }
    if (p.palette != NoPalette) {
      if (pick.stream) {
        const PalettePick& pp = (*in.palettes)[pick.palette];
        const PaletteCapture& pc = g_palettes[p.palette];
        next.bank = pp.bank;
        next.origin = pp.origin;
        next.usedMask = pc.usedMask;
        next.seenMask = pc.seenMask;
        commit.history = pick.palette;
        commit.objectId = p.objectId;
      }
    } else if (prev.bank != NoBank) {
      commit.freeBank = prev.bank;
      next.bank = NoBank;
    }
    if (pick.stream) {
      if (prev.streamed) {
        out.events.push_back({EventType::Restreamed, p.slot});
      }
      next.streamed = true;
      next.vertexFirst = pick.vertexFirst;
      next.vertexCount = p.vtxCount;
      next.indexFirst = pick.indexFirst;
      next.indexCount = p.indexCount;
      next.vtxBlock = pick.vtxBlock;
      next.idxBlock = pick.idxBlock;
      next.staticPending = false;
      ++next.streams;
      if (pick.newAlloc) {
        commit.freeVtx = prev.vtxBlock;
        commit.freeIdx = prev.idxBlock;
      }
    }
  }

  for (const uint32_t slot : in.clusterSlots) {
    const SlotState& st = g_slots[slot];
    const std::array<uint32_t, 3> words{st.clusterWant[0], st.clusterWant[1], NoBounds};
    jobs.push_back(
        {block::TargetInstances, slot * record::Words + record::ClusterFirst, writer.add_data(words.data(), 3), 3});
    out.clusterCommits.emplace_back(slot, st.clusterWant);
    out.events.push_back({EventType::ClustersChanged, slot});
  }

  std::vector<uint32_t> pieces;
  std::vector<uint32_t> vtxChunks;
  std::vector<uint32_t> idxChunks;
  std::vector<uint32_t> rolls;
  std::vector<uint32_t> matrices;
  matrices.reserve((in.palettes != nullptr ? in.palettes->size() : 0) * palette::BankWords);
  uint32_t lastMatrices = RigidMatrix;
  for (size_t i = 0; i < streamed.size(); ++i) {
    const Pick& pick = *streamed[i];
    const Pending& p = g_pending[pick.pending];
    const auto pieceIndex = static_cast<uint32_t>(i);
    const PalettePick* pp = pick.palette != NoPalette ? &(*in.palettes)[pick.palette] : nullptr;
    uint32_t matrixWord = RigidMatrix;
    if (pp != nullptr) {
      if (lastMatrices != RigidMatrix &&
          std::equal(pp->words.begin(), pp->words.end(), matrices.begin() + lastMatrices)) {
        matrixWord = lastMatrices;
        ++out.sharedPalettes;
      } else {
        matrixWord = static_cast<uint32_t>(matrices.size());
        matrices.insert(matrices.end(), pp->words.begin(), pp->words.end());
        lastMatrices = matrixWord;
      }
    }
    std::array<uint32_t, block::PieceWords> piece{};
    std::memcpy(piece.data(), &p.immediates, sizeof(p.immediates));
    piece[block::PieceInstance] = p.slot;
    piece[block::PieceVtxCount] = p.vtxCount;
    piece[block::PieceOutVertexFirst] = pick.vertexFirst;
    piece[block::PieceOutIndexFirst] = pick.indexFirst;
    piece[block::PieceIndexCount] = p.indexCount;
    piece[block::PiecePrim] = p.prim;
    piece[block::PieceIdxByte] = p.idxByte;
    piece[block::PieceMatrixWord] = matrixWord;
    piece[block::PieceFlags] = p.pieceFlags;
    pieces.insert(pieces.end(), piece.begin(), piece.end());
    const uint32_t chunkStart = static_cast<uint32_t>(vtxChunks.size() / block::ChunkWords);
    for (uint32_t first = 0; first < p.vtxCount; first += WorkgroupSize) {
      vtxChunks.push_back(pieceIndex);
      vtxChunks.push_back(first);
    }
    const uint32_t chunkCount = static_cast<uint32_t>(vtxChunks.size() / block::ChunkWords) - chunkStart;
    const Variant& variant = g_variants[p.variant];
    if (!out.variants.empty() && i > 0 && g_pending[streamed[i - 1]->pending].variant == p.variant) {
      out.variants.back().chunkCount += chunkCount;
    } else {
      out.variants.push_back({.pipeline = variant.pipeline,
                              .session = variant.session,
                              .chunkFirst = chunkStart,
                              .chunkCount = chunkCount});
    }
    if (pick.topology) {
      const uint32_t triangles = p.indexCount / 3;
      for (uint32_t first = 0; first < triangles; first += WorkgroupSize) {
        idxChunks.push_back(pieceIndex);
        idxChunks.push_back(first);
      }
    }
    std::array<uint32_t, block::RollJobWords> roll{p.slot, pick.topology ? block::RollFresh : 0u, pieceIndex,
                                                   RigidMatrix};
    if (pp != nullptr) {
      roll[block::RollFlags] = block::RollPalette | (pp->fresh ? block::RollFresh : 0u);
      roll[block::RollPaletteWord] = pick.paletteWord;
      for (uint32_t k = 0; k < 3; ++k) {
        roll[block::RollOriginDelta + k] = f32_bits(pp->delta[k]);
      }
    }
    rolls.insert(rolls.end(), roll.begin(), roll.end());
    out.vertices += p.vtxCount;
  }
  out.pieces = static_cast<uint32_t>(streamed.size());

  std::vector<uint32_t>& words = out.block;
  words.assign(block::HeaderWords, 0);
  const auto place = [&](uint32_t field, const std::vector<uint32_t>& region) {
    words[field] = static_cast<uint32_t>(words.size());
    words.insert(words.end(), region.begin(), region.end());
  };
  place(block::PiecesWord, pieces);
  place(block::VtxChunksWord, vtxChunks);
  place(block::IdxChunksWord, idxChunks);
  place(block::RollJobsWord, rolls);
  place(block::MatricesWord, matrices);
  const auto dataWord = static_cast<uint32_t>(words.size() + jobs.size() * block::WriteJobWords);
  words[block::WriteJobsWord] = static_cast<uint32_t>(words.size());
  for (const Job& job : jobs) {
    words.insert(words.end(), {job.target, job.dst, dataWord + job.src, job.count});
  }
  words.insert(words.end(), writer.data.begin(), writer.data.end());
  words[block::Frame] = in.frameIndex;
  words[block::WriteJobCount] = static_cast<uint32_t>(jobs.size());
  words[block::PieceCount] = out.pieces;
  words[block::VtxChunkCount] = static_cast<uint32_t>(vtxChunks.size() / block::ChunkWords);
  words[block::IdxChunkCount] = static_cast<uint32_t>(idxChunks.size() / block::ChunkWords);
  words[block::RollJobCount] = static_cast<uint32_t>(rolls.size() / block::RollJobWords);
  out.writeJobs = static_cast<uint32_t>(jobs.size());
  out.rollJobs = words[block::RollJobCount];
  out.idxChunks = words[block::IdxChunkCount];
  return out;
}

CameraState current_camera() noexcept {
  const detail::SceneHints& hints = detail::live_scene_hints();
  CameraState camera;
  camera.valid = hints.has(detail::SceneHasMainCamera);
  if (!camera.valid) {
    return camera;
  }
  const Affine view = to_affine(hints.mainCamera.view);
  const Affine inv = inverse(view);
  camera.view = affine_rows(view);
  camera.invView = affine_rows(inv);
  camera.eye = {inv.m[0][3], inv.m[1][3], inv.m[2][3]};
  camera.cutEpoch = hints.mainCamera.cutEpoch;
  return camera;
}

std::shared_ptr<const CheckRequest> make_check(uint32_t frameIndex, const std::vector<Commit>& commits,
                                               std::vector<ParityEntry> parity) {
  auto request = std::make_shared<CheckRequest>();
  request->frame = frameIndex;
  request->wire = !g_wirePath.empty();
  request->proj = detail::live_scene_hints().mainCamera.proj;
  request->view = to_affine(detail::live_scene_hints().mainCamera.view);
  request->parity = std::move(parity);
  std::unordered_map<uint32_t, const SlotState*> next;
  for (const Commit& commit : commits) {
    next[commit.slot] = &commit.next;
  }
  for (uint32_t slot = 0; slot < g_slots.size(); ++slot) {
    const auto it = next.find(slot);
    const SlotState* st = it != next.end() ? it->second : &g_slots[slot];
    if (!st->streamed) {
      continue;
    }
    request->ranges.push_back({.slot = slot,
                               .generation = st->generation,
                               .vertexFirst = st->vertexFirst,
                               .vertexCount = st->vertexCount,
                               .indexFirst = st->indexFirst,
                               .indexCount = st->indexCount,
                               .hasNormalAttr = (st->flags & instance_flag::NoNormal) == 0,
                               .rigidStatic = (st->flags & (instance_flag::CpuDeformed | instance_flag::GpuArrays |
                                                            instance_flag::Palette)) == 0});
  }
  return request;
}

std::vector<ParityEntry> parity_entries(const std::vector<Pick>& picks) {
  std::vector<ParityEntry> out;
  for (const Pick& pick : picks) {
    const Pending& p = g_pending[pick.pending];
    if (p.parity == NoPalette || pick.family == PaletteFamily::None) {
      continue;
    }
    ParityEntry& entry = out.emplace_back(std::move(g_paritySnaps[p.parity]));
    entry.family = pick.family == PaletteFamily::Fresh     ? ParityFamily::Fresh
                   : pick.family == PaletteFamily::Skipped ? ParityFamily::Skipped
                                                           : ParityFamily::Latched;
  }
  return out;
}

bool used_slot(uint32_t mask, uint32_t slot) noexcept { return ((mask >> slot) & 1) != 0; }

std::array<double, 3> snap_origin(const Affine& world) noexcept {
  std::array<double, 3> out{};
  for (int r = 0; r < 3; ++r) {
    out[r] = std::round(world.m[r][3] / palette::OriginGrid) * palette::OriginGrid;
  }
  return out;
}

void palette_worlds(const PaletteCapture& pc, std::array<Affine, palette::Matrices>& out) noexcept {
  for (uint32_t i = 0; i < palette::Matrices; ++i) {
    if (used_slot(pc.usedMask, i)) {
      out[i] = multiply(pc.invView, to_affine(pc.pn[i].pos));
    }
  }
}

bool palette_moved(const PaletteCapture& pc, const SlotState& st, const std::array<Affine, palette::Matrices>& world,
                   double eye) noexcept {
  if (pc.usedMask != st.usedMask) {
    return true;
  }
  const auto& last = g_bankWorld[st.bank];
  for (uint32_t i = 0; i < palette::Matrices; ++i) {
    if (used_slot(pc.usedMask, i) && retained::world_moved(world[i], last[i], eye)) {
      return true;
    }
  }
  return false;
}

bool needs_reorigin(const std::array<Affine, palette::Matrices>& world, uint32_t mask,
                    const std::array<double, 3>& origin) noexcept {
  for (uint32_t i = 0; i < palette::Matrices; ++i) {
    for (int r = 0; r < 3 && used_slot(mask, i); ++r) {
      if (std::abs(world[i].m[r][3] - origin[r]) > palette::ReoriginDistance) {
        return true;
      }
    }
  }
  return false;
}

void bank_words(const PaletteCapture& pc, const std::array<Affine, palette::Matrices>& world,
                const std::array<double, 3>& origin, std::array<uint32_t, palette::BankWords>& out) noexcept {
  out.fill(0);
  const auto& inv = pc.invView.m;
  for (uint32_t i = 0; i < palette::Matrices; ++i) {
    if (!used_slot(pc.usedMask, i)) {
      continue;
    }
    uint32_t* pos = &out[i * palette::MatrixWords];
    uint32_t* nrm = &out[palette::NormalWord + i * palette::MatrixWords];
    const Affine n = to_affine(pc.pn[i].nrm);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        pos[r * 4 + c] = f32_bits(static_cast<float>(world[i].m[r][c]));
        nrm[r * 4 + c] =
            f32_bits(static_cast<float>(inv[r][0] * n.m[0][c] + inv[r][1] * n.m[1][c] + inv[r][2] * n.m[2][c]));
      }
      pos[r * 4 + 3] = f32_bits(static_cast<float>(world[i].m[r][3] - origin[r]));
    }
  }
}

Affine origin_world(const std::array<double, 3>& origin) noexcept {
  Affine out = identity_affine();
  for (int r = 0; r < 3; ++r) {
    out.m[r][3] = origin[r];
  }
  return out;
}

bool place_piece(const Pending& p, const SlotState& st, bool countsChanged, Pick& pick, Tentative& t) {
  if (st.streamed && !countsChanged && st.vtxBlock != NoBlock) {
    pick.vertexFirst = st.vertexFirst;
    pick.indexFirst = st.indexFirst;
    pick.vtxBlock = st.vtxBlock;
    pick.idxBlock = st.idxBlock;
    return true;
  }
  const auto vtx = g_vertexPool->allocate(uint64_t{p.vtxCount} * vertex::Bytes);
  if (!vtx) {
    return false;
  }
  const auto idx = g_indexPool->allocate(uint64_t{p.indexCount} * sizeof(uint32_t));
  if (!idx) {
    g_vertexPool->free(vtx->block);
    return false;
  }
  t.vtxAllocs.push_back(vtx->block);
  t.idxAllocs.push_back(idx->block);
  pick.vertexFirst = static_cast<uint32_t>(vtx->offset / vertex::Bytes);
  pick.indexFirst = static_cast<uint32_t>(idx->offset / sizeof(uint32_t));
  pick.vtxBlock = vtx->block;
  pick.idxBlock = idx->block;
  pick.newAlloc = true;
  return true;
}

void plan_palette(uint32_t index, const SlotState& st, const std::array<Affine, palette::Matrices>& world,
                  uint32_t bank, bool newBank, bool wanted, bool fresh, Pick& pick, Tentative& t) {
  Pending& p = g_pending[index];
  const PaletteCapture& pc = g_palettes[p.palette];
  std::array<double, 3> origin = st.origin;
  if (pick.stream) {
    PalettePick& pp = t.palettes.emplace_back();
    pp.bank = bank;
    pp.world = world;
    const Affine& first = world[pc.usedMask != 0 ? std::countr_zero(pc.usedMask) : 0];
    if (newBank || !st.streamed) {
      origin = snap_origin(first);
    } else if (needs_reorigin(world, pc.usedMask, origin)) {
      const std::array<double, 3> moved = snap_origin(first);
      if (moved != origin) {
        for (int k = 0; k < 3; ++k) {
          pp.delta[k] = static_cast<float>(origin[k] - moved[k]);
        }
        origin = moved;
        ++t.paletteCounts.reorigin;
      }
    }
    pp.origin = origin;
    pp.fresh = fresh || newBank;
    bank_words(pc, world, origin, pp.words);
    pick.palette = static_cast<uint32_t>(t.palettes.size() - 1);
    pick.family = PaletteFamily::Fresh;
    ++t.paletteCounts.streamed;
    t.paletteCounts.oob += (pc.seenMask >> palette::Matrices) != 0 ? 1 : 0;
  } else if (!wanted) {
    const bool immobile = (p.classBits & retained::class_bit::Immobile) != 0;
    pick.family = immobile ? PaletteFamily::Latched : PaletteFamily::Skipped;
    ++(immobile ? t.paletteCounts.latched : t.paletteCounts.skipped);
  }
  pick.paletteWord = (pick.stream ? bank : st.bank) * palette::PairWords;
  p.world = origin_world(origin);
  if (p.parity != NoPalette) {
    ParityEntry& snap = g_paritySnaps[p.parity];
    snap.origin = origin;
    snap.eye = p.eye;
    for (uint32_t i = 0; i < palette::Matrices; ++i) {
      const auto& m = world[i].m;
      snap.slotDistance[i] =
          used_slot(pc.usedMask, i) ? std::sqrt(m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]) : 0.0;
    }
  }
}

void plan_pieces(Tentative& t, bool overrun, bool cut, bool epochChanged, std::vector<Pick>& picks) {
  uint32_t streamVertices = g_frameUse.streamVertices;
  uint32_t newVertices = g_frameUse.newVertices;
  uint32_t pieces = g_frameUse.pieces;
  uint32_t palettes = g_frameUse.palettes;
  for (int pass = 0; pass < 3; ++pass) {
    for (uint32_t i = 0; i < g_pending.size(); ++i) {
      const Pending& p = g_pending[i];
      const bool only = capture_only(p);
      const int category = only ? 0 : category_static(p) ? 2 : 1;
      if (category != pass) {
        continue;
      }
      const SlotState& st = g_slots[p.slot];
      const bool deformed = (p.classBits & retained::class_bit::CpuDeformed) != 0;
      const bool countsChanged = st.streamed && (st.vertexCount != p.vtxCount || st.indexCount != p.indexCount);
      const bool isPalette = p.palette != NoPalette;
      std::array<Affine, palette::Matrices> world{};
      bool paletteMoved = false;
      if (isPalette) {
        const PaletteCapture& pc = g_palettes[p.palette];
        palette_worlds(pc, world);
        const bool immobile = (p.classBits & retained::class_bit::Immobile) != 0;
        paletteMoved = st.bank == NoBank || (!immobile && palette_moved(pc, st, world, p.eye));
      }
      bool stream = !st.streamed || p.newGeometry || deformed || p.gpuArrays || countsChanged || paletteMoved;
      const bool wanted = stream;
      const bool topology = !st.streamed || p.newGeometry || countsChanged;
      t.captureOnly += only ? 1 : 0;
      const auto defer = [&](uint64_t& counter) {
        ++counter;
        stream = false;
        t.lostAtCommit += only ? 1 : 0;
        t.captureOnly -= only ? 1 : 0;
      };
      if (stream && p.variant == NoVariant) {
        defer(g_stats.variantFailed);
      }
      if (stream) {
        const shaders::PipelineStatus status = variant_status(g_variants[p.variant]);
        if (status == shaders::PipelineStatus::Pending) {
          defer(g_stats.variantPending);
        } else if (status == shaders::PipelineStatus::Failed) {
          defer(g_stats.variantFailed);
        }
      }
      if (overrun) {
        if (stream) {
          defer(g_stats.overrunDeferred);
        }
        continue;
      }
      if (stream && !only) {
        const bool fresh = !st.streamed && category_static(p);
        if (pieces >= MaxPiecesPerFrame || streamVertices + p.vtxCount > MaxStreamVerticesPerFrame ||
            (fresh && newVertices + p.vtxCount > MaxNewVerticesPerFrame) ||
            (isPalette && palettes >= MaxPaletteDrawsPerFrame)) {
          defer(g_stats.capDeferred);
        }
      }
      Pick pick{.pending = i};
      uint32_t bank = st.bank;
      bool newBank = false;
      if (stream && isPalette && bank == NoBank) {
        if (g_freeBanks.empty()) {
          defer(g_stats.poolFull);
        } else {
          bank = g_freeBanks.back();
          g_freeBanks.pop_back();
          t.bankAllocs.push_back(bank);
          newBank = true;
        }
      }
      if (stream && !place_piece(p, st, countsChanged, pick, t)) {
        defer(g_stats.poolFull);
        if (newBank) {
          g_freeBanks.push_back(bank);
          t.bankAllocs.pop_back();
          bank = st.bank;
          newBank = false;
        }
      }
      if (!stream && (!st.streamed || (isPalette && st.bank == NoBank))) {
        continue;
      }
      if (stream) {
        ++pieces;
        streamVertices += only ? 0 : p.vtxCount;
        newVertices += !st.streamed && category_static(p) && !only ? p.vtxCount : 0;
        palettes += isPalette && !only ? 1 : 0;
      } else {
        pick.vertexFirst = st.vertexFirst;
        pick.indexFirst = st.indexFirst;
      }
      pick.stream = stream;
      pick.topology = stream && topology;
      const bool seenLast = st.lastSeen != Never && st.lastSeen + 1 == g_hintFrame;
      if (isPalette) {
        const bool fresh = pick.topology || cut || epochChanged || p.teleport || !seenLast;
        plan_palette(i, st, world, bank, newBank, wanted, fresh, pick, t);
      }
      const bool flagsChanged = st.flags != p.flags || st.material != p.material;
      pick.full = !st.recordSent || pick.newAlloc || flagsChanged;
      const bool moved =
          p.latched ? !same_affine(p.world, st.lastSent) : retained::world_moved(p.world, st.lastSent, p.eye);
      pick.moved = !pick.full && st.worldSent && moved;
      pick.uv = !pick.full && p.uv != st.uv;
      pick.clusters = !pick.full && st.recordSent && p.clusters != st.clusters;
      pick.prevValid = !isPalette && st.worldSent && seenLast && !cut && !epochChanged && !p.teleport;
      t.movedLatched += pick.moved && p.latched ? 1 : 0;
      picks.push_back(pick);
    }
  }
  t.use = {.pieces = pieces, .streamVertices = streamVertices, .newVertices = newVertices, .palettes = palettes};
}

void commit_plan(Tentative& t) {
  for (const Commit& commit : t.commits) {
    if (commit.freeVtx != NoBlock) {
      g_vertexPool->free(commit.freeVtx);
    }
    if (commit.freeIdx != NoBlock) {
      g_indexPool->free(commit.freeIdx);
    }
    if (commit.freeBank != NoBank) {
      g_freeBanks.push_back(commit.freeBank);
    }
    if (commit.history != NoPalette) {
      g_bankWorld[commit.next.bank] = t.palettes[commit.history].world;
      if (g_checkMode) {
        g_paletteObjects[commit.objectId] += commit.next.vertexCount;
      }
    }
    SlotState& st = g_slots[commit.slot];
    if (st.staticPending && !commit.next.staticPending) {
      --g_pendingStatics;
    }
    if (st.streamed != commit.next.streamed || st.indexCount != commit.next.indexCount ||
        st.flags != commit.next.flags || st.clusters != commit.next.clusters) {
      g_liveChanged = true;
    }
    g_streamedCount += commit.next.streamed && !st.streamed ? 1 : 0;
    st = commit.next;
  }
  for (const auto& [slot, words] : t.clusterCommits) {
    g_slots[slot].clusters = words;
    g_liveChanged = true;
  }
  if (!g_clusterDirty.empty()) {
    std::erase_if(g_clusterDirty, [](const std::pair<uint32_t, uint32_t>& item) {
      SlotState& st = g_slots[item.first];
      if (st.generation != item.second) {
        return true;
      }
      const bool done = !st.recordSent || st.clusterWant == st.clusters;
      st.clusterQueued = st.clusterQueued && !done;
      return done;
    });
  }
  for (const Freed& freed : t.frees) {
    if (freed.vtx != NoBlock) {
      g_vertexPool->free(freed.vtx);
    }
    if (freed.idx != NoBlock) {
      g_indexPool->free(freed.idx);
    }
    if (freed.bank != NoBank) {
      g_freeBanks.push_back(freed.bank);
    }
  }
  if (!t.frees.empty()) {
    g_liveChanged = true;
  }
  g_lastCommitted = t.plan->serial;
  if (t.plan->clearInstances) {
    g_clearPending = false;
    g_frameReset = true;
  }
  g_prevCamera = t.camera;
  g_frameUse = t.use;
  ++g_stats.committed;
  g_stats.pieces += t.pieces;
  g_stats.paletteStreamed += t.paletteCounts.streamed;
  g_stats.paletteSkipped += t.paletteCounts.skipped;
  g_stats.paletteLatched += t.paletteCounts.latched;
  g_stats.paletteReorigin += t.paletteCounts.reorigin;
  g_stats.paletteShared += t.paletteCounts.shared;
  g_stats.paletteOob += t.paletteCounts.oob;
  if (t.plan->late) {
    ++g_stats.latePlans;
    g_stats.latePieces += t.pieces;
  }
  for (const PlanEvent& event : t.plan->events) {
    g_stats.moved += event.type == EventType::Moved ? 1 : 0;
    g_stats.uvChanged += event.type == EventType::UvChanged ? 1 : 0;
    g_stats.freed += event.type == EventType::Freed ? 1 : 0;
    g_stats.clustersChanged += event.type == EventType::ClustersChanged ? 1 : 0;
  }
  t.committed = true;
  publish_coverage();
  rebuild_live();
}

void resync() noexcept {
  uint32_t lost = 0;
  for (const SlotState& st : g_slots) {
    lost += st.recordSent && st.lastCaptureOnly ? 1 : 0;
  }
  report(lost);
  clear_stream_state();
  ++g_stats.resyncs;
  Log.warn("Retained scene: the stream-out chain broke; every instance streams again");
}

std::vector<uint32_t> dirty_clusters(const std::vector<Pick>& picks) {
  std::vector<uint32_t> out;
  if (g_clusterDirty.empty()) {
    return out;
  }
  std::vector<uint32_t> picked;
  picked.reserve(picks.size());
  for (const Pick& pick : picks) {
    picked.push_back(g_pending[pick.pending].slot);
  }
  std::ranges::sort(picked);
  for (const auto& [slot, generation] : g_clusterDirty) {
    const SlotState& st = g_slots[slot];
    if (st.generation == generation && st.recordSent && st.clusterWant != st.clusters &&
        !std::ranges::binary_search(picked, slot)) {
      out.push_back(slot);
    }
  }
  return out;
}

bool late_work(const Tentative& t, const std::vector<Pick>& picks, const std::vector<uint32_t>& clusterSlots) noexcept {
  if (!g_freed.empty() || !clusterSlots.empty() || t.captureOnly + t.lostAtCommit != 0 || g_undelivered != 0) {
    return true;
  }
  return std::ranges::any_of(
      picks, [](const Pick& pick) { return pick.stream || pick.full || pick.moved || pick.uv || pick.clusters; });
}

void touch(const std::vector<Pick>& picks) noexcept {
  for (const Pick& pick : picks) {
    const Pending& p = g_pending[pick.pending];
    SlotState& st = g_slots[p.slot];
    st.lastSeen = g_hintFrame;
    st.lastCaptureOnly = capture_only(p);
  }
}

std::array<wgpu::Buffer, PoolCount> make_pools(const wgpu::Device& device) {
  constexpr wgpu::BufferUsage Usage =
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
  const std::array<std::pair<const char*, uint64_t>, PoolCount> specs{{
      {"scene.vertices", g_vertexPoolBytes},
      {"scene.indices", IndexPoolBytes},
      {"scene.instances", InstancePoolBytes},
      {"scene.palettes", PalettePoolBytes},
      {"scene.frame", FramePoolBytes},
  }};
  std::array<wgpu::Buffer, PoolCount> out;
  for (size_t i = 0; i < PoolCount; ++i) {
    const wgpu::BufferDescriptor descriptor{.label = specs[i].first, .usage = Usage, .size = specs[i].second};
    out[i] = device.CreateBuffer(&descriptor);
  }
  return out;
}

bool gx_buffer_source(void* context, detail::PersistentBuffer& out) {
  const auto& resources = gfx::detail::resources();
  const auto which = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(context));
  const wgpu::Buffer& buffer = which == binding::GxVertex    ? resources.vertexBuffer
                               : which == binding::GxStorage ? resources.storageBuffer
                                                             : resources.indexBuffer;
  if (!buffer) {
    return false;
  }
  out = {.buffer = buffer, .offset = 0, .size = buffer.GetSize(), .generation = resources.streamingGeneration + 1};
  return true;
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void* state) {
  if (setup.env == nullptr || !setup.env->retained || setup.caps == nullptr || !supported(*setup.caps) ||
      !g_pipelineLayout || g_handles[0].id == 0) {
    builder.skip();
    return;
  }
  const auto* phases = static_cast<const PassPhases*>(state);
  builder.after(phases->after);
  builder.before(phases->before);
  builder.manual_bindings();
  Local local;
  for (uint32_t i = 0; i < PersistentCount; ++i) {
    local.refs[i] = builder.import_persistent(PersistentNames[i], g_handles[i], PersistentAccess[i]).index;
  }
  const std::span<std::byte, detail::PassLocalSize> out = builder.local();
  std::memcpy(out.data(), &local, sizeof(local));
}

bool enabled(const detail::StageData& stage, const void*) {
  return g_active && stage.retained != nullptr && stage.retained->plan != nullptr;
}

void pass_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  for (size_t i = 0; i < PipelineEntries.size(); ++i) {
    if (!g_pipelineLayout) {
      ids[i] = shaders::NeverReady.value;
      continue;
    }
    ids[i] = shaders::request_compute({
                                          .label = PassName,
                                          .shader = LibraryShader,
                                          .entryPoint = PipelineEntries[i],
                                          .layout = g_pipelineLayout,
                                          .layoutKey = LayoutKey,
                                      })
                 .value;
  }
}

const wgpu::BindGroup& bind_group(const std::array<BufferView, PersistentCount>& views) {
  WorkerState& w = g_worker;
  w.lastUsed = w.frame;
  bool same = static_cast<bool>(w.group);
  for (size_t i = 0; i < PersistentCount && same; ++i) {
    same = w.buffers[i].Get() == views[i].buffer.Get() && w.sizes[i] == views[i].size;
  }
  if (same) {
    return w.group;
  }
  std::array<wgpu::BindGroupEntry, PersistentCount> entries{};
  for (uint32_t i = 0; i < PersistentCount; ++i) {
    entries[i] = {.binding = i, .buffer = views[i].buffer, .offset = views[i].offset, .size = views[i].size};
    w.buffers[i] = views[i].buffer;
    w.sizes[i] = views[i].size;
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = "scene.streamout",
      .layout = g_groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  w.group = webgpu::g_device.CreateBindGroup(&descriptor);
  return w.group;
}

void dispatch(const wgpu::ComputePassEncoder& pass, uint32_t blockWord, uint32_t first, uint32_t count, uint32_t value,
              uint32_t groups) {
  for (uint32_t done = 0; done < groups; done += MaxDispatchGroups) {
    const uint32_t n = std::min(groups - done, MaxDispatchGroups);
    const std::array<uint32_t, 4> immediates{blockWord, first + done, count, value};
    pass.SetImmediates(0, immediates.data(), ImmediateSize);
    pass.DispatchWorkgroups(n);
  }
}

CheckLayout check_layout(const CheckRequest& request) {
  CheckLayout layout;
  layout.frameOffset = InstancePoolBytes;
  uint64_t offset = layout.frameOffset + FramePoolBytes;
  uint64_t ranges = 0;
  for (const CheckRange& range : request.ranges) {
    const uint64_t bytes = uint64_t{range.vertexCount} * vertex::Bytes + uint64_t{range.indexCount} * 4;
    if (ranges + bytes > MaxCheckBytes) {
      break;
    }
    layout.vertexOffsets.push_back(offset);
    offset += uint64_t{range.vertexCount} * vertex::Bytes;
    layout.indexOffsets.push_back(offset);
    offset += uint64_t{range.indexCount} * 4;
    ranges += bytes;
  }
  layout.ranges = layout.vertexOffsets.size();
  layout.size = offset;
  return layout;
}

struct WireImage {
  std::vector<uint8_t> pixels = opaque_black();

  static std::vector<uint8_t> opaque_black() {
    std::vector<uint8_t> out(size_t{WireWidth} * WireHeight * 4, 0);
    for (size_t i = 3; i < out.size(); i += 4) {
      out[i] = 255;
    }
    return out;
  }

  void line(double x0, double y0, double x1, double y1) {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const int steps = static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy))));
    if (steps > 4 * static_cast<int>(WireWidth)) {
      return;
    }
    for (int i = 0; i <= steps; ++i) {
      const double t = steps == 0 ? 0.0 : static_cast<double>(i) / steps;
      const auto x = static_cast<int64_t>(std::floor(x0 + dx * t));
      const auto y = static_cast<int64_t>(std::floor(y0 + dy * t));
      if (x < 0 || y < 0 || x >= WireWidth || y >= WireHeight) {
        continue;
      }
      uint8_t* px = &pixels[(static_cast<size_t>(y) * WireWidth + static_cast<size_t>(x)) * 4];
      px[0] = 255;
      px[1] = 255;
      px[2] = 255;
      px[3] = 255;
    }
  }
};

std::optional<std::array<double, 2>> project(const std::array<double, 3>& view, const Mat4x4<float>& proj) {
  const Vec4<float>* rows[4] = {&proj.m0, &proj.m1, &proj.m2, &proj.m3};
  std::array<double, 4> clip{};
  for (int r = 0; r < 4; ++r) {
    clip[r] = (*rows[r])[0] * view[0] + (*rows[r])[1] * view[1] + (*rows[r])[2] * view[2] + (*rows[r])[3];
  }
  if (clip[3] <= 1e-6) {
    return std::nullopt;
  }
  return std::array<double, 2>{(clip[0] / clip[3] * 0.5 + 0.5) * WireWidth,
                               (0.5 - clip[1] / clip[3] * 0.5) * WireHeight};
}

double length3(const std::array<double, 3>& v) noexcept { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

void verify_parity(const CheckRequest& request, const CheckLayout& layout, const uint8_t* bytes) {
  const auto word = [&](uint64_t byteOffset) {
    uint32_t value;
    std::memcpy(&value, bytes + byteOffset, sizeof(value));
    return value;
  };
  const auto fail = [&](std::string message) {
    if (g_parity.failures++ < MaxCheckErrors) {
      Log.error("Retained: parity frame {}: {}", request.frame, message);
    }
  };
  std::unordered_map<uint32_t, size_t> rangeOf;
  for (size_t i = 0; i < layout.ranges; ++i) {
    rangeOf.emplace(request.ranges[i].slot, i);
  }
  for (const ParityEntry& entry : request.parity) {
    const auto it = rangeOf.find(entry.slot);
    if (it == rangeOf.end() || request.ranges[it->second].generation != entry.generation ||
        request.ranges[it->second].vertexCount != entry.vtxCount) {
      ++g_parity.missing;
      continue;
    }
    const size_t range = it->second;
    const uint64_t rec = uint64_t{entry.slot} * record::Bytes;
    const bool fresh = entry.family == ParityFamily::Fresh;
    if (fresh && word(rec + record::StreamFrame * 4) != request.frame) {
      fail(fmt::format("palette slot {} streamed at frame {}", entry.slot, word(rec + record::StreamFrame * 4)));
      continue;
    }
    if (fresh && word(rec + record::PaletteMask * 4) != entry.seenMask) {
      fail(fmt::format("palette slot {} observed slot mask {:#x}, the scan found {:#x}", entry.slot,
                       word(rec + record::PaletteMask * 4), entry.seenMask));
    }
    ++g_parity.entries;
    gx::ShaderConfig config{};
    config.vtxStride = entry.stride;
    config.attrs[GX_VA_POS] = entry.pos;
    config.attrs[GX_VA_PNMTXIDX] = entry.pnmtx;
    gx::vertex_fetch::CpuVertexSource source{.records = entry.records};
    source.arrays[0] = {.bytes = entry.slice, .first = entry.sliceFirst};
    auto& family = g_parity.buckets[static_cast<size_t>(entry.family)];
    for (uint32_t v = 0; v < entry.vtxCount; ++v) {
      gx::vertex_fetch::CpuVertex raw;
      if (!gx::vertex_fetch::decode_vertex(config, source, v, raw)) {
        fail(fmt::format("palette slot {} vertex {} does not decode", entry.slot, v));
        break;
      }
      if (raw.pnMtx > palette::LastSlot) {
        ++g_parity.oob;
        continue;
      }
      const Affine pn = to_affine(entry.pn[raw.pnMtx]);
      const std::array<double, 3> local{raw.pos[0], raw.pos[1], raw.pos[2]};
      const std::array<double, 3> want = transform(pn, local);
      const uint64_t at = layout.vertexOffsets[range] + uint64_t{v} * vertex::Bytes;
      std::array<double, 3> p{};
      for (int k = 0; k < 3; ++k) {
        p[k] = entry.origin[k] + static_cast<double>(bits_f32(word(at + k * 4)));
      }
      const std::array<double, 3> got = transform(request.view, p);
      const double err = length3({got[0] - want[0], got[1] - want[1], got[2] - want[2]});
      ParityBucket& bucket = family[entry.player ? 0 : length3(want) < NearPalette ? 1 : 2];
      bucket.errors.push_back(static_cast<float>(err));
      bucket.max = std::max(bucket.max, err);
      if (entry.family != ParityFamily::Skipped) {
        continue;
      }
      double tolerance = 1e-4 + 1e-6 * std::max(entry.slotDistance[raw.pnMtx], entry.eye) + FreshParityLimit;
      for (int c = 0; c < 3; ++c) {
        const double column = length3({pn.m[0][c], pn.m[1][c], pn.m[2][c]});
        tolerance += 1e-5 * std::max(1.0, column) * std::abs(local[c]);
      }
      g_parity.skippedRatio = std::max(g_parity.skippedRatio, err / tolerance);
      if (err > tolerance) {
        fail(fmt::format("skipped palette slot {} vertex {} is {:.6f} u off, over its {:.6f} u tolerance", entry.slot,
                         v, err, tolerance));
      }
    }
  }
}

void verify(const CheckRequest& request, const CheckLayout& layout, const uint8_t* bytes) {
  const auto word = [&](uint64_t byteOffset) {
    uint32_t value;
    std::memcpy(&value, bytes + byteOffset, sizeof(value));
    return value;
  };
  uint64_t failures = 0;
  uint64_t vertices = 0;
  uint64_t indices = 0;
  const auto fail = [&](std::string message) {
    if (failures++ < MaxCheckErrors) {
      Log.error("Retained: check frame {}: {}", request.frame, message);
    }
  };
  const uint32_t frameIndex = word(layout.frameOffset + frame::FrameIndex * 4);
  const uint32_t breaks = word(layout.frameOffset + frame::BreakCount * 4);
  const uint32_t lateFrame = word(layout.frameOffset + frame::LateFrame * 4);
  if (frameIndex != request.frame) {
    fail(fmt::format("header frame {} is not {}", frameIndex, request.frame));
  }
  if (breaks != 0) {
    fail(fmt::format("breakCount {}", breaks));
  }
  Affine view{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      view.m[r][c] = bits_f32(word(layout.frameOffset + (frame::View + r * 4 + c) * 4));
    }
  }
  std::optional<WireImage> wire;
  if (request.wire) {
    wire.emplace();
  }
  uint32_t worldCurrent = 0;
  uint32_t streamCurrent = 0;
  std::lock_guard lock{g_checkMutex};
  for (size_t i = 0; i < layout.ranges; ++i) {
    const CheckRange& range = request.ranges[i];
    const uint64_t rec = uint64_t{range.slot} * record::Bytes;
    const auto field = [&](uint32_t index) { return word(rec + index * 4); };
    const uint32_t flags = field(record::Flags);
    if ((flags & instance_flag::Valid) == 0) {
      fail(fmt::format("slot {} is not valid", range.slot));
      continue;
    }
    if (field(record::IndexCount) % 3 != 0 || field(record::IndexCount) != range.indexCount ||
        field(record::VertexCount) != range.vertexCount) {
      fail(fmt::format("slot {} counts {}/{}", range.slot, field(record::VertexCount), field(record::IndexCount)));
    }
    std::array<float, 3> lo{};
    std::array<float, 3> hi{};
    for (int k = 0; k < 3; ++k) {
      lo[k] = ordered_to_f32(field(record::CurAabb + k));
      hi[k] = ordered_to_f32(field(record::CurAabb + 3 + k));
    }
    const bool noNormal = (flags & instance_flag::NoNormal) != 0;
    const bool noColor = (flags & instance_flag::NoColor) != 0;
    std::vector<std::array<double, 3>> positions(range.vertexCount);
    for (uint32_t v = 0; v < range.vertexCount; ++v) {
      const uint64_t at = layout.vertexOffsets[i] + uint64_t{v} * vertex::Bytes;
      std::array<float, 3> p{bits_f32(word(at)), bits_f32(word(at + 4)), bits_f32(word(at + 8))};
      const uint32_t aux = word(at + vertex::Aux * 4);
      if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
        fail(fmt::format("slot {} vertex {} is not finite", range.slot, v));
        break;
      }
      if (p[0] < lo[0] || p[1] < lo[1] || p[2] < lo[2] || p[0] > hi[0] || p[1] > hi[1] || p[2] > hi[2]) {
        fail(fmt::format("slot {} vertex {} is outside its box", range.slot, v));
        break;
      }
      if (((aux & vertex::HasNormal) != 0) == noNormal) {
        fail(fmt::format("slot {} vertex {} normal presence disagrees with NoNormal", range.slot, v));
        break;
      }
      if (((aux & vertex::HasColor) != 0) == noColor) {
        fail(fmt::format("slot {} vertex {} colour presence disagrees with NoColor", range.slot, v));
        break;
      }
      positions[v] = {p[0], p[1], p[2]};
    }
    for (uint32_t k = 0; k < range.indexCount; ++k) {
      if (word(layout.indexOffsets[i] + uint64_t{k} * 4) >= range.vertexCount) {
        fail(fmt::format("slot {} index {} is past its {} vertices", range.slot, k, range.vertexCount));
        break;
      }
    }
    const uint32_t streamFrame = field(record::StreamFrame);
    const uint32_t worldFrame = field(record::WorldFrame);
    const std::array<uint32_t, 3> stamp{range.generation, field(record::VertexFirst), streamFrame};
    if (range.rigidStatic) {
      if (const auto it = g_check.lastStream.find(range.slot); it != g_check.lastStream.end() &&
                                                               it->second[0] == stamp[0] && it->second[1] == stamp[1] &&
                                                               it->second[2] != stamp[2]) {
        fail(fmt::format("rigid slot {} streamed again at frame {}", range.slot, streamFrame));
      }
    }
    g_check.lastStream[range.slot] = stamp;
    worldCurrent += worldFrame == request.frame ? 1 : 0;
    streamCurrent += streamFrame == request.frame ? 1 : 0;
    if (i < MaxCheckLog) {
      Log.info("Retained: check frame {} slot {} hash {:08x} flags {:#x} vertices {} worldFrame {} streamFrame {}",
               request.frame, range.slot, field(record::IdentityHash), flags, range.vertexCount, worldFrame,
               streamFrame);
    }
    vertices += range.vertexCount;
    indices += range.indexCount;
    if (wire) {
      Affine world{};
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          world.m[r][c] = bits_f32(field(record::World + r * 4 + c));
        }
        world.m[r][3] = static_cast<double>(bits_f32(field(record::World + r * 4 + 3))) +
                        static_cast<double>(bits_f32(field(record::World + 12 + r)));
      }
      const Affine toView = multiply(view, world);
      for (uint32_t k = 0; k + 2 < range.indexCount; k += 3) {
        std::array<std::optional<std::array<double, 2>>, 3> screen;
        for (uint32_t e = 0; e < 3; ++e) {
          const uint32_t index = word(layout.indexOffsets[i] + uint64_t{k + e} * 4);
          if (index >= range.vertexCount) {
            break;
          }
          screen[e] = project(transform(toView, positions[index]), request.proj);
        }
        for (uint32_t e = 0; e < 3; ++e) {
          const auto& a = screen[e];
          const auto& b = screen[(e + 1) % 3];
          if (a && b) {
            wire->line((*a)[0], (*a)[1], (*b)[0], (*b)[1]);
          }
        }
      }
    }
  }
  verify_parity(request, layout, bytes);
  ++g_check.readbacks;
  g_check.instances += layout.ranges;
  g_check.vertices += vertices;
  g_check.indices += indices;
  g_check.failures += failures;
  Log.info(
      "Retained: check frame {}: {} instances, {} with this frame's world, {} streamed this frame, breakCount {}, "
      "lateFrame {}",
      request.frame, layout.ranges, worldCurrent, streamCurrent, breaks, lateFrame);
  if (wire && !g_wirePath.empty()) {
    if (!gfx::png::write_rgba8_png(g_wirePath, WireWidth, WireHeight, {wire->pixels.data(), wire->pixels.size()})) {
      Log.error("Retained: the wireframe check image could not be written to {}", g_wirePath);
      ++g_check.failures;
    }
  }
}

void queue_check(const wgpu::CommandEncoder& encoder, const std::array<BufferView, PersistentCount>& views,
                 std::shared_ptr<const CheckRequest> request) {
  CheckLayout layout = check_layout(*request);
  wgpu::Buffer buffer;
  {
    std::lock_guard lock{g_checkMutex};
    if (g_check.busy) {
      ++g_check.skipped;
      return;
    }
    if (!g_check.buffer || g_check.bufferSize < layout.size) {
      const wgpu::BufferDescriptor descriptor{
          .label = "Retained check readback",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = layout.size,
      };
      g_check.buffer = webgpu::g_device.CreateBuffer(&descriptor);
      g_check.bufferSize = layout.size;
    }
    g_check.busy = true;
    buffer = g_check.buffer;
  }
  const BufferView& instances = views[binding::Instances];
  const BufferView& frameView = views[binding::Frame];
  encoder.CopyBufferToBuffer(instances.buffer, instances.offset, buffer, 0, InstancePoolBytes);
  encoder.CopyBufferToBuffer(frameView.buffer, frameView.offset, buffer, layout.frameOffset, FramePoolBytes);
  for (size_t i = 0; i < layout.ranges; ++i) {
    const CheckRange& range = request->ranges[i];
    if (range.vertexCount != 0) {
      encoder.CopyBufferToBuffer(views[binding::Vertices].buffer,
                                 views[binding::Vertices].offset + uint64_t{range.vertexFirst} * vertex::Bytes, buffer,
                                 layout.vertexOffsets[i], uint64_t{range.vertexCount} * vertex::Bytes);
    }
    if (range.indexCount != 0) {
      encoder.CopyBufferToBuffer(views[binding::Indices].buffer,
                                 views[binding::Indices].offset + uint64_t{range.indexFirst} * 4, buffer,
                                 layout.indexOffsets[i], uint64_t{range.indexCount} * 4);
    }
  }
  const uint64_t size = layout.size;
  detail::queue_after_frame([buffer, size, request, layout = std::move(layout)] {
    buffer.MapAsync(wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowSpontaneous,
                    [buffer, size, request, layout](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      if (status == wgpu::MapAsyncStatus::Success) {
                        const auto* bytes = static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, size));
                        if (bytes != nullptr) {
                          verify(*request, layout, bytes);
                        }
                        buffer.Unmap();
                      }
                      std::lock_guard lock{g_checkMutex};
                      g_check.busy = false;
                    });
  });
}

using FixedPipelines = std::array<wgpu::ComputePipeline, PipelineEntries.size()>;

void encode_stream(const wgpu::CommandEncoder& encoder, std::string_view name,
                   const std::array<BufferView, PersistentCount>& views, const FixedPipelines& fixed, const Plan& plan,
                   uint32_t serial, uint32_t baseSerial, bool clearInstances) {
  bool bound = true;
  for (const BufferView& view : views) {
    bound = bound && static_cast<bool>(view.buffer);
  }
  bool resolved = bound && std::ranges::all_of(fixed, [](const auto& pipeline) { return static_cast<bool>(pipeline); });
  std::vector<wgpu::ComputePipeline> variants(plan.variants.size());
  for (size_t i = 0; i < variants.size(); ++i) {
    variants[i] = shaders::compute_pipeline(plan.variants[i].pipeline);
    resolved = resolved && static_cast<bool>(variants[i]);
  }
  const wgpu::BindGroup group = bound ? bind_group(views) : wgpu::BindGroup{};
  resolved = resolved && static_cast<bool>(group);
  WorkerState& w = g_worker;
  testing::Chain chain = testing::Chain::Break;
  if (resolved) {
    chain = testing::chain_step(w.chain, serial, baseSerial, clearInstances);
  } else if (w.chain.broken) {
    chain = testing::Chain::Skip;
  } else {
    w.chain.broken = true;
  }
  const uint32_t breaks = g_breakCount.load(std::memory_order_relaxed) + (chain == testing::Chain::Break ? 1 : 0);
  const wgpu::ComputePassDescriptor descriptor{.label = wgpu::StringView{name.data(), name.size()},
                                               .timestampWrites = webgpu::gpu_prof::pass_writes(name)};
  if (chain != testing::Chain::Accept) {
    if (chain == testing::Chain::Break) {
      g_breakCount.store(breaks, std::memory_order_relaxed);
      if (!std::exchange(w.warnedBreak, true)) {
        Log.warn("Retained scene: stream-out plan {} does not follow plan {}; clearing the instance table", serial,
                 w.chain.lastEncoded);
      }
      if (bound) {
        encoder.ClearBuffer(views[binding::Instances].buffer, views[binding::Instances].offset,
                            views[binding::Instances].size);
      }
    }
    if (group && fixed[0]) {
      const wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&descriptor);
      pass.SetBindGroup(0, group);
      pass.SetPipeline(fixed[0]);
      dispatch(pass, plan.blockWord, 0, 1, breaks | 0x80000000u, 1);
      pass.End();
    }
    return;
  }
  if (clearInstances) {
    encoder.ClearBuffer(views[binding::Instances].buffer, views[binding::Instances].offset,
                        views[binding::Instances].size);
  }
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&descriptor);
  pass.SetBindGroup(0, group);
  pass.SetPipeline(fixed[0]);
  dispatch(pass, plan.blockWord, 0, plan.writeJobs, breaks | (clearInstances ? 0x80000000u : 0u), plan.writeJobs);
  if (plan.rollJobs != 0) {
    pass.SetPipeline(fixed[1]);
    dispatch(pass, plan.blockWord, 0, plan.rollJobs, plan.frameIndex, ceil_div(plan.rollJobs, WorkgroupSize));
  }
  if (plan.idxChunks != 0) {
    pass.SetPipeline(fixed[2]);
    dispatch(pass, plan.blockWord, 0, plan.idxChunks, plan.frameIndex, plan.idxChunks);
  }
  for (size_t i = 0; i < variants.size(); ++i) {
    const VariantDispatch& run = plan.variants[i];
    pass.SetPipeline(variants[i]);
    dispatch(pass, plan.blockWord, run.chunkFirst, run.chunkCount, plan.frameIndex, run.chunkCount);
  }
  pass.End();
  g_appliedSerial.store(serial, std::memory_order_release);
  if (plan.check) {
    queue_check(encoder, views, plan.check);
  }
}

void encode(detail::PassContext& ctx, const void*) {
  std::array<uint32_t, 4> params{};
  const std::span<const std::byte> bytes = ctx.params();
  std::memcpy(params.data(), bytes.data(), std::min(bytes.size(), sizeof(params)));
  const auto& frame = ctx.stage_data().retained;
  const Plan* plan = frame != nullptr ? frame->plan.get() : nullptr;
  if (plan == nullptr || params[0] == 0 || plan->serial != params[0]) {
    return;
  }
  Local local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  std::array<BufferView, PersistentCount> views;
  for (uint32_t i = 0; i < PersistentCount; ++i) {
    views[i] = local.refs[i] != NoRef ? ctx.buffer(BufRef{local.refs[i]}) : BufferView{};
  }
  FixedPipelines fixed;
  for (uint32_t i = 0; i < fixed.size(); ++i) {
    fixed[i] = shaders::compute_pipeline(ctx.pipeline(i));
  }
  encode_stream(ctx.encoder(), ctx.name(), views, fixed, *plan, params[0], params[1], params[2] != 0);
}

void destroy_pools() noexcept {
  for (uint32_t i = 0; i < PoolCount; ++i) {
    if (g_handles[binding::Vertices + i].id != 0) {
      detail::set_persistent_buffer(g_handles[binding::Vertices + i], {}, 0, 0);
    }
    g_pools[i] = {};
  }
  g_poolsLive = false;
}

void log_stats() {
  const Stats& s = g_stats;
  if (s.plans == 0) {
    return;
  }
  Log.info(
      "Retained: stream-out plans {} committed {} uncommitted {} header-only {} pieces {} vertices {} moved {} uv "
      "{} freed {} clusters {} resyncs {} variants {} late plans {} late pieces {}",
      s.plans, s.committed, s.uncommitted, s.headerOnly, s.pieces, s.streamedVertices, s.moved, s.uvChanged, s.freed,
      s.clustersChanged, s.resyncs, g_variants.size(), s.latePlans, s.latePieces);
  Log.info(
      "Retained: stream-out skipped transient {} line {} large {} slot {}; deferred variant "
      "{} failed {} cap {} pool {} overrun {} headroom {} refused {}; lost {}",
      s.transientSkipped, s.lineSkipped, s.largeSkipped, s.slotSkipped, s.variantPending, s.variantFailed,
      s.capDeferred, s.poolFull, s.overrunDeferred, s.headroomDeferred, s.refused, s.lost);
  Log.info("Retained: palettes streamed {} skipped {} latched {} reorigin {} shared {} oob {}", s.paletteStreamed,
           s.paletteSkipped, s.paletteLatched, s.paletteReorigin, s.paletteShared, s.paletteOob);
}

void log_parity() {
  std::lock_guard lock{g_checkMutex};
  if (g_parity.entries == 0 && g_parity.failures == 0 && g_parity.missing == 0) {
    return;
  }
  std::string buckets;
  bool freshOver = false;
  for (size_t f = 0; f < FamilyNames.size(); ++f) {
    for (size_t b = 0; b < BucketNames.size(); ++b) {
      ParityBucket& bucket = g_parity.buckets[f][b];
      if (bucket.errors.empty()) {
        continue;
      }
      const size_t at = (bucket.errors.size() - 1) * 99 / 100;
      std::nth_element(bucket.errors.begin(), bucket.errors.begin() + static_cast<std::ptrdiff_t>(at),
                       bucket.errors.end());
      buckets += fmt::format("; {} {} max {:.3g} p99 {:.3g} over {} vertices", FamilyNames[f], BucketNames[b],
                             bucket.max, bucket.errors[at], bucket.errors.size());
      freshOver = freshOver || (f == static_cast<size_t>(ParityFamily::Fresh) && bucket.max > FreshParityLimit);
    }
  }
  const std::string summary = fmt::format(
      "{} entries, {} out-of-range slot vertices, {} missing, worst skipped error {:.3f} of its tolerance{}",
      g_parity.entries, g_parity.oob, g_parity.missing, g_parity.skippedRatio, buckets);
  if (freshOver || g_parity.failures != 0) {
    Log.error("Retained: parity FAILED ({} failures, {})", g_parity.failures, summary);
  } else {
    Log.info("Retained: parity passed ({})", summary);
  }
  std::vector<std::pair<uint64_t, uint64_t>> objects(g_paletteObjects.begin(), g_paletteObjects.end());
  std::ranges::sort(objects, [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string top;
  for (size_t i = 0; i < std::min<size_t>(objects.size(), ReportedObjects); ++i) {
    top += fmt::format("{}{:#x} {}", i == 0 ? "" : ", ", objects[i].first, objects[i].second);
  }
  Log.info("Retained: palette objects by streamed vertices: {}", top);
}

void log_check_result() {
  std::unique_lock lock{g_checkMutex};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{ShutdownWaitMs};
  while (g_check.busy && std::chrono::steady_clock::now() < deadline) {
    lock.unlock();
    if (webgpu::g_instance) {
      webgpu::g_instance.ProcessEvents();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    lock.lock();
  }
  if (!g_checkMode && g_check.readbacks == 0) {
    return;
  }
  if (g_check.readbacks == 0 && g_check.failures == 0) {
    Log.warn("Retained: check mode ran no readback");
    return;
  }
  const uint64_t moved = g_stats.moved;
  if (g_check.failures != 0 || g_check.busy) {
    Log.error("Retained: check FAILED ({} failures, {} readbacks, {} instances, {} vertices, {} indices{})",
              g_check.failures, g_check.readbacks, g_check.instances, g_check.vertices, g_check.indices,
              g_check.busy ? ", a readback never finished" : "");
    return;
  }
  Log.info(
      "Retained: check passed ({} readbacks, {} instances, {} vertices, {} indices, {} skipped, {} moved, {} "
      "breaks)",
      g_check.readbacks, g_check.instances, g_check.vertices, g_check.indices, g_check.skipped, moved,
      g_breakCount.load(std::memory_order_relaxed));
}
} // namespace

bool supported(const Caps& caps) noexcept {
  return caps.valid && caps.core && caps.limits.maxImmediateSize >= ImmediateSize &&
         stage_binding_limits(caps, ShaderStage::Compute).storageBuffers >= StorageBindings;
}

void initialize() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device || !supported(caps())) {
    return;
  }
  std::array<wgpu::BindGroupLayoutEntry, StorageBindings> entries{};
  for (uint32_t i = 0; i < StorageBindings; ++i) {
    const bool readOnly = i <= binding::GxIndex;
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = readOnly ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage},
    };
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{
      .label = "scene.streamout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "scene.streamout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_groupLayout,
      .immediateSize = ImmediateSize,
  };
  g_pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  g_device = device;
  const uint64_t binding = caps().limits.maxStorageBufferBindingSize;
  g_vertexPoolBytes = std::min(VertexPoolBytes, binding / PoolGranule * PoolGranule);
  if (g_vertexPoolBytes < VertexPoolBytes) {
    Log.info("Retained scene: the vertex pool shrinks to {} MiB, the storage binding limit", g_vertexPoolBytes >> 20);
  }
  for (uint32_t i = 0; i <= binding::GxIndex; ++i) {
    g_handles[i] = detail::create_persistent_buffer_source(PersistentNames[i], &gx_buffer_source,
                                                           reinterpret_cast<void*>(static_cast<uintptr_t>(i)));
  }
  for (uint32_t i = binding::Vertices; i < PersistentCount; ++i) {
    g_handles[i] = detail::create_persistent(PersistentNames[i], detail::PersistentKind::Buffer);
  }
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
  if (g_latePass == detail::InvalidPassId) {
    g_latePass = detail::register_pass(late_pass_desc());
  }
  retained::add_settle_predicate(&settled);
}

void shutdown() {
  log_stats();
  log_check_result();
  log_parity();
  for (detail::PassId* pass : {&g_pass, &g_latePass}) {
    if (*pass != detail::InvalidPassId) {
      detail::unregister_pass(*pass);
      *pass = detail::InvalidPassId;
    }
  }
  destroy_pools();
  for (detail::PersistentHandle& handle : g_handles) {
    if (handle.id != 0) {
      detail::destroy_persistent(handle);
      handle = {};
    }
  }
  g_worker = {};
  {
    std::lock_guard lock{g_checkMutex};
    g_check = {};
    g_parity = {};
  }
  g_paletteObjects.clear();
  g_pipelineLayout = {};
  g_groupLayout = {};
  g_device = {};
  g_variants.clear();
  g_lastVariant = NoVariant;
  release_state();
}

void latch_resources(bool active) noexcept {
  if (active == g_poolsLive) {
    return;
  }
  if (!active) {
    destroy_pools();
    return;
  }
  if (!g_device || g_handles[binding::Vertices].id == 0) {
    return;
  }
  g_pools = make_pools(g_device);
  const std::array<uint64_t, PoolCount> sizes{g_vertexPoolBytes, IndexPoolBytes, InstancePoolBytes, PalettePoolBytes,
                                              FramePoolBytes};
  for (uint32_t i = 0; i < PoolCount; ++i) {
    detail::set_persistent_buffer(g_handles[binding::Vertices + i], g_pools[i], 0, sizes[i]);
  }
  g_poolsLive = true;
}

bool begin_frame(bool gate, uint64_t frame) noexcept {
  read_env();
  const bool active = gate && (static_cast<bool>(g_pipelineLayout) || g_variantsReady);
  if (active != g_active) {
    g_active = active;
    if (active) {
      allocate();
      clear_stream_state();
      g_checkMode = retained::check_window(g_checkFirst, g_checkLast);
      if (g_checkFirst == 0 && g_checkLast == UINT64_MAX) {
        g_checkFirst = CheckFirst;
        g_checkLast = CheckLast;
      }
    } else {
      release_state();
    }
  }
  if (!active) {
    return false;
  }
  g_hintFrame = frame;
  g_framePrevCamera = g_prevCamera;
  g_frameReset = false;
  g_frameUse = {};
  const gfx::detail::StreamRoom room = g_room != nullptr ? *g_room : gfx::detail::stream_room();
  for (SealRoom& seal : g_seals) {
    if (std::exchange(seal.sealed, false)) {
      seal.after[seal.ring] = room.lastFrameStorage > seal.end ? room.lastFrameStorage - seal.end : 0;
      seal.ring = (seal.ring + 1) % ReserveFrames;
    }
  }
  if (g_frameBlocks != 0) {
    g_blocks[g_blockRing] = std::exchange(g_frameBlocks, 0);
    g_blockRing = (g_blockRing + 1) % ReserveFrames;
  }
  const uint64_t largest = *std::ranges::max_element(g_blocks);
  if (largest != 0 && g_room == nullptr) {
    gfx::detail::request_streaming_sizes(
        {.vertex = 0, .index = 0, .storage = room.lastFrameStorage + largest, .uniform = 0});
  }
  const uint32_t breaks = g_breakCount.load(std::memory_order_relaxed);
  if (breaks != g_seenBreaks) {
    g_seenBreaks = breaks;
    resync();
    return true;
  }
  return false;
}

void end_frame() noexcept {
  if (!g_active) {
    return;
  }
  rollback();
  clear_frame_captures();
  g_owed = 0;
  g_refusedLost = 0;
  g_undelivered = 0;
  g_stats.breakCount = g_breakCount.load(std::memory_order_relaxed);
  g_stats.variants = static_cast<uint32_t>(g_variants.size());
  g_stats.live = g_streamedCount;
  publish_coverage();
  std::lock_guard lock{g_publishMutex};
  g_published = g_stats;
}

void reset() noexcept {
  if (g_allocated) {
    clear_stream_state();
  }
  clear_frame_captures();
  g_owed = 0;
  g_refusedLost = 0;
  g_undelivered = 0;
  g_seals = {};
  g_blocks = {};
  g_blockRing = 0;
  g_frameBlocks = 0;
  g_prevCamera = {};
  g_framePrevCamera = {};
  g_frameReset = false;
  g_frameUse = {};
  g_coverage.store(0, std::memory_order_relaxed);
  std::lock_guard lock{g_publishMutex};
  g_published = {};
  g_live.reset();
}

bool settled() noexcept { return !g_active || g_pendingStatics == 0; }

void on_capture(const Capture& c) noexcept {
  // The capture's pointers live only for this call, so the stream state copies what it keeps
  if (!g_active || !g_allocated) {
    return;
  }
  const bool only = (c.entryFlags & retained::entry_flag::CaptureOnly) != 0;
  const bool transient = (c.entryFlags & retained::entry_flag::Transient) != 0 || c.slot >= InstanceSlots;
  g_undelivered += !transient && !c.delivered ? 1 : 0;
  const auto skip = [&](uint64_t& counter) {
    ++counter;
    g_owed += only ? 1 : 0;
  };
  if (transient) {
    return skip(g_stats.transientSkipped);
  }
  if (c.prim == GX_LINES || c.prim == GX_LINESTRIP || c.prim == GX_POINTS) {
    return skip(g_stats.lineSkipped);
  }
  if (c.vtxCount > MaxPieceVertices || c.attrs[GX_VA_POS].attrType == GX_NONE) {
    return skip(g_stats.largeSkipped);
  }
  const bool palette = c.attrs[GX_VA_PNMTXIDX].attrType != GX_NONE;
  if (palette && (c.attrs[GX_VA_PNMTXIDX].attrType != GX_DIRECT || c.attrs[GX_VA_PNMTXIDX].offset >= c.stride ||
                  c.invView == nullptr || c.records.size() < size_t{c.vtxCount} * c.stride)) {
    return skip(g_stats.largeSkipped);
  }
  if (!palette && c.pnSlot >= gx::MaxPnMtx + gx::MaxTexMtx) {
    return skip(g_stats.slotSkipped);
  }
  const bool indexed = (c.entryFlags & retained::entry_flag::Indexed) != 0;
  const uint32_t indexCount = indexed ? c.indexCount : derived_index_count(c.prim, c.vtxCount);
  if (indexCount == 0 || indexCount % 3 != 0) {
    return skip(g_stats.largeSkipped);
  }
  SlotState& st = g_slots[c.slot];
  if (st.generation != c.slotGeneration) {
    if (st.recordSent || st.vtxBlock != NoBlock || st.bank != NoBank) {
      on_freed(c.slot);
    }
    st.generation = c.slotGeneration;
  }

  const map_pick::MapSelection maps = map_pick::select_maps(gx::g_gxState);
  const auto tex_present = [&](const map_pick::MapPick& pick) {
    return pick.valid() && pick.texcoordSource && c.attrs[GX_VA_TEX0 + pick.srcTex].attrType != GX_NONE;
  };
  const bool alphaSame = maps.alpha.valid() && maps.albedo.valid() && maps.alpha.coord == maps.albedo.coord;
  const uint8_t uvTex = tex_present(maps.albedo) ? maps.albedo.srcTex : NoTex;
  const uint8_t alphaTex = !alphaSame && tex_present(maps.alpha) ? maps.alpha.srcTex : NoTex;

  Pending& p = g_pending.emplace_back();
  p.slot = c.slot;
  p.generation = c.slotGeneration;
  p.identityHash = c.identityHash;
  p.entryFlags = c.entryFlags;
  p.variant = find_variant(c.attrs, c.stride, uvTex, alphaTex);
  p.classBits = c.classBits;
  p.categories = c.categories;
  p.objectFlags = c.objectFlags;
  p.newGeometry = c.newGeometry;
  p.latched = c.latched;
  p.teleport = c.teleport;
  p.gpuArrays = c.gpuArrays;
  p.hasNormalAttr = c.attrs[GX_VA_NRM].attrType != GX_NONE;
  p.prim = c.prim;
  p.vtxCount = c.vtxCount;
  p.indexCount = indexCount;
  p.idxByte = indexed ? c.idxRange.offset : 0;
  p.immediates = *c.immediates;
  p.world = c.world != nullptr ? *c.world : identity_affine();
  p.eye = c.eye;
  p.objectId = c.objectId;
  p.clusters = {c.clusterFirst, c.clusterWord};
  st.clusterWant = p.clusters;
  if (palette) {
    p.palette = add_palette(c, st);
    p.parity = add_parity(c, g_palettes[p.palette]);
  }

  const bool normalSource = palette || c.pnSlot < gx::MaxPnMtx;
  const bool immobile = (c.classBits & retained::class_bit::Immobile) != 0;
  uint32_t flags = instance_flag::Valid;
  const auto set = [&](bool on, uint32_t bit) { flags |= on ? bit : 0u; };
  set(palette, instance_flag::Palette);
  set((c.classBits & retained::class_bit::CpuDeformed) != 0, instance_flag::CpuDeformed);
  set(c.gpuArrays, instance_flag::GpuArrays);
  set(palette ? immobile : c.latched, instance_flag::Latched);
  set((c.classBits & retained::class_bit::Billboard) != 0, instance_flag::Billboard);
  set(!p.hasNormalAttr || !normalSource, instance_flag::NoNormal);
  set((c.classBits & retained::class_bit::Water) != 0, instance_flag::Water);
  set((c.entryFlags & retained::entry_flag::Late) != 0, instance_flag::Late);
  set((c.classBits & retained::class_bit::Blended) != 0, instance_flag::Blended);
  set(!palette && determinant(p.world) < 0.0, instance_flag::NegativeDeterminant);
  set(alphaSame, instance_flag::AlphaUvIsUv0);
  set(c.attrs[GX_VA_CLR0].attrType == GX_NONE, instance_flag::NoColor);
  flags |= (static_cast<uint32_t>(gx::g_gxState.cullMode) << instance_flag::CullShift) & instance_flag::CullMask;
  std::array<double, 6> xform{};
  if (uvTex != NoTex) {
    set(!compose_uv(maps.albedo.coord, c.attrs, xform), instance_flag::UvNonAffine);
    store_uv(xform, &p.uv[0]);
  } else {
    flags |= instance_flag::UvNonTex;
  }
  if (maps.alpha.valid() && maps.alpha.texcoordSource) {
    if (!compose_uv(maps.alpha.coord, c.attrs, xform)) {
      flags |= instance_flag::UvNonAffine;
    }
    store_uv(xform, &p.uv[6]);
  }
  p.flags = flags;
  p.material = NoMaterial | (maps.alpha.valid() ? static_cast<uint32_t>(maps.alpha.coord) : NoCoord) << 16;
  p.pieceFlags = (p.hasNormalAttr && normalSource ? block::PieceHasNormal : 0u) |
                 (alphaSame ? block::PieceAlphaUvIsUv0 : 0u) | (indexed ? block::PieceIndexed : 0u) |
                 (alphaTex != NoTex ? block::PieceAlphaUv : 0u);
  if (palette) {
    for (int r = 0; r < 3; ++r) {
      p.normal[r * 4 + r] = f32_bits(1.f);
    }
  } else if (normalSource) {
    const Affine nrm = to_affine(gx::g_gxState.pnMtx[c.pnSlot].nrm);
    const Affine& inv = *c.invView;
    for (int r = 0; r < 3; ++r) {
      for (int k = 0; k < 3; ++k) {
        p.normal[r * 4 + k] = f32_bits(
            static_cast<float>(inv.m[r][0] * nrm.m[0][k] + inv.m[r][1] * nrm.m[1][k] + inv.m[r][2] * nrm.m[2][k]));
      }
    }
  }
  if ((c.classBits & StaticBits) != 0 && !st.streamed && !st.staticPending) {
    st.staticPending = true;
    ++g_pendingStatics;
  }
}

void on_freed(uint32_t slot) noexcept {
  if (!g_allocated || slot >= InstanceSlots) {
    return;
  }
  SlotState& st = g_slots[slot];
  if (st.staticPending) {
    --g_pendingStatics;
  }
  if (st.streamed) {
    --g_streamedCount;
  }
  if (st.recordSent || st.vtxBlock != NoBlock || st.idxBlock != NoBlock || st.bank != NoBank) {
    g_freed.push_back(
        {.slot = slot, .vtx = st.vtxBlock, .idx = st.idxBlock, .bank = st.bank, .recordSent = st.recordSent});
  }
  const uint32_t generation = st.generation;
  st = {};
  st.generation = generation;
  g_liveChanged = true;
}

void on_clusters(uint32_t slot, uint32_t slotGeneration, uint32_t first, uint32_t word) noexcept {
  if (!g_active || !g_allocated || slot >= InstanceSlots) {
    return;
  }
  SlotState& st = g_slots[slot];
  if (st.generation != slotGeneration) {
    return;
  }
  st.clusterWant = {first, word};
  if (st.recordSent && st.clusterWant != st.clusters && !std::exchange(st.clusterQueued, true)) {
    g_clusterDirty.emplace_back(slot, slotGeneration);
  }
}

bool streamed(uint32_t slot) noexcept { return g_allocated && slot < g_slots.size() && g_slots[slot].streamed; }

void seal(retained::Frame& out, const detail::StageData& stage) noexcept {
  if (!g_active || !g_allocated) {
    return;
  }
  rollback();
  g_refusedLost = 0;
  const bool late = stage.stage == Stage::PrePost;
  if (!late && stage.stage != Stage::OpaqueDone) {
    return;
  }
  const gfx::detail::StreamRoom room = g_room != nullptr ? *g_room : gfx::detail::stream_room();
  Tentative t;
  t.palettes.reserve(g_palettes.size());
  t.camera = current_camera();
  const bool cut = g_framePrevCamera.valid && t.camera.cutEpoch != g_framePrevCamera.cutEpoch;
  const bool epochChanged = detail::live_scene_hints().sceneEpochChanged;
  std::vector<Pick> picks;
  plan_pieces(t, room.overrun, cut, epochChanged, picks);
  const std::vector<uint32_t> clusterSlots = dirty_clusters(picks);
  if (late && !late_work(t, picks, clusterSlots)) {
    touch(picks);
    return;
  }
  ++g_stats.plans;
  SealRoom& sealRoom = g_seals[late ? 1 : 0];
  const uint64_t reserve = *std::ranges::max_element(sealRoom.after);

  auto plan = std::make_shared<Plan>();
  plan->frame = g_hintFrame;
  plan->frameIndex = stage.frameIndex;
  plan->serial = ++g_serial == 0 ? ++g_serial : g_serial;
  plan->clearInstances = g_clearPending;
  plan->baseSerial = g_clearPending ? 0 : g_lastCommitted;
  plan->late = late;
  const BuildInput base{
      .clusterSlots = clusterSlots,
      .camera = t.camera,
      .prevCamera = g_framePrevCamera.valid ? g_framePrevCamera : t.camera,
      .frameIndex = stage.frameIndex,
      .serial = plan->serial,
      .clearInstances = plan->clearInstances,
      .reset = plan->clearInstances || (late && g_frameReset),
      .late = late,
  };
  BuildInput input = base;
  input.picks = std::move(picks);
  input.palettes = &t.palettes;
  input.frees = g_freed;
  BuildOutput built = build_block(input);
  const uint64_t blockBytes = built.block.size() * sizeof(uint32_t);
  const bool tooLarge = blockBytes > MaxBlockBytes;
  const bool headroom = !room.storageOwned && room.storageUsed + blockBytes + reserve > room.storageCapacity;
  if ((tooLarge || headroom) && !input.picks.empty()) {
    ++g_stats.headroomDeferred;
    if (headroom && g_room == nullptr) {
      gfx::detail::request_streaming_sizes(
          {.vertex = 0, .index = 0, .storage = room.storageCapacity + blockBytes + reserve, .uniform = 0});
    }
    undo_allocations(t);
    t.lostAtCommit += t.captureOnly;
    t.captureOnly = 0;
    t.use = g_frameUse;
    t.palettes.clear();
    t.paletteCounts = {};
    input = base;
    input.frees = g_freed;
    built = build_block(input);
  }
  t.paletteCounts.shared = built.sharedPalettes;
  plan->headerOnly = built.pieces == 0;
  g_stats.headerOnly += plan->headerOnly ? 1 : 0;
  plan->writeJobs = built.writeJobs;
  plan->rollJobs = built.rollJobs;
  plan->idxChunks = built.idxChunks;
  plan->variants = std::move(built.variants);
  plan->events = std::move(built.events);
  if (!late && check_frame()) {
    plan->check = make_check(stage.frameIndex, built.commits, parity_entries(input.picks));
  }
  const std::span<const uint8_t> bytes{reinterpret_cast<const uint8_t*>(built.block.data()), built.block.size() * 4};
  const gfx::Range range = (g_push != nullptr ? g_push : &default_push)(bytes);
  g_block = std::move(built.block);
  if (range.size == 0) {
    ++g_stats.refused;
    undo_allocations(t);
    g_refusedLost = t.captureOnly + t.lostAtCommit;
    if (g_room == nullptr) {
      gfx::detail::request_streaming_sizes(
          {.vertex = 0, .index = 0, .storage = room.storageCapacity + blockBytes + reserve, .uniform = 0});
    }
    return;
  }
  plan->blockWord = range.offset / 4;
  g_stats.streamedVertices += built.vertices;
  t.plan = plan;
  t.commits = std::move(built.commits);
  t.clusterCommits = std::move(built.clusterCommits);
  t.frees = std::move(input.frees);
  g_freed.clear();
  t.blockBytes = static_cast<uint32_t>(bytes.size());
  t.pieces = built.pieces;
  t.sealStorage = room.storageUsed + bytes.size();
  sealRoom.end = t.sealStorage;
  sealRoom.sealed = true;
  g_frameBlocks += bytes.size();
  g_tentative = std::move(t);
  g_lastPlan = plan;
  out.plan = std::move(plan);
}

void skip(const detail::StageData& stage) noexcept {
  if (!g_active || stage.stage != Stage::OpaqueDone) {
    return;
  }
  g_refusedLost = 0;
  rollback();
}

void sealed(const detail::StageData& stage) noexcept {
  if (!g_active || (stage.stage != Stage::OpaqueDone && stage.stage != Stage::PrePost)) {
    return;
  }
  uint32_t lost = std::exchange(g_owed, 0) + std::exchange(g_refusedLost, 0);
  if (g_tentative.plan) {
    lost += g_tentative.committed ? g_tentative.lostAtCommit : g_tentative.captureOnly + g_tentative.lostAtCommit;
    rollback();
  }
  clear_frame_captures();
  g_undelivered = 0;
  report(lost);
}

void drop_pending() noexcept {
  if (!g_active) {
    return;
  }
  rollback();
  clear_frame_captures();
  g_owed = 0;
  g_refusedLost = 0;
  g_undelivered = 0;
}

void commit_stage_params(const detail::StageData& stage, const void*, std::span<std::byte> params) {
  std::array<uint32_t, 4> out{};
  const Plan* plan = stage.retained != nullptr ? stage.retained->plan.get() : nullptr;
  // build_stage_bindings runs this only for passes that survived the marker, pipeline and cull checks
  if (plan != nullptr && g_tentative.plan.get() == plan && !g_tentative.committed) {
    commit_plan(g_tentative);
    out = {plan->serial, plan->baseSerial, plan->clearInstances ? 1u : 0u, plan->blockWord};
  }
  std::memcpy(params.data(), out.data(), std::min(params.size(), sizeof(out)));
}

Coverage coverage() noexcept {
  const uint64_t packed = g_coverage.load(std::memory_order_acquire);
  return {.frame = static_cast<uint32_t>(packed >> 32),
          .pending = static_cast<uint32_t>((packed >> 16) & 0xFFFF),
          .streamed = static_cast<uint32_t>(packed & 0xFFFF)};
}

std::shared_ptr<const std::vector<LiveInstance>> live() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_live;
}

Stats stats() noexcept {
  std::lock_guard lock{g_publishMutex};
  return g_published;
}

bool applied(uint32_t serial) noexcept { return g_appliedSerial.load(std::memory_order_acquire) == serial; }

void worker_frame_begin(uint64_t frameId, const detail::FrameState*) {
  WorkerState& w = g_worker;
  w.frame = frameId;
  if (w.group && frameId > w.lastUsed + ReserveFrames) {
    w.group = {};
    w.buffers = {};
  }
}

detail::PassDesc pass_desc() {
  return {
      .name = std::string{PassName},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 0,
      .runsWithoutMarker = true,
      .paramsSize = ParamsSize,
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = pass_pipelines,
      .stageParams = commit_stage_params,
      .state = &OpaquePhases,
  };
}

detail::PassDesc late_pass_desc() {
  return {
      .name = std::string{LatePassName},
      .stage = Stage::PrePost,
      .kind = PassKind::Compute,
      .cls = detail::PassClass::Required,
      .order = 0,
      .paramsSize = ParamsSize,
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
      .pipelines = pass_pipelines,
      .stageParams = commit_stage_params,
      .state = &LatePhases,
  };
}

namespace testing {
void reset() noexcept {
  streamout::reset();
  g_stats = {};
  release_state();
  g_active = false;
  g_envRead = true;
  g_push = nullptr;
  g_room = nullptr;
  g_variantsReady = false;
  g_checkOverride = false;
  g_checkMode = false;
  g_wirePath.clear();
  g_variants.clear();
  g_lastVariant = NoVariant;
  g_serial = 0;
  g_lastCommitted = 0;
  g_seenBreaks = g_breakCount.load(std::memory_order_relaxed);
  g_appliedSerial.store(0, std::memory_order_relaxed);
  g_block.clear();
  g_lastPlan.reset();
  g_paletteObjects.clear();
}

void set_push(PushFn push) noexcept { g_push = push; }

void set_room(const gfx::detail::StreamRoom* room) noexcept { g_room = room; }

void set_variants_ready(bool ready) noexcept { g_variantsReady = ready; }

void set_check(bool on) noexcept { g_checkOverride = on; }

SlotState slot(uint32_t slot) noexcept {
  if (!g_allocated || slot >= g_slots.size()) {
    return {};
  }
  const auto& st = g_slots[slot];
  return {.streamed = st.streamed,
          .worldSent = st.worldSent,
          .vertexFirst = st.vertexFirst,
          .vertexCount = st.vertexCount,
          .indexFirst = st.indexFirst,
          .indexCount = st.indexCount,
          .flags = st.flags,
          .streams = st.streams,
          .recordWrites = st.recordWrites,
          .lastSent = st.lastSent,
          .clusterFirst = st.clusters[0],
          .clusterWord = st.clusters[1],
          .paletteWord = st.bank != NoBank ? st.bank * palette::PairWords : RigidMatrix,
          .usedMask = st.usedMask,
          .seenMask = st.seenMask,
          .origin = st.origin};
}

std::span<const uint32_t> last_block() noexcept { return g_block; }

std::shared_ptr<const Plan> last_plan() noexcept { return g_lastPlan; }

uint32_t variant_count() noexcept { return static_cast<uint32_t>(g_variants.size()); }

void break_chain() noexcept { g_breakCount.fetch_add(1, std::memory_order_relaxed); }

bool encode_plan(const wgpu::CommandEncoder& encoder, std::span<const BufferView, 8> views, const Plan& plan) {
  FixedPipelines fixed;
  for (size_t i = 0; i < fixed.size(); ++i) {
    const shaders::PipelineId id = shaders::request_compute({
        .label = PassName,
        .shader = LibraryShader,
        .entryPoint = PipelineEntries[i],
        .layout = g_pipelineLayout,
        .layoutKey = LayoutKey,
    });
    if (shaders::wait(id) != shaders::PipelineStatus::Ready) {
      return false;
    }
    fixed[i] = shaders::compute_pipeline(id);
  }
  std::array<BufferView, PersistentCount> bound;
  std::copy(views.begin(), views.end(), bound.begin());
  g_appliedSerial.store(0, std::memory_order_relaxed);
  encode_stream(encoder, PassName, bound, fixed, plan, plan.serial, plan.baseSerial, plan.clearInstances);
  return g_appliedSerial.load(std::memory_order_relaxed) == plan.serial;
}

bool wait_variants(const Plan& plan) {
  return std::ranges::all_of(plan.variants, [](const VariantDispatch& run) {
    return shaders::wait(run.pipeline) == shaders::PipelineStatus::Ready;
  });
}

Chain chain_step(ChainState& state, uint32_t serial, uint32_t baseSerial, bool clearInstances) noexcept {
  if (clearInstances && baseSerial == 0) {
    state.broken = false;
    state.lastEncoded = serial;
    return Chain::Accept;
  }
  if (state.broken) {
    return Chain::Skip;
  }
  if (baseSerial != state.lastEncoded) {
    state.broken = true;
    return Chain::Break;
  }
  state.lastEncoded = serial;
  return Chain::Accept;
}
} // namespace testing
} // namespace aurora::render::scene::streamout

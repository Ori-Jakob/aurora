#include "compact.hpp"

#include "indirect.hpp"
#include "../pass_context.hpp"

#include "../../gx/gx.hpp"
#include "../../internal.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"
#include "../../webgpu/gpu_prof.hpp"

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace aurora::render::compact {
using Key = std::array<uint64_t, 3 * StorageBindings>;

struct BindingAccess {
  static wgpu::BindGroup& group(Binding& binding) noexcept { return binding.m_group; }
  static const wgpu::BindGroup& group(const Binding& binding) noexcept { return binding.m_group; }
  static Key& key(Binding& binding) noexcept { return binding.m_key; }
  static const Key& key(const Binding& binding) noexcept { return binding.m_key; }
};

namespace {
constexpr Module Log{"aurora::render::compact"};
constexpr uint64_t LayoutKey = 0x636F6D7061637401;
constexpr uint64_t DummySize = 32;
constexpr uint32_t PartialsWord = HeaderWords;
constexpr uint64_t IdleFrames = 4;
constexpr uint64_t HeaderBytes = HeaderWords * sizeof(uint32_t);
constexpr uint16_t NoRef = UINT16_MAX;

constexpr uint32_t ModeReduce = 0;
constexpr uint32_t ModeDown = 1;
constexpr uint32_t ModeArgs = 2;
constexpr uint32_t ModeZero = 3;
constexpr uint32_t FlagLevel0 = 1;
constexpr uint32_t FlagCompact = 2;
constexpr uint32_t FlagLive = 4;
constexpr uint32_t FlagParent = 8;
constexpr uint32_t FlagTotal = 16;

enum Slot : uint8_t { SlotInput, SlotOutput, SlotScratch, SlotCount, SlotArgs };

struct Immediates {
  std::array<uint32_t, 16> words{};
};
static_assert(sizeof(Immediates) == ImmediateSize);
static_assert(ImmediateSize <= sizeof(gx::DrawImmediateData));
static_assert(sizeof(indirect::DispatchArgs) == 12 && sizeof(indirect::DrawArgs) == 16 &&
              sizeof(indirect::DrawIndexedArgs) == 20);

struct GraphLocal {
  uint32_t capacity = 0;
  uint32_t countWord = 0;
  uint32_t keepMask = UINT32_MAX;
  uint32_t argsOffset = 0;
  uint32_t argsScale = 1;
  uint32_t argsLimit = 0;
  std::array<uint32_t, 5> record{};
  std::array<uint16_t, StorageBindings> refs{NoRef, NoRef, NoRef, NoRef, NoRef};
  Op op = Op::Scan;
  ArgsKind kind = ArgsKind::None;
  uint8_t field = 0;
};
static_assert(sizeof(GraphLocal) <= detail::PassLocalSize);
static_assert(std::is_trivially_copyable_v<GraphLocal>);

constexpr std::array<std::string_view, 12> RejectNames{
    "None", "Unsupported", "Device",    "Pipeline", "Plan", "KeepMask",
    "Args", "Usage",       "Alignment", "Range",    "Size", "Overlap",
};
static_assert(RejectNames.size() == static_cast<size_t>(Reject::Overlap) + 1);

bool g_supported = false;
wgpu::Device g_device;
wgpu::BindGroupLayout g_groupLayout;
wgpu::PipelineLayout g_pipelineLayout;
std::array<wgpu::Buffer, StorageBindings> g_dummies;
testing::Options g_options;

std::mutex g_pipelineMutex;
uint64_t g_cachedSession = 0;
shaders::PipelineId g_cachedId;
std::optional<shaders::PipelineId> g_forced;

std::atomic_uint64_t g_jobs{0};
std::atomic_uint64_t g_passes{0};
std::atomic_uint64_t g_dispatches{0};
std::atomic_uint64_t g_rejectedCount{0};
std::atomic_uint64_t g_backstops{0};
std::atomic_uint64_t g_bindGroupsCreated{0};
std::atomic_uint64_t g_bindGroupsLastFrame{0};
std::array<std::atomic_bool, RejectNames.size()> g_warnedReject{};
std::atomic_bool g_warnedIndirect{false};
std::atomic_bool g_warnedBinding{false};
std::atomic<std::thread::id> g_owner;

struct CacheEntry {
  std::array<wgpu::Buffer, StorageBindings> buffers;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

// Render worker only
struct WorkerState {
  uint64_t frame = 0;
  uint64_t createdThisFrame = 0;
  uint64_t pipelineSession = 0;
  uint32_t pipelineId = 0;
  wgpu::ComputePipeline pipeline;
  absl::flat_hash_map<Key, CacheEntry> cache;
};
WorkerState g_worker;

inline void bump(std::atomic_uint64_t& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void check_owner() noexcept {
  const auto self = std::this_thread::get_id();
  auto owner = std::thread::id{};
  if (!g_owner.compare_exchange_strong(owner, self, std::memory_order_relaxed)) {
    CHECK(owner == self, "compact: encodes must stay on the thread that made the first one");
  }
}

constexpr uint32_t ceil_div(uint32_t value, uint32_t divisor) noexcept { return (value + divisor - 1) / divisor; }

constexpr uint32_t round4(uint32_t value) noexcept { return (value + 3) & ~3u; }

bool device_matches() noexcept { return g_device && webgpu::g_device.Get() == g_device.Get(); }

uint32_t effective_limit(const Job& job) noexcept {
  const uint32_t capacity = job.plan.capacity;
  if (job.plan.op == Op::Compact) {
    return job.args.limit == 0 ? capacity : std::min(job.args.limit, capacity);
  }
  return job.args.limit;
}

Reject validate_spec(const Job& job, const Caps& caps) noexcept {
  if (!supported(caps)) {
    return Reject::Unsupported;
  }
  if (!job.plan.valid() || job.plan != make_plan(job.plan.op, job.plan.capacity, caps) ||
      job.plan.capacity > max_capacity(caps)) {
    return Reject::Plan;
  }
  if (job.plan.op == Op::Scan && job.keepMask != UINT32_MAX) {
    return Reject::KeepMask;
  }
  const ArgsSpec& args = job.args;
  if (args.kind == ArgsKind::None) {
    return Reject::None;
  }
  if (args.kind > ArgsKind::DrawIndexed || args.scale == 0 || (job.plan.op == Op::Scan && args.limit == 0) ||
      args.offset % 4 != 0) {
    return Reject::Args;
  }
  const uint64_t limit = effective_limit(job);
  if (args.kind == ArgsKind::Dispatch) {
    const uint64_t groups = (limit + args.scale - 1) / args.scale;
    return groups > caps.limits.maxComputeWorkgroupsPerDimension ? Reject::Args : Reject::None;
  }
  if (limit * args.scale > UINT32_MAX || args.field >= 2) {
    return Reject::Args;
  }
  const uint32_t firstInstance = args.kind == ArgsKind::Draw ? args.record[3] : args.record[4];
  if (firstInstance != 0 && !caps.indirectFirstInstance) {
    return Reject::Args;
  }
  return Reject::None;
}

struct View {
  const BufferView* view = nullptr;
  uint64_t need = 0;
};

uint64_t storage_alignment(const Caps& caps) noexcept {
  return std::max<uint64_t>(caps.limits.minStorageBufferOffsetAlignment, 4);
}

Reject check_view(const BufferView& view, const Caps& caps) noexcept {
  if (!view.buffer || !(view.buffer.GetUsage() & wgpu::BufferUsage::Storage)) {
    return Reject::Usage;
  }
  if (view.offset % storage_alignment(caps) != 0 || view.size % 4 != 0) {
    return Reject::Alignment;
  }
  if (view.size == 0 || view.offset + view.size > view.buffer.GetSize() ||
      view.size > caps.limits.maxStorageBufferBindingSize) {
    return Reject::Range;
  }
  return Reject::None;
}

bool bindable(const BufferView& view, const Caps& caps) noexcept { return check_view(view, caps) == Reject::None; }

bool overlaps(const BufferView& a, const BufferView& b) noexcept {
  return a.buffer.Get() == b.buffer.Get() && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

// Dawn's D3D11 path keeps only one UAV per buffer in a dispatch and drops the others
bool conflicts(const BufferView& a, const BufferView& b, const Caps& caps) noexcept {
  return overlaps(a, b) || (caps.backend == BACKEND_D3D11 && a.buffer && a.buffer.Get() == b.buffer.Get());
}

Reject validate_views(const Job& job, const Caps& caps) noexcept {
  const uint64_t elements = uint64_t{job.plan.capacity} * sizeof(uint32_t);
  std::array<View, StorageBindings> views{};
  size_t count = 0;
  views[count++] = {&job.input, elements};
  views[count++] = {&job.output, elements};
  views[count++] = {&job.scratch, job.plan.scratchBytes};
  if (job.count.buffer) {
    views[count++] = {&job.count, (uint64_t{job.countWord} + 1) * sizeof(uint32_t)};
  }
  if (job.args.kind != ArgsKind::None) {
    views[count++] = {&job.argsTarget, uint64_t{job.args.offset} + args_record_words(job.args.kind) * 4};
  }
  const std::span<const View> bound{views.data(), count};
  for (const View& entry : bound) {
    if (const Reject reason = check_view(*entry.view, caps); reason != Reject::None) {
      return reason;
    }
  }
  if (!g_options.skipSizeChecks) {
    for (const View& entry : bound) {
      if (entry.view->size < entry.need) {
        return Reject::Size;
      }
    }
  }
  for (size_t i = 0; i < bound.size(); ++i) {
    for (size_t j = i + 1; j < bound.size(); ++j) {
      if (conflicts(*bound[i].view, *bound[j].view, caps)) {
        return Reject::Overlap;
      }
    }
  }
  return Reject::None;
}

std::array<BufferView, StorageBindings> dummy_slots() {
  std::array<BufferView, StorageBindings> slots;
  for (size_t i = 0; i < slots.size(); ++i) {
    slots[i] = {.buffer = g_dummies[i], .offset = 0, .size = DummySize};
  }
  return slots;
}

std::array<BufferView, StorageBindings> job_slots(const Job& job) {
  std::array<BufferView, StorageBindings> slots = dummy_slots();
  slots[SlotInput] = job.input;
  slots[SlotOutput] = job.output;
  slots[SlotScratch] = job.scratch;
  if (job.count.buffer) {
    slots[SlotCount] = job.count;
  }
  if (job.args.kind != ArgsKind::None) {
    slots[SlotArgs] = job.argsTarget;
  }
  return slots;
}

Key key_of(const std::array<BufferView, StorageBindings>& slots) noexcept {
  Key key{};
  for (size_t i = 0; i < slots.size(); ++i) {
    key[i * 3] = reinterpret_cast<uintptr_t>(slots[i].buffer.Get());
    key[i * 3 + 1] = slots[i].offset;
    key[i * 3 + 2] = slots[i].size;
  }
  return key;
}

wgpu::BindGroup create_group(const std::array<BufferView, StorageBindings>& slots) {
  std::array<wgpu::BindGroupEntry, StorageBindings> entries{};
  for (uint32_t i = 0; i < StorageBindings; ++i) {
    entries[i] = {.binding = i, .buffer = slots[i].buffer, .offset = slots[i].offset, .size = slots[i].size};
  }
  const wgpu::BindGroupDescriptor descriptor{
      .label = "compact",
      .layout = g_groupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return g_device.CreateBindGroup(&descriptor);
}

const wgpu::BindGroup& cached_group(const Job& job) {
  const std::array<BufferView, StorageBindings> slots = job_slots(job);
  const Key key = key_of(slots);
  if (job.binding != nullptr && job.binding->valid()) {
    if (BindingAccess::key(*job.binding) == key) {
      return BindingAccess::group(*job.binding);
    }
    CHECK(false, "compact: a caller binding no longer matches its job's views");
    if (!g_warnedBinding.exchange(true, std::memory_order_relaxed)) {
      Log.warn("compact: a caller binding no longer matches its job's views; using the cache");
    }
  }
  WorkerState& worker = g_worker;
  if (const auto it = worker.cache.find(key); it != worker.cache.end()) {
    it->second.lastUsed = worker.frame;
    return it->second.group;
  }
  if (worker.cache.size() >= MaxCachedBindings) {
    auto oldest = worker.cache.begin();
    for (auto it = worker.cache.begin(); it != worker.cache.end(); ++it) {
      if (it->second.lastUsed < oldest->second.lastUsed) {
        oldest = it;
      }
    }
    worker.cache.erase(oldest);
  }
  CacheEntry entry{.group = create_group(slots), .lastUsed = worker.frame};
  for (size_t i = 0; i < slots.size(); ++i) {
    entry.buffers[i] = slots[i].buffer;
  }
  bump(g_bindGroupsCreated);
  ++worker.createdThisFrame;
  return worker.cache.insert_or_assign(key, std::move(entry)).first->second.group;
}

void dispatch(const wgpu::ComputePassEncoder& pass, const Immediates& immediates, uint32_t groups) {
  pass.SetImmediates(0, immediates.words.data(), ImmediateSize);
  pass.DispatchWorkgroups(groups);
  bump(g_dispatches);
}

Immediates base_immediates(const Job& job) noexcept {
  Immediates out;
  auto& w = out.words;
  w[5] = job.countWord;
  w[6] = job.keepMask;
  w[7] = job.args.offset / 4;
  w[8] = static_cast<uint32_t>(job.args.kind) | uint32_t{job.args.field} << 8;
  w[9] = job.args.scale;
  w[10] = effective_limit(job);
  w[11] = job.args.record[4];
  w[12] = job.args.record[0];
  w[13] = job.args.record[1];
  w[14] = job.args.record[2];
  w[15] = job.args.record[3];
  return out;
}

Immediates step(Immediates out, uint32_t mode, uint32_t flags, uint32_t count, uint32_t level, uint32_t link) noexcept {
  out.words[0] = mode;
  out.words[1] = flags;
  out.words[2] = count;
  out.words[3] = level;
  out.words[4] = link;
  return out;
}

void record_dispatches(const wgpu::ComputePassEncoder& pass, const Job& job) {
  const Immediates base = base_immediates(job);
  const uint32_t n = job.plan.capacity;
  const uint32_t n1 = ceil_div(n, TileSize);
  const uint32_t n2 = ceil_div(n1, TileSize);
  const uint32_t p1 = PartialsWord;
  const uint32_t p2 = PartialsWord + round4(n1);
  const uint32_t level0 =
      FlagLevel0 | (job.plan.op == Op::Compact ? FlagCompact : 0) | (job.count.buffer ? FlagLive : 0);
  switch (job.plan.levels) {
  case 1:
    dispatch(pass, step(base, ModeDown, level0 | FlagTotal, n, 0, 0), 1);
    break;
  case 2:
    dispatch(pass, step(base, ModeReduce, level0, n, 0, p1), n1);
    dispatch(pass, step(base, ModeDown, FlagTotal, n1, p1, 0), 1);
    dispatch(pass, step(base, ModeDown, level0 | FlagParent, n, 0, p1), n1);
    break;
  default:
    dispatch(pass, step(base, ModeReduce, level0, n, 0, p1), n1);
    dispatch(pass, step(base, ModeReduce, 0, n1, p1, p2), n2);
    dispatch(pass, step(base, ModeDown, FlagTotal, n2, p2, 0), 1);
    dispatch(pass, step(base, ModeDown, FlagParent, n1, p1, p2), n2);
    dispatch(pass, step(base, ModeDown, level0 | FlagParent, n, 0, p1), n1);
    break;
  }
  if (job.args.kind != ArgsKind::None) {
    dispatch(pass, step(base, ModeArgs, 0, 0, 0, 0), 1);
  }
}

bool args_zeroable(const Job& job, const Caps& caps) noexcept {
  const ArgsSpec& args = job.args;
  return args.kind != ArgsKind::None && args.kind <= ArgsKind::DrawIndexed && args.offset % 4 == 0 &&
         bindable(job.argsTarget, caps) &&
         job.argsTarget.size >= uint64_t{args.offset} + args_record_words(args.kind) * 4;
}

void backstop(const wgpu::ComputePassEncoder& pass, const Job& job, const Caps& caps) {
  std::array<BufferView, StorageBindings> slots = dummy_slots();
  Immediates immediates;
  if (args_zeroable(job, caps)) {
    slots[SlotArgs] = job.argsTarget;
    immediates.words[7] = job.args.offset / 4;
    immediates.words[8] = static_cast<uint32_t>(job.args.kind);
  }
  if (bindable(job.scratch, caps) && job.scratch.size >= 2 * sizeof(uint32_t)) {
    const BufferView header{
        .buffer = job.scratch.buffer, .offset = job.scratch.offset, .size = std::min(job.scratch.size, HeaderBytes)};
    if (!conflicts(header, slots[SlotArgs], caps)) {
      slots[SlotScratch] = header;
    }
  }
  pass.SetBindGroup(0, create_group(slots));
  immediates.words[0] = ModeZero;
  dispatch(pass, immediates, 1);
  bump(g_backstops);
}

void clear_outputs(const wgpu::CommandEncoder& encoder, const Job& job, const Caps& caps) {
  const auto clearable = [](const wgpu::Buffer& buffer) {
    return static_cast<bool>(buffer.GetUsage() & wgpu::BufferUsage::CopyDst);
  };
  if (args_zeroable(job, caps) && clearable(job.argsTarget.buffer)) {
    encoder.ClearBuffer(job.argsTarget.buffer, job.argsTarget.offset + job.args.offset,
                        args_record_words(job.args.kind) * 4);
  }
  if (bindable(job.scratch, caps) && job.scratch.size >= 2 * sizeof(uint32_t) && clearable(job.scratch.buffer)) {
    encoder.ClearBuffer(job.scratch.buffer, job.scratch.offset, 2 * sizeof(uint32_t));
  }
}

void note_rejection(Reject reason) {
  bump(g_rejectedCount);
  CHECK(g_options.releaseRejections, "compact: rejected a job: {}", reject_name(reason));
  const auto index = static_cast<size_t>(reason);
  if (index < g_warnedReject.size() && !g_warnedReject[index].exchange(true, std::memory_order_relaxed)) {
    Log.warn("compact: rejected a job ({}), so its args and count read zero", reject_name(reason));
  }
}

void warn_indirect(const Job& job) {
  if (job.args.kind != ArgsKind::None && job.argsTarget.buffer &&
      !(job.argsTarget.buffer.GetUsage() & wgpu::BufferUsage::Indirect) &&
      !g_warnedIndirect.exchange(true, std::memory_order_relaxed)) {
    Log.warn("compact: an args target lacks Indirect usage, so no indirect call can read it");
  }
}

bool encode_job(const wgpu::ComputePassEncoder& pass, const Job& job, const Caps& caps) {
  Reject reason = validate(job, caps);
  if (reason == Reject::None) {
    warn_indirect(job);
    const wgpu::BindGroup& group = cached_group(job);
    if (group) {
      pass.SetBindGroup(0, group);
      record_dispatches(pass, job);
      bump(g_jobs);
      return true;
    }
    reason = Reject::Device;
  }
  note_rejection(reason);
  backstop(pass, job, caps);
  return false;
}

bool reject_all(std::span<const Job> jobs, Reject reason) {
  for (size_t i = 0; i < jobs.size(); ++i) {
    note_rejection(reason);
  }
  return false;
}

wgpu::ComputePipeline worker_pipeline() {
  const shaders::PipelineId id = pipeline();
  WorkerState& worker = g_worker;
  const uint64_t session = shaders::session();
  if (worker.pipeline && worker.pipelineId == id.value && worker.pipelineSession == session) {
    return worker.pipeline;
  }
  wgpu::ComputePipeline compute = shaders::compute_pipeline(id);
  if (compute) {
    worker.pipeline = compute;
    worker.pipelineId = id.value;
    worker.pipelineSession = session;
  }
  return compute;
}

bool encode_into(const wgpu::ComputePassEncoder& pass, std::span<const Job> jobs, const Caps& caps) {
  bool ok = true;
  for (const Job& job : jobs) {
    ok = encode_job(pass, job, caps) && ok;
  }
  return ok;
}

GraphLocal read_local(std::span<const std::byte, detail::PassLocalSize> local) noexcept {
  GraphLocal out;
  std::memcpy(&out, local.data(), sizeof(out));
  return out;
}
} // namespace

bool supported(const Caps& caps) noexcept {
  const DeviceLimits& limits = caps.limits;
  return caps.valid && limits.maxImmediateSize >= ImmediateSize &&
         stage_binding_limits(caps, ShaderStage::Compute).storageBuffers >= StorageBindings &&
         limits.maxComputeInvocationsPerWorkgroup >= WorkgroupSize &&
         limits.maxComputeWorkgroupSizeX >= WorkgroupSize &&
         limits.maxComputeWorkgroupStorageSize >= WorkgroupSize * sizeof(uint32_t);
}

uint32_t max_capacity(const Caps& caps) noexcept {
  if (!supported(caps)) {
    return 0;
  }
  const uint64_t binding = caps.limits.maxStorageBufferBindingSize / sizeof(uint32_t);
  const uint64_t groups = uint64_t{caps.limits.maxComputeWorkgroupsPerDimension} * TileSize;
  return static_cast<uint32_t>(std::min({uint64_t{MaxCapacity}, binding, groups}));
}

Plan make_plan(Op op, uint32_t capacity, const Caps& caps) noexcept {
  if (capacity == 0 || capacity > max_capacity(caps)) {
    return {};
  }
  const uint32_t n1 = ceil_div(capacity, TileSize);
  const uint32_t n2 = ceil_div(n1, TileSize);
  if (capacity <= TileSize) {
    return {.op = op, .capacity = capacity, .levels = 1, .dispatches = 1, .scratchBytes = HeaderBytes};
  }
  if (capacity <= TileSize * TileSize) {
    return {.op = op,
            .capacity = capacity,
            .levels = 2,
            .dispatches = 3,
            .scratchBytes = 4 * (uint64_t{HeaderWords} + round4(n1))};
  }
  return {.op = op,
          .capacity = capacity,
          .levels = 3,
          .dispatches = 5,
          .scratchBytes = 4 * (uint64_t{HeaderWords} + round4(n1) + round4(n2))};
}

uint32_t args_record_words(ArgsKind kind) noexcept {
  switch (kind) {
  case ArgsKind::Dispatch:
    return 3;
  case ArgsKind::Draw:
    return 4;
  case ArgsKind::DrawIndexed:
    return 5;
  default:
    return 0;
  }
}

Reject validate(const Job& job, const Caps& caps) noexcept {
  if (const Reject reason = validate_spec(job, caps); reason != Reject::None) {
    return reason;
  }
  return validate_views(job, caps);
}

std::string_view reject_name(Reject reason) noexcept {
  const auto index = static_cast<size_t>(reason);
  return index < RejectNames.size() ? RejectNames[index] : "Unknown";
}

void initialize() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return;
  }
  if (!supported(caps())) {
    Log.info("GPU scan and compaction unavailable: the device lacks the compute limits it needs");
    return;
  }
  std::array<wgpu::BindGroupLayoutEntry, StorageBindings> entries{};
  for (uint32_t i = 0; i < StorageBindings; ++i) {
    entries[i] = {
        .binding = i,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Storage, .minBindingSize = 0},
    };
  }
  const wgpu::BindGroupLayoutDescriptor groupDescriptor{
      .label = "compact",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  wgpu::BindGroupLayout groupLayout = device.CreateBindGroupLayout(&groupDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineDescriptor{
      .label = "compact",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &groupLayout,
      .immediateSize = ImmediateSize,
  };
  wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineDescriptor);
  std::array<wgpu::Buffer, StorageBindings> dummies;
  for (wgpu::Buffer& dummy : dummies) {
    const wgpu::BufferDescriptor descriptor{
        .label = "compact dummy",
        .usage = wgpu::BufferUsage::Storage,
        .size = DummySize,
    };
    dummy = device.CreateBuffer(&descriptor);
  }
  std::lock_guard lock{g_pipelineMutex};
  g_device = device;
  g_groupLayout = std::move(groupLayout);
  g_pipelineLayout = std::move(pipelineLayout);
  g_dummies = std::move(dummies);
  g_supported = true;
}

void shutdown() {
  {
    std::lock_guard lock{g_pipelineMutex};
    g_supported = false;
    g_device = {};
    g_groupLayout = {};
    g_pipelineLayout = {};
    g_dummies = {};
    g_cachedSession = 0;
    g_cachedId = {};
    g_forced.reset();
  }
  g_worker = {};
  g_owner.store(std::thread::id{}, std::memory_order_relaxed);
  g_options = {};
  g_bindGroupsLastFrame.store(0, std::memory_order_relaxed);
}

void worker_frame_begin(uint64_t frameId) {
  WorkerState& worker = g_worker;
  if (!g_supported || (worker.cache.empty() && worker.createdThisFrame == 0)) {
    worker.frame = frameId;
    return;
  }
  check_owner();
  worker.frame = frameId;
  g_bindGroupsLastFrame.store(std::exchange(worker.createdThisFrame, 0), std::memory_order_relaxed);
  absl::erase_if(worker.cache, [frameId](const auto& entry) { return frameId - entry.second.lastUsed > IdleFrames; });
}

shaders::PipelineId pipeline() {
  std::lock_guard lock{g_pipelineMutex};
  if (g_forced) {
    return *g_forced;
  }
  if (!g_supported || !device_matches()) {
    return shaders::NeverReady;
  }
  const uint64_t session = shaders::session();
  if (g_cachedId && g_cachedSession == session) {
    return g_cachedId;
  }
  const shaders::PipelineId id = shaders::request_compute({
      .label = "compact",
      .shader = "scan",
      .entryPoint = "main",
      .layout = g_pipelineLayout,
      .layoutKey = LayoutKey,
  });
  if (!id) {
    return shaders::NeverReady;
  }
  g_cachedSession = session;
  g_cachedId = id;
  return id;
}

bool ready() noexcept { return shaders::ready(pipeline()); }

void pass_pipelines(const detail::PipelineSetup&, const void*, std::span<uint32_t, detail::MaxPassPipelines> ids) {
  ids[0] = pipeline().value;
}

bool declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GraphJob& job) {
  const auto skip = [&] {
    builder.skip();
    return false;
  };
  if (setup.caps == nullptr || !supported(*setup.caps)) {
    return skip();
  }
  const Plan plan = make_plan(job.op, job.capacity, *setup.caps);
  const Job spec{.plan = plan, .keepMask = job.keepMask, .args = job.argsSpec};
  if (!plan.valid() || validate_spec(spec, *setup.caps) != Reject::None) {
    return skip();
  }
  const bool hasArgs = job.argsSpec.kind != ArgsKind::None;
  std::array<std::string_view, StorageBindings> names{job.input, job.output, job.scratch, job.count,
                                                      hasArgs ? job.args : std::string_view{}};
  if (job.input.empty() || job.output.empty() || job.scratch.empty() || (hasArgs && job.args.empty())) {
    return skip();
  }
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = i + 1; j < names.size(); ++j) {
      if (!names[i].empty() && names[i] == names[j]) {
        return skip();
      }
    }
  }
  GraphLocal local{
      .capacity = plan.capacity,
      .countWord = job.countWord,
      .keepMask = job.keepMask,
      .argsOffset = job.argsSpec.offset,
      .argsScale = job.argsSpec.scale,
      .argsLimit = job.argsSpec.limit,
      .record = job.argsSpec.record,
      .op = job.op,
      .kind = job.argsSpec.kind,
      .field = job.argsSpec.field,
  };
  builder.manual_bindings();
  local.refs[SlotInput] = builder.use_buffer(job.input, Access::StorageRead, Missing::DisablePass).index;
  if (!job.count.empty()) {
    local.refs[SlotCount] = builder.use_buffer(job.count, Access::StorageRead, Missing::DisablePass).index;
  }
  local.refs[SlotOutput] =
      builder.create_buffer(job.output, {.size = uint64_t{plan.capacity} * sizeof(uint32_t)}, Access::StorageWrite)
          .index;
  local.refs[SlotScratch] =
      builder
          .create_buffer(job.scratch, {.size = plan.scratchBytes, .extraUsage = wgpu::BufferUsage::CopyDst},
                         Access::StorageWrite)
          .index;
  if (hasArgs) {
    const uint64_t size = uint64_t{job.argsSpec.offset} + args_record_words(job.argsSpec.kind) * 4;
    local.refs[SlotArgs] =
        builder.create_buffer(job.args, {.size = size, .extraUsage = wgpu::BufferUsage::CopyDst}, Access::StorageWrite)
            .index;
  }
  const std::span<std::byte, detail::PassLocalSize> out = builder.local();
  std::memcpy(out.data(), &local, sizeof(local));
  return true;
}

void encode_pass(detail::PassContext& ctx, const void*) {
  const GraphLocal local = read_local(ctx.local());
  const auto view = [&](Slot slot) {
    const uint16_t ref = local.refs[slot];
    return ref != NoRef ? ctx.buffer(BufRef{ref}) : BufferView{};
  };
  const Job job{
      .plan = make_plan(local.op, local.capacity, caps()),
      .input = view(SlotInput),
      .output = view(SlotOutput),
      .scratch = view(SlotScratch),
      .count = view(SlotCount),
      .countWord = local.countWord,
      .keepMask = local.keepMask,
      .args =
          {
              .kind = local.kind,
              .offset = local.argsOffset,
              .field = local.field,
              .scale = local.argsScale,
              .limit = local.argsLimit,
              .record = local.record,
          },
      .argsTarget = view(SlotArgs),
  };
  encode(ctx, std::span{&job, 1});
}

detail::PassDesc graph_pass(std::string name, Stage stage, detail::PassDeclareFn declare, const void* state) {
  return {
      .name = std::move(name),
      .stage = stage,
      .kind = PassKind::Compute,
      .declare = declare,
      .encode = encode_pass,
      .pipelines = pass_pipelines,
      .state = state,
  };
}

Binding make_binding(const Job& job) {
  Binding out;
  if (!g_supported || !device_matches()) {
    return out;
  }
  check_owner();
  if (validate_views(job, caps()) != Reject::None) {
    return out;
  }
  const std::array<BufferView, StorageBindings> slots = job_slots(job);
  BindingAccess::group(out) = create_group(slots);
  BindingAccess::key(out) = key_of(slots);
  return out;
}

bool encode(const wgpu::CommandEncoder& encoder, std::span<const Job> jobs, std::string_view label) {
  if (!encoder) {
    return false;
  }
  if (jobs.empty()) {
    return true;
  }
  if (!g_supported) {
    return reject_all(jobs, Reject::Unsupported);
  }
  if (!device_matches()) {
    return reject_all(jobs, Reject::Device);
  }
  check_owner();
  const Caps& deviceCaps = caps();
  const wgpu::ComputePipeline compute = worker_pipeline();
  if (!compute) {
    for (const Job& job : jobs) {
      note_rejection(Reject::Pipeline);
      clear_outputs(encoder, job, deviceCaps);
    }
    return false;
  }
  const wgpu::ComputePassDescriptor descriptor{
      .label = label,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&descriptor);
  bump(g_passes);
  pass.SetPipeline(compute);
  const bool ok = encode_into(pass, jobs, deviceCaps);
  pass.End();
  return ok;
}

bool encode(const wgpu::CommandEncoder& encoder, const Job& job, std::string_view label) {
  return encode(encoder, std::span{&job, 1}, label);
}

bool encode(const detail::PassContext& ctx, std::span<const Job> jobs) {
  const PassKind kind = ctx.pass().passKind;
  const bool allowed = kind == PassKind::Compute || kind == PassKind::Encoder;
  CHECK(allowed, "compact: '{}' is a {} pass and cannot encode a scan", ctx.name(), detail::pass_kind_name(kind));
  if (!allowed) {
    return false;
  }
  return encode(ctx.encoder(), jobs, ctx.name());
}

bool encode(const wgpu::ComputePassEncoder& pass, std::span<const Job> jobs) {
  if (!pass) {
    return false;
  }
  if (jobs.empty()) {
    return true;
  }
  if (!g_supported) {
    return reject_all(jobs, Reject::Unsupported);
  }
  if (!device_matches()) {
    return reject_all(jobs, Reject::Device);
  }
  check_owner();
  const wgpu::ComputePipeline compute = worker_pipeline();
  CHECK(compute, "compact: encoding into a caller pass needs a ready pipeline");
  if (!compute) {
    return reject_all(jobs, Reject::Pipeline);
  }
  pass.SetPipeline(compute);
  return encode_into(pass, jobs, caps());
}

Stats stats() noexcept {
  return {
      .jobs = g_jobs.load(std::memory_order_relaxed),
      .passes = g_passes.load(std::memory_order_relaxed),
      .dispatches = g_dispatches.load(std::memory_order_relaxed),
      .rejected = g_rejectedCount.load(std::memory_order_relaxed),
      .backstops = g_backstops.load(std::memory_order_relaxed),
      .bindGroupsCreated = g_bindGroupsCreated.load(std::memory_order_relaxed),
      .bindGroupsLastFrame = g_bindGroupsLastFrame.load(std::memory_order_relaxed),
  };
}

namespace testing {
void set_options(const Options& options) { g_options = options; }

void force_pipeline(shaders::PipelineId id) {
  std::lock_guard lock{g_pipelineMutex};
  if (id) {
    g_forced = id;
  } else {
    g_forced.reset();
  }
}
} // namespace testing
} // namespace aurora::render::compact

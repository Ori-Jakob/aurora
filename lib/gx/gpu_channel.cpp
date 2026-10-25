#include "gpu_channel.hpp"

#include "fifo.hpp"
#include "hints.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <utility>

namespace aurora::gx::gpu_channel {
namespace {
using namespace render::gpu;
constexpr Module Log{"aurora::gx::gpu"};
constexpr uint32_t MaxGap = 256;
constexpr uint32_t HeaderSize = GX_AURORA_HINT_HEADER_SIZE;
constexpr uint32_t KnownSystemFlags = SystemGpuOnlyState;
constexpr uint32_t KnownRequirements = RequiresCore | RequiresSubgroups | RequiresShaderF16;
constexpr uint32_t KnownKernelFlags = KernelStateAtomic;
constexpr std::string_view LibraryPrefix = "systems/";

enum class Refusal : uint8_t {
  DisplayList,
  Size,
  Handles,
  Budget,
  Commands,
  Stale,
  ListBind,
  Attribute,
  Systems,
  Desc,
  SystemCap,
  ShadowCap,
  Unknown,
  GpuOnly,
  Bounds,
  Kernel,
  CreationFrame,
  DispatchHandle,
  DispatchInputs,
  DispatchParams,
  Dispatches,
  Count,
};
enum class Rejection : uint8_t { Handle, Gap, Storage, Create, Update, Destroy, Dispatch, Count };

struct GameSlot {
  uint32_t generation = 0;
  bool live = false;
  uint32_t persistentBytes = 0;
  uint32_t flags = 0;
  uint32_t createSerial = 0;
  uint32_t kernelCount = 0;
  bool flushOwed = false;
};

struct GameState {
  uint32_t counter = 0;
  bool latched = false;
  uint64_t budget = 0;
  uint64_t alignment = 4;
  uint64_t cost = 0;
  uint64_t commandBytes = 0;
  uint64_t demand = 0;
  uint64_t requested = 0;
  uint64_t shadowBytes = 0;
  uint32_t liveSystems = 0;
  uint32_t dispatches = 0;
  std::array<GameSlot, MaxSystems> slots{};
  std::array<bool, static_cast<size_t>(Refusal::Count)> warned{};
  uint64_t capChannel = MaxFrameChannelBytes;
  uint64_t capCommand = MaxFrameCommandBytes;
  uint32_t capHandles = MaxFrameHandles;
};

struct FifoSlot {
  uint64_t id = 0;
  uint32_t generation = 0;
  bool live = false;
  bool shadowWritten = false;
  bool pendingFlush = false;
  bool dispatched = false;
  bool stateDiverged = false;
  uint32_t flags = 0;
  uint32_t persistentBytes = 0;
  uint32_t createdSerial = 0;
  uint32_t kernelCount = 0;
  std::vector<uint8_t> payload;
  std::vector<uint8_t> shadow;
};

struct FifoState {
  // FIFO thread only; fifo::end_frame resets it while drained
  std::vector<HandleInfo> handles;
  uint32_t serial = 0;
  std::array<FifoSlot, MaxSystems> slots;
  uint32_t liveSystems = 0;
  uint64_t shadowBytes = 0;
  std::array<bool, static_cast<size_t>(Rejection::Count)> warned{};
};

GameState g_game;
FifoState g_fifo;
std::atomic_uint32_t g_serial{1};
std::atomic<const SystemBackend*> g_backend{nullptr};

// Game thread writes these
std::atomic_uint64_t g_refused{0};
std::atomic_uint64_t g_refusedBytes{0};
std::atomic_uint64_t g_staleBinds{0};
std::atomic_uint64_t g_dispatchesRefused{0};
// FIFO thread writes these
std::atomic_uint64_t g_uploads{0};
std::atomic_uint64_t g_uploadBytes{0};
std::atomic_uint64_t g_reserves{0};
std::atomic_uint64_t g_reserveBytes{0};
std::atomic_uint64_t g_failedHandles{0};
std::atomic_uint64_t g_storageMisses{0};
std::atomic_uint64_t g_rejected{0};
std::atomic_uint64_t g_binds{0};
std::atomic_uint64_t g_invalidBinds{0};
std::atomic_uint64_t g_suppressedDraws{0};
std::atomic_uint64_t g_created{0};
std::atomic_uint64_t g_destroyed{0};
std::atomic_uint64_t g_updates{0};
std::atomic_uint64_t g_updateBytes{0};
std::atomic_uint64_t g_dispatches{0};
std::atomic_uint64_t g_dispatchesRejected{0};
std::atomic_uint64_t g_persistentCopies{0};
std::atomic_uint32_t g_liveSystems{0};

inline void bump(std::atomic_uint64_t& counter, uint64_t n = 1) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

template <typename Message>
void refuse(Refusal kind, uint64_t bytes, Message&& message) noexcept {
  bump(g_refused);
  bump(g_refusedBytes, bytes);
  if (!std::exchange(g_game.warned[static_cast<size_t>(kind)], true)) {
    Log.warn("GPU channel: {}; refused", message());
  }
}

template <typename Message>
void warn_game(Refusal kind, Message&& message) noexcept {
  if (!std::exchange(g_game.warned[static_cast<size_t>(kind)], true)) {
    Log.warn("GPU channel: {}", message());
  }
}

template <typename Message>
void reject(Rejection kind, Message&& message) noexcept {
  bump(g_rejected);
  if (!std::exchange(g_fifo.warned[static_cast<size_t>(kind)], true)) {
    Log.warn("GPU channel: {}; rejected", message());
  }
}

uint32_t handle_serial(uint64_t handle) noexcept { return static_cast<uint32_t>(handle >> 32); }
uint32_t handle_index(uint64_t handle) noexcept { return static_cast<uint32_t>(handle); }

uint64_t cost_of(uint64_t bytes) noexcept { return AURORA_ALIGN(bytes, g_game.alignment) + g_game.alignment; }

void latch_budget() noexcept {
  if (g_game.latched) {
    return;
  }
  const gfx::detail::ChannelBudget budget = gfx::detail::channel_budget();
  g_game.budget = std::min(budget.bytes, g_game.capChannel);
  g_game.alignment = std::max<uint64_t>(budget.alignment, 4);
  g_game.latched = true;
}

template <size_t N, typename Fill>
void write_command(uint16_t id, uint32_t payloadSize, Fill&& fill) noexcept {
  std::array<uint8_t, HeaderSize + N> buffer{};
  hints::wire::Writer out{buffer};
  out.header(id, payloadSize);
  fill(out);
  fifo::write_data(buffer.data(), static_cast<uint32_t>(buffer.size()));
}

bool budget_allows(uint64_t bytes, std::string_view what, uint64_t flushBytes = 0) noexcept {
  latch_budget();
  const uint64_t cost = (bytes != 0 ? cost_of(bytes) : 0) + (flushBytes != 0 ? cost_of(flushBytes) : 0);
  if (g_game.cost + cost > g_game.budget) {
    g_game.demand += cost;
    refuse(Refusal::Budget, bytes, [&] {
      return fmt::format("{} of {} bytes needs {} of a {}-byte frame budget with {} used", what, bytes, cost,
                         g_game.budget, g_game.cost);
    });
    return false;
  }
  return true;
}

void charge(uint64_t bytes, uint64_t commandBytes) noexcept {
  const uint64_t cost = cost_of(bytes);
  g_game.cost += cost;
  g_game.demand += cost;
  g_game.commandBytes += commandBytes;
}

uint64_t owed_flush(const GameSlot& slot) noexcept {
  return slot.flushOwed && current_serial() != slot.createSerial ? slot.persistentBytes : 0;
}

void settle_flush(GameSlot& slot, uint64_t flushBytes) noexcept {
  if (flushBytes != 0) {
    charge(flushBytes, 0);
    slot.flushOwed = false;
  }
}

bool commands_allow(uint64_t bytes, std::string_view what) noexcept {
  if (g_game.commandBytes + bytes > g_game.capCommand) {
    refuse(Refusal::Commands, bytes,
           [&] { return fmt::format("{} would take frame command bytes past {}", what, g_game.capCommand); });
    return false;
  }
  return true;
}

bool handle_available(uint64_t bytes) noexcept {
  latch_budget();
  if (g_game.counter >= g_game.capHandles) {
    refuse(Refusal::Handles, bytes, [] { return fmt::format("more than {} handles this frame", MaxFrameHandles); });
    return false;
  }
  return true;
}

bool outside_display_list(uint64_t bytes, std::string_view what) noexcept {
  if (fifo::in_display_list()) {
    refuse(Refusal::DisplayList, bytes, [&] { return fmt::format("{} inside a display list", what); });
    return false;
  }
  return true;
}

GameSlot* live_slot(SystemId id) noexcept {
  const uint32_t index = static_cast<uint32_t>(id) - 1;
  if (static_cast<uint32_t>(id) == 0 || index >= MaxSystems) {
    return nullptr;
  }
  GameSlot& slot = g_game.slots[index];
  return slot.live && slot.generation == static_cast<uint32_t>(id >> 32) ? &slot : nullptr;
}

FifoSlot* live_fifo_slot(uint64_t id) noexcept {
  const uint32_t index = static_cast<uint32_t>(id) - 1;
  if (static_cast<uint32_t>(id) == 0 || index >= MaxSystems) {
    return nullptr;
  }
  FifoSlot& slot = g_fifo.slots[index];
  return slot.live && slot.id == id ? &slot : nullptr;
}

std::string_view desc_problem(const SystemDesc& desc) noexcept {
  if (!valid_system_name(desc.name)) {
    return "the name needs 1 to 31 letters, digits, '_', '.' or '-'";
  }
  if (desc.wgsl.empty() == desc.library.empty()) {
    return "exactly one of wgsl and library must be set";
  }
  const std::string_view source = desc.wgsl.empty() ? desc.library : desc.wgsl;
  if (source.size() > MaxSystemSourceBytes) {
    return "the source is larger than 256 KiB";
  }
  if (!desc.library.empty() && !desc.library.starts_with(LibraryPrefix)) {
    return "a library source must be an embedded systems/ id";
  }
  if (desc.kernels.empty() || desc.kernels.size() > MaxKernels) {
    return "a system needs 1 to 8 kernels";
  }
  if ((desc.flags & ~KnownSystemFlags) != 0 || (desc.requirements & ~KnownRequirements) != 0) {
    return "unknown flag or requirement bits";
  }
  if (desc.persistentBytes % 4 != 0) {
    return "persistentBytes must be a multiple of 4";
  }
  if ((desc.flags & SystemGpuOnlyState) != 0) {
    if (desc.persistentBytes == 0 || desc.persistentBytes > MaxGpuOnlyStateBytes) {
      return "GPU-only state needs 1 to 256 MiB";
    }
  } else if (desc.persistentBytes > MaxPersistentBytes) {
    return "shadowed state is larger than 16 MiB";
  }
  for (const KernelDesc& kernel : desc.kernels) {
    if (!valid_entry_point(kernel.entryPoint)) {
      return "an entry point needs 1 to 63 identifier characters";
    }
    const auto& size = kernel.workgroupSize;
    if (!valid_workgroup(size[0], size[1], size[2])) {
      return "a workgroup needs every dimension at least 1 and at most 128 invocations";
    }
    if ((kernel.flags & ~KnownKernelFlags) != 0) {
      return "unknown kernel flag bits";
    }
    if ((kernel.flags & KernelStateAtomic) != 0 && desc.persistentBytes == 0) {
      return "atomic state needs persistentBytes";
    }
  }
  return {};
}

HandleInfo* accept_handle(uint64_t handle) noexcept {
  const uint32_t serial = handle_serial(handle);
  const uint32_t index = handle_index(handle);
  if (serial == 0 || index == 0 || index > MaxFrameHandles) {
    reject(Rejection::Handle, [&] { return fmt::format("handle 0x{:016X} is malformed", handle); });
    return nullptr;
  }
  if (g_fifo.serial == 0) {
    g_fifo.serial = serial;
  } else if (serial != g_fifo.serial) {
    reject(Rejection::Handle,
           [&] { return fmt::format("handle 0x{:016X} belongs to frame serial {}", handle, g_fifo.serial); });
    return nullptr;
  }
  auto& handles = g_fifo.handles;
  const size_t size = handles.size();
  if (index <= size) {
    reject(Rejection::Handle, [&] { return fmt::format("handle 0x{:016X} arrived twice", handle); });
    return nullptr;
  }
  const size_t gap = index - size - 1;
  if (gap > MaxGap) {
    reject(Rejection::Handle, [&] { return fmt::format("handle 0x{:016X} skips {} handles", handle, gap); });
    return nullptr;
  }
  if (gap != 0) {
    handles.resize(size + gap, HandleInfo{.state = HandleState::Failed});
    bump(g_failedHandles, gap);
    if (!std::exchange(g_fifo.warned[static_cast<size_t>(Rejection::Gap)], true)) {
      Log.warn("GPU channel: handle 0x{:016X} skips {} handles; they fail", handle, gap);
    }
  }
  return &handles.emplace_back();
}

void storage_missed(HandleInfo& info, uint64_t bytes) noexcept {
  info.state = HandleState::Failed;
  bump(g_failedHandles);
  bump(g_storageMisses);
  if (!std::exchange(g_fifo.warned[static_cast<size_t>(Rejection::Storage)], true)) {
    Log.warn("GPU channel: {} bytes do not fit this frame's streaming storage; the handle fails", bytes);
  }
}

void append_header(std::vector<uint8_t>& out, uint16_t id, uint32_t payloadSize) {
  std::array<uint8_t, HeaderSize> header{};
  hints::wire::Writer writer{header};
  writer.header(id, payloadSize);
  out.insert(out.end(), header.begin(), header.end());
}

void clear_fifo_registry() noexcept {
  for (FifoSlot& slot : g_fifo.slots) {
    slot = {};
  }
  g_fifo.liveSystems = 0;
  g_fifo.shadowBytes = 0;
  g_liveSystems.store(0, std::memory_order_relaxed);
}

Handle game_upload(const void* data, uint32_t bytes) noexcept {
  if (!outside_display_list(bytes, "upload")) {
    return InvalidHandle;
  }
  if (data == nullptr || bytes == 0 || bytes > MaxUploadBytes) {
    refuse(Refusal::Size, bytes,
           [&] { return fmt::format("upload of {} bytes is outside 1 to {}", bytes, MaxUploadBytes); });
    return InvalidHandle;
  }
  if (!handle_available(bytes) || !budget_allows(bytes, "upload") || !commands_allow(bytes, "upload")) {
    return InvalidHandle;
  }
  charge(bytes, bytes);
  const Handle handle = uint64_t{g_serial.load(std::memory_order_relaxed)} << 32 | ++g_game.counter;
  write_command<hints::wire::GpuUploadHeader>(GX_AURORA_GPU_UPLOAD, hints::wire::GpuUploadHeader + bytes,
                                              [&](hints::wire::Writer& out) {
                                                out.u16(AURORA_HINT_GPU_UPLOAD_VERSION);
                                                out.u64(handle);
                                              });
  fifo::write_data(data, bytes);
  return handle;
}

Handle game_reserve(uint32_t bytes, uint32_t flags) noexcept {
  if (!outside_display_list(bytes, "reserve")) {
    return InvalidHandle;
  }
  if (bytes == 0 || bytes > MaxUploadBytes || (flags & ~ReserveZero) != 0) {
    refuse(Refusal::Size, bytes, [&] {
      return fmt::format("reserve of {} bytes with flags 0x{:X} is outside 1 to {} or uses unknown flags", bytes, flags,
                         MaxUploadBytes);
    });
    return InvalidHandle;
  }
  if (!handle_available(bytes) || !budget_allows(bytes, "reserve")) {
    return InvalidHandle;
  }
  charge(bytes, 0);
  const Handle handle = uint64_t{g_serial.load(std::memory_order_relaxed)} << 32 | ++g_game.counter;
  write_command<hints::wire::GpuReserveV1>(GX_AURORA_GPU_RESERVE, hints::wire::GpuReserveV1,
                                           [&](hints::wire::Writer& out) {
                                             out.u16(AURORA_HINT_GPU_RESERVE_VERSION);
                                             out.u64(handle);
                                             out.u32(bytes);
                                             out.u32(flags);
                                           });
  return handle;
}

SystemId game_create(const SystemDesc& desc) noexcept {
  if (!outside_display_list(0, "create_system")) {
    return InvalidSystem;
  }
  if (g_backend.load(std::memory_order_acquire) == nullptr) {
    refuse(Refusal::Systems, 0,
           [] { return std::string{"create_system needs the render module, which is off or uninitialised"}; });
    return InvalidSystem;
  }
  if (const std::string_view problem = desc_problem(desc); !problem.empty()) {
    refuse(Refusal::Desc, 0, [&] { return fmt::format("system '{}': {}", desc.name, problem); });
    return InvalidSystem;
  }
  const bool gpuOnly = (desc.flags & SystemGpuOnlyState) != 0;
  const uint64_t shadowed = gpuOnly ? 0 : desc.persistentBytes;
  if (g_game.shadowBytes + shadowed > MaxShadowBytes) {
    refuse(Refusal::ShadowCap, 0,
           [&] { return fmt::format("system '{}' would take shadowed state past 64 MiB", desc.name); });
    return InvalidSystem;
  }
  const auto freeSlot = std::ranges::find_if(g_game.slots, [](const GameSlot& slot) { return !slot.live; });
  if (freeSlot == g_game.slots.end()) {
    refuse(Refusal::SystemCap, 0, [] { return fmt::format("more than {} live systems", MaxSystems); });
    return InvalidSystem;
  }
  const bool library = !desc.library.empty();
  const std::string_view source = library ? desc.library : desc.wgsl;
  uint64_t size = hints::wire::GpuCreateHeader + desc.name.size() + source.size();
  for (const KernelDesc& kernel : desc.kernels) {
    size += hints::wire::GpuCreateKernelHeader + kernel.entryPoint.size();
  }
  if (!commands_allow(size, "create_system")) {
    return InvalidSystem;
  }
  GameSlot& slot = *freeSlot;
  slot.generation = slot.generation == UINT32_MAX ? 1 : slot.generation + 1;
  slot.live = true;
  slot.persistentBytes = desc.persistentBytes;
  slot.flags = desc.flags;
  slot.createSerial = current_serial();
  slot.kernelCount = static_cast<uint32_t>(desc.kernels.size());
  slot.flushOwed = false;
  ++g_game.liveSystems;
  g_game.shadowBytes += shadowed;
  g_game.commandBytes += size;
  const SystemId id = uint64_t{slot.generation} << 32 | (static_cast<uint32_t>(freeSlot - g_game.slots.begin()) + 1);

  thread_local std::vector<uint8_t> buffer;
  buffer.resize(HeaderSize + size);
  hints::wire::Writer out{buffer};
  out.header(GX_AURORA_GPU_CREATE_SYSTEM, static_cast<uint32_t>(size));
  out.u16(AURORA_HINT_GPU_CREATE_SYSTEM_VERSION);
  out.u64(id);
  out.u32(desc.persistentBytes);
  out.u32(desc.flags | (library ? hints::wire::CreateLibrarySource : 0u));
  out.u32(desc.requirements);
  out.u8(static_cast<uint8_t>(desc.name.size()));
  out.u8(static_cast<uint8_t>(desc.kernels.size()));
  out.u32(static_cast<uint32_t>(source.size()));
  const auto bytes_of = [](std::string_view text) {
    return std::span<const uint8_t>{reinterpret_cast<const uint8_t*>(text.data()), text.size()};
  };
  out.bytes(bytes_of(desc.name));
  for (const KernelDesc& kernel : desc.kernels) {
    out.u8(static_cast<uint8_t>(kernel.entryPoint.size()));
    out.bytes(bytes_of(kernel.entryPoint));
    for (const uint32_t dimension : kernel.workgroupSize) {
      out.u16(static_cast<uint16_t>(dimension));
    }
    out.u8(static_cast<uint8_t>(kernel.flags));
  }
  out.bytes(bytes_of(source));
  fifo::write_data(buffer.data(), static_cast<uint32_t>(buffer.size()));
  return id;
}

void game_update(SystemId id, uint32_t byteOffset, const void* data, uint32_t bytes) noexcept {
  if (!outside_display_list(bytes, "update_persistent")) {
    return;
  }
  GameSlot* slot = live_slot(id);
  if (slot == nullptr) {
    refuse(Refusal::Unknown, bytes, [&] { return fmt::format("system 0x{:016X} is not live", id); });
    return;
  }
  if ((slot->flags & SystemGpuOnlyState) != 0) {
    refuse(Refusal::GpuOnly, bytes, [&] { return fmt::format("system 0x{:016X} keeps its state on the GPU", id); });
    return;
  }
  if (data == nullptr || bytes < 4 || bytes % 4 != 0 || byteOffset % 4 != 0 ||
      uint64_t{byteOffset} + bytes > slot->persistentBytes) {
    refuse(Refusal::Bounds, bytes, [&] {
      return fmt::format("update of {} bytes at {} does not fit the {}-byte state of system 0x{:016X} in words", bytes,
                         byteOffset, slot->persistentBytes, id);
    });
    return;
  }
  const uint64_t flush = owed_flush(*slot);
  if (!budget_allows(bytes, "update_persistent", flush) || !commands_allow(bytes, "update_persistent")) {
    return;
  }
  settle_flush(*slot, flush);
  charge(bytes, bytes);
  if (current_serial() == slot->createSerial) {
    slot->flushOwed = true;
  }
  write_command<hints::wire::GpuUpdateHeader>(GX_AURORA_GPU_UPDATE_PERSISTENT, hints::wire::GpuUpdateHeader + bytes,
                                              [&](hints::wire::Writer& out) {
                                                out.u16(AURORA_HINT_GPU_UPDATE_PERSISTENT_VERSION);
                                                out.u64(id);
                                                out.u32(byteOffset);
                                              });
  fifo::write_data(data, bytes);
}

void game_destroy(SystemId id) noexcept {
  if (!outside_display_list(0, "destroy_system")) {
    return;
  }
  GameSlot* slot = live_slot(id);
  if (slot == nullptr) {
    refuse(Refusal::Unknown, 0, [&] { return fmt::format("system 0x{:016X} is not live", id); });
    return;
  }
  write_command<hints::wire::GpuDestroyV1>(GX_AURORA_GPU_DESTROY_SYSTEM, hints::wire::GpuDestroyV1,
                                           [&](hints::wire::Writer& out) {
                                             out.u16(AURORA_HINT_GPU_DESTROY_SYSTEM_VERSION);
                                             out.u64(id);
                                           });
  slot->live = false;
  --g_game.liveSystems;
  g_game.shadowBytes -= (slot->flags & SystemGpuOnlyState) != 0 ? 0 : slot->persistentBytes;
}

SystemStatus game_status(SystemId id) noexcept {
  const GameSlot* slot = live_slot(id);
  if (slot == nullptr) {
    return SystemStatus::Invalid;
  }
  const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
  if (backend == nullptr || backend->status == nullptr) {
    return SystemStatus::Unavailable;
  }
  const SystemStatus status = backend->status(backend->userData, id);
  if (status == SystemStatus::Ready && slot->persistentBytes != 0 && current_serial() == slot->createSerial) {
    return SystemStatus::Pending;
  }
  return status;
}

template <typename Message>
bool refuse_dispatch(Refusal kind, Message&& message) noexcept {
  bump(g_dispatchesRefused);
  refuse(kind, 0, std::forward<Message>(message));
  return false;
}

bool game_dispatch(SystemId system, uint32_t kernel, std::span<const Handle> inputs, Handle output,
                   std::array<uint32_t, 3> elements, std::span<const uint8_t> params) noexcept {
  if (g_backend.load(std::memory_order_acquire) == nullptr) {
    return refuse_dispatch(Refusal::Systems, [] {
      return std::string{"dispatch needs the render module, which is off or uninitialised"};
    });
  }
  if (fifo::in_display_list()) {
    return refuse_dispatch(Refusal::DisplayList, [] { return std::string{"dispatch inside a display list"}; });
  }
  GameSlot* slot = live_slot(system);
  if (slot == nullptr || kernel >= slot->kernelCount) {
    return refuse_dispatch(Refusal::Kernel, [&] {
      return fmt::format("dispatch of kernel {} names system 0x{:016X}, which is not live or lacks it", kernel, system);
    });
  }
  if (slot->persistentBytes != 0 && current_serial() == slot->createSerial) {
    return refuse_dispatch(Refusal::CreationFrame, [&] {
      return fmt::format("system 0x{:016X} keeps state, which exists only from the frame after its creation", system);
    });
  }
  const bool outputOk = output == InvalidHandle ? slot->persistentBytes != 0 : handle_is_current(output);
  if (!outputOk || !std::ranges::all_of(inputs, [](Handle input) { return handle_is_current(input); })) {
    return refuse_dispatch(Refusal::DispatchHandle, [&] {
      return fmt::format(
          "dispatch of system 0x{:016X} names a handle that is not from this frame, or no output "
          "for a system without state",
          system);
    });
  }
  if (inputs.size() > MaxDispatchInputs) {
    return refuse_dispatch(Refusal::DispatchInputs, [&] {
      return fmt::format("dispatch with {} inputs is past the limit of {}", inputs.size(), MaxDispatchInputs);
    });
  }
  if (params.size() > MaxDispatchParamsSize) {
    return refuse_dispatch(Refusal::DispatchParams, [&] {
      return fmt::format("dispatch with {} parameter bytes is past the limit of {}", params.size(),
                         MaxDispatchParamsSize);
    });
  }
  if (g_game.dispatches >= MaxFrameDispatches) {
    return refuse_dispatch(Refusal::Dispatches,
                           [] { return fmt::format("more than {} dispatches this frame", MaxFrameDispatches); });
  }
  const auto paramWords = static_cast<uint32_t>((params.size() + 3) / 4);
  const auto size =
      static_cast<uint32_t>(hints::wire::GpuDispatchHeaderV1 + inputs.size() * sizeof(uint64_t) + paramWords * 4);
  const uint64_t flush = owed_flush(*slot);
  if (!commands_allow(size, "dispatch") || (flush != 0 && !budget_allows(0, "the owed state flush", flush))) {
    bump(g_dispatchesRefused);
    return false;
  }
  settle_flush(*slot, flush);
  g_game.commandBytes += size;
  ++g_game.dispatches;

  constexpr size_t MaxSize =
      HeaderSize + hints::wire::GpuDispatchHeaderV1 + MaxDispatchInputs * sizeof(uint64_t) + MaxDispatchParamsSize;
  std::array<uint8_t, MaxSize> buffer{};
  hints::wire::Writer out{std::span{buffer.data(), HeaderSize + size}};
  out.header(GX_AURORA_GPU_DISPATCH, size);
  out.u16(AURORA_HINT_GPU_DISPATCH_VERSION);
  out.u64(system);
  out.u32(kernel);
  out.u64(output);
  for (const uint32_t count : elements) {
    out.u32(count);
  }
  out.u8(static_cast<uint8_t>(inputs.size()));
  out.u8(static_cast<uint8_t>(paramWords));
  for (const Handle input : inputs) {
    out.u64(input);
  }
  out.bytes(params);
  fifo::write_data(buffer.data(), HeaderSize + size);
  return true;
}
} // namespace

bool valid_system_name(std::string_view name) noexcept {
  if (name.empty() || name.size() > MaxSystemNameBytes) {
    return false;
  }
  return std::ranges::all_of(name, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
           c == '-';
  });
}

bool valid_entry_point(std::string_view entry) noexcept {
  if (entry.empty() || entry.size() > MaxKernelNameBytes || (entry[0] >= '0' && entry[0] <= '9')) {
    return false;
  }
  return std::ranges::all_of(entry, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  });
}

bool valid_workgroup(uint32_t x, uint32_t y, uint32_t z) noexcept {
  return x >= 1 && y >= 1 && z >= 1 && uint64_t{x} * y * z <= MaxWorkgroupInvocations;
}

void set_system_backend(const SystemBackend* backend) noexcept { g_backend.store(backend, std::memory_order_release); }

bool handle_is_current(uint64_t handle) noexcept {
  const uint32_t index = handle_index(handle);
  return handle_serial(handle) == g_serial.load(std::memory_order_relaxed) && index >= 1 && index <= g_game.counter;
}

uint64_t bind_handle(uint64_t handle) noexcept {
  if (fifo::in_display_list()) {
    if (handle != InvalidHandle) {
      warn_game(Refusal::ListBind, [] {
        return std::string{"GXSetArrayGPU inside a display list binds handle 0, so the list's draws are suppressed"};
      });
    }
    return InvalidHandle;
  }
  if (handle == InvalidHandle || handle_is_current(handle)) {
    return handle;
  }
  bump(g_staleBinds);
  warn_game(Refusal::Stale, [&] {
    return fmt::format("GXSetArrayGPU got handle 0x{:016X}, which is not from this frame; bound as 0", handle);
  });
  return InvalidHandle;
}

void warn_bad_attribute(uint32_t attr) noexcept {
  warn_game(Refusal::Attribute,
            [&] { return fmt::format("GXSetArrayGPU attribute {} has no vertex array; nothing sent", attr); });
}

uint32_t current_serial() noexcept { return g_serial.load(std::memory_order_relaxed); }

bool on_upload(uint64_t handle, std::span<const uint8_t> bytes) noexcept {
  HandleInfo* info = accept_handle(handle);
  if (info == nullptr) {
    return false;
  }
  bump(g_uploads);
  bump(g_uploadBytes, bytes.size());
  const gfx::Range range = gfx::push_channel_storage(bytes);
  if (range.size == 0) {
    storage_missed(*info, bytes.size());
    return false;
  }
  *info = {.range = range, .state = HandleState::Uploaded};
  return true;
}

bool on_reserve(uint64_t handle, uint32_t bytes, uint32_t flags) noexcept {
  HandleInfo* info = accept_handle(handle);
  if (info == nullptr) {
    return false;
  }
  bump(g_reserves);
  bump(g_reserveBytes, bytes);
  const bool zero = (flags & ReserveZero) != 0;
  const gfx::Range range = gfx::reserve_channel_storage(bytes, zero);
  if (range.size == 0) {
    storage_missed(*info, bytes);
    return false;
  }
  *info = {.range = range, .state = zero ? HandleState::Written : HandleState::Reserved};
  return true;
}

bool on_create(const SystemCreate& desc, std::span<const uint8_t> payload) noexcept {
  const uint32_t index = static_cast<uint32_t>(desc.id) - 1;
  const auto generation = static_cast<uint32_t>(desc.id >> 32);
  if (static_cast<uint32_t>(desc.id) == 0 || index >= MaxSystems || generation == 0) {
    reject(Rejection::Create, [&] { return fmt::format("system id 0x{:016X} is malformed", desc.id); });
    return false;
  }
  FifoSlot& slot = g_fifo.slots[index];
  const bool gpuOnly = (desc.flags & SystemGpuOnlyState) != 0;
  const uint64_t shadowed = gpuOnly ? 0 : desc.persistentBytes;
  if (slot.live || generation <= slot.generation || g_fifo.liveSystems >= MaxSystems ||
      g_fifo.shadowBytes + shadowed > MaxShadowBytes) {
    reject(Rejection::Create, [&] {
      return fmt::format("system 0x{:016X} reuses a live or newer slot, or passes the system or shadow caps", desc.id);
    });
    return false;
  }
  slot.id = desc.id;
  slot.generation = generation;
  slot.live = true;
  slot.shadowWritten = false;
  slot.pendingFlush = false;
  slot.dispatched = false;
  slot.stateDiverged = false;
  slot.flags = desc.flags;
  slot.persistentBytes = desc.persistentBytes;
  slot.createdSerial = current_serial();
  slot.kernelCount = static_cast<uint32_t>(desc.kernels.size());
  slot.payload.assign(payload.begin(), payload.end());
  slot.shadow.assign(shadowed, 0);
  ++g_fifo.liveSystems;
  g_fifo.shadowBytes += shadowed;
  g_liveSystems.store(g_fifo.liveSystems, std::memory_order_relaxed);
  bump(g_created);
  if (const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
      backend != nullptr && backend->create != nullptr) {
    backend->create(backend->userData, desc);
  }
  return true;
}

bool on_update(uint64_t id, uint32_t byteOffset, std::span<const uint8_t> bytes) noexcept {
  FifoSlot* slot = live_fifo_slot(id);
  if (slot == nullptr || (slot->flags & SystemGpuOnlyState) != 0 ||
      uint64_t{byteOffset} + bytes.size() > slot->shadow.size()) {
    reject(Rejection::Update, [&] {
      return fmt::format("update of system 0x{:016X} at {} for {} bytes has no shadow to write", id, byteOffset,
                         bytes.size());
    });
    return false;
  }
  std::memcpy(slot->shadow.data() + byteOffset, bytes.data(), bytes.size());
  slot->shadowWritten = true;
  bump(g_updates);
  bump(g_updateBytes, bytes.size());
  const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
  if (backend == nullptr || backend->update == nullptr || slot->persistentBytes == 0) {
    return true;
  }
  if (current_serial() == slot->createdSerial) {
    slot->pendingFlush = true;
    return true;
  }
  bool applied = true;
  if (std::exchange(slot->pendingFlush, false)) {
    applied = backend->update(backend->userData, id, 0, slot->shadow);
    bump(applied ? g_persistentCopies : g_storageMisses);
  }
  if (applied) {
    applied = backend->update(backend->userData, id, byteOffset, bytes);
    bump(applied ? g_persistentCopies : g_storageMisses);
  }
  if (!applied) {
    slot->stateDiverged = true;
  }
  return true;
}

bool on_destroy(uint64_t id) noexcept {
  FifoSlot* slot = live_fifo_slot(id);
  if (slot == nullptr) {
    reject(Rejection::Destroy, [&] { return fmt::format("system 0x{:016X} is not live", id); });
    return false;
  }
  g_fifo.shadowBytes -= slot->shadow.size();
  slot->live = false;
  slot->shadowWritten = false;
  slot->pendingFlush = false;
  slot->dispatched = false;
  slot->stateDiverged = false;
  slot->payload.clear();
  slot->shadow.clear();
  --g_fifo.liveSystems;
  g_liveSystems.store(g_fifo.liveSystems, std::memory_order_relaxed);
  bump(g_destroyed);
  if (const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
      backend != nullptr && backend->destroy != nullptr) {
    backend->destroy(backend->userData, id);
  }
  return true;
}

bool on_dispatch(uint64_t system, uint32_t kernel, uint64_t output, std::span<const uint64_t> inputs,
                 std::array<uint32_t, 3> elements, std::span<const uint8_t> params) noexcept {
  const auto fail = [&](auto&& message) {
    bump(g_dispatchesRejected);
    if (output != InvalidHandle) {
      mark_failed(output);
    }
    if (!std::exchange(g_fifo.warned[static_cast<size_t>(Rejection::Dispatch)], true)) {
      Log.warn("GPU channel: {}; the dispatch is rejected and its output fails", message());
    }
    return false;
  };
  FifoSlot* slot = live_fifo_slot(system);
  if (slot == nullptr || kernel >= slot->kernelCount) {
    return fail([&] { return fmt::format("system 0x{:016X} is not live or has no kernel {}", system, kernel); });
  }
  if (slot->persistentBytes != 0 && current_serial() == slot->createdSerial) {
    return fail([&] { return fmt::format("system 0x{:016X} keeps state and was created this frame", system); });
  }
  const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
  // The GX thread charged the owed flush once these checks passed, so later rejections must not leave it owed
  if (std::exchange(slot->pendingFlush, false)) {
    if (backend == nullptr || backend->update == nullptr ||
        !backend->update(backend->userData, system, 0, slot->shadow)) {
      bump(g_storageMisses);
      slot->stateDiverged = true;
      return fail([&] { return fmt::format("the state of system 0x{:016X} could not be staged", system); });
    }
    bump(g_persistentCopies);
  }
  const HandleInfo target = output != InvalidHandle ? lookup(output) : HandleInfo{};
  if (output != InvalidHandle && target.state != HandleState::Reserved) {
    return fail([&] { return fmt::format("output 0x{:016X} is not a reserve awaiting its writer", output); });
  }
  std::array<gfx::Range, MaxDispatchInputs> ranges{};
  if (inputs.size() > ranges.size()) {
    return fail([&] { return fmt::format("{} inputs are past the limit", inputs.size()); });
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const HandleInfo info = lookup(inputs[i]);
    if (info.state != HandleState::Uploaded && info.state != HandleState::Written) {
      return fail([&] { return fmt::format("input 0x{:016X} has no data", inputs[i]); });
    }
    ranges[i] = {info.range.offset, AURORA_ALIGN(info.range.size, 4u)};
  }
  if (backend == nullptr || backend->dispatch == nullptr) {
    return fail([] { return std::string{"no dispatch backend is installed"}; });
  }
  const DispatchDecode desc{
      .system = system,
      .kernel = kernel,
      .output = target.range,
      .inputs = std::span{ranges.data(), inputs.size()},
      .elements = elements,
      .params = params,
  };
  if (!backend->dispatch(backend->userData, desc)) {
    return fail([&] { return fmt::format("system 0x{:016X} could not record kernel {}", system, kernel); });
  }
  if (output != InvalidHandle) {
    mark_written(output);
  }
  if (slot->persistentBytes != 0) {
    slot->dispatched = true;
  }
  bump(g_dispatches);
  return true;
}

HandleInfo lookup(uint64_t handle) noexcept {
  const uint32_t index = handle_index(handle);
  if (index == 0 || handle_serial(handle) != g_fifo.serial || index > g_fifo.handles.size()) {
    return {};
  }
  return g_fifo.handles[index - 1];
}

bool mark_written(uint64_t handle) noexcept {
  const uint32_t index = handle_index(handle);
  if (index == 0 || handle_serial(handle) != g_fifo.serial || index > g_fifo.handles.size()) {
    return false;
  }
  HandleInfo& info = g_fifo.handles[index - 1];
  if (info.state != HandleState::Reserved) {
    return false;
  }
  info.state = HandleState::Written;
  return true;
}

bool mark_failed(uint64_t handle) noexcept {
  const uint32_t index = handle_index(handle);
  if (index == 0 || handle_serial(handle) != g_fifo.serial || index > g_fifo.handles.size()) {
    return false;
  }
  g_fifo.handles[index - 1].state = HandleState::Failed;
  return true;
}

std::span<const uint8_t> shadow(uint64_t id) noexcept {
  const FifoSlot* slot = live_fifo_slot(id);
  return slot != nullptr ? std::span<const uint8_t>{slot->shadow} : std::span<const uint8_t>{};
}

void note_bind(bool valid) noexcept { bump(valid ? g_binds : g_invalidBinds); }

void note_suppressed_draw() noexcept { bump(g_suppressedDraws); }

void end_frame() noexcept {
  if (const uint64_t demand = std::max(g_game.demand, g_game.requested); demand != 0) {
    gfx::detail::request_channel_storage(demand);
  }
  g_game.counter = 0;
  g_game.latched = false;
  g_game.cost = 0;
  g_game.commandBytes = 0;
  g_game.demand = 0;
  g_game.dispatches = 0;
  uint32_t serial = g_serial.load(std::memory_order_relaxed) + 1;
  if (serial == 0) {
    serial = 1;
  }
  g_serial.store(serial, std::memory_order_relaxed);
  g_fifo.handles.clear();
  g_fifo.serial = 0;
}

void reset() noexcept {
  g_game = {};
  g_serial.store(1, std::memory_order_relaxed);
  reset_decoder();
}

void reset_decoder() noexcept {
  g_fifo.handles.clear();
  g_fifo.serial = 0;
  g_fifo.warned = {};
  clear_fifo_registry();
  if (const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
      backend != nullptr && backend->reset != nullptr) {
    backend->reset(backend->userData);
  }
}

void wait_systems() noexcept {
  if (const SystemBackend* backend = g_backend.load(std::memory_order_acquire);
      backend != nullptr && backend->wait_all != nullptr) {
    backend->wait_all(backend->userData);
  }
}

void write_preamble(std::vector<uint8_t>& out) {
  for (const FifoSlot& slot : g_fifo.slots) {
    if (!slot.live) {
      continue;
    }
    append_header(out, GX_AURORA_GPU_CREATE_SYSTEM, static_cast<uint32_t>(slot.payload.size()));
    out.insert(out.end(), slot.payload.begin(), slot.payload.end());
    if (!slot.shadowWritten) {
      continue;
    }
    append_header(out, GX_AURORA_GPU_UPDATE_PERSISTENT,
                  hints::wire::GpuUpdateHeader + static_cast<uint32_t>(slot.shadow.size()));
    std::array<uint8_t, hints::wire::GpuUpdateHeader> header{};
    hints::wire::Writer writer{header};
    writer.u16(AURORA_HINT_GPU_UPDATE_PERSISTENT_VERSION);
    writer.u64(slot.id);
    writer.u32(0);
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), slot.shadow.begin(), slot.shadow.end());
  }
}

bool uncaptured_gpu_state() noexcept {
  // The worker cannot name the system whose state copy it dropped, so a dropped copy flags every stateful system
  const bool copiesDropped = gfx::detail::compute_failures().copies != 0;
  return std::ranges::any_of(g_fifo.slots, [&](const FifoSlot& slot) {
    return slot.live && ((slot.flags & SystemGpuOnlyState) != 0 || slot.dispatched || slot.stateDiverged ||
                         (copiesDropped && slot.persistentBytes != 0));
  });
}

uint64_t preamble_flush_bytes() noexcept {
  const uint64_t alignment = std::max<uint64_t>(gfx::detail::channel_budget().alignment, 4);
  uint64_t bytes = 0;
  for (const FifoSlot& slot : g_fifo.slots) {
    if (slot.live && slot.shadowWritten) {
      bytes += AURORA_ALIGN(uint64_t{slot.shadow.size()}, alignment) + alignment;
    }
  }
  return bytes;
}

Stats stats() noexcept {
  const auto load = [](const std::atomic_uint64_t& value) { return value.load(std::memory_order_relaxed); };
  return {
      .refused = load(g_refused),
      .refusedBytes = load(g_refusedBytes),
      .staleBinds = load(g_staleBinds),
      .uploads = load(g_uploads),
      .uploadBytes = load(g_uploadBytes),
      .reserves = load(g_reserves),
      .reserveBytes = load(g_reserveBytes),
      .failedHandles = load(g_failedHandles),
      .storageMisses = load(g_storageMisses),
      .rejected = load(g_rejected),
      .binds = load(g_binds),
      .invalidBinds = load(g_invalidBinds),
      .suppressedDraws = load(g_suppressedDraws),
      .created = load(g_created),
      .destroyed = load(g_destroyed),
      .updates = load(g_updates),
      .updateBytes = load(g_updateBytes),
      .liveSystems = g_liveSystems.load(std::memory_order_relaxed),
      .dispatches = load(g_dispatches),
      .dispatchesRefused = load(g_dispatchesRefused),
      .dispatchesRejected = load(g_dispatchesRejected),
      .persistentCopies = load(g_persistentCopies),
      .binderFailures = gfx::detail::compute_failures().binds,
      .droppedCopies = gfx::detail::compute_failures().copies,
  };
}

namespace testing {
void reset_stats() noexcept {
  for (auto* counter :
       {&g_refused,         &g_refusedBytes, &g_staleBinds,      &g_uploads,           &g_uploadBytes,
        &g_reserves,        &g_reserveBytes, &g_failedHandles,   &g_storageMisses,     &g_rejected,
        &g_binds,           &g_invalidBinds, &g_suppressedDraws, &g_created,           &g_destroyed,
        &g_updates,         &g_updateBytes,  &g_dispatches,      &g_dispatchesRefused, &g_dispatchesRejected,
        &g_persistentCopies}) {
    counter->store(0, std::memory_order_relaxed);
  }
  g_game.warned = {};
  g_fifo.warned = {};
}

void set_serial(uint32_t serial) noexcept { g_serial.store(serial, std::memory_order_relaxed); }

void set_frame_caps(uint64_t channelBytes, uint64_t commandBytes, uint32_t handles) noexcept {
  g_game.capChannel = channelBytes;
  g_game.capCommand = commandBytes;
  g_game.capHandles = handles;
  g_game.latched = false;
}
} // namespace testing
} // namespace aurora::gx::gpu_channel

namespace aurora::render::gpu {
bool systems_available() noexcept { return gx::gpu_channel::g_backend.load(std::memory_order_acquire) != nullptr; }

void request_capacity(uint64_t channelBytesPerFrame) noexcept {
  gx::gpu_channel::g_game.requested = channelBytesPerFrame;
}

Handle upload(const void* data, uint32_t bytes) noexcept { return gx::gpu_channel::game_upload(data, bytes); }

Handle reserve(uint32_t bytes, uint32_t flags) noexcept { return gx::gpu_channel::game_reserve(bytes, flags); }

SystemId create_system(const SystemDesc& desc) noexcept { return gx::gpu_channel::game_create(desc); }

void update_persistent(SystemId id, uint32_t byteOffset, const void* data, uint32_t bytes) noexcept {
  gx::gpu_channel::game_update(id, byteOffset, data, bytes);
}

void destroy_system(SystemId id) noexcept { gx::gpu_channel::game_destroy(id); }

SystemStatus system_status(SystemId id) noexcept { return gx::gpu_channel::game_status(id); }

bool dispatch(SystemId system, uint32_t kernel, std::span<const Handle> inputs, Handle output,
              std::array<uint32_t, 3> elements, std::span<const uint8_t> params) noexcept {
  return gx::gpu_channel::game_dispatch(system, kernel, inputs, output, elements, params);
}

Stats stats() noexcept { return gx::gpu_channel::stats(); }
} // namespace aurora::render::gpu

#include "frame.hpp"

#include "depth_peek.hpp"
#include "pipeline_cache.hpp"
#include "recording.hpp"
#include "render_worker.hpp"
#include "resource_cache.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture_replacement.hpp"
#include "../gx/gx.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui/pipeline.hpp"
#endif
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "../render/frame_state.hpp"
#include "../render/module.hpp"
#include "../render/stage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <aurora/gpu_systems.hpp>
#include <SDL3/SDL_stdinc.h>
#include <tracy/Tracy.hpp>
#include <magic_enum.hpp>

namespace aurora::gfx {
using namespace detail;
using webgpu::g_device;
using webgpu::g_instance;
using webgpu::g_queue;

namespace {
constexpr Module Log{"aurora::gfx"};

Resources g_resources;

constexpr uint64_t MiB = 1024 * 1024;
constexpr auto VertexBufferUsage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
// Storage lets later stream-out passes read the u16 indices as u32 words
constexpr auto IndexBufferUsage = wgpu::BufferUsage::Index | wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
constexpr auto StorageBufferUsage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
constexpr auto StagingBufferUsage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;

enum class BufferMapState {
  Unmapped,
  Mapping,
  Mapped,
};

std::array<wgpu::Buffer, StagingBufferCount> g_stagingBuffers;
std::array<std::atomic<BufferMapState>, StagingBufferCount> g_mappingStates;
// Game thread, except that the worker reads a slot's sizes while it maps that slot
std::array<StreamingSizes, StagingBufferCount> g_stagingSizes;
StreamingSizes g_targetSizes;
StreamingSizes g_requestedSizes;
uint32_t g_residentCapacity = 0;
ResidentLayout g_residentLayout{};
uint32_t g_testGrowthFrame = 0;
bool g_loggedStagingCap = false;
uint32_t g_frameIndex = UINT32_MAX;
ChannelBudget g_channelBudget;
uint64_t g_channelDemand = 0;
bool g_warnedChannelBudget = false;
bool g_warnedStorageGrowth = false;

std::array<FramePacket, FrameSlotCount> g_framePackets;
uint64_t g_nextFrameId = 1;
// The packet is reset before the submit callback runs, so the submit hooks keep their own reference
std::shared_ptr<const render::detail::FrameState> g_submitState;
render_worker::FrameSlotPool g_frameSlots{FrameSlotCount};
render_worker::FrameSlotPool g_stagingSlots{StagingBufferCount};

struct RuntimeDrawType {
  std::string label;
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
  uint32_t generation = 1;
};

struct RuntimeEncoderTaskType {
  std::string label;
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
  uint32_t generation = 1;
};

constexpr uint32_t draw_type_index(DrawTypeId id) { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }
constexpr uint32_t draw_type_generation(DrawTypeId id) { return static_cast<uint32_t>(id >> 32); }
constexpr DrawTypeId make_draw_type_id(uint32_t index, uint32_t generation) {
  return static_cast<DrawTypeId>(generation) << 32 | index;
}

std::vector<RuntimeDrawType> g_runtimeDrawTypes;
std::vector<uint32_t> g_freeDrawTypeSlots;
std::vector<RuntimeEncoderTaskType> g_runtimeEncoderTaskTypes;
std::vector<uint32_t> g_freeEncoderTaskTypeSlots;
std::mutex g_runtimeTypeMutex;

// Requires g_runtimeTypeMutex held.
const RuntimeDrawType* find_runtime_draw_type_locked(DrawTypeId id) {
  const auto idx = draw_type_index(id);
  if (id == InvalidDrawType || idx >= g_runtimeDrawTypes.size()) {
    return nullptr;
  }
  const auto& slot = g_runtimeDrawTypes[idx];
  if (slot.generation != draw_type_generation(id) || slot.draw == nullptr) {
    return nullptr;
  }
  return &slot;
}

// Requires g_runtimeTypeMutex held.
const RuntimeEncoderTaskType* find_runtime_encoder_task_type_locked(EncoderTaskId id) {
  const auto idx = draw_type_index(id);
  if (id == InvalidEncoderTask || idx >= g_runtimeEncoderTaskTypes.size()) {
    return nullptr;
  }
  const auto& slot = g_runtimeEncoderTaskTypes[idx];
  if (slot.generation != draw_type_generation(id) || slot.callback == nullptr) {
    return nullptr;
  }
  return &slot;
}

using PresentClock = std::chrono::steady_clock;
constexpr auto PresentFpsWindow = std::chrono::seconds{1};
std::mutex g_presentStatsMutex;
std::deque<PresentClock::time_point> g_presentTimes;
std::atomic_bool g_processEventsQueued = false;
std::atomic_int64_t g_lastPresentNs = 0;
std::atomic_int64_t g_presentPeriodNs = 0;
std::atomic_int64_t g_cpuFrameTimeNs = 0;
PresentClock::time_point g_cpuFrameStart;
constexpr auto FrameStartSafetyMargin = std::chrono::milliseconds{2};
constexpr auto MaxPacingSample = std::chrono::milliseconds{250};
constexpr uint32_t PacingEmaWeight = 8;

int64_t timestamp_ns(PresentClock::time_point time) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

int64_t duration_ns(PresentClock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void update_ema(std::atomic_int64_t& value, int64_t sample) {
  if (sample <= 0 || sample > duration_ns(MaxPacingSample)) {
    return;
  }

  int64_t current = value.load(std::memory_order_acquire);
  while (true) {
    const int64_t next = current == 0 ? sample : current + (sample - current) / static_cast<int64_t>(PacingEmaWeight);
    if (value.compare_exchange_weak(current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }
}

void prune_present_times(PresentClock::time_point now) {
  while (!g_presentTimes.empty() && g_presentTimes.front() + PresentFpsWindow < now) {
    g_presentTimes.pop_front();
  }
}

void process_events() {
  ZoneScopedN("ProcessEvents");
  if (g_instance) {
    g_instance.ProcessEvents();
  }
}

void enqueue_process_events() {
  if (render_worker::is_worker_thread()) {
    process_events();
    return;
  }

  bool expected = false;
  if (!g_processEventsQueued.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  render_worker::enqueue_work([] {
    process_events();
    g_processEventsQueued.store(false, std::memory_order_release);
  });
}

void wait_for_gpu_progress(std::chrono::nanoseconds sleepDuration) {
  if (render_worker::is_idle()) {
    enqueue_process_events();
  }
  std::this_thread::sleep_for(sleepDuration);
}

void pace_frame_start() {
  ZoneScopedN("Frame start pacing");
  if (g_frameSlots.free_count() == FrameSlotCount) {
    return;
  }

  const int64_t lastPresentNs = g_lastPresentNs.load(std::memory_order_acquire);
  const int64_t presentPeriodNs = g_presentPeriodNs.load(std::memory_order_acquire);
  const int64_t cpuFrameTimeNs = g_cpuFrameTimeNs.load(std::memory_order_acquire);
  if (lastPresentNs == 0 || presentPeriodNs == 0 || cpuFrameTimeNs == 0) {
    return;
  }

  const int64_t safetyMarginNs = duration_ns(FrameStartSafetyMargin);
  const int64_t targetStartNs = lastPresentNs + presentPeriodNs - cpuFrameTimeNs - safetyMarginNs;
  int64_t nowNs = timestamp_ns(PresentClock::now());
  if (targetStartNs <= nowNs) {
    return;
  }

  const double initialWaitMs = static_cast<double>(targetStartNs - nowNs) / 1'000'000.0;
  TracyPlot("aurora: frameStartPaceWaitMs", initialWaitMs);
  while (nowNs < targetStartNs) {
    const int64_t remainingNs = targetStartNs - nowNs;
    const auto sleepDuration =
        remainingNs > 1'000'000 ? std::chrono::milliseconds{1} : std::chrono::nanoseconds{remainingNs};
    wait_for_gpu_progress(sleepDuration);
    nowNs = timestamp_ns(PresentClock::now());
  }
}

void map_staging_buffer(size_t slot, bool releaseSlotOnCompletion = false) {
  auto expected = BufferMapState::Unmapped;
  if (!g_mappingStates[slot].compare_exchange_strong(expected, BufferMapState::Mapping, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  g_stagingBuffers[slot].MapAsync(
      wgpu::MapMode::Write, 0, staging_layout(g_stagingSizes[slot]).total, wgpu::CallbackMode::AllowSpontaneous,
      [slot, releaseSlotOnCompletion](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status == wgpu::MapAsyncStatus::CallbackCancelled || status == wgpu::MapAsyncStatus::Aborted) {
          Log.warn("Buffer mapping {}: {}", magic_enum::enum_name(status), message);
          g_mappingStates[slot].store(BufferMapState::Unmapped, std::memory_order_release);
          if (releaseSlotOnCompletion) {
            g_stagingSlots.release(slot);
          }
          return;
        }
        AURORA_ASSERT(status == wgpu::MapAsyncStatus::Success, "Buffer mapping failed: {} {}",
                      magic_enum::enum_name(status), message);
        g_mappingStates[slot].store(BufferMapState::Mapped, std::memory_order_release);
        if (releaseSlotOnCompletion) {
          g_stagingSlots.release(slot);
        }
      });
}

void create_buffer(wgpu::Buffer& out, wgpu::BufferUsage usage, uint64_t size, const char* label,
                   bool mappedAtCreation = false) {
  if (size <= 0) {
    return;
  }
  const wgpu::BufferDescriptor descriptor{
      .label = label,
      .usage = usage,
      .size = size,
      .mappedAtCreation = mappedAtCreation,
  };
  out = g_device.CreateBuffer(&descriptor);
}

void create_staging_buffer(size_t slot, bool mappedAtCreation) {
  const auto label = fmt::format("Staging Buffer {}", slot);
  create_buffer(g_stagingBuffers[slot], StagingBufferUsage, staging_layout(g_stagingSizes[slot]).total, label.c_str(),
                mappedAtCreation);
}

void create_static_bind_group() {
  if (!g_resources.staticBindGroupLayout) {
    constexpr std::array layoutEntries{
        // Vertex data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
        // Storage data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Static bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_resources.staticBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
  }
  const std::array entries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .buffer = g_resources.vertexBuffer,
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .buffer = g_resources.storageBuffer,
      },
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .label = "Static bind group",
      .layout = g_resources.staticBindGroupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_resources.staticBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
}

// Worker only, between the previous frame's submit and this frame's first encode
void ensure_device_buffers(const StreamingSizes& sizes, uint32_t residentCapacity) {
  auto& current = g_resources.deviceSizes;
  bool rebuildStaticGroup = false;
  bool resized = false;
  if (current.vertex < sizes.vertex) {
    create_buffer(g_resources.vertexBuffer, VertexBufferUsage, sizes.vertex, "Shared Vertex Buffer");
    current.vertex = sizes.vertex;
    rebuildStaticGroup = true;
    resized = true;
  }
  if (current.index < sizes.index) {
    create_buffer(g_resources.indexBuffer, IndexBufferUsage, sizes.index, "Shared Index Buffer");
    current.index = sizes.index;
    resized = true;
  }
  if (current.storage < sizes.storage || g_resources.residentCapacity < residentCapacity) {
    resized = resized || current.storage < sizes.storage;
    create_buffer(g_resources.storageBuffer, StorageBufferUsage, sizes.storage + residentCapacity,
                  "Shared Storage Buffer");
    current.storage = sizes.storage;
    g_resources.residentCapacity = residentCapacity;
    rebuildStaticGroup = true;
  }
  if (rebuildStaticGroup) {
    create_static_bind_group();
  }
  if (resized) {
    Log.info("Streaming buffers resized: vertex {} MiB, index {} MiB, storage {} MiB", current.vertex / MiB,
             current.index / MiB, current.storage / MiB);
  } else if (rebuildStaticGroup) {
    Log.info("Resident arena resized: {} MiB", g_resources.residentCapacity / MiB);
  }
}

// Vertex and storage are bound whole in the static group; stream-out binds the index buffer whole
uint64_t buffer_bound() noexcept {
  const auto& limits = g_resources.limits;
  return std::min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
}

uint64_t resident_limit(uint64_t streamingStorage, uint64_t bound) noexcept {
  const auto room = [](uint64_t total, uint64_t used) { return total > used ? total - used : 0; };
  return std::min({bound / 4, room(bound, streamingStorage), room(UINT32_MAX, streamingStorage)});
}

StreamingLimits streaming_limits() noexcept {
  const uint64_t bound = round_streaming_limit(buffer_bound());
  return {.vertex = bound,
          .index = bound,
          .storage = round_streaming_limit(streaming_storage_limit(bound, g_residentCapacity)),
          .staging = g_resources.limits.maxBufferSize};
}

void create_compute_layouts() {
  if (g_resources.computeStreamLayout) {
    return;
  }
  const wgpu::BindGroupLayoutDescriptor emptyDesc{.label = "Compute empty bind group layout"};
  g_resources.computeEmptyLayout = g_device.CreateBindGroupLayout(&emptyDesc);
  const wgpu::BindGroupDescriptor emptyGroupDesc{
      .label = "Compute empty bind group",
      .layout = g_resources.computeEmptyLayout,
  };
  g_resources.computeEmptyGroup = g_device.CreateBindGroup(&emptyGroupDesc);
  const std::array streamEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Storage, .minBindingSize = 0},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Compute,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .hasDynamicOffset = true, .minBindingSize = 256},
      },
  };
  const wgpu::BindGroupLayoutDescriptor streamDesc{
      .label = "Compute stream bind group layout",
      .entryCount = streamEntries.size(),
      .entries = streamEntries.data(),
  };
  g_resources.computeStreamLayout = g_device.CreateBindGroupLayout(&streamDesc);
}

void release_compute_layouts() noexcept {
  g_resources.computeEmptyGroup = {};
  g_resources.computeEmptyLayout = {};
  g_resources.computeStreamLayout = {};
}

uint32_t read_test_growth_frame() noexcept {
  const char* value = SDL_getenv("AURORA_TEST_STREAMING_GROWTH");
  if (value == nullptr) {
    return 0;
  }
  const std::string_view text{value};
  uint32_t frame = 0;
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), frame);
  return ec == std::errc{} && end == text.data() + text.size() ? frame : 0;
}
} // namespace

namespace detail {

Resources& resources() noexcept { return g_resources; }

const wgpu::Buffer& staging_buffer(size_t slot) { return g_stagingBuffers[slot]; }

void request_streaming_sizes(const StreamingSizes& minimum) noexcept {
  // Sizes stay 1 MiB multiples so every staging segment keeps the map and copy alignment
  const auto merge = [](uint64_t& requested, uint64_t wanted) {
    if (wanted > requested) {
      requested = round_streaming_request(wanted);
    }
  };
  merge(g_requestedSizes.vertex, minimum.vertex);
  merge(g_requestedSizes.index, minimum.index);
  merge(g_requestedSizes.storage, minimum.storage);
}

StreamingSizes streaming_sizes() noexcept { return g_targetSizes; }

ChannelBudget channel_budget() noexcept { return g_channelBudget; }

void request_channel_storage(uint64_t bytesPerFrame) noexcept {
  g_channelDemand = std::max(g_channelDemand, bytesPerFrame);
}

void grow_streaming_storage_now(uint64_t storage) noexcept {
  // begin_frame swaps in a staging buffer of the new size, so growing between frames is safe
  g_targetSizes =
      next_streaming_sizes(g_targetSizes, AuroraStats{},
                           {.vertex = 0, .index = 0, .storage = round_streaming_request(storage)}, streaming_limits())
          .sizes;
  if (g_targetSizes.storage < storage && !std::exchange(g_warnedStorageGrowth, true)) {
    Log.warn("Streaming storage stays at {} MiB, below the {} MiB requested", g_targetSizes.storage / MiB,
             (storage + MiB - 1) / MiB);
  }
}

wgpu::BindGroupLayout compute_empty_layout() noexcept { return g_resources.computeEmptyLayout; }

wgpu::BindGroup compute_empty_group() noexcept { return g_resources.computeEmptyGroup; }

wgpu::BindGroupLayout compute_stream_layout() noexcept { return g_resources.computeStreamLayout; }

namespace testing {
StreamingSizes requested_streaming_sizes() noexcept { return g_requestedSizes; }

void reset_streaming_request() noexcept { g_requestedSizes = {}; }

void create_compute_layouts() noexcept { gfx::create_compute_layouts(); }

void release_compute_layouts() noexcept { gfx::release_compute_layouts(); }

void set_staging_buffer(size_t slot, wgpu::Buffer buffer) noexcept { g_stagingBuffers[slot] = std::move(buffer); }
} // namespace testing

StreamingGrowth next_streaming_sizes(const StreamingSizes& current, const AuroraStats& lastFrame,
                                     const StreamingSizes& requested, const StreamingLimits& limits) noexcept {
  const auto target = [](uint64_t size, uint64_t request, uint32_t highWater, uint64_t limit) {
    uint64_t next = std::max(size, request);
    // High water is a u32, so the comparison is formed in 64 bits
    if (uint64_t{highWater} * 4 > next * 3) {
      next *= 2;
    }
    return std::max(std::min(next, limit), size);
  };
  const StreamingSizes wanted{
      .vertex = target(current.vertex, requested.vertex, lastFrame.lastVertSize, limits.vertex),
      .index = target(current.index, requested.index, lastFrame.lastIndexSize, limits.index),
      .storage = target(current.storage, requested.storage, lastFrame.lastStorageSize, limits.storage),
  };
  StreamingGrowth growth{.sizes = current};
  // Steps are taken in order while one staging slot still fits in a single buffer
  for (const auto member : {&StreamingSizes::vertex, &StreamingSizes::index, &StreamingSizes::storage}) {
    if (wanted.*member == current.*member) {
      continue;
    }
    StreamingSizes candidate = growth.sizes;
    candidate.*member = wanted.*member;
    if (staging_layout(candidate).total <= limits.staging) {
      growth.sizes = candidate;
    } else {
      growth.stagingCapped = true;
    }
  }
  return growth;
}

uint32_t next_resident_capacity(uint32_t current, uint64_t request, uint64_t streamingStorage,
                                uint64_t bound) noexcept {
  // A fragmented arena can ask for less than it has, and only growing it bumps the epoch that compacts it
  if (request == 0) {
    return current;
  }
  const uint64_t wanted = std::bit_ceil(std::max<uint64_t>(2 * request, MiB));
  return static_cast<uint32_t>(std::max<uint64_t>(current, std::min(wanted, resident_limit(streamingStorage, bound))));
}

ResidentLayout next_resident_layout(const ResidentLayout& last, uint64_t streamingStorage, uint32_t capacity) noexcept {
  if (streamingStorage == last.streamingStorage && capacity == last.capacity) {
    return last;
  }
  return {.streamingStorage = streamingStorage, .capacity = capacity, .epoch = last.epoch + 1};
}

uint64_t streaming_storage_limit(uint64_t bound, uint32_t residentCapacity) noexcept {
  return bound - residentCapacity;
}

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id) {
  std::lock_guard lock{g_runtimeTypeMutex};
  const auto* slot = find_runtime_draw_type_locked(id);
  if (slot == nullptr) {
    return std::nullopt;
  }
  return RegisteredDrawType{.draw = slot->draw, .userdata = slot->userdata};
}

std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id) {
  std::lock_guard lock{g_runtimeTypeMutex};
  const auto* slot = find_runtime_encoder_task_type_locked(id);
  if (slot == nullptr) {
    return std::nullopt;
  }
  return RegisteredEncoderTaskType{
      .callback = slot->callback,
      .userdata = slot->userdata,
      .afterSubmit = slot->afterSubmit,
  };
}

} // namespace detail

wgpu::Device device() noexcept { return g_device; }

wgpu::Queue queue() noexcept { return g_queue; }

wgpu::TextureFormat color_format() noexcept { return webgpu::g_graphicsConfig.surfaceConfiguration.format; }

wgpu::TextureFormat depth_format() noexcept { return webgpu::g_graphicsConfig.depthFormat; }

uint32_t sample_count() noexcept { return webgpu::g_graphicsConfig.msaaSamples; }

RenderTargetLayout scene_render_target_layout() noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = webgpu::g_frameBuffer.format,
          .width = webgpu::g_frameBuffer.size.width,
          .height = webgpu::g_frameBuffer.size.height,
      }}},
      .depthStencilFormat = webgpu::g_graphicsConfig.depthFormat,
      .sampleCount = webgpu::g_graphicsConfig.msaaSamples,
  };
  if (webgpu::g_graphicsConfig.normalBuffer) {
    layout.colorAttachments[layout.colorAttachmentCount++] = {
        .semantic = ColorAttachmentSemantic::Normal,
        .format = webgpu::g_normalBuffer.format,
        .width = webgpu::g_normalBuffer.size.width,
        .height = webgpu::g_normalBuffer.size.height,
    };
  }
  finalize_render_target_layout(layout);
  return layout;
}

bool uses_reversed_z() noexcept { return gx::UseReversedZ; }

DrawTypeId register_draw_type(const DrawTypeDescriptor& desc) {
  if (desc.draw == nullptr) {
    Log.warn("register_draw_type: draw callback is null");
    return InvalidDrawType;
  }

  std::lock_guard lock{g_runtimeTypeMutex};
  uint32_t idx;
  if (!g_freeDrawTypeSlots.empty()) {
    idx = g_freeDrawTypeSlots.back();
    g_freeDrawTypeSlots.pop_back();
  } else {
    idx = static_cast<uint32_t>(g_runtimeDrawTypes.size());
    g_runtimeDrawTypes.emplace_back();
  }
  auto& slot = g_runtimeDrawTypes[idx];
  slot.label = desc.label != nullptr ? desc.label : "";
  slot.draw = desc.draw;
  slot.userdata = desc.userdata;
  return make_draw_type_id(idx, slot.generation);
}

void unregister_draw_type(DrawTypeId type) noexcept {
  std::lock_guard lock{g_runtimeTypeMutex};
  if (find_runtime_draw_type_locked(type) == nullptr) {
    return;
  }
  const auto idx = draw_type_index(type);
  auto& slot = g_runtimeDrawTypes[idx];
  slot.label.clear();
  slot.draw = nullptr;
  slot.userdata = nullptr;
  ++slot.generation;
  g_freeDrawTypeSlots.push_back(idx);
}

EncoderTaskId register_encoder_task_type(const EncoderTaskDescriptor& desc) {
  if (desc.callback == nullptr) {
    Log.warn("register_encoder_task_type: callback is null");
    return InvalidEncoderTask;
  }

  std::lock_guard lock{g_runtimeTypeMutex};
  uint32_t idx;
  if (!g_freeEncoderTaskTypeSlots.empty()) {
    idx = g_freeEncoderTaskTypeSlots.back();
    g_freeEncoderTaskTypeSlots.pop_back();
  } else {
    idx = static_cast<uint32_t>(g_runtimeEncoderTaskTypes.size());
    g_runtimeEncoderTaskTypes.emplace_back();
  }
  auto& slot = g_runtimeEncoderTaskTypes[idx];
  slot.label = desc.label != nullptr ? desc.label : "";
  slot.callback = desc.callback;
  slot.userdata = desc.userdata;
  slot.afterSubmit = desc.afterSubmit;
  return make_draw_type_id(idx, slot.generation);
}

void unregister_encoder_task_type(EncoderTaskId type) noexcept {
  std::lock_guard lock{g_runtimeTypeMutex};
  if (find_runtime_encoder_task_type_locked(type) == nullptr) {
    return;
  }
  const auto idx = draw_type_index(type);
  auto& slot = g_runtimeEncoderTaskTypes[idx];
  slot.label.clear();
  slot.callback = nullptr;
  slot.userdata = nullptr;
  slot.afterSubmit = nullptr;
  ++slot.generation;
  g_freeEncoderTaskTypeSlots.push_back(idx);
}

void initialize() {
  g_frameIndex = 0;
  g_processEventsQueued.store(false, std::memory_order_release);
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameTimeNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  render_worker::initialize();
  // This appears to take a while and blocks the render thread for periods of time
  // render_worker::set_event_pump([] {
  //   if (g_instance) {
  //     g_instance.ProcessEvents();
  //   }
  // });
  depth_peek::initialize();
  tex_copy_conv::initialize();
  tex_palette_conv::initialize();

  // For uniform & storage buffer offset alignments
  g_device.GetLimits(&g_resources.limits);
  g_testGrowthFrame = read_test_growth_frame();
  g_channelBudget = {
      .bytes = channel_storage_budget(g_targetSizes.storage, 0, false, render::gpu::MaxFrameChannelBytes),
      .alignment = g_resources.limits.minStorageBufferOffsetAlignment,
  };

  create_buffer(g_resources.uniformBuffer, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, UniformBufferSize,
                "Shared Uniform Buffer");
  create_buffer(g_resources.vertexBuffer, VertexBufferUsage, g_targetSizes.vertex, "Shared Vertex Buffer");
  create_buffer(g_resources.indexBuffer, IndexBufferUsage, g_targetSizes.index, "Shared Index Buffer");
  create_buffer(g_resources.storageBuffer, StorageBufferUsage, g_targetSizes.storage, "Shared Storage Buffer");
  g_resources.deviceSizes = g_targetSizes;
  ++g_residentLayout.epoch;
  for (size_t i = 0; i < g_stagingBuffers.size(); ++i) {
    create_staging_buffer(i, false);
  }
  for (auto& state : g_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
  for (size_t slot = 0; slot < g_stagingBuffers.size(); ++slot) {
    map_staging_buffer(slot);
  }

  create_static_bind_group();

  {
    constexpr std::array layoutEntries{
        // Uniform buffer (dynamic offset)
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::Uniform,
                    .hasDynamicOffset = true,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Uniform bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_resources.uniformBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_resources.uniformBuffer,
            .size = gx::MaxUniformSize,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Uniform bind group",
        .layout = g_resources.uniformBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_resources.uniformBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }
  create_compute_layouts();

  gx::initialize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize_pipeline();
#endif
  initialize_pipeline_cache();
  render::detail::initialize();
}

void shutdown() {
  render_worker::synchronize();
  render_worker::shutdown();
  render::detail::quiesce();
  g_processEventsQueued.store(false, std::memory_order_release);
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameTimeNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  shutdown_pipeline_cache();
  depth_peek::shutdown();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  texture_replacement::shutdown();
  gx::shutdown();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown_pipeline();
#endif

  shutdown_resource_cache();
  {
    std::lock_guard lock{g_runtimeTypeMutex};
    g_runtimeDrawTypes.clear();
    g_freeDrawTypeSlots.clear();
    g_runtimeEncoderTaskTypes.clear();
    g_freeEncoderTaskTypeSlots.clear();
  }
  g_resources.vertexBuffer = {};
  g_resources.uniformBuffer = {};
  g_resources.indexBuffer = {};
  g_resources.storageBuffer = {};
  g_resources.deviceSizes = {};
  g_resources.residentCapacity = 0;
  g_stagingBuffers.fill({});
  g_stagingSizes.fill({});
  g_targetSizes = {};
  g_requestedSizes = {};
  g_residentCapacity = 0;
  take_resident_request();
  g_loggedStagingCap = false;
  for (auto& packet : g_framePackets) {
    packet = {};
  }
  render::detail::shutdown();
  shutdown_recording();
  release_compute_layouts();
  g_channelBudget = {};
  g_channelDemand = 0;
  g_warnedChannelBudget = false;
  g_warnedStorageGrowth = false;
  g_resources.staticBindGroup = {};
  g_resources.staticBindGroupLayout = {};
  g_resources.uniformBindGroup = {};
  g_resources.uniformBindGroupLayout = {};
  g_frameIndex = UINT32_MAX;
  g_frameSlots.reset();
  g_stagingSlots.reset();
  for (auto& state : g_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
}

bool wait_for_staging_buffer(size_t slot) {
  ZoneScopedN("Wait for buffer map");
  map_staging_buffer(slot);
  while (true) {
    const auto mappingState = g_mappingStates[slot].load(std::memory_order_acquire);
    if (mappingState == BufferMapState::Mapped) {
      return true;
    }
    if (mappingState == BufferMapState::Unmapped) {
      return false;
    }
    wait_for_gpu_progress(std::chrono::milliseconds{1});
  }
}

size_t acquire_frame_slot() {
  ZoneScopedN("Acquire frame slot");
  const auto waitStart = PresentClock::now();
  while (true) {
    if (const auto slot = g_frameSlots.try_acquire()) {
      const auto waitDuration = PresentClock::now() - waitStart;
      const double waitMs = std::chrono::duration<double, std::milli>{waitDuration}.count();
      TracyPlot("aurora: frameSlotWaitMs", waitMs);
      return *slot;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

std::optional<size_t> acquire_mapped_staging_buffer() {
  ZoneScopedN("Acquire mapped staging buffer");
  while (true) {
    if (auto slot = g_stagingSlots.try_acquire()) {
      if (wait_for_staging_buffer(*slot)) {
        return *slot;
      }
      g_stagingSlots.release(*slot);
      return std::nullopt;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

bool begin_frame() {
  ZoneScoped;
  // pace_frame_start();
  const size_t frameSlot = acquire_frame_slot();
  const auto stagingSlot = acquire_mapped_staging_buffer();
  if (!stagingSlot) {
    g_frameSlots.release(frameSlot);
    return false;
  }

  if (g_stagingSizes[*stagingSlot] != g_targetSizes) {
    // A held slot has no map callback in flight, so its buffer can be swapped for one mapped at creation
    g_stagingSizes[*stagingSlot] = g_targetSizes;
    create_staging_buffer(*stagingSlot, true);
  }

  auto& frame = g_framePackets[frameSlot];
  frame = {};
  frame.frameId = g_nextFrameId++;
  frame.frameIndex = g_frameIndex;
  frame.stagingBuffer = *stagingSlot;
  frame.sizes = g_targetSizes;
  frame.residentCapacity = g_residentCapacity;
  frame.residentLimit = static_cast<uint32_t>(resident_limit(g_targetSizes.storage, buffer_bound()));
  g_residentLayout = next_resident_layout(g_residentLayout, g_targetSizes.storage, g_residentCapacity);
  frame.residentEpoch = g_residentLayout.epoch;
  frame.renderState = render::detail::latch_frame();
  // After latch_frame, so blocks made while latching land in this frame, and before the begin item is queued
  render::detail::on_frame_recording(frame.frameId);
  size_t bufferOffset = 0;
  const auto& stagingBuf = g_stagingBuffers[*stagingSlot];
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{static_cast<u8*>(stagingBuf.GetMappedRange(bufferOffset, size)), static_cast<size_t>(size)};
    bufferOffset += size;
  };
  mapBuffer(frame.verts, frame.sizes.vertex);
  mapBuffer(frame.uniforms, UniformBufferSize);
  mapBuffer(frame.indices, frame.sizes.index);
  mapBuffer(frame.storage, frame.sizes.storage);
  if constexpr (UseTextureBuffer) {
    mapBuffer(frame.textureUpload, TextureUploadSize);
  }

  begin_recording(frame, frameSlot);
  begin_pipeline_frame();
  render_worker::enqueue_begin_frame(frame.frameId, [frameSlot, frameId = frame.frameId, sizes = frame.sizes,
                                                     residentCapacity = frame.residentCapacity,
                                                     renderState = frame.renderState] {
    ensure_device_buffers(sizes, residentCapacity);
    render::detail::on_worker_frame_begin(frameId, renderState.get());
    constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "Redraw encoder"};
    g_framePackets[frameSlot].encoder = g_device.CreateCommandEncoder(&EncoderDescriptor);
    webgpu::gpu_prof::frame_begin(g_framePackets[frameSlot].encoder);
  });
  g_cpuFrameStart = PresentClock::now();
  return true;
}

void end_frame(EndFrameCallback callback) {
  ZoneScoped;
  if (g_cpuFrameStart.time_since_epoch().count() != 0) {
    const auto cpuFrameTime = PresentClock::now() - g_cpuFrameStart;
    update_ema(g_cpuFrameTimeNs, duration_ns(cpuFrameTime));
    const double cpuFrameTimeMs = std::chrono::duration<double, std::milli>{cpuFrameTime}.count();
    TracyPlot("aurora: cpuFrameTimeMs", cpuFrameTimeMs);
  }
  const auto recorded = end_recording();
  auto& frame = *recorded.packet;
  const size_t frameSlot = recorded.frameSlot;
  const uint64_t frameId = frame.frameId;
  end_pipeline_frame();
  ++g_frameIndex;

  // Runs before the end-frame item, which resets the packet and its stats on the worker
  if (g_testGrowthFrame != 0 && frame.frameIndex == g_testGrowthFrame) {
    g_requestedSizes.storage = 2 * g_targetSizes.storage;
  }
  const uint64_t lastStorage = frame.stats.lastStorageSize;
  const uint64_t legacy = lastStorage - std::min(lastStorage, last_channel_storage_bytes());
  if (g_channelDemand != 0) {
    request_streaming_sizes({.vertex = 0, .index = 0, .storage = channel_storage_request(g_channelDemand, legacy)});
  }
  const auto growth = next_streaming_sizes(g_targetSizes, frame.stats, g_requestedSizes, streaming_limits());
  g_targetSizes = growth.sizes;
  g_channelBudget.bytes =
      channel_storage_budget(g_targetSizes.storage, legacy, true, render::gpu::MaxFrameChannelBytes);
  if (g_channelDemand > g_channelBudget.bytes && !std::exchange(g_warnedChannelBudget, true)) {
    Log.warn("GPU channel budget {} bytes stays below its demand of {} bytes; streaming storage is {} MiB of {} MiB",
             g_channelBudget.bytes, g_channelDemand, g_targetSizes.storage / MiB, streaming_limits().storage / MiB);
  }
  g_channelDemand = 0;
  g_residentCapacity =
      next_resident_capacity(g_residentCapacity, take_resident_request(), g_targetSizes.storage, buffer_bound());
  if (growth.stagingCapped && !std::exchange(g_loggedStagingCap, true)) {
    Log.warn("Streaming buffer growth capped: a staging slot would exceed maxBufferSize ({} MiB)",
             g_resources.limits.maxBufferSize / MiB);
  }

  const size_t stagingSlot = frame.stagingBuffer;
  render_worker::enqueue_end_frame(frameId, [frameSlot, stagingSlot, callback = std::move(callback)]() mutable {
    auto& packet = g_framePackets[frameSlot];
    g_stagingBuffers[stagingSlot].Unmap();
    g_mappingStates[stagingSlot].store(BufferMapState::Unmapped, std::memory_order_release);
    const uint64_t stagedStorage = staging_layout(packet.sizes).storage;
    for (const auto& copy : packet.residentCopies) {
      const uint64_t dst = packet.sizes.storage + copy.dst;
      packet.encoder.CopyBufferToBuffer(g_stagingBuffers[stagingSlot], stagedStorage + copy.src,
                                        g_resources.storageBuffer, dst, copy.copySize);
      if (copy.blockSize > copy.copySize) {
        packet.encoder.ClearBuffer(g_resources.storageBuffer, dst + copy.copySize, copy.blockSize - copy.copySize);
      }
    }
    auto encoder = std::move(packet.encoder);
    const auto stats = packet.stats;
    auto afterSubmitCallbacks = std::move(packet.afterSubmitCallbacks);
    g_submitState = std::move(packet.renderState);
    packet = {};
    g_resources.stats.drawCallCount = stats.drawCallCount;
    g_resources.stats.mergedDrawCallCount = stats.mergedDrawCallCount;
    g_resources.stats.lastVertSize = stats.lastVertSize;
    g_resources.stats.lastUniformSize = stats.lastUniformSize;
    g_resources.stats.lastIndexSize = stats.lastIndexSize;
    g_resources.stats.lastStorageSize = stats.lastStorageSize;
    g_resources.stats.lastTextureUploadSize = stats.lastTextureUploadSize;
    if (callback) {
      callback(encoder, std::move(afterSubmitCallbacks));
    }
    g_submitState.reset();
    g_frameSlots.release(frameSlot);
    expire_cached_bind_groups();
    map_staging_buffer(stagingSlot, true);
    process_events();
  });
}

uint32_t current_frame() noexcept { return g_frameIndex; }

void after_submit() noexcept { depth_peek::after_submit(); }

void before_submit() noexcept { render::detail::before_submit(g_submitState.get()); }

void after_frame() noexcept { render::detail::after_frame(g_submitState.get()); }

void gpu_synchronize() { render_worker::synchronize(); }

void synchronize() { render_worker::synchronize(); }

void after_present() noexcept {
  const auto now = PresentClock::now();
  const int64_t nowNs = timestamp_ns(now);
  const int64_t previousPresentNs = g_lastPresentNs.exchange(nowNs, std::memory_order_acq_rel);
  if (previousPresentNs != 0) {
    update_ema(g_presentPeriodNs, nowNs - previousPresentNs);
    const double presentPeriodMs = static_cast<double>(g_presentPeriodNs.load(std::memory_order_acquire)) / 1'000'000.0;
    TracyPlot("aurora: presentPeriodMs", presentPeriodMs);
  }
  std::lock_guard lock{g_presentStatsMutex};
  g_presentTimes.push_back(now);
  prune_present_times(now);
}

float calculate_fps() noexcept {
  const auto now = PresentClock::now();
  std::lock_guard lock{g_presentStatsMutex};
  prune_present_times(now);
  if (g_presentTimes.size() < 2) {
    return 0.f;
  }

  const auto elapsed = std::chrono::duration<float>(g_presentTimes.back() - g_presentTimes.front()).count();
  if (elapsed <= 0.f) {
    return 0.f;
  }
  return static_cast<float>(g_presentTimes.size() - 1) / elapsed;
}
} // namespace aurora::gfx

const AuroraStats* aurora_get_stats() { return &aurora::gfx::detail::resources().stats; }
float aurora_get_fps() { return aurora::gfx::calculate_fps(); }

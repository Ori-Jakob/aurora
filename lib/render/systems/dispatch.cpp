#include "dispatch.hpp"

#include "systems.hpp"

#include "../../gfx/pipeline_cache.hpp"
#include "../../internal.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/gpu_systems.hpp>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace aurora::render::systems {
namespace {
constexpr Module Log{"aurora::render::systems"};
constexpr uint64_t StateSweepInterval = 16;
constexpr uint64_t StateIdleFrames = 32;
constexpr uint64_t DummyStateBytes = 256;

enum class Reason : uint8_t { Unknown, Pending, Failed, Count };

struct ReadyCache {
  uint64_t session = 0;
  std::vector<uint32_t> pipelines;
};

struct WorkerPipeline {
  uint32_t id = 0;
  wgpu::ComputePipeline pipeline;
};

struct StateGroup {
  gpu_heap::BlockHandle block = gpu_heap::BlockHandle::Invalid;
  uint32_t bytes = 0;
  uint32_t generation = 0;
  uint64_t lastFrame = 0;
  wgpu::BindGroup group;
};

// FIFO thread only
ReadyCache g_ready;
std::array<bool, static_cast<size_t>(Reason::Count)> g_warnedReason{};
std::array<bool, magic_enum::enum_count<gfx::RecordResult>()> g_warnedRecord{};

// Render worker only
uint64_t g_workerSession = 0;
std::vector<WorkerPipeline> g_workerPipelines;
std::vector<StateGroup> g_stateGroups;
uint64_t g_workerFrame = 0;

wgpu::Buffer g_dummyState;
wgpu::BindGroup g_dummyGroup;

template <typename Message>
bool reject(Reason reason, Message&& message) noexcept {
  if (!std::exchange(g_warnedReason[static_cast<size_t>(reason)], true)) {
    Log.warn("GPU dispatch rejected: {}", message());
  }
  return false;
}

bool kernel_ready(uint64_t system, shaders::PipelineId pipeline) noexcept {
  const uint64_t session = shaders::session();
  if (g_ready.session != session) {
    g_ready = {.session = session};
  }
  if (std::ranges::find(g_ready.pipelines, pipeline.value) != g_ready.pipelines.end()) {
    return true;
  }
  shaders::PipelineStatus status = shaders::status(pipeline);
  if (status == shaders::PipelineStatus::Pending && gfx::blocking_pipelines()) {
    wait_ready(system);
    status = shaders::status(pipeline);
  }
  if (status == shaders::PipelineStatus::Pending) {
    return reject(Reason::Pending, [&] { return fmt::format("system 0x{:016X} is still compiling", system); });
  }
  if (status == shaders::PipelineStatus::Failed) {
    return reject(Reason::Failed, [&] { return fmt::format("system 0x{:016X} failed to compile", system); });
  }
  g_ready.pipelines.push_back(pipeline.value);
  return true;
}

wgpu::ComputePipeline worker_pipeline(shaders::PipelineId id) {
  const uint64_t session = shaders::session();
  if (g_workerSession != session) {
    g_workerPipelines.clear();
    g_workerSession = session;
  }
  const auto it = std::ranges::find(g_workerPipelines, id.value, &WorkerPipeline::id);
  if (it != g_workerPipelines.end()) {
    return it->pipeline;
  }
  wgpu::ComputePipeline pipeline = shaders::compute_pipeline(id);
  if (pipeline) {
    g_workerPipelines.push_back({.id = id.value, .pipeline = pipeline});
  }
  return pipeline;
}

wgpu::BindGroup state_group(const KernelRef& ref) {
  const gpu_heap::BlockView view = gpu_heap::block_view(ref.state);
  if (!view || view.size < ref.stateBytes) {
    return {};
  }
  auto it = std::ranges::find(g_stateGroups, ref.state, &StateGroup::block);
  if (it == g_stateGroups.end()) {
    it = g_stateGroups.insert(g_stateGroups.end(), StateGroup{.block = ref.state});
  }
  it->lastFrame = g_workerFrame;
  if (it->group && it->generation == view.generation && it->bytes == ref.stateBytes) {
    return it->group;
  }
  const wgpu::BindGroupEntry entry{
      .binding = 0,
      .buffer = view.buffer,
      .offset = view.offset,
      .size = ref.stateBytes,
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "GPU system state",
      .layout = state_layout(),
      .entryCount = 1,
      .entries = &entry,
  };
  it->group = webgpu::g_device.CreateBindGroup(&descriptor);
  it->generation = view.generation;
  it->bytes = ref.stateBytes;
  return it->group;
}
} // namespace

bool bind_kernel(const KernelRef& ref, const wgpu::ComputePassEncoder& pass) {
  const wgpu::ComputePipeline pipeline = worker_pipeline(ref.pipeline);
  if (!pipeline) {
    return false;
  }
  const wgpu::BindGroup group = ref.stateBytes == 0 ? g_dummyGroup : state_group(ref);
  if (!group) {
    return false;
  }
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(2, group);
  return true;
}

gfx::detail::CopyDestination state_target(const StateRef& ref) {
  const gpu_heap::BlockView view = gpu_heap::block_view(ref.state);
  if (!view || view.size < ref.stateBytes) {
    return {};
  }
  return {.buffer = view.buffer, .base = view.offset, .size = ref.stateBytes};
}

void initialize_dispatch() {
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "GPU system dummy state",
      .usage = wgpu::BufferUsage::Storage,
      .size = DummyStateBytes,
  };
  g_dummyState = webgpu::g_device.CreateBuffer(&bufferDescriptor);
  const wgpu::BindGroupEntry entry{.binding = 0, .buffer = g_dummyState, .offset = 0, .size = DummyStateBytes};
  const wgpu::BindGroupDescriptor groupDescriptor{
      .label = "GPU system dummy state",
      .layout = state_layout(),
      .entryCount = 1,
      .entries = &entry,
  };
  g_dummyGroup = webgpu::g_device.CreateBindGroup(&groupDescriptor);
}

void shutdown_dispatch() noexcept {
  g_dummyGroup = {};
  g_dummyState = {};
  g_workerPipelines.clear();
  g_workerSession = 0;
  g_stateGroups.clear();
  g_workerFrame = 0;
  g_ready = {};
  g_warnedReason = {};
  g_warnedRecord = {};
}

void tick_worker(uint64_t frameId) {
  g_workerFrame = frameId;
  if (frameId % StateSweepInterval != 0) {
    return;
  }
  std::erase_if(g_stateGroups, [&](const StateGroup& group) { return group.lastFrame + StateIdleFrames < frameId; });
}

bool decode_dispatch(void*, const gx::gpu_channel::DispatchDecode& desc) noexcept {
  KernelSlot slot;
  if (!kernel_slot(desc.system, desc.kernel, slot) || !slot.pipeline) {
    return reject(Reason::Unknown,
                  [&] { return fmt::format("system 0x{:016X} has no compiled kernel {}", desc.system, desc.kernel); });
  }
  if (!kernel_ready(desc.system, slot.pipeline)) {
    return false;
  }
  if (slot.stateBytes != 0 && slot.state == gpu_heap::BlockHandle::Invalid) {
    return reject(Reason::Failed, [&] { return fmt::format("system 0x{:016X} has no state block", desc.system); });
  }
  std::array<uint32_t, 3> groups{};
  for (size_t d = 0; d < groups.size(); ++d) {
    const uint32_t size = std::max(slot.workgroupSize[d], 1u);
    groups[d] = desc.elements[d] == 0 ? 0 : (desc.elements[d] - 1) / size + 1;
  }
  const KernelRef ref{.pipeline = slot.pipeline, .state = slot.state, .stateBytes = slot.stateBytes};
  const gfx::RecordResult result = gfx::record_dispatch<&bind_kernel>(ref, {
                                                                               .output = desc.output,
                                                                               .inputs = desc.inputs,
                                                                               .groups = groups,
                                                                               .elements = desc.elements,
                                                                               .user = desc.params,
                                                                           });
  if (result != gfx::RecordResult::Recorded) {
    if (!std::exchange(g_warnedRecord[static_cast<size_t>(result)], true)) {
      Log.warn("GPU dispatch of system 0x{:016X} not recorded: {}", desc.system, magic_enum::enum_name(result));
    }
    return false;
  }
  return true;
}

bool apply_update(void*, uint64_t id, uint32_t byteOffset, std::span<const uint8_t> bytes) noexcept {
  KernelSlot slot;
  if (!kernel_slot(id, 0, slot) || slot.state == gpu_heap::BlockHandle::Invalid) {
    return true;
  }
  const gfx::Range staged = gfx::push_channel_storage(bytes);
  if (staged.size == 0) {
    return false;
  }
  const StateRef ref{.state = slot.state, .stateBytes = slot.stateBytes};
  return gfx::record_copy<&state_target>(ref, staged, byteOffset) == gfx::RecordResult::Recorded;
}
} // namespace aurora::render::systems

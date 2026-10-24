#include "systems.hpp"

#include "dispatch.hpp"

#include "../../gfx/frame.hpp"
#include "../../gx/gpu_channel.hpp"
#include "../../internal.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/gpu_systems.hpp>
#include <aurora/render_caps.hpp>
#include <fmt/format.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::systems {
namespace {
using gpu::SystemStatus;
using gx::gpu_channel::KernelCreate;
using gx::gpu_channel::SystemCreate;
constexpr Module Log{"aurora::render::systems"};
constexpr uint64_t WaitNs = 10'000'000'000;
constexpr std::array<std::string_view, 1> AtomicDefines{"AURORA_STATE_ATOMIC"};

struct Slot {
  uint64_t id = 0;
  SystemStatus fixed = SystemStatus::Pending;
  uint32_t kernelCount = 0;
  std::array<shaders::PipelineId, gpu::MaxKernels> pipelines{};
  std::array<std::array<uint32_t, 3>, gpu::MaxKernels> workgroups{};
  gpu_heap::BlockHandle state = gpu_heap::BlockHandle::Invalid;
  uint32_t stateBytes = 0;
};

struct PipelineKey {
  std::string shader;
  std::string entry;
  std::array<uint32_t, 3> size{};
  uint32_t flags = 0;
  bool operator==(const PipelineKey&) const = default;
};

struct CachedPipeline {
  PipelineKey key;
  shaders::PipelineId id;
};

// status runs on the game thread while create and destroy run on the FIFO thread
std::mutex g_slotMutex;
std::array<Slot, gpu::MaxSystems> g_slots;
std::vector<CachedPipeline> g_pipelines;
uint64_t g_pipelineSession = 0;
wgpu::BindGroupLayout g_stateLayout;
wgpu::PipelineLayout g_kernelLayout;

uint32_t slot_index(uint64_t id) noexcept { return static_cast<uint32_t>(id) - 1; }

SystemStatus check(const SystemCreate& desc) noexcept {
  const Caps& deviceCaps = caps();
  if (((desc.requirements & gpu::RequiresCore) != 0 && !deviceCaps.core) ||
      ((desc.requirements & gpu::RequiresSubgroups) != 0 && !deviceCaps.subgroups) ||
      ((desc.requirements & gpu::RequiresShaderF16) != 0 && !deviceCaps.shaderF16)) {
    Log.info("GPU system '{}' is unavailable: the device lacks a feature it requires", desc.name);
    return SystemStatus::Unavailable;
  }
  const auto& limits = deviceCaps.limits;
  const uint64_t stateLimit = std::min<uint64_t>(gpu::MaxGpuOnlyStateBytes, limits.maxStorageBufferBindingSize);
  if (desc.persistentBytes > stateLimit) {
    Log.warn("GPU system '{}' failed: {} bytes of state are past the device's binding limit", desc.name,
             desc.persistentBytes);
    return SystemStatus::Failed;
  }
  for (const KernelCreate& kernel : desc.kernels) {
    const auto& size = kernel.workgroupSize;
    if (size[0] > limits.maxComputeWorkgroupSizeX || size[1] > limits.maxComputeWorkgroupSizeY ||
        size[2] > limits.maxComputeWorkgroupSizeZ ||
        uint64_t{size[0]} * size[1] * size[2] > limits.maxComputeInvocationsPerWorkgroup) {
      Log.warn("GPU system '{}' failed: kernel {} uses a {}x{}x{} workgroup the device cannot run", desc.name,
               kernel.entryPoint, size[0], size[1], size[2]);
      return SystemStatus::Failed;
    }
  }
  return SystemStatus::Pending;
}

std::string shader_id(const SystemCreate& desc) {
  if (desc.library) {
    if (!shaders::has_source(desc.source)) {
      Log.warn("GPU system '{}' failed: no embedded shader {}", desc.name, desc.source);
      return {};
    }
    return std::string{desc.source};
  }
  std::string id = fmt::format("gpusystem/{:016x}", XXH3_64bits(desc.source.data(), desc.source.size()));
  if (!shaders::has_source(id)) {
    std::string text = "//#import aurora/stream\n";
    text.append(desc.source);
    if (!shaders::register_source(id, text)) {
      Log.warn("GPU system '{}' failed: its source could not be registered", desc.name);
      return {};
    }
  }
  return id;
}

shaders::PipelineId pipeline_for(const std::string& shader, std::string_view name, const KernelCreate& kernel) {
  PipelineKey key{
      .shader = shader,
      .entry = std::string{kernel.entryPoint},
      .size = kernel.workgroupSize,
      .flags = kernel.flags,
  };
  const uint64_t session = shaders::session();
  {
    std::lock_guard lock{g_slotMutex};
    if (g_pipelineSession != session) {
      g_pipelines.clear();
      g_pipelineSession = session;
    }
    for (const CachedPipeline& cached : g_pipelines) {
      if (cached.key == key) {
        return cached.id;
      }
    }
  }
  const std::string label = fmt::format("{}:{}", name, kernel.entryPoint);
  const std::array constants{
      shaders::Constant{.name = "aurora_wg_x", .value = static_cast<double>(kernel.workgroupSize[0])},
      shaders::Constant{.name = "aurora_wg_y", .value = static_cast<double>(kernel.workgroupSize[1])},
      shaders::Constant{.name = "aurora_wg_z", .value = static_cast<double>(kernel.workgroupSize[2])},
  };
  const bool atomic = (kernel.flags & gpu::KernelStateAtomic) != 0;
  const shaders::PipelineId id = shaders::request_compute({
      .label = label,
      .shader = shader,
      .entryPoint = kernel.entryPoint,
      .defines = atomic ? std::span<const std::string_view>{AtomicDefines} : std::span<const std::string_view>{},
      .constants = constants,
      .layout = g_kernelLayout,
      .layoutKey = KernelLayoutKey,
  });
  if (id) {
    std::lock_guard lock{g_slotMutex};
    g_pipelines.push_back({.key = std::move(key), .id = id});
  }
  return id;
}

void backend_create(void*, const SystemCreate& desc) noexcept {
  const uint32_t index = slot_index(desc.id);
  if (index >= gpu::MaxSystems) {
    return;
  }
  Slot slot{.id = desc.id, .kernelCount = static_cast<uint32_t>(desc.kernels.size())};
  slot.fixed = check(desc);
  if (slot.fixed == SystemStatus::Pending) {
    const std::string shader = shader_id(desc);
    for (uint32_t i = 0; i < slot.kernelCount && !shader.empty(); ++i) {
      slot.pipelines[i] = pipeline_for(shader, desc.name, desc.kernels[i]);
      if (!slot.pipelines[i]) {
        break;
      }
    }
    if (shader.empty() || std::any_of(slot.pipelines.begin(), slot.pipelines.begin() + slot.kernelCount,
                                      [](shaders::PipelineId id) { return !id; })) {
      slot.fixed = SystemStatus::Failed;
    }
  }
  for (uint32_t i = 0; i < slot.kernelCount; ++i) {
    slot.workgroups[i] = desc.kernels[i].workgroupSize;
  }
  if (slot.fixed == SystemStatus::Pending && desc.persistentBytes != 0) {
    slot.state = gpu_heap::create_block(gpu_heap::Region::SystemState,
                                        AURORA_ALIGN(uint64_t{desc.persistentBytes}, gpu_heap::RegionGranularity));
    slot.stateBytes = desc.persistentBytes;
    if (slot.state == gpu_heap::BlockHandle::Invalid) {
      slot.fixed = SystemStatus::Failed;
    }
  }
  std::lock_guard lock{g_slotMutex};
  g_slots[index] = slot;
}

void release_slot(Slot& slot) noexcept {
  if (slot.state != gpu_heap::BlockHandle::Invalid) {
    gpu_heap::destroy_block(slot.state);
  }
  slot = {};
}

void backend_destroy(void*, uint64_t id) noexcept {
  const uint32_t index = slot_index(id);
  if (index >= gpu::MaxSystems) {
    return;
  }
  std::lock_guard lock{g_slotMutex};
  if (g_slots[index].id == id) {
    release_slot(g_slots[index]);
  }
}

void release_slots() noexcept {
  std::lock_guard lock{g_slotMutex};
  for (Slot& slot : g_slots) {
    release_slot(slot);
  }
}

void backend_reset(void*) noexcept { release_slots(); }

SystemStatus backend_status(void*, uint64_t id) noexcept {
  const uint32_t index = slot_index(id);
  if (index >= gpu::MaxSystems) {
    return SystemStatus::Pending;
  }
  std::lock_guard lock{g_slotMutex};
  const Slot& slot = g_slots[index];
  if (slot.id != id) {
    return SystemStatus::Pending;
  }
  if (slot.fixed != SystemStatus::Pending) {
    return slot.fixed;
  }
  bool ready = true;
  for (uint32_t i = 0; i < slot.kernelCount; ++i) {
    switch (shaders::status(slot.pipelines[i])) {
    case shaders::PipelineStatus::Failed:
      return SystemStatus::Failed;
    case shaders::PipelineStatus::Pending:
      ready = false;
      break;
    case shaders::PipelineStatus::Ready:
      break;
    }
  }
  return ready ? SystemStatus::Ready : SystemStatus::Pending;
}

std::vector<shaders::PipelineId> pending_pipelines(const Slot& slot) {
  if (slot.id == 0 || slot.fixed != SystemStatus::Pending) {
    return {};
  }
  return {slot.pipelines.begin(), slot.pipelines.begin() + slot.kernelCount};
}

void backend_wait_all(void*) noexcept {
  std::vector<shaders::PipelineId> ids;
  {
    std::lock_guard lock{g_slotMutex};
    for (const Slot& slot : g_slots) {
      const auto pipelines = pending_pipelines(slot);
      ids.insert(ids.end(), pipelines.begin(), pipelines.end());
    }
  }
  for (const shaders::PipelineId id : ids) {
    shaders::wait(id, WaitNs);
  }
}

constexpr gx::gpu_channel::SystemBackend Backend{
    .create = backend_create,
    .destroy = backend_destroy,
    .reset = backend_reset,
    .status = backend_status,
    .wait_all = backend_wait_all,
    .dispatch = decode_dispatch,
    .update = apply_update,
};
} // namespace

void initialize() {
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    return;
  }
  const wgpu::BindGroupLayout empty = gfx::detail::compute_empty_layout();
  const wgpu::BindGroupLayout stream = gfx::detail::compute_stream_layout();
  if (!empty || !stream) {
    return;
  }
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Compute,
      .buffer = {.type = wgpu::BufferBindingType::Storage, .minBindingSize = 0},
  };
  const wgpu::BindGroupLayoutDescriptor stateDescriptor{
      .label = "GPU system state",
      .entryCount = 1,
      .entries = &entry,
  };
  g_stateLayout = device.CreateBindGroupLayout(&stateDescriptor);
  const std::array layouts{empty, stream, g_stateLayout};
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "GPU system kernel",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  g_kernelLayout = device.CreatePipelineLayout(&layoutDescriptor);
  initialize_dispatch();
  gx::gpu_channel::set_system_backend(&Backend);
}

void shutdown() {
  gx::gpu_channel::set_system_backend(nullptr);
  shutdown_dispatch();
  release_slots();
  {
    std::lock_guard lock{g_slotMutex};
    g_pipelines.clear();
    g_pipelineSession = 0;
  }
  g_kernelLayout = {};
  g_stateLayout = {};
}

wgpu::BindGroupLayout state_layout() noexcept { return g_stateLayout; }

wgpu::PipelineLayout kernel_layout() noexcept { return g_kernelLayout; }

shaders::PipelineId kernel_pipeline(uint64_t id, uint32_t kernel) noexcept {
  const uint32_t index = slot_index(id);
  if (index >= gpu::MaxSystems) {
    return {};
  }
  std::lock_guard lock{g_slotMutex};
  const Slot& slot = g_slots[index];
  return slot.id == id && kernel < slot.kernelCount ? slot.pipelines[kernel] : shaders::PipelineId{};
}

bool kernel_slot(uint64_t id, uint32_t kernel, KernelSlot& out) noexcept {
  const uint32_t index = slot_index(id);
  if (index >= gpu::MaxSystems) {
    return false;
  }
  std::lock_guard lock{g_slotMutex};
  const Slot& slot = g_slots[index];
  if (slot.id != id || kernel >= slot.kernelCount) {
    return false;
  }
  out = {
      .pipeline = slot.pipelines[kernel],
      .workgroupSize = slot.workgroups[kernel],
      .state = slot.state,
      .stateBytes = slot.stateBytes,
  };
  return true;
}

bool wait_ready(uint64_t id) noexcept {
  const uint32_t index = slot_index(id);
  if (index >= gpu::MaxSystems) {
    return false;
  }
  std::vector<shaders::PipelineId> ids;
  {
    std::lock_guard lock{g_slotMutex};
    if (g_slots[index].id != id) {
      return false;
    }
    ids = pending_pipelines(g_slots[index]);
  }
  for (const shaders::PipelineId pipeline : ids) {
    shaders::wait(pipeline, WaitNs);
  }
  return backend_status(nullptr, id) == SystemStatus::Ready;
}
} // namespace aurora::render::systems

#include "native_step.hpp"

#include "device.hpp"
#include "sentinel.hpp"
#include "teardown.hpp"

#include "../../post/upscale/compose.hpp"
#include "../../../internal.hpp"
#include "../../../webgpu/gpu.hpp"
#include "../../../webgpu/gpu_prof.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace aurora::render::detail::native::d3d12 {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr uint64_t TimestampBytes = 2 * sizeof(uint64_t);

D3D12_RESOURCE_BARRIER transition_barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after) noexcept {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  return barrier;
}
} // namespace

std::unique_ptr<StepRunner> StepRunner::create(const StepDesc& desc) noexcept {
  ID3D12Device* native = device();
  ID3D12CommandQueue* nativeQueue = queue();
  if (native == nullptr || nativeQueue == nullptr || desc.record == nullptr) {
    return nullptr;
  }
  std::unique_ptr<StepRunner> runner{new StepRunner};
  runner->m_desc = desc;
  runner->m_name = std::string{desc.name};
  runner->m_desc.name = runner->m_name;
  runner->m_reportName = "native." + runner->m_name;
  runner->m_fence = NativeFence::create(runner->m_reportName);
  if (!runner->m_fence) {
    return nullptr;
  }
  for (auto& allocator : runner->m_allocators) {
    if (FAILED(native->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) {
      Log.warn("Native step {}: failed to create a command allocator", runner->m_name);
      return nullptr;
    }
  }
  if (FAILED(native->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, runner->m_allocators[0].Get(), nullptr,
                                       IID_PPV_ARGS(&runner->m_list))) ||
      FAILED(runner->m_list->Close())) {
    Log.warn("Native step {}: failed to create its command list", runner->m_name);
    return nullptr;
  }
  const D3D12_QUERY_HEAP_DESC queryDesc{
      .Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
      .Count = static_cast<UINT>(gfx::detail::FrameSlotCount * 2),
  };
  const D3D12_HEAP_PROPERTIES heap{.Type = D3D12_HEAP_TYPE_READBACK};
  const D3D12_RESOURCE_DESC readbackDesc{
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = gfx::detail::FrameSlotCount * TimestampBytes,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_UNKNOWN,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
  };
  if (FAILED(native->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&runner->m_queries))) ||
      FAILED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                             nullptr, IID_PPV_ARGS(&runner->m_readback)))) {
    Log.warn("Native step {}: failed to create its timestamp queries", runner->m_name);
    return nullptr;
  }
  if (FAILED(nativeQueue->GetTimestampFrequency(&runner->m_frequency))) {
    runner->m_frequency = 0;
  }
  const std::wstring label(runner->m_reportName.begin(), runner->m_reportName.end());
  runner->m_list->SetName(label.c_str());
  return runner;
}

StepRunner::~StepRunner() {
  wait_idle();
  if (m_armed) {
    sentinel::disarm(sentinel::Session);
  }
}

gfx::detail::NativeStep StepRunner::step(std::span<SharedTexture* const> inputs, SharedTexture* output) noexcept {
  CHECK(inputs.size() <= MaxStepInputs, "Native step {} has {} inputs", m_name, inputs.size());
  m_inputCount = std::min(inputs.size(), MaxStepInputs);
  std::copy_n(inputs.begin(), m_inputCount, m_inputs.begin());
  m_output = output;
  return {
      .begin_inputs = &begin_inputs,
      .run = &run,
      .begin_output = &begin_output,
      .end_output = &end_output,
      .user = this,
      .name = m_name,
  };
}

void StepRunner::wait_idle() const noexcept {
  if (m_fence && !m_fence->wait_cpu(m_fence->last_signalled())) {
    Log.error("Native step {}: failed to wait for its last native list", m_name);
  }
}

bool StepRunner::begin_inputs(void* user) noexcept {
  auto& self = *static_cast<StepRunner*>(user);
  bool begun = true;
  for (size_t i = 0; i < self.m_inputCount; ++i) {
    begun = self.m_inputs[i]->begin(false) && begun;
  }
  self.m_inputsBegun = begun;
  return begun;
}

bool StepRunner::run(void* user) noexcept {
  auto& self = *static_cast<StepRunner*>(user);
  self.m_value = 0;
  std::array<std::optional<AccessFences>, MaxStepInputs> fences;
  bool ended = true;
  for (size_t i = 0; i < self.m_inputCount; ++i) {
    SharedTexture* input = self.m_inputs[i];
    fences[i] = input->accessing() ? input->end() : std::nullopt;
    ended = ended && fences[i].has_value();
  }
  const size_t oldest = self.m_frames % self.m_slotValues.size();
  for (size_t i = 0; i < self.m_slotValues.size(); ++i) {
    self.read_slot((oldest + i) % self.m_slotValues.size(), false);
  }
  if (!self.m_inputsBegun || !ended || self.m_output == nullptr) {
    self.fail("its inputs could not be handed to the native queue");
    return false;
  }
  return self.execute({fences.data(), self.m_inputCount});
}

bool StepRunner::execute(std::span<const std::optional<AccessFences>> inputs) noexcept {
  ID3D12CommandQueue* nativeQueue = queue();
  const size_t slot = m_frames++ % m_slotValues.size();
  read_slot(slot, true);
  ID3D12CommandAllocator* allocator = m_allocators[slot].Get();
  if (nativeQueue == nullptr || FAILED(allocator->Reset()) || FAILED(m_list->Reset(allocator, nullptr))) {
    fail("its command list could not be reset");
    return false;
  }
  bool waited = true;
  for (const std::optional<AccessFences>& fences : inputs) {
    waited = queue_wait(nativeQueue, *fences) && waited;
  }
  if (m_lastOut) {
    waited = queue_wait(nativeQueue, *m_lastOut) && waited;
  }
  m_lastOut.reset();
  ID3D12GraphicsCommandList* list = m_list.Get();
  bool recorded = false;
  if (waited) {
    const auto query = static_cast<UINT>(slot * 2);
    list->EndQuery(m_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query);
    transition(list, true);
    recorded = m_desc.record(m_desc.user, list);
    transition(list, false);
    list->EndQuery(m_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query + 1);
    list->ResolveQueryData(m_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query, 2, m_readback.Get(),
                           slot * TimestampBytes);
  }
  const bool closed = SUCCEEDED(list->Close());
  const bool executed = waited && closed;
  if (executed) {
    if (!std::exchange(m_armed, true)) {
      sentinel::arm(sentinel::Session);
    }
    ID3D12CommandList* lists[] = {list};
    nativeQueue->ExecuteCommandLists(1, lists);
  }
  // Signalled even when nothing ran, so the waits queued above and Dawn's wait on this value always resolve
  m_value = m_fence->signal(nativeQueue);
  for (size_t i = 0; i < m_inputCount; ++i) {
    m_inputs[i]->mark_used(*m_fence, m_value);
  }
  m_output->mark_used(*m_fence, m_value);
  m_slotValues[slot] = executed ? m_value : 0;
  if (!executed) {
    fail(waited ? "its command list failed to close" : "it could not wait for Dawn's queue");
  } else if (!recorded) {
    fail("its evaluation failed");
  }
  return executed && recorded;
}

void StepRunner::transition(ID3D12GraphicsCommandList* list, bool toStep) const noexcept {
  std::array<D3D12_RESOURCE_BARRIER, MaxStepInputs + 1> barriers{};
  UINT count = 0;
  const auto add = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES state) {
    if (state == D3D12_RESOURCE_STATE_COMMON) {
      return;
    }
    barriers[count++] = toStep ? transition_barrier(resource, D3D12_RESOURCE_STATE_COMMON, state)
                               : transition_barrier(resource, state, D3D12_RESOURCE_STATE_COMMON);
  };
  for (size_t i = 0; i < m_inputCount; ++i) {
    add(m_inputs[i]->resource(), m_desc.inputState);
  }
  add(m_output->resource(), m_desc.outputState);
  if (count != 0) {
    list->ResourceBarrier(count, barriers.data());
  }
}

void StepRunner::read_slot(size_t slot, bool wait) noexcept {
  const uint64_t value = m_slotValues[slot];
  if (value == 0 || (m_fence->completed() < value && (!wait || !m_fence->wait_cpu(value)))) {
    return;
  }
  m_slotValues[slot] = 0;
  const D3D12_RANGE range{slot * TimestampBytes, (slot + 1) * TimestampBytes};
  void* data = nullptr;
  if (m_frequency == 0 || FAILED(m_readback->Map(0, &range, &data))) {
    return;
  }
  uint64_t ticks[2]{};
  std::memcpy(ticks, static_cast<const std::byte*>(data) + range.Begin, sizeof(ticks));
  const D3D12_RANGE written{0, 0};
  m_readback->Unmap(0, &written);
  if (ticks[1] > ticks[0]) {
    const double ns = static_cast<double>(ticks[1] - ticks[0]) * 1e9 / static_cast<double>(m_frequency);
    webgpu::gpu_prof::report_external(m_reportName, static_cast<uint64_t>(ns));
  }
}

void StepRunner::fail(std::string_view reason) noexcept {
  if (std::exchange(m_failed, true)) {
    return;
  }
  Log.warn("Native step {} failed: {}; its frames compose without it", m_name, reason);
  if (m_desc.failed != nullptr) {
    m_desc.failed(m_desc.user);
  }
}

void StepRunner::begin_output(void* user, bool ok) noexcept {
  auto& self = *static_cast<StepRunner*>(user);
  if (self.m_output == nullptr) {
    return;
  }
  bool begun = false;
  if (self.m_value != 0) {
    const std::array fences{self.m_fence->dawn()};
    const std::array values{self.m_value};
    begun = self.m_output->begin(ok, fences, values);
  } else {
    begun = self.m_output->begin(false);
  }
  if (!begun) {
    self.fail("its output could not be handed back to Dawn");
  }
}

void StepRunner::end_output(void* user) noexcept {
  auto& self = *static_cast<StepRunner*>(user);
  if (self.m_output != nullptr && self.m_output->accessing()) {
    self.m_lastOut = self.m_output->end();
  }
}

namespace test_step {
namespace {
struct State {
  int mode = -1;
  bool failed = false;
  std::unique_ptr<StepRunner> runner;
  std::unique_ptr<SharedTexture> in;
  std::unique_ptr<SharedTexture> out;
};

State g_state;

[[maybe_unused]] bool record_copy(void* user, ID3D12GraphicsCommandList* list) noexcept {
  const auto& state = *static_cast<const State*>(user);
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = state.out->resource();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = state.in->resource();
  source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  return true;
}

[[maybe_unused]] void copy_failed(void* user) noexcept { static_cast<State*>(user)->failed = true; }

[[maybe_unused]] void teardown() noexcept {
  g_state.runner.reset();
  g_state.out.reset();
  g_state.in.reset();
  g_state.failed = true;
}

[[maybe_unused]] bool ensure_targets(const webgpu::TextureWithSampler& scene) noexcept {
  if (g_state.in && g_state.in->texture().GetWidth() == scene.size.width &&
      g_state.in->texture().GetHeight() == scene.size.height && g_state.in->format() == scene.format) {
    return true;
  }
  if (g_state.runner) {
    g_state.runner->wait_idle();
  }
  g_state.out.reset();
  g_state.in.reset();
  g_state.in = SharedTexture::create(
      {.format = scene.format, .width = scene.size.width, .height = scene.size.height, .label = "native.copy.in"});
  g_state.out = SharedTexture::create(
      {.format = scene.format, .width = scene.size.width, .height = scene.size.height, .label = "native.copy.out"});
  if (!g_state.runner) {
    g_state.runner = StepRunner::create({
        .name = "copy",
        .inputState = D3D12_RESOURCE_STATE_COPY_SOURCE,
        .outputState = D3D12_RESOURCE_STATE_COPY_DEST,
        .record = &record_copy,
        .failed = &copy_failed,
        .user = &g_state,
    });
  }
  register_teardown(&teardown);
  return g_state.in && g_state.out && g_state.runner;
}
} // namespace

void encode([[maybe_unused]] wgpu::CommandEncoder& cmd) noexcept {
#ifndef NDEBUG
  if (g_state.mode < 0) {
    const char* value = std::getenv("AURORA_NATIVE_TEST_STEP");
    g_state.mode = value != nullptr && std::string_view{value} == "copy" ? 1 : 0;
    if (g_state.mode == 1) {
      Log.warn("AURORA_NATIVE_TEST_STEP=copy: a native copy stands in for the resolver over the split");
    }
  }
  const Caps& deviceCaps = caps();
  if (g_state.mode != 1 || g_state.failed || !available() || !deviceCaps.d3d12SharedTexture ||
      !deviceCaps.d3d12SharedFence) {
    return;
  }
  const webgpu::TextureWithSampler& scene = webgpu::scene_color();
  if (!scene.texture) {
    return;
  }
  if (!ensure_targets(scene)) {
    Log.warn("AURORA_NATIVE_TEST_STEP=copy: its shared textures or native step could not be created");
    g_state.failed = true;
    return;
  }
  const std::array inputs{g_state.in.get()};
  upscale::compose::request_native_step(g_state.runner->step(inputs, g_state.out.get()),
                                        {.view = g_state.out->view(), .format = g_state.out->format()});
  const wgpu::TexelCopyTextureInfo source{.texture = scene.texture};
  const wgpu::TexelCopyTextureInfo destination{.texture = g_state.in->texture()};
  const wgpu::Extent3D extent{scene.size.width, scene.size.height, 1};
  cmd.CopyTextureToTexture(&source, &destination, &extent);
#endif
}
} // namespace test_step
} // namespace aurora::render::detail::native::d3d12

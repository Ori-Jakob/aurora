#include "shared_texture.hpp"

#include "teardown.hpp"

#include "../../../internal.hpp"
#include "../../../webgpu/error_scope.hpp"
#include "../../../webgpu/gpu.hpp"

#include <dawn/native/D3D12Backend.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <utility>

namespace aurora::render::detail::native::d3d12 {
namespace {
constexpr Module Log{"aurora::render::native"};

using Microsoft::WRL::ComPtr;

struct DawnFence {
  wgpu::SharedFence fence;
  ComPtr<ID3D12Fence> native;
};

std::mutex g_mutex;
std::vector<SharedTexture*> g_textures;
std::vector<NativeFence*> g_fences;
std::vector<DawnFence> g_dawnFences;
uint64_t g_sharedBytes = 0;

DXGI_FORMAT dxgi_format(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::BGRA8Unorm:
    return DXGI_FORMAT_B8G8R8A8_UNORM;
  case wgpu::TextureFormat::RGBA8Unorm:
    return DXGI_FORMAT_R8G8B8A8_UNORM;
  case wgpu::TextureFormat::RGBA16Float:
    return DXGI_FORMAT_R16G16B16A16_FLOAT;
  case wgpu::TextureFormat::RG16Float:
    return DXGI_FORMAT_R16G16_FLOAT;
  case wgpu::TextureFormat::R8Unorm:
    return DXGI_FORMAT_R8_UNORM;
  case wgpu::TextureFormat::R32Float:
    return DXGI_FORMAT_R32_FLOAT;
  default:
    return DXGI_FORMAT_UNKNOWN;
  }
}

wgpu::StringView string_view(std::string_view text) noexcept { return {text.data(), text.size()}; }

void set_native_name(ID3D12Object* object, std::string_view label) noexcept {
  const std::wstring name(label.begin(), label.end());
  object->SetName(name.c_str());
}

bool wait_value(ID3D12Fence* fence, uint64_t value, HANDLE event) noexcept {
  if (fence == nullptr || value == 0 || fence->GetCompletedValue() >= value) {
    return true;
  }
  if (event == nullptr || FAILED(fence->SetEventOnCompletion(value, event))) {
    return false;
  }
  return WaitForSingleObject(event, INFINITE) == WAIT_OBJECT_0;
}

void adjust_reservation(uint64_t add, uint64_t remove) noexcept {
  uint64_t total = 0;
  {
    std::lock_guard lock{g_mutex};
    g_sharedBytes = g_sharedBytes + add - std::min(remove, g_sharedBytes + add);
    total = g_sharedBytes;
  }
  const uint64_t reserved = reserve_external_memory(total);
  if (reserved < total) {
    Log.warn("Native D3D12: reserved {} of {} bytes of external memory", reserved, total);
  }
}

void release_shared() noexcept {
  std::vector<SharedTexture*> textures;
  std::vector<NativeFence*> fences;
  {
    std::lock_guard lock{g_mutex};
    textures = g_textures;
    fences = g_fences;
  }
  for (SharedTexture* texture : textures) {
    texture->release();
  }
  for (NativeFence* fence : fences) {
    fence->release();
  }
  std::lock_guard lock{g_mutex};
  g_dawnFences.clear();
}

ID3D12Fence* native_fence(const wgpu::SharedFence& fence) noexcept {
  {
    std::lock_guard lock{g_mutex};
    const auto it =
        std::ranges::find_if(g_dawnFences, [&](const DawnFence& entry) { return entry.fence.Get() == fence.Get(); });
    if (it != g_dawnFences.end()) {
      return it->native.Get();
    }
  }
  wgpu::SharedFenceDXGISharedHandleExportInfo handleInfo{};
  wgpu::SharedFenceExportInfo info{};
  info.nextInChain = &handleInfo;
  fence.ExportInfo(&info);
  ComPtr<ID3D12Fence> opened;
  if (device() == nullptr || info.type != wgpu::SharedFenceType::DXGISharedHandle || handleInfo.handle == nullptr ||
      FAILED(device()->OpenSharedHandle(handleInfo.handle, IID_PPV_ARGS(&opened)))) {
    return nullptr;
  }
  ID3D12Fence* result = opened.Get();
  {
    std::lock_guard lock{g_mutex};
    g_dawnFences.push_back({fence, std::move(opened)});
  }
  register_teardown(&release_shared);
  return result;
}
} // namespace

std::unique_ptr<NativeFence> NativeFence::create(std::string_view label) noexcept {
  ID3D12Device* native = device();
  if (native == nullptr) {
    return nullptr;
  }
  std::unique_ptr<NativeFence> result{new NativeFence};
  if (FAILED(native->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result->m_fence)))) {
    Log.warn("Native D3D12: failed to create fence {}", label);
    return nullptr;
  }
  set_native_name(result->m_fence.Get(), label);
  HANDLE handle = nullptr;
  if (FAILED(native->CreateSharedHandle(result->m_fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) {
    Log.warn("Native D3D12: failed to share fence {}", label);
    return nullptr;
  }
  wgpu::SharedFenceDXGISharedHandleDescriptor handleDescriptor{};
  handleDescriptor.handle = handle;
  const wgpu::SharedFenceDescriptor descriptor{.nextInChain = &handleDescriptor, .label = string_view(label)};
  std::optional<std::string> error;
  {
    std::lock_guard lock{webgpu::error_scope_mutex()};
    webgpu::push_error_scopes();
    result->m_dawn = webgpu::g_device.ImportSharedFence(&descriptor);
    error = webgpu::pop_error_scopes();
  }
  CloseHandle(handle);
  if (error || !result->m_dawn) {
    Log.warn("Native D3D12: Dawn could not import fence {}: {}", label, error.value_or("no fence"));
    return nullptr;
  }
  result->m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (result->m_event == nullptr) {
    Log.warn("Native D3D12: failed to create the wait event for fence {}", label);
    return nullptr;
  }
  {
    std::lock_guard lock{g_mutex};
    g_fences.push_back(result.get());
  }
  register_teardown(&release_shared);
  return result;
}

NativeFence::~NativeFence() {
  release();
  std::lock_guard lock{g_mutex};
  std::erase(g_fences, this);
}

void NativeFence::release() noexcept {
  m_dawn = nullptr;
  m_fence.Reset();
  if (m_event != nullptr) {
    CloseHandle(m_event);
    m_event = nullptr;
  }
}

uint64_t NativeFence::signal(ID3D12CommandQueue* queue) noexcept {
  ++m_value;
  if (queue == nullptr || !m_fence || FAILED(queue->Signal(m_fence.Get(), m_value))) {
    Log.error("Native D3D12: failed to signal fence value {}", m_value);
  }
  return m_value;
}

uint64_t NativeFence::completed() const noexcept { return m_fence ? m_fence->GetCompletedValue() : 0; }

bool NativeFence::wait_cpu(uint64_t value) const noexcept { return wait_value(m_fence.Get(), value, m_event); }

std::unique_ptr<SharedTexture> SharedTexture::create(const Desc& desc) noexcept {
  const DXGI_FORMAT format = dxgi_format(desc.format);
  CHECK(format != DXGI_FORMAT_UNKNOWN, "Native D3D12: format {} cannot be shared", static_cast<uint32_t>(desc.format));
  ID3D12Device* native = device();
  if (format == DXGI_FORMAT_UNKNOWN || native == nullptr || desc.width == 0 || desc.height == 0) {
    return nullptr;
  }
  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
  if (desc.renderTarget) {
    flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  }
  if (desc.unorderedAccess) {
    flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  }
  const D3D12_RESOURCE_DESC resourceDesc{
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = desc.width,
      .Height = desc.height,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = format,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      .Flags = flags,
  };
  const D3D12_HEAP_PROPERTIES heap{.Type = D3D12_HEAP_TYPE_DEFAULT};
  std::unique_ptr<SharedTexture> result{new SharedTexture};
  if (FAILED(native->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&result->m_resource)))) {
    Log.warn("Native D3D12: failed to create {} ({}x{})", desc.label, desc.width, desc.height);
    return nullptr;
  }
  set_native_name(result->m_resource.Get(), desc.label);
  std::optional<std::string> error;
  {
    std::lock_guard lock{webgpu::error_scope_mutex()};
    webgpu::push_error_scopes();
    dawn::native::d3d12::SharedTextureMemoryD3D12ResourceDescriptor nativeDescriptor;
    nativeDescriptor.resource = result->m_resource;
    const wgpu::SharedTextureMemoryDescriptor descriptor{.nextInChain = &nativeDescriptor,
                                                         .label = string_view(desc.label)};
    result->m_memory = webgpu::g_device.ImportSharedTextureMemory(&descriptor);
    if (result->m_memory) {
      result->m_texture = result->m_memory.CreateTexture();
    }
    if (result->m_texture) {
      result->m_view = result->m_texture.CreateView();
    }
    error = webgpu::pop_error_scopes();
  }
  if (error || !result->m_view) {
    Log.warn("Native D3D12: Dawn could not import {}: {}", desc.label, error.value_or("no texture"));
    return nullptr;
  }
  result->m_format = desc.format;
  result->m_bytes = native->GetResourceAllocationInfo(0, 1, &resourceDesc).SizeInBytes;
  {
    std::lock_guard lock{g_mutex};
    g_textures.push_back(result.get());
  }
  register_teardown(&release_shared);
  adjust_reservation(result->m_bytes, 0);
  return result;
}

SharedTexture::~SharedTexture() {
  release();
  std::lock_guard lock{g_mutex};
  std::erase(g_textures, this);
}

bool SharedTexture::begin(bool initialized, std::span<const wgpu::SharedFence> fences,
                          std::span<const uint64_t> values) noexcept {
  CHECK(!m_accessing, "Native D3D12: BeginAccess on a texture that is already accessed");
  CHECK(fences.size() == values.size(), "Native D3D12: {} fences with {} values", fences.size(), values.size());
  if (!m_memory || m_accessing || fences.size() != values.size()) {
    return false;
  }
  const wgpu::SharedTextureMemoryBeginAccessDescriptor descriptor{
      .concurrentRead = false,
      .initialized = initialized,
      .fenceCount = fences.size(),
      .fences = fences.data(),
      .signaledValueCount = values.size(),
      .signaledValues = values.data(),
  };
  if (!m_memory.BeginAccess(m_texture, &descriptor)) {
    Log.warn("Native D3D12: BeginAccess failed");
    return false;
  }
  m_accessing = true;
  return true;
}

std::optional<AccessFences> SharedTexture::end() noexcept {
  if (!m_memory || !m_accessing) {
    return std::nullopt;
  }
  m_accessing = false;
  wgpu::SharedTextureMemoryEndAccessState state;
  if (!m_memory.EndAccess(m_texture, &state)) {
    Log.warn("Native D3D12: EndAccess failed");
    return std::nullopt;
  }
  AccessFences result;
  result.fences.assign(state.fences, state.fences + state.fenceCount);
  result.values.assign(state.signaledValues, state.signaledValues + state.signaledValueCount);
  return result;
}

void SharedTexture::mark_used(const NativeFence& fence, uint64_t value) noexcept {
  m_lastFence = fence.get();
  m_lastUse = value;
}

void SharedTexture::release() noexcept {
  if (!m_resource) {
    return;
  }
  if (m_lastFence && m_lastFence->GetCompletedValue() < m_lastUse) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!wait_value(m_lastFence.Get(), m_lastUse, event)) {
      Log.error("Native D3D12: freeing a shared texture before its last native use completed");
    }
    if (event != nullptr) {
      CloseHandle(event);
    }
  }
  if (m_accessing) {
    end();
  }
  m_view = nullptr;
  m_texture = nullptr;
  m_memory = nullptr;
  m_resource.Reset();
  m_lastFence.Reset();
  m_lastUse = 0;
  adjust_reservation(0, std::exchange(m_bytes, 0));
}

bool queue_wait(ID3D12CommandQueue* queue, const AccessFences& fences) noexcept {
  if (queue == nullptr || fences.fences.size() != fences.values.size()) {
    return false;
  }
  for (size_t i = 0; i < fences.fences.size(); ++i) {
    ID3D12Fence* fence = native_fence(fences.fences[i]);
    if (fence == nullptr || FAILED(queue->Wait(fence, fences.values[i]))) {
      Log.warn("Native D3D12: cannot wait on Dawn's fence value {}", fences.values[i]);
      return false;
    }
  }
  return true;
}

uint64_t shared_bytes() noexcept {
  std::lock_guard lock{g_mutex};
  return g_sharedBytes;
}
} // namespace aurora::render::detail::native::d3d12

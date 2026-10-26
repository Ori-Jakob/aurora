#include "device.hpp"

#include "../../../logging.hpp"

#include <dawn/native/D3D12Backend.h>
#include <SDL3/SDL_stdinc.h>

#include <string_view>
#include <utility>

namespace aurora::render::detail::native::d3d12 {
namespace {
constexpr Module Log{"aurora::render::native"};

using Microsoft::WRL::ComPtr;

struct DeviceState {
  wgpu::Device dawnDevice;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC1 adapterDesc{};
  uint64_t driverVersion = 0;
};

DeviceState g_state;

bool is_d3d12(const wgpu::Adapter& adapter) noexcept {
  wgpu::AdapterInfo info{};
  return adapter.GetInfo(&info) == wgpu::Status::Success && info.backendType == wgpu::BackendType::D3D12;
}
} // namespace

bool killed() noexcept {
  static const bool value = [] {
    const char* env = SDL_getenv("AURORA_NATIVE_D3D12");
    if (env == nullptr || std::string_view{env} != "off") {
      return false;
    }
    Log.warn("AURORA_NATIVE_D3D12=off: native D3D12 interop disabled");
    return true;
  }();
  return value;
}

bool initialize(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept {
  shutdown();
  if (killed() || !device || !adapter || !is_d3d12(adapter)) {
    return false;
  }
  DeviceState state{
      .dawnDevice = device,
      .device = dawn::native::d3d12::GetD3D12Device(device.Get()),
      .queue = dawn::native::d3d12::GetD3D12CommandQueue(device.Get()),
  };
  const ComPtr<IDXGIAdapter> dxgiAdapter = dawn::native::d3d::GetDXGIAdapter(adapter.Get());
  ComPtr<IDXGIAdapter1> adapter1;
  if (!state.device || !state.queue || !dxgiAdapter || FAILED(dxgiAdapter.As(&adapter1)) ||
      FAILED(adapter1->GetDesc1(&state.adapterDesc))) {
    Log.warn("Native D3D12 interop unavailable: Dawn exposed no D3D12 device, queue or adapter");
    return false;
  }
  LARGE_INTEGER umd{};
  if (SUCCEEDED(dxgiAdapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd))) {
    state.driverVersion = static_cast<uint64_t>(umd.QuadPart);
  }
  state.adapter = dxgiAdapter;
  const DXGI_ADAPTER_DESC1& desc = state.adapterDesc;
  const uint64_t v = state.driverVersion;
  Log.info(
      "Native D3D12: vendor {:#06x}, device {:#06x}, subsystem {:#010x}, revision {}, driver {}.{}.{}.{}, "
      "LUID {:08x}:{:08x}",
      desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision, v >> 48, (v >> 32) & 0xFFFF, (v >> 16) & 0xFFFF,
      v & 0xFFFF, static_cast<uint32_t>(desc.AdapterLuid.HighPart), desc.AdapterLuid.LowPart);
  g_state = std::move(state);
  return true;
}

void shutdown() noexcept { g_state = {}; }

bool available() noexcept { return g_state.device != nullptr; }

ID3D12Device* device() noexcept { return g_state.device.Get(); }

ID3D12CommandQueue* queue() noexcept { return g_state.queue.Get(); }

IDXGIAdapter* dxgi_adapter() noexcept { return g_state.adapter.Get(); }

const DXGI_ADAPTER_DESC1& adapter_desc() noexcept { return g_state.adapterDesc; }

uint64_t driver_version() noexcept { return g_state.driverVersion; }

uint64_t reserve_external_memory(uint64_t bytes) noexcept {
  if (!g_state.dawnDevice) {
    return 0;
  }
  return dawn::native::d3d12::SetExternalMemoryReservation(g_state.dawnDevice.Get(), bytes,
                                                           dawn::native::d3d12::MemorySegment::Local);
}
} // namespace aurora::render::detail::native::d3d12

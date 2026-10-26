#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <dxgi1_4.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>

namespace aurora::render::detail::native::d3d12 {
bool killed() noexcept;
bool initialize(const wgpu::Device& device, const wgpu::Adapter& adapter) noexcept;
void shutdown() noexcept;
bool available() noexcept;
ID3D12Device* device() noexcept;
ID3D12CommandQueue* queue() noexcept;
IDXGIAdapter* dxgi_adapter() noexcept;
const DXGI_ADAPTER_DESC1& adapter_desc() noexcept;
uint64_t driver_version() noexcept;
uint64_t reserve_external_memory(uint64_t bytes) noexcept;
} // namespace aurora::render::detail::native::d3d12

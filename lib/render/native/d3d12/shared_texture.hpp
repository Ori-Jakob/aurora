#pragma once

#include "device.hpp"

#include <wrl/client.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::render::detail::native::d3d12 {
struct AccessFences {
  std::vector<wgpu::SharedFence> fences;
  std::vector<uint64_t> values;
};

class NativeFence {
public:
  static std::unique_ptr<NativeFence> create(std::string_view label) noexcept;
  ~NativeFence();
  NativeFence(const NativeFence&) = delete;
  NativeFence& operator=(const NativeFence&) = delete;

  uint64_t signal(ID3D12CommandQueue* queue) noexcept;
  [[nodiscard]] uint64_t last_signalled() const noexcept { return m_value; }
  [[nodiscard]] uint64_t completed() const noexcept;
  bool wait_cpu(uint64_t value) const noexcept;
  [[nodiscard]] ID3D12Fence* get() const noexcept { return m_fence.Get(); }
  [[nodiscard]] const wgpu::SharedFence& dawn() const noexcept { return m_dawn; }
  void release() noexcept;

private:
  NativeFence() = default;

  Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
  wgpu::SharedFence m_dawn;
  HANDLE m_event = nullptr;
  uint64_t m_value = 0;
};

class SharedTexture {
public:
  struct Desc {
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    uint32_t width = 0;
    uint32_t height = 0;
    bool renderTarget = false;
    bool unorderedAccess = false;
    std::string_view label;
  };

  static std::unique_ptr<SharedTexture> create(const Desc& desc) noexcept;
  ~SharedTexture();
  SharedTexture(const SharedTexture&) = delete;
  SharedTexture& operator=(const SharedTexture&) = delete;

  bool begin(bool initialized, std::span<const wgpu::SharedFence> fences = {},
             std::span<const uint64_t> values = {}) noexcept;
  std::optional<AccessFences> end() noexcept;
  void mark_used(const NativeFence& fence, uint64_t value) noexcept;
  void release() noexcept;

  [[nodiscard]] ID3D12Resource* resource() const noexcept { return m_resource.Get(); }
  [[nodiscard]] const wgpu::Texture& texture() const noexcept { return m_texture; }
  [[nodiscard]] const wgpu::TextureView& view() const noexcept { return m_view; }
  [[nodiscard]] wgpu::TextureFormat format() const noexcept { return m_format; }
  [[nodiscard]] uint64_t bytes() const noexcept { return m_bytes; }
  [[nodiscard]] bool accessing() const noexcept { return m_accessing; }

private:
  SharedTexture() = default;

  Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;
  wgpu::SharedTextureMemory m_memory;
  wgpu::Texture m_texture;
  wgpu::TextureView m_view;
  wgpu::TextureFormat m_format = wgpu::TextureFormat::Undefined;
  Microsoft::WRL::ComPtr<ID3D12Fence> m_lastFence;
  uint64_t m_lastUse = 0;
  uint64_t m_bytes = 0;
  bool m_accessing = false;
};

bool queue_wait(ID3D12CommandQueue* queue, const AccessFences& fences) noexcept;
uint64_t shared_bytes() noexcept;
} // namespace aurora::render::detail::native::d3d12

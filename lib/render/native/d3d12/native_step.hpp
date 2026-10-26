#pragma once

#include "shared_texture.hpp"

#include "../../../gfx/frame.hpp"

#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace aurora::render::detail::native::d3d12 {
inline constexpr size_t MaxStepInputs = 4;

struct StepDesc {
  std::string_view name;
  D3D12_RESOURCE_STATES inputState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  D3D12_RESOURCE_STATES outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  bool (*record)(void* user, ID3D12GraphicsCommandList* list) noexcept = nullptr;
  void (*failed)(void* user) noexcept = nullptr;
  void* user = nullptr;
};

// Render worker only
class StepRunner {
public:
  static std::unique_ptr<StepRunner> create(const StepDesc& desc) noexcept;
  ~StepRunner();
  StepRunner(const StepRunner&) = delete;
  StepRunner& operator=(const StepRunner&) = delete;

  [[nodiscard]] gfx::detail::NativeStep step(std::span<SharedTexture* const> inputs, SharedTexture* output) noexcept;
  void wait_idle() const noexcept;
  [[nodiscard]] bool failed() const noexcept { return m_failed; }
  [[nodiscard]] const NativeFence& fence() const noexcept { return *m_fence; }

private:
  StepRunner() = default;

  static bool begin_inputs(void* user) noexcept;
  static bool run(void* user) noexcept;
  static void begin_output(void* user, bool ok) noexcept;
  static void end_output(void* user) noexcept;

  bool execute(std::span<const std::optional<AccessFences>> inputs) noexcept;
  void transition(ID3D12GraphicsCommandList* list, bool toStep) const noexcept;
  void read_slot(size_t slot, bool wait) noexcept;
  void fail(std::string_view reason) noexcept;

  StepDesc m_desc;
  std::string m_name;
  std::string m_reportName;
  std::array<SharedTexture*, MaxStepInputs> m_inputs{};
  size_t m_inputCount = 0;
  SharedTexture* m_output = nullptr;
  std::unique_ptr<NativeFence> m_fence;
  std::array<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>, gfx::detail::FrameSlotCount> m_allocators;
  std::array<uint64_t, gfx::detail::FrameSlotCount> m_slotValues{};
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_list;
  Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_queries;
  Microsoft::WRL::ComPtr<ID3D12Resource> m_readback;
  std::optional<AccessFences> m_lastOut;
  uint64_t m_frequency = 0;
  uint64_t m_frames = 0;
  uint64_t m_value = 0;
  bool m_inputsBegun = false;
  bool m_armed = false;
  bool m_failed = false;
};
} // namespace aurora::render::detail::native::d3d12

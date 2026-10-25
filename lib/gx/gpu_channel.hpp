#pragma once

#include <aurora/gfx.hpp>
#include <aurora/gpu_systems.hpp>
#include <dolphin/gx/GXAurora.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace aurora::gx::gpu_channel {
enum class HandleState : uint8_t { Unknown, Failed, Uploaded, Reserved, Written };
struct HandleInfo {
  gfx::Range range;
  HandleState state = HandleState::Unknown;
};
struct KernelCreate {
  std::string_view entryPoint;
  std::array<uint32_t, 3> workgroupSize;
  uint32_t flags;
};
struct SystemCreate {
  uint64_t id;
  std::string_view name, source;
  bool library;
  std::span<const KernelCreate> kernels;
  uint32_t persistentBytes, flags, requirements;
};
struct DispatchDecode {
  uint64_t system;
  uint32_t kernel;
  gfx::Range output;
  std::span<const gfx::Range> inputs;
  std::array<uint32_t, 3> elements;
  std::span<const uint8_t> params;
};
// Swapped only while the FIFO is drained, like hints::set_sink
struct SystemBackend {
  void* userData = nullptr;
  void (*create)(void*, const SystemCreate&) noexcept = nullptr;
  void (*destroy)(void*, uint64_t id) noexcept = nullptr;
  void (*reset)(void*) noexcept = nullptr;
  render::gpu::SystemStatus (*status)(void*, uint64_t id) noexcept = nullptr;
  void (*wait_all)(void*) noexcept = nullptr;
  bool (*dispatch)(void*, const DispatchDecode&) noexcept = nullptr;
  bool (*update)(void*, uint64_t id, uint32_t byteOffset, std::span<const uint8_t> bytes) noexcept = nullptr;
};
void set_system_backend(const SystemBackend* backend) noexcept;
constexpr bool is_command(uint16_t id) noexcept {
  return id >= GX_AURORA_GPU_UPLOAD && id <= GX_AURORA_GPU_DESTROY_SYSTEM;
}
bool valid_system_name(std::string_view name) noexcept;
bool valid_entry_point(std::string_view entry) noexcept;
bool valid_workgroup(uint32_t x, uint32_t y, uint32_t z) noexcept;
bool handle_is_current(uint64_t handle) noexcept;
uint64_t bind_handle(uint64_t handle) noexcept;
void warn_bad_attribute(uint32_t attr) noexcept;
uint32_t current_serial() noexcept;
bool on_upload(uint64_t handle, std::span<const uint8_t> bytes) noexcept;
bool on_reserve(uint64_t handle, uint32_t bytes, uint32_t flags) noexcept;
bool on_create(const SystemCreate& desc, std::span<const uint8_t> payload) noexcept;
bool on_update(uint64_t id, uint32_t byteOffset, std::span<const uint8_t> bytes) noexcept;
bool on_destroy(uint64_t id) noexcept;
bool on_dispatch(uint64_t system, uint32_t kernel, uint64_t output, std::span<const uint64_t> inputs,
                 std::array<uint32_t, 3> elements, std::span<const uint8_t> params) noexcept;
HandleInfo lookup(uint64_t handle) noexcept;
bool mark_written(uint64_t handle) noexcept;
bool mark_failed(uint64_t handle) noexcept;
std::span<const uint8_t> shadow(uint64_t id) noexcept;
void note_bind(bool valid) noexcept;
void note_suppressed_draw() noexcept;
void end_frame() noexcept;
void reset() noexcept;
void reset_decoder() noexcept;
void wait_systems() noexcept;
void write_preamble(std::vector<uint8_t>& out);
bool uncaptured_gpu_state() noexcept;
uint64_t preamble_flush_bytes() noexcept;
render::gpu::Stats stats() noexcept;

namespace testing {
void reset_stats() noexcept;
void set_serial(uint32_t serial) noexcept;
void set_frame_caps(uint64_t channelBytes, uint64_t commandBytes, uint32_t handles) noexcept;
} // namespace testing
} // namespace aurora::gx::gpu_channel

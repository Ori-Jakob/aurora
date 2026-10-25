#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace aurora::render::gpu {
using Handle = uint64_t;
using SystemId = uint64_t;
inline constexpr Handle InvalidHandle = 0;
inline constexpr SystemId InvalidSystem = 0;
enum ReserveFlags : uint32_t { ReserveZero = 1u << 0 };
enum SystemFlags : uint32_t { SystemGpuOnlyState = 1u << 0 };
enum KernelFlags : uint32_t { KernelStateAtomic = 1u << 0 };
enum SystemRequirements : uint32_t { RequiresCore = 1u << 0, RequiresSubgroups = 1u << 1, RequiresShaderF16 = 1u << 2 };
enum class SystemStatus : uint8_t { Invalid, Pending, Ready, Failed, Unavailable };
inline constexpr uint32_t MaxUploadBytes = 16u << 20;
inline constexpr uint32_t MaxFrameHandles = 1u << 16;
inline constexpr uint64_t MaxFrameChannelBytes = 256ull << 20;
inline constexpr uint64_t MaxFrameCommandBytes = 256ull << 20;
inline constexpr uint32_t MaxPersistentBytes = 16u << 20;
inline constexpr uint64_t MaxShadowBytes = 64ull << 20;
inline constexpr uint32_t MaxGpuOnlyStateBytes = 256u << 20;
inline constexpr uint32_t MaxSystems = 256;
inline constexpr uint32_t MaxKernels = 8;
inline constexpr uint32_t MaxWorkgroupInvocations = 128;
inline constexpr uint32_t MaxSystemSourceBytes = 256u << 10;
inline constexpr uint32_t MaxSystemNameBytes = 31;
inline constexpr uint32_t MaxKernelNameBytes = 63;
inline constexpr uint32_t MaxDispatchInputs = 8;
inline constexpr uint32_t MaxDispatchParamsSize = 128;
inline constexpr uint32_t MaxFrameDispatches = 4096;

struct KernelDesc {
  std::string_view entryPoint;
  std::array<uint32_t, 3> workgroupSize{64, 1, 1};
  uint32_t flags = 0;
};

struct SystemDesc {
  std::string_view name;
  std::string_view wgsl;
  std::string_view library;
  std::span<const KernelDesc> kernels;
  uint32_t persistentBytes = 0;
  uint32_t flags = 0;
  uint32_t requirements = 0;
};

struct Stats {
  uint64_t refused = 0, refusedBytes = 0, staleBinds = 0;
  uint64_t uploads = 0, uploadBytes = 0, reserves = 0, reserveBytes = 0;
  uint64_t failedHandles = 0, storageMisses = 0, rejected = 0;
  uint64_t binds = 0, invalidBinds = 0, suppressedDraws = 0;
  uint64_t created = 0, destroyed = 0, updates = 0, updateBytes = 0;
  uint32_t liveSystems = 0;
  uint64_t dispatches = 0, dispatchesRefused = 0, dispatchesRejected = 0, persistentCopies = 0;
  uint64_t binderFailures = 0, droppedCopies = 0;
};

bool systems_available() noexcept;
void request_capacity(uint64_t channelBytesPerFrame) noexcept;
Handle upload(const void* data, uint32_t bytes) noexcept;
Handle reserve(uint32_t bytes, uint32_t flags = 0) noexcept;
SystemId create_system(const SystemDesc& desc) noexcept;
void update_persistent(SystemId id, uint32_t byteOffset, const void* data, uint32_t bytes) noexcept;
void destroy_system(SystemId id) noexcept;
SystemStatus system_status(SystemId id) noexcept;
bool dispatch(SystemId system, uint32_t kernel, std::span<const Handle> inputs, Handle output,
              std::array<uint32_t, 3> elements, std::span<const uint8_t> params = {}) noexcept;
Stats stats() noexcept;
} // namespace aurora::render::gpu

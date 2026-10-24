#pragma once

#include "../internal.hpp"

#include <aurora/math.hpp>
#include <aurora/render.hpp>
#include <dolphin/gx/GXAurora.h>

#include <cstdint>
#include <span>
#include <string_view>

namespace aurora::gx::hints {
inline constexpr uint16_t FirstId = GX_AURORA_HINT_FIRST;
inline constexpr uint16_t LastId = GX_AURORA_HINT_LAST;
inline constexpr size_t IdCount = LastId - FirstId + 1;

struct Camera {
  AuroraCameraRole role = AURORA_CAMERA_MAIN;
  Mat3x4<float> view{};
  Mat4x4<float> proj{};
  GXProjectionType type = GX_PERSPECTIVE;
  uint32_t cutEpoch = 0;
  // What GXState::proj holds after GXSetProjection(proj, type), so a projection match can compare bits
  Mat4x4<float> gxProj{};
};

struct ObjectBegin {
  uint64_t id = 0;
  uint32_t subId = 0;
  uint32_t generation = 0;
  uint32_t categories = 0;
  uint32_t flags = 0;
  bool hasWorld = false;
  Mat3x4<float> world{};
};

struct ScopePush {
  uint32_t categories = 0;
  uint32_t flags = 0;
  AuroraCameraRole role = AURORA_CAMERA_INHERIT;
};

struct Skeleton {
  uint64_t objectId = 0;
  std::span<const Mat3x4<float>> joints;
};

struct Environment {
  uint16_t version = 0;
  AuroraEnvironment value{};
};

// Each callback runs on the thread that decodes the FIFO, in stream order with the draws around it
struct Sink {
  void* userData = nullptr;
  void (*reset)(void* userData) noexcept = nullptr;
  void (*camera)(void* userData, const Camera& camera) noexcept = nullptr;
  void (*objectBegin)(void* userData, const ObjectBegin& object) noexcept = nullptr;
  void (*objectEnd)(void* userData) noexcept = nullptr;
  void (*scopePush)(void* userData, const ScopePush& scope) noexcept = nullptr;
  void (*scopePop)(void* userData) noexcept = nullptr;
  void (*stage)(void* userData, render::Stage stage) noexcept = nullptr;
  void (*lights)(void* userData, std::span<const AuroraLight> lights) noexcept = nullptr;
  void (*environment)(void* userData, const Environment& environment) noexcept = nullptr;
  void (*material)(void* userData, uint32_t materialClass) noexcept = nullptr;
  void (*skeleton)(void* userData, const Skeleton& skeleton) noexcept = nullptr;
  void (*sceneEpoch)(void* userData, uint32_t epoch) noexcept = nullptr;
  void (*tevPolicy)(void* userData, const AuroraTevPolicy& policy) noexcept = nullptr;
  void (*influencers)(void* userData, std::span<const AuroraInfluencer> influencers) noexcept = nullptr;
  void (*windSources)(void* userData, std::span<const AuroraWindSource> sources) noexcept = nullptr;
};

struct Counters {
  uint64_t decoded = 0;
  uint64_t skipped = 0;
  uint64_t malformed = 0;
  uint64_t late = 0;
  uint64_t bytes = 0;
};

// Swapped only while the FIFO is drained, like guest_memory::set_hooks
void set_sink(const Sink* sink) noexcept;
const Sink* sink() noexcept;
void reset_state() noexcept;

// Consumes the 4-byte size and the payload; the caller has already read the ID
void handle(uint16_t id, ByteReader& reader) noexcept;

Counters counters(uint16_t id) noexcept;
uint64_t total_bytes() noexcept;
uint64_t gpu_command_bytes() noexcept;
void count_late(uint16_t id) noexcept;
// Emitters run on the game thread, so their warn-once flags are kept apart from the decoder's
bool emitter_warned(uint16_t id) noexcept;
void warn_emitter(uint16_t id, std::string_view problem) noexcept;
std::string_view name(uint16_t id) noexcept;
uint16_t latest_version(uint16_t id) noexcept;
Mat4x4<float> gx_projection(const Mat4x4<float>& proj, GXProjectionType type) noexcept;
bool main_camera_usable(const Mat4x4<float>& gxProj) noexcept;

namespace wire {
inline constexpr uint32_t SetArrayImmutableV1 = 19;
inline constexpr uint32_t ReleaseArrayImmutableV1 = 14;
inline constexpr uint32_t CameraV1 = 120;
inline constexpr uint32_t ObjectBeginV1 = 26;
inline constexpr uint32_t ObjectBeginV2 = 74;
inline constexpr uint32_t EmptyV1 = 2;
inline constexpr uint32_t ScopePushV1 = 11;
inline constexpr uint32_t StageV1 = 3;
inline constexpr uint32_t CountedHeader = 4;
inline constexpr uint32_t LightV1 = 68;
inline constexpr uint32_t LightV2 = 72;
inline constexpr uint32_t EnvironmentV1 = 58;
inline constexpr uint32_t EnvironmentV2 = 238;
inline constexpr uint32_t MaterialV1 = 6;
inline constexpr uint32_t SkeletonHeader = 12;
inline constexpr uint32_t Joint = 48;
inline constexpr uint32_t SceneEpochV1 = 6;
inline constexpr uint32_t TevPolicyV1 = 34;
inline constexpr uint32_t Influencer = 32;
inline constexpr uint32_t WindSource = 32;
inline constexpr uint32_t GrassPatchesHeader = 15;
inline constexpr uint32_t GrassLightSet = 48;
inline constexpr uint32_t SkyV1 = 166;
inline constexpr uint32_t FogVolume = 64;
inline constexpr uint32_t GpuUploadHeader = 10;
inline constexpr uint32_t GpuReserveV1 = 18;
inline constexpr uint32_t GpuDispatchHeaderV1 = 36;
inline constexpr uint32_t LoadArrayGpuV1 = 15;
inline constexpr uint32_t GpuUpdateHeader = 14;
inline constexpr uint32_t GpuCreateHeader = 28;
inline constexpr uint32_t GpuCreateKernelHeader = 8;
inline constexpr uint32_t GpuDestroyV1 = 10;
inline constexpr uint32_t CreateLibrarySource = 1u << 1;

// Big-endian like fifo::write_*, so payloads decode with ByteReader::read
class Writer {
public:
  explicit Writer(std::span<uint8_t> out) noexcept : m_out(out) {}
  void u8(uint8_t v) noexcept;
  void u16(uint16_t v) noexcept;
  void u32(uint32_t v) noexcept;
  void u64(uint64_t v) noexcept;
  void f32(float v) noexcept;
  void bytes(std::span<const uint8_t> v) noexcept;
  void header(uint16_t id, uint32_t payloadSize) noexcept;
  [[nodiscard]] size_t size() const noexcept { return m_pos; }

private:
  std::span<uint8_t> m_out;
  size_t m_pos = 0;
};

void write(Writer& out, const AuroraLight& light, uint16_t version) noexcept;
void write(Writer& out, const AuroraEnvironment& environment, uint16_t version) noexcept;
void write(Writer& out, const AuroraTevPolicy& policy) noexcept;
void write(Writer& out, const AuroraInfluencer& influencer) noexcept;
void write(Writer& out, const AuroraWindSource& source) noexcept;
void write(Writer& out, const AuroraGrassLightSet& set) noexcept;
void write(Writer& out, const AuroraSky& sky) noexcept;
void write(Writer& out, const AuroraFogVolume& volume) noexcept;

// One reader per payload family, used by the decoder and by the round-trip tests
bool read(ByteReader& in, AuroraLight& out, uint16_t version) noexcept;
bool read(ByteReader& in, AuroraEnvironment& out, uint16_t version) noexcept;
bool read(ByteReader& in, AuroraTevPolicy& out) noexcept;
bool read(ByteReader& in, AuroraInfluencer& out) noexcept;
bool read(ByteReader& in, AuroraWindSource& out) noexcept;
bool read(ByteReader& in, AuroraGrassLightSet& out) noexcept;
bool read(ByteReader& in, AuroraSky& out) noexcept;
bool read(ByteReader& in, AuroraFogVolume& out) noexcept;
} // namespace wire

namespace testing {
void reset_counters() noexcept;
uint32_t warnings() noexcept;
} // namespace testing
} // namespace aurora::gx::hints

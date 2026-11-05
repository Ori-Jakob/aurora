#pragma once

#include "../../gx/gx.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace aurora::render::scene::meshlets {
inline constexpr uint32_t MaxVertices = 64;
inline constexpr uint32_t MaxTriangles = 124;
inline constexpr float ConeWeight = 0.25f;
inline constexpr uint32_t ClusterWords = 16;
inline constexpr uint32_t ClusterBytes = ClusterWords * 4;
inline constexpr uint64_t PoolBytes = 16ull << 20;
inline constexpr uint64_t CopyBytesPerWindow = 1ull << 20;
inline constexpr uint64_t CopyBytesNonStatic = 256ull << 10;
inline constexpr uint32_t MaxJobsInFlight = 64;
inline constexpr uint32_t MaxJobsNonStatic = 16;
inline constexpr uint64_t UploadBytesPerLatch = 2ull << 20;
inline constexpr uint64_t CpuGeometryBytes = 64ull << 20;
inline constexpr uint8_t ChurnLimit = 2;

namespace cluster {
inline constexpr uint32_t RefRel = 0;
inline constexpr uint32_t TriRel = 1;
inline constexpr uint32_t Counts = 2;
inline constexpr uint32_t Cone = 3;
inline constexpr uint32_t Sphere = 4;
inline constexpr uint32_t AabbMin = 8;
inline constexpr uint32_t AabbMax = 12;
inline constexpr uint32_t BoundsSpaceShift = 16;
} // namespace cluster

struct Key {
  uint64_t lo = 0;
  uint64_t hi = 0;

  [[nodiscard]] bool valid() const noexcept { return lo != 0 || hi != 0; }
  bool operator==(const Key&) const = default;
};

enum class State : uint8_t { None, Deferred, Building, Ready, Failed, NoTriangles, Skipped, Unavailable };
enum class Reason : uint8_t { None, PoolFull, Decode, IndexPastArray, DeformedDirect, GpuArray, Churn };

namespace piece_flag {
inline constexpr uint8_t Static = 1u << 0;
inline constexpr uint8_t Posed = 1u << 1;
inline constexpr uint8_t Deformed = 1u << 2;
} // namespace piece_flag

struct Source {
  GXPrimitive prim = GX_TRIANGLES;
  uint32_t vtxCount = 0;
  uint8_t vtxStride = 0;
  std::span<const uint8_t> records;
  gx::AttrConfig pos{};
  gx::AttrConfig pnmtx{};
  const gx::AttrArray* posArray = nullptr;
  std::span<const gx::PnMtx> palette;
  uint32_t currentPnMtx = 0;
  uint8_t flags = 0;
  bool indexed = false;
  std::span<const uint8_t> indices;
};

struct ClusterRange {
  uint32_t first = 0;
  uint32_t count = 0;
  State state = State::None;
  Reason reason = Reason::None;

  bool operator==(const ClusterRange&) const = default;
};

struct CpuGeometry {
  std::vector<float> positions;
  std::vector<uint32_t> triangles;
  std::array<float, 6> aabb{};
};

struct Stats {
  bool active = false;
  uint64_t requests = 0;
  uint64_t submitted = 0;
  uint64_t built = 0;
  uint64_t inlineBuilds = 0;
  uint64_t shared = 0;
  uint64_t deferred = 0;
  uint64_t noTriangles = 0;
  uint64_t unavailable = 0;
  uint64_t skippedDeformedDirect = 0;
  uint64_t skippedGpuArray = 0;
  uint64_t skippedChurn = 0;
  uint64_t failedIndexPastArray = 0;
  uint64_t failedDecode = 0;
  uint64_t failedPoolFull = 0;
  uint64_t poolFullRetries = 0;
  uint64_t degenerate = 0;
  uint64_t stale = 0;
  uint64_t uploads = 0;
  uint64_t uploadBytes = 0;
  uint32_t entries = 0;
  uint32_t ready = 0;
  uint32_t pending = 0;
  uint32_t pendingStatic = 0;
  uint32_t inFlight = 0;
  uint64_t clusters = 0;
  uint64_t poolUsed = 0;
  uint32_t maxVertices = 0;
  uint32_t maxTriangles = 0;
  uint64_t cpuGeometryBytes = 0;
};

[[nodiscard]] uint32_t count_word(const ClusterRange& range) noexcept;

// FIFO thread, while the source spans live
[[nodiscard]] ClusterRange request(const Source& source, Key* inOut, uint8_t* churn) noexcept;
[[nodiscard]] ClusterRange lookup(const Key& key) noexcept;
void acquire(const Key& key) noexcept;
void release(const Key& key) noexcept;
[[nodiscard]] std::shared_ptr<const CpuGeometry> cpu_geometry(const Key& key) noexcept;
[[nodiscard]] uint32_t pending_static() noexcept;

// Game thread while the FIFO is idle
void initialize();
void shutdown();
void set_active(bool active) noexcept;
void latch_resources(bool active) noexcept;
[[nodiscard]] bool latch() noexcept;
void reset() noexcept;
void set_recording_frame(uint64_t frameId) noexcept;
[[nodiscard]] Stats published_stats() noexcept;

// Render worker
void worker_frame_begin(uint64_t frameId);

namespace testing {
using WriteFn = void (*)(uint64_t offset, std::span<const uint32_t> words) noexcept;

struct Upload {
  uint64_t stamp = 0;
  uint64_t offset = 0;
  uint32_t words = 0;
};

void reset() noexcept;
void set_write(WriteFn write) noexcept;
void set_pool_bytes(uint64_t bytes) noexcept;
[[nodiscard]] std::vector<Upload> staged() noexcept;
[[nodiscard]] std::vector<uint32_t> block(const Key& key) noexcept;
[[nodiscard]] uint32_t refs(const Key& key) noexcept;
} // namespace testing
} // namespace aurora::render::scene::meshlets

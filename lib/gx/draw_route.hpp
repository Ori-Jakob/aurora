#pragma once

#include <dolphin/gx/GXEnum.h>

#include <atomic>
#include <cstdint>
#include <span>

namespace aurora::gx {
struct AttrConfig;
struct DrawData;
struct DrawImmediateData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::gx::draw_route {
enum class Route : uint8_t {
  Push,
  Suppress,
  // Recorded into the overlay side list; replayed after the OPAQUE_DONE stage or put back in place
  Defer,
  // Its bytes are pushed for the capture hook and no draw is recorded, so a retained consumer sees a hidden draw
  Capture,
};

struct Query {
  GXPrimitive prim = GX_TRIANGLES;
  GXVtxFmt fmt = GX_VTXFMT0;
  uint32_t vtxCount = 0;
  bool indexed = false;
  std::span<const uint8_t> vertices;
  std::span<const uint8_t> indices;
};

struct Decision {
  Route route = Route::Push;
  bool breakMerge = false;
};

struct MergedDraw {
  uint32_t vertOffset = 0;
  uint32_t vertSize = 0;
  const DrawData* command = nullptr;
  const PipelineConfig* config = nullptr;
};

struct CapturedDraw {
  GXPrimitive prim = GX_TRIANGLES;
  uint32_t vertOffset = 0;
  uint32_t vertSize = 0;
  uint32_t idxOffset = 0;
  uint32_t idxSize = 0;
  uint32_t vtxCount = 0;
  uint32_t indexCount = 0;
  const DrawImmediateData* immediates = nullptr;
  const AttrConfig* attrs = nullptr;
  uint8_t vtxStride = 0;
};

struct Router {
  Decision (*route)(void* userData, const Query& query) noexcept = nullptr;
  // True when the hook changed the pipeline, bind groups, uniform or instance count, so later draws must not merge
  bool (*push)(void* userData, DrawData& draw, const PipelineConfig& config, const ShaderInfo& info) noexcept = nullptr;
  void* userData = nullptr;
  void (*merge)(void* userData, const MergedDraw& draw) noexcept = nullptr;
  // Without it a Capture decision suppresses the draw
  void (*capture)(void* userData, const CapturedDraw& draw) noexcept = nullptr;
};

inline std::atomic<const Router*> g_router{nullptr};

// Swapped only while the FIFO is drained, like guest_memory::set_hooks
inline void set_router(const Router* router) noexcept { g_router.store(router, std::memory_order_release); }
inline const Router* router() noexcept { return g_router.load(std::memory_order_acquire); }
} // namespace aurora::gx::draw_route

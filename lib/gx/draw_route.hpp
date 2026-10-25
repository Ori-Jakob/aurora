#pragma once

#include <dolphin/gx/GXEnum.h>

#include <atomic>
#include <cstdint>

namespace aurora::gx {
struct DrawData;
struct PipelineConfig;
struct ShaderInfo;
} // namespace aurora::gx

namespace aurora::gx::draw_route {
enum class Route : uint8_t {
  Push,
  Suppress,
};

struct Query {
  GXPrimitive prim = GX_TRIANGLES;
  GXVtxFmt fmt = GX_VTXFMT0;
  uint32_t vtxCount = 0;
  bool indexed = false;
};

struct Decision {
  Route route = Route::Push;
  bool breakMerge = false;
};

struct Router {
  Decision (*route)(void* userData, const Query& query) noexcept = nullptr;
  // True when the hook changed the pipeline, bind groups, uniform or instance count, so later draws must not merge
  bool (*push)(void* userData, DrawData& draw, const PipelineConfig& config, const ShaderInfo& info) noexcept = nullptr;
  void* userData = nullptr;
};

inline std::atomic<const Router*> g_router{nullptr};

// Swapped only while the FIFO is drained, like guest_memory::set_hooks
inline void set_router(const Router* router) noexcept { g_router.store(router, std::memory_order_release); }
inline const Router* router() noexcept { return g_router.load(std::memory_order_acquire); }
} // namespace aurora::gx::draw_route

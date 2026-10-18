#include "views.hpp"

#include "debug.hpp"

#include "../scene/classify.hpp"

#include <atomic>

namespace aurora::render::debug {
namespace {
// Written on the game thread, read at the latch
std::atomic<uint16_t> g_override{0};
std::atomic<uint32_t> g_bit{scene::category::Static};

constexpr scene::ViewHooks g_hooks{.beginFrame = &begin_view_frame, .push = &push_view};
} // namespace

void set_view_override(uint16_t id) noexcept { g_override.store(id, std::memory_order_relaxed); }

uint16_t view_override() noexcept { return g_override.load(std::memory_order_relaxed); }

void set_view_bit(uint32_t bit) noexcept { g_bit.store(bit, std::memory_order_relaxed); }

uint32_t view_bit() noexcept { return g_bit.load(std::memory_order_relaxed); }

void initialize_classification() {
  register_tab("Scene", SceneTabOrder, &draw_scene_tab);
  scene::set_view_hooks(&g_hooks);
}

void shutdown_classification() noexcept {
  scene::set_view_hooks(nullptr);
  scene::set_identity_demand(false);
  set_view_override(0);
  set_view_bit(scene::category::Static);
  testing::reset_views();
}
} // namespace aurora::render::debug

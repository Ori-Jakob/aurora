#include "views.hpp"

#include "capture.hpp"
#include "debug.hpp"
#include "dual_source.hpp"
#include "gbuf_view.hpp"

#include "../scene/classify.hpp"
#include "../settings.hpp"

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

uint16_t selected_view_id(const Settings& settings) noexcept {
  const uint16_t id = view_override();
  return id != 0 ? id : settings.debugView;
}

void views_on_latch() {
  gbuf_view::on_latch(selected_view_id(detail::latched_settings()));
  capture::on_latch();
  dual_source::on_latch();
}

void initialize_classification() {
  register_tab("Views", ViewsTabOrder, &draw_views_tab);
  register_tab("Scene", SceneTabOrder, &draw_scene_tab);
  scene::set_view_hooks(&g_hooks);
  capture::initialize();
  dual_source::initialize();
}

void shutdown_classification() noexcept {
  scene::set_view_hooks(nullptr);
  scene::set_identity_demand(false);
  set_view_override(0);
  set_view_bit(scene::category::Static);
  capture::shutdown();
  dual_source::shutdown();
  gbuf_view::shutdown();
  testing::reset_views();
}
} // namespace aurora::render::debug

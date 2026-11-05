#include "graph_api.hpp"
#include "shaders.hpp"
#include "stage.hpp"
#include "scene/capture_list.hpp"

#include "../logging.hpp"

#include <aurora/render_graph.hpp>

#include <atomic>
#include <string>

namespace aurora::render {
namespace {
constexpr Module Log{"aurora::render"};

std::atomic_bool g_warned{false};
} // namespace

namespace detail {
std::string untraceable_external_pass() { return {}; }

bool stage_samples_scene_depth(const StageData&) noexcept { return false; }

DepthSource stage_scene_depth_source(const StageData&) noexcept { return DepthSource::Unavailable; }
} // namespace detail

PassId register_pass(const ExternalPassDesc&) {
  if (!g_warned.exchange(true)) {
    Log.warn("register_pass: aurora was built without AURORA_ENABLE_RENDER");
  }
  return InvalidPass;
}

void unregister_pass(PassId) noexcept {}

bool set_pass_params(PassId, const void*, size_t) noexcept { return false; }

TexRef PassBuilder::create(std::string_view, const TexDesc&, Access) { return {}; }

BufRef PassBuilder::create_buffer(std::string_view, const BufDesc&, Access) { return {}; }

TexRef PassBuilder::history(std::string_view, const TexDesc&, Access) { return {}; }

TexRef PassBuilder::use(std::string_view, Access, Missing) { return {}; }

BufRef PassBuilder::use_buffer(std::string_view, Access, Missing) { return {}; }

void PassBuilder::after(std::string_view) {}

void PassBuilder::before(std::string_view) {}

void PassBuilder::side_effect() {}

void PassBuilder::manual_bindings() {}

Stage PassBuilder::stage() const noexcept { return Stage::BeforeHud; }

Stage PassContext::stage() const noexcept { return Stage::BeforeHud; }

std::string_view PassContext::name() const noexcept { return {}; }

uint32_t PassContext::frame() const noexcept { return 0; }

std::span<const std::byte> PassContext::params() const noexcept { return {}; }

const wgpu::Device& PassContext::device() const noexcept {
  static const wgpu::Device Empty;
  return Empty;
}

const wgpu::CommandEncoder& PassContext::encoder() const noexcept {
  static const wgpu::CommandEncoder Empty;
  return Empty;
}

wgpu::ComputePassEncoder PassContext::begin_compute() { return {}; }

wgpu::RenderPassEncoder PassContext::begin_render() { return {}; }

wgpu::Texture PassContext::texture(TexRef) const { return {}; }

wgpu::TextureView PassContext::view(TexRef, uint32_t) const { return {}; }

wgpu::TextureView PassContext::prev(TexRef) const { return {}; }

bool PassContext::prev_valid(TexRef) const noexcept { return false; }

BufferView PassContext::buffer(BufRef) const { return {}; }

wgpu::Extent3D PassContext::extent(TexRef) const { return {}; }

wgpu::TextureFormat PassContext::format(TexRef) const { return wgpu::TextureFormat::Undefined; }

uint32_t PassContext::sample_count(TexRef) const { return 1; }

wgpu::ColorTargetState PassContext::color_target(TexRef) const { return {}; }

const wgpu::BindGroupLayout& PassContext::frame_layout() const noexcept {
  static const wgpu::BindGroupLayout Empty;
  return Empty;
}

const wgpu::BindGroupLayout& PassContext::pass_layout() const noexcept { return frame_layout(); }

const wgpu::PipelineLayout& PassContext::pipeline_layout() const noexcept {
  static const wgpu::PipelineLayout Empty;
  return Empty;
}
} // namespace aurora::render

namespace aurora::render::shaders {
// Nothing compiles graph pipelines without the render module, so callers waiting on them return at once
bool wait_idle(uint64_t) { return true; }
} // namespace aurora::render::shaders

namespace aurora::render::scene::capture_list {
Stats stats() noexcept { return {}; }

namespace testing {
// Without the render module no draw is routed, so there is nothing to duplicate or replay
ConsumerId install_check(const wgpu::Color&, float) { return InvalidConsumer; }

void remove_check() noexcept {}

void arm_check(uint32_t) noexcept {}

CheckResult check_result() noexcept { return {}; }
} // namespace testing
} // namespace aurora::render::scene::capture_list

#include "fwd_env.hpp"
#include "records.hpp"

#include "../debug/views.hpp"
#include "../frame_state.hpp"
#include "../graph.hpp"
#include "../pass_context.hpp"
#include "../post/composite.hpp"
#include "../scene/classifier.hpp"
#include "../stage_data.hpp"

#include "../../gx/pipeline.hpp"
#include "../../logging.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace aurora::render::gx_ext::fwd_env {
namespace {
constexpr Module Log{"aurora::render::gx_ext"};
constexpr size_t CacheCapacity = 4;
constexpr uint64_t RetainFrames = 3;
constexpr TexRef AoRef{0};

struct Local {
  uint32_t width = 0;
  uint32_t height = 0;
};
static_assert(sizeof(Local) <= detail::PassLocalSize);

struct CacheEntry {
  wgpu::TextureView view;
  wgpu::BindGroup group;
  uint64_t lastUsed = 0;
};

// Game thread, while the FIFO is idle
bool g_screenAo = false;
std::optional<bool> g_identityOverride;
detail::PassId g_pass = detail::InvalidPassId;
detail::GxReadId g_read = detail::InvalidGxReadId;

// Render worker
std::vector<CacheEntry> g_cache;
bool g_everPublished = false;
bool g_mismatchWarned = false;

std::atomic_uint64_t g_published{0};
std::atomic_uint64_t g_dummy{0};
std::atomic_uint64_t g_mismatch{0};
std::atomic_uint64_t g_neutralAfterPublish{0};
std::atomic_uint32_t g_cacheEntries{0};

inline void bump(std::atomic_uint64_t& counter) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void publish_cache_size() noexcept {
  g_cacheEntries.store(static_cast<uint32_t>(g_cache.size()), std::memory_order_relaxed);
}

bool identity() noexcept { return g_identityOverride.value_or(detail::composite::identity_forced()); }

bool publisher_wanted(const detail::PassSetup& setup) noexcept {
  return detail::composite::declare_wanted(setup) && !identity();
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (!publisher_wanted(setup)) {
    builder.skip();
    return;
  }
  builder.after(detail::composite::Pass);
  builder.before(phase::Overlay);
  builder.side_effect();
  builder.manual_bindings();
  builder.use(
      resource::AoFinal, Access::Sampled,
      detail::UseOptions{.missing = Missing::BindDummyWhite, .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  const Local local{setup.env->sceneSize.width, setup.env->sceneSize.height};
  std::memcpy(builder.local().data(), &local, sizeof(local));
}

void declare_read(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const void*) {
  if (publisher_wanted(setup)) {
    builder.use(resource::AoFinal, Access::Sampled,
                detail::UseOptions{.missing = Missing::BindDummyWhite,
                                   .sampleType = wgpu::TextureSampleType::UnfilterableFloat});
  }
}

bool enabled(const detail::StageData& stage, const void*) {
  // Forward AO must match the opaque pixels, so it follows the composite's decision for this frame
  return detail::composite::enabled_this_frame(stage.frameId);
}

const wgpu::BindGroup& cached_group(const detail::PassContext& ctx, const wgpu::TextureView& view) {
  const uint64_t frame = ctx.frame_id();
  for (CacheEntry& entry : g_cache) {
    if (entry.view.Get() == view.Get()) {
      entry.lastUsed = frame;
      return entry.group;
    }
  }
  if (g_cache.size() >= CacheCapacity) {
    g_cache.erase(std::ranges::min_element(g_cache, {}, &CacheEntry::lastUsed));
  }
  const wgpu::BindGroupEntry entry{.binding = 0, .textureView = view};
  const wgpu::BindGroupDescriptor descriptor{
      .label = "GX Forward Environment Bind Group",
      .layout = gx::forward_environment_layout(),
      .entryCount = 1,
      .entries = &entry,
  };
  CacheEntry& added = g_cache.emplace_back(
      CacheEntry{.view = view, .group = ctx.device().CreateBindGroup(&descriptor), .lastUsed = frame});
  publish_cache_size();
  return added.group;
}

void encode(detail::PassContext& ctx, const void*) {
  Local local;
  std::memcpy(&local, ctx.local().data(), sizeof(local));
  const wgpu::Extent3D extent = ctx.extent(AoRef);
  const wgpu::TextureView view = ctx.view(AoRef);
  if (view && extent.width == local.width && extent.height == local.height) {
    gx::set_forward_environment_group(cached_group(ctx, view));
    bump(g_published);
    g_everPublished = true;
    return;
  }
  if (extent.width == 1 && extent.height == 1) {
    bump(g_dummy);
  } else {
    bump(g_mismatch);
    if (!std::exchange(g_mismatchWarned, true)) {
      Log.warn("Forward environment: ao.final is {}x{} but the scene is {}x{}; translucents stay neutral", extent.width,
               extent.height, local.width, local.height);
    }
  }
  if (g_everPublished) {
    bump(g_neutralAfterPublish);
  }
}

void clear_cache() noexcept {
  g_cache.clear();
  publish_cache_size();
}
} // namespace

detail::PassDesc pass_desc() {
  return {
      .name = std::string{Pass},
      .stage = Stage::OpaqueDone,
      .kind = PassKind::Encoder,
      .cls = detail::PassClass::Required,
      .order = 0,
      .runsWithoutMarker = false,
      .declare = declare,
      .enabled = enabled,
      .encode = encode,
  };
}

detail::GxReadDesc read_desc() noexcept { return {.node = ReadNode, .declare = declare_read}; }

bool screen_ao_receiver(const scene::DrawClass& cls, const gx::PipelineConfig& config) noexcept {
  constexpr uint32_t Layers = scene::category::Terrain | scene::category::Decal;
  constexpr uint32_t Excluded = scene::category::Water | scene::category::NoAo | scene::category::Emissive |
                                scene::category::Particle | scene::category::Sky | scene::category::Ui |
                                scene::category::Overlay | scene::category::FakeShadow | scene::category::ScreenRead;
  const uint32_t categories = cls.key.categories;
  return cls.key.window == scene::Window::Translucent && cls.mainClass && !cls.offscreen && config.colorUpdate &&
         gx::tev_colour_clamped(config.shaderConfig) && config.blendMode == GX_BM_BLEND &&
         config.blendFacSrc == GX_BL_SRCALPHA && config.blendFacDst == GX_BL_INVSRCALPHA && config.depthCompare &&
         !config.depthUpdate && (categories & Layers) != 0 && (categories & Excluded) == 0;
}

bool forward_ao_wanted(const Settings& settings, uint16_t debugView) noexcept {
  // Identity runs and class views must leave translucents untouched
  return detail::composite::wanted(settings) && !identity() && debug::find_view(debugView) == nullptr;
}

void apply_frame_state(const detail::FrameState& state) noexcept {
  g_screenAo = records::gbuffer() && forward_ao_wanted(state.settings, state.debugView);
}

uint32_t record_flags(const scene::DrawClass& cls, const gx::PipelineConfig& config) noexcept {
  return g_screenAo && config.shaderConfig.ext.fwdEnv != 0 && screen_ao_receiver(cls, config) ? records::flag::ScreenAo
                                                                                              : 0u;
}

void initialize() {
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pass_desc());
  }
  if (g_read == detail::InvalidGxReadId) {
    g_read = detail::register_gx_read(read_desc());
  }
}

void shutdown() noexcept {
  const testing::Counters c = testing::counters();
  if (c.frames > 0) {
    Log.info(
        "Forward environment: frames {}, published {}, neutral {} (dummy {}, mismatch {}), neutral after first "
        "publish {}",
        c.frames, c.published, c.neutral, c.dummy, c.mismatch, c.neutralAfterFirstPublish);
  }
  gx::set_forward_environment_group({});
  clear_cache();
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  if (g_read != detail::InvalidGxReadId) {
    detail::unregister_gx_read(g_read);
    g_read = detail::InvalidGxReadId;
  }
  g_screenAo = false;
}

void worker_frame_begin(uint64_t frameId) noexcept {
  gx::set_forward_environment_group({});
  const auto stale = [frameId](const CacheEntry& entry) { return frameId - entry.lastUsed >= RetainFrames; };
  if (std::erase_if(g_cache, stale) != 0) {
    publish_cache_size();
  }
}

namespace testing {
Counters counters() noexcept {
  const uint64_t dummy = g_dummy.load(std::memory_order_relaxed);
  const uint64_t mismatch = g_mismatch.load(std::memory_order_relaxed);
  const uint64_t published = g_published.load(std::memory_order_relaxed);
  return {
      .frames = published + dummy + mismatch,
      .published = published,
      .neutral = dummy + mismatch,
      .dummy = dummy,
      .mismatch = mismatch,
      .neutralAfterFirstPublish = g_neutralAfterPublish.load(std::memory_order_relaxed),
      .cacheEntries = g_cacheEntries.load(std::memory_order_relaxed),
  };
}

bool screen_ao_active() noexcept { return g_screenAo; }

void set_identity_override(std::optional<bool> identity) noexcept { g_identityOverride = identity; }

void reset() noexcept {
  g_screenAo = false;
  g_identityOverride.reset();
  gx::set_forward_environment_group({});
  clear_cache();
  g_everPublished = false;
  g_mismatchWarned = false;
  for (std::atomic_uint64_t* counter : {&g_published, &g_dummy, &g_mismatch, &g_neutralAfterPublish}) {
    counter->store(0, std::memory_order_relaxed);
  }
}
} // namespace testing
} // namespace aurora::render::gx_ext::fwd_env

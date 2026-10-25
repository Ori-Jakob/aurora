#include "pick.hpp"

#include "../pass_context.hpp"

#include "../../gfx/frame_packet.hpp"
#include "../../gfx/pipeline_cache.hpp"
#include "../../internal.hpp"
#include "../../webgpu/gpu.hpp"

#include <aurora/render.hpp>
#include <aurora/render_caps.hpp>

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::render::debug::pick {
namespace {
constexpr TexRef Source{0};
constexpr size_t SlotCount = 3;
constexpr uint64_t ReadbackBytes = sizeof(uint32_t);

enum class SlotState : uint8_t { Available, CopySubmitted, MapPending };

struct Slot {
  wgpu::Buffer buffer;
  SlotState state = SlotState::Available;
  uint32_t sequence = 0;
};

std::mutex g_mutex;
std::array<Slot, SlotCount> g_slots;
size_t g_nextSlot = 0;
std::optional<Readback> g_latest;
std::atomic_uint64_t g_generation{1};

// Game thread
detail::PassId g_pass = detail::InvalidPassId;

void declare_pick(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void*) {
  builder.use(detail::gx_resource_name(detail::GxResource::DebugId), Access::CopySrc, Missing::DisablePass);
}

void complete(size_t index, uint64_t generation, const wgpu::Buffer& buffer, wgpu::MapAsyncStatus status) {
  std::optional<uint32_t> value;
  if (status == wgpu::MapAsyncStatus::Success) {
    if (const void* mapped = buffer.GetConstMappedRange(0, ReadbackBytes)) {
      uint32_t word = 0;
      std::memcpy(&word, mapped, sizeof(word));
      value = word;
    }
    buffer.Unmap();
  }
  std::lock_guard lock{g_mutex};
  if (generation != g_generation.load(std::memory_order_relaxed)) {
    return;
  }
  if (value) {
    g_latest = Readback{*value, g_slots[index].sequence};
  }
  g_slots[index].state = SlotState::Available;
}

void encode_pick(detail::PassContext& ctx, const void*) {
  const std::span<const std::byte> bytes = ctx.params();
  Params params;
  if (bytes.size() < sizeof(params)) {
    return;
  }
  std::memcpy(&params, bytes.data(), sizeof(params));
  const wgpu::Texture texture = ctx.texture(Source);
  if (params.valid == 0 || !texture || params.x >= texture.GetWidth() || params.y >= texture.GetHeight()) {
    return;
  }
  size_t index = SlotCount;
  wgpu::Buffer buffer;
  const uint64_t generation = g_generation.load(std::memory_order_relaxed);
  {
    std::lock_guard lock{g_mutex};
    for (size_t i = 0; i < SlotCount; ++i) {
      const size_t candidate = (g_nextSlot + i) % SlotCount;
      if (g_slots[candidate].state == SlotState::Available) {
        index = candidate;
        break;
      }
    }
    if (index == SlotCount) {
      return;
    }
    Slot& slot = g_slots[index];
    if (!slot.buffer) {
      const wgpu::BufferDescriptor descriptor{
          .label = "Texture pick readback",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = ReadbackBytes,
      };
      slot.buffer = ctx.device().CreateBuffer(&descriptor);
    }
    slot.state = SlotState::CopySubmitted;
    slot.sequence = params.sequence;
    buffer = slot.buffer;
    g_nextSlot = (index + 1) % SlotCount;
  }
  const wgpu::TexelCopyTextureInfo source{
      .texture = texture,
      .origin = {params.x, params.y, 0},
  };
  const wgpu::TexelCopyBufferInfo destination{.buffer = buffer};
  const wgpu::Extent3D extent{1, 1, 1};
  ctx.encoder().CopyTextureToBuffer(&source, &destination, &extent);
  detail::queue_after_frame([index, generation, buffer] {
    {
      std::lock_guard lock{g_mutex};
      if (generation != g_generation.load(std::memory_order_relaxed)) {
        return;
      }
      g_slots[index].state = SlotState::MapPending;
    }
    buffer.MapAsync(wgpu::MapMode::Read, 0, ReadbackBytes, wgpu::CallbackMode::AllowSpontaneous,
                    [index, generation, buffer](wgpu::MapAsyncStatus status, wgpu::StringView) {
                      complete(index, generation, buffer, status);
                    });
  });
}

detail::PassDesc pick_pass_desc() {
  return {
      .name = std::string{PassName},
      .stage = Stage::PrePost,
      .kind = PassKind::Copy,
      .cls = detail::PassClass::Diagnostic,
      .order = 0,
      .runsWithoutMarker = false,
      .paramsSize = sizeof(Params),
      .declare = declare_pick,
      .encode = encode_pick,
  };
}
} // namespace

Capability pick_capability(const CapabilityInputs& inputs) noexcept {
  if (!inputs.device) {
    return {false, "no device"};
  }
  if (!inputs.core) {
    return {false, "needs core features"};
  }
  if (inputs.msaaSamples != 1) {
    return {false, "needs MSAA off"};
  }
  if (!inputs.hints) {
    return {false, "render hints are off"};
  }
  if (inputs.colorBytes > inputs.colorBudget) {
    return {false, "over the colour attachment budget"};
  }
  return {true, {}};
}

CapabilityInputs live_capability_inputs() noexcept {
  return {
      .device = static_cast<bool>(webgpu::g_device),
      .core = webgpu::g_hasCoreFeatures,
      .msaaSamples = webgpu::g_graphicsConfig.msaaSamples,
      .hints = hints_enabled(),
      .colorBytes = layout_color_bytes(armed_layout(gfx::scene_render_target_layout())),
      .colorBudget = caps().limits.maxColorAttachmentBytesPerSample,
  };
}

std::optional<PixelPos> efb_pixel_at(Vec2<float> mouse, Vec2<float> fbScale, Vec2<uint32_t> surfaceSize,
                                     Vec2<uint32_t> efbSize) noexcept {
  const webgpu::Viewport viewport =
      webgpu::calculate_present_viewport(surfaceSize.x, surfaceSize.y, efbSize.x, efbSize.y);
  if (viewport.width <= 0.f || viewport.height <= 0.f) {
    return std::nullopt;
  }
  const float u = (mouse.x * fbScale.x - viewport.left) / viewport.width;
  const float v = (mouse.y * fbScale.y - viewport.top) / viewport.height;
  if (!(u >= 0.f && u < 1.f && v >= 0.f && v < 1.f)) {
    return std::nullopt;
  }
  return PixelPos{
      std::min(static_cast<uint32_t>(u * static_cast<float>(efbSize.x)), efbSize.x - 1),
      std::min(static_cast<uint32_t>(v * static_cast<float>(efbSize.y)), efbSize.y - 1),
  };
}

gfx::RenderTargetLayout armed_layout(const gfx::RenderTargetLayout& live) noexcept {
  gfx::RenderTargetLayout layout = gfx::pipeline_base_layout(live);
  if (layout.colorAttachmentCount >= gfx::MaxColorAttachments) {
    return layout;
  }
  const gfx::ColorAttachmentLayout& scene = layout.colorAttachments[gfx::SceneColorAttachmentIndex];
  layout.colorAttachments[layout.colorAttachmentCount++] = {
      .semantic = gfx::ColorAttachmentSemantic::DebugId,
      .format = webgpu::DebugIdBufferFormat,
      .width = scene.width,
      .height = scene.height,
  };
  gfx::detail::finalize_render_target_layout(layout);
  return layout;
}

uint32_t layout_color_bytes(const gfx::RenderTargetLayout& layout) noexcept {
  std::array<wgpu::TextureFormat, gfx::MaxColorAttachments> formats{};
  const uint32_t count = std::min<uint32_t>(layout.colorAttachmentCount, gfx::MaxColorAttachments);
  for (uint32_t i = 0; i < count; ++i) {
    formats[i] = layout.colorAttachments[i].format;
  }
  return detail::color_bytes_per_sample(std::span{formats}.first(count));
}

bool warm_complete(const WarmInputs& inputs) noexcept {
  if (inputs.observedFrames == 0) {
    return false;
  }
  if (inputs.seconds >= WarmTimeoutSeconds) {
    return true;
  }
  return inputs.total != 0 && uint64_t{inputs.ready} * 100 >= uint64_t{inputs.total} * WarmReadyPercent;
}

ArmState next_arm_state(ArmState current, const ArmInputs& inputs) noexcept {
  if (!inputs.wanted) {
    return ArmState::Off;
  }
  switch (current) {
  case ArmState::Off:
    return ArmState::Warming;
  case ArmState::Warming:
    return inputs.warmComplete ? ArmState::Armed : ArmState::Warming;
  case ArmState::Armed:
    return inputs.baseLayoutChanged ? ArmState::Warming : ArmState::Armed;
  }
  return ArmState::Off;
}

bool resolve_gx_debug_id(const detail::StageData&, Access access, detail::GxBinding& out) {
  const webgpu::TextureWithSampler& ids = webgpu::g_debugIdBuffer;
  if (!webgpu::g_graphicsConfig.debugIdBuffer || !ids.texture) {
    return false;
  }
  if (access == Access::Sampled && !(ids.texture.GetUsage() & wgpu::TextureUsage::TextureBinding)) {
    return false;
  }
  out = {
      .texture = ids.texture,
      .sampleView = ids.view,
      .attachmentView = ids.view,
      .copyTexture = ids.texture,
      .size = ids.size,
      .format = ids.texture.GetFormat(),
      .samples = ids.texture.GetSampleCount(),
  };
  return true;
}

void initialize() { detail::register_gx_resolver(detail::GxResource::DebugId, &resolve_gx_debug_id); }

void shutdown() noexcept {
  if (g_pass != detail::InvalidPassId) {
    detail::unregister_pass(g_pass);
    g_pass = detail::InvalidPassId;
  }
  detail::register_gx_resolver(detail::GxResource::DebugId, nullptr);
  std::lock_guard lock{g_mutex};
  g_generation.fetch_add(1, std::memory_order_relaxed);
  g_slots = {};
  g_nextSlot = 0;
  g_latest.reset();
}

void ensure_pass() {
  if (g_pass == detail::InvalidPassId) {
    g_pass = detail::register_pass(pick_pass_desc());
  }
}

bool pass_registered() noexcept { return g_pass != detail::InvalidPassId; }

void set_params(const Params& params) {
  if (g_pass != detail::InvalidPassId) {
    detail::set_pass_params(g_pass, std::as_bytes(std::span{&params, 1}));
  }
}

std::optional<Readback> take_readback() noexcept {
  std::lock_guard lock{g_mutex};
  return std::exchange(g_latest, std::nullopt);
}

namespace testing {
void set_readback(const Readback& readback) noexcept {
  std::lock_guard lock{g_mutex};
  g_latest = readback;
}
} // namespace testing
} // namespace aurora::render::debug::pick

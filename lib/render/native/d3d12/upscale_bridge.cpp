#include "upscale_bridge.hpp"

#include "device.hpp"
#include "native_step.hpp"
#include "probe.hpp"
#include "sentinel.hpp"
#include "teardown.hpp"

#include "../../post/upscale/compose.hpp"
#include "../../post/upscale/upscale.hpp"
#include "../../../logging.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace aurora::render::detail::native::d3d12::upscale_bridge {
namespace {
constexpr Module Log{"aurora::render::native"};

using upscale::native_upscale::Adapter;
using upscale::native_upscale::Frame;
using upscale::native_upscale::Image;
using upscale::native_upscale::Inputs;
using upscale::native_upscale::Plan;
using upscale::native_upscale::ProbeResult;
using upscale::native_upscale::Targets;

struct Exchange {
  std::optional<Plan> wanted;
  std::optional<Plan> prepared;
  std::optional<Plan> failed;
  GraphExtent render;
  bool release = false;
};

struct Live {
  NativeUpscaler* upscaler = nullptr;
  Plan plan;
  GraphExtent render;
  uint64_t vendorBytes = 0;
  bool armed = false;
  bool failed = false;
  bool warming = false;
  std::unique_ptr<StepRunner> runner;
  std::unique_ptr<SharedTexture> color;
  std::unique_ptr<SharedTexture> depth;
  std::unique_ptr<SharedTexture> motion;
  std::unique_ptr<SharedTexture> reactive;
  std::unique_ptr<SharedTexture> out;
  Frame frame;
};

std::array<NativeUpscaler*, 2> g_upscalers{};
std::mutex g_mutex;
Exchange g_exchange;

// Render worker only
Live g_live;

constexpr size_t index_of(Upscaler option) noexcept { return option == Upscaler::Dlss ? 0 : 1; }

Image image(const SharedTexture* texture) noexcept {
  if (texture == nullptr) {
    return {};
  }
  const wgpu::Texture& handle = texture->texture();
  return {
      .resource = texture->resource(), .format = texture->format(), .size = {handle.GetWidth(), handle.GetHeight()}};
}

bool record(void* user, ID3D12GraphicsCommandList* list) noexcept {
  const auto& live = *static_cast<const Live*>(user);
  if (live.upscaler == nullptr) {
    return false;
  }
  const Frame& frame = live.frame;
  Inputs inputs{
      .commandList = list,
      .color = image(live.color.get()),
      .depth = image(live.depth.get()),
      .motion = image(live.motion.get()),
      .reactive = image(live.reactive.get()),
      .output = image(live.out.get()),
      .camera = frame.camera,
      .frameTimeMs = frame.frameTimeMs,
      .sharpness = frame.sharpness,
      .reset = frame.reset,
  };
  std::copy_n(frame.jitter, 2, inputs.jitter);
  std::copy_n(frame.motionScale, 2, inputs.motionScale);
  return live.upscaler->evaluate(inputs);
}

void step_failed(void* user) noexcept {
  auto& live = *static_cast<Live*>(user);
  live.failed = true;
  if (live.upscaler != nullptr) {
    probe::fail(live.upscaler->option());
  }
}

void release_live() noexcept {
  if (g_live.runner) {
    g_live.runner->wait_idle();
  }
  if (g_live.upscaler != nullptr) {
    g_live.upscaler->destroy();
  }
  const bool armed = g_live.armed;
  const bool vendorBytes = g_live.vendorBytes != 0;
  g_live = {};
  if (vendorBytes) {
    set_vendor_bytes(0);
  }
  if (armed) {
    sentinel::disarm(sentinel::Session);
  }
}

void finish_prepare(const Plan& plan, bool ok) noexcept {
  NativeUpscaler& upscaler = *plan.upscaler;
  g_live.warming = false;
  {
    std::lock_guard lock{g_mutex};
    if (ok) {
      g_exchange.prepared = plan;
      g_exchange.render = g_live.render;
    } else {
      g_exchange.failed = plan;
    }
  }
  if (ok) {
    g_live.vendorBytes = upscaler.vram_bytes();
    set_vendor_bytes(g_live.vendorBytes);
    return;
  }
  Log.warn("Native upscaler {} could not be prepared for {}x{}", upscaler.name(), plan.output.width,
           plan.output.height);
  release_live();
  probe::fail(upscaler.option());
}

void teardown() noexcept {
  release_live();
  std::lock_guard lock{g_mutex};
  g_exchange = {};
}

bool ensure(std::unique_ptr<SharedTexture>& texture, const SharedTexture::Desc& desc) noexcept {
  if (texture && texture->format() == desc.format && texture->texture().GetWidth() == desc.width &&
      texture->texture().GetHeight() == desc.height) {
    return true;
  }
  if (g_live.runner) {
    g_live.runner->wait_idle();
  }
  texture.reset();
  texture = SharedTexture::create(desc);
  return texture != nullptr;
}
} // namespace

ProbeResult PassthroughUpscaler::probe(const Adapter&) noexcept {
  return {.support = {true, Reason::None}, .path = "copy", .autoPick = false};
}

std::string_view PassthroughUpscaler::display_name(UpscalerQuality) const noexcept { return "Copy"; }

std::optional<GraphExtent> PassthroughUpscaler::prepare(const Adapter&, GraphExtent output, UpscalerQuality) noexcept {
  ++m_prepares;
  return upscale::split_size(output, m_ratio);
}

bool PassthroughUpscaler::evaluate(const Inputs& inputs) noexcept {
  ++m_evaluations;
  auto* list = static_cast<ID3D12GraphicsCommandList*>(inputs.commandList);
  if (m_failing || list == nullptr || inputs.color.resource == nullptr || inputs.output.resource == nullptr) {
    return false;
  }
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = static_cast<ID3D12Resource*>(inputs.output.resource);
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = static_cast<ID3D12Resource*>(inputs.color.resource);
  source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  return true;
}

uint32_t PassthroughUpscaler::jitter_phases(float) const noexcept { return 8; }

float PassthroughUpscaler::mip_bias(float ratio) const noexcept { return std::log2(1.f / ratio); }

void register_upscaler(NativeUpscaler& upscaler) noexcept {
  const Upscaler option = upscaler.option();
  if (option == Upscaler::Dlss || option == Upscaler::Xess) {
    g_upscalers[index_of(option)] = &upscaler;
  }
}

NativeUpscaler* upscaler(Upscaler option) noexcept {
  return option == Upscaler::Dlss || option == Upscaler::Xess ? g_upscalers[index_of(option)] : nullptr;
}

Adapter adapter() noexcept {
  Adapter out{.backend = upscale::native_upscale::Backend::D3D12, .device = device(), .queue = queue()};
  if (available()) {
    const DXGI_ADAPTER_DESC1& desc = adapter_desc();
    out.vendorId = desc.VendorId;
    out.deviceId = desc.DeviceId;
    out.subSysId = desc.SubSysId;
    out.revision = desc.Revision;
    out.driverVersion = driver_version();
  }
  return out;
}

bool prepare(const Plan& plan, GraphExtent& render) noexcept {
  std::lock_guard lock{g_mutex};
  if (g_exchange.prepared == plan) {
    render = g_exchange.render;
    return true;
  }
  if (g_exchange.failed != plan) {
    g_exchange.wanted = plan;
  }
  return false;
}

void release() noexcept {
  std::lock_guard lock{g_mutex};
  g_exchange.wanted.reset();
  g_exchange.prepared.reset();
  g_exchange.failed.reset();
  g_exchange.release = true;
}

void worker_frame_begin() noexcept {
  std::optional<Plan> wanted;
  bool releasing = false;
  {
    std::lock_guard lock{g_mutex};
    releasing = std::exchange(g_exchange.release, false);
    if (g_exchange.wanted && g_exchange.wanted != g_exchange.prepared && g_exchange.wanted != g_exchange.failed) {
      wanted = g_exchange.wanted;
    }
  }
  if (releasing) {
    release_live();
  }
  if (!wanted || wanted->upscaler == nullptr) {
    return;
  }
  NativeUpscaler& upscaler = *wanted->upscaler;
  if (!g_live.warming || g_live.upscaler != &upscaler || g_live.plan != *wanted) {
    if (g_live.upscaler != nullptr && g_live.upscaler != &upscaler) {
      release_live();
    }
    if (g_live.runner) {
      g_live.runner->wait_idle();
    }
    if (!std::exchange(g_live.armed, true)) {
      sentinel::arm(sentinel::Session);
    }
    register_teardown(&teardown);
    probe::count_sdk_call();
    const std::optional<GraphExtent> render = upscaler.prepare(adapter(), wanted->output, wanted->quality);
    if (!render || render->width == 0 || render->height == 0) {
      finish_prepare(*wanted, false);
      return;
    }
    g_live.upscaler = &upscaler;
    g_live.plan = *wanted;
    g_live.render = {std::min(render->width, wanted->output.width), std::min(render->height, wanted->output.height)};
    g_live.failed = false;
  }
  const upscale::native_upscale::Warmup state = upscaler.warmup();
  g_live.warming = state == upscale::native_upscale::Warmup::Pending;
  if (!g_live.warming) {
    finish_prepare(*wanted, state == upscale::native_upscale::Warmup::Ready);
  }
}

void wait_idle() noexcept {
  if (g_live.runner) {
    g_live.runner->wait_idle();
  }
}

bool begin(const Frame& frame, Targets& targets) noexcept {
  Live& live = g_live;
  const Caps& deviceCaps = caps();
  if (live.upscaler == nullptr || live.failed || live.render != frame.render || live.plan.output != frame.output ||
      !available() || !deviceCaps.d3d12SharedTexture || !deviceCaps.d3d12SharedFence) {
    return false;
  }
  NativeUpscaler& upscaler = *live.upscaler;
  const bool passthrough = upscaler.passthrough();
  const bool reactive = upscaler.wants_reactive();
  const ColorEncoding encoding = upscaler.color_encoding();
  const bool linear = encoding == ColorEncoding::Linear;
  const wgpu::TextureFormat colorFormat = linear ? wgpu::TextureFormat::RGBA16Float : frame.colorFormat;
  const GraphExtent outSize = passthrough ? frame.render : frame.output;
  const wgpu::TextureFormat outFormat =
      passthrough ? colorFormat : (linear ? wgpu::TextureFormat::RGBA16Float : wgpu::TextureFormat::RGBA8Unorm);
  const uint32_t width = frame.render.width;
  const uint32_t height = frame.render.height;
  bool ok =
      ensure(live.color, {.format = colorFormat,
                          .width = width,
                          .height = height,
                          .renderTarget = linear,
                          .label = "native.in_color"}) &&
      ensure(live.depth, {.format = wgpu::TextureFormat::R32Float,
                          .width = width,
                          .height = height,
                          .renderTarget = true,
                          .label = "native.in_depth"}) &&
      ensure(
          live.motion,
          {.format = wgpu::TextureFormat::RG16Float, .width = width, .height = height, .label = "native.in_motion"}) &&
      (!reactive || ensure(live.reactive, {.format = wgpu::TextureFormat::R8Unorm,
                                           .width = width,
                                           .height = height,
                                           .label = "native.in_reactive"})) &&
      ensure(live.out, {.format = outFormat,
                        .width = outSize.width,
                        .height = outSize.height,
                        .unorderedAccess = !passthrough,
                        .label = "native.out"});
  if (ok && !live.runner) {
    live.runner = StepRunner::create({
        .name = upscaler.name(),
        .inputState = passthrough ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        .outputState = passthrough ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        .record = &record,
        .failed = &step_failed,
        .user = &live,
    });
    ok = live.runner != nullptr;
  }
  if (!ok) {
    Log.warn("Native upscaler {}: its shared textures or native step could not be created", upscaler.name());
    live.failed = true;
    probe::fail(upscaler.option());
    return false;
  }
  live.frame = frame;
  const std::array<SharedTexture*, MaxStepInputs> inputs{live.color.get(), live.depth.get(), live.motion.get(),
                                                         live.reactive.get()};
  const size_t count = reactive ? 4 : 3;
  upscale::compose::request_native_step(live.runner->step(std::span{inputs.data(), count}, live.out.get()),
                                        {.view = live.out->view(), .format = live.out->format(), .encoding = encoding});
  targets = {
      .color = live.color->texture(),
      .colorView = live.color->view(),
      .depthView = live.depth->view(),
      .motion = live.motion->texture(),
      .reactive = reactive ? live.reactive->texture() : wgpu::Texture{},
      .encoding = encoding,
  };
  return true;
}

namespace testing {
Shared shared() noexcept {
  return {
      .color = g_live.color.get(),
      .depth = g_live.depth.get(),
      .motion = g_live.motion.get(),
      .reactive = g_live.reactive.get(),
      .out = g_live.out.get(),
  };
}

void set_upscaler(Upscaler option, NativeUpscaler* upscaler) noexcept {
  if (option == Upscaler::Dlss || option == Upscaler::Xess) {
    g_upscalers[index_of(option)] = upscaler;
  }
}

void reset() noexcept {
  release_live();
  g_upscalers = {};
  std::lock_guard lock{g_mutex};
  g_exchange = {};
}
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::upscale_bridge

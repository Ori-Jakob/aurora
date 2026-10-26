#include "xess.hpp"

#include "upscale_bridge.hpp"

#include "../native.hpp"
#include "../../post/upscale/upscale.hpp"
#include "../../../gfx/pipeline_cache.hpp"
#include "../../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <string_view>
#include <utility>

namespace aurora::render::detail::native::d3d12::xess {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr uint32_t NodeMask = 1;
constexpr double MiB = 1024.0 * 1024.0;

using upscale::native_upscale::Adapter;
using upscale::native_upscale::Backend;
using upscale::native_upscale::Inputs;
using upscale::native_upscale::Warmup;

void wait_bridge() noexcept { upscale_bridge::wait_idle(); }

FenceWait g_fenceWait = &wait_bridge;

bool failed(xess_result_t result) noexcept { return result < XESS_RESULT_SUCCESS; }

bool responsive_env() noexcept {
  static const bool value = [] {
    const char* env = SDL_getenv("AURORA_RENDER_XESS_RESPONSIVE");
    const bool on = env != nullptr && std::string_view{env} == "1";
    if (on) {
      Log.warn("AURORA_RENDER_XESS_RESPONSIVE=1: XeSS-SR reads the reactive mask as its responsive mask");
    }
    return on;
  }();
  return value;
}
} // namespace

xess_quality_settings_t quality_setting(UpscalerQuality quality) noexcept {
  switch (quality) {
  case UpscalerQuality::NativeAa:
    return XESS_QUALITY_SETTING_AA;
  case UpscalerQuality::Quality:
    return XESS_QUALITY_SETTING_QUALITY;
  case UpscalerQuality::Balanced:
    return XESS_QUALITY_SETTING_BALANCED;
  case UpscalerQuality::Performance:
    return XESS_QUALITY_SETTING_PERFORMANCE;
  case UpscalerQuality::UltraPerformance:
    return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
  }
  return XESS_QUALITY_SETTING_QUALITY;
}

uint32_t init_flags(bool responsive) noexcept {
  uint32_t flags = XESS_INIT_FLAG_INVERTED_DEPTH | XESS_INIT_FLAG_LDR_INPUT_COLOR;
  if (responsive) {
    // Intel wants an unjittered mask used only where it clearly helps; M7's reactive mask is jittered
    flags |= XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
  }
  return flags;
}

std::array<float, 2> velocity_scale(GraphExtent render) noexcept {
  return {-0.5f * static_cast<float>(render.width), 0.5f * static_cast<float>(render.height)};
}

void wait_last_use() noexcept {
  if (g_fenceWait != nullptr) {
    g_fenceWait();
  }
}

ColorEncoding XessUpscaler::color_encoding() const noexcept {
  return upscale::color_env().value_or(ColorEncoding::Linear);
}

bool XessUpscaler::wants_reactive() const noexcept { return responsive_env(); }

std::optional<GraphExtent> XessUpscaler::prepare(const Adapter& adapter, GraphExtent output,
                                                 UpscalerQuality quality) noexcept {
  native::assert_worker();
  if (output.width == 0 || output.height == 0) {
    return std::nullopt;
  }
  if (m_context == nullptr) {
    const Api* api = load();
    auto* device = adapter.backend == Backend::D3D12 ? static_cast<ID3D12Device*>(adapter.device) : nullptr;
    if (api == nullptr || device == nullptr) {
      return std::nullopt;
    }
    xess_context_handle_t context = nullptr;
    const xess_result_t created = create_context(*api, device, context);
    if (failed(created) || context == nullptr) {
      Log.warn("{}: no context (result {})", DisplayName, static_cast<int>(created));
      if (context != nullptr) {
        api->destroyContext(context);
      }
      return std::nullopt;
    }
    m_api = api;
    m_context = context;
    m_flags = init_flags(wants_reactive());
    const bool blocking = gfx::blocking_pipelines();
    const xess_result_t built = m_api->d3d12BuildPipelines(m_context, nullptr, blocking, m_flags);
    if (failed(built)) {
      Log.warn("{}: pipeline build failed to start (result {})", DisplayName, static_cast<int>(built));
      destroy();
      return std::nullopt;
    }
    m_building = !blocking;
  }
  const xess_2d_t target{output.width, output.height};
  xess_2d_t optimal{};
  xess_2d_t minimum{};
  xess_2d_t maximum{};
  const xess_result_t sized =
      m_api->getOptimalInputResolution(m_context, &target, quality_setting(quality), &optimal, &minimum, &maximum);
  if (failed(sized) || optimal.x == 0 || optimal.y == 0) {
    Log.warn("{}: no input size for {}x{} (result {})", DisplayName, output.width, output.height,
             static_cast<int>(sized));
    destroy();
    return std::nullopt;
  }
  xess_properties_t properties{};
  const xess_result_t queried = m_api->getProperties(m_context, &target, &properties);
  m_vramBytes = failed(queried) ? 0 : properties.tempBufferHeapSize + properties.tempTextureHeapSize;
  m_output = output;
  m_render = {optimal.x, optimal.y};
  m_quality = quality;
  m_initialized = false;
  return m_render;
}

Warmup XessUpscaler::warmup() noexcept {
  native::assert_worker();
  if (m_context == nullptr) {
    return Warmup::Failed;
  }
  if (m_initialized) {
    return Warmup::Ready;
  }
  if (m_building) {
    if (m_api->getPipelineBuildStatus(m_context) == XESS_RESULT_ERROR_OPERATION_IN_PROGRESS) {
      return Warmup::Pending;
    }
    m_building = false;
  }
  xess_d3d12_init_params_t params{};
  params.outputResolution = {m_output.width, m_output.height};
  params.qualitySetting = quality_setting(m_quality);
  params.initFlags = m_flags;
  params.creationNodeMask = NodeMask;
  params.visibleNodeMask = NodeMask;
  wait_last_use();
  const xess_result_t initialized = m_api->d3d12Init(m_context, &params);
  if (failed(initialized)) {
    Log.warn("{}: init failed for {}x{} (result {})", DisplayName, m_output.width, m_output.height,
             static_cast<int>(initialized));
    return Warmup::Failed;
  }
  m_velocityScale = velocity_scale(m_render);
  m_api->setVelocityScale(m_context, m_velocityScale[0], m_velocityScale[1]);
  m_initialized = true;
  m_executeFailed = false;
  Log.info("{}: {}x{} to {}x{}, temp heaps {:.1f} MiB", DisplayName, m_render.width, m_render.height, m_output.width,
           m_output.height, static_cast<double>(m_vramBytes) / MiB);
  return Warmup::Ready;
}

bool XessUpscaler::evaluate(const Inputs& inputs) noexcept {
  native::assert_worker();
  auto* list = static_cast<ID3D12GraphicsCommandList*>(inputs.commandList);
  const bool responsive = (m_flags & XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK) != 0;
  if (!m_initialized || list == nullptr || inputs.color.resource == nullptr || inputs.depth.resource == nullptr ||
      inputs.motion.resource == nullptr || inputs.output.resource == nullptr ||
      (responsive && inputs.reactive.resource == nullptr)) {
    return false;
  }
  if (inputs.motionScale[0] != m_velocityScale[0] || inputs.motionScale[1] != m_velocityScale[1]) {
    m_velocityScale = {inputs.motionScale[0], inputs.motionScale[1]};
    m_api->setVelocityScale(m_context, m_velocityScale[0], m_velocityScale[1]);
  }
  xess_d3d12_execute_params_t params{};
  params.pColorTexture = static_cast<ID3D12Resource*>(inputs.color.resource);
  params.pVelocityTexture = static_cast<ID3D12Resource*>(inputs.motion.resource);
  params.pDepthTexture = static_cast<ID3D12Resource*>(inputs.depth.resource);
  params.pResponsivePixelMaskTexture = responsive ? static_cast<ID3D12Resource*>(inputs.reactive.resource) : nullptr;
  params.pOutputTexture = static_cast<ID3D12Resource*>(inputs.output.resource);
  params.jitterOffsetX = inputs.jitter[0];
  params.jitterOffsetY = inputs.jitter[1];
  params.exposureScale = 1.f;
  params.resetHistory = inputs.reset ? 1u : 0u;
  params.inputWidth = inputs.color.size.width;
  params.inputHeight = inputs.color.size.height;
  const xess_result_t result = m_api->d3d12Execute(m_context, list, &params);
  if (failed(result)) {
    if (!std::exchange(m_executeFailed, true)) {
      Log.warn("{}: execute failed (result {})", DisplayName, static_cast<int>(result));
    }
    return false;
  }
  return true;
}

void XessUpscaler::destroy() noexcept {
  if (m_context == nullptr) {
    return;
  }
  native::assert_worker();
  wait_last_use();
  m_api->destroyContext(m_context);
  m_context = nullptr;
  m_api = nullptr;
  m_vramBytes = 0;
  m_building = false;
  m_initialized = false;
}

namespace testing {
void set_fence_wait(FenceWait wait) noexcept { g_fenceWait = wait != nullptr ? wait : &wait_bridge; }
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::xess

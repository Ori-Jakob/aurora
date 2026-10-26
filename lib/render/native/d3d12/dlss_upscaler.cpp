#include "dlss.hpp"

#include "../native.hpp"
#include "../../../logging.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace aurora::render::detail::native::d3d12::dlss {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr uint32_t MinJitterPhases = 16;
constexpr double MiB = 1024.0 * 1024.0;

using upscale::native_upscale::Inputs;

Evaluation evaluation(const Inputs& inputs) noexcept {
  return {
      .color = static_cast<ID3D12Resource*>(inputs.color.resource),
      .depth = static_cast<ID3D12Resource*>(inputs.depth.resource),
      .motion = static_cast<ID3D12Resource*>(inputs.motion.resource),
      .output = static_cast<ID3D12Resource*>(inputs.output.resource),
      .jitter = {inputs.jitter[0], inputs.jitter[1]},
      .motionScale = {inputs.motionScale[0], inputs.motionScale[1]},
      .render = inputs.color.size,
      .frameTimeMs = inputs.frameTimeMs,
      .reset = inputs.reset,
  };
}
} // namespace

FeatureFlags feature_flags() noexcept { return {.mvLowRes = true, .depthInverted = true}; }

std::string_view DlssUpscaler::display_name(UpscalerQuality quality) const noexcept {
  return quality == UpscalerQuality::NativeAa ? NativeAaName : DisplayName;
}

bool DlssUpscaler::evaluate(const Inputs& inputs) noexcept {
  native::assert_worker();
  auto* list = static_cast<ID3D12GraphicsCommandList*>(inputs.commandList);
  if (!m_session || list == nullptr || inputs.color.resource == nullptr || inputs.depth.resource == nullptr ||
      inputs.motion.resource == nullptr || inputs.output.resource == nullptr || inputs.color.size != m_wanted.render ||
      inputs.output.size != m_wanted.output) {
    return false;
  }
  if (m_feature == nullptr) {
    m_feature = m_api->create(list, m_wanted);
    if (m_feature == nullptr) {
      Log.warn("{}: NGX could not create its {} feature for {}x{} to {}x{}", DisplayName, mode_name(m_wanted.quality),
               m_wanted.render.width, m_wanted.render.height, m_wanted.output.width, m_wanted.output.height);
      return false;
    }
    m_featureDesc = m_wanted;
    m_vramBytes = m_api->vram().value_or(0);
    Log.info("{}: {}x{} to {}x{} at {}, {:.1f} MiB", DisplayName, m_wanted.render.width, m_wanted.render.height,
             m_wanted.output.width, m_wanted.output.height, mode_name(m_wanted.quality),
             static_cast<double>(m_vramBytes) / MiB);
  }
  return m_api->evaluate(list, m_feature, evaluation(inputs));
}

void DlssUpscaler::destroy() noexcept {
  end_session();
  m_modes = {};
  m_modesOutput = {};
  m_wanted = {};
  m_waiting = false;
}

uint32_t DlssUpscaler::jitter_phases(float ratio) const noexcept {
  return std::max(MinJitterPhases, static_cast<uint32_t>(std::lround(8.f * ratio * ratio)));
}

float DlssUpscaler::mip_bias(float ratio) const noexcept { return std::log2(1.f / ratio); }

void DlssUpscaler::want_feature(const FeatureDesc& desc) noexcept {
  m_wanted = desc;
  if (m_feature != nullptr && m_featureDesc != desc) {
    release_feature();
  }
}

void DlssUpscaler::release_feature() noexcept {
  if (m_feature == nullptr) {
    return;
  }
  wait_last_use();
  drop_feature();
}

void DlssUpscaler::drop_feature() noexcept {
  if (m_feature == nullptr) {
    return;
  }
  m_api->release(std::exchange(m_feature, nullptr));
  m_featureDesc = {};
  m_vramBytes = 0;
}
} // namespace aurora::render::detail::native::d3d12::dlss

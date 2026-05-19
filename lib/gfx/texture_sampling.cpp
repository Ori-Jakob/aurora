#include "texture_sampling.hpp"

#include <algorithm>
#include <cmath>

#include <dolphin/gx/GXTextureSampling.h>

#include "../gx/gx.hpp"

#include <unordered_map>

namespace aurora::gfx {
namespace {
bool s_stochasticSamplingEnabled = false;
Vec4<float> s_stochasticSamplingParams{1.0f, 1.0f, 0.25f, 0.0f};
std::unordered_map<u32, Vec4<float>> s_stochasticSamplingParamOverrides;

Vec4<float> clamp_sampling_params(float cellScale, float jitter, float blendWidth) noexcept {
  return {
      std::clamp(cellScale, 0.1f, 16.0f),
      std::clamp(jitter, 0.0f, 2.5f),
      std::clamp(blendWidth, 0.001f, 4.0f),
      0.0f,
  };
}
}

bool stochastic_sampling_enabled() noexcept { return s_stochasticSamplingEnabled; }

void set_stochastic_sampling_enabled(bool enabled) noexcept {
  s_stochasticSamplingEnabled = enabled;
  gx::g_gxState.stateDirty = true;
}

void set_stochastic_sampling_params(float cellScale, float jitter, float blendWidth) noexcept {
  s_stochasticSamplingParams = clamp_sampling_params(cellScale, jitter, blendWidth);
  gx::g_gxState.stateDirty = true;
}

Vec4<float> stochastic_sampling_params() noexcept { return s_stochasticSamplingParams; }

bool uses_stochastic_sampling(const GXTexObj_& obj) noexcept {
  return (obj.flags & TextureFlagStochasticSampling) != 0;
}

void set_stochastic_sampling(GXTexObj_& obj, bool enabled) noexcept {
  if (enabled) {
    obj.flags |= TextureFlagStochasticSampling;
  } else {
    obj.flags &= ~TextureFlagStochasticSampling;
  }
}

void set_stochastic_sampling_params_override(const GXTexObj_& obj, float cellScale, float jitter,
                                             float blendWidth) noexcept {
  if (obj.texObjId == 0) {
    return;
  }
  s_stochasticSamplingParamOverrides[obj.texObjId] = clamp_sampling_params(cellScale, jitter, blendWidth);
  gx::g_gxState.stateDirty = true;
}

void clear_stochastic_sampling_params_override(const GXTexObj_& obj) noexcept {
  if (obj.texObjId == 0) {
    return;
  }
  s_stochasticSamplingParamOverrides.erase(obj.texObjId);
  gx::g_gxState.stateDirty = true;
}

bool stochastic_sampling_params_override(const GXTexObj_& obj, Vec4<float>& params) noexcept {
  if (obj.texObjId == 0) {
    return false;
  }
  const auto it = s_stochasticSamplingParamOverrides.find(obj.texObjId);
  if (it == s_stochasticSamplingParamOverrides.end()) {
    return false;
  }
  params = it->second;
  return true;
}

void erase_stochastic_sampling_params_override(u32 texObjId) noexcept {
  s_stochasticSamplingParamOverrides.erase(texObjId);
}

Vec4<float> texture_size_bias_sampling_params(const TextureBind& tex) noexcept {
  auto width = static_cast<float>(tex.texObj.width());
  auto height = static_cast<float>(tex.texObj.height());
  const auto vpBias =
      gx::enableLodBias && tex.ref && tex.ref->hasArbitraryMips
          ? log2(std::min(gx::g_gxState.renderViewport.width / std::max(gx::g_gxState.logicalViewport.width, 1.f),
                          gx::g_gxState.renderViewport.height / std::max(gx::g_gxState.logicalViewport.height, 1.f)))
          : 0.f;
  const float stochasticFlag = s_stochasticSamplingEnabled && uses_stochastic_sampling(tex.texObj) ? 1.0f : 0.0f;
  return {width, height, tex.texObj.lod_bias() + vpBias, stochasticFlag};
}

Vec4<float> texture_stochastic_sampling_params(const TextureBind& tex) noexcept {
  Vec4<float> params;
  if (stochastic_sampling_params_override(tex.texObj, params)) {
    return params;
  }
  return stochastic_sampling_params();
}
} // namespace aurora::gfx

extern "C" {
void AuroraSetStochasticSamplingEnabled(GXBool enabled) {
  aurora::gfx::set_stochastic_sampling_enabled(enabled != GX_FALSE);
}

GXBool AuroraGetStochasticSamplingEnabled(void) {
  return aurora::gfx::stochastic_sampling_enabled() ? GX_TRUE : GX_FALSE;
}

void AuroraSetStochasticSamplingParams(f32 cell_scale, f32 jitter, f32 blend_width) {
  aurora::gfx::set_stochastic_sampling_params(cell_scale, jitter, blend_width);
}

void AuroraSetTexObjStochasticSampling(GXTexObj* obj_, GXBool enabled) {
  if (obj_ == nullptr) {
    return;
  }
  auto* obj = reinterpret_cast<GXTexObj_*>(obj_);
  aurora::gfx::set_stochastic_sampling(*obj, enabled != GX_FALSE);
}

GXBool AuroraGetTexObjStochasticSampling(const GXTexObj* obj_) {
  if (obj_ == nullptr) {
    return GX_FALSE;
  }
  const auto* obj = reinterpret_cast<const GXTexObj_*>(obj_);
  return aurora::gfx::uses_stochastic_sampling(*obj) ? GX_TRUE : GX_FALSE;
}

void AuroraSetTexObjStochasticSamplingParams(GXTexObj* obj_, f32 cell_scale, f32 jitter, f32 blend_width) {
  if (obj_ == nullptr) {
    return;
  }
  const auto* obj = reinterpret_cast<const GXTexObj_*>(obj_);
  aurora::gfx::set_stochastic_sampling_params_override(*obj, cell_scale, jitter, blend_width);
}

void AuroraClearTexObjStochasticSamplingParams(GXTexObj* obj_) {
  if (obj_ == nullptr) {
    return;
  }
  const auto* obj = reinterpret_cast<const GXTexObj_*>(obj_);
  aurora::gfx::clear_stochastic_sampling_params_override(*obj);
}

GXBool AuroraGetTexObjStochasticSamplingParams(const GXTexObj* obj_, f32* cell_scale, f32* jitter, f32* blend_width) {
  if (obj_ == nullptr) {
    return GX_FALSE;
  }

  const auto* obj = reinterpret_cast<const GXTexObj_*>(obj_);
  aurora::Vec4<float> params;
  if (!aurora::gfx::stochastic_sampling_params_override(*obj, params)) {
    return GX_FALSE;
  }

  if (cell_scale != nullptr) {
    *cell_scale = params.x();
  }
  if (jitter != nullptr) {
    *jitter = params.y();
  }
  if (blend_width != nullptr) {
    *blend_width = params.z();
  }
  return GX_TRUE;
}
}

#include "stochastic_sampling.hpp"

#include <fmt/format.h>

namespace aurora::gx {
std::string_view stochastic_sampling_wgsl() noexcept {
  return R"WGSL(
fn aurora_hash22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += vec3f(dot(p3, p3.yzx + vec3f(33.33)));
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn aurora_stochastic_sample(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  bias: f32,
  params: vec4f
) -> vec4f {
  let cellScale = max(params.x, 0.001);
  let jitter = max(params.y, 0.0);
  let blendWidth = clamp(params.z, 0.001, 4.0);
  let cellUv = uv * cellScale;
  let cell = floor(cellUv);
  let blend = smoothstep(vec2f(0.5 - blendWidth), vec2f(0.5 + blendWidth), fract(cellUv));
  let offset00 = aurora_hash22(cell + vec2f(0.0, 0.0)) - vec2f(0.5);
  let offset10 = aurora_hash22(cell + vec2f(1.0, 0.0)) - vec2f(0.5);
  let offset01 = aurora_hash22(cell + vec2f(0.0, 1.0)) - vec2f(0.5);
  let offset11 = aurora_hash22(cell + vec2f(1.0, 1.0)) - vec2f(0.5);
  let sample00 = textureSampleBias(tex, samp, uv + offset00 * jitter, bias);
  let sample10 = textureSampleBias(tex, samp, uv + offset10 * jitter, bias);
  let sample01 = textureSampleBias(tex, samp, uv + offset01 * jitter, bias);
  let sample11 = textureSampleBias(tex, samp, uv + offset11 * jitter, bias);
  return mix(mix(sample00, sample10, blend.x), mix(sample01, sample11, blend.x), blend.y);
}

fn aurora_sample_texture(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  size_bias_flags: vec4f,
  params: vec4f
) -> vec4f {
  if (size_bias_flags.w < 0.5) {
    return textureSampleBias(tex, samp, uv, size_bias_flags.z);
  }
  return aurora_stochastic_sample(tex, samp, uv, size_bias_flags.z, params);
}
)WGSL";
}

std::string sampled_texture_expr(uint32_t texMapId, std::string_view uvExpr) {
  return fmt::format("aurora_sample_texture(tex{0}, tex{0}_samp, {1}, ubuf.tex{0}_size_bias, "
                     "ubuf.tex{0}_stochastic_params)",
                     texMapId, uvExpr);
}
} // namespace aurora::gx

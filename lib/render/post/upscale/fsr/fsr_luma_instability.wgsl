// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_luma_instability.h and ffx_fsr3upscaler_luma_instability_pass.hlsl), which carry this notice:
//
// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//#import post/upscale/fsr/fsr_common

@group(1) @binding(0) var dilated_reactive: texture_2d<f32>;
@group(1) @binding(1) var dilated_motion: texture_2d<f32>;
@group(1) @binding(2) var previous_luma_history: texture_2d<f32>;
@group(1) @binding(3) var current_luma: texture_2d<f32>;
@group(1) @binding(4) var luma_history: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var luma_instability: texture_storage_2d<r16float, write>;

struct InstabilityData {
  history: vec4f,
  factor: f32,
}

fn compute_luma_instability_factor(history: vec4f, current: f32) -> InstabilityData {
  var instability = 0.0;
  let diffs0 = current - history[0];
  let similarity0 = min_divided_by_max_or(current, history[0], 1.0);
  var max_similarity = similarity0;
  if (similarity0 < 1.0) {
    for (var i = 1; i <= 3; i++) {
      let diffs1 = current - history[i];
      let similarity1 = min_divided_by_max(current, history[i]);
      if (sign(diffs0) == sign(diffs1)) {
        max_similarity = max(max_similarity, similarity1);
      }
    }
    instability = f32(max_similarity > similarity0);
  }
  let shifted = vec4f(current, history.xyz);
  return InstabilityData(shifted, instability * f32(shifted.w != 0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  let size = render_size();
  if (!is_on_screen(p, size)) {
    return;
  }
  var data = InstabilityData(vec4f(0.0), 0.0);
  let motion = textureLoad(dilated_motion, p, 0).xy;
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let uv_current = uv + fsr.jitter / vec2f(size);
  let reprojected = uv + fsr.prev_jitter / vec2f(size) + motion;
  if (is_uv_inside(reprojected)) {
    let uv_hw = clamp_uv(uv_current, size, size);
    let masks = textureSampleLevel(dilated_reactive, linear_clamp, uv_hw, 0.0);
    let reactive = saturate(masks.r);
    let disocclusion = saturate(masks.g);
    let change = saturate(masks.b);
    let accumulated = saturate(masks.a);
    if (accumulated > 0.9) {
      let current = textureSampleLevel(current_luma, linear_clamp, uv_hw, 0.0).r;
      let history = textureSampleLevel(previous_luma_history, linear_clamp, clamp_uv(reprojected, size, size), 0.0);
      data = compute_luma_instability_factor(history, current);
      let velocity_weight = 1.0 - saturate(get_4k_velocity(motion) / 20.0);
      data.factor *= velocity_weight * (1.0 - disocclusion) * (1.0 - reactive) * (1.0 - change);
    }
  }
  textureStore(luma_history, p, data.history);
  textureStore(luma_instability, p, vec4f(data.factor, 0.0, 0.0, 0.0));
}

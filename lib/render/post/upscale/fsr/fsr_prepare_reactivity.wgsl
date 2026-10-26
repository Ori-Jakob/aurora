// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_prepare_reactivity.h and ffx_fsr3upscaler_prepare_reactivity_pass.hlsl), which carry this notice:
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

@group(1) @binding(0) var<storage, read> reconstructed_depth: array<u32>;
@group(1) @binding(1) var dilated_motion: texture_2d<f32>;
@group(1) @binding(2) var dilated_depth: texture_2d<f32>;
@group(1) @binding(3) var reactive_mask: texture_2d<f32>;
@group(1) @binding(4) var previous_accumulation: texture_2d<f32>;
@group(1) @binding(5) var shading_change: texture_2d<f32>;
@group(1) @binding(6) var current_luma: texture_2d<f32>;
@group(1) @binding(7) var dilated_reactive: texture_storage_2d<rgba8unorm, write>;
//#if FSR_LOCKS_R8
@group(1) @binding(8) var new_locks: texture_storage_2d<r8unorm, write>;
//#else
@group(1) @binding(8) var new_locks: texture_storage_2d<r32float, write>;
//#endif
@group(1) @binding(9) var accumulation: texture_storage_2d<r8unorm, write>;

fn load_reconstructed_depth(p: vec2i) -> f32 {
  let size = render_size();
  let q = clamp_load(p, vec2i(0), size);
  return bitcast<f32>(reconstructed_depth[u32(q.y) * u32(size.x) + u32(q.x)]);
}

fn load_motion(p: vec2i) -> vec2f {
  return textureLoad(dilated_motion, clamp_load(p, vec2i(0), render_size()), 0).xy;
}

fn load_depth(p: vec2i) -> f32 {
  return textureLoad(dilated_depth, clamp_load(p, vec2i(0), render_size()), 0).r;
}

fn compute_disocclusions(uv: vec2f, motion: vec2f, current_depth_view: f32) -> f32 {
  let size = render_size();
  let nearest_depth_m = min(current_depth_view * fsr.view_to_meters, FP16_MAX);
  let threshold = reconstructed_depth_mv_px_threshold(nearest_depth_m);
  let kept = motion * f32(get_4k_velocity(motion) > threshold);
  let bilinear = get_bilinear_sampling_data(uv + kept, size);
  var disocclusion = 0.0;
  var weight_sum = 0.0;
  var potential = true;
  for (var i = 0; i < 4 && potential; i++) {
    let sample_pos = clamp_load(bilinear.base, bilinear_offset(i), size);
    let weight = bilinear.weights[i];
    if (weight > RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD) {
      let prev_depth_view = get_view_space_depth(load_reconstructed_depth(sample_pos));
      let depth_difference = current_depth_view - prev_depth_view;
      potential = potential && depth_difference > FP32_MIN;
      if (potential) {
        let half_viewport_width = length(vec2f(size) * 0.5);
        let depth_threshold = max(current_depth_view, prev_depth_view);
        let required_separation = 1.37e-05 * half_viewport_width * depth_threshold;
        disocclusion += saturate(required_separation / depth_difference) * weight;
        weight_sum += weight;
      }
    }
  }
  return select(0.0, saturate(1.0 - disocclusion / weight_sum), potential && weight_sum > 0.0);
}

fn compute_motion_divergence(uv: vec2f, motion: vec2f, depth: f32) -> f32 {
  let velocity = get_4k_velocity(motion);
  if (velocity <= 0.0) {
    return 0.0;
  }
  let reprojected = vec2i((uv + motion) * vec2f(render_size()));
  let reprojected_depth = load_depth(reprojected);
  let reprojected_velocity = get_4k_velocity(load_motion(reprojected));
  let nucleus_m = get_view_space_depth_in_meters(reprojected_depth);
  let current_m = get_view_space_depth_in_meters(depth);
  let distance_factor = min_divided_by_max(nucleus_m, current_m);
  let velocity_factor = saturate(velocity / 10.0);
  return (1.0 - saturate(reprojected_velocity / velocity)) * distance_factor * velocity_factor;
}

fn load_reactive(p: vec2i) -> f32 {
  let reactive = min(textureLoad(reactive_mask, p, 0).r, fsr.reactive_max) * fsr.reactiveness_scale;
  let invalid = any(abs(load_motion(p)) >= vec2f(INVALID_MOTION));
  return select(reactive, max(reactive, INVALID_MOTION_REACTIVE), invalid);
}

fn dilate_reactive_masks(p: vec2i) -> f32 {
  var dilated = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      dilated = max(dilated, load_reactive(clamp_load(p, vec2i(x, y), render_size())));
    }
  }
  return dilated;
}

fn compute_thin_feature_confidence(p: vec2i) -> f32 {
  let size = render_size();
  var offsets = array<vec2i, 9>(vec2i(0, 0), vec2i(-1, -1), vec2i(0, -1), vec2i(1, -1), vec2i(-1, 0), vec2i(1, 0),
                                vec2i(-1, 1), vec2i(0, 1), vec2i(1, 1));
  var samples: array<f32, 9>;
  var luma_min = FP32_MAX;
  var luma_max = FP32_MIN;
  for (var i = 0; i < 9; i++) {
    samples[i] = textureLoad(current_luma, clamp_load(p, offsets[i], size), 0).r;
    luma_min = min(luma_min, samples[i]);
    luma_max = max(luma_max, samples[i]);
  }
  let range = luma_max - luma_min;
  if (range <= 0.0) {
    return 0.0;
  }
  var dissimilar_min = FP32_MAX;
  var dissimilar_max = 0.0;
  var pattern = 1u;
  for (var i = 1; i < 9; i++) {
    if (abs(samples[i] - samples[0]) / range < 0.9) {
      pattern |= 1u << u32(i);
    } else {
      dissimilar_min = min(dissimilar_min, samples[i]);
      dissimilar_max = max(dissimilar_max, samples[i]);
    }
  }
  if (!(samples[0] > dissimilar_max || samples[0] < dissimilar_min)) {
    return 0.0;
  }
  var rejection = array<u32, 4>(0x17u, 0x2Du, 0xD1u, 0x1A1u);
  for (var i = 0; i < 4; i++) {
    if ((pattern & rejection[i]) == rejection[i]) {
      return 0.0;
    }
  }
  return 1.0 - luma_min / luma_max;
}

fn update_accumulation(p: vec2i, uv: vec2f, motion: vec2f, disocclusion: f32, change: f32) -> f32 {
  let size = render_size();
  let reprojected = uv + motion;
  var accumulated = 0.0;
  if (is_uv_inside(reprojected) && fsr.frame_index > 0.0) {
    accumulated = saturate(textureSampleLevel(previous_accumulation, linear_clamp, clamp_uv(reprojected, size, size),
                                              0.0).r);
  }
  accumulated = mix(accumulated, 0.0, change);
  accumulated = mix(accumulated, min(fsr.min_disocclusion_accumulation, accumulated), disocclusion);
  accumulated *= f32(round(accumulated * 100.0) > 1.0);
  textureStore(accumulation, p, vec4f(saturate(accumulated + fsr.accumulation_per_frame), 0.0, 0.0, 0.0));
  return accumulated;
}

fn compute_shading_change(uv: vec2f) -> f32 {
  let size = vec2i(vec2f(render_size()) * 0.5);
  let change_uv = clamp_uv(uv - fsr.jitter / vec2f(render_size()), size, vec2i(textureDimensions(shading_change)));
  return saturate(textureSampleLevel(shading_change, linear_clamp, change_uv, 0.0).r * fsr.shading_change_scale);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  let size = render_size();
  if (!is_on_screen(p, size)) {
    return;
  }
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let motion = textureLoad(dilated_motion, p, 0).xy;
  let depth = textureLoad(dilated_depth, p, 0).r;
  let disocclusion = compute_disocclusions(uv, motion, get_view_space_depth(depth));
  let change = max(dilate_reactive_masks(p), compute_shading_change(uv));
  let reactive = compute_motion_divergence(uv, motion, depth);
  let accumulated = update_accumulation(p, uv, motion, disocclusion, change);
  textureStore(dilated_reactive, p, vec4f(reactive, disocclusion, change, accumulated));
  let lock = compute_thin_feature_confidence(p);
  if (lock > 0.01) {
    let upscale = vec2i(textureDimensions(new_locks));
    let hr = compute_hr_pos_from_lr_pos(p, upscale);
    if (is_on_screen(hr, upscale)) {
      textureStore(new_locks, hr, vec4f(lock, 0.0, 0.0, 0.0));
    }
  }
}

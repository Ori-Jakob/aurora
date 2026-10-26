// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_shading_change_pyramid.h, ffx_fsr3upscaler_shading_change_pyramid_pass.hlsl and spd/ffx_spd.h),
// which carry this notice:
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

@group(1) @binding(0) var current_luma: texture_2d<f32>;
@group(1) @binding(1) var previous_luma: texture_2d<f32>;
@group(1) @binding(2) var dilated_motion: texture_2d<f32>;
@group(1) @binding(3) var spd_level0: texture_storage_2d<rg16float, write>;
@group(1) @binding(4) var spd_level1: texture_storage_2d<rg16float, write>;
@group(1) @binding(5) var spd_level2: texture_storage_2d<rg16float, write>;

const SET_SIZE: i32 = 5;
const STORED_MIPS: u32 = 3u;

fn set_offset(index: i32) -> vec2i {
  var offsets = array<vec2i, 5>(vec2i(0, 0), vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
  return offsets[index];
}

fn compare_swap(values: ptr<function, array<f32, 5>>, i: i32, j: i32) {
  let lo = min((*values)[i], (*values)[j]);
  (*values)[j] = max((*values)[i], (*values)[j]);
  (*values)[i] = lo;
}

fn sort_set(values: ptr<function, array<f32, 5>>) {
  compare_swap(values, 0, 3);
  compare_swap(values, 1, 4);
  compare_swap(values, 0, 2);
  compare_swap(values, 1, 3);
  compare_swap(values, 0, 1);
  compare_swap(values, 2, 4);
  compare_swap(values, 1, 2);
  compare_swap(values, 3, 4);
  compare_swap(values, 2, 3);
}

fn compute_minimum_difference(current: array<f32, 5>, previous: array<f32, 5>) -> f32 {
  var set0 = current;
  var set1 = previous;
  sort_set(&set0);
  sort_set(&set1);
  var min_diff = FP16_MAX - 1.0;
  var a = 0;
  var b = 0;
  let largest = min(set0[SET_SIZE - 1], set1[SET_SIZE - 1]);
  if (largest > FP32_MIN) {
    for (var i = 0; i < SET_SIZE && min_diff < FP16_MAX; i++) {
      var diff = set0[min(a, SET_SIZE - 1)] - set1[min(b, SET_SIZE - 1)];
      if (abs(diff) > FP16_MIN) {
        diff = sign(diff) * (1.0 - min_divided_by_max(set0[min(a, SET_SIZE - 1)], set1[min(b, SET_SIZE - 1)]));
        min_diff = select(min_diff, diff, abs(diff) < abs(min_diff));
        a += i32(set0[min(a, SET_SIZE - 1)] < set1[min(b, SET_SIZE - 1)]);
        b += i32(set0[min(a, SET_SIZE - 1)] >= set1[min(b, SET_SIZE - 1)]);
      } else {
        min_diff = FP16_MAX;
      }
    }
  }
  return min_diff * f32(min_diff < FP16_MAX - 1.0);
}

fn compute_diff(uv: vec2f, motion: vec2f) -> f32 {
  let size = render_size();
  let current_base = vec2i(floor((uv + fsr.jitter / vec2f(size)) * vec2f(size)));
  var current: array<f32, 5>;
  for (var i = 0; i < SET_SIZE; i++) {
    current[i] = max(textureLoad(current_luma, clamp_load(current_base, set_offset(i), size), 0).r, EPSILON);
  }
  let reprojected = uv + fsr.prev_jitter / vec2f(size) + motion;
  if (!is_uv_inside(reprojected)) {
    return 0.0;
  }
  let previous_base = vec2i(floor(reprojected * vec2f(size)));
  var previous: array<f32, 5>;
  for (var i = 0; i < SET_SIZE; i++) {
    previous[i] = max(textureLoad(previous_luma, clamp_load(previous_base, set_offset(i), size), 0).r, EPSILON);
  }
  return compute_minimum_difference(current, previous);
}

fn load_source(p: vec2i) -> vec4f {
  let size = render_size();
  let q = clamp_load(p, vec2i(0), size);
  let uv = (vec2f(q) + 0.5) / vec2f(size);
  let diff = compute_diff(uv, textureLoad(dilated_motion, q, 0).xy);
  return vec4f(diff, select(0.0, sign(diff), diff != 0.0), 1.0, 0.0);
}

fn reduce_source(base: vec2u) -> vec4f {
  let p = vec2i(base);
  return spd_reduce4(load_source(p), load_source(p + vec2i(0, 1)), load_source(p + vec2i(1, 0)),
                     load_source(p + vec2i(1, 1)));
}

fn store(pix: vec2i, v: vec4f, mip: u32) {
  let value = vec4f(v.xy, 0.0, 0.0);
  if (mip == 0u && is_on_screen(pix, vec2i(textureDimensions(spd_level0)))) {
    textureStore(spd_level0, pix, value);
  } else if (mip == 1u && is_on_screen(pix, vec2i(textureDimensions(spd_level1)))) {
    textureStore(spd_level1, pix, value);
  } else if (mip == 2u && is_on_screen(pix, vec2i(textureDimensions(spd_level2)))) {
    textureStore(spd_level2, pix, value);
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group_id: vec3u, @builtin(local_invocation_index) index: u32) {
  let t = spd_thread(index);
  let wg = group_id.xy;
  var v: array<vec4f, 4>;
  for (var i = 0u; i < 4u; i++) {
    let q = vec2u(i % 2u, i / 2u);
    v[i] = reduce_source(wg * 64u + t * 2u + q * 32u);
    store(vec2i(wg * 32u + t + q * 16u), v[i], 0u);
  }
  for (var i = 0u; i < 4u; i++) {
    v[i] = spd_mip1(t, index, v[i]);
    if (index < 64u) {
      store(vec2i(wg * 16u + t + vec2u(i % 2u, i / 2u) * 8u), v[i], 1u);
    }
  }
  spd_keep_mip1(t, index, v);
  workgroupBarrier();
  let m2 = spd_mip2(t, index);
  if (index < 64u) {
    store(vec2i(wg * 8u + t), m2, STORED_MIPS - 1u);
  }
}

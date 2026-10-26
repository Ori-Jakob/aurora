// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_luma_pyramid.h, ffx_fsr3upscaler_luma_pyramid_pass.hlsl and spd/ffx_spd.h), which carry this
// notice:
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
@group(1) @binding(1) var farthest_depth: texture_2d<f32>;
@group(1) @binding(2) var previous_frame_info: texture_2d<f32>;
@group(1) @binding(3) var farthest_depth_mip1: texture_storage_2d<r16float, write>;
@group(1) @binding(4) var luma_mip5: texture_storage_2d<rg16float, write>;
@group(1) @binding(5) var frame_info: texture_storage_2d<rgba32float, write>;

fn load_source(p: vec2i) -> vec4f {
  let q = clamp_load(p, vec2i(0), render_size());
  let luma = textureLoad(current_luma, q, 0).r;
  let log_luma = max(EPSILON, log(max(luma, FP32_MIN)));
  return vec4f(log_luma, luma, textureLoad(farthest_depth, q, 0).r, 0.0);
}

fn reduce_source(base: vec2u) -> vec4f {
  let p = vec2i(base);
  return spd_reduce4(load_source(p), load_source(p + vec2i(0, 1)), load_source(p + vec2i(1, 0)),
                     load_source(p + vec2i(1, 1)));
}

fn store(pix: vec2i, v: vec4f, mip: u32) {
  if (mip == 5u && is_on_screen(pix, vec2i(textureDimensions(luma_mip5)))) {
    textureStore(luma_mip5, pix, vec4f(v.xy, 0.0, 0.0));
  } else if (mip == 0u && is_on_screen(pix, vec2i(textureDimensions(farthest_depth_mip1)))) {
    textureStore(farthest_depth_mip1, pix, vec4f(v.z, 0.0, 0.0, 0.0));
  }
  if (mip == fsr.spd_mips - 1u && all(pix == vec2i(0))) {
    textureStore(frame_info, vec2i(0), next_frame_info(textureLoad(previous_frame_info, vec2i(0), 0), v));
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group_id: vec3u, @builtin(local_invocation_index) index: u32) {
  let mips = fsr.spd_mips;
  let t = spd_thread(index);
  let wg = group_id.xy;
  var v: array<vec4f, 4>;
  for (var i = 0u; i < 4u; i++) {
    let q = vec2u(i % 2u, i / 2u);
    v[i] = reduce_source(wg * 64u + t * 2u + q * 32u);
    store(vec2i(wg * 32u + t + q * 16u), v[i], 0u);
  }
  if (mips <= 1u) {
    return;
  }
  for (var i = 0u; i < 4u; i++) {
    v[i] = spd_mip1(t, index, v[i]);
    if (index < 64u) {
      store(vec2i(wg * 16u + t + vec2u(i % 2u, i / 2u) * 8u), v[i], 1u);
    }
  }
  spd_keep_mip1(t, index, v);
  if (mips <= 2u) {
    return;
  }
  workgroupBarrier();
  let m2 = spd_mip2(t, index);
  if (index < 64u) {
    store(vec2i(wg * 8u + t), m2, 2u);
  }
  if (mips <= 3u) {
    return;
  }
  workgroupBarrier();
  let m3 = spd_mip3(t, index);
  if (index < 16u) {
    store(vec2i(wg * 4u + t), m3, 3u);
  }
  if (mips <= 4u) {
    return;
  }
  workgroupBarrier();
  let m4 = spd_mip4(t, index);
  if (index < 4u) {
    store(vec2i(wg * 2u + t), m4, 4u);
  }
  if (mips <= 5u) {
    return;
  }
  workgroupBarrier();
  let m5 = spd_mip5(index);
  if (index < 1u) {
    store(vec2i(wg), m5, 5u);
  }
}

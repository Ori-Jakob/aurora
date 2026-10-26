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

@group(1) @binding(0) var luma_mip5: texture_2d<f32>;
@group(1) @binding(1) var previous_frame_info: texture_2d<f32>;
@group(1) @binding(2) var frame_info: texture_storage_2d<rgba32float, write>;

fn load_mip5(p: vec2i) -> vec4f {
  if (!is_on_screen(p, vec2i(textureDimensions(luma_mip5)))) {
    return vec4f(0.0);
  }
  return vec4f(textureLoad(luma_mip5, p, 0).xy, 0.0, 0.0);
}

fn reduce_mip5(base: vec2u) -> vec4f {
  let p = vec2i(base);
  return spd_reduce4(load_mip5(p), load_mip5(p + vec2i(0, 1)), load_mip5(p + vec2i(1, 0)), load_mip5(p + vec2i(1, 1)));
}

fn store(pix: vec2i, v: vec4f, mip: u32) {
  if (mip == fsr.spd_mips - 1u && all(pix == vec2i(0))) {
    textureStore(frame_info, vec2i(0), next_frame_info(textureLoad(previous_frame_info, vec2i(0), 0), v));
  }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) index: u32) {
  let mips = fsr.spd_mips;
  if (mips <= 6u) {
    return;
  }
  let t = spd_thread(index);
  var v: array<vec4f, 4>;
  for (var i = 0u; i < 4u; i++) {
    let q = vec2u(i % 2u, i / 2u);
    v[i] = reduce_mip5(t * 4u + q * 2u);
    store(vec2i(t * 2u + q), v[i], 6u);
  }
  if (mips <= 7u) {
    return;
  }
  let m7 = spd_reduce4(v[0], v[1], v[2], v[3]);
  store(vec2i(t), m7, 7u);
  spd_store_intermediate(t, m7);
  if (mips <= 8u) {
    return;
  }
  workgroupBarrier();
  let m8 = spd_mip2(t, index);
  if (index < 64u) {
    store(vec2i(t), m8, 8u);
  }
  if (mips <= 9u) {
    return;
  }
  workgroupBarrier();
  let m9 = spd_mip3(t, index);
  if (index < 16u) {
    store(vec2i(t), m9, 9u);
  }
  if (mips <= 10u) {
    return;
  }
  workgroupBarrier();
  let m10 = spd_mip4(t, index);
  if (index < 4u) {
    store(vec2i(t), m10, 10u);
  }
  if (mips <= 11u) {
    return;
  }
  workgroupBarrier();
  let m11 = spd_mip5(index);
  if (index < 1u) {
    store(vec2i(0), m11, 11u);
  }
}

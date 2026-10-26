// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_shading_change.h and ffx_fsr3upscaler_shading_change_pass.hlsl), which carry this notice:
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

@group(1) @binding(0) var spd: texture_2d<f32>;
@group(1) @binding(1) var shading_change: texture_storage_2d<r8unorm, write>;

const MIP_LEVELS: i32 = 3;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  let size = vec2i(vec2f(render_size()) * 0.5);
  if (!is_on_screen(p, size)) {
    return;
  }
  let uv = (vec2f(p) + 0.5) / vec2f(size) + fsr.jitter / vec2f(render_size());
  let mip_uv = clamp_uv(uv, size, vec2i(textureDimensions(spd, 0)));
  var change = 0.0;
  for (var mip = 0; mip < MIP_LEVELS; mip++) {
    let s = textureSampleLevel(spd, linear_clamp, mip_uv, f32(mip)).xy;
    let luma = abs(s.x * s.y);
    if (luma > 0.0) {
      change = max(change, luma);
    }
  }
  textureStore(shading_change, p, vec4f(saturate(change), 0.0, 0.0, 0.0));
}

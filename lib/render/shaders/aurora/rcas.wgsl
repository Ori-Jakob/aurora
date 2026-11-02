// The sharpening follows RCAS from AMD FidelityFX FSR 1.0 (GPUOpen-Effects/FidelityFX-FSR), which carries this
// notice:
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

const RCAS_LIMIT: f32 = 0.1875;

fn rcas_tap(t: texture_2d<f32>, p: vec2i) -> vec3f {
  let last = vec2i(textureDimensions(t)) - 1;
  return textureLoad(t, clamp(p, vec2i(0), last), 0).rgb;
}

fn rcas_max3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn rcas_sharpen(t: texture_2d<f32>, pixel: vec2i, sharpness: f32) -> vec3f {
  let e = rcas_tap(t, pixel);
  if (!(sharpness > 0.0)) {
    return e;
  }
  let b = rcas_tap(t, pixel + vec2i(0, -1));
  let d = rcas_tap(t, pixel + vec2i(-1, 0));
  let f = rcas_tap(t, pixel + vec2i(1, 0));
  let h = rcas_tap(t, pixel + vec2i(0, 1));
  return rcas_filter(e, b, d, f, h, sharpness);
}

// Callers that hold linear colour pass stored-space taps, so one strength sharpens every resolver alike
fn rcas_filter(e: vec3f, b: vec3f, d: vec3f, f: vec3f, h: vec3f, sharpness: f32) -> vec3f {
  let mn4 = min(min(b, d), min(f, h));
  let mx4 = max(max(b, d), max(f, h));
  let hit_min = min(mn4, e) / max(4.0 * mx4, vec3f(0.00001));
  let hit_max = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3f(-0.00001));
  let lobe = max(-RCAS_LIMIT, min(rcas_max3(max(-hit_min, hit_max)), 0.0)) * min(sharpness, 1.0);
  let color = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
  return saturate(color);
}

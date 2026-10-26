// The resolve follows Bevy's taa.wgsl (bevyengine/bevy, used under its MIT option), which carries this notice:
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The history filter follows MJP's Catmull-Rom gist (TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1), which carries
// this notice:
//
// MIT License
//
// Copyright (c) 2019 MJP
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
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

//#import aurora/frame

struct TaaParams {
  sharpness: f32,
  flags: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(1) var<uniform> taa_params: TaaParams;
@group(0) @binding(2) var linear_clamp: sampler;

@group(1) @binding(0) var scene_depth: texture_2d<f32>;
@group(1) @binding(1) var scene_motion: texture_2d<f32>;
@group(1) @binding(2) var scene_reactive: texture_2d<f32>;

@group(2) @binding(0) var source_tex: texture_2d<f32>;
@group(2) @binding(1) var history_tex: texture_2d<f32>;

override MIN_BLEND: f32 = 0.015;
override MOVING_SAMPLES: f32 = 10.0;

const RCAS_LIMIT: f32 = 0.1875;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn load_clamped(t: texture_2d<f32>, p: vec2i) -> vec4f {
  let last = vec2i(textureDimensions(t)) - 1;
  return textureLoad(t, clamp(p, vec2i(0), last), 0);
}

fn max3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn finite2(v: vec2f) -> bool {
  let bits = bitcast<vec2u>(v) & vec2u(0x7f800000u);
  return all(bits != vec2u(0x7f800000u));
}

fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn ycocg_to_rgb(c: vec3f) -> vec3f {
  return saturate(vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z));
}

fn clip_towards_aabb_center(history: vec3f, box_min: vec3f, box_max: vec3f) -> vec3f {
  let center = 0.5 * (box_max + box_min);
  let extent = 0.5 * (box_max - box_min) + 0.00000001;
  let offset = history - center;
  let units = max3(abs(offset / extent));
  if (units > 1.0) {
    return center + offset / units;
  }
  return history;
}

fn sample_history(uv: vec2f) -> vec3f {
  return textureSampleLevel(history_tex, linear_clamp, uv, 0.0).rgb;
}

fn history_catmull_rom(position: vec2f, inv_size: vec2f) -> vec3f {
  let center = floor(position - 0.5) + 0.5;
  let f = position - center;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let p0 = (center - 1.0) * inv_size;
  let p3 = (center + 2.0) * inv_size;
  let p12 = (center + w2 / w12) * inv_size;
  let weights = vec4f(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
  let middle = w12.x * w12.y;
  var color = sample_history(vec2f(p12.x, p0.y)) * weights.x;
  color += sample_history(vec2f(p0.x, p12.y)) * weights.y;
  color += sample_history(p12) * middle;
  color += sample_history(vec2f(p3.x, p12.y)) * weights.z;
  color += sample_history(vec2f(p12.x, p3.y)) * weights.w;
  return color / (dot(weights, vec4f(1.0)) + middle);
}

@fragment
fn fs_resolve(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(floor(frag.xy));
  let current = load_clamped(source_tex, pixel).rgb;
//#if TAA_RESET
  return vec4f(current, 1.0);
//#else
  let size = vec2f(textureDimensions(source_tex));

  var closest = pixel;
  var closest_depth = load_clamped(scene_depth, pixel).r;
  for (var i = 0; i < 4; i++) {
    let tap = pixel + vec2i(select(-2, 2, (i & 1) != 0), select(-2, 2, (i & 2) != 0));
    let depth = load_clamped(scene_depth, tap).r;
    if (depth > closest_depth) {
      closest = tap;
      closest_depth = depth;
    }
  }
  let motion = load_clamped(scene_motion, closest).xy;
  let motion_known = finite2(motion) && all(abs(motion) < vec2f(2.0));
  let shift = select(vec2f(0.0), motion, motion_known);
  let prev_position = frag.xy - shift * vec2f(0.5, -0.5) * size;
  let off_screen = !motion_known || any(prev_position < vec2f(0.0)) || any(prev_position >= size);
  let samples = load_clamped(history_tex, vec2i(floor(prev_position))).a;
  var history = history_catmull_rom(prev_position, 1.0 / size);

  var moment1 = vec3f(0.0);
  var moment2 = vec3f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let c = rgb_to_ycocg(load_clamped(source_tex, pixel + vec2i(x, y)).rgb);
      moment1 += c;
      moment2 += c * c;
    }
  }
  let mean = moment1 / 9.0;
  let deviation = sqrt(max(moment2 / 9.0 - mean * mean, vec3f(0.0)));
  history = ycocg_to_rgb(clip_towards_aabb_center(rgb_to_ycocg(history), mean - deviation, mean + deviation));

  let still = all(abs(shift) * 0.5 * size < vec2f(0.01));
  let next = select(min(samples + 1.0, MOVING_SAMPLES), samples + 1.0, still);
  let reactive = saturate(load_clamped(scene_reactive, pixel).r);
  var blend = max(max(1.0 / next, MIN_BLEND), reactive);
  if (off_screen || !(samples >= 1.0)) {
    blend = 1.0;
  }
  let resolved = select(mix(history, current, blend), current, blend >= 1.0);
  return vec4f(resolved, 1.0 / blend);
//#endif
}

@fragment
fn fs_apply(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(floor(frag.xy));
  let e = load_clamped(history_tex, pixel).rgb;
  let sharpness = taa_params.sharpness;
  if (!(sharpness > 0.0)) {
    return vec4f(e, 0.0);
  }
  let b = load_clamped(history_tex, pixel + vec2i(0, -1)).rgb;
  let d = load_clamped(history_tex, pixel + vec2i(-1, 0)).rgb;
  let f = load_clamped(history_tex, pixel + vec2i(1, 0)).rgb;
  let h = load_clamped(history_tex, pixel + vec2i(0, 1)).rgb;
  let mn4 = min(min(b, d), min(f, h));
  let mx4 = max(max(b, d), max(f, h));
  let hit_min = min(mn4, e) / max(4.0 * mx4, vec3f(0.00001));
  let hit_max = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3f(-0.00001));
  let lobe = max(-RCAS_LIMIT, min(max3(max(-hit_min, hit_max)), 0.0)) * min(sharpness, 1.0);
  let color = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
  return vec4f(saturate(color), 0.0);
}

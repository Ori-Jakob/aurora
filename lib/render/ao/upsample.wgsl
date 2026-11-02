// The edge weights follow XeGTAO's denoiser (GameTechDev/XeGTAO), which carries this notice:
//
// MIT License
//
// Copyright (C) 2016-2021, Intel Corporation
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

//#import aurora/ao
//#import aurora/gbuffer

@group(1) @binding(0) var ao_raw: texture_2d<f32>;
@group(1) @binding(1) var ao_hiz: texture_2d<f32>;
@group(1) @binding(2) var scene_depth: texture_2d<f32>;
@group(1) @binding(3) var scene_normal: texture_2d<f32>;

struct UpsampleTap {
  a: f32,
  receiver: bool,
  z: f32,
  edges: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn upsample_unpack_edges(packed_edges: f32) -> vec4f {
  let bits = u32(packed_edges * 255.5);
  return saturate(vec4f(f32((bits >> 6u) & 3u), f32((bits >> 4u) & 3u), f32((bits >> 2u) & 3u), f32(bits & 3u)) / 3.0);
}

fn upsample_tap(p: vec2i) -> UpsampleTap {
  let t = textureLoad(ao_raw, p, 0);
  return UpsampleTap(ao_raw_visibility(t), t.b > 0.5, textureLoad(ao_hiz, p, 0).r, upsample_unpack_edges(t.g));
}

fn upsample_edge(e: vec4f, across: vec4f, down: vec4f, home: u32, i: u32) -> f32 {
  if (i == home) {
    return 1.0;
  }
  let hx = home & 1u;
  let hy = home >> 1u;
  let ix = i & 1u;
  let iy = i >> 1u;
  let horizontal = select(e.x, e.y, ix > hx);
  let vertical = select(e.z, e.w, iy > hy);
  if (iy == hy) {
    return horizontal;
  }
  if (ix == hx) {
    return vertical;
  }
  return 0.5 * (horizontal * select(across.z, across.w, iy > hy) + vertical * select(down.x, down.y, ix > hx));
}

fn upsample_visibility(f: vec2u, dz: f32) -> f32 {
  if ((ao_params.flags & AO_FLAG_IDENTITY_FORCED) != 0u || dz <= 0.0) {
    return 1.0;
  }
  let full = textureDimensions(scene_depth);
  let grid = textureDimensions(ao_raw);
  let d = ao_divisor(full, grid);
  let z = camera_view_depth(ao_params.camera, dz);
  let g = (vec2f(f) - f32(d / 2u)) / f32(d);
  let base = vec2i(floor(g));
  let fraction = g - vec2f(base);
  let last = vec2i(grid) - 1;
  var taps: array<UpsampleTap, 4>;
  var home = 4u;
  var home_gap = 0.0;
  for (var i = 0u; i < 4u; i++) {
    taps[i] = upsample_tap(clamp(base + vec2i(i32(i & 1u), i32(i >> 1u)), vec2i(0), last));
    let gap = abs(taps[i].z - z);
    if (taps[i].receiver && (home == 4u || gap < home_gap)) {
      home = i;
      home_gap = gap;
    }
  }
  let reject = ao_params.depth_reject * z;
  if (home == 4u || home_gap > reject) {
    // No AO sample shares this surface
    return 1.0;
  }
  var sum = 0.0;
  var weights = 0.0;
  var all_open = true;
  for (var i = 0u; i < 4u; i++) {
    let wx = select(1.0 - fraction.x, fraction.x, (i & 1u) != 0u);
    let wy = select(1.0 - fraction.y, fraction.y, (i >> 1u) != 0u);
    let depth_weight = saturate(1.0 - abs(taps[i].z - z) / reject);
    let across = taps[(i & 1u) | (home & 2u)].edges;
    let down = taps[(home & 1u) | (i & 2u)].edges;
    let edge = upsample_edge(taps[home].edges, across, down, home, i);
    let w = select(0.0, (wx * wy + 1e-3) * edge * depth_weight, taps[i].receiver);
    if (w > 0.0) {
      sum += w * taps[i].a;
      weights += w;
      all_open = all_open && taps[i].a >= 1.0;
    }
  }
  if (all_open || weights <= 0.0) {
    return 1.0;
  }
  let v = ao_apply_strength(sum / weights, ao_params.strength, ao_params.min_visibility);
  if (v < 1.0 && gbuf_receiver_code(textureLoad(scene_normal, f, 0).a) == 0u) {
    return 1.0;
  }
  return v;
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let f = vec2u(pos.xy);
  let dz = textureLoad(scene_depth, f, 0).r;
  let key = pack2x16float(vec2f(dz, 0.0)) & 0xFFFFu;
  // G and B carry this pixel's depth, so forward draws take AO only where they lie on it
  return vec4f(upsample_visibility(f, dz), f32(key & 0xFFu) / 255.0, f32(key >> 8u) / 255.0, 0.0);
}

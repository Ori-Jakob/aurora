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

@group(1) @binding(0) var ao_history: texture_2d<f32>;
@group(1) @binding(1) var ao_hiz: texture_2d<f32>;
@group(1) @binding(2) var scene_depth: texture_2d<f32>;
@group(1) @binding(3) var scene_normal: texture_2d<f32>;
@group(0) @binding(3) var nearest_clamp: sampler;

// One packed edge of a tap, unpacked exactly as ao_unpack_edges does
fn upsample_edge_bits(packed_edges: f32, shift: u32) -> f32 {
  return saturate(f32((u32(packed_edges * 255.5) >> shift) & 3u) / 3.0);
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn upsample_visibility(f: vec2u, dz: f32) -> f32 {
  if ((ao_params.flags & AO_FLAG_IDENTITY_FORCED) != 0u || dz <= 0.0) {
    return 1.0;
  }
  let full = textureDimensions(scene_depth);
  let grid = textureDimensions(ao_history);
  let d = ao_divisor(full, grid);
  let z = camera_view_depth(ao_params.camera, dz);
  let g = (vec2f(f) - f32(d / 2u)) / f32(d);
  let base = vec2i(floor(g));
  let fraction = g - vec2f(base);
  // One gather per channel reads the 2x2 footprint, and clamp-to-edge matches clamping the tap index
  let uv = (vec2f(base) + 1.0) / vec2f(grid);
  // Gathers return (0,1) (1,1) (1,0) (0,0); swizzled so lane i is tap (i & 1, i >> 1)
  let a = textureGather(0, ao_history, nearest_clamp, uv).wzxy;
  let receiver = textureGather(1, ao_history, nearest_clamp, uv).wzxy > vec4f(0.0);
  let packed = textureGather(3, ao_history, nearest_clamp, uv).wzxy;
  let tz = textureGather(0, ao_hiz, nearest_clamp, uv).wzxy;
  let gap = abs(tz - z);
  var home = 4u;
  var home_gap = 0.0;
  for (var i = 0u; i < 4u; i++) {
    if (receiver[i] && (home == 4u || gap[i] < home_gap)) {
      home = i;
      home_gap = gap[i];
    }
  }
  let reject = ao_params.depth_reject * z;
  if (home == 4u || home_gap > reject) {
    // No AO sample shares this surface
    return 1.0;
  }
  // Only the home tap's edges and one edge each of its two neighbours are ever read
  let left = (home & 1u) == 0u;
  let top = (home >> 1u) == 0u;
  let home_edges = ao_unpack_edges(packed[home]);
  let horizontal = select(home_edges.x, home_edges.y, left);
  let vertical = select(home_edges.z, home_edges.w, top);
  let across = upsample_edge_bits(packed[home ^ 1u], select(2u, 0u, top));
  let down = upsample_edge_bits(packed[home ^ 2u], select(6u, 4u, left));
  let diagonal = 0.5 * (horizontal * across + vertical * down);
  let k = vec4u(0u, 1u, 2u, 3u) ^ vec4u(home);
  let edge = select(select(vec4f(diagonal), vec4f(vertical), k == vec4u(2u)),
                    select(vec4f(horizontal), vec4f(1.0), k == vec4u(0u)), k < vec4u(2u));
  let wx = vec4f(1.0 - fraction.x, fraction.x, 1.0 - fraction.x, fraction.x);
  let wy = vec4f(1.0 - fraction.y, 1.0 - fraction.y, fraction.y, fraction.y);
  let depth_weight = saturate(1.0 - gap / reject);
  let w = select(vec4f(0.0), (wx * wy + 1e-3) * edge * depth_weight, receiver);
  var sum = 0.0;
  var weights = 0.0;
  var all_open = true;
  for (var i = 0u; i < 4u; i++) {
    if (w[i] > 0.0) {
      sum += w[i] * a[i];
      weights += w[i];
      all_open = all_open && a[i] >= 1.0;
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

// The spatial filter follows XeGTAO's denoiser (GameTechDev/XeGTAO), which carries this notice:
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

// Anti-lag after NRD ReLAX

//#import aurora/ao

@group(1) @binding(0) var ao_raw: texture_2d<f32>;
@group(1) @binding(1) var ao_hiz: texture_2d<f32>;
@group(1) @binding(2) var scene_motion: texture_2d<f32>;
@group(1) @binding(3) var ao_history: texture_storage_2d<rgba16float, write>;
@group(2) @binding(0) var prev_history: texture_2d<f32>;

const DENOISE_DIAGONAL_WEIGHT: f32 = 0.85 * 0.5;
const DENOISE_LEAK_THRESHOLD: f32 = 2.5;
const DENOISE_LEAK_STRENGTH: f32 = 0.5;
const DENOISE_MIN_HISTORY_WEIGHT: f32 = 1e-3;
const DENOISE_INVALID_CONTRAST: f32 = 1e30;

struct DenoiseSpatial {
  value: f32,
  sigma: f32,
  receivers: f32,
}

fn denoise_raw(p: vec2i, last: vec2i) -> vec4f {
  return textureLoad(ao_raw, clamp(p, vec2i(0), last), 0);
}

fn denoise_spatial(t: array<vec4f, 9>) -> DenoiseSpatial {
  let edges_l = ao_unpack_edges(t[3].g);
  let edges_t = ao_unpack_edges(t[1].g);
  let edges_r = ao_unpack_edges(t[5].g);
  let edges_b = ao_unpack_edges(t[7].g);
  var edges_c = ao_unpack_edges(t[4].g) * vec4f(edges_l.y, edges_r.x, edges_t.w, edges_b.z);
  let leak_range = 4.0 - DENOISE_LEAK_THRESHOLD;
  let edginess = saturate(leak_range - dot(edges_c, vec4f(1.0))) / leak_range * DENOISE_LEAK_STRENGTH;
  edges_c = saturate(edges_c + edginess);
  let weight_tl = DENOISE_DIAGONAL_WEIGHT * (edges_c.x * edges_l.z + edges_c.z * edges_t.x);
  let weight_tr = DENOISE_DIAGONAL_WEIGHT * (edges_c.z * edges_t.y + edges_c.y * edges_r.z);
  let weight_bl = DENOISE_DIAGONAL_WEIGHT * (edges_c.w * edges_b.x + edges_c.x * edges_l.w);
  let weight_br = DENOISE_DIAGONAL_WEIGHT * (edges_c.y * edges_r.w + edges_c.w * edges_b.y);
  let weights = array<f32, 9>(weight_tl, edges_c.z, weight_tr, edges_c.x, ao_params.denoise_beta, edges_c.y, weight_bl,
                              edges_c.w, weight_br);
  var sum = 0.0;
  var weight_sum = 0.0;
  var all_open = true;
  var moment1 = 0.0;
  var moment2 = 0.0;
  var receivers = 0.0;
  for (var i = 0u; i < 9u; i++) {
    if (t[i].b <= 0.5) {
      continue;
    }
    let a = ao_raw_visibility(t[i]);
    moment1 += a;
    moment2 += a * a;
    receivers += 1.0;
    if (weights[i] > 0.0) {
      sum += weights[i] * a;
      weight_sum += weights[i];
      all_open = all_open && a >= 1.0;
    }
  }
  let mean = moment1 / receivers;
  let sigma = sqrt(max(moment2 / receivers - mean * mean, 0.0));
  return DenoiseSpatial(select(sum / weight_sum, 1.0, all_open), sigma, receivers);
}

fn denoise_motion(q: vec2u) -> vec2f {
  return textureLoad(scene_motion, q, 0).xy;
}

fn denoise_motion_valid(m: vec2f) -> bool {
  return all(abs(m) < vec2f(2.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(ao_history);
  if (any(id.xy >= size)) {
    return;
  }
  let last = vec2i(size) - 1;
  let p = vec2i(id.xy);
  let z = textureLoad(ao_hiz, p, 0).r;
  let log_z = log2(max(z, 1e-30));
  var t: array<vec4f, 9>;
  for (var i = 0; i < 9; i++) {
    t[i] = denoise_raw(p + vec2i(i % 3 - 1, i / 3 - 1), last);
  }
  let edges = t[4].g;
  if (t[4].b <= 0.5) {
    textureStore(ao_history, id.xy, vec4f(1.0, 0.0, log_z, edges));
    return;
  }
  let spatial = denoise_spatial(t);
  let d = spatial.value;
//#if AO_RESET
  textureStore(ao_history, id.xy, vec4f(d, 1.0, log_z, edges));
//#else
  let full = textureDimensions(scene_motion);
  let divisor = ao_divisor(full, size);
  let q = ao_rep_texel(id.xy, divisor, full);
  let m = denoise_motion(q);
  let uv = (vec2f(q) + 0.5) / vec2f(full);
  let prev_uv = uv - m * vec2f(0.5, -0.5);
  var valid = denoise_motion_valid(m) && all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0));

  let position = (prev_uv * vec2f(full) - f32(divisor / 2u) - 0.5) / f32(divisor);
  let base = vec2i(floor(position));
  let fraction = position - vec2f(base);
  var taps: array<vec4f, 4>;
  for (var i = 0; i < 4; i++) {
    taps[i] = textureLoad(prev_history, clamp(base + vec2i(i & 1, i >> 1u), vec2i(0), last), 0);
  }
  let w = ao_history_tap_weights(fraction, vec4f(taps[0].g, taps[1].g, taps[2].g, taps[3].g),
                                 vec4f(taps[0].b, taps[1].b, taps[2].b, taps[3].b), z, ao_params.depth_reject);
  let history_weight = dot(w, vec4f(1.0));
  valid = valid && history_weight >= DENOISE_MIN_HISTORY_WEIGHT;
  let divide = max(history_weight, DENOISE_MIN_HISTORY_WEIGHT);
  let history_ao = dot(w, vec4f(taps[0].r, taps[1].r, taps[2].r, taps[3].r)) / divide;
  let history_n = dot(w, vec4f(taps[0].g, taps[1].g, taps[2].g, taps[3].g)) / divide;

  var contrast = 0.0;
  for (var i = 0; i < 9; i++) {
    let neighbour = clamp(p + vec2i(i % 3 - 1, i / 3 - 1), vec2i(0), last);
    let mi = denoise_motion(ao_rep_texel(vec2u(neighbour), divisor, full));
    let px = length((mi - m) * 0.5 * vec2f(full));
    contrast = max(contrast, select(DENOISE_INVALID_CONTRAST, px, denoise_motion_valid(mi)));
  }

  let box = AO_CLIP_GAMMA * spatial.sigma / sqrt(max(spatial.receivers, 1.0)) + AO_CLIP_FLOOR;
  // Static ground behind a moving occluder passes the depth test, so lag compares history with the current estimate
  let lag = saturate((abs(history_ao - d) - box) / box);
  let cap = min(ao_temporal_cap(lag, contrast), ao_params.max_samples);
  let n = select(1.0, min(history_n + 1.0, max(floor(cap), 1.0)), valid);
  let clipped = clamp(history_ao, d - box, d + box);
  let ao = select(d, clipped + (d - clipped) / n, valid);
  textureStore(ao_history, id.xy, vec4f(ao, n, log_z, edges));
//#endif
}

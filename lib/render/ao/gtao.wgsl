// The main pass follows XeGTAO's MainPass (GameTechDev/XeGTAO), which carries this notice:
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

// Sector bitmask after Therrien, Levesque and Gilet 2023 (arXiv 2301.11376)

//#import aurora/ao
//#import aurora/gbuffer

@group(1) @binding(0) var ao_hiz: texture_2d<f32>;
@group(1) @binding(1) var scene_normal: texture_2d<f32>;
@group(1) @binding(2) var ao_raw: texture_storage_2d<rgba8unorm, write>;
//#if AO_SSIL
//#import aurora/ssil
@group(1) @binding(3) var ssil_radiance: texture_2d<f32>;
@group(1) @binding(4) var ssil_normal: texture_2d<f32>;
@group(1) @binding(5) var ssil_raw: texture_storage_2d<rgba16float, write>;
//#endif

const GTAO_PI: f32 = 3.14159265358979;
const GTAO_HALF_PI: f32 = 1.57079632679490;
const GTAO_PIXEL_TOO_CLOSE: f32 = 1.3;
const GTAO_CENTRE_PULL: f32 = 0.99999;
const GTAO_GOLDEN: f32 = 0.618033988749895;

fn gtao_depth(p: vec2i, last: vec2i) -> f32 {
  return textureLoad(ao_hiz, clamp(p, vec2i(0), last), 0).r;
}

// Mip sizes are max(1, size >> mip), so no size query per sample
fn gtao_mip_texel(p: vec2i, mip: u32, last: vec2i) -> vec2i {
  return min(clamp(p, vec2i(0), last) >> vec2u(mip), max((last + 1) >> vec2u(mip), vec2i(1)) - 1);
}

fn gtao_calculate_edges(z: f32, l: f32, r: f32, t: f32, b: f32) -> vec4f {
  let edges = vec4f(l, r, t, b) - z;
  let slope_lr = (edges.y - edges.x) * 0.5;
  let slope_tb = (edges.w - edges.z) * 0.5;
  let adjusted = edges + vec4f(slope_lr, -slope_lr, slope_tb, -slope_tb);
  return saturate(1.25 - min(abs(edges), abs(adjusted)) / (z * 0.011));
}

fn gtao_pack_edges(edges: vec4f) -> f32 {
  let q = round(saturate(edges) * 2.9);
  return dot(q, vec4f(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

fn gtao_position(px: vec2f, z: f32, full: vec2f) -> vec3f {
  return camera_view_position_fwd(ao_params.camera, px / full, z, full);
}

fn gtao_depth_normal(edges: vec4f, c: vec3f, l: vec3f, r: vec3f, t: vec3f, b: vec3f) -> vec3f {
  let accepted = saturate(vec4f(edges.x * edges.z, edges.z * edges.y, edges.y * edges.w, edges.w * edges.x) + 0.01);
  let dl = normalize(l - c);
  let dr = normalize(r - c);
  let dt = normalize(t - c);
  let db = normalize(b - c);
  return accepted.x * cross(dl, dt) + accepted.y * cross(dt, dr) + accepted.z * cross(dr, db) +
         accepted.w * cross(db, dl);
}

// Equals projecting the normal onto the slice plane through u, without building that plane's basis
fn gtao_normal_angle(u: vec2f, dir_scale: vec2f, view_vec: vec3f, normal: vec3f, n_dot_v: f32) -> f32 {
  let dir = u * dir_scale;
  let d_dot_v = dot(dir, view_vec.xy);
  let along = dot(normal.xy, dir) - d_dot_v * n_dot_v;
  let ortho_sq = max(dot(dir, dir) - d_dot_v * d_dot_v, 0.0);
  let len_sq = max(n_dot_v * n_dot_v * ortho_sq + along * along, 1e-30);
  let cos_n = saturate(n_dot_v * sqrt(ortho_sq) * inverseSqrt(len_sq));
  return sign(along) * gtao_acos(cos_n);
}

// Abramowitz and Stegun 4.4.45, within 7e-5 rad: a few thousandths of a sector
fn gtao_acos(x: f32) -> f32 {
  let a = abs(x);
  let r = sqrt(1.0 - a) * (1.5707288 + a * (-0.2121144 + a * (0.0742610 - 0.0187293 * a)));
  return select(GTAO_PI - r, r, x >= 0.0);
}

fn gtao_safe_normalize(v: vec3f, fallback: vec3f) -> vec3f {
  let len = length(v);
  return select(fallback, v / len, len > 1e-6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(ao_raw);
  if (any(id.xy >= size)) {
    return;
  }
  let full = textureDimensions(scene_normal);
  let fullf = vec2f(full);
  let d = ao_divisor(full, size);
  let last = vec2i(size) - 1;
  let p = vec2i(id.xy);
  let q = ao_rep_texel(id.xy, d, full);
  let z = gtao_depth(p, last);
  let around = vec4f(gtao_depth(p + vec2i(-1, 0), last), gtao_depth(p + vec2i(1, 0), last),
                     gtao_depth(p + vec2i(0, -1), last), gtao_depth(p + vec2i(0, 1), last));
  let edges = gtao_calculate_edges(z, around.x, around.y, around.z, around.w);
  let packed_edges = gtao_pack_edges(edges);
  let open = 1.0 / AO_RAW_SCALE;
  let normal_texel = textureLoad(scene_normal, q, 0);
  if (ao_is_sky(z) || gbuf_receiver_code(normal_texel.a) == 0u) {
    textureStore(ao_raw, id.xy, vec4f(open, packed_edges, 0.0, 0.0));
//#if AO_SSIL
    textureStore(ssil_raw, id.xy, vec4f(0.0));
//#endif
    return;
  }
  if (ao_params.slice_count == 0u) {
    textureStore(ao_raw, id.xy, vec4f(open, packed_edges, 1.0, 0.0));
//#if AO_SSIL
    textureStore(ssil_raw, id.xy, vec4f(0.0));
//#endif
    return;
  }

  let centre_px = vec2f(q) + 0.5;
  let step_px = f32(d);
  let near_around = select(around, vec4f(z), around >= vec4f(0.5 * AO_SKY_DEPTH));
  let centre = gtao_position(centre_px, z, fullf);
  let surface = gtao_depth_normal(edges, centre, gtao_position(centre_px + vec2f(-step_px, 0.0), near_around.x, fullf),
                                  gtao_position(centre_px + vec2f(step_px, 0.0), near_around.y, fullf),
                                  gtao_position(centre_px + vec2f(0.0, -step_px), near_around.z, fullf),
                                  gtao_position(centre_px + vec2f(0.0, step_px), near_around.w, fullf));
  let pulled_z = z * GTAO_CENTRE_PULL;
  let pix_centre = gtao_position(centre_px, pulled_z, fullf);
  let view_vec = normalize(-pix_centre);
  let depth_normal = gtao_safe_normalize(surface, -view_vec);
  var normal = depth_normal;
  if ((ao_params.flags & AO_FLAG_NORMAL_FROM_DEPTH) == 0u) {
    normal = camera_gx_normal_fwd(gtao_safe_normalize(normal_texel.rgb * 2.0 - 1.0, vec3f(0.0, 0.0, 1.0)));
    if (dot(normal, depth_normal) < 0.0) {
      normal = -normal;
    }
  }
  normal = gtao_safe_normalize(normal + max(0.0, -dot(normal, view_vec)) * view_vec, depth_normal);

  let effect_radius = ao_params.effect_radius;
  let pixel_size = 2.0 * step_px * pulled_z / (ao_params.camera.depth.z * fullf.x);
  let screen_radius = effect_radius / pixel_size;
  if (screen_radius < GTAO_PIXEL_TOO_CLOSE) {
    textureStore(ao_raw, id.xy, vec4f(open, packed_edges, 1.0, 0.0));
//#if AO_SSIL
    textureStore(ssil_raw, id.xy, vec4f(0.0));
//#endif
    return;
  }

  let noise = ao_noise(id.xy, ao_params.noise_index);
  let cam = ao_params.camera.depth;
  let dir_scale = vec2f(1.0 / (fullf.x * cam.z), -1.0 / (fullf.y * cam.w));
  // gtao_position's terms folded once, since only the cell and the depth change per sample
  let inv_full = 1.0 / fullf;
  let ndc_scale = vec2f(2.0, -2.0) * step_px * inv_full;
  let ndc_bias = vec2f(2.0, -2.0) * (f32(d / 2u) + 0.5) * inv_full + vec2f(-1.0, 1.0) + ao_params.camera.offset.xy +
                 vec2f(-2.0, 2.0) * ao_params.camera.offset.zw * inv_full;
  let inv_proj = 1.0 / cam.zw;
  let n_dot_v = dot(normal, view_vec);
  let falloff_range = ao_params.falloff_range * effect_radius;
  let falloff_from = effect_radius * (1.0 - ao_params.falloff_range);
  let falloff_mul = -1.0 / falloff_range;
  let falloff_add = falloff_from / falloff_range + 1.0;
  let min_s = GTAO_PIXEL_TOO_CLOSE / screen_radius;
  let ceil_mode = (ao_params.flags & AO_FLAG_SECTOR_CEIL) != 0u;
  let max_mip = min(textureNumLevels(ao_hiz), AO_MIP_LEVELS) - 1u;
  let slices = ao_params.slice_count;
  let steps = ao_params.steps_per_slice;
  var visibility = 0.0;
//#if AO_SSIL
  let ssil_live = all(textureDimensions(ssil_radiance) == size) && all(textureDimensions(ssil_normal) == size);
  var ssil_sum = vec4f(0.0);
//#endif
  for (var slice_index = 0u; slice_index < slices; slice_index++) {
    let phi = (f32(slice_index) + noise.x) / f32(slices) * GTAO_PI;
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);
    let omega = vec2f(cos_phi, -sin_phi) * screen_radius;
    var mask = 0u;
    for (var step_index = 0u; step_index < steps; step_index++) {
      let step_noise = fract(noise.y + f32(slice_index + step_index * steps) * GTAO_GOLDEN);
      let s = pow(max((f32(step_index) + step_noise) / f32(steps), 1e-6), ao_params.distribution_power) + min_s;
      let offset = s * omega;
      let mip_level = clamp(log2(length(offset)) - ao_params.mip_offset, 0.0, f32(AO_MIP_LEVELS));
      let mip = min(u32(round(mip_level)), max_mip);
      let snapped = round(offset);
      for (var side = 0u; side < 2u; side++) {
        let texel = gtao_mip_texel(p + vec2i(select(-snapped, snapped, side == 0u)), mip, last);
        let sz = textureLoad(ao_hiz, texel, i32(mip)).r;
        if (ao_is_sky(sz)) {
          continue;
        }
        let cell = (vec2f(texel) + 0.5) * f32(1u << mip) - 0.5;
        let towards = cell - vec2f(p);
        let delta = vec3f((cell * ndc_scale + ndc_bias) * (sz * inv_proj), sz) - pix_centre;
        let dist = length(delta);
        let weight = saturate(dist * falloff_mul + falloff_add);
        if (weight <= 0.0 || dist <= 0.0 || dot(towards, towards) < 0.25) {
          continue;
        }
        let n = gtao_normal_angle(select(-towards, towards, side == 0u), dir_scale, view_vec, normal, n_dot_v);
        let delta_v = dot(delta, view_vec);
        let front = gtao_acos(clamp(delta_v / dist, -1.0, 1.0));
        let thickness = ao_params.thickness;
        let back_len_sq = max(dist * dist - 2.0 * thickness * delta_v + thickness * thickness, 1e-30);
        let back = gtao_acos(clamp((delta_v - thickness) * inverseSqrt(back_len_sq), -1.0, 1.0));
        var lo: f32;
        var hi: f32;
        if (side == 0u) {
          lo = 1.0 - weight * (1.0 - (front - n + GTAO_HALF_PI) / GTAO_PI);
          hi = 1.0 - weight * (1.0 - (back - n + GTAO_HALF_PI) / GTAO_PI);
        } else {
          lo = weight * ((GTAO_HALF_PI - back - n) / GTAO_PI);
          hi = weight * ((GTAO_HALF_PI - front - n) / GTAO_PI);
        }
        let bits = ao_sector_bits(lo, hi, ceil_mode);
//#if AO_SSIL
        let new_bits = bits & ~mask;
        if (new_bits != 0u && ssil_live) {
          let r = textureLoad(ssil_radiance, texel, i32(mip));
          if (r.a > 0.0) {
            let emitter = ssil_decode_normal(textureLoad(ssil_normal, texel, i32(mip)));
            let facing = ssil_emitter_facing(emitter, delta / dist);
            ssil_sum += f32(countOneBits(new_bits)) / 32.0 * facing * r;
          }
        }
//#endif
        mask |= bits;
      }
    }
    visibility += 1.0 - f32(countOneBits(mask)) / f32(AO_SECTOR_COUNT);
  }
//#if AO_SSIL
  // Sectors are uniform in angle like AO's, so the occluded share is exactly what visibility leaves out
  let occ = f32(slices) - visibility;
  textureStore(ssil_raw, id.xy, select(vec4f(0.0), ssil_sum / occ, occ > 0.0));
//#endif
  visibility = max(ao_params.min_visibility, pow(max(visibility / f32(slices), 1e-6), AO_FINAL_POWER));
  textureStore(ao_raw, id.xy, vec4f(saturate(visibility / AO_RAW_SCALE), packed_edges, 1.0, 0.0));
}

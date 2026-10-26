// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_accumulate.h, ffx_fsr3upscaler_accumulate_pass.hlsl, ffx_fsr3upscaler_upsample.h,
// ffx_fsr3upscaler_reproject.h and ffx_fsr3upscaler_sample.h), which carry this notice:
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

@group(1) @binding(0) var dilated_reactive: texture_2d<f32>;
@group(1) @binding(1) var dilated_motion: texture_2d<f32>;
@group(1) @binding(2) var previous_upscaled: texture_2d<f32>;
@group(1) @binding(3) var farthest_depth_mip1: texture_2d<f32>;
@group(1) @binding(4) var luma_instability: texture_2d<f32>;
@group(1) @binding(5) var input_color: texture_2d<f32>;
@group(1) @binding(6) var upscaled: texture_storage_2d<rgba16float, write>;
//#if FSR_LOCKS_R8
@group(1) @binding(7) var new_locks: texture_storage_2d<r8unorm, read_write>;
//#else
@group(1) @binding(7) var new_locks: texture_storage_2d<r32float, read_write>;
//#endif

const PI: f32 = 3.141592653589793;

struct AccumParams {
  hr: vec2i,
  hr_uv: vec2f,
  lr_uv_jittered: vec2f,
  lr_uv_hw: vec2f,
  motion: vec2f,
  reprojected_hr_uv: vec2f,
  velocity: f32,
  disocclusion: f32,
  reactive: f32,
  change: f32,
  accumulation: f32,
  instability: f32,
  farthest_m: f32,
  existing: bool,
  new_sample: bool,
}

struct AccumData {
  box_center: vec3f,
  box_vec: vec3f,
  aabb_min: vec3f,
  aabb_max: vec3f,
  box_weight: f32,
  upsampled: vec3f,
  upsampled_weight: f32,
  history: vec3f,
  history_weight: f32,
  lock: f32,
  lock_contribution: f32,
}

fn upscale_size() -> vec2i {
  return vec2i(textureDimensions(new_locks));
}

fn lanczos2_no_clamp(x: f32) -> f32 {
  if (abs(x) < EPSILON) {
    return 1.0;
  }
  return (sin(PI * x) / (PI * x)) * (sin(0.5 * PI * x) / (0.5 * PI * x));
}

fn lanczos2(x: f32) -> f32 {
  return lanczos2_no_clamp(min(abs(x), 2.0));
}

fn lanczos2_approx_sq(x2_in: f32) -> f32 {
  let x2 = min(x2_in, 4.0);
  let a = (2.0 / 5.0) * x2 - 1.0;
  let b = (1.0 / 4.0) * x2 - 1.0;
  return ((25.0 / 16.0) * a * a - (25.0 / 16.0 - 1.0)) * (b * b);
}

fn lanczos2_line(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, t: f32) -> vec4f {
  let w0 = lanczos2(-1.0 - t);
  let w1 = lanczos2(-t);
  let w2 = lanczos2(1.0 - t);
  let w3 = lanczos2(2.0 - t);
  return (w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3) / (w0 + w1 + w2 + w3);
}

fn load_history(base: vec2i, offset: vec2i, size: vec2i) -> vec4f {
  return textureLoad(previous_upscaled, clamp(base + offset, vec2i(1), size - 2), 0);
}

fn history_row(base: vec2i, y: i32, size: vec2i, t: f32) -> vec4f {
  return lanczos2_line(load_history(base, vec2i(-1, y), size), load_history(base, vec2i(0, y), size),
                       load_history(base, vec2i(1, y), size), load_history(base, vec2i(2, y), size), t);
}

fn sample_history(uv: vec2f, size: vec2i) -> vec4f {
  var px = uv * vec2f(size) - 0.5;
  let frac = px - floor(px);
  px = clamp(px, vec2f(0.0), vec2f(size - 1));
  let base = vec2i(floor(px));
  let filtered = lanczos2_line(history_row(base, -1, size, frac.x), history_row(base, 0, size, frac.x),
                               history_row(base, 1, size, frac.x), history_row(base, 2, size, frac.x), frac.y);
  let c11 = load_history(base, vec2i(0, 0), size);
  let c21 = load_history(base, vec2i(1, 0), size);
  let c12 = load_history(base, vec2i(0, 1), size);
  let c22 = load_history(base, vec2i(1, 1), size);
  return clamp(filtered, min(min(c11, c21), min(c12, c22)), max(max(c11, c21), max(c12, c22)));
}

fn init_common(p: vec2i) -> AccumParams {
  let render = render_size();
  var c: AccumParams;
  c.hr = p;
  c.hr_uv = (vec2f(p) + 0.5) / vec2f(upscale_size());
  c.lr_uv_jittered = c.hr_uv + fsr.jitter / vec2f(render);
  c.lr_uv_hw = clamp_uv(c.lr_uv_jittered, render, render);
  c.motion = textureLoad(dilated_motion, clamp_load(vec2i(c.hr_uv * vec2f(render)), vec2i(0), render), 0).xy;
  c.velocity = get_4k_velocity(c.motion);
  c.reprojected_hr_uv = c.hr_uv + c.motion;
  c.existing = is_uv_inside(c.reprojected_hr_uv);
  c.instability = textureSampleLevel(luma_instability, linear_clamp, clamp_uv(c.hr_uv, render, render), 0.0).r;
  let farthest_uv = clamp_uv(c.lr_uv_jittered, render / 2, vec2i(textureDimensions(farthest_depth_mip1)));
  c.farthest_m = textureSampleLevel(farthest_depth_mip1, linear_clamp, farthest_uv, 0.0).r;
  c.new_sample = !c.existing || fsr.frame_index == 0.0;
  let masks = textureSampleLevel(dilated_reactive, linear_clamp, c.lr_uv_hw, 0.0);
  c.reactive = saturate(masks.r);
  c.disocclusion = saturate(masks.g);
  c.change = saturate(masks.b);
  c.accumulation = saturate(masks.a);
  c.accumulation *= f32(round(c.accumulation * 100.0) > 1.0);
  return c;
}

fn reproject_history(c: AccumParams, d: ptr<function, AccumData>) {
  let history = sample_history(c.reprojected_hr_uv, upscale_size());
  (*d).history = rgb_to_ycocg(history.rgb);
  (*d).lock = history.w;
}

fn update_lock_status(c: AccumParams, d: ptr<function, AccumData>) {
  (*d).lock *= f32(!c.new_sample);
  let decrease_factor = max(saturate(c.change), max(c.reactive, c.disocclusion));
  (*d).lock = max(0.0, (*d).lock - decrease_factor * LOCK_MAX);
  (*d).lock_contribution = saturate(saturate((*d).lock - LOCK_THRESHOLD) * (LOCK_MAX - LOCK_THRESHOLD));
  let new_lock = textureLoad(new_locks, c.hr).r * (1.0 - max(c.change * 0.0, c.reactive));
  (*d).lock = max(0.0, min((*d).lock + new_lock, LOCK_MAX));
  let lifetime_decrease = (0.1 / fsr.jitter_sequence_length) * (1.0 - decrease_factor);
  (*d).lock = max(0.0, (*d).lock - lifetime_decrease);
  (*d).lock *= f32(is_uv_inside(c.hr_uv - c.motion));
}

fn compute_base_accumulation_weight(c: AccumParams, d: ptr<function, AccumData>) {
  let base = c.accumulation;
  (*d).history_weight = min(base, mix(base, 0.15, saturate(max(0.0, (c.velocity * fsr.velocity_factor) / 0.5))));
}

fn load_prepared_color(p: vec2i) -> vec3f {
  return rgb_to_ycocg(max(vec3f(0.0), textureLoad(input_color, p, 0).rgb));
}

fn add_box_sample(d: ptr<function, AccumData>, first: bool, color: vec3f, weight: f32) {
  let weighted = color * weight;
  if (first) {
    (*d).aabb_min = color;
    (*d).aabb_max = color;
    (*d).box_center = weighted;
    (*d).box_vec = color * weighted;
    (*d).box_weight = weight;
  } else {
    (*d).aabb_min = min((*d).aabb_min, color);
    (*d).aabb_max = max((*d).aabb_max, color);
    (*d).box_center += weighted;
    (*d).box_vec += color * weighted;
    (*d).box_weight += weight;
  }
}

fn compute_variance_box(d: ptr<function, AccumData>) {
  (*d).box_weight = select(1.0, (*d).box_weight, abs((*d).box_weight) > FP32_MIN);
  (*d).box_center /= (*d).box_weight;
  (*d).box_vec /= (*d).box_weight;
  (*d).box_vec = sqrt(abs((*d).box_vec - (*d).box_center * (*d).box_center));
}

fn compute_upsampled_color_and_weight(c: AccumParams, d: ptr<function, AccumData>) {
  let render = render_size();
  let downscale = vec2f(render) / vec2f(upscale_size());
  let src_output = (vec2f(c.hr) + 0.5) * downscale;
  let src_input = vec2i(floor(src_output));
  let src_unjittered = vec2f(src_input) + 0.5 - fsr.jitter;
  let base_offset = src_unjittered - src_output;
  let flip_col = src_unjittered.x > src_output.x;
  let flip_row = src_unjittered.y > src_output.y;
  let offset_tl = vec2i(select(-1, -2, flip_col), select(-1, -2, flip_row));
  let initial = c.accumulation == 0.0;
  var samples: array<vec3f, 9>;
  for (var i = 0; i < 9; i++) {
    let col_row = vec2i(select(i % 3, 3 - i % 3, flip_col), select(i / 3, 3 - i / 3, flip_row));
    samples[i] = load_prepared_color(clamp_load(src_input + offset_tl + col_row, vec2i(0), render));
  }
  let kernel_bias_max = min(1.99, 1.0 + (1.0 / downscale.x - 1.0));
  let kernel_bias_min = max(1.0, (1.0 + kernel_bias_max) * 0.3);
  let kernel_bias_weight =
      min(1.0 - c.disocclusion * 0.5, min(1.0 - c.change, saturate((*d).history_weight * 5.0)));
  let kernel_bias = mix(kernel_bias_min, kernel_bias_max, kernel_bias_weight);
  for (var i = 0; i < 9; i++) {
    let col_row = vec2i(select(i % 3, 3 - i % 3, flip_col), select(i / 3, 3 - i / 3, flip_row));
    let src_offset = base_offset + vec2f(offset_tl + col_row);
    let on_screen = f32(is_on_screen(src_input + offset_tl + col_row, render));
    if (!initial) {
      let biased = src_offset * kernel_bias;
      let weight = on_screen * lanczos2_approx_sq(dot(biased, biased));
      (*d).upsampled += samples[i] * weight;
      (*d).upsampled_weight += weight;
    }
    let box_weight = exp(-2.3 * dot(src_offset, src_offset)) * on_screen;
    add_box_sample(d, i == 0, samples[i], box_weight);
  }
  compute_variance_box(d);
  (*d).upsampled_weight *= f32((*d).upsampled_weight > EPSILON);
  if ((*d).upsampled_weight > EPSILON) {
    (*d).upsampled /= (*d).upsampled_weight;
    (*d).upsampled_weight *= AVERAGE_LANCZOS_WEIGHT_PER_FRAME;
    (*d).upsampled = clamp((*d).upsampled, (*d).aabb_min, (*d).aabb_max);
  }
  if (initial) {
    (*d).upsampled = (*d).box_center;
    (*d).upsampled_weight = 1.0;
    (*d).history_weight = 0.0;
  }
}

fn rectify_history(c: AccumParams, d: ptr<function, AccumData>) {
  let velocity_factor = saturate(c.velocity / 20.0);
  let distance_factor = saturate(0.75 - c.farthest_m / 20.0);
  let accumulation_factor = 1.0 - c.accumulation;
  let reactive_factor = sqrt(c.reactive);
  let t = max(velocity_factor, max(distance_factor, max(accumulation_factor, max(reactive_factor, c.change))));
  let box_scale = mix(3.0, 1.0, t);
  let scaled_box = (*d).box_vec * vec3f(1.7, 1.0, 1.0) * box_scale;
  let transformed = ((*d).history - (*d).box_center) / max(scaled_box, vec3f(1.193e-7));
  if (length(transformed) > 1.0) {
    let clamped = normalize(transformed) * scaled_box + (*d).box_center;
    let contribution = max(c.instability, (*d).lock_contribution) * c.accumulation * (1.0 - c.disocclusion);
    (*d).history = mix(clamped, (*d).history, saturate(contribution));
  }
}

fn accumulate(d: ptr<function, AccumData>) {
  (*d).history_weight *= f32((*d).history_weight > FP16_MIN);
  (*d).history_weight = max(EPSILON, (*d).history_weight + (*d).upsampled_weight);
  let alpha = saturate((*d).upsampled_weight / (*d).history_weight);
  (*d).history = ycocg_to_rgb(mix((*d).history, (*d).upsampled, alpha));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  if (!is_on_screen(p, upscale_size())) {
    return;
  }
  let c = init_common(p);
  var d = AccumData(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 0.0, vec3f(0.0), 0.0, vec3f(0.0), 1.0, 0.0, 0.0);
  if (c.existing && !c.new_sample) {
    reproject_history(c, &d);
  }
  update_lock_status(c, &d);
  compute_base_accumulation_weight(c, &d);
  compute_upsampled_color_and_weight(c, &d);
  rectify_history(c, &d);
  accumulate(&d);
  textureStore(upscaled, p, vec4f(max(d.history, vec3f(0.0)), d.lock));
  textureStore(new_locks, p, vec4f(0.0));
}

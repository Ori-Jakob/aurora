// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_common.h, ffx_fsr3upscaler_callbacks_hlsl.h and spd/ffx_spd.h), which carry this notice:
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

//#import aurora/frame

struct FsrParams {
  jitter: vec2f,
  prev_jitter: vec2f,
  motion_scale: vec2f,
  device_to_view: vec2f,
  jitter_sequence_length: f32,
  delta_time: f32,
  view_to_meters: f32,
  frame_index: f32,
  velocity_factor: f32,
  reactiveness_scale: f32,
  shading_change_scale: f32,
  accumulation_per_frame: f32,
  min_disocclusion_accumulation: f32,
  reactive_max: f32,
  spd_mips: u32,
  pad0: u32,
}

@group(0) @binding(1) var<uniform> fsr: FsrParams;
@group(0) @binding(2) var linear_clamp: sampler;

const FP16_MIN: f32 = 6.10e-05;
const FP16_MAX: f32 = 65504.0;
const EPSILON: f32 = FP16_MIN;
const FP32_MAX: f32 = 3.402823466e+38;
const FP32_MIN: f32 = 1.175494351e-38;
const RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD: f32 = EPSILON * 10.0;
const AVERAGE_LANCZOS_WEIGHT_PER_FRAME: f32 = 0.74 / 16.0;
const LOCK_THRESHOLD: f32 = 1.0;
const LOCK_MAX: f32 = 2.0;
const INVALID_MOTION: f32 = 2.0;
const INVALID_MOTION_REACTIVE: f32 = 0.9;
const RESET_AUTO_EXPOSURE_SMOOTHING: f32 = 1e4;

fn render_size() -> vec2i {
  return vec2i(aurora_frame.efb_size);
}

fn reconstructed_depth_mv_px_threshold(nearest_depth_m: f32) -> f32 {
  return mix(0.25, 0.75, saturate(nearest_depth_m / 100.0));
}

fn get_4k_velocity(motion: vec2f) -> f32 {
  return length(motion * vec2f(3840.0, 2160.0));
}

fn min_divided_by_max_or(v0: f32, v1: f32, on_zero: f32) -> f32 {
  let m = max(v0, v1);
  return select(on_zero, min(v0, v1) / m, m != 0.0);
}

fn min_divided_by_max(v0: f32, v1: f32) -> f32 {
  return min_divided_by_max_or(v0, v1, 0.0);
}

fn ycocg_to_rgb(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

fn rgb_to_ycocg(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn rgb_to_luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn is_uv_inside(uv: vec2f) -> bool {
  return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn clamp_load(p: vec2i, offset: vec2i, size: vec2i) -> vec2i {
  return clamp(p + offset, vec2i(0), size - 1);
}

fn clamp_uv(uv: vec2f, texture_size: vec2i, resource_size: vec2i) -> vec2f {
  let scaled = uv * vec2f(texture_size);
  let clamped = max(vec2f(0.5), min(scaled, vec2f(texture_size) - 0.5));
  return clamped / vec2f(resource_size);
}

fn is_on_screen(p: vec2i, size: vec2i) -> bool {
  return all(vec2u(p) < vec2u(size));
}

fn compute_hr_pos_from_lr_pos(p: vec2i, upscale: vec2i) -> vec2i {
  let jittered = vec2f(p) + 0.5 - fsr.jitter;
  return vec2i(floor(jittered / vec2f(render_size()) * vec2f(upscale)));
}

fn get_view_space_depth(device_depth: f32) -> f32 {
  return fsr.device_to_view.y / (device_depth - fsr.device_to_view.x);
}

fn get_view_space_depth_in_meters(device_depth: f32) -> f32 {
  return get_view_space_depth(device_depth) * fsr.view_to_meters;
}

fn bilinear_offset(index: i32) -> vec2i {
  return vec2i(index & 1, index >> 1);
}

struct BilinearSamplingData {
  base: vec2i,
  weights: vec4f,
}

fn get_bilinear_sampling_data(uv: vec2f, size: vec2i) -> BilinearSamplingData {
  let sample_px = uv * vec2f(size) - 0.5;
  let frac = sample_px - floor(sample_px);
  return BilinearSamplingData(vec2i(floor(sample_px)), vec4f((1.0 - frac.x) * (1.0 - frac.y), frac.x * (1.0 - frac.y),
                                                              (1.0 - frac.x) * frac.y, frac.x * frac.y));
}

fn compute_auto_exposure_from_lavg(log_luma: f32) -> f32 {
  let lavg = exp(log_luma);
  let s = 100.0;
  let k = 12.5;
  let exposure_iso100 = log2((lavg * s) / k);
  let q = 0.65;
  let lmax = (78.0 / (q * s)) * pow(2.0, exposure_iso100);
  return 1.0 / lmax;
}

fn next_frame_info(previous: vec4f, reduced: vec4f) -> vec4f {
  let prev = select(previous, vec4f(-1.0, 1.0, 0.0, 0.0), fsr.frame_index == 0.0);
  var log_luma = reduced.x;
  if (prev.y < RESET_AUTO_EXPOSURE_SMOOTHING) {
    log_luma = max(0.0, prev.y + (log_luma - prev.y) * (1.0 - exp(-fsr.delta_time)));
  }
  return vec4f(compute_auto_exposure_from_lavg(log_luma), log_luma, reduced.y, prev.w);
}

var<workgroup> spd_intermediate: array<array<vec4f, 16>, 16>;

fn spd_thread(index: u32) -> vec2u {
  let a = index % 64u;
  let x = (extractBits(a, 2u, 3u) & ~1u) | (a & 1u);
  let y = (extractBits(a, 3u, 3u) & ~3u) | (extractBits(a, 1u, 2u) & 3u);
  let sub = vec2u(x, y);
  return sub + vec2u(8u * ((index >> 6u) % 2u), 8u * (index >> 7u));
}

fn spd_load_intermediate(p: vec2u) -> vec4f {
  return spd_intermediate[p.x][p.y];
}

fn spd_store_intermediate(p: vec2u, v: vec4f) {
  spd_intermediate[p.x][p.y] = v;
}

fn spd_reduce4(v0: vec4f, v1: vec4f, v2: vec4f, v3: vec4f) -> vec4f {
  return (v0 + v1 + v2 + v3) * 0.25;
}

fn spd_reduce_intermediate(i0: vec2u, i1: vec2u, i2: vec2u, i3: vec2u) -> vec4f {
  return spd_reduce4(spd_load_intermediate(i0), spd_load_intermediate(i1), spd_load_intermediate(i2),
                     spd_load_intermediate(i3));
}

fn spd_mip1(t: vec2u, index: u32, v: vec4f) -> vec4f {
  spd_store_intermediate(t, v);
  workgroupBarrier();
  var reduced = vec4f(0.0);
  if (index < 64u) {
    reduced = spd_reduce_intermediate(t * 2u, t * 2u + vec2u(1u, 0u), t * 2u + vec2u(0u, 1u), t * 2u + vec2u(1u));
  }
  workgroupBarrier();
  return reduced;
}

fn spd_keep_mip1(t: vec2u, index: u32, v: array<vec4f, 4>) {
  if (index < 64u) {
    spd_store_intermediate(t, v[0]);
    spd_store_intermediate(t + vec2u(8u, 0u), v[1]);
    spd_store_intermediate(t + vec2u(0u, 8u), v[2]);
    spd_store_intermediate(t + vec2u(8u), v[3]);
  }
}

fn spd_mip2(t: vec2u, index: u32) -> vec4f {
  var v = vec4f(0.0);
  if (index < 64u) {
    v = spd_reduce_intermediate(t * 2u, t * 2u + vec2u(1u, 0u), t * 2u + vec2u(0u, 1u), t * 2u + vec2u(1u));
    spd_store_intermediate(vec2u(t.x * 2u + t.y % 2u, t.y * 2u), v);
  }
  return v;
}

fn spd_mip3(t: vec2u, index: u32) -> vec4f {
  var v = vec4f(0.0);
  if (index < 16u) {
    v = spd_reduce_intermediate(vec2u(t.x * 4u, t.y * 4u), vec2u(t.x * 4u + 2u, t.y * 4u),
                                vec2u(t.x * 4u + 1u, t.y * 4u + 2u), vec2u(t.x * 4u + 3u, t.y * 4u + 2u));
    spd_store_intermediate(vec2u(t.x * 4u + t.y, t.y * 4u), v);
  }
  return v;
}

fn spd_mip4(t: vec2u, index: u32) -> vec4f {
  var v = vec4f(0.0);
  if (index < 4u) {
    v = spd_reduce_intermediate(vec2u(t.x * 8u + t.y * 2u, t.y * 8u), vec2u(t.x * 8u + 4u + t.y * 2u, t.y * 8u),
                                vec2u(t.x * 8u + 1u + t.y * 2u, t.y * 8u + 4u),
                                vec2u(t.x * 8u + 5u + t.y * 2u, t.y * 8u + 4u));
    spd_store_intermediate(vec2u(t.x + t.y * 2u, 0u), v);
  }
  return v;
}

fn spd_mip5(index: u32) -> vec4f {
  var v = vec4f(0.0);
  if (index < 1u) {
    v = spd_reduce_intermediate(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(2u, 0u), vec2u(3u, 0u));
  }
  return v;
}

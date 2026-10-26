// Ported from AMD FidelityFX Super Resolution 3.1.5 (GPUOpen-LibrariesAndSDKs/FidelityFX-SDK v2.3.0,
// ffx_fsr3upscaler_prepare_inputs.h and ffx_fsr3upscaler_prepare_inputs_pass.hlsl), which carry this notice:
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

@group(1) @binding(0) var input_motion: texture_2d<f32>;
@group(1) @binding(1) var input_depth: texture_2d<f32>;
@group(1) @binding(2) var input_color: texture_2d<f32>;
@group(1) @binding(3) var dilated_motion: texture_storage_2d<rg16float, write>;
@group(1) @binding(4) var dilated_depth: texture_storage_2d<r32float, write>;
@group(1) @binding(5) var<storage, read_write> reconstructed_depth: array<atomic<u32>>;
@group(1) @binding(6) var farthest_depth: texture_storage_2d<r16float, write>;
@group(1) @binding(7) var current_luma: texture_storage_2d<r16float, write>;

fn load_input_depth(p: vec2i) -> f32 {
  return textureLoad(input_depth, clamp_load(p, vec2i(0), render_size()), 0).r;
}

fn load_input_motion(p: vec2i) -> vec2f {
  let motion = textureLoad(input_motion, clamp_load(p, vec2i(0), render_size()), 0).xy;
  if (any(abs(motion) >= vec2f(2.0))) {
    return vec2f(INVALID_MOTION);
  }
  return motion * fsr.motion_scale;
}

fn reconstruct_prev_depth(p: vec2i, depth: f32, motion: vec2f) {
  let size = render_size();
  let nearest_depth_m = min(get_view_space_depth_in_meters(depth), FP16_MAX);
  let threshold = reconstructed_depth_mv_px_threshold(nearest_depth_m);
  let kept = motion * f32(get_4k_velocity(motion) > threshold);
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let bilinear = get_bilinear_sampling_data(uv + kept, size);
  for (var i = 0; i < 4; i++) {
    if (bilinear.weights[i] > RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD) {
      let store = bilinear.base + bilinear_offset(i);
      if (is_on_screen(store, size)) {
        atomicMax(&reconstructed_depth[u32(store.y) * u32(size.x) + u32(store.x)], bitcast<u32>(depth));
      }
    }
  }
}

struct DepthExtents {
  nearest: f32,
  nearest_coord: vec2i,
  farthest: f32,
}

fn find_depth_extents(p: vec2i) -> DepthExtents {
  let size = render_size();
  var offsets = array<vec2i, 9>(vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(0, -1), vec2i(-1, 0), vec2i(-1, 1),
                                vec2i(1, 1), vec2i(-1, -1), vec2i(1, -1));
  var depth: array<f32, 9>;
  for (var i = 0; i < 9; i++) {
    depth[i] = load_input_depth(p + offsets[i]);
  }
  var extents = DepthExtents(depth[0], p, depth[0]);
  for (var i = 1; i < 9; i++) {
    let q = p + offsets[i];
    if (is_on_screen(q, size) && depth[i] > extents.nearest) {
      extents.farthest = min(extents.farthest, depth[i]);
      extents.nearest_coord = q;
      extents.nearest = depth[i];
    }
  }
  return extents;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec2i(id.xy);
  if (!is_on_screen(p, render_size())) {
    return;
  }
  let extents = find_depth_extents(p);
  let motion = load_input_motion(extents.nearest_coord);
  reconstruct_prev_depth(p, extents.nearest, motion);
  textureStore(dilated_motion, p, vec4f(motion, 0.0, 0.0));
  textureStore(dilated_depth, p, vec4f(extents.nearest, 0.0, 0.0, 0.0));
  let farthest_m = min(get_view_space_depth_in_meters(extents.farthest), FP16_MAX);
  textureStore(farthest_depth, p, vec4f(farthest_m, 0.0, 0.0, 0.0));
  let color = max(vec3f(0.0), textureLoad(input_color, p, 0).rgb);
  textureStore(current_luma, p, vec4f(rgb_to_luma(color), 0.0, 0.0, 0.0));
}

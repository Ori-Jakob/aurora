// The mip filter follows XeGTAO's DepthMIPFilter (GameTechDev/XeGTAO), which carries this notice:
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

@group(1) @binding(0) var hiz_source: texture_2d<f32>;
@group(1) @binding(1) var hiz_dest: texture_storage_2d<r32float, write>;

fn hiz_depth_mip_filter(depths: vec4f) -> f32 {
  let max_depth = max(max(depths.x, depths.y), max(depths.z, depths.w));
  let radius = max(0.75 * ao_params.effect_radius, 1e-6);
  let falloff_range = AO_FALLOFF_RANGE * radius;
  let falloff_from = radius * (1.0 - AO_FALLOFF_RANGE);
  let falloff_mul = -1.0 / falloff_range;
  let falloff_add = falloff_from / falloff_range + 1.0;
  let weights = saturate((vec4f(max_depth) - depths) * falloff_mul + falloff_add);
  return dot(weights, depths) / (weights.x + weights.y + weights.z + weights.w);
}

@compute @workgroup_size(8, 8)
fn linearize(@builtin(global_invocation_id) id: vec3u) {
  let grid = textureDimensions(hiz_dest);
  if (id.x >= grid.x || id.y >= grid.y) {
    return;
  }
  let full = textureDimensions(hiz_source, 0);
  let q = ao_rep_texel(id.xy, ao_divisor(full, grid), full);
  let z = camera_view_depth(ao_params.camera, textureLoad(hiz_source, q, 0).r);
  textureStore(hiz_dest, id.xy, vec4f(z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn filter_mip(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(hiz_dest);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let last = textureDimensions(hiz_source, 0) - 1u;
  let base = id.xy * 2u;
  let depths = vec4f(textureLoad(hiz_source, min(base, last), 0).r,
                     textureLoad(hiz_source, min(base + vec2u(1u, 0u), last), 0).r,
                     textureLoad(hiz_source, min(base + vec2u(0u, 1u), last), 0).r,
                     textureLoad(hiz_source, min(base + vec2u(1u, 1u), last), 0).r);
  textureStore(hiz_dest, id.xy, vec4f(hiz_depth_mip_filter(depths), 0.0, 0.0, 0.0));
}

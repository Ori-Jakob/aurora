//#import aurora/rcas

struct ApplyParams {
  sharpness: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<uniform> apply_params: ApplyParams;
@group(0) @binding(1) var upscaled: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  return vec4f(rcas_sharpen(upscaled, vec2i(floor(frag.xy)), apply_params.sharpness), 0.0);
}

//#import aurora/rcas
//#import aurora/srgb

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

//#if FSR_APPLY_LINEAR
fn stored_tap(p: vec2i) -> vec3f {
  return linear_to_srgb(rcas_tap(upscaled, p));
}
//#endif

@fragment
fn fs_main(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let p = vec2i(floor(frag.xy));
//#if FSR_APPLY_LINEAR
  let e = stored_tap(p);
  if (!(apply_params.sharpness > 0.0)) {
    return vec4f(e, 0.0);
  }
  // Linear history is sharpened on its re-encoded taps, as the stored path is
  return vec4f(rcas_filter(e, stored_tap(p + vec2i(0, -1)), stored_tap(p + vec2i(-1, 0)), stored_tap(p + vec2i(1, 0)),
                           stored_tap(p + vec2i(0, 1)), apply_params.sharpness), 0.0);
//#else
  return vec4f(rcas_sharpen(upscaled, p, apply_params.sharpness), 0.0);
//#endif
}

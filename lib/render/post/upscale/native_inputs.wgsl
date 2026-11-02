//#import aurora/srgb

//#if NATIVE_DEPTH
@group(0) @binding(0) var scene_depth: texture_depth_2d;
//#else
@group(0) @binding(0) var scene_color: texture_2d<f32>;
//#endif

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

//#if NATIVE_DEPTH
@fragment
fn fs_depth(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(scene_depth, vec2i(floor(position.xy)), 0), 0.0, 0.0, 1.0);
}
//#else
@fragment
fn fs_linear(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let stored = textureLoad(scene_color, vec2i(floor(position.xy)), 0);
  return vec4f(srgb_to_linear(stored.rgb), stored.a);
}
//#endif

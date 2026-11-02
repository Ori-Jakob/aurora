//#import aurora/rcas
//#import aurora/srgb

struct ComposeParams {
  scale: vec2f,
  jitter: vec2f,
  scene_size: vec2f,
  output_size: vec2f,
  rcas_strength: f32,
  flags: u32,
  pad: vec2u,
}

struct ComposeStatus {
  value: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ComposeParams;
@group(0) @binding(4) var source_color: texture_2d<f32>;
@group(0) @binding(5) var<uniform> status: ComposeStatus;
//#if !COMPOSE_INPLACE
@group(0) @binding(1) var linear_clamp: sampler;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
//#endif

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn stored_rgb(rgb: vec3f) -> vec3f {
//#if COMPOSE_LINEAR
  return linear_to_srgb(rgb);
//#else
  return rgb;
//#endif
}

fn stored_tap(p: vec2i) -> vec3f {
  return stored_rgb(rcas_tap(source_color, p));
}

fn source_rgb(position: vec2f) -> vec3f {
  let p = vec2i(floor(position));
  let e = stored_tap(p);
  if (!(params.rcas_strength > 0.0)) {
    return e;
  }
  // TAA, FSR and DLSS sharpen stored colour, so a linear source is sharpened on its re-encoded taps too
  return rcas_filter(e, stored_tap(p + vec2i(0, -1)), stored_tap(p + vec2i(-1, 0)), stored_tap(p + vec2i(1, 0)),
                     stored_tap(p + vec2i(0, 1)), params.rcas_strength);
}

//#if COMPOSE_INPLACE
@fragment
fn fs_inplace(@builtin(position) position: vec4f) -> @location(0) vec4f {
  if (status.value != 1u) {
    discard;
  }
  return vec4f(source_rgb(position.xy), 0.0);
}
//#else
struct ComposeOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

// The output EFB is empty until here, so carrying alpha and depth across keeps the game's shadow stencil and depth tests
@fragment
fn fs_split(@builtin(position) position: vec4f) -> ComposeOut {
  // Sampling at the jittered position lines the old scene up with the unjittered resolve
  let scene = position.xy * params.scale + params.jitter;
  let size = vec2i(textureDimensions(scene_color));
  let texel = clamp(vec2i(floor(scene)), vec2i(0), size - 1);
  let nearest = textureLoad(scene_color, texel, 0);
  var rgb = textureSampleLevel(scene_color, linear_clamp, scene / params.scene_size, 0.0).rgb;
  // Only the native copy test hands over a source at render size, so it is resampled like the scene colour
  if (status.value == 1u && all(textureDimensions(source_color) == textureDimensions(scene_color))) {
    rgb = stored_rgb(textureSampleLevel(source_color, linear_clamp, scene / params.scene_size, 0.0).rgb);
  } else if (status.value == 1u) {
    rgb = source_rgb(position.xy);
  }
  var out: ComposeOut;
  out.color = vec4f(rgb, nearest.a);
  out.depth = textureLoad(scene_depth, texel, 0);
  return out;
}
//#endif

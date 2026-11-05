//#import aurora/camera
//#import aurora/gbuffer
//#import aurora/vsm
//#import shadows/vsm_filter

struct SunVisibilityParams {
  camera: StageCamera,
  rotation: ViewRotation,
  flags: u32,
  taps: u32,
  frame: u32,
  normal_scale: f32,
  depth_bias: f32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}

const SUN_VIS_FLAG_PRODUCING: u32 = 1u;
const SUN_VIS_FLAG_COUNTERS: u32 = 2u;
const SUN_VIS_SAMPLE_MASK: u32 = 7u;

@group(0) @binding(1) var<uniform> sun_params: SunVisibilityParams;
@group(0) @binding(5) var sun_compare: sampler_comparison;
@group(1) @binding(0) var sun_depth: texture_2d<f32>;
@group(1) @binding(1) var sun_normal: texture_2d<f32>;
@group(1) @binding(2) var sun_pool: texture_depth_2d;
@group(1) @binding(3) var<storage, read> sun_pt: array<u32>;
@group(1) @binding(4) var<uniform> sun_views: array<vec4u, 144>;
@group(1) @binding(5) var<storage, read_write> sun_stats: array<atomic<u32>>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn sun_count(word: u32, on: bool) {
  if (on && word < arrayLength(&sun_stats)) {
    atomicAdd(&sun_stats[word], 1u);
  }
}

// The filter runs here so the VSM never needs its own screen target; RT and cloud transmittance merge in the same pass later
@fragment
fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let dims = textureDimensions(sun_depth);
  let px = min(vec2u(position.xy), dims - 1u);
  let d = textureLoad(sun_depth, px, 0).r;
  let normal_texel = textureLoad(sun_normal, min(px, textureDimensions(sun_normal) - 1u), 0);
  let cam = sun_params.camera;
  if (d <= 0.0 || gbuf_receiver_code(normal_texel.a) < 2u || (cam.flags & CAMERA_VALID) == 0u) {
    return vec4f(1.0);
  }
  let full = vec2f(dims);
  let z = camera_view_depth(cam, d);
  let pos = camera_view_position_fwd(cam, (vec2f(px) + 0.5) / full, z, full);
  let p_rel = camera_world_dir(sun_params.rotation, pos);
  let raw = camera_gx_normal_fwd(normal_texel.rgb * 2.0 - 1.0);
  let raw_length = length(raw);
  var n_world = vec3f(0.0);
  if (raw_length >= 1e-3) {
    n_world = camera_world_dir(sun_params.rotation, raw / raw_length);
  }
  let level = vsm_filter_select_level(&sun_views, 2.0 * z / (full.y * cam.depth.w), p_rel);
  let params = VsmFilterParams(sun_params.taps, sun_params.normal_scale, sun_params.depth_bias);
  let r = vsm_visibility(&sun_views, &sun_pt, sun_pool, sun_compare, level, p_rel, n_world, params);
  if ((sun_params.flags & SUN_VIS_FLAG_COUNTERS) != 0u) {
    if (r.unshadowed && atomicLoad(&sun_stats[VSM_STAT_SUN_UNSHADOWED]) == 0u) {
      atomicStore(&sun_stats[VSM_STAT_SUN_UNSHADOWED], 1u);
    }
    if (all((px & vec2u(SUN_VIS_SAMPLE_MASK)) == vec2u(0u))) {
      sun_count(VSM_STAT_SUN_SAMPLES, true);
      sun_count(VSM_STAT_SUN_SAMPLED_UNSHADOWED, r.unshadowed);
      sun_count(VSM_STAT_SUN_SAMPLED_FALLBACK, r.fallback);
      sun_count(VSM_STAT_SUN_SAMPLED_STALE, r.stale);
      sun_count(VSM_STAT_SUN_SAMPLED_OUT_OF_RANGE, r.out_of_range);
    }
  }
  return vec4f(r.v, 1.0, 1.0, 1.0);
}

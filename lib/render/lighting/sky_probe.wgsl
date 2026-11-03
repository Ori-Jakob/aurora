//#import aurora/camera
//#import aurora/sky
//#import aurora/srgb

const SKY_SIZE: u32 = 128u;
const SKY_TAPS: u32 = 4u;
const SKY_SH_SIZE: u32 = 32u;
const SKY_FLAG_OBSERVE: u32 = 1u;
const SKY_FLAG_RESET: u32 = 2u;
const SKY_FILTER_CONSTANT: u32 = 1u;
const SKY_FILTER_HAS_ENV: u32 = 2u;
const SKY_FILTER_PHYSICAL: u32 = 4u;
const SKY_FILTER_GLOBAL: u32 = 8u;
const SKY_FILTER_OBSERVED: u32 = 16u;
const SKY_GAIN_MIN: f32 = 0.00390625;
const SKY_GAIN_MAX: f32 = 16.0;

fn sky_lum(rgb: vec3f) -> f32 {
  return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

fn sky_finite(v: vec3f) -> bool {
  return all((bitcast<vec3u>(v) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u));
}

//#if SKY_PROJECT
struct ProjectParams {
  camera: StageCamera,
  rot: ViewRotation,
  flags: u32,
  alpha: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(1) var<uniform> project_params: ProjectParams;
@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var scene_depth: texture_2d<f32>;
@group(1) @binding(2) var scene_fog: texture_2d<u32>;
@group(1) @binding(3) var radiance: texture_storage_2d<rgba16float, write>;
@group(2) @binding(0) var radiance_prev: texture_2d<f32>;

fn sky_observe(texel: vec2u) -> vec4f {
  let cam = project_params.camera;
  let dims = textureDimensions(scene_color);
  let full = vec2f(dims);
  var sum = vec3f(0.0);
  var count = 0u;
  for (var b = 0u; b < SKY_TAPS; b++) {
    for (var a = 0u; a < SKY_TAPS; a++) {
      let uv = (vec2f(texel) + (vec2f(f32(a), f32(b)) + 0.5) / f32(SKY_TAPS)) / f32(SKY_SIZE);
      let dir = sky_oct_decode(uv);
      if (dir.y < 0.0) {
        continue;
      }
      let s = camera_project_fwd(cam, camera_view_dir_fwd(project_params.rot, dir), full);
      if (s.z < 0.5 || any(s.xy < vec2f(0.0)) || any(s.xy >= vec2f(1.0))) {
        continue;
      }
      let px = min(vec2u(s.xy * full), dims - 1u);
      let depth = textureLoad(scene_depth, min(px, textureDimensions(scene_depth) - 1u), 0).r;
      let rec = textureLoad(scene_fog, min(px, textureDimensions(scene_fog) - 1u), 0);
      if (!sky_pixel_is_sky(depth, rec)) {
        continue;
      }
      sum += srgb_to_linear(textureLoad(scene_color, px, 0).rgb);
      count++;
    }
  }
  if (count == 0u) {
    return vec4f(0.0);
  }
  return vec4f(sum / f32(count), 1.0);
}

@compute @workgroup_size(16, 16)
fn project(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(SKY_SIZE))) {
    return;
  }
  var prev = textureLoad(radiance_prev, min(id.xy, textureDimensions(radiance_prev) - 1u), 0);
  if ((project_params.flags & SKY_FLAG_RESET) != 0u || !sky_finite(prev.rgb)) {
    prev = vec4f(0.0);
  }
  let centre = sky_oct_decode((vec2f(id.xy) + 0.5) / f32(SKY_SIZE));
  var out = prev;
  if ((project_params.flags & SKY_FLAG_OBSERVE) != 0u && centre.y >= 0.0) {
    let seen = sky_observe(id.xy);
    if (seen.w > 0.0 && sky_finite(seen.rgb)) {
      out = select(mix(prev, seen, project_params.alpha), seen, prev.w <= 0.0);
    }
  }
  // The lower hemisphere is never observed; the fill gives it the ambient
  if (centre.y < 0.0) {
    out = vec4f(0.0);
  }
  textureStore(radiance, id.xy, out);
}
//#else
struct FilterParams {
  rot: ViewRotation,
  fog: vec4f,
  ambient: vec4f,
  flags: u32,
  constant: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(1) var<uniform> filter_params: FilterParams;
@group(1) @binding(0) var sky_src: texture_2d<f32>;
@group(1) @binding(1) var sky_dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<storage, read_write> sky_out: SkySh9;

var<workgroup> gain_sums: array<vec2f, 256>;
var<workgroup> sh_sums: array<array<vec4f, 7>, 64>;

fn sky_has(flag: u32) -> bool {
  return (filter_params.flags & flag) != 0u;
}

fn sky_ambient() -> vec3f {
  return srgb_to_linear(filter_params.ambient.rgb);
}

fn sky_base(texel: vec2u) -> vec3f {
  let r = textureLoad(sky_src, texel, 0);
  return select(srgb_to_linear(filter_params.fog.rgb), r.rgb, r.w > 0.0);
}

fn sky_mode_of() -> u32 {
  if (sky_has(SKY_FILTER_CONSTANT) || sky_has(SKY_FILTER_PHYSICAL) || sky_has(SKY_FILTER_GLOBAL)) {
    return SKY_MODE_RAW;
  }
  return select(SKY_MODE_PIXEL_FLAT, SKY_MODE_PIXEL,
                sky_has(SKY_FILTER_HAS_ENV) && sky_lum(sky_ambient()) >= SKY_AMBIENT_FLOOR);
}

@compute @workgroup_size(256)
fn gain(@builtin(local_invocation_index) i: u32) {
  var s = vec2f(0.0);
  for (var k = 0u; k < 64u; k++) {
    let t = i * 64u + k;
    let texel = vec2u(t % SKY_SIZE, t / SKY_SIZE);
    let uv = (vec2f(texel) + 0.5) / f32(SKY_SIZE);
    let dir = sky_oct_decode(uv);
    if (dir.y <= 0.0) {
      continue;
    }
    let w = sky_texel_weight(uv, f32(SKY_SIZE)) * dir.y;
    s += vec2f(w * sky_lum(sky_base(texel)), w);
  }
  gain_sums[i] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (i < stride) {
      gain_sums[i] += gain_sums[i + stride];
    }
    workgroupBarrier();
  }
  if (i == 0u) {
    let mean = gain_sums[0].x / max(gain_sums[0].y, 1e-20);
    var g = 1.0;
    if (!sky_has(SKY_FILTER_PHYSICAL) && !sky_has(SKY_FILTER_CONSTANT) && sky_has(SKY_FILTER_HAS_ENV)) {
      g = clamp(sky_lum(sky_ambient()) / max(mean, 1e-6), SKY_GAIN_MIN, SKY_GAIN_MAX);
    }
    sky_out.info = vec4f(0.0, filter_params.fog.w, select(0.0, 1.0, sky_has(SKY_FILTER_OBSERVED)), g);
  }
}

@compute @workgroup_size(8, 8)
fn fill(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(SKY_SIZE))) {
    return;
  }
  let a = sky_ambient();
  let dir = sky_oct_decode((vec2f(id.xy) + 0.5) / f32(SKY_SIZE));
  let c = select(a, sky_out.info.w * sky_base(id.xy), dir.y >= 0.0);
  var out = mix(a, c, filter_params.fog.w);
  if (sky_has(SKY_FILTER_CONSTANT)) {
    out = vec3f(filter_params.constant);
  }
  textureStore(sky_dst, id.xy, vec4f(out, 1.0));
}

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(sky_dst);
  if (any(id.xy >= dims)) {
    return;
  }
  let src = id.xy * 2u;
  let sum = textureLoad(sky_src, src, 0) + textureLoad(sky_src, src + vec2u(1u, 0u), 0) +
            textureLoad(sky_src, src + vec2u(0u, 1u), 0) + textureLoad(sky_src, src + vec2u(1u, 1u), 0);
  textureStore(sky_dst, id.xy, sum * 0.25);
}

fn sky_band_factor(i: u32) -> f32 {
  if (i == 0u) {
    return SKY_PI;
  }
  return select(SKY_PI / 4.0, 2.0 * SKY_PI / 3.0, i < 4u);
}

// Projecting only the difference from the ambient leaves a constant probe free of quadrature error
fn sky_sh_base() -> vec3f {
  return select(sky_ambient(), vec3f(filter_params.constant), sky_has(SKY_FILTER_CONSTANT));
}

@compute @workgroup_size(64)
fn sh9(@builtin(local_invocation_index) i: u32) {
  let base = sky_sh_base();
  var acc = array<vec4f, 7>();
  for (var k = 0u; k < 16u; k++) {
    let t = i * 16u + k;
    let texel = vec2u(t % SKY_SH_SIZE, t / SKY_SH_SIZE);
    let uv = (vec2f(texel) + 0.5) / f32(SKY_SH_SIZE);
    let w = sky_texel_weight(uv, f32(SKY_SH_SIZE));
    let l = (textureLoad(sky_src, texel, 0).rgb - base) * w;
    let y = sky_sh_basis(sky_oct_decode(uv));
    acc[0] += vec4f(l * y[0], y[1] * l.r);
    acc[1] += vec4f(y[1] * l.gb, y[2] * l.rg);
    acc[2] += vec4f(y[2] * l.b, y[3] * l);
    acc[3] += vec4f(y[4] * l, y[5] * l.r);
    acc[4] += vec4f(y[5] * l.gb, y[6] * l.rg);
    acc[5] += vec4f(y[6] * l.b, y[7] * l);
    acc[6] += vec4f(y[8] * l, w);
  }
  sh_sums[i] = acc;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (i < stride) {
      for (var j = 0u; j < 7u; j++) {
        sh_sums[i][j] += sh_sums[i + stride][j];
      }
    }
    workgroupBarrier();
  }
  if (i != 0u) {
    return;
  }
  let sums = sh_sums[0];
  var flat = array<f32, 28>();
  for (var j = 0u; j < 7u; j++) {
    flat[j * 4u] = sums[j].x;
    flat[j * 4u + 1u] = sums[j].y;
    flat[j * 4u + 2u] = sums[j].z;
    flat[j * 4u + 3u] = sums[j].w;
  }
  let norm = 4.0 * SKY_PI / max(flat[27], 1e-20);
  for (var c = 0u; c < 9u; c++) {
    let e = vec3f(flat[c * 3u], flat[c * 3u + 1u], flat[c * 3u + 2u]) * (norm * sky_band_factor(c));
    sky_out.c[c] = vec4f(e, 0.0);
  }
  sky_out.c[0] += vec4f(SKY_PI * sqrt(4.0 * SKY_PI) * base, 0.0);
  sky_out.info.x = 1.0;
  sky_out.amb = vec4f(sky_ambient(), f32(sky_mode_of()));
  sky_out.rot = filter_params.rot.rows;
}
//#endif

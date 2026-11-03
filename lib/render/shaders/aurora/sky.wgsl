// Sky probe v1: linear Rec.709 in legacy-white units with the gain and skyVisibility already applied; consumers pass
// their own ambient a and BAKED_LIGHT flag, and never add ENVIRONMENT ambient or multiply by skyVisibility

struct SkySh9 {
  c: array<vec4f, 9>,
  info: vec4f,
  amb: vec4f,
  rot: array<vec4f, 3>,
}

const SKY_PI: f32 = 3.14159265358979;
const SKY_PROBE_LODS: f32 = 5.0;
const SKY_MODE_RAW: u32 = 0u;
const SKY_MODE_PIXEL: u32 = 1u;
const SKY_MODE_PIXEL_FLAT: u32 = 2u;
const SKY_AMBIENT_FLOOR: f32 = 0.0009765625;
const SKY_SCALE_MAX: f32 = 16.0;

fn sky_sign2(v: vec2f) -> vec2f {
  return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

// +Y sits at the centre, so the observed upper hemisphere is one contiguous diamond
fn sky_oct_point(uv: vec2f) -> vec3f {
  let e = uv * 2.0 - 1.0;
  let y = 1.0 - abs(e.x) - abs(e.y);
  var xz = e;
  if (y < 0.0) {
    xz = (1.0 - abs(e.yx)) * sky_sign2(e);
  }
  return vec3f(xz.x, y, xz.y);
}

fn sky_oct_decode(uv: vec2f) -> vec3f {
  return normalize(sky_oct_point(uv));
}

fn sky_oct_encode(dir: vec3f) -> vec2f {
  let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
  if (!(l1 > 1e-20)) {
    return vec2f(0.5);
  }
  let p = dir / l1;
  var xz = p.xz;
  if (p.y < 0.0) {
    xz = (1.0 - abs(p.zx)) * sky_sign2(p.xz);
  }
  return xz * 0.5 + 0.5;
}

fn sky_texel_weight(uv: vec2f, size: f32) -> f32 {
  let l = length(sky_oct_point(uv));
  return 4.0 / (size * size * l * l * l);
}

fn sky_sh_basis(d: vec3f) -> array<f32, 9> {
  return array<f32, 9>(0.28209479177, 0.48860251190 * d.y, 0.48860251190 * d.z, 0.48860251190 * d.x,
                       1.09254843059 * d.x * d.y, 1.09254843059 * d.y * d.z, 0.31539156525 * (3.0 * d.z * d.z - 1.0),
                       1.09254843059 * d.x * d.z, 0.54627421529 * (d.x * d.x - d.y * d.y));
}

// Coefficients already carry the cosine-lobe factors, so this is irradiance, not radiance
fn sky_sh9_irradiance(sh: SkySh9, n: vec3f) -> vec3f {
  let y = sky_sh_basis(n);
  var e = vec3f(0.0);
  for (var i = 0u; i < 9u; i++) {
    e += sh.c[i].rgb * y[i];
  }
  return max(e, vec3f(0.0));
}

fn sky_world_from_fwd(sh: SkySh9, fwd: vec3f) -> vec3f {
  return sh.rot[0].xyz * fwd.x + sh.rot[1].xyz * fwd.y - sh.rot[2].xyz * fwd.z;
}

fn sky_probe_lod(roughness: f32) -> f32 {
  return saturate(roughness) * SKY_PROBE_LODS;
}

fn sky_probe_radiance(tex: texture_2d<f32>, samp: sampler, dir: vec3f, roughness: f32) -> vec3f {
  return textureSampleLevel(tex, samp, sky_oct_encode(dir), sky_probe_lod(roughness)).rgb;
}

fn sky_sh9_valid(sh: SkySh9) -> bool {
  return sh.info.x == 1.0;
}

fn sky_mode(sh: SkySh9) -> u32 {
  return u32(clamp(sh.amb.w, 0.0, 2.0));
}

fn sky_pixel_is_sky(depth: f32, rec: vec4u) -> bool {
  return depth <= 0.0 && all(rec == vec4u(0u));
}

fn sky_pixel_scale(sh: SkySh9, a: vec3f) -> vec3f {
  if (sky_mode(sh) != SKY_MODE_PIXEL) {
    return vec3f(1.0);
  }
  return clamp(a / max(sh.amb.rgb, vec3f(SKY_AMBIENT_FLOOR)), vec3f(0.0), vec3f(SKY_SCALE_MAX));
}

fn sky_env_irradiance(sh: SkySh9, n: vec3f, a: vec3f, baked: bool) -> vec3f {
  let mode = sky_mode(sh);
  if (mode == SKY_MODE_PIXEL_FLAT || (mode == SKY_MODE_PIXEL && baked)) {
    return SKY_PI * a;
  }
  return sky_pixel_scale(sh, a) * sky_sh9_irradiance(sh, n);
}

fn sky_env_radiance(tex: texture_2d<f32>, samp: sampler, sh: SkySh9, dir: vec3f, roughness: f32, a: vec3f) -> vec3f {
  return sky_pixel_scale(sh, a) * sky_probe_radiance(tex, samp, dir, roughness);
}

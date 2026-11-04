// Bitmask indirect light after Therrien, Levesque and Gilet 2023 (arXiv 2301.11376); cosines replaced for M8's sectors

const SSIL_PI: f32 = 3.14159265358979;
const SSIL_RADIANCE_CLAMP: f32 = 16.0;
const SSIL_FACING_SLOPE: f32 = 4.0;
const SSIL_DEFOG_MIN: f32 = 13.0 / 127.0;
const SSIL_LINEAR_FLOOR: f32 = 0.000244140625;
const SSIL_LUMA: vec3f = vec3f(0.2126, 0.7152, 0.0722);

fn ssil_linear_ao(a: f32) -> f32 {
  return pow(a, 2.2);
}

fn ssil_clamp_radiance(rgb: vec3f) -> vec3f {
  let finite = all((bitcast<vec3u>(rgb) & vec3u(0x7f800000u)) != vec3u(0x7f800000u));
  let c = select(vec3f(0.0), max(rgb, vec3f(0.0)), finite);
  let peak = max(max(c.r, c.g), c.b);
  return select(c, min(c * (SSIL_RADIANCE_CLAMP / peak), vec3f(SSIL_RADIANCE_CLAMP)), peak > SSIL_RADIANCE_CLAMP);
}

fn ssil_encode_normal(n: vec3f) -> vec3f {
  return 0.5 * n + 0.5;
}

fn ssil_decode_normal(t: vec4f) -> vec3f {
  return t.rgb * 2.0 - 1.0;
}

// A zero mean (code 0, mixed footprint) counts as facing, like other sources the normal cannot judge
fn ssil_emitter_facing(n_mean: vec3f, l: vec3f) -> f32 {
  return saturate(1.0 + SSIL_FACING_SLOPE * dot(n_mean, -l));
}

fn ssil_bounce_irradiance(t: vec4f, v_lin: f32) -> vec3f {
  return SSIL_PI * (1.0 - v_lin) * t.rgb;
}

// Dividing by the receiver's luminance lets its own colour act as the albedo tint, so creases keep their hue
fn ssil_legacy_gain(a: f32, bounce: vec3f, receiver_lin: vec3f) -> vec3f {
  let a_lin = ssil_linear_ao(a);
  let x = (1.0 - a_lin) * bounce / max(dot(receiver_lin, SSIL_LUMA), SSIL_LINEAR_FLOOR);
  return max(pow(min(vec3f(a_lin) + x, vec3f(1.0)), vec3f(1.0 / 2.2)), vec3f(a));
}

fn ssil_prev_depth_weights(f: vec2f, log_depths: vec4f, z: f32, reject: f32) -> vec4f {
  let bilinear = vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
  return select(vec4f(0.0), bilinear, abs(exp2(log_depths) - vec4f(z)) <= vec4f(reject * z));
}

// Every consumer reads the bounce here, so its producer can change resolution without touching them
fn ssil_fetch(tex: texture_2d<f32>, p: vec2i) -> vec4f {
  return textureLoad(tex, min(p, vec2i(textureDimensions(tex)) - 1), 0);
}

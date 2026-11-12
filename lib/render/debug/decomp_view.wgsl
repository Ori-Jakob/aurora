//#import aurora/frame
//#import aurora/gbuffer
//#if DECOMP_MULTIPLY
//#import aurora/ao
//#endif
//#if DECOMP_LSM
//#import aurora/lsm
//#endif

@group(1) @binding(0) var color_tex: texture_2d<f32>;
@group(1) @binding(1) var fog_tex: texture_2d<u32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
//#if DECOMP_IDENTITY
@group(1) @binding(3) var legacy_tex: texture_2d<f32>;
//#endif
//#if DECOMP_COVERAGE
@group(1) @binding(3) var reactive_tex: texture_2d<f32>;
//#endif
//#if DECOMP_MULTIPLY
@group(1) @binding(3) var ao_tex: texture_2d<f32>;
//#endif
//#if DECOMP_LSM
@group(1) @binding(3) var lsm_tex: texture_2d<f32>;
@group(1) @binding(4) var hdr_tex: texture_2d<f32>;
//#endif

const IDENTITY_UNCHANGED = vec3f(0.0, 0.0, 0.0);
const IDENTITY_LIFTED = vec3f(255.0, 128.0, 0.0) / 255.0;
const IDENTITY_BRIGHTENED = vec3f(0.0, 255.0, 255.0) / 255.0;
const IDENTITY_CODE_ZERO = vec3f(255.0, 0.0, 0.0) / 255.0;
const IDENTITY_ALPHA = vec3f(255.0, 255.0, 0.0) / 255.0;
const COVERAGE_UNTOUCHED = vec3f(64.0, 64.0, 64.0) / 255.0;
const COVERAGE_FOGGED_NON_RECEIVER = vec3f(40.0, 40.0, 110.0) / 255.0;
const COVERAGE_RECEIVER = vec3f(80.0, 160.0, 255.0) / 255.0;
const COVERAGE_FOGGED_RECEIVER = vec3f(60.0, 200.0, 60.0) / 255.0;
const COVERAGE_LOCKED_RECEIVER = vec3f(255.0, 160.0, 0.0) / 255.0;
const COVERAGE_BAD = vec3f(255.0, 0.0, 255.0) / 255.0;
const DEFOG_BAD = vec3f(255.0, 0.0, 255.0) / 255.0;
const LSM_POST_SHADOW_TINT = vec3f(0.0, 220.0, 220.0) / 255.0;

struct ViewVarying {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

struct Decoded {
  l: vec4f,
  rec: GbufFog,
  s: vec3f,
  bad: bool,
  code: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> ViewVarying {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return ViewVarying(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), vec2f(uv.x, 1.0 - uv.y));
}

// AO may run below scene size and the split may change sizes, so every load maps UV to texels
fn fetch(t: texture_2d<f32>, uv: vec2f) -> vec4f {
  let dims = textureDimensions(t);
  return textureLoad(t, min(vec2u(uv * vec2f(dims)), dims - 1u), 0);
}

fn fetch_record(uv: vec2f) -> vec4u {
  let dims = textureDimensions(fog_tex);
  return textureLoad(fog_tex, min(vec2u(uv * vec2f(dims)), dims - 1u), 0);
}

fn decode(l: vec4f, uv: vec2f) -> Decoded {
  let rec = gbuf_fog_decode(fetch_record(uv));
  let s = l.rgb - rec.q * rec.fc;
  let code = gbuf_receiver_code(fetch(normal_tex, uv).a);
  return Decoded(l, rec, s, any(s < vec3f(-2.0 / 255.0)), code);
}

fn identity_grey(d: f32) -> vec3f {
  return vec3f((63.0 + 8.0 * min(d, 24.0)) / 255.0);
}

fn coverage_touched(c: vec3f) -> vec3f {
  return floor((round(c * 255.0) + 256.0) * 0.5) / 255.0;
}

//#if DECOMP_IDENTITY
fn mode_color(uv: vec2f) -> vec3f {
  let o = fetch(color_tex, uv);
  let d = decode(fetch(legacy_tex, uv), uv);
  if (o.a != d.l.a) {
    return IDENTITY_ALPHA;
  }
  if (all(o.rgb == d.l.rgb)) {
    return IDENTITY_UNCHANGED;
  }
  if (d.code == 0u) {
    return IDENTITY_CODE_ZERO;
  }
  let up = o.rgb > d.l.rgb;
  if (any(up)) {
    return select(IDENTITY_BRIGHTENED, IDENTITY_LIFTED, all(!up | (d.s < vec3f(0.0))));
  }
  let diff = round(abs(o.rgb - d.l.rgb) * 255.0);
  return identity_grey(max(diff.r, max(diff.g, diff.b)));
}
//#else
//#if DECOMP_COVERAGE
fn mode_color(uv: vec2f) -> vec3f {
  let d = decode(fetch(color_tex, uv), uv);
  if (d.bad) {
    return COVERAGE_BAD;
  }
  if (d.code == 0u) {
    return select(COVERAGE_UNTOUCHED, COVERAGE_FOGGED_NON_RECEIVER, d.rec.q > 0.0);
  }
  var base = COVERAGE_RECEIVER;
  if (d.rec.q > 0.0) {
    base = select(COVERAGE_FOGGED_RECEIVER, COVERAGE_LOCKED_RECEIVER, d.rec.lock);
  }
  return select(base, coverage_touched(base), fetch(reactive_tex, uv).r > 0.0);
}
//#else
//#if DECOMP_MULTIPLY
fn mode_color(uv: vec2f) -> vec3f {
  let d = decode(fetch(color_tex, uv), uv);
  return select(d.l.rgb, d.l.rgb * fetch(ao_tex, uv).r, d.code >= 1u);
}
//#else
//#if DECOMP_LSM
fn mode_color(uv: vec2f) -> vec3f {
  let mask = fetch(lsm_tex, uv).rgb;
  let a = fetch(hdr_tex, uv).a;
  if (lsm_pre_shadow(a)) {
    let rec = gbuf_fog_decode(fetch_record(uv));
    return lsm_term(mask, rec.q, rec.fc);
  }
  return select(mask, mix(mask, LSM_POST_SHADOW_TINT, 0.5), a >= 0.5);
}
//#else
fn mode_color(uv: vec2f) -> vec3f {
  let d = decode(fetch(color_tex, uv), uv);
  return select(saturate(d.s), DEFOG_BAD, d.bad);
}
//#endif
//#endif
//#endif
//#endif

@fragment
fn fs_main(in: ViewVarying) -> @location(0) vec4f {
  return vec4f(mode_color(in.uv), 1.0);
}

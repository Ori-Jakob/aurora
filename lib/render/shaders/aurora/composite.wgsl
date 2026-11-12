// Shared by composite.main and later checks; later terms append fields and must keep identity exact
//#import aurora/gbuffer
//#import aurora/srgb
//#import aurora/ssil
//#import aurora/tonemap

const COMPOSITE_FLAG_IDENTITY: u32 = 1u;
const COMPOSITE_FLAG_NO_FOG: u32 = 2u;
const COMPOSITE_FLAG_PBR: u32 = 4u;
const COMPOSITE_FLAG_TONEMAP_AGX: u32 = 8u;
const COMPOSITE_FLAG_SSIL: u32 = 16u;
const COMPOSITE_FLAG_LSM: u32 = 32u;

struct CompositeTerms {
  ao: f32,
  pbr: bool,
  lb: f32,
  m: vec3f,
  h: vec3f,
  e: f32,
  agx: bool,
  bounce: vec3f,
}

struct CompositeEval {
  rgb: vec3f,
  s: vec3f,
  bad: bool,
  keep: bool,
}

fn composite_identity_terms() -> CompositeTerms {
  return CompositeTerms(1.0, false, 1.0, vec3f(1.0), vec3f(0.0), 1.0, false, vec3f(0.0));
}

fn composite_is_identity(t: CompositeTerms) -> bool {
  return t.ao >= 1.0 && !(t.pbr && t.lb < 1.0);
}

fn composite_texel(p: vec2i, dims: vec2u) -> vec2i {
  return min(p, vec2i(dims) - vec2i(1));
}

// The composite and the meter both call this, so their AO and receiver rules cannot drift
fn composite_ao_term(ao_tex: texture_2d<f32>, normal_tex: texture_2d<f32>, p: vec2i, identity: bool) -> f32 {
  let a = textureLoad(ao_tex, composite_texel(p, textureDimensions(ao_tex)), 0).r;
  let code = gbuf_receiver_code(textureLoad(normal_tex, composite_texel(p, textureDimensions(normal_tex)), 0).a);
  return select(1.0, a, !identity && a < 1.0 && code >= 1u);
}

// Exposure applies only while the PBR term is armed; radiance takes .y and legacy bounce .x, so the two cancel
fn composite_units(flags: u32, e: f32, inv_e: f32) -> vec2f {
  return select(vec2f(1.0), vec2f(e, inv_e), (flags & COMPOSITE_FLAG_PBR) != 0u);
}

fn composite_ssil_term(tex: texture_2d<f32>, p: vec2i, e: f32, rho: f32) -> vec3f {
  return rho * e * ssil_fetch(tex, p).rgb;
}

// Bounce gives back at most what AO took, so legacy pixels never pass the original and a = 1 stays identity
fn composite_legacy_scale(t: CompositeTerms, sp: vec3f, q: f32) -> vec3f {
  if (all(t.bounce == vec3f(0.0))) {
    return vec3f(t.ao);
  }
  return ssil_legacy_gain(t.ao, t.bounce, srgb_to_linear(saturate(sp / max(1.0 - q, SSIL_DEFOG_MIN))));
}

fn composite_pbr_tonemap(x: vec3f, agx: bool) -> vec3f {
  return linear_to_srgb(tonemap_apply(x, agx));
}

fn composite_eval(l: vec3f, rec: GbufFog, t: CompositeTerms) -> CompositeEval {
  let s = l - rec.q * rec.fc;
  let bad = any(s < vec3f(-2.0 / 255.0));
  // Subtracting the darkening gives L exactly at a = 1 and never brightens a channel whose fog estimate rounds above it
  var rgb = l - max(s, vec3f(0.0)) * (1.0 - t.ao);
  if (any(t.bounce > vec3f(0.0))) {
    rgb = l - max(s, vec3f(0.0)) * (1.0 - composite_legacy_scale(t, max(s, vec3f(0.0)), rec.q));
  }
  // Full legacy blend selects the AO-only expression itself, so PBR on equals PBR off there (rev3 proof 4)
  if (t.pbr) {
    let sp = max(s, vec3f(0.0));
    let pv = (1.0 - rec.q) * t.m * composite_pbr_tonemap(t.h * t.e, t.agx);
    rgb = select(l - sp + mix(sp * composite_legacy_scale(t, sp, rec.q), pv, 1.0 - t.lb), rgb, t.lb >= 1.0);
  }
  return CompositeEval(rgb, s, bad, bad || composite_is_identity(t));
}
//#if COMPOSITE_RADIANCE

// The meter's q7 114 cut: fog-drowned pixels give darker radiance, never fog colour
const COMPOSITE_RADIANCE_MIN_T: f32 = 13.0 / 127.0;

// Divides GX fog out of the pre-fog result, so screen-space passes never gather fog colour
fn composite_radiance(e: CompositeEval, rec: GbufFog, t: CompositeTerms, inv_e: f32) -> vec4f {
  let sx = max(e.s, vec3f(0.0)) * composite_legacy_scale(t, max(e.s, vec3f(0.0)), rec.q);
  let leg = srgb_to_linear(saturate(sx / max(1.0 - rec.q, COMPOSITE_RADIANCE_MIN_T))) * inv_e;
  let r = select(leg, mix(leg, t.h, 1.0 - t.lb), t.pbr && t.lb < 1.0);
  return select(vec4f(r, 1.0), vec4f(0.0), e.bad);
}
//#endif

// Shared by composite.main and later checks; later terms append fields and must keep identity exact
//#import aurora/gbuffer
//#import aurora/srgb

const COMPOSITE_FLAG_IDENTITY: u32 = 1u;
const COMPOSITE_FLAG_NO_FOG: u32 = 2u;
const COMPOSITE_FLAG_PBR: u32 = 4u;
const COMPOSITE_PBR_EXPOSURE: f32 = 1.0;

struct CompositeTerms {
  ao: f32,
  pbr: bool,
  lb: f32,
  m: f32,
  h: vec3f,
}

struct CompositeEval {
  rgb: vec3f,
  s: vec3f,
  bad: bool,
  keep: bool,
}

fn composite_identity_terms() -> CompositeTerms {
  return CompositeTerms(1.0, false, 1.0, 1.0, vec3f(0.0));
}

fn composite_is_identity(t: CompositeTerms) -> bool {
  return t.ao >= 1.0 && !(t.pbr && t.lb < 1.0);
}

fn composite_texel(p: vec2i, dims: vec2u) -> vec2i {
  return min(p, vec2i(dims) - vec2i(1));
}

// Interim encode until W4 brings exposure and a tonemapper
fn composite_pbr_tonemap(x: vec3f) -> vec3f {
  return linear_to_srgb(x);
}

fn composite_eval(l: vec3f, rec: GbufFog, t: CompositeTerms) -> CompositeEval {
  let s = l - rec.q * rec.fc;
  let bad = any(s < vec3f(-2.0 / 255.0));
  // Subtracting the darkening gives L exactly at a = 1 and never brightens a channel whose fog estimate rounds above it
  var rgb = l - max(s, vec3f(0.0)) * (1.0 - t.ao);
  // Full legacy blend selects the AO-only expression itself, so PBR on equals PBR off there (rev3 proof 4)
  if (t.pbr) {
    let sp = max(s, vec3f(0.0));
    let pv = (1.0 - rec.q) * t.m * composite_pbr_tonemap(t.h * COMPOSITE_PBR_EXPOSURE);
    rgb = select(l - sp + mix(sp * t.ao, pv, 1.0 - t.lb), rgb, t.lb >= 1.0);
  }
  return CompositeEval(rgb, s, bad, bad || composite_is_identity(t));
}

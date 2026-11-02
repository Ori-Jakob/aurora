// Shared by composite.main and later checks; later terms append fields and must keep identity exact
//#import aurora/gbuffer

const COMPOSITE_FLAG_IDENTITY: u32 = 1u;
const COMPOSITE_FLAG_NO_FOG: u32 = 2u;

struct CompositeTerms {
  ao: f32,
}

struct CompositeEval {
  rgb: vec3f,
  s: vec3f,
  bad: bool,
  keep: bool,
}

fn composite_identity_terms() -> CompositeTerms {
  return CompositeTerms(1.0);
}

fn composite_is_identity(t: CompositeTerms) -> bool {
  return t.ao >= 1.0;
}

fn composite_eval(l: vec3f, rec: GbufFog, t: CompositeTerms) -> CompositeEval {
  let s = l - rec.q * rec.fc;
  let bad = any(s < vec3f(-2.0 / 255.0));
  // Subtracting the darkening gives L exactly at a = 1 and never brightens a channel whose fog estimate rounds above it
  let rgb = l - max(s, vec3f(0.0)) * (1.0 - t.ao);
  return CompositeEval(rgb, s, bad, bad || composite_is_identity(t));
}

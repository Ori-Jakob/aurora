// The kept fakes replay with their own GX fog, so the mask carries q*fc that the receiver's fog record removes

const LSM_PRE_SHADOW_ALPHA: f32 = 0.875;

fn lsm_term(mask: vec3f, q: f32, fc: vec3f) -> vec3f {
  let d = vec3f(1.0) - q * fc;
  return select(saturate((mask - q * fc) / d), vec3f(1.0), (mask >= vec3f(1.0)) | (d < vec3f(1.0 / 255.0)));
}

// Alpha 0.75 marks PBR drawn after the game's fake-shadow pass, which the fakes never darkened
fn lsm_pre_shadow(hdr_alpha: f32) -> bool {
  return hdr_alpha > LSM_PRE_SHADOW_ALPHA;
}

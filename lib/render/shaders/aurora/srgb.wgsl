// Resolvers that want linear light decode the stored EFB colour and re-encode their output with this one curve
fn srgb_to_linear(stored: vec3f) -> vec3f {
  return select(pow((stored + 0.055) / 1.055, vec3f(2.4)), stored / 12.92, stored <= vec3f(0.04045));
}

fn linear_to_srgb(rgb: vec3f) -> vec3f {
  let linear = saturate(rgb);
  return select(1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear * 12.92, linear <= vec3f(0.0031308));
}

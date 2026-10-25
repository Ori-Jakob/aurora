struct GbufFog {
  fc: vec3f,
  q: f32,
  lock: bool,
}

fn gbuf_fog_decode(rec: vec4u) -> GbufFog {
  return GbufFog(vec3f(rec.rgb) / 255.0, f32(rec.a >> 1u) / 127.0, (rec.a & 1u) != 0u);
}

fn gbuf_receiver_code(a: f32) -> u32 {
  return u32(round(saturate(a) * 3.0));
}

fn gbuf_motion_pixels(m: vec2f, efb: vec2f) -> vec2f {
  return m * 0.5 * efb * vec2f(1.0, -1.0);
}

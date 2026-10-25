struct AuroraGxFog {
  color: vec4f,
  a: f32,
  b: f32,
  c: f32,
  range_center: f32,
  range_k: array<vec4f, 3>,
  fog_type: u32,
  range_enabled: u32,
  render_width: f32,
  viewport_znear: f32,
  viewport_zfar: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

fn aurora_gx_fog_range_factor(fog: AuroraGxFog, pixel_x: f32) -> f32 {
  let range_k = fog.range_k;
  let screen_x = ((floor(pixel_x) + 0.5) / fog.render_width) * 2.0 - 1.0;
  let offset = screen_x - fog.range_center;
  let range_index = clamp(9.0 - abs(offset) * 9.0, 0.0, 9.0);
  let lower = u32(range_index);
  let upper = min(lower + 1u, 9u);
  let fraction = range_index - f32(lower);
  let k = max(range_k[lower / 4u][lower % 4u] * (1.0 - fraction) + range_k[upper / 4u][upper % 4u] * fraction,
              0.000001);
  return sqrt(offset * offset + k * k) / k;
}

fn aurora_gx_fog_weight(fog: AuroraGxFog, zw: f32, pixel_x: f32) -> f32 {
  if ((fog.fog_type & 7u) == 0u) {
    return 0.0;
  }
  var fogBase: f32;
  if ((fog.fog_type & 8u) != 0u) {
    fogBase = fog.a * zw;
  } else {
    fogBase = fog.a / (fog.b - zw);
  }
  if (fog.range_enabled != 0u) {
    fogBase *= aurora_gx_fog_range_factor(fog, pixel_x);
  }
  var fogF = clamp(fogBase - fog.c, 0.0, 1.0);
  var fogZ = 0.0;
  switch (fog.fog_type & 7u) {
    case 2u: {
      fogZ = fogF;
    }
    case 4u: {
      fogZ = 1.0 - exp2(-8.0 * fogF);
    }
    case 5u: {
      fogZ = 1.0 - exp2(-8.0 * fogF * fogF);
    }
    case 6u: {
      fogZ = exp2(-8.0 * (1.0 - fogF));
    }
    case 7u: {
      fogF = 1.0 - fogF;
      fogZ = exp2(-8.0 * fogF * fogF);
    }
    default: {}
  }
  return clamp(fogZ, 0.0, 1.0);
}

fn aurora_gx_fog_apply(fog: AuroraGxFog, rgb: vec3f, fogZ: f32) -> vec3f {
  return mix(rgb, fog.color.rgb, clamp(fogZ, 0.0, 1.0));
}

fn aurora_gx_fog_record(fog: AuroraGxFog, fogZ: f32) -> vec4u {
  if ((fog.fog_type & 7u) == 0u) {
    return vec4u(0u);
  }
  let lock = select(1u, 0u, fog.fog_type == 2u);
  return vec4u(vec3u(round(fog.color.rgb * 255.0)),
               (min(u32(round(clamp(fogZ, 0.0, 1.0) * 127.0)), 127u) << 1u) | lock);
}

fn aurora_gx_fog_zw(fog: AuroraGxFog, clip: vec4f) -> f32 {
  return fog.viewport_zfar - (clip.z / clip.w) * (fog.viewport_zfar - fog.viewport_znear);
}

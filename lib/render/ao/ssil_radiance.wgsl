//#import aurora/ao
//#import aurora/gbuffer
//#import aurora/ssil

@group(1) @binding(0) var radiance_in: texture_2d<f32>;
@group(1) @binding(1) var history_in: texture_2d<f32>;
@group(1) @binding(2) var scene_motion: texture_2d<f32>;
@group(1) @binding(3) var ao_hiz: texture_2d<f32>;
@group(1) @binding(4) var scene_normal: texture_2d<f32>;
@group(1) @binding(5) var radiance_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var normal_out: texture_storage_2d<rgba8unorm, write>;

fn ssil_depth_mip_weights(depths: vec4f) -> vec4f {
  let max_depth = max(max(depths.x, depths.y), max(depths.z, depths.w));
  let radius = max(0.75 * ao_params.effect_radius, 1e-6);
  let falloff_range = AO_FALLOFF_RANGE * radius;
  let falloff_from = radius * (1.0 - AO_FALLOFF_RANGE);
  let falloff_mul = -1.0 / falloff_range;
  let falloff_add = falloff_from / falloff_range + 1.0;
  return saturate((vec4f(max_depth) - depths) * falloff_mul + falloff_add);
}

fn ssil_source_normal(q: vec2u, pos: vec3f) -> vec3f {
  let texel = textureLoad(scene_normal, q, 0);
  if (gbuf_receiver_code(texel.a) == 0u) {
    return vec3f(0.0);
  }
  let raw = texel.rgb * 2.0 - 1.0;
  let len = length(raw);
  let n = camera_gx_normal_fwd(select(vec3f(0.0, 0.0, 1.0), raw / len, len > 1e-6));
  return select(n, -n, dot(n, -pos) < 0.0);
}

fn ssil_bilinear(f: vec2f) -> vec4f {
  return vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
}

fn ssil_reprojected(q: vec2u, full: vec2u, size: vec2u, d: u32, z: f32) -> vec4f {
  let m = textureLoad(scene_motion, q, 0).xy;
  let uv = (vec2f(q) + 0.5) / vec2f(full);
  let prev_uv = uv - m * vec2f(0.5, -0.5);
  if (!(all(abs(m) < vec2f(2.0)) && all(prev_uv >= vec2f(0.0)) && all(prev_uv <= vec2f(1.0)))) {
    return vec4f(0.0);
  }
  let position = (prev_uv * vec2f(full) - f32(d / 2u) - 0.5) / f32(d);
  let base = vec2i(floor(position));
  let last = vec2i(size) - 1;
  var log_depths = vec4f(0.0);
  for (var i = 0; i < 4; i++) {
    log_depths[i] = textureLoad(history_in, clamp(base + vec2i(i & 1, i >> 1u), vec2i(0), last), 0).b;
  }
  let wd = dot(ssil_prev_depth_weights(position - vec2f(base), log_depths, z, ao_params.depth_reject), vec4f(1.0));
  let dims = textureDimensions(radiance_in);
  let r_position = prev_uv * vec2f(dims) - 0.5;
  let r_base = vec2i(floor(r_position));
  let bilinear = ssil_bilinear(r_position - vec2f(r_base));
  var sum = vec3f(0.0);
  var c0 = 0.0;
  for (var i = 0; i < 4; i++) {
    let tap = textureLoad(radiance_in, clamp(r_base + vec2i(i & 1, i >> 1u), vec2i(0), vec2i(dims) - 1), 0);
    let w = select(0.0, bilinear[i], tap.a >= 0.5);
    sum += w * tap.rgb;
    c0 += w;
  }
  let c = c0 * min(wd, 1.0);
  return select(vec4f(0.0), vec4f(ssil_clamp_radiance(sum / c0) * c, c), c > 0.0);
}

@compute @workgroup_size(8, 8)
fn reproject(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(radiance_out);
  if (any(id.xy >= size)) {
    return;
  }
  let full = textureDimensions(scene_normal);
  let d = ao_divisor(full, size);
  let q = ao_rep_texel(id.xy, d, full);
  let z = textureLoad(ao_hiz, vec2i(id.xy), 0).r;
  if (ao_is_sky(z)) {
    textureStore(radiance_out, id.xy, vec4f(0.0));
    textureStore(normal_out, id.xy, vec4f(ssil_encode_normal(vec3f(0.0)), 0.0));
    return;
  }
  let pos = camera_view_position_fwd(ao_params.camera, (vec2f(q) + 0.5) / vec2f(full), z, vec2f(full));
  textureStore(normal_out, id.xy, vec4f(ssil_encode_normal(ssil_source_normal(q, pos)), 0.0));
//#if SSIL_FURNACE
  textureStore(radiance_out, id.xy, vec4f(1.0));
//#else
//#if SSIL_RESET
  textureStore(radiance_out, id.xy, vec4f(0.0));
//#else
  textureStore(radiance_out, id.xy, ssil_reprojected(q, full, size, d, z));
//#endif
//#endif
}

@compute @workgroup_size(8, 8)
fn filter_mip(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(radiance_out);
  if (any(id.xy >= size)) {
    return;
  }
  let last = textureDimensions(radiance_in) - 1u;
  let base = id.xy * 2u;
  var taps = array<vec2u, 4>(min(base, last), min(base + vec2u(1u, 0u), last), min(base + vec2u(0u, 1u), last),
                             min(base + vec2u(1u, 1u), last));
  var depths = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    depths[i] = textureLoad(ao_hiz, taps[i], 0).r;
  }
  // Built on ao.hiz's grid and weights, so each sample's colour and facing come from the surface its depth came from
  let w = ssil_depth_mip_weights(depths);
  var radiance = vec4f(0.0);
  var normal = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    radiance += w[i] * textureLoad(radiance_in, taps[i], 0);
    normal += w[i] * ssil_decode_normal(textureLoad(history_in, taps[i], 0));
  }
  let total = w.x + w.y + w.z + w.w;
  textureStore(radiance_out, id.xy, radiance / total);
  textureStore(normal_out, id.xy, vec4f(ssil_encode_normal(normal / total), 0.0));
}

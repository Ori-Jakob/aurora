//#import aurora/camera
//#import aurora/gbuffer
//#import aurora/lighting
//#import aurora/openpbr
//#import aurora/sky
//#import aurora/srgb
//#import aurora/surface

struct DeferredParams {
  camera: StageCamera,
  ambient: vec3f,
  flags: u32,
  view_to_world: ViewRotation,
  reserved: vec4u,
}

const DEFERRED_FLAG_SURFACE_AMBIENT: u32 = 1u;
const DEFERRED_MODE_FULL: u32 = 0u;
const DEFERRED_MODE_GREY: u32 = 1u;
const DEFERRED_MODE_DIRECT: u32 = 2u;
const DEFERRED_MODE_INDIRECT: u32 = 3u;
const DEFERRED_MODE_FURNACE: u32 = 4u;
const DEFERRED_MODE_FURNACE_METAL: u32 = 5u;
const DEFERRED_HDR_MAX: f32 = 65504.0;

@group(0) @binding(1) var<uniform> deferred_params: DeferredParams;
@group(0) @binding(2) var linear_clamp: sampler;
@group(1) @binding(0) var surface_rec: texture_2d<u32>;
@group(1) @binding(1) var scene_normal: texture_2d<f32>;
@group(1) @binding(2) var scene_depth: texture_2d<f32>;
@group(1) @binding(3) var ao_final: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> light_list: LightList;
@group(1) @binding(5) var<storage, read> light_grid: array<u32>;
@group(1) @binding(6) var<storage, read> light_classes: array<ClassParams>;
@group(1) @binding(7) var light_lut3d: texture_3d<f32>;
@group(1) @binding(8) var light_lut2d: texture_2d_array<f32>;
@group(1) @binding(9) var scene_hdr: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var surface_ext: texture_2d<u32>;
@group(1) @binding(11) var<storage, read> surface_table: array<vec4u>;
@group(1) @binding(12) var sky_probe: texture_2d<f32>;
@group(1) @binding(13) var<uniform> sky_sh9: SkySh9;

var<workgroup> deferred_union: array<atomic<u32>, 8>;

// M13 (sun VSM), M14 (local VSM and RT) and M15 replace this
fn deferred_light_visibility(l: GpuLight, p: vec3f, n: vec3f) -> f32 {
  return 1.0;
}

// M11 ssil.irr and M21 gi.irr replace this in their ranges; nothing is ever added on top of it
fn deferred_indirect_diffuse(a: vec3f, n: vec3f, baked: bool) -> vec3f {
  // The pixel's own legacy ambient sets the level and the probe adds direction, so actors and rooms keep their ambient
  if (!sky_sh9_valid(sky_sh9)) {
    return OPENPBR_PI * a;
  }
  return sky_env_irradiance(sky_sh9, camera_world_dir(deferred_params.view_to_world, n), a, baked);
}

fn deferred_indirect_specular(a: vec3f, n: vec3f, v: vec3f, roughness: f32) -> vec3f {
  if (!sky_sh9_valid(sky_sh9)) {
    return a;
  }
  let r = camera_world_dir(deferred_params.view_to_world, reflect(-v, n));
  return sky_env_radiance(sky_probe, linear_clamp, sky_sh9, r, roughness, a);
}

fn deferred_texel(t: texture_2d<f32>, p: vec2u) -> vec4f {
  return textureLoad(t, min(p, textureDimensions(t) - 1u), 0);
}

// Interpolated and mapped normals can face away at silhouettes; bending keeps those pixels lit instead of dropping to Sb
fn deferred_bend_normal(raw: vec3f, v: vec3f) -> vec3f {
  let length0 = length(raw);
  var n = select(v, raw / length0, length0 >= 1e-3);
  let nv = dot(n, v);
  if (nv < 1e-3) {
    let bent = n + v * (1e-3 - nv);
    let bent_length = length(bent);
    n = select(v, bent / bent_length, bent_length >= 1e-6);
  }
  return n;
}

fn deferred_bad(h: vec3f) -> bool {
  return any((bitcast<vec3u>(h) & vec3u(0x7fffffffu)) >= vec3u(0x7f800000u));
}

fn deferred_add_light(h: ptr<function, vec3f>, p: OpenPbrPrepared, l: GpuLight, to_light: vec3f, atten: f32,
                      pos: vec3f) {
  let specular = (light_flags(l) & LIGHT_FLAG_NO_SPECULAR) == 0u;
  let f = openpbr_eval_light(p, to_light, specular, light_lut3d, light_lut2d, linear_clamp);
  *h += l.v2.rgb * (f * atten * deferred_light_visibility(l, pos, p.n));
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32) {
  if (local < LIGHT_MASK_WORDS) {
    atomicStore(&deferred_union[local], 0u);
  }
  workgroupBarrier();
  let dims = textureDimensions(scene_hdr);
  let inside = all(id.xy < dims);
  let px = min(id.xy, dims - 1u);
  let cam = deferred_params.camera;
  let rec = gbuf_surface_fetch(surface_rec, px);
  let entry = surface_table[gbuf_surface_entry_index(gbuf_surface_slot(rec), arrayLength(&surface_table))];
  let d = deferred_texel(scene_depth, px).r;
  let cls = gbuf_surface_class(entry);
  let flags = gbuf_surface_flags(entry);
  let lit = inside && gbuf_surface_is_pbr(rec) && cls != 0u && (flags & GBUF_SURFACE_FLAG_UNLIT) == 0u && d > 0.0 &&
            (cam.flags & CAMERA_VALID) != 0u;
  let header = light_list.header;
  let counts = light_counts(header);
  let z = camera_view_depth(cam, d);
  // Each group walks the union of its pixels' clusters, so one pixel's lights never depend on its neighbours
  if (lit) {
    let base = light_cluster(px, dims, z, header) * LIGHT_MASK_WORDS;
    for (var w = 0u; w < counts.words; w++) {
      atomicOr(&deferred_union[w], light_grid[base + w]);
    }
  }
  workgroupBarrier();
  var union_words = array<u32, 8>();
  for (var w = 0u; w < LIGHT_MASK_WORDS; w++) {
    union_words[w] = atomicLoad(&deferred_union[w]);
  }
  if (!inside) {
    return;
  }
  if (!lit) {
    textureStore(scene_hdr, px, vec4f(0.0));
    return;
  }
  let mode = (deferred_params.flags >> 1u) & 7u;
  let furnace = mode == DEFERRED_MODE_FURNACE || mode == DEFERRED_MODE_FURNACE_METAL;
  let full = vec2f(dims);
  let pos = camera_view_position_fwd(cam, (vec2f(px) + 0.5) / full, z, full);
  let v = normalize(-pos);
  let normal_texel = deferred_texel(scene_normal, px);
  let n = deferred_bend_normal(camera_gx_normal_fwd(normal_texel.rgb * 2.0 - 1.0), v);
  var ext = vec4u(0u);
  if (gbuf_surface_reads_ext(entry)) {
    ext = gbuf_surface_fetch(surface_ext, px);
  }
  let s = gbuf_surface_decode(rec, entry, ext);
  var surface = OpenPbrSurface(srgb_to_linear(s.albedo), s.roughness, s.metalness);
  if (mode == DEFERRED_MODE_GREY) {
    surface.base_color = vec3f(0.5);
  }
  if (furnace) {
    surface.base_color = vec3f(1.0);
    surface.metalness = select(0.0, 1.0, mode == DEFERRED_MODE_FURNACE_METAL);
  }
  let row = light_classes[min(cls, arrayLength(&light_classes) - 1u)];
  let p = openpbr_prepare(surface, row, n, v, light_lut3d, light_lut2d, linear_clamp);
  // The AO-off dummy is 1x1, so the load clamps
  var ao = 1.0;
  if (gbuf_receiver_code(normal_texel.a) >= 1u) {
    ao = pow(deferred_texel(ao_final, px).r, 2.2);
  }
  ao = min(ao, s.ao);
  var a = select(deferred_params.ambient, srgb_to_linear(s.ambient),
                 (deferred_params.flags & DEFERRED_FLAG_SURFACE_AMBIENT) != 0u);
  if (furnace) {
    a = vec3f(1.0);
    ao = 1.0;
  }
  var h = vec3f(0.0);
  if (mode != DEFERRED_MODE_DIRECT) {
    let baked = (flags & GBUF_SURFACE_FLAG_BAKED_LIGHT) != 0u;
    h = openpbr_eval_ambient(p, deferred_indirect_diffuse(a, n, baked),
                             deferred_indirect_specular(a, n, v, s.roughness)) * ao;
  }
  if (mode != DEFERRED_MODE_INDIRECT && !furnace) {
    let thin = (flags & GBUF_SURFACE_FLAG_THIN_TRANSLUCENT) != 0u;
    let cat = s.category;
    // The category test covers directionals too: the camera fill is a masked directional
    for (var i = 0u; i < counts.directional; i++) {
      let l = light_list.lights[i];
      if (!light_category_match(l, cat)) {
        continue;
      }
      let to_light = -l.v1.xyz;
      if (dot(n, to_light) <= 0.0 && !thin) {
        continue;
      }
      deferred_add_light(&h, p, l, to_light, 1.0, pos);
    }
    for (var w = 0u; w < counts.words; w++) {
      var bits = union_words[w];
      if (w == counts.locals / 32u) {
        bits &= (1u << (counts.locals % 32u)) - 1u;
      }
      while (bits != 0u) {
        let b = firstTrailingBit(bits);
        bits &= bits - 1u;
        let l = light_list.lights[counts.directional + w * 32u + b];
        let to = l.v0.xyz - pos;
        let d2 = dot(to, to);
        if (d2 >= l.v0.w * l.v0.w) {
          continue;
        }
        let dist = sqrt(d2);
        let to_light = select(n, to / dist, dist > 0.0);
        let spot = light_spot_attenuation(l, to_light);
        if (spot <= 0.0 || !light_category_match(l, cat)) {
          continue;
        }
        let atten = light_distance_attenuation(l, dist) * spot;
        if (atten <= 0.0 || (dot(n, to_light) <= 0.0 && !thin)) {
          continue;
        }
        deferred_add_light(&h, p, l, to_light, atten, pos);
      }
    }
  }
  if (mode == DEFERRED_MODE_FULL) {
    h += s.emissive * row.emission.rgb * row.emission.w;
  }
  if (deferred_bad(h)) {
    textureStore(scene_hdr, px, vec4f(0.0));
    return;
  }
  textureStore(scene_hdr, px, vec4f(clamp(h, vec3f(0.0), vec3f(DEFERRED_HDR_MAX)), 1.0));
}

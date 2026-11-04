//#import aurora/composite

struct CompositeParams {
  flags: u32,
  legacy_blend: f32,
  pad0: u32,
  pad1: u32,
}

struct CompositeExposure {
  e: f32,
  inv_e: f32,
  ratio: f32,
  one: f32,
}

//#if COMPOSITE_RADIANCE
struct CompositeOut {
  @location(0) color: vec4f,
  @location(1) radiance: vec4f,
}

//#endif
@group(0) @binding(1) var<uniform> composite_params: CompositeParams;

@group(1) @binding(0) var legacy_tex: texture_2d<f32>;
@group(1) @binding(1) var fog_record: texture_2d<u32>;
@group(1) @binding(2) var scene_normal: texture_2d<f32>;
@group(1) @binding(3) var ao_final: texture_2d<f32>;
@group(1) @binding(4) var surface_tex: texture_2d<u32>;
@group(1) @binding(5) var scene_hdr: texture_2d<f32>;
@group(1) @binding(6) var<uniform> composite_exposure: CompositeExposure;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
//#if COMPOSITE_RADIANCE
fn fs_main(@builtin(position) pos: vec4f) -> CompositeOut {
//#else
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
//#endif
  // Every input is read at this pixel, so the composite is exact under jitter and at any render size
  let p = vec2i(pos.xy);
  var t = composite_identity_terms();
  t.ao = composite_ao_term(ao_final, scene_normal, p, (composite_params.flags & COMPOSITE_FLAG_IDENTITY) != 0u);
  if ((composite_params.flags & (COMPOSITE_FLAG_PBR | COMPOSITE_FLAG_IDENTITY)) == COMPOSITE_FLAG_PBR) {
    let hdr = textureLoad(scene_hdr, composite_texel(p, textureDimensions(scene_hdr)), 0);
    // H already carries AO on indirect light; alpha 0 (dummy, unlit, class 0) keeps the legacy line
    if (hdr.a >= 0.5) {
      t.pbr = true;
      t.lb = composite_params.legacy_blend;
      t.h = hdr.rgb;
      t.e = composite_exposure.e;
      t.agx = (composite_params.flags & COMPOSITE_FLAG_TONEMAP_AGX) != 0u;
    }
  }
//#if !COMPOSITE_RADIANCE
  if (composite_is_identity(t)) {
    discard;
  }
//#endif
  var rec = gbuf_fog_decode(textureLoad(fog_record, p, 0));
  if ((composite_params.flags & COMPOSITE_FLAG_NO_FOG) != 0u) {
    rec.q = 0.0;
  }
//#if COMPOSITE_RADIANCE
  let l = textureLoad(legacy_tex, p, 0).rgb;
  let e = composite_eval(l, rec, t);
  let units = composite_units(composite_params.flags, composite_exposure.e, composite_exposure.inv_e);
  // Identity and bad pixels rewrite L, which round-trips exactly, so the history covers every pixel
  return CompositeOut(vec4f(select(e.rgb, l, e.keep), 0.0), composite_radiance(e, rec, t, units.y));
//#else
  let e = composite_eval(textureLoad(legacy_tex, p, 0).rgb, rec, t);
  if (e.keep) {
    discard;
  }
  return vec4f(e.rgb, 0.0);
//#endif
}

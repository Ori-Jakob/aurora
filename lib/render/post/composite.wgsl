//#import aurora/composite

struct CompositeParams {
  flags: u32,
  legacy_blend: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(1) var<uniform> composite_params: CompositeParams;

@group(1) @binding(0) var legacy_tex: texture_2d<f32>;
@group(1) @binding(1) var fog_record: texture_2d<u32>;
@group(1) @binding(2) var scene_normal: texture_2d<f32>;
@group(1) @binding(3) var ao_final: texture_2d<f32>;
@group(1) @binding(4) var surface_tex: texture_2d<u32>;
@group(1) @binding(5) var scene_hdr: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // Every input is read at this pixel, so the composite is exact under jitter and at any render size
  let p = vec2i(pos.xy);
  var t = composite_identity_terms();
  var a = textureLoad(ao_final, composite_texel(p, textureDimensions(ao_final)), 0).r;
  if ((composite_params.flags & COMPOSITE_FLAG_IDENTITY) != 0u) {
    a = 1.0;
  }
  if (a < 1.0 && gbuf_receiver_code(textureLoad(scene_normal, p, 0).a) >= 1u) {
    t.ao = a;
  }
  if ((composite_params.flags & (COMPOSITE_FLAG_PBR | COMPOSITE_FLAG_IDENTITY)) == COMPOSITE_FLAG_PBR) {
    let hdr = textureLoad(scene_hdr, composite_texel(p, textureDimensions(scene_hdr)), 0);
    // H already carries AO on indirect light; alpha 0 (dummy, unlit, class 0) keeps the legacy line
    if (hdr.a >= 0.5) {
      t.pbr = true;
      t.lb = composite_params.legacy_blend;
      t.h = hdr.rgb;
    }
  }
  if (composite_is_identity(t)) {
    discard;
  }
  var rec = gbuf_fog_decode(textureLoad(fog_record, p, 0));
  if ((composite_params.flags & COMPOSITE_FLAG_NO_FOG) != 0u) {
    rec.q = 0.0;
  }
  let e = composite_eval(textureLoad(legacy_tex, p, 0).rgb, rec, t);
  if (e.keep) {
    discard;
  }
  return vec4f(e.rgb, 0.0);
}

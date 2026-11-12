//#import aurora/composite
//#import aurora/gbuffer
//#import aurora/lsm

const COUNT_FLAG_SUN: u32 = 1u;
const COUNT_THRESHOLD: f32 = 0.99;

struct CountParams {
  flags: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  sun: CompositeSun,
}

@group(0) @binding(1) var<uniform> count_params: CountParams;
@group(1) @binding(0) var lsm_mask: texture_2d<f32>;
@group(1) @binding(1) var vis_sun: texture_2d<f32>;
@group(1) @binding(2) var scene_hdr: texture_2d<f32>;
@group(1) @binding(3) var gx_normal: texture_2d<f32>;
@group(1) @binding(4) var fog_record: texture_2d<u32>;
@group(1) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 4>;

var<workgroup> count_local: array<atomic<u32>, 4>;

@compute @workgroup_size(8, 8)
fn count(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < 4u) {
    atomicStore(&count_local[li], 0u);
  }
  workgroupBarrier();
  let dims = textureDimensions(gx_normal);
  let p = vec2i(id.xy);
  let normal = textureLoad(gx_normal, min(p, vec2i(dims) - vec2i(1)), 0);
  let code = gbuf_receiver_code(normal.a);
  let hdr = textureLoad(scene_hdr, composite_texel(p, textureDimensions(scene_hdr)), 0);
  let pbr = hdr.a >= 0.5;
  // Post-shadow PBR pixels drew after the fakes, so no kept fake ever reached them
  let counted = all(id.xy < dims) && code >= 2u && (!pbr || lsm_pre_shadow(hdr.a));
  if (counted) {
    let rec = gbuf_fog_decode(textureLoad(fog_record, p, 0));
    let m = lsm_term(textureLoad(lsm_mask, composite_texel(p, textureDimensions(lsm_mask)), 0).rgb, rec.q, rec.fc);
    var v = 1.0;
    if ((count_params.flags & COUNT_FLAG_SUN) != 0u) {
      let vis = textureLoad(vis_sun, composite_texel(p, textureDimensions(vis_sun)), 0).rg;
      // H takes V on the key light and Sb takes Rs, so each pixel type counts the shadow it shows
      v = select(composite_sun_term(vis_sun, gx_normal, p, count_params.sun), vis.x * vis.y, pbr);
    }
    let touched = any(m < vec3f(COUNT_THRESHOLD));
    let shadowed = v < COUNT_THRESHOLD;
    if (touched && shadowed) {
      atomicAdd(&count_local[select(1u, 0u, pbr)], 1u);
    }
    if (touched) {
      atomicAdd(&count_local[2], 1u);
    }
    if (shadowed) {
      atomicAdd(&count_local[3], 1u);
    }
  }
  workgroupBarrier();
  if (li < 4u) {
    let n = atomicLoad(&count_local[li]);
    if (n != 0u) {
      atomicAdd(&counts[li], n);
    }
  }
}

//#import aurora/retained
//#import aurora/vsm
//#import shadows/vsm_caster

//#if VSM_CLIP_DISTANCES
enable clip_distances;
//#endif

@group(0) @binding(0) var<storage, read> vsm_vertices: array<u32>;
@group(0) @binding(1) var<storage, read> vsm_indices: array<u32>;
@group(0) @binding(2) var<storage, read> vsm_instances: array<u32>;
@group(0) @binding(3) var<storage, read> vsm_clusters: array<u32>;
@group(0) @binding(4) var<storage, read> vsm_render_list: array<u32>;
@group(0) @binding(5) var<storage, read> vsm_pairs: array<u32>;
@group(0) @binding(6) var<uniform> vsm_views_u: array<vec4u, 144>;
@group(0) @binding(7) var<storage, read> vsm_materials: array<u32>;
@group(0) @binding(8) var vsm_atlas: texture_2d<f32>;
@group(0) @binding(9) var vsm_atlas_sampler: sampler;

const VSM_RASTER_NO_VERTEX: u32 = 0xFFFFFFFFu;

struct VsmRasterVarying {
  @builtin(position) position: vec4f,
//#if VSM_CLIP_DISTANCES
  @builtin(clip_distances) clip: array<f32, 4>,
//#endif
  @location(0) @interpolate(flat) rect: vec4f,
//#if VSM_CUTOUT
  @location(1) alpha_uv: vec2f,
  @location(2) @interpolate(flat) material: u32,
//#endif
}

struct VsmRasterFragment {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) rect: vec4f,
//#if VSM_CUTOUT
  @location(1) alpha_uv: vec2f,
  @location(2) @interpolate(flat) material: u32,
//#endif
}

fn vsm_raster_hidden() -> VsmRasterVarying {
  var out: VsmRasterVarying;
  out.position = vec4f(2.0, 2.0, 0.0, 1.0);
//#if VSM_CLIP_DISTANCES
  out.clip = array<f32, 4>(-1.0, -1.0, -1.0, -1.0);
//#endif
  out.rect = vec4f(0.0);
//#if VSM_CUTOUT
  out.alpha_uv = vec2f(0.0);
  out.material = RETAINED_NO_MATERIAL;
//#endif
  return out;
}

fn vsm_raster_view(i: u32) -> VsmView {
  let base = VSM_HEADER_WORDS / 4u + min(i, VSM_MAX_VIEWS - 1u) * (VSM_VIEW_WORDS / 4u);
  var w: array<vec4u, 8>;
  for (var k = 0u; k < 8u; k++) {
    w[k] = vsm_views_u[base + k];
  }
  return vsm_view_unpack(w);
}

fn vsm_raster_local(slot: u32, pair: VsmPair, vi: u32) -> u32 {
  let t = vi / 3u;
  let corner = vi % 3u;
  if (pair.chunk) {
    let tri = pair.item * VSM_CHUNK_TRIANGLES + t;
    if (tri >= retained_record_word(&vsm_instances, slot, RETAINED_REC_INDEX_COUNT) / 3u) {
      return VSM_RASTER_NO_VERTEX;
    }
    let w = retained_record_word(&vsm_instances, slot, RETAINED_REC_INDEX_FIRST) + 3u * tri + corner;
    if (w >= arrayLength(&vsm_indices)) {
      return VSM_RASTER_NO_VERTEX;
    }
    return vsm_indices[w];
  }
  let range = retained_cluster_range(&vsm_instances, slot);
  let index = pair.item;
  if (index < range.x || index - range.x >= range.y ||
      (index + 1u) * RETAINED_CLUSTER_WORDS > arrayLength(&vsm_clusters)) {
    return VSM_RASTER_NO_VERTEX;
  }
  let c = retained_cluster(&vsm_clusters, index);
  if (t >= c.triangle_count || c.tri_word + t >= arrayLength(&vsm_clusters)) {
    return VSM_RASTER_NO_VERTEX;
  }
  let ref_word = c.ref_word + retained_cluster_triangle(&vsm_clusters, c, t)[corner];
  if (ref_word >= arrayLength(&vsm_clusters)) {
    return VSM_RASTER_NO_VERTEX;
  }
  return vsm_clusters[ref_word];
}

@vertex
fn vs_caster(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsmRasterVarying {
  // Release builds turn robustness off, so an index past a pool would read another resource
  if (VSM_PAIRS_CUTOUT_BASE >= arrayLength(&vsm_pairs)) {
    return vsm_raster_hidden();
  }
//#if VSM_CUTOUT
  let pair_index = ii + vsm_pairs[VSM_PAIRS_CUTOUT_BASE];
//#else
  let pair_index = ii;
//#endif
  let pw = VSM_PAIRS_HEADER_WORDS + 2u * pair_index;
  if (pw + 1u >= arrayLength(&vsm_pairs)) {
    return vsm_raster_hidden();
  }
  let pair = vsm_pair_unpack(vec2u(vsm_pairs[pw], vsm_pairs[pw + 1u]));
  let page = vsm_render_page(&vsm_render_list, pair.render_index);
  let virtual_page = vsm_render_virtual(page.x);
  let slot = pair.slot;
  if (virtual_page >= VSM_MAX_VIRTUAL_PAGES || page.y >= VSM_MAX_POOL_PAGES ||
      (slot + 1u) * RETAINED_RECORD_WORDS > arrayLength(&vsm_instances)) {
    return vsm_raster_hidden();
  }
  let flags = retained_record_word(&vsm_instances, slot, RETAINED_REC_FLAGS);
  let local = vsm_raster_local(slot, pair, vi);
  let count = retained_record_word(&vsm_instances, slot, RETAINED_REC_VERTEX_COUNT);
  if ((flags & RETAINED_VALID) == 0u || local >= count) {
    return vsm_raster_hidden();
  }
  let base = (retained_record_word(&vsm_instances, slot, RETAINED_REC_VERTEX_FIRST) + local) * RETAINED_VERTEX_WORDS;
  if (base + RETAINED_VERTEX_WORDS > arrayLength(&vsm_vertices)) {
    return vsm_raster_hidden();
  }
  let pos = vec3f(bitcast<f32>(vsm_vertices[base]), bitcast<f32>(vsm_vertices[base + 1u]),
                  bitcast<f32>(vsm_vertices[base + 2u]));
  let anchor_hi = bitcast<vec3f>(vsm_views_u[1].yzw);
  let anchor_lo = bitcast<vec3f>(vsm_views_u[2].xyz);
  let p_anchor = vec4f(pos, 1.0) * retained_world_rel(retained_world(&vsm_instances, slot), anchor_hi, anchor_lo);
  let v = vsm_raster_view(virtual_page / VSM_PAGES_PER_VIEW);
  let pages_x = max(vsm_render_list[VSM_RL_PAGES_X], 1u);
  let rows = max(vsm_render_list[VSM_RL_ROWS], 1u);
  let at = vsm_caster_position(v, vsm_caster_page(v, virtual_page, page.y, pages_x, rows), p_anchor);
  var out: VsmRasterVarying;
  out.position = at.position;
//#if VSM_CLIP_DISTANCES
  out.clip = array<f32, 4>(at.clip.x, at.clip.y, at.clip.z, at.clip.w);
//#endif
  out.rect = at.rect;
//#if VSM_CUTOUT
  var raw: array<u32, 8>;
  for (var k = 0u; k < 8u; k++) {
    raw[k] = vsm_vertices[base + k];
  }
  let vertex = retained_unpack_vertex(raw);
  let uv = retained_uv(retained_uv_xform(&vsm_instances, slot, false), vertex.uv0);
  let alpha_uv = retained_uv(retained_uv_xform(&vsm_instances, slot, true), retained_alpha_uv(vertex, flags));
  out.alpha_uv = select(alpha_uv, uv, (flags & RETAINED_ALPHA_UV_IS_UV0) != 0u);
  out.material = retained_record_word(&vsm_instances, slot, RETAINED_REC_MATERIAL) & 0xFFFFu;
//#endif
  return out;
}

@fragment
fn fs_rect(in: VsmRasterFragment) {
//#if !VSM_CLIP_DISTANCES
  if (any(in.position.xy < in.rect.xy) || any(in.position.xy >= in.rect.zw)) {
    discard;
  }
//#endif
}

//#if VSM_CUTOUT
@fragment
fn fs_cutout(in: VsmRasterFragment) {
//#if !VSM_CLIP_DISTANCES
  if (any(in.position.xy < in.rect.xy) || any(in.position.xy >= in.rect.zw)) {
    discard;
  }
//#endif
  let words = (in.material + 1u) * RETAINED_MATERIAL_WORDS;
  if (in.material == RETAINED_NO_MATERIAL || words > arrayLength(&vsm_materials)) {
    return;
  }
  let m = retained_material(&vsm_materials, in.material);
  let blocked = RETAINED_MAT_ALPHA_NO_UV | RETAINED_MAT_ALPHA_PENDING | RETAINED_MAT_ALPHA_MISSING;
  if (m.alpha_tile == RETAINED_NO_TILE || (m.flags & blocked) != 0u) {
    return;
  }
  let uv = atlas_uv(m.alpha_tile, m.alpha_wrap.x, m.alpha_wrap.y, in.alpha_uv);
  let texel = textureSampleLevel(vsm_atlas, vsm_atlas_sampler, uv, 0.0);
  if (!retained_alpha_pass(m, texel[m.alpha_channel])) {
    discard;
  }
}
//#endif

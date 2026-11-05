//#import aurora/retained
//#import aurora/vsm

fn vsm_cull_hist_box(history: ptr<storage, array<u32>, read>, hb: u32) -> array<vec3f, 2> {
  let w = hb + VSM_HIST_BOX;
  let lo = vec3f(bitcast<f32>((*history)[w]), bitcast<f32>((*history)[w + 1u]), bitcast<f32>((*history)[w + 2u]));
  let hi = vec3f(bitcast<f32>((*history)[w + 3u]), bitcast<f32>((*history)[w + 4u]),
                 bitcast<f32>((*history)[w + 5u]));
  return array<vec3f, 2>(lo, hi);
}

fn vsm_cull_hist_xform(history: ptr<storage, array<u32>, read>, hb: u32) -> mat3x4f {
  return retained_mat3x4(history, hb + VSM_HIST_XFORM);
}

//#if VSM_CHUNK_BOUNDS
@group(1) @binding(0) var<storage, read> vsm_list: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_history: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_indices: array<u32>;
@group(1) @binding(3) var<storage, read> vsm_vertices: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_chunk_boxes: array<u32>;
@group(1) @binding(5) var<storage, read_write> vsm_stats: array<atomic<u32>>;

@compute @workgroup_size(128)
fn vsm_chunk_bounds(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  if (wg.x >= arrayLength(&vsm_list)) {
    return;
  }
  let slot = vsm_list[wg.x];
  let hb = slot * VSM_HIST_WORDS;
  if (slot >= VSM_SLOTS || hb + VSM_HIST_WORDS > arrayLength(&vsm_history)) {
    return;
  }
  let flags = vsm_history[hb + VSM_HIST_FLAGS];
  if ((flags & (VSM_HIST_CHUNKED | VSM_HIST_BOX_VALID)) != (VSM_HIST_CHUNKED | VSM_HIST_BOX_VALID)) {
    return;
  }
  let index_first = vsm_history[hb + VSM_HIST_INDEX_FIRST];
  let index_count = vsm_history[hb + VSM_HIST_INDEX_COUNT];
  let chunks = vsm_chunk_count(index_count);
  if (vsm_chunk_slot(index_first, 0u) == VSM_NO_CHUNK || vsm_chunk_slot(index_first, chunks) > VSM_CHUNK_BOXES) {
    if (lid == 0u) {
      atomicAdd(&vsm_stats[VSM_STAT_CHUNK_MISALIGNED], 1u);
    }
    return;
  }
  let vertex_first = vsm_history[hb + VSM_HIST_VERTEX_FIRST];
  let xform = vsm_cull_hist_xform(&vsm_history, hb);
  let box = vsm_cull_hist_box(&vsm_history, hb);
  for (var c = lid; c < chunks; c += 128u) {
    let first = c * VSM_CHUNK_INDICES;
    let last = min(first + VSM_CHUNK_INDICES, index_count / 3u * 3u);
    var lo = vec3f(3.4e38);
    var hi = vec3f(-3.4e38);
    for (var i = first; i < last; i++) {
      let index = index_first + i;
      if (index >= arrayLength(&vsm_indices)) {
        break;
      }
      let base = (vertex_first + vsm_indices[index]) * RETAINED_VERTEX_WORDS;
      if (base + 2u >= arrayLength(&vsm_vertices)) {
        continue;
      }
      let p = vec3f(bitcast<f32>(vsm_vertices[base]), bitcast<f32>(vsm_vertices[base + 1u]),
                    bitcast<f32>(vsm_vertices[base + 2u]));
      let q = vec4f(p, 1.0) * xform;
      lo = min(lo, q);
      hi = max(hi, q);
    }
    let words = vsm_chunk_box_pack(array<vec3f, 2>(lo, hi), box);
    let w = vsm_chunk_slot(index_first, c) * 2u;
    if (w + 1u < arrayLength(&vsm_chunk_boxes)) {
      vsm_chunk_boxes[w] = words.x;
      vsm_chunk_boxes[w + 1u] = words.y;
    }
  }
  if (lid == 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_CHUNK_BOXES], chunks);
  }
}
//#endif

//#if VSM_CULL_CANDIDATES
@group(1) @binding(0) var<storage, read_write> vsm_history: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_render_list: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_cull_words: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_stats: array<atomic<u32>>;

@compute @workgroup_size(64)
fn vsm_cull_candidates(@builtin(global_invocation_id) id: vec3u) {
  let s = id.x;
  let hb = s * VSM_HIST_WORDS;
  if (s >= min(VSM_SLOTS, arrayLength(&vsm_cull_words)) || hb + VSM_HIST_WORDS > arrayLength(&vsm_history)) {
    return;
  }
  let flags = vsm_history[hb + VSM_HIST_FLAGS];
  var mask = 0u;
  if ((flags & (VSM_HIST_CASTER | VSM_HIST_BOX_VALID)) == (VSM_HIST_CASTER | VSM_HIST_BOX_VALID)) {
    let w = hb + VSM_HIST_BOX;
    let box = array<vec3f, 2>(vec3f(bitcast<f32>(vsm_history[w]), bitcast<f32>(vsm_history[w + 1u]),
                                    bitcast<f32>(vsm_history[w + 2u])),
                              vec3f(bitcast<f32>(vsm_history[w + 3u]), bitcast<f32>(vsm_history[w + 4u]),
                                    bitcast<f32>(vsm_history[w + 5u])));
    let count = min(vsm_views[VSM_HDR_VIEW_COUNT], VSM_MAX_VIEWS);
    for (var i = 0u; i < count; i++) {
      let v = vsm_view(&vsm_views, i);
      if ((v.flags & VSM_VIEW_ACTIVE) == 0u) {
        continue;
      }
      let r = vsm_box_rect(v, box);
      if (r.z < r.x || r.w < r.y) {
        continue;
      }
      let columns = vsm_row_bits(u32(r.x), u32(r.z));
      for (var y = r.y; y <= r.w; y++) {
        let word = VSM_RL_ROW_MASKS + i * VSM_WINDOW_PAGES + u32(y);
        if (word < arrayLength(&vsm_render_list) && (vsm_render_list[word] & columns) != 0u) {
          mask |= 1u << i;
          break;
        }
      }
    }
  }
  vsm_cull_words[s] = mask;
  vsm_history[hb + VSM_HIST_CULL_MASK] = mask;
  if (mask != 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_CULL_SLOTS], 1u);
  }
}
//#endif

//#if VSM_CULL_PAIRS
@group(1) @binding(0) var<storage, read> vsm_list: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_history: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_clusters: array<u32>;
@group(1) @binding(3) var<storage, read> vsm_chunk_boxes: array<u32>;
@group(1) @binding(4) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(5) var<storage, read> vsm_pt: array<u32>;
//#if VSM_CULL_EMIT
@group(1) @binding(6) var<storage, read_write> vsm_offsets: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> vsm_pairs: array<atomic<u32>>;
//#else
@group(1) @binding(6) var<storage, read_write> vsm_counts: array<atomic<u32>>;
//#endif

struct VsmCullItem {
  box: array<vec3f, 2>,
  word: u32,
  chunk: bool,
  posed: bool,
}

fn vsm_cull_item(hb: u32, k: u32, clustered: bool) -> VsmCullItem {
  var item: VsmCullItem;
  item.box = vsm_cull_hist_box(&vsm_history, hb);
  if (clustered) {
    let index = vsm_history[hb + VSM_HIST_CLUSTER_FIRST] + k;
    item.word = index;
    if ((index + 1u) * RETAINED_CLUSTER_WORDS > arrayLength(&vsm_clusters)) {
      return item;
    }
    let c = retained_cluster(&vsm_clusters, index);
    item.posed = c.bounds_space != 0u;
    if (!item.posed) {
      item.box = vsm_box_through(c.aabb_min, c.aabb_max, vsm_cull_hist_xform(&vsm_history, hb));
    }
    return item;
  }
  item.word = VSM_PAIR_CHUNK | k;
  item.chunk = true;
  let w = vsm_chunk_slot(vsm_history[hb + VSM_HIST_INDEX_FIRST], k) * 2u;
  if (w + 1u < arrayLength(&vsm_chunk_boxes)) {
    item.box = vsm_chunk_box(vec2u(vsm_chunk_boxes[w], vsm_chunk_boxes[w + 1u]), item.box);
  }
  return item;
}

var<workgroup> vsm_cull_views: array<VsmView, 16>;

@compute @workgroup_size(256)
fn vsm_cull_pairs(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  if (lid < VSM_MAX_VIEWS) {
    vsm_cull_views[lid] = vsm_view(&vsm_views, lid);
  }
  workgroupBarrier();
  if (wg.x >= arrayLength(&vsm_list)) {
    return;
  }
  let slot = vsm_list[wg.x];
  let hb = slot * VSM_HIST_WORDS;
  if (slot >= VSM_SLOTS || hb + VSM_HIST_WORDS > arrayLength(&vsm_history)) {
    return;
  }
  let flags = vsm_history[hb + VSM_HIST_FLAGS];
  let mask = vsm_history[hb + VSM_HIST_CULL_MASK];
  let cutout = (flags & VSM_HIST_CUTOUT) != 0u;
  let clusters = vsm_history[hb + VSM_HIST_CLUSTER_COUNT] & RETAINED_CLUSTER_COUNT_MASK;
  let items = select(vsm_chunk_count(vsm_history[hb + VSM_HIST_INDEX_COUNT]), clusters, clusters != 0u);
//#if VSM_CULL_EMIT
  let max_pairs = min(atomicLoad(&vsm_pairs[VSM_PAIRS_MAX]), VSM_MAX_PAIRS);
  let total_opaque = atomicLoad(&vsm_pairs[VSM_PAIRS_TOTAL_OPAQUE]);
  let cutout_base = atomicLoad(&vsm_pairs[VSM_PAIRS_CUTOUT_BASE]);
//#endif
  for (var k = lid; k < items; k += 256u) {
    let item = vsm_cull_item(hb, k, clusters != 0u);
    for (var bits = mask; bits != 0u; bits &= bits - 1u) {
      let v = vsm_cull_views[firstTrailingBit(bits)];
      // A depth raster writes any texel whose centre a triangle covers, so only a rect with no centre can be skipped
      let t = vsm_texel_rect(v, item.box);
      if (!vsm_rect_holds_centre(t)) {
        continue;
      }
      let r = vsm_page_rect(v, t);
      for (var y = r.y; y <= r.w; y++) {
        for (var x = r.x; x <= r.z; x++) {
          let index = vsm_page_index(v.table_offset, v.origin + vec2i(x, y));
          if (index >= arrayLength(&vsm_pt)) {
            continue;
          }
          let e = vsm_pt[index];
          if (!vsm_entry_listed(e)) {
            continue;
          }
          let ri = vsm_entry_render_index(e);
//#if VSM_CULL_EMIT
          var at = 0u;
          var end = 0u;
          if (cutout) {
            at = atomicAdd(&vsm_offsets[VSM_MAX_RENDER_PAGES + ri], 1u) - total_opaque + cutout_base;
            end = cutout_base + atomicLoad(&vsm_pairs[VSM_PAIRS_ENDS + VSM_MAX_RENDER_PAGES + ri]);
          } else {
            at = atomicAdd(&vsm_offsets[ri], 1u);
            end = atomicLoad(&vsm_pairs[VSM_PAIRS_ENDS + ri]);
          }
          if (at >= end || at >= max_pairs) {
            atomicAdd(&vsm_pairs[VSM_PAIRS_DROPPED], 1u);
            continue;
          }
          let pair = vsm_pair_pack(item.word, slot, ri);
          atomicStore(&vsm_pairs[VSM_PAIRS_HEADER_WORDS + 2u * at], pair.x);
          atomicStore(&vsm_pairs[VSM_PAIRS_HEADER_WORDS + 2u * at + 1u], pair.y);
          if (item.chunk) {
            atomicAdd(&vsm_pairs[VSM_PAIRS_CHUNK_PAIRS], 1u);
          }
          if (item.posed) {
            atomicAdd(&vsm_pairs[VSM_PAIRS_POSED_PAIRS], 1u);
          }
//#else
          atomicAdd(&vsm_counts[select(0u, VSM_MAX_RENDER_PAGES, cutout) + ri], 1u);
//#endif
        }
      }
    }
  }
}
//#endif

//#if VSM_CULL_SKIP
@group(1) @binding(0) var<storage, read> vsm_counts: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_offsets: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_render_list: array<u32>;
@group(1) @binding(3) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_pt: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> vsm_pairs: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> vsm_raster_args: array<u32>;

fn vsm_skip_word(i: u32) -> u32 {
  return select(0u, vsm_offsets[i], i < arrayLength(&vsm_offsets));
}

@compute @workgroup_size(64)
fn vsm_cull_skip(@builtin(global_invocation_id) id: vec3u) {
  let r = id.x;
  let words = min(arrayLength(&vsm_counts), arrayLength(&vsm_offsets));
  if (r >= VSM_MAX_RENDER_PAGES || 2u * VSM_MAX_RENDER_PAGES > words) {
    return;
  }
  let max_pairs = min(vsm_views[VSM_HDR_MAX_PAIRS], VSM_MAX_PAIRS);
  let total_opaque = vsm_offsets[VSM_MAX_RENDER_PAGES];
  let co = vsm_counts[r];
  let cc = vsm_counts[VSM_MAX_RENDER_PAGES + r];
  let oo = vsm_offsets[r];
  let oc = vsm_offsets[VSM_MAX_RENDER_PAGES + r] - total_opaque;
  atomicStore(&vsm_pairs[VSM_PAIRS_ENDS + r], oo + co);
  atomicStore(&vsm_pairs[VSM_PAIRS_ENDS + VSM_MAX_RENDER_PAGES + r], oc + cc);
  let n = co + cc;
  if (n == 0u || oo + oc + n <= max_pairs) {
    return;
  }
  atomicMax(&vsm_pairs[VSM_PAIRS_SKIP_KEY], VSM_MAX_RENDER_PAGES - r);
  atomicAdd(&vsm_stats[VSM_STAT_SKIPPED_PAGES], 1u);
  let word = vsm_render_page(&vsm_render_list, r).x;
  let v = vsm_render_virtual(word);
  // A page re-rendered only for the check keeps its clean state, so the check never moves allocation
  let check_only = word != VSM_NO_PAGE && (word & VSM_RL_CHECK_ONLY) != 0u;
  if (v < arrayLength(&vsm_pt)) {
    atomicOr(&vsm_pt[v], select(VSM_ENTRY_SKIP | VSM_ENTRY_DIRTY | VSM_ENTRY_PRIORITY, VSM_ENTRY_SKIP, check_only));
  }
}

@compute @workgroup_size(1)
fn vsm_cull_args() {
  let total_opaque = vsm_skip_word(VSM_MAX_RENDER_PAGES);
  let last = 2u * VSM_MAX_RENDER_PAGES - 1u;
  let tail = select(0u, vsm_counts[last], last < arrayLength(&vsm_counts));
  let total_cutout = vsm_skip_word(last) + tail - total_opaque;
  let key = atomicLoad(&vsm_pairs[VSM_PAIRS_SKIP_KEY]);
  var opaque = total_opaque;
  var cutout = total_cutout;
  if (key != 0u) {
    let first = VSM_MAX_RENDER_PAGES - key;
    opaque = vsm_skip_word(first);
    cutout = vsm_skip_word(VSM_MAX_RENDER_PAGES + first) - total_opaque;
  }
  atomicStore(&vsm_pairs[VSM_PAIRS_OPAQUE_COUNT], opaque);
  atomicStore(&vsm_pairs[VSM_PAIRS_CUTOUT_BASE], opaque);
  atomicStore(&vsm_pairs[VSM_PAIRS_CUTOUT_COUNT], cutout);
  atomicStore(&vsm_pairs[VSM_PAIRS_SKIPPED], atomicLoad(&vsm_stats[VSM_STAT_SKIPPED_PAGES]));
  atomicStore(&vsm_pairs[VSM_PAIRS_TOTAL_OPAQUE], total_opaque);
  atomicStore(&vsm_pairs[VSM_PAIRS_MAX], min(vsm_views[VSM_HDR_MAX_PAIRS], VSM_MAX_PAIRS));
  atomicStore(&vsm_stats[VSM_STAT_OPAQUE_PAIRS], opaque);
  atomicStore(&vsm_stats[VSM_STAT_CUTOUT_PAIRS], cutout);
  if (arrayLength(&vsm_raster_args) >= 8u) {
    vsm_raster_args[0] = VSM_VERTICES_PER_PAIR;
    vsm_raster_args[1] = opaque;
    vsm_raster_args[2] = 0u;
    vsm_raster_args[3] = 0u;
    vsm_raster_args[4] = VSM_VERTICES_PER_PAIR;
    vsm_raster_args[5] = cutout;
    vsm_raster_args[6] = 0u;
    vsm_raster_args[7] = 0u;
  }
}
//#endif

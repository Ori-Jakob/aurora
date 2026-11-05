//#import aurora/retained
//#import aurora/vsm

//#if VSM_DIRTY
@group(1) @binding(0) var<storage, read> vsm_list: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_boxes: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_pt: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> vsm_dirty_pages: array<atomic<u32>>;

fn vsm_dirty_box(base: u32) -> array<vec3f, 2> {
  let lo = vec3f(bitcast<f32>(vsm_boxes[base]), bitcast<f32>(vsm_boxes[base + 1u]), bitcast<f32>(vsm_boxes[base + 2u]));
  let hi = vec3f(bitcast<f32>(vsm_boxes[base + 3u]), bitcast<f32>(vsm_boxes[base + 4u]),
                 bitcast<f32>(vsm_boxes[base + 5u]));
  return array<vec3f, 2>(lo, hi);
}

fn vsm_dirty_mark(index: u32, priority: bool) {
  let bit = 1u << (index & 31u);
  let word = index >> 5u;
  if (word >= VSM_DIRTY_PLANE_WORDS || word + VSM_DIRTY_PLANE_WORDS >= arrayLength(&vsm_dirty_pages)) {
    return;
  }
  if ((atomicOr(&vsm_dirty_pages[word], bit) & bit) == 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_FOOTPRINT], 1u);
  }
  if (priority && (atomicOr(&vsm_dirty_pages[VSM_DIRTY_PLANE_WORDS + word], bit) & bit) == 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_FOOTPRINT_PRIORITY], 1u);
  }
  if (index >= arrayLength(&vsm_pt) || (atomicLoad(&vsm_pt[index]) & VSM_ENTRY_ALLOCATED) == 0u) {
    return;
  }
  let prev = atomicOr(&vsm_pt[index], VSM_ENTRY_DIRTY | select(0u, VSM_ENTRY_PRIORITY, priority));
  if ((prev & VSM_ENTRY_DIRTY) == 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_DIRTIED], 1u);
  }
  if (priority && (prev & VSM_ENTRY_PRIORITY) == 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_DIRTIED_PRIORITY], 1u);
  }
}

@compute @workgroup_size(64)
fn vsm_dirty(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  if (wg.x >= arrayLength(&vsm_list)) {
    return;
  }
  let base = vsm_list[wg.x] * VSM_DIRTY_BOX_WORDS;
  if (base + VSM_DIRTY_BOX_WORDS > arrayLength(&vsm_boxes)) {
    return;
  }
  let flags = vsm_boxes[base + VSM_DIRTY_BOX_FLAGS];
  let count = min(vsm_views[VSM_HDR_VIEW_COUNT], VSM_MAX_VIEWS);
  for (var b = 0u; b < 2u; b++) {
    if ((flags & (VSM_DIRTY_HAS_OLD << b)) == 0u) {
      continue;
    }
    let box = vsm_dirty_box(base + select(VSM_DIRTY_BOX_OLD, VSM_DIRTY_BOX_NEW, b == 1u));
    let priority = (flags & (VSM_DIRTY_OLD_PLAYER << b)) != 0u;
    for (var i = 0u; i < count; i++) {
      let v = vsm_view(&vsm_views, i);
      if ((v.flags & VSM_VIEW_ACTIVE) == 0u) {
        continue;
      }
      let r = vsm_box_rect(v, box);
      if (r.z < r.x || r.w < r.y) {
        continue;
      }
      let w = u32(r.z - r.x + 1);
      let pages = w * u32(r.w - r.y + 1);
      for (var p = lid; p < pages; p += 64u) {
        let page = v.origin + vec2i(r.x + i32(p % w), r.y + i32(p / w));
        vsm_dirty_mark(vsm_page_index(v.table_offset, page), priority);
      }
    }
  }
}
//#else
@group(1) @binding(0) var<storage, read> vsm_instances: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_materials: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_history: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> vsm_cast_bits: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> vsm_slot_words: array<u32>;
@group(1) @binding(7) var<storage, read_write> vsm_boxes: array<u32>;

struct VsmScanBox {
  valid: bool,
  lo: vec3f,
  hi: vec3f,
  xform: mat3x4f,
}

fn vsm_scan_box(slot: u32, hi: vec3f, lo: vec3f, pad: f32) -> VsmScanBox {
  var out: VsmScanBox;
  out.xform = retained_world_rel(retained_world(&vsm_instances, slot), hi, lo);
  let base = slot * RETAINED_RECORD_WORDS + RETAINED_REC_CUR_AABB;
  let raw_lo = vec3u(vsm_instances[base], vsm_instances[base + 1u], vsm_instances[base + 2u]);
  let raw_hi = vec3u(vsm_instances[base + 3u], vsm_instances[base + 4u], vsm_instances[base + 5u]);
  // The empty AABB is all-ones then all-zeros in the order-preserving encoding, which decodes to NaN
  out.valid = all(raw_lo <= raw_hi) && all(raw_lo != vec3u(0xFFFFFFFFu)) && all(raw_hi != vec3u(0u));
  if (!out.valid) {
    return out;
  }
  let aabb = retained_aabb(&vsm_instances, slot, false);
  var bmin = vec3f(3.4e38);
  var bmax = vec3f(-3.4e38);
  for (var i = 0u; i < 8u; i++) {
    let corner = select(aabb[0], aabb[1], vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u));
    let p = vec4f(corner, 1.0) * out.xform;
    bmin = min(bmin, p);
    bmax = max(bmax, p);
  }
  out.lo = bmin - vec3f(pad);
  out.hi = bmax + vec3f(pad);
  return out;
}

fn vsm_scan_material(word: u32) -> vec2<bool> {
  let id = word & RETAINED_NO_MATERIAL;
  if (id == RETAINED_NO_MATERIAL || (id + 1u) * RETAINED_MATERIAL_WORDS > arrayLength(&vsm_materials)) {
    return vec2<bool>(false, false);
  }
  let flags = retained_material(&vsm_materials, id).flags;
  let tested = (flags & RETAINED_MAT_ALPHA_TEST) != 0u;
  let pending = (flags & RETAINED_MAT_ALPHA_PENDING) != 0u;
  let missing = (flags & RETAINED_MAT_ALPHA_MISSING) != 0u;
  return vec2<bool>(tested && !pending && !missing, tested && pending);
}

fn vsm_scan_put(base: u32, v: vec3f) {
  vsm_history[base] = bitcast<u32>(v.x);
  vsm_history[base + 1u] = bitcast<u32>(v.y);
  vsm_history[base + 2u] = bitcast<u32>(v.z);
}

fn vsm_scan_row(base: u32, row: vec4f) {
  vsm_history[base] = bitcast<u32>(row.x);
  vsm_history[base + 1u] = bitcast<u32>(row.y);
  vsm_history[base + 2u] = bitcast<u32>(row.z);
  vsm_history[base + 3u] = bitcast<u32>(row.w);
}

fn vsm_scan_box_put(base: u32, lo: vec3f, hi: vec3f) {
  vsm_boxes[base] = bitcast<u32>(lo.x);
  vsm_boxes[base + 1u] = bitcast<u32>(lo.y);
  vsm_boxes[base + 2u] = bitcast<u32>(lo.z);
  vsm_boxes[base + 3u] = bitcast<u32>(hi.x);
  vsm_boxes[base + 4u] = bitcast<u32>(hi.y);
  vsm_boxes[base + 5u] = bitcast<u32>(hi.z);
}

fn vsm_scan_count(stat: u32, on: bool) {
  if (on) {
    atomicAdd(&vsm_stats[stat], 1u);
  }
}

@compute @workgroup_size(64)
fn vsm_scan(@builtin(global_invocation_id) id: vec3u) {
  let s = id.x;
  if (s >= arrayLength(&vsm_slot_words)) {
    return;
  }
  vsm_slot_words[s] = 0u;
  let rec = s * RETAINED_RECORD_WORDS;
  let hb = s * VSM_HIST_WORDS;
  if (s >= VSM_SLOTS || rec + RETAINED_RECORD_WORDS > arrayLength(&vsm_instances) ||
      hb + VSM_HIST_WORDS > arrayLength(&vsm_history)) {
    return;
  }
  let header_flags = vsm_views[VSM_HDR_FLAGS];
  let full = (header_flags & VSM_FULL) != 0u;
  let w0 = vsm_instances[rec + RETAINED_REC_FLAGS];
  let w1 = vsm_instances[rec + RETAINED_REC_CATEGORIES];
  let w2 = vsm_instances[rec + RETAINED_REC_OBJECT_FLAGS];
  let hash = vsm_instances[rec + RETAINED_REC_IDENTITY_HASH];
  let world_frame = vsm_instances[rec + RETAINED_REC_WORLD_FRAME];
  let uv_frame = vsm_instances[rec + RETAINED_REC_UV_FRAME];
  let material_frame = vsm_instances[rec + RETAINED_REC_MATERIAL_FRAME];
  let stream_frame = vsm_instances[rec + RETAINED_REC_STREAM_FRAME];
  let cluster_count = vsm_instances[rec + RETAINED_REC_CLUSTER_COUNT];
  let hflags = vsm_history[hb + VSM_HIST_FLAGS];

  let valid = (w0 & RETAINED_VALID) != 0u;
  let caster_class = vsm_is_caster(w0, w1);
  let same_stream = stream_frame == vsm_history[hb + VSM_HIST_STREAM_FRAME];
  let occupant = (hflags & VSM_HIST_SEEN) != 0u && valid &&
                 ((hflags & VSM_HIST_VALID) == 0u || hash != vsm_history[hb + VSM_HIST_HASH]);
  let pending = valid && same_stream && (occupant || (hflags & VSM_HIST_PENDING_STREAM) != 0u);
  let cast_now = caster_class && !pending;
  let was_caster = (hflags & VSM_HIST_CASTER) != 0u;
  let material = vsm_scan_material(vsm_instances[rec + RETAINED_REC_MATERIAL]);
  let cutout = material.x;
  let alpha_pending = material.y;
  let was_cutout = (hflags & VSM_HIST_CUTOUT) != 0u;
  let player = (w1 & VSM_CAT_PLAYER) != 0u;
  let h = vsm_header(&vsm_views);
  let box = vsm_scan_box(s, h.anchor_hi, h.anchor_lo, h.texel0);

  var reasons = 0u;
  reasons |= select(0u, VSM_REASON_CASTER, cast_now != was_caster);
  reasons |= select(0u, VSM_REASON_HASH, hash != vsm_history[hb + VSM_HIST_HASH]);
  reasons |= select(0u, VSM_REASON_STREAM, !same_stream);
  reasons |= select(0u, VSM_REASON_WORLD, world_frame != vsm_history[hb + VSM_HIST_WORLD_FRAME]);
  reasons |= select(0u, VSM_REASON_MATERIAL,
                    material_frame != vsm_history[hb + VSM_HIST_MATERIAL_FRAME] && (was_cutout || cutout));
  reasons |= select(0u, VSM_REASON_UV, uv_frame != vsm_history[hb + VSM_HIST_UV_FRAME] && cutout);
  reasons |= select(0u, VSM_REASON_ALPHA, (hflags & VSM_HIST_ALPHA_PENDING) != 0u && !alpha_pending);
  reasons |= select(0u, VSM_REASON_CUTOUT, was_cutout != cutout);
  reasons |= select(0u, VSM_REASON_ARRAYS, cast_now && (w0 & RETAINED_GPU_ARRAYS) != 0u);
  // Frees zero only word 0, so a freed slot keeps its old AABB and the Valid bit is what stops it casting
  let changed = !full && (cast_now || was_caster) && reasons != 0u;
  let chunked = cast_now && (cluster_count & RETAINED_CLUSTER_COUNT_MASK) == 0u;
  let chunk_need = chunked && (changed || full);

  if (changed || chunk_need) {
    let db = s * VSM_DIRTY_BOX_WORDS;
    var box_flags = 0u;
    if (changed && was_caster && (hflags & VSM_HIST_BOX_VALID) != 0u) {
      for (var k = 0u; k < 6u; k++) {
        vsm_boxes[db + VSM_DIRTY_BOX_OLD + k] = vsm_history[hb + VSM_HIST_BOX + k];
      }
      box_flags |= VSM_DIRTY_HAS_OLD | select(0u, VSM_DIRTY_OLD_PLAYER, (hflags & VSM_HIST_PLAYER) != 0u);
    }
    if (changed && cast_now && box.valid) {
      vsm_scan_box_put(db + VSM_DIRTY_BOX_NEW, box.lo, box.hi);
      box_flags |= VSM_DIRTY_HAS_NEW | select(0u, VSM_DIRTY_NEW_PLAYER, player);
    }
    vsm_boxes[db + VSM_DIRTY_BOX_FLAGS] = box_flags;
    vsm_slot_words[s] = select(0u, VSM_SLOT_CHANGED, changed) | select(0u, VSM_SLOT_CHUNK_NEED, chunk_need);
  }

  var nf = VSM_HIST_SEEN;
  nf |= select(0u, VSM_HIST_VALID, valid);
  nf |= select(0u, VSM_HIST_CASTER, cast_now);
  nf |= select(0u, VSM_HIST_CUTOUT, cutout);
  nf |= select(0u, VSM_HIST_ALPHA_PENDING, alpha_pending);
  nf |= select(0u, VSM_HIST_PLAYER, player);
  nf |= select(0u, VSM_HIST_CHUNKED, chunked);
  nf |= select(0u, VSM_HIST_PENDING_STREAM, pending);
  nf |= select(0u, VSM_HIST_BOX_VALID, box.valid);
  vsm_history[hb + VSM_HIST_FLAGS] = nf;
  vsm_history[hb + VSM_HIST_HASH] = hash;
  vsm_history[hb + VSM_HIST_STREAM_FRAME] = stream_frame;
  vsm_history[hb + VSM_HIST_WORLD_FRAME] = world_frame;
  vsm_history[hb + VSM_HIST_MATERIAL_FRAME] = material_frame;
  vsm_history[hb + VSM_HIST_UV_FRAME] = uv_frame;
  vsm_history[hb + VSM_HIST_CLUSTER_FIRST] = vsm_instances[rec + RETAINED_REC_CLUSTER_FIRST];
  vsm_history[hb + VSM_HIST_CLUSTER_COUNT] = cluster_count;
  vsm_history[hb + VSM_HIST_INDEX_FIRST] = vsm_instances[rec + RETAINED_REC_INDEX_FIRST];
  vsm_history[hb + VSM_HIST_INDEX_COUNT] = vsm_instances[rec + RETAINED_REC_INDEX_COUNT];
  vsm_scan_put(hb + VSM_HIST_BOX, select(vec3f(0.0), box.lo, box.valid));
  vsm_scan_put(hb + VSM_HIST_BOX + 3u, select(vec3f(0.0), box.hi, box.valid));
  vsm_scan_row(hb + VSM_HIST_XFORM, box.xform[0]);
  vsm_scan_row(hb + VSM_HIST_XFORM + 4u, box.xform[1]);
  vsm_scan_row(hb + VSM_HIST_XFORM + 8u, box.xform[2]);
  for (var k = 28u; k < VSM_HIST_WORDS; k++) {
    vsm_history[hb + k] = 0u;
  }
  let bit = 1u << (s & 31u);
  if ((s >> 5u) < arrayLength(&vsm_cast_bits)) {
    if (cast_now) {
      atomicOr(&vsm_cast_bits[s >> 5u], bit);
    } else {
      atomicAnd(&vsm_cast_bits[s >> 5u], ~bit);
    }
  }

  vsm_scan_count(VSM_STAT_SCAN_VALID, valid);
  vsm_scan_count(VSM_STAT_SCAN_CASTERS, cast_now);
  vsm_scan_count(VSM_STAT_PENDING_STREAM, pending);
  vsm_scan_count(VSM_STAT_PENDING_CASTERS, pending && caster_class);
  vsm_scan_count(VSM_STAT_CHUNK_NEED, chunk_need);
  if (!changed) {
    return;
  }
  vsm_scan_count(VSM_STAT_CHANGED, true);
  vsm_scan_count(VSM_STAT_CHANGED_STATIC,
                 (w1 & VSM_CAT_STATIC) != 0u || (w2 & VSM_OBJ_IMMOBILE) != 0u);
  vsm_scan_count(VSM_STAT_CHANGED_IMMOBILE, (w2 & VSM_OBJ_IMMOBILE) != 0u);
  vsm_scan_count(VSM_STAT_CHANGED_PLAYER, player);
  vsm_scan_count(VSM_STAT_CHANGED_PALETTE, (w0 & RETAINED_PALETTE) != 0u);
  vsm_scan_count(VSM_STAT_NON_PLAYER_OTHER, !player && (reasons & (VSM_REASON_WORLD | VSM_REASON_STREAM)) == 0u);
  for (var k = 0u; k < VSM_REASON_COUNT; k++) {
    vsm_scan_count(VSM_STAT_REASON_FIRST + k, (reasons & (1u << k)) != 0u);
  }
}
//#endif

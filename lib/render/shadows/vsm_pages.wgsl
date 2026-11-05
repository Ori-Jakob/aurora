//#import aurora/vsm

@group(1) @binding(0) var<storage, read_write> vsm_views: array<u32>;
@group(1) @binding(1) var<storage, read_write> vsm_pt: array<u32>;
@group(1) @binding(2) var<storage, read_write> vsm_phys: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_stats: array<atomic<u32>>;

fn vsm_update_free(i: u32, e: u32, stat: u32) {
  vsm_pt[i] = 0u;
  if ((e & VSM_ENTRY_ALLOCATED) == 0u) {
    return;
  }
  let p = vsm_entry_phys(e) * VSM_PHYS_WORDS;
  if (p + VSM_PHYS_LAST_USED < arrayLength(&vsm_phys)) {
    vsm_phys[p + VSM_PHYS_OWNER] = 0u;
    vsm_phys[p + VSM_PHYS_LAST_USED] = 0u;
  }
  atomicAdd(&vsm_stats[stat], 1u);
}

fn vsm_update_origin(word: u32) -> vec2i {
  return bitcast<vec2i>(vec2u(vsm_views[word], vsm_views[word + 1u]));
}

@compute @workgroup_size(64)
fn vsm_update(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= min(VSM_MAX_VIRTUAL_PAGES, arrayLength(&vsm_pt))) {
    return;
  }
  if ((vsm_views[VSM_HDR_FLAGS] & VSM_FULL) != 0u) {
    if ((vsm_pt[i] & VSM_ENTRY_ALLOCATED) != 0u) {
      atomicAdd(&vsm_stats[VSM_STAT_FREED_FULL], 1u);
    }
    vsm_pt[i] = 0u;
    if (i < arrayLength(&vsm_phys)) {
      vsm_phys[i] = 0u;
    }
    return;
  }
  let e = vsm_pt[i];
  if (e == 0u) {
    return;
  }
  let base = VSM_HEADER_WORDS + (i / VSM_PAGES_PER_VIEW) * VSM_VIEW_WORDS;
  let word = vsm_views[base + VSM_VIEW_FLAGS];
  let flags = word & 0xFFu;
  if ((flags & VSM_VIEW_ACTIVE) == 0u || (flags & VSM_VIEW_FREED) != 0u) {
    vsm_update_free(i, e, VSM_STAT_FREED_RELATCH);
    return;
  }
  if ((flags & VSM_VIEW_RECENTRED) != 0u) {
    vsm_update_free(i, e, VSM_STAT_FREED_RECENTRE);
    return;
  }
  if ((flags & VSM_VIEW_WRAP) == 0u) {
    return;
  }
  let local = i - min(i, word >> 16u);
  let slot = vec2u(local % VSM_WINDOW_PAGES, (local / VSM_WINDOW_PAGES) % VSM_WINDOW_PAGES);
  let next = vsm_update_origin(base + VSM_VIEW_ORIGIN);
  let prev = vsm_update_origin(base + VSM_VIEW_PREV_ORIGIN);
  // A toroidal slot that wrapped now names another absolute page, so its old content is dropped
  if (!vsm_kept_after_shift(slot, prev, next)) {
    vsm_update_free(i, e, VSM_STAT_FREED_WRAP);
  }
}

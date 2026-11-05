//#import aurora/camera
//#import aurora/gbuffer
//#import aurora/vsm

//#if VSM_MARK
struct VsmMarkParams {
  camera: StageCamera,
  rotation: ViewRotation,
  size: vec2u,
  flags: u32,
  reserved: u32,
}

@group(0) @binding(1) var<uniform> vsm_mark_params: VsmMarkParams;
@group(1) @binding(0) var vsm_depth: texture_2d<f32>;
@group(1) @binding(1) var vsm_normal: texture_2d<f32>;
@group(1) @binding(2) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_requests: array<atomic<u32>>;
//#if VSM_MARK_CHECK
@group(1) @binding(4) var<storage, read_write> vsm_receivers: array<u32>;
//#endif

var<workgroup> vsm_mark_keys: array<u32, 512>;

@compute @workgroup_size(8, 8)
fn vsm_mark(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
            @builtin(local_invocation_index) li: u32) {
  let dims = textureDimensions(vsm_depth);
  var keys = vsm_no_keys();
  if (all(gid.xy < dims)) {
    let px = vec2i(gid.xy);
    let d = textureLoad(vsm_depth, px, 0).r;
    let a = textureLoad(vsm_normal, px, 0).a;
//#if VSM_MARK_CHECK
    let w = (gid.y * dims.x + gid.x) * 2u;
    if (w + 1u < arrayLength(&vsm_receivers)) {
      vsm_receivers[w] = bitcast<u32>(d);
      vsm_receivers[w + 1u] = bitcast<u32>(a);
    }
//#endif
    let cam = vsm_mark_params.camera;
    if (d > 0.0 && gbuf_receiver_code(a) >= 2u && (cam.flags & CAMERA_VALID) != 0u) {
      let full = vec2f(dims);
      let z = camera_view_depth(cam, d);
      let pos = camera_view_position_fwd(cam, (vec2f(gid.xy) + 0.5) / full, z, full);
      let p_rel = camera_world_dir(vsm_mark_params.rotation, pos);
      keys = vsm_request_keys(&vsm_views, p_rel, 2.0 * z / (full.y * cam.depth.w));
    }
  }
  for (var k = 0u; k < VSM_MARK_KEYS; k++) {
    vsm_mark_keys[li * VSM_MARK_KEYS + k] = keys[k];
  }
  workgroupBarrier();
  for (var k = 0u; k < VSM_MARK_KEYS; k++) {
    let key = keys[k];
    if (key == VSM_NO_KEY) {
      continue;
    }
    var seen = false;
    for (var j = 0u; j < VSM_MARK_KEYS; j++) {
      seen = seen || (lid.x > 0u && vsm_mark_keys[(li - 1u) * VSM_MARK_KEYS + j] == key);
      seen = seen || (lid.y > 0u && vsm_mark_keys[(li - 8u) * VSM_MARK_KEYS + j] == key);
    }
    let index = key & 0xFFFFu;
    if (!seen && index < arrayLength(&vsm_requests)) {
      atomicOr(&vsm_requests[index], key >> 16u);
    }
  }
}
//#endif

//#if VSM_CLASSIFY
@group(1) @binding(0) var<storage, read> vsm_requests: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(2) var<storage, read_write> vsm_pt: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_phys: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> vsm_need: array<u32>;
@group(1) @binding(6) var<storage, read_write> vsm_phys_words: array<u32>;

fn vsm_classify_count(stat: u32, on: bool) {
  if (on) {
    atomicAdd(&vsm_stats[stat], 1u);
  }
}

@compute @workgroup_size(64)
fn vsm_classify(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= min(VSM_MAX_VIRTUAL_PAGES, arrayLength(&vsm_pt)) ||
      VSM_NEED_CLASSES * VSM_MAX_VIRTUAL_PAGES > arrayLength(&vsm_need)) {
    return;
  }
  var req = 0u;
  if (v < arrayLength(&vsm_requests)) {
    req = vsm_requests[v] & VSM_REQUEST_MASK;
  }
  let e = vsm_pt[v];
  let allocated = (e & VSM_ENTRY_ALLOCATED) != 0u;
  let p = vsm_entry_phys(e) * VSM_PHYS_WORDS;
  let owned = allocated && p + VSM_PHYS_LAST_USED < arrayLength(&vsm_phys);
  let stale = (e & VSM_ENTRY_DIRTY) != 0u || (e & VSM_ENTRY_CACHED) == 0u;
  let priority = req != 0u && allocated && stale && (e & VSM_ENTRY_PRIORITY) != 0u;
  let reuse = req != 0u && allocated && stale && !priority;
  let fresh = req != 0u && !allocated;
  if (req != 0u && owned) {
    vsm_phys[p + VSM_PHYS_LAST_USED] = vsm_views[VSM_HDR_CLOCK];
  }
  let drop = req == 0u && allocated && (e & VSM_ENTRY_DIRTY) != 0u;
  if (drop) {
    vsm_pt[v] = 0u;
    if (owned) {
      vsm_phys[p + VSM_PHYS_OWNER] = 0u;
      vsm_phys[p + VSM_PHYS_LAST_USED] = 0u;
    }
  }
  vsm_need[VSM_NEED_PRIORITY * VSM_MAX_VIRTUAL_PAGES + v] = select(0u, 1u, priority);
  vsm_need[VSM_NEED_NEW * VSM_MAX_VIRTUAL_PAGES + v] = select(0u, 1u, fresh);
  vsm_need[VSM_NEED_REUSE * VSM_MAX_VIRTUAL_PAGES + v] = select(0u, 1u, reuse);
  vsm_classify_count(VSM_STAT_REQUESTED, req != 0u);
  vsm_classify_count(VSM_STAT_NEED_PRIORITY, priority);
  vsm_classify_count(VSM_STAT_NEED_NEW, fresh);
  vsm_classify_count(VSM_STAT_NEED_REUSE, reuse);
  vsm_classify_count(VSM_STAT_HITS, req != 0u && allocated && !stale);
  vsm_classify_count(VSM_STAT_FREED_DIRTY, drop);
}

@compute @workgroup_size(64)
fn vsm_phys_classify(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= VSM_MAX_POOL_PAGES || VSM_PHYS_BANDS * VSM_MAX_POOL_PAGES > arrayLength(&vsm_phys_words)) {
    return;
  }
  var band = VSM_PHYS_BANDS;
  let w = p * VSM_PHYS_WORDS;
  if (p < vsm_views[VSM_HDR_POOL_PAGES] && w + VSM_PHYS_LAST_USED < arrayLength(&vsm_phys)) {
    band = vsm_phys_band(vsm_phys[w + VSM_PHYS_OWNER], vsm_phys[w + VSM_PHYS_LAST_USED], vsm_views[VSM_HDR_CLOCK]);
  }
  for (var b = 0u; b < VSM_PHYS_BANDS; b++) {
    vsm_phys_words[b * VSM_MAX_POOL_PAGES + p] = select(0u, 1u, band == b);
  }
  vsm_classify_count(VSM_STAT_EVICTABLE, band < VSM_PHYS_BANDS);
  vsm_classify_count(VSM_STAT_FREE_PAGES, band == VSM_PHYS_FREE);
}
//#endif

//#if VSM_ASSIGN
@group(1) @binding(0) var<storage, read> vsm_need_list: array<u32>;
@group(1) @binding(1) var<storage, read> vsm_free_list: array<u32>;
@group(1) @binding(2) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(3) var<storage, read_write> vsm_pt: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_phys: array<u32>;
@group(1) @binding(5) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> vsm_render_list: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> vsm_clear_args: array<u32>;

fn vsm_assign_put(word: u32, value: u32) {
  if (word < arrayLength(&vsm_render_list)) {
    atomicStore(&vsm_render_list[word], value);
  }
}

fn vsm_assign_row(v: u32) {
  let view = v / VSM_PAGES_PER_VIEW;
  let slot = v % VSM_PAGES_PER_VIEW;
  let origin = bitcast<vec2i>(vec2u(vsm_views[VSM_HEADER_WORDS + view * VSM_VIEW_WORDS + VSM_VIEW_ORIGIN],
                                    vsm_views[VSM_HEADER_WORDS + view * VSM_VIEW_WORDS + VSM_VIEW_ORIGIN + 1u]));
  let local = vsm_slot_local(vec2u(slot % VSM_WINDOW_PAGES, slot / VSM_WINDOW_PAGES), origin);
  let word = VSM_RL_ROW_MASKS + view * VSM_WINDOW_PAGES + local.y;
  if (word < arrayLength(&vsm_render_list)) {
    atomicOr(&vsm_render_list[word], 1u << local.x);
  }
}

@compute @workgroup_size(64)
fn vsm_assign(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let clock = vsm_views[VSM_HDR_CLOCK];
  let held = (vsm_views[VSM_HDR_FLAGS] & VSM_HDR_FLAG_HELD) != 0u;
  let need_p = atomicLoad(&vsm_stats[VSM_STAT_NEED_PRIORITY]);
  let need_a = atomicLoad(&vsm_stats[VSM_STAT_NEED_NEW]);
  let need_r = atomicLoad(&vsm_stats[VSM_STAT_NEED_REUSE]);
  let evictable = atomicLoad(&vsm_stats[VSM_STAT_EVICTABLE]);
  let b = vsm_alloc_budget(need_p, need_a, need_r, evictable, vsm_views[VSM_HDR_BUDGET], held);
  if (i == 0u) {
    let pages = max(vsm_views[VSM_HDR_POOL_PAGES], 1u);
    let pages_x = max(vsm_views[VSM_HDR_POOL_PAGES_X], 1u);
    vsm_assign_put(VSM_RL_COUNT, b.count);
    vsm_assign_put(VSM_RL_PRIORITY, b.priority);
    vsm_assign_put(VSM_RL_NEW, b.fresh);
    vsm_assign_put(VSM_RL_REUSE, b.reuse);
    vsm_assign_put(VSM_RL_BUDGET, vsm_views[VSM_HDR_BUDGET]);
    vsm_assign_put(VSM_RL_PAGES_X, pages_x);
    vsm_assign_put(VSM_RL_ROWS, (pages + pages_x - 1u) / pages_x);
    vsm_assign_put(VSM_RL_FLAGS, select(0u, VSM_HDR_FLAG_HELD, held));
    if (arrayLength(&vsm_clear_args) >= 4u) {
      vsm_clear_args[0] = 6u;
      vsm_clear_args[1] = b.count;
      vsm_clear_args[2] = 0u;
      vsm_clear_args[3] = 0u;
    }
    atomicStore(&vsm_stats[VSM_STAT_RENDERED], b.count);
    atomicStore(&vsm_stats[VSM_STAT_DEFERRED_NEW], b.deferred_new);
    atomicStore(&vsm_stats[VSM_STAT_DEFERRED_REUSE], b.deferred_reuse);
    atomicStore(&vsm_stats[VSM_STAT_OVERFLOW_POOL], b.overflow_pool);
    atomicStore(&vsm_stats[VSM_STAT_PLAYER_DEFERRED], b.player_deferred);
  }
  if (i >= b.count || i >= arrayLength(&vsm_need_list)) {
    return;
  }
  let k = vsm_need_list[i];
  let v = k % VSM_MAX_VIRTUAL_PAGES;
  if (v >= arrayLength(&vsm_pt)) {
    return;
  }
  var page = VSM_NO_PAGE;
  // The lists are compacted in ascending page order, so the same requests always get the same pages
  if (k / VSM_MAX_VIRTUAL_PAGES == VSM_NEED_NEW) {
    let j = i - b.priority;
    if (j < evictable && j < arrayLength(&vsm_free_list)) {
      page = vsm_free_list[j] % VSM_MAX_POOL_PAGES;
      let w = page * VSM_PHYS_WORDS;
      let old = vsm_phys[w + VSM_PHYS_OWNER];
      if (old != 0u && old - 1u < arrayLength(&vsm_pt)) {
        vsm_pt[old - 1u] = 0u;
        atomicAdd(&vsm_stats[VSM_STAT_EVICTED], 1u);
      }
      vsm_pt[v] = page | VSM_ENTRY_ALLOCATED | VSM_ENTRY_RENDER_NOW | VSM_ENTRY_FRESH |
                  (i << VSM_ENTRY_RENDER_INDEX_SHIFT);
      vsm_phys[w + VSM_PHYS_OWNER] = v + 1u;
      vsm_phys[w + VSM_PHYS_LAST_USED] = clock;
    }
  } else {
    let e = vsm_pt[v];
    page = vsm_entry_phys(e);
    vsm_pt[v] = (e & (VSM_ENTRY_CACHED | VSM_ENTRY_DIRTY | VSM_ENTRY_PRIORITY)) | page | VSM_ENTRY_ALLOCATED |
                VSM_ENTRY_RENDER_NOW | (i << VSM_ENTRY_RENDER_INDEX_SHIFT);
  }
  vsm_assign_put(VSM_RL_PAGES + 2u * i, select(v, VSM_NO_PAGE, page == VSM_NO_PAGE));
  vsm_assign_put(VSM_RL_PAGES + 2u * i + 1u, page);
  if (page != VSM_NO_PAGE) {
    vsm_assign_row(v);
  }
}
//#endif

//#if VSM_COMMIT
@group(1) @binding(0) var<storage, read> vsm_render_list: array<u32>;
@group(1) @binding(1) var<storage, read_write> vsm_pt: array<u32>;
@group(1) @binding(2) var<storage, read_write> vsm_stats: array<atomic<u32>>;

@compute @workgroup_size(64)
fn vsm_commit(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= min(vsm_render_list[VSM_RL_COUNT], VSM_MAX_RENDER_PAGES) ||
      VSM_RL_PAGES + 2u * i >= arrayLength(&vsm_render_list)) {
    return;
  }
  let v = vsm_render_list[VSM_RL_PAGES + 2u * i];
  if (v >= min(VSM_MAX_VIRTUAL_PAGES, arrayLength(&vsm_pt))) {
    return;
  }
  let e = vsm_pt[v];
  if ((e & VSM_ENTRY_RENDER_NOW) == 0u) {
    return;
  }
  // Only commit makes a page Cached, so a frame whose raster never ran leaves no cleared page in use
  let keep = e & ~(VSM_ENTRY_RENDER_NOW | VSM_ENTRY_SKIP | VSM_ENTRY_FRESH |
                   (VSM_ENTRY_RENDER_INDEX_MASK << VSM_ENTRY_RENDER_INDEX_SHIFT));
  if ((e & VSM_ENTRY_SKIP) != 0u) {
    vsm_pt[v] = keep;
    atomicAdd(&vsm_stats[VSM_STAT_COMMIT_SKIPPED], 1u);
  } else {
    vsm_pt[v] = (keep | VSM_ENTRY_CACHED) & ~(VSM_ENTRY_DIRTY | VSM_ENTRY_PRIORITY);
    atomicAdd(&vsm_stats[VSM_STAT_COMMITTED], 1u);
  }
}
//#endif

//#if VSM_CLEAR
@group(0) @binding(0) var<storage, read> vsm_render_list: array<u32>;
@group(0) @binding(1) var<storage, read> vsm_pt: array<u32>;

@vertex
fn vs_clear(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let hidden = vec4f(2.0, 2.0, 0.0, 1.0);
  if (ii >= min(vsm_render_list[VSM_RL_COUNT], VSM_MAX_RENDER_PAGES) ||
      VSM_RL_PAGES + 2u * ii + 1u >= arrayLength(&vsm_render_list)) {
    return hidden;
  }
  let v = vsm_render_list[VSM_RL_PAGES + 2u * ii];
  let page = vsm_render_list[VSM_RL_PAGES + 2u * ii + 1u];
  if (v >= min(VSM_MAX_VIRTUAL_PAGES, arrayLength(&vsm_pt)) || page >= VSM_MAX_POOL_PAGES ||
      (vsm_pt[v] & VSM_ENTRY_SKIP) != 0u) {
    return hidden;
  }
  let pages_x = max(vsm_render_list[VSM_RL_PAGES_X], 1u);
  let size = vec2f(f32(pages_x), f32(max(vsm_render_list[VSM_RL_ROWS], 1u)));
  let corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0),
                                vec2f(1.0, 1.0));
  let uv = (vec2f(f32(page % pages_x), f32(page / pages_x)) + corners[vi % 6u]) / size;
  return vec4f(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, VSM_CLEAR_DEPTH, 1.0);
}
//#endif

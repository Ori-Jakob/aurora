const VSM_PAGE_TEXELS: u32 = 128u;
const VSM_WINDOW_PAGES: u32 = 32u;
const VSM_WINDOW_TEXELS: u32 = 4096u;
const VSM_PAGES_PER_VIEW: u32 = 1024u;
const VSM_SUN_LEVELS: u32 = 12u;
const VSM_MAX_VIEWS: u32 = 16u;
const VSM_MAX_VIRTUAL_PAGES: u32 = 16384u;
const VSM_MAX_POOL_PAGES: u32 = 4096u;
const VSM_HEADER_WORDS: u32 = 64u;
const VSM_VIEW_WORDS: u32 = 32u;
const VSM_VIEWS_WORDS: u32 = 576u;
const VSM_PHYS_WORDS: u32 = 2u;
const VSM_STAT_COUNT: u32 = 128u;

const VSM_ENTRY_PHYS_MASK: u32 = 0xFFFu;
const VSM_ENTRY_FRESH: u32 = 0x1000u;
const VSM_ENTRY_ALLOCATED: u32 = 0x2000u;
const VSM_ENTRY_CACHED: u32 = 0x4000u;
const VSM_ENTRY_DIRTY: u32 = 0x8000u;
const VSM_ENTRY_PRIORITY: u32 = 0x10000u;
const VSM_ENTRY_RENDER_NOW: u32 = 0x20000u;
const VSM_ENTRY_RENDER_INDEX_SHIFT: u32 = 18u;
const VSM_ENTRY_RENDER_INDEX_MASK: u32 = 0xFFFu;
const VSM_ENTRY_SKIP: u32 = 0x40000000u;

const VSM_PHYS_OWNER: u32 = 0u;
const VSM_PHYS_LAST_USED: u32 = 1u;

const VSM_HDR_FRAME_INDEX: u32 = 0u;
const VSM_HDR_CLOCK: u32 = 1u;
const VSM_HDR_FLAGS: u32 = 2u;
const VSM_HDR_FULL_REASON: u32 = 3u;
const VSM_HDR_VIEW_COUNT: u32 = 4u;
const VSM_HDR_ANCHOR_HI: u32 = 5u;
const VSM_HDR_ANCHOR_LO: u32 = 8u;
const VSM_HDR_BUDGET: u32 = 11u;
const VSM_HDR_POOL_PAGES: u32 = 12u;
const VSM_HDR_POOL_PAGES_X: u32 = 13u;
const VSM_HDR_LEVEL_BIAS: u32 = 14u;
const VSM_HDR_TEXEL0: u32 = 15u;
const VSM_HDR_SUN_DIR: u32 = 16u;
const VSM_HDR_LIGHT_ID: u32 = 19u;
const VSM_HDR_MAX_PAIRS: u32 = 20u;
const VSM_HDR_PAIRS: u32 = 21u;
const VSM_HDR_GENERATION: u32 = 24u;

const VSM_PRODUCING: u32 = 1u;
const VSM_FULL: u32 = 2u;
const VSM_EXEMPT: u32 = 4u;
const VSM_RELATCH: u32 = 8u;

const VSM_VIEW_CAM: u32 = 0u;
const VSM_VIEW_ANCHOR: u32 = 12u;
const VSM_VIEW_ORIGIN: u32 = 24u;
const VSM_VIEW_PREV_ORIGIN: u32 = 26u;
const VSM_VIEW_TEXEL: u32 = 28u;
const VSM_VIEW_DEPTH_SCALE: u32 = 29u;
const VSM_VIEW_FLAGS: u32 = 30u;
const VSM_VIEW_PAGES: u32 = 31u;

const VSM_VIEW_ACTIVE: u32 = 1u;
const VSM_VIEW_WRAP: u32 = 2u;
const VSM_VIEW_FREED: u32 = 4u;
const VSM_VIEW_RECENTRED: u32 = 8u;
const VSM_KIND_SUN_LEVEL: u32 = 1u;
const VSM_KIND_SPOT: u32 = 2u;
const VSM_KIND_CUBE_FACE: u32 = 3u;

const VSM_STAT_FREED_WRAP: u32 = 0u;
const VSM_STAT_FREED_RELATCH: u32 = 1u;
const VSM_STAT_FREED_RECENTRE: u32 = 2u;
const VSM_STAT_FREED_FULL: u32 = 3u;

struct VsmHeader {
  frame_index: u32,
  clock: u32,
  flags: u32,
  full_reason: u32,
  view_count: u32,
  anchor_hi: vec3f,
  anchor_lo: vec3f,
  budget: u32,
  pool_pages: u32,
  pool_pages_x: u32,
  level_bias: f32,
  texel0: f32,
  sun_dir: vec3f,
  light_id: u32,
  max_pairs: u32,
  generation: u32,
}

struct VsmView {
  cam: mat3x4f,
  anchor: mat3x4f,
  origin: vec2i,
  prev_origin: vec2i,
  texel: f32,
  depth_scale: f32,
  flags: u32,
  kind: u32,
  table_offset: u32,
  pages: vec2u,
}

fn vsm_f32(w: u32) -> f32 {
  return bitcast<f32>(w);
}

fn vsm_header_unpack(w: array<vec4u, 16>) -> VsmHeader {
  var h: VsmHeader;
  h.frame_index = w[0].x;
  h.clock = w[0].y;
  h.flags = w[0].z;
  h.full_reason = w[0].w;
  h.view_count = w[1].x;
  h.anchor_hi = bitcast<vec3f>(w[1].yzw);
  h.anchor_lo = bitcast<vec3f>(w[2].xyz);
  h.budget = w[2].w;
  h.pool_pages = w[3].x;
  h.pool_pages_x = w[3].y;
  h.level_bias = vsm_f32(w[3].z);
  h.texel0 = vsm_f32(w[3].w);
  h.sun_dir = bitcast<vec3f>(w[4].xyz);
  h.light_id = w[4].w;
  h.max_pairs = w[5].x;
  h.generation = w[6].x;
  return h;
}

fn vsm_rows(a: vec4u, b: vec4u, c: vec4u) -> mat3x4f {
  return mat3x4f(bitcast<vec4f>(a), bitcast<vec4f>(b), bitcast<vec4f>(c));
}

fn vsm_view_unpack(w: array<vec4u, 8>) -> VsmView {
  var v: VsmView;
  v.cam = vsm_rows(w[0], w[1], w[2]);
  v.anchor = vsm_rows(w[3], w[4], w[5]);
  v.origin = bitcast<vec2i>(w[6].xy);
  v.prev_origin = bitcast<vec2i>(w[6].zw);
  v.texel = vsm_f32(w[7].x);
  v.depth_scale = vsm_f32(w[7].y);
  v.flags = w[7].z & 0xFFu;
  v.kind = (w[7].z >> 8u) & 0xFFu;
  v.table_offset = w[7].z >> 16u;
  v.pages = vec2u(w[7].w & 0xFFFFu, w[7].w >> 16u);
  return v;
}

fn vsm_quad(views: ptr<storage, array<u32>, read>, word: u32) -> vec4u {
  return vec4u((*views)[word], (*views)[word + 1u], (*views)[word + 2u], (*views)[word + 3u]);
}

fn vsm_header(views: ptr<storage, array<u32>, read>) -> VsmHeader {
  var w: array<vec4u, 16>;
  for (var i = 0u; i < 16u; i++) {
    w[i] = vsm_quad(views, i * 4u);
  }
  return vsm_header_unpack(w);
}

fn vsm_view(views: ptr<storage, array<u32>, read>, i: u32) -> VsmView {
  let base = VSM_HEADER_WORDS + min(i, VSM_MAX_VIEWS - 1u) * VSM_VIEW_WORDS;
  var w: array<vec4u, 8>;
  for (var k = 0u; k < 8u; k++) {
    w[k] = vsm_quad(views, base + k * 4u);
  }
  return vsm_view_unpack(w);
}

fn vsm_level_view(level: u32) -> u32 {
  return VSM_SUN_LEVELS - 1u - level;
}

fn vsm_view_coords(v: VsmView, p_rel: vec3f) -> vec3f {
  return vec4f(p_rel, 1.0) * v.cam;
}

fn vsm_view_coords_anchor(v: VsmView, p_anchor: vec3f) -> vec3f {
  return vec4f(p_anchor, 1.0) * v.anchor;
}

fn vsm_view_dir(v: VsmView) -> vec3f {
  return -normalize(v.anchor[2].xyz);
}

fn vsm_abs_page(origin: i32, slot: u32) -> i32 {
  return origin + i32((slot - u32(origin)) & (VSM_WINDOW_PAGES - 1u));
}

fn vsm_page_index(table_offset: u32, page: vec2i) -> u32 {
  let p = bitcast<vec2u>(page) & vec2u(VSM_WINDOW_PAGES - 1u);
  return table_offset + p.x + VSM_WINDOW_PAGES * p.y;
}

fn vsm_kept_after_shift(slot: vec2u, prev: vec2i, next: vec2i) -> bool {
  let page = vec2i(vsm_abs_page(prev.x, slot.x), vsm_abs_page(prev.y, slot.y));
  let d = page - next;
  return all(d >= vec2i(0)) && all(d < vec2i(i32(VSM_WINDOW_PAGES)));
}

fn vsm_entry_phys(e: u32) -> u32 {
  return e & VSM_ENTRY_PHYS_MASK;
}

fn vsm_entry_render_index(e: u32) -> u32 {
  return (e >> VSM_ENTRY_RENDER_INDEX_SHIFT) & VSM_ENTRY_RENDER_INDEX_MASK;
}

fn vsm_page_usable(e: u32) -> bool {
  return (e & (VSM_ENTRY_ALLOCATED | VSM_ENTRY_CACHED | VSM_ENTRY_RENDER_NOW)) ==
         (VSM_ENTRY_ALLOCATED | VSM_ENTRY_CACHED);
}

fn vsm_page_stale(e: u32) -> bool {
  return vsm_page_usable(e) && (e & VSM_ENTRY_DIRTY) != 0u;
}

const VSM_HDR_FLAG_HELD: u32 = 16u;

const VSM_STAT_SCAN_VALID: u32 = 4u;
const VSM_STAT_SCAN_CASTERS: u32 = 5u;
const VSM_STAT_CHANGED: u32 = 6u;
const VSM_STAT_CHANGED_STATIC: u32 = 7u;
const VSM_STAT_CHANGED_IMMOBILE: u32 = 8u;
const VSM_STAT_CHANGED_PLAYER: u32 = 9u;
const VSM_STAT_CHANGED_PALETTE: u32 = 10u;
const VSM_STAT_PENDING_STREAM: u32 = 11u;
const VSM_STAT_PENDING_CASTERS: u32 = 12u;
const VSM_STAT_CHUNK_NEED: u32 = 13u;
const VSM_STAT_REASON_FIRST: u32 = 14u;
const VSM_STAT_NON_PLAYER_OTHER: u32 = 23u;
const VSM_STAT_FOOTPRINT: u32 = 24u;
const VSM_STAT_FOOTPRINT_PRIORITY: u32 = 25u;
const VSM_STAT_DIRTIED: u32 = 26u;
const VSM_STAT_DIRTIED_PRIORITY: u32 = 27u;

const VSM_SLOTS: u32 = 16384u;
const VSM_HIST_WORDS: u32 = 32u;
const VSM_HIST_FLAGS: u32 = 0u;
const VSM_HIST_HASH: u32 = 1u;
const VSM_HIST_STREAM_FRAME: u32 = 2u;
const VSM_HIST_WORLD_FRAME: u32 = 3u;
const VSM_HIST_MATERIAL_FRAME: u32 = 4u;
const VSM_HIST_UV_FRAME: u32 = 5u;
const VSM_HIST_CLUSTER_FIRST: u32 = 6u;
const VSM_HIST_CLUSTER_COUNT: u32 = 7u;
const VSM_HIST_INDEX_FIRST: u32 = 8u;
const VSM_HIST_INDEX_COUNT: u32 = 9u;
const VSM_HIST_BOX: u32 = 10u;
const VSM_HIST_XFORM: u32 = 16u;

const VSM_HIST_VALID: u32 = 1u;
const VSM_HIST_CASTER: u32 = 2u;
const VSM_HIST_CUTOUT: u32 = 4u;
const VSM_HIST_ALPHA_PENDING: u32 = 8u;
const VSM_HIST_PLAYER: u32 = 16u;
const VSM_HIST_CHUNKED: u32 = 32u;
const VSM_HIST_PENDING_STREAM: u32 = 64u;
const VSM_HIST_SEEN: u32 = 128u;
const VSM_HIST_BOX_VALID: u32 = 256u;

const VSM_SLOT_CHANGED: u32 = 1u;
const VSM_SLOT_CHUNK_NEED: u32 = 2u;

const VSM_DIRTY_BOX_WORDS: u32 = 16u;
const VSM_DIRTY_BOX_OLD: u32 = 0u;
const VSM_DIRTY_BOX_NEW: u32 = 6u;
const VSM_DIRTY_BOX_FLAGS: u32 = 12u;
const VSM_DIRTY_HAS_OLD: u32 = 1u;
const VSM_DIRTY_HAS_NEW: u32 = 2u;
const VSM_DIRTY_OLD_PLAYER: u32 = 4u;
const VSM_DIRTY_NEW_PLAYER: u32 = 8u;
const VSM_DIRTY_PLANE_WORDS: u32 = 512u;

const VSM_REASON_CASTER: u32 = 1u;
const VSM_REASON_HASH: u32 = 2u;
const VSM_REASON_STREAM: u32 = 4u;
const VSM_REASON_WORLD: u32 = 8u;
const VSM_REASON_MATERIAL: u32 = 16u;
const VSM_REASON_UV: u32 = 32u;
const VSM_REASON_ALPHA: u32 = 64u;
const VSM_REASON_CUTOUT: u32 = 128u;
const VSM_REASON_ARRAYS: u32 = 256u;
const VSM_REASON_COUNT: u32 = 9u;

const VSM_CAT_STATIC: u32 = 1u;
const VSM_CAT_PLAYER: u32 = 1024u;
const VSM_OBJ_IMMOBILE: u32 = 64u;

// Bit for bit vsm::is_caster's exclusions, so the GPU and CPU caster sets cannot drift apart
const VSM_EXCLUDED_CLASSES: u32 = 0xA24u;
const VSM_EXCLUDED_CATEGORIES: u32 = 0x4113C0u;

fn vsm_is_caster(flags: u32, categories: u32) -> bool {
  return (flags & 1u) != 0u && (flags & VSM_EXCLUDED_CLASSES) == 0u && (categories & VSM_EXCLUDED_CATEGORIES) == 0u;
}

fn vsm_box_rect(v: VsmView, box: array<vec3f, 2>) -> vec4i {
  var lo = vec2f(3.4e38);
  var hi = vec2f(-3.4e38);
  for (var i = 0u; i < 8u; i++) {
    let corner = select(box[0], box[1], vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u));
    let t = vsm_view_coords_anchor(v, corner).xy;
    lo = min(lo, t);
    hi = max(hi, t);
  }
  let limit = vec2f(v.pages) - 1.0;
  let p0 = clamp(floor(lo / f32(VSM_PAGE_TEXELS)), vec2f(-1.0), limit + 1.0);
  let p1 = clamp(floor(hi / f32(VSM_PAGE_TEXELS)), vec2f(-1.0), limit + 1.0);
  return vec4i(vec2i(max(p0, vec2f(0.0))), vec2i(min(p1, limit)));
}

const VSM_REQUEST_PIXEL: u32 = 1u;
const VSM_REQUEST_FALLBACK: u32 = 2u;
const VSM_REQUEST_VOLUME: u32 = 4u;
const VSM_REQUEST_PRIORITY: u32 = 8u;
const VSM_REQUEST_MASK: u32 = 15u;
const VSM_FILTER_MARGIN_TEXELS: u32 = 4u;
const VSM_MARK_KEYS: u32 = 8u;
const VSM_NO_KEY: u32 = 0xFFFFFFFFu;
const VSM_NO_PAGE: u32 = 0xFFFFFFFFu;
const VSM_CLEAR_DEPTH: f32 = 0.0;

const VSM_NEED_CLASSES: u32 = 3u;
const VSM_NEED_PRIORITY: u32 = 0u;
const VSM_NEED_NEW: u32 = 1u;
const VSM_NEED_REUSE: u32 = 2u;
const VSM_PHYS_BANDS: u32 = 4u;
const VSM_PHYS_FREE: u32 = 0u;
const VSM_PHYS_OLD: u32 = 1u;
const VSM_PHYS_MID: u32 = 2u;
const VSM_PHYS_YOUNG: u32 = 3u;
const VSM_OLD_AGE: u32 = 64u;
const VSM_MID_AGE: u32 = 8u;
const VSM_MAX_RENDER_PAGES: u32 = 4096u;

const VSM_RL_COUNT: u32 = 0u;
const VSM_RL_PRIORITY: u32 = 1u;
const VSM_RL_NEW: u32 = 2u;
const VSM_RL_REUSE: u32 = 3u;
const VSM_RL_BUDGET: u32 = 4u;
const VSM_RL_PAGES_X: u32 = 5u;
const VSM_RL_ROWS: u32 = 6u;
const VSM_RL_FLAGS: u32 = 7u;
const VSM_RL_PAGES: u32 = 16u;
const VSM_RL_ROW_MASKS: u32 = 8208u;
const VSM_RL_WORDS: u32 = 8720u;

const VSM_STAT_REQUESTED: u32 = 28u;
const VSM_STAT_NEED_PRIORITY: u32 = 29u;
const VSM_STAT_NEED_NEW: u32 = 30u;
const VSM_STAT_NEED_REUSE: u32 = 31u;
const VSM_STAT_HITS: u32 = 32u;
const VSM_STAT_FREED_DIRTY: u32 = 33u;
const VSM_STAT_EVICTABLE: u32 = 34u;
const VSM_STAT_FREE_PAGES: u32 = 35u;
const VSM_STAT_RENDERED: u32 = 36u;
const VSM_STAT_DEFERRED_NEW: u32 = 37u;
const VSM_STAT_DEFERRED_REUSE: u32 = 38u;
const VSM_STAT_OVERFLOW_POOL: u32 = 39u;
const VSM_STAT_PLAYER_DEFERRED: u32 = 40u;
const VSM_STAT_EVICTED: u32 = 41u;
const VSM_STAT_COMMITTED: u32 = 42u;
const VSM_STAT_COMMIT_SKIPPED: u32 = 43u;

fn vsm_slot_local(slot: vec2u, origin: vec2i) -> vec2u {
  return (slot - bitcast<vec2u>(origin)) & vec2u(VSM_WINDOW_PAGES - 1u);
}

fn vsm_render_page(render_list: ptr<storage, array<u32>, read>, index: u32) -> vec2u {
  let word = VSM_RL_PAGES + 2u * index;
  if (word + 1u >= arrayLength(render_list)) {
    return vec2u(VSM_NO_PAGE);
  }
  return vec2u((*render_list)[word], (*render_list)[word + 1u]);
}

fn vsm_level_wanted(f: f32, texel0: f32, bias: f32) -> u32 {
  return u32(clamp(ceil(log2(max(f, 1e-30) / texel0) + bias), 0.0, f32(VSM_SUN_LEVELS - 1u)));
}

fn vsm_level_fits(v: VsmView, p_rel: vec3f) -> bool {
  let c = vsm_view_coords(v, p_rel).xy;
  let m = f32(VSM_FILTER_MARGIN_TEXELS);
  return all(c >= vec2f(m)) && all(c < vec2f(f32(VSM_WINDOW_TEXELS) - m));
}

fn vsm_select_level(views: ptr<storage, array<u32>, read>, f: f32, p_rel: vec3f, bias: f32) -> u32 {
  let texel0 = vsm_f32((*views)[VSM_HDR_TEXEL0]);
  if (!(texel0 > 0.0)) {
    return VSM_SUN_LEVELS;
  }
  for (var level = vsm_level_wanted(f, texel0, bias); level < VSM_SUN_LEVELS; level++) {
    if (vsm_level_fits(vsm_view(views, vsm_level_view(level)), p_rel)) {
      return level;
    }
  }
  return VSM_SUN_LEVELS;
}

fn vsm_no_keys() -> array<u32, 8> {
  return array<u32, 8>(VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY, VSM_NO_KEY);
}

fn vsm_window_keys(c: vec2f, origin: vec2i, table_offset: u32, bits: u32, keys: ptr<function, array<u32, 8>>,
                   first: u32) {
  let m = f32(VSM_FILTER_MARGIN_TEXELS);
  let lo = max(vec2i(floor((c - m) / f32(VSM_PAGE_TEXELS))), vec2i(0));
  let hi = min(vec2i(floor((c + m) / f32(VSM_PAGE_TEXELS))), vec2i(i32(VSM_WINDOW_PAGES) - 1));
  var k = first;
  for (var y = lo.y; y <= min(hi.y, lo.y + 1); y++) {
    for (var x = lo.x; x <= min(hi.x, lo.x + 1); x++) {
      (*keys)[k] = vsm_page_index(table_offset, origin + vec2i(x, y)) | (bits << 16u);
      k++;
    }
  }
}

fn vsm_level_keys(v: VsmView, p_rel: vec3f, bits: u32, keys: ptr<function, array<u32, 8>>, first: u32) {
  vsm_window_keys(vsm_view_coords(v, p_rel).xy, v.origin, v.table_offset, bits, keys, first);
}

fn vsm_request_keys(views: ptr<storage, array<u32>, read>, p_rel: vec3f, f: f32) -> array<u32, 8> {
  var keys = vsm_no_keys();
  let level = vsm_select_level(views, f, p_rel, vsm_f32((*views)[VSM_HDR_LEVEL_BIAS]));
  if (level >= VSM_SUN_LEVELS) {
    return keys;
  }
  vsm_level_keys(vsm_view(views, vsm_level_view(level)), p_rel, VSM_REQUEST_PIXEL, &keys, 0u);
  if (level + 1u < VSM_SUN_LEVELS) {
    vsm_level_keys(vsm_view(views, vsm_level_view(level + 1u)), p_rel, VSM_REQUEST_FALLBACK, &keys, 4u);
  }
  return keys;
}

fn vsm_uniform_base(i: u32) -> u32 {
  return (VSM_HEADER_WORDS + min(i, VSM_MAX_VIEWS - 1u) * VSM_VIEW_WORDS) / 4u;
}

// Level selection and requests need only the window position, so they read two matrix rows instead of the whole view
fn vsm_uniform_coords(views: ptr<uniform, array<vec4u, 144>>, i: u32, p_rel: vec3f) -> vec2f {
  let base = vsm_uniform_base(i);
  let p = vec4f(p_rel, 1.0);
  return vec2f(dot(p, bitcast<vec4f>((*views)[base])), dot(p, bitcast<vec4f>((*views)[base + 1u])));
}

fn vsm_coords_fit(c: vec2f) -> bool {
  let m = f32(VSM_FILTER_MARGIN_TEXELS);
  return all(c >= vec2f(m)) && all(c < vec2f(f32(VSM_WINDOW_TEXELS) - m));
}

fn vsm_uniform_level_keys(views: ptr<uniform, array<vec4u, 144>>, level: u32, p_rel: vec3f, bits: u32,
                          keys: ptr<function, array<u32, 8>>, first: u32) {
  let view = vsm_level_view(level);
  let base = vsm_uniform_base(view);
  vsm_window_keys(vsm_uniform_coords(views, view, p_rel), bitcast<vec2i>((*views)[base + 6u].xy),
                  (*views)[base + 7u].z >> 16u, bits, keys, first);
}

fn vsm_uniform_request_keys(views: ptr<uniform, array<vec4u, 144>>, p_rel: vec3f, f: f32) -> array<u32, 8> {
  var keys = vsm_no_keys();
  let texel0 = vsm_f32((*views)[3].w);
  if (!(texel0 > 0.0)) {
    return keys;
  }
  for (var level = vsm_level_wanted(f, texel0, vsm_f32((*views)[3].z)); level < VSM_SUN_LEVELS; level++) {
    if (vsm_coords_fit(vsm_uniform_coords(views, vsm_level_view(level), p_rel))) {
      vsm_uniform_level_keys(views, level, p_rel, VSM_REQUEST_PIXEL, &keys, 0u);
      if (level + 1u < VSM_SUN_LEVELS) {
        vsm_uniform_level_keys(views, level + 1u, p_rel, VSM_REQUEST_FALLBACK, &keys, 4u);
      }
      break;
    }
  }
  return keys;
}

fn vsm_request(requests: ptr<storage, array<atomic<u32>>, read_write>, v: VsmView, page: vec2i, bits: u32) {
  let local = page - v.origin;
  if (any(local < vec2i(0)) || any(local >= vec2i(i32(VSM_WINDOW_PAGES)))) {
    return;
  }
  let index = vsm_page_index(v.table_offset, page);
  if (index < arrayLength(requests)) {
    atomicOr(&(*requests)[index], bits & VSM_REQUEST_MASK);
  }
}

fn vsm_request_world(requests: ptr<storage, array<atomic<u32>>, read_write>, views: ptr<storage, array<u32>, read>,
                     p_rel: vec3f, footprint: f32, bits: u32) {
  let keys = vsm_request_keys(views, p_rel, footprint);
  for (var k = 0u; k < VSM_MARK_KEYS; k++) {
    let key = keys[k];
    let index = key & 0xFFFFu;
    if (key != VSM_NO_KEY && index < arrayLength(requests)) {
      atomicOr(&(*requests)[index], select(VSM_REQUEST_FALLBACK, bits & VSM_REQUEST_MASK, k < 4u));
    }
  }
}

fn vsm_phys_band(owner: u32, last_used: u32, clock: u32) -> u32 {
  if (owner == 0u) {
    return VSM_PHYS_FREE;
  }
  let age = clock - last_used;
  if (age >= VSM_OLD_AGE) {
    return VSM_PHYS_OLD;
  }
  if (age >= VSM_MID_AGE) {
    return VSM_PHYS_MID;
  }
  return select(VSM_PHYS_BANDS, VSM_PHYS_YOUNG, age >= 1u);
}

struct VsmBudget {
  count: u32,
  priority: u32,
  fresh: u32,
  reuse: u32,
  deferred_new: u32,
  deferred_reuse: u32,
  overflow_pool: u32,
  player_deferred: u32,
}

fn vsm_alloc_budget(priority: u32, fresh: u32, reuse: u32, evictable: u32, budget: u32, held: bool) -> VsmBudget {
  var b: VsmBudget;
  if (held) {
    return b;
  }
  let within = min(budget, fresh + reuse);
  b.count = min(priority + within, VSM_MAX_RENDER_PAGES);
  b.priority = min(priority, b.count);
  b.fresh = min(min(fresh, within), b.count - b.priority);
  b.reuse = min(within - min(fresh, within), b.count - b.priority - b.fresh);
  b.deferred_new = fresh - b.fresh;
  b.deferred_reuse = reuse - b.reuse;
  b.overflow_pool = b.fresh - min(b.fresh, evictable);
  b.player_deferred = priority - b.priority;
  return b;
}

const VSM_HIST_VERTEX_FIRST: u32 = 28u;
const VSM_HIST_CULL_MASK: u32 = 29u;

const VSM_MAX_PAIRS: u32 = 524288u;
const VSM_VERTICES_PER_PAIR: u32 = 372u;
const VSM_PAIR_CHUNK: u32 = 0x80000000u;
const VSM_PAIR_SLOT_BITS: u32 = 14u;
const VSM_PAIRS_HEADER_WORDS: u32 = 16u;
const VSM_PAIRS_OPAQUE_COUNT: u32 = 0u;
const VSM_PAIRS_CUTOUT_BASE: u32 = 1u;
const VSM_PAIRS_CUTOUT_COUNT: u32 = 2u;
const VSM_PAIRS_SKIPPED: u32 = 3u;
const VSM_PAIRS_CHUNK_PAIRS: u32 = 4u;
const VSM_PAIRS_POSED_PAIRS: u32 = 5u;
const VSM_PAIRS_TOTAL_OPAQUE: u32 = 6u;
const VSM_PAIRS_SKIP_KEY: u32 = 7u;
const VSM_PAIRS_DROPPED: u32 = 8u;
const VSM_PAIRS_MAX: u32 = 9u;
const VSM_PAIRS_ENDS: u32 = 1048592u;
const VSM_CHUNK_TRIANGLES: u32 = 124u;
const VSM_CHUNK_INDICES: u32 = 372u;
const VSM_INDEX_GRANULE: u32 = 64u;
const VSM_CHUNK_BOXES: u32 = 65536u;
const VSM_NO_CHUNK: u32 = 0xFFFFFFFFu;
const VSM_CHUNK_QUANTA: f32 = 1023.0;
const VSM_CENTRE_SLACK: f32 = 1e-3;

const VSM_STAT_CULL_SLOTS: u32 = 44u;
const VSM_STAT_SKIPPED_PAGES: u32 = 45u;
const VSM_STAT_OPAQUE_PAIRS: u32 = 46u;
const VSM_STAT_CUTOUT_PAIRS: u32 = 47u;
const VSM_STAT_CHUNK_BOXES: u32 = 48u;
const VSM_STAT_CHUNK_MISALIGNED: u32 = 49u;

struct VsmPair {
  item: u32,
  chunk: bool,
  slot: u32,
  render_index: u32,
}

fn vsm_pair_pack(item: u32, slot: u32, render_index: u32) -> vec2u {
  return vec2u(item, (slot & ((1u << VSM_PAIR_SLOT_BITS) - 1u)) | (render_index << VSM_PAIR_SLOT_BITS));
}

fn vsm_pair_unpack(w: vec2u) -> VsmPair {
  var p: VsmPair;
  p.chunk = (w.x & VSM_PAIR_CHUNK) != 0u;
  p.item = w.x & ~VSM_PAIR_CHUNK;
  p.slot = w.y & ((1u << VSM_PAIR_SLOT_BITS) - 1u);
  p.render_index = w.y >> VSM_PAIR_SLOT_BITS;
  return p;
}

fn vsm_chunk_count(index_count: u32) -> u32 {
  return (index_count / 3u + VSM_CHUNK_TRIANGLES - 1u) / VSM_CHUNK_TRIANGLES;
}

fn vsm_chunk_slot(index_first: u32, chunk: u32) -> u32 {
  if (index_first % VSM_INDEX_GRANULE != 0u) {
    return VSM_NO_CHUNK;
  }
  return index_first / VSM_INDEX_GRANULE + chunk;
}

fn vsm_chunk_box_pack(box: array<vec3f, 2>, bounds: array<vec3f, 2>) -> vec2u {
  let extent = max(bounds[1] - bounds[0], vec3f(1e-30));
  let lo = vec3u(clamp(floor((box[0] - bounds[0]) / extent * VSM_CHUNK_QUANTA), vec3f(0.0), vec3f(VSM_CHUNK_QUANTA)));
  let hi = vec3u(clamp(ceil((box[1] - bounds[0]) / extent * VSM_CHUNK_QUANTA), vec3f(0.0), vec3f(VSM_CHUNK_QUANTA)));
  return vec2u(lo.x | (lo.y << 10u) | (lo.z << 20u), hi.x | (hi.y << 10u) | (hi.z << 20u));
}

fn vsm_chunk_box(words: vec2u, bounds: array<vec3f, 2>) -> array<vec3f, 2> {
  let extent = bounds[1] - bounds[0];
  let lo = vec3f(vec3u(words.x, words.x >> 10u, words.x >> 20u) & vec3u(1023u)) / VSM_CHUNK_QUANTA;
  let hi = vec3f(vec3u(words.y, words.y >> 10u, words.y >> 20u) & vec3u(1023u)) / VSM_CHUNK_QUANTA;
  return array<vec3f, 2>(bounds[0] + lo * extent, bounds[0] + hi * extent);
}

fn vsm_box_through(lo: vec3f, hi: vec3f, xform: mat3x4f) -> array<vec3f, 2> {
  var bmin = vec3f(3.4e38);
  var bmax = vec3f(-3.4e38);
  for (var i = 0u; i < 8u; i++) {
    let corner = select(lo, hi, vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u));
    let p = vec4f(corner, 1.0) * xform;
    bmin = min(bmin, p);
    bmax = max(bmax, p);
  }
  return array<vec3f, 2>(bmin, bmax);
}

fn vsm_texel_rect(v: VsmView, box: array<vec3f, 2>) -> vec4f {
  var lo = vec2f(3.4e38);
  var hi = vec2f(-3.4e38);
  for (var i = 0u; i < 8u; i++) {
    let corner = select(box[0], box[1], vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u));
    let t = vsm_view_coords_anchor(v, corner).xy;
    lo = min(lo, t);
    hi = max(hi, t);
  }
  return vec4f(lo, hi);
}

fn vsm_rect_holds_centre(t: vec4f) -> bool {
  let lo = t.xy - vec2f(VSM_CENTRE_SLACK);
  let hi = t.zw + vec2f(VSM_CENTRE_SLACK);
  return all(ceil(lo - 0.5) <= floor(hi - 0.5));
}

fn vsm_page_rect(v: VsmView, t: vec4f) -> vec4i {
  let limit = vec2f(v.pages) - 1.0;
  let p0 = clamp(floor((t.xy - vec2f(VSM_CENTRE_SLACK)) / f32(VSM_PAGE_TEXELS)), vec2f(-1.0), limit + 1.0);
  let p1 = clamp(floor((t.zw + vec2f(VSM_CENTRE_SLACK)) / f32(VSM_PAGE_TEXELS)), vec2f(-1.0), limit + 1.0);
  return vec4i(vec2i(max(p0, vec2f(0.0))), vec2i(min(p1, limit)));
}

fn vsm_row_bits(x0: u32, x1: u32) -> u32 {
  let upper = select((1u << (x1 + 1u)) - 1u, 0xFFFFFFFFu, x1 >= 31u);
  return upper & ~((1u << x0) - 1u);
}

const VSM_ENTRY_CHECK: u32 = 0x80000000u;
const VSM_HDR_FLAG_CHECK: u32 = 32u;
const VSM_NEED_CHECK: u32 = 3u;
const VSM_RL_CHECK_COUNT: u32 = 8u;
const VSM_RL_CHECK_ONLY: u32 = 0x80000000u;

const VSM_STAT_CHECK_PAGES: u32 = 50u;
const VSM_STAT_CHECK_LISTED: u32 = 51u;
const VSM_STAT_CHECK_COMPARED: u32 = 52u;
const VSM_STAT_CHECK_TEXELS: u32 = 53u;
const VSM_STAT_CHECK_MISMATCHED: u32 = 54u;
const VSM_STAT_CHECK_FIRST: u32 = 55u;
const VSM_STAT_CHECK_SKIPPED: u32 = 56u;
const VSM_STAT_PROBE_FOUND: u32 = 60u;
const VSM_STAT_PROBE_OCCLUDED: u32 = 61u;
const VSM_STAT_SUN_UNSHADOWED: u32 = 64u;
const VSM_STAT_SUN_SAMPLES: u32 = 65u;
const VSM_STAT_SUN_SAMPLED_UNSHADOWED: u32 = 66u;
const VSM_STAT_SUN_SAMPLED_FALLBACK: u32 = 67u;
const VSM_STAT_SUN_SAMPLED_STALE: u32 = 68u;
const VSM_STAT_SUN_SAMPLED_OUT_OF_RANGE: u32 = 69u;

fn vsm_render_count(render_list: ptr<storage, array<u32>, read>) -> u32 {
  if (VSM_RL_CHECK_COUNT >= arrayLength(render_list)) {
    return 0u;
  }
  return min((*render_list)[VSM_RL_COUNT] + (*render_list)[VSM_RL_CHECK_COUNT], VSM_MAX_RENDER_PAGES);
}

fn vsm_render_virtual(word: u32) -> u32 {
  return select(word & ~VSM_RL_CHECK_ONLY, VSM_NO_PAGE, word == VSM_NO_PAGE);
}

fn vsm_entry_listed(e: u32) -> bool {
  return (e & VSM_ENTRY_SKIP) == 0u && (e & (VSM_ENTRY_RENDER_NOW | VSM_ENTRY_CHECK)) != 0u;
}

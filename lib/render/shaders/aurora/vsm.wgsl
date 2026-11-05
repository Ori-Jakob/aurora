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
const VSM_STAT_COUNT: u32 = 64u;

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

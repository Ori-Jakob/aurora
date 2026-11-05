const RETAINED_VERTEX_WORDS: u32 = 8u;
const RETAINED_RECORD_WORDS: u32 = 96u;
const RETAINED_FRAME_WORDS: u32 = 64u;

const RETAINED_REC_FLAGS: u32 = 0u;
const RETAINED_REC_CATEGORIES: u32 = 1u;
const RETAINED_REC_OBJECT_FLAGS: u32 = 2u;
const RETAINED_REC_IDENTITY_HASH: u32 = 3u;
const RETAINED_REC_VERTEX_FIRST: u32 = 4u;
const RETAINED_REC_VERTEX_COUNT: u32 = 5u;
const RETAINED_REC_INDEX_FIRST: u32 = 6u;
const RETAINED_REC_INDEX_COUNT: u32 = 7u;
const RETAINED_REC_PALETTE_FIRST: u32 = 8u;
const RETAINED_REC_MATERIAL: u32 = 9u;
const RETAINED_REC_WORLD_FRAME: u32 = 10u;
const RETAINED_REC_GROUP: u32 = 11u;
const RETAINED_REC_WORLD: u32 = 12u;
const RETAINED_REC_PREV_WORLD: u32 = 28u;
const RETAINED_REC_NORMAL: u32 = 44u;
const RETAINED_REC_UV_BASE: u32 = 56u;
const RETAINED_REC_UV_ALPHA: u32 = 62u;
const RETAINED_REC_UV_FRAME: u32 = 68u;
const RETAINED_REC_MATERIAL_FRAME: u32 = 69u;
const RETAINED_REC_DRAW_ATTRIBUTES: u32 = 70u;
const RETAINED_REC_CLUSTER_FIRST: u32 = 73u;
const RETAINED_REC_CLUSTER_COUNT: u32 = 74u;
const RETAINED_REC_BOUNDS_OFFSET: u32 = 75u;
const RETAINED_REC_CUR_AABB: u32 = 80u;
const RETAINED_REC_PREV_AABB: u32 = 86u;
const RETAINED_REC_STREAM_FRAME: u32 = 92u;
const RETAINED_REC_PREV_STREAM_FRAME: u32 = 93u;
const RETAINED_REC_PALETTE_MASK: u32 = 94u;
const RETAINED_REC_GPU_FLAGS: u32 = 95u;

const RETAINED_VALID: u32 = 1u;
const RETAINED_PALETTE: u32 = 2u;
const RETAINED_CPU_DEFORMED: u32 = 4u;
const RETAINED_GPU_ARRAYS: u32 = 8u;
const RETAINED_LATCHED: u32 = 16u;
const RETAINED_BILLBOARD: u32 = 32u;
const RETAINED_NO_NORMAL: u32 = 64u;
const RETAINED_UV_NON_TEX: u32 = 128u;
const RETAINED_UV_NON_AFFINE: u32 = 256u;
const RETAINED_WATER: u32 = 512u;
const RETAINED_LATE: u32 = 1024u;
const RETAINED_BLENDED: u32 = 2048u;
const RETAINED_PREV_WORLD_VALID: u32 = 4096u;
const RETAINED_FALLBACK: u32 = 32768u;
const RETAINED_NEGATIVE_DETERMINANT: u32 = 65536u;
const RETAINED_NO_COLOR: u32 = 131072u;
const RETAINED_ALPHA_UV_IS_UV0: u32 = 262144u;
const RETAINED_ALPHA_UV_CLAMPED: u32 = 1u;
const RETAINED_NO_MATERIAL: u32 = 0xFFFFu;

const RETAINED_VTX_HAS_NORMAL: u32 = 0x01000000u;
const RETAINED_VTX_HAS_COLOR: u32 = 0x02000000u;

const RETAINED_FRAME_INDEX: u32 = 0u;
const RETAINED_FRAME_FLAGS: u32 = 1u;
const RETAINED_FRAME_SERIAL: u32 = 2u;
const RETAINED_FRAME_BREAK_COUNT: u32 = 3u;
const RETAINED_FRAME_CAM_HI: u32 = 4u;
const RETAINED_FRAME_CAM_LO: u32 = 8u;
const RETAINED_FRAME_VIEW: u32 = 12u;
const RETAINED_FRAME_INV_VIEW: u32 = 24u;
const RETAINED_FRAME_PREV_CAM_HI: u32 = 36u;
const RETAINED_FRAME_PREV_CAM_LO: u32 = 40u;
const RETAINED_FRAME_PREV_VIEW: u32 = 44u;
const RETAINED_FRAME_LATE_FRAME: u32 = 56u;
const RETAINED_CAMERA_VALID: u32 = 1u;
const RETAINED_RESET: u32 = 2u;

const RETAINED_PALETTE_MATRIX_WORDS: u32 = 12u;
const RETAINED_PALETTE_NORMAL_WORD: u32 = 120u;
const RETAINED_PALETTE_BANK_WORDS: u32 = 240u;
const RETAINED_PALETTE_LAST_SLOT: u32 = 9u;

const RETAINED_CLUSTER_WORDS: u32 = 16u;
const RETAINED_CLUSTER_COUNT_MASK: u32 = 0x00FFFFFFu;
const RETAINED_CLUSTER_STATE_SHIFT: u32 = 24u;
const RETAINED_CLUSTER_REASON_SHIFT: u32 = 28u;
const RETAINED_CLUSTERS_READY: u32 = 3u;

struct RetainedVertex {
  pos: vec3f,
  nrm: vec3f,
  has_nrm: bool,
  uv0: vec2f,
  alpha_uv_fixed: vec2i,
  alpha: f32,
  luma: f32,
  disp: f32,
  has_clr: bool,
}

struct RetainedCamera {
  frame: u32,
  flags: u32,
  serial: u32,
  break_count: u32,
  hi: vec3f,
  lo: vec3f,
  view: mat3x4f,
  inv_view: mat3x4f,
  prev_hi: vec3f,
  prev_lo: vec3f,
  prev_view: mat3x4f,
  late_frame: u32,
}

struct RetainedWorld {
  rows: mat3x4f,
  lo: vec3f,
}

struct RetainedCluster {
  ref_word: u32,
  tri_word: u32,
  vertex_count: u32,
  triangle_count: u32,
  bounds_space: u32,
  cone_axis: vec3f,
  cone_cutoff: f32,
  center: vec3f,
  radius: f32,
  aabb_min: vec3f,
  aabb_max: vec3f,
}

fn retained_oct_decode(w: u32) -> vec3f {
  let e = unpack2x16snorm(w);
  var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

fn retained_ordered_to_f32(u: u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7FFFFFFFu, (u & 0x80000000u) != 0u));
}

fn retained_unpack_vertex(w: array<u32, 8>) -> RetainedVertex {
  var v: RetainedVertex;
  v.pos = vec3f(bitcast<f32>(w[0]), bitcast<f32>(w[1]), bitcast<f32>(w[2]));
  let aux = w[7];
  v.has_nrm = (aux & RETAINED_VTX_HAS_NORMAL) != 0u;
  v.nrm = select(vec3f(0.0), retained_oct_decode(w[3]), v.has_nrm);
  v.uv0 = vec2f(bitcast<f32>(w[4]), bitcast<f32>(w[5]));
  v.alpha_uv_fixed = vec2i(bitcast<i32>(w[6] << 16u) >> 16u, bitcast<i32>(w[6]) >> 16u);
  v.has_clr = (aux & RETAINED_VTX_HAS_COLOR) != 0u;
  v.alpha = f32(aux & 0xFFu) / 255.0;
  v.luma = f32((aux >> 8u) & 0xFFu) / 255.0;
  v.disp = f32((aux >> 16u) & 0xFFu) / 255.0;
  return v;
}

fn retained_alpha_uv(v: RetainedVertex, flags: u32) -> vec2f {
  if ((flags & RETAINED_ALPHA_UV_IS_UV0) != 0u) {
    return v.uv0;
  }
  return vec2f(v.alpha_uv_fixed) / 4096.0;
}

fn retained_vec4(p: ptr<storage, array<u32>, read>, word: u32) -> vec4f {
  return vec4f(bitcast<f32>(p[word]), bitcast<f32>(p[word + 1u]), bitcast<f32>(p[word + 2u]),
               bitcast<f32>(p[word + 3u]));
}

fn retained_mat3x4(p: ptr<storage, array<u32>, read>, word: u32) -> mat3x4f {
  return mat3x4f(retained_vec4(p, word), retained_vec4(p, word + 4u), retained_vec4(p, word + 8u));
}

fn retained_camera(frame: ptr<storage, array<u32>, read>) -> RetainedCamera {
  var cam: RetainedCamera;
  cam.frame = frame[RETAINED_FRAME_INDEX];
  cam.flags = frame[RETAINED_FRAME_FLAGS];
  cam.serial = frame[RETAINED_FRAME_SERIAL];
  cam.break_count = frame[RETAINED_FRAME_BREAK_COUNT];
  cam.hi = retained_vec4(frame, RETAINED_FRAME_CAM_HI).xyz;
  cam.lo = retained_vec4(frame, RETAINED_FRAME_CAM_LO).xyz;
  cam.view = retained_mat3x4(frame, RETAINED_FRAME_VIEW);
  cam.inv_view = retained_mat3x4(frame, RETAINED_FRAME_INV_VIEW);
  cam.prev_hi = retained_vec4(frame, RETAINED_FRAME_PREV_CAM_HI).xyz;
  cam.prev_lo = retained_vec4(frame, RETAINED_FRAME_PREV_CAM_LO).xyz;
  cam.prev_view = retained_mat3x4(frame, RETAINED_FRAME_PREV_VIEW);
  cam.late_frame = frame[RETAINED_FRAME_LATE_FRAME];
  return cam;
}

fn retained_record_word(inst: ptr<storage, array<u32>, read>, slot: u32, word: u32) -> u32 {
  return inst[slot * RETAINED_RECORD_WORDS + word];
}

fn retained_world_at(inst: ptr<storage, array<u32>, read>, slot: u32, word: u32) -> RetainedWorld {
  let base = slot * RETAINED_RECORD_WORDS + word;
  var w: RetainedWorld;
  w.rows = retained_mat3x4(inst, base);
  w.lo = retained_vec4(inst, base + 12u).xyz;
  return w;
}

fn retained_world(inst: ptr<storage, array<u32>, read>, slot: u32) -> RetainedWorld {
  return retained_world_at(inst, slot, RETAINED_REC_WORLD);
}

fn retained_prev_world(inst: ptr<storage, array<u32>, read>, slot: u32, frame: u32) -> RetainedWorld {
  let flags = retained_record_word(inst, slot, RETAINED_REC_FLAGS);
  let fresh = retained_record_word(inst, slot, RETAINED_REC_WORLD_FRAME) == frame;
  if (fresh && (flags & RETAINED_PREV_WORLD_VALID) != 0u) {
    return retained_world_at(inst, slot, RETAINED_REC_PREV_WORLD);
  }
  return retained_world(inst, slot);
}

fn retained_world_rel(w: RetainedWorld, cam_hi: vec3f, cam_lo: vec3f) -> mat3x4f {
  let t = (vec3f(w.rows[0].w, w.rows[1].w, w.rows[2].w) - cam_hi) + (w.lo - cam_lo);
  return mat3x4f(vec4f(w.rows[0].xyz, t.x), vec4f(w.rows[1].xyz, t.y), vec4f(w.rows[2].xyz, t.z));
}

fn retained_normal_matrix(inst: ptr<storage, array<u32>, read>, slot: u32) -> mat3x4f {
  return retained_mat3x4(inst, slot * RETAINED_RECORD_WORDS + RETAINED_REC_NORMAL);
}

fn retained_palette_matrix(pal: ptr<storage, array<u32>, read>, inst: ptr<storage, array<u32>, read>, slot: u32,
                           bone: u32, prev: bool, normal: bool) -> mat3x4f {
  let first = retained_record_word(inst, slot, RETAINED_REC_PALETTE_FIRST);
  let bank = first + select(0u, RETAINED_PALETTE_BANK_WORDS, prev) + select(0u, RETAINED_PALETTE_NORMAL_WORD, normal);
  return retained_mat3x4(pal, bank + min(bone, RETAINED_PALETTE_LAST_SLOT) * RETAINED_PALETTE_MATRIX_WORDS);
}

fn retained_prev_is_last_frame(inst: ptr<storage, array<u32>, read>, slot: u32, frame: u32) -> bool {
  return retained_record_word(inst, slot, RETAINED_REC_STREAM_FRAME) == frame &&
         retained_record_word(inst, slot, RETAINED_REC_PREV_STREAM_FRAME) == frame - 1u;
}

fn retained_aabb(inst: ptr<storage, array<u32>, read>, slot: u32, prev: bool) -> array<vec3f, 2> {
  let base = slot * RETAINED_RECORD_WORDS + select(RETAINED_REC_CUR_AABB, RETAINED_REC_PREV_AABB, prev);
  let lo = vec3f(retained_ordered_to_f32(inst[base]), retained_ordered_to_f32(inst[base + 1u]),
                 retained_ordered_to_f32(inst[base + 2u]));
  let hi = vec3f(retained_ordered_to_f32(inst[base + 3u]), retained_ordered_to_f32(inst[base + 4u]),
                 retained_ordered_to_f32(inst[base + 5u]));
  return array<vec3f, 2>(lo, hi);
}

fn retained_uv_xform(inst: ptr<storage, array<u32>, read>, slot: u32, alpha: bool) -> array<vec3f, 2> {
  let base = slot * RETAINED_RECORD_WORDS + select(RETAINED_REC_UV_BASE, RETAINED_REC_UV_ALPHA, alpha);
  let r0 = vec3f(bitcast<f32>(inst[base]), bitcast<f32>(inst[base + 1u]), bitcast<f32>(inst[base + 2u]));
  let r1 = vec3f(bitcast<f32>(inst[base + 3u]), bitcast<f32>(inst[base + 4u]), bitcast<f32>(inst[base + 5u]));
  return array<vec3f, 2>(r0, r1);
}

fn retained_uv(xform: array<vec3f, 2>, src: vec2f) -> vec2f {
  let p = vec3f(src, 1.0);
  return vec2f(dot(xform[0], p), dot(xform[1], p));
}

fn retained_cluster_range(inst: ptr<storage, array<u32>, read>, slot: u32) -> vec2u {
  return vec2u(retained_record_word(inst, slot, RETAINED_REC_CLUSTER_FIRST),
               retained_record_word(inst, slot, RETAINED_REC_CLUSTER_COUNT) & RETAINED_CLUSTER_COUNT_MASK);
}

fn retained_cluster(clusters: ptr<storage, array<u32>, read>, index: u32) -> RetainedCluster {
  let base = index * RETAINED_CLUSTER_WORDS;
  var c: RetainedCluster;
  c.ref_word = base + clusters[base];
  c.tri_word = base + clusters[base + 1u];
  let counts = clusters[base + 2u];
  c.vertex_count = counts & 0xFFu;
  c.triangle_count = (counts >> 8u) & 0xFFu;
  c.bounds_space = (counts >> 16u) & 0xFFu;
  let cone = unpack4x8snorm(clusters[base + 3u]);
  c.cone_axis = cone.xyz;
  c.cone_cutoff = cone.w;
  let sphere = vec4f(bitcast<f32>(clusters[base + 4u]), bitcast<f32>(clusters[base + 5u]),
                     bitcast<f32>(clusters[base + 6u]), bitcast<f32>(clusters[base + 7u]));
  c.center = sphere.xyz;
  c.radius = sphere.w;
  c.aabb_min = vec3f(bitcast<f32>(clusters[base + 8u]), bitcast<f32>(clusters[base + 9u]),
                     bitcast<f32>(clusters[base + 10u]));
  c.aabb_max = vec3f(bitcast<f32>(clusters[base + 12u]), bitcast<f32>(clusters[base + 13u]),
                     bitcast<f32>(clusters[base + 14u]));
  return c;
}

fn retained_cluster_vertex(clusters: ptr<storage, array<u32>, read>, c: RetainedCluster, i: u32) -> u32 {
  return clusters[c.ref_word + i];
}

fn retained_cluster_triangle(clusters: ptr<storage, array<u32>, read>, c: RetainedCluster, t: u32) -> vec3u {
  let w = clusters[c.tri_word + t];
  return vec3u(w & 0xFFu, (w >> 8u) & 0xFFu, (w >> 16u) & 0xFFu);
}

const RETAINED_MATERIAL_WORDS: u32 = 8u;
const RETAINED_ATLAS_PAGE: f32 = 4096.0;
const RETAINED_ATLAS_TILE: f32 = 128.0;
const RETAINED_ATLAS_TILES_PER_ROW: u32 = 32u;
const RETAINED_NO_TILE: u32 = 0xFFFFu;
const RETAINED_MAT_ALPHA_TEST: u32 = 1u;
const RETAINED_MAT_ALBEDO_NO_UV: u32 = 2u;
const RETAINED_MAT_ALPHA_NO_UV: u32 = 4u;
const RETAINED_MAT_ALBEDO_PENDING: u32 = 8u;
const RETAINED_MAT_ALPHA_PENDING: u32 = 16u;
const RETAINED_MAT_ALBEDO_MISSING: u32 = 32u;
const RETAINED_MAT_ALPHA_MISSING: u32 = 64u;
const RETAINED_ATTR_WORD_KNOWN: u32 = 1u;
const RETAINED_ATTR_CULL_NONE: u32 = 2u;
const RETAINED_WRAP_CLAMP: u32 = 0u;
const RETAINED_WRAP_REPEAT: u32 = 1u;
const RETAINED_WRAP_MIRROR: u32 = 2u;

struct RetainedMaterial {
  albedo_tile: u32,
  alpha_tile: u32,
  albedo_wrap: vec2u,
  alpha_wrap: vec2u,
  alpha_channel: u32,
  alpha_uses_uv0: bool,
  flags: u32,
  comp0: u32,
  comp1: u32,
  op: u32,
  ref0: u32,
  ref1: u32,
  registry: u32,
}

struct RetainedDrawAttributes {
  word: u32,
  const_color: vec4f,
  receiver: u32,
  word_known: bool,
  cull_none: bool,
}

fn retained_material(materials: ptr<storage, array<u32>, read>, id: u32) -> RetainedMaterial {
  let base = id * RETAINED_MATERIAL_WORDS;
  let w0 = materials[base];
  let w1 = materials[base + 1u];
  let w2 = materials[base + 2u];
  var m: RetainedMaterial;
  m.albedo_tile = w0 & 0xFFFFu;
  m.alpha_tile = w0 >> 16u;
  m.albedo_wrap = vec2u(w1 & 3u, (w1 >> 2u) & 3u);
  m.alpha_wrap = vec2u((w1 >> 4u) & 3u, (w1 >> 6u) & 3u);
  m.alpha_channel = (w1 >> 8u) & 3u;
  m.alpha_uses_uv0 = ((w1 >> 10u) & 1u) != 0u;
  m.flags = w1 >> 16u;
  m.comp0 = w2 & 7u;
  m.comp1 = (w2 >> 3u) & 7u;
  m.op = (w2 >> 6u) & 3u;
  m.ref0 = (w2 >> 8u) & 0xFFu;
  m.ref1 = (w2 >> 16u) & 0xFFu;
  m.registry = materials[base + 3u];
  return m;
}

fn retained_draw_attributes(inst: ptr<storage, array<u32>, read>, slot: u32) -> RetainedDrawAttributes {
  var a: RetainedDrawAttributes;
  a.word = retained_record_word(inst, slot, RETAINED_REC_DRAW_ATTRIBUTES);
  a.const_color = unpack4x8unorm(retained_record_word(inst, slot, RETAINED_REC_DRAW_ATTRIBUTES + 1u));
  let w2 = retained_record_word(inst, slot, RETAINED_REC_DRAW_ATTRIBUTES + 2u);
  a.receiver = w2 & 0xFFu;
  a.word_known = ((w2 >> 8u) & RETAINED_ATTR_WORD_KNOWN) != 0u;
  a.cull_none = ((w2 >> 8u) & RETAINED_ATTR_CULL_NONE) != 0u;
  return a;
}

fn atlas_wrap(c: f32, mode: u32) -> f32 {
  if (mode == RETAINED_WRAP_REPEAT) {
    return fract(c);
  }
  if (mode == RETAINED_WRAP_MIRROR) {
    let t = fract(c * 0.5) * 2.0;
    return select(t, 2.0 - t, t > 1.0);
  }
  return saturate(c);
}

fn atlas_uv(tile: u32, wrap_s: u32, wrap_t: u32, uv: vec2f) -> vec2f {
  let wrapped = vec2f(atlas_wrap(uv.x, wrap_s), atlas_wrap(uv.y, wrap_t));
  let texel = clamp(wrapped * RETAINED_ATLAS_TILE, vec2f(0.5), vec2f(RETAINED_ATLAS_TILE - 0.5));
  let origin = vec2f(f32(tile % RETAINED_ATLAS_TILES_PER_ROW), f32(tile / RETAINED_ATLAS_TILES_PER_ROW)) *
               RETAINED_ATLAS_TILE;
  return (origin + texel) / RETAINED_ATLAS_PAGE;
}

fn retained_alpha_compare(comp: u32, a: u32, limit: u32) -> bool {
  switch comp {
    case 0u: { return false; }
    case 1u: { return a < limit; }
    case 2u: { return a == limit; }
    case 3u: { return a <= limit; }
    case 4u: { return a > limit; }
    case 5u: { return a != limit; }
    case 6u: { return a >= limit; }
    default: { return true; }
  }
}

fn retained_alpha_pass(m: RetainedMaterial, a: f32) -> bool {
  if ((m.flags & RETAINED_MAT_ALPHA_TEST) == 0u) {
    return true;
  }
  let v = u32(round(saturate(a) * 255.0));
  let p0 = retained_alpha_compare(m.comp0, v, m.ref0);
  let p1 = retained_alpha_compare(m.comp1, v, m.ref1);
  switch m.op {
    case 0u: { return p0 && p1; }
    case 1u: { return p0 || p1; }
    case 2u: { return p0 != p1; }
    default: { return p0 == p1; }
  }
}

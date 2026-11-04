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

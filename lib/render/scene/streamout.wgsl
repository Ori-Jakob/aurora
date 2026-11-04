const BLOCK_PIECES: u32 = 1u;
const BLOCK_IDX_CHUNKS: u32 = 3u;
const BLOCK_WRITE_JOBS: u32 = 4u;
const BLOCK_ROLL_JOBS: u32 = 6u;
const PIECE_WORDS: u32 = 32u;
const PIECE_OUT_INDEX_FIRST: u32 = 19u;
const PIECE_INDEX_COUNT: u32 = 20u;
const PIECE_PRIM: u32 = 21u;
const PIECE_IDX_BYTE: u32 = 22u;
const PIECE_FLAGS: u32 = 24u;
const PIECE_INDEXED: u32 = 4u;
const CHUNK_WORDS: u32 = 2u;
const WRITE_JOB_WORDS: u32 = 4u;
const ROLL_JOB_WORDS: u32 = 8u;
const TARGET_FRAME: u32 = 1u;
const ROLL_FRESH: u32 = 1u;
const RECORD_WORDS: u32 = 96u;
const RECORD_CUR_AABB: u32 = 80u;
const RECORD_PREV_AABB: u32 = 86u;
const RECORD_STREAM_FRAME: u32 = 92u;
const RECORD_PREV_STREAM_FRAME: u32 = 93u;
const FRAME_FLAGS: u32 = 1u;
const FRAME_BREAK_COUNT: u32 = 3u;
const FRAME_RESET: u32 = 2u;
const RESET_BIT: u32 = 0x80000000u;
const EMPTY_MIN: u32 = 0xFFFFFFFFu;
const EMPTY_MAX: u32 = 0u;
const PRIM_QUADS: u32 = 0x80u;
const PRIM_TRIANGLESTRIP: u32 = 0x98u;
const PRIM_TRIANGLEFAN: u32 = 0xA0u;

struct StreamPush {
  block_word: u32,
  first: u32,
  count: u32,
  value: u32,
}

@group(0) @binding(1) var<storage, read> abuf: array<u32>;
@group(0) @binding(2) var<storage, read> ibuf: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> instances: array<u32>;
@group(0) @binding(7) var<storage, read_write> frame_words: array<u32>;
var<immediate> push: StreamPush;

fn ordered_from_f32(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn f32_from_ordered(u: u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7FFFFFFFu, (u & 0x80000000u) != 0u));
}

fn shifted(u: u32, delta: f32, empty: u32) -> u32 {
  if (u == empty || delta == 0.0) {
    return u;
  }
  return ordered_from_f32(f32_from_ordered(u) - delta);
}

@compute @workgroup_size(64)
fn write_jobs(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let hw = push.block_word;
  let job = hw + abuf[hw + BLOCK_WRITE_JOBS] + (push.first + wid.x) * WRITE_JOB_WORDS;
  let target_kind = abuf[job];
  let dst = abuf[job + 1u];
  let src = hw + abuf[job + 2u];
  let count = abuf[job + 3u];
  for (var i = li; i < count; i += 64u) {
    var v = abuf[src + i];
    if (target_kind == TARGET_FRAME) {
      let w = dst + i;
      if (w == FRAME_FLAGS && (push.value & RESET_BIT) != 0u) {
        v = v | FRAME_RESET;
      }
      if (w == FRAME_BREAK_COUNT) {
        v = push.value & ~RESET_BIT;
      }
      frame_words[w] = v;
    } else {
      instances[dst + i] = v;
    }
  }
}

@compute @workgroup_size(64)
fn roll_jobs(@builtin(global_invocation_id) gid: vec3u) {
  let hw = push.block_word;
  let j = push.first + gid.x;
  let live = j < push.count;
  var base = 0u;
  var fresh = true;
  var delta = vec3f(0.0);
  var cur: array<u32, 6>;
  var stream_frame = 0u;
  if (live) {
    let job = hw + abuf[hw + BLOCK_ROLL_JOBS] + j * ROLL_JOB_WORDS;
    base = abuf[job] * RECORD_WORDS;
    fresh = (abuf[job + 1u] & ROLL_FRESH) != 0u;
    delta = vec3f(bitcast<f32>(abuf[job + 4u]), bitcast<f32>(abuf[job + 5u]), bitcast<f32>(abuf[job + 6u]));
    for (var k = 0u; k < 6u; k++) {
      cur[k] = instances[base + RECORD_CUR_AABB + k];
    }
    stream_frame = instances[base + RECORD_STREAM_FRAME];
  }
  workgroupBarrier();
  if (!live) {
    return;
  }
  for (var k = 0u; k < 3u; k++) {
    let d = delta[k];
    instances[base + RECORD_PREV_AABB + k] = select(shifted(cur[k], d, EMPTY_MIN), EMPTY_MIN, fresh);
    instances[base + RECORD_PREV_AABB + 3u + k] = select(shifted(cur[3u + k], d, EMPTY_MAX), EMPTY_MAX, fresh);
    instances[base + RECORD_CUR_AABB + k] = EMPTY_MIN;
    instances[base + RECORD_CUR_AABB + 3u + k] = EMPTY_MAX;
  }
  instances[base + RECORD_PREV_STREAM_FRAME] = select(stream_frame, 0u, fresh);
  instances[base + RECORD_STREAM_FRAME] = push.value;
}

fn read_index(byte_offset: u32) -> u32 {
  return (ibuf[byte_offset >> 2u] >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
}

@compute @workgroup_size(64)
fn write_indices(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let hw = push.block_word;
  let chunk = hw + abuf[hw + BLOCK_IDX_CHUNKS] + (push.first + wid.x) * CHUNK_WORDS;
  let pw = hw + abuf[hw + BLOCK_PIECES] + abuf[chunk] * PIECE_WORDS;
  let tri = abuf[chunk + 1u] + li;
  if (tri * 3u >= abuf[pw + PIECE_INDEX_COUNT]) {
    return;
  }
  let prim = abuf[pw + PIECE_PRIM];
  let out = abuf[pw + PIECE_OUT_INDEX_FIRST] + tri * 3u;
  var t = vec3u(tri * 3u, tri * 3u + 1u, tri * 3u + 2u);
  if ((abuf[pw + PIECE_FLAGS] & PIECE_INDEXED) != 0u) {
    let b = abuf[pw + PIECE_IDX_BYTE] + tri * 6u;
    t = vec3u(read_index(b), read_index(b + 2u), read_index(b + 4u));
  } else if (prim == PRIM_QUADS) {
    let q = (tri >> 1u) * 4u;
    t = select(vec3u(q + 2u, q + 3u, q), vec3u(q, q + 1u, q + 2u), (tri & 1u) == 0u);
  } else if (prim == PRIM_TRIANGLEFAN) {
    t = vec3u(0u, tri + 1u, tri + 2u);
  } else if (prim == PRIM_TRIANGLESTRIP) {
    let v = tri + 2u;
    t = select(vec3u(v - 1u, v - 2u, v), vec3u(v - 2u, v - 1u, v), (tri & 1u) == 0u);
  }
  out_indices[out] = t.x;
  out_indices[out + 1u] = t.y;
  out_indices[out + 2u] = t.z;
}

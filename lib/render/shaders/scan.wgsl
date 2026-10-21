struct ScanImmediates {
  a: vec4u, // mode (0 reduce, 1 down, 2 args, 3 zero), flags, count, level
  b: vec4u, // link, live word, keep mask, args word
  c: vec4u, // args kind | field << 8, scale, limit, record[4]
  d: vec4u, // record[0..3]
};
var<immediate> imm: ScanImmediates;
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(3) var<storage, read_write> live: array<u32>;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;
var<workgroup> sums: array<u32, 128>;

const MODE_REDUCE = 0u;
const MODE_DOWN = 1u;
const MODE_ARGS = 2u;
const MODE_ZERO = 3u;
const FLAG_LEVEL0 = 1u;
const FLAG_COMPACT = 2u;
const FLAG_LIVE = 4u;
const FLAG_PARENT = 8u;
const FLAG_TOTAL = 16u;
const KIND_DISPATCH = 1u;
const KIND_DRAW = 2u;
const KIND_DRAW_INDEXED = 3u;
const ITEMS = 8u;
const TILE = 1024u;

fn record_words(kind: u32) -> u32 {
  if (kind == KIND_DISPATCH) {
    return 3u;
  }
  if (kind == KIND_DRAW) {
    return 4u;
  }
  if (kind == KIND_DRAW_INDEXED) {
    return 5u;
  }
  return 0u;
}

fn put_args(index: u32, value: u32) {
  if (index < arrayLength(&args)) {
    args[index] = value;
  }
}

fn write_args() {
  let kind = imm.c.x & 0xffu;
  let field = (imm.c.x >> 8u) & 0xffu;
  let scale = imm.c.y;
  let base = imm.b.w;
  var total = 0u;
  if (0u < arrayLength(&scratch)) {
    total = scratch[0];
  }
  let t = min(total, imm.c.z);
  if (kind == KIND_DISPATCH) {
    put_args(base, t / scale + select(0u, 1u, t % scale != 0u));
    put_args(base + 1u, 1u);
    put_args(base + 2u, 1u);
    return;
  }
  let words = record_words(kind);
  let value = t * scale;
  put_args(base, select(imm.d.x, value, field == 0u));
  put_args(base + 1u, select(imm.d.y, value, field == 1u));
  put_args(base + 2u, select(imm.d.z, value, field == 2u));
  put_args(base + 3u, select(imm.d.w, value, field == 3u));
  if (words == 5u) {
    put_args(base + 4u, select(imm.c.w, value, field == 4u));
  }
}

fn write_zero() {
  if (arrayLength(&scratch) >= 2u) {
    scratch[0] = 0u;
    scratch[1] = 0u;
  }
  let words = record_words(imm.c.x & 0xffu);
  for (var k = 0u; k < words; k++) {
    put_args(imm.b.w + k, 0u);
  }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let mode = imm.a.x;
  let flags = imm.a.y;
  let count = imm.a.z;
  let level = imm.a.w;
  let level0 = (flags & FLAG_LEVEL0) != 0u;
  let keep = (flags & FLAG_COMPACT) != 0u;
  var n = count;
  if (level0 && (flags & FLAG_LIVE) != 0u && imm.b.y < arrayLength(&live)) {
    n = min(count, live[imm.b.y]);
  }
  let first = wg.x * TILE + lid * ITEMS;
  var v: array<u32, 8>;
  var own = 0u;
  for (var k = 0u; k < ITEMS; k++) {
    let i = first + k;
    var x = 0u;
    if (i < n) {
      if (level0) {
        if (i < arrayLength(&src)) {
          x = src[i];
          if (keep) {
            x = select(0u, 1u, (x & imm.b.z) != 0u);
          }
        }
      } else if (level + i < arrayLength(&scratch)) {
        x = scratch[level + i];
      }
    }
    v[k] = x;
    own += x;
  }
  sums[lid] = own;
  workgroupBarrier();

  var t = 0u;
  if (lid >= 1u) {
    t = sums[lid - 1u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 2u) {
    t = sums[lid - 2u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 4u) {
    t = sums[lid - 4u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 8u) {
    t = sums[lid - 8u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 16u) {
    t = sums[lid - 16u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 32u) {
    t = sums[lid - 32u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  t = 0u;
  if (lid >= 64u) {
    t = sums[lid - 64u];
  }
  workgroupBarrier();
  sums[lid] += t;
  workgroupBarrier();

  let inclusive = sums[lid];
  let link = imm.b.x;
  if (mode == MODE_REDUCE) {
    if (lid == 127u && link + wg.x < arrayLength(&scratch)) {
      scratch[link + wg.x] = inclusive;
    }
    return;
  }
  if (mode == MODE_DOWN) {
    var running = inclusive - own;
    if ((flags & FLAG_PARENT) != 0u && link + wg.x < arrayLength(&scratch)) {
      running += scratch[link + wg.x];
    }
    for (var k = 0u; k < ITEMS; k++) {
      let i = first + k;
      if (level0) {
        if (keep) {
          if (v[k] != 0u && running < arrayLength(&dst)) {
            dst[running] = i;
          }
        } else if (i < count && i < arrayLength(&dst)) {
          dst[i] = running;
        }
      } else if (i < count && level + i < arrayLength(&scratch)) {
        scratch[level + i] = running;
      }
      running += v[k];
    }
    if ((flags & FLAG_TOTAL) != 0u && lid == 127u && 0u < arrayLength(&scratch)) {
      scratch[0] = running;
    }
    if (level0 && wg.x == 0u && lid == 0u && 1u < arrayLength(&scratch)) {
      scratch[1] = n;
    }
    return;
  }
  if (wg.x != 0u || lid != 0u) {
    return;
  }
  if (mode == MODE_ARGS) {
    write_args();
  } else if (mode == MODE_ZERO) {
    write_zero();
  }
}

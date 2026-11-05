//#import aurora/srgb

const ATLAS_TILE: u32 = 128u;
const ATLAS_TILES_PER_ROW: u32 = 32u;
const ATLAS_MEAN_THREADS: u32 = 256u;

struct AtlasPush {
  a: u32,
  b: u32,
  c: u32,
  d: u32,
}

var<immediate> push: AtlasPush;

override SRGB_SOURCE: bool = false;

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_blit(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let local = position.xy - vec2f(f32(push.a), f32(push.b));
  let level = f32(push.c);
  let taps = max(push.d, 1u);
  var sum = vec4f(0.0);
  for (var j = 0u; j < taps; j++) {
    for (var i = 0u; i < taps; i++) {
      let offset = (vec2f(f32(i), f32(j)) + 0.5) / f32(taps) - 0.5;
      sum += textureSampleLevel(source, source_sampler, (local + offset) / f32(ATLAS_TILE), level);
    }
  }
  let c = sum / f32(taps * taps);
  if (SRGB_SOURCE) {
    return vec4f(linear_to_srgb(c.rgb), c.a);
  }
  return c;
}

@group(0) @binding(0) var page: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> means: array<u32>;

var<workgroup> partial: array<vec4f, ATLAS_MEAN_THREADS>;

@compute @workgroup_size(ATLAS_MEAN_THREADS)
fn cs_mean(@builtin(local_invocation_index) li: u32) {
  let tile = push.a;
  let origin = vec2u(tile % ATLAS_TILES_PER_ROW, tile / ATLAS_TILES_PER_ROW) * ATLAS_TILE;
  var sum = vec4f(0.0);
  for (var k = li; k < ATLAS_TILE * ATLAS_TILE; k += ATLAS_MEAN_THREADS) {
    sum += textureLoad(page, origin + vec2u(k % ATLAS_TILE, k / ATLAS_TILE), 0);
  }
  partial[li] = sum;
  workgroupBarrier();
  for (var stride = ATLAS_MEAN_THREADS / 2u; stride > 0u; stride >>= 1u) {
    if (li < stride) {
      partial[li] += partial[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    means[tile] = pack4x8unorm(partial[0] / f32(ATLAS_TILE * ATLAS_TILE));
  }
}

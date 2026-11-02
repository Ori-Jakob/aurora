//#import aurora/camera

// ao.final multiplies stored-space colour; linear consumers convert with ao_linear_factor
struct AoParams {
  camera: StageCamera,
  effect_radius: f32,
  thickness: f32,
  falloff_range: f32,
  strength: f32,
  mip_offset: f32,
  distribution_power: f32,
  slice_count: u32,
  steps_per_slice: u32,
  noise_index: u32,
  flags: u32,
  denoise_beta: f32,
  depth_reject: f32,
  max_samples: f32,
  radius_metres: f32,
  thickness_metres: f32,
  min_visibility: f32,
}

@group(0) @binding(1) var<uniform> ao_params: AoParams;

const AO_SKY_DEPTH: f32 = 1e30;
const AO_MIP_LEVELS: u32 = 5u;
const AO_RADIUS_METRES: f32 = 0.73;
const AO_THICKNESS_METRES: f32 = 0.25;
const AO_FALLOFF_RANGE: f32 = 0.615;
const AO_DISTRIBUTION_POWER: f32 = 2.0;
const AO_MIP_OFFSET: f32 = 3.3;
const AO_FINAL_POWER: f32 = 1.0;
const AO_MIN_VISIBILITY: f32 = 0.2;
const AO_DENOISE_BETA: f32 = 1.2;
const AO_DEPTH_REJECT: f32 = 0.1;
const AO_MAX_SAMPLES: f32 = 16.0;
const AO_LAG_SAMPLES: f32 = 2.0;
const AO_MOVING_SAMPLES: f32 = 4.0;
const AO_CLIP_GAMMA: f32 = 2.0;
const AO_CLIP_FLOOR: f32 = 2.0 / 255.0;
const AO_MOTION_CONTRAST_PX: f32 = 0.5;

const AO_FLAG_TEMPORAL_OFF: u32 = 1u;
const AO_FLAG_IDENTITY_FORCED: u32 = 2u;
const AO_FLAG_NORMAL_FROM_DEPTH: u32 = 4u;
const AO_FLAG_SECTOR_CEIL: u32 = 8u;

fn ao_divisor(full: vec2u, grid: vec2u) -> u32 {
  let cells = max(grid, vec2u(1u));
  let d = (full + cells - 1u) / cells;
  return max(max(d.x, d.y), 1u);
}

fn ao_rep_texel(p: vec2u, d: u32, full: vec2u) -> vec2u {
  return min(d * p + vec2u(d / 2u), max(full, vec2u(1u)) - 1u);
}

fn ao_is_sky(z: f32) -> bool {
  return z >= 0.5 * AO_SKY_DEPTH;
}

fn ao_linear_factor(a: f32) -> f32 {
  return pow(a, 2.2);
}

const AO_SECTOR_COUNT: u32 = 32u;
const AO_RAW_SCALE: f32 = 1.5;
const AO_HILBERT_WIDTH: u32 = 64u;
const AO_NOISE_PERIOD: u32 = 64u;
const AO_NOISE_STRIDE: u32 = 288u;
const AO_R2_STEPS: vec2u = vec2u(3242174889u, 2447445414u);

// Round activates a sector only when half of it is covered, so near-horizon slivers do not darken open ground
fn ao_sector_bits(t0: f32, t1: f32, ceil_mode: bool) -> u32 {
  let lo = clamp(t0, 0.0, 1.0);
  let hi = clamp(t1, 0.0, 1.0);
  var start: u32;
  var end: u32;
  if (ceil_mode) {
    start = min(u32(lo * 32.0), 31u);
    end = min(start + u32(max(ceil((hi - lo) * 32.0), 0.0)), 32u);
  } else {
    start = u32(round(lo * 32.0));
    end = u32(round(hi * 32.0));
  }
  if (end <= start) {
    return 0u;
  }
  return (0xFFFFFFFFu >> (32u - (end - start))) << start;
}

fn ao_hilbert_index(px: u32, py: u32) -> u32 {
  var x = px;
  var y = py;
  var index = 0u;
  for (var level = AO_HILBERT_WIDTH / 2u; level > 0u; level = level / 2u) {
    let rx = select(0u, 1u, (x & level) != 0u);
    let ry = select(0u, 1u, (y & level) != 0u);
    index += level * level * ((3u * rx) ^ ry);
    if (ry == 0u) {
      if (rx == 1u) {
        x = AO_HILBERT_WIDTH - 1u - x;
        y = AO_HILBERT_WIDTH - 1u - y;
      }
      let swap = x;
      x = y;
      y = swap;
    }
  }
  return index;
}

fn ao_noise(p: vec2u, noise_index: u32) -> vec2f {
  let index = ao_hilbert_index(p.x, p.y) + AO_NOISE_STRIDE * (noise_index % AO_NOISE_PERIOD);
  return fract(vec2f(0.5) + vec2f(index * AO_R2_STEPS) * (1.0 / 4294967296.0));
}

fn ao_raw_visibility(t: vec4f) -> f32 {
  return saturate(t.r * AO_RAW_SCALE);
}

// Strength scales the darkening; above 1 it deepens mid-tones but stops at the floor
fn ao_apply_strength(v: f32, k: f32, floor_value: f32) -> f32 {
  return max(1.0 - k * (1.0 - v), 1.0 - min(k, 1.0) * (1.0 - floor_value));
}

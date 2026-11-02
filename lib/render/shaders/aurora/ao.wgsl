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

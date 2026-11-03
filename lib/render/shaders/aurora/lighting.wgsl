// Frozen light.list/light.grid v1; M14, M16 and M18 import this, so fields only append
//#import aurora/camera

struct LightListHeader {
  dir_count: u32,
  local_count: u32,
  version: u32,
  flags: u32,
  slice_near: f32,
  slice_log_scale: f32,
  units_per_metre: f32,
  pad0: u32,
}

struct GpuLight {
  v0: vec4f,
  v1: vec4f,
  v2: vec4f,
  v3: vec4f,
  v4: vec4u,
}

struct LightList {
  header: LightListHeader,
  lights: array<GpuLight, 256>,
}

struct LightClusterParams {
  camera: StageCamera,
  flags: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

struct LightCounts {
  directional: u32,
  locals: u32,
  words: u32,
}

const LIGHT_TILES_X: u32 = 16u;
const LIGHT_TILES_Y: u32 = 9u;
const LIGHT_SLICES: u32 = 24u;
const LIGHT_CLUSTER_COUNT: u32 = 3456u;
const LIGHT_MASK_WORDS: u32 = 8u;
const LIGHT_MAX_LIGHTS: u32 = 256u;
const LIGHT_MAX_DIRECTIONAL: u32 = 16u;
const LIGHT_RADIUS_INFLATE: f32 = 1e-3;
const LIGHT_DEPTH_INFLATE: f32 = 1e-5;

const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

const LIGHT_FLAG_NO_SHADOW: u32 = 1u;
const LIGHT_FLAG_NO_SPECULAR: u32 = 2u;
const LIGHT_FLAG_FLICKER: u32 = 4u;
const LIGHT_FLAG_CAMERA_FILL: u32 = 8u;
const LIGHT_FLAG_AFFECTS_TRANSLUCENT: u32 = 16u;
const LIGHT_FLAG_CLOUD_SHADOWED: u32 = 32u;
const LIGHT_FLAG_KEY_ROOM: u32 = 64u;
const LIGHT_FLAG_LANTERN: u32 = 128u;
const LIGHT_FLAG_NO_VOLUMETRIC: u32 = 256u;

const LIGHT_HEADER_TRUNCATED: u32 = 1u;
const LIGHT_HEADER_MASK_WIDENED: u32 = 2u;
const LIGHT_HEADER_STRESS: u32 = 4u;
const LIGHT_HEADER_SKIP_MASK: u32 = 8u;

const LIGHT_PARAMS_BRUTE_FORCE: u32 = 1u;

fn light_counts(h: LightListHeader) -> LightCounts {
  let directional = min(h.dir_count, LIGHT_MAX_DIRECTIONAL);
  let locals = min(h.local_count, LIGHT_MAX_LIGHTS - directional);
  return LightCounts(directional, locals, min((locals + 31u) / 32u, LIGHT_MASK_WORDS));
}

fn light_slice(h: LightListHeader, z: f32) -> u32 {
  if (!(z > h.slice_near)) {
    return 0u;
  }
  let s = floor(log(z / h.slice_near) * h.slice_log_scale);
  return min(u32(clamp(s, 0.0, f32(LIGHT_SLICES - 1u))), LIGHT_SLICES - 1u);
}

fn light_tile(p: u32, size: u32, tiles: u32) -> u32 {
  return min(((2u * p + 1u) * tiles) / (2u * max(size, 1u)), tiles - 1u);
}

fn light_cluster(px: vec2u, dims: vec2u, z: f32, h: LightListHeader) -> u32 {
  let tx = light_tile(px.x, dims.x, LIGHT_TILES_X);
  let ty = light_tile(px.y, dims.y, LIGHT_TILES_Y);
  return (light_slice(h, z) * LIGHT_TILES_Y + ty) * LIGHT_TILES_X + tx;
}

fn light_type(l: GpuLight) -> u32 {
  return l.v4.x & 0xffu;
}

fn light_receiver_byte(l: GpuLight) -> u32 {
  return (l.v4.x >> 8u) & 0xffu;
}

fn light_flags(l: GpuLight) -> u32 {
  return l.v4.x >> 16u;
}

fn light_distance_attenuation(l: GpuLight, d: f32) -> f32 {
  let r = l.v0.w;
  let p = l.v3.x;
  if (p <= 0.0) {
    let x = d / r;
    let w = saturate(1.0 - x * x * x * x);
    return w * w;
  }
  let f = 1.0 / (1.0 + l.v2.w * pow(max(d, 0.0), p));
  return max(0.0, (f - 1.0 / 255.0) / (1.0 - 1.0 / 255.0));
}

fn light_spot_attenuation(l: GpuLight, to_light: vec3f) -> f32 {
  if (light_type(l) != LIGHT_TYPE_SPOT) {
    return 1.0;
  }
  let t = saturate(dot(l.v1.xyz, -to_light) * l.v1.w + l.v3.y);
  return t * t;
}

fn light_category_match(l: GpuLight, cat: u32) -> bool {
  let mask = light_receiver_byte(l);
  return mask == 0u || (mask & cat) != 0u;
}

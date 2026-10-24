const AURORA_HZB_REL_MARGIN: f32 = 0x1p-18f;
const AURORA_HZB_ABS_MARGIN: f32 = 0x1p-30f;
const AURORA_HZB_MIN_W: f32 = 1e-6;
const AURORA_HZB_MAX_F32: f32 = 0x1.fffffep+127f;

struct AuroraHzbBounds {
  ndc_min: vec2f,
  ndc_max: vec2f,
  nearest: f32,
  valid: bool,
}

fn aurora_hzb_bounds() -> AuroraHzbBounds {
  return AuroraHzbBounds(vec2f(AURORA_HZB_MAX_F32), vec2f(-AURORA_HZB_MAX_F32), -AURORA_HZB_MAX_F32, true);
}

fn aurora_hzb_bounds_add(bounds: AuroraHzbBounds, clip: vec4f) -> AuroraHzbBounds {
  var out = bounds;
  if (!(clip.w > AURORA_HZB_MIN_W && all(abs(clip.xyz) <= vec3f(AURORA_HZB_MAX_F32)))) {
    out.valid = false;
    return out;
  }
  let ndc = clip.xyz / clip.w;
  out.ndc_min = min(out.ndc_min, ndc.xy);
  out.ndc_max = max(out.ndc_max, ndc.xy);
  out.nearest = max(out.nearest, ndc.z);
  return out;
}

fn aurora_hzb_bounds_aabb(clip_from_local: mat4x4f, bmin: vec3f, bmax: vec3f) -> AuroraHzbBounds {
  var bounds = aurora_hzb_bounds();
  for (var i = 0u; i < 8u; i++) {
    let corner = select(bmin, bmax, vec3<bool>((i & 1u) != 0u, (i & 2u) != 0u, (i & 4u) != 0u));
    bounds = aurora_hzb_bounds_add(bounds, clip_from_local * vec4f(corner, 1.0));
  }
  return bounds;
}

fn aurora_hzb_occluded_rect(hzb: texture_2d<f32>, rect_min: vec2u, rect_max: vec2u, nearest: f32) -> bool {
  if (any(rect_min > rect_max)) {
    return false;
  }
  let extent = max(rect_max.x - rect_min.x, rect_max.y - rect_min.y) + 1u;
  var level = 0u;
  if (extent > 1u) {
    level = min(firstLeadingBit(extent - 1u), textureNumLevels(hzb) - 1u);
  }
  if (level > 0u) {
    let finer = textureDimensions(hzb, level - 1u) - 1u;
    if (all(min(rect_max >> vec2u(level), finer) - min(rect_min >> vec2u(level), finer) <= vec2u(1u))) {
      level -= 1u;
    }
  }
  let size = textureDimensions(hzb, level);
  let lo = min(rect_min >> vec2u(level + 1u), size - 1u);
  let hi = min(rect_max >> vec2u(level + 1u), size - 1u);
  if (any(hi - lo > vec2u(1u))) {
    return false;
  }
  let a = textureLoad(hzb, lo, level).r;
  let b = textureLoad(hzb, vec2u(hi.x, lo.y), level).r;
  let c = textureLoad(hzb, vec2u(lo.x, hi.y), level).r;
  let d = textureLoad(hzb, hi, level).r;
  return nearest < min(min(a, b), min(c, d));
}

fn aurora_hzb_occluded(hzb: texture_2d<f32>, bounds: AuroraHzbBounds, viewport: vec4f, depth_range: vec2f,
                       efb_size: vec2u) -> bool {
  if (!bounds.valid || any(bounds.ndc_min > bounds.ndc_max) || any(efb_size == vec2u(0u))) {
    return false;
  }
  let x0 = viewport.x + (bounds.ndc_min.x * 0.5 + 0.5) * viewport.z;
  let x1 = viewport.x + (bounds.ndc_max.x * 0.5 + 0.5) * viewport.z;
  let y0 = viewport.y + (0.5 - bounds.ndc_max.y * 0.5) * viewport.w;
  let y1 = viewport.y + (0.5 - bounds.ndc_min.y * 0.5) * viewport.w;
  let limit = vec2f(efb_size - 1u);
  let lo = clamp(floor(vec2f(x0, y0)) - 1.0, vec2f(0.0), limit);
  let hi = clamp(floor(vec2f(x1, y1)) + 1.0, vec2f(0.0), limit);
  let window = depth_range.x + max(bounds.nearest, 0.0) * (depth_range.y - depth_range.x);
  let nearest = window * (1.0 + AURORA_HZB_REL_MARGIN) + AURORA_HZB_ABS_MARGIN;
  return aurora_hzb_occluded_rect(hzb, vec2u(lo), vec2u(hi), nearest);
}

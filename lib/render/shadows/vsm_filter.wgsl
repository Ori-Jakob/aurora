//#import aurora/vsm

const VSM_FILTER_FALLBACK_LEVELS: u32 = 2u;
const VSM_FILTER_NORMAL_BASE: f32 = 0.5;
const VSM_FILTER_NORMAL_SLOPE: f32 = 1.5;
const VSM_FILTER_TAP_INSET: f32 = 0.5;

struct VsmFilterParams {
  taps: u32,
  normal_scale: f32,
  depth_bias: f32,
}

struct VsmReceiver {
  texel: vec2f,
  page: vec2i,
  local: vec2f,
  ref_depth: f32,
  out_of_range: bool,
}

struct VsmLookup {
  phys: u32,
  usable: bool,
  stale: bool,
}

struct VsmTap {
  page: vec2i,
  local: vec2f,
}

struct VsmLevelResult {
  usable: bool,
  stale: bool,
  out_of_range: bool,
  v: f32,
}

struct VsmResult {
  v: f32,
  level: u32,
  fallback: bool,
  stale: bool,
  unshadowed: bool,
  out_of_range: bool,
}

fn vsm_filter_view(views: ptr<uniform, array<vec4u, 144>>, i: u32) -> VsmView {
  let base = VSM_HEADER_WORDS / 4u + min(i, VSM_MAX_VIEWS - 1u) * (VSM_VIEW_WORDS / 4u);
  var w: array<vec4u, 8>;
  for (var k = 0u; k < 8u; k++) {
    w[k] = (*views)[base + k];
  }
  return vsm_view_unpack(w);
}

fn vsm_filter_select_level(views: ptr<uniform, array<vec4u, 144>>, f: f32, p_rel: vec3f) -> u32 {
  let texel0 = vsm_f32((*views)[3].w);
  if (!(texel0 > 0.0)) {
    return VSM_SUN_LEVELS;
  }
  for (var level = vsm_level_wanted(f, texel0, vsm_f32((*views)[3].z)); level < VSM_SUN_LEVELS; level++) {
    if (vsm_level_fits(vsm_filter_view(views, vsm_level_view(level)), p_rel)) {
      return level;
    }
  }
  return VSM_SUN_LEVELS;
}

fn vsm_filter_tap(taps: u32, t: u32) -> vec2f {
  if (taps <= 1u) {
    return vec2f(0.0);
  }
  if (taps <= 4u) {
    return vec2f(f32(t & 1u), f32(t >> 1u)) - 0.5;
  }
  return vec2f(f32(t % 3u), f32(t / 3u)) - 1.0;
}

fn vsm_filter_taps(taps: u32) -> u32 {
  return select(select(9u, 4u, taps <= 4u), 1u, taps <= 1u);
}

fn vsm_normal_offset_texels(params: VsmFilterParams, n_dot_l: f32) -> f32 {
  return (VSM_FILTER_NORMAL_BASE + VSM_FILTER_NORMAL_SLOPE * (1.0 - saturate(n_dot_l))) * params.normal_scale;
}

fn vsm_receiver(v: VsmView, p_rel: vec3f, bias: f32) -> VsmReceiver {
  let c = vsm_view_coords(v, p_rel);
  let lp = clamp(floor(c.xy / f32(VSM_PAGE_TEXELS)), vec2f(0.0), vec2f(f32(VSM_WINDOW_PAGES - 1u)));
  var r: VsmReceiver;
  r.texel = c.xy;
  r.page = v.origin + vec2i(lp);
  r.local = c.xy - lp * f32(VSM_PAGE_TEXELS);
  let d = c.z + bias * v.texel * v.depth_scale;
  r.ref_depth = saturate(d);
  r.out_of_range = !(d >= 0.0 && d <= 1.0);
  return r;
}

fn vsm_tap_page(v: VsmView, texel: vec2f, offset: vec2f) -> VsmTap {
  let c = texel + offset;
  let lp = clamp(floor(c / f32(VSM_PAGE_TEXELS)), vec2f(0.0), vec2f(f32(VSM_WINDOW_PAGES - 1u)));
  var t: VsmTap;
  t.page = v.origin + vec2i(lp);
  t.local = clamp(c - lp * f32(VSM_PAGE_TEXELS), vec2f(VSM_FILTER_TAP_INSET),
                  vec2f(f32(VSM_PAGE_TEXELS) - VSM_FILTER_TAP_INSET));
  return t;
}

fn vsm_lookup(pt: ptr<storage, array<u32>, read>, v: VsmView, page: vec2i) -> VsmLookup {
  let index = vsm_page_index(v.table_offset, page);
  var e = 0u;
  if (index < arrayLength(pt)) {
    e = (*pt)[index];
  }
  return VsmLookup(vsm_entry_phys(e), vsm_page_usable(e), vsm_page_stale(e));
}

fn vsm_pcf(pool: texture_depth_2d, cmp: sampler_comparison, pages_x: u32, phys: u32, local: vec2f,
           ref_depth: f32) -> f32 {
  let px = max(pages_x, 1u);
  let origin = vec2f(vec2u(phys % px, phys / px) * VSM_PAGE_TEXELS);
  let uv = (origin + local) / vec2f(textureDimensions(pool));
  return textureSampleCompareLevel(pool, cmp, uv, ref_depth);
}

fn vsm_level_visibility(pt: ptr<storage, array<u32>, read>, pool: texture_depth_2d, cmp: sampler_comparison,
                        pages_x: u32, v: VsmView, p_rel: vec3f, n_world: vec3f,
                        params: VsmFilterParams) -> VsmLevelResult {
  let n_dot_l = dot(n_world, -vsm_view_dir(v));
  let p = p_rel + n_world * (vsm_normal_offset_texels(params, n_dot_l) * v.texel);
  let rcv = vsm_receiver(v, p, params.depth_bias);
  let taps = vsm_filter_taps(params.taps);
  var out: VsmLevelResult;
  out.out_of_range = rcv.out_of_range;
  var last = vec2i(i32(0x7FFFFFFF));
  var found = VsmLookup(0u, false, false);
  var sum = 0.0;
  for (var t = 0u; t < taps; t++) {
    let tap = vsm_tap_page(v, rcv.texel, vsm_filter_tap(taps, t));
    if (any(tap.page != last)) {
      found = vsm_lookup(pt, v, tap.page);
      last = tap.page;
    }
    if (!found.usable) {
      return out;
    }
    out.stale = out.stale || found.stale;
    sum += vsm_pcf(pool, cmp, pages_x, found.phys, tap.local, rcv.ref_depth);
  }
  out.usable = true;
  out.v = sum / f32(taps);
  return out;
}

// A deferred page keeps its old depth, which beats dropping a whole page of static shadow
fn vsm_visibility(views: ptr<uniform, array<vec4u, 144>>, pt: ptr<storage, array<u32>, read>, pool: texture_depth_2d,
                  cmp: sampler_comparison, level: u32, p_rel: vec3f, n_world: vec3f,
                  params: VsmFilterParams) -> VsmResult {
  var r: VsmResult;
  r.v = 1.0;
  r.level = level;
  r.unshadowed = true;
  let pages_x = (*views)[3].y;
  for (var k = 0u; k <= VSM_FILTER_FALLBACK_LEVELS; k++) {
    let l = level + k;
    if (l >= VSM_SUN_LEVELS) {
      break;
    }
    let v = vsm_filter_view(views, vsm_level_view(l));
    if (k > 0u && !vsm_level_fits(v, p_rel)) {
      continue;
    }
    let s = vsm_level_visibility(pt, pool, cmp, pages_x, v, p_rel, n_world, params);
    if (s.usable) {
      r.v = s.v;
      r.level = l;
      r.fallback = k > 0u;
      r.stale = s.stale;
      r.unshadowed = false;
      r.out_of_range = s.out_of_range;
      return r;
    }
  }
  return r;
}

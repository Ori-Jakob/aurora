//#import aurora/camera
//#import aurora/gbuffer
//#import aurora/vsm
//#import shadows/vsm_filter

struct VsmViewParams {
  camera: StageCamera,
  rotation: ViewRotation,
  flags: u32,
  taps: u32,
  normal_scale: f32,
  depth_bias: f32,
  reserved: vec4u,
}

override VSM_VIEW_MODE: u32 = 0u;

const VIEW_MOSAIC: u32 = 4u;
const VIEW_NO_KEY: u32 = 0xFFFFFFFFu;

@group(0) @binding(1) var<uniform> view_params: VsmViewParams;
@group(0) @binding(5) var view_compare: sampler_comparison;
@group(1) @binding(0) var view_depth: texture_2d<f32>;
@group(1) @binding(1) var view_normal: texture_2d<f32>;
@group(1) @binding(2) var view_pool: texture_depth_2d;
@group(1) @binding(3) var<storage, read> view_pt: array<u32>;
@group(1) @binding(4) var<storage, read> view_phys: array<u32>;
@group(1) @binding(5) var<storage, read> view_requests: array<u32>;
@group(1) @binding(6) var<storage, read> view_render_list: array<u32>;
@group(1) @binding(7) var<uniform> view_views: array<vec4u, 144>;
@group(1) @binding(8) var<storage, read> view_stats: array<u32>;
@group(1) @binding(9) var view_target: texture_storage_2d<rgba8unorm, write>;

fn level_color(level: u32) -> vec3u {
  var colors = array<vec3u, 16>(
      vec3u(255u, 64u, 64u), vec3u(255u, 160u, 0u), vec3u(255u, 240u, 80u), vec3u(160u, 230u, 60u),
      vec3u(60u, 200u, 60u), vec3u(0u, 220u, 160u), vec3u(0u, 220u, 220u), vec3u(80u, 160u, 255u),
      vec3u(60u, 90u, 255u), vec3u(160u, 90u, 255u), vec3u(200u, 60u, 140u), vec3u(255u, 170u, 200u),
      vec3u(200u, 200u, 200u), vec3u(150u, 110u, 70u), vec3u(120u, 120u, 60u), vec3u(90u, 60u, 120u));
  return colors[min(level, 15u)];
}

const PAGE_CACHED: vec3u = vec3u(60u, 200u, 60u);
const PAGE_RENDERED: vec3u = vec3u(255u, 160u, 0u);
const PAGE_STALE: vec3u = vec3u(160u, 90u, 255u);
const PAGE_FALLBACK: vec3u = vec3u(255u, 240u, 80u);
const PAGE_UNSHADOWED: vec3u = vec3u(255u, 0u, 255u);
const PAGE_NOT_RECEIVER: vec3u = vec3u(64u, 64u, 64u);
const TABLE_FREE: vec3u = vec3u(0u, 0u, 0u);
const TABLE_CACHED: vec3u = vec3u(80u, 160u, 255u);
const TABLE_REQUESTED: vec3u = vec3u(60u, 200u, 60u);
const TABLE_RENDERED: vec3u = vec3u(255u, 160u, 0u);
const TABLE_STALE: vec3u = vec3u(160u, 90u, 255u);
const TABLE_MISSING: vec3u = vec3u(230u, 40u, 40u);
const TABLE_CAMERA: vec3u = vec3u(255u, 255u, 255u);
const POOL_UNUSED: vec3u = vec3u(0u, 0u, 96u);
const POOL_RENDERED: vec3u = vec3u(255u, 160u, 0u);
const LETTERBOX: vec3u = vec3u(24u, 24u, 24u);
const NO_SUN: vec3u = vec3u(70u, 82u, 100u);

struct ViewReceiver {
  valid: bool,
  p_rel: vec3f,
  n_world: vec3f,
  footprint: f32,
}

fn view_receiver(px: vec2u, dims: vec2u) -> ViewReceiver {
  var r: ViewReceiver;
  let d = textureLoad(view_depth, px, 0).r;
  let normal_texel = textureLoad(view_normal, min(px, textureDimensions(view_normal) - 1u), 0);
  let cam = view_params.camera;
  if (d <= 0.0 || gbuf_receiver_code(normal_texel.a) < 2u || (cam.flags & CAMERA_VALID) == 0u) {
    return r;
  }
  let full = vec2f(dims);
  let z = camera_view_depth(cam, d);
  let pos = camera_view_position_fwd(cam, (vec2f(px) + 0.5) / full, z, full);
  r.p_rel = camera_world_dir(view_params.rotation, pos);
  let raw = camera_gx_normal_fwd(normal_texel.rgb * 2.0 - 1.0);
  let raw_length = length(raw);
  if (raw_length >= 1e-3) {
    r.n_world = camera_world_dir(view_params.rotation, raw / raw_length);
  }
  r.footprint = 2.0 * z / (full.y * cam.depth.w);
  r.valid = true;
  return r;
}

fn view_rendered(view: u32, origin: vec2i, page: vec2i) -> bool {
  let local = page - origin;
  if (any(local < vec2i(0)) || any(local >= vec2i(i32(VSM_WINDOW_PAGES)))) {
    return false;
  }
  let word = VSM_RL_ROW_MASKS + min(view, VSM_MAX_VIEWS - 1u) * VSM_WINDOW_PAGES + u32(local.y);
  if (word >= arrayLength(&view_render_list)) {
    return false;
  }
  return (view_render_list[word] & (1u << u32(local.x))) != 0u;
}

fn view_entry(index: u32) -> u32 {
  if (index >= arrayLength(&view_pt)) {
    return 0u;
  }
  return view_pt[index];
}

fn view_page_key(px: vec2u, dims: vec2u) -> u32 {
  let r = view_receiver(px, dims);
  if (!r.valid) {
    return VIEW_NO_KEY;
  }
  let level = vsm_filter_select_level(&view_views, r.footprint, r.p_rel);
  if (level >= VSM_SUN_LEVELS) {
    return VIEW_NO_KEY;
  }
  let page = vsm_receiver(vsm_filter_view(&view_views, vsm_level_view(level)), r.p_rel, 0.0).page;
  return (level << 20u) | ((bitcast<u32>(page.y) & 1023u) << 10u) | (bitcast<u32>(page.x) & 1023u);
}

fn view_border(px: vec2u, dims: vec2u) -> bool {
  let key = view_page_key(px, dims);
  if (key == VIEW_NO_KEY) {
    return false;
  }
  var border = false;
  if (px.x + 1u < dims.x) {
    let right = view_page_key(px + vec2u(1u, 0u), dims);
    border = border || (right != VIEW_NO_KEY && right != key);
  }
  if (px.y + 1u < dims.y) {
    let below = view_page_key(px + vec2u(0u, 1u), dims);
    border = border || (below != VIEW_NO_KEY && below != key);
  }
  return border;
}

fn level_mode(px: vec2u, dims: vec2u) -> vec3u {
  let r = view_receiver(px, dims);
  if (!r.valid) {
    return PAGE_NOT_RECEIVER;
  }
  let level = vsm_filter_select_level(&view_views, r.footprint, r.p_rel);
  let params = VsmFilterParams(view_params.taps, view_params.normal_scale, view_params.depth_bias);
  let s = vsm_visibility(&view_views, &view_pt, view_pool, view_compare, level, r.p_rel, r.n_world, params);
  if (s.unshadowed) {
    return PAGE_UNSHADOWED;
  }
  let c = level_color(s.level);
  return select(c, c / 2u, s.fallback);
}

fn pages_mode(px: vec2u, dims: vec2u) -> vec3u {
  let r = view_receiver(px, dims);
  if (!r.valid) {
    return PAGE_NOT_RECEIVER;
  }
  let level = vsm_filter_select_level(&view_views, r.footprint, r.p_rel);
  let params = VsmFilterParams(view_params.taps, view_params.normal_scale, view_params.depth_bias);
  let s = vsm_visibility(&view_views, &view_pt, view_pool, view_compare, level, r.p_rel, r.n_world, params);
  var c = PAGE_CACHED;
  if (s.unshadowed) {
    c = PAGE_UNSHADOWED;
  } else if (s.fallback) {
    c = PAGE_FALLBACK;
  } else {
    let view = vsm_level_view(s.level);
    let v = vsm_filter_view(&view_views, view);
    if (view_rendered(view, v.origin, vsm_receiver(v, r.p_rel, 0.0).page)) {
      c = PAGE_RENDERED;
    } else if (s.stale) {
      c = PAGE_STALE;
    }
  }
  return select(c, c / 2u, view_border(px, dims));
}

fn mosaic_view(cell: u32) -> u32 {
  return select(cell, VSM_SUN_LEVELS - 1u - cell, cell < VSM_SUN_LEVELS);
}

fn table_mode(px: vec2u, dims: vec2u) -> vec3u {
  let side = min(dims.x, dims.y) / (VIEW_MOSAIC * VSM_WINDOW_PAGES) * (VIEW_MOSAIC * VSM_WINDOW_PAGES);
  let origin = (dims - vec2u(side)) / 2u;
  if (side == 0u || any(px < origin) || any(px >= origin + vec2u(side))) {
    return LETTERBOX;
  }
  let cell_px = side / VIEW_MOSAIC;
  let q = px - origin;
  let cell = q / cell_px;
  let inner = q % cell_px;
  if (any(inner == vec2u(0u))) {
    return LETTERBOX;
  }
  let view = mosaic_view(cell.x + VIEW_MOSAIC * cell.y);
  let v = vsm_filter_view(&view_views, view);
  if ((v.flags & VSM_VIEW_ACTIVE) == 0u) {
    return LETTERBOX;
  }
  let local = inner / (cell_px / VSM_WINDOW_PAGES);
  let page = v.origin + vec2i(min(local, vec2u(VSM_WINDOW_PAGES - 1u)));
  let camera = v.origin + vec2i(floor(vsm_view_coords(v, vec3f(0.0)).xy / f32(VSM_PAGE_TEXELS)));
  if (all(page == camera)) {
    return TABLE_CAMERA;
  }
  if (view_rendered(view, v.origin, page)) {
    return TABLE_RENDERED;
  }
  let index = vsm_page_index(v.table_offset, page);
  let e = view_entry(index);
  var requested = false;
  if (index < arrayLength(&view_requests)) {
    requested = (view_requests[index] & VSM_REQUEST_MASK) != 0u;
  }
  let usable = vsm_page_usable(e);
  if (requested && !usable) {
    return TABLE_MISSING;
  }
  if (vsm_page_stale(e)) {
    return TABLE_STALE;
  }
  if (requested) {
    return TABLE_REQUESTED;
  }
  if (!usable) {
    return TABLE_FREE;
  }
  let w = vsm_entry_phys(e) * VSM_PHYS_WORDS + VSM_PHYS_LAST_USED;
  var age = 0u;
  if (w < arrayLength(&view_phys)) {
    age = view_views[0].y - view_phys[w];
  }
  return TABLE_CACHED * (256u - min(age, 48u) * 4u) / 256u;
}

fn rendered_owner(owner: u32) -> bool {
  let v = owner - 1u;
  let view = v / VSM_PAGES_PER_VIEW;
  let slot = v % VSM_PAGES_PER_VIEW;
  if (view >= VSM_MAX_VIEWS) {
    return false;
  }
  let o = vsm_filter_view(&view_views, view).origin;
  let local = vsm_slot_local(vec2u(slot % VSM_WINDOW_PAGES, slot / VSM_WINDOW_PAGES), o);
  return view_rendered(view, o, o + vec2i(local));
}

fn pool_mode(px: vec2u, dims: vec2u) -> vec3u {
  let atlas = textureDimensions(view_pool);
  let scale = min(f32(dims.x) / f32(atlas.x), f32(dims.y) / f32(atlas.y));
  let offset = (vec2f(dims) - vec2f(atlas) * scale) * 0.5;
  let t = (vec2f(px) + 0.5 - offset) / scale;
  if (any(t < vec2f(0.0)) || any(t >= vec2f(atlas))) {
    return LETTERBOX;
  }
  let texel = min(vec2u(t), atlas - 1u);
  let pages_x = max(view_views[3].y, 1u);
  let pp = texel / VSM_PAGE_TEXELS;
  let p = pp.x + pp.y * pages_x;
  if (pp.x >= pages_x || p >= view_views[3].x || p * VSM_PHYS_WORDS >= arrayLength(&view_phys)) {
    return LETTERBOX;
  }
  let owner = view_phys[p * VSM_PHYS_WORDS + VSM_PHYS_OWNER];
  if (owner == 0u) {
    return POOL_UNUSED;
  }
  let local = t - vec2f(pp * VSM_PAGE_TEXELS);
  let edge = min(min(local.x, local.y), min(f32(VSM_PAGE_TEXELS) - local.x, f32(VSM_PAGE_TEXELS) - local.y));
  if (edge < max(2.0 / scale, 1.0) && rendered_owner(owner)) {
    return POOL_RENDERED;
  }
  return vec3u(u32(round(saturate(textureLoad(view_pool, texel, 0)) * 255.0)));
}

@compute @workgroup_size(8, 8)
fn view_main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(view_target);
  if (any(gid.xy >= dims)) {
    return;
  }
  let px = gid.xy;
  let receivers = textureDimensions(view_depth);
  var c = NO_SUN;
  switch VSM_VIEW_MODE {
    case 0u: {
      c = select(PAGE_NOT_RECEIVER, level_mode(px, receivers), all(px < receivers));
    }
    case 1u: {
      c = select(PAGE_NOT_RECEIVER, pages_mode(px, receivers), all(px < receivers));
    }
    case 2u: {
      c = table_mode(px, dims);
    }
    case 3u: {
      c = pool_mode(px, dims);
    }
    default: {
    }
  }
  textureStore(view_target, px, vec4f(vec3f(c) / 255.0, 1.0));
}

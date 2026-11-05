//#import aurora/vsm

@group(1) @binding(0) var vsm_pool: texture_depth_2d;
@group(1) @binding(1) var vsm_copy: texture_depth_2d;
@group(1) @binding(2) var<storage, read> vsm_render_list: array<u32>;
@group(1) @binding(3) var<storage, read> vsm_pt: array<u32>;
@group(1) @binding(4) var<storage, read_write> vsm_stats: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read> vsm_views: array<u32>;
@group(1) @binding(6) var<storage, read_write> vsm_probes: array<vec4f>;

var<workgroup> vsm_check_bad: atomic<u32>;

fn vsm_check_key(v: u32, phys: u32) -> u32 {
  let level = VSM_SUN_LEVELS - 1u - min(v / VSM_PAGES_PER_VIEW, VSM_SUN_LEVELS - 1u);
  return (level << 22u) | ((v % VSM_PAGES_PER_VIEW) << 12u) | (phys & VSM_ENTRY_PHYS_MASK);
}

@compute @workgroup_size(8, 8)
fn vsm_check_cmp(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u,
                 @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&vsm_check_bad, 0u);
  }
  workgroupBarrier();
  let i = vsm_render_list[VSM_RL_COUNT] + wg.x;
  let listed = wg.x < vsm_render_list[VSM_RL_CHECK_COUNT] && i < VSM_MAX_RENDER_PAGES;
  var v = VSM_NO_PAGE;
  var phys = VSM_NO_PAGE;
  if (listed) {
    let page = vsm_render_page(&vsm_render_list, i);
    v = vsm_render_virtual(page.x);
    phys = page.y;
  }
  var skipped = false;
  if (v < min(VSM_MAX_VIRTUAL_PAGES, arrayLength(&vsm_pt))) {
    skipped = (vsm_pt[v] & VSM_ENTRY_SKIP) != 0u;
  }
  let dims = textureDimensions(vsm_pool);
  let pages_x = max(vsm_render_list[VSM_RL_PAGES_X], 1u);
  let compared = v < VSM_MAX_VIRTUAL_PAGES && phys < VSM_MAX_POOL_PAGES && !skipped;
  if (compared) {
    let origin = vec2u(phys % pages_x, phys / pages_x) * VSM_PAGE_TEXELS;
    var bad = 0u;
    for (var y = lid.y; y < VSM_PAGE_TEXELS; y += 8u) {
      for (var x = lid.x; x < VSM_PAGE_TEXELS; x += 8u) {
        let p = origin + vec2u(x, y);
        if (all(p < dims)) {
          let a = bitcast<u32>(textureLoad(vsm_pool, p, 0));
          let b = bitcast<u32>(textureLoad(vsm_copy, p, 0));
          bad += select(0u, 1u, a != b);
        }
      }
    }
    if (bad != 0u) {
      atomicAdd(&vsm_check_bad, bad);
    }
  }
  workgroupBarrier();
  if (li != 0u) {
    return;
  }
  if (skipped) {
    atomicAdd(&vsm_stats[VSM_STAT_CHECK_SKIPPED], 1u);
  }
  if (!compared) {
    return;
  }
  atomicAdd(&vsm_stats[VSM_STAT_CHECK_COMPARED], 1u);
  let total = atomicLoad(&vsm_check_bad);
  if (total != 0u) {
    atomicAdd(&vsm_stats[VSM_STAT_CHECK_TEXELS], total);
    atomicAdd(&vsm_stats[VSM_STAT_CHECK_MISMATCHED], 1u);
    atomicMax(&vsm_stats[VSM_STAT_CHECK_FIRST], ~vsm_check_key(v, phys));
  }
}

fn vsm_probe_depth(v: VsmView, page: vec2i, c: vec3f) -> f32 {
  let index = vsm_page_index(v.table_offset, v.origin + page);
  if (index >= arrayLength(&vsm_pt)) {
    return -1.0;
  }
  let e = vsm_pt[index];
  let filled = (e & VSM_ENTRY_ALLOCATED) != 0u && (e & VSM_ENTRY_SKIP) == 0u &&
               (e & (VSM_ENTRY_CACHED | VSM_ENTRY_RENDER_NOW)) != 0u;
  if (!filled) {
    return -1.0;
  }
  let phys = vsm_entry_phys(e);
  let pages_x = max(vsm_render_list[VSM_RL_PAGES_X], 1u);
  let local = vec2u(clamp(c.xy - vec2f(page) * f32(VSM_PAGE_TEXELS), vec2f(0.0), vec2f(f32(VSM_PAGE_TEXELS) - 1.0)));
  let texel = vec2u(phys % pages_x, phys / pages_x) * VSM_PAGE_TEXELS + local;
  if (any(texel >= textureDimensions(vsm_pool))) {
    return -1.0;
  }
  return textureLoad(vsm_pool, texel, 0);
}

@compute @workgroup_size(32)
fn vsm_check_probe(@builtin(local_invocation_index) i: u32) {
  if (arrayLength(&vsm_probes) == 0u || i >= min(u32(vsm_probes[0].x), 32u) || i + 1u >= arrayLength(&vsm_probes)) {
    return;
  }
  let p = vsm_probes[i + 1u].xyz;
  for (var level = 0u; level < VSM_SUN_LEVELS; level++) {
    let v = vsm_view(&vsm_views, vsm_level_view(level));
    let c = vsm_view_coords_anchor(v, p);
    if (any(c.xy < vec2f(0.0)) || any(c.xy >= vec2f(f32(VSM_WINDOW_TEXELS)))) {
      continue;
    }
    let d = vsm_probe_depth(v, vec2i(floor(c.xy / f32(VSM_PAGE_TEXELS))), c);
    if (d < 0.0) {
      continue;
    }
    atomicOr(&vsm_stats[VSM_STAT_PROBE_FOUND], 1u << i);
    if (d > c.z + 8.0 * v.texel * v.depth_scale) {
      atomicOr(&vsm_stats[VSM_STAT_PROBE_OCCLUDED], 1u << i);
    }
    return;
  }
}

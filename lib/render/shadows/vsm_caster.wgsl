//#import aurora/vsm

struct VsmCasterPage {
  abs_page: vec2i,
  atlas_origin: vec2f,
  atlas_size: vec2f,
}

struct VsmCasterPosition {
  position: vec4f,
  clip: vec4f,
  rect: vec4f,
}

fn vsm_caster_page(v: VsmView, virtual_page: u32, phys: u32, pages_x: u32, rows: u32) -> VsmCasterPage {
  let slot = virtual_page % VSM_PAGES_PER_VIEW;
  var page: VsmCasterPage;
  page.abs_page = vec2i(vsm_abs_page(v.origin.x, slot % VSM_WINDOW_PAGES),
                        vsm_abs_page(v.origin.y, slot / VSM_WINDOW_PAGES));
  page.atlas_origin = vec2f(f32(phys % pages_x), f32(phys / pages_x)) * f32(VSM_PAGE_TEXELS);
  page.atlas_size = vec2f(f32(pages_x), f32(rows)) * f32(VSM_PAGE_TEXELS);
  return page;
}

fn vsm_caster_position(v: VsmView, page: VsmCasterPage, p_anchor: vec3f) -> VsmCasterPosition {
  // The window origin follows the camera, so texels come from the absolute page and a re-render in place matches
  let texels = vec2f(dot(v.anchor[0].xyz, p_anchor), dot(v.anchor[1].xyz, p_anchor)) -
               vec2f(page.abs_page) * f32(VSM_PAGE_TEXELS);
  let raw = dot(v.anchor[2].xyz, p_anchor) + v.anchor[2].w;
//#if VSM_UNCLIPPED
  let depth = raw;
//#else
  // Without depth clip control only the light side is clamped, so straddling triangles keep their plane below it
  let depth = min(raw, 1.0);
//#endif
  let atlas = (page.atlas_origin + texels) / page.atlas_size;
  var out: VsmCasterPosition;
  out.position = vec4f(2.0 * atlas.x - 1.0, 1.0 - 2.0 * atlas.y, depth, 1.0);
  out.clip = vec4f(texels.x, f32(VSM_PAGE_TEXELS) - texels.x, texels.y, f32(VSM_PAGE_TEXELS) - texels.y);
  out.rect = vec4f(page.atlas_origin, page.atlas_origin + vec2f(f32(VSM_PAGE_TEXELS)));
  return out;
}

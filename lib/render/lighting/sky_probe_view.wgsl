//#import aurora/sky
//#import aurora/srgb

struct ViewParams {
  mode: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

const SKY_VIEW_MAGENTA: vec3f = vec3f(1.0, 0.0, 1.0);

@group(0) @binding(1) var<uniform> view_params: ViewParams;
@group(0) @binding(2) var linear_clamp: sampler;
@group(1) @binding(0) var sky_probe: texture_2d<f32>;
@group(1) @binding(1) var sky_radiance: texture_2d<f32>;
@group(1) @binding(2) var<uniform> sky_sh9: SkySh9;
@group(1) @binding(3) var scene_depth: texture_2d<f32>;
@group(1) @binding(4) var scene_fog: texture_2d<u32>;
@group(1) @binding(5) var view_out: texture_storage_2d<rgba8unorm, write>;

fn sky_view_panel(panel: u32, uv: vec2f) -> vec3f {
  if (panel == 0u) {
    let dims = textureDimensions(sky_radiance);
    let r = textureLoad(sky_radiance, min(vec2u(uv * vec2f(dims)), dims - 1u), 0);
    // Unseen upper texels show the fog fill in the probe; magenta marks them here
    return select(r.rgb, SKY_VIEW_MAGENTA, r.w <= 0.0 && sky_oct_decode(uv).y >= 0.0);
  }
  if (panel == 1u) {
    return textureSampleLevel(sky_probe, linear_clamp, uv, 3.0).rgb;
  }
  if (!sky_sh9_valid(sky_sh9)) {
    return SKY_VIEW_MAGENTA;
  }
  return sky_sh9_irradiance(sky_sh9, sky_oct_decode(uv)) / SKY_PI;
}

@compute @workgroup_size(8, 8)
fn view(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(view_out);
  if (any(id.xy >= dims)) {
    return;
  }
  if (view_params.mode == 1u) {
    let depth = textureLoad(scene_depth, min(id.xy, textureDimensions(scene_depth) - 1u), 0).r;
    let rec = textureLoad(scene_fog, min(id.xy, textureDimensions(scene_fog) - 1u), 0);
    textureStore(view_out, id.xy, vec4f(vec3f(select(0.0, 1.0, sky_pixel_is_sky(depth, rec))), 1.0));
    return;
  }
  let side = max(min(dims.x / 3u, dims.y * 3u / 4u), 1u);
  var rgb = vec3f(0.0);
  if (id.y < side && id.x < side * 3u) {
    let panel = id.x / side;
    let uv = (vec2f(f32(id.x - panel * side), f32(id.y)) + 0.5) / f32(side);
    rgb = sky_view_panel(panel, uv);
  } else if (id.y < side + side / 8u && id.x < side * 3u) {
    rgb = select(sky_sh9.amb.rgb, vec3f(f32(sky_mode(sky_sh9)) * 0.5), id.x >= side * 2u);
  }
  textureStore(view_out, id.xy, vec4f(linear_to_srgb(rgb), 1.0));
}

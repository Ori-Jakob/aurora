//#import aurora/frame
//#import aurora/lighting

@group(0) @binding(1) var<uniform> cluster_params: LightClusterParams;
@group(1) @binding(0) var<storage, read_write> light_list: LightList;
@group(1) @binding(1) var<storage, read_write> light_grid: array<u32>;

var<private> cluster_slice_scale: array<f32, 25> = array<f32, 25>(
  0x1p+0f, 0x1.5f627ep+0f, 0x1.e24f3ap+0f, 0x1.4b0218p+1f, 0x1.c65738p+1f, 0x1.37d03p+2f, 0x1.abfe6ap+2f,
  0x1.25bb3ep+3f, 0x1.932cbcp+3f, 0x1.14b2b8p+4f, 0x1.7bcb8p+4f, 0x1.04a712p+5f, 0x1.65c558p+5f, 0x1.eb1338p+5f,
  0x1.510624p+6f, 0x1.ce9916p+6f, 0x1.3d7afp+7f, 0x1.b3c5b2p+7f, 0x1.2b11dcp+8f, 0x1.9a808cp+8f, 0x1.19ba16p+9f,
  0x1.82b288p+9f, 0x1.0963c8p+10f, 0x1.6c45eap+10f, 0x1.f4p+10f,
);

fn cluster_depths(cam: StageCamera, h: LightListHeader, s: u32) -> vec2f {
  let near = clamp(cam.depth.x / (cam.depth.y - 1.0), 0.0, CAMERA_FAR_DEPTH);
  var far = CAMERA_FAR_DEPTH;
  if (cam.depth.y != 0.0) {
    far = clamp(cam.depth.x / cam.depth.y, 0.0, CAMERA_FAR_DEPTH);
  }
  let lo = select(h.slice_near * cluster_slice_scale[s], near, s == 0u);
  let hi = select(h.slice_near * cluster_slice_scale[s + 1u], far, s == LIGHT_SLICES - 1u);
  return vec2f(lo, hi);
}

@compute @workgroup_size(64)
fn assign(@builtin(global_invocation_id) id: vec3u) {
  let cluster = id.x;
  if (cluster >= LIGHT_CLUSTER_COUNT) {
    return;
  }
  let base = cluster * LIGHT_MASK_WORDS;
  if ((cluster_params.flags & LIGHT_PARAMS_BRUTE_FORCE) != 0u) {
    for (var w = 0u; w < LIGHT_MASK_WORDS; w++) {
      light_grid[base + w] = 0xffffffffu;
    }
    return;
  }
  let h = light_list.header;
  let counts = light_counts(h);
  let cam = cluster_params.camera;
  let tx = cluster % LIGHT_TILES_X;
  let ty = (cluster / LIGHT_TILES_X) % LIGHT_TILES_Y;
  let s = cluster / (LIGHT_TILES_X * LIGHT_TILES_Y);
  let dims = vec2f(max(aurora_frame.efb_size, vec2u(1u)));
  let depths = cluster_depths(cam, h, s);
  // Tiles are assigned by pixel centre, so the uv edges bound every pixel a tile owns
  let u = vec2f(f32(tx), f32(tx + 1u)) * (1.0 / 16.0);
  let v = vec2f(f32(ty), f32(ty + 1u)) * (1.0 / 9.0);
  var lo = vec3f(3.4e38);
  var hi = vec3f(-3.4e38);
  for (var i = 0u; i < 8u; i++) {
    let corner = camera_view_position_fwd(cam, vec2f(u[i & 1u], v[(i >> 1u) & 1u]), depths[(i >> 2u) & 1u], dims);
    lo = min(lo, corner);
    hi = max(hi, corner);
  }
  let inflate = LIGHT_DEPTH_INFLATE * max(abs(depths.x), abs(depths.y));
  lo -= vec3f(inflate);
  hi += vec3f(inflate);
  var mask = array<u32, 8>();
  for (var i = 0u; i < counts.locals; i++) {
    let l = light_list.lights[counts.directional + i];
    let c = l.v0.xyz;
    let r = l.v0.w * (1.0 + LIGHT_RADIUS_INFLATE);
    let d = max(max(lo - c, c - hi), vec3f(0.0));
    if (dot(d, d) <= r * r) {
      mask[i / 32u] |= 1u << (i % 32u);
    }
  }
  for (var w = 0u; w < LIGHT_MASK_WORDS; w++) {
    light_grid[base + w] = mask[w];
  }
}
